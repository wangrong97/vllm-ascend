# Mixed GS32 VQ2/MXFP4 inference

This integration loads `vq2_mixed_gs32_source` version 3 checkpoints through
the standard vLLM safetensors loader and the Ascend quantization scheme
registry. Each routed expert projection independently selects VQ2 or MXFP4.
Attention, shared experts and other model components retain their checkpoint
dtypes and use the ordinary model implementation.

The integration targets DeepSeek-V4 on Ascend950 with Model Runner V1 or V2.
It was extracted from development commit `592a69733`; the framework behavior
is based on `0a8761f07`, with the V2 adaptation from `d4b4e6157`. The migration
excludes the experimental fixed I32 prefill graph.

## Supported execution

| Setting | Behavior |
| --- | --- |
| `serial` | Strict eager reference with per-expert staged projections |
| `batched` | Eager execution sharing activation preparation and launch plans |
| `fused` | Eager grouped projections over actual per-expert routed rows |
| `full_graph` | B1 decode graph replay; prefill uses eager device routing |
| `--gs32-fusion none` | Separate activation preparation |
| `--gs32-fusion swiglu_quant` | Optional SwiGLU and Down activation preparation fusion |

Ordinary layers use tensor parallelism; routed experts use whole-expert EP
with linear placement. Supported configuration sizes are 1, 2, 4 and 8. An
EP rank keeps complete local expert parameters and the common MoE pipeline
combines its partial results. Activations remain BF16 through dispatch because
VQ2 permutations and scales are expert-specific.

The grouped native path has no arbitrary Python per-expert M or aggregate R
cap. The scheduler still bounds each prefill step through
`max_num_batched_tokens`, and allocations must fit available device memory.
Graph execution captures only one decode token, with at most eight routed
rows. Increasing the prefill chunk does not enlarge that decode graph.

The configuration rejects unsupported combinations before expert allocation:
tensor-sharded experts, PP/DP/CP greater than one, EPLB, LoRA, speculative
decoding, KV transfer, async scheduling, offload and sleep mode. Full decode
graphs require synchronous scheduling, one sequence, capture sizes `[1]`,
`FULL_DECODE_ONLY`, and the documented Ascend graph settings supplied by the
launcher. `full_graph_prefill` is not an accepted mode.

## Build and checkpoint prerequisites

Build the native mixed kernel and Torch bridge following
`csrc/mixed_w2w4_moe_ffn/INTEGRATION.md` and the runtime build instructions.
Supply the resulting `libvq2a8_gs32_stage.so` and `mixed_w2w4_moe_ffn.o`
explicitly; no developer-specific build directory is assumed.

**The GS32 bridge alone is insufficient for an entire DeepSeek-V4 model.**
Build/install the complete `vllm_ascend_C` extension and its ACLNN operators
for the same target stack. This integration also includes the SAS and QLI
AICPU metadata corrections, which validate the fixed output layout and clear
unused metadata slots. Rebuild those custom operators when migrating the
framework. Each worker registers and checks the required native bindings
after selecting its device.

Existing mixed source-v1 delivery checkpoints can be converted to source-v3:

```bash
python tools/convert_vq2_mixed_gs32_v3.py \
  --source "$SOURCE_MODEL" --output "$MODEL" --dry-run
python tools/convert_vq2_mixed_gs32_v3.py \
  --source "$SOURCE_MODEL" --output "$MODEL"
```

The converter requires an explicit source and a new output directory. It
validates the original source manifest, preserves non-routed tensors and
writes separately named expert tensors. The runtime accepts only source-v3;
older artifact layouts and developer diagnostic loaders are not production
dependencies. Runtime loading validates index/header agreement, root tensor
dtypes, duplicate or missing tensors, and local expert ownership. Indexed
shards must use a uniform directory depth for the standard loader's pattern.

## Start and test a service

Use absolute paths for `MODEL` and `BUILD`; `BUILD` names the directory that
actually contains the artifacts produced by your build.

```bash
export VLLM_USE_V2_MODEL_RUNNER=0
export ASCEND_RT_VISIBLE_DEVICES=4,5

bash tools/tests/serve_vq2a8_256k_1K.sh \
  --model "$MODEL" \
  --gs32-library "$BUILD/libvq2a8_gs32_stage.so" \
  --gs32-binary "$BUILD/mixed_w2w4_moe_ffn.o" \
  --gs32-fusion none \
  --max-num-batched-tokens 8192
```

The convenience launchers preserve explicit device and thread environment
settings. They accept checkpoint/artifact CLI arguments or the existing
`GS32_OUTPUT`, `GS32_EP_LIBRARY` and `GS32_EP_BINARY` script variables. CLI
arguments override those variables. `--dry-run` prints the final command
without initializing the model.

The 256K launcher budgets 16 GiB per worker for KV cache and sets context
capacity to 262144 input plus 1024 output tokens. Startup validates the actual
capacity; this budget is not a general memory sizing guarantee. The 64K and
1M convenience wrappers likewise describe requested test cases, not validated
hardware capacity or a quality result at those lengths.

```bash
python tools/benchmark_vq2a8_serving.py \
  --base-url http://127.0.0.1:8000 --model vq2a8 \
  --input-tokens 128 --max-tokens 128 \
  --warmups 1 --repeats 10 --timeout 1200 --progress --summary

bash tools/tests/run_vq2a8_ep2_256k_1k.sh \
  --base-url http://127.0.0.1:8000 --model vq2a8 \
  --warmups 0 --repeats 2 --timeout 21600 \
  --result-dir benchmark-results/256k-chunk8192-none
```

Compare identical hardware, software revisions, fusion, chunk size, model
and cache settings. Record first-request and steady-state results separately.
Prefix caching is disabled in the supplied service configuration. A local
FFN timing excludes attention, HCCL, token routing and scheduling, so it must
not be reported as end-to-end TTFT improvement. The development branch's
long-prefill performance regression remains unresolved; this migration does
not claim a measured serving speedup or successful 1M acceptance.

## One-command performance suite

After starting the service, run the serving latency suite with one command:

```bash
python tools/tests/run_vq2a8_performance_suite.py \
  --base-url http://127.0.0.1:8000 --model vq2a8 \
  --suite standard --label mrv1-ep2-chunk8192-none \
  --result-dir benchmark-results/performance
```

The client connects to the existing service and sends one request at a time.
It does not start or restart workers. Select the runner, devices, chunk size,
fusion and KV budget when launching the service. The label is your annotation;
the client cannot verify these settings. Keep prefix caching disabled for
these repeated synthetic prompts, and avoid other traffic during comparisons.

| Suite | Cases, in execution order |
| --- | --- |
| `quick` | 128/127/129/224 input tokens, each with 32 output; then 128-to-128 |
| `standard` (default) | `quick`, 64K-to-1K, 256K-to-1K, then short recovery |
| `full` | `quick`, 64K-to-1K, 256K-to-1K, 1M-to-1K, then short recovery |

Boundary cases use no warmup and two measured requests each. The 128-to-128
case uses one warmup and ten measured requests. Long cases use no warmup and
two measured requests. Recovery repeats 128-to-128 three times without a
warmup, so its first request can expose latency left by the long workload.
Recovery here checks completion and latency, not answer correctness.

Every case is checked against the server's reported context capacity before
any generation. If one preflight fails, this suite sends no generation
requests and exits nonzero. It never shortens an input to fit. The standard
suite requires at least **263168** context tokens; full requires **1049600**
because `1m` means **1048576 input plus 1024 output tokens**. Enough reported
context does not guarantee sufficient runtime memory or successful execution.
The existing long-context-only entry point keeps its original behavior of
running other valid cases even if one length fails preflight.

Use `--check-only` for capacity checks without generation, or `--dry-run` for
an offline plan without HTTP requests or result files. Select a subset or
override request counts when needed:

```bash
python tools/tests/run_vq2a8_performance_suite.py --suite full --check-only
python tools/tests/run_vq2a8_performance_suite.py \
  --cases 128x128 256k recovery --warmups 0 --repeats 2 \
  --label mrv2-ep2-chunk8192-none
```

The explicit `1m-window` case instead uses 1047552 input tokens plus 1024
output tokens inside a 1048576-token window. It is never substituted for `1m`.
`--timeout` controls each HTTP socket timeout, not a total suite deadline.

Each invocation creates a unique result directory with `summary.json`,
`summary.csv`, and per-case JSON and live logs. Reports retain individual
warmup and measured requests; means and medians exclude warmups and are
computed separately for each case. CSV `first_ttft_ms` is the first actual
request of that case, including a warmup if configured. It does not establish
a cold server or an empty compilation cache. `client_git_revision` identifies
the benchmark checkout, not the server revision.

TTFT and TPOT reuse the existing client-visible SSE timing implementation.
TPOT uses reported completion token counts, not SSE chunk counts. Tokenization
and prompt construction occur outside timed requests. Returned input/output
counts must match exactly. A failed request stops subsequent generation and
leaves the remaining cases marked `not_run`; incomplete cases have no success
average. Ctrl+C saves partial results and returns 130. Closing the client
connection does not prove the server has already cancelled its computation.

## Correctness and regression validation

### One-command NPU regression

The NPU CI driver reuses the existing nightly tests and starts each stage in
an isolated process. Run it on an idle Linux Ascend950 host with the source-v3
checkpoint, the rebuilt GS32 artifacts, and the complete `vllm_ascend_C`
extension (including SAS/QLI metadata operators). The tests load their own
models; they do not use an HTTP server. Stop serving on the selected cards
before running this suite.

```bash
python tools/tests/run_vq2a8_npu_ci.py \
  --devices 4,5 \
  --model "$MODEL" \
  --gs32-library "$BUILD/libvq2a8_gs32_stage.so" \
  --gs32-binary "$BUILD/mixed_w2w4_moe_ffn.o" \
  --suite pr \
  --result-dir benchmark-results/npu-ci
```

| Suite | Devices | Coverage |
| --- | --- | --- |
| `kernels` | Exactly one, e.g. `--devices 4` | Nine component test files: activation, async validity, routes, native/grouped projections, mixed banks/FFN, EP bank geometry and attention |
| `pr` (default) | Exactly two | Components, then real EP2 eager/decode-graph comparison under both MRV1 and MRV2 |
| `nightly` | Exactly two | PR coverage plus short/16K/short decode-graph replay and recovery under both runners |

The component stage includes large-M boundaries, both local EP2 rank banks
at 4096/8192 tokens, and both activation fusion settings where parameterized
by the existing tests. Local EP-shaped tests do not execute HCCL; the separate
EP2 model stage checks actual dispatch/combine/collectives and expert ownership.
Running both runners checks each against its own eager reference, not direct
V1-versus-V2 numerical parity. `--model-runners v1` or `--model-runners v2`
selects one runner for a focused run. The recovery probe checks each worker's
actual runner class and decode replay count, using its existing 128-token
prefill chunks; it is not 256K/1M acceptance.

By default the EP2 test omits its optional EP1 full-model reference because
the checkpoint may not fit on one card. The same-topology EP2 eager/graph
comparison remains mandatory. Add `--single-card-reference` only when the
entire reference model fits on one card. The summary records this coverage
choice explicitly; EP1-only model tests are not run by default.

Add `--dry-run` to print the commands and device/runner assignments without
initializing an NPU or creating result files. Normal runs produce a unique
directory containing `summary.json`, `environment.log`, per-stage logs and
JUnit XML, and retained pytest temporary directories with model subprocess
logs/reports. Each stage prints its log path; use `tail -f` on that file to
watch detailed progress. The summary records the checkout commit, commands,
selected environment overrides, test counts, elapsed time and result status.

Exit zero requires every selected stage to execute at least one test with
no failures, errors, skipped cases or expected failures. Missing quantization
pairs in a checkpoint therefore prevent complete CI acceptance, even if
pytest would normally skip those cases. Inspect the JUnit skip reasons and
use a checkpoint covering all required pairs. Failures stop later stages,
which remain `not_run`. Missing/malformed JUnit reports also fail the run.
Timeout or cancellation cleans up the stage's owned process group; Ctrl+C
and SIGTERM save the partial report and return 130. `--timeout` defaults to
7200 seconds per stage; existing model probes retain their own shorter
1800-second child-process timeout.

The command is suitable as a job step on a private NPU CI runner; archive
the unique result directory and use the process exit code as the job result.
No GitHub-triggered NPU job or self-hosted runner registration is installed
by this script. If piping output to `tee`, enable `set -o pipefail` so shell
pipelines preserve failures. Existing component timing records stay in the
logs, but there is no automatic timing regression threshold. Use the serving
performance suite separately for TTFT/TPOT and compare equivalent hardware,
artifacts, runner, chunk size and fusion settings.

### Individual test entry points

CPU contract tests can run without a vLLM installation or NPU initialization:

```bash
python -X utf8 tools/run_vq2a8_cpu_tests.py
python -m pytest --noconftest -o addopts= -q \
  tests/ut/tools/test_convert_vq2_mixed_gs32_v3.py \
  tests/ut/tools/test_vq2a8_long_context_latency_runner.py \
  tests/ut/tools/test_vq2a8_long_context_server_launchers.py \
  tests/ut/attention/test_dsa_kv_quant.py
```

CPU tests exercise real tensor arithmetic and model-loading methods with
isolated framework scaffolding. They do not validate CANN compilation, HCCL,
ACLGraph or target hardware performance. Run the hardware suite separately:

```bash
ASCEND_RT_VISIBLE_DEVICES=4 VLLM_USE_V2_MODEL_RUNNER=0 \
python -m pytest -vv -s -x \
  tests/e2e/nightly/test_vq2a8_gs32_batched.py \
  -k 'device_prefill_compare or device_prefill_ep2_capacity' \
  --vq2a8-model "$MODEL" \
  --vq2a8-gs32-library "$BUILD/libvq2a8_gs32_stage.so" \
  --vq2a8-gs32-binary "$BUILD/mixed_w2w4_moe_ffn.o"

ASCEND_RT_VISIBLE_DEVICES=4,5 VLLM_USE_V2_MODEL_RUNNER=0 \
python -m pytest -vv -s -x \
  'tests/e2e/nightly/test_vq2a8_gs32_ep.py::test_gs32_ep_real_model_eager_and_graphs_match_single_card[ep2]' \
  tests/e2e/nightly/test_vq2a8_gs32_ep_attention.py \
  --vq2a8-model "$MODEL" \
  --vq2a8-gs32-library "$BUILD/libvq2a8_gs32_stage.so" \
  --vq2a8-gs32-binary "$BUILD/mixed_w2w4_moe_ffn.o"
```

Select EP2 cases when exposing only two devices. The default EP model probe
compares against a single-card reference; use
`--vq2a8-gs32-ep-skip-single-card` explicitly if that reference cannot fit.
Same-topology eager/decode-graph comparisons remain required. Validate both
EP ranks, all mixed projection pairs, empty experts, padded rows, group-list
counts and prefixes, nondefault streams, and invalid-input recovery.

## Model Runner V2

Select V2 with the existing upstream `VLLM_USE_V2_MODEL_RUNNER=1` environment
variable before starting the process. Set it to `0` to run the V1 comparison.
The launchers preserve this choice; the integration does not change the
upstream default or introduce a new runner-selection flag. Both runners use
the same checkpoint, native artifacts and GS32 execution options. In
`full_graph` mode, prefill still runs eagerly and only B1 decode is captured.

The runners share validity ownership: each expert layer writes a stable
device status view, and one aggregate check rejects invalid output before
sampling. V2 resets these views before each forward, checks generation after
LM Head submission, and checks dummy/profile forwards before returning.
An interrupted GS32 forward clears its pending state so a later sampling
call cannot consume a partial result. Capture warmups validate output outside
capture; captured execution records device writes without a host read.

V2 reuses the worker's existing startup prefill entry point. It temporarily
borrows non-null KV pages while the runner is idle, forces eager execution,
and restores the borrowed pages, input buffers, block tables and attention
metadata on both success and failure. This avoids leaving startup data in
the first real request's buffers. No extra per-layer device-to-host check is
added to the serving path.

Run the same-topology eager/decode-graph comparison above with
`VLLM_USE_V2_MODEL_RUNNER=1`. Then verify the actual runner and decode replay
on every rank across short, 16K and recovery requests:

```bash
ASCEND_RT_VISIBLE_DEVICES=4,5 VLLM_USE_V2_MODEL_RUNNER=1 \
python -m pytest -vv -s -x \
  tests/e2e/nightly/test_vq2a8_gs32_ep_long_context.py -k ep2 \
  --vq2a8-model "$MODEL" \
  --vq2a8-gs32-library "$BUILD/libvq2a8_gs32_stage.so" \
  --vq2a8-gs32-binary "$BUILD/mixed_w2w4_moe_ffn.o"
```

The report includes each worker's runner class, validity-owner count and
actual decode replay count. V2 replay counters are installed only by the
test RPC; they are not production hooks. A configuration claiming V2 while
running V1 fails this probe.

For latency comparisons, restart the service command above with V2 selected,
then repeat the short 128-to-128 and 256K-to-1024 commands with otherwise
identical settings. Compare startup, first-request TTFT and steady-state
TTFT/TPOT separately. Local CPU tests cover lifecycle and failure contracts;
NPU compilation, graph execution, numerical parity and serving performance
must be validated on the target stack. This adaptation makes no measured
speedup claim and does not resolve the recorded long-prefill regression.
