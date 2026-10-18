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

## Test guides

Use the separate guides for the three test workflows. Commands run from the
repository root; each guide explains its inputs, coverage and result format.

| Workflow | Guide | Service ownership |
| --- | --- | --- |
| NPU CI regression | [VQ2A8 NPU CI](vq2a8_npu_ci.md) | Loads its own models on idle selected cards |
| Serving performance | [VQ2A8 performance](vq2a8_performance.md) | Includes card selection and service launch commands, then runs HTTP latency tests |
| Answer accuracy | [VQ2A8 accuracy](vq2a8_accuracy.md) | Evaluates GPQA Diamond / AIME24 / AIME25 against an existing HTTP service |

### Start and test a service

See [the performance quick start](vq2a8_performance.md#one-command-performance-suite)
for the complete service and client commands, including physical NPU card
selection, Model Runner V1/V2 and a common 256K service for `standard`.

### One-command performance suite

The [performance guide](vq2a8_performance.md) covers short inputs, 64K/256K/1M,
context capacity, preflight checks and TTFT/TPOT reports. It also distinguishes
the full 1M input case from the smaller `1m-window` case.

### Correctness and regression validation

Numeric NPU regression, serving latency and model answer accuracy are separate
checks. Follow [the accuracy guide](vq2a8_accuracy.md) for dataset selection,
offline bundles, native grading and baseline score comparisons.

### One-command NPU regression

The [NPU CI guide](vq2a8_npu_ci.md) documents the `kernels`, `pr` and `nightly`
suites, both model runners, artifact requirements, logs and strict pass criteria.

### Individual test entry points

CPU and individual NPU commands are listed in
[the NPU CI guide](vq2a8_npu_ci.md). These remain useful for focused diagnosis
after a suite failure.

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

Run the same-topology eager/decode-graph comparison in the
[NPU CI guide](vq2a8_npu_ci.md) with
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

For latency comparisons, follow the [performance guide](vq2a8_performance.md),
restart the service with V2 selected, and repeat the short 128-to-128 and
256K-to-1024 cases with otherwise identical settings. Compare startup, first-request TTFT and steady-state
TTFT/TPOT separately. Local CPU tests cover lifecycle and failure contracts;
NPU compilation, graph execution, numerical parity and serving performance
must be validated on the target stack. This adaptation makes no measured
speedup claim and does not resolve the recorded long-prefill regression.
