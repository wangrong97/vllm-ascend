# Mixed GS32 VQ2/MXFP4 inference

This integration loads `vq2_mixed_gs32_source` version 3 checkpoints through
the standard vLLM safetensors loader and the Ascend quantization scheme
registry. Each routed expert projection independently selects VQ2 or MXFP4.
Attention, shared experts and other model components retain their checkpoint
dtypes and use the ordinary model implementation.

The initial integration targets DeepSeek-V4 on Ascend950 with Model Runner V1.
It was extracted from development commit `592a69733`; the runtime behavior is
based on `0a8761f07`. The migration excludes the experimental fixed I32 prefill
graph and the separate Model Runner V2 adaptation.

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

## Correctness and regression validation

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
