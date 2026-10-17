# GS32 low-level runtime

This layer connects the standalone mixed W2/W4 kernel to Torch tensors. It
provides the native projection bridge, activation preparation, and device-side
route metadata. Model loading, quantization registration, expert-bank dispatch,
EP communication, and model-runner integration are separate changes.

## Source and interfaces

The implementation is selected from `feat/test` commit
`592a697336e74e2defd9f8920d64793c4f2dac56`. The native operator sources are supplied
by the preceding operator PR and are not changed here.

- `csrc/vq2a8_gs32_stage/torch_binding.cpp` registers
  `torch.ops.vq2a8_gs32_stage` and opaque projection-plan/route handles. It retains
  tensor owners on the submitted stream, validates geometry and aliases, and
  reuses the standalone DownW2 projection stage for natural-order GateUp and
  Down projections.
- Strict `prepare_routes` accepts CPU counts and explicitly uploads owned
  metadata. `prepare_device_routes` accepts device counts, returns device
  validity, and supports deferred validation. The caller must retain and check
  this validity before accepting results.
- Grouped projections use actual route counts. The historical per-expert 4096
  and aggregate 32768 caps are absent. Available memory, integer geometry, and
  the native expert limit still apply. The serial reference retains its
  128-row launch contract.
- The activation kernels preserve permutation, BF16 rounding, K32 E8M0 scales,
  and FP8 bytes. Route kernels handle counts or cumulative endpoints, mixed
  subsets, inactive experts, padding, invalid metadata, and logical-row restore.
- `vq2a8_gs32_format.py` defines the source-v3 and native tensor contracts;
  `vq2a8_gs32_staged.py` defines the numerical reference. The empty `w2a8`
  package initializer deliberately performs no model-scheme registration.

Capture-safe operators do not by themselves enable model graph execution. The
later framework's `full_graph` mode uses decode graph replay and eager chunked
prefill. Fixed-length prefill graph experiments are not included.

## Build on the target host

Use Linux on the target Ascend system with matching CANN, `torch`, and
`torch_npu`. Load that CANN installation's environment first. Both the native
kernel and Torch bridge must come from compatible sources.

```bash
# Run from the repository root. ASCEND_TOOLKIT_HOME names the loaded toolkit.
bash csrc/mixed_w2w4_moe_ffn/run_mixed_w2w4_moe_ffn_esl.sh

python tools/build_vq2a8_gs32_stage.py \
  --cann "${ASCEND_TOOLKIT_HOME:?Set the CANN toolkit path}" \
  --kernel-binary "$PWD/csrc/mixed_w2w4_moe_ffn/mixed_w2w4_moe_ffn.o" \
  --build-dir "$PWD/build/vq2a8-gs32-runtime"
```

The first command compiles AIC/AIV and links the kernel ELF. The Python tool
compiles **only the Torch bridge** around that existing ELF, validates registered
schemas in a fresh process, and copies the ELF beside the `.so`. Its output is
`build/vq2a8-gs32-runtime/libvq2a8_gs32_stage.so` and
`build/vq2a8-gs32-runtime/mixed_w2w4_moe_ffn.o`.

Schema verification establishes interface availability, not NPU numerical
correctness or graph replay. Do not check generated binaries or build logs into
Git. Use `--dry-run` to inspect build inputs without compiling or writing files.

## Validation

CPU contract tests use real CPU Torch arithmetic and independent NumPy/
`ml_dtypes` reference calculations. They isolate the selected modules from
engine initialization and do not emulate an NPU execution success.

```bash
python -m pip install pytest torch numpy regex safetensors ml_dtypes
python -X utf8 tools/run_vq2a8_cpu_tests.py
```

On the target NPU, the following suite exercises activation/route boundaries,
native grouped projections, nondefault streams, invalid-route recovery, and
capture/replay. `$MODEL` must name a source-v3 mixed GS32 checkpoint. Only selected
expert tensors are read; no serving engine is created.

```bash
ASCEND_RT_VISIBLE_DEVICES=4 python -m pytest -vv -s -x \
  tests/e2e/nightly/test_vq2a8_gs32_activation.py \
  tests/e2e/nightly/test_vq2a8_gs32_activation_async.py \
  tests/e2e/nightly/test_vq2a8_gs32_device_routes.py \
  tests/e2e/nightly/test_vq2a8_gs32_grouped.py \
  tests/e2e/nightly/test_vq2a8_gs32_device_native.py \
  --vq2a8-model "$MODEL" \
  --vq2a8-gs32-library "$PWD/build/vq2a8-gs32-runtime/libvq2a8_gs32_stage.so" \
  --vq2a8-gs32-binary "$PWD/build/vq2a8-gs32-runtime/mixed_w2w4_moe_ffn.o"
```

The grouped file includes `large_m and smoke`, `beyond_4096`, and
`beyond_32768` selectors for smaller focused runs. Numerical assertions and
projection timings are separate; these tests do not establish whole-model
accuracy, TTFT, or TPOT.

The migration is prepared on a Windows CPU environment. Linux compilation,
NPU correctness, streams, and graph replay must be verified separately on the
target system; historical `feat/test` results do not replace this validation.
