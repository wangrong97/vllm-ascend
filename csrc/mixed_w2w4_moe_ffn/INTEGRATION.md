# mixed_w2w4_moe_ffn: GS32x32 integration boundary

## Status

The native framework adapter uses this complete routed-expert FFN for
`gs32_fusion=full_ffn`. It fuses Gate/Up GEMM, clipped SwiGLU, intermediate
K32 MXFP8 quantization, and Down GEMM. Router, token dispatch, weighted
expert combine and the shared expert remain outside the operator.
`gs32_fusion=none` uses the same binary for separate projection launches.

Both W2 and W4 Gate paths use the model's `swiglu_limit`: Gate has only
an upper bound, while Up is bounded on both sides. A positive limit enables
clipping; zero disables clipping for diagnostics. The arithmetic remains
BF16 Gate/Up -> FP32 SwiGLU -> BF16 intermediate -> K32 MXFP8.
Raw nonfinite Gate/Up values remain invalid even if clipping could otherwise
turn them finite. Their intermediate E8M0 scale is `0xff`; the framework
combines this with input, route and output validity on device.

## Source provenance

The original source is the user's restored `w2a8-matmul` project from
`operator_code.md`. It was first imported on
`vllm-ascend-vq2a8-v023-gs32_32`, based on `vllm-ascend-vq2a8-v023`
commit `6b2c38d22`. The standalone source was later imported from `feat/test`
commit `592a69733` into `feat/vq2a8_v0.1`, based on official vLLM Ascend
`v0.27.1rc1` commit `3b31886237ed`.

`SOURCE_MANIFEST.json` records all 39 original files, their original hashes,
the maintained hashes, and subsequent transformations. Original hashes are
preserved separately from maintained hashes. Internal `MatFP4` classes and
`MAT_FP4` macros retain their original names. The archive and generated
binaries are not duplicated into this directory. The supplied implementation
retains its original source provenance and licensing obligations; this
integration does not assign it a new license.

## Framework contract

1. **Weights.** The loader uses GS32x32 arbitrary 16x2 E4M3 codebooks,
   natural input-column order and natural Gate-then-Up output order in
   the exported checkpoint. It builds resident mixed-precision banks once,
   including the device's Gate32/Up32 interleave. W2 output scales keep
   their natural output-row order and store FP32 bits in UINT64's low bits.
   The synthetic four-level generator is a fixture, not a restriction on
   real VQ codebooks. Permutation is unsupported.
2. **Precision pairs.** W2/W2, W2/W4 and W4/W4 are supported. W4/W2 is
   rejected before launch. All experts retain their weight/table segments,
   including experts with no current tokens. The stage mask reflects
   resident precision types rather than current counts, so graph replay
   can change which experts receive tokens without changing launch stages.
3. **Tiling and binary ABI.** Version 3 preserves the 64-byte header and
   component offsets 64, 192, 832 and 960. A 32-byte `MixedPostOptions`
   extension starts at offset 1600; its first field is the FP32 limit,
   followed by zeroed reserved fields. Total tiling size is 1632 bytes.
   The 19 kernel arguments are unchanged. The registered kernel stem is
   `mixed_w2w4_moe_ffn_v3`; the binary file remains
   `mixed_w2w4_moe_ffn.o`. Rebuild the Torch bridge and device binary together.
4. **Runtime ownership.** The bridge uses Torch-owned inputs, outputs and
   scratch, a device guard, the selected current raw stream, allocator
   stream recording and queued strong owners. Standalone file I/O,
   allocations, stream creation, synchronization and device reset stay
   outside model forward. Route sanitation and status propagation must
   work for empty groups and all-empty local ranks without CPU sync.

The launch uses 32 AIC blocks. Its full-core barrier runs only when both
Gate and Down stages are enabled; standalone projections avoid this barrier.
Changes to block counts require AIC/AIV scheduling and synchronization
validation. Down consumes E8M0 as numeric scale, not an address or route;
invalid intermediates are rejected by framework status checks.

## Independent validation

CPU checks from the repository root (numerical cases require NumPy and
`ml_dtypes`; import-integrity checks do not):

```bash
python -m unittest discover -s csrc/mixed_w2w4_moe_ffn/tests -p 'test_*.py' -v
```

On a compatible Linux CANN 9.2.0/Ascend target, from this directory:

```bash
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-9.2.0
# Load the matching CANN environment first; verify the device and path.
bash run_mixed_w2w4_moe_ffn_esl.sh
python3 scripts/gen_data.py 3 4096 4096 4096 \
  --group-list 1,1,1 --is-expert-w4 0,0,1 --is-down-w4 0,1,1 \
  --seed 1234 --swiglu-limit 10
./mixed_w2w4_moe_ffn_host 3 4096 4096 4096 1 0 1
python3 cmp.py 3 4096
```

The generator saves the limit in `input/input_swiglu_limit.bin` for the
Host and in `input/metadata.json` for the result record. The target remains
`dav-c310-cube/vec`. CPU checks do not compile or execute device code.
Device 1 in the example must exist and be available. The standalone
comparator's `rtol=atol=0.01` and 99% overall match-rate threshold do not
establish bit-exact or real-model accuracy. Cases share input/output
directories and must not execute concurrently in one working directory.

Hardware acceptance must cover arbitrary real VQ LUTs and W4 banks,
all supported precision pairs, clipping tails, zero groups/all-empty ranks,
non-default streams and queued lifetimes, graph capture/replay with changed
routes, followed by model accuracy and latency. No NPU compilation or
execution is claimed by the local CPU checks.
