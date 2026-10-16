# mixed_w2w4_moe_ffn: GS32x32 integration boundary

## Status

This change imports only `csrc/mixed_w2w4_moe_ffn` from `feat/test`
commit `592a69733` into `feat/vq2a8_v0.1`. The target branch is based on
official vLLM Ascend `v0.27.1rc1`, commit `3b31886237ed`.

**This standalone import does not enable model dispatch.** It contains no
Torch extension, checkpoint loader, model runner changes, or serving backend.
Framework integration and its tests remain separate changes. Results from
other branches, including whole-model and large-M framework tests, are not
validation of this standalone PR.

The imported operator fuses Gate/Up GEMM, ordinary SwiGLU, intermediate
K32 MXFP8 quantization, and Down GEMM. The name `mixed_w2w4_moe_ffn`
describes its mixed W2/W4 expert FFN computation, not a single projection.
Router, token dispatch, and weighted expert combine are outside this operator.
After a public-name change, rebuild the Host and AIC/AIV binaries together;
previously compiled binaries do not validate the renamed entry points.

## Source provenance

The original source is the user's restored `w2a8-matmul` project from
`operator_code.md`. It was first imported on
`vllm-ascend-vq2a8-v023-gs32_32`, based on `vllm-ascend-vq2a8-v023`
commit `6b2c38d22`. `SOURCE_MANIFEST.json` records all 39 original files,
their original hashes, and the hashes of the maintained copies. Changes
already present in the source branch include:

- CRLF to LF for Linux tooling;
- literal `mat_fp4` to `mixed_w2w4_moe_ffn` in filenames and contents, including
  the AIC/AIV symbols, Runtime registration name, binary names, and scripts;
- restoration of compiler-default DCache maintenance for chained kernels;
- a full-core barrier only when both Gate and Down stages are active,
  so standalone projections do not enter an unnecessary cross-stage barrier.

This migration additionally removes trailing whitespace, formats Python
sources, and performs non-semantic lint and documentation cleanup. These
migration changes do not alter kernel mathematics, tiling, buffer layouts,
route logic, or numerical test thresholds; the existing DCache and barrier
fixes above are preserved.
Internal `MatFP4` classes and `MAT_FP4` macros retain their original names.
The archive and generated binaries are not duplicated into this directory.
The supplied implementation retains its original source provenance and
licensing obligations; this import does not assign it a new license.

Original source hashes remain available separately from the maintained file
hashes. Import integrity checks do not imply repository-wide lint passes.

## Framework integration boundary

The following requirements describe integration work outside this standalone
PR. They are not a claim that the source branch lacks an adapter.

1. **Explicit GS32 artifact loader.** Validate `group_size=32`,
   `row_group_size=32`, arbitrary 16x2 E4M3 VQ codebooks, `row_scale`,
   permutation, column scales, no RHT, and no residual. The new checkpoint
   has all experts even in hash layers. Its base config still contains
   `expert_dtype=fp4`; that is not a VQ dispatch instruction. Checkpoint
   schemas and format identities must match the actual exported weights.
2. **Weight and activation semantics.** Repack the original 4-bit vector
   indices and arbitrary pair LUTs without fitting four scalar levels.
   The four-level generator is only a synthetic test fixture. Row scales
   can map to the W2 output-scale inputs (FP32 bits in the low 32 bits of
   each UINT64). Column scales and permutation require per-expert input
   transforms. With canonical K32 codebooks, the Down transform must occur
   inside the fused operator after SwiGLU and before MXFP8 quantization;
   the current ABI has no Down permutation/column-scale inputs. Reordering
   codes alone cannot absorb a cross-K32 permutation without codebook IDs.
   Reject nonzero weight bias unless explicitly implemented. Moving scales
   across quantization or BF16 rounding is not automatically numerically
   equivalent to the standalone model weight decoder.
3. **Model SwiGLU.** DeepSeek-V4 uses `swiglu_limit=10`. The imported
   reference and device helper explicitly use unclipped SwiGLU. The model
   variant must preserve the configured gate/up clipping and establish
   the intended MXFP8 versus rowwise W8A8 numerical reference; changing
   this silently is not a correct backend replacement.
4. **Torch/runtime adapter and expert dispatch.** Replace the complete
   routed expert FFN boundary, not a single `_projection`. Keep router,
   weighted combine, routed scale, and shared expert semantics outside.
   Use Torch-owned tensors, device guard, caller-selected current raw
   stream, allocator `recordStream`, and task-queue strong-owner capture.
   Do not bring standalone allocation, file I/O, stream creation,
   synchronization, or device reset into forward. Use real target CANN
   headers and a separately versioned ABI/library. The raw mixed-kernel
   registration and graph capture need their own hardware acceptance.

The standalone launch is fixed at 32 AIC blocks. Its full-core barrier is
conditional on both Gate and Down stages being active. Changes to block
counts require validation of both AIC/AIV scheduling and synchronization.

## Independent validation

CPU import checks (no CANN/Torch required), from the repository root:

```bash
python -m unittest discover -s csrc/mixed_w2w4_moe_ffn/tests -p 'test_*.py' -v
```

On a compatible Linux CANN 9.2.0/Ascend target, from this directory:

```bash
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-9.2.0
# Load the matching CANN environment first; verify the device and path.
bash run_mixed_w2w4_moe_ffn_esl.sh
python3 scripts/gen_data.py 3 4096 4096 4096 \
  --group-list 1,1,1 --is-expert-w4 0,0,1 --is-down-w4 0,1,1 --seed 1234
./mixed_w2w4_moe_ffn_host 3 4096 4096 4096 1 0 1
python3 cmp.py 3 4096
```

The target remains fixed to `dav-c310-cube/vec`; neither the hardware nor
the CANN installation has been verified by the Windows import checks.
Device 1 in the example must exist and be available. The original
comparator uses rtol=atol=0.01 and an overall 99% match-rate threshold;
it does not establish bit-exact or real-model accuracy. The INI currently
enables two W2+W2 full-FFN cases, including Down. All cases share input/output
directories and must not execute concurrently in the same working directory.

Acceptance order for a future model adapter: real-weight CPU layout and
reference checks, NPU eager correctness, queued lifetime and non-default
stream tests, graph replay tests, routed MoE correctness, then full-model
accuracy and latency. No NPU compilation or execution is claimed here.
