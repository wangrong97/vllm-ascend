# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GS32 preparation kernels; all arithmetic follows the staged MXFP8 ABI."""

import torch
from vllm.triton_utils import tl, triton

MX_GROUP = 32
FINITE_BLOCK = 1024
FLOAT32_MAX = 3.4028234663852886e38


@triton.jit
def _quantize_groups(values, BLOCK_K: tl.constexpr, GROUP: tl.constexpr):
    grouped = tl.reshape(values, (BLOCK_K // GROUP, GROUP))
    maximum = tl.max(tl.abs(grouped), axis=1)
    bits = maximum.to(tl.int32, bitcast=True)
    exponent = (bits >> 23) & 255
    codes = exponent - 8 + ((bits & 0x7FFFFF) > 0x600000).to(tl.int32)
    codes = tl.where(maximum == 0, 127, tl.maximum(codes, 1))
    inverse = ((254 - codes) << 23).to(tl.float32, bitcast=True)
    scaled = (grouped * inverse[:, None]).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    scaled = tl.minimum(tl.maximum(scaled, -448.0), 448.0)
    return tl.reshape(scaled, (BLOCK_K,)), codes


@triton.jit(do_not_specialize=["EXPERTS", "INPUT_ROWS"])
def _prepare_activation_kernel(
    hidden_ptr,
    perm_ptr,
    slot_ptr,
    raw_gate_ptr,
    source_row_ptr,
    route_flags_ptr,
    output_ptr,
    output_scale_ptr,
    flags_ptr,
    K: tl.constexpr,
    EXPERTS,
    INPUT_ROWS,
    BLOCK_K: tl.constexpr,
    HAS_RAW_GATE: tl.constexpr,
    HAS_PERM: tl.constexpr,
    HAS_SOURCE_ROWS: tl.constexpr,
    HAS_ROUTE_FLAGS: tl.constexpr,
    FUSE_SWIGLU: tl.constexpr,
    SWIGLU_LIMIT: tl.constexpr,
    GROUP: tl.constexpr,
    FLOAT_MAX: tl.constexpr,
):
    # Eager grouped inputs can span large routed banks; widen before strides.
    row = tl.program_id(0).to(tl.int64)
    columns = tl.arange(0, BLOCK_K)
    active = columns < K
    if HAS_SOURCE_ROWS:
        source_row = tl.load(source_row_ptr + row)
        source_valid = (source_row >= 0) & (source_row < INPUT_ROWS)
        padding = source_row == -1
    else:
        source_row = row
        source_valid = True
        padding = False
    if HAS_ROUTE_FLAGS:
        route_valid = tl.load(route_flags_ptr) == 1
    else:
        route_valid = True
    # Padding and invalid routes do not touch BF16 input, permutations or slots,
    # and bypass SwiGLU/group reductions. Zero has neutral E8M0 code 127.
    if source_valid & route_valid:
        if HAS_PERM:
            slot = tl.load(slot_ptr + row)
            # Keep the exact inventory bound without compiling each bank size.
            slot_valid = (slot >= 0) & (slot < EXPERTS)
            safe_slot = tl.where(slot_valid, slot, 0)
            permutation = tl.load(perm_ptr + safe_slot * K + columns, mask=active & slot_valid, other=0)
        else:
            slot_valid = True
            permutation = columns
        index_valid = (permutation >= 0) & (permutation < K)
        safe_index = tl.where(index_valid, permutation, 0)
        read_mask = active & slot_valid & index_valid
        if FUSE_SWIGLU:
            # SwiGLU is pointwise, so gathering both inputs by the Down permutation
            # is equivalent to permuting the BF16 result. K32 quantization must
            # follow that permutation; cross-group permutations cannot move later.
            gate = tl.load(raw_gate_ptr + source_row * (2 * K) + safe_index, mask=read_mask, other=0).to(tl.float32)
            up = tl.load(raw_gate_ptr + source_row * (2 * K) + K + safe_index, mask=read_mask, other=0).to(tl.float32)
            raw_finite = (tl.abs(gate) <= FLOAT_MAX) & (tl.abs(up) <= FLOAT_MAX)
            gate = tl.where(raw_finite, gate, 0.0)
            up = tl.where(raw_finite, up, 0.0)
            if SWIGLU_LIMIT > 0:
                gate = tl.minimum(gate, SWIGLU_LIMIT)
                up = tl.minimum(tl.maximum(up, -SWIGLU_LIMIT), SWIGLU_LIMIT)
            # Preserve FP32 SiLU -> FP32 product -> BF16 RNE. This selectable
            # implementation still needs target-NPU comparison against Torch SiLU.
            silu = gate * tl.sigmoid(gate)
            hidden = silu * up
        else:
            hidden = tl.load(hidden_ptr + source_row * K + safe_index, mask=read_mask, other=0).to(tl.float32)
        rounded = hidden.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
        finite = (tl.abs(hidden) <= FLOAT_MAX) & (tl.abs(rounded) <= FLOAT_MAX)
        if FUSE_SWIGLU:
            finite = finite & raw_finite
        valid = slot_valid & index_valid & finite
        if HAS_RAW_GATE:
            gate = tl.load(raw_gate_ptr + source_row * (2 * K) + columns, mask=active, other=0).to(tl.float32)
            up = tl.load(raw_gate_ptr + source_row * (2 * K) + K + columns, mask=active, other=0).to(tl.float32)
            valid = valid & (tl.abs(gate) <= FLOAT_MAX) & (tl.abs(up) <= FLOAT_MAX)
        # Strict callers reject invalid flags before projection; graph callers
        # defer rejection until graph completion. Both receive finite FP8 bytes.
        safe_rounded = tl.where(active & valid, rounded, 0.0)
        scaled, codes = _quantize_groups(safe_rounded, BLOCK_K, GROUP)
        fp8 = scaled.to(output_ptr.dtype.element_ty, fp_downcast_rounding="rtne")
        tl.store(output_ptr + row * K + columns, fp8, mask=active)
        groups = tl.arange(0, BLOCK_K // GROUP)
        tl.store(output_scale_ptr + row * (K // GROUP) + groups, codes.to(tl.uint8), mask=groups < K // GROUP)
        row_valid = tl.min(tl.where(active, valid, True).to(tl.int32), axis=0)
        tl.store(flags_ptr + row, row_valid)

    else:
        # Ascend Triton cannot implicitly cast an integer literal to FP8.
        zero_fp8 = tl.full((BLOCK_K,), 0.0, tl.float32).to(output_ptr.dtype.element_ty, fp_downcast_rounding="rtne")
        tl.store(output_ptr + row * K + columns, zero_fp8, mask=active)
        groups = tl.arange(0, BLOCK_K // GROUP)
        tl.store(output_scale_ptr + row * (K // GROUP) + groups, 127, mask=groups < K // GROUP)
        tl.store(flags_ptr + row, (padding & route_valid).to(tl.int32))


@triton.jit(do_not_specialize=["ELEMENTS"])
def _finite_flags_kernel(tensor_ptr, flags_ptr, ELEMENTS, BLOCK: tl.constexpr, FLOAT_MAX: tl.constexpr):
    # Routed bank sizes change with token contents, even at fixed batch size.
    tile = tl.program_id(0).to(tl.int64)
    offsets = tile * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(tensor_ptr + offsets, mask=offsets < ELEMENTS, other=0).to(tl.float32)
    valid = tl.min((tl.abs(values) <= FLOAT_MAX).to(tl.int32), axis=0)
    tl.store(flags_ptr + tile, valid)


@triton.jit
def _finish_flags_kernel(
    output_ptr,
    gate_flags_ptr,
    down_flags_ptr,
    route_flags_ptr,
    native_flags_ptr,
    result_ptr,
    ELEMENTS: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
    FLAG_BLOCK: tl.constexpr,
    ROUTE_ELEMENTS: tl.constexpr,
    NATIVE_ELEMENTS: tl.constexpr,
    FLOAT_MAX: tl.constexpr,
):
    # One writer owns the final result. Tiled scans bound UB use independently
    # of the FFN output width; no atomics or stale accumulator are involved.
    valid = tl.full((), 1, tl.int32)
    columns = tl.arange(0, BLOCK)
    for start in range(0, ELEMENTS, BLOCK):
        offsets = start + columns
        values = tl.load(output_ptr + offsets, mask=offsets < ELEMENTS, other=0).to(tl.float32)
        valid = valid & tl.min((tl.abs(values) <= FLOAT_MAX).to(tl.int32), axis=0)
    rows = tl.arange(0, FLAG_BLOCK)
    gate = tl.load(gate_flags_ptr + rows, mask=rows < ROWS, other=1)
    down = tl.load(down_flags_ptr + rows, mask=rows < ROWS, other=1)
    upstream_valid = tl.min(((gate == 1) & (down == 1)).to(tl.int32), axis=0)
    for start in range(0, ROUTE_ELEMENTS, BLOCK):
        offsets = start + columns
        flags = tl.load(route_flags_ptr + offsets, mask=offsets < ROUTE_ELEMENTS, other=1)
        upstream_valid = upstream_valid & tl.min((flags == 1).to(tl.int32), axis=0)
    for start in range(0, NATIVE_ELEMENTS, BLOCK):
        offsets = start + columns
        flags = tl.load(native_flags_ptr + offsets, mask=offsets < NATIVE_ELEMENTS, other=1)
        upstream_valid = upstream_valid & tl.min((flags == 1).to(tl.int32), axis=0)
    tl.store(result_ptr, valid & upstream_valid)


def prepare_activation(
    hidden, perm, row_slots, raw_gate=None, *, swiglu_limit=None, source_rows=None, route_flags=None
):
    rows, width = hidden.shape
    if swiglu_limit is not None:
        width //= 2
    output = torch.empty((rows, width), device=hidden.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty((rows, width // MX_GROUP), device=hidden.device, dtype=torch.uint8)
    flags = torch.empty((rows,), device=hidden.device, dtype=torch.int32)
    _prepare_activation_kernel[(rows,)](
        hidden,
        hidden if perm is None else perm,
        row_slots,
        hidden if raw_gate is None else raw_gate,
        row_slots if source_rows is None else source_rows,
        row_slots if route_flags is None else route_flags,
        output,
        scales,
        flags,
        K=width,
        EXPERTS=0 if perm is None else perm.shape[0],
        INPUT_ROWS=hidden.shape[0],
        BLOCK_K=triton.next_power_of_2(width),
        HAS_RAW_GATE=raw_gate is not None,
        HAS_PERM=perm is not None,
        HAS_SOURCE_ROWS=source_rows is not None,
        HAS_ROUTE_FLAGS=route_flags is not None,
        FUSE_SWIGLU=swiglu_limit is not None,
        SWIGLU_LIMIT=0.0 if swiglu_limit is None else float(swiglu_limit),
        GROUP=MX_GROUP,
        FLOAT_MAX=FLOAT32_MAX,
    )
    return output.view(torch.uint8), scales, flags


def finite_flags(tensor):
    count = triton.cdiv(tensor.numel(), FINITE_BLOCK)
    flags = torch.empty((count,), device=tensor.device, dtype=torch.int32)
    _finite_flags_kernel[(count,)](tensor, flags, ELEMENTS=tensor.numel(), BLOCK=FINITE_BLOCK, FLOAT_MAX=FLOAT32_MAX)
    return flags


def finish_flags(output, gate_flags, down_flags, *, route_flags=None, native_flags=None):
    result = torch.empty((1,), device=output.device, dtype=torch.int32)
    _finish_flags_kernel[(1,)](
        output,
        gate_flags,
        down_flags,
        gate_flags if route_flags is None else route_flags,
        gate_flags if native_flags is None else native_flags,
        result,
        ELEMENTS=output.numel(),
        ROWS=output.shape[0],
        BLOCK=FINITE_BLOCK,
        FLAG_BLOCK=triton.next_power_of_2(max(output.shape[0], 1)),
        ROUTE_ELEMENTS=0 if route_flags is None else route_flags.numel(),
        NATIVE_ELEMENTS=0 if native_flags is None else native_flags.numel(),
        FLOAT_MAX=FLOAT32_MAX,
    )
    return result
