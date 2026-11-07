# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GS32 input gather and K32 MXFP8 quantization with deferred validity."""

import torch
from vllm.triton_utils import tl, triton

MX_GROUP = 32
FLOAT32_MAX = 3.4028234663852886e38
FULL_FFN_CHECK_COLUMNS = 4096
FULL_FFN_CHECK_FLAGS = 1024
MXFP8_NONFINITE_SCALE = 255


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


@triton.jit(do_not_specialize=["INPUT_ROWS"])
def _prepare_activation_kernel(
    hidden_ptr,
    source_row_ptr,
    route_flags_ptr,
    route_metadata_ptr,
    output_ptr,
    output_scale_ptr,
    flags_ptr,
    K: tl.constexpr,
    INPUT_ROWS,
    BLOCK_K: tl.constexpr,
    HAS_SOURCE_ROWS: tl.constexpr,
    HAS_ROUTE_FLAGS: tl.constexpr,
    HAS_ROUTE_METADATA: tl.constexpr,
    GROUP: tl.constexpr,
    FLOAT_MAX: tl.constexpr,
):
    # Routed inputs can span large expert banks; widen before strides.
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
    if HAS_ROUTE_METADATA:
        live_rows = tl.load(route_metadata_ptr).to(tl.int64)
        route_valid = route_valid & (tl.load(route_metadata_ptr + 1) == 1)
        route_valid = route_valid & (live_rows >= 0) & (live_rows <= INPUT_ROWS)
        source_valid = row < live_rows
        padding = row >= live_rows
    # Padding and invalid routes do not touch BF16 input,
    # and bypass group reductions. Zero has neutral E8M0 code 127.
    if source_valid & route_valid:
        hidden = tl.load(hidden_ptr + source_row * K + columns, mask=active, other=0).to(tl.float32)
        rounded = hidden.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
        finite = (tl.abs(hidden) <= FLOAT_MAX) & (tl.abs(rounded) <= FLOAT_MAX)
        valid = finite
        # The forward owner checks validity after completion; the native FFN
        # always receives finite FP8 bytes even when inputs are invalid.
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


def prepare_activation(hidden, *, source_rows=None, route_flags=None, route_metadata=None):
    rows, width = hidden.shape
    output = torch.empty((rows, width), device=hidden.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty((rows, width // MX_GROUP), device=hidden.device, dtype=torch.uint8)
    flags = torch.empty((rows,), device=hidden.device, dtype=torch.int32)
    _prepare_activation_kernel[(rows,)](
        hidden,
        hidden if source_rows is None else source_rows,
        hidden if route_flags is None else route_flags,
        hidden if route_metadata is None else route_metadata,
        output,
        scales,
        flags,
        K=width,
        INPUT_ROWS=hidden.shape[0],
        BLOCK_K=triton.next_power_of_2(width),
        HAS_SOURCE_ROWS=source_rows is not None,
        HAS_ROUTE_FLAGS=route_flags is not None,
        HAS_ROUTE_METADATA=route_metadata is not None,
        GROUP=MX_GROUP,
        FLOAT_MAX=FLOAT32_MAX,
    )
    return output.view(torch.uint8), scales, flags


@triton.jit(do_not_specialize=["ROWS"])
def _finish_full_ffn_kernel(
    output_ptr,
    middle_scales_ptr,
    input_flags_ptr,
    route_flags_ptr,
    native_flags_ptr,
    route_metadata_ptr,
    block_flags_ptr,
    ROWS,
    WIDTH: tl.constexpr,
    SCALE_WIDTH: tl.constexpr,
    COLUMN_BLOCK: tl.constexpr,
    SCALE_BLOCK: tl.constexpr,
    COLUMN_BLOCKS: tl.constexpr,
    NONFINITE_SCALE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    column_block = tl.program_id(1)
    live_rows = tl.load(route_metadata_ptr).to(tl.int64)
    route_valid = (tl.load(route_metadata_ptr + 1) == 1) & (live_rows >= 0) & (live_rows <= ROWS)
    route_valid = route_valid & (tl.load(route_flags_ptr) == 1) & (tl.load(native_flags_ptr) == 1)
    selected = route_valid & (row < live_rows)
    columns = column_block.to(tl.int64) * COLUMN_BLOCK + tl.arange(0, COLUMN_BLOCK).to(tl.int64)
    active = columns < WIDTH
    # Do not load padding or invalid routes. Successful output needs no restore
    # copy: only nonfinite live values and excluded rows require stores.
    values = tl.load(output_ptr + row * WIDTH + columns, mask=active & selected, other=0).to(tl.float32)
    finite = (values == values) & (tl.abs(values) != float("inf"))
    tl.store(output_ptr + row * WIDTH + columns, 0, mask=active & ((~finite) | (~selected)))
    valid = route_valid & (tl.load(input_flags_ptr + row) == 1)
    valid = valid & (tl.min(tl.where(active, finite, True).to(tl.int32), axis=0) == 1)
    groups = tl.arange(0, SCALE_BLOCK).to(tl.int64)
    scales = tl.load(middle_scales_ptr + row * SCALE_WIDTH + groups, mask=groups < SCALE_WIDTH, other=127)
    valid = valid & (tl.min((scales != NONFINITE_SCALE).to(tl.int32), axis=0) == 1)
    tl.store(block_flags_ptr + row * COLUMN_BLOCKS + column_block, valid.to(tl.int32))


@triton.jit(do_not_specialize=["FLAGS"])
def _reduce_full_ffn_validity_kernel(block_flags_ptr, validity_ptr, FLAGS, BLOCK: tl.constexpr):
    # A bounded register tile keeps large prefill capacities from generating a
    # capacity-sized reduction vector or a distinct compiled specialization.
    offsets = tl.arange(0, BLOCK).to(tl.int64)
    valid = tl.full((BLOCK,), 1, tl.int32)
    for start in range(0, FLAGS, BLOCK):
        indices = start.to(tl.int64) + offsets
        flags = tl.load(block_flags_ptr + indices, mask=indices < FLAGS, other=1)
        valid = valid & (flags == 1).to(tl.int32)
    tl.store(validity_ptr, tl.min(valid, axis=0) == 1)


def finish_full_ffn(output, middle_scales, input_flags, route_flags, native_flags, route_metadata, validity):
    rows, width = output.shape
    column_block = min(FULL_FFN_CHECK_COLUMNS, triton.next_power_of_2(width))
    column_blocks = triton.cdiv(width, column_block)
    block_flags = torch.empty((rows * column_blocks,), dtype=torch.int32, device=output.device)
    _finish_full_ffn_kernel[(rows, column_blocks)](
        output,
        middle_scales,
        input_flags,
        route_flags,
        native_flags,
        route_metadata,
        block_flags,
        ROWS=rows,
        WIDTH=width,
        SCALE_WIDTH=middle_scales.shape[1],
        COLUMN_BLOCK=column_block,
        SCALE_BLOCK=triton.next_power_of_2(middle_scales.shape[1]),
        COLUMN_BLOCKS=column_blocks,
        NONFINITE_SCALE=MXFP8_NONFINITE_SCALE,
    )
    _reduce_full_ffn_validity_kernel[(1,)](block_flags, validity, FLAGS=block_flags.numel(), BLOCK=FULL_FFN_CHECK_FLAGS)
    return block_flags
