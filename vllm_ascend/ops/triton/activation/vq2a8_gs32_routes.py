# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device routing for GS32 decode, graphs, and full-capacity prefill."""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.fused_moe.vq2a8_gs32_routes import MAX_DECODE_ROUTED_ROWS

COPY_BLOCK = 512
MAX_EXPERT_ROWS = 128
PREFILL_ROW_BLOCK = 256
RESTORE_COLUMN_BLOCK = 4096
# static_range needs constexpr objects, not annotation-only Python integers.
INT64_GATHER_BITS = tl.constexpr(16)
INT64_SUM_BITS = tl.constexpr(32)


@triton.jit(do_not_specialize=["EXPERTS"])
def _route_metadata_kernel(
    group_ptr,
    expert_ptr,
    logical_slot_ptr,
    counts_ptr,
    row_slots_ptr,
    gather_ptr,
    inverse_ptr,
    flags_ptr,
    ROWS: tl.constexpr,
    LOGICAL: tl.constexpr,
    EXPERTS,
    GROUP_TYPE: tl.constexpr,
    LOGICAL_BLOCK: tl.constexpr,
    EXPERT_BLOCK: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    MAX_COUNT: tl.constexpr,
    BUILD_ROWS: tl.constexpr,
    ALLOW_PADDING: tl.constexpr,
    SUBSET: tl.constexpr,
):
    logical = tl.arange(0, LOGICAL_BLOCK)
    logical_active = logical < LOGICAL
    raw = tl.load(group_ptr + logical, mask=logical_active, other=0).to(tl.int64)
    bounded = tl.minimum(tl.maximum(raw, 0), ROWS)
    value_valid = (raw >= 0) & (raw <= ROWS)
    if GROUP_TYPE == 0:
        previous = tl.load(group_ptr + logical - 1, mask=logical_active & (logical > 0), other=0).to(tl.int64)
        # Subtract only bounded values: INT64_MIN/MAX must not overflow.
        counts = bounded - tl.minimum(tl.maximum(previous, 0), ROWS)
    else:
        counts = bounded
    value_valid = value_valid & (counts >= 0) & (counts <= MAX_COUNT)
    counts = tl.where(logical_active, tl.minimum(tl.maximum(counts, 0), ROWS), 0).to(tl.int32)
    slots = tl.load(logical_slot_ptr + logical, mask=logical_active, other=-1)
    # EXPERT_BLOCK is a compile-time bucket; EXPERTS remains the true bound.
    slot_valid = (slots >= 0) & (slots < EXPERTS)
    physical = tl.arange(0, EXPERT_BLOCK)
    physical_active = physical < EXPERTS
    expert = tl.load(expert_ptr + physical, mask=physical_active, other=0)
    expert_valid = (expert >= 0) & (expert < LOGICAL)
    # Keep inventory addresses affine; values from partially masked loads
    # must only feed register lookups, never subsequent GM addresses.
    # Validate original int64 values before narrowing; [-1,255] is exact FP32.
    bounded_slots = tl.where(slot_valid, slots, -1).to(tl.float32)
    bounded_experts = tl.where(expert_valid, expert, -1).to(tl.float32)
    safe_slot = tl.where(slot_valid, slots, 0).to(tl.int32)
    safe_expert = tl.where(expert_valid, expert, 0).to(tl.int32)
    back_id = tl.gather(bounded_experts, safe_slot, axis=0).to(tl.int32)
    back_slot = tl.gather(bounded_slots, safe_expert, axis=0).to(tl.int32)
    mapping_valid = ((slots == -1) & ((counts == 0) | SUBSET)) | (slot_valid & (back_id == logical))
    valid = tl.min(tl.where(logical_active, value_valid & mapping_valid, True).to(tl.int32), axis=0) == 1
    valid = valid & ((tl.sum(counts, axis=0) <= ROWS) if ALLOW_PADDING else (tl.sum(counts, axis=0) == ROWS))
    valid = valid & (
        tl.min(tl.where(physical_active, expert_valid & (back_slot == physical), True).to(tl.int32), axis=0) == 1
    )
    # Ascend gather rejects int32 sources. Counts are already bounded to
    # [0, ROWS<=256], so the FP32 roundtrip preserves every count exactly.
    physical_counts = tl.gather(counts.to(tl.float32), safe_expert, axis=0).to(tl.int32)
    physical_counts = tl.where(physical_active, physical_counts, 0)
    if not SUBSET:
        valid = valid & (tl.sum(physical_counts, axis=0) == (tl.sum(counts, axis=0) if ALLOW_PADDING else ROWS))
    tl.store(counts_ptr + physical, tl.where(valid, physical_counts, 0), mask=physical_active)

    if BUILD_ROWS:
        # Keep the proven decode path: at most eight row-selection iterations.
        row_offsets = tl.arange(0, ROW_BLOCK)
        slot_values = tl.full((ROW_BLOCK,), 0, tl.int32)
        source_values = tl.full((ROW_BLOCK,), 0, tl.int32)
        inverse_values = tl.full((ROW_BLOCK,), -1, tl.int32)
        remaining = physical_counts
        for row in range(ROWS):
            selected = tl.min(tl.where(physical_active & (remaining > 0), physical, EXPERTS), axis=0)
            has_owner = selected < EXPERTS
            selected = tl.where(selected < EXPERTS, selected, 0)
            owner = physical_active & (physical == selected) & has_owner
            selected_expert = tl.sum(tl.where(owner, safe_expert, 0), axis=0)
            consumed = tl.sum(tl.where(owner, physical_counts - remaining, 0), axis=0)
            source_start = tl.sum(tl.where(logical < selected_expert, counts, 0), axis=0)
            source = source_start + consumed
            slot_values = tl.where(row_offsets == row, tl.where(valid, selected, 0), slot_values)
            source_values = tl.where(
                row_offsets == row, tl.where(valid, tl.where(has_owner, source, -1), 0), source_values
            )
            # Build the inverse in registers, never as a data-dependent scatter.
            inverse_values = tl.where(valid & has_owner & (row_offsets == source), row, inverse_values)
            remaining = remaining - owner.to(tl.int32)
        tl.store(row_slots_ptr + row_offsets, slot_values, mask=row_offsets < ROWS)
        tl.store(gather_ptr + row_offsets, source_values, mask=row_offsets < ROWS)
        tl.store(inverse_ptr + row_offsets, inverse_values, mask=row_offsets < ROWS)
    tl.store(flags_ptr, valid.to(tl.int32))


@triton.jit(do_not_specialize=["EXPERTS"])
def _prefill_route_rows_kernel(
    group_ptr,
    expert_ptr,
    counts_ptr,
    row_slots_ptr,
    gather_ptr,
    inverse_ptr,
    flags_ptr,
    ROWS: tl.constexpr,
    LOGICAL: tl.constexpr,
    EXPERTS,
    GROUP_TYPE: tl.constexpr,
    LOGICAL_BLOCK: tl.constexpr,
    EXPERT_BLOCK: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    # One program owns one expert's disjoint output/source intervals. This
    # avoids a ROWS-long decode loop, scans, and indirect vector GM addresses.
    slot = tl.program_id(0)
    valid = tl.load(flags_ptr) == 1
    expert = tl.load(expert_ptr + slot)
    logical = tl.arange(0, LOGICAL_BLOCK)
    raw = tl.load(group_ptr + logical, mask=logical < LOGICAL, other=0).to(tl.int64)
    bounded = tl.minimum(tl.maximum(raw, 0), ROWS)
    if GROUP_TYPE == 0:
        previous = tl.load(group_ptr + logical - 1, mask=(logical < LOGICAL) & (logical > 0), other=0).to(tl.int64)
        logical_counts = bounded - tl.minimum(tl.maximum(previous, 0), ROWS)
    else:
        logical_counts = bounded
    logical_counts = tl.where(logical < LOGICAL, logical_counts, 0).to(tl.int32)
    source_start = tl.sum(tl.where(logical < expert, logical_counts, 0), axis=0)
    physical = tl.arange(0, EXPERT_BLOCK)
    physical_counts = tl.load(counts_ptr + physical, mask=physical < EXPERTS, other=0).to(tl.int32)
    physical_start = tl.sum(tl.where(physical < slot, physical_counts, 0), axis=0)
    count = tl.sum(tl.where(physical == slot, physical_counts, 0), axis=0)
    total_rows = tl.sum(physical_counts, axis=0)

    row_offsets = tl.arange(0, ROW_BLOCK)
    active = row_offsets < ROWS
    clear_invalid = (~valid) & (slot == 0)
    clear_padding = valid & (slot == 0) & (row_offsets >= total_rows)
    physical_mask = active & (
        clear_invalid
        | clear_padding
        | (valid & (row_offsets >= physical_start) & (row_offsets < physical_start + count))
    )
    source_mask = active & valid & (row_offsets >= source_start) & (row_offsets < source_start + count)
    slots = tl.where(valid & ~clear_padding, slot, 0)
    sources = tl.where(valid, tl.where(clear_padding, -1, source_start + row_offsets - physical_start), 0)
    inverse = physical_start + row_offsets - source_start
    # Addresses remain affine even with non-power-of-two expert/row tails.
    tl.store(row_slots_ptr + row_offsets, slots, mask=physical_mask)
    tl.store(gather_ptr + row_offsets, sources, mask=physical_mask)
    tl.store(inverse_ptr + row_offsets, inverse, mask=source_mask)


@triton.jit
def _gather_hidden_kernel(
    hidden_ptr, gather_ptr, flags_ptr, output_ptr, ROWS: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    source = tl.load(gather_ptr + row)
    valid = (tl.load(flags_ptr) == 1) & (source >= 0) & (source < ROWS)
    source = tl.where(valid, source, 0)
    values = tl.load(hidden_ptr + source * WIDTH + columns, mask=valid & (columns < WIDTH), other=0)
    tl.store(output_ptr + row * WIDTH + columns, values, mask=columns < WIDTH)


def prepare_routes(hidden, group_list, group_type, expert_ids, logical_slots, *, allow_padding=False, subset=False):
    rows, width = hidden.shape
    output = torch.empty_like(hidden)
    counts = torch.empty_like(expert_ids)
    row_slots = torch.empty((rows,), device=hidden.device, dtype=torch.int64)
    gather = torch.empty_like(row_slots)
    inverse = torch.full_like(row_slots, -1)
    flags = torch.empty((1,), device=hidden.device, dtype=torch.int32)
    _route_metadata_kernel[(1,)](
        group_list,
        expert_ids,
        logical_slots,
        counts,
        row_slots,
        gather,
        inverse,
        flags,
        ROWS=rows,
        LOGICAL=group_list.numel(),
        EXPERTS=expert_ids.numel(),
        GROUP_TYPE=group_type,
        LOGICAL_BLOCK=triton.next_power_of_2(group_list.numel()),
        EXPERT_BLOCK=triton.next_power_of_2(expert_ids.numel()),
        ROW_BLOCK=triton.next_power_of_2(rows),
        MAX_COUNT=MAX_EXPERT_ROWS,
        BUILD_ROWS=rows <= MAX_DECODE_ROUTED_ROWS,
        ALLOW_PADDING=allow_padding,
        SUBSET=subset,
    )
    if rows > MAX_DECODE_ROUTED_ROWS:
        _prefill_route_rows_kernel[(expert_ids.numel(),)](
            group_list,
            expert_ids,
            counts,
            row_slots,
            gather,
            inverse,
            flags,
            ROWS=rows,
            LOGICAL=group_list.numel(),
            EXPERTS=expert_ids.numel(),
            GROUP_TYPE=group_type,
            LOGICAL_BLOCK=triton.next_power_of_2(group_list.numel()),
            EXPERT_BLOCK=triton.next_power_of_2(expert_ids.numel()),
            ROW_BLOCK=triton.next_power_of_2(rows),
        )
    _gather_hidden_kernel[(rows, triton.cdiv(width, COPY_BLOCK))](
        hidden, gather, flags, output, ROWS=rows, WIDTH=width, BLOCK=COPY_BLOCK
    )
    return output, counts, row_slots, inverse, flags


@triton.jit
def _gather_int64_exact(values, indices):
    # Ascend register gather accepts FP32 but not int64. Every 16-bit piece
    # is exactly representable, including counts/prefixes beyond 2**24.
    result = tl.full(indices.shape, 0, tl.int64)
    for shift in tl.static_range(0, 64, INT64_GATHER_BITS):
        piece = ((values >> shift) & ((1 << INT64_GATHER_BITS) - 1)).to(tl.float32)
        result = result | (tl.gather(piece, indices, axis=0).to(tl.int64) << shift)
    return result


@triton.jit
def _sum_int64_counts(values):
    # Nonnegative int64 inputs and at most 256 experts: each partial sum
    # fits int64 even when their mathematical total does not.
    mask = (1 << INT64_SUM_BITS) - 1
    high_max = (1 << (63 - INT64_SUM_BITS)) - 1
    low = tl.sum(values & mask, axis=0)
    high = tl.sum(values >> INT64_SUM_BITS, axis=0) + (low >> INT64_SUM_BITS)
    total = (tl.minimum(high, high_max) << INT64_SUM_BITS) | (low & mask)
    return total, high <= high_max


@triton.jit(do_not_specialize=["ROWS", "EXPERTS"])
def _prefill_metadata_kernel(
    group_ptr,
    expert_ptr,
    logical_slot_ptr,
    counts_ptr,
    source_starts_ptr,
    physical_starts_ptr,
    tile_ends_ptr,
    restore_metadata_ptr,
    flags_ptr,
    ROWS,
    EXPERTS,
    LOGICAL: tl.constexpr,
    GROUP_TYPE: tl.constexpr,
    LOGICAL_BLOCK: tl.constexpr,
    EXPERT_BLOCK: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    ALLOW_PADDING: tl.constexpr,
    SUBSET: tl.constexpr,
):
    logical = tl.arange(0, LOGICAL_BLOCK)
    logical_active = logical < LOGICAL
    raw = tl.load(group_ptr + logical, mask=logical_active, other=0).to(tl.int64)
    bounded = tl.minimum(tl.maximum(raw, 0), ROWS)
    value_valid = (raw >= 0) & (raw <= ROWS)
    if GROUP_TYPE == 0:
        previous = tl.load(group_ptr + logical - 1, mask=logical_active & (logical > 0), other=0).to(tl.int64)
        # Bound before subtraction, so malicious INT64_MIN/MAX never overflow.
        logical_counts = bounded - tl.minimum(tl.maximum(previous, 0), ROWS)
    else:
        logical_counts = bounded
    value_valid = value_valid & (logical_counts >= 0)
    logical_counts = tl.where(logical_active, tl.maximum(logical_counts, 0), 0).to(tl.int64)
    slots = tl.load(logical_slot_ptr + logical, mask=logical_active, other=-1)
    slot_valid = (slots >= 0) & (slots < EXPERTS)
    physical = tl.arange(0, EXPERT_BLOCK)
    physical_active = physical < EXPERTS
    expert = tl.load(expert_ptr + physical, mask=physical_active, other=0)
    expert_valid = (expert >= 0) & (expert < LOGICAL)
    safe_slot = tl.where(slot_valid, slots, 0).to(tl.int32)
    safe_expert = tl.where(expert_valid, expert, 0).to(tl.int32)
    back_id = tl.gather(tl.where(expert_valid, expert, -1).to(tl.float32), safe_slot, axis=0).to(tl.int32)
    back_slot = tl.gather(tl.where(slot_valid, slots, -1).to(tl.float32), safe_expert, axis=0).to(tl.int32)
    mapping_valid = ((slots == -1) & ((logical_counts == 0) | SUBSET)) | (slot_valid & (back_id == logical))
    valid = tl.min(tl.where(logical_active, value_valid & mapping_valid, True).to(tl.int32), axis=0) == 1
    total, total_valid = _sum_int64_counts(logical_counts)
    valid = valid & total_valid & ((total <= ROWS) if ALLOW_PADDING else (total == ROWS))
    valid = valid & (
        tl.min(tl.where(physical_active, expert_valid & (back_slot == physical), True).to(tl.int32), axis=0) == 1
    )
    physical_counts = tl.where(physical_active, _gather_int64_exact(logical_counts, safe_expert), 0)
    if not SUBSET:
        valid = valid & (tl.sum(physical_counts, axis=0) == total)
    source_starts = _gather_int64_exact(tl.cumsum(logical_counts, axis=0) - logical_counts, safe_expert)
    physical_starts = tl.cumsum(physical_counts, axis=0) - physical_counts
    safe_counts = tl.where(valid, physical_counts, 0)
    # Division before addition preserves INT64_MAX without a count+block overflow.
    tile_counts = safe_counts // ROW_BLOCK + (safe_counts % ROW_BLOCK != 0).to(tl.int64)
    tile_ends = tl.cumsum(tile_counts, axis=0)
    tl.store(counts_ptr + physical, tl.where(valid, physical_counts, 0), mask=physical_active)
    tl.store(source_starts_ptr + physical, tl.where(valid, source_starts, 0), mask=physical_active)
    tl.store(physical_starts_ptr + physical, tl.where(valid, physical_starts, 0), mask=physical_active)
    tl.store(tile_ends_ptr + physical, tile_ends, mask=physical_active)
    tl.store(restore_metadata_ptr, tl.sum(safe_counts, axis=0))
    tl.store(restore_metadata_ptr + 1, valid.to(tl.int64))
    tl.store(flags_ptr, valid.to(tl.int32))


@triton.jit(do_not_specialize=["ROWS", "EXPERTS"])
def _prefill_metadata_rows_kernel(
    counts_ptr,
    source_starts_ptr,
    physical_starts_ptr,
    tile_ends_ptr,
    row_slots_ptr,
    gather_ptr,
    inverse_ptr,
    ROWS,
    EXPERTS,
    EXPERT_BLOCK: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    task = tl.program_id(0).to(tl.int64)
    physical = tl.arange(0, EXPERT_BLOCK)
    tile_ends = tl.load(tile_ends_ptr + physical, mask=physical < EXPERTS, other=0).to(tl.int64)
    slot = tl.min(tl.where((physical < EXPERTS) & (task < tile_ends), physical, EXPERTS), axis=0)
    if slot < EXPERTS:
        previous_tiles = tl.load(tile_ends_ptr + slot - 1, mask=slot > 0, other=0).to(tl.int64)
        tile_start = (task - previous_tiles) * ROW_BLOCK
        count = tl.load(counts_ptr + slot).to(tl.int64)
        source_start = tl.load(source_starts_ptr + slot).to(tl.int64)
        physical_start = tl.load(physical_starts_ptr + slot).to(tl.int64)
        local_rows = tile_start + tl.arange(0, ROW_BLOCK).to(tl.int64)
        physical_rows = physical_start + local_rows
        source_rows = source_start + local_rows
        active = (local_rows < count) & (physical_rows < ROWS) & (source_rows < ROWS)
        # Scalar expert offsets plus a fixed-size affine tile, never a vector
        # whose register footprint or compilation cost grows with row capacity.
        tl.store(row_slots_ptr + physical_rows, slot, mask=active)
        tl.store(gather_ptr + physical_rows, source_rows, mask=active)
        tl.store(inverse_ptr + source_rows, physical_rows, mask=active)


@triton.jit(do_not_specialize=["ROWS", "EXPERTS"])
def _restore_metadata_kernel(
    counts_ptr,
    route_flags_ptr,
    metadata_ptr,
    ROWS,
    EXPERTS,
    EXPERT_BLOCK: tl.constexpr,
):
    physical = tl.arange(0, EXPERT_BLOCK)
    counts = tl.load(counts_ptr + physical, mask=physical < EXPERTS, other=0).to(tl.int64)
    valid_counts = tl.min(((counts >= 0) & (counts <= ROWS)).to(tl.int32), axis=0) == 1
    live_rows, total_valid = _sum_int64_counts(tl.minimum(tl.maximum(counts, 0), ROWS))
    route_valid = (tl.load(route_flags_ptr) == 1) & valid_counts & total_valid & (live_rows <= ROWS)
    tl.store(metadata_ptr, tl.where(route_valid, live_rows, 0))
    tl.store(metadata_ptr + 1, route_valid.to(tl.int64))


@triton.jit(do_not_specialize=["ROWS"])
def _restore_kernel(
    output_ptr,
    inverse_ptr,
    metadata_ptr,
    result_ptr,
    block_flags_ptr,
    ROWS,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    COLUMN_BLOCKS: tl.constexpr,
):
    live_rows = tl.load(metadata_ptr).to(tl.int64)
    route_valid = tl.load(metadata_ptr + 1) == 1
    row = tl.program_id(0).to(tl.int64)
    column_block = tl.program_id(1)
    source = tl.load(inverse_ptr + row).to(tl.int64)
    selected = (source >= 0) & (source < live_rows) & (source < ROWS)
    mapping_valid = (source == -1) | selected
    valid = route_valid & mapping_valid
    # Empty/excluded logical rows do no width-sized loads or reductions.
    if route_valid & selected:
        columns = column_block.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        active = columns < WIDTH
        values = tl.load(output_ptr + source * WIDTH + columns, mask=active, other=0).to(tl.float32)
        finite = (values == values) & (tl.abs(values) != float("inf"))
        tl.store(result_ptr + row * WIDTH + columns, tl.where(finite, values, 0), mask=active)
        valid = tl.min(tl.where(active, finite, True).to(tl.int32), axis=0) == 1
    tl.store(block_flags_ptr + row * COLUMN_BLOCKS + column_block, valid.to(tl.int32))


def prepare_metadata(hidden, group_list, group_type, expert_ids, logical_slots, *, allow_padding=False, subset=False):
    rows = hidden.shape[0]
    experts = expert_ids.numel()
    counts = torch.empty_like(expert_ids)
    source_starts = torch.empty_like(expert_ids)
    physical_starts = torch.empty_like(expert_ids)
    tile_ends = torch.empty_like(expert_ids)
    restore_metadata = torch.empty((2,), device=hidden.device, dtype=torch.int64)
    row_slots = torch.zeros((rows,), device=hidden.device, dtype=torch.int64)
    gather = torch.full_like(row_slots, -1)
    inverse = torch.full_like(row_slots, -1)
    flags = torch.empty((1,), device=hidden.device, dtype=torch.int32)
    _prefill_metadata_kernel[(1,)](
        group_list,
        expert_ids,
        logical_slots,
        counts,
        source_starts,
        physical_starts,
        tile_ends,
        restore_metadata,
        flags,
        ROWS=rows,
        EXPERTS=experts,
        LOGICAL=group_list.numel(),
        GROUP_TYPE=group_type,
        LOGICAL_BLOCK=triton.next_power_of_2(group_list.numel()),
        EXPERT_BLOCK=triton.next_power_of_2(experts),
        ROW_BLOCK=PREFILL_ROW_BLOCK,
        ALLOW_PADDING=allow_padding,
        SUBSET=subset,
    )
    if rows:
        # Sum of expert tile counts is <= ceil(rows/block)+experts-1.
        # This removes the experts*ceil(rows/block) mostly-empty launch grid.
        _prefill_metadata_rows_kernel[(triton.cdiv(rows, PREFILL_ROW_BLOCK) + experts - 1,)](
            counts,
            source_starts,
            physical_starts,
            tile_ends,
            row_slots,
            gather,
            inverse,
            ROWS=rows,
            EXPERTS=experts,
            EXPERT_BLOCK=triton.next_power_of_2(experts),
            ROW_BLOCK=PREFILL_ROW_BLOCK,
        )
    return counts, row_slots, gather, inverse, flags, restore_metadata


def restore(output, inverse, counts, flags, result, *, metadata=None):
    rows, width = output.shape
    block = min(RESTORE_COLUMN_BLOCK, triton.next_power_of_2(width))
    column_blocks = triton.cdiv(width, block)
    block_flags = torch.empty((rows * column_blocks,), device=output.device, dtype=torch.int32)
    if metadata is None:
        metadata = torch.empty((2,), device=output.device, dtype=torch.int64)
        _restore_metadata_kernel[(1,)](
            counts,
            flags,
            metadata,
            ROWS=rows,
            EXPERTS=counts.numel(),
            EXPERT_BLOCK=triton.next_power_of_2(counts.numel()),
        )
    if rows:
        _restore_kernel[(rows, column_blocks)](
            output,
            inverse,
            metadata,
            result,
            block_flags,
            ROWS=rows,
            WIDTH=width,
            BLOCK=block,
            COLUMN_BLOCKS=column_blocks,
        )
    else:
        # No row programs run, but invalid standalone metadata must still fail.
        return metadata[1:2].to(torch.int32)
    return block_flags
