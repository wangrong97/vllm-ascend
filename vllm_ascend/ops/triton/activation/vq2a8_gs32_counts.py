# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-launch, overflow-safe count copying and validation for native routes."""

import torch
from vllm.triton_utils import tl, triton

INT64_SUM_BITS = tl.constexpr(32)


@triton.jit(do_not_specialize=["ROWS", "GROUPS"])
def _sanitize_device_counts_kernel(
    counts_ptr,
    safe_counts_ptr,
    status_ptr,
    ROWS,
    GROUPS,
    BLOCK: tl.constexpr,
    ALLOW_PADDING: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    active = offsets < GROUPS
    raw = tl.load(counts_ptr + offsets, mask=active, other=0).to(tl.int64)
    bounded = tl.minimum(tl.maximum(raw, 0), ROWS)
    values_valid = tl.min(tl.where(active, raw == bounded, True).to(tl.int32), axis=0) == 1
    # Split before reduction: even 256 copies of INT64_MAX cannot overflow
    # either partial sum. Raw counts never participate in an overflowing sum.
    low_mask = (1 << INT64_SUM_BITS) - 1
    high_max = (1 << (63 - INT64_SUM_BITS)) - 1
    low = tl.sum(bounded & low_mask, axis=0)
    high = tl.sum(bounded >> INT64_SUM_BITS, axis=0) + (low >> INT64_SUM_BITS)
    total = (tl.minimum(high, high_max) << INT64_SUM_BITS) | (low & low_mask)
    total_valid = high <= high_max
    fits_rows = (total <= ROWS) if ALLOW_PADDING else (total == ROWS)
    valid = values_valid & total_valid & fits_rows
    safe_counts = tl.where(valid, bounded, 0)
    tl.store(safe_counts_ptr + offsets, safe_counts, mask=active)
    tl.store(status_ptr, valid.to(tl.int32))


def sanitize_device_counts(counts, rows, allow_padding):
    # Fresh allocations are essential: mutation of caller counts after this
    # launch must not affect any subsequent FFN launch through DeviceRoutes.
    safe_counts = torch.empty_like(counts)
    status = torch.empty((1,), dtype=torch.int32, device=counts.device)
    _sanitize_device_counts_kernel[(1,)](
        counts,
        safe_counts,
        status,
        ROWS=rows,
        GROUPS=counts.numel(),
        BLOCK=triton.next_power_of_2(counts.numel()),
        ALLOW_PADDING=allow_padding,
        enable_auto_vectorize_v2=False,
    )
    return safe_counts, status
