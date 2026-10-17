# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adapt TP query heads to the FP8 shared-KV attention kernel's minimum.

Both the attention and its AICPU metadata operator require 64 or 128 heads.
TP can leave only 8/16/32 heads on a rank. Shared-KV attention computes each
query head independently, so zero padding followed by slicing preserves the
local heads without changing weights, KV storage, or output projections.
"""

import torch
import torch.nn.functional as F

KV_QUANT_MIN_QUERY_HEADS = 64


def _kernel_query_heads(num_heads: int) -> int:
    if num_heads <= 0:
        raise ValueError("FP8 shared-KV attention requires a positive query head count.")
    return max(num_heads, KV_QUANT_MIN_QUERY_HEADS)


def kv_quant_sparse_attn_metadata(*, num_heads_q: int, **kwargs) -> torch.Tensor:
    """Build scheduling metadata for the same padded head count as attention."""
    return torch.ops._C_ascend.npu_kv_quant_sparse_attn_sharedkv_metadata(
        num_heads_q=_kernel_query_heads(num_heads_q), **kwargs
    )


def kv_quant_sparse_attn(q: torch.Tensor, *, sinks: torch.Tensor | None = None, **kwargs):
    """Pad small TP shards using capture-compatible device tensor operations."""
    if q.ndim < 2:
        raise ValueError("FP8 shared-KV query must have a head and feature dimension.")
    local_heads = q.shape[-2]
    kernel_heads = _kernel_query_heads(local_heads)
    if kernel_heads == local_heads:
        return torch.ops._C_ascend.npu_kv_quant_sparse_attn_sharedkv(q, sinks=sinks, **kwargs)

    layout = kwargs.get("layout_q", "BSND")
    if (layout, q.ndim) not in (("TND", 3), ("BSND", 4)):
        raise ValueError(f"FP8 shared-KV query padding requires TND or BSND, got {layout} with rank {q.ndim}.")
    padding = kernel_heads - local_heads
    if sinks is not None:
        if sinks.ndim != 1 or sinks.shape[0] != local_heads:
            raise ValueError("FP8 shared-KV sinks must contain one value per local query head.")
        sinks = F.pad(sinks, (0, padding))
    q = F.pad(q, (0, 0, 0, padding))
    output, lse = torch.ops._C_ascend.npu_kv_quant_sparse_attn_sharedkv(q, sinks=sinks, **kwargs)
    # The Torch binding allocates LSE with q's shape and a final dimension of 1.
    # The usual return_softmax_lse=False path returns an empty tensor unchanged.
    if kwargs.get("return_softmax_lse", False):
        lse = lse[..., :local_heads, :].contiguous()
    return output[..., :local_heads, :].contiguous(), lse
