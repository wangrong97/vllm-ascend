# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture-safe routing metadata and output restoration for the GS32 FFN."""

import torch

from vllm_ascend.ops.fused_moe.vq2a8_gs32_activation import _require_npu_tensor

MAX_ROUTE_EXPERTS = 256
NATIVE_ALIGNMENT = 32


def prepare_route_metadata(
    hidden, group_list, group_type, expert_ids, logical_slots, *, allow_padding=False, subset=False, retained=None
):
    from vllm_ascend.ops.triton.activation.vq2a8_gs32_routes import prepare_metadata as launch

    options = {} if retained is None else {"retained": retained}
    return launch(
        hidden, group_list, group_type, expert_ids, logical_slots, allow_padding=allow_padding, subset=subset, **options
    )


def restore_routes(output, inverse, counts, flags, result, *, metadata=None):
    from vllm_ascend.ops.triton.activation.vq2a8_gs32_routes import restore as launch

    return launch(output, inverse, counts, flags, result, metadata=metadata)


def prepare_identity_route_metadata(
    hidden, group_list, group_type, expert_ids, logical_slots, *, allow_padding=False, retained=None
):
    from vllm_ascend.ops.triton.activation.vq2a8_gs32_routes import prepare_identity_metadata as launch

    return launch(hidden, group_list, group_type, expert_ids, logical_slots, allow_padding=allow_padding)


def prepare_identity_metadata(
    hidden, group_list, group_type, expert_ids, logical_slots, *, allow_padding=False, retained=None
):
    """Validate resident counts without constructing row permutations.

    Resident expert IDs must be strictly increasing. Missing logical experts
    must have zero counts, so every valid live row already has resident order.
    Return safe int64 counts, int32 validity and an int64 [live_rows, valid]
    summary. Invalid metadata produces zero counts and zero live rows.
    Padding is the trailing capacity after live_rows and must not be read by
    activation preparation. All returned tensors belong to this forward;
    there are no additional internal scratch tensors to retain for capture.
    """
    _validate_routes(hidden, group_list, group_type, expert_ids, logical_slots, allow_padding, False)
    with torch.npu.device(hidden.device):
        return prepare_identity_route_metadata(
            hidden,
            group_list,
            group_type,
            expert_ids,
            logical_slots,
            allow_padding=allow_padding,
            retained=retained,
        )


def prepare_metadata(
    hidden, group_list, group_type, expert_ids, logical_slots, *, allow_padding=False, subset=False, retained=None
):
    """Return counts, gather, inverse, validity and a restore summary.

    Row vectors retain the full hidden capacity. Unselected or padded rows
    have inverse=-1; physical padding has gather=-1. Invalid dynamic metadata
    produces zero counts and only -1 gather/inverse entries.
    Counts and offsets remain int64 for both decode and prefill capacities.
    The caller must retain and check the returned device validity.
    The int64 [2] summary contains validated live rows and validity, allowing
    restore to reuse that reduction within this same forward.
    During capture, retained collects internal scratch tensors that are not
    part of this public metadata tuple but are still referenced on replay.
    """
    _validate_routes(hidden, group_list, group_type, expert_ids, logical_slots, allow_padding, subset)
    with torch.npu.device(hidden.device):
        options = {} if retained is None else {"retained": retained}
        return prepare_route_metadata(
            hidden,
            group_list,
            group_type,
            expert_ids,
            logical_slots,
            allow_padding=allow_padding,
            subset=subset,
            **options,
        )


def _validate_routes(hidden, group_list, group_type, expert_ids, logical_slots, allow_padding, subset):
    for name, tensor in (
        ("hidden", hidden),
        ("group_list", group_list),
        ("expert_ids", expert_ids),
        ("logical_slots", logical_slots),
    ):
        _require_npu_tensor(tensor, name)
        if tensor.device != hidden.device:
            raise ValueError("GS32 device routes require all tensors on the same NPU.")
    if (
        hidden.dtype != torch.bfloat16
        or hidden.ndim != 2
        or hidden.shape[0] < 0
        or hidden.shape[1] < NATIVE_ALIGNMENT
        or hidden.shape[1] % NATIVE_ALIGNMENT
    ):
        raise ValueError("GS32 device routes require BF16 [R,K], R>=0 and positive K divisible by 32.")
    if type(group_type) is not int or group_type not in (0, 1):
        raise ValueError("GS32 device routes group_type must be 0 (prefix) or 1 (counts).")
    if type(allow_padding) is not bool or type(subset) is not bool:
        raise ValueError("GS32 device routes allow_padding and subset must be bools.")
    if (
        group_list.dtype not in (torch.int32, torch.int64)
        or group_list.ndim != 1
        or not 1 <= group_list.numel() <= MAX_ROUTE_EXPERTS
    ):
        raise ValueError("GS32 device routes require int32/int64 group_list[L], 1<=L<=256.")
    if expert_ids.dtype != torch.int64 or expert_ids.ndim != 1 or not 1 <= expert_ids.numel() <= MAX_ROUTE_EXPERTS:
        raise ValueError("GS32 device routes require int64 expert_ids[E], 1<=E<=256.")
    if logical_slots.dtype != torch.int64 or logical_slots.shape != group_list.shape:
        raise ValueError("GS32 device routes require int64 logical_slots[L].")


def restore(output, inverse, counts, flags, result, *, metadata=None):
    """Write selected logical rows in place and return deferred block validity.

    Only physical rows below sum(counts) are loaded and checked for finite
    values; native output padding may be uninitialized. Unselected result
    rows are untouched. Invalid mappings or nonfinite selected values set
    validity to zero, and nonfinite selected values are replaced by zero.
    Optional metadata must be the unmodified summary returned by
    prepare_metadata for these counts in the same forward and stream.
    """
    for name, tensor in (
        ("output", output),
        ("inverse", inverse),
        ("counts", counts),
        ("flags", flags),
        ("result", result),
    ):
        _require_npu_tensor(tensor, name)
        if tensor.device != output.device:
            raise ValueError("GS32 restore tensors must share the same NPU device.")
    if output.ndim != 2 or output.shape[1] < 1 or output.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("GS32 restore output requires BF16, FP16 or FP32 [R,N].")
    if result.shape != output.shape or result.dtype != output.dtype:
        raise ValueError("GS32 restore result must match output shape and dtype.")
    if inverse.dtype != torch.int64 or inverse.shape != (output.shape[0],):
        raise ValueError("GS32 restore inverse requires int64 [R].")
    if counts.dtype != torch.int64 or counts.ndim != 1 or not 1 <= counts.numel() <= MAX_ROUTE_EXPERTS:
        raise ValueError("GS32 restore counts requires int64 [E], 1<=E<=256.")
    if flags.dtype != torch.int32 or flags.shape != (1,):
        raise ValueError("GS32 restore flags requires int32 [1].")
    if metadata is not None:
        _require_npu_tensor(metadata, "GS32 restore metadata")
        if metadata.device != output.device or metadata.dtype != torch.int64 or metadata.shape != (2,):
            raise ValueError("GS32 restore metadata requires int64 [2] on the output device.")
    with torch.npu.device(output.device):
        if metadata is None:
            return restore_routes(output, inverse, counts, flags, result)
        return restore_routes(output, inverse, counts, flags, result, metadata=metadata)
