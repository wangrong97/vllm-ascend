# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GS32 activation entry points with explicit eager or deferred validation."""

import math

import torch

from vllm_ascend.quantization.methods.w2a8.vq2a8_gs32_staged import MX_GROUP, staged_swiglu_unchecked


def prepare_activation(
    hidden, perm, row_slots, raw_gate=None, *, swiglu_limit=None, source_rows=None, route_flags=None
):
    if swiglu_limit is not None or source_rows is not None or route_flags is not None:
        _validate_prepare(
            hidden, perm, row_slots, raw_gate, swiglu_limit, source_rows=source_rows, route_flags=route_flags
        )
    # Lazy worker loading keeps CPU contract tests independent of Triton/NPU.
    from vllm_ascend.ops.triton.activation.vq2a8_gs32 import prepare_activation as launch

    return launch(
        hidden, perm, row_slots, raw_gate, swiglu_limit=swiglu_limit, source_rows=source_rows, route_flags=route_flags
    )


def finite_flags(tensor):
    from vllm_ascend.ops.triton.activation.vq2a8_gs32 import finite_flags as launch

    return launch(tensor)


def finish_activation_flags(output, gate_flags, down_flags, *, route_flags=None, native_flags=None):
    from vllm_ascend.ops.triton.activation.vq2a8_gs32 import finish_flags as launch

    return launch(output, gate_flags, down_flags, route_flags=route_flags, native_flags=native_flags)


def _require_npu_tensor(tensor, name):
    if not isinstance(tensor, torch.Tensor) or tensor.device.type != "npu" or tensor.device.index is None:
        raise ValueError(f"{name} requires an explicit NPU device; CPU execution is unsupported.")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous.")


def _require_eager():
    capturing = getattr(torch.npu, "is_current_stream_capturing", None)
    if capturing is not None and capturing():
        raise RuntimeError("GS32 fused activation does not support graph capture.")


def _validate_prepare(hidden, perm, row_slots, raw_gate, swiglu_limit=None, *, source_rows=None, route_flags=None):
    if swiglu_limit is not None:
        if (
            isinstance(swiglu_limit, bool)
            or not isinstance(swiglu_limit, (int, float))
            or not math.isfinite(swiglu_limit)
            or swiglu_limit < 0
        ):
            raise ValueError("Fused SwiGLU limit must be a finite non-negative Python number.")
        if raw_gate is not None:
            raise ValueError("Fused SwiGLU uses hidden as GateUp; raw_gate and swiglu_limit cannot be combined.")
    tensors = {"hidden": hidden, "row_slots": row_slots}
    if perm is not None:
        tensors["perm"] = perm
    if raw_gate is not None:
        tensors["raw_gate"] = raw_gate
    if source_rows is not None:
        tensors["source_rows"] = source_rows
    if route_flags is not None:
        tensors["route_flags"] = route_flags
    for name, tensor in tensors.items():
        _require_npu_tensor(tensor, name)
        if tensor.device != hidden.device:
            raise ValueError("GS32 fused activation tensors must share the same device.")
    if hidden.dtype != torch.bfloat16 or hidden.ndim != 2 or hidden.shape[1] < MX_GROUP or hidden.shape[1] % MX_GROUP:
        raise ValueError("GS32 fused activation requires BF16 [M,K] with positive K divisible by 32.")
    rows, width = hidden.shape
    if swiglu_limit is not None:
        if width % (2 * MX_GROUP):
            raise ValueError("Fused SwiGLU requires natural BF16 [M,2K] with positive K divisible by 32.")
        width //= 2
    if perm is not None and (
        perm.dtype not in (torch.int32, torch.int64) or perm.ndim != 2 or perm.shape[0] < 1 or perm.shape[1] != width
    ):
        raise ValueError("GS32 fused activation requires integer perm [E,K], or None for natural MXFP4 columns.")
    if row_slots.dtype != torch.int64 or row_slots.shape != (rows,):
        raise ValueError("GS32 fused activation requires int64 row_slots with shape [M].")
    if raw_gate is not None and (raw_gate.dtype != torch.bfloat16 or raw_gate.shape != (rows, 2 * width)):
        raise ValueError("GS32 fused activation raw_gate must be BF16 [M,2K].")
    if source_rows is not None and (source_rows.dtype != torch.int64 or source_rows.shape != (rows,)):
        raise ValueError("GS32 fused activation source_rows must be int64 [M].")
    if route_flags is not None and (route_flags.dtype != torch.int32 or route_flags.shape != (1,)):
        raise ValueError("GS32 fused activation route_flags must be int32 [1].")
    return rows, width


def prepare(hidden, perm, row_slots, raw_gate=None, *, swiglu_limit=None):
    """Prepare MXFP8; optional SwiGLU consumes natural BF16 [M,2K]."""
    rows, width = _validate_prepare(hidden, perm, row_slots, raw_gate, swiglu_limit)
    with torch.npu.device(hidden.device):
        _require_eager()
        if rows == 0:
            return hidden.new_empty((0, width), dtype=torch.uint8), hidden.new_empty(
                (0, width // MX_GROUP), dtype=torch.uint8
            )
        options = {} if swiglu_limit is None else {"swiglu_limit": swiglu_limit}
        data, scales, flags = prepare_activation(hidden, perm, row_slots, raw_gate, **options)
        # Exactly one small host transfer and decision, before native launch.
        if not all(flags.cpu().tolist()):
            raise ValueError("GS32 fused activation inputs contain non-finite values or invalid indices.")
        return data, scales


def prepare_unchecked(hidden, perm, row_slots, raw_gate=None, *, swiglu_limit=None, source_rows=None, route_flags=None):
    """Return device validity that must be checked before accepting results.

    Metadata checks remain synchronous host checks. Numerical flags remain
    device tensors so a graph owner can combine them without a host transfer.
    The kernel sanitizes invalid numerical inputs before emitting FP8 bytes.
    A non-None limit fuses SwiGLU over hidden [M,2K] before preparing [M,K];
    zero disables clipping, while None retains the ordinary [M,K] path.
    Optional int64 source_rows [M] gathers logical input rows directly into
    quantized physical rows. A source of -1 is valid inactive padding. Other
    out-of-range sources, or route_flags [1] other than one, produce safe zero
    rows and invalid flags without reading the input or computing activation.
    """
    rows, width = _validate_prepare(
        hidden, perm, row_slots, raw_gate, swiglu_limit, source_rows=source_rows, route_flags=route_flags
    )
    with torch.npu.device(hidden.device):
        if rows == 0:
            return (
                hidden.new_empty((0, width), dtype=torch.uint8),
                hidden.new_empty((0, width // MX_GROUP), dtype=torch.uint8),
                hidden.new_empty((0,), dtype=torch.int32),
            )
        options = {} if swiglu_limit is None else {"swiglu_limit": swiglu_limit}
        if source_rows is not None:
            options["source_rows"] = source_rows
        if route_flags is not None:
            options["route_flags"] = route_flags
        return prepare_activation(hidden, perm, row_slots, raw_gate, **options)


def finish_flags(output, gate_flags, down_flags, *, route_flags=None, native_flags=None):
    """Combine both preparation flags and output finiteness without a CPU read."""
    _require_npu_tensor(output, "GS32 output")
    if output.ndim != 2 or output.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("GS32 output requires a BF16, FP16 or FP32 [M,N] tensor.")
    for name, flags in (("gate_flags", gate_flags), ("down_flags", down_flags)):
        _require_npu_tensor(flags, name)
        if flags.device != output.device or flags.dtype != torch.int32 or flags.shape != (output.shape[0],):
            raise ValueError(f"{name} must be int32 [M] on the output device.")
    external = {"route_flags": route_flags, "native_flags": native_flags}
    for name, flags in external.items():
        if flags is not None:
            _require_npu_tensor(flags, name)
            if flags.device != output.device or flags.dtype != torch.int32 or flags.ndim != 1 or not flags.numel():
                raise ValueError(f"{name} must be a nonempty int32 vector on the output device.")
    with torch.npu.device(output.device):
        if route_flags is None and native_flags is None:
            return finish_activation_flags(output, gate_flags, down_flags)
        return finish_activation_flags(output, gate_flags, down_flags, **external)


def require_finite_fused(tensor, name):
    _require_npu_tensor(tensor, name)
    if tensor.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(f"{name} requires a BF16, FP16 or FP32 tensor.")
    with torch.npu.device(tensor.device):
        _require_eager()
        if tensor.numel() and not all(finite_flags(tensor).cpu().tolist()):
            raise ValueError(f"{name} contains NaN or Inf.")


def swiglu_unchecked(gate_up, limit):
    """Keep staged Torch arithmetic; the subsequent prepare checks raw GateUp."""
    return staged_swiglu_unchecked(gate_up, limit)
