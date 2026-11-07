# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Input gather and MXFP8 preparation for the complete GS32 FFN."""

import torch

MX_GROUP = 32


def prepare_activation(hidden, *, source_rows=None, route_flags=None, route_metadata=None):
    # Lazy worker loading keeps CPU contract tests independent of Triton/NPU.
    from vllm_ascend.ops.triton.activation.vq2a8_gs32 import prepare_activation as launch

    return launch(hidden, source_rows=source_rows, route_flags=route_flags, route_metadata=route_metadata)


def _require_npu_tensor(tensor, name):
    if not isinstance(tensor, torch.Tensor) or tensor.device.type != "npu" or tensor.device.index is None:
        raise ValueError(f"{name} requires an explicit NPU device; CPU execution is unsupported.")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous.")


def _validate_prepare(hidden, *, source_rows=None, route_flags=None, route_metadata=None):
    tensors = {"hidden": hidden}
    if source_rows is not None:
        tensors["source_rows"] = source_rows
    if route_flags is not None:
        tensors["route_flags"] = route_flags
    if route_metadata is not None:
        tensors["route_metadata"] = route_metadata
        if source_rows is not None:
            raise ValueError("GS32 identity route_metadata and source_rows are mutually exclusive.")
    for name, tensor in tensors.items():
        _require_npu_tensor(tensor, name)
        if tensor.device != hidden.device:
            raise ValueError("GS32 input preparation tensors must share the same device.")
    if hidden.dtype != torch.bfloat16 or hidden.ndim != 2 or hidden.shape[1] < MX_GROUP or hidden.shape[1] % MX_GROUP:
        raise ValueError("GS32 input preparation requires BF16 [M,K] with positive K divisible by 32.")
    rows, width = hidden.shape
    if source_rows is not None and (source_rows.dtype != torch.int64 or source_rows.shape != (rows,)):
        raise ValueError("GS32 input preparation source_rows must be int64 [M].")
    if route_flags is not None and (route_flags.dtype != torch.int32 or route_flags.shape != (1,)):
        raise ValueError("GS32 input preparation route_flags must be int32 [1].")
    if route_metadata is not None and (route_metadata.dtype != torch.int64 or route_metadata.shape != (2,)):
        raise ValueError("GS32 identity route_metadata must be int64 [2].")
    return rows, width


def prepare_unchecked(hidden, *, source_rows=None, route_flags=None, route_metadata=None):
    """Return MXFP8 bytes, scales and validity for the forward owner to check.

    Host metadata checks remain immediate. Numerical validity stays on device
    until the owning forward completes. Invalid values become safe zero bytes.
    Optional source_rows gathers logical inputs into physical expert order;
    -1 denotes inactive padding, while other invalid indices fail validity.
    Identity route_metadata instead contains [live_rows, valid]; padding never
    reads hidden, even if unused input rows contain NaN or Inf.
    """
    rows, width = _validate_prepare(
        hidden, source_rows=source_rows, route_flags=route_flags, route_metadata=route_metadata
    )
    with torch.npu.device(hidden.device):
        if rows == 0:
            return (
                hidden.new_empty((0, width), dtype=torch.uint8),
                hidden.new_empty((0, width // MX_GROUP), dtype=torch.uint8),
                hidden.new_empty((0,), dtype=torch.int32),
            )
        options = {} if route_metadata is None else {"route_metadata": route_metadata}
        return prepare_activation(hidden, source_rows=source_rows, route_flags=route_flags, **options)


def finish_full_ffn(output, middle_scales, input_flags, route_flags, native_flags, route_metadata, validity):
    """Sanitize live outputs and overwrite the owning forward's device status.

    The native bridge initializes output/scale padding on every execution.
    This also zeros excluded output rows without reading them and checks the
    input flags and all intermediate scale bytes, so an earlier nonfinite
    value cannot disappear behind SwiGLU clipping or quantization.
    Returns scratch block flags that the graph owner retains until release.
    """
    from vllm_ascend.ops.triton.activation.vq2a8_gs32 import finish_full_ffn as launch

    tensors = {
        "output": output,
        "middle_scales": middle_scales,
        "input_flags": input_flags,
        "route_flags": route_flags,
        "native_flags": native_flags,
        "route_metadata": route_metadata,
        "validity": validity,
    }
    for name, tensor in tensors.items():
        _require_npu_tensor(tensor, name)
        if tensor.device != output.device:
            raise ValueError("GS32 full FFN validity tensors must share the same device.")
    if output.dtype != torch.bfloat16 or output.ndim != 2 or min(output.shape) < 1:
        raise ValueError("GS32 full FFN validity requires positive BF16 output [R,D].")
    rows = output.shape[0]
    if (
        middle_scales.dtype != torch.uint8
        or middle_scales.ndim != 2
        or middle_scales.shape[0] != rows
        or middle_scales.shape[1] < 1
    ):
        raise ValueError("GS32 full FFN middle_scales must be uint8 [R,S], S>0.")
    for name, shape in (("input_flags", (rows,)), ("route_flags", (1,)), ("native_flags", (1,))):
        if tensors[name].dtype != torch.int32 or tensors[name].shape != shape:
            raise ValueError(f"GS32 full FFN {name} must be int32 {shape}.")
    if route_metadata.dtype != torch.int64 or route_metadata.shape != (2,):
        raise ValueError("GS32 full FFN route_metadata must be int64 [2].")
    if validity.dtype != torch.bool or validity.shape != ():
        raise ValueError("GS32 full FFN validity must be a boolean scalar.")
    # Output/status writes must not overwrite any of the inputs being checked.
    # Only inspect host storage metadata here; no device synchronization.
    ranges = [
        (name, tensor.data_ptr(), tensor.data_ptr() + tensor.numel() * tensor.element_size())
        for name, tensor in tensors.items()
    ]
    for writable in ("output", "validity"):
        _, start, end = next(entry for entry in ranges if entry[0] == writable)
        if any(name != writable and start < other_end and other_start < end for name, other_start, other_end in ranges):
            raise ValueError("GS32 full FFN output/validity must not overlap other tensors.")
    with torch.npu.device(output.device):
        return launch(output, middle_scales, input_flags, route_flags, native_flags, route_metadata, validity)
