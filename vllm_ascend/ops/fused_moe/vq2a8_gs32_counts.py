# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Register the device count sanitizer after loading the GS32 Torch bridge.

Importing this module registers a dispatcher implementation only. Triton is
loaded lazily when the first route is prepared, before graph capture warmup.
"""

import torch

MAX_ROUTE_EXPERTS = 256
INT64_MAX = (1 << 63) - 1
SANITIZE_OPERATOR = "vq2a8_gs32_stage::_sanitize_device_counts"
SANITIZE_SCHEMA = f"{SANITIZE_OPERATOR}(Tensor counts, int rows, bool allow_padding) -> (Tensor, Tensor)"


def _sanitize_device_counts(counts, rows, allow_padding):
    if (
        not isinstance(counts, torch.Tensor)
        or counts.device.type != "npu"
        or counts.device.index is None
        or counts.dtype != torch.int64
        or counts.ndim != 1
        or not counts.is_contiguous()
        or not 1 <= counts.numel() <= MAX_ROUTE_EXPERTS
    ):
        raise ValueError("GS32 count sanitizer requires contiguous int64 counts[1..256] on an explicit NPU device.")
    if type(rows) is not int or not 1 <= rows <= INT64_MAX or type(allow_padding) is not bool:
        raise ValueError("GS32 count sanitizer requires positive int64 rows and boolean allow_padding.")
    from vllm_ascend.ops.triton.activation.vq2a8_gs32_counts import sanitize_device_counts

    with torch.npu.device(counts.device):
        return sanitize_device_counts(counts, rows, allow_padding)


def register_counts_implementation():
    """Require the new bridge schema; registration allocates no device storage."""
    try:
        schema = torch._C._dispatch_find_schema_or_throw(SANITIZE_OPERATOR, "").schema()
    except RuntimeError as error:
        raise RuntimeError("GS32 count sanitizer schema is missing; rebuild and load the GS32 Torch bridge.") from error
    if " ".join(str(schema).split()) != SANITIZE_SCHEMA:
        raise RuntimeError(f"GS32 count sanitizer schema mismatch: {schema}")
    # The serving package and standalone build checker can import this file
    # under different module names in one process. Retain one dispatcher owner.
    if not torch._C._dispatch_has_kernel_for_dispatch_key(SANITIZE_OPERATOR, "PrivateUse1"):
        torch.library.impl(SANITIZE_OPERATOR, "PrivateUse1")(_sanitize_device_counts)


register_counts_implementation()
