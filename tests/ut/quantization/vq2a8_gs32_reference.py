# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test-only compressed layout fixtures and independent CPU numerical oracles.

Synthetic payloads include column scales to stress transform ordering; this is
not an artifact loader or a serving fallback. Real-weight tests use source-v3
through tests.e2e.nightly.vq2a8_gs32_helpers.
"""

from dataclasses import dataclass

import numpy as np
import torch

from vllm_ascend.quantization.methods.w2a8.vq2a8_gs32_staged import (
    MX_GROUP,
    quantize_mxfp8_k32,
    require_finite,
)
from vllm_ascend.quantization.methods.w2a8.vq2a8_gs32_staged import (
    staged_swiglu as staged_swiglu,
)


@dataclass(frozen=True)
class SyntheticProjectionSpec:
    kind: str
    rows: int
    columns: int


def _validate_payload(payload, spec):
    """Validate test inputs before reference indexing; no checkpoint parsing."""
    n, k = spec.rows, spec.columns
    if n < 32 or k < 32 or n % 32 or k % 32:
        raise ValueError("Synthetic GS32 geometry must use positive N32/K32 dimensions.")
    shapes = {
        "packed_indices": (n * k // 16,),
        "codebooks": (k // 32, n // 32, 16, 2),
        "perm": (k,),
        "weight_scale": (k,),
        "weight_bias": (k,),
        "row_scale": (n,),
    }
    if set(payload) != set(shapes):
        raise ValueError("Synthetic GS32 payload has unexpected fields.")
    for name, shape in shapes.items():
        value = payload[name]
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu" or tuple(value.shape) != shape:
            raise ValueError(f"Invalid synthetic {name} geometry/device.")
        if not value.is_contiguous() or not torch.isfinite(value.float()).all():
            raise ValueError(f"Synthetic {name} must be contiguous and finite.")
    if torch.count_nonzero(payload["weight_bias"]):
        raise ValueError("Nonzero weight bias is unsupported by the native projection.")
    if not torch.equal(torch.sort(payload["perm"].long()).values, torch.arange(k)):
        raise ValueError("Synthetic input permutation is invalid.")


GS32_TILE = 32
GS32_VECTOR_LENGTH = 2
GS32_PAIRS = GS32_TILE // GS32_VECTOR_LENGTH
GS32_ZN_K = 16
GS32_INDICES_PER_WORD = 8


def _indices(payload, spec):
    """Producer order: K tile, N tile, column, adjacent output-row pair."""
    words = payload["packed_indices"].numpy().view(np.uint32)
    shifts = np.arange(GS32_INDICES_PER_WORD, dtype=np.uint32) * np.uint32(4)
    indices = ((words[:, None] >> shifts) & np.uint32(15)).astype(np.uint8)
    return indices.reshape(spec.columns // GS32_TILE, spec.rows // GS32_TILE, GS32_TILE, GS32_PAIRS)


def _row_tiles(spec):
    order = np.arange(spec.rows // GS32_TILE, dtype=np.int64)
    if spec.kind == "gate_up":
        if spec.rows % 128:
            raise ValueError("Fused Gate/Up requires total output width divisible by 128.")
        # The producer stores all gate rows, then all up rows. The device's
        # output tiles hold gate32, up32, gate32, up32, ... . Pair order within
        # each group remains untouched; arbitrary VQ pairs are never split.
        order = order.reshape(2, -1).T.reshape(-1)
    elif spec.kind != "down":
        raise ValueError("GS32 kind must be gate_up or down.")
    return order


def repack_gs32_matrix(payload, spec, *, down_stage=False):
    """Convert one CPU matrix into compressed mixed-FFN consumer fields.

    The uint64 row coefficients deliberately stay in logical output order:
    Gate W2 AIC gathers gate/up scales separately using logical_hidden_offset.
    Down permutation and column scales cannot be folded into these weights.
    ``down_stage=True`` keeps natural row order even for GateUp, for reuse of
    the existing DownW2-only entry as an ordinary projection. Never pass this
    layout to the fused GateW2 stage.
    """
    _validate_payload(payload, spec)
    if spec.columns % 64:
        raise ValueError("The mixed FFN consumer requires K divisible by 64.")
    order = np.arange(spec.rows // GS32_TILE) if down_stage else _row_tiles(spec)
    indices = _indices(payload, spec).transpose(1, 0, 2, 3)[order]
    indices = np.ascontiguousarray(indices).reshape(
        spec.rows // GS32_TILE, spec.columns // GS32_ZN_K, GS32_ZN_K, GS32_PAIRS
    )
    packed = np.ascontiguousarray(indices[..., 0::2] | (indices[..., 1::2] << np.uint8(4)))
    books = payload["codebooks"].view(torch.uint8).numpy().transpose(1, 0, 2, 3)[order]
    pair_lut = np.ascontiguousarray(books).reshape(spec.rows // GS32_TILE, spec.columns // GS32_TILE, 32)
    row_bits = payload["row_scale"].numpy().view(np.uint32).astype(np.uint64)
    return {
        "packed_zn": torch.from_numpy(packed),
        "pair_lut": torch.from_numpy(pair_lut),
        "row_scale_bits": torch.from_numpy(row_bits),
        "perm": payload["perm"].clone(),
        "weight_scale": payload["weight_scale"].clone(),
    }


def decode_gs32_codebook_reference(payload, spec):
    """CPU FP32 lookup before inverse permutation or either BDNorm scale.

    Mirrors the verified standalone codec's _decode_qwt, independently of the
    zN repacker. At most one matrix is decoded; no serving fallback is provided.
    """
    _validate_payload(payload, spec)
    indices = torch.from_numpy(_indices(payload, spec).astype(np.int64))
    books = payload["codebooks"].float()
    weight = torch.empty((spec.rows, spec.columns), dtype=torch.float32, device="cpu")
    for column_tile in range(spec.columns // GS32_TILE):
        tile_indices = indices[column_tile].reshape(spec.rows // GS32_TILE, -1)
        vectors = torch.gather(books[column_tile], 1, tile_indices.unsqueeze(-1).expand(-1, -1, 2))
        block = vectors.reshape(spec.rows // GS32_TILE, GS32_TILE, GS32_TILE).transpose(1, 2)
        weight[:, column_tile * GS32_TILE : (column_tile + 1) * GS32_TILE] = block.reshape(spec.rows, GS32_TILE)
    return weight


def decode_gs32_weight_reference(payload, spec):
    """Producer's FP32 inverse transforms, then exactly one BF16 conversion."""
    weight = decode_gs32_codebook_reference(payload, spec)
    weight = weight[:, torch.argsort(payload["perm"].long())]
    weight = weight * payload["weight_scale"].unsqueeze(0) + payload["weight_bias"].unsqueeze(0)
    weight = weight * payload["row_scale"].unsqueeze(1)
    return weight.to(torch.bfloat16)


def prepare_gs32_activation_reference(hidden, payload, spec):
    """Unquantized FP32 input transform for a future fused consumer oracle.

    This is an algebraic scaled-codebook path, NOT the producer BF16-weight
    oracle. MXFP8 quantization, its rounding, and output scaling are separate.
    """
    _validate_payload(payload, spec)
    if (
        not isinstance(hidden, torch.Tensor)
        or hidden.device.type != "cpu"
        or not hidden.is_floating_point()
        or hidden.ndim != 2
        or hidden.shape[1] != spec.columns
        or not bool(torch.isfinite(hidden.float()).all())
    ):
        raise ValueError("Reference hidden must be finite CPU [tokens, K] floating-point data.")
    return (hidden.float() * payload["weight_scale"]).index_select(1, payload["perm"].long())


def dequantize_mxfp8_k32(data, scales):
    """CPU-only independent projection oracle input decoding."""
    if data.device.type != "cpu" or scales.device.type != "cpu":
        raise ValueError("MXFP8 reference decoding is CPU-only.")
    if data.dtype != torch.uint8 or scales.dtype != torch.uint8 or data.ndim != 2:
        raise ValueError("Expected uint8 MXFP8 data and scale bytes.")
    m, k = data.shape
    if m < 1 or k < MX_GROUP or k % MX_GROUP or scales.shape != (m, k // MX_GROUP):
        raise ValueError("Invalid MXFP8 data/scale shape.")
    if bool((scales == 255).any()) or bool((scales == 0).any()):
        raise ValueError("This quantizer emits E8M0 codes 1..254 only.")
    scale_values = (scales.to(torch.int32) << 23).contiguous().view(torch.float32)
    result = data.contiguous().view(torch.float8_e4m3fn).float().reshape(m, k // MX_GROUP, MX_GROUP)
    result = (result * scale_values.unsqueeze(-1)).reshape(m, k)
    require_finite(result, "Decoded MXFP8 activation")
    return result


def staged_projection_reference(hidden, payload, spec):
    """CPU FP32 GEMM + row scale -> BF16, using quantized transformed input.

    Accumulator order may differ from hardware; this is not a bitwise GEMM
    oracle. It does not materialize or reuse BF16 reconstructed model weights.
    """
    transformed = prepare_gs32_activation_reference(hidden, payload, spec)
    data, scale = quantize_mxfp8_k32(transformed)
    activation = dequantize_mxfp8_k32(data, scale)
    codebook_weight = decode_gs32_codebook_reference(payload, spec)
    output = ((activation @ codebook_weight.T) * payload["row_scale"]).to(torch.bfloat16)
    require_finite(output, "Projection reference")
    return output


def staged_expert_reference(hidden, gate_payload, gate_spec, down_payload, down_spec, limit):
    """Compose the independent projection oracle with the staged activation."""
    gate_up = staged_projection_reference(hidden, gate_payload, gate_spec)
    middle = staged_swiglu(gate_up, limit)
    return staged_projection_reference(middle, down_payload, down_spec)
