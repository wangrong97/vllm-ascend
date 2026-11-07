# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent test-only quantization, activation and compressed-weight decoding.

No native GS32 runtime, graph scheduling or serving fallback is used here.
"""

import math

import torch

MX_GROUP = 32
FP8_MAX = 448.0
GS32_TILE = 32
GS32_K_INNER = 16
GS32_PACKED_ROW_BYTES = 8
GS32_LUT_BYTES = 32
MX_SCALE_PAIR_K = 64


def require_finite(tensor, name):
    if not bool(torch.isfinite(tensor.float()).all()):
        raise ValueError(f"{name} contains NaN or Inf.")


def quantize_mxfp8_k32(data):
    """BF16 RNE -> K32 E8M0 scale -> BF16 scaled RNE -> E4M3FN RNE.

    Matches the imported operator's quantize_h convention, including zero
    groups (scale 127), exponent boundary 1.75, and saturation at +/-448.
    Returns raw uint8 data and E8M0 bytes, not an opaque platform NZ tensor.
    """
    if not isinstance(data, torch.Tensor) or not data.is_floating_point() or data.ndim != 2:
        raise ValueError("MXFP8 input must be a floating-point [M,K] tensor.")
    if data.shape[0] < 1 or data.shape[1] < MX_GROUP or data.shape[1] % MX_GROUP:
        raise ValueError("MXFP8 input requires M>0 and positive K divisible by 32.")
    require_finite(data, "MXFP8 input")
    rounded = data.to(torch.bfloat16).float()
    require_finite(rounded, "BF16-rounded MXFP8 input")
    grouped = rounded.reshape(data.shape[0], data.shape[1] // MX_GROUP, MX_GROUP)
    amax = grouped.abs().amax(dim=-1)
    bits = amax.contiguous().view(torch.int32)
    exponent = (bits >> 23) & 255
    codes = exponent - 8 + ((bits & 0x7FFFFF) > 0x600000).to(torch.int32)
    codes = torch.where(amax == 0, 127, codes.clamp_min(1))
    inverse = ((254 - codes) << 23).contiguous().view(torch.float32)
    scaled = (grouped * inverse.unsqueeze(-1)).to(torch.bfloat16).float().clamp(-FP8_MAX, FP8_MAX)
    quantized = scaled.reshape(data.shape).to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
    return quantized, codes.to(torch.uint8).contiguous()


def _validate_swiglu(gate_up, limit):
    if not math.isfinite(limit) or limit < 0:
        raise ValueError("SwiGLU limit must be finite and non-negative.")
    if gate_up.dtype != torch.bfloat16 or gate_up.ndim != 2 or gate_up.shape[1] % 2:
        raise ValueError("GateUp must be a BF16 [M,2H] tensor in natural row order.")


def _swiglu_arithmetic(gate_up, limit):
    gate, up = gate_up.float().chunk(2, dim=-1)
    if limit > 0:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    return (torch.nn.functional.silu(gate) * up).to(torch.bfloat16).contiguous()


def staged_swiglu_unchecked(gate_up, limit):
    """Same arithmetic; callers must check both raw input and returned output."""
    _validate_swiglu(gate_up, limit)
    return _swiglu_arithmetic(gate_up, limit)


def staged_swiglu(gate_up, limit):
    """Logical Gate-all/Up-all; FP32 clamp/SiLU/product -> one BF16 round."""
    _validate_swiglu(gate_up, limit)
    require_finite(gate_up, "GateUp output")
    hidden = _swiglu_arithmetic(gate_up, limit)
    require_finite(hidden, "SwiGLU output")
    return hidden


def decode_gs32_fp8(packed_zn, pair_lut, *, start=0, stop=None):
    """Decode natural-order source-v3 bytes into contiguous E4M3 ``[K, N]``.

    ``start``/``stop`` select output columns (the source weight's output rows),
    not the reduction axis. The returned FP8 bytes are exact codebook lookups;
    no weight rescaling, quantization or floating-point arithmetic is involved.
    The source-v3 GateUp bank already uses natural Gate-all/Up-all row order.
    """
    if packed_zn.dtype != torch.uint8 or packed_zn.ndim != 4:
        raise ValueError("Reference packed_zn must be uint8 [N/32,K/16,16,8].")
    ntiles, ktiles, inner_k, row_bytes = packed_zn.shape
    if (
        ntiles < 1
        or ktiles < MX_SCALE_PAIR_K // GS32_K_INNER
        or ktiles % (MX_SCALE_PAIR_K // GS32_K_INNER)
        or (inner_k, row_bytes) != (GS32_K_INNER, GS32_PACKED_ROW_BYTES)
    ):
        raise ValueError("Reference packed_zn requires positive N32/K64 geometry.")
    rows, columns = ntiles * GS32_TILE, ktiles * GS32_K_INNER
    if (
        pair_lut.dtype != torch.uint8
        or tuple(pair_lut.shape) != (ntiles, columns // GS32_TILE, GS32_LUT_BYTES)
        or pair_lut.device != packed_zn.device
    ):
        raise ValueError("Reference pair_lut must be uint8 [N/32,K/32,32] on the weight device.")
    stop = rows if stop is None else stop
    if (
        type(start) is not int
        or type(stop) is not int
        or not 0 <= start < stop <= rows
        or start % GS32_TILE
        or stop % GS32_TILE
    ):
        raise ValueError("Reference decode output-column range must be nonempty and N32 aligned.")
    packed = packed_zn[start // GS32_TILE : stop // GS32_TILE]
    lut = pair_lut[start // GS32_TILE : stop // GS32_TILE]
    block_tiles = (stop - start) // GS32_TILE
    # Each packed byte stores indices for two adjacent output-row pairs.
    # Preserve low/high nibble order and the two distinct values per VQ pair.
    indices = torch.stack((packed & 15, packed >> 4), dim=-1)
    indices = indices.reshape(block_tiles, columns // GS32_TILE, GS32_TILE, GS32_TILE // 2)
    offsets = indices.to(torch.int64).unsqueeze(-1) * 2
    offsets = offsets + torch.arange(2, dtype=torch.int64, device=packed.device)
    offsets = offsets.flatten(-2)
    tables = lut.unsqueeze(2).expand(-1, -1, GS32_TILE, -1)
    values = torch.gather(tables, -1, offsets)
    # [N/32,K/32,K32,N32] -> [K,N], retaining raw FP8 encodings including -0.
    decoded = values.permute(1, 2, 0, 3).reshape(columns, stop - start).contiguous()
    return decoded.view(torch.float8_e4m3fn)


def decode_mxfp4_fp8(weight):
    """Decode E2M1 nibbles exactly into E4M3, preserving signed zero."""
    codes = torch.stack((weight & 15, weight >> 4), dim=-1).reshape(weight.shape[0], -1).long()
    magnitudes = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=weight.device)
    values = magnitudes[codes & 7]
    return torch.where(codes >= 8, -values, values).to(torch.float8_e4m3fn)


class OfficialFFNOracle:
    """Selected-expert oracle: CPU quantization/math plus official NPU GMM.

    It reads natural checkpoint layouts, never native full-FFN packing or the
    retired staged serving implementation. Decode storage is bounded by one
    expert/output-column block. Host transfers are intentional test work.
    """

    def __init__(self, layer, spec, device, *, column_chunk_size=256):
        import torch_npu

        if type(column_chunk_size) is not int or column_chunk_size < GS32_TILE or column_chunk_size % GS32_TILE:
            raise ValueError("Oracle column chunks must be positive multiples of 32.")
        self.gmm = torch_npu.npu_grouped_matmul
        self.scale_dtype = torch_npu.float8_e8m0fnu
        self.fields = {
            name: value.detach().cpu().clone()
            for name, value in vars(layer).items()
            if name.startswith(("w13_", "w2_")) and isinstance(value, torch.Tensor)
        }
        self.spec, self.device = spec, device
        self.column_chunk_size = column_chunk_size
        self.num_experts = layer.vq2a8_num_experts

    def project(self, kind, expert, data, scales):
        spec = self.spec.specs[kind]
        prefix = "w13" if kind == "gate_up" else "w2"
        quant_type = self.spec.quant_types[kind][expert]
        same_type = sorted(e for e in self.spec.expert_ids if self.spec.quant_types[kind][e] == quant_type)
        slot = same_type.index(expert)
        rows, k = data.shape
        x = data.to(self.device).view(torch.float8_e4m3fn)
        x_scale = scales.to(self.device).reshape(rows, k // 64, 2)
        counts = torch.tensor([rows], dtype=torch.int64, device=self.device)
        result = torch.empty((rows, spec.rows), dtype=torch.bfloat16)
        for first in range(0, spec.rows, self.column_chunk_size):
            last = min(first + self.column_chunk_size, spec.rows)
            if quant_type == "vq2":
                decoded = decode_gs32_fp8(
                    self.fields[f"{prefix}_packed_zn"][slot],
                    self.fields[f"{prefix}_pair_lut"][slot],
                    start=first,
                    stop=last,
                )
                weight_scale = torch.full((1, k // 64, last - first, 2), 127, dtype=torch.uint8)
                row_scale = (
                    self.fields[f"{prefix}_row_scale_bits"][slot, first:last].to(torch.int32).view(torch.float32)
                )
            else:
                decoded = decode_mxfp4_fp8(self.fields[f"{prefix}_mxfp4_weight"][slot, first:last]).T.contiguous()
                weight_scale = self.fields[f"{prefix}_mxfp4_scale"][slot, first:last]
                weight_scale = weight_scale.reshape(last - first, k // 64, 2).permute(1, 0, 2).unsqueeze(0).contiguous()
                row_scale = torch.ones(last - first)
            projected = self.gmm(
                x=[x],
                weight=[decoded.to(self.device).unsqueeze(0)],
                scale=[weight_scale.to(self.device)],
                per_token_scale=[x_scale],
                group_list=counts,
                group_list_type=1,
                group_type=0,
                split_item=2,
                output_dtype=torch.float32,
                x_dtype=torch.float8_e4m3fn,
                weight_dtype=torch.float8_e4m3fn,
                scale_dtype=self.scale_dtype,
                per_token_scale_dtype=self.scale_dtype,
                bias=None,
            )[0]
            if projected.dtype != torch.float32 or tuple(projected.shape) != (rows, last - first):
                raise RuntimeError("Official GMM must return FP32 [M,N].")
            projected = projected.cpu()
            result[:, first:last] = (projected * row_scale).bfloat16()
        require_finite(result, "Official GMM projection")
        return result

    def __call__(self, hidden, groups, group_type, limit):
        hidden = hidden.detach().cpu()
        values = groups.detach().cpu().tolist()
        counts = values if group_type == 1 else [end - start for start, end in zip([0, *values[:-1]], values)]
        if len(counts) != self.num_experts or any(c < 0 for c in counts) or sum(counts) > hidden.shape[0]:
            raise ValueError("Invalid oracle expert counts.")
        if any(counts[e] for e in range(self.num_experts) if e not in self.spec.expert_ids):
            raise ValueError("Oracle routes include a nonresident expert.")
        intermediate = self.spec.specs["down"].columns
        output = torch.zeros((hidden.shape[0], self.spec.specs["down"].rows), dtype=torch.bfloat16)
        middle = torch.zeros((hidden.shape[0], intermediate), dtype=torch.uint8)
        middle_scales = torch.full((hidden.shape[0], intermediate // 32), 127, dtype=torch.uint8)
        first = 0
        for expert, count in enumerate(counts):
            last = first + count
            if count:
                data, scales = quantize_mxfp8_k32(hidden[first:last])
                gate_up = self.project("gate_up", expert, data, scales)
                data, scales = quantize_mxfp8_k32(staged_swiglu(gate_up, limit))
                middle[first:last], middle_scales[first:last] = data, scales
                output[first:last] = self.project("down", expert, data, scales)
            first = last
        return output, {"data": middle, "scales": middle_scales}
