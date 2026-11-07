# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load-time byte layouts for the mixed W2/W4 complete-FFN kernel.

The checkpoint keeps natural Gate-then-Up rows. Only the resident Gate weight
and table tiles are interleaved; W2 row scales retain their natural row order.
This module deliberately has no torch_npu, vLLM or native-library dependency.
"""

import math

import torch

N_TILE = 32
K_TILE = 16
SCALE_GROUP = 32
FP32_ONE_BITS = 0x3F800000


def interleave_gate_up_blocks(value):
    """Interleave the two halves of the first (N32 tile) axis byte-for-byte."""
    blocks = value.shape[0]
    if blocks == 0 or blocks % 2:
        raise ValueError("Complete FFN Gate rows require an even positive number of N32 tiles.")
    return value.reshape(2, blocks // 2, *value.shape[1:]).transpose(0, 1).contiguous().reshape_as(value)


def pack_mxfp4_zn(weight):
    """Convert row-K-packed E2M1 bytes to raw [N/32,K/16,16,16] zN.

    Source bytes pack adjacent K values; destination bytes pack adjacent N rows.
    Integer nibble operations preserve every code, including negative zero. The
    int16 intermediates avoid depending on unsigned-byte arithmetic on the NPU.
    """
    if weight.dtype != torch.uint8 or weight.ndim != 2:
        raise ValueError("Complete FFN MXFP4 weights must be uint8 [N,K/2].")
    rows, half_columns = weight.shape
    columns = half_columns * 2
    if rows < N_TILE or rows % N_TILE or columns < K_TILE or columns % K_TILE:
        raise ValueError("Complete FFN MXFP4 packing requires positive N32/K16 alignment.")
    tiles = weight.reshape(rows // N_TILE, N_TILE, columns // K_TILE, K_TILE // 2).to(torch.int16)
    even_rows, odd_rows = tiles[:, 0::2], tiles[:, 1::2]
    even_k = (even_rows & 15) | ((odd_rows & 15) << 4)
    odd_k = (even_rows >> 4) | (odd_rows & 240)
    return (
        torch.stack((even_k, odd_k), dim=-1)
        .permute(0, 2, 3, 4, 1)
        .contiguous()
        .reshape(rows // N_TILE, columns // K_TILE, K_TILE, N_TILE // 2)
        .to(torch.uint8)
    )


def validate_full_ffn_geometry(gate_spec, down_spec):
    """Reject unsupported shapes before allocating native weight storage."""
    if (
        gate_spec.columns % 64
        or gate_spec.rows % 128
        or down_spec.rows % N_TILE
        or gate_spec.columns <= 0
        or gate_spec.rows <= 0
        or down_spec.rows <= 0
        or down_spec.columns != gate_spec.rows // 2
    ):
        raise ValueError("Complete FFN requires K64, Gate N128, Down N32 and Down K == Gate N/2.")


def validate_full_ffn_spec(layer_spec):
    validate_full_ffn_geometry(layer_spec.specs["gate_up"], layer_spec.specs["down"])
    experts = tuple(layer_spec.expert_ids)
    if experts != tuple(sorted(set(experts))):
        raise ValueError("Complete FFN resident expert ids must be unique and sorted.")
    for expert in experts:
        pair = tuple(layer_spec.quant_types[kind][expert] for kind in ("gate_up", "down"))
        if pair not in (("vq2", "vq2"), ("vq2", "mxfp4"), ("mxfp4", "mxfp4")):
            raise ValueError(f"Complete FFN does not support expert {expert} precision pair {pair}.")


def build_full_ffn_bank(layer, layer_spec, *, swiglu_limit):
    """Pack both projections in one resident-expert order, without requantizing.

    Call only after source-v3 field validation. The returned bank owns newly
    allocated storage, allowing the caller to release every source parameter.
    CPU precision flags are consumed by the native plan during initialization;
    they are never read back from the device in a model forward.
    """
    if isinstance(swiglu_limit, bool) or not isinstance(swiglu_limit, (int, float)):
        raise ValueError("Complete FFN SwiGLU limit must be a finite non-negative number.")
    if not math.isfinite(swiglu_limit) or swiglu_limit < 0:
        raise ValueError("Complete FFN SwiGLU limit must be a finite non-negative number.")
    validate_full_ffn_spec(layer_spec)
    experts = tuple(layer_spec.expert_ids)

    device = layer.vq2a8_anchor.device
    bank = {"expert_ids": experts, "fields": {}, "weight_offsets": {}, "stage_mask": 0}
    for kind, prefix, native_name, stage_shift in (("gate_up", "w13", "gate", 0), ("down", "w2", "down", 2)):
        spec = layer_spec.specs[kind]
        rows, columns = spec.rows, spec.columns
        quant_types = layer_spec.quant_types[kind]
        typed_slots = {
            quant_type: {expert: slot for slot, expert in enumerate(e for e in experts if quant_types[e] == quant_type)}
            for quant_type in ("vq2", "mxfp4")
        }
        offsets = [0]
        for expert in experts:
            offsets.append(offsets[-1] + rows * columns // (2 if quant_types[expert] == "mxfp4" else 4))
        table_bytes = rows * columns // SCALE_GROUP
        weight = torch.empty(offsets[-1], dtype=torch.uint8, device=device)
        table = torch.empty(len(experts) * table_bytes, dtype=torch.uint8, device=device)
        row_scale = torch.full((len(experts), rows), FP32_ONE_BITS, dtype=torch.int64, device=device)
        flags = tuple(quant_types[expert] == "mxfp4" for expert in experts)
        for slot, expert in enumerate(experts):
            quant_type = quant_types[expert]
            source_slot = typed_slots[quant_type][expert]
            if quant_type == "vq2":
                packed = getattr(layer, f"{prefix}_packed_zn")[source_slot]
                source_table = getattr(layer, f"{prefix}_pair_lut")[source_slot]
                if kind == "gate_up":
                    packed = interleave_gate_up_blocks(packed)
                    source_table = interleave_gate_up_blocks(source_table)
                row_scale[slot].copy_(getattr(layer, f"{prefix}_row_scale_bits")[source_slot])
            else:
                packed = pack_mxfp4_zn(getattr(layer, f"{prefix}_mxfp4_weight")[source_slot])
                source_table = getattr(layer, f"{prefix}_mxfp4_scale")[source_slot]
                if kind == "gate_up":
                    packed = interleave_gate_up_blocks(packed)
                    source_table = interleave_gate_up_blocks(source_table.reshape(rows // N_TILE, N_TILE, -1))
            weight[offsets[slot] : offsets[slot + 1]].copy_(packed.reshape(-1))
            table[slot * table_bytes : (slot + 1) * table_bytes].copy_(source_table.reshape(-1))
        bank["fields"].update(
            {f"{native_name}_weight": weight, f"{native_name}_table": table, f"{native_name}_row_scale": row_scale}
        )
        bank[f"{native_name}_w4"] = torch.tensor(flags, dtype=torch.bool, device="cpu")
        bank["weight_offsets"][kind] = tuple(offsets)
        if any(not flag for flag in flags):
            bank["stage_mask"] |= 1 << stage_shift
        if any(flags):
            bank["stage_mask"] |= 1 << (stage_shift + 1)
    bank["swiglu_limit"] = float(swiglu_limit)
    return bank
