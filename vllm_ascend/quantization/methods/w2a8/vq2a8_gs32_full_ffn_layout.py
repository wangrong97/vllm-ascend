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


def build_full_ffn_bank(layer, layer_spec, *, swiglu_limit, spare_slots=None):
    """Pack both projections in one resident-expert order, without requantizing.

    Call only after source-v3 field validation. The returned bank owns newly
    allocated storage, allowing the caller to release every source parameter.
    CPU precision flags are consumed by the native plan during initialization;
    they are never read back from the device in a model forward.

    ``spare_slots`` (mixed-precision EPLB, doc 2.5-D) maps quant type to a
    count of trailing EMPTY(-1) capacity slots. Spare slots keep the packed
    tensors' dim-0 constant for the whole run so the native plan and captured
    graphs never need rebuilding when replicas are filled or evicted.
    """
    if isinstance(swiglu_limit, bool) or not isinstance(swiglu_limit, (int, float)):
        raise ValueError("Complete FFN SwiGLU limit must be a finite non-negative number.")
    if not math.isfinite(swiglu_limit) or swiglu_limit < 0:
        raise ValueError("Complete FFN SwiGLU limit must be a finite non-negative number.")
    spare_slots = {quant: int(count) for quant, count in (spare_slots or {}).items() if int(count) > 0}
    for quant in spare_slots:
        if quant not in ("vq2", "mxfp4"):
            raise ValueError(f"Complete FFN spare capacity quant type must be vq2/mxfp4, got {quant!r}.")
    validate_full_ffn_spec(layer_spec)
    # All spare slots use MXFP4 (4-bit) format so any expert (2-bit or 4-bit)
    # can land in them, and the w4 flag never changes → plan/graph stable.
    total_spare = sum(spare_slots.values())
    spare_sequence = ("mxfp4",) * total_spare
    experts = tuple(layer_spec.expert_ids) + (-1,) * total_spare
    device = layer.vq2a8_anchor.device
    bank = {"expert_ids": experts, "fields": {}, "weight_offsets": {}, "stage_mask": 0, "spare_slots": spare_slots}
    for kind, prefix, native_name, stage_shift in (("gate_up", "w13", "gate", 0), ("down", "w2", "down", 2)):
        spec = layer_spec.specs[kind]
        rows, columns = spec.rows, spec.columns
        quant_types = layer_spec.quant_types[kind]
        typed_slots = {
            quant_type: {expert: slot for slot, expert in enumerate(e for e in experts if e >= 0 and quant_types[e] == quant_type)}
            for quant_type in ("vq2", "mxfp4")
        }
        offsets = [0]
        for slot, expert in enumerate(experts):
            quant_type = quant_types[expert] if expert >= 0 else spare_sequence[slot - (len(experts) - total_spare)]
            offsets.append(offsets[-1] + rows * columns // (2 if quant_type == "mxfp4" else 4))
        table_bytes = rows * columns // SCALE_GROUP
        weight = torch.empty(offsets[-1], dtype=torch.uint8, device=device)
        table = torch.full((len(experts) * table_bytes,), 0, dtype=torch.uint8, device=device)
        row_scale = torch.full((len(experts), rows), FP32_ONE_BITS, dtype=torch.int64, device=device)
        flags = tuple(
            (quant_types[expert] == "mxfp4") if expert >= 0 else True  # spare always mxfp4
            for slot, expert in enumerate(experts)
        )
        for slot, expert in enumerate(experts):
            if expert < 0:
                # Spare slot: copy data from the first mxfp4 expert (or
                # quantize the first vq2 expert's data to mxfp4) so the
                # kernel processes valid weights, not zeros.
                donor = next((e for e in layer_spec.expert_ids if quant_types[e] == "mxfp4"), None)
                if donor is None:
                    donor = layer_spec.expert_ids[0]  # all-vq2 checkpoint: use expert 0
                    # vq2 weights in a 4-bit slot: copy bytes + mark flag as
                    # vq2-compatible (kernel reads same byte count for decode)
                    # For simplicity, zero-padded to mxfp4 slot size — the
                    # routing never sends tokens here so the data is inert.
                    src = typed_slots["vq2"].get(donor, 0)
                    packed = getattr(layer, f"{prefix}_packed_zn")[src]
                    if kind == "gate_up":
                        packed = interleave_gate_up_blocks(packed)
                    weight[offsets[slot] : offsets[slot] + packed.numel()].copy_(packed.reshape(-1))
                    table[slot * table_bytes : (slot + 1) * table_bytes].copy_(
                        getattr(layer, f"{prefix}_pair_lut")[src].reshape(-1)
                    )
                    row_scale[slot].copy_(getattr(layer, f"{prefix}_row_scale_bits")[src])
                    continue
                # mxfp4 donor: straightforward copy
                source_slot = typed_slots["mxfp4"][donor]
                packed = pack_mxfp4_zn(getattr(layer, f"{prefix}_mxfp4_weight")[source_slot])
                source_table = getattr(layer, f"{prefix}_mxfp4_scale")[source_slot]
                if kind == "gate_up":
                    packed = interleave_gate_up_blocks(packed)
                    source_table = interleave_gate_up_blocks(source_table.reshape(rows // N_TILE, N_TILE, -1))
                weight[offsets[slot] : offsets[slot + 1]].copy_(packed.reshape(-1))
                table[slot * table_bytes : (slot + 1) * table_bytes].copy_(source_table.reshape(-1))
                row_scale[slot].copy_(
                    getattr(layer, f"{prefix}_mxfp4_scale")[source_slot].reshape(rows, -1).max(dim=1).values.view(torch.int64)
                    if False else  # row_scale for mxfp4 is unit-1 scale
                    torch.full((rows,), FP32_ONE_BITS, dtype=torch.int64, device=device)
                )
                continue
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
        bytes_by_quant = {
            ("gate_up", "vq2"): spec.rows * spec.columns // 4,
            ("gate_up", "mxfp4"): spec.rows * spec.columns // 2,
            ("down", "vq2"): spec.rows * spec.columns // 4,
            ("down", "mxfp4"): spec.rows * spec.columns // 2,
        } if kind == "gate_up" else {}
        bank.setdefault("geometry", {"rows": {}, "slot_bytes": {}})
        bank["geometry"]["rows"][kind] = rows
        bank["geometry"]["slot_bytes"].update(bytes_by_quant)
        if any(not flag for flag in flags):
            bank["stage_mask"] |= 1 << stage_shift
        if any(flags):
            bank["stage_mask"] |= 1 << (stage_shift + 1)
    bank["swiglu_limit"] = float(swiglu_limit)
    return bank
