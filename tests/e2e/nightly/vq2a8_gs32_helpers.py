# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read selected source-v3 expert tensors without creating an engine or model."""

from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

SELECTED_EXPERTS = 3


def native_paths(options):
    if not all(options.values()):
        raise ValueError("Model and both GS32 native artifact paths are required.")
    paths = {key: str(Path(value).resolve(strict=True)) for key, value in options.items()}
    if not all(Path(paths[key]).is_file() for key in ("gs32_library", "gs32_binary")):
        raise ValueError("Both GS32 native artifact paths must be files.")
    return paths


def checkpoint_inventory(model):
    from vllm_ascend.quantization.methods.w2a8.vq2a8_gs32_format import (
        GS32_FORMAT,
        GS32_VERSION,
        MXFP4_SOURCE_FIELDS,
        VQ2_SOURCE_FIELDS,
        parse_vq2a8_expert_name,
        read_gs32_json,
    )

    root = Path(model).resolve(strict=True)
    config = read_gs32_json(root / "config.json")
    if config.get("quantization_config") != {
        "quant_method": "vq2a8",
        "format": GS32_FORMAT,
        "format_version": GS32_VERSION,
    }:
        raise ValueError("This probe requires a source-v3 GS32 checkpoint.")
    index = read_gs32_json(root / "model.safetensors.index.json")["weight_map"]
    fields = defaultdict(set)
    for name in index:
        parsed = parse_vq2a8_expert_name(name)
        if parsed is not None:
            layer, expert, projection, field = parsed
            fields[layer, expert, projection].add(field)
    entries = {}
    for (layer, expert, projection), present in fields.items():
        if present == set(VQ2_SOURCE_FIELDS):
            quant_type = "vq2"
        elif present == set(MXFP4_SOURCE_FIELDS):
            quant_type = "mxfp4"
        else:
            raise ValueError(f"Incomplete source-v3 expert: {layer}/{expert}/{projection}")
        entry = entries.setdefault(layer, {"gate_up": {}, "down": {}})
        entry["gate_up" if projection == "w13" else "down"][expert] = quant_type
    for entry in entries.values():
        if entry["gate_up"].keys() != entry["down"].keys():
            raise ValueError("Source-v3 expert projections have different inventories.")
    return config, index, entries


def load_selected_layer(model, *, layer_index=0, expert_count=SELECTED_EXPERTS, expert_positions=None):
    # Only selected experts' payloads are read; no complete model is allocated.
    import torch
    from safetensors import safe_open

    from vllm_ascend.quantization.methods.w2a8.vq2a8_gs32_format import (
        GS32ProjectionSpec,
        convert_expert_tensor,
        source_projection_fields,
    )

    root = Path(model).resolve(strict=True)
    config, index, entries = checkpoint_inventory(root)
    entry = entries[layer_index]
    inventory = tuple(sorted(entry["gate_up"]))
    positions = tuple(range(expert_count)) if expert_positions is None else tuple(expert_positions)
    if (
        not positions
        or any(type(p) is not int or not 0 <= p < len(inventory) for p in positions)
        or len(set(positions)) != len(positions)
    ):
        raise ValueError("The selected layer needs enough distinct experts.")
    selected = tuple(inventory[p] for p in positions)
    specs = {
        "gate_up": GS32ProjectionSpec(2 * config["moe_intermediate_size"], config["hidden_size"]),
        "down": GS32ProjectionSpec(config["hidden_size"], config["moe_intermediate_size"]),
    }
    types = {kind: {expert: entry[kind][expert] for expert in selected} for kind in specs}
    spec = SimpleNamespace(layer_index=layer_index, expert_ids=selected, specs=specs, quant_types=types)
    layer = SimpleNamespace(
        vq2a8_num_experts=config["n_routed_experts"],
        vq2a8_source=str(root),
        vq2a8_positions=positions,
        vq2a8_anchor=torch.empty(0, dtype=torch.uint8),
    )
    names = {
        "packed_indices": "packed_zn",
        "codebooks": "pair_lut",
        "out_dim_scale": "row_scale_bits",
        "perm": "perm",
        "weight": "mxfp4_weight",
        "scale": "mxfp4_scale",
    }
    for kind, projection in (("gate_up", "w13"), ("down", "w2")):
        for quant_type in ("vq2", "mxfp4"):
            experts = sorted(e for e in selected if types[kind][e] == quant_type)
            for field in source_projection_fields(specs[kind], quant_type):
                values = []
                for expert in experts:
                    name = f"layers.{layer_index}.ffn.experts.{expert}.{projection}.{field}"
                    shard = (root / index[name]).resolve(strict=True)
                    if not shard.is_relative_to(root):
                        raise ValueError(f"Checkpoint shard escapes model root: {shard}")
                    with safe_open(shard, framework="pt", device="cpu") as handle:
                        value = handle.get_tensor(name)
                    values.append(convert_expert_tensor(field, value, specs[kind], quant_type))
                if values:
                    setattr(layer, f"{projection}_{names[field]}", torch.stack(values))
    return layer, spec, config
