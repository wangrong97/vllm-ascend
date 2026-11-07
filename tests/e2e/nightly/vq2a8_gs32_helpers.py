# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inspect GS32 checkpoints and runner graph evidence without engine initialization."""

from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

SELECTED_EXPERTS = 3
LEGACY_PREFILL_TOKENS = 32
MODEL_RUNNER_MODULES = (
    ("vllm_ascend.worker.model_runner_v1", 1),
    ("vllm_ascend.worker.v2.model_runner", 2),
)


class _ReplayCounter:
    """Count successful test submissions without replacing graph objects."""

    def __init__(self, replay):
        self.original_replay = replay
        self.num_replays = 0

    def __call__(self, *args, **kwargs):
        result = self.original_replay(*args, **kwargs)
        self.num_replays += 1
        return result


def runner_graph_state(runner):
    """Read the selected runner's graph owner and count actual decode replays."""
    runner_class = type(runner)
    runner_name = f"{runner_class.__module__}.{runner_class.__name__}"
    versions = {
        version
        for module, version in MODEL_RUNNER_MODULES
        if any(cls.__module__ == module for cls in runner_class.__mro__)
    }
    assert len(versions) == 1, f"Unsupported model runner: {runner_name}."
    version = versions.pop()
    state = {name: {"captures": 0, "replays": 0} for name in ("prefill", "decode", "other")}
    compilation = getattr(runner, "compilation_config", None)
    decode_sizes = set(getattr(compilation, "cudagraph_capture_sizes", None) or (1,))
    decode_buckets = {}
    if version == 2:
        # V2 stores graphs on its manager; the graph object and concrete type
        # must survive instrumentation because vLLM resolves its update tasks.
        for descriptor, graph in runner.cudagraph_manager.graphs.items():
            assert callable(getattr(graph, "replay", None)), "Unsupported V2 graph entry: cannot verify replay."
            if not isinstance(graph.replay, _ReplayCounter):
                graph.replay = _ReplayCounter(graph.replay)
            tokens = descriptor.num_tokens
            phase = "decode" if tokens in decode_sizes else "other"
            state[phase]["captures"] += 1
            state[phase]["replays"] += graph.replay.num_replays
            if phase == "decode":
                bucket = decode_buckets.setdefault(str(tokens), {"captures": 0, "replays": 0})
                bucket["captures"] += 1
                bucket["replays"] += graph.replay.num_replays
    else:
        for entry in getattr(runner.model, "concrete_aclgraph_entries", {}).values():
            if entry.aclgraph is None:
                continue
            tokens = entry.batch_descriptor.num_tokens
            phase = "decode" if tokens in decode_sizes else "prefill" if tokens == LEGACY_PREFILL_TOKENS else "other"
            state[phase]["captures"] += 1
            state[phase]["replays"] += entry.num_replays
            if phase == "decode":
                bucket = decode_buckets.setdefault(str(tokens), {"captures": 0, "replays": 0})
                bucket["captures"] += 1
                bucket["replays"] += entry.num_replays
    state.update(model_runner=runner_name, model_runner_version=version, decode_buckets=decode_buckets)
    return state


def native_paths(options):
    """Validate named checkpoint/artifact paths before any model child starts."""
    flags = {
        "model": "--vq2a8-model",
        "draft_model": "--vq2a8-dspark-model",
        "gs32_library": "--vq2a8-gs32-library",
        "gs32_binary": "--vq2a8-gs32-binary",
    }
    required = ("model", "gs32_library", "gs32_binary")
    missing = [flags[name] for name in required if name not in options]
    if missing:
        raise ValueError(f"Required GS32 paths were not provided: {', '.join(missing)}.")
    paths = {}
    for name, value in options.items():
        flag = flags.get(name, name)
        if value is None or not str(value).strip():
            raise ValueError(f"{flag} requires a nonempty path.")
        try:
            path = Path(value).resolve(strict=True)
        except OSError as error:
            raise ValueError(f"{flag}: cannot resolve {str(value)!r}: {error}") from error
        directory = name in ("model", "draft_model")
        if not (path.is_dir() if directory else path.is_file()):
            kind = "directory" if directory else "regular file"
            raise ValueError(f"{flag}: expected a {kind}, got {path}.")
        paths[name] = str(path)
    return paths


def checkpoint_inventory(model):
    from vllm_ascend.quantization.methods.w2a8.vq2a8_gs32_format import (
        MXFP4_SOURCE_FIELDS,
        VQ2_SOURCE_FIELDS,
        parse_vq2a8_expert_name,
        read_gs32_json,
    )

    root = Path(model).resolve(strict=True)
    config = read_gs32_json(root / "config.json")
    from tools.vq2a8_source_contract import source_contract

    source_contract()["validate_source_config"](config.get("quantization_config"))
    index = read_gs32_json(root / "model.safetensors.index.json")["weight_map"]
    fields = defaultdict(set)
    for name in index:
        if name.startswith("mtp."):
            # The main-model probe does not load original DSpark draft experts.
            continue
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
    spec = SimpleNamespace(
        layer_index=layer_index,
        expert_ids=selected,
        specs=specs,
        quant_types=types,
    )
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
