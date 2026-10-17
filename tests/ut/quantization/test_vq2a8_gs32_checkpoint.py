# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source-v3 header inventory; payload allocation and NPU are not needed."""
# ruff: noqa: F401, F811  # Imported pytest fixtures are injected by name.

import json
import struct
from types import SimpleNamespace as NS

import pytest
import torch

from tests.ut.quantization.test_vq2a8_gs32_fixtures import modules, source_field, source_layer_spec
from vllm_ascend.quantization.methods.w2a8.vq2a8_gs32_format import (
    GS32_FORMAT,
    GS32_VERSION,
    read_gs32_header,
    source_projection_fields,
)


def source_config():
    return {"quant_method": "vq2a8", "format": GS32_FORMAT, "format_version": GS32_VERSION}


def write_checkpoint(root, tensors, *, metadata=None):
    header = {
        "__metadata__": metadata or {"format": GS32_FORMAT, "format_version": "3", "row_order": "natural_gate_then_up"}
    }
    payload = bytearray()
    for name, (dtype, tensor) in tensors.items():
        raw = tensor.contiguous().view(torch.uint8).numpy().tobytes()
        header[name] = {
            "dtype": dtype,
            "shape": list(tensor.shape),
            "data_offsets": [len(payload), len(payload) + len(raw)],
        }
        payload.extend(raw)
    data = json.dumps(header).encode()
    data += b" " * (-len(data) % 8)
    (root / "weights.safetensors").write_bytes(struct.pack("<Q", len(data)) + data + payload)
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "weights.safetensors" for name in tensors}})
    )
    (root / "config.json").write_text(json.dumps({"quantization_config": source_config()}))


@pytest.fixture
def source_checkpoint(modules, tmp_path):
    spec = source_layer_spec(modules)
    tensors = {}
    for expert in spec.expert_ids:
        for projection, kind in (("w13", "gate_up"), ("w2", "down")):
            quant_type = spec.quant_types[kind][expert]
            for field, (dtype, _) in source_projection_fields(spec.specs[kind], quant_type).items():
                name = f"layers.0.ffn.experts.{expert}.{projection}.{field}"
                tensors[name] = dtype, source_field(spec.specs[kind], quant_type, field, expert)
    tensors["model.layers.0.self_attn.q_a_proj.weight"] = "F8_E4M3", torch.ones(128, 128).to(torch.float8_e4m3fn)
    tensors["model.layers.0.self_attn.q_a_proj.scale"] = (
        "F8_E8M0",
        torch.full((1, 1), 110, dtype=torch.uint8).view(torch.float8_e8m0fnu),
    )
    tensors["model.layers.0.attn_hc.scale"] = "F32", torch.ones(3)
    tensors["head.weight"] = "BF16", torch.ones(16, 128, dtype=torch.bfloat16)
    write_checkpoint(tmp_path, tensors)
    config = NS(
        model=str(tmp_path),
        hf_config=NS(
            num_hidden_layers=1, n_routed_experts=4, num_hash_layers=1, hidden_size=128, moe_intermediate_size=64
        ),
    )
    return NS(root=tmp_path, config=config, tensors=tensors)


def test_source_v3_inventory_has_independent_projection_types(modules, source_checkpoint):
    quant = modules.config.AscendVQ2A8Config(source_config())
    with pytest.raises(ValueError, match="unbound"):
        quant.layer_spec(0)
    quant.bind_checkpoint(source_checkpoint.config)
    layer = quant.layer_spec(0)
    assert [layer.quant_types["gate_up"][e] for e in range(4)] == ["vq2", "vq2", "mxfp4", "mxfp4"]
    assert [layer.quant_types["down"][e] for e in range(4)] == ["vq2", "mxfp4", "vq2", "mxfp4"]
    assert quant.root_linear_kind("model.layers.0.self_attn.wq_a") == "ds_fp8_block128"
    assert quant.root_linear_kind("lm_head") == "unquantized"
    assert quant.normalize_root_name("model.layers.0.attn_hc.scale") == "model.layers.0.hc_attn_scale"
    quant.bind_checkpoint(source_checkpoint.config)
    assert len(quant.root_tensors) == 4


@pytest.mark.parametrize("hash_layers", [0, 1])
def test_source_v3_sparse_expert_inventory_only_allowed_for_hash_layers(modules, source_checkpoint, hash_layers):
    tensors = {name: value for name, value in source_checkpoint.tensors.items() if ".experts.3." not in name}
    write_checkpoint(source_checkpoint.root, tensors)
    source_checkpoint.config.hf_config.num_hash_layers = hash_layers
    quant = modules.config.AscendVQ2A8Config(source_config())
    if hash_layers:
        quant.bind_checkpoint(source_checkpoint.config)
        assert quant.layer_spec(0).expert_ids == (0, 1, 2)
    else:
        with pytest.raises(ValueError, match="logical expert inventory"):
            quant.bind_checkpoint(source_checkpoint.config)


@pytest.mark.parametrize(
    "fault",
    ["missing", "mixed", "old_name", "dtype", "shape", "remote_layer", "unpaired_scale", "scale_shape", "old_order"],
)
def test_source_v3_header_inventory_rejects_invalid_exports(modules, source_checkpoint, fault):
    tensors = dict(source_checkpoint.tensors)
    key = "layers.0.ffn.experts.0.w13.perm"
    metadata = None
    if fault == "missing":
        del tensors[key]
    elif fault == "mixed":
        tensors["layers.0.ffn.experts.0.w13.weight"] = "U8", torch.ones(128, 64, dtype=torch.uint8)
    elif fault == "old_name":
        tensors["model.layers.0.mlp.experts.w13_perm"] = tensors.pop(key)
    elif fault == "dtype":
        tensors[key] = "I64", tensors[key][1].long()
    elif fault == "shape":
        tensors[key] = "I32", tensors[key][1][:-1]
    elif fault == "remote_layer":
        tensors[key.replace("layers.0", "layers.1")] = tensors.pop(key)
    elif fault == "unpaired_scale":
        del tensors["model.layers.0.self_attn.q_a_proj.scale"]
    elif fault == "scale_shape":
        tensors["model.layers.0.self_attn.q_a_proj.scale"] = (
            "F8_E8M0",
            torch.full((2, 1), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu),
        )
    else:
        metadata = {"row_order": "interleaved_gate_up_32"}
    write_checkpoint(source_checkpoint.root, tensors, metadata=metadata)
    quant = modules.config.AscendVQ2A8Config(source_config())
    with pytest.raises(ValueError):
        quant.bind_checkpoint(source_checkpoint.config)
    assert not quant.layers and quant.checkpoint_root is None


@pytest.mark.parametrize("fault", ["duplicate_json", "wrong_index", "truncated", "trailing", "traversal"])
def test_source_v3_audits_index_and_file_extents(modules, source_checkpoint, fault):
    root = source_checkpoint.root
    index_path = root / "model.safetensors.index.json"
    if fault == "duplicate_json":
        index_path.write_text('{"weight_map":{},"weight_map":{}}')
    elif fault == "wrong_index":
        index = json.loads(index_path.read_text())
        del index["weight_map"][next(iter(index["weight_map"]))]
        index_path.write_text(json.dumps(index))
    elif fault == "traversal":
        index_path.write_text(json.dumps({"weight_map": {"head.weight": "../outside.safetensors"}}))
    else:
        path = root / "weights.safetensors"
        data = path.read_bytes()
        path.write_bytes(data[:-1] if fault == "truncated" else data + b"x")
    quant = modules.config.AscendVQ2A8Config(source_config())
    with pytest.raises(ValueError):
        quant.bind_checkpoint(source_checkpoint.config)


@pytest.mark.parametrize("size", [0, 1, 2**64 - 1])
def test_source_v3_bounded_header_reads(tmp_path, size):
    path = tmp_path / "invalid.safetensors"
    path.write_bytes(struct.pack("<Q", size) + b"{}")
    with pytest.raises(ValueError):
        read_gs32_header(path)
