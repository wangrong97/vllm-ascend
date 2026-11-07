# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source-v3 header inventory; payload allocation and NPU are not needed."""
# ruff: noqa: F401, F811  # Imported pytest fixtures are injected by name.

import json
import struct
from types import SimpleNamespace as NS

import pytest
import torch

from tests.ut.quantization.test_vq2a8_gs32_fixtures import modules, serving_config, source_field, source_layer_spec
from vllm_ascend.quantization.methods.w2a8.vq2a8_gs32_contract import FOLDED_HEAD_CONFIG, FOLDED_ROTATION_CONFIG
from vllm_ascend.quantization.methods.w2a8.vq2a8_gs32_format import (
    GS32_FORMAT,
    GS32_VERSION,
    normalize_vq2a8_root_name,
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
    assert [layer.quant_types["down"][e] for e in range(4)] == ["vq2", "mxfp4", "mxfp4", "mxfp4"]
    assert quant.root_linear_kind("model.layers.0.self_attn.wq_a") == "ds_fp8_block128"
    assert quant.root_linear_kind("lm_head") == "unquantized"
    assert quant.normalize_root_name("model.layers.0.attn_hc.scale") == "model.layers.0.hc_attn_scale"
    quant.bind_checkpoint(source_checkpoint.config)
    assert len(quant.root_tensors) == 4
    assert quant.mtp_tensors == frozenset()


@pytest.mark.parametrize("fault", [None, "perm", "mxfp4", "rotation", "runtime_config"])
def test_folded_source_v3_binds_natural_vq_weights_and_rejects_contradictions(modules, source_checkpoint, fault):
    tensors = {name: value for name, value in source_checkpoint.tensors.items() if ".experts." not in name}
    for expert in range(4):
        for projection in ("w13", "w2"):
            prefix = f"layers.0.ffn.experts.0.{projection}."
            for name, value in source_checkpoint.tensors.items():
                if name.startswith(prefix):
                    tensors[name.replace("experts.0.", f"experts.{expert}.")] = value
    # Auxiliary rotation is deliberately absent: inference must not read it.
    rotation = {**dict(FOLDED_ROTATION_CONFIG), "global_rotation": "optional/quarot.safetensors"}
    config = {**source_config(), **dict(FOLDED_HEAD_CONFIG)}
    if fault == "perm":
        name = "layers.0.ffn.experts.0.w13.perm"
        tensors[name] = "I32", torch.arange(128, dtype=torch.int32)
    if fault == "mxfp4":
        prefix = "layers.0.ffn.experts.1.w2."
        tensors = {name: value for name, value in tensors.items() if not name.startswith(prefix)}
        tensors.update({name: value for name, value in source_checkpoint.tensors.items() if name.startswith(prefix)})
    if fault == "rotation":
        rotation["inverse_rotation_before_final_norm"] = True
    write_checkpoint(source_checkpoint.root, tensors)
    (source_checkpoint.root / "config.json").write_text(json.dumps({"quantization_config": config}))
    (source_checkpoint.root / "rotation_config.json").write_text(json.dumps(rotation))
    quant = modules.config.AscendVQ2A8Config(source_config() if fault == "runtime_config" else config)
    if fault:
        with pytest.raises(
            ValueError, match="six VQ|perm tensors are unsupported|rotation_config|quantization_config differ"
        ):
            quant.bind_checkpoint(source_checkpoint.config)
        assert quant.checkpoint_root is None
    else:
        quant.bind_checkpoint(source_checkpoint.config)
        assert quant.folded_output_head
        assert all(kind == "vq2" for types in quant.layer_spec(0).quant_types.values() for kind in types.values())


def test_folded_runtime_rejects_speculation_before_entering_original_dspark(modules, tmp_path):
    config = serving_config()
    config.additional_config["vq2a8"] = {
        "gs32_execution": "fused",
        "gs32_library": str(tmp_path / "stage.so"),
        "gs32_binary": str(tmp_path / "kernel.o"),
    }
    config.model_config.hf_config.quantization_config = {**source_config(), **dict(FOLDED_HEAD_CONFIG)}
    config.speculative_config = NS(method="dspark")
    with pytest.raises(ValueError, match="main-model decoding only"):
        modules.config.validate_vq2a8_config(config)
    config.speculative_config = None
    assert modules.config.validate_vq2a8_config(config) == {
        **config.additional_config["vq2a8"],
        "gs32_backend": "native",
        "gs32_execution": "fused",
        "gs32_fusion": "full_ffn",
    }


@pytest.mark.parametrize("projection", ["w13", "w2"])
@pytest.mark.parametrize("identity", [False, True])
def test_source_v3_rejects_explicit_perms_even_identity(modules, source_checkpoint, projection, identity):
    tensors = dict(source_checkpoint.tensors)
    width = 128 if projection == "w13" else 64
    value = torch.arange(width, dtype=torch.int32)
    if not identity:
        value = value.roll(1)
    tensors[f"layers.0.ffn.experts.0.{projection}.perm"] = "I32", value
    write_checkpoint(source_checkpoint.root, tensors)
    quant = modules.config.AscendVQ2A8Config(source_config())
    with pytest.raises(ValueError, match="perm tensors are unsupported"):
        quant.bind_checkpoint(source_checkpoint.config)
    assert not quant.layers and quant.checkpoint_root is None


def test_source_v3_natural_columns_require_complete_hash_layer_inventory(modules, source_checkpoint):
    tensors = {name: value for name, value in source_checkpoint.tensors.items() if ".experts.3." not in name}
    write_checkpoint(source_checkpoint.root, tensors)
    quant = modules.config.AscendVQ2A8Config(source_config())
    with pytest.raises(ValueError, match="logical expert inventory"):
        quant.bind_checkpoint(source_checkpoint.config)
    assert not quant.layers and quant.checkpoint_root is None


def test_source_v3_preserves_original_mtp_inventory_separately(modules, source_checkpoint):
    tensors = dict(source_checkpoint.tensors)
    mtp = {
        "mtp.0.ffn.experts.0.w1.weight": ("I8", torch.zeros(32, 16, dtype=torch.int8)),
        "mtp.0.ffn.experts.0.w1.scale": (
            "F8_E8M0",
            torch.full((32, 1), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu),
        ),
        "mtp.1.main_norm.weight": ("BF16", torch.ones(128, dtype=torch.bfloat16)),
        "mtp.2.confidence_head.weight": ("BF16", torch.ones(1, 128, dtype=torch.bfloat16)),
    }
    tensors.update(mtp)
    write_checkpoint(source_checkpoint.root, tensors)
    source_checkpoint.config.hf_config.num_nextn_predict_layers = 1
    quant = modules.config.AscendVQ2A8Config(source_config())
    quant.bind_checkpoint(source_checkpoint.config)
    assert quant.mtp_tensors == frozenset(mtp)
    assert quant.mtp_tensors <= quant.weight_map.keys()
    assert not quant.mtp_tensors & quant.root_tensors.keys()
    assert len(quant.root_tensors) == 4
    assert quant.layer_spec(0).expert_ids == (0, 1, 2, 3)


@pytest.mark.parametrize(
    "name", ["mtp.01.norm.weight", "mtp.-1.norm.weight", "mtp.x.norm.weight", "mtp.0.", "mtp.0..weight"]
)
def test_source_v3_rejects_malformed_mtp_names(modules, source_checkpoint, name):
    tensors = {**source_checkpoint.tensors, name: ("BF16", torch.ones(128, dtype=torch.bfloat16))}
    write_checkpoint(source_checkpoint.root, tensors)
    quant = modules.config.AscendVQ2A8Config(source_config())
    with pytest.raises(ValueError, match="Malformed source-v3 MTP tensor name"):
        quant.bind_checkpoint(source_checkpoint.config)
    assert quant.mtp_tensors == frozenset()
    assert not quant.layers and quant.checkpoint_root is None


def test_source_v3_ffn_root_aliases_bind_without_renaming_payloads(modules, source_checkpoint):
    source_name = "model.layers.0.ffn.gate.tid2eid"
    canonical_name = "model.layers.0.mlp.gate.tid2eid"
    tensors = {**source_checkpoint.tensors, source_name: ("I64", torch.tensor([[0, 1, 2, 3]]))}
    write_checkpoint(source_checkpoint.root, tensors)
    quant = modules.config.AscendVQ2A8Config(source_config())
    quant.bind_checkpoint(source_checkpoint.config)
    tensor = quant.root_tensor_spec(source_name)
    assert tensor["source_name"] == source_name
    assert tensor["name"] == canonical_name
    assert quant.root_by_name[canonical_name] is tensor
    assert source_name in quant.weight_map and canonical_name not in quant.weight_map
    assert quant.layer_spec(0).expert_ids == (0, 1, 2, 3)


@pytest.mark.parametrize(
    "suffix,canonical",
    [
        ("gate.tid2eid", "gate.tid2eid"),
        ("gate.e_score_correction_bias", "gate.e_score_correction_bias"),
        ("gate.weight", "gate.weight"),
        ("gate.scale", "gate.weight_scale"),
        ("shared_experts.gate_proj.weight", "shared_experts.gate_proj.weight"),
        ("shared_experts.gate_proj.scale", "shared_experts.gate_proj.weight_scale"),
        ("shared_experts.up_proj.weight", "shared_experts.up_proj.weight"),
        ("shared_experts.up_proj.scale", "shared_experts.up_proj.weight_scale"),
        ("shared_experts.down_proj.weight", "shared_experts.down_proj.weight"),
        ("shared_experts.down_proj.scale", "shared_experts.down_proj.weight_scale"),
    ],
)
def test_legacy_ffn_roots_share_the_mlp_field_contract(suffix, canonical):
    expected = f"model.layers.0.mlp.{canonical}"
    assert normalize_vq2a8_root_name(f"model.layers.0.ffn.{suffix}") == expected
    assert normalize_vq2a8_root_name(f"model.layers.0.mlp.{suffix}") == expected


@pytest.mark.parametrize("prefixes", [("ffn", "mlp"), ("mlp", "ffn")])
def test_source_v3_rejects_duplicate_ffn_and_mlp_root_aliases(modules, source_checkpoint, prefixes):
    tensors = dict(source_checkpoint.tensors)
    for prefix in prefixes:
        tensors[f"model.layers.0.{prefix}.gate.tid2eid"] = "I64", torch.tensor([[0, 1, 2, 3]])
    write_checkpoint(source_checkpoint.root, tensors)
    quant = modules.config.AscendVQ2A8Config(source_config())
    with pytest.raises(ValueError, match="Duplicate normalized source-v3 root"):
        quant.bind_checkpoint(source_checkpoint.config)
    assert not quant.layers and quant.checkpoint_root is None


@pytest.mark.parametrize(
    "suffix",
    [
        "ffn.gate.unknown",
        "ffn.unknown.weight",
        "ffn.gate_proj.weight",
        "ffn.shared_experts.gate_up_proj.weight",
        "ffnx.gate.tid2eid",
    ],
)
def test_legacy_ffn_prefix_does_not_allow_unknown_root_fields(suffix):
    with pytest.raises(ValueError, match="Unknown source-v3 root tensor"):
        normalize_vq2a8_root_name(f"model.layers.0.{suffix}")


@pytest.mark.parametrize("hash_layers", [0, 1])
def test_source_v3_requires_full_inventory_in_hash_and_non_hash_layers(modules, source_checkpoint, hash_layers):
    tensors = {name: value for name, value in source_checkpoint.tensors.items() if ".experts.3." not in name}
    write_checkpoint(source_checkpoint.root, tensors)
    source_checkpoint.config.hf_config.num_hash_layers = hash_layers
    quant = modules.config.AscendVQ2A8Config(source_config())
    with pytest.raises(ValueError, match="logical expert inventory"):
        quant.bind_checkpoint(source_checkpoint.config)


@pytest.mark.parametrize(
    "fault",
    ["missing", "mixed", "old_name", "dtype", "shape", "remote_layer", "unpaired_scale", "scale_shape", "old_order"],
)
def test_source_v3_header_inventory_rejects_invalid_exports(modules, source_checkpoint, fault):
    tensors = dict(source_checkpoint.tensors)
    key = "layers.0.ffn.experts.0.w13.out_dim_scale"
    metadata = None
    if fault == "missing":
        del tensors["layers.0.ffn.experts.0.w13.codebooks"]
    elif fault == "mixed":
        tensors["layers.0.ffn.experts.0.w13.weight"] = "U8", torch.ones(128, 64, dtype=torch.uint8)
    elif fault == "old_name":
        tensors["model.layers.0.mlp.experts.w13_row_scale_bits"] = tensors.pop(key)
    elif fault == "dtype":
        tensors[key] = "I64", tensors[key][1].long()
    elif fault == "shape":
        tensors[key] = "U64", tensors[key][1][:-1]
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
        index = json.loads(index_path.read_text(encoding="utf-8"))
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
