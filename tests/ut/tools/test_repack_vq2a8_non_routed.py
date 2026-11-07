# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small real-byte checkpoints and numerical checks for non-routed export."""

import hashlib
import json
import math
import struct

import pytest
import torch

from tools import repack_vq2a8_non_routed as repacker


def tensor_bytes(value):
    return value.contiguous().view(torch.uint8).numpy().tobytes()


def write_safetensors(path, tensors):
    header = {"__metadata__": {"format": "pt"}}
    payload = bytearray()
    for name, (dtype, shape, data) in tensors.items():
        start = len(payload)
        payload.extend(data)
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [start, len(payload)]}
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def read_safetensors(path):
    data = path.read_bytes()
    size = struct.unpack_from("<Q", data)[0]
    header = json.loads(data[8 : 8 + size])
    return {
        name: (
            entry["dtype"],
            entry["shape"],
            data[8 + size + entry["data_offsets"][0] : 8 + size + entry["data_offsets"][1]],
        )
        for name, entry in header.items()
        if name != "__metadata__"
    }


def snapshot(root):
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def make_source(root):
    """One layer with a VQ2 expert, an MXFP4 expert and independent HF roots."""
    root.mkdir()
    config = {
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_v4",
        "hidden_size": 128,
        "moe_intermediate_size": 128,
        "num_hidden_layers": 1,
        "n_routed_experts": 2,
        "num_hash_layers": 0,
        "num_experts_per_tok": 1,
        "n_shared_experts": 1,
        "compress_ratios": [0],
        "vocab_size": 128,
        "quantization_config": {"quant_method": "vq2a8", "format": "vq2_mixed_gs32_source", "format_version": 3},
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "tokenizer_config.json").write_text('{"tokenizer_class":"TokenizersBackend"}', encoding="utf-8")
    (root / "tokenizer.json").write_text('{"fixture":"preserve bytes"}', encoding="utf-8")
    (root / "generation_config.json").write_text('{"eos_token_id":1}', encoding="utf-8")

    tensors = {}

    def add(name, dtype, shape, value):
        data = tensor_bytes(torch.full(shape, value, dtype=dtype))
        tensors[name] = ({torch.bfloat16: "BF16", torch.float32: "F32"}[dtype], shape, data)

    quantized_names = [
        "self_attn.q_a_proj",
        "self_attn.q_b_proj",
        "self_attn.kv_proj",
        "self_attn.o_a_proj",
        "self_attn.o_b_proj",
        "mlp.shared_experts.gate_proj",
        "mlp.shared_experts.up_proj",
        "mlp.shared_experts.down_proj",
    ]
    for index, name in enumerate(quantized_names):
        add(f"model.layers.0.{name}.weight", torch.bfloat16, [128, 128], (index + 1) / 8)
    for name in ("self_attn.q_a_norm", "self_attn.kv_norm", "input_layernorm", "post_attention_layernorm"):
        add(f"model.layers.0.{name}.weight", torch.float32, [128], 1)
    add("model.layers.0.self_attn.sinks", torch.float32, [1], -0.125)
    for prefix in ("attn_hc", "ffn_hc"):
        for field, shape in (("fn", [24, 512]), ("base", [24]), ("scale", [3])):
            add(f"model.layers.0.{prefix}.{field}", torch.float32, shape, 0.125)
    add("model.layers.0.mlp.gate.weight", torch.bfloat16, [2, 128], 0.375)
    add("model.layers.0.mlp.gate.e_score_correction_bias", torch.float32, [2], -0.25)
    add("model.embed_tokens.weight", torch.bfloat16, [128, 128], 0.5)
    add("head.weight", torch.bfloat16, [128, 128], -0.5)
    add("model.norm.weight", torch.float32, [128], 1)
    for field, shape in (("hc_fn", [4, 512]), ("hc_base", [4]), ("hc_scale", [1])):
        add(f"model.hc_head.{field}", torch.float32, shape, 0.25)

    for projection, rows in (("w13", 256), ("w2", 128)):
        prefix = f"layers.0.ffn.experts.0.{projection}"
        tensors[f"{prefix}.packed_indices"] = (
            "U8",
            [rows // 32, 8, 16, 8],
            bytes(range(256)) * (rows * 128 // 4 // 256),
        )
        # Include negative zero: a numeric recast would lose this bit.
        tensors[f"{prefix}.codebooks"] = ("F8_E4M3", [rows // 32, 4, 16, 2], b"\x80\x38" * (rows * 128 // 32 // 2))
        tensors[f"{prefix}.out_dim_scale"] = ("U64", [rows], struct.pack("<Q", 0x3F800000) * rows)
        prefix = f"layers.0.ffn.experts.1.{projection}"
        tensors[f"{prefix}.weight"] = ("U8", [rows, 64], bytes(range(256)) * (rows * 64 // 256))
        tensors[f"{prefix}.scale"] = ("F8_E8M0", [rows, 4], b"\x00\x7f\xfe\x80" * rows)
    save_source(root, tensors)
    return tensors


def save_source(root, tensors):
    # Routed and non-routed share the same file, as in the real checkpoint.
    write_safetensors(root / "mixed.safetensors", tensors)
    index = {
        "metadata": {"total_size": sum(len(value[2]) for value in tensors.values())},
        "weight_map": dict.fromkeys(tensors, "mixed.safetensors"),
    }
    (root / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")


def output_tensors(root):
    index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
    result = {}
    for shard in set(index["weight_map"].values()):
        tensors = read_safetensors(root / shard)
        assert not result.keys() & tensors.keys()
        assert {name for name, relative in index["weight_map"].items() if relative == shard} == tensors.keys()
        result.update(tensors)
    assert index["metadata"]["total_size"] == sum(len(value[2]) for value in result.values())
    return result


def test_round_trip_preserves_routed_payload_and_rebuilds_index(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "output"
    before_tensors = make_source(source)
    before = snapshot(source)
    report = repacker.repack(source, destination, quantize_fp8=True, threads=1)
    after = output_tensors(destination)
    assert len(after) == len(before_tensors) + 8
    saved_payload = sum(len(value[2]) for value in before_tensors.values()) - sum(
        len(value[2]) for value in after.values()
    )
    # Eight 128x128 BF16 matrices acquire one scale byte each; five Norms halve.
    assert saved_payload == 8 * (128 * 128 - 1) + 5 * 128 * 2
    for name, tensor in before_tensors.items():
        if ".experts." in name:
            assert after[name] == (tensor[0], list(tensor[1]), tensor[2])
    assert after["embed.weight"] == before_tensors["model.embed_tokens.weight"]
    assert after["head.weight"] == before_tensors["head.weight"]
    assert after["layers.0.ffn.gate.weight"] == before_tensors["model.layers.0.mlp.gate.weight"]
    assert after["hc_head_fn"] == before_tensors["model.hc_head.hc_fn"]
    assert after["norm.weight"] == ("BF16", [128], tensor_bytes(torch.ones(128, dtype=torch.bfloat16)))
    dtype, shape, payload = after["layers.0.attn.wq_a.weight"]
    assert (dtype, shape) == ("F8_E4M3", [128, 128])
    scales = after["layers.0.attn.wq_a.scale"]
    assert scales[:2] == ("F8_E8M0", [1, 1])
    decoded = torch.frombuffer(bytearray(payload), dtype=torch.float8_e4m3fn).float()
    decoded *= 2 ** (scales[2][0] - 127)
    assert torch.equal(decoded, torch.full_like(decoded, 0.125))
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "generation_config.json"):
        assert (destination / name).read_bytes() == (source / name).read_bytes()
    assert snapshot(source) == before
    saved_report = json.loads((destination / "repack_report.json").read_text(encoding="utf-8"))
    assert saved_report == report
    source_bytes = sum(path.stat().st_size for path in source.rglob("*") if path.is_file())
    target_bytes = sum(path.stat().st_size for path in destination.rglob("*") if path.is_file())
    assert report["storage"]["payload"]["saved_bytes"] == saved_payload
    assert report["storage"]["files"]["source_bytes"] == source_bytes
    assert report["storage"]["files"]["target_bytes"] == target_bytes
    assert report["storage"]["files"]["saved_bytes"] == source_bytes - target_bytes
    assert not list(tmp_path.glob(".output.*"))


def test_dry_run_does_not_write_or_quantize(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "output"
    make_source(source)
    before = snapshot(tmp_path)
    repacker.repack(source, destination, dry_run=True)
    assert snapshot(tmp_path) == before
    assert not destination.exists()


def test_fp8_requires_explicit_quantization_flag(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "output"
    make_source(source)
    with pytest.raises(ValueError, match="quantiz"):
        repacker.repack(source, destination)
    assert not destination.exists()


def test_block128_orientation_and_zero_scale():
    values = torch.empty((256, 384), dtype=torch.float32)
    block_values = [[448.0, 0.0, 896.0], [224.0, 1792.0, -448.0]]
    for row in range(2):
        for column in range(3):
            values[row * 128 : (row + 1) * 128, column * 128 : (column + 1) * 128] = block_values[row][column]
    quantized, scales, _ = repacker.quantize_fp8_block128(values)
    assert quantized.dtype == torch.float8_e4m3fn
    assert scales.dtype == torch.uint8
    assert scales.tolist() == [[127, 127, 128], [126, 129, 127]]
    scale_values = torch.exp2(scales.float() - 127).repeat_interleave(128, 0).repeat_interleave(128, 1)
    assert torch.equal(quantized.float() * scale_values, values)


def test_fp8_round_to_nearest_even_and_exponent_boundary():
    values = torch.zeros((128, 256), dtype=torch.float32)
    values[0, 0] = 448.0
    values[0, 1:5] = torch.tensor([1.0625, 1.1875, -1.0625, -1.1875])
    values[0, 128] = torch.nextafter(torch.tensor(448.0), torch.tensor(math.inf))
    quantized, scales, _ = repacker.quantize_fp8_block128(values)
    assert scales.tolist() == [[127, 128]]
    assert quantized.float()[0, 1:5].tolist() == [1.0, 1.25, -1.0, -1.25]


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nonfinite_fp8_input_is_rejected(bad):
    values = torch.zeros((128, 128))
    values[3, 7] = bad
    with pytest.raises(ValueError, match="finite"):
        repacker.quantize_fp8_block128(values)


def test_nonexact_norm_does_not_publish_checkpoint(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "output"
    tensors = make_source(source)
    tensors["model.norm.weight"] = ("F32", [128], tensor_bytes(torch.full((128,), 1.001)))
    save_source(source, tensors)
    before = snapshot(source)
    with pytest.raises(ValueError, match="exact|BF16|bfloat16"):
        repacker.repack(source, destination, quantize_fp8=True, threads=1)
    assert not destination.exists()
    assert snapshot(source) == before


def test_existing_output_is_not_overwritten(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "output"
    make_source(source)
    destination.mkdir()
    marker = destination / "keep.txt"
    marker.write_text("user data", encoding="utf-8")
    with pytest.raises(ValueError, match="exist"):
        repacker.repack(source, destination, quantize_fp8=True)
    assert marker.read_text(encoding="utf-8") == "user data"


def test_output_inside_source_is_rejected(tmp_path):
    source = tmp_path / "source"
    make_source(source)
    before = snapshot(source)
    with pytest.raises(ValueError):
        repacker.repack(source, source / "output", quantize_fp8=True)
    assert snapshot(source) == before


def test_identity_perm_is_rejected(tmp_path):
    source = tmp_path / "source"
    tensors = make_source(source)
    tensors["layers.0.ffn.experts.0.w13.perm"] = ("I64", [128], struct.pack("<128q", *range(128)))
    save_source(source, tensors)
    with pytest.raises(ValueError, match="perm"):
        repacker.plan_model(source)


def test_missing_expert_field_is_rejected(tmp_path):
    source = tmp_path / "source"
    tensors = make_source(source)
    del tensors["layers.0.ffn.experts.1.w2.scale"]
    save_source(source, tensors)
    with pytest.raises(ValueError):
        repacker.plan_model(source)


def test_index_header_disagreement_is_rejected(tmp_path):
    source = tmp_path / "source"
    make_source(source)
    index_path = source / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    del index["weight_map"]["head.weight"]
    index_path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ValueError):
        repacker.plan_model(source)


def test_original_names_preserve_indexer_and_hc_hierarchy():
    names = {
        "model.layers.2.self_attn.compressor.indexer.q_b_proj.weight": "layers.2.attn.indexer.wq_b.weight",
        "model.layers.2.self_attn.compressor.indexer.gate_proj.weight": "layers.2.attn.indexer.compressor.wgate.weight",
        "model.layers.2.self_attn.compressor.indexer.position_bias": "layers.2.attn.indexer.compressor.ape",
        "model.layers.2.self_attn.compressor.indexer.scorer.weights_proj.weight": (
            "layers.2.attn.indexer.weights_proj.weight"
        ),
        "model.layers.2.mlp.shared_experts.up_proj.weight": "layers.2.ffn.shared_experts.w3.weight",
        "model.layers.2.mlp.shared_experts.down_proj.weight": "layers.2.ffn.shared_experts.w2.weight",
        "model.layers.2.attn_hc.scale": "layers.2.hc_attn_scale",
        "model.hc_head.hc_fn": "hc_head_fn",
    }
    for source, expected in names.items():
        assert repacker.original_name(source) == expected
