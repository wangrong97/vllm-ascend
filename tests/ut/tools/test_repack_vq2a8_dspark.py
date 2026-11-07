# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real safetensors byte round trips for the standalone DSpark packager."""

import importlib.util
import json
import struct
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.ut.tools.test_convert_vq2_mixed_gs32_v3 import read_tensors, snapshot, write_safetensors
from tools import inspect_vq2a8_checkpoint as inspect
from tools import repack_vq2a8_dspark as pack
from tools import serve_vq2a8 as serve


def save_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def assets(tmp_path):
    original, target = tmp_path / "original", tmp_path / "target"
    original.mkdir()
    target.mkdir()
    config = {
        "model_type": "deepseek_v4",
        "architectures": ["DeepseekV4ForCausalLM"],
        "hidden_size": 8,
        "vocab_size": 16,
        "hc_mult": 2,
        "num_hidden_layers": 3,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "qk_rope_head_dim": 2,
        "q_lora_rank": 4,
        "o_lora_rank": 4,
        "o_groups": 1,
        "n_routed_experts": 2,
        "num_experts_per_tok": 1,
        "moe_intermediate_size": 4,
        "n_shared_experts": 1,
        "num_hash_layers": 0,
        "expert_dtype": "fp4",
        "dspark_block_size": 5,
        "dspark_target_layer_ids": [0, 1, 2],
        "dspark_markov_rank": 4,
        "dspark_noise_token_id": 15,
        "compress_ratios": [0, 4, 128, 0, 0, 0],
        "quantization_config": {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "scale_fmt": "ue8m0",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
        },
    }
    save_json(original / "config.json", config)
    save_json(original / "tokenizer_config.json", {"tokenizer_class": "fixture"})
    save_json(original / "tokenizer.json", {"version": "1.0"})
    target_config = deepcopy(config)
    target_config["quantization_config"] = {
        "quant_method": "vq2a8",
        "format": "vq2_mixed_gs32_source",
        "format_version": 3,
    }
    target_config["compress_ratios"] = target_config["compress_ratios"][:3]
    for key in list(target_config):
        if key.startswith("dspark_"):
            del target_config[key]
    save_json(target / "config.json", target_config)
    save_json(target / "model.safetensors.index.json", {"weight_map": {"body": "rest/body.safetensors"}})
    (target / "rest").mkdir()
    write_safetensors(target / "rest/body.safetensors", {"body": ("BF16", [1], b"\x80\x3f")})

    bf16 = ("BF16", [2], b"\x80\x3f\x00\x40")
    fp32 = ("F32", [2], struct.pack("<2f", 0.5, -0.5))
    fp8 = ("F8_E4M3", [2, 2], bytes([0, 128, 56, 184]))
    scale = ("F8_E8M0", [1, 1], bytes([127]))
    packed = ("I8", [2, 2], bytes([0, 128, 127, 255]))
    shards = {
        "model-00001-of-00048.safetensors": {"embed.weight": bf16, "layers.0.attn.wq_a.weight": fp8},
        "model-00045-of-00048.safetensors": {
            "head.weight": bf16,
            "hc_head_fn": fp32,
            "hc_head_base": fp32,
            "hc_head_scale": fp32,
            "layers.2.attn.wq_a.weight": fp8,
        },
        "model-00002-of-00048.safetensors": {"layers.1.attn.wq_a.weight": fp8},
    }
    expected = {k: v for tensors in shards.values() for k, v in tensors.items() if not k.startswith("layers.")}
    for stage in range(3):
        tensors = {}
        prefix = f"mtp.{stage}."
        for family in ("attn", "ffn"):
            for part in ("base", "fn", "scale"):
                tensors[prefix + f"hc_{family}_{part}"] = fp32
        for name in (
            "attn.attn_sink",
            "attn.q_norm.weight",
            "attn.kv_norm.weight",
            "attn_norm.weight",
            "ffn_norm.weight",
        ):
            tensors[prefix + name] = bf16
        for name in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"):
            tensors[prefix + f"attn.{name}.weight"] = fp8
            tensors[prefix + f"attn.{name}.scale"] = scale
        for expert in ("shared_experts", "experts.0", "experts.1"):
            for projection in (1, 2, 3):
                tensors[prefix + f"ffn.{expert}.w{projection}.weight"] = fp8 if expert == "shared_experts" else packed
                tensors[prefix + f"ffn.{expert}.w{projection}.scale"] = scale
        tensors[prefix + "ffn.gate.weight"] = bf16
        tensors[prefix + "ffn.gate.bias"] = fp32
        if stage == 0:
            tensors[prefix + "main_proj.weight"] = fp8
            tensors[prefix + "main_proj.scale"] = scale
            tensors[prefix + "main_norm.weight"] = bf16
        if stage == 2:
            for name in (
                "norm.weight",
                "markov_head.markov_w1.weight",
                "markov_head.markov_w2.weight",
                "confidence_head.proj.weight",
            ):
                tensors[prefix + name] = bf16
            for name in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
                tensors[prefix + name] = fp32
        shards[f"model-{46 + stage:05d}-of-00048.safetensors"] = tensors
        expected.update(tensors)
    weight_map = {}
    for filename, tensors in shards.items():
        write_safetensors(original / filename, tensors)
        weight_map.update(dict.fromkeys(tensors, filename))
    save_json(original / "model.safetensors.index.json", {"weight_map": weight_map})
    return original, target, target / "dspark", expected


def test_inspector_recognizes_real_dspark_package(assets, tmp_path):
    source, target, output, _ = assets
    pack.repack(source, output, target=target)
    before = snapshot(output)
    directory, result = inspect.inspect_model(output, tmp_path / "inspection")
    report = result["checkpoints"][0]
    assert snapshot(output) == before
    assert report["classification"] == "dspark_original_package"
    assert not report["findings"]
    assert report["draft"]["critical_weights_missing"] == []
    assert set(report["draft"]["stage_tensor_counts"]) == {"0", "1", "2"}
    assert report["draft"]["payload_hashes_verified"] is False
    assert "--verify-only" in (directory / "summary.txt").read_text()


@pytest.mark.parametrize("problem", ["manifest", "stage", "critical_index_entry", "config", "impossible_stages"])
def test_inspector_reports_broken_dspark_package(assets, tmp_path, problem):
    source, target, output, _ = assets
    pack.repack(source, output, target=target)
    if problem == "manifest":
        path = output / pack.REPORT_NAME
        data = json.loads(path.read_text())
        data["tensors"]["embed.weight"]["shape"] = [99]
        expected = "DSPARK_MANIFEST"
    elif problem == "critical_index_entry":
        path = output / "model.safetensors.index.json"
        data = json.loads(path.read_text())
        del data["weight_map"]["mtp.0.main_proj.weight"]
        expected = "DSPARK_CRITICAL_WEIGHT"
    else:
        path = output / "config.json"
        data = json.loads(path.read_text())
        if problem == "stage":
            data["n_mtp_layers"] = 2
            expected = "DSPARK_STAGES"
        elif problem == "impossible_stages":
            data["n_mtp_layers"] = 10**12
            expected = "DSPARK_STAGES"
        else:
            data["quantization_config"]["quant_method"] = "vq2a8"
            expected = "DSPARK_CONFIG"
    save_json(path, data)
    _, result = inspect.inspect_model(output, tmp_path / "inspection")
    report = result["checkpoints"][0]
    assert report["classification"] == "dspark_original_package"
    assert report["findings"][expected]["severity"] == "error"


def test_package_preserves_every_selected_tensor_and_original_config(assets, monkeypatch):
    source, target, output, expected = assets
    # Exercise tensors crossing streaming chunk boundaries, including FP8 and I8.
    monkeypatch.setattr(pack, "COPY_CHUNK", 3)
    before_source, before_target = snapshot(source), snapshot(target)
    report = pack.repack(source, output, target=target)
    actual = {}
    for path in sorted(output.glob("*.safetensors")):
        _, tensors = read_tensors(path)
        actual.update(tensors)
    assert actual == expected
    assert report["total_tensors"] == len(expected)
    assert report["shared_tensors"] == 5
    assert report["output_shards"] == 5
    assert all(not name.startswith("layers.") for name in actual)
    assert (output / "config.json").read_bytes() == (source / "config.json").read_bytes()
    assert snapshot(source) == before_source
    after_target = {name: digest for name, digest in snapshot(target).items() if not name.startswith("dspark/")}
    assert after_target == before_target
    assert not list(target.glob(".dspark.repacking-*"))
    # It must be independently deployable after the original directory moves.
    source.rename(source.with_name("original-unavailable"))
    assert pack.verify_package(output)["tensors"] == report["tensors"]


def test_packaged_checkpoint_passes_existing_runtime_preflight_and_launcher(assets):
    source, target, output, _ = assets
    pack.repack(source, output, target=target)
    module_path = Path(__file__).resolve().parents[3] / "vllm_ascend/quantization/methods/w2a8/vq2a8_gs32_dspark.py"
    spec = importlib.util.spec_from_file_location("dspark_repack_contract", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = SimpleNamespace(
        use_v2_model_runner=True,
        model_config=SimpleNamespace(
            model=str(target), hf_config=SimpleNamespace(**pack.read_json(target / "config.json"))
        ),
        speculative_config=SimpleNamespace(
            method="dspark",
            num_speculative_tokens=5,
            enforce_eager=False,
            model=str(output),
            draft_model_config=SimpleNamespace(
                model=str(output), hf_config=SimpleNamespace(**pack.read_json(output / "config.json"))
            ),
        ),
    )
    module.validate_vq2a8_dspark_config(config)
    args = SimpleNamespace(dspark=None, dspark_model=output, model=target, num_speculative_tokens=5)
    expected = {
        "method": "dspark",
        "model": str(output),
        "num_speculative_tokens": 5,
        "enforce_eager": False,
    }
    assert serve.dspark_config(args) == expected
    args.dspark, args.dspark_model = True, None
    assert serve.dspark_config(args) == expected
    # Standard loader paths are relative to each checkpoint root. Draft files
    # found by the target's depth-one glob are absent from the target index.
    target_index = pack.read_json(target / "model.safetensors.index.json")["weight_map"]
    selected = [p for p in target.glob("*/*.safetensors") if str(p.relative_to(target)) in set(target_index.values())]
    assert selected == [target / "rest/body.safetensors"]


def test_dry_run_reads_only_selected_headers_and_writes_nothing(assets):
    source, target, output, expected = assets
    # Unselected target-only shard is not needed to extract a draft.
    (source / "model-00002-of-00048.safetensors").unlink()
    before = snapshot(source.parent)
    report = pack.repack(source, output, target=target, dry_run=True)
    assert report["total_tensors"] == len(expected)
    assert snapshot(source.parent) == before


@pytest.mark.parametrize(
    "missing", ["mtp.1.ffn.experts.1.w3.scale", "mtp.2.confidence_head.proj.weight", "head.weight"]
)
def test_missing_draft_dependency_fails_before_writing(assets, missing):
    source, target, output, _ = assets
    path = source / "model.safetensors.index.json"
    index = pack.read_json(path)
    del index["weight_map"][missing]
    save_json(path, index)
    with pytest.raises(ValueError, match="Incomplete original DSpark"):
        pack.repack(source, output, target=target)
    assert not output.exists()
    assert not list(target.glob(".dspark.repacking-*"))


@pytest.mark.parametrize(
    "field,value",
    [
        ("dspark_block_size", 1),
        ("dspark_target_layer_ids", [0, 8]),
        ("compress_ratios", [0, 4, 128]),
        ("expert_dtype", "fp8"),
        ("quantization_config", {"quant_method": "vq2a8"}),
    ],
)
def test_rejects_nonoriginal_or_incomplete_draft_config(assets, field, value):
    source, target, output, _ = assets
    path = source / "config.json"
    config = pack.read_json(path)
    config[field] = value
    save_json(path, config)
    with pytest.raises(ValueError):
        pack.repack(source, output, target=target)
    assert not output.exists()


def test_target_geometry_mismatch_and_existing_output_are_rejected(assets):
    source, target, output, _ = assets
    config = pack.read_json(target / "config.json")
    config["vocab_size"] += 1
    save_json(target / "config.json", config)
    with pytest.raises(ValueError, match="vocab_size"):
        pack.repack(source, output, target=target)
    output.mkdir()
    (output / "keep").write_text("existing")
    with pytest.raises(ValueError, match="already exists"):
        pack.repack(source, output)
    assert (output / "keep").read_text() == "existing"


def test_insufficient_space_and_source_overlap_do_not_write(assets, monkeypatch):
    source, target, output, _ = assets
    with pytest.raises(ValueError, match="overlap"):
        pack.repack(source, source / "dspark")
    monkeypatch.setattr(pack.shutil, "disk_usage", lambda path: SimpleNamespace(free=0))
    before = snapshot(target)
    with pytest.raises(ValueError, match="Insufficient free space"):
        pack.repack(source, output, target=target)
    assert snapshot(target) == before


@pytest.mark.parametrize("escape", ["../outside.safetensors", "/outside.safetensors", "external.safetensors"])
def test_shards_cannot_escape_source(assets, escape):
    source, target, output, _ = assets
    index_path = source / "model.safetensors.index.json"
    index = pack.read_json(index_path)
    first_shard = index["weight_map"]["embed.weight"]
    outside = source.parent / "outside.safetensors"
    outside.write_bytes((source / first_shard).read_bytes())
    (source / "external.safetensors").symlink_to(outside)
    index["weight_map"] = {k: escape if v == first_shard else v for k, v in index["weight_map"].items()}
    save_json(index_path, index)
    with pytest.raises(ValueError, match="Unsafe|external"):
        pack.repack(source, output, target=target)
    assert not output.exists()


def test_corrupt_header_or_unindexed_tensor_is_rejected(assets):
    source, target, output, _ = assets
    path = source / "model-00046-of-00048.safetensors"
    data = path.read_bytes()
    path.write_bytes(data[:-1])
    with pytest.raises(ValueError, match="Truncated"):
        pack.repack(source, output, target=target)
    path.write_bytes(data)
    header, tensors = read_tensors(path)
    tensors["unexpected"] = ("U8", [1], b"\x01")
    write_safetensors(path, tensors)
    with pytest.raises(ValueError, match="index/header mismatch"):
        pack.repack(source, output, target=target)


def test_verify_detects_payload_and_config_tampering(assets):
    source, target, output, _ = assets
    pack.repack(source, output)
    path = next(output.glob("*.safetensors"))
    data = path.read_bytes()
    path.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        pack.verify_package(output)
    path.write_bytes(data)
    with (output / "config.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="SHA256 mismatch: config.json"):
        pack.verify_package(output)


def test_failed_verification_never_publishes_output(assets, monkeypatch):
    source, target, output, _ = assets

    def fail_verification(root):
        raise ValueError("injected readback failure")

    monkeypatch.setattr(pack, "verify_package", fail_verification)
    with pytest.raises(ValueError, match="readback failure"):
        pack.repack(source, output)
    assert not output.exists()
    assert len(list(target.glob(".dspark.repacking-*"))) == 1


def test_source_mutation_during_copy_is_detected(assets, monkeypatch):
    source, target, output, _ = assets
    original_writer = pack.write_shard

    def mutate_after_copy(shard, destination):
        result = original_writer(shard, destination)
        with (source / "config.json").open("a") as stream:
            stream.write(" ")
        return result

    monkeypatch.setattr(pack, "write_shard", mutate_after_copy)
    with pytest.raises(ValueError, match="changed during repack"):
        pack.repack(source, output)
    assert not output.exists()


def test_cli_works_standalone_outside_repository(assets, tmp_path):
    source, target, output, _ = assets
    script = tmp_path / "standalone_repack.py"
    script.write_bytes(Path(pack.__file__).read_bytes())
    command = [sys.executable, str(script), "--source", str(source), "--output", str(output), "--target", str(target)]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "DSPARK_REPACK_STATUS=verified" in result.stdout
    result = subprocess.run(
        [sys.executable, str(script), "--output", str(output), "--verify-only"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "DSPARK_REPACK_STATUS=verified" in result.stdout
