# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read-only diagnostics against real safetensors fixtures, without torch/NPU."""

import gzip
import json
import shutil
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from tests.ut.tools.test_convert_vq2_mixed_gs32_v3 import snapshot, write_json, write_safetensors
from tools import inspect_vq2a8_checkpoint as inspect
from tools.vq2a8_source_contract import source_contract


@pytest.fixture
def checkpoint(tmp_path):
    root = tmp_path / "model"
    (root / "rest").mkdir(parents=True)
    (root / "vq2_mixed_gs32").mkdir()
    write_json(
        root / "config.json",
        {
            "model_type": "deepseek_v4",
            "hidden_size": 64,
            "moe_intermediate_size": 64,
            "num_hidden_layers": 1,
            "n_routed_experts": 2,
            "quantization_config": {"quant_method": "vq2a8", "format": inspect.FORMAT, "format_version": 3},
        },
    )
    tensors = {}
    for expert in range(2):
        for projection, rows in (("w13", 128), ("w2", 64)):
            prefix = f"layers.0.ffn.experts.{expert}.{projection}."
            if (expert == 0) == (projection == "w13"):
                fields = {
                    "packed_indices": ("U8", [rows // 32, 4, 16, 8], bytes(rows * 64 // 4)),
                    "codebooks": ("F8_E4M3", [rows // 32, 2, 16, 2], bytes(rows // 32 * 2 * 32)),
                    "out_dim_scale": ("U64", [rows], struct.pack("<Q", 0x3F800000) * rows),
                }
            else:
                fields = {
                    "weight": ("U8", [rows, 32], bytes(rows * 32)),
                    "scale": ("F8_E8M0", [rows, 2], bytes([127]) * rows * 2),
                }
            tensors.update({prefix + k: v for k, v in fields.items()})
    write_safetensors(root / "vq2_mixed_gs32/layer_00.safetensors", tensors)
    write_safetensors(root / "rest/body.safetensors", {"model.norm.weight": ("BF16", [64], bytes(128))})
    weight_map = dict.fromkeys(tensors, "vq2_mixed_gs32/layer_00.safetensors")
    weight_map["model.norm.weight"] = "rest/body.safetensors"
    write_json(root / "model.safetensors.index.json", {"weight_map": weight_map})
    return root, tensors


def run(root, tmp_path, samples=0):
    directory, report = inspect.inspect_model(root, tmp_path / "reports", sample_count=samples)
    return directory, report["checkpoints"][0]


def errors(report):
    return {code for code, item in report["findings"].items() if item["severity"] == "error"}


@pytest.mark.parametrize("fault", [None, "mixed", "sidecar"])
def test_folded_profile_inspector_agrees_with_runtime_config_and_inventory(checkpoint, tmp_path, fault):
    root, tensors = checkpoint
    contract = source_contract()
    config = json.loads((root / "config.json").read_text())
    config["quantization_config"] = dict((*contract["SOURCE_V3_CONFIG"], *contract["FOLDED_HEAD_CONFIG"]))
    write_json(root / "config.json", config)
    if fault != "mixed":
        converted = {}
        for expert in range(2):
            for projection, source in (("w13", 0), ("w2", 1)):
                prefix = f"layers.0.ffn.experts.{source}.{projection}."
                converted.update(
                    {
                        name.replace(f"experts.{source}.", f"experts.{expert}."): value
                        for name, value in tensors.items()
                        if name.startswith(prefix)
                    }
                )
        write_safetensors(root / "vq2_mixed_gs32/layer_00.safetensors", converted)
        index = json.loads((root / "model.safetensors.index.json").read_text())
        index["weight_map"] = {name: shard for name, shard in index["weight_map"].items() if ".experts." not in name}
        index["weight_map"].update(dict.fromkeys(converted, "vq2_mixed_gs32/layer_00.safetensors"))
        write_json(root / "model.safetensors.index.json", index)
    if fault == "sidecar":
        write_json(root / "rotation_config.json", {**dict(contract["FOLDED_ROTATION_CONFIG"]), "online": True})
    _, result = inspect.inspect_model(root, tmp_path / "reports", contract=inspect.EXPORT_CONTRACT)
    blockers = result["checkpoints"][0]["runtime_compatibility"]["findings"]
    assert "V3_CONFIG" not in blockers
    assert ("FOLDED_EXPERT_FIELDS" in blockers) == (fault == "mixed")
    assert ("ROTATION_CONFIG" in blockers) == (fault == "sidecar")


def test_collects_complete_inventory_without_reading_payloads(checkpoint, tmp_path, monkeypatch):
    root, tensors = checkpoint
    before = snapshot(root)
    original_open = Path.open

    class HeaderOnly:
        def __init__(self, stream):
            self.stream = stream
            length = struct.unpack("<Q", stream.read(8))[0]
            self.end = 8 + length
            stream.seek(0)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def read(self, count=-1):
            assert count >= 0 and self.stream.tell() + count <= self.end
            return self.stream.read(count)

    def checked_open(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        if path.suffix == ".safetensors" and args == ("rb",):
            return HeaderOnly(stream)
        return stream

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", checked_open)
        directory, report = run(root, tmp_path)
    assert snapshot(root) == before
    assert not errors(report)
    assert report["classification"] == "source_v3_candidate"
    assert report["samples"] == []
    with gzip.open(directory / "tensors.jsonl.gz", "rt") as stream:
        records = [json.loads(line) for line in stream]
    assert {r["name"] for r in records} == set(tensors) | {"model.norm.weight"}
    with zipfile.ZipFile(directory / "checkpoint-report.zip") as bundle:
        assert set(bundle.namelist()) == {
            "summary.txt",
            "report.json",
            "metadata.json",
            "files.json",
            "tensors.jsonl.gz",
        }


@pytest.mark.parametrize(
    "field,bad_bytes",
    [
        ("out_dim_scale", struct.pack("<Q", 0x13F800000) * 128),
        ("codebooks", bytes([127]) * 256),
    ],
)
def test_sampled_vq_values(checkpoint, tmp_path, field, bad_bytes):
    root, tensors = checkpoint
    name = f"layers.0.ffn.experts.0.w13.{field}"
    dtype, shape, _ = tensors[name]
    tensors[name] = dtype, shape, bad_bytes
    write_safetensors(root / "vq2_mixed_gs32/layer_00.safetensors", tensors)
    _, report = run(root, tmp_path, samples=8)
    assert "SAMPLED_VALUE" in errors(report)
    assert any(x["name"] == name and x["status"] == "failed" for x in report["samples"])


def test_sampled_mxfp4_nan(checkpoint, tmp_path):
    root, tensors = checkpoint
    name = "layers.0.ffn.experts.0.w2.scale"
    tensors[name] = "F8_E8M0", [64, 2], bytes([255]) * 128
    write_safetensors(root / "vq2_mixed_gs32/layer_00.safetensors", tensors)
    _, report = run(root, tmp_path, samples=8)
    assert "SAMPLED_VALUE" in errors(report)


def test_indexed_tensor_missing_from_shard_is_still_reported(checkpoint, tmp_path):
    root, tensors = checkpoint
    del tensors["layers.0.ffn.experts.0.w13.out_dim_scale"]
    write_safetensors(root / "vq2_mixed_gs32/layer_00.safetensors", tensors)
    _, report = run(root, tmp_path)
    assert "INDEX_TENSOR_MISSING" in errors(report)
    assert "EXPERT_FIELDS" in errors(report)


def test_natural_columns_do_not_skip_shapes_or_samples(checkpoint, tmp_path):
    root, tensors = checkpoint
    prefix = "layers.0.ffn.experts.0.w13."
    dtype, _, data = tensors[prefix + "packed_indices"]
    tensors[prefix + "packed_indices"] = dtype, [len(data)], data
    dtype, shape, data = tensors[prefix + "codebooks"]
    tensors[prefix + "codebooks"] = dtype, shape, bytes([127]) * len(data)
    write_safetensors(root / "vq2_mixed_gs32/layer_00.safetensors", tensors)
    before = snapshot(root)
    directory, report = run(root, tmp_path, samples=8)
    assert snapshot(root) == before
    assert {"EXPERT_DTYPE_SHAPE", "SAMPLED_VALUE"} <= errors(report)
    assert report["expert_checks"]["projection_counts"] == {
        "vq2_natural_columns": 2,
        "mxfp4_complete": 2,
    }
    coverage = report["expert_checks"]["sample_coverage"]
    assert coverage["codebooks"]["failed"] == 1
    assert coverage["out_dim_scale"]["passed"] == 2
    assert "perm" not in coverage
    assert report["expert_checks"]["field_checks"]["mismatched"] == 1
    summary = (directory / "summary.txt").read_text()
    assert "REPORT_SCHEMA_VERSION=2" in summary
    assert "samples.codebooks=" in summary
    assert "expert_layouts:" in summary


def test_partial_projection_and_no_sample_coverage(checkpoint, tmp_path):
    root, tensors = checkpoint
    del tensors["layers.0.ffn.experts.0.w13.codebooks"]
    write_safetensors(root / "vq2_mixed_gs32/layer_00.safetensors", tensors)
    _, report = run(root, tmp_path)
    assert "EXPERT_FIELDS" in errors(report)
    coverage = report["expert_checks"]["sample_coverage"]["out_dim_scale"]
    assert coverage == {"eligible": 2, "selected": 0, "not_selected": 2, "passed": 0, "failed": 0, "skipped": 0}


def test_quarot_header_and_original_roots_are_visible(checkpoint, tmp_path):
    root, _ = checkpoint
    (root / "optional").mkdir()
    write_safetensors(root / "optional/quarot.safetensors", {"R1": ("F32", [2, 2], bytes(16))})
    write_safetensors(root / "rest/body.safetensors", {"embed.weight": ("BF16", [64], bytes(128))})
    index_path = root / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    del index["weight_map"]["model.norm.weight"]
    index["weight_map"]["embed.weight"] = "rest/body.safetensors"
    write_json(index_path, index)
    config_path = root / "config.json"
    config = json.loads(config_path.read_text())
    config["quantization_config"] = {"quant_method": "exporter_custom"}
    write_json(config_path, config)
    directory, report = run(root, tmp_path)
    assert "V3_CONFIG" in errors(report)
    assert "V3_ROOT_NAME" not in errors(report)
    assert "ROTATION_PROVENANCE" in report["findings"]
    artifact = report["rotation_artifacts"][0]
    assert artifact["layouts"][0]["shape"] == [2, 2]
    assert artifact["layouts"][0]["dtype"] == "F32"
    assert not artifact["indexed"] and not artifact["payload_checked"]
    assert all(layout["family"] != "R1" for layout in report["root_layouts"])
    summary = (directory / "summary.txt").read_text()
    assert "exporter_custom" in summary and "R1 F32 [2, 2]" in summary


def test_mtp_count_alone_does_not_identify_dspark(tmp_path):
    root = tmp_path / "mtp"
    root.mkdir()
    write_json(root / "config.json", {"model_type": "deepseek_v4", "n_mtp_layers": 1})
    write_safetensors(root / "draft.safetensors", {"mtp.0.attn.attn_sink": ("BF16", [1], bytes(2))})
    _, report = run(root, tmp_path)
    assert report["classification"] == "mtp_candidate"
    assert report["draft"]["stage_tensor_counts"] == {"0": 1}
    assert report["draft"]["signature_weights_present"] == []
    assert "V3_CONFIG" not in errors(report)


def test_target_with_retained_dspark_config_has_no_draft(checkpoint, tmp_path):
    root, _ = checkpoint
    config_path = root / "config.json"
    config = json.loads(config_path.read_text())
    config["dspark_block_size"] = 5
    write_json(config_path, config)
    _, report = run(root, tmp_path)
    assert not errors(report)
    assert report["draft"]["classification"] == "no_mtp"


def test_wrong_shape_and_missing_projection(checkpoint, tmp_path):
    root, tensors = checkpoint
    name = "layers.0.ffn.experts.0.w13.packed_indices"
    dtype, _, data = tensors[name]
    tensors[name] = dtype, [len(data)], data
    tensors = {n: value for n, value in tensors.items() if not n.startswith("layers.0.ffn.experts.1.w2.")}
    write_safetensors(root / "vq2_mixed_gs32/layer_00.safetensors", tensors)
    _, report = run(root, tmp_path)
    assert {"EXPERT_DTYPE_SHAPE", "MISSING_PROJECTION"} <= errors(report)


def test_missing_and_truncated_shards_still_produce_bundle(checkpoint, tmp_path):
    root, _ = checkpoint
    (root / "rest/body.safetensors").unlink()
    shard = root / "vq2_mixed_gs32/layer_00.safetensors"
    shard.write_bytes(shard.read_bytes()[:-1])
    directory, report = run(root, tmp_path)
    assert "FILE_READ" in errors(report)
    assert (directory / "checkpoint-report.zip").is_file()


def test_duplicate_json_keys_are_not_silently_overwritten(checkpoint, tmp_path):
    root, _ = checkpoint
    (root / "model.safetensors.index.json").write_text('{"weight_map":{},"weight_map":{}}')
    _, report = run(root, tmp_path)
    assert {"JSON_READ", "INDEX_REQUIRED"} <= errors(report)


def test_duplicate_tensor_in_selected_shards(checkpoint, tmp_path):
    root, tensors = checkpoint
    write_safetensors(
        root / "rest/body.safetensors",
        {
            "model.norm.weight": ("BF16", [64], bytes(128)),
            "layers.0.ffn.experts.0.w13.out_dim_scale": tensors["layers.0.ffn.experts.0.w13.out_dim_scale"],
        },
    )
    _, report = run(root, tmp_path)
    assert {"DUPLICATE_TENSOR", "HEADER_NOT_INDEXED"} <= errors(report)


def test_unindexed_shard_does_not_override_target_and_dspark_is_separate(checkpoint, tmp_path):
    root, _ = checkpoint
    write_safetensors(root / "extra.safetensors", {"model.norm.weight": ("BF16", [1], bytes(2))})
    draft = root / "dspark"
    draft.mkdir()
    write_json(draft / "config.json", {"model_type": "deepseek_v4"})
    write_safetensors(draft / "draft.safetensors", {"mtp.0.main_norm.weight": ("BF16", [64], bytes(128))})
    write_json(draft / "model.safetensors.index.json", {"weight_map": {"mtp.0.main_norm.weight": "draft.safetensors"}})
    _, result = inspect.inspect_model(root, tmp_path / "reports")
    target, dspark = result["checkpoints"]
    assert not errors(target)
    assert "UNINDEXED_SHARD" in target["findings"]
    assert target["mtp_tensor_count"] == 0
    assert dspark["mtp_tensor_count"] == 1
    assert dspark["label"] == "dspark"


def test_mtp_in_target_index_and_mixed_depth(checkpoint, tmp_path):
    root, _ = checkpoint
    write_safetensors(root / "draft.safetensors", {"mtp.0.main_norm.weight": ("BF16", [64], bytes(128))})
    index_path = root / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["weight_map"]["mtp.0.main_norm.weight"] = "draft.safetensors"
    write_json(index_path, index)
    _, report = run(root, tmp_path)
    assert "SHARD_DEPTH" in errors(report)
    assert "MTP_IN_TARGET_INDEX" not in errors(report)
    assert report["draft"]["target_loader_action"] == "skip"


def test_source_v1_detected_without_claiming_convertible(tmp_path):
    root = tmp_path / "v1"
    root.mkdir()
    write_json(root / "config.json", {"model_type": "deepseek_v4"})
    write_json(root / "quant.json", {"format": inspect.FORMAT, "format_version": 1})
    write_safetensors(
        root / "old.safetensors", {"model.layers.0.ffn.experts.w13_vq_perm": ("I32", [1, 2], struct.pack("<2i", 0, 1))}
    )
    _, report = run(root, tmp_path)
    assert report["classification"] == "source_v1_candidate"
    assert "--dry-run" in report["recommendations"][0]
    assert "V3_CONFIG" not in errors(report)


def test_unused_old_bank_does_not_reclassify_v3(checkpoint, tmp_path):
    root, _ = checkpoint
    write_safetensors(
        root / "old.safetensors",
        {
            "model.layers.0.ffn.experts.w13_vq_perm": ("I32", [1, 2], struct.pack("<2i", 0, 1)),
        },
    )
    _, report = run(root, tmp_path)
    assert report["classification"] == "source_v3_candidate"
    assert not errors(report)


def test_conflicting_marker_requires_review(checkpoint, tmp_path):
    root, _ = checkpoint
    write_json(root / "quant.json", {"format": inspect.FORMAT, "format_version": 1})
    _, report = run(root, tmp_path)
    assert report["classification"] == "mixed_or_conflicting"
    assert {"FORMAT_CONFLICT", "V3_MARKER"} <= errors(report)


def test_external_shard_is_read_but_runtime_rejection_reported(checkpoint, tmp_path):
    root, _ = checkpoint
    source = root / "rest/body.safetensors"
    external = tmp_path / "body.safetensors"
    source.rename(external)
    source.symlink_to(external)
    _, report = run(root, tmp_path)
    assert "EXTERNAL_V3_SHARD" in errors(report)
    assert any(s["path"] == "rest/body.safetensors" for s in report["shards"])


def test_sample_limit_is_reported_as_unverified_not_corrupt(checkpoint, tmp_path, monkeypatch):
    root, _ = checkpoint
    monkeypatch.setattr(inspect, "SAMPLE_BYTE_LIMIT", 1)
    _, report = run(root, tmp_path, samples=8)
    assert not errors(report)
    assert "SAMPLE_SKIPPED" in report["findings"]
    assert all(s["status"] == "skipped" for s in report["samples"])


def test_output_cannot_modify_target_or_symlinked_draft(checkpoint, tmp_path):
    root, _ = checkpoint
    with pytest.raises(ValueError, match="outside"):
        inspect.inspect_model(root, root / "report")
    assert not (root / "report").exists()
    draft = tmp_path / "external-draft"
    draft.mkdir()
    (root / "dspark").symlink_to(draft, target_is_directory=True)
    with pytest.raises(ValueError, match="outside"):
        inspect.inspect_model(root, draft / "report")


def test_inspector_with_contract_files_needs_no_installed_packages(checkpoint, tmp_path):
    root, _ = checkpoint
    # The inspector shares the runtime contract instead of copying its rules.
    repo = Path(inspect.__file__).resolve().parents[1]
    isolated = tmp_path / "isolated"
    for relative in (
        "tools/inspect_vq2a8_checkpoint.py",
        "tools/vq2a8_source_contract.py",
        "vllm_ascend/quantization/methods/w2a8/vq2a8_gs32_contract.py",
    ):
        target = isolated / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / relative, target)
    script = isolated / "tools/inspect_vq2a8_checkpoint.py"
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            str(script),
            "--model",
            str(root),
            "--output-dir",
            str(tmp_path / "reports"),
            "--sample-values",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "CHECKPOINT_INSPECT_STATUS=collected" in result.stdout
    assert "REPORT_BUNDLE=" in result.stdout


def test_direct_repo_command_does_not_import_tools_bisect(checkpoint, tmp_path):
    root, _ = checkpoint
    result = subprocess.run(
        [sys.executable, inspect.__file__, "--model", str(root), "--output-dir", str(tmp_path / "reports")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "CHECKPOINT_INSPECT_STATUS=collected" in result.stdout


def test_new_export_accepts_three_field_mixed_in_current_loader(checkpoint, tmp_path):
    root, _ = checkpoint
    _, legacy = run(root, tmp_path)
    assert not errors(legacy)
    assert legacy["expert_checks"]["projection_counts"]["vq2_natural_columns"] == 2
    directory, result = inspect.inspect_model(
        root, tmp_path / "reports", sample_count=8, contract=inspect.EXPORT_CONTRACT
    )
    report = result["checkpoints"][0]
    assert not errors(report)
    assert report["expert_checks"]["projection_pairs"] == {"vq2/mxfp4": 1, "mxfp4/vq2": 1}
    assert report["expert_checks"]["by_projection"] == {
        "w13": {"vq2": 1, "mxfp4": 1},
        "w2": {"mxfp4": 1, "vq2": 1},
    }
    assert report["expert_checks"]["perm_tensors"] == 0
    assert report["expert_checks"]["sample_coverage"]["codebooks"]["passed"] == 2
    assert not report["runtime_compatibility"]["findings"]
    assert report["runtime_compatibility"]["status"] == "not_verified"
    assert report["export_contract"]["model_executed"] is False
    assert report["export_contract"]["matches_spec_geometry"] is False
    assert (directory / "compact.json").is_file()


@pytest.mark.parametrize("contract", ["current-loader", inspect.EXPORT_CONTRACT])
@pytest.mark.parametrize("identity", [True, False])
def test_perm_is_rejected_in_every_contract(checkpoint, tmp_path, contract, identity):
    root, tensors = checkpoint
    name = "layers.0.ffn.experts.0.w13.perm"
    values = list(range(64)) if identity else list(reversed(range(64)))
    tensors[name] = "I32", [64], struct.pack("<64i", *values)
    write_safetensors(root / "vq2_mixed_gs32/layer_00.safetensors", tensors)
    index = json.loads((root / "model.safetensors.index.json").read_text())
    index["weight_map"][name] = "vq2_mixed_gs32/layer_00.safetensors"
    write_json(root / "model.safetensors.index.json", index)
    _, result = inspect.inspect_model(root, tmp_path / "reports", contract=contract)
    report = result["checkpoints"][0]
    assert {"PERM_UNSUPPORTED", "EXPERT_FIELDS"} <= errors(report)


def test_natural_columns_require_hash_layer_coverage(checkpoint, tmp_path):
    root, tensors = checkpoint
    tensors = {name: value for name, value in tensors.items() if ".experts.1." not in name}
    write_safetensors(root / "vq2_mixed_gs32/layer_00.safetensors", tensors)
    index = json.loads((root / "model.safetensors.index.json").read_text())
    index["weight_map"] = {name: shard for name, shard in index["weight_map"].items() if ".experts.1." not in name}
    write_json(root / "model.safetensors.index.json", index)
    config = json.loads((root / "config.json").read_text())
    config["num_hash_layers"] = 1
    write_json(root / "config.json", config)
    _, result = inspect.inspect_model(root, tmp_path / "reports", contract=inspect.EXPORT_CONTRACT)
    assert "EXPERT_INVENTORY" in errors(result["checkpoints"][0])
    _, current = run(root, tmp_path)
    assert "EXPERT_INVENTORY" in errors(current)


def test_new_export_compact_retains_layouts_and_reports_only_current_blockers(checkpoint, tmp_path):
    root, _ = checkpoint
    extra = {f"layers.0.extra{i}.weight": ("BF16", [64], bytes(128)) for i in range(16)}
    extra["mtp.0.attn_norm.weight"] = ("BF16", [64], bytes(128))
    write_safetensors(root / "extra.safetensors", extra)
    index = json.loads((root / "model.safetensors.index.json").read_text())
    index["weight_map"].update(dict.fromkeys(extra, "extra.safetensors"))
    write_json(root / "model.safetensors.index.json", index)
    config = json.loads((root / "config.json").read_text())
    config["quantization_config"]["export_note"] = "keep this metadata"
    write_json(root / "config.json", config)
    directory, result = inspect.inspect_model(root, tmp_path / "reports", contract=inspect.EXPORT_CONTRACT)
    full = result["checkpoints"][0]
    assert not errors(full)
    compact = json.loads((directory / "compact.json").read_text())["checkpoints"][0]
    blockers = compact["runtime_compatibility"]["findings"]
    assert {"V3_ROOT_NAME", "SHARD_DEPTH", "V3_CONFIG"} <= set(blockers)
    assert "MTP_IN_TARGET_INDEX" not in blockers
    assert len(blockers["V3_ROOT_NAME"]["examples"]) == 2
    assert len(compact["root_layouts"]) == len(full["root_layouts"]) == 17
    assert len(compact["mtp_layouts"]) == 1
    assert compact["config"] == config
    assert not {"layers", "shards", "samples"} & compact.keys()


@pytest.mark.parametrize("contract", ["current-loader", inspect.EXPORT_CONTRACT])
def test_original_roots_and_retained_dspark_are_not_target_blockers(checkpoint, tmp_path, contract):
    root, _ = checkpoint
    original = {
        "embed.weight": ("BF16", [2, 64], bytes(256)),
        "norm.weight": ("BF16", [64], bytes(128)),
        "hc_head_fn": ("F32", [4, 256], bytes(4096)),
        "layers.0.attn.wq_a.weight": ("F8_E4M3", [64, 64], bytes(4096)),
        "layers.0.attn.wq_a.scale": ("F8_E8M0", [1, 1], bytes([127])),
        "layers.0.ffn.shared_experts.w1.weight": ("F8_E4M3", [64, 64], bytes(4096)),
        "layers.0.ffn.shared_experts.w1.scale": ("F8_E8M0", [1, 1], bytes([127])),
        "mtp.0.ffn.experts.0.w1.weight": ("I8", [64, 32], bytes(2048)),
        "mtp.1.main_norm.weight": ("BF16", [64], bytes(128)),
        "mtp.2.norm.weight": ("BF16", [64], bytes(128)),
    }
    write_safetensors(root / "rest/body.safetensors", original)
    index = json.loads((root / "model.safetensors.index.json").read_text())
    del index["weight_map"]["model.norm.weight"]
    index["weight_map"].update(dict.fromkeys(original, "rest/body.safetensors"))
    write_json(root / "model.safetensors.index.json", index)
    config = json.loads((root / "config.json").read_text())
    config.update(dspark_block_size=5, num_nextn_predict_layers=1, expert_dtype="fp4")
    write_json(root / "config.json", config)
    _, result = inspect.inspect_model(root, tmp_path / "reports", contract=contract)
    report = result["checkpoints"][0]
    assert not errors(report)
    assert not report.get("runtime_compatibility", {}).get("findings")
    assert report["mtp_tensor_count"] == 3
    assert report["draft"]["target_loader_action"] == "skip"
    assert report["draft"]["stage_tensor_counts"] == {"0": 1, "1": 1, "2": 1}


@pytest.mark.parametrize("name", ["mtp.01.norm.weight", "mtp.-1.norm.weight", "mtp.0..weight"])
def test_current_loader_rejects_malformed_mtp_names(checkpoint, tmp_path, name):
    root, _ = checkpoint
    root_tensors = {"model.norm.weight": ("BF16", [64], bytes(128)), name: ("BF16", [64], bytes(128))}
    write_safetensors(root / "rest/body.safetensors", root_tensors)
    index = json.loads((root / "model.safetensors.index.json").read_text())
    index["weight_map"][name] = "rest/body.safetensors"
    write_json(root / "model.safetensors.index.json", index)
    _, report = run(root, tmp_path)
    assert "MTP_NAME" in errors(report)


def test_new_export_still_reports_index_errors_and_cli_compact_path(checkpoint, tmp_path):
    root, _ = checkpoint
    index = json.loads((root / "model.safetensors.index.json").read_text())
    index["weight_map"]["layers.0.ffn.experts.0.w13.absent"] = "rest/body.safetensors"
    write_json(root / "model.safetensors.index.json", index)
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            inspect.__file__,
            "--model",
            str(root),
            "--contract",
            inspect.EXPORT_CONTRACT,
            "--output-dir",
            str(tmp_path / "reports"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "INDEX_TENSOR_MISSING" in result.stdout
    assert "CHECKPOINT_INSPECT_STATUS=issues_found" in result.stdout
    path = next(
        line.removeprefix("COMPACT_REPORT=")
        for line in result.stdout.splitlines()
        if line.startswith("COMPACT_REPORT=")
    )
    assert Path(path).is_absolute() and Path(path).is_file()


@pytest.mark.parametrize("contract", ["current-loader", inspect.EXPORT_CONTRACT])
def test_checkpoint_contract_rejects_noninteger_version(checkpoint, tmp_path, contract):
    root, _ = checkpoint
    config = json.loads((root / "config.json").read_text())
    config["quantization_config"]["format_version"] = 3.0
    write_json(root / "config.json", config)
    _, result = inspect.inspect_model(root, tmp_path / "reports", contract=contract)
    assert "V3_CONFIG" in errors(result["checkpoints"][0])
