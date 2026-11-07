# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.ut.tools.test_convert_vq2_mixed_gs32_v3 import snapshot, write_json, write_safetensors
from tools import inspect_vq2a8_transforms as inspect


def save_model(root, tensors, rotation=None):
    root.mkdir()
    write_json(
        root / "config.json",
        {"hidden_size": 4, "num_hidden_layers": 2, "quantization_config": {"output_head_mode": "norm_fused"}},
    )
    fields = {name: ("F32", list(value.shape), value.astype("<f4").tobytes()) for name, value in tensors.items()}
    write_safetensors(root / "model.safetensors", fields, metadata={"rotation_mode": "folded"})
    write_json(root / "model.safetensors.index.json", {"weight_map": dict.fromkeys(fields, "model.safetensors")})
    if rotation is not None:
        (root / "optional").mkdir()
        write_safetensors(
            root / "optional/quarot.safetensors",
            {"global_rotation": ("F32", list(rotation.shape), rotation.astype("<f4").tobytes())},
        )
    return root


@pytest.fixture
def models(tmp_path):
    rotation = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]], dtype=np.float32)
    head = np.arange(32, dtype=np.float32).reshape(8, 4)
    gamma = np.array([1, 2, 3, 4], dtype=np.float32)
    source = {
        "norm.weight": gamma,
        "head.weight": head,
        "embed.weight": head + 3,
        "hc_head_fn": np.arange(32, dtype=np.float32).reshape(2, 16),
    }
    target = {
        "norm.weight": np.ones(4, dtype=np.float32),
        "head.weight": (head * gamma) @ rotation,
        "embed.weight": (head + 3) @ rotation,
        "hc_head_fn": (source["hc_head_fn"].reshape(-1, 4) @ rotation).reshape(2, 16),
        "mtp.0.norm.weight": np.ones(4, dtype=np.float32),
        "mtp.0.head.weight": head.copy(),
        "mtp.0.e_proj.weight": np.eye(4, dtype=np.float32),
    }
    original = save_model(tmp_path / "original", source)
    exported = save_model(tmp_path / "exported", target, rotation)
    return exported, original


def test_read_only_without_original_reports_measurements_and_unknown_semantics(models, tmp_path):
    exported, original = models
    declarations = {"scheme": "folded", "source_commit": "abc123", "bproj_shift": 16, "bproj_rot_step": 1}
    write_json(exported / "rotation_config.json", declarations)
    before = snapshot(exported)
    run, report = inspect.collect(exported, tmp_path / "reports")
    assert snapshot(exported) == before
    assert report["status"] == "collected"
    assert report["reference_model"] is None
    assert all(item["values"]["all_one"] for item in report["final_norms"])
    assert report["rotation"]["status"] == "measured"
    assert report["rotation"]["candidates"][0]["indexed"] is False
    assert report["rotation"]["probes"]["row_gram_max_abs_error"] == 0
    assert report["rotation"]["probes"]["sampled_symmetry_max_abs_error"] == 2
    assert report["mtp"]["stage_counts"] == {"0": 3}
    assert report["shard_declarations"][0]["key"] == "rotation_mode"
    assert report["exporter_questions"]
    rotation_metadata = next(item for item in report["metadata"] if item["file"] == "rotation_config.json")
    assert {item["key"]: item["value"] for item in rotation_metadata["declarations"]} == declarations
    assert json.loads((run / "summary.json").read_text())["status"] == "collected"


def test_optional_reference_distinguishes_rotation_direction_and_norm_fusion(models, tmp_path):
    exported, original = models
    _, report = inspect.collect(exported, tmp_path / "reports", original)
    head = next(item for item in report["heads"] if item["name"] == "head.weight")
    errors = head["reference"]["candidate_errors"]
    assert errors["(W * original_norm_gamma) @ R (per hidden-sized block)"]["relative_l2"] == 0
    assert errors["(W * original_norm_gamma) @ R.T (per hidden-sized block)"]["relative_l2"] > 0
    assert errors["W @ R (per hidden-sized block)"]["relative_l2"] > 0
    hc = next(item for item in report["heads"] if item["name"] == "hc_head_fn")
    assert hc["reference"]["candidate_errors"]["W @ R (per hidden-sized block)"]["relative_l2"] == 0
    mtp = next(item for item in report["heads"] if item["name"] == "mtp.0.head.weight")
    assert mtp["reference"]["status"] == "name_absent_in_reference"


def test_large_tensor_reads_only_selected_rows_and_names_hash_scope(models):
    exported, _ = models
    checkpoint = inspect.Checkpoint(exported)
    record, values = checkpoint.read("head.weight", full_limit=64)
    assert record["scope"] == "sampled_rows" and record["rows"] == [0, 2, 4, 7]
    assert record["bytes_read"] == 64 and values.shape == (4, 4)
    _, all_values = checkpoint.read("head.weight")
    np.testing.assert_array_equal(values, all_values[record["rows"]])


@pytest.mark.parametrize("dtype", ["BF16", "F32"])
def test_norm_nan_does_not_become_all_one_or_invalid_json(dtype):
    raw = (
        np.array([0x3F80, 0x7FC0], dtype="<u2").tobytes()
        if dtype == "BF16"
        else np.array([1, np.nan], dtype="<f4").tobytes()
    )
    result = inspect.value_stats(inspect.decode(raw, dtype, [2]))
    assert result["non_finite"] == 1 and not result["all_one"]
    json.dumps(result, allow_nan=False)


def test_missing_or_ambiguous_rotation_is_not_guessed(models, tmp_path):
    exported, _ = models
    checkpoint = inspect.Checkpoint(exported)
    report, rotation = inspect.rotation_evidence(checkpoint, "unknown", 4)
    assert report["status"] == "missing" and rotation is None
    (exported / "optional/rotation-copy.safetensors").write_bytes(
        (exported / "optional/quarot.safetensors").read_bytes()
    )
    report, rotation = inspect.rotation_evidence(checkpoint, "global_rotation", 4)
    assert report["status"] == "ambiguous" and rotation is None


def test_zero_target_has_undefined_relative_error_but_valid_report():
    error = inspect.relative_error(np.zeros((2, 2)), np.ones((2, 2)))
    assert error["relative_l2"] is None and error["max_abs"] == 1
    json.dumps(error, allow_nan=False)


def test_refuses_reports_inside_model_and_detects_changed_shard(models):
    exported, _ = models
    with pytest.raises(ValueError, match="outside checkpoint"):
        inspect.collect(exported, exported / "reports")
    checkpoint = inspect.Checkpoint(exported)
    checkpoint.header("model.safetensors")
    path = exported / "model.safetensors"
    with path.open("ab") as stream:
        stream.write(b"x")
    with pytest.raises(ValueError, match="Shard changed"):
        checkpoint.read("head.weight")


def test_direct_cli_needs_no_vllm_or_npu(models, tmp_path):
    exported, _ = models
    child = subprocess.run(
        [
            sys.executable,
            str(Path(inspect.__file__)),
            "--model",
            str(exported),
            "--output-dir",
            str(tmp_path / "reports"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert child.returncode == 0, child.stderr
    assert "SUMMARY_JSON=" in child.stdout
