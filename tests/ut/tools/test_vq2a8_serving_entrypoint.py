# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check the serving runner selection without importing torch, vLLM or NPU extensions."""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from tools import build_vq2a8_gs32_stage as build
from tools import serve_vq2a8 as serve
from tools.vq2a8_source_contract import source_contract


@pytest.fixture
def server_assets(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {"quantization_config": {"quant_method": "vq2a8", "format": "vq2_mixed_gs32_source", "format_version": 3}}
        ),
        encoding="utf-8",
    )
    (tmp_path / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    library, binary = tmp_path / "stage.so", tmp_path / "kernel.o"
    library.touch()
    binary.touch()
    return ["--model", str(tmp_path), "--gs32-library", str(library), "--gs32-binary", str(binary)]


@pytest.fixture
def dspark_assets(tmp_path):
    model = tmp_path / "original"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {"model_type": "deepseek_v4", "dspark_block_size": 5, "quantization_config": {"quant_method": "fp8"}}
        ),
        encoding="utf-8",
    )
    (model / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    return model


@pytest.mark.parametrize("execution", ["fused", "full_graph"])
def test_reference_launch_needs_no_gs32_artifacts_and_preserves_tp4_b64(server_assets, execution):
    args = serve.parse_args(
        server_assets[:2]
        + [
            "--gs32-backend",
            "reference_mxfp8",
            "--gs32-execution",
            execution,
            "--ep-size",
            "4",
            "--max-num-seqs",
            "64",
            "--max-num-batched-tokens",
            "8192",
            "--max-model-len",
            "8192",
            "--no-dspark",
        ]
    )
    command = serve.build_command(args)
    options = json.loads(command[command.index("--additional-config") + 1])["vq2a8"]
    assert options == {"gs32_backend": "reference_mxfp8", "gs32_execution": execution, "gs32_fusion": "none"}
    assert command[command.index("--tensor-parallel-size") + 1] == "4"
    assert command[command.index("--max-num-seqs") + 1] == "64"
    assert command[command.index("--max-num-batched-tokens") + 1] == "8192"
    assert ("--enforce-eager" in command) == (execution == "fused")
    assert "--speculative-config" not in command


@pytest.mark.parametrize("execution,fusion", [("serial", "none"), ("batched", "none"), ("full_graph", "swiglu_quant")])
def test_reference_launch_rejects_unsupported_mode_before_loading(server_assets, execution, fusion):
    args = serve.parse_args(
        server_assets[:2]
        + ["--gs32-backend", "reference_mxfp8", "--gs32-execution", execution, "--gs32-fusion", fusion]
    )
    with pytest.raises(ValueError, match="reference_mxfp8 requires"):
        serve.build_command(args)


def test_native_launch_still_requires_native_artifacts(server_assets):
    with pytest.raises(ValueError, match="requires --gs32-library and --gs32-binary"):
        serve.build_command(serve.parse_args(server_assets[:2]))


@pytest.mark.parametrize("valid", [False, True])
def test_runtime_record_is_written_only_after_preflight_with_resolved_defaults(
    server_assets, monkeypatch, tmp_path, valid
):
    record_path = tmp_path / "service-runtime.json"
    artifacts = {"library_sha256": "1" * 64, "kernel": {"sha256": "2" * 64, "abi_version": 3}}
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setattr(serve, "check_native_dependencies", lambda: "model.so")

    def preflight(args):
        if not valid:
            raise ValueError("stale artifact pair")
        return artifacts

    monkeypatch.setattr(serve, "check_gs32_dependencies", preflight)
    result = serve.main(server_assets + ["--runtime-record", str(record_path), "--check-native-only"])
    assert result == (0 if valid else 1)
    assert record_path.exists() == valid
    if valid:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        assert record["gs32_fusion"] == "full_ffn" and record["gs32_execution"] == "fused"
        assert record["gs32_backend"] == "native" and record["model_path"] == str(tmp_path.resolve())
        assert record["native_artifacts"] == artifacts
        assert record["device_execution_verified"] is False


@pytest.mark.parametrize("fault", [None, "inverse", "bool_type", "unknown", "dspark", "sidecar"])
def test_folded_profile_preflight_preserves_model_and_rejects_unsupported_semantics(
    server_assets, dspark_assets, fault
):
    contract = source_contract()
    quant = dict((*contract["SOURCE_V3_CONFIG"], *contract["FOLDED_HEAD_CONFIG"]))
    if fault == "inverse":
        quant["inverse_rotation_before_final_norm"] = True
    if fault == "bool_type":
        quant["vq_input_permutation"] = 0
    if fault == "unknown":
        quant["online_rotation"] = True
    root = Path(server_assets[1])
    (root / "config.json").write_text(json.dumps({"quantization_config": quant}), encoding="utf-8")
    if fault == "sidecar":
        rotation = dict(contract["FOLDED_ROTATION_CONFIG"])
        rotation["main_mtp_final_norm"] = "unfused"
        (root / "rotation_config.json").write_text(json.dumps(rotation), encoding="utf-8")
    extra = ["--dspark-model", str(dspark_assets)] if fault == "dspark" else ["--no-dspark"]
    args = serve.parse_args(server_assets + extra)
    if fault:
        with pytest.raises(ValueError, match="source-v3|rotation_config"):
            serve.build_command(args)
    else:
        command = serve.build_command(args)
        assert command[command.index("serve") + 1] == str(root.resolve())
        assert "--speculative-config" not in command
        assert json.loads((root / "config.json").read_text())["quantization_config"] == quant


@pytest.mark.parametrize("runner", [None, "1", "0"])
def test_launcher_defaults_to_v2_and_preserves_explicit_selection(server_assets, monkeypatch, capsys, runner):
    if runner is None:
        monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
    else:
        monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", runner)
    expected = "1" if runner is None else runner

    def check_native():
        assert os.environ["VLLM_USE_V2_MODEL_RUNNER"] == expected
        return "fake-native-extension"

    monkeypatch.setattr(serve, "check_native_dependencies", check_native)
    monkeypatch.setattr(serve, "check_gs32_dependencies", lambda args: str(args.gs32_library))
    assert serve.main(server_assets + ["--check-native-only"]) == 0
    assert "VQ2A8_NATIVE_CHECK=PASS" in capsys.readouterr().out
    assert serve.main(server_assets + ["--dry-run"]) == 0
    assert shlex.split(capsys.readouterr().out)[:2] == ["env", f"VLLM_USE_V2_MODEL_RUNNER={expected}"]


@pytest.mark.parametrize("execution", ["serial", "batched", "fused", "full_graph"])
@pytest.mark.parametrize("check_only", [False, True])
def test_launcher_checks_selected_bridge_before_serving(server_assets, monkeypatch, capsys, execution, check_only):
    calls = []
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setattr(serve, "check_native_dependencies", lambda: calls.append("model") or "model.so")
    monkeypatch.setattr(build, "verify_loaded_library", lambda path, mode, **kwargs: calls.append((path, mode)))
    monkeypatch.setattr(build, "verify_runtime_artifacts", lambda path, binary, **kwargs: {"library_path": str(path)})
    monkeypatch.setattr(serve.os, "execv", lambda *args: calls.append("serve"))
    flags = ["--gs32-execution", execution, "--gs32-fusion", "none"]
    if check_only:
        flags.append("--check-native-only")
    assert serve.main(server_assets + flags) == 0
    assert calls[:2] == ["model", (Path(server_assets[3]).resolve(), execution)]
    if check_only:
        assert len(calls) == 2
        output = capsys.readouterr().out
        assert "VQ2A8_NATIVE_CHECK=PASS" in output and "gs32_backend=native" in output
        record = json.loads(
            next(line.split("=", 1)[1] for line in output.splitlines() if line.startswith("VQ2A8_RUNTIME="))
        )
        assert record["native_artifacts"]["library_path"] == str(Path(server_assets[3]).resolve())
    else:
        assert calls[2:] == ["serve"]


@pytest.mark.parametrize("check_only", [False, True])
def test_launcher_rejects_stale_bridge_before_loading_weights(server_assets, monkeypatch, capsys, check_only):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setattr(serve, "check_native_dependencies", lambda: "model.so")
    monkeypatch.setattr(serve.os, "execv", lambda *args: pytest.fail("Serving must not start"))

    def stale_bridge(path, execution, **kwargs):
        raise RuntimeError("GS32 schema mismatch for prepare_device_routes: rebuild the bridge .so")

    monkeypatch.setattr(build, "verify_loaded_library", stale_bridge)
    flags = ["--gs32-execution", "full_graph"] + (["--check-native-only"] if check_only else [])
    assert serve.main(server_assets + flags) == 1
    captured = capsys.readouterr()
    assert "GS32 schema mismatch" in captured.err and "PASS" not in captured.out


def test_reference_preflight_checks_model_extension_without_loading_bridge(server_assets, monkeypatch, capsys):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setattr(serve, "check_native_dependencies", lambda: "model.so")
    monkeypatch.setattr(build, "verify_loaded_library", lambda *args: pytest.fail("Reference must not load GS32"))
    assert (
        serve.main(
            server_assets[:2]
            + ["--gs32-backend", "reference_mxfp8", "--gs32-execution", "fused", "--check-native-only"]
        )
        == 0
    )
    assert "gs32_bridge=not_required" in capsys.readouterr().out


def test_dry_run_does_not_import_torch_or_npu(server_assets):
    source = """
import builtins, sys
original_import = builtins.__import__
def checked_import(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'torch_npu', 'vllm', 'vllm_ascend'):
        raise AssertionError('Unexpected runtime import: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = checked_import
from tools.serve_vq2a8 import main
raise SystemExit(main(sys.argv[1:]))
"""
    result = subprocess.run(
        [sys.executable, "-c", source, *server_assets, "--dry-run"],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--quantization vq2a8" in result.stdout


@pytest.mark.parametrize("runner", ["", "invalid"])
@pytest.mark.parametrize("mode", [[], ["--dry-run"], ["--check-native-only"]])
def test_launcher_rejects_invalid_runner_before_native_loading(server_assets, monkeypatch, capsys, runner, mode):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", runner)
    monkeypatch.setattr(serve, "check_native_dependencies", lambda: pytest.fail("Native loading must not start"))
    monkeypatch.setattr(serve.os, "execv", lambda *args: pytest.fail("Serving must not start"))
    assert serve.main(server_assets + mode) == 1
    assert os.environ["VLLM_USE_V2_MODEL_RUNNER"] == runner
    captured = capsys.readouterr()
    assert "VLLM_USE_V2_MODEL_RUNNER must be 0 (MRV1) or 1 (MRV2)" in captured.err
    assert not captured.out


@pytest.mark.parametrize("batch", [2, 64])
def test_launcher_rejects_multisequence_mrv1_graph_before_native_loading(server_assets, monkeypatch, capsys, batch):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    monkeypatch.setattr(serve, "check_native_dependencies", lambda: pytest.fail("Native loading must not start"))
    assert serve.main(server_assets + ["--gs32-execution", "full_graph", "--max-num-seqs", str(batch)]) == 1
    assert "requires MRV2" in capsys.readouterr().err


def test_launcher_defaults_multisequence_graphs_to_mrv2(server_assets, monkeypatch, capsys):
    monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
    assert serve.main(server_assets + ["--gs32-execution", "full_graph", "--max-num-seqs", "64", "--dry-run"]) == 0
    command = shlex.split(capsys.readouterr().out)
    assert command[:2] == ["env", "VLLM_USE_V2_MODEL_RUNNER=1"]
    assert command[command.index("--max-num-seqs") + 1] == "64"


@pytest.mark.parametrize("width", [2, 4, 8])
@pytest.mark.parametrize("execution", ["serial", "fused", "full_graph"])
def test_dp_ep_launches_distinct_engines_with_whole_experts(server_assets, width, execution):
    args = serve.parse_args(
        server_assets
        + [
            "--parallel-mode",
            "dp-ep",
            "--ep-size",
            str(width),
            "--gs32-execution",
            execution,
            "--max-num-seqs",
            "16",
            "--max-num-batched-tokens",
            "2048",
            "--gs32-fusion",
            "none",
        ]
    )
    command = serve.build_command(args)
    expected = {
        "--tensor-parallel-size": "1",
        "--data-parallel-size": str(width),
        "--data-parallel-size-local": str(width),
        "--data-parallel-backend": "mp",
        "--pipeline-parallel-size": "1",
        "--distributed-executor-backend": "mp",
        "--api-server-count": "1",
        "--max-num-seqs": "16",
        "--max-num-batched-tokens": "2048",
        "--expert-placement-strategy": "linear",
    }
    for option, value in expected.items():
        assert command.count(option) == 1
        assert command[command.index(option) + 1] == value
    assert "--enable-expert-parallel" in command
    if execution == "full_graph":
        compilation = json.loads(command[command.index("--compilation-config") + 1])
        assert compilation["cudagraph_capture_sizes"] == [1, 2, 4, 8, 16]
        assert compilation["max_cudagraph_capture_size"] == 16


@pytest.mark.parametrize("parallel_mode", ["tp-ep", "dp-ep"])
def test_parallel_mode_preserves_explicit_and_default_per_engine_batch(server_assets, parallel_mode):
    args = serve.parse_args(server_assets + ["--parallel-mode", parallel_mode, "--ep-size", "4"])
    command = serve.build_command(args)
    assert command[command.index("--max-num-seqs") + 1] == "1"
    if parallel_mode == "tp-ep":
        assert command[command.index("--tensor-parallel-size") + 1] == "4"
        assert command[command.index("--data-parallel-size") + 1] == "1"


def test_dp_ep_rejects_single_rank_before_native_loading(server_assets, monkeypatch, capsys):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setattr(serve, "check_native_dependencies", lambda: pytest.fail("Native loading must not start"))
    assert serve.main(server_assets + ["--parallel-mode", "dp-ep", "--ep-size", "1"]) == 1
    assert "dp-ep requires --ep-size 2, 4 or 8" in capsys.readouterr().err


@pytest.mark.parametrize("execution", ["serial", "batched", "fused", "full_graph"])
@pytest.mark.parametrize("width", [4, 8])
def test_dp_ep_rejects_mrv1_even_for_batch_one(server_assets, monkeypatch, capsys, execution, width):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    monkeypatch.setattr(serve, "check_native_dependencies", lambda: pytest.fail("Native loading must not start"))
    assert (
        serve.main(server_assets + ["--parallel-mode", "dp-ep", "--ep-size", str(width), "--gs32-execution", execution])
        == 1
    )
    assert "dp-ep requires MRV2" in capsys.readouterr().err


@pytest.mark.parametrize("parallel_mode", ["tp-ep", "dp-ep"])
def test_dspark_keeps_target_batches_and_expands_verification_graph_tokens(server_assets, dspark_assets, parallel_mode):
    args = serve.parse_args(
        server_assets
        + [
            "--dspark-model",
            str(dspark_assets),
            "--parallel-mode",
            parallel_mode,
            "--ep-size",
            "4",
            "--gs32-execution",
            "full_graph",
            "--max-num-seqs",
            "16",
        ]
    )
    command = serve.build_command(args)
    speculation = json.loads(command[command.index("--speculative-config") + 1])
    assert speculation == {
        "method": "dspark",
        "model": str(dspark_assets.resolve()),
        "num_speculative_tokens": 5,
        "enforce_eager": False,
    }
    assert command[command.index("--quantization") + 1] == "vq2a8"
    assert command[command.index("--max-num-seqs") + 1] == "16"
    compilation = json.loads(command[command.index("--compilation-config") + 1])
    assert compilation["mode"] == 0
    assert compilation["cudagraph_mode"] == "FULL_DECODE_ONLY"
    assert compilation["cudagraph_capture_sizes"] == [6, 12, 24, 48, 96]
    assert compilation["max_cudagraph_capture_size"] == 96
    assert "--enforce-eager" not in command


def test_dspark_default_budget_holds_full_verification_batch(server_assets, dspark_assets):
    args = serve.parse_args(server_assets + ["--dspark-model", str(dspark_assets), "--max-num-seqs", "64"])
    command = serve.build_command(args)
    assert command[command.index("--max-num-batched-tokens") + 1] == "384"


@pytest.mark.parametrize("tokens", [0, 1, 4, 6, 10])
def test_dspark_rejects_unvalidated_proposal_count(server_assets, dspark_assets, tokens):
    args = serve.parse_args(
        server_assets + ["--dspark-model", str(dspark_assets), "--num-speculative-tokens", str(tokens)]
    )
    with pytest.raises(ValueError, match="requires --num-speculative-tokens 5"):
        serve.build_command(args)


def test_dspark_requires_explicit_enable_or_checkpoint(server_assets):
    with pytest.raises(ValueError, match="requires --dspark-model"):
        serve.build_command(serve.parse_args(server_assets + ["--num-speculative-tokens", "5"]))


@pytest.mark.parametrize("fault", ["quantization", "model_type", "block_size", "weight_index"])
def test_dspark_rejects_wrong_draft_assets(server_assets, dspark_assets, fault):
    path = dspark_assets / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    if fault == "quantization":
        config["quantization_config"]["quant_method"] = "vq2a8"
    elif fault == "model_type":
        config["model_type"] = "qwen3"
    elif fault == "block_size":
        config["dspark_block_size"] = 6
    else:
        (dspark_assets / "model.safetensors.index.json").unlink()
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="local original DeepSeek-V4 FP8 checkpoint"):
        serve.build_command(serve.parse_args(server_assets + ["--dspark-model", str(dspark_assets)]))


def test_dspark_checks_token_budget_before_native_loading(server_assets, dspark_assets):
    args = serve.parse_args(
        server_assets + ["--dspark-model", str(dspark_assets), "--max-num-seqs", "16", "--max-num-batched-tokens", "95"]
    )
    with pytest.raises(ValueError, match="num_speculative_tokens"):
        serve.build_command(args)


def test_dspark_rejects_mrv1_before_native_loading(server_assets, dspark_assets, monkeypatch, capsys):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    monkeypatch.setattr(serve, "check_native_dependencies", lambda: pytest.fail("Native loading must not start"))
    assert serve.main(server_assets + ["--dspark-model", str(dspark_assets), "--dry-run"]) == 1
    assert "DSpark requires MRV2" in capsys.readouterr().err


def test_dspark_packaged_draft_requires_opt_in_and_allows_explicit_override(server_assets, dspark_assets):
    packaged = dspark_assets.with_name("dspark")
    packaged.mkdir()
    for source in dspark_assets.iterdir():
        (packaged / source.name).write_bytes(source.read_bytes())
    assert "--speculative-config" not in serve.build_command(serve.parse_args(server_assets))
    for extra, expected in (
        (["--dspark"], packaged),
        (["--dspark", "--dspark-model", str(dspark_assets)], dspark_assets),
    ):
        command = serve.build_command(serve.parse_args(server_assets + extra))
        speculation = json.loads(command[command.index("--speculative-config") + 1])
        assert speculation["model"] == str(expected.resolve())
        assert speculation["num_speculative_tokens"] == 5
        assert speculation["enforce_eager"] is False


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--dspark"], "DSpark draft directory not found"),
        (["--no-dspark", "--num-speculative-tokens", "5"], "--no-dspark conflicts"),
    ],
)
def test_invalid_packaged_draft_selection_fails_before_native_loading(
    server_assets, monkeypatch, capsys, extra, message
):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setattr(serve, "check_native_dependencies", lambda: pytest.fail("Native loading must not start"))
    assert serve.main(server_assets + extra) == 1
    assert message in capsys.readouterr().err


def test_no_dspark_rejects_explicit_draft_instead_of_silently_enabling_it(server_assets, dspark_assets):
    with pytest.raises(ValueError, match="--no-dspark conflicts"):
        serve.build_command(serve.parse_args(server_assets + ["--no-dspark", "--dspark-model", str(dspark_assets)]))


@pytest.mark.parametrize("baseline", [False, True])
def test_dspark_performance_shell_uses_packaged_draft_and_matching_baseline(
    server_assets, dspark_assets, tmp_path, baseline
):
    dspark_assets.rename(tmp_path / "dspark")
    executable_dir = tmp_path / "bin"
    executable_dir.mkdir()
    executable = executable_dir / "python3"
    executable.write_text(
        f'#!/bin/sh\nprintf "DEVICES=%s\\n" "$ASCEND_RT_VISIBLE_DEVICES"\nexec {shlex.quote(sys.executable)} "$@"\n',
        encoding="utf-8",
    )
    executable.chmod(0o755)
    environment = dict(
        os.environ,
        PATH=f"{executable_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        ASCEND_RT_VISIBLE_DEVICES="4,5,6,7",
    )
    environment.pop("VLLM_USE_V2_MODEL_RUNNER", None)
    script = Path(__file__).resolve().parents[3] / "tools/tests/serve_vq2a8_dspark_performance.sh"
    command = ["bash", str(script), *server_assets, "--dry-run", "--port", "8103"]
    if baseline:
        command.append("--no-dspark")
    result = subprocess.run(
        command, cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=20, check=False
    )
    assert result.returncode == 0, result.stderr
    devices, invocation = result.stdout.splitlines()
    assert devices == "DEVICES=4,5,6,7"
    generated = shlex.split(invocation)
    assert generated[:2] == ["env", "VLLM_USE_V2_MODEL_RUNNER=1"]
    for option, value in {
        "--tensor-parallel-size": "1",
        "--data-parallel-size": "4",
        "--max-num-seqs": "4",
        "--max-model-len": "8192",
        "--max-num-batched-tokens": "8192",
        "--port": "8103",
    }.items():
        assert generated[generated.index(option) + 1] == value
    compilation = json.loads(generated[generated.index("--compilation-config") + 1])
    assert compilation["cudagraph_capture_sizes"] == ([1, 2, 4] if baseline else [6, 12, 24])
    assert ("--speculative-config" in generated) is not baseline
    if not baseline:
        speculation = json.loads(generated[generated.index("--speculative-config") + 1])
        assert speculation["model"] == str((tmp_path / "dspark").resolve())
    environment["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    rejected = subprocess.run(
        command, cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=20, check=False
    )
    assert rejected.returncode == 1
    assert "requires MRV2" in rejected.stderr
