# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate throughput reporting without a vLLM installation or network access."""

import csv
import importlib.util
import io
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def benchmark():
    path = REPO_ROOT / "tools/tests/run_vq2a8_throughput.py"
    spec = importlib.util.spec_from_file_location("vq2a8_throughput_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def small_args(benchmark, *extra):
    return benchmark.parse_args(
        [
            "--tokenizer",
            "local-tokenizer",
            "--concurrency",
            "1",
            "2",
            "--num-prompts",
            "4",
            "--input-len",
            "8",
            "--output-len",
            "4",
            "--repeats",
            "3",
            *extra,
        ]
    )


def valid_result(args, *, num_prompts=None, throughput=16.0, ttft=10.0, tpot=2.0):
    if num_prompts is None:
        num_prompts = args.num_prompts
    duration = num_prompts * args.output_len / throughput
    return {
        "completed": num_prompts,
        "failed": 0,
        "total_output_tokens": num_prompts * args.output_len,
        "output_lens": [args.output_len] * num_prompts,
        "input_lens": [args.input_len] * num_prompts,
        "errors": [""] * num_prompts,
        "duration": duration,
        "output_throughput": throughput,
        "request_throughput": num_prompts / duration,
        "mean_ttft_ms": ttft,
        "p99_ttft_ms": ttft,
        "mean_tpot_ms": tpot,
        "p99_tpot_ms": tpot,
        "p99_e2el_ms": 20.0,
    }


def valid_run(benchmark, args, concurrency, repeat, **metrics):
    return {
        "concurrency": concurrency,
        "repeat": repeat,
        "valid": True,
        **benchmark.validate_result(valid_result(args, **metrics), args, concurrency),
    }


def startup_record():
    return {
        "schema_version": 1,
        "status": "preflight_passed",
        "model_path": "/models/noperm-export",
        "gs32_backend": "native",
        "gs32_fusion": "full_ffn",
        "gs32_execution": "full_graph",
        "native_artifacts": {"tiling_abi_version": 3},
        "device_execution_verified": False,
    }


def test_runtime_record_matches_live_root_and_preserves_execution_scope(benchmark, tmp_path, monkeypatch):
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(startup_record()), encoding="utf-8")
    snapshot = {"data": [{"id": "vq2a8", "root": "/models/noperm-export"}]}
    fetch = Mock(return_value=io.StringIO(json.dumps(snapshot)))
    monkeypatch.setattr(benchmark, "urlopen", fetch)
    args = small_args(benchmark, "--runtime-record", str(path))
    context = benchmark.runtime_context(args)
    report = benchmark.attach_runtime_context(benchmark.summarize(args, []), context)
    assert report["model_path"] == "/models/noperm-export"
    assert report["server_runtime"]["gs32_fusion"] == "full_ffn"
    assert report["server_runtime"]["device_execution_verified"] is False
    assert "not live HTTP proof" in report["server_runtime_source"]["scope"]
    assert context["raw_record"] == path.read_bytes()
    assert all(row["model_path"] == report["model_path"] for row in report["by_concurrency"])
    fetch.assert_called_once_with("http://127.0.0.1:8003/v1/models", timeout=30)


@pytest.mark.parametrize(
    "change",
    [
        {"gs32_backend": "unknown"},
        {"gs32_execution": "unknown"},
        {"gs32_fusion": "unknown"},
        {"gs32_backend": "reference_mxfp8"},
        {"gs32_execution": "serial"},
        {"model_path": ""},
        {"status": "failed"},
    ],
)
def test_runtime_record_invalid_configuration_rejected_before_network(benchmark, tmp_path, monkeypatch, change):
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps({**startup_record(), **change}), encoding="utf-8")
    fetch = Mock(side_effect=AssertionError("invalid records must fail before HTTP"))
    monkeypatch.setattr(benchmark, "urlopen", fetch)
    with pytest.raises(ValueError):
        benchmark.runtime_context(small_args(benchmark, "--runtime-record", str(path)))
    fetch.assert_not_called()


@pytest.mark.parametrize(
    "models",
    [
        [{"id": "vq2a8", "root": "/models/other"}],
        [],
        [{"id": "vq2a8", "root": "/models/noperm-export"}] * 2,
    ],
)
def test_runtime_record_cannot_be_attached_to_different_or_ambiguous_model(benchmark, tmp_path, monkeypatch, models):
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(startup_record()), encoding="utf-8")
    monkeypatch.setattr(benchmark, "urlopen", Mock(return_value=io.StringIO(json.dumps({"data": models}))))
    with pytest.raises(ValueError, match="unique served model root"):
        benchmark.runtime_context(small_args(benchmark, "--runtime-record", str(path)))


@pytest.mark.parametrize(
    "mode,ep,tp,dp", [("tp-ep", ep, ep, 1) for ep in (1, 2, 4, 8)] + [("dp-ep", ep, 1, ep) for ep in (2, 4, 8)]
)
def test_supported_topologies_describe_server_capacity(benchmark, mode, ep, tp, dp):
    args = benchmark.parse_args(
        ["--tokenizer", "local", "--parallel-mode", mode, "--ep-size", str(ep), "--max-num-seqs", "16"]
    )
    assert (args.tp, args.dp, args.ep_size) == (tp, dp, ep)
    assert args.global_sequence_cap == 16 * dp
    assert args.concurrency[-1] == args.global_sequence_cap


@pytest.mark.parametrize(
    "extra",
    [
        ["--parallel-mode", "dp-ep", "--ep-size", "1"],
        ["--parallel-mode", "dp-ep", "--ep-size", "16"],
        ["--ep-size", "3"],
        ["--max-num-seqs", "0"],
        ["--max-num-seqs", "65"],
        ["--concurrency", "1", "1"],
        ["--concurrency", "0"],
        ["--concurrency", "8", "--num-prompts", "7"],
        ["--num-prompts", "0"],
        ["--num-prompts", "-1"],
        ["--num-warmups", "-1"],
        ["--repeats", "0"],
    ],
)
def test_invalid_topology_and_scan_settings_are_rejected(benchmark, extra):
    with pytest.raises(SystemExit) as error:
        benchmark.parse_args(["--tokenizer", "local", *extra])
    assert error.value.code == 2


@pytest.mark.parametrize("mode,ep_size,cap", [("tp-ep", 2, 64), ("dp-ep", 4, 256), ("dp-ep", 8, 512)])
def test_default_scan_uses_four_requests_per_concurrency_slot(benchmark, tmp_path, mode, ep_size, cap):
    args = benchmark.parse_args(["--tokenizer", "local", "--parallel-mode", mode, "--ep-size", str(ep_size)])
    expected = [value for value in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512) if value <= cap]
    assert args.concurrency == expected
    assert args.global_sequence_cap == cap
    assert args.num_prompts is None
    assert args.repeats == 1
    assert args.input_len == 1024
    assert args.output_len == 128
    assert args.num_warmups == 1
    for concurrency in args.concurrency:
        command = benchmark.bench_command(args, concurrency, tmp_path / "result.json")
        assert command[command.index("--num-prompts") + 1] == str(4 * concurrency)
        assert command[command.index("--max-concurrency") + 1] == str(concurrency)
        assert command[command.index("--num-warmups") + 1] == "1"
        assert command[command.index("--random-input-len") + 1] == "1024"
        assert command[command.index("--random-output-len") + 1] == "128"
        assert command[command.index("--random-range-ratio") + 1] == "0"
        assert "--ignore-eos" in command and "--save-detailed" in command


@pytest.mark.parametrize("num_warmups", [0, 3])
@pytest.mark.parametrize("fixed_count", [False, True])
def test_warmup_count_is_independent_of_concurrency_and_formal_requests(benchmark, tmp_path, num_warmups, fixed_count):
    args = benchmark.parse_args(
        [
            "--tokenizer",
            "local",
            "--concurrency",
            "1",
            "64",
            "--num-warmups",
            str(num_warmups),
            *(["--num-prompts", "256"] if fixed_count else []),
        ]
    )
    for concurrency in args.concurrency:
        command = benchmark.bench_command(args, concurrency, tmp_path / "result.json")
        assert command[command.index("--num-warmups") + 1] == str(num_warmups)
        assert command[command.index("--num-prompts") + 1] == str(256 if fixed_count else 4 * concurrency)


def test_dp8_local_b8_scan_keeps_global_b64(benchmark):
    args = benchmark.parse_args(
        ["--tokenizer", "local", "--parallel-mode", "dp-ep", "--ep-size", "8", "--max-num-seqs", "8"]
    )
    assert args.global_sequence_cap == 64
    assert args.concurrency == [1, 2, 4, 8, 16, 32, 64]
    assert benchmark.requests_per_trial(args, 1) == 4
    assert benchmark.requests_per_trial(args, 64) == 256


@pytest.mark.parametrize("mode,ep,cap,scan", [("tp-ep", 4, 7, [1, 2, 4, 7]), ("dp-ep", 2, 14, [1, 2, 4, 8, 14])])
def test_non_power_of_two_capacity_is_included(benchmark, mode, ep, cap, scan):
    args = benchmark.parse_args(
        ["--tokenizer", "local", "--parallel-mode", mode, "--ep-size", str(ep), "--max-num-seqs", "7"]
    )
    assert args.global_sequence_cap == cap
    assert args.concurrency == scan
    assert [benchmark.requests_per_trial(args, concurrency) for concurrency in scan] == [4 * value for value in scan]


def test_dry_run_has_no_subprocess_or_file_side_effects(benchmark, tmp_path, monkeypatch, capsys):
    run = Mock(side_effect=AssertionError("dry run must not start a child process"))
    metadata = Mock(side_effect=AssertionError("dry run must not inspect packages or Git"))
    monkeypatch.setattr(benchmark.subprocess, "run", run)
    monkeypatch.setattr(benchmark, "client_metadata", metadata)
    monkeypatch.setattr(benchmark.tempfile, "mkdtemp", Mock(side_effect=AssertionError("unexpected directory")))
    destination = tmp_path / "results"
    code = benchmark.main(
        [
            "--tokenizer",
            "missing-tokenizer",
            "--max-num-seqs",
            "2",
            "--repeats",
            "2",
            "--result-dir",
            str(destination),
            "--dry-run",
        ]
    )
    assert code == 0
    run.assert_not_called()
    metadata.assert_not_called()
    assert list(tmp_path.iterdir()) == []
    output = capsys.readouterr().out
    assert output.count("--num-prompts 4") == 2
    assert output.count("--num-prompts 8") == 2
    assert output.count("--num-warmups 1") == 4
    assert output.count("bench serve") == 4


@pytest.mark.parametrize(
    "field,value",
    [
        ("completed", 3),
        ("completed", True),
        ("failed", 1),
        ("failed", False),
        ("total_output_tokens", 15),
        ("output_lens", [4, 4, 4, 3]),
        ("output_lens", [4, 4, 4]),
        ("output_lens", [4, 4, 4, True]),
        ("input_lens", [8, 8, 8, 0]),
        ("errors", ["", "server error", "", ""]),
        ("errors", ["", "", ""]),
        ("duration", 0.0),
        ("duration", 2.0),
        ("output_throughput", 0.0),
        ("output_throughput", 17.0),
        ("request_throughput", 3.0),
        ("request_throughput", float("nan")),
        ("p99_ttft_ms", float("nan")),
        ("p99_tpot_ms", float("inf")),
        ("mean_tpot_ms", -1.0),
    ],
)
def test_result_validation_rejects_failed_short_invalid_or_inconsistent_measurements(benchmark, field, value):
    args = small_args(benchmark)
    result = valid_result(args)
    result[field] = value
    with pytest.raises(ValueError):
        benchmark.validate_result(result, args, 1)


@pytest.mark.parametrize(
    "field",
    [
        "completed",
        "failed",
        "total_output_tokens",
        "output_lens",
        "input_lens",
        "errors",
        "duration",
        "output_throughput",
        "request_throughput",
        "mean_ttft_ms",
        "p99_ttft_ms",
        "mean_tpot_ms",
        "p99_tpot_ms",
        "p99_e2el_ms",
    ],
)
def test_result_validation_requires_all_counts_details_and_metrics(benchmark, field):
    args = small_args(benchmark)
    result = valid_result(args)
    del result[field]
    with pytest.raises(ValueError):
        benchmark.validate_result(result, args, 1)


def test_result_validation_accepts_complete_measurement_and_records_actual_input_range(benchmark):
    args = small_args(benchmark)
    result = valid_result(args)
    result["input_lens"] = [7, 8, 9, 8]
    metrics = benchmark.validate_result(result, args, 1)
    assert metrics["min_input_tokens"] == 7 and metrics["max_input_tokens"] == 9
    assert metrics["output_throughput"] == 16.0


@pytest.mark.parametrize("concurrency", [1, 2, 64])
def test_adaptive_validation_uses_current_trial_count(benchmark, concurrency):
    args = benchmark.parse_args(["--tokenizer", "local"])
    result = valid_result(args, num_prompts=4 * concurrency)
    metrics = benchmark.validate_result(result, args, concurrency)
    assert metrics["completed"] == 4 * concurrency
    assert metrics["total_output_tokens"] == 4 * concurrency * args.output_len
    # A well-formed result for another concurrency must not be accepted.
    with pytest.raises(ValueError, match="completed must equal"):
        benchmark.validate_result(result, args, concurrency * 2)


def test_summary_uses_median_of_repeated_trials_for_peak(benchmark):
    args = small_args(benchmark)
    runs = [
        valid_run(benchmark, args, concurrency, repeat, throughput=value)
        for concurrency, values in ((1, (10.0, 100.0, 12.0)), (2, (20.0, 21.0, 22.0)))
        for repeat, value in enumerate(values, 1)
    ]
    summary = benchmark.summarize(args, runs)
    assert summary["complete"]
    assert summary["peak_measured"]["concurrency"] == 2
    assert summary["peak_measured"]["median_output_throughput"] == 21.0
    assert summary["by_concurrency"][0]["median_output_throughput"] == 12.0
    assert summary["peak_measured_with_slo"] is None


@pytest.mark.parametrize("failure", ["partial", "failed"])
def test_failed_or_partial_scan_never_claims_a_peak(benchmark, failure):
    args = small_args(benchmark)
    runs = [valid_run(benchmark, args, concurrency, repeat) for concurrency in args.concurrency for repeat in (1, 2, 3)]
    if failure == "partial":
        runs.pop()
    else:
        runs[-1] = {"concurrency": 2, "repeat": 3, "valid": False, "error": "failed"}
    summary = benchmark.summarize(args, runs)
    assert not summary["complete"]
    assert summary["peak_measured"] is summary["peak_measured_with_slo"] is None
    assert summary["by_concurrency"][0]["eligible"]
    assert not summary["by_concurrency"][1]["eligible"]


@pytest.mark.parametrize("metric,exceeded", [("ttft", {"ttft": 11.0}), ("tpot", {"tpot": 3.0})])
def test_slo_peak_requires_every_repeat_to_meet_both_ceilings(benchmark, metric, exceeded):
    args = small_args(benchmark, "--max-p99-ttft-ms", "10", "--max-p99-tpot-ms", "2")
    runs = [
        valid_run(
            benchmark,
            args,
            concurrency,
            repeat,
            throughput=concurrency * 10.0,
            **(exceeded if concurrency == 2 and repeat == 3 else {}),
        )
        for concurrency in args.concurrency
        for repeat in (1, 2, 3)
    ]
    summary = benchmark.summarize(args, runs)
    assert summary["peak_measured"]["concurrency"] == 2
    assert summary["peak_measured_with_slo"]["concurrency"] == 1
    assert summary["by_concurrency"][1][f"median_p99_{metric}_ms"] <= getattr(args, f"max_p99_{metric}_ms")
    assert not summary["by_concurrency"][1]["meets_slo"]


def fake_bench(benchmark, monkeypatch, *, failure=None):
    calls = []
    monkeypatch.setattr(benchmark, "client_metadata", lambda: {"repository_head": "test-revision", "packages": {}})

    def run(command, *, stdout, stderr, timeout, check):
        assert command[1:5] == ["-m", "vllm.entrypoints.cli.main", "bench", "serve"]
        assert stderr == subprocess.STDOUT and check is True and timeout > 0
        calls.append(command)
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, timeout)
        args = small_args(
            benchmark,
            "--num-prompts",
            command[command.index("--num-prompts") + 1],
            "--input-len",
            command[command.index("--random-input-len") + 1],
            "--output-len",
            command[command.index("--random-output-len") + 1],
        )
        concurrency = int(command[command.index("--max-concurrency") + 1])
        result = valid_result(args, throughput=concurrency * 20.0)
        result["test_raw_detail"] = "preserved"
        result["model_id"] = "other-model" if failure == "model_id" else "vq2a8"
        if failure == "failed_json":
            result["failed"] = 1
        path = Path(command[command.index("--result-dir") + 1]) / command[command.index("--result-filename") + 1]
        path.write_text(json.dumps(result), encoding="utf-8")
        stdout.write("fake benchmark subprocess output\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(benchmark.subprocess, "run", run)
    return calls


def main_options(destination):
    return [
        "--tokenizer",
        "local-tokenizer",
        "--concurrency",
        "1",
        "2",
        "--num-prompts",
        "4",
        "--input-len",
        "8",
        "--output-len",
        "4",
        "--repeats",
        "2",
        "--result-dir",
        str(destination),
        "--note",
        "test hardware and build",
    ]


@pytest.mark.parametrize("different_model_id", [False, True])
def test_main_records_server_evidence_and_rejects_mismatched_benchmark_model(
    benchmark, tmp_path, monkeypatch, different_model_id
):
    fake_bench(benchmark, monkeypatch, failure="model_id" if different_model_id else None)
    record_path = tmp_path / "runtime.json"
    record_path.write_text(json.dumps(startup_record()), encoding="utf-8")
    snapshot = {"data": [{"id": "vq2a8", "root": "/models/noperm-export"}]}
    monkeypatch.setattr(benchmark, "urlopen", Mock(return_value=io.StringIO(json.dumps(snapshot))))
    destination = tmp_path / "results"
    status = benchmark.main(main_options(destination) + ["--runtime-record", str(record_path)])
    assert status == int(different_model_id)
    (directory,) = destination.iterdir()
    report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    assert report["server_runtime"] == startup_record()
    assert (directory / "service-runtime.json").read_bytes() == record_path.read_bytes()
    assert json.loads((directory / "models.json").read_text(encoding="utf-8")) == snapshot
    if different_model_id:
        assert not report["complete"]
        assert "model_id" in report["runs"][0]["error"]
        assert not (directory / "summary.csv").exists()
    else:
        assert report["complete"]
        assert report["peak_measured"]["model_path"] == "/models/noperm-export"
        assert all(run["model_path"] == "/models/noperm-export" for run in report["runs"])
        with (directory / "summary.csv").open(encoding="utf-8", newline="") as stream:
            assert all(row["model_path"] == "/models/noperm-export" for row in csv.DictReader(stream))


@pytest.mark.parametrize("adaptive", [False, True])
@pytest.mark.parametrize("num_warmups", [0, 1])
def test_main_preserves_raw_results_and_writes_csv_manifest_and_summary(
    benchmark, tmp_path, monkeypatch, adaptive, num_warmups
):
    calls = fake_bench(benchmark, monkeypatch)
    destination = tmp_path / "results"
    options = main_options(destination) + ["--num-warmups", str(num_warmups)]
    if adaptive:
        index = options.index("--num-prompts")
        del options[index : index + 2]
    assert benchmark.main(options) == 0
    assert len(calls) == 4
    assert [command[command.index("--max-concurrency") + 1] for command in calls] == ["1", "2", "1", "2"]
    expected_counts = [4, 8] if adaptive else [4, 4]
    assert [int(command[command.index("--num-prompts") + 1]) for command in calls] == expected_counts * 2
    assert [int(command[command.index("--num-warmups") + 1]) for command in calls] == [num_warmups] * 4
    (directory,) = destination.iterdir()
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    assert summary["complete"] and len(summary["runs"]) == 4
    assert summary["peak_measured"]["concurrency"] == 2
    assert [run["num_prompts"] for run in summary["runs"]] == expected_counts * 2
    assert [run["num_warmups"] for run in summary["runs"]] == [num_warmups] * 4
    assert [run["completed"] for run in summary["runs"]] == expected_counts * 2
    assert [row["num_prompts"] for row in summary["by_concurrency"]] == expected_counts
    for run in summary["runs"]:
        raw = directory / run["result_file"]
        assert json.loads(raw.read_text(encoding="utf-8"))["test_raw_detail"] == "preserved"
        assert "fake benchmark" in raw.with_suffix(".log").read_text(encoding="utf-8")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["client_environment"]["repository_head"] == "test-revision"
    assert manifest["declared_server_and_workload"]["note"] == "test hardware and build"
    assert manifest["declared_server_and_workload"]["num_prompts"] == (None if adaptive else 4)
    assert manifest["declared_server_and_workload"]["num_warmups"] == num_warmups
    assert manifest["requests_by_concurrency"] == {"1": expected_counts[0], "2": expected_counts[1]}
    with (directory / "summary.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["concurrency"]) for row in rows] == [1, 2]
    assert [int(row["num_prompts"]) for row in rows] == expected_counts
    assert [float(row["median_output_throughput"]) for row in rows] == [20.0, 40.0]


@pytest.mark.parametrize("mode", ["tp-ep", "dp-ep"])
@pytest.mark.parametrize("ep_size", [2, 4, 8])
def test_result_directory_matches_client_topology(benchmark, tmp_path, monkeypatch, mode, ep_size):
    calls = fake_bench(benchmark, monkeypatch)
    destination = tmp_path / "results"
    options = main_options(destination) + ["--parallel-mode", mode, "--ep-size", str(ep_size), "--repeats", "1"]
    assert benchmark.main(options) == 0
    tp, dp = (ep_size, 1) if mode == "tp-ep" else (1, ep_size)
    (directory,) = destination.iterdir()
    assert directory.name.startswith(f"tp{tp}-dp{dp}-ep{ep_size}-b64-")
    for command in calls:
        assert Path(command[command.index("--result-dir") + 1]) == directory
        assert (directory / command[command.index("--result-filename") + 1]).is_file()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    declared = manifest["declared_server_and_workload"]
    assert (declared["tp"], declared["dp"], declared["ep_size"]) == (tp, dp, ep_size)
    assert (directory / "summary.csv").is_file()
    assert (directory / "summary.json").is_file()


@pytest.mark.parametrize("failure,error_text", [("failed_json", "failed must equal 0"), ("timeout", "TimeoutExpired")])
def test_main_rejects_failed_json_even_with_zero_exit_and_reports_timeout(
    benchmark, tmp_path, monkeypatch, failure, error_text
):
    calls = fake_bench(benchmark, monkeypatch, failure=failure)
    destination = tmp_path / "results"
    assert benchmark.main(main_options(destination)) == 1
    assert len(calls) == 1
    (directory,) = destination.iterdir()
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    assert not summary["complete"]
    assert summary["peak_measured"] is summary["peak_measured_with_slo"] is None
    assert len(summary["runs"]) == 1 and not summary["runs"][0]["valid"]
    assert error_text in summary["runs"][0]["error"]
    assert summary["runs"][0]["num_prompts"] == 4
    assert (directory / "manifest.json").is_file()
    assert not (directory / "summary.csv").exists()


@pytest.fixture
def performance_launcher(tmp_path):
    if os.name == "nt" or shutil.which("bash") is None:
        pytest.skip("Shell launcher execution requires a POSIX environment with bash.")
    model = tmp_path / "source model with spaces"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "deepseek_v4",
                "quantization_config": {
                    "quant_method": "vq2a8",
                    "format": "vq2_mixed_gs32_source",
                    "format_version": 3,
                },
            }
        ),
        encoding="utf-8",
    )
    (model / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    library = tmp_path / "test library.so"
    binary = tmp_path / "test kernel.o"
    library.touch()
    binary.touch()
    executable_dir = tmp_path / "bin"
    executable_dir.mkdir()
    executable = executable_dir / "python3"
    executable.write_text(
        "#!/bin/sh\n"
        'printf "LAUNCH_ENV=%s,%s,%s,%s,%s\\n" "$ASCEND_RT_VISIBLE_DEVICES" '
        '"$VLLM_USE_V2_MODEL_RUNNER" "$VLLM_WORKER_MULTIPROC_METHOD" "$OMP_NUM_THREADS" "$MKL_NUM_THREADS"\n'
        f'exec {shlex.quote(sys.executable)} "$@"\n',
        encoding="utf-8",
    )
    executable.chmod(0o755)

    def launch(*extra_args, environment=None):
        env = os.environ.copy()
        for name in ("VLLM_USE_V2_MODEL_RUNNER", "VLLM_WORKER_MULTIPROC_METHOD", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            env.pop(name, None)
        env.update(
            {
                "PATH": f"{executable_dir}{os.pathsep}{env.get('PATH', '')}",
                "ASCEND_RT_VISIBLE_DEVICES": "4,5",
                **(environment or {}),
            }
        )
        command = [
            "bash",
            str(REPO_ROOT / "tools/tests/serve_vq2a8_performance.sh"),
            "--model",
            str(model),
            "--gs32-library",
            str(library),
            "--gs32-binary",
            str(binary),
            "--dry-run",
            *extra_args,
        ]
        result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=20, check=False)
        return result, (model, library, binary)

    return launch


def command_argument(command, name):
    return command[command.index(name) + 1]


def test_performance_launcher_defaults_to_mrv2_full_graph_b64_and_preserves_spaced_paths(performance_launcher):
    result, (model, library, binary) = performance_launcher()
    assert result.returncode == 0, result.stderr
    environment, invocation = result.stdout.splitlines()
    assert environment == "LAUNCH_ENV=4,5,1,spawn,1,1"
    command = shlex.split(invocation)
    assert command[:2] == ["env", "VLLM_USE_V2_MODEL_RUNNER=1"]
    assert command_argument(command, "serve") == str(model)
    assert command_argument(command, "--max-model-len") == "8192"
    assert command_argument(command, "--max-num-seqs") == "64"
    assert command_argument(command, "--max-num-batched-tokens") == "8192"
    assert command_argument(command, "--tensor-parallel-size") == "2"
    assert command_argument(command, "--data-parallel-size") == "1"
    assert command_argument(command, "--kv-cache-memory-bytes") == str(16 * 1024**3)
    assert "--enable-expert-parallel" in command and "--enforce-eager" not in command
    compilation = json.loads(command_argument(command, "--compilation-config"))
    assert compilation["cudagraph_mode"] == "FULL_DECODE_ONLY"
    assert compilation["cudagraph_capture_sizes"] == [1, 2, 4, 8, 16, 32, 64]
    native = json.loads(command_argument(command, "--additional-config"))["vq2a8"]
    assert native == {
        "gs32_library": str(library),
        "gs32_binary": str(binary),
        "gs32_execution": "full_graph",
    }


@pytest.mark.parametrize("dp_size,local_batch,local_tokens", [(4, 16, 2048), (8, 8, 1024)])
def test_performance_launcher_accepts_dp_and_per_engine_overrides(
    performance_launcher, dp_size, local_batch, local_tokens
):
    devices = ",".join(map(str, range(dp_size)))
    result, _ = performance_launcher(
        "--parallel-mode",
        "dp-ep",
        "--ep-size",
        str(dp_size),
        "--max-num-seqs",
        str(local_batch),
        "--max-num-batched-tokens",
        str(local_tokens),
        "--port",
        "8004",
        environment={"ASCEND_RT_VISIBLE_DEVICES": devices},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[0] == f"LAUNCH_ENV={devices},1,spawn,1,1"
    command = shlex.split(result.stdout.splitlines()[-1])
    assert command_argument(command, "--tensor-parallel-size") == "1"
    assert command_argument(command, "--data-parallel-size") == str(dp_size)
    assert command_argument(command, "--data-parallel-size-local") == str(dp_size)
    assert command_argument(command, "--data-parallel-backend") == "mp"
    assert command_argument(command, "--max-num-seqs") == str(local_batch)
    assert command_argument(command, "--max-num-batched-tokens") == str(local_tokens)
    assert command_argument(command, "--port") == "8004"
    compilation = json.loads(command_argument(command, "--compilation-config"))
    assert compilation["cudagraph_capture_sizes"] == [size for size in (1, 2, 4, 8, 16) if size <= local_batch]


def test_performance_launcher_preserves_explicit_worker_environment(performance_launcher):
    result, _ = performance_launcher(
        environment={
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "VLLM_WORKER_MULTIPROC_METHOD": "fork",
            "OMP_NUM_THREADS": "12",
            "MKL_NUM_THREADS": "8",
        }
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[0] == "LAUNCH_ENV=4,5,1,fork,12,8"


def test_performance_launcher_rejects_batch_above_supported_range(performance_launcher):
    result, _ = performance_launcher("--max-num-seqs", "65")
    assert result.returncode != 0
    assert "max_num_seqs" in result.stderr
    assert "[1,64]" in result.stderr
