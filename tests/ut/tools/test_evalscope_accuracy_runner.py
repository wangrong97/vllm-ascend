# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

RUNNER = Path(__file__).resolve().parents[3] / "tools/tests/run_evalscope_accuracy.py"


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(RUNNER.parent))
    spec = importlib.util.spec_from_file_location("evalscope_accuracy_runner_test", RUNNER)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("profile,model,port", [("baseline", "baseline", 8001), ("vq2a8", "vq2a8", 8000)])
def test_defaults_preserve_existing_accuracy_conditions(runner, tmp_path, profile, model, port):
    args = runner.parse_args([profile])
    command = runner.build_command(args, tmp_path / "run with spaces")
    assert command[:4] == [sys.executable, "-m", "evalscope.cli.cli", "eval"]
    assert command[command.index("--model") + 1] == model
    assert command[command.index("--api-url") + 1] == f"http://127.0.0.1:{port}/v1"
    assert command[command.index("--datasets") + 1 : command.index("--eval-batch-size")] == [
        "gpqa_diamond",
        "aime24",
        "aime25",
    ]
    assert command[command.index("--eval-batch-size") + 1] == "4"
    assert "--limit" not in command
    assert "--dataset-dir" not in command
    assert "--no-timestamp" in command
    generation = json.loads(command[command.index("--generation-config") + 1])
    assert generation["max_tokens"] == 8192
    assert generation["temperature"] == 0
    assert generation["stream"] is False
    assert generation["extra_body"]["ignore_eos"] is False
    assert "top_p" not in generation
    assert "reasoning_effort" not in generation
    assert "chat_template_kwargs" not in generation["extra_body"]
    assert json.loads(command[command.index("--judge") + 1]) == {"strategy": "rule"}


def test_dataset_cache_directory_passed_to_evalscope_without_loading_data(runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = runner.parse_args(["baseline", "--dataset-dir", "cache with spaces"])
    command = runner.build_command(args, tmp_path / "run")
    assert command[command.index("--dataset-dir") + 1] == str(tmp_path / "cache with spaces")
    assert not (tmp_path / "cache with spaces").exists()


@pytest.mark.parametrize(
    "option,value",
    [
        ("--concurrency", "0"),
        ("--max-tokens", "0"),
        ("--repeats", "0"),
        ("--limit", "0"),
        ("--timeout", "0"),
        ("--temperature", "-1"),
        ("--temperature", "nan"),
        ("--top-p", "1.1"),
        ("--top-p", "nan"),
        ("--datasets", "unknown"),
        ("--reasoning-effort", "unknown"),
    ],
)
def test_invalid_cli_rejected_before_starting_evaluation(runner, option, value):
    with pytest.raises(SystemExit) as exc:
        runner.parse_args(["baseline", option, value])
    assert exc.value.code == 2


@pytest.mark.parametrize("effort", ["none", "low", "high", "max"])
def test_explicit_reasoning_and_sampling_reach_service(runner, effort):
    args = runner.parse_args(
        ["baseline", "--reasoning-effort", effort, "--temperature", "1", "--top-p", "1", "--max-tokens", "32768"]
    )
    generation = runner.generation_config(args)
    assert generation["reasoning_effort"] == effort
    assert generation["extra_body"]["chat_template_kwargs"] == {
        "thinking": effort != "none",
        "reasoning_effort": effort,
    }
    assert generation["temperature"] == 1
    assert generation["top_p"] == 1
    assert generation["max_tokens"] == 32768


@pytest.mark.parametrize(
    "url",
    [
        "[http://127.0.0.1:8001/v1](http://127.0.0.1:8001/v1)",
        "http://127.0.0.1:8001",
        "http://user:password@localhost/v1",
    ],
)
def test_rejects_malformed_or_credential_bearing_api_urls(runner, url):
    with pytest.raises(SystemExit) as exc:
        runner.parse_args(["baseline", "--api-url", url])
    assert exc.value.code == 2


@pytest.mark.parametrize("max_model_len", [8191, 8192, 8193])
def test_service_rejects_output_budget_without_prompt_space(runner, monkeypatch, max_model_len):
    def urlopen(request, *, timeout):
        assert request.full_url == "http://127.0.0.1:8001/v1/models"
        assert timeout > 0
        return io.StringIO(json.dumps({"data": [{"id": "baseline", "max_model_len": max_model_len}]}))

    monkeypatch.setattr(runner, "urlopen", urlopen)
    args = runner.parse_args(["baseline"])
    if max_model_len <= args.max_tokens:
        with pytest.raises(ValueError, match="no room for a prompt"):
            runner.check_service(args)
    else:
        assert runner.check_service(args)["max_model_len"] == max_model_len


def test_service_must_expose_requested_model(runner, monkeypatch):
    monkeypatch.setattr(runner, "urlopen", lambda *args, **kwargs: io.StringIO('{"data": [{"id": "vq2a8"}]}'))
    with pytest.raises(ValueError, match="does not expose 'baseline'"):
        runner.check_service(runner.parse_args(["baseline"]))


@pytest.mark.parametrize("version", ["1.12.0", "1.11.0", None])
def test_dependency_pin_prevents_schema_mismatch(runner, monkeypatch, version):
    def installed(package):
        assert package == "evalscope"
        if version is None:
            raise runner.metadata.PackageNotFoundError(package)
        return version

    monkeypatch.setattr(runner.metadata, "version", installed)
    if version == "1.12.0":
        assert runner.check_evalscope_version() == version
    else:
        with pytest.raises(ValueError, match="not installed|Expected EvalScope"):
            runner.check_evalscope_version()


def test_dry_run_needs_no_evalscope_service_or_output_directory(runner, tmp_path, monkeypatch, capsys):
    def unexpected(*args, **kwargs):
        pytest.fail("dry-run must not contact dependencies or the service")

    monkeypatch.setattr(runner, "check_evalscope_version", unexpected)
    monkeypatch.setattr(runner, "check_service", unexpected)
    monkeypatch.setattr(runner.subprocess, "run", unexpected)
    destination = tmp_path / "uncreated"
    assert runner.main(["baseline", "--work-dir", str(destination), "--dry-run"]) == 0
    assert not destination.exists()
    assert "aime25" in capsys.readouterr().out


@pytest.fixture
def fake_evaluation(runner, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "check_evalscope_version", lambda: "1.12.0")
    monkeypatch.setattr(runner, "check_service", lambda args: {"id": args.model, "max_model_len": 65536})
    monkeypatch.setattr(runner, "client_revision", lambda: "client-commit")

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", run)
    monkeypatch.setattr(runner, "summarize_run", lambda *args: {"invalid": False, "datasets": {}})
    return calls


def test_success_records_audit_and_preserves_live_stdio(runner, fake_evaluation, tmp_path):
    destination = tmp_path / "runs"
    assert runner.main(["baseline", "--work-dir", str(destination)]) == 0
    command, kwargs = fake_evaluation[0]
    for key in ("stdout", "stderr", "stdin"):
        assert kwargs.get(key) is None
    run_dir = Path(command[command.index("--work-dir") + 1])
    assert run_dir.parent == destination
    assert (run_dir / "run.json").is_file()
    assert json.loads((run_dir / "accuracy_audit.json").read_text())["invalid"] is False
    assert json.loads((run_dir / "run.json").read_text())["status"] == "finished"


def test_repeated_invocations_do_not_reuse_or_overwrite_results(runner, fake_evaluation, tmp_path):
    args = ["baseline", "--work-dir", str(tmp_path)]
    assert runner.main(args) == 0
    assert runner.main(args) == 0
    directories = [Path(command[command.index("--work-dir") + 1]) for command, _ in fake_evaluation]
    assert directories[0] != directories[1]
    assert all((directory / "accuracy_audit.json").is_file() for directory in directories)


def test_limit_repeats_and_datasets_reach_audit(runner, fake_evaluation, tmp_path, monkeypatch):
    observed = []

    def summarize(run_dir, datasets, expected):
        observed.append((datasets, expected))
        return {"invalid": False, "datasets": {}}

    monkeypatch.setattr(runner, "summarize_run", summarize)
    args = [
        "vq2a8",
        "--work-dir",
        str(tmp_path),
        "--datasets",
        "aime25",
        "gpqa_diamond",
        "--limit",
        "40",
        "--repeats",
        "2",
    ]
    assert runner.main(args) == 0
    assert observed == [(["aime25", "gpqa_diamond"], {"aime25": 60, "gpqa_diamond": 80})]


def test_child_failure_is_not_reported_as_valid_accuracy(runner, fake_evaluation, tmp_path, monkeypatch):
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=7))

    def unexpected(*args):
        pytest.fail("a failed evaluator must not be summarized as a successful run")

    monkeypatch.setattr(runner, "summarize_run", unexpected)
    assert runner.main(["baseline", "--work-dir", str(tmp_path)]) == 7


def test_invalid_audit_fails_even_when_evaluator_exits_zero(runner, fake_evaluation, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "summarize_run", lambda *args: {"invalid": True, "datasets": {}})
    assert runner.main(["baseline", "--work-dir", str(tmp_path)]) == 1


def test_missing_predictions_fail_instead_of_silent_success(runner, fake_evaluation, tmp_path, monkeypatch):
    def summarize(*args):
        raise ValueError("Expected one prediction file; found 0")

    monkeypatch.setattr(runner, "summarize_run", summarize)
    assert runner.main(["baseline", "--work-dir", str(tmp_path)]) == 1


@pytest.mark.parametrize("profile", ["baseline", "vq2a8"])
def test_bash_wrapper_accepts_dry_run_from_other_directory(profile, tmp_path):
    wrapper = RUNNER.with_name(f"run_{profile}_evalscope_accuracy.sh")
    result = subprocess.run(
        ["bash", str(wrapper), "--dry-run"], cwd=tmp_path, capture_output=True, text=True, check=False, timeout=20
    )
    assert result.returncode == 0, result.stderr
    assert "aime25" in result.stdout


def test_suite_entry_point_supports_aliases_from_another_directory(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER.with_name("run_vq2a8_accuracy_suite.py")),
            "--datasets",
            "GPQA",
            "AIME24",
            "aime25",
            "--base-url",
            "http://localhost:8123",
            "--result-dir",
            str(tmp_path / "uncreated"),
            "--dry-run",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "--datasets gpqa_diamond aime24 aime25" in result.stdout
    assert "http://localhost:8123/v1" in result.stdout
    assert not (tmp_path / "uncreated").exists()


def test_alias_duplicates_rejected(runner):
    with pytest.raises(SystemExit) as exc:
        runner.parse_args(["vq2a8", "--datasets", "gpqa", "gpqa_diamond"])
    assert exc.value.code == 2


def test_offline_source_excludes_online_cache_option(runner, tmp_path):
    with pytest.raises(SystemExit) as exc:
        runner.parse_args(["vq2a8", "--offline-data-dir", str(tmp_path), "--dataset-dir", str(tmp_path)])
    assert exc.value.code == 2


def test_offline_bundle_verified_before_service_and_passed_to_evaluator(runner, fake_evaluation, tmp_path, monkeypatch):
    calls = []
    verifier = ModuleType("prepare_evalscope_datasets")

    def validate(path, names):
        calls.append("bundle")
        assert path == tmp_path / "bundle"
        assert names == ["gpqa_diamond"]
        return {"verified": True}

    def service(args):
        calls.append("service")
        return {"id": args.model}

    verifier.validate_offline_bundle = validate
    monkeypatch.setitem(sys.modules, verifier.__name__, verifier)
    monkeypatch.setattr(runner, "check_service", service)
    assert (
        runner.main(
            [
                "vq2a8",
                "--offline-data-dir",
                str(tmp_path / "bundle"),
                "--datasets",
                "gpqa",
                "--result-dir",
                str(tmp_path / "runs"),
            ]
        )
        == 0
    )
    assert calls == ["bundle", "service"]
    command = fake_evaluation[0][0]
    assert command[command.index("--dataset-dir") + 1] == str(tmp_path / "bundle")
    run_dir = next((tmp_path / "runs").iterdir())
    assert json.loads((run_dir / "run.json").read_text())["offline_bundle"] == {"verified": True}


def test_corrupt_offline_bundle_does_not_contact_model(runner, fake_evaluation, tmp_path, monkeypatch):
    verifier = ModuleType("prepare_evalscope_datasets")

    def validate(*args):
        raise ValueError("Bundle checksum mismatch")

    def unexpected(*args):
        pytest.fail("invalid bundle must fail before any service query")

    verifier.validate_offline_bundle = validate
    monkeypatch.setitem(sys.modules, verifier.__name__, verifier)
    monkeypatch.setattr(runner, "check_service", unexpected)
    destination = tmp_path / "results"
    assert (
        runner.main(["vq2a8", "--offline-data-dir", str(tmp_path / "bad-bundle"), "--result-dir", str(destination)])
        == 1
    )
    assert not fake_evaluation
    assert not destination.exists()


@pytest.mark.parametrize("extra", [[], ["--limit", "2"]])
def test_result_records_scope_and_server_identity(runner, fake_evaluation, tmp_path, extra):
    args = [
        "vq2a8",
        "--result-dir",
        str(tmp_path),
        "--label",
        "mrv2-none",
        "--server-revision",
        "server-commit",
        *extra,
    ]
    assert runner.main(args) == 0
    run_dir = next(tmp_path.iterdir())
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["scope"] == ("smoke" if extra else "full")
    assert summary["label"] == "mrv2-none"
    assert summary["client_revision"] == "client-commit"
    assert summary["server_revision"] == "server-commit"
    assert summary["repeats"] == 1
    assert (run_dir / "summary.csv").is_file()


def test_audited_score_is_published_without_rescoring(runner, fake_evaluation, tmp_path, monkeypatch, capsys):
    dataset = {
        "score": 0.5,
        "records": 2,
        "expected_records": 2,
        "max_tokens": 1,
        "empty_final_content": 0,
        "invalid": False,
        "output_tokens": {"unknown_records": 0},
    }
    monkeypatch.setattr(runner, "summarize_run", lambda *args: {"invalid": False, "datasets": {"aime25": dataset}})
    assert runner.main(["vq2a8", "--datasets", "aime25", "--limit", "2", "--result-dir", str(tmp_path)]) == 0
    run_dir = next(tmp_path.iterdir())
    assert json.loads((run_dir / "summary.json").read_text())["datasets"]["aime25"]["score"] == 0.5
    assert "aime25,0.5,2,2,1,0,False" in (run_dir / "summary.csv").read_text()
    assert "score=0.500000" in capsys.readouterr().out


@pytest.mark.parametrize("exception,status,code", [(KeyboardInterrupt, "interrupted", 130), (OSError, "failed", 1)])
def test_interrupted_or_failed_child_leaves_terminal_manifest(
    runner, fake_evaluation, tmp_path, monkeypatch, exception, status, code
):
    def fail(*args, **kwargs):
        raise exception("test failure")

    monkeypatch.setattr(runner.subprocess, "run", fail)
    assert runner.main(["vq2a8", "--result-dir", str(tmp_path)]) == code
    run_dir = next(tmp_path.iterdir())
    assert json.loads((run_dir / "run.json").read_text())["status"] == status
    assert not (run_dir / "summary.json").exists()


@pytest.mark.parametrize("write_report", [False, True], ids=["missing-official-report", "complete-official-report"])
def test_real_audit_requires_official_score_artifact_before_publishing_success(
    runner, tmp_path, monkeypatch, write_report
):
    # Exercise the actual runner -> summarize_run interface. Only the evaluator
    # process is replaced; a zero exit and valid predictions alone cannot pass.
    monkeypatch.setattr(runner, "check_evalscope_version", lambda: "1.12.0")
    monkeypatch.setattr(runner, "check_service", lambda args: {"id": args.model, "max_model_len": 65536})
    monkeypatch.setattr(runner, "client_revision", lambda: "client-commit")

    def evaluate(command, **kwargs):
        run_dir = Path(command[command.index("--work-dir") + 1])
        prediction_dir = run_dir / "predictions" / "vq2a8"
        prediction_dir.mkdir(parents=True)
        prediction = {
            "index": 0,
            "model": "vq2a8",
            "messages": [{"role": "user", "content": "Fixture question"}],
            "model_output": {
                "choices": [{"message": {"role": "assistant", "content": "Fixture answer"}, "stop_reason": "stop"}],
                "usage": {"input_tokens": 5, "output_tokens": 4, "total_tokens": 9},
                "error": None,
            },
        }
        (prediction_dir / "aime25_default.jsonl").write_text(json.dumps(prediction) + "\n", encoding="utf-8")
        if write_report:
            # Official v1.12.0 Report schema v2, default accuracy:mean. Do not
            # infer or recompute this score from the arbitrary fixture answer.
            identity = {"name": "accuracy", "aggregation": "mean", "dimensions": {}}
            report = {
                "schema_version": 2,
                "dataset_name": "aime25",
                "model_name": "vq2a8",
                "num": 1,
                "primary_metric_identity": identity,
                "primary_metric_unavailable_reason": None,
                "metrics": [
                    {
                        "identity": identity,
                        "score": 0.0,
                        "macro_score": 0.0,
                        "num": 1,
                        "semantics": {"kind": "quality", "semantic_id": "quality.accuracy.ratio"},
                        "categories": [
                            {
                                "name": ["default"],
                                "score": 0.0,
                                "macro_score": 0.0,
                                "num": 1,
                                "subsets": [{"name": "default", "score": 0.0, "num": 1, "is_aggregate": False}],
                            }
                        ],
                    }
                ],
                "execution_summary": {
                    "requested": 1,
                    "succeeded": 1,
                    "errored": 0,
                    "incomplete": False,
                    "subsets": {"default": {"requested": 1, "succeeded": 1, "errored": 0}},
                },
            }
            report_dir = run_dir / "reports" / "vq2a8"
            report_dir.mkdir(parents=True)
            (report_dir / "aime25.json").write_text(json.dumps(report), encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", evaluate)
    result = runner.main(["vq2a8", "--datasets", "aime25", "--limit", "1", "--result-dir", str(tmp_path)])
    run_dir = next(tmp_path.iterdir())
    manifest = json.loads((run_dir / "run.json").read_text())
    audit = json.loads((run_dir / "accuracy_audit.json").read_text())
    assert result == (0 if write_report else 1)
    assert manifest["status"] == ("finished" if write_report else "invalid")
    assert audit["invalid"] is not write_report
    if write_report:
        score = json.loads((run_dir / "summary.json").read_text())["datasets"]["aime25"]
        assert score["score"] == 0.0 and score["num"] == 1
        assert score["records"] == score["expected_records"] == 1
        assert (run_dir / "summary.csv").is_file()
    else:
        assert "report" in audit["error"]
        assert not (run_dir / "summary.json").exists()
        assert not (run_dir / "summary.csv").exists()
