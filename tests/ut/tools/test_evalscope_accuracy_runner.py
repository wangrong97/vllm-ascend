# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import io
import json
import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path
from types import ModuleType

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
    assert command[:5] == [sys.executable, "-u", "-m", "evalscope.cli.cli", "eval"]
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
    assert args.thinking_mode == "service_default"
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
    "options,effort",
    [
        (["--thinking"], "high"),
        (["--no-thinking"], "none"),
        (["--thinking", "--reasoning-effort", "low"], "low"),
        (["--thinking", "--reasoning-effort", "high"], "high"),
        (["--thinking", "--reasoning-effort", "max"], "max"),
        (["--no-thinking", "--reasoning-effort", "none"], "none"),
    ],
)
def test_thinking_switches_reach_evalscope_request_config(runner, tmp_path, options, effort):
    args = runner.parse_args(["vq2a8", *options])
    command = runner.build_command(args, tmp_path / "run")
    generation = json.loads(command[command.index("--generation-config") + 1])
    assert generation["reasoning_effort"] == effort
    assert generation["extra_body"]["chat_template_kwargs"] == {
        "thinking": effort != "none",
        "reasoning_effort": effort,
    }
    # A mode switch does not implicitly change sampling or the output budget.
    assert generation["temperature"] == 0.0
    assert generation["max_tokens"] == 8192
    assert "top_p" not in generation


@pytest.mark.parametrize(
    "options,message",
    [
        (["--thinking", "--no-thinking"], "not allowed with argument --thinking"),
        (["--thinking", "--reasoning-effort", "none"], "--thinking conflicts"),
        (["--no-thinking", "--reasoning-effort", "low"], "--no-thinking requires"),
        (["--no-thinking", "--reasoning-effort", "high"], "--no-thinking requires"),
        (["--no-thinking", "--reasoning-effort", "max"], "--no-thinking requires"),
    ],
)
def test_conflicting_thinking_options_rejected(runner, capsys, options, message):
    with pytest.raises(SystemExit) as exc:
        runner.parse_args(["vq2a8", *options])
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


def test_help_documents_thinking_defaults(runner, capsys):
    with pytest.raises(SystemExit) as exc:
        runner.parse_args(["--help"])
    assert exc.value.code == 0
    output = " ".join(capsys.readouterr().out.split())
    assert "--thinking" in output and "--no-thinking" in output
    assert "defaults --reasoning-effort to high" in output
    assert "equivalent to --reasoning-effort none" in output


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
    monkeypatch.setattr(runner, "execute_evaluator", unexpected)
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

    def run(command, log):
        calls.append((command, log))
        log.write_text("Evaluator fixture stdout/stderr\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(runner, "execute_evaluator", run)
    monkeypatch.setattr(runner, "summarize_run", lambda *args: {"invalid": False, "datasets": {}})
    return calls


def test_success_records_audit_and_complete_child_log(runner, fake_evaluation, tmp_path):
    destination = tmp_path / "runs"
    assert runner.main(["baseline", "--work-dir", str(destination)]) == 0
    command, log = fake_evaluation[0]
    run_dir = Path(command[command.index("--work-dir") + 1])
    assert run_dir.parent == destination
    assert (run_dir / "run.json").is_file()
    assert json.loads((run_dir / "accuracy_audit.json").read_text())["invalid"] is False
    manifest = json.loads((run_dir / "run.json").read_text())
    assert manifest["status"] == "finished"
    assert manifest["log"] == str(log) == str(run_dir / "evalscope.log")
    assert log.read_text() == "Evaluator fixture stdout/stderr\n"


@pytest.mark.parametrize(
    "options,mode,effort",
    [
        ([], "service_default", None),
        (["--thinking"], "thinking", "high"),
        (["--no-thinking"], "no_thinking", "none"),
        (["--thinking", "--reasoning-effort", "max"], "thinking", "max"),
        (["--reasoning-effort", "low"], "thinking", "low"),
        (["--reasoning-effort", "none"], "no_thinking", "none"),
    ],
)
def test_thinking_mode_and_request_settings_recorded_consistently(
    runner, fake_evaluation, tmp_path, capsys, options, mode, effort
):
    assert runner.main(["vq2a8", "--result-dir", str(tmp_path), *options]) == 0
    command = fake_evaluation[0][0]
    generation = json.loads(command[command.index("--generation-config") + 1])
    run_dir = Path(command[command.index("--work-dir") + 1])
    for filename in ("run.json", "summary.json"):
        recorded = json.loads((run_dir / filename).read_text())
        assert recorded["thinking_mode"] == mode
        assert recorded["generation_config"] == generation
        assert recorded["generation_config"].get("reasoning_effort") == effort
    assert f"Thinking mode: {mode}; reasoning effort: {effort or 'service_default'}" in capsys.readouterr().out


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
    monkeypatch.setattr(runner, "execute_evaluator", lambda *args: 7)

    def unexpected(*args):
        pytest.fail("a failed evaluator must not be summarized as a successful run")

    monkeypatch.setattr(runner, "summarize_run", unexpected)
    assert runner.main(["baseline", "--work-dir", str(tmp_path)]) == 7


@pytest.mark.parametrize("returncode", [0, 7])
def test_real_evaluator_streams_before_exit_and_keeps_complete_output(runner, tmp_path, monkeypatch, returncode):
    log = tmp_path / "evalscope.log"
    acknowledgement = tmp_path / "terminal-observed-output"
    observed_disk = []

    class Console(io.StringIO):
        def write(self, text):
            if "PROGRESS" in text:
                observed_disk.append(log.read_bytes())
                acknowledgement.touch()
            return super().write(text)

    console = Console()
    monkeypatch.setattr(runner.sys, "stdout", console)
    program = (
        "import pathlib, sys, time\n"
        f"ack = pathlib.Path({str(acknowledgement)!r})\n"
        "sys.stdout.buffer.write(b'PROGRESS\\r')\n"
        "sys.stderr.buffer.write(b'stderr before exit\\n')\n"
        "deadline = time.monotonic() + 5\n"
        "while not ack.exists() and time.monotonic() < deadline: time.sleep(0.01)\n"
        "if not ack.exists(): raise SystemExit(99)\n"
        "sys.stdout.buffer.write(b'final without newline')\n"
        f"raise SystemExit({returncode})\n"
    )
    assert runner.execute_evaluator([sys.executable, "-u", "-c", program], log) == returncode
    assert observed_disk and b"PROGRESS\r" in observed_disk[0]
    assert log.read_bytes() == b"PROGRESS\rstderr before exit\nfinal without newline"
    assert "PROGRESS\r" in console.getvalue()
    assert "stderr before exit\n" in console.getvalue()
    assert "final without newline" in console.getvalue()


def test_closed_terminal_does_not_discard_evaluator_log(runner, tmp_path, monkeypatch):
    class ClosedConsole:
        def write(self, text):
            raise BrokenPipeError("terminal closed")

    monkeypatch.setattr(runner.sys, "stdout", ClosedConsole())
    log = tmp_path / "evalscope.log"
    program = "import sys; sys.stdout.buffer.write(b'stdout\\n'); sys.stderr.buffer.write(b'stderr\\n')"
    assert runner.execute_evaluator([sys.executable, "-u", "-c", program], log) == 0
    assert log.read_bytes() == b"stdout\nstderr\n"


def test_interruption_stops_real_evaluator_and_drains_last_output(runner, tmp_path, monkeypatch, capsys):
    log = tmp_path / "evalscope.log"
    ready = tmp_path / "child-ready"
    original_popen = subprocess.Popen
    started = []

    def interrupted_popen(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        started.append(process)
        original_wait = process.wait
        interrupted = False

        def wait(timeout=None):
            nonlocal interrupted
            if not interrupted:
                interrupted = True
                deadline = time.monotonic() + 5
                while not ready.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                assert ready.exists(), "child did not initialize"
                raise KeyboardInterrupt
            return original_wait(timeout=timeout)

        process.wait = wait
        return process

    monkeypatch.setattr(runner.subprocess, "Popen", interrupted_popen)
    program = (
        "import pathlib, sys, time\n"
        "sys.stdout.write('before interruption\\n')\n"
        f"pathlib.Path({str(ready)!r}).touch()\n"
        "time.sleep(30)\n"
    )
    with pytest.raises(KeyboardInterrupt):
        runner.execute_evaluator([sys.executable, "-u", "-c", program], log)
    assert started[0].poll() is not None
    assert b"before interruption" in log.read_bytes()
    assert "before interruption" in capsys.readouterr().out


@pytest.mark.skipif(os.name != "posix", reason="OS SIGTERM handling requires POSIX")
def test_sigterm_marks_interrupted_and_restores_handler(runner, fake_evaluation, tmp_path, monkeypatch):
    original_handler = signal.getsignal(signal.SIGTERM)

    def interrupted(*args):
        signal.raise_signal(signal.SIGTERM)
        pytest.fail("SIGTERM must interrupt the active evaluation")

    monkeypatch.setattr(runner, "execute_evaluator", interrupted)
    assert runner.main(["vq2a8", "--result-dir", str(tmp_path)]) == 130
    assert signal.getsignal(signal.SIGTERM) == original_handler
    manifest = json.loads(next(tmp_path.glob("*/run.json")).read_text())
    assert manifest["status"] == "interrupted" and manifest["returncode"] == 130


@pytest.mark.skipif(sys.platform != "linux", reason="process-group worker cleanup requires Linux")
def test_finished_evaluator_does_not_leave_background_worker(runner, tmp_path):
    worker_pid = tmp_path / "worker.pid"
    ready = tmp_path / "worker-ready"
    worker = f"import pathlib,time; pathlib.Path({str(ready)!r}).touch(); time.sleep(30)"
    program = (
        "import pathlib, subprocess, sys, time\n"
        f"worker = subprocess.Popen([sys.executable, '-u', '-c', {worker!r}])\n"
        f"pathlib.Path({str(worker_pid)!r}).write_text(str(worker.pid))\n"
        "deadline = time.monotonic() + 5\n"
        f"while not pathlib.Path({str(ready)!r}).exists() and time.monotonic() < deadline: time.sleep(0.01)\n"
    )
    try:
        assert runner.execute_evaluator([sys.executable, "-u", "-c", program], tmp_path / "evalscope.log") == 0
        pid = int(worker_pid.read_text())
        status = Path(f"/proc/{pid}/stat")
        deadline = time.monotonic() + 5
        while status.exists() and status.read_text().split(")", 1)[1].split()[0] != "Z":
            assert time.monotonic() < deadline, "evaluation worker survived process-group cleanup"
            time.sleep(0.01)
    finally:
        if worker_pid.exists():
            with suppress(ProcessLookupError):
                os.kill(int(worker_pid.read_text()), signal.SIGKILL)


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

    monkeypatch.setattr(runner, "execute_evaluator", fail)
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

    def evaluate(command, log):
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
        return 0

    monkeypatch.setattr(runner, "execute_evaluator", evaluate)
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
