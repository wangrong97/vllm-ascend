# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate NPU CI orchestration using CPU child processes and real JUnit files."""

import importlib.util
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
ENTRYPOINT = REPO_ROOT / "tools/tests/run_vq2a8_npu_ci.py"


@pytest.fixture
def assets(tmp_path):
    model = tmp_path / "model with spaces"
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
    (model / "model.safetensors.index.json").write_text('{"weight_map": {}}', encoding="utf-8")
    library, binary = tmp_path / "stage library.so", tmp_path / "mixed kernel.o"
    library.write_bytes(b"fixture")
    binary.write_bytes(b"fixture")
    results = tmp_path / "results with spaces"
    return NS(
        model=model,
        library=library,
        binary=binary,
        results=results,
        argv=[
            "--model",
            str(model),
            "--gs32-library",
            str(library),
            "--gs32-binary",
            str(binary),
            "--devices",
            "4,5",
            "--result-dir",
            str(results),
        ],
    )


@pytest.fixture
def driver():
    spec = importlib.util.spec_from_file_location("vq2a8_npu_ci_test_driver", ENTRYPOINT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_cli(assets, *arguments):
    return subprocess.run(
        [sys.executable, str(ENTRYPOINT), *assets.argv, *arguments],
        cwd=assets.results.parent,
        env={**os.environ, "PYTHONUTF8": "1"},
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=20,
        check=False,
    )


def read_report(assets):
    paths = list(assets.results.rglob("summary.json"))
    assert len(paths) == 1
    return json.loads(paths[0].read_text(encoding="utf-8")), paths[0].parent


@pytest.mark.parametrize("suite, stage_count", [("kernels", 9), ("pr", 11), ("nightly", 13)])
def test_dry_run_is_offline_and_plans_isolated_stages_with_explicit_devices(assets, suite, stage_count):
    result = run_cli(assets, "--suite", suite, "--devices", "4" if suite == "kernels" else "4,5", "--dry-run")
    assert result.returncode == 0, result.stdout + result.stderr
    plan = json.loads(result.stdout)
    stages = plan["stages"]
    assert len(stages) == stage_count
    assert len({stage["name"] for stage in stages}) == stage_count
    assert not assets.results.exists()
    for index, stage in enumerate(stages):
        command, environment = stage["command"], stage["environment"]
        assert isinstance(command, list) and command[:4] == [sys.executable, "-u", "-m", "pytest"]
        for flag, value in (
            ("--vq2a8-model", assets.model),
            ("--vq2a8-gs32-library", assets.library),
            ("--vq2a8-gs32-binary", assets.binary),
        ):
            assert command[command.index(flag) + 1] == str(value)
        assert environment["ASCEND_RT_VISIBLE_DEVICES"] == ("4" if index < 9 else "4,5")
        assert environment["VLLM_USE_V2_MODEL_RUNNER"] in ("0", "1")
    files = [next(item for item in stage["command"] if "tests/e2e/nightly/" in item) for stage in stages[:9]]
    assert len(set(files)) == 9
    if suite != "kernels":
        ep_stages = [stage for stage in stages if any("test_vq2a8_gs32_ep.py::" in item for item in stage["command"])]
        assert {stage["environment"]["VLLM_USE_V2_MODEL_RUNNER"] for stage in ep_stages} == {"0", "1"}
        assert all("--vq2a8-gs32-ep-skip-single-card" in stage["command"] for stage in ep_stages)
        assert all(
            any("test_vq2a8_gs32_ep.py" in item and "ep2" in item for item in stage["command"]) for stage in ep_stages
        )


def test_default_pr_can_select_only_v2_and_enable_ep1_reference(assets):
    result = run_cli(assets, "--model-runners", "v2", "--single-card-reference", "--dry-run")
    assert result.returncode == 0, result.stdout + result.stderr
    stages = json.loads(result.stdout)["stages"]
    assert len(stages) == 10
    assert stages[-1]["environment"]["VLLM_USE_V2_MODEL_RUNNER"] == "1"
    assert "--vq2a8-gs32-ep-skip-single-card" not in stages[-1]["command"]
    assert not assets.results.exists()


def test_kernels_suite_accepts_one_device_but_pr_requires_two(assets):
    result = run_cli(assets, "--devices", "7", "--suite", "kernels", "--dry-run")
    assert result.returncode == 0, result.stdout + result.stderr
    assert {stage["environment"]["ASCEND_RT_VISIBLE_DEVICES"] for stage in json.loads(result.stdout)["stages"]} == {"7"}
    rejected = run_cli(assets, "--devices", "7", "--suite", "pr", "--dry-run")
    assert rejected.returncode != 0
    assert not assets.results.exists()


@pytest.mark.parametrize(
    "arguments",
    [
        ("--devices", "4,4"),
        ("--devices", "-1,5"),
        ("--timeout", "0"),
        ("--timeout", "nan"),
        ("--model-runners", "v1", "v1"),
    ],
)
def test_invalid_configuration_is_rejected_before_output_creation(assets, arguments):
    result = run_cli(assets, *arguments, "--dry-run")
    assert result.returncode != 0
    assert not assets.results.exists()


@pytest.fixture
def child_plan(tmp_path, driver, monkeypatch):
    helper = tmp_path / "fake pytest child.py"
    trace = tmp_path / "children.jsonl"
    helper.write_text(
        textwrap.dedent("""
        import json
        import os
        import subprocess
        import sys
        import time
        from pathlib import Path

        mode, trace = sys.argv[1:3]
        junit = next(arg.split('=', 1)[1] for arg in sys.argv if arg.startswith('--junitxml='))
        with Path(trace).open('a', encoding='utf-8') as output:
            output.write(json.dumps({'mode': mode, 'pid': os.getpid(),
                'devices': os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
                'runner': os.environ.get('VLLM_USE_V2_MODEL_RUNNER'),
                'inherited': os.environ.get('VQ2A8_TEST_PARENT_SENTINEL')}) + '\\n')
        print('child stage: ' + mode, flush=True)
        if mode in ('sleep', 'tree', 'orphan'):
            if mode in ('tree', 'orphan'):
                grandchild = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
                Path(trace + '.grandchild').write_text(str(grandchild.pid))
            if mode != 'orphan':
                time.sleep(30)
        reports = {
            'pass': '<testsuite tests="1" failures="0" errors="0" skipped="0"><testcase name="ok"/></testsuite>',
            'failure': ('<testsuite tests="1" failures="1" errors="0" skipped="0">'
                '<testcase name="bad"><failure message="wrong"/></testcase></testsuite>'),
            'error': ('<testsuite tests="1" failures="0" errors="1" skipped="0">'
                '<testcase name="collection"><error message="collection failed"/></testcase></testsuite>'),
            'skip': ('<testsuite tests="1" failures="0" errors="0" skipped="1">'
                '<testcase name="missing"><skipped message="no NPU"/></testcase></testsuite>'),
            'xfail': ('<testsuite tests="1" failures="0" errors="0" skipped="1">'
                '<testcase name="expected"><skipped type="pytest.xfail"/></testcase></testsuite>'),
            'empty': '<testsuite tests="0" failures="0" errors="0" skipped="0"/>',
            'malformed': '<testsuite',
            'nonzero': '<testsuite tests="1" failures="0" errors="0" skipped="0"><testcase name="ok"/></testsuite>',
        }
        if mode != 'missing':
            report = reports['pass' if mode == 'orphan' else mode]
            Path(junit).write_text('<testsuites>' + report + '</testsuites>', encoding='utf-8')
        sys.exit(2 if mode == 'nonzero' else 0)
    """),
        encoding="utf-8",
    )
    monkeypatch.setattr(driver, "preflight", lambda *args, **kwargs: None)
    monkeypatch.setenv("VQ2A8_TEST_PARENT_SENTINEL", "preserved")

    def configure(*modes):
        stages = [
            {
                "name": f"stage-{index}-{mode}",
                "command": [sys.executable, str(helper), mode, str(trace)],
                "environment": {"ASCEND_RT_VISIBLE_DEVICES": "4,5", "VLLM_USE_V2_MODEL_RUNNER": str(index % 2)},
            }
            for index, mode in enumerate(modes)
        ]
        monkeypatch.setattr(driver, "build_plan", lambda *args, **kwargs: stages)
        return stages

    return NS(configure=configure, trace=trace, helper=helper)


@pytest.mark.parametrize("field", ["tests", "failures", "errors", "skipped"])
@pytest.mark.parametrize("element", ["testsuite", "testsuites"])
def test_junit_rejects_declared_counts_that_disagree_with_testcases(driver, tmp_path, field, element):
    declared = {"tests": 1, "failures": 0, "errors": 0, "skipped": 0}
    declared[field] += 1
    attributes = " ".join(f'{key}="{value}"' for key, value in declared.items())
    case = '<testcase name="ok"/>'
    inner = case if element == "testsuite" else f'<testsuite tests="1">{case}</testsuite>'
    report = tmp_path / "inconsistent.xml"
    report.write_text(f"<{element} {attributes}>{inner}</{element}>", encoding="utf-8")
    with pytest.raises(ValueError):
        driver.read_junit(report)


@pytest.mark.parametrize("interrupted", [False, True])
def test_preflight_failure_or_interrupt_preserves_report_without_starting_stages(
    assets, driver, child_plan, monkeypatch, interrupted
):
    child_plan.configure("pass", "pass")

    def reject(*args):
        if interrupted:
            raise KeyboardInterrupt
        raise RuntimeError("Invalid NPU environment")

    monkeypatch.setattr(driver, "preflight", reject)
    assert driver.main(assets.argv) == (130 if interrupted else 1)
    report, _ = read_report(assets)
    assert report["status"] == ("interrupted" if interrupted else "failed")
    assert report["error"]
    assert all(stage["status"] == "not_run" for stage in report["stages"])
    assert not child_plan.trace.exists()


def test_sigterm_handler_maps_to_keyboard_interrupt(driver):
    with pytest.raises(KeyboardInterrupt):
        driver.interrupt_handler(signal.SIGTERM, None)


@pytest.mark.parametrize(
    "failure",
    [
        "library_missing",
        "model_type",
        "source_version",
        "index_missing",
        "probe_failure",
        "config_null",
        "config_list",
        "quant_null",
        "quant_list",
        "quant_method",
    ],
)
def test_real_preflight_rejects_assets_or_probe_failure_without_running_pytest(assets, driver, monkeypatch, failure):
    monkeypatch.setattr(driver.sys, "platform", "linux")
    calls = []

    def probe(command, environment, log, timeout):
        calls.append(command)
        return 1

    monkeypatch.setattr(driver, "execute", probe)
    if failure == "library_missing":
        assets.library.unlink()
    elif failure in (
        "model_type",
        "source_version",
        "config_null",
        "config_list",
        "quant_null",
        "quant_list",
        "quant_method",
    ):
        path = assets.model / "config.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        if failure == "model_type":
            config["model_type"] = "unsupported"
        elif failure == "source_version":
            config["quantization_config"]["format_version"] = 1
        elif failure == "quant_method":
            config["quantization_config"]["quant_method"] = "unsupported"
        elif failure.startswith("config_"):
            config = None if failure.endswith("null") else []
        else:
            config["quantization_config"] = None if failure.endswith("null") else []
        path.write_text(json.dumps(config), encoding="utf-8")
    elif failure == "index_missing":
        (assets.model / "model.safetensors.index.json").unlink()
    assert driver.main(assets.argv) != 0
    report, _ = read_report(assets)
    assert report["status"] == "failed" and report["error"]
    assert all(stage["status"] == "not_run" for stage in report["stages"])
    assert len(calls) == (1 if failure == "probe_failure" else 0)
    assert all("pytest" not in command for command in calls)


def test_real_preflight_builds_valid_device_probe_without_importing_hardware(assets, driver, monkeypatch):
    monkeypatch.setattr(driver.sys, "platform", "linux")
    calls = []

    def probe(command, environment, log, timeout):
        calls.append((command, environment, log, timeout))
        code = command[command.index("-c") + 1]
        compile(code, "npu_environment_probe", "exec")
        return 0

    monkeypatch.setattr(driver, "execute", probe)
    assets.results.mkdir()
    driver.preflight(driver.parse_args(assets.argv), assets.results)
    assert len(calls) == 1
    command, environment, log, timeout = calls[0]
    assert command[0] == sys.executable and "pytest" not in command
    assert environment["ASCEND_RT_VISIBLE_DEVICES"] == "4,5"
    assert log == assets.results / "environment.log" and timeout > 0


def test_success_executes_sequential_children_and_saves_commands_counts_logs(assets, driver, child_plan):
    plan = child_plan.configure("pass", "pass")
    assert driver.main(assets.argv) == 0
    report, directory = read_report(assets)
    stages = report["stages"]
    assert [stage["name"] for stage in stages] == [stage["name"] for stage in plan]
    assert all(stage["status"] == "passed" for stage in stages)
    assert any("revision" in key or "sha" in key for key in report), "Record the client source revision"
    children = [json.loads(line) for line in child_plan.trace.read_text(encoding="utf-8").splitlines()]
    assert [child["runner"] for child in children] == ["0", "1"]
    assert all(child["devices"] == "4,5" and child["inherited"] == "preserved" for child in children)
    assert len({child["pid"] for child in children}) == 2
    for stage in stages:
        assert stage["counts"] == {"tests": 1, "passed": 1, "failures": 0, "errors": 0, "skipped": 0}
        assert stage["returncode"] == 0 and stage["elapsed_seconds"] >= 0
        assert Path(stage["junit"]).is_file()
        assert "child stage: pass" in Path(stage["log"]).read_text(encoding="utf-8")
        assert any(item.startswith("--basetemp=") for item in stage["command"])
        assert any(item.startswith("--junitxml=") for item in stage["command"])
    assert not list(directory.glob("*.tmp"))


@pytest.mark.parametrize("mode", ["failure", "error", "skip", "xfail", "empty", "malformed", "missing", "nonzero"])
def test_nonpassing_junit_or_child_exit_never_produces_green_and_stops(assets, driver, child_plan, mode):
    child_plan.configure("pass", mode, "pass")
    assert driver.main(assets.argv) != 0
    report, _ = read_report(assets)
    before, failed, later = report["stages"]
    assert before["status"] == "passed"
    assert failed["status"] != "passed" and failed.get("error")
    assert later["status"] == "not_run"
    children = [json.loads(line) for line in child_plan.trace.read_text(encoding="utf-8").splitlines()]
    assert [child["mode"] for child in children] == ["pass", mode]
    if mode in ("skip", "xfail"):
        assert failed["counts"]["skipped"] == 1 and failed["counts"]["passed"] == 0
    if mode == "error":
        assert failed["counts"]["errors"] == 1
    if mode == "failure":
        assert failed["counts"]["failures"] == 1
    if mode == "nonzero":
        assert failed["returncode"] == 2


@pytest.mark.parametrize(
    "failure, diagnostic",
    [
        ("arguments", "unrecognized arguments: --vq2a8-unknown-option"),
        ("conftest", "ImportError: vq2a8_conftest_import_sentinel"),
    ],
)
def test_real_pytest_startup_failure_preserves_diagnostics_and_stops(
    assets, driver, tmp_path, monkeypatch, capsys, failure, diagnostic
):
    isolated = tmp_path / "isolated pytest"
    isolated.mkdir()
    config = isolated / "pytest.ini"
    config.write_text("[pytest]\n", encoding="utf-8")
    test_file = isolated / "test_smoke.py"
    test_file.write_text("def test_pass():\n    assert True\n", encoding="utf-8")
    command = [
        sys.executable,
        "-u",
        "-m",
        "pytest",
        "-c",
        str(config),
        "--confcutdir",
        str(isolated),
        str(test_file),
    ]
    if failure == "arguments":
        command += ["--noconftest", "--vq2a8-unknown-option"]
    else:
        (isolated / "conftest.py").write_text("raise ImportError('vq2a8_conftest_import_sentinel')\n", encoding="utf-8")
    marker = tmp_path / "later-started"
    plan = [
        {
            "name": "startup-failure",
            "command": command,
            "environment": {"PYTHONUTF8": "1", "PYTEST_ADDOPTS": "", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
        },
        {
            "name": "must-not-start",
            "command": [
                sys.executable,
                "-c",
                f"from pathlib import Path; Path({str(marker)!r}).write_text('started')",
            ],
            "environment": {},
        },
    ]
    monkeypatch.setattr(driver, "preflight", lambda *args: None)
    monkeypatch.setattr(driver, "build_plan", lambda *args: plan)

    assert driver.main([*assets.argv, "--timeout", "20"]) == 1
    console = capsys.readouterr().out
    report, _ = read_report(assets)
    failed, later = report["stages"]
    assert report["status"] == "failed" and failed["status"] == "failed"
    assert failed["returncode"] == 4
    assert not Path(failed["junit"]).exists()
    assert Path(failed["junit"]).name in failed["junit_error"]
    assert "return code 4" in failed["error"]
    assert failed["log"] in failed["error"]
    assert diagnostic in failed["log_tail"]
    assert diagnostic in console
    assert "STAGE=startup-failure RETURN_CODE=4" in console
    assert f"STAGE=startup-failure LOG_TAIL_BEGIN LOG={failed['log']}" in console
    assert "STAGE=startup-failure LOG_TAIL_END" in console
    assert later["status"] == "not_run" and not marker.exists()


@pytest.mark.parametrize("shape", ["many_lines", "long_line"])
def test_failure_log_tail_is_bounded_and_keeps_the_latest_diagnostic(driver, tmp_path, shape):
    log = tmp_path / "child.log"
    prefix = "old diagnostic that must be omitted\n"
    if shape == "many_lines":
        content = prefix + "detail\n" * (driver.LOG_TAIL_LINES * 2)
    else:
        content = prefix + "x" * (driver.LOG_TAIL_BYTES * 2) + "\n"
    log.write_text(content + "latest diagnostic\n", encoding="utf-8")
    stage = {"log": str(log), "status": "failed", "error": "original error"}

    driver.capture_failure_log(stage)

    assert stage["status"] == "failed" and stage["error"] == "original error"
    assert stage["log_tail"].endswith("latest diagnostic")
    assert "old diagnostic" not in stage["log_tail"]
    assert len(stage["log_tail"].splitlines()) <= driver.LOG_TAIL_LINES
    assert len(stage["log_tail"].encode("utf-8")) <= driver.LOG_TAIL_BYTES
    assert "log_tail_error" not in stage


def test_failure_log_tail_replaces_invalid_utf8_without_hiding_the_cause(driver, tmp_path):
    log = tmp_path / "child.log"
    log.write_bytes(b"compiler output: \xff\xfe\nactual failure\n")
    stage = {"log": str(log), "status": "failed", "error": "original error"}

    driver.capture_failure_log(stage)

    assert stage["status"] == "failed" and stage["error"] == "original error"
    assert "\ufffd" in stage["log_tail"]
    assert stage["log_tail"].endswith("actual failure")
    assert "log_tail_error" not in stage


@pytest.mark.parametrize("unreadable", ["missing", "directory"])
def test_unreadable_failure_log_keeps_original_failure(driver, tmp_path, unreadable):
    log = tmp_path / "child.log"
    if unreadable == "directory":
        log.mkdir()
    stage = {"log": str(log), "status": "failed", "error": "original error"}

    driver.capture_failure_log(stage)

    assert stage["status"] == "failed" and stage["error"] == "original error"
    assert stage["log_tail_error"]
    assert "log_tail" not in stage


def test_timeout_terminates_child_and_prevents_later_stage(assets, driver, child_plan, capsys):
    child_plan.configure("sleep", "pass")
    assert driver.main([*assets.argv, "--timeout", "1"]) != 0
    report, _ = read_report(assets)
    first, later = report["stages"]
    assert first["status"] == "timeout" and first["error"]
    assert "child stage: sleep" in first["log_tail"]
    assert "child stage: sleep" in capsys.readouterr().out
    assert later["status"] == "not_run"
    children = [json.loads(line) for line in child_plan.trace.read_text(encoding="utf-8").splitlines()]
    assert [child["mode"] for child in children] == ["sleep"]


def test_keyboard_interrupt_terminates_child_and_preserves_summary(assets, driver, child_plan, monkeypatch):
    child_plan.configure("sleep", "pass")
    real_popen = subprocess.Popen
    children = []

    def spawn(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        # Revision lookup also uses Popen through subprocess.run; only the
        # staged child is interrupted, and cleanup uses its real wait method.
        if "fake pytest child.py" not in str(args):
            return process
        children.append(process)
        wait = process.wait
        interrupted = False

        def interrupt_once(*wait_args, **wait_kwargs):
            nonlocal interrupted
            if not interrupted:
                interrupted = True
                raise KeyboardInterrupt
            return wait(*wait_args, **wait_kwargs)

        process.wait = interrupt_once
        return process

    monkeypatch.setattr(driver.subprocess, "Popen", spawn)
    assert driver.main(assets.argv) == 130
    report, _ = read_report(assets)
    assert report["stages"][0]["status"] == "interrupted"
    assert report["stages"][1]["status"] == "not_run"
    assert len(children) == 1 and children[0].poll() is not None


def assert_process_stopped(pid):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        status = Path(f"/proc/{pid}/status")
        if status.exists() and "State:\tZ" in status.read_text():
            return
        time.sleep(0.01)
    pytest.fail("Descendant remains alive after stage cleanup")


@pytest.mark.skipif(os.name != "posix", reason="Real POSIX process-group cleanup requires POSIX")
@pytest.mark.parametrize("mode", ["tree", "orphan"])
def test_timeout_or_normal_parent_exit_kills_descendant_process_group(assets, driver, child_plan, mode):
    child_plan.configure(mode, "pass")
    descendant = Path(str(child_plan.trace) + ".grandchild")
    try:
        assert driver.main([*assets.argv, "--timeout", "1"]) == (0 if mode == "orphan" else 1)
        pid = int(descendant.read_text())
        assert_process_stopped(pid)
        report, _ = read_report(assets)
        assert report["stages"][1]["status"] == ("passed" if mode == "orphan" else "not_run")
    finally:
        if descendant.exists():
            with suppress(ProcessLookupError):
                os.kill(int(descendant.read_text()), signal.SIGKILL)


@pytest.mark.skipif(os.name != "posix", reason="Real SIGTERM delivery requires POSIX")
def test_sigterm_returns_130_saves_report_and_cleans_workers(assets, child_plan, tmp_path):
    # Run the orchestrator itself out of process so this test can deliver a
    # real termination signal without interrupting the outer pytest session.
    plan = child_plan.configure("tree", "pass")
    harness = tmp_path / "signal harness.py"
    harness.write_text(
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('npu_ci_signal', {str(ENTRYPOINT)!r})\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "module.preflight = lambda *args: None\n"
        f"module.build_plan = lambda *args: {plan!r}\n"
        "raise SystemExit(module.main(sys.argv[1:]))\n",
        encoding="utf-8",
    )
    descendant = Path(str(child_plan.trace) + ".grandchild")
    with (tmp_path / "signal.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, str(harness), *assets.argv], stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 5
            while not descendant.exists() and time.monotonic() < deadline:
                assert process.poll() is None
                time.sleep(0.01)
            assert descendant.exists(), "Stage did not start its descendant"
            os.kill(process.pid, signal.SIGTERM)
            assert process.wait(timeout=15) == 130
            report, _ = read_report(assets)
            assert report["status"] == "interrupted"
            assert [stage["status"] for stage in report["stages"]] == ["interrupted", "not_run"]
            assert_process_stopped(int(descendant.read_text()))
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            if child_plan.trace.exists():
                child = json.loads(child_plan.trace.read_text(encoding="utf-8").splitlines()[0])
                with suppress(ProcessLookupError):
                    os.killpg(child["pid"], signal.SIGKILL)
