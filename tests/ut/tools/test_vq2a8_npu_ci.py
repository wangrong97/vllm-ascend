# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate NPU CI orchestration using CPU child processes and real JUnit files."""

import ast
import importlib.util
import io
import json
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from concurrent.futures import ThreadPoolExecutor
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


@pytest.mark.parametrize("returncode", [0, 7])
def test_child_output_is_live_before_exit_and_preserved_in_log(driver, tmp_path, monkeypatch, returncode):
    release = tmp_path / "release-child"
    log = tmp_path / "stream.log"
    observed = threading.Event()

    class Console(io.StringIO):
        def write(self, text):
            count = super().write(text)
            if "STDOUT-READY" in self.getvalue() and "STDERR-READY" in self.getvalue():
                observed.set()
            return count

    console = Console()
    monkeypatch.setattr(sys, "stdout", console)
    code = textwrap.dedent("""
        import os, sys, time
        from pathlib import Path
        os.write(1, b'STDOUT-READY\\r')
        os.write(2, b'STDERR-READY\\n')
        word = '\u4e2d'.encode('utf-8')
        os.write(1, word[:2])
        deadline = time.monotonic() + 10
        while not Path(sys.argv[1]).exists():
            if time.monotonic() >= deadline:
                raise RuntimeError('Parent did not relay output before child exit')
            time.sleep(0.01)
        os.write(1, word[2:] + b'-FINAL')
        sys.exit(int(sys.argv[2]))
    """)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            driver.execute, [sys.executable, "-u", "-c", code, str(release), str(returncode)], {}, log, 15
        )
        try:
            assert observed.wait(5), "Output was buffered until child exit"
            assert not future.done(), "Child must still be waiting for the parent"
            assert b"STDOUT-READY\rSTDERR-READY\n" in log.read_bytes()
            assert "\ufffd" not in console.getvalue()
        finally:
            release.touch()
        assert future.result(timeout=5) == returncode
    expected = "STDOUT-READY\rSTDERR-READY\n\u4e2d-FINAL"
    assert log.read_bytes() == expected.encode("utf-8")
    assert console.getvalue() == expected + "\n"


def test_closed_terminal_does_not_lose_child_log_or_returncode(driver, tmp_path, monkeypatch):
    class ClosedConsole:
        def write(self, text):
            raise BrokenPipeError("terminal disconnected")

    monkeypatch.setattr(sys, "stdout", ClosedConsole())
    log = tmp_path / "closed-terminal.log"
    command = [sys.executable, "-u", "-c", "import os, sys; os.write(2, b'child error'); sys.exit(7)"]
    assert driver.execute(command, {}, log, 10) == 7
    assert log.read_bytes() == b"child error"


@pytest.mark.parametrize("stage_name", ["activation", "ep2-v2", "recovery-v2"])
def test_real_plan_discovers_e2e_conftest_and_passes_custom_options(assets, driver, tmp_path, monkeypatch, stage_name):
    repository = tmp_path / "isolated repository"
    nightly = repository / driver.TEST_ROOT
    nightly.mkdir(parents=True)
    (repository / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    # Reuse the real option declarations without importing NPU-dependent fixtures.
    conftest_source = (REPO_ROOT / "tests/e2e/conftest.py").read_text(encoding="utf-8")
    option_hook = next(
        node
        for node in ast.parse(conftest_source).body
        if isinstance(node, ast.FunctionDef) and node.name == "pytest_addoption"
    )
    (nightly.parent / "conftest.py").write_text(
        "import os\nimport pytest\n\n" + ast.get_source_segment(conftest_source, option_hook) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(driver, "REPO_ROOT", repository)
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    monkeypatch.setenv("PYTHONUTF8", "1")
    monkeypatch.delenv("PYTEST_PLUGINS", raising=False)
    args = driver.parse_args([*assets.argv, "--suite", "nightly", "--model-runners", "v2", "--timeout", "20"])
    stage = next(stage for stage in driver.build_plan(args) if stage["name"] == stage_name)
    target = next(argument for argument in stage["command"] if argument.startswith(f"{driver.TEST_ROOT}/"))
    relative_path, _, node_id = target.partition("::")
    function = node_id.split("[", 1)[0] if node_id else "test_custom_options"
    parameter = '@pytest.mark.parametrize("parallelism", ["ep2"])\n' if node_id else ""
    parameters = "request, parallelism" if node_id else "request"
    (repository / relative_path).write_text(
        "import os\nimport pytest\n\n"
        + parameter
        + f"def {function}({parameters}):\n"
        + f"    assert request.config.getoption('--vq2a8-model') == {str(assets.model)!r}\n"
        + f"    assert request.config.getoption('--vq2a8-gs32-library') == {str(assets.library)!r}\n"
        + f"    assert request.config.getoption('--vq2a8-gs32-binary') == {str(assets.binary)!r}\n"
        + f"    assert request.config.getoption('--vq2a8-gs32-ep-skip-single-card') is {stage_name == 'ep2-v2'}\n"
        + f"    assert os.environ['ASCEND_RT_VISIBLE_DEVICES'] == {'4' if stage_name == 'activation' else '4,5'!r}\n"
        + f"    assert os.environ['VLLM_USE_V2_MODEL_RUNNER'] == {'0' if stage_name == 'activation' else '1'!r}\n",
        encoding="utf-8",
    )
    assets.results.mkdir()

    # Keep command ordering and directory-based conftest discovery exactly as planned.
    driver.run_stage(stage, args, assets.results, lambda: None)

    log = Path(stage["log"]).read_text(encoding="utf-8")
    assert stage["returncode"] == 0, log
    assert stage["status"] == "passed", log
    assert stage["counts"] == {"tests": 1, "passed": 1, "failures": 0, "errors": 0, "skipped": 0}
    assert Path(stage["junit"]).is_file()


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
        ("--allow-missing-checkpoint-pairs", "unknown_pair"),
        ("--allow-missing-checkpoint-pairs", "mxfp4_to_vq2", "mxfp4_to_vq2"),
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


@pytest.fixture
def checkpoint_gap_plan(tmp_path, driver, monkeypatch):
    """Produce real pytest JUnit records for checkpoint gaps without loading an NPU."""
    repository = tmp_path / "checkpoint coverage repository"
    nightly = repository / "tests/e2e/nightly"
    nightly.mkdir(parents=True)
    config = repository / "pytest.ini"
    config.write_text("[pytest]\n", encoding="utf-8")
    monkeypatch.setattr(driver, "REPO_ROOT", repository)
    monkeypatch.setattr(driver, "preflight", lambda *args: None)
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    monkeypatch.setenv("PYTEST_ADDOPTS", "")
    monkeypatch.setenv("PYTHONUTF8", "1")
    monkeypatch.delenv("PYTEST_PLUGINS", raising=False)

    def configure(mode="missing", *, pair="mxfp4_to_vq2", stage_name="batched", later_failure=False):
        filename = "test_other.py" if mode == "wrong_class" else "test_vq2a8_gs32_batched.py"
        function = "test_unrelated" if mode == "wrong_function" else "test_gs32_real_experts_fusion_compare"
        executions = ("fused", "unexpected") if mode == "wrong_variant" else ("fused", "full_graph")
        fusion_reason = (
            f"No real {pair.replace('_to_', ' -> ')} expert exists in this checkpoint; combination was not tested."
        )
        prefill_reason = (
            f"No real {pair.replace('_to_', ' -> ')} expert exists; this prefill combination was not tested."
        )
        source = (
            "import pytest\n\n"
            + f"@pytest.mark.parametrize('execution', {executions!r})\n"
            + f"@pytest.mark.parametrize('pair', [{pair!r}])\n"
            + f"def {function}(pair, execution):\n"
        )
        if mode == "plain":
            source += "    assert True\n"
        elif mode == "xfail":
            source += f"    pytest.xfail({fusion_reason!r})\n"
        elif mode == "wrong_reason":
            source += "    pytest.skip('no NPU available')\n"
        else:
            source += f"    pytest.skip({fusion_reason!r})\n"
        source += (
            "\n@pytest.mark.parametrize('fusion', ('none', 'swiglu_quant'))\n"
            + f"@pytest.mark.parametrize('pair', [{pair!r}])\n"
            + "def test_gs32_real_mixed_ffn_device_prefill_compare_and_timing(pair, fusion):\n"
            + ("    assert True\n" if mode == "plain" else f"    pytest.skip({prefill_reason!r})\n")
        )
        if mode != "all_skip":
            source += "\ndef test_executed_witness():\n    assert True\n"
        if mode == "other_skip":
            source += "\ndef test_other_skip():\n    pytest.skip('no NPU available')\n"
        (nightly / filename).write_text(source, encoding="utf-8")
        if mode == "nonzero":
            (repository / "conftest.py").write_text(
                "def pytest_sessionfinish(session):\n    session.exitstatus = 7\n", encoding="utf-8"
            )
        (nightly / "test_later.py").write_text(
            f"def test_later():\n    assert {not later_failure!r}\n", encoding="utf-8"
        )
        (nightly / "test_last.py").write_text("def test_last():\n    assert True\n", encoding="utf-8")
        stages = [
            {
                "name": name,
                "command": [
                    sys.executable,
                    "-u",
                    "-m",
                    "pytest",
                    "-q",
                    "-c",
                    str(config),
                    "-o",
                    "addopts=",
                    "-o",
                    "xfail_strict=True",
                    f"tests/e2e/nightly/{test}",
                ],
                "environment": {"ASCEND_RT_VISIBLE_DEVICES": "4,5", "VLLM_USE_V2_MODEL_RUNNER": "0"},
            }
            for name, test in ((stage_name, filename), ("ep2-v1", "test_later.py"), ("ep2-v2", "test_last.py"))
        ]
        monkeypatch.setattr(driver, "build_plan", lambda *args: stages)
        return stages

    return configure


def test_checkpoint_gap_is_strictly_rejected_by_default(assets, driver, checkpoint_gap_plan):
    checkpoint_gap_plan()
    assert driver.main(assets.argv) == 1
    report, _ = read_report(assets)
    assert report["allowed_missing_checkpoint_pairs"] == []
    assert report["coverage_complete"] is False
    assert report["status"] == "failed"
    assert [stage["status"] for stage in report["stages"]] == ["failed", "not_run", "not_run"]
    stage = report["stages"][0]
    assert stage["returncode"] == 0
    assert stage["counts"] == {"tests": 5, "passed": 1, "failures": 0, "errors": 0, "skipped": 4}
    assert len(stage["skipped_tests"]) == 4
    assert {case["type"] for case in stage["skipped_tests"]} == {"pytest.skip"}


@pytest.mark.parametrize("pair", ["vq2_to_vq2", "vq2_to_mxfp4", "mxfp4_to_vq2", "mxfp4_to_mxfp4"])
def test_explicit_checkpoint_gap_continues_and_preserves_real_skip_coverage(assets, driver, checkpoint_gap_plan, pair):
    checkpoint_gap_plan(pair=pair)
    assert driver.main([*assets.argv, "--allow-missing-checkpoint-pairs", pair]) == 0
    report, _ = read_report(assets)
    assert report["status"] == "passed_with_coverage_gaps"
    assert report["allowed_missing_checkpoint_pairs"] == [pair]
    assert report["coverage_complete"] is False
    assert [stage["status"] for stage in report["stages"]] == ["passed_with_coverage_gaps", "passed", "passed"]
    stage = report["stages"][0]
    assert stage["counts"] == {"tests": 5, "passed": 1, "failures": 0, "errors": 0, "skipped": 4}
    assert len(stage["coverage_gaps"]) == len(stage["skipped_tests"]) == len(report["coverage_gaps"]) == 4
    expected_names = {
        f"test_gs32_real_experts_fusion_compare[{pair}-{execution}]" for execution in ("fused", "full_graph")
    } | {
        f"test_gs32_real_mixed_ffn_device_prefill_compare_and_timing[{pair}-{fusion}]"
        for fusion in ("none", "swiglu_quant")
    }
    assert {case["name"] for case in stage["skipped_tests"]} == expected_names
    assert {case["classname"] for case in stage["skipped_tests"]} == {"tests.e2e.nightly.test_vq2a8_gs32_batched"}
    assert {gap["nodeid"] for gap in stage["coverage_gaps"]} == {
        f"tests/e2e/nightly/test_vq2a8_gs32_batched.py::{name}" for name in expected_names
    }
    assert all(gap["pair"] == pair and "was not tested" in gap["reason"] for gap in stage["coverage_gaps"])
    assert report["coverage_gaps"] == [
        {"stage": "batched", "model": str(assets.model), **gap} for gap in stage["coverage_gaps"]
    ]


@pytest.mark.parametrize(
    "mode,stage_name,allowed",
    [
        ("other_skip", "batched", "mxfp4_to_vq2"),
        ("xfail", "batched", "mxfp4_to_vq2"),
        ("wrong_reason", "batched", "mxfp4_to_vq2"),
        ("wrong_class", "batched", "mxfp4_to_vq2"),
        ("wrong_function", "batched", "mxfp4_to_vq2"),
        ("wrong_variant", "batched", "mxfp4_to_vq2"),
        ("all_skip", "batched", "mxfp4_to_vq2"),
        ("nonzero", "batched", "mxfp4_to_vq2"),
        ("missing", "grouped", "mxfp4_to_vq2"),
        ("missing", "batched", "vq2_to_mxfp4"),
    ],
)
def test_checkpoint_gap_opt_in_does_not_allow_other_nonpassing_results(
    assets, driver, checkpoint_gap_plan, mode, stage_name, allowed
):
    checkpoint_gap_plan(mode, stage_name=stage_name)
    assert driver.main([*assets.argv, "--allow-missing-checkpoint-pairs", allowed]) == 1
    report, _ = read_report(assets)
    assert report["status"] == "failed" and report["coverage_complete"] is False
    assert [stage["status"] for stage in report["stages"]] == ["failed", "not_run", "not_run"]
    stage = report["stages"][0]
    assert stage["error"]
    if mode == "all_skip":
        assert stage["counts"]["passed"] == 0 and stage["counts"]["skipped"] == 4
    if mode == "nonzero":
        assert stage["returncode"] == 7
    if mode == "xfail":
        assert "pytest.xfail" in {case["type"] for case in stage["skipped_tests"]}


def test_downstream_failure_takes_priority_and_preserves_earlier_checkpoint_gaps(assets, driver, checkpoint_gap_plan):
    checkpoint_gap_plan(later_failure=True)
    assert driver.main([*assets.argv, "--allow-missing-checkpoint-pairs", "mxfp4_to_vq2"]) == 1
    report, _ = read_report(assets)
    assert report["status"] == "failed" and report["coverage_complete"] is False
    assert [stage["status"] for stage in report["stages"]] == ["passed_with_coverage_gaps", "failed", "not_run"]
    assert len(report["coverage_gaps"]) == 4
    assert report["stages"][1]["counts"]["failures"] == 1


@pytest.mark.parametrize("allow_missing", [False, True])
def test_no_checkpoint_gaps_retains_full_pass_semantics(assets, driver, checkpoint_gap_plan, allow_missing):
    checkpoint_gap_plan("plain")
    extra = ["--allow-missing-checkpoint-pairs", "mxfp4_to_vq2"] if allow_missing else []
    assert driver.main([*assets.argv, *extra]) == 0
    report, _ = read_report(assets)
    assert report["status"] == "passed" and report["coverage_complete"] is True
    assert report["coverage_gaps"] == []
    assert all(stage["status"] == "passed" for stage in report["stages"])
    assert all(stage["skipped_tests"] == [] and stage["coverage_gaps"] == [] for stage in report["stages"])


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
