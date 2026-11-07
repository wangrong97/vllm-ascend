# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise diagnostic evidence and worker cleanup without NPU hardware."""

import importlib.util
import json
import os
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ENTRYPOINT = Path(__file__).resolve().parents[3] / "tools/tests/check_vq2a8_hccl.py"


@pytest.fixture
def driver():
    spec = importlib.util.spec_from_file_location("vq2a8_hccl_check_test_driver", ENTRYPOINT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def passed_report(args, rank):
    return {
        "status": "passed",
        "rank": rank,
        "local_rank": rank,
        "world_size": 2,
        "visibility_entry": args.devices.split(",")[rank],
        "requested_visible_devices": args.devices,
        "ascend_rt_visible_devices": args.devices,
        "checks": [
            {"operation": "all_reduce", "group": "world", "dtype": "float32", "shape": [1]},
            *(
                {"operation": "all_gather_into_tensor", "group": group, "dtype": dtype, "shape": [row, args.columns]}
                for group in ("world", "tp", "ep")
                for dtype in ("float32", "bfloat16")
                for row in args.rows
            ),
        ],
    }


@pytest.mark.parametrize(
    "arguments",
    [
        ["--devices", "0"],
        ["--devices", "0,00"],
        ["--devices", "-1,0"],
        ["--devices", "0,x"],
        ["--devices", "0,1", "--timeout", "nan"],
        ["--devices", "0,1", "--timeout", "0"],
        ["--devices", "0,1", "--rows", "1", "1"],
        ["--devices", "0,1", "--columns", "0"],
        ["--devices", "0,1", "--worker"],
        ["--devices", "0,1", "--rank", "0"],
    ],
)
def test_invalid_arguments_fail_before_launch(driver, arguments):
    with pytest.raises(SystemExit) as error:
        driver.parse_args(arguments)
    assert error.value.code == 2


def test_dry_run_needs_no_site_packages_and_creates_no_files(tmp_path):
    results = tmp_path / "results with spaces"
    completed = subprocess.run(
        [sys.executable, "-S", str(ENTRYPOINT), "--devices", "04,5", "--result-dir", str(results), "--dry-run"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stderr
    assert "ASCEND_RT_VISIBLE_DEVICES=4,5" in completed.stdout
    assert "--rank 0" in completed.stdout and "--rank 1" in completed.stdout
    assert not results.exists()


@pytest.mark.parametrize(
    "case", ["complete", "missing", "invalid_json", "failed", "wrong_rank", "wrong_device", "partial"]
)
def test_acceptance_requires_every_rank_and_check(driver, tmp_path, case):
    args = driver.parse_args(["--devices", "4,5"])
    for rank in range(2):
        report = passed_report(args, rank)
        if rank == 1:
            if case == "missing":
                continue
            if case == "invalid_json":
                (tmp_path / "rank-1.json").write_text("{", encoding="utf-8")
                continue
            if case == "failed":
                report["status"] = "failed"
            if case == "wrong_rank":
                report["rank"] = 0
            if case == "wrong_device":
                report["visibility_entry"] = "0"
            if case == "partial":
                report["checks"].pop()
        driver.write_json(tmp_path / f"rank-{rank}.json", report)
    _, errors = driver.read_rank_reports(tmp_path, args)
    assert bool(errors) is (case != "complete")


@pytest.mark.parametrize("root_kind", ["valid", "invalid", "directory", "missing"])
def test_runtime_paths_report_metadata_without_topology_contents(driver, tmp_path, monkeypatch, root_kind):
    rootinfo, route, hixlep = (tmp_path / name for name in ("rootinfo.json", "route.conf", "hixlep"))
    if root_kind == "valid":
        rootinfo.write_text('{"secret_topology": 7}', encoding="utf-8")
    elif root_kind == "invalid":
        rootinfo.write_text("secret_topology", encoding="utf-8")
    elif root_kind == "directory":
        rootinfo.mkdir()
    route.write_text("secret_topology", encoding="utf-8")
    hixlep.mkdir()
    monkeypatch.setattr(driver, "RUNTIME_PATHS", tuple(map(str, (rootinfo, route, hixlep))))
    report = driver.inspect_runtime_paths()
    assert report[0]["json_readable"] is (root_kind == "valid")
    assert report[0]["type"] == ("file" if root_kind in ("valid", "invalid") else root_kind)
    assert report[1]["type"] == "file" and report[2]["type"] == "directory"
    assert "secret_topology" not in json.dumps(report)


@pytest.fixture
def fake_workers(driver, tmp_path, monkeypatch):
    script = tmp_path / "fake worker.py"
    script.write_text(
        textwrap.dedent("""
            import json, os, signal, sys, time
            from pathlib import Path
            directory, rank, mode, payload = sys.argv[1:]
            assert os.environ['ASCEND_RT_VISIBLE_DEVICES'] == '4,5'
            print('READY rank=' + rank, flush=True)
            if mode == 'hang':
                if os.name == 'posix':
                    signal.signal(signal.SIGTERM, signal.SIG_IGN)
                time.sleep(60)
            elif mode == 'fail':
                sys.exit(7)
            elif mode != 'missing':
                Path(directory, 'rank-' + rank + '.json').write_text(payload, encoding='utf-8')
        """),
        encoding="utf-8",
    )
    modes = ["complete", "complete"]

    def command(args, directory, rank):
        return [
            sys.executable,
            "-u",
            str(script),
            str(directory),
            str(rank),
            modes[rank],
            json.dumps(passed_report(args, rank)),
        ]

    monkeypatch.setattr(driver, "build_worker_command", command)
    monkeypatch.setattr(driver, "TERMINATE_GRACE_SECONDS", 0.1)
    processes = []
    popen = subprocess.Popen

    def tracked_popen(*args, **kwargs):
        process = popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(driver.subprocess, "Popen", tracked_popen)
    return modes, processes


@pytest.mark.parametrize(
    "mode,status", [("complete", "passed"), ("missing", "failed"), ("fail", "failed"), ("hang", "timeout")]
)
def test_real_child_exit_and_reports_determine_status(driver, tmp_path, fake_workers, mode, status):
    modes, processes = fake_workers
    modes[1] = mode
    results = tmp_path / "results with spaces"
    result = driver.main(["--devices", "4,5", "--timeout", "2", "--result-dir", str(results)])
    summaries = list(results.glob("*/summary.json"))
    assert len(summaries) == 1
    summary = json.loads(summaries[0].read_text(encoding="utf-8"))
    assert summary["status"] == status
    assert result == (0 if status == "passed" else 1)
    assert len(processes) == 2 and all(process.poll() is not None for process in processes)
    log = Path(summary["log"]).read_text(encoding="utf-8")
    assert "READY rank=1" in log


def test_partial_launch_failure_stops_already_started_worker(driver, tmp_path, fake_workers, monkeypatch):
    modes, processes = fake_workers
    modes[0] = "hang"
    popen = driver.subprocess.Popen

    def fail_second_launch(*args, **kwargs):
        if processes:
            raise OSError("second worker could not start")
        return popen(*args, **kwargs)

    monkeypatch.setattr(driver.subprocess, "Popen", fail_second_launch)
    args = driver.parse_args(["--devices", "4,5"])
    with pytest.raises(OSError, match="second worker"):
        driver.execute_workers(args, tmp_path, tmp_path / "workers.log")
    assert len(processes) == 1 and processes[0].poll() is not None


def test_existing_filestore_is_not_reused(driver, tmp_path, fake_workers):
    _, processes = fake_workers
    args = driver.parse_args(["--devices", "4,5"])
    (tmp_path / "rendezvous.store").write_text("previous run", encoding="utf-8")
    with pytest.raises(ValueError, match="must not already exist"):
        driver.execute_workers(args, tmp_path, tmp_path / "workers.log")
    assert not processes


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group signal handling")
def test_repeated_interrupt_does_not_abort_cleanup(driver, tmp_path, fake_workers, monkeypatch):
    modes, processes = fake_workers
    modes[:] = ["hang", "hang"]
    original_sleep = driver.time.sleep
    sent = False

    def interrupt_during_cleanup(seconds):
        nonlocal sent
        if not sent and signal.getsignal(signal.SIGTERM) == signal.SIG_IGN:
            sent = True
            os.kill(os.getpid(), signal.SIGTERM)
            os.kill(os.getpid(), signal.SIGINT)
        original_sleep(seconds)

    monkeypatch.setattr(driver.time, "sleep", interrupt_during_cleanup)
    previous = signal.getsignal(signal.SIGTERM)
    result = driver.main(["--devices", "4,5", "--timeout", "2", "--result-dir", str(tmp_path / "results")])
    assert result == 1 and sent
    assert all(process.poll() is not None for process in processes)
    assert signal.getsignal(signal.SIGTERM) == previous
