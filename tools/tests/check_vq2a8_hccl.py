#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check HCCL collectives without loading vLLM or model weights.

Run in the same container, Python environment and visible devices as the failing
server. A pass establishes only these collectives, not GS32 model correctness.
The launcher uses the standard library; NPU imports happen in worker processes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.vq2a8_live_log import LiveChildLog  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_PATHS = ("/etc/hccl_rootinfo.json", "/lib/route.conf", "/etc/hixlep")
GROUP_NAMES = ("world", "tp", "ep")
DTYPE_NAMES = ("float32", "bfloat16")
TERMINATE_GRACE_SECONDS = 10
POLL_SECONDS = 0.05
SCOPE = "HCCL collective diagnostics only; does not certify GS32, vLLM model execution or performance."


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--devices", required=True, help="At least two distinct ASCEND_RT_VISIBLE_DEVICES entries, e.g. 4,5"
    )
    parser.add_argument("--timeout", type=float, default=180, help="Total worker-process wall timeout in seconds")
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 64])
    parser.add_argument("--columns", type=int, default=131072, help="Tensor columns, e.g. one TP vocabulary shard")
    parser.add_argument("--result-dir", type=Path, default=REPO_ROOT / "benchmark-results/hccl-check")
    parser.add_argument("--dry-run", action="store_true", help="Print plan only; no NPU initialization or files")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--rank", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--world-size", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--rendezvous", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    devices = args.devices.split(",")
    if len(devices) < 2 or any(not value.isascii() or not value.isdecimal() for value in devices):
        parser.error("--devices requires at least two comma-separated nonnegative integers")
    devices = [str(int(value)) for value in devices]
    if len(set(devices)) != len(devices):
        parser.error("--devices must contain distinct visibility entries")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    if args.columns < 1 or any(row < 1 for row in args.rows) or len(set(args.rows)) != len(args.rows):
        parser.error("--columns must be positive and --rows must contain distinct positive integers")
    if args.worker and args.dry_run:
        parser.error("--worker cannot be combined with --dry-run")
    if args.worker:
        if args.world_size != len(devices) or args.rank is None or not 0 <= args.rank < args.world_size:
            parser.error("--worker requires --world-size matching --devices and a valid --rank")
        if args.rendezvous is None:
            parser.error("--worker requires --rendezvous")
        args.rendezvous = args.rendezvous.expanduser().resolve()
    elif any(value is not None for value in (args.rank, args.world_size, args.rendezvous)):
        parser.error("--rank, --world-size and --rendezvous are worker-only arguments")
    args.devices = ",".join(devices)
    args.result_dir = args.result_dir.expanduser().resolve()
    return args


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def inspect_runtime_paths():
    """Describe deployment assets without disclosing their topology contents."""
    results = []
    for name in RUNTIME_PATHS:
        path = Path(name)
        record = {"path": name, "type": "unknown", "is_symlink": path.is_symlink(), "json_readable": None}
        try:
            record["type"] = "file" if path.is_file() else "directory" if path.is_dir() else "missing"
            record["readable"] = path.exists() and os.access(path, os.R_OK)
            if name == RUNTIME_PATHS[0]:
                record["json_readable"] = False
                if path.is_file():
                    with path.open(encoding="utf-8") as stream:
                        json.load(stream)
                    record["json_readable"] = True
        except (OSError, ValueError) as error:
            # Exception text can contain source text; record only its type.
            record["inspection_error"] = type(error).__name__
        results.append(record)
    return results


def build_worker_command(args, directory, rank):
    world_size = len(args.devices.split(","))
    return [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--worker",
        "--rank",
        str(rank),
        "--world-size",
        str(world_size),
        "--rendezvous",
        str(directory / "rendezvous.store"),
        "--devices",
        args.devices,
        "--timeout",
        str(args.timeout),
        "--rows",
        *map(str, args.rows),
        "--columns",
        str(args.columns),
        "--result-dir",
        str(directory),
    ]


def run_worker(args):
    """Map each single-node rank to a logical device in the requested visibility."""
    rank = args.rank
    local_rank = rank
    world_size = args.world_size
    visibility_entries = args.devices.split(",")
    report_path = args.result_dir / f"rank-{rank}.json"
    started = time.monotonic()
    report = {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "requested_visible_devices": args.devices,
        "ascend_rt_visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "visibility_entries": visibility_entries,
        "runtime_paths": inspect_runtime_paths(),
        "checks": [],
        "status": "running",
        "scope": SCOPE,
    }

    def event(step, phase, **details):
        record = {
            "utc": datetime.now(UTC).isoformat(),
            "elapsed_seconds": time.monotonic() - started,
            "rank": rank,
            "local_rank": local_rank,
            "step": step,
            "phase": phase,
            **details,
        }
        report["last_event"] = record
        print("HCCL_CHECK_EVENT=" + json.dumps(record, ensure_ascii=False, allow_nan=False), flush=True)
        write_json(report_path, report)

    dist = None
    try:
        event("imports", "enter")
        # Workers own device initialization; the launcher must remain stdlib-only.
        import torch
        import torch.distributed as dist
        import torch_npu

        if world_size != len(visibility_entries) or not 0 <= local_rank < world_size:
            raise ValueError("Worker rank inventory does not match --devices")
        if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != args.devices:
            raise ValueError("Worker visibility does not match requested visibility entries")
        if not torch.npu.is_available() or torch.npu.device_count() != world_size:
            raise RuntimeError("Expected one visible NPU per worker")
        torch.npu.set_device(local_rank)
        report.update(
            torch_version=torch.__version__,
            torch_npu_version=getattr(torch_npu, "__version__", "unknown"),
            logical_device=local_rank,
            visibility_entry=visibility_entries[local_rank],
            device_name=torch.npu.get_device_name(local_rank),
        )
        event("device", "selected", visibility_entry=visibility_entries[local_rank], device_name=report["device_name"])
        timeout = timedelta(seconds=args.timeout)
        event("init_process_group", "enter")
        store = dist.FileStore(str(args.rendezvous), world_size)
        dist.init_process_group(backend="hccl", store=store, rank=rank, world_size=world_size, timeout=timeout)
        event("init_process_group", "returned")

        # A first scalar collective distinguishes basic connectivity from the
        # larger all-gather and from creating multiple same-membership groups.
        step = "all_reduce/world"
        event(step, "enter", dtype="float32", shape=[1])
        value = torch.tensor([rank + 1], dtype=torch.float32, device=f"npu:{local_rank}")
        dist.all_reduce(value)
        event(step, "submitted", dtype="float32", shape=[1])
        torch.npu.synchronize()
        event(step, "synchronized", dtype="float32", shape=[1])
        expected = world_size * (world_size + 1) / 2
        if value.item() != expected:
            raise AssertionError(f"AllReduce result differs from the expected rank sum {expected}")
        report["checks"].append({"operation": "all_reduce", "group": "world", "dtype": "float32", "shape": [1]})
        del value

        groups = [("world", None)]
        members = list(range(world_size))
        for name in GROUP_NAMES[1:]:
            event(f"new_group/{name}", "enter", members=members)
            group = dist.new_group(ranks=members, backend="hccl", timeout=timeout)
            groups.append((name, group))
            event(f"new_group/{name}", "returned", members=members)

        for name, group in groups:
            for dtype_name in DTYPE_NAMES:
                dtype = getattr(torch, dtype_name)
                for rows in args.rows:
                    shape = [rows, args.columns]
                    details = {"group": name, "dtype": dtype_name, "shape": shape}
                    step = f"all_gather/{name}/{dtype_name}/{rows}x{args.columns}"
                    event(step, "enter", **details)
                    value = torch.full(shape, rank + 1, dtype=dtype, device=f"npu:{local_rank}")
                    output = torch.empty((world_size * rows, args.columns), dtype=dtype, device=value.device)
                    dist.all_gather_into_tensor(output, value, group=group)
                    event(step, "submitted", **details)
                    torch.npu.synchronize()
                    event(step, "synchronized", **details)
                    for peer in range(world_size):
                        part = output[peer * rows : (peer + 1) * rows]
                        if not torch.all(part == peer + 1).item():
                            raise AssertionError(
                                f"AllGather {name}/{dtype_name}/{shape} has incorrect rank-{peer} rows"
                            )
                    report["checks"].append({"operation": "all_gather_into_tensor", **details})
                    del value, output, part
                    event(step, "validated", **details)
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        traceback.print_exc()
        write_json(report_path, report)
    finally:
        if dist is not None and dist.is_initialized():
            try:
                event("destroy_process_group", "enter")
                dist.destroy_process_group()
                event("destroy_process_group", "returned")
            except BaseException as error:
                report.update(status="failed", cleanup_error=f"{type(error).__name__}: {error}")
                traceback.print_exc()
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(report_path, report)
    print(f"HCCL_CHECK_RANK={rank} STATUS={report['status']}", flush=True)
    return 0 if report["status"] == "passed" else 1


def read_rank_reports(directory, args):
    """Require complete numerical evidence from every expected worker."""
    reports = []
    errors = []
    visibility_entries = args.devices.split(",")
    expected_checks = {
        ("all_reduce", "world", "float32", (1,)),
        *(
            ("all_gather_into_tensor", group, dtype, (rows, args.columns))
            for group in GROUP_NAMES
            for dtype in DTYPE_NAMES
            for rows in args.rows
        ),
    }
    for rank, visibility_entry in enumerate(visibility_entries):
        path = directory / f"rank-{rank}.json"
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(report, dict):
                raise ValueError("Rank report must be a JSON object")
            reports.append(report)
            if (
                report.get("status") != "passed"
                or any(type(report.get(key)) is not int for key in ("rank", "local_rank", "world_size"))
                or report.get("rank") != rank
                or report.get("local_rank") != rank
                or report.get("world_size") != len(visibility_entries)
                or report.get("visibility_entry") != visibility_entry
                or report.get("requested_visible_devices") != args.devices
                or report.get("ascend_rt_visible_devices") != args.devices
            ):
                raise ValueError("Rank failed or device/rank mapping does not match the requested run")
            checks = report.get("checks")
            if not isinstance(checks, list) or len(checks) != len(expected_checks):
                raise ValueError("Rank report is missing numerical collective checks")
            observed = {(check["operation"], check["group"], check["dtype"], tuple(check["shape"])) for check in checks}
            if observed != expected_checks:
                raise ValueError("Rank report does not cover the requested groups, dtypes and shapes")
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append({"rank": rank, "report": str(path), "error": f"{type(error).__name__}: {error}"})
    return reports, errors


def process_group_exists(process):
    """A worker leader may exit while descendants still own its process group."""
    if os.name != "posix":
        return process.poll() is None
    try:
        os.killpg(process.pid, 0)
    except ProcessLookupError:
        return False
    return True


def stop_worker_groups(processes):
    """Terminate all directly owned rank groups using one shared grace period."""
    previous_handlers = {}
    if threading.current_thread() is threading.main_thread():
        # A second Ctrl-C/SIGTERM must not interrupt the grace period before
        # every owned process group has been killed and its leader reaped.
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, signal.SIG_IGN)
    try:
        for process in processes:
            if os.name == "posix":
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
            elif process.poll() is None:
                process.terminate()
        deadline = time.monotonic() + TERMINATE_GRACE_SECONDS
        while True:
            # Reap leaders during the grace period so an exited leader's zombie
            # does not make an otherwise empty process group appear alive.
            for process in processes:
                process.poll()
            if not any(process_group_exists(process) for process in processes):
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(POLL_SECONDS, remaining))
        for process in processes:
            if os.name == "posix":
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
            elif process.poll() is None:
                process.kill()
        for process in processes:
            process.wait()
    finally:
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)


def execute_workers(args, directory, log_path):
    """Own every rank directly; no intermediate launcher can detach workers."""
    processes = []
    started = time.monotonic()
    world_size = len(args.devices.split(","))
    environment = {**os.environ, "ASCEND_RT_VISIBLE_DEVICES": args.devices}
    # The FileStore rendezvous is a new path inside this run's private
    # directory. Never reuse or delete another run's rendezvous file.
    if (directory / "rendezvous.store").exists():
        raise ValueError("The per-run FileStore rendezvous must not already exist")
    with log_path.open("a", encoding="utf-8") as output, LiveChildLog(log_path, "hccl-check"):
        try:
            for rank in range(world_size):
                processes.append(
                    subprocess.Popen(
                        build_worker_command(args, directory, rank),
                        cwd=REPO_ROOT,
                        env=environment,
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        start_new_session=os.name == "posix",
                    )
                )
            while True:
                returncodes = [process.poll() for process in processes]
                if any(code is not None and code != 0 for code in returncodes):
                    break
                if all(code is not None for code in returncodes):
                    break
                remaining = args.timeout - (time.monotonic() - started)
                if remaining <= 0:
                    raise subprocess.TimeoutExpired("HCCL worker processes", args.timeout)
                time.sleep(min(POLL_SECONDS, remaining))
        finally:
            stop_worker_groups(processes)
    return {str(rank): process.returncode for rank, process in enumerate(processes)}


def interrupt_handler(signum, frame):
    raise KeyboardInterrupt


def main(argv=None):
    args = parse_args(argv)
    if args.worker:
        return run_worker(args)
    if args.dry_run:
        directory = args.result_dir / "dry-run"
    else:
        args.result_dir.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix="hccl-check-", dir=args.result_dir)).resolve()
    commands = [build_worker_command(args, directory, rank) for rank in range(len(args.devices.split(",")))]
    for command in commands:
        print(shlex.join(["env", f"ASCEND_RT_VISIBLE_DEVICES={args.devices}", *command]), flush=True)
    print(SCOPE, flush=True)
    if args.dry_run:
        return 0
    summary = {
        "status": "running",
        "requested_visible_devices": args.devices,
        "rows": args.rows,
        "columns": args.columns,
        "timeout_seconds": args.timeout,
        "commands": commands,
        "log": str(directory / "workers.log"),
        "scope": SCOPE,
    }
    write_json(directory / "summary.json", summary)
    started = time.monotonic()
    previous_sigterm = None
    if os.name == "posix":
        previous_sigterm = signal.signal(signal.SIGTERM, interrupt_handler)
    try:
        summary["returncodes"] = execute_workers(args, directory, directory / "workers.log")
    except KeyboardInterrupt:
        summary.update(status="interrupted", error="Interrupted; owned worker process group stopped")
    except subprocess.TimeoutExpired as error:
        summary.update(status="timeout", error=f"Total worker wall timeout exceeded: {error.timeout} seconds")
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        summary.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        if previous_sigterm is not None:
            signal.signal(signal.SIGTERM, previous_sigterm)
    summary["elapsed_seconds"] = time.monotonic() - started
    summary["ranks"], summary["rank_errors"] = read_rank_reports(directory, args)
    if summary["status"] == "running":
        expected_ranks = {str(rank) for rank in range(len(args.devices.split(",")))}
        succeeded = (
            set(summary["returncodes"]) == expected_ranks
            and all(code == 0 for code in summary["returncodes"].values())
            and not summary["rank_errors"]
        )
        summary["status"] = "passed" if succeeded else "failed"
    write_json(directory / "summary.json", summary)
    print(f"HCCL_CHECK_STATUS={summary['status']} SUMMARY={directory / 'summary.json'}", flush=True)
    if summary["status"] == "interrupted":
        return 130
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
