#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run existing GS32 NPU regression tests sequentially on a private Linux runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from contextlib import suppress
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.vq2a8_live_log import LiveChildLog  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = "tests/e2e/nightly"
COMPONENTS = (
    "activation",
    "device_routes",
    "full_ffn",
    "resident_plan",
    "ep_routes",
    "ep_attention",
)
MODEL_TEST = "test_vq2a8_gs32_ep.py::test_gs32_ep_real_model_eager_and_graphs_match_single_card[ep2]"
RECOVERY_TEST = "test_vq2a8_gs32_ep_long_context.py::test_gs32_ep_long_context_decode_graph_and_recovery[ep2]"
TERMINATE_GRACE_SECONDS = 10
PREFLIGHT_SECONDS = 120
LOG_TAIL_BYTES = 16384
LOG_TAIL_LINES = 64
CHECKPOINT_PAIRS = ("vq2_to_vq2", "vq2_to_mxfp4", "mxfp4_to_mxfp4")
ACCEPTED_STAGE_STATUSES = ("passed", "passed_with_coverage_gaps")
FULL_EP_PROFILE_MARKER = "VQ2A8_GS32_FULL_EP_PROFILE="
FULL_EP_PROFILE_FILE = "full-ep-ffn-profile.json"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Local source-v3 checkpoint directory")
    parser.add_argument("--gs32-library", type=Path, required=True)
    parser.add_argument("--gs32-binary", type=Path, required=True)
    parser.add_argument("--devices", required=True, help="Visible physical device IDs, e.g. 4,5; use idle NPUs")
    parser.add_argument("--suite", choices=("kernels", "pr", "nightly"), default="pr")
    parser.add_argument(
        "--model-runners",
        nargs="+",
        choices=("v1", "v2"),
        default=["v2"],
        help="Default V2; the first runner handles components, V2 adds EP2 acceptance, and nightly adds recovery",
    )
    parser.add_argument("--single-card-reference", action="store_true", help="Also load the entire EP1 reference model")
    parser.add_argument(
        "--allow-missing-checkpoint-pairs",
        nargs="+",
        choices=CHECKPOINT_PAIRS,
        default=[],
        help="Explicitly allow missing real-expert pairs, reported as incomplete numerical coverage",
    )
    parser.add_argument("--timeout", type=int, default=7200, help="Wall timeout in seconds per pytest stage")
    parser.add_argument("--result-dir", type=Path, default=REPO_ROOT / "benchmark-results/npu-ci")
    parser.add_argument("--dry-run", action="store_true", help="Print plan only; no NPU initialization or output files")
    args = parser.parse_args(argv)
    devices = args.devices.split(",")
    if any(not device.isascii() or not device.isdecimal() for device in devices):
        parser.error("--devices must be comma-separated nonnegative integers")
    devices = [str(int(device)) for device in devices]
    if len(set(devices)) != len(devices):
        parser.error("--devices must not contain duplicates")
    if len(devices) != (1 if args.suite == "kernels" else 2):
        parser.error("kernels requires exactly one visible device; pr/nightly require exactly two")
    if args.timeout <= 0 or len(set(args.model_runners)) != len(args.model_runners):
        parser.error("--timeout must be positive and --model-runners must be distinct")
    if len(set(args.allow_missing_checkpoint_pairs)) != len(args.allow_missing_checkpoint_pairs):
        parser.error("--allow-missing-checkpoint-pairs must be distinct")
    args.devices = ",".join(devices)
    for name in ("model", "gs32_library", "gs32_binary", "result_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    return args


def build_plan(args):
    common = [
        sys.executable,
        "-u",
        "-m",
        "pytest",
        "-vv",
        "-s",
        "-x",
        "-ra",
        "-o",
        "addopts=",
        "-o",
        "xfail_strict=True",
    ]
    checkpoint_options = [
        "--vq2a8-model",
        str(args.model),
        "--vq2a8-gs32-library",
        str(args.gs32_library),
        "--vq2a8-gs32-binary",
        str(args.gs32_binary),
    ]
    stages = []

    def add(name, test, devices, runner, extra=()):
        # pytest 8 discovers conftest.py from initial positional paths before
        # it knows our custom options. Put the test ahead of checkpoint paths.
        stages.append(
            {
                "name": name,
                "command": [*common, f"{TEST_ROOT}/{test}", *checkpoint_options, *extra],
                "environment": {
                    "ASCEND_RT_VISIBLE_DEVICES": devices,
                    "VLLM_USE_V2_MODEL_RUNNER": "1" if runner == "v2" else "0",
                    "PYTEST_ADDOPTS": "",
                },
            }
        )

    for component in COMPONENTS:
        add(component, f"test_vq2a8_gs32_{component}.py", args.devices.split(",")[0], args.model_runners[0])
    if args.suite != "kernels":
        extra = () if args.single_card_reference else ("--vq2a8-gs32-ep-skip-single-card",)
        for runner in args.model_runners:
            if runner == "v2":
                add("ep2-v2", MODEL_TEST, args.devices, runner, extra)
            if args.suite == "nightly":
                add(f"recovery-{runner}", RECOVERY_TEST, args.devices, runner)
    return stages


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def stop_process_group(process):
    """The Linux runner owns a new session, including pytest's model workers."""
    if os.name == "posix":
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
    elif process.poll() is None:
        process.terminate()
    with suppress(subprocess.TimeoutExpired):
        process.wait(timeout=TERMINATE_GRACE_SECONDS)
    # A pytest parent can exit before its workers; kill remaining session members.
    if os.name == "posix":
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
    elif process.poll() is None:
        process.kill()
    process.wait()


def execute(command, environment, log, timeout):
    with log.open("w", encoding="utf-8") as output, LiveChildLog(log, f"npu-ci-{log.stem}"):
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env={**os.environ, **environment},
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=os.name == "posix",
        )
        try:
            return process.wait(timeout=timeout)
        finally:
            # Clean up workers even when pytest itself has already exited.
            stop_process_group(process)


def preflight(args, result_dir):
    if sys.platform != "linux":
        raise ValueError("NPU CI execution requires Linux; use --dry-run to inspect the plan elsewhere")
    if not args.model.is_dir() or not all(path.is_file() for path in (args.gs32_library, args.gs32_binary)):
        raise ValueError("Provide an existing model directory and both native artifact files")
    config = json.loads((args.model / "config.json").read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("quantization_config"), dict):
        raise ValueError("config.json and quantization_config must be JSON objects")
    quant = config.get("quantization_config", {})
    if (config.get("model_type"), quant.get("quant_method"), quant.get("format"), quant.get("format_version")) != (
        "deepseek_v4",
        "vq2a8",
        "vq2_mixed_gs32_source",
        3,
    ):
        raise ValueError("This suite requires a DeepSeek-V4 source-v3 mixed GS32 checkpoint")
    if not (args.model / "model.safetensors.index.json").is_file():
        raise ValueError("Missing model.safetensors.index.json")
    evidence_path = result_dir / "native-preflight.json"
    command = [
        sys.executable,
        "-u",
        "-c",
        "\n".join(
            (
                "import json, pathlib, sys, torch, torch_npu, triton, vllm, vllm_ascend",
                "assert torch.npu.is_available(), 'Real NPU hardware is required'",
                f"assert torch.npu.device_count() == {len(args.devices.split(','))}, 'Unexpected visible NPU count'",
                "names = [torch.npu.get_device_name(i) for i in range(torch.npu.device_count())]",
                "assert all('950' in name for name in names), f'GS32 acceptance requires Ascend950: {names}'",
                "source = pathlib.Path(vllm_ascend.__file__).resolve()",
                "assert source.is_relative_to(pathlib.Path.cwd()), 'Wrong Ascend checkout'",
                "from tools.build_vq2a8_gs32_stage import verify_loaded_library, verify_runtime_artifacts",
                "schemas = verify_loaded_library(sys.argv[1], execution='full_graph',",
                "    fusion='full_ffn', binary=sys.argv[2])",
                "evidence = {'environment': {'python': sys.version, 'torch': torch.__version__,",
                "'torch_npu': torch_npu.__version__, 'triton': triton.__version__, 'vllm': vllm.__version__,",
                "'ascend_path': str(source), 'devices': names}, 'schemas': schemas,",
                "'native_artifacts': verify_runtime_artifacts(sys.argv[1], sys.argv[2]),",
                "'scope': 'Environment, ABI and artifact identity only; '",
                "    'no weights loaded or kernel execution verified'}",
                "pathlib.Path(sys.argv[3]).write_text(json.dumps(evidence, indent=2, allow_nan=False) + '\\n',",
                "    encoding='utf-8')",
                "print(json.dumps(evidence, allow_nan=False))",
            )
        ),
        str(args.gs32_library),
        str(args.gs32_binary),
        str(evidence_path),
    ]
    result = execute(
        command,
        {
            "ASCEND_RT_VISIBLE_DEVICES": args.devices,
            "VLLM_USE_V2_MODEL_RUNNER": "1" if args.model_runners[0] == "v2" else "0",
        },
        result_dir / "environment.log",
        PREFLIGHT_SECONDS,
    )
    if result:
        raise RuntimeError(f"NPU environment/native ABI preflight failed ({result}); see environment.log")
    return {"file": str(evidence_path), "sha256": file_sha256(evidence_path)}


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preserve_full_ep_profile(stage, args, result_dir):
    """Archive the actual child report, never synthesize absent measurements."""
    payloads = []
    with Path(stage["log"]).open(encoding="utf-8") as stream:
        for line in stream:
            prefix, marker, payload = line.partition(FULL_EP_PROFILE_MARKER)
            prefix = prefix.strip()
            # -vv can print the nodeid before captured test output. A failure
            # traceback also prints the literal marker inside quoted source;
            # that source is not a second measured report.
            node_prefix = (
                len(prefix.split()) == 1 and ".py::" in prefix and not any(quote in prefix for quote in ("'", '"'))
            )
            if marker and (not prefix or node_prefix):
                payloads.append(payload.strip())
    if len(payloads) != 1:
        raise ValueError(f"Full FFN requires exactly one profile marker; found {len(payloads)}")

    def reject_nonfinite(value):
        raise ValueError(f"Full FFN profile contains non-finite JSON value: {value}")

    profile = json.loads(payloads[0], parse_constant=reject_nonfinite)
    if (
        not isinstance(profile, dict)
        or type(profile.get("schema_version")) is not int
        or profile["schema_version"] != 1
    ):
        raise ValueError("Full FFN profile must be a schema_version=1 JSON object")
    if profile.get("status") not in ("PASS", "ALLOCATION_GROWTH"):
        raise ValueError("Full FFN profile has an unknown or missing status")
    sections = (
        "environment",
        "placement",
        "dimensions",
        "captures",
        "resident_weights",
        "retained_capture_storage",
        "replay_stability",
    )
    if any(not isinstance(profile.get(key), dict) or not profile[key] for key in sections):
        raise ValueError("Full FFN profile is missing capture, inventory or memory measurements")
    if not isinstance(profile.get("timings"), list) or not profile["timings"]:
        raise ValueError("Full FFN profile has no recorded timings")
    stability = profile["replay_stability"]
    if (
        type(stability.get("steps")) is not int
        or stability["steps"] < 1
        or type(stability.get("allocated_stable")) is not bool
        or stability["allocated_stable"] != (profile["status"] == "PASS")
    ):
        raise ValueError("Full FFN profile status disagrees with replay stability measurements")
    model_path = profile.get("model_path")
    if not isinstance(model_path, str) or Path(model_path).resolve() != args.model:
        raise ValueError("Full FFN profile model does not match this run")
    for key, path in (
        ("config_sha256", args.model / "config.json"),
        ("index_sha256", args.model / "model.safetensors.index.json"),
        ("library_sha256", args.gs32_library),
        ("kernel_sha256", args.gs32_binary),
        ("build_manifest_sha256", args.gs32_library.parent / "build-manifest.json"),
    ):
        if profile.get(key) != file_sha256(path):
            raise ValueError(f"Full FFN profile {key} does not match this run")
    destination = result_dir / FULL_EP_PROFILE_FILE
    write_json(destination, profile)
    return {"file": str(destination), "sha256": file_sha256(destination), "status": profile["status"]}


def junit_counts(root):
    cases = list(root.iter("testcase"))
    counts = {"tests": len(cases), "passed": 0, "failures": 0, "errors": 0, "skipped": 0}
    for case in cases:
        if case.find("error") is not None:
            counts["errors"] += 1
        elif case.find("failure") is not None:
            counts["failures"] += 1
        elif case.find("skipped") is not None:
            counts["skipped"] += 1
        else:
            counts["passed"] += 1
    return counts


def read_junit(path):
    root = ET.parse(path).getroot()
    if root.tag not in ("testsuites", "testsuite"):
        raise ValueError("Unrecognized JUnit report")
    for suite in root.iter():
        if suite.tag not in ("testsuites", "testsuite"):
            continue
        counts = junit_counts(suite)
        for key in ("tests", "failures", "errors", "skipped"):
            if key in suite.attrib and int(suite.attrib[key]) != counts[key]:
                raise ValueError(f"JUnit {key} count does not match recorded test cases")
    skipped_tests = [
        {
            "classname": case.get("classname", ""),
            "name": case.get("name", ""),
            "type": skipped.get("type", ""),
            "reason": skipped.get("message", ""),
        }
        for case in root.iter("testcase")
        for skipped in case.findall("skipped")
    ]
    return junit_counts(root), skipped_tests


def checkpoint_coverage_gaps(stage, allowed_pairs):
    """Recognize only supported missing-data skips, never hardware skips or xfails."""
    if stage["name"] != "full_ffn":
        return []
    classname = "tests.e2e.nightly.test_vq2a8_gs32_full_ffn"
    allowed = {}
    for pair in allowed_pairs:
        label = tuple(pair.split("_to_"))
        reason = f"No real {label} expert in this checkpoint; this precision pair was not tested."
        allowed[(f"test_gs32_real_experts_full_ffn_compare[{pair}]", reason)] = pair
    gaps = []
    for skipped in stage["skipped_tests"]:
        pair = allowed.get((skipped["name"], skipped["reason"]))
        if pair is not None and skipped["classname"] == classname and skipped["type"] == "pytest.skip":
            gaps.append(
                {
                    "nodeid": f"{TEST_ROOT}/test_vq2a8_gs32_full_ffn.py::{skipped['name']}",
                    "pair": pair,
                    "reason": skipped["reason"],
                }
            )
    return gaps


def capture_failure_log(stage):
    """Keep bounded child output without hiding the original stage failure."""
    try:
        with Path(stage["log"]).open("rb") as log:
            size = log.seek(0, os.SEEK_END)
            log.seek(max(0, size - LOG_TAIL_BYTES))
            tail = log.read(LOG_TAIL_BYTES).decode("utf-8", errors="replace")
        stage["log_tail"] = "\n".join(tail.splitlines()[-LOG_TAIL_LINES:])
    except OSError as error:
        stage["log_tail_error"] = str(error)


def run_stage(stage, args, result_dir, on_update):
    name = stage["name"]
    stage.update(
        status="running",
        log=str(result_dir / f"{name}.log"),
        junit=str(result_dir / f"{name}.xml"),
        pytest_temp=str(result_dir / f"{name}-tmp"),
    )
    stage["command"] += [f"--junitxml={stage['junit']}", f"--basetemp={stage['pytest_temp']}"]
    on_update()
    print(f"STAGE={name} START LOG={stage['log']}", flush=True)
    start = time.monotonic()
    try:
        stage["returncode"] = execute(stage["command"], stage["environment"], Path(stage["log"]), args.timeout)
        try:
            stage["counts"], stage["skipped_tests"] = read_junit(stage["junit"])
        except (OSError, ValueError, ET.ParseError) as error:
            stage.update(
                status="failed",
                junit_error=str(error),
                error=(
                    f"pytest exited with return code {stage['returncode']}; JUnit report unavailable or invalid: "
                    f"{error}. See {stage['log']} for child-process diagnostics"
                ),
            )
        else:
            counts = stage["counts"]
            stage["coverage_gaps"] = checkpoint_coverage_gaps(stage, args.allow_missing_checkpoint_pairs)
            passed = (
                stage["returncode"] == 0
                and counts["passed"] > 0
                and counts["failures"] == counts["errors"] == 0
                and counts["skipped"] == len(stage["skipped_tests"]) == len(stage["coverage_gaps"])
            )
            stage["status"] = (
                ("passed_with_coverage_gaps" if stage["coverage_gaps"] else "passed") if passed else "failed"
            )
            if not passed:
                stage["error"] = (
                    f"pytest exited with return code {stage['returncode']}; NPU CI requires at least one passed test, "
                    f"no failures/errors, and no unsupported skipped/xfail tests. See {stage['log']}"
                )
    except KeyboardInterrupt:
        stage.update(status="interrupted", error="Interrupted; owned pytest process group stopped")
        raise
    except subprocess.TimeoutExpired:
        stage.update(status="timeout", error=f"Stage exceeded {args.timeout} seconds; process group stopped")
    except (OSError, ValueError, ET.ParseError) as error:
        stage.update(status="failed", error=str(error))
    finally:
        if name == "full_ffn":
            try:
                stage["full_ep_profile"] = preserve_full_ep_profile(stage, args, result_dir)
                if stage["full_ep_profile"]["status"] != "PASS":
                    raise ValueError("Full FFN profile reports allocation growth after warmup")
            except (OSError, ValueError) as error:
                stage["profile_error"] = str(error)
                if stage["status"] in ACCEPTED_STAGE_STATUSES:
                    stage.update(status="failed", error=f"Full FFN profile acceptance failed: {error}")
        stage["elapsed_seconds"] = time.monotonic() - start
        if stage["status"] not in ACCEPTED_STAGE_STATUSES:
            capture_failure_log(stage)
        on_update()
        print(f"STAGE={name} STATUS={stage['status']}", flush=True)
        if stage["status"] == "passed_with_coverage_gaps":
            pairs = ", ".join(sorted({gap["pair"].replace("_to_", " -> ") for gap in stage["coverage_gaps"]}))
            print(
                f"STAGE={name} NOTICE=Checkpoint has no real expert for {pairs}; "
                f"recorded {len(stage['coverage_gaps'])} untested cases and continuing remaining checks. "
                f"MODEL={args.model}",
                flush=True,
            )
        for gap in stage.get("coverage_gaps", []):
            print(f"STAGE={name} COVERAGE_GAP=" + json.dumps(gap, ensure_ascii=False), flush=True)
        if "error" in stage:
            if "returncode" in stage:
                print(f"STAGE={name} RETURN_CODE={stage['returncode']}", flush=True)
            print(f"STAGE={name} ERROR={stage['error']}", flush=True)
            if stage.get("log_tail"):
                print(f"STAGE={name} LOG_TAIL_BEGIN LOG={stage['log']}", flush=True)
                print(stage["log_tail"], flush=True)
                print(f"STAGE={name} LOG_TAIL_END", flush=True)
            elif "log_tail_error" in stage:
                print(f"STAGE={name} LOG_TAIL_ERROR={stage['log_tail_error']}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    stages = build_plan(args)
    report = {
        "suite": args.suite,
        "status": "planned",
        "devices": args.devices,
        "model_runners": [
            runner
            for runner in args.model_runners
            if any(stage["name"] in (f"ep2-{runner}", f"recovery-{runner}") for stage in stages)
        ],
        "single_card_reference": args.single_card_reference
        and any(stage["name"].startswith("ep2-") for stage in stages),
        "automatic_missing_checkpoint_pairs": [],
        "allowed_missing_checkpoint_pairs": args.allow_missing_checkpoint_pairs,
        "coverage_complete": False,
        "coverage_gaps": [],
        "scope": "NPU correctness and regression; local timings are not a serving performance gate",
        "timeout_per_stage_seconds": args.timeout,
        "stages": stages,
    }
    if args.dry_run:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    args.result_dir.mkdir(parents=True, exist_ok=True)
    result_dir = Path(tempfile.mkdtemp(prefix=time.strftime("vq2a8-npu-ci-%Y%m%d_%H%M%S-"), dir=args.result_dir))
    report["result_dir"] = str(result_dir)
    try:
        report["git_revision"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        report["git_revision"] = None
    for stage in stages:
        stage["status"] = "not_run"
    summary = result_dir / "summary.json"

    def save():
        report["full_ep_profile"] = next(
            (stage["full_ep_profile"] for stage in stages if "full_ep_profile" in stage), None
        )
        report["coverage_gaps"] = [
            {"stage": stage["name"], "model": str(args.model), **gap}
            for stage in stages
            for gap in stage.get("coverage_gaps", [])
        ]
        # Complete refers to the selected suite, not unselected EP1/long-context tests.
        report["coverage_complete"] = report["status"] == "passed"
        write_json(summary, report)

    print(f"RESULT_DIR={result_dir}", flush=True)
    report["status"] = "preflight"
    save()
    exit_code = 1
    old_sigterm = None
    if os.name == "posix":
        old_sigterm = signal.signal(signal.SIGTERM, interrupt_handler)
    try:
        report["preflight"] = preflight(args, result_dir)
        report["status"] = "running"
        for stage in stages:
            run_stage(stage, args, result_dir, save)
            if stage["status"] not in ACCEPTED_STAGE_STATUSES:
                report["status"] = "failed"
                break
        else:
            report["status"] = "passed_with_coverage_gaps" if report["coverage_gaps"] else "passed"
            exit_code = 0
    except KeyboardInterrupt:
        report.update(status="interrupted", error="Interrupted by user")
        exit_code = 130
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        report.update(status="failed", error=str(error))
    finally:
        if old_sigterm is not None:
            signal.signal(signal.SIGTERM, old_sigterm)
        save()
        print(f"NPU_CI_STATUS={report['status']} SUMMARY={summary}", flush=True)
        if "error" in report:
            print(f"ERROR={report['error']}", flush=True)
    return exit_code


def interrupt_handler(signum, frame):
    raise KeyboardInterrupt


if __name__ == "__main__":
    raise SystemExit(main())
