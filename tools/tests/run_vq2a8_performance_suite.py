#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run sequential short/long TTFT and TPOT cases against an existing service."""

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from http.client import HTTPException
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools import benchmark_vq2a8_serving as bench  # noqa: E402
from tools.tests import run_vq2a8_ep2_long_context_latency as long_runner  # noqa: E402


@dataclass(frozen=True)
class Case:
    name: str
    input_tokens: int
    output_tokens: int
    warmups: int = 0
    repeats: int = 2


CASES = (
    Case("128x32", 128, 32),
    Case("127x32", 127, 32),
    Case("129x32", 129, 32),
    Case("224x32", 224, 32),
    Case("128x128", 128, 128, warmups=1, repeats=10),
    Case("64k", 65536, 1024),
    Case("256k", 262144, 1024),
    Case("1m", 1048576, 1024),
    Case("1m-window", 1047552, 1024),
    Case("recovery", 128, 128, repeats=3),
)
QUICK_CASES = ("128x32", "127x32", "129x32", "224x32", "128x128")
SUITES = (
    ("quick", QUICK_CASES),
    ("standard", (*QUICK_CASES, "64k", "256k", "recovery")),
    ("full", (*QUICK_CASES, "64k", "256k", "1m", "recovery")),
)
METRICS = ("mean_ttft_ms", "median_ttft_ms", "mean_tpot_ms", "median_tpot_ms")
CSV_FIELDS = (
    "case",
    "status",
    "input_tokens",
    "output_tokens",
    "measured_requests",
    "warmup_requests",
    "first_ttft_ms",
    *METRICS,
    "error",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="vq2a8")
    parser.add_argument("--prompt", default="Explain how photosynthesis works.", help="Repeated synthetic seed text")
    parser.add_argument("--suite", choices=[name for name, _ in SUITES], default="standard")
    parser.add_argument("--cases", nargs="+", choices=[case.name for case in CASES], help="Override suite order")
    parser.add_argument("--warmups", type=int, help="Override every case's warmup count; excluded from averages")
    parser.add_argument("--repeats", type=int, help="Override every case's measured request count")
    parser.add_argument(
        "--timeout", type=float, default=long_runner.DEFAULT_TIMEOUT_SECONDS, help="HTTP socket timeout"
    )
    parser.add_argument("--result-dir", type=Path, default=Path(__file__).resolve().parents[2] / "benchmark-results")
    parser.add_argument("--label", default="", help="User-supplied run label, e.g. mrv1-ep2-chunk8192-none")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true", help="Preflight every case without generation")
    mode.add_argument("--dry-run", action="store_true", help="Print plan without HTTP requests or result files")
    args = parser.parse_args(argv)
    if args.cases and len(set(args.cases)) != len(args.cases):
        parser.error("--cases must not contain duplicates")
    return args


def benchmark_args(args, case):
    return bench.parse_args(
        [
            "--base-url",
            args.base_url,
            "--model",
            args.model,
            "--prompt",
            args.prompt,
            "--input-tokens",
            str(case.input_tokens),
            "--max-tokens",
            str(case.output_tokens),
            "--warmups",
            str(case.warmups if args.warmups is None else args.warmups),
            "--repeats",
            str(case.repeats if args.repeats is None else args.repeats),
            "--timeout",
            str(args.timeout),
            "--progress",
        ]
    )


def client_revision():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def make_record(name, args):
    return {
        "case": name,
        "model": args.model,
        "base_url": args.base_url,
        "input_tokens": args.input_tokens,
        "output_tokens": args.max_tokens,
        "required_context_tokens": args.input_tokens + args.max_tokens,
        "concurrency": 1,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "timeout_seconds": args.timeout,
        "requests": [],
        "warmup_requests": [],
        "status": "pending",
    }


def csv_row(record):
    first = record["warmup_requests"] or record["requests"]
    return {
        **{key: record[key] for key in ("case", "status", "input_tokens", "output_tokens")},
        "measured_requests": len(record["requests"]),
        "warmup_requests": len(record["warmup_requests"]),
        "first_ttft_ms": first[0]["ttft_ms"] if first else None,
        **{key: record.get("summary", {}).get(key) for key in METRICS},
        "error": record.get("error", ""),
    }


def save_report(directory, report):
    long_runner.write_json(directory / "summary.json", report)
    temporary = directory / "summary.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(csv_row(record) for record in report["cases"])
    temporary.replace(directory / "summary.csv")


def preflight(variants, records, directory, report):
    for (_, variant), record in zip(variants, records):
        report["active_case"] = record["case"]
        with long_runner.case_log(directory / f"{record['case']}.log"):
            print(f"CASE={record['case']} PREFLIGHT INPUT={variant.input_tokens} OUTPUT={variant.max_tokens}")
            try:
                variant.prompt = bench.prepare_fixed_prompt(variant)
                record["status"] = "ready"
                print(f"CASE={record['case']} READY")
            except (HTTPException, OSError, ValueError) as error:
                record.update(status="preflight_failed", error=str(error))
                print(f"CASE={record['case']} PREFLIGHT_FAILED: {error}")
            except KeyboardInterrupt:
                record.update(status="interrupted", error="Interrupted during preflight")
            finally:
                long_runner.save_case(directory, record)
                save_report(directory, report)
        if record["status"] == "interrupted":
            break
    report.pop("active_case", None)


def run_cases(variants, records, directory, report):
    for (_, variant), record in zip(variants, records):
        report["active_case"] = record["case"]
        long_runner.measure_case(
            variant,
            record,
            directory,
            summary_prefix="VQ2A8_PERFORMANCE_CASE_SUMMARY",
            on_update=partial(save_report, directory, report),
        )
        if record["status"] != "passed":
            break
    report.pop("active_case", None)


def finish_report(directory, report):
    for record in report["cases"]:
        if record["status"] == "pending" or (record["status"] == "ready" and not report["check_only"]):
            record.update(status="not_run", error="Suite stopped; no generation requests sent for this case")
        long_runner.save_case(directory, record)
    save_report(directory, report)
    print("CASE             STATUS               MEAN_TTFT_MS  MEAN_TPOT_MS")
    for record in report["cases"]:
        metrics = record.get("summary", {})
        ttft = f"{metrics['mean_ttft_ms']:.3f}" if metrics else "-"
        tpot = f"{metrics['mean_tpot_ms']:.3f}" if metrics else "-"
        print(f"{record['case']:<16} {record['status']:<20} {ttft:>12} {tpot:>13}")
    print(f"SUMMARY={directory / 'summary.json'}")
    print(f"CSV={directory / 'summary.csv'}")


def main(argv=None):
    args = parse_args(argv)
    selected = args.cases or dict(SUITES)[args.suite]
    specs = {case.name: case for case in CASES}
    variants = [(name, benchmark_args(args, specs[name])) for name in selected]
    records = [make_record(name, variant) for name, variant in variants]
    report = {
        "suite": "custom" if args.cases else args.suite,
        "label": args.label,
        "check_only": args.check_only,
        "client_git_revision": client_revision(),
        "cases": records,
        "scope": "Sequential client-visible SSE timings; synthetic prompts; not model accuracy or server profiling",
        "server_configuration": "Not verified by this client; label is user-supplied, not server evidence",
    }
    if args.dry_run:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    args.result_dir.mkdir(parents=True, exist_ok=True)
    directory = Path(
        tempfile.mkdtemp(prefix=datetime.now().strftime("vq2a8-performance-%Y%m%d_%H%M%S-"), dir=args.result_dir)
    )
    print(f"RESULT_DIR={directory}", flush=True)
    print("Using existing service. Verify prefix caching is disabled and keep runner/chunk/fusion settings fixed.")
    print("All cases are preflighted before generation. A failure stops the suite; input lengths are never shortened.")
    save_report(directory, report)
    try:
        preflight(variants, records, directory, report)
        if not args.check_only and all(record["status"] == "ready" for record in records):
            run_cases(variants, records, directory, report)
    except KeyboardInterrupt:
        record = next((record for record in records if record["case"] == report.get("active_case")), None)
        if record is not None:
            record.update(status="interrupted", error="Interrupted by user")
        else:
            report["interrupted"] = True
    finally:
        finish_report(directory, report)
    if report.get("interrupted") or any(record["status"] == "interrupted" for record in records):
        return 130
    expected = "ready" if args.check_only else "passed"
    return 0 if all(record["status"] == expected for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
