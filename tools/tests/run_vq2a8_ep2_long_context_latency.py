#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure sequential 64K/256K/1M input, 1K output requests on an existing EP2 graph service."""

import argparse
import json
import statistics
import sys
import tempfile
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
from http.client import HTTPException
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools import benchmark_vq2a8_serving as bench  # noqa: E402

OUTPUT_TOKENS = 1024
CASE_INPUTS = (("64k", 65536), ("256k", 262144), ("1m", 1048576), ("1m-window", 1047552))
DEFAULT_TIMEOUT_SECONDS = 21600


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="vq2a8")
    parser.add_argument("--prompt", default="中国首都是", help="Seed text repeated as token IDs; synthetic workload")
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=[name for name, _ in CASE_INPUTS],
        default=["256k", "1m"],
        help="1m means 1048576 INPUT tokens; 1m-window explicitly reserves 1024 output tokens inside a 1M window",
    )
    parser.add_argument("--warmups", type=int, default=0, help="Full-length warmups per case, excluded from averages")
    parser.add_argument(
        "--repeats", type=int, default=1, help="Sequential measured requests per case; concurrency is 1"
    )
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS, help="HTTP socket timeout in seconds")
    parser.add_argument("--result-dir", type=Path, default=Path(__file__).resolve().parents[2] / "benchmark-results")
    parser.add_argument("--check-only", action="store_true", help="Check context capacity without generating output")
    args = parser.parse_args(argv)
    if len(set(args.cases)) != len(args.cases):
        parser.error("--cases must not contain duplicates")
    return args


def benchmark_args(args, input_tokens):
    # Reuse the existing URL/argument validation, exact tokenization and SSE timing.
    return bench.parse_args(
        [
            "--base-url",
            args.base_url,
            "--model",
            args.model,
            "--prompt",
            args.prompt,
            "--input-tokens",
            str(input_tokens),
            "--max-tokens",
            str(OUTPUT_TOKENS),
            "--warmups",
            str(args.warmups),
            "--repeats",
            str(args.repeats),
            "--timeout",
            str(args.timeout),
            "--progress",
        ]
    )


class Tee:
    def __init__(self, console, log):
        self.console = console
        self.log = log

    def write(self, text):
        self.console.write(text)
        self.log.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.log.flush()


@contextmanager
def case_log(path):
    with path.open("a", encoding="utf-8") as log:
        tee = Tee(sys.stdout, log)
        with redirect_stdout(tee), redirect_stderr(tee):
            yield


def save_case(directory, record):
    (directory / f"{record['case']}.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def measure_case(args, record, directory):
    record["status"] = "running"
    save_case(directory, record)
    with case_log(directory / f"{record['case']}.log"):
        try:
            for index in range(args.warmups):
                bench.run_request(args, "WARMUP", index + 1, args.warmups)
            for index in range(args.repeats):
                result = bench.run_request(args, "REQUEST", index + 1, args.repeats)
                record["requests"].append(result)
                save_case(directory, record)
                print(
                    f"CASE={record['case']} REQUEST={index + 1} "
                    f"TTFT_MS={result['ttft_ms']:.3f} TPOT_MS={result['tpot_ms']:.3f} "
                    f"INPUT_TOKENS={result['prompt_tokens']} OUTPUT_TOKENS={result['completion_tokens']}"
                )
            record["summary"] = {
                f"{aggregation}_{metric}": function(item[metric] for item in record["requests"])
                for aggregation, function in (("mean", statistics.mean), ("median", statistics.median))
                for metric in ("ttft_ms", "tpot_ms")
            }
            record["status"] = "passed"
            print("LONG_CONTEXT_SUMMARY=" + json.dumps(record["summary"]))
        except (HTTPException, OSError, ValueError) as error:
            record.update(status="failed", error=str(error))
            print(f"CASE={record['case']} FAILED: {error}")
        except KeyboardInterrupt:
            record.update(status="interrupted", error="Interrupted by user")
            print(f"CASE={record['case']} INTERRUPTED")
        finally:
            save_case(directory, record)


def main(argv=None):
    args = parse_args(argv)
    lengths = dict(CASE_INPUTS)
    variants = [(name, benchmark_args(args, lengths[name])) for name in args.cases]
    args.result_dir.mkdir(parents=True, exist_ok=True)
    directory = Path(
        tempfile.mkdtemp(prefix=datetime.now().strftime("vq2a8-ep2-long-%Y%m%d_%H%M%S-"), dir=args.result_dir)
    )
    print(f"RESULT_DIR={directory}")
    print("Target server: EP2, full_graph (decode graph), prefix caching disabled; verify its launch configuration.")
    records = []
    # Check every selected length before starting any expensive generation.
    for name, variant in variants:
        record = {
            "case": name,
            "model": args.model,
            "base_url": args.base_url,
            "input_tokens": variant.input_tokens,
            "output_tokens": OUTPUT_TOKENS,
            "required_context_tokens": variant.input_tokens + OUTPUT_TOKENS,
            "concurrency": 1,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "timeout_seconds": args.timeout,
            "requests": [],
            "status": "ready",
        }
        records.append(record)
        with case_log(directory / f"{name}.log"):
            print(f"CASE={name} PREFLIGHT INPUT={variant.input_tokens} OUTPUT={OUTPUT_TOKENS}")
            try:
                variant.prompt = bench.prepare_fixed_prompt(variant)
                print(f"CASE={name} READY")
            except (HTTPException, OSError, ValueError) as error:
                record.update(status="preflight_failed", error=str(error))
                print(f"CASE={name} PREFLIGHT_FAILED: {error}")
        save_case(directory, record)
    if not args.check_only:
        stop = False
        for (_, variant), record in zip(variants, records):
            if record["status"] != "ready":
                continue
            if stop:
                record.update(status="not_run", error="Earlier generation failed; no further requests sent")
                save_case(directory, record)
                continue
            measure_case(variant, record, directory)
            stop = record["status"] != "passed"
    report = {"check_only": args.check_only, "cases": records}
    (directory / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for record in records:
        print(f"CASE={record['case']} STATUS={record['status']} LOG={directory / (record['case'] + '.log')}")
    print(f"SUMMARY={directory / 'summary.json'}")
    expected = "ready" if args.check_only else "passed"
    return 0 if all(record["status"] == expected for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
