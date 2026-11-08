#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sweep an existing VQ2A8 server with vllm bench serve; report measured output TPS.

Topology arguments describe the server; they do not reconfigure or verify it.
Run with the server's vLLM environment. --dry-run needs only Python's stdlib.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import shlex
import statistics
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen

MAX_LOCAL_SEQUENCES = 64
REQUEST_WAVES = 4
LATENCY_FIELDS = ("mean_ttft_ms", "p99_ttft_ms", "mean_tpot_ms", "p99_tpot_ms", "p99_e2el_ms")
REPO_ROOT = Path(__file__).resolve().parents[2]


def runtime_contract():
    # Read the packaged, stdlib-only contract without initializing the plugin.
    path = REPO_ROOT / "vllm_ascend/quantization/vq2a8_gs32_runtime.py"
    spec = importlib.util.spec_from_file_location("vq2a8_gs32_runtime_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FULL_FFN_ABI_VERSION, module.KERNEL_ENTRY


def validate_artifact_record(artifacts):
    """Check recorded identity only; server paths need not exist on this client."""
    abi_version, kernel_entry = runtime_contract()
    if not isinstance(artifacts, dict):
        raise ValueError("runtime-record requires complete native_artifacts from the current server startup")
    kernel = artifacts.get("kernel")
    if (
        type(artifacts.get("abi_version")) is not int
        or artifacts["abi_version"] != abi_version
        or not isinstance(kernel, dict)
        or type(kernel.get("abi_version")) is not int
        or kernel["abi_version"] != abi_version
        or kernel.get("entry") != kernel_entry
    ):
        raise ValueError(
            f"runtime-record requires native_artifacts ABI {abi_version} and kernel entry {kernel_entry}; "
            "rebuild the current runtime and use its new server startup record"
        )
    if kernel.get("present") is not True or type(kernel.get("bytes")) is not int or kernel["bytes"] < 4:
        raise ValueError("runtime-record requires a present, non-empty kernel artifact")
    for value in (artifacts.get("library"), artifacts.get("build_manifest"), kernel.get("path")):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("runtime-record requires library, build_manifest and kernel paths")
    for value in (artifacts.get("library_sha256"), kernel.get("sha256")):
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError("runtime-record requires library and kernel SHA256 identities")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8003")
    parser.add_argument("--model", default="vq2a8", help="Served model name")
    parser.add_argument("--tokenizer", required=True, help="Local model/tokenizer path on the benchmark host")
    parser.add_argument("--tokenizer-mode", default="deepseek_v4")
    parser.add_argument("--parallel-mode", choices=("tp-ep", "dp-ep"), default="tp-ep")
    parser.add_argument("--ep-size", type=int, choices=(1, 2, 4, 8), default=2)
    parser.add_argument(
        "--max-num-seqs", type=int, default=64, help="Per-engine server setting, not client concurrency"
    )
    parser.add_argument("--concurrency", type=int, nargs="+", help="Default: powers of two through DP * max-num-seqs")
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument(
        "--num-prompts", type=int, help="Fixed count at every concurrency; default: 4 * current concurrency"
    )
    parser.add_argument(
        "--num-warmups", type=int, default=1, help="Fixed warmup request count per trial; 0 disables explicit warmup"
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=7200, help="Wall timeout in seconds for each bench subprocess")
    parser.add_argument("--max-p99-ttft-ms", type=float, help="Optional latency ceiling for the separate SLO peak")
    parser.add_argument("--max-p99-tpot-ms", type=float, help="Optional latency ceiling for the separate SLO peak")
    parser.add_argument("--result-dir", type=Path, default=Path("benchmark-results/vq2a8-throughput"))
    parser.add_argument("--note", default="", help="Record hardware, CANN, checkpoint/build IDs and server settings")
    parser.add_argument(
        "--runtime-record",
        type=Path,
        help="Server startup JSON from serve_vq2a8.py; verify model root against /v1/models and preserve it",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without loading vLLM, sending or writing"
    )
    args = parser.parse_args(argv)
    url = urlsplit(args.base_url)
    if url.scheme not in ("http", "https") or not url.netloc or url.path not in ("", "/") or url.query or url.fragment:
        parser.error("--base-url must be an HTTP(S) origin, without /v1, query or fragment")
    args.base_url = args.base_url.rstrip("/")
    if args.parallel_mode == "dp-ep" and args.ep_size not in (2, 4, 8):
        parser.error("dp-ep requires EP=DP=2, 4 or 8, TP=1")
    if not 1 <= args.max_num_seqs <= MAX_LOCAL_SEQUENCES:
        parser.error("--max-num-seqs must be in [1, 64]")
    args.tp = args.ep_size if args.parallel_mode == "tp-ep" else 1
    args.dp = args.ep_size if args.parallel_mode == "dp-ep" else 1
    args.global_sequence_cap = args.dp * args.max_num_seqs
    if args.concurrency is None:
        args.concurrency = []
        value = 1
        while value < args.global_sequence_cap:
            args.concurrency.append(value)
            value *= 2
        args.concurrency.append(args.global_sequence_cap)
    if min(args.concurrency) < 1 or len(set(args.concurrency)) != len(args.concurrency):
        parser.error("--concurrency requires distinct positive integers")
    args.concurrency.sort()
    if args.num_prompts is not None and args.num_prompts < max(args.concurrency):
        parser.error("--num-prompts must be >= the largest concurrency")
    if args.num_warmups < 0:
        parser.error("--num-warmups must be >=0")
    if args.input_len < 1 or args.output_len < 2 or args.repeats < 1 or args.seed < 0:
        parser.error("Require input-len >=1, output-len >=2, repeats >=1 and seed >=0")
    if not args.model.strip() or not args.tokenizer.strip() or not args.tokenizer_mode.strip():
        parser.error("Model, tokenizer and tokenizer-mode cannot be empty")
    for name in ("timeout", "max_p99_ttft_ms", "max_p99_tpot_ms"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    return args


def requests_per_trial(args, concurrency):
    return args.num_prompts if args.num_prompts is not None else REQUEST_WAVES * concurrency


def bench_command(args, concurrency, result_path):
    return [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "bench",
        "serve",
        "--backend",
        "vllm",
        "--base-url",
        args.base_url,
        "--endpoint",
        "/v1/completions",
        "--model",
        args.model,
        "--tokenizer",
        args.tokenizer,
        "--tokenizer-mode",
        args.tokenizer_mode,
        "--dataset-name",
        "random",
        "--random-input-len",
        str(args.input_len),
        "--random-output-len",
        str(args.output_len),
        "--random-range-ratio",
        "0",
        "--random-prefix-len",
        "0",
        "--request-rate",
        "inf",
        "--max-concurrency",
        str(concurrency),
        "--num-prompts",
        str(requests_per_trial(args, concurrency)),
        "--num-warmups",
        str(args.num_warmups),
        "--temperature",
        "0",
        "--ignore-eos",
        "--seed",
        str(args.seed),
        "--percentile-metrics",
        "ttft,tpot,e2el",
        "--metric-percentiles",
        "50,90,99",
        "--save-result",
        "--save-detailed",
        "--result-dir",
        str(result_path.parent),
        "--result-filename",
        result_path.name,
    ]


def validate_result(result, args, concurrency):
    """Fail closed: a zero process exit code does not establish successful requests."""
    if not isinstance(result, dict):
        raise ValueError("Benchmark result must be a JSON object")
    num_prompts = requests_per_trial(args, concurrency)
    expected_counts = {
        "completed": num_prompts,
        "failed": 0,
        "total_output_tokens": num_prompts * args.output_len,
    }
    for key, expected in expected_counts.items():
        if type(result.get(key)) is not int or result[key] != expected:
            raise ValueError(f"{key} must equal {expected}, got {result.get(key)!r}")
    lengths = result.get("output_lens")
    if (
        not isinstance(lengths, list)
        or len(lengths) != num_prompts
        or any(type(length) is not int or length != args.output_len for length in lengths)
    ):
        raise ValueError("output_lens must contain the requested output length for every request")
    inputs = result.get("input_lens")
    if (
        not isinstance(inputs, list)
        or len(inputs) != num_prompts
        or any(type(length) is not int or length < 1 for length in inputs)
    ):
        raise ValueError("Missing positive per-request input_lens")
    errors = result.get("errors")
    if not isinstance(errors, list) or len(errors) != num_prompts or any(errors):
        raise ValueError("Detailed benchmark errors must be empty for every request")
    metrics = {}
    for key in ("duration", "output_throughput", "request_throughput", *LATENCY_FIELDS):
        value = result.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Missing or invalid finite nonnegative metric: {key}")
        metrics[key] = value
    if metrics["duration"] <= 0 or metrics["output_throughput"] <= 0:
        raise ValueError("Duration and output throughput must be positive")
    if not math.isclose(
        metrics["output_throughput"], result["total_output_tokens"] / metrics["duration"], rel_tol=1e-5
    ):
        raise ValueError("Output throughput disagrees with total_output_tokens / duration")
    if not math.isclose(metrics["request_throughput"], num_prompts / metrics["duration"], rel_tol=1e-5):
        raise ValueError("Request throughput disagrees with completed / duration")
    return {**metrics, **expected_counts, "min_input_tokens": min(inputs), "max_input_tokens": max(inputs)}


def summarize(args, runs):
    groups = []
    for concurrency in args.concurrency:
        valid = [run for run in runs if run["concurrency"] == concurrency and run["valid"]]
        row = {
            "concurrency": concurrency,
            "num_prompts": requests_per_trial(args, concurrency),
            "valid_repeats": len(valid),
            "eligible": len(valid) == args.repeats,
        }
        if valid:
            for key in ("output_throughput", "request_throughput", *LATENCY_FIELDS):
                row[f"median_{key}"] = statistics.median(run[key] for run in valid)
            # Require every repetition to meet the supplied latency ceiling.
            row["meets_slo"] = all(
                (args.max_p99_ttft_ms is None or run["p99_ttft_ms"] <= args.max_p99_ttft_ms)
                and (args.max_p99_tpot_ms is None or run["p99_tpot_ms"] <= args.max_p99_tpot_ms)
                for run in valid
            )
        groups.append(row)
    complete = len(runs) == len(args.concurrency) * args.repeats and all(run["valid"] for run in runs)
    candidates = [row for row in groups if row["eligible"]] if complete else []
    slo_requested = args.max_p99_ttft_ms is not None or args.max_p99_tpot_ms is not None
    return {
        "complete": complete,
        "peak_measured": max(candidates, key=lambda row: row["median_output_throughput"], default=None),
        "peak_measured_with_slo": max(
            [row for row in candidates if row["meets_slo"]] if slo_requested else [],
            key=lambda row: row["median_output_throughput"],
            default=None,
        ),
        "by_concurrency": groups,
        "runs": runs,
    }


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def client_metadata():
    versions = {}
    for package in ("vllm", "vllm-ascend", "torch", "torch-npu"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        revision = None
    return {"python": sys.version, "packages": versions, "repository_head": revision}


def runtime_context(args):
    """Match a supplied startup record to a live model listing, without asserting execution."""
    if args.runtime_record is None:
        return None
    raw = args.runtime_record.read_bytes()
    record = json.loads(raw)
    if not isinstance(record, dict) or record.get("schema_version") != 1 or record.get("status") != "preflight_passed":
        raise ValueError("runtime-record must be a schema 1 preflight_passed server startup record")
    backend, fusion, execution = (record.get(key) for key in ("gs32_backend", "gs32_fusion", "gs32_execution"))
    if backend != "native" or fusion != "full_ffn":
        raise ValueError("runtime-record requires native/full_ffn")
    if execution not in ("fused", "full_graph"):
        raise ValueError("runtime-record requires fused/full_graph execution")
    if not isinstance(record.get("model_path"), str) or not record["model_path"].strip():
        raise ValueError("runtime-record must include model_path")
    validate_artifact_record(record.get("native_artifacts"))
    with urlopen(args.base_url + "/v1/models", timeout=30) as response:
        snapshot = json.load(response)
    models = snapshot.get("data") if isinstance(snapshot, dict) else None
    if not isinstance(models, list):
        raise ValueError("/v1/models did not return a model list")
    matches = [model for model in models if isinstance(model, dict) and model.get("id") == args.model]
    if len(matches) != 1 or matches[0].get("root") != record["model_path"]:
        raise ValueError("runtime-record model_path must match the unique served model root from /v1/models")
    return {
        "record": record,
        "raw_record": raw,
        "models": snapshot,
        "model_path": record["model_path"],
        "source": {
            "kind": "server_startup_record",
            "file": "service-runtime.json",
            "sha256": hashlib.sha256(raw).hexdigest(),
            "models_snapshot": "models.json",
            "match": "requested and benchmark model_id == service id; root == startup model_path",
            "scope": "Startup configuration and artifact identity; not live HTTP proof of kernel execution.",
        },
    }


def attach_runtime_context(report, context):
    if context is None:
        return report
    report["model_path"] = context["model_path"]
    report["model_path_source"] = {"kind": "service_models_snapshot", "file": "models.json"}
    report["server_runtime"] = context["record"]
    report["server_runtime_source"] = context["source"]
    for row in report["by_concurrency"]:
        row["model_path"] = context["model_path"]
    return report


def main(argv=None):
    args = parse_args(argv)
    try:
        context = None if args.dry_run else runtime_context(args)
    except (OSError, ValueError) as error:
        print(f"Runtime record check failed: {error}", file=sys.stderr)
        return 1
    prefix = f"tp{args.tp}-dp{args.dp}-ep{args.ep_size}-b{args.max_num_seqs}"
    if args.dry_run:
        directory = args.result_dir.resolve() / f"{prefix}-dry-run"
    else:
        args.result_dir.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix=prefix + "-", dir=args.result_dir)).resolve()
    print(f"Declared TP={args.tp} DP={args.dp} EP={args.ep_size}; global sequence cap={args.global_sequence_cap}")
    requests_by_concurrency = {concurrency: requests_per_trial(args, concurrency) for concurrency in args.concurrency}
    print(
        f"Requests per trial by concurrency={requests_by_concurrency}; "
        f"warmups per trial={args.num_warmups}; repeats={args.repeats}"
    )
    if max(args.concurrency) > args.global_sequence_cap:
        print("Client concurrency exceeds declared sequence capacity: this includes server queuing.")
    if not args.dry_run:
        if context is not None:
            (directory / "service-runtime.json").write_bytes(context["raw_record"])
            write_json(directory / "models.json", context["models"])
        write_json(
            directory / "manifest.json",
            {
                "created_utc": datetime.now(UTC).isoformat(),
                "declared_server_and_workload": {
                    key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
                },
                "client_environment": client_metadata(),
                "requests_by_concurrency": requests_by_concurrency,
                "scope": "Measured output TPS for this workload; topology is user-declared, not remotely verified.",
            },
        )
    runs = []
    for repeat in range(1, args.repeats + 1):
        for concurrency in args.concurrency:
            result_path = directory / f"c{concurrency}-r{repeat}.json"
            command = bench_command(args, concurrency, result_path)
            print(shlex.join(command), flush=True)
            if args.dry_run:
                continue
            run = {
                "concurrency": concurrency,
                "num_prompts": requests_per_trial(args, concurrency),
                "num_warmups": args.num_warmups,
                "repeat": repeat,
                "valid": False,
                "result_file": result_path.name,
            }
            try:
                with result_path.with_suffix(".log").open("w", encoding="utf-8") as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout, check=True)
                result = json.loads(result_path.read_text(encoding="utf-8"))
                if context is not None:
                    if not isinstance(result, dict) or result.get("model_id") != args.model:
                        raise ValueError("Benchmark model_id does not match the verified served model")
                    run["model_path"] = context["model_path"]
                run.update(validate_result(result, args, concurrency))
                run["valid"] = True
                print(f"C={concurrency} repeat={repeat}: output TPS={run['output_throughput']:.3f}", flush=True)
            except (OSError, ValueError, subprocess.SubprocessError, KeyboardInterrupt) as error:
                run["error"] = f"{type(error).__name__}: {error}"
                print(f"Benchmark stopped: {run['error']}; inspect {result_path.with_suffix('.log')}", file=sys.stderr)
            runs.append(run)
            report = attach_runtime_context(summarize(args, runs), context)
            write_json(directory / "summary.json", report)
            if not run["valid"]:
                return 1
    if args.dry_run:
        return 0
    with (directory / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report["by_concurrency"][0]))
        writer.writeheader()
        writer.writerows(report["by_concurrency"])
    peak = report["peak_measured"]
    print(f"Peak measured median output TPS={peak['median_output_throughput']:.3f} at C={peak['concurrency']}")
    print(f"Results: {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
