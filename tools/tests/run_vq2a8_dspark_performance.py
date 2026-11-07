#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check an existing server, then measure DSpark or baseline throughput.

Reuses run_vq2a8_throughput.py and its result validation. Defaults to C=1/2/4,
four measured requests per concurrency, one repeat, 16->16 then 128->512.
The declared variant/topology label does not inspect or configure the server.
"""

import argparse
import json
import math
import shlex
import subprocess
import sys
import tempfile
from http.client import HTTPException
from pathlib import Path
from urllib.request import Request, urlopen

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.tests import run_vq2a8_throughput as throughput  # noqa: E402

SMOKE_TOKENS = 16
BENCHMARK_SCRIPT = Path(__file__).with_name("run_vq2a8_throughput.py")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8003")
    parser.add_argument("--model", default="vq2a8", help="Served model name")
    parser.add_argument("--tokenizer", required=True, help="Target checkpoint/tokenizer on the benchmark host")
    parser.add_argument("--tokenizer-mode", default="deepseek_v4")
    parser.add_argument(
        "--variant",
        choices=("dspark", "baseline"),
        default="dspark",
        help="Declared server mode, not remotely verified",
    )
    parser.add_argument("--parallel-mode", choices=("dp-ep", "tp-ep"), default="dp-ep")
    parser.add_argument("--ep-size", type=int, choices=(1, 2, 4, 8), default=4)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--num-prompts", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=1, help="Repeats of each concurrency trial")
    parser.add_argument("--input-len", type=int, default=128, help="Main throughput workload input length")
    parser.add_argument("--output-len", type=int, default=512, help="Main throughput workload output length")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=7200, help="Wall timeout for each underlying benchmark trial")
    parser.add_argument("--request-timeout", type=float, default=300, help="Timeout for readiness/smoke HTTP requests")
    parser.add_argument("--result-dir", type=Path, default=Path("benchmark-results/performance-vq2a8-dspark"))
    parser.add_argument("--note", default="", help="Record actual server budget, KV, hardware and build settings")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the complete plan without HTTP, files or subprocesses"
    )
    args = parser.parse_args(argv)
    if not math.isfinite(args.request_timeout) or args.request_timeout <= 0:
        parser.error("--request-timeout must be finite and positive")
    # Reuse the existing runner's URL, topology, concurrency and count checks
    # for BOTH stages before any requests or output directories are created.
    args.base_url = args.base_url.rstrip("/")
    for _, stage_args in benchmark_stages(args, args.result_dir):
        throughput.parse_args(stage_args)
    return args


def benchmark_stages(args, directory):
    common = [
        "--base-url",
        args.base_url,
        "--model",
        args.model,
        "--tokenizer",
        args.tokenizer,
        "--tokenizer-mode",
        args.tokenizer_mode,
        "--parallel-mode",
        args.parallel_mode,
        "--ep-size",
        str(args.ep_size),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--concurrency",
        *map(str, args.concurrency),
        "--num-prompts",
        str(args.num_prompts),
        "--repeats",
        str(args.repeats),
        "--seed",
        str(args.seed),
        "--timeout",
        str(args.timeout),
        "--note",
        f"declared_variant={args.variant}; {args.note}",
    ]
    return [
        (
            name,
            [*common, "--input-len", str(inputs), "--output-len", str(outputs), "--result-dir", str(directory / name)],
        )
        for name, inputs, outputs in (
            ("smoke-throughput", SMOKE_TOKENS, SMOKE_TOKENS),
            ("throughput", args.input_len, args.output_len),
        )
    ]


def request(args, endpoint, payload=None):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = Request(args.base_url + endpoint, data=data, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=args.request_timeout) as response:
        if response.status != 200:
            raise ValueError(f"{endpoint} returned HTTP {response.status}, expected 200")
        return response.read()


def check_server(args, directory):
    request(args, "/health")
    models = json.loads(request(args, "/v1/models"))
    throughput.write_json(directory / "models.json", models)
    available = models.get("data") if isinstance(models, dict) else None
    if not isinstance(available, list) or not any(
        isinstance(model, dict) and model.get("id") == args.model for model in available
    ):
        raise ValueError(f"Server does not advertise model {args.model!r}")
    smoke = json.loads(
        request(
            args,
            "/v1/completions",
            {
                "model": args.model,
                "prompt": "Hello",
                "max_tokens": SMOKE_TOKENS,
                "temperature": 0,
            },
        )
    )
    throughput.write_json(directory / "smoke.json", smoke)
    choices = smoke.get("choices", []) if isinstance(smoke, dict) else []
    text = choices[0].get("text") if isinstance(choices, list) and choices and isinstance(choices[0], dict) else None
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Smoke completion returned no nonempty text; benchmark not started")
    print("DSPARK_PERFORMANCE_PREFLIGHT=passed", flush=True)


def main(argv=None):
    args = parse_args(argv)
    if args.dry_run:
        print(f"Preflight: GET {args.base_url}/health; GET /v1/models; POST /v1/completions (Hello, 16 tokens)")
        for name, stage_args in benchmark_stages(args, args.result_dir / f"{args.variant}-dry-run"):
            print(f"Stage: {name}", flush=True)
            throughput.main([*stage_args, "--dry-run"])
        return 0
    args.result_dir.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=args.variant + "-", dir=args.result_dir)).resolve()
    stages = benchmark_stages(args, directory)
    report = {
        "status": "running",
        "declared_settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "scope": "Variant and topology are user-declared; compare matching workloads on the same server configuration.",
        "stages": [{"name": name, "status": "not_run"} for name, _ in stages],
    }
    manifest = directory / "run.json"
    throughput.write_json(manifest, report)
    print(f"RESULT_DIR={directory}", flush=True)
    try:
        check_server(args, directory)
        for record, (_, stage_args) in zip(report["stages"], stages, strict=True):
            command = [sys.executable, "-u", str(BENCHMARK_SCRIPT), *stage_args]
            record.update(status="running", command=command)
            throughput.write_json(manifest, report)
            print(shlex.join(command), flush=True)
            subprocess.run(command, check=True)
            record["status"] = "passed"
            throughput.write_json(manifest, report)
        report["status"] = "passed"
    except (OSError, ValueError, TypeError, HTTPException, subprocess.SubprocessError, KeyboardInterrupt) as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        for record in report["stages"]:
            if record["status"] == "running":
                record["status"] = "failed"
        throughput.write_json(manifest, report)
        print(f"DSPARK_PERFORMANCE_STATUS=failed REPORT={manifest}: {error}", file=sys.stderr, flush=True)
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    throughput.write_json(manifest, report)
    print(f"DSPARK_PERFORMANCE_STATUS=passed REPORT={manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
