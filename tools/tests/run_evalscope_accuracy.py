# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Run pinned EvalScope accuracy benchmarks against an existing vLLM service."""

import argparse
import csv
import json
import math
import os
import shlex
import signal
import subprocess
import sys
import tempfile
from contextlib import suppress
from datetime import datetime
from importlib import metadata
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from summarize_evalscope_accuracy import DATASET_SIZES, summarize_run  # noqa: E402

from tools.vq2a8_live_log import LiveChildLog  # noqa: E402

EVALSCOPE_VERSION = "1.12.0"
REPO_ROOT = Path(__file__).resolve().parents[2]
SERVICE_CHECK_TIMEOUT = 10
TERMINATE_GRACE_SECONDS = 10
DEFAULT_DATASETS = ("gpqa_diamond", "aime24", "aime25")


def dataset_name(value: str) -> str:
    name = "gpqa_diamond" if value.lower() == "gpqa" else value.lower()
    if name not in DATASET_SIZES:
        raise argparse.ArgumentTypeError("choose gpqa (GPQA Diamond), gpqa_diamond, aime24 or aime25")
    return name


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=("baseline", "vq2a8"))
    parser.add_argument("--model", help="Served model name (default: target)")
    endpoint = parser.add_mutually_exclusive_group()
    endpoint.add_argument("--api-url", help="OpenAI API root, including /v1")
    endpoint.add_argument("--base-url", help="Service root, without /v1 (same convention as performance suite)")
    parser.add_argument("--datasets", nargs="+", type=dataset_name, default=list(DEFAULT_DATASETS))
    data_source = parser.add_mutually_exclusive_group()
    data_source.add_argument("--dataset-dir", type=Path, help="Dataset cache root; may download missing data")
    data_source.add_argument(
        "--offline-data-dir", type=Path, help="Verified offline bundle exported by prepare_evalscope_datasets.py"
    )
    parser.add_argument("--concurrency", type=positive_int, default=4)
    parser.add_argument("--max-tokens", type=positive_int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, help="Omit to keep the service default")
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument(
        "--thinking",
        dest="thinking",
        action="store_const",
        const=True,
        help="Enable thinking; defaults --reasoning-effort to high when omitted",
    )
    thinking.add_argument(
        "--no-thinking",
        dest="thinking",
        action="store_const",
        const=False,
        help="Disable thinking; equivalent to --reasoning-effort none",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "high", "max"),
        help="Default: service setting unless --thinking/--no-thinking is selected",
    )
    parser.add_argument("--repeats", type=positive_int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--limit", type=positive_int, help="Smoke test: questions per dataset; omit for full evaluation"
    )
    parser.add_argument("--timeout", type=positive_int, default=3600, help="Per-request timeout in seconds")
    parser.add_argument(
        "--work-dir",
        "--result-dir",
        dest="work_dir",
        type=Path,
        help="Output root; a new subdirectory is created for each run",
    )
    parser.add_argument("--label", default="", help="Run label, e.g. mrv1-ep2-none or bf16-baseline")
    parser.add_argument(
        "--server-revision", default="", help="Optional server commit/build identifier supplied by operator"
    )
    parser.add_argument("--dry-run", action="store_true", help="Print command without contacting the service")
    args = parser.parse_args(argv)
    if args.thinking is True:
        if args.reasoning_effort == "none":
            parser.error("--thinking conflicts with --reasoning-effort none")
        args.reasoning_effort = args.reasoning_effort or "high"
    elif args.thinking is False:
        if args.reasoning_effort not in (None, "none"):
            parser.error("--no-thinking requires --reasoning-effort none or no explicit effort")
        args.reasoning_effort = "none"
    if args.reasoning_effort is None:
        args.thinking_mode = "service_default"
    else:
        args.thinking_mode = "no_thinking" if args.reasoning_effort == "none" else "thinking"
    if not math.isfinite(args.temperature) or args.temperature < 0:
        parser.error("--temperature must be finite and nonnegative")
    if args.top_p is not None and (not math.isfinite(args.top_p) or not 0 < args.top_p <= 1):
        parser.error("--top-p must be in (0, 1]")
    if len(set(args.datasets)) != len(args.datasets):
        parser.error("--datasets must not contain duplicates")
    if args.dataset_dir is not None:
        args.dataset_dir = args.dataset_dir.expanduser().resolve()
    if args.offline_data_dir is not None:
        args.offline_data_dir = args.offline_data_dir.expanduser().resolve()
        args.dataset_dir = args.offline_data_dir
    args.model = args.target if args.model is None else args.model
    if not args.model.strip():
        parser.error("--model must not be empty")
    port = 8001 if args.target == "baseline" else 8000
    args.api_url = (
        args.api_url or (args.base_url.rstrip("/") + "/v1" if args.base_url else f"http://127.0.0.1:{port}/v1")
    ).rstrip("/")
    try:
        url = urlsplit(args.api_url)
        valid_url = (
            url.scheme in ("http", "https")
            and url.hostname
            and url.path == "/v1"
            and not (url.username or url.password or url.query or url.fragment)
            and not any(char.isspace() for char in args.api_url)
        )
        _ = url.port
    except ValueError:
        valid_url = False
    if not valid_url:
        parser.error("--api-url must be a plain API root such as http://127.0.0.1:8001/v1")
    args.work_dir = (
        (args.work_dir or REPO_ROOT / "benchmark-results" / "evalscope" / args.target).expanduser().resolve()
    )
    return args


def generation_config(args: argparse.Namespace) -> dict:
    config = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "timeout": args.timeout,
        "stream": False,
        "extra_body": {"ignore_eos": False},
    }
    if args.top_p is not None:
        config["top_p"] = args.top_p
    if args.reasoning_effort is not None:
        config["reasoning_effort"] = args.reasoning_effort
        # The DeepSeek V4 frontend reads these kwargs when encoding chat messages.
        config["extra_body"]["chat_template_kwargs"] = {
            "thinking": args.reasoning_effort != "none",
            "reasoning_effort": args.reasoning_effort,
        }
    return config


def build_command(args: argparse.Namespace, run_dir: Path) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "-m",
        "evalscope.cli.cli",
        "eval",
        "--model",
        args.model,
        "--api-url",
        args.api_url,
        "--api-key",
        "EMPTY",
        "--eval-type",
        "openai_api",
        "--datasets",
        *args.datasets,
        "--eval-batch-size",
        str(args.concurrency),
        "--repeats",
        str(args.repeats),
        "--seed",
        str(args.seed),
        "--generation-config",
        json.dumps(generation_config(args)),
        "--judge",
        json.dumps({"strategy": "rule"}),
        "--work-dir",
        str(run_dir),
        "--no-timestamp",
    ]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.dataset_dir is not None:
        command.extend(["--dataset-dir", str(args.dataset_dir)])
    return command


def check_evalscope_version() -> str:
    install = f"{shlex.quote(sys.executable)} -m pip install 'evalscope=={EVALSCOPE_VERSION}'"
    try:
        version = metadata.version("evalscope")
    except metadata.PackageNotFoundError as exc:
        raise ValueError(f"EvalScope is not installed in this Python environment. Install it with: {install}") from exc
    if version != EVALSCOPE_VERSION:
        raise ValueError(
            f"Expected EvalScope {EVALSCOPE_VERSION}, found {version}. Use a separate venv and run: {install}"
        )
    return version


def check_service(args: argparse.Namespace) -> dict:
    request = Request(f"{args.api_url}/models", headers={"Authorization": "Bearer EMPTY"})
    try:
        with urlopen(request, timeout=SERVICE_CHECK_TIMEOUT) as response:
            payload = json.load(response)
    except (OSError, URLError, ValueError) as exc:
        raise ValueError(f"Cannot query {args.api_url}/models: {exc}. Start the model service first.") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError("Service /models response does not contain a model list")
    models = [item for item in payload["data"] if isinstance(item, dict)]
    model = next((item for item in models if item.get("id") == args.model), None)
    if model is None:
        raise ValueError(
            f"Service does not expose {args.model!r}; available models: {[item.get('id') for item in models]}"
        )
    max_model_len = model.get("max_model_len")
    if isinstance(max_model_len, int) and max_model_len <= args.max_tokens:
        raise ValueError(
            f"Service max_model_len={max_model_len} leaves no room for a prompt with max_tokens={args.max_tokens}. "
            "Increase the server context length or lower --max-tokens."
        )
    return model


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def client_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL, timeout=5
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def stop_evaluator(process: subprocess.Popen) -> None:
    """Stop only this evaluation's child process group, including workers."""
    if os.name == "posix":
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
    elif process.poll() is None:
        process.terminate()
    with suppress(subprocess.TimeoutExpired):
        process.wait(timeout=TERMINATE_GRACE_SECONDS)
    if os.name == "posix":
        # The evaluator may have exited before its dataset worker processes.
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
    elif process.poll() is None:
        process.kill()
    process.wait()


def execute_evaluator(command: list[str], log: Path) -> int:
    """Keep complete child output on disk while mirroring it to the terminal."""
    with log.open("wb") as output, LiveChildLog(log, "evalscope"):
        process = subprocess.Popen(
            command,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=os.name == "posix",
        )
        try:
            return process.wait()
        finally:
            # Close all child writers before LiveChildLog drains its final bytes.
            stop_evaluator(process)


def write_scores(run_dir: Path, manifest: dict, audit: dict) -> None:
    """Publish the audited native scores, keeping smoke/full and sampling conditions visible."""
    summary = {
        "status": "invalid" if audit["invalid"] else "finished",
        "scope": manifest["scope"],
        "label": manifest["label"],
        "model": manifest["model"],
        "client_revision": manifest["client_revision"],
        "server_revision": manifest["server_revision"],
        "evalscope_version": manifest["evalscope_version"],
        "generation_config": manifest["generation_config"],
        "thinking_mode": manifest["thinking_mode"],
        "repeats": manifest["repeats"],
        "datasets": audit["datasets"],
    }
    write_json(run_dir / "summary.json", summary)
    fields = ("dataset", "score", "records", "expected_records", "max_tokens", "empty_final_content", "invalid")
    with (run_dir / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, item in audit["datasets"].items():
            writer.writerow({"dataset": name, **{field: item[field] for field in fields[1:]}})


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        print(shlex.join(build_command(args, args.work_dir / "DRY_RUN")))
        return 0
    run_dir = None
    manifest = {}

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    previous_sigterm = signal.signal(signal.SIGTERM, interrupt) if os.name == "posix" else None
    try:
        version = check_evalscope_version()
        offline_manifest = None
        if args.offline_data_dir is not None:
            # Optional EvalScope/datasets imports are needed only for actual offline evaluation.
            from prepare_evalscope_datasets import validate_offline_bundle

            offline_manifest = validate_offline_bundle(args.offline_data_dir, args.datasets)
        service_model = check_service(args)
        args.work_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(tempfile.mkdtemp(prefix=f"{stamp}-", dir=args.work_dir))
        command = build_command(args, run_dir)
        expected_counts = {
            dataset: min(DATASET_SIZES[dataset], args.limit or DATASET_SIZES[dataset]) * args.repeats
            for dataset in args.datasets
        }
        manifest = {
            "evalscope_version": version,
            "client_revision": client_revision(),
            "server_revision": args.server_revision or None,
            "label": args.label,
            "scope": "smoke" if args.limit is not None else "full",
            "limit": args.limit,
            "repeats": args.repeats,
            "python": sys.executable,
            "target": args.target,
            "model": args.model,
            "api_url": args.api_url,
            "service_model": service_model,
            "datasets": args.datasets,
            "dataset_dir": str(args.dataset_dir) if args.dataset_dir is not None else None,
            "offline_bundle": offline_manifest,
            "expected_counts": expected_counts,
            "concurrency": args.concurrency,
            "generation_config": generation_config(args),
            "thinking_mode": args.thinking_mode,
            "command": command,
            "log": str(run_dir / "evalscope.log"),
            "status": "running",
        }
        write_json(run_dir / "run.json", manifest)
        print(f"EvalScope run: {run_dir}", flush=True)
        print(f"Datasets: {expected_counts}; concurrent requests: {args.concurrency}", flush=True)
        print(
            f"Thinking mode: {args.thinking_mode}; reasoning effort: {args.reasoning_effort or 'service_default'}",
            flush=True,
        )
        print(f"Live progress below; complete child stdout/stderr: {manifest['log']}", flush=True)
        print(f"Native EvalScope logs: {run_dir / 'logs'}", flush=True)
        returncode = execute_evaluator(command, Path(manifest["log"]))
        manifest.update(returncode=returncode, status="failed" if returncode else "auditing")
        write_json(run_dir / "run.json", manifest)
        if returncode:
            print(f"EvalScope failed ({returncode}); inspect {manifest['log']}", file=sys.stderr)
            return returncode if returncode > 0 else 1
        try:
            summary = summarize_run(run_dir, args.datasets, expected_counts)
        except ValueError as exc:
            write_json(run_dir / "accuracy_audit.json", {"invalid": True, "error": str(exc)})
            manifest["status"] = "invalid"
            write_json(run_dir / "run.json", manifest)
            raise
        write_json(run_dir / "accuracy_audit.json", summary)
        write_scores(run_dir, manifest, summary)
        manifest["status"] = "invalid" if summary["invalid"] else "finished"
        write_json(run_dir / "run.json", manifest)
        for dataset, item in summary["datasets"].items():
            print(
                f"{dataset}: score={item['score']:.6f}, records={item['records']}/{item['expected_records']}, "
                f"max_tokens={item['max_tokens']}, empty_final={item['empty_final_content']}, "
                f"unknown_usage={item['output_tokens']['unknown_records']}, invalid={item['invalid']}"
            )
        print(f"SUMMARY={run_dir / 'summary.json'}; CSV={run_dir / 'summary.csv'}")
        print(f"Native scores: {run_dir / 'reports'}; response audit: {run_dir / 'accuracy_audit.json'}")
        if summary["invalid"]:
            print("Incomplete or invalid responses detected; inspect accuracy_audit.json before comparing scores.")
        return int(summary["invalid"])
    except KeyboardInterrupt:
        if run_dir is not None:
            manifest.update(status="interrupted", returncode=130)
            write_json(run_dir / "run.json", manifest)
        print("EvalScope accuracy interrupted; partial results are not a completed evaluation.", file=sys.stderr)
        return 130
    except (OSError, ValueError) as exc:
        if run_dir is not None:
            manifest.update(status="invalid" if manifest.get("status") == "invalid" else "failed", error=str(exc))
            write_json(run_dir / "run.json", manifest)
        print(f"EvalScope accuracy failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if os.name == "posix":
            signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    raise SystemExit(main())
