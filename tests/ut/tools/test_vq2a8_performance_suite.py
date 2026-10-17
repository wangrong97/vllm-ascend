# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the serving suite with a real local HTTP/SSE server, without NPUs."""

import csv
import importlib.util
import json
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOKEN_IDS = [101, 102, 103]
QUICK_CASES = ["128x32", "127x32", "129x32", "224x32", "128x128"]
CASE_SETTINGS = {
    "128x32": (128, 32, 0, 2),
    "127x32": (127, 32, 0, 2),
    "129x32": (129, 32, 0, 2),
    "224x32": (224, 32, 0, 2),
    "128x128": (128, 128, 1, 10),
    "64k": (65536, 1024, 0, 2),
    "256k": (262144, 1024, 0, 2),
    "1m": (1048576, 1024, 0, 2),
    "1m-window": (1047552, 1024, 0, 2),
    "recovery": (128, 128, 0, 3),
}


@pytest.fixture
def suite_server(tmp_path):
    repo = tmp_path / "test repository"
    (repo / "tools/tests").mkdir(parents=True)
    for relative in (
        "tools/tests/run_vq2a8_performance_suite.py",
        "tools/tests/run_vq2a8_ep2_long_context_latency.py",
        "tools/benchmark_vq2a8_serving.py",
        "tools/vq2a8_live_log.py",
    ):
        shutil.copyfile(REPO_ROOT / relative, repo / relative)
    results = tmp_path / "results with spaces"
    requests = []
    state = NS(context=1049600, failure=None, failure_on=1, generations=0, active=0, peak=0)
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with lock:
                requests.append((self.path, payload))
            if self.path == "/tokenize":
                tokens = [] if state.failure == "tokenize" else TOKEN_IDS
                body = json.dumps({"tokens": tokens, "count": len(tokens), "max_model_len": state.context}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path != "/v1/completions":
                self.send_error(404)
                return
            with lock:
                state.generations += 1
                index = state.generations
                state.active += 1
                state.peak = max(state.peak, state.active)
            failure = state.failure if index == state.failure_on else None
            try:
                if failure == "http":
                    self.send_error(503, "Engine unavailable")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                # Send two text chunks, regardless of requested token count.
                # Usage, rather than SSE chunk count, determines token timing.
                time.sleep(0.004)
                self.wfile.write(b'data: {"choices": [{"index": 0, "text": "first"}]}\n\n')
                self.wfile.flush()
                time.sleep(0.004)
                events = [
                    {"choices": [{"index": 0, "text": "last", "finish_reason": "length"}]},
                    {
                        "choices": [],
                        "usage": {
                            "prompt_tokens": len(payload["prompt"]) - (failure == "prompt_usage"),
                            "completion_tokens": payload["max_tokens"] - (failure == "output_usage"),
                        },
                    },
                ]
                for event in events:
                    self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                if failure != "truncated":
                    self.wfile.write(b"data: [DONE]\n\n")
            finally:
                with lock:
                    state.active -= 1

        def log_message(self, format, *args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
        thread.start()
        base_url = f"http://127.0.0.1:{server.server_port}/v1"

        def run(*arguments, failure=None, failure_on=1, context=1049600):
            state.context, state.failure, state.failure_on = context, failure, failure_on
            state.generations = 0
            before = set(results.glob("vq2a8-performance-*/summary.json"))
            completed = subprocess.run(
                [
                    sys.executable,
                    str(repo / "tools/tests/run_vq2a8_performance_suite.py"),
                    "--base-url",
                    base_url,
                    "--result-dir",
                    str(results),
                    *arguments,
                ],
                cwd=tmp_path,
                env={**os.environ, "PYTHONUTF8": "1", "NO_PROXY": "127.0.0.1", "no_proxy": "127.0.0.1"},
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=False,
                timeout=30,
            )
            created = set(results.glob("vq2a8-performance-*/summary.json")) - before
            assert len(created) <= 1, completed.stdout + completed.stderr
            if not created:
                return NS(process=completed, report=None, directory=None)
            summary_path = created.pop()
            return NS(
                process=completed,
                report=json.loads(summary_path.read_text(encoding="utf-8")),
                directory=summary_path.parent,
            )

        try:
            yield NS(run=run, requests=requests, state=state, results=results, repo=repo, base_url=base_url)
        finally:
            server.shutdown()
            thread.join(timeout=5)


def assert_saved_cases(result):
    cases = result.report["cases"]
    with (result.directory / "summary.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(cases)
    required = {
        "status",
        "input_tokens",
        "output_tokens",
        "measured_requests",
        "warmup_requests",
        "first_ttft_ms",
        "mean_ttft_ms",
        "median_ttft_ms",
        "mean_tpot_ms",
        "median_tpot_ms",
        "error",
    }
    for record, row in zip(cases, rows, strict=True):
        assert required <= row.keys()
        assert row["status"] == record["status"]
        assert int(row["input_tokens"]) == record["input_tokens"]
        assert int(row["output_tokens"]) == record["output_tokens"]
        assert int(row["measured_requests"]) == len(record["requests"])
        assert int(row["warmup_requests"]) == len(record.get("warmup_requests", []))
        saved = json.loads((result.directory / f"{record['case']}.json").read_text(encoding="utf-8"))
        assert saved == record
        if record["status"] != "not_run":
            assert (result.directory / f"{record['case']}.log").is_file()
    return rows


@pytest.mark.parametrize(
    "suite, expected",
    [
        ("quick", QUICK_CASES),
        ("standard", QUICK_CASES + ["64k", "256k", "recovery"]),
        ("full", QUICK_CASES + ["64k", "256k", "1m", "recovery"]),
    ],
)
def test_presets_preflight_all_cases_and_record_defaults_without_generation(suite_server, suite, expected):
    result = suite_server.run("--suite", suite, "--check-only", "--label", "候选 V2")
    assert result.process.returncode == 0, result.process.stdout + result.process.stderr
    assert [record["case"] for record in result.report["cases"]] == expected
    assert result.report["suite"] == suite and result.report["label"] == "候选 V2"
    assert result.report["check_only"] is True
    assert "client_git_revision" in result.report
    assert [path for path, _ in suite_server.requests] == ["/tokenize"] * len(expected)
    for record in result.report["cases"]:
        assert (record["input_tokens"], record["output_tokens"], record["warmups"], record["repeats"]) == CASE_SETTINGS[
            record["case"]
        ]
        assert record["status"] == "ready" and record["requests"] == []
        assert "summary" not in record
    assert_saved_cases(result)


def test_default_suite_is_standard_and_offline_dry_run_creates_nothing(suite_server):
    result = suite_server.run("--dry-run")
    assert result.process.returncode == 0, result.process.stdout + result.process.stderr
    assert suite_server.requests == []
    assert not suite_server.results.exists()
    assert result.report is None
    for name in QUICK_CASES + ["64k", "256k", "recovery"]:
        assert name in result.process.stdout
    assert "262144" in result.process.stdout
    assert "1048576" not in result.process.stdout


@pytest.mark.parametrize(
    "arguments",
    [
        ("--cases", "128x32", "128x32"),
        ("--warmups", "-1"),
        ("--repeats", "0"),
        ("--timeout", "nan"),
        ("--model", ""),
        ("--prompt", ""),
    ],
)
def test_invalid_configuration_fails_before_http_and_result_creation(suite_server, arguments):
    result = suite_server.run(*arguments, "--dry-run")
    assert result.process.returncode != 0
    assert suite_server.requests == []
    assert not suite_server.results.exists()


def test_exact_short_cases_preflight_before_generation_and_override_counts(suite_server):
    names = QUICK_CASES + ["recovery"]
    result = suite_server.run(
        "--suite",
        "full",
        "--cases",
        *names,
        "--warmups",
        "1",
        "--repeats",
        "2",
        "--model",
        "candidate",
        "--prompt",
        "different seed",
        "--timeout",
        "5",
    )
    assert result.process.returncode == 0, result.process.stdout + result.process.stderr
    assert [record["case"] for record in result.report["cases"]] == names
    assert [path for path, _ in suite_server.requests] == ["/tokenize"] * len(names) + ["/v1/completions"] * (
        3 * len(names)
    )
    assert suite_server.state.peak == 1
    for _, payload in suite_server.requests[: len(names)]:
        assert payload == {"model": "candidate", "prompt": "different seed", "add_special_tokens": False}
    rows = assert_saved_cases(result)
    for index, (record, row) in enumerate(zip(result.report["cases"], rows, strict=True)):
        input_tokens, output_tokens, _, _ = CASE_SETTINGS[record["case"]]
        assert record["status"] == "passed"
        assert record["warmups"] == 1 and record["repeats"] == 2
        assert len(record["requests"]) == 2 and len(record["warmup_requests"]) == 1
        first = record["warmup_requests"][0]
        assert float(row["first_ttft_ms"]) == pytest.approx(first["ttft_ms"], abs=0.001)
        for aggregation, operation in (("mean", statistics.mean), ("median", statistics.median)):
            for metric in ("ttft_ms", "tpot_ms"):
                value = operation(request[metric] for request in record["requests"])
                key = f"{aggregation}_{metric}"
                assert record["summary"][key] == value
                assert float(row[key]) == pytest.approx(value, abs=0.001)
        for _, payload in suite_server.requests[len(names) + index * 3 : len(names) + (index + 1) * 3]:
            assert payload["prompt"] == (TOKEN_IDS * ((input_tokens + 2) // 3))[:input_tokens]
            assert payload["max_tokens"] == output_tokens and payload["model"] == "candidate"
            assert payload["ignore_eos"] is True and payload["n"] == 1
            assert payload["stream"] is True and payload["stream_options"] == {"include_usage": True}
            assert payload["add_special_tokens"] is False and payload["skip_special_tokens"] is False


@pytest.mark.parametrize("case", ["64k", "256k", "1m", "1m-window"])
def test_long_cases_send_exact_input_and_output_including_full_1m(suite_server, case):
    input_tokens, output_tokens, _, _ = CASE_SETTINGS[case]
    result = suite_server.run("--cases", case, "--warmups", "0", "--repeats", "1", context=input_tokens + output_tokens)
    assert result.process.returncode == 0, result.process.stdout + result.process.stderr
    assert [path for path, _ in suite_server.requests] == ["/tokenize", "/v1/completions"]
    payload = suite_server.requests[-1][1]
    assert payload["prompt"] == (TOKEN_IDS * ((input_tokens + 2) // 3))[:input_tokens]
    assert payload["max_tokens"] == 1024
    record = result.report["cases"][0]
    assert record["required_context_tokens"] == input_tokens + output_tokens
    assert record["requests"][0]["prompt_tokens"] == input_tokens
    rows = assert_saved_cases(result)
    assert float(rows[0]["first_ttft_ms"]) == pytest.approx(record["requests"][0]["ttft_ms"], abs=0.001)


@pytest.mark.parametrize("failure", ["capacity", "tokenize"])
def test_any_preflight_failure_prevents_all_generation(suite_server, failure):
    result = suite_server.run(
        "--cases",
        "128x32",
        "1m",
        "129x32",
        "--warmups",
        "0",
        "--repeats",
        "1",
        context=1048576,
        failure="tokenize" if failure == "tokenize" else None,
    )
    assert result.process.returncode != 0
    assert [path for path, _ in suite_server.requests] == ["/tokenize"] * 3
    records = result.report["cases"]
    assert records[1]["status"] == "preflight_failed"
    assert records[1]["input_tokens"] == 1048576
    assert all(record["requests"] == [] and "summary" not in record for record in records)
    if failure == "capacity":
        assert "1049600" in records[1]["error"]
    assert_saved_cases(result)


@pytest.mark.parametrize(
    "failure, error",
    [
        ("http", "HTTP 503"),
        ("truncated", "[DONE]"),
        ("prompt_usage", "prompt_tokens=128"),
        ("output_usage", "completion_tokens=31"),
    ],
)
def test_generation_failure_keeps_completed_case_and_stops_later_cases(suite_server, failure, error):
    result = suite_server.run(
        "--cases",
        "128x32",
        "129x32",
        "recovery",
        "--warmups",
        "0",
        "--repeats",
        "1",
        failure=failure,
        failure_on=2,
    )
    assert result.process.returncode != 0
    assert [path for path, _ in suite_server.requests] == ["/tokenize"] * 3 + ["/v1/completions"] * 2
    passed, failed, later = result.report["cases"]
    assert [record["status"] for record in (passed, failed, later)] == ["passed", "failed", "not_run"]
    assert len(passed["requests"]) == 1 and "summary" in passed
    assert failed["requests"] == later["requests"] == []
    assert "summary" not in failed and "summary" not in later
    assert error in failed["error"]
    assert_saved_cases(result)


def test_failed_warmup_is_not_a_measured_request_and_stops_suite(suite_server):
    result = suite_server.run(
        "--cases",
        "128x128",
        "recovery",
        "--warmups",
        "1",
        "--repeats",
        "1",
        failure="http",
    )
    assert result.process.returncode != 0
    assert suite_server.state.generations == 1
    first, later = result.report["cases"]
    assert first["status"] == "failed" and later["status"] == "not_run"
    assert first["requests"] == [] and first.get("warmup_requests", []) == []
    assert_saved_cases(result)


def test_failure_on_second_measurement_preserves_first_without_success_summary(suite_server):
    result = suite_server.run(
        "--cases",
        "128x32",
        "recovery",
        "--warmups",
        "0",
        "--repeats",
        "3",
        failure="truncated",
        failure_on=2,
    )
    assert result.process.returncode != 0
    assert [path for path, _ in suite_server.requests] == ["/tokenize"] * 2 + ["/v1/completions"] * 2
    first, later = result.report["cases"]
    assert first["status"] == "failed" and later["status"] == "not_run"
    assert len(first["requests"]) == 1 and later["requests"] == []
    assert "summary" not in first and "summary" not in later
    rows = assert_saved_cases(result)
    assert rows[0]["measured_requests"] == "1"
    assert rows[0]["mean_ttft_ms"] == rows[0]["median_tpot_ms"] == ""


def test_multiple_invocations_use_unique_directories_without_overwriting(suite_server):
    first = suite_server.run("--cases", "128x32", "--check-only", "--label", "first")
    first_bytes = (first.directory / "summary.json").read_bytes()
    second = suite_server.run("--cases", "recovery", "--check-only", "--label", "second")
    assert first.process.returncode == second.process.returncode == 0
    assert first.directory != second.directory
    assert (first.directory / "summary.json").read_bytes() == first_bytes
    assert first.report["label"] == "first" and second.report["label"] == "second"
    assert first.directory.name.startswith("vq2a8-performance-")
    assert second.directory.name.startswith("vq2a8-performance-")


def load_suite(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    monkeypatch.syspath_prepend(str(REPO_ROOT))
    path = REPO_ROOT / "tools/tests/run_vq2a8_performance_suite.py"
    spec = importlib.util.spec_from_file_location("vq2a8_performance_suite_interrupt", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_keyboard_interrupt_returns_130_and_preserves_partial_measurements(suite_server, monkeypatch):
    module = load_suite(monkeypatch)
    calls = []
    first_result = {"ttft_ms": 12.0, "tpot_ms": 2.0, "prompt_tokens": 128, "completion_tokens": 32}

    def request(args, phase, index, total):
        calls.append((phase, index))
        if len(calls) == 1:
            return first_result
        raise KeyboardInterrupt

    monkeypatch.setattr(module.bench, "run_request", request)
    code = module.main(
        [
            "--base-url",
            suite_server.base_url,
            "--result-dir",
            str(suite_server.results),
            "--cases",
            "128x32",
            "recovery",
            "--warmups",
            "0",
            "--repeats",
            "2",
        ]
    )
    assert code == 130
    assert calls == [("REQUEST", 1), ("REQUEST", 2)]
    (summary_path,) = suite_server.results.glob("vq2a8-performance-*/summary.json")
    report = json.loads(summary_path.read_text(encoding="utf-8"))
    first, later = report["cases"]
    assert first["status"] == "interrupted" and first["requests"] == [first_result]
    assert "summary" not in first
    assert later["status"] == "not_run" and later["requests"] == []
    assert_saved_cases(NS(report=report, directory=summary_path.parent))


@pytest.mark.parametrize("phase", ["preflight", "warmup"])
def test_interrupt_during_preflight_or_warmup_keeps_evidence_without_measurement(suite_server, monkeypatch, phase):
    module = load_suite(monkeypatch)
    calls = []
    original_preflight = module.bench.prepare_fixed_prompt
    warmup = {"ttft_ms": 12.0, "tpot_ms": 2.0, "prompt_tokens": 128, "completion_tokens": 128}

    def preflight(args):
        calls.append("preflight")
        if len(calls) == 2:
            raise KeyboardInterrupt
        return original_preflight(args)

    def request(args, request_phase, index, total):
        calls.append((request_phase, index))
        assert request_phase == "WARMUP"
        if index == 2:
            raise KeyboardInterrupt
        return warmup

    if phase == "preflight":
        monkeypatch.setattr(module.bench, "prepare_fixed_prompt", preflight)
    else:
        monkeypatch.setattr(module.bench, "run_request", request)
    code = module.main(
        [
            "--base-url",
            suite_server.base_url,
            "--result-dir",
            str(suite_server.results),
            "--cases",
            "128x128",
            "129x32",
            "recovery",
            "--warmups",
            "2",
            "--repeats",
            "2",
        ]
    )
    assert code == 130
    (summary_path,) = suite_server.results.glob("vq2a8-performance-*/summary.json")
    report = json.loads(summary_path.read_text(encoding="utf-8"))
    first, second, last = report["cases"]
    assert last["status"] == "not_run"
    assert all(record["requests"] == [] and "summary" not in record for record in report["cases"])
    if phase == "preflight":
        assert calls == ["preflight", "preflight"]
        assert [path for path, _ in suite_server.requests] == ["/tokenize"]
        assert first["status"] == "not_run" and second["status"] == "interrupted"
        assert all(record["warmup_requests"] == [] for record in report["cases"])
    else:
        assert calls == [("WARMUP", 1), ("WARMUP", 2)]
        assert [path for path, _ in suite_server.requests] == ["/tokenize"] * 3
        assert first["status"] == "interrupted" and second["status"] == "not_run"
        assert first["warmup_requests"] == [warmup]
    assert_saved_cases(NS(report=report, directory=summary_path.parent))
