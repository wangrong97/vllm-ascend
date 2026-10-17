# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOKEN_IDS = (101, 102, 103)
FULL_CONTEXT_TOKENS = 1049600


@pytest.fixture
def long_context_runner(tmp_path):
    repo_root = tmp_path / "test repository"
    (repo_root / "tools/tests").mkdir(parents=True)
    for relative_path in (
        "tools/tests/run_vq2a8_ep2_64k_1k.sh",
        "tools/tests/run_vq2a8_ep2_256k_1k.sh",
        "tools/tests/run_vq2a8_ep2_1m_1k.sh",
        "tools/tests/run_vq2a8_ep2_long_context_latency.py",
        "tools/benchmark_vq2a8_serving.py",
        "tools/vq2a8_live_log.py",
    ):
        shutil.copyfile(REPO_ROOT / relative_path, repo_root / relative_path)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    if os.name != "nt":
        (bin_dir / "python3").symlink_to(sys.executable)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "no_proxy": "127.0.0.1",
        "NO_PROXY": "127.0.0.1",
    }

    def run(*arguments, max_model_len=FULL_CONTEXT_TOKENS, failure=None, shell_script=None):
        if shell_script is not None and (os.name == "nt" or shutil.which("bash") is None):
            pytest.skip("Shell wrapper execution requires a POSIX environment with bash.")
        requests = []
        state = {"active": 0, "peak": 0}
        lock = threading.Lock()

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with lock:
                    requests.append((self.path, payload))
                if self.path == "/tokenize":
                    data = json.dumps(
                        {"tokens": TOKEN_IDS, "count": len(TOKEN_IDS), "max_model_len": max_model_len}
                    ).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                if self.path != "/v1/completions":
                    self.send_error(404)
                    return
                if failure == "http":
                    self.send_error(503, "Engine unavailable")
                    return
                with lock:
                    state["active"] += 1
                    state["peak"] = max(state["peak"], state["active"])
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    # Separate first and last content in time, but send only two
                    # chunks for 1024 tokens: TPOT must use reported token usage.
                    time.sleep(0.02)
                    first = {"choices": [{"index": 0, "text": "北京"}]}
                    self.wfile.write(f"data: {json.dumps(first)}\n\n".encode())
                    self.wfile.flush()
                    time.sleep(0.02)
                    events = [
                        {"choices": [{"index": 0, "text": "。", "finish_reason": "length"}]},
                        {
                            "choices": [],
                            "usage": {
                                "prompt_tokens": len(payload["prompt"]),
                                "completion_tokens": 512 if failure == "short" else 1024,
                            },
                        },
                    ]
                    for event in events:
                        self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                finally:
                    with lock:
                        state["active"] -= 1
                self.wfile.write(b"data: [DONE]\n\n")

            def log_message(self, format, *args):
                pass

        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                if shell_script is None:
                    command = [sys.executable, str(repo_root / "tools/tests/run_vq2a8_ep2_long_context_latency.py")]
                else:
                    command = ["bash", str(repo_root / "tools/tests" / shell_script)]
                result = subprocess.run(
                    [
                        *command,
                        "--base-url",
                        f"http://127.0.0.1:{server.server_port}/v1",
                        *arguments,
                    ],
                    cwd=tmp_path,
                    env=env,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=30,
                )
            finally:
                server.shutdown()
                thread.join(timeout=5)
        summaries = list((repo_root / "benchmark-results").glob("vq2a8-ep2-long-*/summary.json"))
        assert len(summaries) == 1, result.stdout + result.stderr
        directory = summaries[0].parent
        summary = json.loads(summaries[0].read_text())
        cases = {record["case"]: record for record in summary["cases"]}
        logs = {case: (directory / f"{case}.log").read_text() for case in cases}
        for case, record in cases.items():
            assert json.loads((directory / f"{case}.json").read_text()) == record
        return result, requests, summary, cases, logs, state["peak"]

    return run


def assert_completion_payload(payload, input_tokens):
    assert payload == {
        "model": "vq2a8",
        "prompt": (list(TOKEN_IDS) * ((input_tokens + len(TOKEN_IDS) - 1) // len(TOKEN_IDS)))[:input_tokens],
        "max_tokens": 1024,
        "temperature": 0,
        "ignore_eos": True,
        "n": 1,
        "stream": True,
        "stream_options": {"include_usage": True},
        "add_special_tokens": False,
        "skip_special_tokens": False,
    }


def test_defaults_preflight_both_cases_then_measure_exact_lengths_and_save_results(long_context_runner):
    result, requests, summary, cases, logs, peak = long_context_runner()

    assert result.returncode == 0, result.stdout + result.stderr
    assert [path for path, _ in requests] == ["/tokenize", "/tokenize", "/v1/completions", "/v1/completions"]
    assert [payload for _, payload in requests[:2]] == [
        {"model": "vq2a8", "prompt": "中国首都是", "add_special_tokens": False}
    ] * 2
    assert_completion_payload(requests[2][1], 262144)
    assert_completion_payload(requests[3][1], 1048576)
    assert peak == 1
    assert summary["check_only"] is False
    assert "WARMUP=" not in result.stdout
    for name, input_tokens in (("256k", 262144), ("1m", 1048576)):
        record = cases[name]
        assert record["status"] == "passed"
        assert record["input_tokens"] == input_tokens
        assert record["output_tokens"] == 1024
        assert record["required_context_tokens"] == input_tokens + 1024
        assert record["concurrency"] == 1
        assert record["warmups"] == 0
        assert record["repeats"] == 1
        assert record["timeout_seconds"] == 21600
        assert len(record["requests"]) == 1
        measured = record["requests"][0]
        assert measured["prompt_tokens"] == input_tokens
        assert measured["completion_tokens"] == 1024
        assert 0 < measured["tpot_ms"] < measured["ttft_ms"]
        assert record["summary"] == {
            f"{aggregation}_{metric}": measured[metric]
            for aggregation in ("mean", "median")
            for metric in ("ttft_ms", "tpot_ms")
        }
        for message in (
            f"REQUEST=1/1 START INPUT_TOKENS={input_tokens} MAX_TOKENS=1024",
            "REQUEST=1/1 FIRST_CONTENT TTFT_MS=",
            "REQUEST=1/1 DONE TOKENS=1024",
            "TPOT_MS=",
            "LONG_CONTEXT_SUMMARY=",
        ):
            assert message in result.stdout
            assert message in logs[name]


def test_one_million_context_rejects_full_1m_without_shortening_and_still_measures_256k(long_context_runner):
    result, requests, _, cases, logs, _ = long_context_runner(max_model_len=1048576)

    assert result.returncode == 1
    assert [path for path, _ in requests] == ["/tokenize", "/tokenize", "/v1/completions"]
    assert_completion_payload(requests[-1][1], 262144)
    assert cases["256k"]["status"] == "passed"
    assert cases["1m"]["status"] == "preflight_failed"
    assert cases["1m"]["input_tokens"] == 1048576
    assert cases["1m"]["requests"] == []
    assert "summary" not in cases["1m"]
    assert "Server max_model_len=1048576" in logs["1m"]
    assert "require at least 1049600" in logs["1m"]
    assert "LONG_CONTEXT_SUMMARY=" not in logs["1m"]


def test_explicit_1m_window_reserves_output_inside_context(long_context_runner):
    result, requests, _, cases, _, _ = long_context_runner("--cases", "1m-window", max_model_len=1048576)

    assert result.returncode == 0, result.stdout + result.stderr
    assert [path for path, _ in requests] == ["/tokenize", "/v1/completions"]
    assert_completion_payload(requests[-1][1], 1047552)
    assert list(cases) == ["1m-window"]
    assert cases["1m-window"]["required_context_tokens"] == 1048576
    assert cases["1m-window"]["status"] == "passed"


@pytest.mark.parametrize(
    "shell_script, case, input_tokens",
    [
        ("run_vq2a8_ep2_64k_1k.sh", "64k", 65536),
        ("run_vq2a8_ep2_256k_1k.sh", "256k", 262144),
        ("run_vq2a8_ep2_1m_1k.sh", "1m", 1048576),
    ],
)
def test_shell_entrypoints_each_run_only_the_named_exact_case(long_context_runner, shell_script, case, input_tokens):
    result, requests, _, cases, _, peak = long_context_runner(
        shell_script=shell_script, max_model_len=input_tokens + 1024
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert [path for path, _ in requests] == ["/tokenize", "/v1/completions"]
    assert_completion_payload(requests[-1][1], input_tokens)
    assert list(cases) == [case]
    assert cases[case]["status"] == "passed"
    assert peak == 1


def test_64k_case_rejects_context_one_token_below_required_without_generation(long_context_runner):
    result, requests, _, cases, logs, peak = long_context_runner(
        shell_script="run_vq2a8_ep2_64k_1k.sh", max_model_len=66559
    )

    assert result.returncode == 1
    assert [path for path, _ in requests] == ["/tokenize"]
    assert list(cases) == ["64k"]
    assert cases["64k"]["status"] == "preflight_failed"
    assert cases["64k"]["input_tokens"] == 65536
    assert cases["64k"]["output_tokens"] == 1024
    assert cases["64k"]["required_context_tokens"] == 66560
    assert cases["64k"]["requests"] == []
    assert "summary" not in cases["64k"]
    assert "Server max_model_len=66559" in logs["64k"]
    assert "require at least 66560" in logs["64k"]
    assert "LONG_CONTEXT_SUMMARY=" not in result.stdout
    assert peak == 0


@pytest.mark.parametrize("failure, error", [("short", "completion_tokens=512"), ("http", "HTTP 503")])
def test_generation_failure_has_no_success_summary_and_stops_all_later_requests(long_context_runner, failure, error):
    result, requests, _, cases, logs, _ = long_context_runner("--repeats", "2", failure=failure)

    assert result.returncode == 1
    assert [path for path, _ in requests] == ["/tokenize", "/tokenize", "/v1/completions"]
    assert cases["256k"]["status"] == "failed"
    assert cases["256k"]["requests"] == []
    assert cases["1m"]["status"] == "not_run"
    assert cases["1m"]["requests"] == []
    assert error in cases["256k"]["error"]
    assert error in logs["256k"]
    assert all("summary" not in record for record in cases.values())
    assert "LONG_CONTEXT_SUMMARY=" not in result.stdout


def test_check_only_preflights_every_case_without_any_generation(long_context_runner):
    result, requests, summary, cases, _, peak = long_context_runner("--check-only")

    assert result.returncode == 0, result.stdout + result.stderr
    assert [path for path, _ in requests] == ["/tokenize", "/tokenize"]
    assert peak == 0
    assert summary["check_only"] is True
    assert all(record["status"] == "ready" and record["requests"] == [] for record in cases.values())
    assert all("summary" not in record for record in cases.values())
    assert "LONG_CONTEXT_SUMMARY=" not in result.stdout
