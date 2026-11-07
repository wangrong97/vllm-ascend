# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the HTTP preflight and benchmark handoff without NPU/vLLM."""

import json
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from tools.tests import run_vq2a8_dspark_performance as performance


@pytest.fixture
def server():
    state = SimpleNamespace(
        calls=[],
        payloads=[],
        responses={
            "/health": (200, {}),
            "/v1/models": (200, {"data": [{"id": "vq2a8"}]}),
            "/v1/completions": (200, {"choices": [{"text": "Hello from the target"}]}),
        },
    )

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state.calls.append((self.command, self.path))
            status, body = state.responses[self.path]
            payload = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):
            state.payloads.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.do_GET()

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    state.url = f"http://127.0.0.1:{httpd.server_port}"
    thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield state
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def arguments(tmp_path, server):
    return [
        "--base-url",
        server.url,
        "--tokenizer",
        "/models/target with spaces",
        "--result-dir",
        str(tmp_path / "results"),
    ]


def read_report(tmp_path):
    return json.loads(next((tmp_path / "results").glob("*/run.json")).read_text(encoding="utf-8"))


def test_runs_http_checks_then_both_sweeps_with_short_counts(server, tmp_path, monkeypatch):
    trials = []

    def bench(command, *, check):
        assert check is True
        assert server.calls == [("GET", "/health"), ("GET", "/v1/models"), ("POST", "/v1/completions")]
        assert command[:3] == [performance.sys.executable, "-u", str(performance.BENCHMARK_SCRIPT)]
        trials.append(performance.throughput.parse_args(command[3:]))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(performance.subprocess, "run", bench)
    assert performance.main(arguments(tmp_path, server)) == 0
    assert [(trial.input_len, trial.output_len) for trial in trials] == [(16, 16), (128, 512)]
    for trial in trials:
        assert (trial.tp, trial.dp, trial.ep_size, trial.max_num_seqs) == (1, 4, 4, 4)
        assert (trial.concurrency, trial.num_prompts, trial.repeats) == ([1, 2, 4], 4, 1)
        assert trial.tokenizer == "/models/target with spaces"
        assert "declared_variant=dspark" in trial.note
    assert server.payloads == [{"model": "vq2a8", "prompt": "Hello", "max_tokens": 16, "temperature": 0}]
    report = read_report(tmp_path)
    assert report["status"] == "passed"
    assert [record["status"] for record in report["stages"]] == ["passed", "passed"]
    assert json.loads(next((tmp_path / "results").glob("*/smoke.json")).read_text())["choices"][0]["text"]


@pytest.mark.parametrize(
    "endpoint, response",
    [
        ("/health", (503, {"error": "not ready"})),
        ("/v1/models", (200, {"data": [{"id": "wrong-model"}]})),
        ("/v1/completions", (200, {"choices": [{"text": "  "}]})),
        ("/v1/completions", (200, {"error": "generation failed"})),
        ("/v1/completions", (200, {"choices": {"text": "wrong schema"}})),
    ],
)
def test_preflight_failure_prevents_measurement_and_records_failure(server, tmp_path, monkeypatch, endpoint, response):
    server.responses[endpoint] = response
    monkeypatch.setattr(performance.subprocess, "run", lambda *args, **kwargs: pytest.fail("Benchmark must not start"))
    assert performance.main(arguments(tmp_path, server)) == 1
    report = read_report(tmp_path)
    assert report["status"] == "failed"
    assert [record["status"] for record in report["stages"]] == ["not_run", "not_run"]
    assert server.calls[-1][1] == endpoint


def test_failed_smoke_sweep_stops_main_workload(server, tmp_path, monkeypatch):
    calls = []

    def fail(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(performance.subprocess, "run", fail)
    assert performance.main(arguments(tmp_path, server)) == 1
    assert len(calls) == 1
    report = read_report(tmp_path)
    assert report["status"] == "failed"
    assert [record["status"] for record in report["stages"]] == ["failed", "not_run"]


def test_dry_run_and_baseline_overrides_do_not_contact_server_or_create_files(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(performance, "request", lambda *args: pytest.fail("Dry run must not contact server"))
    monkeypatch.setattr(performance.subprocess, "run", lambda *args, **kwargs: pytest.fail("Dry run must not launch"))
    argv = [
        "--tokenizer",
        "/models/target",
        "--result-dir",
        str(tmp_path / "results"),
        "--variant",
        "baseline",
        "--parallel-mode",
        "tp-ep",
        "--ep-size",
        "2",
        "--input-len",
        "256",
        "--output-len",
        "1024",
        "--concurrency",
        "1",
        "2",
        "--num-prompts",
        "6",
        "--repeats",
        "2",
        "--note",
        "max_num_batched_tokens=8192",
        "--dry-run",
    ]
    assert performance.main(argv) == 0
    text = capsys.readouterr().out
    assert "Declared TP=2 DP=1 EP=2" in text
    assert "--random-input-len 256 --random-output-len 1024" in text
    assert "--num-prompts 6" in text
    assert list(tmp_path.iterdir()) == []
    args = performance.parse_args(argv)
    for _, stage in performance.benchmark_stages(args, tmp_path):
        trial = performance.throughput.parse_args(stage)
        assert trial.note == "declared_variant=baseline; max_num_batched_tokens=8192"
        assert (trial.num_prompts, trial.repeats) == (6, 2)


@pytest.mark.parametrize(
    "extra",
    [
        ["--base-url", "http://localhost:8003/v1"],
        ["--num-prompts", "2"],
        ["--concurrency", "4", "4"],
        ["--request-timeout", "nan"],
        ["--repeats", "0"],
    ],
)
def test_invalid_run_is_rejected_before_any_io(tmp_path, monkeypatch, extra):
    monkeypatch.setattr(performance, "request", lambda *args: pytest.fail("Invalid request must not contact server"))
    with pytest.raises(SystemExit):
        performance.main(["--tokenizer", "/models/target", "--result-dir", str(tmp_path / "results"), *extra])
    assert list(tmp_path.iterdir()) == []
