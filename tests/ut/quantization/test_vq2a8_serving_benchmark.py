# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import io
import json
import subprocess
import sys
from http.client import IncompleteRead
from urllib.error import HTTPError

import pytest

from tools import benchmark_vq2a8_serving as bench


def test_script_help_runs_without_importing_npu_or_shadowing_standard_library():
    result = subprocess.run([sys.executable, bench.__file__, "--help"], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert "--base-url" in result.stdout and "--warmups" in result.stdout


def event(text="", *, usage=None, empty_choices=False):
    return {"choices": [] if empty_choices else [{"index": 0, "text": text}], "usage": usage}


def stream(*events, done=True):
    data = "".join(f"data: {json.dumps(value)}\r\n\r\n" for value in events)
    if done:
        data += "data: [DONE]\r\n\r\n"
    return io.BytesIO(data.encode())


def test_stream_uses_real_token_count_and_excludes_usage_arrival_from_timing():
    values = stream(
        event(),
        event("several tokens in one chunk"),
        event(" more tokens"),
        event(),
        event(usage={"completion_tokens": 9}, empty_choices=True),
    )
    result = bench.measure_stream(values, 10, clock=iter([10.2, 10.6]).__next__)
    assert result["ttft_ms"] == pytest.approx(200)
    assert result["tpot_ms"] == pytest.approx(50)
    assert result["completion_tokens"] == 9


def test_first_content_notice_is_immediate_once_and_preserves_timing():
    notices = []

    def lines():
        yield from stream(event(), event(empty_choices=True), done=False)
        assert not notices
        yield from stream(event("first"), done=False)
        assert notices == [pytest.approx(200)]
        yield from stream(event("last"), event(usage={"completion_tokens": 9}, empty_choices=True))

    result = bench.measure_stream(lines(), 10, clock=iter([10.2, 10.6]).__next__, on_first_content=notices.append)
    assert notices == [pytest.approx(200)]
    assert result == {"ttft_ms": pytest.approx(200), "tpot_ms": pytest.approx(50), "completion_tokens": 9}


def test_single_token_has_null_tpot_and_whitespace_is_generated_content():
    result = bench.measure_stream(stream(event(" ", usage={"completion_tokens": 1})), 1, clock=lambda: 1.5)
    assert result == {"ttft_ms": 500, "tpot_ms": None, "completion_tokens": 1}


def test_stream_reports_server_prompt_tokens_without_changing_timing():
    result = bench.measure_stream(
        stream(event("hi", usage={"completion_tokens": 4, "prompt_tokens": 2})), 0, clock=lambda: 0.2
    )
    assert result == {"ttft_ms": 200, "tpot_ms": 0, "completion_tokens": 4, "prompt_tokens": 2}


@pytest.mark.parametrize("value", [True, -1, "2"])
def test_stream_rejects_invalid_prompt_token_counts(value):
    with pytest.raises(ValueError, match="prompt_tokens"):
        bench.measure_stream(stream(event("hi", usage={"completion_tokens": 4, "prompt_tokens": value})), 0)


def test_sse_comments_multiline_json_and_utf8():
    payload = (
        ': keepalive\n\nevent: message\ndata: {"choices": [{"text": "天空"}],\n'
        'data: "usage": {"completion_tokens": 3}}\n\ndata: [DONE]\n\n'
    )
    result = bench.measure_stream(io.BytesIO(payload.encode()), 0, clock=lambda: 1)
    assert result == {"ttft_ms": 1000, "tpot_ms": 0, "completion_tokens": 3}


@pytest.mark.parametrize(
    "events,done,match",
    [
        ([event("text")], True, "completion_tokens"),
        ([event(usage={"completion_tokens": 1}, empty_choices=True)], True, "no generated text"),
        ([event("text", usage={"completion_tokens": 1})], False, "missing \\[DONE\\]"),
        ([{"error": {"message": "engine stopped"}}], True, "engine stopped"),
        ([event("text", usage={"completion_tokens": True})], True, "integer"),
        ([event("text", usage={"completion_tokens": -1})], True, "Negative"),
        ([event("text", usage={"completion_tokens": 0})], True, "positive"),
        ([event(None, usage={"completion_tokens": 1})], True, "string"),
        ([{"choices": [{"text": "one"}, {"text": "two"}]}], True, "single"),
    ],
)
def test_invalid_or_incomplete_stream_never_fabricates_metrics(events, done, match):
    with pytest.raises(ValueError, match=match):
        bench.measure_stream(stream(*events, done=done), 0, clock=lambda: 1)


def test_truncated_event_and_invalid_json_fail():
    for raw in (b'data: {"choices": []}', b"data: {broken}\n\n"):
        with pytest.raises(ValueError):
            bench.measure_stream(io.BytesIO(raw), 0)


@pytest.mark.parametrize("base_url", ["http://127.0.0.1:8000", "http://127.0.0.1:8000/v1/"])
def test_http_request_uses_openai_streaming_usage_and_custom_prompt(base_url):
    args = bench.parse_args(["--base-url", base_url, "--prompt", "hello", "--max-tokens", "8"])

    def opener(request, *, timeout):
        assert request.full_url == "http://127.0.0.1:8000/v1/completions"
        assert request.method == "POST" and timeout == 120
        payload = json.loads(request.data)
        assert payload == dict(
            model="vq2a8",
            prompt="hello",
            max_tokens=8,
            temperature=0,
            ignore_eos=True,
            n=1,
            stream=True,
            stream_options={"include_usage": True},
        )
        response = stream(event("one two", usage={"completion_tokens": 2}))
        response.headers = {"Content-Type": "text/event-stream; charset=utf-8"}
        return response

    notices = []
    result = bench.request_completion(
        args, opener=opener, clock=iter([10, 10.2]).__next__, on_first_content=notices.append
    )
    assert notices == [pytest.approx(200)]
    assert result["ttft_ms"] == pytest.approx(200)
    assert result["completion_tokens"] == 2


@pytest.mark.parametrize("base_url", ["http://127.0.0.1:8000", "http://127.0.0.1:8000/v1/"])
def test_fixed_prompt_uses_server_tokenizer_and_sends_exact_token_ids(base_url):
    args = bench.parse_args(
        ["--base-url", base_url, "--prompt", "seed", "--input-tokens", "16384", "--max-tokens", "1024"]
    )
    calls = []

    def opener(request, *, timeout):
        calls.append(request.full_url)
        payload = json.loads(request.data)
        assert request.method == "POST" and timeout == 120
        if request.full_url.endswith("/tokenize"):
            assert request.full_url == "http://127.0.0.1:8000/tokenize"
            assert payload == {"model": "vq2a8", "prompt": "seed", "add_special_tokens": False}
            return io.BytesIO(json.dumps({"tokens": [41, 42, 43], "count": 3, "max_model_len": 17408}).encode())
        assert payload["prompt"] == ([41, 42, 43] * 5462)[:16384]
        assert payload["add_special_tokens"] is False
        assert payload["skip_special_tokens"] is False
        assert payload["max_tokens"] == 1024 and payload["ignore_eos"] is True
        response = stream(
            event("first"),
            event("last"),
            event(usage={"completion_tokens": 1024, "prompt_tokens": 16384}, empty_choices=True),
        )
        response.headers = {"Content-Type": "text/event-stream"}
        return response

    args.prompt = bench.prepare_fixed_prompt(args, opener=opener)
    result = bench.request_completion(args, opener=opener, clock=iter([10, 10.2, 11.223]).__next__)
    assert calls == ["http://127.0.0.1:8000/tokenize", "http://127.0.0.1:8000/v1/completions"]
    assert result["ttft_ms"] == pytest.approx(200)
    assert result["tpot_ms"] == pytest.approx(1)
    assert result["prompt_tokens"] == 16384 and result["completion_tokens"] == 1024


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"tokens": []}, "nonempty"),
        ({"tokens": [True]}, "integer token IDs"),
        ({"tokens": [-1]}, "integer token IDs"),
        ({"tokens": ["1"]}, "integer token IDs"),
        ({"count": 2}, "count"),
        ({"count": True}, "count"),
        ({"max_model_len": None}, "max_model_len"),
        ({"max_model_len": True}, "max_model_len"),
        ({"max_model_len": 128}, "require at least 17408"),
        ({"error": {"message": "unavailable"}}, "unavailable"),
    ],
)
def test_fixed_prompt_rejects_invalid_tokenizer_response_and_short_server_context(changes, match):
    args = bench.parse_args(["--input-tokens", "16384", "--max-tokens", "1024"])
    payload = {"tokens": [1], "count": 1, "max_model_len": 17408, **changes}

    def opener(request, *, timeout):
        return io.BytesIO(json.dumps(payload).encode())

    with pytest.raises(ValueError, match=match):
        bench.prepare_fixed_prompt(args, opener=opener)


def test_fixed_prompt_reports_tokenization_http_errors():
    def failed(*args, **kwargs):
        raise HTTPError("http://localhost/tokenize", 404, "Not Found", {}, io.BytesIO(b"tokenizer unavailable"))

    with pytest.raises(ValueError, match="HTTP 404 from /tokenize: tokenizer unavailable"):
        bench.prepare_fixed_prompt(bench.parse_args(["--input-tokens", "16"]), opener=failed)


@pytest.mark.parametrize(
    "usage",
    [
        {"prompt_tokens": 15, "completion_tokens": 4},
        {"completion_tokens": 4},
        {"prompt_tokens": 16, "completion_tokens": 3},
    ],
)
def test_fixed_length_measurements_require_exact_server_usage(usage):
    args = bench.parse_args(["--input-tokens", "16", "--max-tokens", "4"])
    args.prompt = [41] * 16

    def opener(request, *, timeout):
        response = stream(event("text", usage=usage))
        response.headers = {"Content-Type": "text/event-stream"}
        return response

    with pytest.raises(ValueError, match="Expected 16 input and 4 output tokens"):
        bench.request_completion(args, opener=opener, clock=iter([0, 1]).__next__)


def test_cli_prepares_fixed_prompt_once_before_warmup_and_measurements(monkeypatch):
    calls = []

    def prepare(args):
        assert args.prompt == "seed"
        calls.append("tokenize")
        return [41] * 16

    def request(args):
        assert args.prompt == [41] * 16
        calls.append("completion")
        return {"ttft_ms": 100, "tpot_ms": 10, "prompt_tokens": 16, "completion_tokens": 4}

    monkeypatch.setattr(bench, "prepare_fixed_prompt", prepare)
    monkeypatch.setattr(bench, "request_completion", request)
    assert bench.main(["--prompt", "seed", "--input-tokens", "16", "--max-tokens", "4", "--repeats", "2"]) == 0
    assert calls == ["tokenize", "completion", "completion", "completion"]


def test_cli_rejects_insufficient_server_context_before_any_completion(monkeypatch, capsys):
    prepare = bench.prepare_fixed_prompt

    def opener(request, *, timeout):
        return io.BytesIO(b'{"tokens": [41], "count": 1, "max_model_len": 128}')

    monkeypatch.setattr(bench, "prepare_fixed_prompt", lambda args: prepare(args, opener=opener))
    monkeypatch.setattr(bench, "request_completion", lambda args: pytest.fail("Must not send a completion"))
    assert bench.main(["--input-tokens", "16384", "--max-tokens", "1024"]) == 1
    output = capsys.readouterr()
    assert not output.out and "Server max_model_len=128" in output.err


def test_http_error_and_non_stream_response_are_not_timing_successes():
    def failed(*args, **kwargs):
        raise HTTPError("http://localhost", 503, "Unavailable", {}, io.BytesIO(b"engine unavailable"))

    with pytest.raises(ValueError, match="HTTP 503: engine unavailable"):
        bench.request_completion(bench.parse_args([]), opener=failed)

    def wrong_content(*args, **kwargs):
        response = io.BytesIO(b"{}")
        response.headers = {"Content-Type": "application/json"}
        return response

    with pytest.raises(ValueError, match="text/event-stream"):
        bench.request_completion(bench.parse_args([]), opener=wrong_content)


def test_cli_warmup_is_not_printed_as_a_measurement(monkeypatch, capsys):
    calls = []

    def request(args):
        calls.append(args)
        return {"ttft_ms": 123, "tpot_ms": None, "completion_tokens": 1}

    monkeypatch.setattr(bench, "request_completion", request)
    assert bench.main(["--warmups", "1", "--repeats", "2"]) == 0
    assert len(calls) == 3
    assert capsys.readouterr().out.splitlines() == [
        "REQUEST=1 TTFT_MS=123.000 TPOT_MS=null TOKENS=1",
        "REQUEST=2 TTFT_MS=123.000 TPOT_MS=null TOKENS=1",
    ]


def test_cli_summary_excludes_warmups_and_uses_reported_prompt_tokens(monkeypatch, capsys):
    values = iter(
        [
            {"ttft_ms": 9000, "tpot_ms": 900, "completion_tokens": 4, "prompt_tokens": 20},
            {"ttft_ms": 100, "tpot_ms": 30, "completion_tokens": 4, "prompt_tokens": 2},
            {"ttft_ms": 200, "tpot_ms": 50, "completion_tokens": 4, "prompt_tokens": 2},
        ]
    )
    monkeypatch.setattr(bench, "request_completion", lambda _: next(values))
    assert bench.main(["--warmups", "1", "--repeats", "2", "--summary"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 3 and "PROMPT_TOKENS=2" in lines[0]
    summary = json.loads(lines[-1].split("=", 1)[1])
    assert summary == {
        "requests": 2,
        "prompt_tokens": [2],
        "mean_ttft_ms": 150,
        "median_ttft_ms": 150,
        "mean_tpot_ms": 40,
        "median_tpot_ms": 40,
    }


def test_cli_progress_distinguishes_phases_without_changing_measurement_output(monkeypatch, capsys):
    values = iter(
        [
            {"ttft_ms": 9000, "tpot_ms": 900, "completion_tokens": 4, "prompt_tokens": 16},
            {"ttft_ms": 100, "tpot_ms": 30, "completion_tokens": 4, "prompt_tokens": 16},
        ]
    )

    def request(args, *, on_first_content):
        result = next(values)
        on_first_content(result["ttft_ms"])
        return result

    monkeypatch.setattr(bench, "prepare_fixed_prompt", lambda args: [41] * args.input_tokens)
    monkeypatch.setattr(bench, "request_completion", request)
    assert bench.main(["--input-tokens", "16", "--max-tokens", "4", "--progress", "--summary"]) == 0
    output = capsys.readouterr()
    assert output.err.splitlines() == [
        "WARMUP=1/1 START INPUT_TOKENS=16 MAX_TOKENS=4",
        "WARMUP=1/1 FIRST_CONTENT TTFT_MS=9000.000",
        "WARMUP=1/1 DONE TOKENS=4",
        "REQUEST=1/1 START INPUT_TOKENS=16 MAX_TOKENS=4",
        "REQUEST=1/1 FIRST_CONTENT TTFT_MS=100.000",
        "REQUEST=1/1 DONE TOKENS=4",
    ]
    lines = output.out.splitlines()
    assert len(lines) == 2
    assert lines[0] == "REQUEST=1 TTFT_MS=100.000 TPOT_MS=30.000 TOKENS=4 PROMPT_TOKENS=16"
    summary = json.loads(lines[1].split("=", 1)[1])
    assert summary["requests"] == 1 and summary["mean_ttft_ms"] == 100 and summary["mean_tpot_ms"] == 30


def test_cli_progress_identifies_failed_phase_without_first_content_or_success(monkeypatch, capsys):
    def request(args, *, on_first_content):
        raise TimeoutError("timed out waiting for content")

    monkeypatch.setattr(bench, "request_completion", request)
    assert bench.main(["--progress", "--warmups", "0"]) == 1
    output = capsys.readouterr()
    assert not output.out
    assert output.err.splitlines() == [
        "REQUEST=1/1 START MAX_TOKENS=32",
        "ERROR: timed out waiting for content",
    ]


@pytest.mark.parametrize("error", [ValueError("missing usage"), TimeoutError("timeout"), IncompleteRead(b"partial", 2)])
def test_cli_failure_prints_error_without_metrics(monkeypatch, capsys, error):
    def request(args):
        raise error

    monkeypatch.setattr(bench, "request_completion", request)
    assert bench.main(["--warmups", "0"]) == 1
    output = capsys.readouterr()
    assert not output.out and str(error) in output.err


@pytest.mark.parametrize(
    "flags",
    [
        ["--warmups", "-1"],
        ["--repeats", "0"],
        ["--timeout", "nan"],
        ["--max-tokens", "0"],
        ["--input-tokens", "0"],
        ["--input-tokens", "-1"],
    ],
)
def test_invalid_cli_arguments_fail(flags):
    with pytest.raises(SystemExit):
        bench.parse_args(flags)
