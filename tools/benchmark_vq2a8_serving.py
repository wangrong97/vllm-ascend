#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure client-visible TTFT/TPOT from an OpenAI completions SSE stream.

TPOT uses usage.completion_tokens, never the number of SSE chunks. Transport
buffering can combine tokens, so these are HTTP delivery timings.
Restored from v023 to reproduce its short-prompt measurements. Input token
counts and optional aggregate summaries do not alter the timing boundaries.
Fixed-length synthetic inputs use the server tokenizer before timed requests.
Optional progress notices expose request phases; they do not diagnose worker
stalls. Disable progress for measurements without diagnostic output overhead.
"""

# ruff: noqa: E402
import os
import sys

if not __package__:
    # tools/bisect would otherwise shadow Python's standard-library bisect.
    sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import math
import statistics
import time
from http.client import HTTPException
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def sse_data(lines):
    """Decode data events, including multiline JSON and keepalive comments."""
    data = []
    for raw in lines:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                yield "\n".join(data)
                data = []
        elif line.startswith("data:"):
            value = line[5:]
            data.append(value[1:] if value.startswith(" ") else value)
    if data:
        raise ValueError("Truncated SSE event")


def measure_stream(lines, started, *, clock=time.perf_counter, on_first_content=None):
    first_content = last_content = tokens = None
    prompt_tokens = None
    completed = False
    for data in sse_data(lines):
        if data.strip() == "[DONE]":
            completed = True
            break
        event = json.loads(data)
        if not isinstance(event, dict):
            raise ValueError("Invalid SSE completion payload")
        if event.get("error") is not None:
            raise ValueError(f"Server stream error: {event['error']}")
        choices = event.get("choices", [])
        if not isinstance(choices, list) or len(choices) > 1:
            raise ValueError("Expected a single streamed completion")
        if choices:
            choice = choices[0]
            if not isinstance(choice, dict) or choice.get("index", 0) != 0:
                raise ValueError("Unexpected streamed choice")
            content = choice.get("text", "")
            if not isinstance(content, str):
                raise ValueError("Completion text must be a string")
            if content:
                last_content = clock()
                if first_content is None:
                    first_content = last_content
                    if on_first_content is not None:
                        on_first_content((first_content - started) * 1000)
        usage = event.get("usage")
        if usage is not None:
            if not isinstance(usage, dict) or type(usage.get("completion_tokens")) is not int:
                raise ValueError("Missing integer usage.completion_tokens")
            tokens = usage["completion_tokens"]
            if tokens < 0:
                raise ValueError("Negative usage.completion_tokens")
            if "prompt_tokens" in usage:
                prompt_tokens = usage["prompt_tokens"]
                if type(prompt_tokens) is not int or prompt_tokens < 0:
                    raise ValueError("Invalid usage.prompt_tokens")
    if not completed:
        raise ValueError("Truncated stream: missing [DONE]")
    if first_content is None:
        raise ValueError("Stream contained no generated text")
    if tokens is None or tokens < 1:
        raise ValueError("Missing positive usage.completion_tokens; token count cannot be inferred from chunks")
    result = {
        "ttft_ms": (first_content - started) * 1000,
        "tpot_ms": (last_content - first_content) * 1000 / (tokens - 1) if tokens >= 2 else None,
        "completion_tokens": tokens,
    }
    if prompt_tokens is not None:
        result["prompt_tokens"] = prompt_tokens
    return result


def prepare_fixed_prompt(args, *, opener=urlopen):
    """Build exact-length synthetic token IDs using the serving tokenizer."""
    base = args.base_url.rstrip("/").removesuffix("/v1")
    request = Request(
        base + "/tokenize",
        data=json.dumps({"model": args.model, "prompt": args.prompt, "add_special_tokens": False}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with opener(request, timeout=args.timeout) as response:
            tokenized = json.load(response)
    except HTTPError as error:
        detail = error.read(4096).decode("utf-8", errors="replace")
        error.close()
        raise ValueError(f"HTTP {error.code} from /tokenize: {detail}") from error
    if not isinstance(tokenized, dict):
        raise ValueError("Invalid /tokenize response")
    if tokenized.get("error") is not None:
        raise ValueError(f"Server tokenization error: {tokenized['error']}")
    tokens = tokenized.get("tokens")
    if not isinstance(tokens, list) or not tokens or any(type(token) is not int or token < 0 for token in tokens):
        raise ValueError("/tokenize must return a nonempty list of nonnegative integer token IDs")
    if type(tokenized.get("count")) is not int or tokenized["count"] != len(tokens):
        raise ValueError("/tokenize count does not match its token IDs")
    context = tokenized.get("max_model_len")
    if type(context) is not int or context < 1:
        raise ValueError("/tokenize must report a positive integer max_model_len")
    required = args.input_tokens + args.max_tokens
    if context < required:
        raise ValueError(
            f"Server max_model_len={context}, but {args.input_tokens} input + {args.max_tokens} output tokens "
            f"require at least {required}. Restart the server with sufficient context before benchmarking."
        )
    repetitions = (args.input_tokens + len(tokens) - 1) // len(tokens)
    return (tokens * repetitions)[: args.input_tokens]


def request_completion(args, *, opener=urlopen, clock=time.perf_counter, on_first_content=None):
    base = args.base_url.rstrip("/")
    endpoint = base + ("/completions" if base.endswith("/v1") else "/v1/completions")
    payload = {
        "model": args.model,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "n": 1,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if args.input_tokens is not None:
        if not isinstance(args.prompt, list) or len(args.prompt) != args.input_tokens:
            raise ValueError("Fixed-length prompt must be prepared before requesting completion")
        payload["add_special_tokens"] = False
        # Forced long outputs can contain EOS/special tokens. Keep their text
        # visible so they do not disappear from the final-token timing.
        payload["skip_special_tokens"] = False
    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    started = clock()
    try:
        with opener(request, timeout=args.timeout) as response:
            if response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "text/event-stream":
                raise ValueError("Expected text/event-stream response from /v1/completions")
            result = measure_stream(response, started, clock=clock, on_first_content=on_first_content)
    except HTTPError as error:
        detail = error.read(4096).decode("utf-8", errors="replace")
        error.close()
        raise ValueError(f"HTTP {error.code}: {detail}") from error
    if args.input_tokens is not None:
        if result.get("prompt_tokens") != args.input_tokens or result["completion_tokens"] != args.max_tokens:
            raise ValueError(
                f"Expected {args.input_tokens} input and {args.max_tokens} output tokens; "
                f"server reported prompt_tokens={result.get('prompt_tokens')}, "
                f"completion_tokens={result['completion_tokens']}"
            )
    return result


def run_request(args, phase, index, total):
    if not args.progress:
        return request_completion(args)
    label = f"{phase}={index}/{total}"
    prompt = f" INPUT_TOKENS={args.input_tokens}" if args.input_tokens is not None else ""
    print(f"{label} START{prompt} MAX_TOKENS={args.max_tokens}", file=sys.stderr, flush=True)

    def first_content(ttft_ms):
        print(f"{label} FIRST_CONTENT TTFT_MS={ttft_ms:.3f}", file=sys.stderr, flush=True)

    result = request_completion(args, on_first_content=first_content)
    print(f"{label} DONE TOKENS={result['completion_tokens']}", file=sys.stderr, flush=True)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="vq2a8")
    parser.add_argument("--prompt", default="Explain why the sky is blue in simple terms.")
    parser.add_argument(
        "--input-tokens",
        type=int,
        help="Exact synthetic input length; repeat/truncate --prompt token IDs obtained once from server /tokenize",
    )
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP socket timeout in seconds")
    parser.add_argument("--summary", action="store_true", help="Print a final JSON summary, excluding warmups")
    parser.add_argument(
        "--progress", action="store_true", help="Print diagnostic request phases and first-content TTFT to stderr"
    )
    args = parser.parse_args(argv)
    url = urlsplit(args.base_url)
    if url.scheme not in ("http", "https") or not url.netloc:
        parser.error("--base-url must be an HTTP(S) server URL")
    if not args.prompt or not args.model:
        parser.error("--prompt and --model cannot be empty")
    if args.max_tokens < 1 or args.warmups < 0 or args.repeats < 1:
        parser.error("Require max-tokens >=1, warmups >=0 and repeats >=1")
    if args.input_tokens is not None and args.input_tokens < 1:
        parser.error("--input-tokens must be positive")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        if args.input_tokens is not None:
            args.prompt = prepare_fixed_prompt(args)
        for index in range(args.warmups):
            run_request(args, "WARMUP", index + 1, args.warmups)
        results = []
        for index in range(args.repeats):
            result = run_request(args, "REQUEST", index + 1, args.repeats)
            results.append(result)
            tpot = "null" if result["tpot_ms"] is None else f"{result['tpot_ms']:.3f}"
            prompt = f" PROMPT_TOKENS={result['prompt_tokens']}" if "prompt_tokens" in result else ""
            print(
                f"REQUEST={index + 1} TTFT_MS={result['ttft_ms']:.3f} "
                f"TPOT_MS={tpot} TOKENS={result['completion_tokens']}{prompt}",
                flush=True,
            )
        if args.summary:
            ttfts = [result["ttft_ms"] for result in results]
            tpots = [result["tpot_ms"] for result in results if result["tpot_ms"] is not None]
            print(
                "VQ2A8_SERVING_SUMMARY="
                + json.dumps(
                    {
                        "requests": len(results),
                        "prompt_tokens": sorted(
                            {result["prompt_tokens"] for result in results if "prompt_tokens" in result}
                        ),
                        "mean_ttft_ms": statistics.mean(ttfts),
                        "median_ttft_ms": statistics.median(ttfts),
                        "mean_tpot_ms": statistics.mean(tpots) if tpots else None,
                        "median_tpot_ms": statistics.median(tpots) if tpots else None,
                    }
                ),
                flush=True,
            )
    except (HTTPException, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
