# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in EP decode-graph replay across short and 16K contexts.

Each topology loads the model once, with 128-token chunked prefill and a B1
full decode graph. No eager model or single-card reference is loaded. The
checks establish graph execution, token counts and short-request recovery;
they do not establish eager parity or measure serving TTFT/TPOT.
"""

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.e2e.nightly.test_vq2a8_gs32_ep import (
    EP_SIZES,
    check_inventory,
    graph_state,
    variant_environment,
)
from tests.e2e.nightly.test_vq2a8_gs32_ep import runner_options as ep_runner_options
from tests.e2e.nightly.test_vq2a8_gs32_full_graph import CHILD_TIMEOUT_SECONDS, REPO_ROOT

MODULE_NAME = "tests.e2e.nightly.test_vq2a8_gs32_ep_long_context"
MAX_CONTEXT = 17408
PREFILL_CHUNK_TOKENS = 128
KV_CACHE_BYTES = 4 * 1024**3
# Short decode crosses the C128 compression boundary; the final short request
# checks that long-context metadata and cache reuse do not poison recovery.
REQUEST_SHAPES = ((127, 4), (16384, 1024), (127, 4))
LOGPROB_ATOL = 1e-3


def runner_options(library, binary, ep_size):
    options = ep_runner_options("full_graph", library, binary, ep_size)
    options.update(
        max_model_len=MAX_CONTEXT,
        max_num_batched_tokens=PREFILL_CHUNK_TOKENS,
        enable_chunked_prefill=True,
        kv_cache_memory_bytes=KV_CACHE_BYTES,
    )
    return options


def check_graph_progress(previous, current, layer_count, output_tokens):
    assert [state["rank"] for state in previous] == [state["rank"] for state in current]
    for before, after in zip(previous, current, strict=True):
        assert before["layers"] == after["layers"], "Resident expert inventory changed during generation."
        assert before["validity_owners"] == after["validity_owners"] == layer_count
        assert before["decode"]["captures"] == after["decode"]["captures"] == 1
        assert after["decode"]["replays"] - before["decode"]["replays"] >= output_tokens - 1, (
            f"Rank {after['rank']} did not replay its full decode graph for every decode token."
        )
        for phase in ("prefill", "other"):
            assert before[phase] == after[phase] == {"captures": 0, "replays": 0}, (
                f"Rank {after['rank']} unexpectedly captured or replayed {phase}."
            )


def check_runner_selection(states, expected_version):
    for state in states:
        assert state["model_runner_version"] == expected_version, (
            f"Rank {state['rank']} selected {state['model_runner']}, expected model runner V{expected_version}."
        )


def run_variant(model, library, binary, ep_size, output):
    # Engine and NPU initialization stay inside the isolated child process.
    from vllm import SamplingParams, envs

    from tests.e2e.conftest import VllmRunner
    from tests.e2e.nightly.vq2a8_gs32_helpers import checkpoint_inventory

    config, _, entries = checkpoint_inventory(model)
    layer_count = config["num_hidden_layers"]
    expected_version = 2 if envs.VLLM_USE_V2_MODEL_RUNNER else 1
    report = {
        "ep_size": ep_size,
        "execution": "full_graph",
        "max_model_len": MAX_CONTEXT,
        "prefill_chunk_tokens": PREFILL_CHUNK_TOKENS,
        "model_runner_version": expected_version,
        "requests": [],
        "scope": "long-context graph replay and recovery; not eager parity or serving performance",
    }
    with VllmRunner(str(model), **runner_options(library, binary, ep_size)) as runner:
        llm = runner.model
        previous = graph_state(llm, ep_size)
        check_runner_selection(previous, expected_version)
        check_inventory(previous, config, ep_size, entries)
        check_graph_progress(previous, previous, layer_count, 1)
        report["graph_before"] = previous
        print(
            "VQ2A8_GS32_EP_RUNNER="
            + json.dumps({"ranks": [{key: state[key] for key in ("rank", "model_runner")} for state in previous]}),
            flush=True,
        )
        seed_tokens = llm.get_tokenizer().encode("中国首都是", add_special_tokens=False)
        assert seed_tokens
        for input_tokens, output_tokens in REQUEST_SHAPES:
            prompt = (seed_tokens * ((input_tokens + len(seed_tokens) - 1) // len(seed_tokens)))[:input_tokens]
            sampling = SamplingParams(
                temperature=0,
                max_tokens=output_tokens,
                min_tokens=output_tokens,
                ignore_eos=True,
                seed=0,
                logprobs=1,
            )
            results = llm.generate([{"prompt_token_ids": prompt}], sampling, use_tqdm=False)
            assert len(results) == 1 and results[0].finished and len(results[0].outputs) == 1
            assert list(results[0].prompt_token_ids) == prompt
            completion = results[0].outputs[0]
            assert len(completion.token_ids) == output_tokens
            assert completion.logprobs is not None and len(completion.logprobs) == output_tokens
            assert all(0 <= token < config["vocab_size"] for token in completion.token_ids)
            logprobs = [
                values[token].logprob for token, values in zip(completion.token_ids, completion.logprobs, strict=True)
            ]
            assert all(math.isfinite(value) for value in logprobs)
            current = graph_state(llm, ep_size)
            check_runner_selection(current, expected_version)
            check_graph_progress(previous, current, layer_count, output_tokens)
            record = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "token_ids": list(completion.token_ids),
                "logprobs": logprobs,
                "graph_state": current,
            }
            report["requests"].append(record)
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
            print(
                "VQ2A8_GS32_EP_LONG_CONTEXT="
                + json.dumps({"ep_size": ep_size, "input_tokens": input_tokens, "output_tokens": output_tokens}),
                flush=True,
            )
            previous = current
        first, recovery = report["requests"][0], report["requests"][-1]
        assert recovery["token_ids"] == first["token_ids"], "Short-request output changed after long-context decode."
        assert all(
            abs(observed - expected) <= LOGPROB_ATOL
            for observed, expected in zip(recovery["logprobs"], first["logprobs"], strict=True)
        ), "Short-request logprobs changed after long-context decode."
    report["status"] = "PASS"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


@pytest.mark.parametrize("ep_size", EP_SIZES, ids=lambda size: f"ep{size}")
def test_gs32_ep_long_context_decode_graph_and_recovery(request, tmp_path, ep_size):
    values = {
        name: request.config.getoption(flag, default=None)
        for name, flag in (
            ("model", "--vq2a8-model"),
            ("library", "--vq2a8-gs32-library"),
            ("binary", "--vq2a8-gs32-binary"),
        )
    }
    if not values["model"]:
        pytest.skip("Pass a unified GS32 model and native artifacts to opt in to the EP long-context graph test.")
    assert all(values.values())
    paths = {name: Path(value).resolve(strict=True) for name, value in values.items()}
    config = json.loads((paths["model"] / "config.json").read_text())
    assert config["model_type"] == "deepseek_v4"
    assert config["quantization_config"]["format"] == "vq2_mixed_gs32_source"
    assert (paths["model"] / "model.safetensors.index.json").is_file()
    output = tmp_path / f"ep{ep_size}-long-context.json"
    log = output.with_suffix(".log")
    command = [sys.executable, "-u", "-m", MODULE_NAME, "--ep-size", str(ep_size), "--output", str(output)]
    for name, path in paths.items():
        command.extend([f"--{name}", str(path)])
    with log.open("w") as stream:
        child = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=variant_environment(os.environ),
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=CHILD_TIMEOUT_SECONDS,
        )
    assert child.returncode == 0, f"EP{ep_size} long-context graph failed; see {log}\n{log.read_text()[-16000:]}"
    report = json.loads(output.read_text())
    assert report["status"] == "PASS"
    assert [(entry["input_tokens"], entry["output_tokens"]) for entry in report["requests"]] == list(REQUEST_SHAPES)
    print(
        "VQ2A8_GS32_EP_LONG_CONTEXT_RESULT="
        + json.dumps(
            {
                "status": report["status"],
                "model_runner_version": report["model_runner_version"],
                "ranks": [
                    {
                        key: state[key]
                        for key in ("rank", "model_runner", "validity_owners", "decode", "prefill", "other")
                    }
                    for state in report["requests"][-1]["graph_state"]
                ],
                "report": str(output),
                "log": str(log),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "library", "binary", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--ep-size", type=int, choices=EP_SIZES, required=True)
    args = parser.parse_args()
    run_variant(args.model, args.library, args.binary, args.ep_size, args.output)
