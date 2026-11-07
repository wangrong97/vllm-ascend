# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in four-card DSpark acceptance with original draft weights and a GS32 target.

Each topology loads DSpark off/on sequentially. Passing requires real target
verification, rejection sampling, draft execution and decode graph replay on
every rank, together with identical greedy outputs. This is not a performance
benchmark or proof of long-context capacity.
"""

import argparse
import asyncio
import json
import math
import os
import sys
from pathlib import Path

import pytest

from tests.e2e.nightly.test_vq2a8_gs32_dp import completion_record, request_tokens, sampling_params
from tests.e2e.nightly.test_vq2a8_gs32_ep import _ReplayCounter, _runner_graph_state, variant_environment
from tests.e2e.nightly.test_vq2a8_gs32_full_graph import CHILD_TIMEOUT_SECONDS, LOGPROB_ATOL
from tests.e2e.nightly.test_vq2a8_gs32_full_graph import runner_options as graph_runner_options
from tests.e2e.nightly.vq2a8_gs32_helpers import native_paths
from tools.serve_vq2a8 import DSPARK_SPECULATIVE_TOKENS, decode_capture_sizes
from tools.tests.run_vq2a8_npu_ci import execute

MODULE_NAME = "tests.e2e.nightly.test_vq2a8_gs32_dspark"
STATE_METHOD = "vq2a8_gs32_dspark_state"
WORKER_EXTENSION = f"{MODULE_NAME}.GS32DSparkWorkerExtension"
PARALLEL_MODES = ("tp-ep", "dp-ep")
EP_SIZE = 4
LOCAL_MAX_SEQS = 16
OUTPUT_TOKENS = 32
MAX_MODEL_LEN = 128
TOKEN_BUDGET = 2048
KV_CACHE_BYTES = 4 * 1024**3
STATE_TIMEOUT_SECONDS = 30
WAVE_TIMEOUT_SECONDS = 600
LOG_TAIL_CHARS = 16000
PROMPTS = ("Explain why the sky is blue.", "Describe how plants use sunlight to grow.")


class _CallCounter:
    """Test-local callable instrumentation without inspecting device tensors."""

    def __init__(self, function):
        self.function = function
        self.calls = 0

    def __call__(self, *args, **kwargs):
        result = self.function(*args, **kwargs)
        self.calls += 1
        return result

    def __getattr__(self, name):
        return getattr(self.function, name)


class _VerificationCounter:
    def __init__(self, execute_model):
        self.execute_model = execute_model
        self.steps = 0
        self.draft_tokens = 0
        self.max_spec_requests = 0

    def __call__(self, scheduler_output, *args, **kwargs):
        result = self.execute_model(scheduler_output, *args, **kwargs)
        dummy = kwargs.get("dummy_run", args[1] if len(args) > 1 else False)
        scheduled = scheduler_output.scheduled_spec_decode_tokens
        if not dummy and scheduled:
            drafts = [len(tokens) for tokens in scheduled.values()]
            self.steps += 1
            self.draft_tokens += sum(drafts)
            self.max_spec_requests = max(self.max_spec_requests, sum(count > 0 for count in drafts))
        return result

    def snapshot(self):
        return {
            "steps": self.steps,
            "draft_tokens": self.draft_tokens,
            "max_spec_requests": self.max_spec_requests,
        }


def draft_graph_state(speculator):
    manager = speculator.query_cudagraph_manager
    buckets = {}
    for descriptor, graph in manager.graphs.items():
        assert callable(getattr(graph, "replay", None)), "Cannot verify original draft graph replay."
        if not isinstance(graph.replay, _ReplayCounter):
            graph.replay = _ReplayCounter(graph.replay)
        bucket = buckets.setdefault(str(descriptor.num_tokens), {"captures": 0, "replays": 0})
        bucket["captures"] += 1
        bucket["replays"] += graph.replay.num_replays
    return {
        "captures": sum(bucket["captures"] for bucket in buckets.values()),
        "replays": sum(bucket["replays"] for bucket in buckets.values()),
        "buckets": buckets,
    }


class GS32DSparkWorkerExtension:
    def vq2a8_gs32_dspark_state(self, output_dir):
        from vllm.distributed import get_dp_group, get_ep_group, get_tp_group

        runner = self.model_runner
        if not isinstance(runner.execute_model, _VerificationCounter):
            runner.execute_model = _VerificationCounter(runner.execute_model)
        state = _runner_graph_state(runner)
        state.update(
            rank=get_ep_group().rank_in_group,
            ep_size=get_ep_group().world_size,
            dp_size=get_dp_group().world_size,
            tp_size=get_tp_group().world_size,
            verification=runner.execute_model.snapshot(),
            target_banks=sum(hasattr(layer, "vq2a8_weights") for layer in runner.get_model().modules()),
            draft=None,
        )
        speculator = runner.speculator
        if speculator is not None:
            if not isinstance(speculator.propose, _CallCounter):
                speculator.propose = _CallCounter(speculator.propose)
            assert runner.rejection_sampler is not None, "DSpark needs target rejection sampling."
            if not isinstance(runner.rejection_sampler, _CallCounter):
                runner.rejection_sampler = _CallCounter(runner.rejection_sampler)
            draft_config = speculator.vllm_config.model_config
            state["draft"] = {
                "model": str(draft_config.model),
                "quantization": draft_config.quantization,
                "vq2a8_banks": sum(hasattr(layer, "vq2a8_weights") for layer in speculator.model.modules()),
                "proposal_calls": speculator.propose.calls,
                "rejection_calls": runner.rejection_sampler.calls,
                "graphs": draft_graph_state(speculator),
            }
        path = Path(output_dir) / f"ep{state['rank']}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, allow_nan=False), encoding="utf-8")
        temporary.replace(path)
        return {"rank": state["rank"]}


def runner_options(parallel_mode, draft_model, library, binary):
    if parallel_mode not in PARALLEL_MODES:
        raise ValueError("DSpark acceptance supports tp-ep or dp-ep.")
    dp_size = EP_SIZE if parallel_mode == "dp-ep" else 1
    proposal_count = DSPARK_SPECULATIVE_TOKENS if draft_model is not None else 0
    capture_sizes = decode_capture_sizes(LOCAL_MAX_SEQS, proposal_count)
    options = graph_runner_options("full_graph", library, binary)
    options.update(
        tensor_parallel_size=EP_SIZE if parallel_mode == "tp-ep" else 1,
        data_parallel_size=dp_size,
        data_parallel_size_local=dp_size,
        data_parallel_backend="mp",
        enable_expert_parallel=True,
        all2all_backend="flashinfer_all2allv",
        distributed_executor_backend="mp",
        worker_extension_cls=WORKER_EXTENSION,
        max_num_seqs=LOCAL_MAX_SEQS,
        max_num_batched_tokens=TOKEN_BUDGET,
        max_model_len=MAX_MODEL_LEN,
        block_size=32,
        kv_cache_memory_bytes=KV_CACHE_BYTES,
        enable_chunked_prefill=True,
    )
    options["compilation_config"].update(
        cudagraph_capture_sizes=capture_sizes, max_cudagraph_capture_size=capture_sizes[-1]
    )
    options["additional_config"].update(enable_shared_expert_dp=False, enable_dsa_cp=False)
    if draft_model is not None:
        options["speculative_config"] = {
            "method": "dspark",
            "model": str(draft_model),
            "num_speculative_tokens": proposal_count,
            "enforce_eager": False,
        }
    return options


def wave_specs(parallel_mode):
    if parallel_mode not in PARALLEL_MODES:
        raise ValueError("DSpark acceptance supports tp-ep or dp-ep.")
    dp_size = EP_SIZE if parallel_mode == "dp-ep" else 1

    def wave(name, counts, mixed=False):
        return {
            "name": name,
            "requests": [
                {
                    "request_id": f"{name}-{rank}-{index}",
                    "dp_rank": rank,
                    "prompt_index": (rank + index) % len(PROMPTS),
                    "input_tokens": 33 if mixed and index % 2 else 8,
                    "output_tokens": OUTPUT_TOKENS - (index % 5 if mixed else 0),
                }
                for rank, count in enumerate(counts)
                for index in range(count)
            ],
        }

    return [
        wave("balanced", [LOCAL_MAX_SEQS] * dp_size),
        wave("uneven", [0, 1, 7, LOCAL_MAX_SEQS] if dp_size > 1 else [7], mixed=True),
        wave("recover", [1] + [0] * (dp_size - 1)),
    ]


async def snapshot(engine, output_dir):
    await engine.wait_for_requests_to_drain(drain_timeout=STATE_TIMEOUT_SECONDS)
    output_dir.mkdir(parents=True)
    await engine.collective_rpc(STATE_METHOD, timeout=STATE_TIMEOUT_SECONDS, args=(str(output_dir),))
    paths = sorted(output_dir.glob("ep*.json"))
    assert {path.name for path in paths} == {f"ep{rank}.json" for rank in range(EP_SIZE)}
    states = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    assert sorted(state["rank"] for state in states) == list(range(EP_SIZE))
    return sorted(states, key=lambda state: state["rank"])


def check_states(states, parallel_mode, draft_model):
    expected_dp = EP_SIZE if parallel_mode == "dp-ep" else 1
    expected_tp = EP_SIZE if parallel_mode == "tp-ep" else 1
    for state in states:
        assert state["ep_size"] == EP_SIZE and state["dp_size"] == expected_dp and state["tp_size"] == expected_tp
        assert state["model_runner_version"] == 2
        assert state["target_banks"] > 0, "Target did not load GS32 banks."
        draft = state["draft"]
        if draft_model is None:
            assert draft is None
        else:
            assert Path(draft["model"]).resolve() == Path(draft_model).resolve()
            assert draft["quantization"] == "deepseek_v4_fp8", "Draft inherited target quantization."
            assert draft["vq2a8_banks"] == 0, "Draft must use original weights, not GS32 banks."
            assert draft["graphs"]["captures"] > 0, "Original draft has no captured decode graphs."


def check_wave_progress(before, after, wave, enabled):
    bucket = str(LOCAL_MAX_SEQS * (DSPARK_SPECULATIVE_TOKENS + 1 if enabled else 1))
    for old, new in zip(before, after, strict=True):
        assert old["rank"] == new["rank"]
        assert new["decode"]["replays"] > old["decode"]["replays"], "Rank did not replay a target decode graph."
        assert new["prefill"]["replays"] == new["other"]["replays"] == 0
        if enabled:
            # An idle DP rank participates in target collectives but need not
            # produce drafts or execute rejection sampling for a real request.
            active = new["dp_size"] == 1 or any(spec["dp_rank"] == new["rank"] for spec in wave["requests"])
            if active:
                assert new["verification"]["steps"] > old["verification"]["steps"], "No target verification."
                assert new["verification"]["draft_tokens"] > old["verification"]["draft_tokens"]
                assert new["draft"]["proposal_calls"] > old["draft"]["proposal_calls"], "No draft execution."
                assert new["draft"]["rejection_calls"] > old["draft"]["rejection_calls"], "No rejection sampling."
                assert new["draft"]["graphs"]["replays"] > old["draft"]["graphs"]["replays"], (
                    "Original draft did not replay a decode graph."
                )
            if wave["name"] == "balanced":
                assert new["verification"]["max_spec_requests"] == LOCAL_MAX_SEQS, "B16 was not verified."
                draft_bucket = str(LOCAL_MAX_SEQS * DSPARK_SPECULATIVE_TOKENS)
                assert (
                    new["draft"]["graphs"]["buckets"][draft_bucket]["replays"]
                    > old["draft"]["graphs"]["buckets"][draft_bucket]["replays"]
                ), "B16 original draft did not replay its query graph."
        else:
            assert new["verification"]["draft_tokens"] == 0
        if wave["name"] == "balanced":
            assert new["decode_buckets"][bucket]["replays"] > old["decode_buckets"][bucket]["replays"], (
                f"Rank {new['rank']} did not replay target bucket {bucket}."
            )


async def generate_wave(engine, seeds, wave, parallel_mode):
    pending = []
    try:
        await engine.pause_generation(mode="keep", clear_cache=False)
        try:
            for spec in wave["requests"]:
                tokens = request_tokens(seeds, spec)
                options = {"data_parallel_rank": spec["dp_rank"]} if parallel_mode == "dp-ep" else {}
                collector = await engine.add_request(
                    request_id=spec["request_id"],
                    prompt={"prompt_token_ids": tokens},
                    params=sampling_params(spec),
                    **options,
                )
                pending.append((collector, tokens, spec))
        finally:
            await engine.resume_generation()

        async def receive(collector, tokens, spec):
            result = await collector.get()
            return completion_record(result, tokens, spec)

        return await asyncio.gather(*(receive(*item) for item in pending))
    finally:
        for collector, _, _ in pending:
            collector.close()


def compare_reports(actual, expected):
    assert actual["parallel_mode"] == expected["parallel_mode"]
    assert len(actual["waves"]) == len(expected["waves"])
    for observed_wave, reference_wave in zip(actual["waves"], expected["waves"], strict=True):
        assert observed_wave["spec"] == reference_wave["spec"]
        observed = {record["request_id"]: record for record in observed_wave["requests"]}
        reference = {record["request_id"]: record for record in reference_wave["requests"]}
        requested = {spec["request_id"] for spec in observed_wave["spec"]["requests"]}
        assert len(observed) == len(observed_wave["requests"]) == len(requested)
        assert len(reference) == len(reference_wave["requests"]) == len(requested)
        assert observed.keys() == reference.keys() == requested
        for request_id, record in observed.items():
            target = reference[request_id]
            assert record["prompt_token_ids"] == target["prompt_token_ids"]
            assert record["token_ids"] == target["token_ids"], f"DSpark changed greedy output: {request_id}"
            for token, row, ref in zip(record["token_ids"], record["logprobs"], target["logprobs"], strict=True):
                assert row.keys() == ref.keys(), f"DSpark changed top-logprob token set: {request_id}"
                assert str(token) in row
                for token_id, value in row.items():
                    assert math.isfinite(value) and math.isfinite(ref[token_id])
                    assert math.isclose(value, ref[token_id], abs_tol=LOGPROB_ATOL, rel_tol=0), (
                        f"DSpark changed logprobs: {request_id}, token {token_id}"
                    )


async def run_variant(model, library, binary, draft_model, parallel_mode, output):
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    options = runner_options(parallel_mode, draft_model, library, binary)
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(model=str(model), **options))
    report = {"parallel_mode": parallel_mode, "dspark": draft_model is not None, "waves": []}
    state_root = output.with_suffix("")
    try:
        seeds = [engine.get_tokenizer().encode(text, add_special_tokens=False) for text in PROMPTS]
        assert all(seeds)
        previous = await snapshot(engine, state_root / "before")
        check_states(previous, parallel_mode, draft_model)
        report["before"] = previous
        for wave in wave_specs(parallel_mode):
            records = await asyncio.wait_for(
                generate_wave(engine, seeds, wave, parallel_mode), timeout=WAVE_TIMEOUT_SECONDS
            )
            current = await snapshot(engine, state_root / wave["name"])
            check_states(current, parallel_mode, draft_model)
            check_wave_progress(previous, current, wave, draft_model is not None)
            report["waves"].append({"spec": wave, "requests": records, "states": current})
            output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
            previous = current
    finally:
        engine.shutdown()


@pytest.mark.parametrize("parallel_mode", PARALLEL_MODES, ids=("tp4", "dp4"))
def test_gs32_dspark_matches_target_and_replays_verification_graphs(request, tmp_path, parallel_mode):
    values = {
        name: request.config.getoption(option, default=None)
        for name, option in (
            ("model", "--vq2a8-model"),
            ("gs32_library", "--vq2a8-gs32-library"),
            ("gs32_binary", "--vq2a8-gs32-binary"),
            ("draft_model", "--vq2a8-dspark-model"),
        )
    }
    if values["draft_model"] is None:
        pytest.skip("Pass --vq2a8-dspark-model to opt into four-card DSpark hardware acceptance.")
    paths = native_paths(values)
    environment = variant_environment(os.environ)
    environment["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    reports = {}
    for enabled in (False, True):
        name = "dspark" if enabled else "target"
        output = tmp_path / f"{parallel_mode}-{name}.json"
        log = output.with_suffix(".log")
        command = [sys.executable, "-u", "-m", MODULE_NAME, "--parallel-mode", parallel_mode, "--output", str(output)]
        for field in ("model", "gs32_library", "gs32_binary"):
            command.extend([f"--{field.removeprefix('gs32_')}", str(paths[field])])
        if enabled:
            command.extend(["--draft-model", str(paths["draft_model"])])
        returncode = execute(command, environment, log, CHILD_TIMEOUT_SECONDS)
        assert returncode == 0, (
            f"{parallel_mode} {name} failed; see {log}\n"
            + log.read_text(encoding="utf-8", errors="replace")[-LOG_TAIL_CHARS:]
        )
        reports[name] = json.loads(output.read_text(encoding="utf-8"))
    compare_reports(reports["dspark"], reports["target"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--draft-model", type=Path)
    parser.add_argument("--parallel-mode", choices=PARALLEL_MODES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run_variant(args.model, args.library, args.binary, args.draft_model, args.parallel_mode, args.output))
