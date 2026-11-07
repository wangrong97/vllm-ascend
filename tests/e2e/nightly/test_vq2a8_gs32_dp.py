# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit TP1/DP2/4/8/EP2/4/8 acceptance through the serving AsyncLLM client.

Run this file explicitly with the existing --vq2a8-model and native artifact
options. It is intentionally absent from the one-click NPU CI. A same-card-count
TP/EP eager B1 reference, DP eager and DP decode FULL graphs load sequentially;
every DP engine receives directed requests and must independently report its EP
partition, real/dummy forwards and graph use.
This is correctness coverage, not a serving throughput benchmark.
"""

import argparse
import asyncio
import json
import math
import os
import sys
from pathlib import Path

import pytest

from tests.e2e.nightly.test_vq2a8_gs32_ep import (
    EP_LOGPROB_ATOL,
    _runner_graph_state,
    check_inventory,
    graph_state,
    variant_environment,
)
from tests.e2e.nightly.test_vq2a8_gs32_ep import runner_options as ep_runner_options
from tests.e2e.nightly.test_vq2a8_gs32_full_graph import CHILD_TIMEOUT_SECONDS, LOGPROB_ATOL
from tests.e2e.nightly.test_vq2a8_gs32_full_graph import runner_options as decode_runner_options
from tests.e2e.nightly.vq2a8_gs32_helpers import checkpoint_inventory, native_paths
from tools.tests.run_vq2a8_npu_ci import execute

DP_SIZES = (2, 4, 8)
EXECUTIONS = ("fused", "full_graph")
REFERENCE_EXECUTION = "tp_reference"
MODULE_NAME = "tests.e2e.nightly.test_vq2a8_gs32_dp"
STATE_METHOD = "vq2a8_gs32_dp_state"
WORKER_EXTENSION = f"{MODULE_NAME}.GS32DPWorkerExtension"
CAPTURE_SIZES = (1, 2, 4, 8, 16)
LOCAL_MAX_SEQS = max(CAPTURE_SIZES)
SHORT_PROMPT_TOKENS = 8
LONG_PROMPT_TOKENS = 33
OUTPUT_TOKENS = 8
LOGPROBS = 5
MAX_MODEL_LEN = 64
KV_CACHE_BYTES = 4 * 1024**3
WAVE_TIMEOUT_SECONDS = 300
STATE_TIMEOUT_SECONDS = 30
LOG_TAIL_CHARS = 16000
PROMPTS = ("Explain why the sky is blue in one paragraph.", "Describe how plants use sunlight to grow.")


def runner_options(execution, library, binary, dp_size):
    if type(dp_size) is not int or dp_size not in DP_SIZES:
        raise ValueError("GS32 DP acceptance supports DP2, DP4 or DP8.")
    options = decode_runner_options(execution, library, binary)
    options.update(
        tensor_parallel_size=1,
        data_parallel_size=dp_size,
        data_parallel_size_local=dp_size,
        data_parallel_backend="mp",
        data_parallel_external_lb=False,
        data_parallel_hybrid_lb=False,
        enable_expert_parallel=True,
        enable_eplb=False,
        all2all_backend="flashinfer_all2allv",
        distributed_executor_backend="mp",
        worker_extension_cls=WORKER_EXTENSION,
        max_num_seqs=LOCAL_MAX_SEQS,
        max_num_batched_tokens=LOCAL_MAX_SEQS * SHORT_PROMPT_TOKENS,
        max_model_len=MAX_MODEL_LEN,
        block_size=32,
        kv_cache_memory_bytes=KV_CACHE_BYTES,
        enable_chunked_prefill=True,
    )
    options["additional_config"].update(enable_shared_expert_dp=False, enable_dsa_cp=False)
    if execution == "full_graph":
        options["compilation_config"].update(
            cudagraph_capture_sizes=list(CAPTURE_SIZES),
            max_cudagraph_capture_size=LOCAL_MAX_SEQS,
        )
    return options


def reference_runner_options(library, binary, dp_size):
    if type(dp_size) is not int or dp_size not in DP_SIZES:
        raise ValueError("GS32 DP acceptance supports DP2, DP4 or DP8.")
    options = ep_runner_options("fused", library, binary, dp_size)
    options.update(max_num_seqs=1, max_model_len=MAX_MODEL_LEN, block_size=32, kv_cache_memory_bytes=KV_CACHE_BYTES)
    return options


def wave_specs(dp_size):
    """Stable request identities allow comparisons despite asynchronous scheduling."""
    if type(dp_size) is not int or dp_size not in DP_SIZES:
        raise ValueError("GS32 DP acceptance supports DP2, DP4 or DP8.")

    def wave(name, counts, mixed=False):
        requests = []
        for rank, count in enumerate(counts):
            for index in range(count):
                requests.append(
                    {
                        "request_id": f"{name}-dp{rank}-{index}",
                        "dp_rank": rank,
                        "prompt_index": (rank + index) % len(PROMPTS),
                        "input_tokens": LONG_PROMPT_TOKENS if mixed and index % 2 else SHORT_PROMPT_TOKENS,
                        "output_tokens": OUTPUT_TOKENS - (index % 3 if mixed else 0),
                    }
                )
        return {"name": name, "requests": requests}

    waves = [wave(f"only-dp{rank}", [int(index == rank) for index in range(dp_size)]) for rank in range(dp_size)]
    waves.append(wave("balanced", [LOCAL_MAX_SEQS] * dp_size))
    # First DP engine remains idle, while the last reaches the configured local
    # concurrency. Different prompts/budgets exercise mixed prefill and retirements.
    counts = [0, LOCAL_MAX_SEQS] if dp_size == 2 else [0, 1, 7, LOCAL_MAX_SEQS] * (dp_size // 4)
    waves.append(wave("uneven", counts, mixed=True))
    waves.append(wave("recover-single", [1] + [0] * (dp_size - 1)))
    waves.append(wave("recover-balanced", [3] * dp_size))
    return waves


class _ForwardCounter:
    """Test-only host counters; no device reads or synchronization per forward."""

    def __init__(self, execute_model):
        self.execute_model = execute_model
        self.real = 0
        self.dummy = 0
        self.max_real_requests = 0

    def __call__(self, scheduler_output, *args, **kwargs):
        result = self.execute_model(scheduler_output, *args, **kwargs)
        dummy = kwargs.get("dummy_run", args[1] if len(args) > 1 else False)
        if dummy:
            self.dummy += 1
        elif scheduler_output.total_num_scheduled_tokens:
            self.real += 1
            self.max_real_requests = max(self.max_real_requests, len(scheduler_output.num_scheduled_tokens))
        return result

    def snapshot(self):
        return {"real": self.real, "dummy": self.dummy, "max_real_requests": self.max_real_requests}


class GS32DPWorkerExtension:
    def vq2a8_gs32_dp_state(self, output_dir):
        from vllm.distributed import get_dp_group, get_ep_group, get_tp_group

        runner = self.model_runner
        if not isinstance(runner.execute_model, _ForwardCounter):
            runner.execute_model = _ForwardCounter(runner.execute_model)
        state = _runner_graph_state(runner)
        state.update(
            rank=get_ep_group().rank_in_group,
            ep_size=get_ep_group().world_size,
            dp_rank=get_dp_group().rank_in_group,
            dp_size=get_dp_group().world_size,
            tp_rank=get_tp_group().rank_in_group,
            tp_size=get_tp_group().world_size,
            forwards=runner.execute_model.snapshot(),
            validity_owners=len(getattr(runner, "_quantization_validity_buffers", ())),
            layers=[],
        )
        for layer in runner.get_model().modules():
            bank = getattr(layer, "vq2a8_weights", None)
            if bank is None:
                continue
            moe_config = layer.moe_config
            assert layer.vq2a8_ep_rank == moe_config.ep_rank == state["rank"]
            assert layer.vq2a8_ep_size == moe_config.ep_size == state["ep_size"]
            state["layers"].append(
                {
                    "layer": layer.vq2a8_layer_index,
                    "expert_ids": list(layer.vq2a8_local_global_expert_ids),
                    "moe_ep_rank": moe_config.ep_rank,
                    "moe_ep_size": moe_config.ep_size,
                    "moe_local_experts": moe_config.num_local_experts,
                    "graph_row_capacity": bank.graph_row_capacity,
                    "banks": {
                        kind: {
                            quant_type: {
                                "expert_ids": [expert + layer.vq2a8_expert_start for expert in group["expert_ids"]],
                                "field_names": list(group["fields"]),
                                "field_counts": [value.shape[0] for value in group["fields"].values()],
                            }
                            for quant_type, group in groups.items()
                        }
                        for kind, groups in bank.banks.items()
                    },
                }
            )
        state["layers"].sort(key=lambda item: item["layer"])
        # Internal DP broadcasts utility RPCs but returns only one engine's
        # response. Separate rank files prove that every engine participated.
        path = Path(output_dir) / f"dp{state['dp_rank']}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, allow_nan=False), encoding="utf-8")
        temporary.replace(path)
        return {"dp_rank": state["dp_rank"], "path": str(path)}


def read_states(output_dir, dp_size):
    paths = sorted(Path(output_dir).glob("dp*.json"))
    assert {path.name for path in paths} == {f"dp{rank}.json" for rank in range(dp_size)}, (
        "Worker RPC did not report every DP engine. A single engine's RPC return is insufficient."
    )
    states = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    assert sorted(state["dp_rank"] for state in states) == list(range(dp_size)), "Missing/duplicate DP ranks."
    return sorted(states, key=lambda state: state["dp_rank"])


async def snapshot(engine, output_dir, dp_size):
    # FINAL_ONLY responses can arrive before the final DP dummy/retirement
    # iteration. Drain the tagged v0.30 AsyncLLM before assigning a new wave's
    # counter baseline, rather than guessing from client request completion.
    await engine.wait_for_requests_to_drain(drain_timeout=STATE_TIMEOUT_SECONDS)
    output_dir.mkdir(parents=True)
    await engine.collective_rpc(STATE_METHOD, timeout=STATE_TIMEOUT_SECONDS, args=(str(output_dir),))
    return read_states(output_dir, dp_size)


def check_state_inventory(states, config, dp_size, entries, execution):
    check_inventory(states, config, dp_size, entries)
    for rank, state in enumerate(states):
        assert state["model_runner_version"] == 2
        assert state["dp_rank"] == state["rank"] == rank
        assert state["dp_size"] == state["ep_size"] == dp_size
        assert state["tp_rank"] == 0 and state["tp_size"] == 1
        assert state["validity_owners"] == (len(entries) if execution == "full_graph" else 0)
        for layer in state["layers"]:
            assert layer["moe_ep_rank"] == rank and layer["moe_ep_size"] == dp_size
            assert layer["moe_local_experts"] == config["n_routed_experts"] // dp_size
            if execution == "full_graph":
                assert layer["graph_row_capacity"] == LOCAL_MAX_SEQS * dp_size * config["num_experts_per_tok"]


def check_wave_progress(execution, before, after, spec):
    active = {item["dp_rank"] for item in spec["requests"]}
    assert len(before) == len(after)
    for old, new in zip(before, after, strict=True):
        rank = old["dp_rank"]
        assert rank == new["dp_rank"]
        assert old["layers"] == new["layers"], "Resident experts changed across request waves."
        real = new["forwards"]["real"] - old["forwards"]["real"]
        dummy = new["forwards"]["dummy"] - old["forwards"]["dummy"]
        assert real > 0 if rank in active else real == 0, f"Requests ran on the wrong DP engine: {rank}."
        if rank not in active:
            assert dummy > 0, f"Idle DP{rank} did not join the other engines' MoE collectives."
        if spec["name"] == "balanced":
            assert new["forwards"]["max_real_requests"] == LOCAL_MAX_SEQS, (
                f"DP{rank} did not execute the local B{LOCAL_MAX_SEQS} boundary."
            )
        for phase in ("prefill", "decode", "other"):
            assert old[phase]["captures"] == new[phase]["captures"], "Generation recaptured a graph."
            delta = new[phase]["replays"] - old[phase]["replays"]
            if execution == "full_graph" and phase == "decode":
                assert set(new["decode_buckets"]) == {str(size) for size in CAPTURE_SIZES}
                assert delta > 0, f"DP{rank} did not replay a decode graph, including idle-rank participation."
                if spec["name"] == "balanced":
                    bucket = str(LOCAL_MAX_SEQS)
                    assert new["decode_buckets"][bucket]["replays"] > old["decode_buckets"][bucket]["replays"], (
                        f"DP{rank} captured B{LOCAL_MAX_SEQS} but did not replay it."
                    )
            else:
                assert delta == 0, f"Unexpected {phase} graph replay on DP{rank}."


def compare_reports(actual, expected):
    assert actual["dp_size"] == expected["dp_size"]
    assert [wave["spec"] for wave in actual["waves"]] == [wave["spec"] for wave in expected["waves"]]
    ranks = set()
    for got_wave, want_wave in zip(actual["waves"], expected["waves"], strict=True):
        assert len(got_wave["requests"]) == len(want_wave["requests"]) == len(got_wave["spec"]["requests"])
        for got, want, spec in zip(
            got_wave["requests"], want_wave["requests"], got_wave["spec"]["requests"], strict=True
        ):
            check_request_record(got, spec)
            check_request_record(want, spec)
            assert got["request_id"] == want["request_id"] == spec["request_id"]
            assert got["dp_rank"] == want["dp_rank"] == spec["dp_rank"]
            assert got["prompt_token_ids"] == want["prompt_token_ids"]
            assert got["token_ids"] == want["token_ids"], f"DP{got['dp_rank']} graph changed greedy outputs."
            assert len(got["token_ids"]) == len(got["logprobs"]) == len(want["logprobs"]) == spec["output_tokens"]
            for observed, reference in zip(got["logprobs"], want["logprobs"], strict=True):
                assert observed.keys() == reference.keys(), "Graph changed top-five token IDs."
                for token, value in observed.items():
                    assert math.isfinite(value) and math.isfinite(reference[token])
                    assert abs(value - reference[token]) <= LOGPROB_ATOL, (
                        f"DP{got['dp_rank']} logprob mismatch for token {token}: {value} vs {reference[token]}."
                    )
            ranks.add(got["dp_rank"])
    assert ranks == set(range(actual["dp_size"])), "Numerical comparison omitted a DP engine."


def compare_tp_reference(actual, reference):
    """Catch faults shared by both DP implementations, using the existing TP path."""
    assert reference["dp_size"] == 1 and reference["tp_size"] == actual["dp_size"]
    assert [wave["spec"] for wave in actual["waves"]] == [wave["spec"] for wave in reference["waves"]]
    ranks = set()
    for got_wave, ref_wave in zip(actual["waves"], reference["waves"], strict=True):
        specs = got_wave["spec"]["requests"]
        assert len(got_wave["requests"]) == len(ref_wave["requests"]) == len(specs)
        for got, ref, spec in zip(got_wave["requests"], ref_wave["requests"], specs, strict=True):
            check_request_record(got, spec)
            check_request_record(ref, spec)
            assert got["request_id"] == ref["request_id"] == spec["request_id"]
            assert got["dp_rank"] == ref["candidate_dp_rank"] == spec["dp_rank"]
            assert got["prompt_token_ids"] == ref["prompt_token_ids"]
            assert got["token_ids"] == ref["token_ids"], f"DP{got['dp_rank']} greedy outputs differ from TP reference."
            assert len(got["token_ids"]) == len(got["logprobs"]) == len(ref["logprobs"]) == spec["output_tokens"]
            # TP changes BF16 GEMM/reduction order. Match existing EP acceptance:
            # compare generated-token probabilities, not the entire top-k set.
            for token, observed, expected in zip(got["token_ids"], got["logprobs"], ref["logprobs"], strict=True):
                value, target = observed[str(token)], expected[str(token)]
                assert math.isfinite(value) and math.isfinite(target)
                assert abs(value - target) <= EP_LOGPROB_ATOL, (
                    f"DP{got['dp_rank']} logprob differs from TP reference for token {token}: {value} vs {target}."
                )
            ranks.add(got["dp_rank"])
    assert ranks == set(range(actual["dp_size"])), "TP numerical comparison omitted a DP engine."


def request_tokens(seeds, spec):
    seed = seeds[spec["prompt_index"]]
    assert seed, "Acceptance prompt must tokenize to a nonempty sequence."
    count = spec["input_tokens"]
    return (seed * (count // len(seed) + 1))[:count]


def check_request_record(record, spec):
    assert len(record["token_ids"]) == len(record["logprobs"]) == spec["output_tokens"]
    for token, row in zip(record["token_ids"], record["logprobs"], strict=True):
        assert len(row) >= LOGPROBS, "Missing top-five logprobs."
        assert str(token) in row, "Missing generated-token logprob."
        assert all(math.isfinite(value) for value in row.values()), "Nonfinite logprob."


def completion_record(result, tokens, spec):
    assert result is not None and result.finished and len(result.outputs) == 1
    assert list(result.prompt_token_ids) == tokens
    completion = result.outputs[0]
    assert len(completion.token_ids) == spec["output_tokens"] and completion.logprobs is not None
    record = {
        "request_id": spec["request_id"],
        "prompt_token_ids": tokens,
        "token_ids": list(completion.token_ids),
        "logprobs": [{str(token): value.logprob for token, value in row.items()} for row in completion.logprobs],
    }
    check_request_record(record, spec)
    return record


def sampling_params(spec):
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    return SamplingParams(
        temperature=0,
        max_tokens=spec["output_tokens"],
        min_tokens=spec["output_tokens"],
        ignore_eos=True,
        seed=0,
        logprobs=LOGPROBS,
        output_kind=RequestOutputKind.FINAL_ONLY,
    )


async def generate_request(engine, tokens, spec):
    result = None
    async for result in engine.generate(
        prompt={"prompt_token_ids": tokens},
        sampling_params=sampling_params(spec),
        request_id=spec["request_id"],
        data_parallel_rank=spec["dp_rank"],
    ):
        pass
    return {**completion_record(result, tokens, spec), "dp_rank": spec["dp_rank"]}


async def generate_balanced_wave(engine, seeds, wave):
    """Queue the whole boundary batch with v0.30's public pause/add/resume API.

    In v0.30, pause(mode='keep') prevents scheduling while still accepting adds.
    Awaiting add_request before resume queues all sixteen requests on each DP
    engine; observed forward counts and B16 replays must still prove coverage.
    """
    pending = []
    try:
        await engine.pause_generation(mode="keep", clear_cache=False)
        try:
            for spec in wave["requests"]:
                tokens = request_tokens(seeds, spec)
                collector = await engine.add_request(
                    request_id=spec["request_id"],
                    prompt={"prompt_token_ids": tokens},
                    params=sampling_params(spec),
                    data_parallel_rank=spec["dp_rank"],
                )
                pending.append((collector, tokens, spec))
        finally:
            await engine.resume_generation()

        async def receive(collector, tokens, spec):
            result = await collector.get()  # FINAL_ONLY; finished is asserted below.
            return {**completion_record(result, tokens, spec), "dp_rank": spec["dp_rank"]}

        return await asyncio.gather(*(receive(*item) for item in pending))
    finally:
        for collector, _, _ in pending:
            collector.close()


def run_tp_reference(model, library, binary, dp_size, output):
    from vllm import SamplingParams

    from tests.e2e.conftest import VllmRunner

    config, _, entries = checkpoint_inventory(model)
    report = {"execution": REFERENCE_EXECUTION, "tp_size": dp_size, "dp_size": 1, "local_max_seqs": 1, "waves": []}
    print("VQ2A8_GS32_DP_REFERENCE_STARTUP=" + json.dumps(report), flush=True)
    with VllmRunner(str(model), **reference_runner_options(library, binary, dp_size)) as runner:
        llm = runner.model
        previous = graph_state(llm, dp_size, expected_version=2)
        check_inventory(previous, config, dp_size, entries)
        report["before"] = previous
        seeds = [llm.get_tokenizer().encode(text, add_special_tokens=False) for text in PROMPTS]
        for wave in wave_specs(dp_size):
            prompts = [request_tokens(seeds, spec) for spec in wave["requests"]]
            sampling = [
                SamplingParams(
                    temperature=0,
                    max_tokens=spec["output_tokens"],
                    min_tokens=spec["output_tokens"],
                    ignore_eos=True,
                    seed=0,
                    logprobs=LOGPROBS,
                )
                for spec in wave["requests"]
            ]
            # Queue the same wave while the scheduler admits only one request
            # at a time; no DP dispatch or multi-request reference can mask faults.
            results = llm.generate([{"prompt_token_ids": tokens} for tokens in prompts], sampling, use_tqdm=False)
            assert len(results) == len(wave["requests"])
            records = [
                {**completion_record(result, tokens, spec), "candidate_dp_rank": spec["dp_rank"]}
                for result, tokens, spec in zip(results, prompts, wave["requests"], strict=True)
            ]
            report["waves"].append({"spec": wave, "requests": records})
            output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
            print("VQ2A8_GS32_DP_REFERENCE_WAVE=" + json.dumps({"name": wave["name"]}), flush=True)
        current = graph_state(llm, dp_size, expected_version=2)
        check_inventory(current, config, dp_size, entries)
        assert [state["layers"] for state in previous] == [state["layers"] for state in current]
        report["after"] = current
        output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


async def run_variant(model, library, binary, execution, dp_size, output):
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    config, _, entries = checkpoint_inventory(model)
    report = {"execution": execution, "dp_size": dp_size, "local_max_seqs": LOCAL_MAX_SEQS, "waves": []}
    state_root = output.with_suffix("")
    options = runner_options(execution, library, binary, dp_size)
    print("VQ2A8_GS32_DP_STARTUP=" + json.dumps(report), flush=True)
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(model=str(model), **options))
    try:
        seeds = [engine.get_tokenizer().encode(text, add_special_tokens=False) for text in PROMPTS]
        assert all(seeds)
        previous = await snapshot(engine, state_root / "before", dp_size)
        check_state_inventory(previous, config, dp_size, entries, execution)
        report["before"] = previous
        for wave in wave_specs(dp_size):
            if wave["name"] == "balanced":
                operation = generate_balanced_wave(engine, seeds, wave)
            else:
                operation = asyncio.gather(
                    *(generate_request(engine, request_tokens(seeds, spec), spec) for spec in wave["requests"])
                )
            records = await asyncio.wait_for(operation, timeout=WAVE_TIMEOUT_SECONDS)
            current = await snapshot(engine, state_root / wave["name"], dp_size)
            check_state_inventory(current, config, dp_size, entries, execution)
            check_wave_progress(execution, previous, current, wave)
            report["waves"].append({"spec": wave, "requests": records, "state": current})
            output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
            print("VQ2A8_GS32_DP_WAVE=" + json.dumps({"name": wave["name"], "requests": len(records)}), flush=True)
            previous = current
    finally:
        engine.shutdown()


@pytest.mark.parametrize("dp_size", DP_SIZES, ids=lambda size: f"dp{size}")
def test_gs32_dp_eager_and_decode_graphs(request, tmp_path, dp_size):
    """Opt-in real checkpoint acceptance; never included in default NPU CI."""
    values = {
        name: request.config.getoption(flag, default=None)
        for name, flag in (
            ("model", "--vq2a8-model"),
            ("gs32_library", "--vq2a8-gs32-library"),
            ("gs32_binary", "--vq2a8-gs32-binary"),
        )
    }
    if values["model"] is None:
        pytest.skip("Pass --vq2a8-model to opt into real TP1/DP/EP serving acceptance.")
    paths = native_paths(values)
    environment = variant_environment(os.environ)
    environment["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    reports = {}
    for execution in (REFERENCE_EXECUTION, *EXECUTIONS):
        output = tmp_path / f"dp{dp_size}-{execution}.json"
        log = output.with_suffix(".log")
        command = [
            sys.executable,
            "-u",
            "-m",
            MODULE_NAME,
            "--dp-size",
            str(dp_size),
            "--execution",
            execution,
            "--output",
            str(output),
        ]
        for name, path in paths.items():
            command.extend([f"--{name.removeprefix('gs32_')}", str(path)])
        returncode = execute(command, environment, log, CHILD_TIMEOUT_SECONDS)
        assert returncode == 0, (
            f"DP{dp_size} {execution} failed; see {log}\n"
            + log.read_text(encoding="utf-8", errors="replace")[-LOG_TAIL_CHARS:]
        )
        reports[execution] = json.loads(output.read_text(encoding="utf-8"))
    compare_reports(reports["full_graph"], reports["fused"])
    compare_tp_reference(reports["fused"], reports[REFERENCE_EXECUTION])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execution", choices=(REFERENCE_EXECUTION, *EXECUTIONS), required=True)
    parser.add_argument("--dp-size", type=int, choices=DP_SIZES, required=True)
    args = parser.parse_args()
    if args.execution == REFERENCE_EXECUTION:
        run_tp_reference(args.model, args.library, args.binary, args.dp_size, args.output)
    else:
        asyncio.run(run_variant(args.model, args.library, args.binary, args.execution, args.dp_size, args.output))
