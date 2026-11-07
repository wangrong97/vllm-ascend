# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in EP2/4/8 real-model and graph acceptance, with sequential model loads.

By default each topology is compared with a single-card reference. This reference
can be explicitly omitted when the model cannot fit on one card. Each topology's
eager run is compared with decode full graphs. Every rank must report local
weight ownership and actual graph replay. A separate dispatch + bank + combine + collective graph moves eight routes
between global experts, exercising zero-route ranks through the real dispatcher.
These tests check correctness, not serving latency or throughput.
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path

import pytest

from tests.e2e.nightly.test_vq2a8_gs32_full_graph import (
    CHILD_TIMEOUT_SECONDS,
    LOGPROB_ATOL,
    OUTPUT_TOKENS,
    REPO_ROOT,
    compare_requests,
)
from tests.e2e.nightly.test_vq2a8_gs32_full_graph import runner_options as decode_runner_options
from tests.e2e.nightly.vq2a8_gs32_helpers import _ReplayCounter as _ReplayCounter
from tests.e2e.nightly.vq2a8_gs32_helpers import checkpoint_inventory, native_paths
from tests.e2e.nightly.vq2a8_gs32_helpers import runner_graph_state as _runner_graph_state
from tools.tests.run_vq2a8_npu_ci import execute

EP_SIZES = (2, 4, 8)
EXECUTIONS = ("fused", "full_graph")
MODULE_NAME = "tests.e2e.nightly.test_vq2a8_gs32_ep"
GRAPH_STATE_METHOD = "vq2a8_gs32_ep_state"
ROUTE_METHOD = "vq2a8_gs32_ep_unbalanced_routes"
WORKER_EXTENSION = f"{MODULE_NAME}.GS32EPWorkerExtension"
# TP changes BF16 GEMM/reduction order. Same-topology graph comparisons keep
# the stricter existing 1e-3 tolerance and exact top-five token sets.
EP_LOGPROB_ATOL = 0.05
ROUTE_CAPACITY = 8
ROUTE_RTOL = 0.01
ROUTE_ATOL = 0.01
BATCH_SMOKE_EP_SIZE = 2
BATCH_SMOKE_EP_SIZES = (2, 4)
BATCH_SMOKE_INPUT_TOKENS = 8
BATCH_SMOKE_CAPTURE_SIZES = (1, 2, 4, 8, 16, 32, 64)
BATCH_SMOKE_SIZES = (1, 2, 3, 5, 7, 8, 16, 31, 32, 63, 64, 1, 64)
BATCH_SMOKE_MIXED_OUTPUT_TOKENS = (2, 3, 4, 2, 3, 4, 2, 4)
# Set existing vLLM/OpenMP/MKL controls before the child imports torch. This
# correctness probe uses fresh worker processes and a CPU thread limit already
# established at import time. These are not serving performance defaults.
VARIANT_ENVIRONMENT = (
    ("VLLM_WORKER_MULTIPROC_METHOD", "spawn"),
    ("OMP_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
)


INPUT_TOKENS = 128
PROMPT_TEXTS = ("请用中文简要介绍计算机的工作原理。", "请解释植物如何利用阳光制造养分。")


def build_prompts(tokenizer):
    prompts = []
    for text in PROMPT_TEXTS:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        assert tokens, "Acceptance prompt must tokenize to a nonempty sequence."
        prompts.append((tokens * ((INPUT_TOKENS + 1) // len(tokens) + 1))[: INPUT_TOKENS + 1])
    first, second = prompts[0][:INPUT_TOKENS], prompts[1][:INPUT_TOKENS]
    assert first != second, "Prefill replay must consume different input tokens."
    return [first, second, first[:-1], prompts[0], first]


def variant_environment(parent_environment):
    return {**parent_environment, **dict(VARIANT_ENVIRONMENT)}


def runner_options(execution, library, binary, ep_size, fusion="none"):
    if type(ep_size) is not int or ep_size not in (1, *EP_SIZES):
        raise ValueError("GS32 EP acceptance supports only EP1/2/4/8.")
    options = decode_runner_options(execution, library, binary)
    options["additional_config"]["vq2a8"].update(gs32_backend="native", gs32_fusion=fusion)
    options.update(
        # The recovery sequence includes INPUT_TOKENS + 1; reserve the full
        # generation budget too, independently of the prefill chunk size.
        max_model_len=INPUT_TOKENS + 1 + OUTPUT_TOKENS,
        max_num_batched_tokens=INPUT_TOKENS,
        enable_chunked_prefill=True,
        tensor_parallel_size=ep_size,
        data_parallel_size=1,
        enable_expert_parallel=ep_size > 1,
        enable_eplb=False,
        all2all_backend="flashinfer_all2allv",
        distributed_executor_backend="mp" if ep_size > 1 else "uni",
        worker_extension_cls=WORKER_EXTENSION,
    )
    options["additional_config"].update(enable_shared_expert_dp=False, enable_dsa_cp=False)
    return options


def batch_runner_options(execution, library, binary, ep_size=BATCH_SMOKE_EP_SIZE, fusion="none"):
    if ep_size not in BATCH_SMOKE_EP_SIZES:
        raise ValueError(f"Batch smoke requires ep_size in {BATCH_SMOKE_EP_SIZES}.")
    options = runner_options(execution, library, binary, ep_size, fusion)
    options.update(
        max_num_seqs=max(BATCH_SMOKE_CAPTURE_SIZES),
        max_num_batched_tokens=max(BATCH_SMOKE_CAPTURE_SIZES) * BATCH_SMOKE_INPUT_TOKENS,
        max_model_len=64,
        block_size=32,
        kv_cache_memory_bytes=4 * 1024**3,
    )
    if execution == "full_graph":
        options["compilation_config"].update(
            cudagraph_capture_sizes=list(BATCH_SMOKE_CAPTURE_SIZES),
            max_cudagraph_capture_size=max(BATCH_SMOKE_CAPTURE_SIZES),
        )
    return options


class GS32EPWorkerExtension:
    """Named test RPCs inspect all ranks and count actual graph replay submissions."""

    def vq2a8_gs32_ep_state(self):
        from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size

        runner = self.model_runner
        state = _runner_graph_state(runner)
        state.update(
            rank=get_tensor_model_parallel_rank(),
            ep_size=get_tensor_model_parallel_world_size(),
            validity_owners=len(getattr(runner, "_quantization_validity_buffers", ())),
            layers=[],
            full_ffn=[],
        )
        for layer in runner.get_model().modules():
            bank = getattr(layer, "vq2a8_weights", None)
            if bank is None:
                continue
            assert layer.vq2a8_ep_rank == state["rank"] and layer.vq2a8_ep_size == state["ep_size"]
            if bank.fusion == "full_ffn":
                full = bank.full_ffn_bank
                state["layers"].append(
                    {
                        "layer": layer.vq2a8_layer_index,
                        "expert_ids": list(layer.vq2a8_local_global_expert_ids),
                        "fusion": bank.fusion,
                        "full_ffn": {
                            "expert_ids": [expert + layer.vq2a8_expert_start for expert in full["expert_ids"]],
                            "gate_w4": full["gate_w4"].tolist(),
                            "down_w4": full["down_w4"].tolist(),
                            "field_shapes": {name: list(value.shape) for name, value in full["fields"].items()},
                        },
                    }
                )
                state["full_ffn"].append(
                    {
                        "layer": layer.vq2a8_layer_index,
                        "submissions": bank.full_ffn_submissions,
                        "captures": bank.full_ffn_graph_captures,
                    }
                )
                continue
            state["layers"].append(
                {
                    "layer": layer.vq2a8_layer_index,
                    "expert_ids": list(layer.vq2a8_local_global_expert_ids),
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
        return state

    def vq2a8_gs32_ep_unbalanced_routes(self, output_path):
        import torch

        # A utility RPC does not inherit execute_model's inference context.
        # Cover capture_begin and replay too: both can update inference tensors
        # created by the model's earlier captures, outside bank.apply itself.
        with torch.inference_mode():
            return _run_unbalanced_routes(self.model_runner, output_path)


def _run_unbalanced_routes(runner, output_path):
    import torch
    from vllm.distributed import get_tensor_model_parallel_rank, tensor_model_parallel_all_reduce

    from vllm_ascend.ops.fused_moe.dataclass.moe_quant import MoEQuantParams
    from vllm_ascend.ops.fused_moe.dataclass.router_input import MoeRouterInput
    from vllm_ascend.ops.fused_moe.dataclass.token_dispatcher import MoETokenDispatchInput
    from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithAllGather
    from vllm_ascend.quantization.quant_type import QuantType

    config = runner.vllm_config.model_config.hf_config
    # Hash layers may store only IDs from one EP rank. Choose a complete
    # non-hash layer so the changed target really moves work across ranks.
    layer = next(
        module
        for module in runner.get_model().modules()
        if hasattr(module, "vq2a8_weights") and len(module.vq2a8_expert_ids) == config.n_routed_experts
    )
    bank = layer.vq2a8_weights
    rank = get_tensor_model_parallel_rank()
    global_ids = runner.vllm_config.quant_config.layer_spec(layer.vq2a8_layer_index).expert_ids
    targets = (global_ids[0], global_ids[-1], global_ids[0])
    generator = torch.Generator(device="cpu").manual_seed(12861)
    original = (torch.randn(ROUTE_CAPACITY, bank.specs["gate_up"].columns, generator=generator) * 0.1).bfloat16()
    hidden = original.to(bank.device)
    topk_ids = torch.zeros((ROUTE_CAPACITY, 1), dtype=torch.int32, device=bank.device)
    topk_weights = torch.ones((ROUTE_CAPACITY, 1), dtype=torch.float32, device=bank.device)
    dispatcher = TokenDispatcherWithAllGather(
        top_k=1, num_experts=config.n_routed_experts, num_local_experts=bank.num_experts
    )
    dispatch_input = MoETokenDispatchInput(
        hidden_states=hidden,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        routing=MoeRouterInput(
            expert_map=layer.ascend_expert_map,
            global_redundant_expert_num=0,
            mc2_mask=None,
            apply_router_weight_on_input=False,
        ),
        quant=MoEQuantParams(quant_type=QuantType.VQ2A8),
    )
    owned = set(layer.vq2a8_local_global_expert_ids)
    validity = torch.zeros((), dtype=torch.bool, device=bank.device)
    old_flag, old_defer = bank.validity_buffer, bank.defer_validity
    bank.validity_buffer, bank.defer_validity = validity, True
    graph_mode = bank.execution in ("full_graph")
    report = {"rank": rank, "layer": layer.vq2a8_layer_index, "captures": 0, "replays": 0, "cases": []}
    cpu_outputs = {}

    def populate(target):
        topk_ids.fill_(target)
        hidden.copy_(original)
        validity.zero_()

    def forward():
        dispatched = dispatcher.token_dispatch(dispatch_input)
        experts, event = bank.apply(
            dispatched.hidden_states,
            dispatched.group_list,
            dispatched.group_list_type,
            getattr(config, "swiglu_limit", None),
        )
        assert event is None
        local = dispatcher.token_combine(experts, dispatched.combine_metadata)
        reduced = tensor_model_parallel_all_reduce(local.clone())
        return local, reduced, dispatched.group_list

    try:
        capture_context = nullcontext()
        if graph_mode:
            from vllm_ascend.worker.model_runner_v1 import graph_capture

            capture_context = graph_capture(bank.device)
        with capture_context as capture:
            populate(targets[0])
            if graph_mode:
                for _ in range(2):
                    forward()
                    torch.npu.synchronize()
                    assert bool(validity.cpu())
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph, stream=capture.stream):
                    local, reduced, counts = forward()
                report["captures"] = 1
            for index, target in enumerate(targets):
                populate(target)
                if graph_mode:
                    graph.replay()
                    report["replays"] += 1
                else:
                    local, reduced, counts = forward()
                torch.npu.synchronize()
                if graph_mode:
                    assert bool(validity.cpu()), "A zero-route rank must still write valid status."
                # Fused eager validates internally and has no deferred graph
                # flag contract. Its finite and empty-rank outputs are checked below.
                local_cpu, result = local.cpu(), reduced.cpu()
                assert torch.isfinite(result).all() and bool(result.count_nonzero())
                if target not in owned:
                    assert not bool(local_cpu.count_nonzero()), "A non-owner rank contributed expert output."
                observed_rows = int(counts.cpu().sum())
                assert observed_rows == (ROUTE_CAPACITY if target in owned else 0)
                cpu_outputs[str(index)] = result
                report["cases"].append(
                    {
                        "target": target,
                        "local_rows": observed_rows,
                        "output_sha256": hashlib.sha256(result.view(torch.uint8).numpy().tobytes()).hexdigest(),
                    }
                )
            torch.testing.assert_close(cpu_outputs["0"], cpu_outputs["2"], rtol=0, atol=0)
            if rank == 0:
                torch.save(cpu_outputs, output_path)
    finally:
        bank.validity_buffer, bank.defer_validity = old_flag, old_defer
    return report


def graph_state(llm, ep_size, expected_version=None):
    responses = llm.collective_rpc(GRAPH_STATE_METHOD)
    assert len(responses) == ep_size, "Every EP worker must answer the named state RPC."
    assert sorted(item["rank"] for item in responses) == list(range(ep_size)), "Missing or duplicate EP rank."
    if expected_version is not None:
        for state in responses:
            assert state["model_runner_version"] == expected_version, (
                f"Rank {state['rank']} selected {state['model_runner']}, expected model runner V{expected_version}."
            )
    return sorted(responses, key=lambda item: item["rank"])


def check_inventory(states, config, ep_size, entries):
    assert len(states) == ep_size
    assert sorted(item["rank"] for item in states) == list(range(ep_size)), "Missing or duplicate EP rank."
    logical = config["n_routed_experts"]
    assert logical % ep_size == 0
    capacity = logical // ep_size
    for state in states:
        assert state["ep_size"] == ep_size
        rank = state["rank"]
        assert sorted(layer["layer"] for layer in state["layers"]) == sorted(map(int, entries))
        for layer in state["layers"]:
            types = entries[layer["layer"]]
            expected = sorted(expert for expert in types["gate_up"] if expert // capacity == rank)
            assert layer["expert_ids"] == expected, (
                f"Incorrect resident experts on rank {rank}, layer {layer['layer']}."
            )
            if layer.get("fusion") == "full_ffn":
                full = layer["full_ffn"]
                assert full["expert_ids"] == expected
                for kind, mask in (("gate_up", "gate_w4"), ("down", "down_w4")):
                    assert full[mask] == [types[kind][expert] == "mxfp4" for expert in expected]
                shapes = full["field_shapes"]
                assert set(shapes) == {
                    "gate_weight",
                    "gate_table",
                    "gate_row_scale",
                    "down_weight",
                    "down_table",
                    "down_row_scale",
                }
                for stage, width in (("gate", 2 * config["moe_intermediate_size"]), ("down", config["hidden_size"])):
                    assert shapes[f"{stage}_row_scale"] == [len(expected), width]
                    assert len(shapes[f"{stage}_weight"]) == len(shapes[f"{stage}_table"]) == 1
                continue
            assert layer["banks"].keys() == types.keys()
            for kind, inventory in types.items():
                grouped = {q: [expert for expert in expected if inventory[expert] == q] for q in ("vq2", "mxfp4")}
                assert set(layer["banks"][kind]) == {q for q, ids in grouped.items() if ids}
                for quant_type, group in layer["banks"][kind].items():
                    ids = grouped[quant_type]
                    assert group["expert_ids"] == ids
                    fields = group["field_names"]
                    required = (
                        {"packed_zn", "pair_lut", "row_scale_bits"} if quant_type == "vq2" else {"weight", "scale"}
                    )
                    assert len(fields) == len(set(fields)) and set(fields) == required
                    assert group["field_counts"] == [len(ids)] * len(fields)


def check_graph_progress(execution, previous, current, layer_count, input_tokens, fusion="none"):
    assert [state["rank"] for state in previous] == [state["rank"] for state in current]
    for before, after in zip(previous, current, strict=True):
        assert before["layers"] == after["layers"], "Resident inventory changed during generation."
        expected_owners = 0 if execution == "fused" and fusion != "full_ffn" else layer_count
        assert before["validity_owners"] == after["validity_owners"] == expected_owners
        for phase in ("prefill", "decode", "other"):
            enabled = execution != "fused" and (phase == "decode")
            assert before[phase]["captures"] == after[phase]["captures"] == int(enabled)
            delta = after[phase]["replays"] - before[phase]["replays"]
            expected_replays = OUTPUT_TOKENS - 1 if phase == "decode" else int(input_tokens == 32)
            if enabled and expected_replays:
                assert delta >= expected_replays, f"Rank {after['rank']} did not replay its {phase} graph."
            else:
                assert delta == 0, f"Rank {after['rank']} unexpectedly replayed a {phase} graph."


def compare_topologies(actual, expected):
    assert actual["prompt_token_ids"] == expected["prompt_token_ids"]
    assert actual["token_ids"] == expected["token_ids"], "EP changed greedy output tokens."
    assert len(actual["logprobs"]) == len(expected["logprobs"]) == OUTPUT_TOKENS
    for token, observed, reference in zip(expected["token_ids"], actual["logprobs"], expected["logprobs"], strict=True):
        got, want = observed[str(token)], reference[str(token)]
        assert math.isfinite(got) and math.isfinite(want)
        assert abs(got - want) <= EP_LOGPROB_ATOL, f"EP logprob differs for token {token}: {got} vs {want}."


def check_route_reports(reports, ep_size, graph_mode):
    assert sorted(report["rank"] for report in reports) == list(range(ep_size))
    for report in reports:
        assert report["captures"] == int(graph_mode)
        assert report["replays"] == (3 if graph_mode else 0)
        assert len(report["cases"]) == 3
    for cases in zip(*(report["cases"] for report in reports), strict=True):
        assert len({case["target"] for case in cases}) == len({case["output_sha256"] for case in cases}) == 1
        assert sorted(case["local_rows"] for case in cases) == [0] * (ep_size - 1) + [ROUTE_CAPACITY]


def run_variant(model, library, binary, execution, ep_size, output, fusion="none"):
    startup = {name: os.environ.get(name) for name, _ in VARIANT_ENVIRONMENT}
    print(
        "VQ2A8_GS32_EP_STARTUP=" + json.dumps({"ep_size": ep_size, "execution": execution, "environment": startup}),
        flush=True,
    )
    from vllm import SamplingParams, envs

    from tests.e2e.conftest import VllmRunner

    config, _, entries = checkpoint_inventory(model)
    layer_count = len(entries)
    expected_version = 2 if envs.VLLM_USE_V2_MODEL_RUNNER else 1
    report = {
        "execution": execution,
        "fusion": fusion,
        "ep_size": ep_size,
        "model_runner_version": expected_version,
        "startup_environment": startup,
        "requests": [],
        "scope": "EP correctness; not performance",
    }
    with VllmRunner(str(model), **runner_options(execution, library, binary, ep_size, fusion)) as runner:
        llm = runner.model
        previous = graph_state(llm, ep_size, expected_version=expected_version)
        check_inventory(previous, config, ep_size, entries)
        report["graph_before"] = previous
        sampling = SamplingParams(
            temperature=0, max_tokens=OUTPUT_TOKENS, min_tokens=OUTPUT_TOKENS, ignore_eos=True, seed=0, logprobs=5
        )
        for prompt in build_prompts(llm.get_tokenizer()):
            results = llm.generate([{"prompt_token_ids": prompt}], sampling, use_tqdm=False)
            assert len(results) == 1 and results[0].finished and len(results[0].outputs) == 1
            completion = results[0].outputs[0]
            assert list(results[0].prompt_token_ids) == prompt
            assert len(completion.token_ids) == OUTPUT_TOKENS and completion.logprobs is not None
            assert completion.text.strip()
            current = graph_state(llm, ep_size, expected_version=expected_version)
            check_graph_progress(execution, previous, current, layer_count, len(prompt), fusion)
            report["requests"].append(
                {
                    "prompt_token_ids": list(prompt),
                    "token_ids": list(completion.token_ids),
                    "text": completion.text,
                    "logprobs": [
                        {str(token): value.logprob for token, value in row.items()} for row in completion.logprobs
                    ],
                    "graph_state": current,
                }
            )
            previous = current
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        compare_requests(report["requests"][-1], report["requests"][0])
        routes_path = output.with_suffix(".routes.pt")
        report["routes"] = llm.collective_rpc(ROUTE_METHOD, args=(str(routes_path),))
        check_route_reports(report["routes"], ep_size, execution != "fused")
        report["route_output"] = str(routes_path)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    print(
        "VQ2A8_GS32_EP="
        + json.dumps({"ep_size": ep_size, "execution": execution, "status": "PASS", "report": str(output)}),
        flush=True,
    )


def check_batch_graph_progress(execution, previous, current, layer_count, batch, *, mixed_lengths=False, fusion="none"):
    expected_bucket = str(next(size for size in BATCH_SMOKE_CAPTURE_SIZES if size >= batch))
    for before, after in zip(previous, current, strict=True):
        assert after["model_runner_version"] == 2, "The batch smoke test requires MRV2."
        assert before["rank"] == after["rank"] and before["layers"] == after["layers"]
        if execution == "fused":
            assert after["decode"] == before["decode"] == {"captures": 0, "replays": 0}
            assert after["validity_owners"] == (layer_count if fusion == "full_ffn" else 0)
            continue
        assert after["validity_owners"] == layer_count
        buckets_before, buckets_after = before["decode_buckets"], after["decode_buckets"]
        assert set(buckets_before) == set(buckets_after) == {str(size) for size in BATCH_SMOKE_CAPTURE_SIZES}
        for size in buckets_before:
            assert buckets_before[size]["captures"] == buckets_after[size]["captures"] >= 1
        delta = buckets_after[expected_bucket]["replays"] - buckets_before[expected_bucket]["replays"]
        assert delta >= (1 if mixed_lengths else OUTPUT_TOKENS - 1), (
            f"Rank {after['rank']} B{batch} did not replay the expected B{expected_bucket} decode graph."
        )
        if mixed_lengths:
            # With output budgets (2,3,4,2,3,4,2,4), three requests
            # remain after the third output token and must replay B4.
            assert buckets_after["4"]["replays"] > buckets_before["4"]["replays"], (
                f"Rank {after['rank']} did not switch to B4 as requests completed."
            )
        for phase in ("prefill", "other"):
            assert after[phase]["replays"] == before[phase]["replays"] == 0


def run_batch_variant(model, library, binary, execution, output, ep_size=BATCH_SMOKE_EP_SIZE, fusion="none"):
    """Load once per execution mode and exercise all decode buckets on EP2/EP4."""
    from vllm import SamplingParams

    from tests.e2e.conftest import VllmRunner

    config, _, entries = checkpoint_inventory(model)
    report = {"execution": execution, "fusion": fusion, "ep_size": ep_size, "waves": [], "scope": "batch correctness"}
    with VllmRunner(str(model), **batch_runner_options(execution, library, binary, ep_size, fusion)) as runner:
        llm = runner.model
        previous = graph_state(llm, ep_size, expected_version=2)
        check_inventory(previous, config, ep_size, entries)
        report["graph_before"] = previous
        seeds = [prompt[:BATCH_SMOKE_INPUT_TOKENS] for prompt in build_prompts(llm.get_tokenizer())[:2]]
        budgets = [(OUTPUT_TOKENS,) * batch for batch in BATCH_SMOKE_SIZES]
        budgets.append(BATCH_SMOKE_MIXED_OUTPUT_TOKENS)
        for wave, output_lengths in enumerate(budgets):
            batch = len(output_lengths)
            prompts = [seeds[(wave + index) % len(seeds)] for index in range(batch)]
            sampling = [
                SamplingParams(
                    temperature=0,
                    max_tokens=length,
                    min_tokens=length,
                    ignore_eos=True,
                    seed=0,
                    logprobs=5,
                )
                for length in output_lengths
            ]
            results = llm.generate([{"prompt_token_ids": prompt} for prompt in prompts], sampling, use_tqdm=False)
            assert len(results) == batch
            records = []
            for result, prompt, length in zip(results, prompts, output_lengths, strict=True):
                assert result.finished and len(result.outputs) == 1 and list(result.prompt_token_ids) == prompt
                completion = result.outputs[0]
                assert len(completion.token_ids) == length and completion.logprobs is not None
                records.append(
                    {
                        "prompt_token_ids": list(prompt),
                        "token_ids": list(completion.token_ids),
                        "logprobs": [
                            {str(token): value.logprob for token, value in row.items()} for row in completion.logprobs
                        ],
                    }
                )
            current = graph_state(llm, ep_size, expected_version=2)
            mixed_lengths = len(set(output_lengths)) != 1
            check_batch_graph_progress(
                execution, previous, current, len(entries), batch, mixed_lengths=mixed_lengths, fusion=fusion
            )
            if fusion == "full_ffn":
                check_full_ffn_evidence(current, execution)
            report["waves"].append(
                {
                    "batch": batch,
                    "output_lengths": list(output_lengths),
                    "requests": records,
                    "graph_state": current,
                }
            )
            previous = current
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            print(
                "VQ2A8_GS32_EP_BATCH_WAVE="
                + json.dumps({"execution": execution, "wave": wave, "batch": batch, "mixed_lengths": mixed_lengths}),
                flush=True,
            )


def test_gs32_ep2_decode_batch_smoke(request, tmp_path):
    """Keep the opt-in EP2 entrypoint for existing acceptance commands."""
    _check_decode_batch_smoke(request, tmp_path, ep_size=2)


def test_gs32_ep4_decode_batch_smoke(request, tmp_path):
    """Opt-in EP4 MRV2 B1..B64 smoke; absent from the one-click CI."""
    _check_decode_batch_smoke(request, tmp_path, ep_size=4)


def check_full_ffn_evidence(states, execution):
    """Python launch counts prove native submission; runner replay counters prove graphs."""
    for state in states:
        records = {item["layer"]: item for item in state["full_ffn"]}
        assert len(records) == len(state["layers"])
        for layer in state["layers"]:
            assert layer["fusion"] == "full_ffn"
            record = records[layer["layer"]]
            if layer["expert_ids"]:
                assert record["submissions"] > 0, "No full_ffn_device submission on a resident layer."
                assert (record["captures"] > 0) == (execution == "full_graph")


def test_gs32_ep4_full_ffn_decode_batch_smoke(request, tmp_path):
    """Compare staged eager, full FFN eager and full FFN B1..B64 decode graphs."""
    _check_decode_batch_smoke(
        request,
        tmp_path,
        ep_size=4,
        variants=(("none", "fused"), ("full_ffn", "fused"), ("full_ffn", "full_graph")),
    )


def _check_decode_batch_smoke(request, tmp_path, ep_size, variants=(("none", "fused"), ("none", "full_graph"))):
    values = {
        name: request.config.getoption(flag, default=None)
        for name, flag in (
            ("model", "--vq2a8-model"),
            ("gs32_library", "--vq2a8-gs32-library"),
            ("gs32_binary", "--vq2a8-gs32-binary"),
        )
    }
    if values["model"] is None:
        pytest.skip(f"Select this batch smoke explicitly with GS32 artifacts and {ep_size} visible NPUs.")
    paths = native_paths(values)
    reports = {}
    for fusion, execution in variants:
        label = f"ep{ep_size}-batch-{fusion}-{execution}"
        output, log = tmp_path / f"{label}.json", tmp_path / f"{label}.log"
        command = [
            sys.executable,
            "-u",
            "-m",
            MODULE_NAME,
            "--batch-smoke",
            "--ep-size",
            str(ep_size),
            "--execution",
            execution,
            "--fusion",
            fusion,
            "--output",
            str(output),
        ]
        for name, path in paths.items():
            command.extend([f"--{name.removeprefix('gs32_')}", str(path)])
        returncode = execute(
            command,
            {**variant_environment(os.environ), "VLLM_USE_V2_MODEL_RUNNER": "1"},
            log,
            CHILD_TIMEOUT_SECONDS,
        )
        assert returncode == 0, f"{label} failed; see {log}\n{log.read_text(encoding='utf-8')[-16000:]}"
        report = reports[fusion, execution] = json.loads(output.read_text(encoding="utf-8"))
        assert report["ep_size"] == ep_size and report["execution"] == execution
        assert report.get("fusion", "none") == fusion
        assert [wave["batch"] for wave in report["waves"]] == [
            *BATCH_SMOKE_SIZES,
            len(BATCH_SMOKE_MIXED_OUTPUT_TOKENS),
        ]
    for variant in variants[1:]:
        compare_batch_waves(reports[variant]["waves"], reports[variants[0]]["waves"])
    if ("full_ffn", "full_graph") in reports:
        compare_batch_waves(reports["full_ffn", "full_graph"]["waves"], reports["full_ffn", "fused"]["waves"])
    print(
        "VQ2A8_GS32_EP_BATCH_SMOKE="
        + json.dumps(
            {
                "status": "PASS",
                "model_runner_version": 2,
                "ep_size": ep_size,
                "capture_sizes": BATCH_SMOKE_CAPTURE_SIZES,
                "batches": BATCH_SMOKE_SIZES,
                "variants": variants,
                "mixed_output_lengths": BATCH_SMOKE_MIXED_OUTPUT_TOKENS,
                "report_dir": str(tmp_path),
                "scope": f"short EP{ep_size} batch correctness; not long-context capacity or performance",
            }
        ),
        flush=True,
    )


def compare_batch_waves(actual_waves, expected_waves):
    for actual, expected in zip(actual_waves, expected_waves, strict=True):
        assert actual["batch"] == expected["batch"] and actual["output_lengths"] == expected["output_lengths"]
        for got, want in zip(actual["requests"], expected["requests"], strict=True):
            assert got["prompt_token_ids"] == want["prompt_token_ids"]
            assert got["token_ids"] == want["token_ids"], "Decode graph padding changed greedy tokens."
            for got_row, want_row in zip(got["logprobs"], want["logprobs"], strict=True):
                assert got_row.keys() == want_row.keys(), "Decode graph changed the top-logprob token set."
                for token, reference in want_row.items():
                    assert math.isfinite(got_row[token]) and math.isfinite(reference)
                    assert math.isclose(got_row[token], reference, rel_tol=0, abs_tol=LOGPROB_ATOL)


@pytest.mark.parametrize("ep_size", EP_SIZES, ids=lambda size: f"ep{size}")
def test_gs32_ep_real_model_eager_and_graphs_match_single_card(request, tmp_path, ep_size):
    values = {
        name: request.config.getoption(flag, default=None)
        for name, flag in (
            ("model", "--vq2a8-model"),
            ("library", "--vq2a8-gs32-library"),
            ("binary", "--vq2a8-gs32-binary"),
        )
    }
    if not values["model"]:
        pytest.skip("Pass unified GS32 model and native artifacts; this opt-in test needs 2/4/8 visible NPUs.")
    assert all(values.values())
    paths = {name: Path(value).resolve(strict=True) for name, value in values.items()}
    config = json.loads((paths["model"] / "config.json").read_text())
    assert config["quantization_config"]["format"] == "vq2_mixed_gs32_source"
    assert (paths["model"] / "model.safetensors.index.json").is_file()
    skip_single_card = request.config.getoption("--vq2a8-gs32-ep-skip-single-card", default=False)
    variants = tuple((ep_size, mode) for mode in EXECUTIONS)
    if not skip_single_card:
        variants = ((1, "fused"), *variants)
    reports = {}
    # Child exit releases weights/KV/graph pools before loading the next variant.
    for size, execution in variants:
        label = f"ep{size}-{execution}"
        output, log = tmp_path / f"{label}.json", tmp_path / f"{label}.log"
        command = [
            sys.executable,
            "-u",
            "-m",
            MODULE_NAME,
            "--ep-size",
            str(size),
            "--execution",
            execution,
            "--output",
            str(output),
        ]
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
        assert child.returncode == 0, f"{label} failed; see {log}\n{log.read_text()[-16000:]}"
        reports[(size, execution)] = json.loads(output.read_text())
    eager = reports[(ep_size, "fused")]
    if not skip_single_card:
        single = reports[(1, "fused")]
        for actual, expected in zip(eager["requests"], single["requests"], strict=True):
            compare_topologies(actual, expected)
    import torch

    reference = eager if skip_single_card else reports[(1, "fused")]
    reference_routes = torch.load(reference["route_output"], map_location="cpu", weights_only=True)
    for execution in EXECUTIONS:
        report = reports[(ep_size, execution)]
        for actual, expected in zip(report["requests"], eager["requests"], strict=True):
            compare_requests(actual, expected)
        observed_routes = torch.load(report["route_output"], map_location="cpu", weights_only=True)
        assert observed_routes.keys() == reference_routes.keys()
        for key, reference in reference_routes.items():
            torch.testing.assert_close(observed_routes[key], reference, rtol=ROUTE_RTOL, atol=ROUTE_ATOL)
    summary = {
        "status": "PASS",
        "ep_size": ep_size,
        "single_card_comparison_verified": not skip_single_card,
        "single_card_reference": "skipped_by_option" if skip_single_card else "verified",
        "graph_reference": f"ep{ep_size}-fused",
        "route_reference": f"ep{ep_size}-fused" if skip_single_card else "ep1-fused",
        "reports": {f"ep{size}-{mode}": str(tmp_path / f"ep{size}-{mode}.json") for size, mode in variants},
        "scope": "EP model and graph correctness; not serving latency or throughput",
    }
    (tmp_path / f"ep{ep_size}-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("VQ2A8_GS32_EP_ACCEPTANCE=" + json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "library", "binary", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--ep-size", type=int, choices=(1, *EP_SIZES), required=True)
    parser.add_argument("--execution", choices=EXECUTIONS, required=True)
    parser.add_argument("--fusion", choices=("none", "full_ffn"), default="none")
    parser.add_argument("--batch-smoke", action="store_true")
    args = parser.parse_args()
    if args.batch_smoke:
        if args.ep_size not in BATCH_SMOKE_EP_SIZES:
            parser.error("--batch-smoke requires --ep-size 2 or 4")
        run_batch_variant(args.model, args.library, args.binary, args.execution, args.output, args.ep_size, args.fusion)
    else:
        run_variant(args.model, args.library, args.binary, args.execution, args.ep_size, args.output, args.fusion)
