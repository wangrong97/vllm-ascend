# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks that DP acceptance cannot pass with only one engine's evidence."""

import asyncio
import copy
import json
import os
import subprocess
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.e2e.nightly import test_vq2a8_gs32_dp as acceptance
from tests.e2e.nightly import test_vq2a8_gs32_dspark as dspark_acceptance
from tests.e2e.nightly import test_vq2a8_gs32_ep as ep_acceptance
from tests.e2e.nightly.vq2a8_gs32_helpers import native_paths
from tools.tests import run_vq2a8_npu_ci as child_runner


@pytest.fixture
def acceptance_assets(tmp_path):
    values = {}
    for name in ("model", "draft_model", "gs32_library", "gs32_binary"):
        path = tmp_path / name
        if name in ("model", "draft_model"):
            path.mkdir()
        else:
            path.write_bytes(b"native artifact")
        values[name] = str(path)
    return values


def acceptance_request(values):
    options = {
        "--vq2a8-model": values.get("model"),
        "--vq2a8-dspark-model": values.get("draft_model"),
        "--vq2a8-gs32-library": values.get("gs32_library"),
        "--vq2a8-gs32-binary": values.get("gs32_binary"),
    }
    return SimpleNamespace(config=SimpleNamespace(getoption=lambda flag, default=None: options.get(flag, default)))


@pytest.mark.parametrize(
    ("name", "flag"),
    [
        ("model", "--vq2a8-model"),
        ("draft_model", "--vq2a8-dspark-model"),
        ("gs32_library", "--vq2a8-gs32-library"),
        ("gs32_binary", "--vq2a8-gs32-binary"),
    ],
)
@pytest.mark.parametrize("fault", ("missing", "wrong_type", "empty"))
def test_acceptance_asset_failure_names_the_argument(acceptance_assets, tmp_path, name, flag, fault):
    values = dict(acceptance_assets)
    if fault == "missing":
        values[name] = str(tmp_path / "does-not-exist")
    elif fault == "wrong_type":
        values[name] = values["gs32_library" if name in ("model", "draft_model") else "model"]
    else:
        values[name] = ""
    with pytest.raises(ValueError, match=flag):
        native_paths(values)


def test_dspark_explicit_empty_draft_fails_instead_of_skipping(acceptance_assets, tmp_path):
    acceptance_assets["draft_model"] = ""
    with pytest.raises(ValueError, match="--vq2a8-dspark-model requires a nonempty path"):
        dspark_acceptance.test_gs32_dspark_matches_target_and_replays_verification_graphs(
            acceptance_request(acceptance_assets), tmp_path, "dp-ep"
        )


def test_dspark_without_opt_in_still_skips(tmp_path):
    with pytest.raises(pytest.skip.Exception):
        dspark_acceptance.test_gs32_dspark_matches_target_and_replays_verification_graphs(
            acceptance_request({}), tmp_path, "dp-ep"
        )


@pytest.mark.parametrize("kind", ("dp", "dspark", "ep_batch"))
@pytest.mark.parametrize("failure", ("exit", "timeout", "interrupt"))
def test_new_acceptance_children_use_shared_cleanup_and_stop_after_failure(
    monkeypatch, acceptance_assets, tmp_path, kind, failure
):
    """Exercise the real shared executor's finally path through each entrypoint."""
    started, stopped = [], []

    def wait(timeout):
        assert timeout == acceptance.CHILD_TIMEOUT_SECONDS
        if failure == "timeout":
            raise subprocess.TimeoutExpired("model-child", timeout)
        if failure == "interrupt":
            raise KeyboardInterrupt
        return 7

    process = SimpleNamespace(wait=wait)

    def launch(command, **kwargs):
        started.append(command)
        assert kwargs["start_new_session"] == (os.name == "posix")
        assert kwargs["env"]["VLLM_USE_V2_MODEL_RUNNER"] == "1"
        assert kwargs["env"]["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
        assert kwargs["cwd"] == child_runner.REPO_ROOT
        assert "--library" in command and "--binary" in command
        kwargs["stdout"].write("injected model-child failure\n")
        kwargs["stdout"].flush()
        return process

    monkeypatch.setattr(child_runner.subprocess, "Popen", launch)
    monkeypatch.setattr(child_runner, "LiveChildLog", lambda *args: nullcontext())
    monkeypatch.setattr(child_runner, "stop_process_group", lambda child: stopped.append(child))
    error = {"exit": AssertionError, "timeout": subprocess.TimeoutExpired, "interrupt": KeyboardInterrupt}[failure]
    request = acceptance_request(acceptance_assets)
    with pytest.raises(error):
        if kind == "dp":
            acceptance.test_gs32_dp_eager_and_decode_graphs(request, tmp_path, 4)
        elif kind == "dspark":
            dspark_acceptance.test_gs32_dspark_matches_target_and_replays_verification_graphs(
                request, tmp_path, "dp-ep"
            )
        else:
            ep_acceptance.test_gs32_ep2_decode_batch_smoke(request, tmp_path)
    assert len(started) == 1, "A failed child must prevent all later model loads."
    assert stopped == [process], "Failure/cancellation must pass through shared process-group cleanup."


def states(dp_size, execution="full_graph"):
    logical_experts = 8
    topk = 2
    local_experts = logical_experts // dp_size
    config = {"n_routed_experts": logical_experts, "num_experts_per_tok": topk}
    entries = {
        0: {
            kind: dict.fromkeys(range(logical_experts), quant)
            for kind, quant in (("gate_up", "vq2"), ("down", "mxfp4"))
        }
    }
    result = []
    for rank in range(dp_size):
        ids = list(range(rank * local_experts, (rank + 1) * local_experts))
        layer = {
            "layer": 0,
            "expert_ids": ids,
            "moe_ep_rank": rank,
            "moe_ep_size": dp_size,
            "moe_local_experts": local_experts,
            "graph_row_capacity": acceptance.LOCAL_MAX_SEQS * dp_size * topk,
            "banks": {
                "gate_up": {
                    "vq2": {
                        "expert_ids": ids,
                        "field_names": ["packed_zn", "pair_lut", "row_scale_bits"],
                        "field_counts": [local_experts] * 3,
                    }
                },
                "down": {
                    "mxfp4": {
                        "expert_ids": ids,
                        "field_names": ["weight", "scale"],
                        "field_counts": [local_experts] * 2,
                    }
                },
            },
        }
        graph = execution == "full_graph"
        result.append(
            {
                "rank": rank,
                "ep_size": dp_size,
                "dp_rank": rank,
                "dp_size": dp_size,
                "tp_rank": 0,
                "tp_size": 1,
                "layers": [layer],
                "model_runner_version": 2,
                "validity_owners": int(graph),
                "forwards": {"real": 0, "dummy": 0, "max_real_requests": 0},
                "prefill": {"captures": 0, "replays": 0},
                "decode": {"captures": len(acceptance.CAPTURE_SIZES) if graph else 0, "replays": 0},
                "other": {"captures": 0, "replays": 0},
                "decode_buckets": {str(size): {"captures": 1, "replays": 0} for size in acceptance.CAPTURE_SIZES}
                if graph
                else {},
            }
        )
    return result, config, entries


def report(dp_size):
    waves = []
    for spec in acceptance.wave_specs(dp_size):
        requests = []
        for request in spec["requests"]:
            requests.append(
                {
                    "request_id": request["request_id"],
                    "dp_rank": request["dp_rank"],
                    "prompt_token_ids": [1] * request["input_tokens"],
                    "token_ids": [2] * request["output_tokens"],
                    "logprobs": [
                        {"2": -0.1, "3": -2.0, "4": -3.0, "5": -4.0, "6": -5.0} for _ in range(request["output_tokens"])
                    ],
                }
            )
        waves.append({"spec": spec, "requests": requests})
    return {"dp_size": dp_size, "waves": waves}


def tp_report(dp_report):
    reference = copy.deepcopy(dp_report)
    reference["tp_size"] = reference["dp_size"]
    reference["dp_size"] = 1
    for wave in reference["waves"]:
        for record in wave["requests"]:
            record["candidate_dp_rank"] = record.pop("dp_rank")
    return reference


@pytest.mark.parametrize("dp_size", acceptance.DP_SIZES)
@pytest.mark.parametrize("execution", acceptance.EXECUTIONS)
def test_dp_acceptance_uses_internal_single_node_mp_with_local_limits(dp_size, execution):
    options = acceptance.runner_options(execution, "stage.so", "kernel.o", dp_size)
    assert options["tensor_parallel_size"] == 1
    assert options["data_parallel_size"] == options["data_parallel_size_local"] == dp_size
    assert options["data_parallel_backend"] == options["distributed_executor_backend"] == "mp"
    assert not options["data_parallel_external_lb"] and not options["data_parallel_hybrid_lb"]
    assert options["enable_expert_parallel"] and not options["enable_eplb"]
    assert options["max_num_seqs"] == acceptance.LOCAL_MAX_SEQS
    assert not options["async_scheduling"]
    assert options["worker_extension_cls"] == acceptance.WORKER_EXTENSION
    if execution == "full_graph":
        assert options["compilation_config"]["cudagraph_capture_sizes"] == list(acceptance.CAPTURE_SIZES)
    else:
        assert options["enforce_eager"] and options["compilation_config"]["cudagraph_mode"] == "NONE"


@pytest.mark.parametrize("dp_size", (0, 1, 3, 16, True))
def test_dp_acceptance_rejects_unsupported_topology(dp_size):
    with pytest.raises(ValueError):
        acceptance.runner_options("full_graph", "stage.so", "kernel.o", dp_size)
    with pytest.raises(ValueError):
        acceptance.wave_specs(dp_size)


@pytest.mark.parametrize("dp_size", acceptance.DP_SIZES)
def test_tp_reference_uses_same_cards_but_no_dp_and_b1(dp_size):
    options = acceptance.reference_runner_options("stage.so", "kernel.o", dp_size)
    assert options["tensor_parallel_size"] == dp_size
    assert options["data_parallel_size"] == 1 and options["max_num_seqs"] == 1
    assert options["enable_expert_parallel"] and options["enforce_eager"]
    assert options["compilation_config"]["cudagraph_mode"] == "NONE"
    assert options["worker_extension_cls"] != acceptance.WORKER_EXTENSION


@pytest.mark.parametrize("dp_size", acceptance.DP_SIZES)
def test_dp_waves_cover_every_rank_idle_peers_skew_and_recovery(dp_size):
    waves = acceptance.wave_specs(dp_size)
    for rank in range(dp_size):
        assert [request["dp_rank"] for request in waves[rank]["requests"]] == [rank]
    by_name = {wave["name"]: wave for wave in waves}
    assert len(by_name["balanced"]["requests"]) == acceptance.LOCAL_MAX_SEQS * dp_size
    uneven = by_name["uneven"]["requests"]
    assert 0 not in {request["dp_rank"] for request in uneven}
    expected_counts = [0, 16] if dp_size == 2 else [0, 1, 7, 16] * (dp_size // 4)
    assert [sum(request["dp_rank"] == rank for request in uneven) for rank in range(dp_size)] == expected_counts
    assert len({request["input_tokens"] for request in uneven}) > 1
    assert len({request["output_tokens"] for request in uneven}) > 1
    assert waves[-2]["name"] == "recover-single" and waves[-1]["name"] == "recover-balanced"
    identities = [request["request_id"] for wave in waves for request in wave["requests"]]
    assert len(identities) == len(set(identities))
    assert all(
        request["input_tokens"] + request["output_tokens"] <= acceptance.MAX_MODEL_LEN
        for wave in waves
        for request in wave["requests"]
    )


@pytest.mark.parametrize("dp_size", acceptance.DP_SIZES)
@pytest.mark.parametrize("execution", acceptance.EXECUTIONS)
def test_dp_inventory_checks_real_ep_config_and_aggregate_row_capacity(dp_size, execution):
    values, config, entries = states(dp_size, execution)
    acceptance.check_state_inventory(values, config, dp_size, entries, execution)
    values[-1]["layers"][0]["moe_ep_rank"] = 0
    with pytest.raises(AssertionError):
        acceptance.check_state_inventory(values, config, dp_size, entries, execution)


@pytest.mark.parametrize("fault", [None, "perm", "unknown_field", "missing_field", "expert_dimension"])
def test_ep_dp_inventory_accepts_only_named_complete_natural_vq2_banks(fault):
    values, config, entries = states(4)
    bank = values[0]["layers"][0]["banks"]["gate_up"]["vq2"]
    if fault == "perm":
        bank["field_names"].append("perm")
        bank["field_counts"].append(len(bank["expert_ids"]))
    elif fault == "unknown_field":
        bank["field_names"][0] = "wrong_three_field_bank"
    elif fault == "missing_field":
        bank["field_names"].pop()
        bank["field_counts"].pop()
    elif fault == "expert_dimension":
        bank["field_counts"][0] += 1
    if fault is None:
        acceptance.check_state_inventory(values, config, 4, entries, "full_graph")
    else:
        with pytest.raises(AssertionError):
            acceptance.check_state_inventory(values, config, 4, entries, "full_graph")


@pytest.mark.parametrize("dp_size", [4, 8])
def test_dp_inventory_rejects_unscaled_local_graph_capacity(dp_size):
    values, config, entries = states(dp_size)
    values[-1]["layers"][0]["graph_row_capacity"] //= dp_size
    with pytest.raises(AssertionError):
        acceptance.check_state_inventory(values, config, dp_size, entries, "full_graph")


@pytest.mark.parametrize("dp_size", [4, 8])
def test_dp_rpc_requires_a_report_from_every_engine(tmp_path, dp_size):
    (tmp_path / "dp0.json").write_text(json.dumps({"dp_rank": 0}))
    with pytest.raises(AssertionError, match="every DP engine"):
        acceptance.read_states(tmp_path, dp_size)
    for rank in range(1, dp_size):
        (tmp_path / f"dp{rank}.json").write_text(json.dumps({"dp_rank": rank}))
    assert [state["dp_rank"] for state in acceptance.read_states(tmp_path, dp_size)] == list(range(dp_size))
    (tmp_path / f"dp{dp_size - 1}.json").unlink()
    with pytest.raises(AssertionError, match="every DP engine"):
        acceptance.read_states(tmp_path, dp_size)
    (tmp_path / f"dp{dp_size - 1}.json").write_text(json.dumps({"dp_rank": 0}))
    with pytest.raises(AssertionError, match="Missing/duplicate"):
        acceptance.read_states(tmp_path, dp_size)


@pytest.mark.parametrize("dp_size", [4, 8])
def test_snapshot_drains_dp_cleanup_before_counting_next_wave(tmp_path, dp_size):
    calls = []

    class FakeEngine:
        async def wait_for_requests_to_drain(self, drain_timeout):
            assert drain_timeout == acceptance.STATE_TIMEOUT_SECONDS
            calls.append("drain")

        async def collective_rpc(self, method, timeout, args):
            assert method == acceptance.STATE_METHOD and timeout == acceptance.STATE_TIMEOUT_SECONDS
            assert calls == ["drain"]
            calls.append("rpc")
            for rank in range(dp_size):
                (Path(args[0]) / f"dp{rank}.json").write_text(json.dumps({"dp_rank": rank}))
            return [{"dp_rank": 0}]  # Real internal-DP client returns engine0 only.

    values = asyncio.run(acceptance.snapshot(FakeEngine(), tmp_path / "wave", dp_size))
    assert calls == ["drain", "rpc"]
    assert [value["dp_rank"] for value in values] == list(range(dp_size))


@pytest.mark.parametrize("execution", acceptance.EXECUTIONS)
@pytest.mark.parametrize("dp_size", [4, 8])
def test_dp_progress_requires_idle_join_and_graph_replay_on_each_rank(execution, dp_size):
    before, _, _ = states(dp_size, execution)
    after = copy.deepcopy(before)
    for rank, state in enumerate(after):
        state["forwards"]["real" if rank == 0 else "dummy"] = 2
        if execution == "full_graph":
            state["decode"]["replays"] = 1
    wave = acceptance.wave_specs(dp_size)[0]
    acceptance.check_wave_progress(execution, before, after, wave)
    after[-1]["forwards"]["dummy"] = 0
    with pytest.raises(AssertionError, match=f"Idle DP{dp_size - 1}"):
        acceptance.check_wave_progress(execution, before, after, wave)
    after[-1]["forwards"]["dummy"] = 2
    after[-1]["forwards"]["real"] = 1
    with pytest.raises(AssertionError, match="wrong DP engine"):
        acceptance.check_wave_progress(execution, before, after, wave)
    after[-1]["forwards"]["real"] = 0
    if execution == "full_graph":
        after[-1]["decode"]["replays"] = 0
        with pytest.raises(AssertionError, match=f"DP{dp_size - 1} did not replay"):
            acceptance.check_wave_progress(execution, before, after, wave)


@pytest.mark.parametrize("dp_size", [4, 8])
def test_balanced_wave_requires_observed_b16_and_b16_replay_on_all_dp_ranks(dp_size):
    before, _, _ = states(dp_size)
    after = copy.deepcopy(before)
    wave = next(wave for wave in acceptance.wave_specs(dp_size) if wave["name"] == "balanced")
    for state in after:
        state["forwards"]["real"] = 2
        state["forwards"]["max_real_requests"] = acceptance.LOCAL_MAX_SEQS
        state["decode"]["replays"] = 1
        state["decode_buckets"][str(acceptance.LOCAL_MAX_SEQS)]["replays"] = 1
    acceptance.check_wave_progress("full_graph", before, after, wave)
    after[-1]["forwards"]["max_real_requests"] = 2
    with pytest.raises(AssertionError, match=f"DP{dp_size - 1} did not execute the local B16"):
        acceptance.check_wave_progress("full_graph", before, after, wave)
    after[-1]["forwards"]["max_real_requests"] = acceptance.LOCAL_MAX_SEQS
    after[-1]["decode_buckets"][str(acceptance.LOCAL_MAX_SEQS)]["replays"] = 0
    with pytest.raises(AssertionError, match=f"DP{dp_size - 1} captured B16 but did not replay"):
        acceptance.check_wave_progress("full_graph", before, after, wave)


@pytest.mark.parametrize("dp_size", [4, 8])
def test_balanced_wave_queues_all_directed_requests_before_resume(monkeypatch, dp_size):
    wave = next(wave for wave in acceptance.wave_specs(dp_size) if wave["name"] == "balanced")
    calls = []
    collectors = []
    specs = {spec["request_id"]: spec for spec in wave["requests"]}
    monkeypatch.setattr(acceptance, "sampling_params", lambda spec: spec["output_tokens"])

    class FakeCollector:
        def __init__(self, tokens, budget):
            self.closed = False
            self.tokens = tokens
            self.budget = budget

        async def get(self):
            assert calls[-1] == "resume" and len(collectors) == dp_size * acceptance.LOCAL_MAX_SEQS
            return SimpleNamespace(
                finished=True,
                prompt_token_ids=self.tokens,
                outputs=[
                    SimpleNamespace(
                        token_ids=[2] * self.budget,
                        logprobs=[
                            {token: SimpleNamespace(logprob=-float(token)) for token in range(2, 7)}
                            for _ in range(self.budget)
                        ],
                    )
                ],
            )

        def close(self):
            self.closed = True

    class FakeEngine:
        async def pause_generation(self, mode, clear_cache):
            assert mode == "keep" and not clear_cache
            calls.append("pause")

        async def add_request(self, request_id, prompt, params, data_parallel_rank):
            assert calls[0] == "pause" and "resume" not in calls
            spec = specs[request_id]
            assert data_parallel_rank == spec["dp_rank"] and params == spec["output_tokens"]
            calls.append(request_id)
            collector = FakeCollector(prompt["prompt_token_ids"], params)
            collectors.append(collector)
            return collector

        async def resume_generation(self):
            assert len(collectors) == dp_size * acceptance.LOCAL_MAX_SEQS
            calls.append("resume")

    records = asyncio.run(acceptance.generate_balanced_wave(FakeEngine(), [[10], [11]], wave))
    assert [record["request_id"] for record in records] == list(specs)
    assert all(collector.closed for collector in collectors)


def test_forward_counter_distinguishes_idle_and_real_without_device_access():
    counter = acceptance._ForwardCounter(lambda *args, **kwargs: "result")
    scheduler = SimpleNamespace(total_num_scheduled_tokens=8, num_scheduled_tokens={"a": 4, "b": 4})
    assert counter(scheduler) == "result"
    counter(scheduler, dummy_run=True)
    counter(SimpleNamespace(total_num_scheduled_tokens=0, num_scheduled_tokens={}))
    assert counter.snapshot() == {"real": 1, "dummy": 1, "max_real_requests": 2}


@pytest.mark.parametrize("dp_size", acceptance.DP_SIZES)
def test_dp_numerical_comparison_covers_every_rank(dp_size):
    expected = report(dp_size)
    actual = copy.deepcopy(expected)
    acceptance.compare_reports(actual, expected)
    # Modify a request on the last DP engine: comparing only engine0 is unsafe.
    actual["waves"][dp_size - 1]["requests"][0]["logprobs"][0]["2"] += 0.1
    with pytest.raises(AssertionError, match=f"DP{dp_size - 1} logprob mismatch"):
        acceptance.compare_reports(actual, expected)


@pytest.mark.parametrize("bad_value", (float("nan"), float("inf")))
def test_dp_numerical_comparison_rejects_nonfinite_outputs(bad_value):
    expected = report(2)
    actual = copy.deepcopy(expected)
    actual["waves"][1]["requests"][0]["logprobs"][0]["2"] = bad_value
    with pytest.raises(AssertionError):
        acceptance.compare_reports(actual, expected)


@pytest.mark.parametrize("dp_size", [4, 8])
def test_dp_numerical_comparison_rejects_missing_responses(dp_size):
    expected = report(dp_size)
    actual = copy.deepcopy(expected)
    actual["waves"][dp_size - 1]["requests"] = []
    with pytest.raises(AssertionError):
        acceptance.compare_reports(actual, expected)


@pytest.mark.parametrize("malformation", ("empty", "missing_generated", "only_four"))
def test_comparisons_reject_missing_topfive_or_generated_token(malformation):
    actual = report(4)
    row = actual["waves"][3]["requests"][0]["logprobs"][0]
    if malformation == "empty":
        row.clear()
    elif malformation == "missing_generated":
        row["9"] = row.pop("2")
    else:
        row.pop("6")
    with pytest.raises(AssertionError):
        acceptance.compare_reports(actual, copy.deepcopy(actual))
    with pytest.raises(AssertionError):
        acceptance.compare_tp_reference(actual, tp_report(actual))


@pytest.mark.parametrize("dp_size", acceptance.DP_SIZES)
def test_tp_comparison_checks_every_rank_with_existing_ep_tolerance(dp_size):
    actual = report(dp_size)
    reference = tp_report(actual)
    # Cross-TP top-five membership may change; generated-token value is checked.
    row = reference["waves"][dp_size - 1]["requests"][0]["logprobs"][0]
    row["9"] = row.pop("6")
    row["2"] += acceptance.EP_LOGPROB_ATOL / 2
    acceptance.compare_tp_reference(actual, reference)
    row["2"] += acceptance.EP_LOGPROB_ATOL
    with pytest.raises(AssertionError, match="logprob differs from TP reference"):
        acceptance.compare_tp_reference(actual, reference)


@pytest.mark.parametrize("corruption", ("missing", "misaligned", "nonfinite", "wrong_token"))
@pytest.mark.parametrize("dp_size", [4, 8])
def test_tp_comparison_rejects_incomplete_or_invalid_reference(corruption, dp_size):
    actual = report(dp_size)
    reference = tp_report(actual)
    wave = reference["waves"][dp_size - 1]
    record = wave["requests"][0]
    if corruption == "missing":
        wave["requests"].clear()
    elif corruption == "misaligned":
        record["request_id"] = "wrong-id"
    elif corruption == "nonfinite":
        record["logprobs"][0]["2"] = float("nan")
    else:
        record["token_ids"][0] = 3
    with pytest.raises(AssertionError):
        acceptance.compare_tp_reference(actual, reference)
