# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check DSpark acceptance rejects missing hardware evidence; no NPU claim."""

import copy

import pytest

from tests.e2e.nightly import test_vq2a8_gs32_dspark as acceptance


@pytest.mark.parametrize("parallel_mode", acceptance.PARALLEL_MODES)
@pytest.mark.parametrize("draft", [None, "original-checkpoint"])
def test_dspark_acceptance_uses_b16_and_query_token_graphs(parallel_mode, draft):
    options = acceptance.runner_options(parallel_mode, draft, "stage.so", "kernel.o")
    assert options["max_num_seqs"] == 16
    assert options["tensor_parallel_size"] * options["data_parallel_size"] == 4
    assert options["tensor_parallel_size"] == (4 if parallel_mode == "tp-ep" else 1)
    assert options["compilation_config"]["cudagraph_capture_sizes"] == (
        [6, 12, 24, 48, 96] if draft else [1, 2, 4, 8, 16]
    )
    if draft:
        assert options["speculative_config"]["model"] == draft
        assert options["speculative_config"]["enforce_eager"] is False
        assert options["speculative_config"]["num_speculative_tokens"] == 5
    else:
        assert "speculative_config" not in options


def balanced_states():
    before = [
        {
            "rank": rank,
            "dp_size": 4,
            "decode": {"replays": 0},
            "prefill": {"replays": 0},
            "other": {"replays": 0},
            "verification": {"steps": 0, "draft_tokens": 0, "max_spec_requests": 0},
            "draft": {
                "proposal_calls": 0,
                "rejection_calls": 0,
                "graphs": {"replays": 0, "buckets": {"80": {"replays": 0}}},
            },
            "decode_buckets": {"96": {"replays": 0}},
        }
        for rank in range(4)
    ]
    after = copy.deepcopy(before)
    for state in after:
        state["decode"]["replays"] = 4
        state["verification"] = {"steps": 4, "draft_tokens": 320, "max_spec_requests": 16}
        state["draft"] = {
            "proposal_calls": 5,
            "rejection_calls": 4,
            "graphs": {"replays": 5, "buckets": {"80": {"replays": 4}}},
        }
        state["decode_buckets"]["96"]["replays"] = 4
    return before, after


@pytest.mark.parametrize(
    "missing",
    ["verification", "proposal", "rejection", "b16", "graph", "boundary_graph", "draft_graph", "draft_boundary"],
)
def test_acceptance_rejects_missing_speculation_or_graph_work_on_one_rank(missing):
    before, after = balanced_states()
    wave = acceptance.wave_specs("dp-ep")[0]
    acceptance.check_wave_progress(before, after, wave, enabled=True)
    state = after[3]
    if missing == "verification":
        state["verification"]["steps"] = 0
    elif missing == "proposal":
        state["draft"]["proposal_calls"] = 0
    elif missing == "rejection":
        state["draft"]["rejection_calls"] = 0
    elif missing == "b16":
        state["verification"]["max_spec_requests"] = 15
    elif missing == "graph":
        state["decode"]["replays"] = 0
    elif missing == "boundary_graph":
        state["decode_buckets"]["96"]["replays"] = 0
    elif missing == "draft_graph":
        state["draft"]["graphs"]["replays"] = 0
    else:
        state["draft"]["graphs"]["buckets"]["80"]["replays"] = 0
    with pytest.raises(AssertionError):
        acceptance.check_wave_progress(before, after, wave, enabled=True)


def test_idle_dp_rank_still_requires_graph_participation_but_not_rejection_sampling():
    before, after = balanced_states()
    wave = acceptance.wave_specs("dp-ep")[1]
    after[0]["verification"] = copy.deepcopy(before[0]["verification"])
    after[0]["draft"] = copy.deepcopy(before[0]["draft"])
    acceptance.check_wave_progress(before, after, wave, enabled=True)
    after[0]["decode"]["replays"] = 0
    with pytest.raises(AssertionError, match="target decode graph"):
        acceptance.check_wave_progress(before, after, wave, enabled=True)


def test_dspark_output_difference_fails_acceptance():
    spec = {"name": "balanced", "requests": [{"request_id": "one"}]}
    record = {
        "request_id": "one",
        "prompt_token_ids": [1],
        "token_ids": [2],
        "logprobs": [{"2": -0.1, "3": -2.0}],
    }
    reference = {"parallel_mode": "tp-ep", "waves": [{"spec": spec, "requests": [record]}]}
    actual = copy.deepcopy(reference)
    acceptance.compare_reports(actual, reference)
    actual["waves"][0]["requests"][0]["token_ids"] = [3]
    with pytest.raises(AssertionError, match="greedy output"):
        acceptance.compare_reports(actual, reference)


@pytest.mark.parametrize("records", [[], [{"request_id": "one"}, {"request_id": "one"}]])
def test_missing_or_duplicate_requests_fail_acceptance_even_if_both_runs_match(records):
    report = {
        "parallel_mode": "tp-ep",
        "waves": [{"spec": {"name": "balanced", "requests": [{"request_id": "one"}]}, "requests": records}],
    }
    with pytest.raises(AssertionError):
        acceptance.compare_reports(report, report)
