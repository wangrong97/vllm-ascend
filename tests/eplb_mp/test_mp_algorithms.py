# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests: ReplicaAllocator, TCSB planner, SplitTable determinism."""

import copy

import pytest
import torch

from vllm_ascend.eplb.mp.config import get_mp_eplb_options
from vllm_ascend.eplb.mp.placement import EMPTY_SLOT, build_linear_baseline
from vllm_ascend.eplb.mp.planner import (
    MOVE_FILL,
    MOVE_SWAP,
    ReplicaAllocator,
    TCSBPlanner,
    primary_rank,
)
from vllm_ascend.eplb.mp.split_table import SplitTable, weighted_pick

NBYTES = {"vq2": 10, "mxfp4": 19}


def nbytes_of(expert, qtype):
    return NBYTES[qtype]


def _placement(ep_size=2, num_experts=8, spare=(2, 2), vq2_experts=(0, 1, 4, 5)):
    # Spread both quant types over both ranks so swaps/fills always find a
    # same-type counterpart bank on the peer rank.
    quant = {}
    for kind in ("w13", "w2"):
        quant[kind] = {e: ("vq2" if e in vq2_experts else "mxfp4") for e in range(num_experts)}
    slots = {"vq2": spare[0], "mxfp4": spare[1]}
    return {
        rank: build_linear_baseline(num_experts, ep_size, rank, quant, slots)
        for rank in range(ep_size)
    }


def _options(**over):
    raw = {"eplb": "dynamic"}
    raw.update(over.pop("_raw", {}))
    options = get_mp_eplb_options(raw)
    return options


class TestReplicaAllocator:
    def test_fill_targets_hottest_and_respects_budget(self):
        options = get_mp_eplb_options({"eplb": "static", "eplb_options": {"R_vq2": 2}})
        placement = _placement(spare=(2, 2))
        loads = {0: 100.0, 1: 50.0, **{e: 1.0 for e in range(2, 8)}}
        moves, alloc, _ = ReplicaAllocator(options).allocate(
            placement, loads, nbytes_of, byte_budget_per_rank=10**9, excluded=set()
        )
        fills = [m for m in moves if m.kind == MOVE_FILL]
        vq2_fills = [m for m in fills if m.quant_type == "vq2"]
        # rank0's vq2 experts are {0, 1}; the only copy target is rank 1.
        assert vq2_fills and all(m.expert_id in (0, 1) for m in vq2_fills)
        assert all(m.dst_rank == 1 for m in vq2_fills)
        assert vq2_fills[0].expert_id == 0  # the hottest vq2 expert first
        assert alloc[0] == 2

    def test_typed_budgets_are_independent(self):
        # Only one vq2 slot but two mxfp4 slots; a hot mxfp4 expert must not
        # be starved by cheap vq2 copies (typed-budget split, D3).
        options = get_mp_eplb_options(
            {"eplb": "static", "eplb_options": {"R_vq2": 1, "R_mx4": 2}}
        )
        placement = _placement(spare=(1, 2))
        loads = {0: 100.0, 6: 90.0}  # hot vq2 on rank0, hot mxfp4 on rank1
        moves, alloc, _ = ReplicaAllocator(options).allocate(
            placement, loads, nbytes_of, 10**9, set()
        )
        filled_types = {("vq2" if m.expert_id < 4 else "mxfp4") for m in moves if m.kind == MOVE_FILL}
        assert "mxfp4" in filled_types

    def test_evict_requires_hysteresis(self):
        options = get_mp_eplb_options(
            {"eplb": "static", "eplb_options": {"R_vq2": 2, "revoke_hysteresis": 2, "tau_revoke": 0.5}}
        )
        placement = _placement(spare=(2, 2))
        allocator = ReplicaAllocator(options)
        loads = {0: 100.0, 1: 0.0}
        # First cycle: replicate expert 0 onto rank 1 (slot 4).
        moves, alloc, history = allocator.allocate(placement, loads, nbytes_of, 10**9, set())
        filled = [m for m in moves if m.kind == MOVE_FILL]
        assert any(m.expert_id == 0 and m.dst_rank == 1 for m in filled)
        # Expert 1 went cold; per-copy load 0 < tau_revoke but primary copies
        # are never evicted (I5) — the replica of expert 0 stays hot.
        # Now cool expert 0's replica demand: per-copy load below threshold.
        loads = {0: 0.0}
        working = copy.deepcopy(placement)
        moves2, alloc2, history2 = allocator.allocate(
            working, loads, nbytes_of, 10**9, set(), alloc, history
        )
        assert not [m for m in moves2 if m.kind == "evict"]  # hysteresis not yet met
        moves3, alloc3, _ = allocator.allocate(
            copy.deepcopy(working), loads, nbytes_of, 10**9, set(), alloc2, history2
        )
        assert [m for m in moves3 if m.kind == "evict"]

    def test_primary_copies_are_never_evicted(self):
        options = get_mp_eplb_options({"eplb": "static", "eplb_options": {"R_vq2": 2}})
        placement = _placement(spare=(2, 2))
        loads = {e: 0.0 for e in range(8)}  # everything freezing cold
        moves, _, _ = ReplicaAllocator(options).allocate(placement, loads, nbytes_of, 10**9, set())
        for move in moves:
            if move.kind == "evict":
                table = placement[move.src_rank]
                assert primary_rank(table, move.expert_id) != move.src_rank


class TestTCSB:
    def test_same_type_swap_reduces_imbalance(self):
        options = get_mp_eplb_options({"eplb": "dynamic"})
        placement = _placement()  # vq2 experts {0,1} on rank0, {4,5} on rank1
        loads = {0: 100.0, 1: 100.0, **{e: 0.0 for e in range(2, 8)}}
        unit = {e: (1.0 if e in (0, 1, 4, 5) else 1.8) for e in range(8)}
        plan = TCSBPlanner(options).plan(placement, loads, unit, nbytes_of, 10**9)
        assert plan.moves and all(m.kind == MOVE_SWAP for m in plan.moves)
        vq2_set = {0, 1, 4, 5}
        for move in plan.moves:
            same_type = (move.expert_id in vq2_set) == (move.evicted_expert in vq2_set)
            assert same_type  # typed swaps only (D1)
        assert max(plan.expected_cost) < 100.0 + 1e-6  # strictly improved

    def test_budget_K_bounds_moves(self):
        options = get_mp_eplb_options({"eplb": "dynamic", "eplb_options": {"K": 1}})
        placement = _placement()
        loads = {0: 100.0, 1: 100.0, 2: 100.0, 3: 100.0}
        unit = {e: 1.0 for e in range(8)}
        plan = TCSBPlanner(options).plan(placement, loads, unit, nbytes_of, 10**9)
        assert len(plan.moves) <= 1

    def test_ep1_degenerates_to_empty_plan(self):
        options = get_mp_eplb_options({"eplb": "dynamic"})
        placement = _placement(ep_size=1, spare=(0, 0))
        loads = {0: 100.0}
        unit = {0: 1.0}
        assert TCSBPlanner(options).plan(placement, loads, unit, nbytes_of, 10**9).empty


class TestUniformMode:
    """Width-agnostic variant: one value pool, no byte density."""

    def _uniform_options(self):
        return get_mp_eplb_options({"eplb": "static_uniform", "eplb_options": {"R_vq2": 2, "R_mx4": 2}})

    def test_uniform_pool_picks_pure_value_not_byte_density(self):
        # vq2 copies are ~1.9x cheaper per byte, so the *typed* planner would
        # prefer expert 0; the uniform planner ranks by raw amortized value,
        # so the hotter mxfp4 expert (6) wins the first slot.
        options = self._uniform_options()
        assert options.uniform
        placement = _placement(spare=(2, 2))
        loads = {0: 100.0, 6: 130.0}
        moves, alloc, _ = ReplicaAllocator(options).allocate(
            placement, loads, nbytes_of, 10**9, set()
        )
        fills = [m for m in moves if m.kind == MOVE_FILL]
        assert fills and fills[0].expert_id == 6 and fills[0].quant_type == "mxfp4"
        assert alloc[6] == 2

    def test_uniform_budget_is_pooled(self):
        options = self._uniform_options()
        placement = _placement(spare=(2, 2))
        loads = {0: 100.0, 6: 100.0}  # one hot expert per type, equal value
        moves, _, _ = ReplicaAllocator(options).allocate(
            placement, loads, nbytes_of, 10**9, set()
        )
        fills = [m for m in moves if m.kind == MOVE_FILL]
        # Pool budget is 4, but C6 caps each expert at 2 copies with ep=2:
        # both hot experts get exactly one replica each.
        assert len(fills) == 2
        assert {m.quant_type for m in fills} == {"vq2", "mxfp4"}

    def test_uniform_skips_type_with_no_spare_without_blocking(self):
        options = get_mp_eplb_options({"eplb": "static_uniform", "eplb_options": {"R_vq2": 0, "R_mx4": 2}})
        placement = _placement(spare=(0, 2))
        loads = {0: 100.0, 6: 50.0}  # hot vq2 has no spare slots anywhere
        moves, _, _ = ReplicaAllocator(options).allocate(
            placement, loads, nbytes_of, 10**9, set()
        )
        fills = [m for m in moves if m.kind == MOVE_FILL]
        assert fills and all(m.quant_type == "mxfp4" for m in fills)

    def test_uniform_costs_are_flat(self):
        from vllm_ascend.eplb.mp.cost_model import CostTable, amortized_unit_cost

        table = CostTable.uniform()
        assert table.alpha_sum("vq2") == table.alpha_sum("mxfp4") == 1.0
        assert amortized_unit_cost("mxfp4", 5000, 50, table) == 1.0
        assert amortized_unit_cost("vq2", 8, 8, table) == 1.0


class TestSplitTable:
    def _table(self):
        options = get_mp_eplb_options({"eplb": "dynamic"})
        table = SplitTable(num_experts=4, capacity=4, num_ranks=2, options=options)
        table.add_replica(0, 0)
        table.add_replica(0, 1)
        table.refresh_weights({0: 80.0, 1: 40.0})
        return table

    def test_pick_is_deterministic_and_rank_consistent(self):
        table = self._table()
        tensors = table.materialize()
        picks = {
            weighted_pick(tensors["cdf"][0], tensors["route"][0], 42, 0)
            for _ in range(3)  # repeated calls on this "rank"
        }
        assert len(picks) == 1  # identical inputs -> identical picks (E1/I8)

    def test_pick_splits_roughly_by_weight(self):
        table = self._table()
        tensors = table.materialize()
        counts = {0: 0, 1: 0}
        for seq in range(4000):
            counts[weighted_pick(tensors["cdf"][0], tensors["route"][0], seq, 0)] += 1
        # weights ~ [1/3, 2/3] -> within 5 percentage points.
        assert abs(counts[0] / 4000 - 1 / 3) < 0.05

    def test_refresh_weights_respects_clip(self):
        options = get_mp_eplb_options({"eplb": "dynamic", "eplb_options": {"w_lo": 0.5, "w_hi": 2.0}})
        table = SplitTable(2, 2, 2, options)
        table.add_replica(1, 0)
        table.add_replica(1, 1)
        table.refresh_weights({0: 1e9, 1: 1e-9})  # extreme backgrounds
        raw_ratio = (1 / 1e9) / (1 / 1e-9)
        assert raw_ratio < 1e-9  # unclipped would collapse to one side
        weights = table.weight[1]
        ratio = max(weights) / min(weights)
        assert ratio <= (options.weight_hi / options.weight_lo) + 1e-9

    def test_materialize_shapes_are_fixed(self):
        table = self._table()
        tensors = table.materialize()
        assert tensors["route"].shape == (4, 4)
        assert tensors["cdf"].shape == (4, 4)
        assert float(tensors["cdf"][0][-1 % tensors["cdf"].shape[1]]) in (0.0, 1.0)
        assert int(tensors["route"][0][0]) in (0, 1)
        assert int(tensors["route"][1][0]) == -1  # single-copy expert pads with -1

    def test_ewma_never_enters_selection(self):
        # Selection depends only on route/weight; mutating rank EWMA shares
        # must not change picks until refresh_weights runs (E1).
        table = self._table()
        tensors = table.materialize()
        before = weighted_pick(tensors["cdf"][0], tensors["route"][0], 123, 0)
        table.record_rank_share(0, 999.0)
        after = weighted_pick(tensors["cdf"][0], tensors["route"][0], 123, 0)
        assert before == after
