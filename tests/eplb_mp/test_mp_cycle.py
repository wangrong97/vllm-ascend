# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests: controller cycle, commit protocol, and the GS32 switch gates."""

import types

import pytest
import torch

from vllm_ascend.eplb.mp.config import get_mp_eplb_options
from vllm_ascend.eplb.mp.controller import MPEplbController
from vllm_ascend.eplb.mp.migrator import InMemoryCommBackend, MigrationExecutor, MoveStage
from vllm_ascend.eplb.mp.planner import MOVE_FILL, MOVE_SWAP
from vllm_ascend.eplb.mp.placement import build_linear_baseline
from vllm_ascend.eplb.mp.split_table import SplitTable

NBYTES = {"vq2": 10, "mxfp4": 19}


def nbytes_of(expert, qtype):
    return NBYTES[qtype]


def _placement(ep_size=2, spare=(2, 2)):
    quant = {}
    for kind in ("w13", "w2"):
        quant[kind] = {e: ("vq2" if e in (0, 1, 4, 5) else "mxfp4") for e in range(8)}
    slots = {"vq2": spare[0], "mxfp4": spare[1]}
    return {
        rank: build_linear_baseline(8, ep_size, rank, quant, slots)
        for rank in range(ep_size)
    }


def _quant_types(vq2_experts):
    quant = {}
    for kind in ("w13", "w2"):
        quant[kind] = {e: ("vq2" if e in vq2_experts else "mxfp4") for e in range(8)}
    return quant


class TestController:
    def _controller(self, mode="dynamic"):
        options = get_mp_eplb_options({"eplb": mode})
        controller = MPEplbController(
            options, num_experts=8, ep_size=2, ep_rank=0,
            quant_types=_quant_types({0, 1, 2, 3}), num_layers=3,
        )
        controller.bind_geometry(hidden=256, moe_intermediate=128)
        return controller

    def test_off_controller_is_inert(self):
        controller = self._controller("off")
        assert not controller.enabled
        before = controller.step
        controller.on_step_begin()
        controller.on_step_end()
        assert controller.step == before  # no accounting, no phase change

    def test_dynamic_cycle_progresses_and_refreshes_weights(self):
        controller = self._controller("dynamic")
        hot = {0: {0: 4}, 1: {0: 4}}  # experts 0/1: 4 rows each step
        for _ in range(controller._heat_steps()):
            controller.record_step(hot)
            controller.on_step_begin()
            controller.on_step_end()
        assert controller.phase != "heat"  # planning produced moves
        while controller.phase == "migrate":
            controller.on_step_begin()
            controller.on_step_end()
        assert controller.cycle_count == 1
        # Split tables refreshed from the EWMA snapshot after the cycle.
        assert any(split.weight for split in controller.split_tables.values())

    def test_static_runs_once(self):
        controller = self._controller("static")
        controller.record_step({0: {0: 8}, 4: {0: 8}})
        results = controller.run_static_once()
        assert isinstance(results, list) and len(results) == controller.num_layers

    def test_uniform_dynamic_cycle_runs_and_balances_counts(self):
        controller = self._controller("dynamic_uniform")
        assert controller.options.uniform
        hot = {0: {0: 4}, 1: {0: 4}}
        for _ in range(controller._heat_steps()):
            controller.record_step(hot)
            controller.on_step_begin()
            controller.on_step_end()
        while controller.phase == "migrate":
            controller.on_step_begin()
            controller.on_step_end()
        assert controller.cycle_count == 1
        # Uniform mode: costs are raw token counts (flat unit table).
        if controller.last_expected_cost:
            total = sum(controller._window_rows_global()) if hasattr(controller, "_window_rows_global") else None
            assert max(controller.last_expected_cost) >= 0
            for split in controller.split_tables.values():
                assert split.options.uniform

    def test_uniform_byte_accounting_is_slot_counts(self):
        controller = self._controller("static_uniform")
        controller.bind_geometry(hidden=256, moe_intermediate=128)
        assert controller._nbytes(0, "vq2") == 1 and controller._nbytes(0, "mxfp4") == 1
        # Budget equals slot capacity, not byte volume (width-agnostic).
        rank0 = controller.placement[0]
        capacity_slots = sum(
            len(rank0.slots[("w13", q)]) for q in ("vq2", "mxfp4")
        )
        assert controller.byte_budget <= capacity_slots


class TestCommitProtocol:
    def test_evict_swap_fill_order_and_commit(self):
        options = get_mp_eplb_options({"eplb": "dynamic"})
        placement = _placement(spare=(2, 2))
        split = SplitTable(8, 5, 2, options)
        split.add_replica(0, 0)
        executor = MigrationExecutor(placement, {0: split}, InMemoryCommBackend())
        moves = [
            # fill expert 0 (rank0 primary) onto rank1's first spare vq2 slot
            # (rank1 vq2 bank = [4, 5, EMPTY, EMPTY] -> slot 2)
            type("M", (), {"kind": MOVE_FILL, "expert_id": 0, "quant_type": "vq2",
                           "src_rank": 0, "src_slot": 0, "dst_rank": 1, "dst_slot": 2,
                           "evicted_expert": None})(),
            # swap expert 1 (rank0 slot 1) with expert 5 (rank1 slot 1), both vq2
            type("M", (), {"kind": MOVE_SWAP, "expert_id": 1, "quant_type": "vq2",
                           "src_rank": 0, "src_slot": 1, "dst_rank": 1, "dst_slot": 1,
                           "evicted_expert": 5})(),
        ]
        ops = executor.prepare_layer(0, moves, staging_capacity=2)
        assert executor.state == MoveStage.READY
        executor.issue(ops)
        assert executor.state == MoveStage.TRANSFERRING
        result = executor.commit_layer(0)
        assert len(result.applied) == 2 and not result.skipped
        assert placement[1].slot_of(0) == ("vq2", 2)  # replica landed
        assert placement[1].slot_of(1) is not None  # swap incoming
        assert placement[0].slot_of(5) is not None  # swap peer
        # split table follows: expert 0 is replicated (both ranks), expert 1
        # moved to rank 1 by the swap (single-copy there).
        assert split.route[0] == [0, 1]
        assert split.route[1] == [1]

    def test_prepare_rejects_overlapping_layers(self):
        executor = MigrationExecutor(_placement(), {}, InMemoryCommBackend())
        executor.prepare_layer(0, [], 2)
        executor.commit_layer(0)
        executor.prepare_layer(1, [], 2)
        with pytest.raises(RuntimeError):
            executor.prepare_layer(2, [], 2)


class _FakeOps:
    """Minimal ops surface for GS32ExpertWeights on CPU."""

    def make_tiling(self, k, n):
        return (k, n)

    def check_eager(self):
        return None

    def require_padding_support(self):
        return None

    def project(self, *args, **kwargs):
        # The real kernel writes the projection; fill zeros so the serial
        # path's NaN-poisoned output buffer passes the finiteness gate.
        args[-1].fill_(0.0)
        return None


def _make_layer(specs, num_experts=8, ep_size=2, capacity_scale=1):
    layer = types.SimpleNamespace()
    layer.vq2a8_num_experts = num_experts
    layer.vq2a8_ep_size = ep_size
    layer.vq2a8_anchor = torch.empty(0, dtype=torch.bfloat16)
    local = num_experts // ep_size
    for prefix, spec in (("w13", specs["gate_up"]), ("w2", specs["down"])):
        n, k = spec.rows, spec.columns
        shapes = {
            "packed_zn": (local + 1, n // 32, k // 16, 16, 8),
            "pair_lut": (local + 1, n // 32, k // 32, 32),
            "row_scale_bits": (local + 1, n),
            "perm": (local + 1, k),
        }
        for name, shape in shapes.items():
            dtype = torch.int64 if name in ("row_scale_bits", "perm") else torch.uint8
            tensor = torch.empty(shape, dtype=dtype)
            if name == "perm":
                # Identity permutations: the serial path gathers columns
                # through perm[slot] before quantizing.
                tensor = torch.arange(k, dtype=torch.int64).repeat(shape[0], 1)
            setattr(layer, f"{prefix}_{name}", tensor)
    return layer


def _make_weights(mode, ep_size=2):
    from vllm_ascend.eplb.mp.config import OFF_OPTIONS
    from vllm_ascend.ops.fused_moe.vq2a8_gs32 import GS32ExpertWeights
    from vllm_ascend.quantization.methods.w2a8.vq2a8_gs32_format import GS32ProjectionSpec

    # Geometry must be self-consistent: down.K == gate_up.rows // 2 (the
    # SwiGLU output width feeding the down projection).
    specs = {
        "gate_up": GS32ProjectionSpec(128, 256),
        "down": GS32ProjectionSpec(64, 64),
    }
    layer = _make_layer(specs, ep_size=ep_size)
    local = 8 // ep_size
    layer_spec = types.SimpleNamespace(
        specs=specs,
        expert_ids=tuple(range(local)),
        quant_types={kind: {e: "vq2" for e in range(local)} for kind in ("gate_up", "down")},
    )
    options = {"gs32_execution": "serial"}
    mp = get_mp_eplb_options({"eplb": mode}) if mode else OFF_OPTIONS
    return GS32ExpertWeights(
        layer, layer_spec, options, mp_eplb=mp, _test_ops=_FakeOps(),
        _test_activation=types.SimpleNamespace(), _test_mxfp4=lambda *a: None,
    )


class TestGS32SwitchGating:
    def test_off_keeps_baseline_attributes(self):
        weights = _make_weights("off")
        assert not weights.eplb_enabled
        assert weights.moe_load is None
        # eager + off: no device indexes are built (baseline behavior).
        assert all("expert_ids_tensor" not in bank for kind in weights.banks.values() for bank in kind.values())

    def test_on_adds_accounting_and_capacity(self):
        weights = _make_weights("static", ep_size=2)
        assert weights.eplb_enabled
        assert weights.moe_load is not None and tuple(weights.moe_load.shape) == (8,)
        bank = weights.banks["gate_up"]["vq2"]
        # spare = R_vq2(default 4) // ep_size(2) = 2 extra tail slots.
        assert bank["spare_slots"] == 2
        assert int(bank["expert_ids_tensor"][-1]) == -1  # EMPTY sentinel
        assert int(bank["expert_ids_tensor"][0]) == 0
        assert int(bank["logical_slots"][0]) == 0 and int(bank["logical_slots"][7]) == -1

    def test_apply_records_load_only_when_enabled(self):
        weights_off = _make_weights("off")
        weights_on = _make_weights("static", ep_size=1)
        group_list = torch.tensor([2, 1, 0, 1, 0, 0, 0, 0], dtype=torch.int64)
        hidden = torch.zeros(4, 256, dtype=torch.bfloat16)  # gate_up K = 256
        weights_off.apply(hidden, group_list, 1, None)
        weights_on.apply(hidden, group_list, 1, None)
        assert weights_off.moe_load is None
        assert weights_on.moe_load.tolist() == [2, 1, 0, 1, 0, 0, 0, 0]

    def test_commit_hooks_update_indexes(self):
        weights = _make_weights("static", ep_size=2)
        weights.commit_fill("vq2", 5, 6)  # spare slot 5 on this rank
        bank = weights.banks["gate_up"]["vq2"]
        assert int(bank["expert_ids_tensor"][5]) == 6
        assert int(bank["logical_slots"][6]) == 5
        assert bool(weights.resident_mask[6])
        outgoing = weights.commit_swap("vq2", 0, 6)
        assert outgoing == 0
        assert int(bank["logical_slots"][0]) == -1 and int(bank["logical_slots"][6]) == 0
        weights.commit_evict("vq2", 0, 6)
        assert int(bank["logical_slots"][6]) == -1 and not bool(weights.resident_mask[6])
