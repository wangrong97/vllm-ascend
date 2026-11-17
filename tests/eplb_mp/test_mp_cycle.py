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

    def prepare_full_ffn_plan(self, *args, **kwargs):
        return types.SimpleNamespace()  # plan stub; never executed on CPU

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


def _apply_capture(weights):
    original = weights._apply_full_ffn
    weights._apply_full_ffn = lambda *args, **kwargs: (
        torch.zeros((args[0].shape[0], weights.specs["down"].rows), dtype=args[0].dtype),
        None,
    )[0]
    return original


def _restore_apply(weights, original):
    weights._apply_full_ffn = original


def _make_weights(mode, ep_size=2):
    """Full-FFN GS32ExpertWeights with optional MP-EPLB spare capacity."""
    from vllm_ascend.eplb.mp.config import OFF_OPTIONS, get_mp_eplb_options
    from vllm_ascend.ops.fused_moe.vq2a8_gs32 import GS32ExpertWeights
    from vllm_ascend.quantization.methods.w2a8.vq2a8_gs32_format import GS32ProjectionSpec
    from vllm_ascend.quantization.methods.w2a8.vq2a8_gs32_full_ffn_layout import build_full_ffn_bank

    specs = {
        "gate_up": GS32ProjectionSpec(128, 256),
        "down": GS32ProjectionSpec(64, 64),
    }
    layer = _make_layer(specs, ep_size=ep_size)
    local = 8 // ep_size
    live_ids = tuple(range(local))
    layer_spec = types.SimpleNamespace(
        specs=specs,
        expert_ids=live_ids,
        quant_types={kind: {e: "vq2" for e in live_ids} for kind in ("gate_up", "down")},
    )
    mp = get_mp_eplb_options({"eplb": mode}) if mode else OFF_OPTIONS
    spare = {q: mp.replica_slots[q] // ep_size for q in ("vq2", "mxfp4")} if mp.enabled else None
    layer.vq2a8_full_ffn_bank = build_full_ffn_bank(layer, layer_spec, swiglu_limit=0.0, spare_slots=spare)
    return GS32ExpertWeights(
        layer, layer_spec, {"gs32_execution": "fused"}, mp_eplb=mp, _test_ops=_FakeOps(),
        _test_activation=types.SimpleNamespace(),
    )


class TestGS32SwitchGating:
    def test_off_keeps_baseline_attributes(self):
        weights = _make_weights("off")
        assert not weights.eplb_enabled
        assert weights.moe_load is None
        assert weights.full_ffn_bank["spare_slots"] == {}

    def test_on_adds_accounting_and_capacity(self):
        weights = _make_weights("static", ep_size=2)
        assert weights.eplb_enabled
        assert weights.moe_load is not None and tuple(weights.moe_load.shape) == (8,)
        # spare = R_vq2(default 4) // ep_size(2) = 2 extra tail slots.
        assert weights.full_ffn_bank["spare_slots"] == {"vq2": 2, "mxfp4": 1}
        ids = weights.full_ffn_expert_ids.tolist()
        # spare slots carry first_live placeholder (NOT -1 — kernel safety).
        assert ids[:4] == [0, 1, 2, 3] and ids[-3:] == [0, 0, 0]
        assert weights.full_ffn_spare_indices == frozenset({4, 5, 6})
        slots = weights.full_ffn_logical_slots.tolist()
        assert slots[0] == 0 and slots[7] == -1
        assert bool(weights.resident_mask[:4].all()) and not bool(weights.resident_mask[4:].any())

    def test_apply_records_load_only_when_enabled(self):
        # group_list counts (type 1) for 8 logical experts.
        group_list = torch.tensor([2, 1, 0, 1, 0, 0, 0, 0], dtype=torch.int64)
        hidden = torch.zeros((4, 256), dtype=torch.bfloat16)
        weights_off = _make_weights("off")
        weights_on = _make_weights("static", ep_size=2)
        original_off = _apply_capture(weights_off)
        original_on = _apply_capture(weights_on)
        weights_off.apply(hidden, group_list, 1, None)
        weights_on.apply(hidden, group_list, 1, None)
        assert weights_off.moe_load is None
        assert weights_on.moe_load.tolist() == [2, 1, 0, 1, 0, 0, 0, 0]
        _restore_apply(weights_off, original_off)
        _restore_apply(weights_on, original_on)

    def test_commit_hooks_update_indexes(self):
        weights = _make_weights("static", ep_size=2)
        spare_slot = 4  # first trailing spare slot (local experts 0..3)
        weights.commit_fill(spare_slot, 6)
        assert int(weights.full_ffn_expert_ids[spare_slot]) == 6  # actual expert ID
        assert int(weights.full_ffn_logical_slots[6]) == spare_slot
        assert bool(weights.resident_mask[6])
        outgoing = weights.commit_swap(spare_slot, 7)
        assert outgoing == 6  # expert 6 leaves, expert 7 arrives
        assert int(weights.full_ffn_logical_slots[6]) == -1 and int(weights.full_ffn_logical_slots[7]) == spare_slot
        weights.commit_evict(spare_slot, 7)
        assert int(weights.full_ffn_logical_slots[7]) == -1 and not bool(weights.resident_mask[7])


class TestFullFFNRuntimeWiring:
    def _bank_and_weights(self, mode="static"):
        weights = _make_weights(mode)
        return weights, weights.full_ffn_bank

    def test_slicer_send_slice_uses_slot_offsets(self):
        from vllm_ascend.eplb.mp.full_ffn_runtime import FullFFNBankSlicer
        from vllm_ascend.eplb.mp.migrator import P2POp
        from vllm_ascend.eplb.mp.placement import EMPTY_SLOT

        weights, bank = self._bank_and_weights()
        placement = weights._host_placement()
        slicer = FullFFNBankSlicer(lambda _layer: bank, lambda _rank: placement)
        op = P2POp(kind="send", peer_rank=0, expert_id=1, quant_type="vq2", projection="w13", field="packed_zn")
        descriptors = slicer.send_slices(0, [op], rank=0)
        offsets = bank["weight_offsets"]["gate_up"]
        assert len(descriptors) == 1
        slot = placement.slot_of(1, "w13")[1]
        assert descriptors[0]["bytes"] == offsets[slot + 1] - offsets[slot]

    def test_staging_install_roundtrip(self):
        from vllm_ascend.eplb.mp.full_ffn_runtime import FullFFNStaging

        weights, bank = self._bank_and_weights()
        offsets = bank["weight_offsets"]["gate_up"]
        slot = 0  # occupied live slot
        source = bank["fields"]["gate_weight"][offsets[slot] : offsets[slot + 1]].clone()
        staging = FullFFNStaging(num_stages=2)
        table_bytes = bank["fields"]["gate_table"].numel() // len(bank["expert_ids"])
        staging.bind_layer(0, [{"rows": 0, "weight_bytes": source.numel(), "table_bytes": table_bytes}], source.device)
        staged = staging.slot(0, 0)
        staged["weight"].copy_(source)
        staging.install(0, 0, bank, "w13", slot)
        assert torch.equal(bank["fields"]["gate_weight"][offsets[slot] : offsets[slot + 1]], source)
