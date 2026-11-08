# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full-FFN dispatch/lifetime regressions with CPU transport doubles, not NPU proof."""

from types import SimpleNamespace

import pytest
import torch

from tests.support.vq2a8_runtime_doubles import CpuActivation, CpuNative, CpuRouter, NoHostCounts
from vllm_ascend.ops.fused_moe.vq2a8_gs32 import GS32ExpertWeights


class ActivationCounter(CpuActivation):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def prepare_unchecked(self, *args, **kwargs):
        self.calls += 1
        return super().prepare_unchecked(*args, **kwargs)


class FullRouter(CpuRouter):
    def prepare_metadata(self, *args, **kwargs):
        pytest.fail("A complete FFN must use identity metadata without row vectors.")

    def restore(self, *args, **kwargs):
        pytest.fail("A complete FFN must return its native output without an extra restore.")


class FullNative(CpuNative):
    def __init__(self):
        super().__init__()
        self.plans = []
        self.invalid_scale = False
        self.outputs = []

    def prepare_full_ffn_plan(self, anchor, n, d, gate_w4, down_w4, limit, fields):
        assert len(fields) == 6
        plan = (anchor.shape[1], n, d, tuple(gate_w4.tolist()), tuple(down_w4.tolist()), limit)
        self.plans.append(plan)
        return plan

    def full_ffn_device(self, data, scales, routes, plan, middle, middle_scales, output):
        assert plan is self.plans[0]
        # An intentionally simple output labels each active physical expert.
        # This tests routing/order/padding independently of kernel arithmetic.
        output.zero_()
        middle_scales.fill_(127)
        start = 0
        for slot, count in enumerate(routes):
            output[start : start + count].fill_(slot + 1)
            start += count
        if self.invalid_scale:
            middle_scales[0, 0] = 255
        self.launches += 1
        self.outputs.append(output)


def make_bank(monkeypatch, execution="full_graph", experts=(0, 1, 2), capacity=384):
    fields = {
        name: torch.empty(0)
        for name in ("gate_weight", "gate_table", "gate_row_scale", "down_weight", "down_table", "down_row_scale")
    }
    layer = SimpleNamespace(
        vq2a8_num_experts=3,
        vq2a8_ep_size=4,
        vq2a8_anchor=torch.empty(0),
        vq2a8_full_ffn_bank={
            "expert_ids": experts,
            "fields": fields,
            "swiglu_limit": 10.0,
            "gate_w4": torch.tensor([expert == 2 for expert in experts], dtype=torch.bool),
            "down_w4": torch.tensor([expert != 0 for expert in experts], dtype=torch.bool),
        },
    )
    spec = SimpleNamespace(
        expert_ids=experts,
        specs={
            "gate_up": SimpleNamespace(columns=64, rows=128),
            "down": SimpleNamespace(columns=64, rows=64),
        },
    )
    native, activation = FullNative(), ActivationCounter()
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(is_current_stream_capturing=lambda: native.capturing), raising=False
    )
    bank = GS32ExpertWeights(
        layer,
        spec,
        {"gs32_execution": execution},
        graph_row_capacity=capacity,
        _test_ops=native,
        _test_activation=activation,
    )
    bank.device_router = FullRouter()
    return bank, native, activation


@pytest.mark.parametrize("execution", ["fused", "full_graph"])
def test_full_ffn_reuses_one_plan_and_quantization_for_changing_routes(monkeypatch, execution):
    bank, native, activation = make_bank(monkeypatch, execution)
    bank.defer_validity = True
    # Includes empty -> active experts and 128 -> 129 native tile boundary.
    for counts in ([0, 0, 0], [128, 129, 127], [0, 1, 2], [384, 0, 0]):
        groups = torch.tensor(counts).as_subclass(NoHostCounts)
        result, event = bank.apply(torch.ones(384, 64, dtype=torch.bfloat16), groups, 1, 10.0)
        expected = torch.cat([torch.full((n, 64), slot + 1) for slot, n in enumerate(counts)])
        torch.testing.assert_close(result[: sum(counts)], expected.bfloat16())
        assert not result[sum(counts) :].count_nonzero()
        assert event is None and bank.validity_buffer
        assert result is native.outputs[-1]
    assert native.plans == [(64, 128, 64, (False, False, True), (False, True, True), 10.0)]
    assert activation.calls == native.launches == bank.full_ffn_submissions == 4
    assert not bank.capture_owners


def test_full_ffn_capture_retains_scratch_and_rejects_over_capacity(monkeypatch):
    bank, native, _ = make_bank(monkeypatch, capacity=8)
    bank.defer_validity = True
    native.capturing = True
    bank.apply(torch.ones(8, 64, dtype=torch.bfloat16), torch.tensor([2, 3, 3]), 1, 10.0)
    assert bank.full_ffn_graph_captures == 1
    assert any(owner is bank.device_router.scratch_owners[0] for owner in bank.capture_owners[0])
    with pytest.raises(RuntimeError, match="at most 8"):
        bank.apply(torch.ones(9, 64, dtype=torch.bfloat16), torch.tensor([3, 3, 3]), 1, 10.0)
    assert native.launches == 1


@pytest.mark.parametrize("fault", ["nonfinite", "invalid_count", "changed_limit"])
def test_full_ffn_does_not_hide_invalid_input_or_switch_to_staged(monkeypatch, fault):
    bank, native, _ = make_bank(monkeypatch, "fused")
    native.invalid_scale = fault == "nonfinite"
    groups = torch.tensor([-1, 1, 1] if fault == "invalid_count" else [1, 1, 1])
    with pytest.raises(ValueError, match="validity|immutable"):
        bank.apply(torch.ones(3, 64, dtype=torch.bfloat16), groups, 1, 9.0 if fault == "changed_limit" else 10.0)


def test_empty_resident_bank_does_not_launch_and_still_validates_routes(monkeypatch):
    bank, native, activation = make_bank(monkeypatch, "fused", experts=())
    result, _ = bank.apply(torch.ones(8, 64, dtype=torch.bfloat16), torch.zeros(3, dtype=torch.int64), 1, 10.0)
    assert not result.count_nonzero()
    assert not native.plans and native.launches == activation.calls == 0
    with pytest.raises(ValueError, match="validity"):
        bank.apply(torch.ones(8, 64, dtype=torch.bfloat16), torch.tensor([1, 0, 0]), 1, 10.0)


@pytest.mark.parametrize("group_type", [0, 1])
def test_sparse_resident_inventory_preserves_order_and_rejects_nonresident_counts(monkeypatch, group_type):
    bank, native, _ = make_bank(monkeypatch, "fused", experts=(0, 2), capacity=8)
    hidden = torch.ones(8, 64, dtype=torch.bfloat16)
    counts = torch.tensor([2, 0, 3])
    groups = counts.cumsum(0) if group_type == 0 else counts
    result, _ = bank.apply(hidden, groups, group_type, 10.0)
    assert native.route_calls[-1] == [2, 3]
    assert bool((result[:2] == 1).all()) and bool((result[2:5] == 2).all())
    assert not result[5:].count_nonzero()
    counts = torch.tensor([2, 1, 2])
    groups = counts.cumsum(0) if group_type == 0 else counts
    with pytest.raises(ValueError, match="validity"):
        bank.apply(hidden, groups, group_type, 10.0)


@pytest.mark.parametrize("experts", [(2, 0), (0, 0), (-1, 0), (0, 3)])
def test_identity_runtime_rejects_noncanonical_inventory_before_launch(monkeypatch, experts):
    with pytest.raises(ValueError, match="order|sorted unique"):
        make_bank(monkeypatch, experts=experts)


@pytest.mark.parametrize("execution", ["fused", "full_graph"])
def test_identity_padding_ignores_nonfinite_input_and_later_calls_preserve_returned_storage(monkeypatch, execution):
    bank, native, _ = make_bank(monkeypatch, execution, experts=(0, 2), capacity=8)
    hidden = torch.full((8, 64), float("nan"), dtype=torch.bfloat16)
    hidden[:3] = 1
    first, _ = bank.apply(hidden, torch.tensor([2, 0, 1]), 1, 10.0)
    snapshot = first.clone()
    second, _ = bank.apply(torch.ones_like(hidden), torch.tensor([0, 0, 8]), 1, 10.0)
    assert first is native.outputs[0] and second is native.outputs[1]
    assert first.data_ptr() != second.data_ptr()
    torch.testing.assert_close(first, snapshot)
    assert not first[3:].count_nonzero()
    assert bool((second == 2).all())
    # The same NaN becomes invalid as soon as dynamic counts make it live.
    with pytest.raises(ValueError, match="validity"):
        bank.apply(hidden, torch.tensor([4, 0, 0]), 1, 10.0)
    recovered, _ = bank.apply(hidden, torch.tensor([0, 0, 0]), 1, 10.0)
    assert bank.validity_buffer and not recovered.count_nonzero()


def test_multiple_graph_captures_keep_distinct_workspaces_at_equal_and_different_capacities(monkeypatch):
    bank, native, _ = make_bank(monkeypatch, capacity=8)
    bank.defer_validity = True
    native.capturing = True
    outputs = []
    for rows in (8, 8, 3):
        result, _ = bank.apply(torch.ones(rows, 64, dtype=torch.bfloat16), torch.tensor([rows, 0, 0]), 1, 10.0)
        outputs.append(result)
    assert len({output.data_ptr() for output in outputs}) == 3
    for owners, output in zip(bank.capture_owners, outputs, strict=True):
        assert any(owner is output for owner in owners)
        assert any(owner is bank.validity_buffer for owner in owners)
    assert bank.full_ffn_graph_captures == 3
