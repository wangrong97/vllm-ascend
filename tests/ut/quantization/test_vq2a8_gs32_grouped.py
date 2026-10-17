# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Grouped dispatch contracts with real CPU arithmetic, not NPU evidence."""

from types import SimpleNamespace

import pytest
import torch

from tests.ut.quantization.test_vq2a8_gs32_batch import (
    RecordingOps,
    dispatched_reference,
    distinct_expert_weights,
)
from tests.ut.quantization.vq2a8_gs32_reference import staged_expert_reference
from vllm_ascend.quantization.methods.w2a8.vq2a8_gs32_staged import (
    quantize_mxfp8_k32,
    require_finite,
    staged_swiglu_unchecked,
)


class CpuActivation:
    """Only a flow-test substitute for the separately tested device kernel."""

    def __init__(self):
        self.phases = []

    def prepare(self, hidden, perm, row_slots, *, raw_gate=None):
        self.phases.append("gate" if raw_gate is None else "down")
        if raw_gate is not None:
            require_finite(raw_gate, "GateUp output")
        transformed = torch.stack(
            [row.float() if perm is None else row.float()[perm[slot]] for row, slot in zip(hidden, row_slots.tolist())]
        )
        return quantize_mxfp8_k32(transformed)

    def swiglu_unchecked(self, gate, limit):
        return staged_swiglu_unchecked(gate, limit)

    def require_finite_fused(self, output, name):
        self.phases.append("final")
        require_finite(output, name)


class GroupedOps(RecordingOps):
    def __init__(self):
        super().__init__()
        self.routes = []
        self.grouped = []
        self.corrupt_columns = None

    def prepare_routes(self, hidden, counts):
        assert sum(counts.tolist()) == hidden.shape[0]
        route = SimpleNamespace(counts=counts.tolist())
        self.routes.append(route)
        return route

    def project_grouped(self, data, scales, weight, lut, row_scale, routes, output):
        self.grouped.append((routes, weight.shape[0], data.clone(), scales.clone()))
        first = 0
        for slot, count in enumerate(routes.counts):
            if count:
                super().project(
                    data[first : first + count],
                    scales[first : first + count],
                    weight[slot : slot + 1],
                    lut[slot : slot + 1],
                    row_scale[slot : slot + 1],
                    torch.tensor([count]),
                    torch.zeros(1, dtype=torch.int32),
                    self.make_tiling(data.shape[1], output.shape[1]),
                    output[first : first + count],
                )
            first += count
        if output.shape[1] == self.corrupt_columns:
            output[-1, -1] = float("nan")


def fused_bank(*, reversed_slots=False):
    bank, payloads, specs = distinct_expert_weights(GroupedOps())
    bank.execution = "fused"
    bank.activation = CpuActivation()
    if reversed_slots:
        bank.expert_slots = {2: 0, 0: 1}
        for groups in bank.banks.values():
            group = groups["vq2"]
            group["expert_ids"] = (2, 0)
            group["slots"] = {2: 0, 0: 1}
            group["fields"] = {field: value.flip(0) for field, value in group["fields"].items()}
    return bank, payloads, specs


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("group_type", [0, 1])
@pytest.mark.parametrize("counts", [(1, 0, 2), (0, 0, 3), (32, 0, 6), (128, 0, 1), (129, 0, 3), (257, 0, 1)])
def test_fused_grouped_matches_reference_in_logical_dispatch_order(reverse, group_type, counts):
    bank, payloads, specs = fused_bank(reversed_slots=reverse)
    hidden = (torch.randn(sum(counts), 64, generator=torch.Generator().manual_seed(321)) * 0.01).bfloat16()
    groups = torch.tensor(counts)
    actual, event = bank.apply(hidden, groups.cumsum(0) if group_type == 0 else groups, group_type, 0.5)
    expected = dispatched_reference(hidden, counts, payloads, specs, 0.5)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert event is None
    assert len(bank.ops.grouped) == 2
    assert len(bank.ops.routes) == 2
    assert bank.ops.grouped[0][0] is bank.ops.routes[0]
    assert bank.ops.grouped[1][0] is bank.ops.routes[1]
    assert bank.ops.routes[0].counts == ([counts[2], counts[0]] if reverse else [counts[0], counts[2]])
    assert bank.activation.phases == ["gate", "final", "final", "gate", "final"]


@pytest.mark.parametrize("rows", [512, 4097, 8192, 32769])
def test_fused_large_expert_stays_grouped_without_an_independent_row_limit(rows):
    bank, payloads, specs = fused_bank()
    hidden = (torch.randn(rows + 3, 64, generator=torch.Generator().manual_seed(52)) * 0.01).bfloat16()
    output, _ = bank.apply(hidden, torch.tensor([rows, 0, 3]), 1, 10)
    # Independently decode checkpoint weights, with matching CPU GEMM shapes
    # so BLAS shape-dependent rounding does not obscure the routing contract.
    # Native large-M versus 128-row bitwise comparisons live in the NPU tests.
    expected = torch.cat(
        [
            staged_expert_reference(
                values,
                payloads[expert, "gate_up"],
                specs["gate_up"],
                payloads[expert, "down"],
                specs["down"],
                10,
            )
            for expert, values in ((0, hidden[:rows]), (2, hidden[rows:]))
        ]
    )
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert len(bank.ops.grouped) == 2
    assert bank.ops.routes[0].counts == [rows, 3]


@pytest.mark.parametrize("rows", [32769, 65537])
@pytest.mark.parametrize("padding", [0, 7])
@pytest.mark.parametrize("single_hot_expert", [False, True])
@pytest.mark.parametrize("group_type", [0, 1])
def test_fused_dynamic_total_rows_preserve_counts_prefix_and_padding(
    monkeypatch, rows, padding, single_hot_expert, group_type
):
    bank, _, _ = fused_bank()
    bank.allow_padding = bool(padding)
    first = rows if single_hot_expert else rows // 2
    counts = [first, 0, rows - first]
    calls = []

    def grouped(hidden, actual_counts, limit):
        calls.append(actual_counts)
        assert hidden.shape[0] == rows
        return hidden.new_ones((hidden.shape[0], bank.specs["down"].rows))

    def batched(kind, hidden, jobs, **kwargs):
        pytest.fail("Actual routed rows must not trigger an implicit batched fallback.")

    # Kernel arithmetic is covered separately; observe capacity dispatch
    # for a single hot expert and an even split without a large CPU GEMM.
    monkeypatch.setattr(bank, "_apply_fused", grouped)
    monkeypatch.setattr(bank, "_project_batched", batched)
    groups = torch.tensor(counts)
    output, _ = bank.apply(
        torch.zeros(rows + padding, 64, dtype=torch.bfloat16),
        groups.cumsum(0) if group_type == 0 else groups,
        group_type,
        10,
    )
    assert output.shape == (rows + padding, 64)
    assert calls == [counts]
    assert torch.equal(output[:rows], torch.ones_like(output[:rows]))
    assert torch.equal(output[rows:], torch.zeros_like(output[rows:]))


def test_fused_empty_dispatch_does_not_prepare_routes_or_activation():
    bank, _, _ = fused_bank()
    output, _ = bank.apply(torch.empty(0, 64, dtype=torch.bfloat16), torch.zeros(3, dtype=torch.int64), 1, 10)
    assert output.shape == (0, 64)
    assert bank.ops.routes == bank.ops.grouped == bank.activation.phases == []


@pytest.mark.parametrize("group_type", [0, 1])
@pytest.mark.parametrize("counts", [(32770, 0, -1), (32770, 0, 0), (32768, 1, 0), (32768, 0, 0)])
def test_fused_large_routes_still_validate_counts_prefix_and_expert_inventory(group_type, counts):
    bank, _, _ = fused_bank()
    groups = torch.tensor(counts)
    with pytest.raises(ValueError, match="counts or routed expert inventory"):
        bank.apply(
            torch.zeros(32769, 64, dtype=torch.bfloat16),
            groups.cumsum(0) if group_type == 0 else groups,
            group_type,
            0.5,
        )
    assert bank.ops.routes == bank.ops.grouped == bank.activation.phases == []


@pytest.mark.parametrize("columns,launches", [(128, 1), (64, 2)])
def test_fused_invalid_native_output_fails_and_next_call_recovers(columns, launches):
    bank, payloads, specs = fused_bank()
    bank.ops.corrupt_columns = columns
    hidden = torch.zeros(3, 64, dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        bank.apply(hidden, torch.tensor([1, 0, 2]), 1, 10)
    assert len(bank.ops.grouped) == launches
    bank.ops.corrupt_columns = None
    output, _ = bank.apply(hidden, torch.tensor([1, 0, 2]), 1, 10)
    torch.testing.assert_close(output, dispatched_reference(hidden, (1, 0, 2), payloads, specs, 10), rtol=0, atol=0)


def test_fused_invalid_activation_does_not_launch_projection():
    bank, _, _ = fused_bank()
    hidden = torch.full((1, 64), float("nan"), dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        bank.apply(hidden, torch.tensor([1, 0, 0]), 1, 10)
    assert bank.ops.grouped == []
