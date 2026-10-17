# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU control-flow contracts for padded EP dispatch; no NPU emulation claim."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.ops.fused_moe.vq2a8_gs32 import GS32ExpertWeights, _NativeStageOps
from vllm_ascend.quantization.methods.w2a8.vq2a8_gs32_format import GS32_FIELDS


class NoHostCounts(torch.Tensor):
    def cpu(self, *args, **kwargs):
        raise AssertionError("EP graph must not copy counts/status to CPU")

    def item(self, *args, **kwargs):
        raise AssertionError("EP graph must not synchronize counts/status")

    def tolist(self, *args, **kwargs):
        raise AssertionError("EP graph must not materialize counts/status on host")


def control_bank(monkeypatch, execution="fused", *, empty=False, padding=True):
    bank = GS32ExpertWeights.__new__(GS32ExpertWeights)
    bank.execution = execution
    bank.fusion = "none"
    bank.allow_padding = padding
    bank.device = torch.device("cpu")
    bank.num_experts = 4
    bank.expert_slots = {} if empty else {0: 0, 3: 1}
    bank.resident_mask = torch.zeros(bank.num_experts, dtype=torch.bool)
    bank.resident_mask[list(bank.expert_slots)] = True
    bank.specs = {"gate_up": SimpleNamespace(columns=32), "down": SimpleNamespace(rows=32)}
    bank.ops = SimpleNamespace(check_eager=lambda: None)
    bank.validity_buffer = torch.zeros((), dtype=torch.bool)
    bank.defer_validity = execution.startswith("full_graph")
    bank.capture_owners = []
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(is_current_stream_capturing=lambda: bank.defer_validity), raising=False
    )
    return bank


@pytest.mark.parametrize("execution", ["serial", "batched", "fused"])
@pytest.mark.parametrize("group_type", [0, 1])
def test_ep_eager_bank_only_projects_valid_prefix_and_zeros_tail(monkeypatch, execution, group_type):
    bank = control_bank(monkeypatch, execution)
    hidden = torch.randn(8, 32).bfloat16()
    hidden[3:] = float("nan")  # Unowned dispatcher capacity is not activation data.
    counts = torch.tensor([1, 0, 0, 2])
    groups = counts.cumsum(0) if group_type == 0 else counts
    observed = []

    def projection(kind, values):
        observed.append(values.clone())
        assert torch.isfinite(values).all()
        if kind == "gate_up":
            return torch.cat((values, values), dim=1)
        return values

    bank._project = lambda kind, expert, values: projection(kind, values)
    bank._project_batched = lambda kind, values, jobs: projection(kind, values)
    bank._apply_fused = lambda values, counts, limit: projection("down", values)
    output, event = bank.apply(hidden, groups, group_type, 0.5)
    assert event is None and output.shape == hidden.shape
    assert observed and all(value.shape[0] <= 3 for value in observed)
    assert torch.isfinite(output).all() and torch.count_nonzero(output[3:]) == 0


@pytest.mark.parametrize("execution", ["serial", "batched", "fused", "full_graph"])
@pytest.mark.parametrize("group_type", [0, 1])
def test_ep_empty_inventory_outputs_zeros_without_native_plans(monkeypatch, execution, group_type):
    bank = control_bank(monkeypatch, execution, empty=True)
    hidden = torch.full((8, 32), float("nan"), dtype=torch.bfloat16)
    groups = torch.zeros(4, dtype=torch.int64)
    if execution.startswith("full_graph"):
        groups = groups.as_subclass(NoHostCounts)
        bank.validity_buffer = bank.validity_buffer.as_subclass(NoHostCounts)
    output, _ = bank.apply(hidden, groups, group_type, 0.5)
    assert output.shape == hidden.shape and torch.count_nonzero(output) == 0
    if execution.startswith("full_graph"):
        assert bank.validity_buffer
        groups[0] = 1
        bank.apply(hidden, groups, group_type, 0.5)
        assert not bank.validity_buffer
        groups.zero_()
        bank.apply(hidden, groups, group_type, 0.5)
        assert bank.validity_buffer


@pytest.mark.parametrize("counts", [[-1, 0, 0, 0], [9, 0, 0, 0], [0, 1, 0, 0]])
def test_ep_padded_eager_rejects_invalid_counts(monkeypatch, counts):
    bank = control_bank(monkeypatch)
    with pytest.raises(ValueError, match="counts|inventory"):
        bank.apply(torch.zeros(8, 32, dtype=torch.bfloat16), torch.tensor(counts), 1, 0.5)


def test_single_card_bank_retains_exact_count_requirement(monkeypatch):
    bank = control_bank(monkeypatch, padding=False)
    with pytest.raises(ValueError, match="counts|inventory"):
        bank.apply(torch.zeros(8, 32, dtype=torch.bfloat16), torch.tensor([1, 0, 0, 2]), 1, 0.5)


def test_empty_inventory_constructor_never_builds_zero_group_native_plan():
    layer = SimpleNamespace(vq2a8_num_experts=32, vq2a8_ep_size=8, vq2a8_anchor=torch.empty(0))
    ops = Mock()
    spec = SimpleNamespace(
        expert_ids=(),
        quant_types={"gate_up": {}, "down": {}},
        specs={"gate_up": SimpleNamespace(columns=32, rows=64), "down": SimpleNamespace(columns=32, rows=32)},
    )
    bank = GS32ExpertWeights(layer, spec, {"gs32_execution": "full_graph"}, _test_ops=ops, _test_activation=Mock())
    assert bank.allow_padding
    ops.prepare_grouped_plan.assert_not_called()
    assert bank.banks == {"gate_up": {}, "down": {}}
    assert bank.tilings == {}
    assert not bank.resident_mask.any()


def test_native_padding_is_explicit_and_old_bridge_rejected(monkeypatch):
    ops = _NativeStageOps.__new__(_NativeStageOps)
    ops.ops = SimpleNamespace(prepare_device_routes=Mock(return_value="routes"))
    a, counts = object(), object()
    assert ops.prepare_device_routes(a, counts, allow_padding=True) == "routes"
    ops.ops.prepare_device_routes.assert_called_once_with(a, counts, True)
    for schema, valid in (
        ("prepare_device_routes(Tensor anchor, Tensor counts)", False),
        ("prepare_device_routes(Tensor anchor, Tensor counts, bool allow_padding=False)", True),
    ):
        monkeypatch.setattr(
            torch._C,
            "_dispatch_find_schema_or_throw",
            lambda *args, schema=schema: SimpleNamespace(schema=lambda: schema),
        )
        if valid:
            ops.require_padding_support()
        else:
            with pytest.raises(RuntimeError, match="rebuild|rebuilding"):
                ops.require_padding_support()


@pytest.mark.parametrize("capacity", [8])
@pytest.mark.parametrize("active_rows", [0, 1, 3])
def test_ep_device_bank_passes_padding_contract_and_keeps_capacity(monkeypatch, capacity, active_rows):
    execution = "full_graph"
    bank = control_bank(monkeypatch, execution)
    bank.banks = {
        kind: {
            "vq2": {
                "expert_ids": (0, 3),
                "expert_ids_tensor": torch.tensor([0, 3]),
                "logical_slots": torch.tensor([0, -1, -1, 1]),
                "slots": {0: 0, 3: 1},
                "fields": dict.fromkeys(GS32_FIELDS, object()),
                "plan": object(),
            }
        }
        for kind in bank.specs
    }
    bank.specs["gate_up"].rows = 64
    seen = []

    def route(hidden, groups, kind, expert_ids, logical_slots, *, allow_padding, subset):
        assert allow_padding and subset and hidden.shape[0] == capacity
        physical = hidden.clone()
        physical[active_rows:] = 0
        inverse = torch.full((capacity,), -1, dtype=torch.int64)
        inverse[:active_rows] = torch.arange(active_rows)
        return (
            physical,
            torch.tensor([active_rows, 0]),
            torch.zeros(capacity, dtype=torch.int64),
            inverse,
            torch.ones(1, dtype=torch.int32),
        )

    def native_routes(hidden, counts, *, allow_padding):
        assert allow_padding
        return counts

    def project(data, scales, weight, lut, row_scale, routes, plan, output):
        assert output.shape[0] == capacity
        seen.append(output.shape)
        output.fill_(float("nan"))  # Native GMM need not initialize dispatcher padding.
        output[:active_rows] = 2

    bank.device_router = SimpleNamespace(prepare=route)
    bank.ops = SimpleNamespace(
        prepare_device_routes=native_routes,
        device_routes_status=lambda routes: torch.ones(1, dtype=torch.int32),
        project_grouped_device=project,
    )
    bank.activation = SimpleNamespace(
        prepare_unchecked=lambda value, *args, **kwargs: (value, None, torch.isfinite(value).all().int()),
        swiglu_unchecked=lambda value, limit: value[:, :32],
        finish_flags=lambda output, *flags, **kwargs: torch.stack([*flags, torch.isfinite(output).all().int()]).all(),
    )
    hidden = torch.ones(capacity, 32, dtype=torch.bfloat16)
    hidden[active_rows:] = float("nan")
    output, _ = bank.apply(hidden, torch.tensor([active_rows, 0, 0, 0]).as_subclass(NoHostCounts), 1, 0.5)
    assert output.shape == hidden.shape and len(seen) == 2
    assert (output[:active_rows] == 2).all() and (output[active_rows:] == 0).all()
    assert bank.validity_buffer and bank.capture_owners
