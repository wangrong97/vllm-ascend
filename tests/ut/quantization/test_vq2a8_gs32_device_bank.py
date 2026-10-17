# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mixed source-v3 numerics with CPU transport doubles, not NPU proof."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.ops.fused_moe.vq2a8_gs32 import GS32ExpertWeights
from vllm_ascend.quantization.methods.w2a8.vq2a8_gs32_staged import (
    quantize_mxfp8_k32,
    require_finite,
    staged_swiglu,
    staged_swiglu_unchecked,
)


class NoHostCounts(torch.Tensor):
    def cpu(self, *args, **kwargs):
        raise AssertionError("Production graph counts must remain on device")

    def tolist(self, *args, **kwargs):
        raise AssertionError("Production routes must not become Python lists")

    def item(self, *args, **kwargs):
        raise AssertionError("Production validity must not become a host scalar")


def dequant_activation(data, scale):
    return data.view(torch.float8_e4m3fn).float() * torch.exp2(scale.float() - 127).repeat_interleave(32, -1)


def decode_vq(packed, lut):
    n, k = packed.shape[0] * 32, packed.shape[1] * 16
    result = torch.empty(n, k)
    for row in range(n):
        for col in range(k):
            byte = int(packed[row // 32, col // 16, col % 16, (row % 32) // 4])
            code = (byte >> (4 * ((row % 4) // 2))) & 15
            result[row, col] = lut[row // 32, col // 32].view(torch.float8_e4m3fn)[2 * code + row % 2].float()
    return result


def decode_mx(weight, scale):
    # Explicit CPU oracle uses raw source-v3 bytes, not mocked NZ tensors.
    magnitudes = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    codes = torch.stack((weight & 15, weight >> 4), dim=-1).reshape(weight.shape[0], -1).long()
    values = magnitudes[codes & 7] * torch.where(codes >= 8, -1.0, 1.0)
    return values * torch.exp2(scale.view(torch.uint8).float() - 127).repeat_interleave(32, -1)


class CpuActivation:
    def __init__(self):
        self.fusion_calls = []

    def prepare_unchecked(
        self, hidden, perm, slots, raw_gate=None, *, swiglu_limit=None, source_rows=None, route_flags=None
    ):
        padding = torch.zeros(hidden.shape[0], dtype=torch.bool)
        if source_rows is not None:
            padding = source_rows == -1
            hidden = hidden.index_select(0, source_rows.clamp_min(0))
            hidden = torch.where(padding[:, None], 0, hidden)
            if raw_gate is not None:
                raw_gate = raw_gate.index_select(0, source_rows.clamp_min(0))
                raw_gate = torch.where(padding[:, None], 0, raw_gate)
        if swiglu_limit is not None:
            self.fusion_calls.append((hidden.shape, perm is not None))
            raw_gate = hidden
            hidden = staged_swiglu_unchecked(hidden, swiglu_limit)
        transformed = (
            hidden if perm is None else torch.stack([row[perm[int(slot)]] for row, slot in zip(hidden, slots)])
        )
        valid = torch.isfinite(transformed).all(1)
        if raw_gate is not None:
            valid &= torch.isfinite(raw_gate).all(1)
        if route_flags is not None:
            valid &= (route_flags == 1).all()
        data, scale = quantize_mxfp8_k32(torch.where(valid[:, None], transformed, 0))
        scale[padding] = 127
        return data, scale, valid.int()

    def prepare(self, hidden, perm, slots, raw_gate=None, *, swiglu_limit=None):
        data, scale, flags = self.prepare_unchecked(hidden, perm, slots, raw_gate, swiglu_limit=swiglu_limit)
        assert flags.all()
        return data, scale

    swiglu_unchecked = staticmethod(staged_swiglu_unchecked)
    require_finite_fused = staticmethod(require_finite)
    finite_flags = staticmethod(lambda value: torch.isfinite(value).all().int().reshape(1))


class CpuNative:
    def __init__(self):
        self.capturing = False
        self.corrupt_width = None
        self.launches = 0
        self.route_calls = []
        self.status_calls = []
        self.output_zero_modes = []

    def check_eager(self):
        if self.capturing:
            raise RuntimeError("Unexpected eager API during capture")

    def make_tiling(self, k, n):
        return (k, n)

    def require_padding_support(self):
        pass

    def prepare_grouped_plan(self, anchor, n, groups):
        return (anchor.shape[1], n, groups)

    def prepare_routes(self, hidden, counts):
        return counts.tolist()

    def prepare_device_routes(self, hidden, counts, *, allow_padding=False):
        assert allow_padding
        routes = counts.as_subclass(torch.Tensor).tolist()
        self.route_calls.append(routes)
        return routes

    def device_routes_status(self, routes):
        self.status_calls.append(routes)
        return torch.ones(1, dtype=torch.int32)

    def project_grouped(self, data, scales, weights, lut, row_scales, routes, output):
        start = 0
        for expert, count in enumerate(routes):
            if count:
                weight = decode_vq(weights[expert], lut[expert])
                x = dequant_activation(data[start : start + count], scales[start : start + count])
                value = (x @ weight.T) * row_scales[expert].to(torch.int32).view(torch.float32)
                output[start : start + count].copy_(value.bfloat16())
            start += count
        # Deliberately poison unused capacity: production must mask it.
        output[start:] = float("nan")
        if self.corrupt_width == output.shape[1] and start:
            output[0, 0] = float("inf")
        self.launches += 1

    def project_grouped_device(self, data, scales, weights, lut, row_scales, routes, plan, output, *, zero_output=True):
        self.output_zero_modes.append(zero_output)
        self.project_grouped(data, scales, weights, lut, row_scales, routes, output)

    def project(self, data, scales, weights, lut, row_scales, counts, mask, tiling, output):
        self.project_grouped(data, scales, weights, lut, row_scales, counts.tolist(), output)


class CpuRouter:
    def __init__(self):
        self.metadata_calls = []

    def prepare_metadata(
        self, hidden, groups, group_type, experts, logical_slots, *, allow_padding=False, subset=False
    ):
        assert subset
        values = groups.as_subclass(torch.Tensor).tolist()
        counts = values if group_type == 1 else [end - start for end, start in zip(values, [0, *values[:-1]])]
        capacity = hidden.shape[0]
        valid = all(0 <= value <= capacity for value in values) and all(count >= 0 for count in counts)
        valid &= sum(counts) <= capacity if allow_padding else sum(counts) == capacity
        physical_counts = [counts[int(e)] for e in experts] if valid else [0] * len(experts)
        order, slots = [], []
        for slot, expert in enumerate(experts.tolist()):
            if valid:
                order.extend(range(sum(counts[:expert]), sum(counts[: expert + 1])))
                slots.extend([slot] * counts[expert])
        gather = torch.full((capacity,), -1, dtype=torch.int64)
        inverse = torch.full_like(gather, -1)
        if order:
            gather[: len(order)] = torch.tensor(order)
            inverse[order] = torch.arange(len(order))
        self.metadata_calls.append(tuple(experts.tolist()))
        return (
            torch.tensor(physical_counts, dtype=torch.int64),
            torch.tensor(slots + [0] * (capacity - len(slots)), dtype=torch.int64),
            gather,
            inverse,
            torch.tensor([int(valid)], dtype=torch.int32),
            torch.tensor([sum(physical_counts), int(valid)], dtype=torch.int64),
        )

    def restore(self, output, inverse, counts, flags, result, *, metadata=None):
        assert metadata is not None
        assert metadata.tolist() == [int(counts.sum()), int((flags == 1).all())]
        selected = inverse >= 0
        values = output[inverse[selected]]
        finite = torch.isfinite(values)
        result[selected] = torch.where(finite, values, 0)
        return (finite.all() & (flags == 1).all()).int().reshape(1)

    def prepare(self, hidden, groups, group_type, experts, logical_slots, *, allow_padding=False, subset=False):
        assert subset and allow_padding
        counts = groups.as_subclass(torch.Tensor)
        if group_type == 0:
            counts = counts - torch.cat((counts.new_zeros(1), counts[:-1]))
        counts = counts.tolist()
        valid = all(0 <= count <= 128 for count in counts) and sum(counts) <= hidden.shape[0]
        physical_counts = [counts[int(e)] for e in experts] if valid else [0] * len(experts)
        order, slots = [], []
        for slot, expert in enumerate(experts.tolist()):
            if valid:
                order.extend(range(sum(counts[:expert]), sum(counts[: expert + 1])))
                slots.extend([slot] * counts[expert])
        physical = torch.zeros_like(hidden)
        if order:
            physical[: len(order)] = hidden[order]
        inverse = torch.full((hidden.shape[0],), -1, dtype=torch.int64)
        if order:
            inverse[order] = torch.arange(len(order))
        return (
            physical,
            torch.tensor(physical_counts),
            torch.tensor(slots + [0] * (hidden.shape[0] - len(slots))),
            inverse,
            torch.tensor([int(valid)]),
        )


def mixed_bank(monkeypatch, execution="full_graph", *, experts=(0, 1, 2, 3), ep=False, homogeneous=None, fusion="none"):
    kinds = {"gate_up": ("vq2", "vq2", "mxfp4", "mxfp4"), "down": ("vq2", "mxfp4", "vq2", "mxfp4")}
    if homogeneous:
        kinds = {kind: (homogeneous,) * 4 for kind in kinds}
    specs = {"gate_up": SimpleNamespace(columns=64, rows=128), "down": SimpleNamespace(columns=64, rows=64)}
    quant_types = {kind: {e: types[e] for e in experts} for kind, types in kinds.items()}
    layer = SimpleNamespace(vq2a8_num_experts=4, vq2a8_ep_size=2 if ep else 1, vq2a8_anchor=torch.empty(0))
    reference = {}
    generator = torch.Generator().manual_seed(347)
    for kind, prefix in (("gate_up", "w13"), ("down", "w2")):
        n, k = specs[kind].rows, specs[kind].columns
        for quant_type in ("vq2", "mxfp4"):
            selected = sorted(e for e in experts if quant_types[kind][e] == quant_type)
            if not selected:
                continue
            records = []
            for expert in selected:
                if quant_type == "vq2":
                    packed = torch.randint(256, (n // 32, k // 16, 16, 8), dtype=torch.uint8, generator=generator)
                    lut = (
                        ((torch.arange(n // 32 * k // 32 * 32).reshape(n // 32, k // 32, 32) % 15 - 7) / 64)
                        .to(torch.float8_e4m3fn)
                        .view(torch.uint8)
                    )
                    row_scale = torch.full((n,), 0.5 + expert / 8).view(torch.int32).to(torch.int64)
                    perm = torch.roll(torch.arange(k), expert + 1)
                    fields = dict(packed_zn=packed, pair_lut=lut, row_scale_bits=row_scale, perm=perm)
                    weight = decode_vq(packed, lut) * row_scale.to(torch.int32).view(torch.float32)[:, None]
                else:
                    weight_bytes = torch.randint(256, (n, k // 2), dtype=torch.uint8, generator=generator)
                    scale = torch.full((n, k // 32), 123 + expert % 2, dtype=torch.uint8)
                    fields = dict(weight=weight_bytes, scale=scale)
                    weight, perm = decode_mx(weight_bytes, scale), None
                reference[kind, expert] = (weight, perm)
                records.append(fields)
            for field in records[0]:
                name = f"{prefix}_{field}" if quant_type == "vq2" else f"{prefix}_mxfp4_{field}"
                setattr(layer, name, torch.stack([record[field] for record in records]))
    native = CpuNative()

    def mxfp4(data, scales, fields, counts):
        output = torch.full((data.shape[0], fields["weight"].shape[1]), float("nan"), dtype=torch.bfloat16)
        start = 0
        for slot, count in enumerate(counts.as_subclass(torch.Tensor).tolist()):
            if count:
                x = dequant_activation(data[start : start + count], scales[start : start + count])
                output[start : start + count] = (
                    x @ decode_mx(fields["weight"][slot], fields["scale"][slot]).T
                ).bfloat16()
            start += count
        return output

    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(is_current_stream_capturing=lambda: native.capturing), raising=False
    )
    bank = GS32ExpertWeights(
        layer,
        SimpleNamespace(expert_ids=experts, specs=specs, quant_types=quant_types),
        {"gs32_execution": execution, "gs32_fusion": fusion},
        _test_ops=native,
        _test_activation=CpuActivation(),
        _test_mxfp4=mxfp4,
    )
    if execution in ("full_graph"):
        bank.device_router = CpuRouter()
    return bank, reference


def reference_ffn(hidden, counts, weights, limit):
    output = torch.zeros(hidden.shape[0], 64, dtype=torch.bfloat16)
    offset = 0
    for expert, count in enumerate(counts):
        if not count:
            continue
        x = hidden[offset : offset + count]
        for kind in ("gate_up", "down"):
            weight, perm = weights[kind, expert]
            data, scales = quantize_mxfp8_k32(x if perm is None else x[:, perm])
            x = (dequant_activation(data, scales) @ weight.T).bfloat16()
            if kind == "gate_up":
                x = staged_swiglu(x, limit)
        output[offset : offset + count] = x
        offset += count
    return output


@pytest.mark.parametrize("execution", ["serial", "batched", "fused", "full_graph"])
@pytest.mark.parametrize("group_type", [0, 1])
@pytest.mark.parametrize("fusion", ["none", "swiglu_quant"])
@pytest.mark.parametrize("limit", [0.0, 0.5])
def test_all_four_mixed_projection_pairs_match_oracle(monkeypatch, execution, group_type, fusion, limit):
    bank, weights = mixed_bank(monkeypatch, execution, fusion=fusion)
    hidden = (torch.randn(8, 64, generator=torch.Generator().manual_seed(23)) / 8).bfloat16()
    counts = [1, 2, 3, 2]
    groups = torch.tensor(counts)
    if group_type == 0:
        groups = groups.cumsum(0)
    output, _ = bank.apply(hidden, groups, group_type, limit)
    torch.testing.assert_close(output, reference_ffn(hidden, counts, weights, limit), rtol=0, atol=0)
    assert bool(bank.activation.fusion_calls) == (fusion == "swiglu_quant")
    if fusion == "swiglu_quant":
        assert {has_perm for _, has_perm in bank.activation.fusion_calls} == {False, True}
        assert all(shape[1] == 128 for shape, _ in bank.activation.fusion_calls)


@pytest.mark.parametrize("homogeneous", ["vq2", "mxfp4"])
def test_static_empty_type_bank_and_changed_routes(monkeypatch, homogeneous):
    bank, weights = mixed_bank(monkeypatch, homogeneous=homogeneous)
    hidden = torch.randn(8, 64).bfloat16() / 16
    for counts in ([8, 0, 0, 0], [0, 0, 0, 8], [1, 3, 2, 2]):
        output, _ = bank.apply(hidden, torch.tensor(counts).as_subclass(NoHostCounts), 1, 0.5)
        torch.testing.assert_close(output, reference_ffn(hidden, counts, weights, 0.5), rtol=0, atol=0)
        assert bank.validity_buffer


@pytest.mark.parametrize("fusion", ["none", "swiglu_quant"])
def test_mixed_device_route_keeps_original_source_offsets_and_masks_tail(monkeypatch, fusion):
    bank, weights = mixed_bank(monkeypatch, ep=True, fusion=fusion)
    hidden = torch.randn(8, 64).bfloat16() / 16
    for counts in ([1, 2, 0, 3], [0, 0, 5, 0], [0, 0, 0, 0]):
        hidden[sum(counts) :] = float("nan")
        output, _ = bank.apply(hidden, torch.tensor(counts).as_subclass(NoHostCounts), 1, 0.5)
        torch.testing.assert_close(output, reference_ffn(hidden, counts, weights, 0.5), rtol=0, atol=0)
        assert bank.validity_buffer and torch.isfinite(output).all()
        hidden[:] = 0.125


@pytest.mark.parametrize("fusion", ["none", "swiglu_quant"])
def test_empty_ep_rank_keeps_zero_result_and_rejects_nonzero_routes(monkeypatch, fusion):
    bank, _ = mixed_bank(monkeypatch, experts=(), ep=True, fusion=fusion)
    output, _ = bank.apply(torch.ones(8, 64, dtype=torch.bfloat16), torch.zeros(4, dtype=torch.int64), 1, 0.5)
    assert torch.equal(output, torch.zeros_like(output)) and bank.validity_buffer
    with pytest.raises(ValueError, match="validity"):
        bank.apply(torch.ones(8, 64, dtype=torch.bfloat16), torch.tensor([1, 0, 0, 0]), 1, 0.5)


@pytest.mark.parametrize("fusion", ["none", "swiglu_quant"])
def test_capture_requires_owner_and_keeps_all_recapture_buffers(monkeypatch, fusion):
    bank, _ = mixed_bank(monkeypatch, fusion=fusion)
    bank.ops.capturing = True
    hidden, counts = torch.ones(8, 64, dtype=torch.bfloat16), torch.tensor([2, 2, 2, 2])
    with pytest.raises(RuntimeError, match="owner"):
        bank.apply(hidden, counts, 1, 0.5)
    bank.defer_validity = True
    bank.validity_buffer = bank.validity_buffer.as_subclass(NoHostCounts)
    for _ in range(2):
        bank.apply(hidden, counts.as_subclass(NoHostCounts), 1, 0.5)
    assert len(bank.capture_owners) == 2


@pytest.mark.parametrize("fusion", ["none", "swiglu_quant"])
def test_raw_gate_invalidity_survives_clipping_and_next_replay_recovers(monkeypatch, fusion):
    bank, _ = mixed_bank(monkeypatch, fusion=fusion)
    bank.defer_validity = True
    for corrupt in (128, 64, None):
        bank.ops.corrupt_width = corrupt
        bank.apply(torch.ones(8, 64, dtype=torch.bfloat16), torch.tensor([2, 2, 2, 2]), 1, 0.5)
        assert bool(bank.validity_buffer) == (corrupt is None)


def test_fused_batched_large_interleaved_jobs_keep_expert_rows(monkeypatch):
    bank, weights = mixed_bank(monkeypatch, "batched", fusion="swiglu_quant")
    counts = [129, 5, 130, 7]
    hidden = (torch.randn(sum(counts), 64, generator=torch.Generator().manual_seed(41)) / 16).bfloat16()
    output, _ = bank.apply(hidden, torch.tensor(counts), 1, 0.5)
    torch.testing.assert_close(output, reference_ffn(hidden, counts, weights, 0.5), rtol=0, atol=0)
    assert len(bank.activation.fusion_calls) == 2


@pytest.mark.parametrize("execution", ["fused", "full_graph"])
@pytest.mark.parametrize("fusion", ["none", "swiglu_quant"])
@pytest.mark.parametrize("group_type", [0, 1])
def test_large_grouped_mixed_projections_preserve_experts_without_serial_fallback(
    monkeypatch, execution, fusion, group_type
):
    bank, weights = mixed_bank(monkeypatch, execution, fusion=fusion)
    counts = [4097, 5, 8192, 7]
    hidden = (torch.randn(sum(counts), 64, generator=torch.Generator().manual_seed(42)) / 16).bfloat16()

    def unexpected_batched(*args, **kwargs):
        pytest.fail("An expert above 4096 rows forced the mixed layer out of grouped execution.")

    monkeypatch.setattr(bank, "_project_batched", unexpected_batched)
    groups = torch.tensor(counts)
    output, _ = bank.apply(hidden, groups.cumsum(0) if group_type == 0 else groups, group_type, 0.5)
    torch.testing.assert_close(output, reference_ffn(hidden, counts, weights, 0.5), rtol=0, atol=0)


def test_dynamic_grouped_total_rows_preserve_all_mixed_projection_pairs(monkeypatch):
    bank, weights = mixed_bank(monkeypatch, "full_graph", fusion="swiglu_quant", ep=True)
    counts, padding = [32769, 5, 2, 7], 3
    rows = sum(counts)
    hidden = (torch.randn(rows + padding, 64, generator=torch.Generator().manual_seed(42)) / 16).bfloat16()

    def unexpected_batched(*args, **kwargs):
        pytest.fail("Aggregate routed rows must not force mixed quantization out of grouped execution.")

    monkeypatch.setattr(bank, "_project_batched", unexpected_batched)
    output, _ = bank.apply(hidden, torch.tensor(counts).cumsum(0), 0, 0.5)
    expected = reference_ffn(hidden[:rows], counts, weights, 0.5)
    torch.testing.assert_close(output[:rows], expected, rtol=0, atol=0)
    assert torch.equal(output[rows:], torch.zeros_like(output[rows:]))


@pytest.mark.parametrize("fusion", ["none", "swiglu_quant"])
@pytest.mark.parametrize("group_type", [0, 1])
def test_prefill_device_metadata_defers_checks_and_reuses_only_identical_banks(monkeypatch, fusion, group_type):
    bank, weights = mixed_bank(monkeypatch, ep=True, homogeneous="vq2", fusion=fusion)
    bank.defer_validity = True
    bank.validity_buffer = bank.validity_buffer.as_subclass(NoHostCounts)
    hidden = torch.full((257, 64), 0.125, dtype=torch.bfloat16)
    for counts in ([129, 0, 1, 126], [0, 0, 257, 0], [0, 0, 0, 0]):
        hidden[:] = 0.125
        hidden[sum(counts) :] = float("nan")
        groups = torch.tensor(counts)
        if group_type == 0:
            groups = groups.cumsum(0)
        bank.device_router.metadata_calls.clear()
        bank.ops.route_calls.clear()
        bank.ops.status_calls.clear()
        bank.ops.output_zero_modes.clear()
        output, _ = bank.apply(hidden, groups.as_subclass(NoHostCounts), group_type, 0.5)
        torch.testing.assert_close(output, reference_ffn(hidden, counts, weights, 0.5), rtol=0, atol=0)
        assert bank.validity_buffer and bank.capture_owners == []
        assert bank.device_router.metadata_calls == [(0, 1, 2, 3)]
        assert bank.ops.route_calls == [counts]
        assert bank.ops.status_calls == bank.ops.route_calls
        assert bank.ops.output_zero_modes == [False, False]


@pytest.mark.parametrize("fusion", ["none", "swiglu_quant"])
def test_prefill_mixed_banks_have_independent_metadata_and_preserve_raw_failure(monkeypatch, fusion):
    bank, weights = mixed_bank(monkeypatch, ep=True, fusion=fusion)
    bank.defer_validity = True
    bank.validity_buffer = bank.validity_buffer.as_subclass(NoHostCounts)
    counts = [129, 5, 7, 1]
    hidden = torch.full((257, 64), 0.125, dtype=torch.bfloat16)
    hidden[sum(counts) :] = float("nan")
    for corrupt in (128, 64, None):
        bank.ops.corrupt_width = corrupt
        bank.device_router.metadata_calls.clear()
        bank.ops.route_calls.clear()
        bank.ops.status_calls.clear()
        output, _ = bank.apply(hidden, torch.tensor(counts).as_subclass(NoHostCounts), 1, 0.5)
        assert bool(bank.validity_buffer) == (corrupt is None)
        assert bank.device_router.metadata_calls == [(0, 1), (2, 3), (0, 2), (1, 3)]
        assert bank.ops.route_calls == [[129, 5], [129, 7]]
        assert bank.ops.status_calls == bank.ops.route_calls
        assert torch.isfinite(output).all() and not output[sum(counts) :].any()
    torch.testing.assert_close(output, reference_ffn(hidden, counts, weights, 0.5), rtol=0, atol=0)


@pytest.mark.parametrize("homogeneous", [None, "vq2", "mxfp4"])
def test_prefill_down_preparation_rejects_nonfinite_middle_without_an_extra_scan(monkeypatch, homogeneous):
    bank, _ = mixed_bank(monkeypatch, ep=True, homogeneous=homogeneous)
    bank.defer_validity = True
    hidden = torch.full((257, 64), 0.125, dtype=torch.bfloat16)
    counts = torch.tensor([129, 5, 7, 1]).as_subclass(NoHostCounts)
    original = bank.activation.swiglu_unchecked

    def unexpected_scan(value):
        pytest.fail("Prefill must reuse Down preparation validity instead of rescanning all BF16 values")

    monkeypatch.setattr(bank.activation, "finite_flags", unexpected_scan)
    for invalid in (True, False):

        def swiglu(value, limit, invalid=invalid):
            middle = original(value, limit)
            if invalid:
                middle[0, 0] = float("nan")
            return middle

        monkeypatch.setattr(bank.activation, "swiglu_unchecked", swiglu)
        bank.apply(hidden, counts, 1, 0.5)
        assert bool(bank.validity_buffer) == (not invalid)


@pytest.mark.parametrize(
    "groups,group_type",
    [
        ([-1, 0, 0, 0], 1),
        ([258, 0, 0, 0], 1),
        ([129, 129, 0, 0], 1),
        ([2**63 - 1, 2**63 - 1, 2, 0], 1),
        ([257, 256, 257, 257], 0),
    ],
)
def test_prefill_invalid_device_metadata_flags_and_recovers(monkeypatch, groups, group_type):
    bank, _ = mixed_bank(monkeypatch, ep=True)
    bank.defer_validity = True
    hidden = torch.full((257, 64), 0.125, dtype=torch.bfloat16)
    output, _ = bank.apply(hidden, torch.tensor(groups).as_subclass(NoHostCounts), group_type, 0.5)
    assert not bank.validity_buffer and not output.any()
    bank.apply(hidden, torch.tensor([257, 0, 0, 0]).as_subclass(NoHostCounts), 1, 0.5)
    assert bank.validity_buffer


def test_prefill_nonresident_routes_and_zero_capacity_keep_validity(monkeypatch):
    bank, _ = mixed_bank(monkeypatch, experts=(0,), ep=True)
    bank.defer_validity = True
    hidden = torch.ones(257, 64, dtype=torch.bfloat16)
    output, _ = bank.apply(hidden, torch.tensor([0, 257, 0, 0]).as_subclass(NoHostCounts), 1, 0.5)
    assert not bank.validity_buffer and not output.any()
    for groups, valid in (([0, 0, 0, 0], True), ([1, 0, 0, 0], False)):
        output, _ = bank.apply(hidden[:0], torch.tensor(groups).as_subclass(NoHostCounts), 1, 0.5)
        assert output.shape == (0, 64) and bool(bank.validity_buffer) == valid


@pytest.mark.parametrize("expert_shape", [(), (3,)])
@pytest.mark.parametrize("group_size", [16, 32, 64])
@pytest.mark.parametrize("scale_dtype", [torch.uint8, torch.float8_e8m0fnu])
def test_mxfp4_load_helper_preserves_e8_bytes_and_transposed_stride(expert_shape, group_size, scale_dtype):
    # Execute the real helper with only NZ cast replaced. This verifies its
    # Python storage contract, not the hardware-specific NZ conversion.
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/quantization/methods/w4a8/w4a8_mxfp4.py"
    module = ast.parse(source.read_text())
    helper = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "prepare_mxfp4_projection"
    )
    npu = SimpleNamespace(
        npu_format_cast=Mock(side_effect=lambda value, *args, **kwargs: value), float4_e2m1fn_x2=object()
    )
    namespace = {"torch": torch, "torch_npu": npu}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(source), "exec"), namespace)
    weight = torch.arange(4 * 128, dtype=torch.int32).to(torch.uint8).reshape(4, 128)
    weight = weight.expand(*expert_shape, 4, 128).contiguous()
    raw_scale = torch.arange(weight.numel() * 2 // group_size, dtype=torch.int32).to(torch.uint8)
    raw_scale = raw_scale.reshape(*expert_shape, 4, 256 // group_size)
    loaded_weight, loaded_scale = namespace["prepare_mxfp4_projection"](
        weight, raw_scale.view(scale_dtype), group_size=group_size
    )
    npu.npu_format_cast.assert_called_once()
    assert npu.npu_format_cast.call_args.args[0] is weight
    assert npu.npu_format_cast.call_args.args[1] == 29
    assert npu.npu_format_cast.call_args.kwargs == {
        "customize_dtype": torch.float8_e4m3fn,
        "input_dtype": npu.float4_e2m1fn_x2,
    }
    expected_weight = weight.transpose(-1, -2)
    expected_scale = raw_scale.reshape(*expert_shape, 4, 128 // group_size, 2).transpose(-3, -2)
    assert torch.equal(loaded_weight, expected_weight)
    assert loaded_weight.stride() == expected_weight.stride()
    assert torch.equal(loaded_scale, expected_scale)
    assert loaded_scale.dtype == torch.uint8
    assert loaded_scale.stride() == expected_scale.stride()
    assert loaded_scale.data_ptr() == raw_scale.data_ptr()


def test_mxfp4_projection_uses_ordinary_device_operator_gmm_boundary():
    runtime = GS32ExpertWeights.__new__(GS32ExpertWeights)
    runtime.mxfp4_project = None
    runtime.device_operator = SimpleNamespace(npu_grouped_matmul_gmm2=Mock(return_value=object()))
    runtime.mxfp4_dtype, runtime.e8m0_dtype, runtime.mxfp4_quant_type = object(), object(), object()
    data = torch.arange(3 * 128, dtype=torch.int32).to(torch.uint8).reshape(3, 128)
    scales = torch.arange(12, dtype=torch.uint8).reshape(3, 4)
    fields = {"weight": torch.empty(2, 64, 32), "scale": torch.empty(2, 2, 32, 2, dtype=torch.uint8)}
    counts = torch.tensor([1, 1], dtype=torch.int64)  # Fixed capacity has one unused row.
    result = runtime._mxfp4(data, scales, {"fields": fields}, counts)
    assert result is runtime.device_operator.npu_grouped_matmul_gmm2.return_value
    call = runtime.device_operator.npu_grouped_matmul_gmm2.call_args
    assert not call.args
    kwargs = dict(call.kwargs)
    quantized = kwargs.pop("hidden_states")
    assert quantized.dtype == torch.float8_e4m3fn and quantized.data_ptr() == data.data_ptr()
    pertoken = kwargs.pop("per_token_scale")
    assert torch.equal(pertoken, scales.reshape(3, 2, 2)) and pertoken.data_ptr() == scales.data_ptr()
    assert kwargs.pop("weight") is fields["weight"]
    assert kwargs.pop("weight_scale") is fields["scale"]  # Tensor, avoiding nested list in W4 GMM.
    assert kwargs.pop("group_list") is counts
    assert kwargs == {
        "group_list_type": 1,
        "input_dtype": torch.bfloat16,
        "act_quant_type": torch.float8_e4m3fn,
        "weight_quant_type": runtime.mxfp4_dtype,
        "scale_type": None,
        "per_token_scale_type": runtime.e8m0_dtype,
        "use_bf16": True,
        "use_mxfp_quant": True,
        "bias": None,
        "mxfp_quant_dtype": runtime.mxfp4_quant_type,
    }
