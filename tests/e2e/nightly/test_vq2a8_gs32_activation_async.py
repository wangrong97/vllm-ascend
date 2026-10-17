# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-NPU deferred validity and graph replay, without native projections.

GateUp and final output are supplied tensors so this isolates activation and
validity kernels. It does not claim native MoE or whole-model graph coverage.
Pass a GS32 --vq2a8-model to opt into hardware execution, as in the strict suite.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from tests.e2e.nightly.test_vq2a8_gs32_activation import (
    activation_inputs,
    activation_runtime,  # noqa: F401
)


@pytest.fixture(scope="module")
def async_runtime(request):
    runtime = request.getfixturevalue("activation_runtime")

    from vllm_ascend.ops.fused_moe.vq2a8_gs32_activation import finish_flags, prepare_unchecked, swiglu_unchecked

    return SimpleNamespace(
        **vars(runtime), prepare_async=prepare_unchecked, finish=finish_flags, swiglu=swiglu_unchecked
    )


def host_case(runtime, rows, width, fault=None):
    inputs = list(activation_inputs(runtime, rows, width))
    gate = (inputs[0].float() * 2).bfloat16()
    output = inputs[0].clone()
    if fault == "input_nan":
        inputs[0][0, 0] = float("nan")
    elif fault == "input_inf":
        inputs[0][0, 0] = float("inf")
    elif fault == "slot_oob":
        inputs[2][0] = inputs[1].shape[0]
    elif fault == "perm_oob":
        inputs[1][inputs[2][0], 0] = width
    elif fault == "raw_gate_inf":
        gate[0, 0] = float("inf")
    elif fault == "output_hole":
        output[-1, -1] = float("nan")
    return (*inputs, gate, output)


def enqueue_case(runtime, tensors):
    hidden, perm, slots, gate, output = tensors
    gate_data, gate_scale, gate_flags = runtime.prepare_async(hidden, perm, slots)
    middle = runtime.swiglu(gate, 10.0)
    width = middle.shape[1]
    # Projection outputs are deliberate test inputs, not a simulated native
    # projection. Down preparation itself runs the production NPU kernels.
    down_perm = (perm[:, :width] % width).contiguous()
    down_data, down_scale, down_flags = runtime.prepare_async(middle, down_perm, slots, raw_gate=gate)
    status = runtime.finish(output, gate_flags, down_flags)
    return status, gate_data, gate_scale, down_data, down_scale


def assert_finite_quantized(runtime, results):
    torch = runtime.torch
    for data in (results[1], results[3]):
        host = data.cpu().view(torch.float8_e4m3fn).float()
        assert torch.isfinite(host).all(), "Invalid input reached a non-finite native activation byte."
    for scales in (results[2], results[4]):
        host = scales.cpu()
        assert ((host >= 1) & (host <= 254)).all()


def assert_gate_bytes(runtime, host, results):
    hidden, perm, slots = host[:3]
    transformed = np.stack([hidden[row, perm[slot]].float().numpy() for row, slot in enumerate(slots)])
    data, scales = runtime.reference.quantize_h(transformed)
    np.testing.assert_array_equal(results[1].cpu().numpy(), data.view(np.uint8))
    np.testing.assert_array_equal(results[2].cpu().numpy(), scales)


@pytest.mark.parametrize("fault", ["input_nan", "input_inf", "slot_oob", "perm_oob", "raw_gate_inf", "output_hole"])
def test_gs32_activation_async_chain_rejects_and_recovers(async_runtime, fault):
    runtime, torch = async_runtime, async_runtime.torch
    stream = torch.npu.Stream(device=runtime.device)
    stream.wait_stream(torch.npu.current_stream(runtime.device))
    with torch.npu.stream(stream):
        for injected in (None, fault, None):
            host = host_case(runtime, 3, 128, injected)
            tensors = tuple(value.to(runtime.device) for value in host)
            results = enqueue_case(runtime, tensors)
            # There is no intermediate synchronization or CPU flag read.
            assert results[0].cpu().tolist() == [int(injected is None)]
            assert_finite_quantized(runtime, results)
            if injected is None:
                assert_gate_bytes(runtime, host, results)
    torch.npu.current_stream(runtime.device).wait_stream(stream)


@pytest.mark.parametrize("rows,width", [(1, 128), (8, 4096)])
def test_gs32_activation_async_graph_refreshes_flags_and_inputs(async_runtime, rows, width):
    runtime, torch = async_runtime, async_runtime.torch
    host = host_case(runtime, rows, width)
    tensors = tuple(value.to(runtime.device) for value in host)
    stream = torch.npu.Stream(device=runtime.device)
    stream.wait_stream(torch.npu.current_stream(runtime.device))
    with torch.npu.stream(stream):
        for _ in range(2):
            assert enqueue_case(runtime, tensors)[0].cpu().tolist() == [1]
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            results = enqueue_case(runtime, tensors)
        # Repeatedly refresh the SAME captured buffers and consume the SAME
        # status address, including invalid-to-valid transitions.
        for index, fault in enumerate((None, "raw_gate_inf", None, "input_nan", "output_hole", None)):
            refreshed = list(host_case(runtime, rows, width, fault))
            if fault is None:
                refreshed[0] = (refreshed[0].float() + index / 32).bfloat16()
            for target, source in zip(tensors, refreshed, strict=True):
                target.copy_(source)
            graph.replay()
            assert results[0].cpu().tolist() == [int(fault is None)]
            assert_finite_quantized(runtime, results)
            if fault is None:
                assert_gate_bytes(runtime, refreshed, results)
    torch.npu.current_stream(runtime.device).wait_stream(stream)


@pytest.mark.parametrize("permuted", [False, True])
@pytest.mark.parametrize("limit", [0.0, 10.0])
def test_swiglu_prepare_fusion_graph_refreshes_raw_values_and_validity(async_runtime, permuted, limit):
    runtime, torch = async_runtime, async_runtime.torch
    rows, width = 8, 2048
    host, permutation, slots = activation_inputs(runtime, rows, width)
    raw_host = torch.cat((host * 20, host * 30), dim=1).contiguous()
    raw = raw_host.to(runtime.device)
    permutation = permutation.to(runtime.device) if permuted else None
    slots = slots.to(runtime.device)
    stream = torch.npu.Stream(device=runtime.device)
    stream.wait_stream(torch.npu.current_stream(runtime.device))
    with torch.npu.stream(stream):
        for _ in range(2):
            result = runtime.prepare_async(raw, permutation, slots, swiglu_limit=limit)
            assert result[2].cpu().tolist() == [1] * rows
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            result = runtime.prepare_async(raw, permutation, slots, swiglu_limit=limit)
        addresses = tuple(value.data_ptr() for value in result)
        for step, fault in enumerate((None, "gate_inf", None, "up_nan", None)):
            refreshed = (raw_host.float() + step / 32).bfloat16()
            if fault == "gate_inf":
                refreshed[2, 17] = float("inf")
            elif fault == "up_nan":
                refreshed[2, width + 35] = float("nan")
            raw.copy_(refreshed)
            graph.replay()
            expected_flags = [1] * rows
            if fault:
                expected_flags[2] = 0
            assert result[2].cpu().tolist() == expected_flags
            assert tuple(value.data_ptr() for value in result) == addresses
            assert torch.isfinite(result[0].cpu().view(torch.float8_e4m3fn).float()).all()
            assert ((result[1].cpu() >= 1) & (result[1].cpu() <= 254)).all()
            if fault is None:
                middle = runtime.strict_swiglu(raw, limit)
                expected = runtime.prepare(middle, permutation, slots, raw_gate=raw)
                for found, reference in zip(result[:2], expected, strict=True):
                    assert torch.equal(found.cpu(), reference.cpu())
    torch.npu.current_stream(runtime.device).wait_stream(stream)


@pytest.mark.parametrize(
    "rows,width", [(1, 32), (5, 96), (6, 2048), (3, 4096), (127, 160), (128, 96), (129, 96), (1025, 96), (32769, 96)]
)
@pytest.mark.parametrize("permuted", [False, True])
@pytest.mark.parametrize("mode", ["hidden", "raw_gate", "swiglu", "clipped_swiglu"])
def test_gathered_preparation_matches_explicit_index_select(async_runtime, rows, width, permuted, mode):
    runtime, torch = async_runtime, async_runtime.torch
    host, permutation, slots = activation_inputs(runtime, rows, width)
    raw_host = torch.cat((host * 20, host.roll(17, 1) * 30), dim=1).contiguous()
    hidden = (raw_host if "swiglu" in mode else host).to(runtime.device)
    raw = raw_host.to(runtime.device)
    permutation = permutation.to(runtime.device) if permuted else None
    source_host = torch.arange(rows, dtype=torch.int64).roll(1)
    if rows > 1:
        source_host[-1] = -1
        # Inactive rows do not inspect either source data or the physical slot.
        slots[-1] = -1
    source, slots = source_host.to(runtime.device), slots.to(runtime.device)
    options = {}
    if "swiglu" in mode:
        options["swiglu_limit"] = 10.0 if mode == "clipped_swiglu" else 0.0
    elif mode == "raw_gate":
        options["raw_gate"] = raw
    route_flags = torch.ones(1, device=runtime.device, dtype=torch.int32)
    actual = runtime.prepare_async(hidden, permutation, slots, source_rows=source, route_flags=route_flags, **options)

    # Materialize only the independent comparison input. The implementation
    # under test must gather directly from hidden/raw GateUp in the kernel.
    selected = hidden.index_select(0, source.clamp(min=0))
    reference_slots = slots.clone()
    reference_options = dict(options)
    if mode == "raw_gate":
        reference_options["raw_gate"] = raw.index_select(0, source.clamp(min=0))
    if rows > 1:
        selected[-1].zero_()
        reference_slots[-1] = 0
        if mode == "raw_gate":
            reference_options["raw_gate"][-1].zero_()
    expected = runtime.prepare_async(selected, permutation, reference_slots, **reference_options)
    assert actual[2].cpu().tolist() == [1] * rows
    for found, reference in zip(actual, expected, strict=True):
        assert torch.equal(found.cpu(), reference.cpu())
    if rows > 1:
        assert actual[0][-1].cpu().count_nonzero() == 0
        assert (actual[1][-1].cpu() == 127).all()


@pytest.mark.parametrize("mode", ["raw_gate", "clipped_swiglu"])
@pytest.mark.parametrize(
    "fault",
    [
        "source_negative",
        "source_bound",
        "source_large",
        "route_zero",
        "route_two",
        "raw_nan",
        "raw_inf",
        "slot_bound",
        "perm_bound",
    ],
)
def test_gathered_preparation_faults_recover_with_finite_bytes(async_runtime, mode, fault):
    runtime, torch = async_runtime, async_runtime.torch
    rows, width = 5, 96
    host, permutation_host, slots_host = activation_inputs(runtime, rows, width)
    raw_host = torch.cat((host * 20, host * 30), dim=1).contiguous()
    source_host = torch.tensor([4, 2, -1, 0, 1], dtype=torch.int64)
    slots_host[2] = -1
    hidden = (raw_host if mode == "clipped_swiglu" else host).to(runtime.device)
    raw = hidden if mode == "clipped_swiglu" else raw_host.to(runtime.device)
    permutation, slots = permutation_host.to(runtime.device), slots_host.to(runtime.device)
    source = source_host.to(runtime.device)
    route = torch.ones(1, dtype=torch.int32, device=runtime.device)
    options = {"swiglu_limit": 10.0} if mode == "clipped_swiglu" else {"raw_gate": raw}
    stream = torch.npu.Stream(device=runtime.device)
    stream.wait_stream(torch.npu.current_stream(runtime.device))
    with torch.npu.stream(stream):
        for injected in (None, fault, None):
            refreshed_raw, refreshed_source = raw_host.clone(), source_host.clone()
            refreshed_permutation, refreshed_slots = permutation_host.clone(), slots_host.clone()
            status, expected_flags = 1, [1] * rows
            if injected in ("source_negative", "source_bound", "source_large"):
                refreshed_source[0] = {"source_negative": -2, "source_bound": rows, "source_large": 2**40}[injected]
                expected_flags[0] = 0
            elif injected in ("route_zero", "route_two"):
                status = 0 if injected == "route_zero" else 2
                expected_flags = [0] * rows
            elif injected in ("raw_nan", "raw_inf"):
                refreshed_raw[4, 2 * width - 1] = float("nan") if injected == "raw_nan" else float("inf")
                expected_flags[0] = 0
            elif injected == "slot_bound":
                refreshed_slots[0] = permutation_host.shape[0]
                expected_flags[0] = 0
            elif injected == "perm_bound":
                refreshed_permutation[slots_host[0], width - 1] = width
                expected_flags = [int(index == 2 or slot != slots_host[0]) for index, slot in enumerate(slots_host)]
            raw.copy_(refreshed_raw)
            source.copy_(refreshed_source)
            permutation.copy_(refreshed_permutation)
            slots.copy_(refreshed_slots)
            route.fill_(status)
            result = runtime.prepare_async(hidden, permutation, slots, source_rows=source, route_flags=route, **options)
            assert result[2].cpu().tolist() == expected_flags
            assert torch.isfinite(result[0].cpu().view(torch.float8_e4m3fn).float()).all()
            assert ((result[1].cpu() >= 1) & (result[1].cpu() <= 254)).all()
            if injected is None:
                safe_source = source.clamp(min=0)
                selected = hidden.index_select(0, safe_source)
                selected[2].zero_()
                reference_slots = slots.clone()
                reference_slots[2] = 0
                reference_options = dict(options)
                if mode == "raw_gate":
                    selected_raw = raw.index_select(0, safe_source)
                    selected_raw[2].zero_()
                    reference_options["raw_gate"] = selected_raw
                expected = runtime.prepare_async(selected, permutation, reference_slots, **reference_options)
                assert all(torch.equal(a.cpu(), b.cpu()) for a, b in zip(result, expected, strict=True))
            if injected in ("source_negative", "source_bound", "source_large", "route_zero", "route_two"):
                assert result[0][0].cpu().count_nonzero() == 0 and (result[1][0].cpu() == 127).all()
    torch.npu.current_stream(runtime.device).wait_stream(stream)
