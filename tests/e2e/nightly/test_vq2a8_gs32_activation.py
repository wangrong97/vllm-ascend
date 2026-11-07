# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in real-NPU activation bytes; no model loading or native projection.

The GS32 model option opts into Ascend950 execution and checks the checkpoint
format only. Independent CPU quantization comes from the imported operator's
NumPy/ml_dtypes reference, never the new fused implementation.
"""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture(scope="module")
def activation_runtime(request):
    selected = request.config.getoption("--vq2a8-model", default=None)
    if not selected:
        pytest.skip("Pass a source-v3 GS32 --vq2a8-model to opt into real Ascend activation execution.")
    model = Path(selected).resolve(strict=True)
    config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    assert config["quantization_config"]["format"] == "vq2_mixed_gs32_source"
    assert config["quantization_config"]["format_version"] == 3

    import torch
    import torch_npu  # noqa: F401

    from tests.support.vq2a8_oracle import quantize_mxfp8_k32, staged_swiglu
    from vllm_ascend.ops.fused_moe.vq2a8_gs32_activation import finish_full_ffn, prepare_unchecked

    assert torch.npu.is_available(), "Explicit activation tests require a real Ascend NPU."
    # This is a required independent oracle for explicitly requested hardware
    # tests. Missing ml_dtypes is a failure, not a silently skipped comparison.
    source = Path(__file__).resolve().parents[3] / "csrc/mixed_w2w4_moe_ffn/scripts/swiglu_reference.py"
    spec = importlib.util.spec_from_file_location("_gs32_activation_independent_reference", source)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    device = torch.device("npu", 0)
    torch.npu.set_device(device)
    return SimpleNamespace(
        torch=torch,
        device=device,
        prepare=prepare_unchecked,
        finish=finish_full_ffn,
        strict_quantize=quantize_mxfp8_k32,
        strict_swiglu=staged_swiglu,
        reference=reference,
    )


def activation_inputs(runtime, rows, width):
    torch = runtime.torch
    generator = torch.Generator(device="cpu").manual_seed(952)
    hidden = (torch.randn(rows, width, generator=generator) * 0.25).bfloat16()
    return (hidden,)


def compare_activation(runtime, host_inputs, *, raw_gate=None, check_strict=True):
    torch = runtime.torch
    (hidden,) = host_inputs
    expected_data, expected_scale = runtime.reference.quantize_h(hidden.float().numpy())
    device_inputs = tuple(value.to(runtime.device) for value in host_inputs)
    assert raw_gate is None
    actual_data, actual_scale, flags = runtime.prepare(*device_inputs)
    assert bool(flags.cpu().all())
    torch.npu.synchronize()
    np.testing.assert_array_equal(actual_data.cpu().numpy(), expected_data.view(np.uint8))
    np.testing.assert_array_equal(actual_scale.cpu().numpy(), expected_scale)
    if check_strict:
        strict_data, strict_scale = runtime.strict_quantize(hidden.cpu())
        assert torch.equal(actual_data.cpu(), strict_data.cpu())
        assert torch.equal(actual_scale.cpu(), strict_scale.cpu())
    return actual_data, actual_scale


@pytest.mark.parametrize(
    "rows,width", [(1, 64), (3, 64), (3, 96), (6, 2048), (24, 4096), (32, 4096), (127, 64), (128, 64), (129, 64)]
)
def test_gs32_fused_activation_bytes_match_independent_cpu_and_same_npu(activation_runtime, rows, width):
    runtime = activation_runtime
    compare_activation(runtime, activation_inputs(runtime, rows, width))


def test_gs32_fused_activation_all_finite_bf16_patterns_and_signed_zero(activation_runtime):
    runtime, torch = activation_runtime, activation_runtime.torch
    bits = torch.arange(65536, dtype=torch.int32).to(torch.int16)
    values = bits.view(torch.bfloat16)
    values = values[torch.isfinite(values.float())]
    assert values.numel() == 65280
    hidden = values[:, None].expand(-1, 64).contiguous()
    fields = (hidden,)
    compare_activation(runtime, fields)


def test_gs32_fused_activation_mixed_ties_and_e8m0_boundaries(activation_runtime):
    runtime, torch = activation_runtime, activation_runtime.torch
    values = torch.tensor(
        [
            0.0,
            -0.0,
            1 / 1024,
            -1 / 1024,
            3 / 1024,
            -3 / 1024,
            1.0625,
            -1.0625,
            1.1875,
            -1.1875,
            1.75,
            -1.75,
            1.7578125,
            -1.7578125,
            223.0,
            -223.0,
            224.0,
            -224.0,
            225.0,
            -225.0,
            226.0,
            -226.0,
            447.0,
            -447.0,
            448.0,
            -448.0,
            15 / 1024,
            -15 / 1024,
            1 / 64,
            -1 / 64,
            1 / 512,
            -1 / 512,
        ]
    )
    rows = []
    for exponent in (-120, -16, 0, 16, 110):
        for maximum in (224.0, 226.0, 448.0, 450.0):
            row = values.clamp(-maximum, maximum)
            row[-2], row[-1] = maximum, -maximum
            rows.append((row * (2.0**exponent)).repeat(2))
    hidden = torch.stack(rows).bfloat16()
    compare_activation(
        runtime,
        (hidden,),
    )


def test_gs32_fused_activation_nondefault_stream_repeated_fresh_values(activation_runtime):
    runtime, torch = activation_runtime, activation_runtime.torch
    stream = torch.npu.Stream(device=runtime.device)
    stream.wait_stream(torch.npu.current_stream(runtime.device))
    with torch.npu.stream(stream):
        for delta in (0.03125, -0.0625):
            host = list(activation_inputs(runtime, 7, 2048))
            host[0] = (host[0].float() + delta).bfloat16()
            compare_activation(runtime, tuple(host))
    torch.npu.current_stream(runtime.device).wait_stream(stream)


def test_source_v3_mxfp4_activation_uses_natural_columns(activation_runtime):
    runtime = activation_runtime
    (hidden,) = activation_inputs(runtime, 8, 4096)
    actual, actual_scale, flags = runtime.prepare(hidden.to(runtime.device))
    assert bool(flags.cpu().all())
    expected, expected_scale = runtime.reference.quantize_h(hidden.float().numpy())
    np.testing.assert_array_equal(actual.cpu().numpy(), expected.view(np.uint8))
    np.testing.assert_array_equal(actual_scale.cpu().numpy(), expected_scale)


@pytest.mark.parametrize("capacity", [7, 129])
def test_identity_activation_graph_masks_nan_padding_and_recovers(activation_runtime, capacity):
    runtime, torch = activation_runtime, activation_runtime.torch
    hidden = torch.ones(capacity, 64, dtype=torch.bfloat16, device=runtime.device)
    metadata = torch.tensor([capacity, 1], dtype=torch.int64, device=runtime.device)
    route_flags = torch.ones(1, dtype=torch.int32, device=runtime.device)

    def forward():
        return runtime.prepare(hidden, route_flags=route_flags, route_metadata=metadata)

    for _ in range(2):
        forward()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        data, scales, flags = forward()
    addresses = tuple(value.data_ptr() for value in (data, scales, flags))
    for live, meta_valid, route_valid, active_nan in (
        (1, 1, 1, False),
        (0, 1, 1, False),
        (capacity, 1, 1, False),
        (1, 1, 1, True),
        (-1, 1, 1, False),
        (capacity + 1, 1, 1, False),
        (1, 0, 1, False),
        (1, 1, 0, False),
        (2, 1, 1, False),
    ):
        host = torch.full((capacity, 64), float("nan"), dtype=torch.bfloat16)
        bounded = max(0, min(capacity, live))
        host[:bounded] = -0.1875
        if active_nan and bounded:
            host[0, 0] = float("nan")
        hidden.copy_(host)
        metadata.copy_(torch.tensor([live, meta_valid], dtype=torch.int64))
        route_flags.fill_(route_valid)
        graph.replay()
        torch.npu.synchronize()
        valid = meta_valid == route_valid == 1 and 0 <= live <= capacity and not active_nan
        assert bool(flags.cpu().all()) == valid
        assert addresses == tuple(value.data_ptr() for value in (data, scales, flags))
        if valid:
            safe = torch.zeros_like(host)
            safe[:live] = host[:live]
            expected_data, expected_scale = runtime.strict_quantize(safe)
            expected_scale[live:] = 127
            assert torch.equal(data.cpu(), expected_data)
            assert torch.equal(scales.cpu(), expected_scale)


@pytest.mark.parametrize("capacity,width", [(7, 64), (129, 4097)])
def test_finish_full_ffn_graph_checks_all_states_and_recovers(activation_runtime, capacity, width):
    runtime, torch = activation_runtime, activation_runtime.torch
    output = torch.ones(capacity, width, dtype=torch.bfloat16, device=runtime.device)
    middle_scales = torch.full((capacity, 2), 127, dtype=torch.uint8, device=runtime.device)
    input_flags = torch.ones(capacity, dtype=torch.int32, device=runtime.device)
    route_flags = torch.ones(1, dtype=torch.int32, device=runtime.device)
    native_flags = torch.ones_like(route_flags)
    metadata = torch.tensor([capacity - 1, 1], dtype=torch.int64, device=runtime.device)
    validity = torch.zeros((), dtype=torch.bool, device=runtime.device)
    inputs = (output, middle_scales, input_flags, route_flags, native_flags, metadata, validity)
    for _ in range(2):
        runtime.finish(*inputs)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        block_flags = runtime.finish(*inputs)
    pointers = tuple(value.data_ptr() for value in (*inputs, block_flags))
    faults = (
        "none",
        "live_nan",
        "live_inf",
        "scale255",
        "padding_scale255",
        "input",
        "padding_input",
        "route",
        "native",
        "metadata",
        "negative_live",
        "oversized_live",
        "empty",
    )
    for fault in faults:
        # Replay every fault followed by a clean request: status must overwrite
        # previous success and failure, and padding must not be read.
        for case in (fault, "none"):
            live = capacity - 1
            output.fill_(2)
            output[-1].fill_(float("nan"))
            middle_scales.fill_(127)
            input_flags.fill_(1)
            route_flags.fill_(1)
            native_flags.fill_(1)
            validity.fill_(True)
            if case in ("live_nan", "live_inf"):
                output[0, 0] = float("nan" if case == "live_nan" else "inf")
            if case in ("scale255", "padding_scale255"):
                middle_scales[-1 if case == "padding_scale255" else 0, 0] = 255
            if case in ("input", "padding_input"):
                input_flags[-1 if case == "padding_input" else 0] = 0
            if case == "route":
                route_flags.zero_()
            if case == "native":
                native_flags.zero_()
            if case == "negative_live":
                live = -1
            if case == "oversized_live":
                live = capacity + 1
            if case == "empty":
                live = 0
                output.fill_(float("nan"))
            metadata.copy_(torch.tensor([live, 0 if case == "metadata" else 1], dtype=torch.int64))
            graph.replay()
            torch.npu.synchronize()
            assert bool(validity.cpu()) == (case in ("none", "empty"))
            result = output.cpu()
            assert bool(torch.isfinite(result).all())
            if case in ("route", "native", "metadata", "negative_live", "oversized_live", "empty"):
                assert not result.count_nonzero()
            else:
                assert not result[-1].count_nonzero()
                if case in ("live_nan", "live_inf"):
                    assert result[0, 0] == 0
                else:
                    assert bool((result[:-1] == 2).all())
            assert pointers == tuple(value.data_ptr() for value in (*inputs, block_flags))


@pytest.mark.parametrize("size", [1, 1024, 1025, 2050])
def test_full_ffn_validity_reduction_device_tile_boundaries(activation_runtime, size):
    from vllm_ascend.ops.triton.activation.vq2a8_gs32 import _reduce_full_ffn_validity_kernel

    runtime, torch = activation_runtime, activation_runtime.torch
    flags = torch.ones(size, dtype=torch.int32, device=runtime.device)
    validity = torch.zeros((), dtype=torch.bool, device=runtime.device)
    for bad_index in (None, 0, size - 1, None):
        flags.fill_(1)
        if bad_index is not None:
            flags[bad_index] = 0
        validity.fill_(bad_index is not None)
        _reduce_full_ffn_validity_kernel[(1,)](flags, validity, FLAGS=size, BLOCK=1024)
        assert bool(validity.cpu()) == (bad_index is None)
