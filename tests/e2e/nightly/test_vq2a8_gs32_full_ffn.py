# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in real W2/W4 full FFN equivalence and dynamic NPU graph acceptance."""

import json
import statistics
from types import SimpleNamespace

import pytest

from tests.e2e.nightly.vq2a8_gs32_full_ffn_helpers import (
    FUSION_FP8_MISMATCH_LIMIT,
    CapturedBank,
    fusion_byte_metrics,
    fusion_error_metrics,
    fusion_metrics_accepted,
    route_counts,
    select_fusion_expert,
)
from tests.e2e.nightly.vq2a8_gs32_helpers import checkpoint_inventory, load_selected_layer, native_paths
from tests.support.vq2a8_oracle import OfficialFFNOracle

FULL_FFN_PAIRS = (("vq2", "vq2"), ("vq2", "mxfp4"), ("mxfp4", "mxfp4"))
GRAPH_CAPACITIES = (6, 128, 129, 384)
TIMING_WARMUPS = 3
TIMING_REPEATS = 5


def selected_paths(request):
    values = {
        name: request.config.getoption(flag, default=None)
        for name, flag in (
            ("model", "--vq2a8-model"),
            ("gs32_library", "--vq2a8-gs32-library"),
            ("gs32_binary", "--vq2a8-gs32-binary"),
        )
    }
    if not values["model"]:
        pytest.skip("Pass a real mixed source-v3 checkpoint and rebuilt full FFN native artifacts on Ascend950.")
    return native_paths(values)


def select_mixed_layer(entries):
    """Require one layer with all three pairs so one launch sees mixed offsets."""
    for layer_index, entry in sorted(entries.items()):
        selected = [select_fusion_expert({layer_index: entry}, pair) for pair in FULL_FFN_PAIRS]
        if all(item is not None for item in selected):
            return layer_index, tuple(sorted(item["position"] for item in selected))
    return None


class FullFFNLaunchCapture:
    """Observe the real native call and intermediate buffers without device copies."""

    def __init__(self, ops):
        self.ops = ops
        self.calls = 0
        self.middle = self.scales = None

    def __getattr__(self, name):
        return getattr(self.ops, name)

    def full_ffn_device(self, data, scales, fields, routes, plan, middle, middle_scales, output):
        result = self.ops.full_ffn_device(data, scales, fields, routes, plan, middle, middle_scales, output)
        self.calls += 1
        self.middle, self.scales = middle, middle_scales
        return result


def load_runtime(paths, layer_index, positions, *, execution="fused"):
    import torch
    import torch_npu  # noqa: F401

    from vllm_ascend.ops.fused_moe.vq2a8_gs32 import GS32ExpertWeights
    from vllm_ascend.quantization.methods.w2a8.vq2a8_gs32_full_ffn_layout import build_full_ffn_bank

    assert torch.npu.is_available(), "Explicit full FFN acceptance requires real Ascend hardware."
    device = torch.device("npu", 0)
    torch.npu.set_device(device)
    raw, spec, config = load_selected_layer(paths["model"], layer_index=layer_index, expert_positions=positions)
    oracle = OfficialFFNOracle(raw, spec, device)
    for name, value in vars(raw).copy().items():
        if isinstance(value, torch.Tensor):
            setattr(raw, name, value.to(device))
    limit = config.get("swiglu_limit") or 0.0
    raw.vq2a8_full_ffn_bank = build_full_ffn_bank(raw, spec, swiglu_limit=limit)
    for name in tuple(vars(raw)):
        if name.startswith(("w13_", "w2_")):
            delattr(raw, name)
    raw.vq2a8_ep_size = 2  # Permit zero-route EP ranks and padded tails.
    options = {key: value for key, value in paths.items() if key != "model"}
    bank = GS32ExpertWeights(
        raw,
        spec,
        {**options, "gs32_execution": execution, "gs32_fusion": "full_ffn"},
        graph_row_capacity=max(GRAPH_CAPACITIES),
    )
    launch = FullFFNLaunchCapture(bank.ops)
    bank.ops = launch
    return SimpleNamespace(
        torch=torch, device=device, layer=raw, spec=spec, limit=limit, bank=bank, oracle=oracle, launch=launch
    )


def assert_outputs(runtime, expected, actual, context):
    metrics = fusion_error_metrics(runtime.torch, expected, actual)
    print("VQ2A8_GS32_FULL_FFN_OUTPUT=" + json.dumps({**context, **metrics}), flush=True)
    assert fusion_metrics_accepted(metrics), f"Full FFN exceeded existing fusion thresholds: {context}: {metrics}"


@pytest.mark.parametrize("pair", FULL_FFN_PAIRS, ids=lambda pair: "_to_".join(pair))
def test_gs32_real_experts_full_ffn_compare(request, pair):
    paths = selected_paths(request)
    _, _, entries = checkpoint_inventory(paths["model"])
    selected = select_fusion_expert(entries, pair)
    if selected is None:
        pytest.skip(f"No real {pair} expert in this checkpoint; this precision pair was not tested.")
    runtime = load_runtime(paths, selected["layer"], (selected["position"],))
    torch, spec = runtime.torch, runtime.spec
    generator = torch.Generator(device="cpu").manual_seed(95032)
    for rows in (1, 128, 129):
        hidden = (torch.randn(rows, spec.specs["gate_up"].columns, generator=generator) * 0.1).bfloat16()
        hidden = hidden.to(runtime.device)
        groups = route_counts(torch, runtime.layer, spec, (rows,), runtime.device, 1)
        expected, intermediate = runtime.oracle(hidden, groups, 1, runtime.limit)
        calls = runtime.launch.calls
        actual = runtime.bank.apply(hidden, groups, 1, runtime.limit)[0]
        torch.npu.synchronize()
        assert runtime.launch.calls == calls + 1, "The candidate must submit the actual full_ffn_device op."
        context = {**selected, "rows": rows, "execution": "fused"}
        assert_outputs(runtime, expected, actual, context)
        scales = fusion_byte_metrics(torch, intermediate["scales"], runtime.launch.scales.cpu())
        data = fusion_byte_metrics(torch, intermediate["data"], runtime.launch.middle.cpu())
        print("VQ2A8_GS32_FULL_FFN_MIDDLE=" + json.dumps({**context, "scales": scales, "data": data}), flush=True)
        assert scales["different"] == 0, "Full FFN changed the GS32 E8M0 scale algorithm."
        assert data["fraction"] <= FUSION_FP8_MISMATCH_LIMIT, "Full FFN changed too many intermediate FP8 bytes."


@pytest.mark.parametrize("group_type", (0, 1), ids=("prefix", "counts"))
def test_gs32_full_ffn_mixed_graph_routes_and_buckets(request, group_type):
    paths = selected_paths(request)
    _, _, entries = checkpoint_inventory(paths["model"])
    selected = select_mixed_layer(entries)
    if selected is None:
        pytest.skip("No single real layer contains all W2/W2, W2/W4 and W4/W4 pairs; mixed launch was not tested.")
    runtime = load_runtime(paths, *selected, execution="full_graph")
    eager_runtime = load_runtime(paths, *selected, execution="fused")
    torch, spec = runtime.torch, runtime.spec
    bank = runtime.bank
    generator = torch.Generator(device="cpu").manual_seed(950384)
    originals, captures, scale_buffers = {}, {}, {}
    # Capture with no active routes. Replay must update scheduling on device;
    # no Python branch may freeze an initially inactive precision stage.
    empty = route_counts(torch, runtime.layer, spec, (0, 0, 0), runtime.device, group_type)
    for capacity in GRAPH_CAPACITIES:
        hidden = (torch.randn(capacity, spec.specs["gate_up"].columns, generator=generator) * 0.1).bfloat16()
        originals[capacity] = hidden.to(runtime.device)
        captures[capacity] = CapturedBank(
            runtime, bank, originals[capacity], empty, group_type, validity=bank.validity_buffer
        )
        scale_buffers[capacity] = runtime.launch.scales
    assert len(bank.capture_owners) == len(GRAPH_CAPACITIES)
    assert len({capture.output.data_ptr() for capture in captures.values()}) == len(GRAPH_CAPACITIES)
    submissions = runtime.launch.calls
    assert submissions > 0 and bank.full_ffn_graph_captures == len(GRAPH_CAPACITIES)
    for capacity in (384, 6, 128, 129, 6, 384):
        hidden = originals[capacity]
        cases = (
            (0, 0, 0),
            (capacity, 0, 0),
            (0, capacity, 0),
            (0, 0, capacity),
            (capacity // 3, capacity // 3, capacity - 2 * (capacity // 3)),
            (1, 0, 1),
        )
        for counts in cases:
            groups = route_counts(torch, runtime.layer, spec, counts, runtime.device, group_type)
            inputs = hidden
            if counts == (1, 0, 1):
                inputs = hidden.clone()
                inputs[2:] = float("nan")
                inputs[-1] = float("inf")
            expected, _ = runtime.oracle(inputs, groups, group_type, runtime.limit)
            eager = eager_runtime.bank.apply(inputs, groups, group_type, runtime.limit)[0]
            actual = captures[capacity].replay_checked(inputs, groups)
            assert_outputs(
                runtime, expected, actual, {"capacity": capacity, "counts": counts, "group_type": group_type}
            )
            torch.testing.assert_close(actual, eager.cpu(), rtol=0, atol=0)
            active = sum(counts)
            assert torch.count_nonzero(actual[active:]) == 0
            scales = scale_buffers[capacity]
            assert bool((scales[active:].cpu() == 127).all()), "Graph padding must reset scales on every replay."
    assert runtime.launch.calls == submissions, "Graph replay must not resubmit through the Python launch wrapper."
    # A failed replay must not poison the next graph or retain a stale success flag.
    bad = originals[384].clone()
    bad[0, 0] = float("nan")
    groups = route_counts(torch, runtime.layer, spec, (384, 0, 0), runtime.device, group_type)
    with pytest.raises(ValueError, match="deferred validity"):
        captures[384].replay_checked(bad, groups)
    for bad_counts in ((-1, 1, 0), (385, 0, 0), (2**62, 2**62, 0)):
        invalid = route_counts(torch, runtime.layer, spec, bad_counts, runtime.device, group_type)
        with pytest.raises(ValueError, match="deferred validity"):
            captures[384].replay_checked(originals[384], invalid)
        # Every failed replay must clear its own status on the next valid call.
        recovered_groups = route_counts(torch, runtime.layer, spec, (0, 0, 6), runtime.device, group_type)
        captures[6].replay_checked(originals[6], recovered_groups)
    groups = route_counts(torch, runtime.layer, spec, (0, 0, 6), runtime.device, group_type)
    expected, _ = runtime.oracle(originals[6], groups, group_type, runtime.limit)
    recovered = captures[6].replay_checked(originals[6], groups)
    assert_outputs(runtime, expected, recovered, {"recovery": "large invalid to small W4/W4 graph"})
    # Record measurements without universal pass/fail thresholds. The eager
    # owner check runs outside timing, matching a real forward's deferred check.
    eager_runtime.bank.defer_validity = True
    try:
        for capacity in (min(GRAPH_CAPACITIES), max(GRAPH_CAPACITIES)):
            counts = (capacity // 3, capacity // 3, capacity - 2 * (capacity // 3))
            groups = route_counts(torch, runtime.layer, spec, counts, runtime.device, group_type)
            for _ in range(TIMING_WARMUPS):
                eager_runtime.bank.apply(originals[capacity], groups, group_type, runtime.limit)
                captures[capacity].replay_checked(originals[capacity], groups)
            torch.npu.synchronize()
            eager_ms, graph_ms = [], []
            for _ in range(TIMING_REPEATS):
                eager_runtime.bank.validity_buffer.zero_()
                begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                begin.record()
                eager_output = eager_runtime.bank.apply(originals[capacity], groups, group_type, runtime.limit)[0]
                end.record()
                torch.npu.synchronize()
                assert bool(eager_runtime.bank.validity_buffer.cpu())
                eager_ms.append(begin.elapsed_time(end))
                graph_output = captures[capacity].replay_checked(originals[capacity], groups, timing_ms=graph_ms)
                torch.testing.assert_close(graph_output, eager_output.cpu(), rtol=0, atol=0)
            print(
                "VQ2A8_GS32_FULL_FFN_TIMING="
                + json.dumps(
                    {
                        "capacity": capacity,
                        "group_type": group_type,
                        "counts": counts,
                        "warmups": TIMING_WARMUPS,
                        "repeats": TIMING_REPEATS,
                        "measurement": "NPUEvent; input copies and owner status read excluded",
                        "eager_ms": eager_ms,
                        "graph_ms": graph_ms,
                        "median_eager_ms": statistics.median(eager_ms),
                        "median_graph_ms": statistics.median(graph_ms),
                    }
                ),
                flush=True,
            )
    finally:
        eager_runtime.bank.defer_validity = False
    print(
        "VQ2A8_GS32_FULL_FFN_GRAPH="
        + json.dumps(
            {
                "status": "PASS",
                "experts": spec.expert_ids,
                "group_type": group_type,
                "capacities": GRAPH_CAPACITIES,
                "native_submissions": submissions,
                "captures": bank.full_ffn_graph_captures,
                "replays": sum(capture.replays for capture in captures.values()),
            }
        ),
        flush=True,
    )
