# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in real three-expert GS32 device-route graph acceptance.

This captures one FFN bank, not a full model. Resident weights come from three
experts selected to cover the checkpoint's quantization types. Every replay updates the same input/count addresses
and checks deferred validity outside capture. No CPU emulation is used.
"""

import hashlib
import json
import statistics
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.e2e.nightly.test_vq2a8_gs32_batched import (
    checkpoint_inventory,
    load_selected_layer,
    native_paths,
    prepare_selected_layer,
    reversed_selected_bank,
    route_counts,
)

ROUTED_ROWS = 5
WARMUP_REPEATS = 2
COUNT_CASES = ((1, 1, 3), (0, 5, 0), (3, 0, 2), (5, 0, 0), (0, 0, 5))
# B1..B64 at top-k=6, plus the supported B64/top-k=8 capacity.
# Capture all buckets with the same owner before replaying any of them.
DECODE_ROUTED_CAPACITIES = (6, 12, 24, 48, 96, 192, 384, 512)
FAULT_ROUTED_CAPACITIES = (12, 192, 512)
BUCKET_TIMING_REPEATS = 3


@pytest.fixture(scope="module")
def device_bank_runtime(request):
    options = {
        name: request.config.getoption(flag, default=None)
        for name, flag in (
            ("model", "--vq2a8-model"),
            ("gs32_library", "--vq2a8-gs32-library"),
            ("gs32_binary", "--vq2a8-gs32-binary"),
        )
    }
    if not options["model"]:
        pytest.skip("Pass a unified GS32 --vq2a8-model and both native artifacts on Ascend950.")
    paths = native_paths(options)
    model = paths.pop("model")
    _, _, inventory = checkpoint_inventory(model)
    # Prefer a layer with both quantization types in both projections. Three
    # representative experts cover all four available (projection, type) pairs
    # without allocating the complete bank or assuming particular expert IDs.
    layer_index = max(
        sorted(inventory), key=lambda index: sum(len(set(types.values())) for types in inventory[index].values())
    )
    types = inventory[layer_index]
    expert_ids = sorted(types["gate_up"])
    remaining, selected, covered = expert_ids.copy(), [], set()
    for _ in range(3):
        expert = max(
            remaining, key=lambda expert: len({(kind, values[expert]) for kind, values in types.items()} - covered)
        )
        selected.append(expert_ids.index(expert))
        remaining.remove(expert)
        covered.update((kind, values[expert]) for kind, values in types.items())
    layer, spec, config = load_selected_layer(model, layer_index=layer_index, expert_positions=selected)

    import torch
    import torch_npu  # noqa: F401

    from vllm_ascend.ops.fused_moe.vq2a8_gs32 import GS32ExpertWeights

    assert torch.npu.is_available(), "This opt-in test requires real Ascend hardware."
    device = torch.device("npu", 0)
    torch.npu.set_device(device)
    payload_bytes = prepare_selected_layer(layer, device)
    torch.npu.synchronize()
    baselines = {
        mode: GS32ExpertWeights(layer, spec, {**paths, "gs32_execution": mode, "gs32_fusion": "none"})
        for mode in ("serial", "fused")
    }
    missing = next((expert for expert in range(layer.vq2a8_num_experts) if expert not in spec.expert_ids), None)
    assert missing is not None, (
        "Missing-expert acceptance requires a logical expert outside the three resident experts."
    )
    print(
        "VQ2A8_GS32_DEVICE_BANK_ARTIFACTS="
        + json.dumps(
            {
                "layer": spec.layer_index,
                "experts": spec.expert_ids,
                "resident_weight_bytes": payload_bytes,
                "native": {
                    key: {"path": path, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
                    for key, path in paths.items()
                },
                "scope": "three real experts; not full-model or serving validation",
            }
        ),
        flush=True,
    )
    yield SimpleNamespace(
        torch=torch,
        layer=layer,
        spec=spec,
        paths=paths,
        baselines=baselines,
        device=device,
        missing_expert=missing,
        limit=config.get("swiglu_limit"),
        bank_type=GS32ExpertWeights,
    )
    torch.npu.synchronize()


def reference_output(runtime, hidden, groups, group_type, baselines=None):
    outputs = {}
    for mode, bank in (runtime.baselines if baselines is None else baselines).items():
        output, event = bank.apply(hidden, groups, group_type, runtime.limit)
        assert event is None
        outputs[mode] = output.cpu().clone()
    expected = next(iter(outputs.values()))
    for output in outputs.values():
        runtime.torch.testing.assert_close(output, expected, rtol=0, atol=0)
    return expected


class CapturedBank:
    """Own all graph addresses and perform the required external validity check."""

    def __init__(self, runtime, bank, hidden, groups, group_type, *, validity=None):
        self.runtime = runtime
        self.bank = bank
        self.hidden = hidden.clone()
        self.groups = groups.clone()
        self.group_type = group_type
        self.replays = 0
        torch = runtime.torch
        # This scalar is allocated outside the graph, as in the model runner's
        # shared owner vector. The bank must write it on every execution.
        self.validity = torch.zeros((), dtype=torch.bool, device=runtime.device) if validity is None else validity
        bank.validity_buffer = self.validity
        bank.defer_validity = True
        for _ in range(WARMUP_REPEATS):
            self.validity.zero_()
            bank.apply(self.hidden, self.groups, group_type, runtime.limit)
            assert bool(self.validity.cpu()), "The device bank must pass eager warmup before capture."
        torch.npu.synchronize()
        self.graph = torch.npu.NPUGraph()
        self.validity.zero_()
        # Keep the default torch.npu.graph side stream. Immutable native plans
        # were prepared on the original stream and must support this handoff.
        with torch.npu.graph(self.graph):
            self.output, event = bank.apply(self.hidden, self.groups, group_type, runtime.limit)
        assert event is None
        self.addresses = (
            self.hidden.data_ptr(),
            self.groups.data_ptr(),
            self.output.data_ptr(),
            self.validity.data_ptr(),
        )

    def replay_checked(self, hidden, groups, *, timing_ms=None):
        torch = self.runtime.torch
        self.hidden.copy_(hidden)
        self.groups.copy_(groups)
        self.validity.zero_()
        if timing_ms is not None:
            begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            begin.record()
        self.graph.replay()
        if timing_ms is not None:
            end.record()
        self.replays += 1
        torch.npu.synchronize()
        assert self.addresses == (
            self.hidden.data_ptr(),
            self.groups.data_ptr(),
            self.output.data_ptr(),
            self.validity.data_ptr(),
        )
        if not bool(self.validity.cpu()):
            raise ValueError("GS32 graph replay failed deferred validity checks.")
        if timing_ms is not None:
            timing_ms.append(begin.elapsed_time(end))
        return self.output.cpu().clone()


@pytest.mark.parametrize("group_type", (0, 1), ids=("prefix", "counts"))
@pytest.mark.parametrize("reverse_bank", (False, True), ids=("physical_order", "reversed_physical_order"))
def test_gs32_device_bank_graph_dynamic_routes_and_recovery(device_bank_runtime, group_type, reverse_bank):
    runtime = device_bank_runtime
    torch, layer, spec = runtime.torch, runtime.layer, runtime.spec
    selected_spec = spec
    options = {**runtime.paths, "gs32_execution": "full_graph", "gs32_fusion": "none"}
    bank = (
        reversed_selected_bank(layer, spec, options, runtime.device)
        if reverse_bank
        else runtime.bank_type(layer, spec, options)
    )
    generator = torch.Generator(device="cpu").manual_seed(125)
    original = (torch.randn(ROUTED_ROWS, spec.specs["gate_up"].columns, generator=generator) * 0.1).bfloat16()
    hidden = original.to(runtime.device)
    groups = route_counts(torch, layer, spec, COUNT_CASES[0], runtime.device, group_type)
    captured = CapturedBank(runtime, bank, hidden, groups, group_type)

    reference_routes = []
    for counts in COUNT_CASES:
        groups = route_counts(torch, layer, spec, counts, runtime.device, group_type)
        expected = reference_output(runtime, hidden, groups, group_type)
        actual = captured.replay_checked(hidden, groups)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        reference_routes.append(expected)
    assert any(not torch.equal(value, reference_routes[0]) for value in reference_routes[1:]), (
        "Real route changes must affect the reference, otherwise this fixture cannot detect static captured routing."
    )

    # Same shape and routes, new hidden values: neither the transformed FP8
    # bytes nor captured native output may be reused from the previous replay.
    groups = route_counts(torch, layer, spec, (3, 0, 2), runtime.device, group_type)
    for delta in (0.03125, -0.0625):
        fresh = (original.float() + delta).bfloat16().to(runtime.device)
        expected = reference_output(runtime, fresh, groups, group_type)
        torch.testing.assert_close(captured.replay_checked(fresh, groups), expected, rtol=0, atol=0)

    # Invalid dynamic values use the already captured graph. A device fault or
    # wrong exception fails the test; only the graph-external status may reject.
    missing_counts = torch.zeros(layer.vq2a8_num_experts, dtype=torch.int64)
    missing_counts[runtime.missing_expert] = ROUTED_ROWS
    if group_type == 0:
        missing_counts = missing_counts.cumsum(0)
    bad_hidden = hidden.clone()
    bad_hidden[0, 0] = float("nan")
    bad_cases = {
        "nan_hidden": (bad_hidden, groups),
        "missing_expert": (hidden, missing_counts.to(runtime.device)),
        "negative_count_or_decreasing_prefix": (
            hidden,
            route_counts(torch, layer, spec, (-1, 6, 0), runtime.device, group_type),
        ),
        "zero_count_for_nonempty_rows": (
            hidden,
            route_counts(torch, layer, spec, (0, 0, 0), runtime.device, group_type),
        ),
    }
    recovered = reference_output(runtime, hidden, groups, group_type)
    for name, (invalid_hidden, invalid_groups) in bad_cases.items():
        with pytest.raises(ValueError, match="GS32 graph replay failed deferred validity checks"):
            captured.replay_checked(invalid_hidden, invalid_groups)
        assert not bool(captured.validity.cpu()), name
        torch.testing.assert_close(captured.replay_checked(hidden, groups), recovered, rtol=0, atol=0, msg=name)
    print(
        "VQ2A8_GS32_DEVICE_BANK_GRAPH="
        + json.dumps(
            {
                "group_type": group_type,
                "physical_experts": selected_spec.expert_ids,
                "logical_experts": spec.expert_ids,
                "captures": 1,
                "replays": captured.replays,
                "invalid_cases": list(bad_cases),
                "status": "PASS",
                "scope": "three-expert device-route graph correctness; not model or serving performance",
            }
        ),
        flush=True,
    )


@pytest.mark.parametrize(
    ("group_type", "fusion"),
    ((0, "none"), (1, "none"), (1, "swiglu_quant")),
    ids=("prefix", "counts", "counts_swiglu_quant"),
)
def test_gs32_device_bank_decode_graph_buckets_and_recovery(device_bank_runtime, group_type, fusion):
    """Exercise the same large-row graph path used by B1..B64 decoding.

    One shared weight owner retains every captured bucket. EP padding permits
    an empty local rank and typed banks with no live rows, while hot experts
    deliberately exceed the old 128-row bounded-router limit.
    """
    runtime = device_bank_runtime
    torch, spec = runtime.torch, runtime.spec
    # Share the fixture's resident tensors, but use an independent EP owner so
    # the pre-existing strict-total acceptance tests retain their contract.
    layer = SimpleNamespace(**vars(runtime.layer), vq2a8_ep_size=2)
    bank = runtime.bank_type(
        layer,
        spec,
        {**runtime.paths, "gs32_execution": "full_graph", "gs32_fusion": fusion},
        graph_row_capacity=max(DECODE_ROUTED_CAPACITIES),
    )
    baselines = {
        mode: runtime.bank_type(layer, spec, {**runtime.paths, "gs32_execution": mode, "gs32_fusion": fusion})
        for mode in ("serial", "fused")
    }
    generator = torch.Generator(device="cpu").manual_seed(864512)
    originals, captures = {}, {}
    shared_validity = bank.validity_buffer
    for capacity in DECODE_ROUTED_CAPACITIES:
        host = (torch.randn(capacity, spec.specs["gate_up"].columns, generator=generator) * 0.1).bfloat16()
        hidden = host.to(runtime.device)
        counts = (capacity // 3, capacity // 3, capacity - 2 * (capacity // 3))
        groups = route_counts(torch, layer, spec, counts, runtime.device, group_type)
        originals[capacity] = host
        captures[capacity] = CapturedBank(runtime, bank, hidden, groups, group_type, validity=shared_validity)
    retained_captures = len(bank.capture_owners)
    assert retained_captures == len(captures), "Each bucket must retain its own launch buffers."
    assert len({item.output.data_ptr() for item in captures.values()}) == len(captures)

    # Alternate large and small graphs after all captures exist. This catches
    # a later capture replacing an earlier bucket's metadata or output owner.
    order = (512, 6, 384, 12, 192, 24, 96, 48, 6, 512)
    report = []
    for step, capacity in enumerate(order):
        captured = captures[capacity]
        hidden = (originals[capacity].float() + step / 128).bfloat16().to(runtime.device)
        cases = {
            "balanced": (capacity // 3, capacity // 3, capacity - 2 * (capacity // 3)),
            "first_hot": (capacity, 0, 0),
            "middle_hot": (0, capacity, 0),
            "last_hot": (0, 0, capacity),
            "padded": (capacity // 2, 0, 1),
            "empty_local_rank": (0, 0, 0),
        }
        for name, counts in cases.items():
            groups = route_counts(torch, layer, spec, counts, runtime.device, group_type)
            expected = reference_output(runtime, hidden, groups, group_type, baselines)
            actual = captured.replay_checked(hidden, groups)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=f"R={capacity}: {name}")
            active_rows = sum(counts)
            assert torch.count_nonzero(actual[active_rows:]) == 0, (capacity, name)
        report.append({"capacity": capacity, "patterns": list(cases)})

    invalid_cases = []
    for capacity in FAULT_ROUTED_CAPACITIES:
        captured = captures[capacity]
        hidden = originals[capacity].to(runtime.device)
        groups = route_counts(torch, layer, spec, (capacity, 0, 0), runtime.device, group_type)
        expected = reference_output(runtime, hidden, groups, group_type, baselines)
        nan_hidden = hidden.clone()
        nan_hidden[0, 0] = float("nan")
        missing = torch.zeros(layer.vq2a8_num_experts, dtype=torch.int64)
        missing[runtime.missing_expert] = capacity
        if group_type == 0:
            missing = missing.cumsum(0)
        bad_cases = {
            "nan_hidden": (nan_hidden, groups),
            "missing_expert": (hidden, missing.to(runtime.device)),
            "negative_count_or_decreasing_prefix": (
                hidden,
                route_counts(torch, layer, spec, (-1, capacity, 1), runtime.device, group_type),
            ),
            "capacity_overflow": (
                hidden,
                route_counts(torch, layer, spec, (capacity + 1, 0, 0), runtime.device, group_type),
            ),
        }
        for name, (bad_hidden, bad_groups) in bad_cases.items():
            with pytest.raises(ValueError, match="GS32 graph replay failed deferred validity checks"):
                captured.replay_checked(bad_hidden, bad_groups)
            assert not bool(captured.validity.cpu()), (capacity, name)
            recovered = captured.replay_checked(hidden, groups)
            torch.testing.assert_close(recovered, expected, rtol=0, atol=0, msg=f"R={capacity}: recover {name}")
        invalid_cases.append({"capacity": capacity, "faults": list(bad_cases)})
    # Production keeps one validity scalar per FFN owner across all graphs.
    # A fault in the largest bucket must not contaminate a smaller graph.
    largest_capacity = max(DECODE_ROUTED_CAPACITIES)
    largest = captures[largest_capacity]
    largest_nan = originals[largest_capacity].to(runtime.device)
    largest_nan[0, 0] = float("nan")
    largest_groups = route_counts(torch, layer, spec, (largest_capacity, 0, 0), runtime.device, group_type)
    with pytest.raises(ValueError, match="GS32 graph replay failed deferred validity checks"):
        largest.replay_checked(largest_nan, largest_groups)
    smallest_capacity = min(DECODE_ROUTED_CAPACITIES)
    smallest_hidden = originals[smallest_capacity].to(runtime.device)
    smallest_groups = route_counts(torch, layer, spec, (0, 0, smallest_capacity), runtime.device, group_type)
    expected = reference_output(runtime, smallest_hidden, smallest_groups, group_type, baselines)
    recovered = captures[smallest_capacity].replay_checked(smallest_hidden, smallest_groups)
    torch.testing.assert_close(recovered, expected, rtol=0, atol=0)
    timing = {}
    for capacity, captured in captures.items():
        hidden = originals[capacity].to(runtime.device)
        groups = route_counts(torch, layer, spec, (capacity, 0, 0), runtime.device, group_type)
        expected = reference_output(runtime, hidden, groups, group_type, baselines)
        samples = []
        for _ in range(BUCKET_TIMING_REPEATS):
            actual = captured.replay_checked(hidden, groups, timing_ms=samples)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        timing[capacity] = {"npu_event_median_ms": statistics.median(samples), "samples_ms": samples}
    assert len(bank.capture_owners) == retained_captures, "Replays must not append graph owners."
    print(
        "VQ2A8_GS32_DEVICE_BANK_DECODE_BUCKETS="
        + json.dumps(
            {
                "group_type": group_type,
                "fusion": fusion,
                "capacities": list(captures),
                "captures": len(captures),
                "replays": {capacity: item.replays for capacity, item in captures.items()},
                "projection_bank_types": {kind: sorted(groups) for kind, groups in bank.banks.items()},
                "cases": report,
                "invalid_cases": invalid_cases,
                "graph_replay_timing": timing,
                "status": "PASS",
                "scope": "shared three-expert FFN graph buckets; no whole-model batch or throughput claim",
            }
        ),
        flush=True,
    )
