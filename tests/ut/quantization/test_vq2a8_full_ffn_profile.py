# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for profile selection and measurement bookkeeping, not NPU proof."""

from types import SimpleNamespace

import pytest
import torch

from tests.e2e.nightly.vq2a8_gs32_full_ffn_helpers import (
    full_inventory_route_patterns,
    localize_profile_layer,
    measure_npu_submissions,
    npu_memory_snapshot,
    select_full_ep_inventory,
    tensor_storage_summary,
    timing_statistics,
)


def test_full_ep_selection_keeps_entire_rank_and_sparse_source_positions():
    entries = {
        0: {
            "gate_up": {0: "vq2", 4: "vq2", 5: "vq2", 7: "mxfp4"},
            "down": {0: "vq2", 4: "vq2", 5: "mxfp4", 7: "mxfp4"},
        },
        1: {"gate_up": {0: "vq2", 1: "vq2"}, "down": {0: "vq2", 1: "vq2"}},
    }
    selected = select_full_ep_inventory(entries, 16)
    assert selected == {
        "layer": 0,
        "ep_size": 4,
        "ep_rank": 1,
        "num_global_experts": 16,
        "num_local_experts": 4,
        "expert_start": 4,
        "global_ids": (4, 5, 7),
        "local_ids": (0, 1, 3),
        "positions": (1, 2, 3),
        "precision_pairs": {"vq2_to_vq2": 1, "vq2_to_mxfp4": 1, "mxfp4_to_mxfp4": 1},
    }
    # A dense four-expert rank takes precedence over a convenient three-pair
    # subset: the purpose is to profile the actual complete local inventory.
    entries[2] = {kind: {index: "vq2" for index in range(4)} for kind in ("gate_up", "down")}
    assert select_full_ep_inventory(entries, 16)["global_ids"] == (0, 1, 2, 3)


@pytest.mark.parametrize("num_experts,ep_size", [(0, 4), (15, 4), (16, 0), (16, True), (True, 1)])
def test_full_ep_selection_rejects_invalid_partition(num_experts, ep_size):
    with pytest.raises(ValueError, match="partition"):
        select_full_ep_inventory({}, num_experts, ep_size=ep_size)


@pytest.mark.parametrize("entry", [{"gate_up": {0: "vq2"}, "down": {}}, {"gate_up": {16: "vq2"}, "down": {16: "vq2"}}])
def test_full_ep_selection_rejects_inconsistent_inventory(entry):
    with pytest.raises(ValueError, match="inventories"):
        select_full_ep_inventory({0: entry}, 16)


def test_full_ep_selection_does_not_invent_inventory():
    assert select_full_ep_inventory({}, 256) is None
    entry = {kind: {100: "vq2", 120: "vq2"} for kind in ("gate_up", "down")}
    selected = select_full_ep_inventory({0: entry}, 256)
    assert selected["global_ids"] == (100, 120)
    assert selected["local_ids"] == (36, 56)
    assert selected["num_local_experts"] == 64


def test_localization_matches_runtime_numbering_without_reordering_source_fields():
    layer = SimpleNamespace(vq2a8_num_experts=16, w13_packed_zn=torch.arange(8))
    original_tensor = layer.w13_packed_zn
    spec = SimpleNamespace(
        expert_ids=(4, 7),
        specs={"gate_up": object()},
        quant_types={"gate_up": {4: "vq2", 7: "mxfp4"}, "down": {4: "mxfp4", 7: "mxfp4"}},
    )
    placement = {
        "global_ids": (4, 7),
        "local_ids": (0, 3),
        "expert_start": 4,
        "num_local_experts": 4,
        "ep_size": 4,
        "ep_rank": 1,
    }
    local = localize_profile_layer(layer, spec, placement)
    assert local.expert_ids == (0, 3)
    assert local.quant_types["gate_up"] == {0: "vq2", 3: "mxfp4"}
    assert local.specs is spec.specs
    assert spec.expert_ids == (4, 7)
    assert layer.vq2a8_num_experts == 4 and layer.vq2a8_ep_rank == 1
    assert layer.w13_packed_zn is original_tensor
    with pytest.raises(ValueError, match="complete"):
        localize_profile_layer(layer, spec, {**placement, "global_ids": (4,)})


@pytest.mark.parametrize("rows", [1, 6, 384, 4096])
def test_profile_patterns_cover_inventory_and_real_precision_pairs(rows):
    spec = SimpleNamespace(
        expert_ids=(1, 4, 9),
        quant_types={"gate_up": {1: "vq2", 4: "vq2", 9: "mxfp4"}, "down": {1: "vq2", 4: "mxfp4", 9: "mxfp4"}},
    )
    patterns = dict(full_inventory_route_patterns(spec, rows))
    assert set(patterns) == {"empty", "balanced", "hot_vq2_to_vq2", "hot_vq2_to_mxfp4", "hot_mxfp4_to_mxfp4", "skewed"}
    for name, counts in patterns.items():
        assert len(counts) == 3 and all(type(count) is int and count >= 0 for count in counts)
        assert sum(counts) == (0 if name == "empty" else rows)
    assert max(patterns["balanced"]) - min(patterns["balanced"]) <= 1
    assert patterns["hot_vq2_to_mxfp4"] == (0, rows, 0)


def test_storage_report_deduplicates_base_storage_aliases_references_and_cycles():
    base = torch.arange(8, dtype=torch.float32)
    view, copied = base[2:5], base[2:5].clone()
    opaque = object()
    values = [base, base, {"view": view, "copy": copied, "native_route": opaque}, opaque]
    values.append(values)
    report = tensor_storage_summary(torch, values)
    assert report["tensor_objects"] == 3
    assert report["unique_storages"] == 2
    assert report["storage_bytes"] == 44
    assert report["logical_tensor_bytes"] == 56
    assert report["opaque_objects_excluded"] == 1
    assert tensor_storage_summary(torch, [torch.empty(0)])["storage_bytes"] == 0


@pytest.mark.parametrize("samples", [[], [-1], [float("nan")], [float("inf")]])
def test_profile_rejects_unusable_timing_samples(samples):
    with pytest.raises(ValueError):
        timing_statistics(samples)


def test_profile_statistics_keep_raw_samples_and_explicit_units():
    assert timing_statistics([3.0, 1.0, 2.0]) == {
        "samples_ms": [3.0, 1.0, 2.0],
        "min_ms": 1.0,
        "median_ms": 2.0,
        "max_ms": 3.0,
    }


def test_measurement_bookkeeping_synchronizes_and_checks_outside_submissions():
    calls = []

    class Event:
        def record(self):
            calls.append("event")

        def elapsed_time(self, end):
            assert isinstance(end, Event)
            return 0.5

    def counter(value):
        def get(device):
            assert device == "npu:0"
            return value

        return get

    npu = SimpleNamespace(
        memory_allocated=counter(16),
        memory_reserved=counter(32),
        max_memory_allocated=counter(24),
        max_memory_reserved=counter(48),
        synchronize=lambda: calls.append("sync"),
        reset_peak_memory_stats=lambda device: calls.append("reset"),
        Event=lambda **kwargs: Event(),
    )
    runtime = SimpleNamespace(torch=SimpleNamespace(npu=npu), device="npu:0")
    report = measure_npu_submissions(
        runtime, lambda: calls.append("submit"), lambda: calls.append("check"), warmups=2, repeats=3
    )
    assert calls.count("submit") == 5 and calls.count("sync") == calls.count("check") == 2
    assert calls[:4] == ["submit", "submit", "sync", "check"]
    assert calls[-2:] == ["sync", "check"]
    assert report["device"]["samples_ms"] == [0.5] * 3
    assert len(report["host_submission"]["samples_ms"]) == 3
    assert report["memory_before"] == npu_memory_snapshot(runtime.torch, runtime.device)
    assert report["memory_after"] == {
        "allocated_bytes": 16,
        "reserved_bytes": 32,
        "peak_allocated_bytes": 24,
        "peak_reserved_bytes": 48,
    }


@pytest.mark.parametrize("warmups,repeats", [(0, 3), (3, 0), (-1, 3), (True, 3)])
def test_measurement_rejects_missing_warmup_or_samples(warmups, repeats):
    with pytest.raises(ValueError, match="positive"):
        measure_npu_submissions(None, None, None, warmups=warmups, repeats=repeats)
