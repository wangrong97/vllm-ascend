# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-NPU routing and replay, with independent Python-integer expected rows.

This suite never runs a native projection. It validates the device routing
boundary and its deferred flags; the native route sanitizer is tested separately.
"""

import json
import statistics
from time import perf_counter
from types import SimpleNamespace

import pytest

from tests.e2e.nightly.test_vq2a8_gs32_activation import activation_runtime  # noqa: F401
from tests.ut.quantization.test_vq2a8_gs32_device_routes import route_reference

pytestmark = pytest.mark.filterwarnings(r"error:tl\.where with a non-boolean condition is deprecated.*:UserWarning")

PREFILL_BENCHMARK_ROWS = 32768
PREFILL_BENCHMARK_WARMUPS = 3
PREFILL_BENCHMARK_REPEATS = 5


@pytest.fixture(scope="module")
def route_runtime(request):
    runtime = request.getfixturevalue("activation_runtime")

    from vllm_ascend.ops.fused_moe.vq2a8_gs32_routes import prepare_identity_metadata, prepare_metadata, restore

    return SimpleNamespace(
        **vars(runtime), metadata=prepare_metadata, identity_metadata=prepare_identity_metadata, restore=restore
    )


@pytest.mark.parametrize("group_type", [0, 1])
@pytest.mark.parametrize("dtype_name", ["int32", "int64"])
def test_identity_metadata_graph_matches_general_route_oracle_with_holes(route_runtime, group_type, dtype_name):
    runtime, torch = route_runtime, route_runtime.torch
    capacity, ids, slots = 129, [0, 2, 4], [0, -1, 1, -1, 2]
    hidden = torch.zeros(capacity, 64, dtype=torch.bfloat16, device=runtime.device)
    expert_ids = torch.tensor(ids, dtype=torch.int64, device=runtime.device)
    logical_slots = torch.tensor(slots, dtype=torch.int64, device=runtime.device)
    groups = torch.zeros(5, dtype=getattr(torch, dtype_name), device=runtime.device)

    def forward():
        return runtime.identity_metadata(hidden, groups, group_type, expert_ids, logical_slots, allow_padding=True)

    for _ in range(2):
        forward()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        counts, flags, summary = forward()
    addresses = tuple(value.data_ptr() for value in (counts, flags, summary))
    cases = (
        [0, 0, 0, 0, 0],
        [2, 0, 3, 0, 1],
        [0, 0, 129, 0, 0],
        [2, 1, 3, 0, 0],
        [-1, 0, 2, 0, 0],
        [130, 0, 0, 0, 0],
        [64, 0, 66, 0, 0],
        [0, 0, 0, 0, 128],
    )
    for raw in cases:
        host = torch.tensor(raw, dtype=getattr(torch, dtype_name))
        if group_type == 0:
            host = host.cumsum(0).to(host.dtype)
        groups.copy_(host)
        graph.replay()
        torch.npu.synchronize()
        physical, gather, inverse, valid = route_reference(
            host.tolist(), group_type, ids, slots, capacity, allow_padding=True, max_count=None
        )
        assert counts.cpu().tolist() == physical
        assert flags.cpu().tolist() == [valid]
        assert summary.cpu().tolist() == [sum(physical), valid]
        generic = runtime.metadata(hidden, groups, group_type, expert_ids, logical_slots, allow_padding=True)
        assert torch.equal(counts.cpu(), generic[0].cpu())
        assert torch.equal(flags.cpu(), generic[3].cpu())
        if valid:
            active = sum(physical)
            assert gather == inverse == [*range(active), *([-1] * (capacity - active))]
            assert generic[1].cpu().tolist() == gather
            assert generic[2].cpu().tolist() == inverse
        assert addresses == tuple(value.data_ptr() for value in (counts, flags, summary))


def test_identity_rejects_reordered_or_subset_inventory_and_recovers_on_device(route_runtime):
    runtime, torch = route_runtime, route_runtime.torch
    hidden = torch.zeros(8, 64, dtype=torch.bfloat16, device=runtime.device)
    groups = torch.tensor([2, 0, 3], dtype=torch.int64, device=runtime.device)
    experts = torch.tensor([0, 2], dtype=torch.int64, device=runtime.device)
    slots = torch.tensor([0, -1, 1], dtype=torch.int64, device=runtime.device)
    for ids, mapping, values, expected in (
        ([2, 0], [1, -1, 0], [2, 0, 3], 0),
        ([0, 2], [0, -1, 1], [2, 1, 2], 0),
        ([0, 0], [0, -1, -1], [2, 0, 0], 0),
        ([0, 2], [0, -1, 1], [2, 0, 3], 1),
    ):
        experts.copy_(torch.tensor(ids))
        slots.copy_(torch.tensor(mapping))
        groups.copy_(torch.tensor(values))
        counts, flags, metadata = runtime.identity_metadata(hidden, groups, 1, experts, slots, allow_padding=True)
        assert flags.cpu().tolist() == [expected]
        if expected:
            assert counts.cpu().tolist() == [2, 3] and metadata.cpu().tolist() == [5, 1]
        else:
            assert not counts.cpu().count_nonzero() and metadata.cpu().tolist() == [0, 0]
    # A general subset is still valid, but its row permutation is not identity.
    experts.copy_(torch.tensor([2, 0]))
    slots.copy_(torch.tensor([1, -1, 0]))
    groups.copy_(torch.tensor([2, 3, 1]))
    generic = runtime.metadata(hidden, groups, 1, experts, slots, allow_padding=True, subset=True)
    assert generic[3].cpu().tolist() == [1]
    assert generic[1][:3].cpu().tolist() == [5, 0, 1]


@pytest.fixture(scope="module")
def native_route_runtime(request, route_runtime):
    from tests.e2e.nightly.test_vq2a8_gs32_full_ffn import selected_paths

    paths = selected_paths(request)
    route_runtime.torch.ops.load_library(paths["gs32_library"])
    from vllm_ascend.ops.fused_moe import vq2a8_gs32_counts  # noqa: F401

    return route_runtime


@pytest.mark.parametrize("allow_padding", [False, True])
def test_native_counts_owned_copy_exact_status_and_graph_replay(native_route_runtime, allow_padding):
    runtime, torch = native_route_runtime, native_route_runtime.torch
    ops = torch.ops.vq2a8_gs32_stage
    hidden = torch.zeros(8, 64, dtype=torch.uint8, device=runtime.device)
    counts = torch.tensor([3, 0, 5], dtype=torch.int64, device=runtime.device)
    safe, flags = ops._sanitize_device_counts(counts, 8, allow_padding)
    counts.zero_()
    assert safe.cpu().tolist() == [3, 0, 5], "The sanitizer must own counts independently of caller mutations."
    assert flags.cpu().tolist() == [1]
    counts.copy_(torch.tensor([3, 0, 5]))
    handle = ops.prepare_device_routes(hidden, counts, allow_padding)
    external_status = ops.device_routes_status(handle)
    external_status.zero_()
    assert ops.device_routes_status(handle).cpu().tolist() == [1], (
        "Status reads must not expose mutable internal storage."
    )
    for _ in range(2):
        ops.prepare_device_routes(hidden, counts, allow_padding)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        graph_safe, graph_flag = ops._sanitize_device_counts(counts, 8, allow_padding)
        graph_handle = ops.prepare_device_routes(hidden, counts, allow_padding)
        graph_status = ops.device_routes_status(graph_handle)
    for values in ([3, 0, 5], [-1, 4, 5], [9, 0, 0], [4, 0, 5], [0, 0, 0], [1, 0, 2], [0, 0, 8]):
        counts.copy_(torch.tensor(values, dtype=torch.int64))
        graph.replay()
        torch.npu.synchronize()
        valid = all(0 <= value <= 8 for value in values) and (sum(values) <= 8 if allow_padding else sum(values) == 8)
        assert graph_safe.cpu().tolist() == (values if valid else [0] * len(values))
        assert graph_flag.cpu().tolist() == graph_status.cpu().tolist() == [int(valid)]
    for values, valid in (([2**63 - 1, 0], True), ([2**63 - 1, 1], False), ([2**63 - 1] * 256, False)):
        extreme = torch.tensor(values, dtype=torch.int64, device=runtime.device)
        safe, flags = ops._sanitize_device_counts(extreme, 2**63 - 1, True)
        assert flags.cpu().tolist() == [int(valid)]
        assert safe.cpu().tolist() == (values if valid else [0] * len(values))

    bad_dtype = counts.to(torch.int32)
    with pytest.raises((ValueError, RuntimeError)):
        ops._sanitize_device_counts(bad_dtype, 8, allow_padding)
    with pytest.raises((ValueError, RuntimeError)):
        ops.prepare_device_routes(hidden, bad_dtype, allow_padding)


@pytest.mark.parametrize(
    "counts",
    [
        [0, 0, 0],
        [0, 65535, 65536, 2**24 - 1, 2**24 + 1, 2**31 + 1, 2**32 + 3, 2**48 + 5, 2**62 - 5],
        [2**63 - 1, 0],
        [2**63 - 1, 1],
        [2**63 - 1] * 256,
    ],
    ids=["empty", "exact-bit-boundaries", "int64-max", "sum-overflow", "256-expert-overflow"],
)
def test_prefill_metadata_int64_compilation_and_exactness(route_runtime, counts):
    """Compile production helpers with tiny metadata, without allocating routed rows.

    Large counts exercise exact gather and overflow rejection, not a claim that
    tensors with these capacities fit on the NPU. No weight/native library load.
    """
    from vllm.triton_utils import tl, triton

    from vllm_ascend.ops.triton.activation import vq2a8_gs32_routes as kernels

    runtime, torch = route_runtime, route_runtime.torch
    # Annotation-only globals can pass lookup yet fail inside static_range.
    assert isinstance(kernels.INT64_GATHER_BITS, tl.constexpr)
    assert isinstance(kernels.INT64_SUM_BITS, tl.constexpr)
    experts = list(reversed(range(len(counts))))
    device_counts = torch.tensor(counts, dtype=torch.int64, device=runtime.device)
    device_experts = torch.tensor(experts, dtype=torch.int64, device=runtime.device)
    # Reverse permutation is its own inverse.
    outputs = [torch.empty_like(device_counts) for _ in range(4)]
    summary = torch.empty(2, dtype=torch.int64, device=runtime.device)
    flags = torch.empty(1, dtype=torch.int32, device=runtime.device)
    rows = min(sum(counts), 2**63 - 1)
    block = triton.next_power_of_2(len(counts))
    kernels._prefill_metadata_kernel[(1,)](
        device_counts,
        device_experts,
        device_experts,
        *outputs,
        summary,
        flags,
        ROWS=rows,
        EXPERTS=len(counts),
        LOGICAL=len(counts),
        GROUP_TYPE=1,
        LOGICAL_BLOCK=block,
        EXPERT_BLOCK=block,
        ROW_BLOCK=kernels.PREFILL_ROW_BLOCK,
        ALLOW_PADDING=True,
        SUBSET=False,
        enable_auto_vectorize_v2=False,
    )
    valid = sum(counts) <= rows
    physical = [counts[index] for index in experts]
    tiles = [(count + kernels.PREFILL_ROW_BLOCK - 1) // kernels.PREFILL_ROW_BLOCK for count in physical]
    expected = [
        physical,
        [sum(counts[:index]) for index in experts],
        [sum(physical[:index]) for index in range(len(counts))],
        [sum(tiles[: index + 1]) for index in range(len(counts))],
    ]
    assert flags.cpu().tolist() == [int(valid)]
    for output, values in zip(outputs, expected, strict=True):
        assert output.cpu().tolist() == (values if valid else [0] * len(counts))
    assert summary.cpu().tolist() == ([sum(counts), 1] if valid else [0, 0])
    restored = torch.empty(2, dtype=torch.int64, device=runtime.device)
    route_flags = torch.ones(1, dtype=torch.int32, device=runtime.device)
    kernels._restore_metadata_kernel[(1,)](
        device_counts, route_flags, restored, ROWS=rows, EXPERTS=len(counts), EXPERT_BLOCK=block
    )
    assert restored.cpu().tolist() == ([sum(counts), 1] if valid else [0, 0])


@pytest.mark.parametrize("group_type", [0, 1])
@pytest.mark.parametrize("capacity", [1025, 32769])
def test_prefill_metadata_and_restore_graph_large_rows_empty_bank_and_recovery(route_runtime, group_type, capacity):
    runtime, torch = route_runtime, route_runtime.torch
    width = 64
    experts = [2, 0]
    host_hidden = torch.arange(capacity * width).reshape(capacity, width).remainder(89).bfloat16()
    host_experts = torch.tensor(experts, dtype=torch.int64)
    host_slots = torch.tensor([1, -1, 0], dtype=torch.int64)

    def group_values(counts):
        value = torch.tensor(counts, dtype=torch.int64)
        return value.cumsum(0) if group_type == 0 else value

    initial = [capacity - 130, 1, 129]
    hidden = host_hidden.to(runtime.device)
    groups = group_values(initial).to(runtime.device)
    expert_ids, logical_slots = host_experts.to(runtime.device), host_slots.to(runtime.device)

    def forward():
        metadata = runtime.metadata(
            hidden, groups, group_type, expert_ids, logical_slots, allow_padding=True, subset=True
        )
        counts, gather, inverse, route_flags, summary = metadata
        # Simulate native writes only to the live physical prefix. Poison all
        # other output rows to detect an invalid pre-mask finiteness check.
        physical = hidden.index_select(0, gather.clamp_min(0))
        physical.masked_fill_((gather < 0).unsqueeze(1), float("nan"))
        result = torch.full_like(hidden, 7)
        status = runtime.restore(physical, inverse, counts, route_flags, result, metadata=summary)
        return metadata, result, status

    stream = torch.npu.Stream(device=runtime.device)
    stream.wait_stream(torch.npu.current_stream(runtime.device))
    with torch.npu.stream(stream):
        for _ in range(2):
            forward()
    stream.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        metadata, restored, status = forward()
    stream.synchronize()
    cases = [
        (initial, None),
        ([0, capacity, 0], None),  # Resident bank receives no rows.
        ([0, 0, 0], None),  # Empty local dispatch with a retained capacity.
        ([129, 0, capacity - 129], -(2**63)),
        ([129, 0, capacity - 129], 2**63 - 1),
        ([1, 2, capacity - 3], None),
    ]
    for index, (counts, corrupt) in enumerate(cases):
        host_groups = group_values(counts)
        if corrupt is not None:
            host_groups[0] = corrupt
        refreshed_hidden = host_hidden + index
        with torch.npu.stream(stream):
            hidden.copy_(refreshed_hidden)
            groups.copy_(host_groups)
            graph.replay()
        stream.synchronize()
        expected_counts, gather, inverse, valid = route_reference(
            host_groups.tolist(),
            group_type,
            experts,
            host_slots.tolist(),
            capacity,
            allow_padding=True,
            subset=True,
            max_count=None,
        )
        assert metadata[0].cpu().tolist() == expected_counts
        assert metadata[1].cpu().tolist() == (gather if valid else [-1] * capacity)
        assert metadata[2].cpu().tolist() == inverse
        assert metadata[3].cpu().tolist() == [valid]
        assert metadata[4].cpu().tolist() == [sum(expected_counts), valid]
        assert bool(status.cpu().all()) == bool(valid)
        expected = torch.full_like(host_hidden, 7)
        selected = torch.tensor(inverse) >= 0
        expected[selected] = refreshed_hidden[selected]
        assert torch.equal(restored.cpu(), expected)
    torch.npu.current_stream(runtime.device).wait_stream(stream)


@pytest.mark.parametrize("fault", [None, "live_nan", "live_inf", "inverse_oob", "inverse_negative", "route"])
@pytest.mark.parametrize("width", [64, 4097])
@pytest.mark.parametrize("reuse_metadata", [False, True])
def test_prefill_restore_masks_poisoned_tails_and_rejects_invalid_live_rows(
    route_runtime, fault, width, reuse_metadata
):
    from vllm_ascend.ops.triton.activation.vq2a8_gs32_routes import RESTORE_COLUMN_BLOCK

    runtime, torch = route_runtime, route_runtime.torch
    output = torch.full((513, width), float("nan"), dtype=torch.bfloat16, device=runtime.device)
    output[:2].fill_(3)
    inverse = torch.full((513,), -1, dtype=torch.int64, device=runtime.device)
    inverse[129], inverse[512] = 1, 0
    counts = torch.tensor([0, 2, 0], dtype=torch.int64, device=runtime.device)
    flags = torch.ones(1, dtype=torch.int32, device=runtime.device)
    if fault in ("live_nan", "live_inf"):
        output[1, width - 1] = float("nan") if fault == "live_nan" else float("inf")
    elif fault == "inverse_oob":
        inverse[129] = 2**63 - 1
    elif fault == "inverse_negative":
        inverse[129] = -(2**63)
    elif fault == "route":
        flags.zero_()
    result = torch.full_like(output, 7)
    summary = None
    if reuse_metadata:
        summary = torch.tensor([0, 0] if fault == "route" else [2, 1], dtype=torch.int64, device=runtime.device)
    status = runtime.restore(output, inverse, counts, flags, result, metadata=summary)
    assert status.numel() == output.shape[0] * ((width + RESTORE_COLUMN_BLOCK - 1) // RESTORE_COLUMN_BLOCK)
    assert bool(status.cpu().all()) == (fault is None)
    expected = torch.full_like(output.cpu(), 7)
    if fault != "route":
        expected[512] = 3
        if fault not in ("inverse_oob", "inverse_negative"):
            expected[129] = 3
            if fault in ("live_nan", "live_inf"):
                expected[129, width - 1] = 0
    assert torch.equal(result.cpu(), expected)


@pytest.mark.parametrize("experts", [128, 256])
@pytest.mark.parametrize("pattern", ["balanced", "sparse", "skewed", "empty"])
@pytest.mark.parametrize("group_type", [0, 1])
def test_prefill_metadata_wide_banks_correctness_and_timing(route_runtime, experts, pattern, group_type):
    """Measure warmed metadata work only; no projection, activation or weights."""
    from vllm_ascend.ops.triton.activation import vq2a8_gs32_routes as kernels

    runtime, torch = route_runtime, route_runtime.torch
    rows = PREFILL_BENCHMARK_ROWS
    counts = [0] * experts
    if pattern == "balanced":
        counts = [rows // experts] * experts
    elif pattern == "sparse":
        for index, count in zip((0, experts // 3, 2 * experts // 3, experts - 1), (1, 255, 257, rows - 513)):
            counts[index] = count
    elif pattern == "skewed":
        counts[experts // 2] = rows
    inventory = list(reversed(range(experts)))
    # All experts belong to this bank, in reverse physical order. Empty experts
    # and uneven blocks exercise compact task lookup, including repeated ends.
    host_group = torch.tensor(counts, dtype=torch.int64)
    if group_type == 0:
        host_group = host_group.cumsum(0)
    hidden = torch.empty((rows, 32), dtype=torch.bfloat16, device=runtime.device)
    group = host_group.to(runtime.device)
    expert_ids = torch.tensor(inventory, dtype=torch.int64, device=runtime.device)
    slots = expert_ids.clone()  # Reverse permutation is its own inverse.
    expected = route_reference(
        host_group.tolist(), group_type, inventory, inventory, rows, allow_padding=True, subset=True, max_count=None
    )

    def run():
        return runtime.metadata(hidden, group, group_type, expert_ids, slots, allow_padding=True, subset=True)

    def check(actual):
        for result, reference in zip(actual[:3], expected[:3], strict=True):
            assert result.cpu().tolist() == reference
        assert actual[3].cpu().tolist() == [1]
        assert actual[4].cpu().tolist() == [sum(counts), 1]

    for _ in range(PREFILL_BENCHMARK_WARMUPS):
        check(run())
    samples, wall_samples = [], []
    for _ in range(PREFILL_BENCHMARK_REPEATS):
        begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        torch.npu.synchronize()
        wall_begin = perf_counter()
        begin.record()
        actual = run()
        end.record()
        end.synchronize()
        wall_samples.append((perf_counter() - wall_begin) * 1000)
        samples.append(begin.elapsed_time(end))
        check(actual)
    block = kernels.PREFILL_ROW_BLOCK
    capacity_tiles = (rows + block - 1) // block
    live_tiles = sum((count + block - 1) // block for count in counts)
    print(
        "GS32_PREFILL_METADATA_TIMING="
        + json.dumps(
            {
                "experts": experts,
                "pattern": pattern,
                "group_type": group_type,
                "rows": rows,
                "live_rows": sum(counts),
                "live_tiles": live_tiles,
                "legacy_grid_programs": experts * capacity_tiles,
                "compact_grid_upper_bound": capacity_tiles + experts - 1,
                "npu_event_median_ms": statistics.median(samples),
                "samples_ms": samples,
                "synchronized_wall_median_ms": statistics.median(wall_samples),
                "synchronized_wall_samples_ms": wall_samples,
                "scope": "warm metadata allocation, validation and row mapping; no activation, native/GMM or transfers",
            }
        ),
        flush=True,
    )
