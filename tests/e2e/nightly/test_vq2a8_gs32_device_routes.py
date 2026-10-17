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

    from vllm_ascend.ops.fused_moe.vq2a8_gs32_activation import finish_flags
    from vllm_ascend.ops.fused_moe.vq2a8_gs32_routes import prepare, prepare_metadata, restore

    return SimpleNamespace(
        **vars(runtime), route=prepare, metadata=prepare_metadata, restore=restore, finish=finish_flags
    )


def host_fields(runtime, counts, experts, width=64, group_type=1, dtype=None):
    torch = runtime.torch
    rows, logical = sum(counts), len(counts)
    hidden = torch.arange(rows * width).reshape(rows, width).bfloat16()
    group = torch.tensor(counts, dtype=torch.int64 if dtype is None else dtype)
    if group_type == 0:
        group = group.cumsum(0).to(group.dtype)
    slots = [-1] * logical
    for slot, expert in enumerate(experts):
        slots[expert] = slot
    return hidden, group, torch.tensor(experts, dtype=torch.int64), torch.tensor(slots, dtype=torch.int64)


def enqueue(runtime, tensors, group_type):
    hidden, group, experts, slots = tensors
    return runtime.route(hidden, group, group_type, experts, slots)


def assert_route(runtime, host, group_type, results, *, allow_padding=False, subset=False):
    torch = runtime.torch
    hidden, group, experts, slots = host
    expected = route_reference(
        group.tolist(),
        group_type,
        experts.tolist(),
        slots.tolist(),
        hidden.shape[0],
        allow_padding=allow_padding,
        subset=subset,
    )
    physical, row_slots, source, inverse, valid = expected
    assert results[4].cpu().tolist() == [valid]
    assert results[1].cpu().tolist() == physical
    assert results[2].cpu().tolist() == row_slots
    assert results[3].cpu().tolist() == inverse
    actual = results[0].cpu()
    expected_hidden = torch.zeros_like(hidden)
    if valid:
        for row, source_row in enumerate(source):
            if source_row >= 0:
                expected_hidden[row] = hidden[source_row]
    assert torch.equal(actual, expected_hidden)
    if valid and not allow_padding and not subset:
        assert torch.equal(actual[inverse], hidden)


@pytest.mark.parametrize("group_type", [0, 1])
@pytest.mark.parametrize("dtype_name", ["int32", "int64"])
@pytest.mark.parametrize(
    "counts,experts",
    [
        ([1], [0]),
        ([1, 0, 2, 0], [2, 0, 3]),
        ([0, 0, 0, 8], [3, 2, 0]),
        # Exercise masked tails on both lookup tables, including unequal
        # power-of-two blocks (logical 5 -> 8, physical 3 -> 4).
        ([1, 0, 2, 0, 1], [4, 0, 2]),
        ([0, 1, 0, 2, 0, 1, 2], [6, 0, 3, 1, 5]),
        # Isolate row tails from expert tails: the original abort had both
        # ROWS=3 and EXPERTS=3, so either could have supplied its 3 -> 4 slice.
        ([1, 0, 2, 0], [2, 0, 3, 1]),  # ROWS=3, EXPERTS=4.
        ([1, 0, 3, 0], [2, 0, 3]),  # ROWS=4, EXPERTS=3.
        ([1, 0, 4, 0], [2, 0, 3, 1]),  # ROWS=5, EXPERTS=4.
        ([2, 0, 5, 0], [2, 0, 3, 1]),  # ROWS=7, EXPERTS=4.
    ],
)
def test_gs32_device_routes_match_cpu_order(route_runtime, group_type, dtype_name, counts, experts):
    runtime = route_runtime
    host = host_fields(runtime, counts, experts, group_type=group_type, dtype=getattr(runtime.torch, dtype_name))
    device = tuple(value.to(runtime.device) for value in host)
    assert_route(runtime, host, group_type, enqueue(runtime, device, group_type))


def test_gs32_device_routes_maximum_bank_and_width(route_runtime):
    runtime = route_runtime
    counts = [0] * 256
    for expert in (0, 3, 31, 95, 127, 200, 254, 255):
        counts[expert] = 1
    host = host_fields(runtime, counts, list(reversed(range(256))), width=4096)
    device = tuple(value.to(runtime.device) for value in host)
    assert_route(runtime, host, 1, enqueue(runtime, device, 1))


@pytest.mark.parametrize("capacity", [6, 256])
def test_gs32_routes_runtime_expert_bounds_subset_graph_recovery(route_runtime, capacity):
    runtime, torch = route_runtime, route_runtime.torch
    logical = 128
    for expert_count in (1, 2, 3, 63, 64, 65, 127, 128):
        experts = list(reversed(range(expert_count)))
        slots = torch.full((logical,), -1, dtype=torch.int64)
        for slot, expert in enumerate(experts):
            slots[expert] = slot
        counts = torch.zeros(logical, dtype=torch.int64)
        counts[0] += 1
        counts[expert_count - 1] += 2
        if expert_count < logical:
            counts[-1] = 1  # Active route owned by another quantization bank.
        hidden = torch.arange(capacity * 64).reshape(capacity, 64).bfloat16()
        host = (hidden, counts, torch.tensor(experts, dtype=torch.int64), slots)
        tensors = tuple(value.to(runtime.device) for value in host)

        def forward(tensors=tensors):
            return runtime.route(*tensors[:2], 1, *tensors[2:], allow_padding=True, subset=True)

        stream = torch.npu.Stream(device=runtime.device)
        stream.wait_stream(torch.npu.current_stream(runtime.device))
        with torch.npu.stream(stream):
            for _ in range(2):
                forward()
        stream.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            result = forward()
        stream.synchronize()
        for index, state in enumerate(("valid", "bad_slot", "empty", "recovered")):
            refreshed = [value.clone() for value in host]
            refreshed[0].add_(index)
            if state == "bad_slot":
                refreshed[3][0] = expert_count
            elif state == "empty":
                refreshed[1].zero_()
            with torch.npu.stream(stream):
                for target, value in zip(tensors, refreshed, strict=True):
                    target.copy_(value)
                graph.replay()
            stream.synchronize()
            assert_route(runtime, refreshed, 1, result, allow_padding=True, subset=True)
            assert result[4].cpu().tolist() == [int(state != "bad_slot")]


@pytest.mark.parametrize("group_type", [0, 1])
@pytest.mark.parametrize("dtype_name", ["int32", "int64"])
@pytest.mark.parametrize(
    "counts,experts",
    [
        ([32] * 6, [5, 3, 1, 4, 2, 0]),
        ([32] * 8, list(reversed(range(8)))),
        ([128, 0, 64], [2, 1, 0]),
        ([0, 7, 0, 31, 0, 25, 128], [6, 3, 1, 5, 0]),
        ([1] * 256, list(reversed(range(256)))),
    ],
)
def test_gs32_device_routes_prefill_matches_cpu_order(route_runtime, group_type, dtype_name, counts, experts):
    runtime = route_runtime
    host = host_fields(runtime, counts, experts, group_type=group_type, dtype=getattr(runtime.torch, dtype_name))
    device = tuple(value.to(runtime.device) for value in host)
    assert_route(runtime, host, group_type, enqueue(runtime, device, group_type))


@pytest.mark.parametrize("group_type", [0, 1])
def test_gs32_device_routes_prefill_graph_dynamic_routes_and_recovery(route_runtime, group_type):
    runtime, torch = route_runtime, route_runtime.torch
    experts = [7, 5, 3, 1, 6, 4, 2, 0]
    host = host_fields(runtime, [32] * 6 + [0, 0], experts, group_type=group_type)
    tensors = tuple(value.to(runtime.device) for value in host)
    stream = torch.npu.Stream(device=runtime.device)
    stream.wait_stream(torch.npu.current_stream(runtime.device))
    with torch.npu.stream(stream):
        for _ in range(2):
            enqueue(runtime, tensors, group_type)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            result = enqueue(runtime, tensors, group_type)
        for index, counts in enumerate(
            ([32] * 6 + [0, 0], [0, 0, 0, 0, 0, 0, 64, 128], [0, 0, 0, 0, 0, 0, 63, 129], [24] * 8)
        ):
            # The third replay is invalid although its total row count remains
            # correct. The fourth must recover with the same captured buffers.
            refreshed = list(host_fields(runtime, counts, experts, group_type=group_type))
            refreshed[0].add_(index)
            for target, source in zip(tensors, refreshed, strict=True):
                target.copy_(source)
            graph.replay()
            assert_route(runtime, refreshed, group_type, result)
        for corrupt in (True, False):
            refreshed = list(host_fields(runtime, [24] * 8, experts, group_type=group_type))
            if corrupt:
                refreshed[2][0] = 2**63 - 1
            for target, source in zip(tensors, refreshed, strict=True):
                target.copy_(source)
            graph.replay()
            assert_route(runtime, refreshed, group_type, result)
    torch.npu.current_stream(runtime.device).wait_stream(stream)


@pytest.mark.parametrize("group_type", [0, 1])
@pytest.mark.parametrize(
    "fault",
    [
        "max",
        "min",
        "negative",
        "too_many",
        "missing",
        "zero",
        "bad_id_min",
        "bad_id_max",
        "bad_id_negative",
        "bad_id_oob",
        "bad_slot_min",
        "bad_slot_max",
        "bad_slot_negative",
        "bad_slot_oob",
        "duplicate",
    ],
)
def test_gs32_device_routes_invalid_data_is_safe_and_recovers(route_runtime, group_type, fault):
    runtime = route_runtime
    host = list(host_fields(runtime, [1, 0, 2, 0], [2, 0, 3], group_type=group_type))
    if fault in ("max", "min", "negative", "too_many"):
        host[1][0] = {"max": 2**63 - 1, "min": -(2**63), "negative": -1, "too_many": 4}[fault]
    elif fault == "missing":
        host[1].copy_(runtime.torch.tensor([0, 1, 3, 3] if group_type == 0 else [0, 1, 2, 0]))
    elif fault == "zero":
        host[1].zero_()
    elif fault.startswith("bad_id_"):
        host[2][0] = {
            "bad_id_min": -(2**63),
            "bad_id_max": 2**63 - 1,
            "bad_id_negative": -1,
            "bad_id_oob": host[1].numel(),
        }[fault]
    elif fault.startswith("bad_slot_"):
        host[3][0] = {
            "bad_slot_min": -(2**63),
            "bad_slot_max": 2**63 - 1,
            "bad_slot_negative": -2,
            "bad_slot_oob": host[2].numel(),
        }[fault]
    else:
        host[2][1] = host[2][0]
    device = tuple(value.to(runtime.device) for value in host)
    results = enqueue(runtime, device, group_type)
    assert_route(runtime, host, group_type, results)
    assert results[4].cpu().tolist() == [0]
    fresh = host_fields(runtime, [1, 0, 2, 0], [2, 0, 3], group_type=group_type)
    assert_route(
        runtime, fresh, group_type, enqueue(runtime, tuple(value.to(runtime.device) for value in fresh), group_type)
    )


@pytest.mark.parametrize("group_type", [0, 1])
def test_gs32_device_routes_graph_dynamic_counts_and_status_recovery(route_runtime, group_type):
    runtime, torch = route_runtime, route_runtime.torch
    host = host_fields(runtime, [1, 0, 2, 0], [2, 0, 3], group_type=group_type)
    tensors = tuple(value.to(runtime.device) for value in host)
    stream = torch.npu.Stream(device=runtime.device)
    stream.wait_stream(torch.npu.current_stream(runtime.device))
    with torch.npu.stream(stream):
        flags = torch.ones(3, dtype=torch.int32, device=runtime.device)
        native = torch.ones(1, dtype=torch.int32, device=runtime.device)
        for _ in range(2):
            result = enqueue(runtime, tensors, group_type)
            runtime.finish(result[0], flags, flags, route_flags=result[4], native_flags=native).cpu()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            result = enqueue(runtime, tensors, group_type)
            combined = runtime.finish(result[0], flags, flags, route_flags=result[4], native_flags=native)
        for index, counts in enumerate(([1, 0, 2, 0], [0, 0, 0, 3], [0, 1, 2, 0], [3, 0, 0, 0])):
            refreshed = list(host_fields(runtime, counts, [2, 0, 3], group_type=group_type))
            refreshed[0].add_(index)
            for target, source in zip(tensors, refreshed, strict=True):
                target.copy_(source)
            native.fill_(1)
            graph.replay()
            expected_valid = int(counts[1] == 0)
            assert combined.cpu().tolist() == [expected_valid]
            assert_route(runtime, refreshed, group_type, result)
        # A native validator rejection must survive otherwise valid routes,
        # and the same captured status must recover on its next replay.
        for value in (0, 2, 1):
            native.fill_(value)
            graph.replay()
            assert combined.cpu().tolist() == [int(value == 1)]
    torch.npu.current_stream(runtime.device).wait_stream(stream)


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
        counts, _, gather, inverse, route_flags, summary = metadata
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
        expected_counts, slots, gather, inverse, valid = route_reference(
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
        assert metadata[1].cpu().tolist() == slots
        assert metadata[2].cpu().tolist() == (gather if valid else [-1] * capacity)
        assert metadata[3].cpu().tolist() == inverse
        assert metadata[4].cpu().tolist() == [valid]
        assert metadata[5].cpu().tolist() == [sum(expected_counts), valid]
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
        for result, reference in zip(actual[:4], expected[:4], strict=True):
            assert result.cpu().tolist() == reference
        assert actual[4].cpu().tolist() == [1]
        assert actual[5].cpu().tolist() == [sum(counts), 1]

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
