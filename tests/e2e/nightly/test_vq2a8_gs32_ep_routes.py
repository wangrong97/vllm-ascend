# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real NPU padded-route capture for each EP2/4/8 local-bank geometry.

These tests use one NPU; they validate routing, not distributed collectives.
"""

import pytest

from tests.e2e.nightly.test_vq2a8_gs32_activation import activation_runtime  # noqa: F401
from tests.e2e.nightly.test_vq2a8_gs32_device_routes import route_runtime  # noqa: F401
from tests.ut.quantization.test_vq2a8_gs32_device_routes import route_reference

pytestmark = pytest.mark.filterwarnings(r"error:tl\.where with a non-boolean condition is deprecated.*:UserWarning")


@pytest.mark.parametrize("ep_size", [2, 4, 8])
@pytest.mark.parametrize("capacity", [8, 256])
@pytest.mark.parametrize("group_type", [0, 1])
@pytest.mark.parametrize("dtype_name", ["int32", "int64"])
def test_gs32_ep_routes_padding_graph_dynamic_empty_and_recovery(
    route_runtime,  # noqa: F811 - imported pytest fixture
    ep_size,
    capacity,
    group_type,
    dtype_name,
):
    runtime, torch = route_runtime, route_runtime.torch
    local_experts = 256 // ep_size
    first = 256 - local_experts
    experts = list(reversed(range(first, 256)))
    slots = [-1] * 256
    for slot, expert in enumerate(experts):
        slots[expert] = slot
    hidden = torch.arange(capacity * 64).reshape(capacity, 64).bfloat16()
    expert_ids = torch.tensor(experts, dtype=torch.int64)
    logical_slots = torch.tensor(slots, dtype=torch.int64)
    dtype = getattr(torch, dtype_name)
    partial = torch.zeros(256, dtype=dtype)
    partial[first], partial[-1] = 1, 2
    empty = torch.zeros_like(partial)
    full = torch.zeros_like(partial)
    full[first], full[-1] = min(capacity, 128), max(0, capacity - 128)
    missing = partial.clone()
    missing[0] = 1
    over_group = partial.clone()
    over_group[first] = 129
    over_capacity = full.clone()
    over_capacity[first + 1] = 1
    values = (partial, full, empty, missing, over_group, over_capacity, partial)

    def host_group(counts):
        return counts.cumsum(0).to(dtype) if group_type == 0 else counts

    tensors = tuple(value.to(runtime.device) for value in (hidden, host_group(partial), expert_ids, logical_slots))

    def forward():
        return runtime.route(tensors[0], tensors[1], group_type, tensors[2], tensors[3], allow_padding=True)

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
    for index, counts in enumerate(values):
        current_hidden = hidden + index
        group = host_group(counts)
        with torch.npu.stream(stream):
            tensors[0].copy_(current_hidden)
            tensors[1].copy_(group)
            graph.replay()
        stream.synchronize()
        physical, row_slots, source, inverse, valid = route_reference(
            group.tolist(), group_type, experts, slots, capacity, allow_padding=True
        )
        assert result[4].cpu().tolist() == [valid]
        assert result[1].cpu().tolist() == physical
        assert result[2].cpu().tolist() == row_slots
        assert result[3].cpu().tolist() == inverse
        expected = torch.zeros_like(hidden)
        for destination, source_row in enumerate(source):
            if valid and source_row >= 0:
                expected[destination] = current_hidden[source_row]
        actual = result[0].cpu()
        assert torch.equal(actual, expected)
        if valid:
            live_rows = int(counts.sum())
            assert torch.equal(actual[inverse][:live_rows], current_hidden[:live_rows])
            assert torch.equal(actual[live_rows:], torch.zeros_like(actual[live_rows:]))
