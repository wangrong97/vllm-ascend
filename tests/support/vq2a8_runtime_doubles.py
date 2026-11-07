# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU transport doubles for full FFN routing/lifetimes, never numerical proof."""

import torch

from tests.support.vq2a8_oracle import quantize_mxfp8_k32


class CpuActivation:
    def prepare_unchecked(self, hidden, *, source_rows=None, route_flags=None, route_metadata=None):
        padding = torch.zeros(hidden.shape[0], dtype=torch.bool)
        metadata_valid = True
        if route_metadata is not None:
            assert source_rows is None
            live_rows, metadata_valid = route_metadata.tolist()
            metadata_valid = metadata_valid == 1 and 0 <= live_rows <= hidden.shape[0]
            padding = torch.arange(hidden.shape[0]) >= live_rows
            hidden = torch.where(padding[:, None], 0, hidden)
        if source_rows is not None:
            padding = source_rows == -1
            hidden = hidden.index_select(0, source_rows.clamp_min(0))
            hidden = torch.where(padding[:, None], 0, hidden)
        valid = torch.isfinite(hidden).all(1)
        valid &= metadata_valid
        if route_flags is not None:
            valid &= (route_flags == 1).all()
        data, scale = quantize_mxfp8_k32(torch.where(valid[:, None], hidden, 0))
        scale[padding] = 127
        return data, scale, valid.int()

    def finish_full_ffn(self, output, middle_scales, input_flags, route_flags, native_flags, route_metadata, validity):
        live_rows, metadata_valid = route_metadata.tolist()
        route_valid = (
            metadata_valid == 1
            and 0 <= live_rows <= output.shape[0]
            and bool((route_flags == 1).all())
            and bool((native_flags == 1).all())
        )
        live = (
            torch.arange(output.shape[0]) < live_rows if route_valid else torch.zeros(output.shape[0], dtype=torch.bool)
        )
        finite = torch.isfinite(output[live]).all()
        output[~live] = 0
        output.masked_fill_(~torch.isfinite(output), 0)
        status = (
            finite & route_valid & (input_flags == 1).all() & (native_flags == 1).all() & (middle_scales != 255).all()
        )
        validity.copy_(status)
        return status.int().reshape(1)


class NoHostCounts(torch.Tensor):
    def cpu(self, *args, **kwargs):
        raise AssertionError("Production graph counts must remain on device")

    def tolist(self, *args, **kwargs):
        raise AssertionError("Production routes must not become Python lists")

    def item(self, *args, **kwargs):
        raise AssertionError("Production validity must not become a host scalar")


class CpuNative:
    def __init__(self):
        self.capturing = False
        self.launches = 0
        self.route_calls = []
        self.status_calls = []

    def require_padding_support(self):
        pass

    def check_eager(self):
        if self.capturing:
            raise RuntimeError("Eager execution cannot run inside a graph capture")

    def prepare_device_routes(self, hidden, counts, *, allow_padding=False):
        assert allow_padding
        routes = counts.as_subclass(torch.Tensor).tolist()
        self.route_calls.append(routes)
        return routes

    def device_routes_status(self, routes):
        self.status_calls.append(routes)
        return torch.ones(1, dtype=torch.int32)


class CpuRouter:
    def __init__(self):
        self.metadata_calls = []
        self.scratch_owners = []

    def prepare_identity_metadata(
        self, hidden, groups, group_type, experts, logical_slots, *, allow_padding=False, retained=None
    ):
        # Independent host-integer routing double, never a GPU performance model.
        ids = experts.tolist()
        assert ids == sorted(set(ids))
        values = groups.as_subclass(torch.Tensor).tolist()
        counts = values if group_type == 1 else [end - start for end, start in zip(values, [0, *values[:-1]])]
        capacity = hidden.shape[0]
        valid = all(0 <= count <= capacity for count in counts)
        valid &= sum(counts) <= capacity if allow_padding else sum(counts) == capacity
        valid &= all(count == 0 or index in ids for index, count in enumerate(counts))
        valid &= all(logical_slots[expert] == slot for slot, expert in enumerate(ids))
        physical = [counts[expert] for expert in ids] if valid else [0] * len(ids)
        result = (
            torch.tensor(physical, dtype=torch.int64),
            torch.tensor([int(valid)], dtype=torch.int32),
            torch.tensor([sum(physical), int(valid)], dtype=torch.int64),
        )
        self.metadata_calls.append(tuple(ids))
        if retained is not None:
            self.scratch_owners.append(result[2])
            retained.extend(result)
        return result

    def prepare_metadata(
        self, hidden, groups, group_type, experts, logical_slots, *, allow_padding=False, subset=False, retained=None
    ):
        if retained is not None:
            scratch = torch.empty(3)
            self.scratch_owners.append(scratch)
            retained.append(scratch)
        values = groups.as_subclass(torch.Tensor).tolist()
        counts = values if group_type == 1 else [end - start for end, start in zip(values, [0, *values[:-1]])]
        capacity = hidden.shape[0]
        valid = all(0 <= value <= capacity for value in values) and all(count >= 0 for count in counts)
        valid &= sum(counts) <= capacity if allow_padding else sum(counts) == capacity
        if not subset:
            valid &= all(count == 0 or logical_slots[index] >= 0 for index, count in enumerate(counts))
        physical_counts = [counts[int(e)] for e in experts] if valid else [0] * len(experts)
        order = []
        for expert in experts.tolist():
            if valid:
                order.extend(range(sum(counts[:expert]), sum(counts[: expert + 1])))
        gather = torch.full((capacity,), -1, dtype=torch.int64)
        inverse = torch.full_like(gather, -1)
        if order:
            gather[: len(order)] = torch.tensor(order)
            inverse[order] = torch.arange(len(order))
        self.metadata_calls.append(tuple(experts.tolist()))
        return (
            torch.tensor(physical_counts, dtype=torch.int64),
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
