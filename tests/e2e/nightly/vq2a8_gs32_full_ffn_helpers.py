# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full FFN acceptance utilities independent of retired service runtimes."""

import hashlib
import math
import statistics
from time import perf_counter
from types import SimpleNamespace

WARMUP_REPEATS = 2
FUSION_NRMSE_LIMIT = 0.005
FUSION_PEAK_ERROR_LIMIT = 0.02
FUSION_FP8_MISMATCH_LIMIT = 0.01


def select_full_ep_inventory(entries, num_experts, *, ep_size=4):
    """Choose a complete real EP range, retaining sparse checkpoint holes.

    Select the range with the largest resident inventory, then the widest
    precision coverage. Stable layer/rank traversal resolves ties. The caller
    can remap global IDs to this rank's local logical range before packing.
    """
    if (
        type(num_experts) is not int
        or type(ep_size) is not int
        or ep_size < 1
        or num_experts < 1
        or num_experts % ep_size
    ):
        raise ValueError("Full-inventory profiling requires an equal contiguous EP partition.")
    local_count = num_experts // ep_size
    selected, best_score = None, (-1, -1)
    for layer_index, entry in sorted(entries.items()):
        inventory = tuple(sorted(entry["gate_up"]))
        if set(inventory) != set(entry["down"]) or any(
            type(expert) is not int or not 0 <= expert < num_experts for expert in inventory
        ):
            raise ValueError("Profiling requires matching in-range Gate/Down inventories.")
        for rank in range(ep_size):
            start = rank * local_count
            ids = tuple(expert for expert in inventory if start <= expert < start + local_count)
            if not ids:
                continue
            pairs = {(entry["gate_up"][expert], entry["down"][expert]) for expert in ids}
            score = (len(ids), len(pairs))
            if score > best_score:
                selected = {
                    "layer": layer_index,
                    "ep_size": ep_size,
                    "ep_rank": rank,
                    "num_global_experts": num_experts,
                    "num_local_experts": local_count,
                    "expert_start": start,
                    "global_ids": ids,
                    "local_ids": tuple(expert - start for expert in ids),
                    "positions": tuple(inventory.index(expert) for expert in ids),
                    "precision_pairs": {
                        "_to_".join(pair): sum(
                            (entry["gate_up"][expert], entry["down"][expert]) == pair for expert in ids
                        )
                        for pair in sorted(pairs)
                    },
                }
                best_score = score
    return selected


def localize_profile_layer(layer, spec, placement):
    """Apply the same contiguous EP numbering while preserving source order."""
    if tuple(spec.expert_ids) != tuple(placement["global_ids"]):
        raise ValueError("The profile must load the complete selected EP inventory.")
    start = placement["expert_start"]
    layer.vq2a8_num_experts = placement["num_local_experts"]
    layer.vq2a8_ep_size = placement["ep_size"]
    layer.vq2a8_ep_rank = placement["ep_rank"]
    layer.vq2a8_expert_start = start
    return SimpleNamespace(
        **{
            **vars(spec),
            "expert_ids": tuple(placement["local_ids"]),
            "quant_types": {
                kind: {expert - start: value for expert, value in types.items()}
                for kind, types in spec.quant_types.items()
            },
        }
    )


def full_inventory_route_patterns(spec, rows):
    """Deterministic empty, spread and precision-specific hot expert loads."""
    if type(rows) is not int or rows < 1 or not spec.expert_ids:
        raise ValueError("Profiling requires positive rows and a nonempty inventory.")
    experts = tuple(spec.expert_ids)
    balanced = tuple(rows // len(experts) + int(slot < rows % len(experts)) for slot in range(len(experts)))
    result = [("empty", (0,) * len(experts)), ("balanced", balanced)]
    pairs = {}
    for slot, expert in enumerate(experts):
        pair = (spec.quant_types["gate_up"][expert], spec.quant_types["down"][expert])
        pairs.setdefault(pair, slot)
    for pair, slot in sorted(pairs.items()):
        result.append(
            ("hot_" + "_to_".join(pair), tuple(rows if index == slot else 0 for index in range(len(experts))))
        )
    skewed = [0] * len(experts)
    skewed[0] = rows * 3 // 4
    for index in range(rows - skewed[0]):
        skewed[(index + 1) % len(experts)] += 1
    result.append(("skewed", tuple(skewed)))
    return tuple(result)


def tensor_storage_summary(torch, values):
    """Count Python-visible retained storage once, including aliased views.

    Opaque native route/plan allocations are not introspected; allocator
    measurements complement this logical ownership report.
    """
    tensors, seen_containers, opaque = {}, set(), set()

    def visit(value):
        if isinstance(value, torch.Tensor):
            tensors[id(value)] = value
        elif isinstance(value, (tuple, list, dict)):
            if id(value) in seen_containers:
                return
            seen_containers.add(id(value))
            for child in value.values() if isinstance(value, dict) else value:
                visit(child)
        elif value is not None and not isinstance(value, (bool, int, float, str)):
            opaque.add(id(value))

    visit(values)
    storages = {}
    for tensor in tensors.values():
        storage = tensor.untyped_storage()
        if storage.nbytes():
            storages[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
    return {
        "tensor_objects": len(tensors),
        "unique_storages": len(storages),
        "storage_bytes": sum(storages.values()),
        "logical_tensor_bytes": sum(tensor.numel() * tensor.element_size() for tensor in tensors.values()),
        "opaque_objects_excluded": len(opaque),
        "scope": "Python-visible tensor base storage; opaque native allocations excluded",
    }


def npu_memory_snapshot(torch, device):
    """Sample allocator counters after the caller has synchronized the device."""
    return {
        "allocated_bytes": int(torch.npu.memory_allocated(device)),
        "reserved_bytes": int(torch.npu.memory_reserved(device)),
        "peak_allocated_bytes": int(torch.npu.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.npu.max_memory_reserved(device)),
    }


def timing_statistics(values):
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Timing samples must be a nonempty sequence of nonnegative milliseconds.")
    return {
        "samples_ms": list(values),
        "min_ms": min(values),
        "median_ms": statistics.median(values),
        "max_ms": max(values),
    }


def measure_npu_submissions(runtime, submit, check, *, warmups, repeats):
    """Measure queued NPU execution and host submission separately.

    Correctness reads occur outside the timed range. These are per-FFN metrics,
    not end-to-end latency or a substitute for multi-rank serving benchmarks.
    """
    if type(warmups) is not int or type(repeats) is not int or warmups < 1 or repeats < 1:
        raise ValueError("Profiling requires positive warmup and repeat counts.")
    torch = runtime.torch
    for _ in range(warmups):
        submit()
    torch.npu.synchronize()
    check()
    torch.npu.reset_peak_memory_stats(runtime.device)
    before = npu_memory_snapshot(torch, runtime.device)
    events, host_ms = [], []
    for _ in range(repeats):
        begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        begin.record()
        started = perf_counter()
        submit()
        host_ms.append((perf_counter() - started) * 1000)
        end.record()
        events.append((begin, end))
    torch.npu.synchronize()
    check()
    after = npu_memory_snapshot(torch, runtime.device)
    return {
        "warmups": warmups,
        "repeats": repeats,
        "device": timing_statistics([begin.elapsed_time(end) for begin, end in events]),
        "host_submission": timing_statistics(host_ms),
        "memory_before": before,
        "memory_after": after,
    }


def route_counts(torch, layer, spec, counts, device, group_type):
    logical = torch.zeros(layer.vq2a8_num_experts, dtype=torch.int64)
    for expert, count in zip(spec.expert_ids, counts, strict=True):
        logical[expert] = count
    if group_type == 0:
        logical = logical.cumsum(0)
    return logical.to(device)


def select_fusion_expert(entries, pair):
    """Choose a real pair from any layer, without opening tensor payloads."""
    for layer_index, entry in sorted(entries.items()):
        for position, expert in enumerate(sorted(entry["gate_up"])):
            if (entry["gate_up"][expert], entry["down"][expert]) == tuple(pair):
                return {"layer": layer_index, "expert": expert, "position": position, "pair": tuple(pair)}
    return None


def fusion_error_metrics(torch, reference, actual):
    if reference.shape != actual.shape:
        raise AssertionError(f"Fusion changed output shape: {reference.shape} -> {actual.shape}.")
    # Diagnostic reductions run on the host in FP64 to avoid overflow when
    # finite BF16 outputs have large magnitudes.
    reference, actual = reference.detach().cpu().double(), actual.detach().cpu().double()
    if not bool(torch.isfinite(reference).all() & torch.isfinite(actual).all()):
        return {"finite": False, "nrmse": None, "peak_error_ratio": None, "max_abs_error": None}
    difference = (actual - reference).abs()
    floor = torch.finfo(torch.float32).tiny
    return {
        "finite": True,
        "nrmse": float(difference.square().mean().sqrt() / reference.square().mean().sqrt().clamp_min(floor)),
        "peak_error_ratio": float(difference.max() / reference.abs().max().clamp_min(floor)),
        "max_abs_error": float(difference.max()),
    }


def fusion_byte_metrics(torch, reference, actual):
    assert reference.shape == actual.shape and reference.dtype == actual.dtype == torch.uint8
    different = (reference != actual).flatten().nonzero().flatten()
    return {
        "different": different.numel(),
        "total": reference.numel(),
        "fraction": different.numel() / max(1, reference.numel()),
        "reference_sha256": hashlib.sha256(reference.numpy().tobytes()).hexdigest(),
        "fused_sha256": hashlib.sha256(actual.numpy().tobytes()).hexdigest(),
        "first_differences": [
            {"index": index, "reference": int(reference.flatten()[index]), "fused": int(actual.flatten()[index])}
            for index in different[:8].tolist()
        ],
    }


def fusion_metrics_accepted(metrics):
    return (
        metrics["finite"]
        and metrics["nrmse"] <= FUSION_NRMSE_LIMIT
        and metrics["peak_error_ratio"] <= FUSION_PEAK_ERROR_LIMIT
    )


class CapturedBank:
    """Own all graph addresses and perform the required external validity check."""

    def __init__(self, runtime, bank, hidden, groups, group_type, *, validity=None, measure_capture=False):
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
        self.capture_metrics = None
        if measure_capture:
            torch.npu.reset_peak_memory_stats(runtime.device)
            before = npu_memory_snapshot(torch, runtime.device)
        self.graph = torch.npu.NPUGraph()
        self.validity.zero_()
        # Keep the default torch.npu.graph side stream. Immutable native plans
        # were prepared on the original stream and must support this handoff.
        started = perf_counter()
        with torch.npu.graph(self.graph):
            self.output, event = bank.apply(self.hidden, self.groups, group_type, runtime.limit)
        if measure_capture:
            torch.npu.synchronize()
            self.capture_metrics = {
                "wall_ms": (perf_counter() - started) * 1000,
                "memory_before": before,
                "memory_after": npu_memory_snapshot(torch, runtime.device),
                "retained": tensor_storage_summary(torch, bank.capture_owners[-1]),
            }
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
