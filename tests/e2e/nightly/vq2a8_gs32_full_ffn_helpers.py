# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full FFN acceptance utilities independent of retired service runtimes."""

import hashlib

WARMUP_REPEATS = 2
FUSION_NRMSE_LIMIT = 0.005
FUSION_PEAK_ERROR_LIMIT = 0.02
FUSION_FP8_MISMATCH_LIMIT = 0.01


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
