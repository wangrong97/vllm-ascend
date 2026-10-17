# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in selected-expert GS32 equivalence and warm FFN timing on real NPUs.

The original probes materialize three experts from source-v3 layer 0. The
separate fusion_compare probe selects one real expert per quantization pair
and compares eager fused execution and captured full_graph replay. No LLM,
attention, router, whole-model loading or serving latency is measured.
Input-byte capture intentionally synchronizes; uncaptured comparisons and
timings separately exercise the production submission path.
The device_prefill_ep2_capacity diagnostic loads a full local 128-expert bank
and measures synthetic serving-sized route capacity without attention or HCCL.
"""

import hashlib
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.e2e.nightly.vq2a8_gs32_helpers import checkpoint_inventory, load_selected_layer, native_paths

SELECTED_EXPERTS = 3
WARMUP_REPEATS = 3
TIMING_REPEATS = 5
CASES = ((1, 1, 1), (6, 2, 24), (129, 2, 1), (0, 3, 0), (3, 0, 2), (0, 0, 0))
FUSION_PAIRS = (("vq2", "vq2"), ("vq2", "mxfp4"), ("mxfp4", "mxfp4"), ("mxfp4", "vq2"))
FUSION_WARMUP_REPEATS = 5
FUSION_TIMING_REPEATS = 20
# Operator A/B acceptance only; this does not replace task-accuracy evaluation.
# The GS32 scale algorithm must remain exact. Tiny SiLU rounding differences
# may alter FP8 ties; keep their limits explicit and independent of old tests.
FUSION_NRMSE_LIMIT = 0.005
FUSION_PEAK_ERROR_LIMIT = 0.02
FUSION_FP8_MISMATCH_LIMIT = 0.01
PREFILL_VALIDATION_ROWS = 1024


def prepare_selected_layer(layer, device):
    import torch

    from vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4 import prepare_mxfp4_projection

    size = 0
    for name, value in vars(layer).copy().items():
        if isinstance(value, torch.Tensor):
            size += value.numel() * value.element_size()
            setattr(layer, name, value.to(device))
    for prefix in ("w13", "w2"):
        if hasattr(layer, f"{prefix}_mxfp4_weight"):
            weight, scale = prepare_mxfp4_projection(
                getattr(layer, f"{prefix}_mxfp4_weight"), getattr(layer, f"{prefix}_mxfp4_scale")
            )
            setattr(layer, f"{prefix}_mxfp4_weight", weight)
            setattr(layer, f"{prefix}_mxfp4_scale", scale)
    return size


def reversed_selected_bank(layer, spec, options, device):
    """Reverse raw CPU banks before NZ conversion, then update slot metadata."""
    import torch

    from vllm_ascend.ops.fused_moe.vq2a8_gs32 import GS32ExpertWeights

    reversed_layer, selected_spec, _ = load_selected_layer(
        layer.vq2a8_source, layer_index=spec.layer_index, expert_positions=layer.vq2a8_positions
    )
    for name, value in vars(reversed_layer).copy().items():
        if isinstance(value, torch.Tensor) and name != "vq2a8_anchor":
            setattr(reversed_layer, name, value.flip(0))
    prepare_selected_layer(reversed_layer, device)
    result = GS32ExpertWeights(reversed_layer, selected_spec, options)
    for groups in result.banks.values():
        for bank in groups.values():
            bank["expert_ids"] = tuple(reversed(bank["expert_ids"]))
            bank["slots"] = {expert: slot for slot, expert in enumerate(bank["expert_ids"])}
            if "expert_ids_tensor" in bank:
                bank["expert_ids_tensor"].copy_(torch.tensor(bank["expert_ids"], device=device))
                bank["logical_slots"].fill_(-1)
                bank["logical_slots"][bank["expert_ids_tensor"]] = torch.arange(len(bank["expert_ids"]), device=device)
    return result


class ProjectionCapture:
    """Record actual input bytes around real native launches; never fake an op."""

    def __init__(self, bank):
        self.ops = bank.ops
        self.owners = {
            fields["packed_zn"][slot : slot + 1].data_ptr(): (kind, expert)
            for kind, groups in bank.banks.items()
            if "vq2" in groups
            for fields in (groups["vq2"]["fields"],)
            for expert, slot in groups["vq2"]["slots"].items()
        }
        self.chunks = defaultdict(int)
        self.records = {}
        self.group_routes = {}
        self.bank_experts = {kind: group["vq2"]["expert_ids"] for kind, group in bank.banks.items() if "vq2" in group}
        self.bank_kinds = {
            group["vq2"]["fields"]["packed_zn"].data_ptr(): kind for kind, group in bank.banks.items() if "vq2" in group
        }

    def __getattr__(self, name):
        return getattr(self.ops, name)

    def project(self, *args):
        data, scales, weight, _, _, _, _, _, output = args
        identity = self.owners[weight.data_ptr()]
        key = (*identity, self.chunks[identity])
        self.chunks[identity] += 1
        self.records[key] = {
            "data": data.cpu().clone(),
            "scales": scales.cpu().clone(),
            "input_shape": tuple(data.shape),
            "output_shape": tuple(output.shape),
        }
        return self.ops.project(*args)

    def prepare_routes(self, hidden, counts):
        routes = self.ops.prepare_routes(hidden, counts)
        self.group_routes[id(routes)] = counts.tolist()
        return routes

    def project_grouped(self, data, scales, weight, lut, row_scale, routes, output):
        from vllm_ascend.quantization.methods.w2a8.vq2a8_gs32_format import GS32_MAX_ROWS

        kind = self.bank_kinds[weight.data_ptr()]
        first = 0
        for expert, count in zip(self.bank_experts[kind], self.group_routes[id(routes)], strict=True):
            # Compare the same logical rows even when a grouped launch spans
            # several serial-baseline chunks. This does not split the launch.
            for offset in range(0, count, GS32_MAX_ROWS):
                rows = min(count - offset, GS32_MAX_ROWS)
                start = first + offset
                key = (kind, expert, self.chunks[kind, expert])
                self.chunks[kind, expert] += 1
                self.records[key] = {
                    "data": data[start : start + rows].cpu().clone(),
                    "scales": scales[start : start + rows].cpu().clone(),
                    "input_shape": (rows, data.shape[1]),
                    "output_shape": (rows, output.shape[1]),
                }
            first += count
        return self.ops.project_grouped(data, scales, weight, lut, row_scale, routes, output)


def compare_case(banks, hidden, group_list, group_type, limit, *, capture):
    import torch

    outputs, snapshots = {}, {}
    for mode, bank in banks.items():
        original = bank.ops
        recorder = ProjectionCapture(bank) if capture else None
        if recorder is not None:
            bank.ops = recorder
        try:
            output, _ = bank.apply(hidden, group_list, group_type, limit)
            torch.npu.synchronize()
            outputs[mode] = output.cpu().clone()
        finally:
            bank.ops = original
        if recorder is not None:
            snapshots[mode] = recorder.records
    for mode, output in outputs.items():
        torch.testing.assert_close(
            output, outputs["serial"], rtol=0, atol=0, msg=lambda message, mode=mode: f"{mode}: {message}"
        )
    if not capture:
        return
    for mode in snapshots:
        assert snapshots["serial"].keys() == snapshots[mode].keys()
        for key, baseline in snapshots["serial"].items():
            actual = snapshots[mode][key]
            assert baseline["input_shape"] == actual["input_shape"], (mode, key)
            assert baseline["output_shape"] == actual["output_shape"], (mode, key)
            for field in ("data", "scales"):
                assert torch.equal(baseline[field], actual[field]), (mode, key, field)


def route_counts(torch, layer, spec, counts, device, group_type):
    logical = torch.zeros(layer.vq2a8_num_experts, dtype=torch.int64)
    for expert, count in zip(spec.expert_ids, counts, strict=True):
        logical[expert] = count
    if group_type == 0:
        logical = logical.cumsum(0)
    return logical.to(device)


def warm_timing(torch, banks, hidden, counts, limit):
    for _ in range(WARMUP_REPEATS):
        for bank in banks.values():
            bank.apply(hidden, counts, 1, limit)
    torch.npu.synchronize()
    samples = {mode: [] for mode in banks}
    # Alternate order to reduce systematic temperature/clock drift bias.
    for repeat in range(TIMING_REPEATS):
        order = tuple(banks) if repeat % 2 == 0 else tuple(reversed(banks))
        for mode in order:
            torch.npu.synchronize()
            begin = time.perf_counter()
            banks[mode].apply(hidden, counts, 1, limit)
            torch.npu.synchronize()
            samples[mode].append((time.perf_counter() - begin) * 1000)
    medians = {mode: statistics.median(values) for mode, values in samples.items()}
    return {
        "median_ms": medians,
        "samples_ms": samples,
        "speedup_vs_batched": {mode: medians["batched"] / elapsed for mode, elapsed in medians.items()},
    }


def check_stream_cases(torch, banks, layer, spec, device, limit, *, nondefault, cases=CASES):
    generator = torch.Generator(device="cpu").manual_seed(1234)
    for counts in cases:
        host = (torch.randn(sum(counts), spec.specs["gate_up"].columns, generator=generator) * 0.1).bfloat16()
        for group_type in (0, 1):
            groups = route_counts(torch, layer, spec, counts, device, group_type)
            hidden = host.to(device)
            compare_case(banks, hidden, groups, group_type, limit, capture=True)
            # No byte capture or intermediate CPU copies on these runs.
            # Fresh values expose stale prepared-plan or activation reuse.
            for delta in (0.03125, -0.0625):
                hidden = (host.float() + delta).bfloat16().to(device)
                compare_case(banks, hidden, groups, group_type, limit, capture=False)
        print(
            "VQ2A8_GS32_BATCHED_EQUIVALENCE="
            + json.dumps({"counts": counts, "modes": list(banks), "nondefault_stream": nondefault, "status": "PASS"}),
            flush=True,
        )
        if counts in CASES[:2]:
            groups = route_counts(torch, layer, spec, counts, device, 1)
            report = warm_timing(torch, banks, hidden, groups, limit)
            print(
                "VQ2A8_GS32_BATCHED_TIMING="
                + json.dumps(
                    {**report, "counts": counts, "nondefault_stream": nondefault, "scope": "selected FFN only"}
                ),
                flush=True,
            )


def check_shared_plans_across_streams(torch, banks, layer, spec, device, limit):
    check_stream_cases(torch, banks, layer, spec, device, limit, nondefault=False)
    plans = banks["batched"].ops.plans
    default_plans = dict(plans)
    grouped_plans = dict(banks["fused"].ops.grouped_plans) if "fused" in banks else {}
    assert default_plans, "Default-stream cases must prepare real native plans."
    default_stream = torch.npu.current_stream(device)
    other_stream = torch.npu.Stream(device=device)
    other_stream.wait_stream(default_stream)
    # M=5,7,9 have not appeared on the default stream. Prepare these while
    # reusing the same bank and all previously initialized plans.
    new_counts = (5, 7, 9)
    assert not any(shape[0] in new_counts for shape in default_plans)
    with torch.npu.stream(other_stream):
        check_stream_cases(torch, banks, layer, spec, device, limit, nondefault=True, cases=(*CASES, new_counts))
    assert all(plans[key] is plan for key, plan in default_plans.items())
    assert all(any(shape[0] == rows for shape in plans) for rows in new_counts)
    nondefault_plans = {key: plan for key, plan in plans.items() if key not in default_plans}
    assert nondefault_plans
    default_stream.wait_stream(other_stream)
    # Return to the original stream and reuse the newly created handles too.
    check_stream_cases(torch, banks, layer, spec, device, limit, nondefault=False, cases=(new_counts,))
    assert all(plans[key] is plan for key, plan in nondefault_plans.items())
    if "fused" in banks:
        assert grouped_plans
        assert all(banks["fused"].ops.grouped_plans[key] is plan for key, plan in grouped_plans.items())


def check_reordered_bank(torch, banks, layer, spec, paths, device, limit):
    """Compare a physically reversed bank against the original serial bank.

    Keep logical route counts and input rows unchanged. Reversing every field
    and expert_id together forces the real grouped path to reorder input rows
    and restore output order, including leading/interior/trailing zero groups.
    """
    reordered = reversed_selected_bank(layer, spec, {**paths, "gs32_execution": "fused"}, device)
    compared = {"serial": banks["serial"], "fused_reordered": reordered}
    generator = torch.Generator(device="cpu").manual_seed(597)
    for counts in ((0, 3, 0), (3, 0, 2), (1, 2, 4)):
        host = (torch.randn(sum(counts), spec.specs["gate_up"].columns, generator=generator) * 0.1).bfloat16()
        for group_type in (0, 1):
            groups = route_counts(torch, layer, spec, counts, device, group_type)
            compare_case(compared, host.to(device), groups, group_type, limit, capture=True)
            fresh = (host.float() + 0.03125).bfloat16().to(device)
            compare_case(compared, fresh, groups, group_type, limit, capture=False)
        print(
            "VQ2A8_GS32_REORDERED_EQUIVALENCE="
            + json.dumps(
                {
                    "counts": counts,
                    "original_experts": spec.expert_ids,
                    "reordered_experts": {
                        kind: {quant_type: bank["expert_ids"] for quant_type, bank in groups.items()}
                        for kind, groups in reordered.banks.items()
                    },
                    "status": "PASS",
                }
            ),
            flush=True,
        )


def run_real_experts(request, modes):
    options = {
        name: request.config.getoption(flag)
        for name, flag in (
            ("model", "--vq2a8-model"),
            ("gs32_library", "--vq2a8-gs32-library"),
            ("gs32_binary", "--vq2a8-gs32-binary"),
        )
    }
    if not options["model"]:
        pytest.skip("Pass unified --vq2a8-model and both --vq2a8-gs32-* paths on Ascend950.")
    paths = native_paths(options)
    layer, spec, config = load_selected_layer(paths.pop("model"))

    import torch
    import torch_npu  # noqa: F401

    from vllm_ascend.ops.fused_moe.vq2a8_gs32 import GS32ExpertWeights

    assert torch.npu.is_available(), "This opt-in test requires real Ascend hardware."
    device = torch.device("npu", 0)
    torch.npu.set_device(device)
    payload_bytes = prepare_selected_layer(layer, device)
    torch.npu.synchronize()
    print(
        "VQ2A8_GS32_BATCHED_ARTIFACTS="
        + json.dumps(
            {
                "layer": spec.layer_index,
                "experts": spec.expert_ids,
                "resident_weight_bytes": payload_bytes,
                "native": {
                    key: {"path": path, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
                    for key, path in paths.items()
                },
                "model_integration_verified": False,
                "serving_performance_verified": False,
            }
        ),
        flush=True,
    )
    banks = {mode: GS32ExpertWeights(layer, spec, {**paths, "gs32_execution": mode}) for mode in modes}
    check_shared_plans_across_streams(torch, banks, layer, spec, device, config.get("swiglu_limit"))
    if "fused" in banks:
        check_reordered_bank(torch, banks, layer, spec, paths, device, config.get("swiglu_limit"))


def test_gs32_real_experts_batched(request):
    run_real_experts(request, ("serial", "batched"))


def test_gs32_real_experts_fused(request):
    run_real_experts(request, ("serial", "batched", "fused"))


def select_fusion_expert(entries, pair):
    """Choose a real pair from any layer, without opening tensor payloads."""
    for layer_index, entry in sorted(entries.items()):
        for position, expert in enumerate(sorted(entry["gate_up"])):
            if (entry["gate_up"][expert], entry["down"][expert]) == tuple(pair):
                return {"layer": layer_index, "expert": expert, "position": position, "pair": tuple(pair)}
    return None


class FusionActivationCapture:
    """Observe real activation returns without copies or host reads in capture."""

    def __init__(self, activation):
        self.activation = activation
        self.records = []

    def __getattr__(self, name):
        return getattr(self.activation, name)

    def _prepare(self, method, hidden, perm, slots, *args, **kwargs):
        output = getattr(self.activation, method)(hidden, perm, slots, *args, **kwargs)
        self.records.append(
            {
                "method": method,
                "fusion": kwargs.get("swiglu_limit") is not None,
                "swiglu_limit": kwargs.get("swiglu_limit"),
                "input_shape": tuple(hidden.shape),
                "has_perm": perm is not None,
                "data": output[0],
                "scales": output[1],
            }
        )
        return output

    def prepare(self, hidden, perm, slots, *args, **kwargs):
        return self._prepare("prepare", hidden, perm, slots, *args, **kwargs)

    def prepare_unchecked(self, hidden, perm, slots, *args, **kwargs):
        return self._prepare("prepare_unchecked", hidden, perm, slots, *args, **kwargs)

    def snapshot(self, torch, fusion):
        # One selected expert has exactly one bank in each projection. For a
        # graph these are the capture's final two calls; replay refreshes the
        # same tensors without calling this Python proxy again.
        records = self.records[-2:]
        assert len(records) == 2, "Both projection preparations must execute."
        assert [record["fusion"] for record in records] == [False, fusion == "swiglu_quant"]
        down = records[-1]
        trace = [{key: value for key, value in record.items() if key not in ("data", "scales")} for record in records]
        return {
            "trace": trace,
            "fusion_calls": sum(record["fusion"] for record in records),
            "data": down["data"].view(torch.uint8).cpu().clone(),
            "scales": down["scales"].view(torch.uint8).cpu().clone(),
        }


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


def compare_fusion_outputs(torch, outputs, *, context, quiet=False):
    metrics = fusion_error_metrics(torch, outputs["none"], outputs["swiglu_quant"])
    accepted = fusion_metrics_accepted(metrics)
    if not quiet or not accepted:
        print("VQ2A8_GS32_FUSION_OUTPUT=" + json.dumps({**context, "error": metrics, "accepted": accepted}), flush=True)
    assert accepted, f"Fusion operator A/B exceeded fixed error limits: {metrics}"
    return metrics


def compare_fusion_activations(torch, snapshots, *, context):
    baseline, fused = snapshots["none"], snapshots["swiglu_quant"]
    data = fusion_byte_metrics(torch, baseline["data"], fused["data"])
    scales = fusion_byte_metrics(torch, baseline["scales"], fused["scales"])

    def dequantize(snapshot):
        values = snapshot["data"].view(torch.float8_e4m3fn).float()
        return values * torch.exp2(snapshot["scales"].float() - 127).repeat_interleave(32, dim=-1)

    error = fusion_error_metrics(torch, dequantize(baseline), dequantize(fused))
    accepted = (
        scales["different"] == 0 and data["fraction"] <= FUSION_FP8_MISMATCH_LIMIT and fusion_metrics_accepted(error)
    )
    report = {
        **context,
        "fp8_bytes": data,
        "e8m0_bytes": scales,
        "dequantized_error": error,
        "actual_activation_calls": {mode: snapshot["trace"] for mode, snapshot in snapshots.items()},
        "fusion_calls": {mode: snapshot["fusion_calls"] for mode, snapshot in snapshots.items()},
        "accepted": accepted,
    }
    print("VQ2A8_GS32_FUSION_ACTIVATION=" + json.dumps(report), flush=True)
    assert accepted, "Fusion changed E8M0 bytes or exceeded fixed activation error limits."
    return report


def fusion_event_timing(
    torch,
    runners,
    validate,
    *,
    warmup=FUSION_WARMUP_REPEATS,
    repeats=FUSION_TIMING_REPEATS,
    reference_mode="none",
    candidate_mode="swiglu_quant",
):
    """Time only warmed execution; validate both outputs after every pair."""
    for index in range(warmup):
        outputs = {mode: run() for mode, run in runners.items()}
        torch.npu.synchronize()
        validate(outputs, "warmup", index)
    samples = {mode: [] for mode in runners}
    wall_samples = {mode: [] for mode in runners}
    for repeat in range(repeats):
        order = tuple(runners) if repeat % 2 == 0 else tuple(reversed(runners))
        outputs = {}
        for mode in order:
            begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            torch.npu.synchronize()
            wall_begin = time.perf_counter()
            begin.record()
            outputs[mode] = runners[mode]()
            end.record()
            end.synchronize()
            wall_samples[mode].append((time.perf_counter() - wall_begin) * 1000)
            elapsed = begin.elapsed_time(end)
            assert elapsed > 0, "NPU events must report a positive elapsed time."
            samples[mode].append(elapsed)
        # Host checks/copies are deliberately outside the measured events.
        validate(outputs, "timing", repeat)
    medians = {mode: statistics.median(values) for mode, values in samples.items()}
    wall_medians = {mode: statistics.median(values) for mode, values in wall_samples.items()}
    return {
        "npu_event_median_ms": medians,
        "samples_ms": samples,
        "speedup": medians[reference_mode] / medians[candidate_mode],
        "synchronized_wall_median_ms": wall_medians,
        "synchronized_wall_samples_ms": wall_samples,
        "synchronized_wall_speedup": wall_medians[reference_mode] / wall_medians[candidate_mode],
        "validated_warmup_pairs": warmup,
        "validated_timing_pairs": repeats,
    }


@pytest.mark.parametrize("execution", ("fused", "full_graph"))
@pytest.mark.parametrize("pair", FUSION_PAIRS, ids=lambda pair: "_to_".join(pair))
def test_gs32_real_experts_fusion_compare(request, execution, pair):
    options = {
        name: request.config.getoption(flag, default=None)
        for name, flag in (
            ("model", "--vq2a8-model"),
            ("gs32_library", "--vq2a8-gs32-library"),
            ("gs32_binary", "--vq2a8-gs32-binary"),
        )
    }
    if not options["model"]:
        pytest.skip("Pass source-v3 --vq2a8-model and both native artifacts on Ascend950 for fusion_compare.")
    paths = native_paths(options)
    model = paths.pop("model")
    _, _, entries = checkpoint_inventory(model)
    selected = select_fusion_expert(entries, pair)
    if selected is None:
        message = f"No real {' -> '.join(pair)} expert exists in this checkpoint; combination was not tested."
        print(
            "VQ2A8_GS32_FUSION_MISSING=" + json.dumps({"pair": pair, "execution": execution, "reason": message}),
            flush=True,
        )
        pytest.skip(message)

    import torch
    import torch_npu  # noqa: F401

    from vllm_ascend.ops.fused_moe.vq2a8_gs32 import GS32ExpertWeights

    assert torch.npu.is_available(), "Explicit fusion comparison requires real Ascend hardware."
    device = torch.device("npu", 0)
    torch.npu.set_device(device)
    layer, spec, config = load_selected_layer(
        model, layer_index=selected["layer"], expert_positions=(selected["position"],)
    )
    payload_bytes = prepare_selected_layer(layer, device)
    limit = config.get("swiglu_limit")
    context = {
        **selected,
        "execution": execution,
        "resident_weight_bytes_shared_by_both_modes": payload_bytes,
        "scope": "selected real expert FFN operator A/B; not serving or GPQA/AIME acceptance",
        "thresholds": {
            "nrmse": FUSION_NRMSE_LIMIT,
            "peak_error_ratio": FUSION_PEAK_ERROR_LIMIT,
            "fp8_mismatch_fraction": FUSION_FP8_MISMATCH_LIMIT,
            "e8m0_mismatches": 0,
        },
    }
    print("VQ2A8_GS32_FUSION_CASE=" + json.dumps(context), flush=True)
    # No duplicate weight banks: both runtimes reference the same Parameters.
    banks = {
        mode: GS32ExpertWeights(layer, spec, {**paths, "gs32_execution": execution, "gs32_fusion": mode})
        for mode in ("none", "swiglu_quant")
    }
    for mode, bank in banks.items():
        assert bank.fusion == mode
    generator = torch.Generator(device="cpu").manual_seed(950)
    for rows in (8,) if execution == "full_graph" else (1, 8, 128):
        row_context = {**context, "rows": rows}
        groups = route_counts(torch, layer, spec, (rows,), device, 1)
        original = (torch.randn(rows, spec.specs["gate_up"].columns, generator=generator) * 0.1).bfloat16()
        hidden = original.to(device)
        recorders = {mode: FusionActivationCapture(bank.activation) for mode, bank in banks.items()}
        for mode, bank in banks.items():
            bank.activation = recorders[mode]
        captured = {}
        try:
            if execution == "full_graph":
                from tests.e2e.nightly.test_vq2a8_gs32_device_bank import CapturedBank

                runtime = SimpleNamespace(torch=torch, device=device, limit=limit)
                captured = {mode: CapturedBank(runtime, bank, hidden, groups, 1) for mode, bank in banks.items()}
            for index, delta in enumerate((0.0, 0.03125, -0.0625)):
                hidden = (original.float() + delta).bfloat16().to(device)
                outputs, snapshots = {}, {}
                for mode, bank in banks.items():
                    if captured:
                        outputs[mode] = captured[mode].replay_checked(hidden, groups)
                    else:
                        recorders[mode].records.clear()
                        outputs[mode] = bank.apply(hidden, groups, 1, limit)[0]
                        torch.npu.synchronize()
                    snapshots[mode] = recorders[mode].snapshot(torch, mode)
                step_context = {**row_context, "input_revision": index}
                compare_fusion_outputs(torch, outputs, context=step_context)
                compare_fusion_activations(torch, snapshots, context=step_context)
        finally:
            for mode, bank in banks.items():
                bank.activation = recorders[mode].activation

        def validate(outputs, phase, index, captured=captured, row_context=row_context):
            if captured:
                for mode, graph in captured.items():
                    assert bool(graph.validity.cpu()), f"{mode} invalid during {phase} {index}"
            compare_fusion_outputs(
                torch, outputs, context={**row_context, "phase": phase, "iteration": index}, quiet=True
            )
            # Clear outside the timed events so the next replay must write a
            # fresh success value; a stale True cannot pass a later round.
            for graph in captured.values():
                graph.validity.zero_()

        if captured:
            for graph in captured.values():
                graph.validity.zero_()

            def replay(graph):
                graph.graph.replay()
                return graph.output

            runners = {mode: lambda graph=graph: replay(graph) for mode, graph in captured.items()}
        else:
            runners = {
                mode: lambda bank=bank, hidden=hidden, groups=groups: bank.apply(hidden, groups, 1, limit)[0]
                for mode, bank in banks.items()
            }
        report = fusion_event_timing(torch, runners, validate)
        # This success/speedup record is emitted only after every numerical,
        # byte, warmup and timed-output check above has passed.
        print("VQ2A8_GS32_FUSION_TIMING=" + json.dumps({**row_context, **report, "status": "PASS"}), flush=True)


@pytest.mark.parametrize("fusion", ("none", "swiglu_quant"))
@pytest.mark.parametrize("pair", FUSION_PAIRS, ids=lambda pair: "_to_".join(pair))
def test_gs32_real_mixed_ffn_device_prefill_compare_and_timing(request, pair, fusion):
    """Compare the new eager prefill with strict eager on actual mixed weights."""
    options = {
        name: request.config.getoption(flag, default=None)
        for name, flag in (
            ("model", "--vq2a8-model"),
            ("gs32_library", "--vq2a8-gs32-library"),
            ("gs32_binary", "--vq2a8-gs32-binary"),
        )
    }
    if not options["model"]:
        pytest.skip("Pass source-v3 --vq2a8-model and both native artifacts on Ascend950 for device prefill.")
    paths = native_paths(options)
    model = paths.pop("model")
    _, _, entries = checkpoint_inventory(model)
    selected = select_fusion_expert(entries, pair)
    if selected is None:
        pytest.skip(f"No real {' -> '.join(pair)} expert exists; this prefill combination was not tested.")
    entry = entries[selected["layer"]]
    inventory = tuple(sorted(entry["gate_up"]))
    # Keep one representative of every available pair in this same layer.
    # Different GateUp/Down types must form independently padded subsets.
    positions = {selected["position"]}
    for quant_pair in FUSION_PAIRS:
        for position, expert in enumerate(inventory):
            if (entry["gate_up"][expert], entry["down"][expert]) == quant_pair:
                positions.add(position)
                break
    for position in range(len(inventory)):
        if len(positions) >= SELECTED_EXPERTS:
            break
        positions.add(position)

    import torch
    import torch_npu  # noqa: F401

    from vllm_ascend.ops.fused_moe.vq2a8_gs32 import GS32ExpertWeights

    assert torch.npu.is_available(), "Explicit device-prefill comparison requires real Ascend hardware."
    device = torch.device("npu", 0)
    torch.npu.set_device(device)
    layer, spec, config = load_selected_layer(
        model, layer_index=selected["layer"], expert_positions=tuple(sorted(positions))
    )
    # This is a local FFN padded-capacity test; no EP communication is mocked.
    layer.vq2a8_ep_size = 2
    payload_bytes = prepare_selected_layer(layer, device)
    limit = config.get("swiglu_limit")
    banks = {
        mode: GS32ExpertWeights(layer, spec, {**paths, "gs32_execution": mode, "gs32_fusion": fusion})
        for mode in ("fused", "full_graph")
    }
    candidate = banks["full_graph"]
    # The serving forward owner checks this status after all FFN work. Test
    # that same submission contract and check every output outside timings.
    candidate.defer_validity = True
    target = spec.expert_ids.index(selected["expert"])
    companion = next((slot for slot in range(len(spec.expert_ids)) if slot != target), target)
    generator = torch.Generator(device="cpu").manual_seed(951)
    context = {
        **selected,
        "fusion": fusion,
        "experts": spec.expert_ids,
        "quantization_pairs": {
            expert: (spec.quant_types["gate_up"][expert], spec.quant_types["down"][expert])
            for expert in spec.expert_ids
        },
        "resident_weight_bytes_shared_by_both_modes": payload_bytes,
        "scope": "real selected mixed FFN eager prefill; local padded capacity; no serving latency claim",
    }
    stream = torch.npu.Stream(device=device)
    for capacity, pattern in ((257, "full"), (4097, "split"), (8192, "padded"), (257, "empty")):
        counts = [0] * len(spec.expert_ids)
        if pattern != "empty":
            counts[target] = capacity - (7 if pattern == "padded" else 0)
            if pattern == "split" and companion != target:
                counts[target] -= 1
                counts[companion] = 1
        active_rows = sum(counts)
        original = (torch.randn(capacity, spec.specs["gate_up"].columns, generator=generator) * 0.1).bfloat16()
        case_context = {**context, "capacity": capacity, "counts": counts, "pattern": pattern}

        def validate(outputs, phase, iteration, case_context=case_context, active_rows=active_rows):
            assert bool(candidate.validity_buffer.cpu()), f"Invalid prefill during {phase} {iteration}."
            baseline, actual = outputs["fused"].cpu(), outputs["full_graph"].cpu()
            assert torch.isfinite(baseline.float()).all() and torch.isfinite(actual.float()).all()
            assert torch.equal(actual.view(torch.int16), baseline.view(torch.int16)), (
                f"Device prefill differs from strict eager: {case_context}; "
                f"{fusion_error_metrics(torch, baseline, actual)}"
            )
            assert torch.count_nonzero(actual[active_rows:]) == 0, "Padded/empty output rows must be zero."
            candidate.validity_buffer.zero_()

        for group_type in (1, 0):
            groups = route_counts(torch, layer, spec, counts, device, group_type)
            active_stream = torch.npu.current_stream(device) if group_type == 1 else stream
            active_stream.wait_stream(torch.npu.current_stream(device))
            with torch.npu.stream(active_stream):
                for revision, delta in enumerate((0.0, 0.03125)):
                    hidden = (original.float() + delta).bfloat16().to(device)
                    candidate.validity_buffer.zero_()
                    outputs = {mode: bank.apply(hidden, groups, group_type, limit)[0] for mode, bank in banks.items()}
                    active_stream.synchronize()
                    validate(outputs, f"group_type={group_type}", revision)
            active_stream.synchronize()
            print(
                "VQ2A8_GS32_DEVICE_PREFILL_EQUIVALENCE="
                + json.dumps(
                    {**case_context, "group_type": group_type, "nondefault_stream": group_type == 0, "status": "PASS"}
                ),
                flush=True,
            )
        if pattern == "empty":
            continue
        groups = route_counts(torch, layer, spec, counts, device, 1)
        hidden = original.to(device)
        candidate.validity_buffer.zero_()
        runners = {
            mode: lambda bank=bank, hidden=hidden, groups=groups: bank.apply(hidden, groups, 1, limit)[0]
            for mode, bank in banks.items()
        }
        report = fusion_event_timing(
            torch,
            runners,
            validate,
            warmup=WARMUP_REPEATS,
            repeats=TIMING_REPEATS,
            reference_mode="fused",
            candidate_mode="full_graph",
        )
        print(
            "VQ2A8_GS32_DEVICE_PREFILL_TIMING=" + json.dumps({**case_context, **report, "status": "PASS"}),
            flush=True,
        )


def device_prefill_ep2_counts(tokens, topk, local_experts):
    """Synthetic EP2 loads with the dispatcher's full token-times-topk capacity."""
    if tokens < 1 or local_experts < 2 or not 1 <= topk <= 2 * local_experts:
        raise ValueError("Invalid EP2 token, top-k or local expert geometry.")
    capacity = tokens * topk
    active = capacity // 2
    rows, tail = divmod(active, local_experts)
    balanced = tuple(rows + (slot < tail) for slot in range(local_experts))
    sparse = [0] * local_experts
    sparse_rows = min(tokens, max(2, capacity // 8))
    sparse[0], sparse[-1] = sparse_rows - 1, 1
    return capacity, (("balanced_half", balanced), ("skewed_padded", tuple(sparse)))


def device_prefill_bank_occupancy(spec, counts, capacity):
    """Report each independently prepared quantization bank's padding ratio."""
    result = {}
    for projection, types in spec.quant_types.items():
        result[projection] = {}
        for quant_type in sorted(set(types.values())):
            experts = tuple(expert for expert, kind in types.items() if kind == quant_type)
            live = sum(counts[expert] for expert in experts)
            result[projection][quant_type] = {
                "experts": len(experts),
                "live_rows": live,
                "capacity": capacity,
                "occupancy": live / capacity,
            }
    return result


@pytest.fixture(scope="module", params=(0, 1), ids=lambda rank: f"rank{rank}")
def device_prefill_ep2_bank(request):
    options = {
        name: request.config.getoption(flag, default=None)
        for name, flag in (
            ("model", "--vq2a8-model"),
            ("gs32_library", "--vq2a8-gs32-library"),
            ("gs32_binary", "--vq2a8-gs32-binary"),
        )
    }
    if not options["model"]:
        pytest.skip("Pass source-v3 model and both native artifacts for the full EP2 capacity diagnostic.")
    paths = native_paths(options)
    model = paths.pop("model")
    config, _, inventory = checkpoint_inventory(model)

    # Lazy imports keep NPU initialization out of collection. select_bank only
    # selects a complete local inventory; its old 128-token route cases are unused.
    import torch
    import torch_npu  # noqa: F401

    from tests.e2e.nightly.vq2a8_gs32_ep_helpers import select_bank
    from vllm_ascend.ops.fused_moe.vq2a8_gs32 import GS32ExpertWeights

    layer_index, positions, local_ids, local_count = select_bank(config, request.param, inventory)
    assert local_count == 128, "This diagnostic requires all 128 local experts from a 256-expert EP2 model."
    assert torch.npu.is_available(), "The EP2 capacity diagnostic requires Ascend hardware."
    device = torch.device("npu", 0)
    torch.npu.set_device(device)
    layer, spec, _ = load_selected_layer(model, layer_index=layer_index, expert_positions=positions)
    global_ids = spec.expert_ids
    spec = SimpleNamespace(
        layer_index=layer_index,
        expert_ids=local_ids,
        specs=spec.specs,
        quant_types={
            kind: {local: types[global_] for global_, local in zip(global_ids, local_ids, strict=True)}
            for kind, types in spec.quant_types.items()
        },
    )
    layer.vq2a8_num_experts, layer.vq2a8_ep_size = local_count, 2
    weight_bytes = prepare_selected_layer(layer, device)
    banks = {
        mode: GS32ExpertWeights(layer, spec, {**paths, "gs32_execution": mode, "gs32_fusion": "none"})
        for mode in ("fused", "full_graph")
    }
    banks["full_graph"].defer_validity = True
    yield (
        torch,
        device,
        banks,
        spec,
        config,
        {
            "ep_rank": request.param,
            "layer": layer_index,
            "global_experts": global_ids,
            "local_experts": local_ids,
            "resident_expert_count": local_count,
            "resident_weight_bytes_shared_by_both_modes": weight_bytes,
            "fusion": "none",
            "scope": "full local EP2 bank; synthetic routes; eager FFN only; excludes attention, HCCL and router",
        },
    )
    banks.clear()
    torch.npu.synchronize()


@pytest.mark.parametrize("tokens", (4096, 8192), ids=lambda tokens: f"t{tokens}")
def test_gs32_device_prefill_ep2_capacity(device_prefill_ep2_bank, tokens):
    """Measure padding costs using every local expert and serving-sized capacity."""
    torch, device, banks, spec, config, context = device_prefill_ep2_bank
    capacity, cases = device_prefill_ep2_counts(tokens, config["num_experts_per_tok"], len(spec.expert_ids))
    candidate = banks["full_graph"]
    generator = torch.Generator(device="cpu").manual_seed(951)
    for pattern, counts in cases:
        active_rows = sum(counts)
        host = torch.randn(capacity, spec.specs["gate_up"].columns, dtype=torch.bfloat16, generator=generator)
        host.mul_(0.1)
        host[active_rows:] = float("nan")
        hidden = host.to(device)
        del host
        groups = torch.tensor(counts, dtype=torch.int64, device=device)
        case_context = {
            **context,
            "tokens": tokens,
            "topk": config["num_experts_per_tok"],
            "group_list_type": 1,
            "capacity": capacity,
            "active_rows": active_rows,
            "pattern": pattern,
            "counts": counts,
            "banks": device_prefill_bank_occupancy(spec, counts, capacity),
        }
        print("VQ2A8_GS32_EP2_CAPACITY_CASE=" + json.dumps(case_context), flush=True)

        def validate(outputs, phase, iteration, case_context=case_context, active_rows=active_rows):
            assert bool(candidate.validity_buffer.cpu()), f"Invalid prefill during {phase} {iteration}: {case_context}"
            # Keep host validation bounded for R=token_budget*topk, rather than
            # materializing two complete FP32 FFN outputs after each timed pair.
            for first in range(0, capacity, PREFILL_VALIDATION_ROWS):
                last = first + PREFILL_VALIDATION_ROWS
                reference = outputs["fused"][first:last].cpu()
                actual = outputs["full_graph"][first:last].cpu()
                assert torch.isfinite(reference.float()).all() and torch.isfinite(actual.float()).all()
                assert torch.equal(actual.view(torch.int16), reference.view(torch.int16)), (
                    f"Device prefill differs from compact eager: {case_context}; rows={first}:{last}"
                )
                assert torch.count_nonzero(actual[max(0, active_rows - first) :]) == 0, "Padding must stay zero."
            candidate.validity_buffer.zero_()

        candidate.validity_buffer.zero_()
        runners = {
            mode: lambda bank=bank, hidden=hidden, groups=groups: bank.apply(
                hidden, groups, 1, config.get("swiglu_limit")
            )[0]
            for mode, bank in banks.items()
        }
        report = fusion_event_timing(
            torch,
            runners,
            validate,
            warmup=WARMUP_REPEATS,
            repeats=TIMING_REPEATS,
            reference_mode="fused",
            candidate_mode="full_graph",
        )
        print(
            "VQ2A8_GS32_EP2_CAPACITY_TIMING=" + json.dumps({**case_context, **report, "status": "PASS"}),
            flush=True,
        )
