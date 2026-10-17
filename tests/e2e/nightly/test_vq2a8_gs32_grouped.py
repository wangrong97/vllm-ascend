# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in real-NPU grouped launches, without loading a complete model.

Exercise the first and last resident slots across route-mask word boundaries.
The fixture banks occupy at most a few MiB. CPU codebook reconstruction checks
the mathematical result; separate G=1 launches check grouped output bitwise.
Neither oracle uses the grouped implementation to produce expected results.
The large-M cases load three real source-v3 VQ2 experts for each projection
and compare one grouped launch with the unchanged <=128-row serial adapter.
Use ``-k 'large_m and smoke'`` for the short gate, or ``-k large_m`` for all
M boundaries. ``-k beyond_4096`` covers the former expert-size limit and
``-k beyond_32768`` covers the former aggregate routed-row limit.
Timings exclude input preparation and do not assert speedups.
"""

import json
import statistics
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

LARGE_M_GROUPS = 3
SERIAL_ROW_LIMIT = 128
ACTIVATION_PREPARATION_ROWS = 4096
OUTPUT_COMPARISON_ROWS = 4096
TIMING_REPEATS = 3
LARGE_M_CASES = (
    pytest.param((1, 0, 129), id="smoke-m129"),
    pytest.param((255, 0, 1), id="m255"),
    pytest.param((1, 256, 1), id="m256"),
    pytest.param((0, 257, 1), id="smoke-m257"),
    pytest.param((1, 0, 479), id="m479"),
    pytest.param((480, 1, 0), id="m480"),
    pytest.param((1, 481, 0), id="smoke-m481"),
    pytest.param((512, 1, 512), id="m512"),
    pytest.param((0, 1, 1024), id="m1024"),
    pytest.param((2048, 0, 1), id="m2048"),
    pytest.param((1, 0, 4096), id="m4096"),
    pytest.param((0, 4097, 1), id="beyond_4096-m4097"),
    pytest.param((8191, 1, 0), id="beyond_4096-m8191"),
    pytest.param((1, 0, 8192), id="beyond_4096-m8192"),
    pytest.param((0, 8193, 1), id="beyond_4096-m8193"),
    pytest.param((16383, 0, 1), id="beyond_4096-m16383"),
    pytest.param((0, 1, 16384), id="beyond_4096-m16384"),
    pytest.param((1, 16385, 0), id="beyond_4096-m16385"),
    pytest.param((0, 0, 32768), id="beyond_4096-m32768"),
    pytest.param((16384, 0, 16384), id="beyond_4096-two-m16384"),
    pytest.param((0, 32769, 0), id="beyond_32768-m32769"),
    pytest.param((65535, 0, 0), id="beyond_32768-m65535"),
    pytest.param((0, 0, 65536), id="beyond_32768-m65536"),
    pytest.param((0, 65537, 0), id="beyond_32768-m65537"),
    pytest.param((32768, 0, 32768), id="beyond_32768-two-m32768"),
    pytest.param((32768, 32768, 32768), id="beyond_32768-three-m32768"),
)


@pytest.fixture(scope="module")
def grouped_runtime(request):
    options = {
        name: request.config.getoption(flag, default=None)
        for name, flag in (
            ("model", "--vq2a8-model"),
            ("library", "--vq2a8-gs32-library"),
            ("binary", "--vq2a8-gs32-binary"),
        )
    }
    if not options["model"]:
        pytest.skip("Pass --vq2a8-model and both --vq2a8-gs32-* paths for real Ascend grouped execution.")
    assert all(options.values()), "Explicit grouped tests require both native artifact paths."
    paths = {name: Path(value).resolve(strict=True) for name, value in options.items()}
    config = json.loads((paths["model"] / "config.json").read_text())
    assert config["quantization_config"]["format"] == "vq2_mixed_gs32_source"

    # Imports follow the hardware opt-in; no CPU/fake native execution fallback.
    import torch
    import torch_npu  # noqa: F401

    assert torch.npu.is_available(), "Explicit grouped tests require a real Ascend NPU."
    device = torch.device("npu", 0)
    torch.npu.set_device(device)
    torch.ops.load_library(str(paths["library"]))
    ops = torch.ops.vq2a8_gs32_stage
    for name in ("initialize", "prepare", "project_prepared", "prepare_routes", "prepare_grouped", "project_grouped"):
        assert hasattr(ops, name), f"Rebuild the GS32 bridge: missing {name}."
    assert hasattr(ops, "supports_dynamic_grouped_rows"), (
        "Rebuild the GS32 bridge: missing supports_dynamic_grouped_rows."
    )
    assert ops.supports_dynamic_grouped_rows() is True, (
        "Rebuild the GS32 bridge for grouped execution without fixed expert or aggregate row limits."
    )
    ops.initialize(str(paths["binary"]))
    return SimpleNamespace(torch=torch, device=device, ops=ops, model=paths["model"])


def source_matrix(slot, k, n):
    """Small real producer payloads, with different values in each bank slot."""
    import torch

    from tests.ut.quantization.test_vq2a8_gs32_layout import fixture

    payload, spec, _ = fixture("gate_up" if n == 128 else "down")
    assert n == spec.rows and k in (64, 128)
    spec = replace(spec, columns=k)
    # packed_indices/codebooks both have K tiles as their outermost axis.
    payload["packed_indices"] = payload["packed_indices"].repeat(k // 64).roll(slot)
    books = payload["codebooks"].view(torch.uint8).repeat(k // 64, 1, 1, 1)
    # Avoid FP8 special encodings while distinguishing physical expert slots.
    payload["codebooks"] = ((books.to(torch.int32) + slot * 7) % 120).to(torch.uint8).view(torch.float8_e4m3fn)
    payload["perm"] = torch.roll(torch.arange(k, dtype=torch.int32), 23 + slot)
    payload["weight_scale"] = torch.linspace(0.5, 1.5, k) * (1 + slot / 512)
    payload["weight_bias"] = torch.zeros(k)
    payload["row_scale"] *= 1 + slot / 256
    return payload, spec


def build_case(groups, pattern, k, n):
    """Only active rows get CPU GEMM references; all resident slots are packed."""
    import torch

    from tests.ut.quantization.vq2a8_gs32_reference import (
        prepare_gs32_activation_reference,
        quantize_mxfp8_k32,
        repack_gs32_matrix,
        staged_projection_reference,
    )

    counts = torch.zeros(groups, dtype=torch.int64)
    counts[-1] = 1 if pattern == "last" else 2
    if pattern == "edges" and groups > 1:
        counts[0] = 1
    fields, inputs, scales, references, active = [], [], [], [], []
    generator = torch.Generator(device="cpu").manual_seed(7301)
    for slot, count in enumerate(counts.tolist()):
        payload, spec = source_matrix(slot, k, n)
        fields.append(repack_gs32_matrix(payload, spec, down_stage=True))
        if count:
            hidden = (torch.randn(count, k, generator=generator) / 16).bfloat16()
            transformed = prepare_gs32_activation_reference(hidden, payload, spec)
            data, scale = quantize_mxfp8_k32(transformed)
            inputs.append(data)
            scales.append(scale)
            references.append(staged_projection_reference(hidden, payload, spec))
            active.append((slot, count))
    banks = (
        torch.stack([entry["packed_zn"] for entry in fields]),
        torch.stack([entry["pair_lut"] for entry in fields]),
        torch.stack([entry["row_scale_bits"].view(torch.int64) for entry in fields]),
    )
    return SimpleNamespace(
        counts=counts,
        active=active,
        data=torch.cat(inputs),
        scales=torch.cat(scales),
        banks=banks,
        expected=torch.cat(references),
        groups=groups,
        k=k,
        n=n,
    )


def run_case(runtime, case, *, other_stream=False):
    torch, ops, device = runtime.torch, runtime.ops, runtime.device
    data, scales = case.data.to(device), case.scales.to(device)
    banks = tuple(field.to(device) for field in case.banks)
    rows = data.shape[0]
    plan = ops.prepare_grouped(data, case.n, case.groups)
    counts = case.counts.clone()
    routes = ops.prepare_routes(data, counts)
    # The handle must own a validated snapshot, not the caller's mutable tensor.
    counts.fill_(128)
    output = torch.full((rows, case.n), float("nan"), dtype=torch.bfloat16, device=device)
    repeated = torch.full_like(output, float("nan"))
    torch.npu.synchronize()
    stream = torch.npu.Stream(device=device) if other_stream else torch.npu.current_stream(device)
    with torch.npu.stream(stream):
        ops.project_grouped(data, scales, *banks, routes, plan, output)
        # Reusing route/plan owners must not mutate group offsets between calls.
        ops.project_grouped(data, scales, *banks, routes, plan, repeated)
    stream.synchronize()
    actual = output.cpu()
    assert torch.isfinite(actual.float()).all(), "A grouped row/output tile was not written."
    assert torch.equal(actual.view(torch.int16), repeated.cpu().view(torch.int16))

    serial, offset = [], 0
    for slot, count in case.active:
        # Fresh allocations keep every scale slice aligned even when K/32 <32.
        expert_data = case.data[offset : offset + count].clone().to(device)
        expert_scales = case.scales[offset : offset + count].clone().to(device)
        expert_banks = tuple(field[slot : slot + 1] for field in banks)
        expert_output = torch.full((count, case.n), float("nan"), dtype=torch.bfloat16, device=device)
        single_plan = ops.prepare(expert_data, case.n)
        ops.project_prepared(expert_data, expert_scales, *expert_banks, single_plan, expert_output)
        torch.npu.synchronize()
        serial.append(expert_output.cpu())
        offset += count
    assert offset == rows
    assert torch.equal(actual.view(torch.int16), torch.cat(serial).view(torch.int16)), "Grouped differs from G=1."
    # CPU FP32 accumulation order may differ from the native GEMM. Apply the
    # established stage-probe tolerance to every element, not just a mean.
    torch.testing.assert_close(actual.float(), case.expected.float(), rtol=0.01, atol=0.01)


@pytest.mark.parametrize("groups", [1, 3, 33, 256])
@pytest.mark.parametrize("pattern", ["last", "edges"])
@pytest.mark.parametrize("k,n", [(64, 64), (128, 128)])
def test_gs32_grouped_zero_slots_and_route_mask_boundary(grouped_runtime, groups, pattern, k, n):
    # G33 reads mask word1; G256 reads word7. Zero groups still advance bank
    # offsets and read route bits, so a single trailing live group is essential.
    run_case(grouped_runtime, build_case(groups, pattern, k, n))


def test_gs32_grouped_plan_and_owned_routes_on_nondefault_stream(grouped_runtime):
    run_case(grouped_runtime, build_case(33, "edges", 128, 128), other_stream=True)


@pytest.mark.parametrize(
    "fault", ["float_counts", "device_counts", "negative", "wrong_sum", "count_exceeds_rows", "int64_overflow"]
)
def test_gs32_grouped_routes_reject_invalid_counts_before_launch(grouped_runtime, fault):
    torch, ops, device = grouped_runtime.torch, grouped_runtime.ops, grouped_runtime.device
    rows = 3
    data = torch.empty((rows, 64), dtype=torch.uint8, device=device)
    counts = torch.tensor([1, 0, 2], dtype=torch.int64)
    if fault == "float_counts":
        counts = counts.float()
    elif fault == "device_counts":
        counts = counts.to(device)
    elif fault == "negative":
        counts = torch.tensor([1, -1, 3], dtype=torch.int64)
    elif fault == "wrong_sum":
        counts = torch.tensor([1, 0, 1], dtype=torch.int64)
    elif fault == "count_exceeds_rows":
        counts = torch.tensor([0, 0, rows + 1], dtype=torch.int64)
    elif fault == "int64_overflow":
        # These counts wrap to -1 if summed as int64 without bounded-prefix
        # validation. Reject them against actual rows before any native launch.
        largest = torch.iinfo(torch.int64).max
        counts = torch.tensor([largest, largest, 1], dtype=torch.int64)
    with pytest.raises(RuntimeError, match="dtype|CPU metadata|counts|count|rows"):
        ops.prepare_routes(data, counts)
    torch.npu.synchronize()


def test_gs32_grouped_schema_rejects_raw_counts_in_place_of_owned_routes(grouped_runtime):
    torch, ops, device = grouped_runtime.torch, grouped_runtime.ops, grouped_runtime.device
    case = build_case(3, "edges", 64, 64)
    data, scales = case.data.to(device), case.scales.to(device)
    banks = tuple(field.to(device) for field in case.banks)
    output = torch.full((3, 64), float("nan"), dtype=torch.bfloat16, device=device)
    plan = ops.prepare_grouped(data, 64, 3)
    with pytest.raises((RuntimeError, TypeError), match="GroupedRoutes"):
        ops.project_grouped(data, scales, *banks, case.counts, plan, output)
    torch.npu.synchronize()
    assert torch.isnan(output.cpu().float()).all(), "Invalid routes must not submit a kernel."


@pytest.fixture(scope="module", params=("gate_up", "down"))
def real_grouped_projection(grouped_runtime, request):
    from tests.e2e.nightly.vq2a8_gs32_helpers import checkpoint_inventory, load_selected_layer

    runtime, kind = grouped_runtime, request.param
    torch = runtime.torch
    _, _, inventory = checkpoint_inventory(runtime.model)
    selected = None
    for layer_index, entry in sorted(inventory.items()):
        experts = tuple(sorted(entry[kind]))
        positions = tuple(index for index, expert in enumerate(experts) if entry[kind][expert] == "vq2")
        if len(positions) >= LARGE_M_GROUPS:
            selected = layer_index, positions[:LARGE_M_GROUPS]
            break
    assert selected is not None, (
        f"Need three real VQ2 {kind} experts in one layer; synthetic substitution is forbidden."
    )
    layer, layer_spec, _ = load_selected_layer(runtime.model, layer_index=selected[0], expert_positions=selected[1])
    spec = layer_spec.specs[kind]
    prefix = "w13" if kind == "gate_up" else "w2"
    banks = tuple(
        getattr(layer, f"{prefix}_{field}").to(runtime.device) for field in ("packed_zn", "pair_lut", "row_scale_bits")
    )
    # Preserve the real source permutation and out_dim_scale; no fixture
    # substitutes identity permutations or unit row scales for this gate.
    permutation = getattr(layer, f"{prefix}_perm")
    anchor = torch.empty((1, spec.columns), dtype=torch.uint8, device=runtime.device)
    plan = runtime.ops.prepare_grouped(anchor, spec.rows, LARGE_M_GROUPS)
    torch.npu.synchronize()
    yield SimpleNamespace(
        runtime=runtime,
        kind=kind,
        layer=layer_spec.layer_index,
        experts=layer_spec.expert_ids,
        k=spec.columns,
        n=spec.rows,
        banks=banks,
        permutation=permutation,
        plan=plan,
    )
    torch.npu.synchronize()


def large_m_inputs(projection, counts):
    from tests.ut.quantization.vq2a8_gs32_reference import quantize_mxfp8_k32

    torch, inputs, scales = projection.runtime.torch, [], []
    generator = torch.Generator(device="cpu").manual_seed(7315)
    for slot, count in enumerate(counts):
        # Bound CPU quantization temporaries while keeping a single native
        # grouped input with the full requested expert count.
        for start in range(0, count, ACTIVATION_PREPARATION_ROWS):
            rows = min(ACTIVATION_PREPARATION_ROWS, count - start)
            hidden = (torch.randn(rows, projection.k, generator=generator) / 16).bfloat16()
            transformed = hidden.index_select(1, projection.permutation[slot].long())
            data, scale = quantize_mxfp8_k32(transformed)
            inputs.append(data)
            scales.append(scale)
    return torch.cat(inputs).to(projection.runtime.device), torch.cat(scales).to(projection.runtime.device)


def serial_projection_jobs(projection, data, scales, output, counts):
    """Prepare <=128-row G=1 owners once, outside the measured launch loop."""
    jobs, plans, offset = [], {}, 0
    ops = projection.runtime.ops
    for slot, count in enumerate(counts):
        fields = tuple(bank[slot : slot + 1] for bank in projection.banks)
        for start in range(0, count, SERIAL_ROW_LIMIT):
            first, last = offset + start, offset + min(start + SERIAL_ROW_LIMIT, count)
            if last - first not in plans:
                plans[last - first] = ops.prepare(data[first:last], projection.n)
            jobs.append((data[first:last], scales[first:last], *fields, plans[last - first], output[first:last]))
        offset += count
    assert offset == data.shape[0]
    return jobs


def time_projection(torch, launch):
    """Report synchronized submission wall time, including Python launch cost."""
    timings = []
    for _ in range(TIMING_REPEATS):
        torch.npu.synchronize()
        start = time.perf_counter()
        launch()
        torch.npu.synchronize()
        timings.append((time.perf_counter() - start) * 1000)
    return timings


def compare_large_m_outputs(torch, output, serial):
    """Check every element without materializing full-size CPU FP32 copies."""
    mismatched_elements, max_abs_error = 0, 0.0
    for first in range(0, output.shape[0], OUTPUT_COMPARISON_ROWS):
        last = first + OUTPUT_COMPARISON_ROWS
        actual, expected = output[first:last].cpu(), serial[first:last].cpu()
        actual_float, expected_float = actual.float(), expected.float()
        assert torch.isfinite(actual_float).all(), "Large-M grouped output contains non-finite or unwritten rows."
        assert torch.isfinite(expected_float).all(), "The <=128-row serial baseline must be finite."
        mismatched_elements += int((actual.view(torch.int16) != expected.view(torch.int16)).sum())
        max_abs_error = max(max_abs_error, float((actual_float - expected_float).abs().max()))
    return mismatched_elements, max_abs_error


@pytest.mark.parametrize("counts", LARGE_M_CASES)
def test_gs32_grouped_real_large_m(real_grouped_projection, counts):
    projection = real_grouped_projection
    runtime = projection.runtime
    torch, ops = runtime.torch, runtime.ops
    data, scales = large_m_inputs(projection, counts)
    output = torch.full((sum(counts), projection.n), float("nan"), dtype=torch.bfloat16, device=runtime.device)
    serial = torch.full_like(output, float("nan"))
    caller_counts = torch.tensor(counts, dtype=torch.int64)
    routes = ops.prepare_routes(data, caller_counts)
    caller_counts.zero_()  # The route owner must retain its validated snapshot.
    jobs = serial_projection_jobs(projection, data, scales, serial, counts)

    def grouped_launch():
        # The same module-owned plan was made for M=1 and is reused across
        # every large-M shape; only route owners and activations are per call.
        ops.project_grouped(data, scales, *projection.banks, routes, projection.plan, output)

    def serial_launch():
        for job in jobs:
            ops.project_prepared(*job)

    grouped_launch()
    serial_launch()
    torch.npu.synchronize()
    mismatched_elements, max_abs_error = compare_large_m_outputs(torch, output, serial)
    print(
        "GS32_GROUPED_LARGE_M_ACCURACY="
        + json.dumps(
            {
                "projection": projection.kind,
                "layer": projection.layer,
                "experts": projection.experts,
                "counts": counts,
                "k": projection.k,
                "n": projection.n,
                "plan_anchor_rows": 1,
                "mismatched_elements": mismatched_elements,
                "max_abs_error": max_abs_error,
            }
        ),
        flush=True,
    )
    # Preserve the existing grouped/G=1 exactness gate. If larger native
    # tiling changes rounding, retain the diagnostics for review instead of
    # silently loosening the acceptance threshold before hardware evidence.
    assert mismatched_elements == 0, "Large-M grouped projection differs from the <=128-row G=1 baseline."
    grouped_ms = time_projection(torch, grouped_launch)
    serial_ms = time_projection(torch, serial_launch)
    print(
        "GS32_GROUPED_LARGE_M_TIMING="
        + json.dumps(
            {
                "projection": projection.kind,
                "counts": counts,
                "serial_launches": len(jobs),
                "grouped_ms": grouped_ms,
                "serial_ms": serial_ms,
                "median_grouped_ms": statistics.median(grouped_ms),
                "median_serial_ms": statistics.median(serial_ms),
                "scope": "warm projection submission+execution; excludes routes, activation preparation and transfers",
            }
        ),
        flush=True,
    )
