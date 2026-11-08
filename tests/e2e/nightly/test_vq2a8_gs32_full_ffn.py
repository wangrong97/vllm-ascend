# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in real W2/W4 full FFN equivalence and dynamic NPU graph acceptance."""

import hashlib
import json
import platform
import statistics
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.e2e.nightly.vq2a8_gs32_full_ffn_helpers import (
    FUSION_FP8_MISMATCH_LIMIT,
    CapturedBank,
    full_inventory_route_patterns,
    fusion_byte_metrics,
    fusion_error_metrics,
    fusion_metrics_accepted,
    localize_profile_layer,
    measure_npu_submissions,
    npu_memory_snapshot,
    route_counts,
    select_full_ep_inventory,
    select_fusion_expert,
    tensor_storage_summary,
)
from tests.e2e.nightly.vq2a8_gs32_helpers import checkpoint_inventory, load_selected_layer, native_paths
from tests.support.vq2a8_oracle import OfficialFFNOracle

FULL_FFN_PAIRS = (("vq2", "vq2"), ("vq2", "mxfp4"), ("mxfp4", "mxfp4"))
GRAPH_CAPACITIES = (6, 128, 129, 384)
TIMING_WARMUPS = 3
TIMING_REPEATS = 5
PROFILE_EP_SIZE = 4
PROFILE_GRAPH_ROWS = (6, 384)
PROFILE_PREFILL_ROWS = 4096
PROFILE_REPLAY_STEPS = 100
PROFILE_MEMORY_INTERVAL = 20


def selected_paths(request):
    values = {
        name: request.config.getoption(flag, default=None)
        for name, flag in (
            ("model", "--vq2a8-model"),
            ("gs32_library", "--vq2a8-gs32-library"),
            ("gs32_binary", "--vq2a8-gs32-binary"),
        )
    }
    if not values["model"]:
        pytest.skip("Pass a real mixed source-v3 checkpoint and rebuilt full FFN native artifacts on Ascend950.")
    return native_paths(values)


def select_mixed_layer(entries):
    """Require one layer with all three pairs so one launch sees mixed offsets."""
    for layer_index, entry in sorted(entries.items()):
        selected = [select_fusion_expert({layer_index: entry}, pair) for pair in FULL_FFN_PAIRS]
        if all(item is not None for item in selected):
            return layer_index, tuple(sorted(item["position"] for item in selected))
    return None


class FullFFNLaunchCapture:
    """Observe the real native call and intermediate buffers without device copies."""

    def __init__(self, ops):
        self.ops = ops
        self.calls = 0
        self.middle = self.scales = None

    def __getattr__(self, name):
        return getattr(self.ops, name)

    def full_ffn_device(self, data, scales, routes, plan, middle, middle_scales, output):
        result = self.ops.full_ffn_device(data, scales, routes, plan, middle, middle_scales, output)
        self.calls += 1
        self.middle, self.scales = middle, middle_scales
        return result


def load_runtime(paths, layer_index, positions, *, execution="fused", placement=None, with_oracle=True):
    import torch
    import torch_npu

    from vllm_ascend.ops.fused_moe.vq2a8_gs32 import GS32ExpertWeights
    from vllm_ascend.quantization.methods.w2a8.vq2a8_gs32_full_ffn_layout import build_full_ffn_bank

    assert torch.npu.is_available(), "Explicit full FFN acceptance requires real Ascend hardware."
    device = torch.device("npu", 0)
    torch.npu.set_device(device)
    raw, spec, config = load_selected_layer(paths["model"], layer_index=layer_index, expert_positions=positions)
    if placement is not None:
        spec = localize_profile_layer(raw, spec, placement)
    oracle = OfficialFFNOracle(raw, spec, device) if with_oracle else None
    for name, value in vars(raw).copy().items():
        if isinstance(value, torch.Tensor):
            setattr(raw, name, value.to(device))
    limit = config.get("swiglu_limit") or 0.0
    raw.vq2a8_full_ffn_bank = build_full_ffn_bank(raw, spec, swiglu_limit=limit)
    for name in tuple(vars(raw)):
        if name.startswith(("w13_", "w2_")):
            delattr(raw, name)
    if placement is None:
        raw.vq2a8_ep_size = 2  # Permit zero-route EP ranks and padded tails.
    options = {key: value for key, value in paths.items() if key != "model"}
    bank = GS32ExpertWeights(
        raw,
        spec,
        {**options, "gs32_execution": execution, "gs32_fusion": "full_ffn"},
        graph_row_capacity=max(GRAPH_CAPACITIES),
    )
    launch = FullFFNLaunchCapture(bank.ops)
    bank.ops = launch
    return SimpleNamespace(
        torch=torch,
        torch_npu_version=torch_npu.__version__,
        device=device,
        layer=raw,
        spec=spec,
        limit=limit,
        bank=bank,
        oracle=oracle,
        launch=launch,
    )


def assert_outputs(runtime, expected, actual, context):
    metrics = fusion_error_metrics(runtime.torch, expected, actual)
    print("VQ2A8_GS32_FULL_FFN_OUTPUT=" + json.dumps({**context, **metrics}), flush=True)
    assert fusion_metrics_accepted(metrics), f"Full FFN exceeded existing fusion thresholds: {context}: {metrics}"


@pytest.mark.parametrize("pair", FULL_FFN_PAIRS, ids=lambda pair: "_to_".join(pair))
def test_gs32_real_experts_full_ffn_compare(request, pair):
    paths = selected_paths(request)
    _, _, entries = checkpoint_inventory(paths["model"])
    selected = select_fusion_expert(entries, pair)
    if selected is None:
        pytest.skip(f"No real {pair} expert in this checkpoint; this precision pair was not tested.")
    runtime = load_runtime(paths, selected["layer"], (selected["position"],))
    torch, spec = runtime.torch, runtime.spec
    generator = torch.Generator(device="cpu").manual_seed(95032)
    for rows in (1, 128, 129):
        hidden = (torch.randn(rows, spec.specs["gate_up"].columns, generator=generator) * 0.1).bfloat16()
        hidden = hidden.to(runtime.device)
        groups = route_counts(torch, runtime.layer, spec, (rows,), runtime.device, 1)
        expected, intermediate = runtime.oracle(hidden, groups, 1, runtime.limit)
        calls = runtime.launch.calls
        actual = runtime.bank.apply(hidden, groups, 1, runtime.limit)[0]
        torch.npu.synchronize()
        assert runtime.launch.calls == calls + 1, "The candidate must submit the actual full_ffn_device op."
        context = {**selected, "rows": rows, "execution": "fused"}
        assert_outputs(runtime, expected, actual, context)
        scales = fusion_byte_metrics(torch, intermediate["scales"], runtime.launch.scales.cpu())
        data = fusion_byte_metrics(torch, intermediate["data"], runtime.launch.middle.cpu())
        print("VQ2A8_GS32_FULL_FFN_MIDDLE=" + json.dumps({**context, "scales": scales, "data": data}), flush=True)
        assert scales["different"] == 0, "Full FFN changed the GS32 E8M0 scale algorithm."
        assert data["fraction"] <= FUSION_FP8_MISMATCH_LIMIT, "Full FFN changed too many intermediate FP8 bytes."


@pytest.mark.parametrize("group_type", (0, 1), ids=("prefix", "counts"))
def test_gs32_full_ffn_mixed_graph_routes_and_buckets(request, group_type):
    paths = selected_paths(request)
    _, _, entries = checkpoint_inventory(paths["model"])
    selected = select_mixed_layer(entries)
    if selected is None:
        pytest.skip("No single real layer contains all W2/W2, W2/W4 and W4/W4 pairs; mixed launch was not tested.")
    runtime = load_runtime(paths, *selected, execution="full_graph")
    eager_runtime = load_runtime(paths, *selected, execution="fused")
    torch, spec = runtime.torch, runtime.spec
    bank = runtime.bank
    generator = torch.Generator(device="cpu").manual_seed(950384)
    originals, captures, scale_buffers = {}, {}, {}
    # Capture with no active routes. Replay must update scheduling on device;
    # no Python branch may freeze an initially inactive precision stage.
    empty = route_counts(torch, runtime.layer, spec, (0, 0, 0), runtime.device, group_type)
    for capacity in GRAPH_CAPACITIES:
        hidden = (torch.randn(capacity, spec.specs["gate_up"].columns, generator=generator) * 0.1).bfloat16()
        originals[capacity] = hidden.to(runtime.device)
        captures[capacity] = CapturedBank(
            runtime, bank, originals[capacity], empty, group_type, validity=bank.validity_buffer
        )
        scale_buffers[capacity] = runtime.launch.scales
    assert len(bank.capture_owners) == len(GRAPH_CAPACITIES)
    assert len({capture.output.data_ptr() for capture in captures.values()}) == len(GRAPH_CAPACITIES)
    submissions = runtime.launch.calls
    assert submissions > 0 and bank.full_ffn_graph_captures == len(GRAPH_CAPACITIES)
    for capacity in (384, 6, 128, 129, 6, 384):
        hidden = originals[capacity]
        cases = (
            (0, 0, 0),
            (capacity, 0, 0),
            (0, capacity, 0),
            (0, 0, capacity),
            (capacity // 3, capacity // 3, capacity - 2 * (capacity // 3)),
            (1, 0, 1),
        )
        for counts in cases:
            groups = route_counts(torch, runtime.layer, spec, counts, runtime.device, group_type)
            inputs = hidden
            if counts == (1, 0, 1):
                inputs = hidden.clone()
                inputs[2:] = float("nan")
                inputs[-1] = float("inf")
            expected, _ = runtime.oracle(inputs, groups, group_type, runtime.limit)
            eager = eager_runtime.bank.apply(inputs, groups, group_type, runtime.limit)[0]
            actual = captures[capacity].replay_checked(inputs, groups)
            assert_outputs(
                runtime, expected, actual, {"capacity": capacity, "counts": counts, "group_type": group_type}
            )
            torch.testing.assert_close(actual, eager.cpu(), rtol=0, atol=0)
            active = sum(counts)
            assert torch.count_nonzero(actual[active:]) == 0
            scales = scale_buffers[capacity]
            assert bool((scales[active:].cpu() == 127).all()), "Graph padding must reset scales on every replay."
    assert runtime.launch.calls == submissions, "Graph replay must not resubmit through the Python launch wrapper."
    # A failed replay must not poison the next graph or retain a stale success flag.
    bad = originals[384].clone()
    bad[0, 0] = float("nan")
    groups = route_counts(torch, runtime.layer, spec, (384, 0, 0), runtime.device, group_type)
    with pytest.raises(ValueError, match="deferred validity"):
        captures[384].replay_checked(bad, groups)
    for bad_counts in ((-1, 1, 0), (385, 0, 0), (2**62, 2**62, 0)):
        invalid = route_counts(torch, runtime.layer, spec, bad_counts, runtime.device, group_type)
        with pytest.raises(ValueError, match="deferred validity"):
            captures[384].replay_checked(originals[384], invalid)
        # Every failed replay must clear its own status on the next valid call.
        recovered_groups = route_counts(torch, runtime.layer, spec, (0, 0, 6), runtime.device, group_type)
        captures[6].replay_checked(originals[6], recovered_groups)
    groups = route_counts(torch, runtime.layer, spec, (0, 0, 6), runtime.device, group_type)
    expected, _ = runtime.oracle(originals[6], groups, group_type, runtime.limit)
    recovered = captures[6].replay_checked(originals[6], groups)
    assert_outputs(runtime, expected, recovered, {"recovery": "large invalid to small W4/W4 graph"})
    # Record measurements without universal pass/fail thresholds. The eager
    # owner check runs outside timing, matching a real forward's deferred check.
    eager_runtime.bank.defer_validity = True
    try:
        for capacity in (min(GRAPH_CAPACITIES), max(GRAPH_CAPACITIES)):
            counts = (capacity // 3, capacity // 3, capacity - 2 * (capacity // 3))
            groups = route_counts(torch, runtime.layer, spec, counts, runtime.device, group_type)
            for _ in range(TIMING_WARMUPS):
                eager_runtime.bank.apply(originals[capacity], groups, group_type, runtime.limit)
                captures[capacity].replay_checked(originals[capacity], groups)
            torch.npu.synchronize()
            eager_ms, graph_ms = [], []
            for _ in range(TIMING_REPEATS):
                eager_runtime.bank.validity_buffer.zero_()
                begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                begin.record()
                eager_output = eager_runtime.bank.apply(originals[capacity], groups, group_type, runtime.limit)[0]
                end.record()
                torch.npu.synchronize()
                assert bool(eager_runtime.bank.validity_buffer.cpu())
                eager_ms.append(begin.elapsed_time(end))
                graph_output = captures[capacity].replay_checked(originals[capacity], groups, timing_ms=graph_ms)
                torch.testing.assert_close(graph_output, eager_output.cpu(), rtol=0, atol=0)
            print(
                "VQ2A8_GS32_FULL_FFN_TIMING="
                + json.dumps(
                    {
                        "capacity": capacity,
                        "group_type": group_type,
                        "counts": counts,
                        "warmups": TIMING_WARMUPS,
                        "repeats": TIMING_REPEATS,
                        "measurement": "NPUEvent; input copies and owner status read excluded",
                        "eager_ms": eager_ms,
                        "graph_ms": graph_ms,
                        "median_eager_ms": statistics.median(eager_ms),
                        "median_graph_ms": statistics.median(graph_ms),
                    }
                ),
                flush=True,
            )
    finally:
        eager_runtime.bank.defer_validity = False
    print(
        "VQ2A8_GS32_FULL_FFN_GRAPH="
        + json.dumps(
            {
                "status": "PASS",
                "experts": spec.expert_ids,
                "group_type": group_type,
                "capacities": GRAPH_CAPACITIES,
                "native_submissions": submissions,
                "captures": bank.full_ffn_graph_captures,
                "replays": sum(capture.replays for capture in captures.values()),
            }
        ),
        flush=True,
    )


def profile_file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_gs32_full_ffn_full_ep_inventory_profile(request, tmp_path):
    """Single-NPU full EP-local inventory; excludes communication and attention."""
    paths = selected_paths(request)
    config, _, entries = checkpoint_inventory(paths["model"])
    placement = select_full_ep_inventory(entries, config["n_routed_experts"], ep_size=PROFILE_EP_SIZE)
    assert placement is not None, "The requested checkpoint must contain a nonempty EP-local inventory."
    runtime = load_runtime(
        paths,
        placement["layer"],
        placement["positions"],
        execution="full_graph",
        placement=placement,
        with_oracle=False,
    )
    torch, bank, spec = runtime.torch, runtime.bank, runtime.spec
    assert spec.expert_ids == placement["local_ids"]
    assert len(bank.full_ffn_expert_ids) == len(placement["global_ids"])
    assert bank.num_experts == placement["num_local_experts"]
    bank.defer_validity = True
    generator = torch.Generator(device="cpu").manual_seed(950640)
    hidden_by_rows, captures, routes_by_rows, expected_by_case = {}, {}, {}, {}
    group_type = 1
    for rows in PROFILE_GRAPH_ROWS:
        hidden_by_rows[rows] = (
            (torch.randn(rows, spec.specs["gate_up"].columns, generator=generator) * 0.1).bfloat16().to(runtime.device)
        )
        patterns = full_inventory_route_patterns(spec, rows)
        routes_by_rows[rows] = [
            (name, counts, route_counts(torch, runtime.layer, spec, counts, runtime.device, group_type))
            for name, counts in patterns
        ]
        empty = routes_by_rows[rows][0][2]
        captures[rows] = CapturedBank(
            runtime, bank, hidden_by_rows[rows], empty, group_type, validity=bank.validity_buffer, measure_capture=True
        )
    torch.npu.synchronize()
    report = {
        "schema_version": 1,
        "scope": (
            "Single NPU, complete real EP-local FFN inventory; excludes dispatch/collectives, "
            "attention, sampling and end-to-end service latency"
        ),
        "numerical_scope": (
            "Full inventory eager/graph equality; independent official-GMM and intermediate-byte "
            "validation remains in the separate real precision-pair tests"
        ),
        "timing_scope": (
            "NPU events include validity reset and full FFN submission/replay; host intervals include submission "
            "only. Warmup, synchronization, status reads and input copies are outside those intervals. "
            "Capture wall time includes capture finalization and device synchronization."
        ),
        "environment": {
            "python": platform.python_version(),
            "host_arch": platform.machine(),
            "torch": torch.__version__,
            "torch_npu": runtime.torch_npu_version,
            "device": str(runtime.device),
            "device_name": torch.npu.get_device_name(runtime.device),
        },
        "model_path": paths["model"],
        "config_sha256": profile_file_hash(Path(paths["model"]) / "config.json"),
        "index_sha256": profile_file_hash(Path(paths["model"]) / "model.safetensors.index.json"),
        "library_sha256": profile_file_hash(paths["gs32_library"]),
        "kernel_sha256": profile_file_hash(paths["gs32_binary"]),
        "build_manifest_sha256": profile_file_hash(Path(paths["gs32_library"]).parent / "build-manifest.json"),
        "placement": placement,
        "dimensions": {
            "input": spec.specs["gate_up"].columns,
            "middle": spec.specs["down"].columns,
            "output": spec.specs["down"].rows,
        },
        "captures": {str(rows): capture.capture_metrics for rows, capture in captures.items()},
        "resident_weights": tensor_storage_summary(torch, bank.full_ffn_bank["fields"]),
        "retained_capture_storage": tensor_storage_summary(torch, bank.capture_owners),
        "timings": [],
    }

    def check_validity():
        assert bool(bank.validity_buffer.cpu()), "The measured full FFN must pass its external owner validity check."

    for rows in PROFILE_GRAPH_ROWS:
        for name, counts, groups in routes_by_rows[rows]:
            bank.validity_buffer.zero_()
            eager = bank.apply(hidden_by_rows[rows], groups, group_type, runtime.limit)[0]
            torch.npu.synchronize()
            check_validity()
            expected = eager.cpu().clone()
            expected_by_case[rows, name] = expected
            graph_output = captures[rows].replay_checked(hidden_by_rows[rows], groups)
            torch.testing.assert_close(graph_output, expected, rtol=0, atol=0)
            assert bool(torch.isfinite(expected).all())
            assert not expected[sum(counts) :].count_nonzero()

            def eager_submit(hidden=hidden_by_rows[rows], current_groups=groups):
                bank.validity_buffer.zero_()
                return bank.apply(hidden, current_groups, group_type, runtime.limit)[0]

            def graph_submit(capture=captures[rows]):
                capture.validity.zero_()
                capture.graph.replay()

            report["timings"].append(
                {
                    "phase": "decode",
                    "rows": rows,
                    "pattern": name,
                    "counts": counts,
                    "eager": measure_npu_submissions(
                        runtime, eager_submit, check_validity, warmups=TIMING_WARMUPS, repeats=TIMING_REPEATS
                    ),
                    "graph": measure_npu_submissions(
                        runtime, graph_submit, check_validity, warmups=TIMING_WARMUPS, repeats=TIMING_REPEATS
                    ),
                }
            )
    # Prefill stays eager, as in serving. It is deliberately not captured into
    # an artificial 4096-row graph that the production configuration never uses.
    prefill = (
        (torch.randn(PROFILE_PREFILL_ROWS, spec.specs["gate_up"].columns, generator=generator) * 0.1)
        .bfloat16()
        .to(runtime.device)
    )
    for name, counts in full_inventory_route_patterns(spec, PROFILE_PREFILL_ROWS):
        if name not in ("balanced", "skewed"):
            continue
        groups = route_counts(torch, runtime.layer, spec, counts, runtime.device, group_type)

        def prefill_submit(current_groups=groups):
            bank.validity_buffer.zero_()
            return bank.apply(prefill, current_groups, group_type, runtime.limit)[0]

        report["timings"].append(
            {
                "phase": "prefill_eager",
                "rows": PROFILE_PREFILL_ROWS,
                "pattern": name,
                "counts": counts,
                "eager": measure_npu_submissions(
                    runtime, prefill_submit, check_validity, warmups=TIMING_WARMUPS, repeats=TIMING_REPEATS
                ),
            }
        )
    # No per-replay CPU result copy or event allocation in this stress loop.
    # Every replay's status is retained on device, so a later success cannot
    # hide an earlier failure. Changing routes alternates both graph buckets.
    history = torch.empty(PROFILE_REPLAY_STEPS, dtype=torch.bool, device=runtime.device)
    owner_count = len(bank.capture_owners)
    storage_before = tensor_storage_summary(torch, bank.capture_owners)
    submissions = runtime.launch.calls
    pointers = {rows: capture.addresses for rows, capture in captures.items()}
    torch.npu.synchronize()
    torch.npu.reset_peak_memory_stats(runtime.device)
    samples = [{"step": 0, **npu_memory_snapshot(torch, runtime.device)}]
    last_cases = {}
    for step in range(PROFILE_REPLAY_STEPS):
        rows = PROFILE_GRAPH_ROWS[step % len(PROFILE_GRAPH_ROWS)]
        patterns = routes_by_rows[rows]
        name, _, groups = patterns[(step // len(PROFILE_GRAPH_ROWS)) % len(patterns)]
        capture = captures[rows]
        capture.groups.copy_(groups)
        capture.validity.zero_()
        capture.graph.replay()
        history[step].copy_(capture.validity)
        last_cases[rows] = name
        if (step + 1) % PROFILE_MEMORY_INTERVAL == 0:
            torch.npu.synchronize()
            samples.append({"step": step + 1, **npu_memory_snapshot(torch, runtime.device)})
    torch.npu.synchronize()
    assert bool(history.cpu().all())
    assert runtime.launch.calls == submissions, "Graph replay must not submit through the Python FFN wrapper."
    assert len(bank.capture_owners) == owner_count
    assert tensor_storage_summary(torch, bank.capture_owners) == storage_before
    for rows, name in last_cases.items():
        capture = captures[rows]
        assert pointers[rows] == (
            capture.hidden.data_ptr(),
            capture.groups.data_ptr(),
            capture.output.data_ptr(),
            capture.validity.data_ptr(),
        )
        torch.testing.assert_close(capture.output.cpu(), expected_by_case[rows, name], rtol=0, atol=0)
    # Reserved memory is allocator cache and is reported, not equated to live
    # tensor ownership. Live allocations should plateau after full warmup.
    allocated_stable = all(sample["allocated_bytes"] == samples[0]["allocated_bytes"] for sample in samples)
    report["replay_stability"] = {
        "steps": PROFILE_REPLAY_STEPS,
        "samples": samples,
        "allocated_stable": allocated_stable,
        "capture_owner_count": owner_count,
        "storage": storage_before,
    }
    report["status"] = "PASS" if allocated_stable else "ALLOCATION_GROWTH"
    destination = tmp_path / "full-ep-ffn-profile.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("VQ2A8_GS32_FULL_EP_PROFILE=" + json.dumps({"report_file": str(destination), **report}), flush=True)
    assert allocated_stable, f"Repeated graph replay changed live allocation bytes after warmup; inspect {destination}."
