# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in NPU coverage for immutable resident plans and dynamic buffer checks."""

import gc
import weakref

import pytest

from tests.e2e.nightly.test_vq2a8_gs32_full_ffn import FULL_FFN_PAIRS, load_runtime, selected_paths
from tests.e2e.nightly.vq2a8_gs32_full_ffn_helpers import select_fusion_expert
from tests.e2e.nightly.vq2a8_gs32_helpers import checkpoint_inventory


@pytest.fixture(scope="module")
def resident_runtime(request):
    paths = selected_paths(request)
    _, _, entries = checkpoint_inventory(paths["model"])
    selected = next(
        (candidate for pair in FULL_FFN_PAIRS if (candidate := select_fusion_expert(entries, pair)) is not None),
        None,
    )
    assert selected is not None, "Resident-plan acceptance requires a real supported precision pair."
    return load_runtime(paths, selected["layer"], (selected["position"],), with_oracle=False)


def prepare_plan(runtime, fields):
    torch, bank = runtime.torch, runtime.bank
    return bank.ops.prepare_full_ffn_plan(
        torch.empty((1, runtime.spec.specs["gate_up"].columns), dtype=torch.uint8, device=runtime.device),
        runtime.spec.specs["gate_up"].rows,
        runtime.spec.specs["down"].rows,
        bank.full_ffn_bank["gate_w4"],
        bank.full_ffn_bank["down_w4"],
        runtime.limit,
        fields,
    )


def launch_buffers(runtime):
    torch, bank = runtime.torch, runtime.bank
    input_width = runtime.spec.specs["gate_up"].columns
    middle_width = runtime.spec.specs["down"].columns
    output_width = runtime.spec.specs["down"].rows
    hidden = torch.full((1, input_width), 0.125, dtype=torch.bfloat16, device=runtime.device)
    data, scales, flags = bank.activation.prepare_unchecked(hidden)
    counts = torch.ones(len(runtime.spec.expert_ids), dtype=torch.int64, device=runtime.device)
    routes = bank.ops.prepare_device_routes(data, counts)
    middle = torch.empty((1, middle_width), dtype=torch.uint8, device=runtime.device)
    middle_scales = torch.empty((1, middle_width // 32), dtype=torch.uint8, device=runtime.device)
    output = torch.empty((1, output_width), dtype=torch.bfloat16, device=runtime.device)
    torch.npu.synchronize()
    assert bool(flags.cpu().all())
    assert bank.ops.device_routes_status(routes).cpu().tolist() == [1]
    return data, scales, routes, middle, middle_scales, output


@pytest.mark.parametrize("fault", ("dtype", "rank", "blob_size", "row_scale_shape", "weight_alias"))
def test_full_ffn_plan_rejects_invalid_resident_weights_at_prepare(resident_runtime, fault):
    runtime, torch = resident_runtime, resident_runtime.torch
    fields = dict(runtime.bank.full_ffn_bank["fields"])
    if fault == "dtype":
        fields["gate_weight"] = fields["gate_weight"].to(torch.int8)
    elif fault == "rank":
        fields["gate_weight"] = fields["gate_weight"].unsqueeze(0)
    elif fault == "blob_size":
        fields["gate_weight"] = fields["gate_weight"][:-32]
    elif fault == "row_scale_shape":
        fields["gate_row_scale"] = fields["gate_row_scale"][:, :-1].contiguous()
    else:
        fields["gate_table"] = fields["gate_weight"][: fields["gate_table"].numel()]
    # Preparation must validate static metadata before this plan can launch.
    with pytest.raises(RuntimeError, match="dtype|rank|dimensions|blob sizes|row scales|alias|overlap"):
        prepare_plan(runtime, fields)


@pytest.mark.parametrize("alias", ("input_output", "weight_output"))
def test_full_ffn_launch_still_rejects_dynamic_aliases(resident_runtime, alias):
    runtime, torch = resident_runtime, resident_runtime.torch
    data, scales, routes, middle, middle_scales, output = launch_buffers(runtime)
    if alias == "input_output":
        shared = torch.empty(max(data.numel(), output.numel() * 2), dtype=torch.uint8, device=runtime.device)
        data = shared[: data.numel()].reshape(data.shape)
        output = shared[: output.numel() * 2].view(torch.bfloat16).reshape(output.shape)
    else:
        weight = runtime.bank.full_ffn_bank["fields"]["gate_weight"]
        output = weight[: output.numel() * 2].view(torch.bfloat16).reshape(output.shape)
    with pytest.raises(RuntimeError, match="alias|overlap"):
        runtime.bank.ops.full_ffn_device(
            data, scales, routes, runtime.bank.full_ffn_plan, middle, middle_scales, output
        )


def test_full_ffn_plan_retains_weights_after_python_references_are_released(resident_runtime):
    runtime, torch = resident_runtime, resident_runtime.torch
    data, scales, routes, middle, middle_scales, output = launch_buffers(runtime)
    runtime.bank.ops.full_ffn_device(data, scales, routes, runtime.bank.full_ffn_plan, middle, middle_scales, output)
    torch.npu.synchronize()
    expected = output.cpu().clone()
    expected_middle, expected_scales = middle.cpu().clone(), middle_scales.cpu().clone()
    assert bool(torch.isfinite(expected).all())

    # The cloned storage has no owner in runtime.bank. Only the new C++ plan
    # may keep it alive after these original Python Tensor wrappers disappear.
    cloned = {name: value.clone() for name, value in runtime.bank.full_ffn_bank["fields"].items()}
    references = [weakref.ref(value) for value in cloned.values()]
    plan = prepare_plan(runtime, cloned)
    del cloned
    gc.collect()
    assert all(reference() is None for reference in references)
    runtime.bank.ops.full_ffn_device(data, scales, routes, plan, middle, middle_scales, output)
    torch.npu.synchronize()
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    assert torch.equal(middle.cpu(), expected_middle)
    assert torch.equal(middle_scales.cpu(), expected_scales)


def test_full_ffn_plan_is_independent_of_original_wrapper_metadata(resident_runtime):
    runtime, torch = resident_runtime, resident_runtime.torch
    data, scales, routes, middle, middle_scales, output = launch_buffers(runtime)
    runtime.bank.ops.full_ffn_device(data, scales, routes, runtime.bank.full_ffn_plan, middle, middle_scales, output)
    torch.npu.synchronize()
    expected = output.cpu().clone()
    expected_middle, expected_scales = middle.cpu().clone(), middle_scales.cpu().clone()

    cloned = {name: value.clone() for name, value in runtime.bank.full_ffn_bank["fields"].items()}
    plan = prepare_plan(runtime, cloned)
    original = {name: (value.shape, value.data_ptr()) for name, value in cloned.items()}
    # These edits affect only the caller's Tensor metadata, never weight bytes
    # or allocation lifetime. The resident plan keeps its own detached metadata.
    for name, value in cloned.items():
        if name.endswith("row_scale"):
            value.transpose_(0, 1)
        else:
            value.unsqueeze_(0)
        shape, pointer = original[name]
        assert value.shape != shape
        assert value.data_ptr() == pointer

    runtime.bank.ops.full_ffn_device(data, scales, routes, plan, middle, middle_scales, output)
    torch.npu.synchronize()
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    assert torch.equal(middle.cpu(), expected_middle)
    assert torch.equal(middle_scales.cpu(), expected_scales)
