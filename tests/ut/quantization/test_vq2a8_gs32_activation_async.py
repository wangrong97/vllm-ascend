# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full FFN input preparation contracts; CPU transport is not NPU proof."""

import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.ut.quantization.test_vq2a8_gs32_device_routes import evaluate_row_kernel
from tests.ut.quantization.test_vq2a8_gs32_fused_activation import activation_module, tagged


@pytest.fixture
def activation(monkeypatch):
    monkeypatch.setattr(torch, "npu", SimpleNamespace(device=lambda device: nullcontext()), raising=False)
    return activation_module()


def test_prepare_rejects_cpu(activation):
    with pytest.raises(ValueError, match="NPU"):
        activation.prepare_unchecked(torch.zeros(3, 64, dtype=torch.bfloat16))


def test_prepare_preserves_device_flags_and_original_buffers(activation, monkeypatch):
    hidden = tagged(torch.zeros(3, 64, dtype=torch.bfloat16))
    rows = tagged(torch.tensor([2, 0, -1], dtype=torch.int64))
    route = tagged(torch.tensor([1], dtype=torch.int32))
    sentinel = object()
    calls = []

    def launch(received, **options):
        calls.append((received, options))
        return sentinel

    monkeypatch.setattr(activation, "prepare_activation", launch)
    assert activation.prepare_unchecked(hidden, source_rows=rows, route_flags=route) is sentinel
    assert calls[0][0] is hidden
    assert calls[0][1]["source_rows"] is rows
    assert calls[0][1]["route_flags"] is route


@pytest.mark.parametrize("fault", ["dtype", "width", "row_dtype", "row_shape", "flag_dtype", "flag_shape", "device"])
def test_prepare_rejects_bad_metadata_before_launch(activation, monkeypatch, fault):
    hidden = tagged(torch.zeros(3, 64, dtype=torch.float32 if fault == "dtype" else torch.bfloat16))
    if fault == "width":
        hidden = tagged(torch.zeros(3, 63, dtype=torch.bfloat16))
    rows = tagged(
        torch.zeros(2 if fault == "row_shape" else 3, dtype=torch.int32 if fault == "row_dtype" else torch.int64)
    )
    flags = tagged(
        torch.ones(2 if fault == "flag_shape" else 1, dtype=torch.int64 if fault == "flag_dtype" else torch.int32),
        index=1 if fault == "device" else 0,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid metadata reached device launch")

    monkeypatch.setattr(activation, "prepare_activation", forbidden)
    with pytest.raises(ValueError):
        activation.prepare_unchecked(hidden, source_rows=rows, route_flags=flags)


def test_empty_prepare_does_not_launch(activation, monkeypatch):
    monkeypatch.setattr(activation, "prepare_activation", lambda *args, **kwargs: pytest.fail("empty launch"))
    result = activation.prepare_unchecked(tagged(torch.zeros(0, 64, dtype=torch.bfloat16)))
    assert [tuple(value.shape) for value in result] == [(0, 64), (0, 2), (0,)]
    assert [value.dtype for value in result] == [torch.uint8, torch.uint8, torch.int32]


@pytest.mark.parametrize("legacy", ["raw_gate", "swiglu_limit", "perm"])
def test_removed_partial_fusion_arguments_fail(activation, legacy):
    with pytest.raises(TypeError):
        activation.prepare_unchecked(tagged(torch.zeros(3, 64, dtype=torch.bfloat16)), **{legacy: None})


def test_identity_prepare_forwards_device_summary_without_host_reads(activation, monkeypatch):
    hidden = tagged(torch.zeros(3, 64, dtype=torch.bfloat16))
    summary = tagged(torch.tensor([2, 1], dtype=torch.int64))
    route = tagged(torch.ones(1, dtype=torch.int32))
    sentinel = object()

    def launch(received, *, source_rows, route_flags, route_metadata):
        assert received is hidden and source_rows is None
        assert route_flags is route and route_metadata is summary
        return sentinel

    monkeypatch.setattr(activation, "prepare_activation", launch)
    monkeypatch.setattr(type(summary), "item", lambda *args: pytest.fail("device scalar read"))
    monkeypatch.setattr(type(summary), "cpu", lambda *args: pytest.fail("device host transfer"))
    assert activation.prepare_unchecked(hidden, route_flags=route, route_metadata=summary) is sentinel


@pytest.mark.parametrize("fault", ["dtype", "shape", "device", "noncontiguous", "both_mappings"])
def test_identity_prepare_rejects_invalid_summary_before_launch(activation, monkeypatch, fault):
    hidden = tagged(torch.zeros(3, 64, dtype=torch.bfloat16))
    summary = tagged(
        torch.tensor([2, 1], dtype=torch.int32 if fault == "dtype" else torch.int64), index=int(fault == "device")
    )
    if fault == "shape":
        summary = tagged(torch.ones(3, dtype=torch.int64))
    if fault == "noncontiguous":
        summary = tagged(torch.ones(4, dtype=torch.int64)[::2])
    options = {"route_metadata": summary}
    if fault == "both_mappings":
        options["source_rows"] = tagged(torch.arange(3))
    monkeypatch.setattr(activation, "prepare_activation", lambda *a, **k: pytest.fail("invalid launch"))
    with pytest.raises(ValueError):
        activation.prepare_unchecked(hidden, **options)


def finish_inputs():
    return [
        tagged(torch.ones(3, 64, dtype=torch.bfloat16)),
        tagged(torch.full((3, 2), 127, dtype=torch.uint8)),
        tagged(torch.ones(3, dtype=torch.int32)),
        tagged(torch.ones(1, dtype=torch.int32)),
        tagged(torch.ones(1, dtype=torch.int32)),
        tagged(torch.tensor([2, 1], dtype=torch.int64)),
        tagged(torch.zeros((), dtype=torch.bool)),
    ]


def test_finish_forwards_original_storage_and_status_without_host_reads(activation, monkeypatch):
    inputs = finish_inputs()
    sentinel = object()

    def launch(*received):
        assert all(actual is expected for actual, expected in zip(received, inputs, strict=True))
        return sentinel

    monkeypatch.setitem(
        sys.modules, "vllm_ascend.ops.triton.activation.vq2a8_gs32", SimpleNamespace(finish_full_ffn=launch)
    )
    monkeypatch.setattr(type(inputs[0]), "item", lambda *args: pytest.fail("device scalar read"))
    monkeypatch.setattr(type(inputs[0]), "cpu", lambda *args: pytest.fail("device host transfer"))
    assert activation.finish_full_ffn(*inputs) is sentinel


@pytest.mark.parametrize(
    "fault",
    [
        "output_dtype",
        "zero_rows",
        "scale_dtype",
        "scale_rows",
        "input_dtype",
        "input_rows",
        "route_dtype",
        "native_shape",
        "metadata_dtype",
        "metadata_shape",
        "status_dtype",
        "status_shape",
        "device",
        "noncontiguous",
        "output_alias",
        "status_alias",
    ],
)
def test_finish_rejects_bad_buffers_before_launch(activation, monkeypatch, fault):
    inputs = finish_inputs()
    replacements = {
        "output_dtype": (0, torch.ones(3, 64)),
        "zero_rows": (0, torch.empty(0, 64, dtype=torch.bfloat16)),
        "scale_dtype": (1, torch.ones(3, 2, dtype=torch.int32)),
        "scale_rows": (1, torch.ones(2, 2, dtype=torch.uint8)),
        "input_dtype": (2, torch.ones(3, dtype=torch.int64)),
        "input_rows": (2, torch.ones(2, dtype=torch.int32)),
        "route_dtype": (3, torch.ones(1, dtype=torch.int64)),
        "native_shape": (4, torch.ones(2, dtype=torch.int32)),
        "metadata_dtype": (5, torch.ones(2, dtype=torch.int32)),
        "metadata_shape": (5, torch.ones(3, dtype=torch.int64)),
        "status_dtype": (6, torch.ones((), dtype=torch.uint8)),
        "status_shape": (6, torch.ones(1, dtype=torch.bool)),
        "noncontiguous": (1, torch.ones(3, 4, dtype=torch.uint8)[:, ::2]),
    }
    if fault in replacements:
        index, replacement = replacements[fault]
        inputs[index] = tagged(replacement)
    if fault == "device":
        inputs[4] = tagged(torch.ones(1, dtype=torch.int32), index=1)
    if fault == "output_alias":
        inputs[1] = tagged(inputs[0].view(torch.uint8).reshape(-1)[:6].view(3, 2))
    if fault == "status_alias":
        inputs[6] = tagged(inputs[1].view(torch.bool).reshape(-1)[0])
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.ops.triton.activation.vq2a8_gs32",
        SimpleNamespace(finish_full_ffn=lambda *a: pytest.fail("invalid launch")),
    )
    with pytest.raises(ValueError):
        activation.finish_full_ffn(*inputs)


def activation_kernel_path():
    return Path(__file__).resolve().parents[3] / "vllm_ascend/ops/triton/activation/vq2a8_gs32.py"


@pytest.mark.parametrize("width", [64, 4097])
@pytest.mark.parametrize(
    "fault",
    [
        "none",
        "live_nan",
        "live_inf",
        "scale255",
        "padding_scale255",
        "input",
        "padding_input",
        "route",
        "native",
        "metadata",
        "negative_live",
        "oversized_live",
        "empty",
    ],
)
def test_finish_kernel_register_semantics_ignore_padding_and_keep_all_validity_sources(width, fault):
    rows, live = 3, 2
    output = torch.full((rows, width), 2.0, dtype=torch.bfloat16)
    output[2].fill_(float("nan"))
    tensors = {
        "output_ptr": output,
        "middle_scales_ptr": torch.full((rows, 2), 127, dtype=torch.uint8),
        "input_flags_ptr": torch.ones(rows, dtype=torch.int32),
        "route_flags_ptr": torch.ones(1, dtype=torch.int32),
        "native_flags_ptr": torch.ones(1, dtype=torch.int32),
        "route_metadata_ptr": torch.tensor([live, 1], dtype=torch.int64),
        "block_flags_ptr": torch.empty(rows * ((width + 4095) // 4096), dtype=torch.int32),
    }
    if fault in ("live_nan", "live_inf"):
        output[0, width - 1] = float("nan" if fault == "live_nan" else "inf")
    if fault in ("scale255", "padding_scale255"):
        tensors["middle_scales_ptr"][2 if fault == "padding_scale255" else 0, 0] = 255
    if fault in ("input", "padding_input"):
        tensors["input_flags_ptr"][2 if fault == "padding_input" else 0] = 0
    if fault in ("route", "native"):
        tensors[f"{fault}_flags_ptr"].zero_()
    if fault == "metadata":
        tensors["route_metadata_ptr"][1] = 0
    if fault in ("negative_live", "oversized_live", "empty"):
        live = {"negative_live": -1, "oversized_live": rows + 1, "empty": 0}[fault]
        tensors["route_metadata_ptr"][0] = live
    blocks = (width + 4095) // 4096
    for row in range(rows):
        for block in range(blocks):
            accesses, _ = evaluate_row_kernel(
                "_finish_full_ffn_kernel",
                tensors,
                {
                    "ROWS": rows,
                    "WIDTH": width,
                    "SCALE_WIDTH": 2,
                    "COLUMN_BLOCK": min(4096, 1 << (width - 1).bit_length()),
                    "SCALE_BLOCK": 2,
                    "COLUMN_BLOCKS": blocks,
                    "NONFINITE_SCALE": 255,
                },
                (row, block),
                source_path=activation_kernel_path(),
            )
            if row >= live or fault in ("route", "native", "metadata", "negative_live", "oversized_live"):
                assert all(
                    count == 0 for operation, name, count in accesses if operation == "load" and name == "output_ptr"
                )
    assert bool((tensors["block_flags_ptr"] == 1).all()) == (fault in ("none", "empty"))
    assert bool(torch.isfinite(output).all())
    if fault in ("route", "native", "metadata", "negative_live", "oversized_live", "empty"):
        assert not output.count_nonzero()
    else:
        assert not output[2].count_nonzero()
        if fault in ("live_nan", "live_inf"):
            assert output[0, width - 1] == 0
        else:
            assert bool((output[:2] == 2).all())


@pytest.mark.parametrize("size", [1, 1024, 1025, 2050])
def test_finish_reduction_fixed_tiles_cover_boundary_and_overwrite_status(size):
    flags = torch.ones(size, dtype=torch.int32)
    validity = torch.zeros((), dtype=torch.bool)
    for bad_index in (None, 0, size - 1, None):
        flags.fill_(1)
        if bad_index is not None:
            flags[bad_index] = 0
        validity.fill_(bad_index is not None)
        evaluate_row_kernel(
            "_reduce_full_ffn_validity_kernel",
            {"block_flags_ptr": flags, "validity_ptr": validity},
            {"FLAGS": size, "BLOCK": 1024},
            (0,),
            source_path=activation_kernel_path(),
        )
        assert bool(validity) == (bad_index is None)
