# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU metadata/ownership checks for asynchronous activation, not NPU execution."""

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.ut.quantization.test_vq2a8_gs32_fused_activation import (
    HostFlags,
    activation_module,
    tagged,
    valid_inputs,
)


@pytest.fixture
def activation(monkeypatch):
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(is_current_stream_capturing=lambda: True, device=lambda device: nullcontext()),
        raising=False,
    )
    return activation_module()


def test_gs32_activation_async_prepare_rejects_cpu(activation):
    with pytest.raises(ValueError, match="NPU"):
        activation.prepare_unchecked(
            torch.zeros(1, 64, dtype=torch.bfloat16),
            torch.arange(64).repeat(2, 1),
            torch.zeros(1, dtype=torch.int64),
        )


def test_gs32_activation_async_prepare_preserves_flags_without_host_read(activation, monkeypatch):
    args = valid_inputs()
    raw_gate = tagged(torch.zeros(3, 128, dtype=torch.bfloat16))
    flags = HostFlags([1, 0, 1])
    data, scales = torch.zeros(3, 64, dtype=torch.uint8), torch.ones(3, 2, dtype=torch.uint8)
    calls = []

    def launch(*received):
        calls.append(received)
        return data, scales, flags

    monkeypatch.setattr(activation, "prepare_activation", launch)
    actual = activation.prepare_unchecked(*args, raw_gate=raw_gate)
    assert all(got is expected for got, expected in zip(actual, (data, scales, flags), strict=True))
    assert all(got is expected for got, expected in zip(calls[0], (*args, raw_gate), strict=True))
    assert flags.transfers == 0


@pytest.mark.parametrize("field", ["dtype", "width", "slot_dtype", "raw_gate"])
def test_gs32_activation_async_invalid_metadata_never_launches(activation, monkeypatch, field):
    args, raw_gate = list(valid_inputs()), None
    if field == "dtype":
        args[0] = tagged(torch.zeros(3, 64, dtype=torch.float32))
    elif field == "width":
        args[0] = tagged(torch.zeros(3, 63, dtype=torch.bfloat16))
    elif field == "slot_dtype":
        args[2] = tagged(torch.zeros(3, dtype=torch.int32))
    else:
        raw_gate = tagged(torch.zeros(3, 127, dtype=torch.bfloat16))

    def forbidden(*args):
        pytest.fail("Invalid metadata reached asynchronous preparation.")

    monkeypatch.setattr(activation, "prepare_activation", forbidden)
    with pytest.raises(ValueError):
        activation.prepare_unchecked(*args, raw_gate=raw_gate)


def test_gs32_activation_async_finish_forwards_device_flags_without_reading(activation, monkeypatch):
    output = tagged(torch.zeros(3, 64, dtype=torch.bfloat16))
    gate = tagged(torch.tensor([1, 0, 1], dtype=torch.int32))
    down = tagged(torch.ones(3, dtype=torch.int32))
    final = HostFlags([0])
    calls = []

    def finish(*args):
        calls.append(args)
        return final

    monkeypatch.setattr(activation, "finish_activation_flags", finish)
    assert activation.finish_flags(output, gate, down) is final
    assert all(got is expected for got, expected in zip(calls[0], (output, gate, down), strict=True))
    assert final.transfers == 0


def test_gs32_activation_async_finish_includes_route_and_native_flags(activation, monkeypatch):
    output = tagged(torch.zeros(3, 64, dtype=torch.bfloat16))
    gate = tagged(torch.ones(3, dtype=torch.int32))
    down = tagged(torch.ones(3, dtype=torch.int32))
    route = tagged(torch.zeros(1, dtype=torch.int32))
    native = tagged(torch.ones(1, dtype=torch.int32))
    final, calls = HostFlags([0]), []

    def finish(*args, **kwargs):
        calls.append((args, kwargs))
        return final

    monkeypatch.setattr(activation, "finish_activation_flags", finish)
    assert activation.finish_flags(output, gate, down, route_flags=route, native_flags=native) is final
    assert calls[0][1] == {"route_flags": route, "native_flags": native}
    assert final.transfers == 0


@pytest.mark.parametrize("kind", ["cpu", "dtype", "empty", "rank", "device"])
def test_gs32_activation_async_finish_rejects_invalid_external_flags(activation, monkeypatch, kind):
    output = tagged(torch.zeros(3, 64, dtype=torch.bfloat16))
    flags = tagged(torch.ones(3, dtype=torch.int32))
    external = tagged(torch.ones(1, dtype=torch.int32))
    if kind == "cpu":
        external = torch.ones(1, dtype=torch.int32)
    elif kind == "dtype":
        external = tagged(torch.ones(1, dtype=torch.int64))
    elif kind == "empty":
        external = tagged(torch.empty(0, dtype=torch.int32))
    elif kind == "rank":
        external = tagged(torch.ones(1, 1, dtype=torch.int32))
    else:
        external = tagged(torch.ones(1, dtype=torch.int32), index=1)

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid external flag metadata reached a kernel.")

    monkeypatch.setattr(activation, "finish_activation_flags", forbidden)
    for name in ("route_flags", "native_flags"):
        with pytest.raises(ValueError):
            activation.finish_flags(output, flags, flags, **{name: external})


@pytest.mark.parametrize("field", ["output_cpu", "output_dtype", "rank", "gate_dtype", "down_rows", "device"])
def test_gs32_activation_async_finish_rejects_invalid_metadata(activation, monkeypatch, field):
    output = tagged(torch.zeros(3, 64, dtype=torch.bfloat16))
    gate = tagged(torch.ones(3, dtype=torch.int32))
    down = tagged(torch.ones(3, dtype=torch.int32))
    if field == "output_cpu":
        output = torch.zeros(3, 64, dtype=torch.bfloat16)
    elif field == "output_dtype":
        output = tagged(torch.zeros(3, 64, dtype=torch.int32))
    elif field == "rank":
        output = tagged(torch.zeros(3, 64, 1, dtype=torch.bfloat16))
    elif field == "gate_dtype":
        gate = tagged(torch.ones(3, dtype=torch.int64))
    elif field == "down_rows":
        down = tagged(torch.ones(4, dtype=torch.int32))
    else:
        down = tagged(torch.ones(3, dtype=torch.int32), index=1)

    def forbidden(*args):
        pytest.fail("Invalid metadata reached final validity kernel.")

    monkeypatch.setattr(activation, "finish_activation_flags", forbidden)
    with pytest.raises(ValueError):
        activation.finish_flags(output, gate, down)


def test_gs32_activation_async_does_not_relax_strict_capture_guard(activation):
    with pytest.raises(RuntimeError, match="capture"):
        activation.prepare(*valid_inputs())
    with pytest.raises(RuntimeError, match="capture"):
        activation.require_finite_fused(tagged(torch.ones(3, 64, dtype=torch.bfloat16)), "output")


def test_gs32_activation_async_empty_returns_empty_flags_without_launch(activation, monkeypatch):
    args = list(valid_inputs())
    args[0] = tagged(torch.empty(0, 64, dtype=torch.bfloat16))
    args[2] = tagged(torch.empty(0, dtype=torch.int64))

    def forbidden(*args):
        pytest.fail("Empty preparation launched a kernel.")

    monkeypatch.setattr(activation, "prepare_activation", forbidden)
    data, scales, flags = activation.prepare_unchecked(*args)
    assert data.shape == (0, 64) and scales.shape == (0, 2) and flags.shape == (0,)
    assert data.dtype == scales.dtype == torch.uint8 and flags.dtype == torch.int32


@pytest.mark.parametrize("limit", [0.0, 10.0])
def test_fused_swiglu_async_preserves_device_flags_and_raw_buffer(activation, monkeypatch, limit):
    raw = tagged(torch.ones(3, 128, dtype=torch.bfloat16))
    _, perm, slots = valid_inputs()
    flags, calls = HostFlags([1, 0, 1]), []
    result = torch.zeros(3, 64, dtype=torch.uint8), torch.ones(3, 2, dtype=torch.uint8), flags

    def launch(*args, **kwargs):
        calls.append((args, kwargs))
        return result

    monkeypatch.setattr(activation, "prepare_activation", launch)
    actual = activation.prepare_unchecked(raw, perm, slots, swiglu_limit=limit)
    assert all(a is b for a, b in zip(actual, result, strict=True))
    assert calls[0][0][0] is raw and calls[0][1] == {"swiglu_limit": limit}
    assert flags.transfers == 0


def test_fused_swiglu_async_empty_and_strict_capture_boundary(activation, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Empty or strict captured SwiGLU reached launch.")

    monkeypatch.setattr(activation, "prepare_activation", forbidden)
    raw = tagged(torch.empty(0, 128, dtype=torch.bfloat16))
    slots = tagged(torch.empty(0, dtype=torch.int64))
    data, scales, flags = activation.prepare_unchecked(raw, None, slots, swiglu_limit=10.0)
    assert data.shape == (0, 64) and scales.shape == (0, 2) and flags.shape == (0,)
    with pytest.raises(RuntimeError, match="capture"):
        activation.prepare(raw, None, slots, swiglu_limit=10.0)


@pytest.mark.parametrize("swiglu_limit", [None, 0.0, 10.0])
def test_gathered_prepare_forwards_original_buffers_without_host_reads(activation, monkeypatch, swiglu_limit):
    hidden, perm, slots = valid_inputs()
    raw = tagged(torch.ones(3, 128, dtype=torch.bfloat16))
    source = tagged(torch.tensor([2, -1, 0], dtype=torch.int64))
    route = tagged(torch.ones(1, dtype=torch.int32))
    flags, calls = HostFlags([1, 1, 1]), []
    result = torch.zeros(3, 64, dtype=torch.uint8), torch.full((3, 2), 127, dtype=torch.uint8), flags

    def launch(*args, **kwargs):
        calls.append((args, kwargs))
        return result

    def forbidden(*args, **kwargs):
        pytest.fail("Gathered preparation performed a host read or materialized a Torch gather.")

    monkeypatch.setattr(activation, "prepare_activation", launch)
    for name in ("cpu", "item", "tolist", "index_select", "gather"):
        monkeypatch.setattr(torch.Tensor, name, forbidden)
    options = {"source_rows": source, "route_flags": route}
    if swiglu_limit is None:
        options["raw_gate"] = raw
    else:
        hidden = raw
        options["swiglu_limit"] = swiglu_limit
    actual = activation.prepare_unchecked(hidden, perm, slots, **options)
    assert all(a is b for a, b in zip(actual, result, strict=True))
    assert calls[0][0][0] is hidden
    assert calls[0][0][3] is (raw if swiglu_limit is None else None)
    assert calls[0][1]["source_rows"] is source and calls[0][1]["route_flags"] is route
    assert flags.transfers == 0


@pytest.mark.parametrize("field", ["source_rows", "route_flags"])
@pytest.mark.parametrize("fault", ["cpu", "dtype", "rows", "rank", "device", "contiguous"])
def test_gathered_prepare_rejects_metadata_before_launch(activation, monkeypatch, field, fault):
    rows, dtype = (3, torch.int64) if field == "source_rows" else (1, torch.int32)
    value = torch.ones(rows, dtype=dtype)
    if fault == "dtype":
        value = value.float()
    elif fault == "rows":
        value = torch.ones(rows + 1, dtype=dtype)
    elif fault == "rank":
        value = value[:, None]
    elif fault == "contiguous":
        # Even a one-element vector is contiguous; use a wrong-length strided
        # flag vector to exercise the shared contiguous metadata requirement.
        value = torch.ones(2 * max(rows, 2), dtype=dtype)[::2]
    if fault != "cpu":
        value = tagged(value, index=int(fault == "device"))

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid gathered metadata reached the activation launch.")

    monkeypatch.setattr(activation, "prepare_activation", forbidden)
    with pytest.raises(ValueError):
        activation.prepare_unchecked(*valid_inputs(), **{field: value})


def evaluate_gathered_prepare(hidden, perm, slots, *, raw_gate=None, source_rows=None, route_flags=None, limit=None):
    """Run production indexing/register expressions on CPU, not an NPU emulator."""
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/triton/activation/vq2a8_gs32.py"
    module = ast.parse(source.read_text())

    class CpuExpressions(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            self.generic_visit(node)
            node.decorator_list = []
            for argument in node.args.args:
                argument.annotation = None
            return node

        def visit_Call(self, node):
            self.generic_visit(node)
            if any(keyword.arg == "bitcast" for keyword in node.keywords):
                node.func.attr = "view"
            node.keywords = [
                keyword for keyword in node.keywords if keyword.arg not in ("bitcast", "fp_downcast_rounding")
            ]
            return node

    reads = []

    class Pointer:
        def __init__(self, storage, name, offsets=0):
            self.storage, self.name, self.offsets = storage.flatten(), name, torch.as_tensor(offsets)
            self.dtype = SimpleNamespace(element_ty=storage.dtype)

        def __add__(self, offsets):
            return Pointer(self.storage, self.name, self.offsets + offsets)

    def load(pointer, mask=True, other=0):
        mask, offsets = torch.broadcast_tensors(torch.as_tensor(mask), pointer.offsets)
        reads.append(pointer.name)
        selected = offsets[mask]
        assert ((selected >= 0) & (selected < pointer.storage.numel())).all(), "Unmasked out-of-range input read."
        result = torch.full(offsets.shape, other, dtype=pointer.storage.dtype)
        result[mask] = pointer.storage[selected]
        return result

    def store(pointer, value, mask=True):
        values = torch.as_tensor(value)
        # Torch accepts integer -> FP8 casts that Ascend Triton rejects.
        # Require explicitly typed FP8 values so the CPU oracle cannot hide
        # an unsupported implicit conversion in the production store.
        if pointer.storage.dtype == torch.float8_e4m3fn:
            assert values.dtype == pointer.storage.dtype, "FP8 stores require explicitly typed FP8 values."
        mask, offsets, values = torch.broadcast_tensors(torch.as_tensor(mask), pointer.offsets, values)
        pointer.storage[offsets[mask]] = values[mask].to(pointer.storage.dtype)

    def where(condition, left, right):
        return torch.where(torch.as_tensor(condition), torch.as_tensor(left), torch.as_tensor(right))

    namespace = {
        "tl": SimpleNamespace(
            arange=torch.arange,
            full=lambda shape, value, dtype: torch.full(shape, value, dtype=dtype),
            load=load,
            store=store,
            where=where,
            abs=torch.abs,
            sigmoid=torch.sigmoid,
            min=lambda values, axis: values.amin(dim=axis),
            max=lambda values, axis: values.amax(dim=axis),
            minimum=lambda a, b: torch.minimum(torch.as_tensor(a), torch.as_tensor(b)),
            maximum=lambda a, b: torch.maximum(torch.as_tensor(a), torch.as_tensor(b)),
            reshape=torch.reshape,
            int64=torch.int64,
            int32=torch.int32,
            uint8=torch.uint8,
            float32=torch.float32,
            bfloat16=torch.bfloat16,
        )
    }
    functions = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name in ("_quantize_groups", "_prepare_activation_kernel")
    ]
    expressions = ast.fix_missing_locations(CpuExpressions().visit(ast.Module(body=functions, type_ignores=[])))
    exec(compile(expressions, str(source), "exec"), namespace)
    rows, width = hidden.shape
    if limit is not None:
        width //= 2
    data = torch.empty((rows, width), dtype=torch.float8_e4m3fn)
    scales, flags = torch.empty((rows, width // 32), dtype=torch.uint8), torch.empty(rows, dtype=torch.int32)
    buffers = (
        hidden,
        hidden if perm is None else perm,
        slots,
        hidden if raw_gate is None else raw_gate,
        slots if source_rows is None else source_rows,
        slots if route_flags is None else route_flags,
        data,
        scales,
        flags,
    )
    pointers = [
        Pointer(value, name)
        for value, name in zip(
            buffers,
            ("hidden", "perm", "slots", "raw_gate", "source_rows", "route_flags", "data", "scales", "flags"),
            strict=True,
        )
    ]
    for row in range(rows):
        namespace["tl"].program_id = lambda axis, row=row: torch.tensor(row, dtype=torch.int32)
        namespace["_prepare_activation_kernel"](
            *pointers,
            K=width,
            EXPERTS=0 if perm is None else perm.shape[0],
            INPUT_ROWS=rows,
            BLOCK_K=1 << (width - 1).bit_length(),
            HAS_RAW_GATE=raw_gate is not None,
            HAS_PERM=perm is not None,
            HAS_SOURCE_ROWS=source_rows is not None,
            HAS_ROUTE_FLAGS=route_flags is not None,
            FUSE_SWIGLU=limit is not None,
            SWIGLU_LIMIT=0.0 if limit is None else limit,
            GROUP=32,
            FLOAT_MAX=torch.finfo(torch.float32).max,
        )
    return data.view(torch.uint8), scales, flags, reads


@pytest.mark.parametrize("width", [32, 96, 160])
@pytest.mark.parametrize("permuted", [False, True])
@pytest.mark.parametrize("limit", [None, 0.0, 10.0])
def test_gathered_registers_match_explicit_gather_bytes(width, permuted, limit):
    generator = torch.Generator().manual_seed(3208)
    hidden = (torch.randn(5, width, generator=generator) * 20).bfloat16()
    if limit is not None:
        hidden = torch.cat((hidden, hidden.roll(17, 1) * 2), dim=1)
    perm = torch.stack((torch.arange(width), torch.roll(torch.arange(width), 33))) if permuted else None
    slots = torch.tensor([1, 0, 1, 1, 0], dtype=torch.int64)
    source = torch.tensor([4, 0, 4, 2, 1], dtype=torch.int64)
    actual = evaluate_gathered_prepare(hidden, perm, slots, source_rows=source, limit=limit)
    expected = evaluate_gathered_prepare(hidden.index_select(0, source), perm, slots, limit=limit)
    assert all(torch.equal(a, b) for a, b in zip(actual[:3], expected[:3], strict=True))
    assert actual[2].tolist() == [1] * 5


@pytest.mark.parametrize(
    "source,route,expected", [(-1, 1, 1), (-2, 1, 0), (3, 1, 0), (2**40, 1, 0), (0, 0, 0), (-1, 2, 0)]
)
@pytest.mark.parametrize("limit", [None, 0.0, 10.0])
@pytest.mark.parametrize("width", [96, 4096])
def test_inactive_gathered_registers_skip_input_and_quantization(source, route, expected, limit, width):
    hidden = torch.full((3, width if limit is None else 2 * width), float("nan"), dtype=torch.bfloat16)
    # Poison every input that must not be inspected in the inactive row branch.
    perm, slots = torch.full((1, width), 9999, dtype=torch.int32), torch.full((3,), 9999, dtype=torch.int64)
    result = evaluate_gathered_prepare(
        hidden,
        perm,
        slots,
        source_rows=torch.full((3,), source, dtype=torch.int64),
        route_flags=torch.tensor([route], dtype=torch.int32),
        limit=limit,
    )
    assert result[0].count_nonzero() == 0 and (result[1] == 127).all()
    assert result[2].tolist() == [expected] * 3
    assert set(result[3]) == {"source_rows", "route_flags"}


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("fault", [float("nan"), float("inf")])
def test_gathered_raw_gate_invalidity_survives_mapping_and_clipping(fused, fault):
    width = 96
    raw = torch.ones(3, 2 * width, dtype=torch.bfloat16)
    raw[2, width + 95] = fault
    source, slots = torch.tensor([2, 1, 0]), torch.zeros(3, dtype=torch.int64)
    permutation = torch.roll(torch.arange(width), 33)[None]
    options = {"limit": 10.0} if fused else {"raw_gate": raw}
    hidden = raw if fused else torch.ones(3, width, dtype=torch.bfloat16)
    result = evaluate_gathered_prepare(hidden, permutation, slots, source_rows=source, **options)
    assert result[2].tolist() == [0, 1, 1]
    assert torch.isfinite(result[0].view(torch.float8_e4m3fn).float()).all()
    expected_options = {"limit": 10.0} if fused else {"raw_gate": raw.index_select(0, source)}
    expected = evaluate_gathered_prepare(hidden.index_select(0, source), permutation, slots, **expected_options)
    assert all(torch.equal(a, b) for a, b in zip(result[:3], expected[:3], strict=True))
