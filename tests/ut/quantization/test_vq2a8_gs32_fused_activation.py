# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused activation wrapper contracts; CPU tests never execute an NPU kernel."""

import ast
import importlib
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def activation_module():
    return importlib.import_module("vllm_ascend.ops.fused_moe.vq2a8_gs32_activation")


@dataclass(frozen=True)
class FakeDevice:
    type: str = "npu"
    index: int = 0


class NpuTaggedCpuTensor(torch.Tensor):
    """Metadata double only: storage and arithmetic still live on the CPU."""

    @property
    def device(self):
        return self.fake_device


def tagged(tensor, *, index=0):
    value = tensor.as_subclass(NpuTaggedCpuTensor)
    value.fake_device = FakeDevice(index=index)
    return value


def valid_inputs():
    return (
        tagged(torch.zeros(3, 64, dtype=torch.bfloat16)),
        tagged(torch.arange(64, dtype=torch.int64).repeat(2, 1)),
        tagged(torch.tensor([1, 0, 1], dtype=torch.int64)),
    )


@pytest.mark.parametrize(
    "kernel_name,scalar,blocks",
    [
        ("_prepare_activation_kernel", "EXPERTS", ("K", "BLOCK_K", "GROUP")),
        ("_prepare_activation_kernel", "INPUT_ROWS", ("K", "BLOCK_K", "GROUP")),
        ("_finite_flags_kernel", "ELEMENTS", ("BLOCK",)),
    ],
)
def test_activation_runtime_sizes_do_not_create_exact_shape_specializations(kernel_name, scalar, blocks):
    """Compilation contract only; actual Ascend compilation is tested separately."""
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/triton/activation/vq2a8_gs32.py"
    module = ast.parse(source.read_text())
    kernel = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == kernel_name)
    arguments = {arg.arg: arg.annotation for arg in kernel.args.args}
    assert arguments[scalar] is None
    assert all(ast.unparse(arguments[block]) == "tl.constexpr" for block in blocks)
    decorator = next(node for node in kernel.decorator_list if ast.unparse(node.func) == "triton.jit")
    excluded = next(keyword.value for keyword in decorator.keywords if keyword.arg == "do_not_specialize")
    assert scalar in ast.literal_eval(excluded)


@pytest.mark.parametrize("elements", [1, 1023, 1024, 1025, 2047, 2048, 2049])
@pytest.mark.parametrize("fault", [None, float("nan"), float("inf")])
def test_finite_runtime_size_preserves_tail_mask_and_last_live_element(elements, fault):
    """Evaluate production bounds on CPU, without emulating Triton compilation."""
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/triton/activation/vq2a8_gs32.py"
    module = ast.parse(source.read_text())
    kernel = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "_finite_flags_kernel"
    )
    block = 1024
    tiles = (elements + block - 1) // block
    storage = torch.ones(tiles * block)
    storage[elements:] = float("nan")
    if fault is not None:
        storage[elements - 1] = fault
    flags = torch.empty(tiles, dtype=torch.int32)
    namespace = {
        "tensor_ptr": 0,
        "flags_ptr": 0,
        "ELEMENTS": elements,
        "BLOCK": block,
        "FLOAT_MAX": torch.finfo(torch.float32).max,
    }
    for tile in range(tiles):
        namespace["tl"] = SimpleNamespace(
            program_id=lambda axis, tile=tile: torch.tensor([tile], dtype=torch.int32),
            arange=lambda first, last: torch.arange(first, last),
            load=lambda offsets, mask, other: torch.where(mask, storage[offsets], other),
            abs=torch.abs,
            min=lambda values, axis: values.amin(dim=axis),
            store=lambda offset, value: flags.__setitem__(offset, value),
            float32=torch.float32,
            int32=torch.int32,
            int64=torch.int64,
        )
        exec(compile(ast.Module(body=kernel.body, type_ignores=[]), str(source), "exec"), namespace)
    assert flags.tolist() == [1] * (tiles - 1) + [int(fault is None)]


@pytest.mark.parametrize("width", [2048, 4096])
@pytest.mark.parametrize("row", [2**19, 2**20 + 1])
def test_activation_pointer_strides_do_not_wrap_at_int32(width, row):
    """Evaluate actual address expressions without allocating multi-GiB banks."""
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/triton/activation/vq2a8_gs32.py"
    module = ast.parse(source.read_text())
    kernel = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "_prepare_activation_kernel"
    )
    row_assignment = next(
        node
        for node in kernel.body
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "row" for t in node.targets)
    )
    columns = torch.arange(width, dtype=torch.int32)
    pointers = {
        "hidden_ptr",
        "raw_gate_ptr",
        "output_ptr",
        "output_scale_ptr",
        "flags_ptr",
        "slot_ptr",
        "source_row_ptr",
    }
    namespace = {
        "tl": SimpleNamespace(program_id=lambda axis: torch.tensor([row], dtype=torch.int32), int64=torch.int64),
        "K": width,
        "GROUP": 32,
        "columns": columns,
        "safe_index": torch.roll(columns, 33),
        "groups": torch.arange(width // 32, dtype=torch.int32),
        **dict.fromkeys(pointers, 0),
    }
    exec(compile(ast.Module(body=[row_assignment], type_ignores=[]), str(source), "exec"), namespace)
    namespace["source_row"] = namespace["row"]
    checked = set()
    for call in ast.walk(kernel):
        if (
            not isinstance(call, ast.Call)
            or not isinstance(call.func, ast.Attribute)
            or call.func.attr not in ("load", "store")
        ):
            continue
        expression = call.args[0]
        names = {node.id for node in ast.walk(expression) if isinstance(node, ast.Name)}
        if not names & {"row", "source_row"} or not names & pointers:
            continue
        pointer = (names & pointers).pop()
        actual = eval(compile(ast.Expression(expression), str(source), "eval"), namespace)
        assert actual.dtype == torch.int64
        stride = {
            "hidden_ptr": width,
            "raw_gate_ptr": 2 * width,
            "output_ptr": width,
            "output_scale_ptr": width // 32,
        }.get(pointer, 1)
        assert int(actual.min()) >= row * stride
        assert int(actual.max()) < (row + 1) * stride
        checked.add(pointer)
    assert checked == pointers


@pytest.mark.parametrize("fault", [False, True])
def test_finite_pointer_offsets_above_int32_preserve_tail_mask(fault):
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/triton/activation/vq2a8_gs32.py"
    module = ast.parse(source.read_text())
    kernel = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "_finite_flags_kernel"
    )
    block, tile = 1024, 2**21 + 1
    elements = tile * block + 17
    writes = []

    def load(offsets, mask, other):
        expected = tile * block + torch.arange(block, dtype=torch.int64)
        assert torch.equal(offsets, expected) and offsets.dtype == torch.int64
        assert torch.equal(mask, torch.arange(block) < 17)
        values = torch.ones(block)
        values[17:] = float("nan")
        if fault:
            values[16] = float("nan")
        return torch.where(mask, values, other)

    namespace = {
        "tl": SimpleNamespace(
            program_id=lambda axis: torch.tensor([tile], dtype=torch.int32),
            arange=lambda start, end: torch.arange(start, end, dtype=torch.int32),
            load=load,
            abs=torch.abs,
            min=lambda values, axis: values.amin(dim=axis),
            store=lambda offset, value: writes.append((int(offset), int(value))),
            float32=torch.float32,
            int32=torch.int32,
            int64=torch.int64,
        ),
        "tensor_ptr": 0,
        "flags_ptr": 0,
        "ELEMENTS": elements,
        "BLOCK": block,
        "FLOAT_MAX": torch.finfo(torch.float32).max,
    }
    exec(compile(ast.Module(body=kernel.body, type_ignores=[]), str(source), "exec"), namespace)
    assert writes == [(tile, int(not fault))]


class HostFlags:
    """Count the wrapper's host transfers without claiming device execution."""

    def __init__(self, values):
        self.values = values
        self.transfers = 0

    def cpu(self):
        self.transfers += 1
        return self

    def tolist(self):
        return self.values


@pytest.fixture
def wrapper(monkeypatch):
    activation = activation_module()
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(is_current_stream_capturing=lambda: False, device=lambda device: nullcontext()),
        raising=False,
    )
    return activation


def test_gs32_fused_activation_rejects_cpu_execution():
    activation = activation_module()
    with pytest.raises(ValueError, match="NPU|npu"):
        activation.prepare(
            torch.zeros(1, 64, dtype=torch.bfloat16),
            torch.arange(64, dtype=torch.int64).repeat(2, 1),
            torch.zeros(1, dtype=torch.int64),
        )


@pytest.mark.parametrize(
    "index,replacement",
    [
        (0, lambda: tagged(torch.zeros(3, 64, dtype=torch.float32))),
        (0, lambda: tagged(torch.zeros(3, 63, dtype=torch.bfloat16))),
        (0, lambda: tagged(torch.zeros(3, 64, 1, dtype=torch.bfloat16))),
        (0, lambda: tagged(torch.zeros(64, 3, dtype=torch.bfloat16).T)),
        (1, lambda: tagged(torch.ones(2, 64, dtype=torch.bfloat16))),
        (1, lambda: tagged(torch.ones(2, 32, dtype=torch.float32))),
        (1, lambda: tagged(torch.ones(2, 64, dtype=torch.float32), index=1)),
        (1, lambda: tagged(torch.arange(32, dtype=torch.int64).repeat(2, 1))),
        (2, lambda: tagged(torch.zeros(3, dtype=torch.int32))),
        (2, lambda: tagged(torch.zeros(4, dtype=torch.int64))),
        (2, lambda: tagged(torch.zeros(3, 1, dtype=torch.int64))),
    ],
)
def test_gs32_fused_activation_invalid_metadata_never_launches(wrapper, monkeypatch, index, replacement):
    args = list(valid_inputs())
    args[index] = replacement()

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid metadata reached the fusion kernel.")

    monkeypatch.setattr(wrapper, "prepare_activation", forbidden)
    with pytest.raises(ValueError):
        wrapper.prepare(*args)


@pytest.mark.parametrize(
    "raw_gate",
    [
        lambda: tagged(torch.zeros(3, 127, dtype=torch.bfloat16)),
        lambda: tagged(torch.zeros(2, 128, dtype=torch.bfloat16)),
        lambda: tagged(torch.zeros(3, 128, dtype=torch.float32)),
        lambda: tagged(torch.zeros(3, 128, dtype=torch.bfloat16), index=1),
    ],
)
def test_gs32_fused_activation_invalid_raw_gate_never_launches(wrapper, monkeypatch, raw_gate):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid raw gate metadata reached the fusion kernel.")

    monkeypatch.setattr(wrapper, "prepare_activation", forbidden)
    with pytest.raises(ValueError):
        wrapper.prepare(*valid_inputs(), raw_gate=raw_gate())


def test_gs32_fused_activation_forwards_all_buffers_and_reads_flags_once(wrapper, monkeypatch):
    args = valid_inputs()
    raw_gate = tagged(torch.zeros(3, 128, dtype=torch.bfloat16))
    data = torch.arange(3 * 64, dtype=torch.uint8).reshape(3, 64)
    scale = torch.full((3, 2), 127, dtype=torch.uint8)
    flags = HostFlags([1, 1, 1])
    calls = []

    def launch(*received, **kwargs):
        calls.append((received, kwargs))
        return data, scale, flags

    monkeypatch.setattr(wrapper, "prepare_activation", launch)
    actual_data, actual_scale = wrapper.prepare(*args, raw_gate=raw_gate)
    assert actual_data is data and actual_scale is scale
    assert flags.transfers == 1 and len(calls) == 1
    received, kwargs = calls[0]
    assert all(got is expected for got, expected in zip(received[:3], args, strict=True))
    assert (received[3] if len(received) == 4 else kwargs["raw_gate"]) is raw_gate


@pytest.mark.parametrize("flag_values", [[0, 1, 1], [1, 0, 1], [1, 1, 0]])
def test_gs32_fused_activation_bad_flag_fails_before_native_consumer(wrapper, monkeypatch, flag_values):
    flags = HostFlags(flag_values)
    monkeypatch.setattr(
        wrapper,
        "prepare_activation",
        lambda *args, **kwargs: (torch.zeros(3, 64, dtype=torch.uint8), torch.ones(3, 2, dtype=torch.uint8), flags),
    )
    consumed = []

    def pipeline():
        result = wrapper.prepare(*valid_inputs())
        consumed.append(result)

    with pytest.raises(ValueError, match="finite|invalid"):
        pipeline()
    assert consumed == [] and flags.transfers == 1


def test_gs32_fused_activation_failure_does_not_poison_next_call(wrapper, monkeypatch):
    flags = [HostFlags([1, 0, 1]), HostFlags([1, 1, 1])]

    def launch(*args, **kwargs):
        return torch.zeros(3, 64, dtype=torch.uint8), torch.full((3, 2), 127, dtype=torch.uint8), flags.pop(0)

    monkeypatch.setattr(wrapper, "prepare_activation", launch)
    with pytest.raises(ValueError):
        wrapper.prepare(*valid_inputs())
    data, scale = wrapper.prepare(*valid_inputs())
    assert data.shape == (3, 64) and scale.shape == (3, 2)
    assert flags == []


@pytest.mark.parametrize("flag_values", [[1, 1], [1, 0]])
def test_gs32_fused_activation_output_finite_flags_preserve_name_and_one_transfer(wrapper, monkeypatch, flag_values):
    flags = HostFlags(flag_values)
    output = tagged(torch.zeros(3, 128, dtype=torch.bfloat16))
    monkeypatch.setattr(wrapper, "finite_flags", lambda tensor: flags)
    if all(flag_values):
        wrapper.require_finite_fused(output, "Down native output")
    else:
        with pytest.raises(ValueError, match="Down native output"):
            wrapper.require_finite_fused(output, "Down native output")
    assert flags.transfers == 1


def test_gs32_fused_activation_capture_rejected_before_launch(wrapper, monkeypatch):
    torch.npu.is_current_stream_capturing = lambda: True

    def forbidden(*args, **kwargs):
        pytest.fail("Graph capture reached an eager-only fusion kernel.")

    monkeypatch.setattr(wrapper, "prepare_activation", forbidden)
    with pytest.raises(RuntimeError, match="capture|graph"):
        wrapper.prepare(*valid_inputs())


def test_gs32_fused_activation_empty_dispatch_does_not_launch(wrapper, monkeypatch):
    args = list(valid_inputs())
    args[0] = tagged(torch.empty(0, 64, dtype=torch.bfloat16))
    args[2] = tagged(torch.empty(0, dtype=torch.int64))

    def forbidden(*args, **kwargs):
        pytest.fail("Empty activation preparation launched a kernel.")

    monkeypatch.setattr(wrapper, "prepare_activation", forbidden)
    data, scale = wrapper.prepare(*args)
    assert data.shape == (0, 64) and scale.shape == (0, 2)
    assert data.dtype == scale.dtype == torch.uint8


@pytest.mark.parametrize("limit", [0.0, 10.0])
def test_gs32_fused_activation_unchecked_swiglu_preserves_original_arithmetic(limit):
    from vllm_ascend.quantization.methods.w2a8.vq2a8_gs32_staged import staged_swiglu, staged_swiglu_unchecked

    generator = torch.Generator().manual_seed(314)
    raw = (torch.randn(6, 128, generator=generator) * 20).bfloat16()
    gate, up = raw.float().chunk(2, dim=-1)
    if limit > 0:
        gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
    expected = (torch.nn.functional.silu(gate) * up).bfloat16().contiguous()
    assert torch.equal(staged_swiglu_unchecked(raw, limit).view(torch.int16), expected.view(torch.int16))
    assert torch.equal(staged_swiglu(raw, limit).view(torch.int16), expected.view(torch.int16))


def test_gs32_fused_activation_unchecked_requires_raw_gate_check_after_clamp():
    from vllm_ascend.quantization.methods.w2a8.vq2a8_gs32_staged import staged_swiglu, staged_swiglu_unchecked

    raw = torch.ones(1, 128, dtype=torch.bfloat16)
    raw[0, 0] = float("inf")
    assert torch.isfinite(staged_swiglu_unchecked(raw, 10)).all()
    with pytest.raises(ValueError, match="GateUp output"):
        staged_swiglu(raw, 10)


@pytest.mark.parametrize("limit", [-1.0, float("nan"), float("inf")])
def test_gs32_fused_activation_unchecked_still_validates_swiglu_metadata(limit):
    from vllm_ascend.quantization.methods.w2a8.vq2a8_gs32_staged import staged_swiglu_unchecked

    with pytest.raises(ValueError):
        staged_swiglu_unchecked(torch.ones(1, 128, dtype=torch.bfloat16), limit)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, None])
def test_source_v3_permutation_or_natural_mxfp4_columns(wrapper, monkeypatch, dtype):
    args = list(valid_inputs())
    args[1] = None if dtype is None else tagged(torch.arange(64, dtype=dtype).repeat(2, 1))
    result = (torch.zeros(3, 64, dtype=torch.uint8), torch.ones(3, 2, dtype=torch.uint8), HostFlags([1, 1, 1]))
    monkeypatch.setattr(wrapper, "prepare_activation", lambda *values: result)
    assert wrapper.prepare(*args)[0] is result[0]


@pytest.mark.parametrize("limit", [0, 0.0, 10.0])
@pytest.mark.parametrize("permuted", [False, True])
def test_fused_swiglu_prepare_forwards_raw_gateup_without_middle(wrapper, monkeypatch, limit, permuted):
    hidden, perm, slots = valid_inputs()
    raw = tagged(torch.ones(3, 128, dtype=torch.bfloat16))
    calls, flags = [], HostFlags([1, 1, 1])
    result = torch.zeros(3, 64, dtype=torch.uint8), torch.ones(3, 2, dtype=torch.uint8), flags

    def launch(*args, **kwargs):
        calls.append((args, kwargs))
        return result

    monkeypatch.setattr(wrapper, "prepare_activation", launch)
    actual = wrapper.prepare(raw, perm if permuted else None, slots, swiglu_limit=limit)
    assert all(a is b for a, b in zip(actual, result[:2], strict=True))
    assert calls[0][0][0] is raw and calls[0][0][3] is None
    assert calls[0][1] == {"swiglu_limit": limit} and flags.transfers == 1


@pytest.mark.parametrize("limit", [-1, float("nan"), float("inf"), True, "10", torch.tensor(10.0)])
def test_fused_swiglu_limit_is_validated_without_tensor_conversion(wrapper, monkeypatch, limit):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid SwiGLU limit reached kernel launch.")

    monkeypatch.setattr(wrapper, "prepare_activation", forbidden)
    with pytest.raises(ValueError, match="limit"):
        wrapper.prepare(tagged(torch.ones(3, 128, dtype=torch.bfloat16)), *valid_inputs()[1:], swiglu_limit=limit)


@pytest.mark.parametrize("fault", ["raw_gate", "half_width", "perm_width"])
def test_fused_swiglu_rejects_ambiguous_source_and_wrong_half_geometry(wrapper, monkeypatch, fault):
    raw = tagged(torch.ones(3, 128, dtype=torch.bfloat16))
    _, perm, slots = valid_inputs()
    options = {"swiglu_limit": 10.0}
    if fault == "raw_gate":
        options["raw_gate"] = raw
    elif fault == "half_width":
        raw = tagged(torch.ones(3, 96, dtype=torch.bfloat16))
    else:
        perm = tagged(torch.arange(128).repeat(2, 1))

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid fused geometry reached kernel launch.")

    monkeypatch.setattr(wrapper, "prepare_activation", forbidden)
    with pytest.raises(ValueError):
        wrapper.prepare(raw, perm, slots, **options)


def test_direct_fused_activation_rejects_raw_gate_conflict_before_lazy_import(wrapper):
    raw = tagged(torch.ones(3, 128, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="cannot be combined"):
        wrapper.prepare_activation(raw, None, valid_inputs()[2], raw_gate=raw, swiglu_limit=0.0)


@pytest.mark.parametrize("limit", [0.0, 10.0])
def test_fused_swiglu_empty_dispatch_returns_half_width(wrapper, monkeypatch, limit):
    def forbidden(*args, **kwargs):
        pytest.fail("Empty fused SwiGLU launched a kernel.")

    monkeypatch.setattr(wrapper, "prepare_activation", forbidden)
    data, scale = wrapper.prepare(
        tagged(torch.empty(0, 128, dtype=torch.bfloat16)),
        None,
        tagged(torch.empty(0, dtype=torch.int64)),
        swiglu_limit=limit,
    )
    assert data.shape == (0, 64) and scale.shape == (0, 2)


def evaluate_fused_swiglu_registers(raw, permutation, limit):
    """Execute production register expressions with CPU Torch, not Triton.

    This exercises indexing, clipping, FP32/product/BF16 order and flags;
    it deliberately cannot validate Ascend sigmoid code generation.
    """
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/triton/activation/vq2a8_gs32.py"
    module = ast.parse(source.read_text())
    kernel = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "_prepare_activation_kernel"
    )
    kernel_body = next(
        node.body
        for node in kernel.body
        if isinstance(node, ast.If) and ast.unparse(node.test) == "source_valid & route_valid"
    )
    start = next(
        i for i, node in enumerate(kernel_body) if isinstance(node, ast.If) and ast.unparse(node.test) == "FUSE_SWIGLU"
    )
    stop = next(
        i
        for i, node in enumerate(kernel_body)
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Tuple) for t in node.targets)
    )

    class CpuRounding(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            node.keywords = [kw for kw in node.keywords if kw.arg != "fp_downcast_rounding"]
            return node

    body = CpuRounding().visit(ast.Module(body=kernel_body[start:stop], type_ignores=[]))
    width, row = raw.shape[1] // 2, 1
    namespace = {
        "tl": SimpleNamespace(
            load=lambda address, mask, other: torch.where(mask, raw.flatten()[address], other),
            abs=torch.abs,
            where=torch.where,
            sigmoid=torch.sigmoid,
            minimum=lambda a, b: torch.minimum(a, torch.as_tensor(b)),
            maximum=lambda a, b: torch.maximum(a, torch.as_tensor(b)),
            float32=torch.float32,
            bfloat16=torch.bfloat16,
        ),
        "raw_gate_ptr": 0,
        "row": row,
        "source_row": row,
        "K": width,
        "safe_index": permutation,
        "read_mask": torch.ones(width, dtype=torch.bool),
        "active": torch.ones(width, dtype=torch.bool),
        "index_valid": torch.ones(width, dtype=torch.bool),
        "slot_valid": torch.tensor(True),
        "FUSE_SWIGLU": True,
        "HAS_RAW_GATE": False,
        "SWIGLU_LIMIT": limit,
        "FLOAT_MAX": torch.finfo(torch.float32).max,
    }
    exec(compile(body, str(source), "exec"), namespace)
    return namespace["safe_rounded"], namespace["valid"]


@pytest.mark.parametrize("limit", [0.0, 10.0])
@pytest.mark.parametrize("permuted", [False, True])
def test_fused_swiglu_register_order_matches_staged_cpu_contract(limit, permuted):
    from vllm_ascend.quantization.methods.w2a8.vq2a8_gs32_staged import quantize_mxfp8_k32, staged_swiglu

    raw = torch.ones(2, 128, dtype=torch.bfloat16)
    raw[0].fill_(99)  # Catch wrong row stride when input width is 2K.
    raw[1, :64] = torch.tensor([-100, -20, -10, -1, 0, 1, 10, 20]).repeat(8)
    raw[1, 64:] = torch.tensor([1, -30, 0.25, 30, -0.0, 3, -0.5, 2]).repeat(8)
    permutation = torch.roll(torch.arange(64), 33) if permuted else torch.arange(64)
    actual, valid = evaluate_fused_swiglu_registers(raw, permutation, limit)
    expected = staged_swiglu(raw, limit)[1, permutation].float()
    assert valid.all()
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
    assert all(
        torch.equal(a, b)
        for a, b in zip(quantize_mxfp8_k32(actual[None]), quantize_mxfp8_k32(expected[None]), strict=True)
    )


@pytest.mark.parametrize("fault", ["nan", "gate_inf", "up_inf", "fp32_overflow", "bf16_overflow"])
def test_fused_swiglu_flags_preserve_raw_invalidity_and_both_overflows(fault):
    raw = torch.ones(2, 128, dtype=torch.bfloat16)
    limit = 10.0
    if fault == "nan":
        raw[1, 7] = float("nan")
    elif fault == "gate_inf":
        raw[1, 7] = float("inf")
    elif fault == "up_inf":
        raw[1, 64 + 7] = -float("inf")
    elif fault == "fp32_overflow":
        limit = 0.0
        raw[1, 7] = raw[1, 64 + 7] = torch.finfo(torch.bfloat16).max
    else:
        limit = 0.0
        raw[1, 7], raw[1, 64 + 7] = 181.0 * 2**56, 181.0 * 2**57
    actual, valid = evaluate_fused_swiglu_registers(raw, torch.roll(torch.arange(64), 33), limit)
    assert not valid.all() and (~valid).sum() == 1
    assert torch.isfinite(actual).all() and (actual[~valid] == 0).all()
