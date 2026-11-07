# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused activation wrapper contracts; CPU tests never execute an NPU kernel."""

import ast
import importlib
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


@pytest.mark.parametrize("width", [2048, 4096])
@pytest.mark.parametrize("row", [2**19, 2**20 + 1])
def test_activation_pointer_strides_do_not_wrap_at_int32(width, row):
    """Evaluate actual address expressions without allocating multi-GiB banks."""
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/triton/activation/vq2a8_gs32.py"
    module = ast.parse(source.read_text(encoding="utf-8"))
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
        "output_ptr",
        "output_scale_ptr",
        "flags_ptr",
        "source_row_ptr",
    }
    namespace = {
        "tl": SimpleNamespace(program_id=lambda axis: torch.tensor([row], dtype=torch.int32), int64=torch.int64),
        "K": width,
        "GROUP": 32,
        "columns": columns,
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
            "output_ptr": width,
            "output_scale_ptr": width // 32,
        }.get(pointer, 1)
        assert int(actual.min()) >= row * stride
        assert int(actual.max()) < (row + 1) * stride
        checked.add(pointer)
    assert checked == pointers


@pytest.mark.parametrize("limit", [0.0, 10.0])
def test_gs32_fused_activation_unchecked_swiglu_preserves_original_arithmetic(limit):
    from tests.support.vq2a8_oracle import staged_swiglu, staged_swiglu_unchecked

    generator = torch.Generator().manual_seed(314)
    raw = (torch.randn(6, 128, generator=generator) * 20).bfloat16()
    gate, up = raw.float().chunk(2, dim=-1)
    if limit > 0:
        gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
    expected = (torch.nn.functional.silu(gate) * up).bfloat16().contiguous()
    assert torch.equal(staged_swiglu_unchecked(raw, limit).view(torch.int16), expected.view(torch.int16))
    assert torch.equal(staged_swiglu(raw, limit).view(torch.int16), expected.view(torch.int16))


def test_gs32_fused_activation_unchecked_requires_raw_gate_check_after_clamp():
    from tests.support.vq2a8_oracle import staged_swiglu, staged_swiglu_unchecked

    raw = torch.ones(1, 128, dtype=torch.bfloat16)
    raw[0, 0] = float("inf")
    assert torch.isfinite(staged_swiglu_unchecked(raw, 10)).all()
    with pytest.raises(ValueError, match="GateUp output"):
        staged_swiglu(raw, 10)


@pytest.mark.parametrize("limit", [-1.0, float("nan"), float("inf")])
def test_gs32_fused_activation_unchecked_still_validates_swiglu_metadata(limit):
    from tests.support.vq2a8_oracle import staged_swiglu_unchecked

    with pytest.raises(ValueError):
        staged_swiglu_unchecked(torch.ones(1, 128, dtype=torch.bfloat16), limit)
