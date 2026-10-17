# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Execute the actual DSA output-projection method with CPU operator stand-ins.

These regression tests cover dispatch/layout, not CANN or full attention.
"""

import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch


class Unquantized:
    pass


def projection_method(npu):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/attention/dsa_v1.py"
    methods = [
        node
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.FunctionDef) and node.name in ("_forward_o_proj", "_has_weight_scale")
    ]
    scope = {
        "torch": torch,
        "torch_npu": npu,
        "oproj_tp_enable": lambda: False,
        "olora_tp_enable": lambda: False,
    }
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), scope)
    return scope["_forward_o_proj"]


def column_loader(quantized=False, a5=True):
    """Execute the real column loader; isolate the upstream sharding copy."""
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/linear.py"
    cls = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "AscendColumnParallelLinear"
    )
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "weight_loader")

    class Base:
        def weight_loader(self, param, loaded_weight):
            param.data.copy_(loaded_weight)

    definition = ast.ClassDef(
        name="Column", bases=[ast.Name(id="Base", ctx=ast.Load())], keywords=[], body=[method], decorator_list=[]
    )
    scope = {
        "torch": torch,
        "Parameter": torch.nn.Parameter,
        "Base": Base,
        "UnquantizedLinearMethod": Unquantized,
        "get_current_hardware_profile": lambda: NS(supports=lambda _: a5),
        "HardwareCapability": NS(DYNAMIC_MX_QUANT_FUSION=1),
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[definition], type_ignores=[])), str(path), "exec"), scope)
    owner = scope["Column"]()
    owner.prefix = "model.layers.0.self_attn.wo_a"
    owner.n_local_groups, owner.o_lora_rank, owner.tp_rank = 2, 3, 0
    owner.quant_config = NS(get_name=lambda: "vq2a8")
    owner.quant_method = object() if quantized else Unquantized()
    owner.weight = torch.nn.Parameter(torch.zeros(6, 8, dtype=torch.bfloat16), requires_grad=False)
    return owner


@pytest.mark.parametrize("tokens", [1, 3])
@pytest.mark.parametrize("a5", [False, True])
def test_unquantized_root_load_and_projection_preserve_group_layout(tokens, a5):
    groups, rank, hidden = 2, 3, 8
    weight = ((torch.arange(groups * rank * hidden).reshape(groups * rank, hidden) % 17) / 16).bfloat16()
    x = ((torch.arange(tokens * groups * hidden).reshape(tokens, groups, hidden) % 13) / 16).bfloat16()
    column = column_loader(a5=a5)
    for _ in range(2):
        column.weight_loader(column.weight, weight)
        assert column.weight.shape == (groups, hidden, rank)
    owner = NS(n_local_groups=groups, o_lora_rank=rank, wo_a=column, wo_b=lambda x: x, support_fp8_attention=a5)
    assert not hasattr(column, "weight_scale")
    output = torch.empty(tokens, groups * rank, dtype=x.dtype)
    npu = NS(npu_transpose_batchmatmul=lambda x, w, **_: torch.bmm(x.transpose(0, 1), w).transpose(0, 1))
    result = projection_method(npu)(owner, x, output)
    expected = torch.cat(
        [torch.nn.functional.linear(x[:, group], weight[group * rank : (group + 1) * rank]) for group in range(groups)],
        dim=-1,
    )
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert result is output


def test_a5_quantized_column_does_not_apply_bf16_reshape():
    column = column_loader(quantized=True)
    weight = torch.ones(6, 8, dtype=torch.bfloat16)
    column.weight_loader(column.weight, weight)
    assert column.weight.shape == weight.shape


def test_existing_a5_quantized_branch_keeps_scale_and_permutation_contract():
    calls = []
    x_scale, w_scale = NS(view=lambda _: "x_scale"), NS(view=lambda _: "w_scale")
    npu = NS(
        npu_dynamic_mx_quant=lambda x, **kw: (x, x_scale),
        npu_transpose_quant_batchmatmul=lambda *a, **kw: calls.append(kw) or torch.ones(1, 2, 3),
    )
    owner = NS(
        support_fp8_attention=True,
        n_local_groups=2,
        o_lora_rank=3,
        wo_a=NS(weight=torch.zeros(2, 8, 3).to(torch.float8_e4m3fn), quant_method=object(), weight_scale=w_scale),
        wo_b=lambda x: x,
    )
    projection_method(npu=npu)(owner, torch.zeros(1, 2, 8), torch.empty(1, 6))
    assert calls[0]["x1_scale"] == "x_scale" and calls[0]["x2_scale"] == "w_scale"
    assert calls[0]["group_sizes"] == (0, 0, 32)
    assert calls[0]["perm_x1"] == calls[0]["perm_y"] == (1, 0, 2)


def test_non_a5_grouped_weight_path_unchanged():
    calls = []
    npu = NS(npu_transpose_batchmatmul=lambda *a, **kw: calls.append(kw) or torch.ones(1, 2, 3))
    owner = NS(support_fp8_attention=False, n_local_groups=2, wo_a=NS(weight=torch.zeros(2, 8, 3)), wo_b=lambda x: x)
    projection_method(npu=npu)(owner, torch.zeros(1, 2, 8), torch.empty(1, 6))
    assert calls[0]["scale"] is None and calls[0]["batch_split_factor"] == 1
