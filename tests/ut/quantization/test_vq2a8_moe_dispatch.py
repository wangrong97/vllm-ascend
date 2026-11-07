# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for the v0.30 MoE dispatch boundary; no device emulation."""

import ast
from enum import Enum, auto
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.quantization.quant_type import QuantType

ROOT = Path(__file__).resolve().parents[3]


def load_function(relative, name, scope, owner=None):
    source = ROOT / relative
    tree = ast.parse(source.read_text(encoding="utf-8"))
    nodes = tree.body
    if owner is not None:
        nodes = next(node for node in nodes if isinstance(node, ast.ClassDef) and node.name == owner).body
    function = next(node for node in nodes if isinstance(node, ast.FunctionDef) and node.name == name)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), scope)
    return scope[name]


def mlp_input():
    return NS(
        quant=NS(quant_type=QuantType.VQ2A8, is_quant=True),
        weights=NS(vq2a8=NS(apply=Mock(return_value=(object(), None)))),
        hidden_states=torch.zeros(3, 64, dtype=torch.bfloat16),
        group_list=torch.tensor([1, 0, 2]),
        group_list_type=1,
        dynamic_scale=None,
        dynamic_eplb=False,
        lora_context=None,
        activation="silu",
        swiglu_limit=0.5,
    )


def mlp_dispatch(scope=None):
    return load_function(
        "vllm_ascend/ops/fused_moe/moe_mlp.py",
        "apply_moe_mlp",
        {"QuantType": QuantType, **(scope or {})},
    )


@pytest.mark.parametrize("group_type", [0, 1])
def test_gs32_dispatch_preserves_whole_ffn_and_group_metadata(group_type):
    value = mlp_input()
    value.group_list_type = group_type
    if group_type == 0:
        value.group_list = value.group_list.cumsum(0)
    hooks = Mock()
    assert mlp_dispatch()(value, hooks) is value.weights.vq2a8.apply.return_value
    value.weights.vq2a8.apply.assert_called_once_with(
        value.hidden_states, value.group_list, group_type, value.swiglu_limit
    )
    assert not hooks.mock_calls


@pytest.mark.parametrize("fault", ["payload", "dynamic_scale", "dynamic_eplb", "lora_context", "activation"])
def test_gs32_dispatch_rejects_unsupported_inputs_before_any_projection(fault):
    value = mlp_input()
    bank = value.weights.vq2a8
    if fault == "payload":
        value.weights.vq2a8 = None
    else:
        setattr(value, fault, "gelu" if fault == "activation" else object())
    with pytest.raises(ValueError, match="payload|BF16|silu"):
        mlp_dispatch()(value, Mock())
    bank.apply.assert_not_called()


@pytest.mark.parametrize("fused", [False, True])
def test_other_quant_methods_keep_v030_gmm_hooks(fused):
    value = mlp_input()
    value.quant.quant_type = QuantType.W4A8MXFP
    hidden, scale, result, event = object(), object(), object(), object()
    hooks = Mock()
    hooks.supports_fused_activation.return_value = fused
    hooks.apply_gmm1_act_quant.return_value = hidden, scale
    hooks.apply_act_quant.return_value = hidden, scale
    hooks.apply_gmm2.return_value = result
    npu = NS(current_stream=lambda: NS(record_event=lambda: event))
    activation = Mock(return_value=object())
    dispatch = mlp_dispatch({"torch": NS(npu=npu), "_unified_apply_activation": activation})
    assert dispatch(value, hooks) == (result, event)
    hooks.apply_gmm2.assert_called_once_with(value, hidden, scale)
    value.weights.vq2a8.apply.assert_not_called()
    if fused:
        hooks.apply_gmm1_act_quant.assert_called_once_with(value)
        hooks.apply_gmm1.assert_not_called()
        activation.assert_not_called()
    else:
        hooks.apply_gmm1.assert_called_once_with(value)
        activation.assert_called_once_with(value, hooks.apply_gmm1.return_value, hooks)
        hooks.apply_act_quant.assert_called_once_with(value, activation.return_value)


def test_gs32_scheme_supplies_quant_method_to_v030_communicator():
    builder, comm = Mock(), Mock()
    apply = load_function(
        "vllm_ascend/quantization/methods/w2a8/vq2a8_gs32.py",
        "apply",
        {"_EXTRA_CTX": NS(moe_comm_method=comm), "build_fused_experts_input": builder},
        owner="AscendGS32FusedMoEScheme",
    )
    layer = NS(
        vq2a8_weights=object(),
        ascend_expert_map=None,
        global_redundant_expert_num=0,
        ascend_mc2_mask=None,
        apply_router_weight_on_input=False,
        ascend_pertoken_scale=None,
        activation="silu",
    )
    scheme = NS(quant_type=QuantType.VQ2A8, projection_anchor=lambda *_: object())
    hidden, weights, ids = object(), object(), object()
    assert apply(scheme, layer, hidden, weights, ids) is comm.fused_experts.return_value
    comm.fused_experts.assert_called_once_with(fused_experts_input=builder.return_value, quant_method=scheme)
    assert builder.call_args.kwargs["layer"] is layer
    assert builder.call_args.kwargs["vq2a8"] is layer.vq2a8_weights
    assert builder.call_args.kwargs["hidden_states"] is hidden
    assert builder.call_args.kwargs["topk_weights"] is weights
    assert builder.call_args.kwargs["topk_ids"] is ids


@pytest.mark.parametrize("gs32", [False, True])
def test_gs32_comm_setup_keeps_shape_owned_v030_cache(gs32):
    class CommType(Enum):
        ALLGATHER = auto()
        ALLTOALL = auto()
        MC2 = auto()
        FUSED_MC2 = auto()

    constructors = {
        "AllGatherCommImpl": Mock(side_effect=lambda _: object()),
        "AlltoAllCommImpl": Mock(side_effect=lambda _: object()),
        "MC2CommImpl": Mock(side_effect=lambda _: object()),
        "FusedMC2CommImpl": Mock(side_effect=lambda _: object()),
    }
    scope = {
        "MoECommType": CommType,
        "_MoECommMethods": {},
        "gs32_ep_uses_allgather": lambda _: gs32,
        "get_current_vllm_config": lambda: object(),
        **constructors,
    }
    source = "vllm_ascend/ops/fused_moe/moe_comm_method.py"
    for name in ("_moe_config_key", "get_moe_comm_method", "setup_moe_comm_method"):
        load_function(source, name, scope)
    configs = [NS(ep_size=2, num_experts=experts, num_local_experts=experts // 2) for experts in (128, 256)]
    for config in configs:
        scope["setup_moe_comm_method"](config)
    methods = [scope["get_moe_comm_method"](CommType.ALLGATHER, config) for config in configs]
    assert methods[0] is not None and methods[1] is not None and methods[0] is not methods[1]
    assert len(scope["_MoECommMethods"]) == (2 if gs32 else 8)
    if gs32:
        for name in ("AlltoAllCommImpl", "MC2CommImpl", "FusedMC2CommImpl"):
            constructors[name].assert_not_called()
    else:
        assert all(constructor.call_count == 2 for constructor in constructors.values())
