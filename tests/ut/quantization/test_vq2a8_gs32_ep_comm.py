# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Execute actual communication-selection functions without initializing NPU/HCCL."""

import ast
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[3]


def load_functions(relative, names, namespace):
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in names)
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id in names for target in node.targets)
        )
    ]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), relative, "exec"), namespace)


def config(world=2, *, dp=1, gs32=True, quantization="vq2a8", enabled=True):
    return SimpleNamespace(
        quant_config=SimpleNamespace(is_gs32=gs32),
        model_config=SimpleNamespace(
            quantization=quantization,
            hf_text_config=SimpleNamespace(num_experts_per_tok=6),
            get_num_experts=lambda: 256,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=world,
            data_parallel_size=dp,
            enable_expert_parallel=enabled,
            world_size_across_dp=world * dp,
            pipeline_parallel_size=1,
        ),
    )


@pytest.fixture
def selection():
    kinds = Enum("MoECommType", "ALLGATHER MC2 ALLTOALL FUSED_MC2")
    policies = Enum("MoECommPolicy", "CAPACITY_AND_EXPERT_DENSITY FUSED_OR_CAPACITY CAPACITY_AND_WORLD_SIZE ALLGATHER")
    quant_types = Enum("QuantType", "NONE MXFP8")
    profile = SimpleNamespace(moe_comm_policy=policies.CAPACITY_AND_WORLD_SIZE)
    namespace = {
        "VllmConfig": object,
        "FusedMoEConfig": object,
        "MoECommMethod": object,
        "MoECommType": kinds,
        "MoECommPolicy": policies,
        "QuantType": quant_types,
        "A5_SUPPORT_MEGA_MOE_QUANT_TYPES": (quant_types.MXFP8,),
        "is_mega_moe_supported": lambda: True,
        "get_current_hardware_profile": lambda: profile,
        "is_moe_model": lambda _: True,
        "get_mc2_tokens_capacity": lambda: 512,
        "get_ep_group": lambda: SimpleNamespace(world_size=8),
        "get_ascend_config": lambda: SimpleNamespace(enable_fused_mc2=0),
        "use_cann_megamoe": lambda _: False,
        "logger": Mock(),
    }
    load_functions(
        "vllm_ascend/ascend_forward_context.py",
        {
            "gs32_ep_uses_allgather",
            "select_moe_comm_method",
            "_select_capacity_and_expert_density_moe_comm_method",
            "_select_fused_or_capacity_moe_comm_method",
            "_select_capacity_and_world_size_moe_comm_method",
            "_MOE_COMM_SELECTORS",
        },
        namespace,
    )
    return namespace


@pytest.mark.parametrize("world,dp", [(2, 1), (4, 1), (8, 1), (1, 2), (1, 4), (1, 8)])
@pytest.mark.parametrize("tokens", [1, 32, 128, 1024])
@pytest.mark.parametrize("is_draft_model", [False, True])
def test_gs32_ep_comm_always_selects_allgather(selection, world, dp, tokens, is_draft_model):
    actual = selection["select_moe_comm_method"](
        tokens, config(world, dp=dp), is_draft_model=is_draft_model, draft_moe_quant_type=selection["QuantType"].NONE
    )
    assert actual == selection["MoECommType"].ALLGATHER


@pytest.mark.parametrize("tokens", [1, 8192])
@pytest.mark.parametrize("dp_size", [4, 8])
@pytest.mark.parametrize(
    "policy", ["CAPACITY_AND_EXPERT_DENSITY", "FUSED_OR_CAPACITY", "CAPACITY_AND_WORLD_SIZE", "ALLGATHER"]
)
def test_gs32_ep_comm_dp_does_not_fall_into_hardware_fused_policy(selection, tokens, policy, dp_size):
    selection["get_current_hardware_profile"] = lambda: SimpleNamespace(
        moe_comm_policy=selection["MoECommPolicy"][policy]
    )
    selection["get_ascend_config"] = lambda: SimpleNamespace(enable_fused_mc2=1)
    selection["use_cann_megamoe"] = lambda _: True
    actual = selection["select_moe_comm_method"](tokens, config(1, dp=dp_size))
    assert actual == selection["MoECommType"].ALLGATHER


@pytest.mark.parametrize("supported", [False, True])
def test_non_gs32_draft_retains_hardware_quantization_policy(selection, supported):
    selection["get_ascend_config"] = lambda: SimpleNamespace(enable_fused_mc2=1)
    quant_type = selection["QuantType"].MXFP8 if supported else selection["QuantType"].NONE
    actual = selection["select_moe_comm_method"](
        1, config(gs32=False), is_draft_model=True, draft_moe_quant_type=quant_type
    )
    expected = selection["MoECommType"].FUSED_MC2 if supported else selection["MoECommType"].MC2
    assert actual == expected


@pytest.mark.parametrize("gs32,quantization", [(False, "vq2a8"), (True, "w8a8"), (False, None)])
@pytest.mark.parametrize("world,dp", [(2, 1), (1, 4)])
def test_gs32_ep_comm_does_not_change_other_quantization(selection, gs32, quantization, world, dp):
    actual = selection["select_moe_comm_method"](1, config(world, dp=dp, gs32=gs32, quantization=quantization))
    assert actual == selection["MoECommType"].MC2


def test_gs32_ep_comm_non_moe_still_has_no_method(selection):
    selection["is_moe_model"] = lambda _: False
    assert selection["select_moe_comm_method"](1, config()) is None


@pytest.mark.parametrize(
    "world,dp,gs32,enabled",
    [
        (1, 1, True, False),
        (1, 1, True, True),
        (2, 1, False, True),
        (4, 1, True, False),
        (1, 4, True, False),
        (1, 4, False, True),
    ],
)
def test_gs32_ep_comm_override_requires_both_gs32_and_ep(selection, world, dp, gs32, enabled):
    assert "gs32_ep_uses_allgather" in selection
    assert not selection["gs32_ep_uses_allgather"](config(world, dp=dp, gs32=gs32, enabled=enabled))


@pytest.mark.parametrize("world,dp", [(2, 1), (4, 1), (8, 1), (1, 2), (1, 4)])
def test_gs32_ep_comm_setup_does_not_construct_unsupported_backends(selection, world, dp):
    assert "gs32_ep_uses_allgather" in selection
    selection.update(
        _MoECommMethods={},
        get_current_vllm_config=lambda: config(world, dp=dp),
        AllGatherCommImpl=Mock(return_value=object()),
        AlltoAllCommImpl=Mock(side_effect=AssertionError("GS32 must not initialize AlltoAll")),
        MC2CommImpl=Mock(side_effect=AssertionError("GS32 must not initialize MC2")),
        FusedMC2CommImpl=Mock(side_effect=AssertionError("GS32 must not initialize FusedMC2")),
    )
    load_functions(
        "vllm_ascend/ops/fused_moe/moe_comm_method.py",
        {"setup_moe_comm_method", "_moe_config_key", "get_moe_comm_method"},
        selection,
    )
    moe = SimpleNamespace(ep_size=world * dp, num_experts=256, num_local_experts=256 // (world * dp))
    selection["setup_moe_comm_method"](moe)
    selection["AllGatherCommImpl"].assert_called_once_with(moe)
    assert list(selection["_MoECommMethods"]) == [(selection["MoECommType"].ALLGATHER, (256, 256 // (world * dp)))]
    assert selection["get_moe_comm_method"](selection["MoECommType"].ALLGATHER, moe) is (
        selection["AllGatherCommImpl"].return_value
    )


def test_gs32_ep_comm_setup_preserves_other_models(selection):
    selection.update(_MoECommMethods={}, get_current_vllm_config=lambda: config(gs32=False))
    names = ("AllGatherCommImpl", "AlltoAllCommImpl", "MC2CommImpl", "FusedMC2CommImpl")
    selection.update({name: Mock(return_value=object()) for name in names})
    load_functions(
        "vllm_ascend/ops/fused_moe/moe_comm_method.py",
        {"setup_moe_comm_method", "_moe_config_key", "get_moe_comm_method"},
        selection,
    )
    moe = SimpleNamespace(ep_size=4)
    selection["setup_moe_comm_method"](moe)
    for name in names:
        selection[name].assert_called_once_with(moe)


def test_gs32_ep_comm_setup_keeps_distinct_expert_banks(selection):
    first, second = object(), object()
    selection.update(
        _MoECommMethods={},
        get_current_vllm_config=lambda: config(),
        AllGatherCommImpl=Mock(side_effect=[first, second]),
    )
    load_functions(
        "vllm_ascend/ops/fused_moe/moe_comm_method.py",
        {"setup_moe_comm_method", "_moe_config_key", "get_moe_comm_method"},
        selection,
    )
    target = SimpleNamespace(ep_size=2, num_experts=256, num_local_experts=128)
    draft = SimpleNamespace(ep_size=2, num_experts=128, num_local_experts=64)
    selection["setup_moe_comm_method"](target)
    selection["setup_moe_comm_method"](draft)
    kind = selection["MoECommType"].ALLGATHER
    assert selection["get_moe_comm_method"](kind, target) is first
    assert selection["get_moe_comm_method"](kind, draft) is second
