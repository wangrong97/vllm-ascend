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
    tree = ast.parse((ROOT / relative).read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), relative, "exec"), namespace)


def config(world=2, *, gs32=True, quantization="vq2a8", enabled=True):
    return SimpleNamespace(
        quant_config=SimpleNamespace(is_gs32=gs32),
        model_config=SimpleNamespace(
            quantization=quantization,
            hf_text_config=SimpleNamespace(num_experts_per_tok=6),
            get_num_experts=lambda: 256,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=world,
            enable_expert_parallel=enabled,
            world_size_across_dp=world,
            pipeline_parallel_size=1,
        ),
    )


@pytest.fixture
def selection():
    kinds = Enum("MoECommType", "ALLGATHER MC2 ALLTOALL FUSED_MC2")
    soc = Enum("AscendDeviceType", "A2 A3 A5 _310P")
    namespace = {
        "VllmConfig": object,
        "MoECommType": kinds,
        "AscendDeviceType": soc,
        "is_moe_model": lambda _: True,
        "get_mc2_tokens_capacity": lambda: 512,
        "get_ascend_device_type": lambda: soc.A5,
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
            "_select_a2_moe_comm_method",
            "_select_a3_moe_comm_method",
            "_select_a5_moe_comm_method",
        },
        namespace,
    )
    return namespace


@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("tokens", [1, 32, 128, 1024])
def test_gs32_ep_comm_always_selects_allgather(selection, world, tokens):
    assert selection["select_moe_comm_method"](tokens, config(world)) == selection["MoECommType"].ALLGATHER


@pytest.mark.parametrize("gs32,quantization", [(False, "vq2a8"), (True, "w8a8"), (False, None)])
def test_gs32_ep_comm_does_not_change_other_quantization(selection, gs32, quantization):
    actual = selection["select_moe_comm_method"](1, config(gs32=gs32, quantization=quantization))
    assert actual == selection["MoECommType"].MC2


def test_gs32_ep_comm_non_moe_still_has_no_method(selection):
    selection["is_moe_model"] = lambda _: False
    assert selection["select_moe_comm_method"](1, config()) is None


@pytest.mark.parametrize("world,gs32,enabled", [(1, True, False), (2, False, True), (4, True, False)])
def test_gs32_ep_comm_override_requires_both_gs32_and_ep(selection, world, gs32, enabled):
    assert "gs32_ep_uses_allgather" in selection
    assert not selection["gs32_ep_uses_allgather"](config(world, gs32=gs32, enabled=enabled))


@pytest.mark.parametrize("world", [2, 4, 8])
def test_gs32_ep_comm_setup_does_not_construct_unsupported_backends(selection, world):
    assert "gs32_ep_uses_allgather" in selection
    selection.update(
        _MoECommMethods={},
        get_current_vllm_config=lambda: config(world),
        AllGatherCommImpl=Mock(return_value=object()),
        AlltoAllCommImpl=Mock(side_effect=AssertionError("GS32 must not initialize AlltoAll")),
        MC2CommImpl=Mock(side_effect=AssertionError("GS32 must not initialize MC2")),
        FusedMC2CommImpl=Mock(side_effect=AssertionError("GS32 must not initialize FusedMC2")),
    )
    load_functions("vllm_ascend/ops/fused_moe/moe_comm_method.py", {"setup_moe_comm_method"}, selection)
    moe = SimpleNamespace(ep_size=world)
    selection["setup_moe_comm_method"](moe)
    selection["AllGatherCommImpl"].assert_called_once_with(moe)
    assert list(selection["_MoECommMethods"]) == [selection["MoECommType"].ALLGATHER]


def test_gs32_ep_comm_setup_preserves_other_models(selection):
    selection.update(_MoECommMethods={}, get_current_vllm_config=lambda: config(gs32=False))
    names = ("AllGatherCommImpl", "AlltoAllCommImpl", "MC2CommImpl", "FusedMC2CommImpl")
    selection.update({name: Mock(return_value=object()) for name in names})
    load_functions("vllm_ascend/ops/fused_moe/moe_comm_method.py", {"setup_moe_comm_method"}, selection)
    moe = SimpleNamespace(ep_size=4)
    selection["setup_moe_comm_method"](moe)
    for name in names:
        selection[name].assert_called_once_with(moe)
