# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source-v3 per-expert typed banks under linear whole-expert EP."""
# ruff: noqa: F401, F811  # Imported pytest fixtures are injected by name.

from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from tests.ut.quantization.test_vq2a8_gs32_fixtures import (
    load_source_experts,
    modules,
    source_field,
    source_layer_spec,
    source_parameters,
)
from vllm_ascend.quantization.methods.w2a8.vq2a8_gs32_format import convert_expert_tensor, source_projection_fields

GLOBAL_EXPERTS = 16
STORED_EXPERTS = (0, 3, 5, 8, 14, 15)


def make_scheme(modules, world, rank, ids=STORED_EXPERTS):
    spec = source_layer_spec(modules, ids)
    config = NS(
        num_experts=GLOBAL_EXPERTS,
        num_logical_experts=GLOBAL_EXPERTS,
        num_local_experts=GLOBAL_EXPERTS // world,
        ep_size=world,
        ep_rank=rank,
    )
    return source_parameters(modules, spec, config)


def load_declared_experts(scheme, layer, *, omit=frozenset()):
    for kind, projection in (("gate_up", "w13"), ("down", "w2")):
        spec = scheme.layer_spec.specs[kind]
        for expert in scheme.layer_spec.expert_ids:
            quant_type = scheme.layer_spec.quant_types[kind][expert]
            for field in source_projection_fields(spec, quant_type):
                if (kind, expert, field) in omit:
                    continue
                scheme.load_expert_tensor(
                    layer, expert, projection, field, source_field(spec, quant_type, field, expert)
                )


def test_natural_order_ep_banks_omit_perm_and_localize_experts(modules):
    ids = tuple(range(GLOBAL_EXPERTS))
    scheme, layer = make_scheme(modules, 2, 1, ids)
    assert not hasattr(layer, "w13_perm") and not hasattr(layer, "w2_perm")
    load_declared_experts(scheme, layer)
    scheme.validate_loaded()
    with pytest.raises(ValueError, match="Unexpected source-v3 expert field"):
        scheme.load_expert_tensor(layer, 8, "w13", "perm", torch.arange(128, dtype=torch.int32))
    factory = Mock(return_value=object())
    modules.stub("vllm_ascend.ops.fused_moe.vq2a8_gs32", GS32ExpertWeights=factory)
    modules.stub("vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4", prepare_mxfp4_projection=lambda w, s: (w, s))
    scheme.process_weights_after_loading(layer)
    local_spec = factory.call_args.args[1]
    assert local_spec.expert_ids == tuple(range(8))
    assert local_spec.quant_types == {
        kind: {expert - 8: quant_type for expert, quant_type in types.items() if expert >= 8}
        for kind, types in scheme.layer_spec.quant_types.items()
    }


@pytest.mark.parametrize("field", ["packed_indices", "codebooks", "out_dim_scale"])
def test_natural_order_does_not_allow_other_missing_vq_fields(modules, field):
    scheme, layer = make_scheme(modules, 4, 2, (8,))
    load_declared_experts(scheme, layer, omit=frozenset({("gate_up", 8, field)}))
    with pytest.raises(ValueError, match="Incomplete source-v3"):
        scheme.validate_loaded()


@pytest.mark.parametrize("world", [1, 2, 4, 8])
def test_ep_per_expert_loading_reconstructs_all_projection_banks(modules, world):
    visited = []
    for rank in range(world):
        scheme, layer = make_scheme(modules, world, rank)
        load_source_experts(scheme, layer)
        assert scheme.validate_loaded() == set(dict(layer.named_parameters()))
        visited.extend(scheme.placement.global_ids)
        for name, parameter in layer.named_parameters():
            kind, quant_type, field, spec = scheme.parameter_metadata[name]
            for expert, slot in scheme.parameter_experts[name].items():
                expected = convert_expert_tensor(field, source_field(spec, quant_type, field, expert), spec, quant_type)
                torch.testing.assert_close(parameter[slot], expected, rtol=0, atol=0)
            assert parameter.shape[0] == len(scheme.parameter_experts[name])
            assert all(
                rank * GLOBAL_EXPERTS // world <= e < (rank + 1) * GLOBAL_EXPERTS // world
                for e in scheme.parameter_experts[name]
            )
    assert tuple(visited) == STORED_EXPERTS


@pytest.mark.parametrize("world,rank", [(2, 1), (4, 1), (8, 3)])
@pytest.mark.parametrize("graph_row_capacity", [8, 384, 512])
def test_ep_postload_local_namespace_and_empty_hash_rank(modules, world, rank, graph_row_capacity):
    scheme, layer = make_scheme(modules, world, rank)
    scheme.graph_row_capacity = graph_row_capacity
    load_source_experts(scheme, layer)
    factory = Mock(return_value=object())
    modules.stub("vllm_ascend.ops.fused_moe.vq2a8_gs32", GS32ExpertWeights=factory)
    modules.stub("vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4", prepare_mxfp4_projection=lambda w, s: (w, s))
    scheme.process_weights_after_loading(layer)
    assert factory.call_args.kwargs["graph_row_capacity"] == graph_row_capacity
    local_spec = factory.call_args.args[1]
    expected = tuple(
        e - rank * GLOBAL_EXPERTS // world
        for e in STORED_EXPERTS
        if rank * GLOBAL_EXPERTS // world <= e < (rank + 1) * GLOBAL_EXPERTS // world
    )
    assert local_spec.expert_ids == expected
    for kind in local_spec.specs:
        assert local_spec.quant_types[kind] == {
            e: scheme.layer_spec.quant_types[kind][e + layer.vq2a8_expert_start] for e in expected
        }
    assert layer.vq2a8_local_global_expert_ids == scheme.placement.global_ids
    assert layer.vq2a8_ep_rank == rank
    assert layer.vq2a8_num_experts == GLOBAL_EXPERTS // world
    if not expected:
        assert not list(layer.parameters())
        assert scheme.projection_anchor(layer, "w13") is layer.vq2a8_anchor


@pytest.mark.parametrize("fault", ["shape", "dtype", "payload", "duplicate", "shard"])
def test_ep_invalid_local_field_never_marks_slot_complete(modules, fault):
    scheme, layer = make_scheme(modules, 4, 2)
    parameter = layer.w2_row_scale_bits  # global expert 8
    parameter.data.fill_(-1)
    source = torch.ones(128, dtype=torch.float32).view(torch.int32).to(torch.uint64)
    kwargs = {"expert_id": 8}
    if fault == "shape":
        source = source[:-1]
    elif fault == "dtype":
        source = source.long()
    elif fault == "payload":
        source.zero_()
    elif fault == "duplicate":
        scheme.weight_loader(parameter, source, **kwargs)
    else:
        kwargs["shard_id"] = 0
    before = parameter.detach().clone()
    with pytest.raises(ValueError):
        scheme.weight_loader(parameter, source, **kwargs)
    torch.testing.assert_close(parameter, before)
    assert scheme.loaded_experts["w2_row_scale_bits"] == ({8} if fault == "duplicate" else set())


def test_ep_remote_experts_are_skipped_without_native_placeholder(modules):
    scheme, layer = make_scheme(modules, 8, 3)
    assert not scheme.placement.global_ids and not list(layer.parameters())
    assert (
        scheme.load_expert_tensor(
            layer, 0, "w13", "packed_indices", source_field(scheme.layer_spec.specs["gate_up"], "vq2", "packed_indices")
        )
        is False
    )
    with pytest.raises(ValueError, match="Unexpected source-v3 expert field"):
        scheme.load_expert_tensor(layer, 0, "w13", "perm", torch.arange(128, dtype=torch.int32))
    assert scheme.validate_loaded() == set()


def test_all_mxfp4_projection_uses_its_actual_resident_anchor(modules):
    scheme, layer = make_scheme(modules, 8, 7, ids=(14, 15))
    assert not hasattr(layer, "w13_packed_zn")
    assert scheme.projection_anchor(layer, "w13") is layer.w13_mxfp4_weight


@pytest.mark.parametrize(
    "field,value",
    [("ep_size", 3), ("ep_rank", -1), ("ep_rank", 4), ("num_local_experts", 2), ("num_logical_experts", 15)],
)
def test_ep_rejects_inconsistent_placement(modules, field, value):
    scheme, _ = make_scheme(modules, 4, 1)
    setattr(scheme.moe_config, field, value)
    with pytest.raises(ValueError):
        scheme.get_weight(scheme.moe_config.num_local_experts, 64, 128, torch.bfloat16)


@pytest.mark.parametrize("ids", [(), (0, 0), (-1,), (GLOBAL_EXPERTS,), (True,), (1.0,)])
def test_ep_rejects_invalid_global_inventory(modules, ids):
    with pytest.raises(ValueError, match="inventory"):
        make_scheme(modules, 4, 0, ids)


def test_ep_rejects_matrix_sharding(modules):
    scheme, _ = make_scheme(modules, 4, 0)
    with pytest.raises(ValueError, match="geometry"):
        scheme.get_weight(4, 32, 128, torch.bfloat16)
