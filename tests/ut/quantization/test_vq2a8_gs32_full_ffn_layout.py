# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Byte-preserving complete-FFN layout and resident-bank ownership checks."""
# ruff: noqa: F401, F811  # pytest fixtures are imported by name.

from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from tests.ut.quantization.test_vq2a8_gs32_fixtures import (
    load_source_experts,
    modules,
    source_layer_spec,
    source_parameters,
)
from vllm_ascend.quantization.methods.w2a8.vq2a8_gs32_contract import resolve_gs32_options
from vllm_ascend.quantization.methods.w2a8.vq2a8_gs32_full_ffn_layout import (
    FP32_ONE_BITS,
    build_full_ffn_bank,
    interleave_gate_up_blocks,
    pack_mxfp4_zn,
    validate_full_ffn_geometry,
)


def literal_zn(row_packed):
    """Independent scalar indexing of the destination's adjacent-N nibbles."""
    rows, half_columns = row_packed.shape
    columns = half_columns * 2
    out = torch.empty((rows // 32, columns // 16, 16, 16), dtype=torch.uint8)
    for n in range(0, rows, 2):
        for k in range(columns):
            low = (int(row_packed[n, k // 2]) >> (4 * (k % 2))) & 15
            high = (int(row_packed[n + 1, k // 2]) >> (4 * (k % 2))) & 15
            out[n // 32, k // 16, k % 16, (n % 32) // 2] = low | (high << 4)
    return out


@pytest.mark.parametrize("rows,columns", [(32, 16), (64, 64), (128, 128)])
def test_fp4_raw_zn_preserves_every_code_and_axis(rows, columns):
    weight = torch.randint(
        0, 256, (rows, columns // 2), generator=torch.Generator().manual_seed(183), dtype=torch.uint8
    )
    expected = literal_zn(weight)
    actual = pack_mxfp4_zn(weight)
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # Explicitly include E2M1 negative zero (0x8) as both low/high codes.
    negative_zero = torch.full_like(weight, 0x88)
    assert torch.equal(pack_mxfp4_zn(negative_zero), torch.full_like(actual, 0x88))


def test_interleaving_preserves_bytes_and_does_not_touch_source():
    natural = torch.arange(8 * 3 * 16 * 8).remainder(256).to(torch.uint8).reshape(8, 3, 16, 8)
    saved = natural.clone()
    out = interleave_gate_up_blocks(natural)
    torch.testing.assert_close(out, natural[[0, 4, 1, 5, 2, 6, 3, 7]], rtol=0, atol=0)
    assert out.data_ptr() != natural.data_ptr()
    torch.testing.assert_close(natural, saved, rtol=0, atol=0)


def mixed_source(modules, *, world=1, rank=0):
    spec = source_layer_spec(modules)
    # Local test model has all three supported pairs, with non-contiguous W2
    # typed slots: 22,24,44,22. Every EP rank may have a different composition.
    spec = replace(
        spec,
        quant_types={
            "gate_up": {0: "vq2", 1: "vq2", 2: "mxfp4", 3: "vq2"},
            "down": {0: "vq2", 1: "mxfp4", 2: "mxfp4", 3: "vq2"},
        },
    )
    config = NS(num_experts=4, num_logical_experts=4, num_local_experts=4 // world, ep_size=world, ep_rank=rank)
    config.swiglu_limit = 10.0
    scheme, layer = source_parameters(modules, spec, config)
    scheme.swiglu_limit = 10.0
    load_source_experts(scheme, layer)
    for name, parameter in layer.named_parameters():
        if parameter.dtype == torch.uint8:
            parameter.data.copy_(torch.arange(parameter.numel()).remainder(251).reshape_as(parameter).to(torch.uint8))
        else:
            rows = parameter.shape[-1]
            scales = torch.linspace(0.5, 2.0, parameter.numel()).reshape(-1, rows).view(torch.int32).to(torch.int64)
            parameter.data.copy_(scales)
    return scheme, layer


@pytest.mark.parametrize("world,rank", [(1, 0), (2, 0), (2, 1), (4, 2)])
def test_postload_mixed_bank_preserves_weights_and_releases_old_storage(modules, world, rank):
    scheme, layer = mixed_source(modules, world=world, rank=rank)
    originals = {name: value.detach().clone() for name, value in layer.named_parameters()}
    scheme.native_options = resolve_gs32_options({})
    factory = Mock(return_value=object())
    modules.stub("vllm_ascend.ops.fused_moe.vq2a8_gs32", GS32ExpertWeights=factory)
    official = Mock(side_effect=AssertionError("Complete FFN must not prepare official opaque NZ weights"))
    modules.stub("vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4", prepare_mxfp4_projection=official)
    scheme.process_weights_after_loading(layer)
    bank = layer.vq2a8_full_ffn_bank
    assert bank["expert_ids"] == scheme.placement.local_ids
    assert bank["swiglu_limit"] == 10.0
    assert not list(layer.parameters())
    assert all(not hasattr(layer, name) for name in originals)
    assert not official.called
    assert factory.call_args.args[2]["gs32_fusion"] == "full_ffn"
    assert scheme.projection_anchor(layer, "w13") is bank["fields"]["gate_weight"]
    assert scheme.projection_anchor(layer, "w2") is bank["fields"]["down_weight"]
    stage_mask = 0
    for kind, prefix, native, shift in (("gate_up", "w13", "gate", 0), ("down", "w2", "down", 2)):
        rows, columns = scheme.layer_spec.specs[kind].rows, scheme.layer_spec.specs[kind].columns
        flags = []
        for slot, global_expert in enumerate(scheme.placement.global_ids):
            quant_type = scheme.layer_spec.quant_types[kind][global_expert]
            flags.append(quant_type == "mxfp4")
            if quant_type == "vq2":
                source_slot = scheme.parameter_experts[f"{prefix}_packed_zn"][global_expert]
                packed = originals[f"{prefix}_packed_zn"][source_slot]
                table = originals[f"{prefix}_pair_lut"][source_slot]
                if kind == "gate_up":
                    order = [tile for pair in zip(range(rows // 64), range(rows // 64, rows // 32)) for tile in pair]
                    packed, table = packed[order], table[order]
                scales = originals[f"{prefix}_row_scale_bits"][source_slot]
            else:
                source_slot = scheme.parameter_experts[f"{prefix}_mxfp4_weight"][global_expert]
                packed = literal_zn(originals[f"{prefix}_mxfp4_weight"][source_slot])
                table = originals[f"{prefix}_mxfp4_scale"][source_slot]
                if kind == "gate_up":
                    order = [tile for pair in zip(range(rows // 64), range(rows // 64, rows // 32)) for tile in pair]
                    packed = packed[order]
                    table = table.reshape(rows // 32, 32, columns // 32)[order]
                scales = torch.full((rows,), FP32_ONE_BITS, dtype=torch.int64)
            first, last = bank["weight_offsets"][kind][slot : slot + 2]
            actual = bank["fields"][f"{native}_weight"][first:last]
            torch.testing.assert_close(actual, packed.reshape(-1), rtol=0, atol=0)
            actual_table = bank["fields"][f"{native}_table"].reshape(len(bank["expert_ids"]), -1)[slot]
            torch.testing.assert_close(actual_table, table.reshape(-1), rtol=0, atol=0)
            torch.testing.assert_close(bank["fields"][f"{native}_row_scale"][slot], scales, rtol=0, atol=0)
        assert bank[f"{native}_w4"].device.type == "cpu"
        assert bank[f"{native}_w4"].dtype == torch.bool
        assert bank[f"{native}_w4"].tolist() == flags
        stage_mask |= (1 << shift if not all(flags) else 0) | (1 << (shift + 1) if any(flags) else 0)
    assert bank["stage_mask"] == stage_mask
    for name, value in bank["fields"].items():
        assert getattr(layer, f"vq2a8_full_ffn_{name}") is value


def test_default_full_ffn_rejects_w4_w2_before_allocating(modules):
    spec = source_layer_spec(modules)
    spec.quant_types["down"][2] = "vq2"  # Explicit unsupported W4 Gate/W2 Down.
    scheme = modules.method.AscendGS32FusedMoEScheme(spec)
    assert scheme.native_options["gs32_fusion"] == "full_ffn"
    with pytest.raises(ValueError, match="expert 2 precision pair"):
        scheme.get_weight(4, 64, 128, torch.bfloat16)
    assert scheme.placement is None


def test_empty_hash_rank_has_no_fake_expert_or_weight_payload(modules):
    spec = source_layer_spec(modules, ids=(0,))
    config = NS(num_experts=4, num_logical_experts=4, num_local_experts=1, ep_size=4, ep_rank=3, swiglu_limit=10.0)
    scheme, layer = source_parameters(modules, spec, config)
    scheme.swiglu_limit = 10.0
    load_source_experts(scheme, layer)
    scheme.native_options = resolve_gs32_options({})
    modules.stub("vllm_ascend.ops.fused_moe.vq2a8_gs32", GS32ExpertWeights=Mock())
    scheme.process_weights_after_loading(layer)
    bank = layer.vq2a8_full_ffn_bank
    assert bank["expert_ids"] == () and bank["stage_mask"] == 0
    assert all(value.numel() == 0 for value in bank["fields"].values())
    assert scheme.projection_anchor(layer, "w13") is layer.vq2a8_anchor
    assert scheme.projection_anchor(layer, "w2") is layer.vq2a8_anchor


@pytest.mark.parametrize("limit", [-1, float("inf"), float("nan"), True])
def test_bad_limit_rejected_before_packing(modules, limit):
    scheme, layer = mixed_source(modules)
    with pytest.raises(ValueError, match="limit"):
        build_full_ffn_bank(layer, scheme.layer_spec, swiglu_limit=limit)


@pytest.mark.parametrize("gate,down", [((64, 128), (128, 32)), ((128, 32), (128, 64)), ((128, 128), (48, 64))])
def test_unsupported_geometry_is_rejected(gate, down):
    with pytest.raises(ValueError, match="Complete FFN requires"):
        validate_full_ffn_geometry(NS(rows=gate[0], columns=gate[1]), NS(rows=down[0], columns=down[1]))
