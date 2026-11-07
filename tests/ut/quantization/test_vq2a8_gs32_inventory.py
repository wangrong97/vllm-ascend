# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU coverage of real-checkpoint probes for natural input columns."""
# ruff: noqa: F401, F811  # Imported pytest fixtures are injected by name.

import json

import pytest
import torch

from tests.e2e.nightly.vq2a8_gs32_helpers import checkpoint_inventory, load_selected_layer
from tests.ut.quantization.test_vq2a8_gs32_checkpoint import source_checkpoint, write_checkpoint
from tests.ut.quantization.test_vq2a8_gs32_fixtures import modules
from tools.vq2a8_source_contract import source_contract


def test_probe_loads_selected_real_payloads_in_natural_order_and_skips_mtp(source_checkpoint):
    tensors = dict(source_checkpoint.tensors)
    tensors["mtp.2.ffn.experts.0.w1.weight"] = "I8", torch.zeros(32, 16, dtype=torch.int8)
    write_checkpoint(source_checkpoint.root, tensors)
    config_path = source_checkpoint.root / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(hidden_size=128, moe_intermediate_size=64, n_routed_experts=4)
    config_path.write_text(json.dumps(config))

    _, _, inventory = checkpoint_inventory(source_checkpoint.root)
    assert inventory == {
        0: {
            "gate_up": {0: "vq2", 1: "vq2", 2: "mxfp4", 3: "mxfp4"},
            "down": {0: "vq2", 1: "mxfp4", 2: "mxfp4", 3: "mxfp4"},
        }
    }
    layer, spec, _ = load_selected_layer(source_checkpoint.root, expert_count=3)
    assert layer.w13_packed_zn.shape[0] == 2
    assert layer.w2_packed_zn.shape[0] == 1
    assert not hasattr(layer, "w13_perm") and not hasattr(layer, "w2_perm")
    assert spec.expert_ids == (0, 1, 2)


def test_probe_rejects_three_fields_when_a_required_vq2_payload_is_missing(source_checkpoint):
    tensors = dict(source_checkpoint.tensors)
    del tensors["layers.0.ffn.experts.0.w13.codebooks"]
    write_checkpoint(source_checkpoint.root, tensors)
    with pytest.raises(ValueError, match="Incomplete source-v3 expert"):
        checkpoint_inventory(source_checkpoint.root)


@pytest.mark.parametrize("identity", [False, True])
def test_probe_rejects_explicit_perm_payloads(source_checkpoint, identity):
    value = torch.arange(128, dtype=torch.int32)
    if not identity:
        value = value.roll(1)
    tensors = {**source_checkpoint.tensors, "layers.0.ffn.experts.0.w13.perm": ("I32", value)}
    write_checkpoint(source_checkpoint.root, tensors)
    with pytest.raises(ValueError, match="perm|Incomplete source-v3 expert"):
        checkpoint_inventory(source_checkpoint.root)


@pytest.mark.parametrize("fault", (None, "unknown", "permutation", "incomplete"))
def test_inventory_accepts_complete_folded_contract_and_rejects_contradictions(source_checkpoint, fault):
    config_path = source_checkpoint.root / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    quantization = config["quantization_config"]
    quantization.update(source_contract()["FOLDED_HEAD_CONFIG"])
    if fault == "unknown":
        quantization["unsupported"] = True
    elif fault == "permutation":
        quantization["vq_input_permutation"] = True
    elif fault == "incomplete":
        del quantization["vq_weight_layout"]
    config_path.write_text(json.dumps(config), encoding="utf-8")
    if fault:
        with pytest.raises(ValueError, match="source-v3"):
            checkpoint_inventory(source_checkpoint.root)
    else:
        parsed, _, inventory = checkpoint_inventory(source_checkpoint.root)
        assert parsed["quantization_config"] == quantization
        assert inventory
