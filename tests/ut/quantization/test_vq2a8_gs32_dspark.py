# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only contracts for a separate official DSpark checkpoint."""

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def dspark():
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/quantization/methods/w2a8/vq2a8_gs32_dspark.py"
    spec = importlib.util.spec_from_file_location("gs32_dspark_contract", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def dspark_config(tmp_path, dspark):
    original = tmp_path / "official"
    original.mkdir()
    target = tmp_path / "vq2a8"
    target.mkdir()
    hf = dict.fromkeys(dspark.DSPARK_GEOMETRY_FIELDS, 4)
    hf.update(
        model_type="deepseek_v4",
        num_hidden_layers=3,
        dspark_block_size=5,
        dspark_target_layer_ids=[0, 1, 2],
        compress_ratios=[0, 4, 128, 0, 0, 0],
        quantization_config={"quant_method": "fp8"},
        expert_dtype="fp4",
    )
    (original / "config.json").write_text(json.dumps(hf), encoding="utf-8")
    names = [
        "mtp.0.main_proj.weight",
        "mtp.0.main_proj.scale",
        "mtp.0.main_norm.weight",
        "mtp.2.norm.weight",
        "mtp.2.markov_head.markov_w1.weight",
        "mtp.2.markov_head.markov_w2.weight",
        *(f"mtp.{stage}.attn.wq_a.weight" for stage in range(3)),
        "embed.weight",
    ]
    weights = dict.fromkeys(names, "draft.safetensors")
    # Full official checkpoints include main-model shards. The standard
    # loader still visits those, so the preflight must check them too.
    weights["layers.0.attn.wq_a.weight"] = "main.safetensors"
    (original / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weights}), encoding="utf-8")
    (original / "draft.safetensors").touch()
    (original / "main.safetensors").touch()
    target_hf = deepcopy(hf)
    target_hf["quantization_config"] = {"quant_method": "vq2a8"}
    target_hf["compress_ratios"] = target_hf["compress_ratios"][:3]
    draft_model = SimpleNamespace(
        model=str(original),
        # Reproduce upstream DeepSeek DSpark's target-quantization override.
        quantization="vq2a8",
        hf_config=SimpleNamespace(**hf),
        max_model_len=1048576,
        dtype="bfloat16",
        use_fp64_gumbel=False,
    )
    config = SimpleNamespace(
        use_v2_model_runner=True,
        model_config=SimpleNamespace(
            model=str(target),
            quantization="vq2a8",
            hf_config=SimpleNamespace(**target_hf),
            max_model_len=262144,
            dtype="bfloat16",
            use_fp64_gumbel=True,
        ),
        speculative_config=SimpleNamespace(
            model=str(original),
            method="dspark",
            enforce_eager=False,
            num_speculative_tokens=5,
            quantization=None,
            draft_model_config=draft_model,
        ),
        quant_config=SimpleNamespace(is_gs32=True),
        compilation_config=SimpleNamespace(
            static_forward_context={},
            cudagraph_mode=SimpleNamespace(decode_mode=lambda: SimpleNamespace(name="FULL")),
            cudagraph_capture_sizes=[6, 12, 24, 48, 96],
            max_cudagraph_capture_size=96,
        ),
        load_config=SimpleNamespace(load_format="auto"),
        additional_config={"vq2a8": {"gs32_execution": "full_graph"}, "enable_cpu_binding": True},
    )
    config.speculative_config.target_model_config = config.model_config
    return config


@pytest.mark.parametrize("enforce_eager", [False, True])
def test_official_dspark_preflight_accepts_compatible_separate_weights(dspark, dspark_config, enforce_eager):
    dspark_config.speculative_config.enforce_eager = enforce_eager
    dspark.validate_vq2a8_dspark_config(dspark_config)


@pytest.mark.parametrize("runtime_default_resolved", [False, True])
def test_dspark_preserves_original_fp4_default_when_checkpoint_omits_expert_dtype(
    dspark, dspark_config, runtime_default_resolved
):
    path = Path(dspark_config.speculative_config.model) / "config.json"
    hf = json.loads(path.read_text(encoding="utf-8"))
    del hf["expert_dtype"]
    path.write_text(json.dumps(hf), encoding="utf-8")
    if not runtime_default_resolved:
        del dspark_config.speculative_config.draft_model_config.hf_config.expert_dtype
    dspark.validate_vq2a8_dspark_config(dspark_config)


@pytest.mark.parametrize("expert_dtype", ["fp8", "bf16", None])
def test_dspark_rejects_runtime_expert_dtype_outside_original_w4a8_path(dspark, dspark_config, expert_dtype):
    dspark_config.speculative_config.draft_model_config.hf_config.expert_dtype = expert_dtype
    with pytest.raises(ValueError, match="FP8/FP4"):
        dspark.validate_vq2a8_dspark_config(dspark_config)


@pytest.mark.parametrize(
    "field,value",
    [
        ("method", "mtp"),
        ("num_speculative_tokens", 4),
        ("num_speculative_tokens", 6),
        ("num_speculative_tokens", True),
        ("enforce_eager", None),
        ("enforce_eager", "false"),
        ("enforce_eager", 1),
    ],
)
def test_dspark_rejects_unsupported_drafting_modes(dspark, dspark_config, field, value):
    setattr(dspark_config.speculative_config, field, value)
    with pytest.raises(ValueError, match="exactly 5"):
        dspark.validate_vq2a8_dspark_config(dspark_config)


def test_dspark_requires_mrv2(dspark, dspark_config):
    dspark_config.use_v2_model_runner = False
    with pytest.raises(ValueError, match="MRV2"):
        dspark.validate_vq2a8_dspark_config(dspark_config)


def test_dspark_rejects_target_checkpoint_as_draft(dspark, dspark_config):
    dspark_config.speculative_config.model = dspark_config.model_config.model
    with pytest.raises(ValueError, match="separate"):
        dspark.validate_vq2a8_dspark_config(dspark_config)


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("hidden_size", 128, "hidden_size"),
        ("vocab_size", 128, "vocab_size"),
        ("compress_ratios", [0, 0, 0], "compression"),
    ],
)
def test_dspark_rejects_incompatible_target_geometry(dspark, dspark_config, field, value, error):
    setattr(dspark_config.model_config.hf_config, field, value)
    with pytest.raises(ValueError, match=error):
        dspark.validate_vq2a8_dspark_config(dspark_config)


@pytest.mark.parametrize("shard", ["draft.safetensors", "main.safetensors"])
def test_dspark_preflight_requires_all_official_shards(dspark, dspark_config, shard):
    (Path(dspark_config.speculative_config.model) / shard).unlink()
    with pytest.raises(ValueError, match="shard is missing"):
        dspark.validate_vq2a8_dspark_config(dspark_config)


def test_dspark_preflight_rejects_index_without_mtp(dspark, dspark_config):
    root = Path(dspark_config.speculative_config.model)
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"embed.weight": "main.safetensors"}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="missing draft weights"):
        dspark.validate_vq2a8_dspark_config(dspark_config)


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("dspark_block_size", 6, "block_size=5"),
        ("expert_dtype", "fp8", "FP8/FP4"),
        ("expert_dtype", "bf16", "FP8/FP4"),
        ("quantization_config", {"quant_method": "vq2a8"}, "FP8/FP4"),
        ("quantization_config", None, "FP8/FP4"),
        ("quantization_config", [], "FP8/FP4"),
        ("dspark_target_layer_ids", [0, 1, 3], "auxiliary-layer"),
    ],
)
def test_dspark_preflight_rejects_incompatible_official_config(dspark, dspark_config, field, value, error):
    path = Path(dspark_config.speculative_config.model) / "config.json"
    hf = json.loads(path.read_text())
    hf[field] = value
    path.write_text(json.dumps(hf))
    with pytest.raises(ValueError, match=error):
        dspark.validate_vq2a8_dspark_config(dspark_config)


def test_dspark_preflight_rejects_draft_runtime_layout_override(dspark, dspark_config):
    dspark_config.speculative_config.draft_model_config.hf_config.sample_from_anchor = False
    with pytest.raises(ValueError, match="anchor-as-first"):
        dspark.validate_vq2a8_dspark_config(dspark_config)


@pytest.fixture
def official_quantization_factory(monkeypatch):
    official_quant = SimpleNamespace(get_name=lambda: "deepseek_v4_fp8")
    quant_factory = Mock(return_value=official_quant)
    vllm = ModuleType("vllm")
    config_module = ModuleType("vllm.config")
    config_module.VllmConfig = SimpleNamespace(get_quantization_config=quant_factory)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.config", config_module)
    return quant_factory


def test_dspark_uses_official_quantization_without_mutating_target(
    dspark, dspark_config, official_quantization_factory
):
    runtime = dspark.prepare_vq2a8_dspark_config(dspark_config)
    draft = runtime.speculative_config.draft_model_config
    assert runtime is not dspark_config
    assert runtime.speculative_config is not dspark_config.speculative_config
    assert draft is not dspark_config.speculative_config.draft_model_config
    assert runtime.model_config is draft
    assert draft.quantization == "deepseek_v4_fp8"
    assert draft.model == dspark_config.speculative_config.model
    official_quantization_factory.assert_called_once_with(draft, dspark_config.load_config)
    assert runtime.quant_config is official_quantization_factory.return_value
    assert runtime.compilation_config is not dspark_config.compilation_config
    assert runtime.compilation_config.static_forward_context is dspark_config.compilation_config.static_forward_context
    assert runtime.compilation_config.cudagraph_capture_sizes == [5, 10, 20, 40, 80]
    assert runtime.compilation_config.max_cudagraph_capture_size == 80
    assert dspark_config.compilation_config.cudagraph_capture_sizes == [6, 12, 24, 48, 96]
    assert dspark_config.compilation_config.max_cudagraph_capture_size == 96
    runtime.compilation_config.static_forward_context["mtp.0"] = object()
    assert "mtp.0" in dspark_config.compilation_config.static_forward_context
    assert "vq2a8" not in runtime.additional_config
    assert runtime.additional_config["enable_cpu_binding"] is True
    assert draft.max_model_len == 262144
    assert draft.use_fp64_gumbel is True
    assert runtime.speculative_config.target_model_config is dspark_config.model_config
    assert dspark_config.model_config.quantization == "vq2a8"
    assert dspark_config.speculative_config.draft_model_config.quantization == "vq2a8"
    assert dspark_config.speculative_config.quantization is None
    assert dspark_config.quant_config.is_gs32
    assert "vq2a8" in dspark_config.additional_config


def test_original_w4a8_dspark_keeps_existing_config_identity(dspark):
    config = SimpleNamespace(model_config=SimpleNamespace(quantization="deepseek_v4_fp8"))
    assert dspark.prepare_vq2a8_dspark_config(config) is config


@pytest.mark.parametrize("enforce_eager", [False, True])
def test_eager_target_does_not_require_speculative_graph_buckets(
    dspark, dspark_config, official_quantization_factory, enforce_eager
):
    dspark_config.compilation_config.cudagraph_mode = SimpleNamespace(decode_mode=lambda: SimpleNamespace(name="NONE"))
    dspark_config.compilation_config.cudagraph_capture_sizes = [1, 2, 4, 8, 16]
    dspark_config.compilation_config.max_cudagraph_capture_size = 16
    dspark_config.speculative_config.enforce_eager = enforce_eager
    runtime = dspark.prepare_vq2a8_dspark_config(dspark_config)
    assert runtime.compilation_config.cudagraph_capture_sizes == [1, 2, 4, 8, 16]
    assert runtime.compilation_config.max_cudagraph_capture_size == 16
    assert runtime.compilation_config.cudagraph_mode.decode_mode().name == "NONE"


def test_graph_target_rejects_unaligned_token_buckets(dspark, dspark_config, official_quantization_factory):
    dspark_config.compilation_config.cudagraph_capture_sizes = [1, 2, 4, 8, 16]
    with pytest.raises(ValueError, match="multiples of six"):
        dspark.prepare_vq2a8_dspark_config(dspark_config)


def test_dspark_rejects_runtime_block_size_override(dspark, dspark_config):
    dspark_config.speculative_config.draft_model_config.hf_config.dspark_block_size = 6
    with pytest.raises(ValueError, match="block_size=5"):
        dspark.validate_vq2a8_dspark_config(dspark_config)


@pytest.mark.parametrize("filename", ["config.json", "model.safetensors.index.json"])
@pytest.mark.parametrize("value", [None, []])
def test_dspark_rejects_nonobject_metadata(dspark, dspark_config, filename, value):
    path = Path(dspark_config.speculative_config.model) / filename
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="JSON objects"):
        dspark.validate_vq2a8_dspark_config(dspark_config)


@pytest.mark.parametrize("value", [None, [], 1, "", "../outside.safetensors"])
def test_dspark_rejects_invalid_shard_paths(dspark, dspark_config, value):
    path = Path(dspark_config.speculative_config.model) / "model.safetensors.index.json"
    index = json.loads(path.read_text(encoding="utf-8"))
    index["weight_map"]["embed.weight"] = value
    path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ValueError, match="shard"):
        dspark.validate_vq2a8_dspark_config(dspark_config)
