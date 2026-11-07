# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in full decode graph configuration boundaries."""
# ruff: noqa: F811  # Imported pytest fixture is injected by parameter name.

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.ut.quantization.test_vq2a8_gs32_fixtures import modules, serving_config  # noqa: F401


def graph_config():
    config = serving_config()
    config.additional_config = {
        "vq2a8": {
            "gs32_library": str(Path(__file__).resolve().parent / "stage.so"),
            "gs32_binary": str(Path(__file__).resolve().parent / "kernel.o"),
            "gs32_execution": "full_graph",
            "gs32_backend": "native",
            "gs32_fusion": "full_ffn",
        },
        "ascend_compilation_config": {"enable_npugraph_ex": False},
        "multistream_dsv4_dsa_overlap": False,
    }
    config.model_config.enforce_eager = False
    config.model_config.max_model_len = 128
    config.model_config.hf_config.num_experts_per_tok = 6
    config.scheduler_config.max_num_seqs = 1
    config.scheduler_config.max_num_batched_tokens = 128
    compilation = config.compilation_config
    compilation.cudagraph_mode = "FULL_DECODE_ONLY"
    compilation.cudagraph_capture_sizes = [1]
    compilation.max_cudagraph_capture_size = 1
    compilation.cudagraph_num_of_warmups = 1
    return config


@pytest.mark.parametrize("fusion", ["full_ffn"])
@pytest.mark.parametrize("use_v2", [False, True])
def test_gs32_full_graph_requires_explicit_decode_graph_scope(modules, fusion, use_v2):
    config = graph_config()
    config.use_v2_model_runner = use_v2
    config.additional_config["vq2a8"]["gs32_fusion"] = fusion
    assert modules.config.validate_vq2a8_config(config) == config.additional_config["vq2a8"]


@pytest.mark.parametrize(
    "fault",
    [
        "eager",
        "full_prefill",
        "piecewise",
        "sizes",
        "max_size",
        "batch",
        "warmup",
        "enpu",
        "overlap",
        "topk",
        "context",
        "async",
        "tp",
        "fused",
    ],
)
@pytest.mark.parametrize("use_v2", [False, True])
def test_gs32_graph_rejects_unvalidated_execution_before_weight_allocation(modules, fault, use_v2):
    config = graph_config()
    config.use_v2_model_runner = use_v2
    compilation = config.compilation_config
    if fault == "eager":
        config.model_config.enforce_eager = True
    elif fault == "full_prefill":
        compilation.cudagraph_mode = "FULL"
    elif fault == "piecewise":
        compilation.cudagraph_mode = "PIECEWISE"
    elif fault == "sizes":
        compilation.cudagraph_capture_sizes = [1, 2]
    elif fault == "max_size":
        compilation.max_cudagraph_capture_size = 2
    elif fault == "batch":
        config.scheduler_config.max_num_seqs = 2
    elif fault == "warmup":
        compilation.cudagraph_num_of_warmups = 0
    elif fault == "enpu":
        config.additional_config["ascend_compilation_config"]["enable_npugraph_ex"] = True
    elif fault == "overlap":
        config.additional_config["multistream_dsv4_dsa_overlap"] = True
    elif fault == "topk":
        config.model_config.hf_config.num_experts_per_tok = 9
    elif fault == "context":
        config.model_config.max_model_len = 129
    elif fault == "async":
        config.scheduler_config.async_scheduling = True
    elif fault == "tp":
        config.parallel_config.tensor_parallel_size = 2
    else:
        config.additional_config["vq2a8"]["gs32_execution"] = "fused"
    with pytest.raises(ValueError):
        modules.config.validate_vq2a8_config(config)


def long_graph_config():
    config = graph_config()
    config.model_config.hf_config.model_type = "deepseek_v4"
    config.model_config.max_model_len = 17408
    config.scheduler_config.enable_chunked_prefill = True
    config.scheduler_config.max_num_batched_tokens = 128
    config.cache_config.enable_prefix_caching = False
    return config


@pytest.mark.parametrize("context", [129, 17408, 263168, 1049600])
@pytest.mark.parametrize("budget", [1, 128, 129, 4096, 8192, 98304])
@pytest.mark.parametrize("use_v2", [False, True])
def test_gs32_long_context_keeps_decode_capture_shape_bounded(modules, context, budget, use_v2):
    config = long_graph_config()
    config.use_v2_model_runner = use_v2
    config.model_config.max_model_len = context
    config.scheduler_config.max_num_batched_tokens = budget
    assert modules.config.validate_vq2a8_config(config) == config.additional_config["vq2a8"]
    assert config.compilation_config.cudagraph_capture_sizes == [1]


@pytest.mark.parametrize("fault", ["unchunked", "missing_budget", "model", "nonpositive_context"])
def test_gs32_long_context_rejects_unsupported_prefill_dispatch(modules, fault):
    config = long_graph_config()
    if fault == "unchunked":
        config.scheduler_config.enable_chunked_prefill = False
    elif fault == "missing_budget":
        del config.scheduler_config.max_num_batched_tokens
    elif fault == "model":
        config.model_config.hf_config.model_type = "deepseek_v3"
    else:
        config.model_config.max_model_len = 0
    with pytest.raises(ValueError):
        modules.config.validate_vq2a8_config(config)


@pytest.mark.parametrize("budget", [0, -1, None, True, False, 4096.0, "4096"])
def test_gs32_long_context_requires_positive_integer_prefill_budget(modules, budget):
    config = long_graph_config()
    config.scheduler_config.max_num_batched_tokens = budget
    with pytest.raises(ValueError, match="positive integer token budget"):
        modules.config.validate_vq2a8_config(config)


@pytest.mark.parametrize("batch", range(1, 65))
@pytest.mark.parametrize("top_k", [1, 6, 8])
def test_mrv2_accepts_every_batch_through_64_with_covering_graph_buckets(modules, batch, top_k):
    config = graph_config()
    config.use_v2_model_runner = True
    config.scheduler_config.max_num_seqs = batch
    config.scheduler_config.max_num_batched_tokens = batch
    config.model_config.hf_config.num_experts_per_tok = top_k
    config.compilation_config.cudagraph_capture_sizes = [1] if batch == 1 else [1, batch]
    config.compilation_config.max_cudagraph_capture_size = batch
    assert modules.config.validate_vq2a8_config(config) == config.additional_config["vq2a8"]


@pytest.mark.parametrize("batch", [2, 8, 64])
def test_mrv1_rejects_multisequence_decode_graphs_before_loading(modules, batch):
    config = graph_config()
    config.use_v2_model_runner = False
    config.scheduler_config.max_num_seqs = batch
    config.compilation_config.cudagraph_capture_sizes = [1, batch]
    config.compilation_config.max_cudagraph_capture_size = batch
    with pytest.raises(ValueError, match="requires MRV2"):
        modules.config.validate_vq2a8_config(config)


@pytest.mark.parametrize("batch", [0, -1, 65, 256, True, None, 8.0, "8"])
def test_graph_batch_bound_is_checked_before_loading(modules, batch):
    config = graph_config()
    config.use_v2_model_runner = True
    config.scheduler_config.max_num_seqs = batch
    with pytest.raises(ValueError, match="max_num_seqs in"):
        modules.config.validate_vq2a8_config(config)


@pytest.mark.parametrize("sizes", [None, [], [0, 8], [1, 4], [1, 16], [1, 8, 8], [8, 1], [1, True, 8], [1, 8.0]])
def test_graph_buckets_must_be_valid_and_cover_scheduler_capacity(modules, sizes):
    config = graph_config()
    config.use_v2_model_runner = True
    config.scheduler_config.max_num_seqs = 8
    config.compilation_config.max_cudagraph_capture_size = 8
    config.compilation_config.cudagraph_capture_sizes = sizes
    with pytest.raises(ValueError, match="capture sizes"):
        modules.config.validate_vq2a8_config(config)


def test_decode_token_budget_must_hold_a_full_batch(modules):
    config = graph_config()
    config.use_v2_model_runner = True
    config.scheduler_config.max_num_seqs = 64
    config.scheduler_config.max_num_batched_tokens = 63
    config.compilation_config.cudagraph_capture_sizes = [1, 8, 64]
    config.compilation_config.max_cudagraph_capture_size = 64
    with pytest.raises(ValueError, match="max_num_batched_tokens >= max_num_seqs"):
        modules.config.validate_vq2a8_config(config)


@pytest.mark.parametrize("batch,top_k", [(1, 6), (7, 6), (64, 6), (64, 8)])
@pytest.mark.parametrize("tp_size,dp_size", [(1, 1), (2, 1), (4, 1), (8, 1), (1, 2), (1, 4), (1, 8)])
@pytest.mark.parametrize("query_width", [1, 6])
def test_graph_capacity_reaches_scheme_without_expert_parallel_division(
    modules, monkeypatch, batch, top_k, tp_size, dp_size, query_width
):
    config = graph_config()
    config.use_v2_model_runner = True
    config.scheduler_config.max_num_seqs = batch
    config.scheduler_config.max_num_batched_tokens = batch * query_width
    config.compilation_config.cudagraph_capture_sizes = (
        [query_width] if batch == 1 else [query_width, batch * query_width]
    )
    config.compilation_config.max_cudagraph_capture_size = batch * query_width
    if query_width > 1:
        config.speculative_config = SimpleNamespace(num_speculative_tokens=query_width - 1)
        # Draft asset/config validation is covered by the DSpark contract tests;
        # this regression isolates target graph and MoE capacity propagation.
        monkeypatch.setattr(modules.config, "validate_vq2a8_dspark_config", Mock())
    config.model_config.hf_config.num_experts_per_tok = top_k
    config.model_config.hf_config.model_type = "deepseek_v4"
    config.model_config.hf_config.n_routed_experts = 256
    config.parallel_config.tensor_parallel_size = tp_size
    config.parallel_config.data_parallel_size = dp_size
    config.parallel_config.enable_expert_parallel = tp_size * dp_size > 1
    modules.current.get_current_vllm_config.return_value = config
    quant = modules.config.AscendVQ2A8Config(
        {"quant_method": "vq2a8", "format": "vq2_mixed_gs32_source", "format_version": 3}
    )
    quant.validate_model_config = Mock()
    quant.layer_spec = Mock()
    scheme = Mock()
    monkeypatch.setattr(modules.config, "get_scheme_class", Mock(return_value=scheme))
    layer = SimpleNamespace(moe_config=SimpleNamespace())
    quant.get_quant_method(layer, "model.layers.0.mlp.experts")
    assert scheme.call_args.kwargs["graph_row_capacity"] == dp_size * batch * query_width * top_k
    assert scheme.call_args.kwargs["native_options"] == config.additional_config["vq2a8"]
    assert "graph_row_capacity" not in config.additional_config["vq2a8"]


def dspark_graph_config(modules, monkeypatch):
    config = graph_config()
    config.use_v2_model_runner = True
    config.scheduler_config.max_num_seqs = 16
    config.scheduler_config.max_num_batched_tokens = 96
    config.speculative_config = SimpleNamespace(num_speculative_tokens=5)
    config.compilation_config.cudagraph_capture_sizes = [6, 12, 24, 48, 96]
    config.compilation_config.max_cudagraph_capture_size = 96
    monkeypatch.setattr(modules.config, "validate_vq2a8_dspark_config", Mock())
    return config


def test_dspark_target_graph_uses_six_tokens_per_sequence(modules, monkeypatch):
    config = dspark_graph_config(modules, monkeypatch)
    assert modules.config.validate_vq2a8_config(config) == config.additional_config["vq2a8"]
    modules.config.validate_vq2a8_dspark_config.assert_called_once_with(config)


@pytest.mark.parametrize("sizes", [[1, 2, 4, 8, 16], [6, 12, 24, 48], [1, 6, 96], [6, 95, 96], [6, 96, 102]])
def test_dspark_rejects_sequence_buckets_or_partial_query_padding(modules, monkeypatch, sizes):
    config = dspark_graph_config(modules, monkeypatch)
    config.compilation_config.cudagraph_capture_sizes = sizes
    with pytest.raises(ValueError, match="capture sizes"):
        modules.config.validate_vq2a8_config(config)


def test_dspark_token_budget_must_cover_target_verification_batch(modules, monkeypatch):
    config = dspark_graph_config(modules, monkeypatch)
    config.scheduler_config.max_num_batched_tokens = 95
    with pytest.raises(ValueError, match="max_num_batched_tokens"):
        modules.config.validate_vq2a8_config(config)


@pytest.mark.parametrize("field", ["lora_config", "kv_transfer_config"])
def test_dspark_does_not_enable_unrelated_features(modules, monkeypatch, field):
    config = dspark_graph_config(modules, monkeypatch)
    setattr(config, field, SimpleNamespace())
    with pytest.raises(ValueError, match="LoRA or KV transfer"):
        modules.config.validate_vq2a8_config(config)
