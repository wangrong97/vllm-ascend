# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in B1 full decode graph configuration boundaries."""
# ruff: noqa: F811  # Imported pytest fixture is injected by parameter name.

from pathlib import Path

import pytest

from tests.ut.quantization.test_vq2a8_gs32_fixtures import modules, serving_config  # noqa: F401


def graph_config():
    config = serving_config()
    config.additional_config = {
        "vq2a8": {
            "gs32_library": str(Path(__file__).resolve().parent / "stage.so"),
            "gs32_binary": str(Path(__file__).resolve().parent / "kernel.o"),
            "gs32_execution": "full_graph",
        },
        "ascend_compilation_config": {"enable_npugraph_ex": False},
        "multistream_dsv4_dsa_overlap": False,
    }
    config.model_config.enforce_eager = False
    config.model_config.max_model_len = 128
    config.model_config.hf_config.num_experts_per_tok = 6
    config.scheduler_config.max_num_seqs = 1
    compilation = config.compilation_config
    compilation.cudagraph_mode = "FULL_DECODE_ONLY"
    compilation.cudagraph_capture_sizes = [1]
    compilation.max_cudagraph_capture_size = 1
    compilation.cudagraph_num_of_warmups = 1
    return config


@pytest.mark.parametrize("fusion", ["none", "swiglu_quant"])
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
