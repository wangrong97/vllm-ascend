# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-owned startup deduplication; no native device operations are run."""
# ruff: noqa: F401, F811  # Imported pytest fixtures are injected by name.

import pickle
from copy import copy, deepcopy
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from tests.ut.quantization.test_vq2a8_gs32_dspark import dspark, dspark_config, official_quantization_factory
from tests.ut.quantization.test_vq2a8_gs32_fixtures import modules, serving_config
from vllm_ascend.quantization.methods.w2a8 import vq2a8_gs32_startup as startup


def owner_config(tmp_path):
    config = serving_config()
    config.additional_config["vq2a8"].update(
        gs32_backend="native",
        gs32_fusion="full_ffn",
        gs32_library=str(tmp_path / "bridge.so"),
        gs32_binary=str(tmp_path / "kernel.o"),
    )
    return config


def quant_config(modules):
    return modules.config.AscendVQ2A8Config(
        {"quant_method": "vq2a8", "format": "vq2_mixed_gs32_source", "format_version": 3}
    )


def validation_spies(modules, monkeypatch, options):
    runtime = Mock(return_value=dict(options))
    monkeypatch.setattr(modules.config, "validate_vq2a8_config", runtime)
    return runtime


def test_model_validation_runs_once_and_preserves_independent_returned_options(modules, monkeypatch, tmp_path):
    config = owner_config(tmp_path)
    runtime = validation_spies(modules, monkeypatch, config.additional_config["vq2a8"])
    quant = quant_config(modules)
    quant.validate_model_config = Mock()
    first = quant.validate_runtime_config(config)
    first["gs32_execution"] = "invalid"
    for _ in range(4):
        assert quant.validate_runtime_config(config) == config.additional_config["vq2a8"]
    runtime.assert_called_once_with(config)
    quant.validate_model_config.assert_called_once_with(config.model_config.hf_config)
    other = quant_config(modules)
    other.validate_model_config = Mock()
    other.validate_runtime_config(config)
    assert runtime.call_count == 2
    assert other._startup_state is not quant._startup_state


@pytest.mark.parametrize("stage", ["runtime", "inventory"])
def test_validation_failures_are_not_cached(modules, monkeypatch, tmp_path, stage):
    config = owner_config(tmp_path)
    runtime = validation_spies(modules, monkeypatch, config.additional_config["vq2a8"])
    quant = quant_config(modules)
    quant.validate_model_config = Mock()
    failing = runtime if stage == "runtime" else quant.validate_model_config
    failing.side_effect = ValueError("first validation failed")
    with pytest.raises(ValueError, match="first validation failed"):
        quant.validate_runtime_config(config)
    assert quant._startup_state is None
    failing.side_effect = None
    quant.validate_runtime_config(config)
    quant.validate_runtime_config(config)
    assert runtime.call_count == 2


@pytest.mark.parametrize(
    "change", ["dtype", "geometry", "limit", "budget", "nested_options", "graph_buckets", "draft", "new_config"]
)
def test_configuration_changes_cannot_reuse_successful_validation(modules, monkeypatch, tmp_path, change):
    config = owner_config(tmp_path)
    runtime = validation_spies(modules, monkeypatch, config.additional_config["vq2a8"])
    quant = quant_config(modules)
    quant.validate_model_config = Mock()
    quant.validate_runtime_config(config)
    if change == "dtype":
        config.model_config.dtype = "float16"
    elif change == "geometry":
        config.model_config.hf_config.hidden_size += 32
    elif change == "limit":
        config.model_config.hf_config.swiglu_limit = 9.0
    elif change == "budget":
        config.scheduler_config.max_num_batched_tokens = 4096
    elif change == "nested_options":
        config.additional_config["vq2a8"]["gs32_binary"] += ".other"
    elif change == "graph_buckets":
        config.compilation_config.cudagraph_capture_sizes = [1, 2, 4]
    elif change == "draft":
        config.speculative_config = NS(model="different-draft")
    else:
        config = deepcopy(config)
    with pytest.raises(ValueError, match="configuration changed"):
        quant.validate_runtime_config(config)
    assert runtime.call_count == 1


def test_runtime_graph_inventory_growth_does_not_invalidate_configuration(modules, monkeypatch, tmp_path):
    config = owner_config(tmp_path)
    runtime = validation_spies(modules, monkeypatch, config.additional_config["vq2a8"])
    quant = quant_config(modules)
    quant.validate_model_config = Mock()
    quant.validate_runtime_config(config)
    config.compilation_config.static_forward_context = {"layer.0": object()}
    config.cache_config.num_gpu_blocks = 1024
    config.parallel_config.rank = 3
    config.parallel_config.local_rank = 1
    config.model_config.hf_config._name_or_path = "runtime-copy"
    quant.validate_runtime_config(config)
    runtime.assert_called_once()


def test_each_moe_receives_same_startup_owner_but_creates_its_own_scheme(modules, monkeypatch, tmp_path):
    config = owner_config(tmp_path)
    modules.current.get_current_vllm_config.return_value = config
    runtime = validation_spies(modules, monkeypatch, config.additional_config["vq2a8"])
    quant = quant_config(modules)
    quant.validate_model_config = Mock()
    quant.layer_spec = Mock()
    scheme = Mock(side_effect=[object(), object()])
    monkeypatch.setattr(modules.config, "get_scheme_class", Mock(return_value=scheme))
    layer = NS(moe_config=NS())
    for index in (0, 1):
        quant.get_quant_method(layer, f"model.layers.{index}.mlp.experts")
    runtime.assert_called_once()
    assert scheme.call_count == 2
    assert all(call.kwargs["startup_state"] is quant._startup_state for call in scheme.call_args_list)


def test_dspark_preparation_reuses_model_validation_without_reopening_assets(
    modules, monkeypatch, tmp_path, dspark, dspark_config, official_quantization_factory
):
    options = owner_config(tmp_path).additional_config["vq2a8"]
    check_assets = Mock(wraps=dspark.validate_vq2a8_dspark_config)
    monkeypatch.setattr(dspark, "validate_vq2a8_dspark_config", check_assets)

    def validate(config):
        check_assets(config)
        return options

    monkeypatch.setattr(modules.config, "validate_vq2a8_config", validate)
    quant = quant_config(modules)
    quant.validate_model_config = Mock()
    dspark_config.quant_config = quant
    quant.validate_runtime_config(dspark_config)
    dspark_config.cache_config = NS(num_gpu_blocks=1024)
    dspark_config.parallel_config = NS(rank=3)
    dspark.prepare_vq2a8_dspark_config(dspark_config)
    check_assets.assert_called_once_with(dspark_config)
    dspark_config.speculative_config.draft_model_config.hf_config.sample_from_anchor = False
    with pytest.raises(ValueError, match="configuration changed"):
        dspark.prepare_vq2a8_dspark_config(dspark_config)


def test_new_worker_revalidates_before_creating_native_state(modules, monkeypatch, tmp_path):
    config = owner_config(tmp_path)
    runtime = validation_spies(modules, monkeypatch, config.additional_config["vq2a8"])
    quant = quant_config(modules)
    quant.validate_model_config = Mock()
    quant.validate_runtime_config(config)
    old = quant._startup_state
    monkeypatch.setattr(startup, "getpid", lambda: old.process_id + 1)
    options = config.additional_config["vq2a8"]
    with pytest.raises(RuntimeError, match="current worker"):
        old.native_ops(options["gs32_library"], options["gs32_binary"], "npu:0", execution="fused", factory=Mock())
    quant.validate_runtime_config(config)
    assert runtime.call_count == 2 and quant._startup_state is not old


def test_copy_and_serialization_drop_worker_native_objects(modules, tmp_path):
    config = owner_config(tmp_path)
    quant = quant_config(modules)
    quant._startup_state = startup.GS32StartupState(config, config.additional_config["vq2a8"])
    # A loaded dispatcher need not be serializable. Neither model copies nor
    # worker transport should try to copy it or inherit successful validation.
    quant._startup_state._native_ops["npu:0"] = lambda: None
    for restored in (copy(quant), deepcopy(quant), pickle.loads(pickle.dumps(quant))):
        assert restored._startup_state is None
        assert restored.source_config == quant.source_config
    assert quant._startup_state is not None


def test_native_initialization_is_model_device_scoped_and_failure_retryable(tmp_path):
    config = owner_config(tmp_path)
    options = config.additional_config["vq2a8"]
    state = startup.GS32StartupState(config, options)
    args = options["gs32_library"], options["gs32_binary"]
    factory = Mock(side_effect=[RuntimeError("registration failed"), object(), object(), object()])
    with pytest.raises(RuntimeError, match="registration failed"):
        state.native_ops(*args, "npu:0", execution="fused", factory=factory)
    first = state.native_ops(*args, "npu:0", execution="fused", factory=factory)
    assert state.native_ops(*args, "npu:0", execution="fused", factory=factory) is first
    assert state.native_ops(*args, "npu:1", execution="fused", factory=factory) is not first
    other = startup.GS32StartupState(config, options)
    assert other.native_ops(*args, "npu:0", execution="fused", factory=factory) is not first
    assert factory.call_count == 4


@pytest.mark.parametrize("fault", ["artifacts", "execution", "changed_config"])
def test_native_state_rejects_unvalidated_inputs(tmp_path, fault):
    config = owner_config(tmp_path)
    options = config.additional_config["vq2a8"]
    state = startup.GS32StartupState(config, options)
    library, binary, execution = options["gs32_library"], options["gs32_binary"], "fused"
    if fault == "artifacts":
        binary += ".other"
    elif fault == "execution":
        execution = "full_graph"
    else:
        config.model_config.hf_config.hidden_size += 32
    factory = Mock()
    with pytest.raises(ValueError, match="differ|configuration changed"):
        state.native_ops(library, binary, "npu:0", execution=execution, factory=factory)
    factory.assert_not_called()
