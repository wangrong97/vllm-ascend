# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise runner spec selection on CPU, without initializing vLLM or NPU."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest


@pytest.fixture
def spec_reader():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/model_runner_v1.py"
    runner = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner"
    )
    method = next(
        node for node in runner.body if isinstance(node, ast.FunctionDef) and node.name == "_get_layer_kv_cache_specs"
    )
    # Stand-ins only provide the upstream type identities used by isinstance.
    # The selection logic itself is compiled directly from the runner source.
    scope = {
        "KVCacheConfig": object,
        "KVCacheSpec": object,
        "UniformTypeKVCacheSpecs": type("UniformTypeKVCacheSpecs", (NS,), {}),
        "AscendSFAIndexerCacheSpec": type("AscendSFAIndexerCacheSpec", (NS,), {}),
        "AttentionLayerBase": type("AttentionLayerBase", (NS,), {}),
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
    return NS(**scope)


@pytest.mark.parametrize("block_size,state_block_size", [(128, 8), (64, 4), (32, 2)])
@pytest.mark.parametrize("runner_compilation_config", [True, False])
def test_compressed_cache_reuses_planned_specs_after_block_size_update(
    spec_reader, block_size, state_block_size, runner_compilation_config
):
    specs = {
        "indexer.k_cache": NS(block_size=block_size * 4, storage_block_size=block_size, scale_dim=1),
        "attn": NS(block_size=block_size * 4, storage_block_size=block_size, scale_dim=0),
        "compressor.state_cache": NS(block_size=state_block_size),
    }
    groups = [
        NS(layer_names=[name], kv_cache_spec=spec_reader.UniformTypeKVCacheSpecs(kv_cache_specs={name: spec}))
        for name, spec in specs.items()
    ]
    # EngineCore changes the shared config to min(group.block_size) before
    # the uni executor allocates caches; 8/4/2 are not DSV4 table keys.
    config = NS(cache_config=NS(block_size=state_block_size))

    def query_spec(vc):
        return {block_size: specs["indexer.k_cache"]}[vc.cache_config.block_size]

    layer = spec_reader.AttentionLayerBase(get_kv_cache_spec=Mock(side_effect=query_spec))
    compilation = NS(static_forward_context=dict.fromkeys(specs, layer))
    config.compilation_config = compilation
    owner = NS(vllm_config=config)
    if runner_compilation_config:
        owner.compilation_config = compilation

    actual = spec_reader._get_layer_kv_cache_specs(owner, NS(kv_cache_groups=groups))

    assert set(actual) == set(specs)
    for name, expected in specs.items():
        assert actual[name] is expected
    layer.get_kv_cache_spec.assert_not_called()
    assert config.cache_config.block_size == state_block_size


def test_uniform_group_preserves_per_layer_indexer_layout(spec_reader):
    main = NS(block_size=16, head_size=576)
    indexer = spec_reader.AscendSFAIndexerCacheSpec(block_size=16, head_size=128, scale_dim=1)
    specs = {"attn": main, "indexer": indexer}
    layer = spec_reader.AttentionLayerBase(
        get_kv_cache_spec=Mock(side_effect=AssertionError("must reuse planned spec"))
    )
    owner = NS(compilation_config=NS(static_forward_context=dict.fromkeys(specs, layer)), vllm_config=NS())
    group = NS(layer_names=list(specs), kv_cache_spec=spec_reader.UniformTypeKVCacheSpecs(kv_cache_specs=specs))

    actual = spec_reader._get_layer_kv_cache_specs(owner, NS(kv_cache_groups=[group]))

    assert actual["attn"] is main and actual["indexer"] is indexer
    layer.get_kv_cache_spec.assert_not_called()


@pytest.mark.parametrize("has_context", [True, False])
def test_nonuniform_group_keeps_sfa_indexer_recovery(spec_reader, has_context):
    group_spec = NS(block_size=16)
    indexer = spec_reader.AscendSFAIndexerCacheSpec(block_size=16, scale_dim=1)
    layer = spec_reader.AttentionLayerBase(get_kv_cache_spec=Mock(return_value=indexer))
    main_layer = spec_reader.AttentionLayerBase(get_kv_cache_spec=Mock(return_value=NS(block_size=32)))
    owner = NS(vllm_config=NS())
    if has_context:
        owner.compilation_config = NS(static_forward_context={"indexer": layer, "attn": main_layer})
    group = NS(layer_names=["attn", "indexer"], kv_cache_spec=group_spec)

    actual = spec_reader._get_layer_kv_cache_specs(owner, NS(kv_cache_groups=[group]))

    assert actual["attn"] is group_spec
    assert actual["indexer"] is (indexer if has_context else group_spec)
    if has_context:
        layer.get_kv_cache_spec.assert_called_once_with(owner.vllm_config)
    else:
        layer.get_kv_cache_spec.assert_not_called()
