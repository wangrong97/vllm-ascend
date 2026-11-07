# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check DSV4 cache allocation against its attention plan without an NPU."""

import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


def _source_tree(relative_path):
    path = Path(__file__).resolve().parents[3] / relative_path
    return path, ast.parse(path.read_text(encoding="utf-8"))


def _execute_nodes(path, nodes, scope):
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    tree = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(tree, str(path), "exec"), scope)


class _UpstreamSWACache:
    def __init__(self, head_dim, window_size, dtype, prefix, cache_config):
        self.head_dim = head_dim
        self.window_size = window_size
        self.dtype = dtype
        self.prefix = prefix
        self.cache_config = cache_config


@pytest.fixture
def cache_contract(monkeypatch):
    # Boundary stand-ins model dtype identity, hardware capability and spec
    # records. All dtype selection, shapes and plan selection run real source.
    torch = NS(bfloat16="bf16", float16="fp16", float8_e4m3fn="fp8", uint8="uint8")
    torch.ops = NS(
        _C_ascend=NS(npu_sparse_attn_sharedkv="sparse_non_a5", npu_sparse_attn_sharedkv_metadata="meta_non_a5")
    )
    hardware = NS(a5=True)

    def generic_dtype(cache_dtype, model_config):
        if cache_dtype == "auto":
            return model_config.dtype
        return {
            "fp8": torch.float8_e4m3fn,
            "fp8_e4m3": torch.float8_e4m3fn,
            "float8_e4m3fn": torch.float8_e4m3fn,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "turboquant_4bit_nc": torch.uint8,
        }[cache_dtype]

    scope = {
        "__name__": __name__,
        "torch": torch,
        "dataclass": dataclass,
        "VllmDeepseekV4SWACache": _UpstreamSWACache,
        "AscendSlidingWindowMLASpec": NS,
        "AscendMLAAttentionSpec": NS,
        "HardwareCapability": NS(DSV4_COMPRESSED_CACHE="compressed"),
        "get_current_hardware_profile": lambda: NS(supports=lambda _: hardware.a5),
        "kv_cache_dtype_str_to_dtype": generic_dtype,
        "is_turboquant": lambda config: config.cache_config.cache_dtype == "turboquant_4bit_nc",
        "sparse_flash_mla": "sparse_bf16",
        "sparse_flash_mla_metadata": "meta_bf16",
        "mixed_quant_sparse_flash_mla_fp8": "mixed_fp8",
        "mixed_quant_sparse_flash_mla_metadata_fp8": "meta_fp8",
        "mixed_quant_sparse_flash_mla_turboquant": "mixed_turboquant",
        "mixed_quant_sparse_flash_mla_metadata_turboquant": "meta_turboquant",
    }
    plan_path, plan_tree = _source_tree("vllm_ascend/attention/dsa_attn_kv_plan.py")
    plan_names = {
        "_supports_dsv4_compressed_cache",
        "is_a5_bf16_kv_enabled",
        "get_dsv4_attn_kv_dtype",
        "get_dsa_attn_kv_plan",
        "DsaAttnKvPlan",
    }
    _execute_nodes(
        plan_path,
        [n for n in plan_tree.body if isinstance(n, ast.Assign) or getattr(n, "name", None) in plan_names],
        scope,
    )
    tq_path, tq_tree = _source_tree("vllm_ascend/quantization/methods/kv_cache/turboquant/__init__.py")
    _execute_nodes(tq_path, [n for n in tq_tree.body if isinstance(n, ast.Assign)], scope)

    layer_path, layer_tree = _source_tree("vllm_ascend/models/layer/attention/layer.py")
    _execute_nodes(
        layer_path,
        [
            n
            for n in layer_tree.body
            if isinstance(n, ast.Assign) or getattr(n, "name", None) in {"get_dsv4_block_sizes", "dsv4_block_sizes"}
        ],
        scope,
    )
    # The real SWA constructor imports the block table lazily.
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.models.layer.attention.layer",
        NS(DSV4_BLOCK_SIZES=scope["DSV4_BLOCK_SIZES"]),
    )
    attention = next(n for n in layer_tree.body if isinstance(n, ast.ClassDef) and n.name == "DSAAttention")
    compressed_spec = next(
        n for n in attention.body if isinstance(n, ast.FunctionDef) and n.name == "get_kv_cache_spec"
    )
    _execute_nodes(layer_path, [compressed_spec], scope)
    scope["compressed_spec"] = scope.pop("get_kv_cache_spec")

    model_path, model_tree = _source_tree("vllm_ascend/models/deepseek_v4/model.py")
    swa = next(n for n in model_tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendDeepseekV4SWACache")
    _execute_nodes(model_path, [swa], scope)
    attention = next(n for n in model_tree.body if isinstance(n, ast.ClassDef) and n.name == "DeepseekV4Attention")
    constructor = next(n for n in attention.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    end = next(
        i
        for i, n in enumerate(constructor.body)
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "swa_cache_layer" for t in n.targets)
    )
    dtype_name = next(k.value.id for k in constructor.body[end].value.keywords if k.arg == "dtype")
    start = next(
        i
        for i, n in enumerate(constructor.body[:end])
        if any(isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store) and t.id == dtype_name for t in ast.walk(n))
    )
    # Execute the allocation fragment from the actual attention constructor,
    # including its dtype resolution and TurboQuant override.
    build_swa = ast.parse("def build_swa(self, vllm_config, cache_config, prefix): pass").body[0]
    build_swa.body = [
        *constructor.body[start : end + 1],
        ast.Return(value=ast.Name(id="swa_cache_layer", ctx=ast.Load())),
    ]
    _execute_nodes(model_path, [build_swa], scope)

    def allocate(cache_dtype="auto", model_dtype="bf16", a5=True, ratio=4):
        hardware.a5 = a5
        config = NS(cache_config=NS(cache_dtype=cache_dtype, block_size=32), model_config=NS(dtype=model_dtype))
        owner = NS(head_dim=512, window_size=128)
        cache = scope["build_swa"](owner, config, config.cache_config, "model.layers.0.self_attn")
        swa_spec = cache.get_kv_cache_spec(config)
        compressed = scope["compressed_spec"](
            NS(compress_ratio=ratio, kv_cache_dtype=cache_dtype, head_size=512, n_local_heads=8), config
        )
        plan = scope["get_dsa_attn_kv_plan"](config, ratio)
        assert config.cache_config.cache_dtype == cache_dtype, "Spec construction must preserve the requested policy."
        return swa_spec, compressed, plan

    return NS(allocate=allocate, slot_bytes=scope["SLOT_BYTES"])


@pytest.mark.parametrize("cache_dtype", ["auto", "fp8", "fp8_e4m3"])
@pytest.mark.parametrize("ratio", [4, 128])
def test_a5_fp8_cache_specs_match_attention_plan(cache_contract, cache_dtype, ratio):
    swa, compressed, plan = cache_contract.allocate(cache_dtype, ratio=ratio)
    assert (swa.head_size, swa.dtype) == (608, "fp8")
    assert (compressed.head_size, compressed.dtype) == (608, "fp8")
    assert plan.sparse_attn_op == "mixed_fp8"


@pytest.mark.parametrize("cache_dtype", ["bfloat16", "bf16"])
@pytest.mark.parametrize("ratio", [4, 128])
def test_a5_explicit_bf16_cache_specs_match_attention_plan(cache_contract, cache_dtype, ratio):
    swa, compressed, plan = cache_contract.allocate(cache_dtype, ratio=ratio)
    assert (swa.head_size, swa.dtype) == (512, "bf16")
    assert (compressed.head_size, compressed.dtype) == (512, "bf16")
    assert plan.sparse_attn_op == "sparse_bf16"


@pytest.mark.parametrize(
    "cache_dtype,model_dtype,expected_dtype",
    [("auto", "bf16", "bf16"), ("auto", "fp16", "fp16"), ("bfloat16", "fp16", "bf16")],
)
def test_non_a5_preserves_generic_dtype_resolution(cache_contract, cache_dtype, model_dtype, expected_dtype):
    swa, compressed, plan = cache_contract.allocate(cache_dtype, model_dtype, a5=False)
    assert (swa.head_size, swa.dtype) == (512, expected_dtype)
    assert (compressed.head_size, compressed.dtype) == (512, expected_dtype)
    assert plan.sparse_attn_op == "sparse_non_a5"


@pytest.mark.parametrize("ratio", [4, 128])
def test_turboquant_keeps_its_cache_dtype_overrides(cache_contract, ratio):
    swa, compressed, plan = cache_contract.allocate("turboquant_4bit_nc", a5=False, ratio=ratio)
    assert (swa.head_size, swa.dtype) == (512, "bf16")
    if ratio == 4:
        assert (compressed.head_size, compressed.dtype) == (cache_contract.slot_bytes, "uint8")
        assert plan.sparse_attn_op == "mixed_turboquant"
    else:
        assert (compressed.head_size, compressed.dtype) == (512, "bf16")
