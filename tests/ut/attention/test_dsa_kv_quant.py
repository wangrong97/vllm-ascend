# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for adapting local query heads to the A5 FP8 attention ABI."""

import ast
import importlib.util
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def adapter():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/attention/dsa_kv_quant.py"
    spec = importlib.util.spec_from_file_location("dsa_kv_quant_cpu_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def shared_kv_reference(q, kv, sinks):
    """Independent dense attention reference with a zero-valued sink token."""
    logits = torch.einsum("...hd,kd->...hk", q, kv) / q.shape[-1] ** 0.5
    if sinks is None:
        denominator = torch.logsumexp(logits, dim=-1, keepdim=True)
    else:
        sink_logits = sinks.expand(*logits.shape[:-1]).unsqueeze(-1)
        denominator = torch.logsumexp(torch.cat((logits, sink_logits), dim=-1), dim=-1, keepdim=True)
    probabilities = torch.exp(logits - denominator)
    return torch.einsum("...hk,kd->...hd", probabilities, kv), denominator


def query_inputs(tokens, heads, layout):
    shape = (tokens, heads, 8) if layout == "TND" else (2, tokens, heads, 8)
    return torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float64).sin().reshape(shape)


def install_native(monkeypatch, *, attention=None, metadata=None):
    attention = Mock() if attention is None else attention
    metadata = Mock() if metadata is None else metadata
    monkeypatch.setattr(torch.ops._C_ascend, "npu_kv_quant_sparse_attn_sharedkv", attention, raising=False)
    monkeypatch.setattr(torch.ops._C_ascend, "npu_kv_quant_sparse_attn_sharedkv_metadata", metadata, raising=False)
    return attention, metadata


@pytest.mark.parametrize("ep_size", [2, 4, 8])
@pytest.mark.parametrize("last_rank", [False, True])
@pytest.mark.parametrize("tokens", [1, 32], ids=["decode", "prefill"])
@pytest.mark.parametrize("layout", ["TND", "BSND"])
@pytest.mark.parametrize("return_lse", [False, True])
def test_padded_ep_attention_matches_global_shared_kv(
    adapter, monkeypatch, ep_size, last_rank, tokens, layout, return_lse
):
    global_heads = 64
    global_q = query_inputs(tokens, global_heads, layout)
    global_sinks = torch.linspace(-3.0, 2.0, global_heads, dtype=global_q.dtype)
    kv = torch.arange(7 * global_q.shape[-1], dtype=global_q.dtype).cos().reshape(7, -1)
    expected, expected_lse = shared_kv_reference(global_q, kv, global_sinks)
    local_heads = global_heads // ep_size
    rank = ep_size - 1 if last_rank else 0
    start = rank * local_heads
    q = global_q[..., start : start + local_heads, :].contiguous()
    sinks = global_sinks[start : start + local_heads].clone()
    original_q, original_sinks = q.clone(), sinks.clone()
    block_table = torch.tensor([[0, 1]], dtype=torch.int32)
    metadata = torch.tensor([17], dtype=torch.int32)
    empty_lse = torch.empty(0)
    native_calls = []

    def native(padded_q, **kwargs):
        assert padded_q.shape[-2] == global_heads
        assert padded_q.dtype == q.dtype and padded_q.device == q.device
        assert padded_q.is_contiguous()
        torch.testing.assert_close(padded_q[..., :local_heads, :], original_q)
        assert torch.count_nonzero(padded_q[..., local_heads:, :]) == 0
        assert kwargs["sinks"].shape == (global_heads,)
        torch.testing.assert_close(kwargs["sinks"][:local_heads], original_sinks)
        assert kwargs["sinks"].dtype == sinks.dtype and kwargs["sinks"].device == sinks.device
        assert kwargs["ori_kv"] is kv and kwargs["ori_block_table"] is block_table
        assert kwargs["metadata"] is metadata
        assert kwargs["layout_q"] == layout
        assert kwargs["softmax_scale"] == q.shape[-1] ** -0.5
        assert kwargs["ori_mask_mode"] == 4 and kwargs["ori_win_left"] == 127
        assert kwargs["return_softmax_lse"] is return_lse
        output, lse = shared_kv_reference(padded_q, kwargs["ori_kv"], kwargs["sinks"])
        native_calls.append(padded_q)
        return output, lse if return_lse else empty_lse

    install_native(monkeypatch, attention=native)
    output, lse = adapter.kv_quant_sparse_attn(
        q,
        sinks=sinks,
        ori_kv=kv,
        ori_block_table=block_table,
        metadata=metadata,
        layout_q=layout,
        softmax_scale=q.shape[-1] ** -0.5,
        ori_mask_mode=4,
        ori_win_left=127,
        return_softmax_lse=return_lse,
    )
    assert len(native_calls) == 1
    assert output.shape == q.shape and output.is_contiguous()
    torch.testing.assert_close(output, expected[..., start : start + local_heads, :])
    if return_lse:
        assert lse.shape == (*q.shape[:-1], 1)
        torch.testing.assert_close(lse, expected_lse[..., start : start + local_heads, :])
    else:
        assert lse is empty_lse
    torch.testing.assert_close(q, original_q)
    torch.testing.assert_close(sinks, original_sinks)


@pytest.mark.parametrize("heads", [1, 8, 16, 32, 63])
@pytest.mark.parametrize("layout", ["TND", "BSND"])
def test_padding_without_sinks_matches_reference(adapter, monkeypatch, heads, layout):
    q = query_inputs(2, heads, layout)
    kv = torch.arange(24, dtype=q.dtype).reshape(3, 8) / 10

    def native(padded_q, **kwargs):
        assert kwargs.get("sinks") is None
        assert padded_q.shape[-2] == 64
        return shared_kv_reference(padded_q, kwargs["ori_kv"], None)

    install_native(monkeypatch, attention=native)
    output, lse = adapter.kv_quant_sparse_attn(q, ori_kv=kv, layout_q=layout, return_softmax_lse=True)
    expected, expected_lse = shared_kv_reference(q, kv, None)
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(lse, expected_lse)
    assert output.is_contiguous()


@pytest.mark.parametrize("heads", [64, 65, 128, 256])
def test_native_head_counts_are_forwarded_without_copies(adapter, monkeypatch, heads):
    q = query_inputs(1, heads, "TND")
    sinks = torch.linspace(-1, 1, heads)
    native_output = torch.ones_like(q)
    native_lse = torch.empty(0)
    native = Mock(return_value=(native_output, native_lse))
    install_native(monkeypatch, attention=native)
    output, lse = adapter.kv_quant_sparse_attn(q, sinks=sinks, layout_q="TND")
    assert native.call_args.args[0] is q
    assert native.call_args.kwargs["sinks"] is sinks
    assert output is native_output and lse is native_lse


@pytest.mark.parametrize("heads,expected", [(1, 64), (8, 64), (16, 64), (32, 64), (63, 64), (64, 64), (128, 128)])
def test_metadata_and_attention_use_same_head_count(adapter, monkeypatch, heads, expected):
    metadata_output = torch.tensor([123], dtype=torch.int32)
    sequence_lengths = torch.tensor([1], dtype=torch.int32)
    metadata_native = Mock(return_value=metadata_output)

    def native(q, **kwargs):
        assert q.shape[-2] == expected
        assert kwargs["metadata"] is metadata_output
        return torch.zeros_like(q), torch.empty(0)

    install_native(monkeypatch, attention=native, metadata=metadata_native)
    metadata = adapter.kv_quant_sparse_attn_metadata(
        num_heads_q=heads, num_heads_kv=1, kv_quant_mode=1, seqused_kv=sequence_lengths, layout_q="TND"
    )
    assert metadata is metadata_output
    kwargs = metadata_native.call_args.kwargs
    assert kwargs["num_heads_q"] == expected and kwargs["num_heads_kv"] == 1
    assert kwargs["kv_quant_mode"] == 1 and kwargs["seqused_kv"] is sequence_lengths
    assert kwargs["layout_q"] == "TND"
    adapter.kv_quant_sparse_attn(query_inputs(1, heads, "TND"), metadata=metadata, layout_q="TND")


@pytest.mark.parametrize("heads", [65, 256])
def test_metadata_does_not_expand_unrelated_native_head_counts(adapter, monkeypatch, heads):
    _, native = install_native(monkeypatch)
    adapter.kv_quant_sparse_attn_metadata(num_heads_q=heads)
    assert native.call_args.kwargs["num_heads_q"] == heads


@pytest.mark.parametrize("heads", [0, -1])
def test_metadata_rejects_nonpositive_heads(adapter, monkeypatch, heads):
    _, native = install_native(monkeypatch)
    with pytest.raises(ValueError):
        adapter.kv_quant_sparse_attn_metadata(num_heads_q=heads)
    native.assert_not_called()


@pytest.mark.parametrize(
    "shape,layout,sink_shape",
    [
        ((1, 0, 8), "TND", None),
        ((1, 32, 8), "BAD", None),
        ((1, 32, 8), "BSND", None),
        ((1, 1, 32, 8), "TND", None),
        ((32, 8), "TND", None),
        ((1, 32, 8), "TND", (31,)),
        ((1, 32, 8), "TND", (33,)),
        ((1, 32, 8), "TND", (1, 32)),
        ((1, 32, 8), "TND", ()),
    ],
)
def test_invalid_padding_inputs_fail_before_native(adapter, monkeypatch, shape, layout, sink_shape):
    native, _ = install_native(monkeypatch)
    sinks = None if sink_shape is None else torch.zeros(sink_shape)
    with pytest.raises(ValueError):
        adapter.kv_quant_sparse_attn(torch.zeros(shape), sinks=sinks, layout_q=layout)
    native.assert_not_called()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_noncontiguous_queries_preserve_dtype_and_values(adapter, monkeypatch, dtype):
    q = torch.arange(3 * 32 * 16, dtype=torch.float32).to(dtype).reshape(3, 32, 16)[..., ::2]
    sinks = torch.arange(64, dtype=dtype)[::2]
    assert not q.is_contiguous() and not sinks.is_contiguous()
    original_q, original_sinks = q.clone(), sinks.clone()

    def native(padded_q, **kwargs):
        assert padded_q.dtype == dtype and padded_q.is_contiguous()
        assert kwargs["sinks"].dtype == dtype and kwargs["sinks"].is_contiguous()
        torch.testing.assert_close(padded_q[..., :32, :], original_q)
        torch.testing.assert_close(kwargs["sinks"][:32], original_sinks)
        return padded_q.clone(), torch.empty(0)

    install_native(monkeypatch, attention=native)
    output, _ = adapter.kv_quant_sparse_attn(q, sinks=sinks, layout_q="TND")
    assert output.dtype == dtype and output.is_contiguous()
    torch.testing.assert_close(output, original_q)
    torch.testing.assert_close(q, original_q)
    torch.testing.assert_close(sinks, original_sinks)


@pytest.mark.parametrize(
    "compressed_cache,cache_dtype,expected",
    [
        (True, "auto", "fp8"),
        (True, "float8_e4m3fn", "fp8"),
        (True, "bfloat16", "bf16"),
        (True, "bf16", "bf16"),
        (False, "auto", "legacy"),
        (False, "bfloat16", "legacy"),
    ],
)
def test_actual_kv_plan_selects_paired_ops(adapter, monkeypatch, compressed_cache, cache_dtype, expected):
    # Execute the real plan and cache-dtype selection while excluding imports
    # that initialize vLLM/NPU. Only hardware capability and native ops are fake.
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/attention/dsa_attn_kv_plan.py"
    tree = ast.parse(path.read_text())
    names = {"DsaAttnKvPlan", "is_a5_bf16_kv_enabled", "get_dsa_attn_kv_plan"}
    constants = {
        "_BF16_KV_CACHE_DTYPES",
        "DSA_COMPRESSOR_SLOT_MAPPING_FLAT",
        "DSA_COMPRESSOR_SLOT_MAPPING_BLOCK_OFFSET",
    }
    nodes = [
        node
        for node in tree.body
        if (isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names)
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id in constants for target in node.targets)
        )
    ]
    legacy_attention, legacy_metadata = Mock(), Mock()
    monkeypatch.setattr(torch.ops._C_ascend, "npu_sparse_attn_sharedkv", legacy_attention, raising=False)
    monkeypatch.setattr(torch.ops._C_ascend, "npu_sparse_attn_sharedkv_metadata", legacy_metadata, raising=False)
    bf16_attention, bf16_metadata = Mock(), Mock()
    scope = {
        "__name__": __name__,
        "torch": torch,
        "dataclass": dataclass,
        "Callable": Callable,
        "Any": Any,
        "_supports_dsv4_compressed_cache": lambda: compressed_cache,
        "kv_quant_sparse_attn": adapter.kv_quant_sparse_attn,
        "kv_quant_sparse_attn_metadata": adapter.kv_quant_sparse_attn_metadata,
        "sparse_flash_mla": bf16_attention,
        "sparse_flash_mla_metadata": bf16_metadata,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    plan = scope["get_dsa_attn_kv_plan"](SimpleNamespace(cache_config=SimpleNamespace(cache_dtype=cache_dtype)))
    expected_attention, expected_metadata = {
        "fp8": (adapter.kv_quant_sparse_attn, adapter.kv_quant_sparse_attn_metadata),
        "bf16": (bf16_attention, bf16_metadata),
        "legacy": (legacy_attention, legacy_metadata),
    }[expected]
    assert plan.get_dsa_sparse_attn_op() is expected_attention
    assert plan.get_dsa_sparse_attn_metadata_op() is expected_metadata
    if expected == "fp8":
        assert plan.get_dsa_sparse_attn_base_kwargs() == {"kv_quant_mode": 1, "tile_size": 64, "rope_head_dim": 64}
        assert plan.get_dsa_sparse_attn_metadata_kwargs("npu:0") == {"kv_quant_mode": 1}
    else:
        assert plan.get_dsa_sparse_attn_base_kwargs() == {}
        assert plan.get_dsa_sparse_attn_metadata_kwargs("npu:0") == {"device": "npu:0"}
