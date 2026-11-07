# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the actual single-stream prolog/indexer with CPU quantization stand-ins.

These tests check reuse, fallback and tensor lifetime, not CANN numerics or
graph capture. E4M3 rounding and matrix operations use real CPU PyTorch.
"""

import __future__

import ast
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch


class MXFP8:
    def __init__(self, scale_alg=0, group_size=32):
        self.dynamic_mx_quant_scale_alg = scale_alg
        self.group_size = group_size


class INT8:
    pass


class CPUOps:
    float8_e8m0fnu = torch.float8_e8m0fnu

    def __init__(self):
        self.mx_quant_calls = 0
        self.int8_quant_calls = 0

    def npu_dynamic_mx_quant(self, value, *, dst_type, scale_alg):
        self.mx_quant_calls += 1
        grouped = value.float().reshape(value.shape[0], -1, 32)
        maximum = grouped.abs().amax(-1).clamp_min(1e-10)
        exponent = (maximum / 448).log2()
        exponent = exponent.ceil() if scale_alg == 0 else exponent.floor()
        scale = 2**exponent
        quantized = (grouped / scale[..., None]).clamp(-448, 448).to(dst_type).reshape(value.shape)
        return quantized, (exponent + 127).to(torch.uint8)

    def npu_dynamic_quant(self, value):
        self.int8_quant_calls += 1
        scale = value.float().abs().amax(-1).clamp_min(1e-10) / 127
        return (value.float() / scale[:, None]).round().to(torch.int8), scale

    @staticmethod
    def npu_quant_matmul(value, weight, weight_scale, *, pertoken_scale, bias, output_dtype, **kwargs):
        if value.dtype == torch.float8_e4m3fn:
            scale = 2 ** (pertoken_scale.float() - 127)
            restored = value.float() * scale.repeat_interleave(32, -1)
        else:
            restored = value.float() * pertoken_scale[:, None]
        output = restored @ weight.float()
        if bias is not None:
            output += bias.float()
        return output.to(output_dtype)


def load_methods(npu):
    repo = Path(__file__).resolve().parents[3]
    scope = {
        "torch": torch,
        "torch_npu": npu,
        "AscendW8A8DynamicLinearMethod": INT8,
        "AscendW8A8MXFP8DynamicLinearMethod": MXFP8,
        "MXFP8_INPUT_GROUP_SIZE": 32,
        "_is_w8a8_dynamic": lambda layer: isinstance(layer.quant_method, INT8),
        "DeviceOperator": SimpleNamespace(apply_dsa_q_rms=lambda q, *_: q),
        "rotate_activation": lambda q, _: q,
        "get_current_hardware_profile": lambda: SimpleNamespace(supports=lambda _: False),
        "HardwareCapability": SimpleNamespace(FP8_ATTENTION=1),
    }
    definitions = []
    for file, names in (
        ("vllm_ascend/ops/cv_linear.py", {"CVLinearWrapper"}),
        ("vllm_ascend/attention/dsa_v1.py", {"_mla_prolog_single_stream"}),
        ("vllm_ascend/models/deepseek_v4/indexer.py", {"_indexer_qkv_prepare"}),
    ):
        tree = ast.parse((repo / file).read_text(encoding="utf-8"))
        definitions.extend(
            node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
        )
    exec(
        compile(
            ast.Module(body=definitions, type_ignores=[]),
            "shared_mxfp8",
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        scope,
    )
    return scope


class Linear:
    def __init__(self, ops, method, width, output, *, seed, dtype=torch.bfloat16):
        self.ops = ops
        self.quant_method = method
        self.input_size = width
        self.gather_output = False
        self.skip_bias_add = False
        weights = torch.randn(width, output, generator=torch.Generator().manual_seed(seed)) / 16
        self.weight = weights.to(torch.float8_e4m3fn if isinstance(method, MXFP8) else dtype)
        self.weight_scale = torch.ones(1)
        self.bias = torch.linspace(-0.1, 0.1, output).to(dtype)

    def __call__(self, value):
        bias = None if self.skip_bias_add else self.bias
        if isinstance(self.quant_method, MXFP8):
            data, scale = self.ops.npu_dynamic_mx_quant(
                value, dst_type=torch.float8_e4m3fn, scale_alg=self.quant_method.dynamic_mx_quant_scale_alg
            )
        elif isinstance(self.quant_method, INT8):
            data, scale = self.ops.npu_dynamic_quant(value)
        else:
            output = value.float() @ self.weight.float()
            return (output if bias is None else output + bias.float()).to(value.dtype)
        return self.ops.npu_quant_matmul(
            data, self.weight, self.weight_scale, pertoken_scale=scale, bias=bias, output_dtype=value.dtype
        )

    forward = __call__


def norm(value):
    return (value.float() * (value.float().square().mean(-1, keepdim=True) + 1e-6).rsqrt()).to(value.dtype)


def model(scope, ops, *, dtype=torch.bfloat16, method=MXFP8):
    wrapper = scope["CVLinearWrapper"]
    qa = Linear(ops, method(), 64, 32, seed=1, dtype=dtype)
    kv = Linear(ops, method(), 64, 32, seed=2, dtype=dtype)
    qb = Linear(ops, method(), 32, 32, seed=3, dtype=dtype)
    iq = Linear(ops, method(), 32, 32, seed=4, dtype=dtype)
    indexer = SimpleNamespace(
        wq_b=iq,
        cv_wq_b=wrapper(iq),
        skip_topk=False,
        n_heads=2,
        head_dim=16,
        rope_head_dim=8,
        compressor=object(),
        ops=SimpleNamespace(unpack_dsa_indexer_kv_cache=lambda _: (None,) * 4),
        _get_indexer_cache_metadata=lambda _: (None, None),
    )
    owner = SimpleNamespace(
        wq_a=qa,
        wkv=kv,
        wq_b=qb,
        cv_wq_a=wrapper(qa),
        cv_wkv=wrapper(kv),
        cv_wq_b=wrapper(qb),
        q_norm=norm,
        kv_norm=norm,
        indexer=indexer,
        compress_ratio=4,
        eps=1e-6,
        q_norm_without_weight=None,
        n_local_heads=2,
        head_dim=16,
        nope_head_dim=16,
        rope_head_dim=16,
    )
    owner._write_kv_cache = lambda _, value, *args, **kwargs: setattr(owner, "written_kv", value.clone())
    owner.prolog = MethodType(scope["_mla_prolog_single_stream"], owner)
    indexer.prepare = MethodType(scope["_indexer_qkv_prepare"], indexer)
    return owner


@pytest.fixture
def scope(monkeypatch):
    ops = CPUOps()
    monkeypatch.setattr(torch.ops._C_ascend, "inplace_partial_rotary_mul", lambda *a, **k: None, raising=False)
    return load_methods(ops), ops


def run(owner, inputs, *, write_cache):
    q, qr, scale = owner.prolog(inputs, None, None, None, None, write_swa_cache=write_cache)
    iq = owner.indexer.prepare(inputs, qr, None, None, None, None, scale, write_cache=False)[0]
    return q, iq, getattr(owner, "written_kv", None)


@pytest.mark.parametrize("batch", [1, 3, 64])
@pytest.mark.parametrize("write_cache", [False, True])
def test_shared_mxfp8_matches_independent_quantization_and_reduces_calls(scope, monkeypatch, batch, write_cache):
    definitions, ops = scope
    owner = model(definitions, ops)
    # Exercise bias and skip_bias_add independently on reused consumers.
    owner.wkv.skip_bias_add = True
    owner.indexer.wq_b.skip_bias_add = True
    inputs = torch.randn(batch, 64, generator=torch.Generator().manual_seed(8)).bfloat16()
    actual = run(owner, inputs, write_cache=write_cache)
    assert ops.mx_quant_calls == 2
    monkeypatch.setattr(definitions["CVLinearWrapper"], "can_share_mxfp8_input", lambda *a: False)
    ops.mx_quant_calls = 0
    expected = run(owner, inputs, write_cache=write_cache)
    assert ops.mx_quant_calls == (4 if write_cache else 3)
    for observed, reference in zip(actual, expected):
        if reference is not None:
            torch.testing.assert_close(observed, reference, rtol=0, atol=0)


@pytest.mark.parametrize(
    "difference", ["scale_alg", "group_size", "padding", "width", "partition", "comm", "tuple", "dtype", "scheme"]
)
def test_sharing_requires_compatible_projection_inputs(scope, difference):
    definitions, ops = scope
    owner = model(definitions, ops)
    other = owner.cv_wkv
    if difference == "scale_alg":
        other._mxfp8_quant_method.dynamic_mx_quant_scale_alg = 1
    elif difference == "group_size":
        other._mxfp8_quant_method.group_size = 64
    elif difference == "padding":
        other.linear.mxfp8_tp_padding = (1, 0)
    elif difference == "width":
        other.linear.input_size = 32
    elif difference == "partition":
        other.linear.input_size_per_partition = 32
    elif difference == "comm":
        other._has_communication = True
    elif difference == "tuple":
        other.linear.return_bias = True
    elif difference == "dtype":
        other.linear.weight = other.linear.weight.bfloat16()
    else:
        other._is_mxfp8_dynamic = False
    assert not owner.cv_wq_a.can_share_mxfp8_input(other)


@pytest.mark.parametrize("consumer", ["kv", "indexer"])
@pytest.mark.parametrize("difference", ["scale_alg", "comm"])
def test_incompatible_consumers_use_original_forward(scope, monkeypatch, consumer, difference):
    definitions, ops = scope
    owner = model(definitions, ops)
    other = owner.cv_wkv if consumer == "kv" else owner.indexer.cv_wq_b
    if difference == "scale_alg":
        other._mxfp8_quant_method.dynamic_mx_quant_scale_alg = 1
    else:
        other._has_communication = True
    inputs = torch.randn(3, 64).bfloat16()
    actual = run(owner, inputs, write_cache=True)
    assert ops.mx_quant_calls == 3
    monkeypatch.setattr(definitions["CVLinearWrapper"], "can_share_mxfp8_input", lambda *a: False)
    expected = run(owner, inputs, write_cache=True)
    for observed, reference in zip(actual, expected):
        torch.testing.assert_close(observed, reference, rtol=0, atol=0)


def test_fp16_inputs_keep_original_projection_dtype(scope):
    definitions, ops = scope
    owner = model(definitions, ops, dtype=torch.float16)
    result = run(owner, torch.randn(3, 64).half(), write_cache=True)
    assert ops.mx_quant_calls == 4
    assert all(value.dtype == torch.float16 for value in result)


@pytest.mark.parametrize("method", [object, INT8])
def test_non_mxfp8_paths_keep_original_quantization(scope, monkeypatch, method):
    definitions, ops = scope
    owner = model(definitions, ops, method=method)
    if method is INT8:
        owner.q_norm = SimpleNamespace(weight=torch.ones(32))
        monkeypatch.setattr(
            torch.ops._C_ascend,
            "npu_rms_norm_dynamic_quant",
            lambda value, weight, **kwargs: ops.npu_dynamic_quant(norm(value)),
            raising=False,
        )
    inputs = torch.randn(3, 64).bfloat16()
    actual = run(owner, inputs, write_cache=True)
    assert ops.mx_quant_calls == 0
    assert ops.int8_quant_calls == (2 if method is INT8 else 0)
    qr = norm(owner.wq_a(inputs))
    expected = (
        owner.wq_b(qr).unflatten(-1, (2, 16)),
        owner.indexer.wq_b(qr).view(-1, 2, 16),
        norm(owner.wkv(inputs)).view(-1, 1, 32),
    )
    for observed, reference in zip(actual, expected):
        torch.testing.assert_close(observed, reference, rtol=0, atol=0)


@pytest.mark.parametrize("skip_topk,compress_ratio", [(True, 4), (False, 128), (False, 0)])
def test_unused_indexer_keeps_bf16_qr(scope, skip_topk, compress_ratio):
    definitions, ops = scope
    owner = model(definitions, ops)
    owner.indexer.skip_topk = skip_topk
    owner.compress_ratio = compress_ratio
    _, qr, scale = owner.prolog(torch.randn(3, 64).bfloat16(), None, None, None, None)
    assert qr.dtype == torch.bfloat16
    assert scale is None
    assert ops.mx_quant_calls == 2


def test_quantized_inputs_are_not_cached_across_forwards(scope):
    definitions, ops = scope
    owner = model(definitions, ops)
    first = run(owner, torch.randn(3, 64).bfloat16(), write_cache=True)
    second = run(owner, torch.randn(3, 64).bfloat16(), write_cache=True)
    assert ops.mx_quant_calls == 4
    assert not torch.equal(first[0], second[0])
