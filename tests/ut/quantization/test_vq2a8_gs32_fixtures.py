# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def modules(monkeypatch):
    root = Path(__file__).resolve().parents[3]

    def stub(name, **attrs):
        module = types.ModuleType(name)
        module.__path__ = []
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    for name in (
        "vllm",
        "vllm.model_executor",
        "vllm.model_executor.layers",
        "vllm_ascend.ops",
        "vllm_ascend.ops.fused_moe",
    ):
        stub(name)
    current = stub("vllm.config", get_current_vllm_config=Mock())
    linear = stub("vllm.model_executor.layers.linear", LinearBase=type("LinearBase", (torch.nn.Module,), {}))
    stub("vllm.model_executor.layers.quantization", register_quantization_config=lambda _: lambda cls: cls)
    stub("vllm.model_executor.layers.quantization.base_config", QuantizationConfig=object)
    stub("vllm_ascend.quantization.utils", is_fused_moe_layer=lambda layer: hasattr(layer, "moe_config"))
    ctx = stub("vllm_ascend.ascend_forward_context", _EXTRA_CTX=SimpleNamespace(moe_comm_method=Mock()))
    builder = stub("vllm_ascend.ops.fused_moe.dataclass.fused_experts", build_fused_experts_input=Mock())
    banks = stub(
        "vllm_ascend.ops.fused_moe.vq2a8",
        VQ2A8ExpertWeights=SimpleNamespace(from_layer=Mock(side_effect=lambda *a, **k: object())),
    )
    adapter = stub("vllm_ascend.quantization.method_adapters", AscendFusedMoEMethod=Mock(), AscendLinearMethod=Mock())
    stub("vllm_ascend.ops.linear", AscendUnquantizedLinearMethod=Mock())
    package = stub("vllm_ascend.quantization.methods")
    package.__path__ = [str(root / "vllm_ascend/quantization/methods")]

    def load(name, relative):
        spec = importlib.util.spec_from_file_location(name, root / relative)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    registry = load("vllm_ascend.quantization.methods.registry", "vllm_ascend/quantization/methods/registry.py")
    package.get_scheme_class = registry.get_scheme_class
    method = load(
        "vllm_ascend.quantization.methods.w2a8.vq2a8_gs32", "vllm_ascend/quantization/methods/w2a8/vq2a8_gs32.py"
    )
    config = load("vllm_ascend.quantization.configs.vq2a8_config", "vllm_ascend/quantization/configs/vq2a8_config.py")
    return SimpleNamespace(
        method=method,
        config=config,
        ctx=ctx,
        builder=builder,
        banks=banks,
        adapter=adapter,
        current=current,
        linear=linear,
        registry=registry,
        load=load,
        stub=stub,
    )


def serving_config():
    return SimpleNamespace(
        additional_config={},
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            enforce_eager=True,
            quantization="vq2a8",
            hf_config=SimpleNamespace(
                num_hidden_layers=1, num_hash_layers=1, n_routed_experts=2, hidden_size=4096, moe_intermediate_size=2048
            ),
        ),
        load_config=SimpleNamespace(load_format="auto", device=None),
        device_config=SimpleNamespace(device="npu"),
        compilation_config=SimpleNamespace(mode=0, cudagraph_mode=0),
        scheduler_config=SimpleNamespace(async_scheduling=False),
        cache_config=SimpleNamespace(),
    )


def source_layer_spec(modules, ids=(0, 1, 2, 3), hidden=128, intermediate=64):
    from vllm_ascend.quantization.methods.w2a8.vq2a8_gs32_format import GS32ProjectionSpec

    return modules.config.VQ2A8LayerSpec(
        0,
        tuple(ids),
        {"gate_up": GS32ProjectionSpec(2 * intermediate, hidden), "down": GS32ProjectionSpec(hidden, intermediate)},
        {
            "gate_up": {expert: "vq2" if expert % 4 < 2 else "mxfp4" for expert in ids},
            "down": {expert: "vq2" if expert % 2 == 0 else "mxfp4" for expert in ids},
        },
    )


def source_field(spec, quant_type, field, expert=0):
    from vllm_ascend.quantization.methods.w2a8.vq2a8_gs32_format import source_projection_fields

    dtype, shape = source_projection_fields(spec, quant_type)[field]
    if field == "perm":
        return torch.arange(spec.columns, dtype=torch.int32).roll(expert)
    if field == "out_dim_scale":
        return torch.full(shape, float(expert + 1), dtype=torch.float32).view(torch.int32).to(torch.uint64)
    if dtype == "F8_E4M3":
        return torch.full(shape, expert + 1, dtype=torch.float32).to(torch.float8_e4m3fn)
    if dtype == "F8_E8M0":
        return torch.full(shape, 127 + expert % 3, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    return torch.full(shape, expert + 1, dtype=torch.uint8)


def source_parameters(modules, spec, moe_config=None):
    scheme = modules.method.AscendGS32FusedMoEScheme(spec, moe_config=moe_config)
    count = len(spec.expert_ids) if moe_config is None else moe_config.num_local_experts
    tensors = scheme.get_weight(count, spec.specs["down"].columns, spec.specs["down"].rows, torch.bfloat16)
    layer = torch.nn.Module()
    for name, value in tensors.items():
        param = torch.nn.Parameter(value, requires_grad=False)
        for key, attr in scheme.get_weight_attrs(name).items():
            setattr(param, key, attr)
        layer.register_parameter(name, param)
    return scheme, layer


def load_source_experts(scheme, layer, ids=None):
    from vllm_ascend.quantization.methods.w2a8.vq2a8_gs32_format import source_projection_fields

    for expert in scheme.layer_spec.expert_ids if ids is None else ids:
        for projection, kind in (("w13", "gate_up"), ("w2", "down")):
            spec = scheme.layer_spec.specs[kind]
            quant_type = scheme.layer_spec.quant_types[kind][expert]
            for field in source_projection_fields(spec, quant_type):
                scheme.load_expert_tensor(
                    layer, expert, projection, field, source_field(spec, quant_type, field, expert)
                )
