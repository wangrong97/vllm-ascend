# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Execute real source-v3 model loading with CPU tensors and isolated vLLM scaffolding."""
# ruff: noqa: F401, F811  # Imported pytest fixtures are injected by name.

import ast
import glob
import json
from collections.abc import Iterable
from pathlib import Path
from types import MethodType
from types import SimpleNamespace as NS

import pytest
import regex as re
import torch

from tests.ut.quantization.test_vq2a8_gs32_checkpoint import source_checkpoint, source_config
from tests.ut.quantization.test_vq2a8_gs32_fixtures import modules


def source_functions():
    root = Path(__file__).resolve().parents[3]
    path = root / "vllm_ascend/models/deepseek_v4/model.py"
    model_class = next(
        n
        for n in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(n, ast.ClassDef) and n.name == "AscendDeepseekV4ForCausalLM"
    )
    contract = root / "vllm_ascend/quantization/methods/w2a8/vq2a8_gs32_format.py"
    helpers = [
        n
        for n in ast.parse(contract.read_text(encoding="utf-8")).body
        if isinstance(n, ast.FunctionDef)
        and n.name in ("parse_vq2a8_expert_name", "normalize_deepseek_v4_weight_name", "normalize_vq2a8_root_name")
    ]
    scope = {"torch": torch, "re": re, "Iterable": Iterable}
    exec(compile(ast.Module(body=helpers, type_ignores=[]), str(contract), "exec"), scope)
    methods = [
        n
        for n in model_class.body
        if isinstance(n, ast.FunctionDef) and n.name in ("_prepare_vq2a8_root", "_load_vq2a8_weights", "load_weights")
    ]
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), scope)
    return scope, model_class


class ExpertScheme:
    """Model-boundary stand-in; numerical projection validation is tested by scheme UTs."""

    def __init__(self, layer, ids):
        self.layer, self.ids, self.loaded = layer, tuple(ids), set()
        self.expected = {(expert, projection) for expert in ids for projection in ("w13", "w2")}
        for projection in ("w13", "w2"):
            layer.register_parameter(
                projection + "_bank",
                torch.nn.Parameter(torch.zeros(len(ids), 2, dtype=torch.uint8), requires_grad=False),
            )

    def is_local_expert(self, expert_id):
        return expert_id in self.ids

    def load_expert_tensor(self, layer, expert_id, projection, field, value):
        assert layer is self.layer
        key = (expert_id, projection)
        if key in self.loaded or field != ("packed_indices" if expert_id == 0 else "weight"):
            raise ValueError("Invalid source-v3 expert field")
        getattr(layer, projection + "_bank").data[self.ids.index(expert_id)].copy_(value)
        self.loaded.add(key)
        return True

    def validate_loaded(self, layer):
        assert layer is self.layer
        if self.loaded != self.expected:
            raise ValueError("Incomplete source-v3 expert projection")
        return {"w13_bank", "w2_bank"}


def register_root(owner, name, value):
    dtypes = {
        torch.bfloat16: "BF16",
        torch.float32: "F32",
        torch.int64: "I64",
        torch.float8_e4m3fn: "F8_E4M3",
        torch.float8_e8m0fnu: "F8_E8M0",
    }
    owner.quant_config.root_tensors[name] = {
        "source_name": name,
        "dtype": dtypes[value.dtype],
        "shape": tuple(value.shape),
    }
    owner.quant_config.weight_map[name] = "root.safetensors"
    return name, value


def model_loader(*, local_ids=(0, 1), rank=0, world=1):
    scope, _ = source_functions()
    scope.update(
        get_tensor_model_parallel_rank=lambda: rank,
        get_tensor_model_parallel_world_size=lambda: world,
        default_weight_loader=lambda p, value: p.data.copy_(value),
    )
    owner = torch.nn.Module()
    for name in ("_prepare_vq2a8_root", "_load_vq2a8_weights", "load_weights"):
        setattr(owner, name, MethodType(scope[name], owner))
    owner.config = NS(n_routed_experts=2, num_attention_heads=8)
    owner.model = torch.nn.Module()
    owner.model.layers = torch.nn.ModuleList([torch.nn.Module()])
    mlp = owner.model.layers[0].mlp = torch.nn.Module()
    mlp.gate = torch.nn.Module()
    mlp.gate.weight = torch.nn.Parameter(torch.zeros(2, 4), requires_grad=False)
    mlp.gate.tid2eid = torch.nn.Parameter(torch.zeros(3, 1, dtype=torch.int32), requires_grad=False)
    mlp.experts = torch.nn.Module()
    routed = mlp.experts.routed_experts = torch.nn.Module()
    scheme = ExpertScheme(routed, local_ids)
    routed.quant_method = NS(quant_method=scheme)
    roots = {}
    owner.quant_config = NS(
        get_name=lambda: "vq2a8",
        root_tensors=roots,
        weight_map={},
        mtp_tensors=frozenset(),
        layer_spec=lambda index: NS(expert_ids=(0, 1)),
        normalize_root_name=scope["normalize_vq2a8_root_name"],
    )

    def spec(name):
        if name not in roots:
            raise ValueError(f"Unknown source-v3 root tensor {name}")
        return roots[name]

    owner.quant_config.root_tensor_spec = spec
    values = [
        register_root(owner, "model.layers.0.mlp.gate.weight", torch.arange(8).reshape(2, 4).bfloat16()),
        register_root(owner, "model.layers.0.mlp.gate.tid2eid", torch.tensor([[0], [1], [0]], dtype=torch.int64)),
    ]
    for expert in (0, 1):
        field = "packed_indices" if expert == 0 else "weight"
        for index, projection in enumerate(("w13", "w2")):
            values.append(
                (
                    f"layers.0.ffn.experts.{expert}.{projection}.{field}",
                    torch.full((2,), expert * 10 + index, dtype=torch.uint8),
                )
            )
            owner.quant_config.weight_map[values[-1][0]] = "experts.safetensors"
    return owner, values, scheme


@pytest.mark.parametrize("local_ids", [(0, 1), (0,), (1,), ()])
def test_global_expert_ids_are_dispatched_only_to_owner(local_ids):
    owner, values, scheme = model_loader(local_ids=local_ids)
    assert owner.load_weights(iter(reversed(values))) == set(dict(owner.named_parameters()))
    assert scheme.loaded == scheme.expected
    for slot, expert in enumerate(local_ids):
        assert scheme.layer.w13_bank[slot].tolist() == [expert * 10] * 2
        assert scheme.layer.w2_bank[slot].tolist() == [expert * 10 + 1] * 2
    assert torch.equal(owner.model.layers[0].mlp.gate.tid2eid, values[1][1].int())


def test_legacy_ffn_gate_roots_load_into_existing_mlp_parameters():
    owner, values, _ = model_loader()
    for index, (name, value) in enumerate(values[:2]):
        del owner.quant_config.root_tensors[name]
        del owner.quant_config.weight_map[name]
        values[index] = register_root(owner, name.replace(".mlp.", ".ffn."), value)

    assert owner.load_weights(iter(values)) == set(dict(owner.named_parameters()))
    gate = owner.model.layers[0].mlp.gate
    torch.testing.assert_close(gate.weight, values[0][1].float())
    torch.testing.assert_close(gate.tid2eid, values[1][1].int())


def original_0731_reference():
    path = Path(__file__).resolve().parents[3] / "docs/vq2a8/deepseek_v4_flash_0731_reference.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_original_0731_non_routed_inventory_uses_existing_model_names(modules):
    scope, _ = source_functions()
    observed = original_0731_reference()["main_non_routed_layouts"]
    expected_modules = {
        "embed": "model.embed_tokens",
        "norm": "model.norm",
        "head": "lm_head",
        "attn.wq_a": "self_attn.wq_a",
        "attn.wq_b": "self_attn.wq_b",
        "attn.wkv": "self_attn.wkv",
        "attn.wo_a": "self_attn.wo_a",
        "attn.wo_b": "self_attn.wo_b",
        "attn.q_norm": "self_attn.q_norm",
        "attn.kv_norm": "self_attn.kv_norm",
        "attn.compressor.wgate": "self_attn.compressor.wgate",
        "attn.compressor.wkv": "self_attn.compressor.wkv",
        "attn.compressor.norm": "self_attn.compressor.norm",
        "attn.indexer.wq_b": "self_attn.indexer.wq_b",
        "attn.indexer.weights_proj": "self_attn.indexer.weights_proj",
        "attn.indexer.compressor.wgate": "self_attn.indexer.compressor.wgate",
        "attn.indexer.compressor.wkv": "self_attn.indexer.compressor.wkv",
        "attn.indexer.compressor.norm": "self_attn.indexer.compressor.norm",
        "attn_norm": "input_layernorm",
        "ffn_norm": "post_attention_layernorm",
        "ffn.gate": "mlp.gate",
        "ffn.shared_experts.w1": "mlp.shared_experts.gate_proj",
        "ffn.shared_experts.w2": "mlp.shared_experts.down_proj",
        "ffn.shared_experts.w3": "mlp.shared_experts.up_proj",
    }
    direct = {
        "hc_head_fn": "model.hc_head_fn",
        "hc_head_base": "model.hc_head_base",
        "hc_head_scale": "model.hc_head_scale",
        "hc_attn_fn": "hc_attn_fn",
        "hc_attn_base": "hc_attn_base",
        "hc_attn_scale": "hc_attn_scale",
        "hc_ffn_fn": "hc_ffn_fn",
        "hc_ffn_base": "hc_ffn_base",
        "hc_ffn_scale": "hc_ffn_scale",
        "attn.attn_sink": "self_attn.attn_sink",
        "attn.compressor.ape": "self_attn.compressor.ape",
        "attn.indexer.compressor.ape": "self_attn.indexer.compressor.ape",
        "ffn.gate.bias": "mlp.gate.e_score_correction_bias",
        "ffn.gate.tid2eid": "mlp.gate.tid2eid",
    }
    assert len(observed) == 50
    roots = {}
    for _, dtype, shape, _, name in observed:
        if name.startswith("layers."):
            _, index, suffix = name.split(".", 2)
            prefix = f"model.layers.{index}."
        else:
            prefix, suffix = "", name
        if suffix in direct:
            expected = prefix + direct[suffix]
        else:
            module, field = suffix.rsplit(".", 1)
            expected = prefix + expected_modules[module] + (".weight_scale" if field == "scale" else ".weight")
        assert scope["normalize_vq2a8_root_name"](name) == expected
        assert scope["normalize_deepseek_v4_weight_name"](name) == expected
        roots[expected] = {"source_name": name, "dtype": dtype, "shape": tuple(shape)}
    modules.config.AscendVQ2A8Config._validate_roots(roots)


@pytest.mark.parametrize(
    "name",
    [
        "layers.01.attn.wq_a.weight",
        "layers.0.attn.unknown.weight",
        "layers.0.ffn.shared_experts.w1.unknown",
        "layers.0.attn.wq_a.weight_scale",
        "layers.0.attn..wq_a.weight",
        "unknown_embed.weight",
        "mtp.0.ffn.experts.0.w1.weight",
    ],
)
def test_original_root_names_are_validated_before_permissive_loader_remapping(name):
    scope, _ = source_functions()
    with pytest.raises(ValueError, match="Unknown source-v3 root tensor"):
        scope["normalize_vq2a8_root_name"](name)


@pytest.mark.parametrize("folded", [False, True])
def test_original_norm_head_and_embedding_values_are_loaded_without_extra_transforms(folded):
    owner, values, _ = model_loader()
    owner.quant_config.folded_output_head = folded
    owner.model.norm = torch.nn.Module()
    owner.model.embed_tokens = torch.nn.Module()
    owner.lm_head = torch.nn.Module()
    sources = []
    for name, module, value in (
        (
            "norm.weight",
            owner.model.norm,
            torch.ones(4, dtype=torch.bfloat16) if folded else torch.tensor([0.5, 1.5, 2.0, 3.0], dtype=torch.bfloat16),
        ),
        ("embed.weight", owner.model.embed_tokens, torch.arange(12).reshape(3, 4).bfloat16()),
        ("head.weight", owner.lm_head, torch.arange(12, 24).reshape(3, 4).bfloat16()),
    ):
        parameter = torch.nn.Parameter(torch.zeros_like(value), requires_grad=False)
        module.weight = parameter
        sources.append((parameter, value))
        values.append(register_root(owner, name, value))
    assert owner.load_weights(iter(values)) == set(dict(owner.named_parameters()))
    for parameter, source in sources:
        assert parameter.dtype == source.dtype
        assert torch.equal(parameter, source)


@pytest.mark.parametrize("bad_value", [0.5, float("nan")])
def test_folded_final_norm_rejects_unfused_or_nonfinite_gamma(bad_value):
    owner, _, _ = model_loader()
    owner.quant_config.folded_output_head = True
    value = torch.tensor([1, bad_value, 1, 1], dtype=torch.bfloat16)
    register_root(owner, "norm.weight", value)
    with pytest.raises(ValueError, match="all-one final norm"):
        owner._prepare_vq2a8_root(
            torch.nn.Parameter(torch.ones_like(value)),
            value,
            "model.norm.weight",
            owner.quant_config.root_tensors["norm.weight"],
        )


@pytest.mark.parametrize("mode", ["complete", "missing_main", "missing_mtp", "duplicate_mtp"])
def test_original_mtp_is_skipped_before_expert_parsing_without_weakening_completeness(mode):
    owner, values, _ = model_loader()
    names = frozenset(row[4] for row in original_0731_reference()["mtp_layouts"])
    assert len(names) == 45
    owner.quant_config.mtp_tensors = names
    for name in sorted(names):
        owner.quant_config.weight_map[name] = "mtp.safetensors"
        # Payload is intentionally opaque: the main model must not interpret it.
        values.append((name, None))
    if mode == "missing_main":
        del values[0]
    elif mode == "missing_mtp":
        values.pop()
    elif mode == "duplicate_mtp":
        values.append(values[-1])
    if mode == "complete":
        assert owner.load_weights(iter(values)) == set(dict(owner.named_parameters()))
    else:
        message = "Duplicate source-v3" if mode == "duplicate_mtp" else "Incomplete source-v3"
        with pytest.raises(ValueError, match=message):
            owner.load_weights(iter(values))


def test_shared_name_mapping_preserves_original_vision_router_bias_name():
    scope, _ = source_functions()
    assert scope["normalize_deepseek_v4_weight_name"]("layers.0.ffn.gate.bias_vl") == "model.layers.0.mlp.gate.bias_vl"
    assert (
        scope["normalize_deepseek_v4_weight_name"]("layers.3.ffn.gate.bias")
        == "model.layers.3.mlp.gate.e_score_correction_bias"
    )


@pytest.mark.parametrize("index", [0, 2, 3])
def test_missing_root_or_local_expert_projection_is_rejected(index):
    owner, values, _ = model_loader()
    with pytest.raises(ValueError, match="Incomplete source-v3"):
        owner.load_weights(iter(values[:index] + values[index + 1 :]))


@pytest.mark.parametrize("index", [0, 2, 5])
def test_duplicate_tensor_is_rejected_even_for_remote_experts(index):
    owner, values, _ = model_loader(local_ids=(0,))
    with pytest.raises(ValueError, match="Duplicate source-v3"):
        owner.load_weights(iter([*values, values[index]]))


def test_legacy_stacked_expert_interface_is_rejected():
    owner, values, _ = model_loader()
    values[-1] = ("model.layers.0.mlp.experts.w2_packed_zn", values[-1][1])
    with pytest.raises(ValueError, match="Unknown source-v3 checkpoint tensor"):
        owner.load_weights(iter(values))


@pytest.mark.parametrize("suffix", ["1.w13.unknown_field", "17.w13.weight"])
def test_unknown_remote_expert_tensor_is_rejected_against_bound_index(suffix):
    owner, values, _ = model_loader(local_ids=(0,))
    values.append((f"layers.0.ffn.experts.{suffix}", torch.zeros(2, dtype=torch.uint8)))
    with pytest.raises(ValueError, match="Unknown source-v3 checkpoint tensor"):
        owner.load_weights(iter(values))


def test_missing_remote_expert_tensor_is_rejected_against_bound_index():
    owner, values, _ = model_loader(local_ids=(0,))
    with pytest.raises(ValueError, match="Incomplete source-v3 checkpoint"):
        owner.load_weights(iter(values[:-1]))


def test_hash_routes_outside_global_inventory_are_rejected():
    owner, values, _ = model_loader()
    values[1][1][0, 0] = 2
    with pytest.raises(ValueError, match="Hash table"):
        owner.load_weights(iter(values))


def test_hash_routes_must_reference_sparse_checkpoint_inventory():
    owner, values, _ = model_loader(local_ids=(0,))
    owner.config.n_routed_experts = 32
    owner.quant_config.layer_spec = lambda index: NS(expert_ids=(0, 3))
    # ID 1 is in the model's numeric range, but absent from the source inventory.
    with pytest.raises(ValueError, match="Hash table"):
        owner.load_weights(iter(values))


@pytest.mark.parametrize("bad", ["dtype", "shape"])
def test_root_payload_must_match_preflight_header(bad):
    owner, values, _ = model_loader()
    name, value = values[0]
    values[0] = (name, value.float() if bad == "dtype" else value[:1])
    with pytest.raises(ValueError, match="disagrees with its header"):
        owner.load_weights(iter(values))


def test_preserves_fp32_norm_parameter_and_loader_identity():
    owner, values, _ = model_loader()
    owner.model.norm = torch.nn.Module()
    parameter = torch.nn.Parameter(torch.zeros(4, dtype=torch.bfloat16), requires_grad=False)
    parameter.marker = object()
    marker = parameter.marker
    owner.model.norm.weight = parameter
    value = torch.tensor([1.001, 1.002, 1.003, 1.004], dtype=torch.float32)
    values.append(register_root(owner, "model.norm.weight", value))
    owner.load_weights(iter(values))
    assert owner.model.norm.weight is parameter and parameter.marker is marker
    assert parameter.dtype == torch.float32 and torch.equal(parameter, value)


def add_fp8_root(owner, *, rank=0, world=1, row=False, original=False):
    attention = owner.model.layers[0].self_attn = torch.nn.Module()
    linear = torch.nn.Module()
    setattr(attention, "wo_b" if row else "wq_b", linear)
    n, k = 512, 256
    shape = (n, k // world) if row else (n // world, k)
    linear.weight = torch.nn.Parameter(torch.zeros(shape).to(torch.float8_e4m3fn), requires_grad=False)
    linear.weight_scale = torch.nn.Parameter(torch.zeros(shape[0] // 128, shape[1] // 128), requires_grad=False)
    axis = 1 if row else 0

    def loader(parameter, value):
        width = parameter.shape[axis]
        parameter.data.copy_(value.narrow(axis, rank * width, width))

    linear.weight.weight_loader = loader
    linear.weight_scale.weight_loader = loader
    prefix = "model.layers.0.self_attn.o_b_proj" if row else "model.layers.0.self_attn.q_b_proj"
    if original:
        prefix = "layers.0.attn.wo_b" if row else "layers.0.attn.wq_b"
    weight = torch.arange(n * k).remainder(100).reshape(n, k).to(torch.float8_e4m3fn)
    scale = torch.arange(110, 118, dtype=torch.uint8).reshape(n // 128, k // 128).view(torch.float8_e8m0fnu)
    values = [register_root(owner, prefix + ".weight", weight), register_root(owner, prefix + ".scale", scale)]
    return linear, values


@pytest.mark.parametrize("row", [False, True])
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("original", [False, True])
def test_fp8_weight_and_scale_share_tp_partition_without_dequantizing_weight(row, rank, original):
    owner, values, _ = model_loader(rank=rank, world=2)
    linear, fp8 = add_fp8_root(owner, rank=rank, world=2, row=row, original=original)
    owner.load_weights(iter([*values, *fp8]))
    axis = 1 if row else 0
    for parameter, (_, source) in zip((linear.weight, linear.weight_scale), fp8):
        width = parameter.shape[axis]
        assert torch.equal(parameter.float(), source.narrow(axis, rank * width, width).float())
    assert linear.weight.dtype == torch.float8_e4m3fn
    assert linear.weight_scale.dtype == torch.float32
    width = linear.weight_scale.shape[axis]
    recovered = ((linear.weight_scale.view(torch.int32) >> 23) & 255).to(torch.uint8)
    assert torch.equal(recovered, fp8[1][1].view(torch.uint8).narrow(axis, rank * width, width))


@pytest.mark.parametrize("dtype,bad_value", [(torch.float8_e8m0fnu, 255), (torch.float32, 0), (torch.float32, 1.5)])
def test_root_scale_rejects_invalid_exponent_or_non_power_of_two(dtype, bad_value):
    owner, values, _ = model_loader()
    _, fp8 = add_fp8_root(owner)
    name, scale = fp8[-1]
    scale = scale.float() if dtype == torch.float32 else scale
    if dtype == torch.float32:
        scale[0, 0] = bad_value
    else:
        scale.view(torch.uint8)[0, 0] = bad_value
    fp8[-1] = register_root(owner, name, scale)
    with pytest.raises(ValueError, match="positive powers of two"):
        owner.load_weights(iter([*values, *fp8]))


@pytest.mark.parametrize("missing_scale", [False, True])
@pytest.mark.parametrize("original", [False, True])
def test_shared_fp8_gate_up_weights_and_scales_have_independent_shards(missing_scale, original):
    owner, values, _ = model_loader()
    shared = owner.model.layers[0].mlp.shared_experts = torch.nn.Module()
    fused = shared.gate_up_proj = torch.nn.Module()
    fused.weight = torch.nn.Parameter(torch.zeros(256, 128).to(torch.float8_e4m3fn), requires_grad=False)
    fused.weight_scale = torch.nn.Parameter(torch.zeros(2, 1), requires_grad=False)

    def loader(parameter, value, shard_id):
        width = parameter.shape[0] // 2
        parameter.data[shard_id * width : (shard_id + 1) * width].copy_(value)

    fused.weight.weight_loader = loader
    fused.weight_scale.weight_loader = loader
    for index, projection in enumerate(("gate_proj", "up_proj")):
        prefix = f"model.layers.0.mlp.shared_experts.{projection}"
        if original:
            prefix = f"layers.0.ffn.shared_experts.{'w1' if index == 0 else 'w3'}"
        weight = torch.full((128, 128), index + 1).to(torch.float8_e4m3fn)
        scale = torch.tensor([[110 + index]], dtype=torch.uint8).view(torch.float8_e8m0fnu)
        values.extend(
            [register_root(owner, prefix + ".weight", weight), register_root(owner, prefix + ".scale", scale)]
        )
    if missing_scale:
        with pytest.raises(ValueError, match="Incomplete source-v3"):
            owner.load_weights(iter(values[:-1]))
    else:
        owner.load_weights(iter(reversed(values)))
        assert torch.equal(fused.weight.float()[:128], torch.ones(128, 128))
        assert torch.equal(fused.weight.float()[128:], torch.full((128, 128), 2.0))
        assert fused.weight_scale[:, 0].tolist() == [2**-17, 2**-16]


@pytest.mark.parametrize("rank", [0, 1, 2, 3])
@pytest.mark.parametrize("original", [False, True])
def test_sink_shards_and_hc_scale_mapping(rank, original):
    owner, values, _ = model_loader(rank=rank, world=4)
    decoder = owner.model.layers[0]
    decoder.self_attn = torch.nn.Module()
    decoder.self_attn.attn_sink = torch.nn.Parameter(torch.zeros(2), requires_grad=False)
    decoder.hc_attn_scale = torch.nn.Parameter(torch.zeros(3), requires_grad=False)
    sink, scale = torch.arange(8).float(), torch.arange(3).float()
    values.extend(
        [
            register_root(owner, "layers.0.attn.attn_sink" if original else "model.layers.0.self_attn.sinks", sink),
            register_root(owner, "layers.0.hc_attn_scale" if original else "model.layers.0.attn_hc.scale", scale),
        ]
    )
    owner.load_weights(iter(values))
    assert torch.equal(decoder.self_attn.attn_sink, sink[rank * 2 : (rank + 1) * 2])
    assert torch.equal(decoder.hc_attn_scale, scale)


def construct_model(
    *, weight_map=None, quant_name="vq2a8", quant_config=None, model_config=None, inherited_patterns=None, events=None
):
    """Run the real constructor while isolating unrelated model/NPU allocation."""
    scope, original_class = source_functions()
    constructor = next(n for n in original_class.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    events = [] if events is None else events
    if quant_config is None and quant_name is not None:
        quant_config = NS(get_name=lambda: quant_name, weight_map={})

        def bind_checkpoint(model):
            events.append("bind")
            quant_config.weight_map.update(weight_map or {"head.weight": "weights.safetensors"})

        quant_config.bind_checkpoint = bind_checkpoint

    class Model(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            if quant_config is not None and quant_config.get_name() == "vq2a8":
                assert quant_config.weight_map, "Checkpoint must bind before parameter allocation"
            events.append("allocate")
            self.make_empty_intermediate_tensors = lambda: None

    base = ast.parse("torch.nn.Module", mode="eval").body
    owner_class = ast.ClassDef(name="Owner", bases=[base], keywords=[], body=[constructor], decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[owner_class], type_ignores=[]))
    scope.update(
        Path=Path,
        VllmConfig=object,
        get_current_hardware_profile=lambda: NS(supports=lambda _: False),
        HardwareCapability=NS(DSV4_COMPRESSED_CACHE="cache"),
        get_pp_group=lambda: NS(is_last_rank=False),
        PPMissingLayer=torch.nn.Module,
        LogitsProcessor=lambda _: None,
        maybe_prefix=lambda _, name: name,
    )
    exec(compile(module, "model-constructor", "exec"), scope)
    owner_type = scope["Owner"]
    owner_type.model_cls = Model
    owner_type.set_moe_parameters = lambda self: None
    if inherited_patterns is not None:
        owner_type.allow_patterns_overrides = inherited_patterns
    config = NS(
        model_config=model_config or NS(hf_config=NS(vocab_size=7, num_hidden_layers=1)),
        quant_config=quant_config,
    )
    return owner_type(vllm_config=config), events


def test_checkpoint_binds_before_model_parameter_allocation():
    _, events = construct_model()
    assert events == ["bind", "allocate"]


@pytest.mark.parametrize(
    "shards",
    [
        ("rest-model.safetensors", "layer_00.safetensors"),
        ("rest/rest-model.safetensors", "vq2_mixed_gs32/layer_00.safetensors"),
        ("root/rest/rest-model.safetensors", "experts/vq2_mixed_gs32/layer_00.safetensors"),
    ],
    ids=("flat", "converter_layout", "two_levels"),
)
def test_default_loader_discovers_all_indexed_shards_at_uniform_depth(tmp_path, shards):
    for relative in shards:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    weight_map = {f"tensor_{index}": relative for index, relative in enumerate(shards)}
    weight_map["another_tensor_in_first_shard"] = shards[0]
    owner, events = construct_model(weight_map=weight_map)

    # Upstream DefaultModelLoader stops at the first pattern with any matches.
    # Exercise actual glob discovery rather than reproducing pattern creation.
    discovered = []
    for pattern in owner.allow_patterns_overrides:
        discovered.extend(glob.glob(str(tmp_path / pattern)))
        if discovered:
            break
    assert {Path(path).relative_to(tmp_path).as_posix() for path in discovered} == set(shards)
    assert events == ["bind", "allocate"]


@pytest.mark.parametrize(
    "shards",
    [
        ("rest-model.safetensors", "vq2_mixed_gs32/layer_00.safetensors"),
        ("rest/rest-model.safetensors", "experts/vq2_mixed_gs32/layer_00.safetensors"),
    ],
)
def test_mixed_shard_depth_is_rejected_before_allocation(tmp_path, shards):
    for relative in shards:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    events = []
    with pytest.raises(ValueError, match="Source-v3 safetensors shards must use a uniform directory depth"):
        construct_model(
            weight_map={f"tensor_{index}": relative for index, relative in enumerate(shards)}, events=events
        )
    assert events == ["bind"]


@pytest.mark.parametrize("quant_name", [None, "fp8"])
@pytest.mark.parametrize("inherited_patterns", [None, ["existing/*.safetensors"]])
def test_non_vq2_constructor_preserves_loader_discovery(quant_name, inherited_patterns):
    owner, events = construct_model(quant_name=quant_name, inherited_patterns=inherited_patterns)
    if inherited_patterns is None:
        assert not hasattr(owner, "allow_patterns_overrides")
    else:
        assert owner.allow_patterns_overrides is inherited_patterns
    assert events == ["allocate"]


def test_constructor_still_rejects_a_truly_missing_indexed_shard(modules, source_checkpoint):
    quant = modules.config.AscendVQ2A8Config(source_config())
    (source_checkpoint.root / "weights.safetensors").unlink()
    events = []
    with pytest.raises(ValueError, match="Invalid source-v3 shard path"):
        construct_model(quant_config=quant, model_config=source_checkpoint.config, events=events)
    assert not events
    assert quant.checkpoint_root is None
