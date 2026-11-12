# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source-v3 mixed VQ2/MXFP4 checkpoint configuration.

Like ModelSlim and compressed-tensors, this config describes a checkpoint
format. The VQ2A8 MoE computation is selected through the shared scheme
registry; its implementation lives in methods/w2a8.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import regex as re
import torch
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

from vllm_ascend.quantization.utils import is_fused_moe_layer

from ..methods import get_scheme_class
from ..methods.w2a8.vq2a8_gs32_contract import (
    validate_folded_speculation,
    validate_rotation_config,
    validate_source_config,
)
from ..methods.w2a8.vq2a8_gs32_dspark import validate_vq2a8_dspark_config
from ..methods.w2a8.vq2a8_gs32_ep import GS32_EP_SIZES
from ..methods.w2a8.vq2a8_gs32_format import (
    GS32_ROW_ORDER,
    MXFP4_SOURCE_FIELDS,
    VQ2_SOURCE_FIELDS,
    GS32ProjectionSpec,
    normalize_vq2a8_root_name,
    parse_vq2a8_expert_name,
    read_gs32_header,
    read_gs32_json,
    source_projection_fields,
    validate_gs32_options,
)
from ..methods.w2a8.vq2a8_gs32_startup import GS32StartupState

VQ2A8_MAX_EXPERTS = 256
GS32_SHORT_CONTEXT_TOKENS = 128
GS32_GRAPH_MAX_TOP_K = 8
GS32_GRAPH_MAX_BATCH_SIZE = 64
GS32_DP_SIZES = (2, 4, 8)


@dataclass(frozen=True)
class VQ2A8LayerSpec:
    layer_index: int
    expert_ids: tuple[int, ...]
    specs: Mapping[str, GS32ProjectionSpec]
    quant_types: Mapping[str, Mapping[int, str]]


def _validate_source_config(config):
    return validate_source_config(config)


def _validate_parallel_config(config) -> None:
    """Keep whole experts under TP-only EP or single-node MRV2 DP-only EP."""
    parallel = config.parallel_config
    tp_size = getattr(parallel, "tensor_parallel_size", 1)
    dp_size = getattr(parallel, "data_parallel_size", 1)
    ep_enabled = getattr(parallel, "enable_expert_parallel", False)
    if type(tp_size) is not int or type(dp_size) is not int or tp_size < 1 or dp_size < 1:
        raise ValueError("GS32 TP and DP sizes must be positive integers.")
    ep_size = tp_size * dp_size
    if ep_size not in GS32_EP_SIZES or (ep_size > 1 and not ep_enabled):
        raise ValueError(
            "GS32 requires whole-expert EP of size 2,4,8 when TP*DP >1; tensor-sharded experts are unsupported."
        )
    if dp_size > 1:
        if tp_size != 1 or dp_size not in GS32_DP_SIZES:
            raise ValueError("GS32 DP requires TP1 with DP2, DP4 or DP8; mixed TP+DP is unsupported.")
        if not getattr(config, "use_v2_model_runner", False):
            raise ValueError("GS32 DP execution requires MRV2, including eager execution.")
        if (
            getattr(parallel, "nnodes", 1) != 1
            or getattr(parallel, "data_parallel_size_local", dp_size) != dp_size
            or getattr(parallel, "data_parallel_backend", "mp") != "mp"
            or getattr(parallel, "distributed_executor_backend", "mp") != "mp"
            or getattr(parallel, "data_parallel_external_lb", False)
            or getattr(parallel, "data_parallel_hybrid_lb", False)
            or getattr(parallel, "enable_dbo", False)
            or getattr(parallel, "enable_elastic_ep", False)
        ):
            raise ValueError("GS32 DP requires single-node internal MP data parallelism without DBO or elastic EP.")
    if ep_enabled:
        experts = config.model_config.hf_config.n_routed_experts
        if experts % ep_size or getattr(parallel, "expert_placement_strategy", "linear") != "linear":
            raise ValueError("GS32 EP requires evenly divisible logical experts with linear placement.")
        if getattr(config.model_config.hf_config, "model_type", None) != "deepseek_v4":
            raise ValueError("GS32 EP currently requires DeepSeek-V4.")
    for name in (
        "pipeline_parallel_size",
        "prefill_context_parallel_size",
        "decode_context_parallel_size",
    ):
        if getattr(parallel, name, 1) != 1:
            raise ValueError(f"VQ2A8 currently requires {name}=1.")
    for name in ("enable_eplb", "use_sequence_parallel_moe"):
        if getattr(parallel, name, False):
            raise ValueError(f"VQ2A8 does not yet support {name}.")
    eplb_options = (config.additional_config or {}).get("eplb_config", {})
    if getattr(getattr(parallel, "eplb_config", None), "num_redundant_experts", 0) or any(
        eplb_options.get(name) for name in ("num_redundant_experts", "dynamic_eplb", "expert_map_path")
    ):
        raise ValueError("GS32 requires linear placement without redundant experts or EPLB.")


def validate_vq2a8_config(config) -> dict:
    """Reject unsupported execution before allocating compressed parameters."""
    options = (config.additional_config or {}).get("vq2a8", {})
    if not isinstance(options, dict):
        raise ValueError("GS32 options must be an object.")
    options = validate_gs32_options(config, options)
    _validate_parallel_config(config)
    speculative = getattr(config, "speculative_config", None)
    validate_folded_speculation(getattr(config.model_config.hf_config, "quantization_config", {}) or {}, speculative)
    query_width = 1
    if speculative is not None:
        validate_vq2a8_dspark_config(config)
        query_width += speculative.num_speculative_tokens
    if config.model_config.dtype != torch.bfloat16:
        raise ValueError("VQ2A8 requires BF16 model activations.")
    if config.load_config.load_format not in ("auto", "safetensors") or config.model_config.quantization != "vq2a8":
        raise ValueError("VQ2A8 requires --quantization vq2a8 and the standard auto/safetensors loader.")
    device = getattr(config.load_config, "device", None) or config.device_config.device
    if str(device).split(":", 1)[0] != "npu":
        raise ValueError("VQ2A8 requires parameters resident on an NPU.")
    compilation = config.compilation_config
    if getattr(compilation.mode, "value", compilation.mode) != 0:
        raise ValueError("VQ2A8 requires compilation mode NONE; native banks are not TorchDynamo inputs.")
    graph_mode = getattr(compilation.cudagraph_mode, "name", str(compilation.cudagraph_mode))
    if options.get("gs32_execution") == "full_graph":
        capture_sizes = getattr(compilation, "cudagraph_capture_sizes", None)
        max_num_seqs = getattr(config.scheduler_config, "max_num_seqs", None)
        expected_mode = "FULL_DECODE_ONLY"
        if graph_mode != expected_mode:
            raise ValueError(f"GS32 {options['gs32_execution']} requires {expected_mode}.")
        if type(max_num_seqs) is not int or not 1 <= max_num_seqs <= GS32_GRAPH_MAX_BATCH_SIZE:
            raise ValueError(f"GS32 graph execution requires max_num_seqs in [1,{GS32_GRAPH_MAX_BATCH_SIZE}].")
        if max_num_seqs > 1 and not getattr(config, "use_v2_model_runner", False):
            raise ValueError("GS32 graph execution with max_num_seqs >1 requires MRV2.")
        max_decode_tokens = max_num_seqs * query_width
        if (
            not isinstance(capture_sizes, list)
            or not capture_sizes
            or any(
                type(size) is not int or not 1 <= size <= max_decode_tokens or size % query_width
                for size in capture_sizes
            )
            or capture_sizes != sorted(set(capture_sizes))
            or capture_sizes[-1] != max_decode_tokens
            or type(getattr(compilation, "max_cudagraph_capture_size", None)) is not int
            or compilation.max_cudagraph_capture_size != max_decode_tokens
        ):
            raise ValueError(
                "GS32 graph capture sizes must be increasing positive multiples of the decode query width "
                "ending at max_num_seqs * query_width."
            )
        budget = getattr(config.scheduler_config, "max_num_batched_tokens", None)
        if type(budget) is not int or budget < 1:
            raise ValueError("GS32 graph execution requires a positive integer token budget.")
        if budget < max_decode_tokens:
            raise ValueError("GS32 graph execution requires max_num_batched_tokens >= max_num_seqs * query_width.")
        context = getattr(config.model_config, "max_model_len", 0)
        if context < 1:
            raise ValueError("GS32 graph execution requires max_model_len >=1.")
        if context > GS32_SHORT_CONTEXT_TOKENS:
            # Decode captures one query plus any DSpark candidates per sequence;
            # DSA updates the growing KV history
            # through persistent metadata. Prefill stays outside the decode
            # graph and uses the scheduler's actual token budget; native grouped
            # projections accept the resulting routed rows without a fixed cap.
            if getattr(config.model_config.hf_config, "model_type", None) != "deepseek_v4":
                raise ValueError("GS32 long-context decode graphs require DeepSeek-V4 attention.")
            if not getattr(config.scheduler_config, "enable_chunked_prefill", False):
                raise ValueError(
                    "GS32 long-context decode graphs require chunked prefill with a positive integer token budget."
                )
        top_k = getattr(config.model_config.hf_config, "num_experts_per_tok", 0)
        if type(top_k) is not int or not 1 <= top_k <= GS32_GRAPH_MAX_TOP_K:
            raise ValueError("GS32 full_graph requires top-k in [1,8].")
        if getattr(compilation, "cudagraph_num_of_warmups", 0) < 1:
            raise ValueError("GS32 full_graph requires at least one capture warmup.")
        extra = config.additional_config or {}
        if extra.get("ascend_compilation_config", {}).get("enable_npugraph_ex", True):
            raise ValueError("GS32 full_graph requires enable_npugraph_ex=false.")
        if extra.get("multistream_dsv4_dsa_overlap", True):
            raise ValueError("GS32 full_graph requires multistream_dsv4_dsa_overlap=false.")
    elif not config.model_config.enforce_eager or graph_mode not in ("NONE", "0"):
        raise ValueError("GS32 staged execution requires eager execution and graph mode NONE.")
    # NOTE: kv_transfer_config is allowed for PD disaggregation
    # (MooncakeHybridConnector producer/consumer).
    if any(getattr(config, name, None) is not None for name in ("lora_config",)):
        raise ValueError("VQ2A8 does not yet support LoRA or KV transfer.")
    if getattr(config.scheduler_config, "async_scheduling", False):
        raise ValueError("VQ2A8 currently requires synchronous scheduling.")
    additional = config.additional_config or {}
    if additional.get("multistream_overlap_shared_expert", False):
        raise ValueError(
            "VQ2A8 requires multistream_overlap_shared_expert=false; "
            "GS32 does not expose intermediate shared-expert scheduling events."
        )
    if any(
        additional.get(name)
        for name in ("layer_sharding", "mix_placement", "enable_flashcomm1", "enable_shared_expert_dp", "enable_dsa_cp")
    ):
        raise ValueError("VQ2A8 requires complete experts without mixed placement, shared-expert DP or FlashComm/CP.")
    offload = getattr(config, "offload_config", None)
    if (
        getattr(config.cache_config, "cpu_offload_gb", 0)
        or getattr(getattr(offload, "uva", None), "cpu_offload_gb", 0)
        or getattr(getattr(offload, "prefetch", None), "offload_group_size", 0)
        or getattr(config.model_config, "enable_sleep_mode", False)
    ):
        raise ValueError("VQ2A8 native banks require stable resident parameters; offload/sleep is unsupported.")
    return dict(options)


@register_quantization_config("vq2a8")
class AscendVQ2A8Config(QuantizationConfig):
    def __init__(self, config=None):
        super().__init__()
        self.folded_output_head = _validate_source_config(config)
        self.source_config = dict(config)
        self.layers = {}
        self.is_gs32 = True
        self.checkpoint_root = None
        self.root_tensors = {}
        self.root_by_name = {}
        self.weight_map = {}
        self.mtp_tensors = frozenset()
        self._startup_state = None

    def __getstate__(self):
        # Native registrations belong to the constructing worker/model, not a
        # serialized config or a copied model instance. Revalidate on reuse.
        return {**self.__dict__, "_startup_state": None}

    def bind_checkpoint(self, model_config):
        """Build typed inventories before model construction allocates banks."""
        root = Path(model_config.model).resolve()
        if self.checkpoint_root is not None:
            if self.checkpoint_root != root:
                raise ValueError("A VQ2A8 configuration cannot be rebound to a different checkpoint.")
            return
        if not root.is_dir():
            raise ValueError("Source-v3 requires a local model directory with a safetensors index.")
        checkpoint_config = read_gs32_json(root / "config.json").get("quantization_config")
        _validate_source_config(checkpoint_config)
        if checkpoint_config != self.source_config:
            raise ValueError("Runtime and checkpoint source-v3 quantization_config differ.")
        rotation_path = root / "rotation_config.json"
        if self.folded_output_head and rotation_path.exists():
            validate_rotation_config(read_gs32_json(rotation_path))
        for relative in ("quant.json", "vq2_mixed_gs32/format.json"):
            path = root / relative
            if path.exists():
                metadata = read_gs32_json(path)
                _validate_source_config(
                    {key: metadata.get(key) for key in ("quant_method", "format", "format_version")}
                )
                if metadata.get("row_order") != GS32_ROW_ORDER:
                    raise ValueError("Source-v3 requires natural_gate_then_up row order.")
        index = read_gs32_json(root / "model.safetensors.index.json")
        weight_map = index.get("weight_map")
        if (
            not isinstance(weight_map, dict)
            or not weight_map
            or any(not isinstance(name, str) or not isinstance(shard, str) for name, shard in weight_map.items())
        ):
            raise ValueError("Source-v3 requires a nonempty string safetensors weight_map.")
        tensors = {}
        for shard in sorted(set(weight_map.values())):
            path = (root / shard).resolve()
            if Path(shard).is_absolute() or not path.is_relative_to(root) or not path.is_file():
                raise ValueError(f"Invalid source-v3 shard path: {shard}.")
            header = read_gs32_header(path)
            expected = {name for name, value in weight_map.items() if value == shard}
            if set(header) != expected:
                raise ValueError(f"Source-v3 index/header inventory mismatch: {shard}.")
            for name, value in header.items():
                tensors[name] = {**value, "shape": tuple(value["shape"]), "source_name": name, "shard": shard}
        hf = model_config.hf_config
        if (
            type(hf.num_hidden_layers) is not int
            or hf.num_hidden_layers < 1
            or type(hf.n_routed_experts) is not int
            or not 1 <= hf.n_routed_experts <= VQ2A8_MAX_EXPERTS
            or type(getattr(hf, "num_hash_layers", 0)) is not int
            or not 0 <= getattr(hf, "num_hash_layers", 0) <= hf.num_hidden_layers
        ):
            raise ValueError("Invalid source-v3 model layer/expert geometry.")
        specs = {
            "gate_up": GS32ProjectionSpec(2 * hf.moe_intermediate_size, hf.hidden_size),
            "down": GS32ProjectionSpec(hf.hidden_size, hf.moe_intermediate_size),
        }
        inventory = {}
        roots, canonical_roots = {}, {}
        mtp_tensors = set()
        for name, tensor in tensors.items():
            if name.startswith("mtp."):
                # Original 0731 checkpoints carry several DSpark stages even
                # when num_nextn_predict_layers is 1. The target loader skips
                # these tensors; draft validation remains a separate concern.
                if (
                    re.fullmatch(
                        r"mtp\.(0|[1-9][0-9]*)\.[A-Za-z_][A-Za-z_0-9]*(?:\.(?:[A-Za-z_][A-Za-z_0-9]*|0|[1-9][0-9]*))*",
                        name,
                    )
                    is None
                ):
                    raise ValueError(f"Malformed source-v3 MTP tensor name: {name}.")
                mtp_tensors.add(name)
                continue
            expert = parse_vq2a8_expert_name(name)
            if expert is not None:
                layer, eid, projection, field = expert
                if field == "perm":
                    raise ValueError(f"Source-v3 perm tensors are unsupported; export natural input columns: {name}.")
                if layer >= hf.num_hidden_layers or eid >= hf.n_routed_experts:
                    raise ValueError(f"Expert tensor outside model inventory: {name}.")
                inventory.setdefault((layer, eid, projection), {})[field] = tensor
            else:
                canonical = normalize_vq2a8_root_name(name)
                if canonical in canonical_roots:
                    raise ValueError(f"Duplicate normalized source-v3 root: {canonical}.")
                tensor = {**tensor, "name": canonical}
                roots[name] = canonical_roots[canonical] = tensor
        vq2_fields = set(VQ2_SOURCE_FIELDS)
        layers = {}
        for layer in range(hf.num_hidden_layers):
            expert_ids = tuple(sorted({expert for index, expert, _ in inventory if index == layer}))
            if expert_ids != tuple(range(hf.n_routed_experts)):
                raise ValueError(f"Incomplete source-v3 logical expert inventory for layer {layer}.")
            quant_types = {"gate_up": {}, "down": {}}
            for expert in expert_ids:
                for projection, kind in (("w13", "gate_up"), ("w2", "down")):
                    fields = inventory.get((layer, expert, projection), {})
                    if self.folded_output_head and set(fields) != vq2_fields:
                        raise ValueError(
                            "Folded source-v3 declares six VQ tensors per main expert, without perm or MXFP4 backup: "
                            f"layer {layer}, expert {expert}, {projection}."
                        )
                    if set(fields) == vq2_fields:
                        quant_type = "vq2"
                    elif set(fields) == set(MXFP4_SOURCE_FIELDS):
                        quant_type = "mxfp4"
                    else:
                        raise ValueError(
                            f"Incomplete or mixed source-v3 fields: layer {layer}, expert {expert}, {projection}."
                        )
                    for field, (dtype, shape) in source_projection_fields(specs[kind], quant_type).items():
                        tensor = fields[field]
                        if tensor["dtype"] != dtype or tensor["shape"] != shape:
                            raise ValueError(f"Incorrect source-v3 tensor dtype/shape: {tensor['source_name']}.")
                    quant_types[kind][expert] = quant_type
            layers[layer] = VQ2A8LayerSpec(layer, expert_ids, specs, quant_types)
        self._validate_roots(canonical_roots)
        self.layers, self.root_tensors, self.root_by_name = layers, roots, canonical_roots
        self.weight_map, self.checkpoint_root = dict(weight_map), root
        self.mtp_tensors = frozenset(mtp_tensors)
        self.validate_model_config(hf)

    @staticmethod
    def _validate_roots(roots):
        for name, tensor in roots.items():
            dtype, shape = tensor["dtype"], tensor["shape"]
            if name.endswith(".weight_scale"):
                weight = roots.get(name.removesuffix("_scale"))
                if (
                    dtype != "F8_E8M0"
                    or weight is None
                    or weight["dtype"] != "F8_E4M3"
                    or len(weight["shape"]) != 2
                    or shape != tuple((dim + 127) // 128 for dim in weight["shape"])
                ):
                    raise ValueError(f"Invalid source-v3 FP8 block scale pairing: {name}.")
            elif dtype == "F8_E4M3":
                if not name.endswith(".weight") or len(shape) != 2 or name + "_scale" not in roots:
                    raise ValueError(f"Unpaired source-v3 FP8 root weight: {name}.")
            elif dtype not in ("BF16", "F32", "I64"):
                raise ValueError(f"Unsupported source-v3 root dtype: {name}, {dtype}.")

    @staticmethod
    def normalize_root_name(source_name):
        return normalize_vq2a8_root_name(source_name)

    def root_tensor_spec(self, source_name):
        if source_name not in self.root_tensors:
            raise ValueError(f"Unknown source-v3 root tensor: {source_name}.")
        return self.root_tensors[source_name]

    def root_linear_kind(self, prefix):
        names = [prefix + ".weight"]
        if prefix.endswith(".gate_up_proj"):
            names = [prefix.removesuffix("gate_up_proj") + part + ".weight" for part in ("gate_proj", "up_proj")]
        kinds = set()
        for name in names:
            tensor = self.root_by_name.get(name)
            if tensor is None:
                raise ValueError(f"Source-v3 checkpoint is missing Linear weight {name}.")
            kinds.add("ds_fp8_block128" if tensor["dtype"] == "F8_E4M3" else "unquantized")
        if len(kinds) != 1:
            raise ValueError(f"Shared GateUp projections must use the same root quantization: {prefix}.")
        return kinds.pop()

    @classmethod
    def get_name(cls):
        return "vq2a8"

    @classmethod
    def get_supported_act_dtypes(cls):
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls):
        return 0

    @classmethod
    def get_config_filenames(cls):
        return []

    @classmethod
    def from_config(cls, config):
        return cls(config)

    def layer_spec(self, layer_index: int) -> VQ2A8LayerSpec:
        if layer_index not in self.layers:
            raise ValueError(f"Source-v3 checkpoint inventory is unbound or missing model layer {layer_index}.")
        return self.layers[layer_index]

    def validate_model_config(self, hf_config) -> None:
        num_layers = hf_config.num_hidden_layers
        num_experts = hf_config.n_routed_experts
        hash_layers = getattr(hf_config, "num_hash_layers", 0)
        if (
            type(num_layers) is not int
            or num_layers != len(self.layers)
            or type(num_experts) is not int
            or not 1 <= num_experts <= VQ2A8_MAX_EXPERTS
            or type(hash_layers) is not int
            or not 0 <= hash_layers <= num_layers
        ):
            raise ValueError("VQ2A8 metadata disagrees with the model layer/expert inventory.")
        expected_shapes = {
            "gate_up": (2 * hf_config.moe_intermediate_size, hf_config.hidden_size),
            "down": (hf_config.hidden_size, hf_config.moe_intermediate_size),
        }
        for index, layer in self.layers.items():
            if any(expert >= num_experts for expert in layer.expert_ids):
                raise ValueError(f"VQ2A8 layer {index} has an expert outside the logical model inventory.")
            if layer.expert_ids != tuple(range(num_experts)):
                raise ValueError("VQ2A8 natural-column exports must store every logical expert, including hash layers.")
            for kind, shape in expected_shapes.items():
                spec = layer.specs[kind]
                if (spec.rows, spec.columns) != shape:
                    raise ValueError(f"VQ2A8 layer {index}.{kind} geometry disagrees with model config.")

    def get_quant_method(self, layer, prefix, tid2eid=None):
        if isinstance(layer, LinearBase):
            if self.root_linear_kind(prefix) == "ds_fp8_block128":
                from vllm_ascend.quantization.method_adapters import AscendLinearMethod
                from vllm_ascend.utils import FP8_METHOD

                scheme_cls = get_scheme_class(FP8_METHOD, "ds_linear")
                if scheme_cls is None:
                    raise NotImplementedError("No registered DeepSeek FP8 block128 Linear scheme.")
                return AscendLinearMethod(scheme_cls((128, 128)))
            from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod

            return AscendUnquantizedLinearMethod()
        if not is_fused_moe_layer(layer):
            return None
        match = re.search(r"(?:^|\.)layers\.(\d+)\.mlp\.experts(?:\.routed_experts)?$", prefix)
        if match is None:
            raise ValueError(f"VQ2A8 only supports routed expert layers, got {prefix!r}.")
        config = get_current_vllm_config()
        options = self.validate_runtime_config(config)
        from vllm_ascend.quantization.method_adapters import AscendFusedMoEMethod

        scheme_cls = get_scheme_class("VQ2A8_GS32", "moe")
        if scheme_cls is None:
            raise NotImplementedError("No registered VQ2A8 scheme for layer_type=moe.")

        graph_options = {}
        if options.get("gs32_execution") == "full_graph":
            # DP AllGather concatenates equally padded local graph buckets.
            # EP distributes experts, not tokens: a hot expert rank can receive
            # every routed row, so this capacity must not be divided by EP.
            graph_options["graph_row_capacity"] = (
                getattr(config.parallel_config, "data_parallel_size", 1)
                * max(config.compilation_config.cudagraph_capture_sizes)
                * config.model_config.hf_config.num_experts_per_tok
            )
        scheme = scheme_cls(
            self.layer_spec(int(match.group(1))),
            native_options=options,
            startup_state=self._startup_state,
            moe_config=layer.moe_config,
            swiglu_limit=getattr(config.model_config.hf_config, "swiglu_limit", None),
            **graph_options,
        )
        return AscendFusedMoEMethod(scheme, layer.moe_config, tid2eid=tid2eid)

    def validate_runtime_config(self, config):
        """Validate once per model/worker, rejecting construction-time changes."""
        state = self._startup_state
        if state is not None and state.matches_process():
            state.require_unchanged(config)
            return dict(state.options)
        options = validate_vq2a8_config(config)
        self.validate_model_config(config.model_config.hf_config)
        self._startup_state = GS32StartupState(config, options)
        return dict(options)
