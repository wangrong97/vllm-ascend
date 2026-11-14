# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resident GS32 tensors with the standard Ascend routing/combine pipeline."""

from dataclasses import replace
from typing import Any

import torch

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import build_fused_experts_input

from ..base import AscendMoEScheme, QuantType
from ..registry import register_scheme
from .vq2a8_gs32_contract import resolve_gs32_options
from .vq2a8_gs32_ep import gs32_expert_placement
from .vq2a8_gs32_format import (
    GS32_DTYPES,
    GS32_FIELDS,
    SOURCE_TO_NATIVE,
    convert_expert_tensor,
    gs32_projection_shapes,
    source_projection_fields,
)
from .vq2a8_gs32_full_ffn_layout import build_full_ffn_bank, validate_full_ffn_spec

DEFAULT_GRAPH_ROW_CAPACITY = 8  # Legacy B1 capture for top-k up to eight.


@register_scheme("VQ2A8_GS32", "moe")
class AscendGS32FusedMoEScheme(AscendMoEScheme):
    """Keep the common VQ routing adapter, replace only weight ABI and compute."""

    quant_type = QuantType.VQ2A8
    supports_eplb = False

    def __init__(
        self,
        layer_spec,
        *,
        native_options=None,
        moe_config=None,
        graph_row_capacity=DEFAULT_GRAPH_ROW_CAPACITY,
        swiglu_limit=0.0,
        startup_state=None,
    ):
        self.layer_spec = layer_spec
        self.native_options = resolve_gs32_options(native_options or {})
        # Mixed-precision EPLB switch (vq2a8.eplb); "off" keeps the runtime
        # byte-identical to the baseline (no capacity reservation, no
        # accounting, no split tables).
        from vllm_ascend.eplb.mp.config import get_mp_eplb_options

        self.mp_eplb = get_mp_eplb_options(self.native_options)
        self.moe_config = moe_config
        self.graph_row_capacity = graph_row_capacity
        self.swiglu_limit = 0.0 if swiglu_limit is None else swiglu_limit
        self.startup_state = startup_state
        self.placement = None

    def get_weight(self, num_experts, intermediate_size_per_partition, hidden_sizes, params_dtype):
        if params_dtype != torch.bfloat16:
            raise ValueError("GS32 requires BF16 activations.")
        validate_full_ffn_spec(self.layer_spec)
        self.placement = gs32_expert_placement(self.layer_spec.expert_ids, num_experts, self.moe_config)
        expected = {
            "gate_up": (2 * intermediate_size_per_partition, hidden_sizes),
            "down": (hidden_sizes, intermediate_size_per_partition),
        }
        result = {}
        self.parameter_experts = {}
        self.loaded_experts = {}
        self.parameter_metadata = {}
        # Allocated in the same device context as Parameters; empty hash ranks
        # still need a device anchor for routing and deferred validity.
        self.anchor = torch.empty(0, dtype=torch.bfloat16)
        for prefix, kind in (("w13", "gate_up"), ("w2", "down")):
            spec = self.layer_spec.specs[kind]
            if (spec.rows, spec.columns) != expected[kind]:
                raise ValueError(f"GS32 {kind} geometry does not match the unsharded MoE layer.")
            for quant_type in ("vq2", "mxfp4"):
                ids = tuple(e for e in self.placement.global_ids if self.layer_spec.quant_types[kind][e] == quant_type)
                if not ids:
                    continue
                if quant_type == "vq2":
                    native_fields = dict(zip(GS32_FIELDS, zip(GS32_DTYPES, gs32_projection_shapes(spec)), strict=True))
                else:
                    native_fields = {
                        "mxfp4_weight": (torch.uint8, (spec.rows, spec.columns // 2)),
                        "mxfp4_scale": (torch.uint8, (spec.rows, spec.columns // 32)),
                    }
                for source_field in source_projection_fields(spec, quant_type):
                    native_field = SOURCE_TO_NATIVE[source_field]
                    name = f"{prefix}_{native_field}"
                    dtype, shape = native_fields[native_field]
                    # MP-EPLB capacity reservation (doc 2.5-D): tail spare
                    # slots keep the bank's dim-0 constant for the whole run
                    # so plans/captured graphs never need rebuilding.
                    spare = self.mp_eplb.replica_slots[quant_type] // self.placement.ep_size
                    result[name] = torch.empty((len(ids) + spare, *shape), dtype=dtype)
                    self.parameter_experts[name] = {expert: slot for slot, expert in enumerate(ids)}
                    self.loaded_experts[name] = set()
                    self.parameter_metadata[name] = (kind, quant_type, source_field, spec)
        return result

    @staticmethod
    def parameter_name(projection, field):
        if projection not in ("w13", "w2") or field not in SOURCE_TO_NATIVE:
            raise ValueError(f"Unknown source-v3 projection field: {projection}.{field}.")
        return f"{projection}_{SOURCE_TO_NATIVE[field]}"

    def is_local_expert(self, expert_id):
        if self.placement is None:
            raise ValueError("GS32 parameters have not been allocated.")
        return expert_id in self.placement.global_ids

    def load_expert_tensor(self, layer, expert_id, projection, field, value):
        if expert_id not in self.layer_spec.expert_ids:
            raise ValueError(f"Expert {expert_id} is absent from the source-v3 inventory.")
        kind = {"w13": "gate_up", "w2": "down"}.get(projection)
        if kind is None or field not in source_projection_fields(
            self.layer_spec.specs[kind], self.layer_spec.quant_types[kind][expert_id]
        ):
            raise ValueError(f"Unexpected source-v3 expert field: {projection}.{field}.")
        if not self.is_local_expert(expert_id):
            return False
        name = self.parameter_name(projection, field)
        return self.weight_loader(getattr(layer, name), value, expert_id=expert_id, return_success=True)

    def validate_loaded(self, layer=None):
        missing = {
            name: sorted(set(experts) - self.loaded_experts[name])
            for name, experts in self.parameter_experts.items()
            if set(experts) != self.loaded_experts[name]
        }
        if missing:
            raise ValueError(f"Incomplete source-v3 expert checkpoint: {missing}.")
        return set(self.parameter_experts)

    def process_weights_after_loading(self, layer):
        from vllm_ascend.ops.fused_moe.vq2a8_gs32 import GS32ExpertWeights

        self.validate_loaded(layer)
        placement = self.placement
        if placement is None:
            raise ValueError("GS32 parameters must be allocated before post-load processing.")
        layer.vq2a8_expert_ids = self.layer_spec.expert_ids
        layer.vq2a8_layer_index = self.layer_spec.layer_index
        layer.vq2a8_ep_size = placement.ep_size
        layer.vq2a8_ep_rank = placement.ep_rank
        layer.vq2a8_local_global_expert_ids = placement.global_ids
        layer.vq2a8_local_expert_ids = placement.local_ids
        layer.vq2a8_expert_start = placement.expert_start
        layer.vq2a8_num_experts = placement.num_local_experts
        layer.register_buffer("vq2a8_anchor", self.anchor, persistent=False)
        local_types = {
            kind: {
                expert - placement.expert_start: self.layer_spec.quant_types[kind][expert]
                for expert in placement.global_ids
            }
            for kind in self.layer_spec.specs
        }
        # MP-EPLB capacity (doc 2.5-D): trailing EMPTY(-1) sentinel slots keep
        # the packed bank dim-0 constant; plans/graphs never need rebuilding.
        spare_slots = (
            {quant: self.mp_eplb.replica_slots[quant] // placement.ep_size for quant in ("vq2", "mxfp4")}
            if self.mp_eplb.enabled
            else None
        )
        local_spec = replace(
            self.layer_spec,
            expert_ids=placement.local_ids,
            quant_types=local_types,
        )
        bank = build_full_ffn_bank(layer, local_spec, swiglu_limit=self.swiglu_limit, spare_slots=spare_slots)
        for name, tensor in bank["fields"].items():
            layer.register_buffer(f"vq2a8_full_ffn_{name}", tensor, persistent=False)
        layer.vq2a8_full_ffn_bank = bank
        # The complete-FFN bank owns all bytes after packing; release the source
        # Parameters to avoid retaining a second resident checkpoint copy.
        for name in self.parameter_experts:
            delattr(layer, name)
        layer.vq2a8_weights = GS32ExpertWeights(
            layer,
            local_spec,
            self.native_options,
            graph_row_capacity=self.graph_row_capacity,
            startup_state=self.startup_state,
            mp_eplb=self.mp_eplb,
        )

    def get_dynamic_quant_param(self, num_experts, intermediate_size_per_partition, hidden_sizes, params_dtype):
        return {}

    def get_weight_attrs(self, name: str) -> dict[str, Any]:
        return {"weight_loader": self.weight_loader, "vq2a8_parameter_name": name}

    def weight_loader(
        self, param, loaded_weight, weight_name=None, *, expert_id=None, shard_id=None, return_success=False
    ):
        """Load one independently named expert field into its typed local slot."""
        if type(expert_id) is not int or shard_id is not None:
            raise ValueError("Source-v3 requires an explicit global expert_id and complete w13/w2 projections.")
        name = param.vq2a8_parameter_name
        if expert_id not in self.layer_spec.expert_ids:
            raise ValueError(f"Expert {expert_id} is absent from the source-v3 inventory.")
        if not self.is_local_expert(expert_id):
            return False if return_success else None
        if expert_id not in self.parameter_experts[name]:
            raise ValueError(f"Source-v3 expert {expert_id} has a different quantization type for {name}.")
        if expert_id in self.loaded_experts[name]:
            raise ValueError(f"Duplicate source-v3 expert field: {expert_id}.{name}.")
        _, quant_type, source_field, spec = self.parameter_metadata[name]
        value = convert_expert_tensor(source_field, loaded_weight, spec, quant_type)
        param.data[self.parameter_experts[name][expert_id]].copy_(value)
        self.loaded_experts[name].add(expert_id)
        return True if return_success else None

    @staticmethod
    def projection_anchor(layer, prefix):
        bank = layer.vq2a8_full_ffn_bank
        if bank["expert_ids"]:
            name = "gate_weight" if prefix == "w13" else "down_weight"
            return bank["fields"][name]
        return layer.vq2a8_anchor

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts=None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Consume the routes selected by AscendRoutedExperts exactly once."""
        return _EXTRA_CTX.moe_comm_method.fused_experts(
            fused_experts_input=build_fused_experts_input(
                hidden_states=x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                layer=layer,
                w1=self.projection_anchor(layer, "w13"),
                w2=self.projection_anchor(layer, "w2"),
                vq2a8=layer.vq2a8_weights,
                quant_type=self.quant_type,
                dynamic_eplb=False,
                expert_map=layer.ascend_expert_map,
                global_redundant_expert_num=layer.global_redundant_expert_num,
                mc2_mask=layer.ascend_mc2_mask,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
                pertoken_scale=layer.ascend_pertoken_scale,
                activation=layer.activation,
            ),
            quant_method=self,
        )
