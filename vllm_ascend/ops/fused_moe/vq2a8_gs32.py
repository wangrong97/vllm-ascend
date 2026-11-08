# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Complete GS32 FFN execution over resident mixed W2/W4 expert weights."""

from pathlib import Path

import torch

from vllm_ascend.quantization.methods.w2a8.vq2a8_gs32_contract import resolve_gs32_options

NATIVE_ALIGNMENT = 32
MAX_GROUPED_EXPERTS = 256  # Imported mixed kernel host validation limit.
DEFAULT_GRAPH_ROW_CAPACITY = 8  # One decode query with top-k up to eight.


def initialize_quantization_validity(runner, *, breakable=False):
    """Bind every GS32 layer to status storage shared by either model runner."""
    runner._quantization_validity = None
    runner._quantization_validity_buffers = ()
    config = runner.vllm_config
    options = (config.additional_config or {}).get("vq2a8", {})
    if runner.model_config.quantization != "vq2a8" or not getattr(config.quant_config, "is_gs32", False):
        return
    resolved = resolve_gs32_options(options)
    if breakable and resolved["gs32_execution"] == "full_graph":
        raise ValueError("GS32 full decode graphs require VLLM_USE_BREAKABLE_CUDAGRAPH=0.")
    states = {}
    for module in runner.model.modules():
        weights = getattr(module, "vq2a8_weights", None)
        if weights is None:
            continue
        index = module.vq2a8_layer_index
        if index in states and states[index] is not weights:
            raise RuntimeError(f"Duplicate quantization validity owner for layer {index}.")
        states[index] = weights
    if set(states) != set(config.quant_config.layers):
        raise RuntimeError("Quantization validity must cover every configured expert layer.")
    owners = tuple(states[index] for index in sorted(states))
    if len({id(weights) for weights in owners}) != len(owners):
        raise RuntimeError("Quantization validity owners must be distinct for each expert layer.")
    for weights in owners:
        flag = weights.validity_buffer
        if (
            not isinstance(flag, torch.Tensor)
            or flag.dtype != torch.bool
            or flag.shape != ()
            or flag.device != runner.device
        ):
            raise RuntimeError("Quantization validity requires a device-resident boolean scalar per layer.")
    # Validate all owners before mutating any. Stable views live outside graph
    # pools; zeroing at each forward detects missed writes on graph replay.
    runner._quantization_validity = torch.zeros(len(owners), dtype=torch.bool, device=runner.device)
    runner._quantization_validity_buffers = runner._quantization_validity.unbind()
    for weights, flag in zip(owners, runner._quantization_validity_buffers):
        weights.validity_buffer = flag
        weights.defer_validity = True


def check_quantization_validity(runner):
    """Consume one aggregate device status before sampling, never in capture."""
    validity = getattr(runner, "_quantization_validity", None)
    if validity is not None and not bool(validity.all()):
        raise RuntimeError("Quantized expert forward failed validity checks.")


class _NativeFullFFNOps:
    def __init__(self, library, binary, device, *, execution):
        # Register and validate the complete-FFN runtime once during startup.
        import torch_npu  # noqa: F401

        from vllm_ascend.quantization.vq2a8_gs32_runtime import (
            verify_registered_device_schemas,
            verify_runtime_artifacts,
        )

        self.device = torch.device(device)
        if self.device.type != "npu" or self.device.index is None:
            raise ValueError("GS32 runtime requires an explicit NPU index.")
        with torch.npu.device(self.device):
            verify_runtime_artifacts(Path(library), Path(binary), require_full_ffn=True)
            torch.ops.load_library(str(Path(library).resolve()))
            # The bridge owns the schema; its device count validator is Triton.
            from vllm_ascend.ops.fused_moe import vq2a8_gs32_counts  # noqa: F401

            verify_registered_device_schemas(torch, execution, fusion="full_ffn")
            self.ops = torch.ops.vq2a8_gs32_stage
            self.ops.initialize(str(Path(binary).resolve()))

    def prepare_full_ffn_plan(self, anchor, gate_n, output_d, gate_w4, down_w4, limit, fields):
        self.check_eager()
        return self.ops.prepare_full_ffn(
            anchor,
            gate_n,
            output_d,
            gate_w4,
            down_w4,
            limit,
            fields["gate_weight"],
            fields["gate_table"],
            fields["gate_row_scale"],
            fields["down_weight"],
            fields["down_table"],
            fields["down_row_scale"],
        )

    def full_ffn_device(self, data, scales, routes, plan, middle, middle_scales, output):
        self.ops.full_ffn_device(
            data,
            scales,
            routes,
            plan,
            middle,
            middle_scales,
            output,
        )

    def check_eager(self):
        capturing = getattr(torch.npu, "is_current_stream_capturing", None)
        if capturing is not None and capturing():
            raise RuntimeError("GS32 eager execution cannot run during graph capture.")

    def require_padding_support(self):
        schema = torch._C._dispatch_find_schema_or_throw("vq2a8_gs32_stage::prepare_device_routes", "").schema()
        if "bool allow_padding=False" not in str(schema):
            raise RuntimeError("GS32 EP graph execution requires rebuilding the GS32 Torch bridge .so for padding.")

    def prepare_device_routes(self, hidden, counts, *, allow_padding=False):
        if allow_padding:
            return self.ops.prepare_device_routes(hidden, counts, True)
        return self.ops.prepare_device_routes(hidden, counts)

    def device_routes_status(self, routes):
        return self.ops.device_routes_status(routes)


class GS32ExpertWeights:
    """Resident mixed expert weights with one complete-FFN compute path."""

    def __init__(
        self,
        layer,
        layer_spec,
        options,
        *,
        graph_row_capacity=DEFAULT_GRAPH_ROW_CAPACITY,
        startup_state=None,
        _test_ops=None,
        _test_activation=None,
    ):
        if type(graph_row_capacity) is not int or graph_row_capacity < 1:
            raise ValueError("GS32 graph routed row capacity must be a positive integer.")
        # The serving configuration derives this from the largest decode graph
        # bucket times top-k. It does not limit ordinary eager prefill rows.
        self.graph_row_capacity = graph_row_capacity
        options = resolve_gs32_options(options)
        self.execution = options["gs32_execution"]
        self.backend = options["gs32_backend"]
        self.fusion = options["gs32_fusion"]
        self.activation = _test_activation
        if self.activation is None:
            from vllm_ascend.ops.fused_moe import vq2a8_gs32_activation

            self.activation = vq2a8_gs32_activation
        self.specs = layer_spec.specs
        self.expert_slots = {expert: slot for slot, expert in enumerate(layer_spec.expert_ids)}
        self.num_experts = layer.vq2a8_num_experts
        self.allow_padding = getattr(layer, "vq2a8_ep_size", 1) > 1
        self._initialize_full_ffn(layer, options, _test_ops, startup_state)

    def _initialize_full_ffn(self, layer, options, test_ops, startup_state):
        from vllm_ascend.ops.fused_moe import vq2a8_gs32_routes

        self.full_ffn_bank = layer.vq2a8_full_ffn_bank
        bank = self.full_ffn_bank
        if bank["expert_ids"] != tuple(self.expert_slots):
            raise ValueError("Full FFN resident expert order does not match the loaded layer.")
        if not 1 <= self.num_experts <= MAX_GROUPED_EXPERTS or len(self.expert_slots) > MAX_GROUPED_EXPERTS:
            raise ValueError("GS32 full_ffn requires 1..256 logical and at most 256 resident experts.")
        resident_ids = bank["expert_ids"]
        if any(type(expert) is not int or not 0 <= expert < self.num_experts for expert in resident_ids) or tuple(
            resident_ids
        ) != tuple(sorted(set(resident_ids))):
            raise ValueError("Full FFN identity routes require sorted unique resident expert IDs in logical range.")
        self.device = layer.vq2a8_anchor.device
        self.device_router = vq2a8_gs32_routes
        if test_ops is not None:
            self.ops = test_ops
        elif startup_state is not None:
            self.ops = startup_state.native_ops(
                options["gs32_library"],
                options["gs32_binary"],
                self.device,
                execution=self.execution,
                factory=_NativeFullFFNOps,
            )
        else:
            self.ops = _NativeFullFFNOps(
                options["gs32_library"], options["gs32_binary"], self.device, execution=self.execution
            )
        self.ops.require_padding_support()
        self.defer_validity = False
        self.validity_buffer = torch.zeros((), dtype=torch.bool, device=self.device)
        self.capture_owners = []
        self.full_ffn_submissions = 0  # Python submissions, including capture; not a graph replay counter.
        self.full_ffn_graph_captures = 0
        self.full_ffn_plan = None
        self.full_ffn_limit = bank["swiglu_limit"]
        self.full_ffn_expert_ids = torch.tensor(bank["expert_ids"], dtype=torch.int64, device=self.device)
        self.full_ffn_logical_slots = torch.full((self.num_experts,), -1, dtype=torch.int64, device=self.device)
        if self.expert_slots:
            self.full_ffn_logical_slots[self.full_ffn_expert_ids] = torch.arange(
                len(self.expert_slots), device=self.device
            )
            self.full_ffn_plan = self.ops.prepare_full_ffn_plan(
                torch.empty((1, self.specs["gate_up"].columns), dtype=torch.uint8, device=self.device),
                self.specs["gate_up"].rows,
                self.specs["down"].rows,
                bank["gate_w4"],
                bank["down_w4"],
                self.full_ffn_limit,
                bank["fields"],
            )

    def _apply_full_ffn(self, hidden, group_list, group_type, limit, capturing):
        if limit != self.full_ffn_limit:
            raise ValueError("Full FFN SwiGLU limit must match the immutable loaded plan.")
        if group_list.device != self.device:
            group_list = group_list.to(self.device)
        retained = [hidden, group_list] if capturing else None
        if not hidden.shape[0] or not self.expert_slots:
            result = hidden.new_zeros((hidden.shape[0], self.specs["down"].rows))
            status = (group_list == 0).all()
            if not self.allow_padding and hidden.shape[0]:
                status = status & False
            self._accept_validity(status)
            if capturing:
                self.capture_owners.append((*retained, result, status))
            return result
        # Sorted resident IDs and zero counts for every missing logical expert
        # make the dispatched live prefix already match the packed bank order.
        counts, route_flags, route_metadata = self.device_router.prepare_identity_metadata(
            hidden,
            group_list,
            group_type,
            self.full_ffn_expert_ids,
            self.full_ffn_logical_slots,
            allow_padding=self.allow_padding,
            retained=retained,
        )
        data, scales, flags = self.activation.prepare_unchecked(
            hidden, route_flags=route_flags, route_metadata=route_metadata
        )
        routes = self.ops.prepare_device_routes(data, counts, allow_padding=True)
        native_flags = self.ops.device_routes_status(routes)
        rows, width = hidden.shape[0], self.specs["down"].columns
        middle = torch.empty((rows, width), dtype=torch.uint8, device=self.device)
        middle_scales = torch.empty((rows, width // NATIVE_ALIGNMENT), dtype=torch.uint8, device=self.device)
        output = hidden.new_empty((rows, self.specs["down"].rows))
        self.ops.full_ffn_device(
            data,
            scales,
            routes,
            self.full_ffn_plan,
            middle,
            middle_scales,
            output,
        )
        self.full_ffn_submissions += 1
        # Preserve finite-output sanitization and all preparation/scale checks
        # while writing the stable per-layer status directly on device.
        block_flags = self.activation.finish_full_ffn(
            output,
            middle_scales,
            flags,
            route_flags,
            native_flags,
            route_metadata,
            self.validity_buffer,
        )
        self._check_validity()
        if capturing:
            self.full_ffn_graph_captures += 1
            self.capture_owners.append(
                (
                    *retained,
                    counts,
                    route_flags,
                    route_metadata,
                    data,
                    scales,
                    flags,
                    routes,
                    native_flags,
                    middle,
                    middle_scales,
                    output,
                    block_flags,
                    self.validity_buffer,
                )
            )
        # Eager workspaces die here; the current stream/queued launch owns them
        # until completion. Only actual captures retain per-graph storage.
        return output

    def _accept_validity(self, status):
        self.validity_buffer.copy_(status.reshape(()).bool())
        self._check_validity()

    def _check_validity(self):
        if not self.defer_validity and not bool(self.validity_buffer.cpu()):
            raise ValueError("GS32 device preparation or projection failed validity checks.")

    @torch.inference_mode()
    def apply(self, hidden_states, group_list, group_list_type, swiglu_limit):
        limit = 0.0 if swiglu_limit is None else swiglu_limit
        capturing = False
        graph_rows = self.graph_row_capacity
        if self.execution == "full_graph":
            capture_query = getattr(torch.npu, "is_current_stream_capturing", None)
            capturing = capture_query is not None and capture_query()
            if capturing and not self.defer_validity:
                raise RuntimeError("GS32 whole-graph capture requires a forward owner for deferred validity.")
            if capturing and hidden_states.shape[0] > graph_rows:
                raise RuntimeError(f"GS32 whole-graph execution supports at most {graph_rows} routed rows.")
        else:
            self.ops.check_eager()
        if (
            hidden_states.ndim != 2
            or hidden_states.dtype != torch.bfloat16
            or hidden_states.device != self.device
            or hidden_states.shape[1] != self.specs["gate_up"].columns
        ):
            raise ValueError("GS32 expects dispatched resident BF16 [tokens,hidden].")
        if (
            group_list_type not in (0, 1)
            or group_list.ndim != 1
            or group_list.dtype not in (torch.int32, torch.int64)
            or group_list.numel() != self.num_experts
        ):
            raise ValueError("GS32 group_list must have an integer count/prefix for every logical expert.")
        if self.execution == "full_graph" and group_list.device != self.device:
            raise ValueError("GS32 graph group_list must be on the same device as hidden states.")
        return self._apply_full_ffn(hidden_states, group_list, group_list_type, limit, capturing), None
