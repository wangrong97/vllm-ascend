# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GS32 projections over resident source-v3 parameters, without expert I/O.

The serial path remains the validation baseline. Batched execution preserves
native expert/chunk geometry while sharing activation work and stage checks.
"""

from pathlib import Path

import torch

from vllm_ascend.quantization.methods.w2a8.vq2a8_gs32_format import GS32_MAX_ROWS
from vllm_ascend.quantization.methods.w2a8.vq2a8_gs32_staged import quantize_mxfp8_k32, require_finite, staged_swiglu

NATIVE_ALIGNMENT = 32
MAX_GROUPED_EXPERTS = 256  # Imported mixed kernel host validation limit.
MAX_GRAPH_ROUTED_ROWS = 8  # B1 decode with up to eight routed experts (single-rank view).
# DP-attention gathers tokens across DP ranks before MoE: the whole-graph
# capture therefore sees dp_size * B1 * topk routed rows (e.g. DP8 x 1 x 6 =
# 48). The limit can be raised via the "gs32_graph_routed_rows" vq2a8 option
# when the bridge supports dynamic routed rows (full_graph already requires
# supports_dynamic_device_rows).


def initialize_quantization_validity(runner, *, breakable=False):
    """Bind every GS32 layer to status storage shared by either model runner."""
    runner._quantization_validity = None
    runner._quantization_validity_buffers = ()
    config = runner.vllm_config
    options = (config.additional_config or {}).get("vq2a8", {})
    if (
        runner.model_config.quantization != "vq2a8"
        or not getattr(config.quant_config, "is_gs32", False)
        or options.get("gs32_execution", "serial") != "full_graph"
    ):
        return
    if breakable:
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


class _NativeStageOps:
    def __init__(self, library, binary, device, *, execution="serial"):
        # Extension/runtime registration happens once during worker startup,
        # never within a queued per-projection launch callback.
        import torch_npu  # noqa: F401

        self.device = torch.device(device)
        self.execution = execution
        self.plans = {}
        self.grouped_plans = {}
        if self.device.type != "npu" or self.device.index is None:
            raise ValueError("GS32 runtime requires an explicit NPU index.")
        for path in (library, binary):
            if not Path(path).is_file():
                raise ValueError(f"GS32 native artifact is missing: {path}")
        with torch.npu.device(self.device):
            torch.ops.load_library(str(Path(library).resolve()))
            self.ops = torch.ops.vq2a8_gs32_stage
            required = ("prepare", "project_prepared") if execution != "serial" else ()
            if execution in ("fused", "full_graph"):
                required += ("prepare_routes", "prepare_grouped", "project_grouped", "supports_dynamic_grouped_rows")
            if execution in ("full_graph",):
                required += (
                    "prepare_device_routes",
                    "device_routes_status",
                    "project_grouped_device",
                    "supports_dynamic_device_rows",
                    "supports_sparse_prefill_output",
                )
            if not all(hasattr(self.ops, name) for name in required):
                raise RuntimeError(f"GS32 {execution} execution requires rebuilding the GS32 Torch bridge .so.")
            if execution in ("fused", "full_graph"):
                if not self.ops.supports_dynamic_grouped_rows():
                    raise RuntimeError(
                        "GS32 grouped execution requires a bridge supporting dynamic routed rows "
                        "without a fixed per-expert or aggregate row limit; "
                        "rebuild the GS32 Torch bridge .so."
                    )
            if execution in ("full_graph",) and not self.ops.supports_dynamic_device_rows():
                raise RuntimeError("GS32 prefill requires dynamic device routes; rebuild the GS32 Torch bridge .so.")
            if execution in ("full_graph",) and not self.ops.supports_sparse_prefill_output():
                raise RuntimeError("GS32 prefill requires sparse output support; rebuild the GS32 Torch bridge .so.")
            self.ops.initialize(str(Path(binary).resolve()))

    def make_tiling(self, k, n):
        return self.ops.make_tiling(k, n)

    def check_eager(self):
        capturing = getattr(torch.npu, "is_current_stream_capturing", None)
        if capturing is not None and capturing():
            raise RuntimeError("GS32 staged backend does not support graph capture.")

    def project(self, data, scales, weight, lut, row_scale, group_list, mask, tiling, output):
        self.check_eager()
        if self.execution == "serial":
            self.ops.project(data, scales, weight, lut, row_scale, group_list, mask, tiling, output)
            return
        key = (data.shape[0], data.shape[1], output.shape[1])
        plan = self.plans.get(key)
        if plan is None:
            plan = self.ops.prepare(data, output.shape[1])
            self.plans[key] = plan
        self.ops.project_prepared(data, scales, weight, lut, row_scale, plan, output)

    def prepare_routes(self, hidden, counts):
        self.check_eager()
        return self.ops.prepare_routes(hidden, counts)

    def project_grouped(self, data, scales, weight, lut, row_scale, routes, output):
        self.check_eager()
        key = (weight.shape[0], data.shape[1], output.shape[1])
        plan = self.grouped_plans.get(key)
        if plan is None:
            plan = self.ops.prepare_grouped(data, output.shape[1], weight.shape[0])
            self.grouped_plans[key] = plan
        self.ops.project_grouped(data, scales, weight, lut, row_scale, routes, plan, output)

    def prepare_grouped_plan(self, anchor, n, groups):
        self.check_eager()
        return self.ops.prepare_grouped(anchor, n, groups)

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

    def project_grouped_device(self, data, scales, weight, lut, row_scale, routes, plan, output, *, zero_output=True):
        if zero_output:
            self.ops.project_grouped_device(data, scales, weight, lut, row_scale, routes, plan, output)
        else:
            self.ops.project_grouped_device(data, scales, weight, lut, row_scale, routes, plan, output, False)


class GS32ExpertWeights:
    """Resident source-v3 banks with independent GateUp/Down quantization."""

    decode_mode = "grouped"

    def __init__(
        self, layer, layer_spec, options, *, mp_eplb=None, _test_ops=None, _test_activation=None, _test_mxfp4=None
    ):
        self.execution = options.get("gs32_execution", "serial")
        if self.execution not in ("serial", "batched", "fused", "full_graph"):
            raise ValueError("GS32 execution must be serial, batched, fused, or full_graph.")
        # Whole-graph routed-row budget during capture. Default keeps the
        # single-rank B1 contract; DP-attention deployments set it to
        # dp_size * max_query_len * topk via the vq2a8 option.
        self.graph_routed_rows = max(1, int(options.get("gs32_graph_routed_rows", MAX_GRAPH_ROUTED_ROWS)))
        self.fusion = options.get("gs32_fusion", "none")
        if self.fusion not in ("none", "swiglu_quant"):
            raise ValueError("GS32 fusion must be none or swiglu_quant.")
        # Mixed-precision EPLB switch: "off" (default) adds zero behavior.
        from vllm_ascend.eplb.mp.config import OFF_OPTIONS, get_mp_eplb_options

        self.mp_eplb = mp_eplb if mp_eplb is not None else get_mp_eplb_options(options)
        self.eplb_enabled = self.mp_eplb.enabled
        self.activation = _test_activation
        if (self.execution in ("fused", "full_graph") or self.fusion == "swiglu_quant") and self.activation is None:
            from vllm_ascend.ops.fused_moe import vq2a8_gs32_activation

            self.activation = vq2a8_gs32_activation
        self.specs = layer_spec.specs
        self.expert_slots = {expert: slot for slot, expert in enumerate(layer_spec.expert_ids)}
        self.num_experts = layer.vq2a8_num_experts
        self.allow_padding = getattr(layer, "vq2a8_ep_size", 1) > 1
        self.banks = {}
        for kind, prefix in (("gate_up", "w13"), ("down", "w2")):
            self.banks[kind] = {}
            for quant_type in ("vq2", "mxfp4"):
                experts = tuple(
                    sorted(e for e in layer_spec.expert_ids if layer_spec.quant_types[kind][e] == quant_type)
                )
                if not experts:
                    continue
                fields = (
                    {
                        name: getattr(layer, f"{prefix}_{name}")
                        for name in ("packed_zn", "pair_lut", "row_scale_bits", "perm")
                    }
                    if quant_type == "vq2"
                    else {name: getattr(layer, f"{prefix}_mxfp4_{name}") for name in ("weight", "scale")}
                )
                self.banks[kind][quant_type] = {
                    "fields": fields,
                    "expert_ids": experts,
                    "slots": {expert: slot for slot, expert in enumerate(experts)},
                }
        first_bank = next(iter(self.banks["gate_up"].values()), None)
        self.device = next(iter(first_bank["fields"].values())).device if first_bank else layer.vq2a8_anchor.device
        self.ops = (
            _test_ops
            if _test_ops is not None
            else _NativeStageOps(options["gs32_library"], options["gs32_binary"], self.device, execution=self.execution)
        )
        self.mxfp4_project = _test_mxfp4
        if self.mxfp4_project is None and any("mxfp4" in banks for banks in self.banks.values()):
            # Resolve worker-only dependencies before warmup/capture.
            import torch_npu

            from vllm_ascend.device.device_op import DeviceOperator
            from vllm_ascend.quantization.quant_type import QuantType

            self.device_operator = DeviceOperator
            self.mxfp4_dtype = torch_npu.float4_e2m1fn_x2
            self.e8m0_dtype = torch_npu.float8_e8m0fnu
            self.mxfp4_quant_type = QuantType.W4A8MXFP
        self.tilings = {
            kind: self.ops.make_tiling(spec.columns, spec.rows)
            for kind, spec in self.specs.items()
            if "vq2" in self.banks[kind]
        }
        self.route_mask = torch.zeros(1, dtype=torch.int32, device="cpu")
        self.group_counts = {}
        self.defer_validity = False
        self.validity_buffer = torch.zeros((), dtype=torch.bool, device=self.device)
        self.capture_owners = []
        # MP-EPLB: slot-level load accounting lives on the device (raw token
        # rows per logical expert); consumed by the slow ring only (C.5).
        self.moe_load = (
            torch.zeros(self.num_experts, dtype=torch.int64, device=self.device)
            if self.eplb_enabled
            else None
        )
        if self.execution in ("full_graph",) or self.eplb_enabled:
            from vllm_ascend.ops.fused_moe import vq2a8_gs32_routes

            # A typed subset can have padding even on EP1.
            if any("vq2" in banks for banks in self.banks.values()):
                self.ops.require_padding_support()
            if self.num_experts > MAX_GROUPED_EXPERTS or len(self.expert_slots) > MAX_GROUPED_EXPERTS:
                raise ValueError("GS32 full_graph requires at most 256 logical and resident experts.")
            self.device_router = vq2a8_gs32_routes
            self.resident_mask = torch.zeros(self.num_experts, dtype=torch.bool, device=self.device)
            if self.expert_slots:
                self.resident_mask[list(self.expert_slots)] = True
            for kind, banks in self.banks.items():
                for quant_type, bank in banks.items():
                    active = torch.tensor(bank["expert_ids"], dtype=torch.int64, device=self.device)
                    # MP-EPLB capacity: spare tail slots carry the EMPTY(-1)
                    # sentinel; plans (when built) cover the full capacity so
                    # fill/evict never changes the group count (doc D/I3).
                    spare = (
                        self.mp_eplb.replica_slots[quant_type]
                        // max(1, getattr(layer, "vq2a8_ep_size", 1))
                        if self.eplb_enabled
                        else 0
                    )
                    bank["spare_slots"] = spare
                    if spare:
                        active = torch.cat(
                            [active, torch.full((spare,), -1, dtype=torch.int64, device=self.device)]
                        )
                    bank["expert_ids_tensor"] = active
                    bank["logical_slots"] = torch.full((self.num_experts,), -1, dtype=torch.int64, device=self.device)
                    active_ids = torch.tensor(bank["expert_ids"], dtype=torch.int64, device=self.device)
                    bank["logical_slots"][active_ids] = torch.arange(
                        len(bank["expert_ids"]), device=self.device
                    )
                    if quant_type == "vq2" and self.execution in ("full_graph",):
                        spec = self.specs[kind]
                        bank["plan"] = self.ops.prepare_grouped_plan(
                            torch.empty((1, spec.columns), dtype=torch.uint8, device=self.device),
                            spec.rows,
                            len(bank["expert_ids"]) + spare,
                        )

    def _accept_validity(self, status):
        self.validity_buffer.copy_(status.reshape(()).bool())
        if not self.defer_validity and not bool(self.validity_buffer.cpu()):
            raise ValueError("GS32 device preparation or projection failed validity checks.")

    def _mark_eager_valid(self):
        if self.execution in ("full_graph",):
            self.validity_buffer.fill_(True)

    def _prepare_activation(
        self, hidden, bank, slots, *, swiglu_limit=None, unchecked=False, source_rows=None, route_flags=None
    ):
        prepare = self.activation.prepare_unchecked if unchecked else self.activation.prepare
        options = {} if swiglu_limit is None else {"swiglu_limit": swiglu_limit}
        if source_rows is not None:
            options["source_rows"] = source_rows
        if route_flags is not None:
            options["route_flags"] = route_flags
        return prepare(hidden, bank["fields"].get("perm"), slots, **options)

    def _mxfp4(self, data, scales, bank, counts):
        if self.mxfp4_project is not None:
            return self.mxfp4_project(data, scales, bank["fields"], counts)
        return self.device_operator.npu_grouped_matmul_gmm2(
            hidden_states=data.view(torch.float8_e4m3fn),
            weight=bank["fields"]["weight"],
            weight_scale=bank["fields"]["scale"],
            per_token_scale=scales.reshape(data.shape[0], data.shape[1] // 64, 2),
            group_list=counts,
            group_list_type=1,
            input_dtype=torch.bfloat16,
            act_quant_type=torch.float8_e4m3fn,
            weight_quant_type=self.mxfp4_dtype,
            scale_type=None,
            per_token_scale_type=self.e8m0_dtype,
            use_bf16=True,
            use_mxfp_quant=True,
            bias=None,
            mxfp_quant_dtype=self.mxfp4_quant_type,
        )

    @staticmethod
    def _restore(output, inverse):
        selected = inverse >= 0
        return torch.where(selected[:, None], output.index_select(0, inverse.clamp_min(0)), 0)

    def _project_device(self, kind, hidden, group_list, group_list_type, *, swiglu_limit=None):
        spec = self.specs[kind]
        result = hidden.new_zeros((hidden.shape[0], spec.rows))
        status = torch.ones((), dtype=torch.bool, device=self.device)
        retained = [result]
        for quant_type, bank in self.banks[kind].items():
            physical, counts, slots, inverse, route_flags = self.device_router.prepare(
                hidden,
                group_list,
                group_list_type,
                bank["expert_ids_tensor"],
                bank["logical_slots"],
                allow_padding=True,
                subset=True,
            )
            data, scales, flags = self._prepare_activation(
                physical, bank, slots, swiglu_limit=swiglu_limit, unchecked=True
            )
            retained.extend((physical, counts, slots, inverse, route_flags, data, scales, flags))
            status = status & (route_flags == 1).all() & (flags == 1).all()
            if quant_type == "vq2":
                fields = bank["fields"]
                routes = self.ops.prepare_device_routes(physical, counts, allow_padding=True)
                native_flags = self.ops.device_routes_status(routes)
                output = torch.full(
                    (hidden.shape[0], spec.rows), float("nan"), device=self.device, dtype=torch.bfloat16
                )
                self.ops.project_grouped_device(
                    data,
                    scales,
                    fields["packed_zn"],
                    fields["pair_lut"],
                    fields["row_scale_bits"],
                    routes,
                    bank["plan"],
                    output,
                )
                status = status & (native_flags == 1).all()
                retained.extend((routes, native_flags))
            else:
                output = self._mxfp4(data, scales, bank, counts)
            # CANN's single-bank GMM permits sum(counts)<=M on A5. Its
            # uncomputed tail is unspecified: mask BEFORE finiteness checks.
            live = torch.arange(hidden.shape[0], device=self.device) < counts.sum()
            clean = torch.where(live[:, None], output, 0)
            restored = self._restore(clean, inverse)
            status = status & torch.isfinite(clean).all()
            result.add_(restored)
            retained.extend((output, live, clean, restored))
        return result, status, retained

    def _apply_device(self, hidden, group_list, group_list_type, limit, capturing):
        # Validate the full resident inventory as well as each typed subset.
        bounded = group_list.clamp(0, hidden.shape[0])
        counts = bounded if group_list_type == 1 else bounded - torch.cat((bounded.new_zeros(1), bounded[:-1]))
        status = ((group_list >= 0) & (group_list <= hidden.shape[0])).all() & (counts >= 0).all()
        status = status & ((counts == 0) | self.resident_mask).all()
        total = counts.sum()
        status = status & ((total <= hidden.shape[0]) if self.allow_padding else (total == hidden.shape[0]))
        if not self.expert_slots:
            self._accept_validity(status & (total == 0))
            return hidden.new_zeros((hidden.shape[0], self.specs["down"].rows))
        gate_up, gate_status, retained = self._project_device("gate_up", hidden, group_list, group_list_type)
        # Check the raw result before clipping can hide a non-finite value.
        raw_finite = torch.isfinite(gate_up).all()
        if self.fusion == "swiglu_quant":
            # Route the natural GateUp rows by the DOWN bank. Its preparation
            # fuses SwiGLU, the down permutation, BF16 rounding and MXFP8.
            middle = None
            output, down_status, down_retained = self._project_device(
                "down", gate_up, group_list, group_list_type, swiglu_limit=limit
            )
        else:
            middle = self.activation.swiglu_unchecked(gate_up, limit)
            output, down_status, down_retained = self._project_device("down", middle, group_list, group_list_type)
            status = status & torch.isfinite(middle).all()
        status = status & gate_status & raw_finite & down_status
        self._accept_validity(status)
        if capturing:
            # Retain each graph's tensors: earlier captures may still replay.
            # Replay bypasses this Python path and does not append owners.
            self.capture_owners.append((*retained, gate_up, raw_finite, middle, *down_retained, status, output))
        return output

    def _group_count(self, rows):
        count = self.group_counts.get(rows)
        if count is None:
            count = torch.tensor([rows], dtype=torch.int64, device="cpu")
            self.group_counts[rows] = count
        return count

    def _project_prefill(self, kind, hidden, group_list, group_type, metadata, native_routes, *, swiglu_limit=None):
        """Gather while quantizing, then restore directly into logical rows.

        Metadata is shared only within this forward and only by identical
        ordered expert inventories. GateUp and Down can have different banks.
        No eager prefill buffers are retained by the decode graph owner.
        """
        spec = self.specs[kind]
        result = hidden.new_zeros((hidden.shape[0], spec.rows))
        status = torch.ones((), dtype=torch.bool, device=self.device)
        for quant_type, bank in self.banks[kind].items():
            key = bank["expert_ids"]
            if key not in metadata:
                metadata[key] = self.device_router.prepare_metadata(
                    hidden,
                    group_list,
                    group_type,
                    bank["expert_ids_tensor"],
                    bank["logical_slots"],
                    allow_padding=self.allow_padding,
                    subset=True,
                )
            counts, slots, gather, inverse, route_flags, restore_metadata = metadata[key]
            data, scales, flags = self._prepare_activation(
                hidden,
                bank,
                slots,
                swiglu_limit=swiglu_limit,
                unchecked=True,
                source_rows=gather,
                route_flags=route_flags,
            )
            status = status & (flags == 1).all()
            if quant_type == "vq2":
                fields = bank["fields"]
                if key not in native_routes:
                    routes = self.ops.prepare_device_routes(data, counts, allow_padding=True)
                    native_routes[key] = routes, (self.ops.device_routes_status(routes) == 1).all()
                routes, native_status = native_routes[key]
                output = hidden.new_empty((hidden.shape[0], spec.rows))
                self.ops.project_grouped_device(
                    data,
                    scales,
                    fields["packed_zn"],
                    fields["pair_lut"],
                    fields["row_scale_bits"],
                    routes,
                    bank["plan"],
                    output,
                    zero_output=False,
                )
                status = status & native_status
            else:
                output = self._mxfp4(data, scales, bank, counts)
            # The native/GMM unused tail is unspecified. Restore masks it
            # before reading and checks selected output values in the same pass.
            restored_flags = self.device_router.restore(
                output, inverse, counts, route_flags, result, metadata=restore_metadata
            )
            status = status & (restored_flags == 1).all()
            # Release large eager buffers before allocating the next bank.
            # Native launches record stream ownership; no buffers escape into
            # the decode graph's retained owners.
            del data, scales, flags, output, restored_flags
        return result, status

    def _apply_prefill(self, hidden, group_list, group_type, limit):
        # Typed metadata validates ranges, prefixes and total capacity. This
        # additional inventory check catches dispatch to a nonresident expert.
        groups = group_list.to(torch.int64)
        bounded = groups.clamp(0, hidden.shape[0])
        counts = bounded if group_type == 1 else bounded - torch.cat((bounded.new_zeros(1), bounded[:-1]))
        status = ((counts == 0) | self.resident_mask).all()
        if hidden.shape[0] == 0 or not self.expert_slots:
            self._accept_validity(status & (groups == 0).all())
            return hidden.new_zeros((hidden.shape[0], self.specs["down"].rows))
        metadata, native_routes = {}, {}
        gate_up, gate_status = self._project_prefill("gate_up", hidden, group_list, group_type, metadata, native_routes)
        if self.fusion == "swiglu_quant":
            output, down_status = self._project_prefill(
                "down", gate_up, group_list, group_type, metadata, native_routes, swiglu_limit=limit
            )
        else:
            middle = self.activation.swiglu_unchecked(gate_up, limit)
            # Down preparation checks every live column (permutations are
            # validated as bijections when weights load), so no extra full
            # BF16 scan is needed here. GateUp corruption remains in gate_status.
            output, down_status = self._project_prefill("down", middle, group_list, group_type, metadata, native_routes)
        # Raw GateUp nonfinite values were flagged before restore sanitized
        # them, so clipping cannot hide corruption. Reuse the forward owner's
        # existing deferred check, or perform one check for a standalone call.
        self._accept_validity(status & gate_status & down_status)
        return output

    def _bank_for_expert(self, kind, expert):
        for quant_type, bank in self.banks[kind].items():
            if expert in bank["slots"]:
                return quant_type, bank, bank["slots"][expert]
        raise ValueError(f"Missing {kind} projection for logical expert {expert}.")

    def _project_quantized(self, kind, expert, data, scales, output):
        quant_type, bank, slot = self._bank_for_expert(kind, expert)
        fields = bank["fields"]
        if quant_type == "mxfp4":
            # Cover the MP-EPLB spare tail (zero token counts) so the group
            # list matches the packed weight rows (experts + spare), mirroring
            # the fused path; without EPLB spare_slots is absent and stays 0.
            counts = torch.zeros(
                len(bank["expert_ids"]) + bank.get("spare_slots", 0), dtype=torch.int64, device=self.device
            )
            counts[slot] = data.shape[0]
            output.copy_(self._mxfp4(data, scales, bank, counts))
        else:
            if scales.data_ptr() % NATIVE_ALIGNMENT:
                scales = scales.clone()
            self.ops.project(
                data,
                scales,
                fields["packed_zn"][slot : slot + 1],
                fields["pair_lut"][slot : slot + 1],
                fields["row_scale_bits"][slot : slot + 1],
                self._group_count(data.shape[0]),
                self.route_mask,
                self.tilings[kind],
                output,
            )

    def _project(self, kind, expert, hidden, *, swiglu_limit=None):
        self.ops.check_eager()
        _, bank, slot = self._bank_for_expert(kind, expert)
        if swiglu_limit is None:
            perm = bank["fields"].get("perm")
            transformed = hidden if perm is None else hidden.index_select(1, perm[slot])
            data, scales = quantize_mxfp8_k32(transformed)
        else:
            slots = torch.full((hidden.shape[0],), slot, dtype=torch.int64, device=self.device)
            data, scales = self._prepare_activation(hidden, bank, slots, swiglu_limit=swiglu_limit)
        output = torch.full(
            (hidden.shape[0], self.specs[kind].rows), float("nan"), device=self.device, dtype=torch.bfloat16
        )
        self._project_quantized(kind, expert, data, scales, output)
        require_finite(output, "GS32 projection output")
        return output

    def _project_batched(self, kind, hidden, jobs, *, swiglu_limit=None):
        if swiglu_limit is not None:
            output = torch.full(
                (hidden.shape[0], self.specs[kind].rows), float("nan"), device=self.device, dtype=torch.bfloat16
            )
            # Prepare once per quantization bank, not once per 128-row chunk.
            # This bounds eager validity transfers even for a large prefill.
            for bank in self.banks[kind].values():
                selected = [job for job in jobs if job[0] in bank["slots"]]
                if not selected:
                    continue
                physical = torch.cat([hidden[first:last] for _, first, last in selected])
                slots = torch.tensor(
                    [bank["slots"][expert] for expert, first, last in selected for _ in range(last - first)],
                    dtype=torch.int64,
                    device=self.device,
                )
                data, scales = self._prepare_activation(physical, bank, slots, swiglu_limit=swiglu_limit)
                offset = 0
                for expert, first, last in selected:
                    end = offset + last - first
                    self._project_quantized(kind, expert, data[offset:end], scales[offset:end], output[first:last])
                    offset = end
            require_finite(output, f"GS32 {kind} projection output")
            return output
        transformed = []
        for expert, first, last in jobs:
            _, bank, slot = self._bank_for_expert(kind, expert)
            perm = bank["fields"].get("perm")
            rows = hidden[first:last]
            transformed.append(rows if perm is None else rows.index_select(1, perm[slot]))
        data, scales = quantize_mxfp8_k32(torch.cat(transformed))
        output = torch.full(
            (hidden.shape[0], self.specs[kind].rows), float("nan"), device=self.device, dtype=torch.bfloat16
        )
        for expert, first, last in jobs:
            self._project_quantized(kind, expert, data[first:last], scales[first:last], output[first:last])
        require_finite(output, f"GS32 {kind} projection output")
        return output

    def _project_grouped(self, kind, hidden, row_slots, routes, *, bank=None, swiglu_limit=None):
        bank = self.banks[kind]["vq2"] if bank is None else bank
        fields, spec = bank["fields"], self.specs[kind]
        data, scales = self._prepare_activation(hidden, bank, row_slots, swiglu_limit=swiglu_limit)
        output = torch.full((hidden.shape[0], spec.rows), float("nan"), device=self.device, dtype=torch.bfloat16)
        self.ops.project_grouped(
            data, scales, fields["packed_zn"], fields["pair_lut"], fields["row_scale_bits"], routes, output
        )
        return output

    def _project_fused(self, kind, hidden, counts, *, swiglu_limit=None):
        offsets = [0]
        for count in counts:
            offsets.append(offsets[-1] + count)
        result = hidden.new_zeros((hidden.shape[0], self.specs[kind].rows))
        for quant_type, bank in self.banks[kind].items():
            physical_counts = [counts[e] for e in bank["expert_ids"]]
            if not sum(physical_counts):
                continue
            order = [row for e in bank["expert_ids"] for row in range(offsets[e], offsets[e + 1])]
            indices = torch.tensor(order, dtype=torch.int64, device=self.device)
            physical = hidden.index_select(0, indices)
            slots = torch.tensor(
                [s for s, count in enumerate(physical_counts) for _ in range(count)],
                dtype=torch.int64,
                device=self.device,
            )
            if quant_type == "vq2":
                # MP-EPLB appends spare (redundancy) slots to the packed bank,
                # so plan->Groups() == pair_lut.size(0) counts resident slots,
                # not just experts. Zero-pad the group counts to the resident
                # capacity so GroupedRoutes::CheckMatches sees the same group
                # count as the projection plan.
                counts_tensor = torch.tensor(physical_counts, dtype=torch.int64, device="cpu")
                spare = bank.get("spare_slots", 0)
                if spare:
                    counts_tensor = torch.cat(
                        [counts_tensor, torch.zeros(spare, dtype=torch.int64)]
                    )
                routes = self.ops.prepare_routes(physical, counts_tensor)
                output = self._project_grouped(kind, physical, slots, routes, bank=bank, swiglu_limit=swiglu_limit)
            else:
                data, scales = self._prepare_activation(physical, bank, slots, swiglu_limit=swiglu_limit)
                # Same spare-slot capacity applies to the mxfp4 bank: the
                # grouped matmul weight holds len(experts)+spare rows, so the
                # group list must cover the spare tail (zero token counts).
                counts_dev = torch.tensor(physical_counts, dtype=torch.int64, device=self.device)
                spare = bank.get("spare_slots", 0)
                if spare:
                    counts_dev = torch.cat(
                        [counts_dev, torch.zeros(spare, dtype=torch.int64, device=self.device)]
                    )
                output = self._mxfp4(data, scales, bank, counts_dev)
            self.activation.require_finite_fused(output, f"GS32 {kind} output")
            result.index_copy_(0, indices, output)
        return result

    def _apply_fused(self, hidden, counts, limit):
        gate_up = self._project_fused("gate_up", hidden, counts)
        if self.fusion == "swiglu_quant":
            return self._project_fused("down", gate_up, counts, swiglu_limit=limit)
        middle = self.activation.swiglu_unchecked(gate_up, limit)
        self.activation.require_finite_fused(middle, "GS32 SwiGLU output")
        return self._project_fused("down", middle, counts)

    @torch.inference_mode()
    def apply(self, hidden_states, group_list, group_list_type, swiglu_limit):
        limit = 0.0 if swiglu_limit is None else swiglu_limit
        fused_limit = limit if self.fusion == "swiglu_quant" else None
        capturing = False
        graph_rows = self.graph_routed_rows
        if self.execution in ("full_graph",):
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
        if self.execution in ("full_graph",) and group_list.device != self.device:
            raise ValueError("GS32 graph group_list must be on the same device as hidden states.")
        if self.eplb_enabled:
            # MP-EPLB accounting (doc C.5 raw quantities): accumulate this
            # step's logical-expert row counts; consumed by the slow ring.
            counts = group_list.to(device=self.device, dtype=torch.int64).clamp_min(0)
            if group_list_type == 0:
                counts = counts - torch.cat((counts.new_zeros(1), counts[:-1]))
            self.moe_load[: counts.numel()].add_(counts)
        if self.execution in ("full_graph",) and 0 < hidden_states.shape[0] <= graph_rows:
            return self._apply_device(
                hidden_states, group_list, group_list_type, 0.0 if swiglu_limit is None else swiglu_limit, capturing
            ), None
        if capturing:
            raise RuntimeError("GS32 full_graph requires a nonempty B1 decode shape.")
        if self.execution in ("full_graph",):
            return self._apply_prefill(hidden_states, group_list, group_list_type, limit), None
        counts = group_list.cpu().tolist()
        if group_list_type == 0:
            counts = [end - start for end, start in zip(counts, [0, *counts[:-1]])]
        capacity, active_rows = hidden_states.shape[0], sum(counts)
        if (
            any(count < 0 for count in counts)
            or active_rows > capacity
            or (not self.allow_padding and active_rows != capacity)
            or any(count and expert not in self.expert_slots for expert, count in enumerate(counts))
        ):
            raise ValueError("GS32 group counts or routed expert inventory are invalid.")
        if not active_rows:
            self._mark_eager_valid()
            return hidden_states.new_zeros((capacity, self.specs["down"].rows)), None
        hidden_states = hidden_states[:active_rows]

        def finish(output):
            self._mark_eager_valid()
            if active_rows == capacity:
                return output, None
            padded = output.new_zeros((capacity, output.shape[1]))
            padded[:active_rows].copy_(output)
            return padded, None

        if self.execution in ("fused", "full_graph") and len(self.expert_slots) <= MAX_GROUPED_EXPERTS:
            return finish(self._apply_fused(hidden_states, counts, 0.0 if swiglu_limit is None else swiglu_limit))
        if self.execution in ("batched", "fused", "full_graph"):
            jobs, start = [], 0
            for expert, count in enumerate(counts):
                for offset in range(0, count, GS32_MAX_ROWS):
                    jobs.append((expert, start + offset, start + min(count, offset + GS32_MAX_ROWS)))
                start += count
            gate_up = self._project_batched("gate_up", hidden_states, jobs)
            if fused_limit is not None:
                return finish(self._project_batched("down", gate_up, jobs, swiglu_limit=fused_limit))
            middle = staged_swiglu(gate_up, 0.0 if swiglu_limit is None else swiglu_limit)
            return finish(self._project_batched("down", middle, jobs))
        output = hidden_states.new_empty((hidden_states.shape[0], self.specs["down"].rows))
        start = 0
        for expert, count in enumerate(counts):
            for offset in range(0, count, GS32_MAX_ROWS):
                first, last = start + offset, start + min(count, offset + GS32_MAX_ROWS)
                gate_up = self._project("gate_up", expert, hidden_states[first:last])
                if fused_limit is not None:
                    result = self._project("down", expert, gate_up, swiglu_limit=fused_limit)
                else:
                    middle = staged_swiglu(gate_up, limit)
                    result = self._project("down", expert, middle)
                output[first:last].copy_(result)
            start += count
        return finish(output)

    # -- MP-EPLB P2 commit hooks (value-only index refresh; doc 2.5) ---------

    def _bank_of(self, kind, quant_type):
        bank = self.banks.get(kind, {}).get(quant_type)
        if bank is None or "expert_ids_tensor" not in bank:
            raise ValueError(f"MP-EPLB commit requires device indexes for {kind}/{quant_type}.")
        return bank

    def commit_fill(self, quant_type, slot, expert):
        """P2 for a fill move on this rank (both projections, same slot)."""
        for kind in ("gate_up", "down"):
            bank = self._bank_of(kind, quant_type)
            if int(bank["expert_ids_tensor"][slot]) != -1:
                raise ValueError("commit_fill targets an occupied slot.")
            bank["expert_ids_tensor"][slot] = expert
        bank = self._bank_of("gate_up", quant_type)
        bank["logical_slots"][expert] = slot
        self.resident_mask[expert] = True

    def commit_evict(self, quant_type, slot, expert):
        """P2 for an evict move (index only; weight bytes stay resident)."""
        for kind in ("gate_up", "down"):
            bank = self._bank_of(kind, quant_type)
            if int(bank["expert_ids_tensor"][slot]) != expert:
                raise ValueError("commit_evict slot mismatch.")
            bank["expert_ids_tensor"][slot] = -1
        bank = self._bank_of("gate_up", quant_type)
        bank["logical_slots"][expert] = -1
        self.resident_mask[expert] = False

    def commit_swap(self, quant_type, slot, incoming_expert):
        """P2 for the local end of a swap: outgoing expert leaves this rank."""
        bank = self._bank_of("gate_up", quant_type)
        outgoing = int(bank["expert_ids_tensor"][slot])
        if outgoing < 0:
            raise ValueError("commit_swap targets an empty slot.")
        for kind in ("gate_up", "down"):
            self._bank_of(kind, quant_type)["expert_ids_tensor"][slot] = incoming_expert
        bank["logical_slots"][outgoing] = -1
        bank["logical_slots"][incoming_expert] = slot
        self.resident_mask[outgoing] = False
        self.resident_mask[incoming_expert] = True
        return outgoing
