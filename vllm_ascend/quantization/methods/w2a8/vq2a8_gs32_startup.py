# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-owned GS32 startup validation and worker-local native initialization."""

from collections.abc import Mapping
from enum import Enum
from os import getpid
from pathlib import Path

from .vq2a8_gs32_dspark import DSPARK_GEOMETRY_FIELDS


def _snapshot(value):
    """Freeze configuration data, without copying tensor/checkpoint inventories."""
    kind = type(value)
    if value is None or kind in (bool, int, float, str, bytes):
        return kind, value
    if isinstance(value, Enum):
        return kind, value.name
    if isinstance(value, Mapping):
        return kind, tuple(
            sorted(((key, _snapshot(item)) for key, item in value.items()), key=lambda pair: repr(pair[0]))
        )
    if isinstance(value, (tuple, list)):
        return kind, tuple(_snapshot(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return kind, tuple(sorted((_snapshot(item) for item in value), key=repr))
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return kind, _snapshot({key: item for key, item in vars(value).items() if not key.startswith("_")})
    # Immutable dtype/device/path objects and callable configuration values.
    return kind, str(value)


def _read_field(config, path):
    value = config
    for name in path.split("."):
        if isinstance(value, Mapping):
            if name not in value:
                return False, None
            value = value[name]
        else:
            if not hasattr(value, name):
                return False, None
            value = getattr(value, name)
    return True, value


def startup_config_snapshot(config):
    """Track validator inputs, excluding ranks, KV block counts and graph state."""
    sections = {
        "model_config": (
            "model",
            "dtype",
            "quantization",
            "enforce_eager",
            "max_model_len",
            "enable_sleep_mode",
        ),
        "model_config.hf_config": (
            "model_type",
            "quantization_config",
            "num_hidden_layers",
            "n_routed_experts",
            "num_hash_layers",
            "hidden_size",
            "moe_intermediate_size",
            "num_experts_per_tok",
            "swiglu_limit",
        ),
        "parallel_config": (
            "tensor_parallel_size",
            "data_parallel_size",
            "enable_expert_parallel",
            "nnodes",
            "data_parallel_size_local",
            "data_parallel_backend",
            "distributed_executor_backend",
            "data_parallel_external_lb",
            "data_parallel_hybrid_lb",
            "enable_dbo",
            "enable_elastic_ep",
            "expert_placement_strategy",
            "pipeline_parallel_size",
            "prefill_context_parallel_size",
            "decode_context_parallel_size",
            "enable_eplb",
            "use_sequence_parallel_moe",
            "eplb_config.num_redundant_experts",
        ),
        "load_config": ("load_format", "device"),
        "device_config": ("device",),
        "scheduler_config": ("max_num_seqs", "max_num_batched_tokens", "enable_chunked_prefill", "async_scheduling"),
        "compilation_config": (
            "mode",
            "cudagraph_mode",
            "cudagraph_capture_sizes",
            "max_cudagraph_capture_size",
            "cudagraph_num_of_warmups",
        ),
        "additional_config": (
            "vq2a8",
            "eplb_config",
            "ascend_compilation_config.enable_npugraph_ex",
            "multistream_dsv4_dsa_overlap",
            "multistream_overlap_shared_expert",
            "layer_sharding",
            "mix_placement",
            "enable_flashcomm1",
            "enable_shared_expert_dp",
            "enable_dsa_cp",
        ),
        "offload_config": ("uva.cpu_offload_gb", "prefetch.offload_group_size"),
        "cache_config": ("cpu_offload_gb",),
    }
    paths = [f"{section}.{name}" for section, names in sections.items() for name in names]
    paths.append("use_v2_model_runner")
    data = {path: _read_field(config, path) for path in paths}
    for name in ("lora_config", "kv_transfer_config", "speculative_config"):
        data[name] = getattr(config, name, None) is not None
    if data["speculative_config"]:
        draft_fields = (
            "method",
            "num_speculative_tokens",
            "enforce_eager",
            "model",
            "draft_model_config.model",
        )
        for name in draft_fields:
            path = f"speculative_config.{name}"
            data[path] = _read_field(config, path)
        for prefix in ("model_config.hf_config", "speculative_config.draft_model_config.hf_config"):
            for name in (*DSPARK_GEOMETRY_FIELDS, "compress_ratios"):
                path = f"{prefix}.{name}"
                data[path] = _read_field(config, path)
        for name in (
            "expert_dtype",
            "quantization_config",
            "dspark_block_size",
            "sample_from_anchor",
            "dspark_target_layer_ids",
        ):
            path = f"speculative_config.draft_model_config.hf_config.{name}"
            data[path] = _read_field(config, path)
    return _snapshot(data)


class GS32StartupState:
    """One successful validation shared by layers of one model in one process."""

    def __init__(self, config, options):
        self.process_id = getpid()
        self.config = config
        self.snapshot = startup_config_snapshot(config)
        self.options = dict(options)
        self._artifact_paths = tuple(str(Path(options[name]).resolve()) for name in ("gs32_library", "gs32_binary"))
        self._native_ops = {}

    def matches_process(self):
        return self.process_id == getpid()

    def require_unchanged(self, config):
        if not self.matches_process() or config is not self.config or startup_config_snapshot(config) != self.snapshot:
            raise ValueError(
                "GS32 model construction configuration changed after validation; create a new model instance."
            )

    def native_ops(self, library, binary, device, *, execution, factory):
        if not self.matches_process():
            raise RuntimeError("GS32 native initialization must be validated again in the current worker process.")
        self.require_unchanged(self.config)
        paths = tuple(str(Path(path).resolve()) for path in (library, binary))
        if paths != self._artifact_paths or execution != self.options["gs32_execution"]:
            raise ValueError("GS32 native artifacts/execution differ from the validated model configuration.")
        key = str(device)
        if key not in self._native_ops:
            # Publish only after artifact/schema registration succeeds. A failed
            # initialization can be retried; plans and weights remain per layer.
            native = factory(library, binary, device, execution=execution)
            self._native_ops[key] = native
        return self._native_ops[key]
