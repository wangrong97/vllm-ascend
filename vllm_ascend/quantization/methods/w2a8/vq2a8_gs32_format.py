# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source-v3 checkpoint contract and the existing GS32 native tensor ABI."""

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import regex as re
import torch

GS32_FORMAT = "vq2_mixed_gs32_source"
GS32_VERSION = 3
GS32_ROW_ORDER = "natural_gate_then_up"
# These names describe the native bank, never accepted checkpoint field names.
GS32_FIELDS = ("packed_zn", "pair_lut", "row_scale_bits", "perm")
GS32_DTYPES = (torch.uint8, torch.uint8, torch.int64, torch.int64)
GS32_SAFE_DTYPES = ("U8", "U8", "I64", "I64")
VQ2_SOURCE_FIELDS = ("packed_indices", "codebooks", "perm", "out_dim_scale")
MXFP4_SOURCE_FIELDS = ("weight", "scale")
SOURCE_TO_NATIVE = {
    "packed_indices": "packed_zn",
    "codebooks": "pair_lut",
    "perm": "perm",
    "out_dim_scale": "row_scale_bits",
    "weight": "mxfp4_weight",
    "scale": "mxfp4_scale",
}
GS32_MAX_ROWS = 128  # Serial baseline and device-route per-expert limit.
GS32_MAX_HEADER_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class GS32ProjectionSpec:
    rows: int
    columns: int

    def __post_init__(self):
        if (
            type(self.rows) is not int
            or type(self.columns) is not int
            or self.rows < 32
            or self.rows % 32
            or self.columns < 64
            or self.columns % 64
        ):
            raise ValueError("GS32 staged projection requires positive N32 and K64 alignment.")


def gs32_projection_shapes(spec):
    n, k = spec.rows, spec.columns
    return ((n // 32, k // 16, 16, 8), (n // 32, k // 32, 32), (n,), (k,))


def source_projection_fields(spec, quant_type):
    n, k = spec.rows, spec.columns
    if quant_type == "vq2":
        return {
            "packed_indices": ("U8", (n // 32, k // 16, 16, 8)),
            "codebooks": ("F8_E4M3", (n // 32, k // 32, 16, 2)),
            "perm": ("I32", (k,)),
            "out_dim_scale": ("U64", (n,)),
        }
    if quant_type == "mxfp4":
        return {"weight": ("U8", (n, k // 2)), "scale": ("F8_E8M0", (n, k // 32))}
    raise ValueError(f"Unknown source-v3 projection type: {quant_type}.")


def convert_expert_tensor(field, value, spec, quant_type):
    """Validate a single source-v3 field, then adapt its bytes to the native ABI."""
    fields = source_projection_fields(spec, quant_type)
    if field not in fields:
        raise ValueError(f"Unexpected {quant_type} source-v3 field: {field}.")
    dtype_name, shape = fields[field]
    dtype = {
        "U8": torch.uint8,
        "I32": torch.int32,
        "U64": torch.uint64,
        "F8_E4M3": torch.float8_e4m3fn,
        "F8_E8M0": getattr(torch, "float8_e8m0fnu", None),
    }[dtype_name]
    if dtype is None or value.dtype != dtype or tuple(value.shape) != shape:
        raise ValueError(f"Incorrect source-v3 {field} dtype/shape: expected {dtype_name} {shape}.")
    value = value.contiguous()
    if field == "codebooks":
        raw = value.view(torch.uint8)
        if bool(((raw & 127) == 127).any()):
            raise ValueError("Source-v3 codebooks contain NaN.")
        return raw.reshape(spec.rows // 32, spec.columns // 32, 32)
    if field == "perm":
        result = value.to(torch.int64)
        if not bool((result.sort().values == torch.arange(spec.columns, device=value.device)).all()):
            raise ValueError("Source-v3 perm is not a complete input-column permutation.")
        return result
    if field == "out_dim_scale":
        result = value.view(torch.int64)
        scale = result.to(torch.int32).contiguous().view(torch.float32)
        if not bool(((result >= 0) & (result <= 0xFFFFFFFF)).all() & (torch.isfinite(scale) & (scale > 0)).all()):
            raise ValueError("Invalid source-v3 out_dim_scale bit pattern.")
        return result
    if field == "scale":
        raw = value.view(torch.uint8)
        if bool((raw == 255).any()):
            raise ValueError("Source-v3 MXFP4 scale contains NaN.")
        return raw
    return value


def parse_vq2a8_expert_name(name):
    match = re.fullmatch(r"layers\.(0|[1-9][0-9]*)\.ffn\.experts\.(0|[1-9][0-9]*)\.(w13|w2)\.([a-z_]+)", name)
    if match is not None:
        return int(match[1]), int(match[2]), match[3], match[4]
    if ".experts." in name or ".experts_" in name:
        raise ValueError(f"Expected an independently named source-v3 expert tensor, got {name}.")
    return None


def normalize_vq2a8_root_name(name):
    """Map HF source-v3 roots without conflating HC scales with Linear scales."""
    globals_ = {
        "head.weight": "lm_head.weight",
        "model.embed_tokens.weight": "model.embed_tokens.weight",
        "model.norm.weight": "model.norm.weight",
        "model.hc_head.hc_fn": "model.hc_head_fn",
        "model.hc_head.hc_base": "model.hc_head_base",
        "model.hc_head.hc_scale": "model.hc_head_scale",
    }
    if name in globals_:
        return globals_[name]
    match = re.fullmatch(r"model\.layers\.(0|[1-9][0-9]*)\.(.+)", name)
    if match is None:
        raise ValueError(f"Unknown source-v3 root tensor: {name}.")
    suffix = match[2]
    direct = {
        "attn_hc.fn": "hc_attn_fn",
        "attn_hc.base": "hc_attn_base",
        "attn_hc.scale": "hc_attn_scale",
        "ffn_hc.fn": "hc_ffn_fn",
        "ffn_hc.base": "hc_ffn_base",
        "ffn_hc.scale": "hc_ffn_scale",
        "self_attn.sinks": "self_attn.attn_sink",
        "self_attn.compressor.position_bias": "self_attn.compressor.ape",
        "self_attn.compressor.indexer.position_bias": "self_attn.indexer.compressor.ape",
        "mlp.gate.e_score_correction_bias": "mlp.gate.e_score_correction_bias",
        "mlp.gate.tid2eid": "mlp.gate.tid2eid",
    }
    if suffix in direct:
        return f"model.layers.{match[1]}.{direct[suffix]}"
    modules = {
        "input_layernorm": "input_layernorm",
        "post_attention_layernorm": "post_attention_layernorm",
        "self_attn.q_a_proj": "self_attn.wq_a",
        "self_attn.q_a_norm": "self_attn.q_norm",
        "self_attn.q_b_proj": "self_attn.wq_b",
        "self_attn.kv_proj": "self_attn.wkv",
        "self_attn.kv_norm": "self_attn.kv_norm",
        "self_attn.o_a_proj": "self_attn.wo_a",
        "self_attn.o_b_proj": "self_attn.wo_b",
        "self_attn.compressor.gate_proj": "self_attn.compressor.wgate",
        "self_attn.compressor.kv_proj": "self_attn.compressor.wkv",
        "self_attn.compressor.kv_norm": "self_attn.compressor.norm",
        "self_attn.compressor.indexer.q_b_proj": "self_attn.indexer.wq_b",
        "self_attn.compressor.indexer.scorer.weights_proj": "self_attn.indexer.weights_proj",
        "self_attn.compressor.indexer.gate_proj": "self_attn.indexer.compressor.wgate",
        "self_attn.compressor.indexer.kv_proj": "self_attn.indexer.compressor.wkv",
        "self_attn.compressor.indexer.kv_norm": "self_attn.indexer.compressor.norm",
        "mlp.gate": "mlp.gate",
        "mlp.shared_experts.gate_proj": "mlp.shared_experts.gate_proj",
        "mlp.shared_experts.up_proj": "mlp.shared_experts.up_proj",
        "mlp.shared_experts.down_proj": "mlp.shared_experts.down_proj",
    }
    module, _, field = suffix.rpartition(".")
    if module not in modules or field not in ("weight", "scale"):
        raise ValueError(f"Unknown source-v3 root tensor: {name}.")
    return f"model.layers.{match[1]}.{modules[module]}.{'weight_scale' if field == 'scale' else field}"


def _unique_json_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"Duplicate checkpoint JSON key: {name}.")
        result[name] = value
    return result


def read_gs32_json(path):
    result = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_json_object)
    if not isinstance(result, dict):
        raise ValueError(f"Expected a checkpoint JSON object: {path}.")
    return result


def read_gs32_header(path):
    """Audit safetensors metadata and extents without loading tensor payloads."""
    path = Path(path)
    with path.open("rb") as stream:
        encoded_size = stream.read(8)
        if len(encoded_size) != 8:
            raise ValueError(f"Truncated safetensors header: {path}.")
        size = struct.unpack("<Q", encoded_size)[0]
        if not 2 <= size <= GS32_MAX_HEADER_BYTES or size > path.stat().st_size - 8:
            raise ValueError(f"Invalid safetensors header length: {path}.")
        header = json.loads(stream.read(size), object_pairs_hook=_unique_json_object)
    if not isinstance(header, dict):
        raise ValueError(f"Expected a safetensors header object: {path}.")
    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"Invalid safetensors metadata: {path}.")
    for name, expected in (
        ("format", GS32_FORMAT),
        ("format_version", str(GS32_VERSION)),
        ("row_order", GS32_ROW_ORDER),
    ):
        if name in metadata and metadata[name] != expected:
            # Ordinary HF rest shards may identify their serialization as pt.
            if name == "format" and metadata[name] == "pt":
                continue
            raise ValueError(f"Incompatible source-v3 {name} in {path}.")
    dtype_bytes = {"U8": 1, "F8_E4M3": 1, "F8_E8M0": 1, "I32": 4, "U64": 8, "I64": 8, "F32": 4, "BF16": 2}
    intervals = []
    for name, tensor in header.items():
        if not isinstance(tensor, dict) or set(tensor) != {"dtype", "shape", "data_offsets"}:
            raise ValueError(f"Invalid tensor header: {name}.")
        dtype, shape, offsets = tensor["dtype"], tensor["shape"], tensor["data_offsets"]
        if (
            dtype not in dtype_bytes
            or not isinstance(shape, list)
            or any(type(dim) is not int or dim <= 0 for dim in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(offset) is not int or offset < 0 for offset in offsets)
            or offsets[1] - offsets[0] != math.prod(shape) * dtype_bytes[dtype]
        ):
            raise ValueError(f"Invalid tensor dtype, shape or extent: {name}.")
        intervals.append(tuple(offsets))
    end = 0
    for begin, stop in sorted(intervals):
        if begin != end:
            raise ValueError(f"Overlapping or missing safetensors payload: {path}.")
        end = stop
    if end != path.stat().st_size - 8 - size:
        raise ValueError(f"Truncated or trailing safetensors payload: {path}.")
    return header


def validate_gs32_options(config, options):
    required = {"gs32_library", "gs32_binary"}

    if (
        not required.issubset(options)
        or set(options) - required - {"gs32_execution", "gs32_fusion"}
        or any(not isinstance(options[key], str) or not Path(options[key]).is_absolute() for key in required)
    ):
        raise ValueError(
            "GS32 requires absolute gs32_library/gs32_binary paths and optional gs32_execution/gs32_fusion only."
        )
    execution = options.get("gs32_execution", "serial")
    if execution not in ("serial", "batched", "fused", "full_graph"):
        raise ValueError("GS32 gs32_execution must be serial, batched, fused or full_graph.")
    fusion = options.get("gs32_fusion", "none")
    if not isinstance(fusion, str) or fusion not in ("none", "swiglu_quant"):
        raise ValueError("GS32 gs32_fusion must be none or swiglu_quant.")
    graph = execution == "full_graph"
    if graph and config.model_config.enforce_eager:
        raise ValueError("GS32 graph execution requires enforce_eager=False.")
    if not graph and not config.model_config.enforce_eager:
        raise ValueError("GS32 staged kernels require eager execution; graph capture has not been validated.")
