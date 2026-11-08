# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate and isolate the official DSpark drafter for a GS32 target."""

from __future__ import annotations

import json
from copy import copy
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig

DSPARK_SPECULATIVE_TOKENS = 5
DSPARK_TARGET_QUERY_WIDTH = DSPARK_SPECULATIVE_TOKENS + 1
DSPARK_DEFAULT_MTP_STAGES = 3
DSPARK_QUANTIZATION = "deepseek_v4_fp8"
DSPARK_GEOMETRY_FIELDS = (
    "hidden_size",
    "vocab_size",
    "hc_mult",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "qk_rope_head_dim",
    "q_lora_rank",
    "o_lora_rank",
    "o_groups",
    "n_routed_experts",
    "num_experts_per_tok",
    "moe_intermediate_size",
    "n_shared_experts",
    "num_hash_layers",
)


def validate_vq2a8_dspark_config(config: VllmConfig) -> None:
    """Reject unsupported draft formats before allocating GS32 target weights.

    Structural compatibility does not establish checkpoint provenance. The
    official drafter must come from the same original model as the VQ2 target.
    """
    spec = config.speculative_config
    if (
        spec is None
        or getattr(spec, "method", None) != "dspark"
        or type(getattr(spec, "num_speculative_tokens", None)) is not int
        or spec.num_speculative_tokens != DSPARK_SPECULATIVE_TOKENS
        or type(getattr(spec, "enforce_eager", None)) is not bool
        or not getattr(config, "use_v2_model_runner", False)
    ):
        raise ValueError(
            "VQ2A8 speculation requires MRV2, DSpark with exactly 5 draft tokens and boolean enforce_eager."
        )
    model = getattr(spec, "model", None)
    draft_model = getattr(spec, "draft_model_config", None)
    if not model or draft_model is None:
        raise ValueError("VQ2A8 DSpark requires an explicit local official draft checkpoint.")
    root = Path(model).resolve()
    if not root.is_dir() or root == Path(config.model_config.model).resolve():
        raise ValueError("VQ2A8 DSpark requires a separate local official checkpoint, not the source-v3 target.")
    if Path(draft_model.model).resolve() != root:
        raise ValueError("DSpark draft_model_config does not reference the requested official checkpoint.")
    try:
        hf = json.loads((root / "config.json").read_text(encoding="utf-8"))
        index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("DSpark requires readable official config.json and model.safetensors.index.json.") from exc
    if not isinstance(hf, dict) or not isinstance(index, dict):
        raise ValueError("DSpark config.json and model.safetensors.index.json must contain JSON objects.")
    quantization = hf.get("quantization_config", {})
    draft_hf = draft_model.hf_config
    runtime_quantization = getattr(draft_hf, "quantization_config", None)
    if (
        hf.get("model_type") != "deepseek_v4"
        or not isinstance(quantization, dict)
        or quantization.get("quant_method") != "fp8"
        # Match DeepseekV4FP8Config.expert_dtype: older original checkpoints
        # omit this field and use FP4 experts. An explicit override still
        # needs to select the W4A8 path in the resolved draft configuration.
        or hf.get("expert_dtype", "fp4") != "fp4"
        or getattr(draft_hf, "expert_dtype", "fp4") != "fp4"
        or not isinstance(runtime_quantization, dict)
        or runtime_quantization.get("quant_method") != "fp8"
    ):
        raise ValueError("VQ2A8 DSpark requires the original DeepSeek-V4 FP8/FP4 checkpoint.")
    if (
        hf.get("dspark_block_size") != DSPARK_SPECULATIVE_TOKENS
        or getattr(draft_hf, "dspark_block_size", None) != DSPARK_SPECULATIVE_TOKENS
    ):
        raise ValueError("The official DSpark checkpoint must use dspark_block_size=5.")
    if getattr(draft_hf, "sample_from_anchor", True) is not True:
        raise ValueError("VQ2A8 DSpark requires the original anchor-as-first-prediction draft layout.")

    target_hf = config.model_config.hf_config
    for name in DSPARK_GEOMETRY_FIELDS:
        if hf.get(name) is None or hf[name] != getattr(target_hf, name, None):
            raise ValueError(f"DSpark target/draft geometry differs for {name}.")
        if hf[name] != getattr(draft_hf, name, None):
            raise ValueError(f"DSpark runtime draft geometry differs from its checkpoint for {name}.")
    num_layers = hf["num_hidden_layers"]
    if hf.get("compress_ratios", [])[:num_layers] != list(getattr(target_hf, "compress_ratios", []))[:num_layers]:
        raise ValueError("DSpark target/draft attention compression layouts differ.")
    aux_layers = hf.get("dspark_target_layer_ids")
    if (
        not isinstance(aux_layers, list)
        or not aux_layers
        or any(type(layer) is not int or not 0 <= layer < num_layers for layer in aux_layers)
        or len(set(aux_layers)) != len(aux_layers)
        or aux_layers != list(getattr(draft_hf, "dspark_target_layer_ids", []))
    ):
        raise ValueError("DSpark requires valid matching target auxiliary-layer indices.")
    stages = hf.get("n_mtp_layers") or hf.get("dspark_num_mtp_layers") or DSPARK_DEFAULT_MTP_STAGES
    if type(stages) is not int or stages < 1:
        raise ValueError("DSpark requires a positive MTP stage count.")
    weights = index.get("weight_map")
    if not isinstance(weights, dict) or not weights:
        raise ValueError("DSpark requires a nonempty safetensors weight index.")
    required = {
        "mtp.0.main_proj.weight",
        "mtp.0.main_proj.scale",
        "mtp.0.main_norm.weight",
        f"mtp.{stages - 1}.norm.weight",
        f"mtp.{stages - 1}.markov_head.markov_w1.weight",
        f"mtp.{stages - 1}.markov_head.markov_w2.weight",
    }
    required.update(f"mtp.{stage}.attn.wq_a.weight" for stage in range(stages))
    missing = required - weights.keys()
    if missing:
        raise ValueError(f"DSpark checkpoint is missing draft weights: {sorted(missing)}.")
    if any(not isinstance(shard, str) or not shard for shard in weights.values()):
        raise ValueError("DSpark safetensors index contains an invalid shard path.")
    for shard in set(weights.values()):
        path = (root / shard).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"DSpark official checkpoint shard is missing or outside its directory: {shard}.")


def prepare_vq2a8_dspark_config(config: VllmConfig) -> VllmConfig:
    """Give DSpark its own W4A8 model identity without mutating the target.

    vLLM's DeepSeek DSpark config inherits the target quantization even when a
    separate checkpoint was provided. Its loader subsequently rebuilds the
    quantization from draft_model_config, so repair that field as well as the
    active runtime config. Use separate graph buckets for target verification
    and parallel drafting, while sharing static_forward_context so their
    attention layers still register in the same inventory.
    """
    if config.model_config.quantization != "vq2a8":
        return config
    quant_config = getattr(config, "quant_config", None)
    validate_runtime = getattr(quant_config, "validate_runtime_config", None)
    if getattr(quant_config, "is_gs32", False) and callable(validate_runtime):
        validate_runtime(config)
    else:
        validate_vq2a8_dspark_config(config)
    # Lazy worker-side import: this module is also used by the CPU preflight.
    from vllm.config import VllmConfig

    draft = copy(config.speculative_config.draft_model_config)
    draft.quantization = DSPARK_QUANTIZATION
    draft.max_model_len = config.model_config.max_model_len
    draft.dtype = config.model_config.dtype
    draft.use_fp64_gumbel = config.model_config.use_fp64_gumbel
    spec = copy(config.speculative_config)
    spec.draft_model_config = draft
    spec.quantization = DSPARK_QUANTIZATION
    runtime = copy(config)
    runtime.speculative_config = spec
    runtime.model_config = draft
    runtime.quant_config = VllmConfig.get_quantization_config(draft, config.load_config)
    if runtime.quant_config is None or runtime.quant_config.get_name() != DSPARK_QUANTIZATION:
        raise ValueError("Failed to construct the official DeepSeek-V4 W4A8 DSpark quantization config.")
    runtime.additional_config = dict(config.additional_config or {})
    runtime.additional_config.pop("vq2a8", None)
    # Target verification has six query tokens per request (five proposals
    # plus the next target token); DSpark's anchor layout drafts five. Reusing
    # target buckets rounds the largest draft bucket beyond max_num_seqs and
    # silently drops graph coverage for large batches.
    runtime.compilation_config = copy(config.compilation_config)
    target_sizes = config.compilation_config.cudagraph_capture_sizes
    decode_mode = config.compilation_config.cudagraph_mode.decode_mode()
    if decode_mode.name == "FULL" and target_sizes:
        if any(type(size) is not int or size <= 0 or size % DSPARK_TARGET_QUERY_WIDTH for size in target_sizes):
            raise ValueError("VQ2A8 DSpark target graph buckets must be positive multiples of six.")
        draft_sizes = [size // DSPARK_TARGET_QUERY_WIDTH * DSPARK_SPECULATIVE_TOKENS for size in target_sizes]
        runtime.compilation_config.cudagraph_capture_sizes = draft_sizes
        runtime.compilation_config.max_cudagraph_capture_size = max(draft_sizes)
    return runtime
