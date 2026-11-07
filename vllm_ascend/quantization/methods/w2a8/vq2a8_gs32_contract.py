# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-independent source-v3 declarations shared by runtime and CLI checks.

The dedicated-head profile declares rotations already folded into weights.
Validation checks that declaration; it never applies or reconstructs rotations.
"""


def resolve_gs32_options(options):
    """Resolve the single full-FFN implementation, without device imports.

    Backend/fusion keys remain accepted for existing native full-FFN commands
    and recorded provenance. They no longer select different implementations.
    """
    resolved = dict(options)
    backend = resolved.get("gs32_backend", "native")
    if backend != "native":
        raise ValueError("GS32 serving supports only native full_ffn; reference backends are test-only.")
    fusion = resolved.get("gs32_fusion", "full_ffn")
    if fusion != "full_ffn":
        raise ValueError("GS32 serving supports only fusion=full_ffn; staged and partial fusion were removed.")
    execution = resolved.get("gs32_execution", "full_graph")
    if execution not in ("fused", "full_graph"):
        raise ValueError("GS32 full_ffn execution must be full_graph or fused (eager); serial/batched were removed.")
    resolved.update(gs32_backend=backend, gs32_execution=execution, gs32_fusion=fusion)
    return resolved


SOURCE_V3_CONFIG = (
    ("quant_method", "vq2a8"),
    ("format", "vq2_mixed_gs32_source"),
    ("format_version", 3),
)
FOLDED_HEAD_CONFIG = (
    ("hidden_layer_export_policy", "prefix"),
    ("vq_weight_layout", "n32_k16_16_8"),
    ("non_vq_key_layout", "source_keys_with_dedicated_mtp_embedding_head"),
    ("ordinary_float_weight_layout", "source_layout"),
    ("vq_gate_up_layout", "natural_gate_then_up"),
    ("non_vq_gate_up_layout", "separate_w1_w3"),
    ("vq_codebook_layout", "n32_k32_16_2"),
    ("vq_out_scale_encoding", "u64_low32_fp32"),
    ("vq_input_permutation", False),
    ("mxfp4_weight_layout", "row_k_packed_e2m1"),
    ("mxfp4_scale_encoding", "fp8_e8m0"),
    ("main_routed_expert_active_format", "vq"),
    ("main_expert_tensor_count", 6),
    ("main_routed_expert_mxfp4_backup", False),
    ("mtp_routed_expert_format", "mxfp4"),
    ("output_head_mode", "dedicated_main_mtp_norm_fused"),
    ("mtp_hc_head_rotation", "right_global_hc"),
    ("inverse_rotation_before_final_norm", False),
    ("flex_smooth_enabled", False),
    ("requires_custom_loader", True),
)
FOLDED_ROTATION_CONFIG = (
    ("scheme", "msmodelslim_quarot_dedicated_mtp_heads"),
    ("mtp_hc_head_rotation", "right_global_hc"),
    ("main_mtp_final_norm", "ones_fused_into_dedicated_heads"),
    ("inverse_rotation_before_final_norm", False),
    ("flex_smooth_enabled", False),
    ("compute_dtype", "float32"),
)


def validate_source_config(config) -> bool:
    """Accept legacy source-v3 or the complete supported folded-head profile."""
    if not isinstance(config, dict):
        raise ValueError("VQ2A8 requires a source-v3 quantization_config object.")
    expected = dict(SOURCE_V3_CONFIG)
    folded = set(config) != set(expected)
    if folded:
        expected.update(FOLDED_HEAD_CONFIG)
    if set(config) != set(expected):
        raise ValueError(
            "Unsupported source-v3 declarations: "
            f"missing={sorted(set(expected) - set(config))}, extra={sorted(set(config) - set(expected))}."
        )
    for key, value in expected.items():
        if type(config[key]) is not type(value) or config[key] != value:
            raise ValueError(f"Unsupported source-v3 {key}={config[key]!r}; expected {value!r}.")
    return folded


def validate_rotation_config(rotation) -> None:
    """Reject contradictory sidecar semantics; numeric recipe fields are provenance."""
    if not isinstance(rotation, dict):
        raise ValueError("rotation_config.json must contain an object.")
    for key, expected in FOLDED_ROTATION_CONFIG:
        if type(rotation.get(key)) is not type(expected) or rotation[key] != expected:
            raise ValueError(f"Unsupported rotation_config.json {key}; expected {expected!r}.")
    # These optional explicit declarations must not contradict the folded profile.
    for key in ("online", "runtime_rotation_required"):
        if key in rotation and rotation[key] is not False:
            raise ValueError(f"Folded source-v3 weights require {key}=false.")


def validate_folded_speculation(config, speculation) -> None:
    if speculation is not None and config.get("output_head_mode") == "dedicated_main_mtp_norm_fused":
        raise ValueError(
            "Folded source-v3 currently supports main-model decoding only. "
            "Its dedicated single-stage MTP is not the original DSpark drafter; "
            "use --no-dspark and omit speculative_config."
        )
