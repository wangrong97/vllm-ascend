# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration switch for the mixed-precision EPLB algorithm.

The switch lives inside the existing ``additional_config.vq2a8`` object so no
new top-level option or environment variable is introduced:

    {"vq2a8": {"gs32_library": ..., "gs32_binary": ...,
               "eplb": "dynamic",                # "off" (default) | "static" | "dynamic"
               "eplb_options": {"K": 8, "R_vq2": 4, ...}}}

``off`` (the default) keeps every code path byte-identical to the baseline:
no capacity reservation, no load accounting, no split table, no migration.
``static`` plans once at startup (initial replica allocation); ``dynamic``
additionally runs the periodic swap/fill/evict pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Typed modes are quant-width aware (per-type cost slopes, byte budgets,
# typed replica pools). "*_uniform" modes run the same pipeline but treat
# every expert identically: unit costs are flat (token-count balancing),
# replica value ignores bundle bytes, one pooled replica budget, and no byte
# accounting (the fixed bank capacity already bounds resident memory).
# Bank type rigidity is *physical* and still enforced in both variants:
# a vq2 replica can only land in a spare vq2 slot.
MP_EPLB_MODES = ("off", "static", "dynamic", "static_uniform", "dynamic_uniform")

# Conservative defaults; every knob is overridable through eplb_options.
DEFAULT_MOVE_BUDGET_K = 8
DEFAULT_REPLICA_SLOTS_VQ2 = 4
DEFAULT_REPLICA_SLOTS_MXFP4 = 2
DEFAULT_STAGING_BUFFERS = 2
DEFAULT_TAU_TOKEN = 0.10
DEFAULT_TAU_SLOW = 0.05
DEFAULT_TAU_REVOKE = 0.02
DEFAULT_REVOLVE_HYSTERESIS = 2
DEFAULT_EWMA_LAMBDA = 0.05
DEFAULT_WEIGHT_LO = 0.2
DEFAULT_WEIGHT_HI = 5.0
DEFAULT_REPLICA_REPLAN_EVERY = 4
DEFAULT_IMPROVEMENT_EPS = 1e-3

_OPTION_KEYS = frozenset(
    {
        "K",
        "R_vq2",
        "R_mx4",
        "n_stage",
        "tau_token",
        "tau_slow",
        "tau_revoke",
        "revoke_hysteresis",
        "ewma_lambda",
        "w_lo",
        "w_hi",
        "replica_replan_every",
        "improvement_eps",
    }
)


def _positive_int(value, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"mp-eplb option {name} must be an int, got {value!r}.")
    if value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"mp-eplb option {name} must be positive, got {value}.")
    return value


def _fraction(value, name: str, *, lower: float = 0.0, upper: float = 1.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"mp-eplb option {name} must be a number, got {value!r}.")
    if not lower < value < upper:
        raise ValueError(f"mp-eplb option {name} must lie in ({lower}, {upper}), got {value}.")
    return float(value)


@dataclass(frozen=True)
class MPEplbOptions:
    """Parsed, validated options; ``enabled`` is the master gate."""

    mode: str = "off"
    enabled: bool = False
    dynamic: bool = False
    uniform: bool = False  # width-agnostic variant (no typed costs/bytes)
    move_budget_K: int = DEFAULT_MOVE_BUDGET_K
    replica_slots_vq2: int = DEFAULT_REPLICA_SLOTS_VQ2
    replica_slots_mxfp4: int = DEFAULT_REPLICA_SLOTS_MXFP4
    staging_buffers: int = DEFAULT_STAGING_BUFFERS
    tau_token: float = DEFAULT_TAU_TOKEN
    tau_slow: float = DEFAULT_TAU_SLOW
    tau_revoke: float = DEFAULT_TAU_REVOKE
    revoke_hysteresis: int = DEFAULT_REVOLVE_HYSTERESIS
    ewma_lambda: float = DEFAULT_EWMA_LAMBDA
    weight_lo: float = DEFAULT_WEIGHT_LO
    weight_hi: float = DEFAULT_WEIGHT_HI
    replica_replan_every: int = DEFAULT_REPLICA_REPLAN_EVERY
    improvement_eps: float = DEFAULT_IMPROVEMENT_EPS
    extra: dict = field(default_factory=dict)

    @property
    def replica_slots(self) -> dict[str, int]:
        return {"vq2": self.replica_slots_vq2, "mxfp4": self.replica_slots_mxfp4}


def get_mp_eplb_options(native_options: dict | None) -> MPEplbOptions:
    """Parse ``additional_config.vq2a8.eplb`` (missing/invalid -> hard error).

    ``native_options`` is the dict already validated by
    ``validate_vq2a8_config``; it must contain ``gs32_library``/``gs32_binary``
    (checked upstream) and may contain ``eplb``/``eplb_options``.
    """
    options = dict(native_options or {})
    mode = options.pop("eplb", "off")
    if mode not in MP_EPLB_MODES:
        raise ValueError(f"vq2a8.eplb must be one of {MP_EPLB_MODES}, got {mode!r}.")
    raw = options.pop("eplb_options", None)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("vq2a8.eplb_options must be an object.")
    unknown = set(raw) - _OPTION_KEYS
    if unknown:
        raise ValueError(f"Unknown vq2a8.eplb_options keys: {sorted(unknown)}.")

    parsed = MPEplbOptions(
        mode=mode,
        enabled=mode != "off",
        dynamic=mode.startswith("dynamic"),
        uniform=mode.endswith("_uniform"),
        move_budget_K=_positive_int(raw.get("K", DEFAULT_MOVE_BUDGET_K), "K"),
        replica_slots_vq2=_positive_int(raw.get("R_vq2", DEFAULT_REPLICA_SLOTS_VQ2), "R_vq2", allow_zero=True),
        replica_slots_mxfp4=_positive_int(
            raw.get("R_mx4", DEFAULT_REPLICA_SLOTS_MXFP4), "R_mx4", allow_zero=True
        ),
        staging_buffers=_positive_int(raw.get("n_stage", DEFAULT_STAGING_BUFFERS), "n_stage"),
        tau_token=_fraction(raw.get("tau_token", DEFAULT_TAU_TOKEN), "tau_token"),
        tau_slow=_fraction(raw.get("tau_slow", DEFAULT_TAU_SLOW), "tau_slow"),
        tau_revoke=_fraction(raw.get("tau_revoke", DEFAULT_TAU_REVOKE), "tau_revoke", upper=2.0),
        revoke_hysteresis=_positive_int(
            raw.get("revoke_hysteresis", DEFAULT_REVOLVE_HYSTERESIS), "revoke_hysteresis"
        ),
        ewma_lambda=_fraction(raw.get("ewma_lambda", DEFAULT_EWMA_LAMBDA), "ewma_lambda"),
        weight_lo=_fraction(raw.get("w_lo", DEFAULT_WEIGHT_LO), "w_lo", upper=10.0),
        weight_hi=_fraction(raw.get("w_hi", DEFAULT_WEIGHT_HI), "w_hi", upper=100.0),
        replica_replan_every=_positive_int(
            raw.get("replica_replan_every", DEFAULT_REPLICA_REPLAN_EVERY), "replica_replan_every"
        ),
        improvement_eps=_fraction(
            raw.get("improvement_eps", DEFAULT_IMPROVEMENT_EPS), "improvement_eps", upper=1.0
        ),
    )
    if parsed.weight_lo >= parsed.weight_hi:
        raise ValueError(
            f"vq2a8.eplb_options w_lo ({parsed.weight_lo}) must be < w_hi ({parsed.weight_hi})."
        )
    return parsed


OFF_OPTIONS = MPEplbOptions()
