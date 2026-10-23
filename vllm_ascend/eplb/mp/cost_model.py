# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared typed cost model: `load` is a token count, `c_type` a unit cost.

Both fast and slow rings consume these functions only (design doc 2.4):
``load(e) * c_type(e) == alpha_sum * total_tokens + beta_sum * launches``
holds exactly within one M bucket, so amortization is a pure algebraic
reformulation of the piecewise-linear kernel model ``t(M) = alpha*M + beta``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

QUANT_TYPES = ("vq2", "mxfp4")
PROJECTIONS = ("w13", "w2")

# Bucket boundaries for the piecewise-linear kernel model. Decode graphs cover
# at most MAX_GRAPH_ROUTED_ROWS=8 routed rows; larger Ms use prefill tiles.
DECODE_BUCKET = 8

# Relative-unit defaults (alpha_vq2 == 1). Values come from single-projection
# timing on Ascend950DT / CANN 9.1 (see docs: P0 calibration); they only need
# to be *relatively* correct for planning and splitting.
_DEFAULT_SLOPES = {("vq2", "w13"): 1.0, ("vq2", "w2"): 1.0, ("mxfp4", "w13"): 1.8, ("mxfp4", "w2"): 1.8}
_DEFAULT_LAUNCH = {("vq2", "w13"): 0.02, ("vq2", "w2"): 0.02, ("mxfp4", "w13"): 0.01, ("mxfp4", "w2"): 0.01}


@dataclass(frozen=True)
class CostTable:
    """Per (quant_type, projection, bucket) linear coefficients."""

    slopes: dict
    launch: dict

    @classmethod
    def defaults(cls) -> "CostTable":
        return cls(slopes=dict(_DEFAULT_SLOPES), launch=dict(_DEFAULT_LAUNCH))

    @classmethod
    def uniform(cls) -> "CostTable":
        """Width-agnostic table: flat 0.5 per projection (sum 1.0), no launch
        overhead — every expert costs the same per token, so balancing the
        cost equals balancing raw token counts."""
        slopes = {(q, p): 0.5 for q in QUANT_TYPES for p in PROJECTIONS}
        launch = {(q, p): 0.0 for q in QUANT_TYPES for p in PROJECTIONS}
        return cls(slopes=slopes, launch=launch)

    def bucket(self, mean_rows: float) -> str:
        return "decode" if 0 < mean_rows <= DECODE_BUCKET else "prefill"

    def alpha_sum(self, quant_type: str) -> float:
        return sum(self.slopes[(quant_type, proj)] for proj in PROJECTIONS)

    def beta_sum(self, quant_type: str, mean_rows: float) -> float:
        # Launch overhead stays with the bucket it was measured in.
        return sum(self.launch[(quant_type, proj)] for proj in PROJECTIONS)


def amortized_unit_cost(
    quant_type: str,
    total_tokens: int,
    launches: int,
    table: CostTable | None = None,
) -> float:
    """`c_type(e) = (alpha_w13+alpha_w2) + (beta_w13+beta_w2)/M_bar`.

    ``M_bar = total_tokens / launches`` is the mean rows per active launch in
    the window. The identity ``load * c_type == alpha*total + beta*launches``
    holds by construction, so the returned unit cost is exact for rank-level
    aggregation even though it looks like an approximation.
    """
    if quant_type not in QUANT_TYPES:
        raise ValueError(f"Unknown quant type: {quant_type!r}.")
    if total_tokens < 0 or launches < 0:
        raise ValueError("total_tokens and launches must be non-negative.")
    table = table or CostTable.defaults()
    alpha = table.alpha_sum(quant_type)
    if launches == 0:
        return alpha
    mean_rows = total_tokens / launches
    if mean_rows <= 0:
        return alpha
    return alpha + table.beta_sum(quant_type, mean_rows) / mean_rows


def window_load(slot_rows: dict[tuple[int, int], int]) -> dict[int, int]:
    """Merge per-(expert, slot) windowed row counts into logical loads.

    ``slot_rows`` maps ``(logical_expert, physical_slot_id) -> token rows``;
    replica hits are summed over slots (design doc C.3 step 1). Slots with
    zero rows may be omitted.
    """
    loads: dict[int, int] = {}
    for (expert, _slot), rows in slot_rows.items():
        loads[expert] = loads.get(expert, 0) + int(rows)
    return loads


def ewma_step(previous: torch.Tensor, observed: torch.Tensor, lam: float) -> torch.Tensor:
    """In-place-safe EWMA update over slot-level *time shares*.

    ``observed[s] = rows[s] * alpha_sum(type(slot)) + active[s] * beta_sum``:
    slope*rows plus one launch constant per active slot (doc C.5). Callers
    build ``observed``; this helper only blends it.
    """
    if previous.shape != observed.shape:
        raise ValueError("EWMA tensors must share a fixed shape (graph-stable).")
    return lam * observed + (1.0 - lam) * previous


def expert_bytes(quant_type: str, rows_n: int, cols_k: int) -> int:
    """Bytes of one expert bundle for one projection (logical formula).

    Use ``Tensor.numel() * element_size()`` on device-resident slices for the
    authoritative number (NZ padding after ``prepare_mxfp4_projection``);
    this closed form is for planning-time estimates and layout tests.
    """
    if quant_type == "vq2":
        return rows_n * cols_k // 4 + rows_n * cols_k // 32 + 8 * (rows_n + cols_k)
    if quant_type == "mxfp4":
        return rows_n * cols_k // 2 + rows_n * cols_k // 32
    raise ValueError(f"Unknown quant type: {quant_type!r}.")


def bundle_bytes(quant_type: str, gate_up_rows: int, hidden: int, down_rows: int) -> int:
    """Bytes of a full expert bundle (both projections).

    w13: N=gate_up_rows (2*inter), K=hidden;  w2: N=down_rows, K=inter.
    """
    inter = gate_up_rows // 2
    return expert_bytes(quant_type, gate_up_rows, hidden) + expert_bytes(quant_type, down_rows, inter)
