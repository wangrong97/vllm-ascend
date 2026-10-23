# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fast-ring split table: global-consistent weighted deterministic split.

Design doc 2.4-C. The selection function depends only on globally consistent
slow variables (``route``/``weight``/``cdf``); the locally maintained EWMA
never enters the selection path (EP claim uniqueness, E1/I8). ``weighted_pick``
is a pure function: identical inputs give identical picks on every rank, and
replaying a captured decode graph reproduces the same claims.

Tensor shapes are fixed at creation (graph-stable): route/weight/cdf are
``[E_logical, capacity]`` float/int tensors padded with -1/0; per-slot EWMA
is ``[num_ranks, capacity_max]`` so both replica hits and rank backgrounds
can be tracked without reshaping.
"""

from __future__ import annotations

import torch

ROUTE_PAD = -1


def _stable_hash(key: int, salt: int = 0x9E3779B97F4A7C15) -> int:
    """SplitMix64: integer mixing with good avalanche, fully deterministic."""
    h = (key ^ salt) & 0xFFFFFFFFFFFFFFFF
    h = ((h ^ (h >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    h = ((h ^ (h >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return h ^ (h >> 31)


def weighted_pick(
    cdf_row: torch.Tensor,
    route_row: torch.Tensor,
    seq_id: int,
    expert_id: int,
) -> int:
    """Return the replica rank for (seq_id, expert_id).

    ``cdf_row`` is the prefix-sum of the (normalized) weights; ``route_row``
    lists the replica ranks (``ROUTE_PAD`` for unused capacity). Ties break
    toward the lowest rank for determinism. CPU/NPU tensor ops only.
    """
    h = _stable_hash((seq_id << 32) ^ (expert_id & 0xFFFFFFFF))
    point = (h >> 11) * (1.0 / (1 << 53))  # uniform in [0,1)
    below = (cdf_row > point).nonzero(as_tuple=False).flatten()
    if below.numel() == 0:
        valid = route_row[route_row != ROUTE_PAD]
        return int(valid[0].item())
    index = int(below[0].item())
    ranks = route_row[route_row != ROUTE_PAD]
    if index >= ranks.numel():
        index = ranks.numel() - 1
    return int(ranks[index].item())


class SplitTable:
    """Host-side builder/mirror of the fast-ring device tensors.

    The controller owns one per MoE layer; device tensors are materialized
    via :meth:`materialize` (fixed shapes, refreshed in place at P2 commits).
    """

    def __init__(self, num_experts: int, capacity: int, num_ranks: int, options):
        if capacity < 1 or num_experts < 1 or num_ranks < 1:
            raise ValueError("SplitTable needs positive experts/capacity/ranks.")
        self.num_experts = num_experts
        self.capacity = capacity
        self.num_ranks = num_ranks
        self.options = options
        # route[e] = list of replica ranks (ascending); weight in raw units.
        self.route: list[list[int]] = [[] for _ in range(num_experts)]
        self.weight: list[list[float]] = [[] for _ in range(num_experts)]
        self.ewma_rank: dict[int, float] = {r: 0.0 for r in range(num_ranks)}

    # -- route maintenance (P2 commit side) --------------------------------

    def add_replica(self, expert: int, rank: int) -> None:
        row = self.route[expert]
        if rank in row:
            raise ValueError(f"replica {rank} already registered for expert {expert}.")
        row.append(rank)
        row.sort()

    def remove_replica(self, expert: int, rank: int) -> None:
        self.route[expert].remove(rank)

    def is_multi(self, expert: int) -> bool:
        return len(self.route[expert]) > 1

    # -- weight refresh (slow ring; doc C.3) -------------------------------

    def refresh_weights(self, rank_time_shares: dict[int, float]) -> None:
        """Fold rank-level EWMA time shares into normalized replica weights.

        ``rank_time_shares[r]`` is the type-weighted EWMA of rank r. The clip
        bounds are *relative multipliers* (doc C.3 "[0.2, 5] x"): each inverse
        load is divided by the row mean before clipping, so the bounds cap
        the strongest/weakest replica ratio instead of the absolute 1/L
        magnitude (which depends on the load unit and would flatten every
        row onto w_lo).
        """
        opts = self.options
        eps = 1e-6
        for e in range(self.num_experts):
            row = self.route[e]
            if len(row) < 2:
                self.weight[e] = []
                continue
            inverses = [1.0 / (rank_time_shares.get(r, 0.0) + eps) for r in row]
            mean = sum(inverses) / len(inverses)
            raw = [min(max(v / mean, opts.weight_lo), opts.weight_hi) for v in inverses]
            total = sum(raw)
            self.weight[e] = [w / total for w in raw]

    # -- accounting (doc C.5; results feed the slow ring only) --------------

    def record_rank_share(self, rank: int, share: float) -> None:
        lam = self.options.ewma_lambda
        self.ewma_rank[rank] = lam * share + (1.0 - lam) * self.ewma_rank.get(rank, 0.0)

    def rank_time_shares(self) -> dict[int, float]:
        return dict(self.ewma_rank)

    # -- device materialization (graph-stable shapes) -----------------------

    def materialize(self, device="cpu", dtype=torch.float32) -> dict[str, torch.Tensor]:
        route = torch.full((self.num_experts, self.capacity), ROUTE_PAD, dtype=torch.int64)
        weight = torch.zeros((self.num_experts, self.capacity), dtype=dtype)
        cdf = torch.zeros((self.num_experts, self.capacity), dtype=dtype)
        for e in range(self.num_experts):
            row = self.route[e]
            for i, r in enumerate(row):
                route[e, i] = r
            if len(row) >= 2:
                ws = self.weight[e]
                acc = 0.0
                for i, w in enumerate(ws):
                    acc += w
                    weight[e, i] = w
                    cdf[e, i] = acc
                cdf[e, len(ws) - 1] = 1.0  # close the interval exactly
        return {"route": route.to(device), "weight": weight.to(device), "cdf": cdf.to(device)}
