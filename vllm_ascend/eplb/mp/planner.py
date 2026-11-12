# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Slow-ring planner: ReplicaAllocator (stage A) + TCSB (stage B).

Both stages share the typed cost model and emit ``ExpertMove`` objects; the
executor applies them layer-by-layer (migrator.py). The planner is pure
CPU logic over host placement mirrors: callers pass a *working copy* of the
placement dict (``BankSlotTable`` objects are mutated in place) and keep the
original until the migration commits.

Value conventions (doc 2.4):
- ``loads``: logical-level windowed token counts *after* splitting;
- ``unit_costs``: ``c_type(e)`` amortized time per token (cost_model);
- ``nbytes_of(expert, qtype)``: bundle bytes (cost_model or device-measured).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import MPEplbOptions
from .cost_model import CostTable, QUANT_TYPES
from .placement import BankSlotTable

MOVE_SWAP = "swap"
MOVE_FILL = "fill"
MOVE_EVICT = "evict"


@dataclass(frozen=True)
class ExpertMove:
    kind: str
    expert_id: int
    quant_type: str
    src_rank: int
    src_slot: int
    dst_rank: int
    dst_slot: int
    evicted_expert: int | None = None  # swap peer living on dst_rank


@dataclass
class MigrationPlan:
    moves: list[ExpertMove] = field(default_factory=list)
    replica_alloc: dict[int, int] = field(default_factory=dict)
    expected_cost: list[float] = field(default_factory=list)
    expected_bytes: list[int] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.moves


@dataclass
class TCSBArc:
    """Value-density record for one candidate move (doc 2.4-B)."""

    move: ExpertMove
    gain: float
    delta_bytes: int


def primary_rank(table: BankSlotTable, expert: int) -> int:
    """Linear-baseline primary owner (kept fixed; doc E4)."""
    local_count = table.num_experts // table.ep_size
    return expert // local_count


def _rank_cost(placement, loads, unit_costs) -> list[float]:
    return [
        sum(
            loads.get(e, 0.0) * unit_costs.get(e, 1.0)
            for qtype in QUANT_TYPES
            for e in placement[rank].active("w13", qtype)
        )
        for rank in sorted(placement)
    ]


class ReplicaAllocator:
    """Stage A: typed-budget split replica allocation (doc 2.4-A).

    EPLB-paper amortized greedy — repeatedly give the next replica slot to
    the expert with the largest ``load / (replicas + 1)`` — executed per
    quant-type pool so one type can never exhaust the other's slots or byte
    budget (decision D3). Evictions run first to free bytes (doc 2.5 order).
    """

    def __init__(self, options: MPEplbOptions, table: CostTable | None = None):
        self.options = options
        self.table = table or CostTable.defaults()

    def allocate(
        self,
        placement: dict[int, BankSlotTable],
        loads: dict[int, float],
        nbytes_of,
        byte_budget_per_rank: int,
        excluded: set[int],
        replica_alloc: dict[int, int] | None = None,
        revoke_history: dict[tuple[int, int], int] | None = None,
    ):
        """Mutates ``placement`` with fill/evict; returns (moves, alloc, history)."""
        opts = self.options
        alloc = {e: 1 for e in loads}
        if replica_alloc:
            alloc.update(replica_alloc)
        history = dict(revoke_history or {})
        moves: list[ExpertMove] = []

        # -- revoke pass (before fills, to free bytes) ----------------------
        for rank in sorted(placement):
            table = placement[rank]
            for qtype in QUANT_TYPES:
                for slot, e in enumerate(table.slots[("w13", qtype)]):
                    if e == -1 or primary_rank(table, e) == rank:
                        continue  # empty slot or primary copy (I5)
                    per_copy = loads.get(e, 0.0) / max(1, alloc.get(e, 1))
                    key = (rank, e)
                    if per_copy >= opts.tau_revoke:
                        history.pop(key, None)  # recovered; reset hysteresis
                        continue
                    history[key] = history.get(key, 0) + 1
                    if history[key] < opts.revoke_hysteresis:
                        continue
                    moves.append(ExpertMove(MOVE_EVICT, e, qtype, rank, slot, rank, slot))
                    table.evict(qtype, slot, e)
                    alloc[e] = max(1, alloc.get(e, 1) - 1)
                    history.pop(key, None)

        # -- fill pass: typed pools (D3) or one width-agnostic pool ----------
        if opts.uniform:
            total_budget = (
                opts.replica_slots["vq2"] + opts.replica_slots["mxfp4"] - _used_replica_slots_all(placement)
            )
            skipped: set[int] = set()
            while total_budget > 0:
                best = None
                for e, load in loads.items():
                    if e in excluded or e in skipped or _quant_of(placement, e) is None:
                        continue
                    ep_size = next(iter(placement.values())).ep_size
                    if alloc.get(e, 1) >= max(2, ep_size):  # C6
                        continue
                    value = load / (alloc.get(e, 1) + 1)  # no byte density
                    if best is None or value > best[0]:
                        best = (value, e)
                if best is None:
                    break
                _, e = best
                qtype = _quant_of(placement, e)
                target = _pick_fill_target(placement, e, qtype, nbytes_of, byte_budget_per_rank)
                if target is None:
                    skipped.add(e)  # no spare slot of this type; try others
                    continue
                dst_rank, dst_slot = target
                src_rank, src_slot = _locate(placement, e, qtype)
                moves.append(ExpertMove(MOVE_FILL, e, qtype, src_rank, src_slot, dst_rank, dst_slot))
                placement[dst_rank].fill(qtype, dst_slot, e)
                alloc[e] = alloc.get(e, 1) + 1
                total_budget -= 1
            return moves, alloc, history

        for qtype in QUANT_TYPES:
            budget = opts.replica_slots[qtype] - _used_replica_slots(placement, qtype)
            while budget > 0:
                best = None
                for e, load in loads.items():
                    if e in excluded or _quant_of(placement, e) != qtype:
                        continue
                    ep_size = next(iter(placement.values())).ep_size
                    if alloc.get(e, 1) >= max(2, ep_size):  # C6
                        continue
                    value = load / (alloc.get(e, 1) + 1)
                    density = value / max(1, nbytes_of(e, qtype))
                    if best is None or density > best[0]:
                        best = (density, e)
                if best is None:
                    break
                _, e = best
                target = _pick_fill_target(placement, e, qtype, nbytes_of, byte_budget_per_rank)
                if target is None:
                    break
                dst_rank, dst_slot = target
                src_rank, src_slot = _locate(placement, e, qtype)
                moves.append(ExpertMove(MOVE_FILL, e, qtype, src_rank, src_slot, dst_rank, dst_slot))
                placement[dst_rank].fill(qtype, dst_slot, e)
                alloc[e] = alloc.get(e, 1) + 1
                budget -= 1
        return moves, alloc, history


class TCSBPlanner:
    """Stage B: typed constrained swap balancing (doc 2.4-B).

    Greedy over same-type swaps with ``gain = dimbalance`` (bytes are
    symmetric, so the unified density reduces to the imbalance delta; D1).
    Only same-type swaps keep bank counts — and therefore VQ2 group plans
    and captured decode graphs — unchanged (D4).
    """

    def __init__(self, options: MPEplbOptions, table: CostTable | None = None):
        self.options = options
        self.table = table or CostTable.defaults()

    def plan(self, placement, loads, unit_costs, nbytes_of, byte_budget_per_rank) -> MigrationPlan:
        opts = self.options
        ranks = sorted(placement)
        if len(ranks) < 2:
            return MigrationPlan()  # EP=1: no cross-rank freedom (E4)
        costs = _rank_cost(placement, loads, unit_costs)
        mean = sum(costs) / len(costs)
        moves: list[ExpertMove] = []

        for _ in range(opts.move_budget_K):
            if max(costs) - mean <= opts.tau_slow * mean:
                break
            hot = max(ranks, key=lambda r: costs[r])
            cold = min(ranks, key=lambda r: costs[r])
            best: TCSBArc | None = None
            for qtype in QUANT_TYPES:
                hot_slots = placement[hot].slots[("w13", qtype)]
                cold_slots = placement[cold].slots[("w13", qtype)]
                for slot_h, e_out in enumerate(hot_slots):
                    if e_out == -1:
                        continue
                    for slot_c, e_in in enumerate(cold_slots):
                        if e_in == -1:
                            continue
                        gain = _swap_gain(costs, hot, cold, e_out, e_in, loads, unit_costs)
                        if gain <= 1e-12:
                            continue
                        arc = TCSBArc(
                            ExpertMove(MOVE_SWAP, e_out, qtype, hot, slot_h, cold, slot_c, e_in),
                            gain,
                            0,
                        )
                        if best is None or arc.gain > best.gain:
                            best = arc
            if best is None:
                break
            move = best.move
            placement[move.dst_rank].swap(move.quant_type, move.dst_slot, move.expert_id)
            placement[move.src_rank].swap(move.quant_type, move.src_slot, move.evicted_expert)
            costs = _rank_cost(placement, loads, unit_costs)
            mean = sum(costs) / len(costs)
            moves.append(move)

        if not moves:
            return MigrationPlan()
        return MigrationPlan(
            moves=moves,
            expected_cost=costs,
            expected_bytes=[placement[r].bytes_used(nbytes_of) for r in ranks],
        )


# -- helpers ---------------------------------------------------------------


def _used_replica_slots(placement, qtype) -> int:
    used = 0
    for rank, table in placement.items():
        for e in table.slots[("w13", qtype)]:
            if e != -1 and primary_rank(table, e) != rank:
                used += 1
    return used


def _used_replica_slots_all(placement) -> int:
    return sum(_used_replica_slots(placement, qtype) for qtype in QUANT_TYPES)


def _quant_of(placement, expert) -> str | None:
    for rank in sorted(placement):
        qtype = placement[rank].quant_type_of(expert)
        if qtype is not None:
            return qtype
    return None


def _locate(placement, expert, qtype) -> tuple[int, int]:
    for rank in sorted(placement):
        found = placement[rank].slot_of(expert)
        if found is not None:
            return rank, found[1]
    raise ValueError(f"Expert {expert} not resident anywhere.")


def _pick_fill_target(placement, expert, qtype, nbytes_of, byte_budget):
    bytes_e = nbytes_of(expert, qtype)
    best = None
    for rank in sorted(placement):
        table = placement[rank]
        if table.slot_of(expert) is not None:
            continue  # C6: at most one copy per rank
        free = table.free_slots(qtype)
        if not free:
            continue
        used = table.bytes_used(nbytes_of)
        if used + bytes_e > byte_budget:
            continue
        if best is None or used < best[2]:
            best = (rank, free[0], used)
    return (best[0], best[1]) if best else None


def _swap_gain(costs, hot, cold, e_out, e_in, loads, unit_costs) -> float:
    delta_out = loads.get(e_out, 0.0) * unit_costs.get(e_out, 1.0)
    delta_in = loads.get(e_in, 0.0) * unit_costs.get(e_in, 1.0)
    before = max(costs[hot], costs[cold])
    after = max(costs[hot] - delta_out + delta_in, costs[cold] - delta_in + delta_out)
    return before - after
