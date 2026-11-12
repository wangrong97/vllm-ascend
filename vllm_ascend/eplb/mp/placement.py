# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Placement truth for mixed-precision EPLB (host-side mirror).

``BankSlotTable`` is the host mirror of the device-side truth
(``expert_ids_tensor / logical_slots / resident_mask``). Bank capacity is
fixed at startup (active experts + spare replica slots, per quant type) and
never changes at runtime (design doc D / invariant I3), so every move only
rewrites *values*, never shapes.
"""

from __future__ import annotations

from dataclasses import dataclass

from .cost_model import QUANT_TYPES

EMPTY_SLOT = -1


@dataclass
class BankSlotTable:
    """Per-(projection, quant_type) slot occupancy of one rank and layer.

    ``slots[(proj, qtype)]`` is a list of length ``capacity`` whose entry is a
    global logical expert id or ``EMPTY_SLOT``; entries are unique per bank
    (one logical expert at most one slot per rank, invariant I1/C6).
    """

    num_experts: int
    ep_size: int
    ep_rank: int
    capacity: dict[str, int]
    slots: dict[tuple[str, str], list[int]]

    @classmethod
    def create(
        cls,
        local_experts: list[int],
        quant_types: dict[str, dict[int, str]],
        num_experts: int,
        ep_size: int,
        ep_rank: int,
        spare_slots: dict[str, int],
    ) -> "BankSlotTable":
        """Build the linear baseline with spare tail slots (doc 2.5-D).

        ``quant_types[kind][expert]`` follows ``VQ2A8LayerSpec.quant_types``;
        ``spare_slots[qtype]`` is the per-rank replica budget for that type.
        """
        slots: dict[tuple[str, str], list[int]] = {}
        capacity: dict[str, int] = {}
        for kind in ("w13", "w2"):
            for qtype in QUANT_TYPES:
                active = [e for e in sorted(local_experts) if quant_types[kind][e] == qtype]
                spare = spare_slots.get(qtype, 0)
                slots[(kind, qtype)] = active + [EMPTY_SLOT] * spare
                capacity[qtype] = max(capacity.get(qtype, 0), len(active) + spare)
        return cls(
            num_experts=num_experts,
            ep_size=ep_size,
            ep_rank=ep_rank,
            capacity=capacity,
            slots=slots,
        )

    # -- queries -----------------------------------------------------------

    def active(self, proj: str, qtype: str) -> list[int]:
        return [e for e in self.slots[(proj, qtype)] if e != EMPTY_SLOT]

    def slot_of(self, expert: int, proj: str = "w13") -> tuple[str, int] | None:
        """Locate ``expert``; both projections host it in the same-type bank."""
        for qtype in QUANT_TYPES:
            table = self.slots[(proj, qtype)]
            if expert in table:
                return qtype, table.index(expert)
        return None

    def quant_type_of(self, expert: int) -> str | None:
        found = self.slot_of(expert)
        return found[0] if found else None

    def free_slots(self, qtype: str, proj: str = "w13") -> list[int]:
        return [i for i, e in enumerate(self.slots[(proj, qtype)]) if e == EMPTY_SLOT]

    def bytes_used(self, nbytes_of) -> int:
        """Sum bundle bytes of active slots (staging added by the caller)."""
        total = 0
        seen: set[int] = set()
        for (proj, qtype), table in self.slots.items():
            if proj != "w13":
                continue
            for e in table:
                if e != EMPTY_SLOT and e not in seen:
                    seen.add(e)
                    total += nbytes_of(e, qtype)
        return total

    # -- mutations (value-only, shape-preserving; P2 commit side) ----------

    def swap(self, qtype: str, slot: int, incoming_expert: int) -> int:
        """Replace the expert at ``slot`` with ``incoming_expert`` (both
        projections); returns the expert that was swapped out."""
        old = self.slots[("w13", qtype)][slot]
        if old == EMPTY_SLOT:
            raise ValueError(f"swap needs an active slot, slot {slot} is empty.")
        for proj in ("w13", "w2"):
            table = self.slots[(proj, qtype)]
            if table[slot] != old:
                raise ValueError(f"{proj}/{qtype} slot {slot} disagrees with w13.")
            table[slot] = incoming_expert
        return old

    def fill(self, qtype: str, slot: int, expert: int) -> None:
        for proj in ("w13", "w2"):
            table = self.slots[(proj, qtype)]
            if table[slot] != EMPTY_SLOT:
                raise ValueError(f"fill into occupied slot {slot} of {proj}/{qtype}.")
            table[slot] = expert

    def evict(self, qtype: str, slot: int, expert: int) -> None:
        for proj in ("w13", "w2"):
            table = self.slots[(proj, qtype)]
            if table[slot] != expert:
                raise ValueError(f"evict mismatch at {proj}/{qtype} slot {slot}.")
            table[slot] = EMPTY_SLOT

    # -- device index materialization --------------------------------------

    def device_index(self, qtype: str) -> tuple[list[int], list[int], list[bool]]:
        """(expert_ids_tensor, logical_slots, resident_mask) for one bank type.

        ``logical_slots``/``resident_mask`` cover all logical experts; empty
        tail slots carry the ``EMPTY_SLOT`` sentinel in expert_ids.
        """
        expert_ids = list(self.slots[("w13", qtype)])
        logical = [EMPTY_SLOT] * self.num_experts
        mask = [False] * self.num_experts
        for e in expert_ids:
            if e != EMPTY_SLOT:
                logical[e] = self.slots[("w13", qtype)].index(e)
                mask[e] = True
        return expert_ids, logical, mask

    def expert_map(self, global_to_local: dict[int, dict[int, int]]) -> None:
        """No-op placeholder kept for interface parity with upstream EPLB."""
        return None


def build_linear_baseline(
    num_experts: int,
    ep_size: int,
    ep_rank: int,
    quant_types: dict[str, dict[int, str]],
    spare_slots: dict[str, int],
) -> BankSlotTable:
    """Linear interval placement [rank*L, (rank+1)*L) plus spare tail slots."""
    if num_experts % ep_size:
        raise ValueError("Logical expert count must divide ep_size (E4).")
    local_count = num_experts // ep_size
    start = ep_rank * local_count
    local_experts = list(range(start, start + local_count))
    return BankSlotTable.create(
        local_experts, quant_types, num_experts, ep_size, ep_rank, spare_slots
    )
