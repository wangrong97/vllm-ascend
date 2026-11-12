# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bundle-level migration executor with a two-phase commit protocol.

Design doc 2.5. The executor is transport-agnostic: a ``CommBackend`` builds
the P2P request list (``plan_ops``) and waits on them. In production the
backend wraps ``dist.batch_isend_irecv`` over the EP group (E3); unit tests
inject an in-memory backend, keeping the commit protocol itself testable.

Move ordering per layer is evict -> swap -> fill so bytes are freed before
they are consumed (D5). The P2 commit (index refresh) is a pure value-level
operation on fixed-shape tensors — the graph-stable core of the design.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import torch

from .cost_model import QUANT_TYPES
from .planner import MOVE_EVICT, MOVE_FILL, MOVE_SWAP, ExpertMove
from .split_table import SplitTable

# Per-type resident field names (doc 2.3 ExpertBundle).
BUNDLE_FIELDS = {
    "vq2": ("packed_zn", "pair_lut", "row_scale_bits", "perm"),
    "mxfp4": ("mxfp4_weight", "mxfp4_scale"),
}
PROJECTIONS = ("w13", "w2")


@dataclass(frozen=True)
class BundleSpec:
    """Field layout of one expert bundle inside the banks."""

    quant_type: str
    expert_id: int
    slot: int

    def field_names(self):
        for proj in PROJECTIONS:
            for name in BUNDLE_FIELDS[self.quant_type]:
                yield proj, name


class MoveStage(Enum):
    WAITING = 0  # waiting for a layer's moves from the planner
    READY = 1  # P2P ops issued, transfer in flight (overlaps forward)
    TRANSFERRING = 2  # transfer done pending commit


@dataclass
class P2POp:
    """Transport-neutral description of one bundle-slice transfer."""

    kind: str  # "send" | "recv"
    peer_rank: int
    expert_id: int
    quant_type: str
    projection: str
    field: str
    staging_index: int = 0


class CommBackend:
    """Interface implemented by the production HCCL backend and test mocks."""

    def issue(self, ops: list[P2POp]) -> object:
        raise NotImplementedError

    def wait(self, handle: object) -> None:
        raise NotImplementedError


class InMemoryCommBackend(CommBackend):
    """Single-rank test backend: transfers resolve immediately."""

    def __init__(self):
        self.issued: list[list[P2POp]] = []

    def issue(self, ops: list[P2POp]) -> object:
        self.issued.append(list(ops))
        return len(self.issued) - 1

    def wait(self, handle: object) -> None:
        return None


@dataclass
class LayerCommitResult:
    layer: int
    applied: list[ExpertMove] = field(default_factory=list)
    skipped: list[ExpertMove] = field(default_factory=list)


class MigrationExecutor:
    """Per-layer issue -> wait -> commit pipeline (doc 2.5 pseudocode)."""

    def __init__(
        self,
        placement,  # dict[rank, BankSlotTable] (host mirror, working copy)
        split_tables: dict[int, SplitTable] | None,
        comm: CommBackend,
        device_index_writer=None,  # callable(rank, qtype) -> device tensors; optional
    ):
        self.placement = placement
        self.split_tables = split_tables or {}
        self.comm = comm
        self.device_index_writer = device_index_writer
        self.state = MoveStage.WAITING
        self._pending: list[ExpertMove] = []
        self._handle = None
        self._staging_used = 0

    # -- issue side (forward_before) ----------------------------------------

    def prepare_layer(self, layer: int, moves: list[ExpertMove], staging_capacity: int) -> list[P2POp]:
        if self.state != MoveStage.WAITING:
            raise RuntimeError("Previous layer commit still in flight.")
        ordered = [m for m in moves if m.kind == MOVE_EVICT]
        ordered += [m for m in moves if m.kind == MOVE_SWAP]
        ordered += [m for m in moves if m.kind == MOVE_FILL]
        self._pending = ordered
        self._staging_used = 0
        ops: list[P2POp] = []
        for move in ordered:
            if move.kind == MOVE_EVICT:
                continue  # no transfer
            for staging_index, (proj, name) in enumerate(self._bundle_of(move)):
                if staging_index >= staging_capacity:
                    break
                ops.append(
                    P2POp(
                        kind="send" if move.kind == MOVE_SWAP else "send",
                        peer_rank=move.dst_rank,
                        expert_id=move.expert_id,
                        quant_type=move.quant_type,
                        projection=proj,
                        field=name,
                        staging_index=staging_index,
                    )
                )
                if move.kind == MOVE_SWAP:
                    ops.append(
                        P2POp(
                            kind="recv",
                            peer_rank=move.src_rank,
                            expert_id=move.evicted_expert,
                            quant_type=move.quant_type,
                            projection=proj,
                            field=name,
                            staging_index=staging_index,
                        )
                    )
        self.state = MoveStage.READY if ops else MoveStage.TRANSFERRING
        return ops

    def issue(self, ops: list[P2POp]) -> None:
        if self.state != MoveStage.READY or not ops:
            return
        self._handle = self.comm.issue(ops)
        self.state = MoveStage.TRANSFERRING

    # -- commit side (forward_end) -------------------------------------------

    def commit_layer(self, layer: int) -> LayerCommitResult:
        if self.state != MoveStage.TRANSFERRING:
            raise RuntimeError("commit_layer requires an issued (or empty) transfer.")
        if self._handle is not None:
            self.comm.wait(self._handle)
            self._handle = None
        result = LayerCommitResult(layer=layer)
        for move in self._pending:
            try:
                self._apply(move)
                result.applied.append(move)
            except ValueError:
                result.skipped.append(move)  # rollback: P1 left untouched
        self._pending = []
        self.state = MoveStage.WAITING
        return result

    # -- P1/P2 application ----------------------------------------------------

    def _bundle_of(self, move: ExpertMove):
        return list(BundleSpec(move.quant_type, move.expert_id, move.src_slot).field_names())

    def _apply(self, move: ExpertMove) -> None:
        table = self.placement[move.dst_rank]
        if move.kind == MOVE_EVICT:
            table.evict(move.quant_type, move.dst_slot, move.expert_id)
            self._split_remove(move.expert_id, move.dst_rank)
        elif move.kind == MOVE_FILL:
            table.fill(move.quant_type, move.dst_slot, move.expert_id)
            self._split_add(move.expert_id, move.dst_rank)
        elif move.kind == MOVE_SWAP:
            swapped_out = table.swap(move.quant_type, move.dst_slot, move.expert_id)
            self.placement[move.src_rank].swap(move.quant_type, move.src_slot, move.evicted_expert)
            # split route: primary ownership follows the swap for both experts
            self._split_remove(swapped_out, move.dst_rank)
            self._split_add(swapped_out, move.src_rank)
            self._split_remove(move.expert_id, move.src_rank)
            self._split_add(move.expert_id, move.dst_rank)
        else:
            raise ValueError(f"Unknown move kind {move.kind!r}.")
        if self.device_index_writer is not None:
            self.device_index_writer(move.dst_rank, move.quant_type)

    def _split_add(self, expert: int, rank: int) -> None:
        for split in self.split_tables.values():
            if 0 <= expert < split.num_experts and rank not in split.route[expert]:
                split.add_replica(expert, rank)

    def _split_remove(self, expert: int, rank: int) -> None:
        for split in self.split_tables.values():
            if 0 <= expert < split.num_experts and rank in split.route[expert]:
                split.remove_replica(expert, rank)
