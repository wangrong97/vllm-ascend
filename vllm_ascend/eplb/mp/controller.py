# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mixed-precision EPLB controller: the slow-ring cycle state machine.

Mirrors the ``EplbUpdator`` three-phase cadence (heat collection ->
planning -> layer-wise migration) while staying injectable/mockable: the
runtime hook points (MRV1 forward_before/forward_end, GS32 load accounting)
call into this controller only when the config switch is enabled.

Phases per cycle (design doc 2.2):

    steps [0, heat):            collect slot-level loads (accounting hook)
    step heat-1:                finalize window -> planner inputs
    steps [heat, heat+algo):    plan (ReplicaAllocator + TCSB) off the hot path
    steps after:                layer-by-layer issue/commit migration
    cycle end:                  refresh SplitTable weights from EWMA snapshot

``on_step_begin`` / ``on_step_end`` are the only entry points the runtime
needs; both are no-ops unless ``enabled``.
"""

from __future__ import annotations

from .config import MPEplbOptions
from .cost_model import CostTable, amortized_unit_cost
from .migrator import InMemoryCommBackend, MigrationExecutor
from .placement import BankSlotTable, build_linear_baseline
from .planner import ReplicaAllocator, TCSBPlanner
from .split_table import SplitTable

PHASE_HEAT = "heat"
PHASE_PLAN = "plan"
PHASE_MIGRATE = "migrate"


class MPEplbController:
    """One instance per rank owns the local placement mirror and split tables."""

    def __init__(
        self,
        options: MPEplbOptions,
        num_experts: int,
        ep_size: int,
        ep_rank: int,
        quant_types: dict[str, dict[int, str]],
        num_layers: int,
        layer_load_hook=None,
        comm=None,
        cost_table: CostTable | None = None,
    ):
        self.options = options
        self.enabled = options.enabled
        self.num_experts = num_experts
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.num_layers = num_layers
        # Width-agnostic variant: flat unit costs, byte accounting disabled
        # (the fixed bank capacity already bounds resident memory, I2/I3).
        self.cost_table = cost_table or (
            CostTable.uniform() if options.uniform else CostTable.defaults()
        )

        spare = {q: options.replica_slots[q] // ep_size for q in options.replica_slots}
        self.placement: dict[int, BankSlotTable] = {}
        for rank in range(ep_size):
            self.placement[rank] = build_linear_baseline(
                num_experts, ep_size, rank, quant_types, spare
            )
        self.split_tables = {
            layer: SplitTable(num_experts, self._capacity(), ep_size, options)
            for layer in range(num_layers)
        }
        self.executor = MigrationExecutor(
            self.placement, self.split_tables, comm or InMemoryCommBackend()
        )
        self.replica_allocator = ReplicaAllocator(options, self.cost_table)
        self.tcsb = TCSBPlanner(options, self.cost_table)

        # accounting state
        self._window_rows: dict[int, dict[int, int]] = {e: {} for e in range(num_experts)}
        self._window_launches: dict[int, int] = {e: 0 for e in range(num_experts)}
        self.replica_alloc: dict[int, int] = {}
        self.revoke_history: dict[tuple[int, int], int] = {}
        self._nbytes_fn = None
        self._spare_bytes_by_type = {"vq2": 0, "mxfp4": 0}
        self.byte_budget = 0  # finalized by bind_geometry()

        # cycle state
        self.step = 0
        self.phase = PHASE_HEAT
        self._plan_moves_by_layer: dict[int, list] = {}
        self._migrate_layer = 0
        self.cycle_count = 0
        self.last_expected_cost: list[float] = []
        self._layer_load_hook = layer_load_hook

    # -- construction helpers -------------------------------------------------

    def _capacity(self) -> int:
        per_type = [len(t.slots[("w13", q)]) for t in self.placement.values() for q in ("vq2", "mxfp4")]
        return max(1, max(per_type)) if per_type else 1

    def _default_byte_budget(self) -> int:
        """Rank byte budget B_r: busiest baseline rank plus spare capacity.

        Replicas and staging must fit inside the spare headroom (I2). With no
        geometry bound yet this returns 0 (planning refuses to run).
        """
        if self._nbytes_fn is None:
            return 0
        budgets = []
        for table in self.placement.values():
            active = table.bytes_used(self._nbytes)
            spare = sum(
                len(table.free_slots(q)) * self._spare_bytes_by_type[q] for q in ("vq2", "mxfp4")
            )
            budgets.append(active + spare)
        return max(budgets) if budgets else 0

    def _nbytes(self, expert: int, qtype: str) -> int:
        if self._nbytes_fn is None:
            raise RuntimeError("bind_geometry() must be called before planning.")
        return self._nbytes_fn(expert, qtype)

    def bind_geometry(self, hidden: int, moe_intermediate: int, nbytes_fn=None) -> None:
        """Inject per-expert bundle bytes (hidden/inter from hf config).

        Overrides are accepted for tests or device-measured NZ footprints.
        Uniform mode replaces bytes with per-expert counts of 1, which turns
        every byte constraint into a slot-count constraint — exactly the
        capacity invariant that already holds (I3).
        """
        from .cost_model import bundle_bytes

        if self.options.uniform:
            self._nbytes_fn = nbytes_fn or (lambda expert, qtype: 1)
            self._spare_bytes_by_type = {"vq2": 1, "mxfp4": 1}
            self.byte_budget = self._default_byte_budget()
            return

        gate_up_rows, down_rows = 2 * moe_intermediate, hidden
        cache: dict[tuple[int, str], int] = {}

        def nbytes(expert: int, qtype: str) -> int:
            key = (expert, qtype)
            if key not in cache:
                cache[key] = bundle_bytes(qtype, gate_up_rows, hidden, down_rows)
            return cache[key]

        self._nbytes_fn = nbytes_fn or nbytes
        self._spare_bytes_by_type = {
            q: bundle_bytes(q, gate_up_rows, hidden, down_rows) for q in ("vq2", "mxfp4")
        }
        self.byte_budget = self._default_byte_budget()

    # -- accounting hook (GS32 apply tail; enabled only) ----------------------

    def record_step(self, slot_rows: dict[int, dict[int, int]]) -> None:
        """Accumulate one forward's per-slot row counts.

        ``slot_rows[logical_expert][slot_id] = rows``; launches increment per
        slot with rows > 0 (doc C.5 raw quantities).
        """
        for expert, slots in slot_rows.items():
            for slot, rows in slots.items():
                if rows <= 0:
                    continue
                self._window_rows[expert][slot] = self._window_rows[expert].get(slot, 0) + rows
                self._window_launches[expert] += 1

    # -- cycle machine ----------------------------------------------------------

    def on_step_begin(self) -> None:
        if not (self.enabled and self.options.dynamic):
            return
        if self.phase == PHASE_MIGRATE:
            layer = self._migrate_layer
            moves = self._plan_moves_by_layer.get(layer, [])
            ops = self.executor.prepare_layer(layer, moves, self.options.staging_buffers)
            self.executor.issue(ops)

    def on_step_end(self) -> None:
        if not (self.enabled and self.options.dynamic):
            return
        self.step += 1
        if self.phase == PHASE_HEAT and self._heat_done():
            self._run_planning()
        elif self.phase == PHASE_MIGRATE:
            self.executor.commit_layer(self._migrate_layer)
            self._migrate_layer += 1
            if self._migrate_layer >= self.num_layers:
                self._finish_cycle()

    def _heat_done(self) -> bool:
        return self.step >= self._heat_steps()

    def _heat_steps(self) -> int:
        return max(1, min(64, self.num_layers))  # V1: compact heat window

    def _algo_steps(self) -> int:
        return 2  # planning is synchronous in V1 (still off the forward path)

    def _run_planning(self) -> None:
        import copy

        loads: dict[int, float] = {}
        unit_costs: dict[int, float] = {}
        for expert in range(self.num_experts):
            total = sum(self._window_rows[expert].values())
            launches = max(1, self._window_launches[expert])
            loads[expert] = float(total)
            qtype = self._quant_of(expert) or "vq2"
            unit_costs[expert] = amortized_unit_cost(qtype, total, launches, self.cost_table)

        working = {r: copy.deepcopy(t) for r, t in self.placement.items()}
        excluded: set[int] = set()  # hash layers are filtered by the caller (C5)
        if self.cycle_count % self.options.replica_replan_every == 0:
            moves_a, self.replica_alloc, self.revoke_history = self.replica_allocator.allocate(
                working, loads, self._nbytes, self.byte_budget, excluded,
                self.replica_alloc, self.revoke_history,
            )
        else:
            moves_a = []
        plan = self.tcsb.plan(working, loads, unit_costs, self._nbytes, self.byte_budget)
        all_moves = moves_a + plan.moves
        self.last_expected_cost = plan.expected_cost

        self._plan_moves_by_layer = {layer: [] for layer in range(self.num_layers)}
        for index, move in enumerate(all_moves):
            self._plan_moves_by_layer[index % self.num_layers].append(move)
        self._migrate_layer = 0
        self.phase = PHASE_MIGRATE

    def _finish_cycle(self) -> None:
        # Swap the host mirror to the committed working copy is already done
        # via the executor's placement reference; refresh split weights from
        # the EWMA snapshot (doc C.3 / E1: the only place weights change).
        for split in self.split_tables.values():
            split.refresh_weights(self._rank_shares_snapshot())
        self._window_rows = {e: {} for e in range(self.num_experts)}
        self._window_launches = {e: 0 for e in range(self.num_experts)}
        self.step = 0
        self.phase = PHASE_HEAT
        self.cycle_count += 1

    def _rank_shares_snapshot(self) -> dict[int, float]:
        shares = {r: 0.0 for r in range(self.ep_size)}
        for split in self.split_tables.values():
            for r, v in split.rank_time_shares().items():
                shares[r] = max(shares.get(r, 0.0), v)
        return shares

    def _quant_of(self, expert: int) -> str | None:
        for table in self.placement.values():
            qtype = table.quant_type_of(expert)
            if qtype is not None:
                return qtype
        return None

    # -- static mode -----------------------------------------------------------

    def run_static_once(self) -> list:
        """P0 `static` mode: one planning pass + immediate layer commits."""
        import copy

        loads, unit_costs = self._snapshot_inputs()
        working = {r: copy.deepcopy(t) for r, t in self.placement.items()}
        moves_a, self.replica_alloc, self.revoke_history = self.replica_allocator.allocate(
            working, loads, self._nbytes, self.byte_budget, set(),
            self.replica_alloc, self.revoke_history,
        )
        plan = self.tcsb.plan(working, loads, unit_costs, self._nbytes, self.byte_budget)
        self.last_expected_cost = plan.expected_cost
        moves = moves_a + plan.moves
        results = []
        by_layer = {layer: [] for layer in range(self.num_layers)}
        for index, move in enumerate(moves):
            by_layer[index % self.num_layers].append(move)
        for layer in range(self.num_layers):
            ops = self.executor.prepare_layer(layer, by_layer[layer], self.options.staging_buffers)
            self.executor.issue(ops)
            results.append(self.executor.commit_layer(layer))
        for split in self.split_tables.values():
            split.refresh_weights(self._rank_shares_snapshot())
        return results

    def _snapshot_inputs(self):
        loads: dict[int, float] = {}
        unit_costs: dict[int, float] = {}
        for expert in range(self.num_experts):
            total = sum(self._window_rows[expert].values())
            launches = max(1, self._window_launches[expert])
            loads[expert] = float(total)
            qtype = self._quant_of(expert) or "vq2"
            unit_costs[expert] = amortized_unit_cost(qtype, total, launches, self.cost_table)
        return loads, unit_costs
