# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mixed-precision EPLB for VQ2A8-GS32 (typed swap / replica / token split).

This package implements the design documented in
``EPLB混精结合_规划与权重搬运算法设计.md``:

- a config switch (``additional_config.vq2a8.eplb``) gates every code path;
  with the switch off, the runtime behaves exactly as before;
- a shared typed cost model (``load`` in tokens, ``c_type`` in time/token);
- a two-stage slow-ring planner (``ReplicaAllocator`` + ``TCSB``);
- a fast-ring ``SplitTable`` (global-consistent weighted deterministic split);
- a bundle-level migration plan and a two-phase commit protocol.

Runtime wiring that requires a live multi-rank service (dispatcher
expert_map, MRV1 forward hooks, real HCCL P2P) is intentionally isolated
behind injectable interfaces so the algorithm layer stays CPU-testable.
"""

from .config import MP_EPLB_MODES, MPEplbOptions, get_mp_eplb_options
from .controller import MPEplbController
from .cost_model import CostTable, amortized_unit_cost, window_load
from .migrator import BundleSpec, MigrationExecutor, MoveStage
from .placement import BankSlotTable, build_linear_baseline
from .planner import MigrationPlan, ReplicaAllocator, TCSBPlanner, TCSBArc
from .split_table import SplitTable, weighted_pick

__all__ = [
    "MP_EPLB_MODES",
    "MPEplbOptions",
    "get_mp_eplb_options",
    "MPEplbController",
    "CostTable",
    "amortized_unit_cost",
    "window_load",
    "BundleSpec",
    "MigrationExecutor",
    "MoveStage",
    "BankSlotTable",
    "build_linear_baseline",
    "MigrationPlan",
    "ReplicaAllocator",
    "TCSBPlanner",
    "TCSBArc",
    "SplitTable",
    "weighted_pick",
]
