# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests: switch parsing, cost model, placement mirror."""

import pytest
import torch

from vllm_ascend.eplb.mp.config import MP_EPLB_MODES, MPEplbOptions, get_mp_eplb_options
from vllm_ascend.eplb.mp.cost_model import (
    CostTable,
    amortized_unit_cost,
    bundle_bytes,
    expert_bytes,
    window_load,
)
from vllm_ascend.eplb.mp.placement import EMPTY_SLOT, build_linear_baseline


class TestSwitch:
    def test_default_off(self):
        options = get_mp_eplb_options({"gs32_library": "/x.so", "gs32_binary": "/x.o"})
        assert options.mode == "off" and not options.enabled and not options.dynamic

    def test_static_and_dynamic(self):
        assert get_mp_eplb_options({"eplb": "static"}).enabled
        assert not get_mp_eplb_options({"eplb": "static"}).dynamic
        assert get_mp_eplb_options({"eplb": "dynamic"}).dynamic

    def test_modes_advertised(self):
        assert MP_EPLB_MODES == ("off", "static", "dynamic", "static_uniform", "dynamic_uniform")

    def test_uniform_modes(self):
        static_u = get_mp_eplb_options({"eplb": "static_uniform"})
        dynamic_u = get_mp_eplb_options({"eplb": "dynamic_uniform"})
        assert static_u.enabled and static_u.uniform and not static_u.dynamic
        assert dynamic_u.enabled and dynamic_u.uniform and dynamic_u.dynamic
        assert not get_mp_eplb_options({"eplb": "dynamic"}).uniform

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError, match="must be one of"):
            get_mp_eplb_options({"eplb": "sometimes"})

    def test_unknown_option_rejected(self):
        with pytest.raises(ValueError, match="Unknown"):
            get_mp_eplb_options({"eplb": "static", "eplb_options": {"nope": 1}})

    def test_wlo_below_whi(self):
        with pytest.raises(ValueError, match="w_lo"):
            get_mp_eplb_options({"eplb": "static", "eplb_options": {"w_lo": 6.0}})

    def test_options_parsed(self):
        options = get_mp_eplb_options(
            {"eplb": "dynamic", "eplb_options": {"K": 3, "R_vq2": 8, "tau_slow": 0.2}}
        )
        assert (options.move_budget_K, options.replica_slots_vq2, options.tau_slow) == (3, 8, 0.2)


class TestCostModel:
    def test_identity_load_times_cost_equals_window_time(self):
        table = CostTable.defaults()
        total, launches = 10_000, 500
        unit = amortized_unit_cost("mxfp4", total, launches, table)
        window_time = total * unit
        alpha = table.alpha_sum("mxfp4")
        beta = table.beta_sum("mxfp4", total / launches)
        assert window_time == pytest.approx(alpha * total + beta * launches)

    def test_zero_launches_falls_back_to_slope(self):
        table = CostTable.defaults()
        assert amortized_unit_cost("vq2", 0, 0, table) == table.alpha_sum("vq2")

    def test_mxfp4_costlier_than_vq2(self):
        table = CostTable.defaults()
        assert (
            amortized_unit_cost("mxfp4", 1000, 1000, table)
            > amortized_unit_cost("vq2", 1000, 1000, table)
        )

    def test_window_load_merges_replicas(self):
        merged = window_load({(7, 0): 10, (7, 1): 5, (9, 3): 2})
        assert merged == {7: 15, 9: 2}

    def test_expert_bytes_formulas(self):
        n, k = 128, 256
        assert expert_bytes("vq2", n, k) == n * k // 4 + n * k // 32 + 8 * (n + k)
        assert expert_bytes("mxfp4", n, k) == n * k // 2 + n * k // 32
        # The design-doc ~1.9x claim on DS4 geometry.
        vq2 = bundle_bytes("vq2", 2 * 2048, 4096, 4096)
        mx4 = bundle_bytes("mxfp4", 2 * 2048, 4096, 4096)
        assert 1.7 < mx4 / vq2 < 2.0


def _quant_types(experts_vq2):
    by_kind = {}
    for kind in ("w13", "w2"):
        by_kind[kind] = {e: ("vq2" if e in experts_vq2 else "mxfp4") for e in range(8)}
    return by_kind


class TestPlacement:
    def test_linear_baseline_capacity_and_spares(self):
        # rank0 owns experts 0..3 (all vq2 with this typing); its mxfp4 bank
        # exists only as spare capacity.
        table = build_linear_baseline(8, 2, 0, _quant_types({0, 1, 2, 3}), {"vq2": 2, "mxfp4": 2})
        assert table.active("w13", "vq2") == [0, 1, 2, 3]
        assert table.active("w13", "mxfp4") == []
        assert table.slots[("w13", "vq2")] == [0, 1, 2, 3, EMPTY_SLOT, EMPTY_SLOT]
        assert table.slots[("w13", "mxfp4")] == [EMPTY_SLOT, EMPTY_SLOT]

    def test_fill_evict_roundtrip(self):
        # rank1 owns 4..7 (all mxfp4); its vq2 bank is spare-only, so a vq2
        # replica of expert 0 lands in the first spare slot (index 0).
        table = build_linear_baseline(8, 2, 1, _quant_types({0, 1, 2, 3}), {"vq2": 2, "mxfp4": 2})
        table.fill("vq2", 0, 0)
        assert table.slot_of(0) == ("vq2", 0)
        table.evict("vq2", 0, 0)
        assert table.slot_of(0) is None

    def test_fill_rejects_occupied_slot(self):
        table = build_linear_baseline(8, 2, 0, _quant_types({0, 1, 2, 3}), {"vq2": 0, "mxfp4": 0})
        with pytest.raises(ValueError):
            table.fill("vq2", 1, 0)

    def test_swap_returns_outgoing(self):
        table = build_linear_baseline(8, 2, 0, _quant_types({0, 1, 2, 3}), {"vq2": 0, "mxfp4": 0})
        outgoing = table.swap("vq2", 1, 6)
        assert outgoing == 1
        assert table.slots[("w13", "vq2")][1] == 6
        assert table.slots[("w2", "vq2")][1] == 6

    def test_device_index_and_bytes(self):
        table = build_linear_baseline(8, 2, 0, _quant_types({0, 1, 2, 3}), {"vq2": 1, "mxfp4": 1})
        expert_ids, logical, mask = table.device_index("vq2")
        assert expert_ids == [0, 1, 2, 3, EMPTY_SLOT]
        assert logical[0] == 0 and mask[0] and not mask[7]
        nbytes = lambda e, q: 10 if q == "vq2" else 19  # noqa: E731
        assert table.bytes_used(nbytes) == 4 * 10
