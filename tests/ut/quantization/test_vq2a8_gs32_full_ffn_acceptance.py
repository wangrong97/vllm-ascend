# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for full FFN acceptance selection and evidence, without an engine."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.e2e.nightly import test_vq2a8_gs32_ep as ep_probe
from tests.e2e.nightly.test_vq2a8_gs32_full_ffn import select_mixed_layer


def test_mixed_layer_selection_requires_all_pairs_on_one_layer():
    entries = {
        0: {"gate_up": {0: "vq2"}, "down": {0: "vq2"}},
        1: {"gate_up": {0: "vq2", 1: "mxfp4"}, "down": {0: "mxfp4", 1: "mxfp4"}},
    }
    assert select_mixed_layer(entries) is None
    entries[2] = {
        "gate_up": {9: "vq2", 2: "mxfp4", 4: "vq2", 3: "mxfp4"},
        "down": {9: "mxfp4", 2: "mxfp4", 4: "vq2", 3: "vq2"},
    }
    assert select_mixed_layer(entries) == (2, (0, 2, 3))


@pytest.mark.parametrize("execution", ("fused", "full_graph"))
@pytest.mark.parametrize("fault", (None, "missing_launch", "missing_capture", "missing_layer"))
def test_full_ffn_evidence_requires_actual_launch_and_capture(execution, fault):
    state = {
        "layers": [
            {"layer": 0, "fusion": "full_ffn", "expert_ids": [3]},
            {"layer": 1, "fusion": "full_ffn", "expert_ids": []},
        ],
        "full_ffn": [
            {"layer": 0, "submissions": 8, "captures": 7 if execution == "full_graph" else 0},
            {"layer": 1, "submissions": 0, "captures": 0},
        ],
    }
    if fault == "missing_launch":
        state["full_ffn"][0]["submissions"] = 0
    elif fault == "missing_capture":
        state["full_ffn"][0]["captures"] = 0 if execution == "full_graph" else 1
    elif fault == "missing_layer":
        state["full_ffn"].pop()
    if fault:
        with pytest.raises(AssertionError):
            ep_probe.check_full_ffn_evidence([state], execution)
    else:
        ep_probe.check_full_ffn_evidence([state], execution)


def test_ep4_full_ffn_runs_three_isolated_variants(monkeypatch, tmp_path):
    library, binary = tmp_path / "stage.so", tmp_path / "kernel.o"
    library.touch()
    binary.touch()
    options = {
        "--vq2a8-model": str(tmp_path),
        "--vq2a8-gs32-library": str(library),
        "--vq2a8-gs32-binary": str(binary),
    }
    request = SimpleNamespace(config=SimpleNamespace(getoption=lambda name, default=None: options.get(name, default)))
    variants = []

    def execute(command, environment, log, timeout):
        fusion = command[command.index("--fusion") + 1]
        execution = command[command.index("--execution") + 1]
        variants.append((fusion, execution))
        assert command[command.index("--ep-size") + 1] == "4" and "--batch-smoke" in command
        assert environment["VLLM_USE_V2_MODEL_RUNNER"] == "1"
        budgets = [(ep_probe.OUTPUT_TOKENS,) * batch for batch in ep_probe.BATCH_SMOKE_SIZES]
        budgets.append(ep_probe.BATCH_SMOKE_MIXED_OUTPUT_TOKENS)
        report = {
            "fusion": fusion,
            "execution": execution,
            "ep_size": 4,
            "waves": [
                {
                    "batch": len(lengths),
                    "output_lengths": list(lengths),
                    "requests": [
                        {
                            "prompt_token_ids": [1],
                            "token_ids": [2] * length,
                            "logprobs": [{"2": -0.25} for _ in range(length)],
                        }
                        for length in lengths
                    ],
                }
                for lengths in budgets
            ],
        }
        Path(command[command.index("--output") + 1]).write_text(json.dumps(report), encoding="utf-8")
        return 0

    monkeypatch.setattr(ep_probe, "execute", execute)
    ep_probe.test_gs32_ep4_full_ffn_decode_batch_smoke(request, tmp_path)
    assert variants == [("none", "fused"), ("full_ffn", "fused"), ("full_ffn", "full_graph")]
    assert (
        ep_probe.batch_runner_options("fused", library, binary, 4)["additional_config"]["vq2a8"]["gs32_fusion"]
        == "none"
    )
    candidate = ep_probe.batch_runner_options("fused", library, binary, 4, "full_ffn")
    assert candidate["enforce_eager"] and candidate["max_num_seqs"] == 64
    assert candidate["additional_config"]["vq2a8"]["gs32_fusion"] == "full_ffn"
