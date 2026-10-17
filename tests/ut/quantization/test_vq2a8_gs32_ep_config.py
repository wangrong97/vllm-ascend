# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EP keeps whole experts; TP is used only by ordinary model layers."""
# ruff: noqa: F811

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from tests.ut.quantization.test_vq2a8_gs32_fixtures import modules  # noqa: F401
from tests.ut.quantization.test_vq2a8_gs32_graph_config import graph_config, long_graph_config
from tools import serve_vq2a8 as serve


def ep_config(size):
    config = graph_config()
    config.model_config.hf_config.model_type = "deepseek_v4"
    config.model_config.hf_config.n_routed_experts = 256
    config.parallel_config.tensor_parallel_size = size
    config.parallel_config.enable_expert_parallel = True
    config.parallel_config.expert_placement_strategy = "linear"
    return config


@pytest.mark.parametrize("size", [2, 4, 8])
def test_gs32_ep_config_accepts_whole_expert_parallelism(modules, size):
    config = ep_config(size)
    assert modules.config.validate_vq2a8_config(config) == config.additional_config["vq2a8"]


@pytest.mark.parametrize("size", [2, 4, 8])
@pytest.mark.parametrize("use_v2", [False, True])
def test_gs32_ep_long_context_uses_existing_eager_config(modules, size, use_v2):
    config = ep_config(size)
    config.use_v2_model_runner = use_v2
    config.additional_config["vq2a8"]["gs32_execution"] = "fused"
    config.model_config.enforce_eager = True
    config.model_config.max_model_len = 17408
    config.compilation_config.cudagraph_mode = "NONE"
    config.compilation_config.cudagraph_capture_sizes = []
    config.scheduler_config.enable_chunked_prefill = True
    config.scheduler_config.max_num_batched_tokens = 128
    assert modules.config.validate_vq2a8_config(config) == config.additional_config["vq2a8"]


@pytest.mark.parametrize(
    "fault", ["tp_only", "dp", "pp", "cp", "size", "uneven", "round_robin", "eplb", "sp", "shared_dp", "dsa_cp"]
)
@pytest.mark.parametrize("use_v2", [False, True])
def test_gs32_ep_config_rejects_incompatible_parallel_layouts(modules, fault, use_v2):
    config = ep_config(4)
    config.use_v2_model_runner = use_v2
    parallel = config.parallel_config
    if fault == "tp_only":
        parallel.enable_expert_parallel = False
    elif fault == "dp":
        parallel.data_parallel_size = 2
    elif fault == "pp":
        parallel.pipeline_parallel_size = 2
    elif fault == "cp":
        parallel.prefill_context_parallel_size = 2
    elif fault == "size":
        parallel.tensor_parallel_size = 3
    elif fault == "uneven":
        config.model_config.hf_config.n_routed_experts = 255
    elif fault == "round_robin":
        parallel.expert_placement_strategy = "round_robin"
    elif fault == "eplb":
        parallel.enable_eplb = True
    elif fault == "sp":
        parallel.use_sequence_parallel_moe = True
    else:
        config.additional_config["enable_shared_expert_dp" if fault == "shared_dp" else "enable_dsa_cp"] = True
    with pytest.raises(ValueError):
        modules.config.validate_vq2a8_config(config)


@pytest.mark.parametrize("size", [1, 2, 4, 8])
def test_gs32_ep_serving_selects_mp_and_complete_experts(tmp_path, size):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {"quantization_config": {"quant_method": "vq2a8", "format": "vq2_mixed_gs32_source", "format_version": 3}}
        )
    )
    (tmp_path / "model.safetensors.index.json").write_text("{}")
    library, binary = tmp_path / "stage.so", tmp_path / "kernel.o"
    library.touch()
    binary.touch()
    args = serve.parse_args(
        [
            "--model",
            str(tmp_path),
            "--gs32-library",
            str(library),
            "--gs32-binary",
            str(binary),
            "--ep-size",
            str(size),
            "--gs32-execution",
            "full_graph",
            "--dry-run",
        ]
    )
    command = serve.build_command(args)
    assert command[command.index("--tensor-parallel-size") + 1] == str(size)
    assert command[command.index("--distributed-executor-backend") + 1] == ("mp" if size > 1 else "uni")
    assert ("--enable-expert-parallel" in command) == (size > 1)
    if size > 1:
        assert command[command.index("--expert-placement-strategy") + 1] == "linear"
    assert "--no-enable-prefix-caching" in command


@pytest.mark.parametrize("location", ["parallel", "additional"])
def test_gs32_ep_rejects_redundant_experts_before_allocating(modules, location):
    from types import SimpleNamespace

    config = ep_config(4)
    if location == "parallel":
        config.parallel_config.eplb_config = SimpleNamespace(num_redundant_experts=4)
    else:
        config.additional_config["eplb_config"] = {"num_redundant_experts": 4}
    with pytest.raises(ValueError, match="redundan|EPLB"):
        modules.config.validate_vq2a8_config(config)


@pytest.mark.parametrize("field,value", [("dynamic_eplb", True), ("expert_map_path", "/tmp/custom_expert_map")])
def test_gs32_ep_rejects_indirect_expert_remapping(modules, field, value):
    config = ep_config(4)
    config.additional_config["eplb_config"] = {field: value}
    with pytest.raises(ValueError, match="redundan|EPLB"):
        modules.config.validate_vq2a8_config(config)


@pytest.mark.parametrize("size", [2, 4, 8])
@pytest.mark.parametrize("budget", [128, 4096, 8192, 98304])
@pytest.mark.parametrize("use_v2", [False, True])
def test_gs32_ep_long_context_reuses_b1_decode_graph(modules, size, budget, use_v2):
    config = long_graph_config()
    config.use_v2_model_runner = use_v2
    config.model_config.max_model_len = 263168
    config.scheduler_config.max_num_batched_tokens = budget
    config.model_config.hf_config.n_routed_experts = 256
    config.parallel_config.tensor_parallel_size = size
    config.parallel_config.enable_expert_parallel = True
    config.parallel_config.expert_placement_strategy = "linear"
    assert modules.config.validate_vq2a8_config(config) == config.additional_config["vq2a8"]


@pytest.fixture
def long_context_probe():
    path = Path(__file__).resolve().parents[3] / "tests/e2e/nightly/test_vq2a8_gs32_ep_long_context.py"
    spec = importlib.util.spec_from_file_location("gs32_ep_long_context_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("size", [2, 4, 8])
def test_long_context_probe_uses_decode_graph_and_bounded_prefill(long_context_probe, size):
    options = long_context_probe.runner_options("/tmp/stage.so", "/tmp/kernel.o", size)
    assert options["tensor_parallel_size"] == size
    assert options["enable_expert_parallel"] is True
    assert options["max_model_len"] == 17408
    assert options["max_num_batched_tokens"] == 128
    assert options["max_num_seqs"] == 1
    assert options["enable_chunked_prefill"] is True
    assert options["enable_prefix_caching"] is False
    assert options["enforce_eager"] is False
    assert options["compilation_config"]["cudagraph_mode"] == "FULL_DECODE_ONLY"
    assert options["compilation_config"]["cudagraph_capture_sizes"] == [1]
    assert options["additional_config"]["vq2a8"]["gs32_execution"] == "full_graph"
    assert options["worker_extension_cls"].endswith(".GS32EPWorkerExtension")


@pytest.mark.parametrize("fault", [None, "rank", "inventory", "owners", "capture", "replay", "prefill"])
def test_long_context_probe_requires_each_rank_to_replay_decode(long_context_probe, fault):
    before = [
        {
            "rank": rank,
            "layers": [{"layer": 0, "expert_ids": [rank]}],
            "validity_owners": 1,
            "decode": {"captures": 1, "replays": 0},
            "prefill": {"captures": 0, "replays": 0},
            "other": {"captures": 0, "replays": 0},
        }
        for rank in range(2)
    ]
    after = copy.deepcopy(before)
    for state in after:
        state["decode"]["replays"] = 1023
    failed = after[-1]
    if fault == "rank":
        failed["rank"] = 0
    elif fault == "inventory":
        failed["layers"] = []
    elif fault == "owners":
        failed["validity_owners"] = 0
    elif fault == "capture":
        failed["decode"]["captures"] = 2
    elif fault == "replay":
        failed["decode"]["replays"] = 1022
    elif fault == "prefill":
        failed["prefill"]["captures"] = 1
    if fault:
        with pytest.raises(AssertionError):
            long_context_probe.check_graph_progress(before, after, 1, 1024)
    else:
        long_context_probe.check_graph_progress(before, after, 1, 1024)


@pytest.mark.parametrize("version", [1, 2])
def test_long_context_probe_requires_requested_runner_on_every_rank(long_context_probe, version):
    states = [
        {"rank": rank, "model_runner_version": version, "model_runner": f"runner_v{version}.NPUModelRunner"}
        for rank in (0, 1)
    ]
    long_context_probe.check_runner_selection(states, version)
    states[-1]["model_runner_version"] = 3 - version
    with pytest.raises(AssertionError, match="Rank 1 selected"):
        long_context_probe.check_runner_selection(states, version)
