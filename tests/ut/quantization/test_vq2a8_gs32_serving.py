# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: F811  # Imported pytest fixture is injected by parameter name.
import json
from types import SimpleNamespace

import pytest

from tests.ut.quantization.test_vq2a8_gs32_fixtures import modules, serving_config  # noqa: F401
from tools import serve_vq2a8 as serve


def runtime_config_from_command(command):
    """Feed emitted CLI settings into the real quantization validator on CPU."""
    config = serving_config()

    def value(option):
        return command[command.index(option) + 1]

    config.additional_config = json.loads(value("--additional-config"))
    config.compilation_config = SimpleNamespace(**json.loads(value("--compilation-config")))
    config.model_config.max_model_len = int(value("--max-model-len"))
    config.model_config.enforce_eager = "--enforce-eager" in command
    config.model_config.quantization = value("--quantization")
    config.model_config.hf_config.model_type = "deepseek_v4"
    config.model_config.hf_config.n_routed_experts = 256
    config.model_config.hf_config.num_experts_per_tok = 8
    config.load_config.load_format = value("--load-format")
    config.parallel_config.tensor_parallel_size = int(value("--tensor-parallel-size"))
    config.parallel_config.pipeline_parallel_size = int(value("--pipeline-parallel-size"))
    config.parallel_config.enable_expert_parallel = "--enable-expert-parallel" in command
    if config.parallel_config.enable_expert_parallel:
        config.parallel_config.expert_placement_strategy = value("--expert-placement-strategy")
    config.scheduler_config.max_num_seqs = int(value("--max-num-seqs"))
    config.scheduler_config.max_num_batched_tokens = int(value("--max-num-batched-tokens"))
    config.scheduler_config.enable_chunked_prefill = "--enable-chunked-prefill" in command
    config.scheduler_config.async_scheduling = "--no-async-scheduling" not in command
    config.cache_config.enable_prefix_caching = "--no-enable-prefix-caching" not in command
    return config


@pytest.mark.parametrize("execution", [None, "serial", "batched", "fused", "full_graph"])
@pytest.mark.parametrize("fusion", [None, "none", "swiglu_quant"])
def test_gs32_server_selects_native_stage_and_standard_loader(modules, tmp_path, execution, fusion):
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
        ["--model", str(tmp_path), "--gs32-library", str(library), "--gs32-binary", str(binary), "--dry-run"]
        + ([] if execution is None else ["--gs32-execution", execution])
        + ([] if fusion is None else ["--gs32-fusion", fusion])
    )
    command = serve.build_command(args)
    graph = execution == "full_graph"
    assert ("--enforce-eager" in command) == (not graph)
    compilation = json.loads(command[command.index("--compilation-config") + 1])
    if graph:
        assert compilation == {
            "mode": 0,
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1],
            "max_cudagraph_capture_size": 1,
            "cudagraph_num_of_warmups": 1,
        }
        additional = json.loads(command[command.index("--additional-config") + 1])
        assert additional["ascend_compilation_config"] == {"enable_npugraph_ex": False}
    assert command[command.index("--load-format") + 1] == "safetensors"
    assert command[command.index("--max-model-len") + 1] == "128"
    assert command[command.index("--max-num-batched-tokens") + 1] == "128"
    assert "--no-enable-chunked-prefill" in command
    assert command[command.index("--quantization") + 1] == "vq2a8"
    options = json.loads(command[command.index("--additional-config") + 1])["vq2a8"]
    expected = {"gs32_library": str(library), "gs32_binary": str(binary)}
    if execution in ("batched", "fused", "full_graph"):
        expected["gs32_execution"] = execution
    if fusion == "swiglu_quant":
        expected["gs32_fusion"] = fusion
    assert options == expected
    assert modules.config.validate_vq2a8_config(runtime_config_from_command(command)) == options
    (tmp_path / "config.json").write_text(
        json.dumps({"quantization_config": {"quant_method": "vq2a8", "format": "ascend_vq2a8", "format_version": 1}})
    )
    with pytest.raises(ValueError, match="GS32"):
        serve.build_command(args)


def test_gs32_server_rejects_unknown_fusion_before_loading():
    with pytest.raises(SystemExit) as error:
        serve.parse_args(
            ["--model", "model", "--gs32-library", "stage.so", "--gs32-binary", "stage.o", "--gs32-fusion", "all"]
        )
    assert error.value.code == 2


@pytest.mark.parametrize("execution", ["serial", "batched", "fused", "full_graph"])
def test_long_context_keeps_prefill_workspace_bounded(tmp_path, execution):
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
            "2",
            "--gs32-execution",
            execution,
            "--max-model-len",
            "17408",
            "--dry-run",
        ]
    )
    command = serve.build_command(args)
    assert command[command.index("--max-model-len") + 1] == "17408"
    assert command[command.index("--max-num-batched-tokens") + 1] == "128"
    assert "--enable-chunked-prefill" in command
    assert "--no-enable-chunked-prefill" not in command
    assert "--no-enable-prefix-caching" in command
    assert ("--enforce-eager" in command) == (execution != "full_graph")
    compilation = json.loads(command[command.index("--compilation-config") + 1])
    if execution == "full_graph":
        assert compilation["cudagraph_mode"] == "FULL_DECODE_ONLY"
        assert compilation["cudagraph_capture_sizes"] == [1]
        assert compilation["max_cudagraph_capture_size"] == 1
    assert "--enable-expert-parallel" in command
    assert command[command.index("--max-num-seqs") + 1] == "1"


@pytest.mark.parametrize("context", ["0", "-1"])
def test_nonpositive_context_is_rejected_before_loading_assets(tmp_path, context):
    args = serve.parse_args(
        [
            "--model",
            str(tmp_path),
            "--gs32-library",
            str(tmp_path / "stage.so"),
            "--gs32-binary",
            str(tmp_path / "kernel.o"),
            "--max-model-len",
            context,
        ]
    )
    with pytest.raises(ValueError, match="max_model_len >=1"):
        serve.build_command(args)


@pytest.fixture
def server_assets(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {"quantization_config": {"quant_method": "vq2a8", "format": "vq2_mixed_gs32_source", "format_version": 3}}
        )
    )
    (tmp_path / "model.safetensors.index.json").write_text("{}")
    library, binary = tmp_path / "stage.so", tmp_path / "kernel.o"
    library.touch()
    binary.touch()
    return ["--model", str(tmp_path), "--gs32-library", str(library), "--gs32-binary", str(binary)]


@pytest.mark.parametrize(
    ("context", "budget", "expected", "chunked"),
    [
        (64, None, 64, False),
        (263168, None, 128, True),
        (128, 32, 32, True),
        (263168, 4096, 4096, True),
        (263168, 8192, 8192, True),
        (263168, 65537, 65537, True),
        (263168, 98304, 98304, True),
        (263168, 263168, 263168, True),
        # Preserve explicit vLLM budgets even above the context window.
        (128, 8192, 8192, False),
        (263168, 300000, 300000, True),
    ],
)
@pytest.mark.parametrize("ep_size", [1, 2])
@pytest.mark.parametrize("use_v2_model_runner", [False, True])
def test_scheduler_budget_controls_prefill_without_changing_decode_graph(
    modules, server_assets, context, budget, expected, chunked, ep_size, use_v2_model_runner
):
    arguments = server_assets + [
        "--gs32-execution",
        "full_graph",
        "--max-model-len",
        str(context),
        "--ep-size",
        str(ep_size),
    ]
    if budget is not None:
        arguments += ["--max-num-batched-tokens", str(budget)]
    command = serve.build_command(serve.parse_args(arguments))
    assert command[command.index("--max-num-batched-tokens") + 1] == str(expected)
    assert command[command.index("--max-model-len") + 1] == str(context)
    assert ("--enable-chunked-prefill" in command) is chunked
    assert ("--no-enable-chunked-prefill" in command) is not chunked
    compilation = json.loads(command[command.index("--compilation-config") + 1])
    assert compilation["cudagraph_mode"] == "FULL_DECODE_ONLY"
    assert compilation["cudagraph_capture_sizes"] == [1]
    assert compilation["max_cudagraph_capture_size"] == 1
    runtime = runtime_config_from_command(command)
    runtime.use_v2_model_runner = use_v2_model_runner
    assert modules.config.validate_vq2a8_config(runtime) == runtime.additional_config["vq2a8"]


@pytest.mark.parametrize("budget", ["0", "-1"])
def test_nonpositive_scheduler_budget_is_rejected_before_loading_assets(tmp_path, budget):
    args = serve.parse_args(
        [
            "--model",
            str(tmp_path),
            "--gs32-library",
            str(tmp_path / "stage.so"),
            "--gs32-binary",
            str(tmp_path / "kernel.o"),
            "--max-num-batched-tokens",
            budget,
        ]
    )
    with pytest.raises(ValueError, match="max_num_batched_tokens must be positive"):
        serve.build_command(args)


@pytest.mark.parametrize("budget", ["1.5", "tokens"])
def test_noninteger_scheduler_budget_is_rejected(server_assets, budget):
    with pytest.raises(SystemExit) as error:
        serve.parse_args(server_assets + ["--max-num-batched-tokens", budget])
    assert error.value.code == 2
