# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: F811  # Imported pytest fixture is injected by parameter name.
import json
import shlex
from types import SimpleNamespace

import pytest

from tests.ut.quantization.test_vq2a8_gs32_fixtures import modules, serving_config  # noqa: F401
from tools import serve_vq2a8 as serve
from tools.vq2a8_source_contract import source_contract


def test_full_ffn_defaults_resolve_to_native_decode_graph():
    assert source_contract()["resolve_gs32_options"]({}) == {
        "gs32_backend": "native",
        "gs32_fusion": "full_ffn",
        "gs32_execution": "full_graph",
    }


@pytest.mark.parametrize(
    "option,value",
    [
        ("gs32_backend", "reference_mxfp8"),
        ("gs32_fusion", "none"),
        ("gs32_fusion", "swiglu_quant"),
        ("gs32_execution", "serial"),
        ("gs32_execution", "batched"),
    ],
)
def test_retired_serving_modes_are_rejected_without_fallback(option, value):
    with pytest.raises(ValueError, match="full_ffn"):
        source_contract()["resolve_gs32_options"]({option: value})


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
    for option in ("data-parallel-size", "data-parallel-size-local"):
        if f"--{option}" in command:
            setattr(config.parallel_config, option.replace("-", "_"), int(value(f"--{option}")))
    if "--data-parallel-backend" in command:
        config.parallel_config.data_parallel_backend = value("--data-parallel-backend")
    config.parallel_config.pipeline_parallel_size = int(value("--pipeline-parallel-size"))
    config.parallel_config.distributed_executor_backend = value("--distributed-executor-backend")
    config.parallel_config.enable_expert_parallel = "--enable-expert-parallel" in command
    if config.parallel_config.enable_expert_parallel:
        config.parallel_config.expert_placement_strategy = value("--expert-placement-strategy")
    config.scheduler_config.max_num_seqs = int(value("--max-num-seqs"))
    config.scheduler_config.max_num_batched_tokens = int(value("--max-num-batched-tokens"))
    config.scheduler_config.enable_chunked_prefill = "--enable-chunked-prefill" in command
    config.scheduler_config.async_scheduling = "--no-async-scheduling" not in command
    config.cache_config.enable_prefix_caching = "--no-enable-prefix-caching" not in command
    return config


@pytest.mark.parametrize("execution", [None, "fused", "full_graph"])
@pytest.mark.parametrize("fusion", [None, "full_ffn"])
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
    graph = execution in (None, "full_graph")
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
    expected = {
        "gs32_library": str(library),
        "gs32_binary": str(binary),
        "gs32_backend": "native",
        "gs32_fusion": fusion or "full_ffn",
        "gs32_execution": execution or "full_graph",
    }
    assert options == expected
    assert modules.config.validate_vq2a8_config(runtime_config_from_command(command)) == options
    (tmp_path / "config.json").write_text(
        json.dumps({"quantization_config": {"quant_method": "vq2a8", "format": "ascend_vq2a8", "format_version": 1}})
    )
    with pytest.raises(ValueError, match="source-v3"):
        serve.build_command(args)


def test_gs32_server_rejects_unknown_fusion_before_loading():
    args = serve.parse_args(
        ["--model", "model", "--gs32-library", "stage.so", "--gs32-binary", "stage.o", "--gs32-fusion", "all"]
    )
    with pytest.raises(ValueError, match="full_ffn"):
        serve.build_command(args)


@pytest.mark.parametrize(
    ("batch", "buckets"),
    [(1, [1]), (3, [1, 2, 3]), (7, [1, 2, 4, 7]), (8, [1, 2, 4, 8]), (64, [1, 2, 4, 8, 16, 32, 64])],
)
def test_multisequence_launcher_builds_covering_buckets_accepted_by_runtime(modules, server_assets, batch, buckets):
    args = serve.parse_args(server_assets + ["--gs32-execution", "full_graph", "--max-num-seqs", str(batch)])
    command = serve.build_command(args)
    config = runtime_config_from_command(command)
    config.use_v2_model_runner = True
    assert config.scheduler_config.max_num_seqs == batch
    assert config.compilation_config.cudagraph_capture_sizes == buckets
    assert config.compilation_config.max_cudagraph_capture_size == batch
    assert modules.config.validate_vq2a8_config(config) == config.additional_config["vq2a8"]


@pytest.mark.parametrize("dp_size", [2, 4, 8])
@pytest.mark.parametrize("execution", ["fused", "full_graph"])
def test_dp_launcher_settings_pass_runtime_validation(modules, server_assets, dp_size, execution):
    command = serve.build_command(
        serve.parse_args(
            server_assets
            + [
                "--parallel-mode",
                "dp-ep",
                "--ep-size",
                str(dp_size),
                "--gs32-execution",
                execution,
                "--max-num-seqs",
                "16",
                "--max-model-len",
                "65536",
                "--max-num-batched-tokens",
                "2048",
            ]
        )
    )
    config = runtime_config_from_command(command)
    config.use_v2_model_runner = True
    assert config.parallel_config.tensor_parallel_size == 1
    assert config.parallel_config.data_parallel_size == dp_size
    assert config.parallel_config.data_parallel_size_local == dp_size
    assert config.scheduler_config.max_num_seqs == 16
    assert config.scheduler_config.max_num_batched_tokens == 2048
    assert modules.config.validate_vq2a8_config(config) == config.additional_config["vq2a8"]
    if execution == "full_graph":
        assert config.compilation_config.cudagraph_capture_sizes == [1, 2, 4, 8, 16]


@pytest.mark.parametrize("batch", ["0", "-1", "65", "256"])
def test_launcher_rejects_batch_outside_supported_range(server_assets, batch):
    args = serve.parse_args(server_assets + ["--max-num-seqs", batch])
    with pytest.raises(ValueError, match="max_num_seqs must be an integer"):
        serve.build_command(args)


def test_launcher_rejects_explicit_token_budget_smaller_than_batch(server_assets):
    args = serve.parse_args(server_assets + ["--max-num-seqs", "64", "--max-num-batched-tokens", "32"])
    with pytest.raises(ValueError, match="max_num_batched_tokens must be >= max_num_seqs"):
        serve.build_command(args)


def test_launcher_default_token_budget_can_hold_batch_for_short_context(server_assets):
    args = serve.parse_args(server_assets + ["--max-num-seqs", "64", "--max-model-len", "32"])
    command = serve.build_command(args)
    assert command[command.index("--max-num-batched-tokens") + 1] == "64"


@pytest.mark.parametrize("execution", ["fused", "full_graph"])
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
            "--gs32-fusion",
            "full_ffn",
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


def test_chat_frontend_defaults_preserve_existing_service_command(server_assets):
    command = serve.build_command(serve.parse_args(server_assets))
    assert "--tokenizer-mode" not in command
    assert "--reasoning-parser" not in command


@pytest.mark.parametrize(
    "frontend_options",
    [
        ["--tokenizer-mode", "auto"],
        ["--tokenizer-mode", "deepseek_v4"],
        ["--reasoning-parser", "deepseek_v4"],
        ["--tokenizer-mode", "deepseek_v4", "--reasoning-parser", "deepseek_v4"],
    ],
)
def test_chat_frontend_options_reach_dry_run_without_native_loading(
    server_assets, frontend_options, monkeypatch, capsys
):
    def fail_native_loading():
        pytest.fail("A serving dry run must not initialize native model dependencies.")

    monkeypatch.setattr(serve, "check_native_dependencies", fail_native_loading)
    assert serve.main(server_assets + frontend_options + ["--dry-run"]) == 0
    command = shlex.split(capsys.readouterr().out)
    for option, value in zip(frontend_options[::2], frontend_options[1::2]):
        assert command.count(option) == 1
        assert command[command.index(option) + 1] == value
    # Frontend extraction does not set the model's reasoning mode or sampling policy.
    assert "--default-chat-template-kwargs" not in command
    assert command[command.index("--generation-config") + 1] == "vllm"


@pytest.mark.parametrize("option", ["--tokenizer-mode", "--reasoning-parser"])
def test_chat_frontend_rejects_unknown_names_before_loading(server_assets, option):
    with pytest.raises(SystemExit) as error:
        serve.parse_args(server_assets + [option, "unknown"])
    assert error.value.code == 2


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
