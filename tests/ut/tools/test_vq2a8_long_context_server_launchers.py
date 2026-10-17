# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise long-context shell launchers without importing vLLM or using NPUs."""

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
LAUNCHERS = (("64k", 66560), ("256k", 263168), ("1M", 1049600))
pytestmark = pytest.mark.skipif(
    os.name == "nt" or shutil.which("bash") is None,
    reason="Shell launcher execution requires a POSIX environment with bash.",
)


@pytest.fixture
def assets(tmp_path):
    model = tmp_path / "model with spaces"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "max_position_embeddings": 1048576,
                "quantization_config": {
                    "quant_method": "vq2a8",
                    "format": "vq2_mixed_gs32_source",
                    "format_version": 3,
                },
            }
        )
    )
    (model / "model.safetensors.index.json").write_text("{}")
    library = tmp_path / "test library.so"
    binary = tmp_path / "test kernel.o"
    library.touch()
    binary.touch()
    return model, library, binary


def run_launcher(tmp_path, case, assets, *, cli_override=False, extra_args=()):
    model, library, binary = assets
    executable_dir = tmp_path / "bin"
    executable_dir.mkdir(exist_ok=True)
    executable = executable_dir / "python3"
    executable.write_text(
        "#!/bin/sh\n"
        'printf "LAUNCH_ENV=%s,%s,%s,%s\\n" "$ASCEND_RT_VISIBLE_DEVICES" '
        '"$VLLM_WORKER_MULTIPROC_METHOD" "$OMP_NUM_THREADS" "$MKL_NUM_THREADS"\n'
        f'exec {shlex.quote(sys.executable)} "$@"\n'
    )
    executable.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{executable_dir}{os.pathsep}{env.get('PATH', '')}",
            "GS32_OUTPUT": str(model),
            "GS32_EP_LIBRARY": str(library),
            "GS32_EP_BINARY": str(binary),
            "ASCEND_RT_VISIBLE_DEVICES": "4,5",
            "VLLM_WORKER_MULTIPROC_METHOD": "fork",
            "OMP_NUM_THREADS": "12",
            "MKL_NUM_THREADS": "12",
        }
    )
    command = ["bash", str(REPO_ROOT / "tools/tests" / f"serve_vq2a8_{case}_1K.sh"), "--dry-run"]
    if cli_override:
        for name in ("GS32_OUTPUT", "GS32_EP_LIBRARY", "GS32_EP_BINARY"):
            env[name] = str(tmp_path / "missing env asset")
        command.extend(["--model", str(model), "--gs32-library", str(library), "--gs32-binary", str(binary)])
    command.extend(extra_args)
    return subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=20, check=False)


def argument(command, name):
    return command[command.index(name) + 1]


@pytest.mark.parametrize(("case", "context"), LAUNCHERS)
@pytest.mark.parametrize("cli_override", (False, True))
def test_launchers_build_dual_card_decode_graph_command(tmp_path, assets, case, context, cli_override):
    result = run_launcher(tmp_path, case, assets, cli_override=cli_override)
    assert result.returncode == 0, result.stderr
    environment, invocation = result.stdout.splitlines()
    assert environment == "LAUNCH_ENV=4,5,fork,12,12"
    command = shlex.split(invocation)
    model, library, binary = assets
    assert command[command.index("serve") + 1] == str(model)
    assert argument(command, "--max-model-len") == str(context)
    assert argument(command, "--tensor-parallel-size") == "2"
    assert "--enable-expert-parallel" in command
    assert "--enforce-eager" not in command
    assert "--no-enable-prefix-caching" in command
    assert "--enable-chunked-prefill" in command
    assert argument(command, "--max-num-batched-tokens") == ("4096" if case == "256k" else "128")
    assert argument(command, "--max-num-seqs") == "1"
    assert argument(command, "--kv-cache-memory-bytes") == str(16 * 1024**3)
    assert argument(command, "--port") == "8000"
    compilation = json.loads(argument(command, "--compilation-config"))
    assert compilation["cudagraph_mode"] == "FULL_DECODE_ONLY"
    native = json.loads(argument(command, "--additional-config"))["vq2a8"]
    assert native == {
        "gs32_library": str(library),
        "gs32_binary": str(binary),
        "gs32_execution": "full_graph",
    }


@pytest.mark.parametrize(("case", "context"), LAUNCHERS)
def test_launchers_allow_explicit_port_and_cache_budget(tmp_path, assets, case, context):
    result = run_launcher(
        tmp_path,
        case,
        assets,
        extra_args=("--port", "8123", "--kv-cache-mib", "8192", "--reserve-gib", "8"),
    )
    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout.splitlines()[-1])
    assert argument(command, "--max-model-len") == str(context)
    assert argument(command, "--port") == "8123"
    assert argument(command, "--kv-cache-memory-bytes") == str(8 * 1024**3)


@pytest.mark.parametrize("budget", (128, 8192))
def test_256k_launcher_allows_explicit_prefill_budget(tmp_path, assets, budget):
    result = run_launcher(tmp_path, "256k", assets, extra_args=("--max-num-batched-tokens", str(budget)))
    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout.splitlines()[-1])
    assert argument(command, "--max-model-len") == "263168"
    assert argument(command, "--max-num-batched-tokens") == str(budget)
    assert "--enable-chunked-prefill" in command
    compilation = json.loads(argument(command, "--compilation-config"))
    assert compilation["cudagraph_mode"] == "FULL_DECODE_ONLY"
    assert compilation["cudagraph_capture_sizes"] == [1]


@pytest.mark.parametrize(("case", "context"), LAUNCHERS)
@pytest.mark.parametrize("missing_asset", ("config.json", "library", "binary"))
def test_launchers_fail_before_startup_if_an_asset_is_missing(tmp_path, assets, case, context, missing_asset):
    model, library, binary = assets
    paths = {"config.json": model / "config.json", "library": library, "binary": binary}
    paths[missing_asset].unlink()
    result = run_launcher(tmp_path, case, assets)
    assert result.returncode != 0
    assert result.stderr
    assert "vllm.entrypoints.cli.main" not in result.stdout
