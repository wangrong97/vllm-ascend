#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serve DeepSeek-V4 VQ2A8 through the standard quantization and MoE pipeline."""

import argparse
import json
import math
import os
import shlex
import sys
from pathlib import Path

DEFAULT_CONTEXT = 128
PREFILL_CHUNK_TOKENS = 128


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--gs32-library", type=Path, required=True, help="GS32 Torch bridge .so")
    parser.add_argument("--gs32-binary", type=Path, required=True, help="mixed_w2w4_moe_ffn.o kernel ELF")
    parser.add_argument(
        "--ep-size",
        type=int,
        choices=(1, 2, 4, 8),
        default=1,
        help="whole-expert parallel ranks; ordinary layers use TP across the same cards",
    )
    parser.add_argument(
        "--gs32-execution",
        choices=("serial", "batched", "fused", "full_graph"),
        default="serial",
        help="eager execution or full_graph for B1 decode with eager chunked prefill",
    )
    parser.add_argument(
        "--gs32-fusion",
        choices=("none", "swiglu_quant"),
        default="none",
        help="keep the current path, or fuse SwiGLU, Down permutation and MXFP8 preparation; independent of execution",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        help="default 128; longer contexts use chunked prefill; full_graph keeps B1 decode graph replay",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        help="positive token budget per scheduler step; default min(max_model_len,128)",
    )
    parser.add_argument("--kv-cache-mib", type=int, default=1024)
    parser.add_argument("--memory-fraction", type=float, default=0.9)
    parser.add_argument("--reserve-gib", type=float, default=16.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--tokenizer-mode",
        choices=("auto", "deepseek_v4"),
        help="optional chat tokenizer override; omitted by default",
    )
    parser.add_argument(
        "--reasoning-parser",
        choices=("deepseek_v4",),
        help="separate reasoning from final answers; thinking mode is selected per request",
    )
    parser.add_argument(
        "--profile-dir", type=Path, help="enable vLLM's Torch profiler controls for a separate trace run"
    )
    check_mode = parser.add_mutually_exclusive_group()
    check_mode.add_argument("--dry-run", action="store_true")
    check_mode.add_argument(
        "--check-native-only",
        action="store_true",
        help="check native operator registration without loading model weights",
    )
    return parser.parse_args(argv)


def build_command(args):
    """Validate model assets and execution limits, then build the vLLM serve command."""
    context = DEFAULT_CONTEXT if args.max_model_len is None else args.max_model_len
    if context < 1:
        raise ValueError("Standard serving requires max_model_len >=1.")
    token_budget = (
        min(context, PREFILL_CHUNK_TOKENS) if args.max_num_batched_tokens is None else args.max_num_batched_tokens
    )
    if token_budget < 1:
        raise ValueError("max_num_batched_tokens must be positive.")
    chunked_prefill = context > DEFAULT_CONTEXT or token_budget < context
    if not math.isfinite(args.memory_fraction) or not 0 < args.memory_fraction <= 1:
        raise ValueError("memory_fraction must be finite and in (0,1].")
    if not math.isfinite(args.reserve_gib) or args.reserve_gib < 1:
        raise ValueError("reserve_gib must be finite and >=1.")
    if args.kv_cache_mib <= 0 or args.kv_cache_mib > args.reserve_gib * 1024:
        raise ValueError("Positive KV cache allocation must fit inside reserve_gib.")
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be in [1,65535].")
    model = args.model.resolve(strict=True)
    if not (model / "config.json").is_file():
        raise ValueError("Model config.json is required.")
    additional = {
        "enable_flashcomm1": False,
        "mix_placement": False,
        "multistream_dsv4_dsa_overlap": False,
    }
    config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    quantization = config.get("quantization_config", {})
    if (
        not isinstance(quantization, dict)
        or quantization.get("quant_method") != "vq2a8"
        or quantization.get("format") != "vq2_mixed_gs32_source"
        or type(quantization.get("format_version")) is not int
        or quantization.get("format_version") != 3
        or set(quantization) != {"quant_method", "format", "format_version"}
        or not (model / "model.safetensors.index.json").is_file()
    ):
        raise ValueError("Select a GS32 source-v3 checkpoint with independently named expert tensors.")
    native = {
        "gs32_library": args.gs32_library.resolve(strict=True),
        "gs32_binary": args.gs32_binary.resolve(strict=True),
    }
    if any(not path.is_file() for path in native.values()):
        raise ValueError("GS32 native library and kernel binary must be regular files.")
    additional["vq2a8"] = {name: str(path) for name, path in native.items()}
    if args.gs32_execution != "serial":
        additional["vq2a8"]["gs32_execution"] = args.gs32_execution
    if args.gs32_fusion != "none":
        additional["vq2a8"]["gs32_fusion"] = args.gs32_fusion
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        str(model),
        "--served-model-name",
        "vq2a8",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--dtype",
        "bfloat16",
        "--load-format",
        "safetensors",
        "--tensor-parallel-size",
        str(args.ep_size),
        "--pipeline-parallel-size",
        "1",
        "--distributed-executor-backend",
        "mp" if args.ep_size > 1 else "uni",
        "--enforce-eager",
        "--compilation-config",
        json.dumps({"mode": 0, "cudagraph_mode": "NONE"}),
        "--no-async-scheduling",
        "--no-enable-prefix-caching",
        "--enable-chunked-prefill" if chunked_prefill else "--no-enable-chunked-prefill",
        "--max-num-seqs",
        "1",
        "--max-model-len",
        str(context),
        "--max-num-batched-tokens",
        str(token_budget),
        "--block-size",
        "128",
        "--kv-cache-memory-bytes",
        str(args.kv_cache_mib * 1024**2),
        "--gpu-memory-utilization",
        str(args.memory_fraction),
        "--seed",
        "0",
        "--disable-log-stats",
        "--generation-config",
        "vllm",
        "--additional-config",
        json.dumps(additional),
    ]
    command.extend(["--quantization", "vq2a8"])
    if args.tokenizer_mode is not None:
        command.extend(["--tokenizer-mode", args.tokenizer_mode])
    if args.reasoning_parser is not None:
        command.extend(["--reasoning-parser", args.reasoning_parser])
    if args.ep_size > 1:
        command.extend(
            [
                "--enable-expert-parallel",
                "--expert-placement-strategy",
                "linear",
                "--all2all-backend",
                "flashinfer_all2allv",
            ]
        )
    if args.gs32_execution == "full_graph":
        command.remove("--enforce-eager")
        command[command.index("--compilation-config") + 1] = json.dumps(
            {
                "mode": 0,
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": [1],
                "max_cudagraph_capture_size": 1,
                "cudagraph_num_of_warmups": 1,
            }
        )
        additional["ascend_compilation_config"] = {"enable_npugraph_ex": False}
        command[command.index("--additional-config") + 1] = json.dumps(additional)
    if args.profile_dir is not None:
        command.extend(
            [
                "--profiler-config",
                json.dumps(
                    {
                        "profiler": "torch",
                        "torch_profiler_dir": str(args.profile_dir.resolve()),
                        "torch_profiler_with_stack": False,
                    }
                ),
            ]
        )
    return command


def check_native_dependencies():
    """Check A5 model bindings before loading weights; this does not execute kernels."""
    # Keep CANN/vLLM initialization out of argument parsing and --dry-run.
    from vllm_ascend.models.deepseek_v4.native_ops import require_deepseek_v4_native_ops

    return require_deepseek_v4_native_ops()


def main(argv=None):
    args = parse_args(argv)
    try:
        command = build_command(args)
        if args.dry_run:
            print(shlex.join(command))
        else:
            extension = check_native_dependencies()
            if args.check_native_only:
                print(f"VQ2A8_NATIVE_CHECK=PASS extension={extension} (registration only; no device execution)")
            else:
                os.execv(command[0], command)
        return 0
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    # Resolve repository modules without putting tools ahead of dependencies.
    sys.path[0] = str(Path(__file__).resolve().parents[1])
    raise SystemExit(main())
