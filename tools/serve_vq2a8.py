#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serve DeepSeek-V4 VQ2A8 through the standard quantization and MoE pipeline."""

# Avoid importing the installed Ascend package or tools/bisect during preflight.
# ruff: noqa: E402
import os
import sys

if __name__ == "__main__":
    sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import math
import shlex
import subprocess
from pathlib import Path

from tools.vq2a8_source_contract import source_contract

DEFAULT_CONTEXT = 128
PREFILL_CHUNK_TOKENS = 128
MAX_NUM_SEQS = 64
DP_EP_SIZES = (2, 4, 8)
DSPARK_SPECULATIVE_TOKENS = 5


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--gs32-runtime-dir", type=Path, help="Full-FFN build directory containing the manifest, .so and .o"
    )
    parser.add_argument("--gs32-library", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--gs32-binary", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--gs32-backend", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--enforce-eager", action="store_true", help="Run the same full FFN without decode graphs")
    parser.add_argument(
        "--dspark",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable DSpark using MODEL/dspark by default; --no-dspark selects the ordinary-decode baseline",
    )
    parser.add_argument(
        "--dspark-model",
        type=Path,
        help="Original DeepSeek-V4 FP8 checkpoint or repacked draft directory; also enables DSpark (MRV2 only)",
    )
    parser.add_argument(
        "--num-speculative-tokens",
        type=int,
        help="DSpark proposal count; requires --dspark or --dspark-model and supports 5 (default when enabled)",
    )
    parser.add_argument(
        "--parallel-mode",
        choices=("tp-ep", "dp-ep"),
        default="tp-ep",
        help="tp-ep shards ordinary layers; dp-ep replicates them across MRV2 engines (EP2/EP4/EP8)",
    )
    parser.add_argument(
        "--ep-size",
        type=int,
        choices=(1, 2, 4, 8),
        default=1,
        help="total whole-expert parallel ranks; the selected parallel mode determines TP or DP placement",
    )
    parser.add_argument(
        "--gs32-execution",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--gs32-fusion",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        help="default 128; longer contexts use chunked prefill; full_graph keeps decode graph replay",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        help=(
            "scheduler token budget; defaults to max(max_num_seqs * query_width,min(max_model_len,128)); DSpark width=6"
        ),
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=1,
        help="maximum sequences per engine in [1,64]; DP multiplies the total; graph batches above one require MRV2",
    )
    parser.add_argument("--kv-cache-mib", type=int, default=1024)
    parser.add_argument("--memory-fraction", type=float, default=0.9)
    parser.add_argument(
        "--reserve-gib",
        type=float,
        default=16.0,
        help="validation ceiling for --kv-cache-mib; does not reserve memory or estimate draft/graph memory",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument(
        "--runtime-record", type=Path, help="save resolved serving configuration and native artifact hashes"
    )
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
        help="check model native operators and the selected GS32 bridge ABI without weights or device execution",
    )
    return parser.parse_args(argv)


def decode_capture_sizes(max_num_seqs, num_speculative_tokens=0):
    """Cover every batch with powers of two and an exact final bucket."""
    if type(max_num_seqs) is not int or not 1 <= max_num_seqs <= MAX_NUM_SEQS:
        raise ValueError(f"max_num_seqs must be an integer in [1,{MAX_NUM_SEQS}].")
    if type(num_speculative_tokens) is not int or num_speculative_tokens not in (0, DSPARK_SPECULATIVE_TOKENS):
        raise ValueError("GS32 supports non-speculative decoding or DSpark with 5 speculative tokens.")
    query_width = num_speculative_tokens + 1
    sizes = []
    size = 1
    while size < max_num_seqs:
        sizes.append(size * query_width)
        size *= 2
    return [*sizes, max_num_seqs * query_width]


def dspark_config(args):
    """Keep original draft assets separate from the compressed target checkpoint."""
    if args.dspark is False and (args.dspark_model is not None or args.num_speculative_tokens is not None):
        raise ValueError("--no-dspark conflicts with --dspark-model or --num-speculative-tokens.")
    draft_path = args.dspark_model
    if args.dspark and draft_path is None:
        draft_path = args.model / "dspark"
    if draft_path is None:
        if args.num_speculative_tokens is not None:
            raise ValueError("--num-speculative-tokens requires --dspark-model or --dspark.")
        return None
    count = DSPARK_SPECULATIVE_TOKENS if args.num_speculative_tokens is None else args.num_speculative_tokens
    if count != DSPARK_SPECULATIVE_TOKENS:
        raise ValueError("GS32 DSpark currently requires --num-speculative-tokens 5.")
    if not draft_path.is_dir():
        raise ValueError(
            f"DSpark draft directory not found: {draft_path}. "
            "Create it with tools/repack_vq2a8_dspark.py or pass --dspark-model ORIGINAL."
        )
    model = draft_path.resolve(strict=True)
    config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    quantization = config.get("quantization_config", {})
    if (
        config.get("model_type") != "deepseek_v4"
        or config.get("dspark_block_size") != DSPARK_SPECULATIVE_TOKENS
        or not isinstance(quantization, dict)
        or quantization.get("quant_method") != "fp8"
        or not (model / "model.safetensors.index.json").is_file()
    ):
        raise ValueError(
            "--dspark-model must select a local original DeepSeek-V4 FP8 checkpoint with dspark_block_size=5 "
            "and a weight index."
        )
    if model == args.model.resolve():
        raise ValueError("--dspark-model must be separate from the VQ2A8 target checkpoint.")
    return {
        "method": "dspark",
        "model": str(model),
        "num_speculative_tokens": count,
        "enforce_eager": False,
    }


def build_command(args):
    """Validate model assets and execution limits, then build the vLLM serve command."""
    contract = source_contract()
    if args.enforce_eager:
        if args.gs32_execution not in (None, "fused"):
            raise ValueError("--enforce-eager conflicts with --gs32-execution; omit the legacy execution option")
        args.gs32_execution = "fused"
    resolved = contract["resolve_gs32_options"](
        {
            key: getattr(args, key)
            for key in ("gs32_backend", "gs32_execution", "gs32_fusion")
            if getattr(args, key) is not None
        }
    )
    for key, value in resolved.items():
        setattr(args, key, value)
    if args.gs32_runtime_dir is not None:
        if args.gs32_library is not None or args.gs32_binary is not None:
            raise ValueError("Use --gs32-runtime-dir or explicit --gs32-library/--gs32-binary, not both")
        directory = args.gs32_runtime_dir.resolve(strict=True)
        if not (directory / "build-manifest.json").is_file():
            raise ValueError("Full FFN runtime directory requires build-manifest.json")
        # Resolve locally: manifests may have been produced on another server.
        args.gs32_library = directory / "libvq2a8_gs32_stage.so"
        args.gs32_binary = directory / "mixed_w2w4_moe_ffn.o"
    if args.parallel_mode == "dp-ep" and args.ep_size not in DP_EP_SIZES:
        raise ValueError("GS32 dp-ep requires --ep-size 2, 4 or 8.")
    tp_size = args.ep_size if args.parallel_mode == "tp-ep" else 1
    dp_size = args.ep_size if args.parallel_mode == "dp-ep" else 1
    context = DEFAULT_CONTEXT if args.max_model_len is None else args.max_model_len
    if context < 1:
        raise ValueError("Standard serving requires max_model_len >=1.")
    speculation = dspark_config(args)
    speculative_tokens = speculation["num_speculative_tokens"] if speculation is not None else 0
    capture_sizes = decode_capture_sizes(args.max_num_seqs, speculative_tokens)
    decode_tokens = args.max_num_seqs * (speculative_tokens + 1)
    token_budget = (
        max(decode_tokens, min(context, PREFILL_CHUNK_TOKENS))
        if args.max_num_batched_tokens is None
        else args.max_num_batched_tokens
    )
    if token_budget < 1:
        raise ValueError("max_num_batched_tokens must be positive.")
    if token_budget < decode_tokens:
        raise ValueError("max_num_batched_tokens must be >= max_num_seqs * (num_speculative_tokens + 1).")
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
    folded = contract["validate_source_config"](quantization)
    contract["validate_folded_speculation"](quantization, speculation)
    if folded and (model / "rotation_config.json").exists():
        contract["validate_rotation_config"](json.loads((model / "rotation_config.json").read_text(encoding="utf-8")))
    if not (model / "model.safetensors.index.json").is_file():
        raise ValueError("Select a GS32 source-v3 checkpoint with independently named expert tensors.")
    additional["vq2a8"] = dict(resolved)
    if args.gs32_library is None or args.gs32_binary is None:
        raise ValueError("GS32 full FFN requires --gs32-runtime-dir (or --gs32-library and --gs32-binary).")
    native = {
        "gs32_library": args.gs32_library.resolve(strict=True),
        "gs32_binary": args.gs32_binary.resolve(strict=True),
    }
    if any(not path.is_file() for path in native.values()):
        raise ValueError("GS32 native library and kernel binary must be regular files.")
    additional["vq2a8"].update({name: str(path) for name, path in native.items()})
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
        str(tp_size),
        "--data-parallel-size",
        str(dp_size),
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
        str(args.max_num_seqs),
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
    if dp_size > 1:
        command.extend(
            ["--data-parallel-size-local", str(dp_size), "--data-parallel-backend", "mp", "--api-server-count", "1"]
        )
    command.extend(["--quantization", "vq2a8"])
    if speculation is not None:
        command.extend(["--speculative-config", json.dumps(speculation)])
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
                "cudagraph_capture_sizes": capture_sizes,
                "max_cudagraph_capture_size": capture_sizes[-1],
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


def check_gs32_dependencies(args):
    """Check the full-FFN bridge ABI without initializing a device."""
    # Reuse the build-time ABI checks, keeping Torch imports out of --dry-run.
    from tools.build_vq2a8_gs32_stage import verify_loaded_library, verify_runtime_artifacts

    library = args.gs32_library.resolve(strict=True)
    verify_loaded_library(library, args.gs32_execution, fusion=args.gs32_fusion, binary=args.gs32_binary)
    return verify_runtime_artifacts(library, args.gs32_binary, require_full_ffn=True)


def runtime_record(args, bridge):
    """Record selected server settings, not a claim of kernel execution or API readiness."""
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    record = {
        "schema_version": 1,
        "status": "preflight_passed",
        "model_path": str(args.model.resolve()),
        "gs32_backend": args.gs32_backend,
        "gs32_execution": args.gs32_execution,
        "gs32_fusion": args.gs32_fusion,
        "server_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "host": args.host,
        "port": args.port,
        "native_artifacts": bridge,
        "device_execution_verified": False,
    }
    if args.runtime_record is not None:
        args.runtime_record.parent.mkdir(parents=True, exist_ok=True)
        args.runtime_record.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("VQ2A8_RUNTIME=" + json.dumps(record, ensure_ascii=False), flush=True)
    return record


def main(argv=None):
    args = parse_args(argv)
    try:
        runner = os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")
        if runner not in ("0", "1"):
            raise ValueError("VLLM_USE_V2_MODEL_RUNNER must be 0 (MRV1) or 1 (MRV2).")
        if args.parallel_mode == "dp-ep" and runner != "1":
            raise ValueError("GS32 dp-ep requires MRV2 (VLLM_USE_V2_MODEL_RUNNER=1).")
        if (args.dspark or args.dspark_model is not None) and runner != "1":
            raise ValueError("GS32 DSpark requires MRV2 (VLLM_USE_V2_MODEL_RUNNER=1).")
        command = build_command(args)
        if args.gs32_execution == "full_graph" and args.max_num_seqs > 1 and runner != "1":
            raise ValueError("GS32 full_graph with max_num_seqs >1 requires MRV2 (VLLM_USE_V2_MODEL_RUNNER=1).")
        if args.dry_run:
            print(shlex.join(["env", f"VLLM_USE_V2_MODEL_RUNNER={runner}", *command]))
        else:
            extension = check_native_dependencies()
            bridge = check_gs32_dependencies(args)
            runtime_record(args, bridge)
            if args.check_native_only:
                print(
                    f"VQ2A8_NATIVE_CHECK=PASS extension={extension} gs32_backend={args.gs32_backend} "
                    f"gs32_bridge={bridge} gs32_execution={args.gs32_execution} "
                    "(registration only; no device execution)"
                )
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
