#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Package the original DeepSeek-V4 DSpark weights without requantization.

Python 3.10+, standard library only. No torch, safetensors, GPU or NPU needed.
The source is the ORIGINAL FP8/FP4 checkpoint, not the VQ2A8 checkpoint.
Copy all mtp.* tensors and the original shared embedding/head/HC parameters
into a new directory, usually TARGET/dspark. Keep the original config and
tokenizer assets. Never modify the source or the target model's root index.

Example:
    python tools/repack_vq2a8_dspark.py --source ORIGINAL --target TARGET \
        --output TARGET/dspark --dry-run
    python tools/repack_vq2a8_dspark.py --source ORIGINAL --target TARGET \
        --output TARGET/dspark
    python tools/repack_vq2a8_dspark.py --output TARGET/dspark --verify-only

Every selected tensor is streamed as bytes and verified by rereading the
output. A successful package is published by renaming a temporary directory.
The report establishes byte preservation, not NPU inference correctness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import struct
import sys
from pathlib import Path

COPY_CHUNK = 8 * 1024 * 1024
HEADER_LIMIT = 100 * 1024 * 1024
SPACE_MARGIN = 64 * 1024 * 1024
REPORT_NAME = "dspark_repack_report.json"
PACKAGE_FORMAT = "deepseek_v4_dspark_original_v1"
DSPARK_BLOCK_SIZE = 5
DEFAULT_MTP_STAGES = 3
SHARED_WEIGHTS = ("embed.weight", "head.weight", "hc_head_fn", "hc_head_base", "hc_head_scale")
SIDECARS = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "tokenizer.model",
    "tokenizer.tiktoken",
)
GEOMETRY_FIELDS = (
    "hidden_size",
    "vocab_size",
    "hc_mult",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "qk_rope_head_dim",
    "q_lora_rank",
    "o_lora_rank",
    "o_groups",
    "n_routed_experts",
    "num_experts_per_tok",
    "moe_intermediate_size",
    "n_shared_experts",
    "num_hash_layers",
)
DTYPE_SIZES = (
    ("BOOL", 1),
    ("I8", 1),
    ("U8", 1),
    ("F8_E4M3", 1),
    ("F8_E5M2", 1),
    ("F8_E8M0", 1),
    ("I16", 2),
    ("U16", 2),
    ("F16", 2),
    ("BF16", 2),
    ("I32", 4),
    ("U32", 4),
    ("F32", 4),
    ("I64", 8),
    ("U64", 8),
    ("F64", 8),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    result = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    require(isinstance(result, dict), f"Expected a JSON object: {path}")
    return result


def local_file(root, relative):
    require(isinstance(relative, str) and relative, "Invalid checkpoint path")
    path = Path(relative)
    require(not path.is_absolute() and ".." not in path.parts, f"Unsafe checkpoint path: {relative}")
    resolved = (root / path).resolve()
    require(resolved.is_relative_to(root) and resolved.is_file(), f"Missing or external file: {relative}")
    return resolved


def file_identity(path):
    info = path.stat()
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def remember(path, snapshots):
    snapshots.setdefault(path, file_identity(path))
    return path


def check_unchanged(snapshots):
    for path, identity in snapshots.items():
        require(file_identity(path) == identity, f"Source or target metadata changed during repack: {path}")


def read_header(path):
    size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(8)
        require(len(prefix) == 8, f"Truncated safetensors prefix: {path}")
        length = struct.unpack("<Q", prefix)[0]
        require(0 < length <= HEADER_LIMIT and 8 + length <= size, f"Invalid safetensors header length: {path}")
        header = json.loads(stream.read(length), object_pairs_hook=unique_object)
    require(isinstance(header, dict), f"Invalid safetensors header: {path}")
    metadata = header.pop("__metadata__", {})
    require(isinstance(metadata, dict), f"Invalid safetensors metadata: {path}")
    spans = []
    sizes = dict(DTYPE_SIZES)
    for name, entry in header.items():
        require(isinstance(entry, dict), f"Invalid tensor entry: {name}")
        shape, offsets = entry.get("shape"), entry.get("data_offsets")
        require(isinstance(shape, list) and all(type(x) is int and x >= 0 for x in shape), f"Invalid shape: {name}")
        require(
            isinstance(offsets, list) and len(offsets) == 2 and all(type(x) is int and x >= 0 for x in offsets),
            f"Invalid data offsets: {name}",
        )
        require(entry.get("dtype") in sizes, f"Unsupported safetensors dtype: {name}")
        begin, end = offsets
        require(end - begin == math.prod(shape) * sizes[entry["dtype"]], f"Tensor byte size mismatch: {name}")
        spans.append((begin, end))
    end = 0
    for begin, next_end in sorted(spans):
        require(begin == end, f"Overlapping or non-contiguous tensor offsets: {path}")
        end = next_end
    require(8 + length + end == size, f"Truncated or trailing safetensors data: {path}")
    return header, 8 + length


def validate_config(config, target=None):
    quant = config.get("quantization_config", {})
    require(
        config.get("model_type") == "deepseek_v4"
        and isinstance(quant, dict)
        and quant.get("quant_method") == "fp8"
        and config.get("expert_dtype", "fp4") == "fp4",
        "Source must be the original DeepSeek-V4 FP8/FP4 checkpoint, not a GS32 target",
    )
    require(config.get("dspark_block_size") == DSPARK_BLOCK_SIZE, "Source must contain dspark_block_size=5")
    require(config.get("sample_from_anchor", True) is True, "Only the original DSpark anchor layout is supported")
    for name in (*GEOMETRY_FIELDS, "dspark_markov_rank", "dspark_noise_token_id"):
        require(type(config.get(name)) is int and config[name] >= 0, f"Invalid source config field: {name}")
    stages = config.get("n_mtp_layers") or config.get("dspark_num_mtp_layers") or DEFAULT_MTP_STAGES
    require(type(stages) is int and stages > 0, "Invalid MTP stage count")
    require(config["n_routed_experts"] > 0 and config["num_hidden_layers"] > 0, "Empty model geometry")
    aux = config.get("dspark_target_layer_ids")
    require(
        isinstance(aux, list)
        and aux
        and all(type(x) is int and 0 <= x < config["num_hidden_layers"] for x in aux)
        and len(set(aux)) == len(aux),
        "Invalid dspark_target_layer_ids",
    )
    ratios = config.get("compress_ratios")
    require(
        isinstance(ratios, list)
        and len(ratios) >= config["num_hidden_layers"] + stages
        and ratios[config["num_hidden_layers"] : config["num_hidden_layers"] + stages] == [0] * stages,
        "Source compress_ratios must include the original uncompressed MTP stages",
    )
    if target is not None:
        for name in GEOMETRY_FIELDS:
            require(config[name] == target.get(name), f"Target/source geometry differs for {name}")
        n_layers = config["num_hidden_layers"]
        require(ratios[:n_layers] == target.get("compress_ratios", [])[:n_layers], "Target attention layout differs")
    return stages


def required_weights(config, stages):
    """The original 0731 DSpark schema, including every routed expert projection."""
    required = set(SHARED_WEIGHTS)
    for stage in range(stages):
        prefix = f"mtp.{stage}."
        suffixes = {
            "hc_attn_base",
            "hc_attn_fn",
            "hc_attn_scale",
            "hc_ffn_base",
            "hc_ffn_fn",
            "hc_ffn_scale",
            "attn.attn_sink",
            "attn.q_norm.weight",
            "attn.kv_norm.weight",
            "attn_norm.weight",
            "ffn_norm.weight",
            "ffn.gate.weight",
            "ffn.gate.bias",
        }
        projections = [f"attn.{p}" for p in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b")]
        projections += [f"ffn.shared_experts.w{p}" for p in (1, 2, 3)]
        projections += [f"ffn.experts.{e}.w{p}" for e in range(config["n_routed_experts"]) for p in (1, 2, 3)]
        suffixes.update(f"{projection}.{field}" for projection in projections for field in ("weight", "scale"))
        required.update(prefix + suffix for suffix in suffixes)
    required.update(("mtp.0.main_proj.weight", "mtp.0.main_proj.scale", "mtp.0.main_norm.weight"))
    required.update(
        f"mtp.{stages - 1}.{suffix}"
        for suffix in (
            "norm.weight",
            "markov_head.markov_w1.weight",
            "markov_head.markov_w2.weight",
            "confidence_head.proj.weight",
            "hc_head_fn",
            "hc_head_base",
            "hc_head_scale",
        )
    )
    return required


def plan_package(source, target=None):
    snapshots = {}
    config_path = remember(local_file(source, "config.json"), snapshots)
    config = read_json(config_path)
    target_config = read_json(remember(target / "config.json", snapshots)) if target is not None else None
    stages = validate_config(config, target_config)
    index = read_json(remember(local_file(source, "model.safetensors.index.json"), snapshots))
    weights = index.get("weight_map")
    require(isinstance(weights, dict) and weights, "Missing source weight_map")
    require(all(isinstance(k, str) and isinstance(v, str) and v for k, v in weights.items()), "Invalid weight_map")
    selected = {k: v for k, v in weights.items() if k.startswith("mtp.") or k in SHARED_WEIGHTS}
    missing = required_weights(config, stages) - selected.keys()
    require(not missing, f"Incomplete original DSpark weights ({len(missing)} missing): {sorted(missing)[:12]}")
    for name in selected:
        if name.startswith("mtp."):
            match = re.fullmatch(r"mtp\.(0|[1-9][0-9]*)\..+", name)
            require(match is not None and int(match[1]) < stages, f"Unexpected MTP stage: {name}")
    shards = []
    source_shards = sorted(set(selected.values()))
    for i, relative in enumerate(source_shards, 1):
        path = remember(local_file(source, relative), snapshots)
        header, base = read_header(path)
        expected = {k for k, v in weights.items() if v == relative}
        require(set(header) == expected, f"Source index/header mismatch: {relative}")
        names = [k for k in header if selected.get(k) == relative]
        shards.append(
            {
                "path": path,
                "source": relative,
                "base": base,
                "header": header,
                "names": names,
                "output": f"model-{i:05d}-of-{len(source_shards):05d}.safetensors",
            }
        )
    sidecars = {name: remember(local_file(source, name), snapshots) for name in SIDECARS if (source / name).exists()}
    check_unchanged(snapshots)
    total = sum(
        s["header"][n]["data_offsets"][1] - s["header"][n]["data_offsets"][0] for s in shards for n in s["names"]
    )
    return {
        "shards": shards,
        "sidecars": sidecars,
        "snapshots": snapshots,
        "summary": {
            "format": PACKAGE_FORMAT,
            "source": str(source),
            "target": str(target) if target is not None else None,
            "stages": stages,
            "mtp_tensors": sum(k.startswith("mtp.") for k in selected),
            "shared_tensors": len(SHARED_WEIGHTS),
            "total_tensors": len(selected),
            "payload_bytes": total,
            "payload_gib": total / 1024**3,
            "source_shards": source_shards,
            "output_shards": len(shards),
        },
    }


def stream_bytes(stream, count, output=None):
    digest = hashlib.sha256()
    while count:
        data = stream.read(min(COPY_CHUNK, count))
        require(data, "Truncated tensor payload while streaming")
        digest.update(data)
        if output is not None:
            output.write(data)
        count -= len(data)
    return digest.hexdigest()


def write_shard(shard, destination):
    header, offset = {}, 0
    for name in shard["names"]:
        entry = shard["header"][name]
        size = entry["data_offsets"][1] - entry["data_offsets"][0]
        header[name] = {"dtype": entry["dtype"], "shape": entry["shape"], "data_offsets": [offset, offset + size]}
        offset += size
    encoded = json.dumps({"__metadata__": {"format": "pt"}, **header}, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    tensors = {}
    with shard["path"].open("rb") as source, destination.open("xb") as output:
        output.write(struct.pack("<Q", len(encoded)))
        output.write(encoded)
        for name, entry in header.items():
            begin, end = shard["header"][name]["data_offsets"]
            source.seek(shard["base"] + begin)
            digest = stream_bytes(source, end - begin, output)
            tensors[name] = {
                "dtype": entry["dtype"],
                "shape": entry["shape"],
                "sha256": digest,
                "source_shard": shard["source"],
                "shard": shard["output"],
            }
        output.flush()
        os.fsync(output.fileno())
    return tensors


def write_json(path, data):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def verify_package(root):
    report = read_json(local_file(root, REPORT_NAME))
    require(report.get("format") == PACKAGE_FORMAT, "Unknown DSpark package format")
    tensors = report["tensors"]
    index = read_json(local_file(root, "model.safetensors.index.json"))
    require(index.get("weight_map") == {n: t["shard"] for n, t in tensors.items()}, "Package index/report mismatch")
    require(len(tensors) == report["total_tensors"], "Package tensor count mismatch")
    for relative in sorted({t["shard"] for t in tensors.values()}):
        path = local_file(root, relative)
        header, base = read_header(path)
        expected = {n for n, t in tensors.items() if t["shard"] == relative}
        require(set(header) == expected, f"Package index/header mismatch: {relative}")
        with path.open("rb") as stream:
            for name, entry in header.items():
                original = tensors[name]
                require(
                    (entry["dtype"], entry["shape"]) == (original["dtype"], original["shape"]),
                    f"Layout changed: {name}",
                )
                begin, end = entry["data_offsets"]
                stream.seek(base + begin)
                require(stream_bytes(stream, end - begin) == original["sha256"], f"SHA256 mismatch: {name}")
        print(f"Verified: {relative}", flush=True)
    for relative, digest in report["sidecar_sha256"].items():
        path = local_file(root, relative)
        with path.open("rb") as stream:
            require(stream_bytes(stream, path.stat().st_size) == digest, f"SHA256 mismatch: {relative}")
    config = read_json(root / "config.json")
    require(not (required_weights(config, validate_config(config)) - tensors.keys()), "Package draft weights missing")
    return report


def repack(source, output, *, target=None, dry_run=False):
    # Check before resolve() so a dangling output symlink cannot redirect writes.
    require(not os.path.lexists(output), f"Output already exists: {output}")
    source, output = source.resolve(), output.resolve()
    target = target.resolve() if target is not None else None
    require(source.is_dir(), f"Source directory not found: {source}")
    require(
        not output.is_relative_to(source) and not source.is_relative_to(output), "Source/output directories overlap"
    )
    if target is not None:
        require(target.is_dir() and not target.is_relative_to(output), "Output must not replace or contain target")
    plan = plan_package(source, target)
    summary = {**plan["summary"], "output": str(output), "dry_run": dry_run}
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if dry_run:
        print("DSPARK_REPACK_STATUS=dry-run", flush=True)
        return summary
    parent = output.parent
    existing = parent
    while not existing.exists():
        existing = existing.parent
    sidecar_bytes = sum(p.stat().st_size for p in plan["sidecars"].values())
    require(
        shutil.disk_usage(existing).free >= summary["payload_bytes"] + sidecar_bytes + SPACE_MARGIN,
        "Insufficient free space",
    )
    parent.mkdir(parents=True, exist_ok=True)
    staging = parent / f".{output.name}.repacking-{os.urandom(8).hex()}"
    staging.mkdir()
    print(f"Temporary output: {staging}", flush=True)
    try:
        tensors = {}
        for shard in plan["shards"]:
            check_unchanged(plan["snapshots"])
            print(f"Extracting: {shard['source']} -> {shard['output']} ({len(shard['names'])} tensors)", flush=True)
            tensors.update(write_shard(shard, staging / shard["output"]))
        sidecar_hashes = {}
        for name, path in plan["sidecars"].items():
            with path.open("rb") as src, (staging / name).open("xb") as dst:
                sidecar_hashes[name] = stream_bytes(src, path.stat().st_size, dst)
                dst.flush()
                os.fsync(dst.fileno())
        write_json(
            staging / "model.safetensors.index.json",
            {
                "metadata": {"total_size": summary["payload_bytes"]},
                "weight_map": {n: t["shard"] for n, t in tensors.items()},
            },
        )
        report = {**summary, "tensors": tensors, "sidecar_sha256": sidecar_hashes}
        write_json(staging / REPORT_NAME, report)
        verify_package(staging)
        check_unchanged(plan["snapshots"])
        require(not os.path.lexists(output), f"Output appeared during repack: {output}")
        staging.rename(output)
    except BaseException:
        print(f"Repack failed; unpublished partial files retained at: {staging}", file=sys.stderr, flush=True)
        raise
    print(f"DSPARK_REPACK_STATUS=verified OUTPUT={output} REPORT={output / REPORT_NAME}", flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, help="Original DeepSeek-V4-Flash-0731 FP8/FP4 checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="New draft directory, normally TARGET/dspark")
    parser.add_argument(
        "--target", type=Path, help="Optional VQ2A8 target directory for geometry checks; never modified"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run", action="store_true", help="Validate config/index/headers and show size without writing"
    )
    mode.add_argument(
        "--verify-only", action="store_true", help="Reread the package and check all recorded SHA256 hashes"
    )
    args = parser.parse_args(argv)
    if args.verify_only:
        if args.source is not None or args.target is not None:
            parser.error("--verify-only uses only --output; omit --source and --target")
    elif args.source is None:
        parser.error("--source is required when creating a draft package")
    try:
        if args.verify_only:
            verify_package(args.output.resolve())
            print(f"DSPARK_REPACK_STATUS=verified OUTPUT={args.output.resolve()}", flush=True)
        else:
            repack(args.source, args.output, target=args.target, dry_run=args.dry_run)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"DSPARK_REPACK_STATUS=failed: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
