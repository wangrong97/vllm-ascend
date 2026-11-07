#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Convert natural-column source-v3 non-routed weights to the export contract.

Use --dry-run to inspect the plan. Actual conversion requires --quantize-fp8:
365 BF16 matrices in the full 43-layer checkpoint are quantized to E4M3FN
with E8M0 scales per 128x128 block. Norms must round-trip exactly through
BF16. Tensor names become original DeepSeek V4 names. Routed bytes, other
weights, config and tokenizer are preserved. CPU PyTorch and NumPy suffice.

An independent output directory is required. Every written tensor is reread
and SHA256 verified before publishing the output. A failure leaves the staging
directory for inspection and never modifies the source checkpoint.
"""

from __future__ import annotations

# Direct execution must not let tools/bisect shadow the standard library.
# ruff: noqa: E402
import os
import sys

if __name__ == "__main__":
    sys.path[:] = [os.path.dirname(os.path.dirname(os.path.abspath(__file__)))] + [
        path for path in sys.path if os.path.abspath(path) != os.path.dirname(os.path.abspath(__file__))
    ]

import argparse
import hashlib
import json
import math
import re
import shutil
import struct
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch

from tools.convert_vq2_mixed_gs32_v3 import (
    COPY_CHUNK,
    copy_verified,
    dtype_size,
    local_file,
    read_header,
    read_json,
    require,
    snapshot,
    write_json,
)

BLOCK = 128
FP8_MAX = 448.0
SPEC_NAME = "vq2_mixed_gs32_export_spec_20261103.md"
REPORT_NAME = "repack_report.json"
SOURCE_CONFIG = (("quant_method", "vq2a8"), ("format", "vq2_mixed_gs32_source"), ("format_version", 3))
SIDECARS = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "experts.json")
ROOT_NAMES = (
    ("model.embed_tokens.weight", "embed.weight"),
    ("head.weight", "head.weight"),
    ("model.norm.weight", "norm.weight"),
    ("model.hc_head.hc_fn", "hc_head_fn"),
    ("model.hc_head.hc_base", "hc_head_base"),
    ("model.hc_head.hc_scale", "hc_head_scale"),
)
MODULE_NAMES = (
    ("self_attn.q_a_proj", "attn.wq_a"),
    ("self_attn.q_b_proj", "attn.wq_b"),
    ("self_attn.kv_proj", "attn.wkv"),
    ("self_attn.o_a_proj", "attn.wo_a"),
    ("self_attn.o_b_proj", "attn.wo_b"),
    ("self_attn.q_a_norm", "attn.q_norm"),
    ("self_attn.kv_norm", "attn.kv_norm"),
    ("self_attn.compressor.gate_proj", "attn.compressor.wgate"),
    ("self_attn.compressor.kv_proj", "attn.compressor.wkv"),
    ("self_attn.compressor.kv_norm", "attn.compressor.norm"),
    ("self_attn.compressor.indexer.q_b_proj", "attn.indexer.wq_b"),
    ("self_attn.compressor.indexer.scorer.weights_proj", "attn.indexer.weights_proj"),
    ("self_attn.compressor.indexer.gate_proj", "attn.indexer.compressor.wgate"),
    ("self_attn.compressor.indexer.kv_proj", "attn.indexer.compressor.wkv"),
    ("self_attn.compressor.indexer.kv_norm", "attn.indexer.compressor.norm"),
    ("mlp.shared_experts.gate_proj", "ffn.shared_experts.w1"),
    ("mlp.shared_experts.up_proj", "ffn.shared_experts.w3"),
    ("mlp.shared_experts.down_proj", "ffn.shared_experts.w2"),
    ("mlp.gate", "ffn.gate"),
    ("input_layernorm", "attn_norm"),
    ("post_attention_layernorm", "ffn_norm"),
)
DIRECT_NAMES = (
    ("self_attn.sinks", "attn.attn_sink"),
    ("self_attn.compressor.position_bias", "attn.compressor.ape"),
    ("self_attn.compressor.indexer.position_bias", "attn.indexer.compressor.ape"),
    ("mlp.gate.e_score_correction_bias", "ffn.gate.bias"),
    ("mlp.gate.tid2eid", "ffn.gate.tid2eid"),
    ("attn_hc.fn", "hc_attn_fn"),
    ("attn_hc.base", "hc_attn_base"),
    ("attn_hc.scale", "hc_attn_scale"),
    ("ffn_hc.fn", "hc_ffn_fn"),
    ("ffn_hc.base", "hc_ffn_base"),
    ("ffn_hc.scale", "hc_ffn_scale"),
)


def original_name(name: str) -> str:
    """Map only known non-routed HF aliases, with no tensor layout change."""
    for source, target in ROOT_NAMES:
        if name in (source, target):
            return target
    match = re.fullmatch(r"(model\.)?layers\.(0|[1-9][0-9]*)\.(.+)", name)
    require(match is not None, f"Unknown non-routed name: {name}")
    prefix, suffix = f"layers.{match[2]}.", match[3]
    for source, target in DIRECT_NAMES:
        if suffix in (source, target):
            return prefix + target
    module, _, field = suffix.rpartition(".")
    for source, target in MODULE_NAMES:
        if module in (source, target) and field == "weight":
            return prefix + target + ".weight"
    raise ValueError(f"Unknown non-routed name or already quantized field: {name}")


def expected_roots(config: dict) -> dict:
    """Return original-name target (dtype, shape, action) for the main model."""
    hidden, intermediate = config["hidden_size"], config["moe_intermediate_size"]
    layers, experts = config["num_hidden_layers"], config["n_routed_experts"]
    vocab = config.get("vocab_size", 128)
    heads, head_dim = config.get("num_attention_heads", 1), config.get("head_dim", hidden)
    q_rank, o_rank = config.get("q_lora_rank", hidden), config.get("o_lora_rank", hidden)
    o_groups, hc = config.get("o_groups", 1), config.get("hc_mult", 4)
    hash_layers = config.get("num_hash_layers", 0)
    ratios = config.get("compress_ratios", [0] * layers)
    require(len(ratios) == layers, "Only main-model compress_ratios are supported (no MTP stages)")
    require(0 <= hash_layers <= layers and heads * head_dim % o_groups == 0, "Invalid model geometry")
    require(config.get("n_shared_experts", 1) == 1, "Exactly one shared expert is supported")
    roots = {}

    def add(name, dtype, shape, action="copy"):
        require(all(type(value) is int and value > 0 for value in shape), f"Invalid shape: {name}")
        roots[name] = (dtype, tuple(shape), action)

    add("embed.weight", "BF16", (vocab, hidden))
    add("head.weight", "BF16", (vocab, hidden))
    add("norm.weight", "BF16", (hidden,), "norm_cast")
    add("hc_head_fn", "F32", (hc, hc * hidden))
    add("hc_head_base", "F32", (hc,))
    add("hc_head_scale", "F32", (1,))
    for layer in range(layers):
        prefix = f"layers.{layer}."
        for module, shape in (
            ("attn.wq_a", (q_rank, hidden)),
            ("attn.wq_b", (heads * head_dim, q_rank)),
            ("attn.wkv", (head_dim, hidden)),
            ("attn.wo_a", (o_groups * o_rank, heads * head_dim // o_groups)),
            ("attn.wo_b", (hidden, o_groups * o_rank)),
            ("ffn.shared_experts.w1", (intermediate, hidden)),
            ("ffn.shared_experts.w3", (intermediate, hidden)),
            ("ffn.shared_experts.w2", (hidden, intermediate)),
        ):
            add(prefix + module + ".weight", "F8_E4M3", shape, "quantize_fp8")
        for module, width in (
            ("attn.q_norm", q_rank),
            ("attn.kv_norm", head_dim),
            ("attn_norm", hidden),
            ("ffn_norm", hidden),
        ):
            add(prefix + module + ".weight", "BF16", (width,), "norm_cast")
        add(prefix + "attn.attn_sink", "F32", (heads,))
        add(prefix + "ffn.gate.weight", "BF16", (experts, hidden))
        if layer < hash_layers:
            add(prefix + "ffn.gate.tid2eid", "I64", (vocab, config.get("num_experts_per_tok", 1)))
        else:
            add(prefix + "ffn.gate.bias", "F32", (experts,))
        for part in ("attn", "ffn"):
            add(prefix + f"hc_{part}_fn", "F32", (hc * (hc + 2), hc * hidden))
            add(prefix + f"hc_{part}_base", "F32", (hc * (hc + 2),))
            add(prefix + f"hc_{part}_scale", "F32", (3,))
        ratio = ratios[layer]
        require(ratio in (0, 4, 128), f"Unsupported compressor ratio: {ratio}")
        if ratio:
            width = head_dim * (2 if ratio == 4 else 1)
            add(prefix + "attn.compressor.ape", "F32", (ratio, width))
            add(prefix + "attn.compressor.norm.weight", "BF16", (head_dim,), "norm_cast")
            for part in ("wgate", "wkv"):
                add(prefix + f"attn.compressor.{part}.weight", "BF16", (width, hidden))
        if ratio == 4:
            index_dim, index_heads = config["index_head_dim"], config["index_n_heads"]
            add(prefix + "attn.indexer.wq_b.weight", "F8_E4M3", (index_heads * index_dim, q_rank), "quantize_fp8")
            add(prefix + "attn.indexer.weights_proj.weight", "BF16", (index_heads, hidden))
            add(prefix + "attn.indexer.compressor.ape", "F32", (ratio, 2 * index_dim))
            add(prefix + "attn.indexer.compressor.norm.weight", "BF16", (index_dim,), "norm_cast")
            for part in ("wgate", "wkv"):
                add(prefix + f"attn.indexer.compressor.{part}.weight", "BF16", (2 * index_dim, hidden))
    return roots


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(COPY_CHUNK):
            digest.update(data)
    return digest.hexdigest()


def plan_model(source: Path) -> dict:
    source = source.resolve()
    require(source.is_dir(), f"Source does not exist: {source}")
    source_files = [path for path in source.rglob("*") if path.is_file()]
    before = snapshot(source_files)
    config = read_json(local_file(source, "config.json"))
    quant = config.get("quantization_config")
    require(
        quant == dict(SOURCE_CONFIG) and type(quant.get("format_version")) is int,
        "Expected exactly the three-key source-v3 quantization_config",
    )
    for key in ("hidden_size", "moe_intermediate_size", "num_hidden_layers", "n_routed_experts"):
        require(type(config.get(key)) is int and config[key] > 0, f"Invalid config {key}")
    require(
        config["hidden_size"] % BLOCK == config["moe_intermediate_size"] % BLOCK == 0,
        "Hidden and intermediate sizes must be multiples of 128",
    )
    index_path = local_file(source, "model.safetensors.index.json")
    index = read_json(index_path).get("weight_map")
    require(isinstance(index, dict) and index, "Missing nonempty weight_map")
    require(all(isinstance(k, str) and isinstance(v, str) for k, v in index.items()), "Invalid weight_map")
    allowed = set(SIDECARS) | {"model.safetensors.index.json", SPEC_NAME} | set(index.values())
    for path in source_files:
        rel = path.relative_to(source).as_posix()
        require(rel in allowed, f"Unknown source file, not silently discarded: {rel}")
        local_file(source, rel)
    roots = expected_roots(config)
    seen_roots, expert_fields = set(), {}
    shards, output_names = [], set()
    source_bytes = target_bytes = routed_bytes = 0
    counts = Counter()
    for rel in sorted(set(index.values())):
        require(rel.endswith(".safetensors"), f"Invalid shard extension: {rel}")
        path = local_file(source, rel)
        header, base = read_header(path)
        require(
            set(header) - {"__metadata__"} == {name for name, shard in index.items() if shard == rel},
            f"Index/header mismatch: {rel}",
        )
        jobs, target_header, offset = [], {"__metadata__": {"format": "pt"}}, 0
        for name, entry in sorted(
            ((n, e) for n, e in header.items() if n != "__metadata__"), key=lambda item: item[1]["data_offsets"][0]
        ):
            require("perm" not in name.split(".") and not name.endswith("_perm"), f"perm is unsupported: {name}")
            require(not name.startswith(("mtp.", "model.mtp.")), f"MTP is unsupported: {name}")
            dtype, shape = entry["dtype"], tuple(entry["shape"])
            start, end = entry["data_offsets"]
            expert = re.fullmatch(r"layers\.(0|[1-9][0-9]*)\.ffn\.experts\.(0|[1-9][0-9]*)\.(w13|w2)\.([a-z_]+)", name)
            if expert:
                layer, eid, projection, field = int(expert[1]), int(expert[2]), expert[3], expert[4]
                require(
                    layer < config["num_hidden_layers"] and eid < config["n_routed_experts"],
                    f"Out-of-range routed expert: {name}",
                )
                expert_fields.setdefault((layer, eid, projection), {})[field] = (dtype, shape)
                target, target_dtype, action = name, dtype, "routed_copy"
                routed_bytes += end - start
            else:
                require(".experts." not in name and ".experts_" not in name, f"Malformed routed name: {name}")
                target = original_name(name)
                require(
                    target in roots and target not in seen_roots, f"Unexpected or duplicate non-routed name: {name}"
                )
                seen_roots.add(target)
                target_dtype, expected_shape, action = roots[target]
                allowed_dtypes = (
                    ("BF16", "F32")
                    if action == "norm_cast"
                    else ("BF16",)
                    if action == "quantize_fp8"
                    else (target_dtype,)
                )
                require(shape == expected_shape and dtype in allowed_dtypes, f"Incorrect dtype/shape: {name}")
                if action == "norm_cast" and dtype == "BF16":
                    action = "copy"
                if action == "quantize_fp8":
                    require(all(dim % BLOCK == 0 for dim in shape), f"FP8 matrix needs block128 alignment: {name}")
            outputs = [(target, target_dtype, shape)]
            if action == "quantize_fp8":
                outputs.append(
                    (target.removesuffix("weight") + "scale", "F8_E8M0", tuple(dim // BLOCK for dim in shape))
                )
            for out_name, out_dtype, out_shape in outputs:
                require(out_name not in output_names, f"Duplicate output tensor: {out_name}")
                output_names.add(out_name)
                size = math.prod(out_shape) * dtype_size(out_dtype)
                target_header[out_name] = {
                    "dtype": out_dtype,
                    "shape": list(out_shape),
                    "data_offsets": [offset, offset + size],
                }
                offset += size
            jobs.append(
                {
                    "source_name": name,
                    "target_name": target,
                    "dtype": dtype,
                    "shape": shape,
                    "offset": base + start,
                    "size": end - start,
                    "action": action,
                    "outputs": tuple(out[0] for out in outputs),
                }
            )
            source_bytes += end - start
            counts[action] += 1
        target_bytes += offset
        shards.append({"relative": rel, "source": path, "header": target_header, "jobs": jobs})
    require(seen_roots == set(roots), f"Incomplete non-routed inventory: missing={sorted(set(roots) - seen_roots)}")
    require(
        len(expert_fields) == config["num_hidden_layers"] * config["n_routed_experts"] * 2,
        "Incomplete routed expert inventory",
    )
    hidden, intermediate = config["hidden_size"], config["moe_intermediate_size"]
    for (_, _, projection), fields in expert_fields.items():
        n, k = (2 * intermediate, hidden) if projection == "w13" else (hidden, intermediate)
        vq = {
            "packed_indices": ("U8", (n // 32, k // 16, 16, 8)),
            "codebooks": ("F8_E4M3", (n // 32, k // 32, 16, 2)),
            "out_dim_scale": ("U64", (n,)),
        }
        mx = {"weight": ("U8", (n, k // 2)), "scale": ("F8_E8M0", (n, k // 32))}
        require(fields in (vq, mx), f"Invalid routed {projection} field set/dtype/shape")
    require(snapshot(source_files) == before, "Source changed during planning")
    return {
        "source": source,
        "config": config,
        "shards": shards,
        "snapshot": before,
        "source_files": source_files,
        "counts": dict(counts),
        "source_tensors": len(index),
        "target_tensors": len(output_names),
        "source_payload_bytes": source_bytes,
        "target_payload_bytes": target_bytes,
        "routed_payload_bytes": routed_bytes,
        "source_file_bytes": sum(value[0] for value in before.values()),
    }


def quantize_fp8_block128(values: torch.Tensor):
    """FP32 scale/divide -> E4M3FN RNE; return raw E8M0 bytes and errors."""
    require(values.device.type == "cpu" and values.ndim == 2, "Expected a CPU matrix")
    n, k = values.shape
    require(n > 0 and k > 0 and n % BLOCK == k % BLOCK == 0, "Expected block128-aligned matrix")
    values = values.float()
    require(bool(torch.isfinite(values).all()), "Non-finite FP8 input")
    blocks = values.reshape(n // BLOCK, BLOCK, k // BLOCK, BLOCK).permute(0, 2, 1, 3)
    maxima = blocks.abs().amax(dim=(-1, -2))
    safe = torch.where(maxima == 0, torch.ones_like(maxima), maxima)
    exponents = torch.ceil(torch.log2(safe / FP8_MAX)).to(torch.int32)
    exponents = torch.where(maxima == 0, torch.zeros_like(exponents), exponents)
    require(bool(((exponents >= -127) & (exponents <= 127)).all()), "Scale outside finite E8M0 range")
    scales = torch.ldexp(torch.ones_like(maxima), exponents)
    normalized = (blocks / scales[:, :, None, None]).clamp(-FP8_MAX, FP8_MAX)
    quantized = normalized.permute(0, 2, 1, 3).reshape(n, k).to(torch.float8_e4m3fn).contiguous()
    restored = quantized.float().reshape(n // BLOCK, BLOCK, k // BLOCK, BLOCK).permute(0, 2, 1, 3)
    restored = (restored * scales[:, :, None, None]).permute(0, 2, 1, 3).reshape(n, k)
    error = restored.double() - values.double()
    squared_error, squared_reference = error.square().sum().item(), values.double().square().sum().item()
    metrics = {
        "elements": values.numel(),
        "max_abs_error": error.abs().max().item(),
        "squared_error": squared_error,
        "squared_reference": squared_reference,
        "rmse": math.sqrt(squared_error / values.numel()),
        "relative_l2": math.sqrt(squared_error / squared_reference) if squared_reference else 0.0,
    }
    return quantized, (exponents + 127).to(torch.uint8).contiguous(), metrics


def tensor_bytes(value):
    return value.contiguous().view(torch.uint8).numpy().tobytes()


def read_exact(stream, size):
    value = stream.read(size)
    require(len(value) == size, "Source truncated while reading a tensor")
    return value


def merge_metrics(total, addition):
    total["elements"] += addition["elements"]
    total["max_abs_error"] = max(total["max_abs_error"], addition["max_abs_error"])
    total["squared_error"] += addition["squared_error"]
    total["squared_reference"] += addition["squared_reference"]


def write_shard(shard, destination):
    header = shard["header"]
    encoded = json.dumps(header, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    destination.parent.mkdir(parents=True, exist_ok=True)
    records, expected_digests = [], {}
    with shard["source"].open("rb") as source, destination.open("xb") as output:
        output.write(struct.pack("<Q", len(encoded)))
        output.write(encoded)
        for job in shard["jobs"]:
            source.seek(job["offset"])
            src_hash, dst_hash = hashlib.sha256(), hashlib.sha256()
            record = {"source_name": job["source_name"], "target_name": job["target_name"], "action": job["action"]}
            if job["action"] == "quantize_fp8":
                rows, columns = job["shape"]
                scales = bytearray()
                metrics = {"elements": 0, "max_abs_error": 0.0, "squared_error": 0.0, "squared_reference": 0.0}
                for _ in range(0, rows, BLOCK):
                    raw = read_exact(source, BLOCK * columns * 2)
                    src_hash.update(raw)
                    values = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(BLOCK, columns)
                    quantized, scale, errors = quantize_fp8_block128(values)
                    payload = tensor_bytes(quantized)
                    output.write(payload)
                    dst_hash.update(payload)
                    scales.extend(tensor_bytes(scale))
                    merge_metrics(metrics, errors)
                output.write(scales)
                scale_hash = hashlib.sha256(scales).hexdigest()
                expected_digests[job["outputs"][1]] = scale_hash
                record["scale_sha256"] = scale_hash
                metrics["rmse"] = math.sqrt(metrics["squared_error"] / metrics["elements"])
                metrics["relative_l2"] = (
                    math.sqrt(metrics["squared_error"] / metrics["squared_reference"])
                    if metrics["squared_reference"]
                    else 0.0
                )
                record["error"] = metrics
            elif job["action"] == "norm_cast":
                raw = read_exact(source, job["size"])
                src_hash.update(raw)
                values = torch.frombuffer(bytearray(raw), dtype=torch.float32)
                converted = values.to(torch.bfloat16)
                require(
                    bool(torch.isfinite(values).all()) and torch.equal(converted.float(), values),
                    f"Norm is not exactly representable as BF16: {job['source_name']}",
                )
                payload = tensor_bytes(converted)
                output.write(payload)
                dst_hash.update(payload)
                record["exact_bf16_roundtrip"] = True
            else:
                remaining = job["size"]
                while remaining:
                    raw = read_exact(source, min(remaining, COPY_CHUNK))
                    src_hash.update(raw)
                    output.write(raw)
                    dst_hash.update(raw)
                    remaining -= len(raw)
            record["source_sha256"] = src_hash.hexdigest()
            record["output_sha256"] = dst_hash.hexdigest()
            expected_digests[job["target_name"]] = record["output_sha256"]
            records.append(record)
        output.flush()
        os.fsync(output.fileno())
    actual, base = read_header(destination)
    require(actual == header, f"Written header mismatch: {destination}")
    with destination.open("rb") as stream:
        for name, entry in actual.items():
            if name == "__metadata__":
                continue
            begin, end = entry["data_offsets"]
            stream.seek(base + begin)
            digest, remaining = hashlib.sha256(), end - begin
            while remaining:
                raw = read_exact(stream, min(remaining, COPY_CHUNK))
                digest.update(raw)
                remaining -= len(raw)
            require(digest.hexdigest() == expected_digests[name], f"Output payload verification failed: {name}")
    return records


def size_comparison(source, target):
    return {
        "source_bytes": source,
        "target_bytes": target,
        "saved_bytes": source - target,
        "saved_percent": (source - target) / source * 100 if source else 0.0,
    }


def write_report(path, report, other_file_bytes):
    """Include the report itself in the final logical on-disk file byte count."""
    source_size = report["storage"]["files"]["source_bytes"]
    report["storage"]["files"] = size_comparison(source_size, other_file_bytes)
    initial = (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    # A small trailing JSON-whitespace reserve avoids oscillating float string
    # lengths when the report's own size changes its saved_percent value.
    reserved_size = len(initial) + 256
    report["storage"]["files"] = size_comparison(source_size, other_file_bytes + reserved_size)
    encoded = (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    require(len(encoded) <= reserved_size, "Report metadata exceeded its whitespace reserve")
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.write(b" " * (reserved_size - len(encoded)))
        stream.flush()
        os.fsync(stream.fileno())


def repack(source: Path, output: Path, *, dry_run=False, quantize_fp8=False, threads=4) -> dict:
    source, output = source.resolve(), output.resolve()
    require(not output.exists(), f"Output already exists: {output}")
    require(
        not source.is_relative_to(output) and not output.is_relative_to(source),
        "Source and output must be separate directory trees",
    )
    require(type(threads) is int and threads > 0, "threads must be positive")
    plan = plan_model(source)
    report = {
        "schema_version": 1,
        "status": "planned",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "output": str(output),
        "contract": "export-20261103",
        "source_tensors": plan["source_tensors"],
        "target_tensors": plan["target_tensors"],
        "actions": plan["counts"],
        "source_config_sha256": file_sha256(source / "config.json"),
        "source_index_sha256": file_sha256(source / "model.safetensors.index.json"),
        "routed_payload_bytes": plan["routed_payload_bytes"],
        "quantization": {
            "block_size": [BLOCK, BLOCK],
            "weight_dtype": "F8_E4M3",
            "scale_dtype": "F8_E8M0",
            "scale_formula": "2**ceil(log2(amax/448)); zero block uses 1",
            "rounding": "FP32 division then E4M3FN round-to-nearest-even; no BF16 intermediate",
            "norm_conversion": "F32 to BF16 only if exactly representable",
        },
        "storage": {
            "payload": size_comparison(plan["source_payload_bytes"], plan["target_payload_bytes"]),
            "files": {"source_bytes": plan["source_file_bytes"]},
        },
        "model_executed": False,
        "limitations": [
            "FP8 conversion is lossy; inference accuracy is not certified.",
            "No NPU, HCCL or graph-capture validation.",
        ],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if dry_run:
        return report
    require(quantize_fp8, "Actual conversion requires --quantize-fp8 to select the lossy conversion")
    spec = Path(__file__).resolve().parents[1] / "docs/vq2a8" / SPEC_NAME
    require(spec.is_file(), f"Repository export specification is missing: {spec}")
    output.parent.mkdir(parents=True, exist_ok=True)
    require(
        shutil.disk_usage(output.parent).free >= plan["target_payload_bytes"] + 256 * 1024 * 1024,
        "Insufficient free space for independent checkpoint output",
    )
    staging = output.parent / f".{output.name}.repacking-{os.urandom(8).hex()}"
    staging.mkdir()
    print(f"STAGING_DIR={staging}", flush=True)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        records, weight_map = [], {}
        for number, shard in enumerate(plan["shards"], 1):
            records.extend(write_shard(shard, staging / shard["relative"]))
            for name in shard["header"]:
                if name != "__metadata__":
                    weight_map[name] = shard["relative"]
            print(f"SHARD_VERIFIED={number}/{len(plan['shards'])} {shard['relative']}", flush=True)
        for name in SIDECARS:
            if (source / name).is_file():
                copy_verified(source / name, staging / name)
        copy_verified(spec, staging / SPEC_NAME)
        write_json(
            staging / "model.safetensors.index.json",
            {"metadata": {"total_size": plan["target_payload_bytes"]}, "weight_map": dict(sorted(weight_map.items()))},
        )
        require(snapshot(plan["source_files"]) == plan["snapshot"], "Source files changed during conversion")
        require(len(weight_map) == plan["target_tensors"], "Output inventory count mismatch")
        report["per_tensor"] = records
        report["verification"] = {
            "all_output_tensors_reread_sha256": True,
            "source_snapshot_unchanged": True,
            "routed_payload_byte_identical": all(
                r["source_sha256"] == r["output_sha256"] for r in records if r["action"] == "routed_copy"
            ),
        }
        report["output_index_sha256"] = file_sha256(staging / "model.safetensors.index.json")
        report["export_spec_sha256"] = file_sha256(staging / SPEC_NAME)
        report["torch_version"] = str(torch.__version__)
        report["threads"] = threads
        report["status"] = "finished"
        other_size = sum(path.stat().st_size for path in staging.rglob("*") if path.is_file())
        write_report(staging / REPORT_NAME, report, other_size)
        require(not output.exists(), f"Output appeared during conversion: {output}")
        staging.rename(output)
    except BaseException:
        print(f"Conversion failed; partial output retained at: {staging}", file=sys.stderr, flush=True)
        raise
    finally:
        torch.set_num_threads(old_threads)
    print(f"OUTPUT_DIR={output}", flush=True)
    print(f"REPACK_REPORT={output / REPORT_NAME}", flush=True)
    print(f"STORAGE={json.dumps(report['storage'])}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New independent directory; must not exist")
    parser.add_argument("--dry-run", action="store_true", help="Validate headers and print the conversion/storage plan")
    parser.add_argument(
        "--quantize-fp8", action="store_true", help="Explicitly enable the lossy block128 FP8 conversion"
    )
    parser.add_argument("--threads", type=int, default=4, help="CPU PyTorch threads (default: 4)")
    args = parser.parse_args()
    repack(args.source, args.output, dry_run=args.dry_run, quantize_fp8=args.quantize_fp8, threads=args.threads)


if __name__ == "__main__":
    main()
