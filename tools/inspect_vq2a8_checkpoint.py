#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Collect read-only checkpoint diagnostics; Python 3.10+, standard library only.

No model, tensor framework, pickle, or device code is loaded. The default run
reads JSON and safetensors headers, not weight payloads. --sample-values also
checks a bounded number of source-v3 metadata tensors per field. Reports are
written outside the model directory and do not certify model correctness.
--contract export-20261103 checks the three-field VQ2 export contract separately
from current loader blockers and also writes compact.json.
"""

from __future__ import annotations

# Direct script execution must not let tools/bisect shadow stdlib bisect.
# ruff: noqa: E402
import os
import sys

if __name__ == "__main__":
    sys.path[:] = [os.path.dirname(os.path.dirname(os.path.abspath(__file__)))] + [
        path for path in sys.path if os.path.abspath(path) != os.path.dirname(os.path.abspath(__file__))
    ]

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import math
import re
import struct
import tempfile
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from tools.vq2a8_source_contract import source_contract

FORMAT = "vq2_mixed_gs32_source"
ROW_ORDER = "natural_gate_then_up"
HEADER_LIMIT = 64 * 1024 * 1024
JSON_LIMIT = 64 * 1024 * 1024
SAMPLE_BYTE_LIMIT = 4 * 1024 * 1024
SAMPLES_PER_FIELD = 8
ISSUE_EXAMPLES = 20
SUMMARY_LAYOUT_EXAMPLES = 12
DSPARK_PACKAGE_FORMAT = "deepseek_v4_dspark_original_v1"
DSPARK_DEFAULT_STAGES = 3
DSPARK_BLOCK_SIZE = 5
DSPARK_SHARED_WEIGHTS = ("embed.weight", "head.weight", "hc_head_fn", "hc_head_base", "hc_head_scale")
VQ_FIELDS = frozenset(("packed_indices", "codebooks", "out_dim_scale"))
EXPORT_CONTRACT = "export-20261103"
EXPORT_GEOMETRY = (
    ("hidden_size", 4096),
    ("moe_intermediate_size", 2048),
    ("num_hidden_layers", 43),
    ("n_routed_experts", 256),
)
COMPACT_ISSUE_EXAMPLES = 2
MX_FIELDS = frozenset(("weight", "scale"))
EXPERT_PATTERN = r"layers\.(0|[1-9][0-9]*)\.ffn\.experts\.(0|[1-9][0-9]*)\.(w13|w2)\.([a-z_]+)"
DTYPE_BYTES = (
    ("BOOL", 1),
    ("U8", 1),
    ("I8", 1),
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
METADATA_FILES = (
    "config.json",
    "quant.json",
    "generation_config.json",
    "tokenizer_config.json",
    "model.safetensors.index.json",
    "vq2_mixed_gs32/format.json",
    "vq2_mixed_gs32/experts.json",
    "conversion_report.json",
    "dspark_repack_report.json",
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
    require(path.stat().st_size <= JSON_LIMIT, f"JSON exceeds {JSON_LIMIT} bytes: {path}")
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream, object_pairs_hook=unique_object)
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def read_header(path):
    """Validate physical extents without reading payloads; retain unknown dtypes."""
    size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(8)
        require(len(prefix) == 8, "Truncated safetensors prefix")
        length = struct.unpack("<Q", prefix)[0]
        require(2 <= length <= HEADER_LIMIT and 8 + length <= size, "Invalid safetensors header length")
        raw = stream.read(length)
    header = json.loads(raw, object_pairs_hook=unique_object)
    require(isinstance(header, dict), "Safetensors header must be an object")
    metadata = header.pop("__metadata__", {})
    require(isinstance(metadata, dict), "Invalid safetensors metadata")
    require(all(isinstance(k, str) and isinstance(v, str) for k, v in metadata.items()), "Metadata must be strings")
    spans, unknown = [], set()
    widths = dict(DTYPE_BYTES)
    for name, entry in header.items():
        require(isinstance(entry, dict), f"Invalid tensor entry: {name}")
        shape, offsets, dtype = entry.get("shape"), entry.get("data_offsets"), entry.get("dtype")
        require(isinstance(dtype, str), f"Invalid dtype: {name}")
        require(isinstance(shape, list) and all(type(n) is int and n >= 0 for n in shape), f"Invalid shape: {name}")
        require(isinstance(offsets, list) and len(offsets) == 2, f"Invalid offsets: {name}")
        require(all(type(n) is int and n >= 0 for n in offsets), f"Invalid offsets: {name}")
        begin, end = offsets
        require(begin <= end <= size - 8 - length, f"Tensor extends outside file: {name}")
        if dtype in widths:
            require(end - begin == math.prod(shape) * widths[dtype], f"Byte size disagrees with shape/dtype: {name}")
        else:
            unknown.add(dtype)
        spans.append((begin, end))
    cursor = 0
    for begin, end in sorted(spans):
        require(begin == cursor, "Overlapping or non-contiguous tensor offsets")
        cursor = end
    require(8 + length + cursor == size, "Unaccounted trailing bytes in safetensors file")
    return header, metadata, 8 + length, hashlib.sha256(raw).hexdigest(), sorted(unknown)


class Findings:
    def __init__(self):
        self.items = {}

    def add(self, severity, code, message):
        item = self.items.setdefault(code, {"severity": severity, "count": 0, "examples": []})
        item["count"] += 1
        if len(item["examples"]) < ISSUE_EXAMPLES:
            item["examples"].append(message)


def is_mtp(name):
    return bool(re.search(r"(^|\.)(mtp|dspark)(\.|_)", name))


def supported_original_root(name):
    """Mirror the original 0731 root whitelist without importing torch/vLLM."""
    match = re.fullmatch(r"layers\.(0|[1-9][0-9]*)\.(.+)", name)
    if match is None:
        return False
    suffix = match[2]
    if suffix in {
        "attn.attn_sink",
        "attn.compressor.ape",
        "attn.indexer.compressor.ape",
        "ffn.gate.bias",
        "ffn.gate.tid2eid",
        "hc_attn_fn",
        "hc_attn_base",
        "hc_attn_scale",
        "hc_ffn_fn",
        "hc_ffn_base",
        "hc_ffn_scale",
    }:
        return True
    module, _, field = suffix.rpartition(".")
    return field in ("weight", "scale") and module in {
        "attn_norm",
        "ffn_norm",
        "attn.wq_a",
        "attn.q_norm",
        "attn.wq_b",
        "attn.wkv",
        "attn.kv_norm",
        "attn.wo_a",
        "attn.wo_b",
        "attn.compressor.wgate",
        "attn.compressor.wkv",
        "attn.compressor.norm",
        "attn.indexer.wq_b",
        "attn.indexer.weights_proj",
        "attn.indexer.compressor.wgate",
        "attn.indexer.compressor.wkv",
        "attn.indexer.compressor.norm",
        "ffn.gate",
        "ffn.shared_experts.w1",
        "ffn.shared_experts.w2",
        "ffn.shared_experts.w3",
    }


def projection_contract(projection, hidden, intermediate):
    """Expected individual fields, including fields in incomplete projections."""
    n, k = (2 * intermediate, hidden) if projection == "w13" else (hidden, intermediate)
    return {
        "packed_indices": ("U8", [n // 32, k // 16, 16, 8]),
        "codebooks": ("F8_E4M3", [n // 32, k // 32, 16, 2]),
        "out_dim_scale": ("U64", [n]),
        "weight": ("U8", [n, k // 2]),
        "scale": ("F8_E8M0", [n, k // 32]),
    }


def layout_inventory(records):
    """Collapse numeric layer/expert IDs, but retain every distinct layout."""
    layouts = {}
    for record in records:
        family = re.sub(r"\.[0-9]+(?=\.|$)", ".*", record["name"])
        key = family, record["dtype"], tuple(record["shape"])
        item = layouts.setdefault(
            key, {"family": family, "dtype": key[1], "shape": list(key[2]), "count": 0, "example": record["name"]}
        )
        item["count"] += 1
    return [layouts[key] for key in sorted(layouts)]


def check_sample(path, base, name, entry, field):
    start, end = entry["data_offsets"]
    require(end - start <= SAMPLE_BYTE_LIMIT, "Tensor exceeds sample byte limit; not read")
    with path.open("rb") as stream:
        stream.seek(base + start)
        data = stream.read(end - start)
    require(len(data) == end - start, "Tensor changed or was truncated while reading")
    if field == "out_dim_scale":
        require(all(0 < x[0] < 0x7F800000 for x in struct.iter_unpack("<Q", data)), "Invalid FP32 scale bits in U64")
    elif field == "codebooks":
        require(b"\x7f" not in data and b"\xff" not in data, "E4M3 codebooks contain NaN")
    elif field == "scale":
        require(b"\xff" not in data, "E8M0 scale contains NaN")
    return {"name": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "status": "passed"}


def check_v3(config, active, shards, findings, sample_count, export_contract=False, runtime_findings=None):
    expected_quant = {"quant_method": "vq2a8", "format": FORMAT, "format_version": 3}
    quant = config.get("quantization_config")
    folded = False
    valid_quant = isinstance(quant, dict) and all(
        type(quant.get(key)) is type(value) and quant[key] == value for key, value in expected_quant.items()
    )
    if not valid_quant:
        findings.add(
            "error",
            "V3_CONFIG",
            f"config.json quantization_config={config.get('quantization_config')!r}; expected {expected_quant}",
        )
    else:
        try:
            folded = source_contract()["validate_source_config"](quant)
        except ValueError as exc:
            (runtime_findings if export_contract else findings).add("error", "V3_CONFIG", str(exc))
    dimensions = [
        config.get(k) for k in ("hidden_size", "moe_intermediate_size", "num_hidden_layers", "n_routed_experts")
    ]
    if not all(type(n) is int and n > 0 for n in dimensions):
        findings.add("error", "V3_GEOMETRY", f"Invalid hidden/intermediate/layers/experts: {dimensions}")
        return {}, [], {}
    hidden, intermediate, layers, experts = dimensions
    hash_layers = config.get("num_hash_layers", 0)
    if type(hash_layers) is not int or not 0 <= hash_layers <= layers:
        findings.add("error", "V3_GEOMETRY", f"Invalid num_hash_layers: {hash_layers}")
        hash_layers = 0
    if hidden % 64 or intermediate % 64:
        findings.add("error", "V3_GEOMETRY", "Current GS32 projections require K64/N32 alignment")
    groups = defaultdict(dict)
    for name, record in active.items():
        match = re.fullmatch(EXPERT_PATTERN, name)
        if match:
            layer, expert, projection, field = int(match[1]), int(match[2]), match[3], match[4]
            groups[layer, expert, projection][field] = record
            if layer >= layers or expert >= experts:
                findings.add("error", "EXPERT_OUT_OF_RANGE", name)
        elif is_mtp(name):
            if (
                re.fullmatch(
                    r"mtp\.(0|[1-9][0-9]*)\.[A-Za-z_][A-Za-z_0-9]*(?:\.(?:[A-Za-z_][A-Za-z_0-9]*|0|[1-9][0-9]*))*",
                    name,
                )
                is None
            ):
                (runtime_findings if export_contract else findings).add(
                    "error", "MTP_NAME", f"{name}: malformed or unsupported original MTP tensor name"
                )
        elif ".experts." in name or ".experts_" in name:
            findings.add("error", "EXPERT_NAME", f"Not a source-v3 per-expert name: {name}")
    inventory = {}
    for layer in range(layers):
        ids = sorted({expert for index, expert, _ in groups if index == layer})
        inventory[str(layer)] = {"expert_ids": ids, "projections": dict(Counter(p for i, _, p in groups if i == layer))}
        if ids != list(range(experts)):
            findings.add(
                "error",
                "EXPERT_INVENTORY",
                f"Layer {layer}: {len(ids)} expert IDs, expected {experts}; hash={layer < hash_layers}",
            )
        for expert in ids:
            for projection in ("w13", "w2"):
                if (layer, expert, projection) not in groups:
                    findings.add("error", "MISSING_PROJECTION", f"layers.{layer}.ffn.experts.{expert}.{projection}")
    candidates = defaultdict(list)
    projection_counts = Counter()
    projection_formats = {projection: Counter() for projection in ("w13", "w2")}
    formats = {}
    field_checks = Counter()
    for (layer, expert, projection), fields in sorted(groups.items()):
        prefix = f"layers.{layer}.ffn.experts.{expert}.{projection}"
        field_set = set(fields)
        if folded and field_set != VQ_FIELDS:
            (runtime_findings if export_contract else findings).add(
                "error", "FOLDED_EXPERT_FIELDS", f"{prefix}: folded profile requires three VQ fields without perm"
            )
        family = (
            "vq2_natural_columns"
            if field_set == VQ_FIELDS
            else "mxfp4_complete"
            if field_set == MX_FIELDS
            else "incomplete_or_mixed"
        )
        projection_counts[family] += 1
        quant_type = "vq2" if field_set == VQ_FIELDS else "mxfp4" if field_set == MX_FIELDS else "invalid"
        formats[layer, expert, projection] = quant_type
        projection_formats[projection][quant_type] += 1
        if "perm" in fields:
            findings.add("error", "PERM_UNSUPPORTED", f"{prefix}.perm: only natural input-column order is supported")
        if field_set == VQ_FIELDS:
            findings.add("info", "NATURAL_INPUT_COLUMNS", f"{prefix}: requires natural input-column order")
        if field_set not in (VQ_FIELDS, MX_FIELDS):
            findings.add(
                "error",
                "EXPERT_FIELDS",
                f"{prefix}: {sorted(fields)}; expected VQ2 {sorted(VQ_FIELDS)} or MXFP4 {sorted(MX_FIELDS)}",
            )
        contract = projection_contract(projection, hidden, intermediate)
        for field, (dtype, shape) in contract.items():
            if field not in fields:
                continue
            record = fields[field]
            if record["dtype"] != dtype or record["shape"] != shape:
                field_checks["mismatched"] += 1
                findings.add(
                    "error",
                    "EXPERT_DTYPE_SHAPE",
                    f"{prefix}.{field}: got {record['dtype']} {record['shape']}; expected {dtype} {shape}",
                )
            else:
                field_checks["matched"] += 1
                if field in ("out_dim_scale", "codebooks", "scale"):
                    candidates[field].append((f"{prefix}.{field}", record))
    samples = []
    for field, records in sorted(candidates.items()):
        # Spread samples over the inventory rather than only reading the first layer.
        count = min(sample_count, len(records))
        indices = sorted({i * (len(records) - 1) // max(count - 1, 1) for i in range(count)})
        for index in indices:
            name, record = records[index]
            begin, end = record["data_offsets"]
            if end - begin > SAMPLE_BYTE_LIMIT:
                findings.add("warning", "SAMPLE_SKIPPED", f"{name}: exceeds {SAMPLE_BYTE_LIMIT} byte sample limit")
                samples.append({"name": name, "field": field, "status": "skipped", "reason": "sample_byte_limit"})
                continue
            path, base = shards[record["shard"]]
            try:
                samples.append({**check_sample(path, base, name, record, field), "field": field})
            except (OSError, ValueError, struct.error) as exc:
                findings.add("error", "SAMPLED_VALUE", f"{name}: {exc}")
                samples.append({"name": name, "field": field, "status": "failed", "error": str(exc)})
    coverage = {}
    for field in ("out_dim_scale", "codebooks", "scale"):
        counts = Counter(s["status"] for s in samples if s["field"] == field)
        coverage[field] = {
            "eligible": len(candidates[field]),
            "selected": sum(counts.values()),
            "not_selected": len(candidates[field]) - sum(counts.values()),
            **{status: counts[status] for status in ("passed", "failed", "skipped")},
        }
    return (
        inventory,
        samples,
        {
            "projection_counts": dict(projection_counts),
            "field_checks": dict(field_checks),
            "sample_coverage": coverage,
            **(
                {
                    "by_projection": {key: dict(value) for key, value in projection_formats.items()},
                    "projection_pairs": dict(
                        Counter(
                            f"{formats.get((layer, expert, 'w13'), 'missing')}/"
                            f"{formats.get((layer, expert, 'w2'), 'missing')}"
                            for layer, expert in sorted({(layer, expert) for layer, expert, _ in groups})
                        )
                    ),
                    "perm_tensors": sum(name.endswith(".perm") for name in active),
                }
                if export_contract
                else {}
            ),
        },
    )


def inspect_draft(config, active, metadata, weight_map, findings, is_draft_scope=False):
    """Identify draft evidence; never infer DSpark compatibility from MTP counts."""
    mtp = {name: record for name, record in active.items() if is_mtp(name)}
    stage_counts = Counter()
    for name in mtp:
        match = re.match(r"mtp\.(0|[1-9][0-9]*)\.", name)
        if match:
            stage_counts[match[1]] += 1
    package = metadata.get("dspark_repack_report.json", {})
    packaged = package.get("format") == DSPARK_PACKAGE_FORMAT
    declared = packaged or ("dspark_block_size" in config and (bool(mtp) or is_draft_scope))
    evidence = {
        "classification": "dspark_original_package"
        if packaged
        else "dspark_candidate"
        if declared
        else "mtp_candidate",
        "stage_tensor_counts": dict(sorted(stage_counts.items(), key=lambda pair: int(pair[0]))),
        "unrecognized_stage_names": sum(stage_counts.values()) != len(mtp),
        "config": {
            k: v
            for k, v in config.items()
            if k.startswith("dspark_") or k in ("n_mtp_layers", "sample_from_anchor", "expert_dtype")
        },
        "shared_weights_present": [name for name in DSPARK_SHARED_WEIGHTS if name in active],
        "signature_weights_present": sorted(
            name
            for name in mtp
            if any(part in name for part in (".main_proj.", ".main_norm.", ".markov_head.", ".confidence_head."))
        ),
        "package_format": package.get("format"),
        "payload_hashes_verified": False,
    }
    if not mtp and not declared:
        evidence["classification"] = "no_mtp"
        return evidence
    quant = config.get("quantization_config")
    if not is_draft_scope and isinstance(quant, dict) and quant.get("quant_method") == "vq2a8":
        evidence["target_loader_action"] = "skip"
        evidence["note"] = (
            "Original MTP tensors are retained in the target inventory and skipped during target loading. "
            "This does not configure or validate a separate DSpark drafter."
        )
        return evidence
    if not declared:
        evidence["note"] = "MTP names alone do not establish the DSpark main-projection/Markov-head architecture."
        return evidence
    if (
        config.get("model_type") != "deepseek_v4"
        or not isinstance(quant, dict)
        or quant.get("quant_method") != "fp8"
        or config.get("expert_dtype", "fp4") != "fp4"
        or config.get("dspark_block_size") != DSPARK_BLOCK_SIZE
        or config.get("sample_from_anchor", True) is not True
    ):
        findings.add(
            "error", "DSPARK_CONFIG", "Expected DeepSeek-V4 FP8/FP4, dspark_block_size=5, sample_from_anchor=True"
        )
    stages = config.get("n_mtp_layers") or config.get("dspark_num_mtp_layers") or DSPARK_DEFAULT_STAGES
    evidence["expected_stages"] = stages
    # Every stage needs at least one tensor; bound malformed counts by inventory size.
    if type(stages) is not int or not 1 <= stages <= len(active):
        findings.add("error", "DSPARK_STAGES", f"Invalid or impossible stage count: {stages!r}")
    else:
        if set(stage_counts) != {str(i) for i in range(stages)} or evidence["unrecognized_stage_names"]:
            findings.add("error", "DSPARK_STAGES", f"Observed stages {list(stage_counts)}, expected 0..{stages - 1}")
        # Minimal signature from validate_vq2a8_dspark_config, plus shared inputs.
        # This is not a full architecture/attention-geometry or numerical check.
        required = {
            *DSPARK_SHARED_WEIGHTS,
            "mtp.0.main_proj.weight",
            "mtp.0.main_proj.scale",
            "mtp.0.main_norm.weight",
            f"mtp.{stages - 1}.norm.weight",
            f"mtp.{stages - 1}.markov_head.markov_w1.weight",
            f"mtp.{stages - 1}.markov_head.markov_w2.weight",
            *(f"mtp.{stage}.attn.wq_a.weight" for stage in range(stages)),
        }
        indexed_names = {name for name, record in active.items() if weight_map.get(name) == record["shard"]}
        missing = sorted(required - indexed_names)
        evidence["critical_weights_missing"] = missing
        for name in missing:
            findings.add("error", "DSPARK_CRITICAL_WEIGHT", name)
    if not weight_map:
        findings.add("error", "DSPARK_INDEX", "DSpark requires a nonempty safetensors index")
    if packaged:
        if "HEADER_NOT_INDEXED" in findings.items:
            findings.items["HEADER_NOT_INDEXED"]["severity"] = "error"
        tensors = package.get("tensors")
        if not isinstance(tensors, dict):
            findings.add("error", "DSPARK_MANIFEST", "Package report is missing its tensor map")
        else:
            if set(tensors) != set(weight_map) or len(tensors) != package.get("total_tensors"):
                findings.add("error", "DSPARK_MANIFEST", "Package tensor inventory/count differs from index/report")
            for name, declared_record in tensors.items():
                record = active.get(name)
                if (
                    not isinstance(declared_record, dict)
                    or record is None
                    or any(record.get(key) != declared_record.get(key) for key in ("dtype", "shape", "shard"))
                    or weight_map.get(name) != declared_record.get("shard")
                ):
                    findings.add("error", "DSPARK_MANIFEST", f"{name}: package report disagrees with index/header")
        evidence["package_declared_counts"] = {
            key: package.get(key) for key in ("stages", "mtp_tensors", "shared_tensors", "total_tensors")
        }
    return evidence


def inspect_checkpoint(root, label, tensor_stream, sample_count=0, exclude_dspark=False, contract="current-loader"):
    findings = Findings()
    export_contract = contract == EXPORT_CONTRACT
    runtime_findings = Findings()
    runtime_checks = runtime_findings if export_contract else findings
    metadata = {}
    for relative in METADATA_FILES:
        path = root / relative
        if path.exists():
            try:
                metadata[relative] = read_json(path)
            except (OSError, ValueError) as exc:
                findings.add("error", "JSON_READ", f"{relative}: {exc}")
    config = metadata.get("config.json", {})
    if not config:
        findings.add("error", "CONFIG_MISSING", "No readable config.json")
    index = metadata.get("model.safetensors.index.json", {})
    weight_map = index.get("weight_map", {})
    if not isinstance(weight_map, dict) or any(not isinstance(v, str) for v in weight_map.values()):
        findings.add("error", "INVALID_INDEX", "weight_map must map tensor names to shard paths")
        weight_map = {}
    expected = defaultdict(set)
    for name, relative in weight_map.items():
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            findings.add("error", "UNSAFE_INDEX_PATH", f"{name}: {relative}")
            continue
        expected[relative].add(name)
    files = {}

    def walk_error(exc):
        findings.add("error", "DIRECTORY_READ", str(exc))

    for directory, dirs, names in os.walk(root, followlinks=False, onerror=walk_error):
        if Path(directory) == root and exclude_dspark:
            dirs[:] = [d for d in dirs if d != "dspark"]
        for name in sorted(names):
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            files[relative] = path
    # Index-referenced symlink directories may not be traversed by os.walk.
    for relative in expected:
        files.setdefault(relative, root / relative)
    records, shard_locations, shard_reports, file_reports = {}, {}, [], []
    for relative, path in sorted(files.items()):
        try:
            info = path.stat()
            require(path.is_file(), "Expected a regular file")
            file_reports.append(
                {"path": relative, "bytes": info.st_size, "mtime_ns": info.st_mtime_ns, "symlink": path.is_symlink()}
            )
            if path.suffix != ".safetensors":
                if relative in expected:
                    findings.add("error", "INDEX_SHARD_TYPE", f"{relative}: expected a .safetensors shard")
                continue
            header, shard_meta, base, digest, unknown = read_header(path)
            if len(shard_reports) % 10 == 0:
                print(f"INSPECT_SHARD={label}/{relative}", flush=True)
            for dtype in unknown:
                findings.add("warning", "UNKNOWN_DTYPE", f"{relative}: {dtype}; byte size could not be checked")
            shard_locations[relative] = (path, base)
            shard_reports.append(
                {
                    "path": relative,
                    "bytes": info.st_size,
                    "tensor_count": len(header),
                    "metadata": shard_meta,
                    "header_sha256": digest,
                    "indexed": relative in expected,
                }
            )
            for name, entry in sorted(header.items()):
                record = {
                    "checkpoint": label,
                    "name": name,
                    "shard": relative,
                    **entry,
                    "indexed": weight_map.get(name) == relative,
                }
                tensor_stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                records.setdefault(name, []).append(record)
            if relative in expected:
                for name in sorted(expected[relative] - header.keys()):
                    findings.add("error", "INDEX_TENSOR_MISSING", f"{relative}: {name}")
                for name in sorted(header.keys() - expected[relative]):
                    findings.add(
                        "warning",
                        "HEADER_NOT_INDEXED",
                        f"{relative}: {name}; GS32 source-v3 requires an exact inventory",
                    )
            elif weight_map:
                findings.add("warning", "UNINDEXED_SHARD", f"{relative}: not selected by this checkpoint's root index")
            after = path.stat()
            if (info.st_size, info.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                findings.add("error", "FILE_CHANGED", relative)
        except (OSError, ValueError, struct.error) as exc:
            severity = "error" if relative in expected or not weight_map else "warning"
            findings.add(severity, "FILE_READ", f"{relative}: {exc}")
    active = {}
    for name, entries in records.items():
        selected = [r for r in entries if r["shard"] in expected] if weight_map else entries
        if len(selected) > 1:
            findings.add("error", "DUPLICATE_TENSOR", f"{name}: {[r['shard'] for r in selected]}")
        if selected:
            active[name] = selected[0]
    per_expert = sum(bool(re.fullmatch(EXPERT_PATTERN, name)) for name in active)
    bank_tensors = sum(bool(re.search(r"\.experts\.w(?:13|2)_(?:vq|mxfp4)_", name)) for name in records)
    quant = config.get("quantization_config", {})
    quant = quant if isinstance(quant, dict) else {}
    versions = [obj.get("format_version") for obj in [quant, *metadata.values()] if obj.get("format") == FORMAT]
    source_v3 = 3 in versions or bool(per_expert)
    active_banks = any(re.search(r"\.experts\.w(?:13|2)_(?:vq|mxfp4)_", name) for name in active)
    source_v1 = 1 in versions or active_banks or (not source_v3 and bool(bank_tensors))
    classification = (
        "mixed_or_conflicting"
        if source_v1 and source_v3
        else "source_v3_candidate"
        if source_v3
        else "source_v1_candidate"
        if source_v1
        else "other_or_unknown"
    )
    inventory, samples, expert_checks = {}, [], {}
    if source_v3 or export_contract:
        if not weight_map:
            findings.add("error", "INDEX_REQUIRED", "GS32 source-v3 requires a nonempty model.safetensors.index.json")
        depths = sorted({len(Path(p).parts) - 1 for p in expected})
        if len(depths) > 1:
            runtime_checks.add(
                "error", "SHARD_DEPTH", f"GS32 default loader requires uniform shard depth, got {depths}"
            )
        for relative in expected:
            if not (root / relative).resolve().is_relative_to(root):
                runtime_checks.add(
                    "error",
                    "EXTERNAL_V3_SHARD",
                    f"{relative}: current GS32 loader rejects shards resolving outside the model directory",
                )
        for relative in ("quant.json", "vq2_mixed_gs32/format.json"):
            if relative in metadata:
                obj = metadata[relative]
                if any(
                    obj.get(k) != v
                    for k, v in {
                        "quant_method": "vq2a8",
                        "format": FORMAT,
                        "format_version": 3,
                        "row_order": ROW_ORDER,
                    }.items()
                ):
                    findings.add("error", "V3_MARKER", f"{relative}: incompatible format/version/row_order")
        for shard in shard_reports:
            if shard["indexed"]:
                for key, values in (
                    ("format", (FORMAT, "pt")),
                    ("format_version", ("3",)),
                    ("row_order", (ROW_ORDER,)),
                ):
                    if key in shard["metadata"] and shard["metadata"][key] not in values:
                        findings.add("error", "V3_SHARD_METADATA", f"{shard['path']}: {key}={shard['metadata'][key]}")
        if "HEADER_NOT_INDEXED" in findings.items:
            findings.items["HEADER_NOT_INDEXED"]["severity"] = "error"
        inventory, samples, expert_checks = check_v3(
            config, active, shard_locations, findings, sample_count, export_contract, runtime_findings
        )
        if isinstance(quant, dict) and quant.get("output_head_mode") == "dedicated_main_mtp_norm_fused":
            rotation_path = root / "rotation_config.json"
            if rotation_path.exists():
                try:
                    source_contract()["validate_rotation_config"](read_json(rotation_path))
                except (ValueError, OSError) as exc:
                    runtime_checks.add("error", "ROTATION_CONFIG", str(exc))
    if not shard_reports:
        findings.add("error", "NO_READABLE_SHARDS", "No safetensors headers could be inspected")
    if source_v1 and source_v3:
        findings.add(
            "error", "FORMAT_CONFLICT", "v1 bank layout/markers and v3 names/markers coexist; do not blindly repack"
        )
    mtp = [name for name in active if is_mtp(name)]
    draft = inspect_draft(config, active, metadata, weight_map, runtime_checks, label == "dspark")
    if not source_v1 and not source_v3 and draft["classification"] != "no_mtp":
        classification = draft["classification"]
    roots = [r for name, r in active.items() if not is_mtp(name) and ".experts." not in name]
    if source_v3 or export_contract:
        # Check original 0731 names without importing the runtime. Legacy HF
        # suffix validation and actual parameter coverage remain loader checks.
        for record in roots:
            name = record["name"]
            if name.startswith("layers.") and not supported_original_root(name):
                runtime_checks.add(
                    "error",
                    "V3_ROOT_NAME",
                    f"{name}: original root name is outside the current source-v3 loader whitelist",
                )
    rotation_artifacts = []
    for shard in shard_reports:
        if "quarot" not in shard["path"].lower():
            continue
        entries = [r for group in records.values() for r in group if r["shard"] == shard["path"]]
        rotation_artifacts.append(
            {
                "path": shard["path"],
                "indexed": shard["indexed"],
                "tensor_count": len(entries),
                "layouts": layout_inventory(entries),
                "payload_checked": False,
            }
        )
        findings.add(
            "warning",
            "ROTATION_PROVENANCE",
            f"{shard['path']}: header collected; whether rotation was fused/required is unknown. "
            "An unindexed rotation file does not establish natural input-column order or safe removal.",
        )
    recommendations = []
    if classification == "source_v1_candidate":
        recommendations.append(
            "Possible source-v1 to v3 conversion: run tools/convert_vq2_mixed_gs32_v3.py "
            "--source MODEL --dry-run first; its manifest/layout checks must pass. "
            "Only natural-column sources are supported; perm tensors are rejected. "
            "A version marker alone is not sufficient."
        )
    elif classification == "source_v3_candidate":
        recommendations.append(
            "Do not rerun the v1-to-v3 converter on this candidate. Resolve reported index/field/layout problems; "
            "a matching source-v3 checkpoint can be loaded directly without another repack."
        )
    elif classification in ("dspark_original_package", "dspark_candidate"):
        recommendations.append(
            "Recognized DSpark draft evidence; this scope does not use the GS32 v1-to-v3 converter. "
            "Header/signature checks do not establish complete draft coverage or target compatibility."
        )
        if classification == "dspark_original_package":
            recommendations.append(
                "Stored package hashes were NOT rechecked. For full package integrity use "
                "tools/repack_vq2a8_dspark.py --output DRAFT_DIR --verify-only (reads all draft payloads)."
            )
    elif classification == "mtp_candidate":
        recommendations.append(
            "MTP tensors found, but DSpark package/config evidence is absent. "
            "Do not assume that any MTP checkpoint is the compatible DSpark drafter."
        )
    else:
        recommendations.append(
            "Need export provenance and layout evidence before choosing a repack. "
            "Do not fix this by only renaming tensors or changing format_version."
        )
    if not mtp:
        recommendations.append(
            "No indexed MTP/DSpark tensors found in this checkpoint scope. "
            "This does not prevent ordinary target-model decoding; "
            "inspect the separate dspark scope for a packaged draft."
        )
    if "NATURAL_INPUT_COLUMNS" in findings.items:
        recommendations.append(
            "Three-field VQ2 projections require natural input-column order. "
            "Tensor headers do not prove the exported column order; QuaRot rotation is separate evidence."
        )
    if mtp and (source_v3 or export_contract):
        recommendations.append(
            "Original mtp.* tensors stay in the checkpoint and are skipped by the target loader. "
            "Separate DSpark loading still requires its own configuration and validation."
        )
    if export_contract:
        recommendations = [
            "Return compact.json for loader adaptation. Export header checks and current runtime blockers "
            "are separate; this inspection does not load or execute the model.",
            "Input-column and Gate/Up order are export requirements, not facts proved by tensor shapes. "
            "Do not add identity perm tensors or repack based only on this report.",
        ]
    return (
        {
            "label": label,
            "root": str(root),
            "classification": classification,
            "config": config,
            "index_entries": len(weight_map),
            "readable_shards": len(shard_reports),
            "active_tensors": len(active),
            "per_expert_tensors": per_expert,
            "bank_tensors": bank_tensors,
            "active_dtype_counts": dict(Counter(r["dtype"] for r in active.values())),
            "name_family_counts": dict(Counter(re.sub(r"\.[0-9]+(?=\.|$)", ".*", n) for n in active)),
            "mtp_tensor_count": len(mtp),
            "mtp_examples": mtp[:ISSUE_EXAMPLES],
            "draft": draft,
            "expert_checks": expert_checks,
            "expert_layouts": layout_inventory(r for n, r in active.items() if re.fullmatch(EXPERT_PATTERN, n)),
            "root_layouts": layout_inventory(roots),
            "mtp_layouts": layout_inventory(active[name] for name in mtp),
            "rotation_artifacts": rotation_artifacts,
            "layers": inventory,
            "shards": shard_reports,
            "samples": samples,
            "findings": findings.items,
            "recommendations": recommendations,
            **(
                {
                    "export_contract": {
                        "name": EXPORT_CONTRACT,
                        "header_status": "issues_found"
                        if any(item["severity"] == "error" for item in findings.items.values())
                        else "checked",
                        "geometry": {
                            key: {"expected": value, "observed": config.get(key), "matches": config.get(key) == value}
                            for key, value in EXPORT_GEOMETRY
                        },
                        "matches_spec_geometry": all(config.get(key) == value for key, value in EXPORT_GEOMETRY),
                        "scope": "Routed-expert fields/layouts use config dimensions; spec geometry is separate. "
                        "Index integrity is checked; non-routed names are inventoried only.",
                        "input_column_order_verified": False,
                        "gate_up_order_verified": False,
                        "model_executed": False,
                    },
                    "runtime_compatibility": {
                        "status": "blocked" if runtime_findings.items else "not_verified",
                        "scope": "Known current loader blockers only; not a complete model compatibility test.",
                        "findings": runtime_findings.items,
                    },
                    "export_metadata": {
                        name: metadata[name]
                        for name in ("quant.json", "vq2_mixed_gs32/format.json")
                        if name in metadata
                    },
                }
                if export_contract
                else {}
            ),
        },
        metadata,
        file_reports,
    )


def checkpoint_summary(item):
    lines = [
        f"\n[{item['label']}] {item['classification']}",
        f"shards={item['readable_shards']} active_tensors={item['active_tensors']} mtp={item['mtp_tensor_count']}",
        "quantization_config=" + json.dumps(item["config"].get("quantization_config"), ensure_ascii=False),
    ]
    checks = item["expert_checks"]
    if checks:
        lines.append("expert_projections=" + json.dumps(checks["projection_counts"], sort_keys=True))
        lines.append("expert_field_checks=" + json.dumps(checks["field_checks"], sort_keys=True))
        for field, coverage in checks["sample_coverage"].items():
            lines.append(f"samples.{field}=" + json.dumps(coverage, sort_keys=True))
    for category in ("expert_layouts", "root_layouts"):
        for layout in item[category][:SUMMARY_LAYOUT_EXAMPLES]:
            lines.append(f"{category}: {layout['family']} {layout['dtype']} {layout['shape']} count={layout['count']}")
        if len(item[category]) > SUMMARY_LAYOUT_EXAMPLES:
            lines.append(f"{category}: {len(item[category])} distinct layouts; complete inventory in report.json")
    draft = item["draft"]
    if draft["classification"] != "no_mtp":
        lines.append("mtp_stage_tensors=" + json.dumps(draft["stage_tensor_counts"]))
        lines.append("draft_config=" + json.dumps(draft["config"], ensure_ascii=False))
        lines.append("dspark_signature_weights_present=" + json.dumps(draft["signature_weights_present"]))
        lines.append("draft_shared_weights_present=" + json.dumps(draft["shared_weights_present"]))
    for artifact in item["rotation_artifacts"]:
        lines.append(
            f"rotation_artifact={artifact['path']} indexed={artifact['indexed']} tensors={artifact['tensor_count']}"
        )
        for layout in artifact["layouts"][:SUMMARY_LAYOUT_EXAMPLES]:
            lines.append(f"  {layout['family']} {layout['dtype']} {layout['shape']} count={layout['count']}")
    for code, finding in item["findings"].items():
        lines.append(f"{finding['severity'].upper()} {code} count={finding['count']}")
        lines.extend("  " + example for example in finding["examples"][:3])
    if "export_contract" in item:
        lines.append("export_contract=" + json.dumps(item["export_contract"], ensure_ascii=False))
        for code, finding in item["runtime_compatibility"]["findings"].items():
            lines.append(f"RUNTIME_BLOCKER {code} count={finding['count']}")
            lines.extend("  " + example for example in finding["examples"][:COMPACT_ISSUE_EXAMPLES])
    lines.extend("NEXT: " + suggestion for suggestion in item["recommendations"])
    return lines


def compact_findings(findings):
    return {code: {**item, "examples": item["examples"][:COMPACT_ISSUE_EXAMPLES]} for code, item in findings.items()}


def compact_report(report):
    """Preserve full config and every layout family, without per-expert/tensor lists."""
    columns = ("family", "dtype", "shape", "count", "example")
    checkpoints = []
    for item in report["checkpoints"]:
        checkpoint = {
            key: item[key]
            for key in (
                "label",
                "root",
                "classification",
                "config",
                "index_entries",
                "readable_shards",
                "active_tensors",
                "per_expert_tensors",
                "mtp_tensor_count",
                "expert_checks",
                "export_contract",
                "export_metadata",
                "recommendations",
            )
            if key in item
        }
        checkpoint["findings"] = compact_findings(item["findings"])
        if "runtime_compatibility" in item:
            checkpoint["runtime_compatibility"] = {
                **item["runtime_compatibility"],
                "findings": compact_findings(item["runtime_compatibility"]["findings"]),
            }
        for category in ("expert_layouts", "root_layouts", "mtp_layouts"):
            checkpoint[category] = [[entry[key] for key in columns] for entry in item[category]]
        checkpoint["mtp_stages"] = item["draft"]["stage_tensor_counts"]
        checkpoint["rotation_artifacts"] = item["rotation_artifacts"]
        checkpoints.append(checkpoint)
    return {
        "schema_version": 1,
        "report_kind": "export_adaptation",
        "contract": EXPORT_CONTRACT,
        "created_utc": report["created_utc"],
        "sample_limit_per_field": report["sample_limit_per_field"],
        "layout_columns": list(columns),
        "checkpoints": checkpoints,
        "limitations": report["limitations"],
    }


def inspect_model(model, output_dir, sample_count=0, contract="current-loader"):
    model, output_dir = Path(model).resolve(), Path(output_dir).resolve()
    require(model.is_dir(), f"Model directory does not exist: {model}")
    scopes = [("target", model)]
    if (model / "dspark").is_dir():
        scopes.append(("dspark", (model / "dspark").resolve()))
    require(
        all(not output_dir.is_relative_to(root) for _, root in scopes),
        "Report output must be outside the model and draft directories",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="checkpoint-", dir=output_dir))
    reports, metadata, files = [], {}, {}
    with gzip.open(run_dir / "tensors.jsonl.gz", "wt", encoding="utf-8") as stream:
        for label, root in scopes:
            print(f"INSPECT_SCOPE={label} MODEL={root}", flush=True)
            report, scope_metadata, scope_files = inspect_checkpoint(
                root,
                label,
                stream,
                sample_count,
                label == "target",
                contract if label == "target" else "current-loader",
            )
            reports.append(report)
            metadata[label], files[label] = scope_metadata, scope_files
    packages = {}
    for package in ("torch", "torch-npu", "vllm", "vllm-ascend", "safetensors"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    report = {
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "installed_packages": packages,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sample_limit_per_field": sample_count,
        "checkpoints": reports,
        "limitations": [
            "No model execution, HCCL check, full payload hash or numerical equivalence check.",
            "Root tensor name normalization/completeness and attention geometry "
            "still require the matching runtime loader.",
            "Matching shapes cannot certify tile order, Gate/Up order, input-column order or export provenance.",
            "Sample checks cover only the listed source-v3 tensors; unsampled values remain unverified.",
            "QuaRot payloads and recorded DSpark package hashes are not verified. "
            "DSpark signatures are partial checks, not complete coverage or target/draft equivalence.",
        ],
    }
    for filename, data in (("report.json", report), ("metadata.json", metadata), ("files.json", files)):
        (run_dir / filename).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if contract == EXPORT_CONTRACT:
        (run_dir / "compact.json").write_text(
            json.dumps(compact_report(report), ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8"
        )
    lines = ["REPORT_SCHEMA_VERSION=2", f"MODEL={model}", f"SAMPLED_TENSORS_PER_FIELD={sample_count}"]
    for item in reports:
        lines.extend(checkpoint_summary(item))
    lines.extend(("\nLIMITATIONS:", *report["limitations"]))
    (run_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    bundle = run_dir / "checkpoint-report.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename in ("summary.txt", "report.json", "metadata.json", "files.json", "tensors.jsonl.gz"):
            archive.write(run_dir / filename, filename)
        if contract == EXPORT_CONTRACT:
            archive.write(run_dir / "compact.json", "compact.json")
    errors = sum(f["count"] for item in reports for f in item["findings"].values() if f["severity"] == "error")
    print("\n".join(lines), flush=True)
    print(f"CHECKPOINT_INSPECT_STATUS={'issues_found' if errors else 'collected'}")
    print(f"SUMMARY={run_dir / 'summary.txt'}")
    if contract == EXPORT_CONTRACT:
        print(f"COMPACT_REPORT={run_dir / 'compact.json'}")
    print(f"REPORT_BUNDLE={bundle}")
    return run_dir, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--contract",
        choices=("current-loader", EXPORT_CONTRACT),
        default="current-loader",
        help="Separate the new export contract from historical runtime compatibility checks",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark-results/checkpoint-inspect"))
    parser.add_argument(
        "--sample-values", action="store_true", help="Check up to 8 tensors per metadata field, spread over experts"
    )
    args = parser.parse_args(argv)
    try:
        inspect_model(
            args.model,
            args.output_dir,
            sample_count=SAMPLES_PER_FIELD if args.sample_values else 0,
            contract=args.contract,
        )
    except (OSError, ValueError) as exc:
        print(f"CHECKPOINT_INSPECT_STATUS=collection_failed ERROR={exc}", file=sys.stderr)
        return 1
    # Findings are diagnostic results, not a collection failure or certification.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
