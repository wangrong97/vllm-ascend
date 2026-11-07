#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read actual non-routed/rotation evidence; NumPy CPU only, no model loading."""

from __future__ import annotations

# Avoid tools/bisect shadowing the standard library during direct execution.
# ruff: noqa: E402
import os
import sys

if __name__ == "__main__":
    sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse
import hashlib
import json
import re
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from tools.inspect_vq2a8_checkpoint import DTYPE_BYTES, is_mtp, read_header, read_json, require

FULL_TENSOR_LIMIT = 2 * 1024**2
ROTATION_LIMIT = 128 * 1024**2
SIDECAR_LIMIT = 2 * 1024**2
SAMPLE_ROWS = 4
ROTATION_PROBE_ROWS = 16
DECLARATION_LIMIT = 80
TRANSFORM_KEY = re.compile(
    r"rotat|quarot|smooth|fuse|norm|head|layout|perm|mtp|dspark|export|revision|source_model|source_commit|scheme|bproj",
    re.I,
)
FINAL_NORM = re.compile(r"(?:model\.)?norm\.weight|mtp\.[0-9]+\.norm\.weight")
QUESTIONS = (
    "Residual basis: write the actual row-vector formula x_new = x @ R or x @ R.T, and where it is applied.",
    "Final logits: give the exact HC merge -> rotation/inverse rotation -> RMSNorm -> head formula and order.",
    "Norm gamma: list each fused Norm and destination weight; state whether the saved Norm was replaced by ones.",
    "Attention/shared FFN: list rotations/smoothing on inputs, Q/K/V, outputs and SwiGLU intermediates; "
    "which are folded into weights and which must run at inference?",
    "MTP: describe e_proj/h_proj or main_proj, token/hidden alignment, stage ordering, and use of dedicated "
    "embedding/head/Norm/HC. MTP names alone do not identify a DSpark-compatible architecture.",
    "Provenance: provide the exact original model revision and exporter revision or a short transformation manifest. "
    "Exporter source is optional; tensor shapes cannot recover these semantics.",
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def row_ids(count, limit=SAMPLE_ROWS):
    return sorted(set(np.linspace(0, count - 1, min(count, limit), dtype=np.int64).tolist())) if count else []


def decode(raw, dtype, shape):
    if dtype == "BF16":
        values = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
    else:
        formats = {"F32": "<f4", "F16": "<f2", "F64": "<f8", "I64": "<i8", "I32": "<i4"}
        if dtype not in formats:
            return None
        values = np.frombuffer(raw, dtype=formats[dtype])
    return values.reshape(shape)


def value_stats(values):
    values = values.astype(np.float64, copy=False).reshape(-1)
    finite = values[np.isfinite(values)]
    return {
        "elements": values.size,
        "non_finite": int(values.size - finite.size),
        "all_one": bool(values.size and finite.size == values.size and np.all(finite == 1)),
        "min": float(finite.min()) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
        "mean": float(finite.mean()) if finite.size else None,
        "first_values": [float(x) if np.isfinite(x) else None for x in values[:8]],
    }


class Checkpoint:
    """Reuse the header validator; payload reads are bounded and explicit."""

    def __init__(self, root):
        self.root = Path(root).resolve(strict=True)
        self.config = read_json(self.root / "config.json")
        index_path = self.root / "model.safetensors.index.json"
        self.index_sha256 = digest(index_path.read_bytes())
        self.weight_map = read_json(index_path)["weight_map"]
        require(isinstance(self.weight_map, dict), "weight_map must be an object")
        self.headers = {}
        self.bytes_read = 0
        self.shard_metadata = {}

    def header(self, relative):
        require(isinstance(relative, str) and not Path(relative).is_absolute(), "Expected relative shard path")
        require(".." not in Path(relative).parts, "Parent traversal in shard path")
        if relative not in self.headers:
            path = (self.root / relative).resolve(strict=True)
            # Read-only indexed symlink shards are supported, as in the existing inspector.
            before = path.stat()
            header, metadata, base, header_hash, _ = read_header(path)
            stamp = (before.st_size, before.st_mtime_ns)
            after = path.stat()
            require(stamp == (after.st_size, after.st_mtime_ns), f"Shard changed: {relative}")
            self.headers[relative] = (path, header, base, stamp)
            self.shard_metadata[relative] = {"header_sha256": header_hash, "metadata": metadata}
        return self.headers[relative]

    def locate(self, name, shard=None):
        relative = self.weight_map[name] if shard is None else shard
        path, header, base, stamp = self.header(relative)
        require(name in header, f"Indexed tensor missing from shard: {name}")
        return relative, path, header[name], base, stamp

    def read(self, name, *, shard=None, full_limit=FULL_TENSOR_LIMIT, rows=None):
        relative, path, entry, base, stamp = self.locate(name, shard)
        shape, dtype = entry["shape"], entry["dtype"]
        begin, end = entry["data_offsets"]
        selected = None
        if rows is not None or end - begin > full_limit:
            require(len(shape) == 2, f"Cannot sample rows of oversized non-matrix {name}")
            selected = row_ids(shape[0]) if rows is None else rows
            require(all(type(row) is int and 0 <= row < shape[0] for row in selected), "Invalid sample row")
            require(dtype in dict(DTYPE_BYTES), f"Unsupported sample dtype: {dtype}")
            width = shape[1] * dict(DTYPE_BYTES)[dtype]
            require(len(selected) * width <= full_limit, f"Sample exceeds byte limit: {name}")
            ranges = [(begin + row * width, width) for row in selected]
            read_shape = [len(selected), shape[1]]
        else:
            ranges, read_shape = [(begin, end - begin)], shape
        chunks = []
        with path.open("rb") as stream:
            for offset, length in ranges:
                stream.seek(base + offset)
                chunk = stream.read(length)
                require(len(chunk) == length, f"Truncated tensor payload: {name}")
                chunks.append(chunk)
        after = path.stat()
        require(stamp == (after.st_size, after.st_mtime_ns), f"Shard changed: {relative}")
        raw = b"".join(chunks)
        self.bytes_read += len(raw)
        values = decode(raw, dtype, read_shape)
        record = {
            "name": name,
            "shard": relative,
            "dtype": dtype,
            "shape": shape,
            "scope": "full_tensor" if selected is None else "sampled_rows",
            "rows": selected,
            "bytes_read": len(raw),
            "sha256": digest(raw),
        }
        if values is not None:
            record["values"] = value_stats(values)
        return record, values


def declarations(value, prefix=""):
    result = []
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else key
            if isinstance(child, dict):
                result.extend(declarations(child, name))
            elif TRANSFORM_KEY.search(name):
                encoded = json.dumps(child, ensure_ascii=False)
                result.append({"key": name, "value": child if len(encoded) <= 512 else encoded[:512] + "..."})
    return result


def sidecars(root):
    reports = []
    paths = sorted(
        set(root.glob("*.json")) | set(root.glob("optional/*.json")) | set(root.glob("vq2_mixed_gs32/*.json"))
    )
    for path in paths:
        if path.name in ("config.json", "model.safetensors.index.json", "tokenizer.json"):
            continue
        if path.stat().st_size > SIDECAR_LIMIT:
            reports.append({"file": path.relative_to(root).as_posix(), "status": "skipped_size_limit"})
            continue
        try:
            matches = declarations(read_json(path))
            if matches:
                reports.append(
                    {
                        "file": path.relative_to(root).as_posix(),
                        "sha256": digest(path.read_bytes()),
                        "declarations": matches[:DECLARATION_LIMIT],
                        "truncated": len(matches) > DECLARATION_LIMIT,
                    }
                )
        except (ValueError, OSError) as exc:
            reports.append({"file": path.relative_to(root).as_posix(), "error": str(exc)})
    return reports


def rotation_evidence(checkpoint, name, hidden):
    candidates = []
    shards = set(checkpoint.weight_map.values())
    # Auxiliary transforms are often unindexed; inspect only likely transform files.
    for path in checkpoint.root.rglob("*.safetensors"):
        relative = path.relative_to(checkpoint.root).as_posix()
        if re.search(r"rotat|quarot|smooth", relative, re.I):
            shards.add(relative)
    for relative in sorted(shards):
        _, header, _, _ = checkpoint.header(relative)
        for tensor, entry in header.items():
            if tensor == name or re.search(r"rotat|quarot|smooth", tensor, re.I):
                candidates.append(
                    {
                        "name": tensor,
                        "shard": relative,
                        "dtype": entry["dtype"],
                        "shape": entry["shape"],
                        "indexed": checkpoint.weight_map.get(tensor) == relative,
                    }
                )
    matches = [item for item in candidates if item["name"] == name]
    report = {
        "candidates": candidates,
        "requested_tensor": name,
        "status": "missing" if not matches else "ambiguous" if len(matches) > 1 else "found",
    }
    if len(matches) != 1:
        return report, None
    match = matches[0]
    if match["shape"] != [hidden, hidden] or match["dtype"] not in ("F32", "BF16", "F16", "F64"):
        report["status"] = "unsupported_geometry_or_dtype"
        return report, None
    if hidden * hidden * dict(DTYPE_BYTES)[match["dtype"]] > ROTATION_LIMIT:
        report["status"] = "skipped_size_limit"
        return report, None
    record, rotation = checkpoint.read(name, shard=match["shard"], full_limit=ROTATION_LIMIT)
    report["payload"] = record
    if not np.isfinite(rotation).all():
        report["status"] = "non_finite"
        return report, None
    ids = row_ids(hidden, ROTATION_PROBE_ROWS)
    rows, columns = rotation[ids].astype(np.float64), rotation[:, ids].astype(np.float64)
    report["probes"] = {
        "indices": ids,
        "scope": "Selected row/column Gram matrices, not a full orthogonality test",
        "row_gram_max_abs_error": float(np.max(np.abs(rows @ rows.T - np.eye(len(ids))))),
        "column_gram_max_abs_error": float(np.max(np.abs(columns.T @ columns - np.eye(len(ids))))),
        "sampled_symmetry_max_abs_error": float(np.max(np.abs(rows - columns.T))),
        "sampled_identity_max_abs_error": float(np.max(np.abs(rows - (np.array(ids)[:, None] == np.arange(hidden))))),
    }
    report["status"] = "measured"
    return report, rotation.astype(np.float32, copy=False)


def relative_error(actual, predicted):
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        return {"finite": False, "relative_l2": None, "max_abs": None}
    delta = actual.astype(np.float64) - predicted
    norm = float(np.linalg.norm(actual.astype(np.float64)))
    difference = float(np.linalg.norm(delta))
    relative = difference / norm if norm else 0.0 if difference == 0 else None
    return {
        "finite": True,
        "relative_l2": relative,
        "max_abs": float(np.max(np.abs(delta))) if delta.size else 0.0,
    }


def reference_comparison(target_record, actual, reference, rotation, hidden):
    name = target_record["name"]
    if name not in reference.weight_map:
        return {"status": "name_absent_in_reference"}
    _, _, header, _, _ = reference.locate(name)
    if header["shape"] != target_record["shape"] or header["dtype"] != target_record["dtype"]:
        return {"status": "layout_differs", "reference_dtype": header["dtype"], "reference_shape": header["shape"]}
    source_record, source = reference.read(name, rows=target_record["rows"])
    result = {
        "status": "compared",
        "scope": target_record["scope"],
        "raw_bytes_equal": target_record["sha256"] == source_record["sha256"],
        "reference_sha256": source_record["sha256"],
    }
    if actual is None or source is None:
        return result
    candidates = {"unchanged": relative_error(actual, source)}
    # Compare small row probes only; names/formulas describe hypotheses, not accepted loader behavior.
    if name in ("head.weight", "embed.weight", "hc_head_fn") and source.ndim == 2 and source.shape[1] % hidden == 0:
        bases = [("W", source.astype(np.float32))]
        if name == "head.weight" and "norm.weight" in reference.weight_map:
            _, gamma = reference.read("norm.weight")
            if gamma is not None and gamma.shape == (source.shape[1],):
                fused = source.astype(np.float32) * gamma
                bases.append(("(W * original_norm_gamma)", fused))
                candidates[bases[-1][0]] = relative_error(actual, fused)
        if rotation is not None:
            for label, base in bases:
                blocks = base.reshape(-1, hidden)
                candidates[f"{label} @ R (per hidden-sized block)"] = relative_error(
                    actual, (blocks @ rotation).reshape(base.shape)
                )
                candidates[f"{label} @ R.T (per hidden-sized block)"] = relative_error(
                    actual, (blocks @ rotation.T).reshape(base.shape)
                )
    result["candidate_errors"] = candidates
    result["interpretation"] = "Sample evidence only; no candidate is automatically accepted as the export formula."
    return result


def selected_tensor(name, layers):
    if ".experts." in name:
        return False
    layer = re.match(r"(?:model\.)?layers\.([0-9]+)\.", name)
    if layer and int(layer[1]) not in layers:
        return False
    return bool(
        layer
        or is_mtp(name)
        or name in ("embed.weight", "head.weight", "norm.weight", "hc_head_fn", "hc_head_base", "hc_head_scale")
    )


def shard_declarations(metadata):
    grouped = {}
    for shard, record in metadata.items():
        for item in declarations(record["metadata"]):
            identity = json.dumps(item, sort_keys=True)
            entry = grouped.setdefault(identity, {**item, "shard_count": 0, "examples": []})
            entry["shard_count"] += 1
            if len(entry["examples"]) < 2:
                entry["examples"].append(shard)
    return list(grouped.values())


def collect(model, output_dir, reference_model=None, rotation_tensor="global_rotation"):
    checkpoint = Checkpoint(model)
    reference = Checkpoint(reference_model) if reference_model else None
    output_dir = Path(output_dir).resolve()
    for source in (checkpoint, reference):
        if source is not None:
            require(not output_dir.is_relative_to(source.root), "Write reports outside checkpoint directories")
    hidden = checkpoint.config.get("hidden_size")
    layers = checkpoint.config.get("num_hidden_layers")
    require(type(hidden) is int and hidden > 0 and type(layers) is int and layers > 0, "Invalid model geometry")
    selected_layers = {0, layers - 1}
    print(f"TRANSFORM_INSPECT_MODEL={checkpoint.root}", flush=True)
    rotation, matrix = rotation_evidence(checkpoint, rotation_tensor, hidden)
    tensors, failures = [], []
    for name in sorted(checkpoint.weight_map):
        if not selected_tensor(name, selected_layers):
            continue
        try:
            record, values = checkpoint.read(name)
            if reference:
                record["reference"] = reference_comparison(record, values, reference, matrix, hidden)
            tensors.append(record)
        except (ValueError, OSError, KeyError) as exc:
            failures.append({"name": name, "error": str(exc)})
    norms = [item for item in tensors if FINAL_NORM.fullmatch(item["name"])]
    heads = [
        item
        for item in tensors
        if item["name"] in ("embed.weight", "head.weight", "hc_head_fn", "hc_head_base", "hc_head_scale")
        or re.fullmatch(r"mtp\.[0-9]+\.(?:head\.weight|emb\.tok_emb\.weight|hc_head_(?:fn|base|scale))", item["name"])
    ]
    mtp_names = [name for name in checkpoint.weight_map if is_mtp(name)]
    families = Counter(re.sub(r"(?<=\.)[0-9]+(?=\.)", "*", name) for name in mtp_names)
    comparisons = Counter(item.get("reference", {}).get("status", "not_requested") for item in tensors)
    compared = [item for item in tensors if item.get("reference", {}).get("status") == "compared"]
    summary = {
        "schema_version": 1,
        "report_kind": "non_routed_transform_evidence",
        "status": "collected_with_errors" if failures else "collected",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": str(checkpoint.root),
        "index_sha256": checkpoint.index_sha256,
        "config_sha256": digest((checkpoint.root / "config.json").read_bytes()),
        "reference_model": str(reference.root) if reference else None,
        "reference_index_sha256": reference.index_sha256 if reference else None,
        "config_declarations": declarations(checkpoint.config),
        "metadata": sidecars(checkpoint.root),
        "shard_declarations": shard_declarations(checkpoint.shard_metadata),
        "final_norms": norms,
        "heads": heads,
        "rotation": rotation,
        "mtp": {
            "stage_counts": dict(Counter(name.split(".")[1] for name in mtp_names if name.startswith("mtp."))),
            "families": dict(sorted(families.items())),
        },
        "comparison_summary": {
            "statuses": dict(comparisons),
            "raw_equal": sum(item["reference"]["raw_bytes_equal"] for item in compared),
            "raw_different": sum(not item["reference"]["raw_bytes_equal"] for item in compared),
        },
        "coverage": {
            "main_layers": sorted(selected_layers),
            "selected_tensors": len(tensors),
            "target_payload_bytes_read": checkpoint.bytes_read,
            "reference_payload_bytes_read": reference.bytes_read if reference else 0,
        },
        "failures": failures,
        "exporter_questions": QUESTIONS,
        "limitations": [
            "Declarations describe exporter intent; they are not proof of tensor transformations.",
            "Large weights are read at selected rows only; sampled hashes are not full weight fingerprints.",
            "All-one Norm does not prove gamma fusion. Orthogonal/symmetric rotation does not prove where it is used.",
            "The optional reference must be the exact export base. "
            "Differences may reflect revision or quantization, not just rotation.",
            "No exporter code, vLLM, torch_npu, NPU, model inference or numerical acceptance is executed.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="transforms-", dir=output_dir))
    report = {**summary, "config": checkpoint.config, "shard_metadata": checkpoint.shard_metadata, "tensors": tensors}
    for filename, data in (("summary.json", summary), ("report.json", report)):
        (run_dir / filename).write_text(
            json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8"
        )
    print(f"TRANSFORM_INSPECT_STATUS={summary['status']}")
    print(
        f"SELECTED_TENSORS={len(tensors)} ROTATION_STATUS={rotation['status']} "
        f"MTP_STAGES={summary['mtp']['stage_counts']}"
    )
    print(f"SUMMARY_JSON={run_dir / 'summary.json'}")
    print(f"REPORT_JSON={run_dir / 'report.json'}")
    return run_dir, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference-model", type=Path, help="Optional exact original checkpoint used by the exporter")
    parser.add_argument(
        "--rotation-tensor", default="global_rotation", help="Exact auxiliary tensor name; never guessed from shape"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark-results/transform-inspect"))
    args = parser.parse_args(argv)
    try:
        _, summary = collect(args.model, args.output_dir, args.reference_model, args.rotation_tensor)
    except (OSError, ValueError, KeyError) as exc:
        print(f"TRANSFORM_INSPECT_STATUS=collection_failed ERROR={exc}", file=sys.stderr)
        return 1
    return int(bool(summary["failures"]))


if __name__ == "__main__":
    raise SystemExit(main())
