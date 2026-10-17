# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Repack the mixed GS32 source v1 checkpoint into per-expert source v3.

Python 3.10+; standard library only. No torch, NPU, or safetensors package
is required. Safetensors payloads are handled as bytes, including FP8 values.

Usage (PowerShell or bash; quote Windows paths):
    python tools/convert_vq2_mixed_gs32_v3.py --source "$SOURCE" --dry-run
    python tools/convert_vq2_mixed_gs32_v3.py --source "$SOURCE" --output "$OUTPUT"

Pass the source checkpoint explicitly with --source.
The output must not exist. Allow approximately 90 GiB of additional space
for this model. Only one expert field is buffered at a time. Completed
files are checked before the temporary directory is published as output.
On failure, the temporary directory is retained and its location is printed.

Source v3 uses natural output row order, ALL Gate rows followed by ALL Up
rows. VQ2 indices become
U8 [N/32,K/16,16,8], codebooks F8_E4M3 [N/32,K/32,16,2]; I32 perm and U64
out_dim_scale stay byte-identical. MXFP4 codes become U8 weight; signed
unbiased scale exponents become F8_E8M0 scale bytes. Non-routed weights keep
their existing names, dtypes and bytes. The complete root index is rebuilt.

The result is the source-v3 DELIVERY format, not the existing
ascend_vq2a8_gs32 runtime checkpoint format. Loading it requires the matching
v3 loader; this script does not implement or claim that runtime integration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

FORMAT = "vq2_mixed_gs32_source"
VERSION = 3
TILE = 32
HEADER_LIMIT = 100 * 1024 * 1024
COPY_CHUNK = 8 * 1024 * 1024
SPACE_MARGIN = 64 * 1024 * 1024
VQ_FIELDS = ("packed_indices", "codebooks", "perm", "out_dim_scale")
MX_FIELDS = ("weight", "scale")
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
EXPONENT_BYTES = bytes(((b if b < 128 else b - 256) + 127) % 256 for b in range(256))


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
    with path.open(encoding="utf-8") as stream:
        result = json.load(stream, object_pairs_hook=unique_object)
    require(isinstance(result, dict), f"Expected a JSON object: {path}")
    return result


def dtype_size(dtype):
    for name, size in DTYPE_SIZES:
        if name == dtype:
            return size
    raise ValueError(f"Unsupported safetensors dtype: {dtype}")


def read_header(path: Path) -> tuple[dict, int]:
    """Validate offsets and byte sizes without reading tensor payloads."""
    size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(8)
        require(len(prefix) == 8, f"Truncated safetensors prefix: {path}")
        length = struct.unpack("<Q", prefix)[0]
        require(0 < length <= HEADER_LIMIT and 8 + length <= size, f"Invalid header length: {path}")
        header = json.loads(stream.read(length), object_pairs_hook=unique_object)
    require(isinstance(header, dict), f"Invalid header object: {path}")
    metadata = header.get("__metadata__", {})
    require(isinstance(metadata, dict), f"Invalid metadata: {path}")
    require(all(isinstance(k, str) and isinstance(v, str) for k, v in metadata.items()), "Metadata must be strings")
    spans = []
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        require(isinstance(entry, dict), f"Invalid tensor entry: {path}:{name}")
        shape, offsets = entry.get("shape"), entry.get("data_offsets")
        require(isinstance(shape, list) and all(type(x) is int and x >= 0 for x in shape), f"Bad shape: {name}")
        require(isinstance(offsets, list) and len(offsets) == 2, f"Bad offsets: {name}")
        require(all(type(x) is int and x >= 0 for x in offsets), f"Bad offsets: {name}")
        begin, end = offsets
        expected = math.prod(shape) * dtype_size(entry.get("dtype"))
        require(end - begin == expected, f"Byte size does not match shape/dtype: {path}:{name}")
        spans.append((begin, end))
    end = 0
    for begin, next_end in sorted(spans):
        require(begin == end, f"Overlapping or non-contiguous tensor offsets: {path}")
        end = next_end
    require(8 + length + end == size, f"Truncated or trailing tensor data: {path}")
    return header, 8 + length


def local_file(root, relative):
    relative = Path(relative)
    require(not relative.is_absolute() and ".." not in relative.parts, f"Unsafe source path: {relative}")
    path = (root / relative).resolve()
    require(path.is_relative_to(root) and path.is_file(), f"Missing file or path outside source: {relative}")
    return path


def transpose_tiles(data, rows, columns, block_bytes):
    """Move K32,N32 blocks to N32,K32, preserving every byte within a block."""
    n_tiles, k_tiles = rows // TILE, columns // TILE
    require(len(data) == n_tiles * k_tiles * block_bytes, "Incorrect tile payload length")
    source = memoryview(data)
    result = bytearray(len(data))
    target = memoryview(result)
    for n in range(n_tiles):
        for k in range(k_tiles):
            start = (k * n_tiles + n) * block_bytes
            dest = (n * k_tiles + k) * block_bytes
            target[dest : dest + block_bytes] = source[start : start + block_bytes]
    return bytes(result)


def convert_tensor(data: bytes, field: str, rows: int, columns: int) -> bytes:
    """Transform one source-v1 expert field; validate value-bearing metadata."""
    require(rows > 0 and columns > 0 and rows % TILE == columns % TILE == 0, "N and K must be multiples of 32")
    if field == "packed_indices":
        return transpose_tiles(data, rows, columns, TILE * (TILE // 4))
    if field == "codebooks":
        require(b"\x7f" not in data and b"\xff" not in data, "Codebook contains E4M3FN NaN")
        return transpose_tiles(data, rows, columns, 16 * 2)
    if field == "perm":
        require(len(data) == columns * 4, "Incorrect perm length")
        values = [x[0] for x in struct.iter_unpack("<i", data)]
        require(sorted(values) == list(range(columns)), "perm is not a complete input-column permutation")
    elif field == "out_dim_scale":
        require(len(data) == rows * 8, "Incorrect out_dim_scale length")
        # Positive finite FP32 bits, with all upper U64 bits zero.
        require(all(0 < x[0] < 0x7F800000 for x in struct.iter_unpack("<Q", data)), "Invalid out_dim_scale bits")
    elif field == "weight":
        require(len(data) == rows * columns // 2, "Incorrect MXFP4 weight length")
    elif field == "scale":
        require(len(data) == rows * columns // TILE, "Incorrect MXFP4 scale length")
        require(b"\x80" not in data, "MXFP4 exponent -128 cannot be represented by finite E8M0")
        return data.translate(EXPONENT_BYTES)
    else:
        raise ValueError(f"Unknown conversion field: {field}")
    return data


@dataclass(frozen=True)
class TensorJob:
    name: str
    dtype: str
    shape: tuple[int, ...]
    field: str
    offset: int
    size: int
    rows: int
    columns: int


@dataclass(frozen=True)
class LayerJob:
    source: Path
    relative: str
    layer: int
    tensors: tuple[TensorJob, ...]


def field_contract(field, rows, columns):
    contracts = {
        "packed_indices": ("U8", (rows * columns // 4,), "U8", (rows // TILE, columns // 16, 16, 8)),
        "codebooks": (
            "F8_E4M3",
            (columns // TILE, rows // TILE, 16, 2),
            "F8_E4M3",
            (rows // TILE, columns // TILE, 16, 2),
        ),
        "perm": ("I32", (columns,), "I32", (columns,)),
        "out_dim_scale": ("U64", (rows,), "U64", (rows,)),
        "weight": ("U8", (rows, columns // 2), "U8", (rows, columns // 2)),
        "scale": ("I8", (rows, columns // TILE), "F8_E8M0", (rows, columns // TILE)),
    }
    return contracts[field]


def plan_layer(root, item, n_experts, hidden, intermediate, counts):
    layer = item["layer"]
    relative = f"vq2_mixed_gs32/layer_{layer:02d}.safetensors"
    source = local_file(root, relative)
    header, base = read_header(source)
    require(header.get("__metadata__", {}).get("format_version") == "1", f"Expected source-v1 shard: {source}")
    jobs, expected_keys = [], set()
    require(sorted(item["expert_ids"]) == list(range(n_experts)), f"Incomplete expert_ids in layer {layer}")
    for kind, projection, rows, columns in (
        ("gate_up", "w13", 2 * intermediate, hidden),
        ("down", "w2", hidden, intermediate),
    ):
        info = item[kind]
        require((info["rows"], info["columns"]) == (rows, columns), f"Projection dimensions disagree: {layer}:{kind}")
        banks = info["banks"]
        require(
            set(banks) <= {"vq", "mxfp4", "bf16"} and not banks.get("bf16"),
            "BF16/unknown expert banks are not supported",
        )
        ids = list(banks.get("vq", [])) + list(banks.get("mxfp4", []))
        require(
            all(type(x) is int for x in ids) and sorted(ids) == list(range(n_experts)),
            f"Duplicate/missing experts: {layer}:{kind}",
        )
        for bank, fields in (("vq", VQ_FIELDS), ("mxfp4", MX_FIELDS)):
            bank_ids = banks.get(bank, [])
            counts[f"{projection}_{bank}"] += len(bank_ids)
            for field in fields:
                source_field = "codes" if field == "weight" else field
                key = f"model.layers.{layer}.ffn.experts.{projection}_{bank}_{source_field}"
                if not bank_ids:
                    continue
                expected_keys.add(key)
                require(key in header, f"Missing tensor: {key}")
                entry = header[key]
                src_dtype, src_shape, out_dtype, out_shape = field_contract(field, rows, columns)
                require(
                    entry["dtype"] == src_dtype and entry["shape"] == [len(bank_ids), *src_shape],
                    f"Wrong dtype/shape: {key}",
                )
                size = math.prod(src_shape) * dtype_size(src_dtype)
                for slot, expert in enumerate(bank_ids):
                    jobs.append(
                        TensorJob(
                            f"layers.{layer}.ffn.experts.{expert}.{projection}.{field}",
                            out_dtype,
                            out_shape,
                            field,
                            base + entry["data_offsets"][0] + slot * size,
                            size,
                            rows,
                            columns,
                        )
                    )
    require(set(header) - {"__metadata__"} == expected_keys, f"Unexpected source tensors in layer {layer}")
    # Traverse the source sequentially, including each bank's manifest order.
    return LayerJob(source, relative, layer, tuple(sorted(jobs, key=lambda job: job.offset)))


def snapshot(paths):
    result = {}
    for path in paths:
        info = path.stat()
        result[str(path)] = (info.st_size, info.st_mtime_ns, info.st_dev, info.st_ino)
    return result


def plan_model(source):
    # Capture identity/size/time BEFORE parsing any manifest or shard header.
    initial_snapshot = snapshot(path for path in source.rglob("*") if path.is_file())
    config = read_json(local_file(source, "config.json"))
    quant = read_json(local_file(source, "quant.json"))
    manifest = read_json(local_file(source, "vq2_mixed_gs32/experts.json"))
    marker = read_json(local_file(source, "vq2_mixed_gs32/format.json"))
    for value in (quant, manifest, marker):
        require(
            value.get("format") == FORMAT and value.get("format_version") == 1, "Only mixed GS32 source v1 is supported"
        )
    dimensions = (
        config["num_hidden_layers"],
        config["n_routed_experts"],
        config["hidden_size"],
        config["moe_intermediate_size"],
    )
    n_layers, n_experts, hidden, intermediate = dimensions
    require(all(type(x) is int and x > 0 for x in dimensions), "Invalid model dimensions")
    require(hidden % TILE == intermediate % TILE == 0, "Model dimensions must be multiples of 32")
    require(
        (manifest["n_layers"], manifest["n_experts"], manifest["hidden_size"], manifest["moe_intermediate_size"])
        == dimensions,
        "Manifest dimensions disagree with config",
    )
    require(sorted(x["layer"] for x in manifest["layers"]) == list(range(n_layers)), "Duplicate/missing layers")
    counts = {f"{p}_{b}": 0 for p in ("w13", "w2") for b in ("vq", "mxfp4")}
    layers = [
        plan_layer(source, item, n_experts, hidden, intermediate, counts)
        for item in sorted(manifest["layers"], key=lambda x: x["layer"])
    ]
    index = read_json(local_file(source, "model.safetensors.index.json"))
    weight_map = index["weight_map"]
    require(isinstance(weight_map, dict) and weight_map, "Missing non-routed weight_map")
    has_routed_weights = any(
        part == "experts" or part.startswith("experts_") for name in weight_map for part in name.split(".")
    )
    require(not has_routed_weights, "Source root index must contain only non-routed weights")
    rest, total = {}, sum(job.size for layer in layers for job in layer.tensors)
    for relative in sorted(set(weight_map.values())):
        require(Path(relative).parts[0] == "rest", "Expected non-routed shards under rest/")
        path = local_file(source, relative)
        header, _ = read_header(path)
        expected = {key for key, value in weight_map.items() if value == relative}
        require(set(header) - {"__metadata__"} == expected, f"Root index and rest header disagree: {relative}")
        total += sum(
            entry["data_offsets"][1] - entry["data_offsets"][0]
            for key, entry in header.items()
            if key != "__metadata__"
        )
        rest[relative] = path
    sidecars = [
        path
        for path in source.iterdir()
        if path.is_file()
        and path.suffix in {".json", ".model", ".tiktoken", ".txt", ".py"}
        and path.name not in {"config.json", "quant.json", "model.safetensors.index.json"}
    ]
    paths = [
        source / name
        for name in (
            "config.json",
            "quant.json",
            "model.safetensors.index.json",
            "vq2_mixed_gs32/experts.json",
            "vq2_mixed_gs32/format.json",
        )
    ]
    paths += [layer.source for layer in layers] + list(rest.values()) + sidecars
    planned_snapshot = snapshot(paths)
    require(
        all(initial_snapshot.get(path) == info for path, info in planned_snapshot.items()),
        "Source files changed while building the conversion plan",
    )
    return {
        "config": config,
        "layers": layers,
        "rest": rest,
        "sidecars": sidecars,
        "weight_map": dict(weight_map),
        "total_size": total,
        "counts": counts,
        "snapshot": planned_snapshot,
    }


def write_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def verify_payload(path, expected_header, expected_digest):
    header, base = read_header(path)
    require(header == expected_header, f"Written header mismatch: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(base)
        while data := stream.read(COPY_CHUNK):
            digest.update(data)
    require(digest.hexdigest() == expected_digest, f"Written payload checksum mismatch: {path}")


def write_layer(layer, destination):
    header = {
        "__metadata__": {
            "format": FORMAT,
            "format_version": str(VERSION),
            "quant_method": "vq2a8",
            "layer": str(layer.layer),
            "row_order": "natural_gate_then_up",
        }
    }
    offset = 0
    for job in layer.tensors:
        header[job.name] = {"dtype": job.dtype, "shape": list(job.shape), "data_offsets": [offset, offset + job.size]}
        offset += job.size
    encoded = json.dumps(header, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    digest = hashlib.sha256()
    with layer.source.open("rb") as source, destination.open("xb") as output:
        output.write(struct.pack("<Q", len(encoded)))
        output.write(encoded)
        for job in layer.tensors:
            source.seek(job.offset)
            data = source.read(job.size)
            require(len(data) == job.size, f"Source truncated while reading {job.name}")
            try:
                result = convert_tensor(data, job.field, job.rows, job.columns)
            except ValueError as error:
                raise ValueError(f"{job.name}: {error}") from error
            require(len(result) == job.size, f"Conversion changed payload length: {job.name}")
            output.write(result)
            digest.update(result)
        output.flush()
        os.fsync(output.fileno())
    verify_payload(destination, header, digest.hexdigest())
    return digest.hexdigest()


def copy_verified(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with source.open("rb") as src, destination.open("xb") as dst:
        while data := src.read(COPY_CHUNK):
            digest.update(data)
            dst.write(data)
        dst.flush()
        os.fsync(dst.fileno())
    copied = hashlib.sha256()
    with destination.open("rb") as stream:
        while data := stream.read(COPY_CHUNK):
            copied.update(data)
    require(copied.digest() == digest.digest(), f"Copy checksum mismatch: {destination}")
    return digest.hexdigest()


def convert(source: Path, output: Path, *, dry_run: bool = False) -> dict:
    source, output = source.resolve(), output.resolve()
    require(source.is_dir(), f"Source directory does not exist: {source}")
    require(not output.exists(), f"Output already exists: {output}")
    require(
        not output.is_relative_to(source) and not source.is_relative_to(output),
        "Source and output must be separate directory trees",
    )
    plan = plan_model(source)
    expert_count = sum(len(layer.tensors) for layer in plan["layers"])
    report = {
        "format": FORMAT,
        "format_version": VERSION,
        "row_order": "natural_gate_then_up",
        "source": str(source),
        "output": str(output),
        "layers": len(plan["layers"]),
        "expert_tensors": expert_count,
        "rest_tensors": len(plan["weight_map"]),
        "total_tensors": expert_count + len(plan["weight_map"]),
        "total_size": plan["total_size"],
        "counts": plan["counts"],
        "dry_run": dry_run,
    }
    print(json.dumps(report, indent=2), flush=True)
    if dry_run:
        return report
    output.parent.mkdir(parents=True, exist_ok=True)
    require(
        shutil.disk_usage(output.parent).free >= plan["total_size"] + SPACE_MARGIN,
        "Insufficient free space for an independent output copy",
    )
    # Avoid tempfile: when invoked as tools/script.py, tools/bisect shadows
    # the stdlib bisect imported indirectly by tempfile on some Python versions.
    staging = output.parent / f".{output.name}.converting-{os.urandom(8).hex()}"
    staging.mkdir()
    print(f"Temporary output: {staging}", flush=True)
    try:
        checksums = {}
        for index, layer in enumerate(plan["layers"], 1):
            path = staging / layer.relative
            path.parent.mkdir(parents=True, exist_ok=True)
            checksums[layer.relative] = {"payload_sha256": write_layer(layer, path)}
            for job in layer.tensors:
                require(job.name not in plan["weight_map"], f"Duplicate output tensor: {job.name}")
                plan["weight_map"][job.name] = layer.relative
            print(f"Layer {index}/{len(plan['layers'])} verified: {layer.relative}", flush=True)
        for relative, path in plan["rest"].items():
            checksums[relative] = {"file_sha256": copy_verified(path, staging / relative)}
            print(f"Non-routed shard verified: {relative}", flush=True)
        for path in plan["sidecars"]:
            copy_verified(path, staging / path.name)
        config = dict(plan["config"])
        quantization = {"quant_method": "vq2a8", "format": FORMAT, "format_version": VERSION}
        config["quantization_config"] = quantization
        write_json(staging / "config.json", config)
        write_json(
            staging / "quant.json",
            {
                **quantization,
                "row_order": report["row_order"],
                "counts": plan["counts"],
                "index": "model.safetensors.index.json",
            },
        )
        write_json(staging / "vq2_mixed_gs32/format.json", {**quantization, "row_order": report["row_order"]})
        write_json(
            staging / "model.safetensors.index.json",
            {"metadata": {"total_size": plan["total_size"]}, "weight_map": dict(sorted(plan["weight_map"].items()))},
        )
        require(
            snapshot(Path(path) for path in plan["snapshot"]) == plan["snapshot"],
            "Source files changed during conversion",
        )
        require(len(plan["weight_map"]) == report["total_tensors"], "Output tensor count mismatch")
        write_json(
            staging / "conversion_report.json",
            {**report, "checksums": checksums, "verification": "all output payloads reread and SHA256 checked"},
        )
        require(not output.exists(), f"Output appeared during conversion: {output}")
        staging.rename(output)
    except BaseException:
        print(f"Conversion did not publish the output. Partial files retained at: {staging}", file=sys.stderr)
        raise
    print(f"Completed and verified: {output}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="New directory; default: <source>-v3")
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate all headers/manifests and print the plan; write nothing"
    )
    args = parser.parse_args()
    destination = args.output if args.output is not None else args.source.with_name(args.source.name + "-v3")
    try:
        convert(args.source, destination, dry_run=args.dry_run)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Conversion failed: {error}\n")


if __name__ == "__main__":
    main()
