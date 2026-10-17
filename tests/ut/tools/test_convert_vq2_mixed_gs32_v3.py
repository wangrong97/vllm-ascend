# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standard-library round trips for the mixed source v1 to v3 converter."""

import hashlib
import json
import random
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools import convert_vq2_mixed_gs32_v3 as converter

FORMAT = "vq2_mixed_gs32_source"
VQ_FIELDS = ("packed_indices", "codebooks", "perm", "out_dim_scale")
MX_FIELDS = ("weight", "scale")


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def write_safetensors(path, tensors, metadata=None):
    """Write real safetensors bytes without using the converter's serializer."""
    header = {"__metadata__": metadata or {"format": "pt"}}
    payload = bytearray()
    for name, (dtype, shape, data) in tensors.items():
        start = len(payload)
        payload.extend(data)
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [start, len(payload)]}
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def read_tensors(path):
    """Independently inspect the serialized output and extract each payload."""
    data = path.read_bytes()
    header_size = struct.unpack_from("<Q", data)[0]
    header = json.loads(data[8 : 8 + header_size])
    result = {}
    for name, entry in header.items():
        if name != "__metadata__":
            start, end = entry["data_offsets"]
            result[name] = (entry["dtype"], entry["shape"], data[8 + header_size + start : 8 + header_size + end])
    return header, result


def snapshot(root):
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }


def expected_packed(source, rows, columns):
    result = bytearray()
    for row_tile in range(rows // 32):
        for column in range(columns):
            for byte_in_column in range(8):
                offset = ((column // 32 * (rows // 32) + row_tile) * 32 + column % 32) * 8 + byte_in_column
                result.append(source[offset])
    return bytes(result)


def expected_codebooks(source, rows, columns):
    result = bytearray()
    for row_tile in range(rows // 32):
        for column_tile in range(columns // 32):
            offset = (column_tile * (rows // 32) + row_tile) * 32
            result.extend(source[offset : offset + 32])
    return bytes(result)


def source_payload(rows, columns, seed):
    packed = random.Random(seed).randbytes(rows * columns // 4)
    books = bytearray((index * 37 + seed) % 126 for index in range(rows * columns // 32))
    books[0] = 0x80  # FP8 negative zero must retain its bit pattern.
    permutation = [(index + seed) % columns for index in range(columns)]
    row_bits = [struct.unpack("<I", struct.pack("<f", 0.25 + (index + seed) / 128))[0] for index in range(rows)]
    exponents = (-127, -9, 0, 127)
    return {
        "packed_indices": packed,
        "codebooks": bytes(books),
        "perm": struct.pack(f"<{columns}i", *permutation),
        "out_dim_scale": struct.pack(f"<{rows}Q", *row_bits),
        "weight": random.Random(seed + 1000).randbytes(rows * columns // 2),
        "scale": bytes(exponents[(index + seed) % 4] & 0xFF for index in range(rows * columns // 32)),
    }


def expected_fields(payload, rows, columns, bank):
    if bank == "vq":
        return {
            "packed_indices": (
                "U8",
                [rows // 32, columns // 16, 16, 8],
                expected_packed(payload["packed_indices"], rows, columns),
            ),
            "codebooks": (
                "F8_E4M3",
                [rows // 32, columns // 32, 16, 2],
                expected_codebooks(payload["codebooks"], rows, columns),
            ),
            "perm": ("I32", [columns], payload["perm"]),
            "out_dim_scale": ("U64", [rows], payload["out_dim_scale"]),
        }
    return {
        "weight": ("U8", [rows, columns // 2], payload["weight"]),
        "scale": (
            "F8_E8M0",
            [rows, columns // 32],
            bytes(struct.unpack("b", bytes([value]))[0] + 127 for value in payload["scale"]),
        ),
    }


def make_source(root):
    """Build a complete source-v1 model with all four VQ/MX projection combinations."""
    root.mkdir()
    expert_dir, rest_dir = root / "vq2_mixed_gs32", root / "rest"
    expert_dir.mkdir()
    rest_dir.mkdir()
    config = {
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_v4",
        "hidden_size": 64,
        "moe_intermediate_size": 64,
        "num_hidden_layers": 1,
        "n_routed_experts": 4,
        "max_position_embeddings": 65536,
    }
    write_json(root / "config.json", config)
    write_json(root / "generation_config.json", {"eos_token_id": 1, "temperature": 0.7})
    format_info = {"quant_method": "vq2a8", "format": FORMAT, "format_version": 1}
    counts = {"w13_vq": 2, "w13_mxfp4": 2, "w2_vq": 2, "w2_mxfp4": 2}
    quant = {
        **format_info,
        "model": "DeepSeek-V4-Flash-0731",
        "n_layers": 1,
        "n_experts": 4,
        "hidden": 64,
        "intermediate": 64,
        "experts": {
            "dir": "vq2_mixed_gs32",
            "manifest": "vq2_mixed_gs32/experts.json",
            "files": "vq2_mixed_gs32/layer_00.safetensors",
            "counts": {**counts, "bf16": 0},
        },
        "rest": {"dir": "rest", "index": "model.safetensors.index.json", "keys": 2},
    }
    write_json(root / "quant.json", quant)
    write_json(expert_dir / "format.json", format_info)
    layer = {
        "layer": 0,
        "expert_ids": [0, 1, 2, 3],
        "gate_up": {"rows": 128, "columns": 64, "banks": {"vq": [2, 0], "mxfp4": [3, 1], "bf16": []}},
        "down": {"rows": 64, "columns": 64, "banks": {"vq": [1, 0], "mxfp4": [2, 3], "bf16": []}},
    }
    manifest = {
        **format_info,
        "n_layers": 1,
        "n_experts": 4,
        "hidden_size": 64,
        "moe_intermediate_size": 64,
        "counts": counts,
        "missing_experts": [],
        "layers": [layer],
    }
    write_json(expert_dir / "experts.json", manifest)
    tensors, expected = {}, {}
    for kind, prefix in (("gate_up", "w13"), ("down", "w2")):
        projection = layer[kind]
        rows, columns = projection["rows"], projection["columns"]
        for bank in ("vq", "mxfp4"):
            expert_ids = projection["banks"][bank]
            payloads = [
                source_payload(rows, columns, expert + (7 if kind == "gate_up" else 23)) for expert in expert_ids
            ]
            fields = VQ_FIELDS if bank == "vq" else MX_FIELDS
            for field in fields:
                source_field = "codes" if field == "weight" else field
                if field == "packed_indices":
                    dtype, shape = "U8", [rows * columns // 4]
                elif field == "codebooks":
                    dtype, shape = "F8_E4M3", [columns // 32, rows // 32, 16, 2]
                elif field == "perm":
                    dtype, shape = "I32", [columns]
                elif field == "out_dim_scale":
                    dtype, shape = "U64", [rows]
                elif field == "weight":
                    dtype, shape = "U8", [rows, columns // 2]
                else:
                    dtype, shape = "I8", [rows, columns // 32]
                name = f"model.layers.0.ffn.experts.{prefix}_{bank}_{source_field}"
                tensors[name] = (dtype, [len(expert_ids), *shape], b"".join(payload[field] for payload in payloads))
            for expert, payload in zip(expert_ids, payloads, strict=True):
                for field, entry in expected_fields(payload, rows, columns, bank).items():
                    expected[f"layers.0.ffn.experts.{expert}.{prefix}.{field}"] = entry
    write_safetensors(
        expert_dir / "layer_00.safetensors", tensors, {**{k: str(v) for k, v in format_info.items()}, "layer": "0"}
    )
    roots = {
        "model.norm.weight": ("BF16", [2], struct.pack("<HH", 0x3F80, 0x4000)),
        "model.layers.0.ffn.gate.tid2eid": ("I64", [1, 4], struct.pack("<4q", 0, 1, 2, 3)),
    }
    filename = "rest-model-00001-of-00001.safetensors"
    write_safetensors(rest_dir / filename, roots)
    write_json(
        rest_dir / "model.safetensors.index.json",
        {"metadata": {"format": "bf16succ-rest-0731"}, "weight_map": dict.fromkeys(roots, filename)},
    )
    write_json(
        root / "model.safetensors.index.json",
        {
            "metadata": {"format": FORMAT, "format_version": "1", "experts_manifest": "vq2_mixed_gs32/experts.json"},
            "weight_map": dict.fromkeys(roots, f"rest/{filename}"),
        },
    )
    expected.update(roots)
    return expected, config


class TensorConversionTests(unittest.TestCase):
    def test_layout_has_independent_element_oracle_and_natural_gate_up_order(self):
        rows, columns = 128, 64
        source = source_payload(rows, columns, 17)
        packed = converter.convert_tensor(source["packed_indices"], "packed_indices", rows, columns)
        books = converter.convert_tensor(source["codebooks"], "codebooks", rows, columns)
        self.assertEqual(packed, expected_packed(source["packed_indices"], rows, columns))
        self.assertEqual(books, expected_codebooks(source["codebooks"], rows, columns))
        self.assertEqual(books[0], 0x80)
        for row in range(rows):
            for column in range(columns):
                old_offset = ((column // 32 * (rows // 32) + row // 32) * 32 + column % 32) * 8 + row % 32 // 4
                new_offset = ((row // 32 * (columns // 16) + column // 16) * 16 + column % 16) * 8 + row % 32 // 4
                shift = (row % 4 // 2) * 4
                old_code = (source["packed_indices"][old_offset] >> shift) & 15
                new_code = (packed[new_offset] >> shift) & 15
                self.assertEqual(new_code, old_code)
                old_book = ((column // 32 * (rows // 32) + row // 32) * 16 + old_code) * 2 + row % 2
                new_book = ((row // 32 * (columns // 32) + column // 32) * 16 + new_code) * 2 + row % 2
                self.assertEqual(books[new_book], source["codebooks"][old_book])

    def test_unchanged_fields_and_e8m0_boundary_bytes(self):
        source = source_payload(64, 64, 7)
        for field in ("perm", "out_dim_scale", "weight"):
            with self.subTest(field=field):
                self.assertEqual(converter.convert_tensor(source[field], field, 64, 64), source[field])
        data = bytes((0x81, 0xF7, 0x00, 0x7F)) * 32
        self.assertEqual(converter.convert_tensor(data, "scale", 64, 64), bytes((0x00, 0x76, 0x7F, 0xFE)) * 32)

    def test_invalid_perm_and_scale_values_are_rejected(self):
        valid = source_payload(64, 64, 7)
        invalid = [
            ("perm", struct.pack("<64i", *([0] * 64))),
            ("perm", struct.pack("<64i", *range(-1, 63))),
            ("perm", struct.pack("<64i", *range(1, 65))),
            ("out_dim_scale", struct.pack("<Q", 0) + valid["out_dim_scale"][8:]),
            ("out_dim_scale", struct.pack("<Q", 0x100000001) + valid["out_dim_scale"][8:]),
            ("out_dim_scale", struct.pack("<Q", 0x7F800000) + valid["out_dim_scale"][8:]),
            ("out_dim_scale", struct.pack("<Q", 0xBF800000) + valid["out_dim_scale"][8:]),
            ("scale", b"\x80" + valid["scale"][1:]),
        ]
        for field, data in invalid:
            with self.subTest(field=field, prefix=data[:8].hex()), self.assertRaises(ValueError):
                converter.convert_tensor(data, field, 64, 64)


class CheckpointConversionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_direct_script_dry_run_works_outside_repository(self):
        source, output = self.root / "source with spaces", self.root / "converted with spaces"
        expected, _ = make_source(source)
        before = snapshot(source)
        result = subprocess.run(
            [
                sys.executable,
                "-X",
                "utf8",
                str(Path(converter.__file__).resolve()),
                "--source",
                str(source),
                "--output",
                str(output),
                "--dry-run",
            ],
            cwd=self.root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=15,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(result.stdout)
        self.assertTrue(report["dry_run"])
        self.assertEqual(report["format_version"], 3)
        self.assertEqual(report["row_order"], "natural_gate_then_up")
        self.assertEqual(report["total_tensors"], len(expected))
        self.assertEqual(report["total_size"], sum(len(entry[2]) for entry in expected.values()))
        self.assertFalse(output.exists())
        self.assertEqual(snapshot(source), before)

    def test_complete_round_trip_index_and_source_preservation(self):
        source, output = self.root / "source", self.root / "converted"
        expected, source_config = make_source(source)
        before = snapshot(source)
        self.assertIsInstance(converter.convert(source, output, dry_run=True), dict)
        self.assertFalse(output.exists())
        self.assertEqual(snapshot(source), before)
        self.assertIsInstance(converter.convert(source, output), dict)
        self.assertEqual(snapshot(source), before)
        config = json.loads((output / "config.json").read_text(encoding="utf-8"))
        self.assertEqual({k: config[k] for k in source_config}, source_config)
        self.assertEqual(config["quantization_config"]["quant_method"], "vq2a8")
        self.assertEqual(config["quantization_config"]["format"], FORMAT)
        self.assertEqual(config["quantization_config"]["format_version"], 3)
        index = json.loads((output / "model.safetensors.index.json").read_text(encoding="utf-8"))
        self.assertEqual(set(index["weight_map"]), set(expected))
        self.assertEqual(index["metadata"]["total_size"], sum(len(entry[2]) for entry in expected.values()))
        actual = {}
        for filename in set(index["weight_map"].values()):
            path = output / filename
            self.assertTrue(path.resolve().is_relative_to(output.resolve()))
            header, values = read_tensors(path)
            self.assertFalse(set(actual) & set(values))
            for name in values:
                self.assertEqual(index["weight_map"][name], filename)
            if any(name.startswith("layers.0.ffn.experts.") for name in values):
                self.assertEqual(header["__metadata__"]["format_version"], "3")
            actual.update(values)
        self.assertEqual(actual, expected)
        self.assertEqual(
            (output / "generation_config.json").read_bytes(), (source / "generation_config.json").read_bytes()
        )
        preserved = snapshot(output)
        with self.assertRaises((ValueError, FileExistsError)):
            converter.convert(source, output)
        self.assertEqual(snapshot(output), preserved)
        self.assertEqual(snapshot(source), before)

    def test_nested_output_is_rejected_without_changing_source(self):
        source = self.root / "source"
        make_source(source)
        before = snapshot(source)
        with self.assertRaises((ValueError, FileExistsError)):
            converter.convert(source, source / "converted")
        self.assertEqual(snapshot(source), before)

    def test_duplicate_bank_expert_prevents_publication(self):
        source, output = self.root / "source", self.root / "converted"
        make_source(source)
        manifest_path = source / "vq2_mixed_gs32/experts.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["layers"][0]["gate_up"]["banks"]["vq"] = [2, 2]
        write_json(manifest_path, manifest)
        before = snapshot(source)
        with self.assertRaises(ValueError):
            converter.convert(source, output)
        self.assertFalse(output.exists())
        self.assertEqual(snapshot(source), before)

    def test_header_rejects_truncation_duplicate_keys_and_overlapping_data(self):
        path = self.root / "bad.safetensors"
        malformed = [
            b"short",
            struct.pack("<Q", 1000) + b"{}",
            struct.pack("<Q", 15) + b'{"x":{},"x":{}}',
        ]
        for data in malformed:
            with self.subTest(data=data[:24]):
                path.write_bytes(data)
                with self.assertRaises(ValueError):
                    converter.read_header(path)
        header = {
            "a": {"dtype": "U8", "shape": [2], "data_offsets": [0, 2]},
            "b": {"dtype": "U8", "shape": [2], "data_offsets": [1, 3]},
        }
        encoded = json.dumps(header).encode("utf-8")
        encoded += b" " * (-len(encoded) % 8)
        path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"123")
        with self.assertRaises(ValueError):
            converter.read_header(path)


if __name__ == "__main__":
    unittest.main()
