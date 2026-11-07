# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packaged GS32 bridge capabilities and artifact-pair validation.

No build-tool, Torch, torch_npu or vLLM import is needed to check artifacts.
Schema checking receives the already imported Torch module from the caller.
"""

import hashlib
import json
from pathlib import Path

KERNEL_ENTRY = "mixed_w2w4_moe_ffn_v3"
FULL_FFN_ABI_VERSION = 3

REQUIRED_FULL_FFN_SCHEMAS = (
    ("initialize", "vq2a8_gs32_stage::initialize(str binary) -> ()"),
    (
        "_sanitize_device_counts",
        "vq2a8_gs32_stage::_sanitize_device_counts(Tensor counts, int rows, bool allow_padding) -> (Tensor, Tensor)",
    ),
    (
        "prepare_device_routes",
        "vq2a8_gs32_stage::prepare_device_routes(Tensor anchor, Tensor counts, bool allow_padding=False) -> "
        "__torch__.torch.classes.vq2a8_gs32_stage.DeviceRoutes",
    ),
    (
        "device_routes_status",
        "vq2a8_gs32_stage::device_routes_status("
        "__torch__.torch.classes.vq2a8_gs32_stage.DeviceRoutes routes) -> Tensor",
    ),
    ("full_ffn_abi_version", "vq2a8_gs32_stage::full_ffn_abi_version() -> int"),
    (
        "prepare_full_ffn",
        "vq2a8_gs32_stage::prepare_full_ffn(Tensor anchor, int n, int d, Tensor gate_w4, Tensor down_w4, "
        "float swiglu_limit) -> __torch__.torch.classes.vq2a8_gs32_stage.FullFFNPlan",
    ),
    (
        "full_ffn_device",
        "vq2a8_gs32_stage::full_ffn_device(Tensor input_bytes, Tensor input_scale, Tensor gate_weight, "
        "Tensor gate_table, Tensor gate_row_scale, Tensor down_weight, Tensor down_table, Tensor down_row_scale, "
        "__torch__.torch.classes.vq2a8_gs32_stage.DeviceRoutes routes, "
        "__torch__.torch.classes.vq2a8_gs32_stage.FullFFNPlan plan, Tensor(a!) hidden, "
        "Tensor(b!) hidden_scale, Tensor(c!) output) -> ()",
    ),
)


def verify_registered_device_schemas(torch_module, execution="full_graph", fusion="full_ffn"):
    """Check the full-FFN ABI without executing an NPU op; extra exports are allowed."""
    if execution not in ("fused", "full_graph"):
        raise ValueError(f"Unsupported GS32 execution mode: {execution}; use fused or full_graph")
    if fusion != "full_ffn":
        raise ValueError(f"Unsupported GS32 fusion mode: {fusion}; only full_ffn is supported")
    for name in ("DeviceRoutes", "FullFFNPlan"):
        try:
            getattr(torch_module.classes.vq2a8_gs32_stage, name)
        except (AttributeError, RuntimeError) as error:
            raise RuntimeError(f"GS32 schema check requires the registered {name} class.") from error
    actual = {}
    for name, expected in REQUIRED_FULL_FFN_SCHEMAS:
        try:
            schema = torch_module._C._dispatch_find_schema_or_throw(f"vq2a8_gs32_stage::{name}", "")
        except RuntimeError as error:
            raise RuntimeError(f"GS32 schema check: missing {name}; rebuild the bridge .so.") from error
        actual[name] = str(schema.schema())
        if " ".join(actual[name].split()) != " ".join(expected.split()):
            raise RuntimeError(f"GS32 schema mismatch for {name}: expected {expected}, found {actual[name]}")
    if not torch_module._C._dispatch_has_kernel_for_dispatch_key(
        "vq2a8_gs32_stage::_sanitize_device_counts", "PrivateUse1"
    ):
        raise RuntimeError("GS32 count sanitizer has no PrivateUse1 implementation; import vq2a8_gs32_counts first.")
    if torch_module.ops.vq2a8_gs32_stage.full_ffn_abi_version() != FULL_FFN_ABI_VERSION:
        raise RuntimeError("GS32 complete FFN ABI mismatch; rebuild the bridge .so and kernel .o.")
    return actual


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def kernel_evidence(path, *, require=True, require_abi=True):
    path = Path(path).resolve()
    if not path.is_file():
        if require:
            raise ValueError(f"Existing mixed kernel binary not found: {path}")
        return {"path": str(path), "present": False}
    size = path.stat().st_size
    with path.open("rb") as stream:
        magic = stream.read(4)
    if size < 4 or magic != b"\x7fELF":
        raise ValueError(f"Mixed kernel binary must be a non-empty ELF object: {path}")
    has_abi = KERNEL_ENTRY.encode("ascii") in path.read_bytes()
    if require_abi and not has_abi:
        raise ValueError(f"Mixed kernel lacks full-FFN ABI {FULL_FFN_ABI_VERSION}; rebuild the kernel .o: {path}")
    return {
        "path": str(path),
        "present": True,
        "bytes": size,
        "sha256": sha256(path),
        "abi_version": FULL_FFN_ABI_VERSION if has_abi else None,
        "entry": KERNEL_ENTRY if has_abi else None,
    }


def verify_runtime_artifacts(library, binary, *, require_full_ffn=True):
    """Validate exact build-paired bytes without loading Torch or executing an NPU op."""
    if not require_full_ffn:
        raise ValueError("Only full FFN artifact validation is supported")
    library = Path(library).resolve(strict=True)
    if not library.is_file() or not library.stat().st_size:
        raise ValueError(f"GS32 Torch bridge is missing or empty: {library}")
    kernel = kernel_evidence(binary, require_abi=require_full_ffn)
    library_hash = sha256(library)
    result = {"library": str(library), "library_sha256": library_hash, "kernel": kernel}
    manifest_path = library.parent / "build-manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Full FFN requires build-manifest.json beside the bridge; rebuild the GS32 runtime")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("status") not in ("built", "built_pending_target_validation")
        or manifest.get("abi_version") != FULL_FFN_ABI_VERSION
        or manifest.get("kernel_abi_version") != FULL_FFN_ABI_VERSION
        or manifest.get("library_sha256") != library_hash
        or manifest.get("deployed_kernel_sha256") != kernel["sha256"]
    ):
        raise ValueError("Full FFN bridge/kernel do not match the completed ABI v3 build manifest; rebuild both")
    result.update(abi_version=FULL_FFN_ABI_VERSION, build_manifest=str(manifest_path))
    return result
