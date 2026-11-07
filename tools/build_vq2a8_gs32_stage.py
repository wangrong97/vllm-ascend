#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the GS32 full-FFN Torch adapter around a matching ABI v3 mixed ELF."""

from __future__ import annotations

# Keep tools/bisect out of the direct-script import path.
# ruff: noqa: E402
import os
import sys

if not __package__:
    sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse
import importlib.util
import json
import platform
import shutil
import struct
import subprocess
from pathlib import Path

import regex as re

from tools.vq2a8_live_log import LiveChildLog

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "csrc/vq2a8_gs32_stage"
BINDING = SOURCE / "torch_binding.cpp"
COUNTS_WRAPPER = REPO / "vllm_ascend/ops/fused_moe/vq2a8_gs32_counts.py"
COUNTS_KERNEL = REPO / "vllm_ascend/ops/triton/activation/vq2a8_gs32_counts.py"
MIXED_SOURCE = REPO / "csrc/mixed_w2w4_moe_ffn"
TARGET = "vq2a8_gs32_stage"
LIBRARY_NAME = f"lib{TARGET}.so"
KERNEL_NAME = "mixed_w2w4_moe_ffn.o"
GENERATED_SOURCE_NAME = "host-extension-source"
SCHEMA_CHECK_TIMEOUT_SECONDS = 60
ELF_HEADER_PREFIX_FORMAT = "<16sHH"
ELF_SHARED_OBJECT = 3
ELF_MACHINE_X86_64 = 62
ELF_MACHINE_AARCH64 = 183


def _runtime_helpers():
    # The wheel owns runtime validation. Load that dependency-free file without
    # importing the plugin package: cross-build hosts need neither vLLM nor NPU.
    path = REPO / "vllm_ascend/quantization/vq2a8_gs32_runtime.py"
    spec = importlib.util.spec_from_file_location("vq2a8_gs32_runtime_build", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return (
        module.FULL_FFN_ABI_VERSION,
        module.KERNEL_ENTRY,
        module.REQUIRED_FULL_FFN_SCHEMAS,
        module.kernel_evidence,
        module.sha256,
        module.verify_registered_device_schemas,
        module.verify_runtime_artifacts,
    )


(
    FULL_FFN_ABI_VERSION,
    KERNEL_ENTRY,
    REQUIRED_FULL_FFN_SCHEMAS,
    kernel_evidence,
    sha256,
    verify_registered_device_schemas,
    verify_runtime_artifacts,
) = _runtime_helpers()


def register_counts_implementation():
    # Standalone registration must not import ops/__init__.py, which registers
    # unrelated model operators. The wrapper itself imports only Torch; its
    # Triton kernel remains lazy until actual device route preparation.
    spec = importlib.util.spec_from_file_location("vq2a8_gs32_counts_build", COUNTS_WRAPPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def verify_loaded_library(library, execution="full_graph", fusion="full_ffn", binary=None):
    """Run in a fresh process so an earlier .so cannot mask stale registration."""
    if execution not in ("fused", "full_graph"):
        raise ValueError(f"Unsupported GS32 execution mode: {execution}; use fused or full_graph")
    if fusion != "full_ffn":
        raise ValueError(f"Unsupported GS32 fusion mode: {fusion}; only full_ffn is supported")
    if binary is None:
        raise ValueError("Full FFN preflight requires the kernel binary path")
    verify_runtime_artifacts(library, binary)

    import torch
    import torch_npu  # noqa: F401

    torch.ops.load_library(str(Path(library).resolve(strict=True)))
    register_counts_implementation()
    return verify_registered_device_schemas(torch, execution, fusion)


def verify_library_schemas(library, log):
    # Import/load/query only: no initialize(), device selection, allocation,
    # kernel launch or graph capture. This can run while an NPU is occupied.
    # The manifest is not finalized until this build-time registration check
    # passes. Runtime preflight separately verifies the completed artifact pair.
    command = [
        sys.executable,
        "-c",
        "import json, sys\n"
        "from tools.build_vq2a8_gs32_stage import register_counts_implementation, verify_registered_device_schemas\n"
        "import torch, torch_npu\n"
        "torch.ops.load_library(sys.argv[1])\n"
        "register_counts_implementation()\n"
        "print(json.dumps(verify_registered_device_schemas(torch), sort_keys=True))\n",
        str(library.resolve()),
    ]
    print(f"GS32_STAGE_BUILD_STAGE=schema LOG={log}", flush=True)
    with log.open("w", encoding="utf-8") as stream, LiveChildLog(log, "gs32-stage-schema"):
        result = subprocess.run(
            command,
            cwd=REPO,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=SCHEMA_CHECK_TIMEOUT_SECONDS,
        )
    if result.returncode:
        raise RuntimeError(f"schema check failed ({result.returncode}); inspect {log}")


def source_hashes():
    required = (
        BINDING,
        COUNTS_WRAPPER,
        COUNTS_KERNEL,
        MIXED_SOURCE / "mixed_tiling.h",
        MIXED_SOURCE / "tiling_builders.h",
        *sorted((MIXED_SOURCE / "components").rglob("*tiling*.h")),
    )
    missing = [path.relative_to(REPO).as_posix() for path in required if not path.is_file()]
    if missing:
        raise ValueError("GS32 stage sources missing: " + ", ".join(missing))
    return {path.relative_to(REPO).as_posix(): sha256(path) for path in required}


def target_settings(args):
    host = platform.machine().lower()
    host = {"amd64": "x86_64", "arm64": "aarch64"}.get(host, host)
    target = host if args.target_arch == "native" else args.target_arch
    if target not in ("aarch64", "x86_64"):
        raise ValueError(f"Unsupported host target: {target}")
    cross = args.toolchain_file is not None or target != host
    if args.toolchain_file is not None and args.target_arch == "native":
        raise ValueError("--toolchain-file requires an explicit --target-arch")
    if cross and not all((args.toolchain_file, args.torch_cmake, args.torch_npu_path)):
        raise ValueError("Cross compilation requires --toolchain-file, --torch-cmake and --torch-npu-path")
    if not cross and any((args.torch_cmake, args.torch_npu_path)):
        raise ValueError("Target Torch paths require --target-arch and --toolchain-file")
    return target, cross


def verify_host_elf(path, target_arch):
    """Reject a host library for the wrong CPU without loading or executing it."""
    header_size = struct.calcsize(ELF_HEADER_PREFIX_FORMAT)
    with Path(path).open("rb") as stream:
        header = stream.read(header_size)
    if len(header) != header_size or not header.startswith(b"\x7fELF\x02\x01"):
        raise ValueError(f"Expected a little-endian ELF64 host library: {path}")
    _, elf_type, machine = struct.unpack(ELF_HEADER_PREFIX_FORMAT, header)
    if elf_type != ELF_SHARED_OBJECT:
        raise ValueError(f"Expected an ELF shared library (ET_DYN): {path}")
    architecture = {ELF_MACHINE_X86_64: "x86_64", ELF_MACHINE_AARCH64: "aarch64"}.get(machine)
    if architecture != target_arch:
        raise ValueError(f"Host ELF architecture mismatch: {path}: machine={machine}, expected {target_arch}")
    return architecture


def cross_dependencies(args, target_arch):
    """Inspect the target SDK on disk; never import the build host's Torch."""
    args.toolchain_file.resolve(strict=True)
    torch_cmake = args.torch_cmake.resolve(strict=True)
    torch_npu = args.torch_npu_path.resolve(strict=True)
    if not (torch_cmake / "Torch/TorchConfig.cmake").is_file():
        raise ValueError("--torch-cmake must point to the target torch/share/cmake directory")
    if not (torch_npu / "include").is_dir():
        raise ValueError("--torch-npu-path must contain target TorchNPU headers")
    torch_lib = torch_cmake.parent.parent / "lib"
    for name in ("libtorch.so", "libtorch_cpu.so", "libc10.so"):
        verify_host_elf(torch_lib / name, target_arch)
    verify_host_elf(torch_npu / "lib/libtorch_npu.so", target_arch)
    runtime_dirs = (
        args.cann / "lib64",
        args.cann / f"{target_arch}-linux/devlib",
        args.driver / "lib64/common",
        args.driver / "lib64/driver",
    )
    for name in ("libascendcl.so", "libruntime.so"):
        library = next((directory / name for directory in runtime_dirs if (directory / name).is_file()), None)
        if library is None:
            raise ValueError(f"Target {name} missing under --cann/--driver")
        verify_host_elf(library, target_arch)
    return torch_npu, torch_cmake


def validate_options(args):
    target_arch, cross = target_settings(args)
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    directory = args.build_dir.resolve()
    if (
        directory in (REPO, SOURCE, MIXED_SOURCE)
        or directory in REPO.parents
        or SOURCE in directory.parents
        or MIXED_SOURCE in directory.parents
    ):
        raise ValueError("Choose a dedicated out-of-source build directory")
    generated_source = directory / GENERATED_SOURCE_NAME
    cmake_cache = directory / "cmake-build/CMakeCache.txt"
    if cmake_cache.is_file():
        match = re.search(r"^CMAKE_HOME_DIRECTORY:INTERNAL=(.+)$", cmake_cache.read_text(), re.MULTILINE)
        if match and Path(match.group(1).strip()).resolve() != generated_source.resolve():
            raise ValueError("Build directory belongs to another project; do not overwrite it")
        manifest = directory / "build-manifest.json"
        previous = json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file() else {}
        previous_arch = previous.get("target_architecture", previous.get("host_architecture"))
        toolchain = str(args.toolchain_file.resolve()) if args.toolchain_file else None
        if (
            (previous_arch is not None and previous_arch != target_arch)
            or previous.get("cross_compiling", False) != cross
            or (cross and previous_arch is None)
            or previous.get("toolchain_file") != toolchain
            or (cross and previous.get("toolchain_sha256") != sha256(args.toolchain_file))
            or (
                cross
                and any(
                    previous.get(name) != str(path.resolve())
                    for name, path in (
                        ("torch_cmake", args.torch_cmake),
                        ("torch_npu_path", args.torch_npu_path),
                        ("cann_path", args.cann),
                        ("driver_path", args.driver),
                    )
                )
            )
        ):
            raise ValueError("Build directory has a different target/toolchain/SDK; use a fresh --build-dir")
    return directory


def cmake_project():
    binding = BINDING.resolve().as_posix().replace('"', '\\"')
    return f"""# Generated by tools/build_vq2a8_gs32_stage.py; do not edit.
cmake_minimum_required(VERSION 3.16)
project(vq2a8_gs32_stage LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "Build the GS32 stage adapter on Linux with CANN 9.1 and torch_npu installed")
endif()
if(DEFINED GS32_TARGET_ARCH AND NOT CMAKE_SYSTEM_PROCESSOR STREQUAL GS32_TARGET_ARCH)
  message(FATAL_ERROR
    "Toolchain processor ${{CMAKE_SYSTEM_PROCESSOR}} does not match --target-arch ${{GS32_TARGET_ARCH}}")
endif()
# Match the main CANN build's host-library layout; AI Core targets are
# selected separately by the existing mixed-kernel build script.
set(SYSTEM_PREFIX ${{CMAKE_SYSTEM_PROCESSOR}}-linux)
if(NOT IS_DIRECTORY "${{ASCEND_HOME_PATH}}" OR NOT IS_DIRECTORY "${{TORCH_NPU_PATH}}")
  message(FATAL_ERROR "ASCEND_HOME_PATH and TORCH_NPU_PATH must be existing directories")
endif()
if(GS32_CROSS_COMPILING)
  # The selected target package supplies the C++ ABI flags. Do not let an
  # ARM Torch installation in the build environment override its prefix.
  unset(ENV{{TORCH_INSTALL_PREFIX}})
  find_package(Torch REQUIRED CONFIG PATHS "${{Torch_DIR}}" NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
else()
  find_package(Torch REQUIRED)
endif()
separate_arguments(TORCH_FLAGS NATIVE_COMMAND "${{TORCH_CXX_FLAGS}}")
add_library({TARGET} SHARED "{binding}")
target_compile_options({TARGET} PRIVATE ${{TORCH_FLAGS}})
target_include_directories({TARGET} PRIVATE
  ${{TORCH_INCLUDE_DIRS}}
  "${{TORCH_NPU_PATH}}/include"
  "${{ASCEND_HOME_PATH}}/include")
target_link_directories({TARGET} PRIVATE
  "${{TORCH_NPU_PATH}}/lib"
  "${{ASCEND_HOME_PATH}}/lib64"
  "${{ASCEND_HOME_PATH}}/${{SYSTEM_PREFIX}}/devlib"
  "${{ASCEND_DRIVER_PATH}}/lib64/common"
  "${{ASCEND_DRIVER_PATH}}/lib64/driver")
target_link_libraries({TARGET} PRIVATE ${{TORCH_LIBRARIES}} torch_npu ascendcl runtime)
# Reject unresolved adapter symbols at link time instead of passing a build
# whose library later fails to load through torch.ops.load_library.
target_link_options({TARGET} PRIVATE "LINKER:-z,defs")
set(STAGE_BUILD_RPATH
  "${{TORCH_NPU_PATH}}/lib;${{ASCEND_HOME_PATH}}/lib64"
  "${{ASCEND_DRIVER_PATH}}/lib64/common;${{ASCEND_DRIVER_PATH}}/lib64/driver")
if(GS32_CROSS_COMPILING)
  # SDK/sysroot paths belong to the build host. The target imports TorchNPU
  # and loads its own CANN environment before loading this library.
  set(STAGE_BUILD_RPATH "")
endif()
set_target_properties({TARGET} PROPERTIES
  # CMake otherwise adds link directories to the build RPATH, including
  # devlib stubs. Load only the real CANN/driver libraries at runtime.
  BUILD_WITH_INSTALL_RPATH TRUE
  INSTALL_RPATH "${{STAGE_BUILD_RPATH}}"
  LIBRARY_OUTPUT_DIRECTORY "${{STAGE_OUTPUT_DIRECTORY}}")
"""


def cmake_commands(args, directory, torch_npu_path, torch_cmake):
    target_arch, cross = target_settings(args)
    generated_source = directory / GENERATED_SOURCE_NAME
    cmake_build = directory / "cmake-build"
    commands = [
        [
            "cmake",
            "-S",
            str(generated_source),
            "-B",
            str(cmake_build),
            f"-DASCEND_HOME_PATH={args.cann.resolve()}",
            f"-DASCEND_DRIVER_PATH={args.driver.resolve()}",
            f"-DTORCH_NPU_PATH={torch_npu_path}",
            f"-DCMAKE_PREFIX_PATH={torch_cmake}",
            f"-DSTAGE_OUTPUT_DIRECTORY={directory}",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        ["cmake", "--build", str(cmake_build), "--target", TARGET, "-j", str(args.jobs), "--verbose"],
    ]
    if args.target_arch != "native":
        commands[0].append(f"-DGS32_TARGET_ARCH={target_arch}")
    commands[0].append(f"-DGS32_CROSS_COMPILING={'ON' if cross else 'OFF'}")
    if cross:
        commands[0].extend(
            [
                f"-DCMAKE_TOOLCHAIN_FILE={args.toolchain_file.resolve()}",
                f"-DTorch_DIR={Path(torch_cmake) / 'Torch'}",
                f"-DCaffe2_DIR={Path(torch_cmake) / 'Caffe2'}",
            ]
        )
    return commands


def _save(path, report):
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def build(args):
    directory = validate_options(args)
    target_arch, cross = target_settings(args)
    evidence = kernel_evidence(args.kernel_binary, require=not args.dry_run, require_abi=not args.dry_run)
    report = {
        "status": "planned" if args.dry_run else "building",
        "implementation": "vq2a8_gs32_stage",
        "abi_version": FULL_FFN_ABI_VERSION,
        "kernel_abi_version": FULL_FFN_ABI_VERSION,
        "source_sha256": source_hashes(),
        "kernel_binary": evidence,
        "kernel_is_built_by_this_tool": False,
        "host_extension_only": True,
        "host_system": platform.system(),
        "host_architecture": platform.machine(),
        "target_architecture": target_arch,
        "cross_compiling": cross,
        "toolchain_file": str(args.toolchain_file.resolve()) if cross else None,
        "cann_path": str(args.cann.resolve()),
        "driver_path": str(args.driver.resolve()),
        "python": sys.executable,
        "cann_9_1_compile_verified": False,
        "device_execution_verified": False,
        "host_schema_verified": False,
        "graph_api_available": False,
        "graph_execution_verified": False,
        "model_execution_verified": False,
        "performance_verified": False,
        "default_model_backend": "unchanged",
    }
    if args.dry_run:
        report["commands"] = cmake_commands(
            args,
            directory,
            args.torch_npu_path.resolve() if cross else "<selected-Python-torch_npu>",
            args.torch_cmake.resolve() if cross else "<selected-Python-torch-cmake>",
        )
        report["deployed_kernel"] = str(directory / KERNEL_NAME)
        print(json.dumps(report, indent=2))
        return 0
    if platform.system() != "Linux":
        raise RuntimeError("Actual compilation requires Linux CANN/torch_npu; use --dry-run on Windows")

    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "build-manifest.json"
    _save(manifest, report)
    try:
        cann = args.cann.resolve(strict=True)
        driver = args.driver.resolve(strict=True)
        if not (cann / "include").is_dir():
            raise RuntimeError(f"CANN headers missing under {cann}")
        if not (driver / "lib64/driver").is_dir():
            raise RuntimeError(f"Ascend driver runtime libraries missing under {driver}")

        if cross:
            torch_npu_path, torch_cmake = cross_dependencies(args, target_arch)
            report["toolchain_sha256"] = sha256(args.toolchain_file)
        else:
            import torch

            spec = importlib.util.find_spec("torch_npu")
            if spec is None or not spec.submodule_search_locations:
                raise RuntimeError("torch_npu is not installed in the selected Python")
            torch_npu_path = Path(next(iter(spec.submodule_search_locations))).resolve()
            torch_cmake = torch.utils.cmake_prefix_path
            report["torch"] = torch.__version__
        report["torch_npu_path"] = str(torch_npu_path)
        report["torch_cmake"] = str(torch_cmake)

        generated_source = directory / GENERATED_SOURCE_NAME
        generated_source.mkdir(parents=True, exist_ok=True)
        generated_cmake = generated_source / "CMakeLists.txt"
        generated_cmake.write_text(cmake_project(), encoding="utf-8")
        report["generated_cmake_sha256"] = sha256(generated_cmake)
        report["commands"] = cmake_commands(
            args,
            directory,
            torch_npu_path,
            torch_cmake,
        )
        _save(manifest, report)

        for name, command in zip(("configure", "compile"), report["commands"]):
            report["current_step"] = name
            _save(manifest, report)
            log = directory / f"{name}.log"
            print(f"GS32_STAGE_BUILD_STAGE={name} LOG={log}", flush=True)
            print("COMMAND=" + json.dumps(command), flush=True)
            with log.open("w", encoding="utf-8") as stream, LiveChildLog(log, f"gs32-stage-{name}"):
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
            if result.returncode:
                raise RuntimeError(f"{name} failed ({result.returncode}); inspect {log}")

        library = directory / LIBRARY_NAME
        if not library.is_file() or not library.stat().st_size:
            raise RuntimeError(f"Host extension was not produced: {library}")
        report["library_architecture"] = verify_host_elf(library, target_arch)
        if source_hashes() != report["source_sha256"]:
            raise RuntimeError("Adapter sources changed while compiling; rebuild before validation")
        if kernel_evidence(args.kernel_binary) != evidence:
            raise RuntimeError("Mixed kernel binary changed while compiling; rebuild before validation")

        if not cross:
            report["current_step"] = "schema"
            _save(manifest, report)
            verify_library_schemas(library, directory / "schema.log")

        deployed_kernel = directory / KERNEL_NAME
        if args.kernel_binary.resolve() != deployed_kernel.resolve():
            shutil.copyfile(args.kernel_binary, deployed_kernel)
        report.update(
            status="built_pending_target_validation" if cross else "built",
            library=str(library),
            library_sha256=sha256(library),
            deployed_kernel=str(deployed_kernel),
            deployed_kernel_sha256=sha256(deployed_kernel),
            # A successful target build verifies only declarations/linkage.
            cann_9_1_compile_verified=True,
            host_schema_verified=not cross,
            required_device_schemas=dict(REQUIRED_FULL_FFN_SCHEMAS),
            graph_api_available=not cross,
        )
        report.pop("current_step", None)
        _save(manifest, report)
        result = "CROSS_COMPILED" if cross else "PASS"
        print(f"GS32_STAGE_BUILD={result} LIBRARY={library} KERNEL={deployed_kernel} REPORT={manifest}", flush=True)
        print(
            f"HOST_SCHEMA_VERIFIED={not cross} GRAPH_API_AVAILABLE={not cross} "
            "DEVICE_EXECUTION_VERIFIED=False GRAPH_EXECUTION_VERIFIED=False",
            flush=True,
        )
        if cross:
            print(
                "Run verify_loaded_library(library, binary=kernel) on the target x86/ARM Linux before NPU validation.",
                flush=True,
            )
        return 0
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        _save(manifest, report)
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-binary", type=Path, required=True, help="Existing mixed_w2w4_moe_ffn.o ELF")
    parser.add_argument("--cann", type=Path, default=Path("/usr/local/Ascend/cann-9.1.0"))
    parser.add_argument("--driver", type=Path, default=Path("/usr/local/Ascend/driver"))
    parser.add_argument("--build-dir", type=Path, default=REPO / "build/vq2a8-gs32-stage")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--target-arch", choices=("native", "aarch64", "x86_64"), default="native")
    parser.add_argument("--toolchain-file", type=Path, help="CMake toolchain using a compiler runnable on this host")
    parser.add_argument(
        "--torch-cmake", type=Path, help="Target torch/share/cmake directory; required for cross builds"
    )
    parser.add_argument(
        "--torch-npu-path", type=Path, help="Target torch_npu package directory; required for cross builds"
    )
    parser.add_argument("--dry-run", action="store_true", help="print commands/hashes; do not write or compile")
    return parser.parse_args(argv)


def main(argv=None):
    try:
        return build(parse_args(argv))
    except (OSError, RuntimeError, ValueError, ImportError, subprocess.TimeoutExpired) as exc:
        print(f"GS32_STAGE_BUILD=FAIL ERROR={exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
