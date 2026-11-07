# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-build contracts; these tests do not compile or execute an Ascend op."""

import builtins
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import build_vq2a8_gs32_stage as build


def write_target_elf(path, architecture="x86_64"):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = bytearray(64)
    header[:7] = b"\x7fELF\x02\x01\x01"
    header[16:18] = (3).to_bytes(2, "little")
    header[18:20] = {"aarch64": 183, "x86_64": 62}[architecture].to_bytes(2, "little")
    path.write_bytes(header)
    return path


@pytest.fixture
def cross_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(build.platform, "system", lambda: "Linux")
    monkeypatch.setattr(build.platform, "machine", lambda: "aarch64")
    cann, driver, torch, torch_npu = (tmp_path / name for name in ("cann", "driver", "torch", "torch_npu"))
    torch_cmake = torch / "share/cmake"
    for directory in (cann / "include", torch / "include", torch_npu / "include", torch_cmake / "Torch"):
        directory.mkdir(parents=True)
    (torch_cmake / "Torch/TorchConfig.cmake").write_text("set(Torch_FOUND TRUE)\n", encoding="utf-8")
    libraries = [write_target_elf(torch / "lib" / name) for name in ("libtorch.so", "libtorch_cpu.so", "libc10.so")]
    libraries.extend(
        (
            write_target_elf(torch_npu / "lib/libtorch_npu.so"),
            write_target_elf(cann / "lib64/libascendcl.so"),
            write_target_elf(driver / "lib64/driver/libruntime.so"),
        )
    )
    toolchain = tmp_path / "x86_64-toolchain.cmake"
    # Only the configure test consumes this toolchain. A Python executable is
    # deliberately substituted for C++ and compiler probes are disabled: real
    # CMake still evaluates target architecture and dependency/RPATH policy.
    toolchain.write_text(
        "set(CMAKE_SYSTEM_NAME Linux)\n"
        "set(CMAKE_SYSTEM_PROCESSOR x86_64)\n"
        f'set(CMAKE_CXX_COMPILER "{Path(build.sys.executable).as_posix()}")\n'
        "set(CMAKE_CXX_COMPILER_ID GNU)\n"
        "set(CMAKE_CXX_COMPILER_ID_RUN TRUE)\n"
        "set(CMAKE_CXX_COMPILER_FORCED TRUE)\n"
        "set(CMAKE_CXX_COMPILER_WORKS TRUE)\n",
        encoding="utf-8",
    )
    directory = tmp_path / "output"
    binary = write_target_elf(tmp_path / "mixed.o", "aarch64")
    binary.write_bytes(binary.read_bytes() + build.KERNEL_ENTRY.encode("ascii") + b"\x00")
    arguments = [
        "--kernel-binary",
        str(binary),
        "--build-dir",
        str(directory),
        "--cann",
        str(cann),
        "--driver",
        str(driver),
        "--target-arch",
        "x86_64",
        "--toolchain-file",
        str(toolchain),
        "--torch-cmake",
        str(torch_cmake),
        "--torch-npu-path",
        str(torch_npu),
    ]
    return SimpleNamespace(
        arguments=arguments,
        directory=directory,
        cann=cann,
        driver=driver,
        torch=torch,
        torch_cmake=torch_cmake,
        torch_npu=torch_npu,
        toolchain=toolchain,
        libraries=libraries,
    )


def mock_compile(monkeypatch, inputs, architecture="x86_64"):
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        assert command[0] == "cmake"
        if "--build" in command:
            write_target_elf(inputs.directory / build.LIBRARY_NAME, architecture)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(build.subprocess, "run", run)
    return commands


def test_cross_build_does_not_import_host_torch_or_load_target_library(cross_inputs, monkeypatch, capsys):
    inputs = cross_inputs
    original_import = builtins.__import__

    def import_without_torch(name, *args, **kwargs):
        assert name.split(".", 1)[0] not in ("torch", "torch_npu"), "cross-build imported host Torch"
        return original_import(name, *args, **kwargs)

    def reject_host_query(*args, **kwargs):
        pytest.fail("cross-build must not inspect installed Torch or load the x86 library on ARM")

    monkeypatch.setattr(builtins, "__import__", import_without_torch)
    monkeypatch.setattr(build.importlib.util, "find_spec", reject_host_query)
    monkeypatch.setattr(build, "verify_library_schemas", reject_host_query)
    commands = mock_compile(monkeypatch, inputs)
    assert build.main(inputs.arguments) == 0
    assert len(commands) == 2
    manifest = json.loads((inputs.directory / "build-manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "built_pending_target_validation"
    for claim in (
        "host_schema_verified",
        "graph_api_available",
        "device_execution_verified",
        "graph_execution_verified",
    ):
        assert manifest[claim] is False
    assert (inputs.directory / build.KERNEL_NAME).is_file()
    assert not (inputs.directory / "schema.log").exists()
    assert "CROSS_COMPILED" in capsys.readouterr().out


@pytest.mark.parametrize("missing", ["--toolchain-file", "--torch-cmake", "--torch-npu-path"])
def test_cross_build_requires_explicit_target_dependencies(cross_inputs, missing, capsys):
    arguments = cross_inputs.arguments.copy()
    index = arguments.index(missing)
    del arguments[index : index + 2]
    assert build.main(arguments) == 1
    assert missing in capsys.readouterr().out
    assert not (cross_inputs.directory / build.LIBRARY_NAME).exists()


def test_cross_toolchain_cannot_implicitly_select_native_architecture(cross_inputs, capsys):
    arguments = cross_inputs.arguments.copy()
    arguments[arguments.index("--target-arch") + 1] = "native"
    assert build.main(arguments) == 1
    assert "target-arch" in capsys.readouterr().out


@pytest.mark.parametrize("target_path", ["--torch-cmake", "--torch-npu-path"])
def test_native_build_rejects_partial_target_configuration(cross_inputs, capsys, target_path):
    inputs = cross_inputs
    arguments = [
        "--kernel-binary",
        inputs.arguments[1],
        "--build-dir",
        str(inputs.directory),
        target_path,
        inputs.arguments[inputs.arguments.index(target_path) + 1],
    ]
    assert build.main(arguments) == 1
    assert "toolchain" in capsys.readouterr().out


def test_cross_dry_run_preserves_target_and_does_not_compile_or_write(cross_inputs, monkeypatch, capsys):
    inputs = cross_inputs
    commands = mock_compile(monkeypatch, inputs)
    assert build.main([*inputs.arguments, "--dry-run"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["host_architecture"] == "aarch64"
    assert report["target_architecture"] == "x86_64"
    assert report["cross_compiling"] is True
    assert report["host_schema_verified"] is False
    assert f"-DCMAKE_TOOLCHAIN_FILE={inputs.toolchain.resolve()}" in report["commands"][0]
    assert f"-DTorch_DIR={inputs.torch_cmake / 'Torch'}" in report["commands"][0]
    assert f"-DCaffe2_DIR={inputs.torch_cmake / 'Caffe2'}" in report["commands"][0]
    assert commands == []
    assert not inputs.directory.exists()


@pytest.mark.parametrize("library_index", range(6))
def test_cross_build_rejects_arm_dependency_before_cmake(cross_inputs, monkeypatch, capsys, library_index):
    inputs = cross_inputs
    library = inputs.libraries[library_index]
    write_target_elf(library, "aarch64")
    commands = mock_compile(monkeypatch, inputs)
    assert build.main(inputs.arguments) == 1
    output = capsys.readouterr().out
    assert library.name in output and "x86_64" in output
    assert commands == []


@pytest.mark.parametrize("missing", ["TorchConfig.cmake", "libtorch_npu.so", "libruntime.so"])
def test_cross_build_rejects_missing_target_package_files(cross_inputs, monkeypatch, capsys, missing):
    inputs = cross_inputs
    if missing == "TorchConfig.cmake":
        (inputs.torch_cmake / "Torch/TorchConfig.cmake").unlink()
    else:
        next(path for path in inputs.libraries if path.name == missing).unlink()
    commands = mock_compile(monkeypatch, inputs)
    assert build.main(inputs.arguments) == 1
    output = capsys.readouterr().out
    assert missing in output or (missing == "TorchConfig.cmake" and "--torch-cmake" in output)
    assert commands == []


@pytest.mark.parametrize("architecture", ["aarch64", "x86_64"])
def test_host_elf_verification_checks_machine(tmp_path, architecture):
    library = write_target_elf(tmp_path / "adapter.so", architecture)
    assert build.verify_host_elf(library, architecture) == architecture
    other = "aarch64" if architecture == "x86_64" else "x86_64"
    with pytest.raises((ValueError, RuntimeError), match=other):
        build.verify_host_elf(library, other)


@pytest.mark.parametrize("corruption", ["truncated", "elf32", "big_endian", "not_elf", "executable"])
def test_host_elf_verification_rejects_unusable_header(tmp_path, corruption):
    library = write_target_elf(tmp_path / "adapter.so")
    header = bytearray(library.read_bytes())
    if corruption == "truncated":
        header = header[:18]
    elif corruption == "elf32":
        header[4] = 1
    elif corruption == "big_endian":
        header[5] = 2
    elif corruption == "executable":
        header[16:18] = (2).to_bytes(2, "little")
    else:
        header[:4] = b"NOPE"
    library.write_bytes(header)
    with pytest.raises((ValueError, RuntimeError), match="ELF"):
        build.verify_host_elf(library, "x86_64")


def test_cross_build_rejects_wrong_compiler_output(cross_inputs, monkeypatch, capsys):
    mock_compile(monkeypatch, cross_inputs, "aarch64")
    assert build.main(cross_inputs.arguments) == 1
    assert "x86_64" in capsys.readouterr().out
    manifest = json.loads((cross_inputs.directory / "build-manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["host_schema_verified"] is False
    assert not (cross_inputs.directory / build.KERNEL_NAME).exists()


@pytest.mark.parametrize(
    "change",
    [
        "toolchain_contents",
        "toolchain_path",
        "target_architecture",
        "missing_manifest",
        "--torch-cmake",
        "--torch-npu-path",
        "--cann",
        "--driver",
    ],
)
def test_cross_build_rejects_incompatible_existing_cache(cross_inputs, monkeypatch, capsys, change):
    inputs = cross_inputs
    commands = mock_compile(monkeypatch, inputs)
    assert build.main(inputs.arguments) == 0
    cmake_cache = inputs.directory / "cmake-build/CMakeCache.txt"
    cmake_cache.parent.mkdir(exist_ok=True)
    cmake_cache.write_text(
        f"CMAKE_HOME_DIRECTORY:INTERNAL={inputs.directory / build.GENERATED_SOURCE_NAME}\n", encoding="utf-8"
    )
    if change == "toolchain_contents":
        with inputs.toolchain.open("a", encoding="utf-8") as stream:
            stream.write("# changed sysroot/compiler settings\n")
    elif change == "toolchain_path":
        other = inputs.toolchain.with_name("other-toolchain.cmake")
        shutil.copyfile(inputs.toolchain, other)
        inputs.arguments[inputs.arguments.index("--toolchain-file") + 1] = str(other)
    elif change == "target_architecture":
        inputs.arguments[inputs.arguments.index("--target-arch") + 1] = "aarch64"
    elif change == "missing_manifest":
        (inputs.directory / "build-manifest.json").unlink()
    else:
        index = inputs.arguments.index(change) + 1
        original = Path(inputs.arguments[index])
        replacement = original.with_name(original.name + "-other")
        shutil.copytree(original, replacement)
        inputs.arguments[index] = str(replacement)
    commands.clear()
    capsys.readouterr()
    assert build.main(inputs.arguments) == 1
    assert commands == []
    output = capsys.readouterr().out.lower()
    assert any(word in output for word in ("cache", "directory", "toolchain", "architecture"))


def test_cross_build_reuses_matching_cache(cross_inputs, monkeypatch):
    inputs = cross_inputs
    commands = mock_compile(monkeypatch, inputs)
    assert build.main(inputs.arguments) == 0
    cmake_cache = inputs.directory / "cmake-build/CMakeCache.txt"
    cmake_cache.parent.mkdir(exist_ok=True)
    cmake_cache.write_text(
        f"CMAKE_HOME_DIRECTORY:INTERNAL={inputs.directory / build.GENERATED_SOURCE_NAME}\n", encoding="utf-8"
    )
    commands.clear()
    assert build.main(inputs.arguments) == 0
    assert len(commands) == 2


@pytest.mark.parametrize("toolchain_architecture", ["x86_64", "aarch64"])
def test_cross_cmake_configure_enforces_target_and_omits_build_machine_rpath(cross_inputs, toolchain_architecture):
    cmake, ninja = shutil.which("cmake"), shutil.which("ninja")
    if cmake is None or ninja is None:
        pytest.skip("CMake and Ninja are required for configure-only validation")
    inputs = cross_inputs
    inputs.toolchain.write_text(
        inputs.toolchain.read_text(encoding="utf-8").replace(
            "CMAKE_SYSTEM_PROCESSOR x86_64", f"CMAKE_SYSTEM_PROCESSOR {toolchain_architecture}"
        ),
        encoding="utf-8",
    )
    source = inputs.directory / build.GENERATED_SOURCE_NAME
    source.mkdir(parents=True)
    (source / "CMakeLists.txt").write_text(
        build.cmake_project() + '\nfile(WRITE "${CMAKE_BINARY_DIR}/target-arch.txt" "${CMAKE_SYSTEM_PROCESSOR}")\n',
        encoding="utf-8",
    )
    args = build.parse_args(inputs.arguments)
    command = build.cmake_commands(args, inputs.directory, inputs.torch_npu, inputs.torch_cmake)[0]
    command[0] = cmake
    command.extend(["-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={Path(ninja).as_posix()}"])
    result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=60)
    if toolchain_architecture != "x86_64":
        assert result.returncode != 0
        assert "does not match --target-arch" in result.stdout + result.stderr
        return
    assert result.returncode == 0, result.stdout + result.stderr
    cmake_build = inputs.directory / "cmake-build"
    assert (cmake_build / "target-arch.txt").read_text(encoding="utf-8") == "x86_64"
    ninja_lines = (cmake_build / "build.ninja").read_text(encoding="utf-8").splitlines()
    link_libraries = next(line for line in ninja_lines if line.strip().startswith("LINK_LIBRARIES ="))
    assert "-rpath" not in link_libraries
    assert "aarch64-linux" not in link_libraries
