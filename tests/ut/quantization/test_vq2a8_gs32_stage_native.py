# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU/source contracts for the GS32 projection and complete FFN adapter."""

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import regex as re

from tools import build_vq2a8_gs32_stage as build

REPO = Path(__file__).resolve().parents[3]
SOURCE = REPO / "csrc/vq2a8_gs32_stage/torch_binding.cpp"
MIXED = REPO / "csrc/mixed_w2w4_moe_ffn"


def source_text():
    return SOURCE.read_text(encoding="utf-8")


def function(source, name):
    marker = f"{name}("
    start = source.index(marker)
    begin = source.index("{", start)
    depth = 1
    end = begin + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[begin:end]


def write_elf(path, architecture="aarch64"):
    header = bytearray(64)
    header[:7] = b"\x7fELF\x02\x01\x01"
    header[16:18] = (3).to_bytes(2, "little")
    header[18:20] = {"aarch64": 183, "x86_64": 62}[architecture].to_bytes(2, "little")
    header.extend(build.KERNEL_ENTRY.encode("ascii") + b"\x00")
    path.write_bytes(header)
    return path


def test_binding_preserves_the_existing_downw2_stage_contract():
    source = source_text()
    assert '#include "../mixed_w2w4_moe_ffn/tiling_builders.h"' in source
    assert '#include "../mixed_w2w4_moe_ffn/mixed_tiling.h"' in source
    assert "constexpr uint64_t kGroupCount = 1;" in source
    assert "constexpr uint32_t kKernelCoreCount = 32;" in source
    assert "static_assert(kMixedStageDownW2 == 4" in source
    assert "static_cast<uint64_t>(2 * k)" in function(source, "Project")
    assert "static_cast<uint64_t>(n)" in function(source, "Project")
    assert "rtKernelLaunch(registered->stub, kKernelCoreCount" in function(source, "Project")


def test_make_tiling_preserves_component_offsets_with_versioned_post_options():
    source = source_text()
    body = function(source, "MakeTiling")
    assert "BuildDownW2TilingBlob" in body
    assert "kMixedTilingBytes" in body
    assert "kMixedStageDownW2" in body
    assert "kMixedDownW2TilingOffset" in body
    assert "std::memcpy(bytes + kMixedDownW2TilingOffset" in body
    mixed = (MIXED / "mixed_tiling.h").read_text(encoding="utf-8")
    assert "constexpr uint64_t kMixedPostOptionsOffset = 1600;" in mixed
    assert "constexpr uint64_t kMixedPostOptionsBytes = 32;" in mixed
    assert "constexpr uint64_t kMixedTilingVersion = 3;" in mixed
    assert "constexpr uint64_t kMixedDownW2TilingOffset = 832;" in mixed


def test_resident_expert_views_are_aligned_and_storage_bounded():
    checks = function(source_text(), "CheckNpuTensor")
    assert "storage_offset() == 0" not in checks
    assert "tensor.storage().nbytes()" in checks
    assert "tensor.nbytes()" in checks
    assert "% kRequiredAlignment == 0" in checks
    assert "tensor.storage().data_ptr()" in function(source_text(), "RecordStream")


def test_project_is_fail_closed_on_geometry_dtype_layout_alias_and_metadata():
    body = function(source_text(), "Project")
    for dtype in ("at::kByte", "at::kLong", "at::kInt", "at::kBFloat16"):
        assert dtype in body
    for shape in (
        "input_scale must have shape [M,K/32]",
        "packed_weight must have shape [1,N/32,K/16,16,8]",
        "pair_lut must have shape [1,N/32,K/32,32]",
        "row_scale must have shape [1,N]",
        "group_list must have shape [1] and contain M",
        "route_mask must have shape [1] and contain the all-W2 value zero",
        "output must have shape [M,N]",
    ):
        assert shape in body
    assert "1<=M<=128" in body
    assert "group_list.data_ptr<int64_t>()[0] == m" in body
    assert "route_mask.data_ptr<int32_t>()[0] == 0" in body
    assert "std::memcmp(tiling.data_ptr(), expected_tiling.data_ptr(), kMixedTilingBytes)" in body
    assert "CheckNoAlias(npu_owners)" in body
    assert "CheckNoAlias(cpu_metadata)" in body
    checks = function(source_text(), "CheckNpuTensor") + function(source_text(), "CheckCpuMetadata")
    assert "is_contiguous()" in checks
    assert "storage_offset() == 0" in checks
    assert "% kRequiredAlignment == 0" in checks


def test_cpu_metadata_is_validated_then_copied_and_owned_for_the_queued_launch():
    body = function(source_text(), "Project")
    assert "group_list.to(input_bytes.device(), at::kLong, false, true)" in body
    assert "route_mask.to(input_bytes.device(), at::kInt, false, true)" in body
    assert "tiling.to(input_bytes.device(), at::kByte, false, true)" in body
    assert "device_group_list.data_ptr()" in body
    assert "device_route_mask.data_ptr()" in body
    assert "device_tiling.data_ptr()" in body
    assert "device_group_list, device_route_mask, device_tiling" in body
    assert "RecordStream(" in body


def test_raw_stream_is_resolved_before_queue_callback_and_graph_is_rejected():
    body = function(source_text(), "Project")
    before_queue, callback = body.split("at_npu::native::OpCommand::RunOpApi(", 1)
    assert "CheckNotCapturing(" in before_queue
    assert before_queue.index("OptionalNPUGuard") < before_queue.index("CheckNotCapturing(")
    assert before_queue.index("CheckNotCapturing(") < before_queue.index("CheckNpuTensor(")
    assert before_queue.index("CheckNotCapturing(") < before_queue.index("group_list.to(")
    assert "const auto launch_stream = stream.stream();" in before_queue
    assert "getCurrentNPUStream" not in callback
    assert ".stream()" not in callback
    assert "[registered, launch_stream" in callback
    assert "false);" in callback
    for forbidden in ("synchronize", ".item(", "rtDeviceReset", "aclrtResetDevice"):
        assert forbidden not in body


def test_capture_guard_uses_exported_status_api_without_hidden_context_helpers_or_fallback():
    # Comments may explain the incompatible APIs; executable calls must not use them.
    source = re.sub(r"//[^\n]*|/\*.*?\*/", "", source_text(), flags=re.DOTALL)
    body = function(source, "CheckNotCapturing")
    assert "c10_npu::currentStreamCaptureStatusMayInitCtx()" in body
    assert "TORCH_CHECK(status == c10_npu::CaptureStatus::None" in body
    assert "static_cast<int>(status)" in body
    assert body.index("currentStreamCaptureStatusMayInitCtx()") < body.index("TORCH_CHECK(")
    # None-only acceptance also rejects Invalidated, unlike an Active-only boolean.
    assert not re.search(r"\b(?:if|try|catch|return)\b", body)
    for forbidden in (
        "assertNotCapturing",
        "currentStreamCaptureStatus",
        "IsContextInitialized",
        "isStreamCapturingMayInitCtx",
        "dlsym",
    ):
        assert not re.search(rf"\b{forbidden}\s*\(", source)


def test_registration_is_startup_only_idempotent_and_retains_elf_bytes():
    source = source_text()
    body = function(source, "Initialize")
    assert "CheckNotCapturing(" in body
    assert body.index("CheckNotCapturing(") < body.index("g_registration_mutex")
    assert body.index("CheckNotCapturing(") < body.index("rtDevBinaryRegister")
    assert "g_registration_mutex" in body
    assert "g_registered_kernel->path == binary_path" in body
    assert "ReadElfBinary(binary_path)" in body
    assert "rtDevBinaryRegister" in body
    assert "rtFunctionRegister" in body
    assert "g_registered_kernel = std::move(state)" in body
    assert "rtDeviceReset" not in source
    assert "rtStreamSynchronize" not in source
    assert "#if __has_include(<runtime/rt.h>)" in source


def test_torch_schema_is_exact_and_output_is_mutable():
    source = source_text()
    assert 'm.def("initialize(str binary) -> ()"' in source
    assert 'm.def("make_tiling(int k, int n) -> Tensor"' in source
    assert "Tensor(a!) output) -> ()" in source
    assert "TORCH_LIBRARY_IMPL(vq2a8_gs32_stage, PrivateUse1" in source


def test_complete_ffn_launch_keeps_device_routes_and_workspace_owners_on_the_stream():
    body = function(source_text(), "FullFFNDevice")
    assert "CheckCaptureAllowed(" in body
    assert "routes->CheckMatches(input_bytes, plan->Groups())" in body
    assert "output.zero_()" in body and "hidden_scale.fill_(127)" in body
    assert "RecordStream(" in body and "routes->Status()" in body
    assert "const auto launch_stream = stream.stream(false)" in body
    assert body.count("rtKernelLaunch(") == 1
    assert "[routes, plan, launch_stream" in body
    for forbidden in (".cpu(", ".item(", "rtMemcpy", "Synchronize", "CheckNotCapturing", "kMaxProjectionRows"):
        assert forbidden not in body
    prepare = function(source_text(), "PrepareFullFFN")
    assert "CheckNotCapturing(" in prepare
    assert "!gate_is_w4 || down_is_w4" in prepare
    assert "stages |= gate_is_w4" in prepare and "stages |= down_is_w4" in prepare
    assert "group_list" not in prepare


@pytest.mark.parametrize("fault", [None, "old_version", "missing_launch", "immutable_workspace", "missing_class"])
def test_full_ffn_preflight_rejects_stale_bridge_and_workspace_schema(fault):
    schemas = dict(build.REQUIRED_FULL_FFN_SCHEMAS)
    if fault == "missing_launch":
        schemas.pop("full_ffn_device")
    if fault == "immutable_workspace":
        schemas["full_ffn_device"] = schemas["full_ffn_device"].replace("Tensor(b!)", "Tensor")

    def find_schema(name, overload):
        key = name.removeprefix("vq2a8_gs32_stage::")
        if key not in schemas:
            raise RuntimeError("missing operator")
        return SimpleNamespace(schema=lambda: schemas[key])

    classes = {"DeviceRoutes": object()}
    if fault != "missing_class":
        classes["FullFFNPlan"] = object()
    fake_torch = SimpleNamespace(
        classes=SimpleNamespace(vq2a8_gs32_stage=SimpleNamespace(**classes)),
        _C=SimpleNamespace(_dispatch_find_schema_or_throw=find_schema),
        ops=SimpleNamespace(
            vq2a8_gs32_stage=SimpleNamespace(
                supports_dynamic_grouped_rows=lambda: True,
                supports_dynamic_device_rows=lambda: True,
                supports_sparse_prefill_output=lambda: True,
                full_ffn_abi_version=lambda: 2 if fault == "old_version" else build.FULL_FFN_ABI_VERSION,
            )
        ),
    )
    if fault:
        with pytest.raises(RuntimeError, match="ABI|schema|FullFFNPlan"):
            build.verify_registered_device_schemas(fake_torch, "fused", "full_ffn")
    else:
        assert build.verify_registered_device_schemas(fake_torch, "fused", "full_ffn") == schemas


@pytest.mark.parametrize("fault", [None, "old_kernel", "changed_library", "changed_kernel", "unfinished", "abi"])
def test_full_ffn_artifacts_must_match_completed_build(tmp_path, fault):
    library = write_elf(tmp_path / build.LIBRARY_NAME)
    binary = write_elf(tmp_path / build.KERNEL_NAME)
    manifest = {
        "status": "built",
        "abi_version": build.FULL_FFN_ABI_VERSION,
        "kernel_abi_version": build.FULL_FFN_ABI_VERSION,
        "library_sha256": build.sha256(library),
        "deployed_kernel_sha256": build.sha256(binary),
    }
    if fault == "old_kernel":
        binary.write_bytes(b"\x7fELF" + b"old-kernel")
    elif fault == "changed_library":
        library.write_bytes(library.read_bytes() + b"changed")
    elif fault == "changed_kernel":
        binary.write_bytes(binary.read_bytes() + b"changed")
    elif fault == "unfinished":
        manifest["status"] = "building"
    elif fault == "abi":
        manifest["kernel_abi_version"] = 2
    (tmp_path / "build-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    if fault:
        with pytest.raises(ValueError, match="ABI|manifest"):
            build.verify_runtime_artifacts(library, binary, require_full_ffn=True)
    else:
        result = build.verify_runtime_artifacts(library, binary, require_full_ffn=True)
        assert result["library_sha256"] == manifest["library_sha256"]
        assert result["kernel"]["sha256"] == manifest["deployed_kernel_sha256"]
        assert result["abi_version"] == build.FULL_FFN_ABI_VERSION


def test_full_ffn_requires_manifest_without_affecting_explicit_legacy_preflight(tmp_path):
    library = write_elf(tmp_path / build.LIBRARY_NAME)
    binary = write_elf(tmp_path / build.KERNEL_NAME)
    with pytest.raises(ValueError, match="build-manifest"):
        build.verify_runtime_artifacts(library, binary, require_full_ffn=True)
    assert build.verify_runtime_artifacts(library, binary)["kernel"]["present"]


def test_build_tool_validates_existing_mixed_elf_and_is_host_only(tmp_path):
    kernel = write_elf(tmp_path / "mixed.o")
    evidence = build.kernel_evidence(kernel)
    assert evidence["present"] is True
    assert evidence["bytes"] == kernel.stat().st_size
    assert len(evidence["sha256"]) == 64
    project = build.cmake_project()
    assert "add_library(vq2a8_gs32_stage SHARED" in project
    assert "ascendc_library" not in project
    assert "torch_npu ascendcl runtime" in project
    assert 'target_link_options(vq2a8_gs32_stage PRIVATE "LINKER:-z,defs")' in project
    assert "set(SYSTEM_PREFIX ${CMAKE_SYSTEM_PROCESSOR}-linux)" in project
    assert '"${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/devlib"' in project
    assert "aarch64-linux/devlib" not in project


@pytest.mark.parametrize("host_architecture", ["aarch64", "x86_64"])
def test_cmake_configure_selects_native_link_directory_without_stub_rpath(tmp_path, host_architecture):
    cmake, ninja = shutil.which("cmake"), shutil.which("ninja")
    if cmake is None or ninja is None:
        pytest.skip("CMake and Ninja are required for configure-only validation")
    cann, driver, torch_npu = tmp_path / "cann", tmp_path / "driver", tmp_path / "torch_npu"
    torch_cmake, directory = tmp_path / "torch_cmake", tmp_path / "output"
    for path in (cann, driver, torch_npu, torch_cmake):
        path.mkdir()
    (torch_cmake / "TorchConfig.cmake").write_text("set(Torch_FOUND TRUE)\n", encoding="utf-8")
    # CMake evaluates the real generated project, but this toolchain deliberately
    # skips compiler probing: the test validates paths, not a CANN/Torch build.
    toolchain = tmp_path / "toolchain.cmake"
    toolchain.write_text(
        f"set(CMAKE_SYSTEM_NAME Linux)\n"
        f"set(CMAKE_SYSTEM_PROCESSOR {host_architecture})\n"
        f'set(CMAKE_CXX_COMPILER "{Path(build.sys.executable).as_posix()}")\n'
        "set(CMAKE_CXX_COMPILER_ID GNU)\n"
        "set(CMAKE_CXX_COMPILER_ID_RUN TRUE)\n"
        "set(CMAKE_CXX_COMPILER_FORCED TRUE)\n"
        "set(CMAKE_CXX_COMPILER_WORKS TRUE)\n",
        encoding="utf-8",
    )
    source = directory / build.GENERATED_SOURCE_NAME
    source.mkdir(parents=True)
    (source / "CMakeLists.txt").write_text(
        build.cmake_project() + "\nget_target_property(link_dirs vq2a8_gs32_stage LINK_DIRECTORIES)\n"
        "get_target_property(build_rpath vq2a8_gs32_stage INSTALL_RPATH)\n"
        'file(WRITE "${CMAKE_BINARY_DIR}/link-dirs.txt" "${link_dirs}")\n'
        'file(WRITE "${CMAKE_BINARY_DIR}/build-rpath.txt" "${build_rpath}")\n',
        encoding="utf-8",
    )
    args = build.parse_args(["--kernel-binary", "unused.o", "--cann", str(cann), "--driver", str(driver)])
    command = build.cmake_commands(args, directory, torch_npu.as_posix(), torch_cmake.as_posix())[0]
    command[0] = cmake
    command.extend(
        [
            "-G",
            "Ninja",
            f"-DCMAKE_MAKE_PROGRAM={Path(ninja).as_posix()}",
            f"-DCMAKE_TOOLCHAIN_FILE={toolchain.as_posix()}",
        ]
    )
    result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    cmake_build = directory / "cmake-build"
    link_dirs = (cmake_build / "link-dirs.txt").read_text(encoding="utf-8").split(";")
    rpath = [Path(path).as_posix() for path in (cmake_build / "build-rpath.txt").read_text(encoding="utf-8").split(";")]
    assert f"{cann.as_posix()}/{host_architecture}-linux/devlib" in link_dirs
    other_architecture = "x86_64" if host_architecture == "aarch64" else "aarch64"
    assert not any(f"/{other_architecture}-linux/" in path for path in link_dirs)
    for runtime_path in (torch_npu / "lib", cann / "lib64", driver / "lib64/common", driver / "lib64/driver"):
        assert runtime_path.as_posix() in rpath
    assert all("devlib" not in path for path in rpath)
    # Checking the explicit RPATH alone misses CMake's automatically appended
    # link directories. The actual linker command must also exclude CANN stubs.
    ninja_lines = (cmake_build / "build.ninja").read_text(encoding="utf-8").splitlines()
    link_libraries = next(line for line in ninja_lines if line.strip().startswith("LINK_LIBRARIES ="))
    link_path = next(line for line in ninja_lines if line.strip().startswith("LINK_PATH ="))
    assert f"{cann.as_posix()}/{host_architecture}-linux/devlib" in link_path
    assert "-rpath" in link_libraries
    assert "devlib" not in link_libraries


def test_build_never_writes_inside_the_imported_kernel_directory():
    args = build.parse_args(["--kernel-binary", "unused.o", "--build-dir", str(MIXED / "build"), "--dry-run"])
    with pytest.raises(ValueError, match="out-of-source"):
        build.validate_options(args)


def test_build_dry_run_never_claims_device_or_graph_acceptance(tmp_path, capsys):
    kernel = write_elf(tmp_path / "mixed.o")
    result = build.main(
        [
            "--kernel-binary",
            str(kernel),
            "--build-dir",
            str(tmp_path / "out"),
            "--dry-run",
        ]
    )
    assert result == 0
    output = capsys.readouterr().out
    assert '"host_extension_only": true' in output
    assert '"kernel_is_built_by_this_tool": false' in output
    assert '"device_execution_verified": false' in output
    assert '"host_schema_verified": false' in output
    assert '"graph_api_available": false' in output
    assert '"graph_execution_verified": false' in output
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("payload", [b"", b"NOPE", b"\x7fEL"])
def test_build_rejects_missing_or_non_elf_kernel(tmp_path, payload):
    kernel = tmp_path / "mixed.o"
    kernel.write_bytes(payload)
    with pytest.raises(ValueError, match="ELF"):
        build.kernel_evidence(kernel)


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "missing_op",
        "missing_prepare_grouped",
        "wrong_class",
        "wrong_output_alias",
        "missing_class",
        "old_padding_schema",
        "missing_dynamic_grouped_rows",
        "wrong_dynamic_grouped_rows_schema",
        "unsupported_dynamic_grouped_rows",
        "missing_dynamic_device_rows",
        "wrong_dynamic_device_rows_schema",
        "unsupported_dynamic_device_rows",
        "missing_sparse_prefill_output",
        "unsupported_sparse_prefill_output",
        "old_output_fill_schema",
        "unsafe_output_fill_default",
    ],
)
def test_build_schema_check_requires_typed_device_routes_and_mutable_output(fault):
    schemas = dict(build.REQUIRED_DEVICE_SCHEMAS)
    if fault == "missing_op":
        del schemas["prepare_device_routes"]
    elif fault == "missing_prepare_grouped":
        del schemas["prepare_grouped"]
    elif fault == "wrong_class":
        schemas["project_grouped_device"] = schemas["project_grouped_device"].replace(
            ".DeviceRoutes ", ".GroupedRoutes "
        )
    elif fault == "wrong_output_alias":
        schemas["project_grouped_device"] = schemas["project_grouped_device"].replace("Tensor(a!)", "Tensor")
    elif fault == "old_padding_schema":
        schemas["prepare_device_routes"] = schemas["prepare_device_routes"].replace(", bool allow_padding=False", "")
    elif fault == "missing_dynamic_grouped_rows":
        del schemas["supports_dynamic_grouped_rows"]
        schemas["grouped_max_routed_rows"] = "vq2a8_gs32_stage::grouped_max_routed_rows() -> int"
    elif fault == "wrong_dynamic_grouped_rows_schema":
        schemas["supports_dynamic_grouped_rows"] = schemas["supports_dynamic_grouped_rows"].replace(
            "-> bool", "-> Tensor"
        )
    elif fault == "missing_dynamic_device_rows":
        del schemas["supports_dynamic_device_rows"]
    elif fault == "wrong_dynamic_device_rows_schema":
        schemas["supports_dynamic_device_rows"] = schemas["supports_dynamic_device_rows"].replace("-> bool", "-> int")
    elif fault == "missing_sparse_prefill_output":
        del schemas["supports_sparse_prefill_output"]
    elif fault == "old_output_fill_schema":
        schemas["project_grouped_device"] = schemas["project_grouped_device"].replace(", bool zero_output=True", "")
    elif fault == "unsafe_output_fill_default":
        schemas["project_grouped_device"] = schemas["project_grouped_device"].replace(
            "bool zero_output=True", "bool zero_output=False"
        )

    def find_schema(name, overload):
        assert overload == ""
        key = name.removeprefix("vq2a8_gs32_stage::")
        if key not in schemas:
            raise RuntimeError("missing op")
        return SimpleNamespace(schema=lambda: schemas[key])

    classes = {} if fault == "missing_class" else {"DeviceRoutes": object()}
    fake_torch = SimpleNamespace(
        classes=SimpleNamespace(vq2a8_gs32_stage=SimpleNamespace(**classes)),
        _C=SimpleNamespace(_dispatch_find_schema_or_throw=find_schema),
        ops=SimpleNamespace(
            vq2a8_gs32_stage=SimpleNamespace(
                supports_dynamic_grouped_rows=lambda: fault != "unsupported_dynamic_grouped_rows",
                supports_dynamic_device_rows=lambda: fault != "unsupported_dynamic_device_rows",
                supports_sparse_prefill_output=lambda: fault != "unsupported_sparse_prefill_output",
            )
        ),
    )
    if fault:
        with pytest.raises(RuntimeError, match="schema|DeviceRoutes|does not support"):
            build.verify_registered_device_schemas(fake_torch)
    else:
        assert build.verify_registered_device_schemas(fake_torch) == schemas


@pytest.mark.parametrize("exit_code", [0, 7])
def test_build_schema_check_is_a_separate_host_only_process(tmp_path, monkeypatch, exit_code):
    library, log = tmp_path / "adapter.so", tmp_path / "schema.log"
    library.write_bytes(b"extension")
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        kwargs["stdout"].write("schema child output\n")
        return SimpleNamespace(returncode=exit_code)

    monkeypatch.setattr(build.subprocess, "run", run)
    if exit_code:
        with pytest.raises(RuntimeError, match="schema.*7|schema.log"):
            build.verify_library_schemas(library, log)
    else:
        build.verify_library_schemas(library, log)
    command, options = calls[0]
    assert command[0] == build.sys.executable and command[1] == "-c"
    assert "verify_loaded_library" in command[2]
    assert command[-1] == str(library.resolve())
    assert options["timeout"] == build.SCHEMA_CHECK_TIMEOUT_SECONDS
    assert options["stderr"] is subprocess.STDOUT and options["check"] is False
    assert log.read_text() == "schema child output\n"


@pytest.mark.parametrize("execution", ["serial", "batched", "fused", "full_graph"])
def test_build_load_check_only_imports_and_queries_registration(tmp_path, monkeypatch, execution):
    library = tmp_path / "adapter.so"
    library.write_bytes(b"extension")
    calls = []
    fake_torch = SimpleNamespace(ops=SimpleNamespace(load_library=lambda path: calls.append(path)))
    monkeypatch.setitem(build.sys.modules, "torch", fake_torch)
    monkeypatch.setitem(build.sys.modules, "torch_npu", SimpleNamespace())
    monkeypatch.setattr(
        build, "verify_registered_device_schemas", lambda module, mode: {"checked": module is fake_torch, "mode": mode}
    )
    assert build.verify_loaded_library(str(library), execution) == {"checked": True, "mode": execution}
    assert calls == [str(library.resolve())]


@pytest.mark.parametrize("bridge_version,kernel_version", [(None, 2), (3, 2), (3, 3)])
def test_legacy_projection_preflight_requires_kernel_matching_loaded_bridge(
    tmp_path, monkeypatch, bridge_version, kernel_version
):
    library = write_elf(tmp_path / build.LIBRARY_NAME)
    binary = write_elf(tmp_path / build.KERNEL_NAME)
    if kernel_version == 2:
        binary.write_bytes(b"\x7fELFold-kernel")
    namespace = SimpleNamespace()
    if bridge_version is not None:
        namespace.full_ffn_abi_version = lambda: bridge_version
    fake_torch = SimpleNamespace(ops=SimpleNamespace(load_library=lambda _: None, vq2a8_gs32_stage=namespace))
    monkeypatch.setitem(build.sys.modules, "torch", fake_torch)
    monkeypatch.setitem(build.sys.modules, "torch_npu", SimpleNamespace())
    monkeypatch.setattr(build, "verify_registered_device_schemas", lambda *args: {"checked": True})
    if bridge_version == 3 and kernel_version == 2:
        with pytest.raises(ValueError, match="ABI"):
            build.verify_loaded_library(library, "fused", fusion="none", binary=binary)
    else:
        assert build.verify_loaded_library(library, "fused", fusion="none", binary=binary) == {"checked": True}


@pytest.mark.parametrize(
    "execution,required",
    [
        ("serial", build.REQUIRED_SERIAL_SCHEMAS),
        ("batched", build.REQUIRED_BATCHED_SCHEMAS),
        ("fused", build.REQUIRED_FUSED_SCHEMAS),
    ],
)
def test_build_schema_check_only_requires_selected_execution_ops(execution, required):
    schemas = dict(required)
    queried = []

    def find_schema(name, overload):
        key = name.removeprefix("vq2a8_gs32_stage::")
        queried.append(key)
        assert key in schemas, "Lower execution modes must not require graph-only operators."
        return SimpleNamespace(schema=lambda: schemas[key])

    fake_torch = SimpleNamespace(
        _C=SimpleNamespace(_dispatch_find_schema_or_throw=find_schema),
        ops=SimpleNamespace(vq2a8_gs32_stage=SimpleNamespace(supports_dynamic_grouped_rows=lambda: True)),
    )
    assert build.verify_registered_device_schemas(fake_torch, execution) == schemas
    assert queried == list(schemas)


@pytest.mark.parametrize("schema_error", [False, True])
@pytest.mark.parametrize("host_architecture", ["aarch64", "x86_64"])
def test_build_pass_requires_schema_verification_and_does_not_claim_graph_execution(
    tmp_path, monkeypatch, schema_error, host_architecture
):
    kernel = write_elf(tmp_path / "mixed.o")
    cann, driver, out = tmp_path / "cann", tmp_path / "driver", tmp_path / "out"
    (cann / "include").mkdir(parents=True)
    (driver / "lib64/driver").mkdir(parents=True)
    args = build.parse_args(
        ["--kernel-binary", str(kernel), "--cann", str(cann), "--driver", str(driver), "--build-dir", str(out)]
    )
    monkeypatch.setattr(build.platform, "system", lambda: "Linux")
    monkeypatch.setattr(build.platform, "machine", lambda: host_architecture)
    monkeypatch.setattr(
        build.importlib.util, "find_spec", lambda _: SimpleNamespace(submodule_search_locations=[str(tmp_path)])
    )
    fake_torch = SimpleNamespace(__version__="test", utils=SimpleNamespace(cmake_prefix_path=str(tmp_path)))
    monkeypatch.setitem(build.sys.modules, "torch", fake_torch)

    def compile_only(command, **kwargs):
        write_elf(out / build.LIBRARY_NAME, host_architecture)
        return SimpleNamespace(returncode=0)

    verified = []

    def schemas(library, log):
        verified.append((library, log))
        if schema_error:
            raise RuntimeError("schema check failed")

    monkeypatch.setattr(build.subprocess, "run", compile_only)
    monkeypatch.setattr(build, "verify_library_schemas", schemas)
    if schema_error:
        with pytest.raises(RuntimeError, match="schema"):
            build.build(args)
    else:
        assert build.build(args) == 0
    assert len(verified) == 1
    manifest = json.loads((out / "build-manifest.json").read_text())
    assert manifest["host_system"] == "Linux"
    assert manifest["host_architecture"] == host_architecture
    assert manifest["host_schema_verified"] is (not schema_error)
    assert manifest["graph_api_available"] is (not schema_error)
    assert manifest["graph_execution_verified"] is False
    assert manifest["device_execution_verified"] is False
    assert manifest["status"] == ("failed" if schema_error else "built")
