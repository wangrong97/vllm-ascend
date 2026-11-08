# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU/source contracts for the GS32 complete FFN adapter."""

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


def test_binding_exposes_only_complete_ffn_and_device_routes():
    source = source_text()
    assert '#include "../mixed_w2w4_moe_ffn/tiling_builders.h"' in source
    assert '#include "../mixed_w2w4_moe_ffn/mixed_tiling.h"' in source
    assert "constexpr uint32_t kKernelCoreCount = 32;" in source
    assert "static_assert(kMixedTilingVersion == 4 && kMixedTilingBytes == 5728" in source
    registrations = source[source.index("TORCH_LIBRARY(vq2a8_gs32_stage,") :]
    expected = {
        "initialize",
        "_sanitize_device_counts",
        "prepare_device_routes",
        "device_routes_status",
        "full_ffn_abi_version",
        "prepare_full_ffn",
        "full_ffn_device",
    }
    assert set(re.findall(r'm\.def\("(\w+)\(', registrations)) == expected
    assert dict(build.REQUIRED_FULL_FFN_SCHEMAS).keys() == expected
    assert re.findall(r'm\.class_<.*?>\("(\w+)"\)', registrations) == ["DeviceRoutes", "FullFFNPlan"]
    assert set(re.findall(r'm\.impl\("(\w+)"', registrations)) == {
        "prepare_full_ffn",
        "full_ffn_device",
        "prepare_device_routes",
    }
    for removed in ("ProjectionPlan", "GroupedRoutes", "GroupedProjectionPlan", "ProjectGrouped", "ProjectPrepared"):
        assert removed not in source


def test_full_ffn_tiling_preserves_all_component_offsets_and_post_options():
    body = function(source_text(), "MakeFullFFNTiling")
    assert "kMixedTilingBytes" in body
    for stage in ("GateW2", "GateW4", "DownW2", "DownW4"):
        assert f"copy(kMixed{stage}TilingOffset, kMixed{stage}TilingBytes, Build{stage}TilingBlob(" in body
    assert "const MixedPostOptions post{static_cast<float>(limit), {}}" in body
    assert "std::memcpy(bytes + kMixedPostOptionsOffset, &post, sizeof(post))" in body
    mixed = (MIXED / "mixed_tiling.h").read_text(encoding="utf-8")
    assert "constexpr uint64_t kMixedPostOptionsOffset = 1600;" in mixed
    assert "constexpr uint64_t kMixedPostOptionsBytes = 32;" in mixed
    assert "constexpr uint64_t kMixedTilingVersion = 4;" in mixed
    assert "constexpr uint64_t kMixedDownW2TilingOffset = 832;" in mixed


def test_resident_expert_views_are_aligned_and_storage_bounded():
    checks = function(source_text(), "CheckNpuTensor")
    assert "storage_offset() == 0" not in checks
    assert "tensor.storage().nbytes()" in checks
    assert "tensor.nbytes()" in checks
    assert "% kRequiredAlignment == 0" in checks
    assert "tensor.storage().data_ptr()" in function(source_text(), "RecordStream")


def test_full_ffn_is_fail_closed_on_geometry_dtype_layout_alias_and_metadata():
    source = source_text()
    body = function(source, "FullFFNDevice")
    resident = function(source, "ResidentWeights")
    for dtype in ("at::kByte", "at::kBFloat16"):
        assert dtype in body
    assert "at::kLong" in resident
    for shape in (
        "input_scale must be [R,K/32]",
        "hidden workspace must be [R,N/2]",
        "hidden scale must be [R,N/64]",
        "output must be [R,D]",
    ):
        assert shape in body
    assert "full FFN mixed weight/table blob sizes do not match resident precision masks" in resident
    assert "Gate row scales must be [G,N]" in resident and "Down row scales must be [G,D]" in resident
    assert "CheckNoAlias(" in resident and "CheckDynamicRanges(" in body
    checks = function(source, "CheckNpuTensor") + function(source, "CheckCpuMetadata")
    assert "is_contiguous()" in checks
    assert "storage_offset() == 0" in checks
    assert "% kRequiredAlignment == 0" in checks
    prepare = function(source, "PrepareFullFFN")
    assert "k % 64 == 0" in prepare and "n % 128 == 0" in prepare and "d % 32 == 0" in prepare
    assert "std::isfinite(swiglu_limit)" in prepare and "swiglu_limit >= 0" in prepare
    assert "CheckCpuMetadata(gate_w4, at::kBool" in prepare
    assert "CheckCpuMetadata(down_w4, at::kBool" in prepare


def test_prepared_plan_owns_weight_storage_and_preserves_dynamic_alias_guards():
    source = source_text()
    resident = function(source, "ResidentWeights")
    checks = function(source, "CheckDynamicRanges")
    launch = function(source, "FullFFNDevice")
    assert "tensors[i].detach()" in resident
    assert "tensor.storage().nbytes()" in checks
    assert "tensor.data_ptr()) == ranges_[i].begin" in checks
    assert "capacity - offset" in checks and "must not alias or overlap storage" in checks
    assert "const ResidentWeights weights_;" in source
    assert "BlobBytes()" not in launch
    assert "const auto& weights = plan->Weights()" in launch
    # All six static tensors are kept alive by the captured plan, and recorded
    # on the submission stream alongside the dynamic buffers.
    assert all(f"weights[{index}]" in launch for index in range(6))


def test_resident_metadata_is_prepared_before_capture_and_owned_for_the_launch():
    source = source_text()
    prepare = function(source, "PrepareFullFFN")
    assert prepare.index("CheckNotCapturing(") < prepare.index("gate_mask.to(")
    assert "gate_mask.to(anchor.device(), at::kInt, false, true)" in prepare
    assert "down_mask.to(anchor.device(), at::kInt, false, true)" in prepare
    assert "tiling.to(anchor.device(), at::kByte, false, true)" in prepare
    body = function(source, "FullFFNDevice")
    assert "plan->GateMask(), plan->DownMask(), plan->Tiling()" in body
    assert "routes->GroupList(), routes->Status()" in body
    before_queue, callback = body.split("at_npu::native::OpCommand::RunOpApi(", 1)
    assert "const auto launch_stream = stream.stream(false);" in before_queue
    assert "getCurrentNPUStream" not in callback and ".stream(" not in callback
    assert "[routes, plan, launch_stream" in callback
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


def test_torch_schema_marks_all_full_ffn_workspaces_mutable():
    source = source_text()
    assert 'm.def("initialize(str binary) -> ()"' in source
    assert "Tensor(a!) hidden" in source
    assert "Tensor(b!) hidden_scale, Tensor(c!) output) -> ()" in source
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
        _C=SimpleNamespace(
            _dispatch_find_schema_or_throw=find_schema,
            _dispatch_has_kernel_for_dispatch_key=lambda *args: True,
        ),
        ops=SimpleNamespace(
            vq2a8_gs32_stage=SimpleNamespace(
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
            build.verify_runtime_artifacts(library, binary)
    else:
        result = build.verify_runtime_artifacts(library, binary)
        assert result["library_sha256"] == manifest["library_sha256"]
        assert result["kernel"]["sha256"] == manifest["deployed_kernel_sha256"]
        assert result["abi_version"] == build.FULL_FFN_ABI_VERSION


def test_full_ffn_artifact_verification_requires_manifest_by_default(tmp_path):
    library = write_elf(tmp_path / build.LIBRARY_NAME)
    binary = write_elf(tmp_path / build.KERNEL_NAME)
    with pytest.raises(ValueError, match="build-manifest"):
        build.verify_runtime_artifacts(library, binary)
    with pytest.raises(ValueError, match="Only full FFN"):
        build.verify_runtime_artifacts(library, binary, require_full_ffn=False)


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
    "fault", [None, "missing_routes", "missing_class", "wrong_class", "old_padding_schema", "immutable_output"]
)
def test_build_schema_check_requires_typed_device_routes_and_mutable_output(fault):
    schemas = dict(build.REQUIRED_FULL_FFN_SCHEMAS)
    if fault == "missing_routes":
        del schemas["prepare_device_routes"]
    elif fault == "wrong_class":
        schemas["full_ffn_device"] = schemas["full_ffn_device"].replace(".DeviceRoutes ", ".GroupedRoutes ")
    elif fault == "old_padding_schema":
        schemas["prepare_device_routes"] = schemas["prepare_device_routes"].replace(", bool allow_padding=False", "")
    elif fault == "immutable_output":
        schemas["full_ffn_device"] = schemas["full_ffn_device"].replace("Tensor(c!)", "Tensor")

    def find_schema(name, overload):
        assert overload == ""
        key = name.removeprefix("vq2a8_gs32_stage::")
        if key not in schemas:
            raise RuntimeError("missing op")
        return SimpleNamespace(schema=lambda: schemas[key])

    classes = {"FullFFNPlan": object()}
    if fault != "missing_class":
        classes["DeviceRoutes"] = object()
    fake_torch = SimpleNamespace(
        classes=SimpleNamespace(vq2a8_gs32_stage=SimpleNamespace(**classes)),
        _C=SimpleNamespace(
            _dispatch_find_schema_or_throw=find_schema,
            _dispatch_has_kernel_for_dispatch_key=lambda *args: True,
        ),
        ops=SimpleNamespace(vq2a8_gs32_stage=SimpleNamespace(full_ffn_abi_version=lambda: build.FULL_FFN_ABI_VERSION)),
    )
    if fault:
        with pytest.raises(RuntimeError, match="schema|DeviceRoutes"):
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
    assert "torch.ops.load_library(sys.argv[1])" in command[2]
    assert "verify_registered_device_schemas(torch)" in command[2]
    assert "register_counts_implementation()" in command[2]
    assert command[2].index("torch.ops.load_library(") < command[2].index("register_counts_implementation()")
    assert "verify_loaded_library" not in command[2]
    for forbidden in ("initialize(", "set_device(", "npu.synchronize(", "NPUStream("):
        assert forbidden not in command[2]
    assert command[-1] == str(library.resolve())
    assert options["timeout"] == build.SCHEMA_CHECK_TIMEOUT_SECONDS
    assert options["stderr"] is subprocess.STDOUT and options["check"] is False
    assert log.read_text() == "schema child output\n"


@pytest.mark.parametrize("execution", ["fused", "full_graph"])
def test_build_load_check_validates_artifacts_before_loading_and_queries_only_host_registration(
    tmp_path, monkeypatch, execution
):
    library = tmp_path / "adapter.so"
    library.write_bytes(b"extension")
    binary = tmp_path / "mixed.o"
    calls = []
    fake_torch = SimpleNamespace(ops=SimpleNamespace(load_library=lambda path: calls.append(("load", path))))
    monkeypatch.setitem(build.sys.modules, "torch", fake_torch)
    monkeypatch.setitem(build.sys.modules, "torch_npu", SimpleNamespace())
    monkeypatch.setattr(build, "verify_runtime_artifacts", lambda lib, obj: calls.append(("verify", lib, obj)))
    monkeypatch.setattr(build, "register_counts_implementation", lambda: calls.append(("register",)))
    monkeypatch.setattr(
        build,
        "verify_registered_device_schemas",
        lambda module, mode, fusion: {"checked": module is fake_torch, "mode": mode, "fusion": fusion},
    )
    assert build.verify_loaded_library(library, execution, binary=binary) == {
        "checked": True,
        "mode": execution,
        "fusion": "full_ffn",
    }
    assert calls == [("verify", library, binary), ("load", str(library.resolve())), ("register",)]


@pytest.mark.parametrize(
    "execution,fusion",
    [("serial", "full_ffn"), ("batched", "full_ffn"), ("fused", "none"), ("full_graph", "swiglu_quant")],
)
def test_preflight_rejects_removed_modes_before_loading_torch(execution, fusion):
    with pytest.raises(ValueError, match="Unsupported GS32"):
        build.verify_loaded_library("unused.so", execution, fusion, binary="unused.o")
    with pytest.raises(ValueError, match="Unsupported GS32"):
        build.verify_registered_device_schemas(None, execution, fusion)


def test_default_preflight_requires_abi_v4_kernel_and_manifest_before_loading(tmp_path, monkeypatch):
    library = write_elf(tmp_path / build.LIBRARY_NAME)
    binary = tmp_path / build.KERNEL_NAME
    binary.write_bytes(b"\x7fELFmixed_w2w4_moe_ffn_v3\x00")

    def reject_load(_):
        pytest.fail("an invalid artifact pair must never be loaded")

    monkeypatch.setitem(build.sys.modules, "torch", SimpleNamespace(ops=SimpleNamespace(load_library=reject_load)))
    monkeypatch.setitem(build.sys.modules, "torch_npu", SimpleNamespace())
    with pytest.raises(ValueError, match="kernel binary path"):
        build.verify_loaded_library(library)
    with pytest.raises(ValueError, match="ABI"):
        build.verify_loaded_library(library, binary=binary)
    write_elf(binary)
    with pytest.raises(ValueError, match="build-manifest"):
        build.verify_loaded_library(library, binary=binary)


@pytest.mark.parametrize("execution", ["fused", "full_graph"])
@pytest.mark.parametrize("legacy_exports", [False, True])
def test_schema_check_requires_only_full_ffn_ops_and_ignores_extra_exports(execution, legacy_exports):
    required = dict(build.REQUIRED_FULL_FFN_SCHEMAS)
    schemas = dict(required)
    if legacy_exports:
        schemas["project"] = "legacy schema"
        schemas["supports_dynamic_device_rows"] = "legacy schema"
    queried = []

    def find_schema(name, overload):
        key = name.removeprefix("vq2a8_gs32_stage::")
        queried.append(key)
        return SimpleNamespace(schema=lambda: schemas[key])

    fake_torch = SimpleNamespace(
        classes=SimpleNamespace(vq2a8_gs32_stage=SimpleNamespace(DeviceRoutes=object(), FullFFNPlan=object())),
        _C=SimpleNamespace(
            _dispatch_find_schema_or_throw=find_schema,
            _dispatch_has_kernel_for_dispatch_key=lambda *args: True,
        ),
        ops=SimpleNamespace(vq2a8_gs32_stage=SimpleNamespace(full_ffn_abi_version=lambda: build.FULL_FFN_ABI_VERSION)),
    )
    assert build.verify_registered_device_schemas(fake_torch, execution) == required
    assert queried == list(required)


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


def test_route_sanitization_preserves_opaque_ownership_and_rejects_missing_registration():
    source = source_text()
    body = function(source, "SanitizeDeviceCounts")
    assert "findSchemaOrThrow(" in body and "hasKernelForDispatchKey(c10::DispatchKey::PrivateUse1)" in body
    assert "_sanitize_device_counts" in body
    assert 'CheckNpuTensor(safe_counts, counts, at::kLong, 1, "sanitized_counts")' in body
    assert 'CheckNpuTensor(status, counts, at::kInt, 1, "count_status")' in body
    assert "safe_counts.numel() == counts.numel() && status.numel() == 1" in body
    assert (
        'CheckNoAlias({{"input_counts", counts}, {"sanitized_counts", safe_counts}, {"count_status", status}})' in body
    )
    for removed in ("counts.clone()", ".clamp(", ".sum()", ".cpu(", ".item("):
        assert removed not in body
    prepare = function(source, "PrepareDeviceRoutes")
    assert "RecordStream({counts, sanitized.first, sanitized.second}, stream)" in prepare
    assert "std::move(sanitized.first), std::move(sanitized.second), stream.stream(false)" in prepare
    assert "routes->Status().clone()" in function(source, "DeviceRoutesStatus")


@pytest.mark.parametrize("fault", ("missing_schema", "missing_backend"))
def test_preflight_requires_new_bridge_sanitizer_schema_and_device_implementation(fault):
    schemas = dict(build.REQUIRED_FULL_FFN_SCHEMAS)
    if fault == "missing_schema":
        del schemas["_sanitize_device_counts"]

    def find_schema(name, overload):
        key = name.removeprefix("vq2a8_gs32_stage::")
        if key not in schemas:
            raise RuntimeError("missing operator")
        return SimpleNamespace(schema=lambda: schemas[key])

    fake_torch = SimpleNamespace(
        classes=SimpleNamespace(vq2a8_gs32_stage=SimpleNamespace(DeviceRoutes=object(), FullFFNPlan=object())),
        _C=SimpleNamespace(
            _dispatch_find_schema_or_throw=find_schema,
            _dispatch_has_kernel_for_dispatch_key=lambda *args: False,
        ),
        ops=SimpleNamespace(vq2a8_gs32_stage=SimpleNamespace(full_ffn_abi_version=lambda: build.FULL_FFN_ABI_VERSION)),
    )
    with pytest.raises(RuntimeError, match="sanitize|sanitizer"):
        build.verify_registered_device_schemas(fake_torch)
