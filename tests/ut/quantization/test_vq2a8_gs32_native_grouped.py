# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-only grouped projection contracts; no NPU numerical claim."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import regex as re

SOURCE = Path(__file__).resolve().parents[3] / "csrc/vq2a8_gs32_stage/torch_binding.cpp"


def body(source, name):
    start = source.index(f"{name}(")
    begin = source.index("{", start)
    end = begin + 1
    depth = 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[begin:end]


def test_grouped_schema_retains_diagnostic_abi_and_opaque_handles():
    source = SOURCE.read_text()
    for name in ("GroupedRoutes", "GroupedProjectionPlan"):
        assert f"class {name} : public torch::CustomClassHolder" in source
        assert f'm.class_<vq2a8_gs32_stage::{name}>("{name}");' in source
    for name in ("prepare_routes", "prepare_grouped", "project_grouped", "project", "prepare", "project_prepared"):
        assert f'm.impl("{name}"' in source


def test_routes_validate_cpu_counts_and_make_one_owned_blocking_upload():
    source = SOURCE.read_text()
    prepare = body(source, "PrepareRoutes")
    assert "CheckNotCapturing(" in prepare
    assert "const auto owned_counts = counts.clone();" in prepare
    assert "ValidateGroupCounts(owned_counts, anchor.size(0))" in prepare
    assert prepare.count(".to(") == 1
    assert "owned_counts.to(anchor.device(), at::kLong, false, true)" in prepare
    checks = body(source, "ValidateGroupCounts")
    assert "CheckCpuMetadata(counts, at::kLong, 1" in checks
    assert "count >= 0" in checks
    assert "kMaxGroupedProjectionRows" not in source
    assert "kMaxGroupedRoutedRows" not in source
    assert "TORCH_CHECK(rows > 0," in checks
    assert "count <= rows - total" in checks
    assert "total == rows" in checks


def test_large_grouped_capability_preserves_serial_limit_and_dynamic_device_capacity():
    source = SOURCE.read_text()
    assert 'm.def("supports_dynamic_grouped_rows() -> bool", &vq2a8_gs32_stage::SupportsDynamicGroupedRows);' in source
    assert "return true;" in body(source, "SupportsDynamicGroupedRows")
    limits = {name: int(value) for name, value in re.findall(r"constexpr int64_t (kMax\w+) = (\d+);", source)}
    assert limits == {
        "kMaxProjectionRows": 128,
        "kMaxGroupedExperts": 256,
    }
    assert "m <= kMaxProjectionRows" in body(source, "ValidateProjectionShape")
    assert "owned_counts.clamp(0, rows)" in body(source, "SanitizeDeviceCounts")
    assert 'm.def("supports_dynamic_device_rows() -> bool", &vq2a8_gs32_stage::SupportsDynamicDeviceRows);' in source


def test_grouped_plan_uses_existing_tiling_and_bitmask_word_count():
    source = SOURCE.read_text()
    prepare = body(source, "PrepareGrouped")
    assert "ValidateGroupCount(groups)" in prepare
    assert "MakeTiling(k, n)" in prepare
    assert "(groups + kRouteMaskWordBits - 1) / kRouteMaskWordBits" in prepare
    assert prepare.count(".to(") == 2
    assert "RegisteredKernelSnapshot()" in prepare


def test_grouped_launch_queues_without_build_copy_or_host_scalar_sync():
    source = SOURCE.read_text()
    project = body(source, "ProjectGrouped")
    assert "CheckGroupedProjectionTensors(" in project
    assert "routes->CheckMatches(input_bytes, plan->Groups())" in project
    assert "plan->CheckMatches(input_bytes, output)" in project
    assert "RecordStream(" in project
    assert "routes->GroupList()" in project
    assert "stream.stream(false)" in project
    assert "[routes, plan, launch_stream" in project
    assert "static_cast<uint64_t>(plan->Groups())" in project
    for forbidden in (".cpu(", ".item(", ".to(", "MakeTiling", "Synchronize", "synchronize", "stream.stream()"):
        assert forbidden not in project
    assert "kKernelCoreCount" in project
    assert "static_cast<uint64_t>(2 * plan->K())" in project


@pytest.fixture(scope="module")
def grouped_harness(tmp_path_factory):
    """Execute actual counts/handle/launch code; only NPU/ATen are doubles."""
    compiler = shutil.which("c++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler is required for grouped host contracts")
    source = SOURCE.read_text()
    routes = source[
        source.index("class GroupedRoutes :") : source.index("c10::intrusive_ptr<GroupedRoutes> PrepareRoutes")
    ]
    plan = source[
        source.index("class GroupedProjectionPlan :") : source.index(
            "c10::intrusive_ptr<GroupedProjectionPlan> PrepareGrouped"
        )
    ]
    device_routes = source[
        source.index("class DeviceRoutes :") : source.index("c10::intrusive_ptr<DeviceRoutes> PrepareDeviceRoutes")
    ]
    harness = r"""
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#define TORCH_CHECK(condition, ...) \
  do { if (!(condition)) throw std::runtime_error("rejected"); } while (false)
namespace torch { struct CustomClassHolder {}; }
namespace c10 {
enum class DeviceType { CPU, PrivateUse1 };
struct Device {
  int index = 0;
  DeviceType kind = DeviceType::PrivateUse1;
  DeviceType type() const { return kind; }
  bool operator==(Device other) const { return index == other.index && kind == other.kind; }
};
template <typename T> using intrusive_ptr = std::shared_ptr<T>;
template <typename T, typename... Args> intrusive_ptr<T> make_intrusive(Args&&... args) {
  return std::make_shared<T>(std::forward<Args>(args)...);
}
}
int metadata_uploads = 0;
int output_zero_fills = 0;
namespace at {
enum ScalarType { kByte, kLong, kBFloat16 };
struct Tensor {
  std::vector<int64_t> shape;
  ScalarType dtype;
  c10::Device where;
  std::shared_ptr<std::vector<uint8_t>> owner;
  Tensor(std::vector<int64_t> dimensions, ScalarType type, c10::Device device = {})
      : shape(std::move(dimensions)), dtype(type), where(device),
        owner(std::make_shared<std::vector<uint8_t>>(nbytes())) {}
  bool defined() const { return owner != nullptr; }
  c10::Device device() const { return where; }
  int dim() const { return shape.size(); }
  int64_t size(size_t axis) const { return shape.at(axis); }
  int64_t numel() const { int64_t n = 1; for (auto dim : shape) n *= dim; return n; }
  size_t nbytes() const { return numel() * (dtype == kByte ? 1 : dtype == kLong ? 8 : 2); }
  void* data_ptr() const { return owner->data(); }
  template <typename T> T* data_ptr() const { return reinterpret_cast<T*>(owner->data()); }
  Tensor clone() const { Tensor result(shape,dtype,where); *result.owner = *owner; return result; }
  void zero_() { std::memset(data_ptr(), 0, nbytes()); ++output_zero_fills; }
  Tensor to(c10::Device device, ScalarType type, bool non_blocking, bool copy) const {
    TORCH_CHECK(!non_blocking && copy && type == dtype, "owned blocking copy required");
    Tensor result(shape, dtype, device);
    *result.owner = *owner;
    ++metadata_uploads;
    return result;
  }
};
}
int queue_drains = 0;
int launches = 0;
int records = 0;
int capture_status = 0;
uintptr_t launch_handle = 0;
namespace c10_npu {
struct NPUStream {
  uintptr_t handle = 100;
  void* stream(bool drain = true) const {
    if (drain) ++queue_drains;
    return reinterpret_cast<void*>(handle);
  }
};
NPUStream current;
NPUStream getCurrentNPUStream() { return current; }
struct OptionalNPUGuard { explicit OptionalNPUGuard(c10::Device) {} };
}
std::vector<std::function<int()>> callbacks;
namespace at_npu { namespace native {
struct OpCommand {
  static void RunOpApi(const char*, std::function<int()> callback, bool sync) {
    TORCH_CHECK(!sync, "must queue launch");
    callbacks.push_back(std::move(callback));
  }
};
}}
constexpr uint64_t kMxScaleGroupSize = 32;
constexpr uint64_t kPackedK0 = 16;
constexpr uint64_t kPackedN0 = 32;
constexpr uint32_t kKernelCoreCount = 32;
__LIMITS__
std::vector<int64_t> expected_counts{1, 0, 2};
struct MatFP4Args __ARGS__;
struct RegisteredKernel { const void* stub = reinterpret_cast<void*>(1234); };
void CheckNotCapturing(const char*) { TORCH_CHECK(capture_status == 0, "capture rejected"); }
void CheckCaptureAllowed(const char*) { TORCH_CHECK(capture_status != 2, "capture invalidated"); }
void CheckNpuTensor(const at::Tensor& tensor, const at::Tensor& anchor,
                    at::ScalarType dtype, int rank, const char*) {
  TORCH_CHECK(tensor.defined() && tensor.device() == anchor.device() &&
              tensor.device().type() == c10::DeviceType::PrivateUse1 &&
              tensor.dtype == dtype && tensor.dim() == rank, "bad device tensor");
}
void CheckCpuMetadata(const at::Tensor& tensor, at::ScalarType dtype, int rank, const char*) {
  TORCH_CHECK(tensor.device().type() == c10::DeviceType::CPU && tensor.dtype == dtype &&
              tensor.dim() == rank, "bad CPU metadata");
}
struct NamedRange __RANGE__;
NamedRange TensorRange(const char* name, const at::Tensor& tensor) __TENSOR_RANGE__
void CheckNoAlias(const std::vector<std::pair<const char*, at::Tensor>>& tensors) __ALIAS__
void ValidateProjectionShape(int64_t m, int64_t k, int64_t n) __SHAPE__
void ValidateGroupCount(int64_t groups) __GROUP_COUNT__
bool SupportsDynamicGroupedRows() __DYNAMIC_GROUPED_ROWS__
void ValidateGroupCounts(const at::Tensor& counts, int64_t rows) __COUNTS__
__ROUTES__
__DEVICE_ROUTES__
c10::intrusive_ptr<GroupedRoutes> PrepareRoutes(const at::Tensor& anchor, const at::Tensor& counts) __PREPARE_ROUTES__
__PLAN__
void CheckGroupedProjectionTensors(const at::Tensor& input_bytes, const at::Tensor& input_scale,
                                   const at::Tensor& packed_weight, const at::Tensor& pair_lut,
                                   const at::Tensor& row_scale, const at::Tensor& output, int64_t groups) __CHECKS__
void RecordStream(const std::vector<at::Tensor>& tensors, c10_npu::NPUStream stream) { records += tensors.size(); }
int rtKernelLaunch(const void* stub, uint32_t blocks, void* raw_args, size_t bytes, void*, void* stream) {
  auto& args = *static_cast<MatFP4Args*>(raw_args);
  TORCH_CHECK(stub && blocks == 32 && bytes == 152, "wrong launch ABI");
  TORCH_CHECK(args.group_count == 3 && args.k == 64 && args.n == 128 && args.d == 32, "wrong mixed shape");
  auto counts = static_cast<int64_t*>(args.group_list);
  for (size_t i = 0; i < expected_counts.size(); ++i) {
    TORCH_CHECK(counts[i] == expected_counts[i], "route mutation or cumulative counts");
  }
  ++launches;
  launch_handle = reinterpret_cast<uintptr_t>(stream);
  return 0;
}
void ProjectGrouped(const at::Tensor& input_bytes, const at::Tensor& input_scale,
                    const at::Tensor& packed_weight, const at::Tensor& pair_lut, const at::Tensor& row_scale,
                    const c10::intrusive_ptr<GroupedRoutes>& routes,
                    const c10::intrusive_ptr<GroupedProjectionPlan>& plan, at::Tensor& output) __PROJECT__
void ProjectGroupedDevice(const at::Tensor& input_bytes, const at::Tensor& input_scale,
                    const at::Tensor& packed_weight, const at::Tensor& pair_lut, const at::Tensor& row_scale,
                    const c10::intrusive_ptr<DeviceRoutes>& routes,
                    const c10::intrusive_ptr<GroupedProjectionPlan>& plan, at::Tensor& output,
                    bool zero_output) __PROJECT_DEVICE__
int main(int argc, char** argv) {
  const std::string scenario = argv[1];
  bool accepted = true;
  if (scenario == "validate") {
    at::Tensor counts({argc - 3}, at::kLong, {0, c10::DeviceType::CPU});
    for (int i = 3; i < argc; ++i) counts.data_ptr<int64_t>()[i - 3] = std::stoll(argv[i]);
    try { ValidateGroupCounts(counts, std::stoll(argv[2])); }
    catch (const std::runtime_error&) { accepted = false; }
    std::cout << "{\"accepted\":" << (accepted ? "true" : "false")
              << ",\"dynamic_grouped_rows\":" << (SupportsDynamicGroupedRows() ? "true" : "false") << "}";
    return 0;
  }
  if (scenario == "device_output") {
    const bool allow_padding = std::stoi(argv[2]);
    const bool zero_output = std::stoi(argv[3]);
    const int64_t active = std::stoll(argv[4]);
    expected_counts = {active, 0, 0};
    at::Tensor input({3, 64}, at::kByte), scales({3, 2}, at::kByte), weights({3, 1, 4, 16, 8}, at::kByte),
               lut({3, 1, 2, 32}, at::kByte), row({3, 32}, at::kLong), output({3, 32}, at::kBFloat16),
               mask({4}, at::kByte), tiling({1600}, at::kByte), counts({3}, at::kLong), flags({1}, at::kLong);
    counts.data_ptr<int64_t>()[0] = active;
    std::memset(output.data_ptr(), 0x55, output.nbytes());
    auto routes = std::make_shared<DeviceRoutes>(3,3,c10::Device{},counts,flags,
        c10_npu::current.stream(false),allow_padding);
    auto plan = std::make_shared<GroupedProjectionPlan>(3,64,32,c10::Device{},mask,tiling,
                                                       std::make_shared<RegisteredKernel>());
    ProjectGroupedDevice(input,scales,weights,lut,row,routes,plan,output,zero_output);
    // The launch callback is deliberately deferred; this observes only the
    // production output-initialization policy and ownership before execution.
    const bool cleared = output.owner->at(0) == 0 && output.owner->at(output.nbytes() - 1) == 0;
    const bool untouched = output.owner->at(0) == 0x55 && output.owner->at(output.nbytes() - 1) == 0x55;
    std::weak_ptr<DeviceRoutes> weak_device_routes = routes;
    std::weak_ptr<GroupedProjectionPlan> weak_device_plan = plan;
    routes.reset(); plan.reset();
    const bool retained = !weak_device_routes.expired() && !weak_device_plan.expired();
    for (const auto& callback : callbacks) TORCH_CHECK(callback() == 0, "launch failed");
    callbacks.clear();
    std::cout << "{\"zero_fills\":" << output_zero_fills
              << ",\"cleared\":" << (cleared ? "true" : "false")
              << ",\"untouched\":" << (untouched ? "true" : "false")
              << ",\"retained\":" << (retained ? "true" : "false")
              << ",\"released\":" << ((weak_device_routes.expired() && weak_device_plan.expired()) ? "true" : "false")
              << ",\"launches\":" << launches << ",\"records\":" << records
              << ",\"queue_drains\":" << queue_drains << ",\"uploads\":" << metadata_uploads << "}";
    return 0;
  }
  std::weak_ptr<GroupedRoutes> weak_routes;
  std::weak_ptr<GroupedProjectionPlan> weak_plan;
  {
    at::Tensor input({3, 64}, at::kByte), scales({3, 2}, at::kByte), weights({3, 1, 4, 16, 8}, at::kByte),
               lut({3, 1, 2, 32}, at::kByte), row({3, 32}, at::kLong), output({3, 32}, at::kBFloat16),
               mask({4}, at::kByte), tiling({1600}, at::kByte),
               counts({3}, at::kLong, {0, c10::DeviceType::CPU});
    counts.data_ptr<int64_t>()[0] = 1;
    counts.data_ptr<int64_t>()[2] = 2;
    if (scenario == "large_m") {
      const int64_t m = std::stoll(argv[2]);
      const int64_t tail = 1;
      expected_counts = {m, 0, tail};
      counts.data_ptr<int64_t>()[0] = m;
      counts.data_ptr<int64_t>()[2] = tail;
      input = at::Tensor({m + tail, 64}, at::kByte);
      scales = at::Tensor({m + tail, 2}, at::kByte);
      output = at::Tensor({m + tail, 32}, at::kBFloat16);
    }
    if (scenario == "negative") counts.data_ptr<int64_t>()[0] = -1;
    if (scenario == "sum_small") counts.data_ptr<int64_t>()[2] = 1;
    if (scenario == "sum_large") counts.data_ptr<int64_t>()[2] = 3;
    if (scenario == "all_zero") { counts.data_ptr<int64_t>()[0] = 0; counts.data_ptr<int64_t>()[2] = 0; }
    if (scenario == "huge_count") counts.data_ptr<int64_t>()[0] = std::numeric_limits<int64_t>::max();
    if (scenario == "too_many_groups") counts = at::Tensor({257},at::kLong,{0,c10::DeviceType::CPU});
    if (scenario == "no_groups") counts = at::Tensor({0},at::kLong,{0,c10::DeviceType::CPU});
    if (scenario == "device_counts") counts.where.kind = c10::DeviceType::PrivateUse1;
    if (scenario == "wrong_dtype") counts.dtype = at::kByte;
    if (scenario == "capture_prepare") capture_status = 1;
    try {
      auto routes = PrepareRoutes(input, counts);
      weak_routes = routes;
      // External CPU mutation must not change the already validated plan.
      std::memset(counts.data_ptr(), 0, counts.nbytes());
      auto plan = std::make_shared<GroupedProjectionPlan>(3,64,32,c10::Device{},mask,tiling,
                                                         std::make_shared<RegisteredKernel>());
      weak_plan = plan;
      if (scenario == "other_stream") c10_npu::current.handle = 200;
      if (scenario == "wrong_bank") weights = at::Tensor({2,1,4,16,8},at::kByte);
      if (scenario == "wrong_rows") { input = at::Tensor({2,64},at::kByte); scales = at::Tensor({2,2},at::kByte);
                                       output = at::Tensor({2,32},at::kBFloat16); }
      if (scenario == "wrong_device") {
        for (auto* tensor : {&input,&scales,&weights,&lut,&row,&output}) tensor->where.index = 1;
      }
      if (scenario == "alias") output.owner = input.owner;
      if (scenario == "capture_launch") capture_status = 1;
      if (scenario == "null_routes") routes.reset();
      if (scenario == "null_plan") plan.reset();
      for (int call = 0; call < 2; ++call) ProjectGrouped(input,scales,weights,lut,row,routes,plan,output);
    } catch (const std::runtime_error&) { accepted = false; }
  }
  const bool retained = !weak_routes.expired() && !weak_plan.expired();
  const auto pending = callbacks.size();
  for (const auto& callback : callbacks) TORCH_CHECK(callback() == 0, "launch failed");
  callbacks.clear();
  std::cout << "{\"accepted\":" << (accepted ? "true" : "false")
            << ",\"uploads\":" << metadata_uploads << ",\"launches\":" << launches
            << ",\"queue_drains\":" << queue_drains << ",\"records\":" << records
            << ",\"pending\":" << pending << ",\"retained\":" << (retained ? "true" : "false")
            << ",\"released\":" << ((weak_routes.expired() && weak_plan.expired()) ? "true" : "false")
            << ",\"stream\":" << launch_handle << "}";
}
"""
    for marker, actual in {
        "__ARGS__": body(source.replace("struct MatFP4Args {", "MatFP4Args() {"), "MatFP4Args"),
        "__RANGE__": body(source.replace("struct NamedRange {", "NamedRange() {"), "NamedRange"),
        "__TENSOR_RANGE__": body(source, "TensorRange"),
        "__ALIAS__": body(source, "CheckNoAlias"),
        "__SHAPE__": body(source, "ValidateProjectionShape"),
        "__LIMITS__": "\n".join(re.findall(r"constexpr int64_t kMax\w+ = \d+;", source)),
        "__GROUP_COUNT__": body(source, "ValidateGroupCount"),
        "__DYNAMIC_GROUPED_ROWS__": body(source, "SupportsDynamicGroupedRows"),
        "__COUNTS__": body(source, "ValidateGroupCounts"),
        "__ROUTES__": routes,
        "__DEVICE_ROUTES__": device_routes,
        "__PREPARE_ROUTES__": body(source, "PrepareRoutes"),
        "__PLAN__": plan,
        "__CHECKS__": body(source, "CheckGroupedProjectionTensors"),
        "__PROJECT__": body(source, "ProjectGrouped"),
        "__PROJECT_DEVICE__": body(source, "ProjectGroupedDevice"),
    }.items():
        harness = harness.replace(marker, actual)
    directory = tmp_path_factory.mktemp("gs32_grouped_host")
    cpp = directory / "grouped.cpp"
    executable = directory / "grouped"
    cpp.write_text(harness)
    result = subprocess.run(
        [compiler, "-std=c++17", str(cpp), "-o", str(executable)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    return executable


@pytest.mark.parametrize("allow_padding,active", [(False, 3), (True, 3), (True, 1), (True, 0)])
@pytest.mark.parametrize("zero_output", [True, False])
def test_device_projection_sparse_output_avoids_fill_and_retains_queued_owners(
    grouped_harness, allow_padding, active, zero_output
):
    result = subprocess.run(
        [str(grouped_harness), "device_output", str(int(allow_padding)), str(int(zero_output)), str(active)],
        capture_output=True,
        text=True,
        check=True,
    )
    cleared = allow_padding and zero_output
    assert json.loads(result.stdout) == {
        "zero_fills": int(cleared),
        "cleared": cleared,
        "untouched": not cleared,
        "retained": True,
        "released": True,
        "launches": 1,
        "records": 10,
        "queue_drains": 0,
        "uploads": 0,
    }


@pytest.mark.parametrize("scenario,stream", [("same_stream", 100), ("other_stream", 200)])
def test_grouped_routes_upload_once_for_two_queued_projections(grouped_harness, scenario, stream):
    result = subprocess.run([str(grouped_harness), scenario], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == {
        "accepted": True,
        "uploads": 1,
        "launches": 2,
        "queue_drains": 0,
        "records": 18,
        "pending": 2,
        "retained": True,
        "released": True,
        "stream": stream,
    }


@pytest.mark.parametrize(
    "rows", [129, 255, 256, 257, 479, 480, 481, 512, 1024, 2048, 4096, 4097, 8192, 16384, 32768, 32769, 65536, 131072]
)
def test_large_grouped_launch_retains_true_counts_and_immutable_routes(grouped_harness, rows):
    result = subprocess.run([str(grouped_harness), "large_m", str(rows)], capture_output=True, text=True, check=True)
    report = json.loads(result.stdout)
    assert report["accepted"] is True
    assert report["uploads"] == 1
    assert report["launches"] == report["pending"] == 2
    assert report["queue_drains"] == 0
    assert report["retained"] is True and report["released"] is True


@pytest.mark.parametrize(
    "counts,rows,accepted",
    [
        ([4096], 4096, True),
        ([4096, 0, 1], 4097, True),
        ([4096] * 8, 32768, True),
        ([4096] * 7 + [4095, 1], 32768, True),
        ([128] * 256, 32768, True),
        ([4097], 4097, True),
        ([8192], 8192, True),
        ([16384, 0, 16384], 32768, True),
        ([32768], 32768, True),
        ([32768, 0, 1], 32768, False),
        ([32769], 32769, True),
        ([4096] * 8 + [1], 32769, True),
        ([65536], 65536, True),
        ([65535, 0, 1], 65536, True),
        ([65536, 65536], 131072, True),
        ([512] * 256, 131072, True),
        ([131071, 0, 1], 131072, True),
        ([2**63 - 1], 2**63 - 1, True),
        ([2**63 - 2, 0, 1], 2**63 - 1, True),
        ([2**62, 2**62 - 1], 2**63 - 1, True),
        ([2**63 - 1, 1], 2**63 - 1, False),
        ([2**62, 2**62], 2**63 - 1, False),
        ([2**63 - 1] * 2, 2**63 - 1, False),
        ([2**63 - 1, -1], 2**63 - 2, False),
        ([4096] * 8, 32769, False),
        ([4096], 4095, False),
        ([4095], 4096, False),
        ([0], 0, False),
        ([0], -1, False),
        ([1], 2**63 - 1, False),
        ([2**63 - 1], 1, False),
        ([-(2**63), 1], 1, False),
        ([4096] * 256, 32768, False),
        ([0] * 257, 1, False),
        ([], 1, False),
    ],
)
def test_large_grouped_count_validation_is_bounded_before_accumulating(grouped_harness, counts, rows, accepted):
    command = [str(grouped_harness), "validate", str(rows), *map(str, counts)]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == {"accepted": accepted, "dynamic_grouped_rows": True}


@pytest.mark.parametrize(
    "scenario",
    [
        "negative",
        "sum_small",
        "sum_large",
        "all_zero",
        "huge_count",
        "too_many_groups",
        "no_groups",
        "device_counts",
        "wrong_dtype",
        "capture_prepare",
        "wrong_bank",
        "wrong_rows",
        "wrong_device",
        "alias",
        "capture_launch",
        "null_routes",
        "null_plan",
    ],
)
def test_grouped_routes_reject_invalid_inputs_before_launch(grouped_harness, scenario):
    result = subprocess.run([str(grouped_harness), scenario], capture_output=True, text=True, check=True)
    report = json.loads(result.stdout)
    assert report["accepted"] is False
    assert report["launches"] == 0
    assert report["pending"] == 0
    assert report["records"] == 0
    assert report["queue_drains"] == 0


@pytest.fixture(scope="module")
def grouped_tiling_harness(tmp_path_factory):
    """Check imported scheduling; hardware numerical evidence currently ends at M=4096."""
    compiler = shutil.which("c++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler is required for imported tiling coverage")
    builder = SOURCE.parent.parent / "mixed_w2w4_moe_ffn/components/down_w2/down_w2_tiling_builder.h"
    directory = tmp_path_factory.mktemp("gs32_grouped_tiling")
    cpp, executable = directory / "tiling.cpp", directory / "tiling"
    cpp.write_text(
        r"""
#include <iostream>
#include <string>
#include "down_w2_tiling_builder.h"
void Check(bool valid) { if (!valid) throw std::runtime_error("unsafe M tile"); }
int main(int argc, char** argv) {
  const uint64_t k = std::stoull(argv[1]), n = std::stoull(argv[2]);
  const auto blob = BuildDownW2TilingBlob(k, n);
  MatFP4TilingData tiling{};
  std::memcpy(&tiling, blob.data(), sizeof(tiling));
  Check(tiling.n_only_mode == 0);
  std::vector<uint64_t> row_counts;
  for (uint64_t m = 1; m <= 4096; ++m) row_counts.push_back(m);
  for (uint64_t m : {4097, 8191, 8192, 8193, 16383, 16384, 16385, 32767, 32768,
                     32769, 65535, 65536, 65537, 131072}) row_counts.push_back(m);
  for (uint64_t m : row_counts) {
    std::vector<int> covered(m, 0);
    const auto check_tile = [&](MatFP4TileDesc tile, uint64_t capacity) {
      Check(tile.rows > 0 && tile.rows <= capacity && tile.offset <= m && tile.rows <= m - tile.offset);
      Check((tile.rows + 15) / 16 * 16 <= capacity);
      // Source activation, K32 scales and BF16 output all use this same M
      // interval; no ceil-aligned tail may be treated as a live source row.
      Check((tile.offset + tile.rows) * k <= m * k);
      Check((tile.offset + tile.rows) * (k / 32) <= m * (k / 32));
      Check((tile.offset + tile.rows) * n <= m * n);
      for (uint64_t row = tile.offset; row < tile.offset + tile.rows; ++row) ++covered[row];
    };
    if (!MatFP4IsWideGroup(m, tiling.n_only_mode, tiling.wide_m_start)) {
      for (uint64_t tile = 0; tile < MatFP4NarrowTileCount(m); ++tile) {
        check_tile(MatFP4GetNarrowTile(m, tile, tiling.small_m_limit), tiling.small_m_limit);
      }
    } else {
      const uint64_t first = MatFP4SplitNearHalf16(m);
      for (uint64_t lane = 0; lane < 2; ++lane) {
        const uint64_t begin = lane == 0 ? 0 : first;
        const uint64_t local = lane == 0 ? first : m - first;
        for (uint64_t tile = 0; tile < MatFP4WideTileCount(local, tiling.wide_m_limit); ++tile) {
          auto descriptor = MatFP4GetWideTile(local, tile, tiling.wide_m_limit);
          descriptor.offset += begin;
          check_tile(descriptor, tiling.wide_m_limit);
        }
      }
    }
    for (const int copies : covered) Check(copies == 1);
  }
  std::cout << row_counts.size() << ":" << row_counts.back();
}
"""
    )
    result = subprocess.run(
        [compiler, "-std=c++17", "-I", str(builder.parent), str(cpp), "-o", str(executable)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return executable


@pytest.mark.parametrize("k,n", [(4096, 4096), (2048, 4096)])
def test_imported_grouped_schedule_covers_live_rows_and_extended_boundaries(grouped_tiling_harness, k, n):
    result = subprocess.run([str(grouped_tiling_harness), str(k), str(n)], capture_output=True, text=True, check=True)
    assert result.stdout == "4110:131072"
