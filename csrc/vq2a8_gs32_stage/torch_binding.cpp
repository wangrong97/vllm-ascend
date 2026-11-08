// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Torch-owned complete mixed-W2/W4 FFN launches.

#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/custom_class.h>
#include <torch/library.h>

#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "acl/acl_rt.h"
#if __has_include(<runtime/rt.h>)
// Preferred CANN 9.1 low-level runtime declarations. The target CANN 9.1
// installation must compile-verify this path before the adapter is accepted.
#include <runtime/rt.h>
#else
// Compatibility declaration source imported with the unmodified kernel.
// It is never copied or edited by this adapter.
#include "../mixed_w2w4_moe_ffn/hacl_rt.h"
#endif
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUGraphsUtils.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "../mixed_w2w4_moe_ffn/tiling_builders.h"
#include "../mixed_w2w4_moe_ffn/mixed_tiling.h"
#include "../mixed_w2w4_moe_ffn/weight_offsets.h"

namespace vq2a8_gs32_stage {
namespace {

constexpr char kKernelName[] = "mixed_w2w4_moe_ffn_v4";
constexpr uint32_t kKernelCoreCount = 32;
constexpr uintptr_t kRequiredAlignment = 32;
// Match the imported standalone Host's validated group-count range.
constexpr int64_t kMaxGroupedExperts = 256;
constexpr int64_t kRouteMaskWordBits = 32;

struct MatFP4Args {
  uint64_t group_count;
  uint64_t k;
  uint64_t n;
  uint64_t d;
  void* group_list;
  void* a;
  void* a_scale;
  void* gateup_w;
  void* gateup_table;
  void* down_w;
  void* down_table;
  void* scale_upgate;
  void* scale_down;
  void* is_expert_w4;
  void* is_down_w4;
  void* h;
  void* h_scale;
  void* c;
  void* tiling;
};

static_assert(sizeof(MatFP4Args) == 19 * sizeof(uint64_t), "unexpected mixed kernel argument ABI");
static_assert(kMixedTilingVersion == 4 && kMixedTilingBytes == 5728, "unexpected mixed tiling ABI");

struct RegisteredKernel {
  std::vector<uint8_t> binary;
  std::string path;
  std::string kernel_name{kKernelName};
  void* binary_handle = nullptr;
  const void* stub = nullptr;
};

std::mutex g_registration_mutex;
std::shared_ptr<const RegisteredKernel> g_registered_kernel;

void CheckRt(rtError_t result, const char* operation) {
  TORCH_CHECK(result == RT_ERROR_NONE, operation, " failed with runtime error ", static_cast<int64_t>(result));
}

void CheckNotCapturing(const char* operation) {
  // The Python startup/device guard establishes the selected NPU context.
  // Use the exported query: the inline assertNotCapturing helper references
  // IsContextInitialized, which is not exported by some torch_npu builds.
  // Accept only None, so Active and Invalidated capture states both fail.
  const auto status = c10_npu::currentStreamCaptureStatusMayInitCtx();
  TORCH_CHECK(status == c10_npu::CaptureStatus::None,
              operation, " is eager-only and cannot run during NPU graph capture; status=",
              static_cast<int>(status));
}

void CheckCaptureAllowed(const char* operation) {
  const auto status = c10_npu::currentStreamCaptureStatusMayInitCtx();
  TORCH_CHECK(status == c10_npu::CaptureStatus::None || status == c10_npu::CaptureStatus::Active,
              operation, " cannot run on an invalidated NPU graph capture; status=", static_cast<int>(status));
}

std::vector<uint8_t> ReadElfBinary(const std::string& path) {
  TORCH_CHECK(!path.empty(), "vq2a8_gs32_stage.initialize requires a non-empty binary path");
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  TORCH_CHECK(stream.good(), "cannot open mixed kernel binary: ", path);
  const std::streamoff size = stream.tellg();
  TORCH_CHECK(size >= 4, "mixed kernel binary is empty or truncated: ", path);
  TORCH_CHECK(static_cast<uint64_t>(size) <= std::numeric_limits<size_t>::max(),
              "mixed kernel binary is too large for this process: ", path);
  std::vector<uint8_t> bytes(static_cast<size_t>(size));
  stream.seekg(0, std::ios::beg);
  stream.read(reinterpret_cast<char*>(bytes.data()), size);
  TORCH_CHECK(stream.good(), "failed to read mixed kernel binary: ", path);
  TORCH_CHECK(bytes[0] == 0x7f && bytes[1] == 'E' && bytes[2] == 'L' && bytes[3] == 'F',
              "mixed kernel binary must be an ELF object: ", path);
  // ABI v4 has a different exported entry symbol, so an older ELF cannot be
  // launched with the new post-processing options even outside the launcher.
  TORCH_CHECK(std::search(bytes.begin(), bytes.end(), kKernelName, kKernelName + sizeof(kKernelName) - 1) != bytes.end(),
              "mixed kernel binary lacks ABI v4; rebuild both the kernel .o and Torch bridge .so: ", path);
  return bytes;
}

void Initialize(const std::string& binary_path) {
  CheckNotCapturing("vq2a8_gs32_stage.initialize");
  std::lock_guard<std::mutex> lock(g_registration_mutex);
  if (g_registered_kernel != nullptr) {
    TORCH_CHECK(g_registered_kernel->path == binary_path,
                "vq2a8_gs32_stage is already initialized with a different binary; start a new process");
    return;
  }

  const char* soc = aclrtGetSocName();
  TORCH_CHECK(soc != nullptr && std::strncmp(soc, "Ascend950", 9) == 0,
              "vq2a8_gs32_stage requires an initialized Ascend950 device");

  auto state = std::make_shared<RegisteredKernel>();
  state->path = binary_path;
  state->binary = ReadElfBinary(binary_path);
  rtDevBinary_t descriptor{};
  descriptor.magic = RT_DEV_BINARY_MAGIC_ELF;
  descriptor.version = 0;
  descriptor.data = state->binary.data();
  descriptor.length = state->binary.size();
  CheckRt(rtDevBinaryRegister(&descriptor, &state->binary_handle), "rtDevBinaryRegister");
  TORCH_CHECK(state->binary_handle != nullptr, "rtDevBinaryRegister returned a null binary handle");

  state->stub = reinterpret_cast<const void*>(state->kernel_name.c_str());
  CheckRt(rtFunctionRegister(state->binary_handle, state->stub, state->kernel_name.c_str(),
                             const_cast<void*>(state->stub), 0),
          "rtFunctionRegister");
  // The ELF byte vector and its function-name storage remain alive for the
  // process lifetime. There is intentionally no reset/unregister hot path.
  g_registered_kernel = std::move(state);
}

std::shared_ptr<const RegisteredKernel> RegisteredKernelSnapshot() {
  std::lock_guard<std::mutex> lock(g_registration_mutex);
  TORCH_CHECK(g_registered_kernel != nullptr,
              "vq2a8_gs32_stage.initialize(binary) must be called once during startup before preparing a full FFN plan");
  return g_registered_kernel;
}

void CheckNpuTensor(const at::Tensor& tensor, const at::Tensor& anchor, at::ScalarType dtype, int64_t rank,
                    const char* name) {
  TORCH_CHECK(tensor.defined(), name, ": undefined tensor");
  TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1, name, ": an NPU tensor is required");
  TORCH_CHECK(tensor.device() == anchor.device(), name, ": all tensors must share one NPU device");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, ": incorrect dtype");
  TORCH_CHECK(tensor.dim() == rank, name, ": incorrect rank");
  TORCH_CHECK(tensor.is_contiguous(), name, ": contiguous storage is required");
  // Unified parameters contain all experts. An aligned contiguous slice may
  // start inside the allocation; the queued callback retains its base storage.
  const uint64_t offset = static_cast<uint64_t>(tensor.storage_offset()) * tensor.element_size();
  const uint64_t capacity = tensor.storage().nbytes();
  TORCH_CHECK(offset <= capacity && tensor.nbytes() <= capacity - offset,
              name, ": tensor view exceeds its base storage");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % kRequiredAlignment == 0,
              name, ": storage must be 32-byte aligned");
}

void CheckCpuMetadata(const at::Tensor& tensor, at::ScalarType dtype, int64_t rank, const char* name) {
  TORCH_CHECK(tensor.defined(), name, ": undefined tensor");
  TORCH_CHECK(tensor.device().is_cpu(), name, ": validated CPU metadata is required");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, ": incorrect dtype");
  TORCH_CHECK(tensor.dim() == rank, name, ": incorrect rank");
  TORCH_CHECK(tensor.is_contiguous(), name, ": contiguous storage is required");
  TORCH_CHECK(tensor.storage_offset() == 0, name, ": storage views/offsets are not accepted");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % kRequiredAlignment == 0,
              name, ": storage must be 32-byte aligned");
}

struct NamedRange {
  const char* name;
  uintptr_t begin;
  uintptr_t end;
};

NamedRange TensorRange(const char* name, const at::Tensor& tensor) {
  const uintptr_t begin = reinterpret_cast<uintptr_t>(tensor.data_ptr());
  const size_t bytes = tensor.nbytes();
  TORCH_CHECK(bytes > 0 && bytes <= std::numeric_limits<uintptr_t>::max() - begin,
              name, ": invalid storage byte range");
  return {name, begin, begin + bytes};
}

void CheckNoAlias(const std::vector<std::pair<const char*, at::Tensor>>& tensors) {
  std::vector<NamedRange> ranges;
  ranges.reserve(tensors.size());
  for (const auto& named : tensors) {
    ranges.push_back(TensorRange(named.first, named.second));
  }
  for (size_t left = 0; left < ranges.size(); ++left) {
    for (size_t right = left + 1; right < ranges.size(); ++right) {
      TORCH_CHECK(ranges[left].end <= ranges[right].begin || ranges[right].end <= ranges[left].begin,
                  ranges[left].name, " and ", ranges[right].name,
                  " must not alias or overlap storage");
    }
  }
}

// Hold private Tensor metadata without copying the resident weight bytes. The
// caller can release its Parameters; the plan and queued callback own storage.
// Cached ranges are valid only while the backing allocations remain in place.
class ResidentWeights {
 public:
  ResidentWeights(const at::Tensor& anchor, int64_t groups, int64_t n, int64_t d,
                  const std::array<int64_t, 4>& bytes,
                  const std::array<at::Tensor, 6>& tensors) {
    const std::array<const char*, 6> names{
        "gate_weight", "gate_table", "gate_row_scale", "down_weight", "down_table", "down_row_scale"};
    std::vector<std::pair<const char*, at::Tensor>> named;
    for (size_t i = 0; i < tensors.size(); ++i) {
      const bool scale = i == 2 || i == 5;
      CheckNpuTensor(tensors[i], anchor, scale ? at::kLong : at::kByte, scale ? 2 : 1, names[i]);
      tensors_[i] = tensors[i].detach();
      ranges_[i] = TensorRange(names[i], tensors_[i]);
      named.emplace_back(names[i], tensors_[i]);
    }
    TORCH_CHECK(tensors_[0].numel() == bytes[0] && tensors_[1].numel() == bytes[1] &&
                    tensors_[3].numel() == bytes[2] && tensors_[4].numel() == bytes[3],
                "full FFN mixed weight/table blob sizes do not match resident precision masks");
    TORCH_CHECK(tensors_[2].size(0) == groups && tensors_[2].size(1) == n,
                "full FFN Gate row scales must be [G,N]");
    TORCH_CHECK(tensors_[5].size(0) == groups && tensors_[5].size(1) == d,
                "full FFN Down row scales must be [G,D]");
    CheckNoAlias(named);
  }

  const at::Tensor& operator[](size_t index) const { return tensors_[index]; }

  void CheckDynamicRanges(const std::vector<std::pair<const char*, at::Tensor>>& tensors) const {
    // No repeated weight shape/dtype/precision validation or weight/weight
    // pairwise scan. Still reject relocated/shrunk storage and dynamic aliases.
    for (size_t i = 0; i < tensors_.size(); ++i) {
      const auto& tensor = tensors_[i];
      const uint64_t offset = static_cast<uint64_t>(tensor.storage_offset()) * tensor.element_size();
      const uint64_t capacity = tensor.storage().nbytes();
      TORCH_CHECK(offset <= capacity && tensor.nbytes() <= capacity - offset &&
                      reinterpret_cast<uintptr_t>(tensor.data_ptr()) == ranges_[i].begin,
                  "full FFN resident weight storage changed after plan preparation");
    }
    CheckNoAlias(tensors);
    for (const auto& named : tensors) {
      const auto range = TensorRange(named.first, named.second);
      for (const auto& resident : ranges_) {
        TORCH_CHECK(range.end <= resident.begin || resident.end <= range.begin,
                    range.name, " and ", resident.name, " must not alias or overlap storage");
      }
    }
  }

 private:
  std::array<at::Tensor, 6> tensors_;
  std::array<NamedRange, 6> ranges_;
};

void RecordStream(const std::vector<at::Tensor>& tensors, c10_npu::NPUStream stream) {
  for (const auto& tensor : tensors) {
    c10_npu::NPUCachingAllocator::recordStream(tensor.storage().data_ptr(), stream);
  }
}

void ValidateGroupCount(int64_t groups) {
  TORCH_CHECK(groups > 0 && groups <= kMaxGroupedExperts,
              "full FFN device routes require 1..256 resident experts");
}

std::pair<at::Tensor, at::Tensor> SanitizeDeviceCounts(const at::Tensor& counts, int64_t rows,
                                                       bool allow_padding = false) {
  // The registered helper copies, bounds-checks and sums counts in one device
  // launch. Its fresh outputs never become caller-visible aliases of this
  // opaque route object. Direct prepare_device_routes calls retain all checks.
  const auto operation = c10::Dispatcher::singleton().findSchemaOrThrow(
      "vq2a8_gs32_stage::_sanitize_device_counts", "");
  TORCH_CHECK(operation.hasKernelForDispatchKey(c10::DispatchKey::PrivateUse1),
              "GS32 count sanitizer is not registered; import vllm_ascend.ops.fused_moe.vq2a8_gs32_counts "
              "after loading the bridge");
  auto result = operation.typed<std::tuple<at::Tensor, at::Tensor>(const at::Tensor&, int64_t, bool)>().call(
      counts, rows, allow_padding);
  auto safe_counts = std::get<0>(std::move(result));
  auto status = std::get<1>(std::move(result));
  CheckNpuTensor(safe_counts, counts, at::kLong, 1, "sanitized_counts");
  CheckNpuTensor(status, counts, at::kInt, 1, "count_status");
  TORCH_CHECK(safe_counts.numel() == counts.numel() && status.numel() == 1,
              "GS32 count sanitizer returned incompatible shapes");
  CheckNoAlias({{"input_counts", counts}, {"sanitized_counts", safe_counts}, {"count_status", status}});
  return {std::move(safe_counts), std::move(status)};
}

// Device route data is produced on the current stream and recalculated inside
// each graph replay. A different stream
// cannot use the handle without a dependency, so reject it instead of guessing.
// No constructor or metadata accessor is exposed to Python.
class DeviceRoutes : public torch::CustomClassHolder {
 public:
  DeviceRoutes(int64_t rows, int64_t groups, c10::Device device, at::Tensor group_list,
               at::Tensor status, void* stream, bool allow_padding = false)
      : rows_(rows), groups_(groups), device_(device), group_list_(std::move(group_list)),
        status_(std::move(status)), stream_(stream), allow_padding_(allow_padding) {}

  void CheckStream() const {
    TORCH_CHECK(c10_npu::getCurrentNPUStream().stream(false) == stream_,
                "device routes must be consumed on their producing NPU stream");
  }

  void CheckMatches(const at::Tensor& input_bytes, int64_t groups) const {
    TORCH_CHECK(input_bytes.device() == device_, "device routes belong to a different NPU device");
    TORCH_CHECK(input_bytes.size(0) == rows_ && groups == groups_,
                "device routes do not match routed row count or resident expert count");
    CheckStream();
  }

  c10::Device Device() const { return device_; }
  const at::Tensor& GroupList() const { return group_list_; }
  const at::Tensor& Status() const { return status_; }
  bool AllowsPadding() const { return allow_padding_; }

 private:
  const int64_t rows_;
  const int64_t groups_;
  const c10::Device device_;
  const at::Tensor group_list_;
  const at::Tensor status_;
  void* const stream_;
  const bool allow_padding_;
};

c10::intrusive_ptr<DeviceRoutes> PrepareDeviceRoutes(const at::Tensor& anchor, const at::Tensor& counts,
                                                     bool allow_padding) {
  TORCH_CHECK(anchor.defined() && anchor.device().type() == c10::DeviceType::PrivateUse1 && anchor.dim() == 2,
              "device routes require an NPU [routed_rows,K] anchor");
  const c10_npu::OptionalNPUGuard guard(anchor.device());
  CheckCaptureAllowed("vq2a8_gs32_stage.prepare_device_routes");
  CheckNpuTensor(counts, anchor, at::kLong, 1, "group_counts");
  ValidateGroupCount(counts.numel());
  TORCH_CHECK(anchor.size(0) > 0, "device routes require positive routed row capacity");
  auto sanitized = SanitizeDeviceCounts(counts, anchor.size(0), allow_padding);
  const auto stream = c10_npu::getCurrentNPUStream();
  RecordStream({counts, sanitized.first, sanitized.second}, stream);
  return c10::make_intrusive<DeviceRoutes>(anchor.size(0), counts.numel(), anchor.device(),
                                          std::move(sanitized.first), std::move(sanitized.second), stream.stream(false),
                                          allow_padding);
}

at::Tensor DeviceRoutesStatus(const c10::intrusive_ptr<DeviceRoutes>& routes) {
  TORCH_CHECK(routes != nullptr, "device routes are required");
  const c10_npu::OptionalNPUGuard guard(routes->Device());
  CheckCaptureAllowed("vq2a8_gs32_stage.device_routes_status");
  routes->CheckStream();
  // 1 means valid; 0 means every native group was zeroed. The caller must
  // include this result in its graph-wide validity check before using output.
  // Return an owned copy so Python mutation cannot change the internal result.
  return routes->Status().clone();
}

int64_t FullFFNAbiVersion() { return static_cast<int64_t>(kMixedTilingVersion); }

int64_t CheckedElements(std::initializer_list<int64_t> dimensions) {
  int64_t result = 1;
  for (const auto dimension : dimensions) {
    TORCH_CHECK(dimension > 0 && result <= std::numeric_limits<int64_t>::max() / dimension,
                "full FFN tensor dimensions overflow int64");
    result *= dimension;
  }
  return result;
}

at::Tensor MakeFullFFNTiling(int64_t k, int64_t n, int64_t d, uint64_t stages, double limit,
                           const MixedWeightOffsets& offsets) {
  auto result = at::zeros({static_cast<int64_t>(kMixedTilingBytes)},
                          at::TensorOptions().device(at::kCPU).dtype(at::kByte));
  const MixedTilingHeader header{
      kMixedTilingMagic, kMixedTilingVersion, kMixedGateW2TilingOffset, kMixedGateW4TilingOffset,
      kMixedDownW2TilingOffset, kMixedDownW4TilingOffset, kMixedTilingBytes, stages};
  auto* bytes = result.data_ptr<uint8_t>();
  std::memcpy(bytes, &header, sizeof(header));
  const auto copy = [bytes](uint64_t offset, uint64_t expected, const std::vector<uint8_t>& blob) {
    TORCH_CHECK(blob.size() == expected, "full FFN component tiling has an incompatible ABI");
    std::memcpy(bytes + offset, blob.data(), blob.size());
  };
  copy(kMixedGateW2TilingOffset, kMixedGateW2TilingBytes, BuildGateW2TilingBlob(k, n));
  copy(kMixedGateW4TilingOffset, kMixedGateW4TilingBytes, BuildGateW4TilingBlob(k, n));
  copy(kMixedDownW2TilingOffset, kMixedDownW2TilingBytes, BuildDownW2TilingBlob(n / 2, d));
  copy(kMixedDownW4TilingOffset, kMixedDownW4TilingBytes, BuildDownW4TilingBlob(n / 2, d));
  const MixedPostOptions post{static_cast<float>(limit), {}};
  std::memcpy(bytes + kMixedPostOptionsOffset, &post, sizeof(post));
  std::memcpy(bytes + kMixedGateWeightOffsetsOffset, offsets.gate.data(), kMixedWeightOffsetsBytes);
  std::memcpy(bytes + kMixedDownWeightOffsetsOffset, offsets.down.data(), kMixedWeightOffsetsBytes);
  return result;
}

// Immutable resident metadata. Preparation is outside capture; route counts
// and token capacity are dynamic and are deliberately not part of this plan.
class FullFFNPlan : public torch::CustomClassHolder {
 public:
  FullFFNPlan(int64_t groups, int64_t k, int64_t n, int64_t d, c10::Device device,
              ResidentWeights weights, at::Tensor gate_mask, at::Tensor down_mask,
              at::Tensor tiling, std::shared_ptr<const RegisteredKernel> registered)
      : groups_(groups), k_(k), n_(n), d_(d), device_(device), weights_(std::move(weights)),
        gate_mask_(std::move(gate_mask)), down_mask_(std::move(down_mask)),
        tiling_(std::move(tiling)), registered_(std::move(registered)) {}

  void CheckMatches(const at::Tensor& input, const at::Tensor& output) const {
    TORCH_CHECK(input.device() == device_, "full FFN plan belongs to a different device");
    TORCH_CHECK(input.size(1) == k_ && output.size(1) == d_, "full FFN plan dimensions do not match input/output");
  }
  int64_t Groups() const { return groups_; }
  int64_t K() const { return k_; }
  int64_t N() const { return n_; }
  int64_t D() const { return d_; }
  const ResidentWeights& Weights() const { return weights_; }
  const at::Tensor& GateMask() const { return gate_mask_; }
  const at::Tensor& DownMask() const { return down_mask_; }
  const at::Tensor& Tiling() const { return tiling_; }
  const RegisteredKernel& Kernel() const { return *registered_; }

 private:
  const int64_t groups_;
  const int64_t k_;
  const int64_t n_;
  const int64_t d_;
  const c10::Device device_;
  const ResidentWeights weights_;
  const at::Tensor gate_mask_;
  const at::Tensor down_mask_;
  const at::Tensor tiling_;
  const std::shared_ptr<const RegisteredKernel> registered_;
};

c10::intrusive_ptr<FullFFNPlan> PrepareFullFFN(const at::Tensor& anchor, int64_t n, int64_t d,
                                              const at::Tensor& gate_w4, const at::Tensor& down_w4,
                                              double swiglu_limit,
                                              const at::Tensor& gate_weight, const at::Tensor& gate_table,
                                              const at::Tensor& gate_row_scale, const at::Tensor& down_weight,
                                              const at::Tensor& down_table, const at::Tensor& down_row_scale) {
  TORCH_CHECK(anchor.defined() && anchor.device().type() == c10::DeviceType::PrivateUse1,
              "full FFN preparation requires an NPU anchor");
  const c10_npu::OptionalNPUGuard guard(anchor.device());
  CheckNotCapturing("vq2a8_gs32_stage.prepare_full_ffn");
  CheckNpuTensor(anchor, anchor, at::kByte, 2, "anchor");
  CheckCpuMetadata(gate_w4, at::kBool, 1, "gate_w4");
  CheckCpuMetadata(down_w4, at::kBool, 1, "down_w4");
  const auto groups = gate_w4.numel();
  ValidateGroupCount(groups);
  TORCH_CHECK(down_w4.numel() == groups, "Gate/Down precision metadata must have the same expert count");
  const auto k = anchor.size(1);
  TORCH_CHECK(k > 0 && k % 64 == 0 && n > 0 && n % 128 == 0 && d > 0 && d % 32 == 0,
              "full FFN requires K%64=0, GateUp N%128=0 and Down D%32=0");
  TORCH_CHECK(std::isfinite(swiglu_limit) && swiglu_limit >= 0 &&
                  swiglu_limit <= std::numeric_limits<float>::max(),
              "full FFN swiglu_limit must be finite, non-negative and representable in float32");
  const auto gate_elements = CheckedElements({k, n});
  const auto down_elements = CheckedElements({n / 2, d});
  CheckedElements({groups, gate_elements});
  CheckedElements({groups, down_elements});
  CheckedElements({groups, n, static_cast<int64_t>(sizeof(int64_t))});
  CheckedElements({groups, d, static_cast<int64_t>(sizeof(int64_t))});
  const auto mask_words = (groups + kRouteMaskWordBits - 1) / kRouteMaskWordBits;
  auto gate_mask = at::zeros({mask_words}, at::TensorOptions().device(at::kCPU).dtype(at::kInt));
  auto down_mask = at::zeros_like(gate_mask);
  auto* gate_words = reinterpret_cast<uint32_t*>(gate_mask.data_ptr<int32_t>());
  auto* down_words = reinterpret_cast<uint32_t*>(down_mask.data_ptr<int32_t>());
  std::array<int64_t, 4> blob_bytes{0, groups * (gate_elements / 32), 0, groups * (down_elements / 32)};
  uint64_t stages = 0;
  for (int64_t expert = 0; expert < groups; ++expert) {
    const bool gate_is_w4 = gate_w4.data_ptr<bool>()[expert];
    const bool down_is_w4 = down_w4.data_ptr<bool>()[expert];
    TORCH_CHECK(!gate_is_w4 || down_is_w4, "full FFN does not support W4 GateUp with W2 Down");
    if (gate_is_w4) gate_words[expert / kRouteMaskWordBits] |= uint32_t{1} << (expert % kRouteMaskWordBits);
    if (down_is_w4) down_words[expert / kRouteMaskWordBits] |= uint32_t{1} << (expert % kRouteMaskWordBits);
    blob_bytes[0] += gate_elements / (gate_is_w4 ? 2 : 4);
    blob_bytes[2] += down_elements / (down_is_w4 ? 2 : 4);
    // Include every resident precision, even when it has zero tokens in the
    // capture warmup. All cores must execute the same stage barriers on replay.
    stages |= gate_is_w4 ? kMixedStageGateW4 : kMixedStageGateW2;
    stages |= down_is_w4 ? kMixedStageDownW4 : kMixedStageDownW2;
  }
  ResidentWeights weights(anchor, groups, n, d, blob_bytes,
                          {gate_weight, gate_table, gate_row_scale, down_weight, down_table, down_row_scale});
  const auto offsets = BuildMixedWeightOffsets(groups, k, n, d, gate_words, down_words);
  TORCH_CHECK(offsets.gate_bytes == static_cast<uint64_t>(blob_bytes[0]) &&
                  offsets.down_bytes == static_cast<uint64_t>(blob_bytes[2]),
              "full FFN static weight offsets disagree with resident blob sizes");
  const auto tiling = MakeFullFFNTiling(k, n, d, stages, swiglu_limit, offsets);
  const auto registered = RegisteredKernelSnapshot();
  return c10::make_intrusive<FullFFNPlan>(
      groups, k, n, d, anchor.device(), std::move(weights),
      gate_mask.to(anchor.device(), at::kInt, false, true),
      down_mask.to(anchor.device(), at::kInt, false, true),
      tiling.to(anchor.device(), at::kByte, false, true), registered);
}

void FullFFNDevice(const at::Tensor& input_bytes, const at::Tensor& input_scale,
                   const c10::intrusive_ptr<DeviceRoutes>& routes, const c10::intrusive_ptr<FullFFNPlan>& plan,
                   at::Tensor& hidden, at::Tensor& hidden_scale, at::Tensor& output) {
  TORCH_CHECK(routes != nullptr && plan != nullptr, "full FFN requires prepared routes and resident plan");
  TORCH_CHECK(input_bytes.defined() && input_bytes.device().type() == c10::DeviceType::PrivateUse1,
              "full FFN requires NPU input");
  const c10_npu::OptionalNPUGuard guard(input_bytes.device());
  CheckCaptureAllowed("vq2a8_gs32_stage.full_ffn_device");
  CheckNpuTensor(input_bytes, input_bytes, at::kByte, 2, "input_bytes");
  CheckNpuTensor(input_scale, input_bytes, at::kByte, 2, "input_scale");
  CheckNpuTensor(hidden, input_bytes, at::kByte, 2, "hidden");
  CheckNpuTensor(hidden_scale, input_bytes, at::kByte, 2, "hidden_scale");
  CheckNpuTensor(output, input_bytes, at::kBFloat16, 2, "output");
  plan->CheckMatches(input_bytes, output);
  routes->CheckMatches(input_bytes, plan->Groups());
  const auto rows = input_bytes.size(0);
  TORCH_CHECK(rows > 0, "full FFN requires positive routed row capacity");
  const auto shape_is = [](const at::Tensor& tensor, int64_t first, int64_t second) {
    return tensor.size(0) == first && tensor.size(1) == second;
  };
  TORCH_CHECK(shape_is(input_scale, rows, plan->K() / 32), "full FFN input_scale must be [R,K/32]");
  TORCH_CHECK(shape_is(hidden, rows, plan->N() / 2), "full FFN hidden workspace must be [R,N/2]");
  TORCH_CHECK(shape_is(hidden_scale, rows, plan->N() / 64), "full FFN hidden scale must be [R,N/64]");
  TORCH_CHECK(shape_is(output, rows, plan->D()), "full FFN output must be [R,D]");
  const auto& weights = plan->Weights();
  weights.CheckDynamicRanges({{"input_bytes", input_bytes}, {"input_scale", input_scale},
                             {"hidden", hidden}, {"hidden_scale", hidden_scale}, {"output", output}});
  // Invalid counts are sanitized to zero before reaching the kernel. Define
  // output and scale padding on every replay, including a wholly empty rank.
  output.zero_();
  hidden_scale.fill_(127);
  const auto stream = c10_npu::getCurrentNPUStream();
  RecordStream({input_bytes, input_scale, weights[0], weights[1], weights[2],
                weights[3], weights[4], weights[5], hidden, hidden_scale, output,
                routes->GroupList(), routes->Status(), plan->GateMask(), plan->DownMask(), plan->Tiling()}, stream);
  const auto launch_stream = stream.stream(false);
  at_npu::native::OpCommand::RunOpApi(
      "Vq2a8Gs32FullFFN",
      [routes, plan, launch_stream, input_bytes, input_scale, hidden, hidden_scale, output]() -> int {
        const auto& weights = plan->Weights();
        MatFP4Args arguments{
            static_cast<uint64_t>(plan->Groups()), static_cast<uint64_t>(plan->K()),
            static_cast<uint64_t>(plan->N()), static_cast<uint64_t>(plan->D()), routes->GroupList().data_ptr(),
            input_bytes.data_ptr(), input_scale.data_ptr(), weights[0].data_ptr(), weights[1].data_ptr(),
            weights[3].data_ptr(), weights[4].data_ptr(), weights[2].data_ptr(), weights[5].data_ptr(),
            plan->GateMask().data_ptr(), plan->DownMask().data_ptr(), hidden.data_ptr(), hidden_scale.data_ptr(),
            output.data_ptr(), plan->Tiling().data_ptr()};
        return static_cast<int>(rtKernelLaunch(plan->Kernel().stub, kKernelCoreCount, &arguments,
                                               sizeof(arguments), nullptr, launch_stream));
      },
      false);
}

}  // namespace
}  // namespace vq2a8_gs32_stage

TORCH_LIBRARY(vq2a8_gs32_stage, m) {
  m.class_<vq2a8_gs32_stage::DeviceRoutes>("DeviceRoutes");
  m.class_<vq2a8_gs32_stage::FullFFNPlan>("FullFFNPlan");
  m.def("initialize(str binary) -> ()", &vq2a8_gs32_stage::Initialize);
  m.def("_sanitize_device_counts(Tensor counts, int rows, bool allow_padding) -> (Tensor, Tensor)");
  m.def("full_ffn_abi_version() -> int", &vq2a8_gs32_stage::FullFFNAbiVersion);
  m.def("prepare_full_ffn(Tensor anchor, int n, int d, Tensor gate_w4, Tensor down_w4, float swiglu_limit, "
        "Tensor gate_weight, Tensor gate_table, Tensor gate_row_scale, Tensor down_weight, "
        "Tensor down_table, Tensor down_row_scale) -> "
        "__torch__.torch.classes.vq2a8_gs32_stage.FullFFNPlan");
  m.def("full_ffn_device(Tensor input_bytes, Tensor input_scale, "
        "__torch__.torch.classes.vq2a8_gs32_stage.DeviceRoutes routes, "
        "__torch__.torch.classes.vq2a8_gs32_stage.FullFFNPlan plan, Tensor(a!) hidden, "
        "Tensor(b!) hidden_scale, Tensor(c!) output) -> ()");
  m.def("prepare_device_routes(Tensor anchor, Tensor counts, bool allow_padding=False) -> "
        "__torch__.torch.classes.vq2a8_gs32_stage.DeviceRoutes");
  m.def("device_routes_status(__torch__.torch.classes.vq2a8_gs32_stage.DeviceRoutes routes) -> Tensor",
        &vq2a8_gs32_stage::DeviceRoutesStatus);
}

TORCH_LIBRARY_IMPL(vq2a8_gs32_stage, PrivateUse1, m) {
  m.impl("prepare_full_ffn", &vq2a8_gs32_stage::PrepareFullFFN);
  m.impl("full_ffn_device", &vq2a8_gs32_stage::FullFFNDevice);
  m.impl("prepare_device_routes", &vq2a8_gs32_stage::PrepareDeviceRoutes);
}
