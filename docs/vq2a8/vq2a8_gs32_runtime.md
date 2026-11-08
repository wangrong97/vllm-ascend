# GS32 底层运行时

本层将 W2/W4 混合完整 FFN 接入 Torch 张量，包含初始激活量化、设备侧路由元数据、原生 FFN 和有效性检查。
模型加载、EP 通信及 runner 有效性检查保留原有职责；生产中不再构造分阶段投影或 reference 后端。

## 来源与接口

生产执行序列为：恒等路由 metadata → 初始 MXFP8 激活量化 → 原生完整 FFN → 合并有效性检查。
完整 FFN 在设备内执行 GateUp、裁剪后的 SwiGLU、中间激活量化和 Down；框架优化不改变这个序列的数学逻辑。

- `csrc/vq2a8_gs32_stage/torch_binding.cpp` 注册 `torch.ops.vq2a8_gs32_stage` 的完整 FFN 计划、执行及设备路由接口。
  `prepare_full_ffn` 在捕获前校验并持有六个权重张量；`full_ffn_device` 使用 ABI v4 设备入口，保留张量生命周期、流和动态别名校验。
- `prepare_device_routes` 接收设备计数，`device_routes_status` 提供有效性状态；调用方在接受结果前完成检查。
  桥接层使用独立的 Triton sanitizer，将计数范围、精确总和及安全副本生成合为一次设备调用。
  内部计数仍由 opaque route handle 持有，公开状态仍返回副本；没有跳过校验的 trusted-counts 开关。
  图捕获期间不将状态读回主机；runner 沿用采样前的汇总检查。
- 激活准备使用自然输入列顺序、BF16 舍入、K32 E8M0 缩放及 FP8 字节语义，不接收 `.perm`。
  Python/Triton 路由保留 counts/prefixes、padding、无活跃专家及无效输入的处理。
- `vq2a8_gs32_format.py` 定义 source-v3 格式，`vq2a8_gs32_full_ffn_layout.py` 构造完整 FFN 的专家 bank。
  支持 VQ2/VQ2、VQ2/MXFP4、MXFP4/MXFP4；MXFP4/VQ2 在分配前拒绝。
- 数值 oracle 位于 `tests/support/vq2a8_oracle.py` 和 nightly 测试 helper，只用于独立正确性对照。
  旧 staged/reference runtime 已移除，不再加载其投影计划或捕获其计算图。

默认 `full_graph` 使用 decode 图回放与 eager 分块 prefill，两者均执行完整 FFN。
`--enforce-eager` 只关闭模型图，运行记录使用内部 execution 名称 `fused`。
接口支持图捕获不等于目标硬件的整模型图验收已完成，仍需执行下述 NPU 测试。

## 完整 FFN 框架开销优化

完整 FFN bank 在加载时要求专家 ID 唯一且递增。dispatch 输出按逻辑专家分段，运行时拒绝非驻留专家的非零计数，
因此全部有效行已经与 bank 的物理顺序一致；EP 驻留专家 ID 中有空洞也不改变这一性质。
专用 identity metadata 每次在设备上校验计数、映射和顺序，仅输出 counts、validity 和 live-row summary，
省去 gather/inverse、三组偏移 scratch 及逐行映射 kernel。通用重排/subset helper 仍供独立测试使用，
不能把跳过活跃专家的 subset 路由当作恒等路由。

初始量化按原行顺序读取，仅访问 summary 指定的有效前缀；padding 即使包含 NaN/Inf 也不读取，
其 FP8 字节为零、E8M0 为 127。原生输出直接作为返回张量，省去另一份 BF16 result 及归位拷贝。
finish kernel 合并输入、路由、native 状态、中间 scale=255 和输出有限性检查：非法有效值清零并标记失败，
padding 输出清零，再用固定大小寄存器块归约并覆盖已有的每层 validity 标量。
该固定块循环避免大 prefill 为每个容量生成巨大归约向量；没有省略 runner 的最终汇总检查。

桥接层仍在每次执行时初始化 output 与 middle scales，防止空 rank、非法计数或不同 graph replay 继承旧数据。
不建立跨 layer、跨 stream 或跨图桶共享的可变 workspace：每次 capture 持有自己的张量及 route handle，
eager 返回值也不会被后续调用覆盖。图回放本来不会重复执行 Python 分配，优先减少实际缓冲及设备操作。

设备端升级为 ABI v4：tiling 由 1632 字节扩展为 5728 字节，追加 Gate/Down 各 256 个 uint64 权重字节偏移。
原有四个组件 tiling 和 SwiGLU 参数偏移不变；权重偏移在加载时按混合精度计算，设备只读取当前专家对应项。
不调整 tile、动态 token 路由、同步屏障、裁剪与舍入逻辑。每层计划增加 4096 字节设备元数据，替代设备执行中的权重偏移累计。
升级后必须重建 `.o`、`.so` 和清单；旧符号、旧计划 schema 或哈希不匹配在预检时失败。
Python runtime 仍在加载 `.so` 后注册 `_sanitize_device_counts` 实现。
目前 CPU 测试只能验证约定和边界，实际 Triton 编译、图回放、显存占用及吞吐收益均需目标 NPU 验收。

### 启动和提交开销

同一模型实例、同一 worker 的配置和 DSpark 检查只在首次成功时发布结果，后续层复用；改变关键执行配置需重新构造模型。
原生 `.so/.o` 校验、schema 注册及初始化按该实例和 NPU 设备复用。失败不缓存，新进程或新实例重新验证，
每层仍独立持有权重与计划，不建立全局 Python 缓存。

原生计划在初始化时验证六个权重张量的 dtype、形状、尺寸与相互别名，并持有独立张量元数据及共享底层存储。
执行接口只接收动态输入、路由、计划与工作区，不再重复传入六个权重参数。
每次提交仍检查动态 shape、device、stream、输入/工作区重叠及与权重的重叠，并拒绝底层权重存储被移动或缩小。
计划和提交回调共同保持权重存活；这不能替代权重加载后禁止修改、图存活期间禁止重分配的约定。

继续保留每次执行的 output 清零和 middle scale 初始化，以及每次 eager 独立输出、每图独立工作区。
跨调用 eager 缓冲复用、跨图共享、减少初始化和 tile 调参需根据真实 profile 选择，不能仅凭减少代码量认定提速。

## DP-EP 通信与容量

启动器 `tools/serve_vq2a8.py` 默认保留 `--parallel-mode tp-ep`，`--ep-size 4` 表示 TP4+EP4、DP1。
选择 `--parallel-mode dp-ep --ep-size 4` 则表示 TP1+DP4+EP4，共使用四张卡。
DP 路径允许 MRV2、单节点、内部 MP 调度、DP2/DP4/DP8，保留原有 MRV1 的 TP/EP 路径。
外部或混合负载均衡、多节点、TP 与 DP 同时大于 1、EPLB、CP 和异步调度仍不开放。

DP 模式复制 Attention 和共享专家等普通层，路由专家按 EP rank 连续分片。
DP8+EP8 时 256 个路由专家每卡保存 32 个；普通层按八个 engine 复制。
source-v3 checkpoint 和原生投影格式不变。MoE 前按统一容量填充并收集各 DP engine 的 hidden states、
router logits 和 hash 路由 token IDs；每卡只计算本地专家，经 DP reduce-scatter 后恢复各 engine 的原始 token 顺序。
空闲 engine、不同长度请求和部分请求结束仍需要正确参与通信，不能跳过这些 rank 的 collective。

`--max-num-seqs`、`--max-num-batched-tokens` 均是每个 engine 的限制，默认序列上限仍为 1。
decode 图按本地序列上限生成图桶；专家运行时预留的路由容量为 `DP数 × 最大本地图桶 × top-k`，不除以 EP 数。
例如 DP4、本地 B16、top-k=6 时每个专家 bank 预留 384 行，服务全局活动序列上限为 64。
DP1 的原容量公式保持不变。prefill 使用实际调度 token 预算，不沿用 decode 图容量。

当前主线使用[四卡吞吐测试](packed_weight_batch_throughput.md)。下表仅记录框架可配置的其他拓扑，
不属于新模型的默认测评范围，实际容量及数值仍需对应硬件验证：

| 配置 | 服务端和客户端共同修改的参数 | 服务端 token 预算 |
| --- | --- | --- |
| TP4+EP4 | `--parallel-mode tp-ep --ep-size 4 --max-num-seqs 64` | `--max-num-batched-tokens 8192` |
| TP1+DP4+EP4 | `--parallel-mode dp-ep --ep-size 4 --max-num-seqs 16` | `--max-num-batched-tokens 2048` |
| TP8+EP8 | `--parallel-mode tp-ep --ep-size 8 --max-num-seqs 64` | `--max-num-batched-tokens 8192` |
| TP1+DP8+EP8 | `--parallel-mode dp-ep --ep-size 8 --max-num-seqs 8` | `--max-num-batched-tokens 1024` |

八卡使用 `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`。
表中配置的全局活动序列上限均为 64，扫描并发 1～64，保持输入输出长度、请求数和重复次数一致，并更新测试备注。
DP8 每 engine B64 则对应全局配置上限 512；省略客户端 `--concurrency` 时会扫描到 512，默认每档请求数为当前并发的 4 倍，最高档为 2048 个。
这些是配置上限，实际容量和吞吐以测量为准。
停止前一套服务、释放显存后再启动另一套服务。
相同 KV 字节预算不保证相同有效 token 容量，DP 复制普通层后的权重与图内存也需重新测量。
DP8 已补齐 CPU 通信、容量和启动参数测试，尚未完成八卡 NPU 真权重验收；应先执行
[独立 DP 整模验收](vq2a8_npu_ci.md#dp-ep-整模型验收)，再测吞吐。该验收不加入默认一键 CI。

## 在目标主机上构建

在目标 Ascend 系统的 Linux 环境中使用相互匹配的 CANN、`torch` 和 `torch_npu`，并先加载对应 CANN 安装目录的环境。原生算子和 Torch 桥接层必须来自相互兼容的源码。

Linux `aarch64` 和 `x86_64` 宿主共用这套构建流程与算子源码。设备端仍编译为 Ascend 950 的
`dav-c310-cube/vec`；宿主程序和 Torch 桥接 `.so` 则使用本机编译器及对应架构的 CANN/Torch 库。
桥接构建沿用主项目的 `${CMAKE_SYSTEM_PROCESSOR}-linux` 路径规则，构建清单记录宿主系统和架构。
`devlib` 占位库目录仅用于链接，生成 `.so` 的 RPATH 显式保留 TorchNPU、CANN 和驱动运行库路径。
[BiSheng 编译说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta2/compiler/BishengCompiler/atlas_bisheng_10_0015.html)
也区分了宿主平台与设备 AI Core 编译目标。

切换到 x86 机器时安装 x86_64 的 CANN、Torch、TorchNPU 和 Triton/BiSheng，并从干净的源码检出重新构建。
不要复制 ARM 的 `.so`、宿主可执行文件或 CMake 缓存；`pip install -e .` 的主构建也需要在目标机器重新执行。
主构建日志中的 AICPU ARM 交叉编译器用于 NPU 设备侧，仍需保留。
以下命令在目标机生成完整运行目录；`ASCEND_TOOLKIT_HOME` 指向该机器已加载的 CANN，例如 `/usr/local/Ascend/cann-9.2.0`。
本次设备 ABI 升级为 v4，数学逻辑不变；必须从当前源码重建 `.o`、`.so` 与清单。

```bash
# Run from the repository root. ASCEND_TOOLKIT_HOME names the loaded toolkit.
export BUILD="$PWD/build/vq2a8-gs32-runtime-ffn"
bash csrc/mixed_w2w4_moe_ffn/run_mixed_w2w4_moe_ffn_esl.sh

python tools/build_vq2a8_gs32_stage.py \
  --cann "${ASCEND_TOOLKIT_HOME:?Set the CANN toolkit path}" \
  --kernel-binary "$PWD/csrc/mixed_w2w4_moe_ffn/mixed_w2w4_moe_ffn.o" \
  --build-dir "$BUILD"

file "$BUILD/libvq2a8_gs32_stage.so" "$BUILD/mixed_w2w4_moe_ffn.o"
```

Shell 脚本编译 AIC/AIV 代码并链接生成算子 ELF。Python 工具基于已有 ELF **仅编译 Torch 桥接层**，在新进程中校验注册的算子 schema，并将 ELF 复制到 `.so` 所在目录。输出包括 `$BUILD/libvq2a8_gs32_stage.so`、`$BUILD/mixed_w2w4_moe_ffn.o` 和 `build-manifest.json`。

x86 上 `.so` 应为 x86-64 ELF；`.o` 是 AI Core 设备程序，不要求显示为 x86-64。
只有设备目标、算子源码与 CANN 版本相兼容时才可复用已有 `.o`，上述命令默认在目标机重新构建它。
服务启动传入 `--gs32-runtime-dir "$BUILD"`；pytest 的两个独立产物参数保持不变。当前仅验证了构建逻辑，x86+950 的实际编译、装载、数值与图回放仍待目标机验收。

schema 校验只能确认接口可用，不能证明 NPU 数值正确性或图回放正确性。不要将生成的二进制文件或构建日志提交到 Git。使用 `--dry-run` 可检查构建输入，不执行编译或写入文件。

### 在 ARM 主机交叉编译 x86_64 桥接库

上一节是目标机上的本机构建。ARM 上为 x86+950 生成 `.so` 时，显式指定 `--target-arch x86_64`，
并通过 `--toolchain-file` 复用标准 [CMake 交叉编译配置](https://cmake.org/cmake/help/latest/manual/cmake-toolchains.7.html#cross-compiling-for-linux)。
仅指定处理器名称不能使 ARM 的本机 `g++` 生成 x86 指令。需要准备：

- 能在 ARM 主机上运行、生成 x86_64 Linux ELF 的交叉编译器。
- 与目标机系统匹配的 x86_64 sysroot，包含 libc、libstdc++ 等开发文件和依赖库。
- 目标 x86 机器对应版本的 Torch、TorchNPU、CANN 和驱动头文件/库。它们作为 SDK 文件放在 ARM 主机上，不能用 ARM 已安装的包代替。

Torch 包必须保留 `include`、`lib`、`share/cmake`；TorchNPU 保留 `include` 和 `lib`。
可使用目标机器的软件包文件树或解包对应 x86 wheel；不要在 ARM Python 中导入或安装 x86 wheel。
SDK 中的绝对符号链接必须解析到目标库，不能意外指向 ARM 的 `/usr/local`。
以下路径为示例，需替换为实际交叉工具链和目标 SDK 路径。

先在 ARM 主机准备 `x86_64-linux.cmake`：

```cmake
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR x86_64)
set(CMAKE_CXX_COMPILER /opt/cross/bin/x86_64-linux-gnu-g++)
set(CMAKE_SYSROOT /opt/x86-sysroot)
set(CMAKE_FIND_ROOT_PATH /opt/x86-sysroot)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
```

在 ARM 主机运行同一个构建入口。`--kernel-binary` 指向已有、与目标 Ascend 950/CANN 兼容的 AI Core ELF：

```bash
SDK=/opt/x86-sysroot
TARGET_SITE="$SDK/usr/local/python3.11.10/lib/python3.11/site-packages"
BUILD="$PWD/build/vq2a8-gs32-cross-x86_64"

python tools/build_vq2a8_gs32_stage.py \
  --target-arch x86_64 \
  --toolchain-file "$PWD/x86_64-linux.cmake" \
  --torch-cmake "$TARGET_SITE/torch/share/cmake" \
  --torch-npu-path "$TARGET_SITE/torch_npu" \
  --cann "$SDK/usr/local/Ascend/cann-9.2.0" \
  --driver "$SDK/usr/local/Ascend/driver" \
  --kernel-binary "$PWD/build/vq2a8-gs32-runtime-ffn/mixed_w2w4_moe_ffn.o" \
  --build-dir "$BUILD"
```

该分支不会导入 ARM 主机的 Torch/TorchNPU；目标 Torch 的 CMake 配置提供对应 C++ ABI 编译选项。
构建前检查目标依赖的 ELF 架构，构建后再次确认 `.so` 为 x86_64；错误目标或已更换的工具链不能复用旧构建缓存。
交叉产物不写入构建主机 SDK 的 RPATH。成功输出 `GS32_STAGE_BUILD=CROSS_COMPILED`，
清单状态为 `built_pending_target_validation`，`host_schema_verified` 与 `graph_api_available` 仍为 `false`。
这只表示交叉构建完成，ARM 主机不能据此确认 x86 库的实际加载和 NPU 执行。

将 `$BUILD` 中的 `.so`、`.o` 和构建清单复制到 x86 机器，并在该机器的匹配 Python/CANN 环境中、仓库根目录运行已有 schema 检查：

```bash
python - /path/to/x86-artifacts/libvq2a8_gs32_stage.so <<'PY'
import sys
import torch
import torch_npu
from tools.build_vq2a8_gs32_stage import register_counts_implementation, verify_registered_device_schemas

torch.ops.load_library(sys.argv[1])
register_counts_implementation()
print(verify_registered_device_schemas(torch))
PY
```

通过后再执行下节的 NPU 数值和图回放测试。交叉编译仅覆盖 GS32 宿主桥接库；整套 vLLM Ascend 安装、
其他自定义算子和 x86 Triton/BiSheng 仍需目标侧匹配环境。本地测试只验证构建逻辑，尚未完成 ARM→x86+950 的真实交叉编译及目标机验收。

## 验证

CPU 接口与格式约定测试使用真实的 CPU Torch 运算，以及独立的 NumPy/`ml_dtypes` 参考计算。测试将选定模块与引擎初始化隔离，不以模拟执行代替 NPU 验证。

```bash
python -m pip install pytest torch numpy regex safetensors ml_dtypes
python -X utf8 tools/run_vq2a8_cpu_tests.py
```

在目标 NPU 上，以下测试覆盖初始激活量化、设备路由、真实专家完整 FFN 数值及图捕获回放。
`$MODEL` 必须指向 source-v3 混合 checkpoint；组件测试读取选定专家，不启动整模型服务。

```bash
ASCEND_RT_VISIBLE_DEVICES=4 python -m pytest -vv -s -x \
  tests/e2e/nightly/test_vq2a8_gs32_activation.py \
  tests/e2e/nightly/test_vq2a8_gs32_device_routes.py \
  tests/e2e/nightly/test_vq2a8_gs32_full_ffn.py \
  tests/e2e/nightly/test_vq2a8_gs32_resident_plan.py \
  --vq2a8-model "$MODEL" \
  --vq2a8-gs32-library "$BUILD/libvq2a8_gs32_stage.so" \
  --vq2a8-gs32-binary "$BUILD/mixed_w2w4_moe_ffn.o"
```

完整 FFN 使用测试专用官方 GMM oracle；测试保留动态路由、热点、空 rank、无效输入恢复及多个图桶。
图验收还会输出 `VQ2A8_GS32_FULL_FFN_TIMING`：6/384 行各预热 3 次、测量 5 次，记录 eager 与 graph 的
NPUEvent 原始耗时和中位数，排除输入拷贝及 owner 状态读回。该数据用于同环境版本对比，不设固定时延阈值，
也不能替代整模型吞吐或端到端延迟。
完整 FFN 文件新增 `test_gs32_full_ffn_full_ep_inventory_profile`，在真实 checkpoint 中选择一个 EP4 rank 的完整本地专家库存，
测量 6/384 routed rows 的 eager/graph，以及 4096 rows 的 eager prefill。覆盖空路由、均匀路由、各精度组合热点与倾斜路由；
记录捕获耗时、已分配/保留/峰值显存、capture owner 存储、Host 提交与设备事件耗时，并交替回放两个图桶 100 次检查地址、所有权与活跃显存稳定。
输出 `full-ep-ffn-profile.json`，同一 JSON 以 `VQ2A8_GS32_FULL_EP_PROFILE=` 打印到测试日志。
一键 CI 校验报告与本轮模型、产物的关联后，将报告另存到结果目录，并在汇总中记录路径和 SHA256。
这是单卡上真实 EP 本地库存的测量，不包含 HCCL 和整模型延迟；独立数值 oracle 仍由三种真实精度组合用例承担。
比较优化前后时，固定服务器、卡、checkpoint、路由、图桶和 CANN，分别重建各版本产物并保留哈希；
随后按吞吐文档用相同配置重复测试 TTFT、TPOT 和吞吐，不能把减少 Host 检查直接等同于图回放提速。

整模型多卡、B64、答题精度和服务性能分别按 [NPU CI](vq2a8_npu_ci.md)、[精度](vq2a8_accuracy.md)
和[吞吐](packed_weight_batch_throughput.md)指南验证。CPU 测试和历史结果不能代替本次目标环境验收。
