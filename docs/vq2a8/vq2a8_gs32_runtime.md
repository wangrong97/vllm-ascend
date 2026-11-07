# GS32 底层运行时

本层将独立的 W2/W4 混合算子接入 Torch 张量，提供原生投影桥接、激活预处理和设备侧路由元数据功能。
当前分支还包含模型加载、量化方案注册、专家权重组调度、EP 通信及模型运行器接入；下文区分底层接口与整模型验收。

## 来源与接口

本实现从 `feat/test` 分支的提交 `592a697336e74e2defd9f8920d64793c4f2dac56` 中选取。原生算子源码由前置算子 PR 提供，本次运行时接入不修改这些源码。

- `csrc/vq2a8_gs32_stage/torch_binding.cpp` 注册 `torch.ops.vq2a8_gs32_stage` 以及不透明的投影计划、路由句柄。它在提交所用的流上保留张量所有者，校验张量尺寸关系与别名，并复用独立的 DownW2 投影阶段，实现自然顺序的 GateUp 和 Down 投影。
- 严格校验接口 `prepare_routes` 接收 CPU 上的计数，并显式上传由其持有的元数据。`prepare_device_routes` 接收设备上的计数，返回设备侧有效性状态，并支持延后校验。调用方必须保留该状态，并在接受结果前完成检查。
- 分组投影使用实际路由计数，不再受历史上的单专家 4096 行、合计 32768 行上限约束。可用内存、整数尺寸范围及原生算子的专家数量上限仍然适用。串行参考实现仍保留每次启动 128 行的接口约束。
- 激活算子直接使用自然输入列顺序，保持 BF16 舍入、K32 E8M0 缩放因子及 FP8 字节语义；不接收权重列置换参数。路由算子处理计数或累积结束位置、混合类型子集、无活跃路由的专家、填充、无效元数据以及逻辑行顺序恢复。
- `vq2a8_gs32_format.py` 定义 source-v3 和原生张量格式约定；`vq2a8_gs32_staged.py` 定义数值参考实现。`w2a8` 包的初始化文件有意保持为空，不执行模型量化方案注册。

算子支持安全的图捕获，并不意味着整模型图执行已通过验收。框架接入中的 `full_graph` 模式采用 decode 图回放和 eager 分块 prefill，不包含固定长度 prefill 图实验。

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
以下命令将桥接层与部署产物按宿主架构分开输出；在 x86 机器上 `uname -m` 应为 `x86_64`，`ASCEND_TOOLKIT_HOME` 指向该机器已加载的 CANN，例如 `/usr/local/Ascend/cann-9.2.0`。

```bash
# Run from the repository root. ASCEND_TOOLKIT_HOME names the loaded toolkit.
BUILD="$PWD/build/vq2a8-gs32-runtime-$(uname -m)"
bash csrc/mixed_w2w4_moe_ffn/run_mixed_w2w4_moe_ffn_esl.sh

python tools/build_vq2a8_gs32_stage.py \
  --cann "${ASCEND_TOOLKIT_HOME:?Set the CANN toolkit path}" \
  --kernel-binary "$PWD/csrc/mixed_w2w4_moe_ffn/mixed_w2w4_moe_ffn.o" \
  --build-dir "$BUILD"

file "$BUILD/libvq2a8_gs32_stage.so" "$BUILD/mixed_w2w4_moe_ffn.o"
```

Shell 脚本编译 AIC/AIV 代码并链接生成算子 ELF。Python 工具基于已有 ELF **仅编译 Torch 桥接层**，在新进程中校验注册的算子 schema，并将 ELF 复制到 `.so` 所在目录。输出文件为 `$BUILD/libvq2a8_gs32_stage.so` 和 `$BUILD/mixed_w2w4_moe_ffn.o`。

x86 上 `.so` 应为 x86-64 ELF；`.o` 是 AI Core 设备程序，不要求显示为 x86-64。
只有设备目标、算子源码与 CANN 版本相兼容时才可复用已有 `.o`，上述命令默认在目标机重新构建它。
后续测试和服务启动应显式传入这两个新路径。当前仅验证了构建逻辑，x86+950 的实际编译、装载、数值与图回放仍待目标机验收。

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
  --kernel-binary "$PWD/build/vq2a8-gs32-runtime/mixed_w2w4_moe_ffn.o" \
  --build-dir "$BUILD"
```

该分支不会导入 ARM 主机的 Torch/TorchNPU；目标 Torch 的 CMake 配置提供对应 C++ ABI 编译选项。
构建前检查目标依赖的 ELF 架构，构建后再次确认 `.so` 为 x86_64；错误目标或已更换的工具链不能复用旧构建缓存。
交叉产物不写入构建主机 SDK 的 RPATH。成功输出 `GS32_STAGE_BUILD=CROSS_COMPILED`，
清单状态为 `built_pending_target_validation`，`host_schema_verified` 与 `graph_api_available` 仍为 `false`。
这只表示交叉构建完成，ARM 主机不能据此确认 x86 库的实际加载和 NPU 执行。

将 `$BUILD` 中的 `.so`、`.o` 和构建清单复制到 x86 机器，并在该机器的匹配 Python/CANN 环境中、仓库根目录运行已有 schema 检查：

```bash
python -c 'import sys; from tools.build_vq2a8_gs32_stage import verify_loaded_library; print(verify_loaded_library(sys.argv[1]))' \
  /path/to/x86-artifacts/libvq2a8_gs32_stage.so
```

通过后再执行下节的 NPU 数值和图回放测试。交叉编译仅覆盖 GS32 宿主桥接库；整套 vLLM Ascend 安装、
其他自定义算子和 x86 Triton/BiSheng 仍需目标侧匹配环境。本地测试只验证构建逻辑，尚未完成 ARM→x86+950 的真实交叉编译及目标机验收。

## 验证

CPU 接口与格式约定测试使用真实的 CPU Torch 运算，以及独立的 NumPy/`ml_dtypes` 参考计算。测试将选定模块与引擎初始化隔离，不以模拟执行代替 NPU 验证。

```bash
python -m pip install pytest torch numpy regex safetensors ml_dtypes
python -X utf8 tools/run_vq2a8_cpu_tests.py
```

在目标 NPU 上，以下测试覆盖激活与路由边界条件、原生分组投影、非默认流、无效路由恢复以及图捕获与回放。`$MODEL` 必须指向 source-v3 格式的混合 GS32 检查点。测试只读取选定的专家张量，不创建推理服务引擎。

```bash
ASCEND_RT_VISIBLE_DEVICES=4 python -m pytest -vv -s -x \
  tests/e2e/nightly/test_vq2a8_gs32_activation.py \
  tests/e2e/nightly/test_vq2a8_gs32_activation_async.py \
  tests/e2e/nightly/test_vq2a8_gs32_device_routes.py \
  tests/e2e/nightly/test_vq2a8_gs32_grouped.py \
  tests/e2e/nightly/test_vq2a8_gs32_device_native.py \
  --vq2a8-model "$MODEL" \
  --vq2a8-gs32-library "$BUILD/libvq2a8_gs32_stage.so" \
  --vq2a8-gs32-binary "$BUILD/mixed_w2w4_moe_ffn.o"
```

分组投影测试文件支持通过 `large_m and smoke`、`beyond_4096` 和 `beyond_32768` 选择表达式运行范围更小的专项测试。数值断言与投影计时分别进行；这些测试不能证明整模型精度、TTFT 或 TPOT 达标。

本次迁移在 Windows CPU 环境中准备。Linux 编译、NPU 正确性、流行为及图回放必须在目标系统上另行验证；`feat/test` 的历史结果不能替代本次验证。
