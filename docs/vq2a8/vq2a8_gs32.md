# GS32 VQ2/MXFP4 混合量化推理

本集成通过 vLLM 标准 safetensors 加载器和 Ascend 量化方案注册机制，
加载 `vq2_mixed_gs32_source` 版本 3 的 checkpoint。
每个路由专家的各个投影独立选择 VQ2 或 MXFP4。
Attention、共享专家和其他模型组件保留 checkpoint 中的数据类型，使用常规模型实现。

本集成面向 Ascend950 上的 DeepSeek-V4，支持 Model Runner V1 和 V2。
代码提取自开发提交 `592a69733`；框架行为基于 `0a8761f07`，
V2 适配来自 `d4b4e6157`。本次迁移不包含实验性的固定 I32 prefill 图。

相关文档：[底层运行时](vq2a8_gs32_runtime.md) ·
[混合专家权重交付规范建议](vq2_mixed_gs32_export_spec_20261016.md)。

## 支持的执行模式

| 配置 | 行为 |
| --- | --- |
| `serial` | 严格 eager 参考路径，按专家执行分阶段投影 |
| `batched` | eager 执行，共享激活准备和启动计划 |
| `fused` | 按各专家实际路由行数执行 eager 分组投影 |
| `full_graph` | B1 decode 图回放；prefill 使用 eager 设备侧路由 |
| `--gs32-fusion none` | 独立执行激活准备 |
| `--gs32-fusion swiglu_quant` | 可选的 SwiGLU 与 Down 激活准备融合 |

普通层使用张量并行；路由专家使用整专家 EP，并按线性方式放置。
支持的并行规模为 1、2、4、8。每个 EP rank 保存完整的本地专家参数，
公共 MoE 流程负责合并各 rank 的部分结果。
由于 VQ2 的置换和缩放参数因专家而异，激活在分发过程中保持 BF16。

分组原生路径没有额外施加任意的 Python 侧单专家行数 M 或总路由行数 R 上限。
调度器仍通过 `max_num_batched_tokens` 限制每个 prefill 步骤的规模，
内存分配也必须满足设备可用显存的约束。
图执行仅捕获一个 decode token，最多包含八个路由行。
增大 prefill chunk 不会扩大该 decode 图。

配置检查会在专家内存分配之前拒绝不支持的组合，包括：专家参数按张量切分、
PP/DP/CP 大于 1、EPLB、LoRA、推测解码、KV 传输、异步调度、offload 和 sleep mode。
完整 decode 图要求同步调度、单序列、捕获尺寸 `[1]`、`FULL_DECODE_ONLY`，
以及启动器提供的文档所述 Ascend 图配置。
`full_graph_prefill` 不是可接受的执行模式。

## 构建与 checkpoint 准备

按照[混合算子集成说明](../../csrc/mixed_w2w4_moe_ffn/INTEGRATION.md)
和[运行时构建说明](vq2a8_gs32_runtime.md)构建原生混合算子及 Torch 桥接库。
必须显式提供生成的 `libvq2a8_gs32_stage.so` 和 `mixed_w2w4_moe_ffn.o`，
不依赖开发者本地的固定构建目录。

**仅构建 GS32 桥接库不足以运行完整的 DeepSeek-V4 模型。**
还需要针对同一目标软件栈构建并安装完整的 `vllm_ascend_C` 扩展及其 ACLNN 算子。
本集成还包含 SAS 和 QLI 的 AICPU 元数据修正，用于验证固定输出布局并清空未使用的元数据槽位。
迁移框架时需要重新构建这些自定义算子。
每个 worker 在选择设备后，都会注册并检查所需的原生接口绑定。

已有的混合 source-v1 交付 checkpoint 可以转换为 source-v3：

```bash
python tools/convert_vq2_mixed_gs32_v3.py \
  --source "$SOURCE_MODEL" --output "$MODEL" --dry-run
python tools/convert_vq2_mixed_gs32_v3.py \
  --source "$SOURCE_MODEL" --output "$MODEL"
```

转换工具要求显式指定源目录和一个新的输出目录。
它会验证原始源文件清单，保留非路由专家的张量，并写出按专家独立命名的张量。
运行时仅接受 source-v3；旧版产物布局和开发诊断加载器不属于生产运行依赖。
加载时会检查索引与文件头的一致性、模型主体张量（非路由专家张量）的数据类型、张量重复或缺失，
以及本地专家归属。索引中的分片必须处于一致的目录深度，以匹配标准加载器的文件路径模式。

## 测试指南

三类测试分别使用独立指南。所有命令均在仓库根目录执行，
各指南说明所需输入、覆盖范围及结果格式。

| 测试类型 | 指南 | 服务启动方式 |
| --- | --- | --- |
| NPU CI 回归 | [VQ2A8 NPU CI](vq2a8_npu_ci.md) | 在指定的空闲卡上自行加载模型 |
| 服务性能 | [VQ2A8 性能测试](vq2a8_performance.md) | 提供选卡和启动服务命令，随后执行 HTTP 延迟测试 |
| 回答精度 | [VQ2A8 精度评测](vq2a8_accuracy.md) | 对已有 HTTP 服务执行 GPQA Diamond / AIME24 / AIME25 评测 |

### 启动并测试服务

[性能测试快速开始](vq2a8_performance.md#one-command-performance-suite)
提供完整的服务端和客户端命令，包括物理 NPU 选卡、Model Runner V1/V2 选择，
以及供 `standard` 套件统一使用的 256K 服务配置。

### 一键性能测试

[性能测试指南](vq2a8_performance.md)涵盖短输入、64K/256K/1M、上下文容量、
预检及 TTFT/TPOT 报告，并区分完整 1M 输入用例与输入较短的 `1m-window` 用例。

### 正确性与回归验证

NPU 数值回归、服务延迟和模型回答精度是独立的检查项。
数据集选择、离线数据包、原生评分和 baseline 分数对比见[精度评测指南](vq2a8_accuracy.md)。

### 一键 NPU 回归

[NPU CI 指南](vq2a8_npu_ci.md)介绍 `kernels`、`pr`、`nightly` 套件，
两种 Model Runner、产物要求、日志和通过条件。
如果 checkpoint 中不存在真实的 `mxfp4 -> vq2` GateUp 到 Down 专家组合，
默认命令会打印提示、记录未覆盖用例，并自动继续执行剩余阶段，无须额外添加命令行选项。
其他跳过、xfail 和实际失败仍会导致 CI 失败。
具体适用范围及其他缺失组合的可选处理方式，见
[checkpoint 缺少真实专家组合时](vq2a8_npu_ci.md#checkpoint-缺少真实专家组合时)。
存在覆盖缺口但其余检查成功的运行会报告为 `passed_with_coverage_gaps`，
报告中仍会保留这些覆盖缺口。

### 单项测试入口

CPU 和单项 NPU 测试命令见 [NPU CI 指南](vq2a8_npu_ci.md)，
可用于套件失败后的针对性诊断。

## Model Runner V2

进程启动前，通过上游已有的环境变量 `VLLM_USE_V2_MODEL_RUNNER=1` 选择 V2；
设置为 `0` 则运行 V1 对照。启动器会保留该选择；本集成不改变上游默认值，
也不引入新的 runner 选择开关。
两种 runner 使用相同的 checkpoint、原生产物和 GS32 执行选项。
在 `full_graph` 模式下，prefill 仍以 eager 方式运行，仅捕获 B1 decode。

两种 runner 使用相同的有效性检查职责划分：每个专家层写入固定的设备状态视图，
由一次汇总检查在采样前拒绝无效输出。
V2 在每次 forward 前重置这些视图，在生成路径中提交 LM Head 计算后、采样前汇总检查有效性，
并在 dummy/profile forward 返回前完成检查。
GS32 forward 若被中断，会清除待处理状态，避免后续采样调用使用不完整结果。
图捕获预热在捕获区域外验证输出；捕获期间的执行仅记录设备写入，不在主机端读取设备状态。

V2 复用 worker 已有的启动 prefill 入口。
它在 runner 空闲时临时借用非空 KV 页，强制使用 eager 执行，
并在成功或失败后恢复借用的页面、输入缓冲区、块表和 attention 元数据，
避免启动预热数据残留在第一个真实请求的缓冲区中。
服务执行路径不会因此增加逐层的设备到主机检查。

使用 `VLLM_USE_V2_MODEL_RUNNER=1` 运行 [NPU CI 指南](vq2a8_npu_ci.md)中的
同拓扑 eager/decode 图对比测试。
然后通过短请求、16K 请求和恢复请求，验证每个 rank 实际使用的 runner 及 decode 回放：

```bash
ASCEND_RT_VISIBLE_DEVICES=4,5 VLLM_USE_V2_MODEL_RUNNER=1 \
python -m pytest -vv -s -x \
  tests/e2e/nightly/test_vq2a8_gs32_ep_long_context.py -k ep2 \
  --vq2a8-model "$MODEL" \
  --vq2a8-gs32-library "$BUILD/libvq2a8_gs32_stage.so" \
  --vq2a8-gs32-binary "$BUILD/mixed_w2w4_moe_ffn.o"
```

报告包含各 worker 的 runner 类、有效性状态管理对象数量和实际 decode 回放次数。
V2 回放计数器仅由测试 RPC 安装，不属于生产钩子。
如果配置声明使用 V2，但实际运行 V1，该探测会失败。

进行延迟对比时，按照[性能测试指南](vq2a8_performance.md)选择 V2 并重启服务，
保持其他设置一致，重复执行短输入 128→128 和长输入 256K→1024 用例。
启动耗时、首个请求的 TTFT、稳态 TTFT/TPOT 应分别比较。
本地 CPU 测试覆盖生命周期和失败处理约定；NPU 编译、图执行、数值一致性和服务性能
必须在目标软件栈上验证。本次适配没有给出实测加速结论，也未解决已记录的长 prefill 性能回退。
