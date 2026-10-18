# GS32 底层运行时

本层将独立的 W2/W4 混合算子接入 Torch 张量，提供原生投影桥接、激活预处理和设备侧路由元数据功能。模型加载、量化方案注册、专家权重组调度、EP 通信及模型运行器接入由后续独立改动完成。

## 来源与接口

本实现从 `feat/test` 分支的提交 `592a697336e74e2defd9f8920d64793c4f2dac56` 中选取。原生算子源码由前置算子 PR 提供，本次运行时接入不修改这些源码。

- `csrc/vq2a8_gs32_stage/torch_binding.cpp` 注册 `torch.ops.vq2a8_gs32_stage` 以及不透明的投影计划、路由句柄。它在提交所用的流上保留张量所有者，校验张量尺寸关系与别名，并复用独立的 DownW2 投影阶段，实现自然顺序的 GateUp 和 Down 投影。
- 严格校验接口 `prepare_routes` 接收 CPU 上的计数，并显式上传由其持有的元数据。`prepare_device_routes` 接收设备上的计数，返回设备侧有效性状态，并支持延后校验。调用方必须保留该状态，并在接受结果前完成检查。
- 分组投影使用实际路由计数，不再受历史上的单专家 4096 行、合计 32768 行上限约束。可用内存、整数尺寸范围及原生算子的专家数量上限仍然适用。串行参考实现仍保留每次启动 128 行的接口约束。
- 激活算子保持置换、BF16 舍入、K32 E8M0 缩放因子及 FP8 字节的语义一致。路由算子处理计数或累积结束位置、混合类型子集、无活跃路由的专家、填充、无效元数据以及逻辑行顺序恢复。
- `vq2a8_gs32_format.py` 定义 source-v3 和原生张量格式约定；`vq2a8_gs32_staged.py` 定义数值参考实现。`w2a8` 包的初始化文件有意保持为空，不执行模型量化方案注册。

算子支持安全的图捕获，并不意味着模型已具备图执行能力。后续框架接入中的 `full_graph` 模式采用 decode 图回放和 eager 分块 prefill。本次迁移不包含固定长度 prefill 图实验。

## 在目标主机上构建

在目标 Ascend 系统的 Linux 环境中使用相互匹配的 CANN、`torch` 和 `torch_npu`，并先加载对应 CANN 安装目录的环境。原生算子和 Torch 桥接层必须来自相互兼容的源码。

```bash
# Run from the repository root. ASCEND_TOOLKIT_HOME names the loaded toolkit.
bash csrc/mixed_w2w4_moe_ffn/run_mixed_w2w4_moe_ffn_esl.sh

python tools/build_vq2a8_gs32_stage.py \
  --cann "${ASCEND_TOOLKIT_HOME:?Set the CANN toolkit path}" \
  --kernel-binary "$PWD/csrc/mixed_w2w4_moe_ffn/mixed_w2w4_moe_ffn.o" \
  --build-dir "$PWD/build/vq2a8-gs32-runtime"
```

第一条命令编译 AIC/AIV 代码并链接生成算子 ELF。Python 工具基于已有 ELF **仅编译 Torch 桥接层**，在新进程中校验注册的算子 schema，并将 ELF 复制到 `.so` 所在目录。输出文件为 `build/vq2a8-gs32-runtime/libvq2a8_gs32_stage.so` 和 `build/vq2a8-gs32-runtime/mixed_w2w4_moe_ffn.o`。

schema 校验只能确认接口可用，不能证明 NPU 数值正确性或图回放正确性。不要将生成的二进制文件或构建日志提交到 Git。使用 `--dry-run` 可检查构建输入，不执行编译或写入文件。

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
  --vq2a8-gs32-library "$PWD/build/vq2a8-gs32-runtime/libvq2a8_gs32_stage.so" \
  --vq2a8-gs32-binary "$PWD/build/vq2a8-gs32-runtime/mixed_w2w4_moe_ffn.o"
```

分组投影测试文件支持通过 `large_m and smoke`、`beyond_4096` 和 `beyond_32768` 选择表达式运行范围更小的专项测试。数值断言与投影计时分别进行；这些测试不能证明整模型精度、TTFT 或 TPOT 达标。

本次迁移在 Windows CPU 环境中准备。Linux 编译、NPU 正确性、流行为及图回放必须在目标系统上另行验证；`feat/test` 的历史结果不能替代本次验证。
