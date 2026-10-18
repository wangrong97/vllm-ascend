# VQ2A8 NPU CI 与回归测试指南

导航：[功能与集成说明](vq2a8_gs32.md) · [性能测试](vq2a8_performance.md) · [精度评测](vq2a8_accuracy.md)

本指南用于验证 VQ2A8 相关修改在真实 NPU 上的数值正确性、算子集成、专家并行和图执行回归。
一键入口为 `tools/tests/run_vq2a8_npu_ci.py`，它顺序复用已有 nightly 测试，并为每个阶段创建独立进程。
**测试会自行加载模型，不连接 HTTP 服务。** 使用空闲卡执行；如果服务正在使用卡 4、5，应先停止该服务。

NPU 回归、服务 TTFT/TPOT 和模型答题准确率是三类独立检查。这里通过后，仍需按修改范围进行
[性能测试](vq2a8_performance.md)和 [GPQA Diamond / AIME 精度评测](vq2a8_accuracy.md)。
本入口覆盖 VQ2A8 的选定测试，不等同于整个 vLLM Ascend 项目的全部 CI。

## 运行条件

所有命令均从**当前待验证代码的仓库根目录**执行，使用已安装匹配版本 vLLM、PyTorch、
`torch_npu`、Triton 和 vLLM Ascend 的推理环境。

- 真实运行需要 Linux 和 Ascend950；`kernels` 使用一张卡，`pr`、`nightly` 使用两张卡。
- `MODEL` 沿用已准备好的 **DeepSeek-V4 mixed GS32 source-v3 checkpoint 的绝对路径**。
  目录中必须有 `config.json` 和 `model.safetensors.index.json`；旧版 source/artifact 布局不能替代 source-v3。
- `BUILD` 沿用已有原生构建产物目录的**绝对路径**，其中包含
  `libvq2a8_gs32_stage.so` 和 `mixed_w2w4_moe_ffn.o`。它不是需要新建的固定目录名。
- 整模型测试还需要完整的 `vllm_ascend_C` 扩展及配套 ACLNN 算子，包括 SAS/QLI AICPU metadata 修复。
  仅构建 GS32 bridge 不足以运行整模型；更新相关源码后应重建对应原生产物。
  构建与 checkpoint 说明见[集成文档](vq2a8_gs32.md#build-and-checkpoint-prerequisites)。
- checkpoint 应覆盖测试需要的 VQ2/MXFP4 混合投影组合；默认情况下，缺少组合导致的 pytest skip 会使一键 CI 失败。
  需要先验证其余路径时，可以按下文显式声明缺失组合，并保留覆盖缺口记录。

先确认当前终端中的既有路径：

```bash
: "${MODEL:?请设置已有 source-v3 checkpoint 的绝对路径}"
: "${BUILD:?请设置已有 GS32 原生构建产物目录的绝对路径}"
test -f "$MODEL/config.json"
test -f "$MODEL/model.safetensors.index.json"
test -f "$BUILD/libvq2a8_gs32_stage.so"
test -f "$BUILD/mixed_w2w4_moe_ffn.o"
```

驱动会在子进程中检查 NPU 可用性、可见卡数、Ascend950 型号及 `vllm_ascend` 是否来自当前 checkout。
环境版本和加载路径写入 `environment.log`。驱动主进程不会提前导入 PyTorch 或初始化 NPU。

## 一键执行

日常 PR 回归推荐使用空闲卡 4、5，默认依次验证 MRV1 和 MRV2：

```bash
python tools/tests/run_vq2a8_npu_ci.py \
  --devices 4,5 \
  --model "$MODEL" \
  --gs32-library "$BUILD/libvq2a8_gs32_stage.so" \
  --gs32-binary "$BUILD/mixed_w2w4_moe_ffn.o" \
  --suite pr \
  --result-dir benchmark-results/npu-ci
```

先查看计划时，在同一命令末尾加 `--dry-run`。它只打印命令、设备和 runner 分配，不初始化 NPU，
不创建结果文件，也不执行模型或构建产物可用性检查。

需要包含短请求、16K 请求和恢复请求的图回放检查时，选择 `nightly`：

```bash
python tools/tests/run_vq2a8_npu_ci.py \
  --devices 4,5 \
  --model "$MODEL" \
  --gs32-library "$BUILD/libvq2a8_gs32_stage.so" \
  --gs32-binary "$BUILD/mixed_w2w4_moe_ffn.o" \
  --suite nightly \
  --result-dir benchmark-results/npu-ci-nightly
```

只做组件回归时使用一张空闲卡；此套件仍需要真实 checkpoint 和原生产物：

```bash
python tools/tests/run_vq2a8_npu_ci.py \
  --devices 4 \
  --model "$MODEL" \
  --gs32-library "$BUILD/libvq2a8_gs32_stage.so" \
  --gs32-binary "$BUILD/mixed_w2w4_moe_ffn.o" \
  --suite kernels \
  --result-dir benchmark-results/npu-ci-kernels
```

## checkpoint 缺少真实专家组合时

例如，`batched` 显示 `18 passed, 4 skipped`，四个 skip 的原因均为缺少真实的 `mxfp4 -> vq2` 专家。
这表示 GateUp 使用 MXFP4、Down 使用 VQ2 的同一专家未在 checkpoint 中找到；不是四个数值比较失败。
其中两项检查 `fused/full_graph` 下的激活融合对照，另外两项检查 `none/swiglu_quant` 下的 prefill 对照。
严格模式会在此停止，后续 EP 和 MRV1/MRV2 标为 `not_run`。

若接受该 checkpoint 的已知数据覆盖缺口，希望继续验证其余路径，可显式加入：

```bash
python tools/tests/run_vq2a8_npu_ci.py \
  --devices 4,5 \
  --model "$MODEL" \
  --gs32-library "$BUILD/libvq2a8_gs32_stage.so" \
  --gs32-binary "$BUILD/mixed_w2w4_moe_ffn.o" \
  --suite pr \
  --allow-missing-checkpoint-pairs mxfp4_to_vq2 \
  --result-dir benchmark-results/npu-ci
```

该参数不排除测试，仍然执行全部选定阶段。仅允许 `batched` 中上述两个真实专家测试的确切参数化用例，
且 pytest skip 原因必须与声明组合的“checkpoint 中不存在真实专家”完全一致。
缺少 NPU、依赖、模型参数等其他 skip，以及 xfail、失败、错误、整阶段无实际通过用例，仍会使 CI 失败。
参数支持列出多个组合：`vq2_to_vq2`、`vq2_to_mxfp4`、`mxfp4_to_vq2`、`mxfp4_to_mxfp4`；只声明实际确认缺少的组合。

全部其余检查通过后，汇总为 **`passed_with_coverage_gaps`**，`coverage_complete=false`，退出码为 `0`。
这表示声明范围内的验收通过，不能报告为全组合覆盖通过。原始 `skipped` 计数保留，
`coverage_gaps` 逐项记录阶段、checkpoint、测试 nodeid、组合和原因，终端也实时打印 `COVERAGE_GAP`。
阶段中的 `skipped_tests` 保留全部 skip/xfail 的类型和原因，方便识别未声明的缺口。
后续阶段若失败，总状态仍为 `failed`，之前的缺口记录不会丢失。

补齐含该真实专家组合的 checkpoint 后，移除此参数重新运行严格模式；人工拼接不同专家权重不能替代真实专家覆盖。
内部 CI 若要求完整覆盖，应保持默认严格命令，并要求 `status=passed`、`coverage_complete=true`。
这里的完整覆盖仅指所选套件的必跑用例，不包含未选择的 EP1、长上下文性能或数据集精度评测。

## 三套件的覆盖范围

| 套件 | 设备要求 | 默认覆盖 |
| --- | --- | --- |
| `kernels` | 恰好一张，例如 `--devices 4` | 九个组件测试文件，共九个阶段 |
| `pr`，默认 | 恰好两张，例如 `--devices 4,5` | 九个组件阶段，再运行 MRV1、MRV2 的真实 EP2 eager/decode graph 对照，默认共十一个阶段 |
| `nightly` | 恰好两张 | PR 覆盖加两个 runner 各自的短/16K/短恢复测试，默认共十三个阶段 |

九个组件文件均位于 `tests/e2e/nightly/`，文件名前缀为 `test_vq2a8_gs32_`：

| 文件后缀 | 主要验证内容 |
| --- | --- |
| `activation.py` | 激活准备、量化及动态长度边界 |
| `activation_async.py` | 异步有效性检查 |
| `device_routes.py` | 设备端路由和元数据 |
| `device_native.py` | 原生设备投影 |
| `grouped.py` | grouped 投影、大 M 和尾部边界 |
| `device_bank.py` | 混合权重 bank 的设备执行 |
| `batched.py` | 混合 FFN、prefill 和本地 EP2 容量 |
| `ep_routes.py` | EP 专家归属及路由布局 |
| `ep_attention.py` | attention 相关回归 |

组件阶段始终使用 `--devices` 中的**第一张卡**，并设置 `VLLM_USE_V2_MODEL_RUNNER=0`。
已有参数化测试覆盖大 M、两个本地 EP2 rank bank 的 4096/8192 token 容量，以及适用的两种激活融合设置。
本地构造 EP 形状的 FFN 测试不执行 HCCL；真实 EP2 模型阶段才检查 dispatch、combine、collective 和专家归属。

`--model-runners v1` 或 `--model-runners v2` 可以只选一个 runner，用于聚焦排查；该参数只影响
`pr`、`nightly` 的整模型阶段，不会把九个组件阶段变成 V2 测试。
默认运行两种 runner 时，分别与**同 runner、同 EP2 拓扑的 eager 参考**对比，不是直接比较 V1 与 V2 的数值一致性。

`nightly` 的恢复序列为 `127→4`、`16384→1024`、`127→4`，这里箭头表示输入 token 数到输出 token 数。
测试检查每个 worker 的实际 runner 类型、有效性 owner 和 decode 图回放次数；宣称选择 V2 但实际运行 V1 会失败。
该探针沿用 128-token prefill chunk，**不代表 256K 或 1M 已验收**。

## EP1 参考、超时与退出状态

默认 EP2 阶段不加载可选的 EP1 整模型参考，因为完整 checkpoint 可能无法放入一张卡。
同拓扑 EP2 eager/decode graph 对照仍然必跑；汇总会明确记录未执行 EP1 对照，不把它伪装成已验证。
只有确认整模型能在单卡上运行时，才在一键命令中加入 `--single-card-reference`。
默认不会额外执行独立的 EP1 整模型测试。

`--timeout` 默认是 **7200 秒，每个 pytest 阶段独立计时**，并非整套测试的总时限。
环境预检另有 120 秒超时；现有整模型探针的子进程仍保留 **1800 秒**内部超时。
因此仅增大一键入口的 `--timeout`，不会延长探针自己的内部上限。

退出码及汇总规则：

- 默认模式返回 `0`：所有选定阶段均实际执行至少一个测试，并且无 failure、error、skip 或 xfail。
  驱动同时启用 `xfail_strict=True`，不能用意外通过的预期失败用例掩盖结果。
- 显式声明缺失组合后返回 `0`：其余检查全部通过，可以有符合声明的 checkpoint 数据 skip；
  此时汇总为 `passed_with_coverage_gaps`，不计作完整覆盖通过。若实际没有 skip，仍为普通 `passed`。
- 返回 `1`：环境预检失败、测试失败、超时、零测试，或 JUnit 缺失、格式错误、计数不一致。
  未显式接受的 checkpoint 组合 skip，也属于未完成验收。
- Ctrl+C 或 SIGTERM：保存已完成部分及中断状态，返回 `130`。

出现失败即停止后续阶段，尚未运行的阶段保留为 `not_run`。
阶段退出、超时或取消时，驱动清理该阶段自己创建的进程组及模型 worker，避免测试进程遗留占卡。

## 查看日志并接入内部 CI

每次运行都会创建独立目录，例如：

```text
benchmark-results/npu-ci/vq2a8-npu-ci-<时间戳>-<随机后缀>/
  summary.json
  environment.log
  activation.log
  activation.xml
  activation-tmp/
  ep2-v1.log
  ep2-v1.xml
  ep2-v1-tmp/
  ...
```

终端输出 `RESULT_DIR=...` 和每阶段的 `STAGE=... LOG=...`。
环境预检和 pytest 子进程的 stdout/stderr 默认实时显示在终端，同时完整保存到相应日志文件，
包括用例名称、执行进度、通过/失败信息和异常堆栈；原有启动命令无需增加参数。
没有新输出时，每约 15 秒显示 `PROBE_WAIT=...`、等待时长和日志路径。
也可以在另一终端对打印出的路径执行 `tail -f`。
JUnit XML 保留各用例结果和 skip 原因；pytest 临时目录保留模型子进程的日志及报告。
`summary.json` 记录 checkout commit、实际命令、所选环境覆盖、测试计数、耗时和状态，并在阶段更新时保存。

失败阶段会打印子进程退出码（如果已经取得）和日志末尾，最多读取 16 KiB、显示最后 64 行，
并将其保存到阶段的 `log_tail`。完整日志仍保留在原文件；日志读取失败单独记录，不覆盖原错误。
如果 pytest 在参数解析或加载 `conftest.py` 时退出，可能还没有生成 JUnit XML。
此时汇总会同时记录 `returncode`、`junit_error` 和日志，不能仅根据“缺少 XML”判断是算子精度失败。
例如，对终端打印出的结果目录查看：

```bash
RESULT=/path/to/the/printed/result-directory
sed -n '1,220p' "$RESULT/activation.log"
tail -n 80 "$RESULT/activation.log"
```

可以把上面的一键命令作为内部 Linux NPU CI job 的步骤，归档整个结果目录，使用进程退出码判定 job 成败。
脚本本身不注册 self-hosted runner，也不自动安装 GitHub PR 触发任务。
如果通过 `tee` 保存终端输出，应先执行 `set -o pipefail`，避免管道把失败退出码覆盖掉。

组件测试中的局部计时保留在日志中，但**没有自动性能回归阈值**。
这些数据不能替代 HTTP 服务的 TTFT/TPOT，也不包含所有 attention、通信、调度和请求处理成本。
服务性能对比应固定硬件、checkpoint、原生产物、runner、chunk 和融合设置。

## 单独运行测试与故障定位

CPU 合约测试可以在没有 vLLM 安装和 NPU 初始化的环境运行；仍需相应 CPU 测试依赖：

```bash
python -X utf8 tools/run_vq2a8_cpu_tests.py
python -m pytest --noconftest -o addopts= -q \
  tests/ut/tools/test_convert_vq2_mixed_gs32_v3.py \
  tests/ut/tools/test_vq2a8_long_context_latency_runner.py \
  tests/ut/tools/test_vq2a8_long_context_server_launchers.py \
  tests/ut/attention/test_dsa_kv_quant.py
```

CPU 测试使用隔离的框架测试支架，执行真实 tensor 运算和模型加载方法。
它们不能验证 CANN 编译、HCCL、ACLGraph 或目标硬件性能。

单卡检查混合 FFN prefill 数值和本地 EP2 bank 容量：

```bash
ASCEND_RT_VISIBLE_DEVICES=4 VLLM_USE_V2_MODEL_RUNNER=0 \
python -m pytest -vv -s -x \
  tests/e2e/nightly/test_vq2a8_gs32_batched.py \
  -k 'device_prefill_compare or device_prefill_ep2_capacity' \
  --vq2a8-model "$MODEL" \
  --vq2a8-gs32-library "$BUILD/libvq2a8_gs32_stage.so" \
  --vq2a8-gs32-binary "$BUILD/mixed_w2w4_moe_ffn.o"
```

双卡检查真实 EP2 eager/decode graph 对照及 attention；显式选 EP2 node，避免误收集 EP4/EP8：

```bash
ASCEND_RT_VISIBLE_DEVICES=4,5 VLLM_USE_V2_MODEL_RUNNER=0 \
python -m pytest -vv -s -x \
  'tests/e2e/nightly/test_vq2a8_gs32_ep.py::test_gs32_ep_real_model_eager_and_graphs_match_single_card[ep2]' \
  tests/e2e/nightly/test_vq2a8_gs32_ep_attention.py \
  --vq2a8-model "$MODEL" \
  --vq2a8-gs32-library "$BUILD/libvq2a8_gs32_stage.so" \
  --vq2a8-gs32-binary "$BUILD/mixed_w2w4_moe_ffn.o" \
  --vq2a8-gs32-ep-skip-single-card
```

该单独 pytest 命令也显式省略 EP1 参考。单卡容量足够、需要补做 EP1 对照时，移除最后一个参数。
将 `VLLM_USE_V2_MODEL_RUNNER=0` 改为 `1` 可验证 MRV2 的同拓扑对照。

单独检查 MRV2 的短/16K/短图回放与恢复：

```bash
ASCEND_RT_VISIBLE_DEVICES=4,5 VLLM_USE_V2_MODEL_RUNNER=1 \
python -m pytest -vv -s -x \
  'tests/e2e/nightly/test_vq2a8_gs32_ep_long_context.py::test_gs32_ep_long_context_decode_graph_and_recovery[ep2]' \
  --vq2a8-model "$MODEL" \
  --vq2a8-gs32-library "$BUILD/libvq2a8_gs32_stage.so" \
  --vq2a8-gs32-binary "$BUILD/mixed_w2w4_moe_ffn.o"
```

完整回归应覆盖两个 EP rank、混合投影组合、空专家、padding 行、group-list counts/prefixes、
非默认 stream 和无效输入后的恢复。单独运行 pytest 便于定位问题，但仍须检查 skip，不能只根据 pytest
退出码宣称一键 CI 已通过；正式验收应保留对应套件的完整汇总。
