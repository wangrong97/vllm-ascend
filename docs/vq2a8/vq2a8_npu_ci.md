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
  构建与 checkpoint 说明见[集成文档](vq2a8_gs32.md#构建与-checkpoint-准备)。
- checkpoint 用于验证其中实际存在的 VQ2/MXFP4 混合投影组合。若缺少真实的 `mxfp4 -> vq2` 专家，
  一键 CI 会自动提示、记录覆盖缺口并继续；无需额外参数。其他 skip 和真实测试失败仍会阻断。

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

多卡测试前，还需按[950DT 部署检查](../source/getting_started/installation.md#installation-multi-node-physical)
确认当前运行环境中的 `/etc/hccl_rootinfo.json` 是可读取的 JSON **文件**，
并正确配置本机的 `/lib/route.conf` 和 `/etc/hixlep`。
容器中的挂载目标存在不代表类型正确；若 `ls -ld /etc/hccl_rootinfo.json` 显示目录，
应先检查宿主机配置和容器挂载，不能用空 JSON 或另一台机器的拓扑替代。
单卡 bank/算子通过不代表 HCCL 可用；应先通过相同可见卡配置下的独立 HCCL 通信检查，再加载整模型。
可执行 `python tools/tests/check_vq2a8_hccl.py --devices 4,5`；
具体步骤、报告含义和启动同步定位见 [HcclAllGather 超时排查](vq2a8_startup_hccl.md)。

## 新 repack 权重四卡验收

本节仅检查当前 `-export-20261103` 权重的 TP4/EP4、MRV2 路径，不运行其他卡数或单卡 baseline。
原有两卡入口保持可用。先停止占用所选四张卡的服务，在仓库根目录设置：

```bash
export MODEL=/mnt/share/g00872988/DeepSeek-V4-Flash-0731-32x32-noperm-hot5050125-identity-v3-export-20261103
export BUILD="$PWD/build/vq2a8-gs32-runtime"
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
export TMPDIR=/tmp
export VLLM_RPC_BASE_PATH=/tmp
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

首次使用完整融合须按[ABI v3 重建](packed_weight_batch_throughput.md#0-首次升级完整-ffn重建-abi-v3)一起重建 `.o/.so`。
旧设备对象即使文件名相同也不兼容；预检核对 v3 符号和构建清单 SHA256。
先校验主模型扩展和 native GS32 bridge 的注册接口。此命令不加载权重、不启动服务，也不执行设备算子：

```bash
python tools/serve_vq2a8.py \
  --model "$MODEL" --no-dspark \
  --gs32-backend native \
  --gs32-library "$BUILD/libvq2a8_gs32_stage.so" \
  --gs32-binary "$BUILD/mixed_w2w4_moe_ffn.o" \
  --parallel-mode tp-ep --ep-size 4 \
  --gs32-execution full_graph --gs32-fusion full_ffn \
  --max-model-len 8192 --max-num-batched-tokens 8192 --max-num-seqs 64 \
  --check-native-only
```

正常 native 服务启动也会执行此前置检查；`--dry-run` 仍只检查配置，不加载扩展。
`reference_mxfp8` 不依赖独立 GS32 bridge，但仍需主模型扩展。
接口检查不证明 `.o` 已成功执行或图捕获已通过；若提示重建，按[构建说明](vq2a8_gs32.md#构建与-checkpoint-准备)更新对应产物。

先执行 B1 的同权重 eager/decode graph 对照，覆盖 127/128/129-token 输入边界及恢复请求：

```bash
python -m pytest -vv -s -x -o addopts= \
  'tests/e2e/nightly/test_vq2a8_gs32_ep.py::test_gs32_ep_real_model_eager_and_graphs_match_single_card[ep4]' \
  --vq2a8-model "$MODEL" \
  --vq2a8-gs32-library "$BUILD/libvq2a8_gs32_stage.so" \
  --vq2a8-gs32-binary "$BUILD/mixed_w2w4_moe_ffn.o" \
  --vq2a8-gs32-ep-skip-single-card
```

再做完整 FFN 真实专家对照。此阶段不加载整模型；检查 W2/W2、W2/W4、W4/W4，
以及裁剪后的中间 FP8/E8M0、混精动态计数、零专家/空 rank、128/129/384 行和图重放：

```bash
ASCEND_RT_VISIBLE_DEVICES=4 python -m pytest -vv -s -x -o addopts= \
  tests/e2e/nightly/test_vq2a8_gs32_full_ffn.py \
  --vq2a8-model "$MODEL" \
  --vq2a8-gs32-library "$BUILD/libvq2a8_gs32_stage.so" \
  --vq2a8-gs32-binary "$BUILD/mixed_w2w4_moe_ffn.o"
```

然后执行四卡完整 FFN B64 验收。该用例依次加载 `none+fused`、`full_ffn+fused`、
`full_ffn+full_graph`，比较分步/完整融合及 eager/graph 的同权重结果：

```bash
python -m pytest -vv -s -x -o addopts= \
  tests/e2e/nightly/test_vq2a8_gs32_ep.py::test_gs32_ep4_full_ffn_decode_batch_smoke \
  --vq2a8-model "$MODEL" \
  --vq2a8-gs32-library "$BUILD/libvq2a8_gs32_stage.so" \
  --vq2a8-gs32-binary "$BUILD/mixed_w2w4_moe_ffn.o"
```

覆盖 B1/2/3/5/7/8/16/31/32/63/64、大小 batch 切换以及同批请求先后结束，检查四个 rank 的专家归属、
图桶回放、token 和 logprob。测试使用短上下文、512-token 调度预算及每 worker 4 GiB KV，
用于正确性验收；性能服务仍按吞吐文档使用 8192-token 预算和 16 GiB KV，不能用该测试耗时作性能比较。

以上检查通过后，再按[四卡吞吐](packed_weight_batch_throughput.md)及[四卡精度](vq2a8_accuracy.md)测新目录。
本轮默认 native 使用 `full_ffn`，参考后端仍使用 `none`；本地 CPU 检查不代替上述新模型真机验收。
下方原有一键套件仍可用于组件回归，不能代替本节新增的完整 FFN 和 EP4 对照。
repack 前后的非 routed 量化不同，不能要求两份权重逐 token 一致；eager/graph 对照始终使用同一份权重。

## 一键执行

日常 PR 回归使用空闲卡 4、5，默认运行九个组件阶段和 MRV2 的 `ep2-v2` 整模型阶段，共十个阶段。
一键 CI 默认选择 Model Runner V2；`ep2-v1` 已从自动计划移除，MRV1 的整模型 eager/decode graph 对照仍可手动运行。
下面的命令已自动处理 checkpoint 缺少真实 `mxfp4 -> vq2` 专家的情况，不需要在命令中增加允许跳过的参数。
具体提示与结果含义见[缺失组合处理](#checkpoint-缺少真实专家组合时)。

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
一键 CI **默认自动接受这类缺失组合**，打印提示并继续所选套件的后续阶段，无需修改上面的命令。
只有对应真实专家测试报告该专家组合不存在时，才采用此处理；如果 checkpoint 含有该组合，测试照常执行，失败仍会阻断。

该阶段其余检查通过时，终端会显示类似提示：

```text
STAGE=batched STATUS=passed_with_coverage_gaps
STAGE=batched NOTICE=Checkpoint has no real expert for mxfp4 -> vq2; recorded 4 untested cases and continuing remaining checks. MODEL=/path/to/checkpoint
```

自动处理只适用于 `batched` 中上述两个真实专家测试的四个确切参数化用例，
且 pytest skip 原因必须与“checkpoint 中不存在真实 mxfp4 -> vq2 专家”完全一致。
缺少 NPU、依赖、模型参数等其他 skip，以及 xfail、失败、错误、整阶段无实际通过用例，仍会使 CI 失败。
其他专家组合的缺失默认仍阻断；旧参数 `--allow-missing-checkpoint-pairs` 保留兼容，供明确接受其他组合缺口时使用。
已使用该参数指定 `mxfp4_to_vq2` 的旧命令也继续有效，但现在可以直接移除这一参数。

全部其余检查通过后，汇总为 **`passed_with_coverage_gaps`**，`coverage_complete=false`，退出码为 `0`。
这表示已执行的检查通过，不能报告为全组合覆盖通过。原始 `skipped` 计数保留，
`coverage_gaps` 逐项记录阶段、checkpoint、测试 nodeid、组合和原因，终端也实时打印 `COVERAGE_GAP`。
`automatic_missing_checkpoint_pairs` 记录默认自动处理的组合，`allowed_missing_checkpoint_pairs` 记录自动与显式指定的有效并集。
阶段中的 `skipped_tests` 保留全部 skip/xfail 的类型和原因，方便识别其他缺口。
后续阶段若失败，总状态仍为 `failed`，之前的缺口记录不会丢失。

补齐含该真实专家组合的 checkpoint 后，使用同一命令重新运行；人工拼接不同专家权重不能替代真实专家覆盖。
内部 CI 若额外要求完整覆盖，应检查 `status=passed`、`coverage_complete=true`。
这里的完整覆盖仅指所选套件的必跑用例，不包含已移除的 `ep2-v1`、未选择的 EP1、长上下文性能或数据集精度评测。

## 三套件的覆盖范围

| 套件 | 设备要求 | 默认覆盖 |
| --- | --- | --- |
| `kernels` | 恰好一张，例如 `--devices 4` | 九个组件测试文件，共九个阶段 |
| `pr`，默认 | 恰好两张，例如 `--devices 4,5` | 九个组件阶段，再运行 MRV2 的真实 EP2 eager/decode graph 对照 `ep2-v2`，默认共十个阶段 |
| `nightly` | 恰好两张 | PR 覆盖加 `recovery-v2` 短/16K/短恢复测试，默认共十一个阶段 |

九个组件文件均位于 `tests/e2e/nightly/`，文件名前缀为 `test_vq2a8_gs32_`：

| 文件后缀 | 主要验证内容 |
| --- | --- |
| `activation.py` | 激活准备、量化及动态长度边界 |
| `activation_async.py` | 异步有效性检查 |
| `device_routes.py` | 设备端路由和元数据 |
| `device_native.py` | 原生设备投影 |
| `grouped.py` | grouped 投影、大 M 和尾部边界 |
| `device_bank.py` | 混合权重 bank、多图桶捕获回放及大路由容量 |
| `batched.py` | 混合 FFN、prefill 和本地 EP2 容量 |
| `ep_routes.py` | EP 专家归属及路由布局 |
| `ep_attention.py` | attention 相关回归 |

组件阶段始终使用 `--devices` 中的**第一张卡**，runner 使用 `--model-runners` 中的首个值；
默认是 V2，显式选择 `v1` 或 `v1 v2` 时组件阶段使用 V1。
已有参数化测试覆盖大 M、两个本地 EP2 rank bank 的 4096/8192 token 容量，以及适用的两种激活融合设置。
本地构造 EP 形状的 FFN 测试不执行 HCCL；真实 EP2 模型阶段才检查 dispatch、combine、collective 和专家归属。

`--model-runners` 默认是 `v2`，仍可显式选择 `v1` 或 `v1 v2`；
首个值决定组件阶段使用的 runner，全部所选值决定 `pr`、`nightly` 的整模型阶段；不会重新启用 `ep2-v1`。
九个组件阶段验证各自的算子或组件，不能代替整模型执行验证。

| runner 选择 | `pr` 阶段 | `nightly` 阶段 |
| --- | --- | --- |
| 默认，或 `--model-runners v2` | 九个组件 + `ep2-v2`，共十个 | 九个组件 + `ep2-v2` + `recovery-v2`，共十一个 |
| `--model-runners v1` | 仅九个组件，共九个 | 九个组件 + `recovery-v1`，共十个 |
| `--model-runners v1 v2` | 九个组件 + `ep2-v2`，共十个 | 九个组件 + `ep2-v2` + `recovery-v1` + `recovery-v2`，共十二个 |

**一键 CI 通过不代表 MRV1 的整模型 EP2 eager/decode graph 对照已通过。**
该检查已移出计划，组件阶段和 `recovery-v1` 不能代替它；需要该项验收时，使用下文的手动 EP2 命令。
保留的 `ep2-v2` 与**同 runner、同 EP2 拓扑的 eager 参考**对比，不是直接比较 V1 与 V2 的数值一致性。

`ep2-v2` 及手动 EP eager/decode graph 对照依次使用 `128→4`、`128→4`、`127→4`、`129→4`、`128→4`，
覆盖不同输入、长度边界和恢复请求。测试上下文长度为 `129 + 4 = 133`，为全部输出预留空间；
两个执行模式均开启 chunked prefill，每步预算为 128，图模式仅捕获 B1 decode。
这些参数由测试内部设置，无需修改一键命令，也无需重新编译原生算子。

`nightly` 的恢复序列为 `127→4`、`16384→1024`、`127→4`，这里箭头表示输入 token 数到输出 token 数。
测试检查每个 worker 的实际 runner 类型、有效性 owner 和 decode 图回放次数；宣称选择 V2 但实际运行 V1 会失败。
该探针沿用 128-token prefill chunk，**不代表 256K 或 1M 已验收**。

### 多序列 decode 图验收

MRV2 新增 B1～B64 支持。组件阶段 `device_bank` 捕获并交替回放多个 FFN 路由容量，
包括 top-k=6 的 B64（384 行）和 top-k=8 的 B64（512 行）；
检查动态路由、单专家超过 128 行、空 bank、填充行、无效输入和后续恢复。
该测试复用三专家 fixture，不能代替完整 EP2 模型的多序列验收。

完整模型 batch smoke 是独立手动入口，不加入默认 CI，也不恢复已移除的 `ep2-v1`。
在两张空闲卡上运行：

```bash
ASCEND_RT_VISIBLE_DEVICES=4,5 VLLM_USE_V2_MODEL_RUNNER=1 \
python -m pytest -vv -s -x -o addopts= \
  tests/e2e/nightly/test_vq2a8_gs32_ep.py::test_gs32_ep2_decode_batch_smoke \
  --vq2a8-model "$MODEL" \
  --vq2a8-gs32-library "$BUILD/libvq2a8_gs32_stage.so" \
  --vq2a8-gs32-binary "$BUILD/mixed_w2w4_moe_ffn.o"
```

此用例在 eager 和 decode 图模式下各加载一次 EP2 模型，使用短输入和少量输出，
覆盖 B1、B2、B3、B5、B7、B8、B16、B31、B32、B63、B64、大小 batch 切换，
以及同批请求先后结束。它逐 rank 核对实际图桶回放，并比较同拓扑的 token 和 logprob。
CPU 测试只能验证参数和控制逻辑；新 batch 的数值正确性、实际图回放和内存余量
必须以上述 NPU 测试及真实服务请求为准，不能将配置校验通过当作硬件验收通过。

### DP-EP 整模型验收

TP1+DP2/4/8+EP2/4/8 路径仅面向 MRV2、单节点内部 MP；原有 TP/EP 用例不能替代 DP 通信验收。
以下入口独立运行，不增加默认一键 CI 的阶段，也不修改 MRV1 的保留范围。
以下验收八卡。先停止占用测试卡的服务，使用与服务相同的 source-v3 checkpoint 和原生产物：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 VLLM_USE_V2_MODEL_RUNNER=1 \
python -m pytest -vv -s -x -o addopts= \
  'tests/e2e/nightly/test_vq2a8_gs32_dp.py::test_gs32_dp_eager_and_decode_graphs[dp8]' \
  --vq2a8-model "$MODEL" \
  --vq2a8-gs32-library "$BUILD/libvq2a8_gs32_stage.so" \
  --vq2a8-gs32-binary "$BUILD/mixed_w2w4_moe_ffn.o"
```

四卡可选 `[dp4]` 并将可见卡改为 `0,1,2,3`，两卡可选 `[dp2]` 和 `0,1`。
用例顺序加载三次模型：同卡数 TP/EP eager B1 参考、
TP1+DP/EP eager、TP1+DP/EP `FULL_DECODE_ONLY`，不连接已有 HTTP 服务。
每个 DP engine 的 `max_num_seqs=16`，DP4 对应全局并发上限 64，DP8 对应 128。
均衡波次先暂停调度并将全部请求入队，再恢复执行；必须逐 rank 观测到实际 B16，
且图模式下本波次的 B16 图桶发生回放，否则验收失败。仅客户端发起请求不算覆盖该边界。
其他波次检查每个 rank 单独收到请求时其他 rank 的空闲参与、DP8 的 `[0,1,7,16,0,1,7,16]` 不均衡负载、
不同输入/输出长度以及后续小 batch 恢复，同时核对实际 DP/EP 归属、本地专家分片和路由容量。

数值比较要求 DP eager 与 DP 图的 greedy token 一致、top-5 logprob 绝对误差不超过 `1e-3`；
DP eager 还须与 TP/EP B1 参考逐请求匹配 greedy token，并按已有 EP 验收容差 `0.05`
检查生成 token 的 logprob，避免两条 DP 路径同时出错却互相通过。

这项验收需要真实 NPU；本地 CPU 合同测试不能证明数值、collective、图回放或显存余量已经通过。
通过后再按[多卡对比配置](vq2a8_gs32_runtime.md#dp-ep-通信与容量)执行[吞吐测试](vq2a8_batch_throughput.md)，
不能把 DP 每 engine B64 与 TP 全局 B64 当作同一并发容量。

## EP1 参考、超时与退出状态

默认 `ep2-v2` 阶段不加载可选的 EP1 整模型参考，因为完整 checkpoint 可能无法放入一张卡。
选中 `ep2-v2` 时，同拓扑 EP2 eager/decode graph 对照仍然必跑；汇总会明确记录未执行 EP1 对照，不把它伪装成已验证。
只有确认整模型能在单卡上运行时，才在一键命令中加入 `--single-card-reference`；它只为已选中的 `ep2-v2` 增加参考，
不会重新启用 `ep2-v1`。仅选择 `--model-runners v1` 时没有该 EP 对照阶段。
默认不会额外执行独立的 EP1 整模型测试。

`--timeout` 默认是 **7200 秒，每个 pytest 阶段独立计时**，并非整套测试的总时限。
环境预检另有 120 秒超时；现有整模型探针的子进程仍保留 **1800 秒**内部超时。
因此仅增大一键入口的 `--timeout`，不会延长探针自己的内部上限。

退出码及汇总规则：

- 返回 `0`：所有选定阶段均至少有一个实际通过用例，并且无 failure、error、xfail 或不支持的 skip。
  驱动同时启用 `xfail_strict=True`，不能用意外通过的预期失败用例掩盖结果。
  若仅存在自动接受的 `mxfp4 -> vq2` 缺失，或通过旧参数额外声明的组合缺失，汇总为 `passed_with_coverage_gaps`；
  若实际没有 skip，汇总为 `passed`。两者均不会因为这些已识别的数据缺口报错。
- 返回 `1`：环境预检失败、测试失败、超时、零测试，或 JUnit 缺失、格式错误、计数不一致。
  其他未接受的 skip、xfail 或整阶段全 skip，也属于未完成验收。
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
  ep2-v2.log
  ep2-v2.xml
  ep2-v2-tmp/
  ...
```

终端输出 `RESULT_DIR=...` 和每阶段的 `STAGE=... LOG=...`。
环境预检和 pytest 子进程的 stdout/stderr 默认实时显示在终端，同时完整保存到相应日志文件，
包括用例名称、执行进度、通过/失败信息和异常堆栈；原有启动命令无需增加参数。
没有新输出时，每约 15 秒显示 `PROBE_WAIT=...`、等待时长和日志路径。
也可以在另一终端对打印出的路径执行 `tail -f`。
JUnit XML 保留各用例结果和 skip 原因；pytest 临时目录保留模型子进程的日志及报告。
EP 整模型测试的内层子进程输出单独写入 `ep2-v2-tmp/` 下的 `ep2-fused.log` 和 `ep2-full_graph.log`，
不会透传到外层终端；查看加载、预热或图捕获进度时，应对当前内层日志执行 `tail -f`。
`PROBE_WAIT` 仅表示外层仍在等待，不代表内层模型正在推进。
`summary.json` 记录 checkout commit、实际命令、所选环境覆盖、测试计数、耗时和状态，并在阶段更新时保存。
汇总和 dry-run 的 `model_runners` 只列实际安排整模型阶段的 runner：默认 `pr`、`nightly` 为 `["v2"]`，
`kernels` 和 `pr --model-runners v1` 为 `[]`，`nightly --model-runners v1 v2` 为 `["v1", "v2"]`。
`single_card_reference` 仅在实际包含 EP 对照阶段且指定该开关时为 `true`。

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

### 路由 metadata 内核的 BiSheng 编译失败

如果 `device_bank_decode_graph_buckets_and_recovery[prefix]` 在
`CapturedBank.__init__` 的预热 `bank.apply` 中报
`LLVM ERROR: consolidateToTail produced invalid IR`，此时尚未进入图捕获。
最初首个桶是六个路由行，失败栈指向 `_route_metadata_kernel`，还没有执行该次调用的原生矩阵投影。
这是 BiSheng 后端编译失败，不能据此判断四卡通信、DSpark 或大 batch 数值错误。

在 Ascend950DT_9582 / CANN 9.2.0 / `triton_ascend 3.6.0+dev20261007182438` /
`torch_npu 2.10.0.post7` 上，保留 `tl.static_range(ROWS)` 后仍能复现。
使用缓存的 `_route_metadata_kernel.bcmlir` 和相同编译参数做离线对照：
`--enable-auto-vectorize-v2=true` 退出 134 并报上述错误，设为 `false` 退出 0。
因此 `_route_metadata_kernel` 设置 `enable_auto_vectorize_v2=False`，使用旧向量化路径。
后续实机复测中，六行路由的 prefix/counts 两项通过；bank 测试继续运行到
`_apply_prefill -> prepare_metadata -> _prefill_metadata_kernel`，再次出现同一编译器错误。
大行路径也用于超过八个路由行的 Decode 桶，并不表示请求阶段变成了 Prefill。
现将同一局部开关扩展到 `_prefill_metadata_kernel`，该扩展仍待 NPU 编译、数值及图回放复测。
行构造、激活、原生投影及 restore 内核保持各自的编译选项。
报错命令中添加的 `TRITON_ASCEND_ENABLE_AUTO_VECTORIZE_V2=0` 未使这次 BiSheng 命令带上
`--enable-auto-vectorize-v2=False`，复测应去掉该变量，使用代码中的显式 kernel 参数。
这里的 AutoVectorize V2 是编译器优化，与 MRV2 和 Decode 图模式独立。
六行路由通过不代表 bank 全桶、四卡或 DSpark 已通过；图回放及性能仍需在 NPU 上验收。
修改仅涉及 Python/Triton，无须重新编译 GS32 `.so` 和 `.o`，也无须清除原 Triton 缓存。

先运行只检查路由的用例，覆盖六行以及 12/512 行 metadata、256 个逻辑专家、
prefix/counts 和混合权重子集：

```bash
env -u TRITON_ASCEND_ENABLE_AUTO_VECTORIZE_V2 ASCEND_RT_VISIBLE_DEVICES=4 \
python -m pytest -vv -s -x -ra -o addopts= \
  tests/e2e/nightly/test_vq2a8_gs32_device_routes.py::test_gs32_routes_three_expert_decode_warmup \
  tests/e2e/nightly/test_vq2a8_gs32_device_routes.py::test_gs32_routes_three_expert_metadata_warmup \
  'tests/e2e/nightly/test_vq2a8_gs32_device_routes.py::test_gs32_routes_runtime_expert_bounds_subset_graph_recovery[6-prefix]' \
  'tests/e2e/nightly/test_vq2a8_gs32_device_routes.py::test_gs32_routes_runtime_expert_bounds_subset_graph_recovery[6-counts]' \
  --vq2a8-model "$MODEL"
```

这些用例读取模型配置并执行真实 NPU 路由，不加载整模型或执行原生投影。
通过后再运行原来的 `test_gs32_device_bank_decode_graph_buckets_and_recovery` 三组测试，
最后进行四卡整模型验证。若仍然崩溃，保留异常中打印的 Triton 缓存目录和完整 stderr，
同时记录 `python -m pip show triton-ascend torch torch-npu` 的版本信息；
不要先清空缓存，以免丢失提交编译器问题所需的 IR。

### FFN 与整模型检查

单卡检查混合 FFN prefill 数值和本地 EP2 bank 容量：

```bash
ASCEND_RT_VISIBLE_DEVICES=4 VLLM_USE_V2_MODEL_RUNNER=1 \
python -m pytest -vv -s -x \
  tests/e2e/nightly/test_vq2a8_gs32_batched.py \
  -k 'device_prefill_compare or device_prefill_ep2_capacity' \
  --vq2a8-model "$MODEL" \
  --vq2a8-gs32-library "$BUILD/libvq2a8_gs32_stage.so" \
  --vq2a8-gs32-binary "$BUILD/mixed_w2w4_moe_ffn.o"
```

单独检查 MRV2 的真实 EP2 eager/decode graph 对照及 attention：
显式选 EP2 node，避免误收集 EP4/EP8：

```bash
ASCEND_RT_VISIBLE_DEVICES=4,5 VLLM_USE_V2_MODEL_RUNNER=1 \
python -m pytest -vv -s -x \
  'tests/e2e/nightly/test_vq2a8_gs32_ep.py::test_gs32_ep_real_model_eager_and_graphs_match_single_card[ep2]' \
  tests/e2e/nightly/test_vq2a8_gs32_ep_attention.py \
  --vq2a8-model "$MODEL" \
  --vq2a8-gs32-library "$BUILD/libvq2a8_gs32_stage.so" \
  --vq2a8-gs32-binary "$BUILD/mixed_w2w4_moe_ffn.o" \
  --vq2a8-gs32-ep-skip-single-card
```

该单独 pytest 命令也显式省略 EP1 参考。单卡容量足够、需要补做 EP1 对照时，移除最后一个参数。
需要手动补做 MRV1 对照时，将该命令中的 `VLLM_USE_V2_MODEL_RUNNER=1` 改为 `0`；测试文件和 V1 实现仍保留。

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
