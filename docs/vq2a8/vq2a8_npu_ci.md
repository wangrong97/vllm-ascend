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
  `build-manifest.json`、`libvq2a8_gs32_stage.so` 和 `mixed_w2w4_moe_ffn.o`。
- 整模型测试还需要完整的 `vllm_ascend_C` 扩展及配套 ACLNN 算子，包括 SAS/QLI AICPU metadata 修复。
  仅构建 GS32 bridge 不足以运行整模型；更新相关源码后应重建对应原生产物。
  构建与 checkpoint 说明见[集成文档](vq2a8_gs32.md#构建与-checkpoint-准备)。
- 真实专家对照覆盖生产支持的 VQ2/VQ2、VQ2/MXFP4、MXFP4/MXFP4 三种组合；缺失默认阻断。
  MXFP4/VQ2 不属于完整 FFN 支持范围，不再自动允许这一历史组合的缺失。

先确认当前终端中的既有路径：

```bash
: "${MODEL:?请设置已有 source-v3 checkpoint 的绝对路径}"
: "${BUILD:?请设置已有 GS32 原生构建产物目录的绝对路径}"
test -f "$MODEL/config.json"
test -f "$MODEL/model.safetensors.index.json"
test -f "$BUILD/build-manifest.json"
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
export BUILD="$PWD/build/vq2a8-gs32-runtime-ffn"
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
export TMPDIR=/tmp
export VLLM_RPC_BASE_PATH=/tmp
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

按[完整 FFN 构建说明](packed_weight_batch_throughput.md#0-首次升级完整-ffn重建-abi-v3)更新桥接 `.so` 及清单。
本次保留设备 ABI v3 和数学逻辑；设备对象必须与当前目标和 CANN 兼容，预检核对 v3 符号及 SHA256。
先校验主模型扩展和 native GS32 bridge 的注册接口。此命令不加载权重、不启动服务，也不执行设备算子：

```bash
python tools/serve_vq2a8.py \
  --model "$MODEL" --no-dspark \
  --gs32-runtime-dir "$BUILD" \
  --parallel-mode tp-ep --ep-size 4 \
  --max-model-len 8192 --max-num-batched-tokens 8192 --max-num-seqs 64 \
  --check-native-only
```

正常 native 服务启动也会执行此前置检查；`--dry-run` 仍只检查配置，不加载扩展。
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

然后执行四卡完整 FFN B64 验收。该用例依次加载完整 FFN 的 eager 和 graph，
比较同权重结果并检查实际完整 FFN 图回放：

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
完整 FFN 的独立数值 oracle 仅供测试使用，不作为服务后端。CPU 检查不代替上述真机验收。
下方一键套件包含完整 FFN 组件回归，但不代替本节的 EP4/B64 整模型对照。
repack 前后的非 routed 量化不同，不能要求两份权重逐 token 一致；eager/graph 对照始终使用同一份权重。

## 一键执行

日常 PR 回归使用空闲卡 4、5，默认运行五个组件阶段和 MRV2 的 `ep2-v2` 整模型阶段，共六个阶段。
一键 CI 默认选择 Model Runner V2；`ep2-v1` 已从自动计划移除，MRV1 的整模型 eager/decode graph 对照仍可手动运行。
缺少支持的真实专家组合默认使验收失败；显式接受覆盖缺口的规则见[缺失组合处理](#checkpoint-缺少真实专家组合时)。

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

完整 FFN 测试要求真实 checkpoint 覆盖 `vq2_to_vq2`、`vq2_to_mxfp4` 和 `mxfp4_to_mxfp4`。
混合图测试还要求同一层中存在这三种真实组合，以验证单次完整 FFN 调用中的混合专家布局。
缺少真实组合或混合层时会报告未覆盖，默认阻断验收；不再设置任何自动允许缺失组合。
`mxfp4_to_vq2` 是不支持的生产组合，不能通过允许缺失参数启用。

仅需要检查已有支持组合时，可显式添加例如 `--allow-missing-checkpoint-pairs vq2_to_mxfp4`。
它只接受已识别的真实专家缺失用例，不能忽略混合层缺失、其他 skip、xfail、错误或数值失败。
即使显式接受缺失，阶段也必须有实际通过用例；整阶段全 skip 不算完成。

其余检查全部通过时，显式接受覆盖缺口的运行报告为 `passed_with_coverage_gaps`、
`coverage_complete=false`，退出码为 `0`；没有覆盖缺口时才是 `passed`。
`coverage_gaps` 保留阶段、checkpoint、测试 nodeid、组合和原因，原始 skip 计数也会保留。
`automatic_missing_checkpoint_pairs` 为空，`allowed_missing_checkpoint_pairs` 仅记录显式接受的组合。
后续阶段失败时总状态仍为 `failed`。

完整验收应要求 `status=passed` 且 `coverage_complete=true`，并保存原始日志。
这里的完整覆盖只针对所选套件，不包括未选择的 EP1、EP4/B64、长上下文性能或答题精度。

## 三套件的覆盖范围

| 套件 | 设备要求 | 默认覆盖 |
| --- | --- | --- |
| `kernels` | 恰好一张，例如 `--devices 4` | 五个组件测试文件，共五个阶段 |
| `pr`，默认 | 恰好两张，例如 `--devices 4,5` | 五个组件阶段，再运行 MRV2 的真实 EP2 eager/decode graph 对照 `ep2-v2`，默认共六个阶段 |
| `nightly` | 恰好两张 | PR 覆盖加 `recovery-v2` 短/16K/短恢复测试，默认共七个阶段 |

五个组件文件均位于 `tests/e2e/nightly/`：

| 文件 | 主要验证内容 |
| --- | --- |
| `test_vq2a8_gs32_activation.py` | 完整 FFN 初始激活量化及有效性边界 |
| `test_vq2a8_gs32_device_routes.py` | 设备 metadata、counts/prefixes、padding、无效路由与恢复 |
| `test_vq2a8_gs32_full_ffn.py` | 三种真实专家与独立官方 GMM oracle 对照、中间 FP8/E8M0、混精动态路由、图捕获及恢复 |
| `test_vq2a8_gs32_ep_routes.py` | EP 专家归属及路由布局 |
| `test_vq2a8_gs32_ep_attention.py` | attention 相关回归 |

组件阶段始终使用 `--devices` 中的**第一张卡**，runner 使用 `--model-runners` 中的首个值；
默认是 V2，显式选择 `v1` 或 `v1 v2` 时组件阶段使用 V1。
完整 FFN 组件使用独立 `OfficialFFNOracle`，不依赖已删除的 staged/reference 生产模块。
组件测试不执行多卡 HCCL；真实 EP2 模型阶段才检查 dispatch、combine、collective 和专家归属。

`--model-runners` 默认是 `v2`，仍可显式选择 `v1` 或 `v1 v2`；
首个值决定组件阶段使用的 runner，全部所选值决定 `pr`、`nightly` 的整模型阶段；不会重新启用 `ep2-v1`。
五个组件阶段验证各自的算子或组件，不能代替整模型执行验证。

| runner 选择 | `pr` 阶段 | `nightly` 阶段 |
| --- | --- | --- |
| 默认，或 `--model-runners v2` | 五个组件 + `ep2-v2`，共六个 | 五个组件 + `ep2-v2` + `recovery-v2`，共七个 |
| `--model-runners v1` | 仅五个组件，共五个 | 五个组件 + `recovery-v1`，共六个 |
| `--model-runners v1 v2` | 五个组件 + `ep2-v2`，共六个 | 五个组件 + `ep2-v2` + `recovery-v1` + `recovery-v2`，共八个 |

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

MRV2 支持 B1～B64。完整 FFN 组件捕获并交替回放 6、128、129、384 路由行的图桶，
从空路由捕获后切换到热点或混合专家，检查 padding、无效输入及恢复。
这是组件验收；top-k=6 的 384 行边界不能代替完整 EP2/EP4 模型的多序列验收。

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
  若仅存在显式接受的支持组合缺失，汇总为 `passed_with_coverage_gaps`；无 skip 时为 `passed`。
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
  full_ffn.log
  full_ffn.xml
  full_ffn-tmp/
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
sed -n '1,220p' "$RESULT/full_ffn.log"
tail -n 80 "$RESULT/full_ffn.log"
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

### 路由或图捕获故障定位

完整 FFN 仍依赖 Triton 激活和路由 metadata。需要单独定位这些组件时，运行：

```bash
ASCEND_RT_VISIBLE_DEVICES=4 python -m pytest -vv -s -x -ra -o addopts= \
  tests/e2e/nightly/test_vq2a8_gs32_activation.py \
  tests/e2e/nightly/test_vq2a8_gs32_device_routes.py \
  --vq2a8-model "$MODEL"
```

这些组件测试不替代完整 FFN 或整模型验收。出现 BiSheng 编译错误时保留完整 stderr、
异常中的 Triton 缓存目录及 `python -m pip show triton-ascend torch torch-npu` 版本输出，
再区分失败发生在预热、捕获还是回放。清空缓存会丢失定位编译器问题所需的 IR。

### FFN 与整模型检查

单卡检查真实混合专家的完整 FFN 数值与图桶：

```bash
ASCEND_RT_VISIBLE_DEVICES=4 VLLM_USE_V2_MODEL_RUNNER=1 \
python -m pytest -vv -s -x \
  tests/e2e/nightly/test_vq2a8_gs32_full_ffn.py \
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
