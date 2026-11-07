# Ascend 950 启动时 HcclAllGather 超时排查

本文对应 TP2/EP2 在权重加载完成后、MRV2 `profile_run()` 末尾同步时报
`HcclAllGather / EI0006 / Getting socket times out` 的问题。
目前只能确认通信失败，尚未确认是部署环境、特定通信组还是模型执行路径导致；不能将本次排查工具视为服务修复。
真机位于内网，由使用者执行以下检查并回传结果。

## 已确认的调用路径

- 提供的日志显示两个 worker 都完成权重加载，失败位于 KV 容量初始化期间的 profile。
  `shm_broadcast` 等待是 engine 等待 worker 的现象，不能单独作为根因。
- `worker/worker.py` 即使使用显式 KV 预算，也会运行 profile。
  增加 KV 预算或跳过权重加载不能据此解决通信问题。
- Ascend MRV2 调用 upstream `profile_run()`；本地 upstream 实现包含 dummy forward、dummy sampler 和末尾同步。
  dummy sampler 会调用 `compute_logits`，分片 LM-head 会执行 TP AllGather。
  用户安装版本的具体调用点仍应以同步栈为准。
- 当前 GS32 的标准 TP2/EP2、DP1、PCP1 配置中，MoE 策略虽然名为 `ALLGATHER`，
  专家准备阶段并不实际执行 AllGather，最终使用 TP AllReduce。
  启用其他并行配置或草稿模型时，不能直接套用这个判断。
- 通信组 registry 的 key 包含 `hccl_config.group_name`，真实 TP/EP 名称不同，
  不能仅凭两组成员相同就认定它们复用了同一个 HCCL group。

异步执行会使错误延迟到同步点暴露，因此上述 LM-head 路径只是定位线索。
华为的 [EI0006 说明](https://www.hiascend.com/doc_center/source/en/canncommercial/800/maintenref/troubleshooting/atlaserrorcode_15_0248.html)
列出了对端未及时执行、对端异常和链路问题等原因；仅凭这条错误不能判断哪一种已发生。

## 第一步：不加载模型的通信检查

在**服务实际使用的容器、Python 环境和仓库根目录**执行。使用与失败服务相同的可见设备列表；
下面的 `0,1` 对应 `ASCEND_RT_VISIBLE_DEVICES` 的条目，容器内条目不一定等于宿主机物理卡号。
先停止占用这两张卡的失败服务及残留 worker，再运行：

```bash
python tools/tests/check_vq2a8_hccl.py --devices 0,1
```

默认总执行超时 180 秒，退出清理另有最多约 10 秒宽限。可用 `--timeout 300` 调整探针的等待预算；
这不会修改 `HCCL_CONNECT_TIMEOUT` 等通信配置。
只查看计划、不初始化设备也不创建文件时，添加 `--dry-run`。

探针按相同顺序在全部 rank 执行：

1. 默认 world group 的 FP32 AllReduce，并检查结果。
2. 创建成员相同的两个独立 group，分别以 `tp`、`ep` 标识。
3. 在 world、tp、ep group 上测试 FP32/BF16 AllGather，输入分别为 `[1, 131072]` 和 `[64, 131072]`。
   验证每个 rank 的输出内容，记录调用前、提交后、同步后和校验后的时间。

这些是通用 PyTorch HCCL group，不导入 vLLM，也不复制 vLLM 的 group options 或并行拓扑。
`--rows`、`--columns` 可调整输入规模；默认张量仅用于通信诊断，不代表模型的实际 batch 或词表分片形状。
父进程直接管理每个 rank 的进程组，超时、失败或中断时清理本次启动的 worker。

结果写入独立目录 `benchmark-results/hccl-check/hccl-check-*/`：

| 文件 | 用途 |
| --- | --- |
| `summary.json` | 总状态、退出码、全部 rank 报告和缺失检查 |
| `workers.log` | 实时输出、逐 rank 阶段与原始异常 |
| `rank-N.json` | 设备映射条目、软件版本、配置路径类型、最后阶段和数值校验 |

只有所有 rank 完成全部数值检查且正常退出，才返回 `passed` 和退出码 0。
缺报告、缺检查、错误设备映射、超时或任意 rank 失败均不通过。
路径检查只记录 `/etc/hccl_rootinfo.json`、`/lib/route.conf`、`/etc/hixlep` 的类型和可读性等元数据，
不导出拓扑内容、不修改系统文件，也不把 JSON 可解析当作拓扑正确的证明。

## 第二步：按首个失败阶段处理

| 结果 | 后续定位 |
| --- | --- |
| 初始化或首个 AllReduce 失败 | 故障已在不加载 vLLM 的环境中复现；检查两 rank 最早异常、设备映射、运行环境和链路 |
| AllReduce 通过，AllGather 失败 | 保留失败 group、dtype、shape 和两 rank 时间；进一步检查 HCCL/CANN 的该通信路径 |
| world 通过，其他 group 失败 | 保留多 group 创建和通信日志，检查进程组配置与资源；不能直接归因于模型权重 |
| 所有探针通过，服务仍失败 | 通用通信可用，但尚未验证 vLLM 的 options、通信顺序、内存压力或模型算子；执行下一节同步定位 |

若为使用 UB 拓扑的 950DT 部署，应核对容器挂载来自本机的有效配置。
若 `/etc/hccl_rootinfo.json` 是目录，应修正宿主机文件与容器挂载；空 JSON 或其他机器的配置不能替代本机拓扑。
具体要求参见[仓库部署说明](../source/getting_started/installation.md#installation-multi-node-physical)及
[Ascend 官方 950 部署说明](https://github.com/Ascend/MindIE-Motor/blob/master/docs/zh/user_guide/deployment/k8s/pd_disaggregation_deployment.md)。
配置路径异常是待核实线索，探针不会据此擅自修正部署。

## 第三步：保留原配置，获取同步报错栈

给**原来的完整启动命令**临时添加 `ASCEND_LAUNCH_BLOCKING=1`，保存两 rank 的完整日志。
例如原先用自有启动脚本时，在同一终端执行下面的形式，将脚本路径替换为原脚本：

```bash
mkdir -p benchmark-results/startup-debug
set -o pipefail
env ASCEND_LAUNCH_BLOCKING=1 bash /path/to/original-start.sh \
  2>&1 | tee benchmark-results/startup-debug/service-blocking.log
```

若原先直接执行 `python tools/serve_vq2a8.py ...` 或 `vllm serve ...`，
将上面的 `bash /path/to/original-start.sh` 替换为该完整命令，保留所有模型、TP/DP/EP、batch、KV、图模式和 DSpark 参数。
确认启动脚本没有覆盖此环境变量。该设置用于定位，会降低性能；本命令仅对子进程设置变量，正常服务与性能测试使用原启动命令。
若同步模式下通过，也只能说明执行时序变化影响了结果，仍需复测原异步模式。

同时记录当前版本和实际安装位置：

```bash
git rev-parse HEAD
python -m pip show vllm vllm-ascend torch torch-npu
```

请回传：原始完整启动命令、`summary.json`、`workers.log`、同步模式下的两 rank 完整异常，
以及上述版本信息。保留最早的错误，只有 engine 最后的退出栈不足以定位。
目前不建议通过延长 HCCL 超时、关闭 TLS 或绕过 profile 来宣称修复。

## 服务恢复后的验收

真实 checkpoint 启动成功后，还需确认 `/v1/models` 返回 200，并发送至少一条生成请求，
检查 HTTP 200、非空文本和服务日志。随后在原异步配置下运行
[NPU 回归](vq2a8_npu_ci.md)，再做目标 batch/并行配置测试。
通信探针、CPU 单测或服务 ready 日志均不能替代真实权重请求验证。
