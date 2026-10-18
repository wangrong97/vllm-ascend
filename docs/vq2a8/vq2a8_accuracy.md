# VQ2A8 一键精度评测

所有命令均在仓库根目录执行。

其他指南：[NPU CI 回归](vq2a8_npu_ci.md) · [性能测试](vq2a8_performance.md) ·
[框架接入说明](vq2a8_gs32.md)。

通过已启动服务的 `/v1/chat/completions` 评测模型回答的正确率。
客户端使用固定版本 EvalScope 1.12.0 的数据适配器、提示词和规则评分，
不在客户端加载模型，不启动或重启推理服务，不调用额外裁判模型。
它与 NPU 算子回归 CI、TTFT/TPOT 性能测试互相补充。

| 参数 | 实际数据集 | 完整题数 |
| --- | --- | ---: |
| `gpqa` 或 `gpqa_diamond` | GPQA Diamond，`AI-ModelScope/gpqa_diamond` | 198 |
| `aime24` | AIME 2024，`evalscope/aime24` | 30 |
| `aime25` | AIME 2025，`evalscope/aime25` | 30 |

`gpqa` 明确指 Diamond 子集，不代表 GPQA 的所有子集。
默认运行表中的三个数据集，支持指定一个或多个；数据集别名不区分大小写。

## 安装与服务准备

使用 Python 3.10+。在当前推理容器内安装时，直接在仓库根目录执行：

```bash
python -m pip install -r tools/tests/requirements-evalscope.txt
python -c "from importlib.metadata import version; print(version('evalscope'))"
python -m pip check
```

版本应为 `1.12.0`。也可以先创建独立虚拟环境，或在能够访问推理服务的其他机器上安装。
离线机器需要提前准备对应 Python/操作系统的依赖安装包。
脚本不会自动安装依赖；已安装的 EvalScope 版本必须与固定版本一致。

## 指定 NPU 卡并启动精度评测服务

在终端 A 使用 **物理卡 4、5，MRV1、EP2、B1 decode 图、fusion=none** 启动服务。
沿用已经准备好的 `MODEL` 和 `BUILD`；`BUILD` 指向已有 `.so` 和 `.o` 所在目录。
下面为精度评测配置 **65536 总上下文、8192 prefill chunk**，与长输入性能测试的长度设置独立：

```bash
: "${MODEL:?请设置已有 source-v3 checkpoint 的绝对路径}"
: "${BUILD:?请设置已有 GS32 原生构建产物目录的绝对路径}"
mkdir -p benchmark-results/accuracy
set -o pipefail

ASCEND_RT_VISIBLE_DEVICES=4,5 VLLM_USE_V2_MODEL_RUNNER=0 \
VLLM_WORKER_MULTIPROC_METHOD=spawn OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -u tools/serve_vq2a8.py \
  --model "$MODEL" \
  --gs32-library "$BUILD/libvq2a8_gs32_stage.so" \
  --gs32-binary "$BUILD/mixed_w2w4_moe_ffn.o" \
  --ep-size 2 \
  --gs32-execution full_graph \
  --gs32-fusion none \
  --max-model-len 65536 \
  --max-num-batched-tokens 8192 \
  --kv-cache-mib 16384 \
  --tokenizer-mode deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --host 127.0.0.1 --port 8000 \
  2>&1 | tee benchmark-results/accuracy/serve-mrv1-ep2.log
```

`VLLM_USE_V2_MODEL_RUNNER=1` 选择 MRV2，需要重启服务。服务名固定为 `vq2a8`。
`full_graph` 使用 B1 decode 图，prefill 使用 eager 分块执行，最大活动序列数为 1。
`--kv-cache-mib 16384` 为每个 worker 预留 16 GiB KV cache，实际上下文容量由启动时的内存检查确认。
`--tokenizer-mode deepseek_v4` 显式选择模型的消息编码方式；`--reasoning-parser deepseek_v4`
将思考与最终回答拆开，供评测记录和答案评分使用。parser 本身不强制开启思考，模式仍由每次请求决定。
修改上下文、KV、runner 或设备需要重启；改变数据集、temperature 或思考模式无需重启。
如果已有容量足够且已配置上述 parser 的 256K 服务，可以复用；仅为性能压测启动、未配置 parser 的服务，
应按本节命令重启后再做思考模式的精度评测。评测期间不要同时在相同卡上运行性能或 NPU CI 测试。

终端 B 等待服务就绪并检查名称：

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models
```

## 上下文、输出预算与采样参数

服务启动命令设置运行容量。**`temperature`、`top_p`、`max_tokens` 和思考模式在精度脚本中设置，
由每个 HTTP 请求传给服务**，不作为 `serve_vq2a8.py` 的服务启动参数。
启动器使用 `--generation-config vllm`，评测命令显式指定的采样参数优先于服务默认值。

| 参数 | 设置位置 | 本文完整回归示例 | 含义 |
| --- | --- | --- | --- |
| `--max-model-len` | 服务端 | `65536` | 聊天模板处理后的输入与全部生成 token 的总长度上限 |
| `--max-num-batched-tokens` | 服务端 | `8192` | 每个调度步的 token 预算，不是输出上限，也不会把短题目补到 8192 |
| `--max-tokens` | 评测客户端 | thinking：`32768`；no-thinking：`8192` | 每题生成上限；thinking 时包括思考和最终回答，遇到正常结束可提前停止 |
| `--temperature` | 评测客户端 | `0` | 用于本项目的贪心解码回归对照；不保证跨硬件、并行配置完全一致 |
| `--top-p` | 评测客户端 | `1` | 显式固定采样条件；temperature 为 0 时不进行随机核采样 |
| `--seed` / `--repeats` | 评测客户端 | `0` / `1` | 固定随机种子、每题一次；增加 repeats 会增加请求数 |
| `--concurrency` | 评测客户端 | `1` | 匹配当前服务的 B1 配置 |

必须满足：`输入 token 数（含聊天模板） + 请求的输出预算 <= max_model_len`。
thinking 示例为 32768 输出预留空间，输入可用空间为 32768；no-thinking 示例预留 8192 输出，输入可用空间为 57344。
这不是对任意自定义数据集长度的保证。
预检会拒绝输出预算已经占满上下文的配置，逐题是否超限由服务校验；脚本不会自动截短题目。
此前 128/256-token 上下文的短性能服务不能直接用于这里的完整评测。

**GPQA Diamond、AIME24、AIME25 可以使用同一服务和同一套参数。**
一次命令选择多个数据集时，以上生成参数应用于全部数据集；提示词与答案评分由各自的 EvalScope 适配器处理。
先用统一条件建立回归基线。如果某个数据集出现较多输出截断，再单独运行该数据集并增大输出预算，
必要时增大服务上下文；比较 VQ2A8 与原始模型时，该数据集的两边必须使用相同预算和采样参数。
本文日常回归分别使用 thinking 32768、no-thinking 8192 的输出预算，并分别建立基线。
如果要单独比较思考开关的影响，需显式设置相同输出预算，只改变思考模式。
不同预算的结果不能直接归因于思考模式或量化差异；达到输出上限的题目仍计入评分分母，并统计截断数。
本文参数是项目回归条件，不是三个数据集或模型厂商的官方最优成绩配置。

## 一键执行（离线数据）

以下命令统一读取容器内的 `/mnt/share/g00872988/vq2a8-eval-data` 离线包。
请先复制完整数据目录，保留其中的 `manifest.json` 和 `datasets/`；其他部署路径需同步修改 `--offline-data-dir`。
脚本会校验离线数据后再评测；数据缺失或校验失败时直接报错，不会回退到联网下载。
首次接入可先使用下方的 `--limit 2` 示例检查返回内容、评分和截断情况，再运行完整评测。

```bash
# Thinking：三个数据集完整评测
python tools/tests/run_vq2a8_accuracy_suite.py \
  --base-url http://127.0.0.1:8000 \
  --model vq2a8 \
  --datasets gpqa aime24 aime25 \
  --offline-data-dir /mnt/share/g00872988/vq2a8-eval-data \
  --thinking --reasoning-effort high \
  --temperature 0 --top-p 1 --seed 0 --repeats 1 \
  --concurrency 1 \
  --max-tokens 32768 \
  --label mrv1-ep2-thinking-high-t0-o32768 \
  --result-dir benchmark-results/accuracy

# No thinking：复用同一个服务，每题输出上限改为 8192
python tools/tests/run_vq2a8_accuracy_suite.py \
  --base-url http://127.0.0.1:8000 \
  --model vq2a8 \
  --datasets gpqa aime24 aime25 \
  --offline-data-dir /mnt/share/g00872988/vq2a8-eval-data \
  --no-thinking \
  --temperature 0 --top-p 1 --seed 0 --repeats 1 \
  --concurrency 1 \
  --max-tokens 8192 \
  --label mrv1-ep2-no-thinking-t0-o8192 \
  --result-dir benchmark-results/accuracy

# 只运行 AIME25；--limit 2 明确标记为 smoke，不能作为完整成绩
python tools/tests/run_vq2a8_accuracy_suite.py \
  --datasets aime25 --thinking --limit 2 --concurrency 1 \
  --offline-data-dir /mnt/share/g00872988/vq2a8-eval-data \
  --temperature 0 --top-p 1 --max-tokens 32768

# 只预览完整命令，不校验离线包、不联网、不创建结果目录
python tools/tests/run_vq2a8_accuracy_suite.py \
  --offline-data-dir /mnt/share/g00872988/vq2a8-eval-data \
  --datasets gpqa aime24 --no-thinking --dry-run
```

也可以使用 `bash tools/tests/run_vq2a8_evalscope_accuracy.sh`，参数相同。
Python 和 Shell 入口都会自动把 EvalScope 子进程的完整 stdout/stderr 合并保存到本次结果目录的
`evalscope.log`，并实时镜像到终端，无需额外使用 `tee`。启动时会打印日志绝对路径，
`run.json` 的 `log` 字段也记录该路径。子进程使用 Python 无缓冲输出；连续 15 秒没有新输出时，
终端显示 `PROBE_WAIT=evalscope` 等待提示。该提示表示进程仍未退出，不代表题目已完成。
默认并发为 4；示例使用 1 便于在此前 B1 decode 服务上验证。
服务的活动序列数限制可能导致客户端请求排队。
请求默认超时 3600 秒，可用 `--timeout` 调整。

## Thinking 与 no thinking 的含义

| 客户端选项 | 发送的配置 | 用途 |
| --- | --- | --- |
| `--thinking` | `thinking=true`、`reasoning_effort=high` | 显式开启思考；high 是本脚本的默认选择 |
| `--thinking --reasoning-effort low` 或 `max` | `thinking=true`、指定 effort | 显式调整思考强度 |
| `--no-thinking` | `thinking=false`、`reasoning_effort=none` | 显式关闭思考 |
| 仅 `--reasoning-effort none/low/high/max` | 保持原有 effort 到模式的映射 | 兼容旧命令 |
| 两类选项均不指定 | 不发送思考模式配置 | 沿用服务默认；不能据此认定为 no thinking |

这里的 `thinking` 通过 `extra_body.chat_template_kwargs` 发送，同时传递顶层和模板中的 `reasoning_effort`，
与当前 DeepSeek-V4 前端匹配。模式控制提示词构造，temperature 控制解码采样，两者独立。
`--thinking` 与 `--no-thinking` 不能同时使用；与模式冲突的 effort 也会在发请求前报错。
`run.json`、`summary.json` 记录请求的 `thinking_mode` 与完整 `generation_config`，终端也会显示选择。
这些记录表示客户端请求的配置；真实响应是否有思考内容和最终答案，还应检查 `predictions/`。

不指定采样参数时保持原有默认值：输出上限 8192、温度 0、seed 0、repeats 1；
top-p 与思考模式沿用服务设置。本文 thinking 示例显式使用 32768 输出预算，no-thinking 示例显式使用 8192。
`--thinking` 本身不会自动提高输出上限；手动传入的 `--max-tokens` 决定本次预算。
输出预算是上限，不会强制每题生成这么多 token。关闭 thinking 后，正文仍可能包含长篇解题过程，8192 也可能截断答案。
修改输出预算无需重启服务，只影响新启动的评测；正在运行的评测继续使用启动时的配置。

## 内网离线数据

已有离线包时，将完整目录复制到评测容器可见的 `/mnt/share/g00872988/vq2a8-eval-data`，无需再次下载。
需要重新制作离线包时，在有网络的机器上使用相同 EvalScope 版本准备并导出完整数据集：

```bash
python tools/tests/prepare_evalscope_datasets.py \
  --datasets gpqa aime24 aime25 \
  --output-dir /path/to/vq2a8-eval-data
```

把整个导出目录复制到内网。数据包保留 EvalScope 原生缓存，附带题数、数据源、
版本和文件校验信息。评测时显式指定离线包：

```bash
python tools/tests/run_vq2a8_accuracy_suite.py \
  --base-url http://127.0.0.1:8000 \
  --model vq2a8 \
  --datasets gpqa aime24 aime25 \
  --offline-data-dir /mnt/share/g00872988/vq2a8-eval-data \
  --thinking --reasoning-effort high \
  --temperature 0 --top-p 1 --seed 0 --repeats 1 \
  --max-tokens 32768 \
  --concurrency 1 \
  --result-dir benchmark-results/accuracy
```

离线包缺失、版本不符、题数错误或文件校验失败会在发送评测请求前退出。
包中可以包含多个数据集，运行时仅选择其中需要的部分。
校验说明文件完整性，不能证明手工替换的数据仍属于官方数据集。

有网络时也可用 `--dataset-dir /path/to/cache` 共享普通缓存，
缺失数据会交给 EvalScope 下载。`--dataset-dir` 与 `--offline-data-dir` 不能同时使用。
预下载入口为 `prepare_evalscope_datasets.py` 或同名 `.sh`，不需要模型服务。

## 结果与退出状态

每次运行创建独立子目录，保留原始回答和评分依据：

```text
benchmark-results/accuracy/<timestamp-random>/
  run.json                 # 命令、环境、客户端提交、服务信息、实际参数和运行状态
  evalscope.log            # 子进程完整 stdout/stderr，与终端实时输出对应
  configs/                 # EvalScope 展开的配置
  logs/                    # EvalScope 日志
  predictions/             # 原始回答、停止原因和 token 用量
  reviews/                 # 逐题评分
  reports/                 # 原生准确率报告
  accuracy_audit.json       # 预测记录与报告的一致性审计
  summary.json             # 各数据集分数及评测口径
  summary.csv              # 便于对比的分数表
```

`score` 为 0 到 1 的准确率。默认使用原生 `accuracy:mean`；重复多次时计算所有
回答的平均正确率，不自动变成 pass@k。三个数据集分别报告，不混成一个总分。
`--limit` 运行的 `scope` 标记为 `smoke`，完整运行标记为 `full`。

脚本检查预测记录、重复索引、完成题数和原生评分报告是否一致。
子进程失败、缺失报告、题数不符、请求错误、上下文超限和过滤等返回非零退出码。
输出达到 `max_tokens` 和最终回答为空会明确统计，仍保留在评分分母中；
它们不会被静默丢弃，也不单独被当成基础设施故障。
Ctrl+C 返回 130，`run.json` 标记 `interrupted`，部分结果不能作为完整成绩；Linux 上的 SIGTERM 同样处理。
Linux 上会清理本次 EvalScope 子进程组及其工作进程，保留退出前已写入的日志；不会停止推理服务。
子进程失败时终端同时显示退出码和 `evalscope.log` 路径，可直接检查原始报错。

**退出码 0 表示评测完成且产物审计通过，不表示精度达到合入门槛。**
该入口没有默认精度阈值；成绩为 0 的完整有效运行也可能退出 0。
评测不验证服务器所加载权重或算子与客户端提交一致。
可用 `--label` 标注配置、`--server-revision` 记录操作者提供的服务构建版本；
客户端提交 SHA 单独记录，不冒充服务端版本。

## 与原始模型对比

这里的 baseline 是已经下载到本地的 **官方 `DeepSeek-V4-Flash-0731` checkpoint**。
官方 Ascend 950DT 部署使用 Attention MXFP8、MoE MXFP4/W4A8 路径，不能标记为 BF16 baseline。
原始权重直接由 vLLM 加载；不使用 VQ2A8 启动器、GS32 构建产物或 `--quantization ascend`。
部署依据见 [DeepSeek-V4-Flash 官方指南](https://docs.vllm.ai/projects/ascend/en/v0.27.1rc1/tutorials/models/DeepSeek-V4-Flash.html)。

### 启动官方 baseline 服务

下面使用 **物理卡 4、5、6、7，MRV1、DP4、TP1、EP、decode 图**，端口 `8001`，服务名 `baseline`。
四卡方案沿用仓库中 Ascend 950DT 的部署拓扑，不代表模型的理论最少卡数。
这与前面的 VQ2A8 双卡拓扑不同：可以用于相同评测条件下的精度对照，不能直接作为相同资源配置的性能对比。
请先停止占用这些卡的其他服务，再在终端 A 执行；`BASELINE_MODEL` 改成官方权重的实际目录：

```bash
export BASELINE_MODEL=/mnt/share/g00872988/DeepSeek-V4-Flash-0731
mkdir -p benchmark-results/accuracy-baseline
set -o pipefail

ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 VLLM_USE_V2_MODEL_RUNNER=0 \
VLLM_WORKER_MULTIPROC_METHOD=spawn OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
OMP_PROC_BIND=false HCCL_BUFFSIZE=1024 \
PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
python -u -m vllm.entrypoints.cli.main serve "$BASELINE_MODEL" \
  --served-model-name baseline \
  --host 127.0.0.1 --port 8001 \
  --dtype auto \
  --tensor-parallel-size 1 \
  --data-parallel-size 4 \
  --enable-expert-parallel \
  --api-server-count 1 \
  --max-model-len 65536 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.9 \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --no-async-scheduling \
  --block-size 32 \
  --tokenizer-mode deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --generation-config vllm \
  --seed 0 \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1],"max_cudagraph_capture_size":1,"cudagraph_num_of_warmups":1}' \
  --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":false},"enable_cpu_binding":true,"multistream_overlap_shared_expert":true}' \
  2>&1 | tee benchmark-results/accuracy-baseline/serve-official-0731-mrv1-dp4-compile0.log
```

该命令采用普通解码，不开启 DSpark/MTP。`mode=0` 关闭 torch.compile，显式关闭 `npugraph_ex`；
`FULL_DECODE_ONLY` 保留 B1 decode ACL 图捕获和回放，prefill 分块执行。
仅设置 `FULL_DECODE_ONLY` 不会自动关闭编译优化，两项需要分别配置。
`--max-num-seqs 1` 是每个 DP rank 的活动序列上限，下面客户端仍以全局并发 `1` 评测。
与 VQ2A8 示例保持相同的 65536 总上下文、8192 prefill chunk、tokenizer 和 reasoning parser。
baseline 使用自动内存规划，启动日志中的 KV cache 容量检查必须通过；本命令尚未在目标 NPU 上验证。
temperature、输出预算和思考模式由下方客户端请求指定，切换数据集或思考模式不需要重启服务。

在终端 B 确认服务就绪，`/v1/models` 应返回 `baseline`，再验证一次真实生成请求：

```bash
curl -fsS http://127.0.0.1:8001/health
curl -fsS http://127.0.0.1:8001/v1/models
curl -fsS --max-time 180 http://127.0.0.1:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"baseline","messages":[{"role":"user","content":"Reply with OK."}],"temperature":0,"max_tokens":32,"chat_template_kwargs":{"thinking":false,"reasoning_effort":"none"}}'
```

生成请求应成功并返回非空回答；只有 `/health` 成功不足以证明推理正常。

### 启动 profiling 的 HC 形状错误

若启动栈包含 `profile_run`、`aot_autograd`、`npugraph_ex`，并报错
`The residual.shape[1] should be sequence, actual residual.shape[1] is 8192, sequence is 1`，
说明 `npu_hc_post` 收到的主分支与残差分支 token 维不一致。
这里的 `sequence` 是算子输入 `x.size(1)`，不是服务参数 `max-num-seqs`。
该日志本身不能确定哪个上游算子或编译环节产生了错误形状。

上面的启动配置通过 `mode=0` 和 `enable_npugraph_ex=false` 隔离本次报错的编译路径，
保留上下文、chunk、并行拓扑和原始权重。此配置已完成源码核对，尚待目标 NPU 启动及真实请求验证；
不能将它视为编译后端根因已修复的证据。仅修改启动参数，无需重新编译 GS32 算子。

如果仍然失败，可将整项 `--compilation-config` 替换为下面两项，其他参数保持不变，进行全 eager 对照：

```bash
--enforce-eager \
--compilation-config '{"mode":0,"cudagraph_mode":"NONE"}'
```

全 eager 对照同时关闭 decode 图，结果应单独记录，不能与 decode 图服务混作同配置性能数据。
若 `mode=0` 后仍在 `npu_hc_post` 报相同形状错误，需要继续定位实际前向张量，不能仅归因于编译优化。

### 一键评测官方 baseline（离线数据）

No-thinking 完整评测，三个数据集统一使用 8192 输出上限：

```bash
python tools/tests/run_evalscope_accuracy.py baseline \
  --base-url http://127.0.0.1:8001 \
  --model baseline \
  --datasets gpqa aime24 aime25 \
  --offline-data-dir /mnt/share/g00872988/vq2a8-eval-data \
  --no-thinking \
  --temperature 0 --top-p 1 --seed 0 --repeats 1 \
  --max-tokens 8192 \
  --concurrency 1 \
  --label official-0731-mrv1-dp4-compile0-no-thinking-t0-o8192 \
  --result-dir benchmark-results/accuracy-baseline
```

Thinking 完整评测，复用同一个服务，输出上限为 32768：

```bash
python tools/tests/run_evalscope_accuracy.py baseline \
  --base-url http://127.0.0.1:8001 \
  --model baseline \
  --datasets gpqa aime24 aime25 \
  --offline-data-dir /mnt/share/g00872988/vq2a8-eval-data \
  --thinking --reasoning-effort high \
  --temperature 0 --top-p 1 --seed 0 --repeats 1 \
  --max-tokens 32768 \
  --concurrency 1 \
  --label official-0731-mrv1-dp4-compile0-thinking-high-t0-o32768 \
  --result-dir benchmark-results/accuracy-baseline
```

也可以使用 `bash tools/tests/run_baseline_evalscope_accuracy.sh`，其余参数相同。
两个入口都会实时打印 EvalScope 输出并保存日志；每次运行单独创建结果目录，不覆盖已有评测。
首次检查可追加 `--limit 2`；正式评测移除该选项。离线包缺失或校验失败会报错，不会自动联网下载。

两边使用相同数据包、提示模板、思考配置、采样参数、输出预算和重复次数。
no-thinking 比较两边的 8192 预算结果，thinking 比较两边的 high / 32768 预算结果。
先检查截断、空回答和请求错误，再比较各数据集分数。
AIME 每年只有 30 题，一道题约占 3.33 个百分点；小样本波动不能直接归因于量化。

## 客户端回归测试

```bash
python -m pytest --noconftest -o addopts= -q \
  tests/ut/tools/test_evalscope_accuracy_runner.py \
  tests/ut/tools/test_evalscope_accuracy_summary.py \
  tests/ut/tools/test_evalscope_dataset_prepare.py
```

客户端测试不替代目标 NPU 上真实模型的完整精度验收。

## 接口依据

- [固定 EvalScope 版本](https://pypi.org/project/evalscope/1.12.0/)
- [AIME 适配器](https://github.com/modelscope/evalscope/blob/v1.12.0/evalscope/benchmarks/aime/aime_adapter.py)
- [GPQA 适配器](https://github.com/modelscope/evalscope/blob/v1.12.0/evalscope/benchmarks/gpqa/gpqa_adapter.py)
- [报告 schema](https://github.com/modelscope/evalscope/blob/v1.12.0/evalscope/report/report.py)
- [vLLM DeepSeek-V4 tokenizer 选择](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/config/model.py)
- [vLLM parser 选择与响应处理](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/parser/parser_manager.py)
