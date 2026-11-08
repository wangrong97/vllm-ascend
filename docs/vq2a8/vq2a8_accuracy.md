# 无 perm VQ2A8 混精模型：四卡精度评测

所有命令均在内网 Linux 仓库根目录执行。
当前 VQ2A8 示例适用分支为 `test/vq2a8-perf-20261024`，采用四卡 TP4/EP4、DP1、B64；更新日期：2026-11-07。
模型为 `/mnt/share/g00872988/DeepSeek-V4-Flash-0731-32x32-noperm-hot5050125-identity-v3-export-20261103`，使用新格式主模型和 `--no-dspark`。
Linux 路径按内网实际存放位置调整，服务端和客户端使用同一份模型；本地已检查的目录为 `D:\models\DeepSeek-V4-Flash-0731-32x32-noperm-hot5050125-identity-v3-export-20261103`。
本分支只接受自然输入列顺序，旧 `.perm` 张量（包括恒等排列）会被拒绝。

本页默认测 [repack 后的 FP8 非 routed 权重](repack_non_routed.md)，首次使用先做 [EP4/B64 验收](vq2a8_npu_ci.md#新-repack-权重四卡验收)。
其运行时还包含动态 MXFP8 激活量化；转换报告的权重误差不能替代整模精度评测。
对照 repack 前的 BF16 权重时，切换模型路径，并将标签中的 `nonrouted-fp8` 改为 `nonrouted-bf16`；其余代码、服务和评测参数保持一致。

其他指南：[NPU CI 回归](vq2a8_npu_ci.md) · [性能测试](vq2a8_performance.md) ·
[框架接入说明](vq2a8_gs32.md)。

通过已启动服务的 `/v1/chat/completions` 评测模型回答的正确率。
客户端使用固定版本 EvalScope 1.12.0 的数据适配器、提示词和规则评分，
不在客户端加载模型，不启动或重启推理服务，不调用额外裁判模型。
它与 NPU 算子回归 CI、TTFT/TPOT 性能测试互相补充。
本次权重没有 MTP 张量，即使配置包含 `num_nextn_predict_layers=1`，也不代表有可用草稿。
本文固定 `--no-dspark`；旧 DSpark 指南不作为这份模型的启用说明。

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
python - <<'PY'
from importlib.metadata import version
import numpy
import pyarrow
import datasets
from evalscope.config import TaskConfig

for package in ("evalscope", "numpy", "pyarrow", "datasets"):
    print(f"{package}=={version(package)}")
PY
python -m pip check
```

版本应为 `1.12.0`。也可以先创建独立虚拟环境，或在能够访问推理服务的其他机器上安装。
离线机器需要提前准备对应 Python/操作系统的依赖安装包。
脚本不会自动安装依赖；已安装的 EvalScope 版本必须与固定版本一致。

依赖文件同时限制 `pyarrow<26`，兼容仍需 NumPy 1.x 的 Ascend 环境。
PyArrow 26 在导入时要求 NumPy 2；仅检查包版本或运行 `pip check`
不一定能发现这一运行时限制，因此上面也实际导入数据集和 EvalScope。

如果已有环境报 `pyarrow requires NumPy 2.0 or newer, found 1.26.4`，
保留推理环境要求的 NumPy 1.26.4，只替换 PyArrow：

```bash
python -m pip install --only-binary=:all: --no-deps "pyarrow==25.0.1"
```

然后重新执行上面的导入检查和原评测命令。`--no-deps` 使这次修复仅修改 PyArrow。
[PyArrow 25.0.1](https://github.com/apache/arrow/blob/apache-arrow-25.0.1/docs/source/python/install.rst)
支持 NumPy 1.x，且有 Python 3.11 Linux aarch64 wheel；
[PyArrow 26 的导入检查](https://github.com/apache/arrow/blob/apache-arrow-26.0.0/python/pyarrow/lib.pyx)
要求 NumPy 2。无需重新下载离线数据或重新构建 GS32 算子。
若 `pip check` 仍报告其他包冲突，应按对应包的约束处理，不要为消除本条错误直接升级 NumPy。

## 指定 NPU 卡并启动精度评测服务

在终端 A 使用 **物理卡 4、5、6、7，native、MRV2、TP4/EP4、DP1、B64 decode 图、fusion=full_ffn** 启动服务。
更新后按[完整 FFN 构建说明](packed_weight_batch_throughput.md#0-首次升级完整-ffn重建-abi-v4)重建设备对象、桥接库及清单，旧 ABI v3 产物不能复用。
卡号按实际空闲四卡调整；`BUILD` 指向含 `build-manifest.json`、`.so` 和 `.o` 的匹配产物目录。
下面为精度评测配置 **262144（256K）总上下文、8192 prefill chunk**，为每题最多 128K 输出保留输入空间：

```bash
set -o pipefail
export MODEL=/mnt/share/g00872988/DeepSeek-V4-Flash-0731-32x32-noperm-hot5050125-identity-v3-export-20261103
export BUILD="$PWD/build/vq2a8-gs32-runtime-ffn"
SERVE_DIR="$PWD/benchmark-results/accuracy/serve-noperm-mixed-nonrouted-fp8-native-tp4-b64-$(date +%Y%m%d_%H%M%S)"
mkdir -p benchmark-results/accuracy &&
mkdir "$SERVE_DIR" &&
git rev-parse HEAD > "$SERVE_DIR/commit.txt" &&
git status --short > "$SERVE_DIR/worktree.txt" &&
sha256sum "$MODEL/config.json" "$MODEL/model.safetensors.index.json" > "$SERVE_DIR/model-files.sha256" &&
sha256sum "$BUILD/libvq2a8_gs32_stage.so" "$BUILD/mixed_w2w4_moe_ffn.o" > "$SERVE_DIR/gs32-files.sha256" &&
echo "服务日志：$SERVE_DIR/serve.log" &&

TMPDIR=/tmp VLLM_RPC_BASE_PATH=/tmp \
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 \
VLLM_USE_V2_MODEL_RUNNER=1 VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
VLLM_WORKER_MULTIPROC_METHOD=spawn OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -u tools/serve_vq2a8.py \
  --model "$MODEL" --no-dspark \
  --gs32-runtime-dir "$BUILD" \
  --parallel-mode tp-ep --ep-size 4 \
  --max-model-len 262144 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 64 \
  --kv-cache-mib 16384 --reserve-gib 16 \
  --tokenizer-mode deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --runtime-record "$SERVE_DIR/service-runtime.json" \
  --host 127.0.0.1 --port 8003 \
  2>&1 | tee "$SERVE_DIR/serve.log"
```

服务日志目录以本机时间 `YYYYMMDD_HHMMSS` 结尾，同类目录可按名称排序。
准备命令通过 `&&` 衔接，失败时不再启动服务；若同一秒创建同名目录，等待一秒后重试即可，不会覆盖已有记录。

`service-runtime.json` 保存启动配置、版本及 native 产物 SHA256；请随精度结果保留。
它是启动记录，`device_execution_verified=false` 不表示已经通过设备数值验收。
VQ2A8 启动器默认使用 MRV2；显式设置 `VLLM_USE_V2_MODEL_RUNNER=0` 并重启可切换至 MRV1。服务名固定为 `vq2a8`。
以上命令显式使用 `--max-num-seqs 64`，允许最多 64 个活动序列，自动配置 `[1,2,4,8,16,32,64]` decode 图桶，
prefill 使用 eager 分块执行。启动日志中的 `world_size` 应为 `4`；已有 B1/B4 服务需重启才能使用 B64 活动序列上限。
MRV1 仍只支持 B1 decode 图，切换至 MRV1 时还需将 `--max-num-seqs` 改为 `1`。
`--kv-cache-mib 16384` 为每个 worker 预留 16 GiB KV cache，实际上下文容量由启动时的内存检查确认。
`--reserve-gib 16` 仅用于启动器校验 KV 参数上限，不会额外预留显存。
B64 是活动序列上限，不保证 64 条请求都能同时占满 256K 上下文；实际并发受 KV 容量和调度限制。
`--tokenizer-mode deepseek_v4` 显式选择模型的消息编码方式；`--reasoning-parser deepseek_v4`
将思考与最终回答拆开，供评测记录和答案评分使用。parser 本身不强制开启思考，模式仍由每次请求决定。
修改上下文、KV、runner 或设备需要重启；改变数据集、temperature 或思考模式无需重启。
如果已有容量足够且已配置上述 parser 的四卡 B64、256K 服务，可以复用。
[无 perm 混精四卡吞吐服务](packed_weight_batch_throughput.md) 使用 8192 上下文，需按本节改为 262144 并重启，
才能接收下面的 131072 输出预算。评测期间不要同时在相同卡上运行性能或 NPU CI 测试。

生产入口仅保留完整 FFN。需要检查 eager/graph 差异时，在上述启动命令添加 `--enforce-eager`，
重建服务记录并保持其余评测参数一致。独立数值 oracle 仅用于 [NPU 回归](vq2a8_npu_ci.md)，不作为服务后端。
后文可选的官方 baseline 仍是独立模型服务；同一组卡上的服务依次启动。

终端 B 等待服务就绪并检查名称：

```bash
curl -fsS http://127.0.0.1:8003/health
curl -fsS http://127.0.0.1:8003/v1/models
```

## 上下文、输出预算与采样参数

服务启动命令设置运行容量。**`temperature`、`top_p`、`max_tokens` 和思考模式在精度脚本中设置，
由每个 HTTP 请求传给服务**，不作为 `serve_vq2a8.py` 的服务启动参数。
启动器使用 `--generation-config vllm`，评测命令显式指定的采样参数优先于服务默认值。

| 参数 | 设置位置 | 本文完整回归示例 | 含义 |
| --- | --- | --- | --- |
| `--max-model-len` | 服务端 | `262144`（256K） | 聊天模板处理后的输入与全部生成 token 的总长度上限 |
| `--max-num-batched-tokens` | 服务端 | `8192` | 每个调度步的 token 预算，不是输出上限，也不会把短题目补到 8192 |
| `--max-num-seqs` | 服务端 | `64` | TP4/EP4、DP1 的 engine 最多 64 个活动序列；KV 容量还须满足实际并发请求 |
| `--max-tokens` | 评测客户端 | `131072`（128K） | thinking/no-thinking 均使用此生成上限；thinking 时包括思考和最终回答，遇到正常结束可提前停止 |
| `--temperature` | 评测客户端 | `1` | 使用随机采样；固定 seed 也不保证跨硬件、并行配置完全一致 |
| `--top-p` | 评测客户端 | `1` | 不通过 top-p 截断候选概率质量；与 temperature=1 配合使用 |
| `--seed` / `--repeats` | 评测客户端 | `0` / `20` | 固定随机种子、每题 20 次，与此前完整精度评测口径一致 |
| `--concurrency` | 评测客户端 | `64` | 全局最多 64 个在途请求，匹配 VQ2A8 服务的 B64 配置 |
| `--timeout` | 评测客户端 | `7200` | 每个请求的超时秒数，为长输出留出生成时间 |

必须满足：`输入 token 数（含聊天模板） + 请求的输出预算 <= max_model_len`。
thinking/no-thinking 示例均预留 131072 输出，输入可用空间为 131072（含聊天模板）。
这不是对任意自定义数据集长度的保证。
预检会拒绝输出预算已经占满上下文的配置，逐题是否超限由服务校验；脚本不会自动截短题目。
此前 128/256-token 上下文的短性能服务不能直接用于这里的完整评测。

**GPQA Diamond、AIME24、AIME25 可以使用同一服务和同一套参数。**
一次命令选择多个数据集时，以上生成参数应用于全部数据集；提示词与答案评分由各自的 EvalScope 适配器处理。
先用统一条件建立回归基线。如果某个数据集出现较多输出截断，再单独运行该数据集并增大输出预算，
必要时增大服务上下文；比较 VQ2A8 与原始模型时，该数据集的两边必须使用相同预算和采样参数。
本文日常回归统一使用 temperature=1、131072 输出预算，分别建立 thinking/no-thinking 基线。
比较思考开关的影响时，保持相同输出预算与采样参数，只改变思考模式。
不同预算的结果不能直接归因于思考模式或量化差异；达到输出上限的题目仍计入评分分母，并统计截断数。
本文参数是项目回归条件，不是三个数据集或模型厂商的官方最优成绩配置。

## 终端 B：执行精度评测（离线数据）

以下命令统一读取容器内的 `/mnt/share/g00872988/vq2a8-eval-data` 离线包。
请先复制完整数据目录，保留其中的 `manifest.json` 和 `datasets/`；其他部署路径需同步修改 `--offline-data-dir`。
脚本会校验离线数据后再评测；数据缺失或校验失败时直接报错，不会回退到联网下载。
首次接入可先使用下方的 `--limit 4` 示例检查返回内容、评分和截断情况，再运行完整评测。
完整命令按每题 20 次运行：GPQA Diamond 共 3960 条，AIME24、AIME25 各 600 条。

```bash
# Thinking：三个数据集完整评测
python tools/tests/run_vq2a8_accuracy_suite.py \
  --base-url http://127.0.0.1:8003 \
  --model vq2a8 \
  --datasets gpqa aime24 aime25 \
  --offline-data-dir /mnt/share/g00872988/vq2a8-eval-data \
  --thinking --reasoning-effort high \
  --temperature 1 --top-p 1 --seed 0 --repeats 20 \
  --concurrency 64 \
  --max-tokens 131072 --timeout 7200 \
  --label noperm-mixed-nonrouted-fp8-native-full-ffn-mrv2-tp4-dp1-ep4-b64-c64-thinking-high-t1-r20-o131072 \
  --result-dir benchmark-results/accuracy

# No thinking：复用同一个服务，每题输出上限同样为 131072
python tools/tests/run_vq2a8_accuracy_suite.py \
  --base-url http://127.0.0.1:8003 \
  --model vq2a8 \
  --datasets gpqa aime24 aime25 \
  --offline-data-dir /mnt/share/g00872988/vq2a8-eval-data \
  --no-thinking \
  --temperature 1 --top-p 1 --seed 0 --repeats 20 \
  --concurrency 64 \
  --max-tokens 131072 --timeout 7200 \
  --label noperm-mixed-nonrouted-fp8-native-full-ffn-mrv2-tp4-dp1-ep4-b64-c64-no-thinking-t1-r20-o131072 \
  --result-dir benchmark-results/accuracy

# 只运行 AIME25；--limit 4 明确标记为 smoke，不能作为完整成绩
python tools/tests/run_vq2a8_accuracy_suite.py \
  --base-url http://127.0.0.1:8003 --model vq2a8 \
  --datasets aime25 --thinking --limit 4 --concurrency 4 --repeats 1 \
  --offline-data-dir /mnt/share/g00872988/vq2a8-eval-data \
  --temperature 1 --top-p 1 --max-tokens 131072 --timeout 7200

# 只预览完整命令，不校验离线包、不联网、不创建结果目录
python tools/tests/run_vq2a8_accuracy_suite.py \
  --base-url http://127.0.0.1:8003 --model vq2a8 \
  --offline-data-dir /mnt/share/g00872988/vq2a8-eval-data \
  --temperature 1 --top-p 1 --seed 0 --repeats 20 --max-tokens 131072 --timeout 7200 \
  --datasets gpqa aime24 --no-thinking --concurrency 64 --dry-run
```

也可以使用 `bash tools/tests/run_vq2a8_evalscope_accuracy.sh`，参数相同。
Python 和 Shell 入口都会自动把 EvalScope 子进程的完整 stdout/stderr 合并保存到本次结果目录的
`evalscope.log`，并实时镜像到终端，无需额外使用 `tee`。启动时会打印日志绝对路径，
`run.json` 的 `log` 字段也记录该路径。子进程使用 Python 无缓冲输出；连续 15 秒没有新输出时，
终端显示 `PROBE_WAIT=evalscope` 等待提示。该提示表示进程仍未退出，不代表题目已完成。
本文 VQ2A8 完整评测显式使用 `--concurrency 64`；4 题 smoke 使用并发 4、每题 1 次。
脚本默认并发仍为 4，完整评测需保留命令中的显式参数。
客户端并发只表示同时在途请求数，实际 decode batch 由调度器和 KV 容量决定，
不保证每步达到 B64；超过活动序列上限或 KV 容量的请求会等待调度。
并发增加可能提高总吞吐，但不保证单题更快。对比量化精度时应记录并保持两侧并发条件一致。
脚本请求默认超时 3600 秒，本文命令显式设为 `--timeout 7200`。
按每秒 30 个输出 token 估算，生成满 131072 token 约需 73 分钟，尚未计入输入处理与排队时间；
若实际速度更低或排队较久，应进一步增大超时。提前正常结束的请求无需生成满 128K。

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
top-p 与思考模式沿用服务设置。本文所有精度评测命令显式使用 temperature=1、131072 输出预算。
`--thinking` 本身不会自动提高输出上限；手动传入的 `--max-tokens` 决定本次预算。
输出预算是上限，不会强制每题生成这么多 token。关闭 thinking 后，正文仍可能包含长篇解题过程。
服务上下文足够时，修改输出预算无需重启服务，只影响新启动的评测；正在运行的评测继续使用启动时的配置。
若服务仍使用旧的 65536 上下文，需要先按本文的 256K 配置重启。

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
  --base-url http://127.0.0.1:8003 \
  --model vq2a8 \
  --datasets gpqa aime24 aime25 \
  --offline-data-dir /mnt/share/g00872988/vq2a8-eval-data \
  --thinking --reasoning-effort high \
  --temperature 1 --top-p 1 --seed 0 --repeats 20 \
  --max-tokens 131072 --timeout 7200 \
  --concurrency 64 \
  --label noperm-mixed-nonrouted-fp8-native-full-ffn-mrv2-tp4-dp1-ep4-b64-c64-thinking-high-t1-r20-o131072 \
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

下面使用 **物理卡 4、5、6、7，MRV2、DP4、TP1、EP、B4 decode 图**，端口 `8004`，服务名 `baseline`。
四卡方案沿用仓库中 Ascend 950DT 的部署拓扑，不代表模型的理论最少卡数。
这与前面的 VQ2A8 TP4/EP4、DP1 拓扑不同，两者不能直接作为相同配置的性能对比。
本节保留官方模型的 B4/C4、每题 1 次示例；若与上文 VQ2A8 B64/C64、每题 20 次结果对照，
应先对齐重复次数和并发条件，并分别记录服务拓扑。
请先停止占用这些卡的其他服务，再在终端 A 执行；`BASELINE_MODEL` 改成官方权重的实际目录：

```bash
export BASELINE_MODEL=/mnt/share/g00872988/DeepSeek-V4-Flash-0731
mkdir -p benchmark-results/accuracy-baseline
set -o pipefail

ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 VLLM_USE_V2_MODEL_RUNNER=1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
OMP_PROC_BIND=false HCCL_BUFFSIZE=1024 \
PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
python -u -m vllm.entrypoints.cli.main serve "$BASELINE_MODEL" \
  --served-model-name baseline \
  --host 127.0.0.1 --port 8004 \
  --dtype auto \
  --tensor-parallel-size 1 \
  --data-parallel-size 4 \
  --enable-expert-parallel \
  --api-server-count 1 \
  --max-model-len 262144 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.9 \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --no-async-scheduling \
  --block-size 32 \
  --tokenizer-mode deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --generation-config vllm \
  --seed 0 \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4],"max_cudagraph_capture_size":4,"cudagraph_num_of_warmups":1}' \
  --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":false},"enable_cpu_binding":true,"multistream_overlap_shared_expert":true}' \
  2>&1 | tee benchmark-results/accuracy-baseline/serve-official-0731-mrv2-dp4-b4-compile0.log
```

该命令采用普通解码，不开启 DSpark/MTP。`mode=0` 关闭 torch.compile，显式关闭 `npugraph_ex`；
`FULL_DECODE_ONLY` 保留 `[1,2,4]` decode ACL 图捕获和回放，prefill 分块执行。
仅设置 `FULL_DECODE_ONLY` 不会自动关闭编译优化，两项需要分别配置。
`--max-num-seqs 4` 是每个 DP rank 的活动序列上限，DP4 合计上限为 16；
下面客户端以全局并发 `4` 评测，不是每个 DP rank 各发送 4 个请求。
与 VQ2A8 示例保持相同的 262144 总上下文、8192 prefill chunk、tokenizer 和 reasoning parser。
baseline 使用自动内存规划，启动日志中的 KV cache 容量检查必须通过；本命令尚未在目标 NPU 上验证。
temperature、输出预算和思考模式由下方客户端请求指定，切换数据集或思考模式不需要重启服务。

在终端 B 确认服务就绪，`/v1/models` 应返回 `baseline`，再验证一次真实生成请求：

```bash
curl -fsS http://127.0.0.1:8004/health
curl -fsS http://127.0.0.1:8004/v1/models
curl -fsS --max-time 180 http://127.0.0.1:8004/v1/chat/completions \
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

No-thinking 完整评测，三个数据集统一使用 temperature=1、131072（128K）输出上限：

```bash
python tools/tests/run_evalscope_accuracy.py baseline \
  --base-url http://127.0.0.1:8004 \
  --model baseline \
  --datasets gpqa aime24 aime25 \
  --offline-data-dir /mnt/share/g00872988/vq2a8-eval-data \
  --no-thinking \
  --temperature 1 --top-p 1 --seed 0 --repeats 1 \
  --max-tokens 131072 --timeout 7200 \
  --concurrency 4 \
  --label official-0731-mrv2-dp4-b4-c4-compile0-no-thinking-t1-o131072 \
  --result-dir benchmark-results/accuracy-baseline
```

Thinking 完整评测，复用同一个服务，同样使用 temperature=1、131072（128K）输出上限：

```bash
python tools/tests/run_evalscope_accuracy.py baseline \
  --base-url http://127.0.0.1:8004 \
  --model baseline \
  --datasets gpqa aime24 aime25 \
  --offline-data-dir /mnt/share/g00872988/vq2a8-eval-data \
  --thinking --reasoning-effort high \
  --temperature 1 --top-p 1 --seed 0 --repeats 1 \
  --max-tokens 131072 --timeout 7200 \
  --concurrency 4 \
  --label official-0731-mrv2-dp4-b4-c4-compile0-thinking-high-t1-o131072 \
  --result-dir benchmark-results/accuracy-baseline
```

也可以使用 `bash tools/tests/run_baseline_evalscope_accuracy.sh`，其余参数相同。
两个入口都会实时打印 EvalScope 输出并保存日志；每次运行单独创建结果目录，不覆盖已有评测。
首次检查可追加 `--limit 4`；正式评测移除该选项。离线包缺失或校验失败会报错，不会自动联网下载。

两边使用相同数据包、提示模板、思考配置、采样参数、输出预算和重复次数。
no-thinking 比较两边的 temperature=1 / 131072 预算结果，thinking 比较两边的 high / temperature=1 / 131072 预算结果。
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
