# 原始 DeepSeek-V4-Flash-0731：TP4 / DP4 吞吐测评

适用分支：`test/vq2a8-perf-20261024`；更新日期：2026-11-08。
baseline 权重路径为 `/mnt/share/g00872988/DeepSeek-V4-Flash-0731`，在内网 Linux 仓库根目录执行。

这里的 baseline 是原始 checkpoint：Attention 主矩阵使用 MXFP8，routed MoE 使用 MXFP4/W4A8，不能标记为 BF16 或全 FP8 模型。
原始配置中的 `quant_method=fp8`、`scale_fmt=ue8m0`、`expert_dtype=fp4` 由框架自动识别，直接调用 vLLM，
不指定 `--quantization`，不使用 VQ2A8 启动器或 GS32 构建产物。
部署背景见 [Ascend 官方 DeepSeek-V4-Flash 指南](https://docs.vllm.ai/projects/ascend/en/v0.27.1rc1/tutorials/models/DeepSeek-V4-Flash.html)。
仍需正确安装当前环境的 vLLM、vllm-ascend 和模型所需的 Ascend 原生算子。

本文件提供可复制的启动和吞吐测试脚本；答题精度评测见 [baseline 精度命令](vq2a8_accuracy.md#与原始模型对比)。
先停止占用同一组四卡的其他服务，再选择 TP4 或 DP4。不要在同一组卡上同时运行两套服务。
下面使用物理卡 `4,5,6,7`；更换卡号时同步修改启动环境和压测备注。

## 0. 起始配置与比较口径

两套配置与 [VQ2A8 四卡吞吐测试](packed_weight_batch_throughput.md)的起始容量对齐：

| 参数 | baseline TP4 | baseline DP4 |
| --- | --- | --- |
| 并行拓扑 | TP4 / DP1 / EP4 | TP1 / DP4 / EP4 |
| `max-num-seqs` | 64 | 每 engine 16，全局上限 64 |
| `max-num-batched-tokens` | 8192 | 每 engine 2048，合计上限 8192 |
| decode 图桶 | `[1,2,4,8,16,32,64]` | 每 engine `[1,2,4,8,16]` |
| HTTP 端口 | 8006 | 8007 |
| 模型服务名 | `baseline` | `baseline` |

共同设置：MRV2、总上下文 8192、每 worker KV 16 GiB、block size 128、显存比例 0.9，
关闭 prefix cache 和异步调度，不开启 MTP/DSpark；`mode=0` 关闭 torch.compile，
`FULL_DECODE_ONLY` 保留 decode 图，prefill 分块 eager 执行，并显式关闭 `npugraph_ex`。

baseline 保留 CPU binding 和共享专家多流重叠优化，压测备注中会记录这些设置；
这是原始模型部署与 VQ2A8 部署的对照，不应把所有差异归因于量化位宽。
DP4 复制 Attention、共享专家等普通层，routed experts 仍按 EP4 分片；相同 KV 字节预算不保证相同有效 token 容量。
这些命令已按当前源码核对，目标 NPU 上的启动、图回放、精度及峰值显存需要实际验证。

## 1. 终端 A：启动 baseline TP4

服务目录以时间标签结尾，每次启动更新 `current-baseline-tp4-8006` 链接。

```bash
set -o pipefail
export BASELINE_MODEL=/mnt/share/g00872988/DeepSeek-V4-Flash-0731
RESULT_ROOT="$PWD/benchmark-results/baseline-batch-throughput"
SERVE_DIR="$RESULT_ROOT/serve-official-0731-tp4-ep4-b64-$(date +%Y%m%d_%H%M%S)"
SERVE_LINK="$RESULT_ROOT/current-baseline-tp4-8006"
mkdir -p "$RESULT_ROOT" &&
mkdir "$SERVE_DIR" &&
git rev-parse HEAD > "$SERVE_DIR/commit.txt" &&
git status --short > "$SERVE_DIR/worktree.txt" &&
printf '%s\n' "$BASELINE_MODEL" > "$SERVE_DIR/model-path.txt" &&
printf '%s\n' 'TP4/DP1/EP4; B64; batched_tokens=8192; context=8192; KV=16GiB/worker; port=8006' > "$SERVE_DIR/settings.txt" &&
sha256sum "$BASELINE_MODEL/config.json" "$BASELINE_MODEL/model.safetensors.index.json" > "$SERVE_DIR/model-files.sha256" &&
echo "服务日志：$SERVE_DIR/serve.log" &&
ln -sfnT "$SERVE_DIR" "$SERVE_LINK" &&

TMPDIR=/tmp VLLM_RPC_BASE_PATH=/tmp \
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 \
VLLM_USE_V2_MODEL_RUNNER=1 VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
VLLM_WORKER_MULTIPROC_METHOD=spawn OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
OMP_PROC_BIND=false HCCL_BUFFSIZE=1024 \
PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
python -u -m vllm.entrypoints.cli.main serve "$BASELINE_MODEL" \
  --served-model-name baseline \
  --host 127.0.0.1 --port 8006 \
  --dtype auto --load-format safetensors \
  --tensor-parallel-size 4 --data-parallel-size 1 \
  --pipeline-parallel-size 1 --distributed-executor-backend mp \
  --enable-expert-parallel --expert-placement-strategy linear \
  --api-server-count 1 \
  --max-model-len 8192 \
  --max-num-batched-tokens 8192 --max-num-seqs 64 \
  --kv-cache-memory-bytes 17179869184 --gpu-memory-utilization 0.9 \
  --enable-chunked-prefill --no-enable-prefix-caching --no-async-scheduling \
  --block-size 128 \
  --tokenizer-mode deepseek_v4 --reasoning-parser deepseek_v4 \
  --generation-config vllm --seed 0 \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,32,64],"max_cudagraph_capture_size":64,"cudagraph_num_of_warmups":1}' \
  --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":false},"enable_cpu_binding":true,"multistream_overlap_shared_expert":true}' \
  2>&1 | tee "$SERVE_DIR/serve.log"
```

## 2. 终端 A：启动 baseline DP4

使用 TP1/DP4/EP4、四个本地 engine、内部 MP 分发和单 API 服务。
`max-num-seqs=16`、token 预算 2048 均是每个 engine 的限制，不是整个服务的限制。

```bash
set -o pipefail
export BASELINE_MODEL=/mnt/share/g00872988/DeepSeek-V4-Flash-0731
RESULT_ROOT="$PWD/benchmark-results/baseline-batch-throughput"
SERVE_DIR="$RESULT_ROOT/serve-official-0731-dp4-ep4-b16-$(date +%Y%m%d_%H%M%S)"
SERVE_LINK="$RESULT_ROOT/current-baseline-dp4-8007"
mkdir -p "$RESULT_ROOT" &&
mkdir "$SERVE_DIR" &&
git rev-parse HEAD > "$SERVE_DIR/commit.txt" &&
git status --short > "$SERVE_DIR/worktree.txt" &&
printf '%s\n' "$BASELINE_MODEL" > "$SERVE_DIR/model-path.txt" &&
printf '%s\n' 'TP1/DP4/EP4; local_B=16; batched_tokens=2048/engine; context=8192; KV=16GiB/worker; port=8007' > "$SERVE_DIR/settings.txt" &&
sha256sum "$BASELINE_MODEL/config.json" "$BASELINE_MODEL/model.safetensors.index.json" > "$SERVE_DIR/model-files.sha256" &&
echo "服务日志：$SERVE_DIR/serve.log" &&
ln -sfnT "$SERVE_DIR" "$SERVE_LINK" &&

TMPDIR=/tmp VLLM_RPC_BASE_PATH=/tmp \
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 \
VLLM_USE_V2_MODEL_RUNNER=1 VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
VLLM_WORKER_MULTIPROC_METHOD=spawn OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
OMP_PROC_BIND=false HCCL_BUFFSIZE=1024 \
PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
python -u -m vllm.entrypoints.cli.main serve "$BASELINE_MODEL" \
  --served-model-name baseline \
  --host 127.0.0.1 --port 8007 \
  --dtype auto --load-format safetensors \
  --tensor-parallel-size 1 --data-parallel-size 4 \
  --data-parallel-size-local 4 --data-parallel-backend mp \
  --pipeline-parallel-size 1 --distributed-executor-backend mp \
  --enable-expert-parallel --expert-placement-strategy linear \
  --api-server-count 1 \
  --max-model-len 8192 \
  --max-num-batched-tokens 2048 --max-num-seqs 16 \
  --kv-cache-memory-bytes 17179869184 --gpu-memory-utilization 0.9 \
  --enable-chunked-prefill --no-enable-prefix-caching --no-async-scheduling \
  --block-size 128 \
  --tokenizer-mode deepseek_v4 --reasoning-parser deepseek_v4 \
  --generation-config vllm --seed 0 \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16],"max_cudagraph_capture_size":16,"cudagraph_num_of_warmups":1}' \
  --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":false},"enable_cpu_binding":true,"multistream_overlap_shared_expert":true}' \
  2>&1 | tee "$SERVE_DIR/serve.log"
```

## 3. 终端 B：扫描吞吐、TTFT 和 TPOT

同一套客户端命令支持两种服务：测试第 1 节时设置 `MODE=tp4`，测试第 2 节时设置 `MODE=dp4`。
等待服务就绪后，在同一容器、同一仓库根目录执行；服务目录由对应链接自动读取。

复用 `run_vq2a8_throughput.py` 的负载和统计实现：随机输入 1024、输出 128 tokens，
流式 `/v1/completions`、temperature=0、seed=0、ignore-eos，每档请求数为 `4×并发`，预热 1 次，正式测量 1 次。
脚本名称中的 VQ2A8 不限制模型类型，客户端明确使用服务名 `baseline`。

```bash
(
set -euo pipefail
MODE=tp4  # 测试 DP4 时改为 dp4
MODEL=/mnt/share/g00872988/DeepSeek-V4-Flash-0731
RESULT_ROOT="$PWD/benchmark-results/baseline-batch-throughput"
case "$MODE" in
  tp4)
    PORT=8006
    PARALLEL_MODE=tp-ep
    MAX_NUM_SEQS=64
    TOPOLOGY='TP4/DP1/EP4; B64; batched_tokens=8192'
    ;;
  dp4)
    PORT=8007
    PARALLEL_MODE=dp-ep
    MAX_NUM_SEQS=16
    TOPOLOGY='TP1/DP4/EP4; local_B=16; batched_tokens=2048/engine'
    ;;
  *) echo "MODE 必须为 tp4 或 dp4" >&2; exit 1 ;;
esac
BASE_URL="http://127.0.0.1:$PORT"
if ! SERVE_DIR=$(readlink -e "$RESULT_ROOT/current-baseline-$MODE-$PORT") || \
   [ ! -s "$SERVE_DIR/model-path.txt" ]; then
  echo "未找到本轮 baseline 启动记录，请先启动对应服务并等待就绪。" >&2
  exit 1
fi
RUN_DIR="$RESULT_ROOT/$(date +%Y%m%d_%H%M%S)-bench-official-0731-$MODE-ep4-b$MAX_NUM_SEQS"
mkdir "$RUN_DIR"
echo "服务日志：$SERVE_DIR/serve.log"
echo "结果目录：$RUN_DIR"
git rev-parse HEAD > "$RUN_DIR/commit.txt"
git status --short > "$RUN_DIR/worktree.txt"
printf '%s\n' "$SERVE_DIR" > "$RUN_DIR/service-dir.txt"
for name in commit.txt worktree.txt model-path.txt settings.txt model-files.sha256; do
  cp -- "$SERVE_DIR/$name" "$RUN_DIR/server-$name"
done

curl -fsS --max-time 30 "$BASE_URL/health"
curl -fsS --max-time 30 -D "$RUN_DIR/models.headers" \
  "$BASE_URL/v1/models" > "$RUN_DIR/models.json"
python - "$RUN_DIR/models.json" "$MODEL" "$RUN_DIR/server-model-path.txt" <<'PY'
import json
import sys
from pathlib import Path

snapshot = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = sys.argv[2]
recorded = Path(sys.argv[3]).read_text(encoding="utf-8").strip()
models = snapshot.get("data") if isinstance(snapshot, dict) else None
if not isinstance(models, list):
    raise SystemExit("/v1/models did not return a model list")
matches = [model for model in models if isinstance(model, dict) and model.get("id") == "baseline"]
if recorded != expected or len(matches) != 1 or matches[0].get("root") != expected:
    raise SystemExit("baseline id/root does not match the requested model and startup record")
print("Verified baseline model:", expected)
PY

python -u tools/tests/run_vq2a8_throughput.py \
  --base-url "$BASE_URL" --model baseline --tokenizer "$MODEL" \
  --tokenizer-mode deepseek_v4 \
  --parallel-mode "$PARALLEL_MODE" --ep-size 4 --max-num-seqs "$MAX_NUM_SEQS" \
  --concurrency 1 2 4 8 16 32 64 \
  --input-len 1024 --output-len 128 \
  --num-warmups 1 --repeats 1 --seed 0 \
  --result-dir "$RUN_DIR" \
  --note "original-0731; model=$MODEL; devices=4,5,6,7; $TOPOLOGY; MRV2; full_graph decode; no MTP/DSpark; KV=16GiB/worker; context=8192; block=128; CPU binding; shared expert overlap" \
  2>&1 | tee "$RUN_DIR/benchmark.log"
)
```

这里不传 `--runtime-record`：当前该选项只接受 VQ2A8 native/full_ffn 的 ABI 记录，不能用于原始模型。
上面的独立检查验证 `/v1/models` 中的服务名和权重路径；它不证明实际服务器代码版本、拓扑或算子执行。
`current-baseline-*` 链接在服务退出后仍会保留，重启时重新执行对应启动命令；整个测量期间不要更换服务。
启动日志记录实际配置，`settings.txt` 是人工声明，改变参数时应同步更新它、目录标签和 `--note`。

## 4. 结果与容量探索

压测脚本打印的 `Results: ...` 为本轮内部结果目录，包含 `summary.json`、`summary.csv`、每档 `c<C>-r<R>.json/.log` 和 `manifest.json`。
指标与 VQ2A8 相同：输出 token/s、请求/s、平均及 P99 TTFT/TPOT、P99 E2EL。
输出 TPS 是总输出 token 数除以测量时间；TTFT 包含排队和 prefill；TPOT 为首 token 后的平均输出间隔。
检查 `complete=true`、所有档位 `valid=true` 且失败数为 0；正式比较时保留所有重复结果。

因为没有传 VQ2A8 运行记录，baseline 的汇总不会自动附带 `model_path` 或 `server_runtime`。
请一起保留外层 `RUN_DIR` 的 `models.json`、`server-*` 文件、`service-dir.txt` 和相应服务的 `serve.log`。
客户端 commit 与归档的服务 commit 分别记录；二者及 `/v1/models` 都不能代替设备执行验证。

起始配置用于建立同负载对照，不代表任一拓扑的峰值。先分别复测 TP4 和 DP4，再允许两者独立调优：

| DP4 每 engine B | 全局活动序列上限 | decode 图桶 | 建议增加的客户端并发 |
| --- | --- | --- | --- |
| 16 | 64 | `[1,2,4,8,16]` | `96 128`，用于观察 B16 服务的排队与饱和 |
| 32 | 128 | `[1,2,4,8,16,32]` | `96 128` |
| 64 | 256 | `[1,2,4,8,16,32,64]` | `96 128 192 256` |

扩大 DP4 时同步修改服务的 `--max-num-seqs`、`cudagraph_capture_sizes`、`max_cudagraph_capture_size`，
以及客户端 `dp4` 分支的 `MAX_NUM_SEQS`、`TOPOLOGY` 和 `--concurrency`；重启后重新捕获图并检查显存。
这些是容量探索配置，未在目标机实测。仅增加客户端并发不会扩大服务活动容量。
当前压测 wrapper 限制每 engine B≤64，这不代表原始模型的框架上限；TP4 的 B>64 不在本文件脚本覆盖范围。

KV 已显式设为 16 GiB，单独提高 `--gpu-memory-utilization` 不会增加 KV；如需增加，调整 `--kv-cache-memory-bytes` 并重新测量峰值显存。
token 预算和 KV 可分别调优，但要记录实际配置。候选峰值建议使用更多请求、至少 3 次重复复测；
在线服务比较可增加 `--max-p99-ttft-ms`、`--max-p99-tpot-ms`，两侧使用相同延迟门槛。
未提供门槛时，`meets_slo=true` 不表示通过业务延迟要求，`peak_measured_with_slo` 为 `null`。
保留同并发曲线，同时报告同四卡、同工作负载和相同质量/延迟要求下各自最佳吞吐。
