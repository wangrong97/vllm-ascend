# 无 perm VQ2A8 混精模型：98G 单卡并发吞吐测试

在 Linux 容器的仓库根目录执行，使用现有 `-export-20261103` 权重和 ABI v4 完整 FFN 产物。
本页测试 **TP1/EP1、DP1、MRV2、native full_ffn**：单卡持有全部专家，decode 使用图回放，prefill 使用 eager 完整融合。
服务和客户端均复用现有脚本，不需要更改权重或重新 repack。
构建命令见[重建 ABI v4](packed_weight_batch_throughput.md#0-首次升级完整-ffn重建-abi-v4)，
多卡对照见[四卡吞吐测试](packed_weight_batch_throughput.md)。

## 1. 显存与起步配置

`--memory-fraction 0.97` 会传给 vLLM 的 `--gpu-memory-utilization 0.97`。
这里指显存配置比例，不是保证 NPU 计算利用率达到 97%。当前启动器同时显式传入 KV 字节数：
**KV 大小由 `--kv-cache-mib` 决定，不会按 0.97 自动使用剩余显存。**
worker 仍会在加载权重前检查空闲显存是否达到设备总显存的 97%；显式 KV 模式下，这个比例也不是运行时总显存的硬上限。

本地该权重 index 的 `total_size` 为 96,025,092,316 字节，约 **89.43 GiB**。
这是 checkpoint 张量载荷，不等于模型加载峰值或 NPU 最终驻留显存；还要容纳格式转换、激活、算子工作区、图和 KV。
若设备实际总容量为 98 GiB，97% 为 95.06 GiB，因此单卡先使用 1 GiB KV 和较小的 prefill 调度预算。
实际容量及 GB/GiB 单位以设备和启动日志报告为准；本配置尚未做单卡 NPU 实测，98G 不代表一定能完成 B64 捕获。

| 参数 | 单卡起步值 |
| --- | --- |
| 物理卡 | 示例为 `4`，只暴露一张空闲卡 |
| 并行 | `--parallel-mode tp-ep --ep-size 1`，实际 TP1/EP1/DP1 |
| 显存比例 | `--memory-fraction 0.97` |
| KV | `--kv-cache-mib 1024 --reserve-gib 1`，显式 1 GiB |
| 上下文上限 | `--max-model-len 2048`，覆盖 1024 输入加 128 输出 |
| 每步 token 预算 | `--max-num-batched-tokens 1024` |
| 活动序列上限 | `--max-num-seqs 64`，捕获 `[1,2,4,8,16,32,64]` |
| 请求 | 输入 1024、输出 128 tokens，客户端并发 1～64 |
| 采样与测量 | temperature=0、seed=0、ignore-eos；每档每轮 4×并发条请求，预热 1 条，正式重复 3 轮 |

`--reserve-gib 1` 只校验 KV 参数上限，不额外保留 1 GiB。
B64 和 C64 分别是活动序列上限、客户端在途请求上限；KV 不足时可能排队或抢占，并不保证每步都有 64 条 decode。
本页保留与四卡测试相同的请求长度和并发档位，但 **KV 从 16 GiB/worker 改为 1 GiB，调度预算从 8192 改为 1024**，
上下文上限从 8192 改为 2048。
比较结果时需要记录这些差异，不能把吞吐比例全部归因于卡数。

## 2. 终端 A：启动单卡服务

先停止占用物理卡 4 或端口 8003 的旧服务。以下命令不使用外层子 shell 括号；服务目录以时间结尾。
单卡结果使用独立根目录，终端 B 自动读取本轮启动记录。

```bash
set -o pipefail
export MODEL=/mnt/share/g00872988/DeepSeek-V4-Flash-0731-32x32-noperm-hot5050125-identity-v3-export-20261103
export BUILD="$PWD/build/vq2a8-gs32-runtime-ffn"
RESULT_ROOT="$PWD/benchmark-results/single-card-batch-throughput"
SERVE_DIR="$RESULT_ROOT/serve-noperm-mixed-nonrouted-fp8-native-tp1-b64-$(date +%Y%m%d_%H%M%S)"
SERVE_LINK="$RESULT_ROOT/current-native-8003"
mkdir -p "$RESULT_ROOT" &&
mkdir "$SERVE_DIR" &&
git rev-parse HEAD > "$SERVE_DIR/commit.txt" &&
git status --short > "$SERVE_DIR/worktree.txt" &&
sha256sum "$MODEL/config.json" "$MODEL/model.safetensors.index.json" > "$SERVE_DIR/model-files.sha256" &&
sha256sum "$BUILD/libvq2a8_gs32_stage.so" "$BUILD/mixed_w2w4_moe_ffn.o" > "$SERVE_DIR/gs32-files.sha256" &&
echo "服务日志：$SERVE_DIR/serve.log" &&
ln -sfnT "$SERVE_DIR" "$SERVE_LINK" &&

TMPDIR=/tmp VLLM_RPC_BASE_PATH=/tmp \
ASCEND_RT_VISIBLE_DEVICES=4 \
VLLM_USE_V2_MODEL_RUNNER=1 VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
VLLM_WORKER_MULTIPROC_METHOD=spawn OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -u tools/serve_vq2a8.py \
  --model "$MODEL" --no-dspark \
  --gs32-runtime-dir "$BUILD" \
  --parallel-mode tp-ep --ep-size 1 \
  --max-model-len 2048 \
  --max-num-batched-tokens 1024 --max-num-seqs 64 \
  --memory-fraction 0.97 --kv-cache-mib 1024 --reserve-gib 1 \
  --tokenizer-mode deepseek_v4 --reasoning-parser deepseek_v4 \
  --runtime-record "$SERVE_DIR/service-runtime.json" \
  --host 127.0.0.1 --port 8003 \
  2>&1 | tee "$SERVE_DIR/serve.log"
```

启动器在 EP1 下选择 `uni` executor，不启用多卡专家通信。日志中的 `world_size` 应为 `1`，
服务启动记录应为 `native/full_ffn/full_graph`、ABI v4。
等待全部图捕获和服务启动完成，再执行压测。链接存在并不代表服务就绪；终端 B 还会检查 `/health`。
重复启动会更新链接，旧日志目录保留；压测期间保持同一服务。

## 3. 终端 B：扫描并发 1～64

在同一容器的同一仓库根目录执行，无需手填 `SERVE_DIR`。
使用与服务一致的 Python/vLLM 环境，暂停其他请求；如改了物理卡或服务预算，同步更新 `--note`。

```bash
set -o pipefail
MODEL=/mnt/share/g00872988/DeepSeek-V4-Flash-0731-32x32-noperm-hot5050125-identity-v3-export-20261103
BASE_URL=http://127.0.0.1:8003
RESULT_ROOT="$PWD/benchmark-results/single-card-batch-throughput"
if ! SERVE_DIR=$(readlink -e "$RESULT_ROOT/current-native-8003") || \
   [ ! -s "$SERVE_DIR/service-runtime.json" ]; then
  echo "未找到单卡启动记录，请先按第 2 节启动服务并等待就绪。" >&2
else
  RUN_DIR="$RESULT_ROOT/bench-noperm-mixed-nonrouted-fp8-native-tp1-ep1-b64-$(date +%Y%m%d_%H%M%S)"
  mkdir "$RUN_DIR" &&
  echo "服务记录：$SERVE_DIR/service-runtime.json" &&
  echo "结果目录：$RUN_DIR" &&
  git rev-parse HEAD > "$RUN_DIR/commit.txt" &&
  git status --short > "$RUN_DIR/worktree.txt" &&
  curl -fsS --max-time 30 "$BASE_URL/health" &&
  curl -fsS --max-time 30 -D "$RUN_DIR/models.headers" \
    "$BASE_URL/v1/models" > "$RUN_DIR/models.json" &&

  python -u tools/tests/run_vq2a8_throughput.py \
    --base-url "$BASE_URL" --model vq2a8 --tokenizer "$MODEL" \
    --tokenizer-mode deepseek_v4 \
    --parallel-mode tp-ep --ep-size 1 --max-num-seqs 64 \
    --concurrency 1 2 4 8 16 32 64 \
    --input-len 1024 --output-len 128 \
    --num-warmups 1 --repeats 3 --seed 0 \
    --result-dir "$RUN_DIR" \
    --runtime-record "$SERVE_DIR/service-runtime.json" \
    --note 'nonrouted-fp8; backend=native; device=4; TP1/EP1/DP1; MRV2; full_graph; fusion=full_ffn; no DSpark; context=2048; batched_tokens=1024; KV=1GiB; memory_fraction=0.97' \
    2>&1 | tee "$RUN_DIR/benchmark.log"
fi
```

先做一轮快速扫描时可改为 `--repeats 1`；正式结果保留 3 轮并报告中位数。
每档每轮测量请求数仍为 `4×并发`，更改 repeats 不改变单轮负载。
客户端的 `--ep-size 1` 只记录拓扑声明，不会将多卡服务转换为单卡服务。
启动记录与 `/v1/models` 的核对范围、指标定义沿用[四卡结果说明](packed_weight_batch_throughput.md#3-结果)。

## 4. 结果与显存调整

脚本打印的 `Results: ...` 子目录中包含 `summary.json`、`summary.csv`、`manifest.json`、
`service-runtime.json`、`models.json` 及每档每轮的原始 JSON/日志。
记录输出 tokens/s、请求吞吐、平均和 P99 TTFT/TPOT、P99 E2EL；同时保留终端 A 的 `serve.log`，
用于核对单卡模型加载显存、图捕获时间、KV 容量和抢占情况。请求完成不等同于模型精度验收。

按发生问题的阶段调整，每次重启服务都创建新的目录并更新测试备注：

- **权重加载前空闲显存检查失败**：先检查卡上其他进程及实际总容量；0.97 会要求启动时至少 97% 总显存空闲。
- **加载权重时 OOM**：先核对模型路径、原生产物、设备占用和加载峰值；此时降低客户端并发无法减少权重本身。
- **profile/prefill OOM**：先把 `--max-num-batched-tokens 1024` 降到 `512`，保留图执行路径。
- **B64 图捕获 OOM**：把服务 `--max-num-seqs` 降到 `32`，客户端同步设为 `32`、扫描 `1 2 4 8 16 32`；
  结果标签改为 B32，不能继续标记为 B64。
- **运行成功但 KV 抢占明显**：若实测显存仍有余量，可尝试 `--kv-cache-mib 2048 --reserve-gib 2`，
  重新验证捕获和压测；不要直接照搬四卡的 16 GiB KV。

显存有余量后，可另起一轮尝试 `--max-num-batched-tokens 8192`，更接近四卡的调度设置。
增加 token 预算或 KV 都需要重新验证峰值显存，并在结果中记录实际配置。
