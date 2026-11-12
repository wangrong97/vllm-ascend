#!/usr/bin/env bash
# =============================================================================
# E_new: baseline(FP8) vs W2A8 精度对比 —— 官方 baseline 形态编排(不运行)
#
# 两个实验(官方 vq2a8_accuracy 风格, 各用全部 8 卡 / 端口 8003-8004):
#   B1  baseline(FP8 原始): DP8·compile0·图模式(capture 1,2,4)·port8004
#       serve: python -m vllm.entrypoints.cli.main serve + FULL_DECODE_ONLY
#       eval : run_evalscope_accuracy.py baseline (gpqa/aime24/aime25, repeats=5, concurrency=8)
#   B2  W2A8(packed-native): tp-ep EP8·full_graph·native backend·port8003
#       serve: tools/serve_vq2a8.py(官方脚本, 走 Vq2a8Config 完整校验链)
#       eval : run_vq2a8_accuracy_suite.py (gpqa/aime24/aime25, repeats=5, concurrency=8)
#
# 执行顺序: B1 serve→eval→停 → B2 serve→eval→停 → 汇总对比表
# 断点续跑: 记录每实验每数据集已完成轮数(b1/b2 目录下 *.done 标记),
# 被杀重启后只补缺失轮次; 每个数据集累计跑满 5 遍为止。
# 用法: bash tools/eplb/run_baseline_vs_w2a8.sh [--smoke-only] [--skip-b1|--skip-b2]
# =============================================================================
set -uo pipefail
REPO=/mnt/share/w00608002/work/eplb/vllm-ascend
cd "$REPO"

# /dev/shm 守门: DP 进程间通信需共享内存(容器重置后回落 64M 会导致 serve 启动失败)
shm_size_guard() {
  local avail
  avail=$(df -m /dev/shm 2>/dev/null | awk 'NR==2{print $4}')
  if [[ -n "${avail:-}" && "$avail" -lt 1024 ]]; then
    mount -o remount,size=16G /dev/shm 2>/dev/null || echo "[shm-guard] remount /dev/shm 失败(权限?), 继续" >&2
  fi
}
shm_size_guard

# 所有产物统一归档到 output/(便于回溯): 含 serve.log/eval.log/精度结果
OUT=/mnt/share/w00608002/work/eplb/output/baseline_vs_w2a8_$(date +%m%d_%H%M)
mkdir -p "$OUT"
echo "$OUT" > /tmp/baseline_vs_w2a8_dir
# 运行脚本快照随本次运行归档(便于复现当时配置)
mkdir -p "$OUT/scripts"
cp /mnt/share/w00608002/work/eplb/vllm-ascend/tools/eplb/eval/run_baseline_vs_w2a8.sh "$OUT/scripts/" 2>/dev/null || true
cp /mnt/share/w00608002/work/eplb/output/scripts/run_fp8_dp8.sh "$OUT/scripts/" 2>/dev/null || true
cp /mnt/share/w00608002/work/eplb/output/scripts/run_w2a8_graph.sh "$OUT/scripts/" 2>/dev/null || true
echo "编排: baseline(FP8) vs W2A8 官方形态 $(date '+%F %T')" | tee "$OUT/summary.txt"

# ---- 配置(按用户给定的 baseline/W2A8 片段) ----
BASELINE_MODEL=/mnt/share/w00608002/weight/DeepSeek-V4-Flash-0731
MODEL=${MODEL:-/mnt/share/w00608002/weight/DeepSeek-V4-Flash-0731-vq2a8-mixed-gs32-repack}
BUILD="$PWD/build/vq2a8-gs32-runtime"
EVAL_DATA=/mnt/share/w00608002/weight/vq2a8-eval-data

# 前置检查
[[ -d "$BASELINE_MODEL" ]] || { echo "缺 baseline 模型: $BASELINE_MODEL"; exit 1; }
[[ -f "$BUILD/libvq2a8_gs32_stage.so" && -f "$BUILD/mixed_w2w4_moe_ffn.o" ]] || { echo "缺 GS32 产物: $BUILD"; exit 1; }
[[ -d "$EVAL_DATA" ]] || echo "警告: 评测数据目录不存在: $EVAL_DATA(先检查)"

# ---- 通用函数 ----
wait_free() {
  local c=0
  while true; do
    local pids hbm
    pids=$(npu-smi info 2>/dev/null | awk '/Process id/{f=1;next} f&&/^\| [0-9]/{print $3}' | head -8)
    hbm=$(npu-smi info 2>/dev/null | grep -oE "[0-9]+ / 98304" | awk '{if ($1>5000) c++} END{print c+0}')
    if [[ -z "$pids" && "${hbm:-1}" -eq 0 ]]; then c=$((c+1)); else c=0; fi
    [[ $c -ge 2 ]] && return 0
    sleep 30
  done
}

wait_ready() {  # $1=port $2=log $3=name
  local port=$1 log=$2 name=$3 deadline=$((SECONDS+1800))
  while (( SECONDS < deadline )); do
    curl -s --max-time 5 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1 && { echo "$name 就绪 $(date '+%F %T')" | tee -a "$OUT/summary.txt"; return 0; }
    grep -qE "ValueError|RuntimeError|ACL stream|Traceback" "$log" 2>/dev/null && {
      echo "$name 启动失败: $(grep -E 'ValueError|RuntimeError|ACL stream' "$log" | head -1 | cut -c1-160)" | tee -a "$OUT/summary.txt"; return 1; }
    sleep 15
  done
  echo "$name 就绪超时(30min)" | tee -a "$OUT/summary.txt"; return 1
}

smoke() {  # $1=port $2=model_name $3=tag
  local port=$1 model=$2 tag=$3
  : > "$OUT/${tag}_smoke.txt"
  curl -s --max-time 300 "http://127.0.0.1:$port/v1/chat/completions" -H "Content-Type: application/json" \
    -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"计算 23 * 47 等于多少\"}],\"max_tokens\":200}" >> "$OUT/${tag}_smoke.txt"
  grep -q "1081" "$OUT/${tag}_smoke.txt" && echo "$tag smoke ✅" | tee -a "$OUT/summary.txt" || { echo "$tag smoke ❌ 无1081,跳过评测" | tee -a "$OUT/summary.txt"; return 1; }
}

# 续跑: $1=实验目录(b1/b2) $2=数据集 $3=命令模板函数名 -> 返回还需跑的轮数
need_rounds() {
  local dir=$1 ds=$2
  local done=0
  # 统计 dir 下所有 accuracy_audit.json 中该数据集已完成轮数
  for f in "$dir"/*/accuracy_audit.json "$dir"/*/*/accuracy_audit.json; do
    [[ -f "$f" ]] || continue
    local recs
    recs=$(python3 -c "import json,sys; d=json.load(open('$f')); print(d.get('datasets',{}).get('$ds',{}).get('records',0))" 2>/dev/null || echo 0)
    done=$((done + recs))
  done
  local total=5   # 每数据集目标轮数
  local need=$((total - done))
  [[ $need -lt 0 ]] && need=0
  echo "$need"
}

cleanup_all() {
  ps -ef | grep -E "[v]llm|[s]erve_vq2a8|[e]valscope|[a]ccuracy_suite" | awk '{print $2}' | while read p; do
    case "$(readlink /proc/$p/exe 2>/dev/null)" in *python*) kill -9 "$p" 2>/dev/null;; esac
  done
  sleep 10
}
trap cleanup_all EXIT INT TERM

SKIP_B1=0; SKIP_B2=0; SMOKE_ONLY=0
for a in "$@"; do
  case "$a" in --skip-b1) SKIP_B1=1;; --skip-b2) SKIP_B2=1;; --smoke-only) SMOKE_ONLY=1;; esac
done

# ============ B1: baseline FP8 ============
if [[ $SKIP_B1 -eq 0 ]]; then
  echo "===== B1 baseline(FP8) $(date '+%F %T') =====" | tee -a "$OUT/summary.txt"
  wait_free
  mkdir -p "$OUT/b1"
  ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 VLLM_USE_V2_MODEL_RUNNER=1 \
  VLLM_WORKER_MULTIPROC_METHOD=spawn OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OMP_PROC_BIND=false HCCL_BUFFSIZE=1024 PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
  python -u -m vllm.entrypoints.cli.main serve "$BASELINE_MODEL" \
    --served-model-name baseline --host 127.0.0.1 --port 8004 \
    --dtype auto --tensor-parallel-size 1 --data-parallel-size 8 --enable-expert-parallel \
    --api-server-count 1 --max-model-len 262144 --max-num-batched-tokens 8192 --max-num-seqs 4 \
    --gpu-memory-utilization 0.9 --enable-chunked-prefill --no-enable-prefix-caching \
    --no-async-scheduling --block-size 32 --tokenizer-mode deepseek_v4 --reasoning-parser deepseek_v4 \
    --generation-config vllm --seed 0 \
    --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4],"max_cudagraph_capture_size":4,"cudagraph_num_of_warmups":1}' \
    --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":false},"enable_cpu_binding":true,"multistream_overlap_shared_expert":true}' \
    > "$OUT/b1/serve.log" 2>&1 &
  B1PID=$!
  wait_ready 8004 "$OUT/b1/serve.log" "B1" || exit 1
  smoke 8004 baseline "b1" || exit 1
  if [[ $SMOKE_ONLY -eq 0 ]]; then
    echo "B1 评测开始(续跑,目标每数据集5遍) $(date '+%F %T')" | tee -a "$OUT/summary.txt"
    for ds in gpqa aime24 aime25; do
      need=$(need_rounds "$OUT/b1" "$ds")
      echo "B1 $ds: 已完成轮次记录,$need 轮待跑" | tee -a "$OUT/summary.txt"
      if [[ "$need" -gt 0 ]]; then
        python tools/tests/run_evalscope_accuracy.py baseline \
          --base-url http://127.0.0.1:8004 --model baseline \
          --datasets "$ds" --offline-data-dir "$EVAL_DATA" \
          --thinking --reasoning-effort high --temperature 1 --top-p 1 --seed 0 --repeats "$need" \
          --max-tokens 131072 --timeout 7200 --concurrency 16 \
          --label official-0731-mrv2-dp8-b4-c8-compile0-thinking-high-t1-r${need}-o131072 \
          --result-dir "$OUT/b1" >> "$OUT/b1/eval.log" 2>&1
        echo "B1 $ds 本轮完成(rc=$?) $(date '+%F %T')" | tee -a "$OUT/summary.txt"
      fi
    done
    echo "B1 评测完成 $(date '+%F %T')" | tee -a "$OUT/summary.txt"
  fi
  cleanup_all; sleep 5
fi

# ============ B2: W2A8 packed-native ============
if [[ $SKIP_B2 -eq 0 ]]; then
  echo "===== B2 W2A8(packed-native) $(date '+%F %T') =====" | tee -a "$OUT/summary.txt"
  wait_free
  mkdir -p "$OUT/b2"
  SERVE_DIR=$(mktemp -d "$OUT/b2/serve-XXXXXX")
  git rev-parse HEAD > "$SERVE_DIR/commit.txt" 2>/dev/null || true
  TMPDIR=/tmp VLLM_RPC_BASE_PATH=/tmp \
  ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 VLLM_USE_V2_MODEL_RUNNER=1 VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
  VLLM_WORKER_MULTIPROC_METHOD=spawn OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -u tools/serve_vq2a8.py \
    --model "$MODEL" \
    --gs32-library "$BUILD/libvq2a8_gs32_stage.so" --gs32-binary "$BUILD/mixed_w2w4_moe_ffn.o" \
    --ep-size 8 --gs32-execution full_graph --gs32-fusion none \
    --max-model-len 262144 --max-num-batched-tokens 8192 \
    --kv-cache-mib 16384 --reserve-gib 16 --tokenizer-mode deepseek_v4 --reasoning-parser deepseek_v4 \
    --host 127.0.0.1 --port 8003 \
    > "$SERVE_DIR/serve.log" 2>&1 &
  B2PID=$!
  wait_ready 8003 "$SERVE_DIR/serve.log" "B2" || exit 1
  smoke 8003 vq2a8 "b2" || exit 1
  if [[ $SMOKE_ONLY -eq 0 ]]; then
    echo "B2 评测开始(续跑,目标每数据集5遍) $(date '+%F %T')" | tee -a "$OUT/summary.txt"
    for ds in gpqa aime24 aime25; do
      need=$(need_rounds "$OUT/b2" "$ds")
      echo "B2 $ds: $need 轮待跑" | tee -a "$OUT/summary.txt"
      if [[ "$need" -gt 0 ]]; then
        python tools/tests/run_vq2a8_accuracy_suite.py \
          --base-url http://127.0.0.1:8003 --model vq2a8 \
          --datasets "$ds" --offline-data-dir "$EVAL_DATA" \
          --thinking --reasoning-effort high --temperature 1 --top-p 1 --seed 0 --repeats "$need" \
          --concurrency 64 --max-tokens 131072 --timeout 7200 \
          --label packed-native-none-mrv2-tp1-dp1-ep8-b64-c8-thinking-high-t1-r${need}-o131072 \
          --result-dir "$OUT/b2" >> "$OUT/b2/eval.log" 2>&1
        echo "B2 $ds 本轮完成(rc=$?) $(date '+%F %T')" | tee -a "$OUT/summary.txt"
      fi
    done
    echo "B2 评测完成 $(date '+%F %T')" | tee -a "$OUT/summary.txt"
  fi
  cleanup_all; sleep 5
fi

echo "===== 全部完成 $(date '+%F %T') =====" | tee -a "$OUT/summary.txt"
echo "产物: $OUT (b1/ baseline, b2/ W2A8; 各含 serve.log + eval.log + 精度结果)"
