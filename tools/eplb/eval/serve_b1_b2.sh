#!/usr/bin/env bash
# 单独拉起 B1/B2 评测用的 vllm serve（参数与 run_baseline_vs_w2a8.sh 编排内逐字一致）
#
#   B1: baseline FP8  · DP8×EP · 图模式 FULL_DECODE_ONLY(capture 1,2,4) · port 8004
#   B2: W2A8 packed-native · tp-ep EP8 · gs32 full_graph · port 8003
#
# 用法:
#   bash tools/eplb/eval/serve_b1_b2.sh b1              # 前台启动 B1 (日志 tee 到 stdout+文件)
#   bash tools/eplb/eval/serve_b1_b2.sh b2              # 前台启动 B2
#   bash serve_b1_b2.sh b1 --port 9004 --log-dir /tmp/x # 覆盖端口/日志目录
#   bash serve_b1_b2.sh b2 --devices 4,5,6,7            # 指定卡(B1 自动 DP=卡数, B2 自动 EP=卡数)
# 后台运行: nohup bash serve_b1_b2.sh b1 >/dev/null 2>&1 &
set -euo pipefail

EPLB_ROOT=$(cd "$(dirname "$BASH_SOURCE")/../../.." && pwd)   # vllm-ascend 仓库根
BASELINE_MODEL=/mnt/share/w00608002/weight/DeepSeek-V4-Flash-0731
MODEL=${MODEL:-/mnt/share/w00608002/weight/DeepSeek-V4-Flash-0731-vq2a8-mixed-gs32-repack}
BUILD="$EPLB_ROOT/build/vq2a8-gs32-runtime"
DEFAULT_LOG_DIR="/mnt/share/w00608002/work/eplb/output/serve_standalone"

# ---- 参数解析 ----
SIDE="${1:-}"; [[ "$SIDE" == "b1" || "$SIDE" == "b2" ]] || { echo "用法: $0 {b1|b2} [--port N] [--log-dir DIR]"; exit 1; }
shift || true
PORT=; LOG_DIR=; DEVICES=0,1,2,3,4,5,6,7
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT=$2; shift 2;;
    --log-dir) LOG_DIR=$2; shift 2;;
    --devices) DEVICES=$2; shift 2;;
    *) echo "未知参数: $1"; exit 1;;
  esac
done
NWORLD=$(awk -F, '{print NF}' <<< "$DEVICES")   # 卡数 → B1 dp / B2 ep

# ---- /dev/shm 守门 (DP 进程间通信需共享内存, 容器默认 64M 不够) ----
shm_size_guard() {
  local avail
  avail=$(df -m /dev/shm 2>/dev/null | awk 'NR==2{print $4}')
  if [[ -n "${avail:-}" && "$avail" -lt 1024 ]]; then
    mount -o remount,size=16G /dev/shm 2>/dev/null || echo "[shm-guard] remount /dev/shm 失败(权限?), 继续" >&2
  fi
}
shm_size_guard

# ---- 前置检查 ----
if [[ "$SIDE" == "b1" ]]; then
  PORT=${PORT:-8004}
  [[ -d "$BASELINE_MODEL" ]] || { echo "缺 baseline 模型: $BASELINE_MODEL"; exit 1; }
else
  PORT=${PORT:-8003}
  [[ -d "$MODEL" ]] || { echo "缺 W2A8 模型: $MODEL"; exit 1; }
  [[ -f "$BUILD/libvq2a8_gs32_stage.so" && -f "$BUILD/mixed_w2w4_moe_ffn.o" ]] || { echo "缺 GS32 产物: $BUILD"; exit 1; }
fi
if curl -s --max-time 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
  echo "端口 $PORT 已有服务在跑, 先停掉再启动"; exit 1
fi

LOG_DIR=${LOG_DIR:-$DEFAULT_LOG_DIR}
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/serve_${SIDE}_$(date +%m%d_%H%M).log"
echo "[$SIDE] 端口=$PORT 卡=$DEVICES (dp/ep=$NWORLD) 日志=$LOG (前台运行, Ctrl+C 停止)"

cd "$EPLB_ROOT"
if [[ "$SIDE" == "b1" ]]; then
  # ---- B1: baseline FP8 · DP×EP · FULL_DECODE_ONLY 图模式 ----
  exec >> >(tee -a "$LOG") 2>&1
  ASCEND_RT_VISIBLE_DEVICES=$DEVICES VLLM_USE_V2_MODEL_RUNNER=1 \
  VLLM_WORKER_MULTIPROC_METHOD=spawn OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OMP_PROC_BIND=false HCCL_BUFFSIZE=1024 PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
  python -u -m vllm.entrypoints.cli.main serve "$BASELINE_MODEL" \
    --served-model-name baseline --host 127.0.0.1 --port "$PORT" \
    --dtype auto --tensor-parallel-size 1 --data-parallel-size "$NWORLD" --enable-expert-parallel \
    --api-server-count 1 --max-model-len 262144 --max-num-batched-tokens 8192 --max-num-seqs 4 \
    --gpu-memory-utilization 0.9 --enable-chunked-prefill --no-enable-prefix-caching \
    --no-async-scheduling --block-size 32 --tokenizer-mode deepseek_v4 --reasoning-parser deepseek_v4 \
    --generation-config vllm --seed 0 \
    --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4],"max_cudagraph_capture_size":4,"cudagraph_num_of_warmups":1}' \
    --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":false},"enable_cpu_binding":true,"multistream_overlap_shared_expert":true}'
else
  # ---- B2: W2A8 packed-native · tp-ep EP · gs32 full_graph ----
  exec >> >(tee -a "$LOG") 2>&1
  TMPDIR=/tmp VLLM_RPC_BASE_PATH=/tmp \
  ASCEND_RT_VISIBLE_DEVICES=$DEVICES VLLM_USE_V2_MODEL_RUNNER=1 VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
  VLLM_WORKER_MULTIPROC_METHOD=spawn OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -u "$EPLB_ROOT/tools/serve_vq2a8.py" \
    --model "$MODEL" \
    --gs32-library "$BUILD/libvq2a8_gs32_stage.so" --gs32-binary "$BUILD/mixed_w2w4_moe_ffn.o" \
    --ep-size "$NWORLD" --gs32-execution full_graph --gs32-fusion none \
    --max-model-len 262144 --max-num-batched-tokens 8192 \
    --kv-cache-mib 16384 --reserve-gib 16 --tokenizer-mode deepseek_v4 --reasoning-parser deepseek_v4 \
    --host 127.0.0.1 --port "$PORT"
fi
