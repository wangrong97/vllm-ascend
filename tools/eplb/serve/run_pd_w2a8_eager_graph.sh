#!/usr/bin/env bash
# PD 分离: P(0-3)=W2A8 eager · DP4×EP4 + D(4-7)=W2A8 graph(B64) · DP4×EP4
# 新分支(test/vq2a8-perf-20261024) full_ffn 语义:
#   P: --enforce-eager(同一 full FFN 无 decode 图) — prefill 专用
#   D: full_graph 默认(decode 图桶 [1..64] 自动 + eager 分块 prefill) — 并发打满 B64
# 权重两侧同 W2A8; MooncakeHybrid PD + 双超时 + block32; proxy 8000 可选
# 用法: bash run_pd_w2a8_eager_graph.sh [--with-proxy]
#   EPLB_MODE=off|static|dynamic|...  D 侧混精 eplb 注入(默认 off)
set -uo pipefail

# /dev/shm 守门
shm_size_guard() {
  local avail; avail=$(df -m /dev/shm 2>/dev/null | awk 'NR==2{print $4}')
  if [[ -n "${avail:-}" && "$avail" -lt 1024 ]]; then
    mount -o remount,size=16G /dev/shm 2>/dev/null || echo "[shm-guard] remount 失败, 继续" >&2
  fi
}
shm_size_guard

REPO=/mnt/share/w00608002/work/eplb/vllm-ascend
MODEL_P=${MODEL_P:-/mnt/share/w00608002/weight/DeepSeek-V4-Flash-0731-vq2a8-mixed-gs32-repack}
MODEL_D=${MODEL_D:-$MODEL_P}
BUILD="$REPO/build/vq2a8-gs32-runtime-ffn"     # ABI v4 产物(见下方检查)
EPLB_MODE=${EPLB_MODE:-off}
EPLB_OPTIONS='{"K":8,"R_vq2":16,"R_mx4":8,"n_stage":2,"w_lo":0.2,"w_hi":5.0,"ewma_lambda":0.05}'

# ---- 前置检查: ABI v4 full_ffn 产物(不支持复用 v3) ----
if [[ ! -f "$BUILD/libvq2a8_gs32_stage.so" || ! -f "$BUILD/mixed_w2w4_moe_ffn.o" ]]; then
  echo "缺 ABI v4 产物: $BUILD" >&2
  echo "构建命令(需已加载 CANN 环境):" >&2
  echo "  export BUILD=\"$BUILD\"; cd $REPO/csrc/mixed_w2w4_moe_ffn && bash run_mixed_w2w4_moe_ffn_esl.sh" >&2
  echo "  python3 $REPO/tools/build_vq2a8_gs32_stage.py --kernel-binary $REPO/csrc/mixed_w2w4_moe_ffn/mixed_w2w4_moe_ffn.o --cann \$ASCEND_TOOLKIT_HOME --build-dir $BUILD --jobs 4" >&2
  exit 1
fi
for M in "$MODEL_P" "$MODEL_D"; do
  [[ -f "$M/model.safetensors.index.json" ]] || { echo "缺权重: $M"; exit 1; }
done

export OMP_PROC_BIND=false OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024
export HCCL_EXEC_TIMEOUT=3600 ACL_DEVICE_SYNC_TIMEOUT=3600
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# D 侧 eplb 注入(混精 mp; off 不注入)
D_EPLB_CFG='"eplb":"'$EPLB_MODE'"'
[[ "$EPLB_MODE" != "off" ]] && D_EPLB_CFG="$D_EPLB_CFG,\"eplb_options\":$EPLB_OPTIONS"

COMMON=(
  --dtype bfloat16 --load-format safetensors
  --tensor-parallel-size 1 --data-parallel-size 4 --data-parallel-size-local 4 --data-parallel-backend mp
  --pipeline-parallel-size 1 --distributed-executor-backend mp
  --served-model-name dsv --gpu-memory-utilization 0.9 --block-size 32
  --max-model-len 135000 --max-num-batched-tokens 8192 --max-num-seqs 64
  --kv-cache-memory-bytes 17179869184 --seed 0 --disable-log-stats
  --no-async-scheduling --no-enable-prefix-caching --enable-chunked-prefill
  --api-server-count 1 --tokenizer-mode deepseek_v4 --reasoning-parser deepseek_v4
  --quantization vq2a8 --enable-expert-parallel --expert-placement-strategy linear
  --all2all-backend flashinfer_all2allv
)
VQ2A8_COMMON="\"gs32_backend\": \"native\", \"gs32_fusion\": \"full_ffn\", \"gs32_library\": \"$BUILD/libvq2a8_gs32_stage.so\", \"gs32_binary\": \"$BUILD/mixed_w2w4_moe_ffn.o\""

# ---- P: eager prefill(卡0-3, :9001, producer) ----
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 HCCL_NPU_SOCKET_PORT_RANGE=20000-22499 \
  python -u -m vllm.entrypoints.cli.main serve "$MODEL_P" "${COMMON[@]}" \
  --enforce-eager \
  --data-parallel-rpc-port 29550 --host 127.0.0.1 --port 9001 \
  --additional-config '{"enable_flashcomm1": false, "mix_placement": false, "multistream_dsv4_dsa_overlap": false, "vq2a8": {"gs32_execution": "fused", '"$VQ2A8_COMMON"'}, "ascend_compilation_config": {"enable_npugraph_ex": false}}' \
  --kv-transfer-config '{"kv_connector": "MooncakeHybridConnector", "kv_role": "kv_producer", "kv_port": "20001", "engine_id": "0", "kv_connector_extra_config": {"prefill": {"dp_size": 4, "tp_size": 1}, "decode": {"dp_size": 4, "tp_size": 1}}}' &
P_PID=$!
echo "[PD] P(eager·W2A8·dp4ep4) pid=$P_PID devices=0-3 :9001"

# ---- D: graph B64 decode(卡4-7, :9002, consumer) ----
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 HCCL_NPU_SOCKET_PORT_RANGE=22500-25000 \
  python -u -m vllm.entrypoints.cli.main serve "$MODEL_D" "${COMMON[@]}" \
  --compilation-config '{"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --data-parallel-rpc-port 29551 --host 127.0.0.1 --port 9002 \
  --additional-config '{"enable_flashcomm1": false, "mix_placement": false, "multistream_dsv4_dsa_overlap": false, "vq2a8": {"gs32_execution": "full_graph", '"$VQ2A8_COMMON"', '"$D_EPLB_CFG"'}, "ascend_compilation_config": {"enable_npugraph_ex": false}}' \
  --kv-transfer-config '{"kv_connector": "MooncakeHybridConnector", "kv_role": "kv_consumer", "kv_port": "20002", "engine_id": "1", "kv_connector_extra_config": {"prefill": {"dp_size": 4, "tp_size": 1}, "decode": {"dp_size": 4, "tp_size": 1}}}' &
D_PID=$!
echo "[PD] D(graph·B64·W2A8·dp4ep4${EPLB_MODE:+ eplb=$EPLB_MODE}) pid=$D_PID devices=4-7 :9002"

trap 'echo "[PD] stopping"; kill $P_PID $D_PID 2>/dev/null; sleep 10
  for p in $(ps -eo pid,cmd | awk "/vllm.entrypoints/ && !/awk/ {print \$1}"); do kill -9 "$p" 2>/dev/null; done; sleep 5
  for p in $(npu-smi info | sed -n "/Process id/,\$p" | awk -F"|" "NF>4 {gsub(/ /,\"\",\$6); if (\$6+0>1000) print \$6}"); do kill -9 "$p" 2>/dev/null; done' INT TERM EXIT

if [[ "${1:-}" == "--with-proxy" ]]; then
  # NOTE: 不用子shell+wait — proxy 的子shell fork nohup后立即退出,
  # 会被下方 wait -n 捕获为"one instance exited"导致误杀 P/D。
  # 改为直接后台 nohup(proxy内部有uvicorn retry, 无需等PD就绪):
  nohup python3 "$REPO/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py" \
    --host 127.0.0.1 --port 8000 --prefiller-ports 9001 --decoder-ports 9002 --workers 1 \
    > "$REPO/proxy_8000.log" 2>&1 &
  PROXY_WRAPPER_PID=$!
  echo "[PD] proxy :8000 后台启动(直连,无子shell) wrapper_pid=$PROXY_WRAPPER_PID"
fi

wait -n
echo "[PD] one instance exited; shutting down peer"
kill $P_PID $D_PID 2>/dev/null
wait
