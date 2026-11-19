#!/usr/bin/env bash
# PD 分离: P(0-3)=W2A8 eager · DP4×EP4 + D(4-7)=W2A8 graph(B64) · DP4×EP4
# 新分支(test/vq2a8-perf-20261024) full_ffn 语义:
#   P: --enforce-eager(同一 full FFN 无 decode 图) — prefill 专用
#   D: full_graph 默认(decode 图桶 [1..64] 自动 + eager 分块 prefill) — 并发打满 B64
# 权重两侧同 W2A8; MooncakeHybrid PD + 双超时 + block32; proxy 8000 可选
# 用法: bash run_pd_w2a8_eager_graph.sh [--with-proxy]
#   EPLB_MODE=off|static|dynamic|...  D 侧混精 eplb 注入(默认 off)
#   QUANT=auto|fp8|vq2a8  量化方式(auto=按 MODEL_P 的 config.json 识别;
#                          fp8 时不注入 --quantization/vq2a8 配置, 供 FP8 基线)
#   KV_BYTES/MAX_LEN/MEM_UTIL  资源旋钮(默认与历史行为一致; FP8 权重大可下调防 OOM)
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

# ---- 量化方式: auto=按 MODEL_P config.json 识别(fp8|vq2a8) ----
QUANT=${QUANT:-auto}
if [[ "$QUANT" == "auto" ]]; then
  QUANT=$(python3 -c "import json;print(json.load(open('$MODEL_P/config.json')).get('quantization_config',{}).get('quant_method',''))")
  echo "[PD] QUANT 自动识别: $QUANT (来自 $MODEL_P/config.json)"
fi
[[ "$QUANT" =~ ^(fp8|vq2a8)$ ]] || { echo "无法识别的量化方式: '$QUANT' (支持 fp8|vq2a8)" >&2; exit 1; }

# ---- 资源旋钮(默认=历史值) ----
KV_BYTES=${KV_BYTES:-17179869184}
MAX_LEN=${MAX_LEN:-135000}
MEM_UTIL=${MEM_UTIL:-0.9}

# ---- 前置检查: ABI v4 full_ffn 产物(不支持复用 v3; 仅 vq2a8 需要) ----
if [[ "$QUANT" == "vq2a8" ]]; then
  if [[ ! -f "$BUILD/libvq2a8_gs32_stage.so" || ! -f "$BUILD/mixed_w2w4_moe_ffn.o" ]]; then
    echo "缺 ABI v4 产物: $BUILD" >&2
    echo "构建命令(需已加载 CANN 环境):" >&2
    echo "  export BUILD=\"$BUILD\"; cd $REPO/csrc/mixed_w2w4_moe_ffn && bash run_mixed_w2w4_moe_ffn_esl.sh" >&2
    echo "  python3 $REPO/tools/build_vq2a8_gs32_stage.py --kernel-binary $REPO/csrc/mixed_w2w4_moe_ffn/mixed_w2w4_moe_ffn.o --cann \$ASCEND_TOOLKIT_HOME --build-dir $BUILD --jobs 4" >&2
    exit 1
  fi
fi
for M in "$MODEL_P" "$MODEL_D"; do
  [[ -f "$M/model.safetensors.index.json" ]] || { echo "缺权重: $M"; exit 1; }
done

export OMP_PROC_BIND=false OMP_NUM_THREADS=10
export VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S:-1800}  # 大权重 NFS 加载 600s 默认不够(E4 教训)
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024
export HCCL_EXEC_TIMEOUT=3600 ACL_DEVICE_SYNC_TIMEOUT=3600
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# D 侧 eplb 注入(混精 mp; off 不注入)
D_EPLB_CFG='"eplb":"'$EPLB_MODE'"'
[[ "$EPLB_MODE" != "off" ]] && D_EPLB_CFG="$D_EPLB_CFG,\"eplb_options\":$EPLB_OPTIONS"

# 投机开关: SPEC_TOKENS=dspark 启用 DSpark(vq2a8 唯一支持; D 侧 5 draft tokens)
#   DSPARK_MODEL 覆盖 draft checkpoint 路径
# SPEC_TOKENS=N 走 MTP(N 个 draft token; 仅非 vq2a8 量化模型可用)
SPEC_TOKENS=${SPEC_TOKENS:-}
DSPARK_MODEL=${DSPARK_MODEL:-/mnt/weight/DeepSeek-V4-Flash-DSpark}
SPEC_P=(); SPEC_D=()
# D 侧 decode 图配置: dspark 时 query_width=6, capture sizes 须为 6 的倍数
# (GS32 校验: capture_sizes[-1] == max_num_seqs * query_width)
D_COMPILATION='{"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}'
if [[ "$SPEC_TOKENS" == "dspark" ]]; then
  SPEC_D=(--speculative-config '{"method": "dspark", "num_speculative_tokens": 5, "enforce_eager": true, "model": "'"$DSPARK_MODEL"'"}')
  D_COMPILATION='{"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [6, 12, 24, 48, 96, 144, 192, 240, 288, 336, 384], "max_cudagraph_capture_size": 384}'
elif [[ -n "$SPEC_TOKENS" ]]; then
  SPEC_P=(--speculative-config '{"num_speculative_tokens": 1, "method": "deepseek_mtp"}')
  SPEC_D=(--speculative-config '{"num_speculative_tokens": '"$SPEC_TOKENS"', "method": "deepseek_mtp"}')
fi

COMMON=(
  --dtype bfloat16 --load-format safetensors
  --tensor-parallel-size 1 --data-parallel-size 4 --data-parallel-size-local 4 --data-parallel-backend mp
  --pipeline-parallel-size 1 --distributed-executor-backend mp
  --served-model-name dsv --gpu-memory-utilization "$MEM_UTIL" --block-size 32
  --max-model-len "$MAX_LEN" --max-num-batched-tokens 8192 --max-num-seqs 64
  --kv-cache-memory-bytes "$KV_BYTES" --seed 0
  --no-async-scheduling --no-enable-prefix-caching --enable-chunked-prefill
  --api-server-count 1 --tokenizer-mode deepseek_v4 --reasoning-parser deepseek_v4
  --enable-expert-parallel --expert-placement-strategy linear
  --all2all-backend flashinfer_all2allv
)
# LOG_STATS=0 时关闭周期统计日志(默认开, spec_decode 接受率等指标依赖它)
[[ "${LOG_STATS:-1}" == "0" ]] && COMMON+=(--disable-log-stats)
QUANT_ARGS=()                                   # fp8: 不传 --quantization(config.json 自动推导)
[[ "$QUANT" == "vq2a8" ]] && QUANT_ARGS=(--quantization vq2a8)
VQ2A8_COMMON="\"gs32_backend\": \"native\", \"gs32_fusion\": \"full_ffn\", \"gs32_library\": \"$BUILD/libvq2a8_gs32_stage.so\", \"gs32_binary\": \"$BUILD/mixed_w2w4_moe_ffn.o\""
# vq2a8 注入块(fp8 时置空, additional-config 其余部分不变)
VQ2A8_BLOCK_P=', "vq2a8": {"gs32_execution": "fused", '"$VQ2A8_COMMON"'}'
VQ2A8_BLOCK_D=', "vq2a8": {"gs32_execution": "full_graph", '"$VQ2A8_COMMON"', '"$D_EPLB_CFG"'}'
if [[ "$QUANT" != "vq2a8" ]]; then VQ2A8_BLOCK_P=""; VQ2A8_BLOCK_D=""; fi

# ---- P: eager prefill(卡0-3, :9001, producer) ----
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 HCCL_NPU_SOCKET_PORT_RANGE=20000-22499 \
  python -u -m vllm.entrypoints.cli.main serve "$MODEL_P" "${COMMON[@]}" "${QUANT_ARGS[@]}" \
  --enforce-eager \
  --data-parallel-rpc-port 29550 --host 127.0.0.1 --port 9001 "${SPEC_P[@]}" \
  --additional-config '{"enable_flashcomm1": false, "mix_placement": false, "multistream_dsv4_dsa_overlap": false'"$VQ2A8_BLOCK_P"', "ascend_compilation_config": {"enable_npugraph_ex": false}}' \
  --kv-transfer-config '{"kv_connector": "MooncakeHybridConnector", "kv_role": "kv_producer", "kv_port": "20001", "engine_id": "0", "kv_connector_extra_config": {"prefill": {"dp_size": 4, "tp_size": 1}, "decode": {"dp_size": 4, "tp_size": 1}}}' &
P_PID=$!
echo "[PD] P(eager·$QUANT·dp4ep4) pid=$P_PID devices=0-3 :9001"

# ---- D: graph B64 decode(卡4-7, :9002, consumer) ----
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 HCCL_NPU_SOCKET_PORT_RANGE=22500-25000 \
  python -u -m vllm.entrypoints.cli.main serve "$MODEL_D" "${COMMON[@]}" "${QUANT_ARGS[@]}" \
  --compilation-config "$D_COMPILATION" \
  --data-parallel-rpc-port 29551 --host 127.0.0.1 --port 9002 "${SPEC_D[@]}" \
  --additional-config '{"enable_flashcomm1": false, "mix_placement": false, "multistream_dsv4_dsa_overlap": false'"$VQ2A8_BLOCK_D"', "ascend_compilation_config": {"enable_npugraph_ex": false}}' \
  --kv-transfer-config '{"kv_connector": "MooncakeHybridConnector", "kv_role": "kv_consumer", "kv_port": "20002", "engine_id": "1", "kv_connector_extra_config": {"prefill": {"dp_size": 4, "tp_size": 1}, "decode": {"dp_size": 4, "tp_size": 1}}}' &
D_PID=$!
echo "[PD] D(graph·B64·$QUANT·dp4ep4${EPLB_MODE:+ eplb=$EPLB_MODE}) pid=$D_PID devices=4-7 :9002"

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
