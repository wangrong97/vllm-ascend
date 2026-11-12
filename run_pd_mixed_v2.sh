#!/usr/bin/env bash
# /dev/shm 守门
shm_size_guard() {
    local avail
    avail=$(df -m /dev/shm 2>/dev/null | awk 'NR==2{print $4}')
    if [[ -n "$avail" && "$avail" -lt 1024 ]]; then
        mount -o remount,size=16G /dev/shm 2>/dev/null || echo "[shm-guard] remount /dev/shm 失败(权限?),继续" >&2
    fi
}
shm_size_guard

# E4'': PD 混搭 —— P(卡0-3)=P''(moeW4A4-attnW8A8,合法 vq2a8) + D(卡4-7)=W2A8
# MooncakeHybridConnector + 双超时 + block32 + proxy8000
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024
export HCCL_EXEC_TIMEOUT=3600
export ACL_DEVICE_SYNC_TIMEOUT=3600

REPO=/mnt/share/w00608002/work/eplb/vllm-ascend
MODEL_P=${MODEL_P:-/mnt/share/w00608002/weight/DeepSeek-V4-Flash-0731-moeW4A4-attnW2A8}
MODEL_D=/mnt/share/w00608002/weight/DeepSeek-V4-Flash-0731-vq2a8-mixed-gs32-repack

COMMON=(
  --max-model-len 135000 --max-num-batched-tokens 4096 --max-num-seqs 8
  --served-model-name dsv --gpu-memory-utilization 0.9 --block-size 32
  --no-enable-prefix-caching --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4
  --enable-auto-tool-choice --reasoning-parser deepseek_v4 --trust-remote-code
  --data-parallel-size 4 --tensor-parallel-size 1 --enable-expert-parallel
  --api-server-count 1 --safetensors-load-strategy prefetch --no-async-scheduling --enforce-eager
)

# ---- Prefill: P''(attn 逐比特同 D 的 W8A8, MoE=W4A4), eager ----
PC_P=(); [[ -n "${PROF_DIR_P:-}" ]] && PC_P=(--profiler-config "{\"profiler\": \"torch\", \"torch_profiler_dir\": \"$PROF_DIR_P\", \"torch_profiler_with_stack\": false}")
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 HCCL_NPU_SOCKET_PORT_RANGE=20000-22499 \
  vllm serve "$MODEL_P" "${COMMON[@]}" \
  --data-parallel-rpc-port 29550 --port 9001 "${PC_P[@]}" \
  --additional-config '{"enable_cpu_binding": "True", "vq2a8": {"gs32_library": "'$REPO'/build/vq2a8-gs32-runtime/libvq2a8_gs32_stage.so", "gs32_binary": "'$REPO'/build/vq2a8-gs32-runtime/mixed_w2w4_moe_ffn.o", "gs32_execution": "serial", "gs32_fusion": "none"}}' \
  --kv-transfer-config '{"kv_connector": "MooncakeHybridConnector", "kv_role": "kv_producer", "kv_port": "20001", "engine_id": "0", "kv_connector_extra_config": {"prefill": {"dp_size": 4, "tp_size": 1}, "decode": {"dp_size": 4, "tp_size": 1}}}' &
P_PID=$!
echo "[E4''] P(moeW4A4-attnW8A8) pid=$P_PID devices=0-3 port=9001"

# ---- Decode: W2A8(GS32 serial, eplb 由 EPLB_MODE 控制), eager ----
EPLB_MODE=${EPLB_MODE:-off}
EPLB_OPTIONS=${EPLB_OPTIONS:-'{"K": 8, "R_vq2": 16, "R_mx4": 8, "n_stage": 2, "w_lo": 0.2, "w_hi": 5.0, "ewma_lambda": 0.05}'}
D_EPLB_CFG='"eplb":"'$EPLB_MODE'"'
if [[ "$EPLB_MODE" != "off" ]]; then D_EPLB_CFG="$D_EPLB_CFG,\"eplb_options\":$EPLB_OPTIONS"; fi
PC_D=(); [[ -n "${PROF_DIR_D:-}" ]] && PC_D=(--profiler-config "{\"profiler\": \"torch\", \"torch_profiler_dir\": \"$PROF_DIR_D\", \"torch_profiler_with_stack\": false}")
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 HCCL_NPU_SOCKET_PORT_RANGE=22500-25000 \
  vllm serve "$MODEL_D" "${COMMON[@]}" \
  --data-parallel-rpc-port 29551 --port 9002 "${PC_D[@]}" \
  --additional-config '{"enable_cpu_binding": "True", "ascend_compilation_config": {"enable_npugraph_ex": false}, "multistream_dsv4_dsa_overlap": false, "vq2a8": {"gs32_library": "'$REPO'/build/vq2a8-gs32-runtime/libvq2a8_gs32_stage.so", "gs32_binary": "'$REPO'/build/vq2a8-gs32-runtime/mixed_w2w4_moe_ffn.o", "gs32_execution": "serial", "gs32_fusion": "none", '$D_EPLB_CFG'}}' \
  --kv-transfer-config '{"kv_connector": "MooncakeHybridConnector", "kv_role": "kv_consumer", "kv_port": "20002", "engine_id": "1", "kv_connector_extra_config": {"prefill": {"dp_size": 4, "tp_size": 1}, "decode": {"dp_size": 4, "tp_size": 1}}}' &
D_PID=$!
echo "[E4''] D(w2a8) pid=$D_PID devices=4-7 port=9002"

trap 'echo "[E4''] stopping"; kill $P_PID $D_PID 2>/dev/null; sleep 10; ps -ef | grep -E "[v]llm serve" | awk "{print \$2}" | while read p; do case "$(readlink /proc/$p/exe 2>/dev/null)" in *python*) kill \$p 2>/dev/null;; esac; done; sleep 10; ps -ef | grep -E "[V]LLM::" | awk "{print \$2}" | while read p; do case "$(readlink /proc/$p/exe 2>/dev/null)" in *python*) kill -9 \$p 2>/dev/null;; esac; done' INT TERM EXIT
wait -n
echo "[E4''] one instance exited; shutting down peer"
kill $P_PID $D_PID 2>/dev/null
wait
