#!/usr/bin/env bash
# =============================================================================
# PD 分离(Disaggregated Prefill/Decode)· 8 卡 · DP4 × EP4 · VQ2A8 混精 EPLB
#
# 原单机 DP8 脚本已备份为 run_single_node.sh.bak。
#
# 拓扑:prefill 实例(卡 0-3)+ decode 实例(卡 4-7),每侧
#       data-parallel-size 4 + tensor-parallel-size 1 + enable-expert-parallel
#       + enable-dp-attention(attention 按 DP 复制,MoE 专家并行横跨
#       4 个 DP rank → EP=4),KV 经 SharedStorageConnector 从 P 传 D。
#
# -----------------------------------------------------------------------------
# ⚠️ 兼容性提示(先读再跑):当前 VQ2A8 校验(validate_vq2a8_config,
#    vllm_ascend/quantization/configs/vq2a8_config.py)会拒绝:
#      ① data_parallel_size > 1            (L85-91)
#      ② kv_transfer_config 非 None(PD 必需)(L152-155)
#    本脚本按目标形态写全参数;真跑通前需在上述两处放开校验并完成
#    DP/PD 运行时验证(运行时支持≠校验放开)。原脚本中的
#    --async-scheduling / --speculative-config 也在 VQ2A8 拒绝清单中,
#    已按校验要求移除(注释保留)。
# -----------------------------------------------------------------------------
set -euo pipefail

# ----------------------------- 通用环境(继承自原脚本) -----------------------
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=2
export ASCEND_HOST_LOG_FILE_NUM=1000
export ASCEND_PROCESS_LOG_PATH=/mnt/share/w00608002/plog
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024
export GLOO_SOCKET_IFNAME=lo
# PD 双实例各自组网,显式限定端口段避免 EI0020 端口残留冲突。
export HCCL_NPU_SOCKET_PORT_RANGE=20000-25000

# source /usr/local/Ascend/cann-9.1.0/set_env.sh   # 按本机 CANN 路径打开

# ----------------------------- 模型与构建产物 -------------------------------
MODEL=/mnt/share/w00608002/weights/DeepSeek-V4-Flash-0731-vq2a8-mixed-gs32-repack
# GS32 原生算子产物(按 docs/vq2a8/vq2a8_gs32_runtime.md 构建):
#   bash csrc/mixed_w2w4_moe_ffn/run_mixed_w2w4_moe_ffn_esl.sh
#   python tools/build_vq2a8_gs32_stage.py --cann "$ASCEND_TOOLKIT_HOME" \
#     --kernel-binary csrc/mixed_w2w4_moe_ffn/mixed_w2w4_moe_ffn.o \
#     --build-dir build/vq2a8-gs32-runtime
GS32_LIB=$PWD/build/vq2a8-gs32-runtime/libvq2a8_gs32_stage.so
GS32_BIN=$PWD/build/vq2a8-gs32-runtime/mixed_w2w4_moe_ffn.o

# ----------------------------- 混精 EPLB 开关 -------------------------------
# vllm_ascend/eplb/mp 实现;off 时运行路径与基线完全一致。
#   dynamic          : 类型感知(分 VQ2/MXFP4 代价斜率+字节预算+类型分流冗余池)
#   dynamic_uniform  : 不考虑量化位宽(统一代价=token 条数、单池冗余、无字节核算)
MP_EPLB_MODE="dynamic"            # off | static | dynamic | static_uniform | dynamic_uniform
MP_EPLB_OPTIONS='"K": 8, "R_vq2": 16, "R_mx4": 8, "n_stage": 2, "w_lo": 0.2, "w_hi": 5.0, "ewma_lambda": 0.05'
# 说明:R_vq2/R_mx4 为全局冗余槽预算,per-rank 空槽 = R // ep_size
#      (ep=4 → 每卡 vq2 空槽 4 个 ≈ 27MB、mxfp4 2 个 ≈ 26MB,计入显存)。

# GS32 执行模式。full_graph 需要:--max-num-seqs 1、capture_sizes=[1]、
# FULL_DECODE_ONLY、同步调度(默认)。若 PD/DP 调试期想先绕开图约束,
# 改 GS32_EXECUTION="fused" 并启用下方 --enforce-eager 分支。
GS32_EXECUTION="full_graph"       # serial | batched | fused | full_graph

# PD 共享 KV 落盘目录(SharedStorageConnector;如换 AscendStore/Mooncake
# 连接器,改下方 kv_connector 与 kv_role_params)。
KV_SHARED_DIR=/mnt/share/w00608002/kv_shared
mkdir -p "$KV_SHARED_DIR"

# ----------------------------- additional_config 组装 -----------------------
read -r -d '' ADDITIONAL_CONFIG <<EOF || true
{
  "enable_cpu_binding": "True",
  "vq2a8": {
    "gs32_library": "${GS32_LIB}",
    "gs32_binary": "${GS32_BIN}",
    "gs32_execution": "${GS32_EXECUTION}",
    "gs32_fusion": "none",
    "eplb": "${MP_EPLB_MODE}",
    "eplb_options": { ${MP_EPLB_OPTIONS} }
  }
}
EOF

# 公共参数(两实例同构;--api-server-count 1 由外部网关聚合双实例)
COMMON_ARGS=(
  "$MODEL"
  --max-model-len 135000
  --max-num-batched-tokens 4096
  --served-model-name dsv
  --enable-return-routed-experts
  --gpu-memory-utilization 0.9
  --block-size 128
  --no-enable-prefix-caching
  --tokenizer-mode deepseek_v4
  --tool-call-parser deepseek_v4
  --enable-auto-tool-choice
  --reasoning-parser deepseek_v4
  --data-parallel-size 4
  --tensor-parallel-size 1
  --enable-expert-parallel
  --enable-dp-attention
  --api-server-count 1
  --safetensors-load-strategy prefetch
  --additional-config "$ADDITIONAL_CONFIG"
)
# full_graph 配套约束(见上);若 GS32_EXECUTION != full_graph,切换到:
#   --enforce-eager 并移除 --compilation-config
GRAPH_ARGS=(
  --max-num-seqs 1
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1], "max_cudagraph_capture_size": 1}'
)
# 原 DP8 单机脚本的以下选项与 VQ2A8 校验冲突,PD 形态不携带(留档):
#   --async-scheduling            (VQ2A8 要求同步调度)
#   --speculative-config mtp      (VQ2A8 禁 speculative)

# ----------------------------- Prefill 实例(卡 0-3,producer) ---------------
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 vllm serve "${COMMON_ARGS[@]}" "${GRAPH_ARGS[@]}" \
  --port 9001 \
  --kv-transfer-config "{
    \"kv_connector\": \"SharedStorageConnector\",
    \"kv_role\": \"producer\",
    \"kv_role_params\": { \"secondary_folder\": \"${KV_SHARED_DIR}\" }
  }" &
PREFILL_PID=$!
echo "[PD] prefill(producer) pid=${PREFILL_PID} devices=0,1,2,3 port=9001 eplb=${MP_EPLB_MODE}"

# ----------------------------- Decode 实例(卡 4-7,consumer) ---------------
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 vllm serve "${COMMON_ARGS[@]}" "${GRAPH_ARGS[@]}" \
  --port 9002 \
  --kv-transfer-config "{
    \"kv_connector\": \"SharedStorageConnector\",
    \"kv_role\": \"consumer\",
    \"kv_role_params\": { \"secondary_folder\": \"${KV_SHARED_DIR}\" }
  }" &
DECODE_PID=$!
echo "[PD] decode(consumer) pid=${DECODE_PID} devices=4,5,6,7 port=9002 eplb=${MP_EPLB_MODE}"

# ----------------------------- 生命周期管理 ----------------------------------
trap 'echo "[PD] stopping..."; kill $PREFILL_PID $DECODE_PID 2>/dev/null || true; wait' INT TERM
wait -n
EXIT_CODE=$?
echo "[PD] one instance exited (code=${EXIT_CODE}); shutting down the peer."
kill $PREFILL_PID $DECODE_PID 2>/dev/null || true
wait || true
exit $EXIT_CODE
