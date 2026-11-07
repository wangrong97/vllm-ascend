#!/bin/bash
# GLM-5.3-Flash 单机一键部署 + 对话验证 (8x Ascend950DT, 96GB x 8)
# 布局: dp1 x tp8 x ep8, 288 experts (每 EP rank 36 个)。
# 权重: W8A8-MXFP8 量化 checkpoint (--quantization ascend), 真实权重加载,
#       对话输出正常。如需 dummy 权重做 profiling, 设 LOAD_FORMAT=dummy。
# Spec decode: deepseek_mtp (GLM5.3-Flash 无 DSpark 路径:
# glm5_next target 不输出 DSpark draft 所需的 aux hidden states)。
#
# 用法: bash run_glm53flash.sh [--no-profile] [--stop]
#   --no-profile  不挂 torch profiler
#   --stop        停止已启动的服务并退出

set -e

BASE=/mnt/share/w00608002/work/vllm-ascend
LOG_DIR=${BASE}/logs
MODEL_PATH=${MODEL_PATH:-/mnt/share/w00936111/weights/GLM-5.3-Flash}
PORT=${PORT:-8000}
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/server_single_$(date +%m%d_%H%M%S).log"

# ---- 实验参数(可用环境变量覆盖) ----
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-132096}          # 教程 950DT 单机长
export MAX_NUM_SEQS=${MAX_NUM_SEQS:-64}
export MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}
export GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.9}
export SPEC_TOKENS=${SPEC_TOKENS:-3}                   # deepseek_mtp draft tokens
LOAD_FORMAT=${LOAD_FORMAT:-}                           # 置 dummy 则跳过真实权重
# 量化方式: 本地 checkpoint config.json 标注 quant_method=fp8, 由 vllm-ascend
# 注册的 AscendFp8Config 自动接管, 不能显式传 --quantization ascend (两者冲突)。
# 若换成 quant_method=ascend 的 ModelSlim checkpoint, 设 QUANT_METHOD=ascend。
QUANT_METHOD=${QUANT_METHOD:-}

# ---- 教程要求的环境变量 (GLM5.3-Flash 仅支持 model runner V1) ----
export VLLM_USE_V2_MODEL_RUNNER=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024

stop_server() {
    # 只匹配本脚本的模型路径, 避免误杀共享机器上其他用户的 vllm 任务
    pkill -9 -f "vllm serve.*${MODEL_PATH}" 2>/dev/null && echo "[stop] 已停止 vllm serve" || echo "[stop] 没有运行中的服务"
}
if [[ "${1:-}" == "--stop" ]]; then
    stop_server
    exit 0
fi

# /dev/shm 守门: vLLM 进程间通信需共享内存(容器默认 64M 不够),
# 不足 1G 时自动 remount 扩到 16G(无权限时警告继续)。
shm_size_guard() {
    local avail
    avail=$(df -m /dev/shm 2>/dev/null | awk 'NR==2{print $4}')
    if [[ -n "$avail" && "$avail" -lt 1024 ]]; then
        mount -o remount,size=16G /dev/shm 2>/dev/null || echo "[shm-guard] remount /dev/shm 失败(权限?),继续" >&2
    fi
}
shm_size_guard

# ---- 对话验证(函数, 供快速路径和启动后复用) ----
verify_chat() {
    # GLM-5.3-Flash 是思考模型, reasoning 通常消耗数百 token, max_tokens 要给足
    local resp
    resp=$(curl -s -m 300 "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "glm",
            "messages": [{"role": "user", "content": "你好, 请用一句话介绍你自己"}],
            "max_tokens": 2048
        }')
    echo "${resp}" | python3 -I -c "
import json, sys
data = json.load(sys.stdin)
if 'error' in data:
    print('[chat] 请求失败:', data['error'])
    sys.exit(1)
msg = data['choices'][0]['message']
# glm47 reasoning parser 输出字段为 reasoning, 兼容 reasoning_content
reasoning = (msg.get('reasoning') or msg.get('reasoning_content') or '').strip()
content = (msg.get('content') or '').strip()
finish = data['choices'][0].get('finish_reason')
print('[chat] finish_reason:', finish)
print('[chat] reasoning:', (reasoning[:200] + '...') if reasoning else '(无)')
print('[chat] content  :', content[:300] or '(空!)')
if not content:
    print('[chat] 回复内容为空(可能被思考 token 截断), 请加大 max_tokens 重试验证')
    sys.exit(1)
print('[chat] 对话验证通过 ✓')
"
}

# 端口占用检查: 已有健康服务时直接做对话验证(快速路径)
if curl -s -m 3 "http://127.0.0.1:${PORT}/health" -o /dev/null; then
    echo "[check] 端口 ${PORT} 已有健康服务, 直接进行对话验证(如需重启: bash $0 --stop)"
    verify_chat
    exit 0
fi

PROFILE_ARGS=()
if [[ "${1:-}" != "--no-profile" ]]; then
    PROF_DIR=${BASE}/prof_data/single_ep8_$(date +%m%d_%H%M%S)
    mkdir -p "${PROF_DIR}"
    PROFILE_ARGS=(--profiler-config "{\"profiler\": \"torch\", \"torch_profiler_dir\": \"${PROF_DIR}\"}")
fi

LOAD_ARGS=()
if [[ -n "${LOAD_FORMAT}" ]]; then
    LOAD_ARGS=(--load-format "${LOAD_FORMAT}")
else
    # 网络共享盘加载 306GB 权重, prefetch 先进 page cache(内存充足时显著加速)
    LOAD_ARGS=(--safetensors-load-strategy prefetch)
fi

QUANT_ARGS=()
if [[ -n "${QUANT_METHOD}" ]]; then
    QUANT_ARGS=(--quantization "${QUANT_METHOD}")
fi

echo "[start] 日志: ${LOG_FILE}"
setsid vllm serve "${MODEL_PATH}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --seed 1024 \
  --served-model-name glm \
  --kv-cache-dtype fp8 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --trust-remote-code \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --async-scheduling \
  --enable-prefix-caching \
  --reasoning-parser glm47 \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1,2,4,8,16,32,64,96,128]}' \
  --speculative-config "{\"num_speculative_tokens\": ${SPEC_TOKENS}, \"method\": \"deepseek_mtp\", \"enforce_eager\": true}" \
  --additional-config '{"enable_fused_mc2": 0}' \
  "${LOAD_ARGS[@]}" \
  "${QUANT_ARGS[@]}" \
  "${PROFILE_ARGS[@]}" \
  > "${LOG_FILE}" 2>&1 &

SERVER_PID=$!
# 失败/中断时杀掉整个进程组, 防止 EngineCore/Worker 残留占用显存
cleanup_server() {
    kill -9 -- -"${SERVER_PID}" 2>/dev/null || true
}
trap cleanup_server INT TERM
echo "[start] vllm serve PID=${SERVER_PID}, 等待服务就绪(权重约 306GB, 首次加载可能需要 20-40 分钟)..."

# 等待健康检查通过, 最长 60 分钟
deadline=$((SECONDS + 3600))
while (( SECONDS < deadline )); do
    # EngineCore/Worker 挂掉时 APIServer 可能不退出, 从日志判断致命错误
    if grep -qE "Engine core initialization failed|WorkerProc hit an exception|EngineCore encountered a fatal error" "${LOG_FILE}" 2>/dev/null; then
        echo "[error] 引擎初始化失败, 日志末尾如下:" >&2
        tail -40 "${LOG_FILE}" >&2
        cleanup_server
        exit 1
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[error] 服务进程已退出, 日志末尾如下:" >&2
        tail -40 "${LOG_FILE}" >&2
        cleanup_server
        exit 1
    fi
    if curl -s -m 3 "http://127.0.0.1:${PORT}/health" -o /dev/null 2>&1; then
        break
    fi
    sleep 20
done
if (( SECONDS >= deadline )); then
    echo "[error] 等待服务就绪超时(60 分钟), 请查看日志: ${LOG_FILE}" >&2
    exit 1
fi
echo "[ready] 服务已就绪, 发起对话验证..."
verify_chat

echo ""
echo "[done] 服务运行中: http://0.0.0.0:${PORT} (model name: glm)"
echo "       停止服务: bash $0 --stop"
echo "       查看日志: tail -f ${LOG_FILE}"
