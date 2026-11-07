#!/bin/bash
# GLM-5.3-Flash 单机纯服务启动脚本 (8x Ascend950DT, 96GB x 8)
# 与 run_glm53flash.sh 同参数, 但只起服务: 不做对话验证、不挂 profiler。
# 布局: dp1 x tp8 x ep8, 288 experts (每 EP rank 36 个)。
#
# 用法: bash serve_glm53flash.sh [--stop] [--fg]
#   --stop   停止已启动的服务并退出
#   --fg     前台运行(日志直接打屏, Ctrl-C 停止); 默认后台运行
#
# 环境变量覆盖: MODEL_PATH PORT MAX_MODEL_LEN MAX_NUM_SEQS
#   MAX_NUM_BATCHED_TOKENS GPU_MEM_UTIL SPEC_TOKENS LOAD_FORMAT QUANT_METHOD

set -e

BASE=/mnt/share/w00608002/work/vllm-ascend
LOG_DIR=${BASE}/logs
MODEL_PATH=${MODEL_PATH:-/mnt/share/w00936111/weights/GLM-5.3-Flash}
PORT=${PORT:-8000}
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/server_single_$(date +%m%d_%H%M%S).log"

# ---- 参数(可用环境变量覆盖) ----
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-132096}
export MAX_NUM_SEQS=${MAX_NUM_SEQS:-64}
export MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}
export GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.9}
export SPEC_TOKENS=${SPEC_TOKENS:-3}                   # deepseek_mtp draft tokens
LOAD_FORMAT=${LOAD_FORMAT:-}                           # 置 dummy 则跳过真实权重
# 量化方式: 本地 checkpoint config.json 标注 quant_method=fp8, 由 vllm-ascend
# 注册的 AscendFp8Config 自动接管, 不能显式传 --quantization ascend (两者冲突)。
# 若换成 quant_method=ascend 的 ModelSlim checkpoint, 设 QUANT_METHOD=ascend。
QUANT_METHOD=${QUANT_METHOD:-}
# 编译配置: 可用 COMPILATION_CONFIG 整体覆盖(如 A/B 实验关 quant_fp8 融合:
# COMPILATION_CONFIG='{"cudagraph_mode":"FULL_DECODE_ONLY","custom_ops":["all","-quant_fp8"]}')
COMPILATION_CONFIG=${COMPILATION_CONFIG:-'{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1,2,4,8,16,32,64,96,128]}'}

# ---- 教程要求的环境变量 (GLM5.3-Flash 仅支持 model runner V1) ----
export VLLM_USE_V2_MODEL_RUNNER=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024

stop_server() {
    # 服务以 setsid 启动(独立进程组), 整组杀死可覆盖 APIServer + EngineCore +
    # proctitle 改名后的 Worker。只匹配本脚本的模型路径, 避免误杀其他 vllm 任务。
    local pid pgid
    pid=$(pgrep -f "vllm serve.*${MODEL_PATH}" | head -1)
    if [[ -n "${pid}" ]]; then
        pgid=$(ps -o pgid= -p "${pid}" | tr -d ' ')
        kill -9 -- -"${pgid}" 2>/dev/null
        echo "[stop] 已停止 vllm serve (pgid=${pgid})"
    else
        echo "[stop] 没有运行中的服务"
    fi
    # 兜底: APIServer 已死但 Worker 残留(proctitle 改名后按模型路径匹配不到)。
    # pgrep 只见本命名空间进程; cwd 限定本工作区, 双保险不误伤。
    local wpid cwd
    for wpid in $(pgrep -f "VLLM::" 2>/dev/null); do
        cwd=$(readlink "/proc/${wpid}/cwd" 2>/dev/null) || continue
        if [[ "${cwd}" == /mnt/share/w00608002/* ]]; then
            kill -9 "${wpid}" 2>/dev/null && echo "[stop] 清理残留 worker PID=${wpid}"
        fi
    done
}
if [[ "${1:-}" == "--stop" ]]; then
    stop_server
    exit 0
fi

FG=0
[[ "${1:-}" == "--fg" ]] && FG=1

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

# 端口占用检查
if curl -s -m 3 "http://127.0.0.1:${PORT}/health" -o /dev/null; then
    echo "[check] 端口 ${PORT} 已有健康服务在运行(如需重启: bash $0 --stop)"
    exit 0
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

# SPEC_TOKENS=0 时关闭 MTP 投机解码(A/B 对照用)
SPEC_ARGS=()
if [[ "${SPEC_TOKENS}" != "0" ]]; then
    SPEC_ARGS=(--speculative-config "{\"num_speculative_tokens\": ${SPEC_TOKENS}, \"method\": \"deepseek_mtp\", \"enforce_eager\": true}")
fi

SERVE_CMD=(
  vllm serve "${MODEL_PATH}"
  --host 0.0.0.0
  --port "${PORT}"
  --tensor-parallel-size 8
  --enable-expert-parallel
  --seed 1024
  --served-model-name glm
  --kv-cache-dtype fp8
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --trust-remote-code
  --gpu-memory-utilization "${GPU_MEM_UTIL}"
  --async-scheduling
  --enable-prefix-caching
  --reasoning-parser glm47
  --limit-mm-per-prompt '{"image":0,"video":0}'
  --compilation-config "${COMPILATION_CONFIG}"
  --additional-config '{"enable_fused_mc2": 0}'
  "${SPEC_ARGS[@]}"
  "${LOAD_ARGS[@]}"
  "${QUANT_ARGS[@]}"
)

if [[ "${FG}" == "1" ]]; then
    echo "[start] 前台启动(日志: ${LOG_FILE}), Ctrl-C 停止"
    setsid "${SERVE_CMD[@]}" 2>&1 | tee "${LOG_FILE}"
    exit 0
fi

echo "[start] 日志: ${LOG_FILE}"
setsid "${SERVE_CMD[@]}" > "${LOG_FILE}" 2>&1 &

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

echo ""
echo "[ready] 服务已就绪: http://0.0.0.0:${PORT} (model name: glm)"
echo "        停止服务: bash $0 --stop"
echo "        查看日志: tail -f ${LOG_FILE}"
