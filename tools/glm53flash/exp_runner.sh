#!/bin/bash
# exp_runner.sh 6|7|7o — night2 实验六/七/七o 一键运行
#   6  = W4A4 + indexer bf16 + MXFP8 伪量化(e4m3+E8M0)
#   7  = W4A4 + indexer bf16 + MXFP4 伪量化(C7.25) + Hadamard
#   7o = W4A4 + indexer bf16 + MXFP4 伪量化(C7.25) + OSCAR(校准集拟合)
# 用法: bash tools/glm53flash/exp_runner.sh 7o
# 与 experiment_tracking.md §5.6 中"六/七/七o 运行脚本"保持同步
set +e
BASE=/mnt/share/w00608002/work/vllm-ascend
BENCH=/mnt/share/w00608002/work/benchmark
OUTROOT=/mnt/share/w00608002/work/output/glm53flash
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

# ---- 1) 实验选择(唯一变量=indexer 量化方式) ----
unset INDEXER_KV_DTYPE VLLM_ASCEND_KPOOL_HADAMARD_PRE VLLM_ASCEND_KPOOL_MXFP4_SIM \
      VLLM_ASCEND_KPOOL_MXFP8_SIM VLLM_ASCEND_KPOOL_OSCAR_ROTATION VLLM_ASCEND_KPOOL_OSCAR_CKPT
case "${1:-}" in
  6)  D="${OUTROOT}/night2_exp6_w4a4_mxfp8"
      export VLLM_ASCEND_KPOOL_MXFP8_SIM=1 ;;
  7)  D="${OUTROOT}/night2_exp7_w4a4_mxfp4_hadamard"
      export VLLM_ASCEND_KPOOL_MXFP4_SIM=1 VLLM_ASCEND_KPOOL_HADAMARD_PRE=1 ;;
  7o) D="${OUTROOT}/night2_exp7o_w4a4_mxfp4_oscar"
      export VLLM_ASCEND_KPOOL_MXFP4_SIM=1 VLLM_ASCEND_KPOOL_OSCAR_ROTATION=1
      export VLLM_ASCEND_KPOOL_OSCAR_CKPT=${OUTROOT}/oscar_calib/oscar_indexer_calib.pt ;;
  *) echo "用法: bash $0 6|7|7o"; exit 1 ;;
esac
export MODEL_PATH=/mnt/share/w00608002/weight/GLM-5.3-Flash-FP8-w4a4
export QUANT_METHOD=ascend KV_CACHE_DTYPE=fp8
log "===== 实验 ${1}: ${D} ====="

# ---- 2) 起服务 ----
( cd "${BASE}" && PORT=9000 setsid nohup bash serve_glm53flash.sh > "${D}_serve.log" 2>&1 & )
sleep 20; deadline=$((SECONDS + 1500)); grace=$((SECONDS + 120))
until curl -s -m 3 http://127.0.0.1:9000/health -o /dev/null 2>&1; do
    if ! pgrep -f "[b]in/vllm serve" >/dev/null; then
        if (( SECONDS < grace )) && pgrep -f "serve_glm53flash.sh" >/dev/null; then sleep 5; continue; fi
        log "[error] 服务进程退出"; exit 1
    fi
    (( SECONDS < deadline )) || { log "[error] 就绪超时"; exit 1; }
    sleep 20
done; log "服务就绪"

# ---- 3) aime×5 + gpqa×3(断点 run 级) ----
mkdir -p "${D}"
has() { find "$1" -name "summary_*.csv" -exec tail -1 {} \; 2>/dev/null | awk -F, '{print $NF}' | grep -qE '^[0-9]+\.?[0-9]*$'; }
for ds in aime2024_gen_0_shot_chat_prompt.py gpqa_gen_0_shot_cot_chat_prompt.py; do
    name=${ds%.py}; rep=5; [[ ${name} == gpqa* ]] && rep=3
    for i in $(seq 1 ${rep}); do
        wd="${D}/${name}/run${i}"; has "${wd}" && { log "${name} run${i} 已有成绩, 跳过"; continue; }
        mkdir -p "${wd}"
        ( cd "${BENCH}" && ais_bench --models vllm_api_general_chat_glm \
            --datasets "${ds}" --mode all --dump-eval-details --merge-ds \
            -w "${wd}" > "${wd}/aisbench.log" 2>&1 )
        log "${name} run${i} = $(tail -1 $(ls ${wd}/*/summary/summary_*.csv 2>/dev/null | head -1) 2>/dev/null | awk -F, '{print $NF}')"
    done
done

# ---- 4) 停服务 ----
bash "${BASE}/serve_glm53flash.sh" --stop; log "===== 实验 ${1} 结束 ====="
