#!/bin/bash
# =============================================================================
# GLM-5.3-Flash 量化链路夜间长跑实验 (2026-11-10 规划, main 栈)
#
# 实验一: 官方 FP8 权重, 推理最小量化(indexer bf16 + MLA KV fp8; bfloat16 因 SMLA 长序列失控弃用)
#         aime×5 + gpqa×5
# 实验二: msmodelslim 从 FP8 权重量化 W4A4 (GLM53-flash-w4a4.yaml,
#         输出到独立目录, 不覆盖既有 BF16 来源的 w4a4)
# 实验三: 实验二权重 + indexer fp8 + MLA KV fp8, aime×5 + gpqa×5
# 实验四: 实验二权重 + indexer MXFP4 伪量化 + MLA KV fp8, aime×5 + gpqa×5
#
# 启动时间: 01:00(到点前等待; .started 标记存在则断点续跑跳过等待)
# 断点续跑: 各实验 .done 标记 + 各 run 有效成绩自动跳过
# 看门狗: 每 30 分钟检查, 卡死强制杀当前阶段(下轮 supervisor 拉起续跑)
# 产物: /mnt/share/w00608002/work/output/glm53flash/night2_<exp>/
#
# 用法: setsid nohup bash overnight_glm53flash_quant.sh > /dev/null 2>&1 &
# 停止: pkill -f overnight_glm53flash_quant
# =============================================================================

set +e
BASE=/mnt/share/w00608002/work/vllm-ascend
BENCH=/mnt/share/w00608002/work/benchmark
MSMODEL=/mnt/share/w00608002/work/msmodelslim
FP8_WEIGHTS=/mnt/share/w00936111/weights/GLM-5.3-Flash
W4A4_NEW=/mnt/share/w00608002/weight/GLM-5.3-Flash-FP8-w4a4
OUTROOT=/mnt/share/w00608002/work/output/glm53flash
MAIN_LOG="${OUTROOT}/night2_main.log"
REPEATS=5

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "${MAIN_LOG}"; }

# ---- 01:00 等待(断点续跑跳过) ----
if [[ ! -f "${OUTROOT}/.night2_started" ]]; then
    now=$(date +%s); t0=$(date -d "01:00" +%s)
    (( now >= t0 )) && t0=$((t0 + 86400))
    log "等待 01:00 启动(还有 $(( (t0-now)/60 )) 分钟)"
    sleep $((t0 - now))
    touch "${OUTROOT}/.night2_started"
fi
log "===== 夜间量化链路实验启动 ====="

# 看门狗: 每 30 分钟检查一次; 产物目录与主日志均无更新且超过 35 分钟视为卡死,
# 杀掉 ais_bench(解锁脚本循环)或 vllm(由遍前健康检查重启)。
(
    while sleep 1800; do
        [[ -f "${OUTROOT}/.night2_finished" ]] && exit 0
        latest=$(find "${OUTROOT}" -name "summary_*.csv" -newermt '40 minutes ago' -print -quit 2>/dev/null)
        [[ -n "$latest" ]] && continue
        recent_log=$(find "${MAIN_LOG}" -newermt '40 minutes ago' 2>/dev/null)
        [[ -n "$recent_log" ]] && continue
        if pgrep -f "ais_bench --models" >/dev/null 2>&1; then
            echo "[$(date '+%m-%d %H:%M:%S')] [watchdog] 40 分钟无进展, 杀卡死 ais_bench" >> "${MAIN_LOG}"
            pkill -9 -f "ais_bench --models"
        elif pgrep -f "VLLM::" >/dev/null 2>&1; then
            echo "[$(date '+%m-%d %H:%M:%S')] [watchdog] 40 分钟无进展且无评测, 重置服务" >> "${MAIN_LOG}"
            pkill -9 -f "VLLM::"
        fi
    done
) &
WATCHDOG_PID=$!

run_has_result() {
    find "$1" -name "summary_*.csv" -exec tail -1 {} \; 2>/dev/null \
        | awk -F, '{print $NF}' | grep -qE '^[0-9]+\.?[0-9]*$'
}

# ---- 评测: 数据集×N 遍(有效跳过), 依赖已就绪的服务 ----
run_eval() {  # $1=产物目录 $2=README补充说明
    local dir=$1 note=$2 ds name i wd csv score
    mkdir -p "$dir"
    [[ -f "${dir}/README.md" ]] || cat > "${dir}/README.md" <<EOF
# $(basename "$dir")
- 日期: 2026-11-11(夜间)
- ${note}
- 结果: 待回填
EOF
    for ds in aime2024_gen_0_shot_chat_prompt.py gpqa_gen_0_shot_cot_chat_prompt.py; do
        name="${ds%.py}"
        for i in $(seq 1 "${REPEATS}"); do
            wd="${dir}/${name}/run${i}"
            if [[ -d "$wd" ]] && run_has_result "$wd"; then
                log "[${dir##*/}] ${name} run${i} 已有成绩, 跳过"
                continue
            fi
            # 遍前健康检查: 服务挂了先重启(环境变量由调用方保持导出)
            if ! curl -s -m 3 http://127.0.0.1:9000/health -o /dev/null 2>&1; then
                log "[watch] 服务不健康, 重启后继续"
                stop_server
                start_server || { log "[error] 重启失败, 中止本阶段剩余遍"; return 1; }
            fi
            mkdir -p "$wd"
            ( cd "${BENCH}" && ais_bench --models vllm_api_general_chat_glm \
                --datasets "${ds}" --mode all --dump-eval-details --merge-ds \
                -w "$wd" > "$wd/aisbench.log" 2>&1 )
            csv=$(ls "$wd"/*/summary/summary_*.csv 2>/dev/null | head -1)
            score=$(tail -1 "$csv" 2>/dev/null | awk -F, '{print $NF}')
            log "[${dir##*/}] ${name} run${i} = ${score:-失败}"
            # 失败遍: 记录并在阶段末补跑(本轮先继续)
            [[ -z "${score}" ]] && echo "$wd" >> "${dir}/.failed_runs"
        done
    done
    # 补漏: 失败遍重试一轮(服务如不健康由下一遍的遍前检查处理)
    if [[ -f "${dir}/.failed_runs" ]]; then
        log "[retry] 补跑失败遍: $(wc -l < "${dir}/.failed_runs") 个"
        while read -r wd; do
            rm -rf "$wd"; mkdir -p "$wd"
            ds=$(basename "$(dirname "$wd")").py
            ( cd "${BENCH}" && ais_bench --models vllm_api_general_chat_glm \
                --datasets "${ds}" --mode all --dump-eval-details --merge-ds \
                -w "$wd" > "$wd/aisbench.log" 2>&1 )
            csv=$(ls "$wd"/*/summary/summary_*.csv 2>/dev/null | head -1)
            score=$(tail -1 "$csv" 2>/dev/null | awk -F, '{print $NF}')
            log "[retry] ${wd##*/aime/} = ${score:-仍失败}"
        done < "${dir}/.failed_runs"
        rm -f "${dir}/.failed_runs"
    fi
}

start_server() {  # 各实验环境变量由调用方导出
    log "起服务: MODEL=${MODEL_PATH##*/} IDX=${INDEXER_KV_DTYPE:-auto} KV=${KV_CACHE_DTYPE} SIM=${VLLM_ASCEND_KPOOL_MXFP4_SIM:-0} QUANT=${QUANT_METHOD:-auto}"
    ( cd "${BASE}" && PORT=9000 setsid nohup bash serve_glm53flash.sh \
        >> "${MAIN_LOG}" 2>&1 & )
    # 启动宽限期: 包装脚本预检+setsid exec 有数秒间隙, 立即 pgrep 会误杀竞态
    sleep 20
    local deadline=$((SECONDS + 1500))
    local grace=$((SECONDS + 120))
    until curl -s -m 3 http://127.0.0.1:9000/health -o /dev/null 2>&1; do
        if ! pgrep -f "[b]in/vllm serve" >/dev/null; then
            # 宽限期内可能仍在 exec 间隙, 包装脚本活着就不算死
            if (( SECONDS < grace )) && pgrep -f "serve_glm53flash.sh" >/dev/null; then
                sleep 5; continue
            fi
            log "[error] 服务进程退出"; return 1
        fi
        (( SECONDS < deadline )) || { log "[error] 服务就绪超时"; return 1; }
        sleep 20
    done
    log "服务就绪"
}

stop_server() {
    bash "${BASE}/serve_glm53flash.sh" --stop >> "${MAIN_LOG}" 2>&1
    pkill -9 -f "[V]LLM::" 2>/dev/null
    sleep 10
    log "服务已停"
}

# ---- 实验一: FP8 权重, 最小量化基线(indexer bf16 + KV fp8) ----
D1="${OUTROOT}/night2_exp1_fp8_bf16kv"
if [[ -f "${D1}/.done" ]]; then
    log "实验一已完成, 跳过"
else
    log "===== 实验一: FP8 权重 + 最小量化基线(idx bf16 + KV fp8) ====="
    export MODEL_PATH="${FP8_WEIGHTS}" KV_CACHE_DTYPE=fp8
    unset INDEXER_KV_DTYPE VLLM_ASCEND_KPOOL_MXFP4_SIM QUANT_METHOD
    if start_server; then
        run_eval "${D1}" "官方 FP8 权重, indexer bf16, MLA KV fp8(bfloat16 因 SMLA 长序列失控已弃用), 无伪量化"
        touch "${D1}/.done"
    fi
    stop_server
fi

# ---- 实验二: 从 FP8 权重量化 W4A4 ----
D2="${OUTROOT}/night2_exp2_quant"
if [[ -f "${D2}/.done" ]]; then
    log "实验二已完成, 跳过"
else
    log "===== 实验二: msmodelslim W4A4 量化(输入=FP8 权重) ====="
    mkdir -p "${D2}"
    rm -rf "${W4A4_NEW}"
    ( cd "${MSMODEL}" && CONFIG_PATH="${MSMODEL}/GLM53-flash-w4a4.yaml" \
      SAVE_PATH="${W4A4_NEW}" QUANT_DEVICE=2 \
      bash msmodelslim_glm53flash.sh ) > "${D2}/quant.log" 2>&1
    n_shards=$(find "${W4A4_NEW}" -name "quant_model_weights-*.safetensors" 2>/dev/null | wc -l)
    if grep -qE "SUCCESS" "${D2}/quant.log" && [[ -f "${W4A4_NEW}/quant_model_description.json" ]] && (( n_shards >= 40 )); then
        log "实验二量化完成(${n_shards} 分片)"
        touch "${D2}/.done"
    else
        log "[error] 实验二量化失败(分片 ${n_shards}), 后续实验三/四跳过"
        log "===== 结束(实验二失败) ====="
        exit 1
    fi
fi

# ---- 实验三: 新 W4A4 + indexer fp8 + MLA KV fp8 ----
D3="${OUTROOT}/night2_exp3_w4a4_c8"
if [[ -f "${D3}/.done" ]]; then
    log "实验三已完成, 跳过"
else
    log "===== 实验三: W4A4(fromfp8) + indexer C8 + MLA KV fp8 ====="
    export MODEL_PATH="${W4A4_NEW}" QUANT_METHOD=ascend INDEXER_KV_DTYPE=fp8 KV_CACHE_DTYPE=fp8
    unset VLLM_ASCEND_KPOOL_MXFP4_SIM
    if start_server; then
        run_eval "${D3}" "W4A4(fromfp8) 权重, indexer fp8(INDEXER_KV_DTYPE=fp8), MLA KV fp8"
        touch "${D3}/.done"
    fi
    stop_server
fi

# ---- 实验五(2026-11-11 增): 新 W4A4, 无 indexer 量化(idx bf16) + MLA KV fp8 ----
D5="${OUTROOT}/night2_exp5_w4a4_plain"
if [[ -f "${D5}/.done" ]]; then
    log "实验五已完成, 跳过"
else
    log "===== 实验五: W4A4(fromfp8) + indexer bf16 + MLA KV fp8(仅 W4A4) ====="
    export MODEL_PATH="${W4A4_NEW}" QUANT_METHOD=ascend KV_CACHE_DTYPE=fp8
    unset INDEXER_KV_DTYPE VLLM_ASCEND_KPOOL_MXFP4_SIM
    if start_server; then
        run_eval "${D5}" "W4A4(fromfp8) 权重, indexer bf16(无量化), MLA KV fp8 —— 隔离纯 W4A4 权重量化影响"
        touch "${D5}/.done"
    fi
    stop_server
fi

# ---- 实验四: 新 W4A4 + indexer MXFP4 伪量化 + MLA KV fp8 ----
D4="${OUTROOT}/night2_exp4_w4a4_mxfp4"
if [[ -f "${D4}/.done" ]]; then
    log "实验四已完成, 跳过"
else
    log "===== 实验四: W4A4(fromfp8) + indexer MXFP4 伪量化 + MLA KV fp8 ====="
    export MODEL_PATH="${W4A4_NEW}" QUANT_METHOD=ascend KV_CACHE_DTYPE=fp8
    export VLLM_ASCEND_KPOOL_MXFP4_SIM=1
    unset INDEXER_KV_DTYPE
    if start_server; then
        run_eval "${D4}" "W4A4(fromfp8) 权重, indexer bf16 + MXFP4 伪量化(env), MLA KV fp8"
        touch "${D4}/.done"
    fi
    stop_server
fi

# ---- 汇总报告 ----
log "===== 汇总 ====="
python3 - "${OUTROOT}" >> "${MAIN_LOG}" 2>&1 <<'EOF'
import csv, glob, json, os, statistics, sys
root = sys.argv[1]
exps = {"exp1_fp8_bf16kv": "实验一 FP8+idx bf16+KV fp8", "exp3_w4a4_c8": "实验三 W4A4+C8",
        "exp4_w4a4_mxfp4": "实验四 W4A4+MXFP4伪量化", "exp5_w4a4_plain": "实验五 仅W4A4(idx bf16)"}
lines = ["# 夜间量化链路实验汇总", "", "| 实验 | aime2024 | gpqa |", "| --- | --- | --- |"]
for d, name in exps.items():
    cells = []
    for ds in ("aime2024_gen_0_shot_chat_prompt", "gpqa_gen_0_shot_cot_chat_prompt"):
        vals = []
        for p in glob.glob(os.path.join(root, f"night2_{d}", ds, "run*", "**", "summary_*.csv"), recursive=True):
            try:
                rows = list(csv.reader(open(p)))
                if len(rows) >= 2 and rows[-1][-1].replace('.', '', 1).isdigit():
                    vals.append(float(rows[-1][-1]))
            except Exception:
                pass
        cells.append(f"{statistics.mean(vals):.2f}±{statistics.stdev(vals) if len(vals)>1 else 0:.2f} (n={len(vals)})" if vals else "无")
    lines.append(f"| {name} | {cells[0]} | {cells[1]} |")
rep = os.path.join(root, "night2_report.md")
open(rep, "w").write("\n".join(lines) + "\n")
print(f"报告: {rep}")
print("\n".join(lines))
EOF
[[ -n "${WATCHDOG_PID:-}" ]] && kill "${WATCHDOG_PID}" 2>/dev/null
touch "${OUTROOT}/.night2_finished"
log "===== 全部结束 ====="
