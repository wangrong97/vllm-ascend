#!/bin/bash
# =============================================================================
# GLM-5.3-Flash 夜间实验编排 (8x Ascend950DT)
#
# 流程:
#   0. 等待到半夜 00:00(若启动时已在 00:00-01:00 之间则立即开始)
#   1. 等待 8 卡全部空闲且持续 10 分钟以上
#   2. 任务1: 基线精度 (FP8 权重 + vllm 服务 + ais_bench, 5 数据集 x 5 遍)
#   3. 任务2: msmodelslim W4A4 量化(单卡 7 号, 逐层校准)
#   4. 任务3: W4A4 权重精度 (同任务1)
#   5. 汇总 report.md: 每数据集 5 遍 mean±std, 基线 vs W4A4 对照
#
# 所有产出(含本脚本与关键配置归档)写入 output/, 每任务一个带日期的目录:
#   output/task1_baseline_YYYYMMDD/   基线精度(5 数据集 x 5 遍)
#   output/task2_quant_YYYYMMDD/      量化日志
#   output/task3_w4a4_YYYYMMDD/       W4A4 精度
#   output/scripts_config_YYYYMMDD/   脚本与配置归档
#   output/report_YYYYMMDD.md         精度对照报告
#
# 用法: bash run_glm53flash_pipeline.sh          # 挂后台: nohup bash ... &
# =============================================================================

# 不用 set -e: 单任务失败不应中断后续任务, 各步骤显式判断
set +e

VLLM_ASCEND=/mnt/share/w00608002/work/vllm-ascend
BENCH=/mnt/share/w00608002/work/benchmark
MSMODELSLIM=/mnt/share/w00608002/work/msmodelslim
BASELINE_WEIGHTS=/mnt/share/w00936111/weights/GLM-5.3-Flash
W4A4_WEIGHTS=/mnt/share/w00608002/weight/GLM-5.3-Flash-w4a4
SERVE_PORT=9000    # ais_bench 模型配置 vllm_api_general_chat_glm 写死 9000

cd "${VLLM_ASCEND}"   # 固定工作目录, 避免依赖调用处的 cwd

# 归档结构: output/ 下每个任务一个带日期的目录(日期为任务实际执行日)
STAMP=$(date +%Y%m%d)   # 启动日; 过 00:00 后 init_dirs 重算为执行日
OUT=/mnt/share/w00608002/work/output/glm53flash

init_dirs() {
    STAMP=$(date +%Y%m%d)
    DIR_TASK1="${OUT}/task1_baseline_${STAMP}"
    DIR_TASK2="${OUT}/task2_quant_${STAMP}"
    DIR_TASK3="${OUT}/task3_w4a4_${STAMP}"
    REPORT="${OUT}/report_${STAMP}.md"
    SCORES_JSON="${OUT}/scores_raw_${STAMP}.json"
    mkdir -p "${DIR_TASK1}" "${DIR_TASK2}" "${DIR_TASK3}"
    MAIN_LOG="${OUT}/main_${STAMP}.log"
}

DATASETS=(
  aime2024_gen_0_shot_chat_prompt.py
  gpqa_gen_0_shot_cot_chat_prompt.py
  gsm8k_gen_0_shot_cot_chat_prompt.py
  humaneval_gen_0_shot.py
  math500_gen_0_shot_cot_chat_prompt.py
)
REPEATS=5

mkdir -p "${OUT}"
init_dirs

log() {
    echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "${MAIN_LOG}"
    touch "${OUT}/.activity" 2>/dev/null || true   # 看门狗心跳
}

# ---- 看门狗状态: 当前阶段名/监控目录/服务模型路径 ----
set_phase() {  # $1=阶段名(serve*/eval*/quant*) $2=活跃目录 $3=模型路径(serve 阶段用)
    echo "$1" > "${OUT}/.phase"
    echo "$2" > "${OUT}/.watch_dir"
    echo "${3:-}" > "${OUT}/.phase_model"
    touch "${OUT}/.activity"
}

# 30 分钟无进展判定卡死: 阶段活跃目录无新文件且主日志无心跳
start_watchdog() {
    (
        while [[ ! -f "${OUT}/.done" ]]; do
            sleep 1800
            [[ -f "${OUT}/.done" ]] && break
            [[ -f "${OUT}/.phase" ]] || continue
            local_phase=$(cat "${OUT}/.phase" 2>/dev/null)
            watch_dir=$(cat "${OUT}/.watch_dir" 2>/dev/null)
            recent=""
            [[ -n "${watch_dir}" && -d "${watch_dir}" ]] && \
                recent=$(find "${watch_dir}" -type f -newermt '30 minutes ago' -print -quit 2>/dev/null)
            [[ -z "${recent}" ]] && \
                recent=$(find "${OUT}/.activity" -newermt '30 minutes ago' 2>/dev/null)
            if [[ -z "${recent}" ]]; then
                echo "[$(date '+%m-%d %H:%M:%S')] 看门狗: 阶段 [${local_phase}] 超 30 分钟无进展, 强制终止" >> "${MAIN_LOG}"
                case "${local_phase}" in
                    serve*)  # 释放卡死的 vllm 服务
                        MODEL_PATH=$(cat "${OUT}/.phase_model" 2>/dev/null) \
                            bash "${VLLM_ASCEND}/serve_glm53flash.sh" --stop >> "${MAIN_LOG}" 2>&1
                        ;;
                    eval*)  pkill -9 -f "ais_bench" ;;
                    quant*) pkill -9 -f "msmodelslim quant" ;;
                esac
                touch "${OUT}/.activity"
            fi
        done
    ) &
    WATCHDOG_PID=$!
    log "看门狗已启动(每 30 分钟检查一次进展)"
}

# -----------------------------------------------------------------------------
# 0. 配置与脚本归档(便于回溯): 归档到各任务自己的 scripts/ 子目录
# -----------------------------------------------------------------------------
archive_configs() {
    local eval_scripts=(serve_glm53flash.sh run_glm53flash.sh run_glm53flash_pipeline.sh)
    local t
    for t in "${DIR_TASK1}" "${DIR_TASK3}"; do
        mkdir -p "${t}/scripts"
        for s in "${eval_scripts[@]}"; do
            cp -f "${VLLM_ASCEND}/${s}" "${t}/scripts/" 2>/dev/null
        done
        cp -f "${BENCH}/aisbanch_glm.sh" "${t}/scripts/" 2>/dev/null
        cp -f "${BENCH}/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_chat_glm.py" "${t}/scripts/" 2>/dev/null
    done
    mkdir -p "${DIR_TASK2}/scripts"
    cp -f "${MSMODELSLIM}/msmodelslim_glm53flash.sh" "${DIR_TASK2}/scripts/" 2>/dev/null
    cp -f "${MSMODELSLIM}/GLM53-flash-w4a4.yaml" "${DIR_TASK2}/scripts/" 2>/dev/null
    log "配置归档完成(各任务 scripts/ 子目录)"
}

# -----------------------------------------------------------------------------
# 1. 启动窗口等待: 00:00 前若 8 卡连续空闲 30 分钟则提前启动;
#    到 00:00 或已在 00:00-01:00 窗口则进入常规 10 分钟空闲确认。
#    提前启动时置 EARLY_START=1(已验证 30 分钟稳定, 跳过 10 分钟确认)。
# -----------------------------------------------------------------------------
EARLY_START=0
wait_for_start_window() {
    # 重启动(已有历史进度)不再等窗口: 满足空闲条件即可续跑
    if [[ -f "${OUT}/.started" ]]; then
        log "检测到历史进度标记, 本次为断点续跑, 跳过启动窗口等待"
        return 0
    fi
    local now today0 target stable=0
    now=$(date +%s)
    today0=$(date -d "00:00" +%s)
    if (( now >= today0 && now < today0 + 3600 )); then
        log "当前已在 00:00-01:00 窗口内, 直接做空闲检测"
        return 0
    fi
    target=$(( today0 + 86400 ))
    (( now < today0 )) && target=${today0}
    log "等待启动窗口: 到 00:00 ($(date -d @${target} '+%m-%d %H:%M')), 或 8 卡连续空闲 30 分钟提前启动"
    while true; do
        now=$(date +%s)
        if (( now >= target )); then
            log "已到 00:00"
            return 0
        fi
        if gpus_all_free; then
            stable=$((stable + 1))
            if (( stable >= 60 )); then   # 60 次 x 30s = 30 分钟
                EARLY_START=1
                log "8 卡已连续空闲 30 分钟, 提前启动任务"
                return 0
            fi
        else
            if (( stable > 0 )); then
                log "检测到占用, 提前启动计时重置(已稳定 $((stable / 2)) 分钟)"
            fi
            stable=0
        fi
        sleep 30
    done
}

# -----------------------------------------------------------------------------
# 2. 等 8 卡空闲且持续 10 分钟(每 30s 检查一次, 连续 20 次全空闲才放行)
# -----------------------------------------------------------------------------
gpus_all_free() {
    # 每卡 HBM 占用 < 10000MB 视为空闲(基线约 4.8GB)
    local n
    # 行格式: | ... | 0  0 / 0  <used_MB> / 98304 |  → used 是倒数第 3 列
    n=$(npu-smi info 2>/dev/null | awk '/\/ 98304/ { if ($(NF-3)+0 < 10000) c++ } END { print c+0 }')
    [[ "$n" == "8" ]]
}

wait_gpus_free() {
    local stable=0
    log "开始检测 8 卡空闲状态(需连续 10 分钟全空闲)..."
    while true; do
        if gpus_all_free; then
            stable=$((stable + 1))
            if (( stable >= 20 )); then
                log "8 卡已连续 10 分钟空闲, 放行"
                return 0
            fi
        else
            if (( stable > 0 )); then
                log "检测到占用, 空闲计时重置(已稳定 ${stable} 次)"
            fi
            stable=0
        fi
        sleep 30
    done
}

# -----------------------------------------------------------------------------
# 服务启停封装
# -----------------------------------------------------------------------------
start_server() {  # $1=MODEL_PATH $2=QUANT_METHOD(可空)
    local model=$1 quant=$2
    log "启动 vllm 服务: MODEL_PATH=${model} QUANT_METHOD=${quant:-auto} PORT=${SERVE_PORT}"
    MODEL_PATH="${model}" QUANT_METHOD="${quant}" PORT=${SERVE_PORT} \
        bash "${VLLM_ASCEND}/serve_glm53flash.sh" >> "${MAIN_LOG}" 2>&1
    local rc=$?
    if (( rc != 0 )); then
        log "[error] 服务启动失败(rc=${rc}), 详见 serve 日志"
        return 1
    fi
    log "服务就绪"
    return 0
}

stop_server() {  # $1=MODEL_PATH
    MODEL_PATH="$1" bash "${VLLM_ASCEND}/serve_glm53flash.sh" --stop >> "${MAIN_LOG}" 2>&1
    sleep 10
    log "服务已停止"
}

# 释放 vllm 服务: 停止(serve 脚本按进程组+proctitle 兜底) → 确认显存回落。
# 注意: npu-smi 显示的 PID 是宿主机命名空间的, 容器内 /proc 查不到,
# 残留清理必须用本命名空间的 pgrep(proctitle "VLLM::")+ cwd 限定。
release_service() {  # $1=MODEL_PATH
    stop_server "$1"
    local pid cwd
    for pid in $(pgrep -f "VLLM::" 2>/dev/null); do
        cwd=$(readlink "/proc/${pid}/cwd" 2>/dev/null) || continue
        if [[ "${cwd}" == /mnt/share/w00608002/* ]]; then
            log "清理残留进程 PID=${pid} (cwd=${cwd})"
            kill -9 "${pid}" 2>/dev/null
        fi
    done
    local i
    for i in $(seq 1 10); do
        gpus_all_free && { log "显存已释放(8 卡全部回落)"; return 0; }
        sleep 30
    done
    log "[warn] 5 分钟内显存未完全释放, 继续下一阶段(后续 GPU 空闲检测仍兜底)"
}

# -----------------------------------------------------------------------------
# 精度评测: $1=tag (task1_baseline / task3_w4a4)
# 每个数据集跑 REPEATS 遍, 各自独立 work-dir; 已有有效(数值)成绩的 run 自动跳过
# -----------------------------------------------------------------------------
run_has_result() {  # $1=run 目录: 存在数值 accuracy 的 summary 即视为已完成
    find "$1" -name "summary_*.csv" -exec tail -1 {} \; 2>/dev/null \
        | awk -F, '{print $NF}' | grep -qE '^[0-9]+\.?[0-9]*$'
}

run_eval() {
    local tag=$1
    local eval_dir=$2
    mkdir -p "${eval_dir}"
    for ds in "${DATASETS[@]}"; do
        local name="${ds%.py}"
        for i in $(seq 1 "${REPEATS}"); do
            local wd="${eval_dir}/${name}/run${i}"
            if [[ -d "${wd}" ]] && run_has_result "${wd}"; then
                log "[${tag}] ${name} 第 ${i} 遍已有有效成绩, 跳过"
                continue
            fi
            mkdir -p "${wd}"
            set_phase "eval" "${wd}"
            log "[${tag}] ${name} 第 ${i}/${REPEATS} 遍开始"
            ( cd "${BENCH}" && ais_bench \
                --models vllm_api_general_chat_glm \
                --datasets "${ds}" \
                --mode all --dump-eval-details --merge-ds \
                -w "${wd}" ) > "${wd}/aisbench.log" 2>&1
            local rc=$?
            if (( rc != 0 )); then
                log "[${tag}] ${name} 第 ${i} 遍失败(rc=${rc}), 日志: ${wd}/aisbench.log"
            else
                local score
                score=$(find "${wd}" -name "summary_*.csv" -exec tail -1 {} \; 2>/dev/null | awk -F, '{print $NF}')
                log "[${tag}] ${name} 第 ${i} 遍完成, accuracy=${score:-未知}"
            fi
        done
    done
}

# -----------------------------------------------------------------------------
# 汇总报告: 解析所有 summary csv, 生成 mean±std 对照表
# -----------------------------------------------------------------------------
summarize() {
    python3 - "${DIR_TASK1}" "${DIR_TASK3}" "${REPORT}" "${SCORES_JSON}" <<'EOF'
import csv, glob, json, os, statistics, sys

dirs = {"基线(FP8)": sys.argv[1], "W4A4": sys.argv[2]}
report_path, scores_path = sys.argv[3], sys.argv[4]
# scores[tag][dataset] = [run scores]
scores: dict[str, dict[str, list[float]]] = {}
for tag, base in dirs.items():
    scores[tag] = {}
    for csv_path in glob.glob(os.path.join(base, "*", "run*", "**", "summary_*.csv"), recursive=True):
        rel = os.path.relpath(csv_path, base)
        ds_name = rel.split(os.sep)[0]  # <dataset>/runN/.../summary_*.csv
        try:
            with open(csv_path) as f:
                rows = list(csv.reader(f))
            if len(rows) >= 2:
                scores[tag].setdefault(ds_name, []).append(float(rows[-1][-1]))
        except Exception:
            continue

lines = ["# GLM-5.3-Flash 精度对照报告", "",
         f"- 生成时间: {__import__('datetime').datetime.now()}",
         f"- 基线目录: {sys.argv[1]}",
         f"- W4A4目录: {sys.argv[2]}",
         f"- 每个数据集重复 5 遍, 指标: accuracy", "",
         "| 数据集 | 基线(FP8) mean±std | W4A4 mean±std | 差值 |",
         "| --- | --- | --- | --- |"]
tags = list(dirs)
all_ds = sorted(set(scores[tags[0]]) | set(scores[tags[1]]))
for ds in all_ds:
    cells = []
    for tag in tags:
        vals = scores[tag].get(ds) or []
        cells.append(
            f"{statistics.mean(vals):.2f}±{(statistics.stdev(vals) if len(vals) > 1 else 0.0):.2f} (n={len(vals)})"
            if vals else "无数据"
        )
    base_vals = scores[tags[0]].get(ds)
    w4 = scores[tags[1]].get(ds)
    diff = f"{statistics.mean(w4) - statistics.mean(base_vals):+.2f}" if base_vals and w4 else "-"
    lines.append(f"| {ds} | {cells[0]} | {cells[1]} | {diff} |")

report = "\n".join(lines) + "\n"
with open(report_path, "w") as f:
    f.write(report)
with open(scores_path, "w") as f:
    json.dump(scores, f, indent=2)
print(report)
EOF
}

# =============================================================================
# 主流程(可重入: 各阶段完成后置 .done 标记, 被 kill 重启后自动跳过已完成阶段)
# =============================================================================
log "===== 实验编排启动, 产出目录=${OUT} ====="
wait_for_start_window
init_dirs        # 以实际执行日命名产物目录
log "日期标记=${STAMP} (提前启动=${EARLY_START})"
archive_configs
if [[ "${SKIP_IDLE_WAIT:-0}" == "1" ]]; then
    log "SKIP_IDLE_WAIT=1, 跳过空闲检测直接启动"
elif (( EARLY_START != 1 )); then
    wait_gpus_free   # 提前启动已验证 30 分钟稳定, 无需再确认
fi
touch "${OUT}/.started"   # 历史进度标记: 被 kill 后重启直接进入续跑
start_watchdog

# ---- 任务1: 基线精度(可跳过) ----
if [[ -f "${DIR_TASK1}/.done" ]]; then
    log "===== 任务1 已完成(标记存在), 跳过 ====="
else
    log "===== 任务1: 基线精度评测(FP8 权重) ====="
    set_phase "serve_task1" "${VLLM_ASCEND}/logs" "${BASELINE_WEIGHTS}"
    if start_server "${BASELINE_WEIGHTS}" ""; then
        run_eval task1_baseline "${DIR_TASK1}"
        touch "${DIR_TASK1}/.done"
    else
        log "[error] 任务1本轮失败: 服务启动失败(重启管线后自动重试)"
    fi
    set_phase "release_task1" "${OUT}"
    release_service "${BASELINE_WEIGHTS}"
fi

# ---- 任务2: W4A4 量化(已有有效产物则跳过; FORCE_REQUANT=1 强制重做) ----
quant_ok=1
if [[ -f "${W4A4_WEIGHTS}/config.json" && -f "${W4A4_WEIGHTS}/quant_model_description.json" \
      && "${FORCE_REQUANT:-0}" != "1" ]]; then
    n_existing=$(find "${W4A4_WEIGHTS}" -name "quant_model_weights-*.safetensors" 2>/dev/null | wc -l)
    if (( n_existing >= 30 )); then
        log "===== 任务2: 量化产物已存在且完整(${n_existing} 分片), 跳过 ====="
    else
        quant_ok=0   # 产物不全, 视为失败并重做
    fi
else
    quant_ok=0
fi
if (( quant_ok == 0 )); then
    log "===== 任务2: msmodelslim W4A4 量化 ====="
    rm -rf "${W4A4_WEIGHTS}"   # 清除历史部分产物, 避免旧文件混淆
    set_phase "quant" "${DIR_TASK2}"
    ( cd "${MSMODELSLIM}" && bash msmodelslim_glm53flash.sh ) > "${DIR_TASK2}/quant.log" 2>&1
    quant_ok=1
    # 量化脚本不透传失败码, 用产物完整性判断
    if [[ ! -f "${W4A4_WEIGHTS}/config.json" || ! -f "${W4A4_WEIGHTS}/quant_model_description.json" ]]; then
        quant_ok=0
    fi
    if grep -qE "UnexpectedError|InvalidModelError" "${DIR_TASK2}/quant.log" 2>/dev/null; then
        quant_ok=0
    fi
    n_shards=$(find "${W4A4_WEIGHTS}" -name "quant_model_weights-*.safetensors" 2>/dev/null | wc -l)
    log "量化产物: safetensors 分片数=${n_shards}"
    (( n_shards < 30 )) && quant_ok=0
fi

if (( quant_ok == 0 )); then
    log "[error] 任务2量化失败, 跳过任务3(修复后重启管线自动续跑)"
else
    log "任务2量化完成/已就绪"
    # ---- 任务3: W4A4 精度(可跳过) ----
    if [[ -f "${DIR_TASK3}/.done" ]]; then
        log "===== 任务3 已完成(标记存在), 跳过 ====="
    else
        log "===== 任务3: W4A4 权重精度评测 ====="
        # AscendV1(modelslim)格式需显式 --quantization ascend
        set_phase "serve_task3" "${VLLM_ASCEND}/logs" "${W4A4_WEIGHTS}"
        if start_server "${W4A4_WEIGHTS}" "ascend"; then
            run_eval task3_w4a4 "${DIR_TASK3}"
            touch "${DIR_TASK3}/.done"
        else
            log "[error] 任务3本轮失败: 服务启动失败(重启管线后自动重试)"
        fi
        set_phase "release_task3" "${OUT}"
        release_service "${W4A4_WEIGHTS}"
    fi
fi

# ---- 汇总(每次都会重新生成, 把已有数据汇总进报告) ----
log "===== 汇总精度报告 ====="
set_phase "summarize" "${OUT}"
summarize | tee -a "${MAIN_LOG}"
touch "${OUT}/.done"   # 通知看门狗退出
[[ -n "${WATCHDOG_PID:-}" ]] && kill "${WATCHDOG_PID}" 2>/dev/null
touch "${OUT}/.pipeline_finished"   # 通知 supervisor 全部完成, 不再重启
log "===== 全部结束, 报告: ${REPORT} ====="
