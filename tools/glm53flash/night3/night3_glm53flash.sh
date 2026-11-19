#!/bin/bash
# 夜间续跑: 六(补 gpqa run3) → 七o2A(W4A4 激活 dump+拟合) → 七o2B(评测)
#           → 八A(FP8→W4A4-KDA8 量化) → 八B(KDA8 激活 dump+拟合) → 八C(评测)
#   七o/七/三h 已 done 自动跳过; 01:00 起(.night3_started 跳过等待); 断点续跑; 30min 看门狗
# 七o2 差分: OSCAR 拟合源(官方 FP8 vs W4A4 自身); 八差分: KDA8 量化权重 + 自身拟合矩阵(vs 七o2)
set +e
BASE=/mnt/share/w00608002/work/vllm-ascend
BENCH=/mnt/share/w00608002/work/benchmark
OUTROOT=/mnt/share/w00608002/work/output/glm53flash
MAIN_LOG="${OUTROOT}/night3_main.log"
REPEATS_AIME=5
REPEATS_GPQA=3   # 2026-11-11 用户指定 gpqa 减为 3 轮
# 注意: main 分支 V1 已移除, 必须用 V2 runner(不再 export VLLM_USE_V2_MODEL_RUNNER)
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "${MAIN_LOG}"; }

# ---- 01:00 等待 ----
if [[ ! -f "${OUTROOT}/.night3_started" ]]; then
    now=$(date +%s); t0=$(date -d "01:00" +%s)
    (( now >= t0 )) && t0=$((t0 + 86400))
    log "等待 01:00 启动(还有 $(( (t0-now)/60 )) 分钟)"
    sleep $((t0 - now))
    touch "${OUTROOT}/.night3_started"
fi
log "===== night3 启动: 七o(先) → 七(补run3) → 六; 三h 已 done 自动跳过 ====="

stage_complete() {  # $1=目录: 该目录下 aime 全 5 遍 + gpqa 全 3 遍都有有效成绩
    local dir=$1 ds name n_need total=0 ok=0 r
    for ds in aime2024_gen_0_shot_chat_prompt gpqa_gen_0_shot_cot_chat_prompt; do
        name="${ds}"; n_need=5
        [[ "${name}" == gpqa* ]] && n_need=3
        for r in $(seq 1 "${n_need}"); do
            total=$((total + 1))
            run_has_result "${dir}/${name}/run${r}" && ok=$((ok + 1))
        done
    done
    (( ok == total ))
}

run_has_result() {
    find "$1" -name "summary_*.csv" -exec tail -1 {} \; 2>/dev/null \
        | awk -F, '{print $NF}' | grep -qE '^[0-9]+\.?[0-9]*$'
}

run_eval() {  # $1=目录 $2=说明
    local dir=$1 note=$2 ds name i wd csv score
    mkdir -p "$dir"
    [[ -f "${dir}/README.md" ]] || cat > "${dir}/README.md" <<EOF
# $(basename "$dir")
- 日期: 2026-11-12(夜间续跑)
- ${note}
- 结果: 待回填
EOF
    for ds in aime2024_gen_0_shot_chat_prompt.py gpqa_gen_0_shot_cot_chat_prompt.py; do
        name="${ds%.py}"
        [[ "${name}" == gpqa* ]] && REPEATS=${REPEATS_GPQA} || REPEATS=${REPEATS_AIME}
        for i in $(seq 1 "${REPEATS}"); do
            wd="${dir}/${name}/run${i}"
            if [[ -d "$wd" ]] && run_has_result "$wd"; then
                log "[$(basename $dir)] ${name} run${i} 已有成绩, 跳过"; continue
            fi
            if ! curl -s -m 3 http://127.0.0.1:9000/health -o /dev/null 2>&1; then
                log "[watch] 服务不健康, 重启"
                stop_server; start_server || { log "[error] 重启失败, 中止本阶段"; return 1; }
            fi
            mkdir -p "$wd"
            ( cd "${BENCH}" && ais_bench --models vllm_api_general_chat_glm \
                --datasets "${ds}" --mode all --dump-eval-details --merge-ds \
                -w "$wd" > "$wd/aisbench.log" 2>&1 )
            csv=$(ls "$wd"/*/summary/summary_*.csv 2>/dev/null | head -1)
            score=$(tail -1 "$csv" 2>/dev/null | awk -F, '{print $NF}')
            log "[$(basename $dir)] ${name} run${i} = ${score:-失败}"
            [[ -z "${score}" ]] && echo "$wd" >> "${dir}/.failed_runs"
        done
    done
    if [[ -f "${dir}/.failed_runs" ]]; then
        log "[retry] 补跑 $(wc -l < "${dir}/.failed_runs") 个失败遍"
        while read -r wd; do
            rm -rf "$wd"; mkdir -p "$wd"
            ds=$(basename "$(dirname "$wd")").py
            ( cd "${BENCH}" && ais_bench --models vllm_api_general_chat_glm \
                --datasets "${ds}" --mode all --dump-eval-details --merge-ds \
                -w "$wd" > "$wd/aisbench.log" 2>&1 )
            csv=$(ls "$wd"/*/summary/summary_*.csv 2>/dev/null | head -1)
            log "[retry] ${wd##*/run} = $(tail -1 "$csv" 2>/dev/null | awk -F, '{print $NF}')"
        done < "${dir}/.failed_runs"
        rm -f "${dir}/.failed_runs"
    fi
}

start_server() {
    log "起服务: IDX=${INDEXER_KV_DTYPE:-auto} KV=${KV_CACHE_DTYPE} H=${VLLM_ASCEND_KPOOL_HADAMARD_PRE:-0} M4=${VLLM_ASCEND_KPOOL_MXFP4_SIM:-0} M8=${VLLM_ASCEND_KPOOL_MXFP8_SIM:-0} OSCAR=${VLLM_ASCEND_KPOOL_OSCAR_ROTATION:-0}"
    # 8 卡空闲 gate(11-14 教训: 外部任务占卡时硬起服=相撞失败循环); 链内切换刚 stop 过必然空闲, 秒过无成本
    local gate=$((SECONDS + 7200)) stable=0
    while true; do
        free=$(npu-smi info 2>/dev/null | awk '/\/ 98304/{print $(NF-3)+0}' | awk '$0<10000' | wc -l)
        if [[ "$free" == "8" ]]; then
            stable=$((stable + 15)); (( stable >= 60 )) && break   # 空闲稳定 1 分钟才起
        else
            (( stable > 0 )) && log "[gate] 卡被占(空闲 ${free}/8), 等待释放..."
            stable=0
        fi
        (( SECONDS < gate )) || { log "[error] 等 8 卡空闲超时(2h), 本轮放弃(supervisor 重拉再试)"; return 1; }
        sleep 15
    done
    ( cd "${BASE}" && PORT=9000 setsid nohup bash serve_glm53flash.sh >> "${MAIN_LOG}" 2>&1 & )
    sleep 20
    local deadline=$((SECONDS + 1500)) grace=$((SECONDS + 120))
    until curl -s -m 3 http://127.0.0.1:9000/health -o /dev/null 2>&1; do
        if ! pgrep -f "[b]in/vllm serve" >/dev/null; then
            if (( SECONDS < grace )) && pgrep -f "serve_glm53flash.sh" >/dev/null; then sleep 5; continue; fi
            log "[error] 服务进程退出"; return 1
        fi
        (( SECONDS < deadline )) || { log "[error] 就绪超时"; return 1; }
        sleep 20
    done
    log "服务就绪"
}
stop_server() {
    bash "${BASE}/serve_glm53flash.sh" --stop >> "${MAIN_LOG}" 2>&1
    for p in $(pgrep -f "VLLM::"); do kill -9 $p 2>/dev/null; done
    sleep 60; log "服务已停"  # 11-13: 10→30s(AICPU page fault); 11-14: →60s(ERR02005 亦发于切换点, 设备回收再放宽)
}

# 看门狗: 30 分钟一检, 90 分钟无进展杀卡死 ais/重置服务
# 2026-11-13 修: ①40min→90min(gpqa 单遍 45-50min 且期间无日志, 40min 阈值误杀全部 gpqa)
#               ②summary 检查路径 *night3* 匹配不到 night2_exp* 目录(从未生效), 改目录通配
#               ③新增 predictions 活跃检查: ais_bench 持续写 jsonl 即视为有进展
(
    while sleep 1800; do
        [[ -f "${OUTROOT}/.night3_finished" ]] && exit 0
        latest=$(find "${OUTROOT}"/night2_exp* -name "summary_*.csv" -newermt '90 minutes ago' -print -quit 2>/dev/null)
        [[ -n "$latest" ]] && continue
        [[ -n $(find "${MAIN_LOG}" -newermt '90 minutes ago' 2>/dev/null) ]] && continue
        [[ -n $(find "${OUTROOT}"/night2_exp* -name '*.jsonl' -newermt '20 minutes ago' -print -quit 2>/dev/null) ]] && continue
        if pgrep -f "ais_bench --models" >/dev/null 2>&1; then
            echo "[$(date '+%m-%d %H:%M:%S')] [watchdog] 90min 无进展, 杀卡死 ais" >> "${MAIN_LOG}"
            pkill -9 -f "ais_bench --models"
        elif pgrep -f "VLLM::" >/dev/null 2>&1; then
            echo "[$(date '+%m-%d %H:%M:%S')] [watchdog] 90min 无进展, 重置服务" >> "${MAIN_LOG}"
            pkill -9 -f "VLLM::"
        fi
    done
) &
WATCHDOG_PID=$!

W=/mnt/share/w00608002/weight/GLM-5.3-Flash-FP8-w4a4
COMMON="MODEL_PATH=${W} QUANT_METHOD=ascend KV_CACHE_DTYPE=fp8"

# ---- 实验七o: MXFP4 伪量化 + OSCAR(校准集拟合, 先跑) ----
D7O="${OUTROOT}/night2_exp7o_w4a4_mxfp4_oscar"
if [[ ! -f "${D7O}/.done" ]]; then
    log "===== 七o: W4A4 + MXFP4 伪量化(C7.25) + OSCAR ====="
    export MODEL_PATH="${W}" QUANT_METHOD=ascend KV_CACHE_DTYPE=fp8
    export VLLM_ASCEND_KPOOL_MXFP4_SIM=1 VLLM_ASCEND_KPOOL_OSCAR_ROTATION=1
    export VLLM_ASCEND_KPOOL_OSCAR_CKPT="${OUTROOT}/oscar_calib/oscar_indexer_calib.pt"
    unset INDEXER_KV_DTYPE VLLM_ASCEND_KPOOL_HADAMARD_PRE VLLM_ASCEND_KPOOL_MXFP8_SIM VLLM_ASCEND_INDEXER_DUMP VLLM_ASCEND_INDEXER_DUMP_DIR VLLM_ASCEND_INDEXER_DUMP_MODE VLLM_ASCEND_INDEXER_DUMP_MAX_STEP
    if start_server; then
        run_eval "${D7O}" "W4A4-FP8, indexer bf16 + MXFP4 伪量化(C7.25) + OSCAR(校准集拟合, R→往返→Rᵀ), MLA KV fp8"
        stage_complete "${D7O}" && touch "${D7O}/.done" || log "[warn] 七o 未全部出分, 不标 done(supervisor 将重拉补齐)"
    fi
    stop_server
fi

# ---- 实验三h(续): C8 + Hadamard(已 done, 秒过) ----
D3H="${OUTROOT}/night2_exp3h_w4a4_c8_hadamard"
if [[ ! -f "${D3H}/.done" ]]; then
    log "===== 三h(续): W4A4 + C8 + Hadamard ====="
    export MODEL_PATH="${W}" QUANT_METHOD=ascend KV_CACHE_DTYPE=fp8 INDEXER_KV_DTYPE=fp8 VLLM_ASCEND_KPOOL_HADAMARD_PRE=1
    unset VLLM_ASCEND_KPOOL_MXFP4_SIM VLLM_ASCEND_KPOOL_MXFP8_SIM VLLM_ASCEND_KPOOL_OSCAR_ROTATION
    if start_server; then
        run_eval "${D3H}" "W4A4-FP8, indexer C8 + Hadamard(旋转域 q/k/tail), MLA KV fp8"
        stage_complete "${D3H}" && touch "${D3H}/.done" || log "[warn] 三h 未全部出分, 不标 done(supervisor 将重拉补齐)"
    fi
    stop_server
fi

# ---- 实验七: MXFP4 伪量化 + Hadamard(gpqa run1/2 已有, 补 run3) ----
D7="${OUTROOT}/night2_exp7_w4a4_mxfp4_hadamard"
if [[ ! -f "${D7}/.done" ]]; then
    log "===== 七: W4A4 + MXFP4 伪量化(C7.25) + Hadamard ====="
    export MODEL_PATH="${W}" QUANT_METHOD=ascend KV_CACHE_DTYPE=fp8 VLLM_ASCEND_KPOOL_MXFP4_SIM=1 VLLM_ASCEND_KPOOL_HADAMARD_PRE=1
    unset INDEXER_KV_DTYPE VLLM_ASCEND_KPOOL_MXFP8_SIM VLLM_ASCEND_KPOOL_OSCAR_ROTATION VLLM_ASCEND_KPOOL_OSCAR_CKPT VLLM_ASCEND_INDEXER_DUMP VLLM_ASCEND_INDEXER_DUMP_DIR VLLM_ASCEND_INDEXER_DUMP_MODE VLLM_ASCEND_INDEXER_DUMP_MAX_STEP
    if start_server; then
        run_eval "${D7}" "W4A4-FP8, indexer bf16 + MXFP4 伪量化(C7.25) + Hadamard(H→往返→H), MLA KV fp8"
        stage_complete "${D7}" && touch "${D7}/.done" || log "[warn] 七 未全部出分, 不标 done"
    fi
    stop_server
fi

# ---- 实验六: 已取消(2026-11-14 用户指令: 不补 run3, gpqa 以 n=2 定稿 88.89/90.91) ----
# D6="${OUTROOT}/night2_exp6_w4a4_mxfp8"

# ---- 七o2A: W4A4 权重激活 dump + OSCAR 拟合(七o2 的前置) ----
DUMPW="${OUTROOT}/indexer_dump_calib_w4a4"
CKPT_W4A4="${OUTROOT}/oscar_calib/oscar_indexer_w4a4.pt"
CALIB=/mnt/share/w00608002/weight/calib_corpus_R9tau.jsonl
if [[ ! -f "${CKPT_W4A4}" && ! -f "${OUTROOT}/night2_exp7o2_w4a4_mxfp4_oscar_w4a4fit/.done" ]]; then
    log "===== 七o2A: W4A4 权重激活 dump(校准集 904 条, indexer bf16) ====="
    export MODEL_PATH="${W}" QUANT_METHOD=ascend KV_CACHE_DTYPE=fp8
    unset INDEXER_KV_DTYPE VLLM_ASCEND_KPOOL_HADAMARD_PRE VLLM_ASCEND_KPOOL_MXFP4_SIM \ VLLM_ASCEND_INDEXER_DUMP VLLM_ASCEND_INDEXER_DUMP_DIR VLLM_ASCEND_INDEXER_DUMP_MODE VLLM_ASCEND_INDEXER_DUMP_MAX_STEP
          VLLM_ASCEND_KPOOL_MXFP8_SIM VLLM_ASCEND_KPOOL_OSCAR_ROTATION VLLM_ASCEND_KPOOL_OSCAR_CKPT
    mkdir -p "${DUMPW}"
    round=$(ls -d "${DUMPW}"/round* 2>/dev/null | wc -l)
    export VLLM_ASCEND_INDEXER_DUMP=1 VLLM_ASCEND_INDEXER_DUMP_DIR="${DUMPW}/round$(printf '%02d' "${round}")" \
           VLLM_ASCEND_INDEXER_DUMP_MODE=raw VLLM_ASCEND_INDEXER_DUMP_MAX_STEP=2200
    mkdir -p "${VLLM_ASCEND_INDEXER_DUMP_DIR}"
    PROG="${DUMPW}/progress.json"
    dump_ok=0
    if start_server; then
        idx=0; [[ -f "${PROG}" ]] && idx=$(python3 -c "import json;print(json.load(open('${PROG}'))['sent_upto']+1)" 2>/dev/null || echo 0)
        total=$(python3 -c "print(len(open('${CALIB}').readlines()))")
        log "灌校准集: 第 ${idx}..$((total-1)) 条(round ${round}) → ${VLLM_ASCEND_INDEXER_DUMP_DIR}"
        python3 - "${CALIB}" "${idx}" "${PROG}" <<'PY' 2>&1 | tee -a "${MAIN_LOG}" | grep -E "progress|DONE|failed"
import json, sys, time, urllib.request
calib, start, prog_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
lines = open(calib).readlines()   # 只按 \n 切(JSON 内 u2028 会让 splitlines 产生非法行, 已踩坑)
t0 = time.time()
fail = 0
for i in range(start, len(lines)):
    text = json.loads(lines[i])["text"]
    body = json.dumps({"model": "glm", "prompt": text, "max_tokens": 1,
                       "temperature": 0.0}).encode()
    req = urllib.request.Request("http://127.0.0.1:9000/v1/completions", body,
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            json.load(resp)
    except Exception as e:
        fail += 1
        print(f"[{i}] failed: {e}", flush=True)
        if fail >= 5:
            print("连续失败过多, 退出由主链重启服务续跑", flush=True)
            sys.exit(1)
    with open(prog_path, 'w') as f:
        json.dump({"sent_upto": i}, f)
    n = i - start + 1
    if n % 100 == 0:
        print(f"progress {i+1}/{len(lines)} ({n/(time.time()-t0):.1f} 条/s)", flush=True)
print(f"DUMP_DONE: 至第 {len(lines)-1} 条, 共 {len(lines)-start} 条, 失败 {fail}", flush=True)
PY
        # $? 是管道末端 grep 的退出码, python 真实结果须取 PIPESTATUS[0](11-14 复盘:
        # 曾因 $? 恒 0 在 dump 中断后仍拿半截数据拟合 → 错误矩阵)
        if (( PIPESTATUS[0] == 0 )); then
            log "七o2A dump 发送完成"; dump_ok=1
        else
            log "[error] 七o2A dump 中断(断点保留在 progress.json), 本轮不拟合, supervisor 重拉续跑"
            dump_ok=0
        fi
        stop_server
    fi
    if (( dump_ok )); then
    # ---- flat 化 + OSCAR 拟合(坏 npz 由拟合工具自动跳过) ----
    log "七o2A: flat 化 + OSCAR 拟合"
    python3 - "${DUMPW}" <<'PY' >> "${MAIN_LOG}" 2>&1
import glob, os, re, sys
root = sys.argv[1]
flat = root + "_flat"
os.makedirs(flat, exist_ok=True)
offset = 0; n = 0
for d in sorted(glob.glob(f"{root}/round*")):
    mx = -1
    for f in glob.glob(f"{d}/*_step*.npz"):
        m = re.search(r"_step(\d+)\.npz$", f)
        if m: mx = max(mx, int(m.group(1)))
        base = os.path.basename(f)
        if offset:
            base = re.sub(r"_step(\d+)\.npz$", lambda mm: f"_step{int(mm.group(1))+offset}.npz", base)
        dst = f"{flat}/{base}"
        if not os.path.exists(dst):
            os.symlink(f, dst); n += 1
    if mx >= 0: offset += mx + 1
print(f"flat: {flat} 新增 {n} 链接(next offset={offset})")
PY
    python3 /mnt/share/w00608002/work/vllm-ascend/tools/glm53flash/oscar_indexer_rotation/oscar_indexer_rotation.py \
        "${DUMPW}_flat" "${CKPT_W4A4}" >> "${MAIN_LOG}" 2>&1 \
        && log "七o2A 拟合完成: ${CKPT_W4A4}" \
        || { log "[error] 七o2A 拟合失败(不标 done, supervisor 重拉重试)"; }
    fi   # dump_ok 才 flat+拟合(防半截数据)
fi

# ---- 七o2B: W4A4 + MXFP4 伪量化 + OSCAR(W4A4 激活拟合) 评测 ----
D7O2="${OUTROOT}/night2_exp7o2_w4a4_mxfp4_oscar_w4a4fit"
if [[ ! -f "${D7O2}/.done" ]]; then
    if [[ ! -f "${CKPT_W4A4}" ]]; then
        log "[skip] 七o2B: ckpt 不存在(七o2A 未完成), 留待 supervisor 重拉"
    else
        log "===== 七o2B: W4A4 + MXFP4 伪量化 + OSCAR(W4A4 拟合) ====="
        export MODEL_PATH="${W}" QUANT_METHOD=ascend KV_CACHE_DTYPE=fp8
        export VLLM_ASCEND_KPOOL_MXFP4_SIM=1 VLLM_ASCEND_KPOOL_OSCAR_ROTATION=1 VLLM_ASCEND_KPOOL_OSCAR_CKPT="${CKPT_W4A4}"
        unset INDEXER_KV_DTYPE VLLM_ASCEND_KPOOL_HADAMARD_PRE VLLM_ASCEND_KPOOL_MXFP8_SIM VLLM_ASCEND_INDEXER_DUMP VLLM_ASCEND_INDEXER_DUMP_DIR VLLM_ASCEND_INDEXER_DUMP_MODE VLLM_ASCEND_INDEXER_DUMP_MAX_STEP
        if start_server; then
            run_eval "${D7O2}" "W4A4-FP8, indexer bf16 + MXFP4 伪量化(C7.25) + OSCAR(W4A4 自身激活拟合), MLA KV fp8"
            stage_complete "${D7O2}" && touch "${D7O2}/.done" || log "[warn] 七o2 未全部出分, 不标 done"
        fi
        stop_server
    fi
fi

# ---- 八A: 官方 FP8 → W4A4-KDA8 量化(msmodelslim, kda8 yaml, 单卡数小时) ----
W8=/mnt/share/w00608002/weight/GLM-5.3-Flash-FP8-w4a4-kda8
YAML_KDA8=/mnt/share/w00608002/work/msmodelslim/GLM53-flash-w4a4-kda8.yaml
if [[ ! -f "${W8}/config.json" ]]; then
    log "===== 八A: 官方 FP8 → W4A4-KDA8 量化(kda8 yaml, 单卡逐层校准) ====="
    # 选第一张空闲卡(HBM<10GB); 无空闲卡则等待(11-14 教训: 默认卡硬上会撞外部任务), 2h 上限
    qgate=$((SECONDS + 7200))
    while true; do
        QDEV=$(npu-smi info 2>/dev/null | awk '/\/ 98304/{print $(NF-3)+0}' | awk '$0<10000{print NR-1; exit}')
        [[ -n "${QDEV}" ]] && break
        if (( SECONDS >= qgate )); then log "[error] 等空闲卡超时(2h), 八A 本轮放弃(八B/C 因缺权重自动跳过)"; break; fi
        log "[gate] 无空闲卡(量化需 1 卡), 等待..."; sleep 60
    done
    if [[ -n "${QDEV}" ]]; then
    log "量化用卡: ${QDEV}"
    mkdir -p "${W8}/log"
    export ASCEND_RT_VISIBLE_DEVICES="${QDEV}" ASCEND_SLOG_PRINT_TO_STDOUT=0 \
           ASCEND_GLOBAL_LOG_LEVEL=2 ASCEND_HOST_LOG_FILE_NUM=1000 \
           ASCEND_PROCESS_LOG_PATH="${W8}/log/" HF_DATASETS_OFFLINE=1
    CONF_DIR=$(dirname "${YAML_KDA8}")
    [[ -f "${CONF_DIR}/mix_calib.jsonl" ]] || ln -sf /mnt/share/w00608002/work/msmodelslim/example/common/mix_calib.jsonl "${CONF_DIR}/mix_calib.jsonl"
    ( cd /mnt/share/w00608002/work/msmodelslim && msmodelslim quant \
        --model_path /mnt/share/w00936111/weights/GLM-5.3-Flash \
        --save_path "${W8}" --device npu --model_type GLM-5.3-Flash \
        --trust_remote_code True --config_path "${YAML_KDA8}" ) >> "${MAIN_LOG}" 2>&1 \
        && log "八A 量化完成: ${W8}" \
        || log "[error] 八A 量化失败(supervisor 重拉重试; 断点由 msmodelslim 自身 save_path 续)"
    unset ASCEND_RT_VISIBLE_DEVICES
    fi   # QDEV 非空才量化
fi

# ---- 八B: KDA8 权重激活 dump + OSCAR 拟合 ----
DUMPK="${OUTROOT}/indexer_dump_calib_kda8"
CKPT_KDA8="${OUTROOT}/oscar_calib/oscar_indexer_kda8.pt"
if [[ ! -f "${CKPT_KDA8}" && ! -f "${OUTROOT}/night2_exp8_w4a4kda8_mxfp4_oscar/.done" && -f "${W8}/config.json" ]]; then
    log "===== 八B: W4A4-KDA8 权重激活 dump(校准集 904 条, indexer bf16) ====="
    export MODEL_PATH="${W8}" QUANT_METHOD=ascend KV_CACHE_DTYPE=fp8
    unset INDEXER_KV_DTYPE VLLM_ASCEND_KPOOL_HADAMARD_PRE VLLM_ASCEND_KPOOL_MXFP4_SIM \ VLLM_ASCEND_INDEXER_DUMP VLLM_ASCEND_INDEXER_DUMP_DIR VLLM_ASCEND_INDEXER_DUMP_MODE VLLM_ASCEND_INDEXER_DUMP_MAX_STEP
          VLLM_ASCEND_KPOOL_MXFP8_SIM VLLM_ASCEND_KPOOL_OSCAR_ROTATION VLLM_ASCEND_KPOOL_OSCAR_CKPT
    mkdir -p "${DUMPK}"
    round=$(ls -d "${DUMPK}"/round* 2>/dev/null | wc -l)
    export VLLM_ASCEND_INDEXER_DUMP=1 VLLM_ASCEND_INDEXER_DUMP_DIR="${DUMPK}/round$(printf '%02d' "${round}")" \
           VLLM_ASCEND_INDEXER_DUMP_MODE=raw VLLM_ASCEND_INDEXER_DUMP_MAX_STEP=2200
    mkdir -p "${VLLM_ASCEND_INDEXER_DUMP_DIR}"
    PROG="${DUMPK}/progress.json"
    dump_ok=0
    if start_server; then
        idx=0; [[ -f "${PROG}" ]] && idx=$(python3 -c "import json;print(json.load(open('${PROG}'))['sent_upto']+1)" 2>/dev/null || echo 0)
        log "灌校准集: 第 ${idx} 条起(round ${round}) → ${VLLM_ASCEND_INDEXER_DUMP_DIR}"
        python3 - "${CALIB}" "${idx}" "${PROG}" <<'PY' 2>&1 | tee -a "${MAIN_LOG}" | grep -E "progress|DONE|failed"
import json, sys, time, urllib.request
calib, start, prog_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
lines = open(calib).readlines()
t0 = time.time(); fail = 0
for i in range(start, len(lines)):
    text = json.loads(lines[i])["text"]
    body = json.dumps({"model": "glm", "prompt": text, "max_tokens": 1,
                       "temperature": 0.0}).encode()
    req = urllib.request.Request("http://127.0.0.1:9000/v1/completions", body,
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            json.load(resp)
    except Exception as e:
        fail += 1; print(f"[{i}] failed: {e}", flush=True)
        if fail >= 5: sys.exit(1)
    with open(prog_path, 'w') as f: json.dump({"sent_upto": i}, f)
    n = i - start + 1
    if n % 100 == 0: print(f"progress {i+1}/{len(lines)} ({n/(time.time()-t0):.1f} 条/s)", flush=True)
print(f"DUMP_DONE: 共 {len(lines)-start} 条, 失败 {fail}", flush=True)
PY
        if (( PIPESTATUS[0] == 0 )); then
            log "八B dump 发送完成"; dump_ok=1
        else
            log "[error] 八B dump 中断(断点保留), 本轮不拟合, supervisor 重拉续跑"
            dump_ok=0
        fi
        stop_server
    fi
    if (( dump_ok )); then
    python3 - "${DUMPK}" <<'PY' >> "${MAIN_LOG}" 2>&1
import glob, os, re, sys
root = sys.argv[1]; flat = root + "_flat"
os.makedirs(flat, exist_ok=True)
offset = 0; n = 0
for d in sorted(glob.glob(f"{root}/round*")):
    mx = -1
    for f in glob.glob(f"{d}/*_step*.npz"):
        m = re.search(r"_step(\d+)\.npz$", f)
        if m: mx = max(mx, int(m.group(1)))
        base = os.path.basename(f)
        if offset:
            base = re.sub(r"_step(\d+)\.npz$", lambda mm: f"_step{int(mm.group(1))+offset}.npz", base)
        dst = f"{flat}/{base}"
        if not os.path.exists(dst): os.symlink(f, dst); n += 1
    if mx >= 0: offset += mx + 1
print(f"flat: {flat} 新增 {n} 链接")
PY
    python3 /mnt/share/w00608002/work/vllm-ascend/tools/glm53flash/oscar_indexer_rotation/oscar_indexer_rotation.py \
        "${DUMPK}_flat" "${CKPT_KDA8}" >> "${MAIN_LOG}" 2>&1 \
        && log "八B 拟合完成: ${CKPT_KDA8}" || log "[error] 八B 拟合失败"
    fi   # dump_ok 才 flat+拟合(防半截数据)
fi

# ---- 八C: 实验 八 = W4A4-KDA8 + MXFP4 伪量化 + OSCAR(自身拟合) 评测 ----
D8="${OUTROOT}/night2_exp8_w4a4kda8_mxfp4_oscar"
if [[ ! -f "${D8}/.done" ]]; then
    if [[ ! -f "${CKPT_KDA8}" ]]; then
        log "[skip] 八C: ckpt 不存在(八A/B 未完成), 留待 supervisor 重拉"
    else
        log "===== 八C: W4A4-KDA8 + MXFP4 伪量化 + OSCAR(KDA8 自身拟合) ====="
        export MODEL_PATH="${W8}" QUANT_METHOD=ascend KV_CACHE_DTYPE=fp8
        export VLLM_ASCEND_KPOOL_MXFP4_SIM=1 VLLM_ASCEND_KPOOL_OSCAR_ROTATION=1 VLLM_ASCEND_KPOOL_OSCAR_CKPT="${CKPT_KDA8}"
        unset INDEXER_KV_DTYPE VLLM_ASCEND_KPOOL_HADAMARD_PRE VLLM_ASCEND_KPOOL_MXFP8_SIM VLLM_ASCEND_INDEXER_DUMP VLLM_ASCEND_INDEXER_DUMP_DIR VLLM_ASCEND_INDEXER_DUMP_MODE VLLM_ASCEND_INDEXER_DUMP_MAX_STEP
        if start_server; then
            run_eval "${D8}" "W4A4-KDA8(FP8 量化链), indexer bf16 + MXFP4 伪量化(C7.25) + OSCAR(KDA8 自身激活拟合), MLA KV fp8"
            stage_complete "${D8}" && touch "${D8}/.done" || log "[warn] 八 未全部出分, 不标 done"
        fi
        stop_server
    fi
fi

# ---- 实验四: W4A4-FP8 + MXFP4 伪量化(无旋转对照, 与七/七o 唯一差=旋转) ----
D4="${OUTROOT}/night2_exp4_w4a4_mxfp4"
if [[ ! -f "${D4}/.done" ]]; then
    log "===== 四: W4A4 + MXFP4 伪量化(无旋转) ====="
    export MODEL_PATH="${W}" QUANT_METHOD=ascend KV_CACHE_DTYPE=fp8 VLLM_ASCEND_KPOOL_MXFP4_SIM=1
    unset INDEXER_KV_DTYPE VLLM_ASCEND_KPOOL_HADAMARD_PRE VLLM_ASCEND_KPOOL_MXFP8_SIM \
          VLLM_ASCEND_KPOOL_OSCAR_ROTATION VLLM_ASCEND_KPOOL_OSCAR_CKPT \
          VLLM_ASCEND_INDEXER_DUMP VLLM_ASCEND_INDEXER_DUMP_DIR VLLM_ASCEND_INDEXER_DUMP_MODE VLLM_ASCEND_INDEXER_DUMP_MAX_STEP
    if start_server; then
        run_eval "${D4}" "W4A4-FP8, indexer bf16 + MXFP4 伪量化(C7.25) 无旋转(与七/七o 唯一差), MLA KV fp8"
        stage_complete "${D4}" && touch "${D4}/.done" || log "[warn] 四 未全部出分, 不标 done"
    fi
    stop_server
fi

[[ -n "${WATCHDOG_PID}" ]] && kill "${WATCHDOG_PID}" 2>/dev/null
if [[ -f "${D3H}/.done" && -f "${D7O}/.done" && -f "${D7}/.done" && -f "${D7O2}/.done" && -f "${D8}/.done" && -f "${D4}/.done" ]]; then
    touch "${OUTROOT}/.night3_finished"
    log "===== night3 全部结束(六已取消, 现行六阶段均完整出分) ====="
else
    log "===== 本轮未全部完成(不标 finished), supervisor 将重拉续跑 ====="
fi
