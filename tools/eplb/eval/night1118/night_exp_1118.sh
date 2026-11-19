#!/usr/bin/env bash
# 夜间实验主编排(2026-11-18): E3收尾(attn对齐) → gate(00:00+8卡) → E1 FP8 → E2 W2A8 → E4 混搭
# 机制(借鉴 glm53flash/night3): 阶段断点秒过 / 评测按 summary 成绩断点续跑 / 失败遍补跑 /
#   90min watchdog / 8卡稳定gate / 停服60s缓冲(HCCL脏状态) / E1起服OOM自动降档一次
# 标记: $OUT/.e1_done .e2_done .e4_done .e3_failed .finished .halt(熔断,人工排障后rm)
set -uo pipefail

REPO=/mnt/share/w00608002/work/eplb/vllm-ascend
BENCH=/mnt/share/w00608002/work/benchmark   # ais_bench 配置/数据集全按 cwd 相对解析(2026-11-19 第5坑:
                                             # cd 到 ais_bench 仓库根会 fallback site-packages 配置+找不到数据)
W=/mnt/share/w00608002/weight
OUT=/mnt/share/w00608002/work/eplb/output/night_exp_20261118
MAIN_LOG=$OUT/main.log
MIDNIGHT=$(date -d '2026-11-19 00:00' +%s)
M_FP8=$W/DeepSeek-V4-Flash-0731
M_NOPERM=$W/DeepSeek-V4-Flash-0731-32x32-noperm-hot5050125-identity-v3-export-20261103
M_REPACK=$W/DeepSeek-V4-Flash-0731-vq2a8-mixed-gs32-repack
M_W4A4=$W/DeepSeek-V4-Flash-0731-32x32-noperm-w4a4c8
M_ATTND=$W/DeepSeek-V4-Flash-0731-32x32-noperm-w4a4c8-attnD
REPEATS_AIME=3 REPEATS_GPQA=3

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$MAIN_LOG"; }
mkdir -p "$OUT"

# ---------- 断点/成绩判定 ----------
run_has_result() {  # $1=run目录: summary_*.csv 尾列为数值即有效
    find "$1" -name "summary_*.csv" -exec tail -1 {} \; 2>/dev/null \
        | awk -F, '{print $NF}' | grep -qE '^[0-9]+\.?[0-9]*$'
}
stage_complete() {  # $1=E*_fp8 目录: aime×3 + gpqa×3 全部有效
    local dir=$1 ds n r ok=0 total=0
    for ds in aime2024_gen_0_shot_chat_prompt gpqa_gen_0_shot_cot_chat_prompt; do
        n=3
        for r in $(seq 1 $n); do
            total=$((total+1))
            run_has_result "$dir/$ds/run$r" && ok=$((ok+1))
        done
    done
    (( ok == total ))
}

# ---------- 服务生命周期 ----------
serve_pid_of() { pgrep -f "run_pd_w2a8_eager_graph.sh" | head -1; }
stop_server() {
    local p
    for p in $(pgrep -f "run_pd_w2a8_eager_graph.sh"); do kill -TERM "$p" 2>/dev/null; done
    sleep 15
    for p in $(ps -eo pid,cmd | awk '/vllm[.]entrypoints|load_balance_proxy_server/ && !/awk/ {print $1}'); do kill -9 "$p" 2>/dev/null; done
    for p in $(npu-smi info 2>/dev/null | sed -n '/Process id/,$p' | awk -F'|' 'NF>4 {gsub(/ /,"",$6); if ($6+0>1000) print $6}'); do
        kill -9 "$p" 2>/dev/null; done
    sleep 60   # 11-13/11-14 教训: HCCL/AICPU 设备回收需缓冲, 太快 ERR02005
    log "服务已停清场"
}
gate_cards() {  # 8 卡空闲稳定 60s(deadline 2h)
    local deadline=$((SECONDS + 7200)) stable=0 free
    while :; do
        free=$(npu-smi info 2>/dev/null | awk '/\/ 98304/{print $(NF-3)+0}' | awk '$0<10000' | wc -l)
        if [[ "$free" == "8" ]]; then stable=$((stable+15)); (( stable >= 60 )) && return 0
        else
            (( stable > 0 )) && log "[gate] 卡被占(空闲 ${free}/8), 等待..."
            stable=0
        fi
        (( SECONDS < deadline )) || { log "[error] 等 8 卡空闲超时(2h), 本轮放弃"; return 1; }
        sleep 15
    done
}
start_server() {  # $1=MODEL_P $2=MODEL_D $3=serve日志  $KV_BYTES/$MAX_LEN 可选降档
    local envs=(MODEL_P="$1" MODEL_D="$2" QUANT=auto)
    [[ -n "${KV_BYTES:-}" ]] && envs+=("KV_BYTES=$KV_BYTES")
    [[ -n "${MAX_LEN:-}" ]] && envs+=("MAX_LEN=$MAX_LEN")
    # NOTE: 参数展开产物不会被 bash 识别为赋值前缀(2026-11-19 教训:
    # ${KV_BYTES:+KV_BYTES=$KV_BYTES} 曾被当命令执行 → command not found), 必须用 env 显式传
    ( cd "$REPO" && env "${envs[@]}" setsid nohup bash tools/eplb/serve/run_pd_w2a8_eager_graph.sh \
        --with-proxy >> "$3" 2>&1 & )
    local deadline=$((SECONDS + 2400)) grace=$((SECONDS + 180))
    until curl -s -m 3 http://127.0.0.1:9001/v1/models >/dev/null 2>&1 && \
          curl -s -m 3 http://127.0.0.1:9002/v1/models >/dev/null 2>&1; do
        if ! pgrep -f "vllm[.]entrypoints" >/dev/null; then
            if (( SECONDS < grace )) && pgrep -f "run_pd_w2a8_eager_graph" >/dev/null; then sleep 10; continue; fi
            log "[error] serve 进程退出(见 $3 尾部)"; tail -5 "$3" | tee -a "$MAIN_LOG"; return 1
        fi
        (( SECONDS < deadline )) || { log "[error] serve 就绪超时(40min)"; return 1; }
        sleep 20
    done
    log "serve 就绪 ($1 × $2)"
}
smoke() {  # $1=标签: 输出判定文本; 返回0=通过(含1081且无乱码)
    local label=$1 resp text i
    for i in 1 2 3; do
        resp=$(curl -s -m 180 http://127.0.0.1:8000/v1/chat/completions \
            -H 'Content-Type: application/json' \
            -d '{"model":"dsv","messages":[{"role":"user","content":"请先做一段自我介绍，然后计算 23×47，一步一步算，最后单独一行给出答案数字。"}],"max_tokens":4096,"temperature":0.6}' || true)
        text=$(python3 -c "
import json,sys
try:
    r=json.loads(sys.stdin.read()); m=r['choices'][0]['message']
    print((m.get('reasoning_content') or '')+(m.get('content') or ''))
except Exception: print('')" <<<"$resp" 2>/dev/null)
        [[ "$text" == *1081* ]] && { log "[smoke:$label] ✅ 通过(len=${#text})"; return 0; }
        log "[smoke:$label] 第${i}次未含1081(len=${#text}): $(echo "$text" | head -c 120)"
        sleep 20
    done
    return 1
}

# ---------- 评测 ----------
run_eval() {  # $1=E*目录  $2=说明
    local dir=$1 ds name i wd csv score REPEATS
    mkdir -p "$dir"
    [[ -f "$dir/README.md" ]] || echo "# $(basename "$dir") — $2 — $(date '+%F %R')" > "$dir/README.md"
    for ds in aime2024_gen_0_shot_chat_prompt.py gpqa_gen_0_shot_cot_chat_prompt.py; do
        name="${ds%.py}"
        [[ "$name" == gpqa* ]] && REPEATS=$REPEATS_GPQA || REPEATS=$REPEATS_AIME
        for i in $(seq 1 "$REPEATS"); do
            wd="$dir/$name/run$i"
            if [[ -d "$wd" ]] && run_has_result "$wd"; then log "[$(basename $dir)] ${name#*prompt} run$i 已有成绩, 秒过"; continue; fi
            # 评测前健康检查(P/D 双侧), 不健康重启(断点保证已完成遍不丢)
            if ! curl -s -m 3 http://127.0.0.1:9001/v1/models >/dev/null 2>&1 || \
               ! curl -s -m 3 http://127.0.0.1:9002/v1/models >/dev/null 2>&1; then
                log "[watch] 服务不健康 → 重启"; stop_server; start_server "${CUR_MODEL_P:?}" "${CUR_MODEL_D:?}" "$CUR_SERVE_LOG" || return 1
            fi
            rm -rf "$wd"; mkdir -p "$wd"
            ( cd "$BENCH" && ais_bench --models vllm_api_general_chat --datasets "$ds" \
                --mode all --dump-eval-details --merge-ds -w "$wd" > "$wd/aisbench.log" 2>&1 )
            csv=$(ls "$wd"/*/summary/summary_*.csv 2>/dev/null | head -1)
            score=$(tail -1 "$csv" 2>/dev/null | awk -F, '{print $NF}')
            log "[$(basename $dir)] ${name} run$i = ${score:-失败}"
            [[ -z "${score}" ]] && echo "$wd" >> "$dir/.failed_runs"
        done
    done
    if [[ -f "$dir/.failed_runs" ]]; then  # 失败遍补跑一轮
        log "[retry] 补跑 $(wc -l < "$dir/.failed_runs") 个失败遍"
        while read -r wd; do
            rm -rf "$wd"; mkdir -p "$wd"
            ds="$(basename "$(dirname "$wd")").py"
            ( cd "$BENCH" && ais_bench --models vllm_api_general_chat --datasets "$ds" \
                --mode all --dump-eval-details --merge-ds -w "$wd" > "$wd/aisbench.log" 2>&1 )
            csv=$(ls "$wd"/*/summary/summary_*.csv 2>/dev/null | head -1)
            log "[retry] ${wd%/run*} run${wd##*run} = $(tail -1 "$csv" 2>/dev/null | awk -F, '{print $NF}')"
        done < "$dir/.failed_runs"
        rm -f "$dir/.failed_runs"
    fi
}

# ---------- watchdog: 90min 无进展杀卡死 ais / 重置服务(30min 一检) ----------
(
    while sleep 1800; do
        [[ -f "$OUT/.finished" || -f "$OUT/.halt" ]] && exit 0
        [[ -n $(find "$OUT" -name "summary_*.csv" -newermt '90 minutes ago' -print -quit 2>/dev/null) ]] && continue
        [[ -n $(find "$MAIN_LOG" -newermt '90 minutes ago' 2>/dev/null) ]] && continue
        [[ -n $(find "$OUT" -name '*.jsonl' -newermt '20 minutes ago' -print -quit 2>/dev/null) ]] && continue
        if pgrep -f "ais_bench --models" >/dev/null 2>&1; then
            echo "[$(date '+%m-%d %H:%M:%S')] [watchdog] 90min 无进展, 杀卡死 ais" >> "$MAIN_LOG"
            pkill -9 -f "ais_bench --models"
        elif pgrep -f "vllm[.]entrypoints" >/dev/null 2>&1; then
            echo "[$(date '+%m-%d %H:%M:%S')] [watchdog] 90min 无进展, 重置服务" >> "$MAIN_LOG"
            pkill -9 -f "vllm[.]entrypoints"; pkill -9 -f "run_pd_w2a8_eager_graph"
        fi
    done
) &
WATCHDOG_PID=$!
trap 'kill $WATCHDOG_PID 2>/dev/null' EXIT

# ================= 阶段 0: E3 收尾(等量化 → attn 对齐; CPU 任务可先跑) =================
if [[ ! -d "$M_ATTND" && ! -f "$OUT/.e3_failed" ]]; then
    log "E3: 等待量化进程结束..."
    while pgrep -f "msmodelslim quant" >/dev/null 2>&1; do sleep 60; done
    sleep 30
    if [[ ! -f "$M_W4A4/quant_model_weights.safetensors.index.json" ]]; then
        log "[error] E3 量化产物缺失($M_W4A4), E4 将跳过"; touch "$OUT/.e3_failed"
    else
        log "E3: 量化产物就绪 → attn 对齐(hardlink D=repack)"
        ( cd "$REPO" && python3 -I tools/eplb/convert/make_attnD_vq2a8.py \
            --src "$M_W4A4" --out "$M_ATTND" > "$OUT/e3_attnd.log" 2>&1 )
        if [[ -f "$M_ATTND/model.safetensors.index.json" ]]; then
            log "E3 ✅ attnD 产物就绪: $M_ATTND"
        else
            log "[error] E3 attn 对齐失败(见 e3_attnd.log), E4 将跳过"; tail -5 "$OUT/e3_attnd.log" | tee -a "$MAIN_LOG"; touch "$OUT/.e3_failed"
        fi
    fi
fi

# ================= 阶段 1: E1 FP8 基线 =================
run_stage() {  # $1=阶段名 $2=MODEL_P $3=MODEL_D $4=done标记 $5=严格smoke(1=E4)
    local st=$1 MP=$2 MD=$3 DONE=$4 STRICT=$5
    local SL="$OUT/serve_${st}.log"
    [[ -f "$OUT/$DONE" ]] && { log "$st 已 done, 秒过"; return 0; }
    log "$st: gate(00:00 + 8卡)"
    while (( $(date +%s) < MIDNIGHT )); do sleep 60; done
    gate_cards || return 1
    stop_server
    if ! start_server "$MP" "$MD" "$SL"; then
        log "$st 首次起服失败 → 降档重试(KV=8G, len=65536)"
        stop_server
        KV_BYTES=8589934592 MAX_LEN=65536 start_server "$MP" "$MD" "$SL" || { log "[error] $st 起服失败, 本轮放弃"; return 1; }
    fi
    CUR_MODEL_P=$MP CUR_MODEL_D=$MD CUR_SERVE_LOG=$SL
    if ! smoke "$st"; then
        if [[ "$STRICT" == "1" ]]; then
            log "[halt] $st smoke 未过(混搭乱码?) → 记录并熔断, 人工分析后 rm $OUT/.halt 续跑"
            curl -s -m 60 http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
                -d '{"model":"dsv","messages":[{"role":"user","content":"你好，请自我介绍"}],"max_tokens":512,"temperature":0.6}' \
                > "$OUT/e4_smoke_garbled_sample.json" 2>/dev/null
            touch "$OUT/.halt"
            return 1
        fi
        log "[warn] $st smoke 1081 未过但继续(非门禁组)"
    fi
    run_eval "$OUT/${st}" "$st · P=$MP D=$MD"
    stop_server
    if stage_complete "$OUT/${st}"; then touch "$OUT/$DONE"; log "$st ✅ 全部完成"; else log "[warn] $st 仍有缺遍(supervisor 重拉补)"; return 1; fi
}

run_stage E1_fp8  "$M_FP8"    "$M_FP8"    .e1_done 0 || true
run_stage E2_w2a8 "$M_NOPERM" "$M_NOPERM" .e2_done 0 || true
if [[ -d "$M_ATTND" ]]; then
    run_stage E4_mixed "$M_ATTND" "$M_REPACK" .e4_done 1 || true
else
    log "E4 跳过(E3 产物缺失, 见 .e3_failed)"
fi

# ---- 链尾: done 标记齐全才算真完成(E1/E2 缺 → 非零退出, supervisor 重拉补跑) ----
ALLDONE=1
[[ -f "$OUT/.e1_done" ]] || ALLDONE=0
[[ -f "$OUT/.e2_done" ]] || ALLDONE=0
[[ -f "$OUT/.e4_done" || -f "$OUT/.e3_failed" ]] || ALLDONE=0
if (( ALLDONE )); then
    touch "$OUT/.finished"
    log "夜间链全部结束 ✅"
else
    log "仍有阶段未完成(done 标记缺) → 本轮退出, supervisor 将重拉补跑"
    exit 1
fi
