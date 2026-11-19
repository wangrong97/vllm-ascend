#!/usr/bin/env bash
# evalscope 顺序链(2026-11-19 按用户指令切换评测方式):
# 等 E2 evalscope → 停服 → E1 FP8(evalscope 重跑) → 停服 → E4(smoke 门禁 + evalscope) → 全链完成
# 断点标记: $OUT/.esc_e1_done .esc_e4_done .esc_finished .esc_halt
set -uo pipefail
REPO=/mnt/share/w00608002/work/eplb/vllm-ascend
W=/mnt/share/w00608002/weight
OUT=/mnt/share/w00608002/work/eplb/output/night_exp_20261118
LOG=$OUT/evalscope_chain.log
E2_PID=${E2_PID:-1385035}
M_FP8=$W/DeepSeek-V4-Flash-0731
M_ATTND=$W/DeepSeek-V4-Flash-0731-32x32-noperm-w4a4c8-attnD
M_REPACK=$W/DeepSeek-V4-Flash-0731-vq2a8-mixed-gs32-repack
COMMON_ARGS=(--base-url http://127.0.0.1:8000 --model dsv
  --datasets gpqa aime24 aime25
  --offline-data-dir $W/vq2a8-eval-data
  --thinking --reasoning-effort high
  --temperature 1 --top-p 1 --seed 0 --repeats 3
  --concurrency 64 --max-tokens 131072 --timeout 7200)
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
mkdir -p "$OUT"

stop_server() {
    local p
    for p in $(pgrep -f "run_pd_w2a8_eager_graph[.]sh"); do kill -TERM "$p" 2>/dev/null; done
    sleep 15
    for p in $(ps -eo pid,cmd | awk '/vllm[.]entrypoints|load_balance_proxy_server/ && !/awk/ {print $1}'); do kill -9 "$p" 2>/dev/null; done
    sleep 60   # HCCL 设备回收缓冲
    log "服务已停清场"
}
start_server() {  # $1=MODEL_P $2=MODEL_D $3=serve日志
    ( cd "$REPO" && env MODEL_P="$1" MODEL_D="$2" QUANT=auto \
        setsid nohup bash tools/eplb/serve/run_pd_w2a8_eager_graph.sh --with-proxy >> "$3" 2>&1 & )
    local deadline=$((SECONDS + 2400))
    until curl -s -m 3 http://127.0.0.1:9001/v1/models >/dev/null 2>&1 && \
          curl -s -m 3 http://127.0.0.1:9002/v1/models >/dev/null 2>&1; do
        if ! pgrep -f "vllm[.]entrypoints" >/dev/null; then
            log "[error] serve 进程退出(见 $3)"; tail -5 "$3" | tee -a "$LOG"; return 1; fi
        (( SECONDS < deadline )) || { log "[error] serve 就绪超时"; return 1; }
        sleep 20
    done
    log "serve 就绪 ($1 × $2)"
}
run_esc() {  # $1=target(baseline|vq2a8) $2=label $3=result_dir
    ( cd "$REPO" && python3 -u tools/tests/run_evalscope_accuracy.py "$1" \
        "${COMMON_ARGS[@]}" --label "$2" --result-dir "$3" >> "$LOG" 2>&1 )
    local rc=$?
    log "evalscope[$1 $2] rc=$rc"
    return $rc
}

# ---- 阶段 0: 等 E2 evalscope(已在跑) ----
log "链启动: 等 E2 evalscope(pid=$E2_PID)结束"
while kill -0 "$E2_PID" 2>/dev/null; do sleep 30; done
log "E2 evalscope 已结束(成绩见 evalscope/E2_w2a8)"
stop_server

# ---- 阶段 1: E1 FP8 evalscope 重跑 ----
if [[ ! -f "$OUT/.esc_e1_done" ]]; then
    if start_server "$M_FP8" "$M_FP8" "$OUT/serve_esc_E1.log"; then
        if run_esc baseline fp8-pd-eager-graph-t1-r3-c64 "$OUT/evalscope/E1_fp8"; then
            touch "$OUT/.esc_e1_done"; log "E1 ✅"
        else log "[error] E1 evalscope 失败"; fi
    else log "[error] E1 起服失败"; fi
    stop_server
fi

# ---- 阶段 2: E4 混搭(smoke 门禁 + evalscope) ----
if [[ ! -f "$OUT/.esc_e4_done" ]]; then
    if start_server "$M_ATTND" "$M_REPACK" "$OUT/serve_esc_E4.log"; then
        resp=$(curl -s -m 300 http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
          -d '{"model":"dsv","messages":[{"role":"user","content":"请自我介绍，然后计算 23×47，一步步算，最后单独给出答案数字。"}],"max_tokens":4096,"temperature":0.6}')
        text=$(python3 -c "
import json,sys
try:
    r=json.loads(sys.stdin.read()); m=r['choices'][0]['message']
    print((m.get('reasoning_content') or '')+(m.get('content') or ''))
except Exception: print('')" <<<"$resp" 2>/dev/null)
        if [[ "$text" == *1081* && ${#text} -gt 100 ]]; then
            log "[smoke:E4] ✅ 通过(len=${#text})"
            run_esc vq2a8 mixed-p-w4a4attnD-d-w2a8repack-t1-r3-c64 "$OUT/evalscope/E4_mixed" \
                && { touch "$OUT/.esc_e4_done"; log "E4 ✅"; } || log "[error] E4 evalscope 失败"
        else
            log "[halt] E4 smoke 未过(乱码?) → 留样停链"; touch "$OUT/.esc_halt"
            printf '%s' "$resp" > "$OUT/e4_esc_smoke_sample.json"
        fi
    else log "[error] E4 起服失败"; fi
    stop_server
fi

ALL=1
[[ -f "$OUT/.esc_e1_done" ]] || ALL=0
[[ -f "$OUT/.esc_e4_done" || -f "$OUT/.esc_halt" ]] || ALL=0
if (( ALL )); then touch "$OUT/.esc_finished"; log "evalscope 链全部完成 ✅"; else log "链结束(有未完成阶段, 见标记)"; fi
