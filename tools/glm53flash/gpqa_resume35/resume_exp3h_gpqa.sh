#!/bin/bash
# watcher_b: night3 主链退出后(六跑完/异常), 起三h配置服务补测 gpqa 缺失 35 题 →
# 合并生成三h run1/2/3 完整 summary → 标 done → 兜底七 done → 重启 supervisor(校验全done→finished→calib dump)
set +e
BASE=/mnt/share/w00608002/work/vllm-ascend
BENCH=/mnt/share/w00608002/work/benchmark
OUTROOT=/mnt/share/w00608002/work/output/glm53flash
D3H="${OUTROOT}/night2_exp3h_w4a4_c8_hadamard"
D7="${OUTROOT}/night2_exp7_w4a4_mxfp4_hadamard"
LOG="${OUTROOT}/resume_exp3h.log"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "${LOG}"; }

has_result() { find "$1" -name "summary_*.csv" -exec tail -1 {} \; 2>/dev/null | awk -F, '{print $NF}' | grep -qE '^[0-9]+\.?[0-9]*$'; }

# ---- 1) 等 night3 主链退出 且 卡上无残留服务 ----
log "等待 night3 主链退出..."
while pgrep -f "night3_glm53flash.sh" >/dev/null 2>&1; do sleep 120; done
log "主链已退出"
while pgrep -f "VLLM::" >/dev/null 2>&1; do
    log "卡上仍有 VLLM 残留, 清理"
    for p in $(pgrep -f "VLLM::"); do kill -9 $p 2>/dev/null; done
    sleep 15
done

# ---- 2) 三h 已完整则直接收尾 ----
if [[ -f "${D3H}/.done" ]]; then
    log "三h 已 done, 跳过补测"; finish=1
else
    # ---- 3) 起三h配置服务(W4A4-FP8 + C8 + Hadamard) ----
    log "===== 起服务: 三h 配置(W4A4 + C8 + Hadamard) ====="
    export MODEL_PATH=/mnt/share/w00608002/weight/GLM-5.3-Flash-FP8-w4a4
    export QUANT_METHOD=ascend KV_CACHE_DTYPE=fp8 INDEXER_KV_DTYPE=fp8 VLLM_ASCEND_KPOOL_HADAMARD_PRE=1
    unset VLLM_ASCEND_KPOOL_MXFP4_SIM VLLM_ASCEND_KPOOL_MXFP8_SIM VLLM_ASCEND_INDEXER_DUMP
    ( cd "${BASE}" && PORT=9000 setsid nohup bash serve_glm53flash.sh >> "${LOG}" 2>&1 & )
    sleep 20
    deadline=$((SECONDS + 1500)); grace=$((SECONDS + 120))
    until curl -s -m 3 http://127.0.0.1:9000/health -o /dev/null 2>&1; do
        if ! pgrep -f "[b]in/vllm serve" >/dev/null; then
            if (( SECONDS < grace )) && pgrep -f "serve_glm53flash.sh" >/dev/null; then sleep 5; continue; fi
            log "[error] 服务进程退出"; exit 1
        fi
        (( SECONDS < deadline )) || { log "[error] 就绪超时"; exit 1; }
        sleep 20
    done
    log "服务就绪"

    # ---- 4) 跑 resume35 迷你集 ----
    WD="${OUTROOT}/exp3h_gpqa_resume35"
    mkdir -p "$WD"
    ( cd "${BENCH}" && ais_bench --models vllm_api_general_chat_glm \
        --datasets gpqa_resume35_gen_0_shot_cot_chat_prompt.py \
        --mode all --dump-eval-details --merge-ds \
        -w "$WD" > "$WD/aisbench.log" 2>&1 )
    log "resume35 跑完"

    # ---- 5) 合并判分: run1(tmp198) / run2(188+10) / run3(163+35) ----
    python3 - "${D3H}" "${WD}" <<'PY' >> "${LOG}" 2>&1
import json, re, glob, sys
D3H, WD = sys.argv[1], sys.argv[2]
def ans(t):
    m = re.findall(r"[Aa]nswer[:\s]*\(?([A-D])\)?", t or "")
    return m[-1] if m else None
def qkey(r):  # question 文本作为匹配键(选项顺序不同不可比, question 相同)
    return (r.get('origin_prompt') or '').split('\n\nA)')[0].strip()

# 补测结果(按 question 索引); ais_bench 正常收尾会清 tmp 只留正式 jsonl, 两手准备
rf = (glob.glob(f'{WD}/20261*/predictions/*/tmp/tmp_*.jsonl')
      or glob.glob(f'{WD}/20261*/predictions/*/GPQA_diamond.jsonl'))
res = {}
for l in open(rf[0]):
    r = json.loads(l)
    if (r.get('prediction') or '').strip():
        res[qkey(r)] = ans(r['prediction']) == str(r['gold']).strip()
print(f'补测有效 {len(res)}/35 题')

GP = f'{D3H}/gpqa_gen_0_shot_cot_chat_prompt'
def write_summary(run, ok, n=198, note=''):
    score = ok / n * 100
    with open(f'{GP}/{run}/summary_offline.csv', 'w') as f:
        f.write('dataset,version,metric,mode,vllm-api-general-chat\n')
        f.write(f'GPQA_diamond,b1ed2c,accuracy,gen,{score:.2f}\n')
    print(f'{run}: 正确 {ok}/198 = {score:.2f}% {note}')

for run in ('run1', 'run2', 'run3'):
    import os
    if os.path.exists(f'{GP}/{run}/summary_offline.csv'):
        print(f'{run}: summary 已存在, 跳过'); continue
    tmps = glob.glob(f'{GP}/{run}/20261*/predictions/*/tmp/tmp_*.jsonl')
    if tmps:
        rows = [json.loads(l) for l in open(tmps[0])]
    else:
        # tmp 被正常收尾清理: 正式 jsonl(有效答案) + failed(空答案中断题)
        base = glob.glob(f'{GP}/{run}/20261*/predictions/*/')[0]
        rows = [json.loads(l) for l in open(base + 'GPQA_diamond.jsonl')]
        try:
            rows += [json.loads(l) for l in open(base + 'GPQA_diamond_failed.jsonl')]
        except FileNotFoundError:
            pass
    ok = sum(1 for r in rows
             if (r.get('prediction') or '').strip() and ans(r['prediction']) == str(r['gold']).strip())
    miss = [r for r in rows if not (r.get('prediction') or '').strip()]
    patched = sum(1 for r in miss if res.get(qkey(r), False))
    write_summary(run, ok + patched, note=f'(原答 {ok} 对 + 补测 {patched}/{len(miss)} 对)')
PY

    # ---- 6) 校验与标记 ----
    GP3="${D3H}/gpqa_gen_0_shot_cot_chat_prompt"
    ok=1
    for r in run1 run2 run3; do has_result "${GP3}/$r" || ok=0; done
    (( ok )) && touch "${D3H}/.done" && log "三h .done 已标"

    # 停服务
    bash "${BASE}/serve_glm53flash.sh" --stop >> "${LOG}" 2>&1
    for p in $(pgrep -f "VLLM::"); do kill -9 $p 2>/dev/null; done
    sleep 10; log "服务已停"
fi

# ---- 7) 兜底七 done(run1 offline + run2/3 正常 summary 齐) ----
GP7="${D7}/gpqa_gen_0_shot_cot_chat_prompt"
if [[ ! -f "${D7}/.done" ]]; then
    ok7=1
    for r in run1 run2 run3; do has_result "${GP7}/$r" || ok7=0; done
    (( ok7 )) && touch "${D7}/.done" && log "七 .done 已标(兜底)" || log "[warn] 七 gpqa 未齐, 不标"
fi

# ---- 8) 重启 supervisor: 新实例校验三阶段全 done → finished → calib dump 自动接力 ----
if [[ -f "${D3H}/.done" && -f "${D7}/.done" && -f "${OUTROOT}/night2_exp6_w4a4_mxfp8/.done" ]]; then
    setsid nohup bash "${BASE}/night3_supervisor.sh" > /dev/null 2>&1 &
    log "supervisor 已重启(将快速校验并标 finished)"
else
    log "[warn] 尚有阶段未 done, 不重启 supervisor(supervisor 会重拉 night3 补齐)"
    setsid nohup bash "${BASE}/night3_supervisor.sh" > /dev/null 2>&1 &
    log "supervisor 已重启(重拉 night3 断点补齐)"
fi
log "===== watcher_b 结束 ====="
