#!/bin/bash
# finalize: 等三h补测(resume35)出 summary → 合并判分生成三h run1/2/3 summary → 标 done →
# 停服务 → 停止(不重启 supervisor, 不跑七run3/六/校准dump —— 2026-11-13 用户指令"补测完先停止任务")
set +e
BASE=/mnt/share/w00608002/work/vllm-ascend
OUTROOT=/mnt/share/w00608002/work/output/glm53flash
D3H="${OUTROOT}/night2_exp3h_w4a4_c8_hadamard"
WD="${OUTROOT}/exp3h_gpqa_resume35"
GP3="${D3H}/gpqa_gen_0_shot_cot_chat_prompt"
LOG="${OUTROOT}/finalize_exp3h.log"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "${LOG}"; }
has_result() { find "$1" -name "summary_*.csv" -exec tail -1 {} \; 2>/dev/null | awk -F, '{print $NF}' | grep -qE '^[0-9]+\.?[0-9]*$'; }

log "finalize 启动: 等 resume35 补测出 summary"
# ---- 1) 等补测 summary 落盘 ----
while true; do
    csv=$(find "${WD}" -name "summary_*.csv" 2>/dev/null | head -1)
    [[ -n "$csv" ]] && break
    # ais_bench 全退且 5 分钟无 summary → 异常退出
    if ! pgrep -f "ais_bench --models" >/dev/null 2>&1; then
        quiet=$((quiet + 20)); (( quiet >= 300 )) && { log "[error] ais_bench 已退但无 summary, 放弃合并"; exit 1; }
    else quiet=0; fi
    sleep 20
done
log "补测 summary 已出: $(tail -1 "$csv" | awk -F, '{print $NF}') (35 题小集分数, 仅供参照)"

# ---- 2) 合并判分: 三h run1/2/3 ----
python3 - "${GP3}" "${WD}" <<'PY' >> "${LOG}" 2>&1
import json, re, glob, os, sys
GP3, WD = sys.argv[1], sys.argv[2]
def ans(t):
    m = re.findall(r"[Aa]nswer[:\s]*\(?([A-D])\)?", t or "")
    return m[-1] if m else None
def qkey(r):
    return (r.get('origin_prompt') or '').split('\n\nA)')[0].strip()

# 补测结果(按 question 索引); tmp 被正常收尾清理时回退正式 jsonl
rf = (glob.glob(f'{WD}/20261*/predictions/*/tmp/tmp_*.jsonl')
      or glob.glob(f'{WD}/20261*/predictions/*/GPQA_diamond.jsonl'))
res = {}
for l in open(rf[0]):
    r = json.loads(l)
    if (r.get('prediction') or '').strip():
        res[qkey(r)] = ans(r['prediction']) == str(r['gold']).strip()
print(f'补测有效 {len(res)}/35 题')

for run in ('run1', 'run2', 'run3'):
    if os.path.exists(f'{GP3}/{run}/summary_offline.csv'):
        print(f'{run}: summary 已存在, 跳过'); continue
    tmps = glob.glob(f'{GP3}/{run}/20261*/predictions/*/tmp/tmp_*.jsonl')
    if tmps:
        rows = [json.loads(l) for l in open(tmps[0])]
    else:
        base = glob.glob(f'{GP3}/{run}/20261*/predictions/*/')[0]
        rows = [json.loads(l) for l in open(base + 'GPQA_diamond.jsonl')]
        try: rows += [json.loads(l) for l in open(base + 'GPQA_diamond_failed.jsonl')]
        except FileNotFoundError: pass
    ok = sum(1 for r in rows
             if (r.get('prediction') or '').strip() and ans(r['prediction']) == str(r['gold']).strip())
    miss = [r for r in rows if not (r.get('prediction') or '').strip()]
    patched = sum(1 for r in miss if res.get(qkey(r), False))
    score = (ok + patched) / 198 * 100
    with open(f'{GP3}/{run}/summary_offline.csv', 'w') as f:
        f.write('dataset,version,metric,mode,vllm-api-general-chat\n')
        f.write(f'GPQA_diamond,b1ed2c,accuracy,gen,{score:.2f}\n')
    print(f'{run}: 原答 {ok} 对 + 补测 {patched}/{len(miss)} 对 = {score:.2f}%')
PY

# ---- 3) 校验与标记 ----
ok=1
for r in run1 run2 run3; do has_result "${GP3}/$r" || ok=0; done
(( ok )) && touch "${D3H}/.done" && log "三h .done 已标(gpqa×3 合并出分完成)"
ls "${GP3}"/run*/summary_offline.csv 2>/dev/null | while read -r f; do log "  $f = $(tail -1 "$f" | awk -F, '{print $NF}')"; done

# ---- 4) 停服务(不再起任何新任务) ----
bash "${BASE}/serve_glm53flash.sh" --stop >> "${LOG}" 2>&1
sleep 5
for p in $(pgrep -f "VLLM::"); do kill -9 $p 2>/dev/null; done
sleep 10
log "服务已停, 全部任务停止(七 run3 / 六 / 校准 dump 均未启动, 等用户指令)"
log "===== finalize 结束 ====="
