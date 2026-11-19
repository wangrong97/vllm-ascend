#!/bin/bash
# watcher_a: 监控实验七 gpqa run1 的孤儿 infer 进程(被 watchdog 杀了主进程但 infer 存活),
# 跑完后离线判分 tmp(198 题) 生成 summary_offline.csv, 让 night3 主实例的 stage_complete 能自动标七 .done
set +e
R1=/mnt/share/w00608002/work/output/glm53flash/night2_exp7_w4a4_mxfp4_hadamard/gpqa_gen_0_shot_cot_chat_prompt/run1
LOG=/mnt/share/w00608002/work/output/glm53flash/exp7_run1_watch.log
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
log "watcher_a 启动: 等孤儿 infer(3825842/3825843)跑完 run1"

# 等 infer 进程退出(双 PID 保险), 或 tmp 稳定在 198 行超 5 分钟
while true; do
    if ! kill -0 3825843 2>/dev/null && ! kill -0 3825842 2>/dev/null; then
        log "孤儿 infer 已退出"; break
    fi
    tmp=$(ls "$R1"/20261*/predictions/*/tmp/tmp_*.jsonl 2>/dev/null | head -1)
    if [[ -n "$tmp" ]]; then
        n=$(wc -l < "$tmp")
        if (( n >= 198 )); then
            stable=$((stable + 60))
            (( stable >= 300 )) && { log "tmp 已 198 行且稳定 5min"; break; }
        else
            stable=0
        fi
        log "tmp=${n}/198 行"
    fi
    sleep 60
done

# 判分(与 ais_bench 校准一致的口径: 最后一处 Answer: X)
python3 - "$R1" <<'PY' >> "$LOG" 2>&1
import json, re, glob, sys
R1 = sys.argv[1]
fs = glob.glob(f'{R1}/20261*/predictions/*/tmp/tmp_*.jsonl')
rows = [json.loads(l) for l in open(fs[0])]
def ans(t):
    m = re.findall(r"[Aa]nswer[:\s]*\(?([A-D])\)?", t or "")
    return m[-1] if m else None
done = [r for r in rows if (r.get('prediction') or '').strip()]
ok = sum(1 for r in done if ans(r['prediction']) == str(r['gold']).strip())
score = ok / 198 * 100
print(f"run1 补齐判分: {len(rows)}行/{len(done)}有效, 正确{ok}/198 = {score:.2f}%")
with open(f'{R1}/summary_offline.csv', 'w') as f:
    f.write('dataset,version,metric,mode,vllm-api-general-chat\n')
    f.write(f'GPQA_diamond,b1ed2c,accuracy,gen,{score:.2f}\n')
print(f"summary_offline.csv 已写入: {score:.2f}")
PY
log "watcher_a 结束"
