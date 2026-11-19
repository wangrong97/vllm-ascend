#!/bin/bash
# 三h gpqa 剩余 17 题补跑 + 全量合并出分(手动触发: bash exp3h_resume17.sh)
# 前置: 8 卡空闲; 数据状态见 exp3h_gpqa_resume35/resume_state.json
# 流程: 起三h配置服务 → 跑 gpqa_resume17 → 合并(completed_18 + 本次17题) →
#       生成 run2/3 summary(run1=89.90 已落盘) → 标三h .done → 停服务 → 结束(不启动其他任务)
set +e
BASE=/mnt/share/w00608002/work/vllm-ascend
BENCH=/mnt/share/w00608002/work/benchmark
OUTROOT=/mnt/share/w00608002/work/output/glm53flash
D3H="${OUTROOT}/night2_exp3h_w4a4_c8_hadamard"
GP3="${D3H}/gpqa_gen_0_shot_cot_chat_prompt"
WDP="${OUTROOT}/exp3h_gpqa_resume35"
WD="${OUTROOT}/exp3h_resume17"
LOG="${OUTROOT}/exp3h_resume17.log"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "${LOG}"; }

log "===== 三h 剩余 17 题补跑启动 ====="

# ---- 1) 起三h配置服务 ----
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
log "服务就绪(W4A4 + C8 + Hadamard)"

# ---- 2) 跑 resume17 ----
mkdir -p "$WD"
( cd "${BENCH}" && ais_bench --models vllm_api_general_chat_glm \
    --datasets gpqa_resume17_gen_0_shot_cot_chat_prompt.py \
    --mode all --dump-eval-details --merge-ds \
    -w "$WD" > "$WD/aisbench.log" 2>&1 )
log "resume17 跑完: $(tail -1 $(ls "$WD"/*/summary/summary_*.csv 2>/dev/null | head -1) 2>/dev/null | awk -F, '{print $NF}')"

# ---- 3) 合并出分(id 映射: resume17 行号 -> remain_ids -> 原 id) ----
python3 - "${GP3}" "${WDP}" "${WD}" <<'PY' >> "${LOG}" 2>&1
import json, re, glob, os, sys
GP3, WDP, WD = sys.argv[1], sys.argv[2], sys.argv[3]
def ans(t):
    m = re.findall(r"[Aa]nswer[:\s]*\(?([A-D])\)?", t or "")
    return m[-1] if m else None

state = json.load(open(f'{WDP}/resume_state.json'))
ids, remain_ids = state['ids'], state['remain_ids']
res = {int(k): v for k, v in state['completed'].items()}  # 已完成 18 题判分

# 本次 resume17 结果(行号 id -> 原 id)
rf = (glob.glob(f'{WD}/20261*/predictions/*/tmp/tmp_*.jsonl')
      or glob.glob(f'{WD}/20261*/predictions/*/GPQA_diamond.jsonl'))
for l in open(rf[0]):
    r = json.loads(l)
    if (r.get('prediction') or '').strip():
        res[remain_ids[r['id']]] = ans(r['prediction']) == str(r['gold']).strip()
print(f'合并判分池: {len(res)}/35 题(18 旧 + 17 新)')

for run, base_ok, f in [('run2', 173, '20261113_040913'), ('run3', 155, '20261113_050915')]:
    if os.path.exists(f'{GP3}/{run}/summary_offline.csv'):
        print(f'{run}: 已有 summary, 跳过'); continue
    miss = [int(json.loads(l)['id']) for l in open(
        f'{GP3}/{run}/{f}/predictions/vllm-api-general-chat/GPQA_diamond_failed.jsonl')]
    assert base_ok == sum(1 for r in map(json.loads, open(
        glob.glob(f'{GP3}/{run}/{f}/predictions/*/GPQA_diamond.jsonl')[0]))
        if (r.get('prediction') or '').strip() and ans(r['prediction']) == str(r['gold']).strip()), f'{run} 基线对数不符'
    patched = sum(1 for i in miss if res.get(i, False))
    score = (base_ok + patched) / 198 * 100
    with open(f'{GP3}/{run}/summary_offline.csv', 'w') as fo:
        fo.write('dataset,version,metric,mode,vllm-api-general-chat\n')
        fo.write(f'GPQA_diamond,b1ed2c,accuracy,gen,{score:.2f}\n')
    print(f'{run}: {base_ok} 原答 + {patched}/{len(miss)} 补测 = {score:.2f}%')
PY

# ---- 4) 校验标记 + 停服务(不启动其他任务) ----
ok=1
for r in run1 run2 run3; do
    find "${GP3}/$r" -name "summary_*.csv" -exec tail -1 {} \; 2>/dev/null | awk -F, '{print $NF}' | grep -qE '^[0-9]+\.?[0-9]*$' || ok=0
done
(( ok )) && touch "${D3H}/.done" && log "三h .done 已标(gpqa×3 齐: run1=89.90 + run2/3 合并)"
for f in "${GP3}"/run*/summary_offline.csv; do log "  $(basename $(dirname $f)) = $(tail -1 $f | awk -F, '{print $NF}')"; done

bash "${BASE}/serve_glm53flash.sh" --stop >> "${LOG}" 2>&1
sleep 5
for p in $(pgrep -f "[V]LLM::"); do kill -9 $p 2>/dev/null; done
sleep 10
log "服务已停, 脚本结束(七 run3/六/校准 dump 仍待用户指令)"
log "===== exp3h_resume17 结束 ====="
