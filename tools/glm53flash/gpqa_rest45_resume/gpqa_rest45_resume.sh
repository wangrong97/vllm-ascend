#!/bin/bash
# 实验七 GPQA 剩余 45 题续跑: 起同配置服务(W4A4+MXFP4伪量化+Hadamard) →
# 跑 gpqa_rest45 迷你集 → 手工合并出 run1 完整 summary → 停服务
set +e
BASE=/mnt/share/w00608002/work/vllm-ascend
BENCH=/mnt/share/w00608002/work/benchmark
OUTROOT=/mnt/share/w00608002/work/output/glm53flash
LOG="${OUTROOT}/exp7_gpqa_rest45.log"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "${LOG}"; }

# ---- 起 实验七同配置 服务 ----
log "===== 起服务: W4A4 + MXFP4 伪量化 + Hadamard(同实验七) ====="
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7   # 后四卡
export TP_SIZE=4
export MODEL_PATH=/mnt/share/w00608002/weight/GLM-5.3-Flash-FP8-w4a4
export QUANT_METHOD=ascend KV_CACHE_DTYPE=fp8 MAX_MODEL_LEN=132096
export VLLM_ASCEND_KPOOL_MXFP4_SIM=1 VLLM_ASCEND_KPOOL_HADAMARD_PRE=1
unset INDEXER_KV_DTYPE VLLM_ASCEND_KPOOL_MXFP8_SIM VLLM_ASCEND_INDEXER_DUMP
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

# ---- 跑 45 题迷你集 ----
D7="${OUTROOT}/night2_exp7_w4a4_mxfp4_hadamard"
R1="${D7}/gpqa_gen_0_shot_cot_chat_prompt/run1"
wd="${R1}/rest45"
mkdir -p "$wd"
( cd "${BENCH}" && ais_bench --models vllm_api_general_chat_glm \
    --datasets gpqa_rest45_gen_0_shot_cot_chat_prompt.py \
    --mode all --dump-eval-details --merge-ds \
    -w "$wd" > "$wd/aisbench.log" 2>&1 )
csv=$(ls "$wd"/*/summary/summary_*.csv 2>/dev/null | head -1)
rest_score=$(tail -1 "$csv" 2>/dev/null | awk -F, '{print $NF}')
log "[rest45] accuracy = ${rest_score}"

# ---- 合并 run1 总分(153 题已有 145 对 + 45 题新成绩) ----
python3 - "$R1" "$rest_score" <<'PYEOF'
import json, glob, re, csv as csvmod, os, sys
R1, rest45_score = sys.argv[1], sys.argv[2]
# 已完成 153 题的正确数
fs = glob.glob(f'{R1}/20261*/predictions/*/tmp/tmp_*.jsonl')
rows = [json.loads(l) for l in open(fs[0])]
done = [r for r in rows if r.get('output_tokens')]
def ans(t):
    m = re.findall(r'Answer:\s*([A-D])', str(t)) or re.findall(r'\b([A-D])\b\s*$', str(t).strip())
    return m[-1] if m else None
correct_done = sum(1 for r in done if ans(r.get('prediction','')) == str(r['gold']).strip().upper())
n_done = len(done)
n45 = 45
c45 = round(float(rest45_score) / 100 * n45)
total_n = n_done + n45
total_c = correct_done + c45
merged = total_c / total_n * 100
print(f'已完成 {n_done} 题正确 {correct_done} + rest45 {n45} 题约对 {c45} → 合并 {total_c}/{total_n} = {merged:.2f}%')
os.makedirs(f'{R1}/merged', exist_ok=True)
with open(f'{R1}/merged/summary_merged.csv', 'w', newline='') as f:
    w = csvmod.writer(f)
    w.writerow(['dataset', 'version', 'metric', 'mode', 'vllm-api-general-chat'])
    w.writerow(['gpqa_merged', '-', 'accuracy', 'gen', f'{merged:.2f}'])
PYEOF

# ---- 停服务 ----
bash "${BASE}/serve_glm53flash.sh" --stop >> "${LOG}" 2>&1
for p in $(pgrep -f "VLLM::"); do kill -9 $p 2>/dev/null; done
sleep 10
log "===== 完成 ====="
