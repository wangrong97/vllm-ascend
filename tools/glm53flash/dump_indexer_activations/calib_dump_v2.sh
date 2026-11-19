#!/bin/bash
# 校准集(R9tau, 904条) indexer q/k 激活 dump —— 断点续跑版
# 断点机制:
#   - progress.json 记录已发送条目(sent_upto), 每条发完即时落盘
#   - 每次服务启动用独立 round 目录(hook step 计数归零互不覆盖)
#   - 服务挂/中断 → 重启服务(新 round)从 progress 下一条继续; 连续 2 轮无进展才放弃
# 配置与 indexer_dump_exp1 对齐: 官方 FP8 权重 + idx bf16 + 无旋转 + KV fp8(原始域, 可比)
set +e
BASE=/mnt/share/w00608002/work/vllm-ascend
OUTROOT=/mnt/share/w00608002/work/output/glm53flash
CALIB=/mnt/share/w00608002/weight/calib_corpus_R9tau.jsonl
DUMPROOT=${OUTROOT}/indexer_dump_calib
PROG=${DUMPROOT}/progress.json
LOG="${OUTROOT}/calib_dump_v2.log"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "${LOG}"; }

mkdir -p "${DUMPROOT}"
[[ -f "${OUTROOT}/.calib_dump_done" ]] && { log "已完成(.calib_dump_done), 退出"; exit 0; }

TOTAL=$(wc -l < "${CALIB}")
read_progress() { [[ -f "${PROG}" ]] && python3 -c "import json;print(json.load(open('${PROG}'))['sent_upto']+1)" 2>/dev/null || echo 0; }

idx=$(read_progress)
round=$(ls -d "${DUMPROOT}"/round* 2>/dev/null | wc -l)
log "===== 校准 dump 启动: 条目 ${idx}..$((TOTAL-1)) (共 ${TOTAL}), round=${round} ====="
[[ ${idx} -ge ${TOTAL} ]] && { log "progress 已到尾, 标记完成"; touch "${OUTROOT}/.calib_dump_done"; exit 0; }

stall=0
while (( idx < TOTAL )); do
    DUMPDIR="${DUMPROOT}/round$(printf '%02d' "${round}")"
    mkdir -p "${DUMPDIR}"
    log "---- round ${round}: ${DUMPDIR}, 从第 ${idx} 条起 ----"

    # ---- 起服务(实验一配置) ----
    export MODEL_PATH=/mnt/share/w00936111/weights/GLM-5.3-Flash
    export QUANT_METHOD="" KV_CACHE_DTYPE=fp8 PORT=9000
    unset INDEXER_KV_DTYPE VLLM_ASCEND_KPOOL_HADAMARD_PRE VLLM_ASCEND_KPOOL_MXFP4_SIM VLLM_ASCEND_KPOOL_MXFP8_SIM
    export VLLM_ASCEND_INDEXER_DUMP=1 VLLM_ASCEND_INDEXER_DUMP_DIR="${DUMPDIR}" \
           VLLM_ASCEND_INDEXER_DUMP_MODE=raw VLLM_ASCEND_INDEXER_DUMP_MAX_STEP=2200
    ( cd "${BASE}" && PORT=9000 setsid nohup bash serve_glm53flash.sh >> "${LOG}" 2>&1 & )
    sleep 20
    ready=0
    deadline=$((SECONDS + 1500)); grace=$((SECONDS + 120))
    until curl -s -m 3 http://127.0.0.1:9000/health -o /dev/null 2>&1; do
        if ! pgrep -f "[b]in/vllm serve" >/dev/null; then
            if (( SECONDS < grace )) && pgrep -f "serve_glm53flash.sh" >/dev/null; then sleep 5; continue; fi
            log "[error] round ${round} 服务起失败"; break
        fi
        (( SECONDS < deadline )) || { log "[error] round ${round} 就绪超时"; break; }
        sleep 20
    done
    if curl -s -m 3 http://127.0.0.1:9000/health -o /dev/null 2>&1; then ready=1; log "round ${round} 服务就绪"; fi

    if (( ready )); then
        # ---- 串行灌校准集(completions, max_tokens=1 只抓 prefill; 每条发完即写 progress) ----
        python3 - "${CALIB}" "${idx}" "${PROG}" <<'PY' >> "${LOG}" 2>&1
import json, sys, time, urllib.request
calib, start, prog_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
# 只按 \n 切行: splitlines 会把 JSON 内   等也当切分点产生非法行(已踩坑, 坏行曾致 round 中断)
lines = open(calib).readlines()
t0 = time.time()
for i in range(start, len(lines)):
    text = json.loads(lines[i])["text"]
    body = json.dumps({"model": "glm", "prompt": text, "max_tokens": 1,
                       "temperature": 0.0}).encode()
    req = urllib.request.Request("http://127.0.0.1:9000/v1/completions", body,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        json.load(resp)
    with open(prog_path, 'w') as f:  # 每条即时落盘, 断点粒度=1 条
        json.dump({"sent_upto": i}, f)
    n = i - start + 1
    if n % 100 == 0:
        print(f"progress {i+1}/{len(lines)} ({n/(time.time()-t0):.1f} 条/s)", flush=True)
print(f"ROUND_DONE: 至第 {len(lines)-1} 条(0-based), 共 {len(lines)-start} 条")
PY
        rc=$?
        [[ $rc -eq 0 ]] && log "round ${round} 发送全部完成"
    fi

    # ---- 停服务, 评估进展 ----
    bash "${BASE}/serve_glm53flash.sh" --stop >> "${LOG}" 2>&1
    for p in $(pgrep -f "[V]LLM::"); do kill -9 $p 2>/dev/null; done
    sleep 15
    new_idx=$(read_progress)
    if (( new_idx > idx )); then
        log "round ${round}: 进展 ${idx}→${new_idx}"
        idx=${new_idx}; stall=0
    else
        stall=$((stall + 1))
        log "[warn] round ${round} 无进展(stall ${stall}/2)"
        (( stall >= 2 )) && { log "[error] 连续无进展, 放弃(可用本脚本重连续跑)"; exit 1; }
    fi
    round=$((round + 1))
done

# ---- 收尾 ----
touch "${OUTROOT}/.calib_dump_done"
summarize() {
    log "===== 校准 dump 完成: ${DUMPROOT} ====="
    du -sh "${DUMPROOT}" | tee -a "${LOG}"
    find "${DUMPROOT}" -name "*.npz" | wc -l | xargs -I{} log "npz 文件数: {}"
    for d in "${DUMPROOT}"/round*; do
        log "  $(basename $d): $(ls $d/*.npz 2>/dev/null | wc -l) npz"
    done
}
summarize
