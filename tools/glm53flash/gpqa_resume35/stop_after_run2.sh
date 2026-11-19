#!/bin/bash
# 六 gpqa run2 出分后立即暂停整链(不跑 run3; 断点保留, 重挂 supervisor 可续)
set +e
R2=/mnt/share/w00608002/work/output/glm53flash/night2_exp6_w4a4_mxfp8/gpqa_gen_0_shot_cot_chat_prompt/run2
LOG=/mnt/share/w00608002/work/output/glm53flash/stop_exp6_run2.log
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
log "watcher 启动: 等六 gpqa run2 出分"

while true; do
    csv=$(find "$R2" -name "summary_*.csv" 2>/dev/null | head -1)
    [[ -n "$csv" ]] && break
    sleep 15
done
log "run2 已出分: $(tail -1 "$csv" | awk -F, '{print $NF}')"

# 1) 先杀 supervisor(防 60s 重拉), 再杀 night3 主链, 再清 run3 的 ais
pkill -f "[n]ight3_supervisor" 2>/dev/null; sleep 1
pkill -f "[n]ight3_glm53flash" 2>/dev/null; sleep 2
pkill -9 -f "[a]is_bench --models" 2>/dev/null; pkill -9 -f "[o]penicl_api_infer" 2>/dev/null; sleep 2
# 2) 停服务
bash /mnt/share/w00608002/work/vllm-ascend/serve_glm53flash.sh --stop 2>/dev/null | tail -1
for p in $(pgrep -f "[V]LLM::"); do kill -9 $p 2>/dev/null; done
sleep 5
# 3) 复核
pgrep -f "[n]ight3|[a]is_bench|[V]LLM::" >/dev/null && log "[warn] 仍有残留, 需人工检查" || log "全部暂停(六 run3 未跑, 断点保留)"
npu-smi info | grep -c "No running processes" | xargs -I{} log "空闲卡: {}/8"
