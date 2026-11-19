#!/bin/bash
# watcher_c: 等实验七 gpqa run2 出分(summary csv 落盘) → 停 night3 主链(七 run3 不跑) →
# 交给已挂的 watcher_b(等主链退出后起三h服务补测缺失 35 题并合并出分)
set +e
GP7=/mnt/share/w00608002/work/output/glm53flash/night2_exp7_w4a4_mxfp4_hadamard/gpqa_gen_0_shot_cot_chat_prompt
R2="${GP7}/run2"
LOG=/mnt/share/w00608002/work/output/glm53flash/stop_night3.log
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
log "watcher_c 启动: 等七 gpqa run2 出分"

# ---- 1) 等 run2 summary csv ----
while true; do
    csv=$(find "$R2" -name "summary_*.csv" 2>/dev/null | head -1)
    [[ -n "$csv" ]] && break
    sleep 20
done
score=$(tail -1 "$csv" | awk -F, '{print $NF}')
log "run2 已出分: ${score}"

# ---- 2) 停 night3 主链(先主实例防拉新, 再清残留 ais_bench=刚起的 run3) ----
pkill -f "night3_glm53flash.sh" 2>/dev/null; sleep 3
pkill -9 -f "ais_bench --models" 2>/dev/null; sleep 2
if pgrep -f "night3_glm53flash" >/dev/null 2>&1; then
    log "[warn] night3 仍存活, 强杀"; pkill -9 -f "night3_glm53flash"
fi
log "night3 主链已停(七 run3 未跑, 后续由 supervisor 重拉时断点补齐)"

# ---- 3) watcher_b(已挂)将检测到主链退出 → 清服务 → 起三h配置补测 35 题 → 合并三h run1/2/3 ----
log "等待 watcher_b 接管(见 resume_exp3h.log)"
