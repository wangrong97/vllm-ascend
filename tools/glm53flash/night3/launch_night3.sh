#!/bin/bash
# night3 启动调度器(2026-11-14 用户指令):
#   00:20 检查 8 卡空闲 → 空闲: 立即启动; 不空闲: 改 01:00 启动(night3 原生 01:00 等待)
# 00:20 空闲路径: touch .night3_started 跳过 night3 的 01:00 等待
# 01:00 路径: 不 touch, night3 自行等到 01:00; 起服时 start_server 内置 8 卡 gate 双保险
set +e
OUT=/mnt/share/w00608002/work/output/glm53flash
LOG=$OUT/night3_launch.log
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

# ---- 1) 等到 00:20 ----
now=$(date +%s); t=$(date -d "00:20" +%s)
(( now >= t )) && t=$((t + 86400))
log "调度器启动: 等待 00:20(还有 $(( (t - now) / 60 )) 分钟)"
sleep $((t - now))

# ---- 2) 8 卡空闲检查(间隔 30s 两次确认, 防瞬时波动) ----
free_check() { npu-smi info 2>/dev/null | awk '/\/ 98304/{print $(NF-3)+0}' | awk '$0<10000' | wc -l; }
f1=$(free_check); sleep 30; f2=$(free_check)
rm -f "${OUT}/.night3_started"   # 清 11-14 轮遗留标记, 两条路径都从干净状态开始

# ---- 3) 决定启动时刻并挂 supervisor ----
if [[ "${f1}" == "8" && "${f2}" == "8" ]]; then
    log "00:20 检查: 8 卡空闲(两次确认 ${f1}/${f2}) → 立即启动"
    touch "${OUT}/.night3_started"
else
    log "00:20 检查: 卡被占用(空闲 ${f1}→${f2}/8) → 改为 01:00 启动(night3 原生等待)"
fi
setsid nohup bash /mnt/share/w00608002/work/vllm-ascend/tools/glm53flash/night3/night3_supervisor.sh > /dev/null 2>&1 &
log "supervisor 已挂(链序: 七o2A → 七o2B → 八A → 八B → 八C; 已 done 阶段秒过)"
