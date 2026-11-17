#!/usr/bin/env bash
# =============================================================================
# schedule_perf_collect.sh — 预约夜间启动 EPLB 三组性能采集
#
# 行为(2026-11-16 用户指令):
#   1. 等待到指定时刻(默认 03:30)
#   2. 到点后等待 NPU 空闲(连续 2 次确认: 无 NPU 进程 且 每卡 HBM < 8000MB)
#      —— 必要防护: 采集脚本 run_eplb_perf_gsm8k.sh 的 clean_npu 会杀掉
#      NPU 上所有 python 进程, 若彼时他人任务在跑会被误杀, 故必须确认空闲
#   3. 空闲后 exec 采集脚本(off → dynamic → uniform 三组, 脚本自带
#      就绪超时/smoke校验/12min停滞检测/组间容错)
#
# HBM 阈值 8000: 底噪 4815-5468 横跨 5000(2026-11-03 教训), 真任务起步 20GB+
#
# 用法: bash schedule_perf_collect.sh [HH:MM=03:30]
# =============================================================================
set -uo pipefail

REPO=/mnt/share/w00608002/work/eplb/vllm-ascend
START_AT="${1:-03:30}"

# ---------- 1. 等待到预约时刻 ----------
now_s=$(date +%s)
target_s=$(date -d "today $START_AT" +%s)
if (( target_s <= now_s )); then
    target_s=$(date -d "tomorrow $START_AT" +%s)
fi
echo "[sched] $(date '+%F %T') 已预约 $START_AT 启动 EPLB 三组采集(off/dynamic/uniform)"
sleep $((target_s - now_s))
echo "[sched] $(date '+%F %T') 到点, 等待 NPU 空闲(连续 2 次, 防误杀他人任务)..."

# ---------- 2. 等待 NPU 空闲 ----------
free=0
while (( free < 2 )); do
    pids=$(npu-smi info 2>/dev/null | awk '/Process id/{f=1;next} f&&/^\| [0-9]/{print $3}' | head -8)
    hbm=$(npu-smi info 2>/dev/null | grep -oE "[0-9]+ / 98304" | awk '{if ($1>8000) c++} END{print c+0}')
    if [[ -z "$pids" && "${hbm:-1}" == "0" ]]; then
        free=$((free+1))
    else
        free=0
        echo "[sched] $(date '+%T') NPU 忙(进程 ${#pids} 个/高HBM卡 ${hbm:-?}), 60s 后重查"
    fi
    (( free < 2 )) && sleep 60
done
echo "[sched] $(date '+%F %T') NPU 空闲确认, 启动采集"

# ---------- 3. 启动采集(autopilot 自修理闭环, 替换本进程) ----------
cd "$REPO"
exec bash tools/eplb/eval/autopilot_perf_collect.sh
