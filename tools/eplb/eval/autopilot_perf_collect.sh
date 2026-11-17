#!/usr/bin/env bash
# =============================================================================
# autopilot_perf_collect.sh — EPLB 三组采集的自修理+重启闭环
#
# 行为(2026-11-16 用户指令: 采集失败需自修理+重启, 直到采集成功):
#   循环(上限 MAX_ROUNDS=6 轮, 防根本性故障下无限烧资源):
#     1. 自修理(历史教训全收录):
#        a. 杀残留进程(vllm/proxy/ais_bench/采样器)
#        b. 清 /dev/shm/psm_*(mooncake 传输段残留, 2026-11-04 教训)
#        c. /dev/shm 容量检查(<1G 则 remount 16G, 容器重启场景, 2026-11-02 教训)
#        d. 等 NPU HBM 回落(<8000MB/卡, kill 后驱动 GC 慢, 最多 10 分钟)
#     2. 检查三组产物, 只重跑缺失/失败的组(EPLB_PERF_GROUPS 注入)
#     3. 校验成功(每组 aisbench/summary 有非空 .md); 全过 → 退出
#     4. 失败 → 记录 + 退避(第 n 轮 sleep min(10×n, 30) 分钟) → 下一轮
#
# 用法: bash autopilot_perf_collect.sh [OUT目录]
# =============================================================================
set -uo pipefail

REPO=/mnt/share/w00608002/work/eplb/vllm-ascend
OUT="${1:-/mnt/share/w00608002/work/eplb/output/eplb_perf_$(date +%m%d_%H%M)}"
MAX_ROUNDS="${EPLB_PERF_MAX_ROUNDS:-6}"
mkdir -p "$OUT"
LOG="$OUT/autopilot.log"
log() { echo "[autopilot] $(date '+%F %T') $*" | tee -a "$LOG"; }

# ---------- 自修理 ----------
repair() {
    log "---- 自修理开始 ----"
    # a. 残留进程
    local k
    for pat in '[v]llm.entrypoints' '[r]un_pd_w2a8' '[p]roxy_server_exam' 'ais_bench.*[g]sm8k' '[n]pu_load_sampler'; do
        pkill -9 -f "$pat" 2>/dev/null && log "已清理进程: $pat"
    done
    sleep 5
    # b. mooncake 残留段
    local n; n=$(ls /dev/shm/psm_* 2>/dev/null | wc -l)
    (( n > 0 )) && { rm -f /dev/shm/psm_*; log "已清 $n 个 psm 残留段"; }
    # c. shm 容量(容器重启后回到 64M 的场景)
    local sz; sz=$(df --output=size -m /dev/shm 2>/dev/null | tail -1 | tr -d ' ')
    if [[ -n "$sz" && "$sz" -lt 1024 ]]; then
        if mount -o remount,size=16G /dev/shm 2>/dev/null; then
            log "/dev/shm 已扩到 16G(原 ${sz}M)"
        else
            log "警告: /dev/shm 仅 ${sz}M 且 remount 失败"
        fi
    fi
    # d. 等 NPU HBM 回落(驱动 GC 慢, skill 注意事项 7)
    local w=0
    while (( w < 20 )); do
        local high; high=$(npu-smi info 2>/dev/null | grep -oE "[0-9]+ / 98304" | awk '{if ($1>8000) c++} END{print c+0}')
        [[ "${high:-1}" == "0" ]] && break
        log "等待 HBM 回落(高占用卡 ${high} 张, ${w}/10min)"; sleep 30; w=$((w+1))
    done
    log "---- 自修理完成 ----"
}

# ---------- 成功判定: 每组 aisbench/summary 有非空 .md ----------
group_ok() {
    local tag=$1
    local f
    f=$(find "$OUT/$tag/aisbench" -name '*.md' -size +0 2>/dev/null | head -1)
    [[ -n "$f" ]]
}

missing_groups() {
    local m=()
    for tag in off dynamic uniform; do group_ok "$tag" || m+=("$tag"); done
    echo "${m[*]:-}"
}

# ---------- 主循环 ----------
log "启动: OUT=$OUT 上限 ${MAX_ROUNDS} 轮"
round=1
while (( round <= MAX_ROUNDS )); do
    miss=$(missing_groups)
    if [[ -z "$miss" ]]; then
        log "✔ 三组产物齐全, 无需采集"; exit 0
    fi
    log "==== 第 ${round}/${MAX_ROUNDS} 轮, 待采集组: $miss ===="
    repair
    EPLB_PERF_OUT="$OUT" EPLB_PERF_GROUPS="$miss" \
        bash "$REPO/tools/eplb/eval/run_eplb_perf_gsm8k.sh" >> "$LOG" 2>&1
    rc=$?
    miss2=$(missing_groups)
    if [[ -z "$miss2" ]]; then
        log "✔ 采集成功(第 ${round} 轮, rc=$rc)"; exit 0
    fi
    backoff=$(( round * 10 )); (( backoff > 30 )) && backoff=30
    log "✗ 第 ${round} 轮未完成(缺: $miss2, rc=$rc), 退避 ${backoff} 分钟后重试"
    sleep $(( backoff * 60 ))
    round=$((round+1))
done
log "✗ 达到 ${MAX_ROUNDS} 轮上限仍缺: $(missing_groups) — 需人工介入"
exit 1
