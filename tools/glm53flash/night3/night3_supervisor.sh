#!/bin/bash
# 夜间链守护(强化版 2026-11-16):
#   正常路径: night3 不在且未 finished → 60s 内重拉(偶发故障自愈)
#   空转熔断: 30 分钟内拉起 ≥3 次 = 确定性失败(代码 bug 类) →
#             调 autofix(已知症状幂等补丁) → 修好重试; 未知症状 → touch .night3_halt 停止空转
#   .night3_halt 存在时不拉起(人工排除故障后 rm 该文件再启动本脚本)
OUT=/mnt/share/w00608002/work/output/glm53flash
LOG="${OUT}/night3_supervisor.log"
LAUNCHES="${OUT}/.night3_launch_times"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "${LOG}"; }
mkdir -p "${OUT}"

while true; do
    if [[ -f "${OUT}/.night3_finished" ]]; then
        log "全部完成, 退出"; exit 0
    fi
    if [[ -f "${OUT}/.night3_halt" ]]; then
        log "halt 标记存在(熔断过), 不拉起; 排障后: rm ${OUT}/.night3_halt && bash $0"
        exit 0
    fi
    if ! pgrep -f "night3_glm53flash" >/dev/null 2>&1; then
        # ---- 空转检测: 记录拉起时刻, 淘汰 30min 前的记录 ----
        now=$(date +%s)
        touch "${LAUNCHES}"
        awk -v cutoff=$((now - 1800)) '$1 >= cutoff' "${LAUNCHES}" > "${LAUNCHES}.tmp" 2>/dev/null
        mv "${LAUNCHES}.tmp" "${LAUNCHES}"
        n_recent=$(wc -l < "${LAUNCHES}")
        if (( n_recent >= 2 )); then
            log "快速空转检测: 30min 内已拉起 ${n_recent} 次仍未存活 → 确定性失败"
            if bash /mnt/share/w00608002/work/vllm-ascend/tools/glm53flash/autofix_server_fail.sh >> "${LOG}" 2>&1; then
                log "autofix 成功, 清空计数并重试拉起"
                : > "${LAUNCHES}"
            else
                log "autofix 无法修复(未知症状), 熔断: touch .night3_halt, 错误摘要见上"
                touch "${OUT}/.night3_halt"
                exit 1
            fi
        fi
        echo "${now}" >> "${LAUNCHES}"
        log "管线不在, 拉起(30min 内第 $(wc -l < "${LAUNCHES}") 次)"
        setsid nohup bash /mnt/share/w00608002/work/vllm-ascend/tools/glm53flash/night3/night3_glm53flash.sh > /dev/null 2>&1 &
    fi
    sleep 60
done
