#!/usr/bin/env bash
# 夜间链守护(night3 式): 编排不在且未 finished → 60s 内重拉(偶发故障自愈)
#   空转熔断: 30 分钟内拉起 ≥3 次 = 确定性失败 → touch .halt 停止(人工排障后 rm 再启)
#   .halt 已存在时不拉起; .finished 存在则退出
OUT=/mnt/share/w00608002/work/eplb/output/night_exp_20261118
LOG=$OUT/supervisor.log
LAUNCHES=$OUT/.sup_launch_times
mkdir -p "$OUT"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }

while true; do
    [[ -f "$OUT/.finished" ]] && { log "全部完成, 退出"; exit 0; }
    if [[ -f "$OUT/.halt" ]]; then
        log "halt 标记存在(熔断/乱码门禁), 不拉起; 排障后: rm $OUT/.halt && bash $0"
        exit 0
    fi
    if ! pgrep -f "night_exp_1118[.]sh" >/dev/null 2>&1; then
        now=$(date +%s)
        touch "$LAUNCHES"
        awk -v cutoff=$((now - 1800)) '$1 >= cutoff' "$LAUNCHES" > "${LAUNCHES}.tmp" 2>/dev/null
        mv "${LAUNCHES}.tmp" "$LAUNCHES"
        n_recent=$(wc -l < "$LAUNCHES")
        if (( n_recent >= 3 )); then
            log "空转检测: 30min 内已拉起 ${n_recent} 次 → 确定性失败, 熔断"
            tail -20 /mnt/share/w00608002/work/eplb/output/night_exp_20261118/main.log >> "$LOG" 2>/dev/null
            touch "$OUT/.halt"
            exit 1
        fi
        echo "$now" >> "$LAUNCHES"
        log "编排不在, 拉起(30min 内第 $(wc -l < "$LAUNCHES") 次)"
        setsid nohup bash /mnt/share/w00608002/work/eplb/vllm-ascend/tools/eplb/eval/night1118/night_exp_1118.sh >> /dev/null 2>&1 &
    fi
    sleep 60
done
