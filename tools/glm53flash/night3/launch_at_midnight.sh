#!/bin/bash
# 今晚 00:00 启动 night3(链: 八C 续 run4/5+gpqa → 四 全套; 已 done 阶段秒过)
set +e
LOG=/mnt/share/w00608002/work/output/glm53flash/night3_launch.log
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
now=$(date +%s); t=$(date -d "00:00" +%s)
(( now >= t )) && t=$((t + 86400))
log "调度: 等待 00:00(还有 $(( (t - now) / 60 )) 分钟)→ night3 续八C 并新增实验四"
sleep $((t - now))
setsid nohup bash /mnt/share/w00608002/work/vllm-ascend/tools/glm53flash/night3/night3_supervisor.sh > /dev/null 2>&1 &
log "supervisor 已挂(00:00 准点)"
