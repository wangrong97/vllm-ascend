#!/bin/bash
# gate_then_night3: 8 卡连续空闲 ≥10 分钟 → 直接启动 night3(续八C run2-5 等)
# night3 主链: tools/glm53flash/night3/(started 标记在, 起跑不等 01:00)
set +e
LOG=/mnt/share/w00608002/work/output/glm53flash/night3_gate.log
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
log "gate 启动: 等 8 卡连续空闲 10 分钟(每 30s 检查)"

stable=0
while true; do
    free=$(npu-smi info 2>/dev/null | awk '/\/ 98304/{print $(NF-3)+0}' | awk '$0<10000' | wc -l)
    if [[ "$free" == "8" ]]; then
        stable=$((stable + 30))
        (( stable % 120 == 30 )) && log "空闲累计 ${stable}s/600s"
        (( stable >= 600 )) && { log "8 卡已连续空闲 10 分钟, 启动 night3"; break; }
    else
        (( stable > 0 )) && log "检测到占用(空闲 ${free}/8), 计时重置(此前 ${stable}s)"
        stable=0
    fi
    sleep 30
done
setsid nohup bash /mnt/share/w00608002/work/vllm-ascend/tools/glm53flash/night3/night3_supervisor.sh > /dev/null 2>&1 &
log "supervisor 已挂 → night3 续跑(八C run2-5 + gpqa×3, 预计 ~5.5h)"
