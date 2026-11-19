#!/bin/bash
# 夜间实验守护: 每 60 秒检查
#   - .night2_finished 存在 → 退出
#   - overnight_glm53flash_quant.sh 不在运行 → 重新拉起(脚本自身断点续跑,
#     各阶段 .done + run 级有效成绩自动跳过; 可抵御 03:01 外部清扫)
OUTROOT=/mnt/share/w00608002/work/output/glm53flash
LOG="${OUTROOT}/night2_supervisor.log"
mkdir -p "${OUTROOT}"
while true; do
    if [[ -f "${OUTROOT}/.night2_finished" ]]; then
        echo "[$(date '+%m-%d %H:%M:%S')] 实验全部完成, supervisor 退出" >> "${LOG}"
        exit 0
    fi
    if ! pgrep -f "overnight_glm53flash_quant.sh" >/dev/null 2>&1; then
        echo "[$(date '+%m-%d %H:%M:%S')] 管线不在, 拉起" >> "${LOG}"
        setsid nohup bash /mnt/share/w00608002/work/vllm-ascend/tools/glm53flash/night2/overnight_glm53flash_quant.sh > /dev/null 2>&1 &
    fi
    sleep 60
done
