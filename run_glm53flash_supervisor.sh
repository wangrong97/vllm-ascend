#!/bin/bash
# =============================================================================
# GLM-5.3-Flash 实验管线守护进程
#
# 每 60 秒检查一次: 管线进程(run_glm53flash_pipeline.sh)不在且未全部完成
# (无 output/.pipeline_finished 标记)时自动拉起。管线本身可重入:
# 已完成阶段有 .done 标记会自动跳过, 重启后跳过启动窗口直接做空闲检测续跑。
#
# 用法: setsid nohup bash run_glm53flash_supervisor.sh > /dev/null 2>&1 &
# 停止: pkill -f run_glm53flash_supervisor  (管线不会被它误杀)
# =============================================================================

OUT=/mnt/share/w00608002/work/output/glm53flash
PIPELINE=/mnt/share/w00608002/work/vllm-ascend/run_glm53flash_pipeline.sh
LOG="${OUT}/supervisor.log"

mkdir -p "${OUT}"

while true; do
    if [[ -f "${OUT}/.pipeline_finished" ]]; then
        echo "[$(date '+%m-%d %H:%M:%S')] 管线已全部完成, supervisor 退出" >> "${LOG}"
        exit 0
    fi
    if ! pgrep -f "[r]un_glm53flash_pipeline" > /dev/null; then
        echo "[$(date '+%m-%d %H:%M:%S')] 管线进程不在, 重新拉起" >> "${LOG}"
        setsid nohup bash "${PIPELINE}" > /dev/null 2>&1 < /dev/null &
    fi
    sleep 60
done
