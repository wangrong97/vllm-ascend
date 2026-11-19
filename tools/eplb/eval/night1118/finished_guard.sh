#!/usr/bin/env bash
# finished 守卫(一次性): 编排 touch .finished 后检查三个 done 标记,
# 有缺 → rm .finished 并重启 supervisor 补跑(编排断点保证已完成阶段秒过)。
# 背景: 2026-11-19 00:07 E1 评测遍全崩(ais_bench datasets 遮蔽, 已修),
# 编排尾部无条件 touch .finished 的缺陷由本守卫兜底(编排文件运行中不可安全编辑)。
OUT=/mnt/share/w00608002/work/eplb/output/night_exp_20261118
SUP=/mnt/share/w00608002/work/eplb/vllm-ascend/tools/eplb/eval/night1118/night_exp_supervisor.sh
while true; do
    if [[ -f "$OUT/.finished" ]]; then
        missing=""
        [[ -f "$OUT/.e1_done" ]] || missing="$missing E1"
        [[ -f "$OUT/.e2_done" ]] || missing="$missing E2"
        [[ -f "$OUT/.e4_done" || -f "$OUT/.e3_failed" ]] || missing="$missing E4"
        if [[ -n "$missing" ]]; then
            echo "[$(date '+%m-%d %H:%M:%S')] [guard] .finished 但缺:$missing → rm finished 重启 supervisor 补跑" >> "$OUT/supervisor.log"
            rm -f "$OUT/.finished"
            setsid nohup bash "$SUP" > /dev/null 2>&1 &
        else
            echo "[$(date '+%m-%d %H:%M:%S')] [guard] 全部 done, 正常收工" >> "$OUT/supervisor.log"
        fi
        exit 0
    fi
    sleep 60
done
