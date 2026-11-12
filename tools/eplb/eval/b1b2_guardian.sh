#!/usr/bin/env bash
# B1/B2 评测编排监护: 编排被 kill/崩溃后自动清场并重启(断点续跑补缺轮次), 全部完成后自动退出。
# 用法: nohup bash b1b2_guardian.sh >/dev/null 2>&1 &   (在定时启动编排之后挂起)
# 日志: output/b1b2_guardian.log
REPO=/mnt/share/w00608002/work/eplb/vllm-ascend
SCRIPT=$REPO/tools/eplb/eval/run_baseline_vs_w2a8.sh
GLOG=$REPO/output/b1b2_guardian.log
MAX_RESTART=5
restarts=0

log() { echo "$(date '+%F %T') [guardian] $*" >> "$GLOG"; }

clean_stage() {
  # 清残留 serve/评测进程(cmdline 匹配, awk 自排除防自杀; readlink 过滤非 python)
  local pids
  pids=$(ps -eo pid,cmd | awk '/vllm\.entrypoints|serve_vq2a8|evalscope\.cli|run_baseline_vs_w2a8/ && !/awk/ {print $1}')
  for p in $pids; do
    exe=$(readlink /proc/$p/exe 2>/dev/null)
    case "$exe" in *python*|*bash*) kill -9 "$p" 2>/dev/null && log "清场 kill $p";; esac
  done
  sleep 5
}

log "监护启动 (上限重启 $MAX_RESTART 次)"
while true; do
  if ps -eo cmd | awk '/run_baseline_vs_w2a8\.sh/ && !/awk|guardian/ {found=1} END{exit !found}'; then
    sleep 60; continue   # 编排活着, 正常巡检
  fi
  # 编排不在: 判定是否全部完成
  last=$(ls -td "$REPO"/output/baseline_vs_w2a8_*/ 2>/dev/null | head -1)
  if [[ -n "$last" ]] && grep -q '全部完成' "$last/summary.txt" 2>/dev/null; then
    log "评测全部完成 ($last), 监护结束"
    exit 0
  fi
  # 未完成且无编排 → 清场 + 重启
  if [[ $restarts -ge $MAX_RESTART ]]; then
    log "重启次数达上限 $MAX_RESTART, 放弃监护(人工介入)"
    exit 1
  fi
  restarts=$((restarts+1))
  log "编排消失且未完成 → 第 $restarts/$MAX_RESTART 次清场重启"
  clean_stage
  nohup bash "$SCRIPT" > "/mnt/share/w00608002/work/eplb/output/b1b2_guard_restart${restarts}_$(date +%m%d_%H%M).log" 2>&1 &
  log "已重启编排 pid=$! (断点续跑, 只补缺轮次)"
  sleep 180   # 给编排 3 分钟过 gate/起 serve, 避免重复判定
done
