#!/usr/bin/env bash
# B2→B1 定时评测监护 v2: 评测启动后守护, 进程消失且未完成 → 清场 → 幂等重启串联链(断点续跑保证已完成轮次秒过)
# 特性: 独立 setsid 派生(与被监护链不同批); 完成判定→自动退出; 上限重启5次
# 日志: output/b1b2_timer_1110.log (与定时器同文件, 前缀[guardian])
R=/mnt/share/w00608002/work/eplb/vllm-ascend
LOG=$R/output/b1b2_timer_1110.log
MAX_RESTART=5
restarts=0

log() { echo "$(date '+%F %T') [guardian] $*" >> "$LOG"; }

clean_stage() {
  for p in $(ps -eo pid,cmd | awk '/run_baseline_vs_w2a8|serve_vq2a8|vllm.entrypoints|evalscope.cli/ && !/awk|guardian/ {print $1}'); do
    [[ "$(readlink /proc/$p/exe 2>/dev/null)" == *python* || "$(readlink /proc/$p/exe 2>/dev/null)" == *bash* ]] && kill -9 "$p" 2>/dev/null
  done
  sleep 5
  for p in $(npu-smi info 2>/dev/null | awk '/Process id/{f=1;next} f&&/^\| [0-9]/{print $3}'); do
    [[ "$(readlink /proc/$p/exe 2>/dev/null)" == *python* ]] && kill -9 "$p" 2>/dev/null
  done
  sleep 5
}

log "监护进程启动(等待 01:00 定时触发标志)"
# 等定时器触发: 只认日志标志; 到 11-10 02:00(epochn秒) 仍无标志则代启动
TRIGGER_DEADLINE=$(date -d '2026-11-10 02:00' +%s)
while :; do
  grep -q '\[timer\] B2' "$LOG" 2>/dev/null && break
  sleep 60
  (( $(date +%s) > TRIGGER_DEADLINE )) && { log "02:00 仍未见触发标志(定时器可能已死) → 代为启动评测链"; break; }
done

while true; do
  sleep 60
  grep -q '\[timer\] B1完成.*全部结束' "$LOG" 2>/dev/null && { log "评测全部完成, 监护结束"; exit 0; }
  # 编排存活检测: 排除 bash -c 形式(定时器/监护自身 cmdline 含命令文本会被误匹配)
  if ps -eo pid,cmd | awk '$0 !~ /awk|guardian|bash -c/ && /bash tools\/eplb\/eval\/run_baseline_vs_w2a8/ {f=1} END{exit !f}'; then continue; fi
  # 不在 → 重启
  if (( restarts >= MAX_RESTART )); then log "重启达上限 $MAX_RESTART, 放弃(人工介入)"; exit 1; fi
  restarts=$((restarts+1))
  log "评测链不存在且未完成 → 第 $restarts/$MAX_RESTART 次清场重启"
  clean_stage
  setsid nohup bash -c "
    cd $R
    bash tools/eplb/eval/run_baseline_vs_w2a8.sh --skip-b1 >> $LOG 2>&1
    echo \"\$(date '+%F %T') [timer] B2完成 rc=\$? → 启动 B1\" >> $LOG
    bash tools/eplb/eval/run_baseline_vs_w2a8.sh --skip-b2 >> $LOG 2>&1
    echo \"\$(date '+%F %T') [timer] B1完成 rc=\$? 全部结束\" >> $LOG
  " >/dev/null 2>&1 &
  log "已重启串联链 pid=$!"
  sleep 300  # 给链 5 分钟起 serve, 避免重复判定
done
