#!/usr/bin/env bash
# EPLB 性能采集(PD 分离 DP4×EP4, v2 基于已验证的 v6 PD 启动器)
# 三组: off(基线) vs dynamic(混精感知) vs dynamic_uniform(标准)
# 服务: tools/eplb/serve/run_pd_w2a8_eager_graph.sh(已含proxy修复/warmup跳过/Ninja对齐)
# 压测: ais_bench perf, gsm8k 300条, batch 8, model=dsv @ proxy:8000
# 产物: /mnt/share/w00608002/work/eplb/output/eplb_perf_<ts>/{off,dynamic,uniform}/
set -uo pipefail
REPO=/mnt/share/w00608002/work/eplb/vllm-ascend
SERVE="$REPO/tools/eplb/serve/run_pd_w2a8_eager_graph.sh"
MODEL=/mnt/share/w00608002/weight/DeepSeek-V4-Flash-0731-32x32-noperm-hot5050125-identity-v3-export-20261103
OUT="/mnt/share/w00608002/work/eplb/output/eplb_perf_$(date +%m%d_%H%M)"
mkdir -p "$OUT"; echo "$OUT" > /tmp/eplb_perf_dir
OPTS='{"K":8,"R_vq2":16,"R_mx4":8,"n_stage":2,"w_lo":0.2,"w_hi":5.0,"ewma_lambda":0.05}'
PD_PID=""; PROXY_PID=""

clean_npu() { for round in 1 2 3; do local k=0
    for p in $(npu-smi info 2>/dev/null | awk '/Process id/{f=1;next} f&&/^\| [0-9]/{print $3}'); do
      [[ "$(readlink /proc/$p/exe 2>/dev/null)" == *python* ]] && { kill -9 "$p" 2>/dev/null; k=$((k+1)); }; done
    sleep 5; (( k == 0 )) && break; done; }

stop_pd() {
  [[ -n "$PD_PID" ]] && kill "$PD_PID" 2>/dev/null
  sleep 10
  for p in $(ps -eo pid,cmd | awk '/[v]llm.entrypoints|[r]un_pd_w2a8|[p]roxy_server_exam/ {print $1}'); do kill -9 "$p" 2>/dev/null; done
  sleep 5
  clean_npu
  PD_PID=""
}

run_group() {  # $1=tag $2=eplb模式
  local tag=$1 mode=$2
  echo "===== [$tag] D侧eplb=$mode $(date '+%F %T') =====" | tee -a "$OUT/summary.txt"
  clean_npu; sleep 10; mkdir -p "$OUT/$tag"

  # 起 PD (v6 启动器, noperm 权重, EPLB_MODE 注入 D 侧)
  EPLB_MODE="$mode" EPLB_OPTIONS="$OPTS" \
  MODEL_P="$MODEL" MODEL_D="$MODEL" \
    nohup bash "$SERVE" --with-proxy > "$OUT/$tag/pd.log" 2>&1 &
  PD_PID=$!

  # 等就绪(P+D+proxy)
  local dl=0 P=0 D=0 PR=0
  while (( dl < 90 )); do
    P=$(curl -s --max-time 3 http://127.0.0.1:9001/v1/models >/dev/null 2>&1 && echo 1 || echo 0)
    D=$(curl -s --max-time 3 http://127.0.0.1:9002/v1/models >/dev/null 2>&1 && echo 1 || echo 0)
    PR=$(curl -s --max-time 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && echo 1 || echo 0)
    [[ "$P$D$PR" == "111" ]] && break
    grep -qm1 'one instance exited' "$OUT/$tag/pd.log" 2>/dev/null && { echo "[$tag] PD退出: $(tail -3 "$OUT/$tag/pd.log" | tr '\n' ' ' | cut -c1-120)" | tee -a "$OUT/summary.txt"; stop_pd; return 1; }
    sleep 10; dl=$((dl+1)); done
  [[ "$P$D$PR" == "111" ]] || { echo "[$tag] 就绪超时" | tee -a "$OUT/summary.txt"; stop_pd; return 1; }
  echo "[$tag] PD+proxy就绪($(date '+%H:%M:%S'))" | tee -a "$OUT/summary.txt"

  # smoke(1081)
  local SM=$(curl -s --max-time 120 http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"dsv","messages":[{"role":"user","content":"计算 23 * 47 等于多少"}],"max_tokens":1024,"temperature":0}' | grep -c 1081)
  [[ "$SM" != "1" ]] && { echo "[$tag] smoke失败(无1081)" | tee -a "$OUT/summary.txt"; stop_pd; return 1; }
  echo "[$tag] smoke✅(含1081)" | tee -a "$OUT/summary.txt"

  # NPU 采样
  python "$REPO/tools/eplb/collect/eplb_tools/npu_load_sampler.py" --output "$OUT/$tag/npu.csv" --interval 2 --duration 0 &
  local NP=$!

  # ais_bench perf
  local T0=$(ls -t /mnt/share/w00608002/work/benchmark/outputs/default/ 2>/dev/null | head -1)
  (cd /mnt/share/w00608002/work/benchmark && ais_bench --models vllm_api_stream_chat \
    --datasets gsm8k_gen_0_shot_cot_str_perf --debug --summarizer default_perf --mode perf \
    --num-prompts 300) >> "$OUT/$tag/press.log" 2>&1 &
  local APID=$!

  # 停滞检测: 每2分钟 FIN 零增量×6 → 中止
  local last=-1 zero=0 fin
  while kill -0 $APID 2>/dev/null; do
    sleep 120
    fin=$(tail -1 "$OUT/$tag/press.log" 2>/dev/null | tr '\r' '\n' | tail -1 | grep -oE 'FIN=[0-9]+' | grep -oE '[0-9]+$'); fin=${fin:-0}
    if [[ "$fin" == "$last" ]]; then zero=$((zero+1)); else zero=0; fi; last=$fin
    echo "[$tag] 巡检 FIN=$fin zero=$zero" >> "$OUT/summary.txt"
    (( zero >= 6 )) && { echo "[$tag] !! 停滞12min判hang,中止" | tee -a "$OUT/summary.txt"
      kill -9 $APID 2>/dev/null; pkill -9 -f 'ais_bench.*gsm8k' 2>/dev/null; break; }
  done
  wait $APID 2>/dev/null; local RC=$?
  kill $NP 2>/dev/null

  # 归档
  local T1 tries=0
  while (( tries < 60 )); do T1=$(ls -t /mnt/share/w00608002/work/benchmark/outputs/default/ | head -1)
    [[ "$T1" != "$T0" ]] && break; sleep 5; tries=$((tries+1)); done
  [[ "$T1" != "$T0" ]] && cp -a "/mnt/share/w00608002/work/benchmark/outputs/default/$T1" "$OUT/$tag/aisbench" 2>/dev/null
  echo "[$tag] 完成 rc=$RC aisbench:$T1 $(date '+%H:%M:%S')" | tee -a "$OUT/summary.txt"

  stop_pd
  return 0
}

run_group off off
run_group dynamic dynamic
run_group uniform dynamic_uniform
echo "===== 全部完成 $(date '+%F %T') =====" | tee -a "$OUT/summary.txt"
for tag in off dynamic uniform; do
  echo "----- [$tag] perf 摘要 -----"
  find "$OUT/$tag/aisbench/summary" -name '*.md' 2>/dev/null | head -1 | xargs cat 2>/dev/null | head -25
done
