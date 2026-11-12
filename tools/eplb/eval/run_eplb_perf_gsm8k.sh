#!/usr/bin/env bash
# EPLB 性能采集(PD 分离 DP4xEP4 形态): off 基线 vs 混精 eplb(dynamic) vs 标准 eplb(dynamic_uniform)
# 服务: run_pd_mixed_v2.sh — P(0-3)=W2A8 dp4ep4 + D(4-7)=W2A8 dp4ep4+eplb{EPLB_MODE}, Hybrid+双超时, :9001/:9002, proxy:8000
# 压测: ais_bench perf 模式 gsm8k 300条/batch 8, model=dsv @ proxy 8000
# profiling: 压测前 start(9001+9002), 压测后 stop; NPU 2s 采样
# 产物: output/eplb_perf_<ts>/{off,dynamic,uniform}/ 各含 press.log/npu.csv/profiling/{P,D}/aisbench/
set -uo pipefail
REPO=/mnt/share/w00608002/work/eplb/vllm-ascend
cd "$REPO"
OUT="/mnt/share/w00608002/work/eplb/output/eplb_perf_$(date +%m%d_%H%M)"
mkdir -p "$OUT"; echo "$OUT" > /tmp/eplb_perf_dir
OPTS='{"K":8,"R_vq2":16,"R_mx4":8,"n_stage":2,"w_lo":0.2,"w_hi":5.0,"ewma_lambda":0.05}'
PROXY_PID=""

clean_npu() { for round in 1 2 3; do local k=0
    for p in $(npu-smi info 2>/dev/null | awk '/Process id/{f=1;next} f&&/^\| [0-9]/{print $3}'); do
      [[ "$(readlink /proc/$p/exe 2>/dev/null)" == *python* ]] && { kill -9 "$p" 2>/dev/null; k=$((k+1)); }; done
    sleep 5; (( k == 0 )) && break; done; }

wait_free() { local c=0 t=0 maxt=$(( ${WAIT_FREE_TIMEOUT_MIN:-15} * 2 ))
  while true; do local pids; pids=$(npu-smi info 2>/dev/null | awk '/Process id/{f=1;next} f&&/^\| [0-9]/{print $3}' | head -8)
    [[ -z "$pids" ]] && c=$((c+1)) || c=0
    [[ $c -ge 2 ]] && return 0
    t=$((t+1)); (( t % 4 == 0 )) && echo "[gate] 等卡空闲 ${t}x30s" | tee -a "$OUT/summary.txt"
    (( t >= maxt )) && { echo "[gate] 超时, 强制清场放行" | tee -a "$OUT/summary.txt"; clean_npu; return 0; }
    sleep 30; done; }

run_group() {  # $1=tag $2=eplb模式
  local tag=$1 mode=$2
  echo "===== [$tag] D侧eplb=$mode $(date '+%F %T') =====" | tee -a "$OUT/summary.txt"
  echo "[$tag] 前置清场" | tee -a "$OUT/summary.txt"; clean_npu; wait_free
  mkdir -p "$OUT/$tag"
  EPLB_MODE="$mode" EPLB_OPTIONS="$OPTS" MODEL_P=/mnt/share/w00608002/weight/DeepSeek-V4-Flash-0731-vq2a8-mixed-gs32-repack \
    PROF_DIR_P="$OUT/$tag/profiling/P" PROF_DIR_D="$OUT/$tag/profiling/D" \
    nohup bash run_pd_mixed_v2.sh > "$OUT/$tag/pd.log" 2>&1 &
  local SPID=$!
  # 双实例就绪(最多 15 分钟)
  local dl=0
  while (( dl < 60 )); do
    local P D; P=$(curl -s --max-time 3 http://127.0.0.1:9001/v1/models >/dev/null 2>&1 && echo 1 || echo 0)
    D=$(curl -s --max-time 3 http://127.0.0.1:9002/v1/models >/dev/null 2>&1 && echo 1 || echo 0)
    [[ "$P$D" == "11" ]] && break
    grep -qm1 -E 'ValueError|RuntimeError|Traceback' "$OUT/$tag/pd.log" 2>/dev/null && { echo "[$tag] PD启动失败: $(grep -m1 -E 'ValueError|RuntimeError|Error|error:' "$OUT/$tag/pd.log" | cut -c1-130)" | tee -a "$OUT/summary.txt"; kill $SPID; clean_npu; return 1; }
    sleep 15; dl=$((dl+1)); done
  [[ "$P$D" == "11" ]] || { echo "[$tag] PD就绪超时" | tee -a "$OUT/summary.txt"; kill $SPID; clean_npu; return 1; }
  echo "[$tag] PD双实例就绪($(date '+%H:%M:%S'))" | tee -a "$OUT/summary.txt"
  # proxy(先清旧实例防 8000 占用, 起后验证)
  pkill -9 -f load_balance_proxy_server_example 2>/dev/null; sleep 2
  nohup python3 examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
    --host 127.0.0.1 --port 8000 --prefiller-ports 9001 --decoder-ports 9002 --workers 1 > "$OUT/$tag/proxy.log" 2>&1 &
  PROXY_PID=$!
  sleep 6
  curl -s --max-time 5 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 || { echo "[$tag] proxy未就绪(8000)" | tee -a "$OUT/summary.txt"; kill $SPID $PROXY_PID 2>/dev/null; clean_npu; return 1; }
  # smoke 1 条(PD 链路健康)
  local SM=$(curl -s --max-time 120 http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"dsv","messages":[{"role":"user","content":"计算 23 * 47 等于多少"}],"max_tokens":512,"temperature":0}' | grep -c 1081)
  [[ "$SM" != "1" ]] && { echo "[$tag] smoke失败(无1081)" | tee -a "$OUT/summary.txt"; kill $SPID $PROXY_PID 2>/dev/null; clean_npu; return 1; }
  echo "[$tag] smoke✅(含1081)" | tee -a "$OUT/summary.txt"
  # profiling start(两侧) + NPU 采样 + 压测
  curl -s -X POST http://127.0.0.1:9001/start_profile; curl -s -X POST http://127.0.0.1:9002/start_profile
  python tools/eplb/collect/eplb_tools/npu_load_sampler.py --output "$OUT/$tag/npu.csv" --interval 2 --duration 0 &
  local NP=$!
  local T0=$(ls -t /mnt/share/w00608002/work/benchmark/outputs/default/ | head -1)
  (cd /mnt/share/w00608002/work/benchmark && ais_bench --models vllm_api_stream_chat \
    --datasets gsm8k_gen_0_shot_cot_str_perf --debug --summarizer default_perf --mode perf \
    --num-prompts 300) >> "$OUT/$tag/press.log" 2>&1 &
  local APID=$!
  # 停滞检测: 每2分钟 FIN, 连续6次零增量(12min)判 hang 中止
  local last=-1 zero=0 fin
  while kill -0 $APID 2>/dev/null; do
    sleep 120
    fin=$(tail -1 "$OUT/$tag/press.log" 2>/dev/null | tr '\r' '\n' | tail -1 | grep -oE 'FIN=[0-9]+' | grep -oE '[0-9]+$'); fin=${fin:-0}
    if [[ "$fin" == "$last" ]]; then zero=$((zero+1)); else zero=0; fi; last=$fin
    echo "[$tag] 巡检 FIN=$fin zero=$zero" >> "$OUT/summary.txt"
    (( zero >= 6 )) && { echo "[$tag] !! 压测停滞12min判hang, 中止本组" | tee -a "$OUT/summary.txt"
      kill -9 $APID 2>/dev/null; pkill -9 -f 'ais_bench.*gsm8k' 2>/dev/null; break; }
  done
  wait $APID 2>/dev/null; local RC=$?
  curl -s -X POST http://127.0.0.1:9001/stop_profile; curl -s -X POST http://127.0.0.1:9002/stop_profile
  echo "[$tag] profiling已停止" | tee -a "$OUT/summary.txt"
  kill $NP 2>/dev/null
  # profiling 产物归位(serve 工作目录下找 ascend_pt/trace)
  sleep 10
  find "$REPO" -maxdepth 1 -name 'profiling*' -newer "$OUT/$tag/pd.log" 2>/dev/null | head -2
  # aisbench 输出归档(最多等5分钟, hang中止时无新目录不死等)
  local T1 tries=0; while (( tries < 60 )); do T1=$(ls -t /mnt/share/w00608002/work/benchmark/outputs/default/ | head -1)
    [[ "$T1" != "$T0" ]] && break; sleep 5; tries=$((tries+1)); done
  [[ "$T1" != "$T0" ]] && cp -a "/mnt/share/w00608002/work/benchmark/outputs/default/$T1" "$OUT/$tag/aisbench" 2>/dev/null || echo "[$tag] 无新aisbench目录(可能hang中止), 跳过归档" | tee -a "$OUT/summary.txt"
  echo "[$tag] 压测完成 rc=$RC aisbench:$T1 $(date '+%H:%M:%S')" | tee -a "$OUT/summary.txt"
  # 清理(PD脚本 trap 会自杀, 兜底再清)
  kill $SPID $PROXY_PID 2>/dev/null; sleep 10
  for p in $(ps -eo pid,cmd | awk '/run_pd_mixed|load_balance_proxy/ && !/awk/ {print $1}'); do kill -9 "$p" 2>/dev/null; done
  clean_npu
  return 0
}

run_group off off
run_group dynamic dynamic
run_group uniform dynamic_uniform
echo "===== 全部完成 $(date '+%F %T') =====" | tee -a "$OUT/summary.txt"
for tag in off dynamic uniform; do
  echo "----- [$tag] aisbench perf 摘要 -----"
  find "$OUT/$tag/aisbench/summary" -name '*.md' 2>/dev/null | head -1 | xargs cat 2>/dev/null | head -25
done