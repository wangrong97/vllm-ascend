# 无 perm VQ2A8 混精模型：四卡并发吞吐测试

适用分支：`test/vq2a8-perf-20261024`；更新日期：2026-11-08。
模型为 `/mnt/share/g00872988/DeepSeek-V4-Flash-0731-32x32-noperm-hot5050125-identity-v3-export-20261103`。
Linux 路径按内网实际存放位置调整，服务端和客户端使用同一份模型；本地已检查的目录为 `D:\models\DeepSeek-V4-Flash-0731-32x32-noperm-hot5050125-identity-v3-export-20261103`。
本分支只接受自然输入列顺序，旧 `.perm` 张量（包括恒等排列）会被拒绝。
在内网 Linux 仓库根目录执行，沿用当前环境及与本分支匹配的 GS32 构建产物。
只测 TP4/EP4、DP1，不测 baseline 或其他卡数。
生产入口固定使用 native 完整 FFN；默认 decode 图回放，prefill 以 eager 执行同一完整融合。

默认测的是 [repack 输出](repack_non_routed.md)：非 routed 主矩阵采用 FP8 权重及动态 MXFP8 激活，routed 字节不变。
首次使用新权重先执行 [EP4/B64 验收](vq2a8_npu_ci.md#新-repack-权重四卡验收)。
若对照 repack 前的 BF16 版本，仅切换服务和客户端的模型目录，并将结果标签中的 `nonrouted-fp8` 改为 `nonrouted-bf16`；
两轮保持代码、卡数、KV、token budget 和请求长度一致。切换 eager/graph 不能代替这组量化前后对照。

## 0. 首次升级完整 FFN：重建 ABI v3

更新代码后，在已加载 CANN 环境的仓库根目录执行；`ASCEND_TOOLKIT_HOME` 应指向实际 CANN 安装。
本次框架优化保留设备内核 ABI v3、SwiGLU 裁剪和量化顺序；Torch 桥接新增计数 sanitizer schema，须重建 `.so` 和构建清单。
下面给出完整重建命令。已确认与当前设备目标、CANN 和 ABI v3 一致的 `.o` 可复用为桥接构建输入。

```bash
export BUILD="$PWD/build/vq2a8-gs32-runtime-ffn"
: "${ASCEND_TOOLKIT_HOME:?请先加载匹配的 CANN 环境}"
(
  cd csrc/mixed_w2w4_moe_ffn
  bash run_mixed_w2w4_moe_ffn_esl.sh
)
python tools/build_vq2a8_gs32_stage.py \
  --kernel-binary "$PWD/csrc/mixed_w2w4_moe_ffn/mixed_w2w4_moe_ffn.o" \
  --cann "$ASCEND_TOOLKIT_HOME" --build-dir "$BUILD" --jobs 4
```

构建报告必须为完成状态，预检会核对 v3 符号及 `.o/.so` 的 SHA256 与构建清单。
这只证明构建和接口匹配，不能替代设备执行验收。
本轮省去恒等路由的 gather/inverse 和归位缓冲，并合并有效性检查；详见[运行时说明](vq2a8_gs32_runtime.md#完整-ffn-框架开销优化)。
比较优化前后性能时使用同一服务器、同一四卡、模型、CANN、图桶、KV 与调度预算，先预热后重复测量。
同时记录服务加载后的图捕获时间、峰值显存、吞吐及 TTFT/TPOT；历史两卡和四卡结果不能作为本轮改动的收益证据。

## 1. 终端 A：启动服务

如果已按本节命令启动服务且符合下表，保持运行并直接执行第 2 节。重启前先停止原服务。设备选择及显存占用由使用者确认，
下面的 `4,5,6,7` 可换成实际空闲四卡。启动日志中的 `world_size` 应为 `4`。

| 参数 | 设置 |
| --- | --- |
| 模型执行 | native、MRV2、full_graph decode、eager chunked prefill、fusion=full_ffn |
| 拓扑 | TP4、EP4、DP1 |
| 服务容量 | max-num-seqs=64、max-num-batched-tokens=8192、max-model-len=8192 |
| KV cache | 每个 worker 16384 MiB |
| 投机解码 | `--no-dspark`，当前权重没有 MTP 张量 |

```bash
set -o pipefail
export MODEL=/mnt/share/g00872988/DeepSeek-V4-Flash-0731-32x32-noperm-hot5050125-identity-v3-export-20261103
export BUILD="$PWD/build/vq2a8-gs32-runtime-ffn"
RESULT_ROOT="$PWD/benchmark-results/batch-throughput"
SERVE_DIR="$RESULT_ROOT/serve-noperm-mixed-nonrouted-fp8-native-tp4-$(date +%Y%m%d_%H%M%S)"
SERVE_LINK="$RESULT_ROOT/current-native-8003"
mkdir -p "$RESULT_ROOT" &&
mkdir "$SERVE_DIR" &&
git rev-parse HEAD > "$SERVE_DIR/commit.txt" &&
git status --short > "$SERVE_DIR/worktree.txt" &&
sha256sum "$MODEL/config.json" "$MODEL/model.safetensors.index.json" > "$SERVE_DIR/model-files.sha256" &&
sha256sum "$BUILD/libvq2a8_gs32_stage.so" "$BUILD/mixed_w2w4_moe_ffn.o" > "$SERVE_DIR/gs32-files.sha256" &&
echo "服务日志：$SERVE_DIR/serve.log" &&
ln -sfnT "$SERVE_DIR" "$SERVE_LINK" &&

TMPDIR=/tmp VLLM_RPC_BASE_PATH=/tmp \
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 \
VLLM_USE_V2_MODEL_RUNNER=1 VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
VLLM_WORKER_MULTIPROC_METHOD=spawn OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -u tools/serve_vq2a8.py \
  --model "$MODEL" --no-dspark \
  --gs32-runtime-dir "$BUILD" \
  --parallel-mode tp-ep --ep-size 4 \
  --max-model-len 8192 \
  --max-num-batched-tokens 8192 --max-num-seqs 64 \
  --kv-cache-mib 16384 --reserve-gib 16 \
  --tokenizer-mode deepseek_v4 --reasoning-parser deepseek_v4 \
  --runtime-record "$SERVE_DIR/service-runtime.json" \
  --host 127.0.0.1 --port 8003 \
  2>&1 | tee "$SERVE_DIR/serve.log"
```

服务日志目录以本机时间 `YYYYMMDD_HHMMSS` 结尾，同类目录可按名称排序；测试结果的外层目录仍以时间开头。
准备命令通过 `&&` 衔接，失败时不再启动服务；若同一秒创建同名目录，等待一秒后重试即可，不会覆盖已有记录。
终端 A 每次启动时更新 `current-native-8003` 目录链接，服务退出后保留该链接和历史日志。
终端 B 在同一容器、同一仓库根目录执行，自动解析该链接，无需每次填写 `SERVE_DIR`。

`--reserve-gib` 是启动器对 KV 参数的校验上限，不会另行预留显存。
若出现 `Free memory ... less than desired GPU memory utilization`，是在权重加载前的设备检查失败；
应先处理所选卡的占用。仅修改客户端 `--ep-size` 不会改变服务的实际卡数。

## 2. 终端 B：扫描并发 1～64

使用与服务相同的 Python/vLLM 环境。暂停其他请求，保持服务配置不变。
目标输入 1024、输出 128 tokens；每档正式请求数为 `4×并发`，显式预热 1 次，正式测量 1 次。
使用 `/v1/completions` 流式请求，temperature=0、seed=0、ignore-eos，不套聊天模板。
客户端并发不等于每步实际 decode batch。

```bash
(
set -euo pipefail
MODEL=/mnt/share/g00872988/DeepSeek-V4-Flash-0731-32x32-noperm-hot5050125-identity-v3-export-20261103
BASE_URL=http://127.0.0.1:8003
RESULT_ROOT="$PWD/benchmark-results/batch-throughput"
if ! SERVE_DIR=$(readlink -e "$RESULT_ROOT/current-native-8003") || \
   [ ! -s "$SERVE_DIR/service-runtime.json" ]; then
  echo "未找到本轮 native 启动记录，请先按第 1 节启动服务并等待就绪。" >&2
  exit 1
fi
RUN_DIR="$RESULT_ROOT/$(date +%Y%m%d_%H%M%S)-bench-noperm-mixed-nonrouted-fp8-native-tp4-ep4"
mkdir "$RUN_DIR"
echo "服务记录：$SERVE_DIR/service-runtime.json"
echo "结果目录：$RUN_DIR"
git rev-parse HEAD > "$RUN_DIR/commit.txt"
git status --short > "$RUN_DIR/worktree.txt"

curl -fsS --max-time 30 "$BASE_URL/health"
curl -fsS --max-time 30 -D "$RUN_DIR/models.headers" \
  "$BASE_URL/v1/models" > "$RUN_DIR/models.json"

python -u tools/tests/run_vq2a8_throughput.py \
  --base-url "$BASE_URL" --model vq2a8 --tokenizer "$MODEL" \
  --tokenizer-mode deepseek_v4 \
  --parallel-mode tp-ep --ep-size 4 --max-num-seqs 64 \
  --concurrency 1 2 4 8 16 32 64 \
  --input-len 1024 --output-len 128 \
  --num-warmups 1 --repeats 1 --seed 0 \
  --result-dir "$RUN_DIR" \
  --runtime-record "$SERVE_DIR/service-runtime.json" \
  --note 'nonrouted-fp8; backend=native; devices=4,5,6,7; TP4/EP4/DP1; MRV2; full_graph; fusion=full_ffn; no DSpark; batched_tokens=8192; KV=16GiB/worker' \
  2>&1 | tee "$RUN_DIR/benchmark.log"

)
```

更换卡号时同步更新 `--note`；客户端拓扑参数仅记录声明，不会验证或重配服务。
`SERVE_DIR` 只用于找到 `service-runtime.json`，让测试结果带上服务端的 backend、fusion、execution、代码版本和算子哈希，
不参与推理计算。客户端在开始时解析为固定目录，扫描期间不要重启服务。
`current-native-8003` 只是本地启动记录的入口，不是 HTTP 服务实例证明；绕过本文命令启动的服务不会自动登记。
链接存在不表示服务仍在运行；重启时应重新执行第 1 节以更新记录，并等待服务就绪后再测试。
脚本自动检查每档成功请求数、失败数和每条输出长度；某档失败会停止并保留日志。
CLI 的初始校验请求和显式预热不计入正式指标。

## 3. 结果

脚本打印的 `Results: ...` 是 `RUN_DIR` 下本轮独立子目录。
该内部子目录仍由测试脚本自动生成；每轮的外层 `RUN_DIR` 已按时间命名，按外层目录排序即可。结果包含：

| 文件 | 内容 |
| --- | --- |
| `summary.csv` | 各并发输出 TPS、请求吞吐、平均及 P99 TTFT/TPOT、P99 E2EL、`model_path` |
| `summary.json` | `complete`、`by_concurrency`、`runs`、`peak_measured`、`model_path`、`server_runtime` 及来源 |
| `c<C>-r1.json` / `.log` | 每档原始指标、请求详情及日志 |
| `manifest.json` | 客户端环境、声明的服务拓扑、负载参数和备注 |

提供 `--runtime-record` 后，脚本先读取本轮服务的启动记录，再请求 `/v1/models`，
要求 `vq2a8` 唯一匹配且 `root` 与记录的 `model_path` 相同；每档原始结果的 `model_id` 也必须一致。
不匹配则停止。`model_path` 写入 JSON 顶层、每档结果和 CSV，不从 tokenizer 或标签推断。
`server_runtime` 保存实际解析的 backend/fusion/execution、代码版本及 native 构建产物 SHA256；
`server_runtime_source` 记录原始文件 SHA256 和证据范围。本轮目录同时保留 `service-runtime.json`、`models.json`。
该文件证明启动器的配置和构建产物，通过 `/v1/models` 只核对模型身份，不能证明当前 HTTP 服务实际执行了指定内核；
`device_execution_verified=false` 仍保留，实际执行需结合服务日志及 NPU 验收。整个扫描期间不要更换服务或权重。

输出 TPS 是总输出 token 数除以测量时间；TTFT 包含排队和 prefill；TPOT 为首 token 后的平均输出间隔。
汇总中的 `median_*` 是多轮指标的中位数，本次只有一轮，因此就是该轮结果。
低并发请求较少，P99 仅作本次短测参考。性能测量不代表答案精度通过。
反馈 `summary.json`、`models.json`、`service-runtime.json` 及服务日志即可。

## 4. 完整融合 eager 对照与数值验证

需要隔离图捕获或回放问题时，在第 1 节启动命令中添加 `--enforce-eager`，
保持模型、卡组、容量和完整 FFN 算子一致，重新创建本轮服务记录及结果目录。
客户端备注将 `full_graph` 改为 `fused/eager`；服务记录中的 execution 会记录实际选择。
该开关只改变图执行方式，仍调用完整 FFN，不再提供分阶段或 reference 服务路径。

独立数值参考仅存在于测试中：`tests/support/vq2a8_oracle.py` 定义量化和裁剪语义，
NPU 测试以解码后的权重及官方 GMM 对照完整 FFN。按 [NPU 验收指南](vq2a8_npu_ci.md)
检查真实混精专家、动态路由、图桶和整模型 eager/graph 一致性；这些检查不由吞吐压测代替。
