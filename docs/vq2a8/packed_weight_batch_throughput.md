# 无 perm VQ2A8 混精模型：四卡并发吞吐测试

适用分支：`test/vq2a8-perf-20261024`；更新日期：2026-11-07。
模型为 `/mnt/share/g00872988/DeepSeek-V4-Flash-0731-32x32-noperm-hot5050125-identity-v3-export-20261103`。
Linux 路径按内网实际存放位置调整，服务端和客户端使用同一份模型；本地已检查的目录为 `D:\models\DeepSeek-V4-Flash-0731-32x32-noperm-hot5050125-identity-v3-export-20261103`。
本分支只接受自然输入列顺序，旧 `.perm` 张量（包括恒等排列）会被拒绝。
在内网 Linux 仓库根目录执行，沿用当前环境及与本分支匹配的 GS32 构建产物。
只测 TP4/EP4、DP1，不测 baseline 或其他卡数。
第 1～3 节为默认 native；第 4 节为 `reference_mxfp8`，复用相同压测脚本和负载。
更新代码后再执行 reference 命令：`git pull --ff-only`。

默认测的是 [repack 输出](repack_non_routed.md)：非 routed 主矩阵采用 FP8 权重及动态 MXFP8 激活，routed 字节不变。
首次使用新权重先执行 [EP4/B64 验收](vq2a8_npu_ci.md#新-repack-权重四卡验收)。
若对照 repack 前的 BF16 版本，仅切换服务和客户端的模型目录，并将结果标签中的 `nonrouted-fp8` 改为 `nonrouted-bf16`；
两轮保持代码、卡数、KV、token budget 和请求长度一致。native/reference 都复用相同非 routed 计算路径，不能代替这组量化前后对照。

## 0. 首次升级完整 FFN：重建 ABI v3

更新代码后，在已加载 CANN 环境的仓库根目录执行；`ASCEND_TOOLKIT_HOME` 应指向实际 CANN 安装。
本次修改设备 SwiGLU 和 tiling ABI，旧 `.o` 不能与新 `.so` 混用。

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
  --gs32-backend native \
  --gs32-library "$BUILD/libvq2a8_gs32_stage.so" \
  --gs32-binary "$BUILD/mixed_w2w4_moe_ffn.o" \
  --parallel-mode tp-ep --ep-size 4 \
  --gs32-execution full_graph --gs32-fusion full_ffn \
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

要比较完整融合与分步 native，将第 1 节改为 `--gs32-fusion none`，
保留 `full_graph` 和其他负载参数，重新创建服务日志及结果目录；客户端备注也改为 `fusion=none`。
比较时以同一权重、同一代码/卡组、同一 KV/请求配置为准，分别记录完整融合与分步路径。

## 4. reference_mxfp8 对照路径

此入口复用现有 `ReferenceMXFP8Projector`。
它替换主模型 VQ2 投影（包括 `w13` 和 `w2`）：按专家、每块 256 个输出列解码为 FP8，
复用同一份 MXFP8 激活数据及缩放，调用 `torch_npu.npu_grouped_matmul`，
对 FP32 输出应用行缩放后转换为 BF16。MXFP4 投影、模型加载、EP 路由、Attention、共享专家和输出头继续用本分支实现。
权重在调用之间保持压缩态，不预先把整个模型展开为 FP8，不额外旋转。

reference 不加载 GS32 专用 `.so/.o`，因此不传 `--gs32-library/--gs32-binary`；
仍需现有 CANN、torch_npu、Triton 及 DeepSeek V4 主扩展算子。
native 默认使用 `full_ffn`，reference 保持 `none`；两条路径均不启用 MTP。

### 4.1 终端 A：启动 reference 服务

使用相同四卡依次测量；先停止该组卡上的 native 服务，再执行以下命令。
reference 使用端口 `8004` 和独立日志目录，便于区分结果。
本例同样为 `full_graph`、B64、token 预算 8192、每卡 KV 16 GiB。

```bash
set -o pipefail
export MODEL=/mnt/share/g00872988/DeepSeek-V4-Flash-0731-32x32-noperm-hot5050125-identity-v3-export-20261103
RESULT_ROOT="$PWD/benchmark-results/batch-throughput"
SERVE_DIR="$RESULT_ROOT/serve-noperm-mixed-nonrouted-fp8-reference-tp4-$(date +%Y%m%d_%H%M%S)"
SERVE_LINK="$RESULT_ROOT/current-reference-8004"
mkdir -p "$RESULT_ROOT" &&
mkdir "$SERVE_DIR" &&
git rev-parse HEAD > "$SERVE_DIR/commit.txt" &&
git status --short > "$SERVE_DIR/worktree.txt" &&
sha256sum "$MODEL/config.json" "$MODEL/model.safetensors.index.json" > "$SERVE_DIR/model-files.sha256" &&
echo "reference 服务日志：$SERVE_DIR/serve.log" &&
ln -sfnT "$SERVE_DIR" "$SERVE_LINK" &&

TMPDIR=/tmp VLLM_RPC_BASE_PATH=/tmp \
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 \
VLLM_USE_V2_MODEL_RUNNER=1 VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
VLLM_WORKER_MULTIPROC_METHOD=spawn OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -u tools/serve_vq2a8.py \
  --model "$MODEL" --no-dspark \
  --gs32-backend reference_mxfp8 \
  --parallel-mode tp-ep --ep-size 4 \
  --gs32-execution full_graph --gs32-fusion none \
  --max-model-len 8192 \
  --max-num-batched-tokens 8192 --max-num-seqs 64 \
  --kv-cache-mib 16384 --reserve-gib 16 \
  --tokenizer-mode deepseek_v4 --reasoning-parser deepseek_v4 \
  --runtime-record "$SERVE_DIR/service-runtime.json" \
  --host 127.0.0.1 --port 8004 \
  2>&1 | tee "$SERVE_DIR/serve.log"
```

reference 使用独立的 `current-reference-8004` 链接，每次启动时更新，下一节自动读取。
如修改服务端口，同步修改终端 A/B 的端口和链接名。

### 4.2 终端 B：测 reference 吞吐、TTFT、TPOT

```bash
(
set -euo pipefail
MODEL=/mnt/share/g00872988/DeepSeek-V4-Flash-0731-32x32-noperm-hot5050125-identity-v3-export-20261103
BASE_URL=http://127.0.0.1:8004
RESULT_ROOT="$PWD/benchmark-results/batch-throughput"
if ! SERVE_DIR=$(readlink -e "$RESULT_ROOT/current-reference-8004") || \
   [ ! -s "$SERVE_DIR/service-runtime.json" ]; then
  echo "未找到本轮 reference 启动记录，请先按第 4.1 节启动服务并等待就绪。" >&2
  exit 1
fi
RUN_DIR="$RESULT_ROOT/$(date +%Y%m%d_%H%M%S)-bench-noperm-mixed-nonrouted-fp8-reference-tp4-ep4"
mkdir "$RUN_DIR"
echo "reference 服务记录：$SERVE_DIR/service-runtime.json"
echo "reference 结果目录：$RUN_DIR"
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
  --note 'nonrouted-fp8; backend=reference_mxfp8; devices=4,5,6,7; TP4/EP4/DP1; MRV2; full_graph; fusion=none; no DSpark; batched_tokens=8192; KV=16GiB/worker' \
  2>&1 | tee "$RUN_DIR/benchmark.log"

)
```

结果文件及指标含义与第 3 节相同。区分两次结果时，同时保留 `manifest.json` 中的 backend 备注、
服务日志、`service-runtime.json` 和 `models.json`。backend 备注是客户端声明，`server_runtime` 来自提供的启动记录。
每档超时沿用脚本默认 7200 秒，超时会停止并保留日志。

### 4.3 执行方式与对比范围

reference 是诊断参考实现：每次执行都包含权重分块解码和多次 GMM 调用，
full_graph 还需为本地所有专家捕获固定调用序列，即使某个专家当前没有 token。
启动捕获时间、图内存和推理耗时可能明显高于 native；测得的是这条完整参考路径的服务性能，
不能据此判断官方 GMM 本身的性能上限，也不是原始 FP8 模型的 baseline。

若需要先检查 eager 路径，将 reference 启动命令中的 `--gs32-execution full_graph` 改为 `fused`；
该值表示现有 eager 分组执行，`--gs32-fusion` 仍为 `none`。同步将客户端备注改为 `fused`。
做同执行方式对照时，native 也使用 `fused` 单独测一轮，避免把 eager 和图执行差异归因于投影算子。

本地验证范围为 CPU 字节解码、GMM 接口、数值对照、路由/图控制和启动参数；
实际 Ascend 950 全模型启动、图捕获及吞吐仍待内网验证。本次 native 完整 FFN 升级为 ABI v3，使用 native 时需要一起重建 `.o` 和 `.so`。

### 4.4 启动预跑的 UINT8 MaskedFill 错误

若权重加载完成后，在 `gather_input` 报 `Tensor selfRef not implemented for DT_UINT8`，
原因是当前 NPU 的 `masked_fill_` 不接受 UINT8。已将 reference 激活和 scale 缓冲区的 padding 填充
改为共享存储的 INT8 视图；有效行字节保持不变，padding 数据为 0，E8M0 scale 为 127（数值 1）。
这是 Python 参考路径的兼容修复，无需改权重、重新量化或重编 GS32 算子。

执行 `git pull --ff-only` 后，重新运行第 4.1 节的原启动命令，参数保持不变。
本地回归覆盖高位字节、无 padding、部分 padding 和空专家；完整 NPU 启动仍需重试确认。
