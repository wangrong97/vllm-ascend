# VQ2A8 + DSpark 四卡性能与精度验证

该路径使用 VQ2A8 source-v3 checkpoint 作为目标模型，使用同源原始
DeepSeek-V4-Flash-0731 checkpoint 中的 DSpark 草稿权重。
目标模型继续执行 GS32 VQ2/MXFP4 专家；草稿模型使用原始 DeepSeek-V4 FP8 量化方案，
不读取目标 checkpoint 的 GS32 专家张量。最终 token 由目标模型验证与拒绝采样决定。

当前实现范围为 MRV2、DSpark 5 个候选 token、同步调度，
目标模型使用 `full_graph` decode 图，原始草稿复用 DeepSeek-V4 的 DSpark decode 图路径。
其他推测算法和 PD 分离不在此范围内。
打包后用 `--dspark` 自动读取目标目录下的 `dspark/`；也可以用
`--dspark-model PATH` 指向独立草稿包或完整同源原始模型。
两种启用参数均省略时保持普通 VQ2A8 解码，即使目标目录下已有草稿包也不会自动开启。
`--num-speculative-tokens` 启用后默认 5，目前也只允许 5。

建议按“打包草稿 → B4 性能快测与普通解码对照 → 精度验证”顺序执行。
代码及 CPU 测试不能代替 A5 PR/DT 实机验收。下面的四卡 B4/B16、图回放、
输出一致性、长上下文容量与性能均需要在目标环境验证；尚不能承诺吞吐加速比。

## 将原始草稿打包到目标模型目录

`tools/repack_vq2a8_dspark.py` 从原始 FP8/FP4 checkpoint 提取全部 `mtp.*`，
以及 `embed.weight`、`head.weight`、`hc_head_fn`、`hc_head_base`、`hc_head_scale`。
保留这些公共参数，可以沿用完整原始 checkpoint 的草稿加载方式，
不依赖目标模型的 embedding/head 是否经过其他转换。
原始 `config.json` 原样保留，包括 DSpark 参数、FP8/FP4 量化配置及草稿层的 `compress_ratios`。
脚本生成只引用提取后张量的独立索引，并复制已有的 tokenizer、generation 配置。

2026-10-22 检查 ACS 上的 `DeepSeek-V4-Flash-0731-vq2a8-mixed-gs32`：
全部 56 个分片均没有 MTP 张量，DSpark 配置也未保留；缺失发生在 GS32 repack 之前。
同源原始 0731 的三个 MTP 分片包含 4705 个张量，约 10.12 GiB。
因此 `--source` 应选择原始 0731，不能选择量化目录，也不能只复制三个 MTP 文件到目标根目录。

脚本只使用 Python 3.10+ 标准库，不需要安装 torch、safetensors，也不使用 GPU/NPU。
权重载荷以最多 8 MiB 的分块读取；不反量化、不重新量化、不修改张量名称、dtype、shape 或载荷字节。
打包后逐张量回读并比较 SHA256，全部通过后才将临时目录发布为 `dspark/`。
已存在的输出目录会被拒绝，目标模型的配置、索引和已有权重均不修改。

在同时能访问原始模型与目标目录的服务器上执行：

```bash
ORIGINAL=/path/to/DeepSeek-V4-Flash-0731
MODEL=/path/to/DeepSeek-V4-Flash-vq2a8-mixed-gs32-repack

# 先检查配置、完整草稿字段及分片文件头，并显示空间需求；不写文件。
python tools/repack_vq2a8_dspark.py \
  --source "$ORIGINAL" --target "$MODEL" --output "$MODEL/dspark" --dry-run

# 正式提取并校验。输出还包含约 1.97 GiB 的原始 embedding/head，合计约 12.09 GiB。
python tools/repack_vq2a8_dspark.py \
  --source "$ORIGINAL" --target "$MODEL" --output "$MODEL/dspark"

# 搬运到推理服务器后可再次校验；不再需要原始模型目录。
python tools/repack_vq2a8_dspark.py --output "$MODEL/dspark" --verify-only
```

成功输出 `DSPARK_REPACK_STATUS=verified`，校验清单为
`$MODEL/dspark/dspark_repack_report.json`，包括每个张量的来源、dtype、shape 和 SHA256。
`--target` 用于检查目标/草稿的模型尺寸与主体 attention 压缩布局是否一致；
若在另一台机器提前打包，可省略它并将 `--output` 指向新的独立目录，随后搬运整个草稿目录。
结构一致不能证明训练来源一致，仍须选择目标量化模型对应的原始 0731。

输出的 safetensors 分片按实际来源重新编号；只提取选中的张量，不复制原始主体层。
已检查的 0731 需要读取原始第 1、45、46、47、48 个分片，生成五个草稿包分片。
不需要把其余原始主体分片复制到推理服务器，也不要把草稿张量加入目标的根索引。
失败时保留 `.dspark.repacking-*` 临时目录，错误信息会打印路径；正式 `dspark/` 不会发布。

这个包继续使用现有 DSpark 模型、量化配置隔离和标准 safetensors 加载器。
`verified` 表示打包完整性及字节校验通过；实际服务生成、图回放与精度仍按下文在 NPU 上验收。

## 先测性能：启动 B4 服务

以下命令均在仓库根目录执行。使用已经 repack 的目标模型及 `$MODEL/dspark`，
不再需要在推理服务器部署完整原始模型。先确认所选卡空闲。

终端 A：

```bash
MODEL=/mnt/share/g00872988/DeepSeek-V4-Flash-0731-vq2a8-mixed-gs32-repack
BUILD="$PWD/build/vq2a8-gs32-runtime-ffn"
mkdir -p benchmark-results/performance-vq2a8-dspark
set -o pipefail

ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 VLLM_USE_V2_MODEL_RUNNER=1 \
bash tools/tests/serve_vq2a8_dspark_performance.sh \
  --model "$MODEL" \
  --gs32-runtime-dir "$BUILD" \
  2>&1 | tee benchmark-results/performance-vq2a8-dspark/serve-dp4-b4-dspark5.log
```

脚本复用 `tools/serve_vq2a8.py`，默认参数如下；追加同名参数可以覆盖：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| 草稿 | `--dspark` | 从 `$MODEL/dspark` 加载，DSpark5 |
| 并行 | `--parallel-mode dp-ep --ep-size 4` | TP1、DP4、EP4 |
| 图执行 | 默认完整 FFN、decode 图回放 | 目标 decode 图及原始草稿图路径 |
| 序列上限 | `--max-num-seqs 4` | 每 engine B4，全局上限 16 |
| 上下文上限 | `--max-model-len 8192` | 每个请求输入与输出的总 token 上限 |
| 调度预算 | `--max-num-batched-tokens 8192` | 每轮调度的 token 预算，保持 8192 |
| KV 预算 | `--kv-cache-mib 16384 --reserve-gib 16` | 每 worker 显式 KV 预算及其校验上限 |
| 接口 | `127.0.0.1:8003`，模型名 `vq2a8` | 用于后续客户端测试 |

DSpark5 在 B4 下的目标验证图桶为 `[6,12,24]`；草稿每条序列处理 5 个 query token。
可在启动命令末尾追加 `--dry-run` 查看最终参数，不加载模型。

## 一键性能测试：并发 1、2、4

等待终端 A 显示服务就绪，另开终端 B：

```bash
MODEL=/mnt/share/g00872988/DeepSeek-V4-Flash-0731-vq2a8-mixed-gs32-repack

python tools/tests/run_vq2a8_dspark_performance.py \
  --base-url http://127.0.0.1:8003 --model vq2a8 --tokenizer "$MODEL" \
  --variant dspark --parallel-mode dp-ep --ep-size 4 --max-num-seqs 4 \
  --concurrency 1 2 4 --num-prompts 4 --repeats 1 \
  --input-len 128 --output-len 512 \
  --result-dir benchmark-results/performance-vq2a8-dspark \
  --note 'DSpark5; max_model_len=8192; max_num_batched_tokens=8192; KV=16384MiB; full_graph'
```

脚本先检查 `/health`、`/v1/models`，再用 `Hello` 做一次 16-token 真实生成检查。
随后顺序调用已有 `run_vq2a8_throughput.py`，每个阶段都测 C=1、2、4：

| 阶段 | 每请求输入 / 输出 | 每档正式请求 | 每档测量轮数 |
| --- | --- | --- | --- |
| `smoke-throughput` | 16 / 16 token | 4 | 1 |
| `throughput` | 128 / 512 token | 4 | 1 |

每档另有 C 条预热请求，不计入正式 TPS；vLLM bench 自身的预检也会产生额外请求。
长输出阶段每档正式生成 `4 × 512 = 2048` token。
`--repeats 1` 表示每档只测一轮；`--input-len`、`--output-len` 只调整第二阶段。
使用固定随机种子、`temperature=0` 和 `ignore_eos`，保持各档输出长度一致。
任一就绪检查、生成检查或吞吐试验失败即停止；请求失败、输出长度不足不能算成功吞吐。
追加 `--dry-run` 可查看全部命令，不访问服务、不创建结果。

终端打印每档 output TPS 和结果目录；目录按运行自动隔离。
顶层 `run.json` 记录阶段状态，`models.json`、`smoke.json` 留存预检响应；
各阶段子目录中保留原始 bench JSON、日志、`summary.json` 和 `summary.csv`，
包括总 output TPS、TTFT、TPOT 和尾延迟。
`DSPARK_PERFORMANCE_STATUS=passed` 表示上述请求与结果校验通过，不代表完成精度验收。

C 是全局客户端并发，DP4 会将请求分给不同 engine，因此 C=4 不等于每个 engine 达到 B4。
客户端的 `--variant`、并行参数和 `--note` 只记录实验配置，不会修改或自动验证服务配置；
是否启用 DSpark 应同时核对服务启动日志中的 `speculative_config`。

### 普通解码对照

停止终端 A 的服务并等待 worker 退出，用相同卡、目标模型、并行拓扑和预算重启，
只追加 `--no-dspark`。此时目标图桶恢复为 `[1,2,4]`，不加载草稿：

```bash
MODEL=/mnt/share/g00872988/DeepSeek-V4-Flash-0731-vq2a8-mixed-gs32-repack
BUILD="$PWD/build/vq2a8-gs32-runtime-ffn"
set -o pipefail

ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 VLLM_USE_V2_MODEL_RUNNER=1 \
bash tools/tests/serve_vq2a8_dspark_performance.sh \
  --model "$MODEL" \
  --gs32-runtime-dir "$BUILD" \
  --no-dspark \
  2>&1 | tee benchmark-results/performance-vq2a8-dspark/serve-dp4-b4-baseline.log
```

终端 B 重复上述一键测试命令，将 `--variant dspark` 改为 `--variant baseline`，
并将 `--note` 中的 `DSpark5` 改为 `no DSpark`。
若自行指定过 `--dspark-model` 或 `--num-speculative-tokens`，对照时也须移除；
与 `--no-dspark` 同时传入会报参数冲突。
比较同一负载、同一 C 的总 output TPS 和 TTFT/TPOT：
吞吐比为 `DSpark output TPS / baseline output TPS`。
此前 TP2/EP2 的结果与这里 DP4/EP4 拓扑不同，不能直接作为 DSpark 加速比的分母。
每档 4 请求、1 轮用于快速比较；结果接近时再增加轮数确认波动。

### 可选：长输入性能

短负载完成后，需要测 64K、256K 输入时，先停止当前服务，在 DSpark 启动命令中追加
`--max-model-len 263168` 后重启：`262144` 输入加 `1024` 输出需要这个总长度上限。
仍保持 `--max-num-seqs 4 --max-num-batched-tokens 8192`，以实际 KV 容量检查结果为准。
服务就绪后复用现有长输入测试：

```bash
# 先检查输入长度和服务声明的上下文容量，不生成。
python tools/tests/run_vq2a8_performance_suite.py \
  --base-url http://127.0.0.1:8003 --model vq2a8 \
  --cases 64k 256k recovery --concurrency 1 2 4 \
  --label dspark5-dp4-b4-long --check-only \
  --result-dir benchmark-results/performance-vq2a8-dspark

# 每个长输入档：C=1、2 各测 2 请求，C=4 测 4 请求，才能实际覆盖并发 4。
python tools/tests/run_vq2a8_performance_suite.py \
  --base-url http://127.0.0.1:8003 --model vq2a8 \
  --cases 64k 256k recovery --concurrency 1 2 4 \
  --label dspark5-dp4-b4-long \
  --result-dir benchmark-results/performance-vq2a8-dspark
```

`recovery` 用短请求检查长输入后的恢复；本组主要记录 TTFT/TPOT，结果与前面的总 TPS 分别查看。
这里不传 `--repeats`，由长输入脚本使用默认的 2/2/4 请求数；
该旧脚本的 `--repeats` 表示请求数，与吞吐脚本的“重复测量轮数”含义不同。
做长输入 baseline 对照时也必须使用相同的 `max_model_len=263168` 和其余服务配置。

## 性能检查后：长上下文精度服务

先停止性能服务并等待 worker 退出。下面单独使用 B16、长上下文配置做精度与硬件验收，
不与前面的 B4 性能结果混用；不同端口的服务仍会共享相同卡的显存。

```bash
MODEL=/mnt/share/g00872988/DeepSeek-V4-Flash-0731-vq2a8-mixed-gs32-repack
DRAFT_MODEL="$MODEL/dspark"
BUILD="$PWD/build/vq2a8-gs32-runtime-ffn"
mkdir -p benchmark-results/accuracy-vq2a8-dspark
set -o pipefail

ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 VLLM_USE_V2_MODEL_RUNNER=1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -u tools/serve_vq2a8.py \
  --model "$MODEL" \
  --dspark-model "$DRAFT_MODEL" \
  --num-speculative-tokens 5 \
  --gs32-runtime-dir "$BUILD" \
  --parallel-mode dp-ep --ep-size 4 \
  --max-model-len 262144 \
  --max-num-seqs 16 --max-num-batched-tokens 2048 \
  --kv-cache-mib 16384 --reserve-gib 16 \
  --tokenizer-mode deepseek_v4 --reasoning-parser deepseek_v4 \
  --host 127.0.0.1 --port 8003 \
  2>&1 | tee benchmark-results/accuracy-vq2a8-dspark/serve-dp4-b16-dspark5.log
```

改成 `--parallel-mode tp-ep --ep-size 4` 即为 TP4+EP4、DP1。
同样设置 `--max-num-seqs 16` 时，TP4 服务的全局活动序列上限为 16，
DP4 服务为每个 engine 16、全局最多 64。客户端并发仍是单独设置，
请求分配和上下文容量可能使实际 batch 小于该上限。

DSpark5 每个目标验证序列最多包含 6 个 query token，B16 的目标图桶自动设为
`[6,12,24,48,96]`，最大图 token 数为 96；这不表示允许 96 个活动序列。
草稿每个序列处理 5 个 query token，B16 草稿图对应 80 个 token，
通过 `enforce_eager=false` 复用原始模型的草稿图管理器。
DP4 时专家需要容纳各 engine 的验证 token，路由容量还需乘以 DP 数和 top-k。
`max_num_batched_tokens` 必须不小于 `max_num_seqs × 6`。
目标模型保留 `compilation mode=0` 和 `enable_npugraph_ex=false`，
不要直接复制原始 baseline 的编译配置或启用共享专家多流重叠。

`max_model_len=262144` 是输入与输出总长度上限，为 131072 输出 token 留出输入空间。
16 GiB 是每个 worker 的显式 KV 预算，不保证 B16 的 256K 上下文都能驻留；
草稿权重、草稿 KV 及额外验证图也会消耗内存。以启动时容量检查和实测为准。
GS32 当前还会为每层、每个捕获桶保留中间张量，开销随路由容量和捕获桶总大小增长；
DP 与 DSpark 会进一步放大路由容量。三专家 bank 测试通过不能证明整模型 B16/B64 的显存容量。
整模型验收需同时记录捕获前后的 allocated/reserved 显存和峰值；图池复用优化需另行验证
所有桶的交替回放及空 rank 恢复，不能直接删除 native 句柄或捕获缓冲区的引用。
`--reserve-gib` 当前仅校验显式 KV 预算的上限，不会实际预留显存，也不估算草稿或图的开销；
不能据此认为 16 GiB 预算已覆盖全部额外开销。草稿只加载 MTP 与所需 embedding/head，
不应将整套原始 checkpoint 的文件大小直接作为草稿驻留显存。
使用完整原始 checkpoint 时，标准权重加载器仍会遍历其主体分片文件；
使用上述独立草稿包时只遍历打包后的分片，无须部署完整原始 checkpoint。
两种方式都保留原始草稿格式；磁盘读取量与草稿最终驻留显存是两个不同指标。
当前 GS32 每层量化校验还会重复读取草稿配置与索引、检查分片是否存在，
共享存储上的启动耗时需单独测量；后续可复用一次性 checkpoint 校验，避免增加全局缓存。

## GPQA 精度请求检查

服务就绪后，另开终端运行：

```bash
python tools/tests/run_vq2a8_accuracy_suite.py \
  --base-url http://127.0.0.1:8003 \
  --datasets gpqa --thinking --reasoning-effort high \
  --offline-data-dir /mnt/share/g00872988/vq2a8-eval-data \
  --limit 2 --concurrency 2 \
  --temperature 0 --top-p 1 --seed 0 --repeats 1 \
  --max-tokens 131072 --timeout 14400 \
  --label vq2a8-dp4-dspark5-thinking-smoke \
  --result-dir benchmark-results/accuracy-vq2a8-dspark
```

no-thinking 可改为 `--no-thinking`，同时去掉 `--reasoning-effort high`。
输出预算由客户端独立指定；例如先用 `--max-tokens 8192` 检查 no-thinking。
两条请求只能证明接口完成了生成，不能证明每个 rank 达到 B16、验证图被回放或精度达标。

正式精度对比应分别运行 DSpark 开关两组：
保持目标 checkpoint、拓扑、输入顺序、thinking 模式、采样参数和输出预算一致，
关闭 DSpark 时从服务命令移除 `--dspark-model` 和 `--num-speculative-tokens`。
移除评测端 `--limit` 才是完整 GPQA；保留原始预测、截断数量及错误记录。
若同时比较原始 baseline 与 VQ2A8，应明确分别记录是否启用 DSpark。

## 独立硬件验收

该验收不加入默认一键 CI。每个拓扑顺序加载 DSpark 关闭和开启两个服务引擎，
使用短输入、B16 平衡负载、非均衡负载及单请求恢复；不会测量 128K 输出性能。
需要四张空闲 NPU、独立草稿包（或完整原始 checkpoint）和已构建的 GS32 算子。

```bash
MODEL=/mnt/share/g00872988/DeepSeek-V4-Flash-0731-vq2a8-mixed-gs32-repack
DRAFT_MODEL="$MODEL/dspark"
BUILD="$PWD/build/vq2a8-gs32-runtime-ffn"

ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 VLLM_USE_V2_MODEL_RUNNER=1 \
python -m pytest -vv -s -x \
  'tests/e2e/nightly/test_vq2a8_gs32_dspark.py::test_gs32_dspark_matches_target_and_replays_verification_graphs[dp4]' \
  --vq2a8-model "$MODEL" \
  --vq2a8-dspark-model "$DRAFT_MODEL" \
  --vq2a8-gs32-library "$BUILD/libvq2a8_gs32_stage.so" \
  --vq2a8-gs32-binary "$BUILD/mixed_w2w4_moe_ffn.o"
```

将用例末尾 `[dp4]` 改为 `[tp4]` 验收 TP4+EP4。
通过标准包括：

- 每个 EP rank 的目标模型加载 GS32 bank，草稿保留原始量化格式且没有 GS32 bank。
- 有真实草稿调用、目标验证 token 和拒绝采样执行；不是仅构造 speculative 配置。
- 目标与草稿 decode 图分别发生实际回放，平衡负载达到每 engine B16，
  并回放目标 96-token 验证图和草稿 80-token 图。
- DSpark 开关两组的贪心输出 token 一致，top-logprobs 满足已有容差。
- 空闲 DP rank 参与通信，非均衡请求及请求结束后的恢复不死锁。

原始日志与中间 JSON 保留在 pytest 临时目录，失败信息给出具体子进程日志路径。
前面的性能快测用于尽早观察吞吐变化；正确性、图回放和完整精度仍须独立验收。
扩大[吞吐扫描](vq2a8_batch_throughput.md)前，先确认这些检查通过，
并单独记录最大稳定并发与长上下文容量。
