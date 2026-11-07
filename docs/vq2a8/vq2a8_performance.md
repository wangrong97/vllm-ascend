# VQ2A8 一键性能测试

使用已启动的 VQ2A8 服务，测试短输入、64K、256K、1M 输入的首 token 时延
（TTFT）和后续 token 平均时延（TPOT）。下面的一键命令扫描客户端并发 1、2、4：
短输入每档 4 个测量请求；长输入在并发 1、2 时每档测 2 个请求，并发 4 时测最少的 4 个请求。
各输入和并发档位依次执行，同一档内并发发送请求。客户端不加载模型，也不启动或重启服务。
所有命令均在仓库根目录执行。

其他指南：[并发吞吐测试](vq2a8_batch_throughput.md) ·
[NPU CI 回归](vq2a8_npu_ci.md) · [精度评测](vq2a8_accuracy.md) ·
[框架接入说明](vq2a8_gs32.md)。

## One-command performance suite

### 1. 指定 NPU 卡并启动服务

下面用 **物理卡 4、5，MRV2、EP2、B4、chunk8192、fusion=none** 启动一个 256K 服务，
可以依次完成整个 `standard`，包括里面的短输入和 64K 输入，无须按长度重启。
`MODEL` 指向已准备好的模型目录；`BUILD` 指向已有 `.so` 和 `.o` 的实际目录，
这个变量本身不会创建或编译任何文件。

在终端 A 执行；将示例绝对路径替换成实际路径：

```bash
export MODEL=/path/to/DeepSeek-V4-Flash-vq2a8-mixed-gs32-repack
export BUILD=/path/to/existing/gs32-artifacts

mkdir -p benchmark-results/performance
set -o pipefail

ASCEND_RT_VISIBLE_DEVICES=4,5 VLLM_USE_V2_MODEL_RUNNER=1 \
bash tools/tests/serve_vq2a8_256k_1K.sh \
  --model "$MODEL" \
  --gs32-library "$BUILD/libvq2a8_gs32_stage.so" \
  --gs32-binary "$BUILD/mixed_w2w4_moe_ffn.o" \
  --gs32-fusion none \
  --max-model-len 263168 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 8192 \
  --kv-cache-mib 16384 \
  2>&1 | tee benchmark-results/performance/serve-mrv2-ep2-chunk8192-none.log
```

`ASCEND_RT_VISIBLE_DEVICES=4,5` 选择两张服务用卡；改成 `0,1` 即使用卡 0、1。
VQ2A8 启动器默认使用 MRV2；显式设置 `VLLM_USE_V2_MODEL_RUNNER=0` 并重启可切换至 MRV1。
脚本保留显式设备和 runner 选择。默认地址为 `http://127.0.0.1:8003`，服务模型名为 `vq2a8`。

该启动脚本使用 `full_graph`，上面的 `--max-num-seqs 4` 启用 **B1/B2/B4 decode 图回放，prefill 仍为 eager**；
同时关闭 prefix caching，最多调度 4 个活动序列。不加该覆盖参数时，脚本仍默认 B1。
`--max-num-batched-tokens 8192`
是每个调度步的 token 预算，不要求小请求补成 8192 个真实输入 token。
256K 脚本默认 chunk 为 4096，以上命令显式覆盖为 8192。

`--kv-cache-mib 16384` 为每个 worker 配置 16 GiB KV cache。
脚本中的 `--reserve-gib 16` 只约束这个参数的合法上限，不额外分配 16 GiB。
实际 KV 容量和运行时内存是否够用，由服务启动检查及实际请求验证。

### MRV2 多序列 decode

上述 B4 配置自动捕获 `[1,2,4]`。将 `--max-num-seqs 4` 改为 `--max-num-seqs 64` 即可配置 B1～B64；
启动器自动捕获 `[1,2,4,8,16,32,64]`，不必为每个 batch 单独启动服务或指定图模式。
也可设为 7 等非 2 的幂上限，自动捕获 `[1,2,4,7]`。实际 batch 在区间内变化时，
由调度器补齐到对应图桶；每步 token 预算必须不小于最大活动序列数。
MRV1 继续保留 B1，新增多序列图要求 MRV2；使用 MRV1 时，服务和客户端分别选择 `--max-num-seqs 1`、`--concurrency 1`。

多图捕获和并发长序列需要额外内存，`--kv-cache-mib 16384` 不保证 64 路 256K 请求可同时运行。
先用短输入验证实际吞吐、KV 占用和请求排队，再增加长度。可使用
[精度脚本](vq2a8_accuracy.md)的 `--concurrency 4`、`--concurrency 8` 或 `--concurrency 64` 发起并发请求。
客户端并发不等于每步 decode batch，完整图覆盖应通过
[独立 batch smoke 测试](vq2a8_npu_ci.md#多序列-decode-图验收)检查实际回放桶。
不同并发下的吞吐请使用[并发吞吐测试](vq2a8_batch_throughput.md)中的服务启动命令和扫描脚本。
下面的 `run_vq2a8_performance_suite.py` 按并发档位分别统计每条请求的 TTFT/TPOT，
客户端并发不保证每步实际 decode batch 等于该值，也不验证具体图桶覆盖。

### 2. 服务就绪后预检并测试

等待终端 A 出现 HTTP server started，在终端 B 的同一仓库根目录执行：

```bash
# 确认服务就绪及模型名称
curl -fsS http://127.0.0.1:8003/health
curl -fsS http://127.0.0.1:8003/v1/models

# 只做全部 case 的容量与提示词预检，不发送生成请求
python tools/tests/run_vq2a8_performance_suite.py \
  --base-url http://127.0.0.1:8003 --model vq2a8 \
  --suite standard --concurrency 1 2 4 --check-only

# 扫描 standard：所有输入均测 C=1/2/4；长输入对应请求数为 2/2/4
python tools/tests/run_vq2a8_performance_suite.py \
  --base-url http://127.0.0.1:8003 --model vq2a8 \
  --suite standard --concurrency 1 2 4 --label mrv2-ep2-b4-chunk8192-none \
  --result-dir benchmark-results/performance
```

正常运行本身也包含预检，所以单独的 `--check-only` 是可选步骤。
`--label` 仅为操作者标注，不会设置或验证服务的 runner、卡号、chunk 或 fusion。
修改这些服务参数需要重启服务，再给测试结果使用对应 label。

## 套件和服务容量

| 套件 | 按执行顺序包含的 case | 最小总上下文容量 |
| --- | --- | ---: |
| `quick` | 128/127/129/224 输入各输出 32；然后 128 输入、128 输出 | 256 |
| `standard`（默认） | `quick`，64K/1K，256K/1K，短输入恢复测试 | 263168 |
| `full` | `quick`，64K/1K，256K/1K，1M/1K，短输入恢复测试 | 1049600 |

容量为**输入加输出**。64K=65536，256K=262144，1M=1048576，输出 1K=1024。
`--max-model-len` 决定总上下文上限；`--max-num-batched-tokens` 决定单步预算，
两者可以不同。长输入分块处理，小输入按实际调度量处理。
同一服务可以测所有容量以内的长度；这是同一配置下的横向比较，不代表每个长度的最优配置。

每个输入 case、每个实际执行的并发档位分别采用以下请求次数：

- 四个短边界 case：无 warmup，每档测量 4 个请求。
- `128x128`：warmup 1 个请求，每档测量 4 个请求。
- 每个长输入 case：无 warmup，C=1、2 每档测量 2 个请求，C=4 测量 4 个请求。
- `recovery`：128 输入、128 输出，无 warmup，每档测量 4 个请求，观察长请求后的短请求时延。

这里的 `--repeats` 表示每档的测量请求总数，不是将整档重复若干轮；不会再乘以并发数。
例如短输入 C=4 是同时发送 4 个请求，总共测量 4 个；C=2 是最多同时发送 2 个，总共测量 4 个。
默认请求数不会小于所选并发，因此长输入 C=4 自动使用 4 个请求，所有长度都覆盖 C=1、2、4。
显式指定 `--repeats` 时，必须不小于最大的所选并发；例如 `--concurrency 1 2 4 --repeats 2`
会在发送请求前报错，避免出现标注 C=4 却只发 2 个请求的情况。
不传 `--concurrency` 时保留原有串行 C=1 行为。

全部 case 先完成预检，再开始任何生成请求。只要一个预检失败，整套不发送生成请求，
返回非零退出码；脚本不会自动缩短输入。服务报告的上下文容量足够，也不等于运行时内存一定足够。
原有 `run_vq2a8_ep2_long_context_latency.py` 单独入口仍保留其逐 case 执行行为，
与此套件的整体预检策略不同。

## 包含 1M 输入的 full 测试

先在终端 A 用 Ctrl+C 结束前述服务，等待 worker 退出。
在同一终端沿用已经设置的 `MODEL`、`BUILD`、结果目录和 `pipefail`，改用 1M 启动脚本：

```bash
ASCEND_RT_VISIBLE_DEVICES=4,5 VLLM_USE_V2_MODEL_RUNNER=1 \
bash tools/tests/serve_vq2a8_1M_1K.sh \
  --model "$MODEL" \
  --gs32-library "$BUILD/libvq2a8_gs32_stage.so" \
  --gs32-binary "$BUILD/mixed_w2w4_moe_ffn.o" \
  --gs32-fusion none \
  --max-model-len 1049600 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 8192 \
  --kv-cache-mib 16384 \
  2>&1 | tee benchmark-results/performance/serve-mrv2-ep2-1m-chunk8192-none.log
```

1M 脚本默认 chunk 为 128，命令显式改为 8192，保持与上面的性能对照一致。
模型合法上下文和实际 KV 容量必须支持 **1049600 总 token**；16 GiB KV 只是脚本配置值，
不能保证任意模型和运行环境都能完成 1M 测试，也不代表已验证 1M 回答质量。

服务就绪后，在终端 B 执行：

```bash
python tools/tests/run_vq2a8_performance_suite.py \
  --base-url http://127.0.0.1:8003 --model vq2a8 \
  --suite full --concurrency 1 2 4 --check-only

python tools/tests/run_vq2a8_performance_suite.py \
  --base-url http://127.0.0.1:8003 --model vq2a8 \
  --suite full --concurrency 1 2 4 --label mrv2-ep2-b4-1m-chunk8192-none \
  --result-dir benchmark-results/performance
```

如果模型只允许 1048576 总上下文，把服务命令中的 `--max-model-len` 改为 `1048576`，
客户端明确选择 `--cases 1m-window recovery`。
`1m-window` 使用 1047552 输入加 1024 输出，不能标成完整 1M 输入；
脚本不会把 `full` 中的 `1m` 自动替换成它。

## 挑选 case 和控制次数

```bash
# 离线查看 full 执行计划，不访问服务、不创建结果目录
python tools/tests/run_vq2a8_performance_suite.py --suite full --concurrency 1 2 4 --dry-run

# 单独选择长输入：并发 1、2、4，对应测量请求数 2、2、4
python tools/tests/run_vq2a8_performance_suite.py \
  --base-url http://127.0.0.1:8003 --model vq2a8 \
  --cases 256k --concurrency 1 2 4 --warmups 0 \
  --timeout 21600 --label mrv2-ep2-chunk8192-none \
  --result-dir benchmark-results/performance

# 1048576 总上下文窗口：输入长度明确减少 1024
python tools/tests/run_vq2a8_performance_suite.py \
  --base-url http://127.0.0.1:8003 --model vq2a8 \
  --cases 1m-window recovery --concurrency 1 2 4 --label mrv2-ep2-b4-1m-window-chunk8192-none \
  --result-dir benchmark-results/performance
```

`--cases` 覆盖套件选择和顺序，case 名称可通过 `--help` 查看。
`--timeout` 是每次 HTTP socket 超时，不是整个套件的总时限。

## 结果与比较口径

每次运行在 `--result-dir` 下创建独立子目录，输出 `summary.json`、`summary.csv`、
整套客户端日志 `run.log`，以及每个输入/并发组合的 JSON 和日志。C=1 保留原文件名，
C=2、4 分别加上 `-c2`、`-c4`，例如 `128x128-c4.json`，避免不同并发互相覆盖。
终端会打印并发列和汇总文件路径。
默认实时显示并同步刷新日志文件，无须另外加 `tee`：`PREFLIGHT`、case 结果和汇总走 stdout，
请求的 `START`、`FIRST_CONTENT`、`DONE` 进度走 stderr，两者均写入日志。
原有长输入单独入口也采用同样的双输出方式。

`START` 在发送请求前输出，`FIRST_CONTENT` 在首个内容到达时输出，不等待整条请求完成；
等待首 token 期间没有新的客户端事件时，终端可能保持安静。
日志不逐 token 打印生成内容，也不转发服务端 worker 日志；服务日志仍在终端 A 单独保存。
TTFT/TPOT 的计时点不变，日志仅增加少量客户端 I/O；比较版本时应保持相同的日志输出方式。

| 字段或记录 | 含义 |
| --- | --- |
| `workload` / `concurrency` | 原始输入 case 名称 / 客户端并发上限，不是实测 decode batch |
| `mean_ttft_ms` / `median_ttft_ms` | 测量请求首个内容到达的时延均值 / 中位数 |
| `mean_tpot_ms` / `median_tpot_ms` | 测量请求 TPOT 的均值 / 中位数；使用实际 completion token 数，不按 SSE 消息数量计数 |
| CSV `first_ttft_ms` | 该组合编号为 1 的请求的 TTFT；若有 warmup，则来自第一条 warmup，不按完成顺序选择 |
| `warmup_requests` / `requests` | 分别保留预热请求与测量请求；均值和中位数排除 warmup |
| `client_git_revision` | 测试客户端的提交 SHA，不代表服务端代码或算子版本 |

分词和提示词构造不计入请求时延；返回的输入、输出 token 数必须与请求一致。
每个输入/并发组合独立统计，不将不同长度和并发合成一个平均值。
第一条请求不等于冷启动测量：此前的请求、启动预热和编译缓存都可能影响结果。

检测到某条请求失败后停止提交新请求，终止该档仍在运行的客户端，剩余组合标记为 `not_run`；
不完整组合保留已收到的结果，不输出成功平均值。并发请求使用独立客户端进程，创建进程不计入单请求 TTFT/TPOT。
Ctrl+C 保留部分结果并返回 130，终止仍在运行的客户端；客户端断开不证明服务端计算已立即取消。
**退出码 0 表示所选输入/并发组合全部完成，不表示性能达到门槛。** 当前没有自动性能回归阈值。
`recovery` 仅检查完成情况和时延，模型回答正确性请使用精度评测指南。

跨版本比较时，保持硬件、模型、算子产物、runner、EP、chunk、fusion、KV、缓存、并发和请求数设置一致，
避免其他请求或 CI 同时占用测试卡，并单独保存服务启动日志和版本信息。
启动耗时、首次请求、稳态 TTFT/TPOT 分别比较。
局部 FFN benchmark 不包含 attention、HCCL、路由和调度，不能直接作为端到端加速结论。
此前长 prefill 回归的最终结论仍需目标 NPU 上相同条件的服务测试，本指南不承诺具体加速幅度。
