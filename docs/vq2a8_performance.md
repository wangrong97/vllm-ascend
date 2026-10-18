# VQ2A8 一键性能测试

使用已启动的 VQ2A8 服务，串行测试短输入、64K、256K、1M 输入的首 token 时延
（TTFT）和后续 token 平均时延（TPOT）。客户端不加载模型，也不启动或重启服务。
所有命令均在仓库根目录执行。

其他指南：[NPU CI 回归](vq2a8_npu_ci.md) · [精度评测](vq2a8_accuracy.md) ·
[框架接入说明](vq2a8_gs32.md)。

## One-command performance suite

### 1. 指定 NPU 卡并启动服务

下面用 **物理卡 4、5，MRV1、EP2、chunk8192、fusion=none** 启动一个 256K 服务，
可以依次完成整个 `standard`，包括里面的短输入和 64K 输入，无须按长度重启。
`MODEL` 指向已准备好的模型目录；`BUILD` 指向已有 `.so` 和 `.o` 的实际目录，
这个变量本身不会创建或编译任何文件。

在终端 A 执行；将示例绝对路径替换成实际路径：

```bash
export MODEL=/path/to/DeepSeek-V4-Flash-vq2a8-mixed-gs32-repack
export BUILD=/path/to/existing/gs32-artifacts

mkdir -p benchmark-results/performance
set -o pipefail

ASCEND_RT_VISIBLE_DEVICES=4,5 VLLM_USE_V2_MODEL_RUNNER=0 \
bash tools/tests/serve_vq2a8_256k_1K.sh \
  --model "$MODEL" \
  --gs32-library "$BUILD/libvq2a8_gs32_stage.so" \
  --gs32-binary "$BUILD/mixed_w2w4_moe_ffn.o" \
  --gs32-fusion none \
  --max-model-len 263168 \
  --max-num-batched-tokens 8192 \
  --kv-cache-mib 16384 \
  2>&1 | tee benchmark-results/performance/serve-mrv1-ep2-chunk8192-none.log
```

`ASCEND_RT_VISIBLE_DEVICES=4,5` 选择两张服务用卡；改成 `0,1` 即使用卡 0、1。
`VLLM_USE_V2_MODEL_RUNNER=0` 选择 MRV1，改成 `1` 并重启服务才会使用 MRV2。
脚本保留这些显式环境设置。默认地址为 `http://127.0.0.1:8000`，服务模型名为 `vq2a8`。

该启动脚本使用 `full_graph`，其实际含义是 **B1 decode 图回放，prefill 仍为 eager**；
同时关闭 prefix caching，最多运行一个活动序列。`--max-num-batched-tokens 8192`
是每个调度步的 token 预算，不要求小请求补成 8192 个真实输入 token。
256K 脚本默认 chunk 为 4096，以上命令显式覆盖为 8192。

`--kv-cache-mib 16384` 为每个 worker 配置 16 GiB KV cache。
脚本中的 `--reserve-gib 16` 只约束这个参数的合法上限，不额外分配 16 GiB。
实际 KV 容量和运行时内存是否够用，由服务启动检查及实际请求验证。

### 2. 服务就绪后预检并测试

等待终端 A 出现 HTTP server started，在终端 B 的同一仓库根目录执行：

```bash
# 确认服务就绪及模型名称
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models

# 只做全部 case 的容量与提示词预检，不发送生成请求
python tools/tests/run_vq2a8_performance_suite.py \
  --base-url http://127.0.0.1:8000 --model vq2a8 \
  --suite standard --check-only

# 串行执行完整 standard
python tools/tests/run_vq2a8_performance_suite.py \
  --base-url http://127.0.0.1:8000 --model vq2a8 \
  --suite standard --label mrv1-ep2-chunk8192-none \
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

默认请求次数如下：

- 四个短边界 case：无 warmup，每个测量 2 次。
- `128x128`：warmup 1 次，测量 10 次。
- 每个长输入 case：无 warmup，测量 2 次。
- `recovery`：128 输入、128 输出，无 warmup，测量 3 次，观察长请求后的短请求时延。

全部 case 先完成预检，再开始任何生成请求。只要一个预检失败，整套不发送生成请求，
返回非零退出码；脚本不会自动缩短输入。服务报告的上下文容量足够，也不等于运行时内存一定足够。
原有 `run_vq2a8_ep2_long_context_latency.py` 单独入口仍保留其逐 case 执行行为，
与此套件的整体预检策略不同。

## 包含 1M 输入的 full 测试

先在终端 A 用 Ctrl+C 结束前述服务，等待 worker 退出。
在同一终端沿用已经设置的 `MODEL`、`BUILD`、结果目录和 `pipefail`，改用 1M 启动脚本：

```bash
ASCEND_RT_VISIBLE_DEVICES=4,5 VLLM_USE_V2_MODEL_RUNNER=0 \
bash tools/tests/serve_vq2a8_1M_1K.sh \
  --model "$MODEL" \
  --gs32-library "$BUILD/libvq2a8_gs32_stage.so" \
  --gs32-binary "$BUILD/mixed_w2w4_moe_ffn.o" \
  --gs32-fusion none \
  --max-model-len 1049600 \
  --max-num-batched-tokens 8192 \
  --kv-cache-mib 16384 \
  2>&1 | tee benchmark-results/performance/serve-mrv1-ep2-1m-chunk8192-none.log
```

1M 脚本默认 chunk 为 128，命令显式改为 8192，保持与上面的性能对照一致。
模型合法上下文和实际 KV 容量必须支持 **1049600 总 token**；16 GiB KV 只是脚本配置值，
不能保证任意模型和运行环境都能完成 1M 测试，也不代表已验证 1M 回答质量。

服务就绪后，在终端 B 执行：

```bash
python tools/tests/run_vq2a8_performance_suite.py \
  --base-url http://127.0.0.1:8000 --model vq2a8 \
  --suite full --check-only

python tools/tests/run_vq2a8_performance_suite.py \
  --base-url http://127.0.0.1:8000 --model vq2a8 \
  --suite full --label mrv1-ep2-1m-chunk8192-none \
  --result-dir benchmark-results/performance
```

如果模型只允许 1048576 总上下文，把服务命令中的 `--max-model-len` 改为 `1048576`，
客户端明确选择 `--cases 1m-window recovery`。
`1m-window` 使用 1047552 输入加 1024 输出，不能标成完整 1M 输入；
脚本不会把 `full` 中的 `1m` 自动替换成它。

## 挑选 case 和控制次数

```bash
# 离线查看 full 执行计划，不访问服务、不创建结果目录
python tools/tests/run_vq2a8_performance_suite.py --suite full --dry-run

# 单独选择 case，覆盖每个 case 的 warmup 和测量次数
python tools/tests/run_vq2a8_performance_suite.py \
  --base-url http://127.0.0.1:8000 --model vq2a8 \
  --cases 128x128 256k recovery --warmups 0 --repeats 2 \
  --timeout 21600 --label mrv1-ep2-chunk8192-none \
  --result-dir benchmark-results/performance

# 1048576 总上下文窗口：输入长度明确减少 1024
python tools/tests/run_vq2a8_performance_suite.py \
  --base-url http://127.0.0.1:8000 --model vq2a8 \
  --cases 1m-window recovery --label mrv1-ep2-1m-window-chunk8192-none \
  --result-dir benchmark-results/performance
```

`--cases` 覆盖套件选择和顺序，case 名称可通过 `--help` 查看。
`--timeout` 是每次 HTTP socket 超时，不是整个套件的总时限。

## 结果与比较口径

每次运行在 `--result-dir` 下创建独立子目录，输出 `summary.json`、`summary.csv`，
以及每个 case 的 JSON 和实时日志。终端会打印汇总文件路径。

| 字段或记录 | 含义 |
| --- | --- |
| `mean_ttft_ms` / `median_ttft_ms` | 测量请求首个内容到达的时延均值 / 中位数 |
| `mean_tpot_ms` / `median_tpot_ms` | 测量请求 TPOT 的均值 / 中位数；使用实际 completion token 数，不按 SSE 消息数量计数 |
| CSV `first_ttft_ms` | 该 case 第一条实际请求的 TTFT；若有 warmup，则来自第一条 warmup |
| `warmup_requests` / `requests` | 分别保留预热请求与测量请求；均值和中位数排除 warmup |
| `client_git_revision` | 测试客户端的提交 SHA，不代表服务端代码或算子版本 |

分词和提示词构造不计入请求时延；返回的输入、输出 token 数必须与请求一致。
每个 case 独立统计，不将短输入与长输入合成一个平均值。
第一条请求不等于冷启动测量：此前的请求、启动预热和编译缓存都可能影响结果。

某条请求失败后停止后续生成，剩余 case 标记为 `not_run`；不完整 case 不输出成功平均值。
Ctrl+C 保留部分结果并返回 130，客户端断开不证明服务端计算已立即取消。
**退出码 0 表示所选测试完成，不表示性能达到门槛。** 当前没有自动性能回归阈值。
`recovery` 仅检查完成情况和时延，模型回答正确性请使用精度评测指南。

跨版本比较时，保持硬件、模型、算子产物、runner、EP、chunk、fusion、KV 和缓存设置一致，
避免其他请求或 CI 同时占用测试卡，并单独保存服务启动日志和版本信息。
启动耗时、首次请求、稳态 TTFT/TPOT 分别比较。
局部 FFN benchmark 不包含 attention、HCCL、路由和调度，不能直接作为端到端加速结论。
此前长 prefill 回归的最终结论仍需目标 NPU 上相同条件的服务测试，本指南不承诺具体加速幅度。
