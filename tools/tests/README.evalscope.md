# VQ2A8 一键精度评测

通过已启动服务的 `/v1/chat/completions` 评测模型回答的正确率。
客户端使用固定版本 EvalScope 1.12.0 的数据适配器、提示词和规则评分，
不在客户端加载模型，不启动或重启推理服务，不调用额外裁判模型。
它与 NPU 算子回归 CI、TTFT/TPOT 性能测试互相补充。

| 参数 | 实际数据集 | 完整题数 |
| --- | --- | ---: |
| `gpqa` 或 `gpqa_diamond` | GPQA Diamond，`AI-ModelScope/gpqa_diamond` | 198 |
| `aime24` | AIME 2024，`evalscope/aime24` | 30 |
| `aime25` | AIME 2025，`evalscope/aime25` | 30 |

`gpqa` 明确指 Diamond 子集，不代表 GPQA 的所有子集。
默认运行表中的三个数据集，支持指定一个或多个；数据集别名不区分大小写。

## 安装与服务准备

在独立的 Python 3.10+ 客户端环境安装评测依赖。
客户端可以位于同一台服务器，也可以位于能够访问推理服务的其他机器：

```bash
python3 -m venv /path/to/evalscope-venv
source /path/to/evalscope-venv/bin/activate
python -m pip install -r tools/tests/requirements-evalscope.txt
```

离线机器需要提前准备对应 Python/操作系统的依赖安装包。
脚本不会自动安装依赖；已安装的 EvalScope 版本必须与固定版本一致。

使用现有服务启动脚本启动模型，然后确认 `/v1/models` 中的服务名称。
下面示例假设名称为 `vq2a8`、端口为 8000。
默认输出预算为 8192 tokens，服务上下文必须能够容纳题目、聊天模板和输出之和。
此前仅为短性能测试设置的 256-token 上下文不能用于这里的完整评测。
预检会拒绝完全没有提示词空间的输出预算；逐题是否超限仍由服务响应检查。

## 一键执行

```bash
# 三个数据集完整评测
python tools/tests/run_vq2a8_accuracy_suite.py \
  --base-url http://127.0.0.1:8000 \
  --model vq2a8 \
  --datasets gpqa aime24 aime25 \
  --concurrency 1 \
  --max-tokens 8192 \
  --result-dir benchmark-results/accuracy

# 只运行 AIME25；--limit 2 明确标记为 smoke，不能作为完整成绩
python tools/tests/run_vq2a8_accuracy_suite.py \
  --datasets aime25 --limit 2 --concurrency 1

# 只预览完整命令，不联网、不安装依赖、不创建结果目录
python tools/tests/run_vq2a8_accuracy_suite.py \
  --datasets gpqa aime24 --dry-run
```

也可以使用 `bash tools/tests/run_vq2a8_evalscope_accuracy.sh`，参数相同。
默认并发为 4；示例使用 1 便于在此前 B1 decode 服务上验证。
服务的活动序列数限制可能导致客户端请求排队。
请求默认超时 3600 秒，可用 `--timeout` 调整。

可配置 `--temperature`、`--top-p`、`--seed`、`--max-tokens`、`--repeats`。
默认温度为 0、seed 为 0、重复次数为 1，未指定的 top-p 沿用服务设置。
`--reasoning-effort none|low|high|max` 是针对当前 DeepSeek V4 服务的可选设置，
同时传递其聊天模板参数；不指定时沿用服务的思考配置。
这些是本项目回归评测条件，不宣称复现模型厂商公开榜单成绩。

## 内网离线数据

有网络的机器先使用相同 EvalScope 版本准备并导出完整数据集：

```bash
python tools/tests/prepare_evalscope_datasets.py \
  --datasets gpqa aime24 aime25 \
  --output-dir /path/to/vq2a8-eval-data
```

把整个导出目录复制到内网。数据包保留 EvalScope 原生缓存，附带题数、数据源、
版本和文件校验信息。评测时显式指定离线包：

```bash
python tools/tests/run_vq2a8_accuracy_suite.py \
  --base-url http://127.0.0.1:8000 \
  --model vq2a8 \
  --datasets gpqa aime24 aime25 \
  --offline-data-dir /mnt/share/vq2a8-eval-data \
  --concurrency 1 \
  --result-dir benchmark-results/accuracy
```

离线包缺失、版本不符、题数错误或文件校验失败会在发送评测请求前退出。
包中可以包含多个数据集，运行时仅选择其中需要的部分。
校验说明文件完整性，不能证明手工替换的数据仍属于官方数据集。

有网络时也可用 `--dataset-dir /path/to/cache` 共享普通缓存，
缺失数据会交给 EvalScope 下载。`--dataset-dir` 与 `--offline-data-dir` 不能同时使用。
预下载入口为 `prepare_evalscope_datasets.py` 或同名 `.sh`，不需要模型服务。

## 结果与退出状态

每次运行创建独立子目录，保留原始回答和评分依据：

```text
benchmark-results/accuracy/<timestamp-random>/
  run.json                 # 命令、环境、客户端提交、服务信息、实际参数和运行状态
  configs/                 # EvalScope 展开的配置
  logs/                    # EvalScope 日志
  predictions/             # 原始回答、停止原因和 token 用量
  reviews/                 # 逐题评分
  reports/                 # 原生准确率报告
  accuracy_audit.json       # 预测记录与报告的一致性审计
  summary.json             # 各数据集分数及评测口径
  summary.csv              # 便于对比的分数表
```

`score` 为 0 到 1 的准确率。默认使用原生 `accuracy:mean`；重复多次时计算所有
回答的平均正确率，不自动变成 pass@k。三个数据集分别报告，不混成一个总分。
`--limit` 运行的 `scope` 标记为 `smoke`，完整运行标记为 `full`。

脚本检查预测记录、重复索引、完成题数和原生评分报告是否一致。
子进程失败、缺失报告、题数不符、请求错误、上下文超限和过滤等返回非零退出码。
输出达到 `max_tokens` 和最终回答为空会明确统计，仍保留在评分分母中；
它们不会被静默丢弃，也不单独被当成基础设施故障。
Ctrl+C 返回 130，`run.json` 标记 `interrupted`，部分结果不能作为完整成绩。

**退出码 0 表示评测完成且产物审计通过，不表示精度达到合入门槛。**
该入口没有默认精度阈值；成绩为 0 的完整有效运行也可能退出 0。
评测不验证服务器所加载权重或算子与客户端提交一致。
可用 `--label` 标注配置、`--server-revision` 记录操作者提供的服务构建版本；
客户端提交 SHA 单独记录，不冒充服务端版本。

## 与原始模型对比

同样的客户端可以评测已启动的 baseline 服务：

```bash
python tools/tests/run_evalscope_accuracy.py baseline \
  --base-url http://127.0.0.1:8001 \
  --model baseline \
  --datasets gpqa aime24 aime25 \
  --offline-data-dir /mnt/share/vq2a8-eval-data \
  --concurrency 1 \
  --label bf16-baseline \
  --result-dir benchmark-results/accuracy-baseline
```

两边使用相同数据包、提示模板、思考配置、采样参数、输出预算和重复次数。
先检查截断、空回答和请求错误，再比较各数据集分数。
AIME 每年只有 30 题，一道题约占 3.33 个百分点；小样本波动不能直接归因于量化。

## 客户端回归测试

```bash
python -m pytest --noconftest -o addopts= -q \
  tests/ut/tools/test_evalscope_accuracy_runner.py \
  tests/ut/tools/test_evalscope_accuracy_summary.py \
  tests/ut/tools/test_evalscope_dataset_prepare.py
```

客户端测试不替代目标 NPU 上真实模型的完整精度验收。

## 接口依据

- [固定 EvalScope 版本](https://pypi.org/project/evalscope/1.12.0/)
- [AIME 适配器](https://github.com/modelscope/evalscope/blob/v1.12.0/evalscope/benchmarks/aime/aime_adapter.py)
- [GPQA 适配器](https://github.com/modelscope/evalscope/blob/v1.12.0/evalscope/benchmarks/gpqa/gpqa_adapter.py)
- [报告 schema](https://github.com/modelscope/evalscope/blob/v1.12.0/evalscope/report/report.py)
