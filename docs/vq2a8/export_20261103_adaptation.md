# 无 perm 主线权重：加载适配与内网验证

更新日期：2026-11-06；分支：`test/vq2a8-perf-20261024`。
当前主线模型为 `DeepSeek-V4-Flash-0731-32x32-noperm-hot5050125-identity-v3`。
格式见 [权重导出规范](vq2_mixed_gs32_export_spec_20261103.md)，不再保留旧四字段 VQ2 的 perm 兼容入口。

## 1. 实际权重与复用范围

本地检查了 44 个分片、63,807 个有效 Tensor；其中 routed experts 为 62,608 个、非 routed 为 1,199 个。
43 层共 11,008 个专家，包含 8,256 个 W2/W2、2,064 个 W2/W4、688 个 W4/W4；两个投影独立识别格式。
Attention、共享专家和输出头的普通矩阵保存为 BF16，Norm、HC 等参数为 F32。
本次没有 `.perm` 或 MTP Tensor。

| 能力 | 处理方式 |
| --- | --- |
| VQ2 四维索引、FP8 码本、U64 行缩放 | 复用已有格式及字节视图转换 |
| MXFP4 `weight/scale` | 复用 E2M1、E8M0 加载与 GMM 路径 |
| W2/W4 混精 | 保留按投影建立计算分组、EP 路由及输出恢复 |
| 普通 BF16 / F32 参数 | 复用名称映射、共享专家合并、TP 分片及 F32 Norm 加载 |
| native / reference | 保留两条投影路径；reference 与 native 共享部分加载和激活处理 |
| full_graph | 保留 MRV2 decode 图和 eager chunked prefill，需新权重硬件验收 |
| MTP / DSpark | 本次权重缺失 MTP，不启用 |

## 2. 无 perm 契约

VQ2 投影只接受 `packed_indices`、`codebooks`、`out_dim_scale`；MXFP4 只接受 `weight`、`scale`。
输入列必须对应自然输入激活的同一列。出现旧 `.perm` 时明确拒绝，包括恒等排列；
不生成 identity 参数，不混存三字段/四字段专家，不静默忽略旧字段。
完整专家清单检查保留，三个 hash 层也要求全部专家，避免清理兼容分支后漏检缺失权重。

清理覆盖格式、配置、参数创建、权重加载、激活准备及 Triton 列重排入口。
MoE 的 token 行分发、按专家分组和结果归位保留；这些不是已删除的权重输入列置换。
原生 GS32 投影接口不接收 perm，本次不修改 native ABI，无需仅因此重编 `.so/.o`。
跨分支复用不匹配的 native 产物仍需单独处理。

非 routed 参数按交付值加载，不凭文件名 `identity` 推断变换来源，也不新增额外旋转或重复融合 gamma。
全一 Norm 仍执行 RMS 归一化。保留模型固有的 RoPE、indexer Hadamard 等计算。

## 3. 检查与启动

内网 Linux 路径按实际部署位置调整；示例假定模型存放于 `/mnt/share/g00872988/`。
在仓库根目录先读取结构，检查工具不占用 NPU：

```bash
MODEL=/mnt/share/g00872988/DeepSeek-V4-Flash-0731-32x32-noperm-hot5050125-identity-v3
python tools/inspect_vq2a8_checkpoint.py \
  --model "$MODEL" --contract export-20261103 --sample-values \
  --output-dir benchmark-results/checkpoint-inspect
```

服务启动与测试分开执行：

- [四卡吞吐启动和测试](packed_weight_batch_throughput.md)：TP4/EP4、DP1、B64、8192 上下文，含 native/reference 两条路径。
- [四卡精度启动和测试](vq2a8_accuracy.md)：相同拓扑和 B64，262144 上下文，GPQA Diamond、AIME24、AIME25。

两份文档均显式 `--no-dspark`。配置中的 `num_nextn_predict_layers=1` 不等于存在可用 MTP 权重。
已有完全符合对应配置的服务可以复用；吞吐服务的 8192 上下文不能承载精度评测的 131072 输出预算。

服务就绪后可先发送一次短请求；只复制代码块，不把 JSON 返回结果再粘贴为 Shell 命令：

```bash
curl -fsS http://127.0.0.1:8003/health
curl -fsS http://127.0.0.1:8003/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"vq2a8","messages":[{"role":"user","content":"中国的首都是哪里？"}],"temperature":0,"max_tokens":64,"reasoning_effort":"none","chat_template_kwargs":{"thinking":false,"reasoning_effort":"none"}}'
```

## 4. 验证边界

已读取真实 checkpoint 的全量索引和文件头，并用当前加载逻辑检查首末层三种混精组合的实际 Tensor。
少量码本和 scale 的 payload 抽样不等价于全量数值校验；字段和 shape 也不能证明 Gate/Up 与输入列的实际内容顺序。
本地 CPU 检查不等价于 Ascend 950 的整模加载、首请求、EP 通信、图捕获、精度或吞吐通过。
后续在同一份权重和相同服务参数下比较框架版本，避免把模型精度配置变化归因于代码。
