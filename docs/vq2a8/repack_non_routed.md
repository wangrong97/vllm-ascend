# identity-v3 按新规范转换非 routed 权重

脚本：[tools/repack_vq2a8_non_routed.py](../../tools/repack_vq2a8_non_routed.py)。
目标格式：[导出规范](vq2_mixed_gs32_export_spec_20261103.md)第 7 节；更新日期：2026-11-07。
源模型为 `DeepSeek-V4-Flash-0731-32x32-noperm-hot5050125-identity-v3`，输出目录后缀为 `-export-20261103`。
所有命令从仓库根目录执行。

## 转换范围

| 部分 | 处理 |
| --- | --- |
| Routed VQ2/MXFP4 | 保持 Tensor 名称、dtype、shape 和原始载荷字节，不改变专家编号、Gate/Up 顺序或输入列顺序 |
| 365 个主要 Attention、共享专家及 indexer Q-B 矩阵 | BF16 转为 E4M3FN 权重，每个 128×128 块生成一个 E8M0 scale |
| 235 个 Norm | F32 转为 BF16，并检查往返转换是否精确；不精确则停止 |
| 其他非 routed 参数 | 保持 dtype、shape 和载荷，按原始 0731 名称重命名 |
| 配置与索引 | 保留源模型的三键 source-v3 配置，重建完整权重索引，使用仓库中的最新导出规范 |

例如 `model.layers.0.self_attn.q_a_proj.weight` 改为 `layers.0.attn.wq_a.weight`，
共享专家改用 `ffn.shared_experts.w1/w3/w2`，根参数改用 `embed.weight`、`norm.weight`、`hc_head_fn` 等原始名称。
重命名同时作用于 safetensors 文件头和根索引。源分片中 routed 和非 routed 参数混存，
因此脚本逐 Tensor 重写分片，而非直接复制整个 `layer_*.safetensors`。

FP8 使用当前源权重重新量化，不复制旧模型的 FP8 权重或 scale。块尺度为
`s = 2**ceil(log2(amax/448))`，全零块使用 `s=1`；权重除以尺度后按 E4M3FN 舍入。
这一步会引入数值误差，不能称为纯无损 repack，也不能恢复曾经的旧 FP8 载荷。
脚本记录每个量化矩阵的误差；不会再次旋转、融合 Norm.gamma、生成 `.perm` 或补造 MTP。

## 运行条件与输出

正式转换需要支持 `torch.float8_e4m3fn` 的 PyTorch CPU 环境和 NumPy；不需要 NPU、vLLM、safetensors 包或运行服务。
沿用现有环境即可，不必重装仓库。脚本分块处理权重，避免将完整模型载入内存。
`--threads 4` 控制 CPU 线程数，可按机器资源调整。

输出是独立副本，目标目录必须不存在，并且不能放在源模型目录内。
先运行 `--dry-run` 查看计划和存储估算，再为完整输出预留空间；该模型预计需要约 90 GiB，具体以计划为准。
正式转换需显式传入 `--quantize-fp8`。
失败时不会发布最终输出目录，会保留日志中打印的临时目录，便于排查；源权重不修改。

输出保留 tokenizer、generation 配置等必要文件，写入新索引及 `repack_report.json`。
报告记录源路径和哈希、`per_tensor` 转换操作与误差、输出校验及 `storage` 存储统计。
Routed 字节复制与输出回读校验用于确认未改动其载荷；这不等于整模推理精度验收。

## Linux：内网执行

先检查计划。模型若存放在其他位置，只修改 `SOURCE`：

```bash
SOURCE=/mnt/share/g00872988/DeepSeek-V4-Flash-0731-32x32-noperm-hot5050125-identity-v3
OUTPUT="${SOURCE}-export-20261103"

python tools/repack_vq2a8_non_routed.py \
  --source "$SOURCE" --output "$OUTPUT" --dry-run
```

确认计划和剩余空间后，在同一终端正式转换：

```bash
python tools/repack_vq2a8_non_routed.py \
  --source "$SOURCE" --output "$OUTPUT" \
  --quantize-fp8 --threads 4
```

完成后检查新目录的索引与字段：

```bash
python tools/inspect_vq2a8_checkpoint.py \
  --model "$OUTPUT" \
  --contract export-20261103 \
  --output-dir benchmark-results/checkpoint-inspect-repacked
```

## Windows：本地执行

在 PowerShell 中检查计划：

```powershell
$sourceModel = 'D:\models\DeepSeek-V4-Flash-0731-32x32-noperm-hot5050125-identity-v3'
$outputModel = "${sourceModel}-export-20261103"

python tools/repack_vq2a8_non_routed.py `
  --source "$sourceModel" --output "$outputModel" --dry-run
```

正式转换：

```powershell
python tools/repack_vq2a8_non_routed.py `
  --source "$sourceModel" --output "$outputModel" `
  --quantize-fp8 --threads 4
```

检查输出：

```powershell
python tools/inspect_vq2a8_checkpoint.py `
  --model "$outputModel" `
  --contract export-20261103 `
  --output-dir benchmark-results/checkpoint-inspect-repacked
```

## 如何读取存储与精度结果

存储对比应区分以下口径：

- Tensor 载荷字节数：直接反映 dtype 转换的节省，排除文件头、索引和报告。
- 文件大小：包含 safetensors 文件头、配置、索引、tokenizer、转换报告等实际文件内容。
- 文件系统“占用空间”：受簇大小、压缩、稀疏文件等影响，不能用文件大小代替。

主要节省来自 365 个矩阵由每元素 2 字节降为 1 字节，新增 block128 scale 的开销较小。
235 个 Norm 的存储节省很小，routed 权重的载荷大小保持不变。
实际节省以完成后的报告为准，`--dry-run` 的估算不代表已生成输出。

格式检查完成后，将服务命令中的 `--model` 和客户端 tokenizer 路径改为新输出目录，继续使用 `--no-dspark`。
先验证文本输出，再按[精度测评](vq2a8_accuracy.md)和[吞吐测评](packed_weight_batch_throughput.md)运行。
本地 CPU 转换、哈希和误差检查不替代昇腾上的加载、图捕获及整模精度测试。

## 本地完整转换结果（2026-11-07）

已完成全部 44 个分片的转换，输出目录为：

```text
D:\models\DeepSeek-V4-Flash-0731-32x32-noperm-hot5050125-identity-v3-export-20261103
```

| 统计口径 | 转换前 | 转换后 | 节省 |
| --- | --- | --- | --- |
| 全部 Tensor 载荷 | 94.885912 GiB | 89.430336 GiB | 5.455576 GiB（5.7496%） |
| 目录文件合计，含索引、配置、报告等 | 94.906577 GiB | 89.472258 GiB | 5.434318 GiB（5.7260%） |
| 其中非 routed Tensor 载荷 | 13.694018 GiB | 8.238442 GiB | 5.455576 GiB（39.8391%） |

这里的 GiB 为 `bytes / 1024**3`；文件大小按目录内实际文件重新计数，包含新增的逐 Tensor 报告。
源目录共 101,905,160,754 字节，输出目录共 96,070,105,908 字节，减少 5,835,054,846 字节。
Routed 载荷仍为 81.191895 GiB，占源模型大部分，所以整模存储降幅约为 5.73%。

验证结果：62,608 个 routed Tensor 字节一致；全部 64,172 个输出 Tensor 已回读校验 SHA256；
235 个 Norm 均通过 BF16 精确往返检查。输出非 routed 共 50 类、1,564 个 Tensor，
名称、dtype、shape 和数量与原始 0731 参考清单全部一致。索引检查通过，源目录未改动。

365 个 FP8 矩阵相对源 BF16 权重的汇总相对 L2 误差为 1.1999%，RMSE 为 0.000349751，
单矩阵相对 L2 误差最高为 2.6692%（`layers.31.attn.wkv.weight`）。这些是权重重建误差，不是整模精度下降比例。
尚未在昇腾上执行该输出模型，需继续验收文本输出与精度。

汇总保存在 [本地转换结果 JSON](results/identity_v3_non_routed_repack_20261107.json)，
完整逐 Tensor 信息保存在输出模型目录的 `repack_report.json`。
