# 非 routed、输出头和 MTP 的实际变换取证

更新日期：2026-11-03。适用分支：`test/vq2a8-perf-20261024`。

非 routed 权重可能经过旋转、平滑或 Norm 融合，不能再按“与原始 0731 逐字节一致”判断正确性。
本次 packed_weight 的证据已回传，用户已确认全部旋转折叠、推理不需要额外旋转；无需重复运行采集。
主模型启动见[加载适配与验证](export_20261103_adaptation.md)，交付语义见[权重规范](vq2_mixed_gs32_export_spec_20261103.md)第 7.8 节。
下面保留工具用法，供后续不同导出诊断；不通过删除扩展配置绕过检查。

## 1. 没有原始权重也可以运行

在内网 Linux 仓库根目录执行。使用现有带 NumPy 的 Python 环境，不导入 vLLM、torch 或 torch_npu，
不占 NPU、不启动服务、不执行模型目录里的源码、不修改权重。

```bash
git pull --ff-only origin test/vq2a8-perf-20261024

MODEL=/mnt/share/g00872988/DeepSeek-V4-Flash-packed_weight
python tools/inspect_vq2a8_transforms.py \
  --model "$MODEL" \
  --output-dir benchmark-results/transform-inspect
```

**先回传终端 `SUMMARY_JSON=` 对应的 `summary.json` 内容即可。**
同目录 `report.json` 保存完整选取结果及分片元数据，必要时再查看。
`collected` 仅表示采集完成，不表示可以正确推理；`collected_with_errors` 会列出具体读取失败。

## 2. 可选：与同源原始权重对照

如果内网有导出时实际使用的原始权重，追加 `--reference-model`。没有也不必下载或等待：

```bash
MODEL=/mnt/share/g00872988/DeepSeek-V4-Flash-packed_weight
ORIGINAL=/实际路径/导出时使用的原始DeepSeek-V4-Flash-0731
python tools/inspect_vq2a8_transforms.py \
  --model "$MODEL" \
  --reference-model "$ORIGINAL" \
  --output-dir benchmark-results/transform-inspect
```

不同版本的原始权重会干扰对照，不能把所有数值差异都归因于旋转。
两种用法均不需要量化导出源码。

## 3. 提取哪些证据

| 内容 | 读取范围与用途 |
| --- | --- |
| 配置、导出元数据、safetensors 元数据 | 提取 rotation、smooth、Norm、head、layout、MTP、revision 等声明；声明与实际测量分开 |
| 主模型和 MTP 最终 Norm | 小 Tensor 完整读取，报告 hash、是否全一、范围、非有限值 |
| embedding、head、HC head | 小 Tensor 完整读取，大矩阵抽取最多 4 个分散行；记录行号及 hash 范围 |
| Attention、共享专家、router、HC | 抽取主模型第一层和最后一层；大矩阵只读少量行 |
| MTP | 索引统计全部阶段和字段；读取非 routed 的小 Tensor 或少量行，不读取 MTP 专家大矩阵 |
| 旋转矩阵 | 查找索引及名称含 rotation/quarot/smooth 的辅助文件，完整读取唯一命名的 `global_rotation`，上限 128 MiB |
| 旋转性质 | 对选取的行、列检查 Gram 矩阵、对称性和恒等性；不是全矩阵正交性证明 |
| 可选原始权重对照 | 同名同布局 Tensor 的对应字节比较；head/embed/HC head 计算明确列出的数值候选公式 |

普通 Tensor 完整读取上限为 2 MiB，超过则抽行；4096×4096 F32 旋转矩阵约 64 MiB。
不读取全部模型载荷，也不做整矩阵立方复杂度的正交验证。
辅助矩阵如果不叫 `global_rotation`，先看报告 `rotation.candidates`，再用 `--rotation-tensor 实际名称` 指定。
同名多个候选不会自动选择；不读取 pickle/PT 文件。

原始权重存在时，输出头候选包括 `W`、`W * 原始Norm.gamma`，以及它们右乘 `R` 或 `R.T`。
HC head 按每个 hidden-size 列块分别探测右乘；这只是列明公式的数值实验，不预设导出一定使用该公式。
报告不自动选择加载方案。尤其当 R 对称、Norm.gamma 全一时，部分候选不能据此区分。

## 4. 后续不同导出需说明的信息

下面这些信息无法单凭 shape、hash 或全一 Norm 可靠恢复。可以提供简短公式或变换清单，不必提供完整源码：

1. 隐藏状态的新坐标系：按行向量约定，是 `x_new = x @ R` 还是 `x @ R.T`？哪些位置在运行时做变换？
2. 最终输出的准确顺序：HC 合并、旋转或逆旋转、RMSNorm、head 分别在哪里执行？
3. 哪些 Norm.gamma 融入了哪些权重？保存的全一 Norm 是否仍需执行 RMS 归一化？
4. Attention 和共享专家：输入、Q/K/V、输出、SwiGLU 中间量是否还有局部旋转或平滑？哪些已折叠进权重，哪些需要运行时代码？
5. MTP：使用 `e_proj/h_proj` 还是 `main_proj`，各阶段如何接收主模型 hidden/token，独立 embedding/head/HC/Norm 如何使用？
6. 实际原始权重版本及导出器版本。MTP 是单阶段还是 DSpark 多阶段，应以实际结构和执行定义为准。

`optional/quarot.safetensors` 未被索引引用，仅能说明它没有作为普通参数加载，不能据此断定它不需要参与推理。
同样，全一 Norm 只能证明保存的参数为一，不能单独证明 gamma 已正确融合进 head。
