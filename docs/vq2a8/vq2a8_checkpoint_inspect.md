# VQ2A8 权重信息采集与 repack 判断

权重加载失败时，先采集实际 checkpoint 的结构，再根据报告决定是否转换。
`tools/inspect_vq2a8_checkpoint.py` 仅依赖 Python 3.10+ 标准库，可单独复制到目标服务器运行。
不需要安装 torch、torch-npu、vLLM 或 safetensors，不加载模型、不占用 NPU，也不修改权重。

## 一次采集

在仓库根目录执行，将 `MODEL` 改成当前加载失败的权重目录；采集期间不要同时导出或修改该目录：

```bash
MODEL=/mnt/share/g00872988/DeepSeek-V4-Flash-0731-32x32-noperm-hot5050125-identity-v3

python tools/inspect_vq2a8_checkpoint.py \
  --model "$MODEL" \
  --contract export-20261103 \
  --sample-values \
  --output-dir benchmark-results/checkpoint-inspect
```

默认只读 JSON 配置和所有 safetensors 文件头，核对文件大小、偏移、索引、张量名称、dtype 和 shape。
上面的 `--sample-values` 额外从识别到的 source-v3 专家中分散抽样：
`out_dim_scale`、`codebooks`、MXFP4 `scale` 各最多 8 个 Tensor，每个不超过 4 MiB。
它检查缩放位模式和 NaN，并记录抽样字节的 SHA256；不会扫描全部权重数据。
当前主线要求自然输入列顺序，发现 `.perm`（包括恒等排列）即报告 `PERM_UNSUPPORTED`。
即使某组专家缺少字段，也继续检查现有字段的 dtype/shape，
并对其中符合布局的元数据 Tensor 抽样。缺字段不会再使整组检查被跳过。
如果只想快速提取结构，去掉 `--sample-values` 即可。

每次创建独立报告目录，输出目录必须位于模型和草稿目录之外。终端最后打印：

```text
REPORT_SCHEMA_VERSION=2
CHECKPOINT_INSPECT_STATUS=collected 或 issues_found
SUMMARY=.../summary.txt
REPORT_BUNDLE=.../checkpoint-report.zip
```

请提供 `checkpoint-report.zip` 和服务启动失败时的首个完整异常。
不方便传文件时，先贴 `summary.txt`；详细分析还可能需要压缩包中的元数据。
报告包含模型路径、配置、Tensor 名称和本机已安装包的版本，不包含权重 payload。

## 报告内容

| 文件 | 内容 |
| --- | --- |
| `summary.txt` | 实际量化配置、专家类型计数、布局示例、抽样覆盖、MTP 层号、QuaRot 信息及问题建议 |
| `report.json` | 配置、分片统计、每层专家清单、完整布局分类、DSpark 包结构检查、抽样结果和检查范围 |
| `metadata.json` | 原始配置、完整索引、格式标记、专家 manifest 及已有转换报告 |
| `files.json` | 文件相对路径、大小、修改时间、符号链接标记 |
| `tensors.jsonl.gz` | 每个可解析 Tensor 的名称、dtype、shape、分片、偏移和索引匹配状态 |

如果存在 `MODEL/dspark/`，脚本将主模型和草稿作为独立 checkpoint 分别报告。
主目录中未被索引引用的旧 shard 会单独标注，不直接当成主模型正在加载的权重。
若 MTP Tensor 被写进 GS32 主模型索引，会报告 `MTP_IN_TARGET_INDEX`。
文件截断、目录冒充文件、重复 JSON 键、索引缺失、专家缺字段、混合目录深度等会进入问题清单。
同类问题保留总数和最多 20 个示例；完整 Tensor 清单保存在压缩文件中。

新版摘要包含以下信息，方便直接粘贴分析：

- `quantization_config`：实际配置；`V3_CONFIG` 同时打印当前加载器要求的配置。
- `expert_projections`：自然列序三字段 VQ2（`vq2_natural_columns`）、完整 MXFP4、其他不支持或缺字段的投影数量。
  这是投影数量，一个专家通常有 `w13` 和 `w2` 两组。
- `expert_layouts` / `root_layouts`：专家与非专家权重的名称族、dtype、shape 和数量；摘要各显示最多 12 种，完整列表在 `report.json`。
- `samples.*`：符合抽样布局的 Tensor 总数 `eligible`、选中数 `selected`、未选中数 `not_selected`，
  以及 `passed` / `failed` / `skipped`。未选中或跳过不表示验证通过；缺字段和布局不符另见问题清单。
- `mtp_stage_tensors`：实际 MTP 层号和每层 Tensor 数量；同时列出 main projection、Markov head 等 DSpark 特征权重。
  仅有 `mtp.*` 名称或某个 Tensor 总数，不足以判断可作为 DSpark 草稿。
- `rotation_artifact`：名称含 `quarot` 的 safetensors 文件、是否被索引选中，以及其中 Tensor 的 dtype/shape。
  不读取旋转矩阵 payload，也不能据此确认旋转已融合、该文件可删除或输入列已经是自然顺序。

DSpark 包若带有 `dspark_repack_report.json`，会识别其格式，并核对 manifest、索引和文件头。
此外检查部分关键权重、MTP 层号和 FP8/FP4 配置；这些检查不代表草稿全部权重及主草稿兼容性已验证。
报告中的原有 SHA256 只作采集，**不会重新校验全部 payload**。需要全量包完整性验证时另外执行：

```bash
python tools/repack_vq2a8_dspark.py --output "$MODEL/dspark" --verify-only
```

此命令会读取全部草稿数据。它校验打包时记录的哈希，不能单独证明草稿与当前主模型同源。

`collected` 表示本脚本未发现所覆盖的错误，**不表示模型已通过加载或精度验证**。
`issues_found` 表示已采集并发现问题，报告仍可使用；两种情况均返回退出码 0。
无法完成采集或写出报告时返回 1，并打印 `collection_failed`。

## 如何判断是否 repack

| 报告分类或问题 | 处理方向 |
| --- | --- |
| `source_v1_candidate` | 可能需要现有 v1→v3 转换；先运行下方 dry-run，确认 manifest 与实际布局满足转换器输入要求 |
| `source_v3_candidate` | 符合当前加载器约定的 v3 可直接加载；不要再次套用 v1→v3 转换器 |
| `mixed_or_conflicting` | 配置、格式标记或实际张量布局冲突，先核对导出来源，不能只修改版本号 |
| `dspark_original_package` | 识别到现有 DSpark 打包格式；结合包结构问题判断，无需套用 GS32 v1→v3 转换器 |
| `dspark_candidate` | 有 DSpark 配置依据，但仍需检查完整草稿结构、来源和主模型兼容性 |
| `mtp_candidate` | 发现 MTP Tensor，尚无充分 DSpark 配置或打包依据；不能直接认定为可用 DSpark |
| `other_or_unknown` | 保留原始导出信息，依据张量清单决定适配或转换方式 |
| `PERM_UNSUPPORTED` | 当前框架不再支持 `.perm`，包括 identity；需由导出端交付符合自然列顺序的权重，不能只删除旧字段 |
| `V3_ROOT_NAME` | 非 routed 名称未被现有映射接受；需结合完整布局处理，不能仅改版本号 |
| `ROTATION_PROVENANCE` | 已收集 QuaRot 文件头，仍需导出端说明旋转是否融合及执行方式 |
| `SHARD_DEPTH` / 索引问题 | 先修复文件组织与索引；这不一定需要重新量化或转换 payload |
| MTP 缺失 | 不影响普通主模型解码；DSpark 草稿准备见 [DSpark 指南](vq2a8_dspark.md) |

v1 转换预检只读原始模型，尚不生成新权重：

```bash
python tools/convert_vq2_mixed_gs32_v3.py --source "$MODEL" --dry-run
```

加载器与导出规范统一要求三字段 VQ2、两字段 MXFP4，完整覆盖所有层与专家（包括 hash 层）。
v1→v3 工具仅用于符合自然输入列顺序的源权重布局转换，也拒绝带 `.perm` 的输入。
当前 `32x32-noperm-hot5050125-identity-v3` 已是 v3，结构检查通过后无需再次转换。

文件头不能证明 Gate/Up 行序、码本 tile 轴顺序、自然输入列的数值语义、权重来源或数值正确。
非专家权重的完整名称映射、缺失项和 Attention 形状仍需配套加载器校验；抽样未覆盖的值也未验证。
HCCL 配置、NPU 内存、驱动与算子错误不在本脚本检测范围内。
例如 `/etc/hccl_rootinfo.json` 为目录导致的 `EI0004`，应修复通信部署配置，repack 无法解决。

相关说明：[GS32 加载与转换](vq2a8_gs32.md) ·
[导出格式规范](vq2_mixed_gs32_export_spec_20261103.md) ·
[HCCL 最小检查](vq2a8_startup_hccl.md)。
