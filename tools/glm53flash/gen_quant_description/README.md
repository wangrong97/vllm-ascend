# gen_quant_description.py — AscendV1 量化描述文件离线生成器

不跑量化,直接根据量化配置 yaml + 权重清单,秒级生成 `quant_model_description.json`
(AscendV1 格式,vllm-ascend 部署时据此识别每层的量化方式)。

## 用法

```bash
python3 gen_quant_description.py [config.yaml] [model_dir] [output.json]
```

三个参数均可省略,默认值:

| 参数 | 默认值 |
| --- | --- |
| config.yaml | `/mnt/share/w00608002/weight/GLM-5.3-Flash-w4a4/GLM-5.3-Flash_best_practice.yaml` |
| model_dir | `/mnt/share/w00608002/weight/GLM-5.3-Flash-BF16-fixcfg`(读取 `model.safetensors.index.json` 的权重键清单) |
| output.json | `<config 所在目录>/quant_model_description.json` |

示例:

```bash
# 默认(GLM-5.3-Flash W4A4)
python3 gen_quant_description.py

# 指定其他配置/模型/输出
python3 gen_quant_description.py my_w8a8.yaml /path/to/model /path/to/output/quant_model_description.json
```

## 规则说明

- 从 yaml 的 `spec.process` 中提取所有 `linear_quant` 块,按顺序做 fnmatch 匹配
  (支持 `{}` 逗号展开,如 `*layers.{3,7,43}.self_attn.o_proj`)
- `(act_dtype, weight_dtype)` → 标签映射:
  `mxfp8→W8A8_MXFP8`、`mxfp4→W4A4_MXFP4`、`int8→W8A8_DYNAMIC`、`fp8_e4m3→W8A8_FP8`
  (新组合需在脚本的 `DTYPE_LABEL` 中补充)
- 只处理 `model.language_model.layers.*` 下的 Linear 权重(与量化流水线只逐层
  处理语言模型 decoder 的行为一致);视觉塔、embed/lm_head、norm 等一律 FLOAT
- 命中的量化 Linear 生成 `.weight` + `.weight_scale` 两条目;其余键标 FLOAT
- 末尾写入 `model_quant_type`(取出现的最低比特类型)和 quarot metadata 占位

## 用途与注意

1. **预览/校验**:量化前确认各层量化范围是否符合预期(比跑完 40 分钟再发现
   配置错了要便宜得多)
2. **格式对齐验证**:正式量化跑完后,msmodelslim 会生成权威版本,可 diff 两者
   验证规则理解一致
3. ⚠️ 该文件是离线推演产物,量化参数(scale 数值)只有真实跑完量化才有;
   本文件只描述"哪些层、什么量化方式"
4. GLM-5.3-Flash 相关背景见 `docs/glm3flash/experiment_tracking.md` §6
