# GLM-5.3-Flash W4A4 量化精度对照实验

> 实验跟踪文档。每完成一个阶段或发生状态变化时更新「进度跟踪」和「结果记录」。
> 创建: 2026-11-04 | 最近更新: 2026-11-09

## 1. 实验目标

| 项 | 内容 |
| --- | --- |
| 模型 | GLM-5.3-Flash (glm5_next, 320B MoE / 18B 激活, 45 层 + 1 MTP, 混合 KDA 线性注意力 + DSA 稀疏注意力) |
| 基线 | 官方 FP8 (e4m3 块量化) 权重的精度 |
| 对照 | msmodelslim W4A4 MXFP4 量化(路由专家 W4A4, 其余 W8A8 MXFP8, KDA 层不量化)后的精度 |
| 硬件 | 单机 8x Ascend950DT (96GB/卡) |
| 推理栈 | vllm 0.30.0 + vllm_ascend 0.30.1.dev (releases/v0.30.0 分支) |

## 2. 关键路径

| 用途 | 路径 |
| --- | --- |
| 基线 FP8 权重(只读) | `/mnt/share/w00936111/weights/GLM-5.3-Flash` |
| BF16 权重(量化输入, config 已修正) | `/mnt/share/w00608002/weight/GLM-5.3-Flash-BF16-fixcfg` |
| W4A4 量化输出 | `/mnt/share/w00608002/weight/GLM-5.3-Flash-BF16-w4a4` |
| 量化配置 | `/mnt/share/w00608002/work/msmodelslim/GLM53-flash-w4a4.yaml` |
| 量化脚本 | `/mnt/share/w00608002/work/msmodelslim/msmodelslim_glm53flash.sh` |
| 服务脚本 | `/mnt/share/w00608002/work/vllm-ascend/serve_glm53flash.sh` |
| 实验编排管线 | `/mnt/share/w00608002/work/vllm-ascend/run_glm53flash_pipeline.sh` |
| 管线守护 | `/mnt/share/w00608002/work/vllm-ascend/run_glm53flash_supervisor.sh` |
| 描述文件生成器 | `/mnt/share/w00608002/work/msmodelslim/gen_quant_description.py` |
| 实验产出归档 | `/mnt/share/w00608002/work/output/glm53flash/` |
| 环境一键恢复 | `bash vllm-ascend/.claude/skills/glm53flash-env/restore_env.sh` |

## 3. 任务定义

### 任务1: 基线精度 (FP8)

```bash
PORT=9000 bash serve_glm53flash.sh          # TP8xEP8, fp8 KV, MTP, port 9000
cd /mnt/share/w00608002/work/benchmark       # ais_bench 评测
```

- 数据集 × 5 遍: aime2024 / gpqa / gsm8k / humaneval / math500
- 评测配置: `vllm_api_general_chat_glm`(temperature=1.0, **top_p=0.95**, 无 top_k —— 已对齐官方, max_out_len=131072, batch=32, **端口写死 9000**)

### 任务2: W4A4 量化

```bash
cd /mnt/share/w00608002/work/msmodelslim && bash msmodelslim_glm53flash.sh
# 单卡逐层校准(默认 2 号卡, QUANT_DEVICE=<n> 可换), 约 30-40 分钟
```

量化范围(与官方 FP8 方案对齐, 见 §6 决策记录):

| 模块 | 量化 |
| --- | --- |
| DSA 层 (3,7,...,43,45) q_a/q_b/kv_a/o_proj | W8A8 MXFP8 |
| dense MLP + shared_experts | W8A8 MXFP8 |
| 路由专家 mlp.experts.* | **W4A4 MXFP4** |
| KDA 层全部投影/conv/门控、indexer、kv_b_proj、norm、mHC、视觉塔 | 不量化 (BF16) |

### 任务3: W4A4 精度(同任务1)

```bash
MODEL_PATH=/mnt/share/w00608002/weight/GLM-5.3-Flash-BF16-w4a4 QUANT_METHOD=ascend PORT=9000 bash serve_glm53flash.sh
```

注意: AscendV1 格式的 config.json 无 quantization_config, **必须显式 `QUANT_METHOD=ascend`**。

### 一键全流程(编排管线)

```bash
# 启动(含守护: 被 kill 后自动拉起, 断点续跑)
setsid nohup bash /mnt/share/w00608002/work/vllm-ascend/run_glm53flash_supervisor.sh > /dev/null 2>&1 &

# 彻底停止
pkill -f run_glm53flash_supervisor; pkill -f run_glm53flash_pipeline
```

管线行为: 00:00 前 8 卡连续空闲 30 分钟提前启动, 否则 00:00 后空闲 10 分钟启动;
每任务结束释放服务并确认显存回落; 看门狗每 30 分钟检测卡死并切换阶段;
产物在 `/mnt/share/w00608002/work/output/glm53flash/task{1,2,3}_*_<日期>/`(各任务目录下的 `scripts/` 子目录归档该任务所用脚本与配置), 报告同目录 `report_<日期>.md`。

## 4. 进度跟踪

| 阶段 | 状态 | 时间 | 说明 |
| --- | --- | --- | --- |
| 环境修复 (transformers 5.16.1 / vllm_ascend_C / fixcfg) | ✅ 完成 | 11-04 | 见 §6 |
| 服务脚本跑通 + 对话验证 | ✅ 完成 | 11-04 | 中文回复正常, MTP 接受率 39% |
| 任务1 基线 | 🟢 修复后重测中 | 11-09 | PR #20861 修复后 aime 2 遍 **93.33±4.71**(与 W4A4 同档); 剩 3 遍 aime + 5 遍 GPQA 待续跑 |
| 任务2 量化 | ✅ 完成 | 11-05 12:34 | 卡2, 约 70 分钟, 174GB/44 分片, SUCCESS 无报错 |
| 任务3 W4A4 精度 | 🟢 进行中 | 11-06 | aime2024 五遍完成 92.67±2.4; 其余数据集 03:01 被外部 SIGTERM 打断, 待续跑 |
| 对照报告 | ⬜ 未生成 | - | 模板已验证(report_20261104.md) |

## 5. 结果记录

### 5.1 基线 (FP8)

**修复后基线 (2026-11-09, PR #20861 cherry-pick 后, 64 并发):**

| 数据集 | 各遍 accuracy | mean±std | 备注 |
| --- | --- | --- | --- |
| aime2024 | 90.00 / 96.67 | **93.33±4.71 (n=2)** | ✅ 与 W4A4(92.67±2.36)同档 → 量化路径无损; run1=27题+补测3题合成, run2=96.67 |
| gpqa / gsm8k / humaneval / math500 | - | - | 未跑(用户中断于 run2 后) |

**修复前基线 (2026-11-05/06, 已作废, 仅留档):**

| 数据集 | 各遍 accuracy | mean±std | 备注 |
| --- | --- | --- | --- |
| aime2024 | 63.33 / 63.33 / 46.67 / 46.67 / 46.67 | 53.33±9.13 (n=5) | ⚠️ **根因已定案(见 §6.16)**: prefix 匹配 bug 导致该保 BF16 的层被错误量化, indexer 输出全零; 修复后同口径 93.33±4.71 |
| gpqa / gsm8k / humaneval / math500 | - | - | 未跑 |

**A/B 对照实验 (2026-11-06, aime 单遍, 均在修复前代码上):**

| 实验 | 配置 | accuracy | out_tokens 中位 | 结论 |
| --- | --- | --- | --- | --- |
| A/B1 | 基线 + 关 quant_fp8 融合 (`custom_ops: ["all","-quant_fp8"]`) | 63.33 | 131072(仍打满) | 融合算子排除嫌疑 |
| A/B2 | 基线 + 关 MTP 投机 | 中断(11/11 完成题全部正常收尾) | - | 误导性指向 MTP; 真因是 prefix bug(§6.16) |

**W4A4 对照**: 修复前基线异常时 W4A4 正常(out 中位 17779)——因 ModelSlim 路径不受 prefix bug 影响, 与量化精度无关。

**输出 token 分布对比(aime2024, 30 题, 上限 131072):**

| 运行 | accuracy | out_tokens min | 中位 | max | 打满上限题数 | 无 boxed 答案 |
| --- | --- | --- | --- | --- | --- | --- |
| 基线 run1 | 63.33 | 708 | **131072** | 131072 | 17+/30 | 13 |
| 基线 run2 | 63.33 | 708 | 131072 | 131072 | 18/30 | 13 |
| 基线 run3 | 46.67 | 828 | 131072 | 131072 | ~15/30 | 15 |
| 基线 run4 | 46.67 | 661 | 131072 | 131072 | ~14/30 | 14 |
| 基线 run5 | 46.67 | 596 | 131072 | 131072 | ~12/30 | 12 |
| A/B1 基线关融合 | 63.33 | 708 | 131072 | 131072 | 18/30 | 13 |
| **W4A4 run1** | **93.33** | 816 | 17779 | 131072 | 4/30 | 1 |
| **W4A4 run2** | **93.33** | 888 | 8717 | 131072 | 7/30 | 2 |
| **W4A4 run3** | **90.00** | 800 | 10620 | 131072 | 5/30 | 2 |
| **W4A4 run4** | **90.00** | 965 | 15285 | 131072 | 7/30 | 2 |
| **W4A4 run5** | **96.67** | 912 | 12012 | 131072 | 5/30 | 1 |

> 判读: 基线 FP8 上一半以上请求生成到上限仍未停止(模型退化为无法正常收尾);
> W4A4 五遍中位 8.7K-17.8K token 干净收尾(平均打满仅 5.6/30, 多为难题正常长思考)。
> 截断缺陷为 FP8 服务路径特有, 与评测配置/数据集无关。

**旧参数 (top_k=20, top_p=1.0) 参考值 — 2026-11-04:**

| 数据集 | 各遍 accuracy | mean±std | 备注 |
| --- | --- | --- | --- |
| aime2024 | 50.00 / 56.67 / 36.67 | 47.78±10.18 (n=3) | 旧参数, 不可与新参数直接对比 |

> 官方参考 (A3 PD 部署): GPQA 89.9 (temperature=1.0, top_p=0.95)。
> 基线 aime 偏低已定位为异常截断所致(见上表), 待 FP8 路径修复后重测。

### 5.2 W4A4 量化产物 — 2026-11-05

- 路径: `/mnt/share/w00608002/weight/GLM-5.3-Flash-BF16-w4a4`(174GB, 44 分片 + config + 官方 description)
- 官方 description 统计: W4A4_MXFP4 74305 条 / W8A8_MXFP8 372 条 / FLOAT 1434 条, `model_quant_type=W4A4_MXFP4`
- 与离线预览版 (tools/glm53flash/gen_quant_description) diff: 量化分类完全一致, 差异仅 key 重命名(saver 将 `hc_attn_base`→`attn_hc.base` 等)和 metadata 占位
- FA3 适配: glm_5_next 适配器已实现 inject_fa3_placeholders(未启用), 可用 `CONFIG_PATH=.../GLM53-flash-w4a4c8.yaml` 试跑

### 5.3 W4A4 精度

| 数据集 | 各遍 accuracy | mean±std | 备注 |
| --- | --- | --- | --- |
| aime2024 | 93.33 / 93.33 / 90.00 / 90.00 / 96.67 | **92.67±2.36 (n=5)** | ✅ 完成(11-06 02:35); 生成行为健康(out 中位 17.8K, 仅 4/30 截顶) |
| gpqa / gsm8k / humaneval / math500 | - | - | 03:01 被外部 SIGTERM 打断(gpqa run1 完成 174/198 题), 待续跑 |

### 5.4 KDA 8bit 量化变体 (w4a4-kda8) — 2026-11-07 ✅

- 配置: `msmodelslim/GLM53-flash-w4a4-kda8.yaml`(KDA 层 9 类投影也做 W8A8 MXFP8, 新增 306 个量化模块 = 9类×34层)
- 产物: `/mnt/share/w00608002/weight/GLM-5.3-Flash-BF16-w4a4-kda8`(170GB, 比 w4a4 再省 4GB)
- **aime2024 = 90.00 / 93.33, mean 91.67±2.36 (n=2), 与 w4a4 的 92.67±2.36 同档 → KDA 8bit 量化精度无损**
- 生成行为健康: out 中位 8.5K-13.8K, 打满上限仅 3/30

**精度结果:**

| 数据集 | 各遍 accuracy | mean±std | 备注 |
| --- | --- | --- | --- |
| aime2024 | 90.00 / 93.33 | **91.67±2.36 (n=2)** | ✅ 11-07, 与 w4a4(92.67±2.36)同档 → KDA 8bit 量化精度无损 |
| gpqa / gsm8k / humaneval / math500 | - | - | 未跑, 待补 |

> 注: n=2 是评测触发去重前的两组独立运行(同一服务两轮, 20261107_133016=90.00 / 20261107_133049=93.33);
> 后续如需更高置信度可补满 5 遍。
- **过程中发现并修复 vllm-ascend 两个真实 bug**(建议上游 PR):
  1. `modelslim_config.py` 的 `UPDATED_PACKED_MODULES_MAPPING` 缺 `in_proj_qkvbfg_a`(HF 命名 fused_qkvbfg_a_proj 与运行时模块名不匹配, KDA 量化权重无法部署)
  2. `glm5next/kda.py` 强制 KDA 层 quant_config=None(官方 FP8 无 KDA scale 的假设), 改为检测 checkpoint 是否携带 KDA 量化条目
  - ⚠️ modelslim_config.py 里留有 TEMP-DEBUG 日志, 提 PR 前需移除


### 5.5 main 栈迁移 + kpool indexer C8 — 2026-11-10 🟢

**背景**: v0.30.0 分支上 `--attention_config.indexer_kv_dtype` 对 GLM-5.3-Flash 是 no-op(kpool 被 LI 门控排除); main 已合入 #18447(GLM kpool FP8 C8: 满池 key fp8+per-pool f32 scale, q 逐向量 fp8 打分, CANN pool_key_indexer quant_mode=0)与 #20861 上游原版(6adbb114c)。

**环境迁移(main 验证栈)**:

| 项 | 变更 |
| --- | --- |
| 分支 | glm53flash_v0.30.0 → **main(3ed4259b2)**, 你的提交 47f9e675a 已 cherry-pick(8eb550787, kda.py 与 main 的 use_recoverssm 合并) |
| vllm | 0.30.0+empty → **0.33.0**(main 要求; PyPI wheel 为 CUDA 版, `cumem_allocator.abi3.so` 已禁用走 vllm 自带 ModuleNotFoundError 降级) |
| vllm_ascend_C | 按 main csrc 全量重编(旧 .so/pyc/csrc build 512M+151M 缓存已清) |
| Model Runner | V1 已被 main 移除, GLM 走 V2(serve 脚本去掉 VLLM_USE_V2_MODEL_RUNNER=0) |
| serve 脚本 | 新增 INDEXER_KV_DTYPE 开关注入 --attention-config |

**验证结果**:

| 实验 | 配置 | aime2024 | 备注 |
| --- | --- | --- | --- |
| main 栈冒烟 | FP8 权重 + indexer C8 | -(对话正常, C8 四层证据: 命令行/引擎配置/kpool cache 组/双槽绑定) | 权重 40.48GB/卡, KV 37.97GB(523 万 token) |
| **main 栈 W4A4+C8** | w4a4 权重 + indexer C8 | **93.33** ✅ | 与 v0.30 栈 W4A4(92.67±2.36)同档 → indexer C8 精度无损 + main 栈(V2/vllm0.33)复现正常; 生成健康(out 中位 12156, 打满 5/30); 产物 output/glm53flash/ab_main_w4a4_c8/ |
| **main 栈 MXFP4 伪量化** | FP8 权重 + indexer bf16 + VLLM_ASCEND_KPOOL_MXFP4_SIM=1(q/k 双侧 fp4 往返) | **93.33** ✅ | A4/C4 数值验证无损(out 中位 6683); 算子不支持 fp4 数据, 为伪量化路线; 产物 output/glm53flash/ab_mxfp4_sim/ |
| **W4A4 + MXFP4 伪量化** | w4a4 权重 + 同上伪量化 | **88.89±3.85 (n=3, 未完)** | 86.67/93.33/86.67; run4-5 被停止未跑; 略低于锚(~-3.8), 疑似权重量化×伪量化轻度交互, 待补齐确认; 产物 output/glm53flash/ab_w4a4_mxfp4_sim/ |

### 5.6 main分支量化链路实验(night2) — 2026-11-11 🟢 进行中

**四+1 实验矩阵**(产物 output/glm53flash/night2_*, 详见 night2_report.md):

| 实验 | 权重 | indexer | MLA KV | aime2024 | gpqa | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| 一 | 官方 FP8 | bf16 | fp8 | **92.00±1.82** (90/90/93.33/93.33/93.33) | **90.20±0.98** (89.90/91.41/89.90/88.89/90.91) | ✅ |
| 二 | FP8→W4A4 量化(msmodelslim, 输入=官方 FP8 权重) | — | — | 产物 `weight/GLM-5.3-Flash-FP8-w4a4` | — | ✅ |
| 三 | W4A4-FP8 | **C8** | fp8 | **89.33±2.79** (86.67/90.00/90.00/93.33/86.67) | 84.85 (n=1, 崩溃后未补) | 🟡 |
| 三h(新增) | W4A4-FP8 | **C8+Hadamard** | fp8 | **89.33±2.79** (90.00/86.67/86.67/93.33/90.00) | **90.24±0.58** (89.90/89.90/90.91, 缺题补测合并) | ✅ |
| 五(新增) | W4A4-FP8 | bf16 | fp8 | **91.33±1.82** (90/93.33/93.33/90/90) | **89.90** (n=1, run2-5 未跑) | 🟡 |
| 四 | W4A4-FP8 | bf16+MXFP4 伪量化(无旋转) | fp8 | **89.33±1.49** (90.00×4/86.67) | **89.05±1.17** (90.40/88.38/88.38) | ✅ |
| 六(新增) | W4A4-FP8 | bf16+MXFP8 伪量化(env) | fp8 | **88.00±3.80** (86.67/90.00/83.33/93.33/86.67) | **89.90** (88.89/90.91, n=2 定稿, run3 已取消) | ✅ |
| 七(新增) | W4A4-FP8 | bf16+MXFP4 伪量化(C7.25)+Hadamard | fp8 | **88.00±1.83** (86.67/86.67/90.00/90.00/86.67) | **88.04±1.17** (87.37/87.37/89.39, run3 为 11-14 补) | ✅ |
| 七o(新增) | W4A4-FP8 | bf16+MXFP4 伪量化(C7.25)+**OSCAR**(官方FP8权重激活拟合, 260 万样本/层) | fp8 | **89.33±2.79** (86.67/90.00/93.33/90.00/86.67) | **89.56±1.05** (89.90/88.38/90.40) | ✅ |
| 七o2(新增 11-14) | W4A4-FP8 | bf16+MXFP4 伪量化(C7.25)+**OSCAR(W4A4 自身激活拟合)** | fp8 | **90.67±2.72** (90/90/93.33/86.67/93.33) | **89.06±1.19** (87.37/89.90/89.90) | ✅ |
| 八(新增 11-14) | **W4A4-KDA8**(官方 FP8 输入 × kda8 yaml 量化) | bf16+MXFP4 伪量化(C7.25)+**OSCAR(KDA8 自身激活拟合)** | fp8 | **91.33±4.47** (96.67/93.33/86.67/93.33/86.67) | **89.73±1.27** (88.38/89.90/90.91) | ✅ |

**差分设计**: 一 vs 五 = 纯 W4A4 权重损失; 五 vs 三 = indexer C8 损失; 五 vs 四 = MXFP4 数值损失; 五 vs 六 = MXFP8 数值损失(e4m3+E8M0, SNR≈30.6dB; 实现=VLLM_ASCEND_KPOOL_MXFP8_SIM=1, 与 MXFP4 同注入点 q 打分前/k 写入前, 同开时 MXFP4 优先); 三 vs 三h = Hadamard 对真 C8 的精度影响; 四 vs 七 = Hadamard 对 MXFP4 伪量化的影响(H→往返→H 回原域; 开关 VLLM_ASCEND_KPOOL_HADAMARD_PRE=1, 真量化路径在旋转域存取/打分, q/k/tail 三处同域); **七 vs 七o = OSCAR vs Hadamard 对 MXFP4 伪量化的影响**(同注入点 R→量化往返→Rᵀ; 矩阵=校准集协方差拟合 spread 0.059 vs 固定 Sylvester 0.304, 开关 VLLM_ASCEND_KPOOL_OSCAR_ROTATION=1 + VLLM_ASCEND_KPOOL_OSCAR_CKPT=.../oscar_calib/oscar_indexer_calib.pt, 同开时 OSCAR 优先; 离线证据: MX 误差 11.20→10.97%, 摊平彻底 5 倍); **七o vs 七o2 = OSCAR 矩阵拟合源与部署权重不匹配的代价**(七o 矩阵由官方 FP8 权重激活拟合, 七o2 由 W4A4 权重自身激活拟合——同为校准集 904 条 raw dump + 流式协方差拟合; 若七o2≥七o 则 FP8 参考权重拟合已够用, 若明显更高则量化部署须用自身激活拟合; 前置流程 = W4A4+idx bf16 dump → 拟合 oscar_indexer_w4a4.pt → 评测, 已编排进 night3 链 11-14); **七o2 vs 八 = KDA 层 8bit 量化的整体影响**(八权重 = 官方 FP8 输入 × GLM53-flash-w4a4-kda8.yaml(§5.4, KDA 9 类投影加 W8A8, 再省 4GB)+ 同流程 dump/拟合/评测 → 权重量化深度 × 拟合矩阵匹配度的组合差分; night3 链: 八A 量化(单卡数小时) → 八B dump+拟合 → 八C 评测)。

**六/七/七o 运行脚本**(W4A4-FP8 权重 + indexer 伪量化三变体, 通用参数化版; 同文件沉淀于 `tools/glm53flash/exp_runner.sh`; 断点 run 级续跑, 已有成绩的遍自动跳过):

```bash
#!/bin/bash
# exp_runner.sh 6|7|7o — night2 实验六/七/七o 一键运行
#   6  = W4A4 + indexer bf16 + MXFP8 伪量化(e4m3+E8M0)
#   7  = W4A4 + indexer bf16 + MXFP4 伪量化(C7.25) + Hadamard
#   7o = W4A4 + indexer bf16 + MXFP4 伪量化(C7.25) + OSCAR(校准集拟合)
# 用法: bash tools/glm53flash/exp_runner.sh 7o
set +e
BASE=/mnt/share/w00608002/work/vllm-ascend
BENCH=/mnt/share/w00608002/work/benchmark
OUTROOT=/mnt/share/w00608002/work/output/glm53flash
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

# ---- 1) 实验选择(唯一变量=indexer 量化方式) ----
unset INDEXER_KV_DTYPE VLLM_ASCEND_KPOOL_HADAMARD_PRE VLLM_ASCEND_KPOOL_MXFP4_SIM \
      VLLM_ASCEND_KPOOL_MXFP8_SIM VLLM_ASCEND_KPOOL_OSCAR_ROTATION VLLM_ASCEND_KPOOL_OSCAR_CKPT
case "${1:-}" in
  6)  D="${OUTROOT}/night2_exp6_w4a4_mxfp8"
      export VLLM_ASCEND_KPOOL_MXFP8_SIM=1 ;;
  7)  D="${OUTROOT}/night2_exp7_w4a4_mxfp4_hadamard"
      export VLLM_ASCEND_KPOOL_MXFP4_SIM=1 VLLM_ASCEND_KPOOL_HADAMARD_PRE=1 ;;
  7o) D="${OUTROOT}/night2_exp7o_w4a4_mxfp4_oscar"
      export VLLM_ASCEND_KPOOL_MXFP4_SIM=1 VLLM_ASCEND_KPOOL_OSCAR_ROTATION=1
      export VLLM_ASCEND_KPOOL_OSCAR_CKPT=${OUTROOT}/oscar_calib/oscar_indexer_calib.pt ;;
  *) echo "用法: bash $0 6|7|7o"; exit 1 ;;
esac
export MODEL_PATH=/mnt/share/w00608002/weight/GLM-5.3-Flash-FP8-w4a4
export QUANT_METHOD=ascend KV_CACHE_DTYPE=fp8
log "===== 实验 ${1}: ${D} ====="

# ---- 2) 起服务 ----
( cd "${BASE}" && PORT=9000 setsid nohup bash serve_glm53flash.sh > "${D}_serve.log" 2>&1 & )
sleep 20; deadline=$((SECONDS + 1500)); grace=$((SECONDS + 120))
until curl -s -m 3 http://127.0.0.1:9000/health -o /dev/null 2>&1; do
    if ! pgrep -f "[b]in/vllm serve" >/dev/null; then
        if (( SECONDS < grace )) && pgrep -f "serve_glm53flash.sh" >/dev/null; then sleep 5; continue; fi
        log "[error] 服务进程退出"; exit 1
    fi
    (( SECONDS < deadline )) || { log "[error] 就绪超时"; exit 1; }
    sleep 20
done; log "服务就绪"

# ---- 3) aime×5 + gpqa×3(断点 run 级) ----
mkdir -p "${D}"
has() { find "$1" -name "summary_*.csv" -exec tail -1 {} \; 2>/dev/null | awk -F, '{print $NF}' | grep -qE '^[0-9]+\.?[0-9]*$'; }
for ds in aime2024_gen_0_shot_chat_prompt.py gpqa_gen_0_shot_cot_chat_prompt.py; do
    name=${ds%.py}; rep=5; [[ ${name} == gpqa* ]] && rep=3
    for i in $(seq 1 ${rep}); do
        wd="${D}/${name}/run${i}"; has "${wd}" && { log "${name} run${i} 已有成绩, 跳过"; continue; }
        mkdir -p "${wd}"
        ( cd "${BENCH}" && ais_bench --models vllm_api_general_chat_glm \
            --datasets "${ds}" --mode all --dump-eval-details --merge-ds \
            -w "${wd}" > "${wd}/aisbench.log" 2>&1 )
        log "${name} run${i} = $(tail -1 $(ls ${wd}/*/summary/summary_*.csv 2>/dev/null | head -1) 2>/dev/null | awk -F, '{print $NF}')"
    done
done

# ---- 4) 停服务 ----
bash "${BASE}/serve_glm53flash.sh" --stop; log "===== 实验 ${1} 结束 ====="
```

注意: 起服前自查 `npu-smi info` 8 卡空闲; 七o 确认 OSCAR 生效可 grep 服务日志 `oscar`(矩阵加载打印); 断点续跑=重跑同命令自动跳过已有 summary 的遍(七的 gpqa run1/2 已出分会被跳过, 只补 run3)。

**关键结论(已有数据)**:
- 基线 gpqa 90.20 vs 官方 A3 参考 89.9 —— 基线可信
- **三级差分(aime, exp5 补齐后 2026-11-11 13:27)**: 一 vs 五 = 92.00→91.33(-0.67, 噪声内) → **纯 W4A4 权重(含 FP8→W4A4 链路)在 aime 上无损**; 五 vs 三 = 91.33→89.33(-2.00) → **indexer C8 是精度损失主要来源**; 基线总损失 -2.67 几乎全部来自 indexer C8
- gpqa 侧: 五 89.90(单遍) vs 基线 90.20 —— 与 aime 结论一致(权重无损); 三 84.85 的 -5 差距同样指向 indexer C8(单遍待补 std)
- **Hadamard 修复 C8 的 gpqa 损失(2026-11-13 定案)**: 三h(C8+H) gpqa **90.24±0.58**(89.90/89.90/90.91, n=3)vs 三(C8 无 H) 84.85 → **+5.4, 与基线 90.20±0.98 完全持平**; aime 侧三h 89.33±2.79 与三 89.33±2.79 同档 → **旋转域存取/打分完整修复 indexer C8 的精度损失, 是 indexer 量化的正确形态**(gpqa 长选项题对 q 侧量化敏感, 受益最大); 补测中断题均为两轮中断筛选出的最难长尾(小集仅 58.82%), 合并口径经基线校准
- **七(MXFP4 伪量化+H)组合损失(2026-11-13)**: aime **88.00±1.83**(-4.0 vs 五 91.33)、gpqa **87.37×2**(-2.83 vs 基线, 两遍同分)→ MXFP4(C7.25)伪量化误差大于真 C8 路线; 且末段长题错误率显著高于前段(run1 前 174 题 91.4% → 尾部 24 题 58%), **误差随思考长度累积**
- **OSCAR > Hadamard 端到端定论(2026-11-14)**: 七o(gpqa **89.56±1.05** / aime 89.33±2.79)vs 七(88.04±1.17 / 88.00±1.83)→ 同注入点(R→量化往返→Rᵀ)唯一变量矩阵来源, **OSCAR 双数据集全面 +1.3~+1.5**; gpqa 距基线仅 -0.64(噪声边缘)→ **校准集拟合的 OSCAR 几乎抹平 MXFP4 伪量化损失**, 与离线三层证据(摊平 0.059 vs 0.304 / MXFP4 误差 10.97 vs 11.20 / 曲面平坦)闭环; indexer 旋转方案排序: **OSCAR > Hadamard > 无旋转**
- **MXFP4 旋转收益完整归因(2026-11-18, 四补齐后)**: gpqa 四(无旋转) **89.05±1.17** vs 七(Hadamard) 88.04±1.17 vs 七o(OSCAR) 89.56±1.05 —— gpqa 上 Hadamard 对 MXFP4 反而 **-1.0**(低于无旋转!), OSCAR +0.5 vs 无旋转; 此前"Hadamard 修复 C8 gpqa(-5.35→持平)"的结论**仅适用于真 C8(per-pool scale)路线, 不可外推到 MXFP4(per-32 组自适应 scale)路线**; aime 四 89.33±1.49 与七/七o 同档(aime 对旋转不敏感)。八(KDA8+OSCAR)定稿: aime **91.33±4.47** / gpqa **89.73±1.27**, 与基线同档
- **全链收官(2026-11-18 09:00)**: night3 11-18 00:00 起跑, 八C 续 run4/5+gpqa×3 + 四 全套, 全程零异常零干预; 逐题真实性校验(verify_scores.py)全部通过(30/30、198/198, 截断为题 11/29 等已知边缘题)

**异常记录**:
- 实验一原设计 KV=bfloat16, 实测 SMLA 路径长序列集体失控(24/30 冲 131K, KV 99.6%) → 改 fp8(§6.18)
- 09:25 实验三 gpqa run1 后服务端驱动级崩溃(halSqCqFree drvRetCode=7 → HCCL watchdog → Worker-0 挂),
  脚本误标 exp3 .done 并跳过 exp4 —— 已修正状态, 待续跑补齐(§6.19)

### 5.7 night3 夜间续跑链 + 校准集 dump — 2026-11-12 🟢 进行中

**链路**(`night3_glm53flash.sh` + `night3_supervisor.sh`, 断点 run 级续跑 + 30min 看门狗 + supervisor 60s 重拉): 三h(续 aime run4/5 + gpqa×3)→ 七(aime×5 + gpqa×3)→ 六(aime×5 + gpqa×3)。11-12 22:39 起跑(三h aime run1-3 当日已按 166912 口径重跑出分, run4/5 中午被打断无 summary 由断点机制覆盖), 预计 11-13 上午收官。

**清理无效成绩**(11-12 晚): 六 aime run1=43.33(疑似单遍噪声/截断)、七 gpqa run1=73.23(198 题全量, 补测 45 题错误率偏高拉低总分)已删除重跑; 逐题真实性可用 `tools/glm53flash/verify_scores.py` 复核(基线 92.00/90.20 已验证逐题真实)。

**追加阶段: 校准集 dump indexer 激活**(11-13 完成, `calib_dump_v2.sh` 断点续跑版): 实验一配置服务(官方 FP8 权重 + idx bf16 + 无旋转 + KV fp8, 与 indexer_dump_exp1 同口径, 原始域), msmodelslim 量化校准集 `weight/calib_corpus_R9tau.jsonl`(904 条)逐条串行灌 `/v1/completions`(max_tokens=1 只抓 prefill), raw 模式 dump(**16GB, 12526 npz**, round00+round02; round01 为中断残留; 11-16 起全部 dump 数据集中移至 **`output/glm53flash/dump/`**, 含 README 说明各数据集配置与用途, 后续新 dump 依此约定归档)。断点续跑机制(progress.json 条目级 + round 目录防覆盖)实战验证两次; 数据坑: `splitlines()` 会切 JSON 内 ` ` 产生非法行(改 `readlines()`), 收尾停服会截断在写 npz(拟合时 CRC 校验剔除)。

**OSCAR 校准集拟合**(11-13, `oscar_calib/oscar_indexer_calib.pt`): 12 层 × ~260 万样本/层(168 倍于 exp1), spread 全部 →0.000, 正交误差 ~1e-15。量化误差对比(`compare_calib.py`, 12 层均值): **能量离散度 无旋转 0.709 / Hadamard 0.304 / OSCAR 0.059**(摊平彻底且无 Hadamard 的 L31/L43 退化); MXFP4 误差 11.71→11.20→10.97%, MXFP8 3.06→3.00→2.95% —— **MX 量化(per-32 动态 scale)对旋转不敏感, 收益主要预期在真 C8(per-pool scale)路线**, 与 exp1 小样本结论逐位复现。

**校准分布能量离散度(spread): 三种旋转方式 × 三套权重**(11-16, 同批 40 npz/层抽样, 存 `oscar_calib/spread_3way.log` 第 4 节):

**计算公式**: 对该层 q 激活 `x ∈ [N, 128]`(全部 token×head 展平, 或旋转后 `xR`), 逐维能量 `e_i = E[x_i²] = (1/N)Σₙ x[n,i]²`, **spread = std(e)/mean(e)**, 即 128 维能量向量的变异系数(量纲无关; 协方差恒等式版本 `e = diag(C)`、旋转后 `diag(RᵀCR)`, 与逐样本计算等价)。含义: 0 = 各维能量完全相等(白化, 量化 scale 均匀覆盖所有维); 大 = 能量集中少数离群维, scale 被撑大 → 小能量维相对分辨率损失。正交旋转不改总能量(Σe_i 守恒)只重分配各维 → 旋转的意义即在"只旋转不拉伸"约束下把能量摊平。

| 数据(权重) | 基线(无旋转) | Hadamard | OSCAR(自配矩阵) |
| --- | --- | --- | --- |
| 官方 FP8 | 0.744 | 0.315 | **0.067** |
| W4A4 | 0.714 | 0.303 | **0.070** |
| W4A4-KDA8 | 0.709 | 0.304 | **0.061** |

→ Hadamard 摊平 ~58%、OSCAR 摊平 ~91%(再低 4.5-5 倍, 且无 Hadamard 的层间退化); 基线/Hadamard 两列内三套权重相对差 <5%, OSCAR 列绝对差 <0.01(0.06-0.07 噪声量级)→ 分布几何对量化深度鲁棒; 与端到端自洽(H→OSCAR 的 gpqa +1.5, 即七→七o)。

**三套 dump 交叉对比定论**(11-16, `spread_3way.py` 纯 CPU, 结果 `oscar_calib/spread_3way.log`):
1. **q 分布几何对权重量化鲁棒**: raw spread 逐层几乎逐位一致(FP8/W4A4/KDA8 均值 0.744/0.714/0.709)——量化深度不改变各向异性主轴 → **矩阵在任一权重上拟合一次即可**(推荐官方 FP8 干净权重)
2. **矩阵跨权重完全可移植**: 3×3 交叉(数据×矩阵)旋转后 spread 全部 0.061-0.083, 非对角不比自配差 → **部署无需"量化后重拟合"**(分布层与端到端七o≈七o2 双重证据)
3. **矩阵元素大不相同但功能等价**: 两两 max|R_a R_bᵀ−I| ≈ 1.6-1.9(元素级几乎不相关)——特征值简并子空间内分解不唯一, "摆平"手法不同效果相同

## 6. 决策与排障记录

1. **transformers 必须 ≥5.16.1**: checkpoint 由 5.16.0 保存, `glm5_next` 低版本不识别。
2. **vllm_ascend_C 扩展需手工编译**: wheel 不带; 装 repo + site-packages 两处 (从 repo 目录启动时 repo 包优先)。编译: 仓库根 CMakeLists 直接 cmake, `SOC_VERSION=ascend950dt_9582`。
3. **FP8 权重勿传 `--quantization ascend`**: config 标注 quant_method=fp8, 冲突; AscendV1 (w4a4) 产物才必须显式传 `--quantization ascend`。
4. **KDA 层不量化**: 官方 FP8 `modules_to_not_convert` 与已验证 glm_5_next_w8a8.yaml 均排除 KDA 投影; w4a4 yaml 已从 `*self_attn*` 通配改为 DSA 层精确 include。
5. **FA3 (C8 KV) 未做**: glm_5_next 适配器未实现 FA3QuantAdapterInterface, w4a4c8 中两段 fa3_quant 已摘除, 待 msmodelslim 补支持后恢复。
6. **BF16 权重 config 被反量化工具截断为 4 层** → 用 fixcfg 目录(符号链接+实体 config.json)修正; 文本文件必须实体拷贝(saver 会沿符号链接 chmod 源文件)。
7. **共享机器资源协调**: 本机 8 卡由多个容器/任务共享(含本用户在其他容器 跑的 VLLM 服务)。11-04 晚任务曾两次被 SIGKILL/SIGTERM 打断(卡被其他任务 回收)。管线已有重试/断点续跑机制, 启动前以空闲检测兜底; 排期时注意与 其他容器的任务错开。
8. **对话验证 max_tokens ≥1024**: 思考模型, reasoning 占数百 token, 字段名 是 `reasoning` (非 reasoning_content)。
9. **npu-smi 显存解析**: HBM 用量是倒数第 3 列(`$(NF-3)`), 行首是 `|`。
10. **评测采样参数已对齐官方**(2026-11-05): ais_bench 模型配置改为 `temperature=1.0, top_p=0.95`(去掉 top_k=20), 与权重 generation_config.json 和官方榜单口径一致。注意: 11-04 的部分基线数据(aime2024 n=3)是旧参数 (top_k=20, top_p=1.0)下跑的, 与新参数结果不可直接对比。
11. **服务停止必须按进程组杀**(2026-11-05 修复): Worker 经 setproctitle 改名为 `VLLM::Worker_TP`, 按模型路径 pkill 只能杀 APIServer; 且 npu-smi 报的是宿主机 PID, 容器内 /proc 查不到。修复: serve 脚本 setsid 独立进程组 + `--stop` 按 PGID 整组杀 + `VLLM::` proctitle 兜底清扫(已实机验证显存回落基线)。
12. **基线截断根因已定案**(2026-11-06): 经字节级比对, 本机 BF16 与 HF 官方 GLM-5.3-Flash-BF16 完全一致(含 MTP 层), "W2A8 链路降质"排除。官方 FP8 发布版的 **MTP 层量化带伤**(对 BF16 原版 SNR 仅 32dB, 主干为 55dB), 基线开 MTP 时 draft 提议垃圾 token 导致生成失控(冲到 131072 上限不停)。W4A4 的 MTP 来自健康 BF16 故正常。**结论: 基线低分是官方 FP8 checkpoint 的 MTP 质量问题 + MTP 投机放大, 与推理栈无关; 基线应改测"关 MTP"或换权重重测。**
13. **夜间外部定时清扫**(03:01 整链 SIGTERM): 已连续两晚发生, supervisor 也会被杀, 需协调白名单或错峰。
14. **基线截断真因 = 上游 PR #20861 的 prefix 匹配 bug**(2026-11-09 定案): 官方 FP8 checkpoint 的 `modules_to_not_convert` 用扁平前缀(`model.layers.3...`), 嵌套模型运行时是 `language_model.model.layers.3...`, 精确匹配 0 命中 → 该保 BF16 的层(部分无 weight_scale_inv)被静默错误量化 → indexer wq_b 输出全零、稀疏 top-k 坍塌。已 cherry-pick 修复(suffix 匹配兜底, 3 commits, 纯 Python 无需重编译, UT 20/20)。**修复后基线 aime 93.33±4.71, 与 W4A4 同档; 此前"MTP draft 带伤致截断"的结论(§6.12)为误判**——MTP 层 32dB 的量化差异存在但不主导(教训: 假设机制不通时应推翻重查; 排障需打到运行时模块类型验证)。
15. **并发上限实测**(2026-11-09): 基线 FP8 路径 `MAX_NUM_SEQS` 64 可行, 96/128 初始化 OOM(权重 39.85GB/卡 + 池 41GB 顶死); W4A4 权重小 18GB 或可更高(未验证)。客户端 batch_size 已 32→64。
16. **pkill/pgrep 自匹配陷阱**(反复踩坑): 停止脚本里 pkill -f 的模式若与本脚本命令行字面量相同会自杀且中断后续命令, 必须用 `[V]LLM::` 类括号写法; ais_bench 出分要逐目录读 csv, 不能 find+tail 合并(曾因此误报两组同分)。
17. **bf16 KV 的 SMLA 路径长序列失控**(2026-11-10): GLM-5.3-Flash 在 main 栈用 --kv-cache-dtype bfloat16 时, use_smla=True 路径(sfa_v1.py:264)下 24/30 题集体生成至 131K 上限不收尾, KV 99.6% 近死锁, 吞吐塌至 25 tok/s; 同配置 fp8 KV(C8 打包路径)完全正常(5 分钟 24 题)。疑似 SMLA kernel 长序列正确性 bug, 值得单独提 issue。
18. **09:25 驱动级崩溃 + 脚本误标 done**(2026-11-11): halSqCqFree drvRetCode=7 → HCCL watchdog 异常 → Worker-0 exit -6, 服务整组退出; overnight 脚本的阶段收尾在服务恢复失败时仍 touch .done 且跳过后续实验(恢复流程缺陷, 待修: 失败不应标 done / exp4 被跳未执行)。临时处置: 手动清 .done 与 .night2_finished 后由断点续跑机制补齐。

## 7. 更新日志

- 2026-11-18: **night3 第三夜全链收官(零异常)**: 00:00 定时起, 八C 补 run4/5(93.33/86.67)+gpqa×3(88.38/89.90/90.91) → **八定稿 aime 91.33±4.47 / gpqa 89.73±1.27, ✅**; 四(W4A4-FP8+MXFP4 无旋转)全新跑完 → **aime 89.33±1.49 / gpqa 89.05±1.17, ✅**; MXFP4 旋转归因完整(四 vs 七 vs 七o, Hadamard 在 MXFP4 路线反而 -1.0 的新发现); verify_scores 逐题校验全过。
- 2026-11-16(凌晨): **八B 起服失败空转 2.7h 的教训与自修复机制**: 根因=kda.py cherry-pick 合并引入的分支 bug(`kda_quantized=True` 路径不赋值 `saved_quant_config`), FP8-w4a4-kda8 为 main 栈首个 KDA 条目 checkpoint 首次触发即炸; 确定性代码 bug 对重试型自愈免疫(supervisor 空转循环)。修复: 补丁已打(赋值前置); 机制: **supervisor 强化版**(30min 内 ≥3 次拉起失败 → autofix 已知症状幂等补丁表 → 未知症状 .night3_halt 熔断, `tools/glm53flash/autofix_server_fail.sh`), 同场景 2.7h→~10min。另: 11-15 夜七o2 全出分(aime 90.67±2.72 / gpqa 89.06±1.19 → 拟合源不匹配基本无代价)、八A 量化成功(50min, 44 分片)。
- 2026-11-14(01:00-上午): **night3 第二夜(七o→七→六)**: **七o 端到端确认 OSCAR>Hadamard**(aime 89.33 vs 88.00, gpqa 89.56±1.05 vs 88.04±1.17 → 同注入点 +1.3/+1.5; gpqa 距基线仅 -0.64, 几乎抹平 MXFP4 损失, 与离线摊平证据一致); 七补 run3=89.39 齐遍; **watchdog 修复实战验证通过**(gpqa×3 各 60min 静默零误杀); 07:17 HCCL 崩溃一轮 supervisor 60s 自愈; 六 aime 88.00±3.80 + gpqa run1 88.89, run2/3 跑中, 预计 ~12:10 全链 finished。
- 2026-11-13: **night3 夜间链**: 三h 齐遍收官(aime 89.33±2.79 + **gpqa 90.24±0.58**)、七 aime 齐遍(88.00±1.83)+gpqa 87.37×2; **watchdog 双 bug(40min 阈值+summary 路径)误杀全部 gpqa**, 数据恢复: 孤儿 infer 自跑完(七 run1 198 题离线 87.37)+三h 缺题 35 题补测续跑(tools/glm53flash/gpqa_resume35/, id 映射匹配, 判分口径经实验一校准 89.90=89.90); **Hadamard 修复 C8 的 gpqa 损失(84.85→90.24, 与基线持平)** 为核心新结论; 另: kill -9 服务后 HCCL AICPU 脏状态致起服失败一次(等 60s 自愈)。
- 2026-11-11(下午): **实验五完成 aime 5 遍(91.33±1.82), gpqa run1=89.90**(用户中止 run2-5); 三级差分定论: W4A4 权重无损, 损失来自 indexer C8; 新增 MXFP8 伪量化开关(VLLM_ASCEND_KPOOL_MXFP8_SIM, SNR 30.6dB)。
- 2026-11-11: **夜间链路实验**: 实验一完成(aime 92.00±1.82 / gpqa 90.20±0.98, 基线与官方 89.9 高度一致); 实验二完成(FP8→W4A4 产物); 实验三 aime 89.33±2.79、gpqa 因 09:25 驱动崩溃仅 1 遍; exp4 被误跳; 新增实验五(纯 W4A4 隔离组); bf16 KV SMLA 失控与脚本恢复缺陷已记录(§6.17/18)。
- 2026-11-10: **迁移 main 分支验证**: vllm 0.33.0 升级 + vllm_ascend_C 重编 + cumem CUDA 扩展禁用; **GLM kpool indexer C8(fp8) 在 main 可用并实机验证**(v0.30 为 no-op); 你的提交恢复到 main(含冲突合并); W4A4+C8 aime 对照跑中。
- 2026-11-09: **cherry-pick 上游 PR #20861**(FP8 prefix 匹配 bug 修复, §6.14); 修复后基线重测: run1 合成 90.00(27题+补测3题, 自建 aime_top3 迷你数据集)+ run2 96.67 → **aime2024 93.33±4.71, 与 W4A4 同档, 量化对照实验闭环**; 并发阶梯实测 64 为 FP8 上限(96/128 OOM), 客户端 batch 32→64; 剩余 3 遍 aime + 5 遍 GPQA 待续跑。
- 2026-11-08: 权重一致性核验(本机 FP8/BF16 与官方逐字节比对一致); 官方 config 的 modules_to_not_convert 全量拆解; 整理三版量化策略对照表。
- 2026-11-07: **w4a4-kda8 量化变体完成**(KDA 8bit 精度无损, aime 91.67±2.36); 修复 vllm-ascend 两个 KDA 量化部署 bug(打包映射缺失 + kda.py 强制 quant_config=None)。
- 2026-11-06: 00:00 管线自动重启; 任务1 基线服务 HCCL 启动失败跳过; **任务3 W4A4 aime2024 完成 5 遍: 92.67±2.36**; 03:01 被外部定时清扫(SIGTERM 整链); 发现**基线 FP8 异常截断缺陷**(>50% 请求打满 131072 上限), 基线分数不可用; A/B1 排除 quant_fp8 融合(63.33 仍截断), A/B2(关 MTP)进行中。
- 2026-11-05: 建文档; **任务2 W4A4 量化完成**(卡2, 70min, SUCCESS); FA3 适配器支持已实现并验证注入; 描述文件生成器归档到 tools/glm53flash/; ais_bench 采样参数对齐官方; 下午抢跑任务1: **aime2024 首遍 63.33**(新参数), 14:00 定时停止; 服务 4 分钟就绪(page cache 预热); 修复服务停止后 Worker 残留占卡的 bug(按 PGID 整组杀, 实机验证显存回落); 清理 output 异常轮次产物, 脚本改为归档到各任务目录的 scripts/ 子目录。
- 2026-11-04: 环境修复完成; 服务+对话验证通过; 夜间管线任务1 部分完成 (aime2024 n=3), 任务2 被外部 kill。
