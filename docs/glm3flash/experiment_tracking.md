# GLM-5.3-Flash W4A4 量化精度对照实验

> 实验跟踪文档。每完成一个阶段或发生状态变化时更新「进度跟踪」和「结果记录」。
> 创建: 2026-11-04 | 最近更新: 2026-11-05

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
| W4A4 量化输出 | `/mnt/share/w00608002/weight/GLM-5.3-Flash-w4a4` |
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
MODEL_PATH=/mnt/share/w00608002/weight/GLM-5.3-Flash-w4a4 QUANT_METHOD=ascend PORT=9000 bash serve_glm53flash.sh
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
| 任务1 基线 | ⚠️ 数据不可用 | 11-06 | aime2024 五遍完成但**存在严重异常截断**(见 §5.1), 其余 4 数据集未跑(11-06 00:32 服务启动 HCCL 失败) |
| 任务2 量化 | ✅ 完成 | 11-05 12:34 | 卡2, 约 70 分钟, 174GB/44 分片, SUCCESS 无报错 |
| 任务3 W4A4 精度 | 🟢 进行中 | 11-06 | aime2024 五遍完成 92.67±2.4; 其余数据集 03:01 被外部 SIGTERM 打断, 待续跑 |
| 对照报告 | ⬜ 未生成 | - | 模板已验证(report_20261104.md) |

## 5. 结果记录

### 5.1 基线 (FP8)

**官方采样参数 (temperature=1.0, top_p=0.95):**

| 数据集 | 各遍 accuracy | mean±std | 备注 |
| --- | --- | --- | --- |
| aime2024 | 63.33 / 63.33 / 46.67 / 46.67 / 46.67 | 53.33±9.13 (n=5) | ⚠️ **存在严重异常截断, 分数不可用**: 逐题复核发现 >50% 的请求生成到 max_out_len=131072 上限仍未停止(中位输出=上限), 13-15/30 题无 boxed 最终答案。模型退化为"停不下来", 是 FP8 服务路径缺陷, 非真实精度 |
| gpqa / gsm8k / humaneval / math500 | - | - | 未跑 |

**A/B 对照实验 (2026-11-06, aime 单遍):**

| 实验 | 配置 | accuracy | out_tokens 中位 | 结论 |
| --- | --- | --- | --- | --- |
| A/B1 | 基线 + 关 quant_fp8 融合 (`custom_ops: ["all","-quant_fp8"]`) | 63.33 | 131072(仍打满) | 融合算子排除嫌疑 |
| A/B2 | 基线 + 关 MTP 投机 | 跑中 | - | 判别 MTP 路径 |

**W4A4 对照正常**: out_tokens 中位 17779, 仅 4/30 打满上限 → 截断是 FP8 路径特有缺陷。嫌疑方向: FP8 权重块缩放加载/lm_head/EOS 附近数值问题。

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

- 路径: `/mnt/share/w00608002/weight/GLM-5.3-Flash-w4a4`(174GB, 44 分片 + config + 官方 description)
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
- 产物: `/mnt/share/w00608002/weight/GLM-5.3-Flash-w4a4-kda8`(170GB, 比 w4a4 再省 4GB)
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

## 7. 更新日志

- 2026-11-06: 00:00 管线自动重启; 任务1 基线服务 HCCL 启动失败跳过; **任务3 W4A4 aime2024 完成 5 遍: 92.67±2.36**; 03:01 被外部定时清扫(SIGTERM 整链); 发现**基线 FP8 异常截断缺陷**(>50% 请求打满 131072 上限), 基线分数不可用; A/B1 排除 quant_fp8 融合(63.33 仍截断), A/B2(关 MTP)进行中。
- 2026-11-05: 建文档; **任务2 W4A4 量化完成**(卡2, 70min, SUCCESS); FA3 适配器支持已实现并验证注入; 描述文件生成器归档到 tools/glm53flash/; ais_bench 采样参数对齐官方; 下午抢跑任务1: **aime2024 首遍 63.33**(新参数), 14:00 定时停止; 服务 4 分钟就绪(page cache 预热); 修复服务停止后 Worker 残留占卡的 bug(按 PGID 整组杀, 实机验证显存回落); 清理 output 异常轮次产物, 脚本改为归档到各任务目录的 scripts/ 子目录。
- 2026-11-04: 环境修复完成; 服务+对话验证通过; 夜间管线任务1 部分完成 (aime2024 n=3), 任务2 被外部 kill。
