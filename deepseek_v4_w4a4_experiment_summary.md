# DeepSeek-V4-Flash 量化实验结果总结

> 日期：2026-08-26 ~ 2026-09-04
> 环境：8×Ascend950DT(A5),CANN 9.1.0,vllm-ascend releases/v0.23.0,DP8+EP,图模式 FULL_DECODE_ONLY,MTP×1
> 评测工具：ais_bench(`/mnt/share/rr08002/benchmark`)，模型配置 `vllm_api_general_chat`(model=dsv, port=9001, batch_size=16)

## 1. 背景与目标

对 msmodelslim 量化的 DeepSeek-V4-Flash W4A4 权重（及后续变体）与原生 FP8 权重做性能/精度对比，定位"W4A4 比原生慢/精度低"的来源。

## 2. 权重版本一览

| 简称 | 路径 | 量化配置 |
|---|---|---|
| FP8 原生 | `/mnt/share/weight/DeepSeek-V4-Flash` | 官方 FP8：全线性层 FP8 e4m3 + e8m0 scale(attn/共享专家 128×128 块，MoE 专家实为 **FP4 e2m1 + FP8 激活** 即 W4A8) |
| W4A4 | `/mnt/share/rr08002/weights/DeepSeek-V4-Flash-w4a4` | attn W4A4,wo_a/wo_b/共享专家 W8A8,MoE W4A4 |
| attnW8A8 | `.../DeepSeek-V4-Flash-w4a4-attnw8a8` | attn 提到 W8A8,MoE 保持 W4A4 |
| attnW8A8+moeW4A8 | `.../DeepSeek-V4-Flash-attnw8a8-moew4a8` | attn W8A8,MoE 提到 W4A8(fp4 权重 + fp8 激活) |
| 混合(原生attn+moeW4A8) | `.../DeepSeek-V4-Flash-attnNative-moew4a8` | attn 用**原生官方量化值**(权重字节逐位沿用,128×128 块 scale 展开为 per-32),MoE 用 msmodelslim W4A8。构建脚本 `build_hybrid_ckpt.py` |
| attnw8a8ceil+moeW4A8 | `.../DeepSeek-V4-Flash-attnw8a8ceil-moew4a8` | msmodelslim `calculate_mx_qparam` floor→ceil 修复后重新量化(attn W8A8 + MoE W4A8),**attn 量化值与原生逐位一致**(见 6.5) |
| w4a4-fouroversix | `.../DeepSeek-V4-Flash-w4a4-fouroversix` | attn/MoE 专家 W4A4，权重用 **fouroversix**(每块 Scale-to-6/Scale-to-4 双路径 MSE 择优 + e8m0 最近邻舍入),wo_a/wo_b/共享专家 W8A8 |

BF16 源统一为 `/mnt/share/weight/DeepSeek-V4-Flash-BF16`。

## 3. 工程修复（前置，已完成合入本地代码)

W4A4 ckpt 最初"走不通"：描述文件里 `wo_a` 标为 `W8A8_MXFP8`,modelslim 路径给它配通用 MXFP8 scheme，只做 2D 变换，而 DSA 的 `_forward_o_proj` 要求 `[G,K,N]` 3D 布局（该 reshape 此前只存在于 FP8 专用的 `ds_linear` scheme)。
**修复**:`quantization/methods/w8a8_mxfp8.py` 新增 `AscendW8A8MXFP8DSWoADynamicLinearMethod`（通用 2D 变换 + wo_a 3D 分组 reshape),`modelslim_config.py:get_quant_method` 增加 `deepseek_v4 + wo_a + W8A8_MXFP8` 特判。修复后 W4A4 服务正常启动、推理正常。
详见 `deepseek_v4_w4a4_quant_flow.md`。

## 4. 精度实验

### 4.1 全量 5 数据集（2026-09-02 上午，采样 T=0.6/top_p=0.95，各 3 轮）

| 数据集 | W4A4 | FP8 原生 |
|---|---|---|
| GPQA_diamond | 70.37 | 72.56 |
| AIME2024 | 78.89 | 68.89 |
| GSM8K | 96.59 | 96.97 |
| HumanEval(pass@1)* | 93.09 | 92.68 |
| MATH500 | 95.60 | 95.53 |

*ais_bench 的 humaneval 评测器被 filelock 3.31 的 fork 安全钩子卡死，已用独立脚本直调 `human_eval.evaluate_functional_correctness` 补测（方法学一致）。

### 4.2 GPQA + AIME × 3(2026-09-02 下午起，采样 T=1.0/top_k=20/top_p=1.0)

| 数据集 | W4A4 | attnW8A8 | attnW8A8+moeW4A8 | FP8 原生 | FP8 复测（9/4) |
|---|---|---|---|---|---|
| GPQA | 69.70/70.71/69.70 → **70.24** | 68.69/70.20/69.70 → **69.53** | 70.71/69.19/69.70 → **69.87** | 72.22/71.72/73.74 → **72.56** | 72.73/73.74/75.25 → **73.91** |
| AIME | 83.33/80.00/86.67 → **83.33** | 76.67/83.33/80.00 → **80.00** | 66.67/66.67/80.00 → **71.11** | 73.33/76.67/70.00 → **73.33** | 73.33/73.33/73.33 → **73.33** |

W4A4 量化参数变体（`scale_alg=2, dst_type_max=7.25`,w4a4_mxfp4.py:78):GPQA 70.04 / AIME 80.00——与改参前无差异。

### 4.3 精度结论

1. **GPQA:FP8 基线 ≈ 73.2（两次平均）,W4A4 系三组 69.5~70.2，稳定低 ~3 分**。把 attn 提 W8A8、把 MoE 提 W4A8 均**没有**抬升
2. **AIME：方差主导**(30 题小样本）,W4A4 系在 71~83 波动，与 FP8 73.3 无稳定方向差
3. 其余数据集（GSM8K/HumanEval/MATH500)：两组持平

### 4.4 决定性实验：混合 ckpt(2026-09-04)

将 attn 换成**原生官方量化值**（权重字节逐位沿用，128×128 块 scale 展开为 per-32 e8m0),MoE 保持 msmodelslim W4A8:

| 数据集 | 三轮 | 均值 |
|---|---|---|
| GPQA_diamond | 74.24 / 73.74 / 75.25 | **74.41** |
| AIME2024 | 70.00 / 73.33 / 73.33 | **72.22** |

**GPQA 从 ~70 直接回到 74.41，达到甚至略高于 FP8 原生基线（73.23)**——差距的元凶定位成功：**msmodelslim 的 attn 量化值**（而非运行时流程、MoE、或 scale 粒度）。这也反向证实 msmodelslim 的 MoE W4A8 与运行时路径没有问题。AIME 上 hybrid 72.22 与 FP8 73.33 持平（方差内）。

### 4.5 MoE 路由分布实验（2026-09-04/07):attn 量化差异是否传导到专家选择

**动机**:4.4 证明差距来自 attn 量化值，但还有一个待排除的间接路径——attn 输出变化是否会改变下游 MoE 的专家选择（routing)，从而放大精度损失？

**方法**：服务加 `--enable-return-routed-experts`，对 GPQA 198 题（与评测完全相同的 prompt 构造，含 ABCD 轮转选项）以相同采样参数（T=1.0/top_k=20/top_p=1.0）跑推理，从响应里解析 base64 编码的 numpy 数组 `[num_tokens, 43层, topk6]`，聚合成每层×每专家的路由计数。
*工程修复*：量化 MoE 路径（`w4a8_mxfp4.py` 的 apply）原本没有 capturer 钩子调用（只在 `fused_moe.py` 的非量化路径有），首轮采集全为 0；已在 apply 的 `select_experts` 后补上 `capturer.capture(layer_id, topk_ids)`（注意放在 load-balance 随机打乱之前，记录真实路由）。

**两组对比**（各 ~10.8 万 token):

| 指标 | 混合（原生attn+moeW4A8) vs 原生 FP8 | attnw8a8-moew4a8(msmodelslim) vs 原生 FP8 |
|---|---|---|
| 全局 JS 散度 | 0.000010 | 0.000010 |
| 每层 JS 散度（均值/最大） | 0.00037 / 0.00064 (L40) | 0.00038 / 0.00071 (L39) |
| 每层 top-6 专家集合 Jaccard | 均值 0.953（最低 0.71 @L0) | 均值 0.947（最低 0.71 @L9) |
| 单专家份额最大偏移 | ±0.012% | ±0.012% |

**结论**:attn 量化策略差异（msmodelslim vs 官方）对 MoE 专家选择**没有可观测影响**，两组实验互相印证。GPQA 差距的传导路径是"attn 输出内容变化 → 推理路径不同"，**不经过 MoE 路由**；路由分布对 attn 权重扰动高度鲁棒。

（原生attn+moeW4A8) vs 原生 FP8：

![image](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI2026090412705905/50432326/f00d698607064fa3bcdcef4d990b7c8f.png)

![image](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI2026090412705905/50432348/5cc5a8620dde414aaaef0b60aeb98686.png)

![image](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI2026090412705905/50432379/0f947068935f4199b461311c91fc74bf.png)

attnw8a8-moew4a8(msmodelslim) vs 原生 FP8：

![image](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI2026090412705905/50432466/168a0bef8f6f4409b385f06f2c74118b.png)

![image](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI2026090412705905/50432500/8d5980d2426d4e1e952670892230ac0c.png)

![image](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI2026090412705905/50432549/a2a9900038ed473da60743e92a30cd19.png)

#### 三张图的含义详解

图在 `ais_bench_logs/routed_experts/`(attnw8a8 组带 `_attnw8a8` 后缀）。

**图 1：`moe_routing_expert_scatter[_attnw8a8].png` — 专家份额一致性散点图（log-log)**

- **是什么**：每个点是一个专家；x 坐标 = 该专家在原生 FP8 下的使用份额，y 坐标 = 该专家在对比 ckpt 下的使用份额。虚线为 y=x（完全一致线）
- **怎么读**:256 个点全部紧贴对角线（偏差最大 ±0.012%)；没有任何专家被一边弃用/另一边热捧（没有出现偏离对角线的离群点）
- **若路由被 attn 差异扰动，该图会显示**：点云散开、出现远离对角线的离群专家

**图 2：`moe_routing_js_per_layer[_attnw8a8].png` — 每层路由分布 JS 散度条形图**

- **是什么**：对每一层，把该层 256 个专家被选中次数归一化成概率分布，计算两个 ckpt 分布之间的 **JS 散度**(Jensen-Shannon divergence，两个分布相似度的度量，0 = 完全相同，ln2≈0.693 = 完全不相关）
- **怎么读**：所有 43 层的柱子都在 1e-4 量级（几乎贴零，虚线为均值线）。最大柱 L39/L40 也只有 0.0007——比"完全无关"小 3 个数量级，说明**逐层路由分布几乎没变**；深层（L32+）略有抬升，是可忽略的弱趋势
- **若 attn 差异传导到路由，该图会显示**：部分层柱子显著高出（如 0.01+)

**图 3：`moe_routing_sorted_share[_attnw8a8].png` — 专家使用份额排序曲线**

- **是什么**：把 256 个专家按使用份额从大到小排序后画曲线（y 对数轴），两个 ckpt 各一条。比较的是分布的**形状**（头部集中度、尾部平坦度），不关心具体是哪个专家
- **怎么读**：两条曲线几乎完全重合——头部的热专家集中度和长尾形态完全一致，说明**专家使用的"贫富结构"没有被改变**
- **若路由结构改变，该图会显示**：两曲线分离（如一边头部更陡 = 更集中，或尾部更高 = 更均匀）

### 4.6 ceil 修复实验（2026-09-07):msmodelslim attn 量化恢复到原生水平

按 6.5 的分析，将 msmodelslim `calculate_mx_qparam` 的 scale 取整从 floor 改为 ceil（消除大权重截断）后重新量化（attn W8A8 + MoE W4A8，配置不变）并评测：

| 数据集 | floor（原实现） | **ceil（修复后）** | 混合（原生attn) | FP8 原生基线 |
|---|---|---|---|---|
| GPQA | 69.87(3 轮） | 75.25/71.72/75.76/73.23 → **73.99(4 轮）** | 74.41 | 73.23 |
| AIME | 71.11(3 轮） | 73.33/70.00 → 71.67(2 轮） | 72.22 | 73.33 |

- **权重级验证**：修复后 8 个抽样 attn 张量对参考系 **relFro = 0.00000（逐位一致）**，饱和（|w|=448）元素 51268 → **0**；修复前 ~1%
- **精度级验证**:GPQA 73.99 ≥ 原生基线 73.23，与混合 ckpt(74.41）一致；AIME 71.67 vs 原生 73.33（方差内）
- 过程：当日下午机器频繁出现 vllm 服务被杀（aime 连续失败 7 次），用自动重启管线（`resilient_bench.sh`：失败检测→重启服务→重跑）凑齐有效轮次，GPQA×4 + AIME×2

### 4.7 fouroversix W4A4 实验（2026-09-07)

msmodelslim 的 fouroversix 算法（`core/quantizer/impl/fouroversix.py`)：每个 32 元素块同时尝试 **Scale-to-6**(scale=amax/6，保动态范围）与 **Scale-to-4**(scale=amax/4，保分辨率）两种缩放，scale 按 e8m0 最近邻（银行家规则）舍入后分别计算量化 MSE，**逐块择优**；本模型 48.99% 的块选了 Scale-to-4（日志实测）。attn/MoE 专家 W4A4 用 fouroversix,wo_a/wo_b/共享专家保持 W8A8:

| 数据集 | 三轮 | 均值 |
|---|---|---|
| GPQA_diamond | 73.74 / 74.75 / 73.74 | **74.08** |
| AIME2024 | 70.00 / 86.67 / 83.33 | **80.00** |

GPQA 74.08 超过 ceil 修复版（73.99）与 FP8 基线（73.23),**fouroversix 是 W4A4 路线目前最好的量化算法**——自适应缩放既保留了 mxfp4 的动态范围利用，又避免了 minmax 一刀切（以及旧 floor 实现）的精度损失。

### 4.8 全部精度实验汇总大表

**A. 5 数据集（2026-09-02，采样 T=0.6/top_p=0.95，各 3 轮）**

| ckpt | GPQA(3轮/均值) | AIME(3轮/均值) | GSM8K(3轮/均值) | HumanEval(3轮/均值)* | MATH500(3轮/均值) |
|---|---|---|---|---|---|
| W4A4 | 68.69/74.24/68.18 → **70.37** | 80.00/76.67/80.00 → **78.89** | 96.51/96.66/96.59 → **96.59** | 94.51/92.68/92.07 → **93.09** | 95.40/95.80/95.60 → **95.60** |
| FP8 原生 | 70.71/71.72/75.25 → **72.56** | 70.00/76.67/60.00 → **68.89** | 96.89/96.89/97.12 → **96.97** | 93.29/92.68/92.07 → **92.68** | 95.80/95.40/95.40 → **95.53** |

*HumanEval 为独立脚本补测（ais_bench 评测器被 filelock 3.31 fork 钩子卡死，方法学一致）。

**B. GPQA + AIME(2026-09-02 起，采样 T=1.0/top_k=20/top_p=1.0)**

| ckpt / 实验 | GPQA 各轮 | GPQA 均值 | AIME 各轮 | AIME 均值 |
|---|---|---|---|---|
| FP8 原生（第 1 次） | 72.22 / 71.72 / 73.74 | **72.56** | 73.33 / 76.67 / 70.00 | **73.33** |
| FP8 原生（第 2 次复测） | 72.73 / 73.74 / 75.25 | **73.91** | 73.33 / 73.33 / 73.33 | **73.33** |
| **FP8 原生基线（两次平均）** | — | **73.23** | — | **73.33** |
| W4A4(minmax) | 69.70 / 70.71 / 69.70 | **70.24** | 83.33 / 80.00 / 86.67 | **83.33** |
| attnW8A8 | 68.69 / 70.20 / 69.70 | **69.53** | 76.67 / 83.33 / 80.00 | **80.00** |
| attnW8A8+moeW4A8 | 70.71 / 69.19 / 69.70 | **69.87** | 66.67 / 66.67 / 80.00 | **71.11** |
| W4A4 + scale_alg=2,dst_type_max=7.25 | 67.68 / 70.71 / 71.72 | **70.04** | 83.33 / 76.67 / 80.00 | **80.00** |
| 混合（原生attn+moeW4A8) | 74.24 / 73.74 / 75.25 | **74.41** | 70.00 / 73.33 / 73.33 | **72.22** |
| attnw8a8ceil+moeW4A8(floor→ceil 修复） | 75.25 / 71.72 / 75.76 / 73.23(4 轮） | **73.99** | 73.33 / 70.00(2 轮） | **71.67** |
| **w4a4-fouroversix** | 73.74 / 74.75 / 73.74 | **74.08** | 70.00 / 86.67 / 83.33 | **80.00** |

**GPQA 全景排序**（均值）:fouroversix 74.08 ≈ ceil 73.99 ≈ 混合 74.41 ≥ FP8 基线 73.23 > W4A4 系（minmax/scale_alg/attnW8A8/moeW4A8:69.5~70.2)。
**AIME 为 30 题小样本，方差主导**(60~86.67 均有出现）,W4A4 系与 FP8 无稳定方向差。

## 5. 性能实验结论

### 5.1 128k 长上下文性能对比（2026-09-07)

条件：gsm8k 内容打包至 131,072 token 输入 × 4 条请求（`ais_bench_logs/gsm8k_128k.jsonl`，末尾挂真实问题，输出上限 512)，并发 2,ais_bench `--mode perf`,torch profiler 全程开启（两侧同条件，对比公平）:

| 指标（avg,4 条） | FP8 原生 | fouroversix W4A4 | W4A4 vs FP8 |
|---|---|---|---|
| **TTFT** | 21.06 s | **17.74 s** | **快 15.8%** |
| **Prefill 吞吐** | 6227 tok/s | **7394 tok/s** | **快 18.7%** |
| **TPOT** | 41.7 ms | **33.3 ms** | **快 20%** |
| 总 token 吞吐 | 7966 tok/s | 9447 tok/s | +18.6% |
| 输出 token（均值） | 128.5 | 129.75 | 相当 |

profiler trace:`/mnt/share/rr08002/vllm_profiling/perf_fp8_128k/`、`perf_w4a4fouroversix_128k/`（各 8 rank)。

**结论：128k 长上下文下 fouroversix W4A4 全面快于 FP8 原生（prefill +19%,decode +20%)——W4A4 的带宽优势在长 prefill 场景显著兑现。**

### 5.2 测试过程暴露的环境/引擎问题（记录）

1. **/dev/shm 过小**:ais_bench 推理进程把数据集 dump 到共享内存，`/dev/shm` 默认仅 64MB 且已被崩溃残留占满，导致 infer 进程 SIGBUS(exit -7)、产出 0 输出 token。修复：`mount -o remount,size=32G /dev/shm`
2. **4 并发 128k 请求会杀掉 worker**（原生崩溃，无 Python 栈，FP8 与 W4A4 均复现，与量化无关）；并发 2 稳定。建议作为引擎长上下文高并发稳定性问题单独上报
3. 早前 TPOT 缺失的原因：`default_perf.py:176-181` 在 `output_tokens ≤ 1` 时把 TPOT 置 0，下游剔除该行（此前几轮失败运行的表现）

（注：本节替换了 2026-08-28 的旧 perf 数据——旧数据在短上下文 + 不完整采样下采集，口径不可比，已删除。)

### 5.3 性能数据采集方法与脚本（可复现）

**1. 数据集生成**(`build_gsm8k_128k.py`):

```bash
python3 build_gsm8k_128k.py [输出路径] [行数] [目标token数]
# 默认: ais_bench_logs/gsm8k_128k.jsonl, 4 行, 131072 token/行
```

逻辑：读取 `benchmark/ais_bench/datasets/gsm8k/test.jsonl`，把 Q&A 示例文本逐条拼接到目标 token 数（tokenizer 精确计数截断），末尾挂一条真实 gsm8k 问题（各行不同），输出 `{"question", "answer", "max_tokens": 512}` 的 jsonl——ais_bench custom-dataset 格式（每行 `max_tokens` 控制输出上限，优先级最高）。

**2. 评测命令**（每侧权重各一次）:

```bash
# 1) 启动服务(脚本选择目标权重), 确认 health 后 warmup
bash /mnt/share/rr08002/work/vllm-ascend/vllm_start_dp.sh   # 后台运行
curl -s http://localhost:9001/health
curl -s -X POST http://localhost:9001/v1/completions -d '{"model":"dsv","prompt":"hi","max_tokens":20}'

# 2) 修复 shm(一次性,否则 infer 进程 SIGBUS)
mount -o remount,size=32G /dev/shm

# 3) profiler 打开 → 跑 ais_bench → profiler 关闭
curl -s -X POST http://localhost:9001/start_profile
cd /mnt/share/rr08002/benchmark && ais_bench \
  --models vllm_api_stream_chat \
  --custom-dataset-path /mnt/share/rr08002/work/vllm-ascend/ais_bench_logs/gsm8k_128k.jsonl \
  --custom-dataset-data-type qa --custom-dataset-infer-method gen \
  --summarizer default_perf --mode perf \
  -w /mnt/share/rr08002/work/vllm-ascend/ais_bench_logs/perf_<tag>
curl -s -X POST http://localhost:9001/stop_profile
```

**3. 关键配置注意**:

- `benchmark/ais_bench/.../vllm_api_stream_chat.py`:`batch_size=2`(4 并发 128k 会杀 worker，见 4.2-2);`max_out_len` 被数据集行内 `max_tokens` 覆盖，无需改
- 服务脚本带 `--profiler-config '{torch profiler, dir=/mnt/share/rr08002/vllm_profiling}'`,`/start_profile` 后 trace 按 rank 落盘到该目录（`dp<rank>_..._ascend_pt`)
- 结果 CSV 在 `perf_<tag>/<时间戳>/performances/vllm-api-stream-chat/gsm8k_128k.csv`,TTFT/TPOT/PrefillTokenThroughput 等按行给出 avg/min/max/median/p75/p90/p99


### 5.4 Profiler 算子级解析与单算子验证（2026-09-08)

用 `torch_npu.profiler.profiler.analyse` 解析两组 128k trace(dp0,`ASCEND_PROFILER_OUTPUT/op_statistic.csv`)，并按功能分组统计算子耗时：

**总算子耗时**:FP8 59,954 ms vs fouroversix W4A4 58,215 ms

| 分组 | FP8 原生 | fouroversix W4A4 | Δ |
|---|---|---|---|
| **MoE GMM+Swiglu** | 13,788.8 ms(23.0%) | **5,786.9 ms**(9.9%) | **快 2.4×** |
| Attn(KvQuantSparseAttnSharedkv) | 9,041.9 ms(15.1%) | 11,963.4 ms(20.6%) | 表面 +32% |
| Indexer(VllmQuantLightningIndexer) | 3,903.6 ms(6.5%) | 5,114.9 ms(8.8%) | 表面 +31% |
| QuantMatMul(attn 线性层) | 9,621.1 ms(16.0%) | 10,023.3 ms(17.2%) | +4% |
| 动态量化 / HCCL / 路由 | 12.0s / 5.6s / 2.9s | ≈持平 | — |

**MoE 快 2.4× 的机制**(trace 直接证据）:FP8(W4A8 MoE）走 `GroupedMatmul`(9,879ms)+ `SwigluGroupQuant`(3,910ms）两个算子共 13,789ms;W4A4 走一步融合 `GroupedMatmulSwigluQuantV2`(3,931ms + 残留 GroupedMatmul 1,856ms）共 5,787ms。**MoE 提速是 128k prefill 快 19% 的主因**。

**attn/indexer"+30%"的核查**——构造单算子基准（`bench_indexer_op.py`，真实 128k shape,npu:0):

| 场景 | T(查询 token) | KV 长度 | indexer 单次耗时 |
|---|---|---|---|
| decode 单 token | 1 | 131,137 | 0.248 ms |
| decode MTP / 批量 | 2 / 64 | 131,137 | 0.263 / 0.269 ms |
| prefill 首 chunk | 4096 | 4,096 | 0.491 ms |
| prefill 末 chunk | 4096 | 131,072 | 15.0 ms |

结论：**+30% 不是算子级劣化**。① 两侧 ckpt 在该路径输入 dtype/shape 完全一致（q 是 wq_b 的 bf16 输出量化到 fp8,KV cache 都是 fp8)，不存在 fp4 权重拖慢 indexer 的机制；② indexer decode 极快（0.25ms),trace 的 per-call 均值（1.6~1.7ms）是 decode 与 prefill chunk（满 128k KV 时 15ms/次）的混合均值；③ 两个 run 调用次数不同（+23%，其他算子增减方向还不一致），说明 profiler 墙钟窗口覆盖的算力构成不可比，**该口径不能用于判定单算子优劣**;④ `KvQuantSparseAttnSharedkv` 在裸进程无法独立调用（`torch.ops._C_ascend` NOT FOUND，需要 vllm 运行时 metadata 上下文），其 per-call 差（1.807 vs 1.962ms,+8.6%）同样落在构成波动内。如需严格对比，应按固定 token 步数对齐 profiling 窗口（如只抓固定 N 个 decode step 或某个 prefill chunk)，而非 start/stop 包住整个 bench。

## 6. 关键排查结论

### 6.1 BF16 源与原生 FP8 数值逐位一致（2026-09-03)

编写反量化器（`convert_fp8_to_bf16.py`，处理 FP8 e4m3+128×128 e8m0 块、FP4 e2m1 打包+per-32 e8m0 两种格式），抽查 8 个跨层张量：**dequant（原生 FP8)== 现有 BF16 目录，maxdiff 全部 0.000000**。
→ "BF16 源与原生权重有精度差"的怀疑不成立；重新生成 BF16 无意义。

### 6.2 运行时量化流程等价性（2026-09-04)

逐点核对 attnw8a8-moew4a8 与原生的流程：

- 激活量化：**完全一致**（均为 `npu_dynamic_mx_quant(x, fp8)` 默认 rint,per-token × per-32 × e8m0;norm+quant 融合 pattern 两边都命中）
- MoE：**完全一致**——两边 `process_weights_after_loading` 都做 `npu_format_cast(weight, 29, fp8, input_dtype=fp4_x2)`(fp4→fp8 NZ 无损转换，e2m1 全部值可被 e4m3 精确表示）,apply 参数逐项相同（dispatch/combine 均 fp8 quant_mode=4)
- wo_a：两边最终布局相同（`[G,K,R]` + `[G,K/64,R,2]`)
- **唯一数值差异**:attn/共享专家/wo_a 的**权重 scale 粒度**——原生 128×128 块 vs msmodelslim per-32（本 ckpt **更细**，理论上保真度更高）

→ GPQA ~3 分差距**不来自运行时量化流程**；来源只剩：msmodelslim 量化算法本身（per_block/minmax + mix_calib 校准集，与官方量化器的舍入/校准不同）产生的权重值差异，以及采样方差。

### 6.3 最终定位（2026-09-04，混合 ckpt 实验证实）

4.4 的混合 ckpt（原生 attn 值 + msmodelslim MoE）使 GPQA 恢复到 74.41 ≥ 原生基线：
**GPQA 差距的元凶 = msmodelslim 生成的 attn 权重值**(minmax/per_block 校准方式与官方量化器的差异）。与运行时流程、MoE、scale 粒度、采样参数均无关。

### 6.4 两组 attn 权重数值差异实测（attn_weight_cmp,2026-09-07)

原生（128×128 块 scale）与 attnw8a8-moew4a8(per-32 scale）的 attn 权重分别反量化到浮点，以 BF16 目录（已验证 == 原生值）为参考系逐张量对比（5 类张量 wq_a/wq_b/wkv/wo_a/wo_b × 10 个抽样层）。数据与图在 `ais_bench_logs/attn_weight_cmp/`(metrics.csv + 3 张 PNG)。

| 指标 | 范围 |
|---|---|
| 相对 Frobenius 差 | **0.86% ~ 1.15%**（各层各张量稳定，无层间趋势） |
| 平均绝对差 | ~2×10⁻⁵（权重 abs 均值 0.02，相对 ~0.1%) |
| 最大绝对差 | 0.0078 ~ 0.031(fp8 网格的 1-2 个台阶） |

**图 1:`rel_fro_per_layer.png` — 每张量类型 relFro 随层变化**:5 条线全部贴在 0.01 附近（log 轴），无层间趋势——差异是全局均匀的，不是某几层特别差。
![image](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI2026090412705905/50471938/e87a7fb9ffe344ad94d7260caa7716a7.png)

**图 2:`diff_hist_wq_b_L20.png` — 代表张量差异直方图（log 计数）**：呈完美的**量化网格离散结构**——峰值在 0（绝大多数元素两量化器舍入结果相同），差异值是 ±0.002/±0.004… 的整倍数（fp8 网格台阶），基本对称、无重尾（±0.016 外几乎无点）。即：**差异是少量元素跨了舍入边界，不是系统性偏置**。
![image](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI2026090412705905/50471916/116d3f937d524c33be1c8b8c13687162.png)

**图 3:`scale_granularity_wq_a_L20.png` — scale 粒度对比热图**：左=原生 128×128 块 scale 广播（马赛克，仅 115/116 两级）；中=msmodelslim per-32（细粒度，113~116)；右=指数差（主体 -1 步，局部 -2/0)。精确统计：**128×128 块内 per-32 scale 的指数跨度均值仅 1.15 步（最大 2 步）**,per-32 比 128×128 最多细 4 倍；原生块 scale ≈ 块内最大 per-32 scale 再向上取整（e8m0 取 2 的幂）。
![image](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI2026090412705905/50471980/513e6eb775434ab3b0947519119158d8.png)

**结论**：两组 attn 权重数值分歧 ~1%，形态是离散舍入边界差异；per-32 的粒度优势实际上很小——**粒度不是精度差的原因**，真正的机制在 6.5 的 floor 截断。

### 6.5 原生 vs msmodelslim 的 MXFP8 量化公式差异（2026-09-07，反推验证 + 修复）

**msmodelslim 权重量化算法全景**（源码核对 `msmodelslim/ir/`):

| 类别 | 格式 | scope | 备注 |
|---|---|---|---|
| INT 线性 | int8 / int4 | per_tensor / per_channel / per_group / per_token | observer 仅 minmax |
| FP8 | fp8_e4m3(scale=amax/448,fp32) | per_tensor / per_channel / per_token / per_head | **无 per_block，不支持 128×128 2D 块** |
| MXFP8 | OCP MX 硬编码公式 | 固定 per-32 | `ir/api/impl/mx_quantization.py` |
| 变换类 | quarot / flatquant / svd_residual / smooth_quant | — | 保精度用，与"接近原生"无关 |

两个关键事实：**权重侧 observer 只有 minmax**（无 mse/kl/histogram/awq;histogram 仅激活侧）;**权重量化值与校准集无关**(minmax 只取权重自身 amax，校准集只影响激活量化参数和变换类算法）。

**原生（官方 DeepSeek）约定——从数据完全反推验证**（对 L20 wq_a 全部 256 块拟合）:

```
scale_ue8m0 = 2^ceil(log2(amax / 448))     # 恒 ≥ amax/448,永不截断
weights     = rint(x / scale) → e4m3
block       = 128 × 128
```

验证证据：69/256 个"不匹配"块全部是 `s_nat = 2×candidate`（方向一致、比值恰为 2)，且每块 `amax/(s_nat×448) ∈ [0.5, 1.0]`——与 ceil 约定完全自洽（不匹配仅因参考系 BF16 目录本身是量化导数，原始 amax 略大）。

**msmodelslim mxfp8 公式**(`calculate_mx_qparam`):

```
shared_exp = floor(log2(amax)) - 8
scale      = 2^shared_exp
weights    = rint(x / scale)               # floor(|a|+0.5)
block      = 32(OCP MX 固定)
```

**两个结构性差异**:

1. 块大小 32 vs 128（实测块内 scale 跨度仅 ~1 个指数步，影响小）
2. **floor vs ceil（决定性）**:msmodelslim 的 scale = 2^(floor(log2 amax)-8)，当 `frac(log2 amax) > 0.807`(~19% 的组）时 **scale < amax/448 → 组内最大权重量化后超 448 被截断**；原生 ceil 永不截断

**截断实证**(L20 wq_a):

- **51,268 个元素（1.2%）在 msmodelslim 版饱和在 |w|=448**
- 截断点实例：`ref=-0.0586 → dm=-0.0547`（大权重上 6.7% 误差）
- 误差分布：≤p50 小权重误差 0;p90~p99 平均相对误差 0.38%，最大 6.7%——**误差全部集中在大权重上**

这解释了"per-32 更细却 GPQA 更差"：粒度优势被大权重的系统性截断吃掉。

**修复（一行）**:`floor(log2(amax)) → ceil(log2(amax))`（备份 `mx_quantization.py.orig`)。修复后 scale 恒 ≥ amax/448（与原生同语义），且 per-32 scale 在块内 ≤ 原生块 scale，使原生网格成为其子集 → **attn 量化值与原生逐位一致（relFro=0)**;MoE 的 mxfp4 走独立的 `calculate_mxfp4_qparam`，不受影响。

**msmodelslim 配置空间内的结论**：权重侧没有 mse/kl/awq 类算法可选，mxfp8 per-32 已是其与原生最接近的现成算法；差异不可通过配置消除，只能靠上述源码修复（方案 b）或框架外加 fp8_e4m3 per_block 128×128 + ue8m0(ceil）支持（方案 c，未做）。

## 7. 后续可选实验

- ~~混合 ckpt：原生 attn + msmodelslim MoE~~ → **已完成（5.4/6.3)，定位成功**
- ~~MoE 路由分布对比~~ → **已完成（5.5),attn 量化差异不传导到路由**
- 全 W8A8(attn+MoE 都 W8A8）排除 MoE W4 的影响
- GPQA 差距的 bootstrap 显著性检验
- ~~让 msmodelslim attn 逼近官方~~ → **已完成（5.6/6.4)**:msmodelslim 权重侧只有 minmax observer，无 mse/kl 类算法可选；差异根源是其 scale 取 floor 导致大权重截断，改 ceil 后 attn 量化值与原生逐位一致
- msprof-analyze 解析 `/mnt/share/rr08002/vllm_profiling/{w4a4,fp8}` 的算子级 trace（工具未安装）

## 8. 产物索引

```
/mnt/share/rr08002/work/vllm-ascend/
├── deepseek_v4_w4a4_quant_flow.md      # W4A4 量化代码流程详解(含 MoE 算子级细化 + 源码附录)
├── deepseek_v4_w4a4_experiment_summary.md  # 本文档
├── convert_fp8_to_bf16.py              # FP8→BF16 反量化器(已验证与现有 BF16 目录逐位一致)
├── build_hybrid_ckpt.py                # 混合 ckpt 构建器(原生 attn + 任意 MoE 来源)
├── collect_routed_experts.py           # GPQA 路由采集(--enable-return-routed-experts)
├── plot_routing_cmp.py                 # 路由分布对比分析+绘图
├── attn_weight_cmp.py                  # 两组 attn 权重数值差异对比(反量化+scale 粒度分析)
├── bench_indexer_op.py                 # indexer 单算子 128k shape 性能验证(见 5.4)
├── build_gsm8k_128k.py                 # 128k 输入数据集生成器(gsm8k 打包,见 5.3)
└── ais_bench_logs/
    ├── w4a4_acc/   results/(5数据集,T=0.6)  results_r2_temp1/  results_r3_scale_alg2/
    │               results_attnw8a8/  results_moew4a8/  results_hybrid/  results_attnw8a8ceil/
    │               results_fouroversix/  + 各 server/console 日志
    ├── fp8_acc/    results/  results_r2_temp1/  results_r3/  + 各 server/console 日志
    ├── attn_weight_cmp/  metrics.csv + rel_fro_per_layer.png / diff_hist_wq_b_L20.png / scale_granularity_wq_a_L20.png
    ├── perf_fp8_final/  perf_fouroversix_final/(128k 性能评测 CSV + console 日志,见 4.1)
    ├── gsm8k_128k.jsonl(128k 输入数据集,gsm8k 打包,输出上限 512)
    └── routed_experts/  hybrid.npz / fp8native.npz / attnw8a8.npz(每层×每专家路由计数)
                        moe_routing_{js_per_layer,expert_scatter,sorted_share}[_attnw8a8].png(对比图)
/mnt/share/rr08002/vllm_profiling/{w4a4,fp8}/   # profiler trace(8 rank 各一份)
/mnt/share/rr08002/vllm_profiling/{perf_fp8_128k,perf_w4a4fouroversix_128k}/  # 128k 性能测试 trace(各 8 rank)
msmodelslim 配置:/mnt/share/rr08002/msmodelslim/deepseek-v4-flash-w4a4.yaml(当前为 fouroversix 版;历史备份 .orig/.attnw8a8/.moew4a8)
msmodelslim 修复:/mnt/share/rr08002/msmodelslim/msmodelslim/ir/api/impl/mx_quantization.py(floor→ceil,备份 .orig)
混合 ckpt:/mnt/share/rr08002/weights/DeepSeek-V4-Flash-attnNative-moew4a8
ceil ckpt:/mnt/share/rr08002/weights/DeepSeek-V4-Flash-attnw8a8ceil-moew4a8
fouroversix ckpt:/mnt/share/rr08002/weights/DeepSeek-V4-Flash-w4a4-fouroversix
```

## 9. 运维备忘（这台机器的坑)

- 注释行**不能**夹在 `vllm serve ... \` 与参数行之间（bash 行接续吞参数 → DP=1 → 单卡 OOM)
- 服务被 SIGKILL/会话退出连带杀死后，设备侧 HCCL/显存残留会导致下次启动 EI0020/HcclAllGather 失败，需等待自行回收（~30-60min)；停服务一律 SIGTERM 并等进程全退
- `HCCL_NPU_SOCKET_PORT_RANGE` 在本环境会导致 HCCL AICPU 通信内核初始化失败，勿启用
- ais_bench 必须从 `/mnt/share/rr08002/benchmark` 目录运行（否则加载 site-packages 默认配置）
- 长时任务用 `setsid nohup ... &` 脱离 Claude 会话，否则会话退出会连带杀服务

## 10. msmodelslim量化ymal脚本

```
apiversion: modelslim_v1
metadata:
  config_id: deepseek_v4_flash_w4a4_fouroversix
  score: 90
  verified_model_types:
    - DeepSeek-V4-Flash
  verified_tags:
    DeepSeek-V4-Flash:
      - - vLLM-Ascend
        - Atlas_A5_Inference
  label:
    w_bit: 4
    a_bit: 4
    is_sparse: False
    kv_cache: False

default_w8a8_dynamic: &default_w8a8_dynamic
  act:
    scope: "per_block"
    dtype: "mxfp8"
    symmetric: True
    method: "minmax"
  weight:
    scope: "per_block"
    dtype: "mxfp8"
    symmetric: True
    method: "minmax"

default_w4a4_dynamic_minmax: &default_w4a4_dynamic_minmax
  act:
    scope: "per_block"
    dtype: "mxfp4"
    symmetric: True
    method: "minmax"
  weight:
    scope: "per_block"
    dtype: "mxfp4"
    symmetric: True
    method: "minmax"

default_w4a4_fouroversix: &default_w4a4_fouroversix
  act:
    scope: "per_block"
    dtype: "mxfp4"
    symmetric: True
    method: "minmax"
  weight:
    scope: "per_block"
    dtype: "mxfp4"
    symmetric: True
    method: "fouroversix"

spec:
  process:
    # attention 主线(wq_a/wq_b/wkv/indexer.wq_b): W4A4, 权重用 fouroversix
    - type: "linear_quant"
      qconfig: *default_w4a4_fouroversix
      include:
        - "*attn*"
      exclude:
        - "*wo_a"
        - "*wo_b"
        - "*compressor.wgate"
        - "*compressor.wkv"
        - "*indexer.weights_proj"
        - "*indexer.compressor.wgate"
        - "*indexer.compressor.wkv"
    # wo_a/wo_b 保持 W8A8(与原 W4A4 ckpt 结构一致)
    - type: "linear_quant"
      qconfig: *default_w8a8_dynamic
      include:
        - "*wo_a"
        - "*wo_b"
    # ffn 路由专家: W4A4, 权重用 fouroversix
    - type: "linear_quant"
      qconfig: *default_w4a4_fouroversix
      include:
        - "*ffn*"
      exclude:
        - "*shared_experts*"
    # 共享专家保持 W8A8
    - type: "linear_quant"
      qconfig: *default_w8a8_dynamic
      include:
        - "*shared_experts*"
  dataset: mix_calib.jsonl
  save:
    - type: "ascendv1_saver"
      part_file_size: 4

```
