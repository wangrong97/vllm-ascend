# E4 PD 混搭精度崩溃分析

> 分析日期：2026-11-03（恢复重建：2026-11-06）｜ 分析方式：对标 10T 部署方案的"KV 误差逐层定量统计"方法
> 现场日志：`output/pipeline_E3b5_E4_1103_1527/` ｜ 进展跟踪：`docs/eplb/experiment_progress.md`

---

## 1. 背景与现象

### 实验配置

| 项 | P 侧（卡 0-3，port 9001） | D 侧（卡 4-7，port 9002） |
|---|---|---|
| 权重 | `DeepSeek-V4-Flash-0731-w2a8-bf16-w4a4c8`（**链路版**：W2A8→bf16→W4A4C8，190G） | `...-vq2a8-mixed-gs32-repack`（**MoE=W2A8 + attn=W8A8 混合**，102G） |
| 量化 | `ascend`（attn=W8A8_MXFP8，experts=W4A4_MXFP4） | `vq2a8`（**attn=W8A8**：F8_E4M3+block128 e8m0（ds_fp8_block128），**experts=W2A8**：VQ2/MXFP4 混精 GS32） |
| KV 传输 | MooncakeHybridConnector + 双超时 3600s + block 32 + proxy 8000 | 同左 |

同源派生措施（对标 10T 方案"P 的 W4 由 D 侧 W2A8 生成"）：**experts 已同源**（W2A8→bf16→W4A4），attn 未同源（经 bf16→MXFP8 重量化）。

### 崩溃现象

smoke 请求（"计算 23×47"）输出：
- `finish_reason: length`（300 token 耗尽）、`content` 为空
- **reasoning 字段为乱码循环**：`"6?? No? 91s: 6?? No? 91s: ..."` 持续复读直至截断

对照：P 权重单机 DP8 smoke 正常（E3b.5，含 1081 ✅）；D 权重单机/同体系 PD（W2A8+W2A8，EPLB 五模式矩阵）全部正常。**两侧各自正常，组合乱码**——问题在交互面（KV 传输/表征一致性）。

### 与 10T 方案的对照

| | 10T 方案（参考） | E4 |
|---|---|---|
| 拓扑 | P(W4A4) → KV → D(W2A8+W4A8) | 同构 |
| 独立量化的后果 | KV 漂移逐层累积（cos 0.991→**0.893**，相对误差 0.116→**0.453**）→ AIME/GPQA 崩溃 | 乱码循环（更严重） |
| 同源派生解法 | P 的 W4 由 D 侧生成 → 恢复（AIME 61.8 vs 官方 62.3） | 半套（experts 同源） |

---

## 2. 定量证据一：attn 权重逐层漂移（43 层全量实测）

> ⚠️ **方法局限声明（2026-11-03 修订）**：本节测的是 **wkv 权重矩阵**（静态权重），
> 而非 10T 文档所测的 **KV cache 本身**（两侧完整前向同一输入的逐层累积效应）。
> 单层注入误差小（第 0 层 KV cos 0.9994）**不能推出深层 KV 误差也小**——误差经
> 43 层 attention/MoE 复合传播可能显著放大（10T 数据正是该模式：层 0-9 cos 0.991
> 尚可，30-42 层衰减到 0.893）。**本节数据仅作"单层注入参考值"，不构成对 10T
> 漂移机制的排除**；真正的逐层 KV 累积漂移由 **E5 实验**测定（见 §6）。

**方法**：P/D 两侧 wkv（KV 生成投影，[512,4096]）分别反量化为 fp32 后逐层对比。
- D 侧：fp8_e4m3 + e8m0 scale [4,32]（128×128 block）
- P 侧：fp8_e4m3 + uint8 scale [512,128]（MX per-32 列 e8m0）

| 层组 | cos 均值 | 相对误差（均值/中位/P90） |
|---|---|---|
| 0-9   | 0.9986 | 0.009 / 0.010 / 0.011 |
| 10-19 | 0.9983 | 0.009 / 0.009 / 0.010 |
| 20-29 | 0.9991 | 0.009 / 0.009 / 0.010 |
| 30-39 | 0.9984 | 0.009 / 0.009 / 0.010 |
| 40-42 | 0.9985 | 0.009 / 0.009 / 0.010 |

最差层 L34 cos=0.9978。**第 0 层 KV 输出实测**（随机 hidden 2048×4096 过两侧 wkv）：**cos=0.99942，相对误差=0.0104**。

**解读（修订后）**：单层注入起点优于 10T 独立量化场景（0.9994 vs 0.991），但**深层累积趋势未知**——若累积斜率与 10T 相近，深层 KV 仍可能跌至崩溃区间。

## 3. 定量证据二：config 结构一致性核查

P 的 config 拷自 FP8 目录（`--fp8-config`），D 用 vq2a8 原生 config。AutoConfig 运行时解析后 5 个字段表面不一致，逐一核实代码消费：

| 字段 | 表面差异 | 代码核实 | 结论 |
|---|---|---|---|
| `qk_nope_head_dim` | P 无 / D=448 | 实现不读此键：`nope = head_dim − qk_rope_head_dim`（`model.py:481`） | ✗ 排除 |
| `rope_interleave` | P 无 / D=True | vllm-ascend/vllm 的 DSV4+rope 全链无消费 | ✗ 排除 |
| `compress_rope_parameters` | P 无 / D 有 | 只读 `compress_rope_theta`（=160000 两侧同，`model.py:541`） | ✗ 排除 |
| `kv_lora_rank` / `v_head_dim` | P 无 / D=None | None ≡ 缺失 | ✗ 等价 |
| `dspark_*`（block_size=5 等） | P 有 / D 无 | 仅 `DSparkDeepseekV4ForCausalLM` 变体/投机方法读取；两侧 `model_type` 均 `deepseek_v4` 走普通类 | ✗ 排除 |

**→ 两侧运行时模型结构一致**（rope/维度/压缩参数实际值相同）。

## 4. 定量证据三：KV dtype 一致性（A5 机制）

A5+DSV4 的 KV cache dtype 决策链（`dsa_attn_kv_plan.py`）：

```python
# get_dsv4_attn_kv_dtype (L52-58)
return bf16 if (非A5 or 显式bf16) else torch.float8_e4m3fn
# A5 硬件位: hardware_profile.py capabilities 含 DSV4_COMPRESSED_CACHE
# model.py: fp8 时回写 cache_dtype="float8_e4m3fn", cached_head_size = head_dim + 128(scale 内嵌)
```

- CLI `--kv-cache-dtype auto`（E4 两侧均未显式指定）→ **两侧 KV 同为 fp8_e4m3 + 128 内嵌 scale 布局**
- P/D 走**同一个** `DeepseekV4` 模型类的 `get_kv_cache_spec`，与各自量化方式无关

**→ KV dtype/格式错读排除**（"C8 int8 vs bf16"假说不成立：P 产物 desc 无 `kv_cache_type`/`kv_cache_scale` 键，C8 从未启用）。

### 附：KV 量化的控制机制（澄清）

| 机制 | 控制方 | 与 E4 关系 |
|---|---|---|
| DSV4 内建 fp8 KV | 模型类 + A5 硬件位（`get_dsv4_attn_kv_dtype`），`--kv-cache-dtype bfloat16` 可显式关闭 | 两侧默认同 fp8 |
| vLLM 上游 fp8 KV | `--kv-cache-dtype fp8/e4m3/e5m2` | 未使用 |
| msmodelslim C8 int8 KV | 权重产物 `quant_model_description.json` 的 `kv_cache_type:"C8"`+每层 scale 键（`modelslim_config.py:1107`），**serve 无参数可关**；dense-attention 设计，PD 下仅 consumer 侧 int8（`kv_c8.py:126`） | E4 的 P 产物未启用（label `kv_cache:True` 是标签，spec 未实施） |

---

## 5. 排除项汇总（修订后）

| # | 嫌疑 | 状态 | 证据 |
|---|---|---|---|
| 1 | ~~attn 投影重量化漂移（10T 机制）~~ | **回归待验（E5）** | §2 权重/单层数据只是注入参考，**累积效应未测**——修订后不再视为已排除 |
| 2 | config 结构差异 | ✅ 排除 | 5 字段逐一核实无消费/等价（§3） |
| 3 | KV dtype 格式错读（C8 vs bf16） | ✅ 排除 | C8 未启用；两侧同为模型内建 fp8（§4） |
| 4 | dspark 特性错位 | ✅ 排除 | 普通模型类不读（§3） |
| 5 | P/D 权重各自损坏 | ✅ 排除 | E3b.5 单机 smoke 含 1081 ✅；W2A8 侧全量验证过 |

## 6. 未决嫌疑与验证方案（修订后：双嫌疑并存）

**嫌疑 A（回归）**：**累积 KV 漂移（10T 机制完整形态）**——单层注入 1% 经 43 层 attention/MoE 复合传播后的深层 KV 误差未知；10T 数据（0.991→0.893 衰减）表明该机制足以造成崩溃级输出质量劣化。
**嫌疑 B**：**P 侧 ascend 量化权重体系 × mooncake PD 传输链路的交互缺陷**——只在"跨权重体系传 KV"时发作（单机各自正常、同体系 PD 正常）。

### 实验序列（按判定力/成本排序）

| 序 | 实验 | 方法 | 判定 |
|---|---|---|---|
| **E5（先跑，20 分钟/侧×2）** | **两侧单机逐层 KV 对比**（真正对齐 10T 表格） | 同一固定 prompt 分别单机跑 P 权重与 D 权重，逐层 dump KV pool 首请求 block，逐层 cos/相对误差表（数据区/scale 区分开） | 深层(30-42) cos ≤0.93 或显著单调衰减 → **嫌疑 A 坐实**（修复方向=P 权重 attn 同源不重量化），E6 可免；深层 cos ≥0.99 无衰减 → **嫌疑 A 排除**，转 E6 |
| E6（E5 排除 A 后） | PD 传输 dump 比对（工具已就绪） | `tools/eplb/install_kv_block_hook.sh` + P=W2A8（正常基线）与 P=W4A4C8（乱码）各跑一次 PD smoke，比对 D 收到的 KV block vs P 写入值 | 数据区损/scale 区损分离 → 定位传输具体环节 |
| 二分实验（可选） | P=E2 直量化 W4A4C8 + D=W2A8 重跑 smoke | 同乱码 → 与量化源无关，锁定 ascend 体系×传输；正常 → 链路版权重独有 | |
| bf16 KV 对照（可选） | 两侧 `--kv-cache-dtype bfloat16` | 若 bf16 KV 下 PD 恢复 → 缺陷在 fp8 KV 的 scale 搬运/语义 | |

### E5 实施要点

- **dump 触发**：单机无 mooncake 路径（E6 的两个插桩点不触发）——挂 `/stop_profile` 端点（服务需带 `--profiler-config`）：`stop_profile` 处理时 dump `model_runner.kv_caches` 每层前 N block（复用 `kv_block_hook.dump_side` 的 single 模式）
- **流程**：起服务（P 权重）→ 发固定 prompt（max_tokens=16，temperature=0）→ `curl POST /stop_profile` → dump 落盘 → 换 D 权重重复 → 离线 `kv_layer_compare.py --scale-tail 128`（数据区/scale 区分层输出，10T 同款层组表）
- **注意**：两侧 dump 的 block 需同 prompt 同 block_id 对齐（首请求 prompt ≤32 token → 每层 1 个 block，天然对齐）；DP4/DP8 下请求只落一个 rank，dump 需含 rank 标识

### 修复方向预案（按 E5/E6 判定结果）

- **A 坐实**：重出 P 权重——yaml 仅量化 `*ffn*`（attn/shared_experts 保持 FLOAT），使 attn 与 D 侧数值同源（10T 方案"P 由 D 生成"的完整语义），再跑 E4''
- **B 坐实**：检查 ascend 量化 attention 的 KV 写入路径（`AscendLinearMethod` 与 cache 写解耦）/ fp8 KV scale 区搬运语义；兜底 P 侧 `--kv-cache-dtype bfloat16`

---

## 附录：关键代码位置索引

| 内容 | 位置 |
|---|---|
| KV dtype 决策 | `vllm_ascend/attention/dsa_attn_kv_plan.py:52-58`（`get_dsv4_attn_kv_dtype`） |
| KV spec/回写/head+128 | `vllm_ascend/models/deepseek_v4/model.py:128-132` |
| A5 能力位 | `vllm_ascend/device/hardware_profile.py`（capabilities 含 DSV4_COMPRESSED_CACHE） |
| C8 KV 启用判定 | `vllm_ascend/quantization/configs/modelslim_config.py:1107-1120` |
| C8 实现（dense/PD 设计） | `vllm_ascend/quantization/methods/kv_cache/kv_c8.py:126-137` |
| nope 维度推导 | `vllm_ascend/models/deepseek_v4/model.py:481` |
| attn 逐层漂移数据 | §2（2026-11-03 执行） |
