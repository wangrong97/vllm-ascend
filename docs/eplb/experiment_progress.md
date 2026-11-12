# EPLB 实验进展跟踪文档

> 用途：记录 W2A8/W4A4 量化权重实验链（E0-E6）与 B 系列/性能采集任务的进展与结论，作为**断点续跑依据**。每个阶段状态变化后更新本文档。最近更新：**2026-11-09 晚**（定时 B2→B1 评测布置、EPLB 性能采集工具 v3.1 定稿）

---

## 1. 实验目标与矩阵

核心问题：对比不同量化权重（FP8 / W2A8 / W4A4）在 DeepSeek-V4-Flash 上的精度，并验证 **PD 混搭（P=W4A4, D=W2A8）** 的可行性。精度口径统一：ais_bench，AIME2024(30题) + GPQA_diamond(198题) 各 5 轮（temp=1.0/top_k=20/batch 32），一键脚本 `/mnt/share/w00608002/work/benchmark/aisbench.sh`（共 10 条命令，勿改）。

> **术语澄清**：所谓"W2A8 权重"（`...-vq2a8-mixed-gs32-repack`）实为混合量化——**MoE 路由专家 = W2A8**（VQ2 2bit + MXFP4 4bit 混精，走 GS32 算子；quant.json counts: w13_vq 10313 / w13_mxfp4 695 / w2_vq 8249 / w2_mxfp4 2759），**attention/root = W8A8**（F8_E4M3 + block[128,128] e8m0 scale，走 `ds_fp8_block128` DeepSeek FP8 线性路径，vq2a8_config.py:323）。文中 "D=W2A8 / D(W2A8)" 指该权重目录整体；涉及其 attn 部分时严格说是 W8A8。

| 阶段 | 内容 | 状态 | 结果摘要 |
|---|---|---|---|
| **E0** | FP8 基线：DP8·图模式·port9000 | ✅ 完成（5+5 轮） | **AIME 99.33±1.33 / GPQA 91.11±0.76**（详见 §2 结果表） |
| **E1** | W2A8 基线：DP8·**图模式**·rows48 修复版 | ✅ 完成（4+3 有效轮） | **AIME 94.17±4.93 / GPQA 88.22±1.26**（详见 §2 结果表） |
| E1b | W2A8：DP8·serial+eager·seqs128 | 🛑 用户暂停（smoke✅） | 备选形态，已弃 |
| **E2** | FP8→W4A4C8 量化（msmodelslim） | ✅ 完成 | 48 分片/190G |
| E2.5 | W4A4 对话 smoke | ✅ 通过（eager 形态，含 1081） | |
| **E3a** | W2A8→bf16 反量化（自研脚本） | ✅ 完成 | 34588 键/531G，三重校验通过 |
| E3a.5 | bf16 对话 smoke | 🚫 取消（用户指示） | |
| **E3b** | bf16→W4A4C8（链路版量化） | ✅ 完成 | 48 分片/190G，smoke 含 1081 |
| **E4** | PD 混搭：P=链路W4A4 + D=W2A8 | ❌ 乱码崩溃 | 根因分析见 `docs/eplb/E4_PD混搭精度崩溃分析.md` |
| **E4'** | PD 混搭：P=attnD 数值对齐版 + D=W2A8 | 🚫 已废弃 | 用户指示改用合法 vq2a8 格式（量化逻辑须一致） |
| **E4''** | PD 混搭：P=**moeW4A4-attnW8A8**（合法 vq2a8 格式）+ D=W2A8 | ✅ smoke 通过（2026-11-07） | 3/3 对话正确（含 1081），**E4 乱码根因锁定为 P 侧 attn 量化逻辑不一致**，KV 传输无嫌疑；可接 aisbench |
| **E5** | 两侧单机逐层 KV 累积漂移对比（E4 主判定实验） | ⬜ 规划就绪 | 工具三件套已备，前置 E4'' 结果 |
| **E6** | KV block 逐元素比对（E4 终局定位） | ⬜ 工具就绪 | 前置 E5 判定嫌疑 A 排除 |
| **E4''-4over6** | PD 混搭：P=**4over6-w4a4c8-attnD** + D=W2A8 | ✅ smoke 通过（2026-11-09） | 3/3 对话正确（含 1081）——4over6 MoE 量化在 PD 链路可用；**aisbench 评测因 warmup OOM/worker 孤儿连环事故未完成，待重跑**（并发须 ≤8） |
| **B2** | W2A8 精度：EP8·tp-ep·full_graph·:8003 | 🔄 **定时 11-10 01:00 启动** | evalscope 口径（gpqa/aime24/aime25 各5遍·并发64·seed0）；evalscope 已装、参数链已修 |
| **B1** | FP8 精度：DP8·图模式·:8004 | 🔄 B2 完成后自动接续 | 同口径·并发16（32 会 OOM）；两次编排串联（--skip-b1 → --skip-b2），断点续跑在位 |
| **PERF-EPLB** | EPLB 性能对比：PD 分离 DP4×EP4，D 侧 eplb 三模式（off/dynamic/dynamic_uniform） | ⬜ 工具就绪待跑 | GSM8K 300条·aisbench perf·batch8·profiling(P/D 分目录)；`run_eplb_perf_gsm8k.sh` v3.1 已全链体检（权重同源/端口/IP/profiling/防 hang） |

### 📊 精度总表（所有已完成评测实验）

| 实验 | 权重 | 形态 | AIME 均值±σ | GPQA 均值±σ |
|---|---|---|---|---|
| E0 | FP8 | DP8·图模式 | 99.33±1.33 (n=5) | 91.11±0.76 (n=5) |
| E1 | W2A8 | DP8·图模式 | 94.17±4.93 (n=4) | 88.22±1.26 (n=3) |
| Δ | — | — | -5.16 | -2.89 |
| E4 | 链路W4A4+W2A8 (PD) | P4+D4 | ❌ 乱码崩溃（未评测） | ❌ |

（E4'' 待跑后补入；E1 后 3 轮 aisbench 为空轮不计入，n 为有效轮数）

### 权重资产（/mnt/share/w00608002/weight/）

| 路径 | 说明 | 状态 |
|---|---|---|
| `DeepSeek-V4-Flash-0731` | FP8 原始（156G） | ✅ |
| `DeepSeek-V4-Flash-0731-vq2a8-mixed-gs32-repack` | W2A8 混精（102G） | ✅ |
| `DeepSeek-V4-Flash-0731-w4a4c8` | E2 产物（190G） | ✅ |
| `DeepSeek-V4-Flash-0731-w2a8-bf16` | E3a 产物（531G） | ✅ |
| `DeepSeek-V4-Flash-0731-w2a8-bf16-w4a4c8` | E3b 产物（190G） | ✅ |
| `...-w2a8-bf16-w4a4c8-attnD` | P'（attn 数值对齐版，已废弃） | ⚠️ 可删 |
| `...-moeW4A4-attnW2A8`（attn 实为 W8A8） | **P''（合法 vq2a8 格式，E4'' 的 P 侧：attn/root 逐比特同 D，MoE=W4A4）** | ✅ 146G |

---

## 2. 已完成实验详情

### E0 · FP8 基线 ✅（2026-11-02 22:09 ~ 11-03 02:38，10 轮全）

形态：`run.sh`（FP8 · DP8×EP · 图模式 FULL_DECODE_ONLY · 无 MTP · async · port 9000）。MTP 移除原因：该 FP8 权重的 mtp 键为 `mtp.0.hc_*` 原始命名，与 vLLM MTP 加载器（`model.layers.N.mtp_block.*`）不兼容。

**逐轮精度**（原始数据在 `benchmark/outputs/default/20261102_220909~20261103_020505/`）：

| # | 轮次 | AIME2024 | # | 轮次 | GPQA_diamond |
|---|---|---|---|---|---|
| 1 | 1102_220909 | 100.00 | 1 | 1102_223133 | 91.41 |
| 2 | 1102_231433 | 100.00 | 2 | 1102_233135 | 90.40 |
| 3 | 1103_001505 | 100.00 | 3 | 1103_002813 | 90.40 |
| 4 | 1103_011102 | 96.67 | 4 | 1103_011927 | 90.91 |
| 5 | 1103_015045 | 100.00 | 5 | 1103_020505 | 92.42 |

**统计**：

| 数据集 | n | 均值 | 方差 | 标准差 | min | max |
|---|---|---|---|---|---|---|
| AIME2024 | 5 | **99.33** | 1.7742 | 1.33 | 96.67 | 100.00 |
| GPQA_diamond | 5 | **91.11** | 0.5709 | 0.76 | 90.40 | 92.42 |

### E1 · W2A8 图模式基线 ✅（2026-11-03 22:15 ~ 11-04 09:11，有效 7 轮）

形态：`run_w2a8.sh`（W2A8 · DP8×EP · **图模式 full_graph**（rows48 修复版）· 同步调度 · port 9000）。aisbench 计划 10 轮，实际有效 7 轮（aime 4 + gpqa 3）；后 3 轮为空轮——服务结束时 aisbench 仍在排队，无预测结果，不计入。

**逐轮精度**（原始数据在 `benchmark/outputs/default/20261103_221737~20261104_083207/`）：

| # | 轮次 | AIME2024 | # | 轮次 | GPQA_diamond |
|---|---|---|---|---|---|
| 1 | 1103_221737 | 100.00 | 1 | 1103_231428 | 87.88 |
| 2 | 1104_015145 | 96.67 | 2 | 1104_023718 | 86.87 |
| 3 | 1104_045716 | 93.33 | 3 | 1104_053935 | 89.90 |
| 4 | 1104_083207 | 86.67 | — | — | — |

**统计**：

| 数据集 | n | 均值 | 方差 | 标准差 | min | max |
|---|---|---|---|---|---|---|
| AIME2024 | 4 | **94.17** | 24.2986 | 4.93 | 86.67 | 100.00 |
| GPQA_diamond | 3 | **88.22** | 1.5868 | 1.26 | 86.87 | 89.90 |

**E0 vs E1 对比结论**：W2A8 相对 FP8 有可见精度损失——AIME -5.16 分（方差也大增 1.77→24.30，末轮 86.67 提示 temp=1.0 下稳定性更差），GPQA -2.89 分。注意 E1 样本量（4+3）小于 E0（5+5），结论方向性可信但均值置信度略低。

### E1 · W2A8 图模式修复 ✅ 验证生效（2026-11-03 09:36~09:47）

**原始故障**：DP8 图捕获首帧报 `GS32 whole-graph execution supports at most 8 routed rows`。**根因**：DP-attention 在 MoE 前全局 gather → 图捕获看到 `dp_size × B1 × topk = 8×1×6 = 48` 行，而 python 守门上限 8 是按"单 rank B1×topk≤8"的过时假设（官方仅验证过无 DP 的 EP2）。

**修复（方案 A，3 处）**：

1. `vllm_ascend/ops/fused_moe/vq2a8_gs32.py`：`graph_routed_rows` 可由 options 覆盖（默认 8）
2. `vllm_ascend/quantization/methods/w2a8/vq2a8_gs32_format.py`：options 白名单加 `gs32_graph_routed_rows`
3. `run_w2a8.sh`：vq2a8 块加 `"gs32_graph_routed_rows": 48`

**验证**：服务 2 分钟就绪（capture 通过）+ smoke 含 1081 ✅。C++ 算子图执行天然支持动态行。

### PD 场景图模式验证 ✅（2026-11-03 09:59~10:05，`run_pd_graph.sh`）

形态：P(0-3)=W2A8 serial+eager · D(4-7)=W2A8 full_graph 图模式（B1, rows=**24**=DP4×1×topk6）· MooncakeHybrid + 双超时 + block32 + proxy。结果：3 分钟双实例就绪（D 侧图捕获通过）+ smoke 含 1081 ✅。结论：① PD 场景 D 侧图模式可用（rows 按 DP4 取 24，勿沿用 DP8 的 48）；② 旧"PD+full_graph 507034"确认就是 force_load_balance 的锅，修复后不复现。

### 历史成果（前序调试）

- **VQ2A8 PD 分离打通**：① 连接器必须用 `MooncakeHybridConnector`（V1 对 VQ2A8 的 KV spec 协商有断言缺陷）；② Hybrid 内存注册慢 → 需 `HCCL_EXEC_TIMEOUT=3600` + `ACL_DEVICE_SYNC_TIMEOUT=3600` 双超时；③ block-size 32（V1 路径 128/64 均挂整除断言）。
- **EPLB 全模式矩阵**：off/static/dynamic/static_uniform/dynamic_uniform 在 PD 形态下对话全对；原疑"spare 槽 C++ 缺口"证伪，真凶是 dummy-run 的 force_load_balance VQ2A8 专属分支（已删）。
- **代码修复清单**（本分支累计）：MC2 dispatcher dynamic_scale 契约、serial/fused/量化侧三路 spare 槽数一致性、`vq2a8_gs32.py` placement 笔误等。

---

## 3. 性能验证（PERF-EPLB）

> 目标：回答"**混精感知 eplb（dynamic）相对标准 eplb（dynamic_uniform）与 off 基线，在真并发 MoE 负载下的吞吐/时延增益**"。核心变量隔离：P/D 两侧同权重（W2A8），唯一差异为 D 侧 eplb 模式。

### 3.1 实验矩阵

| 组 | D 侧 eplb | 语义 | 说明 |
|---|---|---|---|
| off | `"eplb":"off"` | 基线 | 字节级等同，无重平衡 |
| dynamic | `"eplb":"dynamic"` + options | **混精感知** | typed pools：VQ2/MXFP4 分组冗余槽（R_vq2=16/R_mx4=8），typed costs 驱动重平衡 |
| dynamic_uniform | `"eplb":"dynamic_uniform"` + options | **标准** | width-agnostic：同 pipeline 不区分精度——与 dynamic 的差异即"混精感知"的贡献 |

eplb_options（三组同款）：`{"K":8,"R_vq2":16,"R_mx4":8,"n_stage":2,"w_lo":0.2,"w_hi":5.0,"ewma_lambda":0.05}`

### 3.2 形态与口径（2026-11-09 v3.1 定稿）

| 项 | 配置 |
|---|---|
| 服务形态 | PD 分离：P(卡0-3)+D(卡4-7)，各自 **DP4×TP1×EP4**·eager·seqs8/rank（D 全局并发容量 32），MooncakeHybrid+双超时+block32，proxy :8000 |
| 权重 | P=D=**W2A8 repack**（同源，显式注入 MODEL_P） |
| 压测 | ais_bench `--mode perf`：`vllm_api_stream_chat`（127.0.0.1:8000·dsv·**batch8**·stream·temp0.01·max_out 512）× GSM8K **300 条** |
| profiling | 压测窗口 start/stop（**P :9001 + D :9002 各自** torch profiler，trace 分落 `profiling/{P,D}`；PD 侧 serve 须带 `--profiler-config`，经 `PROF_DIR_P/PROF_DIR_D` 注入） |
| 辅助 | NPU util/HBM/功耗 2s 时序（npu.csv） |
| 防护 | 前置清场（孤儿 worker 3 轮迭代杀）→ gate 15min 超时 → smoke 门禁（1081）→ 压测停滞检测（FIN 零增量 12min 中止跳组）→ 归档 5min 超时 → 每组结束清场核验 |

### 3.3 工具链

| 文件 | 角色 |
|---|---|
| `tools/eplb/eval/run_eplb_perf_gsm8k.sh`（v3.1） | 主编排：三组串行全自动，产物 `output/eplb_perf_<ts>/{off,dynamic,uniform}/` |
| `run_pd_mixed_v2.sh`（参数化版） | 被调：`MODEL_P`/`EPLB_MODE`/`EPLB_OPTIONS`/`PROF_DIR_P`/`PROF_DIR_D` 环境变量注入 |
| benchmark `vllm_api_stream_chat.py` | 压测配置：host 127.0.0.1:8000·dsv·batch8 |

### 3.4 状态：⬜ 工具就绪待跑（已完成全链体检）

体检结论（2026-11-09 晚）：权重同源 ✓ / 端口两层架构对齐（8000→proxy→9001/9002）✓ / IP 统一 127.0.0.1（本机 localhost 优先 ::1 已绕开）✓ / profiling 生效条件就绪 ✓ / 防坑机制齐备 ✓。预计单组 ~35-60min，三组 ~2.5-3h。

### 3.5 已排除的设计坑（勿再踩）

| 坑 | 结论 |
|---|---|
| 单机 v2 形态（serve_vq2a8 serial） | **弃用**：serial 分支 `--max-num-seqs 1` 写死——单流串行下 eplb 无从发挥，三组对比无意义；且 EP8 serial 客户端并发 ≥32 必 hang（NPU 空转 4-5%） |
| eplb JSON 注入 | 必须无空格拼装（`"eplb":"dynamic"`）——带空格裸变量展开会被 shell 分词成多参数（`unrecognized arguments: "off"}}`） |
| profiling 端口 | vLLM profile API 在**各自 vllm 主端口**（9001/9002），且需 serve 带 profiler-config；12347 不存在 |
| warmup OOM | PD 形态客户端并发 32 会触发 cgroup OOM 杀双侧主进程 → **batch ≤8**（8 并发 smoke 8/8 实证安全） |
| worker 孤儿 | 主进程死后 spawn worker 残留占 NPU——清理只能按 npu-smi 进程表逐 pid 杀（cmdline 匹配不可靠） |

---

## 4. 运行指引（断点续跑从这里继续）

### 通用规则（每个阶段都必须遵守）

1. **NPU 空闲 gate**：连续 2 次（间隔 30s）`npu-smi` 进程表空 且 每卡 HBM<5000MB 才启动
2. **新权重先 smoke 再评测**：3 条对话（自我介绍/首都/23×47），必须含 "1081" 且正常 stop，才允许跑 aisbench
3. **每阶段 2 次重试**，仍失败记录后跳下一阶段（依赖缺失则下游同步跳过）
4. **结束必验证 NPU 归零**（E0 教训：trap 清理失效残留 8 进程，E1 空等 6.5h）
5. **日志用绝对路径**（tee 相对路径在 `cd benchmark` 后失效丢日志）
6. 清理进程用 `readlink /proc/$p/exe` 过滤 python，**勿用** `pgrep -f "bash run.sh"` 类模式匹配（命令行含模式串会自杀）
7. aisbench 端口切换：直连 9000（E0/E1/E2.5/E3 系列）vs proxy 8000（E4）——改 `benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_chat.py` 的 `host_port`
8. **阶段产物归档**：`cp -a /mnt/share/w00608002/work/benchmark/outputs/default/<本阶段时间戳>` 到本阶段输出目录 `eval_raw/` + 生成 accuracy_summary.csv；注意只挑本阶段的时间戳

### 各阶段命令

**E4''（当前断点）**：
```bash
cd /mnt/share/w00608002/work/eplb/vllm-ascend
# 起(等空闲后): bash run_pd_mixed_v2.sh   # P(0-3)=P'' 9001 + D(4-7)=W2A8 9002, Hybrid+双超时
# proxy 8000; smoke 3条含1081 → 正常则接 aisbench(benchmark host_port 改 8000); 异常转 E5
```

**B2→B1 精度评测（定时 11-10 01:00 已布置，断点续跑恢复命令）**：
```bash
cd /mnt/share/w00608002/work/eplb/vllm-ascend
# 被杀/中断后恢复(各自续跑补缺轮, 输出新目录看 /tmp/baseline_vs_w2a8_dir):
bash tools/eplb/eval/run_baseline_vs_w2a8.sh --skip-b1   # 只跑 B2 (W2A8 EP8 并发64)
bash tools/eplb/eval/run_baseline_vs_w2a8.sh --skip-b2   # 只跑 B1 (FP8 DP8 并发16)
# 进度: <输出目录>/summary.txt + b2/b1/eval.log 进度条; evalscope 预测落盘 <输出目录>/<side>/<ts>/predictions/
```

**PERF-EPLB 性能采集（PD 分离 DP4×EP4 三组，工具 v3.1 就绪）**：
```bash
bash tools/eplb/eval/run_eplb_perf_gsm8k.sh   # off → dynamic → dynamic_uniform 串行
# 每组: 前置清场→PD起(MODEL_P=W2A8, PROF_DIR注入)→proxy8000→smoke门禁→profiling(9001+9002)→
#       NPU采样→aisbench perf(gsm8k 300条 batch8)→停滞检测→归档→清场
# 产物: output/eplb_perf_<ts>/{off,dynamic,uniform}/{press.log,npu.csv,profiling/{P,D},aisbench/}
```

**E5 两侧单机逐层 KV 累积漂移对比（E4 主判定实验，E4'' 若异常先跑）**：
```bash
# 0) 安装单机 dump 触发(挂 /stop_profile; 幂等, 卸载 --uninstall)
bash tools/eplb/install_kv_layerdump_hook.sh
# 1) 跑 P 权重: 起服务(DP4+, 带 profiler-config) → 固定 prompt 请求
mkdir -p output/e5_layerdump
KV_DUMP_DIR=$PWD/output/e5_layerdump KV_DUMP_TAG=P bash <P权重启动脚本> &   # 就绪后:
curl -s http://127.0.0.1:9000/v1/chat/completions -d '{"model":"dsv","messages":[{"role":"user","content":"计算 23 * 47 等于多少"}],"max_tokens":16}'
curl -X POST http://127.0.0.1:9000/stop_profile     # 触发 dump: kv_single_side_P.pt(按 rank 分文件)
# 2) 清理 → 同法跑 D 权重(W2A8) KV_DUMP_TAG=D → kv_single_side_D.pt
# 3) 离线比对(10T 同款层组表: cos/误差/中位/P90; 数据区与 scale 区分开)
python3 tools/eplb/kv_layer_compare.py --a output/e5_layerdump/kv_single_side_P.pt --b output/e5_layerdump/kv_single_side_D.pt --scale-tail 128
# 判定: 深层(30-42) cos≤0.93 或单调衰减 → 嫌疑A(累积漂移)坐实,E6 可免;
#       深层 cos≥0.99 → 嫌疑A排除 → 转 E6(PD 传输 dump 比对)
# 4) 卸载: bash tools/eplb/install_kv_layerdump_hook.sh --uninstall
```

**E6 KV block 逐元素比对（E4 终局定位，前置 E5 排除嫌疑 A）**：
```bash
# 0) 安装插桩(幂等;卸载加 --uninstall)
bash tools/eplb/install_kv_block_hook.sh
# 1) 起带 hook 的 PD(两侧环境变量都设 KV_DUMP_DIR)
mkdir -p output/e6_kvdump
KV_DUMP_DIR=$PWD/output/e6_kvdump bash run_pd_mixed_v2.sh &   # P/D 由同一脚本派生,env 继承
# 2) 就绪后经 proxy 8000 发 1 条固定短 prompt(≤32 token → 首请求落前几个 block)
curl -s http://127.0.0.1:8000/v1/chat/completions -d '{"model":"dsv","messages":[{"role":"user","content":"计算 23 * 47 等于多少"}],"max_tokens":16}'
# 3) 两侧 dump 自动落盘后离线比对
python3 tools/eplb/kv_block_compare.py --dump-dir output/e6_kvdump --scale-tail 128
# 判定: 全部 byte_eq=1.0 → 传输无损,崩溃在数值语义层;
#       存在不一致 → 定位传输/写入篡改(区分数据区 vs scale 区受损指向具体环节)
# 建议对照: P=W4A4C8(乱码场景) 与 P=W2A8(正常基线) 各跑一次
# 4) 卸载: bash tools/eplb/install_kv_block_hook.sh --uninstall
```

**E2 量化（FP8→W4A4C8）**：`cd /mnt/share/w00608002/work/msmodelslim && bash msmodelslim.sh`（单卡逐层校准，产物 weight/DeepSeek-V4-Flash-0731-w4a4c8）。

**E3a 反量化（W2A8→bf16）**：
```bash
cd /mnt/share/w00608002/work/eplb/vllm-ascend
python3 tools/convert_vq2a8_to_bf16.py \
  --source /mnt/share/w00608002/weight/DeepSeek-V4-Flash-0731-vq2a8-mixed-gs32-repack \
  --output /mnt/share/w00608002/weight/DeepSeek-V4-Flash-0731-w2a8-bf16 \
  --fp8-config /mnt/share/w00608002/weight/DeepSeek-V4-Flash-0731
# 纯 CPU 不占卡;~22016 投影,输出 ~553G;建议先 --limit 8 预检再全量
```

**E3b 链路版量化**：复制 `msmodelslim.sh` 改 `MODEL_PATH=.../w2a8-bf16`、`SAVE_PATH=.../w2a8-bf16-w4a4c8`（MODEL_TYPE 保持 `DeepSeek-V4-Flash-0731`）。

---

## 5. 已知约束/坑速查

| # | 事项 | 要点 |
|---|---|---|
| 1 | W2A8 图模式 DP8 | 必须 `gs32_graph_routed_rows = dp_size×topk`（已修，见 §2-E1） |
| 2 | MTP 投机 | 现有两份权重 mtp 键命名不兼容 vLLM，勿加 speculative-config |
| 3 | VQ2A8 非图模式 | serial/fused 均要求 enforce-eager + graph mode NONE（校验） |
| 4 | VQ2A8 PD | 只能 Hybrid 连接器 + 双超时 3600 + block 32 |
| 5 | VQ2A8 图模式与 batch | 图模式锁 B1（max_num_seqs=1），高并发诉求只能 eager |
| 6 | /dev/shm | **已全部 run 脚本内置守门**（`shm_size_guard`：avail<1G 时自动 `mount -o remount,size=16G /dev/shm`，无权限则警告继续）。容器重启后回落 64M 不再阻塞 |
| 7 | aisbench 数据集路径 | 必须在 benchmark 仓库根运行（`ais_bench/datasets/` 相对路径） |
| 8 | aisbench 输出 | `benchmark/outputs/default/<时间戳>/`（不是 results/） |

## 6. 更新日志

- **2026-11-09 晚 定时 B2→B1 精度评测**：系统级定时器（setsid pid 1055704）11-10 01:00 触发，先 B2（W2A8·EP8·并发64）后 B1（FP8·DP8·并发16），两次编排串联（--skip-b1/--skip-b2），日志 `output/b1b2_timer_1110.log`，断点续跑在位。当日教训链（B 系列评测多轮启动失败）：① evalscope 未装（秒退 rc=1）已装 1.12.0 并入 skill；② trap 清理后 8 worker 孤儿占卡→后续编排 gate 静默等死（wait_free 已加超时+日志）；③ 编排真身 pid 与 nohup 外壳混淆导致杀错层、僵尸编排 trap 险误杀后继任务；④ 凌晨 03:03 外部 cgroup 级清理杀掉编排+guardian 全进程树（guardian 对该类清理免疫失败，已认知）；⑤ /dev/shm 容器重置回落 64M（shm 守门已补进编排）。
- **2026-11-09 EPLB 性能采集（PD 形态）工具链 v3.1 定稿**：`run_eplb_perf_gsm8k.sh`（PD 分离 DP4×EP4，D 侧 eplb off/dynamic/dynamic_uniform 三组，GSM8K 300条·batch8·profiling 分 P/D 目录）。当日连环坑与修复：单机 v2 形态 serial 分支 `--max-num-seqs 1` 写死（单流串行测不出 eplb 差异→弃）+ 批32 hang（EP8 serial 并发上限<32）+ profiling 端口 12347→实际 8000/900x + `--parallel-mode` 已删参数复燃 + eplb JSON 注入空格分词（"off"}} 断参）+ localhost→::1 回退（统一 127.0.0.1）+ P 侧默认 P'' 混搭权重（显式注入 W2A8 两侧同源）+ 归档死循环/proxy 旧实例/诊断空消息。`run_pd_mixed_v2.sh`：EPLB_MODE/EPLB_OPTIONS/PROF_DIR_P/PROF_DIR_D/MODEL_P 全参数化。aisbench `vllm_api_stream_chat.py` 现配置：127.0.0.1:8000·dsv·batch8。
- **2026-11-09 E4''-4over6 smoke 通过 / aisbench 受阻**：新权重 `4over6-w4a4c8-attnD`（make_attnD_vq2a8.py 构造：attn/root 硬链 D + MoE=4over6；首启 `Incorrect source-v3 tensor dtype/shape` 系 scale U8 标注缺 F8_E8M0，35 分片头改标后过）作 P 侧 3/3 对话正确。aisbench 评测两次失败：warmup 32 并发触发 cgroup OOM 杀 P/D 主进程（batch 降 8）+ worker 孤儿占显存致重启 `Free memory 6.28/91.75GiB`；恢复后单轮 AIME 5/30 时会话重启丢失关联，未再续。
- **2026-11-07 E4'' smoke 通过**：P(0-3)=P'' + D(4-7)=W2A8，MooncakeHybrid+双超时+block32+proxy8000。3/3 对话正确（自我介绍/巴黎/1081 全正常 stop）。对照 E4（链路W4A4 P 侧 attn 走 MXFP8 scheme）乱码崩溃 → **E4 根因锁定 P 侧 attn 量化逻辑不一致**，E5/E6（KV 传输/累积漂移排查）无需执行。期间修复重建版 run_pd_mixed_v2.sh：P 侧 additional-config 补必填 vq2a8 块（gs32_library/binary）。同日另完成：vllm_ascend_C 扩展重编译（仓库重克隆产物缺失，makeself/abseil/protobuf/json/gtest 手动下载绕过代理证书问题）、B1/B2 后四卡（DP4/EP4）对话验证 3/3+3/3 通过、serve_b1_b2.sh 新增（--devices 自适应并行度）、run_baseline_vs_w2a8.sh 的 serve_vq2a8.py 参数修正。
- **2026-11-06 恢复重建**：工作区（/mnt/share/w00608002/work/eplb/vllm-ascend）此前丢失，已按 eplb_export_mix_precision 分支重新克隆并恢复全部代码修复（6 处）+ 工具集（tools/eplb 9 件 + tools/convert_vq2a8_to_bf16.py）。output/ 下历史日志不可恢复，但 E0/E1 评测原始数据在 benchmark/outputs/default 完好。本文档与分析文档按上下文重建（内容从最新终态起重建，历史叙事精简）。
- **2026-11-04 15:00** 全部 9 个 run 脚本加入 `/dev/shm` 守门 `shm_size_guard`。P'' 目录深度修复完成（moe/ 子目录统一，深度集合={1}，缺失分片=0）。
- **2026-11-04 00:30** P''（合法 vq2a8 格式版）+ E4''：用户指出 P' 仅"数值等价"不够——attn 必须走与 D 同款量化逻辑（D 的 attn 实为 W8A8 的 ds_fp8_block128），而 P' 塞在 MXFP8 scheme 逻辑不同。新工具 `make_p_vq2a8.py` 构造 `weight/DeepSeek-V4-Flash-0731-moeW4A4-attnW2A8`（合法 vq2a8 checkpoint）：root/attn 硬链 D 原分片（逐比特=D）、MoE 为 W4A4 数值转 vq2a8 MXFP4 expert 布局（35 分片/146G）。E4'（attnD 版）已停、E4'' 接替。
- **2026-11-03 22:10** P'（attn 对齐版）+ E4'：基于 attn 差异分析（q_b 行重排不入 KV / wkv 数值一致 / o 投影 P 侧 FLOAT 单向差异）手工生成 attnD 版；后按用户"量化逻辑一致"指示废弃改用 P''。
- **2026-11-03 17:55** E5/E6 编号调换：两侧单机逐层 KV 对比为 E5（主判定、先跑），PD dump 比对为 E6（终局定位）。
- **2026-11-03 17:50** E4 分析方法修订：识别出"证据一测的是 wkv 权重而非 KV cache 本身"的局限——单层注入 1% 不代表 43 层累积后仍小（10T 数据 0.991→0.893 的衰减正是累积效应）。
- **2026-11-03 16:00 全部实验停止（用户指令）**。E3b.5 通过（链路 W4A4 smoke 含 1081）；E4 乱码崩溃。
- **2026-11-03 15:27** E3b ✅ 链路版 W4A4C8 完成（48 分片/190G，26 分钟）。bf16 校验：完整性/写入一致性/MXFP4 数值三重通过。
- **2026-11-03 12:21** E2 ✅ FP8→W4A4C8 完成（48 分片/190G，52 分钟）。权重三线齐备。
- **2026-11-03 12:05** E3a ✅ W2A8→bf16 全量转换完成（531G，31 分钟；期间修复 tools/bisect 标准库遮蔽——脚本内已加 sys.path 过滤防御）。
- **2026-11-03 10:05** PD 场景图模式验证 ✅（P=eager + D=full_graph rows24）。
- **2026-11-03 09:53** 文档创建。E0 完成(99.33/91.11)；E1 图模式修复(3处)验证生效。
- 2026-11-03 09:36~09:47 图模式修复链：白名单→rows48→capture+smoke 双过。
- 2026-11-02 全天：PD Hybrid 打通、EPLB 五模式矩阵、GSM8K 采集工具集(tools/eplb)、反量化脚本；E0 启动。
