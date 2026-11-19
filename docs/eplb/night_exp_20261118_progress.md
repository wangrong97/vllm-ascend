# 夜间实验计划（2026-11-18 晚）

## 实验目标

PD 分离（P eager + D graph·B64）+ eplb=off 形态下，**三组精度实验**（AIME24 + GPQA × 3 轮），核心验证：**PD 分离部署下 W2A8 vs W4A4 混搭的可行性**。E3 为**权重生产**（不跑精度评测，产出供 E4 使用）。

## 三组实验（顺序执行）

| # | 名称 | P（prefill） | D（decode） | 权重说明 |
|---|---|---|---|---|
| **E1** | FP8 基线 | FP8 | FP8 | `DeepSeek-V4-Flash-0731`（原始 FP8） |
| **E2** | W2A8 基线 | W2A8 | W2A8 | `32x32-noperm-hot5050125-identity-v3-export-20261103`（**P=D 同权重**） |
| **E4** | **PD 混搭** | **W4A4（E3 产物）** | **W2A8** | P=W4A4 + D=W2A8（**先 smoke 验证对话正常再放评测**） |

## E3：W4A4 权重生产（不跑精度评测）

```
步骤1: 反量化  W2A8 repack → bf16 (~30min, CPU不占卡)
       tools/convert_vq2a8_to_bf16.py --source vq2a8-repack --output <bf16>
步骤2: 量化    bf16 → W4A4C8 (~52min, 单卡)
       cd msmodelslim && bash msmodelslim.sh (deepseek-v4-flash-0731-w4a4.yaml)
步骤3: attn对齐  attn/root 硬链 D 的 W2A8 repack（W8A8 逐比特一致）
       tools/eplb/convert/make_attnD_vq2a8.py --src <w4a4产物> --out <w4a4-attnD>
```

**E3 产物目标**：`weight/DeepSeek-V4-Flash-0731-32x32-noperm-w4a4c8-attnD`（合法 vq2a8 checkpoint，MoE=W4A4、attn/root 与 D 逐比特相同）——供 E4 的 P 侧使用。

## 执行流程（每组精度实验）

```
wait_free(8卡空闲) → 起 PD(脚本) → proxy:8000 → smoke(自我介绍+首都+23×47含1081)
→ ais_bench AIME24(30题×3轮) → ais_bench GPQA(198题×3轮) → 停服清场 → 汇总精度
```

## 关键注意事项

| 点 | 处理 |
|---|---|
| **E4 必须 smoke 通过才放评测** | P=W4A4 + D=W2A8 是混搭，E4 当年因 attn 量化逻辑不一致导致乱码——本次 attn 用 `make_attnD_vq2a8.py` 与 D 的 W2A8（实为 W8A8）逐比特对齐，理论上可解 |
| **E3 attn 对齐** | `make_attnD_vq2a8.py --src <w4a4新权重> --out <w4a4-attnD>`：attn/root 硬链 D 的 W2A8 repack（W8A8/F8_E4M3+block128 E8M0） |
| **W2A8 反量化** | `tools/convert_vq2a8_to_bf16.py`（CPU 不占卡，~22016 投影，~531G 输出；源为 **noperm-export**（用户钦定），perm 走恒等回退——dequant_vq2 内 perm shape 校验不符时 `torch.arange` 恒等） |
| **E4 权重组合** | MODEL_P=E3产物（w4a4c8-attnD），MODEL_D=W2A8 repack |
| **评测口径** | ais_bench `vllm_api_general_chat`（host_port=8000，dsv，temp=1.0，top_k=20，top_p=1.0，batch=32） |
| **eplb** | 三组全部 off（动态负载均衡不影响精度对比） |

## 预期产出

```
eplb/output/night_exp_20261118/
├── E1_fp8/                     # FP8 基线产物（AIME24+GPQA ×3 轮）
├── E2_w2a8/                    # W2A8 基线产物
├── E3_w4a4/                    # W4A4 链路中间产物（bf16反量化、量化、attn对齐日志）
└── E4_mixed/                   # 混搭产物（含 smoke 门禁记录）
```

## 进度跟踪

| 组 | 状态 | AIME24 | GPQA | 备注 |
|---|---|---|---|---|
| E1 FP8 | ✅ 07:24 完成 | **100.0**（100×3） | **90.07**（89.90/89.90/90.40） | FP8×PD×DP4EP4 无 OOM；第三轮修复后一次通过 |
| E2 W2A8 | 🔄 07:27 起 | — | — | noperm-export，E1 后顺序执行 |
| E3 W4A4 | ✅ 21:52 全链完成 | —（不评测） | —（不评测） | 反量化 27.8min/531G → 量化 37min(卡4) → attn 对齐 54min → attnD 产物 146G/45596 键 |
| E4 混搭 | ⬜ 待跑 | — | — | smoke 门禁 |

预计时间线：E3（反量化 ~30min → 量化 ~52min → attn 对齐 ~20min）先行，E1/E2 顺序执行，E4 最后（E3 产物就绪后）。

## 执行日志

- 17:39 午夜定时器布置（pid 1111938，00:00 → E1 FP8 PD，日志 `output/pd_e1_fp8_1118.log`）
- 17:41 反量化首发（nohup 日志丢失，静默死亡）
- 19:22 反量化重启（noperm-export → bf16，日志 `output/night_exp_20261118_dequant.log`）→ 19:50 ✅ 完成（22016 投影，27.8 分钟，531G）
- 19:45 **E1 启动修复**：`run_pd_w2a8_eager_graph.sh` 硬编码 `--quantization vq2a8` 与 FP8 基线权重不匹配（加载必崩）→ 增加 `QUANT=auto|fp8|vq2a8`（按 MODEL_P config.json 识别）+ `KV_BYTES/MAX_LEN/MEM_UTIL` 资源旋钮；shim 回归验证 vq2a8 参数与改前逐字节一致、fp8 模式无 vq2a8 注入 ✅
- 19:59 E3 步骤 2 量化启动（msmodelslim 卡 3，日志 `output/night_exp_20261118_quant.log`，产物 `weight/DeepSeek-V4-Flash-0731-32x32-noperm-w4a4c8`）
- 20:03 ❌ 量化失败①：`decode_fp4 崩溃 16384 vs 1024`。排查：① config.json 沿用了 noperm-export 的非标字段（缺 dspark_target_layer_ids 等）→ 替换为 FP8 原版转换（config.json.noperm.bak 备份）；② 仍崩 → 真根因：`convert_vq2a8_to_bf16.py` 把 non_routed `.to(bfloat16)` 全转，而 msmodelslim `auto_dequant_state_dict`（WEIGHT_SCALE_INV='.scale'）期望 **F8_E4M3+E8M0 原格式**——BF16 落入 decode_fp4 分支 shape 崩。11-03 成功链路（w2a8-bf16）正是保留原格式的
- 20:12 修复：脚本改回原样直拷（含教训注释）；仅重生成 non_routed.safetensors（1564 张量，experts 22019 文件不动）；dtype 与成功链路逐项对齐 ✅
- 00:07 **E1 起服成功 + smoke ✅**（FP8×PD×DP4EP4 无 OOM，未走降档）；但 6 遍评测瞬间全崩：`from datasets import Dataset` 被 `benchmark/ais_bench/datasets/__init__.py`（空文件，数据集目录）遮蔽 → 改名 `__init__.py.shielded_1119` 后 dry-run 验证 ✅（00:12 修复，赶在 E2 评测开始前）
- 00:13 **finished_guard.sh 上线**：编排尾部"无条件 touch .finished"缺陷（E1 缺遍也会终结链）由独立 guard 兜底——finished 出现但 e1/e2/e4 done 不齐 → rm finished + 重启 supervisor 补跑（编排运行中不可安全编辑文件，缺陷待编排退出后修）
- 00:31 **E4 起服两连败**：① 引擎就绪 600s 超时（attnD 大权重 NFS 加载慢）→ serve 脚本加 `VLLM_ENGINE_READY_TIMEOUT_S=1800`；② 降档重试秒败——`${KV_BYTES:+KV_BYTES=$KV_BYTES}` 展开产物不被 bash 识别为赋值前缀 → 被当命令执行（command not found），serve 根本没起 → start_server 改 `env` 数组显式传递
- 00:33 旧编排按旧尾部逻辑误 touch .finished（"全部结束 ✅"假象）→ 尾部改为 done 标记检查（不全则 exit 1 由 supervisor 重拉）
- 00:41-00:44 **E1 重跑轮 6 遍又全 "-"（第 5 坑）**：数据集路径 `ais_bench/datasets/...` 相对解析到 site-packages（无数据文件）。破案：15:31 成功那次的 cwd 是 **`/mnt/share/w00608002/work/benchmark`**（ais_bench 配置/数据全按 cwd 相对解析；cd 到仓库根会 fallback site-packages 配置）→ 编排 BENCH 变量改指 benchmark 目录，dry-run 验证 ✅（仓库配置+数据全通）
- 00:46 新编排拉起（清熔断计数防误熔断），E1 第三轮开始——五坑全修
- 20:25 **night3 式自愈编排上线**（`tools/eplb/eval/night1118/`，用户指令参考 glm night3 机制）：
  - `night_exp_1118.sh` 主编排：E3 收尾（等量化→attn 对齐）→ gate（00:00+8卡稳定60s）→ E1（起服失败自动降档 KV=8G/len=65536 一次）→ smoke → AIME×3+GPQA×3 → E2 → E4（乱码门禁不过→留样+熔断 .halt）
  - `night_exp_supervisor.sh`：60s 重拉 + 30min 内 ≥3 次熔断 → `.halt`
  - 断点：评测按 summary 成绩秒过；失败遍补跑；90min watchdog；停服 60s HCCL 缓冲
  - **旧裸定时器（pid 1111938）已退役**，午夜启动由编排 gate 接管；主日志 `output/night_exp_20261118/main.log`
- 21:55 熔断（机制按设计工作）：`local st=$1 … SL=$OUT/serve_${st}.log` 单行引用同声明的 st → set -u unbound 崩（E3 完成进 run_stage 即死，3 连重拉全崩 → 熔断）。22:37 修复（local 拆两行）+ bash -x 验证 + rm .halt + 重启 supervisor → 22:41 编排进入 E1 gate 正常待午夜。教训：**supervisor 熔断 exit 后需人工重启**（rm .halt 不会自动复活 supervisor）
