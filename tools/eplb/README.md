# tools/eplb — 专家分配 / KV 调试 / 权重转换 / 评测编排 工具集

按功能分四类目录。所有插桩类工具**默认关闭零开销**（由环境变量激活），可 `--uninstall` 还原。

## 目录结构

```
tools/eplb/
├── collect/              # 采集类(EPLB 专家负载 / GSM8K 压测)
│   ├── eplb_tools/       # 专家分配与负载采集
│   └── kv_debug/         # KV block/layer dump hook(E5/E6 实验)
├── convert/              # 权重格式转换(W4A4→vq2a8 / 反量化)
└── eval/                 # 评测编排(baseline vs W2A8 对比)
```

## 1. collect/eplb_tools — 专家分配与负载采集

| 文件 | 职责 | 用法 |
|---|---|---|
| `expert_counter.py` | 运行时专家路由计数 hook(由 install_expert_counter.sh 注入 routed_experts.py；计数留 NPU、退出/定期落盘) | 环境变量激活：`EPLB_EXPORT_DIR=<目录>`；`EPLB_FLUSH_EVERY=2000`；产出 `expert_counts_rank*.json` + `resident_experts_rank*.json` |
| `install_expert_counter.sh` | 幂等安装/卸载插桩 | `bash install_expert_counter.sh`（装）；`--uninstall`（卸） |
| `expert_placement_report.py` | 静态分配报告：解析权重 index.json 得每专家每投影精度(vq2/mxfp4)；placement 支持 block/stride，运行时真值优先 | `python expert_placement_report.py --model <模型目录> --ep-size 4 --out <输出目录> [--runtime-dump <采集目录>]` |
| `npu_load_sampler.py` | npu-smi 时序采样(util/HBM/功耗 → CSV，零依赖) | `python npu_load_sampler.py --output out.csv [--interval 1] [--duration 0]` |
| `gsm8k_client.py` | GSM8K 并发压测(SSE 流式，记 TTFT/时延/token) | `python gsm8k_client.py --endpoint http://127.0.0.1:9000 [--num-prompts 200 --concurrency 8 --max-tokens 512] --output req.csv` |
| `plot_expert_report.py` | 2×2 组合图(静态分配/卡负载时序/专家热力图/卡负载汇总) | `python plot_expert_report.py --dir <采集目录> [--out png]` |

## 2. collect/kv_debug — KV dump 插桩（E5/E6 实验）

| 文件 | 职责 | 用法 |
|---|---|---|
| `kv_block_hook.py` | dump 核心模块（P/D/single 三模式，幂等单次落盘） | 由 install 脚本注入调用 |
| `install_kv_block_hook.sh` | E6 插桩：mooncake_connector.py 两处（P 侧 worker.get_finished + D 侧 batch_transfer_sync_read 后） | `bash install_kv_block_hook.sh`；环境变量 `KV_DUMP_DIR=<目录>` 激活；`--uninstall` 卸载 |
| `install_kv_layerdump_hook.sh` | E5 插桩：worker.py 的 `Worker.profile()` stop 分支 → dump model_runner.kv_caches 逐层 | 同上；服务需带 `--profiler-config`；触发 = `curl POST /stop_profile` |
| `kv_block_compare.py` | E6 比对：P/D 两侧 dump 的逐元素 diff(数据区与 fp8 行尾 128 宽 scale 区分开) | `python kv_block_compare.py --dump-dir <目录> [--scale-tail 128]` |
| `kv_layer_compare.py` | E5 比对：两侧单机 dump 的逐层层组表(10T 同款：cos/rel 均值·中位·P90 + 趋势判定) | `python kv_layer_compare.py --a <P.pt> --b <D.pt> [--scale-tail 128] [--block 0]` |

## 3. convert — 权重格式转换

| 文件 | 职责 | 用法 |
|---|---|---|
| `make_p_vq2a8.py` | 构造 P''：合法 vq2a8 checkpoint（root/attn 硬链 D 原分片逐比特相同，MoE=W4A4 数值转 MXFP4 expert 布局） | `python make_p_vq2a8.py <输出目录>`（源写死：D=W2A8 + W=W4A4C8，改脚本内 D/W 变量） |
| `make_attnD_vq2a8.py` | 构造 attnD 版：attn/root 硬链 D（W8A8/ds_fp8_block128 逐比特同），MoE 保留**任意 msmodelslim 量化权重**(W4A4C8 系)的数值（w1+w3→w13 合并、scale 位模式直拷）。make_p_vq2a8 的参数化泛化版 | `python make_attnD_vq2a8.py --src <msmodelslim量化目录> --out <输出目录> [--limit N]`。D 写死为 W2A8 repack。要求：src 有 `quant_model_weights.safetensors.index.json` 且 expert 键为 `layers.N.ffn.experts.E.{w1,w2,w3}.{weight,weight_scale}`（U8 存储）。输出须与 D 同文件系统（rest/ 硬链）；`--limit N` 只转前 N 层做快速验证，全量必须不带 limit 重跑。例：`python make_attnD_vq2a8.py --src .../w2a8-bf16-4over6-w4a4c8 --out .../4over6-w4a4c8-attnD`（43 层全量约 20 分钟，产物 146G/35 moe 分片） |
| `fix_p2_scale_dtype.py` | P'' 修复：expert `.scale` 键 safetensors dtype 从 U8 改标 F8_E8M0（数据字节不变） | `python fix_p2_scale_dtype.py`（目录写死） |
| `../convert_vq2a8_to_bf16.py`（tools/ 下） | W2A8→bf16 反量化（VQ2 四字段 + MXFP4 解码，输出对齐 FP8 命名） | `python tools/convert_vq2a8_to_bf16.py --source <W2A8目录> --output <bf16目录> [--fp8-config <FP8目录>] [--limit N] [--device cpu]` |

## 4. eval — 评测编排

| 文件 | 职责 | 用法 |
|---|---|---|
| `run_baseline_vs_w2a8.sh` | baseline(FP8) vs W2A8 精度对比编排（两实验串行：serve→smoke→评测→清理；官方 vq2a8_accuracy 形态） | `bash run_baseline_vs_w2a8.sh [--smoke-only] [--skip-b1|--skip-b2]`；默认输出 `output/baseline_vs_w2a8_<ts>/` |
| `serve_b1_b2.sh` | 单独拉起 B1/B2 评测形态的 vllm serve（参数与编排脚本逐字一致；支持指定卡，B1 自动 DP=卡数 / B2 自动 EP=卡数） | `bash serve_b1_b2.sh {b1\|b2} [--port N] [--log-dir DIR] [--devices 0,1,2,3]`。B1=FP8·DP·FULL_DECODE_ONLY(:8004)；B2=W2A8·tp-ep·gs32 full_graph(:8003)。前台运行 + tee 日志到 `output/serve_standalone/` |

## 通用约定

- **插桩类全部默认关闭**：环境变量未设置时 hook 不初始化（仅一次 env 查询开销），采集完 `--uninstall` 还原源码。
- **采集默认走 NPU tensor 不落盘**：expert_counter 的计数在设备上累加，flush/退出时才 D2H。
- **输出目录**默认落在 `output/` 或指定 `--out`；PD 采集按 `prefill/`、`decode/` 分侧。
- 进度跟踪与实验断点续跑见 `docs/eplb/experiment_progress.md`。
