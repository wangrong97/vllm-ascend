# serve/ — PD 分离与单机服务启动脚本

适用分支：`test/vq2a8-perf-20261024`（full_ffn 单路径、ABI v4、B≤64 decode 图）。

## run_pd_w2a8_eager_graph.sh — PD 分离（eager P + graph D，双 W2A8）

### 形态

| 侧 | 卡 | 权重 | 执行 | 并行 | 端口 |
|---|---|---|---|---|---|
| P（prefill） | 0-3 | W2A8 | `--enforce-eager`（full FFN 无 decode 图） | DP4×TP1×EP4 | :9001 |
| D（decode） | 4-7 | W2A8 | `full_graph`：decode 图桶 `[1..64]` + eager 分块 prefill | DP4×TP1×EP4 | :9002 |
| 连接 | — | — | MooncakeHybrid producer/consumer、双超时 3600s、block32、分侧 HCCL 端口段 | | |

D 侧 `--max-num-seqs 64` 并发打满（B64 为该分支图模式上限 `GS32_GRAPH_MAX_BATCH_SIZE`）；
DP4×B64 理论并发容量 256，压测客户端并发 ≥64 即可打满图回放。

### 首次运行前置

1. **构建 ABI v4 GS32 产物**（不能复用 v3；脚本缺产物时会打印同样命令并退出）：
   ```bash
   export BUILD="$PWD/build/vq2a8-gs32-runtime-ffn"
   # 需已加载 CANN 环境(ASCEND_TOOLKIT_HOME)
   (cd csrc/mixed_w2w4_moe_ffn && bash run_mixed_w2w4_moe_ffn_esl.sh)
   python3 tools/build_vq2a8_gs32_stage.py \
     --kernel-binary csrc/mixed_w2w4_moe_ffn/mixed_w2w4_moe_ffn.o \
     --cann "$ASCEND_TOOLKIT_HOME" --build-dir "$BUILD" --jobs 4
   ```
2. 权重：两侧默认 `weight/DeepSeek-V4-Flash-0731-vq2a8-mixed-gs32-repack`（可用
   `MODEL_P` / `MODEL_D` 覆盖）。该分支只接受自然输入列顺序，旧 `.perm` 张量会被拒绝。

### 运行

```bash
cd /mnt/share/w00608002/work/eplb/vllm-ascend
bash tools/eplb/serve/run_pd_w2a8_eager_graph.sh                 # 基础 PD
bash tools/eplb/serve/run_pd_w2a8_eager_graph.sh --with-proxy    # 就绪后自动起 proxy :8000
EPLB_MODE=dynamic bash tools/eplb/serve/run_pd_w2a8_eager_graph.sh --with-proxy
   # D 侧注入混精 eplb(off|static|dynamic|static_uniform|dynamic_uniform, 默认 off)
```

就绪后 smoke（经直连 D 或 proxy）：
```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"dsv","messages":[{"role":"user","content":"计算 23 * 47 等于多少"}],"max_tokens":2048,"temperature":0}'
# 预期含 1081 且正常 stop
```

### 压测（并发打满口径）

```bash
cd /mnt/share/w00608002/work/benchmark
ais_bench --models vllm_api_stream_chat --datasets gsm8k_gen_0_shot_cot_str_perf \
  --debug --summarizer default_perf --mode perf --num-prompts 300
# vllm_api_stream_chat.py 配置: host 127.0.0.1:8000 · model dsv · batch_size 8-64
```

### 停止

`Ctrl+C`（内置 trap 清理双侧 vllm + NPU 进程表兜底）；异常残留按 npu-smi 进程表
容器内 PID 逐个 `kill -9`（勿用 cmdline 模式匹配，worker 是 spawn 形式）。

### 已知边界

- kv-transfer 已在 `vq2a8_config.py` 放行（PD 分离合法）；PD 仅支持 Hybrid 连接器 + block32。
- B>1 图模式仅此分支支持；旧分支（eplb_export_mix_precision）full_graph 锁 B1。
- 权重被 perm 校验拒绝时，需按 `docs/vq2a8/repack_non_routed.md` 转换。
