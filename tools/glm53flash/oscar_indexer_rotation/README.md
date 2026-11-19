# oscar_indexer_rotation — OSCAR 协方差旋转(GLM-5.3-Flash indexer 适配)

- 日期: 2026-11-12
- 来源: gitcode.com/Ascend-Quant/vllm-ascend PR #24(Qwen235B-attn-v16-oscar)
  `vllm_ascend/attention/oscar.py` 的拟合算法, 逐式对齐移植
- 状态: 拟合工具 + 对比工具 + **runtime 集成**全部就绪(单测通过, 未做实机评测)

## 目录

| 文件 | 用途 |
| --- | --- |
| `oscar_indexer_rotation.py` | 拟合 CLI + checkpoint + runtime 辅助类 |
| `compare_rotations.py` | 同批数据对比 无旋转/Hadamard/OSCAR 的量化误差 |

## 算法(与原版逐式对齐)

```
K 协方差:  C = E[q qᵀ](全 query 头展平, fp64 累加)   ← 原版为 GQA per-kv-head
           C ← (C + Cᵀ)/2(对称化)                       分组均值; GLM indexer k
特征分解:  eigh(C) → (λ, V)                              无头维, 用全头协方差
组合(默认 r_h_pbr):  R = V @ Hadamard(128) @ pbr(λ 降序位反转)
目标:      旋转后 q/k 逐维能量均匀化(能量离散度 std/mean → 0),
           使 MXFP4/MXFP8 的 per-32 组 E8M0 scale 覆盖最优
四种组合:  plain(纯特征基) / h(纯 Hadamard) / r_h / r_h_pbr(默认, 原版 OSCAR_K_OBJECTIVE)
```

## 与原版(PR #24)的适配差异

| 项 | 原版(Qwen235B 主注意力) | 本版(GLM kpool indexer) |
| --- | --- | --- |
| 作用对象 | Q/K/V 三路(head_size 128) | q(打分)/ k(池化)两路, head_dim=128 |
| K 协方差 | GQA per-kv-head 分组均值 | 全 q 头展平(k 无头维, 语义等价) |
| V 旋转 | 有(restore_output 回原域) | 无(indexer 无 V) |
| 数据采集 | 内置 OscarCalibrationCollector(改服务代码) | 复用 dump_indexer_activations(raw npz, 零侵入) |
| 应用点 | attention_v1 的 q/k/v | q 打分前 / k 写入前 / 尾池生产点(与 HADAMARD_PRE 同点) |
| 开关 | QuantConfig 字段 | 环境变量(见下) |

## 完整使用链(三步)

```bash
# ① 抓激活(raw 模式; LAYERS 覆盖所有 DSA 层, 真实评测流量最佳)
VLLM_ASCEND_INDEXER_DUMP=1 VLLM_ASCEND_INDEXER_DUMP_MODE=raw \
VLLM_ASCEND_INDEXER_DUMP_DIR=<dump_dir> VLLM_ASCEND_INDEXER_DUMP_LAYERS=3,7 \
VLLM_ASCEND_INDEXER_DUMP_MAX_STEP=8 ... bash serve_glm53flash.sh
#   发流量: bash ../dump_indexer_activations/run_dump_test.sh(4 题对照集)或评测

# ② 拟合 checkpoint(CLI 输出逐层 samples/正交误差/能量离散度前后)
python3 oscar_indexer_rotation.py <dump_dir> oscar_indexer_ckpt.pt \
    [--composition r_h_pbr] [--layers 3,7]

# ③ runtime 起服务(已集成进 vllm_ascend/attention/indexer_kpool.py)
VLLM_ASCEND_KPOOL_OSCAR_ROTATION=1 \
VLLM_ASCEND_KPOOL_OSCAR_CKPT=<ckpt 绝对路径> \
MODEL_PATH=... bash serve_glm53flash.sh
```

## runtime 集成细节

- **统一入口** `_indexer_rotate(x, layer)`:`OSCAR > Hadamard`(两者同开时 OSCAR 优先)
- 四个注入点全部接入: 真 C8 的 q 打分 / 池化 k 写入 / 尾池 k(旋转域存取, RᵀR=I 保点积);
  伪量化 MXFP4/MXFP8(R → 量化往返 → Rᵀ 回原域, OSCAR 非对称须转置)
- 层号从 `k_cache.prefix` 自动提取(`layers.{N}.`); 缺失层: 真量化 KeyError / 伪量化退化无旋转
- checkpoint 每 device 加载一次(`_OSCAR_MATRICES` 缓存), bf16

## 验证记录(2026-11-12, 合成数据)

- Hadamard 正交 1.1e-16 / 位反转置换正交 ✓
- 强各向异性 q(3 主方向能量占优): 能量离散度 **3.885 → 0.000**, 拟合正交误差 1.1e-15 ✓
- runtime 单测: 保范 0.9998 / 缺层报错 / OSCAR+MXFP4 往返 SNR 19.33 dB ✓

## 对比工具

同批 q 激活并列三方案, 输出能量离散度与 MXFP4/MXFP8 伪量化相对误差:

```bash
python3 compare_rotations.py <dump_dir>
```

## 待办

- [ ] 用真实 dump 数据(三h C8+Hadamard 服务采集)跑 compare_rotations, 出 Hadamard vs OSCAR 结论
- [ ] OSCAR 实机 A/B(aime/gpqa 各 1 遍, vs 三h/七基线)
