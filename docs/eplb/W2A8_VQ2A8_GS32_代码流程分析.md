# vllm-ascend `feat/vq2a8_v0.1`:W2A8(VQ2A8-GS32)权重与 MoE 专家级混精代码流程分析

> 分析对象:`/mnt/share/w00608002/work/eplb/vllm-ascend-vq2a8`(分支 `feat/vq2a8_v0.1`,基于官方 vLLM Ascend `v0.27.1rc1`,相对 `main` 共 32 个提交、约 4 万行新增)
> 目标模型:DeepSeek-V4(Ascend950),Model Runner V1 / V2
> 相关现成文档:`docs/vq2a8/`(总览、运行时、checkpoint 交付规范、CI/性能/精度指南)

---

## 目录

1. [总体架构](#1-总体架构)
2. [量化格式与数据布局(source-v3 checkpoint)](#2-量化格式与数据布局source-v3-checkpoint)
3. [代码模块总览](#3-代码模块总览)
4. [权重加载流程](#4-权重加载流程)
5. [推理执行流程](#5-推理执行流程)
6. [Model Runner 集成(v1/v2、图执行、有效性检查)](#6-model-runner-集成v1v2图执行有效性检查)
7. [启动配置、执行模式与并行约束](#7-启动配置执行模式与并行约束)
8. [原生算子层(csrc)与构建](#8-原生算子层csrc与构建)
9. [测试与工具布局](#9-测试与工具布局)
10. [关键文件索引](#10-关键文件索引)

---

## 1. 总体架构

本分支在 vLLM Ascend 上实现了 **VQ2A8(W2A8)与 W4A8(MXFP4)的 MoE 专家级混合精度推理**:

- **W2(VQ2)**:权重 2-bit 矢量量化。每个 `32×32`(N32×K32)tile 拥有 4 个 FP8(E4M3)量化 level,每个权重元素 2-bit code 四选一;配套每行(输出维)FP32 scale 与每专家输入列置换 `perm`;激活为 **MXFP8**(K32 分组 E8M0 scale + E4M3)。
- **W4(MXFP4)**:权重 FP4(E2M1)+ E8M0 scale(K32);激活同为 MXFP8,走 CANN `npu_grouped_matmul_gmm2`。
- **混精粒度**:**每层、每专家、每投影(`w13` GateUp / `w2` Down)独立**选择 VQ2 或 MXFP4(`quant_types[kind][expert_id]`)。因此同一层中可以同时存在 VQ2 专家与 MXFP4 专家,GateUp 与 Down 也可以异构(对应独立算子的 2+2 / 2+4 / 4+4 路径)。
- Attention、共享专家、router 等非路由专家组件保留 checkpoint 原始数据类型(FP8 block128 或 BF16),走常规模型实现。

数据流主线:

```
input_ids
  └─ DeepseekV4 MoE 层
       ├─ router(FP32 logits;hash 层走 tid2eid 表路由)
       ├─ token dispatch(EP 下 ALLGATHER;激活保持 BF16)
       ├─ AscendRoutedExperts(排序/分组 → group_list)
       │    └─ unified_apply_mlp(moe_mlp.py:842,QuantType.VQ2A8 分支)
       │         └─ GS32ExpertWeights.apply(hidden, group_list, type, swiglu_limit)
       │              ├─ Triton 激活准备(perm 收集 + MXFP8 量化 [+ 融合 SwiGLU])
       │              ├─ Triton 设备路由(decode/prefill 重排 + 有效性 flags)
       │              ├─ VQ2 bank → torch.ops.vq2a8_gs32_stage(Torch bridge
       │              │    → mixed_w2w4_moe_ffn 原生 AIC/AIV kernel)
       │              └─ MXFP4 bank → npu_grouped_matmul_gmm2(CANN)
       └─ unpermute + 加权 combine(routed scaling)
```

## 2. 量化格式与数据布局(source-v3 checkpoint)

格式契约定义在 `vllm_ascend/quantization/methods/w2a8/vq2a8_gs32_format.py`。checkpoint 标识:

```json
"quantization_config": {
  "quant_method": "vq2a8",
  "format": "vq2_mixed_gs32_source",
  "format_version": 3
}
```

仅接受 **source-v3**;旧版 source-v1 需用 `tools/convert_vq2_mixed_gs32_v3.py` 转换。GateUp 行序固定 `natural_gate_then_up`(全部 Gate 行在前、Up 行在后)。

### 2.1 专家张量命名与字段

命名:`layers.{L}.ffn.experts.{E}.{w13|w2}.{字段}`(L 全局层号、E 全局专家号,直接写入张量名)。每个投影的字段集合必须完整属于一种类型:

| 量化类型 | checkpoint 字段 | dtype | 形状(N=输出行,K=输入列) |
| --- | --- | --- | --- |
| VQ2 | `packed_indices` | U8 | `[N/32, K/16, 16, 8]` |
| VQ2 | `codebooks` | F8_E4M3 | `[N/32, K/32, 16, 2]` |
| VQ2 | `perm` | I32 | `[K]`(恒等也要显式存储) |
| VQ2 | `out_dim_scale` | U64 | `[N]`(低 32 位为 FP32 位模式,高 32 位为 0,scale 有限且 >0) |
| MXFP4 | `weight` | U8 | `[N, K/2]`(每字节两个 E2M1,低半字节在前) |
| MXFP4 | `scale` | F8_E8M0 | `[N, K/32]`(值为 `2^(b-127)`,禁止 0xFF/NaN) |

### 2.2 VQ2(W2)编码细节

参照 `csrc/mixed_w2w4_moe_ffn/scripts/w2_layout.py` 的布局函数:

- **2-bit code**:每个权重元素的码值 ∈ [0,3],在所属 `(N/32, K/32)` tile 的 4 个 FP8 level 中四选一。
- **packed zN 布局** `[N/32, K/16, 16, 8]`:逻辑 `[N,K]` 转置为 K-major 后按 `(32 行, 16 列)` 分块;每块 512 个 2-bit code = 128 字节,排成 `16 列 × 8 字节`,每字节打包同列连续 4 行的 4 个 code(小端 2-bit)。
- **pair LUT(codebooks)** `[N/32, K/32, 16, 2]`:4 个 level 展开为 16 项查找表,`pair_lut[p] = (levels[p & 3], levels[p >> 2])`。kernel 将字节中相邻两行的 2-bit 码拼成 4-bit 索引 `p`,一次查表取回一对重建值 —— 等价于逐元素 2-bit 解码的查表优化。加载时按字节视图重排为 native `[N/32, K/32, 32]`,禁止数值转换;codebook 中不允许出现 NaN 字节模式。
- **perm `[K]`**:每专家的输入列置换(为量化分组/tiling 优化而设)。加载时校验必须是 `[0,K)` 的完整双射,并转为 I64。
- **out_dim_scale `[N]`**:每输出行一个 scale,以 U64 存 FP32 位模式;加载校验范围/有限性/正值,保存为 I64 位模式(`row_scale_bits`)。

checkpoint → native 字段映射(`SOURCE_TO_NATIVE`):`packed_indices→packed_zn`、`codebooks→pair_lut`、`perm→perm`、`out_dim_scale→row_scale_bits`;MXFP4 的 `weight/scale→mxfp4_weight/mxfp4_scale`。转换仅做字节视图/重排/类型位宽调整(`convert_expert_tensor`),不做任何数值反量化。

### 2.3 激活:MXFP8(K32)

数值契约在 `vq2a8_gs32_staged.py::quantize_mxfp8_k32`(CPU 参考实现)与 Triton kernel(`ops/triton/activation/vq2a8_gs32.py::_quantize_groups`)中保持一致:

`BF16 RNE → 按 32 元素组求 amax → E8M0 编码(指数 -8,尾数 >0x600000 进位;全零组 code=127)→ BF16 缩放舍入 → 饱和 ±448 → E4M3 字节`

SwiGLU:FP32 clamp/SiLU/乘积后一次 BF16 舍入(`staged_swiglu`);`swiglu_limit=0` 表示不裁剪。可选 `gs32_fusion=swiglu_quant` 将 SwiGLU、Down 置换、BF16 舍入与 MXFP8 量化融合进同一个 Triton kernel。

### 2.4 根张量(非路由专家部分)

`normalize_vq2a8_root_name` 将 source-v3 命名映射到模型参数名(HC 头、compressor/indexer、attention 线性层、shared experts、`lm_head` 等)。允许的 dtype:BF16/F32/I64/F8_E4M3(+ 配对的 `weight_scale` F8_E8M0,block128)。FP8 权重对(`F8_E4M3 weight` + `F8_E8M0 _scale`,scale 形状为权重各维 `ceil(/128)`)在 `get_quant_method` 中路由到既有 DeepSeek FP8 block128 Linear scheme;其余走 `AscendUnquantizedLinearMethod`。

## 3. 代码模块总览

| 层次 | 模块 | 职责 |
| --- | --- | --- |
| 量化注册 | `vllm_ascend/quantization/configs/vq2a8_config.py` | `AscendVQ2A8Config`:source-v3 审计(`bind_checkpoint`)、运行时选项校验、`get_quant_method` 分发 |
| 量化方案 | `vllm_ascend/quantization/methods/w2a8/` | `vq2a8_gs32.py`(`AscendGS32FusedMoEScheme`:参数创建/加载/apply)、`vq2a8_gs32_format.py`(格式 ABI 与转换)、`vq2a8_gs32_ep.py`(线性整专家 EP 放置)、`vq2a8_gs32_staged.py`(数值参考契约) |
| 运行时 | `vllm_ascend/ops/fused_moe/vq2a8_gs32.py` | `GS32ExpertWeights`:常驻 bank 管理、四种执行模式、decode/prefill 设备路由执行;`_NativeStageOps`:原生算子封装 |
| 激活 | `ops/fused_moe/vq2a8_gs32_activation.py` + `ops/triton/activation/vq2a8_gs32.py` | MXFP8 激活准备(perm 收集/融合 SwiGLU)、有限性 flags、Triton kernel |
| 路由 | `ops/fused_moe/vq2a8_gs32_routes.py` + `ops/triton/activation/vq2a8_gs32_routes.py` | 设备侧行重排(counts/slots/gather/inverse)、恢复(restory)与延迟有效性 |
| 原生算子 | `csrc/vq2a8_gs32_stage/torch_binding.cpp` | Torch bridge:注册 `torch.ops.vq2a8_gs32_stage`(投影计划/路由 custom class,图捕获安全) |
| 原生算子 | `csrc/mixed_w2w4_moe_ffn/` | 独立混合 W2/W4 MoE FFN 融合算子(AIC/AIV .cce + host tiling),来源 `feat/test@592a6973` |
| 框架接入 | `ops/fused_moe/`(moe_mlp、moe_comm_method、routed_experts、dataclass/fused_experts、router/*) | VQ2A8 分发点、EP ALLGATHER 通信、BF16 分发、路由 FP32 化 |
| 模型 | `models/deepseek_v4/model.py` + `native_ops.py` | checkpoint 绑定、source-v3 load_weights、FP32 router、DSV4 原生绑定检查 |
| Runner | `worker/model_runner_v1.py`、`worker/v2/model_runner.py`、`worker/worker.py`、`worker/v2/aclgraph_utils.py` | 量化有效性机制、full_graph 图捕获、启动 prefill 预热 |
| 工具/测试 | `tools/*vq2a8*`、`tests/ut/quantization/`、`tests/e2e/nightly/test_vq2a8_*` | 构建/转换/服务/基准/CPU 契约/NPU 回归 |

## 4. 权重加载流程

### 4.1 配置注册与启动校验

1. `platform.py` 将 `vq2a8` 加入支持的量化方法(`supported_quantization` 与 CLI choices),并注册 `AscendVQ2A8Config`。
2. `ascend_config.py` 将 `additional_config.vq2a8` 键排除出 Ascend 平台 schema(避免 typo 报错),保留在 `VllmConfig` 上。
3. `validate_vq2a8_config()`(vq2a8_config.py:67)在**专家参数分配之前**拒绝所有不支持的组合(详见第 7 节),并返回 `gs32_library/gs32_binary/gs32_execution/gs32_fusion` 选项。
4. router 侧改动(`AscendFusedTopKRouter`/`AscendGroupedTopKRouter`)详见 5.1:hash 门控 FP32 化、归一化与 routed scaling 解耦、`sqrtsoftplus` 的 Python 参考路径与 `tid2eid` 表路由。

### 4.2 模型构建与 checkpoint 绑定

`DeepseekV4ForCausalLM.__init__`(model.py:1021):

```python
if quant_config.get_name() == "vq2a8":
    quant_config.bind_checkpoint(vllm_config.model_config)   # 先审计 checkpoint,再建参数
    # DefaultModelLoader 的 glob 深度适配(source-v3 分片可在 vq2_mixed_gs32/ 子目录)
    self.allow_patterns_overrides = ["*/" * shard_depth + "*.safetensors"]
```

MoE 构造分支(model.py:303/372):

- `gate`(router)固定 `params_dtype=torch.float32` + `precast_fp32_weight`(保留 FP32 checkpoint router)。
- `apply_routed_scale_to_output=True`:top-k 归一化后再对 routed 输出乘 `routed_scaling_factor`,与 V4 顺序一致。
- hash 层(`layer_idx < num_hash_layers`)持有 `gate.tid2eid [vocab, top_k]` 哈希表;source-v3 允许 hash 层专家库存稀疏(不必包含全部逻辑专家),非 hash 层必须存全部专家。
- GS32 或 A5 上调用 `require_deepseek_v4_native_ops()`(native_ops.py):在每个执行 worker 内检查 `vllm_ascend_C` 扩展含 DSV4 绑定(`npu_hc_pre_v2`、`moe_gating_top_k_hash`、compressor/KV 量化稀疏注意力等 13 个算子)——单独构建 GS32 桥接库不足以运行整模型。

`bind_checkpoint()`(vq2a8_config.py:187)审计内容:

- 本地目录 + safetensors 索引;`config.json`/`quant.json`/`vq2_mixed_gs32/format.json` 中的格式标识;
- `read_gs32_header()` 逐分片解析 safetensors 头(不载载荷):dtype/shape/extent 一致性、payload 无重叠/截断、`__metadata__` 中格式字段;
- 索引 `weight_map` 与每个分片头张量清单一一对应;
- 逐张量名解析:`parse_vq2a8_expert_name` 提取 `(layer, expert, w13|w2, field)`,按 `(layer, expert, projection)` 汇总字段集合,判定 `vq2`/`mxfp4` 并校验每个字段的 dtype/shape 精确匹配 `source_projection_fields`;
- 层/专家几何(层数、`n_routed_experts ≤ 256`、hash 层数)与投影几何(`gate_up = [2H, K]`、`down = [K, H]`)校验,构建 `VQ2A8LayerSpec(layer, expert_ids, specs, quant_types)` 存入 `self.layers`;
- 根张量 dtype/配对校验(`_validate_roots`)。

### 4.3 参数创建(`get_weight`)

`AscendVQ2A8Config.get_quant_method`(vq2a8_config.py:379):

- `LinearBase` → 按 checkpoint 根张量 dtype 分发:FP8 block128 → DeepSeek FP8 Linear scheme;否则 → `AscendUnquantizedLinearMethod`。
- FusedMoE 层(`layers.{L}.mlp.experts` 前缀)→ `AscendFusedMoEMethod(AscendGS32FusedMoEScheme(layer_spec, native_options, moe_config))`,scheme 注册名 `("VQ2A8_GS32", "moe")`,`quant_type = QuantType.VQ2A8`(quant_type.py 新增枚举值 9)。

`AscendGS32FusedMoEScheme.get_weight`(methods/w2a8/vq2a8_gs32.py:39)按 **(投影 w13/w2) × (量化类型 vq2/mxfp4)** 分 bank 分配参数:

- `gs32_expert_placement()`(vq2a8_gs32_ep.py)按线性整专家 EP 计算本 rank 持有的全局专家区间 `[ep_rank*local, (ep_rank+1)*local)`,从 checkpoint 库存中筛出本地专家;
- 对每个 bank 中的每种字段创建常驻张量 `w13_packed_zn [E_bank, N/32, K/16, 16, 8]`、`w13_pair_lut [E_bank, N/32, K/32, 32]`(U8)、`w13_row_scale_bits [E_bank, N]`、`w13_perm [E_bank, K]`(I64)以及 `*_mxfp4_weight/scale`;同时记录 `parameter_experts`(专家→slot 映射)与 `loaded_experts`(加载进度);
- `method_adapters.py` 的 `AscendFusedMoEMethod.create_weights` 合并 scheme 的 `get_weight_attrs`,把 `weight_loader` 与 `vq2a8_parameter_name` 绑到参数属性上;
- 空库存 rank(EP 下 hash 层可能无本地专家)也分配 `vq2a8_anchor` 设备锚点供路由与有效性使用。

### 4.4 权重加载(`load_weights`)

`DeepseekV4ForCausalLM.load_weights` 在 vq2a8 下走 `_load_vq2a8_weights`(model.py:1174):

```
for (source_name, value) in weights:
    expert = parse_vq2a8_expert_name(source_name)
    if expert:                                    # 专家张量
        scheme.load_expert_tensor(layer, expert_id, projection, field, value)
            └─ weight_loader(param, value, expert_id=...)     # 本地专家 → convert_expert_tensor 字节级
                                                              # 转换后 copy_ 进 bank slot;远端专家跳过
    else:                                         # 根张量
        spec = root_tensor_spec(source_name); name = normalize_root_name(source_name)
        _prepare_vq2a8_root(param, value, name, spec)         # dtype/shape 校验;FP8 scale 数值化;
                                                              # tid2eid 对库存校验;norm F32→param 提升
        param.weight_loader(...)                              # TP 切分/attn_sink 头切分/gate-up 合并
```

结束后 `validate_loaded()` 断言每个 bank 的每个本地专家字段齐备且无重复;最终缺失清单(参数、半加载 gate/up、未消费的 checkpoint 张量)为空,否则报错。

关键点:**专家张量不走 vLLM 通用 weight_loader 堆叠路径**,而是由模型侧按独立命名直接分发到 scheme,再由 scheme 写入 typed bank slot;hash 层 `tid2eid` 值会对照该层 `layer_spec.expert_ids` 库存校验。

### 4.5 加载后处理(`process_weights_after_loading`)

`AscendGS32FusedMoEScheme.process_weights_after_loading`(methods/w2a8/vq2a8_gs32.py:113):

1. `validate_loaded`;MXFP4 bank 调 `prepare_mxfp4_projection`(w4a8_mxfp4.py 新抽取的公共函数):`npu_format_cast(29, float4_e2m1fn_x2)` + 转置为 K-major、E8M0 scale 重排为 N-major 微缩放块视图(纯字节视图,无数值转换);
2. 向 layer 写入运行时元数据:`vq2a8_expert_ids / vq2a8_layer_index / vq2a8_ep_size / vq2a8_ep_rank / vq2a8_local_global_expert_ids / vq2a8_local_expert_ids / vq2a8_expert_start / vq2a8_num_experts`、`vq2a8_anchor` buffer;
3. 以**本地化 layer_spec**(local_ids + local quant_types)构造 `GS32ExpertWeights(layer, local_spec, native_options)` 挂到 `layer.vq2a8_weights` —— 至此进入运行时(见第 5 节)。

`GS32ExpertWeights.__init__` 的工作:

- 建立 `banks[kind][quant_type] = {fields(常驻张量), expert_ids, slots}`;
- 实例化 `_NativeStageOps(gs32_library, gs32_binary, device, execution)`:`torch.ops.load_library` 加载 `libvq2a8_gs32_stage.so`,校验当前执行模式所需算子齐全(`prepare/project/project_prepared/prepare_routes/prepare_grouped/project_grouped/prepare_device_routes/device_routes_status/project_grouped_device/supports_*`),并以 `gs32_binary`(`mixed_w2w4_moe_ffn.o` ELF)初始化原生运行时;
- 为 VQ2 bank预建 tiling(`make_tiling(K,N)`);
- `full_graph` 模式额外:校验桥接库支持 padding 的 `prepare_device_routes`、专家数 ≤ 256、构建 `resident_mask`、每 bank 的 `expert_ids_tensor`/`logical_slots`(逻辑专家→物理 slot 的 I64 表,`-1` 表示不在本 bank)、`prepare_grouped_plan` 分组投影计划。

## 5. 推理执行流程

### 5.1 路由与分发

1. **Router**:`gate`(FP32,`params_dtype=float32` + `precast_fp32_weight`)产出 logits → `AscendFusedTopKRouter`;DeepSeek-V4 无分组路由(`num_expert_groups=1`,model.py `set_moe_parameters` 缺省补 1)。hash 层(`layer_idx < num_hash_layers`,forward 必须带 `input_ids`,decoder 层一路传入 `self.mlp(hidden_states, input_ids)`)用 `moe_gating_top_k_hash`(FP32 logits,`routed_scaling_factor=1.0`,算子内 renorm=0,Python 侧 `_renormalize_topk_weights` 后再乘 scaling)或 `sqrtsoftplus+tid2eid` 表路由(grouped_topk_router.py 的 Python 参考路径:直接查表 `topk_ids = table[token_ids]`;无表分支 argmax 迭代选 top-k,平局取最低专家 ID 与 V4 对齐)。产出 `topk_weights` 与 `topk_ids`。
2. **topk_weights 保持 FP32**(routed_experts.py:480 跳过 cast):因为最终 `npu_moe_token_unpermute` 的 `probs` 参数要求 FP32(token_dispatcher.py)。profiling 强制负载均衡时随机改写 `topk_ids` 为本 rank 实际驻留的专家(`self.vq2a8_expert_ids`,routed_experts.py:485),避免路由到外地专家。
3. **通信**:`ascend_forward_context.gs32_ep_uses_allgather()` —— VQ2A8+GS32+EP 且 TP>1 时强制 `MoECommType.ALLGATHER`,`setup_moe_comm_method` 只初始化 ALLGATHER 实现,不建 MC2/AlltoAll 通信域与 fused buffer(moe_comm_method.py:54)。原因:MC2/AlltoAll 的 token 划分与原生权重 ABI 不兼容;ALLGATHER 保留 TP 复制输入、过滤本地专家路由,由公共 MoE 流程恰好一次求和部分专家输出。
4. **激活保持 BF16 通过分发**:`MoEQuantParams.dispatch_with_quant` 对 VQ2A8 为 False(moe_quant.py:73)—— VQ2 的置换/scale 按专家不同,量化必须发生在 dispatch/重排之后。

**dispatch/combine 细节**(EP ALLGATHER 路径):

- `TokenDispatcherWithAllGather.token_dispatch`:`npu_moe_init_routing` 完成排序,产出**按本地专家连续排好的 BF16 hidden**(`[Σ M_e, K]`)与 `group_list`(**本地专家计数,count 模式 type=1**);EP 下非本地专家先由 dispatcher 的 `expert_map` mask + `active_expert_range` 过滤为 0 计数。
- combine:`npu_moe_token_unpermute(probs=topk_weights[FP32])` 加权还原。
- **routed scaling 时机**:`apply_routed_scale_to_output=True` 时 router 以 `routed_scaling_factor=1.0` 返回权重,先完成 unpermute(保持 BF16 专家舍入精度),再在模型侧对 routed 求和结果统一乘 `routed_scaling_factor`(model.py `final_hidden_states *= routed_scaling_factor`)——注释 "VQ preserves BF16 expert rounding before scaling the routed sum"。

### 5.2 进入专家计算

`AscendRoutedExperts.quant_type` 从 `quant_method.quant_method`(= `AscendGS32FusedMoEScheme`)取到 `QuantType.VQ2A8` 后,`AscendGS32FusedMoEScheme.apply`(methods/w2a8/vq2a8_gs32.py:181)把 `layer.vq2a8_weights` 作为 `MoEWeights.vq2a8` 字段(dataclass/fused_experts.py 新增)传入公共管线:

```python
_EXTRA_CTX.moe_comm_method.fused_experts(
    build_fused_experts_input(hidden_states=x, topk_weights, topk_ids,
                              w1=projection_anchor(layer,"w13"), w2=...,
                              vq2a8=layer.vq2a8_weights, quant_type=QuantType.VQ2A8, ...))
```

`w1/w2` 仅作设备锚点(`w13_packed_zn` 或 `w13_mxfp4_weight`,空库存时 `vq2a8_anchor`)。公共 MoE 管线完成 token 排序/分组后,`unified_apply_mlp`(moe_mlp.py:842)按下式分发:

```python
if quant_type == QuantType.VQ2A8:
    return vq2a8.apply(hidden_states, group_list, group_list_type, swiglu_limit)
```

该分支位于 `is_quant`、unquant 与传统 GMM(MXFP/FP8/INT8)路径之前 —— **VQ2A8 完全绕过传统 w1/w2 GMM 权重管线**,只消费 dispatcher 产出的 BF16 排序行 + `group_list`(类型 0=前缀结束位置 / 1=计数)。VQ2A8 拒绝 EPLB/LoRA/`dynamic_scale`/非 silu 激活。

### 5.3 `GS32ExpertWeights.apply` 总入口(vq2a8_gs32.py:639)

```
apply(hidden, group_list, group_list_type, swiglu_limit)
 ├─ full_graph:捕获中?→ 必须有 forward owner(defer_validity);行数 ≤ 8
 ├─ 输入契约:hidden 为本设备 BF16 [tokens, hidden];group_list 覆盖每逻辑专家
 ├─ full_graph 且 0 < rows ≤ 8 → _apply_device()(decode:图内设备路由路径)
 ├─ full_graph(其余行数)      → _apply_prefill()(eager 分块 prefill 路径)
 ├─ rows==0                    → 置零输出并标记有效
 ├─ fused/full_graph 且本地专家 ≤ 256 → _apply_fused()(eager 分组投影)
 ├─ batched/fused/full_graph   → _project_batched()(128 行/块的批处理)
 └─ serial                     → 逐专家逐 128 行 _project()(严格参考路径)
```

四种执行模式:

| 模式 | 路径 | 说明 |
| --- | --- | --- |
| `serial` | `_project` | 每专家每 128 行一次原生启动,激活用 Torch 参考实现 `quantize_mxfp8_k32`;正确性基线 |
| `batched` | `_project_batched` | 拼接多专家行共享一次激活准备与逐块启动;每 bank 只准备一次(swiglu_quant 时) |
| `fused` | `_apply_fused` | 按实际路由计数用 `prepare_routes` + `project_grouped` 一次分组投影;无 Python 侧 M/R 上限(受显存与原生 256 专家上限约束) |
| `full_graph` | `_apply_device` / `_apply_prefill` | B1 decode 图回放(≤ 8 路由行);prefill 恒 eager 设备路由 |

`swiglu_limit=None` 时 GateUp 与 Down 之间用独立 `staged_swiglu`;`gs32_fusion=swiglu_quant` 时把 SwiGLU+Down 置换+BF16 舍入+MXFP8 融合进 Down 的激活准备 kernel(Triton,FUSE_SWIGLU 分支)。

### 5.4 混精 bank 的执行细节

#### 5.4.1 bank:按 (投影 × 量化类型) 分组的常驻权重容器

`GS32ExpertWeights` 把每层的专家权重组织成二级字典(vq2a8_gs32.py:196-217):

```python
self.banks[kind][quant_type] = {
    "fields":      {...该类型的堆叠参数张量...},   # 第 0 维 = 该 bank 内的专家数 E_bank
    "expert_ids":  (…属于该 bank 的全局专家号,升序…),
    "slots":       {expert: slot},                # 专家号 → 张量第 0 维的紧凑下标
}
```

- `kind ∈ {gate_up(w13), down(w2)}`,`quant_type ∈ {vq2, mxfp4}` —— 一层**最多 4 个 bank**,某组合无专家则不创建;GateUp 与 Down 的类型划分可以不同(独立算子的 2+2/2+4/4+4 组合在框架侧的体现);
- 分 bank 的根本原因:两种类型的字段数/形状/dtype 完全不同(VQ2:`packed_zn/pair_lut/row_scale_bits/perm`,每专家 ≈0.28·N·K 字节;MXFP4:`mxfp4_weight/mxfp4_scale`,≈0.53·N·K 字节),无法共存于一张堆叠张量;
- `full_graph` 模式下每个 bank 额外持有设备索引:`expert_ids_tensor`(int64)、`logical_slots`([num_experts] 全局专家→bank 内 slot,-1 表示不在本 bank)以及 VQ2 bank 的分组投影计划 `plan`(加载时 `prepare_grouped_plan` 预建);
- matmul 分工:**VQ2 bank 走新路径**(`torch.ops.vq2a8_gs32_stage` → mixed_w2w4_moe_ffn 的 DownW2 投影阶段),**MXFP4 bank 的 matmul 走既有 CANN 算子**(`npu_grouped_matmul_gmm2`);但激活量化、路由、bank 循环、求和、validity 全部由 GS32 运行时统一编排。

#### 5.4.2 互斥分派与零填充求和(混精正确性的核心)

**每个逻辑专家在一种投影内只属于一个 bank**,因此每个 token 行的每次投影只由一条 kernel 路径计算,两条路径从不"各算一半再叠加":

```python
# vq2a8_gs32.py:315 — decode 路径的恢复
@staticmethod
def _restore(output, inverse):
    selected = inverse >= 0                      # 本 bank 未命中的逻辑行
    return torch.where(selected[:, None], output.index_select(0, inverse.clamp_min(0)), 0)
                                                # → 严格 0(不是未定义值)
...
result.add_(restored)   # 行 r:要么 vq2 有值且 mxfp4 为 0,要么反之
```

- `subset=True` 路由把 `group_list` 过滤成"本 bank 专家"的物理行,`inverse=-1` 的行在恢复时输出 0;
- 四种模式全部遵循该不变式:`_project_device/_project_prefill` 用 `_restore/restore` 掩码;`_apply_fused` 用 `index_copy_` 写本 bank 专家的行(两 bank 行集天然不相交);serial/batched 按专家 `_bank_for_expert` 查 bank 单路径执行;
- **SwiGLU 统一在两 bank 汇总之后**执行(对完整逻辑行的 `gate_up` 做 `swiglu_unchecked`,行级 pointwise,与该行下游走哪条 Down bank 无关);`swiglu_quant` 融合版按 Down 置换 gather 两输入,与逐元素执行等价(kernel 注释:"SwiGLU is pointwise, so gathering both inputs by the Down permutation is equivalent");
- 求和顺序固定(banks dict 顺序 vq2→mxfp4),浮点累加顺序确定,结果可复现。

#### 5.4.3 VQ2 bank 的投影调用

| 执行模式 | 调用链 | 说明 |
| --- | --- | --- |
| serial | `ops.project`(逐专家,每 128 行一次) | 参考 baseline;group_list 为 CPU 张量(`_group_count` 缓存) |
| batched | `ops.prepare/project_prepared` | 按 `(M, K, N)` 缓存 `ProjectionPlan`;激活逐 bank 一次准备 |
| fused | `ops.prepare_routes`(CPU 计数)+ `ops.project_grouped` | 按实际路由计数分组,无 Python 侧 M/R 上限 |
| full_graph decode | `ops.prepare_device_routes(allow_padding=True)` + `ops.device_routes_status` + `ops.project_grouped_device` | 设备侧计数;复用 bank 的预建 `plan`;输出先填 NaN→检查→掩码 |
| full_graph prefill | 同上,`zero_output=False` | 稀疏写出,由 restore 掩码负责 |

共同约定:激活输入为 Triton kernel 产出的 `(E4M3 字节 [M,K], E8M0 [M,K/32])`;`scales` 指针需 32 字节对齐(不对齐时 clone,`_project_quantized`);输出 BF16,eager 路径 `require_finite` 全量检查。

#### 5.4.4 MXFP4 bank 的 GMM 调用

`_mxfp4`(vq2a8_gs32.py:294-313)对 bank 内专家一次调用:

```python
npu_grouped_matmul_gmm2(
    hidden_states=data.view(torch.float8_e4m3fn),     # 与 VQ2 路同一份激活字节
    weight=bank["fields"]["weight"],                   # float4_e2m1fn_x2,K-major(format 29 预处理)
    weight_scale=bank["fields"]["scale"],              # E8M0,N-major 微缩放块视图
    per_token_scale=scales.reshape(M, K // 64, 2),     # 同一份 E8M0 字节,仅重排布局
    group_list=counts, group_list_type=1,              # 本 bank 各专家计数
    input_dtype=torch.bfloat16, act_quant_type=torch.float8_e4m3fn,
    weight_quant_type=float4_e2m1fn_x2, per_token_scale_type=float8_e8m0fnu,
    use_bf16=True, ...)                                # 输出 BF16
```

要点:激活与 VQ2 路完全同源(同一 Triton kernel、同一 scale 字节);权重是加载时 `prepare_mxfp4_projection` 处理过的常驻格式。**CANN 单 bank GMM 在 A5 上允许 `sum(counts) < M`,其尾部输出未定义** —— 因此恢复前先按 `counts.sum()` 掩码(`live`/restore),在读取与有限性检查之前完成。

#### 5.4.5 各执行模式的 bank 循环对照

**decode 设备路由** `_apply_device`(vq2a8_gs32.py:371,B1 图内,≤ 8 路由行):

1. `group_list` → `counts`(设备侧 clamp/差分),校验:范围、非负、`(counts==0)|resident_mask`(禁止派发到非驻留专家)、总数 == 行数(EP padding 时 ≤);
2. **逐 bank**:`device_router.prepare(hidden, group_list, type, bank["expert_ids_tensor"], bank["logical_slots"], allow_padding=True, subset=True)`(Triton `_route_metadata_kernel`)产出物理行序 `physical/counts/slots/inverse/route_flags`;行重排在寄存器内以 ≤ 8 次"行选择"迭代完成;
3. 激活准备 `prepare_unchecked`:输出 FP8 字节 + E8M0 scale + 每行有效 flags(**无效路由/padding 不触碰输入,直接产出零行与 0 flag** —— 延迟校验永远不会变成未检查的原生指针计算);
4. VQ2:`project_grouped_device` 原生分组投影(输出先填 NaN,按 `counts.sum()` 掩码为 0,`_restore` 恢复逻辑行后 `result.add_`);MXFP4:GMM 后同样掩码+恢复;
5. 全部状态(`route_flags/flags/native_flags/有限性`)AND 成一个设备侧 bool,经 `_accept_validity` 写入 `validity_buffer`(图模式延迟,采样前汇总检查;eager 立即同步);
6. 捕获期间把所有中间张量追加进 `capture_owners`(图池外持有,早先捕获的图仍可回放)。

**prefill 路径** `_apply_prefill`(`_project_prefill`):metadata 只在本 forward 内、由相同专家库存的投影共享(GateUp 与 Down 可有不同 bank,各自独立 metadata);激活准备直接带 `source_rows=gather` 在量化时完成逻辑行→物理行收集;`project_grouped_device(zero_output=False)` 稀疏写出,`device_router.restore` 按掩码恢复进逻辑行并返回 flags;大 buffer 用后即删,不进入 decode 图的 owners。

**eager fused** `_project_fused`:逐 bank 计算 `physical_counts`(本 bank 各专家计数)与行序 `order`(本 bank 专家的逻辑行展平),`index_select` 物理化后投影,`index_copy_` 写回 —— 两 bank 写入的行集不相交。

**serial/batched**:按 `jobs`(专家, 行区间)逐个执行;`_project_batched` 在 `swiglu_quant` 时**每 bank 只做一次激活准备**(不再按 128 行块重复),控制 eager 阶段的有效性同步次数。

#### 5.4.6 两路的数值契约与有效性汇总

| 环节 | VQ2 路 | MXFP4 路 | 一致性 |
| --- | --- | --- | --- |
| 激活量化 | 自家 Triton kernel(E4M3 + E8M0/K32) | 同一 kernel、同一 scale 字节(仅 reshape `[M,K/64,2]`) | ✅ |
| SwiGLU | 汇总后统一执行 / 融合版行级等价 | 同左 | ✅ |
| matmul 输出 | 原生 kernel → BF16 | GMM `use_bf16=True` → BF16 | ✅ 同域求和 |
| 未定义尾行 | 输出填 NaN → 检查 → 掩码 | GMM 尾部未定义 → `live`/restore 掩码 | ✅ 都先掩码后读取 |
| validity | flags AND 汇总进 `validity_buffer` | 同左 | ✅ 统一检查 |

误差差异(VQ2 2-bit vs MXFP4 4-bit 权重近似)是混精的预期语义 —— 由 checkpoint 离线决定,推理侧不做跨类型补偿。

#### 5.4.7 性能代价

同一投影要跑**两轮串行 kernel**(bank 循环 `for quant_type, bank in self.banks[kind].items()`),各 bank 行数少于整层,小 bank 算子利用率低;两种 kernel 的 tiling/速度也不同。混精层耗时 ≈ VQ2 路时间 + MXFP4 路时间 —— 这是为"每专家可独立选精度/省显存"付出的串行开销。

### 5.5 有效性(validity)贯穿设计

所有设备侧校验汇聚为每层一个 bool 标量(`GS32ExpertWeights.validity_buffer`):

- eager(serial/batched/fused):立即 `.cpu()` 同步,失败即抛错;
- full_graph:图内只写设备标志,`initialize_quantization_validity`(runner 侧,见第 6 节)把各层标志聚合为 `runner._quantization_validity`,每次 forward 前 `zero_()`,采样前 `all()` 检查一次 —— 坏路由/坏激活在图回放中不会静默产生错误输出。

## 6. Model Runner 集成(v1/v2、图执行、有效性检查)

### 6.1 量化有效性机制(vq2a8_gs32.py:21 + 两个 runner)

- `initialize_quantization_validity(runner)`:仅当 `quantization=="vq2a8"`、`is_gs32` 且 `gs32_execution=="full_graph"` 生效;要求 `VLLM_USE_BREAKABLE_CUDAGRAPH=0`;把每个专家层的 `vq2a8_weights` 绑定到聚合计量的 unbind 视图(池外稳定视图,回放时重复 zero 检测漏写)。
- **MRV1**(worker/model_runner_v1.py):`load_model` 尾部初始化;`_model_forward` 前对 validity `zero_()`(ACLGraphWrapper 之外,覆盖每次 replay);`_compute_logits` 包装为"先提交 LM Head 再检查有效性"(一次 host 等待覆盖两个提交);`execute_model` 仅对 GS32 owner + PP last rank + 非 pooling 传 `defer_quantization_validity=True`。
- **MRV2**(worker/v2/model_runner.py):`load_model` 在 profiling/图捕获前绑定 validity;`execute_model` 每步先消费上一步 pending 检查,失败时清 `execute_model_state`(坏 forward 不得被 `sample_tokens()` 用);`sample()` 扩展为"提交 LM Head → 校验量化专家 → 采样";捕获预热(aclgraph_utils.py)forward 前 zero、forward 后立即检查 —— 预热必须先拒绝坏路由再录制可用图。

### 6.2 启动 prefill 预热(GS32 startup prefill warmup)

#### 6.2.1 定位:是什么、不是什么

这是一次**形状级编译预热(startup compilation warmup)**:在图捕获之前,用调度器的真实 prefill 预算形状跑一次**真实的 eager fresh prefill 前向**,让所有原生算子与 kernel 在真实形状下完成编译/初始化。

- **不是"预取(prefetch)"**:GS32 权重 bank 加载后常驻设备内存(配置明文禁止 offload/sleep,"native banks require stable resident parameters"),不存在提前搬运权重的需求;预热输入是合成的 `input_ids = arange(N) % vocab_size`、`positions = arange(N)`,不含任何真实请求数据。
- **不是"专家等价遍历"**:路由到哪些专家由模型 router 对该合成输入自行决定(hash 层查 `tid2eid` 表、非 hash 层对合成 hidden 做 top-k),**不保证覆盖每个专家或每种量化类型 bank**,也不做专家数值校验(数值正确性由 nightly 测试 `test_vq2a8_gs32_grouped.py` 等覆盖)。EP 下空专家 rank 也必须执行:其 `group_list` 全 0,`GS32ExpertWeights.apply` 走 `rows==0` 分支返回全零输出并标记有效 —— 目的在于让**每个 rank** 都完成编译缓存与通信路径初始化。
- **为什么必须在图捕获前**:`full_graph` 模式的 decode 图只捕获 B1(≤ 8 路由行)形状,prefill 永远走 eager 设备路由(`_apply_prefill`);若不预热,首个真实请求的 TTFT 将包含全部算子的编译时间。

#### 6.2.2 触发时机与 token 数选择

`NPUWorker`(worker.py:779,位于常规 `_dummy_run` 预热之后、`capture_model()` 之前):

```python
if quantization == "vq2a8" and is_gs32 and model_type == "deepseek_v4":
    num_tokens = min(scheduler_config.max_num_batched_tokens, model_config.max_model_len)
    with self.model_runner._gs32_prefill_warmup(num_tokens) as num_tokens:
        latency_ms = self.profile_prefill_latency(num_tokens)   # 真实 eager 前向 + 计时
```

- 用**调度器的 prefill 预算形状**而非任意形状:让最大的合法 prefill 请求不再有首用编译。
- 每 EP rank(含空专家 rank)都执行;token 容量来自公共 KV 配置,与各 rank 路由/专家库存无关,所有 rank 选择同一 N。

#### 6.2.3 MRV1 实现(`model_runner_v1.py:3668` contextmanager)

进入阶段(构造合成 fresh prefill):

1. 前置断言:`input_batch.num_reqs == 0`(空批),`0 < num_tokens ≤ min(max_num_tokens, max_model_len)`。
2. **KV 页二分裁剪**:对每个 KV group 取 `block_size` 与 block_table 容量,先按表容量夹逼 N;再以 `available_blocks = num_blocks - 1` 为上限二分搜索满足 `Σ cdiv(N, block_size_g) ≤ available_blocks` 的最大 N(块 0 是 null page 不借用)。DeepSeek-V4 多 group 可共享物理缓存存储,因此**跨 group 借用互不相交的页区间**(group 0 用块 `[1,1+c0)`,group 1 紧随其后,……),模拟共享分配器的行为。
3. **快照**所有将被触碰的缓冲区(clone):`positions`、`_dsa_positions_cpu_buf`、`input_ids.gpu`、`query_start_loc`、`optimistic_seq_lens_cpu`、`seq_lens`、各 group 借用区间的 **KV cache 视图**、`block_table.cpu/gpu[0]`、`slot_mapping.gpu`;另存 `num_prompt_tokens/num_computed_tokens_cpu/attn_state/query_lens`。
4. **合成输入**:`input_ids = arange(N) % vocab_size`;每个 group 的 block_table 第 0 行写入借用页号 `[first_block, first_block+num_blocks)`,`slot_mapping = arange(N) + first_block * physical_block_size`;置 `num_prompt_tokens[0]=N`、`num_computed_tokens[0]=0`、`query_start_loc = [0, N]`;标记 `_gs32_prefill_warmup_active = True`。

执行阶段(yield 后由 `profile_prefill_latency` 驱动,worker.py:882):临时 `max_num_reqs=1`,调 `_dummy_run(num_tokens, force_attention=True, is_profile=False, cudagraph_runtime_mode=NONE)` —— 一次带真实注意力的 **eager** 前向。`_dummy_run` 内的预热分支(model_runner_v1.py:3905-3994):

- `attn_state = PrefillNoCache`(非常规 dummy 的 `DecodeOnly`)—— 完整 fresh prefill 语义,DSA/KV 全量写入;
- 跳过 `slot_mapping.gpu.fill_(-1)`(dummy 例程通常作废槽位,但预热用的是真实借来的页);
- `use_compress` 时 `positions = arange(N)`(匹配 fresh prefill)而非 decode 哨兵值 127。

退出阶段(finally):先 `torch.npu.synchronize()`(元数据与 DSA 压缩可能用辅助流),然后逐缓冲区 `copy_(snapshot)` 恢复、恢复 block_table 行数/批状态/attn_state、释放 `device_metadata_executor` 在途提交、再次同步。快照只按本次 N 的大小克隆,不随 `max_model_len`/KV 总预算膨胀。

#### 6.2.4 MRV2 实现(`worker/v2/model_runner.py:265`)

结构与 V1 相同,差异在缓冲区布局与生命周期:

- 前置断言用 V2 语义:`req_states.req_id_to_index` 为空且 `execute_model_state is None`(runner 空闲);
- 借页逻辑同(V2 的 `block_tables.input_block_tables`/`blocks_per_kv_block`),块号写入 `arange(first*factor, end*factor)`;
- 快照集合为 V2 的 `input_buffers`(input_ids/positions/seq_lens/query_start_loc/is_padding)、`slot_mappings[:, :N]`、借用 block table 与 KV cache 视图;
- `prepare_dummy_attn(input_batch)`(model_runner.py:339)把 MRV2 dummy batch 从"DecodeOnly + null page"改标为真实 fresh prefill:`attn_state=PrefillNoCache`、`is_prefilling_np=True`、`prefill_len_np=N`、`num_computed_tokens_np=0`、`positions=arange(N)` 并 `update_cos_sin` 重算旋转位置编码、`input_ids=arange(N)%vocab`;
- `execute_model` 中 `startup_prefill` 恒按 eager 执行(`is_profile=True` 强制),前置消费上一步 pending 的量化有效性检查、每步 `validity.zero_()`;finally 额外清 `execute_model_state` 与 `_quantization_validity_pending`(预热 forward 绝不能被 `sample_tokens()` 消费)。

#### 6.2.5 效果

- 原生算子(`mixed_w2w4_moe_ffn` tiling/编译、ACLNN GMM、DSA/compressor 算子)、Triton 激活/路由 kernel、HCCL ALLGATHER 通信在真实 prefill 形状下完成初始化;
- 顺带产出该形状的 prefill 延迟测量(复用 `profile_prefill_latency` 的计时逻辑,记入日志);
- 结束后 runner 状态与 KV cache 完全还原,不产生任何用户可见副作用(预热输出丢弃,KV 借用页内容被快照恢复)。

### 6.3 其它 runner 侧配套

- `acl_graph.py`:`ACLGraphEntry.num_replays` 计数器(测试验证 decode 真实走图回放);
- `device_allocator/camem.py`:`vllm_ascend_C` 改惰性导入(避免在模型选择原生 provider 前注册无关算子);
- KV/注意力配套:`attention/dsa_kv_quant.py`(TP 下 FP8 共享 KV 注意力 head 数补零到 64/128)、`attention/dsa_attn_kv_plan.py`、`models/deepseek_v4/*`(compressor/indexer 等)。

## 7. 启动配置、执行模式与并行约束

配置入口(`tools/serve_vq2a8.py` 或 `vllm serve --additional-config`):

| 配置 | 取值 | 说明 |
| --- | --- | --- |
| `--quantization` | `vq2a8` | 必需;配合 `--dtype bfloat16`、`--load-format auto/safetensors` |
| `vq2a8.gs32_library` | 绝对路径(必需) | `libvq2a8_gs32_stage.so`(Torch 桥接库) |
| `vq2a8.gs32_binary` | 绝对路径(必需) | `mixed_w2w4_moe_ffn.o`(原生算子 ELF) |
| `vq2a8.gs32_execution` | `serial`(默认)/`batched`/`fused`/`full_graph` | 执行模式 |
| `vq2a8.gs32_fusion` | `none`(默认)/`swiglu_quant` | SwiGLU+量化融合 |

`full_graph` 硬性要求:`enforce_eager=False`、`cudagraph_mode=FULL_DECODE_ONLY`、`capture_sizes=[1]`、`max_num_seqs=1`(B1 单序列 decode)、warmup ≥ 1、`enable_npugraph_ex=false`、`multistream_dsv4_dsa_overlap=false`、同步调度、`VLLM_USE_BREAKABLE_CUDAGRAPH=0`、top-k ∈ [1,8];长上下文(>128 token)需 DeepSeek-V4 attention + chunked prefill + 正的 token 预算。其余模式要求 `enforce_eager=True` + 图模式 NONE。图只捕获一个 decode token(≤ 8 路由行),增大 prefill chunk 不扩大 decode 图;`full_graph_prefill` 不是有效模式。

**并行约束**:普通层 TP;路由专家整专家 EP、线性放置,TP/EP 规模 ∈ {1,2,4,8},专家数必须被整除,每 rank 持完整本地专家(矩阵维不切分)。EP 下通信强制 ALLGATHER。拒绝:专家张量切分、PP/DP/CP>1、EPLB(冗余专家/动态重排/专家映射)、LoRA、投机解码、KV 传输、异步调度、offload/sleep、mix_placement/共享专家 DP/FlashComm/DSA-CP。

## 8. 原生算子层(csrc)与构建

### 8.1 `csrc/mixed_w2w4_moe_ffn/`(独立融合算子)

- 融合 Gate/Up GEMM + SwiGLU + 中间 K32 MXFP8 动态量化 + Down GEMM,输出 BF16;**不含** router/dispatch/加权 combine。
- 每专家按 `is_expert_w4[e]`/`is_down_w4[e]` 选择路径:支持 2+2(VQ2 GateUp+VQ2 Down)、2+4(VQ2+MXFP4)、4+4(MXFP4+MXFP4),不支持 4+2。
- 目录:components/{gate_w2,gate_w4,down_w2,down_w4}(AIC/AIV `.cce` kernel + tiling builder)、`mixed_w2w4_moe_ffn_host.cc`(host 启动/校验)、`hacl_rt.h`(CANN 低层运行时兼容层)、scripts(gen_data/w2_layout/w4_layout/swiglu_reference)、独立用例与 ESL 运行脚本。
- 来源:用户恢复的 `w2a8-matmul` 项目(`operator_code.md`),自 `feat/test@592a6973` 导入,`SOURCE_MANIFEST.json` 记录 39 个原始文件哈希;本分支仅做重命名(mat_fp4→mixed_w2w4_moe_ffn)、CRLF/格式清理,不改 kernel 数学/tiling/布局。

### 8.2 `csrc/vq2a8_gs32_stage/torch_binding.cpp`(Torch 桥接)

- 注册 `torch.ops.vq2a8_gs32_stage`:`initialize`(加载 ELF)、`make_tiling`、`prepare/project_prepared/project`(eager 投影)、`prepare_routes/prepare_grouped/project_grouped`(分组)、`prepare_device_routes/device_routes_status/project_grouped_device`(设备侧路由,`allow_padding` 重载)、`supports_dynamic_grouped_rows/supports_dynamic_device_rows/supports_sparse_prefill_output`(能力探测,bridge 过旧时报错要求重建);
- 投影计划/路由句柄为 custom class(不透明对象),张量 owner 记录在提交流上,别名/尺寸校验,图捕获安全(捕获中只记录设备写);
- 只复用独立算子的 **DownW2 投影阶段**实现自然顺序的 GateUp/Down 投影(分组路径无历史上的单专家 4096 行/合计 32768 行上限,serial 保留 128 行/次)。

### 8.3 构建流程(docs/vq2a8/vq2a8_gs32_runtime.md)

```bash
bash csrc/mixed_w2w4_moe_ffn/run_mixed_w2w4_moe_ffn_esl.sh        # 编译 AIC/AIV 并链接算子 ELF
python tools/build_vq2a8_gs32_stage.py \
  --cann "$ASCEND_TOOLKIT_HOME" \
  --kernel-binary "$PWD/csrc/mixed_w2w4_moe_ffn/mixed_w2w4_moe_ffn.o" \
  --build-dir "$PWD/build/vq2a8-gs32-runtime"                      # 仅编译桥接层并校验 schema
# 产物:libvq2a8_gs32_stage.so + mixed_w2w4_moe_ffn.o
```

注意:仅构建 GS32 桥接库**不足以**运行完整 DeepSeek-V4 —— 还需同一软件栈构建安装完整 `vllm_ascend_C` 扩展及 ACLNN 算子(含 SAS/QLI AICPU 元数据修正)。

## 9. 测试与工具布局

### 测试

- `tests/ut/quantization/`(约 30 个文件):CPU 契约测试(激活、checkpoint 加载、EP bank/comm/weights、设备 bank/routes、图配置/owner、grouped/plan 缓存、MRV2、量化 oracle、serving、KV cache、参考实现 `vq2a8_gs32_reference.py` 等);入口 `python -X utf8 tools/run_vq2a8_cpu_tests.py`。
- `tests/e2e/nightly/`(14 个文件):NPU 回归 —— `test_vq2a8_gs32.py`(基础)、`_activation/_activation_async/_device_routes/_grouped/_device_native/_device_bank`(算子边界)、`_batched/_full_graph`(执行模式,含 V1/V2 与 decode 回放探针)、`_ep/_ep_attention/_ep_routes/_ep_long_context`(EP);公共参数 `--vq2a8-model/--vq2a8-gs32-library/--vq2a8-gs32-binary`。

### 工具(tools/)

| 脚本 | 用途 |
| --- | --- |
| `serve_vq2a8.py` | 服务启动器:拼装标准 vllm serve 命令(BF16/safetensors/TP=EP/block128/量化选项;full_graph 时切 FULL_DECODE_ONLY + capture [1] + 关 npugraph_ex) |
| `build_vq2a8_gs32_stage.py` | 基于 ELF 仅编译 Torch 桥接层 + schema 校验 |
| `convert_vq2_mixed_gs32_v3.py` | source-v1 → source-v3 重打包(纯标准库,字节级 FP8 处理) |
| `run_vq2a8_cpu_tests.py` | CPU 契约测试入口(隔离引擎初始化) |
| `benchmark_vq2a8_serving.py` | OpenAI SSE 流式 TTFT/TPOT 基准 |
| `vq2a8_live_log.py` | 子进程日志 tee(磁盘日志 + 有界 reader 线程;识别 MODEL/MOE/KERNEL/... 阶段心跳) |
| `tools/tests/run_vq2a8_npu_ci.py` | 一键 NPU 回归(kernels/pr/nightly 顺序套件,JUnit 报告,覆盖缺口声明) |
| `tools/tests/run_vq2a8_performance_suite.py` / `run_vq2a8_accuracy_suite.py`(EvalScope:GPQA Diamond/AIME24/25)/ `run_vq2a8_ep2_long_context_latency.py`(EP2 64K/256K/1M) | 性能/精度/长上下文套件 |

### 文档(docs/vq2a8/)

`vq2a8_gs32.md`(总览)、`vq2a8_gs32_runtime.md`(运行时与构建)、`vq2_mixed_gs32_export_spec_20261016.md`(checkpoint 交付规范)、`vq2a8_npu_ci.md` / `vq2a8_performance.md` / `vq2a8_accuracy.md`(三类测试指南)。

## 10. 关键文件索引

| 文件 | 行数(新增/变化) | 角色 |
| --- | --- | --- |
| `vllm_ascend/quantization/configs/vq2a8_config.py` | 407 | source-v3 审计、选项校验、方法分发 |
| `vllm_ascend/quantization/methods/w2a8/vq2a8_gs32.py` | 208 | MoE scheme:参数创建/加载/apply |
| `.../w2a8/vq2a8_gs32_format.py` | 270 | 格式契约、字段转换、选项校验 |
| `.../w2a8/vq2a8_gs32_ep.py` | 62 | 线性整专家 EP 放置 |
| `.../w2a8/vq2a8_gs32_staged.py` | 80 | MXFP8/SwiGLU 数值参考契约 |
| `vllm_ascend/ops/fused_moe/vq2a8_gs32.py` | 728 | `GS32ExpertWeights` 运行时(四种模式) |
| `vllm_ascend/ops/fused_moe/vq2a8_gs32_activation.py` | 179 | 激活入口(eager/deferred 校验) |
| `vllm_ascend/ops/fused_moe/vq2a8_gs32_routes.py` | 143 | 设备路由入口与契约 |
| `vllm_ascend/ops/triton/activation/vq2a8_gs32.py` | 237 | MXFP8 量化/融合 SwiGLU/flags kernel |
| `vllm_ascend/ops/triton/activation/vq2a8_gs32_routes.py` | 492 | 路由元数据/行重排/恢复 kernel |
| `csrc/vq2a8_gs32_stage/torch_binding.cpp` | 864 | Torch 桥接(算子注册/计划/路由) |
| `csrc/mixed_w2w4_moe_ffn/`(整目录) | ~6600 | 独立混合 W2/W4 FFN 融合算子 |
| `vllm_ascend/models/deepseek_v4/model.py` | +165 | checkpoint 绑定、source-v3 加载、FP32 router |
| `vllm_ascend/models/deepseek_v4/native_ops.py` | +50 | DSV4 原生绑定检查 |
| `vllm_ascend/worker/model_runner_v1.py` / `worker/v2/model_runner.py` | 各 +199 | validity、图执行、启动 prefill 预热 |
| `vllm_ascend/ops/fused_moe/moe_mlp.py:842` | +10 | VQ2A8 → `vq2a8.apply` 分发点 |
| `vllm_ascend/ascend_forward_context.py` | +20 | GS32 EP 强制 ALLGATHER |
