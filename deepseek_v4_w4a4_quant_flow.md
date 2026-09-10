# DeepSeek-V4-Flash W4A4 权重量化代码流程详解

> 对象：`/mnt/share/rr08002/weights/DeepSeek-V4-Flash-w4a4`(msModelSlim 混精量化 ckpt)
> 代码基线：vllm-ascend releases/v0.23.0（含 wo_a 特判修复）
> 日期：2026-08-31

## 总览

```
quant_model_description.json 存在
  → AscendModelSlimConfig ("ascend" 量化方式)
    → 逐层 get_quant_method 分发 scheme
      → 权重加载后 process_weights_after_loading 变形
        → forward 时按 scheme 走各自的量化算子
```

| 层 | 描述类型 | scheme | 运行时算子 |
|---|---|---|---|
| attn.wq_a / wq_b / wkv、indexer.wq_b | W4A4_MXFP4 | `AscendW4A4MXFP4DynamicLinearMethod` | fp4 动态量化 + fp4 quant_matmul |
| attn.wo_a | W8A8_MXFP8 | `AscendW8A8MXFP8DSWoADynamicLinearMethod`(特判) | 3D 分组 quant batchmatmul |
| attn.wo_b、ffn.shared_experts.w1/w2/w3 | W8A8_MXFP8 | `AscendW8A8MXFP8DynamicLinearMethod` | fp8 动态量化 + fp8 quant_matmul |
| ffn.experts.*(MoE) | W4A4_MXFP4 | `AscendW4A4MXFP4DynamicFusedMoEMethod` | MC2/allgather + fp4 experts GEMM |
| gate / compressor / norm / embed / head / hc_* | FLOAT | Unquantized | bf16/fp32 原生 |

---

## 第 0 步：量化方式探测

`vllm_ascend/quantization/utils.py`（约 90-130 行）:

```python
# Detection priority:
#   1. ModelSlim (Ascend) – quant_model_description.json exists.
#   2. LLM-Compressor     – config.json 里 quantization_config.quant_method == "compressed-tensors"
#   3. None               – 浮点
from vllm_ascend.quantization.modelslim_config import MODELSLIM_CONFIG_FILENAME

# Case 1: ModelSlim — look for quant_model_description.json
modelslim_path = get_model_file(model, MODELSLIM_CONFIG_FILENAME, revision=revision)
if modelslim_path is not None:
    return ASCEND_QUANTIZATION_METHOD        # → "ascend"
```

W4A4 ckpt 的 `config.json` **没有** `quantization_config` 字段（原生 FP8 ckpt 有，`quant_method: fp8`，走的是 `AscendFp8Config`——这是两条完全不同的路径），因此走 ModelSlim 路径，量化配置类为 `AscendModelSlimConfig`(`quantization/modelslim_config.py`)。

## 第 1 步：加载量化描述文件

`AscendModelSlimConfig.maybe_update_config`(modelslim_config.py:760):

```python
config_path = get_model_file(model_name, MODELSLIM_CONFIG_FILENAME, revision=revision)
if config_path is not None:
    with open(config_path) as f:
        self.quant_description = json.load(f)   # 逐张量量化类型表
    self._apply_extra_quant_adaptations()
    self._add_kvcache_quant_metadata()
```

W4A4 ckpt 的描述文件实际内容（抽样）:

```
layers.0.attn.wq_a.weight          -> W4A4_MXFP4
layers.0.attn.wq_b.weight          -> W4A4_MXFP4
layers.0.attn.wkv.weight           -> W4A4_MXFP4
layers.0.attn.wo_a.weight          -> W8A8_MXFP8      ← 注意
layers.0.attn.wo_b.weight          -> W8A8_MXFP8
layers.10.attn.indexer.wq_b.weight -> W4A4_MXFP4
layers.0.ffn.experts.N.w1/w2/w3    -> W4A4_MXFP4
layers.0.ffn.shared_experts.w1..w3 -> W8A8_MXFP8
layers.0.ffn.gate.weight           -> FLOAT
layers.0.attn.compressor.*         -> FLOAT
layers.0.attn.*norm.weight         -> FLOAT
embed.weight / head.weight         -> FLOAT
hc_* / attn_sink                   -> FLOAT
顶层: model_quant_type=W4A4_MXFP4, group_size=32
```

磁盘上的张量格式（safetensors header 实测）:

| 张量 | dtype | shape | 说明 |
|---|---|---|---|
| wq_a.weight | U8 | [1024, 2048] | fp4 e2m1 双打包，K=4096/2 |
| wq_a.weight_scale | U8 | [1024, 128] | e8m0,K/32=128 |
| wo_a.weight | F8_E4M3 | [8192, 4096] | fp8,[G*R, K] |
| wo_a.weight_scale | U8 | [8192, 128] | e8m0,per-32 K 组 |
| experts.N.w1.weight | U8 | [2048, 2048] | fp4 打包 |
| gate.weight | BF16 | [256, 4096] | 不量化 |

同时 `packed_modules_model_mapping["deepseek_v4"]`(modelslim_config.py:107）处理 gate_up_proj 等融合前缀映射：

```python
"deepseek_v4": {
    "gate_up_proj": ["gate_proj", "up_proj"],
    "experts": ["experts.0.gate_proj", "experts.0.up_proj", "experts.0.down_proj"],
},
```

## 第 2 步：逐层 scheme 分发

入口：`AscendModelSlimConfig.get_quant_method`(modelslim_config.py:604)。

### 2.1 LinearBase 分支（含 wo_a 特判，本次修复新增）

```python
if isinstance(layer, LinearBase):
    if self.is_layer_skipped_ascend(prefix, self.packed_modules_mapping):
        # 描述里标 FLOAT 的层(gate、compressor、weights_proj、embed/head)
        from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
        return AscendUnquantizedLinearMethod()

    # DeepSeek-V4's DSA attention consumes wo_a.weight directly via
    # npu_transpose_quant_batchmatmul, which requires a grouped 3D
    # [n_local_groups, K, o_lora_rank] layout. The generic W8A8_MXFP8
    # scheme keeps the weight 2D, so use the wo_a-aware scheme instead.
    if (
        model_type == "deepseek_v4"
        and prefix.endswith("wo_a")
        and get_linear_quant_type(self.quant_description, prefix, self.packed_modules_mapping)
        == "W8A8_MXFP8"
    ):
        from .methods.w8a8_mxfp8 import AscendW8A8MXFP8DSWoADynamicLinearMethod
        return AscendLinearMethod(AscendW8A8MXFP8DSWoADynamicLinearMethod())

    scheme = create_scheme_for_layer(self.quant_description, prefix, "linear",
                                     self.packed_modules_mapping)
    return AscendLinearMethod(scheme)
```

`create_scheme_for_layer` → `get_quant_type_for_layer` 从描述表取该层的 quant_type → 注册表 `get_scheme_class(quant_type, "linear")` 查表：

| quant_type | scheme 类 | 位置 |
|---|---|---|
| W4A4_MXFP4 | `AscendW4A4MXFP4DynamicLinearMethod` | methods/w4a4_mxfp4.py:39 |
| W8A8_MXFP8 | `AscendW8A8MXFP8DynamicLinearMethod` | methods/w8a8_mxfp8.py:43 |

### 2.2 FusedMoE 分支

```python
elif _is_fused_moe_layer(layer):
    ...
    scheme = create_scheme_for_layer(self.quant_description, prefix, "moe", ...)
    return AscendFusedMoEMethod(scheme, layer.moe_config, tid2eid)
```

`("W4A4_MXFP4","moe")` → `AscendW4A4MXFP4DynamicFusedMoEMethod`(w4a4_mxfp4.py:118)。

### 2.3 Attention 分支

描述文件中没有 `fa_quant_type` / `indexer_quant_type` / `kv_cache_type` 顶层键 →
`is_fa_quant_layer` / `is_indexer_quant_layer` / `is_c8_quant_layer` 全 False →
attention 层**不挂量化 method**,KV cache 为 bf16，注意力 kernel 走非量化路径。

### 2.4 Embedding

`embed.weight` / `head.weight` 为 FLOAT → `UnquantizedEmbeddingMethod`(bf16)。

## 第 3 步：权重加载后处理（process_weights_after_loading)

vLLM 加载完权重后由 `method_adapters.AscendLinearMethod.process_weights_after_loading`(method_adapters.py:124）转发给 scheme。

### 3.1 W4A4_MXFP4 linear(w4a4_mxfp4.py:109)

```python
def process_weights_after_loading(self, layer):
    # weight: (output_size, input_size) -> (input_size, output_size)
    # weight_scale: (n_dim, k_dim) -> (k_dim//2, n_dim, 2)
    n_dim, k_dim = layer.weight_scale.data.shape
    layer.weight_scale.data = layer.weight_scale.data.reshape(n_dim, k_dim // 2, 2)
    layer.weight.data = layer.weight.data.transpose(0, 1)
    layer.weight_scale.data = layer.weight_scale.data.transpose(0, 1)
```

参数创建（`get_weight` / `get_pergroup_param`）与 ckpt 格式一一对应：

```python
# get_weight:      weight       = torch.empty(output_size, input_size // 2, dtype=uint8)
# get_pergroup:    weight_scale = torch.empty(output_size, input_size // 32, dtype=uint8)
```

### 3.2 W8A8_MXFP8 linear(w8a8_mxfp8.py:120)

```python
n_dim, k_dim = layer.weight_scale.data.shape
if layer.weight_scale.data.shape[-1] % 2 != 0:          # 奇数 K/32 补零
    layer.weight_scale.data = F.pad(layer.weight_scale.data, (0, 1), ...)
layer.weight_scale.data = layer.weight_scale.data.reshape(n_dim, k_dim // 2, 2)
layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()        # → [K, N]
layer.weight_scale.data = layer.weight_scale.data.transpose(0, 1).contiguous()  # → [K/64, N, 2]
```

注意：这里**不做** fp32→e8m0 的指数提取——W4A4 ckpt 的 scale 本来就是 e8m0 字节。（对比：原生 FP8 ckpt 走的 `AscendW8A8MXFP8DSDynamicLinearMethod`(fp8.py:60）才需要 `view(int32)>>23&0xFF` 提取指数，因为那边描述的是 fp32 的 128×128 block scale。)

### 3.3 wo_a 特判 scheme（本次新增，w8a8_mxfp8.py:194)

```python
class AscendW8A8MXFP8DSWoADynamicLinearMethod(AscendW8A8MXFP8DynamicLinearMethod):
    """W8A8_MXFP8 linear method for DeepSeek-V4 ``wo_a`` loaded via ModelSlim.
    ...DS attention 要求 [n_local_groups, K, o_lora_rank] 3D 布局..."""

    def __init__(self):
        super().__init__()
        vllm_config = get_current_vllm_config()
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        hf_config = vllm_config.model_config.hf_config
        self.n_local_groups = hf_config.o_groups // tp_size     # 8 / tp
        self.o_lora_rank = hf_config.o_lora_rank                # 1024

    def process_weights_after_loading(self, layer):
        # 基类通用 2D 变换: weight -> (K, G*R), scale -> (K//64, G*R, 2)
        super().process_weights_after_loading(layer)
        # weight: (K, G*R) -> (G, R, K) -> (G, K, R)
        layer.weight.data = (
            layer.weight.data.T.reshape(self.n_local_groups, self.o_lora_rank, -1)
            .transpose(1, 2).contiguous()
        )
        # scale: (K//64, G*R, 2) -> (G, R, K//64, 2) -> (G, K//64, R, 2)
        layer.weight_scale.data = (
            layer.weight_scale.data.transpose(0, 1)
            .reshape(self.n_local_groups, self.o_lora_rank, -1, 2)
            .transpose(1, 2).contiguous()
        )
```

shape 推演（TP=1, G=8, R=1024, K=4096):

```
weight: [8192, 4096] → base → [4096, 8192] → 本类 → [8, 4096, 1024]  ([G, K, N])
scale : [8192, 128]  → base → [64, 8192, 2] → 本类 → [8, 64, 1024, 2] ([G, K/64, N, 2])
```

**为什么必须 3D**:wo_a 是分组低秩投影（o_groups=8 组 head 各自独立投影到 o_lora_rank=1024),DSA 实现用 batched matmul 一次算 8 组；checkpoint 把 8 个块沿 dim0 摞成 2D `[G*R, K]` 存储，加载后必须还原出 batch 维。

### 3.4 W4A4_MXFP4 MoE(w4a4_mxfp4.py:250)

```python
def process_weights_after_loading(self, layer):
    g_num, n_size, k_size = layer.w13_weight_scale.shape
    layer.w13_weight_scale.data = layer.w13_weight_scale.data.reshape(g_num, n_size, k_size // 2, 2)
    g_num, n_size, k_size = layer.w2_weight_scale.shape
    layer.w2_weight_scale.data = layer.w2_weight_scale.data.reshape(g_num, n_size, k_size // 2, 2)
    # The A5 MXFP4 fused grouped-matmul-swiglu op relies on the
    # transpose stride to interpret packed FP4 weights as logical K.
    layer.w13_weight.data = layer.w13_weight.data.transpose(1, 2)
    layer.w2_weight.data = layer.w2_weight.data.transpose(1, 2)
    layer.w13_weight_scale.data = layer.w13_weight_scale.data.transpose(1, 2)
    layer.w2_weight_scale.data = layer.w2_weight_scale.data.transpose(1, 2)
```

## 第 4 步：运行时 apply(每次 forward)

### 4.1 W4A4 线性层(wq_a/wq_b/wkv/indexer.wq_b)—— w4a4_mxfp4.py:72

```python
def apply(self, layer, x, bias=None, tp_rank=0):
    original_shape = x.shape
    if x.dim() > 2:
        x = x.view(-1, x.shape[-1])
    quantized_x, dynamic_scale = torch_npu.npu_dynamic_mx_quant(
        x, dst_type=torch_npu.float4_e2m1fn_x2, round_mode="round"   # 激活动态量化到 FP4
    )
    output = torch_npu.npu_quant_matmul(
        quantized_x,
        layer.weight,                # [K, N] fp4 packed
        layer.weight_scale,          # [K/64, N, 2] e8m0
        scale_dtype=FLOAT8_E8M0FNU_DTYPE,
        pertoken_scale=dynamic_scale,
        pertoken_scale_dtype=FLOAT8_E8M0FNU_DTYPE,
        bias=bias,
        output_dtype=output_dtype,   # bf16
        x1_dtype=torch_npu.float4_e2m1fn_x2,
        x2_dtype=torch_npu.float4_e2m1fn_x2,
        group_sizes=[1, 1, self.group_size],   # K 方向 32 粒度
    )
```

### 4.2 W8A8 线性层(wo_b、共享专家)—— w8a8_mxfp8.py:77

```python
quantized_x, pertoken_scale = torch_npu.npu_dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn)
output = torch_npu.npu_quant_matmul(
    quantized_x, layer.weight, layer.weight_scale,
    scale_dtype=FLOAT8_E8M0FNU_DTYPE,
    pertoken_scale=pertoken_scale,
    pertoken_scale_dtype=FLOAT8_E8M0FNU_DTYPE,
    bias=bias, output_dtype=output_dtype,
    group_sizes=[1, 1, self.group_size],
)
```

### 4.3 wo_a —— 不走 apply，由 attention impl 直取权重

`vllm_ascend/attention/dsa_v1.py:1556` `_forward_o_proj`(A5 分支）:

```python
o_proj_input = o_proj_input.view(num_tokens, self.n_local_groups, group_hidden_dim)
if get_ascend_device_type() in {AscendDeviceType.A5}:
    o = o_proj_input
    o, swiglu_out_scale = torch_npu.npu_dynamic_mx_quant(o, dst_type=torch.float8_e4m3fn)
    o = torch_npu.npu_transpose_quant_batchmatmul(
        o,                             # [G, M, K]  (perm_x1=(1,0,2))
        self.wo_a.weight,              # [G, K, R]  ← 第 3.3 节 reshape 的产物
        dtype=torch.bfloat16,
        bias=None,
        group_sizes=(0, 0, 32),
        x1_scale=swiglu_out_scale.view(torch.float8_e8m0fnu),
        x2_scale=self.wo_a.weight_scale.view(torch.float8_e8m0fnu),  # [G, K/64, R, 2]
        perm_x1=(1, 0, 2), perm_x2=(0, 1, 2), perm_y=(1, 0, 2),
    )
    o = o.reshape(num_tokens, -1)
    output[...] = self.wo_b(o)         # wo_b 走 4.2 的 W8A8 apply
```

### 4.4 MoE 路由专家（在线量化，精确到算子）

#### 4.4.1 入口：scheme 的 apply(w4a4_mxfp4.py:163)

```python
def apply(self, layer, x, router_logits, top_k, ...):
    topk_weights, topk_ids = select_experts(...)          # 路由(不做量化)
    moe_comm_method = _EXTRA_CTX.moe_comm_method
    return moe_comm_method.fused_experts(
        fused_experts_input=build_fused_experts_input(
            hidden_states=x, topk_weights=topk_weights, topk_ids=topk_ids,
            w1=layer.w13_weight, w2=layer.w2_weight,
            quant_type=self.quant_type,                        # QuantType.MXFP4
            mxfp_act_quant_type=torch_npu.float4_e2m1fn_x2,    # ← 激活 fp4
            mxfp_weight_quant_type=torch_npu.float4_e2m1fn_x2, # ← 权重 fp4
            mxfp_scale_dtype=FLOAT8_E8M0FNU_DTYPE,             # scale 全 e8m0
            mxfp_per_token_scale_dtype=FLOAT8_E8M0FNU_DTYPE,
            mxfp_use_bf16=(x.dtype in [torch.bfloat16, torch.uint8]),
            w1_scale=layer.w13_weight_scale, w2_scale=layer.w2_weight_scale,
        ), ...)
```

`moe_comm_method` 的选择（`ascend_forward_context.py:285` `_select_a5_moe_comm_method`）与量化类型无关，只取决于 token 数：

```python
if num_tokens <= mc2_tokens_capacity and world_size > 1:
    return MoECommType.MC2          # decode 走这里(融合通信)
if world_size <= num_experts_per_tok:
    return MoECommType.ALLGATHER    # 长 prefill(超 MC2 容量)走这里
return MoECommType.ALLTOALL
```

两条路径共用同一个骨架（`MoECommMethod.fused_experts`,moe_comm_method.py:122):
`token_dispatch` → `unified_apply_mlp`(moe_mlp.py:461)→ `token_combine`。**fp4 量化发生在哪一步，两条路径不同**。

#### 4.4.2 MC2 路径（decode):dispatch 算子内部量化成 fp4

**Dispatch** —— `TokenDispatcherWithMC2.token_dispatch`(token_dispatcher.py:225):

```python
torch_npu.npu_moe_distribute_dispatch_v2(
    x=hidden_states,            # bf16 [num_tokens, 4096]
    expert_ids=topk_ids,
    expert_scales=topk_weights.to(torch.float32),
    expert_shard_type=0, shared_expert_rank_num=0,
    moe_expert_num=256, global_bs=0,
    x_active_mask=mc2_mask,
    quant_mode=4,               # MXFP 通信量化(A5)
    y_dtype=torch.float4_e2m1fn_x2,   # ← apply 传入的 mxfp_act_quant_type,fp4!
    group_ep=..., ep_world_size=8, ep_rank_id=...,
    group_tp=..., tp_world_size=1, tp_rank_id=0,
    comm_alg="hierarchy",
)
# 返回: expand_x(fp4 packed)、dynamic_scale(e8m0, per-token per-32)、
#       assist_info_for_combine、expert_token_nums、ep/tp_recv_counts、expand_scales
```

关键点：激活的 **bf16→fp4 量化在 dispatch 算子内部完成**,token 以 fp4 形式跨卡传输。`quant_mode=4` 的判定（token_dispatcher.py:165-170):MXFP 且 A5 → 4;`y_dtype` 由 `mxfp.act_quant_type` 决定（token_dispatcher.py:202-211),W4A4 时为 fp4。

**MLP** —— `quant_apply_mlp`(moe_mlp.py:88)。此时 `dynamic_scale` 来自 dispatch（非 None)，走"已量化"分支，**不再重复量化**:

```python
# moe_mlp.py:142-146(dynamic_scale 非 None 分支)
pertoken_scale = DeviceOperator.maybe_normalize_mxfp_scale_layout(dynamic_scale)
                 # e8m0 scale [M, K/32] → [M, K/64, 2](两个 32 组成对打包)
quantized_hidden_states = hidden_states        # dispatch 出来的 fp4,直接用
```

随后 `is_mc2=True` 且 `use_gmm_swiglu_quant_fusion=use_mxfp_quant=True`(mxfp 恒走融合）,GMM1+SwiGLU+重量化**一个融合算子**完成（moe_mlp.py:168 → device_op.py:1323 → MXFP4 落入 else 分支 device_op.py:1396):

```python
torch_npu.npu_grouped_matmul_swiglu_quant_v2(
    x=hidden_states,                    # fp4 [Σtokens_per_expert, K]
    weight=[w13_weight],                # fp4 packed [E, 2*I, K](已转置)
    group_list=cumsum_group_list(group_list, group_list_type, 0),
    weight_scale=[w13_weight_scale],    # e8m0 [E, K/64, 2I, 2]
    x_scale=pertoken_scale,             # e8m0
    dequant_mode=2, quant_mode=2,
    dequant_dtype=torch.float32,
    quant_dtype=torch.float4_e2m1fn_x2,      # 输出重量化为 fp4
    x_dtype=torch.float4_e2m1fn_x2,
    weight_dtype=torch.float4_e2m1fn_x2,
    weight_scale_dtype=torch.float8_e8m0fnu,
    x_scale_dtype=torch.float8_e8m0fnu,
)
# 返回: swiglu 后的 fp4 激活 + swiglu_out_scale(e8m0)
```

GMM2(down_proj)—— `DeviceOperator.npu_grouped_matmul_gmm2`(moe_mlp.py:210 → device_op.py:1454;MXFP4 跳过 W4A8MXFP/W4A16MXFP4 特判，落到通用 MXFP 尾部）:

```python
torch_npu.npu_grouped_matmul(
    x=[hidden_states],                  # fp4(swiglu 输出)
    weight=[w2_weight],                 # fp4 packed [E, O, I]
    scale=[w2_weight_scale],            # e8m0
    per_token_scale=[swiglu_out_scale], # e8m0(上一步的输出 scale)
    bias=None, split_item=2, group_type=0,
    group_list=group_list, group_list_type=group_list_type,
    x_dtype=torch.float4_e2m1fn_x2, weight_dtype=torch.float4_e2m1fn_x2,
    scale_dtype=torch.float8_e8m0fnu,
    per_token_scale_dtype=torch.float8_e8m0fnu,
    output_dtype=torch.bfloat16,        # 输出回 bf16
)
```

**Combine** —— `TokenDispatcherWithMC2.token_combine`(token_dispatcher.py:329)→ `get_combine_mc_kwargs`:MXFP4 时 `quant_mode=0`（注释原文：quant_mode=4 目前仅 MXFP8 用）→ `torch_npu.npu_moe_distribute_combine_v2(expand_x=bf16 结果, ...)` 以 **bf16 不量化**传回并加权求和。

```
MC2 路径量化时机总结:
  bf16 hidden ──dispatch_v2 内部量化──> fp4 跨卡传输
      ──> npu_grouped_matmul_swiglu_quant_v2(融合:GMM1+SwiGLU+requant,fp4 进 fp4 出)
      ──> npu_grouped_matmul(GMM2,fp4×fp4 → bf16)
      ──> combine_v2(bf16 传回)
```

#### 4.4.3 AllGather 路径（长 prefill):dispatch 不量化，MLP 入口在线量化

**Dispatch** —— `TokenDispatcherWithAllGather.token_dispatch`(token_dispatcher.py:351):

```python
unquantized_mxfp4_dispatch = quant_type == QuantType.MXFP4 and dynamic_scale is None
# → True:MXFP4 在 allgather dispatch 阶段不量化(代码注释原话:
#   "MXFP4 stays unquantized in dispatch and is quantized again inside the MLP path")
with_quant = ... and not unquantized_mxfp4_dispatch   # → False → quant_mode = -1

sorted_hidden_states, expanded_row_idx, expert_tokens, dynamic_scale =
    DeviceOperator.npu_moe_init_routing(
        hidden_states,                  # bf16(allgather 收集的全量 token)
        topk_ids, scale=None, active_num=num_tokens*top_k,
        expert_num=256, expert_tokens_num_type=1, expert_tokens_num_flag=True,
        active_expert_range=[first_expert_idx, last_expert_idx],
        quant_mode=-1,                  # 只排序/重排,不量化
        act_quant_type=None,
    )
# dynamic_scale 仍为 None
```

**MLP** —— 同一个 `quant_apply_mlp`，但这次 `dynamic_scale is None` → 走在线量化分支（moe_mlp.py:131 → device_op.py:1302):

```python
hidden_states, pertoken_scale = DeviceOperator.npu_dynamic_quant(
    hidden_states=hidden_states, dynamic_scale=None,
    act_quant_type=torch.float4_e2m1fn_x2, use_mxfp_quant=True,
)
# 内部即:torch_npu.npu_dynamic_mx_quant(hidden_states, dst_type=float4_e2m1fn_x2)
#   + maybe_normalize_mxfp_scale_layout(scale)  # [M, K/32] → [M, K/64, 2]
```

之后 GMM1+SwiGLU(`npu_grouped_matmul_swiglu_quant_v2`)、GMM2(`npu_grouped_matmul`）与 MC2 路径**同一组算子、同一组 dtype 参数**(`is_mc2` 只决定走哪个 if 分支，两个分支最终都进 `DeviceOperator.npu_grouped_matmul_swiglu_quant` 融合算子）。

**Combine** —— `DeviceOperator.npu_moe_token_unpermute`(token_dispatcher.py:428,bf16 反置换 + topk 加权）→ prepare_finalize 的 reduce。

```
AllGather 路径量化时机总结:
  bf16 hidden ──allgather(bf16)──> npu_moe_init_routing(quant_mode=-1,只排序)
      ──> npu_dynamic_mx_quant(fp4,在线量化)
      ──> npu_grouped_matmul_swiglu_quant_v2(融合)──> npu_grouped_matmul(GMM2)
      ──> npu_moe_token_unpermute
```

#### 4.4.4 与原生 FP8(W4A8 MXFP MoE）的算子差异

原生 FP8 ckpt 的 MoE 是 `QuantType.W4A8MXFP`(fp4 权重 + **fp8 激活**)，走 `device_op.py` 的专门分支：

- GMM1:`torch_npu.npu_grouped_matmul`(`x_dtype=fp8, weight_dtype=fp4, antiquant_scale, per_token_scale=e8m0`)+ `torch.ops._C_ascend.npu_swiglu_group_quant(dst_type=fp8, quant_mode=2, clamp_value=swiglu_limit)`——GMM 和 SwiGLU **分两个算子**
- GMM2:`torch_npu.npu_grouped_matmul`(`scale=None, antiquant_scale=[w2_scale]`)

即 W4A8 的 GMM1 是"fp8×fp4 GMM + 独立 swiglu_group_quant"两步，而 W4A4 是 `npu_grouped_matmul_swiglu_quant_v2` 一步融合（fp4 进 fp4 出）。

## 第 5 步：图编译期的融合（图模式下)

`vllm_ascend/compilation/passes/norm_quant_fusion_pass.py` 的 MX 融合 pattern **硬编码 fp8**:

```python
class AddRMSNormDynamicMXQuantPattern(BasePattern):
    def get_pattern(self):
        def pattern(rms_norm_input, residual, rms_norm_weight):
            output = torch.ops.npu.npu_add_rms_norm(rms_norm_input, residual, rms_norm_weight, self.eps)
            out0 = output[0]
            out1 = output[2]
            quantized_output = torch.ops.npu.npu_dynamic_mx_quant(out0, dst_type=torch.float8_e4m3fn)
            return quantized_output[0], quantized_output[1], out1
        return pattern
    # replacement → npu_add_rms_norm_dynamic_mx_quant(..., dst_type=torch.float8_e4m3fn)
```

W4A4 的线性层输入端量化是 `dst_type=float4_e2m1fn_x2, round_mode="round"` → **不命中融合** → RMSNorm 与 fp4 量化以两个独立 kernel 执行（每层 2 处：attn_norm/ffn_norm)。这是与原生 FP8 流程在图优化层面的实质差异。

## 不量化的部分

`ffn.gate`（路由器，BF16)、`compressor.wkv/wgate/norm/ape`、`indexer.weights_proj/compressor`、所有 RMSNorm 权重、`embed/head`(BF16)、`hc_*`(F32)、`attn_sink`(F32)——描述文件中均为 FLOAT，走 `AscendUnquantizedLinearMethod` / 原生 module。

## 附：与原生 FP8 ckpt 路径的对照

| | 原生 FP8(`/mnt/share/weight/DeepSeek-V4-Flash`) | W4A4(本目录) |
|---|---|---|
| 探测依据 | config.json `quantization_config.quant_method=fp8` | quant_model_description.json 存在 |
| 配置类 | `AscendFp8Config`(fp8_config.py) | `AscendModelSlimConfig` |
| attn 线性层 | `("FP8","ds_linear")` → `AscendW8A8MXFP8DSDynamicLinearMethod`（含 wo_a reshape + fp32→e8m0 指数提取） | W4A4_MXFP4 / wo_a 走特判 scheme |
| MoE | `("FP8","w4a8_moe")` → fp4 权重 + **fp8 激活** | `("W4A4_MXFP4","moe")` → fp4 权重 + **fp4 激活** |
| scale 源格式 | fp32 的 128×128 block scale(需展开成 per-32 e8m0) | 已是 per-32 e8m0 字节 |
| norm+quant 融合 | 命中（fp8 pattern) | 不命中（fp4) |

---

## 附录：关键量化源码摘录（原样，未经改写）

### vllm_ascend/quantization/utils.py (探测)

```python
    from vllm_ascend.quantization.modelslim_config import MODELSLIM_CONFIG_FILENAME

    # Case 1: ModelSlim — look for quant_model_description.json
    modelslim_path = get_model_file(model, MODELSLIM_CONFIG_FILENAME, revision=revision)
    if modelslim_path is not None:
        return ASCEND_QUANTIZATION_METHOD
```

### modelslim_config.py get_quant_method LinearBase+wo_a

```python
        if isinstance(layer, LinearBase):
            if self.is_layer_skipped_ascend(prefix, self.packed_modules_mapping):
                # Delayed import to avoid circular import
                from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod

                logger.debug("Select AscendUnquantizedLinearMethod for %s (layer=%s)", prefix, "LinearBase")
                return AscendUnquantizedLinearMethod()
            # DeepSeek-V4's DSA attention consumes wo_a.weight directly via
            # npu_transpose_quant_batchmatmul, which requires a grouped 3D
            # [n_local_groups, K, o_lora_rank] layout. The generic W8A8_MXFP8
            # scheme keeps the weight 2D, so use the wo_a-aware scheme instead.
            if (
                model_type == "deepseek_v4"
                and prefix.endswith("wo_a")
                and get_linear_quant_type(self.quant_description, prefix, self.packed_modules_mapping)
                == "W8A8_MXFP8"
            ):
                from .methods.w8a8_mxfp8 import AscendW8A8MXFP8DSWoADynamicLinearMethod

                logger.debug("Select AscendW8A8MXFP8DSWoADynamicLinearMethod for %s (layer=%s)", prefix, "LinearBase")
                return AscendLinearMethod(AscendW8A8MXFP8DSWoADynamicLinearMethod())
            scheme = create_scheme_for_layer(self.quant_description, prefix, "linear", self.packed_modules_mapping)
            logger.debug("Select AscendLinearMethod for %s (layer=%s)", prefix, "LinearBase")
            return AscendLinearMethod(scheme)
        elif isinstance(layer, AttentionLayerBase) and (
            self.is_fa_quant_layer(prefix) or self.is_indexer_quant_layer(prefix)
        ):
```

### w4a4_mxfp4.py linear get_weight/get_pergroup/apply

```python

    def get_weight(self, input_size: int, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        params_dict = {"weight": torch.empty(output_size, input_size // 2, dtype=torch.uint8)}
        return params_dict

    def get_pergroup_param(
        self, input_size: int, output_size: int, params_dtype: torch.dtype, layer_type: str | None = None
    ) -> dict[str, Any]:
        params_dict = {}
        params_dict["weight_scale"] = torch.empty(output_size, input_size // self.group_size, dtype=torch.uint8)
        return params_dict

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        tp_rank: int | None = 0,
    ) -> torch.Tensor:
        # reshape x for Qwen VL models
        original_shape = x.shape
        if x.dim() > 2:
            x = x.view(-1, x.shape[-1])
        quantized_x, dynamic_scale = torch_npu.npu_dynamic_mx_quant(
            x, dst_type=torch_npu.float4_e2m1fn_x2, round_mode="round"
        )
        pertoken_scale = dynamic_scale
        output_dtype = x.dtype
        if bias is not None and bias.dtype != torch.float32:
            bias = bias.to(torch.float32)

        output = torch_npu.npu_quant_matmul(
            quantized_x,
            layer.weight,
            layer.weight_scale,
            scale_dtype=FLOAT8_E8M0FNU_DTYPE,
            pertoken_scale=pertoken_scale,
            pertoken_scale_dtype=FLOAT8_E8M0FNU_DTYPE,
            bias=bias,
            output_dtype=output_dtype,
            x1_dtype=torch_npu.float4_e2m1fn_x2,
            x2_dtype=torch_npu.float4_e2m1fn_x2,
            group_sizes=[1, 1, self.group_size],
        )
        # reshape output for Qwen VL models
        if len(original_shape) > 2:
            output = output.view(*original_shape[:-1], -1)

        return output

    def process_weights_after_loading(self, layer):
        """Process weights after loading for MXFP4 inference.

        This method transforms weights for NPU MXFP4 computation:
        - weight: (output_size, input_size) -> (input_size, output_size)
        - weight_scale: (n_dim, k_dim) -> (k_dim//2, n_dim, 2)
        """

        n_dim, k_dim = layer.weight_scale.data.shape
        layer.weight_scale.data = layer.weight_scale.data.reshape(n_dim, k_dim // 2, 2)
```

### w4a4_mxfp4.py MoE process_weights_after_loading

```python
        )

    def process_weights_after_loading(self, layer):
        g_num, n_size, k_size = layer.w13_weight_scale.shape
        layer.w13_weight_scale.data = layer.w13_weight_scale.data.reshape(g_num, n_size, k_size // 2, 2)
        g_num, n_size, k_size = layer.w2_weight_scale.shape
        layer.w2_weight_scale.data = layer.w2_weight_scale.data.reshape(g_num, n_size, k_size // 2, 2)
        # The A5 MXFP4 fused grouped-matmul-swiglu op relies on the
        # transpose stride to interpret packed FP4 weights as logical K.
        layer.w13_weight.data = layer.w13_weight.data.transpose(1, 2)
        layer.w2_weight.data = layer.w2_weight.data.transpose(1, 2)
        layer.w13_weight_scale.data = layer.w13_weight_scale.data.transpose(1, 2)
        layer.w2_weight_scale.data = layer.w2_weight_scale.data.transpose(1, 2)
```

### w8a8_mxfp8.py DSWoA 类全文

```python
class AscendW8A8MXFP8DSWoADynamicLinearMethod(AscendW8A8MXFP8DynamicLinearMethod):
    """W8A8_MXFP8 linear method for DeepSeek-V4 ``wo_a`` loaded via ModelSlim.

    The DSA attention impl consumes ``wo_a.weight`` directly through
    ``npu_transpose_quant_batchmatmul``, which requires a grouped 3D weight
    ``[n_local_groups, K, o_lora_rank]`` and a matching E8M0 scale
    ``[n_local_groups, K // 64, o_lora_rank, 2]``. ModelSlim checkpoints store
    the weight as 2D ``[n_groups * o_lora_rank, K]`` with per-32 E8M0 scales
    ``[n_groups * o_lora_rank, K // 32]`` (already uint8, unlike the FP8
    checkpoint's fp32 128x128 block scales), so on top of the generic MXFP8
    post-processing this scheme only adds the grouped reshape.
    """

    def __init__(self):
        super().__init__()
        vllm_config = get_current_vllm_config()
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        hf_config = vllm_config.model_config.hf_config
        self.n_local_groups = hf_config.o_groups // tp_size
        self.o_lora_rank = hf_config.o_lora_rank

    def process_weights_after_loading(self, layer):
        # Generic 2D transforms: weight -> (K, G*R), scale -> (K//64, G*R, 2)
        super().process_weights_after_loading(layer)
        # weight: (K, G*R) -> (G, R, K) -> (G, K, R)
        layer.weight.data = (
            layer.weight.data.T.reshape(self.n_local_groups, self.o_lora_rank, -1).transpose(1, 2).contiguous()
        )
        # scale: (K//64, G*R, 2) -> (G, R, K//64, 2) -> (G, K//64, R, 2)
        layer.weight_scale.data = (
            layer.weight_scale.data.transpose(0, 1)
            .reshape(self.n_local_groups, self.o_lora_rank, -1, 2)
            .transpose(1, 2)
            .contiguous()
        )


@register_scheme("W8A8_MXFP8", "moe")
```

### token_dispatcher.py MC2 dispatch quant 判定

```python
        comm_quant_mode = token_dispatch_input.quant.comm_quant_mode

        assert expert_map is not None, "expert_map is required for MC2 token dispatch."
        # NOTE: quant_mode differs by quant feature:
        # - Legacy int communication quantization uses quant_mode=2.
        # - A5 MXFP communication uses quant_mode=4.
        if comm_quant_mode is not None:
            quant_mode = comm_quant_mode
        elif token_dispatch_input.quant.dispatch_with_quant:
            quant_mode = 4 if self.a5_need_extra_args and token_dispatch_input.quant.is_mxfp else 2
        else:
            quant_mode = 0
        self.moe_expert_num = len(expert_map) + global_redundant_expert_num
        expert_token_nums_type = _get_expert_token_nums_type(token_dispatch_input)
        kwargs_mc2 = {
            "x": hidden_states,
            "expert_ids": topk_ids,
            "expert_shard_type": 0,
            "shared_expert_rank_num": 0,
            "moe_expert_num": self.moe_expert_num,
            "global_bs": self.global_bs,
            "expert_token_nums_type": expert_token_nums_type,
        }
        if self.global_bs == 0:
            kwargs_mc2["x_active_mask"] = token_dispatch_input.routing.mc2_mask

        stage1_kwargs = {
            "scales": None,
            "quant_mode": quant_mode,
            "group_ep": self.moe_all_to_all_group_name,
            "ep_world_size": self.ep_world_size,
            "ep_rank_id": self.ep_rank_id,
        }
        if self.need_extra_args:
            stage1_kwargs.update(
                {
                    "group_tp": self.moe_all_to_all_group_name,
                    "tp_world_size": 1,
                    "tp_rank_id": 0,
                }
            )
        # Only dispatch-enabled MXFP paths pass y_dtype through MC2.
        if (
            self.a5_need_extra_args
            and (token_dispatch_input.quant.is_mxfp or token_dispatch_input.quant.is_fp8)
            and token_dispatch_input.quant.dispatch_with_quant
        ):
            y_dtype = torch.float8_e4m3fn
            if (
                token_dispatch_input.quant.mxfp is not None
                and token_dispatch_input.quant.mxfp.act_quant_type is not None
            ):
                y_dtype = token_dispatch_input.quant.mxfp.act_quant_type
            stage1_kwargs.update({"tp_world_size": 1, "tp_rank_id": 0, "y_dtype": y_dtype})
        if self.need_expert_scale or self.a5_need_extra_args:
            stage1_kwargs.update(
                {
                    "expert_scales": topk_weights.to(torch.float32),
                }
            )
        if self.need_comm_alg:
            stage1_kwargs.update({"comm_alg": "hierarchy"})

        kwargs_mc2.update(stage1_kwargs)
        return kwargs_mc2
```

### token_dispatcher.py AllGather dispatch MXFP4 分支

```python
    def token_dispatch(
        self,
        token_dispatch_input: MoETokenDispatchInput,
    ):
        quant_type = token_dispatch_input.quant.quant_type
        dynamic_scale = token_dispatch_input.routing.pertoken_scale
        unquantized_mxfp4_dispatch = quant_type == QuantType.MXFP4 and dynamic_scale is None
        # Without prepare-stage scales, MXFP4 stays unquantized in dispatch and
        # is quantized again inside the MLP path.
        with_quant = token_dispatch_input.quant.dispatch_with_quant and quant_type != QuantType.W8A8FP8
        with_quant = with_quant and not unquantized_mxfp4_dispatch
        is_mxfp = token_dispatch_input.quant.is_mxfp
        hidden_states = token_dispatch_input.hidden_states
        topk_weights = token_dispatch_input.topk_weights
        topk_ids = token_dispatch_input.topk_ids
        expert_map = token_dispatch_input.routing.expert_map
        act_quant_type = (
            token_dispatch_input.quant.mxfp.act_quant_type
            if token_dispatch_input.quant.mxfp is not None and not unquantized_mxfp4_dispatch
            else None
        )
        global_redundant_expert_num = token_dispatch_input.routing.global_redundant_expert_num
        restore_shape = hidden_states.shape
        # Fuse the first dynamic quant of moe_mlp into initrouting when
        # dispatch_with_quant is on but got a None dynamic_scale.
        if with_quant and dynamic_scale is None:
            if quant_type == QuantType.MXFP4:
                quant_mode = 9
            else:
                quant_mode = 3 if is_mxfp else 1
        else:
            quant_mode = -1
```

### moe_mlp.py quant_apply_mlp 量化入口与融合分支

```python
    use_w4a8_per_channel_gmm_swiglu: bool = False,
) -> torch.Tensor:
    input_hidden_dtype = hidden_states.dtype
    use_gmm_swiglu_quant_fusion = use_mxfp_quant or (fusion and not dynamic_eplb)

    if use_mxfp_quant:
        ensure_mxfp8_moe_available("MXFP MoE MLP path")

        if w1_scale_bias is not None or w2_scale_bias is not None:
            raise NotImplementedError("MXFP path does not support scale_bias yet.")
        if w1_offset is not None or w2_offset is not None:
            raise NotImplementedError("MXFP path does not support antiquant offset yet.")

    if w1_offset is not None:
        unquantized_hidden_states = hidden_states
        quantized_hidden_states = None
    elif mxfp_quant_dtype == QuantType.W4A16MXFP4:
        quantized_hidden_states = None
        pertoken_scale = None
    elif dynamic_scale is None:
        unquantized_hidden_states = hidden_states
        hidden_states, pertoken_scale = DeviceOperator.npu_dynamic_quant(
            hidden_states=hidden_states,
            dynamic_scale=None,
            act_quant_type=act_quant_type,
            use_mxfp_quant=use_mxfp_quant,
        )
        dispose_tensor(unquantized_hidden_states)
        quantized_hidden_states = None
    else:
        unquantized_hidden_states = None
        pertoken_scale = (
            DeviceOperator.maybe_normalize_mxfp_scale_layout(dynamic_scale) if use_mxfp_quant else dynamic_scale
        )
        quantized_hidden_states = hidden_states

    bias1, bias2 = None, None
    _output_dtype = w2_scale[0].dtype if isinstance(w2_scale, list) else w2_scale.dtype

    weight_prefetch_method = get_weight_prefetch_method()
    if weight_prefetch_method:
        weight_prefetch_method.maybe_prefetch_moe_weight_postprocess(hidden_states)
    is_mc2 = _EXTRA_CTX.moe_comm_type == MoECommType.MC2
    if w1_scale_bias is None and w1_offset is None and is_mc2:
        if _custom_gmm_swiglu_enabled(fusion, dynamic_eplb) and not use_mxfp_quant:
            # gmm1: gate_up_proj & act_fn: swiglu
            hidden_states, swiglu_out_scale, _ = torch.ops._C_ascend.grouped_matmul_swiglu_quant_weight_nz_tensor_list(
                x=hidden_states,
                weight=w1,
                weight_scale=w1_scale,
                x_scale=pertoken_scale,
                group_list=cumsum_group_list(group_list, group_list_type, 0),
                swiglu_limit=swiglu_limit,
            )
        elif use_gmm_swiglu_quant_fusion:
            # gmm1: gate_up_proj & act_fn: swiglu
            hidden_states, swiglu_out_scale, _ = DeviceOperator.npu_grouped_matmul_swiglu_quant(
                x=hidden_states,
                weight=_require_single_tensor_for_swiglu_quant(w1, name="w1"),
                group_list=cumsum_group_list(group_list, group_list_type, 0),
                weight_scale=_require_single_tensor_for_swiglu_quant(w1_scale, name="w1_scale"),
                x_scale=pertoken_scale,
                bias=None,
                use_mxfp_quant=use_mxfp_quant,
                act_quant_type=act_quant_type,
                weight_quant_type=weight_quant_type,
                swiglu_limit=swiglu_limit,
                mxfp_quant_dtype=mxfp_quant_dtype,
            )
            if quantized_hidden_states is not None:
                dispose_tensor(quantized_hidden_states)
        else:
            if w1_scale[0].dtype != torch.float32:
                w1_scale[0] = w1_scale[0].to(torch.float32)
```

### device_op.py npu_dynamic_quant (A5)

```python
    def npu_dynamic_quant(
        hidden_states: torch.Tensor,
        dynamic_scale: torch.Tensor | None = None,
        *,
        act_quant_type=torch.float8_e4m3fn,
        use_mxfp_quant: bool = False,
    ):
        if not use_mxfp_quant:
            return BaseDeviceAdaptor.npu_dynamic_quant(
                hidden_states,
                dynamic_scale,
                act_quant_type=act_quant_type,
                use_mxfp_quant=False,
            )

        if dynamic_scale is None:
            hidden_states, dynamic_scale = torch_npu.npu_dynamic_mx_quant(hidden_states, dst_type=act_quant_type)

        return hidden_states, A5DeviceAdaptor.maybe_normalize_mxfp_scale_layout(dynamic_scale)

    @staticmethod
```

### device_op.py npu_grouped_matmul_swiglu_quant MXFP else 分支

```python
        else:
            out, out_scale = torch_npu.npu_grouped_matmul_swiglu_quant_v2(
                x=x,
                weight=[weight],
                group_list=group_list,
                weight_scale=[weight_scale],
                x_scale=x_scale,
                dequant_mode=2,
                quant_mode=2,
                dequant_dtype=torch.float32,
                quant_dtype=act_quant_type,
                x_dtype=act_quant_type if act_quant_type in QUANT_DTYPES else None,
                weight_dtype=weight_quant_type if weight_quant_type in QUANT_DTYPES else None,
                weight_scale_dtype=FLOAT8_E8M0FNU_DTYPE,
                x_scale_dtype=FLOAT8_E8M0FNU_DTYPE,
            )
        return out, A5DeviceAdaptor.maybe_normalize_mxfp_scale_layout(out_scale), None
```

### device_op.py npu_grouped_matmul_gmm2 MXFP4 尾部

```python
        if mxfp_quant_dtype == QuantType.W4A8MXFP:
            gmm2_scale = None  # type: ignore[assignment]
            gmm2_kwargs.update({"antiquant_scale": [weight_scale]})

        return torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=gmm2_weight,
            scale=gmm2_scale,
            bias=bias,
            per_token_scale=[per_token_scale],
            split_item=2,
            group_list_type=group_list_type,
            group_type=0,
            group_list=group_list,
            output_dtype=output_dtype,
            **gmm2_kwargs,
        )[0]

    @staticmethod
    def kv_cache_load(cache_kv_c, cache_k_pe, block_table, context_seq_len_npu, seq_offset, key, value):
```

### norm_quant_fusion_pass.py fp8-only pattern

```python

    def get_pattern(self):
        def pattern(rms_norm_input: torch.Tensor, residual: torch.Tensor, rms_norm_weight: torch.Tensor):
            """
            Pattern for AddRMSNormDynamicMXQuant fusion.
            """
            output = torch.ops.npu.npu_add_rms_norm(rms_norm_input, residual, rms_norm_weight, self.eps)
            out0 = output[0]
            out1 = output[2]
            quantized_output = torch.ops.npu.npu_dynamic_mx_quant(out0, dst_type=torch.float8_e4m3fn)
            return quantized_output[0], quantized_output[1], out1

        return pattern

    def get_replacement(self):
        def replacement(rms_norm_input: torch.Tensor, residual: torch.Tensor, rms_norm_weight: torch.Tensor):
            """
            Replacement for the AddRMSNormDynamicMXQuant fusion.
```

