# VQ2、MXFP4 混合专家权重导出规范

整理日期：2026-10-13；修订日期：2026-11-07。

当前主线以 `DeepSeek-V4-Flash-0731-32x32-noperm-hot5050125-identity-v3` 为权重基础。
第 1～6 节定义自然输入列顺序的 routed experts 混精格式；第 7 节规定非 routed 部分使用
原始 `DeepSeek-V4-Flash-0731` 的字段名称，dtype 和 shape 沿用已核对的
`DeepSeek-V4-Flash-0731-vq2a8-mixed-gs32` 格式，与原始 0731 对应张量一致。
现有 `identity-v3` 的非 routed 部分尚需按第 7.7 节转换，不能将其当前 BF16 矩阵格式当作目标规范。
本规范不再支持权重输入列置换；有效权重中出现 `.perm`（包括恒等排列）即为不支持的格式。

## 1. 专家组织与命名

统一名称：

```text
layers.{L}.ffn.experts.{E}.{P}.{参数}
```

| 项目 | 约定 |
| --- | --- |
| `L` | 全局层号，从 0 开始 |
| `E` | 该层的全局专家编号，从 0 开始 |
| `P` | `w13` 表示 GateUp，`w2` 表示 Down |
| 参数 | VQ2 三字段或 MXFP4 两字段，见下文 |
| 名称前缀 | 固定为 `layers.`，不加 `model.` 前缀 |
| Tensor 范围 | 一个专家的一个投影参数，不包含专家批次维度 |

专家 ID 直接写入 Tensor 名称，不依赖 `experts.json` 中的列表位置识别专家。
多个专家的 Tensor 可以保存在同一个 safetensors 分片中，无须为每个专家建立一个文件。

同一 `(L,E,P)` 交付一套完整字段。VQ2 只保存以下三个 Tensor：

```text
layers.{L}.ffn.experts.{E}.{P}.packed_indices
layers.{L}.ffn.experts.{E}.{P}.codebooks
layers.{L}.ffn.experts.{E}.{P}.out_dim_scale
```

MXFP4 保存以下两个 Tensor，不同时保留该投影的 VQ2 字段：

```text
layers.{L}.ffn.experts.{E}.{P}.weight
layers.{L}.ffn.experts.{E}.{P}.scale
```

每个专家的 GateUp 与 Down 独立选择量化格式。因此每个专家可能有 4、5 或 6 个 Tensor：
两个投影均为 MXFP4 时为 4 个，混合格式时为 5 个，均为 VQ2 时为 6 个。

以下给出导出后的完整命名示例，这些名称应同时用于 safetensors 文件头和 `model.safetensors.index.json` 的 `weight_map` 键。

**示例 1：第 42 层专家 248，GateUp 和 Down 均为 VQ2，共 6 个 Tensor。**

```text
layers.42.ffn.experts.248.w13.packed_indices
layers.42.ffn.experts.248.w13.codebooks
layers.42.ffn.experts.248.w13.out_dim_scale
layers.42.ffn.experts.248.w2.packed_indices
layers.42.ffn.experts.248.w2.codebooks
layers.42.ffn.experts.248.w2.out_dim_scale
```

**示例 2：第 42 层专家 250，GateUp 和 Down 均为 MXFP4，共 4 个 Tensor。**

```text
layers.42.ffn.experts.250.w13.weight
layers.42.ffn.experts.250.w13.scale
layers.42.ffn.experts.250.w2.weight
layers.42.ffn.experts.250.w2.scale
```

**示例 3：第 0 层专家 4，GateUp 为 VQ2、Down 为 MXFP4，共 5 个 Tensor。**

```text
layers.0.ffn.experts.4.w13.packed_indices
layers.0.ffn.experts.4.w13.codebooks
layers.0.ffn.experts.4.w13.out_dim_scale
layers.0.ffn.experts.4.w2.weight
layers.0.ffn.experts.4.w2.scale
```

本模型为 43 层、每层 256 个 routed experts；这里不包含 shared experts。
`hidden_size=4096`，`moe_intermediate_size=2048`。

对每次矩阵乘法，M 是当前专家接收的 token 行数，K 是输入特征维度，N 是输出特征维度。
逻辑权重矩阵按 `[输出行,输入列]`，即 `[N,K]` 定义；实际压缩 Tensor 的存储布局见第 2、3 节。逻辑计算为：

```text
输入 X[M,K] @ 权重 W[N,K].T = 输出 Y[M,N]
```

权重逻辑矩阵的第 `j` 列直接对应该投影输入激活 `X` 的第 `j` 个特征。
导出时，压缩索引、码本及缩放参数必须共同描述这一列顺序；VQ2 和 MXFP4 均遵循此约定。

| 逻辑投影 | 输入 shape `[M,K]` | 权重存储 shape `[N,K]` | 输出 shape `[M,N]` |
| --- | --- | --- | --- |
| Gate / w1 | `[M,4096]` | `[2048,4096]` | `[M,2048]` |
| Up / w3 | `[M,4096]` | `[2048,4096]` | `[M,2048]` |
| GateUp / w13 | `[M,4096]` | `[4096,4096]` | `[M,4096]`，含 Gate、Up 各 2048 列 |
| Down / w2 | `[M,2048]` | `[4096,2048]` | `[M,4096]` |

### 1.1 GateUp 合并与算子布局

交付文件保留合并的 `w13`：

逻辑矩阵为 `[N,K]`，输出行使用 **全部 Gate 在前、全部 Up 在后** 的自然顺序。
该约定对应 `row_order=natural_gate_then_up`，同时适用于 VQ2 和 MXFP4 的 `w13`。

当前 Gate、Up 各有 2048 个输出通道，`w13` 的第 0～2047 行为 Gate，第 2048～4095 行为 Up。

## 2. 参数 dtype 与 shape

本文统一按每次投影的 GEMM 维度定义：

- M：当前专家接收的 token 行数，不属于权重 Tensor 的维度。
- K：当前投影的输入特征维度。
- N：当前投影的输出特征维度。

每个 Tensor 仅保存名称中指定的单个专家参数，以下 shape 均不含专家批次维度。

**K、N 随投影变化**：GateUp 为 `K=4096、N=4096`；Down 为 `K=2048、N=4096`。
下表对 GateUp、Down 都适用，分别代入各自的 K、N：

| 格式 | 参数 | dtype | 完整 shape |
| --- | --- | --- | --- |
| VQ2 | `packed_indices` | U8 | `[N/32,K/16,16,8]` |
| VQ2 | `codebooks` | FP8 E4M3FN | `[N/32,K/32,16,2]` |
| VQ2 | `out_dim_scale` | U64 | `[N]` |
| MXFP4 | `weight` | U8 | `[N,K/2]` |
| MXFP4 | `scale` | FP8 E8M0 | `[N,K/32]` |

当前模型的各投影参数 shape 为：

| 参数 | GateUp | Down |
| --- | --- | --- |
| VQ2 `packed_indices` | `[128,256,16,8]` | `[128,128,16,8]` |
| VQ2 `codebooks` | `[128,128,16,2]` | `[128,64,16,2]` |
| VQ2 `out_dim_scale` | `[4096]` | `[4096]` |
| MXFP4 `weight` | `[4096,2048]` | `[4096,1024]` |
| MXFP4 `scale` | `[4096,128]` | `[4096,64]` |

## 3. VQ2 三个参数的精确定义

对单个专家、单个投影，逻辑权重矩阵为 `[N,K]`，N 为输出行数，K 为输入列数。
三个字段通过共同名称前缀 `layers.{L}.ffn.experts.{E}.{P}.` 关联。

### 3.1 packed_indices

`packed_indices` 保存码本编号。**一个 U8 字节存两个 4-bit 索引，每个索引查出两个 FP8 值，共恢复四个值。**

交付 shape 固定为 `[N/32,K/16,16,8]`，dtype 为 U8，连续存储且最后一维变化最快。
四个轴依次为输出行块、输入列存储块、块内输入列、该列的打包字节。

- **读取顺序**：先低 4 bit，再高 4 bit；编号范围均为 `0…15`，查询当前 32×32 块的码本。
- **权重位置**：低位索引对应前两行，高位索引对应后两行；四个值位于**同一输入列、连续四个输出行**，即 `4×1` 竖条。
- **量化分组**：K16 只是存储分块，码本仍按 K32、N32 分组；相邻两个 K16 存储块使用同一个 K32 码本，不改变 A8 的 K32 分组约定。
- **输出行序**：使用自然输出行序，GateUp 的全部 Gate 行在前、全部 Up 行在后，不插入 Gate32/Up32 交错。

令 `n32`、`k16` 为输出行块和输入列存储块编号，`ki=0…15`，`b=0…7`，精确解码关系为：

```text
T = packed_indices[n32,k16,ki,b]
j = 16*k16 + ki
r = 32*n32 + 4*b

q_low  = T & 0x0F
q_high = T >> 4
V[r:r+2, j]   = codebooks[n32,k16//2,q_low, :]
V[r+2:r+4, j] = codebooks[n32,k16//2,q_high, :]
```

其中 `j` 是逻辑输入列编号，直接对应输入激活 `X[:,j]`。文件字段名保持 `packed_indices`；加载器可将其映射到内部 `packed_zn`，无需再次重排索引字节，也无须在文件中保存两份数据。

查表后，各行再乘从 `out_dim_scale` 解码出的 FP32 scale，即得到同一列顺序下的逻辑权重，详见第 3.3 节。

示例：一个字节为 `0xA3`，低位编号为 3，高位编号为 10。
假设码本第 3 项为 `[1.0,-0.5]`，第 10 项为 `[2.0,-1.5]`：

```text
0xA3 → 先查编号 3，再查编号 10

             同一个输入列 k
                    ↓
输出行 n           1.0   ← 低位编号 3，分量 0
输出行 n+1        -0.5   ← 低位编号 3，分量 1
输出行 n+2         2.0   ← 高位编号10，分量 0
输出行 n+3        -1.5   ← 高位编号10，分量 1

以上为查表得到的 FP8 值，尚未应用行缩放系数。
```

### 3.2 codebooks

维度顺序固定为 `[N/32,K/32,16,2]`，即输出行 tile、输入列 tile、码字、两个分量。
交付 dtype 为 FP8 E4M3FN，连续存储且最后一维变化最快；保留原始位模式，不转成整数数值。
同一 `(n32,k32)` 的 16 个码字适用于该 32×32 块，每个码字包含相邻两行的两个值。

加载器可对交付码本做字节视图并合并最后两个轴，得到 U8 `pair_lut`，不需要转置或重新打包：

```python
import torch

pair_lut = codebooks.view(torch.uint8).reshape(N // 32, K // 32, 32)
```

`pair_lut[n32,k32,2*q+d]` 对应 `codebooks[n32,k32,q,d]` 的原始 FP8 字节，其中 `d=0,1`。
这里必须使用 dtype 视图，不能用 `.to(torch.uint8)` 做数值转换。
GateUp 的两个 tile 轴长度恰好都是 128，交换轴前后 shape 相同，但字节顺序不同；仍须按本节的 N tile 在前布局导出。

### 3.3 out_dim_scale

每个输出行一个缩放系数，交付 dtype 为 U64，shape 为 `[N]`。
每个 U64 的低 32 位保存 FP32 scale 的原始位模式，高 32 位必须为零；解码后的 scale 必须有限且严格大于 0。
例如 `1.0` 保存为 `0x000000003F800000`。

令 `scale[r]` 为第 `r` 个元素解码后的 FP32 数值，结合第 3.1 节得到的 `V`，逻辑权重为：

```text
W[r,j] = float32(V[r,j]) * scale[r]
```

## 4. MXFP4 的格式约定

保存 `weight` 和 `scale` 两个字段，后者直接以 FP8 E8M0 保存缩放系数：

```text
weight：每字节两个 E2M1 编码，低半字节对应前一个元素。
scale ：FP8 E8M0 缩放系数，每行沿 K 方向每 32 个权重共享一个。
```

`weight` 保存的是 U8 `[N,K/2]` 的打包 FP4 码字，不是 `[N,K]` 的稠密浮点权重。
本节的 `scale` 是 MXFP4 的独立 Tensor；第 3.3 节公式中的局部变量 `scale` 则由 VQ2 的 `out_dim_scale` 解码得到，两者不能混用。

`scale` 的 safetensors dtype 为 `F8_E8M0`，对应 `float8_e8m0fnu`。
它没有符号位和尾数位。设一个元素的原始字节为 `b`，有效范围为 `0…254`，其浮点值为 `2**(int(b)-127)`。
`0x7F` 表示 `1.0`，`0x00` 表示 `2**(-127)` 而不是零；`0xFF` 表示 NaN，不得出现在交付的 scale 中。

约束：

- 分组沿输入列 K 维进行，不跨行。
- GateUp 合并时，分别沿输出行拼接 Gate、Up 的 `weight` 和 `scale`。
- 每个 scale 必须有限且严格大于零，表示范围为 `2**(-127)…2**127`。
- 交付的是 E8M0 缩放系数，不是未加偏置的整数指数；推理时直接使用其缩放数值，不再对其执行 `2**scale` 或重复加偏置。
- 若消费者需要原始字节，使用 U8 字节视图，不做整数数值转换；文件仍以 `F8_E8M0` dtype 保存。
- MXFP4 投影只保存 `weight` 和 `scale`，按第 1 节定义的输入列顺序恢复权重。

## 5. 专家归属、混合精度与分片索引

### 5.1 每个投影的字段完整性

加载器通过名称中的 `(L,E,P)` 关联同一投影的参数，再根据完整字段集合识别 VQ2 或 MXFP4。
无需额外的列表位置映射；可选的专家清单只用于摘要或完整性检查，不改变 Tensor 名称中的专家身份。

字段组合示例（专家 ID 仅用于说明，不规定各层的混精分配）：

| 层号 | 专家 ID | GateUp / w13 | Down / w2 |
| --- | --- | --- | --- |
| 0 | 4 | VQ2 | MXFP4 |
| 42 | 248 | VQ2 | VQ2 |
| 42 | 250 | MXFP4 | MXFP4 |

示例中的 MXFP4 投影只保存两字段，不额外生成 `packed_indices`。实际精度由每个投影的完整字段集合识别。

必须满足：

- 全局层号为 `0…42`，每层专家 ID 为 `0…255`，每个专家都有 `w13` 和 `w2`。
- 同一 `(L,E,P)` 恰好具有 VQ2 的 `packed_indices/codebooks/out_dim_scale` 或 MXFP4 的 `weight/scale`，禁止缺字段、混合两套字段或保留被回退覆盖的 VQ 副本。
- 每个字段的 dtype 和 shape 必须符合第 2 节，禁止保留专家批次维度。
- 不交付 `.perm`，也不补 identity 张量；框架明确拒绝此旧字段，不能仅删除旧字段来声称原有权重符合自然列序。
- GateUp 与 Down 的量化类型独立判断，不能根据一个投影的格式推断另一个投影。
- 码本及缩放参数必须与名称中同一专家、同一投影的压缩权重对应，且输入列顺序符合第 1 节。
- 交付完整全局权重，不按 EP rank 分包；运行时可以按全局专家 ID 选择本 rank 需要的权重。

### 5.2 model.safetensors.index.json

根目录的 `model.safetensors.index.json` 必须在 `weight_map` 中列出全部专家 Tensor，以及模型的非 routed 权重。
每个条目表示“真实 Tensor 名称 → 相对模型根目录的分片路径”。多个专家可以指向同一个分片。

以下仅展示第 42 层两个专家的 Down 参数，不是完整模型索引：

```json
{
  "weight_map": {
    "layers.42.ffn.experts.248.w2.packed_indices": "layer_42.safetensors",
    "layers.42.ffn.experts.248.w2.codebooks": "layer_42.safetensors",
    "layers.42.ffn.experts.248.w2.out_dim_scale": "layer_42.safetensors",
    "layers.42.ffn.experts.250.w2.weight": "layer_42.safetensors",
    "layers.42.ffn.experts.250.w2.scale": "layer_42.safetensors"
  }
}
```

分片文件头必须实际存在上述名称及其单专家 Tensor，不能只修改索引，给旧堆叠 Tensor 的切片添加别名。
索引中的名称与分片文件头必须一一对应，每个参数只出现一次，不漏列、不重复，也不依赖文件中的排列顺序。
分片可以按层或大小划分，文件名不承担专家编号映射职责。

## 6. 格式版本与推理框架衔接

本次修订用于后续权重导出，字段集合以本文为准。消费这些权重的加载器需按三字段 VQ2、两字段 MXFP4 的约定适配并验证。

保留格式标识 `vq2_mixed_gs32_source`，将四维索引、N tile 在前的码本布局及使用 `weight/scale` 命名、E8M0 缩放的 MXFP4 格式标记为版本 3，以区别此前的源格式。`config.json` 的量化配置片段为：

```json
{
  "quantization_config": {
    "quant_method": "vq2a8",
    "format": "vq2_mixed_gs32_source",
    "format_version": 3
  }
}
```

## 7. 非 routed 权重导出格式与 MTP 边界

### 7.1 格式依据、范围和变换语义

本节的字段名称以原始 `DeepSeek-V4-Flash-0731` 的
revision `7872f01b1d1fe23eabc4c98b48bffcef5a386062` 为准，见[原始权重格式记录](deepseek_v4_flash_0731_reference.json)。
已于 2026-11-07 将该记录与本地 `D:\models\DeepSeek-V4-Flash-0731-vq2a8-mixed-gs32` 的索引、
`quant.json` 及全部 13 个 `rest` 分片文件头对照：名称映射后，主模型非 routed 的 50 类布局、
1,564 个 Tensor 的 dtype、shape 和数量全部一致，载荷约 8.24 GiB，分类如下：

| safetensors dtype | Tensor 数 | 用途 |
| --- | ---: | --- |
| `F8_E4M3` | 365 | Attention、共享专家及 indexer Q-B 的权重 |
| `F8_E8M0` | 365 | 与上述权重配对的 block128 scale |
| `BF16` | 425 | Norm、embedding、head、compressor、router 等 |
| `F32` | 406 | HC、sink、位置参数、路由修正 bias |
| `I64` | 3 | 前三层的 hash 路由表 |

以上数量对应本文 43 层模型的完整非 routed 交付；13 个分片是参考实物的组织方式，不要求固定分片数或文件名。
非 routed 层内名称统一使用原始 `layers.{L}.attn.*`、`layers.{L}.ffn.*` 等形式，
根级使用 `embed.weight`、`head.weight`、`norm.weight`、`hc_head_fn` 等名称，均不加 `model.` 前缀。
`self_attn.*`、`mlp.*`、`gate_proj` 等框架别名只在加载时映射，不作为本规范的交付名称。
routed experts 继续按第 1 节使用 `layers.*.ffn.experts.*` 的独立专家字段，
不沿用旧 source-v1 的堆叠格式、`.perm` 或量化配置。

非 routed 权重可以经过离线旋转、平滑或 Norm 融合，按实际导出值加载，不要求与原始 0731 逐字节相同。
当前运行路径不额外插入旋转、逆旋转或权重重算；导出方应说明所有离线变换及推理所需算子，
确保交付与这一执行路径一致。目录名 `identity`、全一 Norm 或不存在旋转 sidecar 都不能单独证明这些语义。
保存的 Norm.gamma 为 1 时，仍执行 RMS 归一化，不再次融合 gamma。
模型原有 RoPE 和 indexer Hadamard 等计算保留，不能把它们当作旧权重 `.perm` 一并删除。

如需确认变换来源，使用 [非 routed 变换取证](non_routed_transform_inspect.md)，并提供导出端的计算公式或数值对照。
名称和 dtype 对齐不代表与原始 0731 的权重数值相同；已应用的离线变换仍须单独声明。
文件头检查不代表已完成昇腾整模加载、图捕获或精度验收。

### 7.2 Attention 与共享专家的 FP8 block128

下表中的每个模块必须同时保存 `.weight` 和 `.scale`：

| 字段 | safetensors dtype | shape | 含义 |
| --- | --- | --- | --- |
| `weight` | `F8_E4M3`（E4M3FN） | `[N,K]` | 按逻辑输出行、输入列保存的 FP8 权重 |
| `scale` | `F8_E8M0` | `[ceil(N/128),ceil(K/128)]` | 每个 128×128 权重块的反量化乘数 |

逻辑解码为：

```text
W[n,k] = float32(weight[n,k]) * float32(scale[n//128,k//128])
```

`scale` 使用第 4 节描述的 E8M0 数值编码，必须有限且严格大于零，不保存未加偏置的整数指数。
这里的二维 block128 scale 不同于 routed MXFP4 的每输出行 K32 scale，不能混用 shape 或分组规则。
`weight` 必须为有限 E4M3FN 值；矩阵按连续 `[N,K]` 保存，不提前转置为 NPU 内部布局。

以下名称省略 `layers.{L}.` 前缀，`L=0…42`；表中名称后分别追加 `.weight` 和 `.scale`：

| 模块 | `weight` shape | `scale` shape | 存在范围 |
| --- | --- | --- | --- |
| `attn.wq_a` | `[1024,4096]` | `[8,32]` | 全部 43 层 |
| `attn.wq_b` | `[32768,1024]` | `[256,8]` | 全部 43 层 |
| `attn.wkv` | `[512,4096]` | `[4,32]` | 全部 43 层 |
| `attn.wo_a` | `[8192,4096]` | `[64,32]` | 全部 43 层 |
| `attn.wo_b` | `[4096,8192]` | `[32,64]` | 全部 43 层 |
| `attn.indexer.wq_b` | `[8192,1024]` | `[64,8]` | ratio=4，21 层 |
| `ffn.shared_experts.w1` | `[2048,4096]` | `[16,32]` | 全部 43 层 |
| `ffn.shared_experts.w3` | `[2048,4096]` | `[16,32]` | 全部 43 层 |
| `ffn.shared_experts.w2` | `[4096,2048]` | `[32,16]` | 全部 43 层 |

共享专家 Gate、Up、Down 分别命名为 `w1`、`w3`、`w2`；Gate、Up 分别保存，由已有加载器合并，
不套用 routed `w13` 的 VQ2/MXFP4 编码。
本节共 365 个 FP8 矩阵及 365 个 scale。加载器按文件 dtype 和配对 scale 选择已有 FP8 block128 路径，
再执行 TP 分片及算子布局转换；导出文件不保存第二份 BF16 副本。

### 7.3 Attention 辅助参数

主模型第 0、1 层压缩比例为 0，第 2～42 层交替为 4、128，最后一层为 4。
以下名称同样省略 `layers.{L}.`：

| Tensor 后缀 | dtype | shape | 存在范围 |
| --- | --- | --- | --- |
| `attn.attn_sink` | F32 | `[64]` | 全部层 |
| `attn.q_norm.weight` | BF16 | `[1024]` | 全部层 |
| `attn.kv_norm.weight` | BF16 | `[512]` | 全部层 |
| `attn.compressor.ape` | F32 | `[4,1024]` | ratio=4，21 层 |
| `attn.compressor.ape` | F32 | `[128,512]` | ratio=128，20 层 |
| `attn.compressor.norm.weight` | BF16 | `[512]` | ratio=4/128 |
| `attn.compressor.wgate.weight`、`attn.compressor.wkv.weight` | BF16 | `[1024,4096]` | ratio=4 |
| `attn.compressor.wgate.weight`、`attn.compressor.wkv.weight` | BF16 | `[512,4096]` | ratio=128 |
| `attn.indexer.compressor.ape` | F32 | `[4,256]` | ratio=4 |
| `attn.indexer.compressor.norm.weight` | BF16 | `[128]` | ratio=4 |
| `attn.indexer.compressor.wgate.weight`、`attn.indexer.compressor.wkv.weight` | BF16 | `[256,4096]` | ratio=4 |
| `attn.indexer.weights_proj.weight` | BF16 | `[64,4096]` | ratio=4 |

本表的 BF16 投影不生成 FP8 `.scale`；indexer 的 `wq_b` 单独遵循第 7.2 节。

### 7.4 Router、层 Norm 与 HC

以下名称省略 `layers.{L}.`：

| Tensor 后缀 | dtype | shape | 存在范围 |
| --- | --- | --- | --- |
| `ffn.gate.weight` | BF16 | `[256,4096]` | 全部层 |
| `ffn.gate.tid2eid` | I64 | `[129280,6]` | 第 0、1、2 层 |
| `ffn.gate.bias` | F32 | `[256]` | 第 3～42 层 |
| `attn_norm.weight`、`ffn_norm.weight` | BF16 | `[4096]` | 全部层 |
| `hc_attn_fn`、`hc_ffn_fn` | F32 | `[24,16384]` | 全部层 |
| `hc_attn_base`、`hc_ffn_base` | F32 | `[24]` | 全部层 |
| `hc_attn_scale`、`hc_ffn_scale` | F32 | `[3]` | 全部层 |

Norm 以 BF16 保存，HC 参数以 F32 保存；HC `*_scale` 是普通浮点参数，不是 FP8 或 MXFP4 的 E8M0 scale。
`tid2eid` 的专家编号对应主模型全局专家 ID，不能根据分片位置或热点排序重编号。

### 7.5 Embedding、最终 Norm 和输出头

以下为完整 Tensor 名称，均不加 `model.` 前缀：

| Tensor 名称 | dtype | shape |
| --- | --- | --- |
| `embed.weight` | BF16 | `[129280,4096]` |
| `head.weight` | BF16 | `[129280,4096]` |
| `norm.weight` | BF16 | `[4096]` |
| `hc_head_fn` | F32 | `[4,16384]` |
| `hc_head_base` | F32 | `[4]` |
| `hc_head_scale` | F32 | `[1]` |

`tie_word_embeddings=false`，embedding 与 head 分别加载。输出保留 `HC 合并 → RMSNorm → head`。
若导出方将最终 Norm.gamma 融合进 head，仍需保存实际 gamma（融合后为全一）并声明变换；不能删除 RMS 归一化操作。

### 7.6 配置、索引与 MTP

目标交付使用第 6 节的三键量化配置，不复制原始 0731 的全局 FP8 配置或旧 mixed-gs32 的 source-v1 配置，
也不附加旧 `packed_weight` 的专用输出头/MTP 配置。
`expert_dtype=fp4` 等架构元数据不覆盖 routed experts 按字段识别的混精类型。
`config.json` 的模型几何、压缩比例、RoPE、tokenizer 和词表必须与实际权重一致。

原始 0731 另含 MTP/DSpark 权重；旧 `mixed-gs32` 与当前 `identity-v3` 均没有 MTP 张量。
本次目标交付不包含 MTP 或 DSpark 草稿包，采用原始名称不表示同时恢复这些权重。
`num_nextn_predict_layers=1` 只是保留的配置字段，
不能据此自动开启 MTP。当前服务和测评命令固定 `--no-dspark`。
未来若单独交付 MTP，必须明确阶段结构、专用 embedding/head、隐藏状态接口及与主模型的坐标一致性，另行适配和验收。

索引完整列出全部主专家和非 routed 张量。不得为满足旧加载逻辑而生成 identity `.perm`，
也不得仅删除旧 `.perm` 或旋转声明来绕过格式校验；已应用的列顺序变换需要在导出端正确处理。
框架兼容其他已支持的无 perm 名称或 dtype，不代表这些兼容格式自动符合本规范的导出要求。

### 7.7 现有 identity-v3 权重的转换边界

现有 `DeepSeek-V4-Flash-0731-32x32-noperm-hot5050125-identity-v3` 的非 routed 部分为 1,199 个 Tensor，
其逻辑 shape 与本节对应，但字段仍使用框架别名，且尚未采用本节要求的 FP8 矩阵和 BF16 Norm 格式。
转换要求如下：

| 项目 | 现有 identity-v3 | 本规范要求 | 转换性质 |
| --- | --- | --- | --- |
| 非 routed 名称 | `model.layers.*.self_attn.*`、`mlp.*` 等框架别名 | 第 7.2～7.5 节原始名称 | 同步修改分片文件头和根索引，不能只改索引 |
| 第 7.2 节的 365 个矩阵 | BF16，无 scale | F8_E4M3 weight + F8_E8M0 block128 scale | 重新量化，须记录误差并验收精度 |
| 全部 235 个 Norm | F32 | BF16 | 转换 dtype，须检查能否精确表示 |
| 其余非 routed 字段 | BF16 / F32 / I64 | 保持本节对应格式 | 按原始名称重命名，保持 shape 和载荷 |
| Routed experts | 自然列序 VQ2/MXFP4 | 第 1～6 节 | 保持原有字段及载荷 |

235 个 Norm 包括每层 4 个 Norm、41 个 compressor Norm、21 个 indexer compressor Norm 和 1 个最终 Norm。
新增 365 个 scale 后，非 routed 总数为 1,564；名称映射不改变矩阵行列顺序，无须转置或合并共享专家 Gate/Up。
层内名称映射示例（左列省略 `model.layers.{L}.`，右列省略 `layers.{L}.`；投影字段 `.weight/.scale` 随模块名映射）：

| 现有名称 | 交付名称 |
| --- | --- |
| `self_attn.q_a_proj`、`self_attn.q_b_proj` | `attn.wq_a`、`attn.wq_b` |
| `self_attn.kv_proj`、`self_attn.o_a_proj`、`self_attn.o_b_proj` | `attn.wkv`、`attn.wo_a`、`attn.wo_b` |
| `self_attn.compressor.indexer.q_b_proj` | `attn.indexer.wq_b` |
| `self_attn.compressor.indexer.scorer.weights_proj` | `attn.indexer.weights_proj` |
| `mlp.shared_experts.gate_proj`、`mlp.shared_experts.up_proj`、`mlp.shared_experts.down_proj` | `ffn.shared_experts.w1`、`ffn.shared_experts.w3`、`ffn.shared_experts.w2` |
| `input_layernorm`、`post_attention_layernorm` | `attn_norm`、`ffn_norm` |
| `attn_hc.fn/base/scale`、`ffn_hc.fn/base/scale` | `hc_attn_fn/base/scale`、`hc_ffn_fn/base/scale`（分别以 `_` 连接字段） |

根级 `model.embed_tokens.weight` 映射为 `embed.weight`，`model.norm.weight` 映射为 `norm.weight`，
`model.hc_head.hc_fn/hc_base/hc_scale` 分别映射为 `hc_head_fn`、`hc_head_base`、`hc_head_scale`；`head.weight` 保持同名。
转换以新版自身权重值为输入，不直接拷贝旧模型的权重或 scale，也不再次旋转、融合 gamma 或补造 MTP。
BF16 转 FP8 包含量化舍入，不能称为纯无损 repack；输出应单独保存，并记录转换方法及误差。
本次修订只定义目标格式，不表示现有权重已经完成转换或昇腾精度验收。

转换脚本、Linux/Windows 命令及结果说明见 [identity-v3 非 routed 权重转换](repack_non_routed.md)。
