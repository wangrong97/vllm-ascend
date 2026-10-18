# VQ2、MXFP4 混合专家权重交付规范建议

整理日期：2026-10-13；修订日期：2026-10-16。

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
| 参数 | VQ2 四字段或 MXFP4 两字段，见下文 |
| 名称前缀 | 固定为 `layers.`，不加 `model.` 前缀 |
| Tensor 范围 | 一个专家的一个投影参数，不包含专家批次维度 |

专家 ID 直接写入 Tensor 名称，不依赖 `experts.json` 中的列表位置识别专家。
多个专家的 Tensor 可以保存在同一个 safetensors 分片中，无须为每个专家建立一个文件。

同一 `(L,E,P)` 交付一套完整字段。VQ2 保存以下四个 Tensor（没有perm的话只保存另外三个Tensor）：

```text
layers.{L}.ffn.experts.{E}.{P}.packed_indices
layers.{L}.ffn.experts.{E}.{P}.codebooks
layers.{L}.ffn.experts.{E}.{P}.perm
layers.{L}.ffn.experts.{E}.{P}.out_dim_scale
```

MXFP4 保存以下两个 Tensor，不同时保留该投影的 VQ2 字段：

```text
layers.{L}.ffn.experts.{E}.{P}.weight
layers.{L}.ffn.experts.{E}.{P}.scale
```

每个专家的GateUp 与 Down 独立选择量化格式。因此每个专家可能有 4、6 或 8 个 Tensor。

以下给出导出后的完整命名示例，这些名称应同时用于 safetensors 文件头和 `model.safetensors.index.json` 的 `weight_map` 键。

**示例 1：第 42 层专家 248，GateUp 和 Down 均为 VQ2，共 8 个 Tensor。**

```text
layers.42.ffn.experts.248.w13.packed_indices
layers.42.ffn.experts.248.w13.codebooks
layers.42.ffn.experts.248.w13.perm
layers.42.ffn.experts.248.w13.out_dim_scale
layers.42.ffn.experts.248.w2.packed_indices
layers.42.ffn.experts.248.w2.codebooks
layers.42.ffn.experts.248.w2.perm
layers.42.ffn.experts.248.w2.out_dim_scale
```

**示例 2：第 42 层专家 250，GateUp 和 Down 均为 MXFP4，共 4 个 Tensor。**

```text
layers.42.ffn.experts.250.w13.weight
layers.42.ffn.experts.250.w13.scale
layers.42.ffn.experts.250.w2.weight
layers.42.ffn.experts.250.w2.scale
```

**示例 3：第 0 层专家 4，GateUp 为 VQ2、Down 为 MXFP4，共 6 个 Tensor。**

```text
layers.0.ffn.experts.4.w13.packed_indices
layers.0.ffn.experts.4.w13.codebooks
layers.0.ffn.experts.4.w13.perm
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

| 逻辑投影 | 输入 shape `[M,K]` | 权重存储 shape `[N,K]` | 输出 shape `[M,N]` |
| --- | --- | --- | --- |
| Gate / w1 | `[M,4096]` | `[2048,4096]` | `[M,2048]` |
| Up / w3 | `[M,4096]` | `[2048,4096]` | `[M,2048]` |
| GateUp / w13 | `[M,4096]` | `[4096,4096]` | `[M,4096]`，含 Gate、Up 各 2048 列 |
| Down / w2 | `[M,2048]` | `[4096,2048]` | `[M,4096]` |

### 1.1 GateUp 合并与算子布局

交付文件保留合并的 `w13`：

逻辑矩阵为 `[N,K]`，输出行使用 **全部 Gate 在前、全部 Up 在后** 的自然顺序。
该约定与 `convert_vq2_mixed_gs32_v3.py` 实际输出的 `row_order=natural_gate_then_up` 一致，
同时适用于 VQ2 和 MXFP4 的 `w13`。

当前 Gate、Up 各有 2048 个输出通道，`w13` 的第 0～2047 行为 Gate，第 2048～4095 行为 Up。
按 N32 分块后，顺序为 `Gate₀ → Gate₁ → … → Gate₆₃ → Up₀ → Up₁ → … → Up₆₃`。
四维 `packed_indices` 描述块内字节布局，不改变上述输出行序；交付文件不做 Gate32/Up32 交错。

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
| VQ2 | `perm` | I32 | `[K]` |
| VQ2 | `out_dim_scale` | U64 | `[N]` |
| MXFP4 | `weight` | U8 | `[N,K/2]` |
| MXFP4 | `scale` | FP8 E8M0 | `[N,K/32]` |

当前模型的各投影参数 shape 为：

| 参数 | GateUp | Down |
| --- | --- | --- |
| VQ2 `packed_indices` | `[128,256,16,8]` | `[128,128,16,8]` |
| VQ2 `codebooks` | `[128,128,16,2]` | `[128,64,16,2]` |
| VQ2 `perm` | `[4096]` | `[2048]` |
| VQ2 `out_dim_scale` | `[4096]` | `[4096]` |
| MXFP4 `weight` | `[4096,2048]` | `[4096,1024]` |
| MXFP4 `scale` | `[4096,128]` | `[4096,64]` |

## 3. VQ2 四个参数的精确定义

对单个专家、单个投影，逻辑权重矩阵为 `[N,K]`，N 为输出行数，K 为输入列数。
四个字段通过共同名称前缀 `layers.{L}.ffn.experts.{E}.{P}.` 关联。

### 3.1 packed_indices

`packed_indices` 保存码本编号。**一个 U8 字节存两个 4-bit 索引，每个索引查出两个 FP8 值，共恢复四个值。**

交付 shape 固定为 `[N/32,K/16,16,8]`，dtype 为 U8，连续存储且最后一维变化最快。
四个轴依次为输出行块、输入列存储块、块内输入列、该列的打包字节。。

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

其中 `j` 是编码输入列，尚未应用 `perm`。文件字段名保持 `packed_indices`；加载器可将其映射到内部 `packed_zn`，无需再次重排索引字节，也无须在文件中保存两份数据。

查表后，各行再乘从 `out_dim_scale` 解码出的 FP32 scale，并通过 `perm` 恢复原始列位置，详见第 3.3、3.4 节。

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

以上为查表得到的 FP8 值，尚未应用 scale 和 perm。
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

### 3.3 perm

交付 dtype 为 I32。`perm[j]` 表示编码列 j 对应的原始输入列编号。
必须是 `0…K-1` 的完整置换；没有置换时显式保存恒等排列。
当前算子要求 I64 时，由框架在加载阶段转换一次，不在每次推理时转换。
`perm` 只表示当前专家、当前投影内部的输入列排列，不参与专家 ID 或权重分片的映射。

### 3.4 out_dim_scale

每个输出行一个缩放系数，交付 dtype 为 U64，shape 为 `[N]`。
每个 U64 的低 32 位保存 FP32 scale 的原始位模式，高 32 位必须为零；解码后的 scale 必须有限且严格大于 0。
例如 `1.0` 保存为 `0x000000003F800000`。

## 4. MXFP4 的格式约定

保存 `weight` 和 `scale` 两个字段，后者直接以 FP8 E8M0 保存缩放系数：

```text
weight：每字节两个 E2M1 编码，低半字节对应前一个元素。
scale ：FP8 E8M0 缩放系数，每行沿 K 方向每 32 个权重共享一个。
```

`weight` 保存的是 U8 `[N,K/2]` 的打包 FP4 码字，不是 `[N,K]` 的稠密浮点权重。
本节的 `scale` 是 MXFP4 的独立 Tensor；第 3.4 节公式中的局部变量 `scale` 则由 VQ2 的 `out_dim_scale` 解码得到，两者不能混用。

`scale` 的 safetensors dtype 为 `F8_E8M0`，对应 `float8_e8m0fnu`。
它没有符号位和尾数位。设一个元素的原始字节为 `b`，有效范围为 `0…254`，其浮点值为 `2**(int(b)-127)`。
`0x7F` 表示 `1.0`，`0x00` 表示 `2**(-127)` 而不是零；`0xFF` 表示 NaN，不得出现在交付的 scale 中。

约束：

- 分组沿输入列 K 维进行，不跨行。
- GateUp 合并时，分别沿输出行拼接 Gate、Up 的 `weight` 和 `scale`。
- 每个 scale 必须有限且严格大于零，表示范围为 `2**(-127)…2**127`。
- 交付的是 E8M0 缩放系数，不是未加偏置的整数指数；推理时直接使用其缩放数值，不再对其执行 `2**scale` 或重复加偏置。
- 若消费者需要原始字节，使用 U8 字节视图，不做整数数值转换；文件仍以 `F8_E8M0` dtype 保存。
- MXFP4 投影没有 `perm`、`codebooks` 或 `out_dim_scale`，按自身的输入列顺序恢复权重。

## 5. 专家归属、混合精度与分片索引

### 5.1 每个投影的字段完整性

加载器通过名称中的 `(L,E,P)` 关联同一投影的参数，再根据完整字段集合识别 VQ2 或 MXFP4。
无需额外的列表位置映射；可选的专家清单只用于摘要或完整性检查，不改变 Tensor 名称中的专家身份。

当前 mixed 权重的部分专家示例：

| 层号 | 专家 ID | GateUp / w13 | Down / w2 |
| --- | --- | --- | --- |
| 0 | 4 | VQ2 | MXFP4 |
| 42 | 248 | VQ2 | VQ2 |
| 42 | 250 | MXFP4 | MXFP4 |

因此第 42 层专家 250 的两个投影都只保存 MXFP4 两字段，不额外生成 `packed_indices`。

必须满足：

- 全局层号为 `0…42`，每层专家 ID 为 `0…255`，每个专家都有 `w13` 和 `w2`。
- 同一 `(L,E,P)` 恰好具有 VQ2 的 `packed_indices/codebooks/perm/out_dim_scale` 或 MXFP4 的 `weight/scale`，禁止缺字段、混合两套字段或保留被回退覆盖的 VQ 副本。
- 每个字段的 dtype 和 shape 必须符合第 2 节，禁止保留专家批次维度。
- GateUp 与 Down 的量化类型独立判断，不能根据一个投影的格式推断另一个投影。
- `perm` 及 scale 必须与名称中同一专家、同一投影的压缩权重对应。
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
    "layers.42.ffn.experts.248.w2.perm": "layer_42.safetensors",
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
