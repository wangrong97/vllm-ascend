# EPLB × W2A8/W4A8 混精场景:专家规划(Planning)与权重搬运算法设计(含冗余专家与 token 分流)

> 范围:在 vllm-ascend `feat/vq2a8_v0.1` 的 VQ2A8-GS32 混精(W2 VQ2 + W4 MXFP4 专家级混精)之上,引入 EPLB 式动态专家重排,**与冗余专家(redundant experts)流程及 token 分流(token splitting,路由侧负载重分配)整合**。本文覆盖 **planning(规划)** 与 **权重搬运(weight migration)** 两部分,并以 token 分流作为其快环配套;通信、CI 不在本文范围。
> 依据代码基线:`feat/vq2a8_v0.1`(详见姊妹文档 [W2A8_VQ2A8_GS32_代码流程分析.md](W2A8_VQ2A8_GS32_代码流程分析.md)、[MoE专家预取与专家替换机制分析.md](MoE专家预取与专家替换机制分析.md))。

---

## 目录

- [1. 现状](#1-现状)
- [2. 系统方案](#2-系统方案)
- [3. 方案概述](#3-方案概述)

---

## 1. 现状

### 1.1 混精侧(VQ2A8-GS32)

**权重组织:bank = (投影 × 量化类型) 的常驻堆叠张量组**

```
layer.vq2a8_weights (GS32ExpertWeights, vq2a8_gs32.py:175)
banks[kind][quant_type] = {
    "fields": {name: Tensor[E_bank, ...]},        # 常驻,加载后地址不变
    "expert_ids": (全局专家号,升序),
    "slots":    {expert: slot},                    # 紧凑下标 0..E_bank-1
}
```

- 每专家迁移单位是**异构多字段 bundle**,不是一个张量:
  - VQ2 专家(w13 与 w2 各 4 个字段):`packed_zn [N/32,K/16,16,8]`、`pair_lut [N/32,K/32,32]`、`row_scale_bits [N]`、`perm [K]` —— **每专家独立 perm/scale,这正是 `EPLB_EXPERT_WEIGHT_NAMES` 无法表达 VQ2A8 的原因**;
  - MXFP4 专家(各 2 个字段):`mxfp4_weight [N,K/2]`、`mxfp4_scale [N,K/32]`;
- 每 expert 字节数:VQ2 ≈ `0.28·N·K`,MXFP4 ≈ `0.53·N·K`(**≈1.9 倍差**)。

**放置:纯线性,不感知类型/负载**(`gs32_expert_placement`,vq2a8_gs32_ep.py:52):本 rank 持有区间 `[ep_rank·L, (ep_rank+1)·L)`,专家个数天然均衡,但**字节量与热度不均衡**。校验强制 `expert_placement_strategy=="linear"`、无冗余专家、专家数整除(vq2a8_config.py:78-99)。

**运行时强依赖"常驻 + 线性"的三处机制**:

| 机制 | 位置 | 依赖 |
| --- | --- | --- |
| 设备索引 `expert_ids_tensor / logical_slots[num_experts] / resident_mask` | `GS32ExpertWeights.__init__`(vq2a8_gs32.py:256-265) | bank 内容 → 逻辑专家映射;`logical_slots` 是**单值映射**(一逻辑专家在本 rank 至多一个槽) |
| VQ2 分组投影计划 `plan = prepare_grouped_plan(..., E_bank)` | 同上:268-272 | 以 bank 内专家数为组数,捕获进 decode 图 |
| 有效性/图回放 `validity_buffer / capture_owners` | vq2a8_gs32.py:245-246,398-401 | bank 张量池外持有,回放重读 |

**空组语义已具备**(利好):设备路由算子明确支持"无活跃路由的专家、填充"(docs/vq2a8/vq2a8_gs32_runtime.md:12),为**空槽预留**提供运行时基础。

**通信与 token 派发**:EP 下强制 ALLGATHER(`gs32_ep_uses_allgather`)—— TP 复制输入广播给各 rank,各 rank 按 `expert_map`/`active_expert_range` **认领**本地专家的 token 行,公共流程恰好一次求和部分输出。**利好**:非线性放置与副本分流都不破坏通信正确性;token 的 rank 归属本质上是"认领规则",天然支持重分配。

**负载统计现状**:`AscendRoutedExperts.moe_load`(routed_experts.py:347,槽位级 `[local_num_experts]` 设备张量)是通用机制;GS32 路径每步 `apply` 内部本就计算 per-slot `counts`(设备侧),**采集点现成,只差累进**。

**EPLB/offload 全禁**:`supports_eplb=False`、`validate_vq2a8_config` 拒绝 `enable_eplb/dynamic_eplb/num_redundant_experts/expert_map_path`(vq2a8_config.py:92-99)、offload/sleep 禁。

### 1.2 EPLB 侧(两套独立实现 + 冗余专家 + 副本路由)

**A. 华为动态 EPLB 子系统(`vllm_ascend/eplb/`,仅 MRV1)**

- 周期:热度采集(`expert_heat_collection_interval` 步)→ 独立子进程规划(`algorithm_execution_interval` 步)→ **逐层权重更新流水**(`num_moe_layers` 步,每步一层);
- 搬运:`D2DExpertWeightLoader`(三态状态机):`forward_before` 生成逐专家 P2P 并异步发出(与当前层 forward 重叠),`forward_end` 等待 → `do_update_expert_map/log2phy/weight`;
- 适配层:`VllmEplbAdaptor` 的 `EPLB_EXPERT_WEIGHT_NAMES` 假设**同构单列表张量**,bank 化混精权重无法套用。

**B. 上游 EPLB 集成(MRV2)**

- `AscendEplbState.rearrange`(无新负载门控 → 上游 rebalance → 刷新路由表);
- `AscendEplbLayerState.expert_replica_routing_table`:**graph-stable 副本路由表** —— 重排后形状不变则原地 `copy_`,已捕获图地址稳定。

**C. 冗余专家机制(三处实现,同一核心思想)**

冗余副本的选择均为**贪心最小化最大副本平均负载**(EPLB 论文式):每次把下一个副本分给"原始负载 ÷(已有副本数+1)"摊派后仍最热的逻辑专家。

| 实现 | 位置 | 选择算法要点 | 放置约束 |
| --- | --- | --- | --- |
| DefaultEplb | `policy_default_eplb.py::compute_balanced_pack_redundancy` | 摊派贪心定副本数 → 全部物理槽(含副本)按负载降序 LPT 装箱 | 同一逻辑专家多副本必须异卡 |
| SwiftBalanceEplb(默认) | `policy_swift_balancer.py::compute_redundant_assignments` | 同上贪心 + **单专家副本数 < rank 数**;`is_node_redundant`(按 `num_die_per_host` 节点内冗余) | **一 NPU 至多一个副本** |
| 上游 vLLM | `eplb/policy/default.py::replicate_experts` | `argmax(weight / logcnt)` 逐层分配;层级化(节点内复制+装箱) | 副本优先同节点 |

**D. token→副本的现行分配(分流现状)**

`build_expert_replica_routing_table`(ops/fused_moe/eplb.py:10)+ `map_to_physical`:副本选择是 **`(行号 + ep_rank + logical_id) % replica_count` 的朴素 rank 感知轮询** —— 均衡目标是 **token 条数近似均分**,与专家放置解耦、不感知副本所在 rank 的实时压力,更不区分专家精度类型;`map_to_physical_and_record` 在映射同时按物理槽记录负载(副本命中可归属回逻辑专家),是**负载记账回流**的既有范式。

### 1.3 差距分析(Gap)

| 维度 | 既有现状 | 混精场景需要 | 缺口 |
| --- | --- | --- | --- |
| 迁移单位 | 同构单字段张量(`w13_weight_list[i]`) | 异构多字段 bundle(VQ2 8 切片含 per-expert perm / MXFP4 4 切片) | bundle 传输布局 |
| slot 管理 | unbind 独立张量直接覆盖 | bank 第 0 维切片 + 设备索引联动 + **空槽语义** | 切片级替换 + 原子刷新 + 冗余空槽 |
| 均衡目标 | 专家个数 × 热度(等权 token) | 字节容量 + **类型计算代价**(加权负载) | 双约束规划器 |
| 冗余副本 | 按"个数"装箱,副本无字节/类型差异 | 副本字节按类型;fill/evict 改变 rank 字节占用 | 字节感知副本分配(类型预算分流) |
| **token→副本分配** | 朴素轮询,均衡 token 条数,与放置/类型解耦 | **类型加权负载分流 + 序列粘性 + graph-stable + 与规划闭环** | 设备侧加权分流表 |
| 图兼容 | 副本路由表原地刷新 | `logical_slots`(可原地)+ `plan`(组数变化需重捕获) | 容量预留 + 按容量预建 plan |
| 放置校验 | 上游支持 expert_map 非线性 | VQ2A8 强制 linear + 禁 EPLB/冗余 | 放开校验并注入非线性 expert_map |
| hash 层 | 无特殊处理 | 稀疏库存 + `tid2eid` 校验 | 第一版排除 |

---

## 2. 系统方案

### 2.1 设计目标与约束

**目标**

- G1 负载均衡:最小化 `max_r cost(rank_r)`,专家代价按**类型区分**(VQ2/MXFP4 kernel 吞吐不同);
- G2 显存可控:每 rank 权重字节数(含冗余副本与 staging)不超预算 `B_r`,规划期硬约束;
- G3 热点消峰(权重侧):对持续热点专家提供**冗余副本**手段,副本选择字节感知 —— **复制 VQ2 热专家的性价比约为 MXFP4 的 1.9 倍**;
- G4 在线无损:搬运与推理重叠(逐层流水),切换原子、可回滚,不破坏 validity 与图回放正确性;
- G5 最小侵入:复用既有 EPLB 周期框架(`EplbUpdator` 三阶段)与 GS32 运行时,不改动原生算子与 checkpoint 格式;
- G6 热点消峰(路由侧):**token 分流**作为零搬运成本的快环 —— 在多副本专家间按**类型加权负载**重分配 token,响应秒级热度波动;能用路由解决的绝不动权重。

**硬约束(来自现状分析)**

- C1 类型不可变:VQ2↔MXFP4 是离线量化产物,重排与复制只作用于**同类型**权重;
- C2 图稳定:decode 图捕获了 `logical_slots/resident_mask` 的**地址**与 `plan`;方案要求 **bank 容量(活跃专家数 + 冗余空槽)恒定、`plan` 按容量预建**,运行期任何 move 都不改变张量形状与组数(见 2.5-D);**分流决策必须是确定性的纯设备侧运算**(同输入同输出),否则破坏回放一致性;
- C3 权重常驻:staging 与空冗余槽常驻设备(计入 `B_r`);
- C4 提交原子:映射切换与数据就位满足"全有或全无"(层粒度两阶段提交);
- C5 hash 层(稀疏库存 + `tid2eid`)V1 不参与重排/复制/改表;
- C6 副本约束:一个逻辑专家在一个 rank 至多一个副本(与 `logical_slots` 单值映射自洽);副本类型 = 源专家类型;副本数 < rank 数;
- C7 分流语义无损:**不改变 top-k 专家集合与 topk_weights**,只在等价副本间选择归属 —— 每个 token-专家对恰好被一个副本执行一次,输出加权求和严格不变。

### 2.2 总体架构:双时间尺度闭环

复用 EplbUpdator 三阶段周期骨架(慢环),叠加每步执行的 token 分流(快环),两级共用同一**类型感知代价模型**与同一**槽位级负载记账**:

```
┌──────────────── 快环(每个 forward,零搬运成本,纯设备侧)───────────────┐
│  输入:topk_ids/topk_weights(router 决定,不改,C7)                       │
│  分流:多副本逻辑专家 → 全局一致权重表的确定性加权轮询选副本(2.4-C)  │
│  记账:per-slot counts 累计(EWMA 快表 + 周期累计慢表,同一数据源)     │
│  作用域:仅多副本专家;单副本专家无自由度,残余失衡留给慢环            │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ 实测加权负载(分流后真实命中)
┌───────────────────── 慢环(EPLB 周期:heat + algo + L 步流水)──────────┐
│ ① 热度采集(heat_interval 步):槽位 counts → 归并逻辑级(副本命中求和) │
│ ② 规划(algo_interval 步,子进程并行):                                 │
│    阶段 A 冗余分配 ReplicaAllocator(字节感知摊派贪心,类型预算分流)   │
│    阶段 B 放置调整 TCSB(同类型 swap + fill/evict,统一价值密度)       │
│ ③ 逐层搬运流水(L 步):issue 异步 P2P → commit 原子提交                │
│    (调度顺序:evict → swap → fill;提交点同步刷新分流表)              │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ 新放置/副本集 → 重置快环分流表与 EWMA
```

分工:**分流吃噪声(易变热度),迁移治结构(持续偏移)** —— 快环先把 imbalance 压到 `τ_token` 以内,残余的结构性失衡(单副本专家、无副本可分、字节不均)才由慢环用 swap/fill/evict 解决;慢环的每个 fill 新副本立即进入快环可分集合。

**两环的供给-消费关系(自由度绑定)**:token 分流的自由度**完全由冗余副本供给** —— `fill` 创造自由度(`route[e]` 增一项,该专家进入分流作用域)、`evict` 收回自由度(减至单副本即退出作用域)、`swap` 不改变自由度但改变分流背景(副本迁往负载/类型构成不同的 rank,分流选择随之变化)。反向闭环:ReplicaAllocator 的摊派假设 `load/(replicas+1)` 由快环兑现(流量真的被分过去才有收益);EWMA 中副本长期分不到流量 → `value` 跌破 `τ_revoke` → evict 收回槽位。单副本专家自由度为零(token 归属 = 放置本身),只能经 swap 换 rank 或 fill 变多副本后进入快环管辖;初始无副本时快环作用域为空,故 P0 静态规划必须包含初始 `replica_alloc` 以启动双环。

**EP 场景联合约束(方案与整专家 EP 的对接前提)**

- **E1 认领唯一性(正确性关键)**:ALLGATHER 下所有副本 rank 都收到同一 token 行,分流选择必须使**每个 rank 对同一 (token, e) 解出同一个 r\*** —— 选择只依赖全局一致的慢变量(`route`/`weight`,由慢环提交段统一刷新到所有 rank);**本地 EWMA 绝不进入选择路径**(各 rank 本地视图必然不一致,详见 2.4-C);
- **E2 隐式跨 rank 提交屏障**:VQ2A8 强制同步调度使各 rank 逐层 forward 天然对齐,swap 两端在**同层 `forward_end`** 各自提交 P2 —— 否则中间态下 e_B 短暂双 rank 持有,认领冲突(漏算/双算);
- **E3 通信域统一**:bundle P2P 与 MoE ALLGATHER 复用同一 EP 进程组(与 `get_dynamic_eplb_group` 做一致性断言);
- **E4 EP 校验分离**:保留"逻辑专家数整除 ep_size";把 `local_capacity == global/ep` 的线性恒等改为 `capacity = 本地逻辑数 + 分类型空冗余槽`;EP=1 时无跨 rank 自由度,方案自动退化为"仅规划诊断、不搬运";
- **E5 分流零额外通信**:ALLGATHER 输入广播量不变,分流只改各 rank 的认领行;EP padding 语义(`Σcounts ≤ 行数`,`allow_padding`)对多副本认领同样成立;
- **E6 空库存边界**:hash 层空库存 rank(VQ2A8 EP 特有,仅剩 anchor)按 C5 排除,不经 fill 脱离空库存;非 hash 层在线性基线下每 rank 恒有 L 个专家,不存在空库存。

### 2.3 核心数据结构

```python
@dataclass(frozen=True)
class ExpertBundle:                     # 迁移单位:一个专家的全部字段切片
    expert_id: int                      # 全局逻辑专家号
    quant_type: str                     # "vq2" | "mxfp4"(C1:不变)
    slices: dict[tuple[str, str], torch.Tensor]
    # ("w13"|"w2", 字段名) → bank 张量第 0 维切片视图(连续内存)
    #   vq2: packed_zn/pair_lut/row_scale_bits/perm ×2 投影 = 8 切片
    #   mxfp4: mxfp4_weight/mxfp4_scale ×2 投影 = 4 切片
    nbytes: int

@dataclass
class BankSlotTable:                    # 每 rank 每层放置真值(设备+主机双份)
    capacity: dict[tuple[str, str], int]
    # (kind, quant_type) → bank 第 0 维容量 = 本地活跃数 + 预留冗余空槽
    # 启动期确定,运行期恒定(C2);两类型容量各自独立(类型预算分流,2.4-A)
    host_placement: dict[tuple[str, str], dict[int, int]]
    # 设备侧真值 = GS32 已有 expert_ids_tensor/logical_slots/resident_mask

@dataclass(frozen=True)
class ExpertMove:
    kind: str                           # "swap" | "fill" | "evict"
    expert_id: int; quant_type: str
    src_rank: int; src_slot: int
    dst_rank: int; dst_slot: int
    evicted_expert: int | None          # swap 对端
    # swap: 双向同类型对换(活跃槽↔活跃槽),字节对称
    # fill: 副本填充(主副本 rank → 冗余空槽),单向,字节+
    # evict: 撤销副本槽(仅清映射不清权重),字节−

@dataclass(frozen=True)
class MigrationPlan:
    moves: list[ExpertMove]
    new_placement: dict[int, BankSlotTable]
    replica_alloc: dict[int, int]       # 逻辑专家 → 总副本数(含主副本)
    expected_cost: list[float]; expected_bytes: list[int]

@dataclass
class SplitTable:                       # 快环分流状态(全设备张量,graph-stable)
    route: torch.Tensor                 # [E_logical, capacity] 副本 rank 列表(固定形状,-1 填充)
    weight: torch.Tensor                # [E_logical, capacity] 分流份额(慢环下发,全局一致,P2 原地刷新)
    cdf: torch.Tensor                   # [E_logical, capacity] weight 前缀和(与 weight 同点刷新,查表用)
    ewma_load: torch.Tensor             # [R] 各 rank 类型加权 EWMA(本地累计,仅作慢环输入,不参与选择)
    # 选择:按 cdf[e] 的确定性哈希落点,键 (seq_id, e)(C2/序列粘性/E1 认领唯一,见 2.4-C.4)
    # 记账:两级 —— counts[slot](token 量)与 EWMA_time[s](时间份额,见 2.4-C.5);同源
    # 刷新:慢环按 EWMA 全局快照折算 weight/cdf(route/weight/cdf 三表 P2 原子刷新)
```

### 2.4 规划与分流算法(类型/字节感知,快慢两环)

**共享代价模型**(两环唯一口径;`load` 恒为 token 量,`c_type` 恒为单位代价,乘积才是时间):

```
c_type(e) = (α_w13 + α_w2) + (β_w13 + β_w2) / M̄(e)      # 摊销单位代价(时间/token)
   α/β:按 (类型, 投影, M 档位, 执行模式, 形状) 查表,分类型离线标定,
        相对单位 α_vq2 ≡ 1;M̄(e) = TotalTokens(e)/launches(e) 为窗口内
        平均每次有效启动行数(decode 档 M̄≤8 时 β/M̄ 项显著 —— 小 M 下
        launch 开销占比高;摊销为代数变形,同档位内不损精度)
load(e)  = Σ_{t ∈ W} Σ_{s ∈ replicas(e)} M_s(t)          # 采集窗口 W 内命中 token 总量
   数据源:每步 GS32 apply 的槽位级行数 M_s(t)(group_list 差分,设备张量),
   副本命中按槽位累计后归并到逻辑专家;可选除以 |W| 归一
load(e)·c_type(e) = αΣ·TotalTokens + βΣ·launches = 窗口内该专家执行时间
cost_rank(r) = Σ_{l,e ∈ r} load[l][e] · c_type(e)
bytes_rank(r) = Σ bytes[e] + staging常驻                    # 硬约束 ≤ B_r
```

**A/B/C 三部件的关系(导读)**:三者不是并列的独立算法,而是同一闭环按执行时机的分节 —— **A(冗余分配)是 C(分流)的供给侧,C 是 A 的验证侧与输入源**:

- 链路① A 的输入来自 C:`load(e)` 取 C 分流后的真实命中(残余热度),A 只对分流治不了的结构性热点出手;
- 链路② A 的输出改变 C 的作用域:fill/evict 提交即增删 `route[e]`,划定 C 的自由度边界;
- 链路③ A 的摊派假设(`load/(replicas+1)`)由 C 的 `weighted_pick` 兑现 —— 副本分不到流量则假设落空;
- 链路④ A 的撤销决策由 C 的实测触发:EWMA 中副本长期无命中 → `value < τ_revoke` → evict。

B(放置调整)在 A 给定的副本结构下做权重搬运(swap 不改变作用域,只改变 C 的选择背景)。A 与 C 不能合并为一个环节:fill/evict 有 bundle 搬运成本与提交协议,不能跟步级热度抖动;C 无搬运能力,追不上结构漂移 —— **结构慢调、瞬时分配快调**,中间靠全局一致的 `weight` 表单向传递决策(慢环折算 EWMA → weight,C 只读,见 E1)。

**A. 慢环阶段 A:冗余分配(ReplicaAllocator,每 N 周期,类型预算分流)**

```
输入:load(逻辑级,快环分流后的真实命中归并)、两池剩余额度 R_vq2 / R_mx4
价值:value(e) = load(e) / (replica_alloc[e] + 1)        # 摊派后单副本负载
成本:bytes[e](按类型;VQ2 ≈ MXFP4 的 1/1.9 → 同预算可复制更多)
贪心:两池各自独立执行(互不挤占,防一类副本占满另一类的额度与字节):
    VQ2 池:e* = argmax{ value(e) : type(e)=vq2, e∉hash层,
              replica_alloc[e]+1 < num_ranks,           (C6)
              ∃r: e∉r 且 free_slots[r,vq2]≥1 且 bytes_rank[r]+bytes[e] ≤ B_r }
    MXFP4 池:同上(type=mx4,独立额度)
    池间比例启动期按字节对半定(槽位比 ≈ 0.53:0.28),运行期恒定(bank 容量刚性,C2)
反向:value 低于撤销阈值 τ_revoke(带滞回窗口)的副本 → 生成 evict
```

与既有三套实现同构(摊派贪心),差异:**价值引入字节密度(value/bytes)** 与**类型预算分流**;放置沿用"一 rank 一副本 + 副本数 < rank 数"。

**B. 慢环阶段 B:放置调整(TCSB,每周期)**

```
imbalance = max_r cost_rank(r) − mean_r cost_rank(r)     # 用快环 EWMA 期末值
候选 move 三类(swap/fill/evict),统一价值密度:
    gain(move) = Δimbalance(move) / max(1, Δbytes(move))
循环直到 imbalance ≤ τ_slow 或 |moves| = K:
    优先 ① swap(r_hot↔r_cold 同类型对换,字节对称,gain=Δimbalance)
    其次 ② 阶段 A 的 fill/evict(字节核算:fill 增 / evict 减,先 evict 后 fill)
    取 gain 最大者;级联校验至固定点 → MigrationPlan
门控:expected_improvement < ε → 空计划(沿用"无新负载不重排")
单副本专家的失衡无法由快环解决 → 由本阶段 swap 或阶段 A fill 兜底
```

**C. 快环:类型加权副本分流(每 forward,SplitTable)**

**C.1 解决的问题**:多副本逻辑专家的 token 该由哪个副本 rank 执行。要求:语义无损(不改 top-k,C7)、各 rank 决策一致(EP 认领唯一,E1)、可进 decode 图且回放一致(C2)、零 host 同步。

**C.2 输入与执行位置**:输入为本步 `topk_ids/topk_weights`(router 输出,**只读不改**);执行点插在 routed_experts 路由之后、token dispatch 认领之前(2.6-4)—— ALLGATHER 下所有副本 rank 都持有同一 token 行,分流决定"谁认领";单副本专家(`len(route[e])==1`)不经分流直接归属。输出即本地认领位图(喂给 dispatcher 的 `expert_map`/认领逻辑)与本地 `group_list` 统计。

**C.3 权重表 weight 的构造(慢环每周期刷新,快环只读)**

分两步:先由记账折算各 rank 的加权负载 `L[r]`,再逆负载归一化出份额。两因子的定义与共享代价模型一致(`load` 恒为 token 量,`c_type` 恒为单位代价,乘积才是时间):

```
第一步:各 rank 类型加权负载 L[r](来自 C.5 两级记账的周期归并)
  load(e')  = Σ_{t ∈ W} Σ_{s ∈ replicas(e')} M_s(t)       # 采集窗口 W 内该专家
                                                          # 命中 token 总量(副本槽位归并)
  c_type(e') = (α_w13 + α_w2) + (β_w13 + β_w2) / M̄(e')   # 摊销单位代价(时间/token)
     α/β 按 (类型, 投影, M 档位, 执行模式, 形状) 查表;相对单位 α_vq2 ≡ 1;
     M̄(e') = TotalTokens(e')/launches(e')(窗口内平均每次有效启动行数;
     decode 档 M̄≤8 时 β/M̄ 项显著 —— 小 M 下 launch 开销占比高)
  恒等式:load(e')·c_type(e') = αΣ·TotalTokens + βΣ·launches
          = 窗口内该专家执行时间(摊销为代数变形,同档位内精确)
  L[r] = Σ_{e' ∈ r 活跃槽} load(e') · c_type(e')          # rank 的加权负载(时间份额)

第二步:逆负载归一化为副本份额
  对每个多副本专家 e、其每个副本 rank rᵢ ∈ route[e]:
    raw[e][rᵢ]    = clip( 1 / (L[rᵢ] + ε), w_lo, w_hi )  # 逆负载:所在 rank 越重份额越小
    weight[e][rᵢ] = raw[e][rᵢ] / Σⱼ raw[e][rⱼ]           # 归一化为份额(和为 1)
```

要点:

- **类型感知隐含在 `L[r]`**:`c_type` 分 VQ2/MXFP4 标定(α/β 来自两种 kernel —— 原生算子与 CANN GMM —— 的分段线性计时),MXFP4 密集的 rank 天然更重 → 其副本份额更低;weight 公式本身无需类型分支;
- **EWMA 口径**:周期之间用 `EWMA_time[s]`(C.5)做平滑,`L[r]` 取 Σ EWMA_time(等价于 `load·c_type` 的指数加权版),避免窗口边界阶跃;
- **`clip(w_lo, w_hi) 是防极端的保险丝,非常规调节**:作用于归一化前的 raw,限定任意两副本 raw 之比 ≤ `w_hi/w_lo`(示例 [0.2, 5] → 25:1;日常负载差远在界内不生效)。`w_hi` 防最空 rank 吸走全部流量(单点化 + 刷新后落点剧变);`w_lo` 保最重副本的**最低可观测流量** —— 份额一旦归零,副本 EWMA 恒为 0(零流量假象),`τ_revoke` 的 evict 判决失去依据且份额无法自然恢复(正反馈锁死);clip 同时收窄 P2 刷新前后 cdf 变化,抑制刷新瞬间的落点重排;
- **weight 是慢变量**:由慢环提交段(2.5-P2)按 EWMA 全局快照折算后,与 cdf 一并**原子刷新到所有 rank 的同一张表** —— 快环每步只读,不参与计算。

**C.4 选择算法 weighted_pick(确定性加权哈希,非随机、非 argmin)**

把 weight 变成副本的**累积分布(CDF)**,用确定性哈希落点选择:

```
前置(慢环刷新 weight 时一并算好,存固定形状表):
    cdf[e] = 前缀和 [w₁, w₁+w₂, …, 1]        # 例:weight=[1/3,2/3] → cdf=[0.33, 1.0]
每步、每个 (token t, 逻辑专家 e ∈ 多副本集合):
    key = (seq_id(t), e)                       # 确定性键:序列粘性 + 回放一致
    h   = hash(key) ∈ [0,1)                    # 整数哈希 / 2³²,纯设备侧运算
    r*  = cdf[e] 上 h 落入的区间对应的副本      # 一次查表;平局按 rank 序号破平
    本 rank 判定:r* == my_rank ? 认领该行 : 放弃
```

**例子**:专家 e 有两个副本,在 rank A(L=80ms)与 rank B(L=40ms)→ weight = [1/3, 2/3],cdf = [0.33, 1.0]。序列 s 的 `hash(s,e)=0.70` → 落第二段 → 恒由 B 认领;另一序列 h=0.10 → A。全序列的哈希值均匀展开,大数定律下流量按 1:2 分流 —— **单序列确定、总体按比例**。

**C.5 记账与闭环(本地 EWMA 只作慢环输入,绝不回馈本步选择)**

认领发生侧做两级记账(同一数据源,与共享代价模型口径一致):

```
原始量:  counts[slot] += M_s(t)                       # 槽位本步行数(token 量)
时间份额:EWMA_time[s] ← λ·( M_s(t)·(α_w13+α_w2)^{type(e)}
                          + 1[M_s>0]·(β_w13+β_w2)^{type(e)} ) + (1−λ)·EWMA_time[s]
```

即 EWMA 累计的是**槽位时间份额**(斜率×行数 + 有效启动×截距),而非 token 条数;慢环归并 `Σ_{s∈replicas(e)}` 即得 `load(e)·c_type(e)`。该 EWMA **只**在慢环被收集(all_gather 归并、槽位→逻辑求和)后折算进下一周期的 weight —— 本步选择与本步记账之间**没有任何耦合**,否则形成反馈回路,破坏确定性与回放一致。

**C.6 设计依据(三个被排除的替代方案)**

| 替代方案 | 被排除的原因 | 本方案的对应手段 |
| --- | --- | --- |
| `argmin L[r]`(每步最小负载) | L 是各 rank 本地 EWMA,视图不一致 → 同一 (t,e) 各 rank 解出不同 r\*,漏算/双算(combine 求和把双算计入两次,违反 E1/I8);且全压单一副本,无比例分配 | 负载感知改为**慢变量**:经慢环折算进全局一致的 weight,快环只读 |
| 真随机按 weight 抽样 | 非确定 → decode 图回放结果不一致(C2),且破坏序列粘性 | 确定性哈希 key=(seq_id, e) |
| 朴素轮询(基线 `build_expert_replica_routing_table` 的取模) | 不感知负载、不感知类型,均衡的只是"行号均匀" | 保留其确定性与 graph-stable 优点,把"取模基数"升级为"负载折算的 weight" |

实现形态小结:全部为固定形状表上的纯张量运算(哈希、查表、掩码),无 host 判断、无数据依赖分支 —— 可被 decode 图整段捕获;`route/weight/cdf` 三表由慢环 P2 原子刷新(图地址不变,内容生效),与 `map_to_physical_and_record` 的记账范式同构,但均衡目标从 token 条数换成**类型加权时间份额**。

**闭环口径**(防两环互搏):快环 `ewma_load` 与慢环 `load` 出自**同一 per-slot counts 数据源**(GS32 `apply` 内 `group_list`/`counts`),副本命中按槽位累计、归并回逻辑专家 —— 规划看到的永远是"分流之后"的真实负载,不会出现"规划输入与分流输入口径不一"导致的震荡;慢环提交点重置 `ewma_load` 与 `route`(见 2.5-P2)。

### 2.5 权重搬运算法(bundle 级 D2D 流水,支持 swap/fill/evict)

**D(关键设计)容量预留与按容量预建 plan**

- 启动期:每 rank 每 bank 第 0 维容量 = 本地活跃专家数 + 预留冗余空槽(两类型各自预留,见 2.4-A);空槽初始化 `expert_ids_tensor[slot]=-1`、`logical_slots[e]=-1`、`resident_mask[e]=False`;
- **VQ2 分组投影计划按容量(而非活跃数)一次性预建**:`prepare_grouped_plan(..., groups=capacity)` —— 空槽 counts 恒 0,设备路由与原生分组路径已支持空活跃专家(1.1),**任何 move 都不改变组数,plan 与 decode 图全程免重建/免重捕获**(C2);
- `logical_slots` 保持单值映射(一逻辑专家在本 rank 至多一个槽),与 C6 自洽;跨 rank 多副本归属由 `SplitTable` 决定。

**传输布局**:bank 第 0 维切片天然连续,HCCL `isend/irecv` 直接按切片视图收发;bundle 多字段切片各自独立 P2P(VQ2 8 / MXFP4 4),`batch_isend_irecv` 一次批量发出(构造方式复用 `D2DExpertWeightLoader`,张量来源换成 bundle 枚举器)。

**接收侧:常驻类型分组 staging 双缓冲**

```
staging[type] = [ {field: empty_like(bank_field[0])} ] × n_stage   # 默认 2,计入 B_r
```

swap 目标槽仍驻旧专家且可能被本周期前序图回放读取,故统一经 staging(C4);fill 直收空槽为 V2 优化项。

**三类 move 的搬运与提交**

```
forward_before(layer l):
    按 调度顺序(evict → swap → fill)取本层 moves:
      evict: 无传输任务
      swap:  双方互为 src/dst 构造 P2POp(isend 旧槽切片 / irecv staging)
      fill:  src 侧 isend 主副本切片,dst 侧 irecv staging(单向)
    batch_isend_irecv 异步发出                      # 与第 l 层 forward 重叠
forward_end(layer l):
    req.wait() 全部完成
    两阶段提交:
      P1 数据段:staging → 目标槽原地 copy_
          swap: 双端覆盖;fill: 空槽写入(evict 遗留数据直接被覆盖,无需清理)
      P2 提交段(设备索引/分流表原子刷新,全部固定形状原地写,图地址稳定):
          swap: expert_ids_tensor[slot] 互换;logical_slots 两键互换
          fill: expert_ids_tensor[空槽]=e;logical_slots[e]=slot;resident_mask[e]=True
          evict: expert_ids_tensor[slot]=-1;logical_slots[e]=-1;
                resident_mask[e]=False(权重保留槽内,下次 fill 覆盖)
          SplitTable 同步:route[e] 增/删本 rank(原位写);
                本层无剩余 move 时按本轮 EWMA 全局快照刷新 weight/cdf、重置 ewma_load
          host_placement / dispatcher expert_map / replica_alloc 同点更新
      跨 rank 屏障(E2):swap/fill/evict 的 P2 在各 rank 同层 forward_end
      提交 —— 同步调度下逐层 forward 天然对齐,两端同点切换,无中间态
      双持有;P2 全部依赖的表(route/weight/expert_map)为全局一致内容
      失败 → 回滚:P1 逆拷贝,P2 不动;本层 move 全部作废(层粒度原子)
```

**图模式(full_graph)兼容性**

- `logical_slots/resident_mask/expert_ids_tensor/SplitTable`:固定形状原地写 → 已捕获 decode 图的后续回放自动读到新映射与分流结果(与 `AscendEplbLayerState` graph-stable 手法同构);
- 分流的确定性(C2):`argmin` 对平局按 rank 序号破平,同一 (seq, e) 在放置不变期间恒选同一副本 —— 回放一致;
- `plan`:按容量预建、组数恒定 → 零重捕获;
- validity 兜底:提交后首 forward 若认领规则与设备索引不一致,`resident_mask` 校验置 invalid 并在采样前报错(P2 必须与 dispatcher `expert_map` 同点更新,I6)。

**失败与回滚**:传输失败 → staging 丢弃、本层作废、整周期保守取消;提交后五表一致断言;`expected_cost vs 实测时延`、`value(e) vs 副本命中增益`、`快环 EWMA 残余失衡` 记日志,闭环重标 α/β、τ_revoke 与 τ_token/τ_slow 分界。

### 2.6 运行时集成改造点(清单)

| # | 位置 | 改造 |
| --- | --- | --- |
| 1 | `GS32ExpertWeights` | bank 按容量分配(活跃+空槽,两类型独立);`moe_load` 槽位级累计;`enumerate_bundles()`;`commit_move`(P1+P2 三类 move);设备索引原地刷新 |
| 2 | `plan` 构建 | `prepare_grouped_plan(groups=capacity)` 按容量预建;空槽 counts=0 专项回归 |
| 3 | dispatcher `expert_map` / 认领 | 区间线性 → 由 `BankSlotTable.host_placement` + `SplitTable` 构造;多副本行按分流结果认领(ALLGATHER 下天然支持) |
| 4 | **分流执行点** | routed_experts 路由后、dispatch 前插入 `SplitTable` 查表(纯张量运算,可入图);EWMA 维护 |
| 5 | `EplbUpdator/VllmEplbAdaptor` | 混精 adaptor(bundle 枚举替代 `EPLB_EXPERT_WEIGHT_NAMES`);策略注入 `ReplicaAllocator + TCSB`;规划输入改读分流后负载 |
| 6 | `validate_vq2a8_config` | 新增 `vq2a8.eplb="typed_swap_replica_split"` 开关(默认禁用),放开对应校验 |
| 7 | hash 层 | 重排/复制/分流白名单过滤(C5);`tid2eid` 校验改读当前库存快照 |
| 8 | staging/空槽显存 | `n_stage` bundle 位 + 分类型冗余空槽计入 `B_r`;启动期预分配(C3) |
| 9 | EP 校验分离(E4) | `validate_vq2a8_config`/`gs32_expert_placement`:保留逻辑专家数整除,放开 `capacity==global/ep` 恒等,容量 = 本地逻辑数 + 分类型空槽 |
| 10 | 通信域(E3) | bundle P2P 与 ALLGATHER 复用 EP 进程组;`get_dynamic_eplb_group` 一致性断言;EP=1 时方案退化为仅规划诊断 |

### 2.7 安全不变式(验收清单)

- I1 任意时刻,每个逻辑专家至少一个主副本;全部副本分布在不同 rank(C6)且类型一致(C1);
- I2 任意 rank 任意层:`Σ bytes(活跃槽) + staging ≤ B_r`(fill 增量、evict 释放均计入);
- I3 **bank 容量恒定**(两类型各自独立);活跃专家数 ≤ 容量;`plan` 按容量预建一次,运行期零重建 → decode 图零重捕获;
- I4 `logical_slots ↔ expert_ids_tensor ↔ resident_mask ↔ host expert_map ↔ SplitTable.route` 五表一致(P2 原子点校验);
- I5 swap 仅同类型同层;fill 仅入对应类型空冗余槽;evict 仅作用副本槽(主副本不可 evict);
- I6 提交点之后首个 forward,dispatcher 认领视角与 GS32 设备索引/分流表视角完全一致(validity 兜底);
- I7 迁移预算 K 与两类型冗余槽 R_vq2/R_mx4 有界,单周期 P2P 流量可预估;
- I8 **分流不变式**:不改变任何 topk_ids/topk_weights 语义(C7);每个 (token, 逻辑专家) 对恰被一个副本执行一次,且**所有 rank 对同一 (t,e) 解出同一 r\***(选择仅依赖全局一致的 route/weight,本地 EWMA 不进入选择路径,E1);放置不变期间同一 (seq_id, e) 的副本选择恒定(确定性,图回放一致);单副本专家不经分流直接归属;
- I9 **跨 rank 提交对齐**:swap/fill/evict 的 P2 在各 rank 同一层的 `forward_end` 提交(同步调度 = 隐式屏障,E2);任何时刻不存在同一逻辑专家在两个 rank 同时"活跃就位"的中间态(swap 双持有窗口为零);
- I10 **通信域一致**:bundle P2P 与 MoE ALLGATHER 使用同一 EP 进程组(E3);规划/搬运/分流三处视图(主机 placement、设备索引、SplitTable)在 P2 原子点同源。

---

## 3. 方案概述

**核心思想**:把均衡手段按成本分成**两级自由度** —— 路由侧的 **token 分流(快环,零搬运、严格无损)** 与权重侧的 **专家迁移/冗余(慢环,搬运成本高、治结构)** —— 并让两级共用同一**类型感知代价模型**(VQ2/MXFP4 kernel 斜率不同)与同一**槽位级负载记账**(分流后的真实命中),形成闭环:分流吃秒级热度噪声,迁移治持续结构性失衡,冗余副本是两环的衔接点(慢环 fill 出的副本立即进入快环可分集合)。权重侧迁移单位从"同构张量"升级为"**同类型专家的多字段 bundle**"(VQ2 8 切片含 per-expert perm / MXFP4 4 切片),运行期通过"**bank 容量预留 + plan 按容量预建 + 固定形状设备索引/分流表原地刷新**",使交换、复制、撤销与分流调整全部对已捕获 decode 图透明、零重捕获。

**关键设计决策**

| # | 决策 | 理由 |
| --- | --- | --- |
| D1 | 只做**同类型** move(交换/复制) | 精度离线固定(C1);swap 字节对称;副本类型随源 |
| D2 | 两环共用 `cost = Σ load·c_type` 与槽位级记账 | 混精特有失衡源(1.9 倍字节差 + kernel 吞吐差)在规划与分流中同一口径,防互搏 |
| D3 | 冗余分配沿用**摊派贪心**(EPLB 论文式)+ **类型预算分流**(VQ2/MXFP4 冗余槽各自计数) | 复制热 VQ2 专家性价比 ≈ MXFP4 的 1.9 倍;防一类副本占满额度/字节造成另一类饥饿 |
| D4 | **容量预留 + plan 按容量预建,空槽 counts=0** | swap/fill/evict 均不改变组数 → plan/decode 图零重捕获;空组语义运行时已支持 |
| D5 | 三类 move 统一价值密度 `gain = Δimbalance/Δbytes`;提交 = P1 数据段 + P2 索引/分流表原子段 | 复刻 graph-stable 手法;evict→swap→fill 调度序保证字节先释放后占用 |
| D6 | 分流 = **全局一致权重表的确定性加权轮询 + 键 (seq_id, e) + 纯设备侧运算**;EWMA 仅作慢环输入 | 严格无损(C7);EP 各 rank 本地负载视图必然不一致,选择只依赖全局一致慢变量才能保证认领唯一(E1/I8);负载感知经慢环折算进 weight,兼顾图回放一致(C2)与零 host 同步 |
| D7 | 复用 `EplbUpdator` 三阶段周期与 P2P 构造,替换 adaptor/策略/分流表 | 侵入面最小(G5);MRV1 先行,MRV2 上游 EPLB 路径二期对接 |

**预期收益**:线性放置下字节与负载的 rank 间失衡(实测可达 ±30% 量级)由快环先压到 `τ_token`、慢环再压到 `τ_slow`;持续热点专家经冗余副本二次消峰(按 `value/bytes` 选择,显存换吞吐效率最优);重排在线完成,稳态零额外推理停顿,decode 图全程不重捕获。

**主要风险与缓解**

- R1 提交点与 dispatcher 认领/分流表不同步 → 误路由(现有 `resident_mask` validity 硬报错兜底;缓解:P2 同点更新,I6 验收);
- R2 空槽 counts=0 的原生分组路径未在 plan=容量组数下回归过(缓解:P0 专项回归;运行时文档已声明支持);
- R3 staging + 分类型冗余空槽常驻显存挤占 KV(缓解:`n_stage`/R 可配并计入 `B_r`,P0 以小 R 验证);
- R4 代价模型与 τ_revoke 标定偏差 → 规划次优/副本震荡(非正确性问题;闭环重标 + 滞回窗口);
- R5 **快慢环互搏**:分流追平后规划器误判热点已消、撤销副本后热度反弹(缓解:规划输入用 EWMA 慢表而非瞬时值;fill/evict 带滞回;两环目标分界 τ_token/τ_slow 显式化);
- R6 hash 层/空库存 rank 边界(缓解:C5 白名单,V1 排除)。

**分阶段落地**

- **P0(静态)**:采集打通 + 容量预留/plan 预建改造 + `ReplicaAllocator/TCSB` 离线规划 + 启动期按 `expert_map_path` 一次性放置(含初始副本分配),验证非线性 expert_map 与空槽 plan 正确性;
- **P1a(动态交换)**:swap 型 bundle D2D 流水 + 两阶段提交(MRV1);
- **P1b(动态冗余 + 分流)**:fill/evict 流水 + `SplitTable` 快环(类型加权分流、EWMA、闭环记账),与 P1a 共用提交协议;
- **P2(增强)**:节点内冗余(`is_node_redundant`,一机多 die)、FlashLB 式多轮增量、分流权重的自适应刷新频率(快环感知的时间常数调节)、MRV2 对接。
