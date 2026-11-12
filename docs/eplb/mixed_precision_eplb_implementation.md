# 混精 EPLB(VQ2A8 × 冗余专家 × token 分流)实现说明:修改逻辑、调用链与开关

> 对应设计文档:`/mnt/share/w00608002/work/eplb/EPLB混精结合_规划与权重搬运算法设计.md`
> 代码基线:`feat/vq2a8_v0.1`;实现包:`vllm_ascend/eplb/mp/`;测试:`tests/eplb_mp/`(45 用例)
> 本文回答三个问题:**改了什么、怎么被调用、开关如何让修改生效/失效**。

---

## 目录

1. [修改总览](#1-修改总览)
2. [开关体系与解析链](#2-开关体系与解析链)
3. [off 基线路径(默认,零行为差异)](#3-off-基线路径默认零行为差异)
4. [生效路径详解(六条链)](#4-生效路径详解六条链)
5. [uniform 变体(不考虑量化位宽)](#5-uniform-变体不考虑量化位宽)
6. [端到端调用链图](#6-端到端调用链图)
7. [如何启用](#7-如何启用)
8. [测试与验证](#8-测试与验证)
9. [已知边界与接线 TODO](#9-已知边界与接线-todo)

---

## 1. 修改总览

### 1.1 新增文件(`vllm_ascend/eplb/mp/`,约 1600 行)

| 文件 | 核心对象 | 职责 |
| --- | --- | --- |
| `config.py` | `MPEplbOptions` / `get_mp_eplb_options()` / `MP_EPLB_MODES` / `OFF_OPTIONS` | 开关与参数解析校验(五种模式) |
| `cost_model.py` | `CostTable.defaults()/.uniform()`、`amortized_unit_cost()`、`window_load()`、`expert_bytes()`/`bundle_bytes()` | 共享代价模型(`load`=token 量,`c_type`=单位代价) |
| `placement.py` | `BankSlotTable`、`build_linear_baseline()` | 放置真值主机镜像:容量预留、fill/evict/swap 值级变更、设备索引物化(`device_index()`) |
| `planner.py` | `ReplicaAllocator`(阶段 A)、`TCSBPlanner`(阶段 B)、`ExpertMove`、`MigrationPlan`、`primary_rank()` | 冗余分配(摊派贪心)与同类型交换规划 |
| `split_table.py` | `SplitTable`、`weighted_pick()` | 快环分流:route/weight/cdf/ewma 四张固定形状表 + 确定性加权哈希选择 |
| `migrator.py` | `BundleSpec`、`P2POp`、`CommBackend`/`InMemoryCommBackend`、`MigrationExecutor`(`MoveStage`) | bundle 级搬运:evict→swap→fill 两阶段提交状态机(通信后端可注入) |
| `controller.py` | `MPEplbController` | 慢环周期状态机(heat→plan→逐层 migrate→EWMA 快照刷新);`run_static_once()` |
| `__init__.py` | — | 包导出 |

### 1.2 修改的既有文件(全部修改点位于开关之后)

| 文件 | 修改点 | 性质 |
| --- | --- | --- |
| `quantization/methods/w2a8/vq2a8_gs32_format.py` | `validate_gs32_options()`:optional 集合增加 `"eplb"`/`"eplb_options"` 两键 | 放行配置键 |
| `quantization/methods/w2a8/vq2a8_gs32.py`(scheme) | ①`__init__` 调 `get_mp_eplb_options()` 存 `self.mp_eplb`;②`get_weight()` 容量预留(spare 尾槽);③`process_weights_after_loading()` 透传 `mp_eplb` 给 `GS32ExpertWeights` | 开关读取 + 资源预留 |
| `ops/fused_moe/vq2a8_gs32.py`(GS32ExpertWeights) | ①`__init__` 新参 `mp_eplb`,派生 `eplb_enabled`,创建 `moe_load` 与空槽设备索引、plan 按容量;②`apply()` 输入校验后单点记账;③类尾新增 `commit_fill/commit_evict/commit_swap` 三个 P2 提交钩子 | 运行时集成 |

### 1.3 测试(`tests/eplb_mp/`,45 用例)

`test_mp_unit.py`(开关/代价/放置)、`test_mp_algorithms.py`(规划器/分流表/uniform)、`test_mp_cycle.py`(周期/提交协议/GS32 门控)、`conftest.py`(未构建源码树的 `_build_info` stub)。

---

## 2. 开关体系与解析链

### 2.1 五种模式

```
vq2a8.eplb = "off"             # 默认。所有新增代码不执行,运行路径与基线一致
             | "static"         # 类型感知:启动期一次性规划(初始冗余分配)
             | "dynamic"        # 类型感知:周期 swap/fill/evict + 分流
             | "static_uniform"     # 不考虑量化位宽的两种变体
             | "dynamic_uniform"
```

`MPEplbOptions` 派生位:`enabled = mode != "off"`(总闸)、`dynamic = mode.startswith("dynamic")`、`uniform = mode.endswith("_uniform")`。子选项 `vq2a8.eplb_options = {K, R_vq2, R_mx4, n_stage, tau_token, tau_slow, tau_revoke, revoke_hysteresis, ewma_lambda, w_lo, w_hi, replica_replan_every, improvement_eps}`(全部有默认值,未知键/越界值硬拒绝)。

### 2.2 解析链(谁读开关、何时读)

```
run.sh / vllm serve
  --additional_config '{"vq2a8": {"gs32_library":..., "gs32_binary":...,
                                   "eplb": "dynamic", "eplb_options": {...}}}'
        │
        ▼  vllm_config.additional_config["vq2a8"]
validate_vq2a8_config()                       # configs/vq2a8_config.py(未改动)
  └─ validate_gs32_options()                  # format.py:optional 集合已放行 eplb/eplb_options
        │  返回 dict(options)——含 eplb 键,原样下传
        ▼
AscendGS32FusedMoEScheme.__init__(native_options=options)
  └─ self.mp_eplb = get_mp_eplb_options(self.native_options)   # ★ 第一次解析(模块 config.py)
        │
        ├─▶ get_weight():                 if enabled → spare = replica_slots[qtype] // ep_size
        │
        └─▶ process_weights_after_loading():
              GS32ExpertWeights(layer, local_spec, native_options, mp_eplb=self.mp_eplb)
                  └─ self.eplb_enabled = self.mp_eplb.enabled    # ★ 运行时总闸(唯一判断点)
```

要点:**开关只解析两次**(scheme 一次、GS32 一次,后者由前者透传,保证同一对象);运行期所有分支只判断 `self.eplb_enabled` / `self.mp_eplb.*`,不再触碰配置字符串。

---

## 3. off 基线路径(默认,零行为差异)

`mode="off"` 时 `enabled=False`,逐点对照:

| 集成点 | off 行为 | 基线对照 |
| --- | --- | --- |
| `scheme.get_weight()` | `spare = replica_slots[qtype] // ep_size` 中 `replica_slots` 仍被读取但 **`get_weight` 内 spare 计算走 `if self.mp_eplb...`** —— off 时 spare=0,`torch.empty((len(ids)+0, ...))` | 与基线完全一致 |
| `GS32ExpertWeights.__init__` | `moe_load=None`;设备索引分支条件 `execution=="full_graph" or self.eplb_enabled` 不新增触发;spare=0 → `expert_ids_tensor` 无尾槽、plan 组数不变 | 与基线完全一致 |
| `GS32ExpertWeights.apply()` | 记账块被 `if self.eplb_enabled:` 整体包裹,不执行 | 与基线完全一致 |
| `commit_*` 三个方法 | 新增方法,off 时无人调用(纯死代码) | 零开销 |
| `MPEplbController` | **不会被实例化**(实例化由接线层按 `enabled` 决定;单测中显式构造) | 不存在 |

单测 `TestGS32SwitchGating::test_off_keeps_baseline_attributes` 断言了前两条(moe_load 为 None、无 expert_ids_tensor);既有契约测试(`tests/ut/quantization/test_vq2a8_gs32_staged/layout/fixtures` 等 21 例)回归通过。

---

## 4. 生效路径详解(六条链)

### 4.1 链①:静态资源预留(加载期,开关 on 即生效)

```
AscendGS32FusedMoEScheme.get_weight(num_experts, inter, hidden, dtype)   # methods/w2a8/vq2a8_gs32.py
  ├─ self.placement = gs32_expert_placement(...)          # 基线:线性区间(未改动)
  ├─ ★ spare = self.mp_eplb.replica_slots[quant_type] // placement.ep_size
  ├─ result[name] = torch.empty((len(ids) + spare, *shape))   # bank 第 0 维 = 活跃 + 空槽
  └─ parameter_experts/loaded_experts 仍按 len(ids) 记录(空槽不参与加载校验)
```

随后 `process_weights_after_loading()` 把 `mp_eplb` 透传,进入运行时构建:

```
GS32ExpertWeights.__init__(layer, layer_spec, options, mp_eplb=...)   # ops/fused_moe/vq2a8_gs32.py
  ├─ self.mp_eplb / self.eplb_enabled
  ├─ ★ self.moe_load = torch.zeros(num_experts, int64, device)        # 仅 enabled
  └─ if execution == "full_graph" or eplb_enabled:                     # ★ 分支条件扩大
       for bank in banks:
         active = tensor(expert_ids); ★ spare>0 时 cat([-1]*spare)     # 空槽哨兵
         bank["spare_slots"] = spare
         bank["expert_ids_tensor"] = active(含尾槽 -1)
         bank["logical_slots"][active_ids] = arange(len(active))       # 单值映射(活跃专家)
         plan(groups = len(active_ids) + spare)                        # ★ 按容量预建(仅 full_graph)
```

设计依据:容量恒定 → 任何 move 不改变张量形状/组数 → plan 与 decode 图零重捕获(设计文档 D/I3);空槽 `-1` 哨兵的 counts=0 路径运行时已支持。

### 4.2 链②:负载采集(每步 forward,单点)

```
GS32ExpertWeights.apply(hidden, group_list, group_list_type, swiglu_limit)
  ├─(输入契约校验——基线代码)
  ├─ ★ if self.eplb_enabled:                       # 唯一记账点,位于所有执行分支之前
  │     counts = group_list.to(device, int64).clamp_min(0)
  │     type==0 时差分(prefix → 计数)
  │     self.moe_load[:n].add_(counts)
  └─(serial/batched/fused/full_graph 各分支——基线代码,未改动)
```

`moe_load` 即设计文档 C.5 的"原始量"(槽位逻辑专家的 token 行数);类型加权时间份额(EWMA)由慢环侧消费时折算,不在热路径乘斜率。

### 4.3 链③:慢环周期状态机(接线层驱动)

```
接线层(MRV1 forward 钩子,V1 由测试/上层显式驱动)
  ├─ controller.on_step_begin()      # PHASE_MIGRATE 时:prepare_layer + issue(异步 P2P)
  └─ controller.on_step_end()        # 步进计数;heat 满 → _run_planning;migrate 中 → commit_layer
MPEplbController:
  PHASE_HEAT    _heat_steps() 步内累计 record_step()(消费链②的 moe_load 口径)
  _run_planning():
     loads/unit_costs ← _snapshot_inputs()             # cost_model.amortized_unit_cost
     working = deepcopy(placement)                     # 试算副本
     每 replica_replan_every 周期:
        moves_a, replica_alloc, revoke_history = ReplicaAllocator.allocate(working, ...)
     plan = TCSBPlanner.plan(working, ...)             # 试算并产出 MigrationPlan
     moves 按 index % num_layers 分层                   # 逐层流水
     PHASE_MIGRATE
  每层:executor.prepare_layer(evict→swap→fill 排序) → comm.issue(P2POp 列表)
       executor.commit_layer(wait → P1/P2 应用)         # 见链④
  _finish_cycle():SplitTable.refresh_weights(EWMA 快照) → 清窗口 → PHASE_HEAT
static 模式:run_static_once() = 同一套规划 + 立即逐层 commit(无周期)
```

### 4.4 链④:搬运与两阶段提交

```
MigrationExecutor(migrator.py)
  prepare_layer(layer, moves, staging_capacity)      # MoveStage.WAITING→READY
     排序 evict → swap → fill(字节先释放后占用,D5)
     evict 无传输;swap/fill 生成 P2POp(描述:peer/expert/projection/field/staging_index)
  issue(ops) → CommBackend.issue                     # READY→TRANSFERRING(生产:HCCL batch_isend_irecv)
  commit_layer(layer)                                # →WAITING
     comm.wait(全部 P2P 完成)
     逐 move._apply():
       P1(数据段):生产环境 staging→槽位 copy_;单测中直接落主机镜像
       P2(提交段):placement.swap/fill/evict(主机镜像)
                   split 表跟随(_split_add/_remove)
                   device_index_writer 回调 → 生产环境调 GS32 commit_*(链⑤)
       失败 → 该 move 计入 skipped(层粒度回滚:P1 逆拷贝,P2 不动)
```

### 4.5 链⑤:设备索引 P2 提交钩子(GS32 侧,原地值刷新)

```
GS32ExpertWeights.commit_fill(qtype, slot, expert)     # 新增方法
  两投影 bank 的 expert_ids_tensor[slot] = expert(-1 占用校验)
  logical_slots[expert] = slot;resident_mask[expert] = True
GS32ExpertWeights.commit_evict(qtype, slot, expert)
  expert_ids_tensor[slot] = -1;logical_slots[expert] = -1;mask=False(权重字节保留)
GS32ExpertWeights.commit_swap(qtype, slot, incoming)
  返回被换出专家;双专家的 logical_slots/mask 同点翻转
```

全部为**固定形状原地写值** —— 已捕获 decode 图地址不变、内容生效(graph-stable,设计 D4)。

### 4.6 链⑥:快环分流表(周期边界刷新,步级只读)

```
SplitTable.split_table.py
  route[e]:副本 rank 列表(P2 提交时 add/remove_replica 维护)
  refresh_weights(rank_time_shares):                 # 仅慢环 _finish_cycle 调用
     inverses = 1/(L[r]+ε);raw = clip(v/mean, w_lo, w_hi)   # clip 为相对倍数
     weight[e] = normalized(raw)
  materialize():route/weight/cdf 三张固定形状张量(设备侧)
  weighted_pick(cdf_row, route_row, seq_id, expert_id):
     h = SplitMix64((seq<<32)^expert) / 2^53         # 纯函数,确定性
     cdf 落点查区间 → 平局按 rank 序号破平
```

EWMA(`record_rank_share`)只作慢环输入,**绝不进入选择路径**(E1 认领唯一;单测断言改 EWMA 不改 pick)。

---

## 5. uniform 变体(不考虑量化位宽)

同一管线、三处算法差异,全部由 `options.uniform` 门控:

| 环节 | typed(`static`/`dynamic`) | uniform(`*_uniform`) | 门控位置 |
| --- | --- | --- | --- |
| 代价表 | `CostTable.defaults()`(α_vq2=1/α_mx4=1.8 + launch) | `CostTable.uniform()`(每投影 0.5、launch 0 → **c_type≡1.0**,均衡=token 条数) | `controller.__init__` |
| 冗余价值 | `value/bytes` 密度(VQ2 复制性价比 1.9×) | 纯 `value = load/(replicas+1)` | `ReplicaAllocator.allocate` 的 `if opts.uniform` 分支 |
| 冗余预算 | R_vq2/R_mx4 两池分流(防一类占满) | **单池**(R_vq2+R_mx4 总量);某类型无空槽时跳过该专家不阻塞(`skipped` 集) | 同上 |
| 字节核算 | `bind_geometry` 注入真实 bundle bytes,`B_r` 硬约束 | bytes≡1、预算=槽数 —— 字节约束退化为容量约束(I3 天然满足) | `controller.bind_geometry` |

**物理边界两模式一致**:bank 类型刚性(VQ2 副本只落 vq2 空槽)、swap 同类型、`logical_slots` 单值映射 —— 这是运行时正确性,与算法是否感知位宽无关。

---

## 6. 端到端调用链图

```
                        ┌────────────── 加载期(开关 on)──────────────┐
additional_config.vq2a8 ─▶ validate_gs32_options(放行 eplb 键)
   ─▶ AscendGS32FusedMoEScheme.__init__ ─ get_mp_eplb_options ──▶ mp_eplb
   ─▶ get_weight(★spare 尾槽) ─▶ weight_loader(基线)
   ─▶ process_weights_after_loading ─▶ GS32ExpertWeights(mp_eplb=)
        ─▶ ★moe_load / ★空槽 expert_ids_tensor(-1) / ★plan(groups=capacity)
                        └───────────────────────────────────────────┘
                        ┌────────────── 每步 forward ────────────────┐
router/dispatcher(基线,未改) ─▶ GS32ExpertWeights.apply
   ─▶ ★if eplb_enabled: moe_load.add_(counts)        # 链②,唯一热路径新增
   ─▶ serial/batched/fused/full_graph 投影(基线)
                        └───────────────────────────────────────────┘
                        ┌────────────── 慢环周期(dynamic)────────────┐
on_step_end(heat 满) ─▶ _run_planning
   ├─ ReplicaAllocator.allocate(试算:evict 滞回 → typed 双池/uniform 单池 fill)
   ├─ TCSBPlanner.plan(同类型 swap,统一增益)
   └─ moves 分层 → 逐层 prepare_layer/issue/commit_layer
        ├─ P1 数据段(staging→槽位;生产 HCCL,测试 InMemory)
        ├─ P2 提交段:placement 镜像 + SplitTable.route + ★GS32.commit_fill/evict/swap
        └─ _finish_cycle:SplitTable.refresh_weights(EWMA 快照 → weight/cdf)
                        └───────────────────────────────────────────┘
                        ┌────────────── 快环(步级,只读表)────────────┐
weighted_pick(cdf, route, seq_id, e) ─▶ 副本 rank 认领(确定性、跨 rank 一致)
                        └───────────────────────────────────────────┘
static 模式:run_static_once() 一次性走"规划→逐层 commit→refresh_weights"
```

---

## 7. 如何启用

### 7.1 配置最小例

```json
--additional_config '{
  "vq2a8": {
    "gs32_library": "<abs>/libvq2a8_gs32_stage.so",
    "gs32_binary": "<abs>/mixed_w2w4_moe_ffn.o",
    "gs32_execution": "fused",
    "eplb": "dynamic",
    "eplb_options": {"K": 8, "R_vq2": 16, "R_mx4": 8, "n_stage": 2,
                      "w_lo": 0.2, "w_hi": 5.0, "ewma_lambda": 0.05}
  }
}'
```

### 7.2 模式选择

| 场景 | 推荐 |
| --- | --- |
| 回归基线 / 灰度发布 | `off` |
| 先验证规划收益(不在线搬运) | `static` |
| 在线完整能力(类型感知) | `dynamic` |
| 快速基线对比(不区分 VQ2/MXFP4) | `static_uniform` → `dynamic_uniform` |

PD 分离/dp4ep4 的启动脚本见仓库根 `run.sh`(开关变量 `MP_EPLB_MODE`;其与 VQ2A8 校验的兼容性冲突已在脚本头标注,见第 9 节)。

### 7.3 算子产物(开关 on 的前提)

`gs32_library/gs32_binary` 按 `docs/vq2a8/vq2a8_gs32_runtime.md` 构建;`off` 模式不需要它们(但 `validate_gs32_options` 仍要求这两个路径存在 —— 与基线一致)。

---

## 8. 测试与验证

```bash
# 本仓 CPU 单测(45 例;系统 python 3.11.6 + 借用 site-packages 的运行方式):
PYTHONPATH=/usr/local/python3.11.10/lib/python3.11/site-packages \
  /usr/bin/python3 -m pytest tests/eplb_mp -q

# 既有 VQ2A8 契约回归(off 不破坏基线):
PYTHONPATH=... /usr/bin/python3 -m pytest tests/ut/quantization/test_vq2a8_gs32_*.py -q
```

覆盖矩阵:开关解析(5 模式/非法拒绝/未知键/w_lo≥w_hi)、代价恒等式(`load·c_type = αΣ·tokens+βΣ·launches`)与 1.9× 字节比、放置镜像(容量/空槽/三类变更/设备索引)、ReplicaAllocator(最热优先/类型分流/C6/滞回/主副本保护/uniform 单池纯价值)、TCSB(同型交换/K 预算/EP=1 退化)、SplitTable(确定性/比例 ±5%/clip 相对倍数/EWMA 不进选择)、提交协议(顺序/层互斥/split 跟随)、GS32 门控(off 零属性、on 记账+容量、apply 仅 on 记账、commit 钩子刷新)。

---

## 9. 已知边界与接线 TODO

| # | 项 | 状态 |
| --- | --- | --- |
| 1 | 多 rank 真实 HCCL P2P(`CommBackend` 生产实现) | 接口就绪,`InMemoryCommBackend` 仅供测试 |
| 2 | dispatcher 非线性 `expert_map` 构造(设计 2.6-3) | `BankSlotTable.device_index()` 已产出三表,dispatcher 侧接线待做 |
| 3 | MRV1 `forward_before/forward_end` 钩子注册(设计 2.6-4/5) | `controller.on_step_begin/end` 接口就绪,挂载点待接 |
| 4 | 快环 `SplitTable` 查表插入 routed_experts 路由后(设计 2.6-4) | 算法与单测完成,管线插入待接 |
| 5 | PD 分离 / DP>1 形态 | `validate_vq2a8_config` 当前拒绝(`run.sh` 头部标注需放开的校验位置);运行时验证未做 |
| 6 | hash 层 | 按 C5 排除,白名单过滤在接线层实现 |

V1 已交付并验证的是**算法层全部逻辑 + 开关受控的 GS32 资源预留/记账/提交钩子**;第 1-4 项接线完成后即可在多卡服务环境端到端运行。
