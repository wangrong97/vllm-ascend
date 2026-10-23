# vllm-ascend `feat/vq2a8_v0.1`:MoE 专家"预取"与"专家替换"机制分析

> 分析对象:`/mnt/share/w00608002/work/eplb/vllm-ascend-vq2a8`(分支 `feat/vq2a8_v0.1`)
> 姊妹文档:[W2A8_VQ2A8_GS32_代码流程分析.md](W2A8_VQ2A8_GS32_代码流程分析.md)(量化与混精主流程)

---

## 目录

1. [结论摘要](#1-结论摘要)
2. [机制全景](#2-机制全景)
3. [华为动态 EPLB 子系统(vllm_ascend/eplb/)详解](#3-华为动态-eplb-子系统vllm_ascendeplb详解)
4. [vLLM 上游 EPLB 集成路径(MRV2)](#4-vllm-上游-eplb-集成路径mrv2)
5. [通用权重预取卸载(AscendPrefetchOffloader)](#5-通用权重预取卸载ascendprefetchoffloader)
6. [VQ2A8/GS32 与专家迁移机制的关系](#6-vq2a8gs32-与专家迁移机制的关系)
7. [语义澄清:什么是"预取"、什么是"专家替换"](#7-语义澄清什么是预取什么是专家替换)
8. [关键文件索引](#8-关键文件索引)

---

## 1. 结论摘要

当前分支中**没有**名为"专家预取(expert prefetch)"或"专家近似替换(expert approximation)"的独立特性;实际存在的是以下四套机制,其中第 1 套是"专家权重搬运 + 槽位替换"的完整实现:

| # | 机制 | 位置 | 与两个关键词的对应 |
| --- | --- | --- | --- |
| 1 | **华为动态 EPLB 子系统**(D2D 专家权重迁移) | `vllm_ascend/eplb/`(3331 行,仅 MRV1) | **"预取"**:逐层 P2P 传输与推理 forward 重叠(提前发出下一层传输);**"替换"**:新专家权重 D2D 拷入并覆盖本地旧槽位(`local_expert_to_replace`) |
| 2 | vLLM 上游 EPLB 集成(冗余副本重映射) | `AscendEPLBController`/`AscendEplbState` + `ops/fused_moe/eplb.py` | "替换"的逻辑层:逻辑专家→物理专家/副本的路由表重映射 |
| 3 | 层粒度权重 CPU 卸载+预取 | `AscendPrefetchOffloader` | "预取"的通用形态:`prefetch_step` 层提前回搬,**非专家粒度** |
| 4 | VQ2A8 相关映射/替换小机制 | `tid2eid` 哈希表、profiling 随机路由 | 静态映射表与测试用路由替换,均非权重搬运 |

**"近似替换"澄清**:代码中不存在"用低精度/共享/近似专家在运行时数值近似另一个专家"的机制。VQ2(W2)/MXFP4(W4)混精虽在数值上是专家权重的低比特近似,但它是 **checkpoint 离线决定、加载时固定**的,与运行时 EPLB 替换无关,且二者在本分支**互斥**(见第 6 节)。

## 2. 机制全景

```
┌─ MRV1(vllm_ascend/worker/model_runner_v1.py:495-515)
│    dynamic_eplb=True 时启动:
│    D2DExpertWeightLoader + EplbProcess(spawn 子进程) + EplbUpdator
│    forward_before() ── 生成并异步发出本层专家 P2P 传输   ← 传输与 forward 重叠
│    forward_end()   ── 等待完成 → 更新 expert_map/log2phy/权重槽位
│
├─ MRV2(worker/v2/eplb.py)
│    AscendEPLBController(继承上游 EPLBController)
│    AscendEplbState.rearrange() → 上游 rebalance → 刷新 graph-stable 副本路由表
│    (routed_experts.py:49:MRV2 明确不使用华为 D2D 子系统)
│
├─ 通用预取(model_executor/offloader/prefetch.py)
│    AscendPrefetchOffloader:层组 CPU 卸载,提前 prefetch_step 层回搬静态 buffer(NZ 格式)
│
└─ VQ2A8/GS32(quantization/methods/w2a8/)
     supports_eplb=False;offload/EPLB/冗余专家全部禁止;权重常驻设备
     hash 层 tid2eid:静态 token→库存专家映射表(模型原生设计,非运行时替换)
```

## 3. 华为动态 EPLB 子系统(vllm_ascend/eplb/)详解

华为版权的独立子系统(`Copyright (c) 2025 Huawei Technologies Co., Ltd`),通过 `ascend_config.eplb_config` 配置,**仅 Model Runner V1 启用**。

### 3.1 配置(ascend_config.py:116-161,EplbConfig)

| 配置项 | 默认 | 说明 |
| --- | --- | --- |
| `dynamic_eplb` | false | 开关;需环境变量 `DYNAMIC_EPLB=true`(或 `EXPERT_MAP_RECORD=true`) |
| `expert_map_path` | null | 静态初始专家映射 JSON(设置即隐含 dynamic_eplb) |
| `expert_map_record_path` | null | 周期结束时导出专家映射 JSON |
| `expert_heat_collection_interval` | 600 | 热度采集步数 |
| `algorithm_execution_interval` | 50 | 算法执行(等待规划)步数 |
| `num_redundant_experts` | 0 | 冗余专家数(热专家副本) |
| `eplb_policy_type` | 2 | 重排策略:0 Random / 1 DefaultEplb / 2 SwiftBalanceEplb / 3 FlashLB |
| `eplb_heat_collection_stage` | all | 采集阶段过滤(all/prefill/decode;PD 合设时只在单阶段采集) |
| `load_collection_phase` | all | MRV2 专用:上游负载窗口的阶段过滤 |

一个完整 EPLB 周期 = `expert_heat_collection_interval + algorithm_execution_interval + num_moe_layers` 步(权重更新每层一步)。

### 3.2 三阶段周期与推理的重叠(EplbUpdator,eplb_updator.py)

调用点:`model_runner_v1.py` 的 forward 路径(`forward_before` 在 2363/3847,`forward_end` 在 2466/2792/4123):

```
迭代计数 cur_iterations
 ├─ [0, heat_interval): 热度采集。MoE 层每步把路由命中累计进 layer.moe_load
 │     └─ 第 heat_interval-1 步 forward_end:
 │          compute_and_set_moe_load():all_gather 各 rank 的 moe_load → shared_dict
 │          wakeup_eplb_worker():planner_q.put(1) 唤醒规划子进程
 ├─ [heat_interval, heat_interval+algo_interval): 算法执行(子进程并行规划,推理不中断)
 │     └─ 第 heat+algo-1 步 forward_before:block_update_q.get() 取回逐层更新计划
 ├─ 之后 num_moe_layers 步:逐层权重更新流水(每步一层)
 │     ├─ forward_before:取本层 (send_info, recv_info, updated_expert_map,
 │     │    log2phy_map, layer_id) → generate_expert_d2d_transfer_task()
 │     │    → asyn_expert_weight_transfer():dist.batch_isend_irecv 异步发出
 │     │       ★ 传输在当前层 forward 执行期间于通信流上并行 —— 预取/重叠语义
 │     └─ forward_end:update_expert_map_and_weight(reqs):
 │          req.wait() → do_update_expert_map / do_update_log2phy_map
 │          → do_update_expert_weight(把接收 buffer 拷入被替换槽位)
 └─ 周期结束:导出 expert_map(可选)→ clear_all_moe_loads() → 计数清零
```

### 3.3 D2DExpertWeightLoader(core/eplb_device_transfer_loader.py)

三态状态机(`WAITING → READY → TRANSFERRING → WAITING`):

- `generate_expert_d2d_transfer_task(send_info, recv_info, updated_expert_map, layer_id)`:
  - 对每个 `(dst_rank, expert_id)` 发送项:从 `adaptor.expert_param_per_layer[layer][local_id]` 取**逐专家张量列表**,构造 `P2POp(isend, src_tensor, dst_rank)`;
  - 对每个接收项:`P2POp(irecv, buffer_tensor, src_rank)` 收到预分配的 `buffer_tensor_list` 槽,并记录 `(local_expert_to_replace, buffer_tensor_id)` —— 新映射中该全局专家应落到的本地槽位;
  - 同一时间只接受一个在途更新任务(否则 `warning_once` 拒绝)。
- `asyn_expert_weight_transfer(reqs)`:`dist.batch_isend_irecv` 批量异步收发。
- `update_expert_map_and_weight(reqs)`:等待全部 P2P 完成后依次更新映射与权重;最后一层完成时打印完整周期日志。

通信域:`get_dynamic_eplb_group()`(专用 dynamic EPLB 进程组,HCCL buffer 大小单独配置,utils.py:995)。

### 3.4 VllmEplbAdaptor(adaptor/vllm_adaptor.py)

- **层注册**:`VllmEplbAdaptor.register_layer(layer)` 在 MoE 层初始化时调用(仅真实层,`PPMissingLayer` 不注册,天然只含本 PP rank 的层)。
- **权重命名表 `EPLB_EXPERT_WEIGHT_NAMES`**:按 `(QuantType, enable_fused_mc2)` 列出逐专家参数名(`w13_weight_list/w2_weight_list` + 各 scale/bias)。支持 NONE / W8A8 / W4A8 / W4A4MXFP / W8A8MXFP;**W4A8MXFP 直接 raise 不支持**;**VQ2A8 不在表内**(无法配合,见第 6 节)。
- **权重形态**:dynamic_eplb(或 megamoe)时,`AscendRoutedExperts` 把堆叠 `w13_weight/w2_weight [E,N,K]` `unbind` 成**逐专家独立张量列表**(`*_weight_list`,routed_experts.py:96-100)—— 这是逐专家 P2P 传输的前提。
- **接收缓冲**:`init_buffer_tensor()` 为每个权重键 × 每个 `buffer_id`(= num_local_experts)`torch.empty_like` 预分配,irecv 直接收入,避免运行时分配。
- **替换落点**:`do_update_expert_weight(layer_id, local_expert_to_replace, buffer_tensor_id)` —— 把 buffer 中的新专家权重 `copy_` 覆盖本地第 `local_expert_to_replace` 个专家的各张量;同时 `do_update_expert_map`(expert_map)、`do_update_log2phy_map`(逻辑→物理映射)。
- 其余接口:`get_rank_expert_workload()`(汇聚各层 `moe_load`)、`get_global_expert_map()`、`_export_tensor_to_file()`(expert_map_record_path 导出)。

### 3.5 EplbWorker / EplbProcess(core/eplb_worker.py)

`EplbProcess`:spawn 守护子进程,循环 `planner_q.get()` → `worker.do_update()` → 结果放入 `block_update_q`(maxsize=1)。与主进程通过 `Manager().dict()` 共享 `expert_maps`/`moe_load`。

`EplbWorker.do_update()` 规划流程:

1. 首轮从 shared_dict 读初始 expert_maps;之后每轮读取最新 `moe_load`;
2. `calculate_rebalance_experts`:`policy.rebalance_experts(old_placement, load_info)` 计算新放置;
3. `check_expert_placement` 防御性校验:新放置不得丢失逻辑专家、同一 NPU 内不得放置同一逻辑专家的多个副本、不允许"NPU 内部自我搬移"(检测到即整层回退旧放置);
4. `compose_expert_update_info_greedy`(逐层生成器):
   - 本层映射无变化 → 直接 yield 空更新;
   - `recv` = 新表非 -1 且旧表为 -1 的位置(要接收的专家);`send` = 旧表非 -1 且新表为 -1(要发出的专家);
   - 贪心配对:若目标专家没有 rank 外发它,则从任一仍持有该专家的 rank 复制(冗余副本语义);TODO 注释指出未来可按节点内/节点间代价选择源;
5. `pack_update_info`:只保留**本 rank** 的 send/recv 列表 + 本 rank 的新 expert_map + `generate_log2phy_map`(含 tp_size 维度),逐层打包。

Rank 0 额外计算并记录专家热度不平衡度(均值/最大值/逐层列表,ms-service-metric 上报)。

### 3.6 重排策略(core/policy/)

| type | 类 | 说明 |
| --- | --- | --- |
| 0 | `RandomLoadBalance` | 测试用:随机交换 NPU1/NPU3 的最后一个物理专家 |
| 1 | `DefaultEplb` | 基于当前负载的整体专家替换;`add_redundant` 将逻辑专家负载聚合,`original_compute_balanced_pack_redundancy` 把**热专家分裂成冗余副本**(装箱) |
| 2 | `SwiftBalanceEplb`(默认,751 行) | Dynamic EPLB V2:**约束交换次数**的专家替换(限制单轮搬移量,配合逐层 D2D 流水) |
| 3 | `FlashLB`(1026 行) | 联合优化 + 多轮增强(Multi-Shot)+ 增量调整;`eplb_policy_type==3` 时 updator 走 multi_stage 路径,子进程启动时 `warm_up()` |

工厂注释的硬约束(V2 策略):存在冗余专家时,**一块 NPU 只能放一个冗余副本且其物理专家索引必须为 0**。

## 4. vLLM 上游 EPLB 集成路径(MRV2)

另一条独立路径,复用 vllm 主库的 `EplbState/EplbController`,Ascend 侧扩展:

- `AscendEPLBController`(worker/v2/eplb.py):`prepare_forward` 按 `load_collection_phase`(all/prefill/decode)过滤负载记录;`setup_from_mapping` 从新映射重建状态(动态重排 / 静态 expert_map 的应用入口)。
- `AscendEplbState`(distributed/eplb/state.py):
  - `rearrange()`:门控(EP 组 all_reduce "是否有新负载",无则跳过)→ 上游 `rearrange`(负载窗口 → `policy.rebalance_experts` → 权重重绑定)→ `refresh_model_routing_tables()`;
  - `AscendEplbLayerState.expert_replica_routing_table`:**graph-stable** 的逻辑→物理副本路由表 —— 重排后形状不变则原地 `copy_`,保证 ACL 图捕获的地址稳定;
- `ops/fused_moe/eplb.py`:
  - `build_expert_replica_routing_table(logical_to_physical_map, logical_replica_count, ep_rank)`:副本选择 `(row + ep_rank + logical_id) % replica_count`,各 rank 轮询错开冗余副本;
  - `map_to_physical(topk_ids, table)`:把 topk 逻辑专家 ID 查表映射为物理专家 ID(分块 gather,任意长度);
  - `map_to_physical_and_record(...)`:映射同时把命中累计进 `expert_load_view`(负载统计,另有 Triton 版本 `map_to_physical_and_record_triton`);
- MRV1 的 `dynamic_eplb` 在 MRV2 下恒为 False(routed_experts.py:49),两条路径不混用。

## 5. 通用权重预取卸载(AscendPrefetchOffloader)

`vllm_ascend/model_executor/offloader/prefetch.py`(继承 vllm 0.27.1 `PrefetchOffloader`):

- 机制:**decoder layer 粒度**的 CPU 卸载 —— 每 `group_size` 层为一组,组内卸载后 `num_in_group` 层到 CPU(锁页);forward hook 在执行第 `i` 层时**提前 `prefetch_step` 层**把权重异步回搬进循环复用的静态 buffer 池(`StaticBufferPool`,slot 数 = prefetch_step);`join_after_forward` 尾部环绕衔接首层。
- Ascend 扩展:NPU 上 `FRACTAL_NZ` 格式权重的静态 buffer 也保持 NZ(`_format_static_buffers_for_nz`,通过包装 `process_weights_after_loading` 在加载后探测格式并打标 `mark_prefetch_offload_nz_weight`);`wrap_modules` 临时替换上游 `_ModuleOffloader` 为 Ascend 版。
- 挂载:`model_runner_v1.py:321` `set_offloader(create_offloader(self.offload_config))`(vllm 上游 offload_config)。
- **定位澄清**:该机制按**层**搬运(专家权重只是层参数的一部分),不存在"按路由热度选择性预取某几个专家"的专家粒度逻辑;且 VQ2A8/GS32 下被整体禁用。

## 6. VQ2A8/GS32 与专家迁移机制的关系

**互斥,双向不兼容**:

1. 方向一(EPLB 不支持 VQ2A8):`EPLB_EXPERT_WEIGHT_NAMES` 不含 `QuantType.VQ2A8`;该表假设专家权重是"可按专家一维切片的同构张量列表"(w13/w2 + scale),而 GS32 是**按 (投影×量化类型) 分 bank 的 typed 常驻结构**(`w13_packed_zn/pair_lut/row_scale_bits/perm`、`mxfp4_weight/scale`),且每专家带独立 `perm`/scale 布局,无法用逐专家同构 unbind + P2P 拷贝迁移。
2. 方向二(VQ2A8 禁用 EPLB/offload):
   - `AscendGS32FusedMoEScheme.supports_eplb = False`;
   - `validate_vq2a8_config()`(vq2a8_config.py:92-99)拒绝 `enable_eplb`、`use_sequence_parallel_moe`、`eplb_config.num_redundant_experts/dynamic_eplb/expert_map_path`;
   - EP 校验要求 `expert_placement_strategy == "linear"`、专家数整除、无冗余专家;
   - offload/sleep 全禁("native banks require stable resident parameters"):图捕获与 validity 机制都假设 bank 地址常驻不变;
   - `moe_mlp.py:847` VQ2A8 分支显式拒绝 `dynamic_eplb`。
3. GS32 内部的"映射/替换"类小机制(注意区分):
   - **hash 层 `tid2eid` 表**:`[vocab, top_k]` 的静态 token→专家映射(DeepSeek-V4 原生设计,hash 层专家库存可稀疏),加载时校验表值 ∈ 该层库存 —— 是"映射表",不是运行时可变的替换;
   - **profiling 强制负载均衡**(routed_experts.py:485):profile/dummy run 时把 `topk_ids` 随机改写为本 rank 驻留专家 —— 仅替换**路由**用于测试,不搬权重;
   - **CI 的 "missing MXFP4 to VQ2 experts"**:测试对 checkpoint 中 VQ2↔MXFP4 专家类型对覆盖率的报告,与运行时替换无关。

## 7. 语义澄清:什么是"预取"、什么是"专家替换"

| 说法 | 本分支的真实对应 | 准确描述 |
| --- | --- | --- |
| "专家预取" | ① EPLB D2D 流水:本层 forward 期间异步收发**下一层待更新专家**的权重(`forward_before` 发出、`forward_end` 才等待应用);② `AscendPrefetchOffloader`:提前 `prefetch_step` **层**回搬(层粒度,非专家) | 没有按"路由热度预测即将命中的专家"的预取;预取 = 传输与计算重叠 + 提前搬运 |
| "专家(近似)替换" | EPLB 的槽位替换:新专家权重 D2D 拷入 `local_expert_to_replace` 槽位,`log2phy/expert_map` 重映射;冗余专家 = 热专家多副本 | 替换是**整权重搬运 + 逻辑/物理映射更新**,不是数值近似;不存在"低精度版本近似顶替原专家"的运行时机制(混精是离线 checkpoint 决定的) |
| (对照)6.2 节 prefill 预热 | 启动时合成输入的编译预热 | 与数据/专家搬运均无关 |

## 8. 关键文件索引

| 文件 | 行数 | 角色 |
| --- | --- | --- |
| `vllm_ascend/eplb/eplb_updator.py` | 192 | 三阶段周期驱动:`forward_before/forward_end`、热度汇聚、唤醒子进程 |
| `vllm_ascend/eplb/core/eplb_device_transfer_loader.py` | 138 | D2DExpertWeightLoader:逐专家 P2P 任务构造/异步收发/映射与槽位更新 |
| `vllm_ascend/eplb/adaptor/vllm_adaptor.py` | 222 | MoE 层注册、逐专家权重命名表(按量化类型)、接收 buffer、`do_update_expert_weight` |
| `vllm_ascend/eplb/core/eplb_worker.py` | 410 | EplbWorker(规划/校验/贪心 send-recv 配对)+ EplbProcess(spawn 子进程) |
| `vllm_ascend/eplb/core/eplb_utils.py` | 145 | `generate_log2phy_map` 等 |
| `vllm_ascend/eplb/core/policy/policy_default_eplb.py` | 350 | 整体替换 + 冗余专家装箱分裂 |
| `vllm_ascend/eplb/core/policy/policy_swift_balancer.py` | 751 | 默认策略:受约束交换次数的替换(V2) |
| `vllm_ascend/eplb/core/policy/policy_flashlb.py` | 1026 | FlashLB:联合优化/多轮增强/增量调整 |
| `vllm_ascend/ops/fused_moe/eplb.py` | 138 | 副本路由表构建、`map_to_physical(_and_record)`(上游 EPLB 路径) |
| `vllm_ascend/distributed/eplb/state.py` | 179 | AscendEplbState:rearrange 门控、graph-stable 路由表刷新 |
| `vllm_ascend/worker/v2/eplb.py` | 97 | AscendEPLBController(MRV2,阶段过滤负载采集) |
| `vllm_ascend/model_executor/offloader/prefetch.py` | 201 | AscendPrefetchOffloader(层粒度 CPU 卸载预取 + NZ buffer) |
| `vllm_ascend/worker/model_runner_v1.py:495-515,2363,2466` | — | EplbUpdator 启动与 forward 挂接 |
| `vllm_ascend/ops/fused_moe/routed_experts.py:96-100,343-368` | — | dynamic_eplb 时权重 unbind 成逐专家列表;`moe_load/log2phy` 状态 |
| `vllm_ascend/ascend_config.py:116-161` | — | EplbConfig(周期/策略/冗余专家/采集阶段) |
