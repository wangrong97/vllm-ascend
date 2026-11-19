# night3 夜间编排(GLM-5.3-Flash 量化链路实验)

- 位置: 2026-11-16 起统一收纳于 `tools/glm53flash/night3/`(原在仓库根)
- 日志: `output/glm53flash/night3_main.log`(主链)/ `night3_supervisor.log` / `night3_launch.log`
- 断点标记: `output/glm53flash/.night3_started`(跳过 01:00 等待)/ `.night3_finished`(全链完成)/ `.night3_halt`(熔断, 排障后 rm 再启)

## 文件

| 脚本 | 作用 |
| --- | --- |
| `night3_glm53flash.sh` | 主编排: 阶段链(已 done 秒过) 七o2A(dump+OSCAR 拟合)→七o2B(评测)→八A(FP8→KDA8 量化)→八B(dump+拟合)→八C(评测); 含 watchdog(90min+predictions 活跃检查)、8 卡空闲 gate、run 级断点续跑、dump_ok 防半截数据拟合、各段 dump 开关 unset |
| `night3_supervisor.sh` | 强化守护: 60s 重拉 + 30min 内 ≥3 次失败熔断 → autofix(`tools/glm53flash/autofix_server_fail.sh` 已知症状幂等补丁)→ 未知症状 touch .night3_halt |
| `launch_night3.sh` | 定时调度: 00:20 查 8 卡(两次确认), 空闲→立即启动(touch started), 被占→01:00 由 night3 原生等待兜底 |

## 用法

```bash
# 立即续跑(卡空闲时): 
setsid nohup bash /mnt/share/w00608002/work/vllm-ascend/tools/glm53flash/night3/night3_supervisor.sh > /dev/null 2>&1 &
# 定点启动(00:20 检查/01:00 兜底): 
setsid nohup bash /mnt/share/w00608002/work/vllm-ascend/tools/glm53flash/night3/launch_night3.sh > /dev/null 2>&1 &
# 熔断恢复: 排障后 rm output/glm53flash/.night3_halt 再启 supervisor
```

## 历史教训(详见 docs/glm3flash/experiment_tracking.md §6)

watchdog 误杀(阈值/路径双 bug)、kill -9 后 HCCL/AICPU 脏状态(stop 缓冲 60s)、
外部任务占卡(gate)、dump env 泄漏评测(6 处 unset)、`$?` 取管道末端(PIPESTATUS)、
确定性代码 bug 对重试自愈免疫(熔断+autofix)。
