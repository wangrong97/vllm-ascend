# night1118 夜间编排(EPLB 精度三组 + E3 权重生产收尾)

- 机制借鉴 `work/vllm-ascend/tools/glm53flash/night3/`(supervisor 重拉+熔断 / 阶段断点秒过 / 评测成绩断点续跑 / 失败遍补跑 / 90min watchdog / 8卡稳定 gate / 停服 60s 缓冲)
- 日志: `eplb/output/night_exp_20261118/main.log`(主链) + `supervisor.log` + `serve_E*.log` + `e3_attnd.log`
- 标记: `.e1_done/.e2_done/.e4_done`(阶段完成) `.e3_failed`(权重生产失败,E4 跳过) `.finished`(全链结束) `.halt`(熔断/E4 乱码门禁, 排障后 rm 再启)

## 文件

| 脚本 | 作用 |
| --- | --- |
| `night_exp_1118.sh` | 主编排: E3 收尾(等量化→attn 对齐) → gate(00:00+8卡) → E1 FP8(OOM 自动降档一次) → E2 W2A8 → E4 混搭(smoke 乱码门禁不过即熔断留样) |
| `night_exp_supervisor.sh` | 60s 重拉 + 30min 内 ≥3 次熔断 `.halt` |

## 链

| 组 | P | D | 说明 |
| --- | --- | --- | --- |
| E1 | FP8 原始 | FP8 原始 | 基线; serve 脚本 QUANT=auto 识别 |
| E2 | noperm-export | noperm-export | W2A8 基线 |
| E4 | attnD(E3 产物) | repack | smoke(1081+无乱码) 门禁; attn/root 硬链 repack |

每组: AIME24×3 + GPQA×3 (ais_bench vllm_api_general_chat, :8000 proxy, dsv, temp=1.0)

## 用法

```bash
# 立即启动(现在就会跑 E3 收尾 CPU 段, 00:00 gate 后自动进 E1):
setsid nohup bash /mnt/share/w00608002/work/eplb/vllm-ascend/tools/eplb/eval/night1118/night_exp_supervisor.sh > /dev/null 2>&1 &
# 熔断恢复: 排障后 rm output/night_exp_20261118/.halt 再启 supervisor
```

## 历史教训(继承 night3 + 本夜)

watchdog 阈值 90min(gpqa 单遍 45-50min 无日志)、kill 后 HCCL 脏状态缓冲 60s、
确定性 bug 对重试自愈免疫(熔断+人工)、E4 乱码是权重级问题(门禁留样不重试)、
量化链 non_routed 必须保 F8_E4M3 原格式(见 night_exp_20261118_progress.md 20:03 条目)。
