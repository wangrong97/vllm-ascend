# gpqa_resume35 — watchdog 误杀 gpqa 数据恢复与补测

- 日期: 2026-11-13
- 背景: night3 watchdog 双 bug(40min 阈值 < gpqa 单遍 50min + summary 路径 `*night3*` 匹配不到 `night2_exp*` 目录)误杀 6 个 gpqa run 的 ais_bench 主进程。被杀 run 的 **infer 子进程成孤儿继续跑**, 数据部分/全部落盘, 只缺 summary。
- 数据恢复原则(口径已用实验一校准: 离线复算 89.90 = ais_bench 报分):
  - 判分: `re.findall(r"[Aa]nswer[:\s]*\(?([A-D])\)?", prediction)` 取最后一处 vs gold
  - 空 prediction 行 = 被杀瞬间在飞中断(非答错), 需补测; 有效行直接采信

## 恢复方案(2026-11-13 执行)

| run | 数据状态 | 处置 |
| --- | --- | --- |
| 三h gpqa run1 | 198/198 全答, 89.90% | 离线 summary 直接采信 |
| 三h run2 | 188 答(173 对) + 10 空 | 补测 10 题合并 |
| 三h run3 | 163 答(155 对) + 35 空 | 补测 35 题合并 |
| 七 gpqa run1 | 孤儿 infer 存活跑完 198 | watcher_a 等完成后离线 summary |

注: run2 缺的 10 题 ⊆ run3 缺的 35 题, 合并补测集 = 35 题。

## 文件

- `exp7_run1_watch.sh`(watcher_a): 轮询孤儿 infer(3825842/43)→ tmp 198 行 → 生成 run1 `summary_offline.csv`
- `resume_exp3h_gpqa.sh`(watcher_b): 等 night3 主链退出 → 起三h配置服务 → ais_bench 跑 `gpqa_resume35` 迷你集 → 按 question 文本(origin_prompt `'\n\nA)'` 前段)匹配合并 → 生成 run1/2/3 `summary_offline.csv` → 标 `.done` → 重启 supervisor
- 迷你集: `benchmark/ais_bench/datasets/gpqa_resume35/gpqa_resume35.csv`(GPQA 原始格式, 选项排列由 loader 行序 shuffle, question→gold 自洽)+ `benchmark/configs/datasets/gpqa_resume35/`

## 注意

- `summary_offline.csv` 命名匹配 night3 脚本 `run_has_result` 的 `summary_*.csv` glob → 断点续跑机制自动跳过
- 补测题选项顺序与原遍不同(loader 行序 shuffle), 属评测随机性允许范围; 判分按各自 gold, 正确性不受影响
- 中断题备份: `output/glm53flash/killed_runs_backup_1113/`
