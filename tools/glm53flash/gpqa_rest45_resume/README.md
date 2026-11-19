# gpqa_rest45_resume — 实验七 GPQA 剩余 45 题续跑

- 日期: 2026-11-12
- 背景: 实验七(W4A4 + MXFP4伪量化 + Hadamard) gpqa run1 跑到 153/198(94.77%)被停,
  本工具起同配置服务补跑缺失 45 题(id 见 run1/tmp jsonl), 并自动合并 run1 完整总分
- 前置: 迷你数据集 gpqa_rest45(ais_bench datasets/gpqa_rest45/, 已用真实
  GPQADataset loader 生成并验证 gold 153/153 对齐) + 配置已注册
- 用法(需卡 4-7 空闲, 使用后四卡 TP4):
  bash tools/glm53flash/gpqa_rest45_resume/gpqa_rest45_resume.sh
- 输出: run1/rest45/ + run1/merged/summary_merged.csv(153 题旧成绩+45 题新成绩合并)
- 依赖: serve_glm53flash.sh(实验七 env), ais_bench gpqa_rest45 配置
