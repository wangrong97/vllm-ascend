# dump_indexer_activations — kpool indexer q/k 激活值抓取

- 日期: 2026-11-12
- 用途: 在线抓取 Glm5NextKPoolIndexerBackend 的 q(打分前)与 k(池化写入前)激活,
  支持量化/伪量化/旋转前后的分布分析(stats)与原始张量导出(raw)
- 用法: 起服务前设环境变量(总开关 VLLM_ASCEND_INDEXER_DUMP=1), 正常发请求,
  输出目录得 summary.json / config.json / raw npz
- 环境变量: DUMP_DIR(默认 /tmp/indexer_dump) | STAGES(q,k,默认 q,k)
  | LAYERS(层号白名单, 默认全部) | MAX_STEP(默认 20) | MODE(stats|raw)
- 接线: vllm_ascend/utils.py bootstrap_custom_op_env 尾部自动 install(仅开关开启时)
- 注意: 分析用途, 有开销; raw 模式含 D2H 拷贝, 抓完即关

## 内置对照测试集(2026-11-12)

`testset/dump_testset.jsonl` 含 4 道固定题(从历史评测按错误率抽取):
- aime_easy id9(0/5 错, avg_out 79K) / aime_hard id11(5/5 错, 打满 131K)
- gpqa_easy id48(0/5 错, 61K) / gpqa_hard id17(5/5 错, 2.5K 短答即错)

一键测试(服务需以 DUMP=1 起动):
  bash run_dump_test.sh
默认每题 8192 token(快速出分布; 可 MAXTOK=163840 跑完整思考链), 自动汇总
q/k 各层 absmax/p999/absmean, 直接对比两种数据集的分布差异。
