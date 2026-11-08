# VQ2A8 并发吞吐测试入口

单卡 98G 测试使用[单卡启动与压测命令](single_card_batch_throughput.md)：TP1/EP1、MRV2、B64，
显存比例 0.97，起步配置为 1 GiB KV、1024-token 调度预算，扫描并发 1～64。
单卡显存容量需实测确认，完整 FFN 和图回放保持启用。

以下为四卡测试口径：

当前主线模型为 `DeepSeek-V4-Flash-0731-32x32-noperm-hot5050125-identity-v3`。
使用 [四卡启动与压测命令](packed_weight_batch_throughput.md)：终端 A 启动服务，终端 B 扫描并发。
该文档沿用原文件名，内容已更新到无 perm 混精模型。

- 固定 TP4/EP4、DP1、MRV2、B64，关闭 MTP/DSpark。
- 客户端并发为 1、2、4、8、16、32、64，输入 1024、输出 128 tokens。
- 每档正式请求数为 4×并发，预热 1 次，正式测量 1 次。
- 固定使用 native 完整 FFN，默认 decode 图；可用 `--enforce-eager` 做同算子对照。
- 保存吞吐、TTFT、TPOT、P99 E2EL、summary.json 和服务返回的模型路径。

已有完全符合配置的四卡服务可以复用。B64 表示服务活动序列上限，客户端并发不保证实际每步 decode batch。
旧 repack 权重的带 perm 路径已退出支持，不能只删除张量或补 identity 来转换语义。
精度评测使用不同的上下文配置，见 [四卡精度评测](vq2a8_accuracy.md)。
