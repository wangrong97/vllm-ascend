# VQ2A8 精度评测指南

完整说明已移至 [docs/vq2a8_accuracy.md](../../docs/vq2a8_accuracy.md)。

新文档包含指定 NPU 卡启动服务、上下文与采样参数、`--thinking` / `--no-thinking`、
GPQA Diamond / AIME24 / AIME25 选择、EvalScope 依赖安装、内网离线数据、结果审计和 baseline 对比。
一键执行示例统一使用 `--offline-data-dir /mnt/share/g00872988/vq2a8-eval-data`，先复制完整离线包即可运行。
示例输出上限：thinking 为 32768，no-thinking 为 8192；不同预算分别建立基线。
本目录中的脚本路径保持不变。

其他测试指南：

- [NPU CI 回归](../../docs/vq2a8_npu_ci.md)
- [性能测试与指定卡启动服务](../../docs/vq2a8_performance.md)
