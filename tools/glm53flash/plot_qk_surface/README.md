# plot_qk_surface — indexer q/k 激活值 3D 曲面图

- 日期: 2026-11-12
- 用途: 可视化 dump_indexer_activations 抓取的 q/k 激活分布与离群值; 支持离线 Hadamard 旋转对比
- 轴: x = head × head_dim(q: 32×128=4096 列 / k: 128 列), y = token(q)/池(k), z = 数值
- 用法:
  DUMP_DIR=<dump 目录> python3 plot_qk_surface.py               # 原始数据
  DUMP_DIR=<dump 目录> python3 plot_qk_surface.py --hadamard    # Hadamard 旋转后绘图(Sylvester/√128)
  DUMP_DIR=<dump 目录> python3 plot_qk_surface.py --oscar       # per-layer OSCAR 旋转后绘图(读 <DUMP>/oscar_indexer_ckpt.pt, 可 --oscar <路径> 指定)
  # 三模式输出后缀: _raw / _hadamard / _oscar, 可同目录并排对比
  # 默认 DUMP_DIR = glm53flash/indexer_dump_v3
- 输出: q_3d_surface_{raw|hadamard}.png + k_3d_surface_{raw|hadamard}.png, 写入 dump 目录的 plots/
- 防护: 99.9 分位截断、k 哨兵池剔除(末位索引)、nan/inf 批次跳过
- 注意: 旋转作用于最后一维(128)后再展平, 不要先 flatten 成 4096 列再乘 H(会报维度错误)
- OSCAR 为 per-layer 矩阵; k 无层号, 用首个 DSA 层矩阵示意(q 侧每层各自矩阵)
