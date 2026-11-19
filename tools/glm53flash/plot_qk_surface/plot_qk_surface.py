#!/usr/bin/env python3
"""q/k 激活值 3D 曲面图: x=head×head_dim(展平列), y=token, z=数值大小.

参考用户提供的 plot_3d_surface 风格: plot_surface + coolwarm 色板 + colorbar.
q: 每层一张子图(12 层网格); k: 单独一张(池 × 128 维).
支持旋转对比(与服务端 _indexer_rotate 同构):
  --hadamard            128 阶归一化 Hadamard(Sylvester/√128)
  --oscar [ckpt.pt]    per-layer OSCAR 矩阵(默认读 <DUMP>/oscar_indexer_ckpt.pt,
                        由 oscar_indexer_rotation.py 拟合生成)
用于对比 无旋转/Hadamard/OSCAR 三种分布形态与离群值.
"""
import glob
import math
import os

import numpy as np

DUMP = os.environ.get('DUMP_DIR', '/mnt/share/w00608002/work/output/glm53flash/dump/indexer_dump_exp1')
OUT = os.path.join(DUMP, 'plots')
LAYERS = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 45]

# 旋转模式开关(命令行): none | hadamard | oscar
_argv = os.sys.argv
if '--hadamard' in _argv:
    ROT_MODE = 'hadamard'
elif '--oscar' in _argv:
    ROT_MODE = 'oscar'
    i = _argv.index('--oscar')
    OSCAR_CKPT = _argv[i + 1] if i + 1 < len(_argv) and not _argv[i + 1].startswith('-') \
        else os.path.join(DUMP, 'oscar_indexer_ckpt.pt')
else:
    ROT_MODE = 'none'


def build_hadamard(size):
    """Sylvester 构造归一化 Hadamard(size 为 2 的幂), 与服务端 _get_hadamard128 同构."""
    if size & (size - 1):
        raise ValueError(f'Hadamard size must be power of two, got {size}')
    h = np.ones((1, 1))
    while h.shape[0] < size:
        h = np.block([[h, h], [h, -h]])
    return h / math.sqrt(size)


H128 = build_hadamard(128) if ROT_MODE == 'hadamard' else None

OSCAR_MATS = None  # {layer: [128,128] float32}
if ROT_MODE == 'oscar':
    import torch
    _ckpt = torch.load(OSCAR_CKPT, map_location='cpu', weights_only=True)
    OSCAR_MATS = {int(k): v['k_rotation'].numpy().astype(np.float64)
                 for k, v in _ckpt['layers'].items()}
    print(f'OSCAR checkpoint: {OSCAR_CKPT} ({len(OSCAR_MATS)} 层)')


MAX_FILES = int(os.environ.get('MAX_FILES', '0'))  # >0 时每层均匀抽样该数量 npz(大数据集防护), 0=全部
MIN_BYTES = int(os.environ.get('MIN_BYTES', '0'))  # >0 时过滤小于该字节的文件(排除 decode 小步)


def _sample(files):
    files = sorted(files)
    if MIN_BYTES > 0:
        files = [f for f in files if os.path.getsize(f) >= MIN_BYTES]
    if MAX_FILES > 0 and len(files) > MAX_FILES:
        files = files[::max(1, len(files) // MAX_FILES)][:MAX_FILES]
    return files


def load_q(layer, step_min=0):
    out = []
    for f in _sample(glob.glob(f'{DUMP}/q_layer{layer}_step*.npz')):
        if int(f.split('step')[1].split('.')[0]) < step_min:
            continue
        d = np.load(f)['data']
        if np.isnan(d).any() or np.isinf(d).any():
            continue
        out.append(d)
    return np.concatenate(out) if out else None


def load_k(step_min=0):
    out = []
    for f in _sample(glob.glob(f'{DUMP}/k_pooled_layer-1_step*.npz')):
        if int(f.split('step')[1].split('.')[0]) < step_min:
            continue
        d = np.load(f)['data']
        if np.isnan(d).any() or np.isinf(d).any():
            continue
        d = d[:-1]
        out.append(d[np.abs(d).max(1) < 1e3])
    return np.concatenate(out) if out else None


def rot(data, layer=None):
    """按 ROT_MODE 对最后一维(128)施加旋转: data[..., 128] @ R."""
    if ROT_MODE == 'none':
        return data
    flat = data.reshape(-1, data.shape[-1])
    if ROT_MODE == 'hadamard':
        flat = flat @ H128
    else:  # oscar
        if layer is None or layer not in OSCAR_MATS:
            raise KeyError(f'OSCAR 缺 layer {layer}(可用: {sorted(OSCAR_MATS)})')
        flat = flat @ OSCAR_MATS[layer]
    return flat.reshape(data.shape)


def plot_3d_surface(ax, data, title=''):
    """参考风格的 3D 曲面: x=列, y=token/池, z=数值."""
    print(data.shape)
    if data.shape[0] == 1:
        data = np.squeeze(data)
    X, Y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
    surf = ax.plot_surface(X, Y, data, edgecolor='none', cmap='coolwarm')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('head × head_dim (32 heads × 128 dims = 4096 channels)',
                 fontsize=8, labelpad=1)
    ax.set_ylabel(f'Token dim ({data.shape[0]} tokens)', fontsize=8, labelpad=1)
    ax.set_zlabel('Activation value', fontsize=8, labelpad=1)
    ax.tick_params(labelsize=7)
    return surf


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs(OUT, exist_ok=True)
TAG = {'none': 'raw', 'hadamard': 'hadamard', 'oscar': 'oscar'}[ROT_MODE]

# ---- q: 12 层 3×4 网格曲面 ----
fig = plt.figure(figsize=(22, 15))
for i, L in enumerate(LAYERS):
    q = load_q(L)
    if q is None:
        continue
    data = rot(q, layer=L).reshape(q.shape[0], -1).astype(np.float64)   # 先旋转末 128 维, 再展平
    # 数值防护: 99.9 分位截断避免离群尖峰压扁整张曲面
    hi = np.percentile(np.abs(data), 99.9)
    data = np.clip(data, -hi, hi)
    ax = fig.add_subplot(3, 4, i + 1, projection='3d')
    surf = plot_3d_surface(ax, data, title=f'Layer {L} (tokens={data.shape[0]})')
    fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.08)
fig.suptitle(f'q Activation 3D Surface [{TAG}] — {os.path.basename(DUMP)} '
            f'(x: head×dim, y: token, z: value, clipped@p99.9)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUT, f'q_3d_surface_{TAG}.png'), dpi=100, bbox_inches='tight')
plt.close()

# ---- k: 单曲面(OSCAR 为 per-layer 矩阵而 k 无层号, 用首个 DSA 层矩阵示意; Hadamard 全局一致) ----
k = load_k()
if k is not None:
    if ROT_MODE == 'oscar':
        _k_layer = sorted(OSCAR_MATS)[0]
        print(f'k(OSCAR): 无层号, 用 layer {_k_layer} 矩阵示意')
        k = rot(k.astype(np.float64), layer=_k_layer)
    else:
        k = rot(k.astype(np.float64))
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = plot_3d_surface(ax, k, title=f'k_pooled (pools={k.shape[0]})')
    fig.colorbar(surf, ax=ax, shrink=0.7)
    ax.set_ylabel('Pool dim (token/4)', fontsize=9, labelpad=2)
    ax.set_xlabel('head_dim (128 channels)', fontsize=9, labelpad=2)
    plt.title(f'k Activation 3D Surface [{TAG}]')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, f'k_3d_surface_{TAG}.png'), dpi=110, bbox_inches='tight')

print(f'输出({TAG}):', OUT)
for f in sorted(os.listdir(OUT)):
    if f.endswith('.png'):
        print(' ', f)
