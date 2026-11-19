#!/usr/bin/env python3
"""三套校准 dump 的能量离散度(spread)交叉对比 — 纯 CPU。

回答两个问题:
  1. 权重量化深度(FP8→W4A4→KDA8)对 indexer q 分布各向异性的影响(raw spread 对比)
  2. 三套 OSCAR 矩阵的泛化性: 3×3 交叉(数据×矩阵), 对角线=自配, 非对角=移植
  3. 附: 三矩阵两两差异(max|R_a R_bᵀ − I|)
用法: python3 spread_3way.py [每层抽样npz数=40]
"""
import glob
import os
import sys

import numpy as np
import torch

DUMP = "/mnt/share/w00608002/work/output/glm53flash/dump"
OSCAR = "/mnt/share/w00608002/work/output/glm53flash/oscar_calib"
DUMPS = {
    "FP8": f"{DUMP}/indexer_dump_calib_flat",
    "W4A4": f"{DUMP}/indexer_dump_calib_w4a4_flat",
    "KDA8": f"{DUMP}/indexer_dump_calib_kda8_flat",
}
MATS = {
    n: {int(k): v["k_rotation"].numpy().astype(np.float64)
        for k, v in torch.load(f"{OSCAR}/oscar_indexer_{s}.pt", map_location="cpu", weights_only=True)["layers"].items()}
    for n, s in [("FP8", "calib"), ("W4A4", "w4a4"), ("KDA8", "kda8")]
}
LAYERS = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 45]


def load_q(flat, layer, n, min_bytes=100000):
    files = sorted(f for f in glob.glob(f"{flat}/q_layer{layer}_step*.npz")
                   if os.path.getsize(f) >= min_bytes)
    files = files[::max(1, len(files) // n)][:n]
    return np.concatenate([np.load(f)["data"].reshape(-1, 128) for f in files])


def spread(x):
    e = (x ** 2).mean(0)
    return float(e.std() / e.mean())


def main(n=40):
    data = {(dn, L): load_q(d, L, n) for dn, d in DUMPS.items() for L in LAYERS}

    print(f"===== 1. raw spread(量化深度对分布的影响), 每层 {n} npz 抽样 =====")
    print(f"{'L':>4s} " + " ".join(f"{k:>8s}" for k in DUMPS))
    agg = {k: [] for k in DUMPS}
    for L in LAYERS:
        row = [spread(data[(dn, L)]) for dn in DUMPS]
        for k, v in zip(DUMPS, row):
            agg[k].append(v)
        print(f"{L:>4d} " + " ".join(f"{v:>8.3f}" for v in row))
    print(f"{'mean':>5s} " + " ".join(f"{np.mean(agg[k]):>8.3f}" for k in DUMPS))

    print("\n===== 2. 3×3 交叉: 数据 × 矩阵 → 旋转后 spread(12 层均值) =====")
    hdr = "数据" + chr(92) + "矩阵"
    print(f"{hdr:>10s} " + " ".join(f"{k:>8s}" for k in MATS))
    for dn in DUMPS:
        cells = []
        for mn, mats in MATS.items():
            vals = [spread(data[(dn, L)] @ mats[L]) for L in LAYERS]
            cells.append(np.mean(vals))
        tag = " ←自配" if dn in MATS else ""
        print(f"{dn:>10s} " + " ".join(f"{v:>8.4f}" for v in cells) + tag)

    print("\n===== 3. 矩阵两两差异 max|R_a R_bᵀ − I|(12 层最大值; 0=完全一致) =====")
    names = list(MATS)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d = max(np.abs(MATS[a][L] @ MATS[b][L].T - np.eye(128)).max() for L in LAYERS)
            print(f"  R_{a} vs R_{b}: {d:.4f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
