#!/usr/bin/env python3
"""轻量对比: 已保存的 OSCAR checkpoint(校准集拟合) vs Hadamard vs 无旋转,
对 indexer q 在 MXFP4/MXFP8 伪量化往返下的误差与能量离散度(每层抽样, NPU)。

用法: python3 compare_calib.py <dump_flat_dir> <ckpt_path> [每层抽样npz数]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch_npu

sys.path.insert(0, str(Path(__file__).parent))
from oscar_indexer_rotation import HEAD_SIZE, _build_hadamard
from compare_rotations import mxfp4_roundtrip, mxfp8_roundtrip, energy_spread, rel_err


def main(dump_dir: str, ckpt_path: str, sample_n: int = 50):
    dump = Path(dump_dir)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    H = _build_hadamard(HEAD_SIZE).float().npu()
    layers = sorted(int(k) for k in ckpt["layers"])

    print(f"ckpt: {ckpt_path} (composition={ckpt['composition']}, 抽样 {sample_n} npz/层)")
    agg = {}
    print(f"{'L':>4s} {'方案':<18s} {'能量离散':>9s} {'MXFP4误差':>10s} {'MXFP8误差':>10s}")
    for layer in layers:
        files = sorted(dump.glob(f"q_layer{layer}_step*.npz"))[::max(1, len(sorted(dump.glob(f'q_layer{layer}_step*.npz'))) // sample_n)][:sample_n]
        q = torch.cat([torch.from_numpy(np.load(p)["data"]).reshape(-1, HEAD_SIZE)
                       for p in files]).float().npu()
        R = ckpt["layers"][str(layer)]["k_rotation"].float().npu()
        for name, mat in [("无旋转", None), ("Hadamard", H), ("OSCAR(calib)", R)]:
            xr = q if mat is None else q @ mat
            sp = energy_spread(xr)
            e4 = rel_err(xr, mxfp4_roundtrip(xr))
            e8 = rel_err(xr, mxfp8_roundtrip(xr))
            agg.setdefault(name, []).append((sp, e4, e8))
            print(f"{layer:>4d} {name:<18s} {sp:>9.3f} {e4:>10.4f} {e8:>10.4f}")
        del q
    print("\n===== 12 层均值 =====")
    print(f"{'方案':<18s} {'能量离散':>9s} {'MXFP4误差':>10s} {'MXFP8误差':>10s}")
    for name, rows in agg.items():
        m = np.mean(rows, axis=0)
        print(f"{name:<18s} {m[0]:>9.3f} {m[1]:>10.4f} {m[2]:>10.4f}")


if __name__ == "__main__":
    main(*sys.argv[1:3], int(sys.argv[3]) if len(sys.argv) > 3 else 50)
