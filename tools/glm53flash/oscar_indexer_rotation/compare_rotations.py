#!/usr/bin/env python3
"""对比 OSCAR(协方差拟合) vs Hadamard(固定) 对 indexer q/k 量化的离群值抑制效果.

输入: dump_indexer_activations raw npz(q_layer{N}_step{S}.npz)
方法: 同一批 q 激活, 分别过两种旋转后做 MXFP4/MXFP8 伪量化往返,
      统计逐 32 组的组内相对误差与能量离散度, 并列输出.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from oscar_indexer_rotation import (
    HEAD_SIZE,
    _build_hadamard,
    fit_indexer_rotation,
)


def mxfp4_roundtrip(x: torch.Tensor, packed_scale_max: float = 7.25) -> torch.Tensor:
    import torch_npu
    flat = x.reshape(-1, x.shape[-1]).contiguous()
    packed, scale = torch_npu.npu_dynamic_mx_quant(
        flat, dst_type=torch_npu.float4_e2m1fn_x2, scale_alg=2, dst_type_max=packed_scale_max)
    vals = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6],
                        dtype=torch.float32, device=x.device)
    lo = vals[(packed & 0xF).long()]; hi = vals[(packed >> 4).long()]
    deq = torch.stack((lo, hi), dim=-1).flatten(-2)
    n, d = flat.shape
    f = torch.pow(2.0, scale.reshape(n, -1)[:, : d // 32].int() - 127)
    return (deq.view(n, d // 32, 32) * f.unsqueeze(-1)).view(n, d).view(x.shape).to(x.dtype)


def mxfp8_roundtrip(x: torch.Tensor) -> torch.Tensor:
    import torch_npu
    flat = x.reshape(-1, x.shape[-1]).contiguous()
    q, s = torch_npu.npu_dynamic_mx_quant(flat, dst_type=torch.float8_e4m3fn)
    n, d = flat.shape
    f = torch.pow(2.0, s.reshape(n, -1)[:, : d // 32].int() - 127)
    return (q.float().view(n, d // 32, 32) * f.unsqueeze(-1)).view(n, d).view(x.shape).to(x.dtype)


def energy_spread(x: torch.Tensor) -> float:
    e = x.float().pow(2).mean(dim=tuple(range(x.dim() - 1)))
    return float(e.std() / e.mean())


def rel_err(orig: torch.Tensor, rt: torch.Tensor) -> float:
    return float((rt.float() - orig.float()).pow(2).mean().sqrt() / orig.float().pow(2).mean().sqrt())


def main(dump_dir: str):
    dump = Path(dump_dir)
    layers = sorted({int(p.name.split("_")[1].replace("layer", ""))
                     for p in dump.glob("q_layer*_step*.npz")})
    H = _build_hadamard(HEAD_SIZE).float().npu()

    for layer in layers:
        qs = [t for t in (torch.from_numpy(np.load(p)["data"]).npu()
              for p in sorted(dump.glob(f"q_layer{layer}_step*.npz")))
              if not (torch.isnan(t).any() or torch.isinf(t).any())]
        if not qs:
            print(f"\n===== layer {layer}: 全部批次含 nan, 跳过 =====")
            continue
        q = torch.cat([t.reshape(-1, HEAD_SIZE) for t in qs]).npu()  # [N, 128]

        # 拟合该层 OSCAR(基于同批数据)
        import tempfile, os
        with tempfile.TemporaryDirectory() as td:
            np.savez(os.path.join(td, "q_layer%d_step0.npz" % layer), data=q.float().cpu().numpy())
            R = fit_indexer_rotation(td, layer)["rotation"].npu()

        print(f"\n===== layer {layer} (N={q.shape[0]}) =====")
        print(f"{'方案':<22s} {'能量离散度':>10s} {'MXFP4相对误差':>14s} {'MXFP8相对误差':>14s}")
        for name, mat in [("无旋转", None), ("Hadamard", H), ("OSCAR(r_h_pbr)", R)]:
            xr = q if mat is None else q @ mat
            sp = energy_spread(xr)
            e4 = rel_err(xr, mxfp4_roundtrip(xr))
            e8 = rel_err(xr, mxfp8_roundtrip(xr))
            print(f"{name:<22s} {sp:>10.3f} {e4:>14.4f} {e8:>14.4f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/mnt/share/w00608002/work/output/glm53flash/dump_3h_c8")
