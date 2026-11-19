#!/usr/bin/env python3
"""OSCAR 协方差旋转矩阵算法 — GLM-5.3-Flash kpool indexer 适配版。

来源: gitcode.com/Ascend-Quant/vllm-ascend PR #24 (Qwen235B-attn-v16-oscar)
       vllm_ascend/attention/oscar.py — 数据驱动的正交旋转, 拟合 Q/K 协方差
       特征基并组合 Hadamard / 特征值排序位反转, 用于量化前分散离群值。

适配点(与原版的差异):
  1. 作用对象: Qwen 主注意力的 Q/K/V → **GLM kpool indexer 的 q(打分 query)与
     k(池化 key)**。head_size = 128(indexer 宽度), 无 GQA 分组(indexer q 为
     [T, H, 128] 多头, k 为 [T, 128] 单向量/池化), 因此:
       - K 协方差 = 所有 q 头展平后的 qᵀq 协方差(原版 per-kv-head 分组均值,
         这里 k 无头维, 用 q 全头协方差作为 k 旋转的拟合对象 —— 与原版 GQA
         分组均值语义一致: 让 K 旋转同时对齐全部 query 头)
       - V 侧不适用(indexer 无 V), 仅输出 k_rotation
  2. 数据采集: 复用本仓库 tools/glm53flash/dump_indexer_activations 的
     raw npz(q_layer{N}_step{S}.npz / k_pooled_*.npz), 不依赖原版 collector。
  3. 应用点: 与 VLLM_ASCEND_KPOOL_HADAMARD_PRE 同构 —— q 打分前 / k 池化写入前
     / 尾池生产点右乘 R(本文件提供 runtime 辅助, 见 README)。

拟合算法(与原版逐式对齐):
  K: C_k = E[q qᵀ](全头展平, fp64 累加) → 对称化 → eigh → R_k = compose(V, λ)
  compose ∈ {plain, h, r_h, r_h_pbr(默认)}: 特征基 @ Hadamard @ pbr(λ)
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

OSCAR_ROTATION_COMPOSITIONS = ("plain", "h", "r_h", "r_h_pbr")
OSCAR_DEFAULT_ROTATION_COMPOSITION = "r_h_pbr"
HEAD_SIZE = 128  # GLM-5.3-Flash indexer head_dim


# ---- 与原版一致的工具函数 ----

def _build_hadamard(size: int) -> torch.Tensor:
    """Sylvester 构造的归一化 Hadamard(size 为 2 的幂)。"""
    if size & (size - 1):
        raise ValueError(f"Hadamard size must be a power of two, got {size}")
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < size:
        h = torch.cat([torch.cat([h, h], dim=1),
                       torch.cat([h, -h], dim=1)], dim=0)
    return h / math.sqrt(size)


def _bit_reversal_permutation(size: int) -> torch.Tensor:
    """位反转置换矩阵。"""
    nbits = size.bit_length() - 1
    perm = torch.empty(size, dtype=torch.long)
    for i in range(size):
        r = 0
        x = i
        for _ in range(nbits):
            r = (r << 1) | (x & 1)
            x >>= 1
        perm[i] = r
    out = torch.zeros(size, size, dtype=torch.float64)
    out[perm, torch.arange(size)] = 1.0
    return out


def _eigenvalue_sorted_bit_reversal(eigenvalues: torch.Tensor) -> torch.Tensor:
    """按特征值降序应用位反转(原版 pbr: 大特征值维度先做位反转分散)。"""
    size = eigenvalues.shape[0]
    order = torch.argsort(eigenvalues, descending=True)
    pbr = _bit_reversal_permutation(size)
    out = torch.empty_like(pbr)
    out[order] = pbr  # 排序后的行再位反转
    return out


def _compose_rotation(
    eigenvectors: torch.Tensor,
    eigenvalues: torch.Tensor,
    *,
    composition: str,
) -> torch.Tensor:
    if composition not in OSCAR_ROTATION_COMPOSITIONS:
        raise ValueError(
            f"OSCAR composition must be one of {OSCAR_ROTATION_COMPOSITIONS}, got {composition!r}")
    if composition == "plain":
        return eigenvectors
    hadamard = _build_hadamard(eigenvectors.shape[0])
    if composition == "h":
        return hadamard
    if composition == "r_h":
        return eigenvectors @ hadamard
    # r_h_pbr (default)
    return eigenvectors @ hadamard @ _eigenvalue_sorted_bit_reversal(eigenvalues)


# ---- 数据加载(dump 工具的 raw npz) ----

def _load_dump_tensors(dump_dir: Path, stage: str, layer: int):
    """流式读取 dump_indexer_activations raw 模式的 npz: {stage}_layer{N}_step{S}.npz
    (2026-11-13: list → 生成器, 大数据集下避免一次性载入的内存峰值)"""
    files = sorted(dump_dir.glob(f"{stage}_layer{layer}_step*.npz"))
    if not files:
        raise FileNotFoundError(
            f"{dump_dir} 下无 {stage}_layer{layer}_step*.npz(需要 MODE=raw 抓取)")
    for f in files:
        try:
            data = np.load(f)["data"]  # 原始抓取已转 float32
        except Exception as e:  # noqa: BLE001 — 坏 npz(如服务停止时截断, BadZipFile/CRC)跳过不阻断
            print(f"[oscar-fit] 跳过坏文件: {f.name} ({type(e).__name__})")
            continue
        yield torch.from_numpy(data).to(torch.float64)


# ---- 拟合主体 ----

def fit_indexer_rotation(
    dump_dir: str | Path,
    layer: int,
    *,
    composition: str = OSCAR_DEFAULT_ROTATION_COMPOSITION,
) -> dict:
    """从 dump 的 q 激活拟合该层 indexer 的 OSCAR k 旋转。

    K 协方差(与原版 GQA 分组均值语义对齐):
      C = (1/N) Σ_t,h q[t,h,:] q[t,h,:]ᵀ   (全部 query 头展平)
      C ← (C + Cᵀ)/2
      eigh(C) → R = V @ H @ pbr(λ)
    """
    dump_dir = Path(dump_dir)
    qs = _load_dump_tensors(dump_dir, "q", layer)
    cov = torch.zeros(HEAD_SIZE, HEAD_SIZE, dtype=torch.float64)
    n = 0
    clean_any = False
    for q in qs:  # q: [T, H, 128] — 流式累加协方差, 不保留样本(spread 由协方差恒等式算)
        if q.shape[-1] != HEAD_SIZE:
            raise ValueError(f"q 宽度应为 {HEAD_SIZE}, got {q.shape[-1]}")
        if torch.isnan(q).any() or torch.isinf(q).any():
            continue  # 跳过含 nan/inf 的批次
        flat = q.reshape(-1, HEAD_SIZE)
        cov.addmm_(flat.T, flat)
        n += flat.shape[0]
        clean_any = True
        del q, flat
    if not clean_any:
        raise ValueError(f"layer 数据全部含 nan, 无法拟合")
    cov /= n
    cov = (cov + cov.T) / 2

    eigvals, eigvecs = torch.linalg.eigh(cov)
    rotation = _compose_rotation(eigvecs, eigvals, composition=composition)

    # 正交性校验(与原版同阈值)
    orth_err = (rotation.T @ rotation - torch.eye(HEAD_SIZE, dtype=torch.float64)).abs().max().item()
    if orth_err > 1e-3:
        raise ValueError(f"rotation 非正交: max err {orth_err:.3e}")

    # 拟合质量指标: 旋转后 q 的逐维能量方差(越小说明能量越均匀)
    # 2026-11-13: 改用协方差恒等式计算 —— before = diag(C), after = diag(RᵀCR),
    # 与逐样本计算数学等价, 避免大数据集(百万级样本)下 torch.cat 的内存峰值
    energy_before = cov.diagonal()
    energy_after = (rotation.T @ cov @ rotation).diagonal()
    spread_before = float(energy_before.std() / energy_before.mean())
    spread_after = float(energy_after.std() / energy_after.mean())

    return {
        "layer": layer,
        "composition": composition,
        "num_samples": n,
        "rotation": rotation.to(torch.float32),
        "eigenvalues": eigvals.to(torch.float32),
        "orthogonality_error": orth_err,
        "energy_spread_before": spread_before,
        "energy_spread_after": spread_after,  # 越小越好(理想≈1.0, 即无离散)
    }


def fit_all_layers(
    dump_dir: str | Path,
    output_path: str | Path,
    *,
    composition: str = OSCAR_DEFAULT_ROTATION_COMPOSITION,
    layers: list[int] | None = None,
) -> dict:
    """拟合并保存 OSCAR indexer 旋转 checkpoint(兼容原版 checkpoints 结构风格)。"""
    dump_dir = Path(dump_dir)
    if layers is None:
        layers = sorted({int(m.group(1)) for f in dump_dir.glob("q_layer*_step*.npz")
                         if (m := __import__("re").search(r"q_layer(\d+)_", f.name))})
    if not layers:
        raise FileNotFoundError(f"{dump_dir} 下未发现任何 q_layer*_step*.npz")

    ckpt = {
        "format": "glm53flash-indexer-oscar",
        "version": 1,
        "head_size": HEAD_SIZE,
        "composition": composition,
        "layers": {},
    }
    print(f"{'L':>4s} {'samples':>9s} {'orth_err':>10s} {'spread前':>9s} {'spread后':>9s}")
    for layer in layers:
        r = fit_indexer_rotation(dump_dir, layer, composition=composition)
        ckpt["layers"][str(layer)] = {
            "k_rotation": r["rotation"],
            "eigenvalues": r["eigenvalues"],
        }
        print(f"{layer:>4d} {r['num_samples']:>9d} {r['orthogonality_error']:>10.2e} "
              f"{r['energy_spread_before']:>9.3f} {r['energy_spread_after']:>9.3f}")
    torch.save(ckpt, output_path)
    print(f"saved: {output_path} ({len(ckpt['layers'])} 层)")
    return ckpt


# ---- runtime 应用辅助(供 indexer_kpool 集成时参考/复用) ----

class IndexerOscarRotation:
    """加载 checkpoint 并对 indexer q/k 施加旋转(语义同 _hadamard_rotate)。"""

    def __init__(self, ckpt_path: str | Path, device: str = "npu", dtype=torch.bfloat16):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.matrices = {
            int(k): v["k_rotation"].to(device=device, dtype=dtype)
            for k, v in ckpt["layers"].items()
        }
        self.device, self.dtype = device, dtype

    def rotate(self, x: torch.Tensor, layer: int) -> torch.Tensor:
        """x[..., 128] @ R_layer(层号来自 indexer prefix)。"""
        r = self.matrices[layer]
        return (x.to(self.dtype) @ r).to(x.dtype)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dump_dir", help="dump_indexer_activations raw 模式输出目录")
    ap.add_argument("output", help="输出 checkpoint 路径(.pt)")
    ap.add_argument("--composition", default=OSCAR_DEFAULT_ROTATION_COMPOSITION,
                    choices=OSCAR_ROTATION_COMPOSITIONS)
    ap.add_argument("--layers", default="", help="逗号分隔层号(默认自动发现)")
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",") if x.strip()] or None
    fit_all_layers(a.dump_dir, a.output, composition=a.composition, layers=layers)


if __name__ == "__main__":
    main()
