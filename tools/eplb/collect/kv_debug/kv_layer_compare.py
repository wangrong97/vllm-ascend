#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E5: 两侧单机逐层 KV 累积漂移对比(10T 同款层组表)。
输入: --a=P权重 dump(kv_single_side_P.pt) --b=D权重 dump(kv_single_side_D.pt); 输出层组 cos/相对误差 均值·中位·P90"""
import argparse, torch

def load(p): return torch.load(p, weights_only=False)["blocks"]

def group_stats(rows):
    cos = [r[1] for r in rows]; rel = sorted(r[2] for r in rows)
    n = len(rel); med = rel[n // 2]; p90 = rel[min(n - 1, int(round(0.9 * (n - 1))))]
    return sum(cos) / len(cos), sum(rel) / len(rel), med, p90

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--scale-tail", type=int, default=128); ap.add_argument("--block", type=int, default=0)
    args = ap.parse_args()
    A, B = load(args.a), load(args.b)
    layers = sorted(set(A) & set(B), key=lambda x: int(str(x).split("_")[-1]) if str(x).split("_")[-1].isdigit() else 0)
    print(f"A 层: {len(A)} | B 层: {len(B)} | 公共: {len(layers)}")
    rows = []
    print(f"\n{'层':>3} {'数据区cos':>9} {'数据区rel':>9} {'scale区cos':>10} {'scale区rel':>10}")
    for L in layers:
        ta, tb = A[L].get(args.block), B[L].get(args.block)
        if ta is None or tb is None or ta.shape != tb.shape: continue
        tail = args.scale_tail
        if ta.dim() >= 2 and ta.shape[-1] > tail:
            da, db = ta[..., :-tail].float(), tb[..., :-tail].float(); sa, sb = ta[..., -tail:].float(), tb[..., -tail:].float()
        else: da, db = ta.float(), tb.float(); sa = sb = None
        def cr(x, y):
            xf, yf = x.flatten(), y.flatten()
            return (torch.dot(xf, yf) / (xf.norm() * yf.norm() + 1e-12)).item(), ((xf - yf).norm() / (xf.norm() + 1e-12)).item()
        cd, rd = cr(da, db); cs, rs = cr(sa, sb) if sa is not None else (1.0, 0.0)
        print(f"{L:>3} {cd:>9.5f} {rd:>9.4f} {cs:>10.5f} {rs:>10.4f}")
        rows.append((L, cd, rd))
    print("\n=== 10T 同款层组表(数据区) ===")
    print(f"{'层组':<8}{'cos均值':>9} {'rel均值':>8} {'rel中位':>8} {'rel P90':>8}")
    trend = []
    for g in range(0, len(layers), 10):
        grp = [r for r in rows if g <= int(str(r[0]).split("_")[-1]) < g + 10]
        if not grp: continue
        c, rm, med, p90 = group_stats(grp); trend.append(c)
        print(f"{g}-{g+len(grp)-1:<5}{c:>9.4f} {rm:>8.3f} {med:>8.3f} {p90:>8.3f}")
    print("\n10T 参照(独立量化, 崩溃): 0.991→0.893 递减; 同源(恢复): ~0.999 平稳")
    if trend:
        drop = trend[0] - trend[-1]
        verdict = ("❌ 嫌疑A坐实: 深层显著衰减" if trend[-1] <= 0.93 or drop >= 0.02 else "✅ 嫌疑A排除: 无显著衰减 → 转 E6")
        print(f"判定: cos 首组 {trend[0]:.4f} → 末组 {trend[-1]:.4f}(落差 {drop:.4f}) — {verdict}")

if __name__ == "__main__": main()
