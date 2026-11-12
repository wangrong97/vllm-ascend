#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EPLB 采集数据绘图: 2×2 组合图(静态分配/卡负载时序/专家热力图/卡负载汇总)。"""
import argparse, csv, glob, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SEQ_BLUE = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#123c6e"]
SURFACE = "#fcfcfb"
CMAP_SEQ = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
INK, INK2 = "#333333", "#666666"

def setup_cjk():
    import matplotlib.font_manager as fm
    for w in ("Noto Sans CJK", "WenQuanYi", "SimHei", "Source Han Sans"):
        if any(w.lower() in f.name.lower() for f in fm.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [w, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False

def recessive(ax):
    ax.grid(axis="y", linestyle="--", alpha=0.25, color=INK2)
    ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.set_facecolor(SURFACE)

def panel_placement(ax, report):
    cards = report["cards"]; ranks = [c["rank"] for c in cards]
    vq2 = np.array([c["w13_vq2"] for c in cards], float); mx4 = np.array([c["w13_mxfp4"] for c in cards], float)
    x = np.arange(len(ranks))
    ax.bar(x, vq2, width=0.56, color=CAT[0], edgecolor=SURFACE, linewidth=1.5, label="w13·vq2")
    ax.bar(x, mx4, bottom=vq2, width=0.56, color=CAT[1], edgecolor=SURFACE, linewidth=1.5, label="w13·mxfp4")
    for xi, (a, b) in enumerate(zip(vq2, mx4)):
        ax.text(xi, a + b + 1.2, f"{int(a+b)}", ha="center", va="bottom", color=INK, fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels([f"NPU {r}" for r in ranks]); ax.set_ylabel("expert count")
    ax.set_title("各卡专家精度分配(w13 投影)", fontsize=12, color=INK)
    ax.legend(frameon=False, fontsize=9); recessive(ax); ax.set_ylim(0, max(vq2 + mx4) * 1.18)

def panel_timeseries_grid(fig, gs_cell, csv_path):
    sub = gs_cell.subgridspec(2, 1, height_ratios=[3, 2])
    ax_u = fig.add_subplot(sub[0]); ax_h = fig.add_subplot(sub[1], sharex=ax_u)
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    ids = sorted({int(r["npu_id"]) for r in rows})
    data = {n: {"t": [], "u": [], "h": []} for n in ids}
    t0 = None
    for r in rows:
        try:
            h, m, s = r["ts"].split(":")
            sec = int(h) * 3600 + int(m) * 60 + float(s)
        except Exception:
            continue
        if t0 is None: t0 = sec
        d = data[int(r["npu_id"])]
        d["t"].append(sec - t0); d["u"].append(float(r["util_pct"] or 0)); d["h"].append(float(r["hbm_mb"] or 0) / 1024.0)
    for i, n in enumerate(ids):
        c = CAT[i % len(CAT)]; d = data[n]
        ax_u.plot(d["t"], d["u"], color=c, lw=1.6, label=f"NPU {n}")
        ax_h.plot(d["t"], d["h"], color=c, lw=1.6)
    ax_u.set_title("GSM8K 压测期间各卡负载", fontsize=12, color=INK)
    ax_u.set_ylabel("util %"); ax_h.set_ylabel("HBM (GB)"); ax_h.set_xlabel("elapsed (s)")
    ax_u.legend(frameon=False, fontsize=8.5, ncol=4); recessive(ax_u); recessive(ax_h)

def panel_heatmap(ax, files, n_experts):
    mats = []
    for path in sorted(files):
        p = json.load(open(path, encoding="utf-8"))
        mats.append(np.asarray(p["counts"], dtype=np.int64))
    if not mats:
        ax.set_title("缺专家计数数据"); return
    mat = np.vstack(mats); ep = mat.shape[0]; per = n_experts // ep
    slots = np.vstack([m[r * per:(r + 1) * per] for r, m in enumerate(mat)])
    lg = np.log1p(slots)
    im = ax.imshow(lg, aspect="auto", cmap=CMAP_SEQ, vmin=0, vmax=max(1e-9, lg.max()))
    ax.set_yticks(range(ep)); ax.set_yticklabels([f"NPU {r}" for r in range(ep)])
    ax.set_xlabel(f"expert slot (block-linear, 0..{per - 1})")
    ax.figure.colorbar(im, ax=ax, fraction=0.035, pad=0.02).set_label("tokens (log1p)", fontsize=9)
    ax.set_title("专家 token 负载热力图(卡 × 专家槽位)", fontsize=12, color=INK)
    total = mat.sum(axis=0)
    for e in np.argsort(total)[-3:][::-1]:
        r = e // per; s = e % per
        ax.text(s, r, f"#{e}", ha="center", va="center", fontsize=7.5,
                color="#fff" if lg[r, s] > lg.max() * 0.55 else INK)

def panel_card_total(ax, files):
    ranks, totals = [], []
    for path in sorted(files):
        p = json.load(open(path, encoding="utf-8")); ranks.append(p["rank_tag"]); totals.append(int(sum(p["counts"])))
    x = np.arange(len(ranks))
    ax.bar(x, totals, width=0.56, color=[CAT[i % len(CAT)] for i in x], edgecolor=SURFACE, linewidth=1.5)
    for xi, t in zip(x, totals): ax.text(xi, t * 1.01, f"{t:,}", ha="center", va="bottom", color=INK, fontsize=9.5)
    if totals:
        avg = float(np.mean(totals)); ratio = max(totals) / avg if avg else 0
        ax.axhline(avg, color=INK2, linestyle="--", lw=1.2)
        ax.set_title(f"各卡路由 token 总量 · imbalance max/avg={ratio:.2f}", fontsize=12, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([f"NPU {i}" for i in range(len(ranks))]); ax.set_ylabel("routed tokens")
    recessive(ax); ax.set_ylim(0, max(totals) * 1.15 if totals else 1)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--out", default=None)
    args = ap.parse_args()
    setup_cjk()
    out = args.out or os.path.join(args.dir, "expert_report.png")
    prpt = os.path.join(args.dir, "placement_report.json"); ncsv = os.path.join(args.dir, "npu_load.csv")
    cfiles = sorted(glob.glob(os.path.join(args.dir, "expert_counts_rank*.json")))
    report = json.load(open(prpt, encoding="utf-8")) if os.path.isfile(prpt) else None
    n_experts = report["num_experts"] if report else 256
    fig = plt.figure(figsize=(16, 10), dpi=150, facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.22, left=0.06, right=0.97, top=0.92, bottom=0.07)
    ax_a = fig.add_subplot(gs[0, 0])
    if report: panel_placement(ax_a, report)
    else: ax_a.set_title("缺少静态分配报告")
    if os.path.isfile(ncsv): panel_timeseries_grid(fig, gs[0, 1], ncsv)
    else: fig.add_subplot(gs[0, 1]).set_title("缺少负载采样")
    ax_c = fig.add_subplot(gs[1, 0]); panel_heatmap(ax_c, cfiles, n_experts)
    ax_d = fig.add_subplot(gs[1, 1]); panel_card_total(ax_d, cfiles)
    fig.suptitle("W2A8(VQ2A8) · EPLB off · GSM8K — 专家分配与负载报告", fontsize=14, color=INK)
    fig.savefig(out, facecolor=SURFACE); print(f"图已保存: {out}")

if __name__ == "__main__": main()
