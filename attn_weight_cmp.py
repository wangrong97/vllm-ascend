# attn 权重数值差异对比: 原生(128x128 块 scale) vs attnw8a8-moew4a8(per-32 scale)
# 参考系: BF16 目录(已证明 == 原生 FP8 反量化,逐位一致)
# 输出: 每层每张量的相对差异指标 + 3 张图
import json
import numpy as np
import torch
from safetensors import safe_open
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAT = "/mnt/share/weight/DeepSeek-V4-Flash"
MSM = "/mnt/share/rr08002/weights/DeepSeek-V4-Flash-attnw8a8-moew4a8"
REF = "/mnt/share/weight/DeepSeek-V4-Flash-BF16"
OUT = "/mnt/share/rr08002/work/vllm-ascend/ais_bench_logs/attn_weight_cmp"

import os
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8a8a"

nat_idx = json.load(open(f"{NAT}/model.safetensors.index.json"))["weight_map"]
msm_idx = json.load(open(f"{MSM}/quant_model_weights.safetensors.index.json"))["weight_map"]
ref_idx = json.load(open(f"{REF}/model.safetensors.index.json"))["weight_map"]

_handles = {}
def get(idx, base, k):
    fn = idx[k]
    key = (base, fn)
    if key not in _handles:
        _handles[key] = safe_open(f"{base}/{fn}", framework="pt")
    return _handles[key].get_tensor(k)

def dequant_nat(k):
    """原生: fp8 [N,K] + e8m0 [N/128, K/128]"""
    w = get(nat_idx, NAT, k)
    s = get(nat_idx, NAT, k.replace(".weight", ".scale"))
    return w.to(torch.float32) * s.to(torch.float32).repeat_interleave(128, 0).repeat_interleave(128, 1)

def dequant_msm(k):
    """msmodelslim: fp8 [N,K] + e8m0 [N, K/32]"""
    w = get(msm_idx, MSM, k)
    s = get(msm_idx, MSM, k.replace(".weight", ".weight_scale"))
    s = s.view(torch.float8_e8m0fnu).to(torch.float32)  # uint8 -> e8m0(2^(v-127))
    return w.to(torch.float32) * s.repeat_interleave(32, 1)

TENSORS = ["wq_a", "wq_b", "wkv", "wo_a", "wo_b"]
LAYERS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 42]

rows = []
for layer in LAYERS:
    for t in TENSORS:
        k = f"layers.{layer}.attn.{t}.weight"
        ref = get(ref_idx, REF, k).to(torch.float32)
        dn = dequant_nat(k)
        dm = dequant_msm(k)
        assert torch.equal(dn.to(torch.bfloat16), ref.to(torch.bfloat16)), f"{k}: native != ref"
        diff = (dm - ref)
        ref_norm = ref.norm().item()
        rows.append(dict(
            layer=layer, tensor=t,
            rel_fro=(diff.norm() / ref_norm).item(),
            mean_abs=diff.abs().mean().item(),
            max_abs=diff.abs().max().item(),
            ref_abs_mean=ref.abs().mean().item(),
        ))
        print(f"L{layer:2d} {t:5s} relFro={rows[-1]['rel_fro']:.5f} meanAbs={rows[-1]['mean_abs']:.6f} maxAbs={rows[-1]['max_abs']:.4f}", flush=True)

import csv
with open(f"{OUT}/metrics.csv", "w", newline="") as f:
    wcsv = csv.DictWriter(f, fieldnames=rows[0].keys())
    wcsv.writeheader(); wcsv.writerows(rows)

# ============ 图 1: 每张量类型 relFro 随层变化 ============
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c9c9c7", "axes.linewidth": 0.8})
fig, ax = plt.subplots(figsize=(10, 3.8))
colors = {"wq_a": BLUE, "wq_b": ORANGE, "wkv": "#1baf7a", "wo_a": "#4a3aa7", "wo_b": "#e87ba4"}
for t in TENSORS:
    xs = [r["layer"] for r in rows if r["tensor"] == t]
    ys = [r["rel_fro"] for r in rows if r["tensor"] == t]
    ax.plot(xs, ys, marker="o", ms=4, lw=1.6, color=colors[t], label=t)
ax.set_yscale("log")
ax.set_xlabel("Layer"); ax.set_ylabel("relative Frobenius diff (log)")
ax.set_title("attn weight diff vs native FP8 values (msmodelslim attn, dequantized)")
ax.legend(frameon=False, ncol=5)
ax.grid(color="#e4e4e1", lw=0.6); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(f"{OUT}/rel_fro_per_layer.png", dpi=160)

# ============ 图 2: 代表张量的差异分布直方图 ============
K0 = "layers.20.attn.wq_b.weight"
ref0 = get(ref_idx, REF, K0).to(torch.float32)
dm0 = dequant_msm(K0)
d0 = (dm0 - ref0).flatten().numpy()
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.hist(d0, bins=200, color=BLUE, lw=0)
ax.set_yscale("log")
ax.set_xlabel("signed diff (msmodelslim dequant - native values)")
ax.set_ylabel("count (log)")
ax.set_title(f"Diff histogram: {K0}  (std={d0.std():.2e})")
ax.grid(color="#e4e4e1", lw=0.6); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(f"{OUT}/diff_hist_wq_b_L20.png", dpi=160)

# ============ 图 3: scale 粒度对比热图 (layers.20.attn.wq_a) ============
KS = "layers.20.attn.wq_a.weight"
sn = get(nat_idx, NAT, KS.replace(".weight", ".scale")).view(torch.uint8).float()          # [8,32] e8m0 指数
sm = get(msm_idx, MSM, KS.replace(".weight", ".weight_scale")).view(torch.uint8).float()   # [1024,128]
sn_up = sn.repeat_interleave(128, 0).repeat_interleave(4, 1)                                # 广播到 [1024,128]
delta = sm - sn_up                                                                           # per-32 与块 scale 的指数差
fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), constrained_layout=True)
for ax, mat, title, cmap in [
    (axes[0], sn_up.numpy(), "native: 128x128 block scale (broadcast)", "viridis"),
    (axes[1], sm.numpy(), "msmodelslim: per-32 scale", "viridis"),
    (axes[2], delta.numpy(), "delta (exponent diff)", "RdBu_r"),
]:
    im = ax.imshow(mat, aspect="auto", cmap=cmap)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("K group"); ax.set_ylabel("N (output)")
    fig.colorbar(im, ax=ax, shrink=0.8)
axes[2].images[0].set_clim(-abs(delta).max().item(), abs(delta).max().item())
fig.suptitle(f"Scale granularity comparison: {KS}", fontsize=11)
fig.savefig(f"{OUT}/scale_granularity_wq_a_L20.png", dpi=160)

# ============ scale 统计: 每个 128 块内 per-32 scale 的变异 ============
var_ratio = (sm.reshape(8, 128, 32, 4).std(dim=(1, 3)) / (sn + 1e-6)).numpy()
print("\n=== scale 差异摘要 ===")
print(f"per-32 scale 与块 scale 的指数差: mean={delta.mean():.3f} std={delta.std():.3f} max|.|={delta.abs().max():.0f}")
print(f"块内 per-32 scale 相对变异(std/scale): mean={var_ratio.mean():.3f}")
print(f"\n全部指标 -> {OUT}/metrics.csv; 图 -> rel_fro_per_layer.png / diff_hist_wq_b_L20.png / scale_granularity_wq_a_L20.png")
