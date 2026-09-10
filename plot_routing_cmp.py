# MoE 路由分布对比: hybrid(attnNative+moeW4A8) vs 原生 FP8, GPQA 198 题
# 输出 3 张 PNG 到 ais_bench_logs/routed_experts/
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8a8a"
OUT = "/mnt/share/rr08002/work/vllm-ascend/ais_bench_logs/routed_experts"

H = np.load(f"{OUT}/hybrid.npz")["counts"].astype(float)   # [43, 256]
F = np.load(f"{OUT}/fp8native.npz")["counts"].astype(float)

Hp = H / H.sum(axis=1, keepdims=True)
Fp = F / F.sum(axis=1, keepdims=True)


def js_div(p, q, eps=1e-12):
    p = p + eps; q = q + eps
    p /= p.sum(); q /= q.sum()
    m = 0.5 * (p + q)
    kl = lambda a, b: np.sum(a * np.log(a / b))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


js_layers = np.array([js_div(Hp[l], Fp[l]) for l in range(43)])
js_global = js_div(H.sum(0), F.sum(0))
print(f"per-layer JS: mean={js_layers.mean():.6f} max={js_layers.max():.6f}@L{js_layers.argmax()} min={js_layers.min():.6f}@L{js_layers.argmin()}")
print(f"global JS: {js_global:.6f}")

Hg, Fg = H.sum(0), F.sum(0)
Hgs, Fgs = Hg / Hg.sum(), Fg / Fg.sum()
top = np.argsort(-np.abs(Hgs - Fgs))[:6]
print("largest per-expert share deltas:")
for e in top:
    print(f"  expert {e}: hybrid={Hgs[e]:.4%} native={Fgs[e]:.4%} delta={(Hgs[e]-Fgs[e]):+.4%}")
# topk 重合度: 每层每模型 top-6 集合的 Jaccard (粗粒度看路由是否换专家)
jac = []
for l in range(43):
    th = set(np.argsort(-Hp[l])[:6]); tf = set(np.argsort(-Fp[l])[:6])
    jac.append(len(th & tf) / len(th | tf))
jac = np.array(jac)
print(f"per-layer top6-set Jaccard: mean={jac.mean():.3f} min={jac.min():.3f}@L{jac.argmin()}")

plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c9c9c7", "axes.linewidth": 0.8})

# ---- Fig 1: per-layer JS divergence bar ----
fig, ax = plt.subplots(figsize=(10, 3.2))
ax.bar(range(43), js_layers, color=BLUE, width=0.7)
ax.axhline(js_layers.mean(), color=GRAY, lw=1, ls="--")
ax.annotate(f"max L{js_layers.argmax()}: {js_layers.max():.4f}",
            xy=(js_layers.argmax(), js_layers.max()), xytext=(8, 8), textcoords="offset points",
            fontsize=9, color="#0b0b0b")
ax.set_xlabel("Layer"); ax.set_ylabel("JS divergence")
ax.set_title("MoE routing distribution divergence per layer (hybrid vs native FP8, GPQA)")
ax.grid(axis="y", color="#e4e4e1", lw=0.6)
ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(f"{OUT}/moe_routing_js_per_layer.png", dpi=160)

# ---- Fig 2: per-expert share scatter (agreement) ----
fig, ax = plt.subplots(figsize=(5.2, 5.0))
ax.scatter(Fgs * 100, Hgs * 100, s=10, color=BLUE, alpha=0.5, linewidths=0)
lo = min(Fgs[Fgs > 0].min(), Hgs[Hgs > 0].min()) * 100
hi = max(Fgs.max(), Hgs.max()) * 100
ax.plot([lo, hi], [lo, hi], color=GRAY, lw=1, ls="--")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("native FP8 share (%)"); ax.set_ylabel("hybrid share (%)")
ax.set_title("Per-expert usage share (log-log)")
ax.grid(color="#e4e4e1", lw=0.6); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(f"{OUT}/moe_routing_expert_scatter.png", dpi=160)

# ---- Fig 3: sorted usage share curves ----
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.plot(np.sort(Fgs)[::-1] * 100, color=BLUE, lw=1.8, label="native FP8")
ax.plot(np.sort(Hgs)[::-1] * 100, color=ORANGE, lw=1.8, label="hybrid (native attn + moeW4A8)")
ax.set_yscale("log")
ax.set_xlabel("Expert rank (by usage share, desc)"); ax.set_ylabel("Usage share (%)")
ax.set_title("Expert usage share, sorted (GPQA, all layers)")
ax.legend(frameon=False)
ax.grid(color="#e4e4e1", lw=0.6); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(f"{OUT}/moe_routing_sorted_share.png", dpi=160)
print("saved 3 PNGs")
