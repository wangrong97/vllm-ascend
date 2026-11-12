#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""专家静态分配报告:各卡专家数与量化精度(vq2/mxfp4)分布。"""
import argparse, json, os, re, sys
from collections import Counter, defaultdict

KEY_PAT = re.compile(r"layers\.(\d+)\.ffn\.experts\.(\d+)\.(w13|w2)\.(.+)$")
VQ2_FIELDS = {"codebooks", "packed_indices"}

def parse_weight_index(model_dir):
    p = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.isfile(p): sys.exit(f"未找到权重索引: {p}")
    idx = json.load(open(p, encoding="utf-8"))
    votes = defaultdict(Counter)
    for key in idx.get("weight_map", {}):
        m = KEY_PAT.search(key)
        if not m: continue
        e, proj, field = int(m.group(2)), m.group(3), m.group(4)
        if field in VQ2_FIELDS: votes[(e, proj)]["vq2"] += 1
        elif field in ("weight", "scale"): votes[(e, proj)]["mxfp4"] += 1
    result = {}
    for (e, proj), c in votes.items(): result[(e, proj)] = c.most_common(1)[0][0]
    return result

def linear_rank(e, num_experts, ep_size, mode):
    return e // (num_experts // ep_size) if mode == "block" else e % ep_size

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ep-size", type=int, default=4)
    ap.add_argument("--mode", choices=["block", "stride"], default="block")
    ap.add_argument("--runtime-dump", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    quant = parse_weight_index(args.model)
    n = max(e for e, _ in quant) + 1
    rank_experts = {r: [e for e in range(n) if linear_rank(e, n, args.ep_size, args.mode) == r] for r in range(args.ep_size)}
    if args.runtime_dump and os.path.isdir(args.runtime_dump):
        rt = {}
        for name in os.listdir(args.runtime_dump):
            m = re.match(r"resident_experts_rank(\d+)\.json", name)
            if m:
                rt[int(m.group(1))] = json.load(open(os.path.join(args.runtime_dump, name)))["expert_ids"]
        if rt: rank_experts = rt
    cards = []
    for rank in sorted(rank_experts):
        ex = sorted(rank_experts[rank])
        w13 = [quant.get((e, "w13"), "?") for e in ex]; w2 = [quant.get((e, "w2"), "?") for e in ex]
        cards.append({"rank": rank, "num_experts": len(ex), "expert_ids": ex,
                      "w13_vq2": sum(1 for q in w13 if q == "vq2"), "w13_mxfp4": sum(1 for q in w13 if q == "mxfp4"),
                      "w2_vq2": sum(1 for q in w2 if q == "vq2"), "w2_mxfp4": sum(1 for q in w2 if q == "mxfp4")})
    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, "placement_report.json")
    json.dump({"model": os.path.basename(os.path.abspath(args.model)), "num_experts": n,
               "ep_size": args.ep_size, "cards": cards,
               "expert_quant": {f"{e}_{proj}": q for (e, proj), q in sorted(quant.items())}},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{'NPU卡':>6} {'专家数':>6} {'w13-vq2':>8} {'w13-mxfp4':>10} {'w2-mxfp4':>9}")
    for c in cards: print(f"{c['rank']:>6} {c['num_experts']:>6} {c['w13_vq2']:>8} {c['w13_mxfp4']:>10} {c['w2_mxfp4']:>9}")
    print(f"报告: {out}")

if __name__ == "__main__": main()
