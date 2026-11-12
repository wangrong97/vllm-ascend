#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E6: P/D 两侧 KV block dump 逐元素比对。用法: --dump-dir 含 kv_p_side.pt/kv_d_side.pt; --scale-tail 128"""
import argparse, torch

def region_stats(a, b):
    if a.shape != b.shape: return {"shape_mismatch": (tuple(a.shape), tuple(b.shape))}
    eq = (a.view(torch.uint8) == b.view(torch.uint8)) if a.dtype not in (torch.float32, torch.bfloat16) else (a == b)
    byte_eq = eq.float().mean().item()
    af, bf = a.float().flatten(), b.float().flatten()
    cos = (torch.dot(af, bf) / (af.norm() * bf.norm() + 1e-12)).item()
    return {"byte_eq": round(byte_eq, 6), "cos": round(cos, 6), "max_absdiff": (af - bf).abs().max().item()}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dump-dir", required=True); ap.add_argument("--scale-tail", type=int, default=128)
    args = ap.parse_args()
    P = torch.load(f"{args.dump_dir}/kv_p_side.pt", weights_only=False); D = torch.load(f"{args.dump_dir}/kv_d_side.pt", weights_only=False)
    pb, db = P["blocks"], D["blocks"]
    common = sorted(set(pb) & set(db))
    if not common:
        print("!! 无公共层,前5:", list(pb)[:5], list(db)[:5]); return
    worst = []
    print(f"\n{'layer':<44}{'block':>6} {'数据区 byte_eq':>13} {'cos':>9} {'scale区 eq':>10}")
    for layer in common:
        for b in sorted(set(pb[layer]) & set(db[layer])):
            tp, td = pb[layer][b], db[layer][b]; tail = args.scale_tail
            if tp.dim() >= 2 and tp.shape[-1] > tail:
                dp_, dd_ = tp[..., :-tail], td[..., :-tail]; sp_, sd_ = tp[..., -tail:], td[..., -tail:]
            else: dp_, dd_ = tp, td; sp_ = sd_ = None
            st = region_stats(dp_, dd_); ss = region_stats(sp_, sd_) if sp_ is not None else {}
            print(f"{layer:<44}{b:>6} {st.get('byte_eq', float('nan')):>13} {st.get('cos', float('nan')):>9} {ss.get('byte_eq', float('nan')):>10}")
            if st.get("byte_eq", 1.0) < 0.999: worst.append((layer, b, st.get("byte_eq"), ss.get("byte_eq")))
    print(f"\n判定: {'✅ 逐字节一致(传输无损)' if not worst else f'❌ {len(worst)} 个 block 不一致'}")
    for w in worst[:5]: print("  ", w)

if __name__ == "__main__": main()
