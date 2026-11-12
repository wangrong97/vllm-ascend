#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VQ2A8 混合专家权重 → BF16 全精度权重反量化转换器。
依据 docs/vq2a8/vq2_mixed_gs32_export_spec_20261016.md:
  VQ2 四字段(packed_indices/codebooks/perm/out_dim_scale)与 MXFP4(weight/scale)。
输出对齐 FP8 原始权重命名(msmodelslim 已验证兼容):
  layers.{L}.ffn.experts.{E}.w1/w3/w2.weight (bf16); 非 routed 原样复制。"""
import argparse, json, os, re, shutil, sys, time
import torch
from safetensors.torch import load_file, save_file

# 防御: tools/bisect 包遮蔽标准库 bisect(random→bisect 导入链),import torch 前过滤
sys.path = [p for p in sys.path if os.path.basename(os.path.normpath(p)) != "tools" or not os.path.isdir(os.path.join(p, "bisect"))]

EXPERT_PAT = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w13|w2)\.(\w+)$")
E2M1_LUT = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)

def dequant_vq2(packed, codebooks, perm, out_dim_scale, device):
    N = packed.shape[0] * 32; K = packed.shape[1] * 16
    lut = codebooks.view(torch.uint8).reshape(codebooks.shape[0], codebooks.shape[1], 32)
    try:
        lut = lut.view(torch.float8_e4m3fn).to(torch.float32)
    except Exception:
        b = lut.to(torch.int32)
        sign = torch.where(b >= 128, -1.0, 1.0); bits = torch.where(b >= 128, b - 128, b)
        exp = (bits >> 3) & 0xF; man = bits & 0x7
        val = torch.where(exp == 0, man.float() * 2**-9, (1.0 + man.float() / 8) * torch.pow(2.0, exp.float() - 7))
        val = torch.where(bits == 0x7F, torch.zeros_like(val), val); lut = sign * val
    lut = lut.to(device)
    n32, k16, ki, b_ = packed.shape
    idx_low = (packed & 0x0F).to(torch.int64).to(device); idx_high = (packed >> 4).to(torch.int64).to(device)
    n_grid = torch.arange(n32, device=device)[:, None, None, None, None]
    k32_grid = (torch.arange(k16, device=device)[None, :, None, None, None] // 2)
    d = torch.arange(2, device=device)
    cv_low = lut[n_grid, k32_grid, idx_low[..., None] * 2 + d]
    cv_high = lut[n_grid, k32_grid, idx_high[..., None] * 2 + d]
    W = torch.empty((N, K), dtype=torch.float32, device=device)
    rows = (torch.arange(n32, device=device)[:, None, None, None] * 32 + torch.arange(8, device=device)[None, None, None, :] * 4)[..., None]
    cols = (torch.arange(k16, device=device)[None, :, None, None, None] * 16 + torch.arange(ki, device=device)[None, None, :, None, None])
    r_low = rows + d; r_high = rows + 2 + d
    W[r_low.expand_as(cv_low), cols.expand_as(cv_low)] = cv_low
    W[r_high.expand_as(cv_high), cols.expand_as(cv_high)] = cv_high
    bits32 = (out_dim_scale.view(torch.int64) & 0xFFFFFFFF).to(torch.int32)
    scale = bits32.view(torch.float32).to(device)
    W *= scale[:, None]
    W_out = torch.empty_like(W); W_out[:, perm.to(device).to(torch.int64)] = W
    return W_out

def dequant_mxfp4(weight_u8, scale_e8m0, device):
    N, K2 = weight_u8.shape
    w = weight_u8.to(device); lo = (w & 0x0F).to(torch.int64); hi = (w >> 4).to(torch.int64)
    lut = E2M1_LUT.to(device)
    sign_lo = 1.0 - 2.0 * (lo >> 3).to(torch.float32); sign_hi = 1.0 - 2.0 * (hi >> 3).to(torch.float32)
    vals_lo = sign_lo * lut[lo & 7]; vals_hi = sign_hi * lut[hi & 7]
    codes = torch.empty((N, K2 * 2), dtype=torch.float32, device=device)
    codes[:, 0::2] = vals_lo; codes[:, 1::2] = vals_hi
    s = scale_e8m0.view(torch.uint8).to(torch.float32).to(device); s = torch.pow(2.0, s - 127.0)
    s_full = s.repeat_interleave(32, dim=1)[:, : K2 * 2]
    return codes * s_full

def rename_non_routed(key):
    if key.startswith("model.layers."): return key[len("model."):]
    if key == "model.embed_tokens.weight": return "embed.weight"
    if key.startswith("model."): return key[len("model."):]
    return key

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True); ap.add_argument("--output", required=True)
    ap.add_argument("--fp8-config", default=None); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cpu", choices=["cpu", "npu"])
    args = ap.parse_args()
    if os.path.exists(args.output): sys.exit(f"输出目录已存在: {args.output}")
    os.makedirs(args.output)
    device = args.device
    if device == "npu": import torch_npu  # noqa
    idx = json.load(open(os.path.join(args.source, "model.safetensors.index.json")))["weight_map"]
    experts, non_routed = {}, {}
    for key, shard in idx.items():
        m = EXPERT_PAT.match(key)
        if m: experts.setdefault((int(m.group(1)), int(m.group(2)), m.group(3)), {})[m.group(4)] = shard
        else: non_routed[key] = shard
    cache = {}
    def tensors_of(shard):
        if shard not in cache:
            if len(cache) >= 3: cache.clear()
            cache[shard] = load_file(os.path.join(args.source, shard))
        return cache[shard]
    new_idx = {}; t0 = time.time(); done = 0
    for (L, E, P), fields in sorted(experts.items()):
        if args.limit and done >= args.limit: break
        shard = next(iter(fields.values())); ts = tensors_of(shard)
        pre = f"layers.{L}.ffn.experts.{E}.{P}."
        if "packed_indices" in fields:
            W = dequant_vq2(ts[pre + "packed_indices"], ts[pre + "codebooks"], ts[pre + "perm"], ts[pre + "out_dim_scale"], device)
        else:
            W = dequant_mxfp4(ts[pre + "weight"], ts[pre + "scale"], device)
        W = W.to(torch.bfloat16).cpu()
        out_shard = f"layer_{L}_e{E}_{P}.safetensors"
        if P == "w13":
            half = W.shape[0] // 2
            save_file({f"layers.{L}.ffn.experts.{E}.w1.weight": W[:half], f"layers.{L}.ffn.experts.{E}.w3.weight": W[half:]},
                      os.path.join(args.output, out_shard))
            new_idx[f"layers.{L}.ffn.experts.{E}.w1.weight"] = out_shard; new_idx[f"layers.{L}.ffn.experts.{E}.w3.weight"] = out_shard
        else:
            save_file({f"layers.{L}.ffn.experts.{E}.w2.weight": W}, os.path.join(args.output, out_shard))
            new_idx[f"layers.{L}.ffn.experts.{E}.w2.weight"] = out_shard
        done += 1
        if done % 256 == 0:
            rate = done / (time.time() - t0)
            print(f"  进度 {done}/{len(experts)} ({rate:.1f}/s, 剩余 {(len(experts)-done)/max(rate,0.01)/60:.0f} 分钟)", flush=True)
    if not args.limit:
        nr_out = {}
        for key, shard in sorted(non_routed.items()):
            nr_out[rename_non_routed(key)] = tensors_of(shard)[key].to(torch.bfloat16).cpu()
        save_file(nr_out, os.path.join(args.output, "non_routed.safetensors"), metadata={"format": "pt"})
        for k in nr_out: new_idx[k] = "non_routed.safetensors"
    json.dump({"metadata": {"total_size": 0}, "weight_map": new_idx},
              open(os.path.join(args.output, "model.safetensors.index.json"), "w"), indent=1)
    cfg_src = args.fp8_config or args.source
    cfg = json.load(open(os.path.join(cfg_src, "config.json"), encoding="utf-8"))
    cfg.pop("quantization_config", None); cfg["torch_dtype"] = "bfloat16"
    json.dump(cfg, open(os.path.join(args.output, "config.json"), "w"), indent=1, ensure_ascii=False)
    for name in os.listdir(args.source):
        if name.endswith((".json", ".model", ".txt")) and "safetensors" not in name and name != "config.json":
            shutil.copy(os.path.join(args.source, name), os.path.join(args.output, name))
    print(f"完成: {done} 投影 + {len(non_routed)} 非 routed → {args.output} ({(time.time()-t0)/60:.1f} 分钟)")
    if args.limit: print("[调试模式 --limit,勿用于量化]")

if __name__ == "__main__": main()
