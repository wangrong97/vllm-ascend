#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""构造 attnD 版 checkpoint: attention/root 部分与 D(W2A8 repack) 完全一致
(同权重同格式同 loader 路径 ds_fp8_block128), MoE 保留源权重的数值。

与 make_p_vq2a8.py 同一方案, 源参数化(默认 4over6):
布局: root(非 expert)键 → 硬链 D 原分片; expert 键 → 源 w1+w3 合并 w13,
      scale 位模式直拷(U8 视图)。config/quant.json 拷 D(counts 修正全 mxfp4)。

用法:
  python3 make_attnD_vq2a8.py --src <msmodelslim 量化权重目录> \
      --out <输出目录> [--limit N]
  --limit N: 只转换前 N 层(快速验证; 全量转换必须不带 limit 重新跑)
"""
import argparse, json, os, re, shutil, sys
import torch
from safetensors.torch import load_file, save_file

D = "/mnt/share/w00608002/weight/DeepSeek-V4-Flash-0731-vq2a8-mixed-gs32-repack"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="msmodelslim 量化权重目录(含 quant_model_weights.safetensors.index.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="只转换前 N 层(0=全量)")
    args = ap.parse_args()

    W, out = args.src, args.out
    if os.path.exists(out):
        sys.exit(f"输出已存在: {out}")
    os.makedirs(out)
    didx = json.load(open(f"{D}/model.safetensors.index.json"))["weight_map"]
    widx = json.load(open(f"{W}/quant_model_weights.safetensors.index.json"))["weight_map"]

    # 1) root(非 expert)分片硬链 D —— attn 权重+scale+root 全部逐比特与 D 一致
    root_shards = sorted({s for k, s in didx.items() if ".ffn.experts." not in k})
    for s in root_shards:
        dst = f"{out}/{s}"
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.link(f"{D}/{s}", dst)

    # 2) MoE: 源 expert w1+w3→w13, w2 直拷; scale 位模式直拷(U8 视图)
    pat = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w1|w2|w3)\.(weight|weight_scale)$")
    plan = {}
    for k in sorted(widx):
        m = pat.match(k)
        if m:
            L, E, part, field = int(m[1]), int(m[2]), m[3], m[4]
            if args.limit and L >= args.limit:
                continue
            plan.setdefault((L, E), {}).setdefault(part, {})[field] = k

    new_idx, shard_no, shard_dt, shard_sz = {}, 0, {}, 0
    shard_name = None
    TOTAL = 4 * 1024**3

    def flush():
        nonlocal shard_dt, shard_sz
        if not shard_dt:
            return
        save_file(shard_dt, f"{out}/{shard_name}", metadata={"format": "pt"})
        shard_dt, shard_sz = {}, 0

    n_done = 0
    for (L, E), parts in sorted(plan.items()):
        for merged, srcs in (("w13", ("w1", "w3")), ("w2", ("w2",))):
            def get(field, p):
                shard = widx[parts[p][field]]
                return load_file(f"{W}/{shard}")[parts[p][field]]
            w = torch.cat([get("weight", p) for p in srcs], dim=0)
            s = torch.cat([get("weight_scale", p) for p in srcs], dim=0)
            sz = w.numel() + s.numel()
            if shard_name is None or shard_sz + sz > TOTAL:
                flush()
                shard_no += 1
                shard_name = f"moe/moe-w4a4-{shard_no:05d}.safetensors"
                os.makedirs(f"{out}/moe", exist_ok=True)
            shard_dt[f"layers.{L}.ffn.experts.{E}.{merged}.weight"] = w
            shard_dt[f"layers.{L}.ffn.experts.{E}.{merged}.scale"] = s.view(torch.uint8)
            new_idx[f"layers.{L}.ffn.experts.{E}.{merged}.weight"] = shard_name
            new_idx[f"layers.{L}.ffn.experts.{E}.{merged}.scale"] = shard_name
            shard_sz += sz
        n_done += 1
        if n_done % 512 == 0:
            print(f"  已转换 expert 组: {n_done}/{len(plan)}", flush=True)
    flush()

    # 3) index 汇总(root 键沿用 D 的分片引用)
    for k, s in didx.items():
        if ".ffn.experts." not in k:
            new_idx[k] = s
    json.dump({"metadata": {"total_size": 0}, "weight_map": new_idx},
              open(f"{out}/model.safetensors.index.json", "w"), indent=1)

    # 4) config/tokenizer 等拷 D; quant.json counts 修正(单格式 expert 全 mxfp4)
    for f in os.listdir(D):
        if f.endswith((".json", ".model", ".txt")) and "index" not in f:
            shutil.copy(f"{D}/{f}", f"{out}/{f}")
    q = json.load(open(f"{out}/quant.json"))
    q["counts"] = {"w13_vq": 0, "w13_mxfp4": len(plan), "w2_vq": 0, "w2_mxfp4": len(plan)}
    json.dump(q, open(f"{out}/quant.json", "w"), indent=1)
    print(f"attnD 版完成: {out} (expert 组 {len(plan)}, 层数限制={args.limit or '全量'})")


if __name__ == "__main__":
    main()
