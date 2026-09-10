# 将原生 FP8 ckpt(/mnt/share/weight/DeepSeek-V4-Flash)反量化为 BF16 权重
# 两种量化张量:
#   1) FP8:  weight F8_E4M3 [N,K] + scale F8_E8M0 [N/128, K/128]  (128x128 块, attn/共享专家/mtp)
#   2) FP4:  weight I8 [N,K/2](e2m1 双打包) + scale F8_E8M0 [N, K/32]  (per-32, MoE 专家)
# 其余张量原样拷贝。输出目录结构与源一致(去掉 .scale,重写 index)。
import json
import os
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = "/mnt/share/weight/DeepSeek-V4-Flash"
DST = "/mnt/share/rr08002/weights/DeepSeek-V4-Flash-BF16-fromfp8"

# OCP e2m1 查找表: code = s e1 e0 m → (-1)^s * val
_POS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
_LUT = torch.tensor(_POS + [-v for v in _POS], dtype=torch.float32)  # 索引 0-15


def dequant_fp8(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    n, k = w.shape
    sn, sk = s.shape
    assert n == sn * 128 and k == sk * 128, f"fp8 block mismatch: {w.shape} vs {s.shape}"
    out = w.to(torch.float32) * s.to(torch.float32).repeat_interleave(128, 0).repeat_interleave(128, 1)
    return out.to(torch.bfloat16)


def dequant_fp4(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    n, k2 = w.shape
    sn, sk = s.shape
    assert n == sn and k2 * 2 == sk * 32, f"fp4 block mismatch: {w.shape} vs {s.shape}"
    b = w.to(torch.uint8)
    lo = _LUT[(b & 0xF).long()]   # 低 4 位 = 偶数元素(OCP fp4x2)
    hi = _LUT[(b >> 4).long()]    # 高 4 位 = 奇数元素
    vals = torch.stack([lo, hi], dim=2).flatten(1)  # [N, K]
    out = vals * s.to(torch.float32).repeat_interleave(32, 1)
    return out.to(torch.bfloat16)


def main():
    os.makedirs(DST, exist_ok=True)
    idx = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))
    weight_map = idx["weight_map"]
    shards = {}
    for k, fn in weight_map.items():
        shards.setdefault(fn, []).append(k)

    new_map = {}
    total_size = 0
    for si, (fn, keys) in enumerate(sorted(shards.items())):
        sd_path = os.path.join(SRC, fn)
        out_sd = {}
        with safe_open(sd_path, framework="pt") as f:
            keyset = set(keys)
            for k in keys:
                if k.endswith(".scale"):
                    continue
                scale_key = k + ".scale"
                t = f.get_tensor(k)
                if scale_key in keyset:
                    s = f.get_tensor(scale_key)
                    if t.dtype == torch.float8_e4m3fn:
                        t = dequant_fp8(t, s)
                    elif t.dtype == torch.int8:
                        t = dequant_fp4(t, s)
                    else:
                        raise ValueError(f"{k}: unexpected quantized dtype {t.dtype}")
                out_sd[k] = t.contiguous()
        for k, t in out_sd.items():
            new_map[k] = fn
            total_size += t.numel() * t.element_size()
        save_file(out_sd, os.path.join(DST, fn), metadata={"format": "pt"})
        print(f"[{si+1}/{len(shards)}] {fn}: {len(out_sd)} tensors done", flush=True)
        del out_sd

    new_idx = {"metadata": {"total_size": total_size}, "weight_map": new_map}
    with open(os.path.join(DST, "model.safetensors.index.json"), "w") as f:
        json.dump(new_idx, f, indent=2)

    # 拷贝非权重文件;config.json 去掉 quantization_config
    for name in os.listdir(SRC):
        if name.endswith(".safetensors") or name == "model.safetensors.index.json":
            continue
        src_p = os.path.join(SRC, name)
        dst_p = os.path.join(DST, name)
        if os.path.isdir(src_p):
            if os.path.exists(dst_p):
                shutil.rmtree(dst_p)
            shutil.copytree(src_p, dst_p)
        elif name == "config.json":
            cfg = json.load(open(src_p))
            cfg.pop("quantization_config", None)
            json.dump(cfg, open(dst_p, "w"), indent=2)
        else:
            shutil.copy2(src_p, dst_p)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
