# 构建混合 ckpt: attn 用原生 FP8 ckpt 的权重(+128x128 块 scale 展开为 per-32),
# 其余( MoE 专家、共享专家、gate、norm、embed、mtp 非 attn 部分等)用 attnw8a8-moew4a8。
# 目的:验证 GPQA 差距是否来自 msmodelslim 的 attn 量化值。
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

NATIVE = "/mnt/share/weight/DeepSeek-V4-Flash"  # attn 来源(官方量化值)
MOE = "/mnt/share/rr08002/weights/DeepSeek-V4-Flash-attnw8a8-moew4a8"  # 其余来源
DST = "/mnt/share/rr08002/weights/DeepSeek-V4-Flash-attnNative-moew4a8"

# 需要从原生替换的 attn 量化张量(模块名,不含 .weight)
ATTN_QUANT_SUFFIXES = ("wq_a", "wq_b", "wkv", "wo_a", "wo_b", "indexer.wq_b")


def is_attn_quant_key(key: str) -> bool:
    # layers.N.attn.<mod>.weight / layers.N.attn.<mod>.weight_scale / mtp.0.attn.<mod>.*
    parts = key.split(".")
    if ".attn" not in key:
        return False
    if not (key.endswith(".weight") or key.endswith(".weight_scale")):
        return False
    mod = parts[-2]
    return mod in ATTN_QUANT_SUFFIXES


def expand_native_scale(s_e8m0: torch.Tensor) -> torch.Tensor:
    """原生 [N/128, K/128] e8m0 块 scale → [N, K/32] uint8(e8m0 字节,广播到 per-32)"""
    b = s_e8m0.view(torch.uint8)
    b = b.repeat_interleave(128, dim=0).repeat_interleave(4, dim=1)
    return b.contiguous()


def main():
    os.makedirs(DST, exist_ok=True)
    nat_map = json.load(open(os.path.join(NATIVE, "model.safetensors.index.json")))["weight_map"]
    moe_map = json.load(open(os.path.join(MOE, "quant_model_weights.safetensors.index.json")))["weight_map"]
    nat_handles, moe_handles = {}, {}

    def nat_tensor(k):
        fn = nat_map[k]
        if fn not in nat_handles:
            nat_handles[fn] = safe_open(os.path.join(NATIVE, fn), framework="pt")
        return nat_handles[fn].get_tensor(k)

    shards = {}
    for k, fn in moe_map.items():
        shards.setdefault(fn, []).append(k)

    new_map = {}
    total_size = 0
    n_replaced = 0
    for si, (fn, keys) in enumerate(sorted(shards.items())):
        out = {}
        with safe_open(os.path.join(MOE, fn), framework="pt") as f:
            for k in keys:
                if is_attn_quant_key(k):
                    if k.endswith(".weight_scale"):
                        nat_key = k[: -len(".weight_scale")] + ".scale"
                        t = expand_native_scale(nat_tensor(nat_key))
                    else:
                        t = nat_tensor(k)  # FP8 权重字节直接沿用
                    n_replaced += 1
                else:
                    t = f.get_tensor(k)
                out[k] = t.contiguous()
        for k, t in out.items():
            new_map[k] = fn
            total_size += t.numel() * t.element_size()
        save_file(out, os.path.join(DST, fn), metadata={"format": "pt"})
        print(f"[{si+1}/{len(shards)}] {fn} done", flush=True)
        del out

    json.dump({"metadata": {"total_size": total_size}, "weight_map": new_map},
              open(os.path.join(DST, "quant_model_weights.safetensors.index.json"), "w"), indent=2)

    # 非权重文件全部沿用 attnw8a8-moew4a8(含 config.json、tokenizer、quant_model_description.json)
    for name in os.listdir(MOE):
        if name.endswith(".safetensors") or name.endswith(".index.json"):
            continue
        s, d = os.path.join(MOE, name), os.path.join(DST, name)
        if os.path.isdir(s):
            if os.path.exists(d):
                shutil.rmtree(d)
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)
    print(f"ALL DONE, replaced {n_replaced} attn tensors", flush=True)


if __name__ == "__main__":
    main()
