#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P'' 修复: expert .scale 键 safetensors dtype 从 U8 改标 F8_E8M0(数据字节不变)。"""
import json, os, struct, sys
ROOT = "/mnt/share/w00608002/weight/DeepSeek-V4-Flash-0731-moeW4A4-attnW2A8"
idx_p = os.path.join(ROOT, "model.safetensors.index.json")
shards = sorted({s for k, s in json.load(open(idx_p))["weight_map"].items() if s.startswith("moe-")})
print(f"待修分片: {len(shards)}")
for s in shards:
    path = os.path.join(ROOT, s)
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]; header_raw = f.read(hlen); data = f.read()
    header = json.loads(header_raw); changed = 0
    for k, meta in header.items():
        if k.endswith(".scale") and meta["dtype"] == "U8": meta["dtype"] = "F8_E8M0"; changed += 1
    if not changed: continue
    new_raw = json.dumps(header, separators=(",", ":")).encode()
    with open(path + ".tmp", "wb") as f:
        f.write(struct.pack("<Q", len(new_raw))); f.write(new_raw); f.write(data)
    os.replace(path + ".tmp", path)
    print(f"  {s}: {changed} 个 → F8_E8M0")
print("完成")
