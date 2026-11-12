#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KV block 逐元素比对 hook(E6 终局定位实验)。
由 install_kv_block_hook.sh 注入 mooncake_connector.py 两处调用:
  P 侧: MooncakeConnectorWorker.get_finished 首次调用时 dump self.kv_caches 每层前 N 个 block
  D 侧: RecvingThread._transfer_kv_cache_all_groups 的 batch_transfer_sync_read 返回后按 plan 精确切片
环境变量: KV_DUMP_DIR(激活开关), KV_DUMP_BLOCKS(默认 8)"""
import os

_DUMPED = {}

def dump_side(role, kv_caches, plan=None, n_blocks=None, block_size=32, role_tag=None):
    """role='p': kv_caches={layer:pool}, dump 每层前 n_blocks; role='d': plan 精确切片。
    role='single': 同 p 前缀(单机逐层, 文件名带 role_tag)。幂等: 每 (role,role_tag) 只 dump 一次。"""
    d = os.environ.get("KV_DUMP_DIR", "")
    tag = role_tag or ""
    key = f"{role}:{tag}"
    if not d or _DUMPED.get(key):
        return
    try:
        _DUMPED[key] = True
        os.makedirs(d, exist_ok=True)
        blocks = {}
        if role in ("p", "single"):
            nb = int(os.environ.get("KV_DUMP_BLOCKS", n_blocks or 8))
            for layer, pool in (kv_caches or {}).items():
                if pool is None or not hasattr(pool, "shape") or pool.dim() < 1:
                    continue
                nb_eff = min(nb, pool.shape[0])
                blocks[str(layer)] = {i: pool[i].detach().to("cpu", copy=True) for i in range(nb_eff)}
        else:
            for layer, bids in (plan or []):
                pool = kv_caches.get(layer)
                if pool is None: continue
                blocks.setdefault(layer, {})
                for b in bids:
                    b = b[0] if isinstance(b, (list, tuple)) else b
                    if 0 <= int(b) < pool.shape[0]:
                        blocks[layer][int(b)] = pool[int(b)].detach().to("cpu", copy=True)
        import torch
        suffix = f"_{tag}" if tag else ""
        torch.save({"role": role, "blocks": blocks, "meta": {"tag": tag, "ts": __import__("time").strftime("%F %T")}},
                   os.path.join(d, f"kv_{role}_side{suffix}.pt"))
        print(f"[kv-block-hook] {role}{suffix} dumped {sum(len(v) for v in blocks.values())} blocks -> {d}", flush=True)
    except Exception as exc:
        print(f"[kv-block-hook] {role} dump failed: {exc}", flush=True)
