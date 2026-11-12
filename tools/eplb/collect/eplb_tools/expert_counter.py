#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""运行时专家路由计数器(hook 模块)。

由 install_expert_counter.sh 注入 vllm_ascend/ops/fused_moe/routed_experts.py
的插桩点调用:每层 MoE forward 拿到 topk_ids 后,按全局专家 id 累计直方图。
仅当环境变量 EPLB_EXPORT_DIR 非空时激活;未设置时插桩点开销为一次 dict 查询。

环境变量:
  EPLB_EXPORT_DIR     输出目录(激活开关,空 = 关闭)
  EPLB_NUM_EXPERTS    全局逻辑专家数(默认 256,DeepSeek-V4-Flash)
  EPLB_COUNT_STRIDE   每 N 次记录 1 次(默认 1)
  EPLB_FLUSH_EVERY    每 N 次落盘一次(默认 2000)

输出文件(每个 worker 进程一份):
  expert_counts_<rank>.json   256 维 token 计数直方图 + 元信息
  resident_experts_<rank>.json 本 rank 常驻专家表(运行时真值)
"""
import atexit
import json
import os
import time

import torch

_DIR = os.environ.get("EPLB_EXPORT_DIR", "")
_STRIDE = max(1, int(os.environ.get("EPLB_COUNT_STRIDE", "1")))
_FLUSH_EVERY = max(1, int(os.environ.get("EPLB_FLUSH_EVERY", "2000")))
_NUM_EXPERTS = int(os.environ.get("EPLB_NUM_EXPERTS", "256"))

_state = {
    "counts": None,
    "calls": 0,
    "start": time.time(),
    "placement_dumped": False,
    "registered": False,
}


def _rank_tag() -> str:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return f"rank{dist.get_rank()}"
    except Exception:
        pass
    return f"pid{os.getpid()}"


def _atomic_dump(final: bool) -> None:
    if _state["counts"] is None or not _DIR:
        return
    path = os.path.join(_DIR, f"expert_counts_{_rank_tag()}.json")
    tmp = path + ".tmp"
    payload = {
        "rank_tag": _rank_tag(),
        "num_experts": _NUM_EXPERTS,
        "stride": _STRIDE,
        "calls_seen": _state["calls"],
        "wall_seconds": round(time.time() - _state["start"], 1),
        "final": bool(final),
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "counts": _state["counts"].tolist(),
    }
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp, path)
    except Exception:
        pass


def record(module, topk_ids) -> None:
    if not _DIR:
        return
    st = _state
    if st["counts"] is None:
        st["counts"] = torch.zeros(_NUM_EXPERTS, dtype=torch.int64, device=topk_ids.device)
        if not st["registered"]:
            atexit.register(lambda: _atomic_dump(True))
            st["registered"] = True
        if not st["placement_dumped"]:
            ids = getattr(module, "vq2a8_expert_ids", None)
            if ids is not None:
                try:
                    p = os.path.join(_DIR, f"resident_experts_{_rank_tag()}.json")
                    with open(p, "w", encoding="utf-8") as f:
                        json.dump({"rank_tag": _rank_tag(), "expert_ids": [int(i) for i in ids],
                                   "quant_type": str(getattr(module, "quant_type", "?")),
                                   "ts": time.strftime("%Y-%m-%d %H:%M:%S")}, f)
                    st["placement_dumped"] = True
                except Exception:
                    pass
    st["calls"] += 1
    if (st["calls"] - 1) % _STRIDE:
        return
    ids = topk_ids.reshape(-1).to(torch.int64).clamp_(0, _NUM_EXPERTS - 1)
    st["counts"] += torch.bincount(ids, minlength=_NUM_EXPERTS)
    if st["calls"] % _FLUSH_EVERY == 0:
        _atomic_dump(False)
