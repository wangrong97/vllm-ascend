#!/usr/bin/env python3
"""GLM-5.3-Flash kpool indexer q/k 激活值抓取工具(在线 hook 版)。

原理: 在 Glm5NextKPoolIndexerBackend 的 q 打分点与 k 池化写入点各挂一个
forward hook, 按环境变量配置把张量统计(或原始值)落盘 npz/json, 用于:
  - 量化前后的分布对比(离群值/动态范围/scale 覆盖)
  - MXFP4/MXFP8 伪量化误差的逐层定位
  - Hadamard 旋转前后分布变化

用法:
  1) 起服务前导出环境变量:
       VLLM_ASCEND_INDEXER_DUMP=1                     # 总开关
       VLLM_ASCEND_INDEXER_DUMP_DIR=/path/to/out      # 输出目录(默认 /tmp/indexer_dump)
       VLLM_ASCEND_INDEXER_DUMP_STAGES=q,k            # 抓取点(逗号分隔: q,k,均默认 q,k)
       VLLM_ASCEND_INDEXER_DUMP_LAYERS=3,7            # 只抓这些 DSA 层(逗号分隔; 默认全部)
       VLLM_ASCEND_INDEXER_DUMP_MAX_STEP=50           # 最多抓多少个 forward 步(默认 20, 防)
       VLLM_ASCEND_INDEXER_DUMP_MODE=stats            # stats=统计量 | raw=原始张量(慎用)
  2) 正常起服务发请求; 停服或达到 MAX_STEP 后, 输出目录生成:
       q_layer{N}_step{S}.npz / k_layer{N}_step{S}.npz  (raw 模式)
       summary.json                                     (两种模式都写)

注意: 仅用于离线分析, 有性能开销(raw 模式含 D2H 拷贝); 抓完即关。
"""
from __future__ import annotations

import json
import os
import threading

import torch

_DUMPER = None
_LOCK = threading.Lock()


def _env(name, default=None):
    return os.environ.get(name, default)


def _enabled() -> bool:
    return _env("VLLM_ASCEND_INDEXER_DUMP", "0") == "1"


class _IndexerActDumper:
    def __init__(self):
        self.dir = _env("VLLM_ASCEND_INDEXER_DUMP_DIR", "/tmp/indexer_dump")
        self.stages = set(s.strip() for s in _env("VLLM_ASCEND_INDEXER_DUMP_STAGES", "q,k").split(","))
        layers = _env("VLLM_ASCEND_INDEXER_DUMP_LAYERS", "")
        self.layers = set(int(x) for x in layers.split(",") if x.strip()) if layers else None
        self.max_step = int(_env("VLLM_ASCEND_INDEXER_DUMP_MAX_STEP", "20"))
        self.mode = _env("VLLM_ASCEND_INDEXER_DUMP_MODE", "stats")
        self.step = 0
        self.records = []
        os.makedirs(self.dir, exist_ok=True)
        with open(os.path.join(self.dir, "config.json"), "w") as f:
            json.dump({"stages": sorted(self.stages), "layers": sorted(self.layers) if self.layers else "all",
                       "max_step": self.max_step, "mode": self.mode}, f, indent=2)

    def _layer_of(self, module) -> int | None:
        # backend 挂在 attention 层下: prefix 形如 ...layers.{N}.self_attn...
        pfx = getattr(module, "prefix", "") or ""
        import re
        m = re.search(r"layers\.(\d+)\.", pfx)
        return int(m.group(1)) if m else None

    def _rec(self, stage, layer, t):
        if self.step >= self.max_step:
            return
        if self.layers is not None and layer not in self.layers:
            return
        if t is None or t.numel() == 0:
            return
        try:
            if torch.isnan(t).any() or torch.isinf(t).any():
                return  # 跳过含 nan/inf 的批次(保护拟合)
        except Exception:
            pass  # 检查本身失败不阻断
        rec = {"stage": stage, "layer": layer, "step": self.step,
               "shape": list(t.shape), "dtype": str(t.dtype)}
        if self.mode == "raw":
            path = os.path.join(self.dir, f"{stage}_layer{layer}_step{self.step}.npz")
            try:
                import numpy as np
                np.savez(path, data=t.detach().float().cpu().numpy())
                rec["file"] = path
            except Exception as e:  # noqa: BLE001
                rec["error"] = str(e)[:120]
        else:
            tf = t.detach().float()
            rec.update({
                "absmax": float(tf.abs().max()), "absmean": float(tf.abs().mean()),
                "absmin": float(tf.abs().min()), "std": float(tf.std()), "mean": float(tf.mean()),
                "p999": float(tf.abs().flatten().kthvalue(max(1, int(tf.numel() * 0.999)))[0]),
            })
        self.records.append(rec)
        if self.mode == "raw":
            self.flush()  # raw 模式即时落盘, 防服务中止丢 summary

    def on_forward(self, module, args, kwargs):
        """挂在 Glm5NextKPoolIndexerBackend.forward 上: 抓 q_values(打分前)。"""
        if "q" in self.stages and self.step < self.max_step:
            # 层号兜底: backend 可能无 prefix 属性, 取 _oscar_layer, 再退 -1
            layer = self._layer_of(module)
            if layer is None:
                layer = getattr(module, "_oscar_layer", -1)
            # forward(hidden, q_c, k_hidden, indexer_metadata, compute_topk)
            q_c = args[1] if len(args) > 1 else kwargs.get("q_c")
            # 与 backend 同口径: wq_b 投影后的 q_values
            try:
                qv = module.wq_b(q_c)[0].view(q_c.shape[0], module.n_head, module.head_dim)
                self._rec("q", layer, qv)
            except Exception as e:
                if self.step == 0:
                    print(f"[indexer-dump] q capture failed: {e}")

    def on_write(self, module, pooled_key, tail_key=None):
        """挂在写 cache 前: 抓池化 k(压缩后)与尾池 k。module 可为 None(层号记 -1)。"""
        if self.step < self.max_step:
            layer = self._layer_of(module) if module is not None else None
            layer = -1 if layer is None else layer
            if "k" in self.stages:
                self._rec("k_pooled", layer, pooled_key)
            if tail_key is not None:
                self._rec("k_tail", layer, tail_key)

    def tick(self):
        """每个 forward 结束步进一次; 达到上限时落 summary。"""
        self.step += 1
        if self.step == self.max_step:
            self.flush()

    def flush(self):
        with open(os.path.join(self.dir, "summary.json"), "w") as f:
            json.dump({"records": self.records, "total": len(self.records)}, f, indent=2)


def _get_dumper():
    global _DUMPER
    with _LOCK:
        if _DUMPER is None:
            _DUMPER = _IndexerActDumper()
        return _DUMPER


def install() -> bool:
    """给 Glm5NextKPoolIndexerBackend 安装抓取 hook。服务启动时自动调用。"""
    if not _enabled():
        return False
    try:
        from vllm_ascend.attention.indexer_kpool import Glm5NextKPoolIndexerBackend
        d = _get_dumper()
        if getattr(Glm5NextKPoolIndexerBackend, "_act_dump_installed", False):
            return True

        _orig_forward = Glm5NextKPoolIndexerBackend.forward

        def _fwd(self, *a, **kw):
            d.on_forward(self, a, kw)
            try:
                out = _orig_forward(self, *a, **kw)
            finally:
                d.tick()
            return out

        Glm5NextKPoolIndexerBackend.forward = _fwd
        Glm5NextKPoolIndexerBackend._act_dump_installed = True

        # k 侧: 包一层 write_keypool_caches, 在写 cache 前抓 pooled(池化 k)
        import vllm_ascend.ops.triton.glm5_next_cann_keypool as _ck
        if not getattr(_ck.write_keypool_caches, "_act_dump_wrapped", False):
            _orig_write = _ck.write_keypool_caches

            def _write(tail, cache, key, gate, pooled, metadata, tail_metadata, state, scale_cache=None):
                d.on_write(None, pooled, tail_key=key)
                return _orig_write(tail, cache, key, gate, pooled, metadata, tail_metadata, state,
                                   scale_cache=scale_cache)

            _write._act_dump_wrapped = True
            _ck.write_keypool_caches = _write

        print(f"[indexer-dump] hooks installed -> {d.dir} (stages={sorted(d.stages)}, mode={d.mode})")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[indexer-dump] install failed: {e}")
        return False


if __name__ == "__main__":
    # 独立运行: 打印配置说明
    print(__doc__)
