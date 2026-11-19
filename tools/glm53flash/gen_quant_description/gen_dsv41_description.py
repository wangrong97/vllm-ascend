#!/usr/bin/env python3
"""DeepSeek-V4.1-Flash 量化配置 yaml → AscendV1 quant_model_description.json 离线生成。

与 glm53flash 版(tools/glm53flash/gen_quant_description)同思路, 按 V4.1 适配:
  - 键空间: layers.N.*/mtp.N.*/vision.*/aligner.*/embed/head/norm(无 model.language_model 前缀)
  - 只对标准 Linear 权重打标签; norm/sink/bias/gate 等非 Linear 键恒 FLOAT
    (与真实流水线"按模块类型"行为对齐 —— 名义 fnmatch 会误伤 kv_norm/q_norm/attn_norm)
  - mtp*/vision*/aligner* 由 yaml exclude 保证 FLOAT
  - 附 engram embed 的 INT8_SYM 声明(model_adapter.engram_quant_type=int8_sym 时)

用法:
  python3 gen_dsv41_description.py <config.yaml> <model_dir> [output_json]
默认输出到 <config 同目录>/quant_model_description.json
"""
from __future__ import annotations

import fnmatch
import json
import sys

import yaml

FLOAT = "FLOAT"
DTYPE_LABEL = {
    ("mxfp8", "mxfp8"): "W8A8_MXFP8",
    ("mxfp4", "mxfp4"): "W4A4_MXFP4",
    ("int8", "int8"): "W8A8_DYNAMIC",
    ("fp8_e4m3", "fp8_e4m3"): "W8A8_FP8",
}
PRIORITY = ["W4A4_MXFP4", "W8A8_MXFP8", "W8A8_FP8", "W8A8_DYNAMIC"]

# 非 Linear 命名特征(这些权重即使被 include 通配命中, 真实流水线也不量化)
NON_LINEAR_MARKERS = ("norm", "attn_sink", ".bias", "gate.weight", "gate.bias",
                      "embed", "head.weight", "proj.bias")


def load_rules(config_path: str) -> list[tuple[str, list, list]]:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    rules = []
    for proc in cfg["spec"]["process"]:
        if proc.get("type") != "linear_quant":
            continue
        qc = proc.get("qconfig") or {}
        key = (str(qc.get("act", {}).get("dtype")), str(qc.get("weight", {}).get("dtype")))
        if key not in DTYPE_LABEL:
            raise ValueError(f"未支持的 dtype 组合: {key}")
        rules.append((DTYPE_LABEL[key], proc.get("include") or [], proc.get("exclude") or []))
    if not rules:
        raise ValueError("yaml 中没有 linear_quant 块")
    return rules, cfg


def match_any(name: str, patterns) -> bool:
    return any(fnmatch.fnmatchcase(name, str(p)) for p in patterns)


def classify(module: str, rules) -> str | None:
    for label, includes, excludes in rules:
        if match_any(module, includes) and not match_any(module, excludes):
            return label
    return None


def main() -> None:
    config_path = sys.argv[1]
    model_dir = sys.argv[2]
    out_path = sys.argv[3] if len(sys.argv) > 3 else \
        config_path.rsplit("/", 1)[0] + "/quant_model_description.json"

    rules, cfg = load_rules(config_path)
    with open(f"{model_dir}/model.safetensors.index.json") as f:
        weight_keys = sorted(json.load(f)["weight_map"])

    desc: dict[str, object] = {}
    stats: dict[str, int] = {}
    for key in weight_keys:
        # 与产物对齐的两个结构调整:
        #  1. 源 FP8 的 `.scale` 键在量化产物中被折叠为 `.weight_scale`, 不出现
        #  2. engram embed 查表由 adapter 专用通道保存(独立 safetensors + quant_type
        #     声明), 不进 description; engram 其余权重(q/k/wkv)按 FLOAT 正常列出
        if key.endswith(".scale") or ".engram.embed." in key:
            continue
        label = None
        if key.endswith(".weight") and not any(m in key for m in NON_LINEAR_MARKERS):
            label = classify(key[: -len(".weight")], rules)
        if label is None:
            desc[key] = FLOAT
            stats[FLOAT] = stats.get(FLOAT, 0) + 1
        else:
            desc[key] = label
            desc[key[: -len(".weight")] + ".weight_scale"] = label
            stats[label] = stats.get(label, 0) + 1

    # mtp.0 的 embed/head 副本(saver 为 DSpark draft 复制顶层 embed/head, FLOAT)
    for top in ("embed.weight", "head.weight"):
        if top in weight_keys:
            desc[f"mtp.0.{top}"] = FLOAT
            stats[FLOAT] = stats.get(FLOAT, 0) + 1

    # engram embed 的 INT8_SYM 声明(adapter 专用通道, 与参考产物一致)
    adapter = (cfg.get("spec") or {}).get("model_adapter") or {}
    if adapter.get("engram_quant_type") == "int8_sym":
        for key in weight_keys:
            if key.endswith(".engram.embed.weight"):
                layer = key.split(".")[1]
                desc[f"layers.{layer}.engram.embed.quant_type"] = "INT8_SYM"

    desc["group_size"] = 32
    desc["model_quant_type"] = next((t for t in PRIORITY if t in stats), "Unknown")
    desc["metadata"] = {}
    desc["optional"] = {}
    desc["version"] = "1.0.0"

    with open(out_path, "w") as f:
        json.dump(desc, f, indent=2, ensure_ascii=False, sort_keys=True)

    print(f"config: {config_path}")
    print(f"written: {out_path}")
    for label, n in sorted(stats.items()):
        print(f"  {label:14s} modules: {n}")
    print(f"  model_quant_type: {desc['model_quant_type']}, total entries: {len(desc)}")


if __name__ == "__main__":
    main()
