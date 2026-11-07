#!/usr/bin/env python3
"""根据量化配置 yaml 离线生成 AscendV1 格式的 quant_model_description.json。

用法:
  python3 gen_quant_description.py [config.yaml] [model_dir] [output_json]

默认:
  config:  /mnt/share/w00608002/weight/GLM-5.3-Flash-w4a4/GLM-5.3-Flash_best_practice.yaml
  model:   /mnt/share/w00608002/weight/GLM-5.3-Flash-BF16-fixcfg (读取权重键清单)
  output:  <config 同目录>/quant_model_description.json

规则(与 msmodelslim AscendV1Saver 输出对齐):
  - 仅作用于 model.language_model.layers.*(语言模型 decoder 层, 含 MTP 层);
    视觉塔及顶层权重(embed/norm/lm_head)始终 FLOAT —— 与量化流水线只逐层处理
    decoder 的行为一致。
  - 按 yaml spec.process 中 linear_quant 块顺序匹配(include/exclude,
    fnmatch + {} 展开); 命中的 Linear 生成 <prefix>.weight / .weight_scale 两条目;
    其余权重键标 FLOAT。
  - (act_dtype, weight_dtype) → 量化标签: mxfp8→W8A8_MXFP8, mxfp4→W4A4_MXFP4,
    int8→W8A8_DYNAMIC(per_token 动态), fp8_e4m3→W8A8_FP8。
"""
import fnmatch
import json
import sys

import yaml

DEFAULT_CONFIG = "/mnt/share/w00608002/weight/GLM-5.3-Flash-w4a4/GLM-5.3-Flash_best_practice.yaml"
DEFAULT_MODEL = "/mnt/share/w00608002/weight/GLM-5.3-Flash-BF16-fixcfg"
FLOAT = "FLOAT"

DTYPE_LABEL = {
    ("mxfp8", "mxfp8"): "W8A8_MXFP8",
    ("mxfp4", "mxfp4"): "W4A4_MXFP4",
    ("int8", "int8"): "W8A8_DYNAMIC",
    ("fp8_e4m3", "fp8_e4m3"): "W8A8_FP8",
}
# AscendV1Saver.QUANT_TYPE_PRIORITY: 比特数越低优先级越高, 作为 model_quant_type
PRIORITY = ["W4A4_MXFP4", "W8A8_MXFP8", "W8A8_FP8", "W8A8_DYNAMIC"]


def brace_expand(pattern: str) -> list[str]:
    """展开 'a.{1,2}.b' -> ['a.1.b', 'a.2.b']"""
    if "{" not in pattern:
        return [pattern]
    head, rest = pattern.split("{", 1)
    body, tail = rest.split("}", 1)
    return [h + t for item in body.split(",") for h in [head + item] for t in brace_expand(tail)]


def match_any(name: str, patterns) -> bool:
    if not patterns:
        return False
    return any(fnmatch.fnmatchcase(name, p) for pat in patterns for p in brace_expand(str(pat)))


def load_rules(config_path: str) -> list[tuple[str, list, list]]:
    """从 yaml 提取 linear_quant 规则: [(label, include, exclude), ...]"""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    rules = []
    for proc in cfg["spec"]["process"]:
        if proc.get("type") != "linear_quant":
            continue
        qc = proc.get("qconfig") or {}
        key = (str(qc.get("act", {}).get("dtype")), str(qc.get("weight", {}).get("dtype")))
        if key not in DTYPE_LABEL:
            raise ValueError(f"未支持的 dtype 组合: {key}, 请在 DTYPE_LABEL 中补充")
        rules.append((DTYPE_LABEL[key], proc.get("include") or [], proc.get("exclude") or []))
    if not rules:
        raise ValueError("yaml 中没有 linear_quant 块")
    return rules


def to_runtime_name(key: str) -> str:
    """checkpoint(HF) 键名 → msmodelslim 运行时模块名。

    GLM-5.3-Flash 的 KDA 层: checkpoint 里 forget gate 是扁平的
    `self_attn.f_a_proj`, 运行时 msmodelslim 模型里是
    `self_attn.forget_gate.f_a_proj`(与量化产物 description 中的命名一致)。
    """
    for sub in ('f_a_proj', 'f_b_proj', 'A_log', 'dt_bias'):
        if f'.self_attn.{sub}' in key:
            return key.replace(f'.self_attn.{sub}', f'.self_attn.forget_gate.{sub}')
    return key


def classify(module_prefix: str, rules) -> str | None:
    for label, includes, excludes in rules:
        if match_any(module_prefix, includes) and not match_any(module_prefix, excludes):
            return label
    return None


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG
    model_dir = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL
    out_path = sys.argv[3] if len(sys.argv) > 3 else \
        config_path.rsplit("/", 1)[0] + "/quant_model_description.json"

    rules = load_rules(config_path)
    with open(f"{model_dir}/model.safetensors.index.json") as f:
        weight_keys = sorted(json.load(f)["weight_map"])

    desc: dict[str, object] = {}
    stats: dict[str, int] = {}
    for key in weight_keys:
        # 规则匹配与输出键名都用运行时命名(forget_gate 修正)
        key = to_runtime_name(key)
        label = None
        if key.startswith("model.language_model.") and ".layers." in key and key.endswith(".weight"):
            label = classify(key[: -len(".weight")], rules)
        if label is None:
            desc[key] = FLOAT
            stats[FLOAT] = stats.get(FLOAT, 0) + 1
        else:
            desc[key] = label
            desc[key[: -len(".weight")] + ".weight_scale"] = label
            stats[label] = stats.get(label, 0) + 1

    # AscendV1Saver 末尾写入的全局字段
    model_quant_type = next((t for t in PRIORITY if t in stats), "Unknown")
    desc["model_quant_type"] = model_quant_type
    desc["metadata"] = {
        "quarot": {
            "max_tp_size": 4,
            "heads_rotation": {"layers": []},
            "kronecker_rotation": {"layers": []},
        }
    }

    with open(out_path, "w") as f:
        json.dump(desc, f, indent=2, ensure_ascii=False, sort_keys=True)

    print(f"config: {config_path}")
    print(f"written: {out_path}")
    for label, n in sorted(stats.items()):
        print(f"  {label:14s} modules/keys: {n}")
    print(f"  model_quant_type: {model_quant_type}, total entries: {len(desc)}")


if __name__ == "__main__":
    main()
