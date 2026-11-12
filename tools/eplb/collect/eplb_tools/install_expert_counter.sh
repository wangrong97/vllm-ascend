#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TARGET="$REPO/vllm_ascend/ops/fused_moe/routed_experts.py"
COUNTER="$REPO/tools/eplb/expert_counter.py"
B="# >>> eplb-expert-counter (auto-patched; see tools/eplb) >>>"
E="# <<< eplb-expert-counter <<<"
[[ -f "$TARGET" ]] || { echo "缺少 $TARGET"; exit 1; }
python3 - "$TARGET" "$COUNTER" "$B" "$E" "${1:-}" <<'PYEOF'
import sys
target, counter, begin, end, action = sys.argv[1:6]
src = open(target, encoding="utf-8").read()
if begin in src:
    if action == "--uninstall":
        out, inside = [], False
        for ln in src.splitlines(keepends=True):
            if begin in ln: inside = True; continue
            if end in ln: inside = False; continue
            if not inside: out.append(ln)
        open(target, "w", encoding="utf-8").write("".join(out)); print("已卸载插桩")
    else: print("插桩已存在,跳过")
    sys.exit(0)
if action == "--uninstall": print("未发现插桩"); sys.exit(0)
lines = src.splitlines(keepends=True)
anchor = None; in_m = False
for i, ln in enumerate(lines):
    if ln.lstrip().startswith("def _select_experts"): in_m = True; continue
    if in_m:
        if ln.startswith("    def ") or (ln.strip() and not ln.startswith(" ")): break
        if ln == "        return topk_weights, topk_ids\n": anchor = i; break
if anchor is None: sys.exit("未定位锚点,请人工核对")
block = f'''        {begin}
        try:
            _ec = globals().get("_EPLB_COUNTER_MOD")
            if _ec is None and __import__("os").environ.get("EPLB_EXPORT_DIR"):
                import importlib.util as _ilu
                _s = _ilu.spec_from_file_location("eplb_expert_counter", r"{counter}")
                _ec = _ilu.module_from_spec(_s); _s.loader.exec_module(_ec)
                globals()["_EPLB_COUNTER_MOD"] = _ec
            if _ec is not None: _ec.record(self, topk_ids)
        except Exception: pass
        {end}
'''
lines.insert(anchor, block)
open(target, "w", encoding="utf-8").write("".join(lines))
print(f"已安装插桩(第 {anchor+1} 行前,重启服务生效)")
PYEOF
