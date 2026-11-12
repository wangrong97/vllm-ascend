#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TARGET="$REPO/vllm_ascend/worker/worker.py"
HOOK="$REPO/tools/eplb/kv_block_hook.py"
B="# >>> kv-layerdump-hook (auto; see tools/eplb) >>>"
E="# <<< kv-layerdump-hook <<<"
[[ -f "$TARGET" ]] || { echo "缺少 $TARGET"; exit 1; }
python3 - "$TARGET" "$HOOK" "$B" "$E" "${1:-}" <<'PYEOF'
import sys
target, hook, begin, end, action = sys.argv[1:6]
src = open(target, encoding="utf-8").read()
if begin in src:
    if action == "--uninstall":
        out, inside = [], False
        for ln in src.splitlines(keepends=True):
            if begin in ln: inside = True; continue
            if end in ln: inside = False; continue
            if not inside: out.append(ln)
        open(target, "w", encoding="utf-8").write("".join(out)); print("已卸载")
    else: print("已存在,跳过")
    sys.exit(0)
if action == "--uninstall": print("无插桩"); sys.exit(0)
anchor = "            self.profiler.stop()"
assert src.count(anchor) == 1, "stop 锚点不唯一"
block = f'''            {begin}
            try:
                if __import__("os").environ.get("KV_DUMP_DIR") and not getattr(self, "_kv_layerdumped", False):
                    self._kv_layerdumped = True
                    import importlib.util as _ilu, os as _os
                    _s = _ilu.spec_from_file_location("kv_block_hook", r"{hook}")
                    _m = _ilu.module_from_spec(_s); _s.loader.exec_module(_m)
                    _mr = getattr(self, "model_runner", None)
                    _kc = getattr(_mr, "kv_caches", None) if _mr is not None else None
                    if _kc:
                        _m.dump_side("single", {{f"layer_{{i}}": (t[0] if isinstance(t, (list, tuple)) and t else t) for i, t in enumerate(_kc) if t is not None}},
                                     role_tag=_os.environ.get("KV_DUMP_TAG", "run"))
                    else:
                        print("[kv-layerdump-hook] model_runner.kv_caches 不可达", flush=True)
            except Exception as _e:
                print(f"[kv-layerdump-hook] dump failed: {{_e}}", flush=True)
            {end}
'''
src = src.replace(anchor, block + anchor)
open(target, "w", encoding="utf-8").write(src)
print("已安装 stop_profile 触发的单机 KV dump(KV_DUMP_DIR/KV_DUMP_TAG 控制)")
PYEOF
