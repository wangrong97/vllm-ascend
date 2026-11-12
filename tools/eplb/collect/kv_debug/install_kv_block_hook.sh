#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TARGET="$REPO/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py"
HOOK="$REPO/tools/eplb/kv_block_hook.py"
B="# >>> kv-block-dump-hook (auto; see tools/eplb) >>>"
E="# <<< kv-block-dump-hook <<<"
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
anchor_d = "        ret = self.engine.batch_transfer_sync_read(session_id, src_list, dst_list, length_list)"
assert src.count(anchor_d) == 1, "D 侧锚点不唯一"
d_block = f'''        {begin}
        try:
            if __import__("os").environ.get("KV_DUMP_DIR") and not getattr(self, "_kv_dumped", False):
                self._kv_dumped = True
                import importlib.util as _ilu
                _s = _ilu.spec_from_file_location("kv_block_hook", r"{hook}")
                _m = _ilu.module_from_spec(_s); _s.loader.exec_module(_m)
                _m.dump_side("d", self.kv_caches, plan=getattr(self, "_kv_dump_plan", []))
        except Exception: pass
        {end}
'''
src = src.replace(anchor_d, d_block + anchor_d)
anchor_p = "    def get_finished(self) -> tuple[set[str], set[str]]:"
assert src.count(anchor_p) == 1, "P 侧锚点不唯一"
p_block = anchor_p + f'''
        {begin}
        try:
            if __import__("os").environ.get("KV_DUMP_DIR") and not getattr(self, "_kv_dumped", False):
                self._kv_dumped = True
                import importlib.util as _ilu
                _s = _ilu.spec_from_file_location("kv_block_hook", r"{hook}")
                _m = _ilu.module_from_spec(_s); _s.loader.exec_module(_m)
                _m.dump_side("p", getattr(self, "kv_caches", None))
        except Exception: pass
        {end}'''
src = src.replace(anchor_p, p_block)
open(target, "w", encoding="utf-8").write(src)
print("已安装 D+P 两处插桩(KV_DUMP_DIR 控制开关)")
PYEOF
