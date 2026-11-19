#!/bin/bash
# 一键激活活值抓取测试: 向运行中的服务(9000)发 4 道固定题, 产出 dump 目录
# 前提: 服务已用 VLLM_ASCEND_INDEXER_DUMP=1 起动(hook 已装); 否则只发请求不出 dump
set -e
PORT=${PORT:-9000}
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${VLLM_ASCEND_INDEXER_DUMP_DIR:-/mnt/share/w00608002/work/output/indexer_dump_3h_c8}
MAXTOK=${MAXTOK:-8192}
echo "发送 4 道测试题(aim易/难 + gpqa易/难) 到 :${PORT}, dump 目录: ${OUT}"
python3 - "$HERE/testset/dump_testset.jsonl" "$PORT" "$MAXTOK" <<'PY'
import json, sys, urllib.request
path, port, maxtok = sys.argv[1], sys.argv[2], int(sys.argv[3])
for l in open(path):
    r = json.loads(l)
    body = json.dumps({"model": "glm",
                       "messages": [{"role": "user", "content": r["prompt"]}],
                       "max_tokens": maxtok, "temperature": 1.0, "top_p": 0.95}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body,
                                {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as resp:
        d = json.load(resp)
    c = d["choices"][0]["message"].get("content") or ""
    print(f"[{r['tag']}] id={r['id']} finish={d['choices'][0]['finish_reason']} "
          f"tokens={d.get('usage',{}).get('completion_tokens')} content尾60: {c[-60:]!r}")
PY
echo "--- dump 产物 ---"; ls -la "$OUT" 2>/dev/null | head -8
python3 - "$OUT" <<'PY'
import json, sys, os
p = os.path.join(sys.argv[1], "summary.json")
if not os.path.exists(p):
    print("无 summary.json(hook 未安装?)"); sys.exit(0)
recs = json.load(open(p))["records"]
import collections
agg = collections.defaultdict(list)
for r in recs:
    agg[(r["stage"], r["layer"])].append(r)
print(f"{'stage/layer':<24s} {'n':>3s} {'absmax':>8s} {'p999':>8s} {'absmean':>8s}")
for (st, ly), rs in sorted(agg.items()):
    print(f"{st}/L{ly:<3d}{'':{max(0,14-len(st)-len(str(ly)))}}{len(rs):>3d} "
          f"{max(x['absmax'] for x in rs):8.2f} {max(x['p999'] for x in rs):8.2f} "
          f"{sum(x['absmean'] for x in rs)/len(rs):8.3f}")
PY
