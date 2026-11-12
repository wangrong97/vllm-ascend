#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GSM8K 并发压测客户端(流式,记 TTFT/时延)。输出 CSV: req_id,ttft_s,total_s,prompt_tokens,est_completion_tokens,status"""
import argparse, csv, json, re, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
DATA_RE = re.compile(rb"^data:\s*(.+)$", re.M)
def load_prompts(path, limit):
    ps = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line: continue
        q = json.loads(line).get("question", "")
        if q: ps.append(q)
        if len(ps) >= limit: break
    return ps
def one_request(i, ep, model, prompt, max_tokens):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "stream": True, "temperature": 0.0}).encode()
    req = urllib.request.Request(ep.rstrip("/") + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    rec = {"req_id": i, "ttft_s": None, "total_s": None, "prompt_tokens": None, "est_completion_tokens": 0, "status": "ok"}
    t0 = time.time(); first_at, chars = None, 0
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            for raw in resp:
                m = DATA_RE.match(raw.strip(b"\r\n"))
                if not m: continue
                pl = m.group(1)
                if pl == b"[DONE]": break
                try: ch = json.loads(pl)
                except Exception: continue
                delta = (ch.get("choices") or [{}])[0].get("delta", {})
                c = delta.get("content") or delta.get("reasoning_content") or ""
                if c:
                    if first_at is None: first_at = time.time()
                    chars += len(c)
                u = ch.get("usage") or {}
                if u.get("prompt_tokens"): rec["prompt_tokens"] = u["prompt_tokens"]
        rec["ttft_s"] = round(first_at - t0, 4) if first_at else None
        rec["total_s"] = round(time.time() - t0, 4)
        rec["est_completion_tokens"] = max(1, int(chars / 1.6)) if chars else 0
    except Exception as e:
        rec["status"] = f"error: {str(e)[:80]}"; rec["total_s"] = round(time.time() - t0, 4)
    return rec
def pct(v, p):
    if not v: return float("nan")
    s = sorted(v); k = min(len(s) - 1, max(0, int(round(p / 100 * (len(s) - 1))))); return s[k]
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True); ap.add_argument("--model", default="dsv")
    ap.add_argument("--dataset", default="/mnt/share/w00608002/work/benchmark/ais_bench/datasets/gsm8k/test.jsonl")
    ap.add_argument("--num-prompts", type=int, default=200); ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=512); ap.add_argument("--output", required=True)
    a = ap.parse_args()
    ps = load_prompts(a.dataset, a.num_prompts)
    print(f"加载 {len(ps)} 题 | 并发={a.concurrency}")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        recs = [f.result() for f in [pool.submit(one_request, i, a.endpoint, a.model, p, a.max_tokens) for i, p in enumerate(ps)]]
    wall = time.time() - t0
    with open(a.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys())); w.writeheader(); w.writerows(recs)
    ok = [r for r in recs if r["status"] == "ok"]
    tt = [r["ttft_s"] for r in ok if r["ttft_s"]]; tot = [r["total_s"] for r in ok]
    tok = sum(r["est_completion_tokens"] for r in ok)
    print(f"完成 {len(ok)}/{len(recs)} | {wall:.1f}s")
    if tt: print(f"TTFT P50={pct(tt,50):.3f}s P90={pct(tt,90):.3f}s P99={pct(tt,99):.3f}s")
    if tot: print(f"总时延 P50={pct(tot,50):.3f}s P99={pct(tot,99):.3f}s")
    if wall and tok: print(f"吞吐 ≈ {tok/wall:.1f} tokens/s(估)")
if __name__ == "__main__": main()
