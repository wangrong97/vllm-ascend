# 采集 GPQA 198 题的 MoE routed experts 分布
# 用法: python3 collect_routed_experts.py <tag>
# 输出: ais_bench_logs/routed_experts/<tag>.npz (counts_per_req [198,43,256], meta)
import base64
import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import urllib.request

TAG = sys.argv[1] if len(sys.argv) > 1 else "unnamed"
OUT = f"/mnt/share/rr08002/work/vllm-ascend/ais_bench_logs/routed_experts/{TAG}.npz"
MAX_TOKENS = 2048
CONCURRENCY = 16

ALIGN = """Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{question}

A) {A}
B) {B}
C) {C}
D) {D}"""

df = pd.read_csv('/mnt/share/rr08002/benchmark/ais_bench/datasets/gpqa/gpqa_diamond.csv')
# 与 ais_bench GPQADataset.load 一致: 4 种轮转选项顺序
shuffle_patterns = ['ABCD', 'BCDA', 'CDAB', 'DABC']
prompts = []
for cnt, (_, r) in enumerate(df.iterrows()):
    question = r.iloc[7]
    options = {'A': r['Correct Answer'], 'B': r['Incorrect Answer 1'],
               'C': r['Incorrect Answer 2'], 'D': r['Incorrect Answer 3']}
    c = shuffle_patterns[cnt % 4]
    opts = {('ABCD'[i]): options[c[i]] for i in range(4)}
    prompts.append(ALIGN.format(question=question, **opts))
print(f"prompts: {len(prompts)}", flush=True)


def run_one(idx):
    req = urllib.request.Request(
        "http://localhost:9001/v1/chat/completions",
        data=json.dumps({
            "model": "dsv",
            "messages": [{"role": "user", "content": prompts[idx]}],
            "max_tokens": MAX_TOKENS,
            "temperature": 1.0, "top_k": 20, "top_p": 1.0,
        }).encode(),
        headers={"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=900))
            c = r["choices"][0]
            re_b64 = c.get("routed_experts")
            if re_b64 is None:
                raise RuntimeError("no routed_experts in response")
            arr = np.load(io.BytesIO(base64.b64decode(re_b64)))
            # arr: [num_tokens, num_layers, topk]
            cnt = np.zeros((arr.shape[1], 256), dtype=np.int64)
            for t in range(arr.shape[0]):
                for l in range(arr.shape[1]):
                    for e in arr[t, l]:
                        if 0 <= e < 256:
                            cnt[l, e] += 1
            return idx, cnt, arr.shape[0]
        except Exception as e:
            if attempt == 2:
                print(f"[{idx}] FAILED: {e}", flush=True)
                return idx, None, 0
            time.sleep(5)


t0 = time.time()
results = {}
ntoks = 0
with ThreadPoolExecutor(CONCURRENCY) as ex:
    for idx, cnt, ntok in ex.map(run_one, range(len(prompts))):
        if cnt is not None:
            results[idx] = cnt
            ntoks += ntok
        if len(results) % 20 == 0:
            print(f"done {len(results)}/{len(prompts)}, {time.time()-t0:.0f}s", flush=True)

counts = np.zeros((43, 256), dtype=np.int64)
for cnt in results.values():
    counts += cnt
np.savez(OUT, counts=counts, n_requests=len(results), n_tokens=ntoks)
print(f"SAVED {OUT}: {len(results)} requests, {ntoks} tokens, {time.time()-t0:.0f}s", flush=True)
