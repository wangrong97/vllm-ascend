#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NPU 卡级负载时序采样器(解析 npu-smi info 文本,零额外依赖)。
输出 CSV: ts, npu_id, util_pct, hbm_mb, hbm_total_mb, power_w, n_procs"""
import argparse, csv, re, signal, subprocess, sys, time

HEAD_PAT = re.compile(r"^\|\s*(\d+)\s+\|\s*\S+\s+\|\s*OK\s+\|\s*([\d.]+)")
CONT_PAT = re.compile(r"^\|\s*\S*\s*\|\s*\S*\s*\|\s*NA\s+\|\s*(\d+)\s+\d+\s*/\s*\d+\s+(\d+)\s*/\s*(\d+)")

def sample_once():
    out = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=30).stdout
    rows, cur, in_proc, np_ = [], None, False, 0
    for line in out.splitlines():
        if "Process id" in line: in_proc = True; continue
        if in_proc:
            if re.match(r"^\|\s*\d+\s+\|\s*(\d+)", line): np_ += 1
            continue
        m = HEAD_PAT.match(line)
        if m:
            if cur: rows.append(cur)
            cur = {"npu_id": int(m.group(1)), "power_w": float(m.group(2)), "util_pct": None, "hbm_mb": None, "hbm_total_mb": None}
            continue
        if cur is not None:
            m = CONT_PAT.match(line)
            if m: cur["util_pct"] = int(m.group(1)); cur["hbm_mb"] = int(m.group(2)); cur["hbm_total_mb"] = int(m.group(3))
    if cur: rows.append(cur)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    for r in rows: r["ts"] = ts; r["n_procs"] = np_
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True); ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--duration", type=float, default=0)
    args = ap.parse_args()
    running = {"stop": False}
    def _stop(s, f): running["stop"] = True
    signal.signal(signal.SIGINT, _stop); signal.signal(signal.SIGTERM, _stop)
    n = 0
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ts", "npu_id", "util_pct", "hbm_mb", "hbm_total_mb", "power_w", "n_procs"])
        w.writeheader(); t0 = time.time()
        while not running["stop"]:
            try:
                for row in sample_once(): w.writerow(row)
                f.flush(); n += 1
            except Exception as e: print(f"[sampler] 失败: {e}", file=sys.stderr)
            if args.duration > 0 and time.time() - t0 >= args.duration: break
            time.sleep(max(0.2, args.interval - (time.time() - t0) % args.interval))
    print(f"[sampler] {n} 轮 -> {args.output}", file=sys.stderr)

if __name__ == "__main__": main()
