#!/usr/bin/env python3
"""成绩真实性校验: 对每组出分的评测 run, 逐题检查回答完整性.

检查项(每 run 汇总):
  1. 题数是否齐全(30/198)
  2. 每题 finish 是否正常(检测截断: output_tokens >= max_out)
  3. 回答是否非空、是否有可提取的最终答案(aime: \\boxed, gpqa: Answer: X)
  4. 预测分布(全 A / 全同字母 = 异常信号)
输出: 每实验×数据集×run 一行汇总 + 异常题目明细.
用法: python3 verify_scores.py [output_root]
"""
import glob
import json
import os
import re
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else '/mnt/share/w00608002/work/output/glm53flash'
MAX_OUT = 131072   # 当前 ais_bench 口径

AIME_N, GPQA_N = 30, 198


def extract_aime(pred):
    m = re.findall(r'\\boxed\{([^}]*)\}', str(pred))
    return m[-1].strip() if m else None


def extract_gpqa(pred):
    m = re.findall(r'Answer:\s*([A-D])', str(pred)) or re.findall(r'\b([A-D])\b\s*$', str(pred).strip())
    return m[-1] if m else None


def check_run(run_dir, kind):
    fs = sorted(glob.glob(f'{run_dir}/**/predictions/**/{kind}*.jsonl', recursive=True))
    if not fs and kind.startswith('gpqa'):
        fs = sorted(glob.glob(f'{run_dir}/**/predictions/**/GPQA*.jsonl', recursive=True))
    if not fs:
        fs = sorted(glob.glob(f'{run_dir}/**/predictions/**/tmp/tmp_*.jsonl', recursive=True))
    if not fs:
        return None
    rows = [json.loads(l) for l in open(fs[-1])]
    n = AIME_N if kind.startswith('aime') else GPQA_N
    exp = extract_aime if kind.startswith('aime') else extract_gpqa
    done = [r for r in rows if r.get('output_tokens')]
    empty = sum(1 for r in done if not str(r.get('prediction', '')).strip())
    noans = sum(1 for r in done if exp(r.get('prediction', '')) is None)
    trunc = sum(1 for r in done if r.get('output_tokens', 0) >= MAX_OUT)
    ok = sum(1 for r in done if exp(r.get('prediction', '')) is not None
             and str(exp(r.get('prediction', ''))).strip().upper().replace(',', '')
             == str(r.get('gold', '')).strip().upper().replace(',', ''))
    # 预测分布
    import collections
    dist = collections.Counter(str(exp(r.get('prediction', '')) or '?') for r in done)
    bad_ids = [r['id'] for r in done
              if exp(r.get('prediction', '')) is None
              or r.get('output_tokens', 0) >= MAX_OUT
              or not str(r.get('prediction', '')).strip()]
    return {
        'n': len(rows), 'done': len(done), 'expect': n,
        'empty': empty, 'no_answer': noans, 'trunc': trunc,
        'correct': ok, 'acc': ok / len(done) * 100 if done else 0,
        'dist': dict(dist.most_common(5)), 'bad_ids': bad_ids,
    }


def main():
    dirs = sorted(glob.glob(f'{ROOT}/night2_exp*') + glob.glob(f'{ROOT}/night2_exp*')[:0])
    print(f"{'实验':<38s} {'run':<5s} {'题数':>5s} {'有效':>5s} {'对':>4s} {'实算%':>7s} {'无答':>4s} {'截断':>4s} {'预测分布':<24s} 异常题目")
    for d in dirs:
        name = os.path.basename(d)
        for ds in ('aime2024_gen_0_shot_chat_prompt', 'gpqa_gen_0_shot_cot_chat_prompt'):
            for r in (1, 2, 3, 4, 5):
                rd = f'{d}/{ds}/run{r}'
                if not os.path.isdir(rd):
                    continue
                kind = 'aime2024' if ds.startswith('aime') else 'gpqa'
                c = check_run(rd, kind)
                if c is None:
                    continue
                flag = ''
                if c['done'] < c['expect']:
                    flag = f"⚠ 题数不足({c['done']}/{c['expect']})"
                elif c['no_answer'] > c['expect'] * 0.1:
                    flag = f"⚠ 无答案比例高({c['no_answer']}/{c['done']})"
                print(f"{name:<38s} {ds[:5]}{r:<2d} {c['done']:>5d} {c['done']:>5d} {c['correct']:>4d} "
                      f"{c['acc']:>7.2f} {c['no_answer']:>4d} {c['trunc']:>4d} {str(c['dist']):<24s} {flag}")
                if c['bad_ids'] and len(c['bad_ids']) <= 12:
                    print(f"{'':<38s}      异常题id: {c['bad_ids']}")


if __name__ == '__main__':
    main()
