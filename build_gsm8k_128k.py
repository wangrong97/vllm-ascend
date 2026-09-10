# 生成 128k 输入的 ais_bench 自定义数据集(gsm8k 打包)
# 每条约 131,072 token: gsm8k Q&A 示例打包填充 + 末尾挂一条真实问题,输出上限 512
# 用法: python3 build_gsm8k_128k.py [输出路径] [行数] [目标token数]
import json
import sys

from transformers import PreTrainedTokenizerFast

OUT_PATH = sys.argv[1] if len(sys.argv) > 1 else "/mnt/share/rr08002/work/vllm-ascend/ais_bench_logs/gsm8k_128k.jsonl"
N_ROWS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
TARGET_INPUT = int(sys.argv[3]) if len(sys.argv) > 3 else 131072
MAX_OUT = 512

# tokenizer 用任一 DeepSeek-V4 权重目录(词表相同)
tok = PreTrainedTokenizerFast.from_pretrained("/mnt/share/rr08002/weights/DeepSeek-V4-Flash-w4a4")

pool = []
with open("/mnt/share/rr08002/benchmark/ais_bench/datasets/gsm8k/test.jsonl") as f:
    for line in f:
        pool.append(json.loads(line))
print(f"gsm8k rows: {len(pool)}")


def pack_to(target_tokens, seed_offset):
    """把 gsm8k Q&A 文本拼到 target_tokens(按 token 数截断)"""
    text = "以下是一些数学问题及其解答示例:\n\n"
    i = seed_offset
    while True:
        q = pool[i % len(pool)]
        text += f"问题: {q['question']}\n解答: {q['answer']}\n\n"
        i += 1
        if i % 100 == 0:
            ids = tok(text)["input_ids"]
            if len(ids) >= target_tokens:
                return tok.decode(ids[:target_tokens])


out_rows = []
for r in range(N_ROWS):
    q = pool[(r * 17 + 3) % len(pool)]  # 每行末尾挂不同的真实问题
    tail = f"\n\n现在请回答下面这个问题:\n问题: {q['question']}\n请逐步推理并给出最终数字答案。\n"
    tail_len = len(tok(tail)["input_ids"])
    body = pack_to(TARGET_INPUT - tail_len, r * 300)
    full = body + tail
    print(f"row {r}: prompt tokens = {len(tok(full)['input_ids'])}")
    out_rows.append({"question": full, "answer": "", "max_tokens": MAX_OUT})

with open(OUT_PATH, "w") as f:
    for row in out_rows:
        f.write(json.dumps(row) + "\n")
print("saved:", OUT_PATH)
