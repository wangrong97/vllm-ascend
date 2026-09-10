# 单算子性能验证: npu_quant_lightning_indexer @ 128k 上下文真实 shape
# 对照 trace 中的调用: layout TND/PA_BSND, sparse_count=2048, sparse_mode=3
# fp8 query/key + e8m0 scale(两侧 ckpt 该路径输入 dtype/shape 完全一致)
import time

import torch
import torch_npu

DEVICE = "npu:0"
torch.npu.set_device(DEVICE)

N1, D, BLOCK = 64, 128, 128          # index_n_heads, index_head_dim, block_size
SPARSE_COUNT = 2048
CTX = 131072


def make_inputs(num_tokens, ctx_len, b=1):
    n_blocks = (ctx_len + BLOCK - 1) // BLOCK
    q = torch.randn(num_tokens, N1, D, device=DEVICE).to(torch.float8_e4m3fn)
    k_cache = (torch.randn(n_blocks, BLOCK, 1, D, device=DEVICE) * 0.1).to(torch.float8_e4m3fn)
    w = torch.randn(num_tokens, N1, device=DEVICE, dtype=torch.bfloat16).abs() + 0.5
    qs = torch.rand(num_tokens, N1, device=DEVICE, dtype=torch.float32) * 0.01 + 0.005
    ks = torch.rand(n_blocks, BLOCK, 1, device=DEVICE, dtype=torch.float32) * 0.01 + 0.005
    bt = torch.arange(n_blocks, device=DEVICE, dtype=torch.int32).view(b, -1)
    asq = torch.full((b,), num_tokens, device=DEVICE, dtype=torch.int32)
    ask = torch.full((b,), ctx_len, device=DEVICE, dtype=torch.int32)
    return q, k_cache, w, qs, ks, bt, asq, ask


def bench(num_tokens, ctx_len, iters=50):
    args = make_inputs(num_tokens, ctx_len)
    q, k_cache, w, qs, ks, bt, asq, ask = args
    for _ in range(5):  # warmup
        torch_npu.npu_quant_lightning_indexer(
            query=q, key=k_cache, weights=w,
            query_dequant_scale=qs, key_dequant_scale=ks,
            actual_seq_lengths_query=asq, actual_seq_lengths_key=ask,
            block_table=bt, query_quant_mode=0, key_quant_mode=0,
            layout_query="TND", layout_key="PA_BSND",
            sparse_count=SPARSE_COUNT, sparse_mode=3,
        )
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        torch_npu.npu_quant_lightning_indexer(
            query=q, key=k_cache, weights=w,
            query_dequant_scale=qs, key_dequant_scale=ks,
            actual_seq_lengths_query=asq, actual_seq_lengths_key=ask,
            block_table=bt, query_quant_mode=0, key_quant_mode=0,
            layout_query="TND", layout_key="PA_BSND",
            sparse_count=SPARSE_COUNT, sparse_mode=3,
        )
    end.record()
    torch.npu.synchronize()
    ms = start.elapsed_time(end) / iters
    print(f"  T={num_tokens:5d} ctx={ctx_len:7d}: {ms:.3f} ms/call")
    return ms


print("=== npu_quant_lightning_indexer 单算子基准 (128k 上下文) ===")
bench(1, CTX + 65)      # decode 单 token(128k KV)
bench(2, CTX + 65)      # decode MTP 2 token
bench(64, CTX + 65)     # decode 批量
bench(4096, 4096)       # prefill 第一个 chunk
bench(4096, CTX)        # prefill 最后一个 chunk(全量 128k KV)

# 探查 _C_ascend 稀疏注意力算子是否可独立调用
try:
    import vllm_ascend._cann_ops_custom  # noqa
    for opname in ['npu_kv_quant_sparse_attn_sharedkv_metadata', 'npu_kv_quant_sparse_attn_sharedkv']:
        ok = hasattr(torch.ops._C_ascend, opname)
        print(f"torch.ops._C_ascend.{opname}: {'OK' if ok else 'NOT FOUND'}")
except Exception as e:
    print("_cann_ops_custom load failed:", str(e)[:120])
