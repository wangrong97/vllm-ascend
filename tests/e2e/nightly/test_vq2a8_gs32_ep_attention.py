# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small-head FP8 DSA correctness and ACLGraph replay without model loading.

The existing --vq2a8-model option opts in to NPU execution; weights and the
standalone GS32 kernel artifacts are not loaded by this test.
"""

from types import SimpleNamespace

import pytest

HEAD_DIM = 512
PACKED_KV_BYTES = 640
BLOCK_SIZE = 128
KV_TOKENS = 256
NATIVE_HEADS = 64
SPARSE_TOPK = 512
REPLAY_SEQUENCES = (KV_TOKENS, KV_TOKENS - 1, KV_TOKENS)
LONG_KV_TOKENS = 17408
LONG_REPLAY_SEQUENCES = (127, 128, 129, 16384, 16385, 17407, 127)


@pytest.fixture(scope="module")
def attention_runtime(request):
    if not request.config.getoption("--vq2a8-model", default=None):
        pytest.skip("Pass --vq2a8-model to opt in to the real-NPU attention regression (no model is loaded).")

    import torch
    import torch_npu  # noqa: F401 - registers the NPU device

    from vllm_ascend.attention.dsa_kv_quant import kv_quant_sparse_attn, kv_quant_sparse_attn_metadata
    from vllm_ascend.models.deepseek_v4.native_ops import require_deepseek_v4_native_ops

    assert torch.npu.is_available(), "Explicit EP attention tests require a real Ascend NPU."
    device = torch.device("npu", 0)
    torch.npu.set_device(device)
    require_deepseek_v4_native_ops()
    return SimpleNamespace(
        torch=torch,
        device=device,
        native=torch.ops._C_ascend,
        attention=kv_quant_sparse_attn,
        metadata=kv_quant_sparse_attn_metadata,
    )


def packed_cache(runtime, tokens, generator):
    torch = runtime.torch
    blocks = (tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
    cache = torch.zeros((blocks, BLOCK_SIZE, 1, PACKED_KV_BYTES), dtype=torch.uint8).view(torch.float8_e4m3fn)
    cache = cache.to(runtime.device)
    values = (torch.randn((tokens, HEAD_DIM), generator=generator) * 0.125).bfloat16().to(runtime.device)
    slots = torch.arange(tokens, dtype=torch.int64, device=runtime.device)
    runtime.native.kv_compress_epilog(
        kv_compress_cache=cache.view(-1, 1, PACKED_KV_BYTES),
        x=values,
        slot_mapping=slots,
        quant_group_size=64,
        quant_mode=2,
        round_scale_flag=True,
        layout=1,
    )
    table = torch.arange(blocks, dtype=torch.int32, device=runtime.device).unsqueeze(0)
    return cache, table


@pytest.mark.parametrize("heads", [8, 16, 32], ids=["tp8", "tp4", "tp2"])
@pytest.mark.parametrize("tokens", [1, 32], ids=["decode", "prefill"])
@pytest.mark.parametrize("ratio", [1, 4, 128], ids=["swa", "c4", "c128"])
def test_gs32_ep_attention_heads_match_native_and_graph_replay(attention_runtime, heads, tokens, ratio):
    check_attention_replay(attention_runtime, heads, tokens, ratio, KV_TOKENS, REPLAY_SEQUENCES)


@pytest.mark.parametrize("ratio", [1, 4, 128], ids=["swa", "c4", "c128"])
def test_gs32_ep_attention_long_decode_matches_native_and_graph_replay(attention_runtime, ratio):
    # TP2 decode retains one query row while KV history crosses cache and
    # compression boundaries, then shrinks for a subsequent request.
    check_attention_replay(attention_runtime, 32, 1, ratio, LONG_KV_TOKENS, LONG_REPLAY_SEQUENCES)


@pytest.mark.parametrize("operator", ["sas", "qli"])
def test_gs32_attention_metadata_is_independent_of_allocator_contents(attention_runtime, operator):
    """Compare every metadata word, including idle-core slots and reserved tail."""
    runtime = attention_runtime
    torch, device = runtime.torch, runtime.device
    query_start = torch.tensor([0, 1], dtype=torch.int32, device=device)
    lengths = torch.tensor([256], dtype=torch.int32, device=device)

    def generate():
        if operator == "sas":
            return runtime.native.npu_kv_quant_sparse_attn_sharedkv_metadata(
                num_heads_q=64,
                num_heads_kv=1,
                head_dim=512,
                kv_quant_mode=1,
                cu_seqlens_q=query_start,
                seqused_q=query_start[1:],
                seqused_kv=lengths,
                batch_size=1,
                max_seqlen_q=1,
                max_seqlen_kv=256,
                tile_size=64,
                rope_head_dim=64,
                cmp_ratio=1,
                ori_mask_mode=4,
                cmp_mask_mode=3,
                ori_win_left=127,
                ori_win_right=0,
                layout_q="TND",
                layout_kv="PA_ND",
                has_ori_kv=True,
                has_cmp_kv=False,
            )
        return runtime.native.npu_vllm_quant_lightning_indexer_metadata(
            num_heads_q=64,
            num_heads_k=1,
            head_dim=128,
            query_quant_mode=0,
            key_quant_mode=0,
            actual_seq_lengths_query=query_start[1:],
            actual_seq_lengths_key=lengths,
            batch_size=1,
            max_seqlen_q=1,
            max_seqlen_k=256,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=512,
            sparse_mode=3,
            cmp_ratio=4,
            device=str(device),
        )

    expected = generate().cpu()
    assert expected.dtype == torch.int32 and expected.shape == (1024,)
    for sentinel in (0x01010101, 0x31313131, 0x7F7F7F7F):
        scratch = [torch.full((1024,), sentinel, dtype=torch.int32, device=device) for _ in range(16)]
        torch.npu.synchronize()
        del scratch
        actual = generate().cpu()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def check_attention_replay(runtime, heads, tokens, ratio, kv_tokens, replay_sequences):
    torch, device = runtime.torch, runtime.device
    generator = torch.Generator().manual_seed(20261011)
    original_cache, original_table = packed_cache(runtime, kv_tokens, generator)
    q = torch.empty((tokens, heads, HEAD_DIM), dtype=torch.bfloat16, device=device)
    sink = torch.empty(heads, dtype=torch.float32, device=device)
    seq_lens = torch.tensor([kv_tokens], dtype=torch.int32, device=device)
    query_start = torch.tensor([0, tokens], dtype=torch.int32, device=device)
    common = dict(
        kv_quant_mode=1,
        tile_size=64,
        rope_head_dim=64,
        cmp_ratio=ratio,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=BLOCK_SIZE - 1,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_ND",
        cu_seqlens_q=query_start,
        seqused_kv=seq_lens,
    )
    metadata_kwargs = dict(
        **common,
        num_heads_kv=1,
        head_dim=HEAD_DIM,
        seqused_q=torch.tensor([tokens], dtype=torch.int32, device=device),
        batch_size=1,
        max_seqlen_q=tokens,
        max_seqlen_kv=kv_tokens,
        has_ori_kv=True,
        has_cmp_kv=ratio > 1,
        cmp_topk=SPARSE_TOPK if ratio == 4 else 0,
    )
    attention_kwargs = dict(
        **common,
        ori_kv=original_cache,
        ori_block_table=original_table,
        softmax_scale=HEAD_DIM**-0.5,
    )
    indices = None
    if ratio > 1:
        compressed_cache, compressed_table = packed_cache(runtime, kv_tokens // ratio, generator)
        attention_kwargs.update(cmp_kv=compressed_cache, cmp_block_table=compressed_table)
        if ratio == 4:
            indices = torch.full((tokens, 1, SPARSE_TOPK), -1, dtype=torch.int32, device=device)
            attention_kwargs["cmp_sparse_indices"] = indices

    def set_inputs(sequence, iteration):
        full_q = (torch.randn((tokens, NATIVE_HEADS, HEAD_DIM), generator=generator) * 0.25).bfloat16()
        full_sink = torch.linspace(-0.5, 0.5, NATIVE_HEADS) + iteration * 0.25
        q.copy_(full_q[:, :heads].contiguous())
        sink.copy_(full_sink[:heads])
        seq_lens.fill_(sequence)
        if indices is not None:
            selected = torch.full((tokens, 1, SPARSE_TOPK), -1, dtype=torch.int32)
            for row in range(tokens):
                count = (sequence - tokens + row + 1) // ratio
                first = max(0, count - SPARSE_TOPK)
                selected[row, 0, : count - first] = torch.arange(first, count, dtype=torch.int32)
            indices.copy_(selected)
        # The native oracle receives nonzero extra heads, independently of the
        # adapter's zero-padding. Shared KV attention cannot mix query heads.
        native_metadata = runtime.native.npu_kv_quant_sparse_attn_sharedkv_metadata(
            num_heads_q=NATIVE_HEADS, **metadata_kwargs
        )
        expected = runtime.native.npu_kv_quant_sparse_attn_sharedkv(
            full_q.to(device), sinks=full_sink.to(device), metadata=native_metadata, **attention_kwargs
        )[0]
        torch.npu.synchronize()
        return expected[:, :heads].cpu()

    def forward():
        metadata = runtime.metadata(num_heads_q=heads, **metadata_kwargs)
        return runtime.attention(q, sinks=sink, metadata=metadata, **attention_kwargs)[0]

    expected = set_inputs(kv_tokens, 0)
    eager = forward()
    torch.npu.synchronize()
    torch.testing.assert_close(eager.cpu(), expected, rtol=0, atol=0)

    stream = torch.npu.Stream(device=device)
    stream.wait_stream(torch.npu.current_stream(device))
    with torch.npu.stream(stream):
        forward()
    stream.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        captured = forward()
    stream.synchronize()
    for iteration, sequence in enumerate(replay_sequences, 1):
        expected = set_inputs(sequence, iteration)
        with torch.npu.stream(stream):
            graph.replay()
        stream.synchronize()
        assert captured.shape == q.shape and captured.is_contiguous()
        torch.testing.assert_close(captured.cpu(), expected, rtol=0, atol=0)
    torch.npu.current_stream(device).wait_stream(stream)
