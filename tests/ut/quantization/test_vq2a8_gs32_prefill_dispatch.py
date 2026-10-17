# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise decode dispatch and startup prefill warmup with CPU scaffolding."""

import ast
import inspect
import sys
from contextlib import contextmanager, nullcontext
from enum import Enum
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest
import torch


class Mode(Enum):
    NONE = 0
    FULL = 1

    def valid_runtime_modes(self):
        return True


@pytest.fixture
def runner():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/model_runner_v1.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner")
    names = {"_determine_batch_execution_and_padding"}
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    source = "from __future__ import annotations\nclass Runner:\n" + "\n".join(
        "\n".join("    " + line for line in ast.unparse(node).splitlines()) for node in methods
    )
    scope = {
        "np": np,
        "CUDAGraphMode": Mode,
        "BatchDescriptor": lambda num_tokens: NS(num_tokens=num_tokens),
        "enable_sp": lambda _: False,
    }
    exec(compile(source, str(path), "exec"), scope)
    result = scope["Runner"]()
    result._pad_for_sequence_parallelism = lambda tokens: tokens
    result.input_batch = NS(
        num_computed_tokens_cpu=np.array([0, 0]), num_prompt_tokens=np.array([32, 32]), lora_id_to_lora_request={}
    )
    result.use_dcp = False
    result.uniform_decode_query_len = 1
    result.model_config = NS(is_encoder_decoder=False, quantization="vq2a8")
    result.vllm_config = NS(
        quant_config=NS(is_gs32=True),
        additional_config={"vq2a8": {"gs32_execution": "full_graph"}},
        parallel_config=NS(data_parallel_size=1),
        observability_config=NS(cudagraph_metrics=False),
    )

    def dispatch(num_tokens, **kwargs):
        if num_tokens > 32:
            return Mode.NONE, NS(num_tokens=num_tokens)
        return Mode.FULL, NS(num_tokens=1 if num_tokens == 1 else 32)

    result.cudagraph_dispatcher = NS(dispatch=Mock(side_effect=dispatch))
    return result


def run(runner, tokens, *, requests=1, **kwargs):
    return runner._determine_batch_execution_and_padding(
        tokens, requests, np.array([tokens // requests] * requests), tokens // requests, False, **kwargs
    )[:2]


@pytest.mark.parametrize(
    "computed,tokens,expected", [(0, 128, Mode.NONE), (8192, 128, Mode.NONE), (16384, 1, Mode.FULL)]
)
def test_decode_graph_dispatch_after_long_chunked_prefill(runner, computed, tokens, expected):
    runner.vllm_config.additional_config["vq2a8"]["gs32_execution"] = "full_graph"
    runner.input_batch.num_prompt_tokens[0] = 16384
    runner.input_batch.num_computed_tokens_cpu[0] = computed

    def dispatch(num_tokens, *, uniform_decode, **kwargs):
        # FULL_DECODE_ONLY with capture sizes [1] uses the ordinary vLLM
        # dispatcher. Only the query shape, not accumulated KV, is captured.
        mode = Mode.FULL if num_tokens == 1 and uniform_decode else Mode.NONE
        return mode, NS(num_tokens=num_tokens)

    runner.cudagraph_dispatcher.dispatch.side_effect = dispatch
    mode, descriptor = run(runner, tokens)
    assert mode == expected and descriptor.num_tokens == tokens
    assert bool(runner.cudagraph_dispatcher.dispatch.call_args.kwargs["uniform_decode"]) is (expected == Mode.FULL)


def _load_methods(filename, class_name, names, scope):
    path = Path(__file__).resolve().parents[3] / filename
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    cls.bases = []
    cls.decorator_list = []
    source = "from __future__ import annotations\n" + ast.unparse(cls)
    exec(compile(source, str(path), "exec"), scope)
    return scope[class_name]


@pytest.fixture
def startup_prefill(monkeypatch, request):
    capacity = getattr(request, "param", 128)
    synchronize = Mock()
    monkeypatch.setattr(torch, "npu", NS(synchronize=synchronize), raising=False)
    attn_state = NS(PrefillNoCache="prefill", DecodeOnly="decode")
    cls = _load_methods(
        "vllm_ascend/worker/model_runner_v1.py",
        "NPUModelRunner",
        {"_gs32_prefill_warmup", "_dummy_run", "_should_build_dummy_attn_metadata"},
        {
            "contextmanager": contextmanager,
            "torch": torch,
            "np": np,
            "cdiv": lambda a, b: (a + b - 1) // b,
            "CUDAGraphMode": Mode,
            "AscendAttentionState": attn_state,
        },
    )
    runner = cls()
    runner.device = torch.device("cpu")
    runner.max_num_tokens = capacity
    runner.max_model_len = 2 * capacity
    runner.max_num_reqs = 4
    runner.scheduler_config = NS(max_num_batched_tokens=capacity, max_num_seqs=4)
    runner.model_config = NS(get_vocab_size=lambda: 100)
    runner.input_ids = NS(gpu=torch.full((capacity,), 91, dtype=torch.int64))
    runner.positions = torch.full((capacity,), 7, dtype=torch.int64)
    runner._dsa_positions_cpu_buf = runner.positions.clone()
    runner.optimistic_seq_lens_cpu = torch.full((4,), 9, dtype=torch.int32)
    runner.seq_lens = runner.optimistic_seq_lens_cpu.clone()
    query_cpu = torch.full((5,), 3, dtype=torch.int32)
    runner.query_start_loc = NS(cpu=query_cpu, np=query_cpu.numpy(), gpu=query_cpu.clone())
    runner.query_start_loc.copy_to_gpu = lambda: runner.query_start_loc.gpu.copy_(query_cpu)
    query_pos = torch.zeros(capacity, dtype=torch.int32)
    runner.query_pos = NS(cpu=query_pos, np=query_pos.numpy())
    runner.attn_state = "old"
    runner.query_lens = torch.tensor([9])
    runner.dynamic_eplb = False
    runner.use_dcp = False
    runner.use_compress = True
    runner._has_gdn = False
    runner.speculative_config = None
    runner.uniform_decode_query_len = 1
    runner.device_metadata_executor = NS(submission_in_flight=False, release=Mock())
    runner.synchronize_input_prep = nullcontext
    runner._get_cumsum_and_arange = lambda values, _: np.cumsum(values)
    runner._determine_batch_execution_and_padding = lambda **kwargs: (
        Mode.NONE,
        NS(num_tokens=kwargs["num_tokens"], num_reqs=kwargs["num_reqs"]),
        None,
        None,
        None,
    )
    runner._pad_query_start_loc_for_fia = Mock(side_effect=AssertionError("profile_cpp must not pad requests"))

    class Table:
        def __init__(self):
            cpu = torch.full((4, capacity // 2), 7, dtype=torch.int32)
            self.block_table = NS(cpu=cpu, np=cpu.numpy(), gpu=cpu.clone())
            self.slot_mapping = NS(gpu=torch.full((capacity,), -7, dtype=torch.int32))
            self.num_blocks_per_row = np.zeros(4, dtype=np.int32)
            self.blocks_per_phys_block = 1

        def add_row(self, ids, row):
            self.num_blocks_per_row[row] = len(ids)
            self.block_table.np[row, : len(ids)] = ids

    class Tables(list):
        def commit_block_table(self, num_reqs):
            for table in self:
                table.block_table.gpu[:num_reqs].copy_(table.block_table.cpu[:num_reqs])

    tables = Tables(Table() for _ in range(3))
    for table, size in zip(tables, (32, 64, 2)):
        table.physical_block_size = size
    runner.input_batch = NS(
        num_reqs=0,
        num_prompt_tokens=np.array([9, 8, 7, 6]),
        num_computed_tokens_cpu=np.array([4, 3, 2, 1]),
        block_table=tables,
    )
    groups = [NS(kv_cache_spec=NS(block_size=size), layer_names=[str(index)]) for index, size in enumerate((32, 64, 2))]
    num_blocks = sum((capacity + size - 1) // size for size in (32, 64, 2)) + 3
    runner.kv_cache_config = NS(kv_cache_groups=groups, num_blocks=num_blocks)
    cache = torch.arange(num_blocks * 8, dtype=torch.float32).reshape(num_blocks, 2, 4)
    indexer = cache.clone()
    state = cache.clone()
    state[..., 2:] = -torch.inf
    runner.compilation_config = NS(
        static_forward_context={
            "0": NS(kv_cache=[[cache]]),
            "1": NS(kv_cache=[(indexer[..., :2], indexer[..., 2:], indexer)]),
            "2": NS(kv_cache=[[state]]),
        }
    )
    return runner, (cache, indexer, state), synchronize


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("startup_prefill", [128, 4096, 8192], indirect=True)
def test_startup_prefill_restores_touched_cache_pages_and_host_state(startup_prefill, fail):
    runner, caches, synchronize = startup_prefill
    tokens = runner.max_num_tokens
    group_blocks = [tokens // group.kv_cache_spec.block_size for group in runner.kv_cache_config.kv_cache_groups]
    original_caches = [cache.clone() for cache in caches]
    original_positions = runner.positions.clone()
    original_ids = runner.input_ids.gpu.clone()
    original_query = runner.query_start_loc.cpu.clone()
    original_seq = runner.seq_lens.clone()
    old_query_lens = runner.query_lens

    def execute():
        with runner._gs32_prefill_warmup(tokens) as actual:
            assert actual == tokens
            assert runner.input_batch.num_prompt_tokens[0] == tokens
            assert runner.input_batch.num_computed_tokens_cpu[0] == 0
            torch.testing.assert_close(runner.input_ids.gpu, torch.arange(tokens) % 100)
            first_block = 1
            for table, blocks in zip(runner.input_batch.block_table, group_blocks):
                assert table.num_blocks_per_row[0] == blocks
                np.testing.assert_array_equal(
                    table.block_table.np[0, :blocks], np.arange(first_block, first_block + blocks)
                )
                torch.testing.assert_close(
                    table.slot_mapping.gpu,
                    torch.arange(tokens, dtype=torch.int32) + first_block * table.physical_block_size,
                )
                first_block += blocks
            first_block = 1
            for cache, blocks in zip(caches, group_blocks):
                cache[first_block : first_block + blocks].fill_(123)
                first_block += blocks
            runner.positions.fill_(127)
            runner._dsa_positions_cpu_buf.fill_(127)
            runner.query_start_loc.cpu[:2] = torch.tensor([0, tokens])
            runner.query_start_loc.copy_to_gpu()
            runner.seq_lens.fill_(tokens)
            runner.optimistic_seq_lens_cpu.fill_(tokens)
            runner.attn_state = "prefill"
            runner.query_lens = torch.tensor([tokens])
            if fail:
                runner.device_metadata_executor.submission_in_flight = True
                raise RuntimeError("forward failed")

    if fail:
        with pytest.raises(RuntimeError, match="forward failed"):
            execute()
    else:
        execute()
    for original, cache in zip(original_caches, caches):
        torch.testing.assert_close(cache, original)
    for table in runner.input_batch.block_table:
        assert table.num_blocks_per_row[0] == 0
        assert torch.all(table.block_table.cpu == 7) and torch.all(table.block_table.gpu == 7)
        assert torch.all(table.slot_mapping.gpu == -7)
    torch.testing.assert_close(runner.positions, original_positions)
    torch.testing.assert_close(runner._dsa_positions_cpu_buf, original_positions)
    torch.testing.assert_close(runner.input_ids.gpu, original_ids)
    torch.testing.assert_close(runner.query_start_loc.cpu, original_query)
    torch.testing.assert_close(runner.query_start_loc.gpu, original_query)
    torch.testing.assert_close(runner.seq_lens, original_seq)
    torch.testing.assert_close(runner.optimistic_seq_lens_cpu, original_seq)
    assert runner.input_batch.num_prompt_tokens.tolist() == [9, 8, 7, 6]
    assert runner.input_batch.num_computed_tokens_cpu.tolist() == [4, 3, 2, 1]
    assert runner.query_lens is old_query_lens and runner.attn_state == "old"
    assert runner._gs32_prefill_warmup_active is False
    assert synchronize.call_count == 3
    assert runner.device_metadata_executor.release.call_count == int(fail)


@pytest.mark.parametrize(
    "fault", ["active", "pages", "shared_pages", "table", "tensor", "tensor_end", "tokens", "model_len"]
)
def test_startup_prefill_rejects_unsafe_scope_without_mutation(startup_prefill, fault):
    runner, caches, _ = startup_prefill
    original = [cache.clone() for cache in caches]
    tokens = 128
    if fault == "active":
        runner.input_batch.num_reqs = 1
    elif fault == "pages":
        runner.kv_cache_config.num_blocks = 1
    elif fault == "shared_pages":
        runner.kv_cache_config.num_blocks = 3
    elif fault == "table":
        runner.input_batch.block_table[0].blocks_per_phys_block = 128
    elif fault == "tensor":
        runner.compilation_config.static_forward_context["0"].kv_cache = [[caches[0][:1]]]
    elif fault == "tensor_end":
        # The third group needs pages [7,71), not merely 64 pages from index 1.
        runner.compilation_config.static_forward_context["2"].kv_cache = [[caches[2][:65]]]
    elif fault == "model_len":
        runner.max_model_len = tokens - 1
    else:
        tokens = 129
    with pytest.raises((RuntimeError, ValueError)), runner._gs32_prefill_warmup(tokens):
        pytest.fail("unsafe scope entered")
    for before, cache in zip(original, caches):
        torch.testing.assert_close(before, cache)
    assert runner.input_batch.num_prompt_tokens[0] == 9


@pytest.mark.parametrize(
    "requested,kv_pages,table_width,expected",
    [
        (8192, 4483, 4096, 8192),
        (8192, 2241, 4096, 4096),
        (8192, 4483, 2048, 4096),
        (8192, 71, 64, 128),
        (8192, 70, 64, 126),
        (128, 4, 64, 2),
        (128, 71, 1, 2),
        (1, 71, 64, 1),
    ],
)
@pytest.mark.parametrize("startup_prefill", [8192], indirect=True)
def test_startup_prefill_caps_work_and_cache_snapshots_to_shared_capacity(
    startup_prefill, requested, kv_pages, table_width, expected
):
    runner, caches, _ = startup_prefill
    runner.max_num_tokens = runner.max_model_len = 8192
    runner.kv_cache_config.num_blocks = kv_pages
    for table in runner.input_batch.block_table:
        table.block_table.cpu = table.block_table.cpu[:, :table_width]
        table.block_table.gpu = table.block_table.gpu[:, :table_width]
        table.block_table.np = table.block_table.cpu.numpy()
    before = [cache.clone() for cache in caches]
    with runner._gs32_prefill_warmup(requested) as actual:
        assert actual == expected
        assert runner.input_batch.num_prompt_tokens[0] == expected
        for group, table in zip(runner.kv_cache_config.kv_cache_groups, runner.input_batch.block_table):
            assert (
                table.num_blocks_per_row[0]
                == (expected + group.kv_cache_spec.block_size - 1) // group.kv_cache_spec.block_size
            )
    for old, cache in zip(before, caches):
        torch.testing.assert_close(cache, old)


@pytest.mark.parametrize("startup_prefill", [8192], indirect=True)
@pytest.mark.parametrize("available_pages,expected", [(4480, 8192), (2240, 4096)])
@pytest.mark.parametrize("fail", [False, True])
def test_startup_prefill_borrows_disjoint_pages_from_shared_storage(startup_prefill, available_pages, expected, fail):
    runner, caches, _ = startup_prefill
    shared = caches[0]
    original = shared.clone()
    runner.kv_cache_config.num_blocks = available_pages + 1
    for layer in runner.compilation_config.static_forward_context.values():
        layer.kv_cache = [[shared]]
    written_ranges = []

    def execute():
        with runner._gs32_prefill_warmup(8192) as actual:
            assert actual == expected
            for index, table in enumerate(runner.input_batch.block_table):
                blocks = int(table.num_blocks_per_row[0])
                pages = table.block_table.cpu[0, :blocks].tolist()
                assert len(pages) == len(set(pages)) and min(pages) > 0
                assert not any(set(pages) & set(previous) for previous, _ in written_ranges)
                torch.testing.assert_close(
                    table.slot_mapping.gpu[:actual],
                    torch.arange(actual, dtype=torch.int32) + pages[0] * table.physical_block_size,
                )
                value = 100 + index
                shared[pages] = value
                written_ranges.append((pages, value))
                # Later cache groups must not overwrite any preceding group.
                for previous, previous_value in written_ranges:
                    assert torch.all(shared[previous] == previous_value)
            assert sum(len(pages) for pages, _ in written_ranges) == available_pages
            torch.testing.assert_close(shared[0], original[0])
            torch.testing.assert_close(shared[available_pages + 1 :], original[available_pages + 1 :])
            if fail:
                raise RuntimeError("shared-cache forward failed")

    if fail:
        with pytest.raises(RuntimeError, match="shared-cache forward failed"):
            execute()
    else:
        execute()
    torch.testing.assert_close(shared, original)
    assert runner.input_batch.num_prompt_tokens.tolist() == [9, 8, 7, 6]
    assert runner.input_batch.num_computed_tokens_cpu.tolist() == [4, 3, 2, 1]
    assert runner._gs32_prefill_warmup_active is False
    for table in runner.input_batch.block_table:
        assert table.num_blocks_per_row[0] == 0
        assert torch.all(table.block_table.cpu == 7) and torch.all(table.block_table.gpu == 7)
        assert torch.all(table.slot_mapping.gpu == -7)


@pytest.mark.parametrize("startup_prefill", [128, 4096, 8192], indirect=True)
def test_startup_prefill_dummy_builds_fresh_b1_metadata_and_restores_on_error(startup_prefill):
    runner, _, _ = startup_prefill
    tokens = runner.max_num_tokens

    def inspect_metadata(**kwargs):
        assert kwargs["num_reqs"] == kwargs["num_reqs_padded"] == 1
        assert kwargs["num_tokens"] == kwargs["max_query_len"] == tokens
        assert kwargs["for_cudagraph_capture"] is False
        assert runner.attn_state == "prefill"
        torch.testing.assert_close(runner.positions, torch.arange(tokens))
        torch.testing.assert_close(runner._dsa_positions_cpu_buf, torch.arange(tokens))
        torch.testing.assert_close(runner.query_start_loc.gpu[:2], torch.tensor([0, tokens], dtype=torch.int32))
        assert runner.seq_lens.tolist() == [tokens, 0, 0, 0]
        for table in runner.input_batch.block_table:
            first_block = int(table.block_table.cpu[0, 0])
            torch.testing.assert_close(
                table.slot_mapping.gpu,
                torch.arange(tokens, dtype=torch.int32) + first_block * table.physical_block_size,
            )
        raise RuntimeError("metadata inspected")

    runner._build_attention_metadata = inspect_metadata
    with pytest.raises(RuntimeError, match="metadata inspected"), runner._gs32_prefill_warmup(tokens):
        runner._dummy_run(tokens, force_attention=True, profile_cpp=True, cudagraph_runtime_mode=Mode.NONE)
    assert runner.attn_state == "old" and torch.all(runner.positions == 7)


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize(
    "target,max_len,budget,expected",
    [
        (True, 256, 128, 128),
        (True, 64, 128, 64),
        (False, 256, 128, None),
        (True, 263168, 4096, 4096),
        (True, 263168, 8192, 8192),
    ],
)
def test_startup_prefill_worker_runs_on_every_rank_before_capture(monkeypatch, rank, target, max_len, budget, expected):
    events = []
    module_name = "vllm_ascend.model_executor.warmup.kernel_warmup"
    module = ModuleType(module_name)
    module.kernel_warmup = lambda _: events.append("kernel_warmup")
    monkeypatch.setitem(sys.modules, module_name, module)
    cls = _load_methods(
        "vllm_ascend/worker/worker.py",
        "NPUWorker",
        {"compile_or_warm_up_model"},
        {
            "logger": Mock(),
            "CUDAGraphMode": Mode,
            "get_current_hardware_profile": lambda: NS(supports=lambda _: False),
            "HardwareCapability": NS(ATB_WARMUP=0),
            "get_ascend_config": lambda: NS(enable_cpu_binding=False),
            "set_random_seed": lambda _: events.append("reset_seed"),
            "CompilationTimes": lambda **kwargs: kwargs,
        },
    )
    worker = cls()
    worker.local_rank = rank
    worker.model_config = NS(
        quantization="vq2a8" if target else "fp8",
        hf_text_config=NS(model_type="deepseek_v4"),
        enforce_eager=False,
        seed=123,
        max_model_len=max_len,
    )
    worker.vllm_config = NS(
        quant_config=NS(is_gs32=True),
        compilation_config=NS(
            compile_sizes=[],
            cudagraph_mode=Mode.FULL,
            cudagraph_capture_sizes=[1],
            get_compile_ranges=lambda: [],
            compilation_time=0,
        ),
    )
    worker.scheduler_config = NS(max_num_batched_tokens=budget)
    worker.cache_config = NS(kv_cache_memory_bytes=1024)

    @contextmanager
    def scope(tokens):
        assert tokens == expected
        events.append("prefill_enter")
        yield tokens
        events.append("prefill_restore")

    worker.model_runner = NS(_gs32_prefill_warmup=scope, capture_model=lambda: events.append("capture"))
    worker.profile_prefill_latency = Mock(side_effect=lambda tokens: events.append("prefill") or 1000.0)
    worker.compile_or_warm_up_model()
    prefill = ["prefill_enter", "prefill", "prefill_restore"] if target else []
    assert events == [*prefill, "kernel_warmup", "capture", "reset_seed"]
    if target:
        worker.profile_prefill_latency.assert_called_once_with(expected)
    else:
        worker.profile_prefill_latency.assert_not_called()


def test_prefill_latency_helper_preserves_existing_api_and_restores_runner_on_failure(monkeypatch):
    monkeypatch.setattr(torch, "npu", NS(synchronize=Mock()), raising=False)
    cls = _load_methods(
        "vllm_ascend/worker/worker.py",
        "NPUWorker",
        {"profile_prefill_latency"},
        {"torch": torch, "inspect": inspect, "CUDAGraphMode": Mode, "get_pp_group": lambda: NS(is_first_rank=True)},
    )
    worker = cls()
    worker.scheduler_config = NS(max_num_batched_tokens=128)

    def dummy(*, num_tokens, force_attention, is_profile, cudagraph_runtime_mode, profile_cpp=False):
        assert num_tokens == 128 and force_attention and not is_profile and profile_cpp
        assert cudagraph_runtime_mode == Mode.NONE and worker.model_runner.max_num_reqs == 1
        raise RuntimeError("prefill failed")

    worker.model_runner = NS(max_num_reqs=4, _dummy_run=dummy)
    with pytest.raises(RuntimeError, match="prefill failed"):
        worker.profile_prefill_latency(256)
    assert worker.model_runner.max_num_reqs == 4
