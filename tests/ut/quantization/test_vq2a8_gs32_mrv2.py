# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MRV2 GS32 ownership and replay contracts on real CPU tensors; no NPU engine."""
# ruff: noqa: F401, F811  # Imported pytest fixtures are injected by name.

import ast
import importlib.util
import json
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from tests.ut.quantization.test_vq2a8_gs32_fixtures import modules, serving_config, source_layer_spec
from tests.ut.quantization.test_vq2a8_gs32_graph_config import long_graph_config
from tests.ut.quantization.test_vq2a8_gs32_graph_owner import _method_ast

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture
def v2_quantization_runner(monkeypatch):
    """Run MRV2 ownership code on real CPU tensors, without an NPU engine."""
    state = NS(events=[], values=[True, True], skipped=set(), capturing=False, forward_error=None)
    owners = [NS(validity_buffer=torch.zeros((), dtype=torch.bool)) for _ in range(2)]
    layers = [NS(vq2a8_layer_index=index, vq2a8_weights=owner) for index, owner in enumerate(owners)]

    class Parent:
        def load_model(self, *args, **kwargs):
            state.events.append("load")

        def execute_model(self, scheduler_output, **kwargs):
            state.execute_kwargs = kwargs
            if not scheduler_output.total_num_scheduled_tokens:
                return None
            state.events.append("forward")
            if self._quantization_validity is not None:
                assert not self._quantization_validity.any()
                for index, owner in enumerate(owners):
                    if index not in state.skipped:
                        owner.validity_buffer.fill_(state.values[index])
            self.execute_model_state = NS(hidden_states=torch.ones(1, 4))
            if state.forward_error is not None:
                raise state.forward_error

        def sample(self, *args):
            state.events.append("upstream_sample")

        def prepare_dummy_attn(self, batch, valid_state_slots=False):
            state.valid_state_slots = valid_state_slots
            return "upstream_blocks", "upstream_slots"

    scope = {
        "__name__": __name__,
        "torch": torch,
        "np": np,
        "GPUModelRunner": Parent,
        "CUDAGraphMode": NS(NONE=0, FULL=1, FULL_DECODE_ONLY=2),
        "breakable_cudagraph": NS(is_breakable_cudagraph_enabled=lambda: False),
        "nullcontext": nullcontext,
        "get_pd_dspark_aux_layer_ids": lambda _: (),
        "has_kv_transfer_group": lambda: False,
        "should_skip_allreduce_across_dp_group": lambda _: False,
        "IntermediateTensors": type("IntermediateTensors", (), {}),
        "prepare_v41_dummy_ring_state": lambda *_: None,
        "_start_profiling_chunk_timing": lambda *_: None,
        "_finish_profiling_chunk_timing": lambda *_: None,
        "lmhead_tp_enable": lambda: False,
        "contextmanager": contextmanager,
        "AscendAttentionState": NS(PrefillNoCache="prefill"),
        "update_cos_sin": lambda positions: state.events.append(("rope", positions.clone())),
        "lmhead_tp_pad_rows": lambda rows, capacity, _: (torch.nn.functional.pad(rows, (0, capacity - len(rows)))),
    }
    names = (
        "load_model",
        "_check_quantization_validity",
        "execute_model",
        "sample",
        "_gs32_prefill_warmup",
        "prepare_dummy_attn",
        "_pad_query_start_loc_for_fia",
    )
    methods = [_method_ast("vllm_ascend/worker/v2/model_runner.py", "NPUModelRunner", name) for name in names]
    cls = ast.ClassDef(
        name="Runner",
        bases=[ast.Name(id="GPUModelRunner", ctx=ast.Load())],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, cls], type_ignores=[])), "v2_gs32", "exec"), scope)
    runner = scope["Runner"]()
    runner.device = torch.device("cpu")
    runner.model_config = NS(quantization="vq2a8", get_vocab_size=lambda: 7)
    runner.compilation_config = NS(cudagraph_mode=0)
    runner.vllm_config = NS(
        additional_config={"vq2a8": {"gs32_execution": "full_graph"}}, quant_config=NS(is_gs32=True, layers=[0, 1])
    )
    runner.ascend_config = NS(scheduler_config=NS(profiling_chunk_config=None))
    runner.execute_model_state = None
    runner.is_last_pp_rank = True
    runner.is_pooling_model = False
    runner.adaptive_verification = None
    runner.model_state = NS(finish_execution=Mock())
    runner.kvpp = NS(complete_forward=Mock())
    runner._start_dump_data = Mock()
    runner._finalize_dump_data = Mock()
    runner._maybe_send_draft_kv = Mock()
    runner._maybe_extend_slot_mappings = lambda _, slots: slots
    runner.model = NS(
        modules=lambda: layers,
        compute_logits=lambda hidden: state.events.append("logits") or torch.ones(hidden.shape[0], 8),
    )
    runner.rejection_sampler = None
    runner.sampler = lambda *_: state.events.append("sample") or NS(num_sampled=1, num_rejected=0)
    runner.load_model()
    original_bool = torch.Tensor.__bool__

    def read_bool(tensor):
        assert not state.capturing, "No host read is allowed in graph capture."
        if tensor.ndim == 0:
            state.events.append("read")
        return original_bool(tensor)

    monkeypatch.setattr(torch.Tensor, "__bool__", read_bool)
    monkeypatch.setattr(
        torch, "npu", NS(synchronize=Mock(), is_current_stream_capturing=lambda: state.capturing), raising=False
    )
    state.events.clear()
    return NS(
        runner=runner,
        state=state,
        owners=owners,
        scope=scope,
        batch=NS(logits_indices=torch.tensor([0]), num_draft_tokens=0),
    )


@pytest.mark.parametrize("valid_state_slots", [False, True])
@pytest.mark.parametrize("failure", ["flag", "missed_write"])
def test_v2_gs32_generation_checks_after_logits_before_sampling_and_recovers(
    v2_quantization_runner, valid_state_slots, failure
):
    h = v2_quantization_runner
    runner = h.runner
    pointers = [owner.validity_buffer.data_ptr() for owner in h.owners]
    for valid in (True, False, True):
        h.state.events.clear()
        h.state.values = [True, valid]
        h.state.skipped = {1} if not valid and failure == "missed_write" else set()
        runner.execute_model(
            NS(total_num_scheduled_tokens=1, num_scheduled_tokens={"request": 1}),
            valid_dummy_state_slots=valid_state_slots,
        )
        assert runner._quantization_validity_pending
        assert "sample" not in h.state.events and "logits" not in h.state.events
        if valid:
            runner.sample(torch.ones(1, 4), h.batch, None)
            assert h.state.events[-3:] == ["logits", "read", "sample"]
        else:
            with pytest.raises(RuntimeError, match="failed validity"):
                runner.sample(torch.ones(1, 4), h.batch, None)
            assert "sample" not in h.state.events
            assert runner.execute_model_state is None
        assert not runner._quantization_validity_pending
        assert pointers == [owner.validity_buffer.data_ptr() for owner in h.owners]
    assert h.state.execute_kwargs["context_len"] == 0
    assert h.state.execute_kwargs["valid_dummy_state_slots"] is valid_state_slots
    runner.model_state.finish_execution.assert_called_with(failed=False)


@pytest.mark.parametrize("mode", ["dummy", "profile", "pp"])
def test_v2_gs32_no_sampler_paths_check_before_return(v2_quantization_runner, mode):
    h = v2_quantization_runner
    h.state.values[1] = False
    h.runner.is_last_pp_rank = mode != "pp"
    with pytest.raises(RuntimeError, match="failed validity"):
        h.runner.execute_model(
            NS(total_num_scheduled_tokens=1, num_scheduled_tokens={"request": 1}),
            dummy_run=mode == "dummy",
            is_profile=mode == "profile",
        )
    assert h.runner.execute_model_state is None
    assert not h.runner._quantization_validity_pending


@pytest.mark.parametrize("num_requests", [1, 3, 7, 15, 16])
def test_v2_gs32_dspark_checks_target_before_rejection_and_preserves_logits(v2_quantization_runner, num_requests):
    """DSpark verifies six target tokens per request, including the bonus."""
    h = v2_quantization_runner
    query_len = 6
    num_tokens = num_requests * query_len
    # Nontrivial indices expose accidental truncation to request count.
    hidden = torch.arange((num_tokens + 2) * 4, dtype=torch.float32).reshape(-1, 4)
    indices = torch.arange(1, num_tokens + 1).flip(0)
    batch = NS(logits_indices=indices, num_draft_tokens=num_requests * (query_len - 1))
    draft_logits = torch.arange(num_requests * (query_len - 1) * 8).reshape(-1, 8)
    h.runner.speculator = NS(draft_logits=draft_logits)
    observed_hidden = []

    def compute_logits(selected):
        h.state.events.append("logits")
        observed_hidden.append(selected.clone())
        return selected.repeat(1, 2)

    def reject(logits, received_batch, received_draft_logits):
        assert h.state.events[-2:] == ["logits", "read"]
        h.state.events.append("reject")
        assert received_batch is batch
        assert received_draft_logits is draft_logits
        torch.testing.assert_close(logits, hidden[indices].repeat(1, 2))
        return NS(num_sampled=torch.ones(num_requests), num_rejected=torch.zeros(num_requests))

    h.runner.model.compute_logits = compute_logits
    h.runner.rejection_sampler = Mock(side_effect=reject)
    h.runner.sampler = Mock(side_effect=AssertionError("Speculative verification used the ordinary sampler"))
    scheduler = NS(
        total_num_scheduled_tokens=num_tokens,
        num_scheduled_tokens={f"request-{index}": query_len for index in range(num_requests)},
    )
    for valid in (True, False, True):
        h.state.events.clear()
        h.state.values = [True, valid]
        h.runner.rejection_sampler.reset_mock()
        h.runner.execute_model(scheduler)
        if valid:
            output, sampled, rejected = h.runner.sample(hidden, batch, None)
            assert sampled is output.num_sampled and rejected is output.num_rejected
            h.runner.rejection_sampler.assert_called_once()
        else:
            with pytest.raises(RuntimeError, match="failed validity"):
                h.runner.sample(hidden, batch, None)
            h.runner.rejection_sampler.assert_not_called()
            assert h.runner.execute_model_state is None
        torch.testing.assert_close(observed_hidden[-1], hidden[indices])
        assert not h.runner._quantization_validity_pending
    h.runner.sampler.assert_not_called()


@pytest.mark.parametrize("num_requests", [1, 2, 3, 5, 8, 15, 16])
@pytest.mark.parametrize("query_len", [1, 6], ids=["ordinary-decode", "dspark5-verify"])
def test_v2_gs32_uniform_decode_padding_keeps_all_verification_tokens(v2_quantization_runner, num_requests, query_len):
    h = v2_quantization_runner
    runner = h.runner
    runner.decode_query_len = 6
    runner.compilation_config.cudagraph_mode = h.scope["CUDAGraphMode"].FULL_DECODE_ONLY
    bucket = next(size for size in (1, 2, 4, 8, 16) if size >= num_requests)
    tokens = num_requests * query_len
    offsets = np.full(18, tokens, dtype=np.int32)
    offsets[: num_requests + 1] = np.arange(num_requests + 1) * query_len
    padded, padded_requests = runner._pad_query_start_loc_for_fia(
        num_tokens_padded=bucket * query_len,
        num_reqs_padded=bucket,
        num_reqs=num_requests,
        query_start_loc_np=offsets,
        cudagraph_runtime_mode=h.scope["CUDAGraphMode"].FULL,
        batch_desc_num_reqs=bucket,
        uniform_query_len=query_len,
    )
    assert padded_requests == bucket
    np.testing.assert_array_equal(padded[: bucket + 1], np.arange(bucket + 1) * query_len)
    assert padded[num_requests] == tokens


def test_v2_gs32_empty_step_and_unconsumed_step_status(v2_quantization_runner):
    h = v2_quantization_runner
    h.runner.execute_model(NS(total_num_scheduled_tokens=0, num_scheduled_tokens={}))
    assert h.state.events == []
    h.state.values[1] = False
    h.runner.execute_model(NS(total_num_scheduled_tokens=1, num_scheduled_tokens={"request": 1}))
    with pytest.raises(RuntimeError, match="failed validity"):
        h.runner.execute_model(NS(total_num_scheduled_tokens=0, num_scheduled_tokens={}))
    assert h.runner.execute_model_state is None


def test_v2_gs32_failed_forward_discards_partial_state_and_recovers(v2_quantization_runner):
    h = v2_quantization_runner
    runner = h.runner
    # Emulate an interrupted submission after the parent has installed state.
    # Preserve the original exception and make a later sample_tokens a no-op.
    failure = RuntimeError("model submission failed")
    h.state.forward_error = failure
    with pytest.raises(RuntimeError, match="model submission failed") as raised:
        runner.execute_model(NS(total_num_scheduled_tokens=1, num_scheduled_tokens={"request": 1}))
    assert raised.value is failure
    runner.model_state.finish_execution.assert_called_once_with(failed=True)
    assert runner.execute_model_state is None
    assert not runner._quantization_validity_pending
    assert "sample" not in h.state.events

    h.state.forward_error = None
    h.state.events.clear()
    runner.execute_model(NS(total_num_scheduled_tokens=1, num_scheduled_tokens={"request": 1}))
    runner.sample(torch.ones(1, 4), h.batch, None)
    assert h.state.events[-3:] == ["logits", "read", "sample"]
    assert not runner._quantization_validity_pending


def test_v2_gs32_finish_execution_error_discards_partial_state(v2_quantization_runner):
    h = v2_quantization_runner
    h.runner.model_state.finish_execution.side_effect = RuntimeError("metadata cleanup failed")
    with pytest.raises(RuntimeError, match="metadata cleanup failed"):
        h.runner.execute_model(NS(total_num_scheduled_tokens=1, num_scheduled_tokens={"request": 1}))
    h.runner.model_state.finish_execution.assert_called_once_with(failed=False)
    assert h.runner.execute_model_state is None
    assert not h.runner._quantization_validity_pending
    assert "sample" not in h.state.events


def test_v2_other_quantization_keeps_upstream_sampling(v2_quantization_runner):
    h = v2_quantization_runner
    h.runner.model_config.quantization = "ascend"
    h.runner.load_model()
    h.state.events.clear()
    h.runner.execute_model(NS(total_num_scheduled_tokens=1, num_scheduled_tokens={"request": 1}))
    h.runner.sample(torch.ones(1, 4), h.batch, None)
    assert h.state.events == ["forward", "upstream_sample"]
    assert h.runner._quantization_validity is None


def test_v2_gs32_startup_prefill_forces_eager_without_leaving_pending_status(v2_quantization_runner):
    h = v2_quantization_runner
    h.runner._gs32_prefill_warmup_active = True
    h.runner.execute_model(NS(total_num_scheduled_tokens=1, num_scheduled_tokens={"request": 1}), dummy_run=True)
    assert h.state.execute_kwargs["is_profile"] is True
    assert not h.runner._quantization_validity_pending


@pytest.mark.parametrize("auxiliary_outputs", [False, True], ids=["ordinary", "dspark-target"])
def test_v2_gs32_capture_wrapper_has_no_host_check_in_capture(v2_quantization_runner, auxiliary_outputs):
    h = v2_quantization_runner
    source = ast.parse((REPO / "vllm_ascend/worker/v2/aclgraph_utils.py").read_text(encoding="utf-8"))
    cls = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "ModelWithContext")
    context = NS(cudagraph_runtime_mode=1)
    extra = NS(capturing=False)
    scope = {
        "nn": torch.nn,
        "torch": torch,
        "get_forward_context": lambda: context,
        "_EXTRA_CTX": extra,
        "CUDAGraphMode": NS(PIECEWISE=2),
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), "v2_capture", "exec"), scope)

    target_hidden = torch.ones(96, 4)
    result = (target_hidden, [target_hidden + layer for layer in range(3)]) if auxiliary_outputs else target_hidden

    def forward():
        for owner, valid in zip(h.owners, h.state.values):
            owner.validity_buffer.fill_(valid)
        return result

    metadata = NS(begin_forward=Mock(), finish=Mock())
    wrapper = scope["ModelWithContext"](forward, device_metadata=metadata, model_runner=h.runner)
    for capturing in (False, True, False):
        h.state.capturing = capturing
        h.state.values[1] = False
        h.state.events.clear()
        metadata.begin_forward.reset_mock()
        metadata.finish.reset_mock()
        if capturing:
            assert wrapper() is result
            assert h.state.events == []
        else:
            with pytest.raises(RuntimeError, match="failed validity"):
                wrapper()
        metadata.begin_forward.assert_called_once_with()
        metadata.finish.assert_called_once_with()
    h.state.values[1] = True
    assert wrapper() is result


@pytest.mark.parametrize("fail", [False, True])
def test_v2_gs32_startup_prefill_borrows_and_restores_only_live_pages(v2_quantization_runner, fail):
    h = v2_quantization_runner
    runner = h.runner
    runner.max_num_tokens = runner.max_model_len = 16
    runner.req_states = NS(req_id_to_index={})
    runner.model_state = NS(attn_metadata="original")
    runner.kv_cache_config = NS(
        num_blocks=12,
        kv_cache_groups=[
            NS(kv_cache_spec=NS(block_size=2), layer_names=["first"]),
            NS(kv_cache_spec=NS(block_size=4), layer_names=["second"]),
        ],
    )
    caches = [torch.arange(12 * 8).reshape(12, 8).float() for _ in range(2)]
    runner.compilation_config.static_forward_context = {
        name: NS(kv_cache=[cache]) for name, cache in zip(("first", "second"), caches)
    }
    runner.block_tables = NS(
        input_block_tables=[torch.full((1, 16), 9, dtype=torch.int32) for _ in range(2)],
        blocks_per_kv_block=[2, 1],
        slot_mappings=torch.full((2, 16), 99, dtype=torch.int32),
    )
    runner.input_buffers = NS(
        input_ids=torch.full((16,), 5, dtype=torch.int64),
        positions=torch.full((16,), 7, dtype=torch.int64),
        seq_lens=torch.ones(1, dtype=torch.int32),
        seq_lens_cpu=torch.ones(1, dtype=torch.int32),
        query_start_loc=torch.ones(3, dtype=torch.int32),
        is_padding=torch.ones(16, dtype=torch.bool),
    )
    buffers = [
        *vars(runner.input_buffers).values(),
        *runner.block_tables.input_block_tables,
        runner.block_tables.slot_mappings,
        *caches,
    ]
    snapshots = [value.clone() for value in buffers]

    def warmup():
        with runner._gs32_prefill_warmup(16) as tokens:
            # Shared physical allocation: ceil(N/2)+ceil(N/4) <= 11.
            assert tokens == 14
            batch = NS(
                num_reqs=1,
                num_tokens=tokens,
                num_tokens_after_padding=tokens,
                is_prefilling_np=np.zeros(1, dtype=bool),
                prefill_len_np=np.zeros(1),
                num_computed_tokens_np=np.ones(1),
                num_computed_prefill_tokens_np=np.ones(1),
                input_ids=runner.input_buffers.input_ids[:tokens],
                positions=runner.input_buffers.positions[:tokens],
                is_padding=runner.input_buffers.is_padding[:tokens],
            )
            tables, slots = runner.prepare_dummy_attn(batch)
            assert batch.attn_state == "prefill" and batch.is_prefilling_np.all() and batch.has_prefill
            assert torch.equal(batch.positions, torch.arange(tokens))
            assert not batch.is_padding.any()
            assert tables[0][0, :14].tolist() == list(range(2, 16))
            assert tables[1][0, :4].tolist() == list(range(8, 12))
            assert slots[0, 0] == 2 and slots[1, 0] == 32
            caches[0][1:8].zero_()
            caches[1][8:12].zero_()
            runner.model_state.attn_metadata = "warmup"
            if fail:
                raise RuntimeError("warmup failed")

    if fail:
        with pytest.raises(RuntimeError, match="warmup failed"):
            warmup()
    else:
        warmup()
    for actual, expected in zip(buffers, snapshots):
        assert torch.equal(actual, expected)
    assert runner.model_state.attn_metadata == "original"
    assert not runner._gs32_prefill_warmup_active
    assert runner.prepare_dummy_attn(NS(num_reqs=1), valid_state_slots=True) == ("upstream_blocks", "upstream_slots")
    assert h.state.valid_state_slots is True


@pytest.fixture
def probe():
    path = REPO / "tests/e2e/nightly/test_vq2a8_gs32_ep.py"
    spec = importlib.util.spec_from_file_location("gs32_ep_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("version", ["0", "1"])
def test_gs32_ep_environment_preserves_model_runner_selection(probe, version):
    environment = {"VLLM_USE_V2_MODEL_RUNNER": version, "ASCEND_RT_VISIBLE_DEVICES": "4,5"}
    assert probe.variant_environment(environment).items() >= environment.items()


def test_gs32_ep_v1_graph_state_uses_existing_entry_replays(probe):
    cls = type("NPUModelRunner", (), {"__module__": "vllm_ascend.worker.model_runner_v1"})
    runner = cls()
    runner.model = NS(
        concrete_aclgraph_entries={
            1: NS(aclgraph=object(), batch_descriptor=NS(num_tokens=1), num_replays=7),
            32: NS(aclgraph=object(), batch_descriptor=NS(num_tokens=32), num_replays=2),
            128: NS(aclgraph=None),
        }
    )
    state = probe._runner_graph_state(runner)
    assert state["model_runner"] == "vllm_ascend.worker.model_runner_v1.NPUModelRunner"
    assert state["model_runner_version"] == 1
    assert state["decode"] == {"captures": 1, "replays": 7}
    assert state["prefill"] == {"captures": 1, "replays": 2}
    assert state["other"] == {"captures": 0, "replays": 0}


def test_gs32_ep_v2_graph_state_counts_actual_replay_without_reset(probe):
    cls = type("NPUModelRunner", (), {"__module__": "vllm_ascend.worker.v2.model_runner"})
    runner = cls()
    descriptor = type("Descriptor", (), {"num_tokens": 1})()
    original_replay = Mock(return_value="replayed")
    graph_type = type("Graph", (), {})
    graph = graph_type()
    graph.replay = original_replay
    graph.pool = "retained"
    runner.cudagraph_manager = NS(graphs={descriptor: graph})
    first = probe._runner_graph_state(runner)
    assert first["model_runner_version"] == 2
    assert first["model_runner"] == "vllm_ascend.worker.v2.model_runner.NPUModelRunner"
    assert first["decode"] == {"captures": 1, "replays": 0}
    replay = runner.cudagraph_manager.graphs[descriptor]
    assert replay is graph and type(replay) is graph_type
    assert replay.pool == graph.pool
    assert replay.replay() == "replayed"
    assert probe._runner_graph_state(runner)["decode"] == {"captures": 1, "replays": 1}
    assert runner.cudagraph_manager.graphs[descriptor] is replay
    replay.replay()
    original_replay.side_effect = RuntimeError("graph submission failed")
    with pytest.raises(RuntimeError, match="submission"):
        replay.replay()
    assert probe._runner_graph_state(runner)["decode"] == {"captures": 1, "replays": 2}
    assert first["decode"]["replays"] == 0
    assert original_replay.call_count == 3


def test_gs32_ep_v2_graph_state_cannot_report_eager_as_replay(probe):
    cls = type("NPUModelRunner", (), {"__module__": "vllm_ascend.worker.v2.model_runner"})
    runner = cls()
    runner.cudagraph_manager = NS(graphs={})
    assert probe._runner_graph_state(runner)["decode"] == {"captures": 0, "replays": 0}
    descriptor = type("Descriptor", (), {"num_tokens": 1})()
    runner.cudagraph_manager.graphs[descriptor] = NS(aclgraph=object())
    with pytest.raises(AssertionError, match="cannot verify replay"):
        probe._runner_graph_state(runner)


@pytest.mark.parametrize("use_v2", [False, True])
def test_gs32_ep_graph_state_classifies_configured_decode_buckets(probe, use_v2):
    module = "vllm_ascend.worker.v2.model_runner" if use_v2 else "vllm_ascend.worker.model_runner_v1"
    parent = type("NPUModelRunner", (), {"__module__": module})
    runner = type("RunnerWithExtension", (parent,), {})()
    runner.compilation_config = NS(cudagraph_capture_sizes=[1, 4, 32, 64])
    # Different descriptors can share a token bucket; both captures and actual
    # replays must be summed. B32 is decode here, despite the old I32 probe.
    tokens = (1, 4, 32, 32, 64, 128)
    if use_v2:
        graphs = {type("Descriptor", (), {"num_tokens": size})(): NS(replay=Mock()) for size in tokens}
        runner.cudagraph_manager = NS(graphs=graphs)
        initial = probe._runner_graph_state(runner)
        assert initial["decode"] == {"captures": 5, "replays": 0}
        for count, graph in enumerate(graphs.values(), start=1):
            for _ in range(count):
                graph.replay()
    else:
        runner.model = NS(
            concrete_aclgraph_entries={
                index: NS(aclgraph=object(), batch_descriptor=NS(num_tokens=size), num_replays=index + 1)
                for index, size in enumerate(tokens)
            }
        )
    state = probe._runner_graph_state(runner)
    assert state["model_runner_version"] == (2 if use_v2 else 1)
    assert state["decode"] == {"captures": 5, "replays": 15}
    assert state["prefill"] == {"captures": 0, "replays": 0}
    assert state["other"] == {"captures": 1, "replays": 6}
    assert state["decode_buckets"] == {
        "1": {"captures": 1, "replays": 1},
        "4": {"captures": 1, "replays": 2},
        "32": {"captures": 2, "replays": 7},
        "64": {"captures": 1, "replays": 5},
    }
    assert probe._runner_graph_state(runner) == state
    if use_v2:
        assert initial["decode"]["replays"] == 0
        assert all(runner.cudagraph_manager.graphs[key] is graph for key, graph in graphs.items())


def test_gs32_ep_v2_unconfigured_b32_does_not_count_as_decode_or_prefill(probe):
    cls = type("NPUModelRunner", (), {"__module__": "vllm_ascend.worker.v2.model_runner"})
    runner = cls()
    runner.compilation_config = NS(cudagraph_capture_sizes=[1, 2, 4])
    descriptor = type("Descriptor", (), {"num_tokens": 32})()
    runner.cudagraph_manager = NS(graphs={descriptor: NS(replay=Mock())})
    state = probe._runner_graph_state(runner)
    assert state["other"] == {"captures": 1, "replays": 0}
    assert state["prefill"] == state["decode"] == {"captures": 0, "replays": 0}
    assert state["decode_buckets"] == {}


@pytest.mark.parametrize("execution", ["fused", "full_graph"])
@pytest.mark.parametrize("ep_size", [2, 4])
def test_gs32_batch_probe_options_satisfy_runtime_contract(modules, probe, execution, ep_size):
    options = probe.batch_runner_options(execution, Path("stage.so").resolve(), Path("kernel.o").resolve(), ep_size)
    assert options["max_num_seqs"] == 64
    assert options["max_num_batched_tokens"] >= options["max_num_seqs"] * probe.BATCH_SMOKE_INPUT_TOKENS
    assert options["max_model_len"] >= probe.BATCH_SMOKE_INPUT_TOKENS + max(
        probe.OUTPUT_TOKENS, *probe.BATCH_SMOKE_MIXED_OUTPUT_TOKENS
    )
    assert options["tensor_parallel_size"] == ep_size and options["data_parallel_size"] == 1
    assert options["enable_expert_parallel"] and options["distributed_executor_backend"] == "mp"
    assert not options["async_scheduling"] and not options["enable_prefix_caching"]
    assert options["compilation_config"]["mode"] == 0
    if execution == "full_graph":
        assert not options["enforce_eager"]
        assert options["compilation_config"]["cudagraph_mode"] == "FULL_DECODE_ONLY"
        assert options["compilation_config"]["cudagraph_capture_sizes"] == [1, 2, 4, 8, 16, 32, 64]
        assert options["compilation_config"]["max_cudagraph_capture_size"] == 64
    else:
        assert options["enforce_eager"]
        assert options["compilation_config"]["cudagraph_mode"] == "NONE"

    config = long_graph_config()
    config.model_config.hf_config.n_routed_experts = 256
    config.use_v2_model_runner = True
    config.additional_config = options["additional_config"]
    config.compilation_config = NS(**options["compilation_config"])
    for field in ("max_model_len", "enforce_eager", "quantization"):
        setattr(config.model_config, field, options[field])
    for field in ("enable_chunked_prefill", "max_num_batched_tokens", "max_num_seqs", "async_scheduling"):
        setattr(config.scheduler_config, field, options[field])
    for field in ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size", "enable_expert_parallel"):
        setattr(config.parallel_config, field, options[field])
    config.load_config.load_format = options["load_format"]
    config.cache_config.enable_prefix_caching = options["enable_prefix_caching"]
    assert modules.config.validate_vq2a8_config(config) == options["additional_config"]["vq2a8"]
    # Building the batch probe must leave independent B1 invocations unchanged.
    assert probe.runner_options("full_graph", "stage.so", "kernel.o", 2)["compilation_config"][
        "cudagraph_capture_sizes"
    ] == [1]


@pytest.fixture(params=[2, 4], ids=["ep2", "ep4"])
def batch_graph_states(probe, request):
    return [
        {
            "rank": rank,
            "layers": [{"layer": 0, "expert_ids": [rank]}],
            "model_runner_version": 2,
            "validity_owners": 1,
            "decode": {"captures": 7, "replays": 70},
            "decode_buckets": {str(size): {"captures": 1, "replays": 10} for size in probe.BATCH_SMOKE_CAPTURE_SIZES},
            "prefill": {"captures": 0, "replays": 0},
            "other": {"captures": 0, "replays": 0},
        }
        for rank in range(request.param)
    ]


@pytest.mark.parametrize("batch,bucket", [(1, 1), (3, 4), (5, 8), (7, 8), (31, 32), (63, 64), (64, 64)])
def test_gs32_batch_progress_requires_smallest_covering_bucket(probe, batch_graph_states, batch, bucket):
    after = deepcopy(batch_graph_states)
    for state in after:
        state["decode"]["replays"] += probe.OUTPUT_TOKENS - 1
        state["decode_buckets"][str(bucket)]["replays"] += probe.OUTPUT_TOKENS - 1
    probe.check_batch_graph_progress("full_graph", batch_graph_states, after, 1, batch)


@pytest.mark.parametrize(
    "fault",
    [
        "runner",
        "rank",
        "layers",
        "owners",
        "capture",
        "missing_bucket",
        "zero_capture",
        "short_replay",
        "wrong_bucket",
        "prefill",
        "other",
    ],
)
def test_gs32_batch_progress_rejects_invalid_rank_or_bucket_evidence(probe, batch_graph_states, fault):
    after = deepcopy(batch_graph_states)
    for state in after:
        state["decode"]["replays"] += probe.OUTPUT_TOKENS - 1
        state["decode_buckets"]["8"]["replays"] += probe.OUTPUT_TOKENS - 1
    failed = after[-1]
    if fault == "runner":
        failed["model_runner_version"] = 1
    elif fault == "rank":
        failed["rank"] = 0
    elif fault == "layers":
        failed["layers"] = []
    elif fault == "owners":
        failed["validity_owners"] = 0
    elif fault == "capture":
        failed["decode_buckets"]["8"]["captures"] += 1
    elif fault == "missing_bucket":
        del failed["decode_buckets"]["64"]
    elif fault == "zero_capture":
        batch_graph_states[-1]["decode_buckets"]["64"]["captures"] = 0
        failed["decode_buckets"]["64"]["captures"] = 0
    elif fault == "short_replay":
        failed["decode_buckets"]["8"]["replays"] -= 1
    elif fault == "wrong_bucket":
        # Aggregate replay growth is insufficient: B5 must use its B8 bucket.
        failed["decode_buckets"]["8"]["replays"] -= probe.OUTPUT_TOKENS - 1
        failed["decode_buckets"]["64"]["replays"] += probe.OUTPUT_TOKENS - 1
    else:
        failed[fault]["replays"] = 1
    with pytest.raises(AssertionError):
        probe.check_batch_graph_progress("full_graph", batch_graph_states, after, 1, 5)


@pytest.mark.parametrize("missing_bucket", [None, "8", "4"])
def test_gs32_batch_progress_requires_live_batch_shrink_on_every_rank(probe, batch_graph_states, missing_bucket):
    after = deepcopy(batch_graph_states)
    for state in after:
        state["decode_buckets"]["8"]["replays"] += 2
        state["decode_buckets"]["4"]["replays"] += 1
        state["decode"]["replays"] += 3
    if missing_bucket is not None:
        after[-1]["decode_buckets"][missing_bucket]["replays"] = batch_graph_states[-1]["decode_buckets"][
            missing_bucket
        ]["replays"]
        with pytest.raises(AssertionError, match=f"Rank {len(batch_graph_states) - 1}"):
            probe.check_batch_graph_progress("full_graph", batch_graph_states, after, 1, 8, mixed_lengths=True)
    else:
        probe.check_batch_graph_progress("full_graph", batch_graph_states, after, 1, 8, mixed_lengths=True)


@pytest.mark.parametrize("fault", [None, "capture", "replay", "owners"])
def test_gs32_batch_eager_full_ffn_checks_validity_without_graph_execution(probe, batch_graph_states, fault):
    for state in batch_graph_states:
        state.update(decode={"captures": 0, "replays": 0}, decode_buckets={}, validity_owners=1)
    after = deepcopy(batch_graph_states)
    if fault == "owners":
        after[-1]["validity_owners"] = 0
    elif fault:
        after[-1]["decode"]["captures" if fault == "capture" else "replays"] = 1
    if fault:
        with pytest.raises(AssertionError):
            probe.check_batch_graph_progress("fused", batch_graph_states, after, 1, 64)
    else:
        probe.check_batch_graph_progress("fused", batch_graph_states, after, 1, 64)


def test_gs32_batch_progress_rejects_missing_worker(probe, batch_graph_states):
    after = deepcopy(batch_graph_states[:1])
    after[0]["decode_buckets"]["1"]["replays"] += probe.OUTPUT_TOKENS - 1
    with pytest.raises(ValueError, match="shorter"):
        probe.check_batch_graph_progress("full_graph", batch_graph_states, after, 1, 1)


@pytest.mark.parametrize("ep_size", [2, 4])
@pytest.mark.parametrize("fault", [None, "wrong_ep", "missing_wave", "different_tokens"])
def test_gs32_batch_smoke_runs_selected_ep_and_checks_child_reports(probe, tmp_path, monkeypatch, ep_size, fault):
    library, binary = tmp_path / "stage.so", tmp_path / "kernel.o"
    library.touch()
    binary.touch()
    options = {
        "--vq2a8-model": str(tmp_path),
        "--vq2a8-gs32-library": str(library),
        "--vq2a8-gs32-binary": str(binary),
    }
    request = NS(config=NS(getoption=lambda name, default=None: options.get(name, default)))
    executions = []

    def execute(command, environment, log, timeout):
        assert command[command.index("--ep-size") + 1] == str(ep_size)
        assert "--batch-smoke" in command and environment["VLLM_USE_V2_MODEL_RUNNER"] == "1"
        execution = command[command.index("--execution") + 1]
        executions.append(execution)
        budgets = [(probe.OUTPUT_TOKENS,) * batch for batch in probe.BATCH_SMOKE_SIZES]
        budgets.append(probe.BATCH_SMOKE_MIXED_OUTPUT_TOKENS)
        waves = [
            {
                "batch": len(lengths),
                "output_lengths": list(lengths),
                "requests": [
                    {
                        "prompt_token_ids": [1, 2],
                        "token_ids": [3] * length,
                        "logprobs": [{"3": -0.25} for _ in range(length)],
                    }
                    for length in lengths
                ],
            }
            for lengths in budgets
        ]
        report = {"ep_size": ep_size, "execution": execution, "fusion": "full_ffn", "waves": waves}
        if execution == "full_graph":
            if fault == "wrong_ep":
                report["ep_size"] = 8
            elif fault == "missing_wave":
                waves.pop()
            elif fault == "different_tokens":
                waves[0]["requests"][0]["token_ids"][0] = 4
        Path(command[command.index("--output") + 1]).write_text(json.dumps(report), encoding="utf-8")
        return 0

    monkeypatch.setattr(probe, "execute", execute)
    smoke = getattr(probe, f"test_gs32_ep{ep_size}_decode_batch_smoke")
    if fault:
        with pytest.raises(AssertionError):
            smoke(request, tmp_path)
    else:
        smoke(request, tmp_path)
    assert executions == ["fused", "full_graph"]


@pytest.mark.parametrize("use_v2", [False, True])
@pytest.mark.parametrize("world", [2, 4, 8])
def test_gs32_scheme_uses_runner_owned_context_and_global_to_local_map(modules, monkeypatch, use_v2, world):
    """V2 must read its upstream map and additional kwargs, not stale V1 state."""
    context = NS(moe_comm_method=Mock(), additional_kwargs={"moe_comm_method": Mock()})
    namespace = {
        "Any": object,
        "torch": torch,
        "get_forward_context": lambda: context,
        "envs_vllm": NS(VLLM_USE_V2_MODEL_RUNNER=use_v2),
    }
    source = ast.parse((REPO / "vllm_ascend/ascend_forward_context.py").read_text(encoding="utf-8"))
    proxy = next(
        node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "_ExtraForwardContextProxy"
    )
    exec(compile(ast.Module(body=[proxy], type_ignores=[]), "ascend_forward_context.py", "exec"), namespace)
    source = ast.parse((REPO / "vllm_ascend/ops/fused_moe/routed_experts.py").read_text(encoding="utf-8"))
    owner = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "AscendRoutedExperts")
    getter = next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == "ascend_expert_map")
    exec(compile(ast.Module(body=[getter], type_ignores=[]), "routed_experts.py", "exec"), namespace)
    layer = type("ExpertMapOwner", (), {"ascend_expert_map": namespace["ascend_expert_map"]})()
    layer._use_v2_model_runner = use_v2
    # On the last EP rank only this contiguous global range is resident.
    local_experts = 256 // world
    expected_map = torch.full((256,), -1, dtype=torch.int32)
    expected_map[-local_experts:] = torch.arange(local_experts, dtype=torch.int32)
    layer.expert_map = expected_map if use_v2 else torch.full_like(expected_map, -1)
    layer._ascend_expert_map = torch.full_like(expected_map, -1) if use_v2 else expected_map
    layer.vq2a8_full_ffn_bank = {
        "expert_ids": (0,),
        "fields": {name: torch.empty(0, dtype=torch.uint8) for name in ("gate_weight", "down_weight")},
    }
    layer.vq2a8_weights = object()
    layer.global_redundant_expert_num = 0
    layer.ascend_mc2_mask = layer.ascend_pertoken_scale = None
    layer.apply_router_weight_on_input = False
    layer.activation = "silu"
    monkeypatch.setattr(modules.method, "_EXTRA_CTX", namespace["_ExtraForwardContextProxy"]())
    scheme = modules.method.AscendGS32FusedMoEScheme(source_layer_spec(modules))
    hidden = torch.randn(2, 128, dtype=torch.bfloat16)
    topk_ids = torch.tensor([[0, 255], [255 - local_experts, 256 - local_experts]], dtype=torch.int32)
    topk_weights = torch.tensor([[0.123456, 0.876544], [0.333333, 0.666667]], dtype=torch.float32)

    result = scheme.apply(layer, hidden, topk_weights, topk_ids)

    selected = context.additional_kwargs["moe_comm_method"] if use_v2 else context.moe_comm_method
    other = context.moe_comm_method if use_v2 else context.additional_kwargs["moe_comm_method"]
    assert result is selected.fused_experts.return_value
    other.fused_experts.assert_not_called()
    selected.fused_experts.assert_called_once_with(
        fused_experts_input=modules.builder.build_fused_experts_input.return_value, quant_method=scheme
    )
    payload = modules.builder.build_fused_experts_input.call_args.kwargs
    assert payload["expert_map"] is expected_map
    assert payload["hidden_states"] is hidden and payload["topk_weights"] is topk_weights
    assert payload["topk_ids"] is topk_ids and payload["vq2a8"] is layer.vq2a8_weights
    assert payload["pertoken_scale"] is None and payload["dynamic_eplb"] is False
    assert payload["quant_type"] == modules.method.QuantType.VQ2A8
    assert torch.equal(expected_map[topk_ids], torch.tensor([[-1, local_experts - 1], [-1, 0]], dtype=torch.int32))


@pytest.mark.parametrize("execution", ["fused", "full_graph"])
def test_mrv2_reuses_source_v3_eager_and_decode_graph_configuration(modules, execution):
    config = long_graph_config() if execution == "full_graph" else serving_config()
    config.additional_config["vq2a8"] = {
        "gs32_library": str(Path("stage.so").resolve()),
        "gs32_binary": str(Path("stage.o").resolve()),
        "gs32_execution": execution,
        "gs32_backend": "native",
        "gs32_fusion": "full_ffn",
    }
    config.use_v2_model_runner = True
    assert modules.config.validate_vq2a8_config(config) == config.additional_config["vq2a8"]
