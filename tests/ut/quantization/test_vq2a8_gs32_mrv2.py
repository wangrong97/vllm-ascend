# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MRV2 GS32 ownership and replay contracts on real CPU tensors; no NPU engine."""
# ruff: noqa: F401, F811  # Imported pytest fixtures are injected by name.

import ast
import importlib.util
from contextlib import contextmanager
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
    state = NS(events=[], values=[True, True], skipped=set(), capturing=False, version=True, forward_error=None)
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

        def prepare_dummy_attn(self, batch):
            return "upstream_dummy"

    scope = {
        "__name__": __name__,
        "torch": torch,
        "GPUModelRunner": Parent,
        "CUDAGraphMode": NS(NONE=0),
        "breakable_cudagraph": NS(is_breakable_cudagraph_enabled=lambda: False),
        "vllm_version_is": lambda _: state.version,
        "_start_profiling_chunk_timing": lambda *_: None,
        "_finish_profiling_chunk_timing": lambda *_: None,
        "lmhead_tp_enable": lambda: False,
        "contextmanager": contextmanager,
        "AscendAttentionState": NS(PrefillNoCache="prefill"),
        "update_cos_sin": lambda positions: state.events.append(("rope", positions.clone())),
    }
    names = (
        "load_model",
        "_check_quantization_validity",
        "execute_model",
        "sample",
        "_gs32_prefill_warmup",
        "prepare_dummy_attn",
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


@pytest.mark.parametrize("version", [False, True])
@pytest.mark.parametrize("failure", ["flag", "missed_write"])
def test_v2_gs32_generation_checks_after_logits_before_sampling_and_recovers(v2_quantization_runner, version, failure):
    h = v2_quantization_runner
    h.state.version = version
    runner = h.runner
    pointers = [owner.validity_buffer.data_ptr() for owner in h.owners]
    for valid in (True, False, True):
        h.state.events.clear()
        h.state.values = [True, valid]
        h.state.skipped = {1} if not valid and failure == "missed_write" else set()
        runner.execute_model(NS(total_num_scheduled_tokens=1))
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
    assert ("context_len" in h.state.execute_kwargs) is (not version)


@pytest.mark.parametrize("mode", ["dummy", "profile", "pp"])
def test_v2_gs32_no_sampler_paths_check_before_return(v2_quantization_runner, mode):
    h = v2_quantization_runner
    h.state.values[1] = False
    h.runner.is_last_pp_rank = mode != "pp"
    with pytest.raises(RuntimeError, match="failed validity"):
        h.runner.execute_model(
            NS(total_num_scheduled_tokens=1), dummy_run=mode == "dummy", is_profile=mode == "profile"
        )
    assert h.runner.execute_model_state is None
    assert not h.runner._quantization_validity_pending


def test_v2_gs32_empty_step_and_unconsumed_step_status(v2_quantization_runner):
    h = v2_quantization_runner
    h.runner.execute_model(NS(total_num_scheduled_tokens=0))
    assert h.state.events == []
    h.state.values[1] = False
    h.runner.execute_model(NS(total_num_scheduled_tokens=1))
    with pytest.raises(RuntimeError, match="failed validity"):
        h.runner.execute_model(NS(total_num_scheduled_tokens=0))
    assert h.runner.execute_model_state is None


@pytest.mark.parametrize("version", [False, True])
def test_v2_gs32_failed_forward_discards_partial_state_and_recovers(v2_quantization_runner, version):
    h = v2_quantization_runner
    h.state.version = version
    runner = h.runner
    # Emulate an interrupted submission after the parent has installed state.
    # Preserve the original exception and make a later sample_tokens a no-op.
    failure = RuntimeError("model submission failed")
    h.state.forward_error = failure
    with pytest.raises(RuntimeError, match="model submission failed") as raised:
        runner.execute_model(NS(total_num_scheduled_tokens=1))
    assert raised.value is failure
    assert runner.execute_model_state is None
    assert not runner._quantization_validity_pending
    assert "sample" not in h.state.events

    h.state.forward_error = None
    h.state.events.clear()
    runner.execute_model(NS(total_num_scheduled_tokens=1))
    runner.sample(torch.ones(1, 4), h.batch, None)
    assert h.state.events[-3:] == ["logits", "read", "sample"]
    assert not runner._quantization_validity_pending


def test_v2_other_quantization_keeps_upstream_sampling(v2_quantization_runner):
    h = v2_quantization_runner
    h.runner.model_config.quantization = "ascend"
    h.runner.load_model()
    h.state.events.clear()
    h.runner.execute_model(NS(total_num_scheduled_tokens=1))
    h.runner.sample(torch.ones(1, 4), h.batch, None)
    assert h.state.events == ["forward", "upstream_sample"]
    assert h.runner._quantization_validity is None


def test_v2_gs32_startup_prefill_forces_eager_without_leaving_pending_status(v2_quantization_runner):
    h = v2_quantization_runner
    h.runner._gs32_prefill_warmup_active = True
    h.runner.execute_model(NS(total_num_scheduled_tokens=1), dummy_run=True)
    assert h.state.execute_kwargs["is_profile"] is True
    assert not h.runner._quantization_validity_pending


def test_v2_gs32_capture_wrapper_has_no_host_check_in_capture(v2_quantization_runner):
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

    def forward():
        for owner, valid in zip(h.owners, h.state.values):
            owner.validity_buffer.fill_(valid)
        return torch.ones(1)

    wrapper = scope["ModelWithContext"](forward, model_runner=h.runner)
    for capturing in (False, True, False):
        h.state.capturing = capturing
        h.state.values[1] = False
        h.state.events.clear()
        if capturing:
            wrapper()
            assert h.state.events == []
        else:
            with pytest.raises(RuntimeError, match="failed validity"):
                wrapper()
    h.state.values[1] = True
    wrapper()


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
            assert batch.attn_state == "prefill" and batch.is_prefilling_np.all()
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
    assert runner.prepare_dummy_attn(NS()) == "upstream_dummy"


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
    graph = NS(replay=Mock(return_value="replayed"), pool="retained")
    runner.cudagraph_manager = NS(graphs={descriptor: graph})
    first = probe._runner_graph_state(runner)
    assert first["model_runner_version"] == 2
    assert first["model_runner"] == "vllm_ascend.worker.v2.model_runner.NPUModelRunner"
    assert first["decode"] == {"captures": 1, "replays": 0}
    replay = runner.cudagraph_manager.graphs[descriptor]
    assert replay.pool == graph.pool
    assert replay.replay() == "replayed"
    assert probe._runner_graph_state(runner)["decode"] == {"captures": 1, "replays": 1}
    assert runner.cudagraph_manager.graphs[descriptor] is replay
    replay.replay()
    graph.replay.side_effect = RuntimeError("graph submission failed")
    with pytest.raises(RuntimeError, match="submission"):
        replay.replay()
    assert probe._runner_graph_state(runner)["decode"] == {"captures": 1, "replays": 2}
    assert first["decode"]["replays"] == 0
    assert graph.replay.call_count == 3


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
    layer.w13_packed_zn = layer.w2_packed_zn = torch.empty(0, dtype=torch.uint8)
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
        fused_experts_input=modules.builder.build_fused_experts_input.return_value
    )
    payload = modules.builder.build_fused_experts_input.call_args.kwargs
    assert payload["expert_map"] is expected_map
    assert payload["hidden_states"] is hidden and payload["topk_weights"] is topk_weights
    assert payload["topk_ids"] is topk_ids and payload["vq2a8"] is layer.vq2a8_weights
    assert payload["pertoken_scale"] is None and payload["dynamic_eplb"] is False
    assert payload["quant_type"] == modules.method.QuantType.VQ2A8
    assert torch.equal(expected_map[topk_ids], torch.tensor([[-1, local_experts - 1], [-1, 0]], dtype=torch.int32))


@pytest.mark.parametrize("execution", ["serial", "batched", "fused", "full_graph"])
def test_mrv2_reuses_source_v3_eager_and_decode_graph_configuration(modules, execution):
    config = long_graph_config() if execution == "full_graph" else serving_config()
    config.additional_config["vq2a8"] = {
        "gs32_library": str(Path("stage.so").resolve()),
        "gs32_binary": str(Path("stage.o").resolve()),
        "gs32_execution": execution,
    }
    config.use_v2_model_runner = True
    assert modules.config.validate_vq2a8_config(config) == config.additional_config["vq2a8"]
