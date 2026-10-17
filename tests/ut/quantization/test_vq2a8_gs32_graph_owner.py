# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GS32 full-graph owner lifecycle on CPU; graph submission alone is emulated."""

import ast
import dataclasses
import importlib.util
import math
import sys
import types
import weakref
from contextlib import ExitStack, contextmanager, nullcontext
from enum import Enum
from functools import partial
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import pytest
import torch

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture
def quantization_runner(monkeypatch):
    """Execute real runner/wrapper methods; emulate only graph submission."""

    class GraphMode(Enum):
        NONE = 0
        FULL = 1
        PIECEWISE = 2

    context = NS(
        capturing=False, cudagraph_runtime_mode=GraphMode.NONE, batch_descriptor=1, flash_comm_v1_enabled=False
    )
    current_stream = NS(synchronize=Mock())
    internal_stream = NS(synchronize=Mock())
    state = NS(
        stream=current_stream,
        capture=None,
        replay_writes=[],
        graph_kwargs=[],
        bool_reads=0,
        events=[],
        scopes=[],
        model_kwargs=[],
        logits=torch.ones(1, 8),
        logits_error=None,
        is_last_rank=True,
    )

    class Graph:
        def __init__(self):
            self.writes = []

        def replay(self):
            for write in self.writes:
                write()

    @contextmanager
    def graph(graph, **kwargs):
        state.graph_kwargs.append(kwargs)
        previous = state.stream
        state.stream = kwargs.get("stream", internal_stream)
        state.capture = graph
        try:
            yield
        finally:
            state.capture = None
            state.stream = previous

    monkeypatch.setattr(
        torch, "npu", NS(NPUGraph=Graph, graph=graph, current_stream=lambda: state.stream), raising=False
    )
    offloader = types.ModuleType("vllm.model_executor.offloader.base")
    offloader.get_offloader = lambda: NS(sync_prev_onload=lambda: None, join_after_forward=lambda: None)
    monkeypatch.setitem(sys.modules, offloader.__name__, offloader)
    communication = types.ModuleType("vllm_ascend.ops.fused_moe.moe_comm_method")
    communication.get_moe_comm_method = lambda _: None
    monkeypatch.setitem(sys.modules, communication.__name__, communication)

    @contextmanager
    def set_forward_context(**kwargs):
        context.cudagraph_runtime_mode = kwargs["cudagraph_runtime_mode"]
        context.batch_descriptor = kwargs["batch_descriptor"]
        # Deliberately retain capturing; the real Ascend context must reset it.
        yield

    original_bool = torch.Tensor.__bool__

    def read_bool(value):
        assert not context.capturing, "A capture must not read a tensor on the host"
        state.bool_reads += 1
        state.events.append("validity")
        return original_bool(value)

    @contextmanager
    def record_scope(name):
        state.scopes.append(name)
        yield

    monkeypatch.setattr(torch.Tensor, "__bool__", read_bool)
    scope = {
        "__name__": __name__,
        "torch": torch,
        "math": math,
        "dataclasses": dataclasses,
        "partial": partial,
        "CUDAGraphMode": GraphMode,
        "get_forward_context": lambda: context,
        "IntermediateTensors": type("IntermediateTensors", (), {}),
        "_EXTRA_CTX": NS(is_draft_model=False),
        "envs": NS(VLLM_LOGGING_LEVEL="INFO"),
        "current_platform": NS(get_global_graph_pool=lambda: "pool"),
        "CUDAGraphOptions": lambda: NS(debug_log_enable=False, gc_disable=False, weak_ref_output=False),
        "_acl_graph_wrappers": weakref.WeakSet(),
        "ExitStack": ExitStack,
        "patch": patch,
        "validate_cudagraph_capturing_enabled": lambda: None,
        "weak_ref_tensors": lambda output: output,
        "weak_ref_workspaces": lambda _: None,
        "_graph_params": None,
        "_draft_graph_params": None,
        "_draft_graph_prefill_params": None,
        "compilation_counter": NS(num_cudagraph_captured=0),
        "logger": NS(info_once=lambda *_: None),
        "_is_stream_resource_capture_error": lambda _: False,
        "_is_old_hdk_capture_error": lambda _: False,
        "contextmanager": contextmanager,
        "set_forward_context": set_forward_context,
        "set_current_vllm_config": lambda _: nullcontext(),
        "_is_decode_only_node": lambda _: False,
        "use_cann_megamoe": lambda _: False,
        "select_moe_comm_method": lambda *_: "allgather",
        "get_tensor_model_parallel_world_size": lambda: 1,
        "is_moe_model": lambda _: True,
        "enable_sp": lambda _: False,
        "flashcomm2_enable": lambda: False,
        "has_layer_idx": lambda _: False,
        "get_dp_group": lambda: NS(world_size=1),
        "get_mc2_mask": lambda: None,
        "breakable_cudagraph": NS(is_breakable_cudagraph_enabled=lambda: False),
        "record_function_or_nullcontext": record_scope,
        "get_pp_group": lambda: NS(is_last_rank=state.is_last_rank),
        "ExecuteModelState": lambda *args: NS(logits=args[1]),
    }
    scope.update(
        dict.fromkeys(
            (
                "input_ids",
                "positions",
                "intermediate_tensors",
                "inputs_embeds",
                "scheduler_output",
                "spec_decode_metadata",
                "spec_decode_common_attn_metadata",
                "attn_metadata",
                "ec_connector_output",
                "cudagraph_stats",
                "batch_desc",
                "kv_connector_output",
            )
        )
    )
    scope.update(num_tokens_padded=1, model_kwargs={}, logits_indices=torch.tensor([0]))
    methods = [
        _method_ast("vllm_ascend/worker/model_runner_v1.py", "NPUModelRunner", name)
        for name in (
            "_initialize_quantization_validity",
            "_model_forward",
            "_check_quantization_validity",
            "_compute_logits",
        )
    ]
    # Exercise the real generation handoff without constructing the distributed
    # worker, scheduler, KV cache or NPU input preparation on this CPU harness.
    execute = _method_ast("vllm_ascend/worker/model_runner_v1.py", "NPUModelRunner", "execute_model")

    def assigned_name(node):
        return node.targets[0].id if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) else None

    defer = next(node for node in execute.body if assigned_name(node) == "defer_quantization_validity")
    forward = next(
        node
        for node in execute.body
        if isinstance(node, ast.With) and any(assigned_name(child) == "hidden_states" for child in node.body)
    )
    forward_statements = [
        node for node in forward.body if assigned_name(node) in ("hidden_states", "check_quantization_validity")
    ]
    post_process = next(
        node
        for node in execute.body
        if isinstance(node, ast.With)
        and isinstance(node.items[0].context_expr, ast.Call)
        and node.items[0].context_expr.args
        and isinstance(node.items[0].context_expr.args[0], ast.Constant)
        and node.items[0].context_expr.args[0].value == "post process"
    )
    generation = ast.parse("def execute_generation(self): pass").body[0]
    generation.body = [defer, *forward_statements, post_process]
    methods.append(generation)
    runner_class = ast.ClassDef(name="Runner", bases=[], keywords=[], body=methods, decorator_list=[])
    wrapper_source = ast.parse((REPO / "vllm_ascend/compilation/acl_graph.py").read_text(encoding="utf-8"))
    classes = [
        node
        for node in wrapper_source.body
        if isinstance(node, ast.ClassDef) and node.name in ("ACLGraphEntry", "ACLGraphWrapper")
    ]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    context_source = ast.parse((REPO / "vllm_ascend/ascend_forward_context.py").read_text(encoding="utf-8"))
    context_function = next(
        node
        for node in context_source.body
        if isinstance(node, ast.FunctionDef) and node.name == "set_ascend_forward_context"
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[future, runner_class, *classes, context_function], type_ignores=[])
    )
    exec(compile(module, "quantization_runner_lifecycle", "exec"), scope)
    owners = [NS(validity_buffer=torch.zeros((), dtype=torch.bool), defer_validity=False) for _ in range(2)]
    layers = [NS(vq2a8_layer_index=index, vq2a8_weights=owner) for index, owner in enumerate(owners)]
    status = [True, True]
    skipped = set()
    calls = []
    output = torch.ones(1, 4)

    class Model:
        def modules(self):
            return iter(layers)

        def __call__(self, **kwargs):
            calls.append(state.stream)
            state.model_kwargs.append(kwargs)
            state.events.append("forward")
            if state.capture is not None:
                assert state.stream is internal_stream, "Default wrapper capture behavior changed"
            for index, owner in enumerate(owners):
                if index in skipped:
                    continue

                # Stand in only for the device status write, never runner logic.
                def write(index=index, owner=owner):
                    owner.validity_buffer.fill_(status[index])

                write()
                if state.capture is not None:
                    state.capture.writes.append(write)
            return output

        def compute_logits(self, hidden_states):
            assert torch.equal(hidden_states, output)
            state.events.append("logits")
            if state.logits_error is not None:
                raise state.logits_error
            return state.logits

    runner = scope["Runner"]()
    runner.model = Model()
    runner.model_config = NS(quantization="vq2a8")
    runner.vllm_config = NS(
        quant_config=NS(is_gs32=True, layers={0: None, 1: None}),
        compilation_config=NS(),
        additional_config={"vq2a8": {"gs32_execution": "full_graph"}},
    )
    runner.device = torch.device("cpu")
    runner.enable_enpu = False
    runner.compilation_config = NS(cudagraph_mode=GraphMode.FULL)
    runner._update_full_graph_params_if_needed = Mock()
    runner.is_pooling_model = False
    runner.broadcast_pp_output = False
    runner.use_aux_hidden_state_outputs = False
    runner.execute_model_state = None
    return NS(
        runner=runner,
        breakable=scope["breakable_cudagraph"],
        owners=owners,
        layers=layers,
        status=status,
        skipped=skipped,
        context=context,
        state=state,
        mode=GraphMode,
        wrapper=scope["ACLGraphWrapper"],
        calls=calls,
        stream=current_stream,
        output=output,
        forward_context=lambda mode: scope["set_ascend_forward_context"](
            None, runner.vllm_config, num_tokens=1, aclgraph_runtime_mode=mode, batch_descriptor=1
        ),
    )


def test_runner_rejects_breakable_capture_for_full_graph_owner(quantization_runner):
    harness = quantization_runner
    harness.breakable.is_breakable_cudagraph_enabled = lambda: True
    with pytest.raises(ValueError, match="VLLM_USE_BREAKABLE_CUDAGRAPH=0"):
        harness.runner._initialize_quantization_validity()
    harness.runner.compilation_config.cudagraph_mode = harness.mode.NONE
    harness.runner._initialize_quantization_validity()
    assert len(harness.runner._quantization_validity_buffers) == 2


@pytest.mark.parametrize("execution", ["full_graph"])
def test_runner_quantization_validity_binds_distinct_persistent_views(quantization_runner, execution):
    harness = quantization_runner
    runner = harness.runner
    runner.vllm_config.additional_config["vq2a8"]["gs32_execution"] = execution
    runner._initialize_quantization_validity()
    assert isinstance(runner._quantization_validity_buffers, tuple)
    assert len(runner._quantization_validity_buffers) == 2
    assert len({flag.data_ptr() for flag in runner._quantization_validity_buffers}) == 2
    for owner, flag in zip(harness.owners, runner._quantization_validity_buffers):
        assert owner.validity_buffer is flag and owner.defer_validity
        assert flag.untyped_storage().data_ptr() == runner._quantization_validity.untyped_storage().data_ptr()


@pytest.mark.parametrize("invalid", ["missing", "duplicate", "shared", "dtype", "shape", "device"])
def test_runner_quantization_validity_rejects_incomplete_or_aliased_inventory(quantization_runner, invalid):
    harness = quantization_runner
    if invalid == "missing":
        harness.layers.pop()
    elif invalid == "duplicate":
        harness.layers[1].vq2a8_layer_index = 0
    elif invalid == "shared":
        harness.layers[1].vq2a8_weights = harness.owners[0]
    elif invalid == "dtype":
        harness.owners[1].validity_buffer = torch.zeros((), dtype=torch.int32)
    elif invalid == "device":
        harness.owners[1].validity_buffer = torch.zeros((), dtype=torch.bool, device="meta")
    else:
        harness.owners[1].validity_buffer = torch.zeros(1, dtype=torch.bool)
    with pytest.raises(RuntimeError, match="Quantization|quantization"):
        harness.runner._initialize_quantization_validity()
    assert not any(owner.defer_validity for owner in harness.owners)


def test_runner_eager_checks_all_layers_once_and_cannot_reuse_stale_success(quantization_runner):
    harness = quantization_runner
    runner = harness.runner
    runner._initialize_quantization_validity()
    addresses = tuple(flag.data_ptr() for flag in runner._quantization_validity_buffers)
    assert runner._model_forward(1) is harness.output
    assert harness.state.bool_reads == 1
    harness.status[1] = False
    with pytest.raises(RuntimeError, match="failed validity"):
        runner._model_forward(1)
    assert harness.state.bool_reads == 2
    harness.status[1] = True
    runner._model_forward(1)
    harness.skipped.add(1)
    with pytest.raises(RuntimeError, match="failed validity"):
        runner._model_forward(1)
    assert harness.state.bool_reads == 4
    assert addresses == tuple(flag.data_ptr() for flag in runner._quantization_validity_buffers)


def test_runner_capture_and_replay_check_fresh_status_without_recapturing(quantization_runner):
    harness = quantization_runner
    runner = harness.runner
    runner._initialize_quantization_validity()
    runner._model_forward(1)  # Eager warmup uses the caller's capture stream.
    wrapper = harness.wrapper(runner.model, runner.vllm_config, harness.mode.FULL)
    runner.model = wrapper
    harness.context.cudagraph_runtime_mode = harness.mode.FULL
    harness.state.bool_reads = 0
    runner._model_forward(1)
    assert harness.context.capturing and harness.state.bool_reads == 0
    assert harness.state.graph_kwargs == [{"pool": "pool"}]
    entry = wrapper.concrete_aclgraph_entries[1]
    assert entry.num_replays == 0
    pointers = tuple(flag.data_ptr() for flag in runner._quantization_validity_buffers)
    for _ in range(3):
        with harness.forward_context(harness.mode.FULL):
            assert not harness.context.capturing
            assert runner._model_forward(1) is harness.output
    assert entry.num_replays == 3 and harness.state.bool_reads == 3
    assert len(harness.calls) == 2  # Warmup and capture; replay skips Python model.
    harness.status[0] = False
    with pytest.raises(RuntimeError, match="failed validity"):
        runner._model_forward(1)
    assert entry.num_replays == 4 and harness.state.bool_reads == 4
    harness.status[0] = True
    entry.aclgraph.writes.pop()  # Missing graph write must not retain prior True.
    with pytest.raises(RuntimeError, match="failed validity"):
        runner._model_forward(1)
    assert pointers == tuple(flag.data_ptr() for flag in runner._quantization_validity_buffers)
    successful_replays = entry.num_replays
    entry.aclgraph.replay = Mock(side_effect=RuntimeError("Graph submission failed"))
    with pytest.raises(RuntimeError, match="Graph submission failed"):
        runner._model_forward(1)
    assert entry.num_replays == successful_replays
    with harness.forward_context(harness.mode.NONE):
        assert not harness.context.capturing
        assert runner._model_forward(1) is harness.output
    assert entry.num_replays == successful_replays


@pytest.mark.parametrize("graph_replay", [False, True])
def test_runner_generation_submits_logits_before_fresh_validity_check(quantization_runner, graph_replay):
    harness = quantization_runner
    runner = harness.runner
    runner._initialize_quantization_validity()
    mode = harness.mode.FULL if graph_replay else harness.mode.NONE
    if graph_replay:
        runner._model_forward(1)
        runner.model = harness.wrapper(runner.model, runner.vllm_config, harness.mode.FULL)
        with harness.forward_context(mode):
            runner._model_forward(1)
        assert harness.context.capturing

    def generate():
        with harness.forward_context(mode):
            hidden_states = runner._model_forward(1, defer_quantization_validity=True)
        # The actual generation path submits logits after leaving the forward
        # context. It must still inspect this forward's status before sampling.
        return runner._compute_logits(hidden_states, check_quantization_validity=True)

    harness.state.events.clear()
    before_reads = harness.state.bool_reads
    assert generate() is harness.state.logits
    assert harness.state.bool_reads == before_reads + 1
    assert harness.state.events[-2:] == ["logits", "validity"]
    assert harness.state.scopes[-2:] == ["compute logits", "quantization validity"]
    assert "defer_quantization_validity" not in harness.state.model_kwargs[-1]

    harness.status[1] = False
    sample = Mock()
    with pytest.raises(RuntimeError, match="failed validity"):
        sample(generate())
    sample.assert_not_called()
    assert harness.state.events[-2:] == ["logits", "validity"]

    harness.status[1] = True
    assert generate() is harness.state.logits
    if graph_replay:
        runner.model.concrete_aclgraph_entries[1].aclgraph.writes.pop()
    else:
        harness.skipped.add(1)
    with pytest.raises(RuntimeError, match="failed validity"):
        generate()


def test_runner_deferred_validity_checks_even_without_logits(quantization_runner):
    harness = quantization_runner
    runner = harness.runner
    runner._initialize_quantization_validity()
    harness.state.logits = None
    hidden_states = runner._model_forward(1, defer_quantization_validity=True)
    assert runner._compute_logits(hidden_states, check_quantization_validity=True) is None
    assert harness.state.bool_reads == 1
    harness.status[0] = False
    hidden_states = runner._model_forward(1, defer_quantization_validity=True)
    with pytest.raises(RuntimeError, match="failed validity"):
        runner._compute_logits(hidden_states, check_quantization_validity=True)


def test_runner_failed_logits_cannot_reach_sampling(quantization_runner):
    harness = quantization_runner
    runner = harness.runner
    runner._initialize_quantization_validity()
    hidden_states = runner._model_forward(1, defer_quantization_validity=True)
    harness.state.logits_error = RuntimeError("LM Head submission failed")
    sample = Mock()
    with pytest.raises(RuntimeError, match="LM Head submission failed"):
        sample(runner._compute_logits(hidden_states, check_quantization_validity=True))
    sample.assert_not_called()
    assert harness.state.bool_reads == 0


@pytest.mark.parametrize("valid", [False, True])
def test_runner_execute_checks_before_publishing_sampling_state(quantization_runner, valid):
    harness = quantization_runner
    runner = harness.runner
    runner._initialize_quantization_validity()
    harness.status[0] = valid
    if valid:
        runner.execute_generation()
        assert runner.execute_model_state.logits is harness.state.logits
    else:
        with pytest.raises(RuntimeError, match="failed validity"):
            runner.execute_generation()
        assert runner.execute_model_state is None
    assert harness.state.events == ["forward", "logits", "validity"]


@pytest.mark.parametrize("path", ["non_final_rank", "pooling", "broadcast"])
def test_runner_execute_other_paths_keep_forward_validity_boundary(quantization_runner, path):
    harness = quantization_runner
    runner = harness.runner
    runner._initialize_quantization_validity()
    if path == "non_final_rank":
        harness.state.is_last_rank = False
    elif path == "pooling":
        runner.is_pooling_model = True
    else:
        runner.broadcast_pp_output = True
    harness.status[0] = False
    with pytest.raises(RuntimeError, match="failed validity"):
        runner.execute_generation()
    assert harness.state.events == ["forward", "validity"]
    assert runner.execute_model_state is None


def test_runner_execute_does_not_add_private_kwargs_to_other_runners(quantization_runner):
    harness = quantization_runner
    runner = harness.runner
    runner.model_config.quantization = "w4a8"
    runner._initialize_quantization_validity()
    # The 310P subclass forwards unknown keywords directly into its model.
    runner._model_forward = Mock(return_value=harness.output)
    runner.execute_generation()
    assert runner._model_forward.call_args.kwargs == {}
    assert runner.execute_model_state.logits is harness.state.logits
    assert harness.state.bool_reads == 0


@pytest.mark.parametrize("quantization, options", [("w4a8", {}), ("vq2a8", {}), ("vq2a8", {"gs32_execution": "fused"})])
def test_runner_non_full_graph_and_wrapper_default_keep_existing_behavior(quantization_runner, quantization, options):
    harness = quantization_runner
    runner = harness.runner
    runner.model_config.quantization = quantization
    runner.vllm_config.additional_config = {"vq2a8": options}
    runner._initialize_quantization_validity()
    assert runner._quantization_validity is None and runner._quantization_validity_buffers == ()
    assert runner._model_forward(1) is harness.output
    assert harness.state.bool_reads == 0
    assert all(not owner.defer_validity for owner in harness.owners)
    wrapper = harness.wrapper(lambda: harness.output, runner.vllm_config, harness.mode.FULL)
    harness.context.cudagraph_runtime_mode = harness.mode.FULL
    assert wrapper() is harness.output
    assert harness.state.graph_kwargs == [{"pool": "pool"}]


def _method_ast(relative_path, class_name, method_name):
    module = ast.parse((REPO / relative_path).read_text(encoding="utf-8"))
    cls = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    return next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)


def test_gs32_owner_load_binds_before_graph_wrapper():
    load = _method_ast("vllm_ascend/worker/model_runner_v1.py", "NPUModelRunner", "load_model")
    calls = [node for node in ast.walk(load) if isinstance(node, ast.Call)]
    bind = next(
        node
        for node in calls
        if isinstance(node.func, ast.Attribute) and node.func.attr == "_initialize_quantization_validity"
    )
    wrapper = next(node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "ACLGraphWrapper")
    assert bind.lineno < wrapper.lineno


def test_gs32_owner_does_not_enable_on_another_checkpoint_format(quantization_runner):
    harness = quantization_runner
    harness.runner.vllm_config.quant_config.is_gs32 = False
    harness.runner._initialize_quantization_validity()
    assert harness.runner._quantization_validity is None
    assert not any(owner.defer_validity for owner in harness.owners)


@pytest.fixture
def full_graph_probe():
    path = REPO / "tests/e2e/nightly/test_vq2a8_gs32_full_graph.py"
    spec = importlib.util.spec_from_file_location("gs32_full_graph_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gs32_full_graph_probe_uses_named_rpc_and_public_decode_graph(full_graph_probe):
    probe = full_graph_probe
    state = {"captures": 1, "replays": 3, "validity_owners": 43}
    llm = NS(collective_rpc=Mock(return_value=[state]))
    assert probe.graph_state(llm) == state
    llm.collective_rpc.assert_called_once_with("vq2a8_gs32_full_graph_state")
    options = probe.runner_options("full_graph", "/native/bridge.so", "/native/kernel.o")
    assert options["enforce_eager"] is False
    assert options["max_num_seqs"] == 1
    assert options["compilation_config"]["cudagraph_mode"] == "FULL_DECODE_ONLY"
    assert options["compilation_config"]["cudagraph_capture_sizes"] == [1]
    assert options["compilation_config"]["cudagraph_num_of_warmups"] >= 1
    assert options["additional_config"]["vq2a8"]["gs32_execution"] == "full_graph"
    assert probe.runner_options("fused", "/native/bridge.so", "/native/kernel.o")["enforce_eager"] is True


@pytest.mark.parametrize("failure", ["no_graph", "recapture", "missing_replay", "missing_owner"])
def test_gs32_full_graph_probe_rejects_partial_or_fake_graph_acceptance(full_graph_probe, failure):
    probe = full_graph_probe
    previous = {"captures": 1, "replays": 3, "validity_owners": 43}
    current = {"captures": 1, "replays": 6, "validity_owners": 43}
    if failure == "no_graph":
        current["captures"] = 0
    elif failure == "recapture":
        current["captures"] = 2
    elif failure == "missing_replay":
        current["replays"] = 5
    else:
        current["validity_owners"] = 42
    with pytest.raises(AssertionError):
        probe.check_graph_progress("full_graph", previous, current, 43)


@pytest.mark.parametrize("failure", ["token", "logprob", "nonfinite", "top_set"])
def test_gs32_full_graph_probe_rejects_changed_generation(full_graph_probe, failure):
    probe = full_graph_probe
    expected = {"prompt_token_ids": [1] * 32, "token_ids": [2] * 4, "logprobs": [{"2": -0.1} for _ in range(4)]}
    actual = {"prompt_token_ids": [1] * 32, "token_ids": [2] * 4, "logprobs": [{"2": -0.1} for _ in range(4)]}
    probe.compare_requests(actual, expected)
    if failure == "token":
        actual["token_ids"][0] = 3
    elif failure == "logprob":
        actual["logprobs"][0]["2"] = -0.2
    elif failure == "nonfinite":
        actual["logprobs"][0]["2"] = float("nan")
    else:
        actual["logprobs"][0]["3"] = -1.0
    with pytest.raises(AssertionError):
        probe.compare_requests(actual, expected)
