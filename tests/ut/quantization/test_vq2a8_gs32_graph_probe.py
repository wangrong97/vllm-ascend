# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runner identity and graph replay evidence; no engine or NPU initialization."""

import importlib
import json
import sys
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from tests.e2e.nightly.vq2a8_gs32_helpers import runner_graph_state


@pytest.fixture(params=["full_graph", "ep"])
def probe(request):
    return importlib.import_module(f"tests.e2e.nightly.test_vq2a8_gs32_{request.param}")


def make_runner(version, *, captures=True):
    module = "vllm_ascend.worker.v2.model_runner" if version == 2 else "vllm_ascend.worker.model_runner_v1"
    base = type("NPUModelRunner", (), {"__module__": module})
    # Worker extensions may subclass the runner; inspect its inheritance too.
    runner = type("ExtendedRunner", (base,), {})()
    runner._quantization_validity_buffers = [object()] * 43 if captures else []
    if version == 2:
        descriptor = type("Descriptor", (), {"num_tokens": 1})()
        graph = NS(replay=Mock(return_value="submitted"))
        runner.cudagraph_manager = NS(graphs={descriptor: graph} if captures else {})
    else:
        entry = NS(aclgraph=object(), batch_descriptor=NS(num_tokens=1), num_replays=0)
        runner.model = NS(concrete_aclgraph_entries={1: entry} if captures else {})
    return runner


@pytest.mark.parametrize("version", [1, 2])
def test_tp1_probe_counts_selected_runners_decode_replays(version):
    probe = importlib.import_module("tests.e2e.nightly.test_vq2a8_gs32_full_graph")
    runner = make_runner(version)
    worker = probe.GS32GraphStateWorkerExtension()
    worker.model_runner = runner
    before = worker.vq2a8_gs32_full_graph_state()
    assert before["model_runner_version"] == version
    assert before["captures"] == 1 and before["replays"] == 0
    if version == 2:
        graph = next(iter(runner.cudagraph_manager.graphs.values()))
        for _ in range(probe.OUTPUT_TOKENS - 1):
            assert graph.replay() == "submitted"
    else:
        runner.model.concrete_aclgraph_entries[1].num_replays += probe.OUTPUT_TOKENS - 1
    after = worker.vq2a8_gs32_full_graph_state()
    probe.check_graph_progress("full_graph", before, after, 43)
    assert after["decode"] == runner_graph_state(runner)["decode"]
    assert after["replays"] == probe.OUTPUT_TOKENS - 1
    assert before["replays"] == 0
    if version == 2:
        assert next(iter(runner.cudagraph_manager.graphs.values())) is graph


@pytest.mark.parametrize("version", [1, 2])
def test_tp1_eager_probe_retains_runner_identity_without_claiming_replay(version):
    probe = importlib.import_module("tests.e2e.nightly.test_vq2a8_gs32_full_graph")
    worker = probe.GS32GraphStateWorkerExtension()
    worker.model_runner = make_runner(version, captures=False)
    state = worker.vq2a8_gs32_full_graph_state()
    assert state["model_runner_version"] == version
    assert state["captures"] == state["replays"] == state["validity_owners"] == 0
    probe.check_graph_progress("fused", state, state, 43)
    with pytest.raises(AssertionError):
        probe.check_graph_progress("full_graph", state, state, 43)


@pytest.mark.parametrize("version", [1, 2])
def test_every_named_rpc_rejects_wrong_runner_on_any_rank(probe, version):
    ep = probe.MODULE_NAME.endswith("_ep")
    states = [
        {"rank": rank, "model_runner_version": version, "model_runner": f"runner_v{version}"}
        for rank in range(2 if ep else 1)
    ]
    llm = NS(collective_rpc=Mock(return_value=states))
    args = (llm, 2) if ep else (llm,)
    observed = probe.graph_state(*args, expected_version=version)
    assert observed == (states if ep else states[0])
    llm.collective_rpc.assert_called_once_with(probe.GRAPH_STATE_METHOD)
    states[-1]["model_runner_version"] = 3 - version
    with pytest.raises(AssertionError, match=f"Rank {len(states) - 1} selected"):
        probe.graph_state(*args, expected_version=version)


@pytest.mark.parametrize("version", [1, 2])
def test_model_acceptance_checks_runner_selection_before_generation(probe, version, monkeypatch, tmp_path):
    # Exercise the real acceptance entry point with a wrong worker response.
    # This ensures the CLI's requested V2 cannot silently certify a V1 run.
    ep = probe.MODULE_NAME.endswith("_ep")
    states = [
        {"rank": rank, "model_runner_version": 3 - version, "model_runner": "wrong.runner"}
        for rank in range(2 if ep else 1)
    ]
    llm = NS(collective_rpc=Mock(return_value=states), generate=Mock())
    engine = Mock()
    engine.__enter__ = Mock(return_value=NS(model=llm))
    engine.__exit__ = Mock(return_value=False)
    monkeypatch.setitem(sys.modules, "vllm", NS(SamplingParams=Mock(), envs=NS(VLLM_USE_V2_MODEL_RUNNER=version == 2)))
    monkeypatch.setitem(sys.modules, "tests.e2e.conftest", NS(VllmRunner=Mock(return_value=engine)))
    config = {"num_hidden_layers": 43}
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    if ep:
        monkeypatch.setattr(probe, "checkpoint_inventory", lambda _: (config, {}, {0: {}}))
    args = (tmp_path, "stage.so", "kernel.o", "full_graph")
    if ep:
        args = (*args, 2)
    with pytest.raises(AssertionError, match=f"expected model runner V{version}"):
        probe.run_variant(*args, tmp_path / "report.json")
    llm.generate.assert_not_called()


def test_dp_tp_reference_checks_runner_selection_before_generation(monkeypatch, tmp_path):
    probe = importlib.import_module("tests.e2e.nightly.test_vq2a8_gs32_dp")
    states = [{"rank": rank, "model_runner_version": 1, "model_runner": "runner_v1"} for rank in range(2)]
    llm = NS(collective_rpc=Mock(return_value=states), generate=Mock())
    engine = Mock()
    engine.__enter__ = Mock(return_value=NS(model=llm))
    engine.__exit__ = Mock(return_value=False)
    monkeypatch.setitem(sys.modules, "vllm", NS(SamplingParams=Mock()))
    monkeypatch.setitem(sys.modules, "tests.e2e.conftest", NS(VllmRunner=Mock(return_value=engine)))
    monkeypatch.setattr(probe, "checkpoint_inventory", lambda _: ({}, {}, {}))

    with pytest.raises(AssertionError, match="expected model runner V2"):
        probe.run_tp_reference(tmp_path, "stage.so", "kernel.o", 2, tmp_path / "report.json")
    llm.generate.assert_not_called()


def test_unknown_runner_cannot_be_reported_as_v1():
    with pytest.raises(AssertionError, match="Unsupported model runner"):
        runner_graph_state(NS(model=NS(concrete_aclgraph_entries={})))


def test_ambiguous_runner_cannot_be_reported_as_either_version():
    bases = tuple(type(make_runner(version)).__bases__[0] for version in (1, 2))
    runner = type("AmbiguousRunner", bases, {})()
    with pytest.raises(AssertionError, match="Unsupported model runner"):
        runner_graph_state(runner)


def test_tp1_and_ep_probes_share_one_replay_counter():
    tp1 = importlib.import_module("tests.e2e.nightly.test_vq2a8_gs32_full_graph")
    ep = importlib.import_module("tests.e2e.nightly.test_vq2a8_gs32_ep")
    runner = make_runner(2)
    worker = tp1.GS32GraphStateWorkerExtension()
    worker.model_runner = runner
    graph = next(iter(runner.cudagraph_manager.graphs.values()))
    original_replay = graph.replay
    initial = ep._runner_graph_state(runner)
    counter = graph.replay
    assert isinstance(counter, ep._ReplayCounter)
    worker.vq2a8_gs32_full_graph_state()
    assert graph.replay is counter
    assert graph.replay() == "submitted"
    assert worker.vq2a8_gs32_full_graph_state()["decode"] == {"captures": 1, "replays": 1}
    assert ep._runner_graph_state(runner)["decode"] == {"captures": 1, "replays": 1}
    assert initial["decode"]["replays"] == 0
    assert counter.original_replay is original_replay
    assert next(iter(runner.cudagraph_manager.graphs.values())) is graph


@pytest.mark.parametrize("version", [1, 2])
def test_tp1_probe_rejects_a_non_decode_graph_even_when_replayed(version):
    probe = importlib.import_module("tests.e2e.nightly.test_vq2a8_gs32_full_graph")
    runner = make_runner(version)
    if version == 2:
        next(iter(runner.cudagraph_manager.graphs)).num_tokens = 32
    else:
        runner.model.concrete_aclgraph_entries[1].batch_descriptor.num_tokens = 32
    worker = probe.GS32GraphStateWorkerExtension()
    worker.model_runner = runner
    before = worker.vq2a8_gs32_full_graph_state()
    if version == 2:
        graph = next(iter(runner.cudagraph_manager.graphs.values()))
        for _ in range(probe.OUTPUT_TOKENS - 1):
            graph.replay()
    else:
        runner.model.concrete_aclgraph_entries[1].num_replays += probe.OUTPUT_TOKENS - 1
    after = worker.vq2a8_gs32_full_graph_state()
    assert after["captures"] == 1 and after["replays"] == probe.OUTPUT_TOKENS - 1
    with pytest.raises(AssertionError, match="B1 decode graph"):
        probe.check_graph_progress("full_graph", before, after, 43)
