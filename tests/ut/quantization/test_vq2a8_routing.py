# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical routing regressions using actual router methods on CPU tensors.

Only the fused CANN operator is substituted. These tests do not validate its
device implementation or claim NPU performance.
"""

import ast
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]


def router_class(filename, name, base=object):
    path = ROOT / "vllm_ascend/ops/fused_moe/router" / filename
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
    cls.bases = [ast.Name(id="Base", ctx=ast.Load())]
    cls.body = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("_compute_routing", "_renormalize_topk_weights", "_native_grouped_topk")
    ]
    scope = {"torch": torch, "F": F, "Base": base}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(path), "exec"), scope)
    return scope[name]


def router(scoring, *, fused=False, renormalize=True):
    cls = router_class("grouped_topk_router.py", "AscendGroupedTopKRouter")
    if fused:
        cls = router_class("fused_topk_router.py", "AscendFusedTopKRouter", cls)
    owner = cls()
    owner.scoring_func = scoring
    owner.top_k = 2
    owner.renormalize = renormalize
    owner.routed_scaling_factor = 1.75
    owner.use_grouped_topk = False
    owner.num_expert_group = owner.topk_group = 1
    owner.e_score_correction_bias = None
    owner.tid2eid = None
    owner.is_fused_supported = lambda _: True
    return owner


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("renormalize", [False, True])
def test_sqrtsoftplus_preserves_fp32_scores_bias_selection_and_tie_order(dtype, renormalize):
    logits = torch.tensor([[0, 0, -3, -4], [-3, 2, 1, 0]], dtype=dtype)
    owner = router("sqrtsoftplus", renormalize=renormalize)
    owner.e_score_correction_bias = torch.tensor([0.0, 0.0, 0.1, 0.0])
    weights, ids = owner._compute_routing(logits, logits, torch.int64)
    expected_ids = torch.tensor([[0, 1], [1, 2]])
    scores = F.softplus(logits.float()).sqrt().gather(1, expected_ids)
    if renormalize:
        scores /= scores.sum(dim=-1, keepdim=True)
    assert weights.dtype == torch.float32 and ids.dtype == torch.int64
    torch.testing.assert_close(ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(weights, scores * owner.routed_scaling_factor, rtol=0, atol=0)


def test_hash_fallback_reads_token_table_and_masks_dummy_token_id():
    owner = router("sqrtsoftplus")
    owner.tid2eid = torch.tensor([[3, 1], [0, 2]])
    logits = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.bfloat16)
    weights, ids = owner._compute_routing(logits, logits, torch.int32, input_ids=torch.tensor([-1, 1]))
    expected_ids = torch.tensor([[3, 1], [0, 2]])
    scores = F.softplus(logits.float()).sqrt().gather(1, expected_ids)
    torch.testing.assert_close(ids, expected_ids.to(torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(weights, scores / scores.sum(-1, keepdim=True) * 1.75, rtol=0, atol=0)


@pytest.mark.parametrize("renormalize", [False, True])
def test_fused_sqrtsoftplus_normalizes_then_scales_once(monkeypatch, renormalize):
    owner = router("sqrtsoftplus", fused=True, renormalize=renormalize)
    logits = torch.tensor([[-1, 0, 2, 3]], dtype=torch.bfloat16)
    selected = torch.tensor([[2, 3]], dtype=torch.int32)
    scores = F.softplus(logits.float()).sqrt().gather(1, selected.long())
    calls = []

    def native(**kwargs):
        calls.append(kwargs)
        return scores, selected, None

    monkeypatch.setattr(torch.ops._C_ascend, "moe_gating_top_k_hash", native, raising=False)
    weights, ids = owner._compute_routing(logits, logits, torch.int32)
    assert calls[0]["x"].dtype == torch.float32
    assert calls[0]["renorm"] == 0 and calls[0]["routed_scaling_factor"] == 1.0
    expected = scores / scores.sum(-1, keepdim=True) if renormalize else scores
    torch.testing.assert_close(weights, expected * 1.75, rtol=0, atol=0)
    torch.testing.assert_close(ids, selected, rtol=0, atol=0)


@pytest.mark.parametrize("scoring", ["softmax", "sigmoid"])
def test_other_unfused_scoring_paths_preserve_weights_and_indices(scoring):
    owner = router(scoring)
    logits = torch.tensor([[-2.0, 0.25, 1.5, 3.0]])
    weights, ids = owner._compute_routing(logits, logits, torch.int32)
    probabilities = logits.softmax(-1) if scoring == "softmax" else logits.sigmoid()
    values, indices = probabilities.topk(2, dim=-1)
    torch.testing.assert_close(ids, indices.to(torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(weights, values / values.sum(-1, keepdim=True) * 1.75, rtol=0, atol=0)
