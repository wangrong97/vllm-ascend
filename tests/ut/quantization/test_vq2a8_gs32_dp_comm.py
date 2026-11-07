# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU numeric contracts for DP gather/hash/scatter; not HCCL or NPU proof."""

import ast
from contextlib import nullcontext
from pathlib import Path
from types import MethodType
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
NUM_EXPERTS = 8


def load_method(relative, owner, name, scope):
    source = ROOT / relative
    tree = ast.parse(source.read_text(encoding="utf-8"))
    nodes = tree.body
    if owner:
        nodes = next(node for node in nodes if isinstance(node, ast.ClassDef) and node.name == owner).body
    node = next(node for node in nodes if isinstance(node, ast.FunctionDef) and node.name == name)
    node.decorator_list = []
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), scope)
    return scope[name]


def make_prepare(group, width, dp_size):
    scope = {
        "torch": torch,
        "nn": torch.nn,
        "QuantType": NS(NONE=0),
        "MoEPrepareOutput": NS,
        "_EXTRA_CTX": NS(max_tokens_across_dp=width),
        "get_dp_group": lambda: group,
    }
    instance = NS(moe_config=NS(dp_size=dp_size, pcp_size=1, is_sequence_parallel=False, dp_group=group))
    for name in (
        "_use_ep_sequence_parallel",
        "prepare",
        "_prepare_with_dp_group",
        "all_gather_input_ids",
        "finalize",
        "_finalize_with_dp_group",
    ):
        method = load_method(
            "vllm_ascend/ops/fused_moe/prepare_finalize.py", "PrepareAndFinalizeWithAllGather", name, scope
        )
        setattr(instance, name, MethodType(method, instance))
    return instance


def route_with_hash(prepare, hidden, logits, local_ids, table, expected_ids):
    def hash_kernel(**kwargs):
        # The native hash kernel is the boundary: the CPU contract checks its
        # actual gathered inputs, then uses an explicit table as the oracle.
        torch.testing.assert_close(kwargs["input_ids"], expected_ids)
        torch.testing.assert_close(kwargs["x"], logits)
        ids = kwargs["tid2eid"][kwargs["input_ids"]]
        return torch.tensor([0.25, 0.75]).expand(ids.shape), ids, None

    kernel = Mock(side_effect=hash_kernel)
    torch_api = NS(
        int64=torch.int64,
        int32=torch.int32,
        float32=torch.float32,
        where=torch.where,
        ops=NS(_C_ascend=NS(moe_gating_top_k_hash=kernel)),
    )
    scope = {
        "torch": torch_api,
        "MoECommType": NS(ALLGATHER="allgather"),
        "_EXTRA_CTX": NS(moe_comm_type="allgather", moe_comm_method=NS(prepare_finalize=prepare)),
        "DEEPSEEK_V4_IMAGE_SENTINEL_COUNT": 0,
    }
    compute = load_method(
        "vllm_ascend/ops/fused_moe/router/fused_topk_router.py", "AscendFusedTopKRouter", "_compute_routing", scope
    )
    router = NS(
        bias_vl=None,
        is_fused_supported=lambda _: True,
        topk_group=1,
        num_expert_group=1,
        renormalize=True,
        scoring_func="sqrtsoftplus",
        tid2eid=table,
        e_score_correction_bias=None,
        top_k=2,
        routed_scaling_factor=1.0,
        image_sentinel_lo=129257,
    )
    result = compute(router, hidden, logits, torch.int32, input_ids=local_ids)
    kernel.assert_called_once()
    return result


def expert_partial(hidden, ids, weights, rank, dp_size):
    first = rank * (NUM_EXPERTS // dp_size)
    local = (ids >= first) & (ids < first + NUM_EXPERTS // dp_size)
    coefficients = ((ids.float() + 1) * weights * local).sum(1)
    return hidden * coefficients[:, None]


@pytest.mark.parametrize(
    "counts,bucket",
    [
        ((3, 1, 0, 2), None),
        ((3, 1, 0, 2), 4),
        ((4, 4, 4, 4), None),
        ((4, 4, 4, 4), 4),
        ((0, 0, 1, 0), None),
        ((0, 0, 1, 0), 4),
        # DSpark5 verification has six target token rows per request.
        # DP4 local B16 therefore gathers 384 rows, including input IDs
        # used by the hash router, before the experts scatter results back.
        ((96, 96, 96, 96), 96),
        ((0, 6, 42, 96), 96),
        ((0, 6, 42, 96), None),
        ((0, 1, 3, 4, 0, 2, 1, 4), None),
        ((0, 1, 3, 4, 0, 2, 1, 4), 4),
        ((4,) * 8, None),
        ((4,) * 8, 4),
        ((0,) * 7 + (1,), None),
        ((0,) * 7 + (1,), 4),
        ((0, 6, 42, 96) * 2, 96),
    ],
    ids=[
        "eager-uneven",
        "graph-uneven",
        "eager-full",
        "graph-full",
        "eager-idle",
        "graph-idle",
        "dspark5-b16-full",
        "dspark5-b16-uneven-graph",
        "dspark5-b16-uneven-eager",
        "dp8-eager-uneven",
        "dp8-graph-uneven",
        "dp8-eager-full",
        "dp8-graph-full",
        "dp8-eager-only-rank7",
        "dp8-graph-only-rank7",
        "dp8-dspark5-b16-uneven-graph",
    ],
)
def test_gs32_dp_hash_gather_sum_scatter_and_local_shared_output(counts, bucket):
    dp_size = len(counts)
    width = bucket or max(counts)
    local_inputs, local_logits, local_ids = [], [], []
    for rank, count in enumerate(counts):
        ids = torch.arange(rank * 5 + 1, rank * 5 + 1 + count, dtype=torch.int64)
        hidden = torch.stack((ids.float(), ids.float() + 0.5), dim=1)
        logits = ids.float()[:, None].expand(-1, NUM_EXPERTS).clone()
        if bucket:
            hidden = torch.nn.functional.pad(hidden, (0, 0, 0, bucket - count))
            logits = torch.nn.functional.pad(logits, (0, 0, 0, bucket - count))
            ids = torch.nn.functional.pad(ids, (0, bucket - count))
        local_inputs.append(hidden)
        local_logits.append(logits)
        local_ids.append(ids)

    def pad(tensor):
        return torch.cat((tensor, tensor.new_zeros((width - len(tensor), *tensor.shape[1:]))))

    padded = [[pad(tensor) for tensor in tensors] for tensors in (local_inputs, local_logits, local_ids)]
    gathered_hidden, gathered_logits, gathered_ids = [torch.cat(tensors) for tensors in padded]
    token_ids = torch.arange(max(32, int(gathered_ids.max()) + 1))
    table = torch.stack((token_ids % NUM_EXPERTS, (token_ids + 3) % NUM_EXPERTS), dim=1).to(torch.int32)
    expected_routes = table[gathered_ids]
    expected_weights = torch.tensor([0.25, 0.75]).expand(expected_routes.shape)
    partials = [
        expert_partial(gathered_hidden, expected_routes, expected_weights, rank, dp_size) for rank in range(dp_size)
    ]
    total = torch.stack(partials).sum(0)

    for rank, count in enumerate(counts):
        gather_calls = []

        def gather(tensor, dim, *, rank=rank, gather_calls=gather_calls):
            assert dim == 0
            index = len(gather_calls)
            torch.testing.assert_close(tensor, padded[index][rank])
            gather_calls.append(tensor)
            return torch.cat(padded[index])

        def reduce_scatter(tensor, dim, *, rank=rank):
            assert dim == 0
            # Each rank contributes only its own experts; the collective sums
            # all distinct contributions before selecting this rank's rows.
            torch.testing.assert_close(tensor, partials[rank])
            return total[rank * width : (rank + 1) * width].clone()

        group = NS(all_gather=gather, reduce_scatter=Mock(side_effect=reduce_scatter))
        prepare = make_prepare(group, width, dp_size)

        def routed_forward(*, hidden_states, router_logits, input_ids, prepare=prepare, rank=rank):
            prepared = prepare.prepare(hidden_states, router_logits)
            torch.testing.assert_close(prepared.hidden_states, gathered_hidden)
            torch.testing.assert_close(prepared.router_logits, gathered_logits)
            weights, ids = route_with_hash(
                prepare, prepared.hidden_states, prepared.router_logits, input_ids, table, gathered_ids
            )
            partial = expert_partial(prepared.hidden_states, ids, weights, rank, dp_size)
            return prepare.finalize(partial, reduce_results=True), NS()

        shared = NS(
            prepare_input_before_routed=Mock(side_effect=lambda x: NS(hidden_states=x, is_gathered=False)),
            forward=Mock(side_effect=lambda value, *args, **kwargs: value.hidden_states * 2),
        )
        runner = NS(
            _sequence_parallel_context=nullcontext,
            ascend_shared_experts=shared,
            _can_overlap_sp_shared_with=lambda _: False,
            routed_input_transform=None,
            routed_output_transform=None,
            _prepare_router_and_milestones=lambda value, hidden, logits: (logits, None, None, None),
            routed_experts=NS(forward_impl=routed_forward),
        )
        forward = load_method(
            "vllm_ascend/ops/fused_moe/fused_moe.py",
            "AscendMoERunner",
            "_forward_impl",
            {"RoutedMoEMilestones": NS},
        )
        shared_out, routed_out = forward(
            runner, local_inputs[rank], local_logits[rank], None, input_ids=local_ids[rank]
        )
        shared.prepare_input_before_routed.assert_called_once()
        torch.testing.assert_close(shared.prepare_input_before_routed.call_args.args[0], local_inputs[rank])
        shared.forward.assert_called_once()
        assert len(gather_calls) == 3
        group.reduce_scatter.assert_called_once()

        # Run the real final-reduction decision with a physical TP1 identity.
        # It must not select another EP/DP reduction after the sum-scatter.
        tp_reduce = Mock(side_effect=lambda x: x)
        reduction_scope = {
            "get_forward_context": lambda prepare=prepare: NS(
                no_compile_layers={"test": NS(moe_config=prepare.moe_config)}
            ),
            "_EXTRA_CTX": NS(moe_comm_type="allgather"),
            "MoECommType": NS(ALLGATHER="allgather", ALLTOALL="alltoall", MC2="mc2", FUSED_MC2="fused"),
            "tensor_model_parallel_all_reduce": tp_reduce,
        }
        for name in ("_routed_output_is_reduced", "_maybe_all_reduce_tensor_model_parallel_impl"):
            load_method("vllm_ascend/ops/register_custom_ops.py", None, name, reduction_scope)
        final = reduction_scope["_maybe_all_reduce_tensor_model_parallel_impl"](shared_out + routed_out, "test")
        assert tp_reduce.call_count == 1
        expected = local_inputs[rank].clone()
        for row, token_id in enumerate(local_ids[rank].tolist()):
            coefficients = 0.25 * (token_id % NUM_EXPERTS + 1) + 0.75 * ((token_id + 3) % NUM_EXPERTS + 1)
            expected[row] *= coefficients + 2  # Exactly one local shared contribution.
        torch.testing.assert_close(final, expected)
        torch.testing.assert_close(final[:count], expected[:count])
        assert torch.isfinite(final).all()
