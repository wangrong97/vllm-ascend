# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate the test oracle's layout and routing; CPU checks are not NPU evidence."""

import sys
from types import SimpleNamespace

import pytest
import torch

from tests.support.vq2a8_oracle import OfficialFFNOracle, decode_mxfp4_fp8


def test_fp4_decode_covers_all_codes_and_preserves_negative_zero():
    packed = torch.tensor([[low | ((low + 1) << 4) for low in range(0, 16, 2)]], dtype=torch.uint8)
    actual = decode_mxfp4_fp8(packed)
    expected = torch.tensor([[0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]])
    assert torch.equal(actual.view(torch.uint8), expected.to(torch.float8_e4m3fn).view(torch.uint8))


def make_oracle(monkeypatch):
    calls = []

    def gmm(**arguments):
        calls.append(arguments)
        assert arguments["group_list_type"] == 1 and arguments["group_type"] == 0
        assert arguments["output_dtype"] == torch.float32
        data = arguments["x"][0].float()
        weight = arguments["weight"][0][0].float()
        rows, k = data.shape
        width = weight.shape[1]
        x_scales = arguments["per_token_scale"][0].reshape(rows, k // 32).int() - 127
        w_scales = arguments["scale"][0][0].permute(0, 2, 1).reshape(k // 32, width).int() - 127
        data = data * torch.exp2(x_scales.float()).repeat_interleave(32, 1)
        weight = weight * torch.exp2(w_scales.float()).repeat_interleave(32, 0)
        return [data @ weight]

    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu_grouped_matmul=gmm, float8_e8m0fnu=object()))
    raw = SimpleNamespace(
        vq2a8_num_experts=3,
        w13_mxfp4_weight=torch.full((1, 128, 32), 0x22, dtype=torch.uint8),
        w13_mxfp4_scale=torch.full((1, 128, 2), 127, dtype=torch.uint8),
        w2_mxfp4_weight=torch.full((1, 64, 32), 0x22, dtype=torch.uint8),
        w2_mxfp4_scale=torch.full((1, 64, 2), 127, dtype=torch.uint8),
    )
    spec = SimpleNamespace(
        expert_ids=(1,),
        quant_types={kind: {1: "mxfp4"} for kind in ("gate_up", "down")},
        specs={"gate_up": SimpleNamespace(rows=128, columns=64), "down": SimpleNamespace(rows=64, columns=64)},
    )
    return OfficialFFNOracle(raw, spec, torch.device("cpu"), column_chunk_size=32), raw, calls


def test_oracle_uses_source_scale_axes_before_bf16_rounding(monkeypatch):
    oracle, raw, calls = make_oracle(monkeypatch)
    # Distinguish both K32 scale groups and every output column.
    oracle.fields["w13_mxfp4_scale"][0, :, 0] = 128
    oracle.fields["w13_mxfp4_scale"][0, :, 1] = 129
    oracle.fields["w13_mxfp4_scale"][0, 1::2, 1] = 130
    raw.w13_mxfp4_weight.zero_()  # Native repacking/deletion cannot change the oracle snapshot.
    data = torch.ones((2, 64)).to(torch.float8_e4m3fn).view(torch.uint8)
    scales = torch.tensor([[127, 128], [128, 127]], dtype=torch.uint8)
    result = oracle.project("gate_up", 1, data, scales)
    assert len(calls) == 4 and all(c["weight"][0].shape == (1, 64, 32) for c in calls)
    expected = torch.tensor([[320.0, 576.0], [256.0, 384.0]]).repeat(1, 64).bfloat16()
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize("group_type", [0, 1])
def test_oracle_empty_rank_padding_and_nonresident_rejection(monkeypatch, group_type):
    oracle, _, calls = make_oracle(monkeypatch)
    hidden = torch.full((3, 64), float("nan"), dtype=torch.bfloat16)
    empty, intermediate = oracle(hidden, torch.zeros(3, dtype=torch.int64), group_type, 10.0)
    assert not calls and not empty.count_nonzero()
    assert not intermediate["data"].count_nonzero()
    assert bool((intermediate["scales"] == 127).all())
    hidden[0].fill_(1.0)
    counts = torch.tensor([0, 1, 0])
    groups = counts.cumsum(0) if group_type == 0 else counts
    actual, intermediate = oracle(hidden, groups, group_type, 10.0)
    assert torch.isfinite(actual).all() and actual[0].count_nonzero() == 64
    assert not actual[1:].count_nonzero() and bool((intermediate["scales"][1:] == 127).all())
    counts = torch.tensor([1, 0, 0])
    groups = counts.cumsum(0) if group_type == 0 else counts
    with pytest.raises(ValueError, match="nonresident"):
        oracle(hidden, groups, group_type, 10.0)
