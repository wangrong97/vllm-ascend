# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent VQ decode and official MXFP8-call contracts; CPU only."""

import sys
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.ops.fused_moe.vq2a8_gs32_reference import (
    ReferenceMXFP8Projector,
    _GraphProjectionScratch,
    decode_gs32_fp8,
)
from vllm_ascend.quantization.methods.w2a8.vq2a8_gs32_format import validate_gs32_options


@pytest.mark.parametrize("first,count", [(0, 8), (1, 3), (8, 0)])
def test_reference_padding_avoids_uint8_masked_fill_and_preserves_raw_bytes(monkeypatch, first, count):
    # Exercise every byte encoding, including high-bit FP8 data/E8M0 scales.
    data = torch.arange(8 * 1024).remainder(256).to(torch.uint8).reshape(8, 1024)
    scales = torch.arange(256).to(torch.uint8).reshape(8, 32)
    scratch = _GraphProjectionScratch(data, 32)
    pointers = (scratch.data.data_ptr(), scratch.scales.data_ptr())
    original_masked_fill = torch.Tensor.masked_fill_
    calls = []

    def npu_masked_fill(tensor, mask, value):
        assert tensor.dtype != torch.uint8, "Ascend masked_fill_ does not support DT_UINT8"
        calls.append((tensor.dtype, tensor.data_ptr(), value))
        return original_masked_fill(tensor, mask, value)

    monkeypatch.setattr(torch.Tensor, "masked_fill_", npu_masked_fill)
    scratch.gather_input(data, scales, torch.tensor(first), torch.tensor([count]))
    indices = (torch.arange(8) + first).clamp(max=7)
    live = torch.arange(8)[:, None] < count
    assert torch.equal(scratch.data, torch.where(live, data[indices], 0))
    assert torch.equal(scratch.scales, torch.where(live, scales[indices], 127))
    assert scratch.data.dtype == scratch.scales.dtype == torch.uint8
    assert calls == [(torch.int8, pointers[0], 0), (torch.int8, pointers[1], 127)]


@pytest.mark.parametrize("execution", ["fused", "full_graph"])
def test_reference_runtime_options_do_not_require_native_artifacts(execution):
    config = SimpleNamespace(model_config=SimpleNamespace(enforce_eager=execution == "fused"))
    validate_gs32_options(config, {"gs32_backend": "reference_mxfp8", "gs32_execution": execution})


@pytest.mark.parametrize(
    "options",
    [
        {"gs32_backend": "unknown"},
        {"gs32_backend": "reference_mxfp8", "gs32_execution": "serial"},
        {"gs32_backend": "reference_mxfp8", "gs32_execution": "fused", "gs32_fusion": "swiglu_quant"},
        {"gs32_backend": "native", "gs32_execution": "fused"},
    ],
)
def test_reference_and_native_runtime_options_reject_unsupported_settings(options):
    config = SimpleNamespace(model_config=SimpleNamespace(enforce_eager=True))
    with pytest.raises(ValueError):
        validate_gs32_options(config, options)


def make_fields(groups=3, n=96, k=128):
    generator = torch.Generator().manual_seed(93271)
    packed = torch.randint(0, 256, (groups, n // 32, k // 16, 16, 8), generator=generator, dtype=torch.uint8)
    books = torch.randint(0, 256, (groups, n // 32, k // 32, 32), generator=generator, dtype=torch.uint8)
    books[(books & 127) == 127] = 0
    # Non-Cartesian pair codebooks exercise independent even/odd output rows.
    books[:, 0, 0, 0] = 128
    row_scales = torch.linspace(0.73, 1.93, groups * n).reshape(groups, n)
    return {
        "packed_zn": packed,
        "pair_lut": books,
        "row_scale_bits": row_scales.view(torch.int32).to(torch.int64),
    }


def literal_decode(packed, lut):
    """Scalar source-format oracle, independent of vectorized reshape/gather."""
    n, k = packed.shape[0] * 32, packed.shape[1] * 16
    output = torch.empty((k, n), dtype=torch.uint8)
    for row in range(n):
        for column in range(k):
            byte = int(packed[row // 32, column // 16, column % 16, row % 32 // 4])
            index = (byte >> (4 * (row % 4 // 2))) & 15
            output[column, row] = lut[row // 32, column // 32, 2 * index + row % 2]
    return output


class OfficialGMMDouble:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        x, weight, scale = kwargs["x"][0], kwargs["weight"][0], kwargs["scale"][0]
        activation_scale = kwargs["per_token_scale"][0]
        m, k = x.shape
        n = weight.shape[-1]
        assert x.dtype == weight.dtype == torch.float8_e4m3fn
        assert weight.shape == (1, k, n)
        assert weight.is_contiguous()
        assert activation_scale.shape == (m, k // 64, 2)
        assert scale.shape == (1, k // 64, n, 2)
        assert scale.dtype == torch.uint8 and torch.all(scale == 127)
        assert kwargs["output_dtype"] == torch.float32
        assert kwargs["x_dtype"] == kwargs["weight_dtype"] == torch.float8_e4m3fn
        assert kwargs["scale_dtype"] == kwargs["per_token_scale_dtype"] == torch.uint8
        assert kwargs["group_list_type"] == 1 and kwargs["group_type"] == 0 and kwargs["split_item"] == 2
        assert kwargs["bias"] is None
        assert kwargs["group_list"].tolist() == [m]
        self.calls.append((m, k, n))
        scales = torch.exp2(activation_scale.float().reshape(m, k // 32) - 127).repeat_interleave(32, -1)
        return [(x.float() * scales) @ weight[0].float()]


class GraphGMMDouble:
    """Official ABI double with dynamic counts and deliberately poisoned tail."""

    def __init__(self):
        self.calls = []
        self.counts = []
        self.pointers = []

    def __call__(self, **kwargs):
        x, weight, scale = kwargs["x"][0], kwargs["weight"][0], kwargs["scale"][0]
        activation_scale, count = kwargs["per_token_scale"][0], kwargs["group_list"]
        m, k = x.shape
        n = weight.shape[-1]
        assert x.dtype == weight.dtype == torch.float8_e4m3fn
        assert weight.shape == (1, k, n) and weight.is_contiguous()
        assert activation_scale.shape == (m, k // 64, 2)
        assert scale.shape == (1, k // 64, n, 2) and scale.dtype == torch.uint8
        assert kwargs["output_dtype"] == torch.float32
        assert kwargs["x_dtype"] == kwargs["weight_dtype"] == torch.float8_e4m3fn
        assert kwargs["scale_dtype"] == kwargs["per_token_scale_dtype"] == torch.uint8
        assert kwargs["group_list_type"] == 1 and kwargs["group_type"] == 0 and kwargs["split_item"] == 2
        assert kwargs["bias"] is None and count.shape == (1,) and count.dtype == torch.int64
        self.calls.append((m, k, n))
        self.counts.append(count.clone())
        self.pointers.append((x.data_ptr(), weight.data_ptr(), scale.data_ptr(), count.data_ptr()))
        scales = torch.exp2(activation_scale.float().reshape(m, k // 32) - 127).repeat_interleave(32, -1)
        result = (x.float() * scales) @ weight[0].float()
        live = torch.arange(m) < count
        return [torch.where(live[:, None], result, float("nan"))]


def projector(gmm=None, chunk_size=64):
    return ReferenceMXFP8Projector(
        column_chunk_size=chunk_size,
        _test_gmm=OfficialGMMDouble() if gmm is None else gmm,
        _test_scale_dtype=torch.uint8,
    )


def inputs(m=7, k=128):
    generator = torch.Generator().manual_seed(781)
    values = (torch.randn((m, k), generator=generator) * 0.5).to(torch.float8_e4m3fn).view(torch.uint8)
    scales = torch.randint(123, 131, (m, k // 32), generator=generator, dtype=torch.uint8)
    return values, scales


@pytest.mark.parametrize("n,k", [(32, 64), (96, 128), (256, 192)])
def test_decode_matches_literal_lookup_for_every_byte_and_block(n, k):
    fields = make_fields(groups=1, n=n, k=k)
    packed, lut = fields["packed_zn"][0], fields["pair_lut"][0]
    expected = literal_decode(packed, lut)
    actual = decode_gs32_fp8(packed, lut)
    assert actual.dtype == torch.float8_e4m3fn and actual.is_contiguous()
    assert torch.equal(actual.view(torch.uint8), expected)
    for start in range(0, n, 64):
        stop = min(start + 64, n)
        block = decode_gs32_fp8(packed, lut, start=start, stop=stop)
        assert torch.equal(block.view(torch.uint8), expected[:, start:stop])


def test_decoder_preserves_signed_zero_and_arbitrary_pair_values():
    fields = make_fields(groups=1, n=32, k=64)
    packed, lut = fields["packed_zn"][0], fields["pair_lut"][0]
    packed.zero_()
    lut[..., 0], lut[..., 1] = 128, 53
    decoded = decode_gs32_fp8(packed, lut).view(torch.uint8)
    assert torch.all(decoded[:, 0::2] == 128)
    assert torch.all(decoded[:, 1::2] == 53)


def test_official_projection_matches_independent_oracle_and_bounds_decoded_blocks():
    fields = make_fields()
    data, scales = inputs()
    original = {name: tensor.clone() for name, tensor in fields.items()}
    original_data, original_scales = data.clone(), scales.clone()
    gmm = OfficialGMMDouble()
    result = projector(gmm)(data, scales, fields, torch.tensor([2, 0, 3]))
    expected = torch.zeros((7, 96), dtype=torch.bfloat16)
    offset = 0
    for expert, count in enumerate((2, 0, 3)):
        if not count:
            continue
        x = data[offset : offset + count].view(torch.float8_e4m3fn).float()
        x *= torch.exp2(scales[offset : offset + count].float() - 127).repeat_interleave(32, -1)
        weight = literal_decode(fields["packed_zn"][expert], fields["pair_lut"][expert])
        row_scale = fields["row_scale_bits"][expert].to(torch.int32).view(torch.float32)
        expected[offset : offset + count] = ((x @ weight.view(torch.float8_e4m3fn).float()) * row_scale).bfloat16()
        offset += count
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert gmm.calls == [(2, 128, 64), (2, 128, 32), (3, 128, 64), (3, 128, 32)]
    assert torch.count_nonzero(result[5:]) == 0
    assert torch.equal(data, original_data) and torch.equal(scales, original_scales)
    for name in fields:
        assert torch.equal(fields[name], original[name])


def test_row_scale_is_applied_before_the_only_bf16_round():
    fields = make_fields(groups=1, n=32, k=64)
    fields["row_scale_bits"] = torch.full((1, 32), 1.003).view(torch.int32).to(torch.int64)
    data, scales = inputs(m=1, k=64)

    def gmm(**kwargs):
        assert kwargs["output_dtype"] == torch.float32
        return [torch.full((1, 32), 1.003)]

    actual = projector(gmm)(data, scales, fields, torch.tensor([1]))
    expected = (torch.tensor(1.003) * torch.tensor(1.003)).bfloat16()
    prematurely_rounded = (torch.tensor(1.003).bfloat16().float() * torch.tensor(1.003)).bfloat16()
    assert expected != prematurely_rounded
    assert torch.all(actual == expected)


@pytest.mark.parametrize("m", [0, 5])
def test_zero_counts_do_not_decode_or_launch_gmm(monkeypatch, m):
    fields = make_fields(groups=2, n=32, k=64)
    data, scales = inputs(m=m, k=64)
    gmm = OfficialGMMDouble()

    def fail_decode(*args, **kwargs):
        raise AssertionError("Empty experts must not allocate decoded weights")

    monkeypatch.setattr("vllm_ascend.ops.fused_moe.vq2a8_gs32_reference.decode_gs32_fp8", fail_decode)
    output = projector(gmm)(data, scales, fields, torch.zeros(2, dtype=torch.int64))
    assert output.shape == (m, 32) and torch.count_nonzero(output) == 0
    assert not gmm.calls


@pytest.mark.parametrize("returned_dtype", [torch.bfloat16, torch.float16, torch.float64])
def test_rejects_gmm_that_does_not_return_fp32(returned_dtype):
    data, scales = inputs(m=1, k=64)
    with pytest.raises(RuntimeError, match="must return FP32"):
        projector(lambda **kwargs: [torch.ones((1, 32), dtype=returned_dtype)])(
            data, scales, make_fields(groups=1, n=32, k=64), torch.tensor([1])
        )


@pytest.mark.parametrize("chunk_size", [0, -32, 1, 33, True, 32.0])
def test_rejects_invalid_output_chunk_size(chunk_size):
    with pytest.raises(ValueError, match="chunk size"):
        projector(chunk_size=chunk_size)


@pytest.mark.parametrize("start,stop", [(-32, 32), (0, 0), (0, 65), (1, 32), (True, 32), (0, 32.0)])
def test_rejects_invalid_decode_range(start, stop):
    fields = make_fields(groups=1, n=64, k=64)
    with pytest.raises(ValueError, match="range"):
        decode_gs32_fp8(fields["packed_zn"][0], fields["pair_lut"][0], start=start, stop=stop)


@pytest.mark.parametrize(
    "counts", [torch.tensor([-1, 2]), torch.tensor([4, 4]), torch.tensor([1.0, 1.0]), torch.tensor([1])]
)
def test_rejects_invalid_counts(counts):
    data, scales = inputs()
    with pytest.raises(ValueError, match="counts"):
        projector()(data, scales, make_fields(groups=2), counts)


def test_graph_capture_is_rejected_before_reading_counts(monkeypatch):
    monkeypatch.setattr(torch, "npu", SimpleNamespace(is_current_stream_capturing=lambda: True), raising=False)
    with pytest.raises(RuntimeError, match="eager"):
        projector()(None, None, None, None)


def test_test_double_requires_explicit_scale_type():
    with pytest.raises(ValueError, match="test scale dtype"):
        ReferenceMXFP8Projector(_test_gmm=OfficialGMMDouble())


def test_default_backend_selects_official_gmm_without_native_library_and_rejects_cpu(monkeypatch):
    official = OfficialGMMDouble()
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu_grouped_matmul=official, float8_e8m0fnu="e8m0"))

    def forbid_native(*args, **kwargs):
        raise AssertionError("The reference backend must not load a native library")

    monkeypatch.setattr(torch.ops, "load_library", forbid_native)
    reference = ReferenceMXFP8Projector()
    assert reference.gmm is official and reference.scale_dtype == "e8m0"
    data, scales = inputs(m=1, k=64)
    with pytest.raises(ValueError, match="explicit NPU"):
        reference(data, scales, make_fields(groups=1, n=32, k=64), torch.tensor([1]))
    assert not official.calls


@pytest.mark.parametrize("defect", ["packed_dtype", "packed_shape", "k_alignment", "lut_dtype", "lut_shape"])
def test_decoder_rejects_incompatible_source_layout(defect):
    fields = make_fields(groups=1, n=32, k=128)
    packed, lut = fields["packed_zn"][0], fields["pair_lut"][0]
    if defect == "packed_dtype":
        packed = packed.int()
    elif defect == "packed_shape":
        packed = packed[..., :7]
    elif defect == "k_alignment":
        packed = packed[:, :3]
    elif defect == "lut_dtype":
        lut = lut.int()
    else:
        lut = lut[..., :31]
    with pytest.raises(ValueError):
        decode_gs32_fp8(packed, lut)


@pytest.mark.parametrize(
    "defect", ["input_dtype", "input_stride", "scales_shape", "scales_dtype", "weight", "lut", "row_scale"]
)
def test_projection_rejects_invalid_inputs_before_gmm(defect):
    fields = make_fields(groups=1, n=32, k=128)
    data, scales = inputs(m=2, k=128)
    if defect == "input_dtype":
        data = data.float()
    elif defect == "input_stride":
        data = data[:, ::2]
    elif defect == "scales_shape":
        scales = scales[:, :2].contiguous()
    elif defect == "scales_dtype":
        scales = scales.float()
    elif defect == "weight":
        fields["packed_zn"] = fields["packed_zn"].int()
    elif defect == "lut":
        fields["pair_lut"] = fields["pair_lut"][..., :31]
    else:
        fields["row_scale_bits"] = fields["row_scale_bits"].int()
    gmm = OfficialGMMDouble()
    with pytest.raises(ValueError):
        projector(gmm)(data, scales, fields, torch.tensor([2]))
    assert not gmm.calls


@pytest.mark.parametrize("counts", [(2, 0, 3), (0, 0, 7), (7, 0, 0), (0, 0, 0), (0, 1, 1)])
@pytest.mark.parametrize("n,chunk_size", [(32, 64), (96, 64), (256, 128)])
def test_graph_projection_matches_eager_with_empty_experts_and_padding(counts, n, chunk_size):
    fields = make_fields(n=n)
    data, scales = inputs()
    expected = projector(chunk_size=chunk_size)(data, scales, fields, torch.tensor(counts))
    gmm = GraphGMMDouble()
    actual = projector(gmm, chunk_size).project_graph(data, scales, fields, torch.tensor(counts))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.isfinite(actual).all()
    assert len(set(gmm.pointers)) == 1
    assert len(gmm.calls) >= len(counts)
    assert all(call[0] == data.shape[0] and call[2] <= chunk_size for call in gmm.calls)


def test_graph_path_does_not_read_tensor_values_on_host_during_capture(monkeypatch):
    fields, (data, scales), counts = make_fields(), inputs(), torch.tensor([1, 2, 0])
    gmm = GraphGMMDouble()
    reference = projector(gmm)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(is_current_stream_capturing=lambda: True), raising=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("Graph execution must not read device tensor values on the host")

    with monkeypatch.context() as guarded:
        for name in ("cpu", "item", "tolist", "numpy"):
            guarded.setattr(torch.Tensor, name, forbidden)
        result = reference.project_graph(data, scales, fields, counts)
    assert result.shape == (7, 96)
    assert len(gmm.calls) == 9


def test_graph_capture_owners_retain_one_reused_decode_workspace_per_call():
    fields, (data, scales) = make_fields(), inputs()
    reference = projector(GraphGMMDouble())
    owners = [[], []]
    first = reference.project_graph(data, scales, fields, torch.tensor([2, 0, 3]), retained=owners[0])
    first_copy = first.clone()
    second = reference.project_graph(data, scales, fields, torch.tensor([0, 7, 0]), retained=owners[1])
    assert owners[0][0].decoded.shape == owners[1][0].decoded.shape == (128, 32)
    assert owners[0][0].decoded.data_ptr() != owners[1][0].decoded.data_ptr()
    assert len(owners[0]) == len(owners[1]) == 7
    assert first.data_ptr() != second.data_ptr()
    torch.testing.assert_close(first, first_copy, rtol=0, atol=0)


def test_graph_launch_sequence_is_unchanged_when_zero_counts_become_active():
    fields, (data, scales) = make_fields(), inputs()
    counts = torch.zeros(3, dtype=torch.int64)
    gmm = GraphGMMDouble()
    reference = projector(gmm)
    empty = reference.project_graph(data, scales, fields, counts)
    empty_launches = list(gmm.calls)
    counts.copy_(torch.tensor([0, 2, 5]))
    data.bitwise_xor_(128)
    active = reference.project_graph(data, scales, fields, counts)
    assert gmm.calls[len(empty_launches) :] == empty_launches
    assert torch.count_nonzero(empty) == 0
    torch.testing.assert_close(active, projector()(data, scales, fields, counts), rtol=0, atol=0)


@pytest.mark.parametrize("counts", [(-1, 2), (4, 4), (8, 0), (2**63 - 1, 1)])
def test_graph_invalid_counts_never_reach_gmm_and_propagate_deferred_failure(counts):
    fields, (data, scales) = make_fields(groups=2), inputs()
    gmm = GraphGMMDouble()
    result = projector(gmm).project_graph(data, scales, fields, torch.tensor(counts))
    assert torch.isnan(result).all()
    assert all(torch.equal(count, torch.tensor([0])) for count in gmm.counts)


def test_graph_zero_capacity_returns_empty_without_launches():
    fields, (data, scales) = make_fields(), inputs(m=0)
    gmm = GraphGMMDouble()
    retained = []
    result = projector(gmm).project_graph(data, scales, fields, torch.zeros(3, dtype=torch.int64), retained=retained)
    assert result.shape == (0, 96) and not gmm.calls
    assert len(retained) == 1 and retained[0] is result


def test_graph_requires_device_resident_counts():
    fields, (data, scales) = make_fields(), inputs()
    with pytest.raises(ValueError, match="counts must be on the input device"):
        projector(GraphGMMDouble()).project_graph(
            data, scales, fields, torch.empty(3, dtype=torch.int64, device="meta")
        )


def test_graph_scales_fp32_before_bf16_conversion_and_masks_poisoned_tail():
    fields = make_fields(groups=1, n=32, k=64)
    fields["row_scale_bits"] = torch.full((1, 32), 1.003).view(torch.int32).to(torch.int64)
    data, scales = inputs(m=2, k=64)

    def gmm(**kwargs):
        assert kwargs["output_dtype"] == torch.float32
        return [torch.tensor([1.003, float("nan")])[:, None].expand(-1, 32).clone()]

    actual = projector(gmm).project_graph(data, scales, fields, torch.tensor([1]))
    expected = (torch.tensor(1.003) * torch.tensor(1.003)).bfloat16()
    assert torch.all(actual[0] == expected)
    assert torch.count_nonzero(actual[1]) == 0


@pytest.mark.parametrize("dtype,shape", [(torch.bfloat16, (2, 32)), (torch.float32, (1, 32))])
def test_graph_rejects_incompatible_gmm_output(dtype, shape):
    fields, (data, scales) = make_fields(groups=1, n=32, k=64), inputs(m=2, k=64)
    with pytest.raises(RuntimeError, match="must return FP32"):
        projector(lambda **kwargs: [torch.zeros(shape, dtype=dtype)]).project_graph(
            data, scales, fields, torch.tensor([1])
        )
