# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent VQ decode and official MXFP8-call contracts; CPU only."""

import pytest
import torch

from tests.support.vq2a8_oracle import decode_gs32_fp8


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


@pytest.mark.parametrize("start,stop", [(-32, 32), (0, 0), (0, 65), (1, 32), (True, 32), (0, 32.0)])
def test_rejects_invalid_decode_range(start, stop):
    fields = make_fields(groups=1, n=64, k=64)
    with pytest.raises(ValueError, match="range"):
        decode_gs32_fp8(fields["packed_zn"][0], fields["pair_lut"][0], start=start, stop=stop)


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
