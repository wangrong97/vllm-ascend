// SPDX-License-Identifier: Apache-2.0
#ifndef MIXED_MOE_WEIGHT_OFFSETS_H
#define MIXED_MOE_WEIGHT_OFFSETS_H

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "mixed_tiling.h"

// Pure Host metadata construction, shared by the Torch bridge and standalone
// harness. Counts are deliberately absent: empty experts still own weights.
struct MixedWeightOffsets {
    std::array<uint64_t, kMixedMaxExperts> gate {};
    std::array<uint64_t, kMixedMaxExperts> down {};
    uint64_t gate_bytes = 0;
    uint64_t down_bytes = 0;
};

namespace MixedWeightOffsetDetail {
inline uint64_t Product(uint64_t left, uint64_t right)
{
    if (right != 0 && left > std::numeric_limits<uint64_t>::max() / right) {
        throw std::overflow_error("mixed weight element count overflows uint64");
    }
    return left * right;
}

inline uint64_t Add(uint64_t left, uint64_t right)
{
    if (left > std::numeric_limits<uint64_t>::max() - right) {
        throw std::overflow_error("mixed packed-weight byte count overflows uint64");
    }
    return left + right;
}
} // namespace MixedWeightOffsetDetail

inline MixedWeightOffsets BuildMixedWeightOffsets(
    uint64_t groups, uint64_t k, uint64_t n, uint64_t d,
    const uint32_t *gate_mask, const uint32_t *down_mask)
{
    if (groups == 0 || groups > kMixedMaxExperts || gate_mask == nullptr || down_mask == nullptr) {
        throw std::invalid_argument("mixed weight offsets require 1..256 experts and both precision masks");
    }
    if (k == 0 || k % 64 != 0 || n == 0 || n % 128 != 0 || d == 0 || d % 32 != 0) {
        throw std::invalid_argument("mixed weight offsets require K%64=0, N%128=0 and D%32=0");
    }
    const auto gate_elements = MixedWeightOffsetDetail::Product(k, n);
    const auto down_elements = MixedWeightOffsetDetail::Product(n / 2, d);
    MixedWeightOffsets result;
    for (uint64_t expert = 0; expert < groups; ++expert) {
        const bool gate_w4 = ((gate_mask[expert >> 5] >> (expert & 31)) & 1U) != 0;
        const bool down_w4 = ((down_mask[expert >> 5] >> (expert & 31)) & 1U) != 0;
        if (gate_w4 && !down_w4) {
            throw std::invalid_argument("mixed weight offsets do not support W4 GateUp with W2 Down");
        }
        result.gate[expert] = result.gate_bytes;
        result.down[expert] = result.down_bytes;
        result.gate_bytes = MixedWeightOffsetDetail::Add(result.gate_bytes, gate_elements / (gate_w4 ? 2 : 4));
        result.down_bytes = MixedWeightOffsetDetail::Add(result.down_bytes, down_elements / (down_w4 ? 2 : 4));
    }
    return result;
}

#endif
