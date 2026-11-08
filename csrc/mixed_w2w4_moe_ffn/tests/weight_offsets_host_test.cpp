// SPDX-License-Identifier: Apache-2.0
// Exercise the production Host helper without CANN or any device allocation.
#include "weight_offsets.h"

#include <iostream>
#include <random>
#include <string>
#include <vector>

void Check(bool condition, const char *message)
{
    if (!condition) throw std::runtime_error(message);
}

template <typename Error, typename Callback>
void Reject(Callback callback)
{
    try {
        callback();
    } catch (const Error &) {
        return;
    }
    throw std::runtime_error("invalid offset metadata was accepted");
}

int main()
{
    try {
        std::mt19937 generator(95032);
        const uint64_t group_cases[] {1, 2, 31, 32, 33, 64, 127, 128, 129, 255, 256};
        uint64_t cases = 0;
        for (const auto groups : group_cases) {
            for (uint64_t trial = 0; trial < 64; ++trial) {
                const uint64_t k = (1 + generator() % 80) * 64;
                const uint64_t n = (1 + generator() % 48) * 128;
                const uint64_t d = (1 + generator() % 96) * 32;
                std::vector<uint32_t> gate((groups + 31) / 32, 0), down(gate.size(), 0);
                std::vector<uint64_t> counts(groups, 0);
                for (uint64_t expert = 0; expert < groups; ++expert) {
                    // Include homogeneous, alternating and random mixed banks.
                    const auto mode = trial < 3 ? trial : (trial == 3 ? expert % 3 : generator() % 3);
                    if (mode == 2) gate[expert >> 5] |= uint32_t {1} << (expert & 31);
                    if (mode != 0) down[expert >> 5] |= uint32_t {1} << (expert & 31);
                    const uint64_t row_cases[] {0, 1, 2, 127, 128, 129, 257};
                    counts[expert] = trial % 8 == 0 ? 0 : row_cases[generator() % 7];
                }
                const auto actual = BuildMixedWeightOffsets(groups, k, n, d, gate.data(), down.data());
                for (uint64_t phase = 0; phase < 2; ++phase) {
                    const auto &mask = phase == 0 ? gate : down;
                    const auto &offsets = phase == 0 ? actual.gate : actual.down;
                    const auto reduction = phase == 0 ? k : n / 2;
                    const auto columns = phase == 0 ? n : d;
                    const auto elements = reduction * columns;
                    // Independent oracle: append every expert's packed blob,
                    // including inactive experts, exactly as the old scan did.
                    uint64_t old_offset = 0;
                    uint64_t old_table_offset = 0;
                    uint64_t row_begin = 0;
                    for (uint64_t expert = 0; expert < groups; ++expert) {
                        const auto w4 = (mask[expert >> 5] >> (expert & 31)) & 1U;
                        Check(offsets[expert] == old_offset, "mixed packed offset disagrees with sequential scan");
                        Check(expert * (elements / 32) == old_table_offset, "constant table/scale stride changed");
                        if (counts[expert] != 0) {
                            // Both precision branches keep the same dynamic
                            // row prefix and consume the same packed ranges.
                            Check(offsets[expert] + elements / (w4 ? 2 : 4) ==
                                  old_offset + elements / (w4 ? 2 : 4), "active expert range changed");
                            row_begin += counts[expert];
                        }
                        old_offset += elements / (w4 ? 2 : 4);
                        old_table_offset += w4 ? elements / 32 : (reduction / 32) * (columns / 32) * 32;
                    }
                    const auto bytes = phase == 0 ? actual.gate_bytes : actual.down_bytes;
                    Check(bytes == old_offset, "packed blob size changed");
                    for (uint64_t expert = groups; expert < kMixedMaxExperts; ++expert) {
                        Check(offsets[expert] == 0, "unused offset slots must be initialized");
                    }
                    uint64_t expected_rows = 0;
                    for (auto count : counts) expected_rows += count;
                    Check(row_begin == expected_rows, "empty expert handling changed");
                }
                ++cases;
            }
        }
        std::array<uint32_t, 8> zeros {}, ones {};
        ones.fill(std::numeric_limits<uint32_t>::max());
        Reject<std::invalid_argument>([&] { BuildMixedWeightOffsets(0, 64, 128, 32, zeros.data(), zeros.data()); });
        Reject<std::invalid_argument>([&] { BuildMixedWeightOffsets(257, 64, 128, 32, zeros.data(), zeros.data()); });
        Reject<std::invalid_argument>([&] { BuildMixedWeightOffsets(1, 64, 128, 32, nullptr, zeros.data()); });
        Reject<std::invalid_argument>([&] { BuildMixedWeightOffsets(1, 64, 128, 32, zeros.data(), nullptr); });
        Reject<std::invalid_argument>([&] { BuildMixedWeightOffsets(1, 64, 128, 32, ones.data(), zeros.data()); });
        Reject<std::invalid_argument>([&] { BuildMixedWeightOffsets(1, 0, 128, 32, zeros.data(), zeros.data()); });
        Reject<std::invalid_argument>([&] { BuildMixedWeightOffsets(1, 65, 128, 32, zeros.data(), zeros.data()); });
        Reject<std::invalid_argument>([&] { BuildMixedWeightOffsets(1, 64, 64, 32, zeros.data(), zeros.data()); });
        Reject<std::invalid_argument>([&] { BuildMixedWeightOffsets(1, 64, 128, 33, zeros.data(), zeros.data()); });
        Reject<std::overflow_error>([&] { BuildMixedWeightOffsets(1, uint64_t {1} << 57, 128, 32, zeros.data(), zeros.data()); });
        Reject<std::overflow_error>([&] { BuildMixedWeightOffsets(256, uint64_t {1} << 51, 128, 32, ones.data(), ones.data()); });
        Check(kMixedTilingVersion == 4 && kMixedTilingBytes == 5728, "unexpected v4 tiling ABI");
        Check(kMixedGateWeightOffsetsOffset == 1632 && kMixedDownWeightOffsetsOffset == 3680,
              "unexpected static weight offset layout");
        std::cout << "mixed weight offsets: " << cases << " random/boundary banks and 11 invalid cases passed\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
