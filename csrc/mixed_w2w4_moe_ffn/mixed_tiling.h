#ifndef MIXED_MOE_TILING_H
#define MIXED_MOE_TILING_H

#include <cstdint>

// The four original tiling payloads are kept byte-for-byte.  Only this small
// fixed layout is new, so each component still reads its own native ABI.
constexpr uint64_t kMixedTilingAlignment = 32;
constexpr uint64_t kMixedGateW2TilingOffset = 64;
constexpr uint64_t kMixedGateW2TilingBytes = 13 * sizeof(uint64_t);
constexpr uint64_t kMixedGateW4TilingOffset = 192;
constexpr uint64_t kMixedGateW4TilingBytes = 640;
constexpr uint64_t kMixedDownW2TilingOffset = 832;
constexpr uint64_t kMixedDownW2TilingBytes = 13 * sizeof(uint64_t);
constexpr uint64_t kMixedDownW4TilingOffset = 960;
constexpr uint64_t kMixedDownW4TilingBytes = 640;
// ABI v3 appends post-processing options without moving the component payloads.
constexpr uint64_t kMixedPostOptionsOffset = 1600;
constexpr uint64_t kMixedPostOptionsBytes = 32;
constexpr uint64_t kMixedTilingBytes = kMixedPostOptionsOffset + kMixedPostOptionsBytes;

constexpr uint64_t kMixedTilingMagic = 0x4D4F455732573441ULL; // "MOEW2W4A"
constexpr uint64_t kMixedTilingVersion = 3;

constexpr uint64_t kMixedStageGateW2 = 1ULL << 0;
constexpr uint64_t kMixedStageGateW4 = 1ULL << 1;
constexpr uint64_t kMixedStageDownW2 = 1ULL << 2;
constexpr uint64_t kMixedStageDownW4 = 1ULL << 3;
constexpr uint64_t kMixedStageGateMask = kMixedStageGateW2 | kMixedStageGateW4;
constexpr uint64_t kMixedStageDownMask = kMixedStageDownW2 | kMixedStageDownW4;

struct MixedTilingHeader {
    uint64_t magic;
    uint64_t version;
    uint64_t gate_w2_offset;
    uint64_t gate_w4_offset;
    uint64_t down_w2_offset;
    uint64_t down_w4_offset;
    uint64_t total_bytes;
    // Uniform across all cores. For graph replay this describes resident
    // precisions, including experts whose current token count is zero.
    uint64_t stage_mask;
};

struct MixedPostOptions {
    // Zero disables clipping. Positive limits cap Gate above and Up on both
    // sides before SiLU. The host validates that the value is finite.
    float swiglu_limit;
    uint32_t reserved[7];
};

static_assert(sizeof(MixedTilingHeader) == 64, "unexpected mixed tiling header layout");
static_assert(sizeof(MixedPostOptions) == kMixedPostOptionsBytes, "unexpected post options layout");
static_assert(kMixedPostOptionsOffset % kMixedTilingAlignment == 0, "unaligned post options");
static_assert(kMixedGateW2TilingOffset % kMixedTilingAlignment == 0, "unaligned gate W2 tiling");
static_assert(kMixedGateW4TilingOffset % kMixedTilingAlignment == 0, "unaligned gate W4 tiling");
static_assert(kMixedDownW2TilingOffset % kMixedTilingAlignment == 0, "unaligned down W2 tiling");
static_assert(kMixedDownW4TilingOffset % kMixedTilingAlignment == 0, "unaligned down W4 tiling");

#endif
