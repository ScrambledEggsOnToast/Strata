#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace strata::core { struct SessionState; }
namespace strata::prefill {

inline constexpr int kPrefillOwners = 4;
inline constexpr int64_t kPrefillRows = 2048;
inline bool packed_prefill_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("STRATA_PACKED_PREFILL");
        return value && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}
// Private preparation is shared by union-row packing and the serial retained-prompt lifecycle.
inline bool retained_prompt_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("STRATA_RETAINED_PROMPT");
        return value && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}
inline bool private_preparation_enabled() { return packed_prefill_enabled() || retained_prompt_enabled(); }


// Immutable until the physical stage chain drains. tokens names exactly [position, position + rows).
// Each stage writes an independently owned destination; the prompt arena never becomes an owner.
struct OwnerSpan {
    uint64_t request = 0, incarnation = 0;
    bool steering = true;
    int slot = -1;
    int64_t position = 0, rows = 0;
    const int64_t* tokens = nullptr;
    const int32_t* next_tokens = nullptr; // same cells' immutable position+1 tokens, for the MTP callback
    std::array<core::SessionState*, 4> stages{};
};
struct OwnerSegment {
    OwnerSpan owner;
    int64_t row = 0;
};

enum class PrefillCompletion { complete, pending_command, failed };
struct SegmentedResult {
    PrefillCompletion completion = PrefillCompletion::failed;
    std::array<int64_t, kPrefillOwners> frontier{};
    uint32_t touched = 0;
    uint64_t execution = 0;
};

// Bounded max-min allocation. Short suffixes receive their whole remainder before a longer owner
// receives the unused share. No delay to collect another owner and no minimum arithmetic span.
inline std::array<int64_t, kPrefillOwners> prefill_shares(
        const std::array<int64_t, kPrefillOwners>& remaining, int count, int64_t capacity) {
    std::array<int64_t, kPrefillOwners> result{};
    while (capacity > 0) {
        int live = 0;
        for (int i = 0; i < count; ++i) live += result[i] < remaining[i];
        if (!live) break;
        const int64_t share = std::max<int64_t>(1, capacity / live);
        for (int i = 0; i < count && capacity > 0; ++i) {
            const int64_t take = std::min({share, remaining[i] - result[i], capacity});
            if (take <= 0) continue;
            result[i] += take;
            capacity -= take;
        }
    }
    return result;
}
} // namespace strata::prefill
