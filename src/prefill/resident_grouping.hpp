// The resident-only device route never changes expert ownership or borrows a missing blob.
#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace strata::prefill::detail {

inline bool resident_grouping_requested() {
#if defined(STRATA_USE_HIP)
    return false;
#else
    static const bool on = [] {
        const char* value = std::getenv("STRATA_PREFILL_DEVICE_GROUPING");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    return on;
#endif
}

inline bool resident_grouping_eligible(bool requested, bool native_grouped, bool has_cache,
                                       const int32_t* resident, int64_t n_expert, int64_t live_slots,
                                       bool has_peer, bool layer_stream_empty) {
    if (!requested || !native_grouped || !has_cache || !resident || n_expert < 1 || n_expert > 1024 ||
        live_slots <= 0 || has_peer || !layer_stream_empty)
        return false;
    for (int64_t e = 0; e < n_expert; ++e)
        if (resident[e] < 0 || resident[e] >= live_slots) return false;
    return true;
}

}  // namespace strata::prefill::detail
