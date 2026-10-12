// #195: host-only eligibility. Keep architecture/format/shape decisions testable without CUDA.
#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace strata::prefill::detail {

inline bool iq_wmma_requested() {
    static const bool on = [] {
        const char* value = std::getenv("STRATA_PREFILL_IQ_WMMA");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    return on;
}

constexpr bool iq_wmma_layer_eligible(bool requested, int cc, bool native, bool mmq, bool other_experts,
                                      int gu_type, int down_type, int64_t n_embd, int64_t n_ff) {
    return requested && cc == 700 && native && mmq && !other_experts && n_embd == 2560 && n_ff == 640 &&
           (gu_type == 18 || gu_type == 21 || gu_type == 22 || gu_type == 23) &&
           (down_type == 20 || down_type == 42);
}

constexpr bool iq_wmma_group_eligible(int experts, int64_t min_rows) {
    // Provisional screen floor, fixed before evaluation; not a measured crossover.
    return experts > 0 && experts <= 16 && min_rows >= 16;
}

}  // namespace strata::prefill::detail
