#pragma once
#include <cstdint>
#include <cstdio>

namespace strata::program {

inline constexpr const char* kBatchStaticResidency =
    "--batch implies static expert residency; the adaptive residency planner is inert in batch windows; "
    "misses use startup/first-use ordering for the whole batch run";

// Once, after the actual cache sizes and startup fills are known, before READY.
// Holds queries the owning stage and any peer/helpers for each model pair, so
// duplicates and spare slots on another stage cannot hide a missing expert.
// An unprofiled single-device cache can still fill on first use without eviction;
// sufficient capacity is not a claim that it is already populated.
template<class Holds>
inline bool admit_batch_residency(FILE* out, bool serve, int batch, int64_t layers, int64_t experts,
                                  bool profiled, int64_t unprofiled_slots, int adapt_every,
                                  int adapt_swaps, int adapt_async, Holds holds) {
    if (!serve) return true;
    const int64_t required = layers * experts; // the engine's resolved, validated model geometry
    int64_t resident = 0;
    for (int64_t l = 0; l < layers; ++l)
        for (int64_t e = 0; e < experts; ++e)
            if (holds(l, e)) ++resident;
    if (resident == required) return true;
    if (batch <= 0) {
        std::fprintf(out, "strata serve: WARNING: incomplete startup expert residency (%lld/%lld experts); "
                          "--batch is off, so the solo/pipelined adaptive residency planner remains reachable, "
                          "governed by --adapt-every %d, --adapt-swaps %d, --adapt-async %d and tier prerequisites.\n",
                     (long long) resident, (long long) required, adapt_every, adapt_swaps, adapt_async);
        return true;
    }
    if (!profiled && unprofiled_slots >= required) {
        std::fprintf(out, "strata serve: WARNING: %s. The unprofiled cache can hold the full expert set "
                          "but startup residency is not established (%lld/%lld experts); first-use filling "
                          "is not adaptive eviction. Use a complete --expert-profile for startup residency.\n",
                     kBatchStaticResidency, (long long) resident, (long long) required);
        return true;
    }
    std::fprintf(out, "strata serve: admission refused: limiting=batch:expert_residency "
                      "resident_experts=%lld required_experts=%lld reason=%s. "
                      "Use a complete --expert-profile and enough cache capacity on every owning stage/tier, "
                      "or omit --batch/--slots.\n",
                 (long long) resident, (long long) required, kBatchStaticResidency);
    return false;
}

// Explicit VRAM release is an operator-requested availability trade, not startup
// admission. Preserve it, but say once when it makes a batch cache too small.
inline void warn_batch_residency_resize(FILE* out, int batch, int64_t slots, int64_t required, bool& warned) {
    if (batch <= 0 || slots >= required || warned) return;
    warned = true;
    std::fprintf(out, "strata serve: WARNING: VRAM release leaves --batch with %lld slots for %lld experts; %s. "
                      "Explicit VRAM release is retained; restore full cache residency or omit --batch/--slots.\n",
                 (long long) slots, (long long) required, kBatchStaticResidency);
}

} // namespace strata::program
