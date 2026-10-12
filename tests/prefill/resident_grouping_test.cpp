// Resident device grouping eligibility: no model, CUDA runtime, or GPU.
#include "resident_grouping.hpp"

#include <cstdio>
#include <numeric>
#include <vector>

namespace d = strata::prefill::detail;

int main() {
    int failures = 0, cases = 0;
    const auto check = [&](bool got, bool expected, const char* what) {
        ++cases;
        if (got != expected) { ++failures; std::fprintf(stderr, "FAIL: %s\n", what); }
    };
    std::vector<int32_t> resident(512);
    std::iota(resident.rbegin(), resident.rend(), 16);   // slots are neither expert IDs nor contiguous from zero
    const auto eligible = [&](bool requested = true, bool native = true, bool cache = true,
                              int64_t slots = 528, bool peer = false, bool stream_empty = true) {
        return d::resident_grouping_eligible(requested, native, cache, resident.data(), resident.size(),
                                             slots, peer, stream_empty);
    };
    check(eligible(), true, "all experts resident in a permuted slot layout");
    check(eligible(false), false, "rollback preserves the host path");
    check(eligible(true, false), false, "MMQ/FP16 arithmetic is not silently changed");
    check(eligible(true, true, false), false, "cache required");
    check(eligible(true, true, true, 0), false, "live cache required");
    check(eligible(true, true, true, 527), false, "scratch-loaned final slot is no longer resident");
    check(eligible(true, true, true, 528, true), false, "CPU/remote/peer ownership is not bypassed");
    check(eligible(true, true, true, 528, false, false), false, "streamed experts retain the host route");
    resident[255] = -1;
    check(eligible(), false, "a missing unselected expert also refuses the fast path");
    resident[255] = 272;
    check(eligible(), true, "the next request sees restored residency");
    check(d::resident_grouping_eligible(true, true, true, nullptr, 512, 528, false, true), false,
          "residency table required");
    check(d::resident_grouping_eligible(true, true, true, resident.data(), 0, 528, false, true), false,
          "empty geometry unsupported");
    check(d::resident_grouping_eligible(true, true, true, resident.data(), 1025, 2048, false, true), false,
          "oversized expert scan rejected before indexing");
    resident.assign(1024, 0);
    std::iota(resident.begin(), resident.end(), 0);
    check(d::resident_grouping_eligible(true, true, true, resident.data(), 1024, 1024, false, true), true,
          "maximum scan geometry");
    std::printf("resident_grouping_test: %d cases, %d failures\n", cases, failures);
    return failures ? 1 : 0;
}
