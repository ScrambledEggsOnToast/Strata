#include "iq_wmma_policy.hpp"

#include <cstdio>
#include <initializer_list>

int main() {
    using namespace strata::prefill::detail;
    int failures = 0;
    auto check = [&](bool value, const char* label) {
        if (!value) { std::fprintf(stderr, "%s\n", label); ++failures; }
    };
    auto eligible = [](bool on, int cc, bool native, bool mmq, bool other, int gu, int down, int n, int ff) {
        return iq_wmma_layer_eligible(on, cc, native, mmq, other, gu, down, n, ff);
    };
    for (int gu : {18, 21, 22, 23})
        for (int down : {20, 42}) {
            check(eligible(true, 700, true, true, false, gu, down, 2560, 640), "active-pack pair eligible on SM70");
            check(!eligible(false, 700, true, true, false, gu, down, 2560, 640), "default remains off");
            for (int cc : {600, 750, 800, 860, 900})
                check(!eligible(true, cc, true, true, false, gu, down, 2560, 640), "non-Volta path preserved");
        }
    check(!eligible(true, 700, false, true, false, 21, 20, 2560, 640), "canonical layout rejected");
    check(!eligible(true, 700, true, false, false, 21, 20, 2560, 640), "MMQ-off rollback respected");
    check(!eligible(true, 700, true, true, true, 21, 20, 2560, 640), "native-grouped/fused paths preserved");
    for (int gu : {16, 17, 20, 29, 42})
        check(!eligible(true, 700, true, true, false, gu, 20, 2560, 640), "unsupported gate/up format rejected");
    check(!eligible(true, 700, true, true, false, 21, 8, 2560, 640), "unsupported down format rejected");
    check(!eligible(true, 700, true, true, false, 21, 20, 2559, 640), "input-column tail falls back");
    check(!eligible(true, 700, true, true, false, 21, 20, 2560, 639), "FF dimension tail falls back");
    for (int experts : {1, 2, 15, 16}) {
        check(iq_wmma_group_eligible(experts, 16), "bounded group at screen floor eligible");
        check(iq_wmma_group_eligible(experts, 17), "partial token tile eligible");
        check(iq_wmma_group_eligible(experts, 257), "multi-chunk tail eligible");
        check(!iq_wmma_group_eligible(experts, 15), "small group falls back");
        check(!iq_wmma_group_eligible(experts, 0), "empty group falls back");
    }
    check(!iq_wmma_group_eligible(0, 64) && !iq_wmma_group_eligible(17, 64), "table capacity cannot be exceeded");
    // Even the minimum ten routed rows price much more q8_1 H storage than the two tables alias.
    static_assert(2 * 16 * sizeof(void*) <= 10 * (640 / 32) * 36);
    std::printf("IQ WMMA policy: %s (CPU only)\n", failures ? "FAILED" : "passed");
    return failures ? 1 : 0;
}
