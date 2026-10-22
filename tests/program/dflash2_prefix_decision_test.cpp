// #295's preselected historical proposal0. This is derived decision compatibility,
// not a second GPU verifier execution or a numerical-quality gate. The binding
// receipt pins the captured input, CPU result, actual output stream and window log.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {
struct Decision {
    int accepted, keep, emit_begin, emit_rows;
    int32_t correction;
};
// Only row zero was retained as an actual target result for this diagnostic.
// Reject every later access instead of fabricating unobserved target outputs.
struct RecordedTarget {
    int32_t operator[](std::size_t row) const {
        if (row != 0) throw std::runtime_error("decision consumed an unobserved target row");
        return 198;
    }
};
Decision decide(const std::array<int32_t, 3>& proposal) {
    const bool merged = false;
    const int T = 4;
    const std::array<int32_t, 4> window{248068, proposal[0], proposal[1], proposal[2]};
    const RecordedTarget outv;
    const int64_t max_new = 8, produced_n = 1;
    const struct { std::vector<int64_t> eos_ids; } o{{248044, 248046}};
    // Generated verbatim from the production serial acceptance/clipping/EOS code.
#include "dflash2_serial_prefix.inc"
    return {a, keep, emit_begin, emit_rows, outv[std::size_t(a)]};
}
void require(bool okay) {
    if (!okay) throw std::runtime_error("retained paired prefix decision disagrees");
}
}
int main() {
    try {
        const Decision native = decide({271, 332, 1628});
        const Decision cpu = decide({271, 332, 20});
        for (const Decision arm : {native, cpu})
            require(arm.accepted == 0 && arm.keep == 1 && arm.emit_begin == 0 &&
                    arm.emit_rows == 1 && arm.correction == 198);
        std::printf("DFLASH2_PREFIX_DECISION kind=derived retained_actual_target_row=0 "
                    "frontier=18 T=4 anchor=248068 target_output=198 native_a=%d cpu_a=%d "
                    "drift=%d native_keep=%d cpu_keep=%d second_target_run=0\n",
                    native.accepted, cpu.accepted, cpu.accepted - native.accepted,
                    native.keep, cpu.keep);
        std::puts("PASS dflash2_prefix_decision_test");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL dflash2_prefix_decision_test: %s\n", e.what());
        return 1;
    }
}
