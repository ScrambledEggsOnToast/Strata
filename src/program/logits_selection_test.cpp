// Tests the production stride selection and header row-count contract.
// Native writer coverage comes from the protected CLI artifact validation.
#include "strata/program/logits_selection.hpp"

#include <cstdio>
#include <string>

using namespace strata::program::logits_selection;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what.c_str()); ++g_fail; }
}

}  // namespace

int main() {
    {   // parse_stride: the strict positive decimal contract
        int64_t v = 0;
        check(parse_stride("1", v) && v == 1, "stride 1 parses");
        check(parse_stride("4096", v) && v == 4096, "stride 4096 parses");
        for (const char* bad : {"", "0", "-1", "+1", " 1", "1 ", "1x", "0x10", "1.5",
                                "9223372036854775808", "92233720368547758070"})
            check(!parse_stride(bad, v), std::string("stride '") + bad + "' refused");
    }
    {   // the count the header claims (the 11-vs-10 defect): a 5-token prompt with --max-new 6 writes 10 rows
        check(row_count(5 - 1 + 6, 1) == 10, "5-token prompt, max-new 6: 10 rows, not 11");
        check(row_count(1 - 1 + 1, 1) == 1, "one token in, one token out: 1 row");
        check(row_count(10, 3) == 4, "total 10 stride 3: rows at 0, 3, 6, 9");
        check(row_count(10, 9) == 2, "total 10 stride 9: rows at 0 and the final position 9");
        check(row_count(9, 9) == 2, "total 9 stride 9: position 0 is a multiple AND 8 is final: 2 rows");
        check(row_count(0, 1) == 0 && row_count(5, 0) == 0, "degenerate totals and strides count 0 rows");
    }
    {   // selected: multiples of the stride plus the final position, exactly once
        check(selected(0, 10, 3) && selected(9, 10, 3), "first and final positions selected");
        check(selected(3, 10, 3) && selected(6, 10, 3) && !selected(4, 10, 3), "multiples only in between");
        check(selected(9, 10, 9), "a final position that is also a multiple stays one row");
        for (int64_t p : {-1, 10, 11}) check(!selected(p, 10, 3), "outside [0, total) never selected");
    }
    {   // selected and row_count agree over a sweep: the count of selected positions IS row_count
        for (int64_t total = 1; total <= 257; ++total)
            for (int64_t stride = 1; stride <= 64; ++stride) {
                int64_t n = 0;
                for (int64_t p = 0; p < total; ++p) n += selected(p, total, stride);
                if (n != row_count(total, stride)) {
                    check(false, "selected count " + std::to_string((long long) n) + " != row_count " +
                                     std::to_string((long long) row_count(total, stride)) + " (total " +
                                     std::to_string((long long) total) + ", stride " +
                                     std::to_string((long long) stride) + ")");
                }
                check(selected(total - 1, total, stride),
                      "the final row is always selected (total " + std::to_string((long long) total) + ")");
            }
    }
    if (g_fail == 0) std::printf("logits_selection_test: all checks passed\n");
    return g_fail == 0 ? 0 : 1;
}
