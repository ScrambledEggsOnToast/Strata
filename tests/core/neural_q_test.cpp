#include "strata/core/spec_prob.hpp"

#include <array>
#include <cstdio>
#include <cstring>
#include <limits>
#include <span>
#include <string>

namespace {
using Row = std::array<int32_t, strata::core::kSpecQStride>;
int failures = 0;
void probability(Row& row, int index, float value) {
    std::memcpy(&row[size_t(strata::core::kSpecQEntries + index)], &value, sizeof(value));
}
Row conditional() {
    Row row{};
    std::fill_n(row.begin(), strata::core::kSpecQEntries, -1);
    for (int i = 0; i < 16; ++i) { row[size_t(i)] = i; probability(row, i, 0.0625f); }
    return row;
}
void check(const char* name, const Row& row, int32_t proposed, bool expected) {
    std::string error;
    const bool accepted = strata::core::validate_spec_q_row(row.data(), proposed, 1024, error);
    if (accepted != expected) { std::fprintf(stderr, "%s: unexpected conditional-q admission\n", name); ++failures; }
}
}

int main() {
    const Row valid = conditional();
    check("normalized selector distribution", valid, 15, true);
    check("proposed token absent must not become a point mass", valid, 16, false);
    Row changed = valid;
    changed[1] = 0;
    check("duplicate candidate", changed, 15, false);
    changed = valid;
    changed[1] = 1024;
    check("out-of-vocabulary candidate", changed, 15, false);
    changed = valid;
    probability(changed, 1, -0.0625f);
    check("negative conditional probability", changed, 15, false);
    changed = valid;
    probability(changed, 1, std::numeric_limits<float>::quiet_NaN());
    check("nonfinite conditional probability", changed, 15, false);
    changed = valid;
    probability(changed, 1, 0.5f);
    check("unnormalized conditional distribution", changed, 15, false);
    changed = valid;
    probability(changed, 0, 0.0f);
    probability(changed, 1, 0.125f);
    check("unselected zero-probability candidate", changed, 15, true);
    check("selected zero-probability candidate must not become a point mass", changed, 0, false);
    changed = valid;
    changed[5] = -1;
    check("hidden candidate after terminator", changed, 15, false);
    changed = valid;
    probability(changed, 32, 0.25f);
    check("probability outside declared support", changed, 15, false);
    {
        std::array<int32_t, 3 * strata::core::kSpecQStride> prefix{};
        for (int row = 0; row < 3; ++row)
            std::copy(valid.begin(), valid.end(), prefix.begin() + row * strata::core::kSpecQStride);
        const int32_t drafts[] = {15, 14, 13, 22, 23};
        auto coverage = [&](int available, int required, std::span<const int32_t> submitted) {
            std::string error;
            return strata::core::validate_spec_q_prefix(prefix.data(), available, required, submitted, 1024, error);
        };
        auto require = [&](bool condition, const char* name) {
            if (!condition) { std::fprintf(stderr, "%s\n", name); ++failures; }
        };
        require(coverage(3, 3, drafts), "neural q prefix followed by explicit point-mass lookup rows refused");
        require(!coverage(2, 3, drafts), "missing neural q row became a point mass");
        require(!coverage(4, 3, drafts), "undeclared q on a point-mass lookup row was accepted");
        require(!coverage(3, -1, drafts), "negative neural coverage was accepted");
        require(!coverage(3, 3, std::span(drafts, 2)), "q coverage exceeded the submitted window");
        const int32_t missing[] = {15, 14, 42, 22, 23};
        require(!coverage(3, 3, missing), "last neural proposal absent from q became a point mass");
        std::string error;
        require(!strata::core::validate_spec_q_prefix(nullptr, 3, 3, drafts, 1024, error),
                "missing conditional transport was accepted");
    }
    std::printf("neural_q_test: %d failures\n", failures);
    return failures ? 1 : 0;
}
