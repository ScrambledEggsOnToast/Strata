// Startup metadata and exact diagnostics only: no model, CUDA calls or weight copies.
#include "strata/program/batch_residency.hpp"
#include <cstdio>
#include <cstring>

namespace {
constexpr const char* partial =
    "strata serve: admission refused: limiting=batch:expert_residency "
    "resident_experts=4 required_experts=6 reason=--batch implies static expert residency; "
    "the adaptive residency planner is inert in batch windows; misses use startup/first-use ordering "
    "for the whole batch run. Use a complete --expert-profile and enough cache capacity on every "
    "owning stage/tier, or omit --batch/--slots.\n";

bool output_is(FILE* out, bool actual, bool expected, const char* message) {
    std::rewind(out);
    char text[2048] = {};
    const size_t n = std::fread(text, 1, sizeof(text) - 1, out);
    const bool ok = actual == expected && !std::ferror(out) &&
                    n == std::strlen(message) && std::strcmp(text, message) == 0;
    std::fclose(out);
    if (!ok) std::fprintf(stderr, "batch_residency_test: admission=%d expected=%d; expected %sreceived %s\n",
                          actual, expected, message, text);
    return ok;
}

template<class Holds>
bool startup_is(bool serve, int batch, int64_t layers, int64_t experts, bool profiled, int64_t slots,
                int every, int swaps, int async, Holds holds, bool admitted, const char* message) {
    FILE* out = std::tmpfile();
    if (!out) { std::perror("batch_residency_test: tmpfile"); return false; }
    const bool actual = strata::program::admit_batch_residency(out, serve, batch, layers, experts,
                                                              profiled, slots, every, swaps, async, holds);
    return output_is(out, actual, admitted, message);
}
} // namespace

int main() {
    auto incomplete = [](int64_t, int64_t e) { return e < 2; }; // four of six pairs
    auto empty = [](int64_t, int64_t) { return false; };
    // No adaptive flag can make a partial batch profile adaptive, including async.
    for (int settings = 0; settings < 3; ++settings) {
        const int every = settings == 0 ? 0 : 4, swaps = settings == 0 ? 0 : 96;
        const int async = settings == 2 ? 1 : 0;
        if (!startup_is(true, 2, 2, 3, true, 4, every, swaps, async, incomplete, false, partial)) return 1;
    }
    // Spare capacity does not complete a frozen, trimmed startup profile.
    if (!startup_is(true, 2, 2, 3, true, 6, 4, 96, 0, incomplete, false, partial)) return 1;

    // A primary's spare/duplicate entries for a later stage are not its owner.
    // Helpers and the peer are a union of actual pairs, never summed slot counts.
    bool primary[2][3] = {{true, true, true}, {true, true, true}};
    bool stage[2][3] = {{false, false, false}, {true, false, false}};
    bool peer[2][3] = {{true, false, false}, {false, true, false}};
    bool helper[2][3] = {{true, false, false}, {false, true, false}};
    auto owned = [&](int64_t l, int64_t e) {
        return (l == 0 ? primary[l][e] : stage[l][e]) || peer[l][e] || helper[l][e];
    };
    if (!startup_is(true, 2, 2, 3, true, 6, 0, 0, 0, owned, false,
        "strata serve: admission refused: limiting=batch:expert_residency "
        "resident_experts=5 required_experts=6 reason=--batch implies static expert residency; "
        "the adaptive residency planner is inert in batch windows; misses use startup/first-use ordering "
        "for the whole batch run. Use a complete --expert-profile and enough cache capacity on every "
        "owning stage/tier, or omit --batch/--slots.\n")) return 1;
    helper[1][2] = true;
    if (!startup_is(true, 2, 2, 3, true, 3, 4, 96, 1, owned, true, "")) return 1;

    // Daily profile regression: all 24,576 pairs, --adapt-swaps 0, no new log or refusal.
    int64_t queries = 0;
    auto complete = [&](int64_t, int64_t) { ++queries; return true; };
    if (!startup_is(true, 4, 48, 512, true, 24576, 4, 0, 0, complete, true, "") || queries != 24576) return 1;
    // Non-serving generation is untouched, even with a requested batch count.
    queries = 0;
    if (!startup_is(false, 2, 2, 3, true, 4, 4, 96, 0, complete, true, "") || queries != 0) return 1;

    if (!startup_is(true, 2, 2, 3, false, 5, 4, 96, 0, empty, false,
        "strata serve: admission refused: limiting=batch:expert_residency "
        "resident_experts=0 required_experts=6 reason=--batch implies static expert residency; "
        "the adaptive residency planner is inert in batch windows; misses use startup/first-use ordering "
        "for the whole batch run. Use a complete --expert-profile and enough cache capacity on every "
        "owning stage/tier, or omit --batch/--slots.\n")) return 1;
    // Full-capacity first-use filling is legitimate, but not established startup residency.
    if (!startup_is(true, 2, 2, 3, false, 6, 4, 96, 1, empty, true,
        "strata serve: WARNING: --batch implies static expert residency; the adaptive residency planner "
        "is inert in batch windows; misses use startup/first-use ordering for the whole batch run. "
        "The unprofiled cache can hold the full expert set but startup residency is not established "
        "(0/6 experts); first-use filling is not adaptive eviction. Use a complete --expert-profile "
        "for startup residency.\n")) return 1;

    // Partial caches remain admitted outside batch; do not promise swaps with the flags off.
    if (!startup_is(true, 0, 2, 3, true, 4, 4, 96, 1, incomplete, true,
        "strata serve: WARNING: incomplete startup expert residency (4/6 experts); --batch is off, "
        "so the solo/pipelined adaptive residency planner remains reachable, governed by --adapt-every 4, "
        "--adapt-swaps 96, --adapt-async 1 and tier prerequisites.\n")) return 1;
    if (!startup_is(true, 0, 2, 3, true, 4, 0, 0, 0, incomplete, true,
        "strata serve: WARNING: incomplete startup expert residency (4/6 experts); --batch is off, "
        "so the solo/pipelined adaptive residency planner remains reachable, governed by --adapt-every 0, "
        "--adapt-swaps 0, --adapt-async 0 and tier prerequisites.\n")) return 1;

    FILE* out = std::tmpfile();
    if (!out) { std::perror("batch_residency_test: tmpfile"); return 1; }
    bool warned = false;
    strata::program::warn_batch_residency_resize(out, 2, 6, 6, warned);
    strata::program::warn_batch_residency_resize(out, 0, 5, 6, warned);
    if (warned) { std::fclose(out); return 1; }
    strata::program::warn_batch_residency_resize(out, 2, 5, 6, warned);
    strata::program::warn_batch_residency_resize(out, 2, 4, 6, warned);
    if (!output_is(out, warned, true,
        "strata serve: WARNING: VRAM release leaves --batch with 5 slots for 6 experts; "
        "--batch implies static expert residency; the adaptive residency planner is inert in batch windows; "
        "misses use startup/first-use ordering for the whole batch run. Explicit VRAM release is retained; "
        "restore full cache residency or omit --batch/--slots.\n")) return 1;

    std::puts("batch_residency_test: refusal, full residency and one-time warning cases passed");
    return 0;
}
