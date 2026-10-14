// Execute generate.cpp's complete read_part lambda on the CPU with the real batch_read_piece and TimeBudget.
// Only GPU work and the clock are synthetic. This proves control-flow reachability, not device/state exactness.
#include "strata/program/batch_read.hpp"
#include "strata/program/time_budget.hpp"

#include <chrono>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct Clock {
    using time_point = std::chrono::time_point<std::chrono::steady_clock, std::chrono::milliseconds>;
    static inline int64_t ms = 0;
    static time_point now() { return time_point(std::chrono::milliseconds(ms)); }
};

struct ConvCheckpoint { std::vector<int32_t> ids; };
namespace strata::program {
struct ResidentAdmission { int unused; const ConvCheckpoint* checkpoint; };
}
// Resident parking/BSTOP are not this test's seam. Fail if a fixture reaches them.
struct NoControlRequests {
    void cancel(int) { throw std::logic_error("unexpected BSTOP"); }
    bool available(int) { throw std::logic_error("unexpected BYIELD"); }
    bool park(int, const std::vector<int32_t>&, const strata::program::ResidentAdmission&, bool, std::string&) {
        throw std::logic_error("unexpected park");
    }
};
struct Result {
    std::vector<std::pair<int64_t, int64_t>> reads;
    std::vector<int64_t> decode_positions;
    int64_t interleave_boundaries = 0;
    double estimate_ms = 0;
    bool okay = false, fatal = false;
    int drains = 0;
    std::string error;
};

// Inject Prefill's terminal outcome at the existing read_part seam. This
// exercises caller error/cancel handling only, never CUDA cleanup or reuse.
namespace strata::prefill {
struct Prefill {
    const int64_t* begin;
    Result& result;
    const char* retained_error = nullptr;
    int64_t position = 0;
    int64_t chunk() const { return 2048; }
    bool run(const int64_t* data, int64_t count, int64_t pos, std::string& e) {
        if (data != begin + pos || pos != position || count <= 0) { e = "invalid prefill range"; return false; }
        result.reads.emplace_back(pos, count);
        position += count;
        Clock::ms += 250 * (count / chunk() + (count % chunk() != 0));
        return true;
    }
    bool run_retained(const int64_t* data, int64_t count, int64_t pos, std::string& e,
                      const std::function<bool(Prefill*, int64_t, std::string&)>&, bool) {
        if (retained_error && *retained_error) { e = retained_error; return false; }
        return run(data, count, pos, e);
    }
};
}
namespace strata::core {
struct Progress { struct Busy { void store(bool) { throw std::logic_error("unexpected device callback"); } } busy; };
inline Progress& progress() { static Progress p; return p; }
}

static Result read_prompt(double budget_ms, bool piped, bool split, bool active, int batch = 4,
                          const char* retained_error = nullptr, bool drain_ok = true) {
    Clock::ms = 0;
    Result result;
    struct Options { int batch; int prompt_cache = 6; int64_t short_read = 32; double foreground_budget_ms; };
    const Options o{batch, 6, 32, budget_ms};
    const int64_t n = 10000;
    std::vector<int64_t> ids((size_t) n);
    struct Stage { strata::prefill::Prefill sp; };
    std::vector<std::unique_ptr<Stage>> stages(split ? 2 : 0);
    strata::program::TimeBudget time_budget(budget_ms, 0.25);
    auto last_decode = Clock::now();
    const auto batch_on = [&] { return active; };
    strata::prefill::Prefill sp{ids.data(), result, retained_error};
    const bool retained_prefill = retained_error != nullptr, retained_layer_yields = true;
    int retained_blocked_stage = -1;
    std::vector<int> stage_group(3, -1);
    struct Group { bool inflight = false; };
    std::vector<Group> pg(2);
    std::string err;
    const auto score_controls = [](bool) { throw std::logic_error("unexpected retained callback"); };
    const auto pump_mtp = [](bool) -> bool { throw std::logic_error("unexpected retained callback"); };
    const auto pipe_drain = [&] {
        ++result.drains;
        err.clear(); // The real caller can share its error string with the peer pipeline.
        if (!drain_ok) err = "descendant drain failed";
        return drain_ok;
    };
    auto decode_round = [&]() -> bool {
        result.decode_positions.push_back(sp.position);
        Clock::ms += 5;
        last_decode = Clock::now();
        time_budget.decoded(5);
        return true;
    };
    const double decode_share = 0.5;
    bool batch_fatal = false, in_prompt_read = false;
    int64_t il_parts = 0, yielded_at = -1;
    double il_ms = 0;
    int yielded_slot = -1;
    std::mutex in_mu;
    std::vector<std::string> in_lines;
    NoControlRequests resident;
    const int admit_slot = -1;
    const std::vector<int> req_imgs;
    const std::vector<ConvCheckpoint> checks;
    const int64_t read_from = 0;
    const bool resumed_from0 = false, trace = false;
    struct Slot { bool failed = false; };
    const std::vector<Slot> bs(4);

    // CMake extracts this verbatim from generate.cpp and regenerates it whenever that source changes.
#include "prefill_read_part.inc"

    std::string& e = err;
    result.okay = read_part(0, n, e);
    result.fatal = batch_fatal;
    result.error = e;
    if (!retained_prefill && (!result.okay || !e.empty() || batch_fatal || in_prompt_read || yielded_at != -1 || yielded_slot != -1))
        throw std::runtime_error("read_part did not finish: " + e);
    result.interleave_boundaries = il_parts;
    result.estimate_ms = time_budget.chunk_ms();
    return result;
}

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

int main() {
    // Piped groups return piece=0 in the unchanged upstream helper, even beside decoding slots.
    CHECK(strata::program::batch_read_piece(true, true, true, true, 2048, 8) == 0);
    const std::vector<std::pair<int64_t, int64_t>> quanta{{0, 2048}, {2048, 6144}, {8192, 1808}};
    const std::vector<std::pair<int64_t, int64_t>> whole{{0, 10000}};
    for (bool piped : {false, true}) {
        const auto positive = read_prompt(1000, piped, true, true);
        CHECK(positive.reads == quanta);                    // cold chunk, whole-chunk quantum, original tail
        CHECK(positive.decode_positions == std::vector<int64_t>{8192}); // decode between prompt boundaries
        CHECK(positive.interleave_boundaries == 2);
        CHECK(positive.estimate_ms == 250);                 // actually reached time_budget.prefilled
    }
    // The initial solo split also reaches the quantum loop, but must not invent a decoder.
    for (bool piped : {false, true}) {
        const auto solo = read_prompt(1000, piped, true, false);
        CHECK(solo.reads == quanta);
        CHECK(solo.decode_positions.empty() && solo.interleave_boundaries == 0);
        CHECK(solo.estimate_ms == 250);
    }
    // Budget 0 retains both legacy whole-run bypasses, with no learned timing or decode callback.
    const auto legacy_piped = read_prompt(0, true, true, true);
    const auto legacy_solo_split = read_prompt(0, false, true, false);
    for (const auto* legacy : {&legacy_piped, &legacy_solo_split}) {
        CHECK(legacy->reads == whole);
        CHECK(legacy->decode_positions.empty() && legacy->interleave_boundaries == 0);
        CHECK(legacy->estimate_ms == 0);
    }
    // Legacy non-pipelined decoding still uses one chunk per piece, not budget-selected quanta.
    const auto legacy_chunked = read_prompt(0, false, true, true);
    const std::vector<std::pair<int64_t, int64_t>> chunks{
        {0, 2048}, {2048, 2048}, {4096, 2048}, {6144, 2048}, {8192, 1808}};
    CHECK(legacy_chunked.reads == chunks);
    CHECK(legacy_chunked.decode_positions.size() == 100 && legacy_chunked.interleave_boundaries == 4);
    CHECK(legacy_chunked.estimate_ms == 0);
    // The outer no-batch bypass remains unconditional, even for an invalid positive-budget CLI combination.
    const auto no_batch = read_prompt(1000, true, true, false, 0);
    CHECK(no_batch.reads == whole && no_batch.decode_positions.empty() && no_batch.estimate_ms == 0);
    const auto cancelled = read_prompt(0, true, true, true, 4, "cancelled");
    CHECK(!cancelled.okay && !cancelled.fatal && cancelled.error == "cancelled" && cancelled.drains == 1);
    for (const char* failure : {"prefill: retained cancellation cleanup failed", "partial publication failed"}) {
        const auto failed = read_prompt(0, true, true, true, 4, failure);
        CHECK(!failed.okay && failed.fatal && failed.error == failure && failed.drains == 1);
    }
    const auto failed_drain = read_prompt(0, true, true, true, 4, "cancelled", false);
    CHECK(!failed_drain.okay && failed_drain.fatal && failed_drain.error == "descendant drain failed" && failed_drain.drains == 1);
    const auto retained = read_prompt(0, true, true, true, 4, "");
    CHECK(retained.okay && !retained.fatal && retained.error.empty() && retained.drains == 1 && retained.reads == chunks);
    if (fails == 0) std::printf("prefill_interleave_test: ok (positive piped: prefill/decode/prefill; zero: legacy)\n");
    return fails == 0 ? 0 : 1;
}
