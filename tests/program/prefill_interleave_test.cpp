// Execute generate.cpp's complete read_part lambda on the CPU with the real batch_read_piece and TimeBudget.
// Only GPU work and the clock are synthetic. This proves control-flow reachability, not device/state exactness.
#include "strata/program/batch_read.hpp"
#include "strata/program/time_budget.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
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
// Parking/cancellation are not this test's seam. Fail if a fixture accidentally reaches them.
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
};

static Result read_prompt(double budget_ms, bool piped, bool split, bool active, int batch = 4) {
    Clock::ms = 0;
    Result result;
    struct Options { int batch; int prompt_cache = 6; int64_t short_read = 32; double foreground_budget_ms; };
    const Options o{batch, 6, 32, budget_ms};
    const int64_t n = 10000;
    std::vector<int64_t> ids((size_t) n);
    std::vector<int> stages(split ? 2 : 0);
    strata::program::TimeBudget time_budget(budget_ms, 0.25);
    auto last_decode = Clock::now();
    const auto batch_on = [&] { return active; };
    struct PrefillSink {
        const int64_t* begin;
        Result& result;
        int64_t position = 0;
        int64_t chunk() const { return 2048; }
        bool run(const int64_t* data, int64_t count, int64_t pos, std::string& e) {
            if (data != begin + pos || pos != position || count <= 0) { e = "invalid prefill range"; return false; }
            result.reads.emplace_back(pos, count);
            position += count;
            // A drained memory-admitted chunk costs 250 ms, including the original short tail.
            Clock::ms += 250 * (count / chunk() + (count % chunk() != 0));
            return true;
        }
    } sp{ids.data(), result};
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

    std::string e;
    if (!read_part(0, n, e) || !e.empty() || batch_fatal || in_prompt_read || yielded_at != -1 || yielded_slot != -1)
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
    if (fails == 0) std::printf("prefill_interleave_test: ok (positive piped: prefill/decode/prefill; zero: legacy)\n");
    return fails == 0 ? 0 : 1;
}
