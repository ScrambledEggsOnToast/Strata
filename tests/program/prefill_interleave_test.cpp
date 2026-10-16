// Execute generate.cpp's complete read_part lambda on the CPU with the real batch_read_piece and TimeBudget.
// Only GPU work and the clock are synthetic. This proves control-flow reachability, not device/state exactness.
#include "strata/program/batch_read.hpp"
#include "strata/program/time_budget.hpp"
#include "strata/program/serve_controls.hpp"

#include <chrono>
#include <atomic>
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
struct ResidentAdmission {
    int32_t next_token = 0;
    const std::vector<ConvCheckpoint>* checkpoints = nullptr;
    const float* residual = nullptr;
    int64_t checkpoint_limit = -1;
};
}
// The narrow parking sink records the production lambda's chosen frontier;
// device transfers and slot ownership are exercised by the real-model probe.
struct NoControlRequests {
    bool allow_park = false;
    int64_t parked_at = -1;
    void cancel(int) { throw std::logic_error("unexpected BSTOP"); }
    bool available(int slot) {
        if (!allow_park) throw std::logic_error("unexpected BYIELD");
        return slot == 0;
    }
    bool park(int, const std::vector<int32_t>& prefix, const strata::program::ResidentAdmission&, bool, std::string&) {
        if (!allow_park || parked_at >= 0) throw std::logic_error("unexpected park");
        parked_at = (int64_t) prefix.size();
        return true;
    }
};
struct Result {
    std::vector<std::pair<int64_t, int64_t>> reads;
    std::vector<int64_t> decode_positions;
    int64_t interleave_boundaries = 0;
    std::vector<int64_t> store_positions;
    int64_t parked_at = -1;
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
    bool pipelined = false;
    std::deque<std::string>* incoming = nullptr;
    std::atomic<bool>* cancel_after_run = nullptr;
    int64_t chunk() const { return 2048; }
    bool run(const int64_t* data, int64_t count, int64_t pos, std::string& e) {
        if (data != begin + pos || pos != position || count <= 0) { e = "invalid prefill range"; return false; }
        result.reads.emplace_back(pos, count);
        position += count;
        const int64_t chunks = count / chunk() + (count % chunk() != 0);
        Clock::ms += count == 1024 ? 125 : 250 * chunks - (pipelined ? 150 * (chunks - 1) : 0);
        if (incoming) { incoming->push_back("BYIELD 0"); incoming = nullptr; }
        if (cancel_after_run) cancel_after_run->store(true);
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
struct Progress { struct Busy { void store(bool) {} } busy; };
inline Progress& progress() { static Progress p; return p; }
inline void progress_at(const char*) {}
}

static void split_setting(const char* value) {
    constexpr const char* name = "STRATA_PREFILL_SPLIT_OVER_BUDGET";
#if defined(_WIN32)
    if (_putenv_s(name, value ? value : "") != 0) throw std::runtime_error("cannot set split gate");
#else
    if ((value ? setenv(name, value, 1) : unsetenv(name)) != 0) throw std::runtime_error("cannot set split gate");
#endif
}

static Result read_prompt(double budget_ms, bool piped, bool split, bool active, int batch = 4,
                          const char* retained_error = nullptr, bool drain_ok = true, int64_t n = 10000,
                          double prior_chunk_ms = 0, bool inject_yield = false, int64_t yield_on_decode_at = -1,
                          bool empty_decode_credit = false, bool cancel_first_launch = false,
                          int64_t prior_chunk_tokens = 2048) {
    Clock::ms = 0;
    Result result;
    const bool expects_yield = inject_yield || yield_on_decode_at >= 0;
    struct Options { int batch; int prompt_cache = 6; int64_t short_read = 32; double foreground_budget_ms; };
    const Options o{batch, 6, 32, budget_ms};
    std::vector<int64_t> ids((size_t) n);
    struct Stage { strata::prefill::Prefill sp; };
    std::vector<std::unique_ptr<Stage>> stages(split ? 2 : 0);
    strata::program::TimeBudget time_budget(budget_ms, 0.25);
    if (prior_chunk_ms > 0) time_budget.prefilled(prior_chunk_ms, 1, prior_chunk_tokens);
    // Seed the historical 2048-token cost without letting old credit pay for
    // the midpoint decode: the first half must earn that credit itself.
    if (empty_decode_credit) while (time_budget.decode_allowed(0)) time_budget.decoded(5);
    auto last_decode = Clock::now();
    double credited_prefill_ms = 0, resident_round_ms = 0;
    std::atomic<bool> stop_req{false};
    const auto credit_service = [] {};
    const auto retire_stopped = [] { return true; };
    const auto batch_on = [&] { return active; };
    strata::prefill::Prefill sp{ids.data(), result, retained_error};
    sp.pipelined = piped;
    if (cancel_first_launch) sp.cancel_after_run = &stop_req;
    const bool retained_prefill = retained_error != nullptr, retained_layer_yields = true;
    int retained_blocked_stage = -1;
    std::vector<int> stage_group(3, -1);
    struct Group { bool inflight = false; };
    std::vector<Group> pg(2);
    const auto pipe_inflight = [&] {
        return std::any_of(pg.begin(), pg.end(), [](const Group& group) { return group.inflight; });
    };
    std::string err;
    const auto score_controls = [](bool) {};
    const auto pump_mtp = [](bool) -> bool { throw std::logic_error("unexpected retained callback"); };
    const auto pipe_drain = [&] {
        ++result.drains;
        err.clear(); // The real caller can share its error string with the peer pipeline.
        if (!drain_ok) err = "descendant drain failed";
        return drain_ok;
    };
    std::mutex in_mu;
    std::deque<std::string> in_lines;
    auto decode_round = [&]() -> bool {
        result.decode_positions.push_back(sp.position);
        Clock::ms += 5;
        last_decode = Clock::now();
        time_budget.decoded(5);
        if (time_budget.enabled()) resident_round_ms += 5;
        if (yield_on_decode_at >= 0 && sp.position >= yield_on_decode_at) {
            in_lines.push_back("BYIELD 0");
            yield_on_decode_at = -1;
        }
        return true;
    };
    const double decode_share = 0.5;
    bool batch_fatal = false, in_prompt_read = false;
    int64_t il_parts = 0, yielded_at = -1;
    double il_ms = 0;
    int yielded_slot = -1;
    NoControlRequests resident;
    resident.allow_park = expects_yield;
    if (inject_yield) sp.incoming = &in_lines;
    const auto resident_control = [&](const std::string& command) {
        if (command.rfind("BSTOP ", 0) == 0) resident.cancel(std::atoi(command.c_str() + 6));
    };
    const auto park_replaced = [](int) { return true; }; // GPU snapshotting is outside this CPU control-flow seam.
    const int admit_slot = -1;
    const std::vector<int> req_imgs;
    const std::vector<ConvCheckpoint> checks;
    const int64_t read_from = 0;
    const bool resumed_from0 = false, trace = false;
    struct Slot { bool failed = false; };
    const std::vector<Slot> bs(4);

    // Disk publication is exercised by the real-engine persistence qualification, not this scheduling fixture.
    const auto store_seam = [&] { result.store_positions.push_back(sp.position); };

    // CMake extracts this verbatim from generate.cpp and regenerates it whenever that source changes.
#include "prefill_read_part.inc"

    std::string& e = err;
    result.okay = read_part(0, n, e);
    result.fatal = batch_fatal;
    result.error = e;
    if (!retained_prefill && !expects_yield && !cancel_first_launch &&
        (!result.okay || !e.empty() || batch_fatal || in_prompt_read || yielded_at != -1 || yielded_slot != -1))
        throw std::runtime_error("read_part did not finish: " + e);
    result.interleave_boundaries = il_parts;
    result.estimate_ms = time_budget.chunk_ms();
    result.parked_at = resident.parked_at;
    return result;
}

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

int main() {
    split_setting(nullptr); // Ambient experiment settings must not change the legacy regressions.
    // Piped groups return piece=0 in the unchanged upstream helper, even beside decoding slots.
    CHECK(strata::program::batch_read_piece(true, true, true, true, 2048, 8) == 0);
    const std::vector<std::pair<int64_t, int64_t>> whole{{0, 10000}};
    for (bool piped : {false, true}) {
        const auto positive = read_prompt(1000, piped, true, true);
        int64_t frontier = 0;
        for (const auto& read : positive.reads) {
            CHECK(read.first == frontier && read.second > 0);
            if (piped) CHECK(read.second <= 2048);
            frontier += read.second;
            CHECK(read.second % 2048 == 0 || frontier == 10000);
        }
        CHECK(frontier == 10000 && !positive.decode_positions.empty());
        CHECK(positive.decode_positions.front() > 0 && positive.decode_positions.back() < frontier);
    }
    // Idle pipeline filling preserves arithmetic chunks and the unchanged tail,
    // never invents a decoder, and cannot cheapen a known isolated-C cost.
    for (int64_t n : {2047, 2048, 2049, 4095, 4096, 4097, 8193}) {
        const auto solo = read_prompt(1000, true, true, false, 4, nullptr, true, n, 1200);
        int64_t frontier = 0;
        for (const auto& read : solo.reads) {
            CHECK(read.first == frontier && read.second > 0 && read.second <= 4096);
            frontier += read.second;
            CHECK(read.second % 2048 == 0 || frontier == n);
        }
        CHECK(frontier == n && solo.decode_positions.empty() && solo.interleave_boundaries == 0);
        if (n == 4096) CHECK(solo.reads.front().second == 4096 && solo.estimate_ms == 1200);
        if (n == 4097) CHECK(solo.reads.front().second == 2048);
        if (n < 2048) CHECK(solo.estimate_ms == 1200);
    }
    // An arrival during 2C+tail or 3C must still reach the eligible C frontier,
    // not skip over it and make the waiter sit through the unparkable tail.
    for (int64_t n : {4097, 6144}) {
        const auto yielded = read_prompt(1000, true, true, false, 4, nullptr, true, n, 1200, true);
        CHECK(!yielded.okay && !yielded.fatal && yielded.error == "yield");
        CHECK(yielded.parked_at == 2048 && yielded.decode_positions.empty());
    }
    // A control arriving during either boundary decode or a credited burst
    // parks at the first advanced frontier, before another prefill launch.
    for (int64_t trigger : {0, 1}) {
        const auto yielded = read_prompt(1000, true, true, true, 4, nullptr, true, 10000, 1200, false, trigger);
        CHECK(!yielded.okay && !yielded.fatal && yielded.error == "yield" && yielded.parked_at == 2048);
        CHECK(yielded.reads.size() == 1 && yielded.reads.front().second == 2048);
        CHECK(std::count(yielded.decode_positions.begin(), yielded.decode_positions.end(), 2048) == (trigger == 1 ? 1 : 0));
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
    // A measured over-budget C is still allocated as C=2048, but the opt-in
    // launch reaches the same drained midpoint on ordinary and grouped reads.
    split_setting("1024");
    const std::vector<std::pair<int64_t, int64_t>> split_tail{
        {0, 1024}, {1024, 1024}, {2048, 1024}, {3072, 1024}, {4096, 1}};
    for (bool piped : {false, true}) {
        const auto divided = read_prompt(1000, piped, piped, true, 4, nullptr, true, 4097, 1200, false, -1, true);
        CHECK(divided.okay && divided.reads == split_tail);
        CHECK(!divided.decode_positions.empty() && divided.decode_positions.front() == 1024);
        CHECK((divided.store_positions == std::vector<int64_t>{1024, 2048, 3072, 4096, 4097}));
        CHECK(divided.estimate_ms == 1200); // Half launches and the original tail cannot cheapen C.
    }
    // Complete each selected C as two halves even when only its second half
    // remains; leave the original final tail intact, including tails below 1024.
    for (int64_t n : {2047, 2048, 2049, 3071, 3072, 4095, 4096, 4097, 10000}) {
        const auto divided = read_prompt(1000, true, true, true, 4, nullptr, true, n, 1200);
        int64_t frontier = 0;
        for (const auto& read : divided.reads) {
            CHECK(read.first == frontier && read.second > 0);
            const int64_t expected = frontier < n - n % 2048 ? 1024 : n % 2048;
            CHECK(read.second == expected);
            frontier += read.second;
        }
        CHECK(frontier == n && divided.estimate_ms == 1200);
    }
    // The midpoint re-enters the existing store/control/cancellation seam,
    // rather than a private second-half loop that hides arrivals.
    const auto midpoint_yield = read_prompt(1000, true, true, true, 4, nullptr, true, 4097, 1200, true);
    CHECK(!midpoint_yield.okay && !midpoint_yield.fatal && midpoint_yield.error == "yield");
    CHECK(midpoint_yield.parked_at == 1024 && midpoint_yield.reads.size() == 1);
    CHECK((midpoint_yield.store_positions == std::vector<int64_t>{1024}));
    const auto midpoint_cancel = read_prompt(1000, true, true, true, 4, nullptr, true, 4097, 1200,
                                            false, -1, false, true);
    CHECK(!midpoint_cancel.okay && !midpoint_cancel.fatal && midpoint_cancel.error == "cancelled");
    CHECK(midpoint_cancel.reads.size() == 1 && midpoint_cancel.reads.front().second == 1024);
    CHECK((midpoint_cancel.store_positions == std::vector<int64_t>{1024}));
    const std::vector<std::pair<int64_t, int64_t>> unsplit_tail{{0, 2048}, {2048, 2048}, {4096, 1}};
    for (bool piped : {false, true}) {
        for (double estimate : {900.0, 1000.0}) {
            const auto within = read_prompt(1000, piped, true, true, 4, nullptr, true, 4097, estimate);
            CHECK(within.reads == unsplit_tail);
        }
        const auto idle = read_prompt(1000, piped, true, false, 4, nullptr, true, 2048, 1200);
        CHECK((idle.reads == std::vector<std::pair<int64_t, int64_t>>{{0, 2048}}));
        CHECK(idle.decode_positions.empty());
    }
    const auto idle_fill = read_prompt(1000, true, true, false, 4, nullptr, true, 4096, 1200);
    CHECK((idle_fill.reads == std::vector<std::pair<int64_t, int64_t>>{{0, 4096}}));
    CHECK(idle_fill.estimate_ms == 1200 && idle_fill.decode_positions.empty());
    const auto cold = read_prompt(1000, true, true, true, 4, nullptr, true, 4097);
    CHECK(cold.reads.front().second == 2048);
    // A cost learned for another allocation geometry is not a measured C:
    // execute the cold whole 2048 first and learn its actual cost normally.
    for (bool piped : {false, true}) {
        for (int64_t prior_geometry : {1024, 4096}) {
            const auto mismatched = read_prompt(1000, piped, piped, true, 4, nullptr, true, 4097, 1200,
                                                false, -1, false, false, prior_geometry);
            CHECK(mismatched.reads == unsplit_tail && mismatched.estimate_ms == 250);
        }
    }
    CHECK(read_prompt(0, true, true, true).reads == legacy_piped.reads);
    CHECK(read_prompt(0, false, true, true).reads == legacy_chunked.reads);
    // Only the exact supported value can partition a launch. Positive
    // unsupported values are also refused by the engine's startup validation.
    const char* const disabled_settings[] = {nullptr, "0", "1", "512", "2048", "4096", "01024", "1024junk"};
    for (const char* value : disabled_settings) {
        split_setting(value);
        const auto unchanged = read_prompt(1000, true, true, true, 4, nullptr, true, 4097, 1200);
        CHECK(unchanged.reads == unsplit_tail);
    }
    split_setting(nullptr);
    if (fails == 0) std::printf("prefill_interleave_test: ok (budget interleave, opt-in split, legacy)\n");
    return fails == 0 ? 0 : 1;
}
