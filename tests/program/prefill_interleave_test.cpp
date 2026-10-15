// Execute generate.cpp's complete read_part lambda on the CPU with the real batch_read_piece and TimeBudget.
// Only GPU work, the clock and the conversation store are synthetic. This proves control-flow reachability,
// not device/state exactness.
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
struct ResidentAdmission { int unused; const ConvCheckpoint* checkpoint; };
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
    int64_t parked_at = -1;
    double estimate_ms = 0;
    bool okay = false, fatal = false;
    int drains = 0;
    std::string error;
    // The synthetic store: handovers the read performed, publications that then finished, captures still waiting.
    int64_t store_seams = 0, store_published = 0;
    size_t store_queued = 0;
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
    int64_t chunk() const { return 2048; }
    bool run(const int64_t* data, int64_t count, int64_t pos, std::string& e) {
        if (data != begin + pos || pos != position || count <= 0) { e = "invalid prefill range"; return false; }
        result.reads.emplace_back(pos, count);
        position += count;
        const int64_t chunks = count / chunk() + (count % chunk() != 0);
        Clock::ms += 250 * chunks - (pipelined ? 150 * (chunks - 1) : 0);
        if (incoming) { incoming->push_back("BYIELD 0"); incoming = nullptr; }
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

static Result read_prompt(double budget_ms, bool piped, bool split, bool active, int batch = 4,
                          const char* retained_error = nullptr, bool drain_ok = true, int64_t n = 10000,
                          double prior_chunk_ms = 0, bool inject_yield = false, int64_t yield_on_decode_at = -1,
                          size_t queued_captures = 0) {
    Clock::ms = 0;
    Result result;
    const bool expects_yield = inject_yield || yield_on_decode_at >= 0;
    struct Options { int batch; int prompt_cache = 6; int64_t short_read = 32; double foreground_budget_ms; };
    const Options o{batch, 6, 32, budget_ms};
    std::vector<int64_t> ids((size_t) n);
    struct Stage { strata::prefill::Prefill sp; };
    std::vector<std::unique_ptr<Stage>> stages(split ? 2 : 0);
    strata::program::TimeBudget time_budget(budget_ms, 0.25);
    if (prior_chunk_ms > 0) time_budget.prefilled(prior_chunk_ms, 1, 2048);
    auto last_decode = Clock::now();
    double credited_prefill_ms = 0, resident_round_ms = 0;
    std::atomic<bool> stop_req{false};
    const auto credit_service = [] {};
    const auto retire_stopped = [] { return true; };
    const auto batch_on = [&] { return active; };
    strata::prefill::Prefill sp{ids.data(), result, retained_error};
    sp.pipelined = piped;
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

    // The store's side of a handover, reduced to the part the seam must do: exactly one publication is admitted at a
    // time, the writer finishes within one prefill chunk (a chunk is seconds of GPU work, a publication is the same
    // order), and nothing starts the next one until this thread's pump hands it over.  `queued_captures` is the
    // backlog a serving run accumulates; without a seam it would only leave the queue when a request boundary came.
    struct StoreSink {
        size_t queued = 0;
        bool busy = false;
        int64_t seams = 0, published = 0;
        void pump() {
            ++seams;
            if (busy) { busy = false; ++published; }   // the admitted publication finished before this seam
            if (queued) { --queued; busy = true; }     // and the oldest capture is handed over now
        }
    } store;
    store.queued = queued_captures;
    const auto store_seam = [&] { store.pump(); };

    // CMake extracts this verbatim from generate.cpp and regenerates it whenever that source changes.
#include "prefill_read_part.inc"

    std::string& e = err;
    result.okay = read_part(0, n, e);
    result.fatal = batch_fatal;
    result.error = e;
    result.store_seams = store.seams;
    result.store_published = store.published;
    result.store_queued = store.queued;
    if (!retained_prefill && !expects_yield &&
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
    // A capture the engine already paid for waits for the store's one admitted writer, and only a seam starts the
    // next publication: a queued backlog must advance once per prefill chunk.  Before the chunk loop pumped, a
    // backlog of 6 stayed at 6 through a whole 8-21 s prompt read (the writer idle), and the leftovers reached
    // shutdown; the pump is now reached once per chunk, hands one capture over each time, and reaps the writer's
    // completion at the next one.
    for (bool piped : {false, true}) {
        const auto backlog = read_prompt(1000, piped, true, true, 4, nullptr, true, 10000, 1200, false, -1, 6);
        CHECK(backlog.store_seams == (int64_t) backlog.reads.size());   // one handover per chunk, none inside one
        CHECK(backlog.store_seams > 1);
        CHECK(backlog.store_queued == (size_t) std::max<int64_t>(0, 6 - backlog.store_seams));
        CHECK(backlog.store_published == std::min<int64_t>(6, backlog.store_seams) - 1);   // the writer is kept busy
    }
    // The retained (peer-expert) read pumps at its own 2048-token boundaries, which are the only seams it has.
    const auto retained_backlog = read_prompt(0, true, true, true, 4, "", true, 10000, 0, false, -1, 3);
    CHECK(retained_backlog.store_seams == (int64_t) retained_backlog.reads.size());
    CHECK(retained_backlog.store_queued == 0 && retained_backlog.store_published == 3);
    // Nothing queued: the seams still run, and no publication is invented for a writer with nothing to write.
    const auto cold = read_prompt(1000, true, true, true, 4, nullptr, true, 10000, 1200);
    CHECK(cold.store_seams == (int64_t) cold.reads.size() && cold.store_published == 0 && cold.store_queued == 0);
    if (fails == 0) std::printf("prefill_interleave_test: ok (positive piped: prefill/decode/prefill; zero: legacy)\n");
    return fails == 0 ? 0 : 1;
}
