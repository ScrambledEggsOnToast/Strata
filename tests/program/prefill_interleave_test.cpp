// Execute generate.cpp's complete read_part lambda on the CPU with the real batch_read_piece and TimeBudget.
// Only GPU work and the clock are synthetic. This proves control-flow reachability, not device/state exactness.
#include "strata/program/batch_read.hpp"
#include "strata/program/time_budget.hpp"
#include "strata/program/serve_controls.hpp"
#include "strata/program/phase_receipts.hpp"
#include "strata/program/checkpoint_preservation.hpp"
#include "strata/program/execution_plan.hpp"

#include <chrono>
#include <atomic>
#include <condition_variable>
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <thread>
#include <vector>

namespace phase = strata::program::phase;

struct Clock {
    using time_point = std::chrono::time_point<std::chrono::steady_clock, std::chrono::milliseconds>;
    static inline std::atomic<int64_t> ms{0};
    static time_point now() { return time_point(std::chrono::milliseconds(ms.load())); }
};

struct ConvCheckpoint {
    std::vector<int32_t> ids;
    // --prefill-pipe: the plan a prompt segment was read in (generate.cpp keeps it on the checkpoint it takes inside
    // the segment).  The mock never plans, so it only has to exist for the extracted read_part to compile.
    struct PromptPlan {
        int64_t chunk = 0, from = -1, to = -1, lend = 0;
        double pipe = 0.0;
        bool single = false, chunked = false;
        bool operator==(const PromptPlan&) const = default;
    } plan;
};
namespace strata::program {
struct ResidentAdmission {
    int32_t next_token = 0;
    const std::vector<ConvCheckpoint>* checkpoints = nullptr;
    const float* residual = nullptr;
    int64_t checkpoint_limit = -1;
    int64_t prefill_split_until = 0;
    ConvCheckpoint::PromptPlan prefill_plan;
};
}
// The narrow parking sink records the production lambda's chosen frontier;
// device transfers and slot ownership are exercised by the real-model probe.
struct NoControlRequests {
    bool allow_park = false;
    int64_t parked_at = -1;
    int64_t prefill_split_until = 0;
    std::vector<int32_t> parked_prefix;
    int64_t take_prefill_split_until() { return std::exchange(prefill_split_until, 0); }
    void cancel(int) { throw std::logic_error("unexpected BSTOP"); }
    bool available(int slot) {
        if (!allow_park) throw std::logic_error("unexpected BYIELD");
        return slot == 0;
    }
    bool park(int, const std::vector<int32_t>& prefix, const strata::program::ResidentAdmission& request,
              bool, std::string&) {
        if (!allow_park || parked_at >= 0) throw std::logic_error("unexpected park");
        parked_at = (int64_t) prefix.size();
        parked_prefix = prefix;
        prefill_split_until = request.prefill_split_until;
        return true;
    }
};
struct Result {
    std::vector<std::pair<int64_t, int64_t>> reads;
    std::vector<int64_t> decode_positions;
    int64_t interleave_boundaries = 0;
    std::vector<int64_t> store_positions;
    int64_t parked_at = -1;
    int64_t parked_split_until = 0;
    int64_t final_frontier = 0;
    std::vector<int32_t> parked_prefix;
    double estimate_ms = 0;
    bool okay = false, fatal = false;
    bool drain_observed = false;
    std::string error;
    std::vector<std::array<int, 2>> retained_starts;
    std::vector<int64_t> retained_start_clocks;
    // Stage/group pairs are semantic work receipts, not callback-count checks.
    std::vector<std::pair<int, int>> completed_stages, drained_stages;
    std::vector<std::array<int, 3>> completions_under_exclusion;
    std::vector<int> reacquired_with_peer;
    std::vector<std::array<int, 2>> seam_rounds;
    std::array<int, 2> completed_rounds{}, starts_after_cancel{}, peer_completions_after_cancel{};
    int pump_calls = 0, cancel_controls = 0;
    int ownership_violations = 0, exclusion_violations = 0, fence_violations = 0, budget_mutations = 0;
    int final_blocked_stage = -1;
    bool final_bounded_round = false, final_inflight = false, decode_credit = false;
    double credited_ms = 0, resident_ms = 0, decode_estimate_ms = 0, decode_age_ms = 0;
    std::vector<int64_t> checkpoint_positions;
    bool prefill_phase_observed = false, copyin_phase_observed = false;
    bool decode_phase_unavailable = false, parked_identity_preserved = false;
    int concurrent_pumps = 0, concurrent_starts = 0, plain_pumps = 0, mtp_reservations = 0;
    int concurrent_chunks = 0, credit_after_overlap = 0;
    double request_scope_credit_ms = 0, credit_after_exit = 0;
    std::array<int, 3> doorbell_served{}, external_on{}, external_off{};
    bool external_balanced = false;
    int uncredited_starts = 0;
    int64_t first_concurrent_start_ms = -1;
    int doorbell_context_errors = 0, starts_after_retirement = 0;
    double retirement_decode_ms = 0;
};

struct ConcurrentSchedule {
    bool enabled = true, mtp = true, pipeline = true, third = false, source_refused = false;
    bool yield = false, cancel = false, fail_read = false, fail_pump = false, fail_retire = false;
    bool cancel_resident = false;
};
struct ReaderVerifier {
    int64_t windows = 0;
    double ms_host = 0;
    std::atomic<bool> busy{false}, external{false};
    std::atomic<int> served{0}, on{0}, off{0};
    void* expected_pool = nullptr;
    std::atomic<int> wrong_pool{0};
    bool batch_busy() const { return busy.load(); }
    int batch_serve(int, void* pool) {
        if (!external.load()) return -1;
        if (busy.exchange(false)) {
            if (pool != expected_pool) { ++wrong_pool; return -1; }
            ++served;
        }
        return 0;
    }
    void set_external_serve(bool value) {
        external.store(value);
        if (value) ++on; else ++off;
    }
};
struct ConcurrentState {
    std::array<ReaderVerifier, 3> verifier;
    std::atomic<int> chunks{0}, polls{0};
    bool inflight = false;
    std::array<int, 3> pool{};
    bool cancel_pending = false, retirement_done = false;
    std::mutex mu;
    std::condition_variable cv;
};

struct RetainedSchedule {
    int64_t stage_ms = 20, pump_ms = 5;
    int64_t checkpoint_every = 0;
    int stop_boundary = -1, cancel_boundary = -1, fail_pump_call = -1;
    bool foreign_callback = false, admission_before_cancel = false;
    // Each launched stage completes only after its own number of pump polls.
    std::array<std::array<int, 3>, 2> completion_polls{{{{1, 1, 1}}, {{1, 1, 1}}}};
    bool reverse_completions = false;
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
    const RetainedSchedule* schedule = nullptr;
    std::vector<Prefill*> owners;
    const std::vector<int>* stage_group = nullptr;
    strata::program::TimeBudget* budget = nullptr;
    std::atomic<bool>* stop = nullptr;
    std::deque<std::string>* controls = nullptr;
    bool* retained_running = nullptr;
    std::function<bool()> consumers_live;
    std::vector<int64_t>* checkpoint_next = nullptr;
    int64_t* next_checkpoint = nullptr;
    const ConcurrentSchedule* concurrent = nullptr;
    ConcurrentState* concurrent_state = nullptr;
    std::mutex* control_mu = nullptr;
    int64_t stopped = -1;
    int64_t chunk() const { return 2048; }
    // The real reader sets this hook only while its helper thread owns Prefill.
    std::function<bool()> stop_at_chunk;
    int64_t stopped_at() const { return stopped; }
    bool reserve_uploads(int64_t) { return true; }
    // --prefill-pipe stays off in this harness: the extracted read_part takes the unplanned path, which is what the
    // BYIELD and retained-read assertions are about.
    static double pipe_setting() { return 0.0; }
    static int64_t pipe_key(double) { return -1; }
    static void log_plan(int64_t, int64_t, int) {}
    int64_t plan_chunk(int64_t) const { return 0; }
    void force_chunk(int64_t, bool, bool) {}
    int stage_count() const { return 1; }
    bool run(const int64_t* data, int64_t count, int64_t pos, std::string& e) {
        if (data != begin + pos || pos != position || count <= 0) { e = "invalid prefill range"; return false; }
        result.reads.emplace_back(pos, count);
        if (concurrent && stop_at_chunk) {
            stopped = -1;
            for (int64_t offset = 0; offset < count;) {
                if (stop && stop->load()) { e = "cancelled"; return false; }
                if (stop_at_chunk()) { stopped = offset; return true; }
                const int polls = concurrent_state->polls.load();
                const int64_t rows = std::min(chunk(), count - offset);
                position += rows;
                offset += rows;
                Clock::ms += 20;
                const int chunks = ++concurrent_state->chunks;
                if (concurrent->yield && chunks == 1) {
                    std::lock_guard<std::mutex> lock(*control_mu);
                    controls->push_back("BYIELD 0");
                }
                if (concurrent->cancel_resident && chunks == 1) {
                    std::lock_guard<std::mutex> lock(*control_mu);
                    controls->push_back("BSTOP 0");
                }
                if (concurrent->cancel && chunks == 1) stop->store(true);
                if (concurrent->fail_read) { e = "injected reader failure"; return false; }
                std::unique_lock<std::mutex> lock(concurrent_state->mu);
                concurrent_state->cv.wait_for(lock, std::chrono::milliseconds(5), [&] {
                    return concurrent_state->polls.load() > polls || (stop && stop->load());
                });
            }
            return true;
        }
        position += count;
        const int64_t chunks = count / chunk() + (count % chunk() != 0);
        Clock::ms += count == 1024 ? 125 : 250 * chunks - (pipelined ? 150 * (chunks - 1) : 0);
        if (incoming) { incoming->push_back("BYIELD 0"); incoming = nullptr; }
        if (cancel_after_run) cancel_after_run->store(true);
        return true;
    }
    bool run_retained(const int64_t* data, int64_t count, int64_t pos, std::string& e,
                      const std::function<bool(Prefill*, int64_t, std::string&)>& step, bool) {
        if (retained_error && *retained_error) { e = retained_error; return false; }
        if (!schedule) return run(data, count, pos, e);
        if (data != begin + pos || pos != position || count <= 0) { e = "invalid prefill range"; return false; }
        struct Exit { bool& running; ~Exit() { running = false; } } exit{*retained_running};
        *retained_running = true;
        result.reads.emplace_back(pos, count);
        result.retained_starts.push_back({});
        int boundary = 0;
        const auto invoke = [&](int stage, int64_t layer) {
            // The production callback must reacquire only this device.
            if (schedule->stop_boundary == boundary) stop->store(true);
            if (schedule->cancel_boundary == boundary) {
                if (schedule->admission_before_cancel) controls->push_back("BGEN queued-admission");
                controls->push_back("BSTOP 0");
            }
            ++boundary;
            const double chunk_ms = budget->chunk_ms(), decode_ms = budget->decode_ms();
            const bool credit = budget->decode_allowed(0);
            bool okay = false;
            if (schedule->foreign_callback) {
                std::thread foreign([&] { okay = step(owners[(size_t)stage], layer, e); });
                foreign.join();
            } else okay = step(owners[(size_t)stage], layer, e);
            if (budget->chunk_ms() != chunk_ms || budget->decode_ms() != decode_ms ||
                budget->decode_allowed(0) != credit) ++result.budget_mutations;
            if (okay && (*stage_group)[(size_t)stage] >= 0) ++result.exclusion_violations;
            if (okay && std::any_of(stage_group->begin(), stage_group->end(), [](int group) { return group >= 0; }))
                result.reacquired_with_peer.push_back(stage);
            if (!okay) return false;
            // This fixture injects Prefill's existing validated cancellation
            // receipt. CUDA cleanup/state validity is not established here.
            if (stop->load()) { e = "cancelled"; return false; }
            return true;
        };
        for (int stage = 0; stage < (int)owners.size(); ++stage) {
            if (!invoke(stage, 0)) return false;
            Clock::ms += schedule->stage_ms;
            if (!invoke(stage, 1)) return false;
            if (schedule->checkpoint_every > 0 && pos + count >= (*checkpoint_next)[(size_t)stage]) {
                if (consumers_live()) ++result.fence_violations;
                result.checkpoint_positions.push_back(pos + count);
                (*checkpoint_next)[(size_t)stage] = pos + count + schedule->checkpoint_every;
            }
        }
        for (int stage = (int)owners.size() - 2; stage >= 0; --stage)
            if (!invoke(stage, 1)) return false;
        if (schedule->checkpoint_every > 0 && pos + count >= *next_checkpoint) {
            if (consumers_live()) ++result.fence_violations;
            *next_checkpoint = pos + count + schedule->checkpoint_every;
        }
        position += count;
        if (incoming) { incoming->push_back("BYIELD 0"); incoming = nullptr; }
        if (cancel_after_run) cancel_after_run->store(true);
        return true;
    }
};
}
namespace strata::core {
struct Progress { struct Busy { void store(bool) {} } busy; };
inline Progress& progress() { static Progress p; return p; }
inline void progress_at(const char*) {}
}
namespace strata::aux_cpus {
inline void place_off_host() {} // placement itself is covered by aux_cpus_test, not this scheduler fixture
}

static void split_setting(const char* value) {
    constexpr const char* name = "STRATA_PREFILL_SPLIT_OVER_BUDGET";
#if defined(_WIN32)
    if (_putenv_s(name, value ? value : "") != 0) throw std::runtime_error("cannot set split gate");
#else
    if ((value ? setenv(name, value, 1) : unsetenv(name)) != 0) throw std::runtime_error("cannot set split gate");
#endif
}

struct RetainedOptions {
    double budget_ms = 100, prior_chunk_ms = 60;
    int64_t tokens = 2048, begin_at = 0, pending_split_until = 0;
    bool active = true, yield = false, drain_ok = true;
};

static Result read_prompt(double budget_ms, bool piped, bool split, bool active, int batch = 4,
                          const char* retained_error = nullptr, bool drain_ok = true, int64_t n = 10000,
                          double prior_chunk_ms = 0, bool inject_yield = false, int64_t yield_on_decode_at = -1,
                          bool empty_decode_credit = false, bool cancel_first_launch = false,
                          int64_t prior_chunk_tokens = 2048, int64_t begin_at = 0, int64_t pending_split_until = 0,
                          const RetainedSchedule* retained_schedule = nullptr,
                          const ConcurrentSchedule* concurrent = nullptr) {
    // Compile Phase0's real diagnostic scopes, not a second fixture-only API.
    // Their host clock is deliberately independent of the synthetic scheduler.
    phase::Leg phase_leg(phase::parse_input(
        "GEN 16 phase_request=retained-fixture phase_leg=1 "
        "phase_incarnation=0123456789abcdef0123456789abcdef phase_priority=background 1 2"), 1, phase::now_ns());
    phase::Leg* phase_current = &phase_leg;
    const bool phase_receipts = true;
    Clock::ms = 0;
    Result result;
    const bool expects_yield = inject_yield || yield_on_decode_at >= 0 || (concurrent && concurrent->yield);
    ConcurrentState concurrent_state;
    for (size_t owner = 0; owner < concurrent_state.verifier.size(); ++owner)
        concurrent_state.verifier[owner].expected_pool = &concurrent_state.pool[owner];
    struct Options {
        int batch;
        int prompt_cache = 6;
        int64_t short_read = 32;
        double foreground_budget_ms;
        int64_t prompt_cache_every = 0;
        // #255: the concurrent gate reads these; the harness never enables it.
        const char* pcie_mode = "pinned";
        int expert_cache_remote[3] = {0, 0, 0};
        bool resident_multiplex = false;
        bool adapt_async = false, spec_split = false;
        bool batch_mtp_third = false;
    };
    const Options o{batch, 6, 32, budget_ms, retained_schedule ? retained_schedule->checkpoint_every : 0,
                    "pinned", {0, 0, 0}, false, false, false, concurrent && concurrent->third};
    std::vector<int64_t> ids((size_t) n);
    for (int64_t i = 0; i < n; ++i) ids[(size_t)i] = i;
    struct Stage { strata::prefill::Prefill sp; };
    std::vector<std::unique_ptr<Stage>> stages(split ? 2 : 0);
    int64_t pp_next_check = o.prompt_cache_every > 0 ? begin_at + o.prompt_cache_every : INT64_MAX;
    std::vector<int64_t> part_next(stages.size() + 1, pp_next_check);
    strata::program::TimeBudget time_budget(budget_ms, 0.25);
    if (prior_chunk_ms > 0) time_budget.prefilled(prior_chunk_ms, 1, prior_chunk_tokens);
    // Seed the historical 2048-token cost without letting old credit pay for
    // the midpoint decode: the first half must earn that credit itself.
    if (empty_decode_credit) while (time_budget.decode_allowed(0)) time_budget.decoded(5);
    auto last_decode = Clock::now();
    double credited_prefill_ms = 0, resident_round_ms = 0;
    std::atomic<bool> stop_req{false};
    bool budget_service = time_budget.enabled();
    auto service_at = Clock::now();
    double prefill_mark = 0, resident_mark = 0;
    const auto credit_service = [&] {
        if (!concurrent || !budget_service) return;
        const auto now = Clock::now();
        const double elapsed = std::chrono::duration<double, std::milli>(now - service_at).count();
        const double wall = std::max(0.0, elapsed - (credited_prefill_ms - prefill_mark) -
                                                  (resident_round_ms - resident_mark));
        time_budget.background_work(wall);
        result.request_scope_credit_ms += wall;
        service_at = now;
        prefill_mark = credited_prefill_ms;
        resident_mark = resident_round_ms;
    };
    std::function<bool()> retire_stopped;
    // One active resident slot per two-slot group; cancelling slot 0 must not
    // disable slot 2. The real resident authority, not this adapter, owns reuse.
    std::array<bool, 4> slot_active{{active, false, active, false}};
    std::array<bool, 2> cancelled_group{};
    const auto batch_on = [&] {
        return retained_schedule ? std::any_of(slot_active.begin(), slot_active.end(), [](bool live) { return live; }) : active;
    };
    strata::prefill::Prefill sp{ids.data(), result, retained_error};
    sp.pipelined = piped;
    sp.position = begin_at;
    sp.concurrent = concurrent;
    sp.concurrent_state = &concurrent_state;
    sp.stop = &stop_req;
    if (cancel_first_launch) sp.cancel_after_run = &stop_req;
    const bool retained_prefill = retained_error != nullptr, retained_layer_yields = budget_ms <= 0;
    bool retained_running = false, bounded_round = false;
    int retained_blocked_stage = -1;
    std::vector<int> stage_group(3, -1);
    struct Group { bool inflight = false, round_started = false; int stage = 0; };
    std::vector<Group> pg(2);
    std::array<int, 3> remaining_polls{};
    bool draining = false;
    const auto group_active = [&](int group) {
        return retained_schedule ? slot_active[(size_t)group * 2] || slot_active[(size_t)group * 2 + 1] : active;
    };
    const auto pipe_inflight = [&] {
        if (concurrent && concurrent_state.inflight) return true;
        return std::any_of(pg.begin(), pg.end(), [](const Group& group) { return group.inflight; });
    };
    std::string err;
    int64_t bt_tokens = 0, bt_rows = 0;
    uint64_t batch_commit_epoch = 0;
    const auto concurrent_pump = [&](bool may_start, bool mtp) {
        ++result.concurrent_pumps;
        if (!mtp) ++result.plain_pumps;
        if (concurrent_state.inflight) {
            concurrent_state.inflight = false;
            ++bt_rows;
            bt_tokens += 2;
            batch_commit_epoch += concurrent->mtp ? 2 : 1;
        }
        if (may_start && active) {
            if (time_budget.enabled() && !time_budget.decode_allowed(0)) ++result.uncredited_starts;
            if (concurrent_state.retirement_done) ++result.starts_after_retirement;
            if (result.first_concurrent_start_ms < 0) result.first_concurrent_start_ms = Clock::ms;
            ++result.concurrent_starts;
            concurrent_state.inflight = true;
            const int owners = concurrent->pipeline ? (concurrent->third ? 3 : 2) : 1;
            for (int owner = 0; owner < owners; ++owner)
                concurrent_state.verifier[(size_t)owner].busy.store(true);
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
            for (;;) {
                bool busy = false;
                for (int owner = 0; owner < owners; ++owner)
                    busy |= concurrent_state.verifier[(size_t)owner].batch_busy();
                if (!busy) break;
                if (std::chrono::steady_clock::now() >= deadline) {
                    err = "a group-owned doorbell was not served";
                    return false;
                }
                std::this_thread::yield();
            }
        }
        Clock::ms += 5;
        ++concurrent_state.polls;
        concurrent_state.cv.notify_all();
        if (concurrent->fail_pump && may_start) { err = "injected pump failure"; return false; }
        return true;
    };
    const auto pump_mtp = [&](bool may_start) -> bool {
        if (concurrent) return concurrent_pump(may_start, true);
        if (!retained_schedule) throw std::logic_error("unexpected retained callback");
        ++result.pump_calls;
        if (result.pump_calls > 4096) throw std::logic_error("synthetic pump failed to converge");
        Clock::ms += retained_schedule->pump_ms;
        if (result.pump_calls == retained_schedule->fail_pump_call) {
            err = "retained pump failed";
            return false;
        }
        for (int order = 0; order < (int)stage_group.size(); ++order) {
            const int stage = retained_schedule->reverse_completions ? (int)stage_group.size() - 1 - order : order;
            const int gi = stage_group[(size_t)stage];
            if (gi < 0 || --remaining_polls[(size_t)stage] > 0) continue;
            stage_group[(size_t)stage] = -1;
            pg[(size_t)gi].stage = stage + 1;
            result.completed_stages.emplace_back(stage, gi);
            if (draining) result.drained_stages.emplace_back(stage, gi);
            if (retained_blocked_stage >= 0)
                result.completions_under_exclusion.push_back({retained_blocked_stage, stage, gi});
            if (stage + 1 == (int)stage_group.size()) {
                pg[(size_t)gi].inflight = false;
                ++result.completed_rounds[(size_t)gi];
                if (result.cancel_controls > 0 && !cancelled_group[(size_t)gi])
                    ++result.peer_completions_after_cancel[(size_t)gi];
            }
        }
        for (int stage = (int)stage_group.size() - 1; stage >= 0; --stage) {
            if (stage == retained_blocked_stage || stage_group[(size_t)stage] >= 0) continue;
            for (size_t gi = 0; gi < pg.size(); ++gi) {
                auto& group = pg[gi];
                if (stage == 0) {
                    if (!may_start || !group_active((int)gi) || group.inflight || (bounded_round && group.round_started)) continue;
                    group.inflight = true;
                    group.round_started = bounded_round;
                    group.stage = 0;
                    if (cancelled_group[gi]) ++result.starts_after_cancel[gi];
                    if (retained_running) {
                        ++result.retained_starts.back()[gi];
                        result.retained_start_clocks.push_back(Clock::ms - retained_schedule->pump_ms);
                    }
                } else if (!group.inflight || group.stage != stage) continue;
                stage_group[(size_t)stage] = (int)gi;
                remaining_polls[(size_t)stage] = retained_schedule->completion_polls[gi][(size_t)stage];
                break;
            }
        }
        // A consumer already occupying the excluded device may finish there;
        // only new launches onto that device are forbidden above.
        return true;
    };
    const auto pipe_drain = [&] {
        result.drain_observed = true;
        if (retained_running) ++result.ownership_violations;
        err.clear(); // The real caller can share its error string with the peer pipeline.
        if (!drain_ok) err = "descendant drain failed";
        if (!drain_ok) return false;
        draining = true;
        while (pipe_inflight()) {
            if (!pump_mtp(false)) { draining = false; return false; }
        }
        draining = false;
        return true;
    };
    retire_stopped = [&] {
        if (!concurrent) return true;
        if (concurrent_state.cancel_pending) {
            const auto begin = Clock::now();
            const bool drained = pipe_drain();
            Clock::ms += 1000; // an indivisible cancellation drain must consume, not mint, credit
            const double wall = std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
            resident_round_ms += wall;
            result.retirement_decode_ms += wall;
            concurrent_state.cancel_pending = false;
            concurrent_state.retirement_done = true;
            if (!drained) return false;
        }
        return !concurrent->fail_retire || concurrent_state.chunks == 0;
    };
    std::mutex in_mu;
    std::deque<std::string> in_lines;
    sp.control_mu = &in_mu;
    if (concurrent) sp.controls = &in_lines;
    if (retained_schedule) {
        sp.schedule = retained_schedule;
        sp.owners.push_back(&sp);
        for (auto& stage : stages) {
            stage = std::make_unique<Stage>(Stage{strata::prefill::Prefill{ids.data(), result}});
            sp.owners.push_back(&stage->sp);
        }
        sp.stage_group = &stage_group;
        sp.budget = &time_budget;
        sp.stop = &stop_req;
        sp.controls = &in_lines;
        sp.retained_running = &retained_running;
        sp.consumers_live = pipe_inflight;
        sp.checkpoint_next = &part_next;
        sp.next_checkpoint = &pp_next_check;
    }
    auto decode_round = [&]() -> bool {
        if (retained_running) ++result.ownership_violations;
        result.decode_positions.push_back(sp.position);
        const auto begin = Clock::now();
        if (retained_schedule) {
            const auto before = result.completed_rounds;
            bounded_round = true;
            for (auto& group : pg) group.round_started = group.inflight;
            for (;;) {
                bool pending = pipe_inflight();
                for (size_t gi = 0; gi < pg.size(); ++gi)
                    pending = pending || (!pg[gi].round_started && group_active((int)gi));
                if (!pending) break;
                if (!pump_mtp(true)) { bounded_round = false; return false; }
            }
            bounded_round = false;
            result.seam_rounds.push_back({result.completed_rounds[0] - before[0],
                                         result.completed_rounds[1] - before[1]});
            if (pipe_inflight()) ++result.fence_violations;
        } else Clock::ms += 5;
        last_decode = Clock::now();
        const double wall_ms = std::chrono::duration<double, std::milli>(last_decode - begin).count();
        time_budget.decoded(wall_ms);
        if (time_budget.enabled()) resident_round_ms += wall_ms;
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
    resident.prefill_split_until = pending_split_until;
    if (inject_yield) sp.incoming = &in_lines;
    const auto resident_control = [&](const std::string& command) {
        if (command.rfind("BSTOP ", 0) == 0) {
            const int slot = std::atoi(command.c_str() + 6);
            if (concurrent) {
                concurrent_state.cancel_pending = true;
            } else if (!retained_schedule) {
                resident.cancel(slot);
                active = false;
            } else if (slot >= 0 && slot < (int)slot_active.size()) {
                slot_active[(size_t)slot] = false;
                cancelled_group[(size_t)slot / 2] = !group_active(slot / 2);
            }
            ++result.cancel_controls;
        }
    };
    const auto generation_controls = [&](bool consume_yield) {
        for (;;) {
            std::string control;
            {
                std::lock_guard<std::mutex> lock(in_mu);
                if (!strata::program::pop_control_prefix(in_lines, control, consume_yield)) break;
            }
            resident_control(control);
        }
    };
    const auto executor_retire = [&](int) {
        if (in_prompt_read || pipe_inflight() || !result.drain_observed) ++result.fence_violations;
        return true;
    };
    const auto park_replaced = [&](int) {
        if (retained_running || pipe_inflight() || bounded_round) ++result.fence_violations;
        return true;
    }; // GPU snapshotting is outside this CPU control-flow seam.
    const int admit_slot = -1;
    const std::vector<int> req_imgs;
    const std::vector<ConvCheckpoint> checks;
    const int64_t read_from = begin_at;
    const bool resumed_from0 = false, trace = false;
    struct Slot { bool failed = false; };
    const std::vector<Slot> bs(4);
    // The compiled production loop names these store authorities, but this
    // scheduling fixture never authorizes disk/store ownership changes.
    const bool store_on = false;
    constexpr uint64_t kSessionOwner = 0, kResidentOwnerFirst = 1;
    struct StoreOwner {
        bool force_fresh = false;
        phase::Identity phase_producer;
        void borrow() { throw std::logic_error("unexpected store borrow"); }
    } no_store;
    const auto store_owner = [&](uint64_t) -> StoreOwner& { return no_store; };

    // Disk publication is exercised by the real-engine persistence qualification, not this scheduling fixture.
    const auto store_seam = [&] {
        if (retained_prefill && time_budget.enabled() &&
            (retained_running || pipe_inflight() || bounded_round)) ++result.fence_violations;
        result.store_positions.push_back(sp.position);
    };

    // CMake extracts this verbatim from generate.cpp and regenerates it whenever that source changes.
    // Ambient names the extracted region reads from read_prompt's enclosing scope in generate.cpp (--prefill-pipe's
    // resume plan and the last lend's chunk).  This harness plans and resumes nothing, so they stay empty and the
    // extracted code takes the unplanned path the assertions below are about.
    ConvCheckpoint::PromptPlan resume_plan, seg_plan;
    int64_t resume_plan_at = -1, resume_plan_end = -1;
    int64_t seg_lend = 0;
    bool capture_device_fault = false;   // #252: the park source's own class, set by a failed device sync
    const bool conc_env = concurrent && concurrent->enabled;
    const bool private_preparation = false;
    const bool batch_mtp = concurrent && concurrent->mtp;
    const bool batch_mtp_pipeline = batch_mtp && concurrent->pipeline;
    strata::program::ExecutionPlanInput plan_input;
    plan_input.serve = plan_input.prompt_cache = true;
    plan_input.batch_requested = batch;
    plan_input.piped_concurrent = conc_env;
    plan_input.batch_mtp = batch_mtp;
    plan_input.batch_mtp_pipeline = batch_mtp_pipeline;
    plan_input.mtp_configured = batch_mtp;
    plan_input.foreground_budget_ms = budget_ms;
    const auto execution_plan = strata::program::resolve_execution_plan(plan_input);
    strata::program::ConcurrentReadSetup concurrent_read_setup;
    concurrent_read_setup.other_source = concurrent && concurrent->source_refused;
    std::condition_variable in_cv;
    bool use_mtp = batch_mtp;
    struct MtpStub {
        Result& result;
        bool reserve_prefill(int64_t, std::string&) { ++result.mtp_reservations; return true; }
    } mtp{result};
    const int n_pipe = 1;
    // (only passed through to the stub below; the real PoolMultiFn would pull the CUDA headers into this CPU harness)
    const int win_pool_fn = 0;
    void* win_pool_user = &concurrent_state.pool[0];
    auto group_pool_user = [&](int owner) -> void* { return &concurrent_state.pool[(size_t)owner]; };
    auto stage_verifier = [&](int) -> ReaderVerifier& { return concurrent_state.verifier[0]; };
    auto stage_peer = [&](int) -> ReaderVerifier& { return concurrent_state.verifier[1]; };
    auto stage_third = [&](int) -> ReaderVerifier& { return concurrent_state.verifier[2]; };
    auto pump = [&](bool may_start) { return concurrent ? concurrent_pump(may_start, false) : true; };
    auto pipeline_step = [&](bool may_start) {
        return batch_mtp_pipeline ? pump_mtp(may_start) : concurrent ? concurrent_pump(may_start, batch_mtp) : pump(may_start);
    };
    std::mutex bell_mu;
    std::condition_variable bell_cv;
    std::atomic<bool> bell_waiting{false};
    bool conc_serving = false;
    // This fixture restores legacy scheduling coverage; native capture has its own lifecycle fixture.
    const bool use_dflash = false;
    int64_t pp_reached = begin_at;
    bool pp_exact = true;
    struct { struct { uint64_t request_generation = 1; } d; } drive;
    const auto report_stage_state = [](const char*, uint64_t, int64_t, int, std::string&) { return true; };
#include "prefill_read_part.inc"

    std::string& e = err;
    result.okay = read_part(begin_at, n, e);
    result.fatal = batch_fatal;
    result.error = e;
    if (concurrent) {
        result.concurrent_chunks = concurrent_state.chunks;
        for (int probe = 0; probe <= 1000; ++probe)
            if (time_budget.decode_allowed(probe)) ++result.credit_after_overlap;
        credit_service(); // the production request-scope destructor also runs after the read
        for (int probe = 0; probe <= 1000; ++probe)
            if (time_budget.decode_allowed(probe)) ++result.credit_after_exit;
        result.external_balanced = true;
        for (size_t owner = 0; owner < concurrent_state.verifier.size(); ++owner) {
            const auto& v = concurrent_state.verifier[owner];
            result.doorbell_served[owner] = v.served;
            result.external_on[owner] = v.on;
            result.external_off[owner] = v.off;
            result.external_balanced &= !v.external && v.on == v.off;
            result.doorbell_context_errors += v.wrong_pool;
        }
    }
    if (!concurrent && !retained_prefill && !expects_yield && !cancel_first_launch &&
        (!result.okay || !e.empty() || batch_fatal || in_prompt_read || yielded_at != -1 || yielded_slot != -1))
        throw std::runtime_error("read_part did not finish: " + e);
    result.interleave_boundaries = il_parts;
    result.estimate_ms = time_budget.chunk_ms();
    result.parked_at = resident.parked_at;
    result.parked_split_until = resident.prefill_split_until;
    result.parked_prefix = resident.parked_prefix;
    result.final_frontier = sp.position;
    result.final_blocked_stage = retained_blocked_stage;
    result.final_bounded_round = bounded_round;
    result.final_inflight = pipe_inflight();
    result.decode_credit = time_budget.decode_allowed(0);
    result.credited_ms = credited_prefill_ms;
    result.resident_ms = resident_round_ms;
    result.decode_estimate_ms = time_budget.decode_ms();
    result.prefill_phase_observed = phase_leg.totals[size_t(phase::Stage::prefill_run)].count > 0;
    result.copyin_phase_observed = phase_leg.totals[size_t(phase::Stage::resident_copyin)].count > 0;
    result.decode_phase_unavailable = !phase_leg.totals[size_t(phase::Stage::decode_interleave)].known;
    result.parked_identity_preserved = no_store.phase_producer.valid() &&
        std::strcmp(no_store.phase_producer.request, phase_leg.input.identity.request) == 0 &&
        std::strcmp(no_store.phase_producer.incarnation, phase_leg.input.identity.incarnation) == 0 &&
        no_store.phase_producer.leg == phase_leg.input.identity.leg &&
        no_store.phase_producer.priority == phase_leg.input.identity.priority;
    result.decode_age_ms = std::chrono::duration<double, std::milli>(Clock::now() - last_decode).count();
    return result;
}

static Result read_retained(const RetainedSchedule& schedule, const RetainedOptions& o = {}) {
    return read_prompt(o.budget_ms, true, true, o.active, 4, "", o.drain_ok, o.tokens, o.prior_chunk_ms,
                       o.yield, -1, false, false, 2048, o.begin_at, o.pending_split_until, &schedule);
}

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

static void test_retained_budget() {
    // A complete bounded callback quantum is settled once at the idle seam.
    // Moving this reviewed policy fixture here keeps the change in program scope.
    strata::program::TimeBudget retained(100, .25);
    retained.prefilled(60, 1, 2048);
    retained.decoded(250);
    CHECK(!retained.decode_allowed(0) && retained.decode_ms() == 250);
    retained.background_work(40);
    CHECK(!retained.decode_allowed(0));
    retained.background_work(10);
    CHECK(!retained.decode_allowed(0));
    retained.background_work(1);
    CHECK(retained.decode_allowed(0) && !retained.decode_allowed(3));
    retained.decoded(1000, false);
    CHECK(retained.decode_allowed(0));
    const auto idle = [](const Result& result) {
        return result.final_blocked_stage == -1 && !result.final_bounded_round && !result.final_inflight &&
               result.ownership_violations == 0 && result.exclusion_violations == 0 &&
               result.fence_violations == 0 && result.budget_mutations == 0;
    };
    const RetainedSchedule serial;
    const auto guarded = read_retained(serial);
    CHECK(guarded.okay && !guarded.fatal && idle(guarded));
    CHECK(guarded.prefill_phase_observed && guarded.decode_phase_unavailable);
    CHECK((guarded.retained_starts == std::vector<std::array<int, 2>>{{1, 1}}));
    CHECK(!guarded.retained_start_clocks.empty() && guarded.retained_start_clocks.front() == 40);
    CHECK(guarded.decode_positions.empty() && guarded.seam_rounds.empty());
    CHECK((guarded.completed_rounds == std::array<int, 2>{1, 1}));
    CHECK(guarded.estimate_ms == 60 && guarded.credited_ms == 60 && guarded.resident_ms == 30);
    CHECK(guarded.decode_estimate_ms == 30 && guarded.decode_credit);
    CHECK((guarded.store_positions == std::vector<int64_t>{2048}));
    RetainedSchedule long_stage = serial;
    long_stage.stage_ms = 1000;
    const auto old_decode = read_retained(long_stage);
    CHECK(old_decode.okay && idle(old_decode) && old_decode.decode_age_ms == 1000);

    // Cold/idle chunks cannot invent credit or decoders; a retained run never
    // adopts the ordinary idle 4096 fill. Tails do not train the 2048 estimate.
    const auto cold = read_retained(serial, {.prior_chunk_ms = 0});
    CHECK(cold.okay && idle(cold) && cold.completed_stages.empty() && cold.decode_credit && cold.estimate_ms == 60);
    const auto solo = read_retained(serial, {.prior_chunk_ms = 0, .tokens = 4097, .active = false});
    CHECK(solo.okay && idle(solo) && solo.completed_stages.empty() && solo.decode_positions.empty());
    CHECK((solo.reads == std::vector<std::pair<int64_t, int64_t>>{{0, 2048}, {2048, 2048}, {4096, 1}}));
    CHECK(solo.drain_observed && solo.estimate_ms == 60);

    // Existing checkpoint-part and full-checkpoint callbacks occur inside the
    // bulk call, so their chunks use global closure instead of stage loans.
    RetainedSchedule checkpoint = serial;
    checkpoint.checkpoint_every = 4096;
    const auto snapshots = read_retained(checkpoint, {.tokens = 4096});
    CHECK(snapshots.okay && idle(snapshots));
    CHECK((snapshots.retained_starts == std::vector<std::array<int, 2>>{{1, 1}, {0, 0}}));
    CHECK((snapshots.checkpoint_positions == std::vector<int64_t>{4096, 4096, 4096}));

    // Independent stage latencies leave peers live at local handoff and at the
    // final seam. Both completion traversal orders must preserve all work.
    for (bool reverse : {false, true}) {
        RetainedSchedule delayed = serial;
        delayed.completion_polls = {{{{2, 3, 5}}, {{3, 12, 9}}}};
        delayed.reverse_completions = reverse;
        const auto result = read_retained(delayed);
        CHECK(result.okay && idle(result));
        CHECK((result.completed_rounds == std::array<int, 2>{1, 1}));
        CHECK((result.retained_starts == std::vector<std::array<int, 2>>{{1, 1}}));
        CHECK(!result.reacquired_with_peer.empty() && !result.drained_stages.empty());
        CHECK(std::any_of(result.completions_under_exclusion.begin(), result.completions_under_exclusion.end(),
                          [](const auto& event) { return event[0] == 1 && event[1] == 0 && event[2] == 1; }));
        for (int stage = 0; stage < 3; ++stage)
            for (int group = 0; group < 2; ++group)
                CHECK(std::count(result.completed_stages.begin(), result.completed_stages.end(),
                                 std::make_pair(stage, group)) == 1);
        // These independent consumers become ready on the same poll. Invert
        // their completion order without changing the set of completed work.
        const auto peer = std::find(result.completed_stages.begin(), result.completed_stages.end(), std::make_pair(0, 1));
        const auto local = std::find(result.completed_stages.begin(), result.completed_stages.end(), std::make_pair(1, 0));
        CHECK((peer < local) == !reverse);
        CHECK((result.store_positions == std::vector<int64_t>{2048}));
    }

    // In-chunk service must not replace the ordinary seam's additional complete
    // rounds. A long bulk-only suffix makes the seam due; each paid round must
    // finish both groups, rather than merely advance the synthetic clock.
    const auto seam = read_retained(long_stage, {.tokens = 4096});
    CHECK(seam.okay && idle(seam) && seam.seam_rounds.size() >= 2);
    CHECK(std::all_of(seam.seam_rounds.begin(), seam.seam_rounds.end(),
                      [](const auto& round) { return round == std::array<int, 2>{1, 1}; }));
    CHECK(!seam.decode_positions.empty() &&
          std::all_of(seam.decode_positions.begin(), seam.decode_positions.end(), [](int64_t q) { return q == 2048; }));

    // An indivisible slow round carries its entire measured service as debt.
    // The next chunk must repay it, not restart or miscredit it as prefill.
    RetainedSchedule slow = serial;
    slow.pump_ms = 100;
    const auto debt = read_retained(slow, {.tokens = 4096});
    CHECK(debt.okay && idle(debt) && !debt.decode_credit && debt.decode_positions.empty() && debt.seam_rounds.empty());
    CHECK((debt.retained_starts == std::vector<std::array<int, 2>>{{1, 1}, {0, 0}}));
    CHECK(debt.credited_ms == 120 && debt.resident_ms == 600 && debt.estimate_ms == 60);

    // STOP and contiguous BSTOP make progress at the existing stage callbacks.
    // A queued admission must still prevent a later slot-number cancellation.
    RetainedSchedule stopped = serial;
    stopped.stop_boundary = 3;
    const auto cancelled = read_retained(stopped);
    CHECK(!cancelled.okay && !cancelled.fatal && cancelled.error == "cancelled" && idle(cancelled));
    CHECK(cancelled.store_positions.empty() && cancelled.final_frontier == 0);
    RetainedSchedule resident_stop = serial;
    resident_stop.cancel_boundary = 4; // slot 0 already owns a stage; slot 2 is independent
    const auto stop = read_retained(resident_stop, {.tokens = 4096});
    CHECK(stop.okay && idle(stop) && stop.cancel_controls > 0);
    CHECK((stop.starts_after_cancel == std::array<int, 2>{0, 0}));
    CHECK(stop.retained_starts.size() == 2 && stop.retained_starts[0][0] == 1 && stop.retained_starts[1][0] == 0);
    CHECK(stop.completed_rounds[0] == 1 && stop.peer_completions_after_cancel[1] > 0);
    // This proves the caller keeps pumping the peer and does not restart the
    // cancelled group. Actual slot-generation/output fencing remains GPU-only.
    resident_stop.admission_before_cancel = true;
    const auto ordered = read_retained(resident_stop);
    CHECK(ordered.okay && idle(ordered) && ordered.cancel_controls == 0);
    CHECK((ordered.completed_rounds == std::array<int, 2>{1, 1}));

    const auto yielded = read_retained(serial, {.tokens = 10000, .yield = true});
    CHECK(!yielded.okay && !yielded.fatal && yielded.error == "yield" && idle(yielded));
    CHECK(yielded.parked_at == 2048 && yielded.final_frontier == yielded.parked_at && yielded.parked_split_until == 0);
    CHECK(yielded.copyin_phase_observed && yielded.parked_identity_preserved);
    CHECK(yielded.parked_prefix.size() == (size_t)yielded.parked_at);
    for (size_t i = 0; i < yielded.parked_prefix.size(); ++i) CHECK(yielded.parked_prefix[i] == (int32_t)i);
    const auto resumed = read_retained(serial, {.tokens = 10000, .begin_at = yielded.parked_at,
                                               .pending_split_until = yielded.parked_split_until});
    const auto uninterrupted = read_retained(serial, {.tokens = 10000});
    CHECK(resumed.okay && idle(resumed) && resumed.final_frontier == 10000 && resumed.parked_split_until == 0);
    auto joined = yielded.reads;
    joined.insert(joined.end(), resumed.reads.begin(), resumed.reads.end());
    CHECK(joined == uninterrupted.reads);
    int64_t frontier = 0;
    for (const auto& range : joined) {
        CHECK(range.first == frontier && range.second > 0);
        frontier += range.second;
    }
    CHECK(frontier == 10000);

    // Failure while the local exclusion is live resets it before the main
    // owner's full drain. Error-string aliasing must not turn failure into STOP.
    RetainedSchedule broken = serial;
    broken.fail_pump_call = 4;
    const auto failure = read_retained(broken);
    CHECK(!failure.okay && failure.fatal && failure.error == "retained pump failed" && idle(failure));
    CHECK(!failure.drained_stages.empty() && failure.store_positions.empty());
    CHECK((failure.completed_rounds == std::array<int, 2>{1, 1}));
    const auto drain = read_retained(serial, {.drain_ok = false});
    CHECK(!drain.okay && drain.fatal && drain.error == "descendant drain failed" && drain.store_positions.empty());
    CHECK(drain.final_blocked_stage == -1 && !drain.final_bounded_round);
    RetainedSchedule foreign = serial;
    foreign.foreign_callback = true;
    const auto refused_owner = read_retained(foreign);
    CHECK(!refused_owner.okay && refused_owner.fatal && idle(refused_owner));
    CHECK(refused_owner.error == "retained prefill: budget pump owner changed" && refused_owner.completed_stages.empty());
    const auto split = read_retained(serial, {.tokens = 4097, .begin_at = 1024, .pending_split_until = 2048});
    CHECK(!split.okay && split.reads.empty() && split.error == "retained prefill: cannot resume a split 2048 chunk");

    for (const char* error : {"cancelled", "prefill: retained cancellation cleanup failed", "partial publication failed"}) {
        const auto terminal = read_prompt(1000, true, true, true, 4, error);
        CHECK(!terminal.okay && terminal.fatal == (std::strcmp(error, "cancelled") != 0));
        CHECK(terminal.error == error && terminal.drain_observed && idle(terminal));
    }
    const auto legacy = read_retained(serial, {.budget_ms = 0, .prior_chunk_ms = 0, .active = false});
    CHECK(legacy.okay && legacy.drain_observed && legacy.estimate_ms == 0 && legacy.credited_ms == 0);
}

static void test_concurrent_reader() {
    const auto run = [](const ConcurrentSchedule& schedule, double budget = 1000, bool debt = false, bool drain = true) {
        return read_prompt(budget, true, true, true, 4, nullptr, drain, 16384, 60,
                           false, -1, debt, false, 2048, 0, 0, nullptr, &schedule);
    };
    for (bool pipeline : {false, true}) {
        const ConcurrentSchedule on{.pipeline = pipeline, .third = pipeline};
        const auto overlap = run(on);
        CHECK(overlap.okay && !overlap.fatal && overlap.final_frontier == 16384);
        CHECK(overlap.concurrent_chunks == 8 && overlap.concurrent_starts > 0 && overlap.plain_pumps == 0);
        CHECK(overlap.mtp_reservations == 1 && overlap.drain_observed && !overlap.final_inflight);
        CHECK(overlap.uncredited_starts == 0 && overlap.first_concurrent_start_ms < 1000);
        CHECK(overlap.estimate_ms == 60 && overlap.decode_estimate_ms == 0);
        CHECK(overlap.credit_after_exit == overlap.credit_after_overlap && overlap.request_scope_credit_ms == 0);
        CHECK(overlap.external_balanced && overlap.doorbell_context_errors == 0);
        for (int owner = 0; owner < (pipeline ? 3 : 1); ++owner) {
            CHECK(overlap.doorbell_served[(size_t)owner] > 0);
            CHECK(overlap.external_on[(size_t)owner] == 1 && overlap.external_off[(size_t)owner] == 1);
        }
        const auto debt = run(on, 1000, true);
        CHECK(debt.okay && debt.concurrent_starts > 0 && debt.uncredited_starts == 0);
        CHECK(debt.decode_estimate_ms == 5 && debt.estimate_ms == 60);
        CHECK(debt.credit_after_exit == debt.credit_after_overlap && debt.request_scope_credit_ms == 0);
        const auto no_budget = run(on, 0);
        CHECK(no_budget.okay && no_budget.concurrent_starts > 0 && no_budget.external_balanced);

        auto off = on;
        off.enabled = false;
        const auto control = run(off);
        const auto sequential = read_prompt(1000, true, true, true, 4, nullptr, true, 16384, 60);
        CHECK(control.okay && control.concurrent_chunks == 0 && control.concurrent_starts == 0);
        CHECK(control.reads == sequential.reads && control.decode_positions == sequential.decode_positions);
        CHECK(control.estimate_ms == sequential.estimate_ms && control.mtp_reservations == 0);
        off = on;
        off.source_refused = true;
        const auto gated = run(off);
        CHECK(gated.okay && gated.reads == sequential.reads && gated.concurrent_chunks == 0);
    }
    ConcurrentSchedule plain;
    plain.mtp = plain.pipeline = false;
    const auto legacy_overlap = run(plain);
    CHECK(legacy_overlap.okay && legacy_overlap.plain_pumps > 0 && legacy_overlap.mtp_reservations == 0);

    ConcurrentSchedule yielded;
    yielded.yield = true;
    const auto parked = run(yielded);
    CHECK(!parked.okay && !parked.fatal && parked.error == "yield");
    CHECK(parked.parked_at > 0 && parked.parked_at < 16384 && parked.parked_at % 2048 == 0);
    CHECK(parked.parked_at == parked.final_frontier && parked.drain_observed && !parked.final_inflight);
    CHECK(parked.external_balanced && parked.fence_violations == 0);
    ConcurrentSchedule stopped;
    stopped.cancel = true;
    const auto cancelled = run(stopped);
    CHECK(!cancelled.okay && !cancelled.fatal && cancelled.error == "cancelled");
    CHECK(cancelled.final_frontier == 2048 && cancelled.external_balanced && cancelled.drain_observed);
    CHECK(!cancelled.final_inflight && cancelled.credit_after_exit == cancelled.credit_after_overlap);
    ConcurrentSchedule resident_stop;
    resident_stop.cancel_resident = true;
    const auto retired = run(resident_stop);
    CHECK(retired.okay && retired.cancel_controls == 1 && retired.retirement_decode_ms >= 1000);
    CHECK(retired.starts_after_retirement == 0 && !retired.decode_credit);
    CHECK(retired.decode_estimate_ms == 0 && retired.external_balanced && !retired.final_inflight);
    for (int failure = 0; failure < 3; ++failure) {
        ConcurrentSchedule failed;
        failed.fail_read = failure == 0;
        failed.fail_pump = failure == 1;
        failed.fail_retire = failure == 2;
        const auto terminal = run(failed);
        CHECK(!terminal.okay && (failure == 0 || terminal.fatal));
        CHECK(terminal.drain_observed && !terminal.final_inflight && terminal.external_balanced);
        CHECK(terminal.credit_after_exit == terminal.credit_after_overlap);
    }
    const auto failed_drain = run({}, 1000, false, false);
    CHECK(!failed_drain.okay && failed_drain.fatal && failed_drain.external_balanced && failed_drain.drain_observed);
}

int main() {
    split_setting(nullptr); // Ambient experiment settings must not change the legacy regressions.
    test_retained_budget();
    test_concurrent_reader();
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
    CHECK(!cancelled.okay && !cancelled.fatal && cancelled.error == "cancelled" && cancelled.drain_observed);
    for (const char* failure : {"prefill: retained cancellation cleanup failed", "partial publication failed"}) {
        const auto failed = read_prompt(0, true, true, true, 4, failure);
        CHECK(!failed.okay && failed.fatal && failed.error == failure && failed.drain_observed);
    }
    const auto failed_drain = read_prompt(0, true, true, true, 4, "cancelled", false);
    CHECK(!failed_drain.okay && failed_drain.fatal && failed_drain.error == "descendant drain failed" && failed_drain.drain_observed);
    const auto retained = read_prompt(0, true, true, true, 4, "");
    CHECK(retained.okay && !retained.fatal && retained.error.empty() && retained.drain_observed && retained.reads == chunks);
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
    CHECK(midpoint_yield.parked_split_until == 2048);
    // A parked midpoint is a nonzero-frontier continuation, not a new chunk
    // origin. Its first 1024 must finish the original [0,2048), even if the
    // estimate is cold by the time this request resumes.
    const std::vector<std::pair<int64_t, int64_t>> resumed_cold{
        {0, 1024}, {1024, 1024}, {2048, 2048}, {4096, 1}};
    for (bool piped : {false, true}) {
        const auto first = read_prompt(1000, piped, piped, true, 4, nullptr, true, 4097, 1200, true);
        CHECK(first.parked_at == 1024 && first.parked_split_until == 2048);
        for (double estimate : {0.0, 1200.0}) {
            const auto next = read_prompt(1000, piped, piped, true, 4, nullptr, true, 4097, estimate,
                                          false, -1, false, false, 2048, first.parked_at, first.parked_split_until);
            auto joined = first.reads;
            joined.insert(joined.end(), next.reads.begin(), next.reads.end());
            CHECK(next.okay && joined == (estimate == 0 ? resumed_cold : split_tail));
            CHECK(next.reads.front().first == 1024 && next.reads.front().second == 1024);
            CHECK(next.parked_split_until == 0); // the continuation is consumed, not retained for a later request
        }
    }
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
    if (fails == 0) std::printf("prefill_interleave_test: ok (retained ownership/budget, interleave, opt-in split, legacy)\n");
    return fails == 0 ? 0 : 1;
}
