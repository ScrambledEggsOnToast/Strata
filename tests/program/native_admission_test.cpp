// Execute generate.cpp's native_select verbatim. Only resource/capture sinks are
// synthetic: this proves poison-reset/affinity/reservation control-flow ordering,
// not CUDA state exactness (the protected real-engine vehicle covers resources).
#include "strata/program/native_admission.hpp"
#include "strata/program/checkpoint_preservation.hpp"
#include "strata/program/phase_receipts.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <deque>
#include <cstdio>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using Clock = std::chrono::steady_clock;
namespace phase = strata::program::phase;
struct Scope { bool valid() const { return false; } };
struct Request { int scope = 1; bool steering = true; Scope shared_scope; int64_t pin = 0; };
namespace strata::program {
struct ResidentPreparation { int slot = -1; uint64_t request = 0, incarnation = 0; };
struct ResidentPrepareInput { int slot = -1; const std::vector<int64_t>* tokens = nullptr; Request request; };
}
struct Leg {
    Leg(int, uint64_t, uint64_t) {}
    int slot = -1;
    uint64_t shared_prefix_tokens = 0;
};
struct Owner {
    std::string tag;
    Request input;
    std::vector<int64_t> tokens;
    uint64_t submission_sequence = 0, admit_sequence = 0;
    bool foreground = true, solo = false, input_ready = false, pending = false;
    const char* cancel_reason = nullptr;
    Clock::time_point accepted = Clock::now(), deadline = accepted + std::chrono::seconds(10), started, service_ready;
    double admission_wait_s = 0, service_wait_s = 0;
    strata::program::ResidentPreparation lease;
    bool work = false;
    std::array<int, 4> seams{};
    int phase_input = 0;
    std::optional<Leg> phase_leg;
};
struct Seat {
    std::vector<int64_t> ids{11, 22};
    bool cached = true;
    bool* poison = nullptr;
    bool* affinity_saw_poison = nullptr;
    bool reusable(int, bool) const {
        *affinity_saw_poison |= *poison;
        return cached;
    }
};
struct Resources {
    Seat& seat;
    bool reserved = false;
    std::vector<int64_t> retained_source;
    bool available(int) const { return !reserved; }
    bool reserve(const strata::program::ResidentPrepareInput* input, int count,
                 strata::program::ResidentPreparation* lease, std::string&) {
        if (count != 1 || reserved) return false;
        // Like ResidentSlots::reserve: source selection is copied now, not at prepare.
        if (seat.cached) retained_source = seat.ids;
        *lease = {input->slot, 1, 1};
        reserved = true;
        return true;
    }
    void priority(int, bool) {}
};

static int failures = 0;
static void check(bool okay, const char* message) {
    if (!okay) { std::fprintf(stderr, "FAIL: %s\n", message); ++failures; }
}
static void selection_case(bool poisoned, bool matching) {
    const bool retained_prompt = true, piped = false, phase_receipts = false;
    bool nan_flush = poisoned, affinity_saw_poison = false, capture_saw_poison = false;
    int resets = 0;
    std::array<Seat, 1> bs;
    bs[0].poison = &nan_flush;
    bs[0].affinity_saw_poison = &affinity_saw_poison;
    Resources resident{bs[0], false, {}};
    struct { int batch_groups = 1, turn_token = -1; } o;
    constexpr size_t packed_seats = 1;
    std::array<Owner, 2> packed_submissions;
    std::span<Owner> packed_owners(packed_submissions.data(), packed_seats);
    auto& waiting = packed_submissions[1];
    waiting.tag = "waiting";
    waiting.submission_sequence = 1;
    waiting.input_ready = true;
    waiting.tokens = matching ? std::vector<int64_t>{11, 22, 33} : std::vector<int64_t>{44, 55, 66};
    strata::program::NativeAdmissionPolicy native_policy;
    uint64_t native_seat_used[1]{}, packed_admit_sequence = 0, phase_sequence = 0;
    auto packed_poison_reset = [&] {
        if (!nan_flush) return;
        ++resets;
        nan_flush = false;
        bs[0].cached = false;
        bs[0].ids.clear();
    };
    auto park_completed = [&](int, int, const void*) {
        capture_saw_poison |= nan_flush;
        return true;
    };
    auto capture_failure = [] { return strata::program::PreservationFailure::durability; };
    auto drop_capture = [](const char*) {};
    auto packed_terminal = [](Owner& owner, int, const char*) { owner = {}; };
    struct Executor {
        int submissions = 0;
        bool submit(const strata::program::ResidentPreparation& lease, int, std::string&) {
            ++submissions;
            return lease.request && lease.incarnation;
        }
    };
    std::array<Executor, 1> resident_executors;
    const int execution_plan = 0;
    std::string err;
    auto executor_retire = [](int) { return true; };
#include "native_select.inc"
    check(native_select(), "native selection completes at its owner safe point");
    check(resident.reserved && packed_owners[0].pending && packed_submissions[1].tag.empty(),
          "the same waiting submission receives exactly one physical lease");
    check(!affinity_saw_poison && !capture_saw_poison,
          "poison reset precedes every native affinity lookup and old-resource capture");
    check(resident.retained_source.size() == (poisoned ? 0U : 2U),
          "a post-poison reservation cannot copy a retained source before invalidation");
    check(resets == (poisoned ? 1 : 0), "poison reset occurs once, and healthy cache reuse is unchanged");
    check(packed_owners[0].work && resident_executors[0].submissions == 1,
          "native selection binds its reserved generation to exactly one executor call scope");
}
static void control_case() {
    std::deque<std::string> in_lines;
    const auto enqueue = [&](const std::string& incoming) {
        const std::string_view text(incoming);
        const size_t tag_end = text.find(' ', text.find(' ') + 1);
#include "native_control_coalesce.inc"
        in_lines.push_back(incoming);
    };
    const std::string cancel = "BQCANCEL victim 0 17 3";
    const std::string priority = "BQPRIORITY victim 0 17 3 foreground";
    const std::string resume = "BQRESUME victim 0 17 3 32 1 1 0 128 32 0";
    for (const auto& valid : {cancel, priority, resume}) {
        in_lines = {valid, "BQREGISTER peer foreground eligible 100000 4"};
        auto stale = valid;
        stale.replace(stale.find(" 17 3"), 5, " 17 2");
        enqueue(stale);
        check(std::find(in_lines.begin(), in_lines.end(), valid) != in_lines.end(),
              "a late incarnation cannot replace an unconsumed valid logical control");
    }
    in_lines = {cancel};
    enqueue("BQCANCEL victim 1 17 3");
    check(std::find(in_lines.begin(), in_lines.end(), cancel) != in_lines.end(),
          "a stale physical hint cannot overwrite valid cancellation");
    in_lines = {resume};
    enqueue("BQRESUME victim 0 17 3 64 1 1 0 128 32 0");
    check(std::find(in_lines.begin(), in_lines.end(), resume) != in_lines.end(),
          "a stale retained cursor cannot overwrite a valid resume");
    in_lines = {priority};
    enqueue("BQPRIORITY victim 0 17 3 background");
    check(in_lines.size() == 1 && in_lines.front().ends_with("background"),
          "same-lease priorities remain a bounded latest intention");
    in_lines = {cancel};
    enqueue(cancel);
    check(in_lines.size() == 1, "identical cancellation is coalesced without dropping its intention");
}
static void decoder_boundary_case(bool retained_prompt, bool previously_ready, bool newly_ready,
                                  bool cancel, bool pending, bool suspended, bool allowed) {
    struct { bool suspended; } cursor{suspended};
    struct { decltype(cursor) prompt_cursor; } state{cursor};
    std::array<decltype(state), 1> bs{state};
    struct { bool pending; strata::program::ResidentPreparation lease; } owner{pending, {0, 1, 1}};
    std::array<decltype(owner), 1> packed_owners{owner};
    bool ready = previously_ready;
    // This is the decision made before final-token verification. first_token()
    // activates a previously preparing owner only after that transaction drains.
    const bool decoder_slice_due = retained_prompt && ready;
    ready = ready || newly_ready;
    int collections = 0, rounds = 0;
    bool charged = false;
    auto packed_collect = [&] { ++collections; if (cancel) ready = false; return true; };
    auto packed_decoding = [&] { return ready; };
    struct {
        bool allowed;
        bool enabled() const { return true; }
        bool decode_allowed(int) const { return allowed; }
    } time_budget{allowed};
    auto decode_round = [&](bool charge) { ++rounds; charged = charge; return true; };
#include "native_decoder_slice.inc"
    check(decoder_slice(), "a drained prompt boundary succeeds");
    const bool charge = !retained_prompt || (pending && !suspended);
    const bool expected = retained_prompt && (previously_ready || newly_ready) && !cancel && (!charge || allowed);
    check(rounds == int(expected),
          "the drained boundary serves a newly active decoder once without waiting for another prompt quantum");
    check(!rounds || charged == charge, "suspended peers retain the existing decode-budget charge policy");
    check(!retained_prompt || collections == 1, "retained boundary consumes cancellation before readiness");
}

static void decoder_boundary_cases() {
    decoder_boundary_case(true, false, true, false, true, true, true);
    decoder_boundary_case(true, true, true, false, true, false, true);
    decoder_boundary_case(true, false, false, false, true, true, true);
    decoder_boundary_case(true, false, true, true, true, true, true);
    decoder_boundary_case(true, false, true, false, true, false, false);
    decoder_boundary_case(true, false, true, false, true, true, false);
    decoder_boundary_case(true, false, true, false, false, false, false);
    decoder_boundary_case(false, false, true, false, true, false, true);
}


int main() {
    selection_case(false, true);
    selection_case(true, true);
    selection_case(true, false);
    control_case();
    decoder_boundary_cases();
    if (!failures) std::puts("native_admission_test: all passed (real selector, synthetic resource sinks)");
    return failures ? 1 : 0;
}