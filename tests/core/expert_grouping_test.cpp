// HET-038: the cross-request expert grouping seam.  The dispatch derives each request's verification span
// from the per-row metadata before anything is claimed, and every result row belongs to exactly one
// (request, layer, speculative position).  These tests pin that seam's behavior: span derivation and its
// fail-closed identity refusals, the bounded merged/per-request decision, and exactly-once ownership with
// a cancelled member, on the real ledger and helper scheduler.  No CUDA, no engine, no wiring echoes.
#include "strata/core/expert_worker.hpp"

#include <stdexcept>
#include <array>
#include <vector>

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

strata::core::ExpertGroupPlan analyze(const std::vector<int64_t>& positions,
                                      const std::vector<int>& slots,
                                      const std::vector<uint64_t>& requests, int64_t k,
                                      const strata::core::ExpertGroupBounds& bounds,
                                      std::vector<strata::core::ExpertGroupSpan>& spans,
                                      const char*& error) {
    spans.resize(8);
    int32_t n = 0;
    const auto plan = strata::core::analyze_expert_groups(positions.data(), slots.data(), requests.data(),
                                                          (int64_t) positions.size(), k, bounds,
                                                          spans.data(), n, error);
    spans.resize((size_t) n);
    return plan;
}

void two_request_spans_group_into_one_window() {
    // slot 2 holds a two-row verification span, slot 5 a single row: the batch-MTP shape.
    const std::vector<int64_t> positions = {40, 41, 17};
    const std::vector<int> slots = {2, 2, 5};
    const std::vector<uint64_t> requests = {101, 101, 202};
    std::vector<strata::core::ExpertGroupSpan> spans;
    const char* error = nullptr;
    const auto plan = analyze(positions, slots, requests, 10, {}, spans, error);
    require(plan == strata::core::ExpertGroupPlan::merged, "two requests' spans must merge into one group");
    require(error == nullptr, "a valid window names no error");
    require(spans.size() == 2, "the window holds two request spans");
    require(spans[0].request == 101 && spans[0].slot == 2 && spans[0].first_position == 40 &&
            spans[0].first_row == 0 && spans[0].rows == 2, "the first request's span is its two rows");
    require(spans[1].request == 202 && spans[1].slot == 5 && spans[1].first_position == 17 &&
            spans[1].first_row == 2 && spans[1].rows == 1, "the second request's span is its one row");
}

void solo_window_is_one_span() {
    // A solo verify window: T consecutive speculative positions, no slots, one dispatch generation.
    const std::vector<int64_t> positions = {9, 10, 11, 12};
    const std::vector<int> slots(4, -1);
    const std::vector<uint64_t> requests(4, 7);
    std::vector<strata::core::ExpertGroupSpan> spans;
    const char* error = nullptr;
    const auto plan = analyze(positions, slots, requests, 10, {}, spans, error);
    require(plan == strata::core::ExpertGroupPlan::merged, "a solo window merges into one span");
    require(spans.size() == 1 && spans[0].rows == 4 && spans[0].slot == -1 && spans[0].first_position == 9,
            "the solo window's span is its whole verification span");
}

void mixed_bound_and_unbound_identities_refused() {
    const std::vector<int64_t> positions = {0, 1};
    const std::vector<int> slots = {0, 1};
    const std::vector<uint64_t> requests = {55, 0};   // one row lost its published request identity
    std::vector<strata::core::ExpertGroupSpan> spans;
    const char* error = nullptr;
    const auto plan = analyze(positions, slots, requests, 10, {}, spans, error);
    require(plan == strata::core::ExpertGroupPlan::refused, "mixed identities must fail closed");
    require(spans.empty(), "a refused window leaves no spans");
    require(error != nullptr, "the refusal names the reason");
}

void one_request_two_slots_refused() {
    // The verifier binds one request to one slot; rows of the same request on two slots could deliver a
    // result to a request at a position its span never held.
    const std::vector<int64_t> positions = {3, 4};
    const std::vector<int> slots = {1, 2};
    const std::vector<uint64_t> requests = {88, 88};
    std::vector<strata::core::ExpertGroupSpan> spans;
    const char* error = nullptr;
    const auto plan = analyze(positions, slots, requests, 10, {}, spans, error);
    require(plan == strata::core::ExpertGroupPlan::refused, "a request owning two slots is refused");
}

void noncontiguous_span_refused() {
    const std::vector<int64_t> positions = {3, 9};   // same request, same slot, positions not consecutive
    const std::vector<int> slots = {1, 1};
    const std::vector<uint64_t> requests = {88, 88};
    std::vector<strata::core::ExpertGroupSpan> spans;
    const char* error = nullptr;
    const auto plan = analyze(positions, slots, requests, 10, {}, spans, error);
    require(plan == strata::core::ExpertGroupPlan::refused, "a broken span is refused, never split silently");
}

void negative_position_refused() {
    std::vector<strata::core::ExpertGroupSpan> spans;
    const char* error = nullptr;
    const auto plan = analyze({-1}, {0}, {88}, 10, {}, spans, error);
    require(plan == strata::core::ExpertGroupPlan::refused, "negative token positions cannot enter expert groups");
}

void oversized_window_falls_back_per_span() {
    // Merged bounds exceeded with two servable spans: independent per-request jobs, same ownership.
    const std::vector<int64_t> positions = {0, 1};
    const std::vector<int> slots = {0, 1};
    const std::vector<uint64_t> requests = {1, 2};
    const strata::core::ExpertGroupBounds bounds{1, 4, 2, 8};   // the window holds one row, entries cap 4
    std::vector<strata::core::ExpertGroupSpan> spans;
    const char* error = nullptr;
    auto plan = analyze(positions, slots, requests, 3, bounds, spans, error);
    require(plan == strata::core::ExpertGroupPlan::per_span, "an oversized multi-request window falls back");
    require(spans.size() == 2, "the fallback still knows both requests' spans");
    // One oversized span alone: nothing to fall back to.
    const std::vector<int64_t> solo_positions = {0, 1};
    const std::vector<int> solo_slots = {-1, -1};
    const std::vector<uint64_t> solo_requests = {5, 5};
    plan = analyze(solo_positions, solo_slots, solo_requests, 3, bounds, spans, error);
    require(plan == strata::core::ExpertGroupPlan::refused, "a single oversized span cannot be split");
}

void span_claims_complete_exactly_once() {
    // Two spans' rows in one ledger operation: each row claims once, each owner completes once, the window
    // is publishable only when every claimed row is done, and a second claim can never sneak in.
    strata::core::ExpertCompletion ledger(8);
    const strata::core::ExpertOperation op{77, 6, 30, 3, 1};
    require(ledger.begin(op, 3), "the window's ledger begins");
    require(ledger.claim(0, 1, 1) && ledger.claim(1, 1, 1), "the CPU claims span A's and B's first rows");
    require(ledger.claim(2, 4, 1), "the helper claims span B's second row");
    require(!ledger.claim(0, 4, 1), "a claimed row cannot be claimed again");
    require(!ledger.publishable(), "unfinished rows are not publishable");
    require(ledger.complete(op, 1, 1), "the CPU's completion lands once");
    require(!ledger.complete(op, 1, 1), "a second completion for the same owner is refused");
    require(!ledger.publishable(), "the helper's row is still outstanding");
    require(ledger.complete(op, 4, 1), "the helper's completion lands");
    require(ledger.publishable(), "every claimed row done: the window is publishable");
    require(ledger.release(op), "the window's ownership releases after the final consumer");
    require(!ledger.active(), "the ledger retires the operation");
}

void cancelled_span_never_publishes() {
    // One member cancelled while its rows are in flight: nothing of it publishes, the rest of the window
    // is poisoned with it, and ownership releases only after the staged work has drained.
    strata::core::ExpertCompletion ledger(8);
    const strata::core::ExpertOperation op{77, 6, 30, 3, 1};
    require(ledger.begin(op, 3), "the window's ledger begins");
    require(ledger.claim(0, 1, 1) && ledger.claim(2, 4, 1), "the CPU and helper claim their rows");
    require(ledger.matches(op, 3), "the window matches before cancellation");
    ledger.cancel();
    require(!ledger.matches(op, 3), "a cancelled window never matches for publication");
    require(!ledger.publishable(), "a cancelled window is never publishable");
    require(ledger.drained(op, 4), "the helper's staged rows drain");
    require(ledger.drained(op, 1), "the CPU's rows drain");
    require(ledger.release(op), "ownership releases after the drains");
    require(!ledger.active(), "the ledger retires the cancelled operation");
}

void helper_grouping_preserves_span_ownership() {
    // A helper holding one expert claims exactly its rows - from either span - and the scheduler writes the
    // assignment back before finishing; the CPU's rows stay distinguishable for exactly-once delivery.
    class PickExpert final : public strata::core::ExpertWorker {
    public:
        PickExpert(int32_t expert, float value) : expert_(expert), value_(value) {}
        uint64_t residency_version() const override { return 1; }
        bool begin(const strata::core::ExpertWork& work, std::string&) override {
            count_ = work.operation.token_count * work.routed_width;
            owned_.assign((size_t) count_, false);
            for (int64_t row = 0; row < count_; ++row)
                owned_[(size_t) row] = work.experts[row] == expert_ && work.assigned[row] < 0;
            return true;
        }
        bool owns(int64_t row) const override {
            return row >= 0 && row < count_ && owned_[(size_t) row];
        }
        bool finish(float* output, std::string&) override {
            for (int64_t row = 0; row < count_; ++row)
                if (owned_[(size_t) row]) output[row] = value_;
            return true;
        }
        bool cancel(std::string&) override { return true; }
    private:
        int32_t expert_;
        float value_;
        int64_t count_ = 0;
        std::vector<bool> owned_;
    };

    // Three routed entries across two spans: rows 0 and 2 route to expert 3 (span A row, span B row),
    // row 1 routes elsewhere.  The helper owns expert 3 wherever it appears.
    std::array<int32_t, 3> ids = {3, 9, 3};
    std::array<int32_t, 3> assigned = {-1, -1, -1};
    PickExpert helper(3, 5.0f);
    strata::core::ExpertCompletion ledger(8);
    strata::core::ExpertHelperScheduler scheduler({&helper});
    strata::core::ExpertWork work;
    work.operation = {77, 6, 30, 3, 1};
    work.weights = {1, 1, &helper, 4, 4, 0, 0};
    work.experts = ids.data();
    work.routed_width = 1;
    work.assigned = assigned.data();
    require(ledger.begin(work.operation, 3), "the window's ledger begins");
    work.assigned = assigned.data();
    require(scheduler.begin(work, assigned.data(), ledger), "the helper submits over both spans");
    require(assigned[0] == 2 && assigned[2] == 2, "the helper's rows are marked in routing order");
    require(assigned[1] == -1, "the other row stays unassigned");
    require(ledger.owner(0) == 4 && ledger.owner(2) == 4 && ledger.owner(1) == 0,
            "the ledger holds the helper's ownership per row");
    std::array<float, 3> output = {0.f, 0.f, 0.f};
    require(ledger.claim(1, 1, 1), "the CPU claims the row the helper does not own");
    require(scheduler.finish(output.data()), "the helper completes its rows");
    require(output[0] == 5.0f && output[2] == 5.0f && output[1] == 0.f,
            "each helper row lands at its own routing index exactly once");
    require(!ledger.complete(work.operation, 4, 1), "duplicate helper completion is rejected");
    require(ledger.complete(work.operation, 1, 1), "the CPU's completion lands");
    require(ledger.publishable(), "the grouped window is publishable");
    require(ledger.release(work.operation), "the window releases");
}
} // namespace

int main() {
    two_request_spans_group_into_one_window();
    solo_window_is_one_span();
    mixed_bound_and_unbound_identities_refused();
    one_request_two_slots_refused();
    noncontiguous_span_refused();
    negative_position_refused();
    oversized_window_falls_back_per_span();
    span_claims_complete_exactly_once();
    cancelled_span_never_publishes();
    helper_grouping_preserves_span_ownership();
    return 0;
}
