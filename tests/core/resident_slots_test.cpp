// CPU-only lifecycle preflight. Real graph/isolation checks run through the protected model vehicle.
#include "strata/program/resident_slots.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/graph.hpp"
#include "strata/core/mtp.hpp"
#include "strata/core/verify.hpp"
#include "strata/prefill/prefill.hpp"

#include <cstdio>
#include <type_traits>

int main() {
    using namespace strata;
    static_assert(!std::is_move_constructible_v<program::ResidentSlots>);
    core::ModelGeometry geometry;
    core::SessionOwner main, a, b;
    core::Verifier verifier;
    core::MtpDrafter draft;
    prefill::Prefill prompt;
    core::ExpertDispatch dispatch;
    dispatch.completion.emplace_back(core::ExpertCompletion::kCapacity);
    program::ResidentStage stage{&main.state(), &verifier, {&a.state(), &b.state()}, -1};
    program::ResidentSlots slots(geometry, {stage}, draft, prompt, dispatch, 64, true);
    std::string error;
    int failures = 0;
    auto check = [&](bool ok, const char* message) {
        if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); ++failures; }
    };
    const core::GraphCaptureProbe disabled_probe({"resident-default", "no-observer"});
    for (const auto phase : {core::GraphCapturePhase::BeforeCapture, core::GraphCapturePhase::AfterCapture,
                            core::GraphCapturePhase::AfterInstantiate, core::GraphCapturePhase::AfterUpload,
                            core::GraphCapturePhase::Retained})
        check(disabled_probe.check(phase, error),
              "an absent capture observer performs no CUDA query, including with CUDA_VISIBLE_DEVICES=-1");
    {
        program::ResidentSlot completed;
        completed.active = true;
        completed.request = 17;
        completed.input.scope.value[0] = 1;
        completed.ids = {11, 22, 33};
        completed.sampling.seed = 99;
        completed.produced = 2;
        completed.p = 3;
        completed.prefill_split_until = 2048;
        completed.complete(true); // same policy called after serial and pipelined row drains
        const core::ConversationScope other{{2}};
        check(completed.reusable(completed.input.scope, true),
              "completed pipelined turn retains a same-conversation prefix");
        check(!completed.reusable(other, true) && !completed.reusable({}, true) &&
              !completed.reusable(completed.input.scope, false),
              "cold conversation, absent scope and changed steering all miss");
        check(!completed.active && completed.request == 17 && completed.produced == 2 &&
              completed.p == 3 && completed.sampling.seed == 99 && completed.ids.size() == 3,
              "retention changes neither committed output, identity, RNG nor position");
        check(completed.prefill_split_until == 0, "completion clears any pending prefill partition endpoint");
        completed.input.checkpoint = false;
        completed.complete(true);
        check(!completed.cached, "one-shot completed turns cannot become reusable");
        completed.input.checkpoint = true;
        completed.img = true;
        completed.complete(true);
        check(!completed.cached, "image slots stay ineligible");
        completed.img = false;
        completed.failed = true;
        completed.complete(true);
        check(!completed.cached, "failed state is never reusable");
        completed.failed = false;
        completed.complete(false);
        check(!completed.cached, "prompt-cache zero still disables retention");
        completed.tainted = true;                   // #879: this run met non-finite logits
        completed.complete(true);
        check(!completed.cached && !completed.reusable(completed.input.scope, true),
              "a tainted completion is neither cached nor reusable");
    }
    {
        // `tainted` belongs to the identity that ran, never to the seat (#879): a fresh (non-continuing)
        // admission retires the old identity and its stale taint together - bind clears the flag where it
        // retires, and rebuilds the seat's cells - so the fresh identity completes reusable. A continuing
        // rebind is the same identity and run: its taint stays, and no completion of it becomes reusable.
        program::ResidentSlot seat;
        seat.request = 23;                          // the poisoned run's own identity
        seat.input.scope.value[0] = 1;
        seat.input.checkpoint = true;
        seat.ids = {11, 22};
        seat.tainted = true;
        seat.complete(true);
        check(!seat.cached && !seat.reusable(seat.input.scope, true),
              "a tainted completion is not cached, so the poisoned prefix is never handed on");
        seat.tainted = false;                       // retired with the old identity by the fresh admission's bind
        seat.request = 29;                          // the identity the fresh (non-continuing) bind publishes
        seat.complete(true);
        check(seat.cached && seat.reusable(seat.input.scope, true),
              "a fresh non-continuing admission rebuilds the seat and completes reusable");
        seat.tainted = true;                        // kept by a continuing rebind of the same identity (29)
        seat.complete(true);
        check(!seat.cached && !seat.reusable(seat.input.scope, true),
              "a continuing tainted rebind stays non-reusable");
    }
    {
        program::ResidentSlot parked;
        parked.request = 17;
        parked.partial = true;
        parked.prefill_split_until = 2048;
        program::ResidentWorking other;
        other.request = 18;
        other.resume_slot = 0;
        parked.restored(other, 0, true);
        check(other.prefill_split_until == 0 && parked.prefill_split_until == 2048 && parked.partial,
              "prefix reuse by a new identity cannot claim or erase the parked request's pending half");
        program::ResidentWorking continued;
        continued.request = 17;
        continued.resume_slot = 1;
        parked.restored(continued, 0, true);
        check(continued.prefill_split_until == 0 && parked.prefill_split_until == 2048 && parked.partial,
              "the same identity cannot take a different physical slot's pending half");
        continued.resume_slot = 0;
        parked.restored(continued, 0, true);
        check(continued.prefill_split_until == 2048 && parked.prefill_split_until == 0 && !parked.partial,
              "full exact continuation transfers the original endpoint once");
        parked.partial = true;
        parked.prefill_split_until = 2048;
        parked.restored(continued, 0, false);
        check(continued.prefill_split_until == 0 && parked.prefill_split_until == 0 && !parked.partial,
              "checkpoint rewind discards the old live-frontier interval");
    }
    program::ResidentRequest oversized;
    oversized.max_new = 55; // two prompt cells + 55 output + the unchanged eight-cell margin > 64
    check(!slots.begin(-1, {11, 22}, oversized, error) && error.find("exceeds the context") != std::string::npos,
          "full output budget refuses before binding working consumers");
    oversized.max_new = INT64_MAX;
    check(!slots.begin(-1, {11, 22}, oversized, error) && error.find("overflowing") != std::string::npos,
          "overflowing output budget refuses before state pricing or binding");
    check(slots.working().request == 0 && dispatch.request_generation == 0 &&
          slots.slots()[0].request == 0 && slots.slots()[1].request == 0,
          "budget refusal publishes no identity and changes no resident owner");
    const core::ExpertOperation operation{71, 0, 0, 1, 1};
    check(dispatch.completion[0].begin(operation, 0), "operation lease acquired");
    program::ResidentRequest config;
    config.max_new = 8;
    check(!slots.begin(-1, {11, 22}, config, error), "working request refuses retained consumers");
    check(slots.working().request == 0 && dispatch.request_generation == 0 &&
          dispatch.completion[0].active(), "refusal changes neither request nor operation ownership");
    check(dispatch.completion[0].release(operation), "operation lease drains independently");
    check(!slots.begin(-1, {11, 22}, config, error) && slots.working().request == 0,
          "uninitialized working verifier cannot publish an identity");
    const std::vector<int32_t> prefix{11, 22};
    program::ResidentAdmission request;
    request.next_token = 33;
    check(!slots.admit(0, prefix, request, error), "admission before resource initialization refuses");
    check(slots.slots()[0].request == 0 && !slots.slots()[0].active &&
          slots.slots()[0].ids.empty() && dispatch.slot_requests[0] == 0,
          "refused admission publishes no request identity or history");
    check(!slots.init_resources(error), "uninitialized verifier cannot create resident resources");
    check(!slots.admit(1, prefix, request, error) && slots.slots()[1].request == 0,
          "failed resource initialization cannot expose another ready slot");
    const bool failed_before_pricing = slots.slots()[0].failed;
    program::PrefixWork restore_work{123, 456};
    check(!slots.resume_work(0, nullptr, restore_work, error) && restore_work.bytes == 0 &&
          restore_work.route == 0, "read-only pricing refuses an unpublished resident owner without stale costs");
    check(!slots.resume_work(-1, nullptr, restore_work, error),
          "read-only pricing refuses a nonexistent seat before reading state");
    check(slots.working().request == 0 && slots.slots()[0].ids.empty() &&
          slots.slots()[0].failed == failed_before_pricing,
          "pricing refusal changes no request identity, history or failure state");
    error = "stale pricing error";
    check(!slots.completed_snapshot_source(-1, error) && error.empty(),
          "missing completed source is a clean cache miss before GPU access");
    check(!slots.completed_snapshot_source(0, error) && error.empty(),
          "an unpublished seat cannot supply a conversation snapshot");
    check(slots.working().request == 0 && slots.slots()[0].ids.empty() &&
          slots.slots()[0].failed == failed_before_pricing && dispatch.request_generation == 0,
          "snapshot refusal changes neither working identity nor resident ownership");
    std::printf("resident lifecycle preflight: failures=%d\n", failures);
    return failures ? 1 : 0;
}
