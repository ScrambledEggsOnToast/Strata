// CPU-only lifecycle preflight. Real graph/isolation checks run through the protected model vehicle.
#include "strata/program/resident_slots.hpp"
#include "strata/core/expert_source.hpp"
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
    {
        program::ResidentSlot completed;
        completed.active = true;
        completed.request = 17;
        completed.input.scope.value[0] = 1;
        completed.ids = {11, 22, 33};
        completed.sampling.seed = 99;
        completed.produced = 2;
        completed.p = 3;
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
    std::printf("resident lifecycle preflight: failures=%d\n", failures);
    return failures ? 1 : 0;
}
