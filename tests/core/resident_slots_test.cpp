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
    std::printf("resident lifecycle preflight: failures=%d\n", failures);
    return failures ? 1 : 0;
}
