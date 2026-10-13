#include "strata/program/serial_judge.hpp"
#include "strata/prefill/prefill.hpp"
#include "strata/core/native_head.hpp"
#include <cstdio>
#include <limits>
#include <type_traits>

namespace {
int failures = 0;
void check(bool value, const char* name) {
    if (!value) { std::fprintf(stderr, "FAIL %s\n", name); ++failures; }
}
}
int main() {
    using namespace strata;
    core::ModelGeometry geometry;
    core::qsa_set_kv_resident(0);
    core::qsa_set_kv_int8(false);
    core::qsa_set_kv_q4(false);
    core::qsa_set_kv_hybrid(true);
    const auto placement = core::qsa_state_placement(geometry, 4096, 0, 4096);
    const int64_t ranges[] = {0, 20, 39, 48};
    uint64_t checkpoint = 0, sessions = 0, readback = 0;
    for (int stage = 0; stage < 3; ++stage) {
        program::JudgeStageBytes bytes;
        check(program::judge_stage_bytes(geometry, 4096, 10, ranges[stage], ranges[stage + 1],
                                        248320, stage == 2, placement, bytes), "daily short-owner price available");
        core::SessionAllocationBytes session;
        check(core::session_allocation_bytes(geometry, 4096, 10, session, ranges[stage], ranges[stage + 1], &placement) &&
              bytes.session.device == session.device && bytes.session.host_pinned == session.host_pinned &&
              bytes.session.host_heap == session.host_heap && bytes.session.host_transient == session.host_transient,
              "the complete independent SessionOwner price is retained");
        check(bytes.device_canary == 280 && bytes.host_canary == 108, "operation canary priced per stage");
        check(bytes.device_head_output == (stage == 2 ? 248320ull * 4 : 0) &&
              bytes.host_readback == bytes.device_head_output, "head output and readback only on actual head stage");
        check(bytes.device_bytes() % (2ull << 20) == 0, "individual allocation granules are charged");
        check(bytes.session.host_transient == 1048576, "short-state initialization transient explicit");
        checkpoint += bytes.host_checkpoint; sessions += bytes.session.device; readback += bytes.host_readback;
        std::printf("JUDGE_PRICE stage=%d session_device=%llu total_device=%llu host_checkpoint=%llu total_host=%llu init_transient=%llu\n",
                    stage, (unsigned long long)bytes.session.device, (unsigned long long)bytes.device_bytes(),
                    (unsigned long long)bytes.host_checkpoint, (unsigned long long)bytes.host_bytes(),
                    (unsigned long long)bytes.session.host_transient);
    }
    check(checkpoint > 120ull << 20 && checkpoint < 140ull << 20, "all stage-local GDN/PLE/indexer branch state charged");
    check(sessions > 180ull << 20 && sessions < 200ull << 20, "short lane is not a full-context generation slot");
    check(readback == 248320ull * 4, "no hidden full-logit matrix");
    program::JudgeStageBytes invalid;
    for (int64_t cells : {int64_t(0), int64_t(4097), std::numeric_limits<int64_t>::max()})
        check(!program::judge_stage_bytes(geometry, cells, 10, 0, 20, 248320, false, placement, invalid),
              "unknown or oversized short-context demand refuses rather than shrinking");
    auto streamed = placement; streamed.kv.mode = 1;
    check(!program::judge_stage_bytes(geometry, 4096, 10, 0, 20, 248320, false, streamed, invalid),
          "unpriced streamed rebind rejected");
    check(!program::judge_stage_bytes(geometry, 4096, 10, 20, 20, 248320, false, placement, invalid), "empty stage rejected");
    check(!program::judge_stage_bytes(geometry, 4096, 10, 0, 20, 0, false, placement, invalid), "unknown vocabulary rejected");
    plan::DeviceCost device;
    device.name = "CUDA0";
    device.vram.add_unknown("judge_library_first_use_driver", "missing finite first-use bound");
    plan::DeviceTelemetry telemetry;
    telemetry.name = "CUDA0"; telemetry.measured = true;
    telemetry.total_bytes = telemetry.free_bytes = 31ull << 30;
    const auto admission = plan::admit_configuration({device}, {telemetry}, {}, {}, {});
    check(!admission.verified && !admission.admitted && !admission.unknown_classes.empty(), "unknown classes never admit");
    prefill::Prefill empty_prompt;
    core::SessionState unowned;
    std::string error;
    check(!empty_prompt.validate_rebind(unowned, nullptr, 1, error), "no uninitialized or manual execution owner accepted");
    program::SerialJudgeLane lane;
    core::NativeHead head;
    check(!lane.ready() && !lane.run(1, nullptr, 0, error), "unadmitted lane cannot execute");
    check(!lane.init(geometry, 4096, 10, 248320, placement, {}, head, nullptr, admission, error),
          "complete verified admission and actual loaded execution owners required");
    static_assert(!std::is_move_constructible_v<program::SerialJudgeLane>);
    return failures ? 1 : 0;
}
