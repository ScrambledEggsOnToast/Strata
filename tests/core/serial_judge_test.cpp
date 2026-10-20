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
    struct Ledger {
        uint64_t checkpoint = 0, sessions = 0, readback = 0, shared_tables = 0;
    };
    const auto price_mode = [&](const char* mode, const int64_t* ranges, int stages, int generation_slots) {
        Ledger ledger;
        int64_t resident_pairs = 0;
        for (int stage = 0; stage < stages; ++stage) {
            program::JudgeStageBytes bytes;
            check(program::judge_stage_bytes(geometry, 4096, 10, ranges[stage], ranges[stage + 1],
                                            248320, stage + 1 == stages, placement, bytes), "stage-local short-owner price available");
            core::SessionAllocationBytes session, generation;
            check(core::session_allocation_bytes(geometry, 4096, 10, session, ranges[stage], ranges[stage + 1], &placement) &&
                  bytes.session.device == session.device && bytes.session.host_pinned == session.host_pinned &&
                  bytes.session.host_heap == session.host_heap && bytes.session.host_transient == session.host_transient,
                  "the complete independent SessionOwner price is retained");
            const auto long_placement = core::qsa_state_placement(geometry, 262144, 0, 262144);
            check(core::session_allocation_bytes(geometry, 262144, 10, generation, ranges[stage], ranges[stage + 1], &long_placement) &&
                  generation.device > session.device, "judge price is short, never the full-context generation arena");
            check(bytes.device_canary == 280 && bytes.host_canary == 108, "operation canary priced per stage");
            check(bytes.device_head_output == (stage + 1 == stages ? 248320ull * 4 : 0) &&
                  bytes.host_readback == bytes.device_head_output, "head output and readback only on actual head stage");
            check(bytes.device_bytes() % (2ull << 20) == 0, "individual allocation granules are charged");
            check(bytes.session.host_transient == 1048576,
                  "conservative short-table/page-table bound retained even when borrow builds no table");
            const auto table = core::shared_rope_device_bytes(geometry, 262144);
            check(table == (core::shared_rope_enabled() ? 67108864ull : 0),
                  "immutable process table is admitted once per device, not per generation or judge owner");
            ledger.checkpoint += bytes.host_checkpoint;
            ledger.sessions += bytes.session.device;
            ledger.readback += bytes.host_readback;
            ledger.shared_tables += table;
            resident_pairs += (ranges[stage + 1] - ranges[stage]) * geometry.n_expert;
            std::printf("JUDGE_PRICE mode=%s generation_slots=%d stage=%d session_device=%llu total_device=%llu "
                        "host_pinned=%llu host_heap=%llu host_checkpoint=%llu total_host=%llu init_transient=%llu shared_table=%llu\n",
                        mode, generation_slots, stage, (unsigned long long)bytes.session.device,
                        (unsigned long long)bytes.device_bytes(), (unsigned long long)bytes.session.host_pinned,
                        (unsigned long long)bytes.session.host_heap, (unsigned long long)bytes.host_checkpoint,
                        (unsigned long long)bytes.host_bytes(), (unsigned long long)bytes.session.host_transient,
                        (unsigned long long)table);
        }
        check(ledger.checkpoint > 120ull << 20 && ledger.checkpoint < 140ull << 20,
              "all stage-local GDN/PLE/indexer branch state charged");
        check(ledger.sessions > 170ull << 20 && ledger.sessions < 200ull << 20,
              "short lane is not a full-context generation slot");
        check(ledger.readback == 248320ull * 4, "no hidden full-logit matrix");
        check(resident_pairs == 24576, "both priced splits cover every required expert pair without borrowing residency");
        return ledger;
    };
    const int64_t three_ranges[] = {0, 20, 39, 48}, pair_ranges[] = {0, 25, 48};
    const auto three = price_mode("three_gpu", three_ranges, 3, 4);
    const auto pair = price_mode("two_v100_serial", pair_ranges, 2, 1);
    check(pair.sessions < three.sessions && pair.checkpoint < three.checkpoint,
          "two-V100 one-generation-slot mode is priced independently, not scaled from the three-GPU total");
    check(three.shared_tables == (core::shared_rope_enabled() ? 3 * 67108864ull : 0) &&
          pair.shared_tables == (core::shared_rope_enabled() ? 2 * 67108864ull : 0),
          "short borrow adds zero tables in either hardware composition");
    check(core::qsa_state_bytes(geometry, 4096, true, 0, &placement) -
          core::qsa_state_bytes(geometry, 4096, false, 0, &placement) ==
          (core::shared_rope_enabled() ? 0 : 1048576ull),
          "shared session arenas contain no hidden dedicated judge RoPE allocation");
    program::JudgeStageBytes invalid;
    for (int64_t cells : {int64_t(0), int64_t(4097), int64_t(262144), std::numeric_limits<int64_t>::max()})
        check(!program::judge_stage_bytes(geometry, cells, 10, 0, 20, 248320, false, placement, invalid),
              "unknown or oversized short-context demand refuses rather than shrinking");
    auto streamed = placement; streamed.kv.mode = 1;
    check(!program::judge_stage_bytes(geometry, 4096, 10, 0, 20, 248320, false, streamed, invalid),
          "unpriced streamed rebind rejected");
    auto elastic = placement; elastic.elastic = true;
    check(!program::judge_stage_bytes(geometry, 4096, 10, 0, 20, 248320, false, elastic, invalid),
          "unpriced elastic judge placement rejected");
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
