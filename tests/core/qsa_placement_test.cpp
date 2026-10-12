// Header-only placement/pricing/carving seam; no CUDA calls or model fixtures.
#include "strata/plan/qsa_memory.hpp"
#include <cstdio>
#include <stdexcept>

namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
void test_resolved_placement_prices_the_carved_extent() {
    using namespace strata::plan;
    using namespace strata::kernels;
    const auto s = qsa_real_shapes();
    for (int fmt : {kKvF16, kKvInt8, kKvQ4, kKvHybrid}) {
        for (int64_t window : {int64_t{0}, int64_t{20480}, int64_t{32768}}) {
            const auto placement = qsa_resolve_placement(s, 262144, window, 0, fmt);
            for (bool rope : {false, true}) {
                const uint64_t price = qsa_device_bytes(s, placement.cells, placement.kv,
                                                       placement.format, rope, placement.elastic);
                const auto arena = qsa_arena_layout(s, placement, rope, price);
                require(arena.valid && arena.extent == price, "resolved price differs from carved extent");
                require(arena.payload_extent <= price, "QSA arrays overlap the following allocation");
                require(!qsa_arena_layout(s, placement, rope, price - 1).valid,
                        "undersized QSA arena accepted");
            }
        }
    }
}
void test_slot_policy_and_startup_floor() {
    using namespace strata::plan;
    using namespace strata::kernels;
    const auto s = qsa_real_shapes();
    const auto inherited = qsa_resolve_roles(s, 262144, kKvF16, 32768, -1, 32768, 8);
    require(inherited.working_main.kv.slots == 8192 && inherited.slot_main.kv.slots == 8192,
            "absent slot override changed the global placement");
    require(inherited.working_draft.kv.slots == 8216 && inherited.slot_draft.kv.slots == 8216,
            "absent slot override changed draft ring inheritance");
    const auto mixed = qsa_resolve_roles(s, 262144, kKvHybrid, 0, 65536, 32768, 8);
    require(mixed.working_main.kv.mode == 0 && mixed.slot_main.kv.mode == 1 &&
            mixed.slot_main.kv.slots == 16384, "slot override retargeted the working session");
    require(mixed.working_main.format == mixed.slot_main.format &&
            mixed.working_draft.format == kKvInt8 && mixed.slot_draft.format == kKvInt8,
            "residency override changed a role's format");
    require(mixed.working_draft.kv.mode == 0 && mixed.slot_draft.kv.mode == 2 &&
            mixed.slot_draft.kv.slots == 16384, "slot draft does not follow its requested window");
    const auto full_slots = qsa_resolve_roles(s, 262144, kKvF16, 32768, 0, 32768, 8);
    require(full_slots.working_main.kv.mode == 1 && full_slots.slot_main.kv.mode == 0 &&
            full_slots.slot_draft.kv.mode == 0, "explicit zero did not make slots resident");
    require(qsa_slot_window_error(s, -1, 8, 2051) != nullptr, "negative window accepted");
    require(qsa_slot_window_error(s, 20479, 8, 2051) != nullptr, "engine floor silently rounded up");
    require(qsa_slot_window_error(s, 20480, 8, 2051) == nullptr, "minimum legal main window refused");
    require(qsa_stream_call_min_cells(s, 8, 2051) == 16448, "call-capacity floor conflated with engine floor");
    require(qsa_slot_window_error(s, 20480, 8, 4096) != nullptr, "larger call-capacity floor ignored");
    require(qsa_slot_window_error(s, 20480, 8, 2051, 32864) != nullptr, "undersized dense draft ring accepted");
    require(qsa_slot_window_error(s, 0, 8, 2051, 32864) == nullptr, "resident slot needs a streamed floor");
}
void test_absence_is_byte_identical_to_legacy_prices() {
    using namespace strata::plan;
    using namespace strata::kernels;
    const auto s = qsa_real_shapes();
    const auto daily = qsa_resolve_roles(s, 131072, kKvF16, 0, -1, 32768, 8);
    require(qsa_device_bytes(s, 131072, daily.working_main.kv, kKvF16, true) == 318901712 &&
            qsa_device_bytes(s, 131072, daily.slot_main.kv, kKvF16, false) == 285347280,
            "absent option changed the legacy 128K FP16 QSA prices");
    require(qsa_pinned_bytes(s, daily.working_main.kv, kKvF16) == 116 &&
            qsa_pinned_bytes(s, daily.slot_main.kv, kKvF16) == 116,
            "absent option added host backing to resident roles");
    const auto arena = qsa_arena_layout(s, daily.working_main, true);
    require(arena.valid && arena.extent == qsa_device_bytes(
                s, daily.working_main.cells, daily.working_main.kv, daily.working_main.format, true),
            "legacy FP16 carved extent differs from its price");
    const auto streamed = qsa_resolve_roles(s, 262144, kKvF16, 20480, -1, 32768, 8);
    require(qsa_device_bytes(s, 262144, streamed.working_main.kv, kKvF16, false) == 75865712 &&
            qsa_device_bytes(s, 262144, streamed.slot_main.kv, kKvF16, false) == 75865712,
            "absent option changed legacy streamed-main/slot prices");
    const auto hybrid = qsa_resolve_roles(s, 262144, kKvHybrid, 0, -1, 32768, 8);
    require(qsa_device_bytes(s, 262144, hybrid.working_main.kv, kKvHybrid, true) == 314838544 &&
            qsa_device_bytes(s, 262144, hybrid.working_draft.kv, kKvInt8, false) == 310644240,
            "absent option changed hybrid main or whole-INT8 draft prices");
}
void test_draft_and_elastic_carves_are_bounded() {
    using namespace strata::plan;
    using namespace strata::kernels;
    const auto s = qsa_real_shapes();
    for (int fmt : {kKvF16, kKvInt8, kKvQ4}) {
        const auto ring = qsa_resolve_placement(s, 262144, 65536, 65536, fmt);
        const uint64_t price = qsa_device_bytes(s, ring.cells, ring.kv, fmt, false);
        const auto arena = qsa_arena_layout(s, ring, false, price);
        require(arena.valid && arena.extent == price && ring.kv.pooled_rows == 2,
                "draft ring price differs from its bounded carve");
        QsaArenaCursor pinned{qsa_pinned_bytes(s, ring.kv, fmt) - 116};
        (void) qsa_pool_layout(s, ring.kv.pages, fmt, pinned);
        require(pinned.valid, "pinned draft pools exceed the role's host price");
    }
    auto elastic = qsa_resolve_placement(s, 20481, 0, 0, kKvF16, false, true);
    require(qsa_arena_layout(s, elastic, true).valid, "legacy elastic carve is not priced");
    elastic.kv.pages += 1;
    require(!qsa_arena_layout(s, elastic, true).valid, "mismatched frozen placement accepted");
}
}
int main() {
    try {
        test_resolved_placement_prices_the_carved_extent();
        test_slot_policy_and_startup_floor();
        test_absence_is_byte_identical_to_legacy_prices();
        test_draft_and_elastic_carves_are_bounded();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "qsa_placement_test: %s\n", e.what());
        return 1;
    }
    return 0;
}
