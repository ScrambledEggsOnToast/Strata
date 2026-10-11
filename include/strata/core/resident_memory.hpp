// Pure preallocation adapter: invoke BEFORE the expert auto-cache grant.
#pragma once
#include "strata/core/session.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/plan/resident.hpp"
#include "strata/plan/qsa_memory.hpp"

namespace strata::core {
static_assert(sizeof(SessionOwner) <= strata::plan::kResidentOwnerBytes,
              "increase resident owner ABI envelope in native and Python planner");
static_assert(sizeof(QsaState) <= strata::plan::kResidentQsaBytes,
              "increase resident QSA ABI envelope in native and Python planner");
// Use the drafter's whole format without temporarily mutating process dispatch.
inline strata::plan::ResidentPayload resident_draft_payload(const ModelGeometry& g, int64_t cells,
                                                           int64_t ring_cells = -1,
                                                           const strata::plan::QsaPlacement* placement = nullptr) {
    const auto resolved = placement ? *placement : qsa_state_placement(g, cells, ring_cells, -1, true);
    return {qsa_state_bytes(g, cells, false, ring_cells, &resolved),
            qsa_state_host_bytes(g, cells, ring_cells, &resolved)};
}

// Price the frozen role placements used by the allocator, including quantized
// pools and authoritative streamed backing. Unknown demand still fails closed.
inline bool resident_allocation_bytes(const ModelGeometry& g, int64_t cells, int64_t k,
                                      int requested_slots, bool mtp,
                                      const strata::plan::QsaRolePlacements& placements,
                                      strata::plan::ResidentBytes& out, const char*& unpriced_class) {
    out = {};
    const auto shape = strata::kernels::qsa_real_shapes();
    unpriced_class = "resident_slot_count";
    if (requested_slots < 0 || requested_slots > (int) strata::plan::kResidentMaxSlots) return false;
    unpriced_class = strata::plan::resident_role_unpriced(shape, cells, placements, mtp);
    if (unpriced_class) return false;
    SessionAllocationBytes working, slot;
    unpriced_class = "session_state";
    if (!session_allocation_bytes(g, cells, k, working, 0, -1, &placements.working_main)) return false;
    unpriced_class = "resident_slot_main_device";
    if (!session_allocation_bytes(g, cells, k, slot, 0, -1, &placements.slot_main)) return false;
    const strata::plan::ResidentPayload main{working.device, working.host_pinned};
    const strata::plan::ResidentPayload slot_main{slot.device, slot.host_pinned};
    const auto draft = mtp ? resident_draft_payload(g, cells, -1, &placements.working_draft)
                           : strata::plan::ResidentPayload{};
    const auto slot_draft = mtp ? resident_draft_payload(g, cells, -1, &placements.slot_draft)
                                : strata::plan::ResidentPayload{};
    unpriced_class = "resident_ownership_payload";
    if (!strata::plan::resident_bytes((uint64_t) requested_slots, (uint64_t) cells,
        (uint64_t) g.n_qsa_layers(), (uint64_t) (shape.idx_block - 1) * g.idx_key_dim,
        main, slot_main, std::max(working.host_transient, slot.host_transient), draft, slot_draft, mtp, out))
        return false;
    unpriced_class = nullptr;
    return true;
}
} // namespace strata::core
