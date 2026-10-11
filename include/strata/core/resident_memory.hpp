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

// FP16 fully resident scope matches the dry-run planner. Do not silently price a
// streamed/quantized configuration as zero; caller must emit an unknown class or refuse.
inline bool resident_allocation_bytes(const ModelGeometry& g, int64_t cells, int64_t k,
                                      int requested_slots, bool mtp,
                                      strata::plan::ResidentBytes& out,
                                      const strata::plan::QsaRolePlacements* placements = nullptr) {
    out = {};
    if (requested_slots < 0 || qsa_kv_resident() != 0 || qsa_kv_int8() || qsa_kv_q4() || qsa_kv_hybrid())
        return false;
    SessionAllocationBytes working, slot;
    if (!session_allocation_bytes(g, cells, k, working, 0, -1,
                                  placements ? &placements->working_main : nullptr) ||
        !session_allocation_bytes(g, cells, k, slot, 0, -1,
                                  placements ? &placements->slot_main : nullptr)) return false;
    const auto shape = strata::kernels::qsa_real_shapes();
    const strata::plan::ResidentPayload main{working.device, working.host_pinned};
    const strata::plan::ResidentPayload slot_main{slot.device, slot.host_pinned};
    const auto draft = mtp ? resident_draft_payload(g, cells, -1,
        placements ? &placements->working_draft : nullptr) : strata::plan::ResidentPayload{};
    const auto slot_draft = mtp ? resident_draft_payload(g, cells, -1,
        placements ? &placements->slot_draft : nullptr) : strata::plan::ResidentPayload{};
    return strata::plan::resident_bytes((uint64_t) requested_slots, (uint64_t) cells,
        (uint64_t) g.n_qsa_layers(), (uint64_t) (shape.idx_block - 1) * g.idx_key_dim,
        main, slot_main, std::max(working.host_transient, slot.host_transient), draft, slot_draft, mtp, out);
}
} // namespace strata::core
