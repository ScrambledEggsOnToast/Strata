// Pure preallocation adapter: invoke BEFORE the expert auto-cache grant.
#pragma once
#include "strata/core/session.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/plan/resident.hpp"

namespace strata::core {
static_assert(sizeof(SessionOwner) <= strata::plan::kResidentOwnerBytes,
              "increase resident owner ABI envelope in native and Python planner");
static_assert(sizeof(QsaState) <= strata::plan::kResidentQsaBytes,
              "increase resident QSA ABI envelope in native and Python planner");
// FP16 fully resident scope matches the dry-run planner. Do not silently price a
// streamed/quantized configuration as zero; caller must emit an unknown class or refuse.
inline bool resident_allocation_bytes(const ModelGeometry& g, int64_t cells, int64_t k,
                                      int requested_slots, bool mtp,
                                      strata::plan::ResidentBytes& out) {
    out = {};
    if (requested_slots < 0 || qsa_kv_resident() != 0 || qsa_kv_int8() || qsa_kv_q4() || qsa_kv_hybrid())
        return false;
    SessionAllocationBytes base;
    if (!session_allocation_bytes(g, cells, k, base)) return false;
    const auto shape = strata::kernels::qsa_real_shapes();
    const uint64_t draft = mtp ? qsa_state_bytes(g, cells, false, -1) : 0;
    const uint64_t pinned = mtp ? qsa_state_host_bytes(g, cells, -1) : 0;
    return strata::plan::resident_bytes((uint64_t) requested_slots, (uint64_t) cells,
        (uint64_t) g.n_qsa_layers(), (uint64_t) (shape.idx_block - 1) * g.idx_key_dim,
        base.device, base.host_pinned, base.host_transient, draft, pinned, mtp, out);
}
} // namespace strata::core
