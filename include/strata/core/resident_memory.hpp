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
// The drafter's load/init_slots select a whole format, never hybrid K8V4.
// Preserve the process settings while invoking the same state sizing path.
inline strata::plan::ResidentPayload resident_draft_payload(const ModelGeometry& g, int64_t cells,
                                                           int64_t ring_cells = -1) {
    struct RestoreFormat {
        bool hybrid = qsa_kv_hybrid(), int8 = qsa_kv_int8(), q4 = qsa_kv_q4();
        ~RestoreFormat() { qsa_set_kv_q4(q4); qsa_set_kv_int8(int8); qsa_set_kv_hybrid(hybrid); }
    } restore;
    const int fmt = strata::plan::qsa_draft_format(
        strata::plan::qsa_state_format(restore.hybrid, restore.q4, restore.int8, -1));
    qsa_set_kv_hybrid(fmt == strata::kernels::kKvHybrid);
    qsa_set_kv_int8(fmt == strata::kernels::kKvInt8);
    qsa_set_kv_q4(fmt == strata::kernels::kKvQ4);
    return {qsa_state_bytes(g, cells, false, ring_cells), qsa_state_host_bytes(g, cells, ring_cells)};
}

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
    const strata::plan::ResidentPayload main{base.device, base.host_pinned};
    const auto draft = mtp ? resident_draft_payload(g, cells) : strata::plan::ResidentPayload{};
    return strata::plan::resident_bytes((uint64_t) requested_slots, (uint64_t) cells,
        (uint64_t) g.n_qsa_layers(), (uint64_t) (shape.idx_block - 1) * g.idx_key_dim,
        main, main, base.host_transient, draft, draft, mtp, out);
}
} // namespace strata::core
