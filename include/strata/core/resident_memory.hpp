// Pure preallocation adapter: invoke BEFORE the expert auto-cache grant.
#pragma once
#include "strata/core/session.hpp"
#include "strata/core/conversation_cache.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/plan/resident.hpp"
#include "strata/plan/qsa_memory.hpp"

namespace strata::core {
static_assert(sizeof(SessionOwner) <= strata::plan::kResidentOwnerBytes,
              "increase resident owner ABI envelope in native and Python planner");
static_assert(sizeof(QsaState) <= strata::plan::kResidentQsaBytes,
              "increase resident QSA ABI envelope in native and Python planner");
static_assert(sizeof(ConversationCheckpoint) <= strata::plan::kResidentCheckpointMetadataBytes &&
              sizeof(ConversationKv) <= strata::plan::kResidentCheckpointMetadataBytes,
              "increase checkpoint/image metadata envelope in native and Python planner");

// Explicit splits use the same layer counts as SessionOwner. For an unresolved
// auto split, the full model bounds the largest stage, with one extra primary
// QSA per later stage (a stage with no attention still allocates that primary).
inline bool resident_host_payload(const ModelGeometry& g, int64_t cells, bool mtp,
                                   const strata::plan::QsaRolePlacements& roles,
                                   uint64_t stages, const int64_t* split_at,
                                   strata::plan::ResidentHostBytes& out) {
    out = {};
    if (!stages || stages > (uint64_t) g.n_layers || cells <= 0 || cells > INT32_MAX ||
        g.qsa_interval <= 0) return false;
    auto shape = strata::kernels::qsa_real_shapes();
    shape.n_head_kv = g.n_head_kv;
    shape.head_dim = g.head_dim;
    shape.idx_dim = g.idx_key_dim;
    strata::plan::ResidentCheckpointPayload checkpoint;
    int64_t lo = 0;
    for (uint64_t stage = 0; stage < stages; ++stage) {
        const int64_t hi = split_at && stage + 1 < stages ? split_at[stage] : g.n_layers;
        if (split_at && (hi <= lo || hi > g.n_layers)) return false;
        const int64_t q_range = split_at ? hi / g.qsa_interval - lo / g.qsa_interval
                                         : stage == 0 ? g.n_qsa_layers() : 0;
        const uint64_t nq = std::max<int64_t>(q_range, g.n_qsa_layers() ? 1 : 0);
        const uint64_t ng = split_at ? (hi - lo) - q_range : stage == 0 ? g.n_gdn_layers() : 0;
        strata::plan::CheckpointStateBytes state;
        if (!strata::plan::checkpoint_state_bytes(ng, g.ssm_state_size, g.ssm_v_heads,
            g.ssm_conv_channels, g.ssm_d_conv, shape, state) ||
            !strata::plan::resident_checkpoint_stage(cells, nq, state, checkpoint)) return false;
        lo = hi;
    }
    strata::plan::QsaImageBytes main, draft;
    if (g.n_qsa_layers() && !strata::plan::qsa_image_bytes(shape, cells, roles.working_main.format, true, main)) return false;
    if (mtp && !strata::plan::qsa_image_bytes(shape, cells, roles.working_draft.format, false, draft)) return false;
    return strata::plan::resident_host_bytes(cells, checkpoint, main.bytes, draft.bytes, out);
}
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
                                      strata::plan::ResidentBytes& out, const char*& unpriced_class,
                                      uint64_t stage_count = 1, const int64_t* split_at = nullptr) {
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
    strata::plan::ResidentHostBytes host;
    unpriced_class = "resident_checkpoints";
    if (requested_slots && !resident_host_payload(g, cells, mtp, placements, stage_count, split_at, host)) return false;
    unpriced_class = "resident_ownership_payload";
    if (!strata::plan::resident_bytes((uint64_t) requested_slots, (uint64_t) cells,
        (uint64_t) g.n_qsa_layers(), (uint64_t) (shape.idx_block - 1) * g.idx_key_dim,
        main, slot_main, std::max(working.host_transient, slot.host_transient), draft, slot_draft, mtp, host, out))
        return false;
    unpriced_class = nullptr;
    return true;
}
} // namespace strata::core
