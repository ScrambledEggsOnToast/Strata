// HET-035/HET-017 resident-slot payload contract. No CUDA dependencies.
#pragma once
#include "strata/plan/qsa_memory.hpp"
#include "strata/kernels/ngram.hpp"
#include <algorithm>
#include <cstdint>

namespace strata::plan {
// Validated [lo, hi) session ranges own their recurrent rows and attention
// pools. Even an attention-free range retains a primary QSA when the model
// has attention; shared scratch, PLE and the primary RoPE are never prorated.
struct SessionLayerCounts {
    int64_t gdn = 0, qsa = 0;
};
inline SessionLayerCounts session_layer_counts(int64_t n_layers, int64_t interval,
                                                int64_t lo, int64_t hi) {
    const int64_t qsa = hi / interval - lo / interval;
    return {hi - lo - qsa, std::max<int64_t>(qsa, n_layers / interval ? 1 : 0)};
}
inline uint64_t session_arena_bytes(const SessionLayerCounts& layers, uint64_t shared,
                                     uint64_t recurrent, uint64_t qsa_first, uint64_t qsa_rest) {
    uint64_t bytes = shared + (uint64_t) layers.gdn * recurrent;
    if (layers.qsa) bytes += qsa_first + (uint64_t) (layers.qsa - 1) * qsa_rest;
    return (bytes + 255) / 256 * 256 + 4096;
}

// Conservative ABI envelopes, enforced against the source types by core/resident_memory.hpp.
// These are payload bounds, not measured allocator/driver/graph overhead.
inline constexpr uint64_t kResidentOwnerBytes = 4096;
inline constexpr uint64_t kResidentQsaBytes = 4096;
inline constexpr uint64_t kResidentMaxSlots = 8;
// Fresh checkpoint/vector copies allocate their logical extents. Add 12.5% of
// the whole retained seat payload for capacity/alignment slack, plus 4 KiB per
// stage for checkpoint objects/directories (ABI-checked by the core adapter).
// Serial transfer images receive the same percentage and one 4-KiB envelope.
// Allocator/driver/graph overhead still needs the separate enforced allowance.
inline constexpr uint64_t kResidentPayloadMarginDivisor = 8;
inline constexpr uint64_t kResidentCheckpointMetadataBytes = 4096;
struct CheckpointStateBytes {
    uint64_t gdn = 0, ple = 0, tail = 0, dead = 0, block_pos = 0;
};
// The definitions consumed by conversation_checkpoint_save/restore. GDN is
// stage-local; PLE is duplicated in every stage; indexer terms are per layer.
inline bool checkpoint_state_bytes(uint64_t gdn_layers, uint64_t state_size, uint64_t v_heads,
                                    uint64_t conv_channels, uint64_t d_conv,
                                    const strata::kernels::QsaShapes& s, CheckpointStateBytes& out) {
    out = {};
    if (!state_size || !v_heads || !conv_channels || !d_conv || s.idx_block <= 0 || s.idx_dim <= 0) return false;
    LogicalBytes count;
    CheckpointStateBytes b;
    const uint64_t recurrence = count.product({state_size, v_heads, state_size});
    const uint64_t convolution = count.product({conv_channels, d_conv - 1});
    b.gdn = count.product({gdn_layers, count.sum({recurrence, convolution}), sizeof(float)});
    b.ple = count.product({strata::kernels::NG_HIST, strata::kernels::NG_HC_DIM, sizeof(float)});
    b.tail = count.product({(uint64_t) (s.idx_block - 1), (uint64_t) s.idx_dim, sizeof(float)});
    b.dead = count.product({(uint64_t) s.idx_dim, sizeof(float)});
    b.block_pos = sizeof(int32_t);
    if (!count.valid) return false;
    out = b;
    return true;
}
struct ResidentCheckpointPayload {
    uint64_t bytes = 0, stage_peak = 0, stages = 0;
};
inline bool resident_checkpoint_stage(uint64_t cells, uint64_t qsa_layers,
                                       const CheckpointStateBytes& state, ResidentCheckpointPayload& out) {
    LogicalBytes count;
    const uint64_t layer = count.sum({state.tail, state.dead, state.block_pos});
    const uint64_t payload = count.sum({state.gdn, state.ple, count.product({qsa_layers, layer}),
                                        count.product({cells, sizeof(int32_t)})});
    const uint64_t total = count.sum({out.bytes, payload});
    if (!count.valid || out.stages == UINT64_MAX) return false;
    out.bytes = total;
    out.stage_peak = std::max(out.stage_peak, payload);
    ++out.stages;
    return true;
}
struct ResidentHostBytes {
    // Histories/checkpoints are per seat; transfer is one serial operation peak.
    uint64_t histories = 0, checkpoints = 0, transfer_transient = 0;
};
inline bool resident_host_bytes(uint64_t cells, const ResidentCheckpointPayload& checkpoint,
                                 uint64_t main_image, uint64_t draft_image, ResidentHostBytes& out) {
    out = {};
    if (!cells || cells > INT32_MAX || !checkpoint.stages || !checkpoint.bytes ||
        !checkpoint.stage_peak || checkpoint.stage_peak > checkpoint.bytes) return false;
    LogicalBytes count;
    const auto margin = [](uint64_t n) {
        return n / kResidentPayloadMarginDivisor + (n % kResidentPayloadMarginDivisor != 0);
    };
    ResidentHostBytes b;
    // The third old "history" was the checkpoint ID vector: charge it in each
    // stage's checkpoint instead. ResidentSlot itself retains only ids/prompt.
    b.histories = count.product({cells, 2, sizeof(int32_t)});
    b.checkpoints = count.sum({checkpoint.bytes, margin(count.sum({checkpoint.bytes, b.histories})),
                               count.product({checkpoint.stages, kResidentCheckpointMetadataBytes})});
    // copy_to: one stage checkpoint + one main layer image; draft.save_slot
    // starts only after copy_to returns. Neither seats nor stages multiply it.
    const uint64_t transfer = std::max(count.sum({checkpoint.stage_peak, main_image}), draft_image);
    b.transfer_transient = count.sum({transfer, margin(transfer), kResidentCheckpointMetadataBytes});
    if (!count.valid) return false;
    out = b;
    return true;
}
struct ResidentPayload {
    uint64_t device = 0, pinned = 0;
};
struct ResidentBytes {
    // Working payloads are single owners; slot payloads are requested-count totals.
    ResidentPayload working_main, slot_main, working_draft, slot_draft;
    uint64_t slot_main_owner = 0, slot_draft_owner = 0;
    uint64_t histories = 0, checkpoints = 0;
    uint64_t verifier_device = 0, verifier_pinned = 0, verifier_history = 0;
    uint64_t transfer_transient = 0, init_transient = 0;
};
// Format is shared by main roles; only residency may differ. Unknown layouts or
// elastic growth must not become a zero payload in the admission adapter.
inline const char* resident_role_unpriced(const strata::kernels::QsaShapes& s, int64_t cells,
                                          const QsaRolePlacements& roles, bool mtp) {
    const auto main_valid = [&](const QsaPlacement& p) {
        return qsa_placement_matches(s, cells, p) && p.kv.mode != 2;
    };
    if (roles.working_main.elastic || roles.slot_main.elastic ||
        (mtp && (roles.working_draft.elastic || roles.slot_draft.elastic)))
        return "elastic_kv_growth";
    if (!main_valid(roles.working_main)) return "session_state";
    if (!main_valid(roles.slot_main) || roles.slot_main.format != roles.working_main.format ||
        roles.slot_main.rotate != roles.working_main.rotate) return "resident_slot_main_device";
    if (mtp) {
        const int fmt = qsa_draft_format(roles.working_main.format);
        if (!qsa_placement_matches(s, cells, roles.working_draft) || roles.working_draft.format != fmt)
            return "resident_working_draft_device";
        if (!qsa_placement_matches(s, cells, roles.slot_draft) || roles.slot_draft.format != fmt ||
            roles.slot_draft.rotate != roles.working_draft.rotate) return "resident_slot_draft_device";
    }
    return nullptr;
}
// slots is the REQUESTED --batch count, additional to the main working session.
// Never replace it with a count trimmed by a later runtime fallback. Calls are serial;
// weights, verifier/MTP scratch and opaque graph allowance are shared, not multiplied.
// One state-only checkpoint (with stage-local IDs/PLE) and two seat histories.
// The supplied host bound is derived from the actual checkpoint/image definitions.
inline bool resident_bytes(uint64_t slots, uint64_t cells, uint64_t nq, uint64_t tail_floats,
                           const ResidentPayload& working_main, const ResidentPayload& slot_main, uint64_t init_transient,
                           const ResidentPayload& working_draft, const ResidentPayload& slot_draft, bool mtp,
                           const ResidentHostBytes& host, ResidentBytes& out) {
    out = {};
    const auto bounded = [](const ResidentPayload& p) { return p.device <= (1ull << 50) && p.pinned <= (1ull << 50); };
    if (slots > kResidentMaxSlots || cells == 0 || cells > INT32_MAX || nq > 48 ||
        tail_floats > (1ull << 20) || !working_main.device || !slot_main.device ||
        !bounded(working_main) || !bounded(slot_main) || !bounded(working_draft) || !bounded(slot_draft) ||
        init_transient > (1ull << 50) || (mtp && (!working_draft.device || !slot_draft.device))) return false;
    out.working_main = working_main;
    if (mtp) out.working_draft = working_draft;
    if (!slots) return true;
    if (!host.checkpoints || !host.transfer_transient || host.histories > (1ull << 50) ||
        host.checkpoints > (1ull << 50) || host.transfer_transient > (1ull << 50)) return false;
    const auto align256 = [](uint64_t n) { return (n + 255) & ~255ull; };
    out.slot_main = {slots * slot_main.device, slots * slot_main.pinned};
    out.slot_main_owner = slots * (kResidentOwnerBytes + nq * kResidentQsaBytes);
    out.histories = slots * host.histories;
    out.checkpoints = slots * host.checkpoints;
    const uint64_t commit = slots * (2 + kResidentMaxSlots) * sizeof(int32_t);
    out.verifier_device = align256(align256(commit) + slots * std::max<uint64_t>(nq, 1) * tail_floats * 4)
                          + kResidentMaxSlots * (4096 + 1) * sizeof(int32_t);
    out.verifier_pinned = commit + 16;
    out.verifier_history = kResidentMaxSlots * 4096 * sizeof(int32_t);
    if (mtp) out.slot_draft = {slots * slot_draft.device, slots * slot_draft.pinned};
    out.slot_draft_owner = mtp ? slots * (kResidentQsaBytes + 3 * sizeof(uint64_t)) : 0;
    out.transfer_transient = host.transfer_transient;
    out.init_transient = init_transient;
    return true;
}
} // namespace strata::plan
