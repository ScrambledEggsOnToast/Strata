// HET-035/HET-017 resident-slot payload contract. No CUDA dependencies.
#pragma once
#include "strata/plan/qsa_memory.hpp"
#include <algorithm>
#include <cstdint>

namespace strata::plan {
// Conservative ABI envelopes, enforced against the source types by core/resident_memory.hpp.
// These are payload bounds, not measured allocator/driver/graph overhead.
inline constexpr uint64_t kResidentOwnerBytes = 4096;
inline constexpr uint64_t kResidentQsaBytes = 4096;
inline constexpr uint64_t kResidentMaxSlots = 8;
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
// One retained checkpoint and three context-bounded int32 token histories per slot.
// Transfer bound covers a slot checkpoint plus the larger main/draft image in either direction.
inline bool resident_bytes(uint64_t slots, uint64_t cells, uint64_t nq, uint64_t tail_floats,
                           const ResidentPayload& working_main, const ResidentPayload& slot_main, uint64_t init_transient,
                           const ResidentPayload& working_draft, const ResidentPayload& slot_draft, bool mtp, ResidentBytes& out) {
    out = {};
    const auto bounded = [](const ResidentPayload& p) { return p.device <= (1ull << 50) && p.pinned <= (1ull << 50); };
    if (slots > kResidentMaxSlots || cells == 0 || cells > INT32_MAX || nq > 48 ||
        tail_floats > (1ull << 20) || !working_main.device || !slot_main.device ||
        !bounded(working_main) || !bounded(slot_main) || !bounded(working_draft) || !bounded(slot_draft) ||
        init_transient > (1ull << 50) || (mtp && (!working_draft.device || !slot_draft.device))) return false;
    out.working_main = working_main;
    if (mtp) out.working_draft = working_draft;
    if (!slots) return true;
    const auto align256 = [](uint64_t n) { return (n + 255) & ~255ull; };
    out.slot_main = {slots * slot_main.device, slots * slot_main.pinned};
    out.slot_main_owner = slots * (kResidentOwnerBytes + nq * kResidentQsaBytes);
    out.histories = slots * cells * 3 * sizeof(int32_t);
    out.checkpoints = slots * slot_main.device; // full slot arena bounds one state-only checkpoint
    const uint64_t commit = slots * (2 + kResidentMaxSlots) * sizeof(int32_t);
    out.verifier_device = align256(align256(commit) + slots * std::max<uint64_t>(nq, 1) * tail_floats * 4)
                          + kResidentMaxSlots * (4096 + 1) * sizeof(int32_t);
    out.verifier_pinned = commit + 16;
    out.verifier_history = kResidentMaxSlots * 4096 * sizeof(int32_t);
    if (mtp) out.slot_draft = {slots * slot_draft.device, slots * slot_draft.pinned};
    out.slot_draft_owner = mtp ? slots * (kResidentQsaBytes + 3 * sizeof(uint64_t)) : 0;
    out.transfer_transient = slot_main.device + std::max(working_main.device, slot_main.device) +
        cells * 3 * sizeof(int32_t) + (mtp ? std::max(working_draft.device, slot_draft.device) : 0);
    out.init_transient = init_transient;
    return true;
}
} // namespace strata::plan
