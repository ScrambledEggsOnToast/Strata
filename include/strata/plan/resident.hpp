// HET-035/HET-017 resident-slot payload contract. No CUDA dependencies.
#pragma once
#include <algorithm>
#include <cstdint>

namespace strata::plan {
// Conservative ABI envelopes, enforced against the source types by core/resident_memory.hpp.
// These are payload bounds, not measured allocator/driver/graph overhead.
inline constexpr uint64_t kResidentOwnerBytes = 4096;
inline constexpr uint64_t kResidentQsaBytes = 4096;
inline constexpr uint64_t kResidentMaxSlots = 8;
struct ResidentBytes {
    uint64_t session_device = 0, session_pinned = 0, session_owner = 0;
    uint64_t histories = 0, checkpoints = 0;
    uint64_t verifier_device = 0, verifier_pinned = 0, verifier_history = 0;
    uint64_t mtp_device = 0, mtp_pinned = 0, mtp_owner = 0;
    uint64_t transfer_transient = 0, init_transient = 0;
};
// slots is the REQUESTED --batch count, additional to the main working session.
// Never replace it with a count trimmed by a later runtime fallback. Calls are serial;
// weights, verifier/MTP scratch and opaque graph allowance are shared, not multiplied.
// One retained checkpoint and three context-bounded int32 token histories per slot.
// Transfer bound covers one checkpoint plus one layer KV image (including segment payloads).
inline bool resident_bytes(uint64_t slots, uint64_t cells, uint64_t nq, uint64_t tail_floats,
                           uint64_t session_device, uint64_t session_pinned, uint64_t init_transient,
                           uint64_t draft_device, uint64_t draft_pinned, bool mtp, ResidentBytes& out) {
    out = {};
    if (slots > kResidentMaxSlots || cells == 0 || cells > INT32_MAX || nq > 48 ||
        tail_floats > (1ull << 20) || !session_device ||
        session_device > (1ull << 50) || session_pinned > (1ull << 50) ||
        init_transient > (1ull << 50) || draft_device > (1ull << 50) || draft_pinned > (1ull << 50) ||
        (mtp && !draft_device)) return false;
    if (!slots) return true;
    const auto align256 = [](uint64_t n) { return (n + 255) & ~255ull; };
    out.session_device = slots * session_device;
    out.session_pinned = slots * session_pinned;
    out.session_owner = slots * (kResidentOwnerBytes + nq * kResidentQsaBytes);
    out.histories = slots * cells * 3 * sizeof(int32_t);
    out.checkpoints = slots * session_device; // full arena bounds one state-only checkpoint
    const uint64_t commit = slots * (2 + kResidentMaxSlots) * sizeof(int32_t);
    out.verifier_device = align256(align256(commit) + slots * std::max<uint64_t>(nq, 1) * tail_floats * 4)
                          + kResidentMaxSlots * (4096 + 1) * sizeof(int32_t);
    out.verifier_pinned = commit + 16;
    out.verifier_history = kResidentMaxSlots * 4096 * sizeof(int32_t);
    out.mtp_device = mtp ? slots * draft_device : 0;
    out.mtp_pinned = mtp ? slots * draft_pinned : 0;
    out.mtp_owner = mtp ? slots * (kResidentQsaBytes + 3 * sizeof(uint64_t)) : 0;
    out.transfer_transient = 2 * session_device + cells * 3 * sizeof(int32_t) + (mtp ? draft_device : 0);
    out.init_transient = init_transient;
    return true;
}
} // namespace strata::plan
