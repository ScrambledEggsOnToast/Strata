#pragma once

#include "strata/core/conversation_cache.hpp"
#include <cstddef>

namespace strata::test {

inline bool equal(const core::ConversationKv& a, const core::ConversationKv& b) {
    return a.k == b.k && a.v == b.v && a.k_scale == b.k_scale &&
           a.v_scale == b.v_scale && a.pooled == b.pooled;
}

// Compare live request and authoritative K/V payloads, not residency metadata.
// Callers retain their scenario-specific assertion messages.
inline bool same_snapshot(const core::SavedConversation& a, const core::SavedConversation& b) {
    if (a.live.ids != b.live.ids || a.live.gdn != b.live.gdn || a.live.ple != b.live.ple ||
        a.live.tails != b.live.tails || a.live.dead != b.live.dead ||
        a.live.block_pos != b.live.block_pos || a.kv.size() != b.kv.size() ||
        a.draft_begin != b.draft_begin || a.draft_cells != b.draft_cells ||
        a.draft_next_token != b.draft_next_token || a.draft_boundary_residual != b.draft_boundary_residual) return false;
    for (std::size_t i = 0; i < a.kv.size(); ++i)
        if (!equal(a.kv[i], b.kv[i])) return false;
    return true;
}

}  // namespace strata::test
