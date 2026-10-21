#pragma once
// The row layout of a batch window (Verifier::stage_batch), as pure logic so a CPU test can pin it down.
//
// A batch window carries S rows, each a token of one slot.  A slot owns one row (plain --batch) or a contiguous group
// of rows at consecutive positions (--batch-mtp: the slot's current token and proposals; a native DFlash2 decode
// window has at most two owners of up to four rows each; a layer split's every stage sees the same layout).
// The rows travel between the stages of a layer split in hand-off buffers of
// `handoff_rows` rows, the window's group starting at row `hbase`.

#include <array>
#include <cstdint>

namespace strata::core {

/// nullptr when the layout is valid, else why not.  `rows[t]` is row t's slot, `pos[t]` its position.  `n_slots`: the
/// slots the verifier holds; `max_t`: its window capacity; `via_handoff`: the stage hands its rows on or takes them from
/// another stage (a layer split), so [hbase, hbase + S) must fit the `handoff_rows` rows of the hand-off buffers.
inline const char* batch_rows_error(const int* rows, int S, const int64_t* pos, int n_slots, int max_t, int hbase,
                                    bool via_handoff, int handoff_rows) {
    if (S < 1 || S > max_t || hbase < 0 || (via_handoff && hbase + S > handoff_rows))
        return "verify: batch rows out of range (init_slots)";
    for (int t = 0; t < S; ++t) {
        if (rows[t] < 0 || rows[t] >= n_slots) return "verify: a batch row's slot is out of range";
        for (int u = 0; u < t - 1; ++u)
            if (rows[u] == rows[t] && rows[t - 1] != rows[t]) return "verify: a slot's proposed rows must be contiguous";
        if (t > 0 && rows[t] == rows[t - 1] && pos[t] != pos[t - 1] + 1)
            return "verify: proposed rows must have consecutive positions";
    }
    return nullptr;
}

/// The exact window/commit graph-cache key, shared with Verifier and its CPU capacity proof.
/// `rows` and S must already describe a validated window of at most eight rows. The residency variant bit is
/// stored unchanged: Verifier passes ar_off_, so true selects its doorbell graph rather than the all-resident one.
using VerifyBatchKey = std::array<int, 11>;
inline constexpr VerifyBatchKey verify_batch_graph_key(const int* rows, int S, int hbase, bool has_residency) {
    VerifyBatchKey key{};
    key[0] = hbase;
    key[1] = S;
    key[2] = has_residency ? 1 : 0;
    for (int t = 0; t < S; ++t) key[3 + t] = rows[t];
    return key;
}

/// Native DFlash2's priced production families, without restricting the general row validator:
/// decode has at most two contiguous owners of depth 1..4; retained known-input target reads have
/// one owner of depth 1..8; final prompts have at most four distinct one-row owners, in any order.
/// Positions and handoff extents remain batch_rows_error's responsibility.
inline constexpr bool dflash2_batch_rows_supported(const int* rows, int S) {
    if (!rows || S < 1 || S > 8) return false;
    unsigned seen = 0;
    int owners = 0;
    bool depth_one = true, depth_at_most_four = true;
    for (int t = 0; t < S;) {
        const int owner = rows[t];
        if (owner < 0 || owner >= 4 || (seen & (1u << owner))) return false;
        seen |= 1u << owner;
        ++owners;
        const int first = t++;
        while (t < S && rows[t] == owner) ++t;
        depth_one = depth_one && t - first == 1;
        depth_at_most_four = depth_at_most_four && t - first <= 4;
    }
    return owners == 1 || (owners <= 2 && depth_at_most_four) || depth_one;
}

/// A slot's entire verified row range is invalid when any of its logits were nonfinite,
/// including a rejected draft row. Masks are local to the verifier owner, not handoff offsets.
inline bool batch_rows_nonfinite(uint32_t mask, int first, int count) {
    if (first < 0 || first >= 32 || count <= 0 || count > 32 - first) return false;
    const uint32_t rows = (UINT32_MAX >> (32 - count)) << first;
    return (mask & rows) != 0;
}

}  // namespace strata::core
