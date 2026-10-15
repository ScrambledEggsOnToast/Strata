// Incremental reuse of an already-stored image: the byte intervals a capture actually kept from the image whose K/V
// it retained.  CPU-only metadata: it reads no payload, hashes no payload and copies no payload, and it never claims
// a byte the capture did not itself keep (conversation_snapshot's retained prefixes).
//
// The proof is structural, not a comparison.  `conversation_snapshot_save` keeps the leading bytes of a retained
// K/V part and copies the rest, so for those bytes the outgoing part is the *same allocation* (or, after a detach,
// the same bytes) as the donor image's own part at the same stream offsets.  A stream identity is a stable name -
// the store maps it to a physical identity and refuses a donor that maps elsewhere - so the interval below is a
// claim the store re-validates: donor still authorized, in this bucket, captured in this epoch, every adopted chunk
// still present, and every boundary on the donor's own chunk grid.
#pragma once

#include "strata/core/conversation_store.hpp"

#include <cstdint>
#include <vector>

namespace strata::core {

// One K/V part of one stage that a capture kept from the image it retained: `bytes` of the head of that part are the
// donor's own bytes, out of `donor_bytes` in the donor's image.  `stage` is the caller's own tagging of the capture
// (0 = the image, k+1 = its k-th stage image); the capture module reports the layer and part and leaves `stage` 0.
struct ConversationRetainedPrefix {
    uint32_t stage = 0;
    uint32_t layer = 0;   // the K/V layer index: the store's stream ordinal for every kv_* kind
    uint32_t part = 0;    // 0 = k, 1 = v, 2 = k_scale, 3 = v_scale, 4 = pooled (the order of the kv_* kinds)
    uint64_t bytes = 0;
    uint64_t donor_bytes = 0;   // the donor's own length of this part (the retained buffer's size at capture)
};

// The reuse intervals this capture may claim for `donor_head` (0 = none, and an epoch mismatch claims nothing: a
// donor from a retired execution domain is not authorized to be adopted).
//
// `streams` is the *outgoing* image's canonical stream table (conversation_store_streams), so the interval names the
// same stream identity the store will look up.  A stream whose identity is absent is skipped rather than guessed.
// `chunk_bytes` is the store's own object size: K/V part streams are byte streams (element 1), so every donor chunk
// is exactly one object except the last one, and an interval that ends on a chunk boundary of both streams is
// adoptable as a whole number of chunks.  The one interval that may end off that grid is one that ends *both*
// streams: an immutable chunk is never partially rewritten, and a whole-stream retain ends on the donor's own last
// chunk.
inline std::vector<ConversationStoreReuse> conversation_store_reuse_intervals(
    const std::vector<ConversationStoreStreamInfo>& streams, const std::vector<ConversationRetainedPrefix>& retained,
    uint64_t donor_head, uint64_t donor_epoch, uint64_t capture_epoch, uint64_t chunk_bytes) {
    std::vector<ConversationStoreReuse> out;
    if (donor_head == 0 || capture_epoch == 0 || donor_epoch != capture_epoch) return out;
    const uint64_t chunk = chunk_bytes < 4096 ? 4096 : chunk_bytes;
    if (retained.empty() || streams.empty()) return out;
    out.reserve(retained.size());
    for (const ConversationRetainedPrefix& part : retained) {
        if (part.bytes == 0 || part.part > 4) continue;
        const ConversationStoreStreamId id{part.stage,
                                          uint16_t(uint16_t(ConversationStoreStreamKind::kv_k) + part.part), 0,
                                          part.layer};
        const ConversationStoreStreamInfo* info = nullptr;
        for (const ConversationStoreStreamInfo& stream : streams)
            if (stream.id == id) {
                info = &stream;
                break;
            }
        if (info == nullptr) continue;
        const uint64_t proven = part.bytes < part.donor_bytes ? part.bytes : part.donor_bytes;
        if (proven == 0 || proven > info->bytes) continue;
        // Ends on the donor's own last chunk and on this image's end as well: no alignment is needed, and no chunk
        // is partially rewritten.  Otherwise only a whole number of objects may be adopted.
        const bool whole_streams = proven == part.donor_bytes && proven == info->bytes;
        const uint64_t bytes = whole_streams ? proven : (proven / chunk) * chunk;
        if (bytes == 0) continue;
        ConversationStoreReuse interval;
        interval.head_id = donor_head;
        interval.stream = id;
        interval.source_offset = 0;
        interval.offset = 0;
        interval.bytes = bytes;
        out.push_back(interval);
    }
    return out;
}

}  // namespace strata::core
