// Experimental target-K/V sharing policy. No CUDA, environment reads or allocation.
#pragma once
#include "strata/plan/qsa_memory.hpp"
#include "strata/program/continuation.hpp"
#include <cstring>

namespace strata::plan {
enum class KvShareMode : uint8_t { off, pool, alias };
inline bool kv_share_mode(const char* text, KvShareMode& out) {
    out = KvShareMode::off;
    if (!text || !*text || std::strcmp(text, "off") == 0) return true;
    if (std::strcmp(text, "pool") == 0) { out = KvShareMode::pool; return true; }
    if (std::strcmp(text, "alias") == 0) { out = KvShareMode::alias; return true; }
    return false;
}
inline constexpr uint64_t kKvShareHostLimit = 2ull << 30;
inline constexpr uint64_t kKvShareChunk = 2ull << 20;
inline constexpr uint64_t kKvShareExtentBytes = 64;
inline constexpr uint64_t kKvShareBindingBytes = 16;
inline constexpr uint64_t kKvShareOwnerBytes = 128;
inline constexpr uint64_t kKvSharePoolBytes = 512;
// Process-wide driver entry/once caches and bounded streamed-read control blocks.
inline constexpr uint64_t kKvShareBackendHostBytes = 16ull << 10;

struct KvShareBudget {
    KvShareMode mode = KvShareMode::off;
    uint64_t cells = 0, owners = 0, layers = 0, stages = 0;
    uint64_t granularity = 0, cache_bytes = 0, cache_leases = 0;
};
struct KvShareQuote {
    bool known = false, admitted = false;
    const char* refusal = "unknown_layout";
    uint64_t complete_pages = 0, gross_kv_bytes = 0, rounded_release = 0;
    uint64_t host_metadata = 0, cow_host = 0, cow_device = 0, cow_payload = 0;
    uint64_t old_device = 0, common_device = 0, image_room = 0;
    int64_t net_steady = 0, net_cow_peak = 0;
};
// The quote retains ALL capacity records, even those an alias makes temporarily
// free. Unlike the original optimistic replay, freed vector capacity is not RAM
// released. Existing four-byte device page tables are not charged twice.
inline KvShareQuote kv_share_quote(const strata::kernels::QsaShapes& s, int format,
                                   const KvShareBudget& b, int64_t prefix) {
    KvShareQuote q;
    if (b.mode == KvShareMode::off) { q.known = q.admitted = true; q.refusal = "none"; return q; }
    if (s.page_size != 4 || s.n_head_kv != 2 || s.head_dim != 256 ||
        format != strata::kernels::kKvHybrid || !b.cells || b.cells > 262144 ||
        b.cells % 4 || b.owners < 2 || b.owners > 8 || !b.layers || b.layers > 48 ||
        !b.stages || b.stages > 4 || b.granularity != kKvShareChunk ||
        prefix < 0 || uint64_t(prefix) > b.cells) return q;
    LogicalBytes n;
    const uint64_t pages = b.cells / 4;
    const uint64_t all = n.product({b.owners, pages});
    q.complete_pages = uint64_t(prefix) / 4;
    const uint64_t unique = all - (b.mode == KvShareMode::alias ? q.complete_pages : 0);
    // K codes, K scales, V Q4_0. No indexer, recurrence, PLE or draft credit.
    constexpr uint64_t parts[] = {2048, 64, 1152};
    uint64_t common_layer = 0, directory_chunks = 0;
    for (uint64_t part : parts) {
        const uint64_t bytes = n.product({unique, part});
        const uint64_t rounded = n.sum({bytes, b.granularity - 1}) / b.granularity * b.granularity;
        common_layer = n.sum({common_layer, rounded});
        const uint64_t cow = n.sum({2 * part, b.granularity - 1}) / b.granularity * b.granularity;
        q.cow_device = n.sum({q.cow_device, n.product({b.layers, cow})});
        const uint64_t capacity_bytes = n.product({n.sum({all, 2}), part});
        directory_chunks = n.sum({directory_chunks,
            n.sum({capacity_bytes, b.granularity - 1}) / b.granularity});
    }
    q.old_device = n.product({b.layers, all, 3264});
    q.common_device = n.product({b.layers, common_layer});
    q.gross_kv_bytes = b.mode == KvShareMode::alias ? n.product({b.layers, q.complete_pages, 3264}) : 0;
    q.rounded_release = q.old_device >= q.common_device ? q.old_device - q.common_device : 0;
    q.cow_payload = n.product({2, b.layers, 3264});
    // Besides bindings, retain a stable int32 upload mirror, one vector/device
    // table pointer per owner, and per-chunk handle/occupancy directories. This
    // implementation owns descriptors per layer, not only per stage.
    q.host_metadata = n.sum({n.product({b.layers, all, kKvShareExtentBytes}),
        n.product({b.layers, all, kKvShareBindingBytes + sizeof(int32_t)}),
        n.product({b.layers, directory_chunks, sizeof(uint64_t) + sizeof(uint32_t)}),
        n.product({b.layers, kKvSharePoolBytes}),
        n.product({b.owners, b.layers, kKvShareOwnerBytes + uint64_t{32}}),
        n.product({b.owners, uint64_t{128}}), kKvShareBackendHostBytes});
    q.cow_host = n.product({2, b.layers, kKvShareExtentBytes + kKvShareBindingBytes});
    const uint64_t charged_host = n.sum({q.host_metadata, q.cow_host});
    const uint64_t host_used = n.sum({charged_host, b.cache_leases});
    if (!n.valid || std::max({q.old_device, q.common_device, charged_host, q.cow_device}) > INT64_MAX) return {};
    q.known = true;
    q.net_steady = int64_t(q.old_device) - int64_t(q.common_device) - int64_t(q.host_metadata);
    q.net_cow_peak = q.net_steady - int64_t(q.cow_device) - int64_t(q.cow_host);
    if (charged_host > kKvShareHostLimit) { q.refusal = "host_growth_limit"; return q; }
    if (host_used > b.cache_bytes) { q.refusal = "host_cache_envelope"; return q; }
    q.image_room = b.cache_bytes - host_used;
    if (b.mode == KvShareMode::alias && (q.net_steady <= 0 || q.net_cow_peak < 0)) {
        q.refusal = "negative_memory_first_peak"; return q;
    }
    q.admitted = true; q.refusal = "none";
    return q;
}
struct KvShareEligibility {
    KvShareMode mode = KvShareMode::off;
    uint64_t request = 0, incarnation = 0, allocation = 0;
    uint64_t expected_allocation = 0;
    int64_t prefix = 0, capacity = 0;
    bool same_pool = false, compatible = false, preparing_cancelled = false, drained = false;
};
inline bool kv_share_eligible(const KvShareEligibility& in,
                              const strata::program::ContinuationCertificate& cut) {
    return in.mode == KvShareMode::alias && !in.preparing_cancelled && in.drained &&
        in.same_pool && in.compatible && in.allocation && in.allocation == in.expected_allocation &&
        cut.certified && cut.qsa && cut.gdn && cut.ple && cut.stages &&
        in.request && in.incarnation && in.request == cut.request && in.incarnation == cut.incarnation &&
        cut.valid_begin == 0 && in.prefix >= 4 && in.prefix <= cut.valid_end &&
        cut.valid_end <= cut.committed_tokens && cut.committed_tokens <= in.capacity;
}
enum class KvCowDecision : uint8_t { private_write, detach, refuse };
inline KvCowDecision kv_share_cow(uint32_t references, bool generation_matches, bool drained,
                                  uint64_t pages_to_rewrite, uint64_t available_cow_pages) {
    if (!references || !generation_matches || !drained) return KvCowDecision::refuse;
    if (references == 1) return KvCowDecision::private_write;
    return pages_to_rewrite && pages_to_rewrite <= available_cow_pages ?
        KvCowDecision::detach : KvCowDecision::refuse;
}
// External pools omit only the K/V carve. The off branch calls the historical
// initializer unchanged, including alignment, private indexer and trailing slack.
inline QsaArenaLayout kv_share_arena_layout(const strata::kernels::QsaShapes& s,
        const QsaPlacement& p, bool rope, KvShareMode mode, uint64_t capacity = UINT64_MAX) {
    if (mode == KvShareMode::off) return qsa_arena_layout(s, p, rope, capacity);
    if (p.kv.mode || p.format != strata::kernels::kKvHybrid || p.elastic) return {};
    auto external = p;
    external.elastic = true;
    return qsa_arena_layout(s, external, rope, capacity);
}
} // namespace strata::plan
