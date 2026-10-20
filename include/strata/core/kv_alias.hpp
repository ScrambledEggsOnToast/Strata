// Generation-fenced common-pool ownership. One host submitter; no CUDA calls.
#pragma once
#include "strata/plan/kv_share.hpp"
#include <array>
#include <atomic>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace strata::core {
struct KvOwnerKey {
    uint64_t request = 0, incarnation = 0, allocation = 0;
    bool valid() const { return request && incarnation && allocation; }
    bool operator==(const KvOwnerKey& b) const {
        return request == b.request && incarnation == b.incarnation && allocation == b.allocation;
    }
};
struct KvPageBinding {
    uint64_t extent = UINT64_MAX, generation = 0;
    bool operator==(const KvPageBinding& b) const { return extent == b.extent && generation == b.generation; }
};
struct KvExtentRecord {
    uint64_t generation = 0, acquires = 0, releases = 0;
    uint64_t reserved[4]{};
    uint32_t references = 0, reserved32 = 0;
};
struct KvAliasOwner {
    KvOwnerKey key;
    uint64_t certified_pages = 0;
    uint64_t pending_source = UINT64_MAX, pending_pages = 0, pending_page = 0;
    KvPageBinding pending_private;
    uint64_t pins = 0;
    bool active = false, preparing = false, cancelled = false, retained = false;
    uint8_t reserved[44]{};
};
static_assert(sizeof(KvPageBinding) == plan::kKvShareBindingBytes);
static_assert(sizeof(KvExtentRecord) == plan::kKvShareExtentBytes);
static_assert(sizeof(KvAliasOwner) == plan::kKvShareOwnerBytes);

// One controller acquires/advances; a finished I/O reader may release from a
// different thread. A same-key reset still advances the lifetime generation.
class KvReadPins {
public:
    bool advance(uint64_t owner) {
        if (owner >= slots_.size() || pinned(owner) || generation(owner) == UINT64_MAX) return false;
        slots_[owner].generation.fetch_add(1, std::memory_order_release); return true;
    }
    uint64_t acquire(uint64_t owner) {
        if (owner >= slots_.size() || !generation(owner) ||
            slots_[owner].count.load(std::memory_order_relaxed) == UINT64_MAX) return 0;
        slots_[owner].count.fetch_add(1, std::memory_order_acq_rel);
        return generation(owner);
    }
    bool release(uint64_t owner, uint64_t expected) {
        if (!expected || owner >= slots_.size() || generation(owner) != expected) return false;
        auto count = slots_[owner].count.load(std::memory_order_acquire);
        while (count) {
            if (slots_[owner].count.compare_exchange_weak(count, count - 1,
                    std::memory_order_acq_rel, std::memory_order_acquire)) return true;
        }
        return false;
    }
    uint64_t generation(uint64_t owner) const {
        return owner < slots_.size() ? slots_[owner].generation.load(std::memory_order_acquire) : 0;
    }
    bool pinned(uint64_t owner) const {
        return owner >= slots_.size() || slots_[owner].count.load(std::memory_order_acquire) != 0;
    }
    bool any() const {
        for (const auto& slot : slots_) if (slot.count.load(std::memory_order_acquire)) return true;
        return false;
    }
private:
    struct Slot { std::atomic<uint64_t> generation{0}, count{0}; };
    std::array<Slot, 8> slots_{};
};

// Acquire precedes device publication. bind/detach_commit/abort/terminal require
// the adapter's existing drain, and consume the pending references exactly once.
// cancel only sets a flag: it never drops a reference or certifies a drain.
class KvAliasState {
public:
    KvAliasState(uint64_t owners, uint64_t pages) : pages_(pages), owners_((size_t) checked_owners(owners, pages)),
        extents_((size_t) (owners * pages + 2)), bindings_((size_t) (owners * pages)) {}
    KvAliasState(const KvAliasState&) = delete;
    KvAliasState& operator=(const KvAliasState&) = delete;
    bool owns(uint64_t owner, const KvOwnerKey& key) const {
        return owner < owners_.size() && key.valid() && owners_[(size_t) owner].active &&
            owners_[(size_t) owner].key == key;
    }
    bool begin(uint64_t owner, const KvOwnerKey& key) {
        if (owner >= owners_.size() || !key.valid() || owners_[(size_t) owner].active ||
            free_pages() < pages_) return false;
        auto& o = owners_[(size_t) owner];
        o = {}; o.key = key; o.active = o.preparing = true;
        for (uint64_t page = 0; page < pages_; ++page) at(owner, page) = new_extent();
        return true;
    }
    bool ready(uint64_t owner, const KvOwnerKey& key) {
        if (!owns(owner, key) || owners_[(size_t) owner].cancelled || pending(owner)) return false;
        owners_[(size_t) owner].preparing = false;
        return true;
    }
    bool cancel(uint64_t owner, const KvOwnerKey& key) {
        if (!owns(owner, key) || owners_[(size_t) owner].retained) return false;
        owners_[(size_t) owner].cancelled = true;
        return true;
    }
    bool acquire(uint64_t target, const KvOwnerKey& target_key, uint64_t source, const KvOwnerKey& source_key,
                 int64_t prefix, const program::ContinuationCertificate& certificate, bool drained) {
        if (source == target || !owns(source, source_key) || !owns(target, target_key) ||
            pending(target) || pending(source) || owners_[(size_t) source].preparing ||
            owners_[(size_t) source].cancelled || owners_[(size_t) target].cancelled ||
            owners_[(size_t) target].retained || owners_[(size_t) target].pins) return false;
        const plan::KvShareEligibility in{plan::KvShareMode::alias, source_key.request, source_key.incarnation,
            source_key.allocation, source_key.allocation, prefix, int64_t(pages_ * 4), true, true, false, drained};
        if (!plan::kv_share_eligible(in, certificate)) return false;
        const uint64_t count = uint64_t(prefix) / 4;
        for (uint64_t p = 0; p < count; ++p) {
            const auto b = at(source, p);
            if (!valid(b) || extents_[(size_t) b.extent].references == UINT32_MAX) {
                // No publication happened. Roll back exactly, including audit
                // counters, instead of paying a second full validation pass.
                for (uint64_t prior = 0; prior < p; ++prior) {
                    auto& e = extents_[(size_t) at(source, prior).extent];
                    --e.references; --e.acquires;
                }
                return false;
            }
            add(b);
        }
        auto& o = owners_[(size_t) target];
        o.pending_source = source; o.pending_pages = count;
        ++owners_[(size_t) source].pins;
        return true;
    }
    bool bind(uint64_t target, const KvOwnerKey& key, bool drained) {
        if (!owns(target, key) || !drained) return false;
        auto& o = owners_[(size_t) target];
        if (o.pending_source == UINT64_MAX || !o.pending_pages) return false;
        const uint64_t source = o.pending_source;
        for (uint64_t p = 0; p < o.pending_pages; ++p) drop(at(target, p));
        // Owners are distinct; the acquired generation bindings are a POD run.
        std::memcpy(&at(target, 0), &at(source, 0), (size_t) o.pending_pages * sizeof(KvPageBinding));
        o.certified_pages = o.pending_pages;
        --owners_[(size_t) source].pins;
        o.pending_source = UINT64_MAX; o.pending_pages = 0;
        return true;
    }
    plan::KvCowDecision cow(uint64_t owner, const KvOwnerKey& key, uint64_t page, bool drained) const {
        if (!owns(owner, key) || page >= pages_ || owners_[(size_t) owner].cancelled ||
            owners_[(size_t) owner].retained || owners_[(size_t) owner].pins || pending(owner)) return plan::KvCowDecision::refuse;
        const auto b = at(owner, page);
        return plan::kv_share_cow(valid(b) ? extents_[(size_t) b.extent].references : 0,
                                  valid(b), drained, 1, free_pages() ? 1 : 0);
    }
    bool detach(uint64_t owner, const KvOwnerKey& key, uint64_t page, bool drained) {
        if (cow(owner, key, page, drained) != plan::KvCowDecision::detach) return false;
        auto& o = owners_[(size_t) owner];
        o.pending_private = new_extent(); o.pending_page = page;
        return true;
    }
    bool detach_commit(uint64_t owner, const KvOwnerKey& key, bool drained) {
        if (!owns(owner, key) || !drained) return false;
        auto& o = owners_[(size_t) owner];
        if (!valid(o.pending_private)) return false;
        drop(at(owner, o.pending_page));
        at(owner, o.pending_page) = o.pending_private;
        o.certified_pages = std::min(o.certified_pages, o.pending_page);
        o.pending_private = {}; o.pending_page = 0;
        return true;
    }
    bool abort(uint64_t owner, const KvOwnerKey& key, bool drained) {
        if (!owns(owner, key) || !drained) return false;
        auto& o = owners_[(size_t) owner];
        if (o.pending_source != UINT64_MAX) {
            for (uint64_t p = 0; p < o.pending_pages; ++p) drop(at(o.pending_source, p));
            --owners_[(size_t) o.pending_source].pins;
            o.pending_source = UINT64_MAX; o.pending_pages = 0;
        }
        if (valid(o.pending_private)) drop(o.pending_private);
        o.pending_private = {}; o.pending_page = 0;
        return true;
    }
    bool terminal(uint64_t owner, const KvOwnerKey& key, bool drained, bool retain = false) {
        if (!owns(owner, key) || !drained || owners_[(size_t) owner].pins || pending(owner)) return false;
        auto& o = owners_[(size_t) owner];
        if (retain) {
            if (o.preparing || o.cancelled || o.retained || !o.certified_pages) return false;
            o.retained = true;
            return true;
        }
        for (uint64_t p = 0; p < pages_; ++p) { drop(at(owner, p)); at(owner, p) = {}; }
        o = {};
        return true;
    }
    const KvPageBinding& binding(uint64_t owner, uint64_t page) const { return at(owner, page); }
    const KvAliasOwner& owner(uint64_t index) const { return owners_[(size_t) index]; }
    const KvExtentRecord& extent(uint64_t index) const { return extents_[(size_t) index]; }
    uint64_t extent_count() const { return extents_.size(); }
    uint64_t pages() const { return pages_; }
    bool valid(const KvPageBinding& b) const {
        return b.extent < extents_.size() && b.generation &&
            extents_[(size_t) b.extent].generation == b.generation && extents_[(size_t) b.extent].references;
    }
    uint64_t host_bytes() const {
        return sizeof(*this) + owners_.capacity() * sizeof(KvAliasOwner) +
            extents_.capacity() * sizeof(KvExtentRecord) + bindings_.capacity() * sizeof(KvPageBinding);
    }
    uint64_t free_pages() const { return extents_.size() - live_ - exhausted_; }
private:
    static uint64_t checked_owners(uint64_t owners, uint64_t pages) {
        if (owners < 2 || owners > 8 || !pages || pages > 65536)
            throw std::invalid_argument("KV alias state: unpriced owner/page capacity");
        return owners;
    }
    bool pending(uint64_t owner) const {
        const auto& o = owners_[(size_t) owner];
        return o.pending_source != UINT64_MAX || o.pending_private.extent != UINT64_MAX;
    }
    KvPageBinding new_extent() {
        for (uint64_t count = 0; count < extents_.size(); ++count) {
            const uint64_t id = next_++ % extents_.size();
            auto& e = extents_[(size_t) id];
            if (e.references || e.generation == UINT64_MAX) continue;
            ++e.generation; ++e.acquires; e.references = 1; ++live_;
            return {id, e.generation};
        }
        return {};
    }
    void add(const KvPageBinding& b) { auto& e = extents_[(size_t) b.extent]; ++e.references; ++e.acquires; }
    void drop(const KvPageBinding& b) {
        auto& e = extents_[(size_t) b.extent];
        ++e.releases;
        if (!--e.references) {
            --live_;
            if (e.generation == UINT64_MAX) ++exhausted_;
        }
    }
    KvPageBinding& at(uint64_t owner, uint64_t page) { return bindings_[(size_t) (owner * pages_ + page)]; }
    const KvPageBinding& at(uint64_t owner, uint64_t page) const { return bindings_[(size_t) (owner * pages_ + page)]; }
    uint64_t pages_ = 0, live_ = 0, next_ = 0, exhausted_ = 0;
    std::vector<KvAliasOwner> owners_;
    std::vector<KvExtentRecord> extents_;
    std::vector<KvPageBinding> bindings_;
};
} // namespace strata::core
