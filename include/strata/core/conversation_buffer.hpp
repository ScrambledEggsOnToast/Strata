#pragma once

#include "strata/core/progress.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <initializer_list>
#include <utility>
#include <vector>

namespace strata::core {

inline bool prefix_backing_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("STRATA_PREFIX_BACKING");
        return v && std::atoi(v) != 0;
    }();
    return enabled;
}

// Snapshot-owned host storage. Opt-in 64 KiB extents can share complete immutable
// allocations; mutable visits detach first. Admission charges every reference in
// full AND the worst-case detach peak. Ordinary copies remain deep copies.
class ConversationBuffer {
public:
    static constexpr size_t segment_bytes = 16 * 1024 * 1024;
    static constexpr size_t prefix_extent_bytes = 64 * 1024;
    // Explicit policy is also used by CPU fixtures without process environment changes.
    ConversationBuffer() : ConversationBuffer(prefix_backing_enabled()) {}
    explicit ConversationBuffer(bool sharing) : sharing_(sharing) {}
    ConversationBuffer(std::initializer_list<uint8_t> values) : ConversationBuffer() {
        resize(values.size());
        visit(0, size(), [&](uint8_t* p, size_t n, size_t at) {
            std::memcpy(p, values.begin() + at, n);
            return true;
        });
    }
    ConversationBuffer(const ConversationBuffer&) = default;
    ConversationBuffer& operator=(const ConversationBuffer&) = default;
    ConversationBuffer(ConversationBuffer&& other) noexcept
        : segments_(std::move(other.segments_)), size_(std::exchange(other.size_, 0)), sharing_(other.sharing_) {}
    ConversationBuffer& operator=(ConversationBuffer&& other) noexcept {
        if (this == &other) return *this;
        segments_ = std::move(other.segments_);
        size_ = std::exchange(other.size_, 0);
        sharing_ = other.sharing_;
        return *this;
    }

    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    size_t bytes() const {
        size_t n = segments_.capacity() * sizeof(Segment);
        for (const auto& segment : segments_) n += segment.bytes();
        return n;
    }
    // Includes the old directory while reserve allocates its replacement.
    // Saturation makes an overflowing estimate fail admission before resize.
    size_t allocation_peak(size_t n) const {
        size_t total = bytes();
        // A rewrite or partial shrink can detach every shared allocation. Count
        // the old backing and all replacements simultaneously, even for shrink.
        for (const auto& segment : segments_)
            if (segment.shared() && !add(total, segment.bytes())) return SIZE_MAX;
        if (n <= size_) return total;
        const size_t limit = extent_bytes();
        const size_t spare = segments_.empty() ? 0 : segments_.back().capacity() - segments_.back().size();
        const size_t extra = n - size_ - std::min(n - size_, spare);
        const size_t full = extra / limit, tail = extra % limit;
        if (!add(total, full * limit)) return SIZE_MAX;
        if (tail && !add(total, segment_capacity(tail, full ? limit :
                segments_.empty() ? 0 : segments_.back().capacity(), size_ != 0))) return SIZE_MAX;
        const size_t new_segments = full + (tail != 0);
        if (sharing_ && (new_segments > SIZE_MAX / Segment::owner_bytes ||
            !add(total, new_segments * Segment::owner_bytes))) return SIZE_MAX;
        size_t count = segments_.size();
        if (!add(count, full + (tail != 0))) return SIZE_MAX;
        if (count > segments_.capacity() &&
            (count > SIZE_MAX / sizeof(Segment) || !add(total, count * sizeof(Segment)))) return SIZE_MAX;
        return total;
    }
    void resize(size_t n, uint8_t value = 0) {
        if (n > size_) {
            const size_t limit = extent_bytes();
            const bool growing = size_ != 0;
            const size_t spare = segments_.empty() ? 0 : segments_.back().capacity() - segments_.back().size();
            const size_t extend = std::min(n - size_, spare);
            const size_t extra = n - size_ - extend;
            segments_.reserve(segments_.size() + extra / limit + (extra % limit != 0));
            if (extend) {
                segments_.back().resize(segments_.back().size() + extend, value);
                size_ += extend;
            }
            while (size_ < n) {
                const size_t count = std::min(limit, n - size_);
                const size_t previous = segments_.empty() ? 0 : segments_.back().capacity();
                Segment segment(sharing_);
                segment.reserve(segment_capacity(count, previous, growing));
                segment.resize(count, value);
                segments_.push_back(std::move(segment));
                size_ += count;
                progress_beat();   // #2049: zero-filling 16 MiB of fresh pages is slow when the host is paging
            }
        } else {
            size_t retained = size_, count = segments_.size();
            while (count && retained - segments_[count - 1].size() >= n)
                retained -= segments_[--count].size();
            // Detach the surviving partial extent before discarding any tail.
            // A failed detach must leave this receiver, not just its donors, intact.
            if (retained > n)
                segments_[count - 1].resize(segments_[count - 1].size() - (retained - n));
            segments_.resize(count);
            size_ = n;
        }
    }
    void pop_back() { resize(size_ - 1); }

    // fn(pointer, count, absolute_offset); false aborts a transfer immediately.
    // Mutable pointers must be retired before another mutation or share_prefix.
    // Production callers perform synchronous transfers and never retain writers.
    template<class Fn> bool visit(size_t offset, size_t count, Fn fn) {
        return visit_impl(*this, offset, count, fn);
    }
    template<class Fn> bool visit(size_t offset, size_t count, Fn fn) const {
        return visit_impl(*this, offset, count, fn);
    }
    bool read(void* target, size_t offset, size_t count) const {
        return visit(offset, count, [&](const uint8_t* p, size_t n, size_t at) {
            std::memcpy(static_cast<uint8_t*>(target) + at - offset, p, n);
            return true;
        });
    }
    bool operator==(const ConversationBuffer& other) const {
        if (size_ != other.size_) return false;
        size_t a = 0, b = 0, x = 0, y = 0;
        while (a < segments_.size() && b < other.segments_.size()) {
            const auto& left = segments_[a];
            const auto& right = other.segments_[b];
            const size_t n = std::min(left.size() - x, right.size() - y);
            if (std::memcmp(left.data() + x, right.data() + y, n)) return false;
            x += n; y += n;
            if (x == left.size()) { ++a; x = 0; }
            if (y == right.size()) { ++b; y = 0; }
        }
        return true;
    }

    // Caller must authorize the complete checkpoint/layout first. This method
    // independently verifies bytes and only replaces whole, exact-capacity
    // extents with matching boundaries. No prefix/tail payload is relocated.
    size_t share_prefix(const ConversationBuffer& source, size_t cut) {
        if (!sharing_ || !source.sharing_ || cut > size_ || cut > source.size_) return 0;
        size_t retained = 0, at = 0, j = 0, source_at = 0;
        for (auto& segment : segments_) {
            const size_t n = segment.size();
            if (n > cut - at) break;
            while (j < source.segments_.size() && source_at < at)
                source_at += source.segments_[j++].size();
            if (j == source.segments_.size()) break;
            const auto& donor = source.segments_[j];
            if (source_at == at && n == prefix_extent_bytes && segment.capacity() == n &&
                donor.size() == n && donor.capacity() == n && donor.owner &&
                segment.owner != donor.owner &&
                std::memcmp(std::as_const(segment).data(), donor.data(), n) == 0) {
                segment.owner = donor.owner;
                retained += n;
                progress_beat();
            }
            at += n;
        }
        return retained;
    }
    // Freeze for immutable sharing (the NVMe store's publication lease): every
    // payload moves into a shared allocation - no byte is copied and this buffer
    // keeps reading exactly the bytes it held. A later mutation detaches this
    // side's allocation first, so an alias taken after this call never observes a
    // byte written here afterwards. Each frozen segment is charged one owner.
    void freeze() {
        for (auto& segment : segments_) segment.freeze();
    }
    bool frozen() const {
        for (const auto& segment : segments_) if (!segment.frozen()) return false;
        return true;
    }
    // A cheap alias: every payload is shared with `source`, which must be frozen
    // (a segment that is not frozen yet is copied - still correct, just not free).
    // Neither side may be mutated without the detach that mutation performs.
    ConversationBuffer frozen_alias() const {
        ConversationBuffer out(sharing_);
        out.segments_.reserve(segments_.size());
        for (const auto& segment : segments_) out.segments_.push_back(segment.frozen_alias());
        out.size_ = size_;
        return out;
    }
    // Allocation identities, not pairwise byte comparisons. Visitors cannot
    // mutate payload. Used by the bounded, opt-in retained-backing diagnostic.
    template<class Fn> void for_each_backing(Fn fn) const {
        for (const auto& s : segments_)
            if (!fn(s.data(), s.capacity(), s.owner ? Segment::owner_bytes : 0)) break;
    }

private:
    struct Segment {
        std::vector<uint8_t> local;
        std::shared_ptr<std::vector<uint8_t>> owner;
        // Same accounting convention as CheckpointBytes: vector owner plus two
        // pointers for the shared control block, excluding allocator bookkeeping.
        static constexpr size_t owner_bytes = sizeof(std::vector<uint8_t>) + 2 * sizeof(void*);
        explicit Segment(bool sharing = false) {
            if (sharing) owner = std::make_shared<std::vector<uint8_t>>();
        }
        Segment(const Segment& other) : Segment(bool(other.owner)) { write() = other.read(); }
        Segment& operator=(const Segment& other) {
            if (this != &other) { Segment copy(other); *this = std::move(copy); }
            return *this;
        }
        Segment(Segment&&) noexcept = default;
        Segment& operator=(Segment&&) noexcept = default;
        const std::vector<uint8_t>& read() const { return owner ? *owner : local; }
        std::vector<uint8_t>& write() {
            if (shared()) {
                // Allocate/copy first: failure leaves this and every sibling intact.
                auto next = std::make_shared<std::vector<uint8_t>>();
                next->reserve(owner->capacity());
                next->assign(owner->begin(), owner->end());
                owner = std::move(next);
                progress_beat();
            }
            return owner ? *owner : local;
        }
        bool shared() const { return owner && owner.use_count() > 1; }
        // Move the payload into a shared allocation (O(1); the local vector is
        // left empty). A write through any other reference copies first.
        void freeze() {
            if (!owner) owner = std::make_shared<std::vector<uint8_t>>(std::move(local));
        }
        bool frozen() const { return bool(owner); }
        Segment frozen_alias() const {
            Segment out(false);   // no empty owner to replace
            if (owner) out.owner = owner;
            else out.local = local;   // not frozen: a real copy, still correct
            return out;
        }
        size_t size() const { return read().size(); }
        size_t capacity() const { return read().capacity(); }
        size_t bytes() const { return capacity() + (owner ? owner_bytes : 0); }
        const uint8_t* data() const { return read().data(); }
        uint8_t* data() { return write().data(); }
        void reserve(size_t n) { write().reserve(n); }
        void resize(size_t n, uint8_t v = 0) { write().resize(n, v); }
    };
    size_t extent_bytes() const { return sharing_ ? prefix_extent_bytes : segment_bytes; }
    // A fresh capture allocates exactly its payload. Later appends reserve
    // geometrically growing segments, bounded by 16 MiB. Small chat turns then
    // extend the final allocation instead of adding one transfer per turn.
    size_t segment_capacity(size_t count, size_t previous, bool growing) const {
        if (!growing) return count;
        return std::max(count, std::min(extent_bytes(), std::max<size_t>(65536, std::min(previous, extent_bytes()/2)*2)));
    }
    static bool add(size_t& n, size_t extra) {
        if (extra > SIZE_MAX - n) return false;
        n += extra;
        return true;
    }
    template<class Self, class Fn> static bool visit_impl(Self& self, size_t offset, size_t count, Fn fn) {
        if (offset > self.size_ || count > self.size_ - offset) return false;
        size_t begin = 0;
        for (auto& segment : self.segments_) {
            if (!count) break;
            const size_t end = begin + segment.size();
            if (offset < end) {
                const size_t n = std::min(count, end - offset);
                if (!fn(segment.data() + (offset - begin), n, offset)) return false;
                offset += n; count -= n;
            }
            begin = end;
        }
        return count == 0;
    }
    std::vector<Segment> segments_;
    size_t size_ = 0;
    bool sharing_ = false;
};

} // namespace strata::core
