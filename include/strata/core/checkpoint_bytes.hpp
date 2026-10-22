#pragma once

#include "strata/core/progress.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

namespace strata::core {

inline bool immutable_checkpoints_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("STRATA_IMMUTABLE_CHECKPOINTS");
        return v && std::atoi(v) != 0;
    }();
    return enabled;
}

namespace checkpoint_detail {
// Only the explicit overwrite-resize uses zero-argument construction. Normal
// growth supplies a value, preserving zero-filled storage for other builders.
template<class T> struct OverwriteAllocator : std::allocator<T> {
    using value_type = T;
    template<class U> struct rebind { using other = OverwriteAllocator<U>; };
    OverwriteAllocator() noexcept = default;
    template<class U> OverwriteAllocator(const OverwriteAllocator<U>&) noexcept {}
    template<class U> bool operator==(const OverwriteAllocator<U>&) const noexcept { return true; }
    template<class U> void construct(U* p) {
        static_assert(std::is_same_v<U, uint8_t>);
        ::new (static_cast<void*>(p)) U;
    }
    template<class U, class... Args> void construct(U* p, Args&&... args) {
        std::construct_at(p, std::forward<Args>(args)...);
    }
};
}

// Builders own mutable bytes; published payloads have only const readers. Copies
// of a sealed payload retain its allocation. LRU, pins and token metadata remain
// in ConversationCheckpoint. A writer detaches before changing a sealed image.
// Builder references from write() must be retired before seal() or copying.
class CheckpointBytes {
public:
    using Storage = std::vector<uint8_t, checkpoint_detail::OverwriteAllocator<uint8_t>>;
    CheckpointBytes() = default;
    CheckpointBytes(Storage bytes) : building_(std::move(bytes)) {}
    CheckpointBytes(std::initializer_list<uint8_t> bytes) : building_(bytes) {}
    const Storage& read() const { return sealed_ ? *sealed_ : building_; }
    Storage& write() {
        if (sealed_) {
            Storage detached;
            detached.reserve(sealed_->size());
            constexpr size_t step = 1u << 20;
            for (size_t at = 0; at < sealed_->size(); at += step) {
                detached.insert(detached.end(), sealed_->begin() + at,
                                sealed_->begin() + std::min(sealed_->size(), at + step));
                progress_beat();
            }
            building_ = std::move(detached);
            sealed_.reset();
        }
        return building_;
    }
    void seal() {
        if (!sealed_ && !building_.empty())
            sealed_ = std::make_shared<const Storage>(std::move(building_));
    }
    bool sealed() const { return bool(sealed_); }
    size_t size() const { return read().size(); }
    size_t capacity() const { return read().capacity(); }
    bool empty() const { return read().empty(); }
    const uint8_t* data() const { return read().data(); }
    const uint8_t& operator[](size_t i) const { return read()[i]; }
    void resize(size_t n, uint8_t v = 0) { write().resize(n, v); }
    // The caller must overwrite every newly exposed byte before any read,
    // seal or publication. A failed fill must discard the unpublished builder.
    void resize_for_overwrite(size_t n) { write().resize(n); }
    void assign(size_t n, uint8_t v) { write().assign(n, v); }
    void push_back(uint8_t v) { write().push_back(v); }
    void pop_back() { write().pop_back(); }
    // Charge each reference in full (conservative), including the shared owner.
    // As with existing vector accounting this excludes allocator bookkeeping.
    static constexpr size_t owner_bytes = sizeof(Storage) + sizeof(std::shared_ptr<const void>);
    size_t bytes() const { return capacity() + (sealed_ ? owner_bytes : 0); }
    size_t freeze_growth_bytes() const { return !sealed_ && !empty() ? owner_bytes : 0; }
    bool operator==(const CheckpointBytes& b) const { return read() == b.read(); }
    friend bool operator==(const std::vector<uint8_t>& a, const CheckpointBytes& b) {
        return a.size() == b.size() && std::equal(a.begin(), a.end(), b.read().begin());
    }
    friend bool operator==(const CheckpointBytes& a, const std::vector<uint8_t>& b) { return b == a; }
private:
    Storage building_;
    std::shared_ptr<const Storage> sealed_;
};

} // namespace strata::core
