#pragma once

#include "strata/core/progress.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <memory>
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

// Builders own mutable bytes; published payloads have only const readers. Copies
// of a sealed payload retain its allocation. LRU, pins and token metadata remain
// in ConversationCheckpoint. A writer detaches before changing a sealed image.
// Builder references from write() must be retired before seal() or copying.
class CheckpointBytes {
public:
    CheckpointBytes() = default;
    CheckpointBytes(std::vector<uint8_t> bytes) : building_(std::move(bytes)) {}
    CheckpointBytes(std::initializer_list<uint8_t> bytes) : building_(bytes) {}
    const std::vector<uint8_t>& read() const { return sealed_ ? *sealed_ : building_; }
    std::vector<uint8_t>& write() {
        if (sealed_) {
            std::vector<uint8_t> detached;
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
            sealed_ = std::make_shared<const std::vector<uint8_t>>(std::move(building_));
    }
    bool sealed() const { return bool(sealed_); }
    size_t size() const { return read().size(); }
    size_t capacity() const { return read().capacity(); }
    bool empty() const { return read().empty(); }
    const uint8_t* data() const { return read().data(); }
    const uint8_t& operator[](size_t i) const { return read()[i]; }
    void resize(size_t n, uint8_t v = 0) { write().resize(n, v); }
    void assign(size_t n, uint8_t v) { write().assign(n, v); }
    void push_back(uint8_t v) { write().push_back(v); }
    void pop_back() { write().pop_back(); }
    // Charge each reference in full (conservative), including the shared owner.
    // As with existing vector accounting this excludes allocator bookkeeping.
    static constexpr size_t owner_bytes = sizeof(std::vector<uint8_t>) + sizeof(std::shared_ptr<const void>);
    size_t bytes() const { return capacity() + (sealed_ ? owner_bytes : 0); }
    bool operator==(const CheckpointBytes& b) const { return read() == b.read(); }
    friend bool operator==(const std::vector<uint8_t>& a, const CheckpointBytes& b) { return a == b.read(); }
    friend bool operator==(const CheckpointBytes& a, const std::vector<uint8_t>& b) { return a.read() == b; }
private:
    std::vector<uint8_t> building_;
    std::shared_ptr<const std::vector<uint8_t>> sealed_;
};

} // namespace strata::core
