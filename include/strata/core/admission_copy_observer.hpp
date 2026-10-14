#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::core::admission_copy {
using Clock = std::chrono::steady_clock;
enum Part { gdn, ple, index_tail, index_dead, index_position, index_repair,
            key, value, key_scale, value_scale, index_pooled, other, part_count };
enum Direction { capture, restore, direct, direction_count };
struct Counter { uint64_t bytes, calls, failed, ns; };
class Scope;
inline thread_local Scope* current = nullptr;
inline thread_local uint64_t next_scope = 0;
inline bool enabled() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_ADMISSION_COPY_BYTES");
        return v && std::strcmp(v, "0") != 0;
    }();
    return on;
}

// One synchronous stage transfer, never a new device owner or allocation.
// Counts actual successful transfer calls, including index spare-row repair.
class Scope {
public:
    Scope(const char* operation, int slot, int device, int64_t cells)
        : operation_(operation), slot_(slot), device_(device), cells_(cells), previous_(current), on_(enabled()) {
        if (on_) { counters = {}; id_ = ++next_scope; start_ = Clock::now(); current = this; }
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
    void complete() { complete_ = true; }
    ~Scope() {
        if (!on_) return;
        const auto wall = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start_).count();
        current = previous_;
        static constexpr const char* names[] = {"gdn", "ple", "index_tail", "index_dead", "index_position",
            "index_repair", "key", "value", "key_scale", "value_scale", "index_pooled", "other"};
        std::fprintf(stderr, "ADMISSION_COPY id=%llu operation=%s slot=%d device=%d cells=%lld complete=%d wall_ns=%lld\n",
                     (unsigned long long)id_, operation_, slot_, device_, (long long)cells_, complete_ ? 1 : 0, (long long)wall);
        static constexpr const char* directions[] = {"capture", "restore", "direct"};
        for (int d = 0; d < direction_count; ++d) for (int p = 0; p < part_count; ++p) {
            const auto& c = counters[d * part_count + p];
            if (!c.calls) continue;
            std::fprintf(stderr, "ADMISSION_COPY_PART id=%llu operation=%s slot=%d device=%d cells=%lld "
                         "direction=%s state=%s completed_bytes=%llu calls=%llu failed=%llu host_copy_ns=%llu\n",
                         (unsigned long long)id_, operation_, slot_, device_, (long long)cells_, directions[d], names[p],
                         (unsigned long long)c.bytes, (unsigned long long)c.calls,
                         (unsigned long long)c.failed, (unsigned long long)c.ns);
        }
    }
    std::array<Counter, direction_count * part_count> counters;
private:
    const char* operation_;
    int slot_, device_;
    int64_t cells_;
    uint64_t id_ = 0;
    Scope* previous_;
    bool on_, complete_ = false;
    Clock::time_point start_{};
};

// No clock reads, pointer inspection, synchronization or printing with no scope.
class Transfer {
public:
    Transfer(Part part, Direction direction, size_t bytes)
        : owner_(current), part_(part), direction_(direction), bytes_(bytes) {
        if (owner_) start_ = Clock::now();
    }
    void finish(bool success) {
        if (!owner_) return;
        auto& c = owner_->counters[direction_ * part_count + part_];
        ++c.calls;
        if (success) c.bytes += bytes_; else ++c.failed;
        c.ns += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start_).count();
    }
private:
    Scope* owner_;
    Part part_;
    Direction direction_;
    size_t bytes_;
    Clock::time_point start_{};
};
static_assert(sizeof(Scope) <= 1536 && sizeof(Transfer) <= 128,
              "admission copy observation must remain bounded stack metadata");
} // namespace strata::core::admission_copy
