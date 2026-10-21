#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace strata::core {
enum class DraftKind : uint8_t { none = 0, mtp = 1, dflash2 = 2 };

// Content identities, never paths or allocation addresses. load_generation binds
// borrowed target allocations within a process; persistent restores additionally
// compare the target content digest and rebind to the verified current load.
struct DraftIdentity {
    DraftKind kind = DraftKind::none;
    uint32_t format = 0;
    std::array<uint8_t, 32> weights{}, configuration{}, borrowed_target{};
    uint64_t load_generation = 0;
    // layers, hidden, kv heads, head width, retained rows, block, absolute limit, kind-specific K/V encoding
    std::array<int64_t, 8> geometry{};
    bool operator==(const DraftIdentity&) const = default;
    bool content_matches(const DraftIdentity& other) const {
        return kind == other.kind && format == other.format && weights == other.weights &&
            configuration == other.configuration && borrowed_target == other.borrowed_target &&
            geometry == other.geometry;
    }
    bool bound() const {
        if (kind == DraftKind::none || !format) return false;
        for (auto byte : weights) if (byte) return true;
        return false;
    }
};

struct DraftOwner {
    int seat = 8;
    uint64_t request = 0, incarnation = 0, generation = 0;
    bool operator==(const DraftOwner&) const = default;
};

struct DraftCoverage {
    DraftOwner owner;
    int64_t materialized_begin = -1, materialized_end = -1;
    int64_t certified_begin = -1, certified_end = -1;
    int64_t target_input_frontier = -1;
    int32_t next_token = -1;
    bool present = false;
    bool coherent = true;
};

struct DraftAllocationPlan {
    int device = -1;
    uint64_t shared_device = 0, shared_host = 0;
    uint64_t owner_device = 0, owner_host = 0;
    uint64_t transfer_device = 0, transfer_host = 0;
    uint64_t snapshot_host = 0;
};
} // namespace strata::core
