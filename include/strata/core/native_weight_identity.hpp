#pragma once

#include <array>
#include <cstdint>
#include <string>

namespace strata::core {
// Populated only by an opt-in native load. The digest covers the exact unchanged
// GGUF tensor bytes copied into the allocation, not a path, header or declaration.
struct NativeSourceIdentity {
    std::string path;
    std::array<uint64_t, 7> stamp{};
    bool operator==(const NativeSourceIdentity&) const = default;
};
struct NativeTensorIdentity {
    std::string sha256;
    NativeSourceIdentity source;
    uint64_t bytes = 0, generation = 0;
    int64_t input = 0, output = 0;
    int type = -1, device = -1;
    bool operator==(const NativeTensorIdentity&) const = default;
};

// Source replacement or mutation during an opt-in load/verification refuses.
NativeSourceIdentity native_source_identity(const std::string& path);
uint64_t next_native_weight_generation();
} // namespace strata::core
