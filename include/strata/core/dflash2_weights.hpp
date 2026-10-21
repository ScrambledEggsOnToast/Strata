#pragma once

#include "strata/core/dflash2_bridge.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace strata::core::dflash2 {
struct RuntimeTensor {
    std::span<const uint16_t> values;
    std::vector<int64_t> shape;
    size_t offset_bytes = 0;
};

// Host-only, strict reader for tools/dflash2_cpu/runtime.py's converted format.
// The manifest digest and target identity MUST come from independent release
// evidence and the actual borrowed target, respectively, not this directory.
// Files must remain immutable for this object's lifetime. Views remain valid
// until destruction; a device upload must complete before that destruction.
class RuntimeWeights {
public:
    RuntimeWeights(const std::string& directory, const std::string& independent_manifest_sha256,
                   const TargetIdentity& borrowed_target);
    ~RuntimeWeights();
    RuntimeWeights(const RuntimeWeights&) = delete;
    RuntimeWeights& operator=(const RuntimeWeights&) = delete;

    const RuntimeTensor& tensor(std::string_view name) const;
    std::span<const uint16_t> dense() const;
    const TargetIdentity& target_identity() const;
    const std::string& manifest_sha256() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strata::core::dflash2
