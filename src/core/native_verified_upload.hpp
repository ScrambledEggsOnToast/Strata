#pragma once

#include "conversation_store_sha256.hpp"

#include <algorithm>
#include <cstring>
#include <memory>
#include <span>

namespace strata::core {
inline constexpr size_t native_identity_staging_bytes = 4u << 20;
// `upload` must consume this owned immutable chunk synchronously. Hash exactly
// those bytes; no digest is published for a failed/partial upload. Only opt-in
// verified loads use this bounded staging allocation.
template<class Upload>
bool native_verified_upload(const void* source, size_t bytes, std::string& digest, Upload&& upload) {
    digest.clear();
    if (!source || bytes == 0) return false;
    const size_t capacity = std::min(bytes, native_identity_staging_bytes);
    auto staging = std::make_unique_for_overwrite<std::byte[]>(capacity);
    conversation_store_detail::Sha256 hash;
    for (size_t offset = 0; offset < bytes;) {
        const size_t count = std::min(bytes - offset, capacity);
        std::memcpy(staging.get(), static_cast<const std::byte*>(source) + offset, count);
        hash.update(staging.get(), count);
        if (!upload(offset, std::span<const std::byte>{staging.get(), count})) return false;
        offset += count;
    }
    digest = conversation_store_detail::sha256_hex(hash.digest());
    return true;
}
} // namespace strata::core
