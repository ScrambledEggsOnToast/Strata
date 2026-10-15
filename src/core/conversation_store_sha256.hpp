// The durable store's content-address, prefix-chain and catalog/manifest integrity primitive.  Vetted EVP SHA-256
// from OpenSSL only: no handwritten rounds, no dlopen, no non-cryptographic substitute (the legacy SessionHasher is
// deliberately NOT used as a content address).  Internal to src/core/conversation_store*.cpp.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace strata::core::conversation_store_detail {

using Digest = std::array<uint8_t, 32>;

// Streaming digest.  digest() leaves the stream usable: it finalizes a copy of the EVP context, so a caller can
// record a digest at an intermediate point (the prefix chain) and keep feeding bytes.
class Sha256 {
public:
    Sha256();
    Sha256(const Sha256& other);
    Sha256& operator=(const Sha256& other);
    ~Sha256();
    void update(const void* data, size_t n);
    Digest digest() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

Digest sha256(const void* data, size_t n);

// Lower-case hex, and its strict inverse (exactly 64 hex digits, no other characters).
std::string sha256_hex(const Digest& digest);
bool sha256_hex_parse(std::string_view text, Digest& out);

// First eight little-endian bytes: ordering and in-memory keys only, never an identity check by itself.
uint64_t digest_u64(const Digest& digest);
bool digest_nonzero(const Digest& digest);

} // namespace strata::core::conversation_store_detail
