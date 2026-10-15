#include "conversation_store_sha256.hpp"

#include <openssl/evp.h>

#include <cstring>
#include <stdexcept>

namespace strata::core::conversation_store_detail {
namespace {

EVP_MD_CTX* make_context() {
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (ctx == nullptr) throw std::bad_alloc();
    if (EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) != 1) {
        EVP_MD_CTX_free(ctx);
        throw std::runtime_error("openssl: EVP_sha256 is unavailable");
    }
    return ctx;
}

} // namespace

struct Sha256::Impl {
    EVP_MD_CTX* ctx = nullptr;
};

Sha256::Sha256() : impl_(std::make_unique<Impl>()) { impl_->ctx = make_context(); }

Sha256::Sha256(const Sha256& other) : impl_(std::make_unique<Impl>()) {
    impl_->ctx = EVP_MD_CTX_new();
    if (impl_->ctx == nullptr) throw std::bad_alloc();
    if (EVP_MD_CTX_copy_ex(impl_->ctx, other.impl_->ctx) != 1) {
        EVP_MD_CTX_free(impl_->ctx);
        impl_->ctx = nullptr;
        throw std::runtime_error("openssl: cannot copy a SHA-256 context");
    }
}

Sha256& Sha256::operator=(const Sha256& other) {
    if (this == &other) return *this;
    if (impl_->ctx == nullptr) {
        impl_->ctx = EVP_MD_CTX_new();
        if (impl_->ctx == nullptr) throw std::bad_alloc();
    }
    if (EVP_MD_CTX_copy_ex(impl_->ctx, other.impl_->ctx) != 1)
        throw std::runtime_error("openssl: cannot copy a SHA-256 context");
    return *this;
}

Sha256::~Sha256() {
    if (impl_ && impl_->ctx != nullptr) EVP_MD_CTX_free(impl_->ctx);
}

void Sha256::update(const void* data, size_t n) {
    if (n == 0) return;
    if (data == nullptr) throw std::invalid_argument("openssl: hashing a null buffer");
    // EVP takes an int length; feed longer buffers in bounded pieces so a huge stream is still exact.
    const uint8_t* p = static_cast<const uint8_t*>(data);
    constexpr size_t kStep = 1u << 28;
    while (n > 0) {
        const size_t part = n < kStep ? n : kStep;
        if (EVP_DigestUpdate(impl_->ctx, p, part) != 1)
            throw std::runtime_error("openssl: EVP_DigestUpdate failed");
        p += part;
        n -= part;
    }
}

Digest Sha256::digest() const {
    EVP_MD_CTX* copy = EVP_MD_CTX_new();
    if (copy == nullptr) throw std::bad_alloc();
    Digest out{};
    unsigned int written = 0;
    if (EVP_MD_CTX_copy_ex(copy, impl_->ctx) != 1 ||
        EVP_DigestFinal_ex(copy, out.data(), &written) != 1 || written != out.size()) {
        EVP_MD_CTX_free(copy);
        throw std::runtime_error("openssl: EVP_DigestFinal_ex failed");
    }
    EVP_MD_CTX_free(copy);
    return out;
}

Digest sha256(const void* data, size_t n) {
    Sha256 hash;
    hash.update(data, n);
    return hash.digest();
}

std::string sha256_hex(const Digest& digest) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::string out;
    out.resize(digest.size() * 2);
    for (size_t i = 0; i < digest.size(); ++i) {
        out[2 * i] = kHex[digest[i] >> 4];
        out[2 * i + 1] = kHex[digest[i] & 0x0f];
    }
    return out;
}

bool sha256_hex_parse(std::string_view text, Digest& out) {
    if (text.size() != out.size() * 2) return false;
    auto nibble = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    };
    Digest parsed{};
    for (size_t i = 0; i < parsed.size(); ++i) {
        const int hi = nibble(text[2 * i]);
        const int lo = nibble(text[2 * i + 1]);
        if (hi < 0 || lo < 0) return false;
        parsed[i] = static_cast<uint8_t>((hi << 4) | lo);
    }
    out = parsed;
    return true;
}

uint64_t digest_u64(const Digest& digest) {
    uint64_t value = 0;
    std::memcpy(&value, digest.data(), sizeof(value));
    return value;
}

bool digest_nonzero(const Digest& digest) {
    for (uint8_t byte : digest)
        if (byte != 0) return true;
    return false;
}

} // namespace strata::core::conversation_store_detail
