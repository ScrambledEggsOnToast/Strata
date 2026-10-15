#pragma once

#include "strata/core/conversation_cache.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace strata::program {

// Private frontend/native control, never a client-facing endpoint. Only exact
// text prompts are offered; image preparation retains its ordinary owner gates.
struct ConversationOffer {
    uint64_t request_id = 0;
    uint64_t generation = 0;
    uint64_t deadline_ns = 0; // CLOCK_MONOTONIC, original request deadline
    core::ConversationScope scope;
    bool cvec = true;
    std::vector<int32_t> tokens;
};

struct ConversationOfferStats {
    uint64_t accepted = 0;
    uint64_t cancelled = 0;
    uint64_t rejected = 0;
    uint64_t accepted_token_bytes = 0;
};

// Returns an inherited descriptor >=3, or -1. Consumes the environment variable
// so a later child cannot accidentally inherit this channel's ownership.
int inherited_conversation_offer_fd();

class ConversationOfferChannel {
public:
    // Callbacks run on the CPU reader thread and must never touch GPU state.
    // offer takes ownership only when it returns true. cancel retires exactly
    // this request/generation. Both must provide their own storage synchronization.
    using Offer = std::function<bool(ConversationOffer&&)>;
    using Cancel = std::function<void(uint64_t, uint64_t)>;

    // Owns fd, including on start failure. Frontier is 1 or 2. The byte limit
    // covers outstanding token descriptors, not permission for image allocation.
    ConversationOfferChannel(int fd, size_t max_tokens, size_t frontier,
                             size_t metadata_bytes, Offer offer, Cancel cancel);
    ~ConversationOfferChannel();
    ConversationOfferChannel(const ConversationOfferChannel&) = delete;
    ConversationOfferChannel& operator=(const ConversationOfferChannel&) = delete;

    // Sends the fixed SCC1 HELLO then starts bounded offer reception. A disabled
    // store sends no key and accepts no offers. No key/prompt bytes are logged.
    bool start(const std::array<uint8_t, 32>& issuer_key, uint64_t epoch,
               bool enabled, std::string& why);
    void stop() noexcept;
    ConversationOfferStats stats() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace strata::program
