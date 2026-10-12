// Scheduling at drained, memory-admitted prefill/decode boundaries. No model arithmetic lives here.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace strata::program {
class TimeBudget {
public:
    TimeBudget(double foreground_ms, double background_share)
        : foreground_ms_(foreground_ms), background_share_(background_share) {}
    static bool valid(double foreground_ms, double background_share) {
        return std::isfinite(foreground_ms) && foreground_ms >= 0 && foreground_ms <= 60000 &&
               std::isfinite(background_share) && background_share >= 0.01 && background_share <= 0.99;
    }
    bool enabled() const { return foreground_ms_ > 0; }
    // Select a scheduling quantum, never a new arithmetic chunk. The memory-admitted
    // chunk stays fixed; even a cold/over-budget estimate runs one indivisible chunk.
    int64_t prefill_tokens(int64_t remaining, int64_t memory_chunk, double decode_age_ms) const {
        if (remaining <= 0 || memory_chunk <= 0) return 0;
        if (remaining <= memory_chunk || chunk_ms_ <= 0 || chunk_tokens_ != memory_chunk)
            return std::min(remaining, memory_chunk);
        const int64_t full = remaining / memory_chunk;
        const double room = std::max(0.0, foreground_ms_ - std::max(0.0, decode_age_ms) - decode_ms_);
        const double count = std::clamp(std::floor(room / chunk_ms_), 1.0, (double) full);
        // Clamp before conversion and multiplication: full * memory_chunk <= remaining.
        const int64_t chunks = count >= (double) full ? full : (int64_t) count;
        return chunks * memory_chunk;
    }
    void prefilled(double wall_ms, int64_t chunks, int64_t memory_chunk) {
        if (!std::isfinite(wall_ms) || wall_ms <= 0 || chunks <= 0 || memory_chunk <= 0) return;
        // A short request's smaller admitted chunk cannot train a larger chunk's cost downward.
        chunk_ms_ = std::max(wall_ms / (double) chunks, chunk_tokens_ == memory_chunk ? chunk_ms_ * 0.9 : 0.0);
        chunk_tokens_ = memory_chunk;
        // Bounded credit: idle/background-only work cannot buy an unbounded decode burst.
        decode_credit_ms_ = std::min(foreground_ms_, decode_credit_ms_ +
                                    wall_ms * (1.0 - background_share_) / background_share_);
    }
    bool decode_due(double age_ms, int64_t memory_chunk) const {
        return chunk_tokens_ != memory_chunk || chunk_ms_ <= 0 ||
               age_ms + chunk_ms_ + decode_ms_ >= foreground_ms_;
    }
    bool decode_allowed(double already_ms) const {
        // One complete round may overrun; it cannot be interrupted or skipped midway.
        return already_ms < decode_credit_ms_;
    }
    void decoded(double wall_ms, bool charge = true) {
        if (!std::isfinite(wall_ms) || wall_ms <= 0) return;
        decode_ms_ = std::max(wall_ms, decode_ms_ * 0.9);
        // Carry the indivisible-round overrun forward rather than forgiving it
        // at each boundary. Decode-only periods update cost, not background debt.
        if (charge) decode_credit_ms_ -= wall_ms;
    }
    double chunk_ms() const { return chunk_ms_; }
    double decode_ms() const { return decode_ms_; }
private:
    double foreground_ms_, background_share_;
    double chunk_ms_ = 0, decode_ms_ = 0, decode_credit_ms_ = 0;
    int64_t chunk_tokens_ = 0;
};
} // namespace strata::program
