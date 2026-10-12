// Runtime-only continuation economics. Eligibility and state ownership stay with the caller.
#pragma once
#include <array>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

namespace strata::program {
enum class PrefixTransfer { checkpoint, resident_live, resident_checkpoint, parked, park, count };
struct PrefixWork {
    uint64_t bytes = 0;
    uint64_t route = 0; // resolved devices and authoritative K/V modes; never a model-state identity
    void add(uint64_t n) {
        bytes = n > UINT64_MAX - bytes ? UINT64_MAX : bytes + n;
    }
    void path(int device, int mode) {
        route = (route ^ (uint64_t(uint32_t(device)) << 32 | uint32_t(mode))) * 1099511628211ull;
    }
};

class PrefixCosts {
public:
    // Eight recent observations per real copy path, no probes, allocations or persisted calibration.
    void transfer(PrefixTransfer kind, PrefixWork work, double ms) {
        if (!valid(ms) || !work.bytes || work.bytes == UINT64_MAX) return;
        auto& history = transfers_[size_t(kind)];
        history.samples[history.next++ % history.samples.size()] = {work, ms};
    }
    std::optional<double> transfer_ms(PrefixTransfer kind, PrefixWork work) const {
        if (!work.bytes) return 0.0;
        if (work.bytes == UINT64_MAX) return {};
        double rate = 0;
        size_t count = 0;
        for (const auto& sample : transfers_[size_t(kind)].samples) {
            if (!sample.work.bytes || sample.work.route != work.route) continue;
            const double ratio = double(work.bytes) / double(sample.work.bytes);
            // Bounded interpolation/extrapolation only. A new layout or a >4x size change must warm up again.
            if (ratio < 0.25 || ratio > 4.0) continue;
            rate += sample.ms / double(sample.work.bytes);
            ++count;
        }
        if (!count) return {};
        return rate / double(count) * double(work.bytes);
    }
    void prefill(int64_t begin, int64_t end, bool windows, double ms, uint64_t route = 0) {
        if (begin < 0 || end <= begin || !valid(ms)) return;
        auto& rate = prefill_[windows ? 1 : 0][bucket(end)];
        const int64_t tokens = end - begin;
        const double ratio = rate.tokens ? double(tokens) / double(rate.tokens) : 0;
        if (rate.route != route || ratio < 0.25 || ratio > 4.0) rate = {0, 0, route, 0};
        rate.tokens = tokens;
        const double observed = ms / double(tokens);
        rate.samples = std::min(rate.samples + 1, 8u);
        rate.ms_per_token += (observed - rate.ms_per_token) / rate.samples;
    }
    std::optional<double> prefill_ms(int64_t begin, int64_t end, bool windows, uint64_t route = 0) const {
        if (begin < 0 || end < begin) return {};
        if (end == begin) return 0.0;
        const auto& rates = prefill_[windows ? 1 : 0];
        const size_t at = bucket(end);
        // Same context scale first, then at most one neighboring power-of-two context bucket.
        for (const size_t i : {at, at ? at - 1 : at, std::min(at + 1, rates.size() - 1)}) {
            const double ratio = rates[i].tokens ? double(end - begin) / double(rates[i].tokens) : 0;
            if (rates[i].samples && rates[i].route == route && ratio >= 0.25 && ratio <= 4.0)
                return rates[i].ms_per_token * double(end - begin);
        }
        return {};
    }
private:
    static bool valid(double ms) { return std::isfinite(ms) && ms > 0; }
    static size_t bucket(int64_t context) {
        size_t i = 0;
        for (uint64_t n = uint64_t(std::max<int64_t>(context, 1)); n > 1 && i < 31; n >>= 1) ++i;
        return i;
    }
    struct TransferSample { PrefixWork work; double ms = 0; };
    struct History { std::array<TransferSample, 8> samples{}; size_t next = 0; };
    struct Rate { double ms_per_token = 0; unsigned samples = 0; uint64_t route = 0; int64_t tokens = 0; };
    std::array<History, size_t(PrefixTransfer::count)> transfers_{};
    std::array<std::array<Rate, 32>, 2> prefill_{};
};
static_assert(sizeof(PrefixCosts) <= 4096, "prefix observations must stay inside their bounded RAM charge");

// Enumerate in the old tie order: active, resident slots in seat order, parked newest first.
// Cold is also a candidate. Unknown alternatives never win; an unpriced longest
// prefix keeps the old choice. Ties and insufficient savings retain that same rollback.
class PrefixChoice {
public:
    explicit PrefixChoice(bool measured, double minimum_savings_ms = 0)
        : measured_(measured), minimum_savings_ms_(minimum_savings_ms) {}
    void consider(size_t id, int64_t tokens, std::optional<double> total_ms) {
        if (tokens < 0) return;
        const bool known = total_ms && std::isfinite(*total_ms) && *total_ms >= 0;
        unknown_ |= !known;
        if (!have_ || tokens > longest_tokens_) {
            longest_ = id;
            longest_tokens_ = tokens;
            longest_ms_ = known ? *total_ms : std::numeric_limits<double>::infinity();
        }
        if (known && (!cheapest_ms_ || *total_ms < *cheapest_ms_)) {
            cheapest_ = id;
            cheapest_ms_ = *total_ms;
        }
        have_ = true;
    }
    size_t selected() const {
        return using_cost() ? cheapest_ : longest_;
    }
    bool using_cost() const {
        return measured_ && std::isfinite(longest_ms_) && cheapest_ms_ &&
            *cheapest_ms_ + minimum_savings_ms_ < longest_ms_;
    }
    bool unknown() const { return unknown_; }
    bool longest_known() const { return std::isfinite(longest_ms_); }
    size_t longest() const { return longest_; }
    size_t cheapest() const { return cheapest_; }
private:
    bool measured_, have_ = false, unknown_ = false;
    double minimum_savings_ms_, longest_ms_ = 0;
    int64_t longest_tokens_ = 0;
    size_t longest_ = 0, cheapest_ = 0;
    std::optional<double> cheapest_ms_;
};
} // namespace strata::program
