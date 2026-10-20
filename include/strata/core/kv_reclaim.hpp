// Physical backing stays charged until the driver has actually released it.
#pragma once
#include "strata/plan/kv_share.hpp"
#include <atomic>
#include <bitset>
#include <memory>
#include <string>

namespace strata::core {
struct KvReclaimStatus {
    uint64_t charged_bytes = 0, pending_bytes = 0, released_chunks = 0, peak_pending_bytes = 0;
};
// One controller prepares a batch, then hands it to one owned worker. The
// controller must join that batch before mapping or preparing another one.
// Concurrent observers only read atomic counters, never the bitmap directory.
class KvReclaimLedger {
public:
    static constexpr uint64_t chunk_limit = 1024;
    bool mapped(uint64_t chunk) {
        if (chunk >= chunk_limit || mapped_[chunk] || pending_.load(std::memory_order_acquire)) return false;
        change_begin(); mapped_.set(chunk); charged_.fetch_add(plan::kKvShareChunk, std::memory_order_relaxed); change_end();
        return true;
    }
    uint64_t begin() {
        if (pending_.load(std::memory_order_acquire) || epoch_ == UINT64_MAX) return 0;
        return ++epoch_;
    }
    bool queue(uint64_t ticket, uint64_t chunk) {
        if (!ticket || ticket != epoch_ || chunk >= chunk_limit || !mapped_[chunk] || queued_[chunk]) return false;
        change_begin(); queued_.set(chunk);
        const auto bytes = pending_.fetch_add(plan::kKvShareChunk, std::memory_order_relaxed) + plan::kKvShareChunk;
        if (bytes > peak_.load(std::memory_order_relaxed)) peak_.store(bytes, std::memory_order_relaxed);
        change_end(); return true;
    }
    // Call ONLY after BOTH unmap and physical release succeeded. A stale or
    // duplicate completion never consumes another chunk or replacement batch.
    bool complete(uint64_t ticket, uint64_t chunk) {
        if (!ticket || ticket != epoch_ || chunk >= chunk_limit || !queued_[chunk] || !mapped_[chunk]) return false;
        change_begin(); queued_.reset(chunk); mapped_.reset(chunk);
        charged_.fetch_sub(plan::kKvShareChunk, std::memory_order_relaxed);
        released_.fetch_add(1, std::memory_order_relaxed);
        pending_.fetch_sub(plan::kKvShareChunk, std::memory_order_relaxed);
        change_end(); return true;
    }
    KvReclaimStatus snapshot() const {
        for (;;) {
            const auto before = sequence_.load(std::memory_order_acquire);
            if (before & 1) continue;
            const KvReclaimStatus result{charged_.load(std::memory_order_relaxed), pending_.load(std::memory_order_relaxed),
                released_.load(std::memory_order_relaxed), peak_.load(std::memory_order_relaxed)};
            std::atomic_thread_fence(std::memory_order_acquire);
            if (sequence_.load(std::memory_order_relaxed) == before) return result;
        }
    }
    bool can_reserve(uint64_t limit, uint64_t other_charged, uint64_t incoming) const {
        const auto charged = charged_.load(std::memory_order_acquire);
        return charged <= limit && other_charged <= limit - charged && incoming <= limit - charged - other_charged;
    }
private:
    void change_begin() { sequence_.fetch_add(1, std::memory_order_acq_rel); }
    void change_end() { sequence_.fetch_add(1, std::memory_order_release); }
    std::bitset<chunk_limit> mapped_, queued_;
    uint64_t epoch_ = 0;
    std::atomic<uint64_t> sequence_{0}, charged_{0}, pending_{0}, released_{0}, peak_{0};
};
// start() cannot accept reclamation jobs before it succeeds. Consume a failed
// worker here so a poisoned owner never mistakes failed startup for failed work.
template<class Worker>
bool kv_reclaim_start_worker(std::unique_ptr<Worker>& worker, std::string& error) {
    if (worker->start(error)) return true;
    worker.reset();
    return false;
}
} // namespace strata::core
