// Physical backing stays charged until the driver has actually released it.
#pragma once
#include "strata/core/kv_alias.hpp"
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
    KvReclaimLedger() = default; // Retains the #268 diagnostic worker's legacy path.
    explicit KvReclaimLedger(plan::KvTopologyKey key) : topology_(key) {
        if (!key.valid()) throw std::invalid_argument("KV reclaim ledger: invalid topology identity");
    }
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
    // The worker and the CPU policy consume the same physical ledger. No shadow
    // freed-byte estimate can turn a pending release into admission headroom.
    plan::KvReclaimDispatch dispatch(const plan::KvDeviceReclaimPolicy& policy, const plan::KvTopologyKey& key,
            const KvPipelineLifetime& pipeline, const KvPipelineKey& submission, uint64_t ready_chunks,
            const plan::KvReclaimWindow& window) const {
        if (!topology_.valid() || !(topology_ == key)) return {};
        const auto physical = snapshot();
        return plan::kv_reclaim_dispatch(policy, key, physical.charged_bytes, physical.pending_bytes,
            ready_chunks, pipeline.drained_for(submission, key.stage), window);
    }
private:
    void change_begin() { sequence_.fetch_add(1, std::memory_order_acq_rel); }
    void change_end() { sequence_.fetch_add(1, std::memory_order_release); }
    std::bitset<chunk_limit> mapped_, queued_;
    uint64_t epoch_ = 0;
    const plan::KvTopologyKey topology_{};
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
struct KvReclaimProgress {
    plan::KvReclaimAction action = plan::KvReclaimAction::refuse;
    bool success = false;
    uint64_t released_chunks = 0;
};
// CPU policy adapter for one exclusively handed-off contiguous run. Real backends
// supply nonthrowing unmap/physical-release callbacks. Separate handles still need
// separate releases; batching is scheduling, not a claimed single driver call.
class KvReclaimRun {
public:
    KvReclaimRun(plan::KvTopologyKey key, plan::KvDeviceReclaimPolicy policy, KvPipelineKey pipeline,
                 uint64_t first_chunk, uint64_t chunks)
        : key_(key), policy_(policy), pipeline_(pipeline), first_(first_chunk), chunks_(chunks) {
        if (!policy.valid_for(key) || !pipeline.valid() || !chunks ||
            chunks > UINT64_MAX / plan::kKvShareChunk || first_chunk > UINT64_MAX - chunks)
            throw std::invalid_argument("KV reclaim: invalid device policy/run identity");
    }
    ~KvReclaimRun() { if (held_) std::terminate(); }
    KvReclaimRun(const KvReclaimRun&) = delete;
    KvReclaimRun& operator=(const KvReclaimRun&) = delete;
    uint64_t charged_bytes() const { return (chunks_ - released_) * plan::kKvShareChunk; }
    uint64_t unmapped_bytes() const { return unmapped_ * plan::kKvShareChunk; }
    uint64_t released_bytes() const { return released_ * plan::kKvShareChunk; }
    // Only `enqueue` requests a handoff; `wait_worker` must not queue twice.
    // The held handle survives physical-free failure for an exact retry.
    template<class Unmap, class Release>
    KvReclaimProgress reclaim(const KvPipelineLifetime& lifetime, const plan::KvDeviceUuid& executor,
                              bool worker, bool flush, Unmap&& unmap, Release&& release) {
        using plan::KvReclaimAction;
        if (executor != key_.device_uuid) return {};
        const auto selected = plan::kv_reclaim_plan(policy_, key_, chunks_ - released_,
            lifetime.drained_for(pipeline_, key_.stage), flush || batch_remaining_ != 0);
        switch (selected.action) {
        case KvReclaimAction::wait_pipeline:
        case KvReclaimAction::hold:
        case KvReclaimAction::complete: return {selected.action, true, 0};
        case KvReclaimAction::enqueue:
            if (!worker) {
                if (queued_) return {KvReclaimAction::wait_worker, true, 0};
                batch_remaining_ = selected.chunks;
                queued_ = true; return {KvReclaimAction::enqueue, true, 0};
            }
            if (!queued_) return {};
            break;
        case KvReclaimAction::release_inline:
            if (worker) return {};
            break;
        default: return {};
        }
        if (!batch_remaining_) batch_remaining_ = selected.chunks;
        const auto action = worker ? KvReclaimAction::release_worker : KvReclaimAction::release_inline;
        const uint64_t before = released_;
        const uint64_t batch = batch_remaining_;
        for (uint64_t count = 0; count < batch; ++count) {
            if (!held_) {
                held_ = unmap(key_, first_ + unmapped_);
                if (!held_) return {action, false, released_ - before};
                ++unmapped_;
            }
            if (!release(key_, held_)) return {action, false, released_ - before};
            held_ = 0; ++released_; --batch_remaining_;
        }
        queued_ = false;
        return {action, true, released_ - before};
    }
private:
    const plan::KvTopologyKey key_;
    const plan::KvDeviceReclaimPolicy policy_;
    const KvPipelineKey pipeline_;
    const uint64_t first_, chunks_;
    uint64_t unmapped_ = 0, released_ = 0, held_ = 0, batch_remaining_ = 0;
    bool queued_ = false;
};
} // namespace strata::core
