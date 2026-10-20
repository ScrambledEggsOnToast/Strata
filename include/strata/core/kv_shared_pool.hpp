// Bounded research adapter for stage/layer-local, graph-stable K8V4 pools.
#pragma once
#include "strata/core/kv_alias.hpp"
#include "strata/core/vmm.hpp"
#include "strata/core/logical_kv.hpp"
#include "strata/core/kv_reclaim.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include <array>
#include <memory>
#include <string>

namespace strata::core {
struct KvAliasReceipt {
    double quote_us = 0, entry_drain_us = 0, acquire_us = 0, table_copy_us = 0;
    double publication_drain_us = 0, bind_us = 0, directory_us = 0, unmap_us = 0, free_us = 0, total_us = 0;
    uint64_t table_bytes = 0, unmaps = 0, releases = 0;
    uint64_t table_calls = 0, incoming_events = 0, publication_events = 0;
    double ready_us = 0, queue_us = 0, available_us = 0;
    uint64_t reclaim_peak_bytes = 0;
};
struct QsaState;
class KvSharedPool {
public:
    explicit KvSharedPool(const plan::KvShareBudget& budget);
    ~KvSharedPool();
    KvSharedPool(const KvSharedPool&) = delete;
    KvSharedPool& operator=(const KvSharedPool&) = delete;
    bool init(std::string& error);
    // Fresh private bytes are unspecified. A producer must initialize every
    // certified/read page; logical reset and private-indexer initialization stay
    // explicit. COW copies a complete old page before its binding is published.
    bool begin(uint64_t owner, const KvOwnerKey& key, int32_t* page_table, std::string& error);
    bool ready(uint64_t owner, const KvOwnerKey& key) { return state_.ready(owner, key); }
    // Non-null streams must order ALL source/target consumers of this cut.
    // Otherwise use the conservative whole-device fence, never infer idleness.
    bool alias(uint64_t target, const KvOwnerKey& key, uint64_t source, const KvOwnerKey& source_key,
               int64_t prefix, const program::ContinuationCertificate& cut, std::string& error,
               KvAliasReceipt* receipt = nullptr, void* source_stream = nullptr, void* target_stream = nullptr);
    // Deferred aliases return after publication; old backing remains charged.
    // Mapping, COW and drained retirement join the owned worker before reuse.
    bool wait_reclaim(std::string& error, KvAliasReceipt* receipt = nullptr);
    KvReclaimStatus reclaim_status() const;
    bool can_reserve(uint64_t limit, uint64_t other_charged, uint64_t incoming) const;
    // A caller must ask before issuing ANY write, including captured appends.
    // Zero/multi-page rewrites are not silently converted into a wider COW plan.
    bool writable(uint64_t owner, const KvOwnerKey& key, uint64_t page, std::string& error);
    bool cancel(uint64_t owner, const KvOwnerKey& key) { return state_.cancel(owner, key); }
    bool terminal(uint64_t owner, const KvOwnerKey& key, std::string& error);
    bool mount(QsaState& state, uint64_t owner, const KvOwnerKey& key, std::string& error);
    bool reset(uint64_t owner, const KvOwnerKey& key, std::string& error);
    // The pool must outlive this read lease. Copies keep the owner immutable;
    // release all copies before its existing drained reset/retirement.
    std::shared_ptr<KvSharedPool> borrow(uint64_t owner, const KvOwnerKey& key, std::string& error);
    bool usable(uint64_t owner, const KvOwnerKey& key) const { return !poisoned_ && state_.owns(owner, key); }
    strata::kernels::QsaAttnPools pools(uint64_t owner) const;
    uint64_t mapped_bytes() const;
    uint64_t released_chunks() const { return reclaim_status().released_chunks; }
    uint64_t host_bytes() const;
    const KvAliasState& ownership() const { return state_; }
    const std::vector<int32_t>& table(uint64_t owner) const { return tables_[(size_t) owner]; }
    int device() const { return device_; }
private:
    bool drain(std::string& error) const;
    bool fence(void* source, void* target, bool publication, std::string& error, KvAliasReceipt* receipt);
    bool publish(uint64_t owner, std::string& error);
    bool retain_extent(uint64_t extent, std::string& error);
    void release_extent(uint64_t extent);
    uint64_t prepare_reclaim(std::string& error);
    bool reclaim(std::string& error, KvAliasReceipt* receipt = nullptr, uint64_t ticket = 0);
    static bool fail(std::string& error, const char* operation);
    const plan::KvShareBudget budget_;
    KvAliasState state_;
    VmmRange range_;
    KvReadPins readers_;
    std::array<uint64_t, 3> offsets_{};
    inline static constexpr std::array<uint64_t, 3> per_page_{{2048, 64, 1152}};
    std::array<void*, 3> fences_{};
    std::vector<uint32_t> chunk_users_;
    std::vector<std::vector<int32_t>> tables_;
    std::vector<int32_t*> device_tables_;
    uint64_t released_chunks_ = 0;
    int device_ = -1;
    std::atomic<bool> poisoned_{false};
    struct ReclaimWorker;
    std::unique_ptr<ReclaimWorker> worker_;
};
bool kv_shared_valid(const QsaState& state);
KvLogicalPages kv_logical_pages(const QsaState& state);
bool kv_shared_reset(const QsaState& state, std::string& error);
} // namespace strata::core
