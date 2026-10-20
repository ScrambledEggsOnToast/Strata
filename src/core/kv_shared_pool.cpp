#include "strata/core/kv_shared_pool.hpp"
#include "strata/core/layer.hpp"
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <stdexcept>

namespace strata::core {
static_assert(sizeof(KvSharedPool) <= plan::kKvSharePoolBytes);
namespace {
uint64_t checked_pages(const plan::KvShareBudget& b) {
    auto pool = b;
    pool.mode = plan::KvShareMode::pool;
    const auto quote = plan::kv_share_quote(kernels::qsa_real_shapes(), kernels::kKvHybrid, pool, 0);
    if (!quote.known || !quote.admitted || b.mode == plan::KvShareMode::off)
        throw std::invalid_argument("KV shared pool: invalid or unpriced capacity");
    return b.cells / 4;
}
}
KvSharedPool::KvSharedPool(const plan::KvShareBudget& budget)
    : budget_(budget), state_(budget.owners, checked_pages(budget)),
      tables_((size_t) budget.owners), device_tables_((size_t) budget.owners, nullptr) {}
KvSharedPool::~KvSharedPool() {
    if (readers_.any()) {
        std::fprintf(stderr, "KV shared pool: destruction while a streamed source still borrows its owner\n");
        std::terminate();
    }
    if (device_ < 0) return;
    int previous = -1;
    if (cudaGetDevice(&previous) != cudaSuccess || cudaSetDevice(device_) != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "KV shared pool: teardown drain failed; no storage reuse is safe\n");
        std::terminate();
    }
    range_.release();
    for (void* fence : fences_) if (fence && cudaEventDestroy(static_cast<cudaEvent_t>(fence)) != cudaSuccess)
        std::terminate();
    if (cudaSetDevice(previous) != cudaSuccess) std::terminate();
}
bool KvSharedPool::fail(std::string& error, const char* operation) {
    error = std::string("KV shared pool: ") + operation;
    return false;
}
bool KvSharedPool::drain(std::string& error) const {
    int current = -1;
    if (poisoned_ || device_ < 0 || cudaGetDevice(&current) != cudaSuccess || current != device_)
        return fail(error, "poisoned, uninitialized or foreign-device owner");
    if (cudaDeviceSynchronize() != cudaSuccess) return fail(error, "consumer drain failed");
    return true;
}
bool KvSharedPool::fence(void* source, void* target, bool publication, std::string& error, KvAliasReceipt* receipt) {
    if (!source || !target) return drain(error);
    int current = -1;
    if (poisoned_ || cudaGetDevice(&current) != cudaSuccess || current != device_)
        return fail(error, "foreign-device event fence");
    auto record = [&](size_t index, void* stream) {
        return cudaEventRecord(static_cast<cudaEvent_t>(fences_[index]), static_cast<cudaStream_t>(stream)) == cudaSuccess;
    };
    auto join = [&](size_t index) { return cudaEventSynchronize(static_cast<cudaEvent_t>(fences_[index])) == cudaSuccess; };
    const bool complete = publication ? record(2, target) && join(2) :
        record(0, source) && (source == target || record(1, target)) && join(0) && (source == target || join(1));
    if (!complete) { poisoned_ = true; return fail(error, "owner event fence failed"); }
    if (receipt) {
        if (publication) ++receipt->publication_events;
        else receipt->incoming_events += source == target ? 1 : 2;
    }
    return true;
}
bool KvSharedPool::init(std::string& error) {
    if (device_ >= 0 || !vmm_available() || vmm_granularity() != budget_.granularity)
        return fail(error, "VMM unavailable, already initialized or unqualified granularity");
    if (cudaGetDevice(&device_) != cudaSuccess) { device_ = -1; return fail(error, "device identity unavailable"); }
    uint64_t bytes = 0;
    for (size_t part = 0; part < per_page_.size(); ++part) {
        offsets_[part] = bytes;
        bytes += (state_.extent_count() * per_page_[part] + budget_.granularity - 1) /
                 budget_.granularity * budget_.granularity;
    }
    if (!range_.reserve(bytes)) { poisoned_ = true; return fail(error, "address reservation failed"); }
    chunk_users_.assign((size_t) range_.chunks(), 0);
    for (auto& table : tables_) table.resize((size_t) state_.pages(), -1);
    for (void*& slot : fences_) {
        cudaEvent_t event = nullptr;
        if (cudaEventCreateWithFlags(&event, cudaEventDisableTiming) != cudaSuccess) {
            poisoned_ = true; return fail(error, "owner fence allocation failed");
        }
        slot = event;
    }
    return true;
}
bool KvSharedPool::retain_extent(uint64_t extent, std::string& error) {
    for (size_t part = 0; part < per_page_.size(); ++part) {
        const uint64_t first_byte = offsets_[part] + extent * per_page_[part];
        const uint64_t first = first_byte / plan::kKvShareChunk;
        const uint64_t end = (first_byte + per_page_[part] + plan::kKvShareChunk - 1) / plan::kKvShareChunk;
        for (uint64_t chunk = first; chunk < end; ++chunk) {
            if (!range_.mapped((int64_t) chunk)) {
                // New private capacity is initialized by its producer/reset;
                // COW overwrites the entire page before publication. Do not
                // clear unrelated, currently unbound bytes in a 2 MiB chunk.
                if (!range_.map_range((int64_t) chunk, (int64_t) chunk + 1, [] { return VmmChunk{0}; })) {
                    poisoned_ = true; return fail(error, "map private capacity failed");
                }
            }
            ++chunk_users_[(size_t) chunk];
        }
    }
    return true;
}
void KvSharedPool::release_extent(uint64_t extent) {
    if (state_.extent(extent).references) return;
    for (size_t part = 0; part < per_page_.size(); ++part) {
        const uint64_t first_byte = offsets_[part] + extent * per_page_[part];
        const uint64_t first = first_byte / plan::kKvShareChunk;
        const uint64_t end = (first_byte + per_page_[part] + plan::kKvShareChunk - 1) / plan::kKvShareChunk;
        for (uint64_t chunk = first; chunk < end; ++chunk) --chunk_users_[(size_t) chunk];
    }
}
bool KvSharedPool::reclaim(std::string& error, KvAliasReceipt* receipt) {
    // The caller has drained both page-table publication and old consumers. Only
    // a successful unmap plus physical release contributes a measured chunk.
    for (int64_t chunk = 0; chunk < range_.chunks(); ++chunk) {
        if (chunk_users_[(size_t) chunk] || !range_.mapped(chunk)) continue;
        const auto unmap_start = receipt ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        const VmmChunk handle = range_.unmap(chunk);
        const auto free_start = receipt ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        if (receipt) {
            receipt->unmap_us += std::chrono::duration<double, std::micro>(free_start - unmap_start).count();
            ++receipt->unmaps;
        }
        if (!handle) { poisoned_ = true; return fail(error, "drained chunk unmap failed"); }
        if (!vmm_chunk_free(handle)) { poisoned_ = true; return fail(error, "physical chunk release failed"); }
        if (receipt) {
            receipt->free_us += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - free_start).count();
            ++receipt->releases;
        }
        ++released_chunks_;
    }
    return true;
}
bool KvSharedPool::publish(uint64_t owner, std::string& error) {
    const auto& table = tables_[(size_t) owner];
    if (cudaMemcpy(device_tables_[(size_t) owner], table.data(), table.size() * sizeof(int32_t),
                   cudaMemcpyHostToDevice) != cudaSuccess || !drain(error)) {
        poisoned_ = true; return fail(error, "private page-table publication failed");
    }
    return true;
}
bool KvSharedPool::begin(uint64_t owner, const KvOwnerKey& key, int32_t* page_table, std::string& error) {
    if (!page_table || readers_.pinned(owner) || readers_.generation(owner) == UINT64_MAX ||
        !drain(error) || !state_.begin(owner, key)) return fail(error, "owner begin refused");
    if (!readers_.advance(owner)) { poisoned_ = true; return fail(error, "owner lifetime advance failed"); }
    device_tables_[(size_t) owner] = page_table;
    for (uint64_t page = 0; page < state_.pages(); ++page) {
        const auto b = state_.binding(owner, page);
        if (!retain_extent(b.extent, error)) return false;
        tables_[(size_t) owner][(size_t) page] = (int32_t) b.extent;
    }
    return publish(owner, error);
}
bool KvSharedPool::alias(uint64_t target, const KvOwnerKey& key, uint64_t source, const KvOwnerKey& source_key,
                         int64_t prefix, const program::ContinuationCertificate& cut, std::string& error,
                         KvAliasReceipt* receipt, void* source_stream, void* target_stream) {
    using Clock = std::chrono::steady_clock;
    const auto started = receipt ? Clock::now() : Clock::time_point{};
    auto mark = started;
    auto phase = [&](double& field) {
        const auto now = Clock::now(); field += std::chrono::duration<double, std::micro>(now - mark).count();
        mark = now;
    };
    if (receipt) *receipt = {};
    const auto quote = plan::kv_share_quote(kernels::qsa_real_shapes(), kernels::kKvHybrid, budget_, prefix);
    if (budget_.mode != plan::KvShareMode::alias || !quote.admitted) return fail(error, quote.refusal);
    if (readers_.pinned(target)) return fail(error, "alias target is borrowed by a streamed source");
    if (receipt) phase(receipt->quote_us);
    if (!fence(source_stream, target_stream, false, error, receipt)) return false;
    if (receipt) phase(receipt->entry_drain_us);
    if (!state_.acquire(target, key, source, source_key, prefix, cut, true))
        return fail(error, "certified generation-bound acquire refused");
    if (receipt) phase(receipt->acquire_us);
    const size_t count = (size_t) prefix / 4;
    // Acquire pins the source generation and published device table. A single
    // ordered D2D run avoids pageable-host staging; old backing stays charged.
    if (cudaMemcpyAsync(device_tables_[(size_t) target], device_tables_[(size_t) source], count * sizeof(int32_t),
                        cudaMemcpyDeviceToDevice, static_cast<cudaStream_t>(target_stream)) != cudaSuccess) {
        poisoned_ = true; return fail(error, "alias publication failed; references quarantined");
    }
    if (receipt) {
        receipt->table_bytes = count * sizeof(int32_t); ++receipt->table_calls; phase(receipt->table_copy_us);
    }
    if (!fence(source_stream, target_stream, true, error, receipt)) { poisoned_ = true; return false; }
    if (receipt) phase(receipt->publication_drain_us);
    if (!state_.bind(target, key, true)) { poisoned_ = true; return fail(error, "alias bind failed after publication"); }
    if (receipt) phase(receipt->bind_us);
    for (size_t page = 0; page < count; ++page) release_extent((uint64_t) tables_[(size_t) target][page]);
    std::memcpy(tables_[(size_t) target].data(), tables_[(size_t) source].data(), count * sizeof(int32_t));
    if (receipt) phase(receipt->directory_us);
    const bool complete = reclaim(error, receipt);
    if (receipt) receipt->total_us = std::chrono::duration<double, std::micro>(Clock::now() - started).count();
    return complete;
}
bool KvSharedPool::writable(uint64_t owner, const KvOwnerKey& key, uint64_t page, std::string& error) {
    int current = -1;
    if (!usable(owner, key) || readers_.pinned(owner) || cudaGetDevice(&current) != cudaSuccess || current != device_)
        return fail(error, "stale, foreign or borrowed write owner");
    const auto decision = state_.cow(owner, key, page, true); // pure permission, not a drain claim
    if (decision == plan::KvCowDecision::private_write) return true;
    if (!drain(error)) return false;
    if (decision != plan::KvCowDecision::detach || !state_.detach(owner, key, page, true))
        return fail(error, "unpriced, pinned or stale COW refused");
    const uint64_t old = state_.binding(owner, page).extent;
    const uint64_t fresh = state_.owner(owner).pending_private.extent;
    if (!retain_extent(fresh, error)) return false;
    for (size_t part = 0; part < per_page_.size(); ++part) {
        auto* base = range_.base() + offsets_[part];
        if (cudaMemcpy(base + fresh * per_page_[part], base + old * per_page_[part], (size_t) per_page_[part],
                       cudaMemcpyDeviceToDevice) != cudaSuccess) {
            poisoned_ = true; return fail(error, "COW copy failed; old extent retained");
        }
    }
    const int32_t binding = (int32_t) fresh;
    if (cudaMemcpy(device_tables_[(size_t) owner] + page, &binding, sizeof(binding), cudaMemcpyHostToDevice) != cudaSuccess ||
        !drain(error) || !state_.detach_commit(owner, key, true)) {
        poisoned_ = true; return fail(error, "COW publication failed; references quarantined");
    }
    tables_[(size_t) owner][(size_t) page] = binding;
    release_extent(old);
    return reclaim(error);
}
bool KvSharedPool::terminal(uint64_t owner, const KvOwnerKey& key, std::string& error) {
    if (readers_.pinned(owner) || !drain(error) || !state_.terminal(owner, key, true))
        return fail(error, "stale, borrowed, pinned or duplicate terminal refused");
    for (const int32_t extent : tables_[(size_t) owner]) release_extent((uint64_t) extent);
    return reclaim(error);
}
bool KvSharedPool::mount(QsaState& state, uint64_t owner, const KvOwnerKey& key, std::string& error) {
    if (!state_.owns(owner, key) || state.kv_mode || state.kv_elastic >= 0 || state.host.present() ||
        state.kv_int8 || state.kv_q4) return fail(error, "incompatible common-pool mount");
    if (readers_.pinned(owner) ||
        (state.shared_pool && state.shared_pool->readers_.pinned(state.shared_owner)))
        return fail(error, "mounted state is borrowed by a streamed source");
    const auto p = pools(owner);
    state.shared_pool = this; state.shared_owner = owner;
    state.shared_request = key.request; state.shared_incarnation = key.incarnation;
    state.allocation_id = key.allocation;
    state.kv_hybrid = true;
    state.k_q = const_cast<int8_t*>(p.k_q); state.k_scale = const_cast<uint16_t*>(p.k_scale);
    state.v_q4 = const_cast<uint8_t*>(p.v_q4); state.page_table = device_tables_[(size_t) owner];
    state.n_pages = (int64_t) state_.pages(); state.max_cells = (int64_t) budget_.cells;
    state.n_slots = (int64_t) state_.extent_count();
    return true;
}
bool KvSharedPool::reset(uint64_t owner, const KvOwnerKey& key, std::string& error) {
    if (!usable(owner, key) || state_.owner(owner).cancelled || readers_.pinned(owner))
        return fail(error, "cancelled or stale common-pool reset");
    int32_t* table = device_tables_[(size_t) owner];
    // Drop only this owner's references AFTER the existing drain. Peers retain
    // their pages. Reacquire private capacity before any restore/zero writes;
    // common bases and private device-table addresses stay graph-stable.
    return terminal(owner, key, error) && begin(owner, key, table, error);
}
std::shared_ptr<KvSharedPool> KvSharedPool::borrow(uint64_t owner, const KvOwnerKey& key, std::string& error) {
    if (!usable(owner, key) || state_.owner(owner).cancelled) {
        fail(error, "stale or cancelled streamed source"); return {};
    }
    const uint64_t generation = readers_.acquire(owner);
    if (!generation) { fail(error, "streamed source lifetime unavailable"); return {}; }
    return {this, [owner, generation](KvSharedPool* pool) { pool->readers_.release(owner, generation); }};
}
bool kv_shared_valid(const QsaState& state) {
    if (!state.shared_pool) return true;
    const auto& p = *state.shared_pool;
    const KvOwnerKey key{state.shared_request, state.shared_incarnation, state.allocation_id};
    return state.kv_mode == 0 && state.kv_hybrid && !state.kv_q4 && !state.kv_int8 &&
        p.usable(state.shared_owner, key) &&
        state.page_table == p.pools(state.shared_owner).page_table &&
        state.k_q == p.pools(state.shared_owner).k_q && state.k_scale == p.pools(state.shared_owner).k_scale &&
        state.v_q4 == p.pools(state.shared_owner).v_q4 &&
        state.n_pages == (int64_t) p.ownership().pages() &&
        state.n_slots == (int64_t) p.ownership().extent_count();
}
KvLogicalPages kv_logical_pages(const QsaState& state) {
    if (!state.shared_pool) return {};
    const auto& p = *state.shared_pool;
    return {p.table(state.shared_owner).data(), (uint64_t) state.n_pages, (uint64_t) state.n_slots};
}
bool kv_shared_reset(const QsaState& state, std::string& error) {
    if (!state.shared_pool) return true;
    if (!kv_shared_valid(state)) { error = "KV shared pool: stale mounted owner"; return false; }
    return state.shared_pool->reset(state.shared_owner,
        {state.shared_request, state.shared_incarnation, state.allocation_id}, error);
}
kernels::QsaAttnPools KvSharedPool::pools(uint64_t owner) const {
    kernels::QsaAttnPools p;
    p.k_q = reinterpret_cast<int8_t*>(range_.base() + offsets_[0]);
    p.k_scale = reinterpret_cast<uint16_t*>(range_.base() + offsets_[1]);
    p.v_q4 = range_.base() + offsets_[2];
    p.page_table = device_tables_[(size_t) owner];
    return p;
}
uint64_t KvSharedPool::mapped_bytes() const { return (uint64_t) range_.mapped_count() * budget_.granularity; }
uint64_t KvSharedPool::host_bytes() const {
    uint64_t bytes = sizeof(*this) + state_.host_bytes() - sizeof(state_) +
        chunk_users_.capacity() * sizeof(uint32_t) + (uint64_t) range_.chunks() * sizeof(VmmChunk) +
        tables_.capacity() * sizeof(std::vector<int32_t>) + device_tables_.capacity() * sizeof(int32_t*);
    for (const auto& table : tables_) bytes += table.capacity() * sizeof(int32_t);
    return bytes;
}
} // namespace strata::core
