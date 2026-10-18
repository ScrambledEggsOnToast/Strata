#pragma once

#include "strata/core/conversation_store.hpp"

#include <algorithm>
#include <cstdint>
#include <new>
#include <optional>
#include <type_traits>

namespace strata::program {

struct ConversationStoreRamBudget {
    uint64_t envelope = 0;
    uint64_t preparation = 0;
    uint64_t outside = 0;

    uint64_t allowance(const core::ConversationCache& cache, uint64_t local_bytes = 0) const {
        uint64_t available = envelope;
        for (uint64_t charge : {preparation, uint64_t(cache.envelope_bytes()), outside, local_bytes})
            available -= std::min(available, charge);
        return available;
    }
};

// Suspend new reader starts while the owner changes its cache charge. This is not
// a lock or a promise of immediate service: existing charged work remains alive.
class StoreRamAdmission {
public:
    StoreRamAdmission(core::ConversationStore* store, core::ConversationCache& cache,
                      ConversationStoreRamBudget budget)
        : store_(store), cache_(cache), budget_(budget) {
        if (store_) store_->suspend_read_admission();
    }
    StoreRamAdmission(const StoreRamAdmission&) = delete;
    StoreRamAdmission& operator=(const StoreRamAdmission&) = delete;
    ~StoreRamAdmission() { publish(); }
    void publish(uint64_t local_bytes = 0) {
        if (store_) store_->publish_read_allowance(budget_.allowance(cache_, local_bytes));
    }
    uint64_t held() const {
        if (!store_) return 0;
        const uint64_t held = store_->ram_held_bytes();
        return budget_.preparation > UINT64_MAX - held ? UINT64_MAX : held + budget_.preparation;
    }
private:
    core::ConversationStore* store_;
    core::ConversationCache& cache_;
    ConversationStoreRamBudget budget_;
};

struct ConversationStoreDemandAdmission {
    bool accepted = false;
    bool recovery_attempted = false;
    bool retried = false;
    uint64_t required_increment_bytes = 0;
    size_t released_bytes = 0;
    uint64_t allowance_bytes = 0;
};

// Disk mount policy only: the reader leaves MAN1 draft coverage unknown. MTP
// requires the explicit interval on the final physical stage before any GPU write.
// RAM-parked restore keeps its own certification contract.
inline bool conversation_store_restore_compatible(const core::SavedConversation& image, bool use_mtp) {
    if (!use_mtp) return true;
    const auto& draft = image.stage_images.empty() ? image : image.stage_images.back();
    return draft.draft_begin >= 0 && draft.draft_cells >= draft.draft_begin;
}

// One authoritative RAM rollback candidate survives every demand in a shared ->
// private-disk -> RAM chain. Only a disk candidate belongs in private_disk: a
// second copy of ram would retain a stale cache index after admission compacts it.
template<class Candidate>
struct ConversationStoreDemandFallback {
    Candidate ram;
    std::optional<Candidate> private_disk;

    const Candidate& after_failure(bool shared) const noexcept {
        return shared && private_disk ? *private_disk : ram;
    }
};

// The actual owner-side sequence, shared by serving and CPU ownership regressions.
// Only a fresh peak-budget refusal can reclaim RAM. Accepted/adopted reads retain
// their existing reservation. The fallback's parked index is protected/rebased;
// no publication pump or GPU-owner mutation is permitted in either view callback.
// Refresh may allocate. Invalidate must publish absent coverage without allocation
// and preserve the caller's dirty flag, so failed optional diagnostics cannot turn
// successful reclamation into a request failure or advertise stale RAM coverage.
template<class Now, class Cancelled, class Refresh, class Invalidate>
ConversationStoreDemandAdmission admit_store_demand(
    core::ConversationStore& store, core::ConversationCache& cache,
    const core::ConversationStoreQuery& query, std::string& why,
    ConversationStoreRamBudget budget, size_t* protected_index,
    Now&& now, Cancelled&& cancelled, Refresh&& refresh, Invalidate&& invalidate) {
    static_assert(std::is_nothrow_invocable_v<Invalidate>);
    ConversationStoreDemandAdmission out;
    const auto current = [&] { return !cancelled() && query.deadline_ns && now() < query.deadline_ns; };
    if (!current()) return out;
    store.publish_read_allowance(budget.allowance(cache));
    core::ConversationStoreReadRefusal refusal;
    out.accepted = store.demand(query, why, &refusal);
    if (out.accepted || !refusal.additional_ram_bytes || !current()) return out;

    out.recovery_attempted = true;
    out.required_increment_bytes = refusal.additional_ram_bytes;
    const size_t before = cache.envelope_bytes(), count = cache.size();
    bool recovered = false;
    {
        StoreRamAdmission reservation(&store, cache, budget);
        const uint64_t held = reservation.held();
        if (refusal.additional_ram_bytes <= SIZE_MAX && held <= SIZE_MAX &&
            budget.outside <= SIZE_MAX - held && current())
            recovered = cache.reserve_snapshot(size_t(refusal.additional_ram_bytes),
                size_t(held + budget.outside), protected_index);
        if (cache.size() != count) {
            try { refresh(); }
            catch (const std::bad_alloc&) { invalidate(); }
        }
    }
    out.released_bytes = before - std::min(before, cache.envelope_bytes());
    out.allowance_bytes = store.ram_read_budget();
    if (recovered && out.released_bytes && current()) {
        // Allowance publication can restart speculation, and the catalog can grow.
        // The store revalidates both; a second refusal never starts another reclaim.
        out.retried = true;
        why.clear();
        out.accepted = store.demand(query, why);
    }
    return out;
}

} // namespace strata::program
