// CPU-only topology/admission/reclaim policy for the default-off K8V4 experiment.
#pragma once
#include "strata/plan/kv_share.hpp"
#include <array>
#include <span>

namespace strata::plan {
inline constexpr uint32_t kKvShareMaxStages = 4;
using KvDeviceUuid = std::array<uint8_t, 16>;
inline bool kv_device_uuid_valid(const KvDeviceUuid& uuid) {
    for (uint8_t byte : uuid) if (byte) return true;
    return false;
}
enum class KvLayerFamily : uint8_t { unknown, target_k8v4, recurrent, draft, ple };
// This is the pool allocation's generation, not an owner's request incarnation.
// Individual layers still require the existing same-pool/continuation checks.
struct KvTopologyKey {
    KvDeviceUuid device_uuid{};
    uint32_t stage = UINT32_MAX;
    KvLayerFamily family = KvLayerFamily::unknown;
    uint64_t allocation_generation = 0;
    bool valid() const {
        return kv_device_uuid_valid(device_uuid) && stage < kKvShareMaxStages &&
            family == KvLayerFamily::target_k8v4 && allocation_generation;
    }
    bool operator==(const KvTopologyKey&) const = default;
};
struct KvStageAlias { KvTopologyKey source, target; };
struct KvTopologyDecision {
    bool alias = false;
    const char* refusal = "copy_path";
    uint32_t refused_stage = UINT32_MAX;
};
// This decision is all-or-copy, never permission to publish a partial pipeline.
// UUIDs are CUDA's raw UUID bytes, not visible ordinals. No P2P/bounce alias route.
inline KvTopologyDecision kv_share_topology(KvShareMode mode, std::span<const KvStageAlias> stages,
                                            uint32_t mandatory_stages) {
    if (mode != KvShareMode::alias) return {};
    constexpr uint32_t all = (1u << kKvShareMaxStages) - 1;
    if (!mandatory_stages || (mandatory_stages & ~all) || stages.size() > kKvShareMaxStages)
        return {false, "invalid_stage_set"};
    uint32_t seen = 0;
    for (const auto& stage : stages) {
        const auto index = stage.source.stage;
        if (index >= kKvShareMaxStages) return {false, "invalid_topology", index};
        const uint32_t bit = 1u << index;
        if (seen & bit) return {false, "duplicate_stage", index};
        seen |= bit;
        if (!(mandatory_stages & bit)) continue;
        if (!stage.source.valid() || !stage.target.valid() || !(stage.source == stage.target))
            return {false, "topology_mismatch", index};
    }
    if ((seen & mandatory_stages) != mandatory_stages) return {false, "missing_mandatory_stage"};
    return {true, "none"};
}
// Topology adds to (never replaces) the existing generation/certification gate.
inline KvTopologyDecision kv_share_topology_eligible(const KvShareEligibility& in,
        const strata::program::ContinuationCertificate& cut, std::span<const KvStageAlias> stages,
        uint32_t mandatory_stages) {
    const auto topology = kv_share_topology(in.mode, stages, mandatory_stages);
    if (!topology.alias) return topology;
    if (!kv_share_eligible(in, cut)) return {false, "owner_ineligible"};
    return topology;
}
// All inputs are disjoint physical byte classes. Shared backing is supplied once,
// not once per alias; mutable_tail excludes that backing. resident_other excludes
// every stage demand below. Scratch and COW coexist at the conservative peak.
struct KvDeviceBudget {
    KvDeviceUuid device_uuid{};
    uint64_t capacity = 0, resident_other = 0, reserve = 0;
};
struct KvStageDemand {
    KvTopologyKey key;
    uint64_t shared_bytes = 0, private_recurrent = 0, private_draft = 0, private_ple = 0;
    uint64_t mutable_tail = 0, operation_scratch = 0, transient_cow = 0;
};
// cache_images includes leased images, not an additional copy of their bytes.
// The cache envelope is partitioned, not charged again on top of this sum.
struct KvShareHostBudget {
    uint64_t capacity = 0, resident_other = 0, reserve = 0;
    uint64_t cache_images = 0, shared_metadata = 0, transient_cow = 0;
};
struct KvStageQuote {
    uint64_t steady = 0, peak = 0;
    uint32_t device_scope = UINT32_MAX;
};
struct KvScopeQuote { uint64_t steady = 0, peak = 0, required = 0; };
struct KvShareStageQuote {
    bool known = false, admitted = false;
    const char* refusal = "unknown_layout";
    uint32_t refused_stage = UINT32_MAX;
    std::array<KvStageQuote, kKvShareMaxStages> stages{};
    std::array<KvScopeQuote, kKvShareMaxStages> devices{};
    KvScopeQuote host;
};
// No signed net saving enters admission. Multiple stages on one UUID share one
// capacity/reserve; different UUIDs and host RAM are independently constrained.
inline KvShareStageQuote kv_share_stage_quote(KvShareMode mode, std::span<const KvDeviceBudget> devices,
        std::span<const KvStageDemand> stages, const KvShareHostBudget& host) {
    KvShareStageQuote q;
    if (mode == KvShareMode::off) { q.known = q.admitted = true; q.refusal = "none"; return q; }
    if (devices.empty() || devices.size() > kKvShareMaxStages ||
        stages.empty() || stages.size() > kKvShareMaxStages) return q;
    LogicalBytes bytes;
    for (size_t i = 0; i < devices.size(); ++i) {
        if (!kv_device_uuid_valid(devices[i].device_uuid)) return q;
        for (size_t j = 0; j < i; ++j) if (devices[i].device_uuid == devices[j].device_uuid) return q;
        q.devices[i].steady = q.devices[i].peak = devices[i].resident_other;
    }
    uint32_t seen = 0;
    for (const auto& demand : stages) {
        if (!demand.key.valid() || (seen & (1u << demand.key.stage))) return q;
        seen |= 1u << demand.key.stage;
        size_t scope = 0;
        while (scope < devices.size() && devices[scope].device_uuid != demand.key.device_uuid) ++scope;
        if (scope == devices.size()) return q;
        auto& stage = q.stages[demand.key.stage];
        stage.device_scope = uint32_t(scope);
        stage.steady = bytes.sum({demand.shared_bytes, demand.private_recurrent, demand.private_draft,
                                 demand.private_ple, demand.mutable_tail});
        stage.peak = bytes.sum({stage.steady, demand.operation_scratch, demand.transient_cow});
        q.devices[scope].steady = bytes.sum({q.devices[scope].steady, stage.steady});
        q.devices[scope].peak = bytes.sum({q.devices[scope].peak, stage.peak});
    }
    for (size_t i = 0; i < devices.size(); ++i)
        q.devices[i].required = bytes.sum({q.devices[i].peak, devices[i].reserve});
    q.host.steady = bytes.sum({host.resident_other, host.cache_images, host.shared_metadata});
    q.host.peak = bytes.sum({q.host.steady, host.transient_cow});
    q.host.required = bytes.sum({q.host.peak, host.reserve});
    const uint64_t host_growth = bytes.sum({host.shared_metadata, host.transient_cow});
    if (!bytes.valid) return {};
    q.known = true;
    for (const auto& stage : stages) {
        const auto scope = q.stages[stage.key.stage].device_scope;
        if (q.devices[scope].required > devices[scope].capacity) {
            q.refusal = "stage_capacity"; q.refused_stage = stage.key.stage; return q;
        }
    }
    if (host_growth > kKvShareHostLimit) { q.refusal = "host_growth_limit"; return q; }
    if (q.host.required > host.capacity) { q.refusal = "host_capacity"; return q; }
    q.admitted = true; q.refusal = "none";
    return q;
}
enum class KvReclaimMode : uint8_t { synchronous, coalesced, deferred };
struct KvDeviceReclaimPolicy {
    KvDeviceUuid device_uuid{};
    KvReclaimMode mode = KvReclaimMode::synchronous;
    uint64_t min_run_chunks = 1, max_batch_chunks = 1;
    bool valid_for(const KvTopologyKey& key) const {
        if (!key.valid() || device_uuid != key.device_uuid || !min_run_chunks ||
            max_batch_chunks < min_run_chunks) return false;
        switch (mode) {
        case KvReclaimMode::synchronous: return min_run_chunks == 1 && max_batch_chunks == 1;
        case KvReclaimMode::coalesced:
        case KvReclaimMode::deferred: return true;
        }
        return false;
    }
};
enum class KvReclaimAction : uint8_t {
    refuse, wait_pipeline, hold, enqueue, wait_worker, release_inline, release_worker, complete
};
struct KvReclaimPlan {
    KvReclaimAction action = KvReclaimAction::refuse;
    uint64_t chunks = 0;
};
// ready_chunks is one contiguous, unreferenced run, not scattered pages whose
// aggregate happens to reach the threshold. Flush permits the final short run.
// Batch sizes are explicit per UUID; these are policy inputs, not hardware tuning.
inline KvReclaimPlan kv_reclaim_plan(const KvDeviceReclaimPolicy& policy, const KvTopologyKey& key,
        uint64_t ready_chunks, bool pipeline_drained, bool flush) {
    if (!policy.valid_for(key)) return {};
    if (!ready_chunks) return {KvReclaimAction::complete, 0};
    if (!pipeline_drained) return {KvReclaimAction::wait_pipeline, 0};
    if (!flush && ready_chunks < policy.min_run_chunks) return {KvReclaimAction::hold, 0};
    return {policy.mode == KvReclaimMode::deferred ? KvReclaimAction::enqueue : KvReclaimAction::release_inline,
            std::min(ready_chunks, policy.max_batch_chunks)};
}
// One device scope at a scheduler-designated idle window. `other_charged` excludes
// this pool's ledger bytes; incoming includes private/scratch/COW peaks. Reserve is
// charged once. An idle signal is not a measured duration/cost prediction.
struct KvReclaimWindow {
    uint64_t capacity = 0, other_charged = 0, incoming = 0, reserve = 0;
    bool idle = false, flush = false;
};
struct KvReclaimDispatch {
    bool known = false, admission_ready = false;
    KvReclaimAction action = KvReclaimAction::refuse;
    uint64_t chunks = 0, required_before = 0, headroom_bytes = 0;
};
// Pending bytes are a subset of charged bytes, never credit and never a second
// charge. This is a single-controller decision, not a concurrent reservation.
inline KvReclaimDispatch kv_reclaim_dispatch(const KvDeviceReclaimPolicy& policy, const KvTopologyKey& key,
        uint64_t charged, uint64_t pending, uint64_t ready_chunks, bool pipeline_drained,
        const KvReclaimWindow& window) {
    KvReclaimDispatch q;
    if (!policy.valid_for(key) || !window.capacity || pending > charged) return q;
    LogicalBytes bytes;
    const uint64_t held = bytes.sum({charged, window.other_charged, window.reserve});
    q.required_before = bytes.sum({held, window.incoming});
    if (!bytes.valid) return {};
    q.known = true;
    q.admission_ready = q.required_before <= window.capacity;
    q.headroom_bytes = held <= window.capacity ? window.capacity - held : 0;
    if (!pipeline_drained) { q.action = KvReclaimAction::wait_pipeline; return q; }
    if (pending) { q.action = KvReclaimAction::wait_worker; return q; }
    if (ready_chunks > charged / kKvShareChunk) return {};
    const bool pressure = !q.admission_ready;
    const auto selected = kv_reclaim_plan(policy, key, ready_chunks, true, window.flush || pressure);
    q.action = selected.action;
    if (selected.action == KvReclaimAction::complete || selected.action == KvReclaimAction::hold) return q;
    if (policy.mode != KvReclaimMode::synchronous && !window.idle && !pressure && !window.flush) {
        q.action = KvReclaimAction::hold; return q;
    }
    // A capacity deadline or explicit retirement flush cannot be satisfied by
    // early worker return. Release inline (or join above); re-quote after release.
    q.action = policy.mode == KvReclaimMode::deferred && window.idle && !pressure && !window.flush ?
        KvReclaimAction::enqueue : KvReclaimAction::release_inline;
    q.chunks = selected.chunks;
    return q;
}
} // namespace strata::plan
