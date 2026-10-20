#include "strata/core/kv_alias.hpp"
#include "strata/core/logical_kv.hpp"
#include "strata/core/kv_reclaim.hpp"
#include "strata/plan/kv_share_topology.hpp"
#include <cstdio>
#include <stdexcept>
#include <thread>

namespace {
using namespace strata;
void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
program::ContinuationCertificate cut(core::KvOwnerKey key, int64_t cells) {
    program::ContinuationInput in;
    in.request = key.request; in.incarnation = key.incarnation;
    in.committed_tokens = cells;
    in.needs_qsa = in.needs_gdn = in.needs_ple = in.needs_stages = true;
    return program::certify_continuation(in);
}
plan::KvTopologyKey topology(uint32_t stage, uint8_t device, uint64_t generation = 7) {
    plan::KvTopologyKey key;
    key.device_uuid[0] = device; key.stage = stage;
    key.family = plan::KvLayerFamily::target_k8v4; key.allocation_generation = generation;
    return key;
}
void topology_alias() {
    using namespace plan;
    std::array<KvStageAlias, 3> stages{};
    for (uint32_t i = 0; i < stages.size(); ++i) stages[i] = {topology(i, uint8_t(i + 1)), topology(i, uint8_t(i + 1))};
    require(kv_share_topology(KvShareMode::alias, stages, 7).alias, "same-stage/device/generation placement refused");
    KvShareEligibility eligible{KvShareMode::alias, 1, 2, 3, 3, 65539, 262144, true, true, false, true};
    auto certificate = cut({1, 2, 3}, 65539);
    require(kv_share_topology_eligible(eligible, certificate, stages, 7).alias,
        "topology-aware continuation eligibility refused a compatible cut");
    certificate.certified = false;
    require(!kv_share_topology_eligible(eligible, certificate, stages, 7).alias,
        "matching topology bypassed continuation certification");
    certificate = cut({1, 2, 3}, 65539);
    stages[1].target.stage = 2;
    require(!kv_share_topology(KvShareMode::alias, stages, 7).alias, "cross-stage alias accepted");
    stages[1].target = topology(1, 3);
    const auto foreign = kv_share_topology(KvShareMode::alias, stages, 7);
    require(!foreign.alias && foreign.refused_stage == 1, "cross-device alias accepted or refusal scope lost");
    require(!kv_share_topology_eligible(eligible, certificate, stages, 7).alias,
        "valid continuation bypassed the foreign-device refusal");
    stages[1].target = topology(1, 2, 8);
    require(!kv_share_topology(KvShareMode::alias, stages, 7).alias, "stale pool allocation generation aliased");
    stages[1].target = topology(1, 2);
    stages[1].target.family = KvLayerFamily::draft;
    require(!kv_share_topology(KvShareMode::alias, stages, 7).alias, "cross-family alias accepted");
    stages[1].source.family = KvLayerFamily::draft;
    require(!kv_share_topology(KvShareMode::alias, stages, 7).alias, "private draft family shared");
    stages[1] = {topology(1, 2), topology(1, 2)};
    require(!kv_share_topology(KvShareMode::alias, std::span(stages).first(2), 7).alias,
        "missing mandatory downstream stage produced a partial alias plan");
    stages[2] = stages[1];
    require(!kv_share_topology(KvShareMode::alias, stages, 7).alias, "duplicate stage hid missing placement");
    stages[2] = {topology(2, 3), topology(2, 3)};
    stages[2].target.device_uuid = {};
    require(!kv_share_topology(KvShareMode::alias, stages, 7).alias, "unknown device UUID admitted");
    require(kv_share_topology(KvShareMode::alias, std::span(stages).first(2), 3).alias,
        "matching declared mandatory stages refused");
    require(!kv_share_topology(KvShareMode::off, stages, 7).alias &&
        !kv_share_topology(KvShareMode::pool, stages, 7).alias &&
        !kv_share_topology(KvShareMode::alias, stages, 0).alias, "off/pool/empty placement enabled aliasing");
}
void topology_admission() {
    using namespace plan;
    constexpr uint64_t mib = 1ull << 20, gib = 1ull << 30;
    std::array<KvDeviceBudget, 3> devices{};
    std::array<KvStageDemand, 3> stages{};
    for (uint32_t i = 0; i < stages.size(); ++i) {
        const auto key = topology(i, uint8_t(i + 1));
        devices[i] = {key.device_uuid, (i == 2 ? 24 : 32) * gib, 20 * gib, gib};
        stages[i] = {key, 2 * gib, 128 * mib, 256 * mib, 32 * mib, 64 * mib, 512 * mib, 256 * mib};
    }
    KvShareHostBudget host{12 * gib, gib, gib, 8 * gib, 256 * mib, mib};
    auto q = kv_share_stage_quote(KvShareMode::alias, devices, stages, host);
    require(q.known && !q.admitted && q.refused_stage == 2 &&
        std::strcmp(q.refusal, "stage_capacity") == 0, "24GiB pressure borrowed aggregate V100 credit");
    require(q.stages[2].steady == 2528 * mib && q.stages[2].peak == 3296 * mib &&
        q.devices[2].required == 24800 * mib, "stage private/scratch/COW demand underpriced or counted twice");
    require(q.host.required == 10497 * mib, "host image/metadata/COW scopes counted twice");
    devices[2].resident_other -= 224 * mib;
    q = kv_share_stage_quote(KvShareMode::alias, devices, stages, host);
    require(q.admitted && q.devices[2].required == 24 * gib, "exact 24GiB fit refused");
    ++stages[2].transient_cow;
    require(!kv_share_stage_quote(KvShareMode::alias, devices, stages, host).admitted,
        "one-byte transient stage overage admitted");
    --stages[2].transient_cow;
    host.capacity = host.resident_other + host.reserve + host.cache_images + host.shared_metadata + host.transient_cow - 1;
    q = kv_share_stage_quote(KvShareMode::pool, devices, stages, host);
    require(q.known && !q.admitted && std::strcmp(q.refusal, "host_capacity") == 0,
        "host pressure borrowed device capacity");
    host.capacity = 12 * gib;
    stages[1].key.device_uuid = stages[0].key.device_uuid;
    devices[0].capacity = 27 * gib;
    q = kv_share_stage_quote(KvShareMode::alias, devices, stages, host);
    require(q.known && !q.admitted && q.devices[0].required == 28096 * mib,
        "two stages on one device each spent the same capacity");
    stages[1].key = stages[0].key;
    require(!kv_share_stage_quote(KvShareMode::alias, devices, stages, host).known,
        "duplicate stage double counted instead of refusing");
    stages[1].key = topology(1, 2);
    stages[0].shared_bytes = UINT64_MAX;
    require(!kv_share_stage_quote(KvShareMode::alias, devices, stages, host).known, "overflow became admission credit");
    const auto off = kv_share_stage_quote(KvShareMode::off, devices, stages, host);
    require(off.known && off.admitted && off.devices[0].required == 0 && off.host.required == 0,
        "default-off topology pricing added allocator demand");
}
void pipeline_lifetime() {
    using namespace core;
    std::array<KvAliasState, 3> pools{{{2, 4}, {2, 4}, {2, 4}}};
    const KvOwnerKey owner{77, 2, 9}, peer{78, 1, 10};
    const KvPipelineKey submission{owner, 11}, stale{owner, 10};
    KvPipelineLifetime lifetime(submission, 7, {13056, 13056, 13056, 0});
    for (uint32_t i = 0; i < pools.size(); ++i)
        require(pools[i].begin(0, owner) && pools[i].begin(1, peer) &&
            pools[i].bind_pipeline(0, owner, lifetime, i), "pipeline owner/binding setup failed");
    require(!lifetime.local_drain(submission, 2) && !lifetime.local_drain(stale, 0) &&
        !lifetime.terminal_drain(submission), "out-of-order or stale pipeline drain accepted");
    require(lifetime.local_drain(submission, 0) && !lifetime.local_drain(submission, 0),
        "local pipeline drain was not exactly once");
    require(lifetime.cancel(submission) && lifetime.cancelled() &&
        !pools[0].terminal_pipeline(0, owner, lifetime, 0) && lifetime.charged(0) == 13056 &&
        pools[0].extent(pools[0].binding(0, 0).extent).references == 1,
        "cancel/local drain released pages still visible downstream");
    require(lifetime.local_drain(submission, 1) && !lifetime.terminal_drain(submission) &&
        lifetime.charged(1) == 13056, "intermediate stage drain released a pipeline charge");
    require(lifetime.local_drain(submission, 2) && !pools[2].terminal_pipeline(0, owner, lifetime, 2),
        "local drains were mistaken for the explicit terminal pipeline barrier");
    require(lifetime.terminal_drain(submission) && !lifetime.terminal_drain(submission) &&
        lifetime.charged(2) == 13056, "terminal drain itself invented a physical/reference release");
    require(!pools[0].terminal_pipeline(1, peer, lifetime, 0) &&
        !pools[0].terminal_pipeline(0, owner, lifetime, 3) &&
        !pools[1].terminal_pipeline(0, owner, lifetime, 0), "foreign owner/stage/pool consumed pipeline proof");
    for (uint32_t i = 0; i < pools.size(); ++i) {
        require(pools[i].terminal_pipeline(0, owner, lifetime, i) && lifetime.charged(i) == 0 &&
            !pools[i].terminal_pipeline(0, owner, lifetime, i), "pipeline retirement leaked or released twice");
        require(pools[i].owns(1, peer) && pools[i].terminal(1, peer, true), "pipeline retirement damaged peer");
    }
}
void stale_pipeline_reuse() {
    using namespace core;
    KvAliasState pool(2, 4);
    const KvOwnerKey owner{81, 1, 13};
    const KvPipelineKey old_submission{owner, 14};
    KvPipelineLifetime old(old_submission, 1, {13056, 0, 0, 0});
    require(pool.begin(0, owner) && pool.bind_pipeline(0, owner, old, 0) && old.local_drain(old_submission, 0) &&
        old.terminal_drain(old_submission), "stale pipeline setup failed");
    require(pool.terminal(0, owner, true) && pool.begin(0, owner), "same-key legacy reset setup failed");
    require(!pool.terminal_pipeline(0, owner, old, 0) && old.charged(0) == 13056 &&
        pool.owns(0, owner), "old unconsumed pipeline proof retired a same-key replacement owner");
    require(!pool.bind_pipeline(0, owner, old, 0), "drained old proof rebound after same-key reset");
    const KvPipelineKey current_submission{owner, 15};
    KvPipelineLifetime current(current_submission, 1, {13056, 0, 0, 0});
    KvPipelineLifetime duplicate({owner, 16}, 1, {13056, 0, 0, 0});
    require(pool.bind_pipeline(0, owner, current, 0) && !pool.bind_pipeline(0, owner, duplicate, 0) &&
        !pool.terminal_pipeline(0, owner, old, 0) && current.charged(0) == 13056,
        "stale/duplicate lifetime consumed replacement ownership or charge");
    require(current.local_drain(current_submission, 0) && current.terminal_drain(current_submission) &&
        pool.terminal_pipeline(0, owner, current, 0) && !current.charged(0),
        "current bound pipeline could not retire after rejecting stale proof");
    std::printf("KV_SHARE_PIPELINE same_key_reset_stale_proof_refused=1 current_replacement_retired=1\n");
}
void reclaim_policy() {
    using namespace plan;
    using namespace core;
    const auto a = topology(0, 1), b = topology(1, 2), c = topology(2, 3);
    const KvDeviceReclaimPolicy synchronous{a.device_uuid, KvReclaimMode::synchronous, 1, 1};
    const KvDeviceReclaimPolicy coalesced{b.device_uuid, KvReclaimMode::coalesced, 4, 8};
    const KvDeviceReclaimPolicy deferred{c.device_uuid, KvReclaimMode::deferred, 8, 32};
    require(kv_reclaim_plan(synchronous, a, 25, true, false).chunks == 1 &&
        kv_reclaim_plan(coalesced, b, 25, true, false).chunks == 8 &&
        kv_reclaim_plan(deferred, c, 25, true, false).action == KvReclaimAction::enqueue,
        "per-device synchronous/coalesced/deferred choice collapsed to uniform calls");
    require(kv_reclaim_plan(coalesced, b, 3, true, false).action == KvReclaimAction::hold &&
        kv_reclaim_plan(coalesced, b, 3, true, true).chunks == 3 &&
        kv_reclaim_plan(deferred, c, 25, false, true).action == KvReclaimAction::wait_pipeline &&
        kv_reclaim_plan(deferred, a, 25, true, true).action == KvReclaimAction::refuse,
        "run threshold/flush/pipeline/UUID reclaim contract weakened");
    const KvOwnerKey owner{91, 1, 17};
    const KvPipelineKey submission{owner, 18};
    KvPipelineLifetime lifetime(submission, 7, {0, 0, 3 * kKvShareChunk, 0});
    const KvDeviceReclaimPolicy worker_policy{c.device_uuid, KvReclaimMode::deferred, 1, 2};
    KvReclaimRun run(c, worker_policy, submission, 40, 3);
    uint64_t unmap_calls = 0, free_calls = 0;
    bool free_ok = false, unmap_ok = true;
    auto unmap = [&](const KvTopologyKey& key, uint64_t chunk) {
        require(key == c && chunk == 40 + unmap_calls, "reclaim crossed placement or unmapped a chunk twice");
        ++unmap_calls;
        return unmap_ok ? chunk + 100 : uint64_t{0};
    };
    auto release = [&](const KvTopologyKey& key, uint64_t handle) {
        require(key == c && handle >= 140 && handle <= 142, "physical release lost device/run identity");
        ++free_calls; return free_ok;
    };
    auto step = [&](bool worker, bool flush = false) {
        return run.reclaim(lifetime, c.device_uuid, worker, flush, unmap, release);
    };
    require(step(false).action == KvReclaimAction::wait_pipeline &&
        run.charged_bytes() == 3 * kKvShareChunk && !unmap_calls, "reclaim ran before terminal pipeline drain");
    require(lifetime.local_drain(submission, 0) && lifetime.local_drain(submission, 1) &&
        lifetime.local_drain(submission, 2) && lifetime.terminal_drain(submission), "reclaim pipeline setup failed");
    require(step(false).action == KvReclaimAction::enqueue && step(false).action == KvReclaimAction::wait_worker &&
        run.released_bytes() == 0 && run.charged_bytes() == 3 * kKvShareChunk && !free_calls,
        "deferred enqueue released physical credit or queued twice");
    require(!run.reclaim(lifetime, a.device_uuid, true, false, unmap, release).success && !unmap_calls,
        "foreign-device worker touched a reclaim run");
    require(!step(true).success && unmap_calls == 1 && free_calls == 1 &&
        run.unmapped_bytes() == kKvShareChunk && run.released_bytes() == 0 &&
        run.charged_bytes() == 3 * kKvShareChunk, "unmap/failed physical release was reported as released");
    free_ok = true;
    require(step(true).success && unmap_calls == 2 && free_calls == 3 &&
        run.released_bytes() == 2 * kKvShareChunk && run.charged_bytes() == kKvShareChunk,
        "retry re-unmapped a held handle or double counted release");
    require(step(false).action == KvReclaimAction::enqueue && step(true).success &&
        run.unmapped_bytes() == 3 * kKvShareChunk && run.released_bytes() == 3 * kKvShareChunk &&
        !run.charged_bytes() && unmap_calls == 3 && free_calls == 4 &&
        step(false).action == KvReclaimAction::complete, "final deferred run accounting is not exact");
    KvReclaimRun failed(a, synchronous, submission, 43, 1);
    unmap_ok = false;
    const auto failed_unmap = failed.reclaim(lifetime, a.device_uuid, false, true,
        [&](const KvTopologyKey&, uint64_t) { ++unmap_calls; return uint64_t{0}; }, release);
    require(!failed_unmap.success && !failed.unmapped_bytes() && !failed.released_bytes() &&
        failed.charged_bytes() == kKvShareChunk && free_calls == 4, "failed unmap released or lost physical charge");
    KvReclaimRun partial(b, coalesced, submission, 10, 7);
    uint64_t physical_frees = 0;
    bool fail_once = true;
    auto unmap_partial = [&](const KvTopologyKey&, uint64_t chunk) {
        if (chunk == 14 && fail_once) { fail_once = false; return uint64_t{0}; }
        return chunk + 100;
    };
    auto free_partial = [&](const KvTopologyKey&, uint64_t) { ++physical_frees; return true; };
    require(!partial.reclaim(lifetime, b.device_uuid, false, false, unmap_partial, free_partial).success &&
        partial.charged_bytes() == 3 * kKvShareChunk && physical_frees == 4,
        "partial coalesced unmap failure lost successful physical releases");
    require(partial.reclaim(lifetime, b.device_uuid, false, false, unmap_partial, free_partial).success &&
        partial.released_bytes() == 7 * kKvShareChunk && !partial.charged_bytes() && physical_frees == 7,
        "coalesced retry stranded the short remainder of an already admitted batch");
}
void policy() {
    using namespace plan;
    KvShareMode mode = KvShareMode::alias;
    require(kv_share_mode(nullptr, mode) && mode == KvShareMode::off, "sharing is not default off");
    require(kv_share_mode("off", mode) && mode == KvShareMode::off, "explicit off refused");
    require(kv_share_mode("pool", mode) && mode == KvShareMode::pool, "pool mode refused");
    require(kv_share_mode("alias", mode) && mode == KvShareMode::alias, "alias mode refused");
    require(!kv_share_mode("1", mode) && mode == KvShareMode::off, "unknown selection enabled sharing");
    const auto s = kernels::qsa_real_shapes();
    KvShareBudget b{KvShareMode::alias, 262144, 4, 12, 3, kKvShareChunk, 12ull << 30, 3299132688};
    const auto short_q = kv_share_quote(s, kernels::kKvHybrid, b, 32768);
    const auto q = kv_share_quote(s, kernels::kKvHybrid, b, 65536);
    require(short_q.known && !short_q.admitted && short_q.net_cow_peak < 0 &&
        std::strcmp(short_q.refusal, "negative_memory_first_peak") == 0, "32K memory-first refusal lost");
    require(q.admitted && q.complete_pages == 16384 && q.gross_kv_bytes == 641728512 &&
        q.rounded_release == 629145600, "64K K/V-only rounded quote differs");
    require(q.host_metadata == 264331056 && q.cow_host == 1920 && q.cow_device == 75497472 &&
        q.cow_payload == 78336 && q.host_metadata + q.cow_host <= kKvShareHostLimit,
        "full-capacity metadata or COW underpriced");
    require(q.net_cow_peak == 289315152 && short_q.net_cow_peak == -37840560,
        "retained metadata capacity incorrectly counted as released RAM");
    require(q.image_room + q.host_metadata + q.cow_host + b.cache_leases == b.cache_bytes,
        "metadata was not funded inside the host cache envelope");
    for (int64_t p : {65537, 65539}) {
        const auto u = kv_share_quote(s, kernels::kKvHybrid, b, p);
        require(u.complete_pages == q.complete_pages && u.rounded_release == q.rounded_release,
            "partial final page credited");
    }
    b.mode = KvShareMode::pool;
    const auto pool = kv_share_quote(s, kernels::kKvHybrid, b, 65536);
    require(pool.admitted && pool.rounded_release == 0 && pool.gross_kv_bytes == 0,
        "pool-only mode claimed alias saving");
    b.mode = KvShareMode::off;
    const auto off = kv_share_quote(s, kernels::kKvHybrid, b, 65536);
    require(off.admitted && off.host_metadata == 0 && off.cow_device == 0,
        "mode off added allocator demand");
    b.mode = KvShareMode::alias; b.granularity = 4096;
    require(!kv_share_quote(s, kernels::kKvHybrid, b, 65536).known, "unqualified granularity admitted");
    b.granularity = kKvShareChunk; b.cache_bytes = b.cache_leases;
    require(!kv_share_quote(s, kernels::kKvHybrid, b, 65536).admitted, "host cache budget exceeded");
    b.cache_bytes = 12ull << 30; b.cells = 262144; b.owners = 8; b.layers = 48; b.stages = 4;
    const auto maximum = kv_share_quote(s, kernels::kKvHybrid, b, 262140);
    require(maximum.known && maximum.host_metadata + maximum.cow_host <= kKvShareHostLimit,
        "maximum admitted capacity exceeds 2GiB host growth");
    b.cells = 262148;
    require(!kv_share_quote(s, kernels::kKvHybrid, b, 65536).admitted, "unpriced context capacity admitted");
    for (int fmt : {kernels::kKvF16, kernels::kKvInt8, kernels::kKvQ4, kernels::kKvHybrid}) {
        const auto p = qsa_resolve_placement(s, 262144, 0, 0, fmt);
        for (bool rope : {false, true}) {
            const auto legacy = qsa_arena_layout(s, p, rope);
            const auto disabled = kv_share_arena_layout(s, p, rope, KvShareMode::off);
            require(legacy.extent == disabled.extent && legacy.payload_extent == disabled.payload_extent &&
                legacy.pool.k == disabled.pool.k && legacy.pool.v == disabled.pool.v &&
                legacy.pool.k_scale == disabled.pool.k_scale && legacy.pool.v_scale == disabled.pool.v_scale &&
                legacy.page_table == disabled.page_table && legacy.idx_pooled == disabled.idx_pooled &&
                legacy.step == disabled.step, "mode-off carve changed");
        }
    }
    core::KvOwnerKey key{1, 2, 3};
    auto cert = cut(key, 65539);
    KvShareEligibility e{KvShareMode::alias, 1, 2, 3, 3, 65539, 262144, true, true, false, true};
    require(kv_share_eligible(e, cert), "certified unaligned cut refused");
    e.expected_allocation = 4; require(!kv_share_eligible(e, cert), "stale allocation accepted");
    e.expected_allocation = 3; e.preparing_cancelled = true;
    require(!kv_share_eligible(e, cert), "cancelled preparation eligible");
    e.preparing_cancelled = false; cert.certified = false;
    require(!kv_share_eligible(e, cert), "uncertified cut accepted");
    cert = cut(key, 65539); cert.valid_begin = 4;
    require(!kv_share_eligible(e, cert), "non-prefix interval accepted");
    require(kv_share_cow(1, true, true, 1, 0) == KvCowDecision::private_write &&
        kv_share_cow(2, true, true, 1, 1) == KvCowDecision::detach &&
        kv_share_cow(2, true, true, 2, 1) == KvCowDecision::refuse &&
        kv_share_cow(2, true, false, 1, 1) == KvCowDecision::refuse,
        "COW decision weakened the drain or bounded rewrite contract");
    std::printf("KV_SHARE_POLICY prefix32_net_cow=%lld prefix64_release_quote=%llu prefix64_host=%llu prefix64_net_cow=%lld\n",
        (long long) short_q.net_cow_peak, (unsigned long long) q.rounded_release,
        (unsigned long long) q.host_metadata, (long long) q.net_cow_peak);
}
void ownership() {
    core::KvAliasState pool(2, 5);
    const core::KvOwnerKey a{1, 1, 11}, b{2, 1, 22}, stale{2, 2, 22};
    require(pool.begin(0, a) && pool.begin(1, b) && pool.ready(0, a), "owner acquisition failed");
    const auto original_a = pool.binding(0, 0), original_b = pool.binding(1, 0);
    const auto partial = pool.binding(1, 4);
    require(!pool.acquire(1, stale, 0, a, 19, cut(a, 19), true), "stale generation acquired refs");
    require(!pool.acquire(1, b, 0, a, 19, cut(a, 19), false), "undrained acquire accepted");
    require(pool.acquire(1, b, 0, a, 19, cut(a, 19), true), "certified acquire failed");
    require(pool.extent(original_a.extent).references == 2 && pool.binding(1, 0) == original_b &&
        pool.extent(original_b.extent).references == 1, "old capacity released before publication drain");
    require(!pool.terminal(0, a, true) && !pool.bind(1, stale, true) && !pool.bind(1, b, false),
        "pending source or stale/undrained publication changed ownership");
    require(pool.bind(1, b, true) && !pool.bind(1, b, true), "binding not exactly once");
    require(pool.binding(1, 0) == original_a && pool.binding(1, 4) == partial &&
        pool.extent(original_b.extent).references == 0, "partial page shared or private binding retained");
    require(pool.ready(1, b), "target readiness failed");
    const auto before = pool.extent(original_a.extent).acquires;
    require(pool.cow(1, b, 0, true) == plan::KvCowDecision::detach &&
        pool.extent(original_a.extent).acquires == before, "COW decision mutated state");
    require(pool.detach(1, b, 0, true), "detach acquisition failed");
    const auto fresh = pool.owner(1).pending_private;
    require(!(fresh == original_a) && pool.binding(1, 0) == original_a &&
        pool.extent(original_a.extent).references == 2, "detach pre-credited old reference");
    require(pool.detach_commit(1, b, true) && !pool.detach_commit(1, b, true) &&
        pool.binding(1, 0) == fresh && pool.binding(0, 0) == original_a &&
        pool.extent(original_a.extent).references == 1, "COW publication isolation failed");
    const auto held = pool.binding(1, 1);
    const auto releases = pool.extent(held.extent).releases;
    require(pool.cancel(1, b) && pool.extent(held.extent).releases == releases &&
        !pool.terminal(1, b, false) && !pool.terminal(1, stale, true), "cancel or stale terminal released refs");
    require(pool.terminal(1, b, true) && !pool.terminal(1, b, true) &&
        pool.extent(held.extent).releases == releases + 1 && pool.valid(pool.binding(0, 1)),
        "drained terminal not exactly once or survivor invalidated");
    require(pool.terminal(0, a, true) && pool.free_pages() == pool.extent_count(), "last reference leaked");
    require(pool.begin(1, stale) && !pool.terminal(1, b, true) && pool.terminal(1, stale, true),
        "late prior-generation terminal touched replacement");
    for (uint64_t i = 0; i < pool.extent_count(); ++i)
        require(pool.extent(i).acquires == pool.extent(i).releases, "reference count unbalanced");
}
void cancelled_preparation() {
    core::KvAliasState pool(2, 4);
    const core::KvOwnerKey a{1, 1, 1}, b{2, 1, 2};
    require(pool.begin(0, a) && pool.begin(1, b) && pool.ready(0, a), "preparation setup failed");
    const auto page = pool.binding(0, 0);
    require(pool.cancel(1, b) && !pool.acquire(1, b, 0, a, 16, cut(a, 16), true) &&
        pool.extent(page.extent).references == 1, "cancelled preparation acquired aliases");
    require(pool.terminal(1, b, true) && pool.terminal(0, a, true), "preparation terminal failed");
    require(pool.begin(0, a) && pool.begin(1, b) && pool.ready(0, a) &&
        pool.acquire(1, b, 0, a, 16, cut(a, 16), true), "pending cancel setup failed");
    require(pool.cancel(1, b) && !pool.terminal(1, b, true) && !pool.abort(1, b, false) &&
        pool.abort(1, b, true) && pool.terminal(1, b, true) && pool.terminal(0, a, true),
        "pending cancellation bypassed drain/abort ownership");
}
void retention() {
    core::KvAliasState pool(2, 4);
    const core::KvOwnerKey a{1, 1, 1}, b{2, 1, 2};
    require(pool.begin(0, a) && pool.begin(1, b) && pool.ready(0, a) &&
        pool.acquire(1, b, 0, a, 16, cut(a, 16), true) && pool.bind(1, b, true) && pool.ready(1, b),
        "retained cut setup failed");
    const auto shared = pool.binding(1, 0);
    require(pool.terminal(1, b, true, true) && !pool.terminal(1, b, true, true) &&
        !pool.cancel(1, b) && pool.extent(shared.extent).references == 2,
        "retention released a reference or accepted a duplicate terminal/late cancel");
    require(!pool.acquire(1, b, 0, a, 16, cut(a, 16), true) &&
        pool.cow(1, b, 0, true) == plan::KvCowDecision::refuse,
        "retained immutable ownership was writable before a new generation");
    require(!pool.terminal(1, b, false) && pool.terminal(1, b, true) &&
        pool.extent(shared.extent).references == 1 && pool.terminal(0, a, true),
        "retained references did not survive until drained retirement");
}
void logical_images() {
    constexpr size_t width = 8;
    std::vector<uint8_t> physical((131073ull) * width, 0);
    const int32_t bindings[] = {131072, 2, 3};
    const core::KvLogicalPages pages{bindings, 3, 131073};
    std::fill(physical.begin() + 131072 * width, physical.end(), uint8_t{7});
    std::fill(physical.begin() + 2 * width, physical.begin() + 4 * width, uint8_t{11});
    std::vector<uint8_t> captured(3 * width), restored(physical.size(), uint8_t{0xa5});
    require(pages.valid(3), "valid COW page view refused");
    require(pages.visit(width, 0, captured.size(), [&](size_t at, size_t n, size_t logical) {
        std::copy_n(physical.data() + at, n, captured.data() + logical); return true;
    }), "logical raw image capture failed");
    require(captured[0] == 7 && captured[width] == 11 && physical[0] == 0,
        "COW raw image used identity pool offsets");
    require(pages.visit(width, 0, captured.size(), [&](size_t at, size_t n, size_t logical) {
        std::copy_n(captured.data() + logical, n, restored.data() + at); return true;
    }), "logical raw image restore failed");
    const auto verify = [&] {
        return pages.visit(width, 0, captured.size(), [&](size_t at, size_t n, size_t logical) {
            return std::equal(restored.data() + at, restored.data() + at + n, captured.data() + logical);
        });
    };
    require(verify() && restored[0] == 0xa5, "logical image roundtrip overwrote foreign physical page");
    restored[131072 * width] ^= 1;
    require(!verify(), "logical verifier accepted a COW-bound byte mismatch");
    size_t visited = 0;
    require(pages.visit(width, 3, 2 * width, [&](size_t, size_t n, size_t) {
        visited += n; return true;
    }) && visited == 2 * width, "unaligned image segment visit lost bytes");
    const int32_t invalid[] = {131073};
    require(!core::KvLogicalPages{invalid, 1, 131073}.valid(1), "out-of-pool page binding accepted");
}
void borrowed_sources() {
    core::KvReadPins pins;
    require(!pins.acquire(0) && !pins.advance(8) && !pins.release(0, 0), "invalid borrow lifetime accepted");
    require(pins.advance(0), "first borrow lifetime refused");
    const auto old = pins.acquire(0);
    require(old && pins.pinned(0) && pins.any() && !pins.advance(0), "borrow allowed a same-key reset");
    require(pins.release(0, old) && !pins.release(0, old) && !pins.any(), "borrow was not released exactly once");
    require(pins.advance(0), "replacement lifetime refused");
    const auto replacement = pins.acquire(0);
    require(replacement > old && !pins.release(0, old) && pins.pinned(0),
        "old-generation release consumed a replacement borrow");
    require(pins.release(0, replacement) && !pins.any(), "replacement borrow leaked");
}
void acquire_rollback() {
    core::KvAliasState state(2, 4);
    const core::KvOwnerKey a{51, 1, 101}, b{52, 1, 102};
    require(state.begin(0, a) && state.begin(1, b) && state.ready(0, a), "rollback setup failed");
    const auto first = state.binding(0, 0), invalid = state.binding(0, 2);
    const auto acquired = state.extent(first.extent).acquires;
    auto& record = const_cast<core::KvExtentRecord&>(state.extent(invalid.extent));
    for (bool stale : {false, true}) {
        if (stale) ++record.generation; else record.references = UINT32_MAX;
        require(!state.acquire(1, b, 0, a, 16, cut(a, 16), true) &&
            state.extent(first.extent).references == 1 && state.extent(first.extent).acquires == acquired &&
            state.owner(0).pins == 0 && state.owner(1).pending_source == UINT64_MAX,
            "failed fused acquire did not roll back every reference and audit count");
        if (stale) --record.generation; else record.references = 1;
    }
    require(state.acquire(1, b, 0, a, 16, cut(a, 16), true) && state.bind(1, b, true) &&
        state.terminal(1, b, true) && state.terminal(0, a, true), "valid acquire failed after rollback");
}
void deferred_charge() {
    using namespace plan;
    KvShareBudget b{KvShareMode::alias, 262144, 4, 12, 3, kKvShareChunk, 12ull << 30, 3299132688};
    const auto base = kv_share_quote(kernels::qsa_real_shapes(), kernels::kKvHybrid, b, 65539);
    b.deferred_reclaim = true;
    const auto q = kv_share_quote(kernels::qsa_real_shapes(), kernels::kKvHybrid, b, 65539);
    require(q.admitted && q.host_metadata == base.host_metadata + b.layers * kKvShareReclaimHostBytes &&
        q.rounded_release == base.rounded_release && q.cow_device == base.cow_device &&
        q.net_cow_peak == base.net_cow_peak - int64_t(b.layers * kKvShareReclaimHostBytes),
        "deferred worker stack/control was not charged without changing K/V credit");
    require(!kv_share_quote(kernels::qsa_real_shapes(), kernels::kKvHybrid, b, 32768).admitted,
        "deferred reclamation weakened 32K refusal");
    core::KvReclaimLedger ledger;
    require(ledger.mapped(3) && ledger.mapped(8) && !ledger.mapped(3), "physical backing charged twice");
    const auto ticket = ledger.begin();
    require(ticket && ledger.queue(ticket, 3) && ledger.queue(ticket, 8) &&
        !ledger.queue(ticket, 3) && !ledger.begin(), "pending reclaim enqueued twice");
    auto status = ledger.snapshot();
    require(status.charged_bytes == 2 * kKvShareChunk && status.pending_bytes == status.charged_bytes &&
        status.released_chunks == 0 && status.peak_pending_bytes == 2 * kKvShareChunk &&
        !ledger.can_reserve(2 * kKvShareChunk, 0, kKvShareChunk),
        "enqueue credited unavailable physical memory");
    require(!ledger.complete(ticket + 1, 3) && !ledger.complete(ticket, 4) &&
        ledger.complete(ticket, 3) && !ledger.complete(ticket, 3), "stale/duplicate physical completion credited bytes");
    status = ledger.snapshot();
    require(status.charged_bytes == kKvShareChunk && status.pending_bytes == kKvShareChunk &&
        status.released_chunks == 1 && ledger.can_reserve(2 * kKvShareChunk, 0, kKvShareChunk) &&
        !ledger.can_reserve(2 * kKvShareChunk, kKvShareChunk, kKvShareChunk),
        "admission double-counted pending or already committed headroom");
    require(ledger.complete(ticket, 8) && !ledger.complete(ticket, 8) &&
        ledger.snapshot().charged_bytes == 0 && ledger.snapshot().released_chunks == 2,
        "final physical release was not exactly once");
    require(ledger.mapped(3), "released chunk could not be reused");
    const auto replacement = ledger.begin();
    require(replacement > ticket && ledger.queue(replacement, 3) && !ledger.complete(ticket, 3) &&
        ledger.snapshot().charged_bytes == kKvShareChunk && ledger.complete(replacement, 3),
        "stale completion consumed replacement backing");
    require(!ledger.can_reserve(UINT64_MAX, UINT64_MAX, 1) &&
        !ledger.mapped(core::KvReclaimLedger::chunk_limit), "capacity overflow or unpriced directory accepted");
    std::printf("KV_SHARE_DEFERRED_POLICY worker_host_bytes=%llu prefix64_host=%llu prefix64_net_cow=%lld\n",
        (unsigned long long) (b.layers * kKvShareReclaimHostBytes), (unsigned long long) q.host_metadata,
        (long long) q.net_cow_peak);
}
void concurrent_reclaim_observer() {
    core::KvReclaimLedger ledger;
    constexpr uint64_t chunks = 32, bytes = chunks * plan::kKvShareChunk;
    for (uint64_t chunk = 0; chunk < chunks; ++chunk) require(ledger.mapped(chunk), "observer mapping failed");
    const auto ticket = ledger.begin();
    for (uint64_t chunk = 0; chunk < chunks; ++chunk) require(ledger.queue(ticket, chunk), "observer queue failed");
    std::atomic<bool> release{false};
    bool worker_ok = true, consistent = true;
    std::thread worker([&] {
        while (!release.load(std::memory_order_acquire)) std::this_thread::yield();
        for (uint64_t chunk = 0; chunk < chunks; ++chunk) worker_ok &= ledger.complete(ticket, chunk);
    });
    consistent &= !ledger.can_reserve(bytes, 0, bytes);
    release.store(true, std::memory_order_release);
    for (;;) {
        const auto status = ledger.snapshot();
        consistent &= status.charged_bytes == status.pending_bytes &&
            status.charged_bytes + status.released_chunks * plan::kKvShareChunk == bytes &&
            status.peak_pending_bytes == bytes;
        if (!status.pending_bytes) break;
    }
    worker.join();
    require(worker_ok && consistent && ledger.can_reserve(bytes, 0, bytes),
        "concurrent admission observer saw early credit or an inconsistent physical ledger");
}
void failed_worker_startup() {
    struct Counts {
        int starts = 0, destroys = 0, joins = 0;
        std::atomic<int> exits{0};
        bool destroyed_poisoned = false;
    };
    struct CpuWorker {
        Counts& counts;
        bool& poisoned;
        const int outcome; // 0: creation fails, 1: thread startup fails, 2: ready
        std::thread thread;
        CpuWorker(Counts& c, bool& p, int value) : counts(c), poisoned(p), outcome(value) {}
        ~CpuWorker() {
            if (thread.joinable()) { thread.join(); ++counts.joins; }
            counts.destroyed_poisoned = poisoned; ++counts.destroys;
        }
        bool start(std::string& error) {
            ++counts.starts;
            if (outcome) thread = std::thread([this] { ++counts.exits; });
            if (outcome == 2) return true;
            poisoned = true; error = "injected pre-job worker startup failure"; return false;
        }
    };
    for (int outcome : {0, 1, 2}) {
        Counts counts;
        bool poisoned = false;
        core::KvReclaimLedger backing;
        require(backing.mapped(3), "startup test backing charge failed");
        auto worker = std::make_unique<CpuWorker>(counts, poisoned, outcome);
        std::string error;
        const bool started = core::kv_reclaim_start_worker(worker, error);
        const bool retained = bool(worker);
        const int destroyed_at_return = counts.destroys;
        worker.reset(); worker.reset();
        require(started == (outcome == 2) && retained == started &&
            destroyed_at_return == int(!started) && counts.starts == 1 && counts.destroys == 1 &&
            counts.joins == int(outcome != 0) && counts.exits.load() == int(outcome != 0),
            "failed startup retained poisoned worker or hung/double-joined its thread");
        require(poisoned == !started && counts.destroyed_poisoned == !started &&
            (started ? error.empty() : error == "injected pre-job worker startup failure") &&
            backing.snapshot().charged_bytes == plan::kKvShareChunk &&
            backing.snapshot().pending_bytes == 0 && backing.snapshot().released_chunks == 0,
            "startup failure changed backing credit or refused safe poisoned-worker cleanup");
    }
    std::puts("KV_SHARE_WORKER_STARTUP create_failure=no_thread thread_failure=joined_once ready=retained poison_cleanup=safe release_credit=0");
}
void headroom_idle_dispatch() {
    using namespace plan;
    using namespace core;
    constexpr uint64_t gib = 1ull << 30, chunk = kKvShareChunk;
    const auto v100 = topology(0, 1), rtx = topology(2, 3);
    const KvDeviceReclaimPolicy sync{v100.device_uuid, KvReclaimMode::synchronous, 1, 1};
    const KvDeviceReclaimPolicy deferred{rtx.device_uuid, KvReclaimMode::deferred, 2, 4};
    KvReclaimLedger ledger(rtx), v100_ledger(v100);
    for (uint64_t i = 0; i < 4; ++i)
        require(ledger.mapped(i) && v100_ledger.mapped(i), "headroom mapping failed");
    const KvPipelineKey submission{{101, 1, 111}, 1};
    KvPipelineLifetime pipeline(submission, 7, {});
    KvReclaimWindow window{24 * gib, 23 * gib - 5 * chunk, chunk, gib, false, false};
    auto dispatch = [&] { return ledger.dispatch(deferred, rtx, pipeline, submission, 4, window); };
    require(dispatch().action == KvReclaimAction::wait_pipeline &&
        pipeline.local_drain(submission, 0) && pipeline.local_drain(submission, 1) &&
        pipeline.local_drain(submission, 2) && pipeline.terminal_drain(submission),
        "headroom/idle policy bypassed terminal pipeline order");
    auto decision = dispatch();
    require(decision.known && decision.admission_ready && decision.required_before == 24 * gib &&
        decision.headroom_bytes == chunk && decision.action == KvReclaimAction::hold,
        "busy window launched reclaim or double-counted charged backing/reserve");
    window.idle = true;
    decision = dispatch();
    require(decision.action == KvReclaimAction::enqueue && decision.chunks == 4 &&
        ledger.snapshot().charged_bytes == 4 * chunk, "idle window lost per-device batch or pre-credited its release");
    ++window.incoming;
    decision = dispatch();
    require(!decision.admission_ready && decision.action == KvReclaimAction::release_inline,
        "24GiB pressure was deferred despite insufficient current headroom");
    auto larger = window; larger.capacity = 32 * gib;
    const auto v100_decision = v100_ledger.dispatch(sync, v100, pipeline, submission, 4, larger);
    require(v100_decision.admission_ready && v100_decision.action == KvReclaimAction::release_inline &&
        v100_decision.chunks == 1 && !dispatch().admission_ready,
        "V100 headroom or policy rescued the independently constrained 24GiB stage");
    require(ledger.dispatch(sync, v100, pipeline, submission, 4, larger).action == KvReclaimAction::refuse,
        "foreign device dispatched using another worker's physical ledger");
    auto replacement_key = rtx; ++replacement_key.allocation_generation;
    require(ledger.dispatch(deferred, replacement_key, pipeline, submission, 4, window).action == KvReclaimAction::refuse,
        "replacement pool generation borrowed a stale worker ledger");
    const auto ticket = ledger.begin();
    for (uint64_t i = 0; i < 4; ++i) require(ledger.queue(ticket, i), "idle dispatch enqueue failed");
    decision = dispatch();
    require(!decision.admission_ready && decision.action == KvReclaimAction::wait_worker &&
        ledger.snapshot().charged_bytes == 4 * chunk && ledger.snapshot().pending_bytes == 4 * chunk,
        "pending worker debt was discounted or dispatched a second time");
    require(ledger.complete(ticket, 0), "idle worker physical completion failed");
    decision = dispatch();
    require(decision.admission_ready && decision.action == KvReclaimAction::wait_worker &&
        decision.headroom_bytes == 2 * chunk && decision.required_before == 24 * gib - chunk + 1,
        "completed bytes or pending debt counted twice in headroom");
    for (uint64_t i = 1; i < 4; ++i) require(ledger.complete(ticket, i), "idle worker completion leaked");
    require(ledger.mapped(5), "short retirement run setup failed");
    window.idle = false; window.flush = true;
    require(ledger.dispatch(deferred, rtx, pipeline, submission, 1, window).action == KvReclaimAction::release_inline &&
        ledger.dispatch(deferred, rtx, pipeline, submission, 1, window).chunks == 1,
        "explicit retirement flush stranded a short run outside an idle window");
    auto stale = submission; ++stale.submission;
    require(ledger.dispatch(deferred, rtx, pipeline, stale, 4, window).action == KvReclaimAction::wait_pipeline,
        "stale dispatcher submission borrowed another pipeline barrier");
    window.other_charged = UINT64_MAX;
    require(!dispatch().known && dispatch().action == KvReclaimAction::refuse,
        "overflow created fictitious stage headroom");
}
}
int main() {
    try {
        topology_alias(); topology_admission(); pipeline_lifetime(); stale_pipeline_reuse(); reclaim_policy();
        policy(); ownership(); cancelled_preparation(); retention(); logical_images(); borrowed_sources(); acquire_rollback();
        deferred_charge(); concurrent_reclaim_observer(); failed_worker_startup(); headroom_idle_dispatch();
    }
    catch (const std::exception& e) { std::fprintf(stderr, "kv_share_test: %s\n", e.what()); return 1; }
    std::printf("kv_share_test: topology, per-stage admission, pipeline lifetime, reclaim accounting, "
        "headroom/idle-window, default-off carve, retention, generation, COW and concurrent deferred charging passed (CPU/no-load)\n");
    return 0;
}
