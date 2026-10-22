// Opt-in strata-resident-test boundary fixture, after real full-model startup.
// The target tap VALUES and running-state markers below are synthetic. The
// initialized three-device/four-seat owners, native weights, five-tap producer,
// GPU context projections, certification and conversation transfers are real.
// This is not target-inference, proposal-quality or physical tap-UUID evidence.
#if defined(STRATA_DFLASH2_CAPTURE)
#include "strata/program/resident_slots.hpp"
#include "strata/core/dflash2_drafter.hpp"
#include "strata/core/expert_source.hpp"
#include "conversation_snapshot_equivalence.hpp"
#include <cuda_runtime.h>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <memory>
#include <future>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace strata;
constexpr int64_t native_cells = core::dflash2::context_rows + 1;
constexpr uint64_t physical_reserve = uint64_t{12} << 30;
constexpr uint64_t diagnostic_allowance = uint64_t{2} << 30;
constexpr size_t raw_values = size_t(core::dflash2::context_rows) * core::dflash2::residual_width;
constexpr size_t raw_bytes = raw_values * sizeof(float);
constexpr uint64_t compact_native_bytes = uint64_t(core::dflash2::context_rows) * 5 * 2 * 8 * 128 * 2;

// Declared AFTER the borrowed payload allocation: an exception must drain all
// consumers before that payload is freed. An undrainable device is fail-stop,
// never a recover/retry path. The native Operation also owns its stream drain.
class FixtureDrain {
public:
    explicit FixtureDrain(const std::vector<program::ResidentStage>& stages) : stages_(stages) {
        if (cudaGetDevice(&previous_) != cudaSuccess) std::terminate();
    }
    ~FixtureDrain() { if (active_) finish(); }
    void finish() noexcept {
        for (const auto& stage : stages_)
            if (cudaSetDevice(stage.device) != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess)
                std::terminate();
        if (cudaSetDevice(previous_) != cudaSuccess) std::terminate();
        active_ = false;
    }
private:
    const std::vector<program::ResidentStage>& stages_;
    int previous_ = -1;
    bool active_ = true;
};

const core::SavedConversation& stage_image(const core::SavedConversation& image, size_t stage) {
    return stage ? image.stage_images[stage - 1] : image;
}
bool same_payload(const core::SavedConversation& a, const core::SavedConversation& b) {
    if (a.stage_images.size() != b.stage_images.size()) return false;
    for (size_t i = 0; i <= a.stage_images.size(); ++i) {
        const auto& left = stage_image(a, i);
        const auto& right = stage_image(b, i);
        if (!test::same_snapshot(left, right) || left.draft_identity != right.draft_identity ||
            left.draft_origin != right.draft_origin || left.geometry != right.geometry ||
            left.layer_lo != right.layer_lo || left.layer_hi != right.layer_hi) return false;
    }
    return true;
}
// Pending inputs deliberately make native observation unavailable. Target-only
// production snapshots can still compare every actual SessionState payload;
// this does not strip or invent a certificate in an image used for restore.
bool same_target_payload(const core::SavedConversation& certified, const core::SavedConversation& target) {
    if (certified.stage_images.size() != target.stage_images.size()) return false;
    for (size_t i = 0; i <= target.stage_images.size(); ++i) {
        const auto& a = stage_image(certified, i);
        const auto& b = stage_image(target, i);
        if (a.live.ids != b.live.ids || a.live.gdn != b.live.gdn || a.live.ple != b.live.ple ||
            a.live.tails != b.live.tails || a.live.dead != b.live.dead ||
            a.live.block_pos != b.live.block_pos || a.kv.size() < b.kv.size()) return false;
        for (size_t k = 0; k < b.kv.size(); ++k)
            if (!test::equal(a.kv[k], b.kv[k])) return false;
    }
    return true;
}

// Seeded channel/position/stream patterns, not a scalar rescaling that RMS
// normalization could erase. The same bounded FP32 array is reused for all taps.
void synthetic_tap(float* out, int start, int count, int tap, uint32_t seed) {
    for (int row = 0; row < count; ++row)
        for (int stream = 0; stream < 4; ++stream)
            for (int channel = 0; channel < core::dflash2::hidden_size; ++channel) {
                uint32_t word = seed ^ (uint32_t(start + row + 1) * 0x9e3779b9u) ^
                    (uint32_t(channel + 1) * 0x85ebca6bu) ^ (uint32_t(tap + 1) * 0xc2b2ae35u) ^
                    (uint32_t(stream + 1) * 0x27d4eb2fu);
                word ^= word >> 16; word *= 0x7feb352du; word ^= word >> 15;
                out[(size_t(row) * 4 + size_t(stream)) * core::dflash2::hidden_size + size_t(channel)] =
                    float(int(word & 1023u) - 511) / 1024.0f;
            }
}
} // namespace

int resident_native_model_checks(strata::program::ResidentSlots& owner, strata::core::ExpertDispatch& dispatch,
        const strata::core::ModelGeometry& geometry, const std::vector<strata::program::ResidentStage>& stages,
        strata::core::dflash2::DflashNativeDrafter& draft) {
    using namespace strata;
    std::string error;
    int checks = 0, identity_refusals = 0;
    auto require = [&](bool ok, const char* message) {
        ++checks;
        if (!ok) throw std::runtime_error(std::string(message) + ": " + error);
    };
    auto device = [&](int ordinal) {
        require(cudaSetDevice(ordinal) == cudaSuccess, "select actual state-owning device");
        require(cudaDeviceSynchronize() == cudaSuccess, "drain actual state-owning device");
    };
    try {
        require(stages.size() == 3 && owner.slots().size() == 4 && owner.available(0) && owner.available(1) &&
                geometry.n_layers == 48 && geometry.n_embd == core::dflash2::hidden_size && geometry.hc == 4 &&
                draft.kind() == core::DraftKind::dflash2 && draft.identity().bound() &&
                draft.identity().load_generation && draft.legacy_state() == nullptr &&
                draft.device() == stages.back().device,
                "native fixture requires the initialized full target and native drafter, not MTP storage");
        for (size_t i = 0; i < stages.size(); ++i) {
            const auto& stage = stages[i];
            require(stage.main && stage.verifier && stage.batch_peer && stage.slots.size() == 4 &&
                    stage.main->max_cells >= native_cells + 2 && stage.main->allocation_device == stage.device,
                    "three complete stages retain four seats and both replay owners");
            for (const auto* session : stage.slots)
                require(session && session->allocation_device == stage.device && session->max_cells >= native_cells + 2,
                        "each actual physical seat admits the chronological cut");
            for (size_t j = 0; j < i; ++j)
                require(stage.device != stages[j].device, "all three target stages use distinct actual devices");
        }
        const std::vector<int32_t> priced_ids(size_t(native_cells), 11);
        const std::vector<core::ConversationImageKey> images;
        const std::vector<core::ConversationCheckpoint> checkpoints;
        const core::ConversationView priced_view{priced_ids, images, checkpoints, true};
        uint64_t target_image_bound = 0;
        for (const auto& stage : stages) {
            size_t bytes = 0;
            require(core::conversation_snapshot_bytes(priced_view, *stage.slots[0], geometry, nullptr, bytes, error) &&
                    bytes && bytes <= diagnostic_allowance - target_image_bound,
                    "price the real three-stage target formats before diagnostic payload allocation");
            target_image_bound += bytes;
        }
        const auto allocation = draft.allocation_plan();
        constexpr uint64_t control_bound = uint64_t{8} << 20;
        require(allocation.snapshot_host >= compact_native_bytes &&
                allocation.snapshot_host < diagnostic_allowance - target_image_bound &&
                target_image_bound + allocation.snapshot_host <=
                    (diagnostic_allowance - raw_bytes - control_bound) / 5,
                "five bounded diagnostic images plus the single tap payload fit the 2 GiB cap");
        // Original, peer, untouched ambient, absent-source and one comparison
        // image may coexist. Refusal aliases share frozen payloads; the control
        // bound covers their token/directory copies. Native readback workspace
        // is already allocated and charged by the production native owner.
        const uint64_t diagnostic_bound = raw_bytes + control_bound +
            5 * (target_image_bound + allocation.snapshot_host);
        uint64_t available = 0, physical = 0;
        require(core::available_memory_bytes(available, nullptr, &physical) && available >= diagnostic_bound &&
                physical >= physical_reserve && diagnostic_bound <= physical - physical_reserve,
                "actual diagnostic demand preserves the unchanged 12 GiB physical reserve");
        std::printf("DFLASH2_NATIVE_LIFECYCLE_ALLOCATION raw_host_bytes=%zu target_image_bound=%llu "
                    "native_image_bound=%llu diagnostic_bound=%llu allowance=%llu reserve=%llu fixture_device_payload_bytes=0\n",
                    raw_bytes, (unsigned long long)target_image_bound,
                    (unsigned long long)allocation.snapshot_host, (unsigned long long)diagnostic_bound,
                    (unsigned long long)diagnostic_allowance, (unsigned long long)physical_reserve);
        auto raw = std::make_unique_for_overwrite<float[]>(raw_values);
        FixtureDrain drain(stages);

        // Only this helper's last-stage path includes the native adapter. The
        // other stages use the established target-only conversation pipeline.
        auto capture = [&](int slot, const std::vector<int32_t>& ids, bool native) {
            core::SavedConversation result;
            for (size_t i = 0; i < stages.size(); ++i) {
                auto& part = i ? result.stage_images[i - 1] : result;
                const auto& session = slot < 0 ? *stages[i].main : *stages[i].slots[size_t(slot)];
                const core::ConversationView view{ids, images, checkpoints, true};
                device(stages[i].device);
                require(native && i + 1 == stages.size()
                    ? core::conversation_neural_save(part, view, session, geometry, draft, slot < 0 ? 8 : slot, error)
                    : core::conversation_snapshot_save(part, view, session, geometry, nullptr, error),
                    "capture actual target state and optional production-certified native context");
                if (i == 0) result.stage_images.resize(stages.size() - 1);
            }
            if (slot >= 0) result.scope = owner.slots()[size_t(slot)].input.scope;
            result.freeze();
            return result;
        };
        const auto ambient = capture(-1, priced_ids, false);
        const auto ambient_request = dispatch.request_generation;
        const auto ambient_working = owner.working().request;
        const std::vector<int64_t> prompt(size_t(native_cells + 1), 11);
        const std::vector<int64_t> peer_prompt{22, 33};
        program::ResidentRequest request, peer_request;
        request.max_new = peer_request.max_new = 1;
        request.scope.value[0] = 111; peer_request.scope.value[0] = 112;
        request.sampling.seed = 17; peer_request.sampling.seed = 23;
        request.sampling.greedy = peer_request.sampling.greedy = true;
        request.sampling.temperature = peer_request.sampling.temperature = 0;
        program::ResidentPreparation source, peer;
        require(owner.prepare(0, prompt, request, source, error) && source.position == 0 &&
                owner.prepare(1, peer_prompt, peer_request, peer, error) && peer.position == 0,
                "prepare actual independent native resident owners without an ambient-state handoff");

        auto ingest = [&](int slot, int start, int count, uint32_t seed, int64_t stale_generation = 0) {
            const auto expected = draft.owner(slot);
            core::DraftOwner ticket;
            require(draft.reserve_inputs(slot, expected, start, count, ticket, error),
                    "reserve the real target-feature interval on its actual native owner");
            core::NeuralDrafter::Operation operation(draft, error, slot, nullptr, 0, start, &ticket);
            require(bool(operation) && draft.begin_capture(operation, core::dflash2::CapturePhase::prefill,
                                                           start, count, error),
                    "begin the actual lease-bound capture producer");
            auto& producer = draft.capture_producer(slot);
            for (size_t tap = 0; tap < core::dflash2::tap_layers.size(); ++tap) {
                synthetic_tap(raw.get(), start, count, int(tap), seed);
                const int layer = core::dflash2::tap_layers[tap];
                core::dflash2::TapView view{start, count, count, layer, layer,
                    core::dflash2::CapturePhase::prefill,
                    core::dflash2::CapturePoint::completed_second_residual_write,
                    {raw.get(), size_t(count) * core::dflash2::residual_width}, producer.owner()};
                require(producer.consume(view, error, &operation.guard()),
                        "consume each honestly synthetic, host-complete tap through the real producer");
            }
            require(producer.publish(error), "publish only after all five matching tap completions");
            const auto features = producer.store().committed_since(start);
            if (stale_generation) {
                require(count == 1 && features.generations[0].size() == 1 && features.generations[1].empty(),
                        "same-owner replay probe is one real contiguous published feature row");
                auto stale = features;
                const std::array<int64_t, 1> stale_tags{stale_generation};
                stale.generations[0] = stale_tags;
                require(!draft.ingest_committed(operation, stale, error) &&
                        error == "dflash2 drafter: stale feature generation",
                        "same-incarnation preparation preserves the consumed-generation replay floor");
                error.clear();
            }
            require(features.start == start && features.count == count &&
                    draft.ingest_committed(operation, features, error) && operation.finish(error),
                    "project the actual committed BF16 feature ring on GPU and drain before certification");
            require(draft.inputs_completed(slot, start, count) && !draft.inputs_pending(slot),
                    "only the production completion owner certifies drained feature projection");
        };
        auto mark_target = [&](int slot, float marker) {
            for (size_t i = 0; i < stages.size(); ++i) {
                auto& session = *stages[i].slots[size_t(slot)];
                core::ConversationStateSizes sizes;
                require(core::conversation_session_sizes(geometry, session, sizes, error) && sizes.gdn >= sizeof(float),
                        "synthetic running-state marker has an actual allocated GDN destination");
                const float value = marker + float(i) / 8;
                device(stages[i].device);
                require(cudaMemcpy(session.gdn_state, &value, sizeof(value), cudaMemcpyHostToDevice) == cudaSuccess,
                        "write a finite diagnostic marker to the existing target allocation");
            }
        };
        auto full_context = [&](uint32_t seed, float marker) {
            mark_target(0, marker);
            ingest(0, 0, core::dflash2::context_rows, seed);
            ingest(0, core::dflash2::context_rows, 1, seed);
            require(owner.prefilled(source, native_cells, error),
                    "publish the synthetic cut with the actual native N-cell completion, not MTP N-1");
        };
        ingest(1, 0, 1, 0xa817c3u);
        mark_target(1, 0.75f);
        require(owner.prefilled(peer, 1, error), "publish the independent peer's real native certificate");
        const auto peer_image = capture(1, owner.slots()[1].ids, true);
        const auto peer_identity = owner.identity(1);
        full_context(0x1935bdu, 0.25f);
        {
            const auto prepared_source = owner.prepared_snapshot_source(source, error);
            require(prepared_source.has_value() && prepared_source->draft_cells == native_cells &&
                    prepared_source->draft_begin == 1 && prepared_source->draft == nullptr,
                    "real common resident source exposes native N cells without a fabricated legacy QSA or R");
        }
        const auto saved = capture(0, owner.slots()[0].ids, true);
        const auto& native = stage_image(saved, stages.size() - 1);
        require(native.draft_cells == native_cells && native.draft_begin == 1 && native.draft_origin == 0 &&
                draft.origin(0) == 0 && draft.capture_producer(0).committed_rows() == native_cells &&
                native.draft_next_token == -1 && native.draft_boundary_residual.empty() &&
                native.kv.size() == size_t(stages.back().slots[0]->qsa_alloc) + 5,
                "chronological ingestion naturally evicts row zero; retained begin is not a forged reset origin");
        const size_t native_first = native.kv.size() - 5;
        for (size_t i = native_first; i < native.kv.size(); ++i) {
            const auto& kv = native.kv[i];
            require(kv.format == 0 && kv.cells == native_cells && kv.heads == 8 && kv.head_dim == 128 &&
                    kv.page_size == 1 && !kv.pooled_rows && !kv.idx_dim &&
                    kv.k.size() == compact_native_bytes / 10 && kv.v.size() == compact_native_bytes / 10 &&
                    kv.k_scale.empty() && kv.v_scale.empty() && kv.pooled.empty(),
                    "save five compact chronological BF16 native K/Vs, not target-shaped MTP storage");
        }
        uint64_t original_hash = 0, changed_hash = 0, restored_hash = 0;
        require(draft.state_fingerprint(0, native_cells, original_hash, error),
                "fingerprint the actual projected native payload");
        const auto before_reset = draft.owner(0);
        require(draft.reset_owner(0, before_reset, 0, error), "reset the same real native owner before changing its payload");
        full_context(0xb49e07u, -0.5f);
        require(draft.state_fingerprint(0, native_cells, changed_hash, error) && changed_hash != original_hash,
                "a non-collinear synthetic pattern changes actual projected native bytes");
        {
            const auto changed = capture(0, owner.slots()[0].ids, true);
            for (size_t i = 0; i < stages.size(); ++i)
                require(stage_image(changed, i).live.gdn != stage_image(saved, i).live.gdn,
                        "each actual target stage is different before restore, not just its image metadata");
        }
        auto restore_native = [&](const core::SavedConversation& image, int slot) {
            const auto& state = owner.slots()[size_t(slot)];
            const auto expected = draft.owner(slot);
            require(core::conversation_neural_restore(image, *stages.back().slots[size_t(slot)], geometry,
                        draft, slot, int64_t(state.prompt.size()), state.prompt[size_t(state.p)],
                        core::ConversationSnapshotExpectation::ordinary, error, nullptr, &expected) ==
                        core::ConversationRestore::restored,
                    "restore and certify through the actual production common native adapter");
        };
        // Whole-source validation precedes every target-stage write, as in the
        // production resident handoff. No replacement lifecycle machine exists.
        for (size_t i = 0; i < stages.size(); ++i) {
            const auto& part = stage_image(saved, i);
            require(i + 1 == stages.size()
                ? core::conversation_neural_validate(part, *stages[i].slots[0], geometry, draft, 0,
                    core::ConversationSnapshotExpectation::ordinary, error)
                : core::conversation_snapshot_validate(part, *stages[i].slots[0], geometry, nullptr, error),
                "prevalidate the entire real source before restoring any target stage");
        }
        for (size_t i = 0; i + 1 < stages.size(); ++i) {
            device(stages[i].device);
            require(core::conversation_snapshot_restore(stage_image(saved, i), *stages[i].slots[0], geometry,
                        nullptr, error) == core::ConversationRestore::restored,
                    "restore earlier actual target stages with the existing conversation workspace");
        }
        restore_native(native, 0);
        require(draft.state_fingerprint(0, native_cells, restored_hash, error) && restored_hash == original_hash &&
                same_payload(saved, capture(0, owner.slots()[0].ids, true)),
                "restore exactly the original actual native fingerprint and every target semantic payload");

        auto unchanged = [&] {
            require(owner.owns(0, source.request, source.incarnation) &&
                    owner.owns(1, peer_identity.request, peer_identity.incarnation) &&
                    owner.slots()[0].preparing && owner.slots()[1].preparing &&
                    !owner.slots()[0].failed && !owner.slots()[1].failed &&
                    owner.slots()[0].p == native_cells && owner.slots()[1].p == 1 &&
                    dispatch.slot_requests[0] == source.request && dispatch.slot_requests[1] == peer.request &&
                    dispatch.request_generation == ambient_request && owner.working().request == ambient_working,
                    "refusal preserves real destination, peer and ambient ownership");
            require(same_payload(saved, capture(0, owner.slots()[0].ids, true)),
                    "refusal changes no actual target/native destination payload");
            require(same_payload(peer_image, capture(1, owner.slots()[1].ids, true)),
                    "refusal changes no actual independent peer payload");
        };

        // A real unconsumed producer/ticket, not an invented live-operation bit.
        const auto pending_before = draft.owner(0);
        const auto committed_before = draft.capture_producer(0).committed_rows();
        core::DraftOwner ticket;
        require(draft.reserve_inputs(0, pending_before, native_cells, 1, ticket, error) &&
                draft.capture_producer(0).begin(core::dflash2::CapturePhase::prefill, native_cells, 1, error),
                "retain an actual pending target-feature ticket before feature commitment");
        const auto capture_owner = draft.capture_producer(0).owner();
        auto refused_output = native.frozen_alias();
        const core::ConversationView pending_view{owner.slots()[0].ids, images, checkpoints, true};
        require(!draft.context_idle(error) &&
                !core::conversation_neural_save(refused_output, pending_view, *stages.back().slots[0], geometry,
                                                draft, 0, error) &&
                same_payload(native, refused_output) && draft.owner(0) == ticket && draft.inputs_pending(0) &&
                draft.capture_producer(0).owner() == capture_owner &&
                draft.capture_producer(0).frontier() == native_cells &&
                draft.capture_producer(0).committed_rows() == committed_before,
                "pending ticket refuses common snapshot/context-idle before any capture publication");
        require(same_target_payload(saved, capture(0, owner.slots()[0].ids, false)) &&
                same_target_payload(peer_image, capture(1, owner.slots()[1].ids, false)),
                "pending snapshot refusal writes neither destination nor peer target state");
        // A healthy reader projection owns the same public operation lease.
        // Cancellation must wait for its drained final reads, not fail merely
        // because that independent peer currently owns scratch.
        std::future<bool> discarded;
        std::promise<void> discard_started;
        std::string discard_error;
        const auto reader_owner = draft.owner(1);
        {
            core::NeuralDrafter::Operation reader(draft, error, 1, nullptr, 0, 1,
                                                 &reader_owner, nullptr, true);
            require(bool(reader), "healthy independent reader acquires the real shared native lease");
            discarded = std::async(std::launch::async, [&] {
                discard_started.set_value();
                return draft.discard_inputs(ticket, discard_error);
            });
            discard_started.get_future().wait();
            require(discarded.wait_for(std::chrono::milliseconds(50)) == std::future_status::timeout,
                    "drained cancellation waits while the healthy reader still owns its final scratch reads");
            require(reader.finish(error), "complete and drain the actual independent read-only operation");
        }
        require(discarded.get() && !draft.inputs_pending(0) &&
                !draft.capture_producer(0).active() && draft.context_idle(error) &&
                draft.owner(1) == reader_owner,
                "drained cancellation completes after bounded handoff without harming peer ownership");
        restore_native(native, 0); // discard intentionally invalidates certification
        unchanged();

        // Produce an absent-draft SOURCE through the real API. No image fields,
        // certificate owners or coverage intervals are cleared by the fixture.
        draft.invalidate_coverage();
        const auto absent = capture(0, owner.slots()[0].ids, true);
        const auto& absent_native = stage_image(absent, stages.size() - 1);
        require(absent_native.draft_identity.kind == core::DraftKind::none && absent_native.draft_cells < 0 &&
                absent_native.draft_begin < 0 && absent_native.draft_boundary_residual.empty() &&
                absent_native.kv.size() == size_t(stages.back().slots[0]->qsa_alloc),
                "actual producer invalidation yields a genuinely absent native source on production capture");
        restore_native(native, 0);
        restore_native(stage_image(peer_image, stages.size() - 1), 1);
        auto refuse = [&](const core::SavedConversation& image) {
            require(core::conversation_neural_validate(native, *stages.back().slots[0], geometry, draft, 0,
                        core::ConversationSnapshotExpectation::ordinary, error),
                    "the unchanged positive image is valid on this otherwise healthy destination");
            const auto expected = draft.owner(0);
            error.clear();
            require(!core::conversation_neural_validate(image, *stages.back().slots[0], geometry, draft, 0,
                        core::ConversationSnapshotExpectation::ordinary, error) &&
                    core::conversation_neural_restore(image, *stages.back().slots[0], geometry, draft, 0,
                        int64_t(prompt.size()), prompt.back(), core::ConversationSnapshotExpectation::ordinary,
                        error, nullptr, &expected) == core::ConversationRestore::invalid &&
                    draft.owner(0) == expected,
                    "actual common preflight refuses before target/native writes or owner generation changes");
            unchanged();
        };
        refuse(absent_native);
        for (int negative = 0; negative < 9; ++negative) {
            auto foreign = native.frozen_alias();
            switch (negative) {
            case 0: foreign.draft_identity.kind = core::DraftKind::mtp; break;
            case 1: ++foreign.draft_identity.format; break;
            case 2: foreign.draft_identity.configuration[0] ^= 1; break;
            case 3: ++foreign.draft_identity.geometry[0]; break;
            case 4: foreign.draft_identity.weights[0] ^= 1; break;
            case 5: foreign.draft_identity.borrowed_target[0] ^= 1; break;
            case 6: foreign.draft_identity.load_generation = native.draft_identity.load_generation == 1 ? 2 : 1; break;
            case 7: ++foreign.geometry[0]; break;
            case 8: ++foreign.kv[native_first].format; break;
            }
            refuse(foreign);
            ++identity_refusals;
        }
        require(same_payload(ambient, capture(-1, priced_ids, false)),
                "all native transitions preserve the actual ambient target sessions on three GPUs");

        // A prepared seat cannot become a retained prefix by faking first_token:
        // that API requires an actual final verifier row. Instead use the
        // existing public working -> parked-owner path and actual common restore.
        require(owner.begin(2, prompt, request, error), "bind a real working request for the retained-resume boundary");
        for (size_t i = 0; i < stages.size(); ++i) {
            const auto& part = stage_image(saved, i);
            require(i + 1 == stages.size()
                ? core::conversation_neural_validate(part, *stages[i].main, geometry, draft, 8,
                    core::ConversationSnapshotExpectation::ordinary, error)
                : core::conversation_snapshot_validate(part, *stages[i].main, geometry, nullptr, error),
                "validate the actual working destinations before any parked-prefix setup writes");
        }
        for (size_t i = 0; i + 1 < stages.size(); ++i) {
            device(stages[i].device);
            require(core::conversation_snapshot_restore(stage_image(saved, i), *stages[i].main, geometry,
                        nullptr, error) == core::ConversationRestore::restored,
                    "restore actual working target payload for the public parked-owner route");
        }
        const auto working_destination = draft.owner(8);
        require(core::conversation_neural_restore(native, *stages.back().main, geometry, draft, 8,
                    int64_t(prompt.size()), prompt.back(), core::ConversationSnapshotExpectation::ordinary,
                    error, nullptr, &working_destination) == core::ConversationRestore::restored,
                "restore the real working native context without forging a verifier completion");
        const program::ResidentAdmission retained_admission{int32_t(prompt.back())};
        require(owner.park(2, priced_ids, retained_admission, true, error) &&
                owner.begin(2, prompt, request, error) && owner.working().resume_slot == 2,
                "park and reselect the exact real native retained continuation");
        program::PrefixWork positive_work;
        require(owner.resume_work(2, nullptr, positive_work, error) && positive_work.bytes > 0,
                "positive public resume_work preflights an otherwise valid native retained source");
        const auto retained_identity = owner.identity(2);
        const auto retained_draft = draft.owner(2), working_draft = draft.owner(8);
        uint64_t retained_hash = 0;
        require(draft.state_fingerprint(2, native_cells, retained_hash, error) && retained_hash == original_hash,
                "the actual parked native owner holds the same compact semantic payload");
        owner.inject_stale_continuation();
        program::PrefixWork refused_work;
        require(!owner.resume_work(2, nullptr, refused_work, error) && refused_work.bytes == 0 &&
                owner.owns(2, retained_identity.request, retained_identity.incarnation) &&
                draft.owner(2) == retained_draft && draft.owner(8) == working_draft &&
                owner.resume_work(2, nullptr, positive_work, error),
                "actual public resume_work refuses the stale generation before mutation, then accepts the valid source");
        // End each comparison's full expression before allocating the next
        // diagnostic image: at most the quoted fifth image may coexist.
        require(same_payload(saved, capture(2, owner.slots()[2].ids, true)),
                "stale resume_work preserves the actual retained target/native payload");
        require(same_payload(saved, capture(-1, priced_ids, true)),
                "stale resume_work preserves the actual working target/native destination");
        require(same_payload(saved, capture(0, owner.slots()[0].ids, true)),
                "stale resume_work preserves the independent prepared source payload");
        require(same_payload(peer_image, capture(1, owner.slots()[1].ids, true)) &&
                owner.owns(0, source.request, source.incarnation) &&
                owner.owns(1, peer_identity.request, peer_identity.incarnation),
                "stale resume_work preserves both actual peer payload and peer ownership");
        owner.cancel(2);

        // The legacy GEN continuation rebinds working seat 8 without a snapshot
        // transfer. Exercise the actual producer and GPU ingest after that seam.
        const std::vector<int64_t> direct_prompt(6, 11);
        require(owner.begin(-1, direct_prompt, request, error) &&
                draft.prepare_slot(8, owner.working().request, direct_prompt.size(), 0, true, error),
                "prepare a fresh actual working owner for direct retained continuation");
        ingest(8, 0, 1, 0x71a3u);
        ingest(8, 1, 1, 0x71a3u);
        const auto previous_capture_owner = draft.capture_producer(8).owner();
        require(previous_capture_owner.generation == 2 &&
                owner.begin(-1, direct_prompt, request, error) &&
                draft.owner(8).incarnation > previous_capture_owner.incarnation &&
                draft.prepare_slot(8, owner.working().request, direct_prompt.size(), 2, false, error) &&
                draft.capture_producer(8).owner().generation == 0,
                "verified newer working incarnation restarts its capture domain over retained context");
        ingest(8, 2, 1, 0x71a3u);
        const auto resumed_capture_owner = draft.capture_producer(8).owner();
        require(resumed_capture_owner.generation == 1 &&
                draft.prepare_slot(8, owner.working().request, direct_prompt.size(), 3, false, error) &&
                draft.capture_producer(8).owner() == resumed_capture_owner,
                "same-incarnation direct preparation preserves the existing capture domain");
        ingest(8, 3, 1, 0x71a3u, 1);
        require(draft.capture_producer(8).owner().generation == 2 &&
                same_payload(peer_image, capture(1, owner.slots()[1].ids, true)),
                "fresh continuation succeeds after stale replay refusal and preserves the peer");
        require(owner.abort_preparation(source, error) && owner.abort_preparation(peer, error) &&
                draft.idle(error) && owner.release_resources(error),
                "retire both real prepared owners and release only drained resident resources");
        drain.finish();
        std::printf("DFLASH2_NATIVE_LIFECYCLE source_origin=%lld retained_begin=%lld certified_end=%lld "
                    "certified_cells=%lld retained_rows=%d restored=1 absent_refused=1 peer_unchanged=1 "
                    "identity_refusals=%d pending_refused=1 stale_resume_work_refused=1 stages=3 seats=4 "
                    "retained_generation_rebind=1 same_owner_replay_refused=1 "
                    "synthetic_host_taps=1 native_gpu_projection=1 target_inference=0 checks=%d failures=0\n",
                    (long long)native.draft_origin, (long long)native.draft_begin, (long long)native.draft_cells,
                    (long long)native.draft_cells,
                    core::dflash2::context_rows, identity_refusals, checks);
        return 0;
    } catch (const std::exception& exception) {
        std::fprintf(stderr, "DFLASH2_NATIVE_LIFECYCLE checks=%d failures=1 error=%s\n", checks, exception.what());
        return 1;
    }
}
#endif
