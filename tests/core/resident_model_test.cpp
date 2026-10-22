// Runs only in strata-resident-test after the normal protected full-model startup.
// No fake CUDA success: all transfers forward to the real runtime; one injected
// transfer failure proves that partial state is unavailable, not GPU recovery.
#include "strata/program/resident_slots.hpp"
#include "strata/program/session_fingerprint.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/graph.hpp"
#include "strata/core/mtp.hpp"
#include "strata/core/verify.hpp"
#include "strata/core/resident_memory.hpp"
#include "conversation_snapshot_equivalence.hpp"
#include "strata/kernels/qsa.hpp"
#include <cuda_runtime.h>
#include <array>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {
using strata::test::same_snapshot;
void* fail_destination = nullptr;
bool transfer_failed = false;

// GNU/ELF wrapping of ordinary C++ allocations made by the statically linked
// engine: vectors, segmented images and checkpoint payloads. No interception of
// CUDA/driver C allocations or RSS claim; those keep their opaque allowance.
// The fixed thread-local directory cannot allocate or perturb measured peaks.
// Resident transfer payloads use ordinary vector new/delete, not aligned new.
struct HostTransferScratch {
    struct Entry { void* pointer; size_t bytes; };
    Entry entries[16384]{};
    uint64_t live = 0, peak = 0;
    bool active = false, overflow = false;
    void begin() {
        for (auto& entry : entries) entry = {};
        live = peak = 0;
        overflow = false;
        active = true;
    }
    void allocated(void* pointer, size_t bytes) {
        if (!active) return;
        if (bytes > UINT64_MAX - live) { overflow = true; return; }
        for (auto& entry : entries) if (!entry.pointer) {
            entry = {pointer, bytes};
            live += bytes;
            peak = std::max(peak, live);
            return;
        }
        overflow = true;
    }
    void released(void* pointer) {
        if (!active || !pointer) return;
        for (auto& entry : entries) if (entry.pointer == pointer) {
            live -= entry.bytes;
            entry = {};
            return;
        }
    }
};
thread_local HostTransferScratch host_transfer_scratch;

}
extern "C" cudaError_t __real_cudaMemcpy(void*, const void*, size_t, cudaMemcpyKind);
extern "C" void* __real__Znwm(size_t);
extern "C" void __real__ZdlPv(void*);
extern "C" void __real__ZdlPvm(void*, size_t);
extern "C" void* __wrap__Znwm(size_t bytes) {
    void* pointer = __real__Znwm(bytes);
    host_transfer_scratch.allocated(pointer, bytes);
    return pointer;
}
extern "C" void __wrap__ZdlPv(void* pointer) {
    host_transfer_scratch.released(pointer);
    __real__ZdlPv(pointer);
}
extern "C" void __wrap__ZdlPvm(void* pointer, size_t bytes) {
    host_transfer_scratch.released(pointer);
    __real__ZdlPvm(pointer, bytes);
}
extern "C" cudaError_t __wrap_cudaMemcpy(void* dst, const void* src, size_t size, cudaMemcpyKind kind) {
    if (dst == fail_destination && size) {
        fail_destination = nullptr;
        transfer_failed = true;
        return cudaErrorInvalidValue;
    }
    return __real_cudaMemcpy(dst, src, size, kind);
}

// Snapshot transfers use a private stream. Forward async calls to the real
// runtime too, so the existing partial-write fixture reaches today's copy path.
extern "C" cudaError_t __real_cudaMemcpyAsync(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t);
extern "C" cudaError_t __wrap_cudaMemcpyAsync(void* dst, const void* src, size_t size,
                                               cudaMemcpyKind kind, cudaStream_t stream) {
    if (dst == fail_destination && size) {
        fail_destination = nullptr;
        transfer_failed = true;
        return cudaErrorInvalidValue;
    }
    return __real_cudaMemcpyAsync(dst, src, size, kind, stream);
}
namespace {
// Real ResidentSlots and real transfers, but no token arithmetic. Keep the
// terminal injected-copy failure in this opt-in vehicle, outside the normal
// lifecycle fixture, so its poisoned working owner is never reused.
int prefill_split_owner_checks(strata::program::ResidentSlots& owner, strata::core::SessionState& main) {
    using namespace strata;
    std::string error;
    int checks = 0;
    auto require = [&](bool ok, const char* message) {
        ++checks;
        if (!ok) throw std::runtime_error(std::string(message) + ": " + error);
    };
    try {
        require(main.max_cells >= 4096 && owner.slots().size() >= 2 && owner.available(0),
                "split-owner fixture needs admitted context >=4096 and initialized resident slots");
        const std::vector<int64_t> prompt(2049, 11);
        const std::vector<int32_t> prefix(1024, 11);
        program::ResidentRequest request;
        request.max_new = 2;
        request.scope.value[0] = 71;
        request.sampling.seed = 17;
        program::ResidentAdmission half;
        half.prefill_split_until = 2048;
        require(owner.begin(0, prompt, request, error), "begin actual split-prefill request");
        const auto first_id = owner.working().request;
        require(owner.park(0, prefix, half, true, error), "park actual midpoint with a pending endpoint");
        require(owner.slots()[0].request == first_id && owner.slots()[0].partial &&
                owner.slots()[0].prefill_split_until == 2048,
                "production parking publishes the nonzero endpoint on its actual resident owner");
        require(owner.begin(0, prompt, request, error) && owner.working().request == first_id &&
                owner.working().resume_slot == 0 && owner.resume(0, nullptr, error),
                "restore the exact continuing request through ResidentSlots");
        require(owner.slots()[0].prefill_split_until == 0 && owner.working().prefill_split_until == 2048 &&
                owner.take_prefill_split_until() == 2048 && owner.take_prefill_split_until() == 0,
                "production restore transfers the original endpoint exactly once");
        require(owner.park(0, prefix, half, true, error) && owner.slots()[0].prefill_split_until == 2048,
                "repark the continuing owner with a real pending half");
        owner.cancel(0);
        require(!owner.slots()[0].partial && owner.slots()[0].prefill_split_until == 0,
                "production cancellation clears a previously nonzero endpoint");
        require(owner.begin(0, prompt, request, error) && owner.working().request != first_id &&
                owner.working().resume_slot == -1 && owner.working().prefill_split_until == 0,
                "a cancelled continuation cannot transfer its endpoint to a successor");
        const auto cancelled_successor = owner.working().request;
        require(owner.park(0, prefix, half, true, error) && owner.slots()[0].prefill_split_until == 2048,
                "publish a nonzero endpoint before actual identity replacement");
        auto replacement = request;
        replacement.sampling.seed = 23;
        require(owner.begin(0, prompt, replacement, error) &&
                owner.working().request != cancelled_successor && owner.working().resume_slot == -1,
                "changed request policy creates the replacement identity");
        const auto replacement_id = owner.working().request;
        const program::ResidentAdmission unsplit;
        require(owner.park(0, prefix, unsplit, true, error) &&
                owner.slots()[0].request == replacement_id && owner.slots()[0].prefill_split_until == 0,
                "production replacement clears the prior owner's pending endpoint");
        require(owner.begin(0, prompt, replacement, error) && owner.resume(0, nullptr, error) &&
                owner.take_prefill_split_until() == 0 &&
                owner.park(0, prefix, half, true, error) && owner.slots()[0].prefill_split_until == 2048,
                "prepare a real nonzero parked endpoint for partial-restore failure");
        require(owner.begin(0, prompt, replacement, error) && owner.working().request == replacement_id &&
                owner.working().resume_slot == 0 && main.ple_hist,
                "select the exact owner and actual PLE destination for restore refusal");
        transfer_failed = false;
        fail_destination = main.ple_hist;
        require(!owner.resume(0, nullptr, error) && transfer_failed,
                "actual partial restore reaches the injected copy refusal");
        require(owner.slots()[0].failed && !owner.available(0) &&
                owner.slots()[0].prefill_split_until == 0 && owner.working().prefill_split_until == 0 &&
                owner.take_prefill_split_until() == 0,
                "production restore failure clears both pending-endpoint owners");
        require(owner.release_resources(error), "release actual resident resources after the refusal");
        std::printf("PREFILL_SPLIT_OWNER_CHECKS checks=%d prefix=1024 endpoint=2048 actual_owner=1 "
                    "copy_refusal=1 failures=0\n", checks);
        return 0;
    } catch (const std::exception& exception) {
        fail_destination = nullptr;
        std::fprintf(stderr, "PREFILL_SPLIT_OWNER_CHECKS checks=%d failures=1 error=%s\n", checks, exception.what());
        return 1;
    }
}

// Opt-in lifecycle vehicle over the real resident allocation and verifier rows.
// Prefix rows use the verifier only to supply genuine final residuals to the
// uncaptured destination MTP front; packed arithmetic is exercised by BGENQ.
int packed_resident_checks(strata::program::ResidentSlots& owner, strata::core::ExpertDispatch& dispatch,
        const strata::core::ModelGeometry& geometry, strata::core::SessionState& main,
        strata::core::SessionState& a, strata::core::SessionState& b, strata::core::MtpDrafter& draft,
        strata::core::Verifier& verifier, strata::core::PoolMultiFn pool, void* pool_user) {
    using namespace strata;
    std::string error;
    int checks = 0;
    auto require = [&](bool ok, const char* message) {
        ++checks;
        if (!ok) throw std::runtime_error(std::string(message) + ": " + error);
    };
    auto fingerprint = [&](core::SessionState& session, const core::QsaState& kv, int64_t cells) {
        program::SessionFingerprint result;
        require(program::session_fingerprint(session, kv, geometry, cells, result, error),
                "read actual owner payload");
        return result;
    };
    const auto same = [](const auto& left, const auto& right) {
        return left.gdn == right.gdn && left.ple == right.ple && left.kv == right.kv && left.draft == right.draft;
    };
    try {
        const char* enabled = std::getenv("STRATA_PACKED_PREFILL");
        require(enabled && std::strcmp(enabled, "1") == 0 && owner.slots().size() >= 2 &&
                owner.available(0) && owner.available(1) && draft.slot_state(0) && draft.slot_state(1),
                "direct preparation fixture needs exact packed opt-in and initialized real owners");
        const auto ambient = fingerprint(main, draft.kv_state(), 2);
        const auto untouched_a = fingerprint(a, *draft.slot_state(0), 2);
        const auto untouched_b = fingerprint(b, *draft.slot_state(1), 2);
        const auto working_request = owner.working().request;
        const auto ambient_request = dispatch.request_generation;
        const std::vector<int64_t> tokens_a{11, 22, 33}, tokens_b{44};
        program::ResidentRequest request_a, request_b;
        request_a.max_new = 1; request_a.scope.value[0] = 91; request_a.pin = 1;
        request_a.shared_scope.value[0] = 93;
        request_b.max_new = 2; request_b.scope.value[0] = 92;
        request_a.sampling.greedy = request_b.sampling.greedy = true;
        request_a.sampling.temperature = request_b.sampling.temperature = 0;
        const program::ResidentPrepareInput inputs[]{{0, &tokens_a, request_a}, {1, &tokens_b, request_b}};
        program::ResidentPreparation leases[2];
        const program::ResidentPrepareInput duplicate[]{{0, &tokens_a, request_a}, {0, &tokens_b, request_b}};
        require(!owner.reserve(duplicate, 2, leases, error) && !owner.slots()[0].request &&
                !owner.slots()[1].request, "duplicate atomic reservation leaves both owners unpublished");
        const core::ExpertOperation operation{71, 0, 0, 1, 1};
        auto& ledger = dispatch.completion.at(0);
        require(ledger.begin(operation, 0) && ledger.retain_submission(operation), "retain actual operation lease");
        require(!owner.reserve(inputs, 2, leases, error), "reservation refuses launched consumers");
        ledger.cancel();
        require(!owner.reserve(inputs, 2, leases, error) && !owner.slots()[0].request,
                "cancelling a consumer is not a reservation drain");
        require(ledger.drain_submission(operation) && ledger.release(operation), "drain and release actual consumer");
        require(owner.reserve(inputs, 2, leases, error), "atomically reserve two independent physical owners");
        require(leases[0].request != leases[1].request && leases[0].incarnation && leases[1].incarnation &&
                leases[0].endpoint == 2 && leases[1].endpoint == 0 && owner.active() &&
                !owner.available(0) && !owner.available(1) && !owner.slots()[0].active &&
                !owner.completed_snapshot_source(0, error) && !owner.release_resources(error),
                "reserved owners exclude ordinary admission, snapshots and resource release");
        require(!owner.prepared_snapshot_source(leases[0], error) && error.empty(),
                "a reserved but unwritten preparation cannot supply snapshot state");
        require(same(untouched_a, fingerprint(a, *draft.slot_state(0), 2)) &&
                same(untouched_b, fingerprint(b, *draft.slot_state(1), 2)),
                "CPU reservations perform no destination writes before independent ACK publication");
        require(owner.prepare(leases[0], error) && owner.prepare(leases[1], error),
                "bind identities and directly reset each physical owner");
        require(draft.slot_cells(0) == 0 && draft.slot_begin(0) == 0 && draft.slot_prompt_len(0) == 3 &&
                draft.slot_cells(1) == 0 && a.ple_prev[0] == -1 && a.ple_prev[1] == -1 &&
                b.ple_prev[0] == -1 && b.ple_prev[1] == -1,
                "destination reset includes owner-specific draft interval and PLE history");
        const auto quiet_b = fingerprint(b, *draft.slot_state(1), 2);
        const auto row = [&](int slot, int32_t token, int64_t position) {
            int32_t picked = -1;
            require(verifier.run_slot_rows(&slot, 1, &token, &position, pool, pool_user, &picked, error) &&
                    verifier.commit_slots(error) && verifier.context_idle(error) && !dispatch.failed,
                    "execute and commit an actual physical verifier row");
            return picked;
        };
        core::ConversationCheckpoint first_cut;
        const auto prefix = [&](const program::ResidentPreparation& lease, int64_t position) {
            const auto& state = owner.slots()[(size_t) lease.slot];
            const int32_t token = state.prompt[(size_t) position];
            const int32_t next = state.prompt[(size_t) position + 1];
            (void) row(lease.slot, token, position);
            require(draft.prefill_slot(lease.slot, verifier.final_R(0), &next, 1, position, error) &&
                    owner.prefilled(lease, position + 1, error),
                    "uncaptured short-span draft front certifies only its actual destination interval");
            core::ConversationCheckpoint cut;
            cut.ids = owner.slots()[(size_t) lease.slot].ids;
            auto& session = lease.slot ? b : a;
            require(core::conversation_checkpoint_save(cut, session, geometry, error) &&
                    owner.checkpoint(lease, cut, error), "retain bounded checkpoint from actual committed prefix");
            if (!position) first_cut = cut.frozen_alias();
        };
        require(owner.prefilled(leases[0], 0, error) && owner.prefilled(leases[1], 0, error),
                "empty prefixes certify the n-1 endpoint of a one-token prompt");
        prefix(leases[0], 0);
        // A legal interior cut, while its resident owner is still preparing.
        // Cell0 already consumed private token1=22; only main R0 may be shared.
        core::ConversationBuffer interior_boundary(true);
        core::SavedConversation interior_shared;
        {
            auto prepared = owner.prepared_snapshot_source(leases[0], error);
            require(prepared && &prepared->session(0) == &a && prepared->owner.p == 1 &&
                    prepared->draft_cells == 1 && prepared->draft_begin == 0 &&
                    !owner.completed_snapshot_source(0, error),
                    "prepared snapshot borrows the actual certified seat without admitting it as completed");
            interior_boundary.resize(size_t(geometry.hc * geometry.n_embd) * sizeof(float));
            require(interior_boundary.visit(0, interior_boundary.size(), [&](uint8_t* dst, size_t n, size_t at) {
                        return cudaMemcpy(dst, reinterpret_cast<const uint8_t*>(verifier.final_R(0)) + at,
                                          n, cudaMemcpyDeviceToHost) == cudaSuccess;
                    }), "capture the interior owner's R0 before any later verifier row");
            interior_boundary.freeze();
            const std::vector<core::ConversationImageKey> no_images;
            const std::vector<core::ConversationCheckpoint> no_checks;
            core::ConversationView view{prepared->owner.ids, no_images, no_checks, true, true};
            view.draft_begin = prepared->draft_begin; view.draft_tokens = 0;
            view.draft_owner = prepared->owner.request; view.draft_next_token = -1;
            view.draft_boundary_residual = &interior_boundary; view.draft_boundary_position = 0;
            view.draft_boundary_owner = prepared->owner.request;
            require(core::conversation_snapshot_save(interior_shared, view, prepared->session(0), geometry,
                        prepared->draft, error) &&
                    core::conversation_snapshot_validate(interior_shared, a, geometry, prepared->draft,
                        core::ConversationSnapshotExpectation::exact_cut, error) &&
                    interior_shared.live.ids == std::vector<int32_t>{11} &&
                    interior_shared.checkpoints.empty() && interior_shared.draft_cells == 0 &&
                    interior_shared.draft_boundary_residual == interior_boundary,
                    "prepared exact cut sanitizes the consumed private draft cell and keeps only its owned residual");
            interior_shared.scope = request_a.shared_scope;
            interior_shared.freeze();
        }
        const auto immutable_interior = interior_shared.frozen_alias();
        prefix(leases[0], 1);
        require(same(quiet_b, fingerprint(b, *draft.slot_state(1), 2)),
                "one owner's real prefix and MTP writes do not touch its peer");
        auto stale = leases[0];
        ++stale.incarnation;
        require(!owner.prefilled(stale, 2, error), "stale physical incarnation cannot publish a frontier");
        require(!owner.prepared_snapshot_source(stale, error),
                "stale physical incarnation cannot lend a prepared snapshot");
        auto foreign = leases[0];
        foreign.stages[0] = &b;
        require(!owner.prefilled(foreign, 2, error), "swapped physical owner cannot publish a frontier");
        require(!owner.prepared_snapshot_source(foreign, error),
                "swapped physical destination cannot lend a prepared snapshot");
        const int32_t first = row(0, 33, 2);
        // Acquire the exact row while this verifier still owns final_R, before
        // executing the peer. No destination token at position 3 is captured.
        core::ConversationBuffer boundary_row;
        boundary_row.resize(size_t(geometry.hc * geometry.n_embd) * sizeof(float));
        require(boundary_row.visit(0, boundary_row.size(), [&](uint8_t* dst, size_t n, size_t at) {
                    return cudaMemcpy(dst, reinterpret_cast<const uint8_t*>(verifier.final_R(0)) + at,
                                      n, cudaMemcpyDeviceToHost) == cudaSuccess;
                }), "capture owned final main residual before another verifier row");
        boundary_row.freeze();
        const std::vector<int32_t> shared_ids{11, 22, 33};
        const std::vector<core::ConversationImageKey> shared_images;
        const std::vector<core::ConversationCheckpoint> shared_checks;
        core::ConversationView shared_view{shared_ids, shared_images, shared_checks, true, true};
        draft.certify_view(shared_view, 0);
        shared_view.draft_boundary_residual = &boundary_row; shared_view.draft_boundary_position = 2;
        shared_view.draft_boundary_owner = leases[0].request;
        core::SavedConversation shared;
        require(core::conversation_snapshot_save(shared, shared_view, a, geometry, *draft.slot_state(0), error) &&
                shared.draft_cells == 2 && shared.draft_next_token == -1 &&
                shared.draft_boundary_residual == boundary_row,
                "production capture preserves actual N-1 coverage and position-bound R without tokenN");
        shared.scope = request_a.scope;
        shared.freeze();
        const auto immutable_shared = shared.frozen_alias();
        owner.cancel(0); // cancellation after issue still owes mandatory draft catchup
        require(!owner.abort_preparation(leases[0], error) && owner.slots()[0].preparing,
                "final verifier commitment cannot be aborted before draft catchup");
        require(owner.first_token(leases[0], first, verifier.final_R(0), error) &&
                draft.slot_cells(0) == 3 && owner.slots()[0].p == 3 && owner.slots()[0].produced == 1 &&
                owner.slots()[0].stop && owner.slots()[0].ids == std::vector<int32_t>({11, 22, 33}),
                "max_new1 and cancellation publish only after final prompt and draft advance exactly once");
        require(!owner.first_token(leases[0], first, verifier.final_R(0), error) && owner.finish(0, error),
                "first-token lease is consumed once and its coherent cancelled state may finish");
        const int32_t second = row(1, 44, 0);
        require(owner.first_token(leases[1], second, verifier.final_R(0), error) && draft.slot_cells(1) == 1,
                "one-token prompt includes its mandatory first-token draft cell");
        const int32_t third = row(1, second, 1);
        require(owner.committed(1, third, verifier.final_R(0), error) && draft.slot_cells(1) == 2 &&
                owner.slots()[1].p == 2 && owner.slots()[1].produced == 2 && owner.finish(1, error),
                "ordinary post-first-token commitment continues the same actual physical state");
        const auto peer = fingerprint(b, *draft.slot_state(1), 2);
        const std::vector<core::ConversationImageKey> images;
        core::SavedConversation image;
        const auto& completed = owner.slots()[0];
        core::ConversationView source_view{completed.ids, images, completed.checks, completed.cvec};
        draft.certify_view(source_view, 0);
        require(core::conversation_snapshot_save(image, source_view, a, geometry, *draft.slot_state(0), error) &&
                image.draft_cells == draft.slot_cells(0) && image.draft_begin == draft.slot_begin(0) &&
                image.draft_next_token == draft.slot_next_token(0),
                "production private capture propagates actual source certification without manual receipt repair");
        const int32_t saved_draft_next = draft.slot_next_token(0);
        image.scope = request_a.scope;
        image.freeze();
        const std::vector<int64_t> branch{11, 22, 77};
        program::ResidentPreparation rewind;
        require(owner.prepare(0, branch, request_a, rewind, error) && rewind.position == 1 &&
                owner.slots()[0].p == 1 && draft.slot_cells(0) == 1 && draft.slot_begin(0) == 0 &&
                a.ple_prev[0] == -1 && a.ple_prev[1] == 11,
                "retained divergent suffix rewinds recurrent state, draft cells and PLE into its own destination");
        core::ConversationCheckpoint rewound;
        rewound.ids = {11};
        require(core::conversation_checkpoint_save(rewound, a, geometry, error) &&
                rewound.gdn == first_cut.gdn && rewound.ple == first_cut.ple &&
                rewound.tails == first_cut.tails && rewound.dead == first_cut.dead &&
                rewound.block_pos == first_cut.block_pos,
                "direct retained rewind restores the original real checkpoint payloads");
        owner.cancel(0);
        require(!owner.prepared_snapshot_source(rewind, error) && error.empty(),
                "cancellation closes the prepared capture seam before the owner is retired");
        require(owner.abort_preparation(rewind, error) && owner.available(0) &&
                !owner.prefilled(rewind, 1, error) && !owner.completed_snapshot_source(0, error),
                "drained cancellation discards unpublished owner and invalidates its old lease");
        const int32_t boundary = saved_draft_next == 88 ? 89 : 88;
        const std::vector<int64_t> extension{11, 22, 33, boundary};
        const program::ResidentPrepareInput restore_input{0, &extension, request_a};
        program::ResidentPreparation restored;
        require(owner.reserve(&restore_input, 1, &restored, error) && restored.position == 0,
                "reserve and ACK a cold seat before materializing its store source");
        image.cvec = false;
        const auto before_refusal = fingerprint(a, *draft.slot_state(0), 2);
        require(!owner.prepare(restored, &image, nullptr, core::ConversationSnapshotExpectation::ordinary, error) &&
                owner.slots()[0].preparing && !owner.slots()[0].failed &&
                same(before_refusal, fingerprint(a, *draft.slot_state(0), 2)),
                "whole-source preflight refusal writes no physical destination");
        image.cvec = true;
        auto unknown = image.frozen_alias();
        unknown.draft_begin = unknown.draft_cells = -1; unknown.draft_next_token = -1;
        require(!owner.prepare(restored, &unknown, nullptr, core::ConversationSnapshotExpectation::ordinary, error) &&
                error.find("resident preparation: source draft reuse refused:") == 0 &&
                owner.slots()[0].p == 0 && owner.slots()[0].preparing && !owner.slots()[0].failed &&
                same(before_refusal, fingerprint(a, *draft.slot_state(0), 2)),
                "uncertified external draft source is an explicit pre-write refusal, not qualified cold reuse");
        require(owner.prepare(restored, &image, nullptr, core::ConversationSnapshotExpectation::ordinary, error) &&
                owner.slots()[0].p == 2 && draft.slot_cells(0) == 2 &&
                a.ple_prev[0] == 11 && a.ple_prev[1] == 22,
                "post-ACK source override restores its actual coherent cut, not the reserved start or live endpoint");
        prefix(restored, 2);
        const int32_t last = row(0, boundary, 3);
        require(owner.first_token(restored, last, verifier.final_R(0), error) &&
                draft.slot_cells(0) == 4 && owner.finish(0, error),
                "restored suffix reaches a coherent first token through the production lifecycle");
        require(same(peer, fingerprint(b, *draft.slot_state(1), 2)) &&
                same(ambient, fingerprint(main, draft.kv_state(), 2)) &&
                owner.working().request == working_request && dispatch.request_generation == ambient_request,
                "direct prep, suffix rewind and restore preserve peer and ambient ownership and payloads");
        program::ResidentPreparation failed;
        const program::ResidentPrepareInput failure_input{0, &branch, request_a};
        require(owner.reserve(&failure_input, 1, &failed, error) && failed.position == 1 && a.ple_hist,
                "select real retained checkpoint for terminal partial-write refusal");
        transfer_failed = false;
        fail_destination = a.ple_hist;
        require(!owner.prepare(failed, error) && transfer_failed && owner.slots()[0].failed &&
                !owner.available(0) && !owner.slots()[0].preparing &&
                !owner.completed_snapshot_source(0, error),
                "partial destination rewind failure leaves only that physical seat unavailable");
        require(same(peer, fingerprint(b, *draft.slot_state(1), 2)) &&
                same(ambient, fingerprint(main, draft.kv_state(), 2)),
                "failed destination changes neither peer nor ambient payload");
        for (const int32_t token_n : {77, 78}) {
            auto follower = request_a;
            follower.scope.value[0] = uint8_t(token_n + 20);
            const std::vector<int64_t> suffix{11, token_n, 33};
            const program::ResidentPrepareInput input{1, &suffix, follower};
            program::ResidentPreparation mounted;
            require(owner.reserve(&input, 1, &mounted, error) &&
                    owner.prepare(mounted, &interior_shared, nullptr,
                        core::ConversationSnapshotExpectation::exact_cut, error) &&
                    owner.slots()[1].p == 1 && draft.slot_cells(1) == 1 &&
                    draft.slot_next_token(1) == token_n,
                    "different private conversations catch up a prepared interior cut with their own tokenN");
            prefix(mounted, 1);
            const int32_t next = row(1, 33, 2);
            require(owner.first_token(mounted, next, verifier.final_R(0), error) &&
                    draft.slot_cells(1) == 3 && owner.finish(1, error) &&
                    same_snapshot(interior_shared, immutable_interior),
                    "both real suffixes finish without mutating the prepared shared source or mixing private rows");
        }
        // The same immutable N-token image resumes twice at full live N. Each
        // destination must supply its own tokenN to exactly one missing KV cell.
        for (const int32_t token_n : {88, 89}) {
            const std::vector<int64_t> suffix{11, 22, 33, token_n};
            const program::ResidentPrepareInput shared_input{1, &suffix, request_a};
            program::ResidentPreparation mounted;
            require(owner.reserve(&shared_input, 1, &mounted, error) &&
                    owner.prepare(mounted, &shared, nullptr, core::ConversationSnapshotExpectation::exact_cut, error) &&
                    owner.slots()[1].p == 3 && draft.slot_cells(1) == 3 &&
                    draft.slot_next_token(1) == token_n,
                    "full-live shared mount completes exactly its destination-owned missing cell");
            require(same_snapshot(shared, immutable_shared) &&
                    shared.draft_next_token == -1 && shared.live.ids.size() == 3 &&
                    shared.draft_boundary_residual == boundary_row,
                    "mount and advance never modify shared source or persist either private tokenN");
            const int32_t next = row(1, token_n, 3);
            require(owner.first_token(mounted, next, verifier.final_R(0), error) &&
                    draft.slot_cells(1) == 4 && owner.finish(1, error),
                    "catchup-completed source continues normal verifier/first-token lifecycle");
        }
        require(same(ambient, fingerprint(main, draft.kv_state(), 2)) && owner.release_resources(error),
                "shared catchup preserves ambient owner and releases all drained resources");
        std::printf("PACKED_RESIDENT_CHECKS checks=%d actual_verifier=1 direct_prepare=1 short_mtp=1 "
                    "rewind=1 max_new1=1 cancel_after_issue=1 partial_write=1 prepared_interior_shared=1 "
                    "external_store_qualified=0 failures=0\n", checks);
        return 0;
    } catch (const std::exception& exception) {
        fail_destination = nullptr;
        std::fprintf(stderr, "PACKED_RESIDENT_CHECKS checks=%d failures=1 error=%s\n", checks, exception.what());
        return 1;
    }
}
} // namespace

int resident_model_checks(strata::program::ResidentSlots& owner, strata::core::ExpertDispatch& dispatch,
                          const strata::core::ModelGeometry& geometry, strata::core::SessionState& main,
                          strata::core::SessionState& a, strata::core::SessionState& b,
                          strata::core::MtpDrafter& draft, strata::core::Verifier& verifier,
                          strata::core::PoolMultiFn pool, void* pool_user) {
    using namespace strata;
    std::string error;
    int checks = 0;
    auto require = [&](bool ok, const char* message) {
        ++checks;
        if (!ok) throw std::runtime_error(std::string(message) + ": " + error);
    };
    try {
        if (std::getenv("STRATA_TEST_PREFILL_SPLIT_OWNER")) return prefill_split_owner_checks(owner, main);
        if (std::getenv("STRATA_TEST_PACKED_RESIDENT"))
            return packed_resident_checks(owner, dispatch, geometry, main, a, b, draft, verifier, pool, pool_user);
        if (std::getenv("STRATA_TEST_MIXED_RESIDENCY")) {
            // Exercise the production Verifier, not a single-sequence parity
            // kernel or a hand-written imitation of its row compaction.
            require(main.qsa_alloc>0 && main.qsa_states[main.qsa_primary()].kv_mode==0 &&
                    a.qsa_states[a.qsa_primary()].kv_mode==1 && b.qsa_states[b.qsa_primary()].kv_mode==1,
                    "mixed fixture needs resident working state and streamed slots");
            const std::vector<int32_t> prefix{11,22};
            const std::vector<core::ConversationImageKey> images;
            const std::vector<core::ConversationCheckpoint> checkpoints;
            auto capture = [&](core::SessionState& state, const std::vector<int32_t>& ids) {
                core::SavedConversation saved;
                require(core::conversation_snapshot_save(saved,{ids,images,checkpoints,true},state,geometry,nullptr,error),
                        "capture complete real mixed-owner state");
                return saved;
            };
            std::array<core::SavedConversation,2> opening{capture(main,prefix),capture(a,prefix)};
            const std::array<int32_t,2> previous{main.ple_prev[0],main.ple_prev[1]};
            const auto old_plan=dispatch.plan;
            const auto old_generation=dispatch.request_generation;
            const std::array<uint64_t,2> old_requests{dispatch.slot_requests[0],dispatch.slot_requests[1]};
            require(owner.release_resources(error),"release original batch bindings before mixed-owner binding");
            require(verifier.init_slots({&main,&a},error),"bind resident and streamed owners in one real verifier");
            kernels::SamplerParams sampling;
            sampling.greedy=true; sampling.temperature=0; sampling.penalty_last_n=0; sampling.seed=17;
            auto bind = [&] {
                for (int slot=0;slot<2;++slot) {
                    dispatch.slot_requests[slot]=1000+slot;
                    require(verifier.set_slot_context(slot,dispatch.slot_requests[slot],sampling,&prefix,true,error),
                            "bind independent real mixed-row identities");
                }
                dispatch.request_generation=1000;
                dispatch.plan=verifier.plan_sink();
            };
            bind();
            auto reset = [&] {
                require(verifier.context_idle(error),"drain mixed window before restoring its opening state");
                require(core::conversation_snapshot_restore(opening[0],main,geometry,nullptr,error)==core::ConversationRestore::restored &&
                        core::conversation_snapshot_restore(opening[1],a,geometry,nullptr,error)==core::ConversationRestore::restored,
                        "restore both mixed owners independently");
            };
            struct Observed {
                std::array<core::SavedConversation,2> state;
                std::array<std::vector<float>,2> residual;
                std::array<int32_t,2> picks{-1,-1};
            };
            auto window = [&](const std::vector<int>& slots, const std::array<int32_t,2>& tokens,
                              core::BatchRowMask other_mask) {
                core::BatchRow rows[2];
                int32_t output[2]={-7,-7};
                std::array<bool,2> active{};
                for (size_t t=0;t<slots.size();++t) {
                    const int slot=slots[t];
                    auto& row=rows[t];
                    row.slot=slot; row.state=slot?&a:&main; row.request_id=1000+slot;
                    row.cancellation_generation=row.request_id; row.token=tokens[slot];
                    row.position=2; row.sequence_end=3; row.output_offset=(int)t;
                    row.mask=core::BatchRowMask::active;
                    active[slot]=true;
                }
                int count=(int)slots.size();
                if (count==1) {
                    rows[1].slot=1-slots[0]; rows[1].state=slots[0]?&main:&a;
                    rows[1].mask=other_mask; rows[1].output_offset=1; count=2;
                }
                if (!count) {
                    for (int slot=0;slot<2;++slot) {
                        rows[slot].slot=slot; rows[slot].state=slot?&a:&main;
                        rows[slot].mask=other_mask; rows[slot].output_offset=slot;
                    }
                    count=2;
                }
                require(verifier.run_slot_rows(rows,count,pool,pool_user,output,error) &&
                        verifier.commit_slots(error) && verifier.context_idle(error) && !dispatch.failed,
                        "bounded production mixed window executes and commits");
                Observed observed;
                for (int slot=0;slot<2;++slot) {
                    auto ids=prefix;
                    if (active[slot]) ids.push_back(tokens[slot]);
                    observed.state[slot]=capture(slot?a:main,ids);
                }
                for (size_t t=0;t<slots.size();++t) {
                    const int slot=slots[t];
                    observed.picks[slot]=output[t];
                    observed.residual[slot].resize((size_t)geometry.hc*geometry.n_embd);
                    require(cudaMemcpy(observed.residual[slot].data(),verifier.final_R((int)t),
                                       observed.residual[slot].size()*4,cudaMemcpyDeviceToHost)==cudaSuccess,
                            "read the actual committed row residual");
                    require(verifier.validated_slot_request_id(slot)==1000+(uint64_t)slot,
                            "GPU commit canary validates the mixed row's owner");
                }
                if (slots.size()==1) require(output[1]==-1,"inactive or cancelled row never produces an output");
                if (slots.empty()) require(output[0]==-1 && output[1]==-1,"all-inert mixed window produces no outputs");
                return observed;
            };
            auto matches = [&](const Observed& got, const Observed& expected, int slot) {
                require(same_snapshot(got.state[slot],expected.state[slot]) &&
                        got.picks[slot]==expected.picks[slot] && got.residual[slot]==expected.residual[slot],
                        "mixed graph agrees byte-for-byte with the independent owner window");
            };
            const std::array<int32_t,2> first{33,44}, second{55,66};
            reset();
            const auto solo0=window({0},first,core::BatchRowMask::padding);
            require(same_snapshot(solo0.state[1],opening[1]),"unselected streamed owner stays unchanged");
            reset();
            const auto solo1=window({1},first,core::BatchRowMask::padding);
            require(same_snapshot(solo1.state[0],opening[0]),"unselected resident owner stays unchanged");
            reset();
            const size_t before=verifier.batch_resources().graph_keys;
            const auto captured=window({1,0},first,core::BatchRowMask::padding);
            const size_t captured_keys=verifier.batch_resources().graph_keys;
            require(captured_keys==before+1,"mixed row order captures exactly one new window/commit key");
            matches(captured,solo0,0); matches(captured,solo1,1);
            require(captured.residual[0]!=captured.residual[1],"distinct owners/tokens yield distinct actual residuals");
            reset();
            const auto replayed=window({1,0},first,core::BatchRowMask::padding);
            require(verifier.batch_resources().graph_keys==captured_keys,"second mixed window replays the retained graph");
            matches(replayed,captured,0); matches(replayed,captured,1);
            reset();
            const auto next0=window({0},second,core::BatchRowMask::padding);
            reset();
            const auto next1=window({1},second,core::BatchRowMask::padding);
            reset();
            const auto changed=window({1,0},second,core::BatchRowMask::padding);
            require(verifier.batch_resources().graph_keys==captured_keys,"changed tokens reuse the same mixed graph");
            matches(changed,next0,0); matches(changed,next1,1);
            require(changed.residual[0]!=captured.residual[0] && changed.residual[1]!=captured.residual[1],
                    "replay reads new row inputs rather than capture-time tokens");
            for (auto mask : {core::BatchRowMask::padding,core::BatchRowMask::finished,core::BatchRowMask::cancelled})
                for (int active : {0,1}) {
                    reset();
                    const auto inert=window({active},first,mask);
                    matches(inert,active?solo1:solo0,active);
                    require(same_snapshot(inert.state[1-active],opening[1-active]),
                            "inactive/finished/cancelled peer keeps every private running and KV byte");
                }
            for (auto mask : {core::BatchRowMask::padding,core::BatchRowMask::finished,core::BatchRowMask::cancelled}) {
                reset();
                const size_t keys=verifier.batch_resources().graph_keys;
                const auto inert=window({},first,mask);
                require(same_snapshot(inert.state[0],opening[0]) && same_snapshot(inert.state[1],opening[1]) &&
                        verifier.batch_resources().graph_keys==keys,"all-inert mixed window neither mutates owners nor captures a graph");
            }
            reset();
            require(verifier.release_batch_resources(error),"release real mixed-owner graphs and scratch");
            require(verifier.n_slots()==0 && verifier.batch_resources().graph_keys==0 &&
                    verifier.batch_resources().device_bytes==0 && verifier.batch_resources().pinned_bytes==0,
                    "release retires every mixed graph and batch allocation");
            require(same_snapshot(capture(main,prefix),opening[0]) && same_snapshot(capture(a,prefix),opening[1]),
                    "batch release preserves both residency owners' private state");
            require(verifier.init_slots({&main,&a},error),"re-initialize the same stable mixed owners");
            bind();
            const auto reinitialized=window({1,0},first,core::BatchRowMask::padding);
            matches(reinitialized,captured,0); matches(reinitialized,captured,1);
            reset();
            require(verifier.release_batch_resources(error),"retire temporary mixed owner bindings");
            dispatch.plan=old_plan; dispatch.request_generation=old_generation;
            dispatch.slot_requests[0]=old_requests[0]; dispatch.slot_requests[1]=old_requests[1];
            main.ple_prev[0]=previous[0]; main.ple_prev[1]=previous[1];
            require(owner.init_resources(error),"rebind original resident lifecycle owner");
            std::printf("MIXED_RESIDENCY_WINDOWS capture=1 replay=2 inactive=2 finished=2 cancelled=2 all_inert=3 reinitialized=1 failures=0\n");
            require(draft.slot_state(0) && draft.slot_state(1) && draft.kv_state().kv_mode==0 &&
                    draft.slot_state(0)->kv_mode==2 && draft.kv_state().kv_int8 && draft.slot_state(0)->kv_int8,
                    "hybrid process uses actual resident and ring INT8 draft states");
            core::ConversationKv original,peer,marker,changed_draft;
            require(core::conversation_kv_save(original,draft.kv_state(),geometry,2,false,error) &&
                    core::conversation_kv_save(peer,*draft.slot_state(1),geometry,2,false,error),"capture real working and independent peer draft");
            marker=original;
            for (auto* part : {&marker.k,&marker.v,&marker.k_scale,&marker.v_scale})
                require(part->visit(0,part->size(),[](uint8_t* p,size_t n,size_t){std::memset(p,0x3c,n);return true;}),
                        "mark each real draft payload run");
            require(draft.bind_request(0, 1000, error), "bind standalone draft transfer fixture destination");
            require(core::conversation_kv_restore(marker,draft.kv_state(),geometry,2,false,error) &&
                    draft.save_slot(0,2,error),"save marked working draft into host-backed slot ring");
            uint64_t hash=0;
            require(core::conversation_kv_verify(marker,*draft.slot_state(0),geometry,2,false,hash,error),
                    "real draft save refills both codes and both scale runs of the ring");
            changed_draft=marker;
            for (auto* part : {&changed_draft.k,&changed_draft.v,&changed_draft.k_scale,&changed_draft.v_scale})
                require(part->visit(0,part->size(),[](uint8_t* p,size_t n,size_t){std::memset(p,0x5a,n);return true;}),
                        "replace working draft with distinct outgoing bytes");
            require(core::conversation_kv_restore(changed_draft,draft.kv_state(),geometry,2,false,error) &&
                    !core::conversation_kv_verify(marker,draft.kv_state(),geometry,2,false,hash,error),
                    "draft restore fixture really overwrites the working payload");
            require(draft.restore_slot(0,2,error) &&
                    core::conversation_kv_verify(marker,draft.kv_state(),geometry,2,false,hash,error),
                    "restore the saved ring draft over the overwritten resident working draft");
            require(core::conversation_kv_verify(peer,*draft.slot_state(1),geometry,2,false,hash,error),
                    "draft save/restore cannot alter the independent peer's payload");
            require(core::conversation_kv_restore(original,draft.kv_state(),geometry,2,false,error) &&
                    draft.save_slot(0,2,error),"restore original draft before resident lifecycle regressions");
            std::puts("DRAFT_RESIDENCY_ROUNDTRIP working=resident slot=ring format=int8 failures=0");
        }
        const uint64_t w4 = core::Verifier::planned_device_bytes(geometry, main.max_cells, 150000, 4);
        const uint64_t w8 = core::Verifier::planned_device_bytes(geometry, main.max_cells, 150000, 8);
        require(w4 > 0 && w8 > w4 && w8 != UINT64_MAX, "full eight-row verifier uses bounded actual arena sequence");
        require(core::Verifier::planned_device_bytes(geometry, main.max_cells, 150000, 9) == UINT64_MAX,
                "pricing refuses unsupported window width");
        const std::vector<int64_t> prompt{11, 22, 33, 44};
        const std::vector<int32_t> prefix{11, 22};
        program::ResidentRequest request;
        request.max_new = 8;
        request.scope.value[0] = 1;
        request.shared_scope.value[0] = 71;
        request.pin = 2;
        request.sampling.seed = 0; // chosen seed must survive repeated yields
        request.sampling.penalty_last_n = 2;
        require(!owner.init_resources(error), "repeated init refuses");
        const core::ExpertOperation operation{71, 0, 0, 1, 1};
        auto& ledger = dispatch.completion.at(0);
        require(ledger.begin(operation, 0) && ledger.retain_submission(operation), "retain submitted operation");
        require(!owner.begin(0, prompt, request, error), "begin refuses in-flight scratch");
        require(owner.working().request == 0 && dispatch.request_generation == 0, "refusal leaves identity unpublished");
        ledger.cancel();
        require(!owner.begin(0, prompt, request, error), "cancellation is not a consumer drain");
        require(ledger.drain_submission(operation) && ledger.release(operation), "consumer drain releases operation");
        require(owner.begin(-1, prompt, request, error), "begin request A");
        const uint64_t aid = owner.working().request, seed = owner.working().sampling.seed;
        require(seed != 0, "automatic seed resolved");
        std::vector<core::ConversationCheckpoint> checkpoint_chain(1);
        auto& checkpoint = checkpoint_chain.front();
        checkpoint.ids = prefix;
        require(core::conversation_checkpoint_save(checkpoint, main, geometry, error), "save real working checkpoint");
        program::ResidentAdmission admission{33, &checkpoint_chain};
        const auto* main_gdn=main.gdn_state;
        const auto* slot_gdn=a.gdn_state;
        const auto* slot_qsa=a.qsa_states;
        core::SavedConversation opening, parked;
        const std::vector<core::ConversationImageKey> images;
        const std::vector<core::ConversationCheckpoint> checks_at_open;
        const core::ConversationView view{prefix,images,checks_at_open,true};
        require(core::conversation_snapshot_save(opening,view,main,geometry,draft.kv_state(),error),
                "capture real working state before resident admission");
        require(owner.park(0, prefix, admission, true, error), "park request A");
        require(core::conversation_snapshot_save(parked,view,a,geometry,*draft.slot_state(0),error) &&
                same_snapshot(opening,parked), "resident admission preserves every main and draft payload byte");
        require(main.gdn_state==main_gdn && a.gdn_state==slot_gdn && a.qsa_states==slot_qsa,
                "resident admission preserves working and stationary slot addresses");
        const auto* history = &owner.slots()[0].ids;
        require(!owner.park(1, prefix, admission, true, error), "same request cannot own two slots");
        require(owner.slots()[1].request == 0 && owner.slots()[1].ids.empty(), "duplicate refusal leaves B empty");
        require(owner.release_resources(error) && !owner.release_resources(error), "release once only");
        require(!owner.available(0) && owner.slots()[0].ids == prefix, "released resources retain prefix but refuse admission");
        require(owner.begin(-1, prompt, request, error) && owner.working().request == aid &&
                owner.working().sampling.seed == seed && owner.working().resume_slot == 0,
                "released solo resend retains identity and automatic seed");
        require(!owner.begin(0, prompt, request, error), "released scratch still refuses BGEN admission");
        require(owner.resume(0, nullptr, error), "restore same yielded solo while scratch is released");
        core::SavedConversation resumed;
        require(core::conversation_snapshot_save(resumed,view,main,geometry,draft.kv_state(),error) &&
                same_snapshot(opening,resumed) && main.gdn_state==main_gdn && a.gdn_state==slot_gdn,
                "resident resume preserves payloads and addresses while scratch is released");
        require(owner.init_resources(error), "reinitialize retained contexts");
        require(owner.park(0, prefix, admission, true, error), "retain solo continuation before promotion");
        require(owner.begin(1, prompt, request, error) && owner.working().request != aid &&
                owner.working().resume_slot == -1, "different physical destination cannot claim parked identity");
        auto altered_request = request;
        altered_request.sampling.penalty_repeat += 0.1f;
        require(owner.begin(0, prompt, altered_request, error) && owner.working().request != aid &&
                owner.working().resume_slot == -1, "different settings cannot claim parked identity");
        altered_request = request;
        altered_request.shared_scope.value[0] ^= 1;
        require(owner.begin(0, prompt, altered_request, error) && owner.working().request != aid &&
                owner.working().resume_slot == -1, "a changed shared authorization cannot claim a yielded identity");
        altered_request = request;
        ++altered_request.pin;
        require(owner.begin(0, prompt, altered_request, error) && owner.working().request != aid &&
                owner.working().resume_slot == -1, "a changed exact pin cut cannot claim a yielded identity");
        const std::vector<int64_t> other_prompt{11, 22, 33, 45};
        require(owner.begin(0, other_prompt, request, error) && owner.working().request != aid &&
                owner.working().resume_slot == -1, "different full prompt cannot claim parked identity");
        require(owner.begin(0, prompt, request, error), "promote yielded solo request into its parked slot");
        require(owner.working().request == aid && owner.working().sampling.seed == seed && owner.working().resume_slot == 0,
                "solo-to-resident promotion retains identity and automatic seed");
        require(owner.resume(0, nullptr, error), "restore yielded request");
        auto aliased_source = owner.completed_snapshot_source(0, error);
        require(aliased_source.has_value(), "completed source exposes the actual mutable checkpoint chain");
        admission.checkpoints = &aliased_source->checkpoints;
        require(owner.park(0, prefix, admission, true, error), "yield same identity again");
        require(owner.slots()[0].request == aid && &owner.slots()[0].ids == history, "repark keeps stable owner/history object");
        require(owner.slots()[0].checks.size() == 1 && owner.slots()[0].checks[0].ids == prefix,
                "aliased checkpoint survives repark");
        admission.checkpoints = &checkpoint_chain;
        owner.cancel(0);
        require(!owner.slots()[0].partial && !owner.slots()[0].active, "cancel invalidates yielded continuation");
        require(owner.begin(0, prompt, request, error), "cancelled yield starts a new request");
        const uint64_t cid = owner.working().request;
        require(cid > aid && owner.working().resume_slot == -1, "cancelled request cannot resurrect identity");
        require(owner.admit(0, prefix, admission, error), "reuse A slot for C");
        require(owner.slots()[0].request == cid && owner.slots()[0].ids == prefix && !owner.slots()[0].stop,
                "C starts with only its private context");
        require(!owner.release_resources(error), "active request prevents resource release");
        require(ledger.begin(operation, 0) && ledger.retain_submission(operation), "retain C consumer");
        owner.cancel(0);
        require(!owner.finish(0, error) && !owner.available(0), "finish refuses a retained consumer");
        ledger.cancel();
        require(!owner.finish(0, error) && owner.slots()[0].active,
                "cancelled submission still prevents finish/reuse");
        require(ledger.drain_submission(operation) && ledger.release(operation), "drain C consumer");
        require(owner.finish(0, error), "finish C");
        require(owner.release_resources(error), "release batch scratch with C cached");
        require(owner.begin(-1, prompt, request, error) && owner.resume(0, nullptr, error),
                "solo restore uses cached C while batch scratch is released");
        require(owner.slots()[0].request == cid && owner.slots()[0].ids == prefix &&
                !owner.available(0) && owner.init_resources(error),
                "solo restoration retains C without making batch admission ready");
        require(owner.begin(1, prompt, request, error), "begin independent B");
        require(owner.admit(1, prefix, admission, error), "admit B");
        const auto bid = owner.slots()[1].request;
        program::SessionFingerprint before, after;
        require(program::session_fingerprint(b, *draft.slot_state(1), geometry, 2, before, error), "fingerprint B before failed A transition");
        require(owner.begin(0, prompt, request, error), "begin replacement A");
        if (std::getenv("STRATA_TEST_RING_COPY_REFUSAL")) {
            // A terminal fixture: a failed draft operation intentionally keeps
            // its scratch lease poisoned, so no later lifecycle test can reuse it.
            const auto* ring=draft.slot_state(0);
            require(ring && ring->kv_mode==2,"ring refusal needs an actual host-backed draft slot");
            auto payload=[](const core::QsaState& st, bool value, bool resident) -> void* {
                const bool host=st.kv_mode!=0 && !resident;
                if (st.kv_q4 || (st.kv_hybrid && value))
                    return value?(host?st.host.v_q4:st.v_q4):(host?st.host.k_q4:st.k_q4);
                if (st.kv_int8 || st.kv_hybrid)
                    return value?(host?st.host.v_q:st.v_q):(host?st.host.k_q:st.k_q);
                return value?(host?st.host.v_pool:st.v_pool):(host?st.host.k_pool:st.k_pool);
            };
            void* ring_k=payload(*ring,false,true);
            void* ring_v=payload(*ring,true,true);
            const uint32_t marker=0x3c3c3c3c;
            require(cudaMemcpy(payload(draft.kv_state(),false,false),&marker,sizeof(marker),cudaMemcpyDefault)==cudaSuccess &&
                    cudaMemcpy(payload(draft.kv_state(),true,false),&marker,sizeof(marker),cudaMemcpyDefault)==cudaSuccess &&
                    cudaMemset(ring_k,0xa5,sizeof(marker))==cudaSuccess &&
                    cudaMemset(ring_v,0xa5,sizeof(marker))==cudaSuccess && cudaDeviceSynchronize()==cudaSuccess,
                    "distinguish real ring pages from the incoming authoritative draft");
            require(cudaGetLastError()==cudaSuccess,"ring refusal starts with a clear runtime error slot");
            fail_destination=ring_v;
            transfer_failed=false;
            require(!owner.admit(0,prefix,admission,error) && transfer_failed &&
                    error.find(cudaGetErrorString(cudaErrorInvalidValue))!=std::string::npos,
                    "actual draft ring copy refusal reaches resident admission");
            require(cudaDeviceSynchronize()==cudaSuccess && cudaGetLastError()==cudaSuccess,
                    "software refusal leaves no runtime error for a second last-error read");
            uint32_t key=0,value=0;
            require(cudaMemcpy(&key,ring_k,sizeof(key),cudaMemcpyDeviceToHost)==cudaSuccess &&
                    cudaMemcpy(&value,ring_v,sizeof(value),cudaMemcpyDeviceToHost)==cudaSuccess &&
                    key==marker && value==0xa5a5a5a5,
                    "ring refusal happened after key mutation and before value mutation");
            require(owner.slots()[0].failed && !owner.slots()[0].active && !owner.available(0) &&
                    !owner.slots()[0].request && !dispatch.slot_requests[0],
                    "failed draft ring cannot publish its resident target");
            require(!draft.context_idle(error) && !draft.bind_request(0,999,error) &&
                    !draft.bind_request(8,999,error) && !draft.save_slot(0,2,error) &&
                    !draft.restore_slot(0,2,error),
                    "poisoned draft scratch refuses both rebinding and save/restore reuse");
            require(!owner.begin(-1,prompt,request,error) && !owner.admit(0,prefix,admission,error),
                    "resident owner cannot reuse a poisoned draft through a successor request");
            require(program::session_fingerprint(b,*draft.slot_state(1),geometry,2,after,error) &&
                    before.gdn==after.gdn && before.ple==after.ple && before.kv==after.kv && before.draft==after.draft &&
                    before.tail==after.tail && before.dead==after.dead && before.pooled==after.pooled &&
                    owner.slots()[1].request==bid && owner.slots()[1].active && owner.slots()[1].ids==prefix,
                    "draft ring refusal preserves independent peer state and identity");
            std::printf("RESIDENT_RING_REFUSAL checks=%d poisoned=1 rebind_refused=1 reuse_refused=1 failures=0\n",checks);
            return 0;
        }
        // GDN is copied before PLE. Mark it so the partial write is observable.
        require(cudaMemset(main.gdn_state, 0x3c, sizeof(float)) == cudaSuccess, "mark source state");
        fail_destination = a.ple_hist;
        require(fail_destination != nullptr, "full-model PLE transfer target exists");
        require(!owner.admit(0, prefix, admission, error) && transfer_failed, "actual partial transfer fails");
        uint32_t changed = 0;
        require(cudaMemcpy(&changed, a.gdn_state, sizeof(changed), cudaMemcpyDeviceToHost) == cudaSuccess && changed == 0x3c3c3c3c,
                "failed transfer occurred after destination mutation");
        require(owner.slots()[0].failed && !owner.available(0) && !owner.slots()[0].active,
                "partial target cannot be published or reused");
        require(program::session_fingerprint(b, *draft.slot_state(1), geometry, 2, after, error), "fingerprint unaffected B");
        require(before.gdn == after.gdn && before.ple == after.ple && before.kv == after.kv && before.draft == after.draft &&
                before.tail == after.tail && before.dead == after.dead && before.pooled == after.pooled &&
                owner.slots()[1].request == bid && owner.slots()[1].active && owner.slots()[1].ids == prefix,
                "failure cannot mutate B state or identity");
        owner.cancel(1);
        require(owner.finish(1, error), "cancel survivor after failed peer");
        require(owner.release_resources(error) && owner.init_resources(error) && !owner.available(0) && owner.available(1),
                "resource re-init restores valid contexts only");
        // #879 taint ownership through the real bind transition: the taint belongs to the identity that
        // ran, never to the seat. A fresh (non-continuing) admission retires it with the old identity and
        // the seat caches again; a continuing rebind of the same identity keeps it, and that identity's
        // completion stays out of the cache.
        require(owner.begin(1, prompt, request, error), "begin the run that will be tainted");
        const uint64_t poisoned = owner.working().request;
        require(owner.admit(1, prefix, admission, error), "admit the run that will be tainted");
        owner.taint(1);
        require(owner.finish(1, error), "finish the poisoned run");
        require(!owner.slots()[1].cached && !owner.completed_snapshot_source(1, error).has_value(),
                "a tainted completion is neither cached nor reusable");
        require(owner.begin(1, prompt, request, error), "begin a fresh identity over the tainted seat");
        require(owner.working().request != poisoned && owner.working().resume_slot == -1,
                "a finished poisoned identity is never continued");
        require(owner.admit(1, prefix, admission, error), "the fresh non-continuing admission takes the seat");
        require(!owner.slots()[1].tainted, "a fresh non-continuing bind retires the seat's stale taint");
        const uint64_t fresh = owner.working().request;
        require(owner.finish(1, error), "finish the fresh run");
        require(owner.completed_snapshot_source(1, error).has_value(), "the fresh identity completes reusable");
        require(owner.park(1, prefix, admission, true, error), "yield the fresh identity");
        require(owner.begin(1, prompt, request, error) && owner.working().request == fresh &&
                owner.working().resume_slot == 1, "promote the yielded identity for its continuing rebind");
        owner.taint(1);
        require(owner.park(1, prefix, admission, true, error), "rebind the same tainted identity continuing");
        require(owner.slots()[1].tainted, "a continuing rebind keeps the identity's own taint");
        require(!owner.completed_snapshot_source(1, error).has_value(),
                "the continuing tainted identity stays non-reusable");
        require(owner.begin(1, prompt, request, error), "select retained B for restore failure");
        transfer_failed = false;
        fail_destination = main.ple_hist;
        require(!owner.resume(1, nullptr, error) && transfer_failed, "partial working restore fails");
        require(!owner.begin(-1, prompt, request, error) && !owner.available(1),
                "partial restore poisons working reuse and its source context");
        require(owner.release_resources(error), "release test resources");
        std::printf("RESIDENT_MODEL_CHECKS checks=%d partial_transfer=1 failures=0\n", checks);
        return 0;
    } catch (const std::exception& exception) {
        fail_destination = nullptr;
        std::fprintf(stderr, "RESIDENT_MODEL_CHECKS checks=%d failures=1 error=%s\n", checks, exception.what());
        return 1;
    }
}

// Capture-only behavior checks at the real Verifier seam. The loaded empty
// logical sessions are observed, not reset or advanced by fake inference.
int verifier_startup_model_checks(const strata::core::ModelGeometry& geometry,
        const std::vector<strata::program::ResidentStage>& stages, int owner_rows, int peer_base) {
    using namespace strata;
    std::string error;
    int checks = 0;
    auto require = [&](bool ok, const char* message) {
        ++checks;
        if (!ok) throw std::runtime_error(std::string(message) + ": " + error);
    };
    struct Observation {
        core::GraphCapturePhase deny = core::GraphCapturePhase::BeforeCapture;
        bool rejecting = false;
        size_t attempts = 0, retained = 0;
    } observation;
    auto observe = +[](void* opaque, const core::GraphCaptureKey&, core::GraphCapturePhase phase,
                      const core::GraphCaptureSample&, std::string& why) -> bool {
        auto& seen = *static_cast<Observation*>(opaque);
        seen.attempts += phase == core::GraphCapturePhase::BeforeCapture;
        seen.retained += phase == core::GraphCapturePhase::Retained;
        if (seen.rejecting && phase == seen.deny) {
            why = "startup fixture: deliberate capture observer refusal"; return false;
        }
        return true;
    };
    auto fingerprint = [&] {
        uint64_t result = program::SessionFingerprint::basis;
        const core::QsaState no_draft{};
        for (const auto& stage : stages)
            for (int slot = -1; slot < int(stage.slots.size()); ++slot) {
                const auto& state = *(slot < 0 ? stage.main : stage.slots[size_t(slot)]);
                program::SessionFingerprint h;
                require(program::session_fingerprint(state, no_draft, geometry, 0, h, error),
                        "read actual loaded empty logical state on every owning device");
                for (const uint64_t word : {h.gdn, program::fingerprint_ple(h, state), program::fingerprint_kv(h),
                                           state.allocation_id})
                    result = program::fingerprint_word(result, word);
            }
        return result;
    };
    auto idle = [&] {
        for (const auto& stage : stages)
            for (auto* verifier : {stage.verifier, stage.batch_peer})
                require(verifier && verifier->context_idle(error) && !verifier->batch_busy() &&
                        !verifier->in_flight() && verifier->windows == 0 && verifier->spec_windows == 0,
                        "all physical owners retain idle unlaunched scratch and zero inference counters");
    };
    try {
        require(stages.size() == 3 && owner_rows >= 1 && owner_rows <= 4 && peer_base >= owner_rows &&
                (peer_base == 4 || peer_base == 8) &&
                stages[0].verifier && stages[0].batch_peer,
                "use actual bounded three-stage replay chains");
        for (const auto& stage : stages)
            require(stage.main && stage.slots.size() == 4 && stage.verifier && stage.batch_peer,
                    "all stationary stage and resident owners are initialized");
        idle();
        const uint64_t opening = fingerprint();
        auto& first = *stages[0].verifier;
        observation.rejecting = true;
        observation.deny = core::GraphCapturePhase::AfterInstantiate;
        {
            const core::GraphCaptureObservationScope scope(observe, &observation);
            require(!first.prepare_startup_graphs(0, owner_rows, error) && observation.attempts == 1 &&
                    observation.retained == 0, "post-instantiate refusal destroys an unretained ordinary executable");
        }
        error.clear();
        idle();
        observation = {};
        {
            const core::GraphCaptureObservationScope scope(observe, &observation);
            require(first.prepare_startup_graphs(0, owner_rows, error) && observation.retained > 0,
                    "retry capture-only preparation after an unlaunched failure");
        }
        auto& peer = *stages[0].batch_peer;
        observation = {};
        {
            const core::GraphCaptureObservationScope scope(observe, &observation);
            require(peer.prepare_startup_graphs(peer_base, owner_rows, error) && observation.retained > 0,
                    "prepare the distinct physical peer chain at its resolved handoff base");
        }
        const size_t expected = size_t(owner_rows);
        for (const auto& stage : stages)
            require(stage.verifier->batch_resources().graph_keys == expected &&
                    stage.batch_peer->batch_resources().graph_keys == expected,
                    "each physical chain retains only slot zero's hot singleton depths under its own full keys");
        require(opening == fingerprint(), "capture/upload changes no logical session payload or PLE frontier");
        observation = {};
        observation.rejecting = true;
        {
            const core::GraphCaptureObservationScope scope(observe, &observation);
            require(first.prepare_startup_graphs(0, owner_rows, error) &&
                    peer.prepare_startup_graphs(peer_base, owner_rows, error) && observation.attempts == 0,
                    "both physical chains reuse keys when every new capture is denied");
        }
        const int other_base = peer_base;
        const auto before = first.batch_resources();
        first.set_batch_graph_limit(before.graph_keys);
        require(!first.prepare_startup_graphs(other_base, owner_rows, error) &&
                error.find("bounded batch graph cache exhausted") != std::string::npos &&
                first.batch_resources().graph_keys == before.graph_keys,
                "unseen valid base reaches the selected key cap, without preflight refusal or eviction");
        first.set_batch_graph_limit(0);
        error.clear();
        idle();
        observation = {};
        observation.rejecting = true;
        observation.deny = core::GraphCapturePhase::AfterInstantiate;
        {
            const core::GraphCaptureObservationScope scope(observe, &observation);
            require(!first.prepare_startup_graphs(other_base, owner_rows, error) &&
                    observation.attempts == 1 && first.batch_resources().graph_execs == before.graph_execs &&
                    first.batch_resources().graph_keys == before.graph_keys + 1,
                    "unseen full key is not aliased; refusal retains only its bounded null reservation");
        }
        error.clear();
        idle();
        observation = {};
        {
            const core::GraphCaptureObservationScope scope(observe, &observation);
            require(first.prepare_startup_graphs(other_base, owner_rows, error) && observation.retained > 0,
                    "unseen valid keys still capture after explicit refusal is removed");
        }
        observation = {};
        observation.rejecting = true;
        {
            const core::GraphCaptureObservationScope scope(observe, &observation);
            require(first.prepare_startup_graphs(other_base, owner_rows, error) && observation.attempts == 0,
                    "retried alternative-base keys are reusable, not recaptured");
        }
        require(!first.prepare_startup_graphs(0, 5, error), "unsupported preparation extent refuses before mutation");
        error.clear();
        idle();
        require(opening == fingerprint(), "all success/refusal paths preserve every loaded logical owner");
        std::printf("VERIFY_GRAPH_STARTUP_CHECKS checks=%d failures=0 stages=%zu slots=4 owner_rows=%d "
            "logical_cells=0 capture_only=1 key_reuse=1 unseen_base=1 cap_refusal=1 "
            "post_instantiate_refusal=1 ownership_idle=1 peer_chain=1 peer_base=%d\n",
            checks, stages.size(), owner_rows, peer_base);
        return 0;
    } catch (const std::exception& exception) {
        std::fprintf(stderr, "VERIFY_GRAPH_STARTUP_CHECKS checks=%d failures=1 error=%s\n",
                     checks, exception.what());
        return 1;
    }
}

// Host counterpart of the single-device fixture: retain the actual three-stage,
// four-seat allocation and read each payload on its owning device. No inference
// performance or mixed/ring residency claim is made by this lifecycle fixture.
int resident_host_model_checks(strata::program::ResidentSlots& owner, strata::core::ExpertDispatch& dispatch,
                               const strata::core::ModelGeometry& geometry,
                               const std::vector<strata::program::ResidentStage>& stages,
                               strata::core::MtpDrafter& draft) {
    using namespace strata;
    std::string error;
    int checks = 0;
    auto require = [&](bool ok, const char* message) {
        ++checks;
        if (!ok) throw std::runtime_error(std::string(message) + ": " + error);
    };
    auto device = [&](int ordinal) {
        require(cudaSetDevice(ordinal) == cudaSuccess, "select actual payload device");
        require(cudaDeviceSynchronize() == cudaSuccess, "join payload device");
    };
    try {
        require(stages.size() == 3 && owner.slots().size() == 4, "retain three stages and four seats");
        if (std::getenv("STRATA_TEST_PREFILL_SPLIT_OWNER"))
            return prefill_split_owner_checks(owner, *stages[0].main);
        if (std::getenv("STRATA_TEST_MTP_RING_HANDOFF")) {
            const auto* slot = draft.slot_state(0);
            require(draft.kv_state().kv_mode == 0 && slot && slot->kv_mode == 2,
                    "MTP ring handoff fixture needs a resident working owner and ring-backed slot");
            std::puts("MTP_RING_HANDOFF working=resident slot=ring");
        }
        const int64_t cells = stages[0].main->max_cells;
        int64_t boundaries[2] = {stages[0].main->layer_hi, stages[1].main->layer_hi};
        plan::QsaRolePlacements roles;
        roles.working_main.format = core::qsa_kv_format(
            stages[0].main->qsa_states[stages[0].main->qsa_primary()]);
        roles.working_draft.format = core::qsa_kv_format(draft.kv_state());
        plan::ResidentHostBytes host_price;
        require(core::resident_host_payload(geometry, cells, true, roles, stages.size(), boundaries,
                                            core::resident_direct_copy_enabled(), owner.checkpoint_chain_limit(), host_price),
                "price the actual frozen owner formats and stage boundaries");
        auto transfer = [&](const char* operation, auto&& run) {
            host_transfer_scratch.begin();
            bool ok = false;
            try { ok = run(); }
            catch (...) { host_transfer_scratch.active = false; throw; }
            host_transfer_scratch.active = false;
            require(!host_transfer_scratch.overflow && host_transfer_scratch.peak <= host_price.transfer_transient,
                    "actual engine-owned C++ host transfer payload stays inside the admitted transient");
            std::printf("RESIDENT_HOST_SCRATCH operation=%s direct=%d peak_cpp_bytes=%llu bound=%llu "
                        "stages=%zu scope=resident_owner failures=0\n", operation,
                        core::resident_direct_copy_enabled(), (unsigned long long)host_transfer_scratch.peak,
                        (unsigned long long)host_price.transfer_transient, stages.size());
            return ok;
        };
        const std::vector<int32_t> prefix{11, 22};
        const std::vector<int64_t> prompt{11, 22, 33, 44};
        const std::vector<core::ConversationImageKey> images;
        const std::vector<core::ConversationCheckpoint> empty_checks;
        const int64_t index_block = kernels::qsa_real_shapes().idx_block;
        require(index_block > 1, "index block supports aligned and partial cuts");
        const std::vector<int32_t> advanced_prefix(size_t(2 * index_block + 1), 11);
        const std::vector<int64_t> rewind_prompt(advanced_prefix.size() + 1, 11);
        const core::ConversationView diagnostic_view{advanced_prefix, images, empty_checks, true};
        size_t snapshot_bytes = core::conversation_kv_bytes(
            draft.kv_state(), geometry, (int64_t) advanced_prefix.size(), false);
        for (const auto& stage : stages) {
            require(stage.main && stage.slots.size() == 4, "complete stationary stage owners");
            size_t bytes = 0;
            require(core::conversation_snapshot_bytes(diagnostic_view, *stage.main, geometry, nullptr, bytes, error),
                    "price full-model diagnostic image");
            require(bytes <= SIZE_MAX - snapshot_bytes, "bounded diagnostic image sum");
            snapshot_bytes += bytes;
        }
        constexpr uint64_t fixture_allowance = uint64_t{2} << 30;
        uint64_t available = 0, physical = 0;
        // A cap-sized target, one mutable refusal input, the original/peer
        // checkpoints and sequential comparison snapshots coexist. Shared
        // refusal aliases do not duplicate state. Ten image-equivalents beyond
        // the target cap conservatively cover the remaining snapshots/metadata;
        // add the deliberately over-reserved array's entire admitted bound.
        const uint64_t diagnostic_images = uint64_t(owner.checkpoint_chain_limit()) + 10;
        require(owner.checkpoint_byte_limit() < fixture_allowance &&
                snapshot_bytes <= (fixture_allowance - owner.checkpoint_byte_limit()) / diagnostic_images &&
                core::available_memory_bytes(available, nullptr, &physical) &&
                available >= fixture_allowance && physical >= fixture_allowance + (uint64_t{12} << 30),
                "diagnostic owners fit the unchanged 2 GiB allowance above 12 GiB reserve");
        struct Image {
            std::array<core::SavedConversation, 3> main;
            core::ConversationKv draft;
        };
        auto capture_at = [&](int slot, const std::vector<int32_t>& ids) {
            const core::ConversationView at_view{ids, images, empty_checks, true};
            Image result;
            for (size_t i = 0; i < stages.size(); ++i) {
                device(stages[i].device);
                auto* state = slot < 0 ? stages[i].main : stages[i].slots[slot];
                require(core::conversation_snapshot_save(result.main[i], at_view, *state, geometry, nullptr, error),
                        "capture all authoritative stage payloads");
            }
            device(stages.back().device);
            const auto* state = slot < 0 ? &draft.kv_state() : draft.slot_state(slot);
            require(state && core::conversation_kv_save(result.draft, *state, geometry, (int64_t) ids.size(), false, error),
                    "capture actual draft owner on final device");
            return result;
        };
        auto capture = [&](int slot) { return capture_at(slot, prefix); };
        auto equal = [](const Image& a, const Image& b) {
            for (size_t i = 0; i < a.main.size(); ++i)
                if (!same_snapshot(a.main[i], b.main[i])) return false;
            return test::equal(a.draft, b.draft);
        };
        std::array<std::array<const void*, 3>, 3> addresses{};
        for (size_t i = 0; i < stages.size(); ++i)
            addresses[i] = {stages[i].main->gdn_state, stages[i].slots[0]->gdn_state, stages[i].slots[0]->qsa_states};
        auto stationary = [&] {
            for (size_t i = 0; i < stages.size(); ++i)
                require(addresses[i] == std::array<const void*, 3>{
                            stages[i].main->gdn_state, stages[i].slots[0]->gdn_state, stages[i].slots[0]->qsa_states},
                        "all graph-facing owner addresses remain stationary");
        };
        program::ResidentRequest request;
        request.max_new = 8;
        request.scope.value[0] = 1;
        request.sampling.seed = 0;
        request.sampling.penalty_last_n = 2;
        require(!owner.init_resources(error), "repeated initialization refuses");
        const core::ExpertOperation operation{71, 0, 0, 1, 1};
        auto& ledger = dispatch.completion.at(0);
        require(ledger.begin(operation, 0) && ledger.retain_submission(operation), "retain real consumer");
        require(!owner.begin(0, prompt, request, error), "live consumer prevents rebinding");
        ledger.cancel();
        require(!owner.begin(0, prompt, request, error), "cancel is not consumer retirement");
        require(ledger.drain_submission(operation) && ledger.release(operation), "retire retained consumer");
        require(owner.begin(-1, prompt, request, error), "begin working request");
        const auto identity = owner.working().request, seed = owner.working().sampling.seed;
        require(seed != 0, "resolved seed retained across yields");
        std::vector<core::ConversationCheckpoint> checkpoint_chain(1);
        auto& checkpoint = checkpoint_chain.front();
        checkpoint.ids = prefix;
        device(stages[0].device);
        require(core::conversation_checkpoint_save(checkpoint, *stages[0].main, geometry, error),
                "capture first-stage checkpoint");
        checkpoint.stage_parts.resize(stages.size() - 1);
        for (size_t i = 1; i < stages.size(); ++i) {
            device(stages[i].device);
            checkpoint.stage_parts[i - 1].ids = prefix;
            require(core::conversation_checkpoint_save(checkpoint.stage_parts[i - 1], *stages[i].main, geometry, error),
                    "capture every later-stage checkpoint");
        }
        program::ResidentAdmission admission{33, &checkpoint_chain};
        auto opening = capture(-1);
        require(transfer("fresh", [&] { return owner.park(0, prefix, admission, true, error); }),
                "park working request");
        require(equal(opening, capture(0)), "admission preserves every stage and draft payload");
        stationary();
        require(!owner.park(1, prefix, admission, true, error) && !owner.slots()[1].request,
                "one request cannot own two seats");
        require(owner.release_resources(error) && !owner.release_resources(error), "release exactly once");
        require(owner.begin(-1, prompt, request, error) && owner.working().request == identity &&
                owner.working().sampling.seed == seed, "released resend preserves identity and seed");
        require(transfer("live", [&] { return owner.resume(0, nullptr, error); }) && equal(opening, capture(-1)),
                "live resume preserves every stage and draft payload");
        require(owner.init_resources(error) && owner.park(0, prefix, admission, true, error),
                "reinitialize and retain the continuation");
        require(owner.begin(0, prompt, request, error) && owner.working().request == identity &&
                owner.working().sampling.seed == seed &&
                transfer("selected_cut_initial", [&] { return owner.resume(0, &checkpoint, error); }) &&
                equal(opening, capture(-1)), "selected-cut resume preserves payload and identity");
        stationary();
        owner.cancel(0);
        require(owner.begin(0, prompt, request, error) && owner.working().request > identity &&
                owner.working().resume_slot == -1 && owner.admit(0, prefix, admission, error),
                "cancelled identity cannot resurrect in successor");
        owner.cancel(0);
        require(owner.finish(0, error), "finish cancelled successor with no submitted row consumer");
        // Peer admits a disjoint payload owner, not an alias of the retained cuts.
        std::vector<core::ConversationCheckpoint> peer_chain(1);
        peer_chain.front().ids = prefix;
        peer_chain.front().stage_parts.resize(stages.size() - 1);
        for (size_t i = 0; i < stages.size(); ++i) {
            auto& part = i ? peer_chain.front().stage_parts[i - 1] : peer_chain.front();
            part.ids = prefix;
            device(stages[i].device);
            require(core::conversation_checkpoint_save(part, *stages[i].main, geometry, error),
                    "capture independent peer checkpoint");
        }
        program::ResidentAdmission peer_admission{33, &peer_chain};
        require(owner.begin(1, prompt, request, error) && owner.admit(1, prefix, peer_admission, error),
                "admit independent peer");
        opening = capture(1);
        const auto peer_identity = owner.slots()[1].request;
        {
            const size_t limit = (size_t) owner.checkpoint_chain_limit();
            require(limit >= 4, "multi-pin ownership fixture needs the unchanged working cache cap >= 4");
            auto make_cut = [&](size_t ordinal, bool pinned, size_t count) {
                core::ConversationCheckpoint cut;
                cut.ids.assign(prefix.begin(), prefix.begin() + count);
                cut.used = ordinal + 1;
                cut.pinned = pinned;
                cut.stage_parts.resize(stages.size() - 1);
                for (size_t i = 0; i < stages.size(); ++i) {
                    auto& part = i ? cut.stage_parts[i - 1] : cut;
                    part.ids = cut.ids;
                    // Shape comes from the independent native capture. These
                    // opaque synthetic bytes are not inference state; do not
                    // copy a full GPU checkpoint only to overwrite every byte.
                    const auto& shape = i ? checkpoint.stage_parts[i - 1] : checkpoint;
                    const uint8_t marker = uint8_t(1 + ordinal * stages.size() + i);
                    part.gdn.assign(shape.gdn.size(), marker);
                    part.ple.assign(shape.ple.size(), marker);
                    part.tails.assign(shape.tails.size(), marker);
                    part.dead.assign(shape.dead.size(), marker);
                    part.block_pos.assign(shape.block_pos.size(), marker);
                }
                return cut;
            };
            std::vector<core::ConversationCheckpoint> cuts;
            cuts.reserve(limit + 1);
            // Root and latest are separately unpinned; equal-length interior
            // pins remain distinguishable by used, ordering and payload markers.
            cuts.push_back(make_cut(0, false, 1));
            cuts.push_back(make_cut(1, false, prefix.size()));
            for (size_t i = 2; i < limit; ++i) cuts.push_back(make_cut(i, true, prefix.size()));
            // Reserve that is either priced as array backing or not retained
            // must not turn a valid checkpoint into a rejected request.
            cuts.front().gdn.write().reserve(cuts.front().gdn.size() + 1);
            cuts.front().ids.reserve(prefix.size() + 8);
            cuts.front().imgs.reserve(1);
            cuts.front().stage_parts.reserve(stages.size() + 4);
            std::array<std::array<size_t, 5>, 3> expected_sizes{};
            for (size_t i = 0; i < stages.size(); ++i) {
                const auto& part = i ? cuts.front().stage_parts[i - 1] : cuts.front();
                expected_sizes[i] = {part.gdn.size(), part.ple.size(), part.tails.size(),
                                     part.dead.size(), part.block_pos.size()};
            }
            auto oracle = [&](const auto& chain) {
                if (chain.size() != limit) return false;
                for (size_t n = 0; n < limit; ++n) {
                    const auto& cut = chain[n];
                    const std::vector<int32_t> ids(prefix.begin(), prefix.begin() + (n ? prefix.size() : 1));
                    if (cut.ids != ids || cut.used != n + 1 || cut.pinned != (n >= 2) ||
                        cut.stage_parts.size() != stages.size() - 1) return false;
                    for (size_t i = 0; i < stages.size(); ++i) {
                        const auto& part = i ? cut.stage_parts[i - 1] : cut;
                        if (part.ids != ids) return false;
                        const uint8_t marker = uint8_t(1 + n * stages.size() + i);
                        size_t field = 0;
                        for (const auto* bytes : {&part.gdn, &part.ple, &part.tails, &part.dead, &part.block_pos}) {
                            if (bytes->size() != expected_sizes[i][field++]) return false;
                            for (auto byte : bytes->read()) if (byte != marker) return false;
                        }
                    }
                }
                return true;
            };
            require(oracle(cuts), "input membership pins order and synthetic payload match independent oracle");
            program::ResidentAdmission retained{33, &cuts};
            require(!owner.slots()[0].checks.empty(), "old nonempty target overlaps cap-sized alias construction");
            require(owner.begin(0, prompt, request, error) &&
                    transfer("cap_chain", [&] { return owner.park(0, prefix, retained, true, error); }),
                    "bind root/latest and multiple pins under the admitted chain cap");
            cuts.clear();
            cuts.shrink_to_fit();
            require(oracle(owner.slots()[0].checks), "complete input release preserves independently specified chain");
            const auto target_image = capture(0);
            // Compare the same live object's representation, including every
            // request/sampling field and vector descriptor, without payload copies.
            const auto metadata = [](const program::ResidentSlot& s) {
                std::array<unsigned char, sizeof(program::ResidentSlot)> bytes{};
                std::memcpy(bytes.data(), &s, bytes.size());
                return bytes;
            };
            const auto old_metadata = metadata(owner.slots()[0]);
            const auto old_ids = owner.slots()[0].ids, old_prompt = owner.slots()[0].prompt;
            const auto peer_metadata = metadata(owner.slots()[1]);
            const auto unchanged = [&] {
                {
                    const auto now = capture(0);
                    require(metadata(owner.slots()[0]) == old_metadata &&
                            owner.slots()[0].ids == old_ids && owner.slots()[0].prompt == old_prompt &&
                            oracle(owner.slots()[0].checks) && equal(target_image, now),
                            "refusal preserves target metadata device state and retained chain");
                }
                {
                    const auto now = capture(1);
                    require(metadata(owner.slots()[1]) == peer_metadata &&
                            owner.slots()[1].request == peer_identity && equal(opening, now),
                            "refusal preserves the independent peer");
                }
            };
            auto seals_match = [&](bool shared_tail = false) {
                for (size_t n = 0; n < cuts.size(); ++n) for (size_t i = 0; i < stages.size(); ++i) {
                    const auto& part = i ? cuts[n].stage_parts[i - 1] : cuts[n];
                    for (const auto* bytes : {&part.gdn, &part.ple, &part.tails, &part.dead, &part.block_pos})
                        if (bytes->sealed() != (shared_tail && n != 0 && !bytes->empty())) return false;
                }
                return true;
            };
            const auto check_refusal = [&](const char* operation, auto&& run, bool shared_tail = false) {
                const bool rejected = !transfer(operation, run);
                const bool source_preserved = seals_match(shared_tail);
                std::printf("RESIDENT_HOST_REFUSAL operation=%s rejected=%d source_seals_preserved=%d\n",
                            operation, rejected, source_preserved);
                require(rejected, "invalid retained input refuses");
                require(source_preserved, "refusal preserves source sealing");
                unchanged();
            };
            cuts.push_back(make_cut(0, true, prefix.size())); // first source remains mutable
            for (size_t i = 0; i < limit; ++i) {
                cuts.push_back(owner.slots()[0].checks[i]); // share sealed payloads, not another full mutable chain
                cuts.back().pinned = true;
            }
            require(seals_match(true) && owner.begin(0, prompt, request, error), "prepare mixed-ownership over-cap source");
            check_refusal("over_cap_refusal", [&] { return owner.park(0, prefix, retained, true, error); }, true);
            cuts.clear();
            cuts.push_back(make_cut(0, false, prefix.size()));
            cuts.front().stage_parts.front().stage_parts.emplace_back();
            check_refusal("nested_refusal", [&] { return owner.park(0, prefix, retained, true, error); });
            cuts.front().stage_parts.front().stage_parts.clear();
            cuts.front().stage_parts.front().gdn.push_back(0);
            check_refusal("shape_refusal", [&] { return owner.park(0, prefix, retained, true, error); });
            cuts.clear();
            cuts.push_back(make_cut(0, false, prefix.size()));
            auto& oversized = cuts.front().gdn.write();
            oversized.reserve(size_t(owner.checkpoint_byte_limit()) + 1);
            check_refusal("capacity_refusal", [&] { return owner.park(0, prefix, retained, true, error); });
            program::ResidentAdmission no_checks{33, nullptr};
            require(owner.resume(0, nullptr, error) && owner.admit(0, prefix, no_checks, error),
                    "activate parked seat through valid resume and admission");
            owner.cancel(0);
            require(owner.finish(0, error), "finish active retained-chain discriminator");
        }
        // Complete the selected cut's pooled block in a longer-lived owner.
        // No token arithmetic is needed: distinct keys make the restore order
        // observable through the real park/begin/resume and copy_from path.
        for (const int64_t cut : {index_block, index_block + 1}) {
            const std::vector<int32_t> cut_ids(size_t(cut), 11);
            std::vector<core::ConversationCheckpoint> cuts(1);
            auto& selected = cuts.front();
            selected.ids = cut_ids;
            selected.stage_parts.resize(stages.size() - 1);
            require(owner.begin(2, rewind_prompt, request, error), "begin retained-cut discriminator");
            const auto cut_identity = owner.working().request;
            const auto cut_seed = owner.working().sampling.seed;
            for (size_t i = 0; i < stages.size(); ++i) {
                device(stages[i].device);
                auto& state = *stages[i].main;
                require(state.qsa_alloc > 0 && state.max_cells > (int64_t) advanced_prefix.size(),
                        "every stage owns an indexer and the advanced prefix fits");
                const std::vector<float> key(size_t(geometry.idx_key_dim), float(10 + i));
                const int32_t position = int32_t(cut % index_block);
                for (int64_t j = 0; j < state.qsa_alloc; ++j) {
                    const auto& qsa = state.qsa_states[state.qsa_ord0 + j];
                    require(cudaMemcpy(qsa.idx_dead, key.data(), key.size() * sizeof(float),
                                       cudaMemcpyHostToDevice) == cudaSuccess &&
                            cudaMemcpy(qsa.idx_pooled + (cut / index_block) * geometry.idx_key_dim,
                                       key.data(), key.size() * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess &&
                            cudaMemcpy(qsa.idx_block_pos, &position, sizeof(position),
                                       cudaMemcpyHostToDevice) == cudaSuccess,
                            "seed selected checkpoint and its repaired pooled row");
                }
                auto& part = i == 0 ? selected : selected.stage_parts[i - 1];
                part.ids = cut_ids;
                require(core::conversation_checkpoint_save(part, state, geometry, error),
                        "capture selected aligned or partial cut on every stage");
            }
            const auto expected = capture_at(-1, cut_ids);
            for (size_t i = 0; i < stages.size(); ++i) {
                device(stages[i].device);
                auto& state = *stages[i].main;
                const std::vector<float> completed(size_t(geometry.idx_key_dim), float(20 + i));
                const int32_t position = int32_t(advanced_prefix.size() % size_t(index_block));
                for (int64_t j = 0; j < state.qsa_alloc; ++j) {
                    const auto& qsa = state.qsa_states[state.qsa_ord0 + j];
                    require(cudaMemcpy(qsa.idx_pooled + (cut / index_block) * geometry.idx_key_dim,
                                       completed.data(), completed.size() * sizeof(float),
                                       cudaMemcpyHostToDevice) == cudaSuccess &&
                            cudaMemcpy(qsa.idx_block_pos, &position, sizeof(position),
                                       cudaMemcpyHostToDevice) == cudaSuccess,
                            "advance past cut block with a different completed pooled key");
                }
            }
            const program::ResidentAdmission cut_admission{11, &cuts};
            require(transfer("advanced", [&] { return owner.park(2, advanced_prefix, cut_admission, true, error); }) &&
                    owner.slots()[2].ids.size() / size_t(index_block) > size_t(cut / index_block) &&
                    owner.slots()[2].checks.size() == 1,
                    "park advanced owner with its earlier selected checkpoint");
            const auto& retained_cut = owner.slots()[2].checks.front();
            const auto retained = capture_at(2, advanced_prefix);
            auto check_rows = [&](bool restored) {
                for (size_t i = 0; i < stages.size(); ++i) {
                    device(stages[i].device);
                    const auto& state = restored ? *stages[i].main : *stages[i].slots[2];
                    const auto& part = i == 0 ? retained_cut : retained_cut.stage_parts[i - 1];
                    const size_t row_bytes = size_t(geometry.idx_key_dim) * sizeof(float);
                    std::vector<float> row(size_t(geometry.idx_key_dim));
                    for (int64_t j = 0; j < state.qsa_alloc; ++j) {
                        const auto& qsa = state.qsa_states[state.qsa_ord0 + j];
                        require(cudaMemcpy(row.data(), qsa.idx_pooled + (cut / index_block) * geometry.idx_key_dim,
                                           row_bytes, cudaMemcpyDeviceToHost) == cudaSuccess,
                                "read cut pooled row from its actual owning device");
                        const bool selected_row = std::memcmp(row.data(), part.dead.data() + size_t(j) * row_bytes,
                                                               row_bytes) == 0;
                        require(selected_row == restored, restored ?
                                "selected checkpoint dead key wins over advanced pooled row on every stage" :
                                "advanced retained pooled row really differs from selected checkpoint dead key");
                    }
                }
            };
            check_rows(false);
            require(owner.begin(2, rewind_prompt, request, error) &&
                    owner.working().request == cut_identity && owner.working().sampling.seed == cut_seed &&
                    owner.working().resume_slot == 2, "select retained-cut request identity");
            {
                const auto before_refusal = capture_at(-1, advanced_prefix);
                auto invalid_cut = retained_cut;
                invalid_cut.stage_parts.pop_back();
                require(!transfer("invalid_cut", [&] { return owner.resume(2, &invalid_cut, error); }) &&
                        equal(before_refusal, capture_at(-1, advanced_prefix)) &&
                        equal(retained, capture_at(2, advanced_prefix)) &&
                        equal(opening, capture(1)) && owner.slots()[1].request == peer_identity &&
                        owner.slots()[2].request == cut_identity && !owner.slots()[2].failed &&
                        owner.working().request == cut_identity && owner.working().sampling.seed == cut_seed,
                        "invalid cut refuses before writes or identity and peer mutation");
            }
            error.clear(); // the rejected malformed cut above is an expected refusal
            require(transfer("selected_cut", [&] { return owner.resume(2, &retained_cut, error); }),
                    "resume selected cut through ResidentSlots::copy_from");
            check_rows(true);
            require(equal(expected, capture_at(-1, cut_ids)) &&
                    equal(retained, capture_at(2, advanced_prefix)) &&
                    owner.slots()[2].request == cut_identity && !owner.slots()[2].failed &&
                    equal(opening, capture(1)) && owner.slots()[1].request == peer_identity &&
                    owner.slots()[1].active,
                    "cut restore preserves full payload, retained source, identities and independent peer");
            stationary();
            require(owner.admit(2, cut_ids, cut_admission, error),
                    "admit restored selected-cut request before active-slot retirement");
            owner.cancel(2);
            require(owner.finish(2, error), "retire selected-cut discriminator without poisoning reuse");
            std::printf("RESIDENT_CUT_REPAIR direct=%d cut=%lld advanced=%zu stages=%zu failures=0\n",
                        core::resident_direct_copy_enabled(), (long long) cut, advanced_prefix.size(), stages.size());
        }
        require(owner.begin(0, prompt, request, error), "begin replacement before injected failure");
        transfer_failed = false;
        const bool draft_failure = std::getenv("STRATA_TEST_DRAFT_COPY_REFUSAL") != nullptr;
        if (draft_failure) {
            device(stages.back().device);
            const auto* target = draft.slot_state(0);
            require(target && target->kv_mode == 0 && draft.kv_state().kv_mode == 0,
                    "draft refusal exercises canonical resident owners, not a changed ring profile");
            auto payload = [](const core::QsaState& state, bool value) -> void* {
                if (state.kv_q4 || (state.kv_hybrid && value)) return value ? state.v_q4 : state.k_q4;
                if (state.kv_int8 || state.kv_hybrid) return value ? state.v_q : state.k_q;
                return value ? state.v_pool : state.k_pool;
            };
            require(cudaMemset(payload(draft.kv_state(), false), 0x3c, 4) == cudaSuccess &&
                    cudaMemset(payload(*target, false), 0xa5, 4) == cudaSuccess &&
                    cudaMemset(payload(*target, true), 0xa5, 4) == cudaSuccess &&
                    cudaDeviceSynchronize() == cudaSuccess, "mark distinct draft key/value payloads");
            fail_destination = payload(*target, true);
            require(!transfer("draft_refusal", [&] { return owner.admit(0, prefix, admission, error); }) &&
                    transfer_failed, "partial draft admission refuses");
            device(stages.back().device);
            uint32_t key = 0, value = 0;
            require(cudaMemcpy(&key, payload(*target, false), 4, cudaMemcpyDeviceToHost) == cudaSuccess &&
                    cudaMemcpy(&value, payload(*target, true), 4, cudaMemcpyDeviceToHost) == cudaSuccess &&
                    key == 0x3c3c3c3c && value == 0xa5a5a5a5, "draft refusal follows K mutation, before V mutation");
            require(!draft.context_idle(error) && !draft.bind_request(0, 999, error) &&
                    !draft.save_slot(0, 2, error) && !draft.restore_slot(0, 2, error) &&
                    !owner.begin(-1, prompt, request, error), "partial draft poisons all owner reuse");
        } else {
            device(stages[0].device);
            require(cudaMemset(stages[0].main->gdn_state, 0x3c, 4) == cudaSuccess, "mark main source");
            fail_destination = stages[0].slots[0]->ple_hist;
            require(fail_destination &&
                    !transfer("main_refusal", [&] { return owner.admit(0, prefix, admission, error); }) && transfer_failed,
                    "partial main admission refuses");
            device(stages[0].device);
            uint32_t changed = 0;
            require(cudaMemcpy(&changed, stages[0].slots[0]->gdn_state, 4, cudaMemcpyDeviceToHost) == cudaSuccess &&
                    changed == 0x3c3c3c3c, "main refusal follows real destination mutation");
        }
        require(owner.slots()[0].failed && !owner.slots()[0].active && !owner.available(0) &&
                equal(opening, capture(1)) && owner.slots()[1].request == peer_identity && owner.slots()[1].active,
                "partial target is unavailable and every peer stage remains intact");
        if (!draft_failure) {
            owner.cancel(1);
            require(owner.finish(1, error) && owner.release_resources(error) && owner.init_resources(error) &&
                    !owner.available(0) && owner.available(1), "reinitialize only valid contexts");
            require(owner.begin(1, prompt, request, error), "select peer for working-resume failure");
            transfer_failed = false;
            fail_destination = stages[0].main->ple_hist;
            require(!transfer("live_refusal", [&] { return owner.resume(1, nullptr, error); }) && transfer_failed &&
                    !owner.begin(-1, prompt, request, error) && !owner.available(1),
                    "partial working resume poisons reuse");
            require(owner.release_resources(error), "release fixture resources");
        }
        std::printf("RESIDENT_HOST_MODEL_CHECKS stages=3 seats=4 checks=%d draft_failure=%d failures=0\n",
                    checks, draft_failure);
        return 0;
    } catch (const std::exception& exception) {
        fail_destination = nullptr;
        std::fprintf(stderr, "RESIDENT_HOST_MODEL_CHECKS checks=%d failures=1 error=%s\n", checks, exception.what());
        return 1;
    }
}
