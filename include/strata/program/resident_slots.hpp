// Resident request transitions. SessionOwner, Verifier and MTP retain resource ownership.
#pragma once
#include "strata/core/conversation_snapshot.hpp"
#include "strata/core/neural_drafter.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/program/prefix_selection.hpp"
#include "strata/program/continuation.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace strata::core { class Verifier; struct ExpertDispatch; }
namespace strata::prefill { class Prefill; }
namespace strata::program {

struct ResidentStage {
    core::SessionState* main = nullptr;
    core::Verifier* verifier = nullptr;
    std::vector<core::SessionState*> slots;
    int device = -1;
    // Optional second group owner: the same private slot states, distinct retained replay scratch.
    core::Verifier* batch_peer = nullptr;
    // Optional third chain owns its own replay and two worker ledgers per stage lane.
    core::Verifier* batch_third = nullptr;
    int third_worker_lane = -1;
};

struct ResidentRequest {
    kernels::SamplerParams sampling; // seed 0 resolves once, retained across yield
    int64_t max_new = 0;
    bool steering = true, images = false;
    double pcie_fraction = 0, spec_min_probability = 0;
    core::ConversationScope scope;
    core::ConversationScope shared_scope;
    int64_t pin = 0;
    bool checkpoint = true;
};

struct ResidentWorking {
    uint64_t request = 0;
    kernels::SamplerParams sampling;
    int resume_slot = -1;
    int64_t prefill_split_until = 0; // exact full-slot continuation; consumed by the prompt loop
    core::ConversationCheckpoint::PromptPlan prefill_plan; // the piped read's plan, carried across a BYIELD
};

// The post-prefill request setup installs the resident owner's already-resolved policy.
// Setter-only targets let the CPU regression exercise this production seam without a device.
template<class Verifier, class Drafter>
void apply_request_sampling(const ResidentWorking& owner, Verifier& verifier, Verifier* peer, Drafter* draft) {
    verifier.set_sampling(owner.sampling);
    if (peer) peer->set_sampling(owner.sampling);
    if (draft) draft->set_draft_sampling(owner.sampling);
}

// Value-only continuation of private preparation. The resident seat owns it;
// scratch/events stay with Prefill, whose complete-chain drain precedes suspension.
struct ResidentPromptCursor {
    core::ConversationCheckpoint::PromptPlan plan;
    uint64_t plan_id = 0, operation = 0;
    int64_t next = 0, split_until = 0;
    bool suspended = false;
    bool windows = false; // original short segment: known-input private verifier rows, never sampled output

    int64_t interval_end() const {
        if (plan.chunk <= 0 || next < plan.from || next >= plan.to) return next;
        return split_until > next ? std::min(next + 1024, split_until) :
            next + std::min(plan.chunk, plan.to - next);
    }
    bool matches(const ResidentPromptCursor& other) const {
        return plan_id && plan_id == other.plan_id && operation == other.operation &&
            next == other.next && split_until == other.split_until &&
            plan.from == other.plan.from && plan.to == other.plan.to && plan.chunk == other.plan.chunk &&
            plan.lend == other.plan.lend && plan.pipe == other.plan.pipe &&
            plan.single == other.plan.single && plan.chunked == other.plan.chunked && windows == other.windows;
    }
    bool valid(int64_t frontier, int64_t endpoint) const {
        if (!plan_id || plan.chunk < 1 || plan.from < 0 || plan.from > frontier ||
            plan.to <= plan.from || plan.to > endpoint || frontier > plan.to || next != frontier) return false;
        if (split_until)
            return plan.chunk == 2048 && split_until > frontier && split_until <= plan.to &&
                (split_until - frontier == 1024 || split_until - frontier == 2048) &&
                (split_until - plan.from) % plan.chunk == 0;
        return frontier == plan.to || (frontier - plan.from) % plan.chunk == 0;
    }
};

// Read-only to the scheduler. The vector objects and graph-referenced sessions
// never move while bound. Token storage is reserved once per admitted slot.
struct ResidentSlot {
    bool active = false, preparing = false, stop = false, foreground = true;
    kernels::SamplerParams sampling;
    uint64_t request = 0;
    uint64_t incarnation = 0;
    int32_t x = 0;
    core::DraftProposal proposal;
    int64_t p = 0, produced = 0, max_new = 0;
    std::chrono::steady_clock::time_point t0;
    std::vector<int32_t> ids;
    std::vector<int32_t> prompt; // yielded full prompt; second seat-owned token vector
    ResidentRequest input;
    int original_slot = -1;
    bool cached = false, cvec = true, img = false;
    std::vector<core::ConversationCheckpoint> checks;
    bool partial = false, partial_from0 = false;
    int64_t prefill_split_until = 0; // pending second 1024 of the original selected 2048-token interval
    core::ConversationCheckpoint::PromptPlan prefill_plan; // pending plan of a partial handoff; consumed on resume
    ResidentPromptCursor prompt_cursor; // retained private preparation; never a legacy parked alias
    bool failed = false;
    // #879: a window met non-finite logits while this slot was running: its state may have started from the
    // poisoned state (a checkpoint, a parked conversation), so it is not kept as a cache when it ends.
    bool tainted = false;

    // Called only after the operation owner certifies this row's drain.
    void complete(bool cache_enabled) {
        active = preparing = false;
        prefill_split_until = 0;
        prefill_plan = {};
        prompt_cursor = {};
        cached = cache_enabled && input.checkpoint && input.scope.valid() && !img && !failed && !tainted;
    }
    // Metadata handoff after a successful drained restore. Foreign identities
    // cannot claim it; a rewind checkpoint discards the abandoned interval.
    void restored(ResidentWorking& working, int slot, bool full_prefix) {
        if (working.resume_slot != slot || working.request != request) return;
        const int64_t pending = std::exchange(prefill_split_until, 0);
        const core::ConversationCheckpoint::PromptPlan carried = std::exchange(prefill_plan, {});
        working.prefill_split_until = full_prefix ? pending : 0;
        working.prefill_plan = full_prefix ? carried : core::ConversationCheckpoint::PromptPlan{};
        partial = false;
    }
    bool reusable(const core::ConversationScope& scope, bool steering) const {
        return !active && !preparing && !failed && cached && request && cvec == steering && input.scope.matches(scope);
    }
};

// Borrowed until the next resident mutation. Capture must return the checkpoint
// chain after split/merge; private GPU state and owner identity stay read-only.
class ResidentSnapshotSource {
public:
    const ResidentSlot& owner;
    std::vector<core::ConversationCheckpoint>& checkpoints;
    const core::QsaState* draft;
    core::NeuralDrafter& neural;
    const int draft_slot;
    const int64_t draft_begin, draft_cells;
    const int32_t draft_next_token;
    // The certified committed cut this source may publish. Speculative scratch is not part of it.
    ContinuationCertificate certificate;

    const core::SessionState& session(size_t stage) const {
        return *stages_[stage].slots[(size_t) slot_];
    }
private:
    friend class ResidentSlots;
    ResidentSnapshotSource(ResidentSlot& owner, const std::vector<ResidentStage>& stages,
                           core::NeuralDrafter& neural, const core::QsaState* draft, int slot,
                           int64_t begin, int64_t cells, int32_t next)
        : owner(owner), checkpoints(owner.checks), draft(draft), neural(neural), draft_slot(slot),
          draft_begin(begin), draft_cells(cells), draft_next_token(next), stages_(stages), slot_(slot) {}
    const std::vector<ResidentStage>& stages_;
    int slot_;
};

struct ResidentAdmission {
    int32_t next_token = 0;
    // Eligible cuts must be flat and stage-shape-valid. Retained array capacity,
    // copied IDs and owner metadata must fit the preadmitted per-checkpoint bytes;
    // caller-only vector reserve is not retained. Root/latest/all pins must fit
    // the admitted chain cap before source freezing or target mutation.
    std::vector<core::ConversationCheckpoint>* checkpoints = nullptr;
    const float* residual = nullptr;
    int64_t checkpoint_limit = -1;   // inclusive; -1 uses the admitted prefix length
    int64_t prefill_split_until = 0; // park only; zero once the selected interval has finished
    core::ConversationCheckpoint::PromptPlan prefill_plan; // park only: the plan of the segment being parked
};

// A physical seat incarnation, not an ambient working-session handle. Tokens
// and stage destinations remain stable until first_token or abort_preparation.
struct ResidentPreparation {
    int slot = -1;
    uint64_t request = 0, incarnation = 0;
    int64_t position = 0, endpoint = 0;
    std::array<core::SessionState*, 4> stages{};
    std::array<uint64_t, 4> allocations{}; // exact independently owned stage allocation generations
};

struct ResidentPrepareInput {
    int slot = -1;
    const std::vector<int64_t>* tokens = nullptr;
    ResidentRequest request;
    const core::SavedConversation* snapshot = nullptr;
    const core::ConversationCheckpoint* cut = nullptr;
    core::ConversationSnapshotExpectation expectation = core::ConversationSnapshotExpectation::ordinary;
};

struct ResidentIdentity {
    uint64_t request = 0, incarnation = 0; // request 0 means the seat is unbound
    bool owns(uint64_t request_id, uint64_t seat_incarnation) const {
        return request && incarnation && request == request_id && incarnation == seat_incarnation;
    }
};

// Only these pre-write source refusals permit retrying the original reservation.
enum class ResidentSourceRefusal { none, uncertified_interval, incoherent_boundary };

// One host submitter, the same as Verifier. Scheduling and protocol emission
// remain outside; callers never rebind identity/history/dispatch/MTP separately.
// Failed preflight leaves every slot unchanged. Failure after mutation invalidates
// only the target; a poisoned resource owner continues to refuse subsequent work.
class ResidentSlots {
public:
    ResidentSlots(const core::ModelGeometry& geometry, std::vector<ResidentStage> stages,
                  core::NeuralDrafter& draft, prefill::Prefill& prompt, core::ExpertDispatch& dispatch,
                  int64_t context, bool cache, int64_t checkpoint_chain_limit, uint64_t checkpoint_byte_limit);
    ~ResidentSlots();
    ResidentSlots(const ResidentSlots&) = delete;
    ResidentSlots& operator=(const ResidentSlots&) = delete;
    ResidentSlots(ResidentSlots&&) = delete;
    ResidentSlots& operator=(ResidentSlots&&) = delete;
    const std::vector<ResidentSlot>& slots() const { return slots_; }
    int64_t checkpoint_chain_limit() const { return checkpoint_chain_limit_; }
    uint64_t checkpoint_byte_limit() const { return checkpoint_byte_limit_; }
    // Bind the working verifier, prefill, draft and dispatch together. An exact
    // yielded resend or solo-to-parked-slot promotion retains identity and seed.
    bool begin(int destination, const std::vector<int64_t>& prompt, const ResidentRequest& request,
               std::string& error);
    const ResidentWorking& working() const { return working_; }
    int64_t take_prefill_split_until() { return std::exchange(working_.prefill_split_until, 0); }
    // The plan of the read being resumed after a BYIELD, so it goes on in the same chunks (read_part).
    core::ConversationCheckpoint::PromptPlan take_prefill_plan() { return std::exchange(working_.prefill_plan, {}); }
    bool active() const;
    bool available(int slot) const;
    // Read-only wire identity; nonexistent seats return an unbound identity.
    ResidentIdentity identity(int slot) const;
    bool owns(int slot, uint64_t request, uint64_t incarnation) const;
    // One committed cut across token history and every required state family, with draft readiness separate.
    // expected_* nonzero requires the cut to belong to that generation (a stale generation is refused).
    ContinuationCertificate certify_continuation(int slot, const core::ConversationCheckpoint* cut = nullptr,
                                                 uint64_t expected_request = 0,
                                                 uint64_t expected_incarnation = 0,
                                                 std::string* error = nullptr) const;
    // Diagnostic only: the next resume_work refuses as a stale generation. Used to exercise the
    // request-local continuation refusal and peer isolation on device; never set by a normal caller.
    void inject_stale_continuation() { stale_continuation_ = true; }
    // Diagnostic only: remove only the chosen seat's optional draft certificate, not target state/KV.
    void inject_absent_draft(int slot);
    void cancel(int slot);
    /// #879: the running slot's state may hold the poison: `complete` must not keep it as a cache when it ends.
    void taint(int slot);
    /// #879 for a whole batch: taint every running slot and drop every idle slot's cached conversation
    /// (`cached=false`, `partial=false`, `checks.clear()`).  Returns (running, idle-cleared) for the report.
    std::pair<size_t, size_t> invalidate_after_nonfinite();
    void priority(int slot, bool foreground);
    bool init_resources(std::string& error);
    bool release_resources(std::string& error);
    bool admit(int slot, const std::vector<int32_t>& prefix, const ResidentAdmission& request, std::string& error);
    // CPU-only, all-or-none reservation of distinct seats. Publish protocol ACKs
    // after reserve, before prepare performs any destination write.
    bool reserve(const ResidentPrepareInput* inputs, int count, ResidentPreparation* leases, std::string& error);
    bool prepare(const ResidentPreparation& lease, std::string& error);
    bool prepare(const ResidentPreparation& lease, const core::SavedConversation* snapshot,
                 const core::ConversationCheckpoint* cut, core::ConversationSnapshotExpectation expectation,
                 std::string& error, ResidentSourceRefusal* refusal = nullptr);
    bool prepare(int slot, const std::vector<int64_t>& tokens, const ResidentRequest& request,
                 ResidentPreparation& lease, std::string& error);
    // Drained segmented work publishes only its certified prefix. The final
    // prompt token belongs to a separate verifier row, followed by first_token.
    bool prefilled(const ResidentPreparation& lease, int64_t frontier, std::string& error);
    bool checkpoint(const ResidentPreparation& lease, core::ConversationCheckpoint& cut, std::string& error);
    bool first_token(const ResidentPreparation& lease, int32_t next, const float* residual, std::string& error);
    bool abort_preparation(const ResidentPreparation& lease, std::string& error);
    // Plan installation and suspend/resume are preflight-only until every owner,
    // allocation, frontier and draft check succeeds. Refusal never taints a peer.
    bool plan_prompt(const ResidentPreparation& lease, const ResidentPromptCursor& cursor, std::string& error);
    bool suspend_prompt(const ResidentPreparation& lease, uint64_t operation,
                        ResidentPromptCursor& cursor, std::string& error);
    bool resume_prompt(const ResidentPreparation& lease, const ResidentPromptCursor& cursor, std::string& error);
    bool park(int slot, const std::vector<int32_t>& prefix, const ResidentAdmission& request,
              bool from_start, std::string& error);
    // No transfers, synchronization or owner mutation. Prices the selected
    // copy path and prevalidates all checkpoint parts before any working-state write.
    bool resume_work(int slot, const core::ConversationCheckpoint* checkpoint,
                     PrefixWork& work, std::string& error) const;
    // Restores private model state into the working session. A partial restore
    // returns false; it is never advertised as a usable prefix by this interface.
    bool resume(int slot, const core::ConversationCheckpoint* checkpoint, std::string& error);
    // Requires drained consumers; never mounts foreign state into the working request.
    // Ineligible sources return nullopt with no error; capture uses the source's
    // own scope and must restore its borrowed checkpoint chain before replacement.
    std::optional<ResidentSnapshotSource> completed_snapshot_source(int slot, std::string& error);
    // Preparation-only borrowed source at the last certified, drained frontier.
    // A live lease must still own every physical stage and coherent draft row;
    // stopped, tainted or unready owners are clean misses. No completed owner is
    // made eligible here, and capture must not advance/mutate the preparation.
    std::optional<ResidentSnapshotSource> prepared_snapshot_source(const ResidentPreparation& lease,
                                                                  std::string& error);
    // Called only for a GPU-committed row. No state migration or allocation.
    bool committed(int slot, int32_t next, const float* residual, std::string& error, bool observe = true);
    bool propose(int slot, const float* residual, std::string& error);
    bool finish(int slot, std::string& error, bool diagnostic = true);
private:
    friend class ResidentExecutor;
    friend struct ResidentExecutorFixture;
    friend struct ResidentDraftReadinessFixture;
    template<class Token>
    bool draft_ready(int slot, const core::ConversationCheckpoint* cut, const std::vector<Token>& tokens,
                     uint64_t expected_request, uint64_t expected_incarnation, std::string& error) const;
    enum class Resources { released, ready, failed };
    struct PreparationSource {
        const core::SavedConversation* image = nullptr;
        const core::ConversationCheckpoint* cut = nullptr;
        core::ConversationSnapshotExpectation expectation = core::ConversationSnapshotExpectation::ordinary;
        int rewind = -1;
        int32_t tail_token = -1;
        bool retained = false, ready = false, mutated = false;
    };
    bool preparation_valid(const ResidentPreparation& lease, std::string& error) const;
    bool validate_checkpoint(const core::ConversationCheckpoint& cut, int slot, std::string& error) const;
    template<class Token>
    bool select_checkpoints(const std::vector<core::ConversationCheckpoint>& cuts,
                            const std::vector<Token>& tokens, int64_t limit, int slot,
                            std::vector<core::ConversationCheckpoint>& selected, std::string& error) const;
    template<class Token>
    bool validate_source(const core::SavedConversation& image, const core::ConversationCheckpoint*& cut,
                         core::ConversationSnapshotExpectation expectation, int slot,
                         const std::vector<Token>& tokens, const ResidentRequest& request,
                         std::vector<core::ConversationCheckpoint>& selected, int64_t& position,
                         std::string& error, ResidentSourceRefusal* refusal = nullptr) const;
    bool idle(std::string& error);
    bool slot_idle(int slot, std::string& error);
    bool retained(int slot) const;
    bool bind(int slot, const std::vector<int32_t>& prefix, const ResidentAdmission& request,
              bool parked, bool from_start, std::string& error);
    bool copy_to(int slot, const std::vector<int32_t>& prefix, std::string& error);
    bool copy_from(int slot, const core::ConversationCheckpoint* checkpoint, std::string& error);
    bool observe(const char* phase, int slot, int64_t cells, std::string& error, const float* residual = nullptr);
    void residency(const char* phase, int slot) const;
    void retire(int slot);
    bool fail(int slot);
    const core::ModelGeometry& geometry_;
    std::vector<ResidentStage> stages_;
    core::NeuralDrafter& draft_;
    prefill::Prefill& prompt_;
    core::ExpertDispatch& dispatch_;
    std::vector<ResidentSlot> slots_;
    int64_t context_;
    std::vector<PreparationSource> preparations_;
    int64_t checkpoint_chain_limit_;
    uint64_t checkpoint_byte_limit_;
    ResidentWorking working_;
    ResidentRequest input_;
    const std::vector<int64_t>* working_prompt_ = nullptr; // borrowed only through this request's park/admit
    uint64_t next_request_ = 0;
    int destination_ = -1;
    bool working_failed_ = false;
    mutable bool stale_continuation_ = false;
    bool cache_, touched_ = false;
    Resources resources_ = Resources::released;
};
} // namespace strata::program
