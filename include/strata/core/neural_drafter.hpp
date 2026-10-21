#pragma once

#include "strata/core/conversation_snapshot.hpp"
#include "strata/core/draft_state.hpp"
#include "strata/core/operation_canary.hpp"
#include "strata/core/operation_lease.hpp"
#include "strata/core/spec_prob.hpp"
#include "strata/kernels/sampler.hpp"

#include <array>
#include <span>

namespace strata::core {
inline constexpr int kNeuralDraftMax = kernels::kVerifyMaxT - 1;

struct DraftProposal {
    enum class Distribution : uint8_t { point_mass, conditional_q };
    DraftOwner owner;
    int64_t frontier = -1;
    int count = 0;
    std::array<int32_t, kNeuralDraftMax> tokens{};
    // Draft-family chosen-token confidence, not calibrated target acceptance.
    std::array<float, kNeuralDraftMax> confidence{};
    Distribution distribution = Distribution::point_mass;
    std::array<int32_t, kNeuralDraftMax * kSpecQStride> q{};
    void clear() { count = 0; frontier = -1; tokens.fill(-1); distribution = Distribution::point_mass; }
    const int32_t* probability_rows() const {
        return count && distribution == Distribution::conditional_q ? q.data() : nullptr;
    }
    bool validate(int64_t vocabulary, std::string& error, DraftKind kind = DraftKind::none) const;
};

struct DraftSlotState {
    int64_t cells = 0, completed_cells = -1, begin = 0, prompt_len = 0;
    bool certified = false;
    int32_t tail_token = -1;
    uint64_t request = 0;
    DraftOwner producer;
    // Numerical adapters may cache a completed point-mass row. Its ownership and
    // lifetime are checked by the common owner before publishing a proposal.
    int32_t proposal = -1;
    int64_t proposal_cell = -1;
    int32_t proposal_token = -1;
    float proposal_confidence = 0;
};

// The single lifecycle owner for both numerical adapters. Every source read,
// context projection, transfer and proposal is enclosed in this same bounded
// lease. A failed operation drains, invalidates coverage and poisons the lease;
class NeuralDrafter {
public:
    class Operation;
    virtual ~NeuralDrafter();
    NeuralDrafter(const NeuralDrafter&) = delete;
    NeuralDrafter& operator=(const NeuralDrafter&) = delete;
    DraftKind kind() const { return identity_.kind; }
    const DraftIdentity& identity() const { return identity_; }
    virtual bool enabled() const = 0;
    int device() const { return device_; }
    int64_t window() const { return window_; }
    int64_t origin(int slot = 8) const { return snapshot_origin(slot); }
    bool batch_drafting() const { return batch_drafting_; }
    void set_batch_drafting(bool on) { batch_drafting_ = on; }
    virtual void set_prompt_len(int64_t n) { prompt_len_ = n; }
    virtual void set_draft_sampling(const kernels::SamplerParams&) = 0;
    virtual uint64_t vram_bytes() const = 0;
    bool set_slot_sampling(int slot, const kernels::SamplerParams&, std::string& error);
    virtual DraftAllocationPlan allocation_plan() const = 0;

    bool bind_request(int slot, uint64_t request, std::string& error, uint64_t incarnation = 0);
    DraftOwner owner(int slot = 8) const;
    bool owns(const DraftOwner&) const;
    bool reserve_inputs(int slot, const DraftOwner& expected, int64_t start, int64_t count,
                         DraftOwner& ticket, std::string& error);
    virtual bool discard_inputs(const DraftOwner& ticket, std::string& error);
    bool inputs_completed(int slot, int64_t start, int64_t count) const;
    bool inputs_pending(int slot) const {
        return slot >= 0 && slot <= 8 && pending_inputs_[size_t(slot)].active;
    }
    bool idle(std::string& error);
    bool context_idle(std::string& error);
    bool scratch_idle(std::string& error, bool wait);
    bool slot_idle(int slot, std::string& error, bool wait = false);
    bool retire(int slot, uint64_t request, uint64_t incarnation, std::string& error);
    // CPU-only reservation fence. Keep actual retained storage/certification for
    // a separately prevalidated prefix handoff, but unbind the old producer.
    bool release_identity(int slot, uint64_t request, uint64_t incarnation, std::string& error);
    virtual void invalidate_coverage();
    void invalidate_slot_coverage(int slot);
    DraftCoverage slot_coverage(int slot) const;
    ConversationDraftCoverage completed_coverage() const {
        return async_owned_ || pending_inputs_[8].active ? ConversationDraftCoverage{} : coverage_;
    }
    int64_t slot_cells(int slot) const;
    int64_t slot_begin(int slot) const;
    int64_t slot_prompt_len(int slot) const;
    int32_t slot_next_token(int slot) const;
    int64_t slot_resume_frontier(int slot, int64_t cells) const;
    bool slot_covers(int slot, int64_t cells, int64_t prompt_length = 0) const;
    bool cached_proposal(int slot, int32_t anchor, int64_t cell, DraftProposal&, std::string&) const;
    virtual bool state_fingerprint(int slot, int64_t cells, uint64_t&, std::string&);
    void restore_boundary_token(int slot, int32_t token);
    void certify_view(ConversationView&, int slot = 8, const DraftOwner* displaced_source = nullptr) const;
    void restored_snapshot(const SavedConversation&, int64_t cells);
    void set_boundary_capture(int64_t position);
    bool capture_boundary(const float*, int64_t position, uint64_t request, std::string& error);

    virtual bool init_slots(const std::vector<SessionState*>&, std::string&,
                            const plan::QsaPlacement* placement = nullptr) = 0;
    virtual bool save_slot(int slot, int64_t cells, std::string&) = 0;
    virtual bool restore_slot(int slot, int64_t cells, std::string&) = 0;
    virtual bool prepare_slot(int slot, uint64_t request, int64_t prompt_length,
                              int64_t cells, bool reset, std::string&) = 0;
    virtual bool prefill_slot(int slot, const float* residuals, const int32_t* next_tokens,
                              int64_t rows, int64_t position, std::string&) = 0;
    virtual bool advance_slot(int slot, const float* residual, int32_t next_token,
                              int64_t cell, std::string&) = 0;
    bool propose_slot(int slot, const float* residual, int32_t anchor, int64_t cell,
                       DraftProposal&, std::string& error);
    // Existing MTP storage remains available only as an explicitly MTP adapter
    // view. DFlash never allocates or impersonates a target-shaped QsaState.
    virtual const QsaState* legacy_state(int slot = 8) const = 0;
    virtual uint64_t copy_bytes(int slot, int64_t cells, bool direct, std::string&) const = 0;
    virtual bool snapshot_capture(SavedConversation&, Operation&, std::string&) = 0;
    virtual bool snapshot_preflight(const SavedConversation&, int slot, int64_t cells,
                                     ConversationSnapshotExpectation, std::string&) const = 0;
    virtual bool snapshot_restore(const SavedConversation&, Operation&, int64_t cells,
                                   int64_t prompt_length, int32_t next_token,
                                   ConversationSnapshotExpectation, std::string&) = 0;

    class Operation {
    public:
        Operation(NeuralDrafter&, std::string&, int slot = 8, cudaStream_t external = nullptr,
                  int32_t token = 0, int32_t position = 0, const DraftOwner* expected = nullptr,
                  const DraftOwner* borrowed_source = nullptr, bool read_only = false);
        ~Operation();
        explicit operator bool() const { return acquired_ && !finished_; }
        bool finish(std::string& error);
        void writes(int64_t begin, int64_t end, int32_t next_token);
        bool boundary(const float*, int64_t begin, int64_t count, std::string&);
        void detach() { finished_ = true; owner_.async_owned_ = true; lease_.complete(); }
        const DraftOwner& identity() const { return identity_; }
        const OperationLease::Guard& guard() const { return lease_; }
    private:
        NeuralDrafter& owner_;
        OperationLease::Guard lease_;
        DraftOwner identity_;
        bool acquired_ = false, finished_ = false, read_only_ = false;
        int slot_ = 8;
        cudaStream_t external_ = nullptr;
    };

protected:
    NeuralDrafter() = default;
    virtual DraftSlotState* slot_metadata(int) = 0;
    virtual const DraftSlotState* slot_metadata(int) const = 0;
    virtual int64_t state_capacity(int slot) const = 0;
    virtual int64_t snapshot_origin(int) const { return 0; }
    virtual bool proposal_preflight(int, const float*, int32_t, int64_t, std::string&) const = 0;
    virtual bool propose_impl(Operation&, const float*, int32_t, int64_t, DraftProposal&, std::string&) = 0;
    virtual bool inspect_proposal(Operation&, const DraftProposal&, std::string&) { return true; }
    virtual bool drain_adapter(std::string&) = 0;
    bool idle_owned(std::string&);
    void complete_writes();
    bool discard_inputs_owned(const DraftOwner&, std::string&);
    const char* resource_name() const { return kind() == DraftKind::dflash2 ? "dflash2" : "mtp"; }
    DraftIdentity identity_;
    OperationLease scratch_;
    OperationCanary canary_;
    const ModelGeometry* g_ = nullptr;
    int device_ = -1;
    cudaStream_t cs_ = nullptr;
    bool chain_live_ = false, batch_drafting_ = false, async_owned_ = false;
    ConversationDraftCoverage coverage_, pending_coverage_;
    uint64_t working_request_ = 0, coverage_owner_ = 0;
    DraftOwner coverage_identity_;
    int pending_slot_ = 8;
    bool pending_writes_ = false, pending_boundary_ = false;
    int64_t boundary_capture_ = -1, boundary_position_ = -1;
    uint64_t boundary_owner_ = 0;
    ConversationBuffer boundary_residual_;
    int64_t window_ = 0, prompt_len_ = 0;
    std::array<DraftOwner, 9> owners_{};
    DraftOwner pending_owner_;
    std::array<kernels::SamplerParams, 9> sampling_{};
    struct PendingInputs { DraftOwner ticket; int64_t start = 0, count = 0; bool active = false; };
    std::array<PendingInputs, 9> pending_inputs_{};
    bool inputs_pending() const {
        for (const auto& input : pending_inputs_) if (input.active) return true;
        return false;
    }
};

// Target-only validation/copy still goes through ConversationBuffer and the
// existing conversation pipeline. The draft adapter preflights BEFORE any
// target write, then restores its own kind-specific layers exactly once.
bool conversation_neural_read_limits(SessionReadLimits&, const SessionState&, const ModelGeometry&,
                                     const NeuralDrafter&, int slot, uint64_t max_tokens,
                                     uint64_t max_checkpoints, std::string& error);
bool conversation_neural_save(SavedConversation&, ConversationView, const SessionState&, const ModelGeometry&,
                              NeuralDrafter&, int slot, std::string&, ConversationKvReuse reuse = {},
                              size_t* reused_bytes = nullptr,
                              std::vector<ConversationRetainedPrefix>* retained = nullptr,
                              const DraftOwner* expected = nullptr);
bool conversation_neural_validate(const SavedConversation&, const SessionState&, const ModelGeometry&,
                                  const NeuralDrafter&, int slot, ConversationSnapshotExpectation,
                                  std::string& error);
ConversationRestore conversation_neural_restore(const SavedConversation&, SessionState&, const ModelGeometry&,
                                                NeuralDrafter&, int slot, int64_t prompt_length,
                                                int32_t next_token, ConversationSnapshotExpectation,
                                                std::string&, const ConversationCheckpoint* selected = nullptr,
                                                const DraftOwner* expected = nullptr);
} // namespace strata::core
