#pragma once

#include "strata/kernels/dflash2_selector.hpp"

#include "strata/core/neural_drafter.hpp"
#include "strata/core/dflash2_capture.hpp"
#include "strata/core/dflash2_target.hpp"

#include <memory>

namespace strata::core::dflash2 {
class RuntimeWeights;
class DeviceForward;
class DeviceSelector;

struct ProposalCapture {
    DraftOwner owner;
    int origin = 0, frontier = 0;
    int32_t anchor = -1;
    FeatureView context;
    const uint16_t* raw_embeddings_device = nullptr; // four raw BF16 rows
    const uint16_t* hidden_device = nullptr;         // four normalized BF16 rows
    const float* logits_device = nullptr;           // three full FP32 vocabulary rows
    DraftProposal proposal;
    std::array<std::array<int32_t, 16>, kernels::dflash2::selector_rows> candidates{};
    std::array<std::array<float, 16>, kernels::dflash2::selector_rows> conditional_probabilities{};
    float temperature = 0;
    std::array<float, 3> uniforms{};
    int device = -1;
    void* stream = nullptr;
};
using ProposalObserver = bool (*)(void*, const ProposalCapture&, std::string&);
// Direct-device native target borrowing: the real loaded embedding/head and
// verified receipts outlive this adapter. No shadow target/MTP or host logits
// bridge is loaded. All contexts/rings/scratch are allocated before admission.
class DflashNativeDrafter final : public NeuralDrafter {
public:
    DflashNativeDrafter(const RuntimeWeights&, VerifiedDailyTarget,
                        const NativeEmbed&, const NativeHead&, int device);
    ~DflashNativeDrafter() override;
    bool enabled() const override { return true; }
    uint64_t vram_bytes() const override;
    DraftAllocationPlan allocation_plan() const override;
    // Allocations made on embedding/head GPUs, if distinct from the drafter.
    DraftAllocationPlan target_allocation_plan(int device) const;
    static DraftAllocationPlan planned_allocation(int device, size_t dense_bytes,
                                                  int embedding_device, int head_device,
                                                  size_t head_scratch_bytes);
    static DraftAllocationPlan planned_target_allocation(int device, int drafter_device,
        int embedding_device, int head_device, size_t head_scratch_bytes);
    bool init_slots(const std::vector<SessionState*>&, std::string&,
                    const plan::QsaPlacement* = nullptr) override;
    bool prepare_slot(int, uint64_t, int64_t, int64_t, bool, std::string&) override;
    bool save_slot(int, int64_t, std::string&) override;
    bool restore_slot(int, int64_t, std::string&) override;
    bool prefill_slot(int, const float*, const int32_t*, int64_t, int64_t, std::string&) override;
    bool advance_slot(int, const float*, int32_t, int64_t, std::string&) override;
    const QsaState* legacy_state(int = 8) const override { return nullptr; }
    uint64_t copy_bytes(int, int64_t, bool, std::string&) const override;
    bool state_fingerprint(int slot, int64_t cells, uint64_t&, std::string&) override;
    bool discard_inputs(const DraftOwner&, std::string&) override;
    void invalidate_coverage() override;
    // Optional synchronous diagnostic observer, OFF by default. All borrowed
    // views expire when the callback returns. Refusal poisons its real operation.
    void set_proposal_observer(ProposalObserver observer, void* user);

    // Hold Operation across target capture, all five feature transfers and
    // context projection. The producer owns THIS request's committed ring and
    // borrows the common tentative arena only while that same lease is held.
    CaptureProducer& capture_producer(int slot = 8);
    bool begin_capture(NeuralDrafter::Operation&, CapturePhase, int64_t start,
                       int64_t count, std::string& error);
    bool ingest_committed(NeuralDrafter::Operation&, const FeatureView&, std::string&);
    bool ingest_committed(int slot, const DraftOwner& expected, const FeatureView&, std::string&);
    bool reset_owner(int slot, const DraftOwner& expected, int origin, std::string&);
    bool propose_at(int slot, const DraftOwner& expected, int frontier, int32_t anchor,
                    const std::array<float, 3>& uniforms, DraftProposal&, std::string&);
    // Existing engine owns Philox. This merely installs its three already drawn
    // kSpecCtrDraft uniforms; the selector never owns an RNG or acceptance loop.
    bool set_proposal_uniforms(int slot, const DraftOwner&, int frontier,
                               const std::array<float, 3>&, std::string&);
    void set_draft_sampling(const kernels::SamplerParams&) override;
    bool snapshot_capture(SavedConversation&, Operation&, std::string&) override;
    bool snapshot_preflight(const SavedConversation&, int slot, int64_t cells,
                            ConversationSnapshotExpectation, std::string&) const override;
    bool snapshot_restore(const SavedConversation&, Operation&, int64_t cells, int64_t prompt_length,
                          int32_t next_token, ConversationSnapshotExpectation, std::string&) override;
private:
    DraftSlotState* slot_metadata(int) override;
    const DraftSlotState* slot_metadata(int) const override;
    int64_t state_capacity(int) const override { return max_position; }
    bool proposal_preflight(int, const float*, int32_t, int64_t, std::string&) const override;
    bool propose_impl(Operation&, const float*, int32_t, int64_t, DraftProposal&, std::string&) override;
    bool inspect_proposal(Operation&, const DraftProposal&, std::string&) override;
    bool drain_adapter(std::string&) override;
    int64_t snapshot_origin(int slot) const override;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strata::core::dflash2
