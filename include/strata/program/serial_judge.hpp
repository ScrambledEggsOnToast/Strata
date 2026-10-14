#pragma once
#include "strata/core/session.hpp"
#include "strata/plan/admission.hpp"
#include <functional>
#include <memory>

namespace strata::prefill { class Prefill; }
namespace strata::core { class Verifier; class MtpDrafter; class NativeHead; }
namespace strata::program {

// Incremental ownership only. Shared weights, Prefill/Gemm/workspace and their
// existing driver/library allowance remain in the generation ledger, never deducted.
struct JudgeStageBytes {
    core::SessionAllocationBytes session;
    uint64_t device_canary = 0, device_head_output = 0, device_allocator_padding = 0;
    uint64_t host_canary = 0, host_checkpoint = 0, host_readback = 0, host_metadata = 0;
    uint64_t device_bytes() const;
    uint64_t host_bytes() const;
};
bool judge_stage_bytes(const core::ModelGeometry& geometry, int64_t cells, int64_t k,
                       int64_t lo, int64_t hi, int64_t vocab, bool head_stage,
                       const plan::QsaPlacement& placement, JudgeStageBytes& out);

struct JudgeStage {
    core::SessionOwner* generation = nullptr;
    prefill::Prefill* prompt = nullptr;
    core::Verifier* verifier = nullptr;
    core::Verifier* batch_peer = nullptr;
    const core::WeightTable* weights = nullptr;
    void* stream = nullptr;
};

// One host submitter, at a drained scheduler boundary (no active generation
// requests). Borrows the actual generation execution owners, not a second chain.
// Every state owner, checkpoint and readback is allocated before request admission.
// No scoring owner enters the generation slot/MTP vectors. Failure poisons the
// lane; generation can resume only after every binding/callback/RoPE table restores.
class SerialJudgeLane {
public:
    SerialJudgeLane();
    ~SerialJudgeLane();
    SerialJudgeLane(const SerialJudgeLane&) = delete;
    SerialJudgeLane& operator=(const SerialJudgeLane&) = delete;
    bool init(const core::ModelGeometry& geometry, int64_t cells, int64_t k, int64_t vocab,
              const plan::QsaPlacement& placement, const std::vector<JudgeStage>& stages,
              const core::NativeHead& head, core::MtpDrafter* draft,
              const plan::AdmissionDecision& admission, std::string& error);
    // A clean cancellation returns true with no logits, after every owner has drained.
    bool run(uint64_t request, const int64_t* tokens, int64_t count, std::string& error,
             const std::function<bool()>& cancelled = {});
    const float* logits() const;
    int64_t vocab() const;
    bool ready() const;
    /// Capture complete-run allocation high-water before a serving frontend can stop the child.
    void observe_generation() const;
private:
    friend bool judge_stage_bytes(const core::ModelGeometry&, int64_t, int64_t, int64_t, int64_t,
                                  int64_t, bool, const plan::QsaPlacement&, JudgeStageBytes&);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strata::program
