// Resident request transitions. SessionOwner, Verifier and MTP retain resource ownership.
#pragma once
#include "strata/core/conversation_snapshot.hpp"
#include "strata/kernels/sampler.hpp"
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::core { class Verifier; class MtpDrafter; struct ExpertDispatch; }
namespace strata::prefill { class Prefill; }
namespace strata::program {

struct ResidentStage {
    core::SessionState* main = nullptr;
    core::Verifier* verifier = nullptr;
    std::vector<core::SessionState*> slots;
    int device = -1;
};

struct ResidentRequest {
    kernels::SamplerParams sampling; // seed 0 resolves once, retained across yield
    int64_t max_new = 0;
    bool steering = true, images = false;
    double pcie_fraction = 0, spec_min_probability = 0;
};

struct ResidentWorking {
    uint64_t request = 0;
    kernels::SamplerParams sampling;
    int resume_slot = -1;
};

// Read-only to the scheduler. The vector objects and graph-referenced sessions
// never move while bound. Token storage is reserved once per admitted slot.
struct ResidentSlot {
    bool active = false, stop = false, foreground = true;
    kernels::SamplerParams sampling;
    uint64_t request = 0;
    int32_t x = 0;
    int64_t p = 0, produced = 0, max_new = 0;
    std::chrono::steady_clock::time_point t0;
    std::vector<int32_t> ids;
    std::vector<int32_t> prompt; // yielded full prompt; third accounted token history
    ResidentRequest input;
    int original_slot = -1;
    bool cached = false, cvec = true, img = false;
    std::vector<core::ConversationCheckpoint> checks;
    bool partial = false, partial_from0 = false;
    bool failed = false;
};

struct ResidentAdmission {
    int32_t next_token = 0;
    const core::ConversationCheckpoint* checkpoint = nullptr;
};

// One host submitter, the same as Verifier. Scheduling and protocol emission
// remain outside; callers never rebind identity/history/dispatch/MTP separately.
// Failed preflight leaves every slot unchanged. Failure after mutation invalidates
// only the target; a poisoned resource owner continues to refuse subsequent work.
class ResidentSlots {
public:
    ResidentSlots(const core::ModelGeometry& geometry, std::vector<ResidentStage> stages,
                  core::MtpDrafter& draft, prefill::Prefill& prompt, core::ExpertDispatch& dispatch,
                  int64_t context, bool cache);
    ~ResidentSlots();
    ResidentSlots(const ResidentSlots&) = delete;
    ResidentSlots& operator=(const ResidentSlots&) = delete;
    ResidentSlots(ResidentSlots&&) = delete;
    ResidentSlots& operator=(ResidentSlots&&) = delete;
    const std::vector<ResidentSlot>& slots() const { return slots_; }
    // Bind the working verifier, prefill, draft and dispatch together. An exact
    // yielded resend or solo-to-parked-slot promotion retains identity and seed.
    bool begin(int destination, const std::vector<int64_t>& prompt, const ResidentRequest& request,
               std::string& error);
    const ResidentWorking& working() const { return working_; }
    bool active() const;
    bool available(int slot) const;
    void cancel(int slot);
    void priority(int slot, bool foreground);
    bool init_resources(std::string& error);
    bool release_resources(std::string& error);
    bool admit(int slot, const std::vector<int32_t>& prefix, const ResidentAdmission& request, std::string& error);
    bool park(int slot, const std::vector<int32_t>& prefix, const ResidentAdmission& request,
              bool from_start, std::string& error);
    // Restores private model state into the working session. A partial restore
    // returns false; it is never advertised as a usable prefix by this interface.
    bool resume(int slot, const core::ConversationCheckpoint* checkpoint, std::string& error);
    // Called only for a GPU-committed row. No state migration or allocation.
    bool committed(int slot, int32_t next, const float* residual, std::string& error, bool observe = true);
    bool finish(int slot, bool keep_cache, std::string& error, bool diagnostic = true);
private:
    enum class Resources { released, ready, failed };
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
    core::MtpDrafter& draft_;
    prefill::Prefill& prompt_;
    core::ExpertDispatch& dispatch_;
    std::vector<ResidentSlot> slots_;
    int64_t context_;
    ResidentWorking working_;
    ResidentRequest input_;
    const std::vector<int64_t>* working_prompt_ = nullptr; // borrowed only through this request's park/admit
    uint64_t next_request_ = 0;
    int destination_ = -1;
    bool working_failed_ = false;
    bool cache_, touched_ = false;
    Resources resources_ = Resources::released;
};
} // namespace strata::program
