// Runs only in strata-resident-test after the normal protected full-model startup.
// No fake CUDA success: all transfers forward to the real runtime; one injected
// transfer failure proves that partial state is unavailable, not GPU recovery.
#include "strata/program/resident_slots.hpp"
#include "strata/program/session_fingerprint.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/mtp.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace {
void* fail_destination = nullptr;
bool transfer_failed = false;
}
extern "C" cudaError_t __real_cudaMemcpy(void*, const void*, size_t, cudaMemcpyKind);
extern "C" cudaError_t __wrap_cudaMemcpy(void* dst, const void* src, size_t size, cudaMemcpyKind kind) {
    if (dst == fail_destination && size) {
        fail_destination = nullptr;
        transfer_failed = true;
        return cudaErrorInvalidValue;
    }
    return __real_cudaMemcpy(dst, src, size, kind);
}

int resident_model_checks(strata::program::ResidentSlots& owner, strata::core::ExpertDispatch& dispatch,
                          const strata::core::ModelGeometry& geometry, strata::core::SessionState& main,
                          strata::core::SessionState& a, strata::core::SessionState& b,
                          strata::core::MtpDrafter& draft) {
    using namespace strata;
    std::string error;
    int checks = 0;
    auto require = [&](bool ok, const char* message) {
        ++checks;
        if (!ok) throw std::runtime_error(std::string(message) + ": " + error);
    };
    try {
        const std::vector<int64_t> prompt{11, 22, 33, 44};
        const std::vector<int32_t> prefix{11, 22};
        program::ResidentRequest request;
        request.max_new = 8;
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
        core::ConversationCheckpoint checkpoint;
        checkpoint.ids = prefix;
        require(core::conversation_checkpoint_save(checkpoint, main, geometry, error), "save real working checkpoint");
        program::ResidentAdmission admission{33, &checkpoint};
        require(owner.park(0, prefix, admission, true, error), "park request A");
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
        require(owner.init_resources(error), "reinitialize retained contexts");
        require(owner.park(0, prefix, admission, true, error), "retain solo continuation before promotion");
        require(owner.begin(1, prompt, request, error) && owner.working().request != aid &&
                owner.working().resume_slot == -1, "different physical destination cannot claim parked identity");
        auto altered_request = request;
        altered_request.sampling.penalty_repeat += 0.1f;
        require(owner.begin(0, prompt, altered_request, error) && owner.working().request != aid &&
                owner.working().resume_slot == -1, "different settings cannot claim parked identity");
        const std::vector<int64_t> other_prompt{11, 22, 33, 45};
        require(owner.begin(0, other_prompt, request, error) && owner.working().request != aid &&
                owner.working().resume_slot == -1, "different full prompt cannot claim parked identity");
        require(owner.begin(0, prompt, request, error), "promote yielded solo request into its parked slot");
        require(owner.working().request == aid && owner.working().sampling.seed == seed && owner.working().resume_slot == 0,
                "solo-to-resident promotion retains identity and automatic seed");
        require(owner.resume(0, nullptr, error), "restore yielded request");
        admission.checkpoint = &owner.slots()[0].checks.at(0);
        require(owner.park(0, prefix, admission, true, error), "yield same identity again");
        require(owner.slots()[0].request == aid && &owner.slots()[0].ids == history, "repark keeps stable owner/history object");
        require(owner.slots()[0].checks.size() == 1 && owner.slots()[0].checks[0].ids == prefix,
                "aliased checkpoint survives repark");
        admission.checkpoint = &checkpoint;
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
        require(!owner.finish(0, true, error) && !owner.available(0), "finish refuses a retained consumer");
        ledger.cancel();
        require(!owner.finish(0, true, error) && owner.slots()[0].active,
                "cancelled submission still prevents finish/reuse");
        require(ledger.drain_submission(operation) && ledger.release(operation), "drain C consumer");
        require(owner.finish(0, true, error), "finish C");
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
        require(owner.finish(1, true, error), "cancel survivor after failed peer");
        require(owner.release_resources(error) && owner.init_resources(error) && !owner.available(0) && owner.available(1),
                "resource re-init restores valid contexts only");
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
