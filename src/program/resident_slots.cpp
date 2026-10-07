#include "strata/program/resident_slots.hpp"
#include "strata/program/session_fingerprint.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/mtp.hpp"
#include "strata/core/verify.hpp"
#include "strata/core/resident_memory.hpp"
#include "strata/prefill/prefill.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace strata::program {
using Clock = std::chrono::steady_clock;
static_assert(sizeof(ResidentSlot) + sizeof(core::SessionOwner) <= plan::kResidentOwnerBytes,
              "resident metadata must fit the existing admission envelope");

ResidentSlots::ResidentSlots(const core::ModelGeometry& geometry, std::vector<ResidentStage> stages,
                             core::MtpDrafter& draft, prefill::Prefill& prompt, core::ExpertDispatch& dispatch,
                             int64_t context, bool cache)
    : geometry_(geometry), stages_(std::move(stages)), draft_(draft), prompt_(prompt), dispatch_(dispatch),
      slots_(stages_.empty() ? 0 : stages_[0].slots.size()), context_(context), cache_(cache) {
    next_request_ = dispatch_.request_generation;
    if (context <= 0 || slots_.size() > plan::kResidentMaxSlots)
        throw std::invalid_argument("resident slot/context extent invalid");
    for (const auto& stage : stages_)
        if (!stage.main || !stage.verifier || stage.slots.size() != slots_.size() ||
            std::find(stage.slots.begin(), stage.slots.end(), nullptr) != stage.slots.end())
            throw std::invalid_argument("resident stage has missing or inconsistent sessions");
    for (auto& slot : slots_) {
        slot.ids.reserve((size_t) context_);
        slot.prompt.reserve((size_t) context_);
        slot.checks.reserve(1);
    }
}

namespace {
bool same_request(const ResidentRequest& a, const ResidentRequest& b) {
    const auto& x = a.sampling;
    const auto& y = b.sampling;
    return a.max_new == b.max_new && a.steering == b.steering && a.images == b.images &&
           a.pcie_fraction == b.pcie_fraction && a.spec_min_probability == b.spec_min_probability &&
           x.top_k == y.top_k && x.top_p == y.top_p && x.min_p == y.min_p &&
           x.temperature == y.temperature && x.min_keep == y.min_keep &&
           x.penalty_last_n == y.penalty_last_n && x.penalty_repeat == y.penalty_repeat &&
           x.penalty_freq == y.penalty_freq && x.penalty_present == y.penalty_present &&
           x.seed == y.seed && x.counter == y.counter && x.greedy == y.greedy;
}
}

bool ResidentSlots::begin(int destination, const std::vector<int64_t>& prompt,
                           const ResidentRequest& request, std::string& error) {
    if (working_failed_ || stages_.empty() || prompt.empty() || prompt.size() >= (uint64_t) context_ ||
        request.max_new < 1 || destination < -1 || (destination >= 0 && !available(destination))) {
        error = "resident working request: invalid or unavailable context"; return false;
    }
    if (!idle(error)) return false;
    ResidentWorking next;
    for (size_t b = 0; b < slots_.size(); ++b) {
        const auto& slot = slots_[b];
        if (!retained((int) b) || !slot.partial || slot.stop ||
            (slot.original_slot != destination && !(slot.original_slot == -1 && destination == (int) b)) ||
            (destination >= 0 && destination != (int) b) || !same_request(slot.input, request) ||
            slot.prompt.size() != prompt.size() ||
            !std::equal(slot.prompt.begin(), slot.prompt.end(), prompt.begin())) continue;
        next.request = slot.request;
        next.sampling = slot.sampling;
        next.resume_slot = (int) b;
        break;
    }
    if (!next.request) {
        if (next_request_ == UINT64_MAX) { error = "resident request identity exhausted"; return false; }
        next.request = ++next_request_;
        next.sampling = request.sampling;
        if (!next.sampling.seed) next.sampling.seed = (uint64_t) Clock::now().time_since_epoch().count();
    }
    working_failed_ = true;
    if (!stages_[0].verifier->set_request_id(next.request, error) ||
        !draft_.bind_request(8, next.request, error) || !prompt_.bind_request(next.request, error)) {
        return false;
    }
    stages_[0].verifier->set_sampling(next.sampling);
    draft_.set_draft_sampling(next.sampling);
    dispatch_.request_generation = next.request;
    working_ = next;
    input_ = request;
    destination_ = destination;
    working_prompt_ = &prompt;
    working_failed_ = false;
    return true;
}

ResidentSlots::~ResidentSlots() {
    // The resource owners outlive us. Drain before histories borrowed by a
    // verifier are destroyed, including on a failed transition/early return.
    if (touched_) for (const auto& stage : stages_)
        if (!stage.verifier->release_gpu_waits(5000)) std::terminate();
    if (touched_) for (const auto& stage : stages_) {
        const core::OnDevice on(stage.device);
        if (cudaDeviceSynchronize() != cudaSuccess) std::terminate();
    }
    for (size_t slot = 0; slot < slots_.size(); ++slot) retire((int) slot);
}

bool ResidentSlots::active() const {
    for (const auto& slot : slots_) if (slot.active) return true;
    return false;
}

bool ResidentSlots::available(int slot) const {
    return resources_ == Resources::ready && slot >= 0 && (size_t) slot < slots_.size() &&
           !slots_[(size_t) slot].active && !slots_[(size_t) slot].failed;
}

bool ResidentSlots::retained(int slot) const {
    return resources_ != Resources::failed && slot >= 0 && (size_t) slot < slots_.size() &&
           !slots_[(size_t) slot].active && !slots_[(size_t) slot].failed &&
           slots_[(size_t) slot].request && slots_[(size_t) slot].cached;
}

void ResidentSlots::cancel(int slot) {
    if (slot < 0 || (size_t) slot >= slots_.size()) return;
    auto& state = slots_[(size_t) slot];
    state.stop = true;
    if (state.partial) {
        residency("cancel", slot);
        state.partial = false;
    }
}

void ResidentSlots::priority(int slot, bool foreground) {
    if (slot >= 0 && (size_t) slot < slots_.size()) slots_[(size_t) slot].foreground = foreground;
}

bool ResidentSlots::idle(std::string& error) {
    // No worker is cancelled here: its operation owner alone may certify a drain.
    for (const auto& completion : dispatch_.completion) if (completion.active()) {
        error = "resident transition: expert consumers still own an operation"; return false;
    }
    for (const auto& stage : stages_)
        if (!stage.verifier->context_idle(error)) return false;
    return draft_.context_idle(error) && prompt_.context_idle(error);
}

bool ResidentSlots::slot_idle(int slot, std::string& error) {
    for (size_t k = 0; k < stages_.size(); ++k) {
        bool expert_pending = false;
        for (size_t group = 2 * k; group < std::min(2 * k + 2, dispatch_.completion.size()); ++group)
            expert_pending = expert_pending || dispatch_.completion[group].active();
        if (!stages_[k].verifier->slot_consumers_idle(slot, expert_pending, error)) return false;
    }
    // A pipelined operation for other rows may continue. Its snapshots don't
    // borrow this slot's history and its stage reports no consumer for this row.
    return draft_.context_idle(error);
}

bool ResidentSlots::init_resources(std::string& error) {
    if (resources_ != Resources::released || active() || stages_.empty() || slots_.empty()) {
        error = "resident resources: release idle resources before initialization"; return false;
    }
    if (!idle(error)) return false;
    for (const auto& stage : stages_) if (stage.verifier->n_slots() != 0) {
        error = "resident resources: verifier already has slot bindings"; return false;
    }
    // Failed partial allocation must still drain before borrowed histories die.
    touched_ = true;
    resources_ = Resources::failed;
    for (const auto& stage : stages_) {
        const core::OnDevice on(stage.device);
        if (!stage.verifier->init_slots(stage.slots, error)) {
            resources_ = Resources::failed; return false;
        }
    }
    for (size_t b = 0; b < slots_.size(); ++b) {
        const auto& slot = slots_[b];
        if (!slot.request || slot.failed) continue;
        if (!stages_[0].verifier->set_slot_context((int) b, slot.request, slot.sampling,
                                                  &slot.ids, slot.cvec, error)) {
            resources_ = Resources::failed; return false;
        }
    }
    resources_ = Resources::ready;
    if (std::getenv("STRATA_STATE_HASH")) std::fprintf(stderr, "strata serve: STATE_RESOURCES phase=initialized\n");
    return true;
}

bool ResidentSlots::release_resources(std::string& error) {
    if (resources_ != Resources::ready || active()) {
        error = "resident resources: release requires initialized idle requests"; return false;
    }
    if (!idle(error)) return false;
    if (!stages_[0].verifier->release_batch_resources(error)) {
        resources_ = Resources::failed; return false;
    }
    resources_ = Resources::released;
    if (std::getenv("STRATA_STATE_HASH")) std::fprintf(stderr, "strata serve: STATE_RESOURCES phase=released\n");
    return true;
}

void ResidentSlots::residency(const char* phase, int slot) const {
    const auto& state = slots_[(size_t) slot];
    if (state.request && std::getenv("STRATA_STATE_HASH"))
        std::fprintf(stderr, "STATE_RESIDENCY request=%llu slot=%d phase=%s cells=%zu\n",
                     (unsigned long long) state.request, slot, phase, state.ids.size());
}

void ResidentSlots::retire(int slot) {
    auto& state = slots_[(size_t) slot];
    if (state.active || state.failed) residency("failure", slot);
    residency("release", slot);
    state.request = 0;
    dispatch_.slot_requests[slot] = 0;
}

bool ResidentSlots::fail(int slot) {
    auto& state = slots_[(size_t) slot];
    state.failed = true;
    state.active = state.cached = state.partial = false;
    return false;
}

bool ResidentSlots::admit(int slot, const std::vector<int32_t>& prefix,
                          const ResidentAdmission& request, std::string& error) {
    return bind(slot, prefix, request, false, false, error);
}

bool ResidentSlots::park(int slot, const std::vector<int32_t>& prefix,
                         const ResidentAdmission& request, bool from_start, std::string& error) {
    return bind(slot, prefix, request, true, from_start, error);
}

bool ResidentSlots::bind(int slot, const std::vector<int32_t>& prefix, const ResidentAdmission& request,
                         bool parked, bool from_start, std::string& error) {
    if (working_failed_ || !available(slot) || !working_.request || prefix.empty() || prefix.size() >= (uint64_t) context_ ||
        (!parked && (input_.max_new <= 1 || destination_ != slot)) ||
        (parked && (!cache_ || input_.images || !working_prompt_ || (destination_ >= 0 && destination_ != slot) ||
         prefix.size() >= working_prompt_->size() || !std::equal(prefix.begin(), prefix.end(), working_prompt_->begin())))) {
        error = "resident admission: invalid or unavailable slot/request"; return false;
    }
    if (request.checkpoint && (request.checkpoint->stage_parts.size() + 1 != stages_.size() ||
        request.checkpoint->ids.size() > prefix.size() ||
        !std::equal(request.checkpoint->ids.begin(), request.checkpoint->ids.end(), prefix.begin()))) {
        error = "resident admission: checkpoint is not a prefix of the admitted context"; return false;
    }
    for (size_t b = 0; b < slots_.size(); ++b)
        if ((int) b != slot && slots_[b].request == working_.request) {
            error = "resident admission: request already owns another slot"; return false;
        }
    if (!idle(error)) return false;
    auto& state = slots_[(size_t) slot];
    const bool continuing = state.request == working_.request && working_.resume_slot == slot;
    // Capture possibly aliased checkpoint metadata before clearing the target.
    std::vector<core::ConversationCheckpoint> selected;
    try {
        if (request.checkpoint && !input_.images) selected.push_back(*request.checkpoint);
    } catch (const std::bad_alloc&) {
        error = "resident admission: checkpoint allocation failed"; return false;
    }
    // Retire before overwriting, but don't publish a new identity until every
    // consumer is bound. A yielded continuation keeps the existing owner.
    if (!continuing) retire(slot);
    state.active = state.cached = state.partial = false;
    state.failed = true; // every exception after this point leaves an unavailable target
    touched_ = true;
    state.checks.swap(selected);
    selected.clear();
    try {
        if (!copy_to(slot, prefix, error) || !draft_.bind_request(slot, working_.request, error) ||
            !draft_.save_slot(slot, (int64_t) prefix.size(), error)) return fail(slot);
    } catch (const std::bad_alloc&) {
        error = "resident admission: transfer allocation failed"; return fail(slot);
    }
    state.ids.assign(prefix.begin(), prefix.end());
    if (parked) {
        state.prompt.assign(working_prompt_->begin(), working_prompt_->end());
        state.input = input_;
        state.original_slot = destination_;
    } else state.prompt.clear();
    state.sampling = working_.sampling;
    state.request = working_.request;
    state.x = request.next_token;
    state.p = (int64_t) prefix.size();
    state.produced = parked ? 0 : 1;
    state.max_new = input_.max_new;
    state.t0 = Clock::now();
    state.cvec = input_.steering;
    state.img = input_.images;
    state.stop = false;
    state.partial_from0 = parked && from_start;
    dispatch_.slot_requests[slot] = working_.request;
    if (request.residual && !draft_.propose_slot(slot, request.residual, state.x, state.p - 1,
                                                state.draft, error)) return fail(slot);
    if (!stages_[0].verifier->set_slot_context(slot, working_.request, state.sampling,
                                              &state.ids, state.cvec, error) ||
        !observe(continuing ? "resume" : "admit", slot, (int64_t) prefix.size(), error)) {
        // Retain a continuing identity for balanced failure/release receipts.
        // A new request has not published its start yet.
        if (!continuing) state.request = dispatch_.slot_requests[slot] = 0;
        return fail(slot);
    }
    if (!continuing) residency("start", slot);
    state.failed = false;
    state.active = !parked;
    state.cached = state.partial = parked;
    if (parked) residency("pause", slot);
    return true;
}

bool ResidentSlots::copy_to(int slot, const std::vector<int32_t>& prefix, std::string& error) {
    const int64_t upto = (int64_t) prefix.size();
    for (const auto& stage : stages_) {
        auto& from = *stage.main;
        auto& to = *stage.slots[(size_t) slot];
        const core::OnDevice on(stage.device);
        if (cudaDeviceSynchronize() != cudaSuccess) { error = "batch admission: device sync failed"; return false; }
        core::ConversationCheckpoint checkpoint;
        checkpoint.ids = prefix;
        if (!core::conversation_checkpoint_save(checkpoint, from, geometry_, error) ||
            !core::conversation_checkpoint_restore(checkpoint, to, geometry_, error)) return false;
        for (int64_t j = 0; j < from.qsa_alloc; ++j) {
            core::ConversationKv image;
            if (!core::conversation_kv_save(image, from.qsa_states[from.qsa_ord0 + j], geometry_, upto, true, error))
                return false;
            if (cudaDeviceSynchronize() != cudaSuccess) { error = "batch admission: device sync failed"; return false; }
            if (!core::conversation_kv_restore(image, to.qsa_states[to.qsa_ord0 + j], geometry_, upto, true, error))
                return false;
        }
        if (cudaDeviceSynchronize() != cudaSuccess) { error = "batch admission: device sync failed"; return false; }
    }
    return true;
}

bool ResidentSlots::resume(int slot, const core::ConversationCheckpoint* checkpoint, std::string& error) {
    if (working_failed_ || !working_.request || !retained(slot)) {
        error = "resident resume: no valid idle prefix"; return false;
    }
    if (checkpoint && (checkpoint->stage_parts.size() + 1 != stages_.size() ||
        checkpoint->ids.size() > slots_[(size_t) slot].ids.size() ||
        !std::equal(checkpoint->ids.begin(), checkpoint->ids.end(), slots_[(size_t) slot].ids.begin()))) {
        error = "resident resume: checkpoint is not a retained prefix"; return false;
    }
    if (!idle(error)) return false;
    const auto cells = (int64_t) (checkpoint ? checkpoint->ids.size() : slots_[(size_t) slot].ids.size());
    working_failed_ = true; // any partial write or exception makes the working context unusable
    try {
        if (!copy_from(slot, checkpoint, error) || !draft_.restore_slot(slot, cells, error) ||
            !observe("resume", slot, cells, error)) {
            return fail(slot);
        }
    } catch (const std::bad_alloc&) {
        error = "resident resume: transfer allocation failed";
        return fail(slot);
    }
    working_failed_ = false;
    residency("resume", slot);
    if (working_.resume_slot == slot) slots_[(size_t) slot].partial = false;
    return true;
}

bool ResidentSlots::copy_from(int slot, const core::ConversationCheckpoint* at, std::string& error) {
    const auto& prefix = at ? at->ids : slots_[(size_t) slot].ids;
    const int64_t upto = (int64_t) prefix.size();
    for (size_t k = 0; k < stages_.size(); ++k) {
        const auto& stage = stages_[k];
        auto& from = *stage.slots[(size_t) slot];
        auto& to = *stage.main;
        const core::OnDevice on(stage.device);
        if (cudaDeviceSynchronize() != cudaSuccess) { error = "batch slot restore: device sync failed"; return false; }
        if (at) {
            if (!core::conversation_checkpoint_restore(k == 0 ? *at : at->stage_parts[k - 1], to, geometry_, error))
                return false;
        } else {
            core::ConversationCheckpoint checkpoint;
            checkpoint.ids = prefix;
            if (!core::conversation_checkpoint_save(checkpoint, from, geometry_, error) ||
                !core::conversation_checkpoint_restore(checkpoint, to, geometry_, error)) return false;
        }
        for (int64_t j = 0; j < from.qsa_alloc; ++j) {
            core::ConversationKv image;
            if (!core::conversation_kv_save(image, from.qsa_states[from.qsa_ord0 + j], geometry_, upto, true, error))
                return false;
            if (cudaDeviceSynchronize() != cudaSuccess) { error = "batch slot restore: device sync failed"; return false; }
            if (!core::conversation_kv_restore(image, to.qsa_states[to.qsa_ord0 + j], geometry_, upto, true, error))
                return false;
        }
        if (cudaDeviceSynchronize() != cudaSuccess) { error = "batch slot restore: device sync failed"; return false; }
    }
    return true;
}

bool ResidentSlots::committed(int slot, int32_t next, const float* residual, std::string& error, bool diagnostic) {
    if (slot < 0 || (size_t) slot >= slots_.size() || !slots_[(size_t) slot].active ||
        slots_[(size_t) slot].ids.size() >= (uint64_t) context_) {
        error = "resident commit: no active request or context exhausted"; return false;
    }
    auto& state = slots_[(size_t) slot];
    if (!slot_idle(slot, error)) return false;
    if (!draft_.advance_slot(slot, residual, next, state.p, error)) return fail(slot);
    state.ids.push_back(state.x);
    if (diagnostic && !observe("commit", slot, (int64_t) state.ids.size(), error, residual)) return fail(slot);
    ++state.produced;
    state.x = next;
    ++state.p;
    return true;
}

bool ResidentSlots::propose(int slot, const float* residual, std::string& error) {
    if (slot < 0 || (size_t) slot >= slots_.size() || !slots_[(size_t) slot].active ||
        !slot_idle(slot, error)) return false;
    auto& state = slots_[(size_t) slot];
    if (!draft_.propose_slot(slot, residual, state.x, state.p - 1, state.draft, error)) return fail(slot);
    return true;
}


bool ResidentSlots::finish(int slot, bool keep_cache, std::string& error, bool diagnostic) {
    if (slot < 0 || (size_t) slot >= slots_.size() || !slots_[(size_t) slot].active) {
        error = "resident completion: no active request"; return false;
    }
    if (!slot_idle(slot, error)) return false;
    auto& state = slots_[(size_t) slot];
    if (state.stop) {
        if (diagnostic && !observe("cancel", slot, (int64_t) state.ids.size(), error)) return fail(slot);
        residency("cancel", slot);
    }
    state.active = false;
    residency("pause", slot);
    state.cached = keep_cache && cache_ && !state.img;
    return true;
}

bool ResidentSlots::observe(const char* phase, int slot, int64_t cells, std::string& error, const float* residual) {
    if (!std::getenv("STRATA_STATE_HASH")) return true;
    const auto* draft = draft_.slot_state(slot);
    if (!draft) { error = "resident draft state missing"; return false; }
    auto& session = *stages_[0].slots[(size_t) slot];
    SessionFingerprint h;
    if (!session_fingerprint(session, *draft, geometry_, cells, h, error, residual)) return false;
    const uint64_t id = slots_[(size_t) slot].request;
    if (!id || dispatch_.slot_requests[slot] != id) { error = "resident request owner mismatch"; return false; }
    const bool released = resources_ == Resources::released;
    const uint64_t observed = stages_[0].verifier->validated_slot_request_id(slot);
    if ((!released && observed != id) || (released && (observed || std::strcmp(phase, "resume")))) {
        error = "resident graph canary mismatch"; return false;
    }
    // Private session bytes outlive batch graphs. A retained fingerprint is not
    // a fresh graph-canary check; re-init must upload and validate the owner again.
    std::fprintf(stderr, "strata serve: %s phase=%s slot=%d request=%llu cells=%lld gdn=%016llx "
                         "ple=%016llx kv=%016llx draft=%016llx canary=%llu\n",
        released ? "STATE_RETAINED" : "STATE_SLOT", phase, slot,
        (unsigned long long) id, (long long) cells, (unsigned long long) h.gdn,
        (unsigned long long) fingerprint_ple(h, session), (unsigned long long) fingerprint_kv(h),
        (unsigned long long) h.draft, (unsigned long long) observed);
    if (residual && std::getenv("STRATA_BATCH_DIAGNOSTICS")) report_batch_state(phase, id, slot, cells, session, h);
    return true;
}
} // namespace strata::program
