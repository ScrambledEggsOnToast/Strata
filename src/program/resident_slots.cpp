#include "strata/program/resident_slots.hpp"
#include "strata/program/session_fingerprint.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/mtp.hpp"
#include "strata/core/admission_copy_observer.hpp"
#include "strata/core/verify.hpp"
#include "strata/core/resident_memory.hpp"
#include "strata/prefill/prefill.hpp"
#include "strata/plan/admission.hpp"
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
                             int64_t context, bool cache, int64_t checkpoint_chain_limit, uint64_t checkpoint_byte_limit)
    : geometry_(geometry), stages_(std::move(stages)), draft_(draft), prompt_(prompt), dispatch_(dispatch),
      slots_(stages_.empty() ? 0 : stages_[0].slots.size()), context_(context), preparations_(slots_.size()),
      checkpoint_chain_limit_(checkpoint_chain_limit), checkpoint_byte_limit_(checkpoint_byte_limit), cache_(cache) {
    next_request_ = dispatch_.request_generation;
    if (context <= 0 || context > INT32_MAX || checkpoint_chain_limit <= 0 || checkpoint_chain_limit > INT32_MAX ||
        slots_.size() > plan::kResidentMaxSlots || (!slots_.empty() && !checkpoint_byte_limit_))
        throw std::invalid_argument("resident slot/context extent invalid");
    for (const auto& stage : stages_)
        if (!stage.main || !stage.verifier || stage.slots.size() != slots_.size() ||
            std::find(stage.slots.begin(), stage.slots.end(), nullptr) != stage.slots.end())
            throw std::invalid_argument("resident stage has missing or inconsistent sessions");
    for (auto& slot : slots_) {
        slot.ids.reserve((size_t) context_);
        slot.prompt.reserve((size_t) context_);
        slot.checks.reserve((size_t) checkpoint_chain_limit_);
    }
}

namespace {
bool admission_timing_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_ADMISSION_TIMING");
        return v != nullptr && std::strcmp(v, "0") != 0; // cached: unset/=0 off
    }();
    return on;
}

void report_admission_timing(int slot, const char* component, int device, int64_t items, Clock::time_point start) {
    const double ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    std::fprintf(stderr, "strata admission timing: slot=%d component=%s device=%d items=%lld host_ms=%.3f\n",
                 slot, component, device, (long long) items, ms);
}

bool same_request(const ResidentRequest& a, const ResidentRequest& b) {
    const auto& x = a.sampling;
    const auto& y = b.sampling;
    return a.scope.matches(b.scope) && a.shared_scope.value == b.shared_scope.value && a.pin == b.pin &&
           a.checkpoint == b.checkpoint &&
           a.max_new == b.max_new && a.steering == b.steering && a.images == b.images &&
           a.pcie_fraction == b.pcie_fraction && a.spec_min_probability == b.spec_min_probability &&
           x.top_k == y.top_k && x.top_p == y.top_p && x.min_p == y.min_p &&
           x.temperature == y.temperature && x.min_keep == y.min_keep &&
           x.penalty_last_n == y.penalty_last_n && x.penalty_repeat == y.penalty_repeat &&
           x.penalty_freq == y.penalty_freq && x.penalty_present == y.penalty_present &&
           x.seed == y.seed && x.counter == y.counter && x.greedy == y.greedy;
}

template<class Token>
bool preparation_prefix(const core::ConversationCheckpoint& cut, const std::vector<Token>& tokens, int64_t limit) {
    return !cut.ids.empty() && cut.imgs.empty() && (int64_t) cut.ids.size() <= limit &&
           cut.ids.size() <= tokens.size() && std::equal(cut.ids.begin(), cut.ids.end(), tokens.begin());
}

const core::SavedConversation& preparation_stage(const core::SavedConversation& image, size_t stage) {
    return stage ? image.stage_images[stage - 1] : image;
}

const core::ConversationCheckpoint* preparation_stage_cut(const core::SavedConversation& image,
        const core::ConversationCheckpoint& cut, size_t stage, size_t stages) {
    if (!stage) return &cut;
    if (stage >= stages) return nullptr;
    const auto& part = preparation_stage(image, stage);
    if (cut.ids == part.live.ids && cut.imgs == part.live.imgs) return &part.live;
    const auto found = std::find_if(part.checkpoints.begin(), part.checkpoints.end(), [&](const auto& candidate) {
        return candidate.ids == cut.ids && candidate.imgs == cut.imgs;
    });
    return found == part.checkpoints.end() ? nullptr : &*found;
}
}

bool ResidentSlots::begin(int destination, const std::vector<int64_t>& prompt,
                           const ResidentRequest& request, std::string& error) {
    if (working_failed_ || stages_.empty() || prompt.empty() || prompt.size() >= (uint64_t) context_ ||
        request.max_new < 1 || destination < -1 || (destination >= 0 && !available(destination))) {
        error = "resident working request: invalid or unavailable context"; return false;
    }
    plan::RequestDemand demand;
    demand.prompt_tokens = (int64_t) prompt.size();
    demand.max_new_tokens = request.max_new;
    const auto admission = plan::admit_request(demand, context_);
    if (!admission.admitted) { error = admission.reason; return false; }
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
    working_.prefill_split_until = 0;
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

bool ResidentSlots::begin_score(std::string& error) {
    if (working_failed_ || stages_.empty() || next_request_ == UINT64_MAX) {
        error = "score working owner unavailable"; return false;
    }
    if (!idle(error)) return false;
    working_failed_ = true;
    working_.prefill_split_until = 0;
    ResidentWorking next;
    next.request = ++next_request_;
    next.sampling.greedy = true;
    next.sampling.temperature = 0;
    if (!stages_[0].verifier->set_request_id(next.request, error)) return false;
    stages_[0].verifier->set_sampling(next.sampling);
    dispatch_.request_generation = next.request;
    working_ = next;
    input_ = {};
    destination_ = -1;
    working_prompt_ = nullptr;
    working_failed_ = false;
    return true;
}

ResidentSlots::~ResidentSlots() {
    // The resource owners outlive us. Drain before histories borrowed by a
    // verifier are destroyed, including on a failed transition/early return.
    if (touched_) for (const auto& stage : stages_) {
        if (!stage.verifier->release_gpu_waits(5000) ||
            (stage.batch_peer && !stage.batch_peer->release_gpu_waits(5000)) ||
            (stage.batch_third && !stage.batch_third->release_gpu_waits(5000))) std::terminate();
    }
    if (touched_) for (const auto& stage : stages_) {
        const core::OnDevice on(stage.device);
        if (cudaDeviceSynchronize() != cudaSuccess) std::terminate();
    }
    for (size_t slot = 0; slot < slots_.size(); ++slot) retire((int) slot);
}

bool ResidentSlots::active() const {
    for (const auto& slot : slots_) if (slot.active || slot.preparing) return true;
    return false;
}

bool ResidentSlots::available(int slot) const {
    return resources_ == Resources::ready && slot >= 0 && (size_t) slot < slots_.size() &&
           !slots_[(size_t) slot].active && !slots_[(size_t) slot].preparing && !slots_[(size_t) slot].failed;
}

ResidentIdentity ResidentSlots::identity(int slot) const {
    if (slot < 0 || (size_t) slot >= slots_.size()) return {};
    const auto& state = slots_[(size_t) slot];
    return {state.request, state.incarnation};
}

bool ResidentSlots::owns(int slot, uint64_t request, uint64_t incarnation) const {
    return identity(slot).owns(request, incarnation);
}

ContinuationCertificate ResidentSlots::certify_continuation(int slot, const core::ConversationCheckpoint* cut,
        uint64_t expected_request, uint64_t expected_incarnation, std::string* error) const {
    ContinuationInput in;
    if (slot < 0 || (size_t) slot >= slots_.size() || stages_.empty()) {
        if (error) *error = "continuation: unbound seat";
        return {};
    }
    const auto& state = slots_[(size_t) slot];
    in.request = state.request;
    in.incarnation = state.incarnation;
    in.expected_request = expected_request;
    in.expected_incarnation = expected_incarnation;
    in.committed_tokens = (int64_t) state.ids.size();
    in.cut_tokens = cut ? (int64_t) cut->ids.size() : -1;
    in.cut_is_live_prefix = !cut || (cut->ids.size() <= state.ids.size() &&
        std::equal(cut->ids.begin(), cut->ids.end(), state.ids.begin()));
    // Resident images are not parkable; text cuts are certified here.
    in.live_images = 0;
    in.cut_images = 0;
    in.needs_qsa = geometry_.n_qsa_layers() > 0;
    in.has_qsa = std::any_of(stages_.begin(), stages_.end(), [slot](const ResidentStage& stage) {
        return (size_t) slot < stage.slots.size() && stage.slots[(size_t) slot] &&
               stage.slots[(size_t) slot]->qsa_alloc > 0;
    });
    in.needs_gdn = geometry_.n_gdn_layers() > 0;
    in.has_gdn = stages_[0].main && stages_[0].main->gdn_state != nullptr;
    in.needs_ple = stages_[0].main && stages_[0].main->ple_hist != nullptr;
    in.has_ple = in.needs_ple;
    in.needs_stages = stages_.size() > 1;
    in.has_stages = std::all_of(stages_.begin(), stages_.end(), [slot](const ResidentStage& stage) {
        return (size_t) slot < stage.slots.size() && stage.slots[(size_t) slot] != nullptr;
    });
    const auto* draft = draft_.kv_state().max_cells > 0 ? draft_.slot_state(slot) : nullptr;
    if (draft) {
        const int64_t frontier = draft_.slot_resume_frontier(slot, in.committed_tokens);
        if (draft_.slot_covers(slot, frontier)) {
            in.draft_present = true;
            in.draft_begin = draft_.slot_begin(slot);
            in.draft_end = std::max(frontier, in.draft_begin);
            in.draft_frontier = frontier;
            in.draft_boundary = draft_.slot_next_token(slot);
            in.token_at_frontier = frontier < (int64_t) state.ids.size()
                ? state.ids[(size_t) frontier] : in.draft_boundary;
        }
    }
    ContinuationCertificate certificate = ::strata::program::certify_continuation(in);
    if (std::getenv("STRATA_CONTINUATION_TRACE"))
        std::fprintf(stderr, "CONTINUATION slot=%d certified=%d reason=%s request=%llu incarnation=%llu "
                             "committed=%lld cut=%lld draft=%d\n", slot, int(certificate.certified),
                     certificate.reason(), (unsigned long long) certificate.request,
                     (unsigned long long) certificate.incarnation, (long long) certificate.committed_tokens,
                     (long long) certificate.valid_end, int(certificate.draft_ready));
    if (!certificate.certified && error) *error = std::string("continuation: ") + certificate.reason();
    return certificate;
}

bool ResidentSlots::retained(int slot) const {
    return resources_ != Resources::failed && slot >= 0 && (size_t) slot < slots_.size() &&
           !slots_[(size_t) slot].active && !slots_[(size_t) slot].preparing && !slots_[(size_t) slot].failed &&
           slots_[(size_t) slot].request && slots_[(size_t) slot].cached;
}

void ResidentSlots::cancel(int slot) {
    if (slot < 0 || (size_t) slot >= slots_.size()) return;
    auto& state = slots_[(size_t) slot];
    state.stop = true;
    state.prefill_split_until = 0;
    state.prefill_plan = {};
    if (state.partial) {
        residency("cancel", slot);
        state.partial = false;
    }
}

void ResidentSlots::taint(int slot) {
    if (slot >= 0 && (size_t) slot < slots_.size()) {
        slots_[(size_t) slot].tainted = true;
        slots_[(size_t) slot].prefill_split_until = 0;
        slots_[(size_t) slot].prefill_plan = {};
    }
}

std::pair<size_t, size_t> ResidentSlots::invalidate_after_nonfinite() {
    size_t running = 0, idle = 0;
    working_.prefill_split_until = 0;
    for (auto& sl : slots_) {
        sl.prefill_split_until = 0;
        sl.prefill_plan = {};
        if (sl.active || sl.preparing) { sl.tainted = true; ++running; }
        else if (sl.cached) { sl.cached = false; sl.partial = false; sl.checks.clear(); ++idle; }
    }
    return {running, idle};
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
        if (!stage.verifier->context_idle(error) ||
            (stage.batch_peer && !stage.batch_peer->context_idle(error)) ||
            (stage.batch_third && !stage.batch_third->context_idle(error))) return false;
    return draft_.context_idle(error) && prompt_.context_idle(error);
}

bool ResidentSlots::slot_idle(int slot, std::string& error) {
    for (size_t k = 0; k < stages_.size(); ++k) {
        bool expert_pending = false;
        for (size_t group = 2 * k; group < std::min(2 * k + 2, dispatch_.completion.size()); ++group)
            expert_pending = expert_pending || dispatch_.completion[group].active();
        const auto& stage = stages_[k];
        bool third_pending = false;
        if (stage.batch_third) {
            const size_t lane = (size_t)stage.third_worker_lane;
            if (stage.third_worker_lane < 0 || 2 * lane + 1 >= dispatch_.completion.size()) {
                error = "resident third owner: missing worker lane"; return false;
            }
            third_pending = dispatch_.completion[2 * lane].active() || dispatch_.completion[2 * lane + 1].active();
        }
        if (!stage.verifier->slot_consumers_idle(slot, expert_pending &&
                (!stage.batch_peer || stage.verifier->batch_busy()), error) ||
            (stage.batch_peer && !stage.batch_peer->slot_consumers_idle(slot,
                expert_pending && stage.batch_peer->batch_busy(), error)) ||
            (stage.batch_third && !stage.batch_third->slot_consumers_idle(slot, third_pending, error))) return false;
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
    for (const auto& stage : stages_)
        if (stage.verifier->n_slots() != 0 || (stage.batch_peer && stage.batch_peer->n_slots() != 0) ||
            (stage.batch_third && stage.batch_third->n_slots() != 0)) {
            error = "resident resources: verifier already has slot bindings"; return false;
        }
    // Failed partial allocation must still drain before borrowed histories die.
    touched_ = true;
    resources_ = Resources::failed;
    for (const auto& stage : stages_) {
        const core::OnDevice on(stage.device);
        if (!stage.verifier->init_slots(stage.slots, error) ||
            (stage.batch_peer && !stage.batch_peer->init_slots(stage.slots, error)) ||
            (stage.batch_third && !stage.batch_third->init_slots(stage.slots, error))) {
            resources_ = Resources::failed; return false;
        }
    }
    for (size_t b = 0; b < slots_.size(); ++b) {
        const auto& slot = slots_[b];
        if (!slot.request || slot.failed) continue;
        if (!stages_[0].verifier->set_slot_context((int) b, slot.request, slot.sampling,
                                                  &slot.ids, slot.cvec, error) ||
            (stages_[0].batch_peer && !stages_[0].batch_peer->set_slot_context((int) b, slot.request,
                                                  slot.sampling, &slot.ids, slot.cvec, error)) ||
            (stages_[0].batch_third && !stages_[0].batch_third->set_slot_context((int)b, slot.request,
                                                  slot.sampling, &slot.ids, slot.cvec, error))) {
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
    if (!stages_[0].verifier->release_batch_resources(error) ||
        (stages_[0].batch_peer && !stages_[0].batch_peer->release_batch_resources(error)) ||
        (stages_[0].batch_third && !stages_[0].batch_third->release_batch_resources(error))) {
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
    state.preparing = false;
    state.prefill_split_until = 0;
    dispatch_.slot_requests[slot] = 0;
}

bool ResidentSlots::fail(int slot) {
    auto& state = slots_[(size_t) slot];
    state.failed = true;
    state.active = state.preparing = state.cached = state.partial = false;
    state.prefill_split_until = 0;
    preparations_[(size_t) slot] = {};
    if (state.request == working_.request) working_.prefill_split_until = 0;
    if (state.request == working_.request) working_.prefill_plan = {};
    return false;
}

bool ResidentSlots::preparation_valid(const ResidentPreparation& lease, std::string& error) const {
    if (resources_ != Resources::ready || lease.slot < 0 || (size_t) lease.slot >= slots_.size()) {
        error = "resident preparation: invalid destination"; return false;
    }
    const auto& state = slots_[(size_t) lease.slot];
    if (!state.preparing || state.failed || !lease.request || state.request != lease.request ||
        !lease.incarnation || state.incarnation != lease.incarnation || state.prompt.empty() ||
        lease.endpoint != (int64_t) state.prompt.size() - 1) {
        error = "resident preparation: stale request or seat incarnation"; return false;
    }
    for (size_t stage = 0; stage < lease.stages.size(); ++stage)
        if (lease.stages[stage] != (stage < stages_.size() ? stages_[stage].slots[(size_t) lease.slot] : nullptr)) {
            error = "resident preparation: foreign physical destination"; return false;
        }
    return true;
}

bool ResidentSlots::validate_checkpoint(const core::ConversationCheckpoint& cut, int slot, std::string& error) const {
    if (cut.ids.empty() || !cut.imgs.empty() || cut.stage_parts.size() + 1 != stages_.size()) {
        error = "resident preparation: incomplete checkpoint stages"; return false;
    }
    plan::LogicalBytes capacity;
    uint64_t retained_bytes = 0;
    for (size_t stage = 0; stage < stages_.size(); ++stage) {
        const auto& part = stage ? cut.stage_parts[stage - 1] : cut;
        if (part.ids != cut.ids || !part.imgs.empty() || (stage && !part.stage_parts.empty())) {
            error = "resident preparation: checkpoint parts describe different prefixes"; return false;
        }
        if (!core::conversation_checkpoint_validate(part, *stages_[stage].slots[(size_t) slot], geometry_, error))
            return false;
        const auto bytes = [](const core::CheckpointBytes& value) { return value.empty() ? size_t(0) : value.capacity(); };
        retained_bytes = capacity.sum({retained_bytes, sizeof(core::ConversationCheckpoint),
            5 * core::CheckpointBytes::owner_bytes, capacity.product({part.ids.size(), sizeof(int32_t)}),
            bytes(part.gdn), bytes(part.ple), bytes(part.tails), bytes(part.dead), bytes(part.block_pos)});
    }
    if (!capacity.valid || retained_bytes > checkpoint_byte_limit_) {
        error = "resident preparation: checkpoint capacity exceeds admitted allowance"; return false;
    }
    return true;
}

template<class Token>
bool ResidentSlots::select_checkpoints(const std::vector<core::ConversationCheckpoint>& cuts,
        const std::vector<Token>& tokens, int64_t limit, int slot,
        std::vector<core::ConversationCheckpoint>& selected, std::string& error) const {
    const core::ConversationCheckpoint* root = nullptr;
    const core::ConversationCheckpoint* latest = nullptr;
    size_t count = 0;
    for (const auto& cut : cuts) {
        if (!preparation_prefix(cut, tokens, limit)) continue;
        if (!validate_checkpoint(cut, slot, error)) return false;
        if (!root || cut.ids.size() < root->ids.size()) root = &cut;
        if (!latest || cut.ids.size() > latest->ids.size()) latest = &cut;
        if (cut.pinned) ++count;
    }
    if (root && !root->pinned) ++count;
    if (latest && latest != root && !latest->pinned) ++count;
    if (count > (uint64_t) checkpoint_chain_limit_) {
        error = "resident preparation: selected checkpoint chain exceeds admitted limit"; return false;
    }
    selected.clear();
    selected.reserve(count);
    for (const auto& cut : cuts)
        if (preparation_prefix(cut, tokens, limit) && (cut.pinned || &cut == root || &cut == latest))
            selected.push_back(cut.frozen_alias());
    return true;
}

template<class Token>
bool ResidentSlots::validate_source(const core::SavedConversation& image, const core::ConversationCheckpoint*& cut,
        core::ConversationSnapshotExpectation expectation, int slot, const std::vector<Token>& tokens,
        const ResidentRequest& request, std::vector<core::ConversationCheckpoint>& selected,
        int64_t& position, std::string& error, ResidentSourceRefusal* refusal) const {
    const int64_t endpoint = (int64_t) tokens.size() - 1;
    const int64_t selected_limit = cut ? std::min(endpoint, (int64_t) cut->ids.size()) : endpoint;
    if (image.stage_images.size() + 1 != stages_.size() || image.cvec != request.steering ||
        !image.live.imgs.empty() ||
        (!image.scope.matches(request.scope) && !(request.pin > 0 && image.scope.matches(request.shared_scope)))) {
        error = "resident preparation: snapshot stage, steering or authorization mismatch"; return false;
    }
    const bool mtp = draft_.kv_state().max_cells > 0;
    for (size_t stage = 0; stage < stages_.size(); ++stage) {
        const auto& part = preparation_stage(image, stage);
        const auto* draft = mtp && stage + 1 == stages_.size() ? draft_.slot_state(slot) : nullptr;
        if (stage && (!part.stage_images.empty() || part.live.ids != image.live.ids ||
                      part.live.imgs != image.live.imgs || part.cvec != image.cvec)) {
            error = "resident preparation: snapshot stages describe different owners"; return false;
        }
        if (!core::conversation_snapshot_validate(part, *stages_[stage].slots[(size_t) slot],
                                                 geometry_, draft, expectation, error)) return false;
    }
    // Neither page-rounded payloads nor the live token count certify skipped
    // draft rows. Source certification is supplied by the capture/store owner;
    // unknown intervals are explicit preflight refusals, never inferred reuse.
    const auto& last = preparation_stage(image, stages_.size() - 1);
    const int64_t live = (int64_t) image.live.ids.size();
    const int64_t draft_limit = mtp ? last.draft_cells : live;
    const int64_t draft_begin = mtp ? last.draft_begin : 0;
    if (mtp && (draft_begin < 0 || draft_limit < draft_begin || draft_limit > live ||
                draft_limit > last.kv.back().cells)) {
        if (refusal) *refusal = ResidentSourceRefusal::uncertified_interval;
        error = "resident preparation: source draft reuse refused: uncertified retained interval"; return false;
    }
    const auto eligible = [&](const core::ConversationCheckpoint& candidate) {
        const int64_t cells = (int64_t) candidate.ids.size();
        const int64_t needed = mtp && draft_.window_ > 0 ?
            std::max<int64_t>(0, (int64_t) tokens.size() - draft_.window_) : 0;
        const bool catchup = mtp && cells == live && draft_limit == live - 1 &&
                             !last.draft_boundary_residual.empty();
        if (!preparation_prefix(candidate, tokens, selected_limit) || (cells > draft_limit && !catchup) ||
            (mtp && std::min(draft_begin, cells) > needed)) return false;
        if (catchup) return true; // destination tokenN is consumed only after its one-row catchup
        if (!mtp || draft_begin >= cells) return true; // an empty interval owes no boundary token
        const int32_t boundary = cells < live ? image.live.ids[(size_t) cells] : last.draft_next_token;
        return boundary >= 0 && boundary == tokens[(size_t) cells];
    };
    if (cut && !preparation_prefix(*cut, tokens, endpoint)) {
        error = "resident preparation: selected snapshot cut is not the request prefix"; return false;
    }
    if (cut && cut != &image.live && std::none_of(image.checkpoints.begin(), image.checkpoints.end(),
            [&](const auto& candidate) { return &candidate == cut; })) {
        error = "resident preparation: selected checkpoint does not belong to source"; return false;
    }
    if (!cut || !eligible(*cut)) {
        cut = eligible(image.live) ? &image.live : nullptr;
        for (const auto& candidate : image.checkpoints)
            if (eligible(candidate) && (!cut || candidate.ids.size() > cut->ids.size())) cut = &candidate;
    }
    position = cut ? (int64_t) cut->ids.size() : 0;
    selected.clear();
    if (!cut) {
        if (mtp && refusal) *refusal = ResidentSourceRefusal::incoherent_boundary;
        error = mtp ? "resident preparation: source draft reuse refused: no coherent checkpoint/boundary token" :
                      "resident preparation: source has no matching checkpoint";
        return false;
    }
    std::vector<core::ConversationCheckpoint> combined;
    const core::ConversationCheckpoint* root = cut;
    size_t count = cut->pinned ? 0 : 1;
    for (const auto& candidate : image.checkpoints)
        if (preparation_prefix(candidate, tokens, position)) {
            if (candidate.ids.size() < root->ids.size()) root = &candidate;
            if (candidate.pinned) ++count;
        }
    if (cut->pinned && std::none_of(image.checkpoints.begin(), image.checkpoints.end(),
            [&](const auto& candidate) { return &candidate == cut; })) ++count;
    if (root != cut && !root->pinned) ++count;
    if (count > (uint64_t) checkpoint_chain_limit_) {
        error = "resident preparation: snapshot checkpoint chain exceeds admitted limit"; return false;
    }
    combined.reserve(count);
    const auto append = [&](const core::ConversationCheckpoint& candidate) {
        core::ConversationCheckpoint whole = candidate.frozen_alias();
        if (whole.stage_parts.empty()) {
            whole.stage_parts.reserve(stages_.size() - 1);
            for (size_t stage = 1; stage < stages_.size(); ++stage) {
                const auto* part = preparation_stage_cut(image, candidate, stage, stages_.size());
                if (!part) { error = "resident preparation: missing snapshot checkpoint part"; return false; }
                whole.stage_parts.push_back(part->frozen_alias());
            }
        }
        if (!validate_checkpoint(whole, slot, error)) return false;
        combined.push_back(std::move(whole));
        return true;
    };
    bool kept_cut = false;
    for (const auto& candidate : image.checkpoints)
        if (preparation_prefix(candidate, tokens, position) &&
            (candidate.pinned || &candidate == root || &candidate == cut)) {
            if (!append(candidate)) return false;
            kept_cut = kept_cut || &candidate == cut;
        }
    if (!kept_cut && !append(*cut)) return false;
    selected.swap(combined);
    return true;
}

bool ResidentSlots::reserve(const ResidentPrepareInput* inputs, int count, ResidentPreparation* leases,
                            std::string& error) {
    if (!prefill::packed_prefill_enabled()) {
        error = "resident preparation: STRATA_PACKED_PREFILL=1 is required"; return false;
    }
    if (!inputs || !leases || count < 1 || count > 4 || stages_.empty() || stages_.size() > 4 ||
        next_request_ > UINT64_MAX - (uint64_t) count) {
        error = "resident preparation: invalid reservation extent"; return false;
    }
    std::array<PreparationSource, 4> sources{};
    std::array<int64_t, 4> positions{};
    std::array<std::vector<core::ConversationCheckpoint>, 4> selected;
    try {
        for (int index = 0; index < count; ++index) {
            const auto& input = inputs[index];
            if (!available(input.slot) || !input.tokens || input.tokens->empty() ||
                input.tokens->size() >= (uint64_t) context_ || input.request.images ||
                slots_[(size_t) input.slot].incarnation == UINT64_MAX) {
                error = "resident preparation: invalid or unavailable slot/request"; return false;
            }
            for (int earlier = 0; earlier < index; ++earlier)
                if (inputs[earlier].slot == input.slot) {
                    error = "resident preparation: duplicate destination"; return false;
                }
            if (std::any_of(input.tokens->begin(), input.tokens->end(),
                    [](int64_t token) { return token < 0 || token > INT32_MAX; })) {
                error = "resident preparation: token outside int32 range"; return false;
            }
            plan::RequestDemand demand;
            demand.prompt_tokens = (int64_t) input.tokens->size();
            demand.max_new_tokens = input.request.max_new;
            const auto admission = plan::admit_request(demand, context_);
            if (input.request.max_new < 1 || !admission.admitted) {
                error = admission.admitted ? "resident preparation: empty output budget" : admission.reason;
                return false;
            }
            const auto& state = slots_[(size_t) input.slot];
            auto& source = sources[(size_t) index];
            auto& position = positions[(size_t) index];
            const int64_t endpoint = (int64_t) input.tokens->size() - 1;
            if (input.snapshot) {
                source.image = input.snapshot; source.cut = input.cut; source.expectation = input.expectation;
                if (!validate_source(*source.image, source.cut, source.expectation, input.slot, *input.tokens,
                                     input.request, selected[(size_t) index], position, error)) return false;
            } else if (input.cut) {
                error = "resident preparation: snapshot cut without source"; return false;
            } else if (state.reusable(input.request.scope, input.request.steering) && !state.tainted) {
                const bool mtp = draft_.kv_state().max_cells > 0;
                const auto boundary_matches = [&](int64_t cells) {
                    if (!mtp || draft_.slot_begin(input.slot) >= cells) return true;
                    const int32_t boundary = cells < (int64_t) state.ids.size() ? state.ids[(size_t) cells] :
                        draft_.slot_next_token(input.slot);
                    return boundary >= 0 && boundary == (*input.tokens)[(size_t) cells];
                };
                if (!state.ids.empty() && (int64_t) state.ids.size() <= endpoint &&
                    std::equal(state.ids.begin(), state.ids.end(), input.tokens->begin()) &&
                    (!mtp || draft_.slot_covers(input.slot, (int64_t) state.ids.size(), (int64_t) input.tokens->size())) &&
                    boundary_matches((int64_t) state.ids.size()))
                    position = (int64_t) state.ids.size();
                for (const auto& cut : state.checks)
                    if (preparation_prefix(cut, state.ids, (int64_t) state.ids.size()) &&
                        preparation_prefix(cut, *input.tokens, endpoint) && (int64_t) cut.ids.size() > position &&
                        (!mtp || draft_.slot_covers(input.slot, (int64_t) cut.ids.size(), (int64_t) input.tokens->size())) &&
                        boundary_matches((int64_t) cut.ids.size()))
                        position = (int64_t) cut.ids.size();
                source.retained = position > 0;
                if (mtp && position > draft_.slot_begin(input.slot)) source.tail_token = (int32_t) (*input.tokens)[(size_t) position];
                if (!select_checkpoints(state.checks, *input.tokens, position, input.slot,
                                        selected[(size_t) index], error)) return false;
                if (position && position != (int64_t) state.ids.size()) {
                    const auto& cuts = selected[(size_t) index];
                    const auto at = std::find_if(cuts.begin(), cuts.end(),
                        [&](const auto& cut) { return (int64_t) cut.ids.size() == position; });
                    if (at == cuts.end()) { error = "resident preparation: selected rewind cut was not retained"; return false; }
                    source.rewind = (int) (at - cuts.begin());
                }
            }
        }
    } catch (const std::bad_alloc&) {
        error = "resident preparation: checkpoint preflight allocation failed"; return false;
    }
    if (!idle(error)) return false;
    for (int index = 0; index < count; ++index) {
        const auto& input = inputs[index];
        auto& state = slots_[(size_t) input.slot];
        const int64_t position = positions[(size_t) index];
        retire(input.slot);
        ++state.incarnation;
        state.request = ++next_request_;
        state.sampling = input.request.sampling;
        if (!state.sampling.seed) state.sampling.seed = (uint64_t) Clock::now().time_since_epoch().count();
        state.input = input.request;
        state.prompt.assign(input.tokens->begin(), input.tokens->end());
        state.ids.assign(input.tokens->begin(), input.tokens->begin() + position);
        state.checks.swap(selected[(size_t) index]);
        state.active = state.cached = state.partial = state.failed = state.tainted = state.stop = false;
        state.preparing = true;
        state.foreground = true;
        state.cvec = input.request.steering; state.img = false;
        state.p = position; state.produced = 0; state.max_new = input.request.max_new;
        state.x = 0; state.draft = -1; state.original_slot = input.slot;
        state.partial_from0 = false; state.prefill_split_until = 0; state.prefill_plan = {};
        state.t0 = Clock::now();
        preparations_[(size_t) input.slot] = sources[(size_t) index];
        ResidentPreparation lease;
        lease.slot = input.slot; lease.request = state.request; lease.incarnation = state.incarnation;
        lease.position = position; lease.endpoint = (int64_t) state.prompt.size() - 1;
        for (size_t stage = 0; stage < stages_.size(); ++stage) lease.stages[stage] = stages_[stage].slots[(size_t) input.slot];
        leases[index] = lease;
    }
    return true;
}

bool ResidentSlots::prepare(int slot, const std::vector<int64_t>& tokens, const ResidentRequest& request,
                            ResidentPreparation& lease, std::string& error) {
    const ResidentPrepareInput input{slot, &tokens, request};
    return reserve(&input, 1, &lease, error) && prepare(lease, error);
}

bool ResidentSlots::prepare(const ResidentPreparation& lease, std::string& error) {
    if (!preparation_valid(lease, error)) return false;
    const auto& source = preparations_[(size_t) lease.slot];
    return prepare(lease, source.image, source.cut, source.expectation, error);
}

bool ResidentSlots::prepare(const ResidentPreparation& lease, const core::SavedConversation* snapshot,
        const core::ConversationCheckpoint* cut, core::ConversationSnapshotExpectation expectation,
        std::string& error, ResidentSourceRefusal* refusal) {
    if (refusal) *refusal = ResidentSourceRefusal::none;
    if (!preparation_valid(lease, error)) return false;
    auto& source = preparations_[(size_t) lease.slot];
    auto& state = slots_[(size_t) lease.slot];
    if (source.ready || source.mutated) { error = "resident preparation: destination already written"; return false; }
    std::vector<core::ConversationCheckpoint> selected;
    int64_t position = state.p;
    try {
        if (snapshot && !validate_source(*snapshot, cut, expectation, lease.slot, state.prompt,
                                         state.input, selected, position, error, refusal)) return false;
    } catch (const std::bad_alloc&) {
        error = "resident preparation: snapshot preflight allocation failed"; return false;
    }
    if (cut && !snapshot) { error = "resident preparation: cut without snapshot"; return false; }
    if (!idle(error)) return false;
    if (snapshot) {
        source.image = snapshot; source.cut = cut; source.expectation = expectation;
        source.retained = false; source.rewind = -1;
        state.checks.swap(selected);
        state.ids.assign(state.prompt.begin(), state.prompt.begin() + position);
        state.p = position;
    }
    source.mutated = touched_ = true;
    state.failed = true;
    dispatch_.slot_requests[lease.slot] = lease.request;
    if (!stages_[0].verifier->set_slot_context(lease.slot, lease.request, state.sampling, &state.ids, state.cvec, error) ||
        (stages_[0].batch_peer && !stages_[0].batch_peer->set_slot_context(lease.slot, lease.request,
                                                  state.sampling, &state.ids, state.cvec, error)) ||
        (stages_[0].batch_third && !stages_[0].batch_third->set_slot_context(lease.slot, lease.request,
                                                  state.sampling, &state.ids, state.cvec, error)) ||
        !prompt_.bind_owner(lease.slot, lease.request, lease.incarnation, lease.stages, error) ||
        !draft_.bind_request(lease.slot, lease.request, error)) return fail(lease.slot);
    try {
        const bool mtp = draft_.kv_state().max_cells > 0;
        if (source.image && position > 0) {
            std::optional<core::MtpDrafter::Operation> operation;
            if (mtp) {
                const core::OnDevice on(draft_.device());
                operation.emplace(draft_, error, lease.slot);
                if (!*operation) return fail(lease.slot);
            }
            for (size_t stage = 0; stage < stages_.size(); ++stage) {
                const auto& image = preparation_stage(*source.image, stage);
                const auto* part = preparation_stage_cut(*source.image, *source.cut, stage, stages_.size());
                const auto* draft = mtp && stage + 1 == stages_.size() ? draft_.slot_state(lease.slot) : nullptr;
                const core::OnDevice on(stages_[stage].device);
                if (!part || core::conversation_snapshot_restore(image, *lease.stages[stage], geometry_,
                        draft, source.expectation, error, part == &image.live ? nullptr : part) !=
                            core::ConversationRestore::restored) return fail(lease.slot);
            }
            if (mtp) {
                const core::OnDevice on(draft_.device());
                if (!operation->finish(error)) return fail(lease.slot);
                auto& target = draft_.slots_[(size_t) lease.slot];
                const auto& last = preparation_stage(*source.image, stages_.size() - 1);
                target.cells = std::min(position, last.draft_cells);
                target.completed_cells = target.cells;
                target.begin = std::min(target.cells, last.draft_begin);
                target.certified = true;
                target.tail_token = target.cells == position && position > target.begin
                    ? state.prompt[size_t(position)] : -1; // token identity was proven by source preflight
                target.prompt_len = (int64_t) state.prompt.size();
                target.proposal = target.proposal_token = -1; target.proposal_cell = -1;
                // The immutable prefix never stores tokenN. Advance exactly its
                // missing cell with this destination's tokenN, using the existing
                // Rin_ arena and KV-only front, before advertising full coverage.
                if (target.cells != position &&
                    !draft_.catchup_slot(lease.slot, last.draft_boundary_residual,
                        state.prompt[size_t(position)], position - 1, error)) return fail(lease.slot);
            }
        } else {
            for (size_t stage = 0; stage < stages_.size(); ++stage) {
                const core::OnDevice on(stages_[stage].device);
                if (!source.retained) {
                    core::session_zero(*lease.stages[stage], geometry_, nullptr, nullptr);
                    if (cudaStreamSynchronize(nullptr) != cudaSuccess) {
                        error = "resident preparation: destination reset failed"; return fail(lease.slot);
                    }
                } else if (source.rewind >= 0) {
                    const auto& whole = state.checks[(size_t) source.rewind];
                    const auto& part = stage ? whole.stage_parts[stage - 1] : whole;
                    if (!core::conversation_checkpoint_restore(part, *lease.stages[stage], geometry_, error))
                        return fail(lease.slot);
                }
            }
            if (mtp && !draft_.prepare_slot(lease.slot, lease.request, (int64_t) state.prompt.size(),
                                            position, !source.retained, error)) return fail(lease.slot);
            if (mtp) draft_.slots_[(size_t) lease.slot].tail_token = source.tail_token;
        }
    } catch (const std::bad_alloc&) {
        error = "resident preparation: direct destination allocation failed"; return fail(lease.slot);
    }
    source.image = nullptr; source.cut = nullptr; source.ready = true;
    state.failed = false;
    residency("start", lease.slot);
    return true;
}

bool ResidentSlots::prefilled(const ResidentPreparation& lease, int64_t frontier, std::string& error) {
    if (!preparation_valid(lease, error)) return false;
    auto& state = slots_[(size_t) lease.slot];
    if (!preparations_[(size_t) lease.slot].ready || frontier < state.p || frontier > lease.endpoint) {
        error = "resident preparation: invalid certified frontier"; return false;
    }
    if (!idle(error)) return false;
    if (draft_.kv_state().max_cells > 0 &&
        (draft_.slot_cells(lease.slot) != frontier ||
         !draft_.slot_covers(lease.slot, frontier, (int64_t) state.prompt.size()) ||
         (frontier > draft_.slot_begin(lease.slot) &&
          draft_.slot_next_token(lease.slot) != state.prompt[(size_t) frontier]))) {
        error = "resident preparation: draft interval or boundary token is not coherent"; return fail(lease.slot);
    }
    state.ids.assign(state.prompt.begin(), state.prompt.begin() + frontier);
    state.p = frontier;
    for (size_t stage = 0; stage < stages_.size(); ++stage) {
        auto& session = *lease.stages[stage];
        session.ple_prev[0] = frontier >= 2 ? state.prompt[(size_t) frontier - 2] : -1;
        session.ple_prev[1] = frontier >= 1 ? state.prompt[(size_t) frontier - 1] : -1;
    }
    return true;
}

bool ResidentSlots::checkpoint(const ResidentPreparation& lease, core::ConversationCheckpoint& cut, std::string& error) {
    if (!preparation_valid(lease, error)) return false;
    auto& state = slots_[(size_t) lease.slot];
    if (!preparations_[(size_t) lease.slot].ready ||
        !preparation_prefix(cut, state.prompt, state.p) || !validate_checkpoint(cut, lease.slot, error)) {
        if (error.empty()) error = "resident preparation: invalid checkpoint frontier";
        return false;
    }
    try {
        std::vector<core::ConversationCheckpoint> candidates;
        candidates.reserve(state.checks.size() + 1);
        for (const auto& existing : state.checks)
            if (existing.ids != cut.ids) candidates.push_back(existing.frozen_alias());
        cut.pinned = cut.pinned || ((int64_t) cut.ids.size() == state.input.pin && state.input.pin > 0) ||
            std::any_of(state.checks.begin(), state.checks.end(),
                [&](const auto& existing) { return existing.pinned && existing.ids == cut.ids; });
        cut.freeze();
        candidates.push_back(cut.frozen_alias());
        std::vector<core::ConversationCheckpoint> selected;
        if (!select_checkpoints(candidates, state.prompt, lease.endpoint, lease.slot, selected, error)) return false;
        state.checks.swap(selected);
    } catch (const std::bad_alloc&) {
        error = "resident preparation: checkpoint retention allocation failed"; return false;
    }
    return true;
}

bool ResidentSlots::first_token(const ResidentPreparation& lease, int32_t next, const float* residual, std::string& error) {
    if (!preparation_valid(lease, error)) return false;
    auto& state = slots_[(size_t) lease.slot];
    if (!preparations_[(size_t) lease.slot].ready || state.p != lease.endpoint ||
        state.ids.size() != (size_t) lease.endpoint) {
        error = "resident first token: prefix endpoint not certified"; return false;
    }
    if (!slot_idle(lease.slot, error) || !prompt_.context_idle(error)) return false;
    for (const auto& stage : stages_)
        if (!stage.verifier->slot_committed_row(
                lease.slot, lease.request, lease.endpoint, (int32_t) state.prompt.back())) {
            error = "resident first token: verifier owner has not committed its final row"; return fail(lease.slot);
        }
    if (next < 0) { error = "resident first token: invalid verifier pick"; return fail(lease.slot); }
    if (draft_.kv_state().max_cells > 0 &&
        !draft_.advance_slot(lease.slot, residual, next, lease.endpoint, error)) return fail(lease.slot);
    state.ids.push_back(state.prompt.back());
    state.p = (int64_t) state.ids.size(); state.produced = 1; state.x = next;
    state.draft = -1;
    if (draft_.kv_state().max_cells > 0 && draft_.batch_drafting_) {
        const auto& target = draft_.slots_[(size_t) lease.slot];
        if (target.proposal_cell == lease.endpoint && target.proposal_token == next) state.draft = target.proposal;
    }
    state.preparing = false; state.active = true;
    preparations_[(size_t) lease.slot] = {};
    if (!observe("admit", lease.slot, state.p, error, residual)) return fail(lease.slot);
    return true;
}

bool ResidentSlots::abort_preparation(const ResidentPreparation& lease, std::string& error) {
    if (!preparation_valid(lease, error) || !idle(error)) return false;
    if (preparations_[(size_t) lease.slot].ready &&
        stages_[0].verifier->slot_committed_row(lease.slot, lease.request, lease.endpoint,
                                               (int32_t) slots_[(size_t) lease.slot].prompt.back())) {
        error = "resident preparation: final verifier row requires first-token draft catchup"; return false;
    }
    auto& state = slots_[(size_t) lease.slot];
    residency("cancel", lease.slot);
    retire(lease.slot);
    state.active = state.preparing = state.cached = state.partial = false;
    state.ids.clear(); state.prompt.clear(); state.checks.clear();
    state.p = state.produced = 0; state.draft = -1; state.prefill_split_until = 0;
    state.prefill_plan = {};
    preparations_[(size_t) lease.slot] = {};
    return true;
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
    const bool timing = admission_timing_enabled();
    auto phase = timing ? Clock::now() : Clock::time_point{};
    if (request.checkpoint_limit < -1 || working_failed_ || !available(slot) || !working_.request || prefix.empty() || prefix.size() >= (uint64_t) context_ ||
        (!parked && (input_.max_new <= 1 || destination_ != slot)) ||
        (parked && (!cache_ || input_.images || !working_prompt_ || (destination_ >= 0 && destination_ != slot) ||
         prefix.size() >= working_prompt_->size() || !std::equal(prefix.begin(), prefix.end(), working_prompt_->begin())))) {
        error = "resident admission: invalid or unavailable slot/request"; return false;
    }
    if (request.prefill_split_until != 0 &&
        (!parked || request.prefill_split_until <= (int64_t) prefix.size() ||
         request.prefill_split_until - (int64_t) prefix.size() != 1024 ||
         request.prefill_split_until > (int64_t) working_prompt_->size())) {
        error = "resident admission: invalid pending prefill interval"; return false;
    }
    const size_t checkpoint_limit = request.checkpoint_limit < 0 ? prefix.size() :
        std::min(prefix.size(), size_t(request.checkpoint_limit));
    const auto eligible = [&](const core::ConversationCheckpoint& checkpoint) {
        return !checkpoint.ids.empty() && checkpoint.ids.size() <= checkpoint_limit &&
            checkpoint.imgs.empty() &&
            std::equal(checkpoint.ids.begin(), checkpoint.ids.end(), prefix.begin());
    };
    core::ConversationCheckpoint* root = nullptr;
    core::ConversationCheckpoint* latest = nullptr;
    size_t selected_count = 0;
    if (request.checkpoints && !input_.images)
        for (auto& checkpoint : *request.checkpoints) {
            if (!eligible(checkpoint)) continue;
            if (!validate_checkpoint(checkpoint, slot, error)) return false;
            if (!root || checkpoint.ids.size() < root->ids.size()) root = &checkpoint;
            if (!latest || checkpoint.ids.size() > latest->ids.size()) latest = &checkpoint;
            if (checkpoint.pinned) ++selected_count;
        }
    if (root && !root->pinned) ++selected_count;
    if (latest && latest != root && !latest->pinned) ++selected_count;
    if (selected_count > (uint64_t) checkpoint_chain_limit_) {
        error = "resident admission: selected checkpoint chain exceeds admitted limit";
        return false;
    }
    for (size_t b = 0; b < slots_.size(); ++b)
        if ((int) b != slot && slots_[b].request == working_.request) {
            error = "resident admission: request already owns another slot"; return false;
        }
    if (!idle(error)) return false;
    if (timing) report_admission_timing(slot, "validation_idle", -1, 1, phase);
    auto& state = slots_[(size_t) slot];
    const bool continuing = state.request == working_.request && working_.resume_slot == slot;
    if (state.incarnation == UINT64_MAX) { error = "resident admission: seat incarnation exhausted"; return false; }
    if (timing) phase = Clock::now();
    // Capture possibly aliased checkpoint metadata before clearing the target.
    std::vector<core::ConversationCheckpoint> selected;
    try {
        selected.reserve(selected_count);
        if (request.checkpoints && !input_.images)
            for (auto& checkpoint : *request.checkpoints)
                if (eligible(checkpoint) && (checkpoint.pinned || &checkpoint == root || &checkpoint == latest)) {
                    checkpoint.freeze();
                    selected.push_back(checkpoint.frozen_alias());
                }
    } catch (const std::bad_alloc&) {
        error = "resident admission: checkpoint allocation failed"; return false;
    }
    if (timing) report_admission_timing(slot, "checkpoint_clone", -1, (int64_t) selected.size(), phase);
    if (timing) phase = Clock::now();
    // Retire before overwriting, but don't publish a new identity until every
    // consumer is bound. A yielded continuation keeps the existing owner.
    if (!continuing) {
        retire(slot);
        // #879: the taint belongs to the identity that ran, not to the seat. Its poison described the
        // retired run's cells, and this bind rebuilds them from the working context, so it must not sit
        // over the new identity's completion. A continuing rebind is the same identity and run: the
        // taint stays, and `complete` still refuses that run the cache.
        state.tainted = false;
    }
    ++state.incarnation;
    state.active = state.preparing = state.cached = state.partial = false;
    state.failed = true; // every exception after this point leaves an unavailable target
    state.prefill_split_until = 0;
    touched_ = true;
    state.checks.swap(selected);
    selected.clear();
    // Release the old top-level directory too: it must not overlap transfer
    // scratch, whose bound is a serial maximum rather than a sum.
    std::vector<core::ConversationCheckpoint>().swap(selected);
    if (timing) report_admission_timing(slot, "retire_checkpoint_bind", -1, 1, phase);
    try {
        if (!copy_to(slot, prefix, error)) return fail(slot);
        if (draft_.kv_state().max_cells > 0) {
            if (timing) phase = Clock::now();
            if (!draft_.bind_request(slot, working_.request, error)) return fail(slot);
            if (timing) report_admission_timing(slot, "draft_request_bind", draft_.device(), 1, phase);
            if (!draft_.save_slot(slot, (int64_t) prefix.size(), error)) return fail(slot);
        }
    } catch (const std::bad_alloc&) {
        error = "resident admission: transfer allocation failed"; return fail(slot);
    }
    if (timing) phase = Clock::now();
    state.ids.assign(prefix.begin(), prefix.end());
    if (parked) {
        state.prompt.assign(working_prompt_->begin(), working_prompt_->end());
        state.original_slot = destination_;
    } else state.prompt.clear();
    state.input = input_;
    state.sampling = working_.sampling;
    state.request = working_.request;
    state.x = request.next_token;
    state.draft = -1;
    state.p = (int64_t) prefix.size();
    state.produced = parked ? 0 : 1;
    state.max_new = input_.max_new;
    state.t0 = Clock::now();
    state.cvec = input_.steering;
    state.img = input_.images;
    state.stop = false;
    state.partial_from0 = parked && from_start;
    state.prefill_split_until = parked ? request.prefill_split_until : 0;
    state.prefill_plan = parked ? request.prefill_plan : core::ConversationCheckpoint::PromptPlan{};
    dispatch_.slot_requests[slot] = working_.request;
    if (timing) report_admission_timing(slot, "slot_metadata", -1, (int64_t) prefix.size(), phase);
    if (timing) phase = Clock::now();
    if (request.residual && !draft_.propose_slot(slot, request.residual, state.x, state.p - 1,
                                                state.draft, error)) return fail(slot);
    if (timing) report_admission_timing(slot, "initial_proposal", draft_.device(), request.residual ? 1 : 0, phase);
    if (timing) phase = Clock::now();
    if (!stages_[0].verifier->set_slot_context(slot, working_.request, state.sampling,
                                              &state.ids, state.cvec, error) ||
        (stages_[0].batch_peer && !stages_[0].batch_peer->set_slot_context(slot, working_.request,
                                              state.sampling, &state.ids, state.cvec, error)) ||
        (stages_[0].batch_third && !stages_[0].batch_third->set_slot_context(slot, working_.request,
                                              state.sampling, &state.ids, state.cvec, error)) ||
        !observe(continuing ? "resume" : "admit", slot, (int64_t) prefix.size(), error)) {
        // Retain a continuing identity for balanced failure/release receipts.
        // A new request has not published its start yet.
        if (!continuing) state.request = dispatch_.slot_requests[slot] = 0;
        return fail(slot);
    }
    if (timing) report_admission_timing(slot, "context_canary_bind", -1, 1, phase);
    if (timing) phase = Clock::now();
    if (!continuing) residency("start", slot);
    state.failed = false;
    state.active = !parked;
    state.cached = state.partial = parked;
    if (parked) residency("pause", slot);
    if (timing) report_admission_timing(slot, "publication", -1, 1, phase);
    return true;
}

bool ResidentSlots::copy_to(int slot, const std::vector<int32_t>& prefix, std::string& error) {
    const int64_t upto = (int64_t) prefix.size();
    const bool timing = admission_timing_enabled();
    for (const auto& stage : stages_) {
        auto& from = *stage.main;
        auto& to = *stage.slots[(size_t) slot];
        const core::OnDevice on(stage.device);
        core::admission_copy::Scope observed("main_admit", slot, stage.device, upto);
        auto phase = timing ? Clock::now() : Clock::time_point{};
        if (core::resident_direct_copy_enabled()) {
            // idle() joined consumers; session copy validates all layers before
            // owning the entry device fence on this stage's selected device.
            if (!core::conversation_session_copy(from, to, geometry_, prefix, nullptr, error)) return false;
            if (timing) report_admission_timing(slot, "main_direct_transfer", stage.device, from.qsa_alloc, phase);
            observed.complete();
            continue;
        }
        if (cudaDeviceSynchronize() != cudaSuccess) { error = "batch admission: device sync failed"; return false; }
        if (timing) report_admission_timing(slot, "stage_sync", stage.device, 1, phase);
        if (timing) phase = Clock::now();
        core::ConversationCheckpoint checkpoint;
        checkpoint.ids = prefix;
        if (!core::conversation_checkpoint_save(checkpoint, from, geometry_, error) ||
            !core::conversation_checkpoint_restore(checkpoint, to, geometry_, error)) return false;
        if (timing) report_admission_timing(slot, "checkpoint_transfer", stage.device, 1, phase);
        if (timing) phase = Clock::now();
        for (int64_t j = 0; j < from.qsa_alloc; ++j) {
            core::ConversationKv image;
            if (!core::conversation_kv_save(image, from.qsa_states[from.qsa_ord0 + j], geometry_, upto, true, error))
                return false;
            if (cudaDeviceSynchronize() != cudaSuccess) { error = "batch admission: device sync failed"; return false; }
            if (!core::conversation_kv_restore(image, to.qsa_states[to.qsa_ord0 + j], geometry_, upto, true, error))
                return false;
        }
        if (cudaDeviceSynchronize() != cudaSuccess) { error = "batch admission: device sync failed"; return false; }
        if (timing) report_admission_timing(slot, "main_kv_transfer", stage.device, from.qsa_alloc, phase);
        observed.complete();
    }
    return true;
}

std::optional<ResidentSnapshotSource> ResidentSlots::completed_snapshot_source(int slot, std::string& error) {
    error.clear();
    if (!retained(slot)) return std::nullopt;
    auto& source = slots_[(size_t) slot];
    if (source.partial || source.img || source.tainted || !source.input.scope.valid() || source.ids.empty())
        return std::nullopt;
    if (!idle(error)) return std::nullopt;
    const auto* draft = draft_.kv_state().max_cells > 0 ? draft_.slot_state(slot) : nullptr;
    if (draft_.kv_state().max_cells > 0 && !draft) {
        error = "resident snapshot: missing draft state"; return std::nullopt;
    }
    const auto certificate = certify_continuation(slot, nullptr, 0, 0, &error);
    if (!certificate.certified) return std::nullopt;
    ResidentSnapshotSource result{source, stages_, draft, slot, draft ? draft_.slot_begin(slot) : 0,
        draft ? draft_.slots_[size_t(slot)].completed_cells : 0, draft ? draft_.slot_next_token(slot) : -1};
    result.certificate = certificate;
    return result;
}

std::optional<ResidentSnapshotSource> ResidentSlots::prepared_snapshot_source(
        const ResidentPreparation& lease, std::string& error) {
    error.clear();
    if (!preparation_valid(lease, error)) return std::nullopt;
    auto& source = slots_[(size_t) lease.slot];
    if (!preparations_[(size_t) lease.slot].ready || source.stop || source.tainted || source.img ||
        !source.input.checkpoint || !source.input.scope.valid() || source.ids.empty())
        return std::nullopt;
    if (source.active || source.p != (int64_t) source.ids.size() || source.p < lease.position ||
        source.p > lease.endpoint || !std::equal(source.ids.begin(), source.ids.end(), source.prompt.begin()) ||
        dispatch_.slot_requests[lease.slot] != lease.request) {
        error = "resident prepared snapshot: incoherent owner frontier"; return std::nullopt;
    }
    if (!idle(error)) return std::nullopt;
    const auto* draft = draft_.kv_state().max_cells > 0 ? draft_.slot_state(lease.slot) : nullptr;
    if (draft_.kv_state().max_cells > 0) {
        if (!draft || draft_.slots_[(size_t) lease.slot].request != lease.request ||
            draft_.slot_cells(lease.slot) != source.p ||
            draft_.slots_[(size_t) lease.slot].completed_cells != source.p ||
            !draft_.slot_covers(lease.slot, source.p, (int64_t) source.prompt.size()) ||
            (source.p > draft_.slot_begin(lease.slot) &&
             draft_.slot_next_token(lease.slot) != source.prompt[(size_t) source.p])) {
            error = "resident prepared snapshot: incoherent draft owner or interval"; return std::nullopt;
        }
    }
    const auto certificate = certify_continuation(lease.slot, nullptr, lease.request, lease.incarnation, &error);
    if (!certificate.certified) return std::nullopt;
    ResidentSnapshotSource result{source, stages_, draft, lease.slot, draft ? draft_.slot_begin(lease.slot) : 0,
        draft ? draft_.slots_[(size_t) lease.slot].completed_cells : 0,
        draft ? draft_.slot_next_token(lease.slot) : -1};
    result.certificate = certificate;
    return result;
}

bool ResidentSlots::resume_work(int slot, const core::ConversationCheckpoint* checkpoint,
                                PrefixWork& work, std::string& error) const {
    work = {};
    if (working_failed_ || !working_.request || !retained(slot) ||
        !slots_[(size_t) slot].input.scope.matches(input_.scope)) {
        error = "resident resume: no valid idle prefix"; return false;
    }
    if (stale_continuation_) {
        stale_continuation_ = false;
        error = "resident resume: stale generation"; return false;
    }
    if (checkpoint && (checkpoint->stage_parts.size() + 1 != stages_.size() ||
        checkpoint->ids.size() > slots_[(size_t) slot].ids.size() ||
        !std::equal(checkpoint->ids.begin(), checkpoint->ids.end(), slots_[(size_t) slot].ids.begin()))) {
        error = "resident resume: checkpoint is not a retained prefix"; return false;
    }
    const int64_t cells = (int64_t) (checkpoint ? checkpoint->ids.size() : slots_[(size_t) slot].ids.size());
    const bool direct = core::resident_direct_copy_enabled();
    if (direct) work.path(-1, -2); // don't reuse host-bounce cost samples
    for (size_t k = 0; k < stages_.size(); ++k) {
        const auto& stage = stages_[k];
        const auto& from = *stage.slots[(size_t) slot];
        core::ConversationStateSizes z;
        if (cells < 1 || cells > from.max_cells || cells > stage.main->max_cells ||
            !core::conversation_session_sizes(geometry_, from, z, error)) return false;
        if (checkpoint) {
            const auto& part = k == 0 ? *checkpoint : checkpoint->stage_parts[k - 1];
            if (part.ids != checkpoint->ids || part.imgs != checkpoint->imgs ||
                !core::conversation_checkpoint_validate(part, *stage.main, geometry_, error)) return false;
        }
        work.path(stage.device, -1);
        const uint64_t state = z.gdn + (from.ple_hist ? z.ple : 0) +
            uint64_t(from.qsa_alloc) * (z.tail + z.dead + z.block_pos);
        work.add(state);
        if (!checkpoint && !direct) work.add(state); // live host bounce captures then restores
        work.add(uint64_t(from.qsa_alloc) * z.dead); // restore refreshes the moving pooled row too
        for (int64_t j = 0; j < from.qsa_alloc; ++j) {
            const auto& qsa = from.qsa_states[from.qsa_ord0 + j];
            std::array<uint64_t, 5> parts{};
            if (!core::conversation_kv_part_sizes(qsa, geometry_, cells, true, parts, error)) return false;
            work.path(stage.device, qsa.kv_mode);
            work.path(stage.device, int(qsa.n_slots));
            const auto& target = stage.main->qsa_states[stage.main->qsa_ord0 + j];
            work.path(stage.device, target.kv_mode);
            work.path(stage.device, int(target.n_slots));
            for (uint64_t n : parts) { work.add(n); if (!direct) work.add(n); }
        }
    }
    if (draft_.kv_state().max_cells > 0) {
        // A park copies every draft row its slot holds but certifies only the cells the chain completed
        // (MtpDrafter::save_slot), so the resumed resident prefix can legitimately reach past the end of
        // the certificate. The interval is therefore judged at its own frontier: asking it to reach the
        // resident prefix refused a real continuation, and "restoring yielded prefix" ended the engine on
        // 2026-10-18. The certificate stays - prefill_slot only advances a certified slot - and the read
        // that follows re-opens an interval at this frontier, exactly as a gap below `begin` starts one
        // (mtp.cpp slot_covers).
        const auto& chain = draft_.slots_[(size_t) slot];
        const int64_t frontier = chain.certified ? draft_.slot_resume_frontier(slot, cells) : cells;
        if (chain.certified &&
            (!working_prompt_ || !draft_.slot_covers(slot, frontier, (int64_t) working_prompt_->size()))) {
            error = "resident resume: draft interval does not cover this prompt"; return false;
        }
        if (chain.certified && frontier > draft_.slot_begin(slot)) {
            const auto& ids = slots_[(size_t) slot].ids;
            const int32_t boundary = frontier < (int64_t) ids.size() ? ids[(size_t) frontier] :
                draft_.slot_next_token(slot);
            if (boundary < 0 || boundary != (*working_prompt_)[(size_t) frontier]) {
                error = "resident resume: draft boundary token does not match this prompt"; return false;
            }
        }
        const auto* from = draft_.slot_state(slot);
        std::array<uint64_t, 5> parts{};
        if (!from || !core::conversation_kv_part_sizes(*from, geometry_, cells, false, parts, error)) return false;
        work.path(draft_.device(), from->kv_mode);
        work.path(draft_.device(), int(from->n_slots));
        work.path(draft_.device(), draft_.kv_state().kv_mode);
        work.path(draft_.device(), int(draft_.kv_state().n_slots));
        for (uint64_t n : parts) { work.add(n); if (!direct) work.add(n); }
    }
    return work.bytes != UINT64_MAX;
}

bool ResidentSlots::resume(int slot, const core::ConversationCheckpoint* checkpoint, std::string& error) {
    PrefixWork work;
    if (!resume_work(slot, checkpoint, work, error) || !idle(error)) return false;
    const auto cells = (int64_t) (checkpoint ? checkpoint->ids.size() : slots_[(size_t) slot].ids.size());
    working_failed_ = true; // any partial write or exception makes the working context unusable
    working_.prefill_split_until = 0;
    try {
        if (!copy_from(slot, checkpoint, error) ||
            (draft_.kv_state().max_cells > 0 && !draft_.restore_slot(slot, cells, error)) ||
            !observe("resume", slot, cells, error)) {
            return fail(slot);
        }
    } catch (const std::bad_alloc&) {
        error = "resident resume: transfer allocation failed";
        return fail(slot);
    }
    working_failed_ = false;
    if (draft_.coverage_.known() && draft_.coverage_.cells == cells && cells > draft_.coverage_.begin &&
        working_prompt_ && cells < int64_t(working_prompt_->size()))
        draft_.coverage_.next_token = (*working_prompt_)[size_t(cells)]; // resume_work proved this source boundary
    residency("resume", slot);
    slots_[(size_t) slot].restored(working_, slot, checkpoint == nullptr);
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
        core::admission_copy::Scope observed(at ? "main_resume_cut" : "main_resume_live", slot, stage.device, upto);
        if (core::resident_direct_copy_enabled()) {
            const auto* part = at ? (k == 0 ? at : &at->stage_parts[k - 1]) : nullptr;
            if (!core::conversation_session_copy(from, to, geometry_, prefix, part, error)) return false;
            observed.complete();
            continue;
        }
        if (cudaDeviceSynchronize() != cudaSuccess) { error = "batch slot restore: device sync failed"; return false; }
        if (!at) {
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
        // The retained slot may have completed the cut's pooled block. Restore
        // its selected spare-row repair last, after copying the later KV image.
        if (at && !core::conversation_checkpoint_restore(k == 0 ? *at : at->stage_parts[k - 1], to, geometry_, error))
            return false;
        if (cudaDeviceSynchronize() != cudaSuccess) { error = "batch slot restore: device sync failed"; return false; }
        observed.complete();
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
    if (draft_.kv_state().max_cells > 0 && !draft_.advance_slot(slot, residual, next, state.p, error)) return fail(slot);
    state.ids.push_back(state.x);
    if (diagnostic && !observe("commit", slot, (int64_t) state.ids.size(), error, residual)) return fail(slot);
    ++state.produced;
    state.x = next;
    state.draft = -1;
    if (draft_.kv_state().max_cells > 0 && draft_.batch_drafting_) {
        const auto& target = draft_.slots_[(size_t) slot];
        if (target.proposal_cell == state.p && target.proposal_token == next) state.draft = target.proposal;
    }
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


bool ResidentSlots::finish(int slot, std::string& error, bool diagnostic) {
    if (slot < 0 || (size_t) slot >= slots_.size() || !slots_[(size_t) slot].active) {
        error = "resident completion: no active request"; return false;
    }
    if (!slot_idle(slot, error)) return false;
    auto& state = slots_[(size_t) slot];
    if (state.stop) {
        if (diagnostic && !observe("cancel", slot, (int64_t) state.ids.size(), error)) return fail(slot);
        residency("cancel", slot);
    }
    state.complete(cache_);
    residency("pause", slot);
    return true;
}

bool ResidentSlots::observe(const char* phase, int slot, int64_t cells, std::string& error, const float* residual) {
    if (!std::getenv("STRATA_STATE_HASH")) return true;
    const core::QsaState empty_draft{};
    const auto* draft = draft_.kv_state().max_cells > 0 ? draft_.slot_state(slot) : &empty_draft;
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
    if (stages_.size() > 1) {
        uint64_t history = SessionFingerprint::basis;
        const auto& owner = slots_[(size_t) slot];
        for (int32_t token : owner.ids) history = fingerprint_word(history, (uint32_t) token);
        for (size_t stage = 0; stage < stages_.size(); ++stage) {
            const auto& private_state = std::strcmp(phase, "resume") == 0
                ? *stages_[stage].main : *stages_[stage].slots[(size_t) slot];
            SessionFingerprint part;
            const auto& private_draft = stage + 1 != stages_.size() ? empty_draft
                : std::strcmp(phase, "resume") == 0 ? draft_.kv_state() : *draft;
            if (!session_fingerprint(private_state, private_draft, geometry_, cells, part, error)) return false;
            std::fprintf(stderr, "STATE_STAGE phase=%s stage=%zu slot=%d request=%llu cells=%lld "
                                 "gdn=%016llx ple=%016llx kv=%016llx draft=%016llx history=%016llx "
                                 "seed=%llu counter=%llu\n", phase, stage, slot, (unsigned long long) id,
                         (long long) cells, (unsigned long long) part.gdn,
                         (unsigned long long) fingerprint_ple(part, private_state),
                         (unsigned long long) fingerprint_kv(part), (unsigned long long) part.draft,
                         (unsigned long long) history, (unsigned long long) owner.sampling.seed,
                         (unsigned long long) owner.sampling.counter);
        }
    }
    if (residual && std::getenv("STRATA_BATCH_DIAGNOSTICS")) report_batch_state(phase, id, slot, cells, session, h);
    return true;
}
} // namespace strata::program
