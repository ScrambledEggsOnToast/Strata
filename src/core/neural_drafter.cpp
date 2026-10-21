#include "strata/core/neural_drafter.hpp"
#include "strata/core/on_device.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <new>
#include <optional>

namespace strata::core {
NeuralDrafter::~NeuralDrafter() {
    const OnDevice on(device_);
    if (cs_ && cudaStreamSynchronize(cs_) != cudaSuccess) std::terminate();
    if (cs_ && cudaStreamDestroy(cs_) != cudaSuccess) std::terminate();
}
bool DraftProposal::validate(int64_t vocabulary, std::string& error, DraftKind kind) const {
    if (!owner.request || !owner.incarnation || !owner.generation || frontier < 0 ||
        count < 1 || count > kNeuralDraftMax || (kind == DraftKind::dflash2 && count > 3)) {
        error = "neural draft: unowned or unbounded proposal"; return false;
    }
    for (int row = 0; row < count; ++row) {
        if (tokens[row] < 0 || tokens[row] >= vocabulary) {
            error = "neural draft: proposal token outside vocabulary"; return false;
        }
        if (!std::isfinite(confidence[row]) || confidence[row] < 0 || confidence[row] > 1) {
            error = "neural draft: invalid draft-family chosen-token confidence"; return false;
        }
        if (distribution == Distribution::point_mass) continue;
        const int32_t* ids = q.data() + row * kSpecQStride;
        if (!validate_spec_q_row(ids, tokens[row], vocabulary, error)) return false;
        if (kind == DraftKind::dflash2 && (ids[15] < 0 || ids[16] != -1)) {
            error = "dflash2: conditional proposal requires exactly sixteen candidates"; return false;
        }
    }
    return true;
}

NeuralDrafter::Operation::Operation(NeuralDrafter& owner, std::string& error, int slot,
        cudaStream_t external, int32_t token, int32_t position, const DraftOwner* expected,
        const DraftOwner* borrowed_source, bool read_only)
    : owner_(owner), lease_(owner.scratch_), read_only_(read_only), slot_(slot), external_(external) {
    const OnDevice on(owner_.device_);
    if (!lease_) {
        error = lease_.failure() == OperationLease::Failure::timeout
            ? "neural draft: operation lease wait exceeded 2000 ms"
            : "neural draft: operation scratch is poisoned by a failed operation";
        return;
    }
    if (slot < 0 || slot > 8 || (slot != 8 && !owner_.slot_metadata(slot)) ||
        (expected && !owner_.owns(*expected)) || (borrowed_source && !owner_.owns(*borrowed_source)) ||
        owner_.chain_live_) {
        lease_.complete();
        error = "neural draft: stale owner or outstanding pipelined operation"; return;
    }
    if (owner_.pending_inputs_[size_t(slot)].active &&
        (read_only_ || !expected || owner_.pending_inputs_[size_t(slot)].ticket != *expected)) {
        lease_.complete(); error = "neural draft: target features still own this request frontier"; return;
    }
    if (owner_.async_owned_ && !owner_.idle_owned(error)) return;
    auto& identity = owner_.owners_[size_t(slot)];
    if (!read_only_ && identity.generation == UINT64_MAX) {
        lease_.complete(); error = "neural draft: operation generation exhausted"; return;
    }
    identity.seat = slot; if (!read_only_) ++identity.generation;
    identity_ = identity;
    owner_.pending_owner_ = identity_;
    acquired_ = true;
    owner_.pending_writes_ = owner_.pending_boundary_ = false;
    owner_.pending_slot_ = slot;
    owner_.canary_.begin(slot, owner_.cs_, token, position);
}

NeuralDrafter::Operation::~Operation() {
    if (acquired_ && !finished_) {
        const OnDevice on(owner_.device_);
        for (cudaStream_t stream : {external_, owner_.cs_})
            if (stream && cudaStreamSynchronize(stream) != cudaSuccess) std::terminate();
        std::string error;
        if (!owner_.drain_adapter(error)) std::terminate();
        owner_.invalidate_coverage();
    }
}

bool NeuralDrafter::Operation::finish(std::string& error) {
    if (!acquired_ || finished_) { error = "neural draft: operation was not acquired or is already complete"; return false; }
    const OnDevice on(owner_.device_);
    if (!owner_.owns(identity_)) { error = "neural draft: stale completion owner"; return false; }
    if (external_ && cudaStreamSynchronize(external_) != cudaSuccess) std::terminate();
    owner_.canary_.check(slot_, owner_.cs_);
    if (!owner_.idle_owned(error) || !owner_.canary_.validate(owner_.resource_name(), error)) return false;
    if (!owner_.owns(identity_)) { error = "neural draft: owner changed while draining"; return false; }
    const auto& input = owner_.pending_inputs_[size_t(slot_)];
    if (input.active && (!owner_.pending_writes_ ||
        input.ticket.request != identity_.request || input.ticket.incarnation != identity_.incarnation ||
        input.ticket.generation == UINT64_MAX || input.ticket.generation + 1 != identity_.generation ||
        owner_.pending_coverage_.cells <= input.start ||
        owner_.pending_coverage_.cells > input.start + input.count)) {
        error = "neural draft: target feature/context completion is absent or outside its reserved interval";
        return false;
    }
    owner_.complete_writes();
    finished_ = true; lease_.complete();
    return true;
}

void NeuralDrafter::Operation::writes(int64_t begin, int64_t end, int32_t token) {
    if (!acquired_ || finished_ || !owner_.owns(identity_)) return;
    if (read_only_) std::terminate(); // a read-only observation cannot publish numerical completion
    if (!owner_.pending_writes_) {
        if (slot_ == 8) owner_.pending_coverage_ =
            owner_.coverage_identity_.request == identity_.request &&
            owner_.coverage_identity_.incarnation == identity_.incarnation
                ? owner_.coverage_ : ConversationDraftCoverage{};
        else {
            const auto& source = *owner_.slot_metadata(slot_);
            owner_.pending_coverage_ = source.certified && source.producer.request == identity_.request &&
                source.producer.incarnation == identity_.incarnation
                ? ConversationDraftCoverage{source.begin, source.completed_cells, source.tail_token}
                : ConversationDraftCoverage{};
        }
    }
    owner_.pending_writes_ = true;
    if (owner_.kind() == DraftKind::dflash2) owner_.pending_coverage_.clear(); // exact currently retained five-layer ring
    owner_.pending_coverage_.completed(begin, end, token);
}

bool NeuralDrafter::Operation::boundary(const float* rows, int64_t begin, int64_t count, std::string& error) {
    if (owner_.kind() != DraftKind::mtp || slot_ != 8 || owner_.boundary_capture_ < begin ||
        owner_.boundary_capture_ - begin >= count) return true;
    if (!owner_.g_ || !rows || !owner_.owns(identity_)) {
        error = "neural draft: boundary source has no current owner"; return false;
    }
    const size_t bytes = size_t(owner_.g_->hc * owner_.g_->n_embd) * sizeof(float);
    owner_.boundary_position_ = -1; owner_.boundary_owner_ = 0;
    owner_.boundary_residual_.resize(bytes);
    const auto* source = reinterpret_cast<const uint8_t*>(rows +
        (owner_.boundary_capture_ - begin) * owner_.g_->hc * owner_.g_->n_embd);
    if (!owner_.boundary_residual_.visit(0, bytes, [&](uint8_t* dst, size_t n, size_t at) {
        return cudaMemcpyAsync(dst, source + at, n, cudaMemcpyDeviceToHost,
                               external_ ? external_ : owner_.cs_) == cudaSuccess;
    })) { error = "neural draft: boundary residual capture failed"; return false; }
    owner_.pending_boundary_ = true;
    return true;
}

DraftOwner NeuralDrafter::owner(int slot) const {
    return slot >= 0 && slot <= 8 ? owners_[size_t(slot)] : DraftOwner{};
}
bool NeuralDrafter::set_slot_sampling(int slot, const kernels::SamplerParams& sampling, std::string& error) {
    if (!enabled()) return true;
    if (slot < 0 || slot > 8 || !owner(slot).request) {
        error = "neural draft: sampling policy has no bound request"; return false;
    }
    OperationLease::Guard lease(scratch_, std::try_to_lock);
    if (!lease || async_owned_ || chain_live_) {
        if (lease) lease.complete();
        error = "neural draft: sampling rebind while operation is outstanding or poisoned"; return false;
    }
    sampling_[size_t(slot)] = sampling;
    lease.complete(); return true;
}
bool NeuralDrafter::owns(const DraftOwner& expected) const {
    return expected.seat >= 0 && expected.seat <= 8 && owners_[size_t(expected.seat)] == expected;
}

void NeuralDrafter::invalidate_slot_coverage(int slot) {
    if (slot == 8) { coverage_.clear(); coverage_owner_ = 0; coverage_identity_ = {}; return; }
    if (auto* state = slot_metadata(slot)) {
        state->certified = false; state->begin = state->completed_cells = -1; state->tail_token = -1;
        state->producer = {};
        state->proposal = state->proposal_token = -1; state->proposal_cell = -1;
    }
}
void NeuralDrafter::invalidate_coverage() {
    coverage_.clear(); coverage_owner_ = 0; coverage_identity_ = {};
    for (int slot = 0; slot < 8; ++slot) invalidate_slot_coverage(slot);
    pending_writes_ = pending_boundary_ = false;
    boundary_position_ = -1; boundary_owner_ = 0; boundary_residual_ = {};
}
void NeuralDrafter::complete_writes() {
    if (!owns(pending_owner_)) return;
    if (pending_writes_) {
        if (pending_slot_ == 8) {
            coverage_ = pending_coverage_; coverage_owner_ = working_request_; coverage_identity_ = pending_owner_;
        }
        if (auto* target = slot_metadata(pending_slot_)) {
            target->begin = pending_coverage_.begin; target->cells = pending_coverage_.cells;
            target->completed_cells = pending_coverage_.cells;
            target->tail_token = pending_coverage_.next_token; target->certified = pending_coverage_.known();
            target->producer = pending_owner_;
        }
    }
    if (pending_boundary_) {
        boundary_residual_.freeze(); boundary_position_ = boundary_capture_; boundary_owner_ = working_request_;
    }
    pending_writes_ = pending_boundary_ = false;
    pending_inputs_[size_t(pending_slot_)].active = false;
}

bool NeuralDrafter::idle_owned(std::string& error) {
    const OnDevice on(device_);
    const auto stream_status = cs_ ? cudaStreamSynchronize(cs_) : cudaSuccess;
    if (!drain_adapter(error)) std::terminate();
    if (stream_status != cudaSuccess) {
        // A failed per-stream join may report a completed fault. Returning the
        // poisoned owner is safe only after the entire device proves drained.
        if (cudaDeviceSynchronize() != cudaSuccess) std::terminate();
        invalidate_coverage();
        if (error.empty()) error = "neural draft: stream/transfer/context drain failed";
        return false;
    }
    if (async_owned_ && !chain_live_) {
        if (!canary_.validate(resource_name(), error) || !owns(pending_owner_)) {
            invalidate_coverage(); return false;
        }
        complete_writes(); async_owned_ = false;
    }
    return true;
}
bool NeuralDrafter::idle(std::string& error) {
    OperationLease::Guard lease(scratch_, std::try_to_lock);
    if (!lease || chain_live_ || inputs_pending()) {
        error = "neural draft: operation scratch is in flight or poisoned";
        if (lease) lease.complete(); return false;
    }
    if (!idle_owned(error)) return false;
    lease.complete(); return true;
}
bool NeuralDrafter::scratch_idle(std::string& error, bool wait) {
    OperationLease::Guard lease(scratch_, wait ? OperationLease::wait_limit : std::chrono::milliseconds(0));
    if (!lease || chain_live_ || inputs_pending()) {
        if (lease) lease.complete();
        error = "neural draft: operation scratch is in flight or poisoned"; return false;
    }
    if (!idle_owned(error)) return false;
    lease.complete(); return true;
}
bool NeuralDrafter::context_idle(std::string& error) { return scratch_idle(error, false); }
bool NeuralDrafter::slot_idle(int slot, std::string& error, bool wait) {
    OperationLease::Guard lease(scratch_, wait ? OperationLease::wait_limit : std::chrono::milliseconds(0));
    if (!lease || slot < 0 || slot > 8 || chain_live_ || inputs_pending(slot)) {
        if (lease) lease.complete();
        error = "neural draft: selected owner still has pending work or poisoned scratch"; return false;
    }
    if (!idle_owned(error)) return false;
    lease.complete(); return true;
}

bool NeuralDrafter::bind_request(int slot, uint64_t request, std::string& error, uint64_t incarnation) {
    if (!enabled()) return true;
    if (!request || slot < 0 || slot > 8 || (slot != 8 && !slot_metadata(slot))) {
        error = "neural draft: invalid request destination"; return false;
    }
    const OnDevice on(device_);
    OperationLease::Guard lease(scratch_, std::try_to_lock);
    if (!lease || async_owned_ || chain_live_ || inputs_pending()) {
        if (lease) lease.complete();
        error = "neural draft: owner rebind while operation is in flight or poisoned"; return false;
    }
    auto& current = owners_[size_t(slot)];
    if (!incarnation) {
        incarnation = current.request == request && current.incarnation ? current.incarnation : current.incarnation + 1;
        if (!incarnation) { lease.complete(); error = "neural draft: incarnation exhausted"; return false; }
    }
    if (incarnation < current.incarnation ||
        (current.request != request && incarnation <= current.incarnation) ||
        (kind() == DraftKind::dflash2 && incarnation > uint64_t(INT64_MAX))) {
        lease.complete(); error = "neural draft: stale seat incarnation"; return false;
    }
    if (!canary_.bind(slot, request, error)) return false;
    if (slot == 8) working_request_ = request;
    if (auto* target = slot_metadata(slot)) {
        if (current.request != request || current.incarnation != incarnation) {
            target->proposal = target->proposal_token = -1; target->proposal_cell = -1;
        }
        target->request = request;
    }
    current.seat = slot; current.request = request; current.incarnation = incarnation;
    lease.complete(); return true;
}
bool NeuralDrafter::retire(int slot, uint64_t request, uint64_t incarnation, std::string& error) {
    const auto current = owner(slot);
    if (!current.request) return true;
    if (current.request != request || current.incarnation != incarnation) {
        error = "neural draft: stale retirement"; return false;
    }
    if (!idle(error)) return false;
    invalidate_slot_coverage(slot);
    auto& state = owners_[size_t(slot)];
    state.request = 0; // retain incarnation and generation to reject late work after reuse
    return true;
}
bool NeuralDrafter::release_identity(int slot, uint64_t request, uint64_t incarnation, std::string& error) {
    if (!enabled()) return true;
    OperationLease::Guard lease(scratch_, std::try_to_lock);
    if (!lease) { error = "neural draft: identity fence scratch is busy/poisoned"; return false; }
    const auto current = owner(slot);
    if (!current.request) { lease.complete(); return true; }
    if (current.request != request || current.incarnation != incarnation || async_owned_ || chain_live_ ||
        inputs_pending()) {
        lease.complete(); error = "neural draft: stale identity fence or undrained input owner"; return false;
    }
    owners_[size_t(slot)].request = 0;
    lease.complete(); return true;
}

bool NeuralDrafter::reserve_inputs(int slot, const DraftOwner& expected, int64_t start, int64_t count,
                                   DraftOwner& ticket, std::string& error) {
    ticket = {};
    OperationLease::Guard lease(scratch_);
    if (!lease) { error = "neural draft: input reservation scratch is busy/poisoned"; return false; }
    if (!owns(expected) || expected.seat != slot || !expected.request || !expected.incarnation ||
        slot < 0 || slot > 8 || pending_inputs_[size_t(slot)].active || async_owned_ || chain_live_ ||
        start < 0 || count < 1 || start > state_capacity(slot) - count ||
        expected.generation == UINT64_MAX) {
        lease.complete(); error = "neural draft: stale/unbounded or overlapping input reservation"; return false;
    }
    const auto coverage = slot_coverage(slot);
    if (coverage.present && !coverage.coherent) {
        lease.complete(); error = "neural draft: input reservation has a foreign retained producer"; return false;
    }
    const int64_t frontier = coverage.present ? coverage.certified_end : snapshot_origin(slot);
    if (start != frontier) {
        lease.complete(); error = "neural draft: input reservation does not continue its actual frontier"; return false;
    }
    auto& current = owners_[size_t(slot)];
    ++current.generation; ticket = current;
    pending_inputs_[size_t(slot)] = {ticket, start, count, true};
    lease.complete(); return true;
}
bool NeuralDrafter::inputs_completed(int slot, int64_t start, int64_t count) const {
    const auto coverage = slot_coverage(slot);
    const int64_t origin = snapshot_origin(slot);
    if (!coverage.present || !coverage.coherent || start < origin || count < 0 ||
        start > state_capacity(slot) - count || start + count != coverage.certified_end) return false;
    return kind() == DraftKind::dflash2
        ? coverage.certified_begin == std::max(origin, coverage.certified_end - int64_t(2047))
        : start >= coverage.certified_begin;
}
bool NeuralDrafter::discard_inputs(const DraftOwner& ticket, std::string& error) {
    OperationLease::Guard lease(scratch_, std::try_to_lock);
    if (!lease) { error = "neural draft: input discard scratch is busy/poisoned"; return false; }
    if (!owns(ticket) || !ticket.request || !inputs_pending(ticket.seat) ||
        pending_inputs_[size_t(ticket.seat)].ticket != ticket || async_owned_ || chain_live_) {
        lease.complete(); error = "neural draft: stale/unowned input discard"; return false;
    }
    if (!discard_inputs_owned(ticket, error)) return false;
    lease.complete(); return true;
}
bool NeuralDrafter::discard_inputs_owned(const DraftOwner& ticket, std::string& error) {
    if (!owns(ticket) || !ticket.request || !inputs_pending(ticket.seat) ||
        pending_inputs_[size_t(ticket.seat)].ticket != ticket || async_owned_ || chain_live_) {
        error = "neural draft: stale/unowned input discard"; return false;
    }
    if (!idle_owned(error)) return false;
    pending_inputs_[size_t(ticket.seat)].active = false;
    invalidate_slot_coverage(ticket.seat);
    return true;
}

DraftCoverage NeuralDrafter::slot_coverage(int slot) const {
    const auto bound = owner(slot);
    DraftCoverage result; result.owner = bound;
    const auto* state = slot_metadata(slot);
    const auto interval = slot == 8 ? coverage_ : state && state->certified
        ? ConversationDraftCoverage{state->begin, state->completed_cells, state->tail_token}
        : ConversationDraftCoverage{};
    // The producer's claimed owner is independent of the current seat binding.
    // A malformed present certificate is not silently converted to "absent".
    result.owner = slot == 8 ? coverage_identity_ : state ? state->producer : DraftOwner{};
    result.present = (slot == 8 ? interval.known() : state && state->certified) &&
                     !async_owned_ && !inputs_pending(slot);
    result.materialized_begin = interval.begin;
    result.materialized_end = slot == 8 ? interval.cells : state ? state->cells : -1;
    result.certified_begin = interval.begin; result.certified_end = interval.cells;
    result.target_input_frontier = interval.cells; result.next_token = interval.next_token;
    result.coherent = !result.present || (interval.known() && bound.request &&
        result.owner.request == bound.request && result.owner.incarnation == bound.incarnation &&
        interval.cells <= result.materialized_end && result.materialized_end <= state_capacity(slot) &&
        (!state || state->request == bound.request));
    return result;
}
int64_t NeuralDrafter::slot_cells(int slot) const {
    const auto* state = slot_metadata(slot); return slot == 8 ? coverage_.cells : state ? state->cells : -1;
}
int64_t NeuralDrafter::slot_begin(int slot) const {
    const auto* state = slot_metadata(slot);
    return slot == 8 ? coverage_.begin : state && state->certified ? state->begin : -1;
}
int64_t NeuralDrafter::slot_prompt_len(int slot) const {
    const auto* state = slot_metadata(slot); return slot == 8 ? prompt_len_ : state ? state->prompt_len : -1;
}
int32_t NeuralDrafter::slot_next_token(int slot) const {
    const auto* state = slot_metadata(slot); return slot == 8 ? coverage_.next_token : state ? state->tail_token : -1;
}
int64_t NeuralDrafter::slot_resume_frontier(int slot, int64_t cells) const {
    const auto coverage = slot_coverage(slot);
    return coverage.present ? std::min(cells, coverage.certified_end) : cells;
}
bool NeuralDrafter::slot_covers(int slot, int64_t cells, int64_t prompt_length) const {
    const auto coverage = slot_coverage(slot);
    if (!coverage.present || !coverage.owner.request || cells < coverage.certified_begin ||
        cells > coverage.certified_end || prompt_length < 0 || prompt_length > state_capacity(slot) ||
        (prompt_length && cells >= prompt_length)) return false;
    if (!prompt_length) return true;
    const int64_t needed = kind() == DraftKind::dflash2
        ? std::max(snapshot_origin(slot), cells - int64_t(2047))
        : window_ > 0 ? std::max<int64_t>(0, prompt_length - window_) : 0;
    return coverage.certified_begin <= std::min(cells, needed);
}
void NeuralDrafter::restore_boundary_token(int slot, int32_t token) {
    if (kind() != DraftKind::mtp) return;
    if (slot == 8) { if (coverage_.known()) coverage_.next_token = token; }
    else if (auto* s = slot_metadata(slot)) s->tail_token = token;
}
bool NeuralDrafter::cached_proposal(int slot, int32_t anchor, int64_t cell,
                                    DraftProposal& out, std::string& error) const {
    out.clear();
    const auto* state = slot_metadata(slot);
    const auto coverage = slot_coverage(slot);
    if (!state || !coverage.present || !coverage.coherent || state->proposal < 0 ||
        state->proposal_cell != cell || state->proposal_token != anchor) return true;
    out.owner = owner(slot); out.frontier = cell + 1;
    out.count = 1; out.tokens[0] = state->proposal;
    out.confidence[0] = state->proposal_confidence;
    return out.validate(g_ ? g_->n_vocab : 248320, error, kind());
}
bool NeuralDrafter::state_fingerprint(int slot, int64_t cells, uint64_t& hash, std::string& error) {
    hash = 14695981039346656037ull;
    const auto* state = legacy_state(slot);
    if (!enabled()) return true;
    if (!state || !g_) { error = "neural draft: missing fingerprint storage"; return false; }
    const auto expected = owner(slot);
    Operation operation(*this, error, slot, nullptr, 0, 0, &expected, nullptr, true);
    ConversationKv image;
    if (!operation || !conversation_kv_save(image, *state, *g_, cells, false, error)) return false;
    for (const auto* part : {&image.k, &image.v, &image.k_scale, &image.v_scale})
        if (!part->visit(0, part->size(), [&](const uint8_t* p, size_t n, size_t) {
            for (size_t i = 0; i < n; ++i) hash = (hash ^ p[i]) * 1099511628211ull;
            return true;
        })) return false;
    return operation.finish(error);
}
void NeuralDrafter::set_boundary_capture(int64_t position) {
    boundary_capture_ = position; boundary_position_ = -1; boundary_owner_ = 0; boundary_residual_ = {};
}
bool NeuralDrafter::capture_boundary(const float* residual, int64_t position, uint64_t request, std::string& error) {
    if (kind() != DraftKind::mtp || !residual || !request || request != working_request_ ||
        coverage_owner_ != working_request_ || !coverage_.known() || position < coverage_.begin ||
        coverage_.cells != position) { error = "neural draft: boundary owner/position mismatch"; return false; }
    const OnDevice on(device_);
    Operation operation(*this, error);
    if (!operation) return false;
    set_boundary_capture(position);
    return operation.boundary(residual, position, 1, error) && operation.finish(error);
}
void NeuralDrafter::certify_view(ConversationView& view, int slot, const DraftOwner* displaced_source) const {
    const auto* supplied_boundary = view.draft_boundary_residual;
    const int64_t supplied_position = view.draft_boundary_position;
    const uint64_t supplied_owner = view.draft_boundary_owner;
    const auto interval = slot_coverage(slot);
    const auto bound = owner(slot);
    const bool displaced = displaced_source && slot == 8 && displaced_source->seat == slot &&
        displaced_source->request == interval.owner.request &&
        displaced_source->incarnation == interval.owner.incarnation &&
        displaced_source->generation == bound.generation &&
        interval.owner.generation <= displaced_source->generation &&
        interval.certified_begin >= 0 && interval.certified_end >= interval.certified_begin &&
        interval.certified_end <= interval.materialized_end && interval.materialized_end <= state_capacity(slot);
    view.draft_tokens = view.draft_begin = -1; view.draft_next_token = -1; view.draft_owner = 0;
    view.draft_boundary_residual = nullptr; view.draft_boundary_position = -1; view.draft_boundary_owner = 0;
    view.draft_identity = identity_; view.draft_origin = snapshot_origin(slot);
    if (!interval.present || (!interval.coherent && !displaced)) return;
    const int64_t live = int64_t(view.ids.size());
    const int64_t end = view.exact_cut && kind() == DraftKind::mtp ? live - 1 :
        std::min(interval.certified_end, live);
    if (end < interval.certified_begin || end > interval.certified_end) return;
    if (kind() == DraftKind::dflash2 && end != live) return;
    view.draft_owner = interval.owner.request;
    view.draft_begin = interval.certified_begin; view.draft_tokens = end;
    if (kind() == DraftKind::mtp && !view.exact_cut && end == live && interval.certified_end == live)
        view.draft_next_token = interval.next_token;
    if (kind() == DraftKind::mtp && end == live && view.draft_next_token < 0) {
        view.draft_tokens = view.draft_begin = -1; return;
    }
    if (kind() == DraftKind::mtp && end == live - 1 && supplied_boundary &&
        supplied_boundary->frozen() && supplied_position == end && supplied_owner == interval.owner.request) {
        // Private resident capture already owns this actual immutable verifier R.
        // Recertification may preserve its witness, never invent or retag one.
        view.draft_boundary_residual = supplied_boundary;
        view.draft_boundary_position = supplied_position; view.draft_boundary_owner = supplied_owner;
    }
    if (kind() == DraftKind::mtp && slot == 8 && end == live - 1 && boundary_owner_ == coverage_owner_ &&
        boundary_owner_ && boundary_position_ == live - 1 && !boundary_residual_.empty()) {
        view.draft_boundary_residual = &boundary_residual_;
        view.draft_boundary_position = boundary_position_; view.draft_boundary_owner = boundary_owner_;
    }
}
void NeuralDrafter::restored_snapshot(const SavedConversation& image, int64_t cells) {
    // Only a completed restore may install the producer's certificate. Unknown
    // coverage remains unknown; neither image size nor a token witness creates it.
    coverage_ = {image.draft_begin, image.draft_cells, image.draft_next_token};
    coverage_owner_ = working_request_; coverage_identity_ = owner(8);
    if (!coverage_.known() || cells < 0 ||
        (image.draft_identity.kind != DraftKind::none && !identity_.content_matches(image.draft_identity))) {
        coverage_.clear(); coverage_owner_ = 0; coverage_identity_ = {};
    }
    else if (cells < coverage_.cells) {
        coverage_.rewind(cells);
        if (kind() == DraftKind::mtp && cells < int64_t(image.live.ids.size()) && cells > coverage_.begin)
            coverage_.next_token = image.live.ids[size_t(cells)];
    }
    set_boundary_capture(-1);
}
bool NeuralDrafter::propose_slot(int slot, const float* residual, int32_t anchor, int64_t cell,
                                 DraftProposal& out, std::string& error) {
    out.clear();
    if (!enabled()) return true;
    const auto current = owner(slot);
    const auto coverage = slot_coverage(slot);
    if (!current.request || !current.incarnation || !coverage.present || !coverage.coherent) {
        error = "neural draft: proposal has no completed current request owner"; return false;
    }
    if (!proposal_preflight(slot, residual, anchor, cell, error)) return false;
    Operation operation(*this, error, slot, nullptr, anchor, int32_t(cell), &current);
    if (!operation) return false;
    DraftProposal candidate;
    if (!propose_impl(operation, residual, anchor, cell, candidate, error)) return false;
    candidate.owner = operation.identity(); candidate.frontier = cell + 1;
    if (!candidate.validate(g_ ? g_->n_vocab : 248320, error, kind()) ||
        !inspect_proposal(operation, candidate, error) || !operation.finish(error)) return false;
    out.owner = candidate.owner; out.frontier = candidate.frontier; out.distribution = candidate.distribution;
    std::copy_n(candidate.tokens.begin(), candidate.count, out.tokens.begin());
    std::copy_n(candidate.confidence.begin(), candidate.count, out.confidence.begin());
    if (candidate.distribution == DraftProposal::Distribution::conditional_q)
        std::copy_n(candidate.q.begin(), candidate.count * kSpecQStride, out.q.begin());
    out.count = candidate.count; // publish only after complete validated delivery
    return true;
}

bool conversation_neural_read_limits(SessionReadLimits& limits, const SessionState& session,
        const ModelGeometry& geometry, const NeuralDrafter& draft, int slot, uint64_t max_tokens,
        uint64_t max_checkpoints, std::string& error) {
    if (!conversation_session_read_limits(limits, session, geometry,
            draft.enabled() ? draft.legacy_state(slot) : nullptr, max_tokens, max_checkpoints, error)) return false;
    limits.draft_identity = draft.enabled() ? draft.identity() : DraftIdentity{};
    if (draft.enabled() && draft.kind() == DraftKind::dflash2) {
        const uint64_t rows = std::min<uint64_t>(limits.max_tokens, 2047);
        const uint64_t bytes = rows * 8 * 128 * sizeof(uint16_t);
        limits.max_kv_layers += 5;
        for (int layer = 0; layer < 5; ++layer) limits.max_kv_bytes.push_back({bytes, bytes, 0, 0, 0});
        limits.max_draft_boundary_bytes = 0;
    }
    return true;
}
bool conversation_neural_save(SavedConversation& image, ConversationView view, const SessionState& session,
        const ModelGeometry& geometry, NeuralDrafter& draft, int slot, std::string& error,
        ConversationKvReuse reuse, size_t* reused_bytes, std::vector<ConversationRetainedPrefix>* retained,
        const DraftOwner* expected) {
    const auto bound = draft.owner(slot);
    if (draft.enabled() && (!bound.request || !bound.incarnation)) {
        error = "neural draft: unbound snapshot capture owner"; return false;
    }
    if (expected && *expected != bound) {
        // The successor may be logically reserved before its predecessor's
        // outgoing bytes are captured, but may not have issued another operation.
        draft.certify_view(view, slot, expected);
        if (expected->seat != slot || view.draft_owner != expected->request || view.draft_tokens < 0) {
            error = "neural draft: stale displaced snapshot source"; return false;
        }
    }
    if (!draft.context_idle(error)) return false;
    if (!expected || *expected == bound) draft.certify_view(view, slot);
    if (draft.enabled() && view.exact_cut && view.draft_tokens < 0) {
        error = "neural draft: exact cut has no actual producer certificate"; return false;
    }
    std::optional<NeuralDrafter::Operation> operation;
    if (draft.enabled()) {
        operation.emplace(draft, error, slot, nullptr, 0, 0, &bound, nullptr, true);
        if (!*operation) return false;
    }
    if (view.draft_tokens < 0) { view.draft_identity = {}; view.draft_origin = -1; }
    SavedConversation captured;
    try {
        {
            const OnDevice on(session.allocation_device);
            if (!conversation_snapshot_save(captured, view, session, geometry,
                    draft.enabled() ? draft.legacy_state(slot) : nullptr, error,
                    std::move(reuse), reused_bytes, retained)) return false;
        }
        if (operation && view.draft_tokens >= 0 && !draft.snapshot_capture(captured, *operation, error))
            return false;
    } catch (const std::bad_alloc&) {
        // Readback has not changed numerical state. Only this known host refusal
        // may release a healthy operation; any failed device drain remains fatal.
        std::string drain_error;
        if (operation && !operation->finish(drain_error)) std::terminate();
        throw;
    }
    if (operation && !operation->finish(error)) return false;
    image = std::move(captured);
    return true;
}

bool conversation_neural_validate(const SavedConversation& image, const SessionState& session,
        const ModelGeometry& geometry, const NeuralDrafter& draft, int slot,
        ConversationSnapshotExpectation expectation, std::string& error) {
    if (!draft.enabled())
        return conversation_snapshot_validate(image, session, geometry, nullptr, expectation, error);
    const int64_t cells = int64_t(image.live.ids.size());
    if (!draft.snapshot_preflight(image, slot, cells, expectation, error)) return false;
    return conversation_snapshot_validate(image, session, geometry, draft.legacy_state(slot), expectation,
                                          error, &draft.identity());
}
ConversationRestore conversation_neural_restore(const SavedConversation& image, SessionState& session,
        const ModelGeometry& geometry, NeuralDrafter& draft, int slot, int64_t prompt_length,
        int32_t next_token, ConversationSnapshotExpectation expectation, std::string& error,
        const ConversationCheckpoint* selected, const DraftOwner* expected) {
    const auto bound = draft.owner(slot);
    if ((expected && (*expected != bound || expected->seat != slot)) ||
        (draft.enabled() && (!bound.request || !bound.incarnation))) {
        error = "neural draft: stale/unbound snapshot restore owner"; return ConversationRestore::invalid;
    }
    if (!conversation_neural_validate(image, session, geometry, draft, slot, expectation, error))
        return ConversationRestore::invalid;
    if (!draft.enabled())
        return conversation_snapshot_restore(image, session, geometry, nullptr, expectation, error, selected);
    const int64_t cells = int64_t(selected ? selected->ids.size() : image.live.ids.size());
    const bool native = draft.kind() == DraftKind::dflash2;
    if (cells < image.draft_begin ||
        cells > image.draft_cells + (!native &&
            (!image.draft_boundary_residual.empty() ||
             (expectation == ConversationSnapshotExpectation::ordinary &&
              cells == int64_t(image.live.ids.size()) && image.draft_cells == cells - 1)) ? 1 : 0) ||
        (native && std::max(image.draft_origin, cells - int64_t(2047)) != image.draft_begin)) {
        error = "neural draft: selected cut needs absent/evicted certified draft cells";
        return ConversationRestore::invalid;
    }
    if (!native && cells <= image.draft_cells && cells > image.draft_begin) {
        const int32_t witness = cells < int64_t(image.live.ids.size())
            ? image.live.ids[size_t(cells)] : image.draft_next_token;
        if (witness < 0 || witness != next_token) {
            error = "neural draft: destination token mismatches the actual MTP boundary witness";
            return ConversationRestore::invalid;
        }
    }
    if (!draft.context_idle(error)) return ConversationRestore::invalid;
    NeuralDrafter::Operation operation(draft, error, slot, nullptr, next_token, int32_t(cells), &bound);
    if (!operation) return ConversationRestore::invalid;
    ConversationRestore result;
    {
        const OnDevice on(session.allocation_device);
        result = conversation_snapshot_restore(image, session, geometry, draft.legacy_state(slot),
                                                expectation, error, selected, &draft.identity());
    }
    if (result != ConversationRestore::restored) return result;
    if (!draft.snapshot_restore(image, operation, cells, prompt_length, next_token, expectation, error) ||
        !operation.finish(error)) return ConversationRestore::transfer_failed;
    return ConversationRestore::restored;
}
} // namespace strata::core
