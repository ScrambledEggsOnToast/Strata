#include "strata/program/resident_executor.hpp"
#include "strata/core/verify.hpp"
#include <cstdlib>

namespace strata::program {
namespace {
bool executor_trace() {
    static const bool enabled = [] {
        const auto* value = std::getenv("STRATA_EXECUTOR_TRACE");
        return value && std::string_view(value) == "1";
    }();
    return enabled;
}
void trace_work(const char* event, ResidentWorkHandle work, int slot) {
    if (executor_trace())
        std::fprintf(stderr, "RESIDENT_EXECUTOR event=%s slot=%d request=%llu incarnation=%llu work=%llu\n",
                     event, slot, (unsigned long long)work.request, (unsigned long long)work.incarnation,
                     (unsigned long long)work.generation);
}
}

ResidentSnapshotLease::ResidentSnapshotLease(ResidentSnapshotLease&& other) noexcept
    : work_(other.work_), issuer_(other.issuer_), generation_(std::exchange(other.generation_, 0)),
      source_(std::move(other.source_)) {
    other.source_.reset();
}

ResidentWorkHandle ResidentExecutor::submit(const ResidentPreparation& lease, const ExecutionPlan& plan,
                                           std::string& error) {
    error.clear();
    const ResidentWorkHandle refused{lease.request, lease.incarnation, 0};
    if (plan.refused || plan.mode != ServeMode::resident || plan.stages != int(resident_.stages_.size()) ||
        lease.slot < 0 || lease.slot >= plan.batch_slots) {
        error = "resident executor: refused or incompatible resident plan";
        return refused;
    }
    if (cursor_.borrow_generation || cursor_.final_pending ||
        (cursor_.generation && (!cursor_.drained || !cursor_.terminal_seen)) ||
        (cursor_.generation && lease_.request == lease.request && lease_.incarnation == lease.incarnation)) {
        error = "resident executor: previous call scope has not retired";
        return refused;
    }
    if (cursor_.generation == UINT64_MAX || !resident_.preparation_valid(lease, error)) {
        if (error.empty()) error = "resident executor: work generation exhausted";
        return refused;
    }
    const uint64_t generation = cursor_.generation + 1;
    lease_ = lease;
    cursor_ = {};
    cursor_.generation = generation;
    trace_work("submit", {lease.request, lease.incarnation, generation}, lease.slot);
    return {lease.request, lease.incarnation, generation};
}

bool ResidentExecutor::current(ResidentWorkHandle work, std::string& error) const {
    if (!work || work.request != lease_.request || work.incarnation != lease_.incarnation ||
        work.generation != cursor_.generation || lease_.slot < 0 ||
        size_t(lease_.slot) >= resident_.slots_.size()) {
        error = "resident executor: stale work handle";
        return false;
    }
    const auto& state = resident_.slots_[size_t(lease_.slot)];
    const bool retired = cursor_.drained && cursor_.retired && !state.request &&
        state.incarnation == lease_.incarnation && !state.active && !state.preparing;
    if (resident_.resources_ == ResidentSlots::Resources::failed || state.failed ||
        (!retired && !resident_.owns(lease_.slot, work.request, work.incarnation))) {
        error = "resident executor: stale or failed resident owner";
        return false;
    }
    for (size_t stage = 0; stage < lease_.stages.size(); ++stage) {
        const auto* physical = stage < resident_.stages_.size()
            ? resident_.stages_[stage].slots[size_t(lease_.slot)] : nullptr;
        if (lease_.stages[stage] != physical || (physical &&
            (!lease_.allocations[stage] || physical->allocation_id != lease_.allocations[stage] ||
             physical->allocation_device != resident_.stages_[stage].device))) {
            error = "resident executor: stage allocation generation changed";
            return false;
        }
    }
    return true;
}

ResidentObservation ResidentExecutor::result(ResidentWorkHandle work, ResidentEvent event, bool fresh) const {
    return {work, event, fresh, std::max<int64_t>(0, cursor_.frontier), cursor_.produced, -1, cursor_.reason};
}

ResidentObservation ResidentExecutor::terminal(ResidentWorkHandle work) {
    if (cursor_.produced < cursor_.retired_produced) {
        auto observation = result(work, ResidentEvent::idle);
        observation.committed = cursor_.retired_produced;
        return observation; // resources retired; only observe may advance the output watermark
    }
    const bool fresh = !std::exchange(cursor_.terminal_seen, true);
    return result(work, ResidentEvent::terminal, fresh);
}

ResidentObservation ResidentExecutor::observe(ResidentWorkHandle work, std::string& error) {
    error.clear();
    if (!current(work, error)) return result(work, ResidentEvent::refused);
    if (cursor_.drained && cursor_.produced == cursor_.retired_produced) return terminal(work);
    const auto& state = resident_.slots_[size_t(lease_.slot)];
    if (state.p < cursor_.frontier || state.produced < cursor_.produced || state.p != int64_t(state.ids.size())) {
        error = "resident executor: committed publication moved backwards or lacks history";
        return result(work, ResidentEvent::refused);
    }
    if (state.produced > cursor_.produced) {
        // ResidentSlots retains every committed input and the newest predictor token. Recover each
        // unobserved output in order even if a drained verifier window accepted multiple rows.
        const int64_t index = lease_.endpoint + 1 + cursor_.produced;
        if (state.preparing || state.produced < 1 || state.p != lease_.endpoint + state.produced ||
            index < 0 || index > int64_t(state.ids.size())) {
            error = "resident executor: incoherent committed token publication";
            return result(work, ResidentEvent::refused);
        }
        const int32_t token = index == int64_t(state.ids.size()) ? state.x : state.ids[size_t(index)];
        cursor_.frontier = state.p;
        ++cursor_.produced;
        auto observation = result(work, ResidentEvent::committed_token, true);
        observation.token = token;
        return observation;
    }
    if (cursor_.final_pending) return result(work, ResidentEvent::idle);
    if (!state.active && !state.preparing) {
        cursor_.frontier = state.p;
        cursor_.retired_produced = state.produced;
        cursor_.drained = true; // ResidentSlots publishes completion only after its consumers drain.
        if (state.stop && cursor_.reason == ResidentTerminal::completed) cursor_.reason = ResidentTerminal::cancelled;
        return terminal(work);
    }
    if (state.preparing && cursor_.frontier != state.p) {
        cursor_.frontier = state.p;
        return result(work, ResidentEvent::prompt_frontier, true);
    }
    return result(work, ResidentEvent::idle);
}

bool ResidentExecutor::prepare(ResidentWorkHandle work, std::string& error) {
    error.clear();
    return current(work, error) && !cursor_.drained && resident_.prepare(lease_, error);
}

bool ResidentExecutor::prepare(ResidentWorkHandle work, const core::SavedConversation* snapshot,
        const core::ConversationCheckpoint* cut, core::ConversationSnapshotExpectation expectation,
        std::string& error, ResidentSourceRefusal* refusal) {
    error.clear();
    return current(work, error) && !cursor_.drained &&
        resident_.prepare(lease_, snapshot, cut, expectation, error, refusal);
}

bool ResidentExecutor::prefilled(ResidentWorkHandle work, int64_t frontier, std::string& error) {
    error.clear();
    return current(work, error) && !cursor_.drained && !cursor_.final_pending &&
        resident_.prefilled(lease_, frontier, error);
}

bool ResidentExecutor::committed(ResidentWorkHandle work, int32_t next, const float* residual,
                                 std::string& error, bool diagnostic) {
    error.clear();
    return current(work, error) && !cursor_.drained && !cursor_.final_pending &&
        resident_.committed(lease_.slot, next, residual, error, diagnostic);
}

bool ResidentExecutor::propose(ResidentWorkHandle work, const float* residual, std::string& error) {
    error.clear();
    return current(work, error) && !cursor_.drained && resident_.propose(lease_.slot, residual, error);
}

bool ResidentExecutor::request_cancel(ResidentWorkHandle work, ResidentTerminal reason, std::string& error) {
    error.clear();
    if (!current(work, error)) return false;
    if (cursor_.drained) return true;
    if (reason == ResidentTerminal::completed || cursor_.borrow_generation) {
        error = "resident executor: cancellation needs a reason and an unborrowed owner";
        return false;
    }
    const auto& state = resident_.slots_[size_t(lease_.slot)];
    if (!state.active && !state.preparing) return true; // completion won before this intention
    if (cursor_.reason == ResidentTerminal::completed) cursor_.reason = reason;
    trace_work("cancel_intent", work, lease_.slot);
    resident_.cancel(lease_.slot);
    return true;
}

ResidentObservation ResidentExecutor::cancel(ResidentWorkHandle work, ResidentTerminal reason, std::string& error) {
    if (!request_cancel(work, reason, error)) return result(work, ResidentEvent::refused);
    return drain(work, error);
}

bool ResidentExecutor::finish(ResidentWorkHandle work, std::string& error, bool diagnostic) {
    error.clear();
    if (!current(work, error)) return false;
    if (cursor_.borrow_generation || cursor_.final_pending) {
        error = cursor_.final_pending ? "resident executor: final transaction consumers have not drained" :
                                      "resident executor: return snapshot before draining";
        return false;
    }
    if (cursor_.drained) return true;
    const auto& state = resident_.slots_[size_t(lease_.slot)];
    const int64_t frontier = state.p, produced = state.produced;
    const bool preparing = state.preparing;
    if (preparing) {
        // A committed final row still requires catch-up, even outside this transaction seam.
        if (!resident_.abort_preparation(lease_, error)) return false;
        cursor_.retired = true;
    } else if (state.active && !resident_.finish(lease_.slot, error, diagnostic)) {
        return false;
    }
    cursor_.frontier = frontier;
    cursor_.retired_produced = produced;
    cursor_.drained = true;
    if ((preparing || state.stop) && cursor_.reason == ResidentTerminal::completed)
        cursor_.reason = ResidentTerminal::cancelled;
    return true;
}

ResidentObservation ResidentExecutor::drain(ResidentWorkHandle work, std::string& error) {
    if (!finish(work, error)) return result(work, ResidentEvent::refused);
    return terminal(work);
}

ResidentSnapshotLease ResidentExecutor::borrow_snapshot(ResidentWorkHandle work, std::string& error) {
    error.clear();
    ResidentSnapshotLease snapshot(work, this);
    if (!current(work, error)) return snapshot;
    if (!cursor_.terminal_seen) return snapshot; // do not hand over history before committed output/terminal observation
    if (cursor_.borrow_generation || cursor_.next_borrow == UINT64_MAX) {
        error = "resident executor: snapshot already borrowed or generation exhausted";
        return snapshot;
    }
    const auto& state = resident_.slots_[size_t(lease_.slot)];
    if (state.active || state.preparing || state.stop || state.tainted || state.failed) return snapshot;
    auto source = resident_.completed_snapshot_source(lease_.slot, error);
    if (!source) return snapshot;
    const auto& certificate = source->certificate;
    if (!certificate.certified || certificate.request != work.request || certificate.incarnation != work.incarnation) {
        error = "resident executor: stale snapshot certificate";
        return snapshot;
    }
    snapshot.source_.emplace(*source);
    snapshot.generation_ = cursor_.borrow_generation = ++cursor_.next_borrow;
    return snapshot;
}

ResidentObservation ResidentExecutor::return_snapshot(ResidentSnapshotLease& snapshot, std::string& error) {
    error.clear();
    const auto work = snapshot.work_;
    if (!current(work, error)) return result(work, ResidentEvent::refused);
    if (snapshot.issuer_ != this || !snapshot.source_ || !snapshot.generation_ ||
        snapshot.generation_ != cursor_.borrow_generation) {
        error = "resident executor: stale snapshot borrow generation";
        return result(work, ResidentEvent::refused);
    }
    const auto& state = resident_.slots_[size_t(lease_.slot)];
    const auto& certificate = snapshot.source_->certificate;
    if (state.active || state.preparing || state.stop || state.tainted || !certificate.certified ||
        certificate.request != work.request || certificate.incarnation != work.incarnation ||
        certificate.committed_tokens != int64_t(state.ids.size()) || certificate.valid_end != state.p) {
        error = "resident executor: borrowed committed cut changed";
        return result(work, ResidentEvent::refused);
    }
    snapshot.source_.reset();
    snapshot.generation_ = cursor_.borrow_generation = 0;
    return result(work, ResidentEvent::idle);
}

bool ResidentExecutor::retire(ResidentWorkHandle work, std::string& error) {
    error.clear();
    if (!current(work, error)) return false;
    if (!cursor_.drained || !cursor_.terminal_seen || cursor_.borrow_generation || cursor_.final_pending) {
        error = "resident executor: owner still has consumers, observations or a snapshot borrow";
        return false;
    }
    trace_work("retire", work, lease_.slot);
    lease_ = {};
    return true;
}

bool ResidentExecutor::final_ready(ResidentWorkHandle work, const core::BatchRow& row, int index,
                                   std::string& error) const {
    if (!current(work, error) || !resident_.preparation_valid(lease_, error)) return false;
    const auto& state = resident_.slots_[size_t(lease_.slot)];
    if (cursor_.drained || cursor_.borrow_generation || cursor_.final_pending ||
        !resident_.preparations_[size_t(lease_.slot)].ready || state.stop ||
        state.prompt_cursor.suspended || state.p != lease_.endpoint ||
        state.ids.size() != size_t(lease_.endpoint) || state.prompt.empty() ||
        row.request_id != work.request || row.cancellation_generation != work.request ||
        row.state != lease_.stages[0] || row.slot != lease_.slot ||
        row.position != lease_.endpoint || row.token != state.prompt.back() ||
        row.output_offset != index || row.sequence_begin != 0 ||
        row.sequence_end != row.position + 1 || row.verification_span != 1 ||
        row.mask != core::BatchRowMask::active) {
        error = "resident executor: final row does not match the selected owner/input";
        return false;
    }
    return true;
}

bool ResidentFinalTransaction::issue(const ResidentFinalSelection* owners, const core::BatchRow* rows,
                                     int count, std::string& error) {
    error.clear();
    if (count_ || released_ || !owners || !rows || count < 1 || count > int(owners_.size())) {
        error = "resident executor: invalid or repeated final transaction";
        return false;
    }
    for (int row = 0; row < count; ++row) {
        const auto& selected = owners[row];
        if (!selected.executor || !selected.executor->final_ready(selected.work, rows[row], row, error)) return false;
        for (int previous = 0; previous < row; ++previous)
            if (owners[previous].executor == selected.executor ||
                &owners[previous].executor->resident_ != &selected.executor->resident_ ||
                owners[previous].executor->lease_.slot == selected.executor->lease_.slot) {
                error = "resident executor: final transaction needs distinct owners of one resident";
                return false;
            }
    }
    count_ = count;
    for (int row = 0; row < count; ++row) {
        auto& executor = *owners[row].executor;
        owners_[size_t(row)] = &executor;
        receipts_[size_t(row)] = {owners[row].work, executor.lease_, row, rows[row].position, rows[row].token};
        executor.cursor_.final_pending = true;
        trace_work("final_issue", owners[row].work, executor.lease_.slot);
    }
    return true;
}

bool ResidentFinalTransaction::consume(const ResidentFinalReceipt& receipt, int32_t next, const float* residual,
                                       std::string& error) {
    error.clear();
    if (released_ || receipt.row < 0 || receipt.row >= count_ || (consumed_ & (1u << receipt.row))) {
        error = "resident executor: stale or already consumed final receipt";
        return false;
    }
    const auto& expected = receipts_[size_t(receipt.row)];
    auto& executor = *owners_[size_t(receipt.row)];
    const auto& a = receipt.lease;
    const auto& b = expected.lease;
    if (receipt.work != expected.work || a.slot != b.slot || a.request != b.request ||
        a.incarnation != b.incarnation || a.position != b.position || a.endpoint != b.endpoint ||
        a.stages != b.stages || a.allocations != b.allocations || receipt.position != expected.position ||
        receipt.token != expected.token || !executor.cursor_.final_pending) {
        error = "resident executor: final receipt owner/allocation/input fence mismatch";
        return false;
    }
    if (!executor.current(receipt.work, error)) return false;
    const auto& state = executor.resident_.slots()[size_t(b.slot)];
    if (state.p != receipt.position || state.prompt.empty() || state.prompt.back() != receipt.token) {
        error = "resident executor: committed input changed before final receipt consumption";
        return false;
    }
    if (!executor.resident_.first_token(executor.lease_, next, residual, error)) return false;
    consumed_ |= 1u << receipt.row;
    if (executor_trace())
        std::fprintf(stderr, "RESIDENT_EXECUTOR event=final_consume row=%d slot=%d request=%llu incarnation=%llu "
                             "work=%llu position=%lld token=%d allocation=%llu stopped=%d\n",
                     receipt.row, b.slot, (unsigned long long)receipt.work.request,
                     (unsigned long long)receipt.work.incarnation, (unsigned long long)receipt.work.generation,
                     (long long)receipt.position, receipt.token, (unsigned long long)b.allocations[0], int(state.stop));
    return true;
}

bool ResidentFinalTransaction::release(std::string& error) {
    error.clear();
    if (released_ || !count_ || consumed_ != (1u << count_) - 1) {
        error = "resident executor: final transaction still has mandatory consumers";
        return false;
    }
    for (int row = 0; row < count_; ++row)
        if (!owners_[size_t(row)]->current(receipts_[size_t(row)].work, error)) return false;
    for (int row = 0; row < count_; ++row) {
        trace_work("final_release", receipts_[size_t(row)].work, owners_[size_t(row)]->lease_.slot);
        owners_[size_t(row)]->cursor_.final_pending = false;
        owners_[size_t(row)] = nullptr;
        receipts_[size_t(row)] = {};
    }
    released_ = true;
    return true;
}

} // namespace strata::program
