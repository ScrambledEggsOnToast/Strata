#include "strata/program/resident_executor.hpp"

namespace strata::program {

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
    if (cursor_.borrow_generation || (cursor_.generation && (!cursor_.drained || !cursor_.terminal_seen)) ||
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

ResidentObservation ResidentExecutor::cancel(ResidentWorkHandle work, ResidentTerminal reason, std::string& error) {
    error.clear();
    if (!current(work, error)) return result(work, ResidentEvent::refused);
    if (cursor_.drained) return terminal(work);
    if (reason == ResidentTerminal::completed || cursor_.borrow_generation) {
        error = "resident executor: cancellation needs a reason and an unborrowed owner";
        return result(work, ResidentEvent::refused);
    }
    const auto& state = resident_.slots_[size_t(lease_.slot)];
    if (!state.active && !state.preparing) return drain(work, error); // completion won before this cancel
    if (cursor_.reason == ResidentTerminal::completed) cursor_.reason = reason;
    resident_.cancel(lease_.slot);
    return drain(work, error);
}

ResidentObservation ResidentExecutor::drain(ResidentWorkHandle work, std::string& error) {
    error.clear();
    if (!current(work, error)) return result(work, ResidentEvent::refused);
    if (cursor_.borrow_generation) {
        error = "resident executor: return snapshot before draining";
        return result(work, ResidentEvent::refused);
    }
    if (cursor_.drained) return terminal(work);
    const auto& state = resident_.slots_[size_t(lease_.slot)];
    const int64_t frontier = state.p, produced = state.produced;
    const bool preparing = state.preparing;
    if (preparing) {
        // abort_preparation refuses an issued final row: mandatory first-token draft catch-up
        // cannot be replaced by cancellation, nor can this facade pump controls to finish it.
        if (!resident_.abort_preparation(lease_, error)) return result(work, ResidentEvent::refused);
        cursor_.retired = true;
    } else if (state.active && !resident_.finish(lease_.slot, error)) {
        return result(work, ResidentEvent::refused);
    }
    cursor_.frontier = frontier;
    cursor_.retired_produced = produced;
    cursor_.drained = true;
    if ((preparing || state.stop) && cursor_.reason == ResidentTerminal::completed)
        cursor_.reason = ResidentTerminal::cancelled;
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

} // namespace strata::program
