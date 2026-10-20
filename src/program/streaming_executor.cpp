#include "strata/program/streaming_executor.hpp"
#include <cstdio>
#include <cstdlib>

namespace strata::program {
namespace {
void trace_work(const char* event, StreamingWorkHandle work, int64_t frontier, int64_t committed,
                StreamingTerminal reason) {
    static const bool enabled = [] {
        const char* value = std::getenv("STRATA_EXECUTOR_TRACE");
        return value && value[0] == '1' && value[1] == '\0';
    }();
    if (enabled)
        std::fprintf(stderr, "STREAMING_EXECUTOR event=%s request=%llu work=%llu frontier=%lld committed=%lld "
                             "reason=%d slot=-1\n", event, (unsigned long long) work.request,
                     (unsigned long long) work.generation, (long long) frontier, (long long) committed, int(reason));
}
}
PlanDecision streaming_capability(const ExecutionPlan& resolved, const plan::AdmissionDecision& admission,
                                  const StreamingSelection& selection) {
    const auto refuse = [](const char* reason) { return PlanDecision{Support::refused, reason}; };
    if (!selection.serve || selection.effective_batch != 0 || selection.destination != -1)
        return refuse("serial serving requires explicit serve, effective batch zero and working destination -1");
    if (resolved.refused || resolved.mode != ServeMode::streaming || resolved.batch_slots != 0)
        return refuse("refused or incompatible serial plan (requested resident modes keep their existing path)");
    if (selection.stages < 1 || selection.stages != resolved.stages)
        return refuse("working stage count does not match the resolved plan");
    if (resolved.admission.support != Support::off || resolved.suspension_mode != Suspension::none ||
        resolved.suspension.support != Support::off || resolved.concurrent_reader.support != Support::off ||
        resolved.budget.support != Support::off || (resolved.proposal != ProposalShape::none &&
        resolved.proposal != ProposalShape::serial_single_draft))
        return refuse("serial working owner excludes native seats, suspension, concurrent readers and batched work");
    if (selection.source == StreamingSource::unresolved || !selection.source_ready || selection.source_retired)
        return refuse("streaming requires an explicitly resolved ready, nonretired expert source");
    if (selection.source_lanes < 1 || (selection.source == StreamingSource::windowed &&
        selection.source_lanes != selection.stages))
        return refuse("live windowed prefill requires one independent source lane per stage");
    if (!admission.admitted || !admission.verified || !admission.unknown_classes.empty())
        return refuse("resource admission is refused or unresolved; unknown demand is not zero");
    if (!selection.cpu_share_resolved || resolved.memory_classes.empty())
        return refuse("original CPU-share choice and resource prices must be resolved before submit");
    bool share_priced = false;
    for (const auto& price : resolved.memory_classes) {
        if (price.unknown || price.scope.empty() || price.name.empty() || price.source.empty())
            return refuse("resource demand has an unknown or unsourced price; refusing rather than estimating zero");
        if (price.scope == "guest" && price.name == "prefill_cpu_share_buffers") share_priced = true;
    }
    if (!share_priced)
        return refuse("missing prefill_cpu_share_buffers price; an omitted demand is not a known zero");
    return {Support::on, {}};
}

StreamingWorkHandle StreamingExecutor::submit(StreamingWorkingBinding owner, const ExecutionPlan& resolved,
        const plan::AdmissionDecision& admission, const StreamingSelection& selection, std::string& error) {
    error.clear();
    const auto capability = streaming_capability(resolved, admission, selection);
    if (capability.support != Support::on) { error = capability.reason; return {}; }
    if (!owner.request || !*owner.request || !owner.stop) {
        error = "streaming executor: missing working owner identity or stop control"; return {};
    }
    if (work_) { error = "streaming executor: retire the previous call scope before rebinding"; return {}; }
    if (next_generation_ == UINT64_MAX) { error = "streaming executor: work generation exhausted"; return {}; }
    owner_ = owner;
    work_ = {*owner.request, ++next_generation_};
    frontier_ = produced_ = observed_ = 0;
    seen_frontier_ = -1;
    pending_ = -1;
    reason_ = StreamingTerminal::completed;
    drained_ = terminal_seen_ = false;
    trace_work("submit", work_, frontier_, produced_, reason_);
    return work_; // Do not clear a STOP that already belongs to this admitted request.
}

bool StreamingExecutor::current(StreamingWorkHandle work, std::string& error) const {
    if (!work || work != work_ || !owner_.request || !owner_.stop || *owner_.request != work.request) {
        error = "streaming executor: stale working owner or call generation"; return false;
    }
    return true;
}
bool StreamingExecutor::writable(StreamingWorkHandle work, std::string& error) const {
    if (!current(work, error)) return false;
    if (drained_) { error = "streaming executor: call scope already drained"; return false; }
    return true;
}
StreamingObservation StreamingExecutor::result(StreamingEvent event, bool fresh) const {
    return {work_, event, fresh, frontier_, produced_, -1, reason_};
}
StreamingObservation StreamingExecutor::terminal() {
    if (observed_ != produced_) return result(StreamingEvent::idle);
    const bool fresh = !terminal_seen_;
    terminal_seen_ = true;
    if (fresh) trace_work("terminal", work_, frontier_, produced_, reason_);
    return result(StreamingEvent::terminal, fresh);
}

bool StreamingExecutor::prefilled(StreamingWorkHandle work, int64_t frontier, std::string& error) {
    error.clear();
    if (!writable(work, error)) return false;
    if (produced_ || frontier < frontier_) {
        error = "streaming executor: prompt frontier moved backwards or followed output"; return false;
    }
    frontier_ = frontier;
    return true;
}
bool StreamingExecutor::committed(StreamingWorkHandle work, int32_t token, int64_t frontier, std::string& error) {
    error.clear();
    if (!writable(work, error)) return false;
    if (pending_ >= 0 || token < 0 || frontier <= frontier_ || produced_ == INT64_MAX) {
        error = "streaming executor: unobserved output, invalid committed publication or exhausted counter"; return false;
    }
    frontier_ = frontier;
    pending_ = token;
    ++produced_;
    trace_work("committed", work_, frontier_, produced_, reason_);
    return true;
}
StreamingObservation StreamingExecutor::observe(StreamingWorkHandle work, std::string& error) {
    error.clear();
    if (!current(work, error)) return {work};
    if (pending_ >= 0) {
        auto observation = result(StreamingEvent::committed_token, true);
        observation.token = pending_;
        pending_ = -1;
        observation.committed = ++observed_;
        seen_frontier_ = frontier_;
        return observation;
    }
    if (drained_) return terminal();
    if (seen_frontier_ != frontier_) {
        seen_frontier_ = frontier_;
        return result(StreamingEvent::prompt_frontier, true);
    }
    return result(StreamingEvent::idle);
}
bool StreamingExecutor::request_cancel(StreamingWorkHandle work, StreamingTerminal reason, std::string& error) {
    error.clear();
    if (!current(work, error)) return false;
    if (reason != StreamingTerminal::cancelled && reason != StreamingTerminal::timeout &&
        reason != StreamingTerminal::disconnected) {
        error = "streaming executor: cancellation requires a cancellation reason"; return false;
    }
    if (drained_) return true; // Completed resources/terminal won; never set a later owner's STOP.
    if (reason_ == StreamingTerminal::completed) reason_ = reason;
    owner_.stop->store(true, std::memory_order_release);
    trace_work("cancel_intent", work_, frontier_, produced_, reason_);
    return true;
}
StreamingObservation StreamingExecutor::drain(StreamingWorkHandle work, const StreamingDrainReceipt& receipt,
        std::string& error, StreamingTerminal reason) {
    error.clear();
    if (!current(work, error)) return {work};
    if (receipt.work != work) { error = "streaming executor: stale drain receipt"; return {work}; }
    if (drained_) return terminal();
    if (!receipt.verifier || !receipt.prefill_chain || !receipt.draft || !receipt.experts || !receipt.transfers ||
        receipt.committed != produced_ || receipt.frontier < frontier_) {
        error = "streaming executor: consumers or committed publications have not drained"; return {work};
    }
    if (reason == StreamingTerminal::failed || reason_ == StreamingTerminal::completed) reason_ = reason;
    if (reason_ == StreamingTerminal::completed && owner_.stop->load(std::memory_order_acquire))
        reason_ = StreamingTerminal::cancelled;
    frontier_ = receipt.frontier;
    drained_ = true;
    trace_work("drain", work_, frontier_, produced_, reason_);
    return terminal(); // Output remains in its scalar register until observe; no implicit delivery.
}
bool StreamingExecutor::retire(StreamingWorkHandle work, std::string& error) {
    error.clear();
    if (!current(work, error)) return false;
    if (!drained_ || !terminal_seen_ || pending_ >= 0 || observed_ != produced_) {
        error = "streaming executor: outstanding consumers, output or terminal observation"; return false;
    }
    trace_work("retire", work_, frontier_, produced_, reason_);
    owner_ = {};
    work_ = {};
    return true;
}
} // namespace strata::program
