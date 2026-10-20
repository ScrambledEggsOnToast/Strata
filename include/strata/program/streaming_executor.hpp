// Serial serving call scope. All expert, CPU, scratch and KV storage stays with its existing owner.
#pragma once
#include "strata/program/execution_plan.hpp"
#include "strata/plan/admission.hpp"
#include <atomic>
#include <cstdint>
#include <string>

namespace strata::program {

enum class StreamingSource { unresolved, arena, mapped, windowed };
struct StreamingSelection {
    bool serve = false;
    int effective_batch = -1, destination = -2, stages = 0;
    StreamingSource source = StreamingSource::unresolved;
    bool source_ready = false, source_retired = false;
    int source_lanes = 0;
    bool cpu_share_resolved = false; // the original admission choice, never re-armed by this adapter
};

// Pure selection: mode alone never selects working-owner ownership. Prices and the admission
// decision must come from the same resolved configuration; a report-only/unknown price refuses.
PlanDecision streaming_capability(const ExecutionPlan& resolved, const plan::AdmissionDecision& admission,
                                  const StreamingSelection& selection);

struct StreamingWorkHandle {
    uint64_t request = 0, generation = 0; // no ResidentSlot, physical-seat incarnation or snapshot lease
    explicit operator bool() const { return request && generation; }
    bool operator==(const StreamingWorkHandle&) const = default;
};
enum class StreamingEvent { refused, idle, prompt_frontier, committed_token, terminal };
enum class StreamingTerminal { completed, cancelled, timeout, disconnected, failed };
struct StreamingObservation {
    StreamingWorkHandle work;
    StreamingEvent event = StreamingEvent::refused;
    bool fresh = false;
    int64_t frontier = 0, committed = 0;
    int32_t token = -1;
    StreamingTerminal reason = StreamingTerminal::completed;
    explicit operator bool() const { return event != StreamingEvent::refused; }
};
struct StreamingWorkingBinding {
    const uint64_t* request = nullptr; // &resident.working().request, stable until explicit retire
    std::atomic<bool>* stop = nullptr; // the existing serial stop_req; no resident cancel(-1)
};

// The one host caller supplies this only after the existing resource owners drain. In particular,
// source_ready/a source mutex cannot certify escaped CPU/stager/GPU reader completion. Default is
// not-drained, and every receipt names the exact call scope and existing publication counts.
struct StreamingDrainReceipt {
    StreamingWorkHandle work;
    int64_t frontier = 0, committed = 0;
    bool verifier = false, prefill_chain = false, draft = false, experts = false, transfers = false;
};

class StreamingExecutor {
public:
    StreamingExecutor() = default;
    StreamingExecutor(const StreamingExecutor&) = delete;
    StreamingExecutor& operator=(const StreamingExecutor&) = delete;
    StreamingExecutor(StreamingExecutor&&) = delete;
    StreamingExecutor& operator=(StreamingExecutor&&) = delete;
    StreamingWorkHandle submit(StreamingWorkingBinding owner, const ExecutionPlan& resolved,
        const plan::AdmissionDecision& admission, const StreamingSelection& selection, std::string& error);
    bool prefilled(StreamingWorkHandle work, int64_t frontier, std::string& error);
    // One scalar pending token, not another history vector. The serial emitter observes each
    // publication before publishing its successor. An issued commit remains legal after STOP.
    bool committed(StreamingWorkHandle work, int32_t token, int64_t frontier, std::string& error);
    StreamingObservation observe(StreamingWorkHandle work, std::string& error);
    bool request_cancel(StreamingWorkHandle work, StreamingTerminal reason, std::string& error);
    // Never waits, pumps controls, consumes output or releases external resources. An incomplete
    // receipt refuses. Drained resources with pending output return idle until observe consumes it.
    StreamingObservation drain(StreamingWorkHandle work, const StreamingDrainReceipt& receipt,
        std::string& error, StreamingTerminal reason = StreamingTerminal::completed);
    bool retire(StreamingWorkHandle work, std::string& error);
    StreamingWorkHandle work() const { return work_; }
private:
    StreamingWorkingBinding owner_;
    StreamingWorkHandle work_;
    uint64_t next_generation_ = 0;
    int64_t frontier_ = 0, seen_frontier_ = -1, produced_ = 0, observed_ = 0;
    int32_t pending_ = -1;
    StreamingTerminal reason_ = StreamingTerminal::completed;
    bool drained_ = false, terminal_seen_ = false;
    bool current(StreamingWorkHandle work, std::string& error) const;
    bool writable(StreamingWorkHandle work, std::string& error) const;
    StreamingObservation result(StreamingEvent event, bool fresh = false) const;
    StreamingObservation terminal();
};
} // namespace strata::program
