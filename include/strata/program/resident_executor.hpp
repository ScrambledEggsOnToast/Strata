// Call-scope resident lifecycle facade. ResidentSlots remains the only request/state owner.
#pragma once
#include "strata/program/execution_plan.hpp"
#include "strata/program/resident_slots.hpp"

namespace strata::program {
class ResidentExecutor;


struct ResidentWorkHandle {
    uint64_t request = 0, incarnation = 0, generation = 0;
    explicit operator bool() const { return request && incarnation && generation; }
    bool operator==(const ResidentWorkHandle&) const = default;
};

enum class ResidentEvent { refused, idle, prompt_frontier, committed_token, terminal };
enum class ResidentTerminal { completed, cancelled, timeout, disconnected, refused };

struct ResidentObservation {
    ResidentWorkHandle work;
    ResidentEvent event = ResidentEvent::refused;
    // A repeated terminal is a successful, drained observation, not another event to emit.
    bool fresh = false;
    int64_t frontier = 0, committed = 0;
    int32_t token = -1;
    ResidentTerminal reason = ResidentTerminal::completed;
    explicit operator bool() const { return event != ResidentEvent::refused; }
};

// A synchronous borrow of the existing certified source, not a new snapshot or checkpoint chain.
// Return it before any resident mutation. Moving transfers the borrow; copying is forbidden.
class ResidentSnapshotLease {
public:
    ResidentSnapshotLease(ResidentSnapshotLease&& other) noexcept;
    ResidentSnapshotLease(const ResidentSnapshotLease&) = delete;
    ResidentSnapshotLease& operator=(const ResidentSnapshotLease&) = delete;
    ResidentSnapshotLease& operator=(ResidentSnapshotLease&&) = delete;
    const ResidentWorkHandle& work() const { return work_; }
    uint64_t generation() const { return generation_; }
    ResidentSnapshotSource* source() { return source_ ? &*source_ : nullptr; }
    const ResidentSnapshotSource* source() const { return source_ ? &*source_ : nullptr; }
    explicit operator bool() const { return source_.has_value(); }
private:
    friend class ResidentExecutor;
    ResidentSnapshotLease(ResidentWorkHandle work, const ResidentExecutor* issuer) : work_(work), issuer_(issuer) {}
    ResidentWorkHandle work_;
    const ResidentExecutor* issuer_;
    uint64_t generation_ = 0;
    std::optional<ResidentSnapshotSource> source_;
};

// One host caller, one admitted lease. No queue, thread, slot registry, token/history copy, or
// scheduling/decoding predicate. The resource owners must outlive both this facade and its borrow.
//
// This contract-only seam attaches to a reservation; it does not yet issue model work. Existing
// Verifier/Prefill/MtpDrafter transitions publish into ResidentSlots, and observe consumes only
// those committed publications. In particular it never drains a graph merely to poll progress.
// The shared packed final-row transaction is deliberately not migrated into this per-request scope.
class ResidentExecutor {
public:
    explicit ResidentExecutor(ResidentSlots& resident) : resident_(resident) {}
    ResidentExecutor(const ResidentExecutor&) = delete;
    ResidentExecutor& operator=(const ResidentExecutor&) = delete;
    ResidentExecutor(ResidentExecutor&&) = delete;
    ResidentExecutor& operator=(ResidentExecutor&&) = delete;

    // Bind only an existing reserved preparation and an accepted resident plan. Refusal changes
    // neither the reservation nor the previous cursor. No model allocation or GPU work occurs.
    ResidentWorkHandle submit(const ResidentPreparation& lease, const ExecutionPlan& plan, std::string& error);
    ResidentObservation observe(ResidentWorkHandle work, std::string& error);
    // Caller chooses the terminal reason. These methods never emit a token/protocol frame or
    // pump controls. If consumers have not drained, refuse and leave their ownership intact.
    // Successful idle means resources retired but committed-token observations remain. Keep the
    // owner/history bound and call observe until its fresh terminal; drain never consumes output.
    ResidentObservation cancel(ResidentWorkHandle work, ResidentTerminal reason, std::string& error);
    ResidentObservation drain(ResidentWorkHandle work, std::string& error);
    // Ineligible completed owners are clean misses (empty lease, empty error). Preparing,
    // stopped, tainted, failed or not-yet-observed terminal owners are never borrowed.
    // Return is generation-fenced.
    ResidentSnapshotLease borrow_snapshot(ResidentWorkHandle work, std::string& error);
    ResidentObservation return_snapshot(ResidentSnapshotLease& snapshot, std::string& error);
private:
    struct Cursor {
        uint64_t generation = 0, borrow_generation = 0, next_borrow = 0;
        int64_t frontier = -1, produced = 0, retired_produced = 0; // observed output is not resource retirement
        ResidentTerminal reason = ResidentTerminal::completed;
        bool drained = false, terminal_seen = false, retired = false;
    } cursor_;
    ResidentSlots& resident_;
    ResidentPreparation lease_;

    bool current(ResidentWorkHandle work, std::string& error) const;
    ResidentObservation result(ResidentWorkHandle work, ResidentEvent event, bool fresh = false) const;
    ResidentObservation terminal(ResidentWorkHandle work);
};

} // namespace strata::program
