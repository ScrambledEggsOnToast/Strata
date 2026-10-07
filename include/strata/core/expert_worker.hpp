#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>
#include <stdexcept>

namespace strata::core {

// The generation distinguishes retries/rejected speculative spans at the same position.
struct ExpertOperation {
    uint64_t request = 0;
    int64_t layer = 0;
    int64_t first_position = 0;
    int64_t token_count = 0;
    uint64_t generation = 0;

    bool operator==(const ExpertOperation& other) const {
        return request == other.request && layer == other.layer &&
               first_position == other.first_position && token_count == other.token_count &&
               generation == other.generation;
    }
};

// Immutable for the lifetime of an operation. A residency version describes a
// published weight set, not a mutable device address or a quantization conversion.
struct ExpertWeights {
    // Opaque immutable process weight-set generation, not a cryptographic revision.
    uint64_t model_generation = 0;
    uint64_t residency_version = 0;
    const void* handle = nullptr;
    int64_t input_width = 0;
    int64_t hidden_width = 0;
    int32_t gate_up_format = 0;
    int32_t down_format = 0;
};

struct ExpertWork {
    ExpertOperation operation;
    ExpertWeights weights;
    const float* input = nullptr;
    const int32_t* experts = nullptr;
    int64_t routed_width = 0;
    const int64_t* positions = nullptr;
    const int* request_slots = nullptr;
    const uint64_t* request_ids = nullptr; // per-token request generation, distinct from reusable slot index
    // Nonnegative entries are already owned. Workers may only select negative
    // entries; the dispatcher publishes ownership before consulting the next worker.
    const int32_t* assigned = nullptr;
    const int32_t* primary_residency = nullptr;
};

// begin must synchronously snapshot caller-owned input and routing/position/request
// metadata into adapter-owned staging before returning. Caller buffers may be
// overwritten by primary-graph teardown even when helper cancellation fails.
// Immutable weights and adapter-owned scratch remain leased until finish/cancel drains.
// finish writes unweighted rows at their router indices. The layer's existing
// combine is the sole reduction owner. A false return poisons the whole layer.
// This interface owns routed rows only. Shared experts are layer-owned, execute
// on the ordered primary stream, and contribute once through the separate shared
// argument of moe_combine/native_moe_combine. They are never router indices,
// helper assignments, or worker output rows, even when their numeric IDs coincide.
// Failed begin/finish may retain staging: cancel must be attempted even for a
// partially submitted begin. Only successful finish/cancel certifies a drain.
// Adapters must not throw; their destruction must drain retained staging.
// No CUDA type crosses this seam; asynchronous completion belongs to the adapter.
class ExpertWorker {
public:
    virtual ~ExpertWorker() = default;
    virtual uint64_t residency_version() const { return 1; }
    virtual bool begin(const ExpertWork& work, std::string& error) = 0;
    virtual bool owns(int64_t row) const = 0;
    virtual bool finish(float* output, std::string& error) = 0;
    virtual bool cancel(std::string& error) = 0;
};

// Existing captured graph executes the arithmetic. Submission publishes its plan;
// only an exact matching final-consumer boundary acknowledges completion.
class GraphExpertWorker final : public ExpertWorker {
public:
    void set_submission(void* context, void (*submit)(void*)) { context_ = context; submit_ = submit; }
    bool begin(const ExpertWork& work, std::string& error) override {
        if (active_ || work.operation.token_count < 0 || work.routed_width < 0 ||
            work.routed_width > 128 || work.operation.token_count > 128 / std::max<int64_t>(work.routed_width, 1)) {
            error = "primary graph operation active or exceeds routing capacity"; return false;
        }
        if (!work.weights.handle || !work.weights.model_generation || work.weights.input_width <= 0 ||
            work.weights.hidden_width <= 0) {
            error = "primary graph immutable weight identity missing"; return false;
        }
        count_ = work.operation.token_count * work.routed_width;
        if (count_ && !work.assigned) { error = "primary graph routing missing"; return false; }
        operation_ = work.operation;
        weights_ = work.weights;
        for (int64_t i = 0; i < count_; ++i) owned_[i] = work.assigned[i] == 0 || work.assigned[i] == 1;
        active_ = true;
        consumed_ = false;
        if (submit_) submit_(context_);
        return true;
    }
    bool owns(int64_t row) const override { return active_ && row >= 0 && row < count_ && owned_[row]; }
    bool consumed(const ExpertOperation& operation) {
        if (!active_ || consumed_ || !(operation == operation_)) return false;
        consumed_ = true;
        return true;
    }
    bool finish(float*, std::string& error) override {
        if (!active_ || !consumed_) { error = "primary graph consumer has not completed"; return false; }
        active_ = false;
        return true;
    }
    bool cancel(std::string& error) override {
        if (!active_) return true;
        return finish(nullptr, error);
    }
private:
    ExpertOperation operation_;
    ExpertWeights weights_;
    bool owned_[128] = {};
    int64_t count_ = 0;
    bool active_ = false;
    bool consumed_ = false;
    void* context_ = nullptr;
    void (*submit_)(void*) = nullptr;
};

// One operation owns these preallocated rows until the final consumer releases them.
// Only the dispatch thread mutates this ledger. Workers signal their own completion
// primitives; the dispatch thread records completion after waiting on those primitives.
// Completion is not publication: cancellation/failure permanently poisons this operation.
class ExpertCompletion {
public:
    static constexpr size_t kCapacity = 128;
    explicit ExpertCompletion(size_t capacity) : rows_(capacity) {}

    bool begin(const ExpertOperation& operation, size_t count) {
        if (active_ || count > rows_.size() || operation.layer < 0 ||
            operation.first_position < 0 || operation.token_count < 0) return false;
        operation_ = operation;
        count_ = count;
        std::fill(rows_.begin(), rows_.begin() + count_, Row{});
        active_ = true;
        cancelled_ = false;
        failed_ = false;
        submissions_ = 0;
        return true;
    }

    bool claim(size_t row, uint32_t owner, uint64_t residency_version) {
        if (!active_ || cancelled_ || failed_ || row >= count_ || owner == 0 ||
            rows_[row].owner != 0) return false;
        rows_[row] = {owner, residency_version, false};
        return true;
    }

    bool complete(const ExpertOperation& operation, uint32_t owner, uint64_t residency_version) {
        if (!active_ || !(operation == operation_) || owner == 0) return false;
        bool found = false;
        for (size_t i = 0; i < count_; ++i) {
            const Row& row = rows_[i];
            if (row.owner != owner) continue;
            if (row.version != residency_version || row.done) return false;
            found = true;
        }
        if (!found) return false;
        for (size_t i = 0; i < count_; ++i)
            if (rows_[i].owner == owner) rows_[i].done = true;
        return true;
    }

    void cancel() { if (active_) cancelled_ = true; }
    void fail() { if (active_) failed_ = true; }
    bool drained(const ExpertOperation& operation, uint32_t owner) {
        if (!active_ || !(operation == operation_) || (!cancelled_ && !failed_)) return false;
        for (size_t i = 0; i < count_; ++i)
            if (rows_[i].owner == owner) rows_[i].done = true;
        return true;
    }
    bool active() const { return active_; }
    bool matches(const ExpertOperation& operation, size_t count) const {
        return active_ && !cancelled_ && !failed_ && operation == operation_ && count == count_;
    }
    // A submitted worker can retain staging even without claiming any rows.
    bool retain_submission(const ExpertOperation& operation) {
        if (!active_ || cancelled_ || failed_ || !(operation == operation_) ||
            submissions_ == std::numeric_limits<size_t>::max()) return false;
        ++submissions_;
        return true;
    }
    bool drain_submission(const ExpertOperation& operation) {
        if (!active_ || !(operation == operation_) || submissions_ == 0) return false;
        --submissions_;
        return true;
    }
    uint32_t owner(size_t row) const { return active_ && row < count_ ? rows_[row].owner : 0; }
    bool has_owner(uint32_t owner) const {
        for (size_t i = 0; i < count_; ++i) if (rows_[i].owner == owner) return true;
        return false;
    }

    bool publishable() const {
        if (!active_ || cancelled_ || failed_ || submissions_ != 0) return false;
        for (size_t i = 0; i < count_; ++i)
            if (rows_[i].owner == 0 || !rows_[i].done) return false;
        return true;
    }

    // Caller certifies the final consumer has finished. Cancelled/failed operations
    // release only after submitted staging and claimed rows have both drained.
    // Unclaimed rows need no completion; retain_submission covers partial launches.
    bool release(const ExpertOperation& operation) {
        if (!active_ || !(operation == operation_) || submissions_ != 0) return false;
        for (size_t i = 0; i < count_; ++i)
            if (rows_[i].owner != 0 && !rows_[i].done) return false;
        if (!cancelled_ && !failed_ && !publishable()) return false;
        active_ = false;
        return true;
    }

    size_t storage_bytes() const { return rows_.capacity() * sizeof(Row); }
    static constexpr size_t planned_bytes(size_t capacity) { return capacity * sizeof(Row); }

private:
    struct Row {
        uint32_t owner = 0;
        uint64_t version = 0;
        bool done = false;
    };
    static_assert(sizeof(Row) == 24, "update the planner ownership-row ABI together");
    std::vector<Row> rows_;
    ExpertOperation operation_;
    size_t count_ = 0;
    size_t submissions_ = 0;
    bool active_ = false;
    bool cancelled_ = false;
    bool failed_ = false;
};

// ================================ HET-038: CROSS-REQUEST EXPERT GROUPING ================================
//
// A verify window's staged rows are independent requests' rows: one row per slot, or one request's short
// verification span as consecutive rows. The expert dispatch groups those rows by layer, expert, weight
// format and worker while every result still belongs to exactly one (request, layer, speculative position).
// The analysis below is the pure, device-independent decision the dispatch and its tests share: it derives
// each request's span from the per-row metadata the pool already receives, validates the ownership identity
// (wrong-delivery and partial-cancellation guards fail closed here, before any worker is submitted), and
// decides whether the window forms one merged group or falls back to per-request jobs when the merged
// formation bounds are exceeded. No CUDA type, allocation or waiting crosses this seam.
struct ExpertGroupSpan {
    uint64_t request = 0;      ///< the rows' request generation (0 = unbound, solo/legacy callers only)
    int32_t slot = -1;         ///< the batch slot, -1 for solo windows
    int64_t first_position = 0;///< the span's first speculative position
    int32_t first_row = 0;     ///< the span's first row in the window
    int32_t rows = 0;          ///< the span's packed rows (its verification-span length)
};

enum class ExpertGroupPlan {
    merged,    ///< one grouped formation over every staged row (the production path)
    per_span,  ///< merged bounds exceeded: independent per-request CPU jobs inside the same window ownership
    refused,   ///< the window's identity or shape cannot be served; the caller must fail the layer closed
};

struct ExpertGroupBounds {
    int64_t max_window_rows = 8;      ///< the window tables' row bound (MAXT)
    int64_t max_window_entries = 128; ///< the merged formation's routed-entry bound (the window tables)
    int64_t max_span_rows = 8;        ///< one request span's rows bound
    int32_t max_spans = 8;            ///< the span table's capacity
};

/// Walks one window's packed rows into per-request verification spans. `worker_requests[r]` is row r's
/// request generation (the hook substitutes each slotted row's published binding; solo rows carry the
/// dispatch's own window generation). A span extends while the request id, slot and consecutive positions
/// continue; every other invariant the resident verifier enforces upstream is re-checked here so the
/// dispatch fails closed on any window its result scatter could not be audited for. Refusals name the
/// reason and leave the span table cleared.
inline ExpertGroupPlan analyze_expert_groups(const int64_t* positions, const int* request_slots,
                                             const uint64_t* worker_requests, int64_t n_tok, int64_t k,
                                             const ExpertGroupBounds& bounds, ExpertGroupSpan* spans,
                                             int32_t& n_spans, const char*& error) {
    n_spans = 0;
    error = nullptr;
    if (n_tok < 0 || k < 0 || (n_tok > 0 && (positions == nullptr || worker_requests == nullptr))) {
        error = "expert group analysis: the window's request metadata is missing";
        return ExpertGroupPlan::refused;
    }
    bool bound = false, unbound = false;
    for (int64_t r = 0; r < n_tok; ++r) {
        if (positions[r] < 0 || positions[r] == std::numeric_limits<int64_t>::max()) {
            error = "expert group analysis: token position is outside the representable sequence extent";
            n_spans = 0;
            return ExpertGroupPlan::refused;
        }
        const int slot = request_slots ? request_slots[r] : -1;
        if (slot < -1) {
            error = "expert group analysis: a batch slot index is negative";
            n_spans = 0;
            return ExpertGroupPlan::refused;
        }
        const uint64_t id = worker_requests[r];
        if (id == 0) unbound = true; else bound = true;
        const bool extends = n_spans > 0 && spans[n_spans - 1].request == id &&
                             spans[n_spans - 1].slot == slot &&
                             positions[r] == spans[n_spans - 1].first_position + spans[n_spans - 1].rows;
        if (!extends) {
            // One request owns rows in one slot and one contiguous span: anything else could deliver a
            // result to a request at a position its verification span never held.
            for (int32_t s = 0; s < n_spans; ++s) {
                if (spans[s].request == id) {
                    error = spans[s].slot != slot
                          ? "expert group analysis: a request owns rows in more than one slot"
                          : "expert group analysis: a request's rows are not one contiguous verification span";
                    n_spans = 0;
                    return ExpertGroupPlan::refused;
                }
            }
            if (n_spans >= bounds.max_spans) {
                error = "expert group analysis: the window holds more request spans than the span table";
                n_spans = 0;
                return ExpertGroupPlan::refused;
            }
            spans[n_spans] = {id, slot, positions[r], (int32_t) r, 0};
            ++n_spans;
        }
        ++spans[n_spans - 1].rows;
    }
    if (bound && unbound) {
        error = "expert group analysis: the window mixes bound and unbound request identities";
        n_spans = 0;
        return ExpertGroupPlan::refused;
    }
    if (n_tok <= bounds.max_window_rows && n_tok * k <= bounds.max_window_entries)
        return ExpertGroupPlan::merged;
    if (n_spans <= 1) {
        error = "expert group analysis: the window exceeds the merged formation bounds";
        n_spans = 0;
        return ExpertGroupPlan::refused;
    }
    for (int32_t s = 0; s < n_spans; ++s)
        if (spans[s].rows > bounds.max_span_rows || (int64_t) spans[s].rows * k > bounds.max_window_entries) {
            error = "expert group analysis: one request span exceeds the per-request formation bounds";
            n_spans = 0;
            return ExpertGroupPlan::refused;
        }
    return ExpertGroupPlan::per_span;
}

// Ordered, non-owning helper dispatch. Construct once, outside dispatch; workers
// and any active ledger must outlive this module. One helper operation may be
// active at a time. The caller owns ledger.begin(), CPU/primary/peer claims and
// completions, final combine, and ledger.release() after its final consumer.
//
// begin preserves ExpertWork, substituting only each helper's residency version.
// assigned must equal work.assigned and remain alive until finish/cancel drains.
// Nonnegative assignments and existing ledger claims are never helper work;
// helper claims use first_owner + ordered index, with assignment_value written
// back before the next helper begins. Reserve that owner range for this module.
//
// No scheduler storage grows during dispatch. Adapters retain their own scratch
// and must keep error messages within kErrorCapacity to avoid string growth.
// Rejected preflight leaves existing ledger/work unchanged; the caller decides
// whether to retry corrected input or fail its layer. After any submission,
// failure poisons the ledger and attempts every outstanding drain. A failed
// cancel keeps submissions pinned; retry cancel before reuse/release/destruction.
// Destruction makes one final drain attempt; adapters remain responsible for
// retaining staging until their own successful drain if that attempt fails.
class ExpertHelperScheduler {
public:
    static constexpr size_t kErrorCapacity = 512;

    explicit ExpertHelperScheduler(const std::vector<ExpertWorker*>& workers,
                                   uint32_t first_owner = 4, int32_t assignment_value = 2)
        : workers_(workers.size()), first_owner_(first_owner), assignment_value_(assignment_value) {
        if (first_owner == 0 || assignment_value < 0 ||
            workers.size() > (uint64_t) std::numeric_limits<uint32_t>::max() - first_owner + 1)
            throw std::invalid_argument("invalid helper ownership range");
        for (size_t i = 0; i < workers.size(); ++i) {
            if (workers[i] == nullptr || std::find(workers.begin(), workers.begin() + i, workers[i]) != workers.begin() + i)
                throw std::invalid_argument("helper workers must be distinct and nonnull");
            workers_[i].worker = workers[i];
        }
        error_.reserve(kErrorCapacity);
        drain_error_.reserve(kErrorCapacity);
    }

    ExpertHelperScheduler(const ExpertHelperScheduler&) = delete;
    ExpertHelperScheduler& operator=(const ExpertHelperScheduler&) = delete;
    ExpertHelperScheduler(ExpertHelperScheduler&&) = delete;
    ExpertHelperScheduler& operator=(ExpertHelperScheduler&&) = delete;
    ~ExpertHelperScheduler() { if (active_) cancel(); }

    bool begin(const ExpertWork& work, int32_t* assigned, ExpertCompletion& completion) {
        if (active_) {
            if (error_.empty()) error_ = "helper operation is still active";
            return false;
        }
        error_.clear();
        if (work.operation.token_count < 0 || work.routed_width < 0 ||
            (work.routed_width != 0 && work.operation.token_count >
                std::numeric_limits<int64_t>::max() / work.routed_width))
            return reject("invalid helper routing geometry");
        const uint64_t count = (uint64_t) work.operation.token_count * (uint64_t) work.routed_width;
        if (count > std::numeric_limits<size_t>::max() ||
            !completion.matches(work.operation, (size_t) count))
            return reject("helper operation does not match the active ledger");
        if (assigned != work.assigned || (count != 0 && (assigned == nullptr || work.experts == nullptr)))
            return reject("helper routing buffers are missing or inconsistent");
        // Check before starting any worker: primary/CPU owner IDs must not alias
        // the reserved helper range, and every existing assignment needs a claim.
        for (size_t row = 0; row < (size_t) count; ++row) {
            const uint32_t owner = completion.owner(row);
            if ((owner != 0) != (assigned[row] >= 0) || helper_owner(owner))
                return reject("helper assignments disagree with existing ownership");
        }
        operation_ = work.operation;
        count_ = (size_t) count;
        completion_ = &completion;
        active_ = true;
        for (size_t i = 0; i < workers_.size(); ++i) {
            auto& state = workers_[i];
            state.version = state.worker->residency_version();
            if (!completion.retain_submission(operation_)) return fail("helper submission could not retain staging");
            state.pending = true; // A failed begin may already have launched work.
            ExpertWork submitted = work;
            submitted.weights.residency_version = state.version;
            if (!state.worker->begin(submitted, error_)) return fail("helper begin failed");
            if (state.worker->residency_version() != state.version)
                return fail("helper residency changed during submission");
            for (size_t row = 0; row < count_; ++row) {
                if (!state.worker->owns((int64_t) row)) continue;
                if (assigned[row] >= 0 || !completion.claim(row, owner_id(i), state.version))
                    return fail("expert row has multiple execution owners");
                assigned[row] = assignment_value_;
            }
        }
        return true;
    }

    // Ownership comes from router-index claims, never a later adapter query.
    bool owns(int64_t row) const {
        return active_ && row >= 0 && (uint64_t) row < count_ &&
               helper_owner(completion_->owner((size_t) row));
    }

    bool finish(float* output) {
        error_.clear();
        if (!active_) return reject("no helper operation is active");
        if (!completion_->matches(operation_, count_)) return fail("helper operation is no longer publishable");
        for (size_t i = 0; i < workers_.size(); ++i) {
            auto& state = workers_[i];
            if (state.worker->residency_version() != state.version)
                return fail("helper residency changed before completion");
            if (!state.worker->finish(output, error_)) return fail("helper finish failed");
            state.pending = false;
            completion_->drain_submission(operation_);
            if (state.worker->residency_version() != state.version)
                return fail("helper residency changed during completion");
            const uint32_t owner = owner_id(i);
            if (completion_->has_owner(owner) && !completion_->complete(operation_, owner, state.version))
                return fail("stale helper completion");
        }
        active_ = false;
        completion_ = nullptr;
        return true;
    }

    bool cancel() {
        if (!active_) return true;
        completion_->cancel();
        return drain();
    }

    bool active() const { return active_; }
    const std::string& error() const { return error_; }
    size_t storage_bytes() const {
        return workers_.capacity() * sizeof(State) + error_.capacity() + 1 + drain_error_.capacity() + 1;
    }
    static constexpr size_t planned_bytes(size_t workers) {
        return workers * sizeof(State) + 2 * (kErrorCapacity + 1);
    }

private:
    struct State {
        ExpertWorker* worker = nullptr;
        uint64_t version = 0;
        bool pending = false;
    };
    uint32_t owner_id(size_t index) const { return first_owner_ + (uint32_t) index; }
    bool helper_owner(uint32_t owner) const {
        return owner >= first_owner_ && (uint64_t) owner - first_owner_ < workers_.size();
    }
    bool reject(const char* message) { error_ = message; return false; }
    bool fail(const char* fallback) {
        if (error_.empty()) error_ = fallback;
        completion_->fail();
        drain();
        return false;
    }
    bool drain() {
        bool drained = true;
        for (size_t i = 0; i < workers_.size(); ++i) {
            auto& state = workers_[i];
            if (state.pending) {
                drain_error_.clear();
                if (!state.worker->cancel(drain_error_)) {
                    drained = false;
                    if (error_.empty()) {
                        if (drain_error_.empty()) error_ = "helper cancellation failed";
                        else error_ = drain_error_;
                    }
                    continue;
                }
                state.pending = false;
                completion_->drain_submission(operation_);
            }
            // Also covers finish succeeding before detecting a stale version.
            completion_->drained(operation_, owner_id(i));
        }
        if (drained) {
            active_ = false;
            completion_ = nullptr;
        }
        return drained;
    }

    std::vector<State> workers_;
    uint32_t first_owner_;
    int32_t assignment_value_;
    ExpertOperation operation_;
    size_t count_ = 0;
    ExpertCompletion* completion_ = nullptr;
    bool active_ = false;
    std::string error_;
    std::string drain_error_;
};

} // namespace strata::core
