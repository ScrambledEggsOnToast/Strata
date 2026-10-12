#include "strata/core/expert_worker.hpp"

#include <stdexcept>
#include <array>
#include <atomic>
#include <chrono>
#include <future>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

class MockDevice final : public strata::core::ExpertWorker {
public:
    explicit MockDevice(int expert, float result, uint64_t version = 1)
        : version(version), expert_(expert), result_(result) {}
    ~MockDevice() override { if (pending_.valid()) pending_.get(); }
    uint64_t residency_version() const override { return version; }
    bool begin(const strata::core::ExpertWork& work, std::string& error) override {
        ++begins;
        seen = work;
        if (pending_.valid()) { error = "mock busy"; return false; }
        count_ = work.operation.token_count * work.routed_width;
        if (count_ < 0 || count_ > (int64_t) selected_.size()) {
            error = "mock capacity";
            return false;
        }
        for (int64_t row = 0; row < count_; ++row)
            selected_[(size_t) row] = work.experts[row] == expert_ &&
                                    (ignore_assignments || work.assigned[row] < 0);
        completed.store(false);
        pending_ = std::async(std::launch::async, [this] {
            if (gate.valid()) gate.wait();
            completed.store(true);
            if (completion_signal != nullptr) completion_signal->set_value();
            return result_;
        });
        if (change_on_begin) ++version;
        if (fail_begin) { error = "mock begin failure"; return false; }
        return true;
    }
    bool owns(int64_t row) const override {
        return row >= 0 && row < count_ && selected_[(size_t) row];
    }
    bool finish(float* output, std::string& error) override {
        ++finishes;
        if (fail_finish) { error = "mock finish failure"; return false; }
        if (!pending_.valid()) { error = "mock inactive"; return false; }
        const float result = pending_.get();
        for (int64_t row = 0; row < count_; ++row) if (owns(row))
            for (int64_t column = 0; column < seen.weights.input_width; ++column)
                output[row * seen.weights.input_width + column] = result + (float) column;
        if (change_on_finish) ++version;
        return true;
    }
    bool cancel(std::string& error) override {
        ++cancels;
        if (cancel_failures != 0) {
            --cancel_failures;
            error = "mock cancel failure";
            return false;
        }
        if (pending_.valid()) pending_.get();
        selected_.fill(false);
        return true;
    }
    bool pending() const { return pending_.valid(); }

    uint64_t version;
    strata::core::ExpertWork seen;
    std::shared_future<void> gate;
    std::promise<void>* completion_signal = nullptr;
    std::atomic<bool> completed{false};
    int begins = 0;
    int finishes = 0;
    int cancels = 0;
    int cancel_failures = 0;
    bool fail_begin = false;
    bool fail_finish = false;
    bool ignore_assignments = false;
    bool change_on_begin = false;
    bool change_on_finish = false;
private:
    int expert_;
    float result_;
    int64_t count_ = 0;
    std::array<bool, 16> selected_{};
    std::future<float> pending_;
};

strata::core::ExpertWork make_work(const int32_t* ids, int32_t* assigned,
                                  int64_t tokens, int64_t routes, int64_t width = 1) {
    strata::core::ExpertWork work;
    work.operation = {12, 3, 17, tokens, 9};
    work.weights = {71, 42, ids, width, width * 3, 11, 17};
    work.experts = ids;
    work.routed_width = routes;
    work.assigned = assigned;
    return work;
}

void distinct_workers_preserve_rows_and_metadata() {
    using namespace strata::core;
    MockDevice first(9, 3.0f, 7), second(4, -2.0f, 19), duplicate(9, 80.0f, 23);
    ExpertCompletion rows(6);
    ExpertHelperScheduler scheduler({&first, &second, &duplicate});
    const size_t storage = scheduler.storage_bytes();
    int32_t ids[] = {9, 4, 9, 4, 9, 77};
    int32_t assigned[] = {-1, -1, -1, -1, 0, -1};
    float input[9]{};
    float output[18];
    std::fill(std::begin(output), std::end(output), 99.0f);
    int64_t positions[] = {17, 81, 6};
    int slots[] = {2, 0, 7};
    int32_t residency[] = {0};
    auto work = make_work(ids, assigned, 3, 2, 3);
    work.input = input;
    work.positions = positions;
    work.request_slots = slots;
    work.primary_residency = residency;
    require(rows.begin(work.operation, 6) && rows.claim(4, 2, 6), "primary claim failed");
    require(scheduler.begin(work, assigned, rows), "ordered helper submission failed");
    require(rows.owner(0) == 4 && rows.owner(1) == 5 && rows.owner(2) == 4 &&
            rows.owner(3) == 5 && rows.owner(4) == 2 && rows.owner(5) == 0,
            "helpers did not claim exact router indices");
    require(assigned[0] == 2 && assigned[1] == 2 && assigned[4] == 0 && assigned[5] == -1,
            "helper assignments overwrote primary or CPU routing");
    require(scheduler.owns(0) && scheduler.owns(3) && !scheduler.owns(4) &&
            !scheduler.owns(5) && !scheduler.owns(-1) && !scheduler.owns(6),
            "scheduler ownership does not match router claims");
    for (const auto* worker : {&first, &second, &duplicate}) {
        const auto& seen = worker->seen;
        require(seen.operation == work.operation && seen.input == input && seen.experts == ids &&
                seen.routed_width == 2 && seen.positions == positions && seen.request_slots == slots &&
                seen.assigned == assigned && seen.primary_residency == residency,
                "scheduler discarded input work metadata");
        require(seen.weights.model_generation == 71 && seen.weights.handle == ids &&
                seen.weights.input_width == 3 && seen.weights.hidden_width == 9 &&
                seen.weights.gate_up_format == 11 && seen.weights.down_format == 17 &&
                seen.weights.residency_version == worker->version,
                "scheduler discarded weight metadata or helper residency");
    }
    require(work.weights.residency_version == 42, "scheduler mutated caller weight metadata");
    require(rows.claim(5, 1, 3), "CPU remainder claim failed");
    require(rows.complete(work.operation, 1, 3), "CPU completion failed");
    require(!rows.publishable(), "helpers published unfinished operation");
    require(!scheduler.begin(work, assigned, rows), "active helper scratch was reused");
    require(scheduler.finish(output), "ordered helper completion failed");
    for (size_t row = 0; row < 6; ++row)
        for (size_t column = 0; column < 3; ++column) {
            const float expected = row < 4 ? (row % 2 == 0 ? 3.0f : -2.0f) + (float) column : 99.0f;
            require(output[row * 3 + column] == expected, "helper output lost exact row or width");
        }
    require(duplicate.finishes == 1 && duplicate.completed.load(), "zero-owned helper did not drain");
    require(!rows.publishable(), "helper finish completed primary ownership");
    require(rows.complete(work.operation, 2, 6) && rows.publishable(), "primary completion refused");
    require(rows.release(work.operation), "consumer could not release completed operation");
    require(scheduler.storage_bytes() == storage, "scheduler storage grew during dispatch");
}

void ordered_overlap_prefers_first_worker() {
    using namespace strata::core;
    for (bool reverse : {false, true}) {
        MockDevice first(9, 3.0f), second(9, 8.0f);
        ExpertCompletion rows(2);
        ExpertHelperScheduler scheduler(reverse ? std::vector<ExpertWorker*>{&second, &first}
                                               : std::vector<ExpertWorker*>{&first, &second}, 20, 7);
        int32_t ids[] = {9, 9}, assigned[] = {-1, -1};
        float output[] = {99, 99};
        auto work = make_work(ids, assigned, 2, 1);
        require(rows.begin(work.operation, 2) && scheduler.begin(work, assigned, rows), "overlap begin failed");
        require(rows.owner(0) == 20 && rows.owner(1) == 20 && assigned[0] == 7 && assigned[1] == 7,
                "ordered helper priority was ignored");
        require(scheduler.finish(output) && rows.publishable(), "overlap completion failed");
        require(output[0] == (reverse ? 8.0f : 3.0f) && output[1] == output[0], "wrong ordered helper won");
        require(rows.release(work.operation), "overlap release failed");
    }
}

void out_of_order_helpers_wait_for_publication() {
    using namespace strata::core;
    MockDevice first(9, 3.0f), second(4, 8.0f);
    ExpertCompletion rows(2);
    ExpertHelperScheduler scheduler({&first, &second});
    int32_t ids[] = {9, 4}, assigned[] = {-1, -1};
    float output[] = {99, 99};
    auto work = make_work(ids, assigned, 1, 2);
    std::promise<void> release_first, second_done;
    first.gate = release_first.get_future().share();
    second.completion_signal = &second_done;
    require(rows.begin(work.operation, 2) && scheduler.begin(work, assigned, rows), "out-of-order setup failed");
    second_done.get_future().wait();
    const bool incomplete = !first.completed.load() && second.completed.load() && !rows.publishable();
    std::promise<void> started;
    auto finishing = std::async(std::launch::async, [&] { started.set_value(); return scheduler.finish(output); });
    started.get_future().wait();
    const bool blocked = finishing.wait_for(std::chrono::milliseconds(10)) == std::future_status::timeout;
    release_first.set_value();
    require(finishing.get() && blocked && incomplete, "out-of-order completion published or skipped a wait");
    require(output[0] == 3 && output[1] == 8 && rows.publishable(), "out-of-order completion lost router rows");
    require(rows.release(work.operation), "out-of-order consumer release failed");
}

void scheduler_setup_rejects_ambiguous_workers() {
    using namespace strata::core;
    MockDevice worker(9, 3.0f);
    bool duplicate = false, null_worker = false, invalid_owner = false;
    try { ExpertHelperScheduler scheduler({&worker, &worker}); }
    catch (const std::invalid_argument&) { duplicate = true; }
    try { ExpertHelperScheduler scheduler({nullptr}); }
    catch (const std::invalid_argument&) { null_worker = true; }
    try { ExpertHelperScheduler scheduler({&worker}, 0); }
    catch (const std::invalid_argument&) { invalid_owner = true; }
    require(duplicate && null_worker && invalid_owner, "ambiguous helper configuration was accepted");
    ExpertCompletion rows(0);
    ExpertHelperScheduler empty({});
    auto work = make_work(nullptr, nullptr, 0, 0);
    require(rows.begin(work.operation, 0) && empty.begin(work, nullptr, rows), "empty helper list was rejected");
    require(empty.finish(nullptr) && rows.publishable() && rows.release(work.operation), "empty helper list did not finish");
}

void failures_drain_every_submission() {
    using namespace strata::core;
    for (bool during_begin : {true, false}) {
        MockDevice first(9, 3.0f), failed(4, -2.0f), last(7, 8.0f);
        failed.fail_begin = during_begin;
        failed.fail_finish = !during_begin;
        ExpertCompletion rows(3);
        ExpertHelperScheduler scheduler({&first, &failed, &last});
        int32_t ids[] = {9, 4, 7}, assigned[] = {-1, -1, -1};
        float output[] = {99, 99, 99};
        auto work = make_work(ids, assigned, 1, 3);
        require(rows.begin(work.operation, 3), "failure ledger begin failed");
        if (during_begin) require(!scheduler.begin(work, assigned, rows), "failed begin was accepted");
        else {
            require(scheduler.begin(work, assigned, rows), "failure setup failed");
            require(!scheduler.finish(output), "failed finish was accepted");
        }
        require(scheduler.error() == (during_begin ? "mock begin failure" : "mock finish failure"),
                "draining overwrote original worker error");
        require(failed.cancels == 1 && !failed.pending() && failed.completed.load(), "failed worker was not drained");
        require(!first.pending() && first.completed.load(), "earlier worker was not drained");
        require(during_begin ? (last.begins == 0 && last.cancels == 0)
                             : (last.cancels == 1 && !last.pending() && last.completed.load()),
                "later submissions were not drained exactly when submitted");
        require(!scheduler.active() && !rows.publishable(), "failed operation remained publishable or active");
        require(rows.release(work.operation), "drained failure retained ledger");
    }
}

void cancellation_waits_and_failed_drains_pin() {
    using namespace strata::core;
    MockDevice first(9, 3.0f), second(4, -2.0f);
    ExpertCompletion rows(2);
    ExpertHelperScheduler scheduler({&first, &second});
    int32_t ids[] = {9, 4}, assigned[] = {-1, -1};
    auto work = make_work(ids, assigned, 1, 2);
    std::promise<void> release;
    first.gate = release.get_future().share();
    require(rows.begin(work.operation, 2) && scheduler.begin(work, assigned, rows), "cancellation setup failed");
    std::promise<void> started;
    auto cancelling = std::async(std::launch::async, [&] { started.set_value(); return scheduler.cancel(); });
    started.get_future().wait();
    const bool blocked = cancelling.wait_for(std::chrono::milliseconds(10)) == std::future_status::timeout;
    release.set_value();
    require(cancelling.get() && blocked, "cancel returned before asynchronous staging drained");
    require(first.completed.load() && second.completed.load() && first.cancels == 1 && second.cancels == 1,
            "cancel skipped an outstanding helper");
    require(!rows.publishable() && rows.release(work.operation), "cancelled operation published or leaked");
    require(scheduler.cancel(), "drained cancellation was not idempotent");

    // A partially submitted begin can retain staging before it claims any rows.
    assigned[0] = assigned[1] = -1;
    ++work.operation.generation;
    first.fail_begin = true;
    first.cancel_failures = 1;
    require(rows.begin(work.operation, 2), "retry ledger begin failed");
    require(!scheduler.begin(work, assigned, rows) && scheduler.active(), "failed drain lost active staging");
    require(rows.owner(0) == 0 && !rows.release(work.operation), "unclaimed partial submission released staging");
    require(!scheduler.begin(work, assigned, rows), "failed cancellation allowed staging reuse");
    require(scheduler.cancel() && !scheduler.active(), "cancellation retry could not drain");
    require(scheduler.error() == "mock begin failure", "retry discarded originating failure");
    require(!rows.publishable() && rows.release(work.operation), "retried cancellation published or leaked");

    // A failed first cancellation must not prevent cancellation of later workers.
    first.fail_begin = false;
    first.cancel_failures = 1;
    assigned[0] = assigned[1] = -1;
    ++work.operation.generation;
    require(rows.begin(work.operation, 2) && scheduler.begin(work, assigned, rows), "multi-cancel setup failed");
    const int second_cancels = second.cancels;
    require(!scheduler.cancel() && scheduler.active(), "failed cancellation was accepted");
    require(second.cancels == second_cancels + 1 && !second.pending(), "failed cancellation skipped later workers");
    require(!rows.release(work.operation), "failed worker released staging");
    require(scheduler.cancel() && rows.release(work.operation), "remaining worker could not drain");
}

void stale_residency_never_publishes() {
    using namespace strata::core;
    for (int when = 0; when < 3; ++when) {
        MockDevice worker(9, 3.0f, 17), other(4, 8.0f, 27);
        worker.change_on_begin = when == 0;
        worker.change_on_finish = when == 2;
        ExpertCompletion rows(2);
        ExpertHelperScheduler scheduler({&worker, &other});
        int32_t ids[] = {9, 4}, assigned[] = {-1, -1};
        float output[] = {99, 99};
        auto work = make_work(ids, assigned, 1, 2);
        require(rows.begin(work.operation, 2), "stale residency setup failed");
        if (when == 0) require(!scheduler.begin(work, assigned, rows), "changed submission residency accepted");
        else {
            require(scheduler.begin(work, assigned, rows), "stale completion setup failed");
            if (when == 1) ++worker.version;
            require(!scheduler.finish(output), "stale completion was accepted");
        }
        require(!scheduler.error().empty() && !scheduler.active() && !rows.publishable(),
                "stale residency was published");
        require(!worker.pending() && !other.pending(), "stale residency leaked worker staging");
        require(when != 1 || (worker.finishes == 0 && output[0] == 99), "known stale weights reached finish");
        require(rows.release(work.operation), "stale residency failed to drain ledger");
    }
}

void duplicate_ownership_fails_closed() {
    using namespace strata::core;
    MockDevice first(9, 3.0f), second(9, 8.0f);
    second.ignore_assignments = true;
    ExpertCompletion rows(2);
    ExpertHelperScheduler scheduler({&first, &second});
    int32_t ids[] = {9, 9}, assigned[] = {-1, -1};
    auto work = make_work(ids, assigned, 1, 2);
    require(rows.begin(work.operation, 2), "duplicate claim setup failed");
    require(!scheduler.begin(work, assigned, rows), "worker stole previously claimed router rows");
    require(first.cancels == 1 && second.cancels == 1 && !first.pending() && !second.pending(),
            "ownership failure did not drain both submissions");
    require(!rows.publishable() && rows.release(work.operation), "duplicate ownership published or leaked");
}

void zero_rows_variable_widths_and_bounds() {
    using namespace strata::core;
    MockDevice worker(9, 3.0f);
    ExpertCompletion rows(6);
    ExpertHelperScheduler scheduler({&worker});
    const size_t storage = scheduler.storage_bytes();
    int32_t ids[] = {9, 9, 9, 9, 9, 9};
    int32_t assigned[6];
    for (int64_t width : {1, 3, 5}) {
        for (int64_t routes : {0, 1, 3}) {
            for (int64_t tokens : {0, 2}) {
                std::fill(std::begin(assigned), std::end(assigned), -1);
                const int64_t count = tokens * routes;
                auto work = make_work(count ? ids : nullptr, count ? assigned : nullptr, tokens, routes, width);
                work.operation.generation = (uint64_t) (100 + width * 30 + routes * 3 + tokens);
                float output[30];
                std::fill(std::begin(output), std::end(output), 99.0f);
                require(rows.begin(work.operation, (size_t) count) &&
                        scheduler.begin(work, count ? assigned : nullptr, rows), "bounded variable shape begin failed");
                require(!rows.publishable(), "zero-row submitted helper did not pin staging");
                require(scheduler.finish(count ? output : nullptr) && rows.publishable(), "variable shape finish failed");
                for (int64_t i = 0; i < 30; ++i)
                    require(output[i] == (i < count * width ? 3.0f + (float) (i % width) : 99.0f),
                            "variable shape output crossed a row or buffer bound");
                require(rows.release(work.operation), "variable shape ledger release failed");
                require(scheduler.storage_bytes() == storage, "variable shape grew scheduler storage");
            }
        }
    }
    auto work = make_work(ids, assigned, 2, 3);
    std::fill(std::begin(assigned), std::end(assigned), -1);
    require(rows.begin(work.operation, 6), "bounds ledger setup failed");
    const int begins = worker.begins;
    work.routed_width = 4;
    require(!scheduler.begin(work, assigned, rows), "routing exceeded active row bound");
    work.routed_width = std::numeric_limits<int64_t>::max();
    require(!scheduler.begin(work, assigned, rows), "routing count overflow was accepted");
    work.routed_width = -1;
    require(!scheduler.begin(work, assigned, rows), "negative routing width was accepted");
    work.routed_width = 3;
    ++work.operation.generation;
    require(!scheduler.begin(work, assigned, rows), "stale operation was accepted");
    --work.operation.generation;
    require(!scheduler.begin(work, nullptr, rows), "inconsistent ownership pointer was accepted");
    assigned[0] = 0;
    require(!scheduler.begin(work, assigned, rows), "unclaimed primary assignment was accepted");
    assigned[0] = -1;
    require(rows.claim(0, 4, 1), "owner collision setup failed");
    assigned[0] = 2;
    require(!scheduler.begin(work, assigned, rows), "reserved helper owner collision was accepted");
    require(worker.begins == begins, "invalid work reached a helper");
    rows.fail();
    require(rows.drained(work.operation, 4) && rows.release(work.operation), "bounds setup release failed");
}

void scheduler_destruction_drains() {
    using namespace strata::core;
    MockDevice worker(9, 3.0f);
    ExpertCompletion rows(1);
    int32_t ids[] = {9}, assigned[] = {-1};
    auto work = make_work(ids, assigned, 1, 1);
    require(rows.begin(work.operation, 1), "destruction setup failed");
    {
        ExpertHelperScheduler scheduler({&worker});
        require(scheduler.begin(work, assigned, rows), "destruction begin failed");
    }
    require(worker.cancels == 1 && worker.completed.load() && !worker.pending(), "scheduler destruction did not drain");
    require(!rows.publishable() && rows.release(work.operation), "destruction published or leaked");
}
void graph_worker_requires_matching_consumer() {
    using namespace strata::core;
    GraphExpertWorker primary;
    int32_t ids[] = {3, 7, 3}, assigned[] = {0, -1, 1};
    auto work = make_work(ids, assigned, 1, 3);
    std::string error;
    int publications = 0;
    primary.set_submission(&publications, [](void* p) { ++*static_cast<int*>(p); });
    require(primary.begin(work, error), "primary graph submission failed");
    assigned[0] = -1;
    require(primary.owns(0) && !primary.owns(1) && primary.owns(2), "graph ownership did not retain routing snapshot");
    require(publications == 1 && !primary.finish(nullptr, error), "publication falsely completed graph");
    auto stale = work.operation;
    ++stale.generation;
    require(!primary.consumed(stale), "stale graph consumer acknowledged current operation");
    require(!primary.cancel(error), "undrained graph cancellation released weights");
    require(primary.consumed(work.operation) && primary.finish(nullptr, error), "matching consumer did not complete graph");
    require(!primary.finish(nullptr, error), "duplicate primary completion accepted");
}

void split_stage_consumers_cannot_release_other_lanes() {
    using namespace strata::core;
    std::array<GraphExpertWorker, 4> primary;
    std::array<ExpertCompletion, 4> completion{
        ExpertCompletion(3), ExpertCompletion(3), ExpertCompletion(3), ExpertCompletion(3)};
    std::array<ExpertOperation, 4> operation;
    int32_t ids[] = {3, 7, 3}, assigned[] = {0, -1, 1};
    std::string error;
    for (size_t lane = 0; lane < 2; ++lane) {
        for (size_t group = 0; group < 2; ++group) {
            const size_t index = 2 * lane + group;
            auto work = make_work(ids, assigned, 1, 3);
            work.operation.layer = lane == 0 ? 3 : 19;
            work.operation.first_position += (int64_t) group;
            operation[index] = work.operation;
            require(completion[index].begin(work.operation, 3) &&
                    primary[index].begin(work, error), "split ownership lane did not start");
            require(completion[index].claim(0, 2, 1) && completion[index].claim(2, 2, 1) &&
                    completion[index].claim(1, 1, 1) && completion[index].complete(work.operation, 1, 1),
                    "split CPU/primary claims did not conserve rows");
        }
    }
    // Equal request/position/generation is not an acknowledgement from the other
    // stage. Both primary graphs may still own rows after the shared helpers drain.
    for (size_t group = 0; group < 2; ++group) {
        require(!primary[group].consumed(operation[2 + group]) &&
                !completion[group].complete(operation[2 + group], 2, 1) &&
                !completion[group].release(operation[group]),
                "later stage consumer released an earlier stage's rows");
        require(primary[2 + group].consumed(operation[2 + group]) &&
                primary[2 + group].finish(nullptr, error) &&
                completion[2 + group].complete(operation[2 + group], 2, 1) &&
                completion[2 + group].release(operation[2 + group]),
                "later stage could not release its own matching rows");
        require(!primary[group].finish(nullptr, error) && primary[group].owns(0),
                "later stage completion drained earlier stage primary work");
        require(primary[group].consumed(operation[group]) && primary[group].finish(nullptr, error) &&
                completion[group].complete(operation[group], 2, 1) &&
                completion[group].release(operation[group]),
                "earlier stage could not release its own matching rows");
    }
}

// Independent scalar work is evaluated in router order. Grouped evaluation may
// run in any group/member order, but writes unweighted original router indices.
float synthetic_expert(int expert, float input, int64_t layer, int column) {
    const float sign = expert & 1 ? -1.0f : 1.0f;
    return std::fma(input, sign * (float) (expert + 1),
                    std::ldexp((float) (layer + column + 1), (expert % 9) - 4));
}

void grouped_scatter_reduction_oracle() {
    using namespace strata::core;
    std::mt19937 random(38);
    uint64_t checked_entries = 0, checked_groups = 0, overlap = 0;
    for (int trial = 0; trial < 2048; ++trial) {
        const int tokens = trial % 9, routes = 1 + (trial / 9) % 16, columns = 1 + trial % 5;
        std::array<int32_t, 128> ids{}, assigned{}, workers{};
        std::array<uint64_t, 8> requests{};
        std::array<int64_t, 8> positions{};
        std::array<int, 8> slots{};
        std::array<float, 8> input{};
        std::array<float, 128> coefficients{};
        for (int t = 0; t < tokens; ++t) {
            // Two positions per request, unrelated requests at different positions,
            // and physical slots unrelated to the packed residual row index.
            requests[t] = 100 + (uint64_t) (t / 2);
            positions[t] = 700 * (t / 2) + t % 2;
            slots[t] = 7 - t / 2;
            input[t] = std::ldexp((float) (t + 1), trial % 11 - 5);
        }
        for (int i = 0; i < tokens * routes; ++i) {
            ids[i] = (int32_t) (random() % (trial % 2 ? 512 : 7));
            workers[i] = 1 + (int32_t) (random() % 5);
            coefficients[i] = (float) (random() % 17) / 16.0f; // includes exact zero
        }
        auto work = make_work(ids.data(), assigned.data(), tokens, routes, columns);
        work.operation.layer = trial % 48;
        work.input = input.data();
        work.request_ids = requests.data();
        work.positions = positions.data();
        work.request_slots = slots.data();
        ExpertRouteGroups groups;
        require(groups.form(work, nullptr, 512), "bounded ready group formation failed");
        require(groups.resolve_workers(workers.data()) || groups.form(work, workers.data(), 512),
                "exact resolved-worker partition failed");
        require(groups.entries() == (size_t) (tokens * routes), "group lost routed entries");
        std::array<unsigned, 128> visits{};
        std::array<float, 128 * 5 + 2> parts;
        parts.fill(12345);
        // Reverse completion order deliberately; destinations, not completions,
        // determine the final reduction order.
        for (size_t g = groups.size(); g-- > 0;) {
            const auto& group = groups.group(g);
            uint16_t previous = 0;
            for (size_t m = 0; m < group.count; ++m) {
                const auto entry = groups.entry(g, m);
                require(entry.destination < tokens * routes && entry.token < tokens &&
                        entry.route < routes && entry.destination == entry.token * routes + entry.route,
                        "scatter crossed the routed shape");
                require(ids[entry.destination] == entry.expert && workers[entry.destination] == entry.worker,
                        "expert/worker grouping changed the resolved placement");
                require(!m || previous < entry.destination, "group changed member order");
                previous = entry.destination;
                ++visits[entry.destination];
                require(groups.matches(entry, requests[entry.token], positions[entry.token],
                                       slots[entry.token], work.operation, work.weights),
                        "request/position identity was lost");
                for (int c = 0; c < columns; ++c)
                    parts[1 + entry.destination * columns + c] =
                        synthetic_expert(entry.expert, input[entry.token], work.operation.layer, c);
            }
        }
        for (int t = 0; t < tokens; ++t)
            for (int c = 0; c < columns; ++c) {
                float separate = 0, grouped = 0;
                unsigned coefficient_visits = 0;
                for (int route = 0; route < routes; ++route) {
                    const int i = t * routes + route;
                    require(visits[i] == 1, "selected contribution was omitted or duplicated");
                    separate = std::fma(coefficients[i],
                        synthetic_expert(ids[i], input[t], work.operation.layer, c), separate);
                    grouped = std::fma(coefficients[i], parts[1 + i * columns + c], grouped);
                    ++coefficient_visits;
                }
                require(coefficient_visits == (unsigned) routes &&
                        std::memcmp(&grouped, &separate, sizeof(float)) == 0,
                        "grouping changed routing coefficients or FP32 reduction order");
            }
        for (size_t i = 1 + (size_t) tokens * routes * columns; i < parts.size(); ++i)
            require(parts[i] == 12345, "scatter overwrote unused result storage");
        require(parts[0] == 12345, "scatter overwrote leading canary");
        checked_entries += groups.entries();
        checked_groups += groups.size();
        overlap += groups.cross_request_groups();
    }
    // This oracle would fail if a later implementation reduced in expert order.
    const float a = std::fma(1.0f, -1e20f, std::fma(1.0f, 1.0f, 1e20f));
    const float b = std::fma(1.0f, 1.0f, std::fma(1.0f, -1e20f, 1e20f));
    require(a != b, "summation-order negative control has no power");
    std::printf("scatter/reduction: 2048 mixed windows, %llu entries exactly once, %llu groups, "
                "%llu cross-request groups; bit-exact router-order reduction; launch savings unmeasured\n",
                (unsigned long long) checked_entries, (unsigned long long) checked_groups,
                (unsigned long long) overlap);
}

void group_snapshot_cancellation_and_stale_delivery() {
    using namespace strata::core;
    int32_t ids[] = {9, 4, 9, 7, 9, 4, 9, 7}, workers[] = {2, 1, 2, 1, 2, 1, 2, 1};
    int32_t assigned[8]{};
    uint64_t requests[] = {41, 41, 77, 77};
    int64_t positions[] = {17, 18, 901, 902};
    int slots[] = {7, 7, 2, 2};
    auto work = make_work(ids, assigned, 4, 2);
    work.request_ids = requests; work.positions = positions; work.request_slots = slots;
    ExpertRouteGroups groups;
    require(groups.form(work, workers, 512) && groups.cross_request_groups() == 3,
            "mixed-request shared experts were not grouped");
    const ExpertOperation operation = work.operation;
    const ExpertWeights weights = work.weights;
    std::promise<void> started, release;
    auto ready = started.get_future();
    auto gate = release.get_future().share();
    // A helper completes after the caller observes cancellation and destroys its
    // input metadata. The immutable group and its outputs remain alive until drain.
    auto helper = std::async(std::launch::async, [groups, gate, &started] {
        started.set_value();
        gate.wait();
        std::array<float, 8> result{};
        for (size_t g = 0; g < groups.size(); ++g)
            for (size_t m = 0; m < groups.group(g).count; ++m) {
                const auto e = groups.entry(g, m);
                result[e.destination] = synthetic_expert(e.expert, (float) e.position, groups.operation().layer, 0);
            }
        return result;
    });
    ready.wait();
    requests[0] = requests[1] = 0; // retired request, while the group is still in flight
    std::fill(std::begin(ids), std::end(ids), -1);
    std::fill(std::begin(positions), std::end(positions), -1);
    std::fill(std::begin(workers), std::end(workers), -1);
    release.set_value();
    const auto result = helper.get();
    float delivered[8];
    std::fill(std::begin(delivered), std::end(delivered), 12345);
    unsigned discarded = 0, committed = 0;
    for (size_t g = 0; g < groups.size(); ++g)
        for (size_t m = 0; m < groups.group(g).count; ++m) {
            const auto e = groups.entry(g, m);
            if (!groups.matches(e, requests[e.token], e.position, e.slot, operation, weights)) { ++discarded; continue; }
            delivered[e.destination] = result[e.destination];
            ++committed;
        }
    require(discarded == 4 && committed == 4, "partial cancellation affected the other request");
    for (int i = 0; i < 8; ++i)
        require(delivered[i] == (i < 4 ? 12345 : synthetic_expert(i % 2 ? (i == 5 ? 4 : 7) : 9,
                    (float) (901 + (i - 4) / 2), operation.layer, 0)),
                "cancelled member leaked or surviving speculative position was corrupted");
    const auto e = groups.entry(0, 0);
    auto op = operation; ++op.layer;
    require(!groups.matches(e, e.request, e.position, e.slot, op, weights), "wrong layer accepted result");
    op = operation; ++op.generation;
    require(!groups.matches(e, e.request, e.position, e.slot, op, weights), "rejected span retry accepted stale result");
    auto version = weights; ++version.residency_version;
    require(!groups.matches(e, e.request, e.position, e.slot, operation, version), "stale weights accepted result");
    version = weights; ++version.gate_up_format;
    require(!groups.matches(e, e.request, e.position, e.slot, operation, version), "mixed formats accepted result");
    require(!groups.matches(e, e.request + 1, e.position, e.slot, operation, weights) &&
            !groups.matches(e, e.request, e.position + 1, e.slot, operation, weights) &&
            !groups.matches(e, e.request, e.position, e.slot - 1, operation, weights),
            "request, speculative position or reused slot accepted stale output");
    auto forged = e; ++forged.expert;
    require(!groups.matches(forged, e.request, e.position, e.slot, operation, weights), "wrong expert accepted output");
    forged = e; ++forged.worker;
    require(!groups.matches(forged, e.request, e.position, e.slot, operation, weights), "wrong worker accepted output");
    std::printf("asynchronous scatter: cancelled 4 entries after submission, committed 4 surviving entries; "
                "stale request/layer/position/span/worker/format/residency refused\n");
}

void group_bounds_and_worker_partitions() {
    using namespace strata::core;
    int32_t ids[128]{}, assigned[128]{}, workers[128]{};
    std::fill(std::begin(workers), std::end(workers), 1);
    auto work = make_work(ids, assigned, 8, 16);
    ExpertRouteGroups groups;
    require(groups.form(work, workers, 512) && groups.entries() == 128 &&
            groups.size() == 1 && groups.group(0).count == 128, "full repeated-expert extent failed");
    require(groups.resolve_workers(workers) && groups.group(0).worker == 1,
            "static placement could not resolve the formed expert group");
    for (int i = 0; i < 128; ++i) workers[i] = 1 + i % 3;
    require(!groups.resolve_workers(workers) && groups.size() == 1,
            "row-selective placement was merged into one worker");
    require(groups.form(work, workers, 512) && groups.size() == 3, "same expert crossed worker partitions");
    work.operation.token_count = 9;
    require(!groups.form(work, workers, 512) && groups.size() == 3, "oversized window replaced valid metadata");
    work.operation.token_count = 8; work.routed_width = 17;
    require(!groups.form(work, workers, 512), "entry cap overflow accepted");
    work.routed_width = INT64_MAX;
    require(!groups.form(work, workers, 512), "multiplication overflow accepted");
    work.routed_width = 16; ids[127] = 512;
    require(!groups.form(work, workers, 512), "invalid expert accepted");
    ids[127] = 0;
    work.operation.token_count = 0;
    require(groups.form(work, workers, 512) && !groups.entries() && !groups.size(), "empty ready window retained work");
    require(sizeof(groups) == 2048, "global group scratch grew");
}

} // namespace

int main() {
    using namespace strata::core;
    grouped_scatter_reduction_oracle();
    group_snapshot_cancellation_and_stale_delivery();
    group_bounds_and_worker_partitions();
    graph_worker_requires_matching_consumer();
    split_stage_consumers_cannot_release_other_lanes();
    distinct_workers_preserve_rows_and_metadata();
    ordered_overlap_prefers_first_worker();
    out_of_order_helpers_wait_for_publication();
    scheduler_setup_rejects_ambiguous_workers();
    failures_drain_every_submission();
    cancellation_waits_and_failed_drains_pin();
    stale_residency_never_publishes();
    duplicate_ownership_fails_closed();
    zero_rows_variable_widths_and_bounds();
    scheduler_destruction_drains();
    ExpertCompletion rows(4);
    const ExpertOperation operation{7, 3, 19, 2, 11};
    require(rows.begin(operation, 3), "bounded operation refused");
    require(rows.claim(0, 1, 4), "first row refused");
    require(!rows.claim(0, 2, 4), "two workers own the same routed row");
    require(rows.claim(1, 2, 9), "second worker refused");
    require(rows.claim(2, 1, 4), "duplicate expert on another token refused");
    require(!rows.publishable(), "unfinished layer became publishable");
    require(rows.complete(operation, 1, 4), "first completion refused");
    require(!rows.publishable(), "partial layer became publishable");
    require(!rows.begin(operation, 1), "in-flight scratch was reused");
    require(rows.complete(operation, 2, 9), "second completion refused");
    require(rows.publishable(), "complete layer was not publishable");
    require(!rows.begin(operation, 1), "scratch reused before last consumer");
    require(rows.release(operation), "last consumer could not release");
    require(rows.begin(ExpertOperation{7, 3, 21, 1, 12}, 0), "empty routing refused");
    require(rows.publishable(), "empty routing is not complete");
    const ExpertOperation empty{7, 3, 21, 1, 12};
    require(rows.release(empty), "empty operation release refused");
    const ExpertOperation cancelled{8, 3, 21, 2, 13};
    require(rows.begin(cancelled, 2), "cancel operation refused");
    require(rows.claim(0, 1, 5), "cancel first claim refused");
    require(rows.claim(1, 2, 10), "cancel second claim refused");
    rows.cancel();
    require(!rows.complete(operation, 1, 5), "stale operation completed current work");
    require(!rows.complete(cancelled, 1, 6), "changed residency completed old work");
    require(!rows.release(cancelled), "cancellation freed in-flight scratch");
    require(rows.complete(cancelled, 2, 10), "out-of-order drain refused");
    require(rows.complete(cancelled, 1, 5), "cancelled work could not drain");
    require(!rows.publishable(), "cancelled layer became publishable");
    require(!rows.complete(cancelled, 1, 5), "duplicate completion accepted");
    require(rows.release(cancelled), "drained cancellation did not release");
    require(!rows.begin(cancelled, 5), "row budget exceeded");
    const ExpertOperation failed{9, 4, 23, 1, 14};
    require(rows.begin(failed, 2), "failed operation refused");
    require(rows.claim(0, 4, 12) && rows.claim(1, 5, 17), "helper claims refused");
    require(!rows.drained(failed, 4), "normal operation accepted cancellation drain");
    rows.fail();
    require(!rows.drained(cancelled, 4), "stale cancellation drained current owner");
    require(rows.drained(failed, 4), "successful helper cancellation not recorded");
    require(!rows.release(failed), "undrained helper ownership was released");
    require(rows.drained(failed, 5), "second helper drain refused");
    require(!rows.publishable(), "failed operation became publishable after drain");
    require(rows.release(failed), "fully drained failed operation retained scratch");
    return 0;
}
