#include "strata/core/expert_worker.hpp"
#include <chrono>
#include <future>
#include <stdexcept>
#include <thread>

namespace {
void require(bool ok) { if (!ok) throw std::runtime_error("helper cost/ownership failure"); }
class DelayedWorker final : public strata::core::ExpertWorker {
public:
    std::shared_future<void> gate;
    std::future<void> job;
    bool pending = false;
    uint64_t residency_version() const override { return 1; }
    bool begin(const strata::core::ExpertWork&, std::string&) override {
        pending = true;
        job = std::async(std::launch::async, [g=gate] { g.wait(); });
        return true;
    }
    bool owns(int64_t row) const override { return row == 0; }
    bool finish(float* output, std::string&) override {
        job.get(); pending=false; output[0]=7; return true;
    }
    bool cancel(std::string&) override { job.get(); pending=false; return true; }
};
}
int main() {
    using namespace strata::core;
    DelayedWorker worker;
    ExpertHelperScheduler scheduler({&worker});
    ExpertCompletion ledger(1);
    int32_t experts[]={3}, assigned[]={-1}; float input=2, output=99;
    ExpertWork work;
    work.operation={1,2,4,1,3}; work.weights={1,1,experts,1,1,0,0};
    work.input=&input; work.experts=experts; work.assigned=assigned; work.routed_width=1;
    std::promise<void> release; worker.gate=release.get_future().share();
    require(ledger.begin(work.operation,1) && scheduler.begin(work,assigned,ledger));
    require(worker.pending && !ledger.publishable() && scheduler.worker_cost(0).queue_service_us==0);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    release.set_value();
    require(scheduler.finish(&output) && output==7 && ledger.publishable());
    const auto cost=scheduler.worker_cost(0);
    require(cost.submissions==1 && cost.queue_service_us>=19000);
    require(ledger.release(work.operation));
    // Fresh operation cannot inherit the earlier queue cost or publish stale completion.
    ++work.operation.generation; assigned[0]=-1;
    require(ledger.begin(work.operation,1) && scheduler.begin(work,assigned,ledger));
    require(scheduler.worker_cost(0).queue_service_us==0);
    require(scheduler.cancel() && !worker.pending && !ledger.publishable());
    require(ledger.release(work.operation));
}
