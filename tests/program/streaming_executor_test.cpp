// Pure production selector/lifecycle; scalar working identity and scripted drain receipts only.
// No model, CUDA, ResidentSlot, source ring, CPU pool, scratch or KV allocation is constructed.
#include "strata/program/streaming_executor.hpp"
#include "strata/prefill/cpu_share_memory.hpp"
#include <cstdio>
#include <cstdlib>
#include <new>
#include <type_traits>

namespace { size_t allocations = 0; int failures = 0, assertions = 0; }
void* operator new(std::size_t bytes) {
    if (auto* pointer = std::malloc(bytes ? bytes : 1)) { ++allocations; return pointer; }
    throw std::bad_alloc();
}
void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { std::free(pointer); }

namespace {
using namespace strata::program;
void check(bool value, const char* message) {
    ++assertions;
    if (!value) { ++failures; std::fprintf(stderr, "FAIL: %s\n", message); }
}
struct Fixture {
    uint64_t request = 17;
    std::atomic<bool> stop{false};
    ExecutionPlan plan;
    StreamingSelection selection;
    strata::plan::Scope guest, host;
    std::vector<strata::plan::DeviceCost> devices{1};
    std::vector<strata::plan::DeviceTelemetry> telemetry{1};
    strata::plan::GuestHostPolicy policy;
    strata::plan::AdmissionDecision admission;
    std::string error;
    Fixture(bool share = false) {
        ExecutionPlanInput input;
        input.serve = true;
        plan = resolve_execution_plan(input);
        selection.serve = true;
        selection.effective_batch = 0; selection.destination = -1; selection.stages = 1;
        selection.source = StreamingSource::windowed; selection.source_ready = true;
        selection.source_lanes = 1; selection.cpu_share_resolved = true;
        guest.add("prefill_cpu_share_buffers", share ? strata::prefill::cpu_share_host_bytes(512, 256) : 0,
                  share ? "native share: existing CPU-free sizing helper" : "resolved explicitly disabled share");
        guest.add("source_ring", 4096, "synthetic fixed source envelope, no allocation");
        guest.mirror_all_into(host);
        devices[0].vram.add("working_state", 4096, "synthetic fixed state envelope, no allocation");
        telemetry[0].name = devices[0].name;
        telemetry[0].measured = true;
        telemetry[0].total_bytes = telemetry[0].free_bytes = 16ull << 30;
        policy.guest_total_measured = policy.host_available_measured = true;
        policy.guest_total_bytes = policy.host_available_bytes = 32ull << 30;
        error.reserve(512);
        resolve_prices();
    }
    void resolve_prices() {
        admission = strata::plan::admit_configuration(devices, telemetry, guest, host, policy);
        plan.memory_classes.clear();
        for (const auto& device : devices) for (const auto& price : device.vram.classes)
            plan.memory_classes.push_back({device.name, price.name, price.source, price.bytes, price.unknown});
        for (const auto& price : guest.classes)
            plan.memory_classes.push_back({"guest", price.name, price.source, price.bytes, price.unknown});
        for (const auto& price : host.classes)
            plan.memory_classes.push_back({"host", price.name, price.source, price.bytes, price.unknown});
    }
    StreamingWorkHandle submit(StreamingExecutor& executor) {
        return executor.submit({&request, &stop}, plan, admission, selection, error);
    }
};
StreamingDrainReceipt receipt(StreamingWorkHandle work, int64_t frontier, int64_t count) {
    return {work, frontier, count, true, true, true, true, true};
}
bool supported(const Fixture& fixture) {
    return streaming_capability(fixture.plan, fixture.admission, fixture.selection).support == Support::on;
}
}
int main() {
    using namespace strata::program;
    static_assert(!std::is_copy_constructible_v<StreamingExecutor> && !std::is_move_constructible_v<StreamingExecutor>);
    {
        Fixture f;
        check(f.admission.admitted && supported(f), "explicit verified serial selection accepts a known disabled-share zero");
        for (auto source : {StreamingSource::arena, StreamingSource::mapped, StreamingSource::windowed}) {
            f.selection.source = source;
            check(supported(f), "existing serial expert source choices keep distinct external owners");
        }
        Fixture enabled(true);
        check(enabled.guest.find("prefill_cpu_share_buffers")->bytes > 0 && supported(enabled),
              "native CPU-share price uses the existing helper, not an omitted demand");
        check(strata::prefill::cpu_share_host_bytes(0, 256) == UINT64_MAX,
              "unresolved/invalid share shape is an overflow refusal sentinel, not zero");
        for (int mutant = 0; mutant < 14; ++mutant) {
            Fixture invalid;
            switch (mutant) {
            case 0: invalid.selection.serve = false; break;
            case 1: invalid.selection.effective_batch = 2; break;
            case 2: invalid.selection.destination = 0; break;
            case 3: invalid.selection.stages = 2; break;
            case 4: invalid.plan.mode = ServeMode::resident; break;
            case 5: invalid.plan.admission.support = Support::conditional; break;
            case 6: invalid.plan.suspension_mode = Suspension::in_place; break;
            case 7: invalid.plan.concurrent_reader.support = Support::conditional; break;
            case 8: invalid.plan.proposal = ProposalShape::batched_single_draft; break;
            case 9: invalid.selection.source = StreamingSource::unresolved; break;
            case 10: invalid.selection.source_ready = false; break;
            case 11: invalid.selection.source_retired = true; break;
            case 12: invalid.selection.source_lanes = 2; break;
            case 13: invalid.selection.cpu_share_resolved = false; break;
            }
            StreamingExecutor executor;
            check(!supported(invalid) && !invalid.submit(executor) && !invalid.stop.load() && !executor.work(),
                  "invalid composition refuses before identity/control mutation");
        }
        for (int scope = 0; scope < 3; ++scope) {
            Fixture unknown;
            auto& demand = scope == 0 ? unknown.devices[0].vram : scope == 1 ? unknown.guest : unknown.host;
            demand.add_unknown("unresolved_demand", "deliberately unmeasured fixture");
            unknown.resolve_prices();
            check(!unknown.admission.admitted && !unknown.admission.unknown_classes.empty() && !supported(unknown),
                  "actual resource admission refuses unknown zero-byte device/guest/host demand");
        }
        for (int missing = 0; missing < 3; ++missing) {
            Fixture unknown;
            if (missing == 0) unknown.plan.memory_classes.clear();
            if (missing == 1) unknown.plan.memory_classes[1].name = "not_the_prefill_share_price";
            if (missing == 2) unknown.plan.memory_classes[1].unknown = true;
            check(!supported(unknown), "omitted or unresolved share prices cannot pass as zero even with a copied admission");
        }
    }
    {
        Fixture f;
        StreamingExecutor executor;
        const size_t before = allocations;
        const auto work = f.submit(executor);
        check(bool(work) && work.request == f.request, "submit binds only the existing working identity");
        check(executor.observe(work, f.error).event == StreamingEvent::prompt_frontier,
              "initial prompt frontier is observable");
        check(executor.observe(work, f.error).event == StreamingEvent::idle, "frontier is not replayed");
        check(executor.prefilled(work, 2, f.error), "drained prompt progress publishes without a token copy");
        check(executor.observe(work, f.error).frontier == 2, "observed prompt frontier is the selected drained frontier");
        check(executor.committed(work, 7, 3, f.error), "first committed output publishes");
        auto output = executor.observe(work, f.error);
        check(output.token == 7 && output.committed == 1 && output.fresh, "first committed token observed once");
        check(executor.committed(work, 8, 4, f.error), "second committed output publishes");
        output = executor.observe(work, f.error);
        check(output.token == 8 && output.committed == 2, "serial committed order is preserved");
        auto done = executor.drain(work, receipt(work, 4, 2), f.error);
        check(done.event == StreamingEvent::terminal && done.fresh && done.reason == StreamingTerminal::completed,
              "drained completion emits one fresh terminal after all output");
        check(!executor.observe(work, f.error).fresh && !executor.drain(work, receipt(work, 4, 2), f.error).fresh,
              "repeated observation/drain cannot replay completion");
        check(executor.request_cancel(work, StreamingTerminal::cancelled, f.error) && !f.stop.load(),
              "completion wins over late cancellation without changing stop control");
        check(executor.retire(work, f.error) && !executor.work(), "retire forgets the borrowed working binding");
        check(allocations == before, "successful production lifecycle allocates no heap storage");
    }
    {
        Fixture f;
        StreamingExecutor executor;
        const auto work = f.submit(executor);
        check(!f.submit(executor), "duplicate submit cannot discard a live cursor");
        check(!executor.prefilled(work, -1, f.error) && !executor.committed(work, -1, 1, f.error),
              "invalid publications refuse before mutation");
        check(executor.committed(work, 7, 1, f.error), "one issued output remains pending");
        check(!executor.committed(work, 8, 2, f.error), "unobserved output cannot be overwritten");
        check(executor.request_cancel(work, StreamingTerminal::timeout, f.error) && f.stop.load(),
              "cancellation is a control intention, not a resource release");
        for (int consumer = 0; consumer < 5; ++consumer) {
            auto pending = receipt(work, 1, 1);
            switch (consumer) {
            case 0: pending.verifier = false; break;
            case 1: pending.prefill_chain = false; break;
            case 2: pending.draft = false; break;
            case 3: pending.experts = false; break;
            case 4: pending.transfers = false; break;
            }
            check(!executor.drain(work, pending, f.error) && !executor.retire(work, f.error),
                  "a still-live consumer prevents drain and retirement without consuming output");
        }
        check(executor.observe(work, f.error).token == 7, "cancel/refused drain retains prior committed output");
        check(executor.committed(work, 8, 2, f.error), "an already-issued commit can publish after STOP");
        auto wrong = receipt(work, 2, 1);
        check(!executor.drain(work, wrong, f.error), "drain cannot undercount an issued committed publication");
        wrong = receipt(work, 2, 2); ++wrong.work.generation;
        check(!executor.drain(work, wrong, f.error), "drain receipt cannot name another work generation");
        auto drained = executor.drain(work, receipt(work, 2, 2), f.error);
        check(drained.event == StreamingEvent::idle && !drained.fresh && !executor.retire(work, f.error),
              "resource drain keeps pending output and blocks retirement");
        check(executor.observe(work, f.error).token == 8, "late committed token precedes cancellation terminal");
        auto done = executor.observe(work, f.error);
        check(done.event == StreamingEvent::terminal && done.fresh && done.committed == 2 &&
              done.reason == StreamingTerminal::timeout, "first cancellation reason survives full drain");
        check(!executor.observe(work, f.error).fresh && executor.retire(work, f.error), "terminal is idempotent before retire");
        ++f.request; f.stop.store(false);
        const auto successor = f.submit(executor);
        check(successor.generation > work.generation, "successor binds a fresh call-scope generation");
        check(!executor.request_cancel(work, StreamingTerminal::cancelled, f.error) && !f.stop.load() &&
              !executor.observe(work, f.error) && !executor.drain(work, receipt(work, 2, 2), f.error),
              "stale controls/publications/drain cannot touch the successor");
        check(executor.drain(successor, receipt(successor, 0, 0), f.error).fresh &&
              executor.retire(successor, f.error), "follow-up completion drains and retires independently");
    }
    {
        Fixture f;
        StreamingExecutor executor;
        f.stop.store(true);
        const auto work = f.submit(executor);
        check(bool(work) && f.stop.load(), "submit does not erase an earlier request-local STOP");
        auto done = executor.drain(work, receipt(work, 0, 0), f.error);
        check(done.fresh && done.committed == 0 && done.reason == StreamingTerminal::cancelled,
              "cancellation before prompt work drains a zero-output terminal");
    }
    std::printf("streaming executor: assertions=%d failures=%d executor_bytes=%zu (CPU contract only)\n",
                assertions, failures, sizeof(StreamingExecutor));
    return failures ? 1 : 0;
}
