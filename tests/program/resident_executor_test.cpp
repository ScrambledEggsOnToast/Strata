// No-load contract fixture: real ResidentSlots transitions, no model/graphs/device allocation.
// The fixture seeds only the admitted physical owners and the verifier's committed-row witness;
// it does not implement another lifecycle or replace the production drain/cancellation rules.
#include "strata/program/resident_executor.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/mtp.hpp"
#include "strata/core/verify.hpp"
#include "strata/prefill/prefill.hpp"

#include <cstdio>
#include <type_traits>

namespace strata::program {
struct ResidentExecutorFixture {
    static core::ModelGeometry empty_geometry() {
        core::ModelGeometry geometry;
        geometry.n_layers = 0;
        return geometry;
    }
    core::ModelGeometry geometry = empty_geometry();
    core::SessionOwner main, a, b;
    core::Verifier verifier;
    core::MtpDrafter draft;
    prefill::Prefill prompt;
    core::ExpertDispatch dispatch;
    ResidentSlots slots{geometry, {{&main.state(), &verifier, {&a.state(), &b.state()}, -1}},
                        draft, prompt, dispatch, 64, true, 6, 4096};
    ResidentPreparation lease;
    ExecutionPlan plan;
    std::string error;

    ResidentExecutorFixture() {
        slots.resources_ = ResidentSlots::Resources::ready;
        a.state().allocation_id = 11;
        a.state().allocation_device = -1;
        b.state().allocation_id = 12;
        b.state().allocation_device = -1;
        // No streams, graphs, weights or buffers. context_idle/wait_commit use their real
        // empty-operation path and never query CUDA for a negative device ordinal.
        verifier.g_ = &geometry;
        verifier.ss_ = &main.state();
        verifier.slots_ = {&a.state(), &b.state()};
        dispatch.completion.emplace_back(core::ExpertCompletion::kCapacity);
        ExecutionPlanInput input;
        input.serve = true;
        input.batch_requested = 2;
        input.spec_requested = 2;
        plan = resolve_execution_plan(input);
        reserve();
        auto& peer = slots.slots_[1];
        peer.request = 23; peer.incarnation = 9; peer.active = true;
        peer.ids = {21, 22, 23}; peer.p = 3; peer.produced = 2; peer.sampling.seed = 99;
        peer.prompt_cursor.operation = 51;
        dispatch.slot_requests[1] = 23;
    }
    void reserve(uint64_t request = 17, uint64_t incarnation = 3) {
        auto& owner = slots.slots_[0];
        owner.active = owner.stop = owner.failed = owner.tainted = owner.cached = false;
        owner.preparing = true;
        owner.request = request; owner.incarnation = incarnation;
        owner.ids.clear(); owner.prompt = {11, 12, 13};
        owner.p = owner.produced = 0; owner.max_new = owner.input.max_new = 8;
        owner.input.scope.value[0] = 1;
        slots.preparations_[0] = {};
        slots.preparations_[0].ready = true;
        dispatch.slot_requests[0] = request;
        verifier.committed_slot_row_[0] = {};
        lease = {};
        lease.slot = 0; lease.request = request; lease.incarnation = incarnation; lease.endpoint = 2;
        lease.stages[0] = &a.state(); lease.allocations[0] = a.state().allocation_id;
    }
    void final_witness() {
        auto& row = verifier.committed_slot_row_[0];
        row.request_id = lease.request; row.position = lease.endpoint; row.token = 13;
    }
    bool first(int32_t token = 31) {
        if (!slots.prefilled(lease, lease.endpoint, error)) return false;
        final_witness();
        return slots.first_token(lease, token, nullptr, error);
    }
    bool peer_untouched() const {
        const auto& peer = slots.slots()[1];
        return peer.request == 23 && peer.incarnation == 9 && peer.active && !peer.stop && !peer.failed &&
            peer.ids == std::vector<int32_t>({21, 22, 23}) && peer.p == 3 && peer.produced == 2 &&
            peer.sampling.seed == 99 && peer.prompt_cursor.operation == 51 && dispatch.slot_requests[1] == 23;
    }
};
} // namespace strata::program

namespace {
int failures = 0;
void check(bool condition, const char* message) {
    if (!condition) { std::fprintf(stderr, "FAIL: %s\n", message); ++failures; }
}
}

int main() {
    using namespace strata::program;
    static_assert(!std::is_copy_constructible_v<ResidentExecutor> && !std::is_move_constructible_v<ResidentExecutor>);
    static_assert(!std::is_copy_constructible_v<ResidentSnapshotLease> &&
                   std::is_move_constructible_v<ResidentSnapshotLease>);
    {
        ResidentExecutorFixture f;
        ResidentExecutor executor(f.slots);
        const auto work = executor.submit(f.lease, f.plan, f.error);
        check(bool(work) && work.request == 17 && work.incarnation == 3, "submit returns the admitted identity and generation");
        const auto* history = f.slots.slots()[0].ids.data();
        const auto initial = executor.observe(work, f.error);
        check(initial && initial.event == ResidentEvent::prompt_frontier && initial.frontier == 0 && initial.fresh &&
              initial.work == work, "submit -> observe publishes the reserved prompt frontier with its identity");
        check(executor.observe(work, f.error).event == ResidentEvent::idle, "polling a frontier does not replay it");
        check(f.slots.prefilled(f.lease, 1, f.error), "the real owner publishes a drained prompt interval");
        const auto frontier = executor.observe(work, f.error);
        check(frontier.event == ResidentEvent::prompt_frontier && frontier.frontier == 1, "observe consumes certified prompt progress");
        check(f.first(), "the real first-token transition accepts a committed final-row witness");
        const auto first = executor.observe(work, f.error);
        check(first.event == ResidentEvent::committed_token && first.token == 31 && first.committed == 1 &&
              first.work == work, "observe reports the first committed token, not a draft");
        check(f.slots.committed(0, 32, nullptr, f.error) && f.slots.committed(0, 33, nullptr, f.error),
              "two real committed transitions can precede observation");
        check(executor.observe(work, f.error).token == 32 && executor.observe(work, f.error).token == 33,
              "a multi-row publication is observed in order without copying token history");
        const auto done = executor.drain(work, f.error);
        check(done && done.event == ResidentEvent::terminal && done.fresh && done.committed == 3 &&
              done.frontier == 5 && done.reason == ResidentTerminal::completed && done.work == work,
              "drain finishes the real owner and reports one identity-bound terminal");
        check(!executor.drain(work, f.error).fresh && !executor.observe(work, f.error).fresh &&
              !executor.cancel(work, ResidentTerminal::cancelled, f.error).fresh && !f.slots.slots()[0].stop,
              "repeated drain/observe/late cancel cannot duplicate or change the completed terminal");
        check(f.slots.slots()[0].ids.data() == history && f.peer_untouched(),
              "the facade neither reallocates committed history nor changes the peer");
        auto snapshot = executor.borrow_snapshot(work, f.error);
        check(snapshot && snapshot.work() == work && snapshot.source()->certificate.certified &&
              snapshot.source()->certificate.request == 17 && snapshot.source()->certificate.incarnation == 3 &&
              snapshot.source()->certificate.valid_end == 5 &&
              &snapshot.source()->checkpoints == &f.slots.slots()[0].checks,
              "borrow returns the existing certified cut/checkpoint chain, not a copy");
        const auto generation = snapshot.generation();
        auto duplicate = executor.borrow_snapshot(work, f.error);
        check(!duplicate && !f.error.empty(), "a second outstanding snapshot borrow is refused");
        check(!executor.drain(work, f.error) && snapshot, "drain refuses while the certified source is borrowed");
        ResidentSnapshotLease moved(std::move(snapshot));
        check(moved && !snapshot && moved.generation() == generation, "moving transfers exactly one snapshot borrow");
        check(bool(executor.return_snapshot(moved, f.error)) && !moved, "return releases the exact borrowed generation");
        auto next = executor.borrow_snapshot(work, f.error);
        check(next && next.generation() > generation && !executor.return_snapshot(moved, f.error) && next,
              "an old returned lease cannot release a newer borrow generation");
        check(bool(executor.return_snapshot(next, f.error)) && f.peer_untouched(), "a refused old return leaves the current borrow and peer intact");
        f.slots.taint(0);
        check(!executor.borrow_snapshot(work, f.error) && f.error.empty(), "a tainted completed owner is a clean snapshot miss");
    }
    {
        ResidentExecutorFixture f;
        ResidentExecutor executor(f.slots);
        const auto work = executor.submit(f.lease, f.plan, f.error);
        check(f.first() && f.slots.committed(0, 32, nullptr, f.error) &&
              f.slots.committed(0, 33, nullptr, f.error), "three outputs commit before any output observation");
        const auto drained = executor.drain(work, f.error);
        check(drained && drained.event == ResidentEvent::idle && !drained.fresh && drained.committed == 3 &&
              !f.slots.slots()[0].active && f.slots.slots()[0].ids == std::vector<int32_t>({11, 12, 13, 31, 32}),
              "drain retires resources but retains unobserved committed output and its history");
        check(executor.drain(work, f.error).event == ResidentEvent::idle &&
              executor.cancel(work, ResidentTerminal::disconnected, f.error).event == ResidentEvent::idle,
              "repeated drain and late cancellation cannot skip outstanding committed observations");
        for (int32_t token : {31, 32, 33}) {
            const auto output = executor.observe(work, f.error);
            check(output && output.event == ResidentEvent::committed_token && output.token == token &&
                  output.work == work && output.fresh, "observe after drain preserves every committed token in order");
        }
        const auto terminal = executor.observe(work, f.error);
        check(terminal && terminal.event == ResidentEvent::terminal && terminal.fresh && terminal.committed == 3 &&
              terminal.reason == ResidentTerminal::completed && !executor.observe(work, f.error).fresh &&
              !executor.drain(work, f.error).fresh && f.peer_untouched(),
              "the only fresh terminal follows all previously unobserved output, with the peer untouched");
    }
    {
        ResidentExecutorFixture f;
        ResidentExecutor executor(f.slots);
        const auto work = executor.submit(f.lease, f.plan, f.error);
        const auto cancelled = executor.cancel(work, ResidentTerminal::timeout, f.error);
        check(cancelled && cancelled.event == ResidentEvent::terminal && cancelled.fresh && cancelled.work == work &&
              cancelled.reason == ResidentTerminal::timeout && !f.slots.slots()[0].request &&
              !f.slots.slots()[0].preparing, "cancel immediately after submit aborts the real reservation and drains one terminal");
        const auto again = executor.cancel(work, ResidentTerminal::disconnected, f.error);
        check(again && !again.fresh && again.reason == ResidentTerminal::timeout &&
              !executor.drain(work, f.error).fresh && !executor.observe(work, f.error).fresh && f.peer_untouched(),
              "cancel/drain/observe are idempotent after abort retired the request identity");
        f.reserve(18, 4);
        const auto successor = executor.submit(f.lease, f.plan, f.error);
        check(successor && successor.generation > work.generation, "a retired call scope can attach to a new admitted generation");
        const auto stale = executor.observe(work, f.error);
        check(!stale && stale.work == work && !executor.cancel(work, ResidentTerminal::cancelled, f.error) &&
              !executor.drain(work, f.error) && !executor.borrow_snapshot(work, f.error) &&
              f.slots.slots()[0].request == 18 && !f.slots.slots()[0].stop && f.peer_untouched(),
              "stale observe/cancel/drain/borrow results cannot claim a successor or affect a peer");
        check(bool(executor.cancel(successor, ResidentTerminal::cancelled, f.error)), "the successor retains its own usable cancellation lease");
    }
    {
        ResidentExecutorFixture f;
        ResidentExecutor executor(f.slots);
        auto refused = f.plan;
        refused.refused = true;
        const auto no_work = executor.submit(f.lease, refused, f.error);
        check(!no_work && no_work.request == 17 && no_work.incarnation == 3 && f.slots.slots()[0].preparing,
              "a refused resolved plan returns its identity without changing a reservation");
        const auto work = executor.submit(f.lease, f.plan, f.error);
        check(!executor.submit(f.lease, f.plan, f.error), "duplicate live submission cannot reset the observation cursor");
        auto stale = work;
        ++stale.generation;
        check(!executor.observe(stale, f.error) && !executor.cancel(stale, ResidentTerminal::cancelled, f.error) &&
              !f.slots.slots()[0].stop, "a stale work generation is refused before mutation even with the right request/incarnation");
        ++f.a.state().allocation_id;
        check(!executor.cancel(work, ResidentTerminal::cancelled, f.error) && !f.slots.slots()[0].stop &&
              f.peer_untouched(), "physical allocation generation fencing precedes cancellation");
        --f.a.state().allocation_id;
        check(bool(executor.cancel(work, ResidentTerminal::cancelled, f.error)), "allocation-fence refusal did not consume the current cursor");
    }
    {
        ResidentExecutorFixture f;
        ResidentExecutor executor(f.slots);
        const auto work = executor.submit(f.lease, f.plan, f.error);
        check(f.slots.prefilled(f.lease, 2, f.error), "the final-row case reaches its prompt endpoint");
        f.final_witness();
        const auto pending = executor.cancel(work, ResidentTerminal::cancelled, f.error);
        check(!pending && !pending.fresh && f.slots.slots()[0].preparing && f.slots.slots()[0].stop &&
              f.error.find("first-token draft catchup") != std::string::npos && f.peer_untouched(),
              "an issued final row refuses premature terminal/abort until mandatory catch-up completes");
        check(f.slots.first_token(f.lease, 31, nullptr, f.error), "cancellation does not replace the mandatory real first-token catch-up");
        const auto drained = executor.drain(work, f.error);
        check(drained && drained.event == ResidentEvent::idle && !drained.fresh && drained.committed == 1,
              "catch-up then drain does not discard its committed first output");
        const auto output = executor.observe(work, f.error);
        const auto terminal = executor.observe(work, f.error);
        check(output.event == ResidentEvent::committed_token && output.token == 31 && output.fresh &&
              terminal.event == ResidentEvent::terminal && terminal.fresh && terminal.committed == 1 &&
              terminal.reason == ResidentTerminal::cancelled && !executor.drain(work, f.error).fresh &&
              !executor.observe(work, f.error).fresh,
              "final-row cancellation exposes mandatory committed output before exactly one cancelled terminal");
        check(!executor.borrow_snapshot(work, f.error) && f.error.empty(), "a stopped owner cannot supply a snapshot");
    }
    {
        ResidentExecutorFixture f;
        ResidentExecutor executor(f.slots);
        const auto work = executor.submit(f.lease, f.plan, f.error);
        const strata::core::ExpertOperation peer_operation{23, 1, 51, 9, 1};
        check(f.dispatch.completion[0].begin(peer_operation, 0), "a peer retains its existing expert-operation lease");
        check(!executor.cancel(work, ResidentTerminal::cancelled, f.error) &&
              f.dispatch.completion[0].active() && f.peer_untouched(),
              "drain refuses rather than cancel or steal a peer's live operation");
        check(f.dispatch.completion[0].release(peer_operation) && executor.drain(work, f.error).fresh &&
              f.peer_untouched(), "the operation owner alone drains its witness before executor retirement");
    }
    {
        ResidentExecutorFixture f;
        ResidentExecutor executor(f.slots);
        const auto work = executor.submit(f.lease, f.plan, f.error);
        check(f.first() && executor.observe(work, f.error).event == ResidentEvent::committed_token &&
              executor.drain(work, f.error).fresh, "the stale-snapshot case completes its real resident owner");
        auto snapshot = executor.borrow_snapshot(work, f.error);
        check(bool(snapshot), "a certified source is borrowed before replacement");
        f.reserve(18, 4); // external replacement violates the synchronous borrow: return must not touch it
        const auto stale = executor.return_snapshot(snapshot, f.error);
        check(!stale && stale.work == work && snapshot && f.slots.slots()[0].request == 18 &&
              f.slots.slots()[0].incarnation == 4 && !f.slots.slots()[0].stop && f.peer_untouched(),
              "return of a stale incarnation is refused without applying it to the successor");
    }
    std::printf("resident executor contract: failures=%d executor_bytes=%zu snapshot_lease_bytes=%zu\n",
                failures, sizeof(ResidentExecutor), sizeof(ResidentSnapshotLease));
    return failures ? 1 : 0;
}
