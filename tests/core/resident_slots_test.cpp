// CPU-only lifecycle preflight. Real graph/isolation checks run through the protected model vehicle.
#include "strata/program/resident_slots.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/graph.hpp"
#include "strata/core/mtp.hpp"
#include "strata/core/verify.hpp"
#include "strata/core/resident_memory.hpp"
#include "strata/prefill/prefill.hpp"

#include <cstdio>
#include <type_traits>

namespace strata::program {
struct ResidentDraftReadinessFixture {
    template<class Check> static void run(Check&& check) {
        core::ModelGeometry geometry;
        geometry.n_layers = 0; // no-load target with no model-required state families
        core::SessionOwner main, a, b;
        core::Verifier verifier;
        core::MtpDrafter draft;
        prefill::Prefill prompt;
        core::ExpertDispatch dispatch;
        dispatch.completion.emplace_back(core::ExpertCompletion::kCapacity);
        const core::ExpertOperation peer_operation{23, 1, 51, 9, 1};
        check(dispatch.completion[0].begin(peer_operation, 0), "a peer owns its own live operation");
        ResidentStage stage{&main.state(), &verifier, {&a.state(), &b.state()}, -1};
        ResidentSlots slots(geometry, {stage}, draft, prompt, dispatch, 64, true, 6, 4096);
        slots.resources_ = ResidentSlots::Resources::ready;
        auto& source = slots.slots_[0];
        source.request = 17; source.incarnation = 3; source.cached = source.partial = true;
        source.input.scope.value[0] = 1;
        for (int token = 100; token < 148; ++token) source.prompt.push_back(token);
        source.ids.assign(source.prompt.begin(), source.prompt.begin() + 16);
        source.p = 16;
        std::vector<int64_t> tokens(source.prompt.begin(), source.prompt.end());
        slots.working_.request = 17; slots.working_.resume_slot = 0;
        slots.input_ = source.input; slots.working_prompt_ = &tokens;
        auto& peer = slots.slots_[1];
        peer.request = 23; peer.incarnation = 9; peer.active = true;
        peer.ids = {21, 22, 23}; peer.p = 3; peer.produced = 2; peer.sampling.seed = 99;
        peer.prompt_cursor.operation = 51;
        dispatch.slot_requests[0] = 17; dispatch.slot_requests[1] = 23;
        draft.st_.max_cells = 64; draft.window_ = 40;
        draft.slots_.resize(2);
        draft.owners_[0] = {0, 17, 3, 0};
        draft.owners_[1] = {1, 23, 9, 0};
        auto& chain = draft.slots_[0];
        chain.state.max_cells = 64; chain.request = 17; chain.certified = true;
        chain.producer = draft.owners_[0];
        chain.begin = 8; chain.cells = chain.completed_cells = 16; chain.prompt_len = 48;
        chain.tail_token = 116;
        auto& peer_chain = draft.slots_[1];
        peer_chain.request = 23; peer_chain.certified = true;
        peer_chain.producer = draft.owners_[1]; peer_chain.state.max_cells = 64;
        peer_chain.begin = 0; peer_chain.cells = peer_chain.completed_cells = 3;
        peer_chain.tail_token = 24;
        std::string error;
        check(decide_draft_readiness(slots.certify_continuation(0), 48, 40, 40960).ready,
              "a certified target and a nonzero valid draft interval are ready without reconstruction");
        // #257: parked rows may extend past their independently completed draft frontier.
        chain.completed_cells = 12; chain.tail_token = 112;
        const auto parked = slots.certify_continuation(0);
        check(chain.begin < chain.completed_cells && chain.completed_cells < chain.cells &&
              parked.certified && parked.refusal == ContinuationRefusal::none &&
              parked.draft_frontier == 12 && parked.draft_end == 12 && parked.valid_end == 16 &&
              parked.draft_ready,
              "a partial parked certificate is ready at its frontier inside the parked interval");
        check(decide_draft_readiness(parked, 48, 40, 40960).ready,
              "the retained window is covered at the completed frontier, not the parked extent");
        chain.tail_token = 116;
        const auto wrong_boundary = slots.certify_continuation(0);
        check(!wrong_boundary.certified && !wrong_boundary.draft_ready &&
              wrong_boundary.refusal == ContinuationRefusal::draft_boundary,
              "the boundary token must match ids[12], not the token at the parked extent");
        chain.completed_cells = 16; chain.tail_token = 116;
        const auto unchanged = [&] {
            return source.request == 17 && source.incarnation == 3 && source.partial && !source.failed &&
                source.ids.size() == 16 && source.ids.front() == 100 && source.ids.back() == 115 &&
                source.p == 16 && peer.request == 23 && peer.incarnation == 9 && peer.active && !peer.failed &&
                peer.ids == std::vector<int32_t>({21, 22, 23}) && peer.p == 3 && peer.produced == 2 &&
                peer.sampling.seed == 99 && peer.prompt_cursor.operation == 51 &&
                dispatch.slot_requests[0] == 17 && dispatch.slot_requests[1] == 23 &&
                dispatch.completion[0].active() && peer_chain.request == 23 && peer_chain.certified &&
                peer_chain.cells == 3 && peer_chain.completed_cells == 3 && peer_chain.tail_token == 24;
        };
        const auto refused = [&] {
            PrefixWork work{123, 456};
            error.clear();
            check(!slots.resume_work(0, nullptr, work, error) && work.bytes == 0 && work.route == 0,
                  "readiness refusal happens before target/draft pricing, writes or owner mutation");
            check(unchanged(), "refusal preserves target identity/history and the peer's state/operation witness");
        };
        slots.inject_absent_draft(0);
        check(slots.certify_continuation(0).certified && !slots.certify_continuation(0).draft_ready,
              "removing optional draft readiness leaves the target cut certified");
        refused();
        const auto absent = decide_draft_readiness(slots.certify_continuation(0), 48, 40, 40960);
        check(!absent.ready && absent.refusal == ContinuationRefusal::draft_absent &&
              absent.residual_rows == 8 && absent.residual_bytes_known && absent.residual_bytes == 327680 &&
              !absent.reconstruction_priced,
              "missing draft history has bounded required inputs but no invented reconstruction quote");
        chain.certified = true; chain.producer = draft.owners_[0];
        chain.begin = 8; chain.completed_cells = 16; chain.tail_token = 116;
        chain.request = 18; chain.producer.request = 18;
        refused();
        chain.request = 17; chain.producer.request = 17; chain.completed_cells = 17;
        refused();
        chain.completed_cells = 16; chain.begin = 17;
        refused();
        chain.begin = 8; chain.tail_token = 999;
        refused();
        chain.tail_token = 116; chain.begin = 9;
        refused();
        chain.begin = 8; tokens.resize(16);
        refused();
        tokens.assign(source.prompt.begin(), source.prompt.end());
        core::ConversationCheckpoint cut;
        cut.ids.assign(source.ids.begin(), source.ids.begin() + 12);
        const auto rewind = slots.certify_continuation(0, &cut, 17, 3);
        check(rewind.certified && rewind.draft_frontier == 12 && rewind.draft_end == 16,
              "a checkpoint rewind validates at its own frontier, not the live draft endpoint");
        check(decide_draft_readiness(rewind, 48, 40, 40960).ready,
              "a checkpoint with the certified draft window and matching boundary remains selectable");
        chain.begin = 13;
        const auto before_draft = decide_draft_readiness(slots.certify_continuation(0, &cut, 17, 3), 48, 40, 40960);
        check(!before_draft.ready && before_draft.refusal == ContinuationRefusal::draft_interval && unchanged(),
              "a target checkpoint before the retained draft interval is correctly refused without mutation");
        chain.begin = 9;
        const auto missing_window = decide_draft_readiness(slots.certify_continuation(0, &cut, 17, 3), 48, 40, 40960);
        check(slots.certify_continuation(0, &cut, 17, 3).certified && !missing_window.ready &&
              missing_window.refusal == ContinuationRefusal::draft_window && unchanged(),
              "target certification alone cannot bypass missing preceding draft-window history");
        chain.begin = 8;
        source.preparing = source.stop = true; source.cached = false;
        ResidentPreparation lease;
        lease.slot = 0; lease.request = 17; lease.incarnation = 3; lease.endpoint = 47;
        lease.stages[0] = &a.state(); lease.allocations[0] = a.state().allocation_id = 1;
        slots.preparations_[0].ready = true;
        ResidentPromptCursor cursor;
        check(!slots.resume_prompt(lease, cursor, error) && source.stop && source.preparing && peer.active,
              "cancellation while preparing wins before readiness work or another operation is issued");
        source.request = 18; source.incarnation = 4;
        draft.owners_[0] = {0, 18, 4, 0};
        check(!slots.resume_prompt(lease, cursor, error) && source.request == 18 && peer.request == 23,
              "a stale readiness completion cannot claim a successor's seat incarnation");
        check(dispatch.completion[0].release(peer_operation), "the untouched peer drains its own operation");
    }
};
} // namespace strata::program

int main() {
    using namespace strata;
    static_assert(!std::is_move_constructible_v<program::ResidentSlots>);
    core::ModelGeometry geometry;
    core::SessionOwner main, a, b;
    core::Verifier verifier;
    core::MtpDrafter draft;
    prefill::Prefill prompt;
    core::ExpertDispatch dispatch;
    dispatch.completion.emplace_back(core::ExpertCompletion::kCapacity);
    program::ResidentStage stage{&main.state(), &verifier, {&a.state(), &b.state()}, -1};
    plan::ResidentCheckpointPayload checkpoint;
    plan::ResidentHostBytes host;
    if (!core::conversation_checkpoint_payload(geometry, 64, 1, nullptr, checkpoint) ||
        !plan::resident_host_bytes(64, checkpoint, 0, 0, 1, host)) return 1;
    program::ResidentSlots slots(geometry, {stage}, draft, prompt, dispatch, 64, true, 6, host.checkpoints);
    std::string error;
    int failures = 0;
    auto check = [&](bool ok, const char* message) {
        if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); ++failures; }
    };
    program::ResidentDraftReadinessFixture::run(check);
    check(slots.identity(0).request == 0 && slots.identity(0).incarnation == 0 &&
          slots.identity(-1).request == 0 && slots.identity(2).request == 0,
          "unbound and nonexistent seats expose no request identity");
    check(!slots.owns(0, 0, 0) && !slots.owns(0, 17, 2) &&
          !slots.owns(-1, 17, 2) && !slots.owns(2, 17, 2),
          "an unbound or nonexistent seat cannot own a cancellation");
    {
        // The production accessor's value comparison is pure; binding a live verifier needs a device.
        const program::ResidentIdentity first{17, 2}, continued{17, 3}, successor{18, 4};
        check(first.owns(17, 2), "a bound identity recognizes its exact request and seat incarnation");
        check(!first.owns(18, 2) && !first.owns(17, 3) && !first.owns(0, 2) && !first.owns(17, 0),
              "request, incarnation and unbound controls cannot match another owner");
        check(continued.owns(17, 3) && !continued.owns(17, 2) &&
              successor.owns(18, 4) && !successor.owns(17, 3),
              "rebind and slot reuse reject the old incarnation even when the request continues");
        check(!program::ResidentIdentity{}.owns(0, 0) && !program::ResidentIdentity{0, 4}.owns(0, 4),
              "request zero always means unbound, including after a seat has been retired");
    }
    const core::GraphCaptureProbe disabled_probe({"resident-default", "no-observer"});
    for (const auto phase : {core::GraphCapturePhase::BeforeCapture, core::GraphCapturePhase::AfterCapture,
                            core::GraphCapturePhase::AfterInstantiate, core::GraphCapturePhase::AfterUpload,
                            core::GraphCapturePhase::Retained})
        check(disabled_probe.check(phase, error),
              "an absent capture observer performs no CUDA query, including with CUDA_VISIBLE_DEVICES=-1");
    {
        program::ResidentSlot completed;
        completed.active = true;
        completed.request = 17;
        completed.input.scope.value[0] = 1;
        completed.ids = {11, 22, 33};
        completed.sampling.seed = 99;
        completed.produced = 2;
        completed.p = 3;
        completed.prefill_split_until = 2048;
        completed.complete(true); // same policy called after serial and pipelined row drains
        const core::ConversationScope other{{2}};
        check(completed.reusable(completed.input.scope, true),
              "completed pipelined turn retains a same-conversation prefix");
        check(!completed.reusable(other, true) && !completed.reusable({}, true) &&
              !completed.reusable(completed.input.scope, false),
              "cold conversation, absent scope and changed steering all miss");
        check(!completed.active && completed.request == 17 && completed.produced == 2 &&
              completed.p == 3 && completed.sampling.seed == 99 && completed.ids.size() == 3,
              "retention changes neither committed output, identity, RNG nor position");
        check(completed.prefill_split_until == 0, "completion clears any pending prefill partition endpoint");
        completed.preparing = true;
        check(!completed.reusable(completed.input.scope, true),
              "a reserved incarnation cannot expose its previous cache while destination preparation is pending");
        completed.complete(true);
        check(!completed.preparing, "drained completion clears the preparation marker");
        completed.input.checkpoint = false;
        completed.complete(true);
        check(!completed.cached, "one-shot completed turns cannot become reusable");
        completed.input.checkpoint = true;
        completed.img = true;
        completed.complete(true);
        check(!completed.cached, "image slots stay ineligible");
        completed.img = false;
        completed.failed = true;
        completed.complete(true);
        check(!completed.cached, "failed state is never reusable");
        completed.failed = false;
        completed.complete(false);
        check(!completed.cached, "prompt-cache zero still disables retention");
        completed.tainted = true;                   // #879: this run met non-finite logits
        completed.complete(true);
        check(!completed.cached && !completed.reusable(completed.input.scope, true),
              "a tainted completion is neither cached nor reusable");
    }
    {
        // `tainted` belongs to the identity that ran, never to the seat (#879): a fresh (non-continuing)
        // admission retires the old identity and its stale taint together - bind clears the flag where it
        // retires, and rebuilds the seat's cells - so the fresh identity completes reusable. A continuing
        // rebind is the same identity and run: its taint stays, and no completion of it becomes reusable.
        program::ResidentSlot seat;
        seat.request = 23;                          // the poisoned run's own identity
        seat.input.scope.value[0] = 1;
        seat.input.checkpoint = true;
        seat.ids = {11, 22};
        seat.tainted = true;
        seat.complete(true);
        check(!seat.cached && !seat.reusable(seat.input.scope, true),
              "a tainted completion is not cached, so the poisoned prefix is never handed on");
        seat.tainted = false;                       // retired with the old identity by the fresh admission's bind
        seat.request = 29;                          // the identity the fresh (non-continuing) bind publishes
        seat.complete(true);
        check(seat.cached && seat.reusable(seat.input.scope, true),
              "a fresh non-continuing admission rebuilds the seat and completes reusable");
        seat.tainted = true;                        // kept by a continuing rebind of the same identity (29)
        seat.complete(true);
        check(!seat.cached && !seat.reusable(seat.input.scope, true),
              "a continuing tainted rebind stays non-reusable");
    }
    {
        program::ResidentSlot parked;
        parked.request = 17;
        parked.partial = true;
        parked.prefill_split_until = 2048;
        program::ResidentWorking other;
        other.request = 18;
        other.resume_slot = 0;
        parked.restored(other, 0, true);
        check(other.prefill_split_until == 0 && parked.prefill_split_until == 2048 && parked.partial,
              "prefix reuse by a new identity cannot claim or erase the parked request's pending half");
        program::ResidentWorking continued;
        continued.request = 17;
        continued.resume_slot = 1;
        parked.restored(continued, 0, true);
        check(continued.prefill_split_until == 0 && parked.prefill_split_until == 2048 && parked.partial,
              "the same identity cannot take a different physical slot's pending half");
        continued.resume_slot = 0;
        parked.restored(continued, 0, true);
        check(continued.prefill_split_until == 2048 && parked.prefill_split_until == 0 && !parked.partial,
              "full exact continuation transfers the original endpoint once");
        parked.partial = true;
        parked.prefill_split_until = 2048;
        parked.restored(continued, 0, false);
        check(continued.prefill_split_until == 0 && parked.prefill_split_until == 0 && !parked.partial,
              "checkpoint rewind discards the old live-frontier interval");
    }
    {
        program::ResidentPromptCursor cursor;
        cursor.plan_id = 7; cursor.operation = 11;
        cursor.plan.from = 321; cursor.plan.to = 10321; cursor.plan.chunk = 2048;
        cursor.plan.pipe = 1; cursor.plan.chunked = true;
        cursor.next = 321;
        std::vector<std::pair<int64_t, int64_t>> intervals;
        while (cursor.next < cursor.plan.to) {
            check(cursor.valid(cursor.next, 11000), "retained interval stays on the original nonzero plan grid");
            const auto saved = cursor;
            cursor.suspended = true;
            check(cursor.matches(saved), "suspension changes no arithmetic plan, cursor or operation witness");
            auto stale = saved;
            for (int field = 0; field < 5; ++field) {
                stale = saved;
                if (field == 0) ++stale.operation;
                if (field == 1) ++stale.plan_id;
                if (field == 2) ++stale.next;
                if (field == 3) ++stale.plan.from;
                if (field == 4) stale.plan.chunked = !stale.plan.chunked;
                check(!cursor.matches(stale), "stale operation/frontier/plan/shape cannot resume the retained cursor");
            }
            const int64_t end = cursor.interval_end();
            intervals.emplace_back(cursor.next, end);
            cursor.next = end; ++cursor.operation; cursor.suspended = false;
        }
        check(intervals.size() == 5 && intervals.front().first == 321 &&
              intervals.back() == std::pair<int64_t, int64_t>{8513, 10321},
              "resuming retains efficient 2048 chunks and exactly the original 1808-token tail");
        cursor.next = 1345; cursor.split_until = 2369;
        check(cursor.valid(1345, 11000) && cursor.interval_end() == 2369,
              "a retained midpoint completes the original pending split endpoint before any tail");
        auto wrong_half = cursor; ++wrong_half.split_until;
        check(!cursor.matches(wrong_half) && !wrong_half.valid(1345, 11000),
              "pending split identity cannot drift to an unrelated interval");
        const auto peer = cursor;
        cursor.next = 2369; cursor.split_until = 0;
        check(peer.next == 1345 && peer.split_until == 2369,
              "retained witnesses have no borrowed mutable storage or peer cursor alias");
    }
    program::ResidentRequest oversized;
    oversized.max_new = 55; // two prompt cells + 55 output + the unchanged eight-cell margin > 64
    check(!slots.begin(-1, {11, 22}, oversized, error) && error.find("exceeds the context") != std::string::npos,
          "full output budget refuses before binding working consumers");
    oversized.max_new = INT64_MAX;
    check(!slots.begin(-1, {11, 22}, oversized, error) && error.find("overflowing") != std::string::npos,
          "overflowing output budget refuses before state pricing or binding");
    check(slots.working().request == 0 && dispatch.request_generation == 0 &&
          slots.slots()[0].request == 0 && slots.slots()[1].request == 0,
          "budget refusal publishes no identity and changes no resident owner");
    const core::ExpertOperation operation{71, 0, 0, 1, 1};
    check(dispatch.completion[0].begin(operation, 0), "operation lease acquired");
    program::ResidentRequest config;
    config.max_new = 8;
    check(!slots.begin(-1, {11, 22}, config, error), "working request refuses retained consumers");
    check(slots.working().request == 0 && dispatch.request_generation == 0 &&
          dispatch.completion[0].active(), "refusal changes neither request nor operation ownership");
    check(dispatch.completion[0].release(operation), "operation lease drains independently");
    check(!slots.begin(-1, {11, 22}, config, error) && slots.working().request == 0,
          "uninitialized working verifier cannot publish an identity");
    const std::vector<int32_t> prefix{11, 22};
    {
        program::ResidentPreparation lease;
        lease.slot = 7;
        lease.request = 91;
        const std::vector<int64_t> tokens{11, 22};
        const program::ResidentPrepareInput inputs[]{{0, &tokens, config}, {0, &tokens, config}};
        check(!slots.reserve(inputs, 0, &lease, error) && lease.slot == 7 && lease.request == 91,
              "empty reservation cannot replace a caller lease");
        program::ResidentPreparation leases[2];
        check(!slots.reserve(inputs, 2, leases, error) && !slots.slots()[0].preparing &&
              slots.slots()[0].request == 0 && slots.slots()[1].request == 0,
              "unavailable/duplicate reservation publishes no partial owner or preparation");
        check(!slots.prepare(lease, error) && !slots.prefilled(lease, 1, error) &&
              !slots.first_token(lease, 33, nullptr, error) && !slots.abort_preparation(lease, error),
              "foreign preparation leases refuse every transition before GPU access");
        program::ResidentPromptCursor cursor;
        check(!slots.plan_prompt(lease, cursor, error) && !slots.suspend_prompt(lease, 1, cursor, error) &&
              !slots.resume_prompt(lease, cursor, error) && slots.slots()[1].request == 0,
              "foreign retained-owner transitions refuse before mutation and leave a peer untouched");
        check(draft.slot_cells(-1) == -1 && draft.slot_begin(0) == -1 &&
              draft.slot_prompt_len(0) == -1 && !draft.slot_covers(0, 0),
              "missing draft destination never advertises a retained interval");
        check(!draft.prepare_slot(-1, 91, 2, 0, true, error) &&
              !draft.prefill_slot(0, nullptr, nullptr, 1, 0, error),
              "invalid direct draft preparation and span refuse before device access");
    }
    program::ResidentAdmission request;
    request.next_token = 33;
    check(!slots.admit(0, prefix, request, error), "admission before resource initialization refuses");
    check(slots.slots()[0].request == 0 && !slots.slots()[0].active &&
          slots.slots()[0].ids.empty() && dispatch.slot_requests[0] == 0,
          "refused admission publishes no request identity or history");
    check(slots.identity(0).request == 0 && !slots.owns(0, 0, slots.identity(0).incarnation),
          "refused admission cannot publish an identity or match an unbound cancellation");
    check(!slots.init_resources(error), "uninitialized verifier cannot create resident resources");
    check(!slots.admit(1, prefix, request, error) && slots.slots()[1].request == 0,
          "failed resource initialization cannot expose another ready slot");
    const bool failed_before_pricing = slots.slots()[0].failed;
    program::PrefixWork restore_work{123, 456};
    check(!slots.resume_work(0, nullptr, restore_work, error) && restore_work.bytes == 0 &&
          restore_work.route == 0, "read-only pricing refuses an unpublished resident owner without stale costs");
    check(!slots.resume_work(-1, nullptr, restore_work, error),
          "read-only pricing refuses a nonexistent seat before reading state");
    check(slots.working().request == 0 && slots.slots()[0].ids.empty() &&
          slots.slots()[0].failed == failed_before_pricing,
          "pricing refusal changes no request identity, history or failure state");
    error = "stale pricing error";
    check(!slots.completed_snapshot_source(-1, error) && error.empty(),
          "missing completed source is a clean cache miss before GPU access");
    check(!slots.completed_snapshot_source(0, error) && error.empty(),
          "an unpublished seat cannot supply a conversation snapshot");
    check(slots.working().request == 0 && slots.slots()[0].ids.empty() &&
          slots.slots()[0].failed == failed_before_pricing && dispatch.request_generation == 0,
          "snapshot refusal changes neither working identity nor resident ownership");
    std::printf("resident lifecycle preflight: failures=%d\n", failures);
    return failures ? 1 : 0;
}
