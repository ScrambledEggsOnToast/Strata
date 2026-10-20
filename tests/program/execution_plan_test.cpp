// tests/program/execution_plan_test.cpp - the resolved execution contract (header-only, CPU).
//
//   1. a valid selected resident plan reports requested and effective speculation, groups, suspension and lookup;
//   2. an unsupported composition is refused before allocation, with the reason kept;
//   3. the foreground budget's prerequisites are a resolved decision, not a second check;
//   4. a nondividing group count resolves to one group with a note;
//   5. packed mode keeps its seat (in-place suspension); legacy batch resends; single-request serving has neither;
//   6. the concurrent reader names the first option-level reason it is off;
//   7. the JSON report is a single machine-readable line carrying those values.
#include "strata/program/execution_plan.hpp"
#include "strata/program/native_admission.hpp"
#include <array>

#include <cstdio>
#include <string>

namespace ep = strata::program;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-78s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
bool has(const std::string& s, const char* part) { return s.find(part) != std::string::npos; }

std::string report(const ep::ExecutionPlan& p, int expected_status) {
    FILE* out = std::tmpfile();
    check(out != nullptr, "the no-load report capture opens");
    if (!out) return {};
    check(ep::report_execution_plan(out, p) == expected_status, "report-plan exit status matches the effective contract");
    std::rewind(out);
    std::string text;
    char buf[1024];
    while (const size_t n = std::fread(buf, 1, sizeof(buf), out)) text.append(buf, n);
    std::fclose(out);
    check(text == "EPLAN " + p.to_json() + "\n", "report-plan emits exactly one tagged JSON line");
    return text;
}

ep::batch_groups::Input group_input(const ep::ExecutionPlanInput& in) {
    ep::batch_groups::Input bin;
    bin.set = in.groups_set;
    bin.auto_given = in.groups_auto;
    bin.given = in.groups_given;
    bin.batch = ep::batch_groups::normalized_slots(in.batch_requested, 8);
    bin.later_stages = in.later_stages;
    bin.batch_mtp = in.batch_mtp;
    bin.adapt_async = in.adapt_async;
    bin.mtp_serial = in.mtp_serial;
    return bin;
}
ep::ExecutionPlanInput resident_batch() {
    ep::ExecutionPlanInput in;
    in.serve = true;
    in.batch_requested = 4;
    in.groups_set = true;
    in.groups_auto = true;
    in.later_stages = 2;
    in.batch_mtp = true;
    in.batch_mtp_pipeline = true;
    in.mtp_configured = true;
    in.spec_requested = 4;
    in.suffix_draft = 12;
    in.lookup_chain = 2;
    in.piped_concurrent = true;
    return in;
}
}  // namespace

int main() {
    std::printf("execution_plan_test\n");
    {
        const ep::ExecutionPlan p = ep::resolve_execution_plan(resident_batch());
        check(!p.refused, "a valid resident batch plan is not refused");
        check(p.mode == ep::ServeMode::resident && p.stages == 3, "the selected mode and stage ownership are reported");
        check(p.batch_slots == 4 && !p.batch_slots_resolved, "the requested slots are reported, with fit still conditional");
        check(p.groups == 2 && p.groups_requested == 2 && p.groups_conditional, "the pipelined group shape is reported");
        check(p.spec_requested == 4 && p.spec == 8 && p.mtp_max_t == 4,
              "suffix and chaining resolve the effective speculative widths once");
        check(p.proposal == ep::ProposalShape::batched_pipeline, "the proposal shape is the pipelined batch MTP");
        check(p.suspension.support == ep::Support::conditional, "legacy batch suspension is a resend, not in place");
        check(p.lookup.support == ep::Support::conditional && has(p.lookup.reason, "solo serial") &&
              has(p.lookup.reason, "resident decode"),
              "lookup is conditional: solo serial serving can execute it, resident decode cannot");
        check(p.concurrent_reader.support == ep::Support::conditional && p.concurrent_reader.reason.empty(),
              "batched MTP composes with the opt-in reader, subject to runtime source checks");
        check(p.budget.support == ep::Support::off, "no budget was requested");
        const std::string json = p.to_json();
        check(has(json, "\"mode\":\"resident\"") && has(json, "\"proposal\":\"batched_pipeline\""),
              "the JSON report carries the resolved mode and proposal");
        check(has(json, "\"suspension\":{\"mode\":\"resend\""), "the JSON report carries the suspension mode");
    }
    {
        for (bool mtp : {false, true}) {
            auto in = resident_batch();
            in.batch_mtp = in.batch_mtp_pipeline = mtp;
            in.foreground_budget_ms = 1000;
            in.prompt_cache = true;
            const auto p = ep::resolve_execution_plan(in);
            check(!p.refused && p.budget.support == ep::Support::on &&
                  p.concurrent_reader.support == ep::Support::conditional && p.concurrent_reader.reason.empty(),
                  "the foreground budget composes with the reader, with and without batched MTP");
            check(has(p.to_json(), "\"concurrent_reader\":{\"support\":\"conditional\",\"reason\":\"\"}"),
                  "the report retains the reader's conditional support and empty option refusal");
        }
        using Input = ep::ExecutionPlanInput;
        const std::pair<bool Input::*, const char*> refusals[] = {
            {&Input::retained_prefill, "retained prefill"}, {&Input::resident_multiplex, "--resident-multiplex"},
            {&Input::adapt_async, "--adapt-async"},
            {&Input::spec_split, "--spec-split"}, {&Input::prefill_pipe, "STRATA_PREFILL_PIPE"}
        };
        for (const auto& [field, reason] : refusals) {
            auto in = resident_batch();
            in.*field = true;
            const auto p = ep::resolve_execution_plan(in);
            check(p.concurrent_reader.support == ep::Support::off && p.concurrent_reader.reason == reason,
                  "unrelated reader exclusions keep their original precise reason");
        }
        auto in = resident_batch();
        in.piped_concurrent = false;
        check(ep::resolve_execution_plan(in).concurrent_reader.support == ep::Support::off,
              "unsetting the opt-in keeps the sequential path selected");
    }
    {
        ep::ExecutionPlan p = ep::resolve_execution_plan(resident_batch());
        p.memory_classes.push_back({"CUDA0", "expert_cache", "profile", 123456, false});
        p.memory_classes.push_back({"host", "physical_host_snapshot", "unmeasured", 0, true});
        const std::string json = p.to_json();
        check(has(json, "\"memory_classes\":[{\"scope\":\"CUDA0\",\"name\":\"expert_cache\""),
              "the report carries the priced device classes");
        check(has(json, "\"bytes\":123456") && has(json, "\"unknown\":true"),
              "unknown classes keep their name and no invented byte estimate");
    }
    {
        ep::ExecutionPlanInput in;
        in.serve = true;
        in.batch_requested = 2;
        in.batch_mtp = true;  // no --mtp
        const ep::ExecutionPlan p = ep::resolve_execution_plan(in);
        check(p.refused && has(p.refusal, "--batch-mtp needs --mtp"),
              "an unsupported composition is refused with its reason");
    }
    {
        ep::ExecutionPlanInput in;
        in.serve = true;
        in.batch_requested = 2;
        in.prompt_cache = true;
        in.foreground_budget_ms = 6000;
        in.background_share = 0.25;
        const ep::ExecutionPlan p = ep::resolve_execution_plan(in);
        check(!p.refused && p.budget.support == ep::Support::on, "a budget with its prerequisites resolves on");
        in.batch_requested = 1;
        const ep::ExecutionPlan q = ep::resolve_execution_plan(in);
        check(q.refused && has(q.refusal, "--batch >= 2"), "a budget without two slots is refused");
        in.batch_requested = 2;
        in.background_share = 1.0;
        const ep::ExecutionPlan r = ep::resolve_execution_plan(in);
        check(r.refused && has(r.refusal, "share"), "an out-of-range share is refused");
    }
    {
        ep::ExecutionPlanInput in;
        in.serve = true;
        in.batch_requested = 6;
        in.groups_set = true;
        in.groups_given = 4;
        in.later_stages = 2;
        const ep::ExecutionPlan p = ep::resolve_execution_plan(in);
        check(p.groups_requested == 4 && p.groups == 1 && p.notes.size() == 1 &&
              p.notes.front() == "--batch-groups requested=4 -> effective=1; slots=6: 1 group of 6 slots (no group pipeline; requested groups do not divide slots)",
              "a non-dividing group count reports the exact requested-to-effective schedule");
        const auto text = report(p, 0);
        check(has(text, "\"groups_requested\":4") && has(text, "\"groups\":1") &&
              has(text, "\"resolved\":false") && has(text, "\"groups_conditional\":true") &&
              has(text, "no group pipeline"),
              "no-load report-plan keeps requested groups visible and fit explicitly conditional");
        check(p.batch_group_info() ==
              " batch_slots=6 batch_groups=1 batch_groups_requested=4 batch_groups_requested_auto=0",
              "INFO uses the same raw request and effective count as EPLAN without needing MTP");
        in.batch_mtp = in.mtp_configured = true;
        in.spec_requested = 4;
        const auto mtp = ep::resolve_execution_plan(in);
        check(!mtp.refused && mtp.groups == ep::batch_groups::resolve(group_input(in)).groups &&
              mtp.proposal == ep::ProposalShape::batched_single_draft,
              "report-plan and runtime MTP gate both use the effective serial fallback, not raw four");
        report(mtp, 0);
        in.groups_given = 3;
        const auto dividing_mtp = ep::resolve_execution_plan(in);
        check(dividing_mtp.refused && ep::batch_groups::resolve(group_input(in)).groups == 3,
              "a genuinely dividing pipeline still refuses synchronous MTP");
        report(dividing_mtp, 1);
    }
    {
        auto in = resident_batch();
        in.batch_mtp = in.batch_mtp_pipeline = false;
        in.groups_auto = false;
        in.groups_given = 4;
        in.batch_requested = 8;
        auto p = ep::resolve_execution_plan(in);
        auto bin = group_input(in);
        const auto before = ep::batch_groups::resolve(bin);
        p.memory_classes.push_back({"CUDA0", "expert_cache", "admission", 1234, false});
        bin.batch = 6;
        const auto fitted = ep::batch_groups::resolve_after_fit(bin, before);
        ep::resolve_batch_fit(p, bin.batch, bin.later_stages, fitted);
        check(p.batch_slots_requested == 8 && p.batch_slots == 6 && p.batch_slots_resolved &&
              p.groups_requested == 4 && p.groups == 1 && !p.groups_conditional,
              "post-fit plan retains the request and finalizes actual slots and groups");
        check(p.notes.size() == 1 && has(p.notes.front(), "after slot fit") &&
              has(p.notes.front(), "requested=4 -> effective=1; slots=6"),
              "post-fit fallback records one explicit actual schedule note");
        check(p.memory_classes.size() == 1 && p.memory_classes.front().bytes == 1234,
              "finalizing fit preserves the admitted memory classes");
        check(p.batch_group_info() ==
              " batch_slots=6 batch_groups=1 batch_groups_requested=4 batch_groups_requested_auto=0",
              "post-fit INFO cannot print the post-clamp value as the request");
        const auto text = report(p, 0);
        check(has(text, "\"batch\":{\"requested\":8,\"slots\":6,\"resolved\":true") &&
              has(text, "\"groups_requested\":4") && has(text, "\"groups\":1") &&
              has(text, "\"groups_conditional\":false"),
              "post-fit EPLAN and INFO agree on effective slots/groups without stale conditional fields");
        bin.batch = 4;
        auto control = ep::resolve_execution_plan(in);
        ep::resolve_batch_fit(control, 4, 2, ep::batch_groups::resolve_after_fit(bin, before));
        check(control.groups == 4 && control.batch_slots / control.groups == 1 && control.notes.empty(),
              "dividing fit preserves the same four-group/one-slot schedule with no fallback note");
    }
    {
        auto in = resident_batch();
        in.batch_mtp = in.batch_mtp_pipeline = false;
        in.batch_requested = 8;
        auto p = ep::resolve_execution_plan(in);
        auto bin = group_input(in);
        const auto before = ep::batch_groups::resolve(bin);
        bin.batch = 6;
        ep::resolve_batch_fit(p, 6, 2, ep::batch_groups::resolve_after_fit(bin, before));
        check(p.groups_requested_auto && p.groups_requested == 2 && p.groups == 3 &&
              p.batch_slots % p.groups == 0 && p.notes.empty(),
              "auto fit keeps its initial attribution and reselects a divisor without a fallback");
        check(has(p.batch_group_info(), "batch_groups=3 batch_groups_requested=2 batch_groups_requested_auto=1") &&
              has(p.to_json(), "\"groups_requested_auto\":true,\"groups\":3"),
              "auto INFO/EPLAN distinguish initial and fitted divisor counts");
    }
    {
        auto in = resident_batch();
        in.batch_mtp = in.batch_mtp_pipeline = false;
        in.groups_auto = false;
        in.groups_given = 2;
        check(ep::report_plan_later_stages("20,39", "", 1) == 2 &&
              ep::report_plan_later_stages("auto", "", 3) == 2,
              "report-plan derives explicit and auto requested split stages without model allocation");
        in.later_stages = ep::report_plan_later_stages("20", "0", 3);
        in.same_device_split = ep::report_plan_same_device_split("20", "0");
        const auto p = ep::resolve_execution_plan(in);
        check(in.later_stages == 0 && p.stages == 1 && p.groups == 1 && p.groups_requested == 2 &&
              p.notes.size() == 1 && has(p.notes.front(), "no layer split"),
              "report-plan same-device split reports the effective one-group schedule");
        report(p, 0);
        in.groups_given = 1;
        in.resident_multiplex = true;
        const auto multiplex = ep::resolve_execution_plan(in);
        check(multiplex.refused && has(multiplex.refusal, "without a layer split"),
              "report-plan refuses resident multiplex on a same-device split even with one effective group");
        report(multiplex, 1);
        in.resident_multiplex = false;
        in.batch_mtp = in.mtp_configured = true;
        in.spec_requested = 4;
        const auto synchronous_mtp = ep::resolve_execution_plan(in);
        check(synchronous_mtp.refused && has(synchronous_mtp.refusal, "each layer-split stage on its own GPU"),
              "report-plan refuses synchronous MTP on a same-device split despite one effective group");
        report(synchronous_mtp, 1);
        in.batch_mtp = false;
        in.groups_given = 2;
        in.same_device_split = ep::report_plan_same_device_split("", "");
        in.later_stages = ep::report_plan_later_stages("", "", 3);
        const auto no_split = ep::resolve_execution_plan(in);
        check(no_split.groups == 1 && no_split.groups_requested == 2 &&
              has(no_split.to_json(), "no layer split"),
              "no-layer-split report retains the raw explicit count and fallback reason");
        report(no_split, 0);
        auto bin = group_input(in);
        auto finalized = no_split;
        ep::resolve_batch_fit(finalized, 4, 0,
            ep::batch_groups::resolve_after_fit(bin, ep::batch_groups::resolve(bin)));
        check(finalized.notes.size() == 1 && finalized.batch_slots_resolved && !finalized.groups_conditional,
              "finalization replaces the pre-fit fallback note rather than duplicating it");
    }
    {
        auto in = resident_batch();
        in.batch_mtp = in.batch_mtp_pipeline = false;
        in.batch_requested = 8;
        auto p = ep::resolve_execution_plan(in);
        auto bin = group_input(in);
        const auto before = ep::batch_groups::resolve(bin);
        bin.batch = 0;
        ep::resolve_batch_fit(p, 0, 2, ep::batch_groups::resolve_after_fit(bin, before));
        check(p.mode == ep::ServeMode::streaming && p.batch_slots_requested == 8 && p.batch_slots == 0 &&
              p.batch_slots_resolved && p.groups == 1 && p.groups_requested == 2 &&
              p.groups_requested_auto && !p.groups_conditional,
              "zero fitted slots finalizes streaming mode while retaining initial slot/group requests");
        check(has(p.batch_group_info(), "batch_slots=0 batch_groups=1 batch_groups_requested=2") &&
              has(p.to_json(), "\"batch\":{\"requested\":8,\"slots\":0,\"resolved\":true") &&
              has(p.to_json(), "\"groups_conditional\":false"),
              "zero-fit INFO/EPLAN helpers have no stale effective slot count or conditional group flag");
        report(p, 0);
    }
    for (const bool mtp_serial : {false, true}) {
        auto in = resident_batch();
        in.groups_set = in.groups_auto = false;
        in.batch_mtp_pipeline = false;
        in.batch_mtp = in.mtp_serial = mtp_serial;
        in.adapt_async = !mtp_serial;
        in.batch_requested = 8;
        auto p = ep::resolve_execution_plan(in);
        auto bin = group_input(in);
        const auto before = ep::batch_groups::resolve(bin);
        bin.batch = 6;
        ep::resolve_batch_fit(p, 6, 2, ep::batch_groups::resolve_after_fit(bin, before));
        check(!p.refused && p.groups == 1 && p.groups_requested == 2 && p.groups_requested_auto &&
              p.batch_slots == 6 && p.batch_slots_resolved && !p.groups_conditional && p.notes.empty(),
              "post-fit serial MTP/adapt-async plans keep their serial override and initial auto attribution");
        check(has(p.batch_group_info(), "batch_groups=1 batch_groups_requested=2 batch_groups_requested_auto=1"),
              "post-fit serial override INFO retains the requested auto count rather than the clamped count");
        report(p, 0);
    }
    {
        ep::ExecutionPlanInput in;
        in.serve = true;
        in.batch_requested = 2;
        in.packed_prefill = true;
        in.batch_mtp = true;
        in.batch_mtp_pipeline = true;
        in.mtp_configured = true;
        in.spec_requested = 4;
        in.groups_set = true;
        in.groups_given = 2;
        in.later_stages = 2;
        in.piped_concurrent = true;
        const ep::ExecutionPlan p = ep::resolve_execution_plan(in);
        check(!p.refused && p.suspension.support == ep::Support::on && has(p.suspension.reason, "destination"),
              "packed preparation keeps its seat and frontier in place");
        check(p.concurrent_reader.support == ep::Support::off && has(p.concurrent_reader.reason, "packed prefill"),
              "packed prefill is the first concurrent-reader conflict");
    }
    {
        auto in = resident_batch();
        in.retained_prompt = true;
        const auto p = ep::resolve_execution_plan(in);
        check(!p.refused && !in.packed_prefill && !in.retained_prefill && p.suspension.support == ep::Support::conditional &&
              has(p.suspension.reason, "without resending") && has(p.suspension.reason, "pending stage allocation"),
              "serial retained prompt reports allocation-conditional capability without packing or layer yields");
        check(p.admission.support == ep::Support::conditional &&
              has(p.to_json(), "\"admission\":{\"authority\":\"native\",\"support\":\"conditional\""),
              "native admission is conditional until retained allocations and residency validate");
        auto resolved = p;
        resolved.admission.support = ep::Support::on;
        check(has(resolved.to_json(), "\"admission\":{\"authority\":\"native\",\"support\":\"on\""),
              "resolved native admission advertises the exact cohort capability");
        check(p.concurrent_reader.support == ep::Support::off && has(p.concurrent_reader.reason, "retained prompt"),
              "retained prompt keeps the independent reader disabled");
        in.serve = false;
        check(ep::resolve_execution_plan(in).refused, "non-serving retained prompts refuse in the no-load resolver");
        in.serve = true; in.mtp_configured = false;
        check(ep::resolve_execution_plan(in).refused, "missing retained draft readiness refuses before allocation");
        in.mtp_configured = true; in.prefill_chunk = 1024;
        check(ep::resolve_execution_plan(in).refused, "unqualified retained scratch geometry refuses before allocation");
        in = {};
        in.serve = in.retained_prompt = in.batch_mtp = in.mtp_configured = in.prompt_cache = true;
        in.batch_requested = 1; in.spec_requested = 4; in.foreground_budget_ms = 1000;
        const auto single = ep::resolve_execution_plan(in);
        check(!single.refused && single.suspension_mode == ep::Suspension::in_place &&
              single.budget.support == ep::Support::on,
              "one private retained owner consistently satisfies both MTP and foreground-budget minimums");
    }
    {
        ep::ExecutionPlanInput pair;
        pair.serve = true;
        pair.later_stages = 1;
        pair.prefill_chunk = 1024;
        const auto serial = ep::resolve_execution_plan(pair);
        check(!serial.refused && serial.stages == 2 && serial.batch_slots == 0 &&
              serial.proposal == ep::ProposalShape::none,
              "the two-V100 one-generation-owner shape composes serially, without invented peers or MTP slots");
    }
    {
        ep::ExecutionPlanInput in;
        in.spec_requested = 4;
        in.suffix_draft = 12;
        const ep::ExecutionPlan p = ep::resolve_execution_plan(in);
        check(!p.refused && p.lookup.support == ep::Support::on && p.suspension.support == ep::Support::off,
              "serial generation consumes lookup and has no suspension");
        in.serve = true;
        const auto serial = ep::resolve_execution_plan(in);
        check(serial.mode == ep::ServeMode::streaming && serial.batch_slots == 0 &&
              serial.lookup.support == ep::Support::on && serial.admission.support == ep::Support::off,
              "explicit zero-seat serving is streaming and consumes serial lookup without native admission");
        check(has(serial.to_json(), "\"mode\":\"streaming\""), "serial-serving JSON reports the working-owner path");
        in.spec_requested = 1;
        const ep::ExecutionPlan q = ep::resolve_execution_plan(in);
        check(q.lookup.support == ep::Support::off && has(q.lookup.reason, "spec"),
              "a serving lookup without the speculative width is off with its reason");
    }
    {
        ep::ExecutionPlanInput in;
        in.serve = true;
        in.resident_multiplex = true;
        in.batch_requested = 2;
        in.later_stages = 1;
        const ep::ExecutionPlan p = ep::resolve_execution_plan(in);
        check(p.refused && has(p.refusal, "single device"), "resident multiplexing is refused on a layer split");
    }
    {
        ep::ExecutionPlanInput in;
        in.serve = true;
        in.batch_requested = 4;
        in.batch_mtp = true;
        in.batch_mtp_floating = true;
        in.batch_mtp_pipeline = true;
        in.mtp_configured = true;
        in.spec_requested = 4;
        in.groups_set = true;
        in.groups_given = 2;
        in.later_stages = 1;
        const ep::ExecutionPlan p = ep::resolve_execution_plan(in);
        check(!p.refused && p.proposal == ep::ProposalShape::batched_floating, "floating owners are the selected shape");
        in.batch_mtp_third = true;
        const ep::ExecutionPlan q = ep::resolve_execution_plan(in);
        check(!q.refused && q.proposal == ep::ProposalShape::batched_third, "the third owner is the selected shape");
        in.batch_requested = 3;
        const ep::ExecutionPlan r = ep::resolve_execution_plan(in);
        check(r.refused, "the third owner without four slots is refused");
    }
    {
        int hint = -2;
        check(ep::native_control_slot("-1", hint) && hint == -1,
              "an unseated logical control carries no physical-seat claim");
        check(ep::native_control_slot("7", hint) && hint == 7,
              "a stale well-formed seat hint is fenced by logical lease matching, not parser fatal");
        check(!ep::native_control_slot("-2", hint) && !ep::native_control_slot("seat", hint) &&
              !ep::native_control_slot("2147483648", hint),
              "malformed or overflowing seat hints remain protocol refusals");
        ep::NativeAdmissionPolicy policy;
        std::array<ep::AdmissionCandidate, 3> waiting{{
            {0, 2, true, false, true}, {1, 1, true, false, true}, {2, 3, false, false, true}}};
        check(policy.select(waiting)->index == 1, "native admission is FIFO within a priority class");
        for (int i = 0; i < 3; ++i) policy.granted(true, false);
        check(policy.select(waiting)->index == 2, "background bypass occurs after three successful foreground grants");
        policy.granted(false, false);
        waiting[1].ready = false;
        check(!policy.select(waiting), "an earlier registered unready input cannot be bypassed by tokenization completion");
        waiting[1].ready = true;
        waiting[0].solo = true;
        check(policy.select(waiting)->index == 0, "foreground solo preference precedes ordinary FIFO");
        for (int i = 0; i < 4; ++i) policy.granted(true, true);
        waiting[2].foreground = true;
        check(policy.select(waiting)->index == 1, "eligible foreground bypass occurs after four solo grants");
        policy = {};
        policy.short_first = 8;
        waiting[0] = {0, 1, true, false, true, 100, 0};
        waiting[1] = {1, 2, true, false, true, 4, 0};
        waiting[2] = {2, 3, true, false, true, 100, 0};
        check(policy.select(waiting)->index == 1, "optional short-first orders only the chosen admission class");
        waiting[0].wait_s = 30;
        check(policy.select(waiting)->index == 0, "aged native input retains its FIFO admission opportunity");
        check(ep::resolve_execution_plan(resident_batch()).admission.support == ep::Support::off,
              "legacy resident and daily plans do not claim native admission");
    }
    std::printf("%s\n", g_fail == 0 ? "execution_plan_test: all passed" : "execution_plan_test: FAILED");
    return g_fail == 0 ? 0 : 1;
}
