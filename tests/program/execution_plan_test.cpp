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
        check(p.lookup.support == ep::Support::off && has(p.lookup.reason, "not consumed"),
              "lookup is not advertised where no serving decode path consumes it");
        check(p.concurrent_reader.support == ep::Support::off && has(p.concurrent_reader.reason, "batched MTP"),
              "the concurrent reader names batched MTP as its first conflict");
        check(p.budget.support == ep::Support::off, "no budget was requested");
        const std::string json = p.to_json();
        check(has(json, "\"mode\":\"resident\"") && has(json, "\"proposal\":\"batched_pipeline\""),
              "the JSON report carries the resolved mode and proposal");
        check(has(json, "\"suspension\":{\"mode\":\"resend\""), "the JSON report carries the suspension mode");
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
        check(p.groups_requested == 4 && p.groups == 1 && !p.notes.empty(),
              "a group count that does not divide the slots resolves to one group with a note");
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
        ep::ExecutionPlanInput in;
        in.spec_requested = 4;
        in.suffix_draft = 12;
        const ep::ExecutionPlan p = ep::resolve_execution_plan(in);
        check(!p.refused && p.lookup.support == ep::Support::on && p.suspension.support == ep::Support::off,
              "serial generation consumes lookup and has no suspension");
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
    std::printf("%s\n", g_fail == 0 ? "execution_plan_test: all passed" : "execution_plan_test: FAILED");
    return g_fail == 0 ? 0 : 1;
}
