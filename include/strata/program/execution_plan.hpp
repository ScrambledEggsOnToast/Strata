// One resolved execution contract for the selected serving mode. Pure: no GPU, model or environment read.
//
// The engine's flags describe a request; several later decisions - speculative width, slot groups, the foreground
// budget, the concurrent reader and prompt suspension - used to be re-derived at different points from overlapping
// subsets of them. This module resolves the option-level part once into requested and effective values with the
// reason a capability is off. Facts that need hardware or allocation (which slots fit, a same-GPU split, runtime
// helper caches) stay explicit inputs and are reported as conditional, never as observed capabilities.
#pragma once
#include "strata/program/batch_groups.hpp"
#include "strata/program/time_budget.hpp"

#include <algorithm>
#include <string>
#include <vector>

namespace strata::program {

enum class ServeMode { streaming, resident };
enum class ProposalShape { none, serial_single_draft, batched_single_draft, batched_pipeline, batched_third, batched_floating };
enum class Suspension { none, resend, in_place };
// off: not requested; on: resolved and available; conditional: still depends on an input not known before allocation;
// refused: the composition is unsupported (the engine must exit before allocating).
enum class Support { off, conditional, on, refused };

struct PlanDecision {
    Support support = Support::off;
    std::string reason;
};

struct ExecutionPlanInput {
    bool serve = false;
    int batch_requested = 0;
    bool groups_set = false, groups_auto = false;
    int groups_given = 1;
    int later_stages = 0;   ///< layer-split stages after the primary (each on its own GPU)
    bool batch_mtp = false, batch_mtp_pipeline = false, batch_mtp_third = false, batch_mtp_floating = false;
    bool mtp_configured = false;
    int spec_requested = 0, mtp_max_requested = 0;
    int suffix_draft = 0, lookup_chain = 0;
    bool adapt_async = false, mtp_serial = false;
    bool resident_multiplex = false, prompt_cache = false, packed_prefill = false, retained_prefill = false;
    double foreground_budget_ms = 0, background_share = 0.25;
    bool piped_concurrent = false;
    bool judge_lane = false, spec_split = false, prefill_pipe = false;
};

// One admitted memory class, as priced by the admission decision's scope. `unknown` classes carry no byte
// estimate and are never summed; `source` names where a price came from.
struct PricedClass {
    std::string scope;    ///< "CUDA0", "guest" or "host"
    std::string name;
    std::string source;
    uint64_t bytes = 0;
    bool unknown = false;
};

struct ExecutionPlan {
    bool refused = false;
    std::string refusal;             ///< the reason the engine must exit before allocating
    ServeMode mode = ServeMode::streaming;
    int stages = 1;
    int batch_slots = 0;             ///< requested logical slots (normalized to the verify window); fit is conditional
    bool batch_slots_resolved = true;
    int groups_requested = 1, groups = 1;
    bool groups_requested_auto = false; ///< `--batch-groups auto` (or the layer-split default) asked for a resolved count
    bool groups_conditional = false; ///< the slots that fit can still reduce the groups
    bool groups_from_auto = false;
    int spec_requested = 0, spec = 0, mtp_max_t = 0;
    ProposalShape proposal = ProposalShape::none;
    PlanDecision lookup;             ///< on only where the selected execution path consumes the suffix drafter
    PlanDecision budget;
    PlanDecision concurrent_reader;
    PlanDecision suspension;         ///< on == the request keeps its seat without a resend
    std::vector<PricedClass> memory_classes;  ///< filled after admission; empty before allocation
    std::vector<std::string> notes;
    std::string to_json() const;
};

inline const char* serve_mode_name(ServeMode m) { return m == ServeMode::resident ? "resident" : "streaming"; }

inline const char* proposal_name(ProposalShape p) {
    switch (p) {
    case ProposalShape::serial_single_draft: return "serial_single_draft";
    case ProposalShape::batched_single_draft: return "batched_single_draft";
    case ProposalShape::batched_pipeline: return "batched_pipeline";
    case ProposalShape::batched_third: return "batched_third";
    case ProposalShape::batched_floating: return "batched_floating";
    case ProposalShape::none: break;
    }
    return "none";
}

inline const char* suspension_name(Suspension s) {
    switch (s) {
    case Suspension::in_place: return "in_place";
    case Suspension::resend: return "resend";
    case Suspension::none: break;
    }
    return "none";
}

inline const char* support_name(Support s) {
    switch (s) {
    case Support::on: return "on";
    case Support::conditional: return "conditional";
    case Support::refused: return "refused";
    case Support::off: break;
    }
    return "off";
}

// Resolve the option-level contract. The caller supplies values that already passed their own type/syntax checks;
// this function owns the composition rules that decide whether the selected mode can run at all.
inline ExecutionPlan resolve_execution_plan(const ExecutionPlanInput& in) {
    ExecutionPlan p;
    p.mode = in.serve ? ServeMode::resident : ServeMode::streaming;
    p.stages = 1 + std::max(0, in.later_stages);
    p.batch_slots = batch_groups::normalized_slots(in.batch_requested, 8);
    // A slot's session must fit before the count is effective; keep the request and label the fit conditional.
    p.batch_slots_resolved = in.batch_requested <= 0;
    p.spec_requested = in.spec_requested;

    // ---- speculative width (the existing suffix/lookup sizing, resolved once) --------------------------------
    int spec = in.spec_requested;
    int mtp_max = in.mtp_max_requested;
    if (in.suffix_draft > 0 && spec >= 2 && mtp_max == 0) {
        mtp_max = spec;                       // the MTP keeps its own windows; lookup may go two longer
        spec = std::min(spec + 2, 8);         // kVerifyMaxT
    }
    if (in.lookup_chain > 0 && spec >= 2) {
        if (mtp_max == 0) mtp_max = spec;
        spec = std::max(spec, std::min(std::max(spec, mtp_max) + in.lookup_chain, 8));
    }
    p.spec = spec;
    p.mtp_max_t = std::min(mtp_max, spec);

    // ---- slot groups --------------------------------------------------------------------------------------
    batch_groups::Input bin;
    bin.set = in.groups_set;
    bin.auto_given = in.groups_auto;
    bin.given = in.groups_given;
    bin.batch = p.batch_slots;
    bin.later_stages = in.later_stages;
    bin.batch_mtp = in.batch_mtp;
    bin.adapt_async = in.adapt_async;
    bin.mtp_serial = in.mtp_serial;
    const batch_groups::Result groups = batch_groups::resolve(bin);
    p.groups = groups.groups;
    p.groups_from_auto = groups.from_auto;
    p.groups_requested_auto = groups.from_auto;
    p.groups_requested = groups.from_auto ? groups.would_be : in.groups_given;
    // A requested number that a layer split cannot divide, or that has no split to pipeline through, resolves to one
    // group with a warning rather than a fractional schedule.
    if (p.batch_slots > 0 && p.groups > 1 &&
        (in.later_stages <= 0 || p.batch_slots % p.groups != 0))
        p.groups = 1;
    if (p.groups != p.groups_requested && !p.groups_from_auto)
        p.notes.push_back("--batch-groups does not divide the requested slots or has no layer split; one group");
    p.groups_conditional = p.batch_slots > 0 && !p.batch_slots_resolved;

    // ---- proposal shape -----------------------------------------------------------------------------------
    const bool mtp_shape = in.mtp_configured && p.spec >= 2;
    if (mtp_shape && p.batch_slots > 0 && in.batch_mtp_third) p.proposal = ProposalShape::batched_third;
    else if (mtp_shape && p.batch_slots > 0 && in.batch_mtp_floating) p.proposal = ProposalShape::batched_floating;
    else if (mtp_shape && p.batch_slots > 0 && in.batch_mtp_pipeline) p.proposal = ProposalShape::batched_pipeline;
    else if (mtp_shape && p.batch_slots > 0 && in.batch_mtp) p.proposal = ProposalShape::batched_single_draft;
    else if (mtp_shape) p.proposal = ProposalShape::serial_single_draft;

    // ---- lookup -------------------------------------------------------------------------------------------
    p.lookup.support = Support::off;
    if (in.suffix_draft > 0 || in.lookup_chain > 0) {
        // The suffix drafter and chaining run in the serial generation loop. Resident serving constructs the
        // drafter but no serving decode path consumes it, so advertising it as effective would be false.
        if (spec < 2) {
            p.lookup.reason = "--suffix-draft/--lookup-chain need --spec T (T >= 2)";
        } else if (p.mode == ServeMode::resident) {
            // A solo serving GEN runs the serial loop and can execute lookup after its first window; a resident
            // decode window does not, and a BGEN admission leg is a single first window (lookup skips it).
            p.lookup.support = Support::conditional;
            p.lookup.reason = "executes in a solo serial serving window; not in a resident decode window, and the "
                              "BGEN single admission window skips it";
        } else {
            p.lookup.support = Support::on;
        }
    }

    // ---- foreground budget --------------------------------------------------------------------------------
    p.budget.support = Support::off;
    if (!TimeBudget::valid(in.foreground_budget_ms, in.background_share)) {
        p.refused = true;
        p.refusal = "require budget 0..60000 ms, share 0.01..0.99";
        p.budget.support = Support::refused;
        p.budget.reason = p.refusal;
    } else if (in.foreground_budget_ms > 0) {
        const int min_slots = in.packed_prefill ? 1 : 2;
        if (!in.serve) p.budget.reason = "a positive budget needs --serve";
        else if (p.batch_slots < min_slots) p.budget.reason = "a positive budget needs --batch >= " + std::to_string(min_slots);
        else if (!in.prompt_cache) p.budget.reason = "a positive budget needs --prompt-cache > 0";
        if (p.budget.reason.empty()) p.budget.support = Support::on;
        else { p.budget.support = Support::refused; p.refused = true; p.refusal = p.budget.reason; }
    }

    // ---- concurrent reader --------------------------------------------------------------------------------
    p.concurrent_reader.support = Support::off;
    if (in.piped_concurrent) {
        if (p.batch_slots <= 0) p.concurrent_reader.reason = "--batch";
        else if (in.packed_prefill) p.concurrent_reader.reason = "packed prefill";
        else if (in.retained_prefill) p.concurrent_reader.reason = "retained prefill";
        else if (in.batch_mtp || in.batch_mtp_pipeline) p.concurrent_reader.reason = "batched MTP";
        else if (in.resident_multiplex) p.concurrent_reader.reason = "--resident-multiplex";
        else if (in.judge_lane) p.concurrent_reader.reason = "the judge lane";
        else if (in.adapt_async) p.concurrent_reader.reason = "--adapt-async";
        else if (in.spec_split) p.concurrent_reader.reason = "--spec-split";
        else if (in.prefill_pipe) p.concurrent_reader.reason = "STRATA_PREFILL_PIPE";
        else if (in.foreground_budget_ms > 0) p.concurrent_reader.reason = "the foreground budget";
        if (!p.concurrent_reader.reason.empty()) p.concurrent_reader.support = Support::off;
        else p.concurrent_reader.support = Support::conditional;  // runtime source/helper checks remain
    }

    // ---- prompt suspension --------------------------------------------------------------------------------
    if (in.packed_prefill) {
        p.suspension.support = Support::on;
        p.suspension.reason = "packed preparation keeps its destination and frontier";
    } else if (p.batch_slots > 0) {
        p.suspension.support = Support::conditional;
        p.suspension.reason = "a yield parks the seat and the frontend resends the retained prompt";
    } else {
        p.suspension.reason = "single-request serving has no yield boundary";
    }

    // ---- composition refusals -----------------------------------------------------------------------------
    if (!p.refused && in.resident_multiplex) {
        if (!in.serve) p.refusal = "--resident-multiplex needs --serve";
        else if (p.batch_slots < 2) p.refusal = "--resident-multiplex needs --batch >= 2";
        else if (in.later_stages > 0) p.refusal = "--resident-multiplex needs a single device";
        else if (p.groups != 1) p.refusal = "--resident-multiplex needs one group";
        if (!p.refusal.empty()) { p.refused = true; p.concurrent_reader.support = Support::off; }
    }
    if (!p.refused && in.batch_mtp) {
        const int min_slots = in.packed_prefill ? 1 : 2;
        if (p.batch_slots < min_slots) p.refusal = "--batch-mtp needs --batch " + std::to_string(min_slots) + " or more";
        else if (!in.mtp_configured) p.refusal = "--batch-mtp needs --mtp";
        else if (p.spec < 2) p.refusal = "--batch-mtp needs --spec T (T >= 2)";
        else if (!in.serve) p.refusal = "--batch-mtp needs --serve";
        else if (in.later_stages > 0 && !in.batch_mtp_pipeline && p.groups > 1)
            p.refusal = groups.mtp_dropped
                ? "a layer split with --batch pipelines the slots in groups by default and the pipelined groups run no "
                  "slot drafts; --batch-groups 1 or STRATA_BATCH_MTP_SERIAL=1 keeps --batch-mtp with serial windows"
                : "--batch-mtp does not combine with --batch-groups (pipelined slot groups) yet";
        if (!p.refusal.empty()) p.refused = true;
    }
    if (!p.refused && in.batch_mtp_pipeline) {
        if (!in.batch_mtp) p.refusal = "--batch-mtp-pipeline needs --batch-mtp";
        else if (in.later_stages <= 0) p.refusal = "--batch-mtp-pipeline needs a distinct-device layer split";
        else if (!batch_groups::pipeline_slots_supported(p.batch_slots, 8))
            p.refusal = "--batch-mtp-pipeline needs effective even --batch 2 through 8";
        else if (p.groups != 2) p.refusal = "--batch-mtp-pipeline needs --batch-groups 2";
        else if (in.resident_multiplex) p.refusal = "--batch-mtp-pipeline excludes resident multiplexing";
        if (!p.refusal.empty()) p.refused = true;
    }
    if (!p.refused && in.batch_mtp_floating) {
        if (!in.batch_mtp_pipeline || p.batch_slots != 4 || p.groups != 2)
            p.refusal = "--batch-mtp-floating needs --batch-mtp-pipeline --batch 4 --batch-groups 2";
        if (!p.refusal.empty()) p.refused = true;
    }
    if (!p.refused && in.batch_mtp_third) {
        if (!in.serve || !in.batch_mtp_floating || !in.batch_mtp_pipeline || p.batch_slots != 4 || p.groups != 2)
            p.refusal = "--batch-mtp-third needs serving floating four-slot/two-group MTP";
        if (!p.refusal.empty()) p.refused = true;
    }
    return p;
}

namespace detail {
inline std::string json_quote(const std::string& s) {
    std::string out = "\"";
    for (const char c : s) {
        switch (c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default: out += c;
        }
    }
    out += '"';
    return out;
}
inline std::string decision_json(const PlanDecision& d) {
    std::string out = "{\"support\":";
    out += json_quote(support_name(d.support));
    out += ",\"reason\":";
    out += json_quote(d.reason);
    out += '}';
    return out;
}
} // namespace detail

inline std::string ExecutionPlan::to_json() const {
    std::string out = "{\"version\":1,\"mode\":";
    out += detail::json_quote(serve_mode_name(mode));
    out += ",\"stages\":" + std::to_string(stages);
    out += ",\"batch\":{\"requested\":" + std::to_string(batch_slots) +
           ",\"resolved\":" + (batch_slots_resolved ? "true" : "false") +
           ",\"groups_requested\":" + std::to_string(groups_requested) +
           ",\"groups_requested_auto\":" + (groups_requested_auto ? "true" : "false") +
           ",\"groups\":" + std::to_string(groups) +
           ",\"groups_conditional\":" + (groups_conditional ? "true" : "false") + '}';
    out += ",\"speculation\":{\"requested\":" + std::to_string(spec_requested) +
           ",\"spec\":" + std::to_string(spec) +
           ",\"mtp_max\":" + std::to_string(mtp_max_t) +
           ",\"proposal\":" + detail::json_quote(proposal_name(proposal)) + '}';
    out += ",\"lookup\":" + detail::decision_json(lookup);
    out += ",\"budget\":" + detail::decision_json(budget);
    out += ",\"concurrent_reader\":" + detail::decision_json(concurrent_reader);
    out += ",\"suspension\":{\"mode\":";
    out += detail::json_quote(suspension.support == Support::on ? "in_place"
                              : suspension.support == Support::conditional ? "resend" : "none");
    out += ",\"support\":";
    out += detail::json_quote(support_name(suspension.support));
    out += ",\"reason\":";
    out += detail::json_quote(suspension.reason);
    out += '}';
    out += ",\"memory_classes\":[";
    for (size_t i = 0; i < memory_classes.size(); ++i) {
        const PricedClass& c = memory_classes[i];
        if (i) out += ',';
        out += "{\"scope\":" + detail::json_quote(c.scope) +
               ",\"name\":" + detail::json_quote(c.name) +
               ",\"bytes\":" + std::to_string(c.bytes) +
               ",\"unknown\":" + (c.unknown ? "true" : "false") +
               ",\"source\":" + detail::json_quote(c.source) + '}';
    }
    out += ']';
    out += ",\"refused\":" + std::string(refused ? "true" : "false");
    out += ",\"refusal\":" + detail::json_quote(refusal);
    out += ",\"notes\":[";
    for (size_t i = 0; i < notes.size(); ++i) {
        if (i) out += ',';
        out += detail::json_quote(notes[i]);
    }
    out += "]}";
    return out;
}

} // namespace strata::program
