// include/strata/program/batch_groups.hpp - how many pipelined slot groups `--batch` runs with.
//
// `--batch-groups N` pipelines N groups of slots through the stages of a layer split; `--batch-groups auto` picks the
// most groups (at most one per stage) that divide the slots.  Since 0.1.41 a layer split with `--batch` >= 2 and no
// `--batch-groups` at all means `auto`. The ordinary pipelined path runs no MTP drafts. Native generate
// refuses a requested --batch-mtp if it resolves to multiple groups, unless --batch-mtp-pipeline owns
// the two-group route. STRATA_BATCH_MTP_SERIAL=1 or --batch-groups 1 selects serial MTP windows.
// --adapt-async 1 with groups NOT given still resolves to one group. Explicit numbers or auto are honoured.
//
// Pure (no GPU, no environment read): generate.cpp passes in what it knows.
#pragma once
#include <string>

namespace strata::program::batch_groups {

/// What kept `auto` from pipelining.
enum class Serial { None, BatchMtp, AdaptAsync };
enum class Fallback { None, NoSplit, NonDividing };

struct Input {
    bool set = false;          ///< --batch-groups was given (a number or `auto`)
    bool auto_given = false;   ///< ... as `auto`
    int given = 1;             ///< the number given (1 when none)
    int batch = 0;             ///< the slots
    int later_stages = 0;      ///< the layer split's stages after CUDA0 (0: no layer split)
    bool batch_mtp = false;    ///< --batch-mtp is on (it passed the gates that do not concern the groups)
    bool adapt_async = false;  ///< --adapt-async 1 was asked for
    bool mtp_serial = false;   ///< STRATA_BATCH_MTP_SERIAL=1: --batch-mtp with no --batch-groups runs one serial group (0.1.43)
};

struct Result {
    int groups = 1;
    int requested = 1;                 ///< raw explicit count, or the initial auto divisor, before any clamp
    Fallback fallback = Fallback::None;
    int fallback_slots = 0;            ///< slots at the fallback decision; a pre-fit serial choice is sticky
    bool after_fit = false;
    bool from_auto = false;            ///< the number came from `auto` (given or the 0.1.41 default)
    Serial serial = Serial::None;      ///< `auto` would have pipelined, but this feature needs the serial windows
    int would_be = 1;                  ///< what `auto` would have picked without it (== groups unless `serial`)
    bool mtp_dropped = false;          ///< draftless default pipeline: native caller refuses MTP without its pipeline owner
};

/// Admission and runtime normalize the request before the pipeline's effective-count gate.
constexpr int normalized_slots(int requested, int capacity) {
    return requested < 0 ? 0 : requested > capacity ? capacity : requested;
}
constexpr bool pipeline_slots_supported(int effective, int capacity) {
    return effective >= 2 && effective <= capacity && effective % 2 == 0;
}

/// The most groups, at most one per stage of the layer split (`later_stages` + 1 GPUs), that divide `batch`.
inline int auto_groups(int later_stages, int batch) {
    int best = 1;
    if (later_stages <= 0) return best;
    for (int d = 2; d <= later_stages + 1 && d <= batch; ++d)
        if (batch % d == 0) best = d;
    return best;
}

inline Result resolve(const Input& in) {
    Result r;
    r.requested = r.groups = r.would_be = in.given;
    const bool defaulted = !in.set && in.later_stages > 0 && in.batch > 1;   // the 0.1.41 default
    if (!in.auto_given && !defaulted) {
        if (in.batch > 0 && r.groups > 1) {
            if (in.later_stages <= 0) r.fallback = Fallback::NoSplit;
            else if (in.batch % r.groups != 0) r.fallback = Fallback::NonDividing;
            if (r.fallback != Fallback::None) {
                r.groups = 1;
                r.fallback_slots = in.batch;
            }
        }
        return r;
    }
    r.from_auto = true;
    r.requested = r.would_be = auto_groups(in.later_stages, in.batch);
    r.groups = r.would_be;
    if (!in.set && r.would_be > 1) {   // only the default gives way; `--batch-groups auto` asked for the pipeline
        if (in.batch_mtp && in.mtp_serial) r.serial = Serial::BatchMtp;
        else if (in.adapt_async) r.serial = Serial::AdaptAsync;
        r.mtp_dropped = in.batch_mtp && r.serial == Serial::None;
        if (r.serial != Serial::None) r.groups = 1;
    }
    return r;
}

// Preserve a pre-fit explicit fallback: shrinking the slots must not resurrect a rejected pipeline.
// Auto, unlike an explicit count, is chosen again from the fitted slots.
inline Result resolve_after_fit(Input in, const Result& before) {
    if (before.from_auto) in.auto_given = true;
    else in.given = before.groups;
    Result r = resolve(in);
    r.requested = before.requested;
    if (before.fallback != Fallback::None) {
        r.fallback = before.fallback;
        r.fallback_slots = before.fallback_slots;
        r.after_fit = before.after_fit;
    }
    else if (r.fallback != Fallback::None) r.after_fit = true;
    if (in.batch <= 0) r.groups = 1;
    return r;
}

inline std::string fallback_message(const Result& r, int slots) {
    if (r.fallback == Fallback::None) return {};
    return "--batch-groups requested=" + std::to_string(r.requested) +
        " -> effective=" + std::to_string(r.groups) + "; slots=" + std::to_string(slots) +
        (slots > 0 ? ": 1 group of " + std::to_string(slots) + " slots" : ": no resident batch") +
        " (no group pipeline; " + (r.after_fit ? "after slot fit; " : "") +
        (r.fallback == Fallback::NoSplit ? "no layer split)" :
            r.fallback_slots == slots ? "requested groups do not divide slots)" :
            "requested groups do not divide pre-fit slots=" + std::to_string(r.fallback_slots) + ")");
}

inline const char* serial_what(Serial s) {
    return s == Serial::BatchMtp ? "--batch-mtp" : s == Serial::AdaptAsync ? "--adapt-async 1" : "";
}

}  // namespace strata::program::batch_groups
