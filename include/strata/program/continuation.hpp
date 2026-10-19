// One certified committed cut across the state families a resident request owns. Pure: no GPU, session or wire
// state. The certificate names the request generation, the live committed extent and the reusable cut, the families
// present at that cut, and the separately owned draft readiness. Speculative scratch is deliberately absent: it is
// borrowed only while a consumer drains and never becomes committed history.
#pragma once
#include <cstdint>


namespace strata::program {

enum class ContinuationRefusal : uint8_t {
    none,
    unbound_identity,
    stale_identity,
    no_tokens,
    cut_not_prefix,
    missing_qsa,
    missing_gdn,
    missing_ple,
    missing_stage,
    draft_interval,
    draft_boundary,
};

inline const char* continuation_refusal_name(ContinuationRefusal refusal) {
    switch (refusal) {
    case ContinuationRefusal::unbound_identity: return "unbound_identity";
    case ContinuationRefusal::stale_identity: return "stale_identity";
    case ContinuationRefusal::no_tokens: return "no_tokens";
    case ContinuationRefusal::cut_not_prefix: return "cut_not_prefix";
    case ContinuationRefusal::missing_qsa: return "missing_qsa";
    case ContinuationRefusal::missing_gdn: return "missing_gdn";
    case ContinuationRefusal::missing_ple: return "missing_ple";
    case ContinuationRefusal::missing_stage: return "missing_stage";
    case ContinuationRefusal::draft_interval: return "draft_interval";
    case ContinuationRefusal::draft_boundary: return "draft_boundary";
    case ContinuationRefusal::none: break;
    }
    return "none";
}

struct ContinuationInput {
    uint64_t request = 0, incarnation = 0;
    // Optional caller generation. Nonzero means the cut must belong to exactly this identity.
    uint64_t expected_request = 0, expected_incarnation = 0;
    int64_t committed_tokens = 0;   ///< the live token history extent
    int64_t cut_tokens = -1;        ///< the checkpoint cut's extent; -1 means the live extent
    bool cut_is_live_prefix = true; ///< the cut's ids are a prefix of the live history
    int64_t live_images = 0, cut_images = 0;
    // What the selected geometry/state actually requires. A family that is not required may be absent.
    bool needs_qsa = false, needs_gdn = false, needs_ple = false, needs_stages = false;
    bool has_qsa = true, has_gdn = true, has_ple = true, has_stages = true;
    // Draft readiness is separate from the committed cut. An absent draft is not a refusal.
    bool draft_present = false;
    int64_t draft_begin = 0, draft_end = 0;
    int64_t draft_frontier = -1;
    int32_t draft_boundary = -1;
    int32_t token_at_frontier = -1;
};

struct ContinuationCertificate {
    bool certified = false;
    ContinuationRefusal refusal = ContinuationRefusal::none;
    uint64_t request = 0, incarnation = 0;
    int64_t committed_tokens = 0;
    int64_t valid_begin = 0, valid_end = 0;
    bool qsa = false, gdn = false, ple = false, stages = false;
    bool draft_ready = false;
    int64_t draft_begin = 0, draft_end = 0;

    const char* reason() const { return continuation_refusal_name(refusal); }
};

inline ContinuationCertificate certify_continuation(const ContinuationInput& in) {
    ContinuationCertificate out;
    out.request = in.request;
    out.incarnation = in.incarnation;
    out.committed_tokens = in.committed_tokens;
    if (!in.request || !in.incarnation) {
        out.refusal = ContinuationRefusal::unbound_identity;
        return out;
    }
    if ((in.expected_request && in.expected_request != in.request) ||
        (in.expected_incarnation && in.expected_incarnation != in.incarnation)) {
        out.refusal = ContinuationRefusal::stale_identity;
        return out;
    }
    if (in.committed_tokens < 1) {
        out.refusal = ContinuationRefusal::no_tokens;
        return out;
    }
    const int64_t cut = in.cut_tokens < 0 ? in.committed_tokens : in.cut_tokens;
    if (cut < 1 || cut > in.committed_tokens || !in.cut_is_live_prefix ||
        (in.cut_images != 0 && (in.live_images == 0 || in.cut_images > in.live_images))) {
        out.refusal = ContinuationRefusal::cut_not_prefix;
        return out;
    }
    // Token history and every required family must be present at the same common cut.
    if (in.needs_qsa && !in.has_qsa) { out.refusal = ContinuationRefusal::missing_qsa; return out; }
    if (in.needs_gdn && !in.has_gdn) { out.refusal = ContinuationRefusal::missing_gdn; return out; }
    if (in.needs_ple && !in.has_ple) { out.refusal = ContinuationRefusal::missing_ple; return out; }
    if (in.needs_stages && !in.has_stages) { out.refusal = ContinuationRefusal::missing_stage; return out; }
    if (in.draft_present) {
        if (in.draft_frontier < in.draft_begin || in.draft_frontier > in.draft_end ||
            in.draft_frontier > cut) {
            out.refusal = ContinuationRefusal::draft_interval;
            return out;
        }
        if (in.draft_boundary < 0 || in.token_at_frontier < 0 ||
            in.draft_boundary != in.token_at_frontier) {
            out.refusal = ContinuationRefusal::draft_boundary;
            return out;
        }
    }
    out.certified = true;
    out.valid_begin = 0;
    out.valid_end = cut;
    out.qsa = in.has_qsa;
    out.gdn = in.has_gdn;
    out.ple = in.has_ple;
    out.stages = in.has_stages;
    out.draft_ready = in.draft_present;
    out.draft_begin = in.draft_present ? in.draft_begin : 0;
    out.draft_end = in.draft_present ? in.draft_end : 0;
    return out;
}

} // namespace strata::program
