// One certified committed cut across the state families a resident request owns. Pure: no GPU, session or wire
// state. The certificate names the request generation, the live committed extent and the reusable cut, the families
// present at that cut, and the separately owned draft readiness. Speculative scratch is deliberately absent: it is
// borrowed only while a consumer drains and never becomes committed history.
#pragma once
#include "strata/core/draft_state.hpp"
#include <cstdint>
#include <algorithm>


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
    draft_owner,
    draft_interval,
    draft_boundary,
    draft_absent,
    draft_window,
    draft_reconstruction_unpriced,
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
    case ContinuationRefusal::draft_owner: return "draft_owner";
    case ContinuationRefusal::draft_interval: return "draft_interval";
    case ContinuationRefusal::draft_boundary: return "draft_boundary";
    case ContinuationRefusal::draft_absent: return "draft_absent";
    case ContinuationRefusal::draft_window: return "draft_window";
    case ContinuationRefusal::draft_reconstruction_unpriced: return "draft_reconstruction_unpriced";
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
    uint64_t draft_request = 0;
    bool draft_shape_valid = true;
    int64_t draft_begin = 0, draft_end = 0;
    int64_t draft_frontier = -1;
    int32_t draft_boundary = -1;
    int32_t token_at_frontier = -1;
    core::DraftKind draft_kind = core::DraftKind::mtp;
    uint64_t draft_incarnation = 0;
    int64_t draft_origin = 0;
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
    int64_t draft_frontier = -1;
    core::DraftKind draft_kind = core::DraftKind::none;
    int64_t draft_origin = 0;

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
        if (in.draft_request != in.request ||
            (in.draft_incarnation && in.draft_incarnation != in.incarnation)) {
            out.refusal = ContinuationRefusal::draft_owner;
            return out;
        }
        if (!in.draft_shape_valid || in.draft_begin < 0 || in.draft_end < in.draft_begin ||
            in.draft_frontier < in.draft_begin || in.draft_frontier > in.draft_end ||
            in.draft_frontier > cut) {
            out.refusal = ContinuationRefusal::draft_interval;
            return out;
        }
        // An empty certified interval has no row that consumed a boundary token.
        if (in.draft_kind == core::DraftKind::mtp && in.draft_frontier > in.draft_begin &&
            (in.draft_boundary < 0 || in.token_at_frontier < 0 || in.draft_boundary != in.token_at_frontier)) {
            out.refusal = ContinuationRefusal::draft_boundary;
            return out;
        }
        if (in.draft_kind == core::DraftKind::dflash2 &&
            (in.draft_origin < 0 || in.draft_begin < in.draft_origin ||
             in.draft_end - in.draft_begin > 2047 ||
             in.draft_frontier != cut || in.draft_begin != std::max(in.draft_origin, cut - 2047))) {
            out.refusal = ContinuationRefusal::draft_interval;
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
    out.draft_frontier = in.draft_present ? in.draft_frontier : -1;
    out.draft_kind = in.draft_present ? in.draft_kind : core::DraftKind::none;
    out.draft_origin = in.draft_origin;
    return out;
}

// #282: readiness is a separate, read-only decision, not a consequence of target certification.
// The existing one-cell snapshot catchup has its own owned residual and preceding certified KV.
// Resident cuts retain tokens/checkpoints/KV, not the final residual rows needed to rebuild a missing
// draft interval. Price that input lower bound, but never claim a priced replay/scratch plan or run one.
struct DraftReadinessDecision {
    bool ready = false;
    ContinuationRefusal refusal = ContinuationRefusal::none;
    int64_t residual_rows = 0;
    uint64_t residual_bytes = 0;
    bool residual_bytes_known = false;
    bool reconstruction_priced = false;

    const char* reason() const { return continuation_refusal_name(refusal); }
};

inline DraftReadinessDecision decide_draft_readiness(const ContinuationCertificate& cut,
        int64_t prompt_tokens, int64_t window, uint64_t residual_row_bytes) {
    DraftReadinessDecision out;
    if (!cut.certified) {
        out.refusal = cut.refusal;
        return out;
    }
    if (prompt_tokens <= cut.valid_end || window < 0) {
        out.refusal = ContinuationRefusal::draft_interval;
        return out;
    }
    const int64_t needed = cut.draft_kind == core::DraftKind::dflash2
        ? std::max(cut.draft_origin, cut.valid_end - int64_t(2047))
        : window > 0 && prompt_tokens > window ? prompt_tokens - window : 0;
    const int64_t frontier = cut.draft_ready ? cut.draft_frontier : cut.valid_end;
    const int64_t begin = frontier < needed ? frontier : needed;
    if (cut.draft_ready && cut.draft_begin <= begin) {
        out.ready = out.residual_bytes_known = out.reconstruction_priced = true;
        return out; // certified reuse, no reconstruction work or additional scratch
    }
    out.refusal = cut.draft_ready ? ContinuationRefusal::draft_window : ContinuationRefusal::draft_absent;
    out.residual_rows = (cut.draft_ready ? cut.draft_begin : frontier) - begin;
    if (!residual_row_bytes || uint64_t(out.residual_rows) > UINT64_MAX / residual_row_bytes) {
        out.refusal = ContinuationRefusal::draft_reconstruction_unpriced;
        return out;
    }
    out.residual_bytes = uint64_t(out.residual_rows) * residual_row_bytes;
    out.residual_bytes_known = true;
    // This is required input storage, not an allowance or a successful reconstruction quote.
    // Even zero missing rows do not authorize manufacturing a new draft producer certificate.
    return out;
}

} // namespace strata::program
