// tests/program/continuation_test.cpp - the certified continuation cut (header-only, CPU).
//
//   1. a live cut and a checkpoint cut are certified only when every required family is present and the cut is a
//      prefix of the committed history;
//   2. a stale request generation, an empty history and a non-prefix cut are refused with distinct reasons;
//   3. an absent draft is a valid committed cut (readiness is separate), while an inconsistent draft interval or
//      boundary token refuses the cut.
#include "strata/program/continuation.hpp"

#include <cstdio>
#include <string>

namespace cp = strata::program;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-78s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
cp::ContinuationInput base() {
    cp::ContinuationInput in;
    in.request = 7;
    in.incarnation = 3;
    in.committed_tokens = 100;
    in.cut_tokens = 80;
    in.needs_qsa = true;
    in.needs_gdn = true;
    in.needs_ple = true;
    in.needs_stages = true;
    in.has_qsa = in.has_gdn = in.has_ple = in.has_stages = true;
    in.draft_present = true;
    in.draft_request = in.request;
    in.draft_begin = 40;
    in.draft_end = 90;
    in.draft_frontier = 80;
    in.draft_boundary = 42;
    in.token_at_frontier = 42;
    return in;
}
}  // namespace

int main() {
    std::printf("continuation_test\n");
    {
        const auto c = cp::certify_continuation(base());
        check(c.certified && c.refusal == cp::ContinuationRefusal::none,
              "a checkpoint cut with all families and a matching draft boundary is certified");
        check(c.request == 7 && c.incarnation == 3 && c.valid_begin == 0 && c.valid_end == 80,
              "the certificate names the generation and the reusable interval");
        check(c.qsa && c.gdn && c.ple && c.stages && c.draft_ready,
              "the certified families and separate draft readiness are reported");
        cp::ContinuationInput live = base();
        live.cut_tokens = -1;
        live.draft_frontier = 100;
        live.draft_end = 100;
        live.token_at_frontier = 42;
        const auto l = cp::certify_continuation(live);
        check(l.certified && l.valid_end == 100, "a live cut reaches the committed extent");
        cp::ContinuationInput no_draft = base();
        no_draft.draft_present = false;
        const auto n = cp::certify_continuation(no_draft);
        check(n.certified && !n.draft_ready,
              "an absent draft leaves the committed cut valid and readiness false");
    }
    {
        cp::ContinuationInput stale = base();
        stale.expected_request = 8;
        stale.expected_incarnation = 4;
        check(cp::certify_continuation(stale).refusal == cp::ContinuationRefusal::stale_identity,
              "a different expected generation is a stale identity");
        cp::ContinuationInput empty = base();
        empty.committed_tokens = 0;
        check(cp::certify_continuation(empty).refusal == cp::ContinuationRefusal::no_tokens,
              "an empty committed history has no cut");
        cp::ContinuationInput cut = base();
        cut.cut_tokens = 120;
        check(cp::certify_continuation(cut).refusal == cp::ContinuationRefusal::cut_not_prefix,
              "a cut beyond the live history is not a prefix");
        cut = base();
        cut.cut_is_live_prefix = false;
        check(cp::certify_continuation(cut).refusal == cp::ContinuationRefusal::cut_not_prefix,
              "a checkpoint whose ids are not a live prefix is refused");
        cp::ContinuationInput image = base();
        image.cut_images = 2;
        image.live_images = 1;
        check(cp::certify_continuation(image).refusal == cp::ContinuationRefusal::cut_not_prefix,
              "a cut with more images than the live history is refused");
        check(cp::certify_continuation(image).reason() == std::string("cut_not_prefix"),
              "the refusal names a stable reason");
    }
    {
        cp::ContinuationInput qsa = base();
        qsa.has_qsa = false;
        check(cp::certify_continuation(qsa).refusal == cp::ContinuationRefusal::missing_qsa,
              "a required QSA family absent at the cut is refused");
        cp::ContinuationInput gdn = base();
        gdn.has_gdn = false;
        check(cp::certify_continuation(gdn).refusal == cp::ContinuationRefusal::missing_gdn,
              "a required GDN family absent at the cut is refused");
        cp::ContinuationInput ple = base();
        ple.has_ple = false;
        check(cp::certify_continuation(ple).refusal == cp::ContinuationRefusal::missing_ple,
              "a required PLE family absent at the cut is refused");
        cp::ContinuationInput stage = base();
        stage.has_stages = false;
        check(cp::certify_continuation(stage).refusal == cp::ContinuationRefusal::missing_stage,
              "a required stage part absent at the cut is refused");
        cp::ContinuationInput not_required = base();
        not_required.needs_qsa = false;
        not_required.has_qsa = false;
        check(cp::certify_continuation(not_required).certified,
              "a family the geometry does not require may be absent");
    }
    {
        cp::ContinuationInput interval = base();
        interval.draft_frontier = 30;   // before begin
        check(cp::certify_continuation(interval).refusal == cp::ContinuationRefusal::draft_interval,
              "a draft frontier outside its interval is refused");
        interval = base();
        interval.draft_frontier = 120;  // beyond the cut
        check(cp::certify_continuation(interval).refusal == cp::ContinuationRefusal::draft_interval,
              "a draft frontier beyond the committed cut is refused");
        cp::ContinuationInput boundary = base();
        boundary.token_at_frontier = 43;
        check(cp::certify_continuation(boundary).refusal == cp::ContinuationRefusal::draft_boundary,
              "a draft boundary token that differs from the prompt is refused");
        cp::ContinuationInput owner = base();
        ++owner.draft_request;
        check(cp::certify_continuation(owner).refusal == cp::ContinuationRefusal::draft_owner,
              "a present draft owned by another request is refused, not reclassified absent");
        interval = base();
        interval.draft_shape_valid = false;
        check(cp::certify_continuation(interval).refusal == cp::ContinuationRefusal::draft_interval,
              "an inconsistent completed/materialized/ring extent remains a malformed present draft");
        interval = base();
        interval.draft_begin = -1;
        check(cp::certify_continuation(interval).refusal == cp::ContinuationRefusal::draft_interval,
              "a negative begin cannot certify a present draft");
    }
    {
        const auto ready = cp::decide_draft_readiness(cp::certify_continuation(base()), 100, 60, 40960);
        check(ready.ready && ready.reconstruction_priced && ready.residual_rows == 0 &&
              ready.residual_bytes_known && ready.residual_bytes == 0,
              "certified nonzero-window reuse has an exact zero reconstruction/scratch price");
        auto absent = base();
        absent.draft_present = false;
        const auto target = cp::certify_continuation(absent);
        const auto refused = cp::decide_draft_readiness(target, 100, 60, 40960);
        check(target.certified && !target.draft_ready && !refused.ready &&
              refused.refusal == cp::ContinuationRefusal::draft_absent,
              "a valid target without a draft is refused without pretending readiness or replay");
        check(refused.residual_rows == 40 && refused.residual_bytes_known &&
              refused.residual_bytes == 1638400 && !refused.reconstruction_priced,
              "the missing 40 residual rows are bounded but not a priced reconstruction plan");
        const auto unpriced = cp::decide_draft_readiness(target, 100, 60, 0);
        check(!unpriced.ready && !unpriced.residual_bytes_known && !unpriced.reconstruction_priced &&
              unpriced.refusal == cp::ContinuationRefusal::draft_reconstruction_unpriced,
              "an unknown input-storage price cannot authorize reconstruction");
        const auto overflow = cp::decide_draft_readiness(target, 100, 60, UINT64_MAX);
        check(!overflow.ready && !overflow.residual_bytes_known &&
              overflow.refusal == cp::ContinuationRefusal::draft_reconstruction_unpriced,
              "overflowing reconstruction input bytes are unknown, never wrapped into a cheap price");
        const auto uncovered = cp::decide_draft_readiness(cp::certify_continuation(base()), 100, 80, 40960);
        check(!uncovered.ready && uncovered.refusal == cp::ContinuationRefusal::draft_window &&
              uncovered.residual_rows == 20 && uncovered.residual_bytes == 819200,
              "a valid but stale retained window refuses with the missing input lower bound");
        auto parked = base();
        parked.committed_tokens = 55249;
        parked.cut_tokens = -1;
        parked.draft_begin = parked.draft_frontier = parked.draft_end = 53201;
        parked.draft_boundary = parked.token_at_frontier = -1;
        const auto historical = cp::decide_draft_readiness(cp::certify_continuation(parked), 58231, 4096, 40960);
        check(historical.ready && historical.reconstruction_priced && historical.residual_rows == 0,
              "the historical completed frontier is preserved, not confused with the target extent");
        const auto wrong_frontier = cp::decide_draft_readiness(target, 80, 60, 40960);
        check(!wrong_frontier.ready && wrong_frontier.refusal == cp::ContinuationRefusal::draft_interval,
              "a resumed frontier must leave an actual prompt token for the next operation");
        const auto skipped = cp::decide_draft_readiness(target, 1000, 60, 40960);
        check(!skipped.ready && skipped.residual_rows == 0 && !skipped.reconstruction_priced,
              "zero presently needed rows do not manufacture a new draft producer certificate");
    }
    std::printf("%s\n", g_fail == 0 ? "continuation_test: all passed" : "continuation_test: FAILED");
    return g_fail == 0 ? 0 : 1;
}
