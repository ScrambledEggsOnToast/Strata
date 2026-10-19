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
    }
    std::printf("%s\n", g_fail == 0 ? "continuation_test: all passed" : "continuation_test: FAILED");
    return g_fail == 0 ? 0 : 1;
}
