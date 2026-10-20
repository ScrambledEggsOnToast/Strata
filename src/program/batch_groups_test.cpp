// src/program/batch_groups_test.cpp - how many pipelined slot groups --batch runs with (header-only, CPU).
//
//   1. `auto` picks the most groups, at most one per stage, that divide the slots;
//   2. a layer split with --batch >= 2 and no --batch-groups means `auto` (0.1.41);
//   3. --adapt-async 1 or STRATA_BATCH_MTP_SERIAL turns an unspecified default into one group;
//   4. explicit counts divide the slots on a layer split, or fall back loudly to the existing serial schedule;
//   5. slot fit reselects auto, but never resurrects a pre-fit explicit fallback.
#include "strata/program/batch_groups.hpp"

#include <cstdio>

namespace bg = strata::program::batch_groups;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-78s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
bg::Input in(int batch, int stages) {
    bg::Input i;
    i.batch = batch;
    i.later_stages = stages;
    return i;
}
}  // namespace

int main() {
    std::printf("batch_groups_test\n");
    check(bg::auto_groups(0, 4) == 1, "auto: no layer split, one group");
    check(bg::auto_groups(1, 2) == 2 && bg::auto_groups(1, 4) == 2 && bg::auto_groups(1, 3) == 1,
          "auto: two GPUs, 2 groups when the slots divide by 2");
    check(bg::auto_groups(3, 8) == 4 && bg::auto_groups(3, 6) == 3 && bg::auto_groups(3, 2) == 2,
          "auto: at most one group per GPU, and a divisor of the slots");
    {
        const int requested = 9;
        const int effective = bg::normalized_slots(requested, 8);
        check(requested == 9 && effective == 8 && bg::pipeline_slots_supported(effective, 8),
              "requested nine slots normalize to eight before the even-count pipeline gate");
        check(bg::normalized_slots(7, 8) == 7 && !bg::pipeline_slots_supported(7, 8),
              "an odd effective count still refuses the two-group MTP pipeline");
        check(bg::pipeline_slots_supported(2, 8) && bg::pipeline_slots_supported(6, 8) &&
              !bg::pipeline_slots_supported(1, 8) && !bg::pipeline_slots_supported(9, 8),
              "only effective even counts two through eight are supported");
        check(bg::pipeline_slots_supported(4, 8) && !bg::pipeline_slots_supported(3, 8),
              "the post-fit count must still pass the same gate");
    }

    {   // the 0.1.41 default
        const auto r = bg::resolve(in(2, 1));
        check(r.groups == 2 && r.from_auto && r.serial == bg::Serial::None, "layer split, --batch 2, nothing given: 2 groups");
    }
    {
        const auto r = bg::resolve(in(1, 1));
        check(r.groups == 1 && !r.from_auto, "--batch 1: no default");
    }
    {
        const auto r = bg::resolve(in(2, 0));
        check(r.groups == 1 && !r.from_auto, "no layer split: one group");
    }

    {   // the production shape: parallel 2 on a 2-GPU split with --batch-mtp
        bg::Input i = in(2, 1);
        i.batch_mtp = true;
        auto r = bg::resolve(i);
        check(r.groups == 2 && r.from_auto && r.serial == bg::Serial::None && r.mtp_dropped,
              "--batch-mtp, nothing given: draftless pipeline resolves; native caller refuses without its MTP pipeline owner");
        i.mtp_serial = true;
        r = bg::resolve(i);
        check(r.groups == 1 && r.from_auto && r.serial == bg::Serial::BatchMtp && r.would_be == 2 && !r.mtp_dropped,
              "STRATA_BATCH_MTP_SERIAL: one group, because of --batch-mtp");
        check(bg::serial_what(r.serial)[0] == '-', "the reason names the feature");
    }
    {
        bg::Input i = in(2, 1);
        i.adapt_async = true;
        const auto r = bg::resolve(i);
        check(r.groups == 1 && r.serial == bg::Serial::AdaptAsync, "--adapt-async 1, nothing given: one group");
    }
    {
        bg::Input i = in(2, 1);
        i.batch_mtp = true;
        i.adapt_async = true;
        check(bg::resolve(i).serial == bg::Serial::AdaptAsync, "both: --adapt-async 1 still gives one group");
        i.mtp_serial = true;
        check(bg::resolve(i).serial == bg::Serial::BatchMtp, "both with STRATA_BATCH_MTP_SERIAL: --batch-mtp is named");
    }
    {   // nothing to give way: the slots do not divide, so auto is one group anyway and there is nothing to say
        bg::Input i = in(3, 1);
        i.batch_mtp = true;
        const auto r = bg::resolve(i);
        check(r.groups == 1 && r.serial == bg::Serial::None && r.would_be == 1, "auto would be 1 group anyway: no reason");
    }
    {   // explicit values are honoured
        bg::Input i = in(4, 1);
        i.set = true;
        i.given = 2;
        i.batch_mtp = true;
        i.adapt_async = true;
        const auto r = bg::resolve(i);
        check(r.groups == 2 && !r.from_auto && r.serial == bg::Serial::None, "--batch-groups 2 with --batch-mtp: 2");
        i.given = 1;
        check(bg::resolve(i).groups == 1, "--batch-groups 1: 1 (the opt-out)");
        i.given = 4;
        check(bg::resolve(i).groups == 4, "--batch-groups 4 divides four slots and is returned as given");
    }
    {
        bg::Input i = in(2, 1);
        i.set = true;
        i.auto_given = true;
        i.batch_mtp = true;
        const auto r = bg::resolve(i);
        check(r.groups == 2 && r.from_auto && r.serial == bg::Serial::None,
              "--batch-groups auto with --batch-mtp: auto asked for the pipeline, 2 groups");
    }
    {
        bg::Input i = in(2, 0);
        i.set = true;
        i.given = 2;
        const auto r = bg::resolve(i);
        check(r.groups == 1 && r.requested == 2 && r.fallback == bg::Fallback::NoSplit,
              "an explicit count without a split retains the request and resolves one group");
        check(bg::fallback_message(r, 2) ==
              "--batch-groups requested=2 -> effective=1; slots=2: 1 group of 2 slots (no group pipeline; no layer split)",
              "no-split fallback names the requested and actual schedule in one line");
    }
    {
        bg::Input i = in(6, 2);
        i.set = true;
        i.given = 4;
        const auto r = bg::resolve(i);
        check(r.groups == 1 && r.requested == 4 && r.fallback == bg::Fallback::NonDividing,
              "an explicit non-divisor keeps requested four and runs one group");
        check(bg::fallback_message(r, 6) ==
              "--batch-groups requested=4 -> effective=1; slots=6: 1 group of 6 slots (no group pipeline; requested groups do not divide slots)",
              "non-dividing fallback names requested groups, fitted slots and the serial schedule");
        i.batch = 4;
        const auto fit = bg::resolve_after_fit(i, r);
        check(fit.groups == 1 && fit.requested == 4 && fit.fallback == bg::Fallback::NonDividing,
              "a pre-fit fallback is not resurrected when the fitted count happens to divide");
        check(bg::fallback_message(fit, 4).find("pre-fit slots=6") != std::string::npos,
              "a sticky fallback reports the original incompatible count, not a false claim about fitted slots");
    }
    {
        bg::Input i = in(8, 2);
        i.set = true;
        i.given = 4;
        const auto before = bg::resolve(i);
        i.batch = 6;
        const auto fit = bg::resolve_after_fit(i, before);
        check(before.groups == 4 && fit.groups == 1 && fit.requested == 4 && fit.after_fit,
              "post-fit shrink from eight slots to six retains requested four and selects one group");
        check(bg::fallback_message(fit, 6) ==
              "--batch-groups requested=4 -> effective=1; slots=6: 1 group of 6 slots (no group pipeline; after slot fit; requested groups do not divide slots)",
              "post-fit fallback is explicitly reported rather than silently clamped");
        i.batch = 4;
        const auto dividing = bg::resolve_after_fit(i, before);
        check(dividing.groups == 4 && dividing.requested == 4 && bg::fallback_message(dividing, 4).empty(),
              "dividing post-fit shape keeps four one-slot pipeline groups without a fallback warning");
    }
    for (int slots = 2; slots <= 8; ++slots) {
        for (int groups = 1; groups <= 8; ++groups) {
            bg::Input i = in(slots, 2);
            i.set = true;
            i.given = groups;
            const auto r = bg::resolve(i);
            check(r.groups == (slots % groups == 0 ? groups : 1) && r.requested == groups &&
                  (bg::fallback_message(r, slots).empty() == (slots % groups == 0)),
                  "explicit count preserves every dividing schedule; every non-divisor reports fallback");
        }
        bg::Input i = in(slots, 2);
        i.set = i.auto_given = true;
        const auto r = bg::resolve(i);
        check(slots % r.groups == 0 && r.fallback == bg::Fallback::None && bg::fallback_message(r, slots).empty(),
              "auto always selects a divisor and never reports an explicit fallback");
    }
    {
        bg::Input i = in(8, 2);
        const auto before = bg::resolve(i);
        i.batch = 6;
        const auto fit = bg::resolve_after_fit(i, before);
        check(before.groups == 2 && fit.groups == 3 && fit.requested == 2 && fit.from_auto,
              "post-fit auto reselects the divisor while retaining the initial requested shape");
        i.batch = 0;
        const auto off = bg::resolve_after_fit(i, before);
        check(off.groups == 1 && off.from_auto && off.requested == 2,
              "no fitted slots selects one group and retains the auto request attribution");
    }
    for (const bool mtp_serial : {false, true}) {
        bg::Input i = in(8, 2);
        i.batch_mtp = mtp_serial;
        i.mtp_serial = mtp_serial;
        i.adapt_async = !mtp_serial;
        const auto before = bg::resolve(i);
        i.batch = 6;
        const auto fit = bg::resolve_after_fit(i, before);
        check(before.groups == 1 && fit.groups == 1 && fit.from_auto && fit.requested == 2 &&
              fit.would_be == 3 && fit.serial == (mtp_serial ? bg::Serial::BatchMtp : bg::Serial::AdaptAsync) &&
              bg::fallback_message(fit, 6).empty(),
              "post-fit default retains the MTP-serial/adapt-async override while reselecting the auto divisor");
        i.set = i.auto_given = true;
        const auto explicit_auto = bg::resolve_after_fit(i, bg::resolve(i));
        check(explicit_auto.groups == 3 && explicit_auto.serial == bg::Serial::None,
              "explicit auto still overrides serial defaults after fit");
    }

    std::printf("%s\n", g_fail == 0 ? "batch_groups_test: all passed" : "batch_groups_test: FAILED");
    return g_fail == 0 ? 0 : 1;
}
