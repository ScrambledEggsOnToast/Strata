// src/program/draft_kv_plan_test.cpp - which path computes the draft layer's prompt K/V (header-only, CPU).
//
//   1. STRATA_SPLIT_MTP_BATCH: unset, empty, 0 and text that is not a number are off; any other whole number is on;
//   2. one GPU: the batched pass is tried whatever the variable says (the default before this change);
//   3. a layer split: the default stays the drafter's own pass, silently; with the variable on the batched pass is
//      tried, unless the pack has no GGUF-form token table, which is reported instead;
//   4. no draft layer: nothing is tried and nothing is said.
//   5. shared MTP reads retain the cut's prompt horizon, with immutable R captured only for a certified store;
//      native DFlash keeps full-prompt metadata and never requests the MTP R boundary.
#include "strata/program/draft_kv_plan.hpp"

#include <cstdio>
#include <initializer_list>

namespace dkv = strata::program::draft_kv;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-72s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

void prompt_horizons() {
    using strata::core::DraftKind;
    struct Case {
        DraftKind kind;
        int64_t resume, reread_to;
        bool shared, certified_store;
        int64_t read_from, prompt_len, boundary_capture;
        const char* description;
    };
    const Case cases[] = {
        {DraftKind::mtp, 0, -1, false, true, 0, 65536, -1, "private fresh MTP uses the full prompt without a shared boundary"},
        {DraftKind::mtp, 4096, -1, false, true, 4096, 65536, -1, "private resumed MTP uses the full prompt"},
        {DraftKind::mtp, 16384, 16384, false, true, 0, 65536, -1, "private reread starts at zero but keeps the full prompt"},
        {DraftKind::mtp, 0, -1, true, true, 0, 8193, 8191, "fresh shared MTP retains the cut and captures its immutable R"},
        {DraftKind::mtp, 4096, -1, true, true, 4096, 8193, 8191, "MTP resumed before the pin retains the shared-cut horizon"},
        {DraftKind::mtp, 8192, -1, true, true, 8192, 65536, -1, "MTP resumed exactly at the pin does not recreate the cut"},
        {DraftKind::mtp, 16384, -1, true, true, 16384, 65536, -1, "MTP resumed past the pin keeps the ordinary horizon"},
        {DraftKind::mtp, 16384, 16384, true, true, 0, 8193, 8191, "shared MTP reread recreates the cut despite a deeper resume"},
        {DraftKind::mtp, 16384, 0, true, true, 16384, 65536, -1, "a zero reread limit does not restart at token zero"},
        {DraftKind::mtp, 4096, -1, true, false, 4096, 8193, -1, "legacy shared MTP retains required rows without capturing R"},
        {DraftKind::mtp, 16384, 16384, true, false, 0, 8193, -1, "legacy shared MTP reread retains the horizon but not the R boundary"},
        {DraftKind::dflash2, 0, -1, false, true, 0, 65536, -1, "private DFlash keeps full-prompt metadata"},
        {DraftKind::dflash2, 0, -1, true, true, 0, 65536, -1, "fresh shared DFlash has no MTP horizon or R boundary"},
        {DraftKind::dflash2, 4096, -1, true, true, 4096, 65536, -1, "DFlash resumed before the pin keeps full-prompt metadata"},
        {DraftKind::dflash2, 8192, -1, true, true, 8192, 65536, -1, "DFlash resumed at the pin keeps full-prompt metadata"},
        {DraftKind::dflash2, 16384, -1, true, true, 16384, 65536, -1, "DFlash resumed past the pin keeps full-prompt metadata"},
        {DraftKind::dflash2, 16384, 16384, true, true, 0, 65536, -1, "certified shared DFlash reread does not request MTP R capture"},
        {DraftKind::dflash2, 16384, 16384, true, false, 0, 65536, -1, "legacy shared DFlash reread also keeps full-prompt metadata"},
        {DraftKind::none, 0, -1, true, true, 0, 65536, -1, "no neural drafter has no shared draft boundary"},
    };
    for (const auto& test : cases) {
        const dkv::PromptHorizon result = dkv::prompt_horizon(
            test.kind, 65536, test.resume, test.reread_to, test.shared, 8192, test.certified_store);
        check(result.read_from == test.read_from && result.prompt_len == test.prompt_len &&
              result.boundary_capture == test.boundary_capture, test.description);
    }
}
}  // namespace

int main() {
    std::printf("draft_kv_plan_test\n");
    check(!dkv::split_env_on(nullptr), "unset is off");
    check(!dkv::split_env_on(""), "empty is off");
    check(!dkv::split_env_on("0"), "0 is off");
    check(!dkv::split_env_on("off"), "text that is not a number is off");
    check(dkv::split_env_on("1"), "1 is on");
    check(dkv::split_env_on("2"), "any other whole number is on");

    for (const bool env : {false, true})
        for (const bool native : {false, true}) {
            const dkv::Plan p = dkv::plan(true, false, env, native);
            check(p.try_batched && p.why == nullptr, "one GPU: the batched pass is tried, nothing is reported");
        }

    {
        const dkv::Plan p = dkv::plan(true, true, false, true);
        check(!p.try_batched && p.why == nullptr, "split, variable off: the drafter's own pass, silently");
    }
    {
        const dkv::Plan p = dkv::plan(true, true, false, false);
        check(!p.try_batched && p.why == nullptr, "split, variable off, no token table: still silent");
    }
    {
        const dkv::Plan p = dkv::plan(true, true, true, true);
        check(p.try_batched && p.why == nullptr, "split, variable on: the batched pass is tried on the last stage");
    }
    {
        const dkv::Plan p = dkv::plan(true, true, true, false);
        check(!p.try_batched && p.why != nullptr, "split, variable on, no token table: declined with a reason");
    }
    for (const bool multi : {false, true})
        for (const bool env : {false, true}) {
            const dkv::Plan p = dkv::plan(false, multi, env, true);
            check(!p.try_batched && p.why == nullptr, "no draft layer: nothing tried, nothing said");
        }

    prompt_horizons();

    std::printf(g_fail == 0 ? "draft_kv_plan_test: all ok\n" : "draft_kv_plan_test: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
