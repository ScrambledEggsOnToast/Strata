// Host-only: the draft chain's coverage decision at the frontier. CUDA is linked but must never be
// initialized; this exercises only the predicate, never a graph or a transfer.
//
// 2026-10-18 daily crash: a 12,375-token batch request reused 11,756 tokens from a parked image whose
// draft rows were not certified, so the chain legitimately began at the resume point. The admission's
// initial proposal was refused ("certified=1 covers=0 slot_begin=11756") and the engine exited.
#include "strata/core/mtp.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::core {
struct MtpDraftFrontierFixture {
    static void run(void (*check)(bool, const char*)) {
        MtpDrafter draft;
        draft.window_ = 32768;
        draft.slots_.resize(1);
        auto& slot = draft.slots_[0];
        slot.state.max_cells = 262144;
        slot.request = 9;
        slot.certified = true;
        slot.begin = 11756;
        slot.cells = slot.completed_cells = 12375;
        slot.prompt_len = 12375;
        // The frontier operations: the proposal at cells-1, the advance at cells.
        check(draft.slot_covers(0, 12374) && draft.slot_covers(0, 12375),
              "a chain certified from the resume point covers its proposal and advance cells");
        check(!(slot.certified && !draft.slot_covers(0, 12374)),
              "batch admission accepts a proposal from a chain certified at 11756");
        check(!draft.slot_covers(0, 11755) && !draft.slot_covers(0, 12376),
              "cells outside the certified interval stay uncovered");
        // Reusing the chain for a prompt is a different question: it must reach the retained window's
        // first cell, which a 12,375-token prompt inside a 32,768-cell window makes token 0.
        check(!draft.slot_covers(0, 12374, 12375),
              "prompt-scoped reuse still refuses a chain that misses the retained window");
        slot.begin = 0;
        check(draft.slot_covers(0, 12374, 12375),
              "a chain certified from token 0 covers the whole retained window");
        slot.begin = 11756;
        slot.certified = false;
        check(!draft.slot_covers(0, 12374), "an uncertified slot covers nothing");
        slot.certified = true;
        slot.completed_cells = 12374;
        check(!draft.slot_covers(0, 12375),
              "a cell past the completed extent is not covered, even when the logical extent reached it");
    }
};

// 2026-10-18, second crash of the day ("restoring yielded prefix: resident resume: draft interval does
// not cover this prompt"): a 58,231-token read gave way at 55,249 and was resumed under load. The park
// had copied every draft row its slot held, but the chain had completed only 53,201 cells, so a resume
// that asked the certificate to reach the resident prefix refused it. It is judged at its own frontier.
struct MtpResumedParkFixture {
    static void run(void (*check)(bool, const char*)) {
        MtpDrafter draft;
        draft.window_ = 4096;
        draft.slots_.resize(1);
        auto& slot = draft.slots_[0];
        slot.state.max_cells = 262144;
        slot.request = 11;
        slot.certified = true;
        slot.begin = 53201;
        slot.cells = 55249;              // the parked copy's extent
        slot.completed_cells = 53201;    // what the chain actually completed
        slot.prompt_len = 58231;
        check(!draft.slot_covers(0, 55249, 58231),
              "the parked copy's extent reaches past the chain's certificate");
        const int64_t frontier = draft.slot_resume_frontier(0, 55249);
        check(frontier == 53201, "a resume judges the chain at its own certified extent");
        check(draft.slot_covers(0, frontier, 58231),
              "the resumed interval covers the prompt's retained window at that frontier");
        slot.cells = slot.completed_cells = 55249;
        check(draft.slot_resume_frontier(0, 55249) == 55249,
              "a chain that completed the whole copy keeps its frontier");
        slot.certified = false;
        check(draft.slot_resume_frontier(0, 55249) == 55249,
              "an uncertified slot is priced at the copy's extent");
    }
};
}

namespace {
int failures = 0;
void check(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); ++failures; }
}
}

int main() {
    strata::core::MtpDraftFrontierFixture::run(check);
    strata::core::MtpResumedParkFixture::run(check);
    if (failures) { std::fprintf(stderr, "%d check(s) failed\n", failures); return 1; }
    std::printf("mtp frontier: all checks passed\n");
    return 0;
}
