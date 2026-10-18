// Per-request packed-selection policy (#249): no GPU, no model, no process state.
#include <cstdio>
#include <vector>

#include "strata/prefill/packing_policy.hpp"

using namespace strata::prefill;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

static PackedSelection select(std::vector<PackedCandidate> waiting, int cap = 0, unsigned streak = 0) {
    return select_packed_seats(std::span<const PackedCandidate>(waiting.data(), waiting.size()), cap, streak);
}

static PackedCandidate ordinary(int slot, bool foreground = true) { return PackedCandidate{slot, false, foreground}; }
static PackedCandidate solo(int slot, bool foreground = true, uint64_t order = 0) {
    return PackedCandidate{slot, true, foreground, order};
}

int main() {
    // Request classification is exact.  A typo refuses rather than choosing a composition.
    {
        PackingEligibility value = PackingEligibility::solo;
        CHECK(parse_packing_eligibility("eligible", value) && value == PackingEligibility::eligible);
        CHECK(parse_packing_eligibility("solo", value) && value == PackingEligibility::solo);
        CHECK(!parse_packing_eligibility("", value));
        CHECK(!parse_packing_eligibility("packed", value));
        CHECK(!parse_packing_eligibility("SOLO", value));
        CHECK(!parse_packing_eligibility("solo ", value));
    }

    // Ordinary eligible owners keep their existing multi-owner composition.
    {
        const PackedSelection s = select({ordinary(0), ordinary(1), ordinary(2)});
        CHECK(s.seats == 0b111u);
        CHECK(s.kind == PackedPackKind::packed);
        CHECK(s.solo_streak == 0);
    }

    // Mixed membership: a waiting solo request owns the next pack alone; ordinary owners remain.
    {
        const PackedSelection s = select({ordinary(0), solo(1), ordinary(2)});
        CHECK(s.seats == 0b010u);
        CHECK(s.kind == PackedPackKind::solo);
        CHECK(s.solo_streak == 1);
        const PackedSelection again = select({ordinary(0), solo(1), ordinary(2)}, 0, s.solo_streak);
        CHECK(again.seats == 0b010u);
        CHECK(again.solo_streak == 2);
    }

    // Late priority arrival: the ordinary pack selected before the arrival finishes (its rows were
    // already issued), then the next boundary switches to the solo owner.
    {
        const PackedSelection before = select({ordinary(0), ordinary(1)});
        CHECK(before.seats == 0b011u && before.kind == PackedPackKind::packed);
        const PackedSelection arrived = select({ordinary(0), ordinary(1), solo(2)});
        CHECK(arrived.seats == 0b100u && arrived.kind == PackedPackKind::solo);
        const PackedSelection after = select({ordinary(0), ordinary(1)});
        CHECK(after.seats == 0b011u && after.kind == PackedPackKind::packed);
    }

    // Cancellation removes the owner before selection: the cancelled solo request neither gets a
    // seat nor keeps ordinary work out of the pack.
    {
        const PackedSelection cancelled = select({ordinary(0), ordinary(1)});
        CHECK(cancelled.seats == 0b011u);
        CHECK(cancelled.kind == PackedPackKind::packed);
    }

    // Bounded ordinary progress under sustained solo traffic: four solo packs, then one ordinary
    // pack, then solo resumes.  A fairness pack never contains the solo owner.
    {
        unsigned streak = 0;
        for (unsigned i = 0; i < kPackedSoloPacksBeforeOrdinary; ++i) {
            const PackedSelection s = select({solo(0), ordinary(1)}, 0, streak);
            CHECK(s.kind == PackedPackKind::solo);
            CHECK(s.seats == 0b01u);
            streak = s.solo_streak;
        }
        CHECK(streak == kPackedSoloPacksBeforeOrdinary);
        const PackedSelection fair = select({solo(0), ordinary(1)}, 0, streak);
        CHECK(fair.kind == PackedPackKind::fairness);
        CHECK(fair.seats == 0b10u);
        CHECK(fair.solo_streak == 0);
        const PackedSelection resumed = select({solo(0), ordinary(1)}, 0, fair.solo_streak);
        CHECK(resumed.kind == PackedPackKind::solo && resumed.seats == 0b01u);
    }
    {
        // With no ordinary owner waiting, the bound costs nothing and solo continues.
        const PackedSelection s = select({solo(0)}, 0, kPackedSoloPacksBeforeOrdinary);
        CHECK(s.kind == PackedPackKind::solo);
        CHECK(s.seats == 0b1u);
        CHECK(s.solo_streak == kPackedSoloPacksBeforeOrdinary + 1);
    }
    {
        // The earliest admitted solo owner goes first even when a later one reserved a lower slot.
        const PackedSelection s = select({solo(2, true, 1), solo(0, true, 2)});
        CHECK(s.seats == 0b100u);
        CHECK(s.kind == PackedPackKind::solo);
        const PackedSelection t = select({solo(2, true, 2), solo(0, true, 1)});
        CHECK(t.seats == 0b001u);
    }

    // The existing foreground cap still orders eligible owners: foreground first in candidate
    // order, background fills the remaining seats, and it never waits to fill the cap.
    {
        const PackedSelection s = select({ordinary(0), ordinary(1), ordinary(2), ordinary(3, false)}, 2);
        CHECK(s.seats == 0b0011u);
        CHECK(s.kind == PackedPackKind::packed);
    }
    {
        const PackedSelection s = select({ordinary(0, false), ordinary(1), ordinary(2, false), ordinary(3)}, 2);
        CHECK(s.seats == 0b1010u);
    }
    {
        // No foreground owner waits: the cap is not armed, exactly as before.
        const PackedSelection s = select({ordinary(0, false), ordinary(1, false)}, 1);
        CHECK(s.seats == 0b11u);
    }
    {
        // More seats than owners: no time is spent waiting to fill the cap.
        const PackedSelection s = select({ordinary(0), ordinary(1)}, 4);
        CHECK(s.seats == 0b11u);
    }
    {
        // A solo owner is one seat; the cap does not split or widen it.
        const PackedSelection s = select({ordinary(0), solo(1)}, 1);
        CHECK(s.seats == 0b10u);
        CHECK(s.kind == PackedPackKind::solo);
    }
    {
        // Fairness uses the ordinary composition, so the cap still applies to that pack.
        const PackedSelection s = select({solo(0), ordinary(1), ordinary(2), ordinary(3)}, 2,
                                         kPackedSoloPacksBeforeOrdinary);
        CHECK(s.kind == PackedPackKind::fairness);
        CHECK(s.seats == 0b0110u);
    }

    // Decoder interleaving: a streaming priority answer slices ordinary prompt work; no issued
    // rows means there is nothing to slice.
    {
        CHECK(packed_slice_due(true, false, 1));
        CHECK(packed_slice_due(false, true, 1));
        CHECK(!packed_slice_due(true, true, 0));
        CHECK(!packed_slice_due(false, false, 2048));
    }

    // A streaming solo answer caps each ordinary pack to the streaming quantum, split across the
    // selected owners; a pack that is already shorter is not extended.
    {
        CHECK(packed_span_rows(true, 1, 1000) == 128);
        CHECK(packed_span_rows(true, 2, 1000) == 64);
        CHECK(packed_span_rows(true, 4, 4096) == 32);
        CHECK(packed_span_rows(true, 4, 20) == 20);
        CHECK(packed_span_rows(false, 4, 1000) == 1000);
        CHECK(packed_span_rows(true, 0, 1000) == 128);
        CHECK(packed_span_rows(true, 2, 0) == 0);
    }

    // Selection is constexpr-safe for the fixed engine sizes.
    static_assert(kPrefillOwners == 4);
    static_assert(kPackedSoloPacksBeforeOrdinary == 4);

    if (failed == 0) std::printf("packed_selection_test: ok\n");
    return failed ? 1 : 0;
}
