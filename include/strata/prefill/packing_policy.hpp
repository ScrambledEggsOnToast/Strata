#pragma once

// Per-request composition policy for segmented packed prefill (#249).
//
// Packed prefill remains one engine/model allocation selected by the command verb and
// `STRATA_PACKED_PREFILL`; this header only decides which already-prepared owners share the next
// bounded pack.  An ordinary request is packing-eligible.  A request that asked for the
// latency-oriented path (`packing=solo`, frontend `strata_packing: "solo"`) never shares a pack
// with another request: it runs as the only owner of its next pack, at the next scheduling
// boundary, and an ordinary pack in flight is not interrupted (its rows are already issued).
//
// Solo traffic must not starve ordinary requests.  After `kPackedSoloPacksBeforeOrdinary`
// consecutive solo packs, one ordinary pack runs before solo resumes.  Both directions are
// deterministic: no timer, no second scheduler and no process-wide toggle per request.

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

#include "strata/prefill/segments.hpp"

namespace strata::prefill {

enum class PackingEligibility : uint8_t { eligible, solo };

// Exact, case-sensitive: an unrecognised value refuses the request rather than silently choosing
// a composition.
inline bool parse_packing_eligibility(std::string_view text, PackingEligibility& out) {
    if (text == "eligible") {
        out = PackingEligibility::eligible;
        return true;
    }
    if (text == "solo") {
        out = PackingEligibility::solo;
        return true;
    }
    return false;
}

// Consecutive solo packs allowed before a waiting ordinary owner gets one pack.
inline constexpr unsigned kPackedSoloPacksBeforeOrdinary = 4;

// One owner with prefill rows left.  `foreground` is the existing admission/decode class and is
// deliberately separate from packing eligibility, so this policy never changes dispatch priority.
struct PackedCandidate {
    int slot = 0;          // physical resident slot, 0 .. kPrefillOwners-1
    bool solo = false;
    bool foreground = true;
};

enum class PackedPackKind : uint8_t { none, packed, solo, fairness };

struct PackedSelection {
    uint32_t seats = 0;   // bit `slot` selects that physical owner
    PackedPackKind kind = PackedPackKind::none;
    unsigned solo_streak = 0;
};

// `foreground_cap` keeps the existing `STRATA_PACKED_PREFILL_FOREGROUND_CAP` rule for eligible
// owners (0 = no cap): when a foreground owner waits, take foreground seats first in candidate
// order, then background seats, and never wait to fill the cap.  A solo owner is one seat by
// definition, so the cap does not compose solo packs.
inline PackedSelection select_packed_seats(std::span<const PackedCandidate> waiting, int foreground_cap,
                                           unsigned solo_streak) {
    PackedSelection choice;
    choice.solo_streak = solo_streak;
    const size_t count = waiting.size() < size_t(kPrefillOwners) ? waiting.size() : size_t(kPrefillOwners);
    if (count == 0) return choice;

    const auto select_ordinary = [&](PackedPackKind kind) {
        uint32_t all = 0;
        bool foreground_waiting = false;
        for (size_t i = 0; i < count; ++i) {
            if (waiting[i].solo) continue;
            all |= uint32_t(1) << waiting[i].slot;
            foreground_waiting = foreground_waiting || waiting[i].foreground;
        }
        uint32_t chosen = all;
        if (foreground_cap > 0 && foreground_cap < kPrefillOwners && foreground_waiting) {
            chosen = 0;
            int seats = 0;
            for (int pass = 0; pass < 2 && seats < foreground_cap; ++pass)
                for (size_t i = 0; i < count && seats < foreground_cap; ++i) {
                    if (waiting[i].solo || waiting[i].foreground != (pass == 0)) continue;
                    chosen |= uint32_t(1) << waiting[i].slot;
                    ++seats;
                }
        }
        choice.seats = chosen;
        choice.kind = kind;
        choice.solo_streak = 0;
    };

    bool any_solo = false, any_ordinary = false;
    for (size_t i = 0; i < count; ++i) {
        any_solo = any_solo || waiting[i].solo;
        any_ordinary = any_ordinary || !waiting[i].solo;
    }
    if (!any_solo) {
        select_ordinary(PackedPackKind::packed);
        return choice;
    }
    if (solo_streak >= kPackedSoloPacksBeforeOrdinary && any_ordinary) {
        select_ordinary(PackedPackKind::fairness);
        return choice;
    }
    // One solo owner per pack, in physical-slot order (the caller passes candidates that way).
    for (size_t i = 0; i < count; ++i)
        if (waiting[i].solo) {
            choice.seats = uint32_t(1) << waiting[i].slot;
            choice.kind = PackedPackKind::solo;
            choice.solo_streak = solo_streak + 1;
            return choice;
        }
    return choice;
}

// A pack is drained early to offer a decode round when the global decoder-slice control is on or
// a priority answer is streaming, so excluding that answer from packing does not move its latency
// into longer streaming pauses.  Zero issued rows means there is nothing to slice.
inline bool packed_slice_due(bool decoder_slice, bool solo_streaming, int64_t issued_rows) {
    return issued_rows > 0 && (decoder_slice || solo_streaming);
}

} // namespace strata::prefill
