#pragma once

// #252: which failed capture ends serving.  A checkpoint, a parked conversation snapshot and a shared prefix are all
// optional reuse: the request that produced them is already correct without them, so a copy, allocation, validation or
// publication failure drops that capture and the engine keeps serving.  A failed device synchronisation is different:
// the GPU's stream is no longer trustworthy, so decoding on from it would serve unknown bytes and the request ends.
//
// The two outcomes are kept apart by the caller, which knows whether a device call failed; this header owns the rule and
// the one log line a drop emits.  A drop is never an ERR/FATAL protocol line, so the frontend's request terminal (and
// with it the client) is untouched.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string_view>

namespace strata::program {

enum class PreservationFailure : uint8_t {
    durability,   ///< a copy, allocation, validation or publication failure: this capture's reuse is lost
    device,       ///< a device sync/launch failed: the device state itself is no longer trustworthy
};

/// Whether that failure must end the request (and, through the request terminal, the engine).
constexpr bool preservation_ends_serving(PreservationFailure failure) noexcept {
    return failure == PreservationFailure::device;
}

/// One bounded stderr line naming the dropped capture and why.  The reason is truncated so a pathological allocator or
/// driver message cannot flood the log, and the line carries no ERR/FATAL marker.
inline void preservation_report_drop(std::FILE* out, std::string_view what, std::string_view why) noexcept {
    constexpr std::size_t kMaxReason = 192;
    if (why.size() > kMaxReason) why = why.substr(0, kMaxReason);
    std::fprintf(out, "strata serve: %.*s dropped (%.*s)\n", int(what.size()), what.data(), int(why.size()), why.data());
}

}  // namespace strata::program
