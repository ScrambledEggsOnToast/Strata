#pragma once

// Admission policy over the existing submission owners, not another owning queue.
// A missing input holds its registered FIFO position. Only successful seat grants
// advance fairness; cancellation, timeout and reservation refusal do not.
#include <cstddef>
#include <charconv>
#include <cstdint>
#include <span>

#include <string_view>
namespace strata::program {
inline constexpr size_t native_queue_limit = 256;
inline constexpr size_t native_submission_limit = native_queue_limit + 4;
inline constexpr size_t native_owner_metadata_bytes = 8192;

// A physical number is only a hint on a logical control. A well-formed stale
// number outside today's effective seats is ignored by lease matching, not an
// engine-fatal parse error that would terminate unrelated owners.
inline bool native_control_slot(std::string_view text, int& slot) {
    if (text.empty()) return false;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), slot);
    return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size() && slot >= -1;
}


struct AdmissionCandidate {
    size_t index = 0;
    uint64_t sequence = 0;
    bool foreground = true, solo = false, ready = false;
    size_t prompt_tokens = 0;
    double wait_s = 0;
};

struct NativeAdmissionPolicy {
    unsigned foreground_grants = 0, solo_grants = 0;
    size_t short_first = 0;
    double max_short_wait_s = 30;

    const AdmissionCandidate* select(std::span<const AdmissionCandidate> waiting) const {
        const AdmissionCandidate *foreground = nullptr, *background = nullptr, *solo = nullptr, *eligible = nullptr;
        const auto oldest = [](const AdmissionCandidate*& previous, const AdmissionCandidate& next) {
            if (!previous || next.sequence < previous->sequence) previous = &next;
        };
        for (const auto& next : waiting) {
            if (!next.foreground) oldest(background, next);
            else {
                oldest(foreground, next);
                oldest(next.solo ? solo : eligible, next);
            }
        }
        const AdmissionCandidate* chosen = background && (!foreground || foreground_grants >= 3) ? background :
            solo && (!eligible || solo_grants < 4) ? solo : eligible ? eligible : foreground;
        if (!chosen || !chosen->ready) return nullptr;
        if (short_first && chosen->wait_s < max_short_wait_s && chosen->prompt_tokens > short_first) {
            for (const auto& next : waiting) {
                if (next.foreground != chosen->foreground || next.solo != chosen->solo || !next.ready) continue;
                if ((next.prompt_tokens <= short_first || next.wait_s >= max_short_wait_s) &&
                    ((chosen->prompt_tokens > short_first && chosen->wait_s < max_short_wait_s) ||
                     next.sequence < chosen->sequence)) chosen = &next;
            }
        }
        return chosen;
    }
    void granted(bool foreground, bool solo) {
        foreground_grants = foreground ? (foreground_grants < 3 ? foreground_grants + 1 : 3) : 0;
        solo_grants = foreground && solo ? (solo_grants < 4 ? solo_grants + 1 : 4) : 0;
    }
};
} // namespace strata::program
