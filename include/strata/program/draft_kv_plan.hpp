// include/strata/program/draft_kv_plan.hpp - which code computes the draft layer's prompt K/V for a chunk.
//
// One GPU: the prompt path's batched pass (E-9, `Prefill::draft_kv`) tries first and the drafter's own
// per-8-row pass (`MtpDrafter::prefill`) takes what it declines.  A layer split: only the drafter's own pass
// ran, until `STRATA_SPLIT_MTP_BATCH=1` (opt-in) lets the last stage's prompt path try E-9 as well.  The
// drafter's K/V then come from Q8_1 x Q8_0 MMQ instead of its own mmvq, so its drafts can differ a little; the
// target's tokens are decided by the verify window and do not.
// Shared MTP cuts also need a prompt horizon independent of the request's private suffix.
//
// Pure (no GPU, no environment read): generate.cpp passes in what it knows.
#pragma once

#include "strata/core/draft_state.hpp"

#include <cstdint>
#include <cstdlib>

namespace strata::program::draft_kv {

/// `STRATA_SPLIT_MTP_BATCH`'s value: any whole number other than 0 turns it on; unset, empty, `0` and text
/// that is not a number leave it off.
inline bool split_env_on(const char* value) { return value != nullptr && std::atoi(value) != 0; }

struct Plan {
    bool try_batched = false;      ///< call Prefill::draft_kv for this chunk
    const char* why = nullptr;     ///< when `try_batched` is false only because of a missing prerequisite: what is missing
};

/// `use_mtp`: the engine runs a draft layer. `multi_gpu`: a layer split. `split_env`: split_env_on(...) of the
/// variable. `has_native_embed`: the GGUF-form token embedding table exists (a --native pack): the split's
/// last stage reads it, a single GPU's path can also gather from its own weights.
inline Plan plan(bool use_mtp, bool multi_gpu, bool split_env, bool has_native_embed) {
    Plan p;
    if (!use_mtp) return p;
    if (!multi_gpu) {
        p.try_batched = true;
        return p;
    }
    if (!split_env) return p;   // the default on a split: the drafter's own pass, nothing to say
    if (!has_native_embed) {
        p.why = "the split's last stage has no GGUF-form token embedding table (not --native)";
        return p;
    }
    p.try_batched = true;
    return p;
}

struct PromptHorizon {
    int64_t read_from;          ///< effective start, including a diagnostic reread from zero
    int64_t prompt_len;         ///< metadata used by the drafter's prompt-row pruning
    int64_t boundary_capture;   ///< immutable MTP R row, or -1 when no capture is needed
};

/// A shared MTP cut must retain rows without regard to a later private suffix. Its horizon
/// is pin+1 whenever this read crosses the pin; only a certified store needs the pin-1 R row.
/// DFlash keeps full-prompt metadata and has separate feature certification, not an MTP R boundary.
constexpr PromptHorizon prompt_horizon(core::DraftKind kind, int64_t prompt_len, int64_t resume,
                                      int64_t reread_to, bool shared_scope, int64_t pin,
                                      bool certified_store) {
    const int64_t read_from = reread_to > 0 ? 0 : resume;
    const bool shared_mtp_cut = kind == core::DraftKind::mtp && shared_scope && read_from < pin;
    return {read_from, shared_mtp_cut ? pin + 1 : prompt_len,
            shared_mtp_cut && certified_store ? pin - 1 : -1};
}

}  // namespace strata::program::draft_kv
