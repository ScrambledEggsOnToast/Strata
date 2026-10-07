// src/prefill/allocation.cpp - FORK (HET-017), not upstream: the admission gate's pre-allocation pricing.
//
// Why this is a separate file: the gate must price a prompt configuration BEFORE a SessionState (and before any
// device allocation) exists, while upstream's counter takes the session.  The counter therefore grew one
// session-free entry point - `Prefill::bytes_needed_from(facts)` in prefill.cpp, which `bytes_needed` also calls,
// so the two can never disagree about the allocation sequence.  What is left for this file is what the gate adds
// on top: the configuration snapshot, the loan/owned split, and the host-side classes.
//
// The rule this file exists to keep: the price is the allocation.  Device bytes come from the same counted
// sequence `init` runs, and the ring is priced at the slot count the configuration names (which
// `allocation_config` snapshots from the engine's own `ring_slots_for`), never at a literal this file carries.

#include "strata/prefill/prefill.hpp"

#include "strata/core/layout.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/prefill/moe_mmq.hpp"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <string>

namespace strata::prefill {

namespace {

/// The allocator's one rounding rule, from prefill.cpp's `Alloc::take`: every payload costs
/// ceil(payload / 256) * 256 + 256 (alignment plus guard).
uint64_t alloc_bytes(uint64_t payload) { return (payload + 511) & ~UINT64_C(255); }
uint64_t owned_bytes(uint64_t payload) {
    constexpr uint64_t granule = 2ull << 20;
    return (payload + granule - 1) / granule * granule;
}

}  // namespace

bool Prefill::allocation_config(int64_t max_cells, int64_t chunk, int kv_mode, bool kv_int8, bool kv_q4,
                                AllocationConfig& out, std::string& err, bool kv_hybrid) {
    out = {};
    err.clear();
    // The artifact's fixed geometry gives its own bounds; these are the admission-side index limits: every
    // element count below (and inside the counter) stays representable, and a caller cannot ask the gate to
    // price a context or a chunk the engine's int32 indices cannot address.
    if (max_cells <= 0 || max_cells > INT32_MAX || chunk <= 0 || chunk > INT32_MAX / 10 /* K */) {
        err = "prefill: context/chunk exceeds supported index range";
        return false;
    }
    if (kv_mode < 0 || kv_mode > 2 || (int) kv_int8 + (int) kv_q4 + (int) kv_hybrid > 1) {
        err = "prefill: invalid KV configuration";
        return false;
    }
    out.max_cells = max_cells;
    out.n_pages = (max_cells + 3) / 4;
    out.chunk = chunk;
    out.kv_mode = kv_mode;
    out.kv_int8 = kv_int8;
    out.kv_q4 = kv_q4;
    out.kv_hybrid = kv_hybrid;
    out.kv_stage_own = kv_stage_own_env();
    out.gr_unfused = gr_unfused_env();
    out.ring = (int) ring_slots_for(chunk);
    out.stager_ring = stager_ring_env();
    const char* env = std::getenv("STRATA_PREFILL_MMQ");
    out.mmq = env == nullptr || std::atoi(env) != 0;
    if (mmq::built() && !mmq::device_config(out.device_cc, out.device_sms, out.device_shared_bytes)) {
        err = "prefill: device properties unavailable for MMQ qualification";
        return false;
    }
    return true;
}

bool Prefill::stager_max_jobs(const core::ModelGeometry& g, const AllocationConfig& c, int64_t& max_jobs) {
    max_jobs = 0;
    const int64_t end = c.layer_end < 0 ? g.n_layers : c.layer_end;
    if (g.n_expert < 1 || c.layer_begin < 0 || end <= c.layer_begin || end > g.n_layers) return false;
    // The stager's claim word packs the job count into 16 bits (`(gen << 32) | (n << 16) | next`): the bound
    // must be encodable, never clamped.  The quotient check also rules out any product overflow.
    if ((uint64_t) (end - c.layer_begin) > (uint64_t) 0xffff / (uint64_t) g.n_expert) return false;
    max_jobs = (int64_t) (end - c.layer_begin) * g.n_expert;
    return true;
}

bool Prefill::allocation_needed(const core::ModelGeometry& g, const AllocationConfig& c,
                                const kernels::cpu::ExpertLayout& experts, AllocationBytes& out, std::string& err) {
    out = {};
    err.clear();
    const int64_t end = c.layer_end < 0 ? g.n_layers : c.layer_end;
    // Fail closed on anything the counter would price without complaining.  The geometry literals are the
    // artifact's fixed shape (prefill.cpp's N/HC/LR/C/ZV/HV/K); a different one is not this engine's model.
    if (g.n_embd != 2560 || g.hc != 4 || g.hc_lr != 320 || g.n_expert < 1 || g.n_expert > 512 ||
        g.n_layers < 1 || g.n_layers > 48 || g.n_ff != 640 || g.ssm_conv_channels != 10240 ||
        g.ssm_value_dim != 6144 || g.ssm_v_heads != 48 || g.ssm_state_size != 128 || g.ssm_k_heads != 16 ||
        g.ssm_d_conv != 4 || g.n_head != 24 || g.n_head_kv != 2 || g.head_dim != 256 || g.idx_q_heads != 4 ||
        g.idx_key_dim != 128 || c.max_cells <= 0 || c.max_cells > INT32_MAX || c.chunk <= 0 ||
        c.chunk > INT32_MAX / 10 || c.n_pages != (c.max_cells + 3) / 4 || c.kv_mode < 0 || c.kv_mode > 2 ||
        ((int) c.kv_int8 + (int) c.kv_q4 + (int) c.kv_hybrid > 1) || c.ring < 8 || c.ring > ring_hard_cap() || c.stager_ring < 2 ||
        c.stager_ring > 256 || c.layer_begin < 0 || c.layer_begin >= end || end > g.n_layers ||
        experts.n_layers != g.n_layers || experts.n_expert != g.n_expert ||
        (experts.native && (experts.fmt.size() != (size_t) g.n_layers ||
                            experts.bytes.size() != (size_t) g.n_layers)) ||
        experts.max_blob == 0) {
        err = "prefill: invalid allocation geometry/configuration";
        return false;
    }
    // The counter prices the pack that is loaded for the paths that read the global layout (fused/MMQ), and the
    // caller's layout for the ring's blob: the engine's own rule, so a candidate layout is priced as it would run.
    if (experts.native) {
        for (int64_t l = 0; l < g.n_layers; ++l) {
            const auto& f = experts.fmt[(size_t) l];
            if (f.n_embd != 2560 || f.n_ff != 640 || f.bytes == 0 || experts.bytes[(size_t) l] < f.bytes) {
                err = "prefill: native expert geometry/blob size mismatch";
                return false;
            }
        }
    }
    uint64_t blob = experts.native ? experts.max_blob : std::max<uint64_t>(experts.max_blob, kernels::cpu::BLOB);
    for (int64_t l = 0; experts.native && l < g.n_layers; ++l)
        blob = std::max(blob, experts.bytes[(size_t) l]);
    if (blob == 0 || blob > std::numeric_limits<uint64_t>::max() - 511) {
        err = "prefill: expert staging allocation overflow";
        return false;
    }
    // The device loan: upstream's counted init sequence, with the ring priced at the configuration's own slot
    // count (`allocation_config` took it from the engine's ring_slots_for; a caller may name a smaller ring) and
    // at this layout's blob size.  A ring whose accumulation cannot be represented is refused here, before the
    // counter can wrap it.
    const uint64_t per_slot = alloc_bytes(blob);
    if (per_slot != 0 && (uint64_t) c.ring > std::numeric_limits<uint64_t>::max() / per_slot) {
        err = "prefill: ring accumulation overflow";
        return false;
    }
    out.loanable_device = bytes_needed_from(g, c.facts(), c.chunk, c.ring, blob, c.kv_stage_own ? 1 : 0);
    if (out.loanable_device == 0 || out.loanable_device == UINT64_MAX) {
        err = "prefill: the allocation count overflowed";
        return false;
    }
    // Owned device: the token ids, the streaming identity page table, and an owned KV stage when the stage owns
    // it (the loan counter deliberately excludes the last: it is not lent).
    const uint64_t max = std::numeric_limits<uint64_t>::max();
    uint64_t owned = owned_bytes((uint64_t) c.chunk * sizeof(int32_t));
    if (c.kv_mode == 1) {
        const uint64_t table = owned_bytes((uint64_t) c.n_pages * sizeof(int32_t));
        if (owned > max - table) {
            err = "prefill: owned device count overflow";
            return false;
        }
        owned += table;
    }
    if (c.kv_stage_own) {
        const uint64_t stage = owned_stage_bytes(g, c.facts());
        if (stage == 0 || owned > max - stage) {
            err = "prefill: owned KV stage count unavailable or overflowing";
            return false;
        }
        owned += stage;
    }
    if (out.loanable_device > max - owned) {
        err = "prefill: device allocation total overflow";
        return false;
    }
    out.owned_device = owned;
    const uint64_t standalone = bytes_needed_from(g, c.facts(), c.chunk, c.ring, blob,
                                                 c.kv_stage_own ? 1 : 0, true);
    const uint64_t separate = owned - (c.kv_stage_own ? owned_stage_bytes(g, c.facts()) : 0);
    if (standalone > max - separate) {
        err = "prefill: standalone allocation count overflow"; return false;
    }
    out.standalone_device = standalone + separate;
    out.mmq_workspace = 0;
    if (c.mmq && mmq::built() &&
        !mmq::workspace_bytes(c.device_cc, c.device_sms, c.device_shared_bytes, out.mmq_workspace)) {
        err = "prefill: invalid MMQ device configuration";
        return false;
    }
    int64_t max_jobs = 0;
    if (!stager_max_jobs(g, c, max_jobs)) {
        err = "prefill: the stager's job bound exceeds the claim word's range";
        return false;
    }
    if (!host_allocation_from(g, c.facts(), c.chunk, c.stager_ring, c.mmq, max_jobs, out.host_payload,
                              out.host_dynamic)) {
        err = "prefill: host allocation arithmetic overflow";
        return false;
    }
    return true;
}

}  // namespace strata::prefill
