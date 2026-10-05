#pragma once

#include "strata/core/conversation_snapshot.hpp"
#include "strata/core/on_device.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/qsa.hpp"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdint>
#include <string>
#include <cstdlib>

namespace strata::program {

struct SessionFingerprint {
    static constexpr uint64_t basis = 1469598103934665603ull;
    uint64_t gdn = basis, ple = basis, tail = basis, pooled = basis;
    uint64_t kv = basis, draft = basis, stale = basis, dead = basis, pooled_full = basis;
};

inline uint64_t fingerprint_word(uint64_t h, uint64_t word) {
    for (int i = 0; i < 8; ++i) { h ^= (word >> (i * 8)) & 255; h *= 1099511628211ull; }
    return h;
}

inline uint64_t fingerprint_ple(const SessionFingerprint& h, const core::SessionState& session) {
    return fingerprint_word(fingerprint_word(h.ple, (uint32_t) session.ple_prev[0]),
                            (uint32_t) session.ple_prev[1]);
}

inline uint64_t fingerprint_kv(const SessionFingerprint& h) {
    return fingerprint_word(fingerprint_word(fingerprint_word(fingerprint_word(h.kv, h.tail), h.dead),
                                             h.pooled), h.pooled_full);
}

// Diagnostic only. Hash authoritative logical cells, excluding speculative/stale
// cells from the equality contract. One fixed 64-KiB transfer buffer, no allocation.
inline bool session_fingerprint(const core::SessionState& session, const core::QsaState& draft,
                                const core::ModelGeometry& g, int64_t cells,
                                SessionFingerprint& out, std::string& err) {
    const core::OnDevice on(session.allocation_device);
    if (cells < 0 || cells > session.max_cells || (draft.max_cells > 0 && cells > draft.max_cells) ||
        cudaDeviceSynchronize() != cudaSuccess) {
        err = "state fingerprint: invalid extent or device drain failed"; return false;
    }
    core::ConversationStateSizes sizes;
    if (!core::conversation_session_sizes(g, session, sizes, err)) return false;
    const auto shape = kernels::qsa_real_shapes();
    std::array<uint8_t, 65536> buffer;
    bool ok = true;
    auto bytes = [&](const void* p, size_t n, uint64_t h) {
        if (n && !p) { ok = false; return h; }
        for (size_t off = 0; ok && off < n;) {
            const size_t take = std::min(buffer.size(), n - off);
            if (cudaMemcpy(buffer.data(), static_cast<const uint8_t*>(p) + off, take,
                           cudaMemcpyDefault) != cudaSuccess) { ok = false; break; }
            for (size_t i = 0; i < take; ++i) { h ^= buffer[i]; h *= 1099511628211ull; }
            off += take;
        }
        return h;
    };
    auto pool_cells = [&](const void* pool, int64_t width, int64_t first, int64_t end, uint64_t h) {
        const int64_t page = shape.page_size;
        for (int64_t pg = first / page; pg * page < end; ++pg)
            for (int64_t head = 0; head < g.n_head_kv; ++head) {
                const int64_t a = std::max(first, pg * page) - pg * page;
                const int64_t b = std::min(end, (pg + 1) * page) - pg * page;
                const size_t off = (size_t) (((pg * g.n_head_kv + head) * page + a) * width);
                h = bytes(static_cast<const uint8_t*>(pool) + off, (size_t) ((b - a) * width), h);
            }
        return h;
    };
    auto kv = [&](const core::QsaState& state, int64_t first, int64_t end, uint64_t h) {
        const bool host = state.kv_mode != 0;
        struct Pool { const void* p; int64_t width; };
        std::array<Pool, 4> pools{};
        int n = 2;
        const int64_t q4 = (int64_t) kernels::kv_q4_bytes_per_head((int) g.head_dim);
        const int64_t scale = (g.head_dim / 64) * 2;
        if (state.kv_q4) {
            pools[0] = {host ? state.host.k_q4 : state.k_q4, q4};
            pools[1] = {host ? state.host.v_q4 : state.v_q4, q4};
        } else if (state.kv_hybrid) {
            pools[0] = {host ? state.host.k_q : state.k_q, g.head_dim};
            pools[1] = {host ? state.host.v_q4 : state.v_q4, q4};
            pools[2] = {host ? state.host.k_scale : state.k_scale, scale}; n = 3;
        } else if (state.kv_int8) {
            pools[0] = {host ? state.host.k_q : state.k_q, g.head_dim};
            pools[1] = {host ? state.host.v_q : state.v_q, g.head_dim};
            pools[2] = {host ? state.host.k_scale : state.k_scale, scale};
            pools[3] = {host ? state.host.v_scale : state.v_scale, scale}; n = 4;
        } else {
            pools[0] = {host ? state.host.k_pool : state.k_pool, g.head_dim * 2};
            pools[1] = {host ? state.host.v_pool : state.v_pool, g.head_dim * 2};
        }
        for (int i = 0; i < n; ++i) {
            if (!pools[i].p && end > first) { ok = false; break; }
            h = pool_cells(pools[i].p, pools[i].width, first, end, h);
        }
        return h;
    };
    out = {};
    out.gdn = bytes(session.gdn_state, sizes.gdn, out.gdn);
    if (std::getenv("STRATA_STATE_HASH_GDN") && session.gdn_alloc > 0) {
        std::fprintf(stderr, "strata serve: STATE_HASH_GDN ");
        const size_t per = sizes.gdn / (size_t) session.gdn_alloc;
        for (int64_t i = 0; i < session.gdn_alloc; ++i)
            std::fprintf(stderr, "%04llx ", (unsigned long long) (bytes(
                (const uint8_t*) session.gdn_state + i * per, per, SessionFingerprint::basis) & 0xffff));
        std::fprintf(stderr, "\n");
    }
    out.ple = bytes(session.ple_hist, session.ple_hist ? sizes.ple : 0, out.ple);
    const int64_t end = std::min<int64_t>(((cells + shape.page_size - 1) / shape.page_size) * shape.page_size,
                                        session.max_cells);
    for (int64_t j = 0; j < session.qsa_alloc; ++j) {
        const auto& state = session.qsa_states[session.qsa_ord0 + j];
        out.tail = bytes(state.idx_tail, sizes.tail, out.tail);
        out.dead = bytes(state.idx_dead, sizes.dead, out.dead);
        out.dead = bytes(state.idx_block_pos, sizeof(int32_t), out.dead);
        out.pooled = bytes(state.idx_pooled, (size_t) (cells / shape.idx_block) * g.idx_key_dim * 4, out.pooled);
        out.pooled_full = bytes(state.idx_pooled, (size_t) (cells > 0 ? cells / shape.idx_block + 1 : 0) *
                               g.idx_key_dim * 4, out.pooled_full);
        out.kv = kv(state, 0, cells, out.kv);
        out.stale = kv(state, cells, end, out.stale);
    }
    if (draft.max_cells > 0) out.draft = kv(draft, 0, cells, out.draft);
    if (!ok) err = "state fingerprint: device read failed";
    return ok;
}
} // namespace strata::program
