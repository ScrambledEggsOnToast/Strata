// CUDA-free QSA allocation arithmetic, shared by the allocator and plan partner.
#pragma once
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include <algorithm>
#include <cstdint>

namespace strata::plan {
inline constexpr int64_t kQsaResidentMinCells = 20480;
struct QsaMemoryLayout {
    int mode = 0;
    int64_t pages = 0, slots = 0, pooled_rows = 0;
};

// Match qsa_state_init: hybrid overrides other formats, but never a positive ring.
inline int qsa_state_format(bool hybrid, bool q4, bool int8, int64_t ring_cells) {
    using namespace strata::kernels;
    return hybrid && ring_cells <= 0 ? kKvHybrid : q4 ? kKvQ4 : int8 ? kKvInt8 : kKvF16;
}
// MtpDrafter::load disables hybrid and enables INT8 only when the process is hybrid.
inline int qsa_draft_format(int process_format) {
    return process_format == strata::kernels::kKvHybrid ? strata::kernels::kKvInt8 : process_format;
}

inline QsaMemoryLayout qsa_memory_layout(const strata::kernels::QsaShapes& s, int64_t cells,
                                        int64_t resident_cells, int64_t ring_cells, bool disabled = false) {
    QsaMemoryLayout p;
    p.pages = (cells + s.page_size - 1) / s.page_size;
    p.slots = p.pages;
    p.pooled_rows = cells / s.idx_block + 2;
    if (resident_cells <= 0 || ring_cells < 0 || disabled) return p;
    const int64_t window = ring_cells > 0 ? ring_cells : std::max(resident_cells, kQsaResidentMinCells);
    const int64_t slots = (window + s.page_size - 1) / s.page_size;
    if (slots < p.pages) {
        p.mode = ring_cells > 0 ? 2 : 1;
        p.slots = slots;
        if (p.mode == 2) p.pooled_rows = 2;
    }
    return p;
}

// kv_block_bytes delegates here: this is also the pinned allocator's per-page price.
inline uint64_t kv_page_bytes(const strata::kernels::QsaShapes& s, int fmt) {
    using namespace strata::kernels;
    const uint64_t rows = (uint64_t) (s.n_head_kv * s.page_size);
    if (fmt == kKvHybrid)
        return rows * (uint64_t) s.head_dim + rows * (uint64_t) (s.head_dim / KV_Q8_GROUP) * 2 +
               rows * kv_q4_bytes_per_head((int) s.head_dim);
    if (fmt == kKvQ4) return rows * kv_q4_bytes_per_head((int) s.head_dim) * 2;
    return fmt == kKvInt8 ? rows * (uint64_t) s.head_dim * 2 + rows * (uint64_t) (s.head_dim / KV_Q8_GROUP) * 2 * 2
                          : rows * (uint64_t) s.head_dim * 2 * 2;
}
inline uint64_t kv_pool_bytes(const strata::kernels::QsaShapes& s, int64_t pages, int fmt) {
    return (uint64_t) pages * kv_page_bytes(s, fmt) + (fmt == strata::kernels::kKvF16 ? 0 : 64);
}

inline uint64_t qsa_device_bytes(const strata::kernels::QsaShapes& s, int64_t cells,
                                 const QsaMemoryLayout& p, int fmt, bool with_rope, bool elastic = false) {
    uint64_t n = 0;
    if (!(elastic && p.mode == 0)) n += kv_pool_bytes(s, p.slots, fmt) + 4 * 16;
    n += (uint64_t) p.pages * 4;
    if (p.mode == 1) n += strata::kernels::kv_stream_map_bytes(p.slots) + 6 * 16;
    n += (uint64_t) (s.idx_block - 1) * s.idx_dim * 4;
    n += (uint64_t) s.idx_dim * 4;
    n += (uint64_t) p.pooled_rows * s.idx_dim * 4;
    n += 16;
    if (with_rope) n += (uint64_t) cells * (s.n_rot / 2) * 4 * 2;
    n += strata::kernels::qsa_step_bytes() + 16;
    n += (uint64_t) s.n_head * 4;
    return ((n + 15) & ~15ull) + 256;
}
inline uint64_t qsa_pinned_bytes(const strata::kernels::QsaShapes& s, const QsaMemoryLayout& p, int fmt) {
    return strata::kernels::qsa_step_bytes() + sizeof(int32_t) + (uint64_t) s.n_head * sizeof(int32_t) +
           (p.mode == 0 ? 0 : (uint64_t) p.pages * kv_page_bytes(s, fmt) + 4 * 256);
}
} // namespace strata::plan
