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

// A value resolved before allocation; pricing and carving never reread policy.
// Main roles share the process format. The draft keeps its existing whole-format rule.
struct QsaPlacement {
    int64_t cells = 0;
    QsaMemoryLayout kv;
    int format = strata::kernels::kKvF16;
    bool elastic = false, rotate = false;
};
struct QsaRolePlacements {
    QsaPlacement working_main, slot_main, working_draft, slot_draft;
};
inline QsaPlacement qsa_resolve_placement(const strata::kernels::QsaShapes& s, int64_t cells,
                                           int64_t resident_cells, int64_t ring_cells, int fmt,
                                           bool disabled = false, bool elastic = false, bool rotate = false) {
    return {cells, qsa_memory_layout(s, cells, resident_cells, ring_cells, disabled), fmt, elastic, rotate};
}
inline bool qsa_placement_matches(const strata::kernels::QsaShapes& s, int64_t cells, const QsaPlacement& p) {
    if (cells <= 0 || cells > INT32_MAX || s.page_size <= 0 || s.idx_block <= 0 ||
        p.cells != cells || p.format < strata::kernels::kKvF16 || p.format > strata::kernels::kKvHybrid)
        return false;
    const int64_t pages = (cells + s.page_size - 1) / s.page_size;
    return p.kv.pages == pages && p.kv.slots > 0 && p.kv.slots <= pages &&
           p.kv.mode >= 0 && p.kv.mode <= 2 &&
           (p.kv.mode == 0 ? p.kv.slots == pages : p.kv.slots < pages) &&
           p.kv.pooled_rows == (p.kv.mode == 2 ? 2 : cells / s.idx_block + 2);
}

inline int64_t qsa_draft_live_cells(int64_t cells, int64_t window, int64_t max_t) {
    return window > 0 && window < cells ? std::min(cells, window + 4 * max_t + 64) : cells;
}
// slot_window == -1 is absence, not explicit zero: preserve the global policy.
// A positive draft override is a ring of that exact window, not a sparse stream map.
inline QsaRolePlacements qsa_resolve_roles(const strata::kernels::QsaShapes& s, int64_t cells, int fmt,
                                           int64_t resident_cells, int64_t slot_window,
                                           int64_t draft_window, int64_t max_t,
                                           bool main_disabled = false, bool ring_disabled = false,
                                           bool elastic = false, bool rotate = false) {
    QsaRolePlacements p;
    p.working_main = qsa_resolve_placement(s, cells, resident_cells, 0, fmt, main_disabled, elastic, rotate);
    p.slot_main = slot_window < 0 ? p.working_main :
        qsa_resolve_placement(s, cells, slot_window, 0, fmt, false, false, rotate);
    const int64_t ring = draft_window > 0 && draft_window < cells ? draft_window + 4 * max_t + 64 : 0;
    const int draft_fmt = qsa_draft_format(fmt);
    p.working_draft = qsa_resolve_placement(s, cells, resident_cells, ring, draft_fmt,
                                            ring > 0 ? ring_disabled : main_disabled, elastic, rotate);
    p.slot_draft = slot_window < 0 ? p.working_draft :
        qsa_resolve_placement(s, cells, slot_window, slot_window > 0 ? slot_window : -1,
                              draft_fmt, false, false, rotate);
    return p;
}
// This is the resolve call's capacity contract, not the engine's 20480-cell floor
// and not the CUDA implementation's RT=1024 clock-sweep guard.
inline int64_t qsa_stream_call_min_cells(const strata::kernels::QsaShapes& s, int64_t n_q, int64_t cap) {
    return n_q * (cap / s.page_size + 2) * s.page_size;
}
inline const char* qsa_slot_window_error(const strata::kernels::QsaShapes& s, int64_t window,
                                         int64_t n_q, int64_t cap, int64_t draft_live_cells = 0) {
    if (window < 0 || window > INT32_MAX) return "--slot-kv-window must be in [0, INT32_MAX]";
    if (window == 0) return nullptr;
    if (window < kQsaResidentMinCells) return "--slot-kv-window is below the engine floor (20480 cells)";
    if ((window + s.page_size - 1) / s.page_size < n_q * (cap / s.page_size + 2))
        return "--slot-kv-window is below the streamed call-capacity floor";
    if ((window + s.page_size - 1) / s.page_size < (draft_live_cells + s.page_size - 1) / s.page_size)
        return "--slot-kv-window is below the MTP live-window capacity (--mtp-window plus lookahead)";
    return nullptr;
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

// Offsets used by the real initializer as well as the hermetic carve test.
// The historical price includes alignment/slack. Reserve that trailing padding too,
// so the returned extent is the allocation price, without changing legacy prices.
struct QsaArenaCursor {
    uint64_t capacity, used = 0;
    bool valid = true;
    uint64_t take_bytes(uint64_t bytes) {
        if (!valid || used > capacity || used > UINT64_MAX - 15 ||
            bytes > capacity - used || bytes > UINT64_MAX - used - 15) {
            valid = false;
            return 0;
        }
        const uint64_t offset = used, next = (used + bytes + 15) & ~15ull;
        if (next > capacity) { valid = false; return 0; }
        used = next;
        return offset;
    }
    template <typename T> uint64_t take(uint64_t count) {
        if (count > UINT64_MAX / sizeof(T)) { valid = false; return 0; }
        return take_bytes(count * sizeof(T));
    }
};
struct QsaPoolLayout {
    uint64_t k = 0, v = 0, k_scale = 0, v_scale = 0;
};
inline QsaPoolLayout qsa_pool_layout(const strata::kernels::QsaShapes& s, int64_t pages, int fmt,
                                     QsaArenaCursor& c) {
    using namespace strata::kernels;
    const uint64_t rows = (uint64_t) pages * s.n_head_kv * s.page_size;
    QsaPoolLayout p;
    if (fmt == kKvHybrid) {
        p.k = c.take<int8_t>(rows * s.head_dim);
        p.k_scale = c.take<uint16_t>(rows * (s.head_dim / KV_Q8_GROUP));
        p.v = c.take<uint8_t>(rows * kv_q4_bytes_per_head((int) s.head_dim));
    } else if (fmt == kKvQ4) {
        p.k = c.take<uint8_t>(rows * kv_q4_bytes_per_head((int) s.head_dim));
        p.v = c.take<uint8_t>(rows * kv_q4_bytes_per_head((int) s.head_dim));
    } else if (fmt == kKvInt8) {
        p.k = c.take<int8_t>(rows * s.head_dim);
        p.v = c.take<int8_t>(rows * s.head_dim);
        p.k_scale = c.take<uint16_t>(rows * (s.head_dim / KV_Q8_GROUP));
        p.v_scale = c.take<uint16_t>(rows * (s.head_dim / KV_Q8_GROUP));
    } else {
        p.k = c.take<uint16_t>(rows * s.head_dim);
        p.v = c.take<uint16_t>(rows * s.head_dim);
    }
    return p;
}
struct QsaArenaLayout {
    QsaPoolLayout pool;
    uint64_t page_table = 0, slot_block = 0, slot_stamp = 0, slot_ref = 0, miss_block = 0, miss_slot = 0, ctl = 0;
    uint64_t idx_tail = 0, idx_dead = 0, idx_pooled = 0, idx_block_pos = 0, cos_tab = 0, sin_tab = 0;
    uint64_t step = 0, attention_status = 0, pos_dev = 0, payload_extent = 0, extent = 0;
    bool valid = false;
};
inline QsaArenaLayout qsa_arena_layout(const strata::kernels::QsaShapes& s, const QsaPlacement& p,
                                       bool with_rope, uint64_t capacity = UINT64_MAX) {
    QsaArenaLayout a;
    if (!qsa_placement_matches(s, p.cells, p)) return a;
    QsaArenaCursor c{capacity};
    if (!(p.elastic && p.kv.mode == 0)) a.pool = qsa_pool_layout(s, p.kv.slots, p.format, c);
    a.page_table = c.take<int32_t>((uint64_t) p.kv.pages);
    if (p.kv.mode == 1) {
        a.slot_block = c.take<int32_t>((uint64_t) p.kv.slots);
        a.slot_stamp = c.take<int32_t>((uint64_t) p.kv.slots);
        a.slot_ref = c.take<int32_t>((uint64_t) p.kv.slots);
        a.miss_block = c.take<int32_t>((uint64_t) p.kv.slots);
        a.miss_slot = c.take<int32_t>((uint64_t) p.kv.slots);
        a.ctl = c.take<int32_t>((uint64_t) strata::kernels::kKvCtlInts);
    }
    a.idx_tail = c.take<float>((uint64_t) (s.idx_block - 1) * s.idx_dim);
    a.idx_dead = c.take<float>((uint64_t) s.idx_dim);
    a.idx_pooled = c.take<float>((uint64_t) p.kv.pooled_rows * s.idx_dim);
    a.idx_block_pos = c.take<int32_t>(1);
    if (with_rope) {
        a.cos_tab = c.take<float>((uint64_t) p.cells * (s.n_rot / 2));
        a.sin_tab = c.take<float>((uint64_t) p.cells * (s.n_rot / 2));
    }
    a.step = c.take<int32_t>((uint64_t) strata::kernels::kStepCount);
    a.attention_status = c.take<int32_t>(1);
    a.pos_dev = c.take<int32_t>((uint64_t) s.n_head);
    a.payload_extent = c.used;
    const uint64_t price = qsa_device_bytes(s, p.cells, p.kv, p.format, with_rope, p.elastic);
    a.valid = c.valid && c.used <= price && price <= capacity;
    if (a.valid) a.extent = price; // reserve the legacy trailing alignment/slack, not an unpriced carve
    return a;
}
} // namespace strata::plan
