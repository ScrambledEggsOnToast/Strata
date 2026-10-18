#include "strata/core/conversation_snapshot.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "conversation_checked.hpp"
#include "strata/core/admission_copy_observer.hpp"
#include "strata/plan/qsa_memory.hpp"
#include "conversation_copy.hpp"
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>

namespace strata::core {
namespace {
struct Layout {
    int format;
    int64_t cells, pooled_rows, page_size;
    size_t data, scales, pooled, value_data, value_scales;
};
static_assert(sizeof(Layout) <= 80, "direct copy layout must remain fixed-size metadata");

bool valid_extent(const QsaState& st, int64_t upto, std::string& error) {
    const int64_t page = strata::kernels::qsa_real_shapes().page_size;
    if (upto < 0 || upto > st.max_cells || upto > std::numeric_limits<int64_t>::max() - (page - 1)) {
        error = "conversation snapshot: invalid K/V extent";
        return false;
    }
    return true;
}

bool layout(const QsaState& st, const ModelGeometry& g, int64_t upto, bool index, Layout& l, std::string& error) {
    if (!valid_extent(st, upto, error)) return false;
    if (st.kv_hybrid && (st.kv_q4 || st.kv_int8)) {
        error = "conversation snapshot: hybrid K8V4 needs distinct format flags";
        return false;
    }
    auto shape = strata::kernels::qsa_real_shapes();
    shape.n_head_kv = g.n_head_kv;
    shape.head_dim = g.head_dim;
    shape.idx_dim = g.idx_key_dim;
    const int format = st.kv_hybrid ? strata::kernels::kKvHybrid : qsa_kv_format(st);
    strata::plan::QsaImageBytes bytes;
    if (!strata::plan::qsa_image_bytes(shape, upto, format, index, bytes) || bytes.bytes > SIZE_MAX) {
        error = "conversation snapshot: invalid K/V geometry or byte count overflow";
        return false;
    }
    // Format 3 identifies snapshot K8V4 and the block movers' three-run hybrid layout. Rotated INT8 K/V (#293,
    // STRATA_KV_ROT=1) is another format too (+16): its bytes mean nothing to a state that does not rotate, so a
    // snapshot or prompt checkpoint taken with the rotation never restores into one without it, nor the reverse.
    // (Q4_0 is always rotated: nothing to tell apart; without the rotation the format is the one it always was.)
    const int rotated = st.kv_rot && !st.kv_q4 && !st.kv_hybrid ? 16 : 0;
    l = {format + rotated, (int64_t) bytes.cells, (int64_t) bytes.pooled_rows, shape.page_size,
         (size_t) bytes.k, (size_t) bytes.k_scale, (size_t) bytes.pooled, (size_t) bytes.v, (size_t) bytes.v_scale};
    return true;
}

std::array<void*, 5> pools(const QsaState& st, bool resident = false) {
    const bool host = st.kv_mode != 0 && !resident;
    if (st.kv_hybrid)
        return {host ? st.host.k_q : st.k_q, host ? st.host.v_q4 : st.v_q4, host ? st.host.k_scale : st.k_scale,
                nullptr, st.idx_pooled};
    if (st.kv_q4)
        return {host ? st.host.k_q4 : st.k_q4, host ? st.host.v_q4 : st.v_q4, nullptr, nullptr, st.idx_pooled};
    if (st.kv_int8)
        return {host ? st.host.k_q : st.k_q, host ? st.host.v_q : st.v_q,
                host ? st.host.k_scale : st.k_scale, host ? st.host.v_scale : st.v_scale, st.idx_pooled};
    return {host ? st.host.k_pool : st.k_pool, host ? st.host.v_pool : st.v_pool, nullptr, nullptr, st.idx_pooled};
}

bool valid(const QsaState& st, const Layout& l, int64_t upto, std::string& error) {
    if (upto < 0 || upto > st.max_cells || st.n_pages < 0 || l.cells / l.page_size > st.n_pages ||
        st.kv_mode < 0 || st.kv_mode > 2 || st.n_slots <= 0 ||
        (st.kv_mode == 0 && st.n_slots < st.n_pages) ||
        l.pooled_rows > st.idx_pooled_rows || (st.kv_mode != 0 && !st.host.present())) {
        error = "conversation snapshot: invalid K/V extent or missing authoritative host pool";
        return false;
    }
    if (st.kv_mode == 1) {
        const auto& m = st.map;
        if (m.n_blocks != st.n_pages || m.n_slots != st.n_slots || !m.page_table || !m.slot_block ||
            !m.slot_stamp || !m.slot_ref || !m.ctl || !m.miss_block || !m.miss_slot) {
            error = "conversation snapshot: invalid streaming map";
            return false;
        }
    }
    if (st.kv_mode == 2 && (!st.page_table ||
        (st.kv_hybrid ? !st.k_q || !st.v_q4 || !st.k_scale :
         st.kv_q4 ? !st.k_q4 || !st.v_q4 : st.kv_int8 ? !st.k_q || !st.v_q || !st.k_scale || !st.v_scale
                                                                   : !st.k_pool || !st.v_pool))) {
        error = "conversation snapshot: invalid draft ring";
        return false;
    }
    const auto authoritative = pools(st);
    const std::array<size_t,5> sizes = {l.data, l.value_data, l.scales, l.value_scales, l.pooled};
    for (size_t i = 0; i < sizes.size(); ++i)
        if (sizes[i] && !authoritative[i]) {
            error = "conversation snapshot: missing authoritative state buffer";
            return false;
        }
    return true;
}

bool transfer(void* dst, const void* src, size_t n, std::string& error, admission_copy::Part part,
              admission_copy::Direction direction, bool queued = false) {
    if (!n) return true;
    if (!src || !dst) { error = "conversation snapshot: missing state buffer"; return false; }
    // Default handles both device allocations and device-mapped host pool aliases.
    if (queued) {
        const cudaError_t e = conversation_detail::copy_nonblocking(dst, src, n, true);
        if (e == cudaSuccess) return true;
        error = std::string("conversation snapshot copy: ") + cudaGetErrorString(e);
        return false;
    }
    admission_copy::Transfer observed(part, direction, n);
    const cudaError_t e = conversation_detail::copy_nonblocking(dst, src, n);
    observed.finish(e == cudaSuccess);
    if (e == cudaSuccess) return true;
    error = std::string("conversation snapshot copy: ") + cudaGetErrorString(e);
    return false;
}
bool restore_residency(const QsaState& st, const ModelGeometry& g, int64_t upto, std::string& error,
                       bool queued = false) {
    // VRAM slots still contain the outgoing conversation. Resolve must refill
    // them from the restored authoritative pools before any attention reads.
    if (st.kv_mode == 1) strata::kernels::kv_stream_reset(st.map, nullptr);
    if (st.kv_mode == 2 && upto > 0) {
        auto shapes = strata::kernels::qsa_real_shapes();
        shapes.n_head_kv = g.n_head_kv; shapes.head_dim = g.head_dim;
        const int64_t end = (upto + shapes.page_size - 1) / shapes.page_size;
        if (!strata::kernels::kv_ring_restore(qsa_attn_pools(st), st.host, qsa_kv_format(st),
                                             std::max<int64_t>(0, end - st.n_slots), end, st.n_slots,
                                             shapes, nullptr, error)) return false;
    }
    auto status = cudaGetLastError();
    if (status != cudaSuccess) {
        error = std::string("conversation snapshot residency restore: ") + cudaGetErrorString(status);
        return false;
    }
    // Repairs use stream 0, which a nonblocking compute/copy stream does not
    // join. Complete them before returning or publishing the restored owner,
    // without stalling unrelated nonblocking work.
    if (!queued && st.kv_mode != 0) {
        status = cudaStreamSynchronize((cudaStream_t) 0);
        if (status != cudaSuccess) {
            error = std::string("conversation snapshot residency synchronize: ") + cudaGetErrorString(status);
            return false;
        }
    }
    return true;
}

bool copy_layout(const QsaState& from, const QsaState& to, const ModelGeometry& g,
                 int64_t upto, bool index, Layout& source, std::string& error) {
    Layout target{};
    if (!layout(from, g, upto, index, source, error) || !valid(from, source, upto, error) ||
        !layout(to, g, upto, index, target, error) || !valid(to, target, upto, error)) return false;
    if (source.format != target.format || source.cells != target.cells ||
        source.page_size != target.page_size || source.pooled_rows != target.pooled_rows ||
        source.data != target.data || source.value_data != target.value_data ||
        source.scales != target.scales || source.value_scales != target.value_scales ||
        source.pooled != target.pooled) {
        error = "resident direct copy: incompatible K/V layouts";
        return false;
    }
    return true;
}

bool copy_payload(const QsaState& from, const QsaState& to, const ModelGeometry& g,
                  int64_t upto, const Layout& l, std::string& error, bool queued = false) {
    const auto src = pools(from), dst = pools(to);
    const std::array<size_t, 5> sizes = {l.data, l.value_data, l.scales, l.value_scales, l.pooled};
    for (size_t i = 0; i < sizes.size(); ++i) {
        // Keep the snapshot path's bounded transfer/progress cadence without
        // allocating its segments. The destination is always the real owner.
        for (size_t at = 0; at < sizes[i];) {
            const size_t bytes = std::min(ConversationBuffer::segment_bytes, sizes[i] - at);
            if (!transfer(static_cast<uint8_t*>(dst[i]) + at, static_cast<const uint8_t*>(src[i]) + at,
                          bytes, error, static_cast<admission_copy::Part>(admission_copy::key + i),
                          admission_copy::direct, queued)) return false;
            at += bytes;
        }
    }
    if (queued && to.kv_mode != 0) {
        const cudaError_t e = conversation_detail::order_after_queued();
        if (e != cudaSuccess) {
            error = std::string("conversation snapshot direct copy: ") + cudaGetErrorString(e); return false;
        }
    }
    return restore_residency(to, g, upto, error, queued);
}

} // namespace

bool resident_direct_copy_enabled() {
    // Freeze once, before admission. Resident/MTP handoffs must never switch to
    // the allocating snapshot branch after a direct-copy envelope was priced.
    static const bool enabled = [] {
        const char* value = std::getenv("STRATA_RESIDENT_DIRECT_COPY");
        return value && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}

bool conversation_kv_copy_validate(const QsaState& from, const QsaState& to, const ModelGeometry& g,
                                   int64_t upto, bool index, std::string& error) {
    Layout source{};
    return copy_layout(from, to, g, upto, index, source, error);
}

bool conversation_kv_copy(const QsaState& from, const QsaState& to, const ModelGeometry& g,
                          int64_t upto, bool index, std::string& error, bool queued) {
    Layout source{};
    return copy_layout(from, to, g, upto, index, source, error) &&
           copy_payload(from, to, g, upto, source, error, queued);
}

bool conversation_detail::copy_prevalidated_kv(const QsaState& from, const QsaState& to, const ModelGeometry& g,
                                              int64_t upto, bool index, std::string& error) {
    // The session already validated every layer before its entry fence. Rebuild
    // only this layer's byte counts, without a context-sized array of layouts.
    Layout source{};
    return layout(from, g, upto, index, source, error) && copy_payload(from, to, g, upto, source, error);
}

bool conversation_kv_capture_bytes(const ConversationKv& image, const QsaState& st, const ModelGeometry& g,
                                   int64_t upto, bool index, size_t& bytes, std::string& error) {
    Layout l{};
    if (!layout(st, g, upto, index, l, error) || !valid(st, l, upto, error)) return false;
    const std::array<size_t,5> sizes = {l.data, l.value_data, l.scales, l.value_scales, l.pooled};
    const std::array<const ConversationBuffer*,5> buffers = {&image.k, &image.v, &image.k_scale, &image.v_scale, &image.pooled};
    bytes = 0;
    for (size_t i = 0; i < buffers.size(); ++i) {
        const size_t n = buffers[i]->allocation_peak(sizes[i]);
        if (n == SIZE_MAX || !conversation_detail::add(bytes, n)) {
            error = "conversation snapshot: segmented K/V allocation overflow";
            return false;
        }
    }
    return true;
}

size_t conversation_kv_bytes(const QsaState& st, const ModelGeometry& g, int64_t upto, bool index) {
    size_t bytes = 0;
    std::string error;
    return conversation_kv_capture_bytes({}, st, g, upto, index, bytes, error) ? bytes : 0;
}

bool conversation_kv_save(ConversationKv& image, const QsaState& st, const ModelGeometry& g,
                          int64_t upto, bool index, std::string& error,
                          int64_t unchanged_tokens, size_t* reused_bytes, std::array<uint64_t, 5>* retained) {
    Layout l{};
    if (!layout(st, g, upto, index, l, error)) return false;
    if (!valid(st, l, upto, error)) return false;
    if (unchanged_tokens < 0 || unchanged_tokens > upto || unchanged_tokens > image.cells) {
        error = "conversation snapshot: invalid unchanged prefix";
        return false;
    }
    if (retained) *retained = {};
    const int64_t whole_cells = (unchanged_tokens / l.page_size) * l.page_size;
    image.format = l.format;
    image.cells = l.cells;
    image.heads = g.n_head_kv;
    image.head_dim = g.head_dim;
    image.page_size = l.page_size;
    image.pooled_rows = l.pooled_rows;
    image.idx_dim = g.idx_key_dim;
    const auto src = pools(st);
    const std::array<size_t,5> sizes = {l.data, l.value_data, l.scales, l.value_scales, l.pooled};
    const std::array<ConversationBuffer*,5> dst = {&image.k, &image.v, &image.k_scale, &image.v_scale, &image.pooled};
    for (size_t i = 0; i < dst.size(); ++i) {
        // Recopy the partial page and the indexer's moving spare row. Completed
        // pages/rows strictly before the first rewritten token remain identical.
        const size_t keep = i == 4 ? (index ? size_t(unchanged_tokens / strata::kernels::qsa_real_shapes().idx_block) * g.idx_key_dim * 4 : 0)
                                  : l.cells ? (sizes[i] / size_t(l.cells)) * size_t(whole_cells) : 0;
        if (keep > dst[i]->size()) { error = "conversation snapshot: missing reusable prefix"; return false; }
        dst[i]->resize(sizes[i]);
        if (!dst[i]->visit(keep, sizes[i] - keep, [&](uint8_t* p, size_t n, size_t at) {
                return transfer(p, src[i] ? static_cast<const uint8_t*>(src[i]) + at : nullptr, n, error, static_cast<admission_copy::Part>(admission_copy::key + i), admission_copy::capture);
            })) return false;
        if (reused_bytes) *reused_bytes += keep;
        // Exactly the bytes this capture did not write: at this point they are still the retained image's own bytes
        // (the same allocation, or its detach-free copy), so a caller may offer them as immutable reuse intervals.
        if (retained) (*retained)[i] = uint64_t(keep);
    }
    return true;
}

namespace {
// Authoritative K/V and scale runs all have the same page/head/cell ordering,
// but different row widths (including hybrid K8/V4).
template<class Image, class Fn>
bool inactive_rows(Image& image, int64_t cells, int64_t begin, int64_t end, Fn fn) {
    if (cells <= 0 || image.page_size <= 0 || image.heads <= 0 ||
        cells % image.page_size != 0 || uint64_t(cells) > SIZE_MAX / uint64_t(image.heads)) return false;
    const size_t rows = size_t(cells) * size_t(image.heads);
    const std::array<decltype(&image.k), 4> buffers = {&image.k, &image.v, &image.k_scale, &image.v_scale};
    for (auto* buffer : buffers) {
        if (buffer->empty()) continue;
        if (buffer->size() % rows != 0) return false;
        const size_t width = buffer->size() / rows;
        const std::array<std::pair<int64_t, int64_t>, 2> inactive{{{0, begin}, {end, cells}}};
        for (const auto interval : inactive) {
            if (interval.first == interval.second) continue;
            for (int64_t page = interval.first / image.page_size;
                 page <= (interval.second - 1) / image.page_size; ++page) {
                const int64_t first = std::max(interval.first, page * image.page_size) - page * image.page_size;
                const int64_t last = std::min(interval.second, (page + 1) * image.page_size) - page * image.page_size;
                for (int64_t head = 0; head < image.heads; ++head) {
                    const size_t offset = ((size_t(page) * size_t(image.heads) + size_t(head)) *
                                           size_t(image.page_size) + size_t(first)) * width;
                    if (!fn(*buffer, offset, size_t(last - first) * width)) return false;
                }
            }
        }
    }
    return true;
}
}

bool conversation_kv_clear_inactive(ConversationKv& image, int64_t tokens, int64_t begin, int64_t end,
                                    std::string& error) {
    if (tokens <= 0 || image.page_size <= 0 || tokens > INT64_MAX - (image.page_size - 1) ||
        begin < 0 || end < begin || end > tokens ||
        image.cells != (tokens + image.page_size - 1) / image.page_size * image.page_size ||
        !inactive_rows(image, image.cells, begin, end, [](auto& buffer, size_t offset, size_t bytes) {
            return buffer.visit(offset, bytes, [](uint8_t* p, size_t n, size_t) {
                std::memset(p, 0, n);
                return true;
            });
        })) {
        error = "conversation snapshot: invalid exact-cut K/V extent";
        return false;
    }
    return true;
}

bool conversation_kv_validate(const ConversationKv& image, const QsaState& st, const ModelGeometry& g,
                              int64_t upto, bool index, ConversationSnapshotExpectation expectation,
                              std::string& error) {
    Layout l{};
    if (!layout(st, g, upto, index, l, error)) return false;
    if (!valid(st, l, upto, error)) return false;
    const std::array<size_t,5> sizes = {l.data, l.value_data, l.scales, l.value_scales, l.pooled};
    const std::array<const ConversationBuffer*,5> src = {&image.k, &image.v, &image.k_scale, &image.v_scale, &image.pooled};
    // An authorized shared mount chooses this expectation, never image.cells.
    // Its payload is rounded to N but its draft owns exactly [0,N-1): row N-1
    // consumes the first private suffix token and cannot be part of the record.
    const bool exact = expectation == ConversationSnapshotExpectation::exact_cut;
    if (exact && (upto <= 0 || (!index && image.cells != upto - 1))) {
        error = "conversation snapshot: shared draft extent must equal N-1";
        return false;
    }
    // Keep the ordinary/private interpretation of older shortened draft images
    // separate; it cannot downgrade an explicit shared expectation.
    const bool exact_draft = !index && image.cells >= 0 && image.cells < upto;
    if (image.format != l.format || (!exact_draft && image.cells != l.cells) || image.heads != g.n_head_kv ||
        image.head_dim != g.head_dim || image.page_size != l.page_size || image.pooled_rows != l.pooled_rows ||
        image.idx_dim != g.idx_key_dim) {
        error = "conversation snapshot: incompatible K/V geometry";
        return false;
    }
    for (size_t i = 0; i < src.size(); ++i)
        if (src[i]->size() != sizes[i]) { error = "conversation snapshot: invalid K/V payload size"; return false; }
    const int64_t retained_end = exact ? (index ? upto : upto - 1) : image.cells;
    if ((exact || exact_draft) && !inactive_rows(image, l.cells, 0, retained_end,
        [](const auto& buffer, size_t offset, size_t bytes) {
            return buffer.visit(offset, bytes, [](const uint8_t* p, size_t n, size_t) {
                return std::all_of(p, p + n, [](uint8_t b) { return b == 0; });
            });
        })) {
        error = "conversation snapshot: exact cut contains excluded K/V bytes";
        return false;
    }
    const auto dst = pools(st);
    for (size_t i = 0; i < src.size(); ++i)
        if (sizes[i] && !dst[i]) { error = "conversation snapshot: missing target state buffer"; return false; }
    return true;
}

bool conversation_kv_restore(const ConversationKv& image, const QsaState& st, const ModelGeometry& g,
                             int64_t upto, bool index, ConversationSnapshotExpectation expectation,
                             std::string& error, int64_t captured_tokens) {
    if (captured_tokens < 0) captured_tokens = upto;
    if (upto < 0 || upto > captured_tokens ||
        !conversation_kv_validate(image, st, g, captured_tokens, index, expectation, error)) return false;
    Layout cut{};
    if (!layout(st, g, upto, index, cut, error) || !valid(st, cut, upto, error)) return false;
    const std::array<size_t, 5> sizes = {cut.data, cut.value_data, cut.scales, cut.value_scales, cut.pooled};
    const std::array<const ConversationBuffer*,5> src = {&image.k, &image.v, &image.k_scale, &image.v_scale, &image.pooled};
    const auto dst = pools(st);
    for (size_t i = 0; i < src.size(); ++i)
        if (!src[i]->visit(0, sizes[i], [&](const uint8_t* p, size_t n, size_t at) {
                return transfer(static_cast<uint8_t*>(dst[i]) + at, p, n, error, static_cast<admission_copy::Part>(admission_copy::key + i), admission_copy::restore);
            })) return false;
    return restore_residency(st, g, upto, error);
}

bool conversation_kv_restore_prefix(const ConversationKv& image, const QsaState& st, const ModelGeometry& g,
                                    int64_t captured, int64_t upto, bool index, std::string& error) {
    return conversation_kv_restore(image, st, g, upto, index, error, captured);
}

bool conversation_kv_part_sizes(const QsaState& st, const ModelGeometry& g, int64_t upto, bool index,
                                std::array<uint64_t, 5>& sizes, std::string& error) {
    Layout l{};
    if (!layout(st, g, upto, index, l, error)) return false;
    sizes = {l.data, l.value_data, l.scales, l.value_scales, l.pooled};
    return true;
}

bool conversation_kv_source(SessionKvSource& out, const QsaState& st, const ModelGeometry& g,
                            int64_t upto, bool index, std::string& error) {
    Layout l{};
    if (!layout(st, g, upto, index, l, error) || !valid(st, l, upto, error)) return false;
    SessionKvSource s;
    s.format = l.format; s.cells = l.cells; s.heads = g.n_head_kv; s.head_dim = g.head_dim;
    s.page_size = l.page_size; s.pooled_rows = l.pooled_rows; s.idx_dim = g.idx_key_dim;
    s.sizes = {l.data, l.value_data, l.scales, l.value_scales, l.pooled};
    const auto src = pools(st);
    const auto sizes = s.sizes;
    auto why = std::make_shared<std::string>();
    s.read = [src, sizes, why](size_t part, size_t offset, void* dst, size_t n) {
        if (part >= 5 || offset > sizes[part] || n > sizes[part] - offset) { *why = "K/V read out of range"; return false; }
        return transfer(dst, src[part] ? static_cast<const uint8_t*>(src[part]) + offset : nullptr, n, *why, static_cast<admission_copy::Part>(admission_copy::key + part), admission_copy::capture);
    };
    s.error = [why] { return *why; };
    out = std::move(s);
    return true;
}

bool conversation_kv_verify(const ConversationKv& image, const QsaState& st, const ModelGeometry& g,
                            int64_t upto, bool index, ConversationSnapshotExpectation expectation,
                            uint64_t& fingerprint, std::string& error, int64_t captured_tokens) {
    if (captured_tokens < 0) captured_tokens = upto;
    if (upto < 0 || upto > captured_tokens ||
        !conversation_kv_validate(image, st, g, captured_tokens, index, expectation, error)) return false;
    Layout cut{};
    if (!layout(st, g, upto, index, cut, error)) return false;
    const std::array<size_t, 5> sizes = {cut.data, cut.value_data, cut.scales, cut.value_scales, cut.pooled};
    std::array<uint8_t, 65536> buffer;
    uint64_t hash = 1469598103934665603ull;
    auto compare = [&](const void* device, const uint8_t* expected, size_t bytes, bool include_hash) {
        for (size_t offset = 0; offset < bytes;) {
            const size_t n = std::min(buffer.size(), bytes - offset);
            if (!transfer(buffer.data(), static_cast<const uint8_t*>(device) + offset, n, error, admission_copy::other, admission_copy::capture)) return false;
            if (std::memcmp(buffer.data(), expected + offset, n) != 0) {
                error = "conversation snapshot: restored K/V bytes differ";
                return false;
            }
            if (include_hash) for (size_t i = 0; i < n; ++i) { hash ^= buffer[i]; hash *= 1099511628211ull; }
            offset += n;
        }
        return true;
    };
    const std::array<const ConversationBuffer*, 5> saved = {&image.k, &image.v, &image.k_scale, &image.v_scale, &image.pooled};
    const auto authoritative = pools(st);
    for (size_t i = 0; i < saved.size(); ++i)
        if (!saved[i]->visit(0, sizes[i], [&](const uint8_t* p, size_t n, size_t at) {
                return compare(static_cast<const uint8_t*>(authoritative[i]) + at, p, n, true);
            })) return false;
    if (st.kv_mode == 2 && cut.cells > 0) {
        const auto resident = pools(st, true);
        const int64_t end = cut.cells / image.page_size;
        const int64_t begin = std::max<int64_t>(0, end - st.n_slots);
        for (size_t i = 0; i < 4; ++i) {
            if (saved[i]->empty()) continue;
            const size_t page_bytes = sizes[i] / size_t(end);
            for (int64_t page = begin; page < end; ++page)
                if (!saved[i]->visit(size_t(page) * page_bytes, page_bytes,
                        [&](const uint8_t* p, size_t n, size_t at) {
                            const size_t offset = size_t(page % st.n_slots) * page_bytes + at - size_t(page) * page_bytes;
                            return compare(static_cast<const uint8_t*>(resident[i]) + offset, p, n, false);
                        })) return false;
        }
    }
    fingerprint = hash;
    return true;
}

// Ordinary/private entry points, also implemented by the other backends. Shared
// native mounts use the explicit-expectation overloads above, including verify.
bool conversation_kv_validate(const ConversationKv& image, const QsaState& st, const ModelGeometry& g,
                              int64_t upto, bool index, std::string& error) {
    return conversation_kv_validate(image, st, g, upto, index, ConversationSnapshotExpectation::ordinary, error);
}
bool conversation_kv_restore(const ConversationKv& image, const QsaState& st, const ModelGeometry& g,
                             int64_t upto, bool index, std::string& error, int64_t captured_tokens) {
    return conversation_kv_restore(image, st, g, upto, index, ConversationSnapshotExpectation::ordinary,
                                   error, captured_tokens);
}
bool conversation_kv_verify(const ConversationKv& image, const QsaState& st, const ModelGeometry& g,
                            int64_t upto, bool index, uint64_t& fingerprint, std::string& error,
                            int64_t captured_tokens) {
    return conversation_kv_verify(image, st, g, upto, index, ConversationSnapshotExpectation::ordinary,
                                  fingerprint, error, captured_tokens);
}

} // namespace strata::core
