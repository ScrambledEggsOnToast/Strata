#include "strata/core/dflash2_bridge.hpp"
#include "strata/core/conversation_snapshot.hpp"
#include "conversation_checked.hpp"
#include "strata/plan/resident.hpp"
#include "conversation_copy.hpp"
#include "strata/core/admission_copy_observer.hpp"

#include <algorithm>
#include <array>
#include <limits>

namespace strata::core {

// Exported: the RAM cache and the persistent conversation store both bucket on exactly this key, so a caller that has
// to build a store selector before it owns an image (a prepared offer, a demand lookup) uses the same 18 fields the
// captured images carry.  A mismatch is an incompatible runtime geometry, never a conversion.
std::array<int64_t, 18> conversation_geometry_key(const ModelGeometry& g) {
    return {g.n_embd, g.n_layers, g.qsa_interval, g.ssm_state_size, g.ssm_k_heads,
            g.ssm_v_heads, g.ssm_d_conv, g.ssm_conv_channels, g.ssm_value_dim,
            g.n_head, g.n_head_kv, g.head_dim, g.idx_q_heads, g.idx_key_dim,
            g.hc, g.hc_lr, g.n_expert, g.n_ff};
}

namespace {
using conversation_detail::add;
using conversation_detail::product;

std::array<int64_t, 18> geometry_key(const ModelGeometry& g) {
    return conversation_geometry_key(g);
}

bool fail(std::string& error, const char* message) {
    error = std::string("conversation snapshot: ") + message;
    return false;
}

bool sync(std::string& error) {
    const auto status = cudaDeviceSynchronize();
    if (status == cudaSuccess) return true;
    error = std::string("conversation snapshot synchronize: ") + cudaGetErrorString(status);
    return false;
}

bool copy(void* dst, const void* src, size_t bytes, std::string& error, admission_copy::Part part,
          admission_copy::Direction direction, bool queued = false) {
    if (!bytes) return true;
    if (!dst || !src) return fail(error, "missing running-state buffer");
    // Synchronous admission receipts must never label an enqueued transfer as completed.
    if (queued) {
        const auto status = conversation_detail::copy_nonblocking(dst, src, bytes, true);
        if (status == cudaSuccess) return true;
        error = std::string("conversation snapshot running-state copy: ") + cudaGetErrorString(status);
        return false;
    }
    admission_copy::Transfer observed(part, direction, bytes);
    const auto status = conversation_detail::copy_nonblocking(dst, src, bytes);
    observed.finish(status == cudaSuccess);
    if (status == cudaSuccess) return true;
    error = std::string("conversation snapshot running-state copy: ") + cudaGetErrorString(status);
    return false;
}

// The session's carve (#216): `qsa_states` keeps global ordinals and holds the owned ones
// [qsa_ord0, qsa_ord0 + qsa_alloc); `gdn_state` holds `gdn_alloc` rows. A whole-model session owns every layer.
size_t owned_qsa(const SessionState& ss) { return (size_t) std::max<int64_t>(ss.qsa_alloc, 0); }
const QsaState& owned(const SessionState& ss, size_t j) { return ss.qsa_states[(size_t) ss.qsa_ord0 + j]; }

bool image_keys(const std::vector<ConversationImageKey>& images, size_t tokens) {
    int64_t previous = -1;
    for (const auto& image : images) {
        if (image.start <= previous || image.start < 0 || (uint64_t) image.start >= tokens) return false;
        previous = image.start;
    }
    return true;
}

bool checkpoint_targets(const SessionState& ss, const ModelGeometry& g, size_t tokens,
                        ConversationStateSizes& z, std::string& error) {
    if (!conversation_session_sizes(g, ss, z, error)) return false;
    if (ss.max_cells < 0 || tokens > (uint64_t) ss.max_cells || (z.gdn && !ss.gdn_state) ||
        (owned_qsa(ss) && !ss.qsa_states)) return fail(error, "invalid session running-state targets");
    for (size_t j = 0; j < owned_qsa(ss); ++j) {
        const auto& st = owned(ss, j);
        const auto block = strata::kernels::qsa_real_shapes().idx_block;
        size_t pooled_bytes = 0;
        if (!st.idx_tail || !st.idx_dead || !st.idx_block_pos || !st.idx_pooled ||
            st.max_cells < 0 || tokens > (uint64_t) st.max_cells ||
            (tokens && tokens / (uint64_t) block >= (uint64_t) std::max<int64_t>(0, st.idx_pooled_rows)) ||
            !product(pooled_bytes, {tokens / (uint64_t) block + 1, (uint64_t) g.idx_key_dim, sizeof(float)}))
            return fail(error, "invalid indexer running-state target");
    }
    return true;
}

bool view_validate(const ConversationView& view, const SessionState& ss,
                   const ModelGeometry& g, std::string& error) {
    ConversationStateSizes z;
    if (!checkpoint_targets(ss, g, view.ids.size(), z, error)) return false;
    if (view.ids.empty() || !image_keys(view.images, view.ids.size()) ||
        std::any_of(view.ids.begin(), view.ids.end(), [](int32_t id) { return id < 0; }))
        return fail(error, "invalid live token/image metadata");
    for (const auto& c : view.checkpoints) {
        if (!c.stage_parts.empty() || c.ids.empty() || c.ids.size() > view.ids.size() ||
            !std::equal(c.ids.begin(), c.ids.end(), view.ids.begin()))
            return fail(error, "checkpoint is not a live token prefix");
        if (!conversation_checkpoint_validate(c, ss, g, error)) return false;
        size_t image = 0;
        for (const auto& key : view.images) {
            if ((uint64_t) key.start >= c.ids.size()) break;
            if (image >= c.imgs.size() || !(c.imgs[image++] == key))
                return fail(error, "checkpoint image identity differs");
        }
        if (image != c.imgs.size()) return fail(error, "checkpoint image prefix differs");
    }
    return true;
}

bool metadata_bytes(const ConversationCheckpoint& c, size_t& total) {
    size_t ids = 0, images = 0;
    if (!product(ids, {c.ids.size(), sizeof(int32_t)}) ||
        !product(images, {c.imgs.size(), sizeof(ConversationImageKey)})) return false;
    for (size_t n : {ids, images, c.gdn.size(), c.ple.size(), c.tails.size(), c.dead.size(), c.block_pos.size()})
        if (!add(total, n)) return false;
    if (immutable_checkpoints_enabled())
        for (const auto* payload : {&c.gdn, &c.ple, &c.tails, &c.dead, &c.block_pos}) {
            const size_t extra = payload->sealed() ? payload->bytes() - payload->size() :
                                 payload->empty() ? 0 : CheckpointBytes::owner_bytes;
            if (!add(total, extra)) return false;
        }
    return true;
}
} // namespace

bool conversation_state_sizes(const ModelGeometry& g, ConversationStateSizes& z, std::string& error) {
    z = {};
    const auto key = geometry_key(g);
    for (size_t i = 0; i < key.size(); ++i)
        if (key[i] < 0 || (i != 1 && key[i] == 0)) return fail(error, "invalid model geometry");
    auto shape = strata::kernels::qsa_real_shapes();
    shape.idx_dim = g.idx_key_dim;
    strata::plan::CheckpointStateBytes bytes;
    if (!strata::plan::checkpoint_state_bytes((uint64_t) g.n_gdn_layers(),
        (uint64_t) g.ssm_state_size, (uint64_t) g.ssm_v_heads, (uint64_t) g.ssm_conv_channels,
        (uint64_t) g.ssm_d_conv, shape, bytes))
        return fail(error, "running-state byte count overflow");
    strata::plan::LogicalBytes count;
    const uint64_t tails = count.product({(uint64_t) g.n_qsa_layers(), bytes.tail});
    const uint64_t dead = count.product({(uint64_t) g.n_qsa_layers(), bytes.dead});
    const uint64_t positions = count.product({(uint64_t) g.n_qsa_layers(), bytes.block_pos});
    if (!count.valid || std::max({bytes.gdn, bytes.ple, bytes.tail, bytes.dead, bytes.block_pos,
                                 tails, dead, positions}) > SIZE_MAX)
        return fail(error, "indexer byte count overflow");
    z = {(size_t) bytes.gdn, (size_t) bytes.ple, (size_t) bytes.tail, (size_t) bytes.dead, (size_t) bytes.block_pos};
    return true;
}

bool conversation_session_sizes(const ModelGeometry& g, const SessionState& ss, ConversationStateSizes& z,
                                std::string& error) {
    if (!conversation_state_sizes(g, z, error)) return false;
    const int64_t n_gdn = g.n_gdn_layers(), n_qsa = g.n_qsa_layers();
    if (ss.gdn_alloc < 0 || ss.gdn_alloc > n_gdn || ss.qsa_ord0 < 0 || ss.qsa_alloc < 0 ||
        ss.qsa_ord0 > n_qsa || ss.qsa_alloc > n_qsa - ss.qsa_ord0)
        return fail(error, "invalid session layer carve");
    // z.gdn is n_gdn whole rows; the session holds gdn_alloc of them (all of them for the default full range)
    z.gdn = n_gdn > 0 ? z.gdn / (size_t) n_gdn * (size_t) ss.gdn_alloc : 0;
    return true;
}

bool conversation_checkpoint_validate(const ConversationCheckpoint& c, const SessionState& ss,
                                      const ModelGeometry& g, std::string& error) {
    ConversationStateSizes z;
    if (!checkpoint_targets(ss, g, c.ids.size(), z, error)) return false;
    const size_t layers = owned_qsa(ss);
    if (c.gdn.size() != z.gdn || c.ple.size() != (ss.ple_hist ? z.ple : 0) ||
        c.tails.size() != layers * z.tail || c.dead.size() != layers * z.dead ||
        c.block_pos.size() != layers * z.block_pos || !image_keys(c.imgs, c.ids.size()))
        return fail(error, "invalid checkpoint running-state payload");
    return true;
}

#if defined(STRATA_GPU_METAL)
}  // namespace strata::core
namespace strata::metal_engine {
// the running state lives in the Metal decode window (metal/src/engine/window.cpp), not in the session's buffers
bool window_checkpoint_save(strata::core::ConversationCheckpoint& c, std::string& error);
bool window_checkpoint_restore(const strata::core::ConversationCheckpoint& c, std::string& error);
}  // namespace strata::metal_engine
namespace strata::core {
#endif

bool conversation_checkpoint_save(ConversationCheckpoint& c, const SessionState& ss,
                                  const ModelGeometry& g, std::string& error) {
#if defined(STRATA_GPU_METAL)
    (void) ss; (void) g;
    return strata::metal_engine::window_checkpoint_save(c, error);
#endif
    ConversationStateSizes z;
    if (!checkpoint_targets(ss, g, c.ids.size(), z, error)) return false;
    const size_t layers = owned_qsa(ss);
    c.gdn.resize(z.gdn); c.ple.resize(ss.ple_hist ? z.ple : 0);
    c.tails.resize(layers * z.tail); c.dead.resize(layers * z.dead); c.block_pos.resize(layers * z.block_pos);
    if (!copy(c.gdn.write().data(), ss.gdn_state, c.gdn.size(), error, admission_copy::gdn, admission_copy::capture) ||
        !copy(c.ple.write().data(), ss.ple_hist, c.ple.size(), error, admission_copy::ple, admission_copy::capture)) return false;
    for (size_t j = 0; j < layers; ++j) {
        const auto& st = owned(ss, j);
        if (!copy(c.tails.write().data() + j * z.tail, st.idx_tail, z.tail, error, admission_copy::index_tail, admission_copy::capture) ||
            !copy(c.dead.write().data() + j * z.dead, st.idx_dead, z.dead, error, admission_copy::index_dead, admission_copy::capture) ||
            !copy(c.block_pos.write().data() + j * z.block_pos, st.idx_block_pos, z.block_pos, error, admission_copy::index_position, admission_copy::capture)) return false;
    }
    if (immutable_checkpoints_enabled()) c.seal();
    return true;
}

bool conversation_checkpoint_restore(const ConversationCheckpoint& c, SessionState& ss,
                                     const ModelGeometry& g, std::string& error) {
#if defined(STRATA_GPU_METAL)
    (void) g;
    if (!strata::metal_engine::window_checkpoint_restore(c, error)) return false;
    ss.ple_prev[0] = c.ids.size() >= 2 ? c.ids[c.ids.size() - 2] : -1;
    ss.ple_prev[1] = c.ids.size() >= 1 ? c.ids[c.ids.size() - 1] : -1;
    return true;
#endif
    if (!conversation_checkpoint_validate(c, ss, g, error)) return false;
    ConversationStateSizes z;
    if (!conversation_session_sizes(g, ss, z, error)) return false;
    if (!sync(error)) return false;   // the copies below no longer ride the legacy stream's implicit ordering
    if (!copy(ss.gdn_state, c.gdn.data(), c.gdn.size(), error, admission_copy::gdn, admission_copy::restore) ||
        !copy(ss.ple_hist, c.ple.data(), c.ple.size(), error, admission_copy::ple, admission_copy::restore)) return false;
    for (size_t j = 0; j < owned_qsa(ss); ++j) {
        const auto& st = owned(ss, j);
        if (!copy(st.idx_tail, c.tails.data() + j * z.tail, z.tail, error, admission_copy::index_tail, admission_copy::restore) ||
            !copy(st.idx_dead, c.dead.data() + j * z.dead, z.dead, error, admission_copy::index_dead, admission_copy::restore) ||
            !copy(st.idx_block_pos, c.block_pos.data() + j * z.block_pos, z.block_pos, error, admission_copy::index_position, admission_copy::restore)) return false;
        if (!c.ids.empty()) {
            const size_t row = c.ids.size() / strata::kernels::qsa_real_shapes().idx_block;
            if (!copy(st.idx_pooled + row * g.idx_key_dim, c.dead.data() + j * z.dead, z.dead, error, admission_copy::index_repair, admission_copy::restore)) return false;
        }
    }
    const size_t tokens = c.ids.size();
    ss.ple_prev[0] = tokens >= 2 ? c.ids[tokens - 2] : -1;
    ss.ple_prev[1] = tokens >= 1 ? c.ids[tokens - 1] : -1;
    return sync(error);
}

bool conversation_session_copy(const SessionState& from, SessionState& to, const ModelGeometry& g,
                               const std::vector<int32_t>& ids, const ConversationCheckpoint* at,
                               std::string& error) {
    ConversationStateSizes source, target;
    if (!checkpoint_targets(from, g, ids.size(), source, error) ||
        !checkpoint_targets(to, g, ids.size(), target, error)) return false;
    if (from.layer_lo != to.layer_lo || from.layer_hi != to.layer_hi ||
        from.gdn_ord0 != to.gdn_ord0 || from.gdn_alloc != to.gdn_alloc ||
        from.qsa_ord0 != to.qsa_ord0 || from.qsa_alloc != to.qsa_alloc ||
        bool(from.ple_hist) != bool(to.ple_hist) ||
        from.allocation_device != to.allocation_device)
        return fail(error, "incompatible direct running-state owners");
    if (at && (at->ids != ids || !conversation_checkpoint_validate(*at, to, g, error)))
        return fail(error, "invalid direct-copy checkpoint");
    // Reject a late-layer mismatch before the first write. No layout conversion
    // or host-image fallback is allowed in this path.
    for (size_t j = 0; j < owned_qsa(from); ++j)
        if (!conversation_kv_copy_validate(owned(from, j), owned(to, j), g, (int64_t) ids.size(), true, error))
            return false;
    if (!sync(error)) return false;
    for (size_t j = 0; j < owned_qsa(from); ++j)
        if (!conversation_detail::copy_prevalidated_kv(owned(from, j), owned(to, j), g,
                                                       (int64_t) ids.size(), true, error)) return false;
    if (at) return conversation_checkpoint_restore(*at, to, g, error);
    if (!copy(to.gdn_state, from.gdn_state, source.gdn, error, admission_copy::gdn, admission_copy::direct) ||
        !copy(to.ple_hist, from.ple_hist, from.ple_hist ? source.ple : 0, error,
              admission_copy::ple, admission_copy::direct)) return false;
    for (size_t j = 0; j < owned_qsa(from); ++j) {
        const auto& src = owned(from, j);
        const auto& dst = owned(to, j);
        if (!copy(dst.idx_tail, src.idx_tail, source.tail, error, admission_copy::index_tail, admission_copy::direct) ||
            !copy(dst.idx_dead, src.idx_dead, source.dead, error, admission_copy::index_dead, admission_copy::direct) ||
            !copy(dst.idx_block_pos, src.idx_block_pos, source.block_pos, error,
                  admission_copy::index_position, admission_copy::direct)) return false;
        if (!ids.empty()) {
            const size_t row = ids.size() / strata::kernels::qsa_real_shapes().idx_block;
            if (!copy(dst.idx_pooled + row * g.idx_key_dim, src.idx_dead, source.dead, error,
                      admission_copy::index_repair, admission_copy::direct)) return false;
        }
    }
    to.ple_prev[0] = ids.size() >= 2 ? ids[ids.size() - 2] : -1;
    to.ple_prev[1] = ids.size() >= 1 ? ids.back() : -1;
    return sync(error);
}

namespace {
bool draft_view_validate(const ConversationView& view, const ModelGeometry& g, bool owns,
                         size_t& boundary_bytes, std::string& error) {
    boundary_bytes = 0;
    const int64_t live = int64_t(view.ids.size());
    const bool native = view.draft_identity.kind == DraftKind::dflash2;
    owns = owns || native;
    const bool unknown = view.draft_begin == -1 && view.draft_tokens == -1 && view.draft_next_token == -1;
    if (!unknown && (!owns || view.draft_begin < 0 || view.draft_tokens < view.draft_begin ||
        view.draft_tokens > live || (native ? view.draft_tokens != live || view.draft_next_token != -1 :
            (view.draft_tokens == live ? view.draft_next_token < 0 : view.draft_next_token != -1))))
        return fail(error, "incoherent completed draft interval");
    if (native && (!view.draft_identity.bound() || !view.draft_owner ||
        view.draft_origin < 0 || view.draft_origin > view.draft_begin ||
        view.draft_tokens - view.draft_begin > 2047 || view.draft_boundary_residual))
        return fail(error, "native draft lacks drained committed context certification");
    if (view.draft_boundary_residual) {
        if (!owns || unknown || view.draft_tokens != live - 1 || !view.draft_owner ||
            view.draft_boundary_owner != view.draft_owner || view.draft_boundary_position != live - 1 ||
            !product(boundary_bytes, {uint64_t(g.n_embd), uint64_t(g.hc), sizeof(float)}) ||
            !boundary_bytes || view.draft_boundary_residual->size() != boundary_bytes ||
            !view.draft_boundary_residual->frozen())
            return fail(error, "unowned or wrong-position boundary residual");
    } else if (view.draft_boundary_position != -1 || view.draft_boundary_owner)
        return fail(error, "boundary ownership without a residual");
    if (view.exact_cut && (!view.images.empty() || !view.checkpoints.empty() ||
        (owns && (unknown || view.draft_tokens != live - (native ? 0 : 1) ||
                  (!native && view.draft_owner && !view.draft_boundary_residual)))))
        return fail(error, "exact cut lacks its kind-specific producer certificate");
    return true;
}
}
bool conversation_checkpoint_copy(const SessionState& from, SessionState& to, const std::vector<int32_t>& ids,
                                  const ModelGeometry& g, std::string& error, bool queued) {
    // conversation_checkpoint_save(from) + conversation_checkpoint_restore(to) without the host image, for two
    // sessions carved alike on the same device (a stage's own session and its batch slots): the same bytes into
    // the same places, through copy() like the restore (the private non-blocking stream, complete on return unless
    // `queued`: then the caller synchronizes the device).
    ConversationStateSizes zf, zt;
    if (!checkpoint_targets(from, g, ids.size(), zf, error) || !checkpoint_targets(to, g, ids.size(), zt, error))
        return false;
    if (from.layer_lo != to.layer_lo || from.layer_hi != to.layer_hi ||
        from.gdn_ord0 != to.gdn_ord0 || from.qsa_ord0 != to.qsa_ord0 ||
        from.allocation_device != to.allocation_device ||
        zf.gdn != zt.gdn || zf.tail != zt.tail || zf.dead != zt.dead || zf.block_pos != zt.block_pos ||
        owned_qsa(from) != owned_qsa(to) || (from.ple_hist != nullptr) != (to.ple_hist != nullptr))
        return fail(error, "direct copy between differently carved sessions");
    auto acopy = [&](void* dst, const void* src, size_t bytes, admission_copy::Part part) {
        return copy(dst, src, bytes, error, part, admission_copy::direct, queued);
    };
    if (!acopy(to.gdn_state, from.gdn_state, zf.gdn, admission_copy::gdn) ||
        !acopy(to.ple_hist, from.ple_hist, from.ple_hist ? zf.ple : 0, admission_copy::ple)) return false;
    for (size_t j = 0; j < owned_qsa(from); ++j) {
        const auto& a = owned(from, j);
        const auto& b = owned(to, j);
        if (!acopy(b.idx_tail, a.idx_tail, zf.tail, admission_copy::index_tail) ||
            !acopy(b.idx_dead, a.idx_dead, zf.dead, admission_copy::index_dead) ||
            !acopy(b.idx_block_pos, a.idx_block_pos, zf.block_pos, admission_copy::index_position)) return false;
        if (!ids.empty()) {
            const size_t row = ids.size() / strata::kernels::qsa_real_shapes().idx_block;
            if (!acopy(b.idx_pooled + row * g.idx_key_dim, a.idx_dead, zf.dead, admission_copy::index_repair)) return false;
        }
    }
    const size_t tokens = ids.size();
    to.ple_prev[0] = tokens >= 2 ? ids[tokens - 2] : -1;
    to.ple_prev[1] = tokens >= 1 ? ids[tokens - 1] : -1;
    return true;
}

bool conversation_snapshot_bytes(const ConversationView& view, const SessionState& ss,
                                 const ModelGeometry& g, const QsaState* draft, size_t& bytes, std::string& error) {
    bytes = sizeof(ConversationScope) + SavedConversation::draft_metadata_bytes;
    if (!view_validate(view, ss, g, error)) return false;
    size_t boundary_bytes = 0;
    if (!draft_view_validate(view, g, draft != nullptr, boundary_bytes, error)) return false;
    if (view.draft_boundary_residual && !add(bytes, view.draft_boundary_residual->bytes()))
        return fail(error, "boundary residual byte count overflow");
    ConversationStateSizes z;
    if (!conversation_session_sizes(g, ss, z, error)) return false;
    size_t ids = 0, images = 0, checkpoints = 0, layers = 0, tails = 0, dead = 0, positions = 0;
    const auto qsa = (uint64_t) owned_qsa(ss);
    if (!product(ids, {view.ids.size(), sizeof(int32_t)}) ||
        !product(images, {view.images.size(), sizeof(ConversationImageKey)}) ||
        !product(checkpoints, {view.checkpoints.size(), sizeof(ConversationCheckpoint)}) ||
        !product(layers, {qsa + (draft ? 1 : 0), sizeof(ConversationKv)}) ||
        !product(tails, {qsa, z.tail}) || !product(dead, {qsa, z.dead}) ||
        !product(positions, {qsa, z.block_pos})) return fail(error, "snapshot metadata byte count overflow");
    if (view.draft_identity.kind == DraftKind::dflash2) {
        const uint64_t rows = uint64_t(view.draft_tokens - view.draft_begin);
        ConversationBuffer buffer;
        const size_t part = buffer.allocation_peak(size_t(rows) * 8 * 128 * sizeof(uint16_t));
        size_t native = 0;
        if (part == SIZE_MAX || !product(native, {10, part}) ||
            !add(bytes, native) || !add(bytes, 5 * sizeof(ConversationKv)))
            return fail(error, "native draft snapshot byte count overflow");
    }
    for (size_t n : {ids, images, checkpoints, layers, tails, dead, positions, z.gdn, ss.ple_hist ? z.ple : 0})
        if (!add(bytes, n)) return fail(error, "snapshot byte count overflow");
    if (immutable_checkpoints_enabled()) {
        // The live checkpoint is sealed after capture, before publication.
        const size_t payloads = (z.gdn != 0) + (ss.ple_hist && z.ple != 0) +
                                (tails != 0) + (dead != 0) + (positions != 0);
        if (!add(bytes, payloads * CheckpointBytes::owner_bytes)) return fail(error, "checkpoint owner overflow");
    }
    for (const auto& c : view.checkpoints)
        if (!metadata_bytes(c, bytes)) return fail(error, "checkpoint byte count overflow");
    const int64_t upto = (int64_t) view.ids.size(); // view_validate bounds this by signed max_cells
    for (uint64_t i = 0; i < qsa + (draft ? 1 : 0); ++i) {
        const auto& st = i == qsa ? *draft : owned(ss, (size_t) i);
        const size_t n = conversation_kv_bytes(st, g, upto, i != qsa);
        if (!n || !add(bytes, n)) return fail(error, "invalid or overflowing K/V byte estimate");
    }
    return true;
}

bool conversation_snapshot_capture_bytes(const ConversationKvReuse& reuse, const ConversationView& view,
                                         const SessionState& ss, const ModelGeometry& g,
                                         const QsaState* draft, size_t& bytes, std::string& error) {
    if (!conversation_snapshot_bytes(view, ss, g, draft, bytes, error)) return false;
    if (view.exact_cut && !reuse.kv.empty())
        return fail(error, "an exact shared cut cannot retain bytes from another captured image");
    if (reuse.kv.empty()) return true;
    const size_t layers = owned_qsa(ss) + (draft ? 1 : 0);
    const bool native = view.draft_identity.kind == DraftKind::dflash2;
    const size_t directory_layers = layers + (native ? 5 : 0);
    if (reuse.kv.size() != directory_layers || reuse.unchanged_tokens < 0 ||
        reuse.unchanged_tokens > reuse.captured_tokens || reuse.unchanged_tokens > int64_t(view.ids.size()))
        return fail(error, "invalid retained K/V prefix");
    for (size_t i = 0; i < layers; ++i) {
        const bool index = i < owned_qsa(ss);
        const auto& st = index ? owned(ss, i) : *draft;
        if (!conversation_kv_validate(reuse.kv[i], st, g, reuse.captured_tokens, index, error)) return false;
        const size_t fresh = conversation_kv_bytes(st, g, int64_t(view.ids.size()), index);
        size_t retained = 0;
        if (!conversation_kv_capture_bytes(reuse.kv[i], st, g, int64_t(view.ids.size()), index, retained, error)) return false;
        bytes -= fresh;
        if (!add(bytes, retained)) return fail(error, "retained K/V allocation overflow");
    }
    if (native) {
        const size_t rows = size_t(view.draft_tokens - view.draft_begin);
        const size_t part = rows * 8 * 128 * sizeof(uint16_t);
        ConversationBuffer empty;
        const size_t fresh = empty.allocation_peak(part);
        for (size_t layer = layers; layer < directory_layers; ++layer)
            for (const auto* buffer : {&reuse.kv[layer].k, &reuse.kv[layer].v}) {
                bytes -= fresh;
                const size_t peak = buffer->allocation_peak(part);
                if (peak == SIZE_MAX || !add(bytes, peak))
                    return fail(error, "native draft retained buffer allocation overflow");
            }
    }
    size_t directory = 0;
    if (!product(directory, {reuse.kv.capacity() - directory_layers, sizeof(ConversationKv)}) ||
        !add(bytes, directory))
        return fail(error, "retained K/V directory overflow");
    return true;
}

bool conversation_snapshot_save(SavedConversation& image, const ConversationView& view,
                                const SessionState& ss, const ModelGeometry& g,
                                const QsaState* draft, std::string& error,
                                ConversationKvReuse reuse, size_t* reused_bytes,
                                std::vector<ConversationRetainedPrefix>* retained) {
    size_t estimate = 0;
    if (!conversation_snapshot_capture_bytes(reuse, view, ss, g, draft, estimate, error) || !sync(error)) return false;
    if (retained) retained->clear();
    // Build into a new object so a failure cannot publish a partial snapshot.
    SavedConversation captured;
    captured.geometry = geometry_key(g);
    captured.layer_lo = ss.layer_lo; captured.layer_hi = ss.layer_hi;
    captured.live.ids = view.ids; captured.live.imgs = view.images;
    captured.cvec = view.cvec; captured.checkpoints = conversation_detail::copy_checkpoints_beating(view.checkpoints);
    captured.draft_identity = view.draft_identity;
    captured.draft_origin = view.draft_origin;
    if ((draft || view.draft_identity.kind == DraftKind::dflash2) && view.draft_owner) {
        captured.draft_begin = view.draft_begin; captured.draft_cells = view.draft_tokens;
        captured.draft_next_token = view.draft_next_token;
        if (view.draft_boundary_residual)
            captured.draft_boundary_residual = view.draft_boundary_residual->frozen_alias();
    }
    const int64_t unchanged = reuse.kv.empty() ? 0 : reuse.unchanged_tokens;
    const size_t layers = owned_qsa(ss);
    // The donor's own part lengths, read before the capture resizes those buffers in place: an offered reuse interval
    // is bounded by what the donor actually holds, never by what this image happens to need.
    std::vector<std::array<uint64_t, 5>> donor_bytes;
    if (retained) {
        donor_bytes.resize(layers + (draft ? 1 : 0));
        for (size_t j = 0; j < donor_bytes.size() && j < reuse.kv.size(); ++j) {
            const ConversationKv& source = reuse.kv[j];
            const std::array<const ConversationBuffer*, 5> parts = {&source.k, &source.v, &source.k_scale,
                                                                    &source.v_scale, &source.pooled};
            for (size_t part = 0; part < parts.size(); ++part) donor_bytes[j][part] = uint64_t(parts[part]->size());
        }
    }
    auto report = [retained, &donor_bytes](uint32_t layer, const std::array<uint64_t, 5>& kept) {
        if (!retained) return;
        for (uint32_t part = 0; part < kept.size(); ++part) {
            if (kept[part] == 0) continue;
            ConversationRetainedPrefix entry;
            entry.layer = layer;
            entry.part = part;
            entry.bytes = kept[part];
            entry.donor_bytes = layer < donor_bytes.size() ? donor_bytes[layer][part] : 0;
            retained->push_back(entry);
        }
    };
    captured.kv = std::move(reuse.kv);
    captured.kv.resize(layers + (draft ? 1 : 0) +
                       (view.draft_identity.kind == DraftKind::dflash2 ? 5 : 0));
    if (!conversation_checkpoint_save(captured.live, ss, g, error)) return false;
    const int64_t upto = (int64_t) view.ids.size();
    for (size_t j = 0; j < layers; ++j) {
        std::array<uint64_t, 5> kept{};
        if (!conversation_kv_save(captured.kv[j], owned(ss, j), g, upto, true, error,
                                  unchanged, reused_bytes, retained ? &kept : nullptr)) return false;
        report((uint32_t) j, kept);
    }
    // The draft's final cell may not have been computed when the output cap was
    // reached. Refresh that page even when the main prefix continued unchanged.
    if (draft) {
        std::array<uint64_t, 5> kept{};
        if (!conversation_kv_save(captured.kv.back(), *draft, g, upto, false, error,
                                  std::max<int64_t>(0, unchanged - 1), reused_bytes,
                                  retained ? &kept : nullptr)) return false;
        report((uint32_t) layers, kept);
    }
    if (view.exact_cut) {
        // The active recurrence and PLE were captured exactly at this seam. Raw
        // index tails past the current partial block and block-position scratch
        // are not state a continuation reads; they may still name an old owner.
        ConversationStateSizes z;
        if (!conversation_session_sizes(g, ss, z, error)) return false;
        // A carved stage can allocate a primary QSA only to lend its RoPE table,
        // and every stage allocates PLE history although only layer zero uses it.
        // Neither unused owner's bytes belong to the shared prefix.
        if (ss.layer_lo > 0) {
            auto& history = captured.live.ple.write();
            std::fill(history.begin(), history.end(), 0);
        }
        const size_t active = size_t(upto % strata::kernels::qsa_real_shapes().idx_block) *
                              size_t(g.idx_key_dim) * sizeof(float);
        auto& tails = captured.live.tails.write();
        for (size_t j = 0; j < layers; ++j) {
            const int64_t layer = (ss.qsa_ord0 + int64_t(j) + 1) * g.qsa_interval - 1;
            const bool active_layer = layer >= ss.layer_lo && layer < ss.layer_hi;
            if (!active_layer) {
                std::fill(tails.begin() + j * z.tail, tails.begin() + (j + 1) * z.tail, 0);
                auto& dead = captured.live.dead.write();
                std::fill(dead.begin() + j * z.dead, dead.begin() + (j + 1) * z.dead, 0);
                if (!conversation_kv_clear_inactive(captured.kv[j], upto, 0, 0, error) ||
                    !captured.kv[j].pooled.visit(0, captured.kv[j].pooled.size(),
                        [](uint8_t* p, size_t n, size_t) { std::memset(p, 0, n); return true; }))
                    return fail(error, "invalid inactive stage K/V");
                continue;
            }
            std::fill(tails.begin() + j * z.tail + active, tails.begin() + (j + 1) * z.tail, 0);
            if (!conversation_kv_clear_inactive(captured.kv[j], upto, 0, upto, error)) return false;
            // The spare is the prefix's constant dead key, never a stale pooled row.
            if (!captured.kv[j].pooled.visit(captured.kv[j].pooled.size() - z.dead, z.dead,
                [&](uint8_t* p, size_t n, size_t at) {
                    std::memcpy(p, captured.live.dead.data() + j * z.dead +
                                   at - (captured.kv[j].pooled.size() - z.dead), n);
                    return true;
                })) return fail(error, "invalid exact indexer spare");
        }
        auto& positions = captured.live.block_pos.write();
        std::fill(positions.begin(), positions.end(), 0);
        if (draft) {
            if (!conversation_kv_clear_inactive(captured.kv.back(), upto, view.draft_begin,
                                                view.draft_tokens, error)) return false;
            captured.kv.back().cells = view.draft_tokens;
        }
    }
    image = std::move(captured);
    return true;
}

bool conversation_snapshot_sources(SavedConversation& meta, std::vector<SessionKvSource>& sources,
                                   const ConversationView& view, const SessionState& ss, const ModelGeometry& g,
                                   const QsaState* draft, std::string& error) {
    if (view.exact_cut) return fail(error, "an exact cut requires the sanitizing snapshot capture");
    if (view.draft_identity.kind == DraftKind::dflash2 ||
        (view.draft_identity.kind == DraftKind::mtp && !draft))
        return fail(error, "kind-tagged native capture requires its neural owner, not a QSA source");
    size_t boundary_bytes = 0;
    if (!draft_view_validate(view, g, draft != nullptr, boundary_bytes, error) ||
        !view_validate(view, ss, g, error) || !sync(error)) return false;
    SavedConversation captured;
    captured.geometry = geometry_key(g);
    captured.layer_lo = ss.layer_lo; captured.layer_hi = ss.layer_hi;
    captured.live.ids = view.ids; captured.live.imgs = view.images;
    captured.cvec = view.cvec; captured.checkpoints = conversation_detail::copy_checkpoints_beating(view.checkpoints);
    captured.draft_identity = view.draft_identity; captured.draft_origin = view.draft_origin;
    if (draft && view.draft_owner) {
        captured.draft_begin = view.draft_begin; captured.draft_cells = view.draft_tokens;
        captured.draft_next_token = view.draft_next_token;
        if (view.draft_boundary_residual)
            captured.draft_boundary_residual = view.draft_boundary_residual->frozen_alias();
    }
    if (!conversation_checkpoint_save(captured.live, ss, g, error)) return false;
    const int64_t upto = (int64_t) view.ids.size();
    const size_t layers = owned_qsa(ss);
    std::vector<SessionKvSource> out(layers + (draft ? 1 : 0));
    for (size_t j = 0; j < layers; ++j)
        if (!conversation_kv_source(out[j], owned(ss, j), g, upto, true, error)) return false;
    if (draft && !conversation_kv_source(out.back(), *draft, g, upto, false, error)) return false;
    meta = std::move(captured);
    sources = std::move(out);
    return true;
}

bool conversation_session_read_limits(SessionReadLimits& limits, const SessionState& ss, const ModelGeometry& g,
                                      const QsaState* draft, uint64_t max_tokens, uint64_t max_checkpoints,
                                      std::string& error) {
    ConversationStateSizes z;
    if (!conversation_session_sizes(g, ss, z, error)) return false;
    if (ss.max_cells < 0) return fail(error, "invalid session cells");
    const uint64_t tokens = std::min<uint64_t>(max_tokens, (uint64_t) ss.max_cells);
    const size_t layers = owned_qsa(ss);
    if ((owned_qsa(ss) && !ss.qsa_states)) return fail(error, "invalid session running-state targets");
    size_t tails = 0, dead = 0, block_pos = 0;
    if (!product(tails, {layers, z.tail}) || !product(dead, {layers, z.dead}) ||
        !product(block_pos, {layers, z.block_pos})) return fail(error, "indexer byte count overflow");
    SessionReadLimits l = limits;   // keeps the caller's admit / progress / max_file_bytes
    l.max_tokens = tokens;
    l.max_checkpoints = max_checkpoints;
    l.max_kv_layers = layers + (draft ? 1 : 0);
    l.geometry = geometry_key(g);
    l.layer_range = std::make_pair(ss.layer_lo, ss.layer_hi);
    l.max_state_bytes = {z.gdn, ss.ple_hist ? z.ple : 0, tails, dead, block_pos};
    size_t boundary_limit = 0;
    if (draft && !product(boundary_limit, {uint64_t(g.n_embd), uint64_t(g.hc), sizeof(float)}))
        return fail(error, "draft boundary read limit overflow");
    l.max_draft_boundary_bytes = boundary_limit;
    l.max_kv_bytes.assign(l.max_kv_layers, {});
    for (size_t j = 0; j < layers; ++j) {
        const auto& st = owned(ss, j);
        const int64_t upto = (int64_t) std::min<uint64_t>(tokens, (uint64_t) std::max<int64_t>(st.max_cells, 0));
        if (!conversation_kv_part_sizes(st, g, upto, true, l.max_kv_bytes[j], error)) return false;
    }
    if (draft) {
        const int64_t dupto = (int64_t) std::min<uint64_t>(tokens, (uint64_t) std::max<int64_t>(draft->max_cells, 0));
        if (!conversation_kv_part_sizes(*draft, g, dupto, false, l.max_kv_bytes.back(), error)) return false;
    }
    limits = std::move(l);
    return true;
}

bool conversation_snapshot_validate(const SavedConversation& image, const SessionState& ss,
                                    const ModelGeometry& g, const QsaState* draft,
                                    ConversationSnapshotExpectation expectation, std::string& error,
                                    const DraftIdentity* identity) {
    const bool native = image.draft_identity.kind == DraftKind::dflash2;
    if (!image.live.stage_parts.empty())
        return fail(error, "a running checkpoint's stage parts belong in the stage images, not in live.stage_parts");
    if ((image.draft_identity.kind != DraftKind::none &&
         (!identity || !identity->content_matches(image.draft_identity))) || (native && draft))
        return fail(error, "foreign draft kind/weights/borrowed target/geometry");
    if (image.geometry != geometry_key(g)) return fail(error, "incompatible runtime geometry");
    // an image holds exactly one carve's running state and K/V: same layer range or nothing
    if (image.layer_lo != ss.layer_lo || image.layer_hi != ss.layer_hi)
        return fail(error, "snapshot from another session layer range");
    const ConversationView view{image.live.ids, image.live.imgs, image.checkpoints, image.cvec};
    if (!view_validate(view, ss, g, error) || !conversation_checkpoint_validate(image.live, ss, g, error)) return false;
    const size_t layers = owned_qsa(ss);
    if (image.kv.size() != layers + (native ? 5 : draft ? 1 : 0))
        return fail(error, "invalid kind-specific K/V layer count");
    const int64_t upto = (int64_t) image.live.ids.size();
    size_t boundary_bytes = 0;
    const bool unknown = image.draft_begin == -1 && image.draft_cells == -1 && image.draft_next_token == -1;
    if (!unknown && ((!draft && !native) || image.draft_begin < 0 || image.draft_cells < image.draft_begin ||
        image.draft_cells > upto || image.draft_cells > image.kv.back().cells ||
        (native ? image.draft_cells != upto || image.draft_next_token != -1 :
         (image.draft_cells == upto ? image.draft_next_token < 0 : image.draft_next_token != -1))))
        return fail(error, "incoherent saved draft interval");
    if (native) {
        if (unknown || !image.draft_identity.bound() || image.draft_identity.format != 1 ||
            image.draft_identity.geometry != dflash2::native_geometry ||
            image.draft_origin < 0 || image.draft_origin > image.draft_begin ||
            image.draft_cells > 262144 || image.draft_cells - image.draft_begin > 2047 ||
            !image.draft_boundary_residual.empty())
            return fail(error, "invalid native context identity/origin/certification");
        const uint64_t bytes = uint64_t(image.draft_cells - image.draft_begin) * 8 * 128 * sizeof(uint16_t);
        for (size_t layer = layers; layer < image.kv.size(); ++layer) {
            const auto& kv = image.kv[layer];
            if (kv.format != 0 || kv.cells != image.draft_cells || kv.heads != 8 ||
                kv.head_dim != 128 || kv.page_size != 1 || kv.pooled_rows || kv.idx_dim ||
                kv.k.size() != bytes || kv.v.size() != bytes || !kv.k_scale.empty() ||
                !kv.v_scale.empty() || !kv.pooled.empty())
                return fail(error, "invalid five-layer 8x128 BF16 native K/V payload");
        }
    }
    if (!image.draft_boundary_residual.empty() && (!draft || unknown || image.draft_cells != upto - 1 ||
        !product(boundary_bytes, {uint64_t(g.n_embd), uint64_t(g.hc), sizeof(float)}) ||
        !boundary_bytes || image.draft_boundary_residual.size() != boundary_bytes))
        return fail(error, "invalid saved boundary residual");
    const bool exact = expectation == ConversationSnapshotExpectation::exact_cut;
    if (exact && (!image.live.imgs.empty() || !image.checkpoints.empty()))
        return fail(error, "shared exact cut contains private image/checkpoint metadata");
    if (exact && draft && !unknown && (image.draft_cells != upto - 1 ||
                          image.draft_next_token != -1 || image.draft_boundary_residual.empty()))
        return fail(error, "shared exact cut lacks certified draft coverage/boundary");
    for (size_t j = 0; j < layers; ++j)
        if (!conversation_kv_validate(image.kv[j], owned(ss, j), g, upto, true, expectation, error)) return false;
    if (draft && !conversation_kv_validate(image.kv.back(), *draft, g, upto, false, expectation, error)) return false;
    if (exact) {
        // Mirror the capture's excluded running-state bytes for every carve.
        // These checks never depend on a record-controlled K/V extent.
        const auto zeros = [](const CheckpointBytes& bytes, size_t begin, size_t end) {
            const auto& source = bytes.read();
            return begin <= end && end <= source.size() &&
                std::all_of(source.begin() + begin, source.begin() + end,
                            [](uint8_t byte) { return byte == 0; });
        };
        if ((ss.layer_lo > 0 && !zeros(image.live.ple, 0, image.live.ple.size())) ||
            !zeros(image.live.block_pos, 0, image.live.block_pos.size()))
            return fail(error, "shared exact cut contains excluded PLE/position bytes");
        ConversationStateSizes z;
        if (!conversation_session_sizes(g, ss, z, error)) return false;
        const size_t active = size_t(upto % strata::kernels::qsa_real_shapes().idx_block) *
                              size_t(g.idx_key_dim) * sizeof(float);
        for (size_t j = 0; j < layers; ++j) {
            const int64_t layer = (ss.qsa_ord0 + int64_t(j) + 1) * g.qsa_interval - 1;
            const bool active_layer = layer >= ss.layer_lo && layer < ss.layer_hi;
            if (!zeros(image.live.tails, j * z.tail + (active_layer ? active : 0), (j + 1) * z.tail))
                return fail(error, "shared exact cut contains excluded index-tail bytes");
            const auto& kv = image.kv[j];
            if (!active_layer) {
                if (!zeros(image.live.dead, j * z.dead, (j + 1) * z.dead))
                    return fail(error, "shared exact cut contains inactive indexer state");
                for (const auto* part : {&kv.k, &kv.v, &kv.k_scale, &kv.v_scale, &kv.pooled})
                    if (!part->visit(0, part->size(), [](const uint8_t* p, size_t n, size_t) {
                        return std::all_of(p, p + n, [](uint8_t byte) { return byte == 0; });
                    })) return fail(error, "shared exact cut contains inactive stage K/V");
            } else if (!kv.pooled.visit(kv.pooled.size() - z.dead, z.dead,
                [&](const uint8_t* p, size_t n, size_t at) {
                    return std::memcmp(p, image.live.dead.data() + j * z.dead +
                        at - (kv.pooled.size() - z.dead), n) == 0;
                })) return fail(error, "shared exact cut has a foreign indexer spare");
        }
    }
    return true;
}

ConversationRestore conversation_snapshot_restore(const SavedConversation& image, SessionState& ss,
                                                   const ModelGeometry& g, const QsaState* draft,
                                                   ConversationSnapshotExpectation expectation, std::string& error,
                                                   const ConversationCheckpoint* selected,
                                                   const DraftIdentity* identity) {
    if (!conversation_snapshot_validate(image, ss, g, draft, expectation, error, identity))
        return ConversationRestore::invalid;
    // A cut is an existing sealed checkpoint of this exact image, never an arbitrary token junction.
    if (selected && std::none_of(image.checkpoints.begin(), image.checkpoints.end(),
            [&](const auto& c) { return &c == selected; })) {
        fail(error, "selected checkpoint does not belong to image");
        return ConversationRestore::invalid;
    }
    if (!sync(error)) return ConversationRestore::transfer_failed;
    const auto& running = selected ? *selected : image.live;
    const int64_t upto = (int64_t) running.ids.size();
    const int64_t captured = (int64_t) image.live.ids.size();
    for (size_t j = 0; j < owned_qsa(ss); ++j)
        if (!conversation_kv_restore(image.kv[j], owned(ss, j), g, upto, true, expectation, error, captured))
            return ConversationRestore::transfer_failed;
    if ((draft && !identity && !conversation_kv_restore(image.kv.back(), *draft, g, upto, false,
                                                      expectation, error, captured)) ||
        !conversation_checkpoint_restore(running, ss, g, error)) return ConversationRestore::transfer_failed;
    return ConversationRestore::restored;
}

// the draft layer's K/V included (the stage that owns the draft head, or no layer split)
bool conversation_snapshot_bytes(const ConversationView& view, const SessionState& ss, const ModelGeometry& g,
                                 const QsaState& draft, size_t& bytes, std::string& error) {
    return conversation_snapshot_bytes(view, ss, g, &draft, bytes, error);
}
bool conversation_snapshot_capture_bytes(const ConversationKvReuse& reuse, const ConversationView& view,
                                         const SessionState& ss, const ModelGeometry& g, const QsaState& draft,
                                         size_t& bytes, std::string& error) {
    return conversation_snapshot_capture_bytes(reuse, view, ss, g, &draft, bytes, error);
}
bool conversation_snapshot_save(SavedConversation& image, const ConversationView& view, const SessionState& ss,
                                const ModelGeometry& g, const QsaState& draft, std::string& error,
                                ConversationKvReuse reuse, size_t* reused_bytes,
                                std::vector<ConversationRetainedPrefix>* retained) {
    return conversation_snapshot_save(image, view, ss, g, &draft, error, std::move(reuse), reused_bytes, retained);
}
bool conversation_snapshot_validate(const SavedConversation& image, const SessionState& ss, const ModelGeometry& g,
                                    const QsaState& draft, ConversationSnapshotExpectation expectation,
                                    std::string& error) {
    if (!image.stage_images.empty()) return fail(error, "a layer split's image restored as a single session");
    return conversation_snapshot_validate(image, ss, g, &draft, expectation, error);
}
ConversationRestore conversation_snapshot_restore(const SavedConversation& image, SessionState& ss,
                                                   const ModelGeometry& g, const QsaState& draft,
                                                   ConversationSnapshotExpectation expectation, std::string& error) {
    return conversation_snapshot_restore(image, ss, g, &draft, expectation, error);
}
bool conversation_snapshot_validate(const SavedConversation& image, const SessionState& ss, const ModelGeometry& g,
                                    const QsaState* draft, std::string& error) {
    return conversation_snapshot_validate(image, ss, g, draft, ConversationSnapshotExpectation::ordinary, error);
}
ConversationRestore conversation_snapshot_restore(const SavedConversation& image, SessionState& ss,
                                                   const ModelGeometry& g, const QsaState* draft, std::string& error,
                                                   const ConversationCheckpoint* selected) {
    return conversation_snapshot_restore(image, ss, g, draft, ConversationSnapshotExpectation::ordinary, error, selected);
}
bool conversation_snapshot_validate(const SavedConversation& image, const SessionState& ss, const ModelGeometry& g,
                                    const QsaState& draft, std::string& error) {
    // the whole-session form: a layer split's image (stage images) is restored stage by stage, never through here
    if (!image.stage_images.empty()) return fail(error, "a layer split's image restored as a single session");
    return conversation_snapshot_validate(image, ss, g, &draft, error);
}
ConversationRestore conversation_snapshot_restore(const SavedConversation& image, SessionState& ss,
                                                   const ModelGeometry& g, const QsaState& draft, std::string& error) {
    return conversation_snapshot_restore(image, ss, g, &draft, error);
}
ConversationRestore conversation_snapshot_restore_prefix(const SavedConversation& image, int64_t upto,
                                                          SessionState& ss, const ModelGeometry& g,
                                                          const QsaState& draft, std::string& error) {
    return conversation_snapshot_restore_prefix(image, upto, ss, g, &draft, error);
}
ConversationRestore conversation_snapshot_restore_prefix(const SavedConversation& image, int64_t upto,
                                                          SessionState& ss, const ModelGeometry& g,
                                                          const QsaState* draft, std::string& error) {
    // a layer split's image (stage images) is restored stage by stage, never in part
    if (!image.stage_images.empty()) {
        fail(error, "a layer split's image restored as a single session");
        return ConversationRestore::invalid;
    }
    if (!conversation_snapshot_validate(image, ss, g, draft, error)) return ConversationRestore::invalid;
    const int64_t captured = (int64_t) image.live.ids.size();
    if (upto < 1 || upto > captured) {
        fail(error, "the borrowed prefix is not inside the parked conversation");
        return ConversationRestore::invalid;
    }
    if (!sync(error)) return ConversationRestore::transfer_failed;
    for (size_t j = 0; j < owned_qsa(ss); ++j)
        if (!conversation_kv_restore_prefix(image.kv[j], owned(ss, j), g, captured, upto, true, error))
            return ConversationRestore::transfer_failed;
    if (draft && !conversation_kv_restore_prefix(image.kv.back(), *draft, g, captured, upto, false, error))
        return ConversationRestore::transfer_failed;
    return ConversationRestore::restored;
}
} // namespace strata::core
