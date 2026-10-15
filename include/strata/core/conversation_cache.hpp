// CPU-only ownership and matching policy for --serve's parked conversations.
// Token equality, image identity, and steering mode are all required for reuse.
#pragma once

#include "strata/core/conversation_buffer.hpp"
#include "strata/core/checkpoint_bytes.hpp"

#include <algorithm>
#include <cassert>
#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <utility>
#include <string_view>
#include <type_traits>
#include <vector>

namespace strata::core {

// Opaque frontend-issued authorization/conversation/render identity. The loaded
// engine's model, arithmetic and placement are immutable for this cache lifetime.
// An absent identity never authorizes prefix reuse, even for identical tokens.
struct ConversationScope {
    std::array<uint8_t, 32> value{};
    bool valid() const {
        return std::any_of(value.begin(), value.end(), [](uint8_t v) { return v != 0; });
    }
    bool matches(const ConversationScope& other) const { return valid() && *this == other; }
    bool operator==(const ConversationScope&) const = default;
};

inline bool conversation_scope_parse(std::string_view text, ConversationScope& out) {
    ConversationScope candidate;
    if (text.size() != 64) return false;
    auto hex = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    };
    for (size_t i = 0; i < candidate.value.size(); ++i) {
        const int hi = hex(text[2 * i]), lo = hex(text[2 * i + 1]);
        if (hi < 0 || lo < 0) return false;
        candidate.value[i] = uint8_t(16 * hi + lo);
    }
    if (!candidate.valid()) return false;
    out = candidate;
    return true;
}

struct ConversationImageKey {
    int64_t start = 0;
    uint64_t hash = 0;
    bool operator==(const ConversationImageKey&) const = default;
};

struct ConversationCheckpoint {
    std::vector<int32_t> ids;
    std::vector<ConversationImageKey> imgs;
    CheckpointBytes gdn, ple, tails, dead, block_pos;
    uint64_t used = 0; // upstream root-pinned/LRU checkpoint retention
    // A shared-prefix pin (the request key pin=N): this checkpoint is the read-only prefix many suffix queries branch
    // from, so retention never evicts it (conv_cache.hpp) and a parked conversation holding it stays parked.  A run-time
    // mark only: it is not in the session file, a request that pins the same prefix again sets it.
    bool pinned = false;
    // Ordinary layer-split checkpoints retain each device's running state.
    // Whole-session parking is currently single-GPU and rejects these parts.
    std::vector<ConversationCheckpoint> stage_parts;

    void seal() {
        gdn.seal(); ple.seal(); tails.seal(); dead.seal(); block_pos.seal();
        for (auto& part : stage_parts) part.seal();
    }
    // Freeze every payload for immutable sharing (ConversationBuffer::freeze):
    // the state arrays seal into shared allocations, this checkpoint's and every
    // stage part's. No byte is copied; a later write detaches first, so an alias
    // taken afterwards never observes a byte written through this checkpoint.
    void freeze() {
        gdn.seal(); ple.seal(); tails.seal(); dead.seal(); block_pos.seal();
        for (auto& part : stage_parts) part.freeze();
    }
    /// Cheap alias: the token/image/LRU metadata is copied, every payload is
    /// shared with this (frozen) checkpoint.
    ConversationCheckpoint frozen_alias() const {
        ConversationCheckpoint out;
        out.ids = ids; out.imgs = imgs; out.used = used; out.pinned = pinned;
        out.gdn = gdn; out.ple = ple; out.tails = tails; out.dead = dead; out.block_pos = block_pos;
        out.stage_parts.reserve(stage_parts.size());
        for (const auto& part : stage_parts) out.stage_parts.push_back(part.frozen_alias());
        return out;
    }

    size_t bytes() const {
        size_t n = ids.capacity() * sizeof(int32_t) + imgs.capacity() * sizeof(ConversationImageKey) +
               gdn.bytes() + ple.bytes() + tails.bytes() + dead.bytes() + block_pos.bytes() +
               stage_parts.capacity() * sizeof(ConversationCheckpoint);
        for (const auto& part : stage_parts) n += part.bytes();
        return n;
    }
};

// Identity-layout K/V pages and completed indexer rows. For streamed layers the
// source is the authoritative host pool, NOT the replaceable VRAM slots.
struct ConversationKv {
    int format = 0;
    int64_t cells = 0, heads = 0, head_dim = 0, page_size = 0, pooled_rows = 0, idx_dim = 0;
    ConversationBuffer k, v, k_scale, v_scale, pooled;
    void freeze() { k.freeze(); v.freeze(); k_scale.freeze(); v_scale.freeze(); pooled.freeze(); }
    /// Cheap alias: the layer's identity fields are copied, its payloads shared.
    ConversationKv frozen_alias() const {
        ConversationKv out;
        out.format = format; out.cells = cells; out.heads = heads; out.head_dim = head_dim;
        out.page_size = page_size; out.pooled_rows = pooled_rows; out.idx_dim = idx_dim;
        out.k = k.frozen_alias(); out.v = v.frozen_alias();
        out.k_scale = k_scale.frozen_alias(); out.v_scale = v_scale.frozen_alias();
        out.pooled = pooled.frozen_alias();
        return out;
    }
    size_t bytes() const {
        return k.bytes() + v.bytes() + k_scale.bytes() + v_scale.bytes() + pooled.bytes();
    }
};

struct ConversationKvReuse {
    std::vector<ConversationKv> kv;
    // Original image extent for validation, and the earliest subsequent rewrite.
    int64_t captured_tokens = 0, unchanged_tokens = 0;
    // With a layer split: the later stages' own retained K/V, one per stage, same extents (empty: none)
    std::vector<ConversationKvReuse> stages;
    size_t bytes() const {
        size_t n = kv.capacity() * sizeof(ConversationKv) + stages.capacity() * sizeof(ConversationKvReuse);
        for (const auto& layer : kv) n += layer.bytes();
        for (const auto& s : stages) n += s.bytes();
        return n;
    }
};

/// Layer-split parking: each running checkpoint holds the first stage's state and, in `stage_parts`, one part per
/// later stage; a parked conversation keeps them in one image per stage.  `split` MOVES the checkpoints apart (no
/// running state is copied): selected complete checkpoints into `stage0` and `parts[k]`,
/// every part carrying ids / images / LRU / pins, the others aside. `merge` restores
/// the entire original chain, including cuts omitted from the parked image.
struct ConversationCheckpointSplit {
    std::vector<ConversationCheckpoint> stage0;
    std::vector<std::vector<ConversationCheckpoint>> parts;   ///< [stage][checkpoint]
    std::vector<ConversationCheckpoint> rest;                 ///< incomplete or not selected for capture
    std::vector<bool> complete;                               ///< per original position: in stage0/parts or in rest
};
enum class ConversationCheckpointRetention { all, parked_history };

inline ConversationCheckpointSplit conversation_checkpoints_split(std::vector<ConversationCheckpoint>&& checks,
                                                                  size_t stages, ConversationCheckpointRetention retention) {
    ConversationCheckpointSplit out;
    out.parts.resize(stages);
    out.complete.reserve(checks.size());
    size_t root = checks.size(), newest = checks.size();
    if (retention == ConversationCheckpointRetention::parked_history) {
        for (size_t i = 0; i < checks.size(); ++i) {
            if (checks[i].stage_parts.size() != stages) continue;
            if (root == checks.size()) root = i;
            if (newest == checks.size() || checks[i].ids.size() >= checks[newest].ids.size()) newest = i;
        }
    }
    size_t whole = 0;
    for (size_t i = 0; i < checks.size(); ++i) {
        const auto& c = checks[i];
        // Keep the root, latest cut and every explicit pin in RAM parking.
        // Older unpinned rewind cuts stay working but need not multiply across
        // every queued history. This never manufactures a recurrent state.
        const bool complete = c.stage_parts.size() == stages &&
            (retention == ConversationCheckpointRetention::all || i == root || i == newest || c.pinned);
        out.complete.push_back(complete);
        whole += complete ? 1 : 0;
    }
    out.stage0.reserve(whole);
    out.rest.reserve(checks.size() - whole);
    for (auto& parts : out.parts) parts.reserve(whole);
    // Allocate and copy identity metadata before moving any running state.
    // If allocation fails, every input checkpoint remains usable by the caller.
    for (size_t i = 0; i < checks.size(); ++i) {
        const auto& c = checks[i];
        if (!out.complete[i]) continue;
        for (size_t k = 0; k < stages; ++k) {
            ConversationCheckpoint part;
            part.ids = c.ids; part.imgs = c.imgs; part.used = c.used; part.pinned = c.pinned;
            out.parts[k].push_back(std::move(part));
        }
    }
    size_t w = 0;
    for (size_t i = 0; i < checks.size(); ++i) {
        auto& c = checks[i];
        if (!out.complete[i]) { out.rest.push_back(std::move(c)); continue; }
        for (size_t k = 0; k < stages; ++k) {
            auto& dst = out.parts[k][w];
            auto& src = c.stage_parts[k];
            dst.gdn = std::move(src.gdn); dst.ple = std::move(src.ple);
            dst.tails = std::move(src.tails); dst.dead = std::move(src.dead);
            dst.block_pos = std::move(src.block_pos);
        }
        c.stage_parts.clear();
        out.stage0.push_back(std::move(c));
        ++w;
    }
    checks.clear();
    return out;
}

/// false (and `out` empty) when the parts do not line up: a stage with another number of checkpoints, or a part
/// whose ids differ from its checkpoint's.
inline bool conversation_checkpoints_merge(ConversationCheckpointSplit&& split, std::vector<ConversationCheckpoint>& out) {
    out.clear();
    for (const auto& p : split.parts)
        if (p.size() != split.stage0.size()) return false;
    size_t whole = 0, other = 0;
    for (bool c : split.complete) (c ? whole : other) += 1;
    if (!split.complete.empty() && (whole != split.stage0.size() || other != split.rest.size())) return false;
    if (split.complete.empty()) split.complete.assign(split.stage0.size(), true);   // a restored image: all whole
    for (size_t j = 0; j < split.stage0.size(); ++j)
        for (const auto& p : split.parts)
            if (p[j].ids != split.stage0[j].ids) return false;
    size_t w = 0, r = 0;
    for (bool c : split.complete) {
        if (!c) { out.push_back(std::move(split.rest[r++])); continue; }
        ConversationCheckpoint cp = std::move(split.stage0[w]);
        for (auto& p : split.parts) cp.stage_parts.push_back(std::move(p[w]));
        ++w;
        out.push_back(std::move(cp));
    }
    return true;
}

struct SavedConversation {
    // RAM-only namespace; session files keep their existing model/config schema.
    ConversationScope scope;
    // Runtime compatibility only; NOT a model/weights identity or disk schema.
    std::array<int64_t, 18> geometry{};
    // The session's layer carve the image was captured from ([0, n_layers) on one GPU); restore requires the same.
    int64_t layer_lo = 0, layer_hi = 0;
    ConversationCheckpoint live;
    std::vector<ConversationCheckpoint> checkpoints;
    std::vector<ConversationKv> kv; // main layers followed by the draft layer
    bool cvec = true;
    // with a layer split, the later stages' own images, one per stage, in stage order
    std::vector<SavedConversation> stage_images;

    /// Freeze every payload this image owns for immutable sharing: the running
    /// checkpoint, every parked cut, every K/V layer and every stage image. No
    /// byte is copied.
    void freeze() {
        live.freeze();
        for (auto& c : checkpoints) c.freeze();
        for (auto& layer : kv) layer.freeze();
        for (auto& stage : stage_images) stage.freeze();
    }
    /// Cheap alias: metadata is copied, every payload is shared with this image
    /// (which must be frozen for the alias to be free).
    SavedConversation frozen_alias() const {
        SavedConversation out;
        out.scope = scope; out.geometry = geometry;
        out.layer_lo = layer_lo; out.layer_hi = layer_hi;
        out.live = live.frozen_alias();
        out.checkpoints.reserve(checkpoints.size());
        for (const auto& c : checkpoints) out.checkpoints.push_back(c.frozen_alias());
        out.kv.reserve(kv.size());
        for (const auto& layer : kv) out.kv.push_back(layer.frozen_alias());
        out.cvec = cvec;
        out.stage_images.reserve(stage_images.size());
        for (const auto& stage : stage_images) out.stage_images.push_back(stage.frozen_alias());
        return out;
    }

    /// Holds a pinned shared prefix (see ConversationCheckpoint::pinned): the parked-conversation budget keeps it.
    bool pinned() const {
        for (const auto& c : checkpoints) if (c.pinned) return true;
        return false;
    }

    size_t bytes() const {
        size_t n = sizeof(scope) + live.bytes() + checkpoints.capacity() * sizeof(ConversationCheckpoint) +
                   kv.capacity() * sizeof(ConversationKv);
        for (const auto& s : stage_images) n += s.bytes();
        for (const auto& c : checkpoints) n += c.bytes();
        for (const auto& k : kv) n += k.bytes();
        return n;
    }
};

/// Freeze a captured image in place and return an immutable alias that shares its
/// payloads: the NVMe store's publication lease. No payload byte is copied (only
/// token/image metadata and the frozen segment directories). The returned image
/// keeps observing exactly the bytes frozen here even if `image` is later mutated
/// (that mutation detaches its own allocation first) or moved from (the shared
/// allocations outlive the move through this handle). The store treats the image
/// as const and must drop the handle when its publication ends.
inline std::shared_ptr<const SavedConversation> conversation_image_freeze_share(SavedConversation& image) {
    image.freeze();
    return std::make_shared<const SavedConversation>(image.frozen_alias());
}

template<class Token>
int64_t conversation_prefix(const ConversationCheckpoint& c, const std::vector<Token>& prompt,
                            const std::vector<ConversationImageKey>& images) {
    const size_t n = c.ids.size();
    // The last prompt token always starts the next verify window.
    if (n == 0 || n >= prompt.size() || !std::equal(c.ids.begin(), c.ids.end(), prompt.begin())) return 0;
    size_t j = 0;
    for (const auto& image : images) {
        if (image.start >= (int64_t) n) continue;
        if (j == c.imgs.size() || !(c.imgs[j++] == image)) return 0;
    }
    if (j != c.imgs.size()) return 0;
    return (int64_t) n;
}

class ConversationCache {
public:
    struct Match {
        size_t index = 0;
        int64_t tokens = 0;
        bool live = false;
    };

    ConversationCache(size_t budget, size_t slots, int64_t min_tokens = 0)
        : budget_(budget), slots_(slots), min_tokens_(min_tokens) {}
    bool enabled() const { return budget_ != 0 && slots_ != 0; }
    /// `--conversation-cache-min-tokens`: whether a conversation of this length is worth one of `slots`.  Every
    /// parked conversation costs a whole slot whatever its length, so a client that interleaves short side requests
    /// with one long conversation spends the cache on the side requests and evicts the long one (oldest first) -
    /// exactly the reuse it wanted.  0 parks every conversation, whatever its size.
    bool wants(int64_t tokens) const { return min_tokens_ <= 0 || tokens >= min_tokens_; }
    size_t bytes() const { return bytes_ + reuse_.bytes(); }
    size_t size() const { return entries_.size(); }
    size_t evictions() const { return evictions_; }

    // ---- Immutable publication leases -------------------------------------
    // The store holds an immutable alias of a parked image's payloads while it
    // writes them (conversation_image_freeze_share). The lease is charged in full
    // even though its payload is physically shared with the parked entry - the
    // same "every reference in full" convention the buffers use - so the envelope
    // never over-commits; the peak is one admitted writer plus one incoming
    // restore, and release is idempotent per reservation. Refusal is a safe cache
    // miss: the caller skips publication, not the request.
    bool try_reserve_lease(size_t bytes) {
        if (bytes > budget_ - std::min(budget_, held_bytes_)) return false;
        held_bytes_ += bytes;
        return true;
    }
    void release_lease(size_t bytes) { held_bytes_ -= std::min(held_bytes_, bytes); }
    size_t leased_bytes() const { return held_bytes_; }
    /// What the configured budget has to cover: payload + retained reuse + leases.
    size_t envelope_bytes() const { return bytes() + held_bytes_; }
    // the longest parked conversation, in tokens (--kv-grow keeps the K/V that long while it could be restored)
    int64_t longest_tokens() const {
        int64_t n = 0;
        for (const auto& e : entries_) n = std::max<int64_t>(n, (int64_t) e.live.ids.size());
        return n;
    }

    // Retain only the restored K/V buffers, not duplicate running checkpoints.
    // This optimization never evicts a parked conversation to make itself fit.
    // `stage_kv`: with a layer split, the later stages' restored K/V (one per stage), retained with the first's.
    void retain(std::vector<ConversationKv>&& kv, int64_t tokens,
                std::vector<std::vector<ConversationKv>>&& stage_kv = {}) {
        reuse_ = {};
        ConversationKvReuse candidate{std::move(kv), tokens, tokens, {}};
        for (auto& k : stage_kv) candidate.stages.push_back(ConversationKvReuse{std::move(k), tokens, tokens, {}});
        if (enabled() && candidate.bytes() <= budget_ - bytes_) reuse_ = std::move(candidate);
    }
    void limit_reuse(int64_t first_dirty) {
        reuse_.unchanged_tokens = std::min(reuse_.unchanged_tokens, first_dirty);
        for (auto& s : reuse_.stages) s.unchanged_tokens = std::min(s.unchanged_tokens, first_dirty);
        if (reuse_.unchanged_tokens <= 0) reuse_ = {};
    }
    ConversationKvReuse take_reuse() { return std::exchange(reuse_, {}); }
    size_t retained_bytes() const { return reuse_.bytes(); }
    int64_t unchanged_tokens() const { return reuse_.kv.empty() ? 0 : reuse_.unchanged_tokens; }
    bool can_fit(size_t incoming, size_t held = 0) const {
        return enabled() && held <= budget_ && incoming <= room(held) &&
               entries_.size() < slots_ && bytes() <= room(held) - incoming;
    }

    // Read-only enumeration: a caller can validate and price every authorized source
    // before take(), eviction, or an outgoing-state overwrite.
    template<class Token, class Visitor>
    void for_each_match(const std::vector<Token>& prompt, const std::vector<ConversationImageKey>& images,
                        bool cvec, const ConversationScope& scope, Visitor&& visit) const {
        if (!scope.valid()) return;
        for (size_t i = entries_.size(); i-- > 0;) {
            const auto& e = entries_[i];
            if (!e.scope.matches(scope) || e.cvec != cvec) continue;
            auto consider = [&](const ConversationCheckpoint& c, bool live) {
                const int64_t n = conversation_prefix(c, prompt, images);
                if (n > 0) visit(Match{i, n, live}, e, c);
            };
            consider(e.live, true);
            for (const auto& c : e.checkpoints) consider(c, false);
        }
    }

    template<class Token>
    Match best(const std::vector<Token>& prompt, const std::vector<ConversationImageKey>& images, bool cvec,
               const ConversationScope& scope) const {
        Match best;
        // Ties prefer the newest parked branch; the caller prefers an equally
        // long active or resident prefix.
        for_each_match(prompt, images, cvec, scope, [&](Match match, const auto&, const auto&) {
            if (match.tokens > best.tokens) best = match;
        });
        return best;
    }

    // Functional observers borrow only synchronous const references; never retain them across cache mutation.
    template<class Visitor> void for_each_image(Visitor&& visit) const {
        for (size_t i = 0; i < entries_.size(); ++i) visit(i, entries_[i]);
    }

    SavedConversation take(size_t index) {
        SavedConversation out = std::move(entries_.at(index));
        bytes_ -= out.bytes();
        entries_.erase(entries_.begin() + (std::ptrdiff_t) index);
        return out;
    }

    // Admit capture bytes, not a published entry. A full entry table must not
    // evict a queued history before capture can fail or replace its older image.
    // held is a taken incoming image, still alive and fully charged during capture.
    bool reserve_snapshot(size_t incoming, size_t held = 0) {
        if (!enabled() || held > budget_ || incoming > room(held)) return false;
        if (bytes() > room(held) - incoming) reuse_ = {};
        while (!entries_.empty() && bytes_ > room(held) - incoming) {
            // the oldest entry that does not hold a pinned shared prefix leaves; with only pinned ones left the new
            // image does not fit (the caller skips parking it - the pinned prefix is what the queries come back to)
            if (!evict_oldest()) return false;
        }
        return true;
    }

    // Release the oldest unpinned image for byte/physical-memory pressure.
    // Buffer extents are released before the caller resamples available memory.
    // False when none can go (empty, or only entries that hold a pinned shared prefix are left).
    bool evict_oldest() {
        auto victim = std::find_if(entries_.begin(), entries_.end(), [](const SavedConversation& e) { return !e.pinned(); });
        if (victim == entries_.end()) return false;
        bytes_ -= victim->bytes();
        entries_.erase(victim);
        ++evictions_;
        return true;
    }

    // #879: every parked conversation (pinned ones too) and the retained K/V go - after non-finite logits any of them
    // may carry the poison.  Returns how many were parked.
    size_t clear() {
        const size_t n = entries_.size();
        entries_.clear();
        bytes_ = 0;
        reuse_ = {};
        return n;
    }

    // The slot count, so a caller that evicts in a loop has a bound it did not invent.
    size_t slots() const { return slots_; }

    // Latest-history retention, not state-equivalent deduplication: an older
    // unpinned rewind cut may cold-read after an exact live-prefix extension.
    // Divergent branches and every explicitly pinned cut remain protected.
    size_t superseded() const { return superseded_; }

    // A non-consuming restore also reserves its eventual publication entry.
    bool put(SavedConversation&& image, size_t held = 0, size_t held_entries = 0) {
        static_assert(std::is_nothrow_move_constructible_v<SavedConversation>);
        static_assert(std::is_nothrow_move_assignable_v<SavedConversation>);
        static_assert(std::is_nothrow_destructible_v<SavedConversation>);
        const size_t n = image.bytes();
        if (!enabled() || !image.scope.valid() || held_entries >= slots_ ||
            held > budget_ || n > room(held)) return false;
        const size_t available_entries = slots_ - held_entries;
        const size_t limit = room(held) - n;
        size_t count = entries_.size(), remaining = bytes_, replaced = 0, victims = 0;
        for (const auto& old : entries_) if (replaces(old, image)) {
            --count;
            remaining -= old.bytes();
            ++replaced;
        }
        const size_t reuse_bytes = reuse_.bytes();
        const bool discard_reuse = remaining > limit || reuse_bytes > limit - remaining;
        const size_t payload_limit = discard_reuse ? limit : limit - reuse_bytes;
        for (const auto& old : entries_) {
            if (count < available_entries && remaining <= payload_limit) break;
            if (!old.pinned() && !replaces(old, image)) {
                --count;
                remaining -= old.bytes();
                ++victims;
            }
        }
        if (count >= available_entries || remaining > payload_limit) return false;

        // Allocate/construct publication storage before destroying any old image.
        // noexcept movement keeps the candidate unchanged if deque growth fails.
        const size_t old_count = entries_.size();
        entries_.push_back(std::move(image));
        const auto& next = entries_.back();
        size_t write = 0, ordinary_left = victims;
        // Compact old entries without erase/pop: next stays at its stable final
        // index until every comparison is done, and each survivor moves at most once.
        for (size_t read = 0; read < old_count; ++read) {
            const auto& old = entries_[read];
            if (replaces(old, next)) continue;
            if (ordinary_left && !old.pinned()) { --ordinary_left; continue; }
            if (write != read) entries_[write] = std::move(entries_[read]);
            ++write;
        }
        assert(ordinary_left == 0);
        if (write != old_count) entries_[write] = std::move(entries_.back());
        while (entries_.size() > write + 1) entries_.pop_back();
        if (discard_reuse) reuse_ = {};
        bytes_ = remaining + n;
        superseded_ += replaced;
        evictions_ += victims;
        return true;
    }

private:
    static bool same_pin(const ConversationCheckpoint& a, const ConversationCheckpoint& b) noexcept {
        if (a.pinned != b.pinned || a.ids != b.ids || a.imgs != b.imgs ||
            a.stage_parts.size() != b.stage_parts.size()) return false;
        for (size_t i = 0; i < a.stage_parts.size(); ++i)
            if (!same_pin(a.stage_parts[i], b.stage_parts[i])) return false;
        return true;
    }
    static bool covered_live(const SavedConversation& old, const SavedConversation& next) noexcept {
        return !old.live.ids.empty() &&
            ((old.live.ids == next.live.ids && old.live.imgs == next.live.imgs) ||
             conversation_prefix(old.live, next.live.ids, next.live.imgs) > 0);
    }
    static bool compatible_pins(const SavedConversation& old, const SavedConversation& next) noexcept {
        if (!covered_live(old, next) || old.geometry != next.geometry || old.layer_lo != next.layer_lo ||
            old.layer_hi != next.layer_hi || old.cvec != next.cvec || old.kv.size() != next.kv.size() ||
            old.stage_images.size() != next.stage_images.size()) return false;
        for (size_t i = 0; i < old.kv.size(); ++i) {
            const auto& a = old.kv[i];
            const auto& b = next.kv[i];
            if (a.format != b.format || a.heads != b.heads || a.head_dim != b.head_dim ||
                a.page_size != b.page_size || a.idx_dim != b.idx_dim) return false;
        }
        for (const auto& pin : old.checkpoints) {
            if (!pin.pinned) continue;
            if (std::none_of(next.checkpoints.begin(), next.checkpoints.end(),
                             [&](const auto& candidate) { return same_pin(pin, candidate); })) return false;
        }
        for (size_t i = 0; i < old.stage_images.size(); ++i)
            if (!compatible_pins(old.stage_images[i], next.stage_images[i])) return false;
        return true;
    }
    static bool replaces(const SavedConversation& old, const SavedConversation& next) noexcept {
        return old.scope.matches(next.scope) && compatible_pins(old, next);
    }

    // Bytes the envelope can still admit for a new image: budget minus this
    // image's own held charge, the outstanding publication leases and nothing
    // else (0 when the charges alone already exceed the budget).
    size_t room(size_t held) const {
        if (held_bytes_ > budget_ || held > budget_ - held_bytes_) return 0;
        return budget_ - held_bytes_ - held;
    }

    size_t budget_ = 0, slots_ = 0, bytes_ = 0, evictions_ = 0, superseded_ = 0;
    size_t held_bytes_ = 0;    // outstanding immutable publication leases (see try_reserve_lease)
    int64_t min_tokens_ = 0;   // --conversation-cache-min-tokens: 0 parks every conversation, whatever its size
    std::deque<SavedConversation> entries_; // least recently active first
    ConversationKvReuse reuse_;
};

} // namespace strata::core
