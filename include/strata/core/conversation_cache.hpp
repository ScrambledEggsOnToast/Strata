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
#include <optional>
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

inline bool conversation_shared_scope_parse(std::string_view text, ConversationScope& out) {
    if (std::any_of(text.begin(), text.end(), [](char c) {
            return !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'));
        })) return false;
    return conversation_scope_parse(text, out);
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
    // --prefill-pipe on a layer split: the plan a prompt segment [from, to) is read in - its chunk (0: none), the loan
    // its buffers were laid out for (which experts the prompt reads in place and which it streams: that sets bits
    // too), the pipe value it was planned with, whether it is one chunk and whether the plan changed the buffers'
    // chunks.  A periodic checkpoint taken inside such a segment keeps it, so a read that goes on from there reads the
    // rest in the same chunks with the same loan.  Run-time only, like `pinned` (a layer split saves no session files).
    struct PromptPlan {
        int64_t chunk = 0, from = -1, to = -1, lend = 0;
        double pipe = 0.0;
        bool single = false, chunked = false;
        bool operator==(const PromptPlan&) const = default;
    } plan;
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
        out.ids = ids; out.imgs = imgs; out.used = used; out.pinned = pinned; out.plan = plan;
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
    // Private captures use page-rounded cells. An exact shared draft cut uses
    // cells == live.ids.size()-1: its meaningful extent is explicit in the same
    // serialized field, while the rounded payload contains zeros afterwards.
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
    // Provenance of these bytes: the persistent-store head this K/V was retained from and the execution-domain epoch
    // it was captured in. 0 (a RAM-only restore) authorizes no reuse interval: nothing else can name the image these
    // buffers came from, and a donor that cannot be named is never guessed at.
    uint64_t donor_head = 0;
    uint64_t donor_epoch = 0;
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
            part.ids = c.ids; part.imgs = c.imgs; part.used = c.used; part.pinned = c.pinned; part.plan = c.plan;
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

// A single contiguous interval of completed writes. Rewrites abandon the old
// suffix; a gap starts a new interval, never a min/max union over missing cells.
struct ConversationDraftCoverage {
    int64_t begin = -1, cells = -1;
    int32_t next_token = -1;
    bool known() const { return begin >= 0 && cells >= begin; }
    void clear() { begin = cells = -1; next_token = -1; }
    void completed(int64_t first, int64_t end, int32_t token) {
        if (first < 0 || end < first) { clear(); return; }
        if (!known() || first < begin || first > cells) begin = first;
        cells = end;
        next_token = end > begin ? token : -1;
    }
    void rewind(int64_t end) {
        if (!known() || end < begin || end > cells) { clear(); return; }
        if (end != cells) next_token = -1;
        cells = end;
    }
};

struct SavedConversation {
    // Incremental inline metadata charge for this certification extension.
    // Embedded stage objects already pay it through their directory capacity.
    static constexpr size_t draft_metadata_bytes =
        sizeof(int64_t) * 2 + sizeof(int32_t) + sizeof(ConversationBuffer);
    // RAM-only namespace; session files keep their existing model/config schema.
    ConversationScope scope;
    // Transient sole publication owner; never serialized or propagated to an alias.
    uint64_t publication_ticket = 0;
    // Runtime compatibility only; NOT a model/weights identity or disk schema.
    std::array<int64_t, 18> geometry{};
    // The session's layer carve the image was captured from ([0, n_layers) on one GPU); restore requires the same.
    int64_t layer_lo = 0, layer_hi = 0;
    ConversationCheckpoint live;
    std::vector<ConversationCheckpoint> checkpoints;
    std::vector<ConversationKv> kv; // main layers followed by the draft layer
    bool cvec = true;
    // Certified populated draft interval, not the rounded allocation extent.
    // Unknown coverage must refuse direct source reuse before any destination write.
    int64_t draft_begin = -1, draft_cells = -1;
    // The token consumed with the last retained main residual when draft_cells == live length.
    int32_t draft_next_token = -1;
    // Immutable R_(live length-1), never the destination's private next token.
    // Only the draft-owning stage may carry it, with cells == live length-1.
    ConversationBuffer draft_boundary_residual;
    // with a layer split, the later stages' own images, one per stage, in stage order
    std::vector<SavedConversation> stage_images;

    /// Freeze every payload this image owns for immutable sharing: the running
    /// checkpoint, every parked cut, every K/V layer and every stage image. No
    /// byte is copied.
    void freeze() {
        live.freeze();
        for (auto& c : checkpoints) c.freeze();
        for (auto& layer : kv) layer.freeze();
        draft_boundary_residual.freeze();
        for (auto& stage : stage_images) stage.freeze();
    }
    /// Cheap alias: metadata is copied, every payload is shared with this image
    /// (which must be frozen for the alias to be free). Publication ownership is not copied.
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
        out.draft_begin = draft_begin; out.draft_cells = draft_cells;
        out.draft_next_token = draft_next_token;
        out.draft_boundary_residual = draft_boundary_residual.frozen_alias();
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
        size_t n = sizeof(scope) + draft_metadata_bytes + draft_boundary_residual.bytes() + live.bytes() +
                   checkpoints.capacity() * sizeof(ConversationCheckpoint) + kv.capacity() * sizeof(ConversationKv) +
                   stage_images.capacity() * sizeof(SavedConversation);
        // Each directory slot already owns its stage's inline metadata/buffer.
        for (const auto& s : stage_images)
            n += s.bytes() - sizeof(s.scope) - draft_metadata_bytes;
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
        uint64_t serial = 0;   // the entry's identity; its index shifts when entries before it go
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

    // ---- Fixed ownership and immutable publication leases ------------------
    // Waiting publications protect their existing parked owner, without a
    // second image or charge. take_publication transfers that exact charge to a
    // held lease when the writer acquires it. Separately reserved aliases remain
    // charged in full, so every live ownership path stays inside one envelope.
    // A startup resident-checkpoint lease uses the same held ownership and is
    // never refunded by request/publication cleanup. It reduces RAM parking
    // and read headroom for the entire cache lifetime, including poison clears.
    bool try_reserve_lease(size_t bytes) {
        const size_t occupied = envelope_bytes();
        if (occupied > budget_ || bytes > budget_ - occupied) return false;
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
    // `donor_head`/`donor_epoch` name the persistent-store image these bytes were restored from (0: a RAM-only
    // restore), so a later capture can offer exactly the bytes it kept from it as reuse intervals. The bytes here are
    // charged in full by envelope_bytes(): they are still live host memory even though no parked entry owns them.
    void retain(std::vector<ConversationKv>&& kv, int64_t tokens,
                std::vector<std::vector<ConversationKv>>&& stage_kv = {},
                uint64_t donor_head = 0, uint64_t donor_epoch = 0) {
        reuse_ = {};
        ConversationKvReuse candidate{std::move(kv), tokens, tokens, {}};
        candidate.donor_head = donor_head;
        candidate.donor_epoch = donor_epoch;
        for (auto& k : stage_kv) {
            ConversationKvReuse part{std::move(k), tokens, tokens, {}};
            part.donor_head = donor_head;
            part.donor_epoch = donor_epoch;
            candidate.stages.push_back(std::move(part));
        }
        const size_t available = room(0);
        if (enabled() && bytes_ <= available && candidate.bytes() <= available - bytes_)
            reuse_ = std::move(candidate);
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
                if (n > 0) visit(Match{i, n, live, serials_[i]}, e, c);
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

    // A pending publication keeps its sole parked owner: restore gets a frozen
    // alias and must not reserve a return slot or put that alias back afterwards.
    SavedConversation take(size_t index, bool& retained_in_cache) {
        retained_in_cache = false;
        auto& entry = entries_.at(index);
        if (entry.publication_ticket != 0) {
            auto alias = entry.frozen_alias();
            retained_in_cache = true;
            return alias;
        }
        SavedConversation out = std::move(entry);
        bytes_ -= out.bytes();
        erase(index);
        return out;
    }

    bool has_publication(uint64_t ticket) const noexcept {
        if (ticket == 0) return false;
        return std::any_of(entries_.begin(), entries_.end(), [&](const auto& entry) {
            return entry.publication_ticket == ticket;
        });
    }

    // The caller allocates its shared root first. Commit only moves ownership:
    // no allocation or payload copy, and the total envelope stays identical.
    std::optional<SavedConversation> take_publication(uint64_t ticket) noexcept {
        if (ticket == 0) return std::nullopt;
        for (size_t i = 0; i < entries_.size(); ++i) {
            auto& entry = entries_[i];
            if (entry.publication_ticket != ticket) continue;
            if (guarded_ != 0 && serials_[i] == guarded_) return std::nullopt;
            const size_t n = entry.bytes();
            std::optional<SavedConversation> out(std::in_place, std::move(entry));
            bytes_ -= n;
            held_bytes_ += n;
            erase(i);
            return out;
        }
        return std::nullopt;
    }

    // A checkpoint inside a parked conversation that a NEW conversation starts with - a sibling subagent sharing the
    // system prompt and tool list, a compacted history - is borrowed: restored from the entry, which stays parked.
    // take() would move the whole conversation into the session and the new request would overwrite everything after
    // that checkpoint, so the other conversation's next turn would read its history again.  A match on the entry's
    // live state or on its deepest checkpoint is the same conversation going on (the deepest one: its client
    // re-rendered the last reply), whose tail is stale: that is still taken.  So is a match that reaches a pinned
    // shared prefix (pin=N): a borrow copies the checkpoints up to the match, and taking it instead keeps one
    // conversation holding the pin, as pin=N expects.
    bool borrows(const Match& m) const {
        if (m.tokens <= 0 || m.live || m.index >= entries_.size() || serials_[m.index] != m.serial) return false;
        size_t deepest = 0;
        for (const auto& c : entries_[m.index].checkpoints) {
            if (c.pinned && (int64_t) c.ids.size() <= m.tokens) return false;
            deepest = std::max(deepest, c.ids.size());
        }
        return (size_t) m.tokens < deepest;
    }
    // The entry with this identity: its index (SIZE_MAX: gone), or the entry itself (nullptr: gone; the pointer is
    // valid until the cache changes).
    size_t index_of(uint64_t serial) const {
        for (size_t i = 0; i < entries_.size(); ++i)
            if (serials_[i] == serial) return i;
        return SIZE_MAX;
    }
    const SavedConversation* find(uint64_t serial) const {
        const size_t i = index_of(serial);
        return i == SIZE_MAX ? nullptr : &entries_[i];
    }
    // While a borrowed entry (the donor) is being restored, parking the outgoing conversation must not evict or drop
    // it: guard() it.  When the outgoing image fits only without it, make_room refuses before evicting anything and
    // guard_blocked() says so: the caller then takes the entry as before, so the conversation that was active a
    // moment ago keeps its place.
    void guard(uint64_t serial) { guarded_ = serial; guard_blocked_ = false; }
    void unguard() { guarded_ = 0; }
    bool guard_blocked() const { return guard_blocked_; }
    size_t borrowed() const { return borrowed_; }
    void count_borrow() { ++borrowed_; }

    // Reserve before allocating a snapshot. held is an incoming image removed
    // with take() but still alive during the exchange; count it against RAM too.
    bool make_room(size_t incoming, size_t held = 0) {
        guard_blocked_ = false;
        if (!enabled() || held > budget_ || incoming > room(held)) return false;
        const size_t limit = room(held) - incoming;
        size_t remaining = bytes_, count = entries_.size();
        for (size_t i = 0; i < entries_.size() && (count >= slots_ || remaining > limit); ++i) {
            if (!evictable(i)) continue;
            --count;
            remaining -= entries_[i].bytes();
        }
        if (count >= slots_ || remaining > limit) {
            guard_blocked_ = guarded_ != 0 && index_of(guarded_) != SIZE_MAX;
            return false; // no owner or optional reuse is discarded by an impossible reservation.
        }
        if (bytes() > limit) reuse_ = {};
        while (entries_.size() >= slots_ || bytes_ > limit) {
            if (!evict_oldest()) return false;
        }
        return true;
    }

    // Drop/invalidation removes only parked protection, never a separately
    // acquired lease. Missing tickets (including already acquired ones) are safe.
    void release_publication(uint64_t ticket) noexcept {
        if (ticket == 0) return;
        for (auto& entry : entries_) if (entry.publication_ticket == ticket) {
            entry.publication_ticket = 0;
            return;
        }
    }

    // Admit snapshot/materialization bytes, not a parked entry. Plan before mutation:
    // impossible reservations preserve every owner, retained reuse and the fallback index.
    // held is outside ownership, still alive and fully charged. A protected parked
    // fallback survives successful reclamation and its index follows compaction.
    //
    // ADMISSION CONTRACT (#250). Pins and pending publications are never spent: evictable() rejects them, so a cache
    // whose every parked entry is pinned refuses here (and in put()/make_room()) rather than choosing a victim. There
    // is deliberately no all-pinned fallback, unlike the working chain's own eviction (program/conv_cache.hpp): the
    // pin=N promise is that the image stays parked, and silently dropping it would break a concurrent sibling's
    // reuse. The carrier skips parking and the request still completes; the skip line names how many entries are
    // pinned. Only clear() drops pins, and only for the non-finite-logit poison path.
    bool reserve_snapshot(size_t incoming, size_t held = 0, size_t* protected_index = nullptr) {
        if (!enabled() || held > budget_ || incoming > room(held) ||
            (protected_index && *protected_index >= entries_.size())) return false;
        const size_t limit = room(held) - incoming;
        if (bytes() <= limit) return true;
        const size_t keep = protected_index ? *protected_index : entries_.size();
        size_t remaining = bytes_, victims = 0;
        for (size_t i = 0; i < entries_.size() && remaining > limit; ++i) {
            if (i != keep && evictable(i)) {
                remaining -= entries_[i].bytes();
                ++victims;
            }
        }
        if (remaining > limit) return false;

        // Optional retained K/V goes first. Movement, destruction and deque
        // shrinkage need no new allocation after the plan has been accepted.
        reuse_ = {};
        size_t write = 0, left = victims;
        for (size_t read = 0; read < entries_.size(); ++read) {
            if (left && read != keep && evictable(read)) { --left; continue; }
            if (protected_index && read == keep) *protected_index = write;
            if (write != read) {
                entries_[write] = std::move(entries_[read]);
                serials_[write] = serials_[read];
            }
            ++write;
        }
        while (entries_.size() > write) { entries_.pop_back(); serials_.pop_back(); }
        bytes_ = remaining;
        evictions_ += victims;
        return true;
    }

    // Release the oldest unpinned, nonpending image for byte/physical-memory pressure.
    // Buffer extents are released before the caller resamples available memory.
    // False when empty or all remaining entries are pinned/pending.
    bool evict_oldest() {
        for (size_t i = 0; i < entries_.size(); ++i) {
            if (!evictable(i)) continue;
            bytes_ -= entries_[i].bytes();
            erase(i);
            ++evictions_;
            return true;
        }
        return false;
    }

    // #879: every parked conversation (pinned/pending too) and retained K/V go -
    // after non-finite logits any may carry poison. Acquired leases stay charged.
    // Returns how many were parked.
    size_t clear() {
        const size_t n = entries_.size();
        entries_.clear();
        serials_.clear();
        bytes_ = 0;
        reuse_ = {};
        guarded_ = 0;
        guard_blocked_ = false;
        return n;
    }

    // The slot count, so a caller that evicts in a loop has a bound it did not invent.
    size_t slots() const { return slots_; }

    // Latest-history retention commits in put(), never before a snapshot can fail.
    // Divergent branches, explicit pins, pending publications and borrowed donors remain protected.
    size_t superseded() const { return superseded_; }
    /// --conversation-cache-park-gain N: the longest parked entry this conversation's outgoing state supersedes, in
    /// tokens, 0 when it has none. The same scoped, pin-preserving live-extension rule as put()
    /// decides whether a park would replace an older copy. Planning never deletes that fallback.
    size_t parked_prefix_tokens(const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& images,
                                const std::vector<ConversationCheckpoint>& checkpoints, bool cvec,
                                const ConversationScope& scope) const {
        size_t longest = 0;
        for (size_t i = 0; i < entries_.size(); ++i) {
            const auto& e = entries_[i];
            if (e.publication_ticket != 0 || !e.scope.matches(scope) || e.cvec != cvec ||
                (guarded_ != 0 && serials_[i] == guarded_) || e.live.ids.empty() ||
                !((e.live.ids == ids && e.live.imgs == images) || conversation_prefix(e.live, ids, images) > 0))
                continue;
            bool pins_kept = true;
            for (const auto& pin : e.checkpoints)
                if (pin.pinned && std::none_of(checkpoints.begin(), checkpoints.end(),
                    [&](const auto& next) { return same_pin(pin, next); })) { pins_kept = false; break; }
            if (pins_kept) longest = std::max(longest, e.live.ids.size());
        }
        return longest;
    }

    // A non-consuming restore of a removed owner reserves its eventual return
    // entry. Pending owners already remain parked and need no such reservation.
    // Ticketed candidates freeze before sizing/admission; refusal may leave the
    // candidate frozen, but never consumes it or changes the cache.
    bool put(SavedConversation&& image, size_t held = 0, size_t held_entries = 0) {
        static_assert(std::is_nothrow_move_constructible_v<SavedConversation>);
        static_assert(std::is_nothrow_move_assignable_v<SavedConversation>);
        static_assert(std::is_nothrow_destructible_v<SavedConversation>);
        guard_blocked_ = false;
        if (!enabled() || !image.scope.valid() || held_entries >= slots_ || held > budget_ ||
            has_publication(image.publication_ticket)) return false;
        if (image.publication_ticket != 0) image.freeze();
        const size_t n = image.bytes();
        if (n > room(held)) return false;
        const size_t available_entries = slots_ - held_entries;
        const size_t limit = room(held) - n;
        size_t count = entries_.size(), remaining = bytes_, replaced = 0, victims = 0;
        for (size_t i = 0; i < entries_.size(); ++i) if (replaces(i, image)) {
            --count;
            remaining -= entries_[i].bytes();
            ++replaced;
        }
        const size_t reuse_bytes = reuse_.bytes();
        const bool discard_reuse = remaining > limit || reuse_bytes > limit - remaining;
        const size_t payload_limit = discard_reuse ? limit : limit - reuse_bytes;
        for (size_t i = 0; i < entries_.size(); ++i) {
            if (count < available_entries && remaining <= payload_limit) break;
            if (evictable(i) && !replaces(i, image)) {
                --count;
                remaining -= entries_[i].bytes();
                ++victims;
            }
        }
        if (count >= available_entries || remaining > payload_limit) {
            guard_blocked_ = guarded_ != 0 && index_of(guarded_) != SIZE_MAX;
            return false;
        }

        // Allocate/construct publication storage before destroying any old image.
        // noexcept movement keeps the candidate unchanged if deque growth fails.
        const size_t old_count = entries_.size();
        serials_.push_back(next_serial_);
        try {
            entries_.push_back(std::move(image));
        } catch (...) {
            serials_.pop_back();
            throw;
        }
        ++next_serial_;
        const auto& next = entries_.back();
        size_t write = 0, ordinary_left = victims;
        // Compact old entries without erase/pop: next stays at its stable final
        // index until every comparison is done, and each survivor moves at most once.
        for (size_t read = 0; read < old_count; ++read) {
            if (replaces(read, next)) continue;
            if (ordinary_left && evictable(read)) { --ordinary_left; continue; }
            if (write != read) {
                entries_[write] = std::move(entries_[read]);
                serials_[write] = serials_[read];
            }
            ++write;
        }
        assert(ordinary_left == 0);
        if (write != old_count) {
            entries_[write] = std::move(entries_.back());
            serials_[write] = serials_.back();
        }
        while (entries_.size() > write + 1) { entries_.pop_back(); serials_.pop_back(); }
        if (discard_reuse) reuse_ = {};
        bytes_ = remaining + n;
        superseded_ += replaced;
        evictions_ += victims;
        return true;
    }

private:
    static bool same_pin(const ConversationCheckpoint& a, const ConversationCheckpoint& b) noexcept {
        if (a.pinned != b.pinned || a.ids != b.ids || a.imgs != b.imgs || a.plan != b.plan ||
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
    bool replaces(size_t index, const SavedConversation& next) const noexcept {
        const auto& old = entries_[index];
        return (guarded_ == 0 || serials_[index] != guarded_) &&
            old.publication_ticket == 0 && old.scope.matches(next.scope) && compatible_pins(old, next);
    }
    static bool evictable(const SavedConversation& image) noexcept {
        return image.publication_ticket == 0 && !image.pinned();
    }
    bool evictable(size_t index) const noexcept {
        return evictable(entries_[index]) && (guarded_ == 0 || serials_[index] != guarded_);
    }

    // Bytes the envelope can still admit for a new image: budget minus this
    // image's own held charge and all fixed/publication leases, and nothing
    // else (0 when the charges alone already exceed the budget).
    size_t room(size_t held) const {
        if (held_bytes_ > budget_ || held > budget_ - held_bytes_) return 0;
        return budget_ - held_bytes_ - held;
    }

    void erase(size_t index) {
        entries_.erase(entries_.begin() + (std::ptrdiff_t) index);
        serials_.erase(serials_.begin() + (std::ptrdiff_t) index);
    }
    size_t budget_ = 0, slots_ = 0, bytes_ = 0, evictions_ = 0, superseded_ = 0, borrowed_ = 0;
    size_t held_bytes_ = 0;    // fixed resident ownership plus outstanding immutable publication leases
    int64_t min_tokens_ = 0;   // --conversation-cache-min-tokens: 0 parks every conversation, whatever its size
    std::deque<SavedConversation> entries_; // least recently active first
    std::deque<uint64_t> serials_;          // entries_[i]'s identity, never reused
    uint64_t next_serial_ = 1, guarded_ = 0;
    bool guard_blocked_ = false;
    ConversationKvReuse reuse_;
};

} // namespace strata::core
