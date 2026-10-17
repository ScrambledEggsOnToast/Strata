// The durable store's pure byte/tree layer: the canonical stream enumeration of one SavedConversation tree, the
// manifest and catalog encodings, the exact-prefix chain digests a lookup compares, and the bounded restore that
// turns a manifest back into a complete image.  No file I/O, no threads, no GPU, no global state: every entry point
// either produces the exact metadata the caller then writes, or refuses with a reason before allocating.
//
// Encoding rules (STCSMAN1 for legacy captures, STCSMAN2 for explicit draft certification, STCSCAT1 catalog;
// little-endian throughout; raw int32/image-key vectors use the asserted little-endian host encoding):
//   integers        8 bytes for u64/i64, 4 for u32, 2 for u16, 1 for u8/bool
//   byte strings    u64 length then the bytes
//   digests         32 raw bytes
//   every file      the tag, the body, then SHA-256 of everything before it
// A reader never trusts a count: each is bounded by ConversationStoreLimits AND by the bytes actually left in the
// buffer before anything is allocated from it.
#include "conversation_store_format.hpp"

#include "conversation_checked.hpp"

#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <tuple>

namespace strata::core::conversation_store_detail {

using conversation_detail::add;

static_assert(std::endian::native == std::endian::little,
              "the store format is little-endian; a big-endian host would need an explicit byte swap");
static_assert(sizeof(ConversationImageKey) == 16, "the image-key stream encodes the struct as raw little-endian bytes");
static_assert(sizeof(int32_t) == 4, "the token-id stream encodes int32 little-endian");

namespace {

// ---------------------------------------------------------------------------------------------------------------
// Encoding primitives
// ---------------------------------------------------------------------------------------------------------------

constexpr char kManifestTag[8] = {'S', 'T', 'C', 'S', 'M', 'A', 'N', '1'};
constexpr char kCertifiedManifestTag[8] = {'S', 'T', 'C', 'S', 'M', 'A', 'N', '2'};
constexpr char kCatalogTag[8] = {'S', 'T', 'C', 'S', 'C', 'A', 'T', '1'};
constexpr char kBucketTag[10] = {'S', 'T', 'C', 'S', 'B', 'U', 'C', 'K', 'E', 'T'};
constexpr char kObjectTag[8] = {'S', 'T', 'C', 'S', 'O', 'B', 'J', '1'};
constexpr char kObjectScopeTag[10] = {'S', 'T', 'C', 'S', 'S', 'C', 'O', 'P', 'E', '1'};
constexpr char kPrefixTag[12] = {'S', 'T', 'C', 'S', 'P', 'R', 'E', 'F', 'I', 'X', '0', '1'};
constexpr size_t kTrailer = 32;
// Hard ceilings for a structure that has no configured limit (the catalog): a corrupt file can still not make the
// parser allocate an unbounded table.
constexpr uint32_t kCatalogCeiling = 1u << 22;
// The format's own structural ceilings: what a record can represent at all, whatever figures the run that wrote it had
// admitted.  A stage part belongs to an image stage and never owns further parts, so one level of parts below the
// image is everything the format encodes and the deepest stage index it can name is this bound.
constexpr uint32_t kFormatMaxStages = 64;
// The smallest encoded form of every section a manifest carries, one figure per section, so the decoder's count
// bounds, the retention figures and the decode's workspace all derive from the same numbers: one container (its two
// flags, scope, record metadata, geometry and the two section counts), one checkpoint record (token count, retained
// size, pin flag, part count), one K/V layer's metadata (seven geometry fields and five part lengths), one stream
// (identity, length and its chunk count), one chunk (offset, length, content address), one object (its content
// address and size) and one cut (length, digest, two flags).  A manifest that fits `max_manifest_bytes` cannot hold
// more of a section than that ceiling divided by the section's own figure, and Reader::count bounds every count by
// the bytes left in the file as well.  The container figure is exactly what manifest_encode() writes for the
// shortest container, so bounding the container count by it - rather than by a looser allowance - is what lets the
// whole workspace of a decode be counted from the encoded size alone: no section can then be charged for more
// decoded bytes than the file spends encoding it (see manifest_workspace_bound).
constexpr uint64_t kContainerEncodedMin = 244;
constexpr uint64_t kCheckpointEncodedMin = 21;
constexpr uint64_t kKvLayerEncodedMin = 96;
constexpr uint64_t kStreamEncodedMin = 34;
constexpr uint64_t kChunkEncodedMin = 44;
constexpr uint64_t kObjectEncodedMin = 40;
constexpr uint64_t kCutEncodedMin = 42;
// The widest single chunk the format can declare: a chunk's length is a uint32 field.
constexpr uint64_t kMaxChunkBytes = std::numeric_limits<uint32_t>::max();

class Writer {
public:
    void u8(uint8_t v) { bytes_.push_back(v); }
    void u16(uint16_t v) { put(v, 2); }
    void u32(uint32_t v) { put(v, 4); }
    void u64(uint64_t v) { put(v, 8); }
    void i64(int64_t v) { put(uint64_t(v), 8); }
    void raw(const void* data, size_t n) {
        const uint8_t* p = static_cast<const uint8_t*>(data);
        bytes_.insert(bytes_.end(), p, p + n);
    }
    void digest(const Digest& d) { raw(d.data(), d.size()); }
    const std::vector<uint8_t>& data() const { return bytes_; }
    size_t size() const { return bytes_.size(); }
    // Reserves the exact size the caller already counted, so an encode never reallocates as it grows, and hands the
    // finished bytes out by move: `data()` would copy the whole encoding into the return value.
    void reserve(size_t capacity) { bytes_.reserve(capacity); }
    std::vector<uint8_t> take() { return std::move(bytes_); }

private:
    void put(uint64_t v, int n) {
        for (int i = 0; i < n; ++i) bytes_.push_back(uint8_t((v >> (8 * i)) & 0xff));
    }
    std::vector<uint8_t> bytes_;
};

class Reader {
public:
    Reader(const uint8_t* data, size_t size) : data_(data), size_(size) {}
    size_t remaining() const { return size_ - at_; }
    bool empty() const { return at_ == size_; }
    bool need(size_t n) const { return n <= size_ - at_; }
    bool u8(uint8_t& out) {
        if (!need(1)) return false;
        out = data_[at_++];
        return true;
    }
    bool u16(uint16_t& out) { return get(out, 2); }
    bool u32(uint32_t& out) { return get(out, 4); }
    bool u64(uint64_t& out) { return get(out, 8); }
    bool i64(int64_t& out) {
        uint64_t v = 0;
        if (!get(v, 8)) return false;
        out = int64_t(v);
        return true;
    }
    bool raw(void* dst, size_t n) {
        if (!need(n)) return false;
        std::memcpy(dst, data_ + at_, n);
        at_ += n;
        return true;
    }
    bool digest(Digest& out) { return raw(out.data(), out.size()); }
    // A count bounded by the configured limit AND by the bytes left for its smallest possible entry, so a crafted
    // file cannot drive an allocation before the data that would justify it exists.
    bool count(uint32_t& out, uint32_t limit, size_t min_entry) {
        uint32_t value = 0;
        if (!u32(value)) return false;
        if (value > limit) return false;
        if (min_entry && uint64_t(value) * min_entry > remaining()) return false;
        out = value;
        return true;
    }
    // u32 length followed by that many bytes: the pattern every byte string uses.
    bool span(uint32_t& length, uint32_t limit) { return count(length, limit, 1); }

private:
    template<class T> bool get(T& out, int n) {
        uint64_t v = 0;
        if (!need(size_t(n))) return false;
        for (int i = 0; i < n; ++i) v |= uint64_t(data_[at_ + size_t(i)]) << (8 * i);
        at_ += size_t(n);
        out = T(v);
        return true;
    }
    const uint8_t* data_;
    size_t size_;
    size_t at_ = 0;
};

// ---------------------------------------------------------------------------------------------------------------
// Canonical stream order
//
// For every container, in container order: the container's own record (its token ids, its image keys, then its five
// running-state arrays), its optional boundary residual, then its checkpoints in retention order, then its K/V layers
// in layer order. A container is either a whole image (kinds live_*/cp_*) or one record's stage part (kinds part_*).
// `owner` names the record: 0 for a container's live record, 1 + checkpoint index for a checkpoint, and the owner's key
// for every stream of a part container; `ordinal` is the K/V layer index for kv_* and 0 otherwise.
// ---------------------------------------------------------------------------------------------------------------

enum class Buffer {
    ids,
    imgs,
    gdn,
    ple,
    tails,
    dead,
    block_pos,
    draft_boundary,
    kv_k,
    kv_v,
    kv_k_scale,
    kv_v_scale,
    kv_pooled,
};

struct KindSet {
    uint16_t ids;
    uint16_t imgs;
    uint16_t state[5];
};

constexpr KindSet kImageLive{uint16_t(ConversationStoreStreamKind::ids),
                             uint16_t(ConversationStoreStreamKind::imgs),
                             {uint16_t(ConversationStoreStreamKind::live_gdn),
                              uint16_t(ConversationStoreStreamKind::live_ple),
                              uint16_t(ConversationStoreStreamKind::live_tails),
                              uint16_t(ConversationStoreStreamKind::live_dead),
                              uint16_t(ConversationStoreStreamKind::live_block_pos)}};
constexpr KindSet kImageCheckpoint{uint16_t(ConversationStoreStreamKind::cp_ids),
                                   uint16_t(ConversationStoreStreamKind::cp_imgs),
                                   {uint16_t(ConversationStoreStreamKind::cp_gdn),
                                    uint16_t(ConversationStoreStreamKind::cp_ple),
                                    uint16_t(ConversationStoreStreamKind::cp_tails),
                                    uint16_t(ConversationStoreStreamKind::cp_dead),
                                    uint16_t(ConversationStoreStreamKind::cp_block_pos)}};
constexpr KindSet kPart{uint16_t(ConversationStoreStreamKind::part_ids),
                        uint16_t(ConversationStoreStreamKind::part_imgs),
                        {uint16_t(ConversationStoreStreamKind::part_gdn),
                         uint16_t(ConversationStoreStreamKind::part_ple),
                         uint16_t(ConversationStoreStreamKind::part_tails),
                         uint16_t(ConversationStoreStreamKind::part_dead),
                         uint16_t(ConversationStoreStreamKind::part_block_pos)}};

constexpr Buffer kStateBuffer[5] = {Buffer::gdn, Buffer::ple, Buffer::tails, Buffer::dead, Buffer::block_pos};
constexpr Buffer kKvBuffer[5] = {Buffer::kv_k, Buffer::kv_v, Buffer::kv_k_scale, Buffer::kv_v_scale,
                                 Buffer::kv_pooled};
constexpr uint16_t kKvKind[5] = {uint16_t(ConversationStoreStreamKind::kv_k),
                                 uint16_t(ConversationStoreStreamKind::kv_v),
                                 uint16_t(ConversationStoreStreamKind::kv_k_scale),
                                 uint16_t(ConversationStoreStreamKind::kv_v_scale),
                                 uint16_t(ConversationStoreStreamKind::kv_pooled)};

struct Slot {
    ConversationStoreStreamId id;
    Buffer buffer = Buffer::ids;
    uint32_t container = 0;
    uint32_t kv = 0;
    uint64_t record_key = 0;
};

// Walks the canonical stream order implied by container metadata.  `visit(slot, container, why)` returns false to
// abort with a reason.  This is the single source of truth for that order: the layout, the manifest checks and the
// restore all walk through here, so they can never disagree about a stream's identity or position.
template<class Visitor>
bool walk_streams(const std::vector<ContainerMeta>& containers, Visitor&& visit, std::string& why) {
    for (uint32_t ci = 0; ci < containers.size(); ++ci) {
        const ContainerMeta& c = containers[ci];
        const bool image = c.is_image;
        const KindSet& own = image ? kImageLive : kPart;
        const uint64_t own_owner = image ? 0 : c.parent_owner;
        auto slot = [&](uint16_t kind, uint64_t owner, uint64_t ordinal, Buffer buffer, uint32_t kv,
                        uint64_t key) {
            Slot s;
            s.id.stage = c.stage;
            s.id.kind = kind;
            s.id.owner = owner;
            s.id.ordinal = ordinal;
            s.buffer = buffer;
            s.container = ci;
            s.kv = kv;
            s.record_key = key;
            return s;
        };
        if (!visit(slot(own.ids, own_owner, 0, Buffer::ids, 0, own_owner), c, why)) return false;
        if (!visit(slot(own.imgs, own_owner, 0, Buffer::imgs, 0, own_owner), c, why)) return false;
        for (int i = 0; i < 5; ++i)
            if (!visit(slot(own.state[i], own_owner, 0, kStateBuffer[i], 0, own_owner), c, why)) return false;
        if (!image) {
            if (!c.checkpoints.empty() || !c.kv.empty()) {
                why = "a stage part container must not carry checkpoints or K/V layers";
                return false;
            }
            continue;
        }
        if (c.draft_boundary_bytes &&
            !visit(slot(kDraftBoundaryKind, 0, 0, Buffer::draft_boundary, 0, 0), c, why)) return false;
        for (uint32_t j = 0; j < c.checkpoints.size(); ++j) {
            const uint64_t key = uint64_t(j) + 1;
            if (!visit(slot(kImageCheckpoint.ids, key, 0, Buffer::ids, 0, key), c, why)) return false;
            if (!visit(slot(kImageCheckpoint.imgs, key, 0, Buffer::imgs, 0, key), c, why)) return false;
            for (int i = 0; i < 5; ++i)
                if (!visit(slot(kImageCheckpoint.state[i], key, 0, kStateBuffer[i], 0, key), c, why)) return false;
        }
        for (uint32_t k = 0; k < c.kv.size(); ++k)
            for (int i = 0; i < 5; ++i)
                if (!visit(slot(kKvKind[i], 0, k, kKvBuffer[i], k, 0), c, why)) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------------------------------------------
// Container metadata from a live image tree
// ---------------------------------------------------------------------------------------------------------------

struct TreeContainer {
    ContainerMeta meta;
    const ConversationCheckpoint* own = nullptr;             // the live record, or the part's own record
    std::vector<const ConversationCheckpoint*> checkpoints;  // image containers only
    std::vector<const ConversationKv*> kv;                   // image containers only
    const ConversationBuffer* boundary = nullptr;
};

RecordMeta record_meta(const ConversationCheckpoint& record) {
    RecordMeta meta;
    meta.tokens = record.ids.size();
    meta.used = record.used;
    meta.pinned = record.pinned;
    meta.parts = uint32_t(record.stage_parts.size());
    return meta;
}

KvMeta kv_meta(const ConversationKv& layer) {
    KvMeta meta;
    meta.format = layer.format;
    meta.cells = layer.cells;
    meta.heads = layer.heads;
    meta.head_dim = layer.head_dim;
    meta.page_size = layer.page_size;
    meta.pooled_rows = layer.pooled_rows;
    meta.idx_dim = layer.idx_dim;
    meta.part[0] = layer.k.size();
    meta.part[1] = layer.v.size();
    meta.part[2] = layer.k_scale.size();
    meta.part[3] = layer.v_scale.size();
    meta.part[4] = layer.pooled.size();
    return meta;
}

// Scalar-only: admission and decode share this before allocating any payload.
template<class Kv>
bool check_draft(const std::array<int64_t, 18>& g, int64_t lo, int64_t hi, uint64_t tokens,
                 int64_t begin, int64_t cells, int32_t next, uint64_t boundary,
                 const Kv* kv, size_t count, std::string& why) {
    if (begin == -1 && cells == -1 && next == -1 && boundary == 0) return true;
    if (begin < 0 || cells < begin || uint64_t(cells) > tokens ||
        (uint64_t(cells) == tokens ? next < 0 : next != -1)) {
        why = "incoherent draft certification interval or tail token";
        return false;
    }
    if (g[1] <= 0 || g[2] <= 0 || lo < 0 || hi < lo || hi != g[1]) {
        why = "draft certification is not on a draft-owning stage";
        return false;
    }
    const int64_t main = std::max<int64_t>(hi / g[2] - lo / g[2], g[1] / g[2] ? 1 : 0);
    if (uint64_t(main) >= count || count - size_t(main) != 1 || kv[count - 1].cells < cells) {
        why = "draft certification exceeds the serialized draft extent";
        return false;
    }
    if (boundary && (tokens == 0 || uint64_t(cells) != tokens - 1 || next != -1 ||
        g[0] <= 0 || g[14] <= 0 || uint64_t(g[0]) > UINT64_MAX / uint64_t(g[14]) ||
        uint64_t(g[0]) * uint64_t(g[14]) > UINT64_MAX / sizeof(float) ||
        boundary != uint64_t(g[0]) * uint64_t(g[14]) * sizeof(float))) {
        why = "draft boundary residual has invalid position or row length";
        return false;
    }
    return true;
}

bool check_image_draft(const SavedConversation& image, uint32_t stage, uint32_t stages,
                       const ConversationStoreLimits& limits, std::string& why) {
    if (image.draft_begin >= 0 && stage + 1 != stages) {
        why = "draft certification belongs only to the final stage";
        return false;
    }
    const uint64_t boundary = image.draft_boundary_residual.size();
    if (boundary > limits.max_state_bytes || boundary > limits.max_stream_bytes) {
        why = "draft boundary residual exceeds the state/stream budget";
        return false;
    }
    return check_draft(image.geometry, image.layer_lo, image.layer_hi, image.live.ids.size(),
                       image.draft_begin, image.draft_cells, image.draft_next_token, boundary,
                       image.kv.data(), image.kv.size(), why);
}

bool image_container(const SavedConversation& image, uint32_t stage, const ConversationStoreLimits& limits,
                     TreeContainer& out, std::string& why) {
    if (!image.scope.valid()) {
        why = "an image has no valid authorized scope";
        return false;
    }
    ContainerMeta& meta = out.meta;
    meta = ContainerMeta{};
    meta.is_image = true;
    meta.stage = stage;
    meta.parent_stage = stage;
    meta.carve_known = true;
    meta.scope = image.scope;
    meta.geometry = image.geometry;
    meta.layer_lo = image.layer_lo;
    meta.layer_hi = image.layer_hi;
    meta.cvec = image.cvec;
    meta.draft_begin = image.draft_begin;
    meta.draft_cells = image.draft_cells;
    meta.draft_next_token = image.draft_next_token;
    meta.draft_boundary_bytes = image.draft_boundary_residual.size();
    out.boundary = &image.draft_boundary_residual;
    meta.record = record_meta(image.live);
    if (meta.record.tokens > limits.max_tokens) {
        why = "an image record holds more tokens than ConversationStoreLimits::max_tokens";
        return false;
    }
    if (image.checkpoints.size() > limits.max_checkpoints) {
        why = "an image holds more checkpoints than ConversationStoreLimits::max_checkpoints";
        return false;
    }
    if (image.kv.size() > limits.max_kv_layers) {
        why = "an image holds more K/V layers than ConversationStoreLimits::max_kv_layers";
        return false;
    }
    meta.checkpoints.reserve(image.checkpoints.size());
    out.checkpoints.reserve(image.checkpoints.size());
    for (const auto& checkpoint : image.checkpoints) {
        const RecordMeta record = record_meta(checkpoint);
        if (record.tokens > limits.max_tokens) {
            why = "a checkpoint holds more tokens than ConversationStoreLimits::max_tokens";
            return false;
        }
        meta.checkpoints.push_back(record);
        out.checkpoints.push_back(&checkpoint);
    }
    meta.kv.reserve(image.kv.size());
    out.kv.reserve(image.kv.size());
    for (const auto& layer : image.kv) {
        meta.kv.push_back(kv_meta(layer));
        out.kv.push_back(&layer);
    }
    out.own = &image.live;
    return true;
}

bool part_container(const ConversationCheckpoint& part, const ContainerMeta& owner, uint64_t owner_key,
                    uint32_t part_index, const ContainerMeta* carve, const ConversationScope& scope,
                    const ConversationStoreLimits& limits, TreeContainer& out, std::string& why) {
    if (part.stage_parts.size() != 0) {
        why = "a stage part that owns further stage parts is not representable";
        return false;
    }
    ContainerMeta& meta = out.meta;
    meta = ContainerMeta{};
    meta.is_image = false;
    meta.stage = owner.stage + 1 + part_index;
    meta.parent_stage = owner.stage;
    meta.parent_owner = owner_key;
    meta.part_index = part_index;
    meta.carve_known = carve != nullptr;
    if (carve) {
        meta.geometry = carve->geometry;
        meta.layer_lo = carve->layer_lo;
        meta.layer_hi = carve->layer_hi;
        meta.cvec = carve->cvec;
    }
    meta.scope = scope;
    meta.record = record_meta(part);
    if (meta.record.tokens > limits.max_tokens) {
        why = "a stage part holds more tokens than ConversationStoreLimits::max_tokens";
        return false;
    }
    out.own = &part;
    return true;
}

bool build_tree(const SavedConversation& image, const ConversationStoreLimits& limits, std::vector<TreeContainer>& out,
                std::string& why) {
    for (const auto& stage : image.stage_images) {
        if (!stage.stage_images.empty()) {
            why = "a stage image that owns its own stage images is not representable";
            return false;
        }
    }
    if (image.stage_images.size() + 1 > limits.max_stages) {
        why = "the image declares more stages than ConversationStoreLimits::max_stages";
        return false;
    }
    const uint32_t stages = uint32_t(image.stage_images.size()) + 1;
    // Check every certification before reserving even the metadata skeleton, not only before transferring payload.
    if (!check_image_draft(image, 0, stages, limits, why)) return false;
    for (uint32_t i = 0; i + 1 < stages; ++i)
        if (!check_image_draft(image.stage_images[i], i + 1, stages, limits, why)) return false;
    out.clear();
    out.reserve(stages);
    if (!image_container(image, 0, limits, out.emplace_back(), why)) return false;
    for (uint32_t i = 0; i + 1 < stages; ++i)
        if (!image_container(image.stage_images[i], i + 1, limits, out.emplace_back(), why)) return false;

    // The image containers are all built now, so a part can take the carve of the stage image that declares its
    // stage.  A part with no such stage keeps carve_known = false.
    const size_t image_containers = out.size();
    for (size_t ci = 0; ci < image_containers; ++ci) {
        const TreeContainer owners = out[ci];
        for (size_t r = 0; r <= owners.checkpoints.size(); ++r) {
            const ConversationCheckpoint& record = r == 0 ? *owners.own : *owners.checkpoints[r - 1];
            const uint64_t owner_key = r == 0 ? 0 : uint64_t(r);
            for (uint32_t p = 0; p < record.stage_parts.size(); ++p) {
                const uint32_t stage = owners.meta.stage + 1 + p;
                const ContainerMeta* carve = stage < stages ? &out[stage].meta : nullptr;
                if (out.size() >= limits.max_images) {
                    why = "the image has more containers than ConversationStoreLimits::max_images";
                    return false;
                }
                TreeContainer part;
                if (!part_container(record.stage_parts[p], owners.meta, owner_key, p, carve, owners.meta.scope,
                                    limits, part, why))
                    return false;
                out.push_back(std::move(part));
            }
        }
    }
    return true;
}

// ---------------------------------------------------------------------------------------------------------------
// Cuts
// ---------------------------------------------------------------------------------------------------------------

// A cut of the top image is advertised only when every later stage can also be placed at exactly that length: that
// stage's own image must hold the cut (its live record or one of its checkpoints).  A record's stage parts are
// restored verbatim but never manufacture a cut, so an ambiguous part-to-stage mapping cannot advertise a prefix the
// store could not actually compose.
bool cut_available(const SavedConversation& image, uint64_t tokens) {
    for (const auto& stage : image.stage_images) {
        bool found = stage.live.ids.size() == tokens;
        for (const auto& checkpoint : stage.checkpoints)
            if (checkpoint.ids.size() == tokens) found = true;
        if (!found) return false;
    }
    return true;
}

void build_cuts(const SavedConversation& image, const Digest& seed, const ConversationStoreLimits& limits,
                std::vector<CutRecord>& out) {
    out.clear();
    const size_t cap = size_t(limits.max_checkpoints) + 1;
    auto add_cut = [&](const ConversationCheckpoint& record, bool live) {
        const uint64_t tokens = record.ids.size();
        if (tokens == 0 || tokens > limits.max_tokens) return;
        if (!cut_available(image, tokens)) return;
        if (out.size() >= cap) return;
        const PrefixChain chain(seed, record.ids.data(), record.ids.size(), record.imgs);
        Digest digest{};
        if (!chain.digest(tokens, digest)) return;
        for (auto& existing : out) {
            if (existing.tokens != tokens || existing.digest != digest) continue;
            existing.live = existing.live || live;
            existing.pinned = existing.pinned || record.pinned;
            return;
        }
        CutRecord cut;
        cut.tokens = tokens;
        cut.digest = digest;
        cut.live = live;
        cut.pinned = record.pinned;
        out.push_back(cut);
    };
    add_cut(image.live, true);
    for (const auto& checkpoint : image.checkpoints) add_cut(checkpoint, false);
    std::stable_sort(out.begin(), out.end(), [](const CutRecord& a, const CutRecord& b) {
        if (a.tokens != b.tokens) return a.tokens > b.tokens;
        return a.pinned && !b.pinned;
    });
}

// ---------------------------------------------------------------------------------------------------------------
// Manifest consistency: verified once, so both a decoded file and a freshly built layout are held to it
// ---------------------------------------------------------------------------------------------------------------

bool check_containers(const ManifestMeta& meta, const ConversationStoreLimits& limits, std::string& why) {
    if (meta.containers.empty()) {
        why = "the manifest has no containers";
        return false;
    }
    if (meta.containers.size() > limits.max_images) {
        why = "the manifest has more containers than ConversationStoreLimits::max_images";
        return false;
    }
    if (meta.stages == 0 || meta.stages > limits.max_stages) {
        why = "the manifest declares an unsupported stage count";
        return false;
    }
    if (meta.stages > meta.containers.size()) {
        why = "the manifest declares more stages than it has containers";
        return false;
    }
    for (uint32_t i = 0; i < meta.containers.size(); ++i) {
        const ContainerMeta& c = meta.containers[i];
        if (!c.is_image && (c.draft_begin != -1 || c.draft_cells != -1 ||
                            c.draft_next_token != -1 || c.draft_boundary_bytes)) {
            why = "a checkpoint stage part carries draft certification";
            return false;
        }
        if (!check_draft(c.geometry, c.layer_lo, c.layer_hi, c.record.tokens, c.draft_begin, c.draft_cells,
                         c.draft_next_token, c.draft_boundary_bytes, c.kv.data(), c.kv.size(), why))
            return false;
        if (c.draft_begin >= 0 && c.stage + 1 != meta.stages) {
            why = "draft certification belongs only to the final stage";
            return false;
        }
        if (!c.scope.valid()) {
            why = "a container has no valid authorized scope";
            return false;
        }
        if (c.record.tokens > limits.max_tokens) {
            why = "a record holds more tokens than ConversationStoreLimits::max_tokens";
            return false;
        }
        if (c.is_image) {
            if (i >= meta.stages || c.stage != i) {
                why = "the image containers are not the leading containers in stage order";
                return false;
            }
            if (c.checkpoints.size() > limits.max_checkpoints) {
                why = "a container holds more checkpoints than ConversationStoreLimits::max_checkpoints";
                return false;
            }
            if (c.kv.size() > limits.max_kv_layers) {
                why = "a container holds more K/V layers than ConversationStoreLimits::max_kv_layers";
                return false;
            }
            for (const auto& checkpoint : c.checkpoints)
                if (checkpoint.tokens > limits.max_tokens) {
                    why = "a checkpoint holds more tokens than ConversationStoreLimits::max_tokens";
                    return false;
                }
            for (const auto& layer : c.kv)
                for (int p = 0; p < 5; ++p)
                    if (layer.part[p] > limits.max_state_bytes) {
                        why = "a K/V part is larger than ConversationStoreLimits::max_state_bytes";
                        return false;
                    }
            continue;
        }
        if (c.parent_stage >= meta.stages || !meta.containers[c.parent_stage].is_image) {
            why = "a stage part names a parent stage that is not an image container";
            return false;
        }
        if (c.part_index >= limits.max_stages) {
            why = "a stage part's index is out of range";
            return false;
        }
        if (c.stage != c.parent_stage + 1 + c.part_index) {
            why = "a stage part does not name the stage its position implies";
            return false;
        }
        if (c.parent_owner > meta.containers[c.parent_stage].checkpoints.size()) {
            why = "a stage part names a checkpoint its parent does not hold";
            return false;
        }
        if (c.record.parts != 0) {
            why = "a stage part that owns further stage parts is not representable";
            return false;
        }
        if (!c.checkpoints.empty() || !c.kv.empty()) {
            why = "a stage part container must not carry checkpoints or K/V layers";
            return false;
        }
        const bool carve_known = c.stage < meta.stages;
        if (carve_known != c.carve_known) {
            why = "a stage part's carve flag disagrees with the stages the image declares";
            return false;
        }
        if (carve_known) {
            const ContainerMeta& declared = meta.containers[c.stage];
            if (declared.layer_lo != c.layer_lo || declared.layer_hi != c.layer_hi ||
                declared.geometry != c.geometry || declared.cvec != c.cvec) {
                why = "a stage part's carve disagrees with the stage image that declares it";
                return false;
            }
        } else if (c.layer_lo != 0 || c.layer_hi != 0) {
            why = "a stage part without a declared carve carries a layer range";
            return false;
        }
    }
    uint32_t images = 0;
    for (const auto& c : meta.containers) images += c.is_image ? 1 : 0;
    if (images != meta.stages) {
        why = "the manifest's stage count disagrees with its image containers";
        return false;
    }
    // Every record's declared part count must be exactly the parts the manifest carries for it, in part order and
    // grouped by owner: a restore attaches them by position, so a gap or a reordering would misplace state.
    size_t next_part = meta.stages;
    for (uint32_t s = 0; s < meta.stages; ++s) {
        const ContainerMeta& c = meta.containers[s];
        for (size_t r = 0; r <= c.checkpoints.size(); ++r) {
            const uint32_t parts = r == 0 ? c.record.parts : c.checkpoints[r - 1].parts;
            if (parts > limits.max_stages || parts > meta.containers.size() - next_part) {
                why = "a record declares more stage parts than the manifest contains";
                return false;
            }
            for (uint32_t p = 0; p < parts; ++p) {
                const ContainerMeta& part = meta.containers[next_part++];
                if (part.parent_stage != s || part.parent_owner != r || part.part_index != p) {
                    why = "the manifest's stage parts are not grouped by owner in part order";
                    return false;
                }
            }
        }
    }
    if (next_part != meta.containers.size()) {
        why = "the manifest's stage part count disagrees with the records that own them";
        return false;
    }
    return true;
}

bool check_streams(const ManifestMeta& meta, const ConversationStoreLimits& limits, std::string& why) {
    if (meta.streams.size() > limits.max_streams) {
        why = "the manifest holds more streams than ConversationStoreLimits::max_streams";
        return false;
    }
    std::map<Digest, uint64_t> objects;
    for (const auto& object : meta.objects) {
        if (object.second == 0) {
            why = "the manifest declares a zero-length object";
            return false;
        }
        if (!objects.emplace(object.first, object.second).second) {
            why = "the manifest declares the same object twice";
            return false;
        }
    }
    std::set<std::tuple<uint32_t, uint16_t, uint64_t, uint64_t>> seen;
    uint64_t payload = 0;
    for (uint32_t i = 0; i < meta.streams.size(); ++i) {
        const StreamRecord& stream = meta.streams[i];
        if (stream.bytes > limits.max_stream_bytes) {
            why = "a stream is longer than ConversationStoreLimits::max_stream_bytes";
            return false;
        }
        if (!seen.emplace(stream.id.stage, stream.id.kind, stream.id.owner, stream.id.ordinal).second) {
            why = "the manifest declares the same stream identity twice";
            return false;
        }
        uint64_t at = 0;
        for (const auto& chunk : stream.chunks) {
            if (chunk.length == 0 || chunk.offset != at || chunk.length > limits.max_stream_bytes ||
                !add(at, chunk.length)) {
                why = "a stream's chunks do not cover it contiguously with non-zero lengths";
                return false;
            }
            const auto object = objects.find(chunk.object);
            if (object == objects.end()) {
                why = "a chunk names an object the manifest does not declare";
                return false;
            }
            if (object->second != chunk.length) {
                why = "a chunk's length disagrees with the object it names";
                return false;
            }
        }
        if (at != stream.bytes) {
            why = "a stream's chunks do not add up to its length";
            return false;
        }
        if (!add(payload, stream.bytes)) {
            why = "the manifest's payload byte count overflows";
            return false;
        }
    }
    if (payload != meta.payload_bytes) {
        why = "the manifest's payload byte count disagrees with its streams";
        return false;
    }
    if (meta.payload_bytes > limits.max_image_bytes) {
        why = "the image is larger than ConversationStoreLimits::max_image_bytes";
        return false;
    }
    // The stream list must be exactly the canonical order of the container metadata: complete, correctly ordered and
    // carrying nothing the containers do not describe.
    size_t index = 0;
    auto visit = [&](const Slot& slot, const ContainerMeta&, std::string& reason) {
        if (index >= meta.streams.size()) {
            reason = "the container metadata describes more streams than the manifest holds";
            return false;
        }
        if (!(meta.streams[index].id == slot.id)) {
            reason = "a manifest stream does not match the canonical identity of its position";
            return false;
        }
        // The state figure is a *capacity* figure, not a structural one: bind_streams() refuses a running-state array
        // or a K/V part longer than ConversationStoreLimits::max_state_bytes, so the predicate has to apply exactly
        // the same rule.  Otherwise a run that serves from a smaller state budget than the profile that wrote the
        // record would read a perfectly sound record as a contradiction in itself - and recovery, which deletes what
        // it believes is corrupt, would discard it.  It is checked here, beside every other figure that describes this
        // run's admitted capacity, so the retention decode can classify the record as capacity it lacks instead.
        if (slot.buffer >= Buffer::gdn && meta.streams[index].bytes > limits.max_state_bytes) {
            reason = "a state stream is larger than ConversationStoreLimits::max_state_bytes";
            return false;
        }
        if (slot.buffer == Buffer::draft_boundary &&
            meta.streams[index].bytes != meta.containers[slot.container].draft_boundary_bytes) {
            reason = "draft boundary stream disagrees with its declared row length";
            return false;
        }
        ++index;
        return true;
    };
    if (!walk_streams(meta.containers, visit, why)) return false;
    if (index != meta.streams.size()) {
        why = "the manifest holds streams the container metadata does not describe";
        return false;
    }
    return true;
}

bool check_cuts(const ManifestMeta& meta, const ConversationStoreLimits& limits, std::string& why) {
    if (meta.cuts.size() > size_t(limits.max_checkpoints) + 1) {
        why = "the manifest advertises more cuts than ConversationStoreLimits::max_checkpoints";
        return false;
    }
    for (size_t i = 0; i < meta.cuts.size(); ++i) {
        const CutRecord& cut = meta.cuts[i];
        if (cut.tokens == 0 || cut.tokens > limits.max_tokens) {
            why = "a cut has an unsupported token length";
            return false;
        }
        if (i && meta.cuts[i - 1].tokens < cut.tokens) {
            why = "the manifest's cuts are not ordered longest first";
            return false;
        }
    }
    return true;
}

bool check_manifest(const ManifestMeta& meta, const ConversationStoreLimits& limits, std::string& why) {
    return check_containers(meta, limits, why) && check_streams(meta, limits, why) && check_cuts(meta, limits, why);
}

// ---------------------------------------------------------------------------------------------------------------
// Restore binding
// ---------------------------------------------------------------------------------------------------------------

// Writes into the final buffer of one stream, bounds-checked against the length the manifest declared.
RestoreTarget target_for(std::vector<int32_t>& ids) {
    RestoreTarget target;
    target.bytes = ids.size() * sizeof(int32_t);
    target.write = [&ids](uint64_t offset, const uint8_t* src, size_t n, std::string& why) {
        const uint64_t bytes = ids.size() * sizeof(int32_t);
        if (offset > bytes || n > bytes - offset) {
            why = "a token-id write is out of range";
            return false;
        }
        std::memcpy(reinterpret_cast<uint8_t*>(ids.data()) + offset, src, n);
        return true;
    };
    return target;
}

RestoreTarget target_for(std::vector<ConversationImageKey>& imgs) {
    RestoreTarget target;
    target.bytes = imgs.size() * sizeof(ConversationImageKey);
    target.write = [&imgs](uint64_t offset, const uint8_t* src, size_t n, std::string& why) {
        const uint64_t bytes = imgs.size() * sizeof(ConversationImageKey);
        if (offset > bytes || n > bytes - offset) {
            why = "an image-key write is out of range";
            return false;
        }
        std::memcpy(reinterpret_cast<uint8_t*>(imgs.data()) + offset, src, n);
        return true;
    };
    return target;
}

RestoreTarget target_for(CheckpointBytes& bytes, const char* what) {
    RestoreTarget target;
    target.bytes = bytes.size();
    target.write = [&bytes, what](uint64_t offset, const uint8_t* src, size_t n, std::string& why) {
        const uint64_t total = bytes.size();
        if (offset > total || n > total - offset) {
            why = std::string(what) + " write is out of range";
            return false;
        }
        std::vector<uint8_t>& dst = bytes.write();
        std::memcpy(dst.data() + offset, src, n);
        return true;
    };
    return target;
}

RestoreTarget target_for(ConversationBuffer& buffer, const char* what) {
    RestoreTarget target;
    target.bytes = buffer.size();
    target.write = [&buffer, what](uint64_t offset, const uint8_t* src, size_t n, std::string& why) {
        if (offset > buffer.size() || n > buffer.size() - offset) {
            why = std::string(what) + " write is out of range";
            return false;
        }
        size_t copied = 0;
        const bool ok = buffer.visit(size_t(offset), n, [&](uint8_t* dst, size_t count, size_t) {
            std::memcpy(dst, src + copied, count);
            copied += count;
            return true;
        });
        if (!ok || copied != n) {
            why = std::string(what) + " write failed";
            return false;
        }
        return true;
    };
    return target;
}

const char* buffer_name(Buffer buffer) {
    switch (buffer) {
        case Buffer::gdn: return "gdn";
        case Buffer::ple: return "ple";
        case Buffer::tails: return "tails";
        case Buffer::dead: return "dead";
        case Buffer::block_pos: return "block_pos";
        default: break;
    }
    return "state";
}

const char* kv_name(Buffer buffer) {
    switch (buffer) {
        case Buffer::kv_k: return "k";
        case Buffer::kv_v: return "v";
        case Buffer::kv_k_scale: return "k_scale";
        case Buffer::kv_v_scale: return "v_scale";
        case Buffer::kv_pooled: return "pooled";
        default: break;
    }
    return "part";
}

int kv_part_index(Buffer buffer) {
    switch (buffer) {
        case Buffer::kv_k: return 0;
        case Buffer::kv_v: return 1;
        case Buffer::kv_k_scale: return 2;
        case Buffer::kv_v_scale: return 3;
        case Buffer::kv_pooled: return 4;
        default: break;
    }
    return -1;
}

// Binds every stream of the canonical order to its final buffer, sized from the manifest's own declared lengths.
bool bind_streams(const ManifestMeta& meta, const ConversationStoreLimits& limits, SavedConversation& out,
                  std::vector<RestoreTarget>& targets, std::string& why) {
    const uint32_t stages = meta.stages;
    out.scope = meta.containers[0].scope;
    out.geometry = meta.containers[0].geometry;
    out.layer_lo = meta.containers[0].layer_lo;
    out.layer_hi = meta.containers[0].layer_hi;
    out.cvec = meta.containers[0].cvec;
    out.checkpoints.assign(meta.containers[0].checkpoints.size(), ConversationCheckpoint{});
    for (uint32_t j = 0; j < meta.containers[0].checkpoints.size(); ++j) {
        out.checkpoints[j].used = meta.containers[0].checkpoints[j].used;
        out.checkpoints[j].pinned = meta.containers[0].checkpoints[j].pinned;
    }
    out.live.used = meta.containers[0].record.used;
    out.live.pinned = meta.containers[0].record.pinned;
    out.kv.assign(meta.containers[0].kv.size(), ConversationKv{});
    for (uint32_t k = 0; k < meta.containers[0].kv.size(); ++k) {
        const KvMeta& source = meta.containers[0].kv[k];
        ConversationKv& layer = out.kv[k];
        layer.format = int(source.format);
        layer.cells = source.cells;
        layer.heads = source.heads;
        layer.head_dim = source.head_dim;
        layer.page_size = source.page_size;
        layer.pooled_rows = source.pooled_rows;
        layer.idx_dim = source.idx_dim;
    }
    out.stage_images.resize(stages - 1);
    for (uint32_t s = 1; s < stages; ++s) {
        const ContainerMeta& c = meta.containers[s];
        SavedConversation& image = out.stage_images[s - 1];
        image.scope = c.scope;
        image.geometry = c.geometry;
        image.layer_lo = c.layer_lo;
        image.layer_hi = c.layer_hi;
        image.cvec = c.cvec;
        image.live.used = c.record.used;
        image.live.pinned = c.record.pinned;
        image.checkpoints.assign(c.checkpoints.size(), ConversationCheckpoint{});
        for (uint32_t j = 0; j < c.checkpoints.size(); ++j) {
            image.checkpoints[j].used = c.checkpoints[j].used;
            image.checkpoints[j].pinned = c.checkpoints[j].pinned;
        }
        image.kv.assign(c.kv.size(), ConversationKv{});
        for (uint32_t k = 0; k < c.kv.size(); ++k) {
            const KvMeta& source = c.kv[k];
            ConversationKv& layer = image.kv[k];
            layer.format = int(source.format);
            layer.cells = source.cells;
            layer.heads = source.heads;
            layer.head_dim = source.head_dim;
            layer.page_size = source.page_size;
            layer.pooled_rows = source.pooled_rows;
            layer.idx_dim = source.idx_dim;
        }
    }
    // Every record's stage parts exist before the walk, so the parts are attached purely by container position.
    for (uint32_t s = 0; s < stages; ++s) {
        const ContainerMeta& c = meta.containers[s];
        SavedConversation& image = s == 0 ? out : out.stage_images[s - 1];
        image.draft_begin = c.draft_begin;
        image.draft_cells = c.draft_cells;
        image.draft_next_token = c.draft_next_token;
        image.live.stage_parts.resize(c.record.parts);
        for (uint32_t j = 0; j < c.checkpoints.size(); ++j)
            image.checkpoints[j].stage_parts.resize(c.checkpoints[j].parts);
    }

    targets.clear();
    targets.reserve(meta.streams.size());
    size_t index = 0;
    auto visit = [&](const Slot& slot, const ContainerMeta& c, std::string& reason) {
        if (index >= meta.streams.size()) {
            reason = "the manifest holds fewer streams than its containers describe";
            return false;
        }
        const StreamRecord& stream = meta.streams[index++];
        if (!(stream.id == slot.id)) {
            reason = "a manifest stream does not match the canonical identity of its position";
            return false;
        }
        ConversationCheckpoint* record = nullptr;
        ConversationKv* layer = nullptr;
        uint64_t declared_tokens = 0;
        if (c.is_image) {
            SavedConversation& image = slot.container == 0 ? out : out.stage_images[slot.container - 1];
            if (slot.buffer >= Buffer::kv_k) {
                if (slot.kv >= image.kv.size()) {
                    reason = "a K/V stream names a layer the image does not hold";
                    return false;
                }
                layer = &image.kv[slot.kv];
            } else if (slot.record_key == 0) {
                record = &image.live;
                declared_tokens = c.record.tokens;
            } else {
                if (slot.record_key > image.checkpoints.size()) {
                    reason = "a stream names a checkpoint the container does not hold";
                    return false;
                }
                record = &image.checkpoints[size_t(slot.record_key) - 1];
                declared_tokens = c.checkpoints[size_t(slot.record_key) - 1].tokens;
            }
        } else {
            SavedConversation& owner = c.parent_stage == 0 ? out : out.stage_images[c.parent_stage - 1];
            std::vector<ConversationCheckpoint>& parts =
                c.parent_owner == 0 ? owner.live.stage_parts : owner.checkpoints[size_t(c.parent_owner) - 1].stage_parts;
            if (c.part_index >= parts.size()) {
                reason = "a stage part stream names a part the record does not hold";
                return false;
            }
            record = &parts[c.part_index];
            record->used = c.record.used;
            record->pinned = c.record.pinned;
            declared_tokens = c.record.tokens;
        }
        switch (slot.buffer) {
            case Buffer::ids: {
                if (stream.bytes % sizeof(int32_t) != 0) {
                    reason = "a token-id stream is not a whole number of int32 elements";
                    return false;
                }
                const uint64_t tokens = stream.bytes / sizeof(int32_t);
                if (tokens > limits.max_tokens) {
                    reason = "a token-id stream is longer than ConversationStoreLimits::max_tokens";
                    return false;
                }
                if (tokens != declared_tokens) {
                    reason = "a token-id stream disagrees with the record's declared token count";
                    return false;
                }
                record->ids.assign(size_t(tokens), 0);
                targets.push_back(target_for(record->ids));
                return true;
            }
            case Buffer::imgs: {
                if (stream.bytes % sizeof(ConversationImageKey) != 0) {
                    reason = "an image-key stream is not a whole number of keys";
                    return false;
                }
                record->imgs.assign(size_t(stream.bytes / sizeof(ConversationImageKey)), ConversationImageKey{});
                targets.push_back(target_for(record->imgs));
                return true;
            }
            case Buffer::draft_boundary: {
                SavedConversation& image = slot.container == 0 ? out : out.stage_images[slot.container - 1];
                image.draft_boundary_residual.resize(size_t(stream.bytes));
                targets.push_back(target_for(image.draft_boundary_residual, "draft boundary residual"));
                return true;
            }
            case Buffer::kv_k:
            case Buffer::kv_v:
            case Buffer::kv_k_scale:
            case Buffer::kv_v_scale:
            case Buffer::kv_pooled: {
                const int part = kv_part_index(slot.buffer);
                if (!layer || part < 0 || stream.bytes != c.kv[slot.kv].part[part]) {
                    reason = "a K/V stream disagrees with the K/V layout the manifest declares";
                    return false;
                }
                if (stream.bytes > limits.max_state_bytes) {
                    reason = "a K/V part is larger than ConversationStoreLimits::max_state_bytes";
                    return false;
                }
                ConversationBuffer* buffer = nullptr;
                switch (slot.buffer) {
                    case Buffer::kv_k: buffer = &layer->k; break;
                    case Buffer::kv_v: buffer = &layer->v; break;
                    case Buffer::kv_k_scale: buffer = &layer->k_scale; break;
                    case Buffer::kv_v_scale: buffer = &layer->v_scale; break;
                    default: buffer = &layer->pooled; break;
                }
                buffer->resize(size_t(stream.bytes));
                targets.push_back(target_for(*buffer, kv_name(slot.buffer)));
                return true;
            }
            default: {
                if (stream.bytes > limits.max_state_bytes) {
                    reason = "a state stream is larger than ConversationStoreLimits::max_state_bytes";
                    return false;
                }
                CheckpointBytes* bytes = nullptr;
                switch (slot.buffer) {
                    case Buffer::gdn: bytes = &record->gdn; break;
                    case Buffer::ple: bytes = &record->ple; break;
                    case Buffer::tails: bytes = &record->tails; break;
                    case Buffer::dead: bytes = &record->dead; break;
                    default: bytes = &record->block_pos; break;
                }
                bytes->resize(size_t(stream.bytes));
                targets.push_back(target_for(*bytes, buffer_name(slot.buffer)));
                return true;
            }
        }
    };
    if (!walk_streams(meta.containers, visit, why)) return false;
    if (index != meta.streams.size()) {
        why = "the manifest holds streams its containers do not describe";
        return false;
    }
    return true;
}

bool container_meta_equal(const ContainerMeta& a, const ContainerMeta& b) {
    auto same_record = [](const RecordMeta& x, const RecordMeta& y) {
        return x.tokens == y.tokens && x.used == y.used && x.pinned == y.pinned && x.parts == y.parts;
    };
    if (a.is_image != b.is_image || a.stage != b.stage || a.parent_stage != b.parent_stage ||
        a.parent_owner != b.parent_owner || a.part_index != b.part_index || a.carve_known != b.carve_known)
        return false;
    if (!(a.scope == b.scope) || !same_record(a.record, b.record) || a.geometry != b.geometry ||
        a.layer_lo != b.layer_lo || a.layer_hi != b.layer_hi || a.cvec != b.cvec)
        return false;
    if (a.draft_begin != b.draft_begin || a.draft_cells != b.draft_cells ||
        a.draft_next_token != b.draft_next_token || a.draft_boundary_bytes != b.draft_boundary_bytes)
        return false;
    if (a.checkpoints.size() != b.checkpoints.size()) return false;
    for (size_t i = 0; i < a.checkpoints.size(); ++i)
        if (!same_record(a.checkpoints[i], b.checkpoints[i])) return false;
    if (a.kv.size() != b.kv.size()) return false;
    for (size_t i = 0; i < a.kv.size(); ++i) {
        const KvMeta& x = a.kv[i];
        const KvMeta& y = b.kv[i];
        if (x.format != y.format || x.cells != y.cells || x.heads != y.heads || x.head_dim != y.head_dim ||
            x.page_size != y.page_size || x.pooled_rows != y.pooled_rows || x.idx_dim != y.idx_dim ||
            x.part[0] != y.part[0] || x.part[1] != y.part[1] || x.part[2] != y.part[2] || x.part[3] != y.part[3] ||
            x.part[4] != y.part[4])
            return false;
    }
    return true;
}

bool stream_bytes_equal(const StreamSource& a, const StreamSource& b, std::string& why) {
    if (a.info.bytes != b.info.bytes) return false;
    const size_t slab = 64 * 1024;
    std::vector<uint8_t> left(slab), right(slab);
    uint64_t at = 0;
    while (at < a.info.bytes) {
        const size_t n = size_t(std::min<uint64_t>(slab, a.info.bytes - at));
        if (!a.read(at, left.data(), n, why) || !b.read(at, right.data(), n, why)) return false;
        if (std::memcmp(left.data(), right.data(), n) != 0) return false;
        at += n;
    }
    return true;
}

void seal_record(ConversationCheckpoint& record) { record.seal(); }

void seal_tree(SavedConversation& image) {
    image.draft_boundary_residual.freeze();
    seal_record(image.live);
    for (auto& checkpoint : image.checkpoints) seal_record(checkpoint);
    for (auto& stage : image.stage_images) seal_tree(stage);
}

} // namespace

// ---------------------------------------------------------------------------------------------------------------
// Enumerating a live tree
// ---------------------------------------------------------------------------------------------------------------

ConversationStoreLimits unconstrained_limits() {
    ConversationStoreLimits limits;
    limits.max_image_bytes = std::numeric_limits<uint64_t>::max();
    limits.max_stream_bytes = std::numeric_limits<uint64_t>::max();
    limits.max_manifest_bytes = std::numeric_limits<uint64_t>::max();
    limits.max_tokens = std::numeric_limits<uint64_t>::max();
    limits.max_state_bytes = std::numeric_limits<uint64_t>::max();
    limits.max_checkpoints = std::numeric_limits<uint32_t>::max() / 2;
    limits.max_kv_layers = std::numeric_limits<uint32_t>::max() / 2;
    limits.max_streams = std::numeric_limits<uint32_t>::max() / 2;
    limits.max_stages = kFormatMaxStages;
    limits.max_images = std::numeric_limits<uint32_t>::max() / 2;
    return limits;
}

bool image_layout(const SavedConversation& image, const Digest& seed, const ConversationStoreLimits& limits,
                  ImageLayout& out, std::string& why) {
    out = ImageLayout{};
    std::vector<TreeContainer> tree;
    if (!build_tree(image, limits, tree, why)) return false;
    out.containers.reserve(tree.size());
    for (const auto& container : tree) out.containers.push_back(container.meta);
    out.stages = 1;
    for (const auto& container : out.containers)
        if (container.is_image && container.stage + 1 > out.stages) out.stages = container.stage + 1;

    std::set<std::tuple<uint32_t, uint16_t, uint64_t, uint64_t>> seen;
    auto add_stream = [&](const Slot& slot, uint64_t bytes, uint32_t element, StreamRead read) {
        if (out.streams.size() >= limits.max_streams) {
            why = "the image has more streams than ConversationStoreLimits::max_streams";
            return false;
        }
        if (bytes > limits.max_stream_bytes) {
            why = "a stream is longer than ConversationStoreLimits::max_stream_bytes";
            return false;
        }
        if (!seen.emplace(slot.id.stage, slot.id.kind, slot.id.owner, slot.id.ordinal).second) {
            why = "the image has two streams with the same identity";
            return false;
        }
        StreamSource source;
        source.info.id = slot.id;
        source.info.index = uint32_t(out.streams.size());
        source.info.bytes = bytes;
        source.element = element;
        source.read = std::move(read);
        out.streams.push_back(std::move(source));
        if (!add(out.payload_bytes, bytes)) {
            why = "the image's payload byte count overflows";
            return false;
        }
        return true;
    };

    auto visit = [&](const Slot& slot, const ContainerMeta&, std::string& reason) {
        const TreeContainer& container = tree[slot.container];
        const bool image_container_ = container.meta.is_image;
        const ConversationCheckpoint* record =
            image_container_ && slot.record_key != 0 ? container.checkpoints[size_t(slot.record_key) - 1]
                                                     : container.own;
        if (!record) {
            reason = "the image does not hold the record a stream belongs to";
            return false;
        }
        switch (slot.buffer) {
            case Buffer::ids: {
                const std::vector<int32_t>* ids = &record->ids;
                const uint64_t bytes = uint64_t(ids->size()) * sizeof(int32_t);
                return add_stream(slot, bytes, uint32_t(sizeof(int32_t)),
                                  [ids, bytes](uint64_t offset, uint8_t* dst, size_t n, std::string& why) {
                                      if (offset > bytes || n > bytes - offset) {
                                          why = "a token-id read is out of range";
                                          return false;
                                      }
                                      std::memcpy(dst, reinterpret_cast<const uint8_t*>(ids->data()) + offset, n);
                                      return true;
                                  });
            }
            case Buffer::imgs: {
                const std::vector<ConversationImageKey>* imgs = &record->imgs;
                const uint64_t bytes = uint64_t(imgs->size()) * sizeof(ConversationImageKey);
                return add_stream(slot, bytes, uint32_t(sizeof(ConversationImageKey)),
                                  [imgs, bytes](uint64_t offset, uint8_t* dst, size_t n, std::string& why) {
                                      if (offset > bytes || n > bytes - offset) {
                                          why = "an image-key read is out of range";
                                          return false;
                                      }
                                      std::memcpy(dst, reinterpret_cast<const uint8_t*>(imgs->data()) + offset, n);
                                      return true;
                                  });
            }
            case Buffer::draft_boundary: {
                const ConversationBuffer* buffer = container.boundary;
                const uint64_t bytes = buffer->size();
                return add_stream(slot, bytes, 1,
                    [buffer, bytes](uint64_t offset, uint8_t* dst, size_t n, std::string& why) {
                        if (offset > bytes || n > bytes - offset || !buffer->read(dst, size_t(offset), n)) {
                            why = "a draft boundary residual read failed";
                            return false;
                        }
                        return true;
                    });
            }
            case Buffer::kv_k:
            case Buffer::kv_v:
            case Buffer::kv_k_scale:
            case Buffer::kv_v_scale:
            case Buffer::kv_pooled: {
                if (!image_container_ || slot.kv >= container.kv.size()) {
                    reason = "the image does not hold the K/V layer a stream belongs to";
                    return false;
                }
                const ConversationKv& layer = *container.kv[slot.kv];
                const ConversationBuffer* buffer = nullptr;
                switch (slot.buffer) {
                    case Buffer::kv_k: buffer = &layer.k; break;
                    case Buffer::kv_v: buffer = &layer.v; break;
                    case Buffer::kv_k_scale: buffer = &layer.k_scale; break;
                    case Buffer::kv_v_scale: buffer = &layer.v_scale; break;
                    default: buffer = &layer.pooled; break;
                }
                const uint64_t bytes = buffer->size();
                return add_stream(slot, bytes, 1,
                                  [buffer, bytes](uint64_t offset, uint8_t* dst, size_t n, std::string& why) {
                                      if (offset > bytes || n > bytes - offset) {
                                          why = "a K/V read is out of range";
                                          return false;
                                      }
                                      if (!buffer->read(dst, size_t(offset), n)) {
                                          why = "a K/V read failed";
                                          return false;
                                      }
                                      return true;
                                  });
            }
            default: {
                const CheckpointBytes* bytes = nullptr;
                switch (slot.buffer) {
                    case Buffer::gdn: bytes = &record->gdn; break;
                    case Buffer::ple: bytes = &record->ple; break;
                    case Buffer::tails: bytes = &record->tails; break;
                    case Buffer::dead: bytes = &record->dead; break;
                    default: bytes = &record->block_pos; break;
                }
                const std::vector<uint8_t>* payload = &bytes->read();
                const uint64_t length = payload->size();
                return add_stream(slot, length, 1,
                                  [payload, length](uint64_t offset, uint8_t* dst, size_t n, std::string& why) {
                                      if (offset > length || n > length - offset) {
                                          why = "a state read is out of range";
                                          return false;
                                      }
                                      std::memcpy(dst, payload->data() + offset, n);
                                      return true;
                                  });
            }
        }
    };
    if (!walk_streams(out.containers, visit, why)) return false;
    if (out.payload_bytes > limits.max_image_bytes) {
        why = "the image is larger than ConversationStoreLimits::max_image_bytes";
        return false;
    }
    build_cuts(image, seed, limits, out.cuts);
    return true;
}

uint64_t image_payload_bytes(const SavedConversation& image) {
    ImageLayout layout;
    std::string why;
    if (!image_layout(image, Digest{}, unconstrained_limits(), layout, why)) return 0;
    return layout.payload_bytes;
}

// ---------------------------------------------------------------------------------------------------------------
// Allocation-free counting (see the header).  Nothing below allocates, reads a payload or depends on a configured
// maximum: every figure is the actual object's own element count plus a documented per-allocation allowance.
// ---------------------------------------------------------------------------------------------------------------

namespace {

uint64_t sat_add(uint64_t a, uint64_t b) { return a > UINT64_MAX - b ? UINT64_MAX : a + b; }
uint64_t sat_mul(uint64_t a, uint64_t b) {
    return a != 0 && b > UINT64_MAX / a ? UINT64_MAX : a * b;
}

// The most elements of `min_entry` encoded bytes a file within `ceiling` bytes could hold, saturating at half the
// count a uint32 count field can carry (the decoder derives the cut ceiling as this plus one, and unconstrained_limits
// saturates the same way).  Past this the file's own length is the binding bound, never this figure.
uint32_t encoded_ceiling(uint64_t ceiling, uint64_t min_entry) {
    const uint64_t count = min_entry ? ceiling / min_entry : ceiling;
    constexpr uint64_t kCeiling = std::numeric_limits<uint32_t>::max() / 2;
    return count > kCeiling ? uint32_t(kCeiling) : uint32_t(count);
}

// One record's streams: its token ids, its image keys and its five running-state arrays.  `parts` are counted by the
// caller, which turns each into a container of its own.
bool count_record(const ConversationCheckpoint& record, const ConversationStoreLimits& limits, ImageShape& out,
                  std::string& why) {
    if (record.ids.size() > size_t(limits.max_tokens)) {
        why = "an image record holds more tokens than ConversationStoreLimits::max_tokens";
        return false;
    }
    const uint64_t states[5] = {uint64_t(record.gdn.size()), uint64_t(record.ple.size()),
                                uint64_t(record.tails.size()), uint64_t(record.dead.size()),
                                uint64_t(record.block_pos.size())};
    uint64_t bytes = sat_mul(uint64_t(record.ids.size()), sizeof(int32_t));
    bytes = sat_add(bytes, sat_mul(uint64_t(record.imgs.size()), sizeof(ConversationImageKey)));
    for (uint64_t state : states) {
        if (state > limits.max_stream_bytes) {
            why = "a stream is longer than ConversationStoreLimits::max_stream_bytes";
            return false;
        }
        if (state > limits.max_state_bytes) {
            why = "a state stream is longer than ConversationStoreLimits::max_state_bytes";
            return false;
        }
        bytes = sat_add(bytes, state);
    }
    if (sat_mul(uint64_t(record.ids.size()), sizeof(int32_t)) > limits.max_stream_bytes ||
        sat_mul(uint64_t(record.imgs.size()), sizeof(ConversationImageKey)) > limits.max_stream_bytes) {
        why = "a stream is longer than ConversationStoreLimits::max_stream_bytes";
        return false;
    }
    out.records = sat_add(out.records, 1);
    out.streams = sat_add(out.streams, 7);
    out.payload_bytes = sat_add(out.payload_bytes, bytes);
    out.max_record_images = std::max(out.max_record_images, uint64_t(record.imgs.size()));
    out.max_stream_bytes = std::max(out.max_stream_bytes, sat_mul(uint64_t(record.ids.size()), sizeof(int32_t)));
    out.max_stream_bytes = std::max(out.max_stream_bytes,
                                    sat_mul(uint64_t(record.imgs.size()), sizeof(ConversationImageKey)));
    for (uint64_t state : states) out.max_stream_bytes = std::max(out.max_stream_bytes, state);
    return true;
}

bool count_kv_layer(const ConversationKv& layer, const ConversationStoreLimits& limits, ImageShape& out,
                    std::string& why) {
    const uint64_t parts[5] = {uint64_t(layer.k.size()), uint64_t(layer.v.size()), uint64_t(layer.k_scale.size()),
                               uint64_t(layer.v_scale.size()), uint64_t(layer.pooled.size())};
    uint64_t bytes = 0;
    for (uint64_t part : parts) {
        if (part > limits.max_stream_bytes) {
            why = "a stream is longer than ConversationStoreLimits::max_stream_bytes";
            return false;
        }
        if (part > limits.max_state_bytes) {
            why = "a K/V part is longer than ConversationStoreLimits::max_state_bytes";
            return false;
        }
        bytes = sat_add(bytes, part);
    }
    out.kv_layers = sat_add(out.kv_layers, 1);
    out.streams = sat_add(out.streams, 5);
    out.payload_bytes = sat_add(out.payload_bytes, bytes);
    for (uint64_t part : parts) out.max_stream_bytes = std::max(out.max_stream_bytes, part);
    return true;
}

// One container's own records: the live record when `live` is the top record of a whole image, or the single record
// of a stage part.
bool count_container(const ConversationCheckpoint& record, const ConversationCheckpoint* checkpoints,
                     size_t checkpoint_count, const ConversationKv* kv, size_t kv_count,
                     const ConversationStoreLimits& limits, ImageShape& out, std::string& why) {
    out.containers = sat_add(out.containers, 1);
    if (!count_record(record, limits, out, why)) return false;
    for (size_t i = 0; i < checkpoint_count; ++i)
        if (!count_record(checkpoints[i], limits, out, why)) return false;
    for (size_t i = 0; i < kv_count; ++i)
        if (!count_kv_layer(kv[i], limits, out, why)) return false;
    return true;
}

// The containers one whole image's records contribute: one per retained stage part, refused exactly as build_tree()
// refuses them (a part that owns further parts, the container ceiling).
bool count_parts(const ConversationCheckpoint& record, const ConversationStoreLimits& limits, ImageShape& out,
                 std::string& why) {
    for (const auto& part : record.stage_parts) {
        if (!part.stage_parts.empty()) {
            why = "a stage part that owns further stage parts is not representable";
            return false;
        }
        if (out.containers >= limits.max_images) {
            why = "the image has more containers than ConversationStoreLimits::max_images";
            return false;
        }
        if (!count_container(part, nullptr, 0, nullptr, 0, limits, out, why)) return false;
    }
    return true;
}

} // namespace

uint64_t stream_chunk_count(uint64_t bytes, uint32_t element, uint64_t chunk_limit) {
    if (bytes == 0) return 0;
    uint64_t size = chunk_limit;
    if (element > 1) size -= size % element;
    if (size == 0) return bytes;   // a degenerate chunk size can never merge two bytes into one object
    return bytes <= size ? 1 : (bytes + size - 1) / size;
}

uint64_t donor_chunks_in_interval(const StreamRecord& donor, uint64_t offset, uint64_t bytes) {
    const uint64_t end = sat_add(offset, bytes);
    uint64_t count = 0;
    for (const auto& chunk : donor.chunks) {
        if (sat_add(chunk.offset, chunk.length) <= offset) continue;
        if (chunk.offset >= end) continue;
        count = sat_add(count, 1);
    }
    return count;
}

uint64_t prefix_chain_workspace(uint64_t image_keys) {
    // The chain's own sorted index (one pair per image key), one state per group and the allocator's header for each.
    // A state is a Sha256 plus the EVP context a copy of it allocates, which this tree cannot size: see
    // kChainBytesPerImageKey.
    return sat_add(sat_mul(image_keys, sizeof(std::pair<uint64_t, size_t>) + kChainBytesPerImageKey),
                   sat_add(sat_mul(image_keys, 8), 2 * kAllocSlackBytes));
}

uint64_t image_layout_workspace(const ImageShape& shape) {
    uint64_t bytes = 0;
    // build_tree() holds its own roster (a ContainerMeta and two pointer vectors per container) while the layout
    // copies those metadata out, so both exist at once.
    bytes = sat_add(bytes, sat_mul(shape.containers, sizeof(TreeContainer) + sizeof(ContainerMeta) +
                                                   2 * kAllocSlackBytes));
    bytes = sat_add(bytes, sat_mul(shape.records, sizeof(RecordMeta) + kAllocSlackBytes));
    bytes = sat_add(bytes, sat_mul(shape.kv_layers, sizeof(KvMeta) + kAllocSlackBytes));
    // One StreamSource per stream, and one heap block for the closure each source's StreamRead holds.
    bytes = sat_add(bytes, sat_mul(shape.streams, sizeof(StreamSource) + 2 * kAllocSlackBytes));
    // The duplicate-identity set image_layout() keeps while it builds, one node per stream.
    bytes = sat_add(bytes, sat_mul(shape.streams, kAllocSlackBytes));
    bytes = sat_add(bytes, sat_mul(shape.cuts, sizeof(CutRecord) + kAllocSlackBytes));
    bytes = sat_add(bytes, prefix_chain_workspace(shape.max_record_images));
    return sat_add(bytes, 8 * kAllocSlackBytes);
}

bool image_shape(const SavedConversation& image, const ConversationStoreLimits& limits, ImageShape& out,
                 std::string& why) {
    out = ImageShape{};
    for (const auto& stage : image.stage_images) {
        if (!stage.stage_images.empty()) {
            why = "a stage image that owns its own stage images is not representable";
            return false;
        }
    }
    if (image.stage_images.size() + 1 > limits.max_stages) {
        why = "the image declares more stages than ConversationStoreLimits::max_stages";
        return false;
    }
    // The whole images, in build_tree()'s order: the top image, then every later stage's own image.
    uint32_t stage_index = 0;
    const auto count_image = [&](const SavedConversation& container) -> bool {
        if (!check_image_draft(container, stage_index++, uint32_t(image.stage_images.size()) + 1, limits, why))
            return false;
        const uint64_t boundary = container.draft_boundary_residual.size();
        out.certified = out.certified || container.draft_begin != -1;
        if (boundary) {
            out.streams = sat_add(out.streams, 1);
            out.payload_bytes = sat_add(out.payload_bytes, boundary);
            out.max_stream_bytes = std::max(out.max_stream_bytes, boundary);
        }
        if (!container.scope.valid()) {
            why = "an image has no valid authorized scope";
            return false;
        }
        if (container.checkpoints.size() > size_t(limits.max_checkpoints)) {
            why = "an image holds more checkpoints than ConversationStoreLimits::max_checkpoints";
            return false;
        }
        if (container.kv.size() > size_t(limits.max_kv_layers)) {
            why = "an image holds more K/V layers than ConversationStoreLimits::max_kv_layers";
            return false;
        }
        if (!count_container(container.live, container.checkpoints.data(), container.checkpoints.size(),
                             container.kv.data(), container.kv.size(), limits, out, why))
            return false;
        if (out.containers >= limits.max_images) {
            why = "the image has more containers than ConversationStoreLimits::max_images";
            return false;
        }
        return true;
    };
    if (!count_image(image)) return false;
    for (const auto& stage : image.stage_images)
        if (!count_image(stage)) return false;
    // Every record's stage parts become containers of their own, taking their carve from the stage image that
    // declares their stage.
    const auto count_record_parts = [&](const SavedConversation& container) -> bool {
        if (!count_parts(container.live, limits, out, why)) return false;
        for (const auto& checkpoint : container.checkpoints)
            if (!count_parts(checkpoint, limits, out, why)) return false;
        return true;
    };
    if (!count_record_parts(image)) return false;
    for (const auto& stage : image.stage_images)
        if (!count_record_parts(stage)) return false;
    // Only the top image advertises cuts, and only for records that every later stage can also be placed at.
    out.cuts = sat_add(1, uint64_t(image.checkpoints.size()));
    if (out.streams > limits.max_streams) {
        why = "the image has more streams than ConversationStoreLimits::max_streams";
        return false;
    }
    if (out.payload_bytes > limits.max_image_bytes) {
        why = "the image is larger than ConversationStoreLimits::max_image_bytes";
        return false;
    }
    return true;
}


bool images_equal(const SavedConversation& a, const SavedConversation& b, std::string& why) {
    const ConversationStoreLimits limits = unconstrained_limits();
    ImageLayout left, right;
    if (!image_layout(a, Digest{}, limits, left, why)) return false;
    if (!image_layout(b, Digest{}, limits, right, why)) return false;
    if (left.stages != right.stages || left.payload_bytes != right.payload_bytes) return false;
    if (left.containers.size() != right.containers.size()) return false;
    for (size_t i = 0; i < left.containers.size(); ++i)
        if (!container_meta_equal(left.containers[i], right.containers[i])) return false;
    if (left.streams.size() != right.streams.size()) {
        why = "the two images have different stream counts";
        return false;
    }
    for (size_t i = 0; i < left.streams.size(); ++i) {
        if (!(left.streams[i].info.id == right.streams[i].info.id)) {
            why = "the two images disagree about a stream's identity";
            return false;
        }
        if (left.streams[i].info.bytes != right.streams[i].info.bytes) {
            why = "the two images disagree about a stream's length";
            return false;
        }
        if (!stream_bytes_equal(left.streams[i], right.streams[i], why)) {
            why = "the two images disagree about a stream's bytes";
            return false;
        }
    }
    if (left.cuts.size() != right.cuts.size()) {
        why = "the two images advertise different cuts";
        return false;
    }
    for (size_t i = 0; i < left.cuts.size(); ++i)
        if (left.cuts[i].tokens != right.cuts[i].tokens || left.cuts[i].digest != right.cuts[i].digest ||
            left.cuts[i].live != right.cuts[i].live || left.cuts[i].pinned != right.cuts[i].pinned) {
            why = "the two images advertise different cuts";
            return false;
        }
    return true;
}

// ---------------------------------------------------------------------------------------------------------------
// Manifest
// ---------------------------------------------------------------------------------------------------------------

// ---------------------------------------------------------------------------------------------------------------
// Encoded sizes, counted rather than encoded
// ---------------------------------------------------------------------------------------------------------------

uint64_t manifest_encoded_size_of(const ManifestShape& shape) {
    // tag, payload, epoch, generation, parent, published_order, stages, bucket, domain
    uint64_t bytes = 116;
    bytes = sat_add(bytes, 4);                                        // the container count
    bytes = sat_add(bytes, sat_mul(shape.containers, 244));           // 236 fixed + the checkpoint and K/V counts
    if (shape.certified) bytes = sat_add(bytes, sat_mul(shape.containers, 28));
    bytes = sat_add(bytes, sat_mul(shape.container_records, 21));
    bytes = sat_add(bytes, sat_mul(shape.kv_layers, 96));
    bytes = sat_add(bytes, 4);                                        // the stream count
    bytes = sat_add(bytes, sat_mul(shape.streams, 34));               // identity, length and the chunk count
    bytes = sat_add(bytes, sat_mul(shape.chunks, 44));                // offset, length, content address
    bytes = sat_add(bytes, 4);                                        // the object count
    bytes = sat_add(bytes, sat_mul(shape.objects, 40));
    bytes = sat_add(bytes, 4);                                        // the cut count
    bytes = sat_add(bytes, sat_mul(shape.cuts, 42));
    return sat_add(bytes, 32);                                        // the trailing digest
}

ManifestShape manifest_shape(const ManifestMeta& meta) {
    ManifestShape shape;
    shape.containers = uint64_t(meta.containers.size());
    for (const auto& container : meta.containers) shape.container_records = sat_add(shape.container_records, uint64_t(container.checkpoints.size()));
    for (const auto& container : meta.containers) shape.kv_layers = sat_add(shape.kv_layers, uint64_t(container.kv.size()));
    shape.streams = uint64_t(meta.streams.size());
    for (const auto& stream : meta.streams) shape.chunks = sat_add(shape.chunks, uint64_t(stream.chunks.size()));
    shape.objects = uint64_t(meta.objects.size());
    shape.cuts = uint64_t(meta.cuts.size());
    for (const auto& c : meta.containers)
        if (c.draft_begin != -1 || c.draft_cells != -1 || c.draft_next_token != -1 || c.draft_boundary_bytes)
            shape.certified = true;
    return shape;
}

uint64_t manifest_encoded_size(const ManifestMeta& meta) { return manifest_encoded_size_of(manifest_shape(meta)); }

uint64_t manifest_workspace_bound(uint64_t encoded_bytes) {
    // Every decoded element, including explicit certification on a legacy-sized container, is less than twice
    // its encoded bytes. Each vector carries one allocator header, at most one per 21 encoded bytes in the
    // densest section. Twice the encoding plus a quarter covers both, including legacy 244-byte containers.
    //
    // This is an upper bound for *any* limits set, not only the configured one: every count the decoder accepts is
    // bounded by Reader::count against the bytes left for that section's smallest encoded entry (kContainerEncodedMin
    // and its siblings), so the sections' decoded bytes are bounded by the encoding times the largest element-to-entry
    // ratio in the format - a corner where each section is charged its own maximum at once cannot arise, because the
    // sections' own encoded bytes have to cover the whole file between them.
    return sat_add(sat_mul(encoded_bytes, 2), sat_add(encoded_bytes / 4, 2 * kAllocSlackBytes));
}

uint64_t manifest_read_workspace_bound(uint64_t encoded_bytes) {
    // Everything one manifest read and decode holds at once: the raw file, the decoded metadata
    // (manifest_workspace_bound) and the two tables the consistency pass builds - one kMapNodeBytes node per declared
    // object and one per declared stream.  The tables are bounded by the encoding itself rather than by a decode: the
    // object and stream sections are the format's densest, at kObjectEncodedMin and kStreamEncodedMin encoded bytes
    // per entry, so no more than encoded_bytes / 34 nodes of either can exist at all; four times the encoding covers
    // both generously (128/34 is 3.77) and no configured maximum, quota or decoded count takes part.  A caller that
    // has already reserved this figure for the same decode (a read's whole peak, a writer's donor preparation) must
    // not reserve it twice (see ConversationStore::Impl::load_manifest_of).
    const uint64_t tables = sat_mul(encoded_bytes, 4);
    return sat_add(sat_add(encoded_bytes, manifest_workspace_bound(encoded_bytes)), tables);
}

std::vector<uint8_t> manifest_encode(const ManifestMeta& meta) {
    Writer w;
    const auto shape = manifest_shape(meta);
    w.reserve(size_t(manifest_encoded_size_of(shape)));
    const bool certified = shape.certified;
    w.raw(certified ? kCertifiedManifestTag : kManifestTag, sizeof kManifestTag);
    w.u64(meta.payload_bytes);
    w.u64(meta.epoch);
    w.u64(meta.generation);
    w.u64(meta.parent);
    w.u64(meta.published_order);
    w.u32(meta.stages);
    w.digest(meta.bucket);
    w.digest(meta.domain);
    w.u32(uint32_t(meta.containers.size()));
    for (const auto& c : meta.containers) {
        w.u8(c.is_image ? 1 : 0);
        w.u32(c.stage);
        w.u32(c.parent_stage);
        w.u64(c.parent_owner);
        w.u32(c.part_index);
        w.u8(c.carve_known ? 1 : 0);
        w.raw(c.scope.value.data(), c.scope.value.size());
        w.u64(c.record.tokens);
        w.u64(c.record.used);
        w.u8(c.record.pinned ? 1 : 0);
        w.u32(c.record.parts);
        for (int64_t g : c.geometry) w.i64(g);
        w.i64(c.layer_lo);
        w.i64(c.layer_hi);
        w.u8(c.cvec ? 1 : 0);
        if (certified) {
            w.i64(c.draft_begin);
            w.i64(c.draft_cells);
            w.u32(uint32_t(c.draft_next_token));
            w.u64(c.draft_boundary_bytes);
        }
        w.u32(uint32_t(c.checkpoints.size()));
        for (const auto& checkpoint : c.checkpoints) {
            w.u64(checkpoint.tokens);
            w.u64(checkpoint.used);
            w.u8(checkpoint.pinned ? 1 : 0);
            w.u32(checkpoint.parts);
        }
        w.u32(uint32_t(c.kv.size()));
        for (const auto& layer : c.kv) {
            w.i64(layer.format);
            w.i64(layer.cells);
            w.i64(layer.heads);
            w.i64(layer.head_dim);
            w.i64(layer.page_size);
            w.i64(layer.pooled_rows);
            w.i64(layer.idx_dim);
            for (uint64_t part : layer.part) w.u64(part);
        }
    }
    w.u32(uint32_t(meta.streams.size()));
    for (const auto& stream : meta.streams) {
        w.u32(stream.id.stage);
        w.u16(stream.id.kind);
        w.u64(stream.id.owner);
        w.u64(stream.id.ordinal);
        w.u64(stream.bytes);
        w.u32(uint32_t(stream.chunks.size()));
        for (const auto& chunk : stream.chunks) {
            w.u64(chunk.offset);
            w.u32(chunk.length);
            w.digest(chunk.object);
        }
    }
    w.u32(uint32_t(meta.objects.size()));
    for (const auto& object : meta.objects) {
        w.digest(object.first);
        w.u64(object.second);
    }
    w.u32(uint32_t(meta.cuts.size()));
    for (const auto& cut : meta.cuts) {
        w.u64(cut.tokens);
        w.digest(cut.digest);
        w.u8(cut.live ? 1 : 0);
        w.u8(cut.pinned ? 1 : 0);
    }
    const Digest trailer = sha256(w.data().data(), w.size());
    w.digest(trailer);
    return w.take();
}

Digest manifest_id_of(const uint8_t* bytes, size_t size) { return sha256(bytes, size); }

bool manifest_decode(const uint8_t* bytes, size_t size, const ConversationStoreLimits& limits, ManifestMeta& out,
                     std::string& why) {
    out = ManifestMeta{};
    if (!bytes || size < sizeof(kManifestTag) + kTrailer + 4) {
        why = "the manifest is too short";
        return false;
    }
    if (size > limits.max_manifest_bytes) {
        why = "the manifest is larger than ConversationStoreLimits::max_manifest_bytes";
        return false;
    }
    const bool certified = std::memcmp(bytes, kCertifiedManifestTag, sizeof kCertifiedManifestTag) == 0;
    if (!certified && std::memcmp(bytes, kManifestTag, sizeof kManifestTag) != 0) {
        why = "the manifest does not start with the store's schema tag";
        return false;
    }
    Digest trailer{};
    std::memcpy(trailer.data(), bytes + size - kTrailer, kTrailer);
    if (sha256(bytes, size - kTrailer) != trailer) {
        why = "the manifest's checksum does not match its bytes";
        return false;
    }
    Reader r(bytes + sizeof kManifestTag, size - sizeof kManifestTag - kTrailer);
    uint32_t count = 0;
    if (!r.u64(out.payload_bytes) || !r.u64(out.epoch) || !r.u64(out.generation) || !r.u64(out.parent) ||
        !r.u64(out.published_order) || !r.u32(out.stages) || !r.digest(out.bucket) || !r.digest(out.domain)) {
        why = "the manifest's header is truncated";
        return false;
    }
    if (!r.count(count, limits.max_images, kContainerEncodedMin + (certified ? 28 : 0))) {
        why = "the manifest's container count is invalid";
        return false;
    }
    out.containers.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        ContainerMeta c;
        uint8_t flag = 0;
        if (!r.u8(flag) || !r.u32(c.stage) || !r.u32(c.parent_stage) || !r.u64(c.parent_owner) ||
            !r.u32(c.part_index)) {
            why = "a manifest container is truncated";
            return false;
        }
        c.is_image = flag != 0;
        if (!r.u8(flag)) {
            why = "a manifest container is truncated";
            return false;
        }
        c.carve_known = flag != 0;
        if (!r.raw(c.scope.value.data(), c.scope.value.size()) || !r.u64(c.record.tokens) || !r.u64(c.record.used) ||
            !r.u8(flag) || !r.u32(c.record.parts)) {
            why = "a manifest container is truncated";
            return false;
        }
        c.record.pinned = flag != 0;
        for (int64_t& g : c.geometry)
            if (!r.i64(g)) {
                why = "a manifest container's geometry is truncated";
                return false;
            }
        if (!r.i64(c.layer_lo) || !r.i64(c.layer_hi) || !r.u8(flag)) {
            why = "a manifest container is truncated";
            return false;
        }
        c.cvec = flag != 0;
        if (certified) {
            uint32_t token = 0;
            if (!r.i64(c.draft_begin) || !r.i64(c.draft_cells) || !r.u32(token) ||
                !r.u64(c.draft_boundary_bytes)) {
                why = "a manifest's draft certification is truncated";
                return false;
            }
            c.draft_next_token = int32_t(token);
        }
        uint32_t checkpoints = 0;
        if (!r.count(checkpoints, limits.max_checkpoints, kCheckpointEncodedMin)) {
            why = "a manifest container's checkpoint count is invalid";
            return false;
        }
        c.checkpoints.reserve(checkpoints);
        for (uint32_t j = 0; j < checkpoints; ++j) {
            RecordMeta record;
            if (!r.u64(record.tokens) || !r.u64(record.used) || !r.u8(flag) || !r.u32(record.parts)) {
                why = "a manifest checkpoint is truncated";
                return false;
            }
            record.pinned = flag != 0;
            c.checkpoints.push_back(record);
        }
        uint32_t layers = 0;
        if (!r.count(layers, limits.max_kv_layers, kKvLayerEncodedMin)) {
            why = "a manifest container's K/V layer count is invalid";
            return false;
        }
        c.kv.reserve(layers);
        for (uint32_t j = 0; j < layers; ++j) {
            KvMeta layer;
            if (!r.i64(layer.format) || !r.i64(layer.cells) || !r.i64(layer.heads) || !r.i64(layer.head_dim) ||
                !r.i64(layer.page_size) || !r.i64(layer.pooled_rows) || !r.i64(layer.idx_dim)) {
                why = "a manifest K/V layer is truncated";
                return false;
            }
            for (uint64_t& part : layer.part)
                if (!r.u64(part)) {
                    why = "a manifest K/V layer's part lengths are truncated";
                    return false;
                }
            c.kv.push_back(layer);
        }
        out.containers.push_back(std::move(c));
    }
    uint32_t streams = 0;
    if (!r.count(streams, limits.max_streams, kStreamEncodedMin)) {
        why = "the manifest's stream count is invalid";
        return false;
    }
    out.streams.reserve(streams);
    for (uint32_t i = 0; i < streams; ++i) {
        StreamRecord stream;
        if (!r.u32(stream.id.stage) || !r.u16(stream.id.kind) || !r.u64(stream.id.owner) || !r.u64(stream.id.ordinal) ||
            !r.u64(stream.bytes)) {
            why = "a manifest stream is truncated";
            return false;
        }
        uint32_t chunks = 0;
        if (!r.count(chunks, kCatalogCeiling, kChunkEncodedMin)) {
            why = "a manifest stream's chunk count is invalid";
            return false;
        }
        stream.chunks.reserve(chunks);
        for (uint32_t j = 0; j < chunks; ++j) {
            ChunkRecord chunk;
            if (!r.u64(chunk.offset) || !r.u32(chunk.length) || !r.digest(chunk.object)) {
                why = "a manifest chunk is truncated";
                return false;
            }
            stream.chunks.push_back(chunk);
        }
        out.streams.push_back(std::move(stream));
    }
    uint32_t objects = 0;
    if (!r.count(objects, kCatalogCeiling, kObjectEncodedMin)) {
        why = "the manifest's object count is invalid";
        return false;
    }
    out.objects.reserve(objects);
    for (uint32_t i = 0; i < objects; ++i) {
        Digest digest{};
        uint64_t bytes = 0;
        if (!r.digest(digest) || !r.u64(bytes)) {
            why = "a manifest object entry is truncated";
            return false;
        }
        out.objects.emplace_back(digest, bytes);
    }
    uint32_t cuts = 0;
    if (!r.count(cuts, limits.max_checkpoints + 1, kCutEncodedMin)) {
        why = "the manifest's cut count is invalid";
        return false;
    }
    out.cuts.reserve(cuts);
    for (uint32_t i = 0; i < cuts; ++i) {
        CutRecord cut;
        uint8_t live = 0, pinned = 0;
        if (!r.u64(cut.tokens) || !r.digest(cut.digest) || !r.u8(live) || !r.u8(pinned)) {
            why = "a manifest cut is truncated";
            return false;
        }
        cut.live = live != 0;
        cut.pinned = pinned != 0;
        out.cuts.push_back(cut);
    }
    if (!r.empty()) {
        why = "the manifest holds trailing bytes after its last section";
        return false;
    }
    out.manifest_id = manifest_id_of(bytes, size);
    return check_manifest(out, limits, why);
}

bool manifest_within_limits(const ManifestMeta& meta, const ConversationStoreLimits& limits, std::string& why) {
    // The same structural *and* capacity predicate the decoder applies to a record it has just decoded, asked of
    // metadata that was decoded under the wider retention figures.  A read decodes once and classifies with this, so
    // a record this run may not serve is refused (refused_limits) rather than decoded a second time to find that out.
    return check_manifest(meta, limits, why);
}

// ---------------------------------------------------------------------------------------------------------------
// Retention: the figures a record must satisfy to be *kept*, as opposed to the figures that decide whether a run may
// create or serve it
// ---------------------------------------------------------------------------------------------------------------

// A store root is shared by every serving profile that opens it, so a record's own figures and the *current* run's
// admitted ones disagree in the ordinary case: another profile captures a different stage count, keeps a different
// number of checkpoints and serves from a different image budget (see ConversationStoreLimits).  Whether such a
// record exists at all is a structural question and is answered from the record itself; whether this run may serve it
// is answered from the configured figures and is enforced on every read and every publication.  A recovery pass that
// mixed the two would delete a history this profile cannot read but the profile that wrote it still can.
//
// Retention is therefore bounded by what the *format* can represent, never by this run's capacity and never by the
// physical quota:
//   * the byte figures (image, stream and state) and the token ceiling by the widest record a manifest within the
//     configured manifest ceiling could declare at all - its encoded chunks, times the widest chunk its own length
//     field can carry.  A physical bound would be wrong in both directions here: identical chunks share one physical
//     object (object_hasher() excludes the chunk's offset and the unchanging-prefix rule charges an object once), so
//     a perfectly sound record's logical bytes may exceed the whole quota, and a quota reduction alone could then
//     turn a retained history into corruption;
//   * the stage count by the number of containers the manifest can encode;
//   * every per-element count (containers, checkpoints, K/V layers and streams) by what a manifest within the
//     configured manifest ceiling could encode at all - the manifest is read under that ceiling, so nothing wider
//     can be observed, and a reduction of any of these is a smaller parsing ceiling for *this* run, never a reason to
//     call a record this run did not write corrupt.
// The physical quota is enforced where it belongs: against the unique files the store's ledger charges, at
// admission.  No parsing ceiling is relaxed - the manifest ceiling itself stays exactly as configured and every count
// is still bounded by the bytes actually left in the file - and the derived set is never narrower than the configured
// one, so a run always keeps everything it could itself create.
ConversationStoreLimits retention_limits(const ConversationStoreLimits& configured) {
    ConversationStoreLimits kept = configured;
    // Every logical byte a manifest describes comes from a chunk it encodes, and within the configured manifest
    // ceiling there can be no more than that ceiling divided by kChunkEncodedMin chunks, each at most kMaxChunkBytes
    // long.  Their product bounds one stream's bytes, an image's whole payload and a record's token-id stream (an
    // int32 per token), independently of every physical figure.  Saturating multiplication: a ceiling wide enough to
    // overflow the product saturates to "no representable limit" rather than wrapping to a small one.
    const uint64_t chunks = configured.max_manifest_bytes / kChunkEncodedMin;
    const uint64_t widest = sat_mul(chunks, kMaxChunkBytes);
    kept.max_stream_bytes = std::max(configured.max_stream_bytes, widest);
    kept.max_state_bytes = std::max(configured.max_state_bytes, widest);
    kept.max_image_bytes = std::max(configured.max_image_bytes, widest);
    kept.max_tokens = std::max(configured.max_tokens, widest / uint64_t(sizeof(int32_t)));
    kept.max_images = std::max(configured.max_images, encoded_ceiling(configured.max_manifest_bytes, kContainerEncodedMin));
    kept.max_stages = std::max(configured.max_stages, kept.max_images);
    kept.max_checkpoints =
        std::max(configured.max_checkpoints, encoded_ceiling(configured.max_manifest_bytes, kCheckpointEncodedMin));
    kept.max_kv_layers =
        std::max(configured.max_kv_layers, encoded_ceiling(configured.max_manifest_bytes, kKvLayerEncodedMin));
    kept.max_streams = std::max(configured.max_streams, encoded_ceiling(configured.max_manifest_bytes, kStreamEncodedMin));
    return kept;
}

std::vector<ConversationStoreStreamInfo> manifest_stream_infos(const ManifestMeta& meta) {
    std::vector<ConversationStoreStreamInfo> out;
    out.reserve(meta.streams.size());
    for (size_t i = 0; i < meta.streams.size(); ++i) {
        ConversationStoreStreamInfo info;
        info.id = meta.streams[i].id;
        info.index = uint32_t(i);
        info.bytes = meta.streams[i].bytes;
        out.push_back(info);
    }
    return out;
}

// ---------------------------------------------------------------------------------------------------------------
// Root catalog: the only authority.  A corrupt catalog means cold caching; a directory never resurrects a history.
// ---------------------------------------------------------------------------------------------------------------

uint64_t catalog_encoded_size(const Catalog& catalog) {
    // tag, issuer digest, publication counter, the domain count and section, then the head count and section.
    uint64_t bytes = 8 + 32 + 8 + 4;
    bytes = sat_add(bytes, sat_mul(uint64_t(catalog.domains.size()), 40));
    bytes = sat_add(bytes, 4);
    // Per head: head id, manifest, bucket, domain index, epoch, generation, parent, payload, manifest and publication
    // sizes, stages, pin and protected flags, then the cut count and section.
    bytes = sat_add(bytes, sat_mul(uint64_t(catalog.heads.size()), 130 + 4));
    for (const auto& head : catalog.heads) bytes = sat_add(bytes, sat_mul(uint64_t(head.cuts.size()), 42));
    return sat_add(bytes, 32);
}

uint64_t catalog_workspace_bound(uint64_t encoded_bytes) {
    // The same derivation as a manifest's: one decoded head costs about 1.2x its encoded bytes plus one vector
    // header each.
    return sat_add(sat_mul(encoded_bytes, 2), sat_add(encoded_bytes / 4, 2 * kAllocSlackBytes));
}

std::vector<uint8_t> catalog_encode(const Catalog& catalog) {
    Writer w;
    w.reserve(size_t(catalog_encoded_size(catalog)));
    w.raw(kCatalogTag, sizeof kCatalogTag);
    w.digest(catalog.issuer_key_digest);
    w.u64(catalog.published_order_counter);
    w.u32(uint32_t(catalog.domains.size()));
    for (const auto& domain : catalog.domains) {
        w.digest(domain.digest);
        w.u64(domain.epoch);
    }
    w.u32(uint32_t(catalog.heads.size()));
    for (const auto& head : catalog.heads) {
        w.u64(head.head_id);
        w.digest(head.manifest_id);
        w.digest(head.bucket);
        w.u32(head.domain);
        w.u64(head.epoch);
        w.u64(head.generation);
        w.u64(head.parent);
        w.u64(head.payload_bytes);
        w.u64(head.manifest_bytes);
        w.u64(head.published_order);
        w.u32(head.stages);
        w.u8(head.pinned ? 1 : 0);
        w.u8(head.protected_seg ? 1 : 0);
        w.u32(uint32_t(head.cuts.size()));
        for (const auto& cut : head.cuts) {
            w.u64(cut.tokens);
            w.digest(cut.digest);
            w.u8(cut.live ? 1 : 0);
            w.u8(cut.pinned ? 1 : 0);
        }
    }
    const Digest trailer = sha256(w.data().data(), w.size());
    w.digest(trailer);
    return w.take();
}

bool catalog_decode(const uint8_t* bytes, size_t size, Catalog& out, std::string& why) {
    out = Catalog{};
    if (!bytes || size < sizeof kCatalogTag + kTrailer + 4) {
        why = "the catalog is too short";
        return false;
    }
    if (std::memcmp(bytes, kCatalogTag, sizeof kCatalogTag) != 0) {
        why = "the catalog does not start with the store's schema tag";
        return false;
    }
    Digest trailer{};
    std::memcpy(trailer.data(), bytes + size - kTrailer, kTrailer);
    if (sha256(bytes, size - kTrailer) != trailer) {
        why = "the catalog's checksum does not match its bytes";
        return false;
    }
    Reader r(bytes + sizeof kCatalogTag, size - sizeof kCatalogTag - kTrailer);
    if (!r.digest(out.issuer_key_digest) || !r.u64(out.published_order_counter)) {
        why = "the catalog's header is truncated";
        return false;
    }
    uint32_t domains = 0;
    if (!r.count(domains, kCatalogCeiling, 40)) {
        why = "the catalog's domain count is invalid";
        return false;
    }
    out.domains.reserve(domains);
    for (uint32_t i = 0; i < domains; ++i) {
        CatalogDomain domain;
        if (!r.digest(domain.digest) || !r.u64(domain.epoch)) {
            why = "a catalog domain is truncated";
            return false;
        }
        out.domains.push_back(domain);
    }
    uint32_t heads = 0;
    if (!r.count(heads, kCatalogCeiling, 106)) {
        why = "the catalog's head count is invalid";
        return false;
    }
    out.heads.reserve(heads);
    std::set<uint64_t> seen;
    for (uint32_t i = 0; i < heads; ++i) {
        CatalogHead head;
        uint8_t pinned = 0, protected_seg = 0;
        if (!r.u64(head.head_id) || !r.digest(head.manifest_id) || !r.digest(head.bucket) || !r.u32(head.domain) ||
            !r.u64(head.epoch) || !r.u64(head.generation) || !r.u64(head.parent) || !r.u64(head.payload_bytes) ||
            !r.u64(head.manifest_bytes) || !r.u64(head.published_order) || !r.u32(head.stages) || !r.u8(pinned) ||
            !r.u8(protected_seg)) {
            why = "a catalog head is truncated";
            return false;
        }
        head.pinned = pinned != 0;
        head.protected_seg = protected_seg != 0;
        if (head.head_id == 0 || head.domain >= out.domains.size()) {
            why = "a catalog head has no valid identity or names an unknown domain";
            return false;
        }
        if (!seen.insert(head.head_id).second) {
            why = "the catalog lists the same head twice";
            return false;
        }
        if (head.published_order > out.published_order_counter) {
            why = "a catalog head is newer than the catalog's publication counter";
            return false;
        }
        uint32_t cuts = 0;
        if (!r.count(cuts, kCatalogCeiling, 42)) {
            why = "a catalog head's cut count is invalid";
            return false;
        }
        head.cuts.reserve(cuts);
        for (uint32_t j = 0; j < cuts; ++j) {
            CutRecord cut;
            uint8_t live = 0, cut_pinned = 0;
            if (!r.u64(cut.tokens) || !r.digest(cut.digest) || !r.u8(live) || !r.u8(cut_pinned)) {
                why = "a catalog cut is truncated";
                return false;
            }
            cut.live = live != 0;
            cut.pinned = cut_pinned != 0;
            head.cuts.push_back(cut);
        }
        for (size_t j = 0; j < head.cuts.size(); ++j) {
            if (head.cuts[j].tokens == 0) {
                why = "a catalog cut has no token length";
                return false;
            }
            if (j && head.cuts[j - 1].tokens < head.cuts[j].tokens) {
                why = "a catalog head's cuts are not ordered longest first";
                return false;
            }
        }
        out.heads.push_back(std::move(head));
    }
    if (!r.empty()) {
        why = "the catalog holds trailing bytes after its last section";
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------------------------------------------
// Restore
// ---------------------------------------------------------------------------------------------------------------

bool restore_prepare(const ManifestMeta& meta, const ConversationStoreLimits& limits, SavedConversation& out,
                     std::vector<RestoreTarget>& targets, std::string& why) {
    if (!check_manifest(meta, limits, why)) return false;
    out = SavedConversation{};
    targets.clear();
    if (!bind_streams(meta, limits, out, targets, why)) return false;
    if (targets.size() != meta.streams.size()) {
        why = "the restore did not bind exactly the manifest's streams";
        return false;
    }
    for (size_t i = 0; i < targets.size(); ++i)
        if (targets[i].bytes != meta.streams[i].bytes) {
            why = "a restore target disagrees with its stream's length";
            return false;
        }
    return true;
}

void restore_finish(SavedConversation& out) { seal_tree(out); }

// ---------------------------------------------------------------------------------------------------------------
// Identity
// ---------------------------------------------------------------------------------------------------------------

Digest store_bucket_digest(const ConversationStoreOptions& options, const ConversationStoreSelector& selector) {
    Writer w;
    w.reserve(sizeof kBucketTag + 96 + 32 + 1 + 18 * 8 + 16);
    w.raw(kBucketTag, sizeof kBucketTag);
    w.digest(options.domain);
    w.digest(options.model);
    w.digest(options.engine);
    w.raw(selector.scope.value.data(), selector.scope.value.size());
    w.u8(selector.cvec ? 1 : 0);
    for (int64_t g : selector.geometry) w.i64(g);
    w.i64(selector.layer_lo);
    w.i64(selector.layer_hi);
    // The immutable chunk size is a storage detail: changing it must not orphan a single stored byte.
    return sha256(w.data().data(), w.size());
}

Digest object_scope_digest(uint16_t kind, const KvMeta* kv, const ConversationScope& authorization,
                           const Digest& model) {
    Sha256 hash;
    hash.update(kObjectScopeTag, sizeof kObjectScopeTag);
    hash.update(authorization.value.data(), authorization.value.size());
    hash.update(model.data(), model.size());
    hash.update(&kind, sizeof kind);
    if (kv) {
        const int64_t interpretation[] = {kv->format, kv->heads, kv->head_dim, kv->page_size, kv->idx_dim};
        hash.update(interpretation, sizeof interpretation);
    }
    return hash.digest();
}

namespace {
bool stream_kv_layout(const std::vector<ContainerMeta>& containers, const ConversationStoreStreamId& id,
                      const KvMeta*& kv, uint64_t& ordinal, std::string& why) {
    kv = nullptr;
    ordinal = id.ordinal;
    if (conversation_store_kv_kind(id.kind)) {
        const ContainerMeta* stage = nullptr;
        for (const auto& container : containers)
            if (container.is_image && container.stage == id.stage) {
                stage = &container;
                break;
            }
        if (!stage || id.ordinal >= stage->kv.size() || stage->geometry[1] <= 0 ||
            stage->geometry[2] <= 0 || stage->layer_lo < 0 || stage->layer_hi < stage->layer_lo ||
            stage->layer_hi > stage->geometry[1]) {
            why = "a K/V object has no valid model-layer interpretation";
            return false;
        }
        const int64_t interval = stage->geometry[2];
        const int64_t count = stage->geometry[1] / interval;
        const int64_t first = count ? std::min(stage->layer_lo / interval, count - 1) : 0;
        const uint64_t allocated = uint64_t(std::max<int64_t>(
            stage->layer_hi / interval - stage->layer_lo / interval, count ? 1 : 0));
        if (stage->kv.size() < allocated || stage->kv.size() - allocated > 1) {
            why = "a K/V object count disagrees with the captured attention layers";
            return false;
        }
        // The extra pool is the drafter, not the next main-model attention layer.
        ordinal = id.ordinal == allocated ? uint64_t{1} << 63 : uint64_t(first) + id.ordinal;
        kv = &stage->kv[size_t(id.ordinal)];
    }
    return true;
}
} // namespace

bool stream_object_identity(const std::vector<ContainerMeta>& containers, const ConversationStoreStreamId& id,
                            const Digest& model, Digest& scope, uint64_t& ordinal, std::string& why) {
    if (containers.empty() || !containers.front().scope.valid()) {
        why = "a physical object has no authorized conversation";
        return false;
    }
    const KvMeta* kv = nullptr;
    if (!stream_kv_layout(containers, id, kv, ordinal, why)) return false;
    scope = object_scope_digest(id.kind, kv, containers.front().scope, model);
    return true;
}

ConversationStoreReadComponent stream_read_component(uint16_t kind, uint64_t object_ordinal) {
    if (kind == kDraftBoundaryKind) return ConversationStoreReadComponent::draft;
    if (conversation_store_kv_kind(kind) && object_ordinal == (uint64_t{1} << 63))
        return ConversationStoreReadComponent::draft;
    switch (ConversationStoreStreamKind(kind)) {
        case ConversationStoreStreamKind::live_gdn:
        case ConversationStoreStreamKind::cp_gdn:
        case ConversationStoreStreamKind::part_gdn: return ConversationStoreReadComponent::recurrent;
        case ConversationStoreStreamKind::live_ple:
        case ConversationStoreStreamKind::cp_ple:
        case ConversationStoreStreamKind::part_ple: return ConversationStoreReadComponent::ple;
        case ConversationStoreStreamKind::live_tails:
        case ConversationStoreStreamKind::live_dead:
        case ConversationStoreStreamKind::live_block_pos:
        case ConversationStoreStreamKind::cp_tails:
        case ConversationStoreStreamKind::cp_dead:
        case ConversationStoreStreamKind::cp_block_pos:
        case ConversationStoreStreamKind::part_tails:
        case ConversationStoreStreamKind::part_dead:
        case ConversationStoreStreamKind::part_block_pos:
        case ConversationStoreStreamKind::kv_pooled: return ConversationStoreReadComponent::indexer;
        case ConversationStoreStreamKind::kv_k:
        case ConversationStoreStreamKind::kv_v:
        case ConversationStoreStreamKind::kv_k_scale:
        case ConversationStoreStreamKind::kv_v_scale: return ConversationStoreReadComponent::main_kv;
        default: return ConversationStoreReadComponent::metadata;
    }
}

bool restore_progress_totals(const ManifestMeta& meta, uint64_t encoded_bytes,
                             ConversationStoreReadProgress& out, std::string& why) {
    decltype(out.components) components{};
    components[size_t(ConversationStoreReadComponent::metadata)].total = encoded_bytes;
    for (const auto& stream : meta.streams) {
        const KvMeta* kv = nullptr;
        uint64_t ordinal = 0;
        if (!stream_kv_layout(meta.containers, stream.id, kv, ordinal, why)) return false;
        auto& component = components[size_t(stream_read_component(stream.id.kind, ordinal))];
        if (stream.bytes > UINT64_MAX - component.total) {
            why = "a restore component byte count overflows";
            return false;
        }
        component.total += stream.bytes;
    }
    components[size_t(ConversationStoreReadComponent::metadata)].acknowledged = encoded_bytes;
    out.components = components;
    out.totals_known = true;
    return true;
}

Sha256 object_hasher(const Digest& scope, uint16_t kind, uint64_t ordinal, uint64_t length) {
    Sha256 hash;
    hash.update(kObjectTag, sizeof kObjectTag);
    hash.update(scope.data(), scope.size());
    hash.update(&kind, sizeof kind);
    hash.update(&ordinal, sizeof ordinal);
    hash.update(&length, sizeof length);
    return hash;
}

Digest object_digest(const Digest& scope, uint16_t kind, uint64_t ordinal, uint64_t length, const uint8_t* bytes) {
    Sha256 hash = object_hasher(scope, kind, ordinal, length);
    if (bytes && length) hash.update(bytes, length);
    return hash.digest();
}

// ---------------------------------------------------------------------------------------------------------------
// Prefix chain
// ---------------------------------------------------------------------------------------------------------------

namespace {

void chain_tag(Sha256& hash, uint8_t tag) { hash.update(&tag, 1); }

void chain_u64(Sha256& hash, uint64_t value) {
    uint8_t bytes[8];
    for (int i = 0; i < 8; ++i) bytes[i] = uint8_t((value >> (8 * i)) & 0xff);
    hash.update(bytes, sizeof bytes);
}

// One run of the token prefix: a tag, its length, then the raw little-endian int32 elements.
void chain_ids(Sha256& hash, const int32_t* ids, size_t from, size_t to) {
    if (to <= from) return;
    chain_tag(hash, 'T');
    chain_u64(hash, uint64_t(to - from));
    hash.update(ids + from, (to - from) * sizeof(int32_t));
}

// The whole group of image keys that sits at one token position, in the caller's order.
void chain_images(Sha256& hash, const std::vector<ConversationImageKey>& images,
                  const std::vector<std::pair<uint64_t, size_t>>& ordered, size_t from, size_t to) {
    if (to <= from) return;
    chain_tag(hash, 'I');
    chain_u64(hash, uint64_t(to - from));
    for (size_t i = from; i < to; ++i) hash.update(&images[ordered[i].second], sizeof(ConversationImageKey));
}

Sha256 chain_seeded(const Digest& seed) {
    Sha256 hash;
    hash.update(kPrefixTag, sizeof kPrefixTag);
    hash.update(seed.data(), seed.size());
    return hash;
}

} // namespace

PrefixChain::PrefixChain(const Digest& seed, const int32_t* ids, size_t count,
                         const std::vector<ConversationImageKey>& images) {
    ids_ = ids;
    count_ = count;
    zero_ = chain_seeded(seed);
    // Images are keyed by the token position they follow; an image at or beyond the prefix cannot contribute, and a
    // negative start is treated as position 0 for grouping (its own value is still what the digest encodes).
    std::vector<std::pair<uint64_t, size_t>> ordered;
    ordered.reserve(images.size());
    for (size_t i = 0; i < images.size(); ++i) {
        const int64_t raw = images[i].start;
        const uint64_t start = raw <= 0 ? 0 : uint64_t(raw);
        if (start > count) continue;
        ordered.emplace_back(start, i);
    }
    std::stable_sort(ordered.begin(), ordered.end(),
                     [](const std::pair<uint64_t, size_t>& a, const std::pair<uint64_t, size_t>& b) {
                         return a.first < b.first;
                     });
    Sha256 state = zero_;
    size_t at = 0;
    size_t group = 0;
    states_.reserve(ordered.size());
    segment_begin_.reserve(ordered.size());
    while (group < ordered.size()) {
        const size_t start = size_t(ordered[group].first);
        chain_ids(state, ids_, at, start);
        at = start;
        size_t end = group;
        while (end < ordered.size() && ordered[end].first == ordered[group].first) ++end;
        chain_images(state, images, ordered, group, end);
        segment_begin_.push_back(start);
        states_.push_back(state);
        group = end;
    }
    tail_ = state;
    chain_ids(tail_, ids_, at, count_);
}

bool PrefixChain::digest(uint64_t n, Digest& out) const {
    if (n > count_) return false;
    if (n == count_) {
        out = tail_.digest();
        return true;
    }
    const size_t target = size_t(n);
    // The last image group strictly behind `n`: a group at exactly `n` is not part of a prefix of length `n`.
    size_t state = 0;
    while (state < segment_begin_.size() && segment_begin_[state] < target) ++state;
    Sha256 chain = state == 0 ? zero_ : states_[state - 1];
    const size_t at = state == 0 ? 0 : segment_begin_[state - 1];
    if (target > at && !ids_) return false;
    chain_ids(chain, ids_, at, target);
    out = chain.digest();
    return true;
}

} // namespace strata::core::conversation_store_detail

// ---------------------------------------------------------------------------------------------------------------
// The public seams the header exposes: the crypto a caller must not hand-roll, and the stream enumeration callers
// plan reuse against.
// ---------------------------------------------------------------------------------------------------------------

namespace strata::core {

std::array<uint8_t, 32> conversation_store_domain_digest(const void* bytes, size_t count) {
    static const uint8_t empty = 0;
    if (!bytes || count == 0) return conversation_store_detail::sha256(&empty, 0);
    return conversation_store_detail::sha256(bytes, count);
}

void ConversationStoreDigestBuilder::add_u64(uint64_t value) {
    for (int i = 0; i < 8; ++i) bytes_.push_back(uint8_t((value >> (8 * i)) & 0xff));
}

void ConversationStoreDigestBuilder::add_i64(int64_t value) { add_u64(uint64_t(value)); }

void ConversationStoreDigestBuilder::add_bytes(const void* data, size_t count) {
    add_u64(count);
    if (!data || count == 0) return;
    const uint8_t* p = static_cast<const uint8_t*>(data);
    bytes_.insert(bytes_.end(), p, p + count);
}

std::array<uint8_t, 32> ConversationStoreDigestBuilder::finish() const {
    return conversation_store_domain_digest(bytes_.data(), bytes_.size());
}

size_t ConversationStoreDigestBuilder::size() const { return bytes_.size(); }

std::vector<ConversationStoreStreamInfo> conversation_store_streams(const SavedConversation& image) {
    conversation_store_detail::ImageLayout layout;
    std::string why;
    if (!conversation_store_detail::image_layout(image, conversation_store_detail::Digest{},
                                                 conversation_store_detail::unconstrained_limits(), layout, why))
        return {};
    std::vector<ConversationStoreStreamInfo> out;
    out.reserve(layout.streams.size());
    for (const auto& stream : layout.streams) out.push_back(stream.info);
    return out;
}

} // namespace strata::core
