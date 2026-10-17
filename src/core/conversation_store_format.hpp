// On-disk encoding of the durable store's manifests and root catalog, the canonical stream enumeration of a
// SavedConversation tree, and the exact-prefix chain digests that make a lookup both cheap and exact.  Pure CPU
// byte/tree logic: no file I/O, no threads, no GPU, no global state, every count bounded before an allocation.
//
// The store's format is deliberately NOT the legacy v1 session file: v1 cannot represent stage images, pins, a
// checkpoint frontier, epochs or per-object chunk references (see docs/DETAILS.md, Session files).  What is reused
// from that path is the safety shape - explicit lengths, bounded parsing, a checksum that is verified before the
// payload is trusted, and a trailing marker so a truncated file cannot be mistaken for a complete one.
#pragma once

#include "strata/core/conversation_store.hpp"

#include "conversation_store_sha256.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace strata::core::conversation_store_detail {

// Reads `count` bytes of one stream at `offset` into `dst`; false with a reason when the range is outside the stream
// or the source itself fails.  Never allocates and never mutates the image.
using StreamRead = std::function<bool(uint64_t offset, uint8_t* dst, size_t count, std::string& why)>;
// Writes `count` bytes of one stream at `offset` from `src`; the restore's only way into a final buffer.
using StreamWrite = std::function<bool(uint64_t offset, const uint8_t* src, size_t count, std::string& why)>;

// One immutable chunk reference: [offset, offset + length) of a stream is the whole object `object`.
struct ChunkRecord {
    uint64_t offset = 0;
    uint32_t length = 0;
    Digest object{};
};

struct StreamRecord {
    ConversationStoreStreamId id;
    uint64_t bytes = 0;             // total bytes of the stream (== sum of its chunk lengths)
    std::vector<ChunkRecord> chunks;
};

// One ConversationCheckpoint's metadata.  Used for a container's live record, each of its checkpoints and each stage
// part (`parts` is only non-zero for a checkpoint that owns later stages).
struct RecordMeta {
    uint64_t tokens = 0;            // == ids.size()
    uint64_t used = 0;
    bool pinned = false;
    uint32_t parts = 0;             // stage parts owned by this record
};

struct KvMeta {
    int64_t format = 0, cells = 0, heads = 0, head_dim = 0, page_size = 0, pooled_rows = 0, idx_dim = 0;
    uint64_t part[5] = {0, 0, 0, 0, 0};   // k, v, k_scale, v_scale, pooled byte lengths
};

// One stage container: either a whole SavedConversation (`is_image`) or one record's stage part.
struct ContainerMeta {
    bool is_image = true;
    uint32_t stage = 0;             // 0 = the top image; k >= 1 = the k-th stage's device
    uint32_t parent_stage = 0;      // for a part: the container stage whose record owns it
    uint64_t parent_owner = 0;      // for a part: 0 = the owner's live record, else the owning checkpoint's key
    uint32_t part_index = 0;        // for a part: its position in the owning record's stage_parts
    // A part carries no carve of its own: it is taken from the stage image that declares that stage.  False means no
    // stage image declares it, so the manifest records that the part's layer range is unknown.  It is manifest
    // metadata only - physical payload identity deliberately ignores devices and carves.
    bool carve_known = true;
    ConversationScope scope;
    RecordMeta record;              // an image's live record, or the part's own record
    std::array<int64_t, 18> geometry{};
    int64_t layer_lo = 0, layer_hi = 0;
    bool cvec = true;
    std::vector<RecordMeta> checkpoints;
    std::vector<KvMeta> kv;
};

// One restorable token/image prefix of a whole image tree, with the exact prefix-chain digest that proves a query's
// tokens (and its image prefix) are that same prefix.
struct CutRecord {
    uint64_t tokens = 0;
    Digest digest{};
    bool live = false;
    bool pinned = false;
};

// A stream of an image plus the way to read its bytes back out of the (const) image.
struct StreamSource {
    ConversationStoreStreamInfo info;
    // 4 for the token-id stream, 16 for the image-key stream, 1 for opaque state bytes: the store aligns immutable
    // chunk boundaries to it, so an element is never split between two objects.
    uint32_t element = 1;
    StreamRead read;
};

struct ImageLayout {
    uint32_t stages = 0;
    uint64_t payload_bytes = 0;
    std::vector<ContainerMeta> containers;
    std::vector<StreamSource> streams;
    std::vector<CutRecord> cuts;
};

// Limits with no configured ceiling: for diagnostics and equality checks that must not refuse a legitimate image for
// being large.  A restore path never uses this - it is derived from the caller's admitted figures.
ConversationStoreLimits unconstrained_limits();

// The limits a *retained* record must satisfy: the figures under which a record this run did not write may still be
// kept on disk and indexed for the shared physical ledger.  Derived from the format's own representability bounds and
// the configured parsing ceilings, never from this run's admitted capacity and never from the physical quota:
// identical chunks share one physical object, so a sound record's logical bytes can exceed the whole quota, and a
// quota or profile reduction must never turn it into corruption.  The byte figures and the token ceiling come from
// the widest record a manifest within the configured manifest ceiling could declare (its encoded chunks times the
// widest chunk its own length field can carry, saturating rather than wrapping), the stage depth from the format's
// structural ceiling, and each per-element count from what such a manifest could encode at all.  The derived set is
// never narrower than `configured` (a run always keeps what it can itself create) and never wider than the format:
// no parsing ceiling is relaxed (the manifest ceiling stays exactly as configured and every count is still checked
// against the bytes left in the file).  Recovery, the index and the planner paths decode with these; every read and
// every publication is admitted against `configured` itself, and a record a run may not serve is refused there rather
// than discarded here.  The physical quota is enforced against the unique files the ledger charges, at admission.
ConversationStoreLimits retention_limits(const ConversationStoreLimits& configured);

// Canonical container/stream/cut enumeration of one image tree.  Fails (with a reason) on anything the format cannot
// represent exactly: a stage deeper than one level, a record's own parts having parts, duplicate stream identities,
// a malformed id/image byte count, a record missing one of its five state streams, or an exceeded limit.
bool image_layout(const SavedConversation& image, const Digest& seed, const ConversationStoreLimits& limits,
                  ImageLayout& out, std::string& why);

// ---------------------------------------------------------------------------------------------------------------
// Allocation-free counting: what an image will cost, and how big its encodings will be
//
// The writer charges its whole workspace before it allocates any of it, so the counts below are computed from the
// image tree/encoded sizes alone: no layout is built, no manifest is decoded and no vector is reserved to find out
// how much something would take.  Every figure is an over-estimate by construction (a container's own element plus
// the allocator's header and rounding); over-estimating is the safe direction, and no figure is ever derived from a
// configured maximum instead of the actual object.
// ---------------------------------------------------------------------------------------------------------------

// One small allocation's header and granularity.  A per-allocation figure, never a per-element one.
inline constexpr uint64_t kAllocSlackBytes = 64;
// One node of a std::map keyed by a 32-byte digest, including its bucket and the allocator's header.
inline constexpr uint64_t kMapNodeBytes = 128;
// One PrefixChain state: sizeof(Sha256) plus the EVP_MD_CTX and provider-side context a copy of it allocates
// (conversation_store_sha256.cpp: EVP_MD_CTX_new()/EVP_DigestInit_ex()).  A hard numerical bound for the provider's
// internals cannot be derived from this tree, so this is a fixed, deliberately generous allowance per image-key
// group rather than a silent omission.
inline constexpr uint64_t kChainBytesPerImageKey = 1024;

// Counts of the structures one image materializes.  No allocation, no I/O, no payload read.
struct ImageShape {
    uint64_t containers = 0;
    uint64_t records = 0;              // live records and checkpoints, across every container
    uint64_t kv_layers = 0;
    uint64_t streams = 0;
    uint64_t cuts = 0;                 // cut records the top image can advertise (duplicates merged later)
    uint64_t payload_bytes = 0;
    uint64_t max_stream_bytes = 0;     // the longest single stream: the largest chunk one read can produce
    uint64_t max_record_images = 0;    // the largest image-key list in one record: the PrefixChain's own scratch
};

// Counts what image_layout() would build, and refuses exactly what it would refuse (an unrepresentable stage depth,
// a scope-less image, an exceeded limit, an overflowing count).  Never allocates.
bool image_shape(const SavedConversation& image, const ConversationStoreLimits& limits, ImageShape& out,
                 std::string& why);

// The workspace (host bytes) the layout skeleton, the per-record prefix chains and the cut list need for `shape`.
uint64_t image_layout_workspace(const ImageShape& shape);

// The workspace of one source read of `bytes`: the chain over its own image keys.  Used by the read pipeline's own
// preparation charge as well as by the writer.
uint64_t prefix_chain_workspace(uint64_t image_keys);

// Immutable chunks one stream of `bytes` splits into at `chunk_limit`, element aligned: the count only, never the
// payload.  `element` is the stream's own alignment (4 for tokens, 16 for image keys, 1 for opaque bytes).
uint64_t stream_chunk_count(uint64_t bytes, uint32_t element, uint64_t chunk_limit);

// How many of a decoded donor stream's own chunks fall inside [offset, offset + bytes) of the same stream identity:
// the most chunks one adopted reuse interval can contribute to the destination stream.  Allocation-free.
uint64_t donor_chunks_in_interval(const StreamRecord& donor, uint64_t offset, uint64_t bytes);

struct ManifestMeta {
    Digest bucket{};                // the exactness namespace: domain, scope, cvec, geometry, layer carve
    Digest domain{};                // the execution-domain digest, copied from the options
    uint64_t epoch = 0;
    uint64_t generation = 0;        // capture ordering
    uint64_t parent = 0;            // the head this image was captured from (0 = none)
    uint64_t published_order = 0;   // store-assigned, monotonic
    uint32_t stages = 0;
    std::vector<ContainerMeta> containers;
    std::vector<StreamRecord> streams;                       // canonical order
    std::vector<std::pair<Digest, uint64_t>> objects;         // object id -> bytes, unique
    std::vector<CutRecord> cuts;                              // longest first; exact duplicates merged
    uint64_t payload_bytes = 0;
    Digest manifest_id{};                                     // == SHA-256 of the encoded bytes
};

// `meta.manifest_id` is not encoded; the caller computes it with manifest_id_of() over the encoded bytes.
std::vector<uint8_t> manifest_encode(const ManifestMeta& meta);
Digest manifest_id_of(const uint8_t* bytes, size_t size);
// Verifies the trailing digest and every bounded count before returning the decoded metadata.
bool manifest_decode(const uint8_t* bytes, size_t size, const ConversationStoreLimits& limits, ManifestMeta& out,
                     std::string& why);
// Whether already-decoded metadata satisfies these figures at all: the same structural and capacity predicate the
// decoder applies, exposed so metadata decoded under the wider retention figures can be classified against this run's
// admitted ones without decoding the file a second time.  A record this predicate refuses while the retention figures
// accept is one this run cannot serve: refused capacity, never a defect in the record.
bool manifest_within_limits(const ManifestMeta& meta, const ConversationStoreLimits& limits, std::string& why);
std::vector<ConversationStoreStreamInfo> manifest_stream_infos(const ManifestMeta& meta);

// The exact encoded size of a manifest with these counts, and of a manifest that already exists.  Exact, so the
// writer can charge the catalog/floor for a file before it encodes one byte, and reserve the encoder's own buffer
// once instead of reallocating it as it grows.
struct ManifestShape {
    uint64_t containers = 0;
    uint64_t container_records = 0;    // checkpoints, summed over the containers
    uint64_t kv_layers = 0;
    uint64_t streams = 0;
    uint64_t chunks = 0;
    uint64_t objects = 0;
    uint64_t cuts = 0;
};
uint64_t manifest_encoded_size_of(const ManifestShape& shape);
ManifestShape manifest_shape(const ManifestMeta& meta);
uint64_t manifest_encoded_size(const ManifestMeta& meta);

// The workspace one manifest decode needs beyond its own encoded bytes: the decoded metadata, its per-element
// vectors and the allocator's overhead.  Derived from the encoded size (whose every element is backed by at least
// one encoded byte), so it needs no decode to compute.
uint64_t manifest_workspace_bound(uint64_t encoded_bytes);

// Everything one manifest read and decode holds at once: the raw file, the decoded metadata
// (manifest_workspace_bound) and the two tables the consistency pass builds (one node per declared object and one
// per declared stream).  Counted from the size the read is capped at, so it needs neither a decode nor a configured
// maximum, and charged before a manifest is read or decoded (see ConversationStoreOptions::metadata_bytes).  A
// caller that has already reserved this figure for the same decode must not reserve it twice.
uint64_t manifest_read_workspace_bound(uint64_t encoded_bytes);

struct CatalogHead {
    uint64_t head_id = 0;
    Digest manifest_id{};           // the manifest file's content address (its name on disk)
    Digest bucket{};
    uint32_t domain = 0;            // index into Catalog::domains
    uint64_t epoch = 0;
    uint64_t generation = 0;
    uint64_t parent = 0;
    uint64_t payload_bytes = 0;
    uint64_t manifest_bytes = 0;
    uint64_t published_order = 0;
    uint32_t stages = 0;
    bool pinned = false;
    bool protected_seg = false;
    std::vector<CutRecord> cuts;
};

struct CatalogDomain {
    Digest digest{};
    uint64_t epoch = 0;
};

struct Catalog {
    Digest issuer_key_digest{};
    std::vector<CatalogDomain> domains;
    std::vector<CatalogHead> heads;
    uint64_t published_order_counter = 0;
};

std::vector<uint8_t> catalog_encode(const Catalog& catalog);
bool catalog_decode(const uint8_t* bytes, size_t size, Catalog& out, std::string& why);

// The exact encoded size of the given catalog: the same figure catalog_encode().size() returns, computed without
// allocating the encoding.  The store charges and reserves the catalog's next copy from this, so a measurement must
// never be the reason something is allocated.
uint64_t catalog_encoded_size(const Catalog& catalog);
// The workspace one catalog decode needs beyond the bytes it read, derived from the encoded size like a manifest's.
uint64_t catalog_workspace_bound(uint64_t encoded_bytes);

// Where one stream's payload bytes go.  Built by restore_prepare() for every stream, in the manifest's order.
struct RestoreTarget {
    uint64_t bytes = 0;
    StreamWrite write;
};

// Builds the complete image skeleton (every container, record, checkpoint, stage part and K/V part, with every
// buffer resized to its recorded length) and one writer per stream, so the payload can be streamed straight into the
// final buffers with no second copy.  Fails before any large allocation when the metadata is incomplete.
bool restore_prepare(const ManifestMeta& meta, const ConversationStoreLimits& limits, SavedConversation& out,
                     std::vector<RestoreTarget>& targets, std::string& why);
// Seals every restored CheckpointBytes: the handed-out image is read-only.
void restore_finish(SavedConversation& out);

bool images_equal(const SavedConversation& a, const SavedConversation& b, std::string& why);
uint64_t image_payload_bytes(const SavedConversation& image);

// The exactness namespace of one query (and of every manifest that may answer it).
Digest store_bucket_digest(const ConversationStoreOptions& options, const ConversationStoreSelector& selector);
// Content address of one immutable physical payload: its canonical interpretation, its state kind, its layer ordinal
// and its exact bytes.  Neither the execution domain nor a device or stage carve is part of it: those belong to the
// manifest's compatibility namespace, and excluding them is what lets two key layouts share one physical payload and
// one quota.  What must match exactly is the interpretation, so a K/V payload carries its whole layout (see
// object_scope_digest); token ids, image keys and running-state arrays are canonical by construction.
Digest object_digest(const Digest& scope, uint16_t kind, uint64_t ordinal, uint64_t length, const uint8_t* bytes);
Sha256 object_hasher(const Digest& scope, uint16_t kind, uint64_t ordinal, uint64_t length);

// Stable row interpretation and authorized model/conversation identity. Extent counts belong only in the
// manifest: appending cells must not rename an unchanged prefix chunk.
Digest object_scope_digest(uint16_t kind, const KvMeta* kv, const ConversationScope& authorization,
                           const Digest& model);
bool stream_object_identity(const std::vector<ContainerMeta>& containers, const ConversationStoreStreamId& id,
                            const Digest& model, Digest& scope, uint64_t& ordinal, std::string& why);

// Allocation-free restore diagnostics. Classification shares the physical object's draft-ordinal rule; totals
// include every declared logical stream once and the validated manifest bytes, never unique-object accounting.
ConversationStoreReadComponent stream_read_component(uint16_t kind, uint64_t object_ordinal);
bool restore_progress_totals(const ManifestMeta& meta, uint64_t encoded_bytes,
                             ConversationStoreReadProgress& out, std::string& why);

// Exact authorized-prefix digests of one token/image prefix, at the cut lengths a lookup needs.  Built once per
// record (publish) or per query (lookup), then asked for any prefix length: the chain states at the image positions
// make each answer O(the tokens since the last image), and the full prefix is already recorded.  The caller keeps the
// token vector alive for as long as the chain is used.
class PrefixChain {
public:
    PrefixChain(const Digest& seed, const int32_t* ids, size_t count,
                const std::vector<ConversationImageKey>& images);
    bool digest(uint64_t n, Digest& out) const;   // false when n > count

private:
    const int32_t* ids_ = nullptr;
    size_t count_ = 0;
    std::vector<size_t> segment_begin_;   // token index where each image group sits, ascending
    std::vector<Sha256> states_;          // the chain after ids[0, begin_k) and the images at begin_k
    Sha256 tail_;                         // the chain of the whole prefix (seed, ids, every image)
    Sha256 zero_;                         // the chain of the seed alone
};

} // namespace strata::core::conversation_store_detail
