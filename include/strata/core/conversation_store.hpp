// Durable, multi-tenant NVMe backing for --serve's parked conversations.  CPU-only: this module owns the disk
// authority (catalog, epochs, immutable objects, manifests, quota), the storage threads and every byte it hands
// back.  It never touches GPU state, never calls a callback that could, and never reads a live session.
//
// Shape of one root directory (created 0700, files 0600):
//   owner.lock      exclusive, non-blocking; a second owner serves without persistent caching
//   issuer.key      the private 32-byte issuer key the frontend derives its authorized scope from
//   catalog         the authoritative, atomically replaced root catalog: domains, epochs, run intent, heads
//   intent-<hex>    one durably created run-intent marker per execution domain (see "Recovery")
//   objects/xx/...  immutable, content-addressed payload chunks (default 1 MiB)
//   manifests/...   immutable complete manifests (payload layout + the whole conversation tree)
//   tmp/            files being written; anything left there by a crash is an accounted orphan
//
// Only the catalog is authority.  Files that are not catalog-authorized are accounted and then deleted; a corrupt
// catalog means cold caching, never a directory-based resurrection.  A manifest describes one complete image: the
// live record, every retained checkpoint (root/latest/rewinds) with its exact pin flags, every stage's own running
// state, every K/V layer part (main layers and the draft) and the exact token/image prefixes the image was captured
// at.  Bytes are reused across turns only through explicit, caller-proven unchanged byte intervals, so an
// immutable chunk that already exists is never re-read or re-hashed.
//
// Threading: one admitted writer thread and a bounded read pipeline (Options::max_reads incoming images).  Nothing in
// this module blocks under a caller's admission lock: a caller that wants a restore at its own safe seam calls
// wait_ready(), and every other entry point returns immediately - including confirmed_restore() and
// invalidate_domain(), whose durable catalog commits are handed to the storage thread (invalidate_domain then waits
// for that commit without holding the catalog mutex).  No storage wait ever runs while the module holds its own
// catalog mutex, and no filesystem call is ever made on a path component the module has not opened itself.
#pragma once

#include "strata/core/conversation_cache.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace strata::core {

// ---------------------------------------------------------------------------------------------------------------
// Payload streams
//
// An image is serialized as an ordered set of named byte streams.  The identity below is stable across turns, so
// the same logical stream can be named in a reuse request against an earlier manifest.
// ---------------------------------------------------------------------------------------------------------------

// Kinds are disjoint by role, so one stream identity can never be read back into the wrong field: `ids`/`live_*`
// belong to a stage container's live record, `cp_*` to one of its checkpoints and `part_*` to a stage part of
// either (a checkpoint's own later-stage running state).  A stage part has no K/V of its own.
enum class ConversationStoreStreamKind : uint16_t {
    ids = 1,          // the live record's token ids, little-endian int32
    imgs = 2,         // the live record's ConversationImageKey list, little-endian
    live_gdn = 3,
    live_ple = 4,
    live_tails = 5,
    live_dead = 6,
    live_block_pos = 7,
    cp_ids = 8,
    cp_imgs = 9,
    cp_gdn = 10,
    cp_ple = 11,
    cp_tails = 12,
    cp_dead = 13,
    cp_block_pos = 14,
    part_ids = 15,
    part_imgs = 16,
    part_gdn = 17,
    part_ple = 18,
    part_tails = 19,
    part_dead = 20,
    part_block_pos = 21,
    kv_k = 22,
    kv_v = 23,
    kv_k_scale = 24,
    kv_v_scale = 25,
    kv_pooled = 26,
};

inline bool conversation_store_state_kind(uint16_t kind) {
    return (kind >= uint16_t(ConversationStoreStreamKind::live_gdn) &&
            kind <= uint16_t(ConversationStoreStreamKind::live_block_pos)) ||
           (kind >= uint16_t(ConversationStoreStreamKind::cp_gdn) &&
            kind <= uint16_t(ConversationStoreStreamKind::cp_block_pos)) ||
           (kind >= uint16_t(ConversationStoreStreamKind::part_gdn) &&
            kind <= uint16_t(ConversationStoreStreamKind::part_block_pos));
}
inline bool conversation_store_kv_kind(uint16_t kind) {
    return kind >= uint16_t(ConversationStoreStreamKind::kv_k) &&
           kind <= uint16_t(ConversationStoreStreamKind::kv_pooled);
}
inline bool conversation_store_part_kind(uint16_t kind) {
    return kind >= uint16_t(ConversationStoreStreamKind::part_ids) &&
           kind <= uint16_t(ConversationStoreStreamKind::part_block_pos);
}

// `stage` 0 is the image (the top container); stage k >= 1 is that container's k-th stage: the chunk of the image
// whose state lives on device k, either a SavedConversation::stage_images[k - 1] subtree or the stage part of a
// checkpoint/live record.  `owner` names the record a `part_*` stream belongs to: 0 for the container's live record,
// the checkpoint's cut (token count) for a checkpoint.  `ordinal` distinguishes the several streams of one kind: the
// K/V layer index (kv_*), the image index (imgs) or 0.
struct ConversationStoreStreamId {
    uint32_t stage = 0;
    uint16_t kind = 0;
    uint64_t owner = 0;
    uint64_t ordinal = 0;
    bool operator==(const ConversationStoreStreamId&) const = default;
};

struct ConversationStoreStreamInfo {
    ConversationStoreStreamId id;
    uint32_t index = 0;   // position in the canonical serialization order of the image
    uint64_t bytes = 0;
};

// The canonical, deterministic stream order of one image, exactly as the store serializes it.  Returns empty for an
// image the format cannot represent (deeper than one stage level, duplicate stream identities, non-multiple-of-four
// id or image byte counts, a kv part count that is not one of the five kinds).
std::vector<ConversationStoreStreamInfo> conversation_store_streams(const SavedConversation& image);

// ---------------------------------------------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------------------------------------------

struct ConversationStoreOptions {
    // Private directory on the cache device; empty disables persistence.  It must be an absolute path owned by this
    // account and not writable by group or others: a missing component is created 0700 with its parent flushed, an
    // existing component or leaf that is a symlink, another account's or group/other-writable is refused, and every
    // later access is descriptor-relative and never follows a symlink.  The issuer key is created only for a
    // genuinely new, empty store: an existing store that has lost its key disables persistence instead of silently
    // rotating the identity its old records were authorized under.
    std::string root;
    // Verified identities.  `domain` is the strong execution-domain digest (state format, kernel contract, resolved
    // layout/arithmetic/draft).  `model` is the verified common model identity and `engine` the verified engine
    // identity; both must come from release data, never a sampled fingerprint, alias or profile name.  A zero value
    // disables persistence safely instead of authorizing the wrong records.
    std::array<uint8_t, 32> domain{};
    std::array<uint8_t, 32> model{};
    std::array<uint8_t, 32> engine{};
    // The quota covers *everything* this module owns on the device: objects, manifests, the catalog and its
    // temporary, the issuer key, the run-intent marker and any file awaiting deletion.  Every byte is charged at the
    // conservative allocated size (block-rounded, plus one block per file for its directory entry), and a charge is
    // credited only once the file is confirmed gone.  Nothing is ever admitted against a promise to delete later.
    uint64_t quota_bytes = uint64_t(128) << 30;    // one shared physical quota for every layout
    // Filesystem floor, resampled before every reservation.  A failed or zero sample refuses the reservation: an
    // unreadable filesystem is never treated as an empty one.
    uint64_t min_free_bytes = uint64_t(128) << 30;
    // Hard ceiling on this module's in-RAM index *and* preparation: no job descriptor, token/image copy or index
    // entry is allocated that would cross it.
    uint64_t metadata_bytes = uint64_t(64) << 20;
    // Hard retained-head ceiling: a publication that cannot obtain a slot (every head pinned or leased) is refused
    // instead of overrunning it.  The protected segment is capped at three quarters of it.
    size_t head_limit = 128;
    uint64_t chunk_bytes = uint64_t(1) << 20;      // immutable object size (the tail may be shorter)
    // One materialized incoming image at a time by default.  The slot is held from the moment a read starts
    // allocating *through the whole lease* of the image it produced, so `max_reads` bounds concurrent incoming
    // images, not merely concurrently running workers.  Read *descriptors* are bounded separately by
    // `max_read_requests`, and an in-flight prefetch's speculative resources are revoked when a demand needs them.
    uint32_t max_reads = 1;
    // Bounded read identifiers (demand and prefetch, including misses).  A request over this bound is refused
    // *before* any descriptor is allocated.
    uint32_t max_read_requests = 16;
    // Extra *same-lineage* publications that may replace the one in flight (see publish()).  Never a second
    // admitted image for a different branch.
    uint32_t max_pending_publishes = 1;
    // 0: no read allowance has been published yet.  No read materializes and a speculative read is refused, and no
    // publication is admitted either: reads and publications are both admitted against this one envelope, and a
    // publication must fit it as its image *plus* the writer workspace it needs (see publish()).  There is no
    // admission path that skips the check, and publish_read_allowance() is what makes both absolute.
    uint64_t read_budget_bytes = 0;
};

// ---------------------------------------------------------------------------------------------------------------
// Identity digests
//
// Vetted SHA-256 (OpenSSL EVP) for the identity fields above.  The caller owns the canonical field selection and
// encoding (the module cannot know which engine fields define a domain); the module owns the crypto, so no caller
// hand-rolls a hash.  `Options::domain` is the digest of the *resolved* execution configuration; `model` and
// `engine` stay the verified identity flags.  The store mixes all three into every bucket, manifest and object
// identity, so a record can only ever be found by a caller that presents the same three.
// ---------------------------------------------------------------------------------------------------------------

std::array<uint8_t, 32> conversation_store_domain_digest(const void* bytes, size_t count);

// The same digest over a canonical, length-delimited encoding, so two different field splits can never hash to the
// same byte sequence: every integer is 8 bytes little-endian, every byte string is a u64 length then its bytes.
// `finish()` is idempotent and never mutates the builder.
class ConversationStoreDigestBuilder {
public:
    void add_u64(uint64_t value);
    void add_i64(int64_t value);
    void add_bytes(const void* data, size_t count);
    void add_str(const std::string& text) { add_bytes(text.data(), text.size()); }
    std::array<uint8_t, 32> finish() const;
    size_t size() const;

private:
    std::vector<uint8_t> bytes_;
};

// What the caller knows about an image before it is parsed.  Every field is checked before any allocation, and a
// manifest that exceeds one of these is rejected as invalid rather than truncated or partially restored.  The
// defaults are deliberately open enough not to become the next bottleneck, and the owner MUST override them from its
// own admitted runtime figures: max_tokens from the resolved context, max_kv_layers from the layer count (+ the
// draft), max_image_bytes/max_stream_bytes from the admitted image budget, max_checkpoints from the checkpoint
// policy.  Nothing here promises more than that admitted budget - it only stops an incidental cap from silently
// refusing a long admitted image.
//
// These figures describe *this run's* admitted capacity.  They decide whether this run may create or serve a record -
// never whether an existing record is kept.  A store root is shared by every profile that opens it, so records
// captured under other figures (another stage count, another checkpoint policy, another image budget) are retained,
// indexed and left on disk; a run that cannot serve one refuses the read or the publication (refused_limits) instead
// of treating it as corrupt - including when only `max_state_bytes` is smaller than the record's state, which is a
// capacity figure for a state stream exactly as it is for a K/V part.  The retention figures are derived by the store
// from the format's own representability bounds and these configured ceilings - never from the physical quota, since
// identical chunks share one physical object and a sound record's logical bytes can therefore exceed the whole quota -
// so this contract cannot be widened or narrowed by an owner by accident, and a change of quota or profile alone
// never discards a valid record.  A record this run cannot even read under its own parsing ceiling or preparation
// budget is refused by declining to serve - recovery disables persistence (the caller's cold path) and leaves the
// catalog, every file and every charge untouched - rather than being deleted or labelled corrupt.
struct ConversationStoreLimits {
    uint64_t max_image_bytes = uint64_t(16) << 30;
    uint64_t max_stream_bytes = uint64_t(8) << 30;
    uint64_t max_manifest_bytes = uint64_t(64) << 20;
    uint64_t max_tokens = uint64_t(1) << 20;
    uint64_t max_state_bytes = uint64_t(4) << 30;
    uint32_t max_checkpoints = 256;
    uint32_t max_kv_layers = 4096;
    uint32_t max_streams = 65536;
    uint32_t max_stages = 8;
    uint32_t max_images = 1048576;
};

// ---------------------------------------------------------------------------------------------------------------
// Lookup
// ---------------------------------------------------------------------------------------------------------------

// Everything that must be equal for two images to describe the same state: the authorized scope, the steering mode,
// the resolved geometry and the session's layer carve.  Compatibility namespaces never partition capacity.
struct ConversationStoreSelector {
    ConversationScope scope;
    bool cvec = true;
    std::array<int64_t, 18> geometry{};
    int64_t layer_lo = 0, layer_hi = 0;
};

// An exact, authorized prefix the store can restore: `tokens` is the length of the matched cut (the image's live
// record or one of its checkpoints).  Produced from catalog metadata only - no payload is read and no GPU pointer,
// image or buffer is exposed.
struct ConversationStoreMatch {
    uint64_t head_id = 0;
    int64_t tokens = 0;
    bool live = false;
    bool pinned = false;
    uint32_t stages = 0;
    uint64_t payload_bytes = 0;
    // The manifest's share of the shared physical charge: its encoded bytes rounded up to the filesystem's
    // allocation granularity, plus one block for its directory entry - never a smaller figure than the file occupies.
    uint64_t manifest_bytes = 0;
    uint64_t published_order = 0;
};

struct ConversationStoreQuery {
    ConversationStoreSelector selector;
    std::vector<int32_t> tokens;                  // the prepared request's exact prompt tokens
    std::vector<ConversationImageKey> images;
    uint64_t request_id = 0;
    uint64_t generation = 0;
    uint64_t deadline_ns = 0;                     // original request deadline, monotonic; 0 = none
};

// ---------------------------------------------------------------------------------------------------------------
// Reads
// ---------------------------------------------------------------------------------------------------------------

enum class ConversationStoreReadStatus : uint8_t {
    none = 0,     // request_status(): still queued or being materialized
    ready,        // a complete, verified image is attached
    missing,      // no eligible stored prefix (an ordinary cache miss)
    corrupt,      // an eligible record failed validation: treat as a miss, never restore it
    cancelled,    // the caller cancelled it, or a newer one replaced it
    stale,        // the execution-domain epoch advanced before it completed
    refused_budget,   // the read allowance did not admit it (prefetch only)
    refused_quota,    // the store is full: skip storage, keep serving
    refused_limits,   // the record exceeds ConversationStoreLimits: refused, never discarded (it may belong to another profile)
    failed,       // I/O or allocation failure
    disabled,     // persistence is off; this is always a miss
};

const char* conversation_store_read_status_name(ConversationStoreReadStatus status);

struct ConversationStoreRead {
    ConversationStoreReadStatus status = ConversationStoreReadStatus::none;
    uint64_t request_id = 0;
    uint64_t generation = 0;
    uint64_t head_id = 0;
    // The execution-domain epoch this read was admitted in.  Never relabelled: a completion of an older epoch is
    // retired rather than delivered under the current one.
    uint64_t epoch = 0;
    int64_t tokens = 0;
    // True when the selected cut is the image's live record and false when it is a checkpoint: the caller needs this
    // to know which part of the restored image the cut describes.
    bool live = false;
    uint32_t stages = 0;
    uint64_t payload_bytes = 0;
    // Complete, byte-verified image: the live record, every checkpoint with its pins, every stage and every K/V
    // layer, exactly as it was captured.  Keep this alias instead of copying the image: a SavedConversation copy
    // deep-copies every payload buffer, while an alias of this shared_ptr shares them.  The image's RAM stays
    // charged to the store until release_read().
    std::shared_ptr<const SavedConversation> image;
    std::string detail;   // bounded human-readable reason; never contains key bytes or credentials
};

// ---------------------------------------------------------------------------------------------------------------
// Publication
// ---------------------------------------------------------------------------------------------------------------

// A caller-proven unchanged byte interval of one payload stream: [source_offset, source_offset + bytes) of the same
// stream identity in `head_id` is byte-identical to [offset, offset + bytes) of the outgoing image's stream.  Only
// supply an interval the engine has actually proven unchanged (conversation_snapshot's retained/reused extents);
// equal token ids alone do not qualify.  Reused intervals must end on a chunk boundary of both streams (or end both
// streams), so no immutable chunk is ever partially rewritten.
//
// `head_id` is *the* donor of the interval and is resolved independently of Publish::supersedes: the store reads
// that head's own manifest, requires it to be currently authorized, to be in this image's own compatibility bucket
// and to have been captured in this epoch, and adopts whole aligned chunks only when its stream identity maps to the
// same physical identity as the outgoing image's.  An unverified or differently mapped donor is ignored, never
// guessed at, and every donor is protected from eviction while the publication is in flight.
struct ConversationStoreReuse {
    uint64_t head_id = 0;
    ConversationStoreStreamId stream;
    uint64_t source_offset = 0;
    uint64_t offset = 0;
    uint64_t bytes = 0;
};

struct ConversationStorePublish {
    uint64_t request_id = 0;
    uint64_t generation = 0;                 // capture ordering, not worker completion
    // The execution-domain epoch (epoch()) sampled at the *actual capture seam*: the store refuses a publication
    // whose capture predates an epoch advance, so an image captured before a safety invalidation can never be
    // authorized afterwards.  A production caller MUST set this explicitly; 0 means "bind the epoch at publish()
    // admission" and exists only for admission-only workflows and tests.
    uint64_t epoch = 0;
    uint64_t supersedes = 0;                 // the head this image was captured from (0 = a new branch)
    bool cover = false;                      // the caller proved it covers `supersedes`' compatible prefix
    // The bytes of *this image's* lease that the caller has already accounted for outside this store's published
    // allowance, because a shared cache handed the entry over while still holding a lease for exactly those bytes.
    // The store fits this one publication against `ram_read_budget() + external_lease_bytes - what it already holds`;
    // it never grants the credit to read admission, and it still charges the full image *and* its whole counted
    // writer workspace, so a transferred lease can never become a second shared allowance.
    // Caller contract, decided at *admission* (see ConversationStorePublishAdmission): an `accepted` image transfers
    // its accounting to this store at that instant, so the caller MUST release its matching lease and republish the
    // allowance straight away - whatever the attempt later settles as, because the store charges the whole image from
    // then on.  A `deferred` or `refused` image transfers nothing: the caller MUST keep both its image and its lease,
    // and a deferred one may be offered again unchanged at a later seam.  The store clamps the figure to the image's
    // own size, so a caller can never claim more credit than the bytes it is handing over, and a publication whose
    // actual counted peak does not fit is refused before any payload is allocated.
    uint64_t external_lease_bytes = 0;
    std::vector<ConversationStoreReuse> reuse;
};

enum class ConversationStorePublishStatus : uint8_t {
    stored = 0,             // a new branch head
    replaced,               // superseded its predecessor in the same branch and epoch
    retained_branch,        // the predecessor keeps a pinned/divergent cut, so both heads are retained
    stale,                  // an older generation, a different epoch, a retired predecessor or a cancelled ticket
    refused_quota,
    refused_floor,
    // The image exceeds ConversationStoreLimits, the store's index or head ceiling is reached, or the publication
    // image does not fit the published RAM allowance (see publish()).
    refused_limits,
    refused_busy,           // the bounded writer and its pending slots are full
    coalesced,              // replaced by a newer covered publication of the same lineage (lease released)
    failed,
    disabled,
};

const char* conversation_store_publish_status_name(ConversationStorePublishStatus status);

// What one publish() call decided about the caller's image and its lease.  The split is the difference between
// resources that are *occupied* and a shape that is *over cap*:
//   accepted  the counted peak fits an envelope whose only occupants are work this store may not free, so the image
//             was admitted: the store owns the image's accounting from this instant (the caller releases its lease),
//             and the eventual outcome of the attempt is reported by take_publish()/wait_publish().
//   deferred  the counted peak fits the allowance and the preparation ceiling on its own, but resources that are
//             occupied right now - an active reader's reservation, a handed-out or demanded image's lease, a
//             publication in flight - keep it out, and nothing this store may free is left to revoke.  Nothing was
//             allocated, charged or transferred: the caller keeps its image and its lease and may offer the same
//             publication again later.  A deferral is not an attempt.
//   refused   the image is refused on its own terms: it is not representable under this profile's figures, its
//             counted peak (image plus writer workspace) can never fit the published allowance or the preparation
//             ceiling, another branch is already admitted, or this store is not serving.  Offering the same image
//             again cannot change that.
enum class ConversationStorePublishAdmission : uint8_t { accepted = 0, deferred = 1, refused = 2 };

const char* conversation_store_publish_admission_name(ConversationStorePublishAdmission admission);

struct ConversationStorePublishResult {
    ConversationStorePublishStatus status = ConversationStorePublishStatus::disabled;
    uint64_t request_id = 0;
    uint64_t generation = 0;
    uint64_t head_id = 0;
    uint64_t epoch = 0;
    uint64_t superseded = 0;
    uint32_t stages = 0;
    uint64_t payload_bytes = 0;
    uint64_t new_bytes = 0;      // objects actually written by this publication
    uint64_t reused_bytes = 0;   // immutable chunks taken from `reuse`
    bool released = true;        // the store dropped its shared_ptr<const SavedConversation> alias
    std::string detail;
};

// ---------------------------------------------------------------------------------------------------------------
// Accounting
// ---------------------------------------------------------------------------------------------------------------

struct ConversationStoreCounters {
    uint64_t objects = 0, object_bytes = 0, shared_objects = 0;
    uint64_t manifests = 0, manifest_bytes = 0, heads = 0, pinned_heads = 0;
    uint64_t published = 0, replaced = 0, retained_branches = 0, coalesced_publishes = 0;
    uint64_t published_chunks = 0, reused_chunks = 0, reused_bytes = 0, dedup_chunks = 0;
    uint64_t matches = 0, misses = 0, reads_served = 0, reads_cancelled = 0, reads_stale = 0;
    uint64_t reads_handed_out = 0, reads_released = 0;
    uint64_t reads_corrupt = 0, reads_failed = 0, refused_budget = 0, refused_quota = 0, refused_limits = 0;
    uint64_t evictions = 0, reclaimed_objects = 0, reclaimed_bytes = 0, deferred_reclaims = 0;
    uint64_t bytes_written = 0, bytes_read = 0, peak_bytes = 0;
    uint64_t orphan_bytes = 0, pending_delete_bytes = 0, corrupt_records = 0, epoch_advances = 0;
    uint64_t refusals_stale = 0, refusals_busy = 0, refusals_floor = 0, refusals_quota = 0;
    uint64_t deferred_publishes = 0;   // admissions held back by occupied resources (see publish())
    uint64_t lock_conflicts = 0, aborted_publications = 0, recovered_orphans = 0;
};

// Cheap catalog metadata of one retained head (for diagnostics and for planning a reuse interval).
struct ConversationStoreManifestInfo {
    uint64_t head_id = 0;
    uint64_t epoch = 0;
    uint64_t generation = 0;
    uint64_t parent_head_id = 0;
    uint64_t payload_bytes = 0;
    // The manifest's share of the shared charge (allocation-rounded), never a smaller figure than the file occupies.
    uint64_t manifest_bytes = 0;
    uint64_t published_order = 0;
    uint32_t stages = 0;
    bool pinned = false;
    std::vector<int64_t> cuts;   // every cut this image can restore (tokens), longest first
};

// ---------------------------------------------------------------------------------------------------------------
// The store
// ---------------------------------------------------------------------------------------------------------------

class ConversationStore {
public:
    explicit ConversationStore(ConversationStoreOptions options, ConversationStoreLimits limits = {});
    ~ConversationStore();
    ConversationStore(const ConversationStore&) = delete;
    ConversationStore& operator=(const ConversationStore&) = delete;

    // Persistence is on only when the root opened, the exclusive lock was taken, the issuer key and catalog are
    // sound and every required identity is present.  A disabled store serves nothing and stores nothing: every
    // lookup is a miss and every publication is refused.  `disabled_reason()` is empty when enabled.
    bool enabled() const;
    const char* disabled_reason() const;
    // Enabled and not faulted.  A fault is *latched* for the life of this instance: a failure that must stop reuse (a
    // safety fault, an authority write that cannot be trusted) is never cleared, and a faulted instance never clears
    // its run-intent marker, never writes the catalog, and never looks like a clean close.
    bool healthy() const;
    void set_unhealthy(const char* reason);      // the owner observed a failure that must stop reuse (e.g. a safety fault)

    // The execution domain's current epoch, the issuer key the frontend derives authorized scopes from, and the
    // durable run-intent marker this owner holds.  The key is 32 bytes of cryptographic entropy and is immutable
    // after a successful open: an existing store whose key cannot be read or is not exactly 32 bytes stays disabled
    // rather than minting a new authorization identity.
    uint64_t epoch() const;
    const std::array<uint8_t, 32>& issuer_key() const;
    bool run_intent_active() const;
    uint64_t domain_digest_truncated() const;
    // The durable publication-ordering floor of this store's own execution domain: the greatest generation the
    // catalog records for a head of this domain, across every head it has not retired yet.  A cover is admitted only
    // when its generation is *strictly* newer than the predecessor head's, and that predecessor's figure was written
    // by the process that captured it - which may have been an earlier one.  A caller that mints its own generations
    // (Publish::generation) must therefore start a fresh process above this value and mint upward from there: a
    // counter restarting at zero would name a figure this domain has already used, and its cover of the head it just
    // restored would be refused as a stale completion.  The ordering is per domain because the store compares a cover
    // only with a predecessor of this domain and epoch, and it is a floor rather than a reservation: a refused
    // publication leaves the value behind the numbers its caller minted, which is harmless while the caller keeps
    // minting upward.  Cheap, read-only; 0 when this domain retains no head (an instance that never read a catalog
    // has none to report).
    uint64_t publication_generation_floor() const;

    // --- lookup (read-only, cheap, CPU only) ---------------------------------------------------------------
    // Exact authorized cuts for one selector, longest first and, within equal lengths, newest first.  Never exposes
    // a payload, an image or a GPU pointer.
    void for_each_match(const ConversationStoreSelector& selector, const int32_t* ids, size_t count,
                        const ConversationImageKey* images, size_t image_count,
                        const std::function<void(const ConversationStoreMatch&)>& visit) const;
    template<class Visitor>
    void for_each_match(const ConversationStoreSelector& selector, const std::vector<int32_t>& ids,
                        const std::vector<ConversationImageKey>& images, Visitor&& visit) const {
        for_each_match(selector, ids.data(), ids.size(), images.data(), images.size(),
                       std::function<void(const ConversationStoreMatch&)>(std::forward<Visitor>(visit)));
    }
    std::vector<ConversationStoreMatch> matches(const ConversationStoreSelector& selector,
                                               const std::vector<int32_t>& ids,
                                               const std::vector<ConversationImageKey>& images) const;
    bool manifest_info(uint64_t head_id, ConversationStoreManifestInfo& out) const;
    // The stream table of a retained head, read from its manifest (bounded, read-only, no catalog lock held): for
    // planning reuse intervals.  It exposes no payload and authorizes nothing.  Decoded under the store's retention
    // figures, so a head another profile captured is still readable as long as it is structurally sound.
    bool manifest_streams(uint64_t head_id, std::vector<ConversationStoreStreamInfo>& out) const;

    // --- reads --------------------------------------------------------------------------------------------
    // demand: required by a request that will consume a slot; admit its whole peak against the published allowance
    //   *together with every other byte the store holds* - the index, the live descriptors' own copies, the admitted
    //   reads' decode preparation and any diagnostic decode - or, when no allowance is published yet, let it wait.
    //   When it can never fit it is refused immediately as refused_budget, so the caller falls back to its ordinary
    //   prefill.
    // prefetch: optional, only admitted while spare allowance exists, never overtaking a demand, and refused (never
    //   left pending) when the allowance has no room for it.
    // Both select the longest exact authorized prefix themselves, deduplicate by (request_id, generation), enforce
    // the read-descriptor and preparation ceilings *before* allocating anything, bind the request to the epoch it was
    // admitted in, and return immediately; the caller consumes the result with take_ready()/wait_ready() at its own
    // safe seam.  A demand never revokes anything of its own; it does revoke outstanding speculation.
    bool demand(const ConversationStoreQuery& query, std::string& why);
    bool prefetch(const ConversationStoreQuery& query, std::string& why);
    // Explicit read cancellation leaves a terminal verdict until its owner calls release_read().
    // Issued filesystem work may finish, but no cancelled image is delivered.
    void cancel_read(uint64_t request_id, uint64_t generation);
    // Withdraw optional work only. Once demand adopts a read, the advisory channel cannot cancel
    // or release it. Unadopted descriptors are retired; a later demand may submit a fresh query.
    void withdraw_prefetch(uint64_t request_id, uint64_t generation);
    // Non-blocking: false when nothing is ready.  A returned image is complete and byte-verified, and it was
    // materialized in the epoch this store still serves: a completion of an older epoch, or one produced after the
    // store faulted or closed, is never delivered and never relabelled.  The caller receives the only alias of an
    // image whose RAM stays charged to this module (see ram_lease_bytes()) until it calls release_read(); nothing is
    // refunded while the caller still holds bytes.  The store does not drop its accounting when the returned
    // shared_ptr is destroyed, because an alias derived from it (a mounted copy that shares payloads) can outlive it:
    // only the caller knows when it is truly done.
    bool take_ready(ConversationStoreRead& out);
    // Bounded wait for the next ready result (own thread only: never call it while holding a caller lock that
    // cancel_read() or demand() needs). timeout_ms 0 = non-blocking. This and take_ready() above deliver the next
    // completion of *any* request, so a caller that mounts the image must check request_id/generation or, better, use
    // the request-specific overloads below.
    bool wait_ready(ConversationStoreRead& out, uint64_t timeout_ms);
    // Request-specific delivery: only that read's own completion is ever consumed.  A caller that mounts what it
    // receives must use these - a global take_ready() can hand back another request's completion, and mounting that
    // would restore the wrong history.  False while the request has not settled (and for an unknown request).
    bool take_ready(uint64_t request_id, uint64_t generation, ConversationStoreRead& out);
    bool wait_ready(uint64_t request_id, uint64_t generation, ConversationStoreRead& out, uint64_t timeout_ms);
    // The single lease release: the caller reports that this read is finished AND that every alias of the returned
    // image is gone.  One unconditional, idempotent transition: a queued, active, ready or handed-out request is
    // retired here, its charge is refunded once its buffers are actually dropped, and a released request can never be
    // delivered or completed afterwards.  Safe and counted for unknown or already-released identifiers; a release
    // that never fires leaves the bytes charged, which is a caller bug the module cannot detect - and charging is the
    // fail-safe direction, never a silent refund of live bytes.
    void release_read(uint64_t request_id, uint64_t generation);
    ConversationStoreReadStatus request_status(uint64_t request_id, uint64_t generation) const;
    // Recency promotion after a *confirmed* restore (never on prefetch or on a mere append): moves the head into the
    // protected segment so eviction reclaims other histories first, demoting the oldest protected head when the
    // segment is at its three-quarter ceiling.  The durable catalog write is handed to this module's storage thread,
    // so this call performs no storage wait; the promotion is a policy hint, and a failed commit leaves it pending for
    // the next one rather than risking the authority.
    void confirmed_restore(uint64_t head_id);

    // --- publication --------------------------------------------------------------------------------------
    // The image is an owning frozen lease: the store keeps this alias until the publication finishes and the caller
    // keeps its own.  One admitted writer image at a time, and it is admitted against the published RAM allowance
    // exactly like a read: the image *and* the bounded writer workspace this attempt will allocate (its copied
    // request, layout, prefix chains, parent/donor decode, chunk vectors and scratch, object/orphan records, encoded
    // manifest and catalog copy) must both fit the envelope, and the workspace must also fit the preparation ceiling
    // beside the index, the live read descriptors' own copies and any diagnostic decode.  Admission is classified
    // (see ConversationStorePublishAdmission): `accepted` when the counted peak fits an envelope whose only occupants
    // are work this store may not free, `deferred` when it fits that envelope on its own but resources occupied right
    // now keep it out - after every *speculative* read that may be revoked has been revoked and the admission retried
    // under the same lock (an active worker, a demand and a handed-out image are never revoked: their charges are
    // released only when they settle) - and `refused` when the shape itself can never fit, another branch is already
    // admitted, or this store is not serving.  While the writer is busy, a new publication is admitted only
    // when it is proven to be the *same lineage* as the one in flight (both carry `cover` for the same nonzero
    // `supersedes` and this one is the newer generation): it then
    // replaces it, and the replaced request's result is reported as `coalesced`.  A different branch - including a
    // second new branch - is never coalesced: it is refused as `refused_busy`, the in-flight predecessor is left
    // untouched and the newer, optional store is dropped as a safe miss.  The call returns immediately.
    ConversationStorePublishAdmission publish(std::shared_ptr<const SavedConversation> image,
                                              const ConversationStorePublish& request, std::string& why);
    // Cancel only a not-yet-started publication, never a read with coincident identifiers.
    void cancel_publication(uint64_t request_id, uint64_t generation);
    bool take_publish(ConversationStorePublishResult& out);
    bool wait_publish(ConversationStorePublishResult& out, uint64_t timeout_ms);
    bool writer_busy() const;

    // --- host admission (RAM reserve handshake) -----------------------------------------------------------
    // Two authoritative numbers in two scopes that must never be mixed: a caller neither subtracts the shared disk
    // quota from a RAM envelope, nor treats a RAM lease as free disk.
    //   ram_held_bytes()       every byte this module currently owns in host memory *or has committed to
    //                          allocating*: index metadata, bounded preparation and read descriptors, diagnostic
    //                          decode workspace, the full peak of every read that has begun allocating, materialized
    //                          read images, the shared image of an in-flight publication, and I/O scratch.  No
    //                          allocation is ever made that this figure did not already include, and a read or a
    //                          publication whose counted peak cannot fit is refused or deferred instead of overrunning
    //                          the allowance.  The figure stays inside the published allowance except in the two
    //                          documented cases: a caller that lowers the allowance under admitted work (see
    //                          publish_read_allowance(), which never revokes what is already admitted), and a
    //                          publication admitted on its own transferred-image credit (`external_lease_bytes`),
    //                          which widens that one ticket's fit and nothing else.
    //   ram_lease_bytes()      the handed-out share of it: a read image returned by take_ready()/wait_ready() and not
    //                          yet released, plus the publication image alias.  Charge this to your own envelope; the
    //                          module will not hand those bytes to another read, and never refunds them on its own.
    //   ram_read_budget()      the absolute admission bound in force.
    //   ram_available_bytes()  == max(0, ram_read_budget() - ram_held_bytes()).
    //   disk_charged_bytes()   committed objects, manifests and control files (catalog, issuer key, run intent,
    //                          temporaries) still present under the root, including bytes waiting to be reclaimed -
    //                          credited only once the deletion has completed.
    //   disk_available_bytes() == max(0, disk_quota_bytes() - disk_charged_bytes()).
    //   disk_free_bytes()      last sampled filesystem space under the root, checked against min_free_bytes.
    // suspend_read_admission() stops admitting *new* read work and is serialized with the reader's start-of-
    // allocation admission, so when it returns every byte this module can still allocate is already counted in
    // ram_held_bytes(): queued work can allocate nothing while admission is suspended.  A caller may publish a new
    // allowance afterwards without racing an allocation.
    uint64_t ram_held_bytes() const;
    uint64_t ram_lease_bytes() const;
    uint64_t ram_read_budget() const;
    uint64_t ram_available_bytes() const;
    void suspend_read_admission();
    // Sets the read-admission budget ABSOLUTELY to `bytes` and returns the value in force.  A value below
    // ram_held_bytes() never revokes admitted work: admitted reads still complete, and no new read is admitted until
    // the accounting drops below the new bound.
    uint64_t publish_read_allowance(uint64_t bytes);
    uint64_t disk_charged_bytes() const;
    uint64_t disk_quota_bytes() const;
    uint64_t disk_available_bytes() const;
    uint64_t disk_free_bytes() const;   // last sampled filesystem available bytes under the root

    // --- lifecycle ----------------------------------------------------------------------------------------
    // Durably advances the execution-domain epoch, invalidating every stored record and outstanding completion of
    // the previous epoch.  Called when the engine discards state for safety (a non-finite/poison detection).  The
    // durable commit runs on this module's storage thread and this call waits for it (bounded); it returns true only
    // when that commit succeeded.  A failure - including one this call could not confirm in time - latches the store
    // faulted and leaves the run intent armed: the next start must not reuse anything.
    bool invalidate_domain(const char* reason);
    // Healthy drain: finishes or cancels storage work, durably commits a final catalog, and only then clears and
    // flushes this domain's run-intent marker - in that order, so a close that cannot finish its commit never removes
    // the marker that tells the next start to distrust this domain.  Only a healthy close promises that eligible
    // completed publications survive a restart; a faulted close, and the destructor, deliberately abandon the intent
    // (an unclean exit must not look healthy).  An instance that does not own the root lock performs no authority
    // write and removes no marker at all.
    void close();
    ConversationStoreCounters counters() const;
    // Diagnostics for qualification evidence: one bounded line per stage of the open sequence.
    std::string describe() const;

    // Test-only seams (documented, deterministic): arm a named publication/transaction step to fail, and count the
    // bytes this module actually read from a payload source.  Steps accumulate, and each one fires exactly once;
    // an empty name disarms every armed step.  Recognised steps:
    //   "layout"          the image's layout is refused before any payload is read
    //   "objects"         the next immutable object write fails
    //   "manifest"        the manifest write fails
    //   "catalog"         the catalog commit is refused before it plans anything
    //   "catalog-rename"  the catalog rename fails: nothing durable changed
    //   "catalog-sync"    the rename succeeded and the root directory flush failed: durability is uncertain
    //   "catalog-apply"   the rename succeeded and the apply could not allocate: durability is uncertain
    //   "intent-unlink-sync"  the flush after removing the run-intent marker failed (it is then re-armed)
    //   "intent-rearm-sync"   the flush of the re-armed run-intent marker failed
    // An unrecognised step is never taken, so arming one costs nothing.
    void set_publish_fault_for_test(const char* step);
    uint64_t source_bytes_read_for_test() const;
#ifdef STRATA_STORE_TEST_HOOKS
    // Pause the real worker after reservation, outside the store mutex. Absent from production builds.
    void set_read_gate_for_test(void (*gate)(void*), void* context);
#endif

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace strata::core
