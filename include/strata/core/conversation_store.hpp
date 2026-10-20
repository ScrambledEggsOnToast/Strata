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
// Threading: one admitted writer thread and a bounded read pipeline (Options::max_reads incoming images). Callers
// perform explicit waits only on their own thread, without holding an admission lock: wait_ready(), wait_publish(),
// invalidate_domain() and finish_shutdown() may wait. request_shutdown() performs no I/O; finish_shutdown() may wait
// indefinitely for filesystem-blocked work and must not be mistaken for a deadline-bounded join. No storage wait
// runs while the module holds its catalog mutex, and filesystem access is descriptor-relative to validated paths.
#pragma once
//
// STRATA_STORE_TIMING=1 enables bounded, metadata-only store_read/store_head receipts (off by default).
// `planned` is not removal: only `retired` after durable catalog apply names a committed retirement, with
// its applied catalog_version and applied_ns. Pin/lease/protected/recency fields describe the victim just
// before apply; protected is preferential retention, not a pin. Capacity evictions count only committed
// removals, excluding covered replacements. Candidate rejections are capped at eight per selection;
// candidate_rejections_omitted reports the remainder without hiding ownership transitions. No tokens,
// prompt bytes, scopes, credentials, paths or content digests enter these receipts.

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
    // MAN2 only: one position-bound main residual on the draft-owning stage; never a K/V donor.
    draft_boundary_residual = 27,
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
    // credited only once the file is confirmed gone.  A file a failed write never reached its own name carries its
    // one charge under the name it did reach - the temporary - and never under both.  Nothing is ever admitted
    // against a promise to delete later.
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
    // Completed, unharvested publication results plus the admitted writer are bounded by 4 * this value + 4.
    // Admission defers at that bound until take_publish()/wait_publish() consumes a result; no verdict is evicted.
    uint32_t max_pending_publishes = 1;
    // 0: no read allowance has been published yet.  No read materializes and a speculative read is refused, and no
    // publication is admitted either: reads and publications are both admitted against this one envelope, and a
    // publication must fit it as its image *plus* the writer workspace it needs (see publish()).  There is no
    // admission path that skips the check, and publish_read_allowance() is what makes both absolute.
    uint64_t read_budget_bytes = 0;
    // Sample successful catalog-apply time for publication receipts. False adds
    // no clock read; catalog versions remain available without timing.
    bool publication_timing = false;
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

// The exact catalog snapshot a lookup examined. Both fields are sampled while
// holding the catalog mutex; `observed_ns` is steady-clock time in this process.
struct ConversationStoreLookupFrontier {
    uint64_t observed_ns = 0;
    uint64_t catalog_version = 0;
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
    none = 0,     // no verdict: inspect request_progress() to distinguish an unknown, queued or active read
    ready,        // a complete, verified image is attached
    missing,      // no eligible stored prefix (an ordinary cache miss)
    corrupt,      // an eligible record failed validation: treat as a miss, never restore it
    cancelled,    // the caller cancelled it, or a newer one replaced it
    stale,        // the execution-domain epoch advanced before it completed
    refused_budget,   // the published read allowance did not admit the image
    refused_quota,    // the store is full: skip storage, keep serving
    refused_limits,   // the record exceeds ConversationStoreLimits: refused, never discarded (it may belong to another profile)
    failed,       // I/O or allocation failure
    disabled,     // persistence is off; this is always a miss
};

const char* conversation_store_read_status_name(ConversationStoreReadStatus status);

// Descriptor ownership, independent of the verdict above. `settled` includes failures and revoked/deferred
// speculation: it promises neither success nor delivery, and an explicit retry starts a new attempt.
enum class ConversationStoreReadState : uint8_t {
    unknown = 0,  // no descriptor (including an already-erased release or terminal hand-over)
    queued,       // accepted selection waiting for a reader, allowance or incoming-image slot
    materializing, // a reader owns the reservation; no partial image can be consumed
    settled,      // the allocation frame unwound; inspect status before attempting delivery
    handed_out,   // the complete image was handed over and its lease awaits owner release
    released,     // owner released it, but an active allocation frame has not yet unwound
};

const char* conversation_store_read_state_name(ConversationStoreReadState state);

enum class ConversationStoreReadComponent : uint8_t { metadata, recurrent, ple, indexer, main_kv, draft, count };

struct ConversationStoreReadBytes {
    uint64_t total = 0;
    uint64_t acknowledged = 0;
    bool operator==(const ConversationStoreReadBytes&) const = default;
};

struct ConversationStoreReadProgress {
    ConversationStoreReadState state = ConversationStoreReadState::unknown;
    ConversationStoreReadStatus status = ConversationStoreReadStatus::none;
    // Refresh/retry may replace the manifest under the same request identity. Totals and acknowledgements belong
    // only to this attempt; its nonzero identity changes and all component counts reset at that transition.
    uint64_t attempt_id = 0;
    uint64_t head_id = 0;
    bool totals_known = false; // false until manifest validation and authorized-cut checks have succeeded
    // Diagnostic only. `materialize_started` means this attempt was dispatched
    // to the reader, not merely queued or refused (and does not acknowledge I/O).
    // The prior bit survives every reissue of this request/generation: only a
    // ready result exposes a duration, and replaced attempts have no such sample.
    bool materialize_started = false;
    bool prior_materialize_unmeasured = false;
    // Logical bytes, not unique objects or physical disk traffic. Metadata includes the validated encoded manifest,
    // token IDs and image keys; recurrent is GDN; PLE is its history; indexer includes tails/dead/block positions and
    // main-model pooled keys; main_kv is main K/V and scales; draft is every part of the extra draft pool.
    // Within an attempt, acknowledged only grows after successful verification/copy, never exceeds total, and
    // equals total for every component on `ready`. Even then, only take_ready()/wait_ready() can hand out an image.
    std::array<ConversationStoreReadBytes, size_t(ConversationStoreReadComponent::count)> components{};

    // A watchdog may beat only for acknowledged bytes, never a state change, timer or condition-variable wake.
    // A new attempt has a zero baseline: discarded bytes from an earlier manifest cannot count as new work.
    bool acknowledged_after(const ConversationStoreReadProgress& previous) const {
        for (size_t i = 0; i < components.size(); ++i)
            if (components[i].acknowledged > (attempt_id == previous.attempt_id ?
                                             previous.components[i].acknowledged : 0)) return true;
        return false;
    }
};

// Advisory recovery information for a fresh demand refused solely by its image peak.
// Includes descriptor ownership and a conservative bound for post-decode restore scaffolding.
// The owner must also cover existing store-held bytes; a retry always revalidates admission.
struct ConversationStoreReadRefusal {
    uint64_t additional_ram_bytes = 0;
};

struct ConversationStoreRead {
    ConversationStoreReadStatus status = ConversationStoreReadStatus::none;
    uint64_t request_id = 0;
    uint64_t generation = 0;
    uint64_t head_id = 0;
    // Receipt identity and origin of the attempt that produced this result. Demand may adopt an existing
    // prefetch without relabelling it; a demand redo has a new attempt_id and prefetched=false.
    uint64_t attempt_id = 0;
    bool prefetched = false;
    // The execution-domain epoch this read was admitted in.  Never relabelled: a completion of an older epoch is
    // retired rather than delivered under the current one.
    uint64_t epoch = 0;
    int64_t tokens = 0;
    // True when the selected cut is the image's live record and false when it is a checkpoint: the caller needs this
    // to know which part of the restored image the cut describes.
    bool live = false;
    uint32_t stages = 0;
    uint64_t payload_bytes = 0;
    // Successful whole-image filesystem/page-cache read, verification, allocation and host materialization.
    // Always measured, independent of tracing. Excludes reader queue/admission and ready residence; not raw SSD bandwidth.
    double materialize_ms = 0;
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
//
// Protection is not limited to the named heads.  Every object the publication will *name without creating* - an
// adopted donor chunk, a chunk the outgoing image reproduces byte for byte, or a complete file adopted from an
// accounted orphan - is leased for the life of the attempt: a head that happens to be its only owner may be evicted
// for room while the manifest is still being written, and the file must survive until the transaction settles, or the
// committed manifest would name an object the store has already deleted and credited.  The lease ends with the
// attempt: an object no retained head and no other publication still names is then reclaimable again.
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

// Admission owns the accounting transition, independently of the asynchronous result:
// accepted transfers the image's charge to the store; the caller releases its matching lease immediately.
// deferred means occupied resources prevent admission; the caller retains image and lease for a later seam.
// refused means an invalid/over-cap image, an incompatible writer already admitted, or a non-serving store.
// Neither deferred nor refused transfers ownership. Demand/handed leases and active worker reservations
// cannot be refunded to admit another operation.
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
    // The successful apply that made this result visible to catalog lookups.
    // Both remain zero when no durable catalog apply occurred.
    uint64_t applied_ns = 0;
    uint64_t catalog_version = 0;
    bool released = true;        // the store dropped its shared_ptr<const SavedConversation> alias
    std::string detail;
};

#ifdef STRATA_STORE_TEST_HOOKS
// One bounded, metadata-only summary of a started writer. Missing phase clocks remain zero (not measured).
// Durations are inclusive/nested: file fsync is part of payload/manifest/catalog, never an additional cost.
// calls_complete=false means an exception interrupted per-call timing; the outer writer clocks remain usable.
// The observer is copied at admission and called only after settlement, outside the store mutex. It must be
// noexcept, bounded, and perform no I/O; its context must outlive finish_shutdown(). No production instrumentation.
struct ConversationStorePublishTrace {
    uint64_t request_id = 0, generation = 0, epoch = 0;
    uint64_t admitted_ns = 0, writer_started_ns = 0, body_finished_ns = 0, result_ready_ns = 0;
    uint64_t payload_started_ns = 0, payload_finished_ns = 0;
    uint64_t manifest_started_ns = 0, manifest_finished_ns = 0;
    uint64_t catalog_started_ns = 0, catalog_finished_ns = 0, catalog_applied_ns = 0;
    uint64_t logical_payload_bytes = 0, source_bytes = 0, source_chunks = 0, hashed_bytes = 0;
    uint64_t source_read_ns = 0, hash_ns = 0;
    uint64_t file_write_ns = 0, file_write_bytes = 0, file_fsync_ns = 0, file_fsync_calls = 0;
    uint64_t dir_fsync_ns = 0, dir_fsync_calls = 0;
    uint64_t thread_cpu_ns = 0;
    bool thread_cpu_known = false;
    bool calls_complete = true;
    ConversationStorePublishStatus status = ConversationStorePublishStatus::disabled;
};
#endif

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
    // Idle policy transactions that committed/refused; restore calls sharing a pending hint; pending hints consumed
    // by publish/evict/invalidate candidate copies (absorption counts the attempt, not its durable outcome).
    uint64_t promotions_applied = 0, promotions_coalesced = 0, promotions_absorbed = 0, promotions_refused = 0;
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

// pending means admission is fenced, not that workers have stopped. Both terminal states prove quiescence and lock
// release; only durable_clean also proves the final catalog durable and this run's intent durably removed.
enum class ConversationStoreShutdownState : uint8_t {
    none = 0,
    pending,
    quiescent_unclean,
    durable_clean,
};

const char* conversation_store_shutdown_state_name(ConversationStoreShutdownState state);

// One mutex-coherent, allocation-free observation; no inference-readiness or admission promise.
struct ConversationStoreReadiness {
    bool enabled = false;
    bool healthy = false;
    bool run_intent = false;
    uint64_t epoch = 0;
    std::array<char, 192> last_error{};
};

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
    ConversationStoreReadiness readiness() const;

    // The execution domain's current epoch, the issuer key the frontend derives authorized scopes from, and the
    // durable run-intent marker this owner holds. The secret remains exactly 32 bytes of cryptographic entropy,
    // immutable after successful open. Its existing disk authority accepts a legacy raw secret or a recognized
    // minimum-reader envelope containing that same secret. Unreadable/unsupported/malformed authority disables
    // persistence rather than minting a new authorization identity or reconciling histories.
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
                        const std::function<void(const ConversationStoreMatch&)>& visit,
                        ConversationStoreLookupFrontier* frontier = nullptr) const;
    template<class Visitor>
    void for_each_match(const ConversationStoreSelector& selector, const std::vector<int32_t>& ids,
                        const std::vector<ConversationImageKey>& images, Visitor&& visit,
                        ConversationStoreLookupFrontier* frontier = nullptr) const {
        for_each_match(selector, ids.data(), ids.size(), images.data(), images.size(),
                       std::function<void(const ConversationStoreMatch&)>(std::forward<Visitor>(visit)), frontier);
    }
    std::vector<ConversationStoreMatch> matches(const ConversationStoreSelector& selector,
                                               const std::vector<int32_t>& ids,
                                               const std::vector<ConversationImageKey>& images) const;
    // Catalog-only presence, without allocation or I/O; not a lease or promise of future retention.
    bool retains_head(uint64_t head_id) const;
    bool manifest_info(uint64_t head_id, ConversationStoreManifestInfo& out) const;
    // The stream table of a retained head, read from its manifest (bounded, read-only, no catalog lock held): for
    // planning reuse intervals.  It exposes no payload and authorizes nothing.  Decoded under the store's retention
    // figures, so a head another profile captured is still readable as long as it is structurally sound.
    bool manifest_streams(uint64_t head_id, std::vector<ConversationStoreStreamInfo>& out) const;

    // --- reads --------------------------------------------------------------------------------------------
    // demand: required by a request that will consume a slot; admit its whole peak against the published allowance
    //   *together with every other byte the store holds* - the index, the live descriptors' own copies, the admitted
    //   reads' decode preparation and any diagnostic decode - or, when no allowance is published yet, let it wait.
    //   When its peak exceeds the current allowance it is refused immediately. An optional refusal quote lets
    //   the owner reclaim eligible cache memory before one revalidated attempt; otherwise use ordinary prefill.
    // prefetch: optional, only materialized while spare allowance exists, never overtaking a demand. An initial
    //   admission refusal returns false and owns nothing. An accepted selected descriptor that later settles
    //   refused_budget retains its bounded query/deadline/head hold, but no image reservation or incoming slot;
    //   publish_read_allowance retries it once usable capacity recovers enough for its unchanged peak.
    // Both select the longest exact authorized prefix themselves, deduplicate by (request_id, generation), enforce
    // the read-descriptor and preparation ceilings *before* allocating anything, bind the request to the epoch it was
    // admitted in, and return immediately; the caller consumes the result with take_ready()/wait_ready() at its own
    // safe seam.  A demand never revokes anything of its own; it does revoke outstanding speculation.
    // Selection owns a bounded metadata/file hold, not a RAM image reservation. Internal speculative revocation
    // drops materialized bytes and incoming slots but keeps that selection, original query and deadline within
    // max_read_requests/metadata_bytes. Demand can retry it without losing its cut to eviction or a covered
    // replacement. Cancellation, withdrawal, deadline expiry, domain retirement and release end the hold.
    // Retirement staging and selection are serialized by the catalog mutex; a selection made after staging but
    // before durable apply owns the captured immutable manifest until release, even if the catalog retires it.
    // refusal is cleared on entry and populated only for the fresh peak-budget refusal, never an accepted read.
    bool demand(const ConversationStoreQuery& query, std::string& why,
                ConversationStoreReadRefusal* refusal = nullptr);
    bool prefetch(const ConversationStoreQuery& query, std::string& why);
    // Explicit read cancellation leaves a terminal verdict until its owner calls release_read().
    // Issued filesystem work may finish, but no cancelled image is delivered.
    void cancel_read(uint64_t request_id, uint64_t generation);
    // Withdraw optional work only. Once demand adopts a read, the advisory channel cannot cancel
    // or release it. Unadopted descriptors are retired; a later demand may submit a fresh query.
    void withdraw_prefetch(uint64_t request_id, uint64_t generation);
    // Re-run an outstanding speculative prefetch's catalog selection. This does not admit initially refused offers,
    // and the production offer channel declines missing heads before calling prefetch. A demanded, handed-out,
    // cancelled, active or failed read is never touched; a valid ready image is refreshed only for a longer cut.
    // An unmaterialized selection may also rebind to a newer equal cut. Unchanged budget deferrals wait for
    // publish_read_allowance, not catalog polling. Returns false when nothing is eligible.
    bool refresh_speculation(uint64_t request_id, uint64_t generation);
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
    // would restore the wrong history. False means no result was delivered: a timeout, unknown/erased descriptor,
    // already-handed result, cancellation or stopped/faulted store can all return false. A wake is not success;
    // inspect request_progress()'s state AND status to distinguish pending work from an undeliverable verdict.
    bool take_ready(uint64_t request_id, uint64_t generation, ConversationStoreRead& out);
    bool wait_ready(uint64_t request_id, uint64_t generation, ConversationStoreRead& out, uint64_t timeout_ms);
    // The single lease release: the caller reports that this read is finished AND that every alias of the returned
    // image is gone.  One unconditional, idempotent transition: a queued, active, ready or handed-out request is
    // retired here, its charge is refunded once its buffers are actually dropped, and a released request can never be
    // delivered or completed afterwards.  Safe and counted for unknown or already-released identifiers; a release
    // that never fires leaves the bytes charged, which is a caller bug the module cannot detect - and charging is the
    // fail-safe direction, never a silent refund of live bytes.
    // A final lease on a retired head also wakes the existing writer to delete its files in bounded idle passes;
    // confirmed deletion returns their physical charge without waiting for another publication or close.
    void release_read(uint64_t request_id, uint64_t generation);
    ConversationStoreReadStatus request_status(uint64_t request_id, uint64_t generation) const;
    // Side-effect-free, mutex-consistent ownership/verdict/byte snapshot; allocates no image and triggers no retry.
    // Prefer an unreleased descriptor if an old released worker still unwinds under the same request identity.
    // Released descriptors disappear when that frame unwinds; no unbounded per-identifier history is retained.
    ConversationStoreReadProgress request_progress(uint64_t request_id, uint64_t generation) const;
    // Recency promotion after a *confirmed* restore (never on prefetch or on a mere append): moves the head into the
    // protected segment so eviction reclaims other histories first, demoting the oldest protected head when the
    // segment is at its three-quarter ceiling. This call allocates no work item and performs no storage wait: one
    // mutex-protected hint coalesces restores until a catalog candidate absorbs them. Urgent invalidations and runnable
    // publications precede an idle policy transaction; a restore after a candidate copy re-arms the next hint. An
    // ordinary refusal leaves the live policy for the next transaction, without retrying it in a busy loop.
    void confirmed_restore(uint64_t head_id);

    // --- publication --------------------------------------------------------------------------------------
    // The image is an owning frozen lease: the store keeps this alias until the publication finishes and the caller
    // keeps its own.  One admitted writer image at a time, and it is admitted against the published RAM allowance
    // exactly like a read: the image *and* the bounded writer workspace this attempt will allocate (its copied
    // request, layout, prefix chains, parent/donor decode, chunk vectors and scratch, object/orphan records, encoded
    // manifest and catalog copy) must both fit the envelope, and the workspace must also fit the preparation ceiling
    // beside the index, read descriptors, admitted read preparation and diagnostic decode.
    // Eligible speculation is revoked before a RAM/preparation deferral; active workers keep their reservations
    // until settlement and handed-out images keep their leases until owner release.
    // Each accepted request reserves one terminal-result slot until take_publish()/wait_publish() consumes it,
    // including coalesced, cancelled and close-time outcomes. A full result queue defers admission (counted in
    // deferred_publishes), retaining the caller's image and lease; deferred/refused calls produce no later result.
    // While the writer is busy, a new publication is admitted only
    // when it is proven to be the *same lineage* as the one in flight (both carry `cover` for the same nonzero
    // `supersedes` and this one is the newer generation): it then
    // replaces it, and the replaced request's result is reported as `coalesced`.  A different branch - including a
    // second new branch - is never coalesced: it is refused as `refused_busy`, the in-flight predecessor is left
    // untouched and the newer, optional store is dropped as a safe miss.  The call returns immediately.
    ConversationStorePublishAdmission publish(std::shared_ptr<const SavedConversation> image,
                                              const ConversationStorePublish& request, std::string& why);
    // Cancel only a not-yet-started publication, never a read with coincident identifiers.
    void cancel_publication(uint64_t request_id, uint64_t generation);
    // Consume each admitted request's terminal result exactly once, in settlement order; no result expires.
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
    // Observation-only coherent ledger view; never reserves, refreshes or admits.
    ConversationStoreRamTrace ram_trace() const;
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
    // Immediate, no I/O: fence new work, retire unhanded reads and cancel unstarted publication. Existing handed-out
    // leases remain valid and release_read() stays legal. finalize_before_ms uses steady_clock milliseconds since
    // its epoch, as the native serving loop does. 0 forbids starting optional finalization; UINT64_MAX is unlimited.
    // Repeated requests only tighten this deadline (0 is strictest), never reopen admission or regain permission.
    // Once finalization has begun, a later request cannot interrupt it or change the eventual durable verdict.
    void request_shutdown(uint64_t finalize_before_ms);
    // Cheap mutex-guarded observation, including while finish_shutdown() is blocked on worker or filesystem I/O.
    ConversationStoreShutdownState shutdown_state() const;
    // Potentially UNBOUNDED: cooperatively stop/join workers, settle ownership, then optionally finalize. The
    // deadline governs only whether finalization may BEGIN after draining; it cannot interrupt filesystem calls,
    // joins or a finalization already begun. Always keeps the live owner and root lock until quiescence is proven.
    // An absent request implies request_shutdown(0). Concurrent/repeated finish calls share one terminal result.
    // durable_clean requires a durable final catalog followed by proven durable intent removal; every other
    // terminal result is quiescent_unclean, with intent left armed. Non-owners never write authority or remove intent.
    ConversationStoreShutdownState finish_shutdown();
    // Legacy healthy drain: request an unlimited deadline and finish. A prior stricter shutdown request is honored,
    // not upgraded. A faulted owner, or the destructor (request deadline 0 + finish), deliberately leaves intent
    // armed. Completed publications survive restart only after durable_clean; no promise covers an uncommitted tail.
    void close();
    ConversationStoreCounters counters() const;
    // Diagnostics include shutdown state and one bounded line per stage of the open sequence.
    std::string describe() const;

    // Test-only seams (documented, deterministic): arm a named publication/transaction step to fail, and count the
    // bytes this module actually read from a payload source.  Steps accumulate, and each one fires exactly once;
    // an empty name disarms every armed step.  Recognised steps:
    //   "layout"          the image's layout is refused before any payload is read
    //   "objects"         the next immutable object write fails
    //   "object-parent-sync" the objects/ flush fails after mkdir (default), during revival, or at the batch drain
    //   "object-shard-sync"  the shard flush fails after object rename or during revival, before manifest creation
    //   "manifest"        the manifest write fails
    //   "catalog"         the catalog commit is refused before it plans anything
    //   "catalog-rename"  the catalog rename fails: nothing durable changed
    //   "catalog-sync"    the rename succeeded and the root directory flush failed: durability is uncertain
    //   "catalog-apply"   the rename succeeded and the apply could not allocate: durability is uncertain
    //   "catalog-reclaim" the transaction committed and the reclamation that follows it could not allocate: the
    //                     commit stands, nothing is deleted or credited, and the files stay queued and charged
    //   "intent-unlink-sync"  the flush after removing the run-intent marker failed (it is then re-armed)
    //   "intent-rearm-sync"   the flush of the re-armed run-intent marker failed
    // An unrecognised step is never taken, so arming one costs nothing.
    void set_publish_fault_for_test(const char* step);
    uint64_t source_bytes_read_for_test() const;
#ifdef STRATA_STORE_TEST_HOOKS
    // With STRATA_STORE_DIR_FSYNC_BATCH=1, before any manifest is created (test builds only):
    //   "batch-shard-open"     fail before opening the next touched shard
    //   "batch-shard-sync"     fail before flushing the next opened shard
    //   "batch-shard-partial"  fail before a second touched shard, after at least one was flushed
    //   "batch-parent-sync"    fail before the final objects-parent flush, including existing shards
    //   "batch-shard-allocate" throw bad_alloc while owning the opened shard
    // Arm these through set_publish_fault_for_test(); generic object flush faults work in either mode.
#endif
#ifdef STRATA_STORE_TEST_HOOKS
    // Pause the real worker after reservation, outside the store mutex. Absent from production builds.
    void set_read_gate_for_test(void (*gate)(void*), void* context);
    // Pause after publishing validated manifest totals or acknowledging a verified chunk, outside the mutex.
    void set_read_progress_gate_for_test(void (*gate)(void*), void* context);
    // Pause a real catalog transaction outside the mutex, before staging (false) or after staging (true).
    void set_catalog_gate_for_test(void (*gate)(void*, bool staged), void* context);
    // Pause a nonempty object-directory batch after payload emission/revalidation, before opening any shard.
    // Outside the mutex; absent from production. Context must outlive the publication using it.
    void set_object_batch_gate_for_test(void (*gate)(void*), void* context);
    // Pause after payload emission, before the serving/shutdown fence and before any manifest. Outside the mutex.
    void set_publish_payload_gate_for_test(void (*gate)(void*), void* context);
    void set_publish_trace_for_test(
        void (*observer)(void*, const ConversationStorePublishTrace&) noexcept, void* context);
#endif

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace strata::core
