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
// Threading: one admitted writer thread and a bounded read pipeline (Options::max_reads materializations).  Nothing
// in this module blocks under a caller's admission lock: a caller that wants a restore at its own safe seam calls
// wait_ready(), and every other entry point returns immediately.  No storage wait ever runs while the module holds
// its own catalog mutex.
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
    std::string root;                      // private directory on the cache device; empty disables persistence
    // Verified identities.  `domain` is the strong execution-domain digest (state format, kernel contract, resolved
    // layout/arithmetic/draft).  `model` is the verified common model identity and `engine` the verified engine
    // identity; both must come from release data, never a sampled fingerprint, alias or profile name.  A zero value
    // disables persistence safely instead of authorizing the wrong records.
    std::array<uint8_t, 32> domain{};
    std::array<uint8_t, 32> model{};
    std::array<uint8_t, 32> engine{};
    uint64_t quota_bytes = uint64_t(128) << 30;    // one shared physical quota for every layout
    uint64_t min_free_bytes = uint64_t(128) << 30; // filesystem floor, resampled before new reservations
    uint64_t metadata_bytes = uint64_t(64) << 20;  // ceiling on this module's in-RAM index
    size_t head_limit = 128;                       // retained branch heads across the whole store
    uint64_t chunk_bytes = uint64_t(1) << 20;      // immutable object size (the tail may be shorter)
    // One materialized incoming restore at a time by default: read *descriptors* are bounded separately, and an
    // in-flight prefetch's speculative resources are revoked when a demand request needs the slot.
    uint32_t max_reads = 1;
    uint32_t max_read_requests = 16;               // bounded queued read identifiers (demand and prefetch)
    // Extra *same-lineage* publications that may replace the one in flight (see publish()).  Never a second
    // admitted image for a different branch.
    uint32_t max_pending_publishes = 1;
    uint64_t read_budget_bytes = 0;                // 0: no read admission until publish_read_allowance()
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
    refused_limits,   // the record exceeds ConversationStoreLimits
    failed,       // I/O or allocation failure
    disabled,     // persistence is off; this is always a miss
};

const char* conversation_store_read_status_name(ConversationStoreReadStatus status);

struct ConversationStoreRead {
    ConversationStoreReadStatus status = ConversationStoreReadStatus::none;
    uint64_t request_id = 0;
    uint64_t generation = 0;
    uint64_t head_id = 0;
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
    uint64_t supersedes = 0;                 // the head this image was captured from (0 = a new branch)
    bool cover = false;                      // the caller proved it covers `supersedes`' compatible prefix
    std::vector<ConversationStoreReuse> reuse;
};

enum class ConversationStorePublishStatus : uint8_t {
    stored = 0,             // a new branch head
    replaced,               // superseded its predecessor in the same branch and epoch
    retained_branch,        // the predecessor keeps a pinned/divergent cut, so both heads are retained
    stale,                  // an older generation, a different epoch or a retired predecessor
    refused_quota,
    refused_floor,
    refused_limits,         // the image exceeds ConversationStoreLimits, or the store's index ceiling is reached
    refused_busy,           // the bounded writer and its pending slots are full
    coalesced,              // replaced by a newer covered publication of the same lineage (lease released)
    failed,
    disabled,
};

const char* conversation_store_publish_status_name(ConversationStorePublishStatus status);

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
    uint64_t lock_conflicts = 0, aborted_publications = 0, recovered_orphans = 0;
};

// Cheap catalog metadata of one retained head (for diagnostics and for planning a reuse interval).
struct ConversationStoreManifestInfo {
    uint64_t head_id = 0;
    uint64_t epoch = 0;
    uint64_t generation = 0;
    uint64_t parent_head_id = 0;
    uint64_t payload_bytes = 0;
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
    bool healthy() const;                       // enabled and no fatal I/O or authority failure
    void set_unhealthy(const char* reason);      // the owner observed a failure that must stop reuse (e.g. a safety fault)

    // The execution domain's current epoch, the issuer key the frontend derives authorized scopes from, and the
    // durable run-intent marker this owner holds.
    uint64_t epoch() const;
    const std::array<uint8_t, 32>& issuer_key() const;
    bool run_intent_active() const;
    uint64_t domain_digest_truncated() const;

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
    // The stream table of a retained head, read from its manifest (bounded); for planning reuse intervals.
    bool manifest_streams(uint64_t head_id, std::vector<ConversationStoreStreamInfo>& out) const;

    // --- reads --------------------------------------------------------------------------------------------
    // demand: required by a request that will consume a slot, admitted as soon as the read allowance admits it.
    // prefetch: optional, only admitted while spare allowance exists, never overtaking a demand request.
    // Both select the longest exact authorized prefix themselves, deduplicate by (request_id, generation) and return
    // immediately; the caller consumes the result with take_ready()/wait_ready() at its own safe seam.
    bool demand(const ConversationStoreQuery& query, std::string& why);
    bool prefetch(const ConversationStoreQuery& query, std::string& why);
    // Cancels a request or a prefetch for (request_id, generation): an issued filesystem operation may finish, but
    // its completion is dropped and its bytes are refunded.  Safe with unknown or already-completed identifiers.
    void cancel(uint64_t request_id, uint64_t generation);
    // Non-blocking: false when nothing is ready.  A returned image is complete and byte-verified.  The caller
    // receives the only alias of an image whose RAM stays charged to this module (see ram_lease_bytes()) until it
    // calls release_read(); nothing is refunded while the caller still holds bytes.  The store does not drop its
    // accounting when the returned shared_ptr is destroyed, because an alias derived from it (a mounted copy that
    // shares payloads) can outlive it: only the caller knows when it is truly done.
    bool take_ready(ConversationStoreRead& out);
    // Bounded wait for the next ready result (own thread only: never call it while holding a caller lock that
    // cancel() or demand() needs).  timeout_ms 0 = non-blocking.  This and take_ready() above deliver the next
    // completion of *any* request, so a caller that mounts the image must check request_id/generation or, better, use
    // the request-specific overloads below.
    bool wait_ready(ConversationStoreRead& out, uint64_t timeout_ms);
    // Request-specific delivery: only that read's own completion is ever consumed.  A caller that mounts what it
    // receives must use these - a global take_ready() can hand back another request's completion, and mounting that
    // would restore the wrong history.  False while the request has not settled (and for an unknown request).
    bool take_ready(uint64_t request_id, uint64_t generation, ConversationStoreRead& out);
    bool wait_ready(uint64_t request_id, uint64_t generation, ConversationStoreRead& out, uint64_t timeout_ms);
    // The single lease release: the caller reports that this read is finished AND that every alias of the returned
    // image is gone.  The caller owns that lifetime centrally - one guard per read that fires on success, failure,
    // cancellation and every fast or exception path - because the module cannot see inside a shared alias.  Until it
    // fires, the image's RAM stays charged and no other read may use it.  Safe and counted for unknown or
    // already-released identifiers; a release that never fires leaves the bytes charged, which is a caller bug the
    // module cannot detect - and charging is the fail-safe direction, never a silent refund of live bytes.
    void release_read(uint64_t request_id, uint64_t generation);
    ConversationStoreReadStatus request_status(uint64_t request_id, uint64_t generation) const;
    // Recency promotion after a *confirmed* restore (never on prefetch or on a mere append): moves the head to the
    // protected segment so eviction reclaims other histories first.
    void confirmed_restore(uint64_t head_id);

    // --- publication --------------------------------------------------------------------------------------
    // The image is an owning frozen lease: the store keeps this alias until the publication finishes and the caller
    // keeps its own.  One admitted writer image at a time.  While it is busy, a new publication is admitted only
    // when it is proven to be the *same lineage* as the one in flight (identical `supersedes`, `cover` set and a
    // newer generation): it then replaces it, and the replaced request's result is reported as `coalesced`.  A
    // different branch is never coalesced - it is refused as `refused_busy`, the in-flight predecessor is left
    // untouched and the newer, optional store is dropped as a safe miss.  The call returns immediately.
    bool publish(std::shared_ptr<const SavedConversation> image, const ConversationStorePublish& request,
                 std::string& why);
    bool take_publish(ConversationStorePublishResult& out);
    bool wait_publish(ConversationStorePublishResult& out, uint64_t timeout_ms);
    bool writer_busy() const;

    // --- host admission (RAM reserve handshake) -----------------------------------------------------------
    // Two authoritative numbers in two scopes that must never be mixed: a caller neither subtracts the shared disk
    // quota from a RAM envelope, nor treats a RAM lease as free disk.
    //   ram_held_bytes()       every byte this module currently owns in host memory, including what it holds on a
    //                          caller's behalf: index metadata, admitted reservations that have already allocated,
    //                          materialized read images, the shared image of an in-flight publication, I/O scratch.
    //   ram_lease_bytes()      the handed-out share of it: a read image returned by take_ready()/wait_ready() and not
    //                          yet released, plus the publication image alias.  Charge this to your own envelope; the
    //                          module will not hand those bytes to another read, and never refunds them on its own.
    //   ram_read_budget()      the absolute admission bound in force.
    //   ram_available_bytes()  == max(0, ram_read_budget() - ram_held_bytes()).
    //   disk_charged_bytes()   committed objects and manifests still authorized by the catalog, including bytes
    //                          waiting to be reclaimed (orphans, pending deletions, retired objects - credited only
    //                          after the authority record is durable and the file is gone).
    //   disk_available_bytes() == max(0, disk_quota_bytes() - disk_charged_bytes()).
    //   disk_free_bytes()      last sampled filesystem space under the root, checked against min_free_bytes.
    // suspend_read_admission() stops admitting *new* read work (queued demand stays queued); every byte the module
    // can still allocate was already counted in ram_held_bytes() at admission time, so a caller may observe it while
    // suspended and then publish_read_allowance() without racing an allocation.
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
    // the previous epoch.  Called when the engine discards state for safety (a non-finite/poison detection).  A
    // failed commit leaves the run intent armed: the next start must not reuse anything.
    bool invalidate_domain(const char* reason);
    // Healthy drain: finishes/cancels storage work, then durably clears this domain's run intent.  Only a clean
    // close promises that eligible completed publications survive a restart; the destructor deliberately abandons
    // the intent instead (an unclean exit must not look healthy).
    void close();
    ConversationStoreCounters counters() const;
    // Diagnostics for qualification evidence: one bounded line per stage of the open sequence.
    std::string describe() const;

    // Test-only seams (documented, deterministic): force the next publication to fail at a named step, and count the
    // bytes this module actually read from a payload source.  Empty by default.
    void set_publish_fault_for_test(const char* step);
    uint64_t source_bytes_read_for_test() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace strata::core
