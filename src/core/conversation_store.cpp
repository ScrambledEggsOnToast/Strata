// The durable NVMe conversation store: its disk authority (root, lock, issuer key, catalog, run intent), the shared
// physical quota, the one admitted writer and the bounded read pipeline, and every byte it hands back.
//
// Rules this file keeps, in the order they are enforced:
//   * An object is durable before a manifest names it; a manifest is durable before the catalog authorizes it; the
//     catalog is the only authority.  A file the catalog does not authorize is an accounted orphan, never a
//     resurrected history; a catalog that cannot be trusted means cold caching.
//   * Nothing is deleted until the authority that named it has been removed *durably*, and no byte is credited before
//     the file is confirmed gone.  A failed deletion stays queued, stays charged, and is retried.
//   * Every byte this module creates is charged first, at a conservative allocated size (block-rounded, plus one
//     block for the file's directory entry): objects, manifests, the catalog and its temporary, the issuer key, the
//     run-intent marker and orphans.  Nothing is ever admitted against a promise to delete something later.
//   * The in-memory catalog is only ever replaced by a write that succeeded.  A failed commit leaves the durable
//     authority intact and latches the store faulted instead of continuing from a divergent view.
//   * Every entry point reports an ordinary failure as a *cache* failure; only genuine authority uncertainty or an
//     observed safety fault latches the store.
//
// Threading rules:
//   * The catalog mutex guards state, never I/O.  Every storage wait happens with that mutex released.
//   * One storage thread owns every durable catalog transaction, so a planned commit can never interleave with
//     another.  Callers that need durability hand work to it and wait for it, bounded, without holding the mutex.
//   * Descriptor-relative access only: the root and its three directories are opened once, validated (owner, type,
//     mode) and never re-derived from a path, so no component can be swapped for a symlink underneath us.
#include "strata/core/conversation_store.hpp"

#include "conversation_store_format.hpp"
#include "conversation_store_sha256.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cinttypes>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <dirent.h>
#include <fcntl.h>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <sys/file.h>
#include <sys/random.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

namespace strata::core {
namespace {

using conversation_store_detail::Catalog;
using conversation_store_detail::CatalogDomain;
using conversation_store_detail::CatalogHead;
using conversation_store_detail::ContainerMeta;
using conversation_store_detail::CutRecord;
using conversation_store_detail::Digest;
using conversation_store_detail::digest_nonzero;
using conversation_store_detail::digest_u64;
using conversation_store_detail::kAllocSlackBytes;
using conversation_store_detail::kDraftBoundaryKind;
using conversation_store_detail::kMapNodeBytes;
using conversation_store_detail::ManifestMeta;
using conversation_store_detail::sha256;
using conversation_store_detail::sha256_hex;
using conversation_store_detail::StreamRecord;
using conversation_store_detail::StreamWrite;

constexpr uint64_t kObjectBytesEstimate = 64;  // the RAM one object-table entry costs, at most
constexpr size_t kMaxReaderThreads = 8;
constexpr uint64_t kBlockBytes = 4096;         // conservative allocation granularity of the cache filesystem
constexpr uint64_t kDirectoryCharge = 4096;    // one block per file for its directory entry and inode
constexpr uint64_t kReadSlabBytes = uint64_t(1) << 20;
constexpr uint64_t kReadFixedOverhead = uint64_t(64) << 10;
constexpr uint64_t kJobFixedOverhead = 1024;
// Fixed 256-bit tracker, a stack-only two-character shard name plus NUL, and one transient descriptor.
constexpr uint64_t kShardBatchWorkspace = 4 * sizeof(uint64_t) + 3 * sizeof(char) + sizeof(int);
constexpr uint64_t kMaxScanEntries = uint64_t(1) << 20;
constexpr uint64_t kMaxReclaimPerPass = 65536;
constexpr uint64_t kTempNameAttempts = 32;
constexpr uint32_t kDurableWaitMs = 30000;
constexpr uint64_t kIdlePollMs = 20;
constexpr char kIssuerEnvelopeTag[8] = {'S', 'T', 'C', 'S', 'K', 'E', 'Y', '2'};
constexpr size_t kIssuerEnvelopeBytes = 64;
constexpr uint64_t kDraftCertificationBytes = SavedConversation::draft_metadata_bytes;
// Envelope/checksum preparation, its temporary bookkeeping, and the SHA provider's existing conservative bound.
constexpr uint64_t kReaderFloorWorkspace = kIssuerEnvelopeBytes + 2 * kAllocSlackBytes +
                                           conversation_store_detail::kChainBytesPerImageKey + 1024;

// Existing authority, not a second catalog: legacy readers require exactly 32 bytes and stop in open_root().
// Original key bytes and all derived authorization identities remain unchanged.
std::array<uint8_t, kIssuerEnvelopeBytes> issuer_envelope(const std::array<uint8_t, 32>& key) {
    std::array<uint8_t, kIssuerEnvelopeBytes> out{};
    std::memcpy(out.data(), kIssuerEnvelopeTag, 8);
    out[8] = 2;  // little-endian envelope version
    out[12] = 2; // little-endian minimum reader
    std::memcpy(out.data() + 16, key.data(), key.size());
    const auto checksum = sha256(out.data(), 48);
    std::memcpy(out.data() + 48, checksum.data(), 16);
    return out;
}

uint64_t sat_add(uint64_t a, uint64_t b) { return a > UINT64_MAX - b ? UINT64_MAX : a + b; }
uint64_t sat_mul(uint64_t a, uint64_t b) { return a != 0 && b > UINT64_MAX / a ? UINT64_MAX : a * b; }

// The most one committed change can add to the catalog: a single head's own encoding (its fixed fields plus every cut
// one image can advertise), the publication counter and one allocation's rounding.  A commit adds at most one head,
// and an eviction, promotion or invalidation only shrinks or repeats one, so this is what makes the next catalog
// copy's cost predictable before it is written.
uint64_t catalog_growth_bytes(const ConversationStoreLimits& limits) {
    return 512 + 130 + 4 + (uint64_t(limits.max_checkpoints) + 1) * 42;
}

// The bytes this store will read of one stored manifest: the head's own recorded charge when that is smaller than
// the configured parsing ceiling, never the ceiling alone.  A manifest file's charge is block-rounded and always at
// least its real size, and the writer's reservation is built from those same recorded charges, so a read can never
// need more bytes than the reservation, or the decode workspace, already counted for it.
uint64_t manifest_read_cap(uint64_t recorded, const ConversationStoreLimits& limits) {
    return recorded != 0 && recorded < limits.max_manifest_bytes ? recorded : limits.max_manifest_bytes;
}

// The outcome of one catalog transaction.  `unchanged` means not one durable byte changed and the previously
// committed authority is still the authority; `durable_applied` means the new catalog was renamed into place and
// every directory that changed was flushed; `uncertain` means the rename happened but its durability could not be
// confirmed, so nothing may be deleted, credited or reused until a restart re-reads the authority.
enum class CommitOutcome : uint8_t { unchanged = 0, durable_applied = 1, uncertain = 2 };

// Private, opt-in read diagnostics. A batch belongs to one mutex scope (or one reader iteration), not a background
// logger. Callers declare it before their lock so destruction emits only after unlock. No query, scope, path, digest,
// error detail or image bytes enter a record. IDs are native store identities, not frontend request identities.
// All times are steady-clock nanoseconds. admit_ns owns a peak/slot; start_ns enters read_body; finish_ns is sampled
// after that allocation frame unwinds, before reacquiring the catalog mutex. A missing start means the body has not
// run; a missing finish means it has not unwound. Queue, dispatch, body-lock wait, completion-lock wait and ready
// residence are separate. materialize_ns is the logical body's wall time less its catalog-lock acquisition waits:
// filesystem/page-cache access, hashing, copying and restore work, never a claim of physical disk latency.
uint64_t store_timing_ns() {
    return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now().time_since_epoch()).count());
}

// A read's existing catalog lock, with opt-in acquisition wait accounting. The counter is updated only after the
// mutex is held. This introduces neither another lock nor a diagnostic clock call on the disabled path.
class StoreReadTimedLock {
    uint64_t started_;
    std::lock_guard<std::mutex> guard_;
public:
    StoreReadTimedLock(std::mutex& mutex, bool timing, uint64_t& waited)
        : started_(timing ? store_timing_ns() : 0), guard_(mutex) {
        if (timing) waited += store_timing_ns() - started_;
    }
};

std::atomic<uint64_t> store_read_attempt{0}; // process-wide: a retired descriptor may overlap its replacement

struct StoreReadTrace {
    const char* event;
    const char* reason;
    ConversationStoreReadStatus status;
    uint64_t ts_ns, request_id, generation, attempt_id, previous_attempt_id, head, cut, epoch;
    uint64_t call_ns, call_lock_wait_ns, enqueue_ns, admit_ns, start_ns, finish_ns, ready_ns, ready_end_ns;
    uint64_t body_lock_wait_ns, completion_lock_wait_ns;
    uint64_t peak_bytes, reserved_bytes, lease_bytes, resident_bytes, budget_bytes, prepared_bytes;
    uint64_t preparation_bytes, metadata_limit, incoming, max_reads, queued, descriptors, wait_mask;
    uint64_t demand, prefetched, active, holds_slot, cancelled, io_attempted, bytes_read, omitted_rejections;
    bool head_event;
    uint64_t plan_version, catalog_version, replacement_head, head_leases, pinned, protected_seg, recency;
    uint64_t charged_bytes, free_bytes, quota_bytes, floor_bytes;
    uint64_t live_tokens, stages, live_cut;
};

struct StoreReadTraceBatch {
    // Deliberately fixed and allocation-free. Ordinary operations produce at most a few records per descriptor.
    // An unusually large configured request set reports loss explicitly; an overflow is not evidence of no work.
    StoreReadTrace records[64];
    size_t count = 0;
    uint64_t dropped = 0;
    uint64_t call_ns = 0, call_lock_wait_ns = 0; // this API call only; workers have no caller lock interval
    bool io_allowed = true; // request_shutdown may retire work but must not perform diagnostic writes

    void add(const StoreReadTrace& record) {
        if (count < sizeof records / sizeof records[0]) {
            records[count] = record;
            records[count].call_ns = call_ns;
            records[count].call_lock_wait_ns = call_lock_wait_ns;
            ++count;
        } else ++dropped;
    }
    void emit() {
        if (!io_allowed || (!count && !dropped)) return; // disabled: no clocks, formatting, allocations or I/O
        const int saved_errno = errno;
        for (size_t i = 0; i < count; ++i) {
            const auto& r = records[i];
            if (r.head_event) {
                char line[1024];
                const int length = std::snprintf(line, sizeof line,
                    "store_head event=%s reason=%s state_ns=%" PRIu64 " applied_ns=%" PRIu64
                    " request_id=%" PRIu64 " generation=%" PRIu64
                    " head=%" PRIu64 " cut=%" PRIu64 " epoch=%" PRIu64
                    " live_tokens=%" PRIu64 " stages=%" PRIu64 " live_cut=%" PRIu64
                    " plan_version=%" PRIu64 " catalog_version=%" PRIu64 " replacement_head=%" PRIu64
                    " head_leases=%" PRIu64 " pinned=%" PRIu64 " protected=%" PRIu64 " recency=%" PRIu64
                    " charged_bytes=%" PRIu64 " free_bytes=%" PRIu64 " quota_bytes=%" PRIu64 " floor_bytes=%" PRIu64 "\n",
                    r.event, r.reason, r.ts_ns, r.finish_ns, r.request_id, r.generation, r.head, r.cut, r.epoch,
                    r.live_tokens, r.stages, r.live_cut,
                    r.plan_version, r.catalog_version, r.replacement_head, r.head_leases, r.pinned, r.protected_seg,
                    r.recency, r.charged_bytes, r.free_bytes, r.quota_bytes, r.floor_bytes);
                if (length > 0 && size_t(length) < sizeof line) write_line(line, size_t(length));
                continue;
            }
            const uint64_t queue_ns = r.admit_ns ? r.admit_ns - r.enqueue_ns : 0;
            const uint64_t dispatch_ns = r.admit_ns && (r.start_ns || r.finish_ns)
                ? (r.start_ns ? r.start_ns : r.finish_ns) - r.admit_ns : 0;
            const uint64_t read_ns = r.finish_ns && r.start_ns ? r.finish_ns - r.start_ns : 0;
            const uint64_t materialize_ns = read_ns - std::min(read_ns, r.body_lock_wait_ns);
            const uint64_t residence = r.ready_ns ? (r.ready_end_ns ? r.ready_end_ns : r.ts_ns) - r.ready_ns : 0;
            char line[2048];
            const int length = std::snprintf(line, sizeof line,
                "store_read event=%s reason=%s status=%s ts_ns=%" PRIu64
                " request_id=%" PRIu64 " generation=%" PRIu64 " attempt_id=%" PRIu64
                " previous_attempt_id=%" PRIu64 " head=%" PRIu64 " cut=%" PRIu64 " epoch=%" PRIu64
                " call_ns=%" PRIu64 " call_lock_wait_ns=%" PRIu64
                " enqueue_ns=%" PRIu64 " admit_ns=%" PRIu64 " start_ns=%" PRIu64
                " finish_ns=%" PRIu64 " ready_ns=%" PRIu64 " ready_end_ns=%" PRIu64
                " queue_ns=%" PRIu64 " dispatch_ns=%" PRIu64 " read_ns=%" PRIu64
                " body_lock_wait_ns=%" PRIu64 " materialize_ns=%" PRIu64 " completion_lock_wait_ns=%" PRIu64
                " ready_residence_ns=%" PRIu64
                " peak_bytes=%" PRIu64 " reserved_bytes=%" PRIu64 " lease_bytes=%" PRIu64
                " resident_bytes=%" PRIu64 " budget_bytes=%" PRIu64 " prepared_bytes=%" PRIu64
                " preparation_bytes=%" PRIu64 " metadata_limit=%" PRIu64 " incoming=%" PRIu64
                " max_reads=%" PRIu64 " queued=%" PRIu64 " descriptors=%" PRIu64 " wait_mask=%" PRIu64
                " demand=%" PRIu64 " origin=%s active=%" PRIu64 " holds_slot=%" PRIu64 " cancelled=%" PRIu64
                " io_attempted=%" PRIu64 " bytes_read=%" PRIu64 " omitted_rejections=%" PRIu64 "\n",
                r.event, r.reason, conversation_store_read_status_name(r.status), r.ts_ns,
                r.request_id, r.generation, r.attempt_id, r.previous_attempt_id, r.head, r.cut, r.epoch,
                r.call_ns, r.call_lock_wait_ns, r.enqueue_ns, r.admit_ns, r.start_ns, r.finish_ns, r.ready_ns, r.ready_end_ns,
                queue_ns, dispatch_ns, read_ns, r.body_lock_wait_ns, materialize_ns, r.completion_lock_wait_ns, residence,
                r.peak_bytes, r.reserved_bytes, r.lease_bytes, r.resident_bytes, r.budget_bytes, r.prepared_bytes,
                r.preparation_bytes, r.metadata_limit, r.incoming, r.max_reads, r.queued, r.descriptors, r.wait_mask,
                r.demand, r.prefetched ? "prefetch" : "demand", r.active, r.holds_slot, r.cancelled,
                r.io_attempted, r.bytes_read, r.omitted_rejections);
            if (length > 0 && size_t(length) < sizeof line) write_line(line, size_t(length));
        }
        if (dropped) {
            char line[128];
            const int length = std::snprintf(line, sizeof line, "store_read event=trace_overflow dropped=%" PRIu64 "\n",
                                             dropped);
            if (length > 0 && size_t(length) < sizeof line) write_line(line, size_t(length));
        }
        count = 0;
        dropped = 0;
        errno = saved_errno;
    }
    void flush(std::unique_lock<std::mutex>& lock) {
        if (!count && !dropped) return;
        lock.unlock();
        emit();
        lock.lock();
    }
    ~StoreReadTraceBatch() { emit(); }

private:
    static void write_line(const char* line, size_t bytes) {
        while (bytes) {
            const ssize_t written = ::write(STDERR_FILENO, line, bytes);
            if (written < 0 && errno == EINTR) continue;
            if (written <= 0) return;
            line += written;
            bytes -= size_t(written);
        }
    }
};

// ---------------------------------------------------------------------------------------------------------------
// POSIX helpers.  Each is exact or fails with a reason, none throws, and all are descriptor relative: no path is
// ever re-walked, so no component can be replaced between two calls.
// ---------------------------------------------------------------------------------------------------------------

bool write_all(int fd, const void* data, size_t size, std::string& why) {
    const uint8_t* at = static_cast<const uint8_t*>(data);
    size_t left = size;
    while (left) {
        const ssize_t n = ::write(fd, at, left);
        if (n <= 0) {
            if (n < 0 && errno == EINTR) continue;
            why = std::string("write failed: ") + std::strerror(errno);
            return false;
        }
        at += size_t(n);
        left -= size_t(n);
    }
    return true;
}

// Where a file the store may have to delete lives.  `root` is the catalog, the issuer key, the run-intent marker or
// the lock file; `object` names one content address inside a two-hex-digit shard.
enum class Where : uint8_t { root = 0, objects = 1, manifests = 2, tmp = 3 };

struct PendingFile {
    Where where = Where::tmp;
    Digest object{};        // object identity when `where` is objects: the refcount is rechecked before removal
    std::string shard;      // two hex digits, when `where` is objects
    std::string name;
    uint64_t charge = 0;    // this file's share of the shared quota, credited only once it is confirmed gone
    uint64_t head_id = 0;   // the retired head it belonged to; 0 never defers
};

constexpr uint64_t kPendingMetadataBytes = sizeof(PendingFile) + 2 * kAllocSlackBytes;

// Retired selected heads can outlive their publisher. Their existing index nodes and eventual retire records must
// remain counted without relying on that writer's transient reservation; moving a last lease into the retire queue
// consumes this reserved metadata rather than allocating an uncharged second lifetime.
uint64_t head_hold_metadata_bytes(size_t objects) {
    return sat_add(3 * kMapNodeBytes + kPendingMetadataBytes,
                   sat_mul(uint64_t(objects), sizeof(Digest) + kPendingMetadataBytes));
}

// Every host byte one admitted publication can hold at once, counted before any of it is allocated.  Each field is an
// upper bound on one phase's structures; `peak` is the largest sum of them that can be live together, so the writer
// never allocates beyond what it reserved (see publish() and publish_body()).
struct WriterWorkspace {
    uint64_t request_copy = 0;    // the copied ConversationStorePublish and its reuse intervals
    uint64_t layout = 0;          // the layout skeleton, its per-record prefix chains and the identity set
    uint64_t donor_decoded = 0;   // every manifest this attempt decodes: its raw bytes, its decoded metadata and its
                                  // consistency tables, one read and decode at a time (see DonorPrep::decoded)
    uint64_t chunks = 0;          // the most immutable chunks it can emit; every one is assumed unique
    uint64_t buffer = 0;          // the largest single chunk it reads into one buffer
    uint64_t manifest = 0;        // its own encoded manifest
    uint64_t catalog_apply = 0;   // the candidate catalog copy, the apply staging and the retire queue's room
    uint64_t shard_batch = 0;     // opt-in fixed shard tracker and sequential drain workspace

    // Every structure that is live while one chunk is read and written, kept apart from the manifest and the catalog
    // plan because those overlap with it rather than replace it: the chunk records, the referenced and charged object
    // maps' nodes, the new-object map's nodes, the object-lease map's nodes, the attempt's own lease list, the
    // persistent index's entries, the head's own object-list slots and the pending-file entries a write adds.
    uint64_t chunk_state() const {
        uint64_t state = sat_mul(chunks, sizeof(conversation_store_detail::ChunkRecord) + kAllocSlackBytes);
        // Every prospective chunk may be a distinct object.  An object this attempt *creates* spends a node in each
        // of the two new-object maps beside its referenced-map node; an object it merely *names* - a deduplicated
        // chunk, an adopted donor interval or a file revived from an accounted orphan - spends no new-object node and
        // charges nothing, and holds its place in the object-lease map and in this attempt's own lease list instead
        // (see reclaim_pending()).  A chunk is one or the other, never both, so the four-node and two-Digest budget
        // below already covers either side of that choice: at most two of the nodes are ever spent on one object's
        // write phase, and the persistent index contributes the last.
        state = sat_add(state, sat_mul(chunks, 4 * conversation_store_detail::kMapNodeBytes + 2 * sizeof(Digest) + 40 +
                                                 2 * kAllocSlackBytes));
        // Every object this attempt writes is accounted before it is written, by name and by shard.
        state = sat_add(state, sat_mul(chunks, sizeof(PendingFile) + 2 * kAllocSlackBytes));
        return sat_add(state, 2 * kAllocSlackBytes);   // the per-stream chunk reservations
    }

    // I + J + chunk_state + max(B, W_manifest/catalog/apply).  The chunk, map, orphan and index state is *retained*
    // while the manifest is encoded and the catalog transaction is planned, so it is added rather than maximized
    // away; the single chunk buffer and the manifest/apply phase do not coexist.
    uint64_t peak() const {
        const uint64_t base = sat_add(sat_add(request_copy, layout), donor_decoded);
        return sat_add(sat_add(sat_add(base, shard_batch), chunk_state()),
                       std::max(buffer, sat_add(manifest, catalog_apply)));
    }
};

// The bounded preparation one publication's parent and reuse donors need.  Built from the named heads' own recorded
// charges - never from head_limit x max_manifest_bytes and never from a default maximum - and computed from catalog
// metadata alone, with no I/O and no allocation.
struct DonorPrep {
    // Every manifest this attempt decodes, summed over the heads it names: the whole read+decode workspace of one
    // manifest (its raw bytes, its decoded metadata and the tables the consistency pass builds), so the reservation
    // covers the transient peak of a decode that has not happened yet.  The heads are decoded one at a time and their
    // metadata is retained, so this sum is never below what is really held.
    uint64_t decoded = 0;
    uint64_t chunks = 0;      // the most chunks those donors can contribute through reuse intervals, per donor
                              // whole-manifest bound times that donor's heaviest single-stream interval count
};

// The shared charge of one file: its blocks plus one block for its directory entry (and inode).  Deliberately
// conservative: the quota stops the store before the device does, so over-estimating is the safe direction.
uint64_t allocated_charge(uint64_t size) {
    return (size + kBlockBytes - 1) / kBlockBytes * kBlockBytes + kDirectoryCharge;
}

bool private_owner(const struct stat& info) { return info.st_uid == ::geteuid(); }
bool private_directory(const struct stat& info) {
    return S_ISDIR(info.st_mode) && private_owner(info) && (info.st_mode & 0077) == 0;
}
bool private_regular(const struct stat& info) {
    return S_ISREG(info.st_mode) && private_owner(info) && (info.st_mode & 0077) == 0;
}

bool is_hex_name(const std::string& name) {
    if (name.empty()) return false;
    for (char c : name)
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
    return true;
}

struct RootDirs {
    int root = -1, objects = -1, manifests = -1, tmp = -1;

    ~RootDirs() { close_all(); }
    void close_all() {
        for (int* fd : {&root, &objects, &manifests, &tmp}) {
            if (*fd >= 0) ::close(*fd);
            *fd = -1;
        }
    }
    int of(Where where) const {
        switch (where) {
            case Where::root: return root;
            case Where::objects: return objects;
            case Where::manifests: return manifests;
            case Where::tmp: return tmp;
        }
        return -1;
    }
};

int open_child_dir(int parent, const char* name, std::string& why, bool private_leaf = true) {
    const int fd = ::openat(parent, name, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (fd < 0) {
        why = std::string("cannot open the directory ") + name + ": " + std::strerror(errno);
        return -1;
    }
    struct stat info {};
    const bool inspected = ::fstat(fd, &info) == 0;
    const bool trusted_ancestor = inspected && S_ISDIR(info.st_mode) &&
        (info.st_uid == 0 || private_owner(info)) &&
        ((info.st_mode & 0022) == 0 || (info.st_uid == 0 && (info.st_mode & S_ISVTX)));
    if (!inspected || (private_leaf ? !private_directory(info) : !trusted_ancestor)) {
        why = std::string("the directory ") + name + " has unsafe ownership or permissions";
        ::close(fd);
        errno = EPERM;
        return -1;
    }
    return fd;
}

bool make_child_dir(int parent, const char* name, bool& created, std::string& why) {
    created = false;
    if (::mkdirat(parent, name, 0700) == 0) {
        created = true;
        return true;
    }
    if (errno != EEXIST) {
        why = std::string("cannot create ") + name + ": " + std::strerror(errno);
        return false;
    }
    struct stat info {};
    if (::fstatat(parent, name, &info, AT_SYMLINK_NOFOLLOW) != 0 || !private_directory(info)) {
        why = std::string(name) + " exists and is not a private directory this account owns";
        return false;
    }
    return true;
}

bool fsync_dir(int fd, const char* label, std::string& why) {
    if (::fsync(fd) != 0) {
        why = std::string("directory fsync failed for ") + label + ": " + std::strerror(errno);
        return false;
    }
    return true;
}

// Creates a fresh file exclusively (O_EXCL), so a name collision can never truncate another file, and flushes its
// data before it can be renamed into place.
bool write_new_file(int dir, const std::string& name, const void* data, size_t size, std::string& why) {
    const int fd = ::openat(dir, name.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
    if (fd < 0) {
        why = std::string("cannot create ") + name + ": " + std::strerror(errno);
        return false;
    }
    bool ok = write_all(fd, data, size, why);
    if (ok && ::fsync(fd) != 0) {
        why = std::string("fsync failed for ") + name + ": " + std::strerror(errno);
        ok = false;
    }
    if (::close(fd) != 0 && ok) {
        why = std::string("close failed for ") + name + ": " + std::strerror(errno);
        ok = false;
    }
    return ok;
}

// Reads at most `limit` bytes of a regular file this account owns.  The size is checked from the open descriptor
// before anything is allocated, so a declared length never becomes an allocation.
bool read_file(int dir, const std::string& name, uint64_t limit, std::vector<uint8_t>& out, std::string& why) {
    const int fd = ::openat(dir, name.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) {
        why = std::string("cannot open ") + name + ": " + std::strerror(errno);
        return false;
    }
    struct stat info {};
    if (::fstat(fd, &info) != 0 || !private_regular(info)) {
        why = name + " is not a regular file this account owns";
        ::close(fd);
        return false;
    }
    if (info.st_size < 0 || uint64_t(info.st_size) > limit) {
        why = name + " is larger than the limit this store admits";
        ::close(fd);
        return false;
    }
    out.resize(size_t(info.st_size));
    size_t at = 0;
    while (at < out.size()) {
        const ssize_t n = ::read(fd, out.data() + at, out.size() - at);
        if (n < 0) {
            if (errno == EINTR) continue;
            why = std::string("read failed for ") + name + ": " + std::strerror(errno);
            ::close(fd);
            return false;
        }
        if (n == 0) break;
        at += size_t(n);
    }
    ::close(fd);
    if (at != out.size()) {
        why = name + " shrank while it was read";
        return false;
    }
    return true;
}

bool remove_file(int dir, const std::string& name, std::string& why) {
    if (dir < 0) {
        why = "the store does not own the directory this file lives in";
        return false;
    }
    if (::unlinkat(dir, name.c_str(), 0) == 0) return true;
    if (errno == ENOENT) return true;   // already gone: nothing to credit beyond what was charged
    why = std::string("cannot remove ") + name + ": " + std::strerror(errno);
    return false;
}

struct DirEntry {
    std::string name;
    bool directory = false;
    bool regular = false;
    bool ours = false;
    uint64_t size = 0;
};

// One bounded listing: the entry count is bounded *while* reading, so a directory full of junk cannot become an
// unbounded allocation before a bound is checked.
bool list_entries(int dir, std::vector<DirEntry>& out, uint64_t limit, std::string& why) {
    out.clear();
    // dup() shares the directory cursor and would make subsequent scans miss entries.
    const int dup_fd = ::openat(dir, ".", O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (dup_fd < 0) {
        why = std::string("cannot reopen a directory for scanning: ") + std::strerror(errno);
        return false;
    }
    DIR* stream = ::fdopendir(dup_fd);
    if (!stream) {
        why = std::string("cannot list a directory: ") + std::strerror(errno);
        ::close(dup_fd);
        return false;
    }
    while (struct dirent* entry = ::readdir(stream)) {
        const std::string name = entry->d_name;
        if (name == "." || name == "..") continue;
        if (out.size() >= limit) {
            why = "a store directory holds more entries than this store accounts for";
            ::closedir(stream);
            return false;
        }
        DirEntry item;
        item.name = name;
        struct stat info {};
        if (::fstatat(dir, name.c_str(), &info, AT_SYMLINK_NOFOLLOW) == 0) {
            item.directory = S_ISDIR(info.st_mode) != 0;
            item.regular = S_ISREG(info.st_mode) != 0;
            item.ours = private_owner(info) && !S_ISLNK(info.st_mode);
            item.size = info.st_size > 0 ? uint64_t(info.st_size) : 0;
        }
        out.push_back(std::move(item));
    }
    ::closedir(stream);
    return true;
}

bool fill_random(uint8_t* out, size_t size) {
    size_t at = 0;
    while (at < size) {
        const ssize_t n = ::getrandom(out + at, size - at, 0);
        if (n > 0) {
            at += size_t(n);
            continue;
        }
        if (n < 0 && errno == EINTR) continue;
        break;
    }
    if (at != size) {
        const int fd = ::open("/dev/urandom", O_RDONLY | O_CLOEXEC);
        if (fd >= 0) {
            while (at < size) {
                const ssize_t n = ::read(fd, out + at, size - at);
                if (n > 0) {
                    at += size_t(n);
                    continue;
                }
                if (n < 0 && errno == EINTR) continue;
                break;
            }
            ::close(fd);
        }
    }
    return at == size;
}

uint64_t filesystem_free_bytes(int root_fd) {
    struct statvfs info {};
    if (::fstatvfs(root_fd, &info) != 0) return 0;
    return uint64_t(info.f_bavail) * uint64_t(info.f_frsize);
}

bool hex_to_digest(const std::string& text, Digest& out) {
    return conversation_store_detail::sha256_hex_parse(text, out);
}

// One object shard, opened under `objects/` without ever following a symlink at either step.
int open_shard_fd(const RootDirs& dirs, const std::string& shard, std::string& why) {
    return open_child_dir(dirs.objects, shard.c_str(), why);
}

struct ShardFd {
    int fd;
    explicit ShardFd(int value) : fd(value) {}
    ShardFd(const ShardFd&) = delete;
    ShardFd& operator=(const ShardFd&) = delete;
    ~ShardFd() { if (fd >= 0) ::close(fd); }
};

// Own the current validated object descriptor through its synchronous hash/copy, including error exits.
struct ReadObject {
    int fd = -1;
    ReadObject() = default;
    ReadObject(const ReadObject&) = delete;
    ReadObject& operator=(const ReadObject&) = delete;
    ~ReadObject() { if (fd >= 0) ::close(fd); }
};

bool open_read_object(const RootDirs& dirs, const Digest& object, uint64_t size, ReadObject& out,
                      std::string& why) {
    const std::string name = sha256_hex(object);
    const int shard = open_shard_fd(dirs, name.substr(0, 2), why);
    if (shard < 0) return false;
    const int fd = ::openat(shard, name.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    const int open_error = errno;
    ::close(shard);
    if (fd < 0) {
        why = std::string("cannot open an object: ") + std::strerror(open_error);
        return false;
    }
    struct stat info {};
    if (::fstat(fd, &info) != 0 || !S_ISREG(info.st_mode) || uint64_t(info.st_size) != size) {
        why = "an object's size does not match the manifest";
        ::close(fd);
        return false;
    }
    out.fd = fd;
    return true;
}

// Hash and restore from the same validated descriptor. No image is delivered until every object digest matches.
bool hash_and_copy_object(int fd, uint64_t size, conversation_store_detail::Sha256& hash,
                          StreamWrite& write, uint64_t offset, std::vector<uint8_t>& slab, Digest& out,
                          std::string& why) {
    uint64_t left = size;
    while (left) {
        const ssize_t n = ::read(fd, slab.data(), size_t(std::min<uint64_t>(slab.size(), left)));
        if (n < 0) {
            if (errno == EINTR) continue;
            why = std::string("cannot read an object: ") + std::strerror(errno);
            return false;
        }
        if (n == 0) {
            why = "an object is shorter than the manifest says";
            return false;
        }
        hash.update(slab.data(), size_t(n));
        if (!write(offset + size - left, slab.data(), size_t(n), why)) {
            return false;
        }
        left -= uint64_t(n);
    }
    out = hash.digest();
    return true;
}

// ---------------------------------------------------------------------------------------------------------------
// Exact prefix confirmation of a materialized image
// ---------------------------------------------------------------------------------------------------------------

// Does this record hold exactly the request's own token prefix, and does its own prefix chain reproduce the
// advertised digest?  The chain is rebuilt from the *restored* bytes, so an advertisement can never be satisfied by
// metadata alone.
bool record_holds_cut(const ConversationCheckpoint& record, const ConversationStoreQuery& query, uint64_t tokens,
                      const Digest& seed, Digest& digest) {
    if (record.ids.size() != tokens || tokens > query.tokens.size()) return false;
    if (tokens && std::memcmp(record.ids.data(), query.tokens.data(), size_t(tokens) * sizeof(int32_t)) != 0)
        return false;
    const conversation_store_detail::PrefixChain chain(seed, record.ids.data(), record.ids.size(), record.imgs);
    return chain.digest(tokens, digest);
}

bool restored_cut_matches(const SavedConversation& image, const conversation_store_detail::ManifestMeta& meta,
                          const ConversationStoreQuery& query, uint64_t tokens, bool live, const Digest& advertised,
                          std::string& why) {
    Digest digest{};
    const ConversationCheckpoint* selected = nullptr;
    if (live) {
        if (record_holds_cut(image.live, query, tokens, meta.bucket, digest)) selected = &image.live;
    } else {
        for (const auto& checkpoint : image.checkpoints)
            if (record_holds_cut(checkpoint, query, tokens, meta.bucket, digest)) {
                selected = &checkpoint;
                break;
            }
    }
    if (!selected || !(digest == advertised)) {
        why = "the materialized image does not reproduce the cut the manifest advertised";
        return false;
    }
    // A split image carries the later stages in one of two encodings, and a cut is only coherent when the encoding it
    // uses names exactly one part per later stage.  A record-container image keeps the parts on the record itself
    // (`stage_parts`); the native parked split leaves a primary record's `stage_parts` empty and carries each part as
    // that stage's own image record, which the stage loop below confirms holds this exact prefix.
    // Live records always use the per-stage image representation rather than embedded checkpoint parts.
    if (!selected->stage_parts.empty() && (live || selected->stage_parts.size() + 1 != meta.stages)) {
        why = "the selected cut has an inconsistent stage-part count";
        return false;
    }
    for (const auto& part : selected->stage_parts) {
        Digest part_digest{};
        if (!record_holds_cut(part, query, tokens, meta.bucket, part_digest) || part_digest != advertised) {
            why = "a selected checkpoint part does not hold the restored prefix";
            return false;
        }
    }
    // A cut is only ever advertised when every later stage can be placed at that same length; the restored stages
    // must really carry that same prefix, byte for byte, or the image is not a coherent all-stage restore.
    for (const auto& stage : image.stage_images) {
        if (stage.scope != image.scope || stage.cvec != image.cvec) {
            why = "a later stage belongs to a different scope or steering mode";
            return false;
        }
        bool found = false;
        Digest stage_digest{};
        if (record_holds_cut(stage.live, query, tokens, meta.bucket, stage_digest) &&
            stage_digest == advertised) found = true;
        for (const auto& checkpoint : stage.checkpoints)
            if (!found && record_holds_cut(checkpoint, query, tokens, meta.bucket, stage_digest) &&
                stage_digest == advertised) found = true;
        if (!found) {
            why = "a later stage does not hold the restored prefix";
            return false;
        }
    }
    return true;
}

}  // namespace

// ---------------------------------------------------------------------------------------------------------------
// The implementation
// ---------------------------------------------------------------------------------------------------------------

struct ConversationStore::Impl {
    // ---- configuration (immutable after construction) ----
    ConversationStoreOptions options;
    ConversationStoreLimits limits;
    // The figures a record must satisfy to be *kept* and indexed, as opposed to the ones `limits` puts on creating
    // and serving one (see conversation_store_detail::retention_limits): a root is shared by every profile that opens
    // it, so a record another profile captured is retained, accounted and left on disk.  They come from the format's
    // own representability bounds and the configured parsing ceilings - never from the physical quota, which records
    // as *logical* bytes can exceed because identical chunks share one physical object.  Recovery, the publication
    // planner and the stream-table diagnostic decode under these; every read and publication is admitted against
    // `limits` itself, and a record this run may not serve is classified there with manifest_within_limits().
    ConversationStoreLimits retained_limits;
    std::string root;                 // diagnostics only: every access goes through `dirs`
    RootDirs dirs;
    uint32_t reader_threads = 1;
    bool read_timing = false;          // STRATA_STORE_TIMING=1, sampled once before workers start
    bool dir_fsync_batch = false;     // STRATA_STORE_DIR_FSYNC_BATCH=1, sampled before workers start

    // ---- authority ----
    std::mutex mutex;
    std::condition_variable cv_ready;
    std::condition_variable cv_publish;
    std::condition_variable cv_work;
    std::condition_variable cv_durable;
    std::condition_variable cv_shutdown;
    std::array<uint8_t, 32> issuer_key{};
    bool certified_reader_floor = false; // sticky existing issuer authority, never reset by catalog transactions
    Catalog catalog;
    uint64_t catalog_charge = 0;
    uint64_t epoch = 0;
    uint64_t catalog_version = 0;      // bumped by every applied commit
    uint32_t domain_index = 0;
    bool intent_active = false;
    bool on = false;
    ConversationStoreShutdownState shutdown = ConversationStoreShutdownState::none;
    uint64_t finalize_before_ms = 0;    // steady-clock epoch; repeated requests only tighten, 0 forbids finalization
    bool finishing = false;            // exactly one caller owns worker joins and finalization
    bool faulted = false;              // latched for the life of this instance
    bool initialized = false;          // the open sequence completed and the authority is trustworthy
    bool lock_owned = false;           // this instance holds the exclusive root lock
    bool unclean_start = false;        // this domain's previous run did not close cleanly: retire its records
    std::string reason;
    std::string fault_reason;
    int lock_fd = -1;

    // ---- disk accounting (unique physical bytes) ----
    struct ObjectState {
        uint64_t bytes = 0;            // its conservative allocated charge
        uint32_t refs = 0;
    };
    std::map<Digest, ObjectState> objects;                  // catalog and request-held retired object references
    std::map<uint64_t, std::vector<Digest>> head_objects;   // retained or selected retired head -> referenced objects
    // In-flight publication leases: every object a publication will *reference without creating* is pinned here from
    // the instant the publication finds it until that transaction settles.  A deduplicated chunk is usually owned by
    // some retained head, and make_room() may evict that head - its sole owner - while the publication that names it
    // is still writing.  Reclamation treats a leased object as referenced (see reclaim_pending()), which is what
    // keeps a file this publication will name from being deleted and credited before it commits.
    struct ObjectLease {
        uint64_t charge = 0;    // the object's own recorded charge, credited only once the file is confirmed gone
    };
    std::map<Digest, ObjectLease> object_leases;
    std::map<uint64_t, uint64_t> head_manifest_bytes;       // head id -> its manifest's allocated charge
    std::map<uint64_t, uint32_t> head_leases;               // selected metadata holds, through image owner release
    std::map<uint64_t, uint64_t> head_recency;              // RAM-only eviction recency (published order is fallback)
    uint64_t recency_counter = 0;
    std::set<uint64_t> eviction_excluded;                   // heads protected by the in-flight publication
    std::vector<PendingFile> pending_files;                 // orphans and retired files awaiting confirmed deletion
    bool reclaim_pending_work = false;                     // mutex-guarded idle writer hint, not another work queue
    uint64_t charged = 0;         // allocated bytes of every file this module owns that still exists
    uint64_t reclaimable = 0;     // the part of `charged` no longer referenced but still on disk
    uint64_t free_sample = 0;     // last sampled filesystem space under the root
    uint64_t metadata = 0;        // estimated RAM of the in-RAM index
    uint64_t prep = 0;            // the live read descriptors' own query copies
    uint64_t decode = 0;          // the live manifest read+decode workspace charged to the preparation ceiling (see
                                  // reserve_decode_locked); a read's and a writer's own decodes are charged to their
                                  // reservations instead, so this is only ever a path that has no reservation
    uint64_t read_prep = 0;       // the live read descriptors' own preparation beyond `prep` - the manifest read and
                                  // decode each admitted read reserved, plus the workspace it grew into while it
                                  // restored.  It is a *view* of bytes the read's RAM reservation already holds, not
                                  // another RAM charge: it is what makes readers, the writer and the diagnostic
                                  // decodes share one preparation total (see preparation_locked())

    // ---- RAM ledger ----
    uint64_t ram_held = 0;        // admitted read peaks, materialized images, the publication alias, scratch
    uint64_t ram_lease = 0;       // the handed-out (or publication) share of it
    uint64_t ram_budget = 0;      // the absolute published allowance; 0 = none published yet
    bool admission_suspended = false;

    // ---- reads ----
    struct ReadJob {
        ConversationStoreQuery query;
        ConversationStoreReadStatus status = ConversationStoreReadStatus::none;
        bool demand = false;
        bool prefetched = false;          // immutable origin of this attempt; demand adoption does not relabel it
        bool refresh_on_completion = false; // one demand-time catalog recheck after an active prefetch unwinds
        bool active = false;              // a reader thread is materializing it right now
        std::atomic<bool> cancelled{false};
        bool speculation_revoked = false; // internal eviction is retryable demand, unlike caller cancellation
        bool budget_deferred = false;    // accepted speculation, not an initial admission refusal
        uint64_t deferred_available = 0;  // last usable capacity; retry only on a meaningful recovery
        bool released = false;
        bool handed = false;
        bool leased = false;              // this job holds a slot in head_leases
        bool holds_slot = false;          // one of Options::max_reads incoming-image slots
        uint64_t reserved_bytes = 0;      // the peak charged to ram_held while it is not yet a lease
        uint64_t lease_bytes = 0;         // the finished image's charge, refunded by release_read()
        uint64_t peak_bytes = 0;          // the peak its admission reserved
        uint64_t prep_bytes = 0;          // its own query copy, charged to `prep`
        uint64_t prepared_bytes = 0;      // the part of `reserved_bytes` that is this read's own preparation - its
                                          // manifest read and decode and the scaffolding it restores into - charged to
                                          // `read_prep` while the read is live
        uint64_t epoch_at_admission = 0;
        uint64_t head_id = 0;
        uint64_t tokens = 0;
        uint64_t published_order = 0;    // equal-cut rebinds may only move to a newer publication
        Digest manifest_id{};             // authority captured atomically with the selected-head lease
        uint64_t manifest_bytes = 0;
        bool live = false;
        uint32_t stages = 0;
        uint64_t payload_bytes = 0;
        double materialize_ms = 0;
        ConversationStoreReadProgress progress; // fixed-size counters, protected by the existing catalog mutex
        std::shared_ptr<const SavedConversation> image;
        std::string detail;
        // Diagnostic scalars only. They do not participate in cancellation, admission or accounting.
        uint64_t trace_attempt = 0, trace_previous = 0;
        uint64_t trace_enqueue = 0, trace_admit = 0, trace_start = 0, trace_finish = 0, trace_ready = 0;
        uint64_t trace_ready_end = 0; // adoption/retirement ends ready residence; caller lease time is not ready time
        uint64_t trace_body_lock_wait = 0, trace_completion_lock_wait = 0;
        uint64_t trace_wait_mask = UINT64_MAX;
        const char* trace_retire_reason = "none";
        bool trace_io_attempted = false;
        uint64_t trace_bytes_read = 0;
    };
    static_assert(sizeof(ReadJob) <= kJobFixedOverhead, "read progress must fit the admitted descriptor overhead");
    std::vector<std::shared_ptr<ReadJob>> jobs;      // queued, active, ready or handed out (bounded)
    std::deque<std::shared_ptr<ReadJob>> queue;      // waiting for the allowance and a slot (demand before prefetch)
    std::deque<std::shared_ptr<ReadJob>> ready;      // settled, waiting for take_ready()
    uint64_t incoming = 0;                           // jobs holding one of Options::max_reads slots
#ifdef STRATA_STORE_TEST_HOOKS
    void (*read_gate)(void*) = nullptr;
    void* read_gate_context = nullptr;
    void (*read_progress_gate)(void*) = nullptr;
    void* read_progress_gate_context = nullptr;
    void (*catalog_gate)(void*, bool) = nullptr;
    void* catalog_gate_context = nullptr;
    void (*object_batch_gate)(void*) = nullptr;
    void* object_batch_gate_context = nullptr;
#endif

    // ---- publication ----
    struct PublishJob {
        std::shared_ptr<const SavedConversation> image;
        ConversationStorePublish request;
        uint64_t epoch = 0;               // the capture epoch: bound at the seam, or at publish() admission
        uint64_t ram_bytes = 0;           // the image alias: charged to ram_held *and* ram_lease
        uint64_t workspace_bytes = 0;     // the writer's own counted workspace: charged to ram_held only
        uint64_t catalog_apply_bytes = 0; // the apply structures this attempt may copy (counted at admission)
        // The part of this image's lease the caller had already accounted for outside this store's allowance.  It
        // widens *this* publication's own fit check and nothing else: reads are never admitted against it, and the
        // ledger still charges the full image and the whole counted workspace.
        uint64_t external_credit_bytes = 0;
        conversation_store_detail::ImageShape shape;   // the allocation-free count this admission was charged for
        bool started = false;             // the writer has begun reading this image: it may never be replaced
        bool cancelled = false;
        bool coalesced = false;
        // The catalog transaction renamed a new catalog into place but could not confirm its durability.  Every file
        // this attempt created may be referenced by either catalog, so none of them may be deleted.
        bool retain_files = false;
        std::vector<PendingFile> orphans; // files this attempt has created and not yet authorized
        // The objects this attempt references without creating, in the order it took their leases: released once the
        // transaction settles, so an abandoned attempt's objects become reclaimable again.
        std::vector<Digest> leased_objects;
    };
    std::shared_ptr<PublishJob> publishing;
    // publish() reserves one result slot per admitted job, including a coalesced predecessor. Finished results plus
    // the active writer never exceed max_pending_publishes * 4 + 4; only harvesting frees a slot.
    std::deque<ConversationStorePublishResult> publish_done;

    struct CommitPlan {
        enum class Kind : uint8_t { publish, eviction, invalidate, promote };
        Kind kind = Kind::promote;
        Catalog candidate;                        // exact for publish/eviction; re-derived at commit otherwise
        uint64_t version = 0;                     // the catalog version an exact candidate was planned against
        uint64_t recency_at_copy = 0;             // preserves policy hints newer than this immutable snapshot
        std::vector<uint64_t> retiring;           // heads the applied catalog drops
        uint64_t new_head = 0;                    // publish only
        const char* eviction_reason = "space";
        uint64_t replaced_head = 0;               // diagnostics: distinguish a covered replacement from head pressure
        uint64_t request_id = 0, generation = 0, live_tokens = 0; // publication diagnostics, never authority
        uint64_t applied_ns = 0, applied_catalog_version = 0; // exact successful visibility boundary
        bool defer_reclaim = false;               // final catalog: reclaim only after proven intent removal
        std::vector<Digest> new_objects;          // publish only: everything the new manifest references
        std::map<Digest, uint64_t> new_object_charges;   // the ones this publication created (already charged)
        uint64_t new_manifest_bytes = 0;
        uint64_t new_catalog_charge = 0;          // filled in by commit_plan
        uint64_t epoch = 0;                       // invalidate: the advanced epoch
        uint32_t domain_index = 0;                // invalidate
        // Everything the apply after the rename needs, built *before* it: fresh object-index nodes (merged without
        // allocating) and the head's own index entries.  A commit that cannot allocate these changes nothing.
        std::map<Digest, ObjectState> staged_objects;
        bool staged_head_objects = false;
        bool staged_head_manifest = false;
        bool staged_head_recency = false;
    };

    struct DurableWork {
        CommitPlan plan;
        bool done = false;
        bool ok = false;
        std::string detail;
    };

    // ---- workers ----
    std::thread writer;
    std::vector<std::thread> readers;
    std::deque<std::shared_ptr<DurableWork>> durable; // urgent invalidations only; never coalesced
    bool promote_pending = false;                 // optional policy hint, guarded by mutex; no per-restore allocation
    std::atomic<bool> stop{false};

    // ---- counters and test seams ----
    ConversationStoreCounters counters;
    std::set<std::string, std::less<>> fault_steps; // transparent lookup avoids per-object temporary fault keys
    std::atomic<uint64_t> source_bytes_read{0};
    std::atomic<uint64_t> name_counter{0};

    // ------------------------------ helpers ------------------------------
    // Every one of these assumes the caller holds `mutex` unless its name says otherwise; not one of them performs a
    // storage wait while the lock is held.
    uint64_t sample_free_locked();
    uint64_t resident_locked() const { return ram_held + metadata + prep + decode; }
    // Preparation excludes payload images. Reader and writer workspace is already charged in ram_held;
    // these views enforce the separate metadata ceiling without charging RAM a second time.
    uint64_t preparation_locked() const {
        const uint64_t writer = publishing ? publishing->workspace_bytes : 0;
        return sat_add(sat_add(metadata, prep), sat_add(decode, sat_add(read_prep, writer)));
    }
    std::string temp_name();
    std::string intent_name() const;
    bool store_is_empty(std::string& why);
    void latch_fault_locked(const std::string& why);
    void recount_metadata_locked();
    bool revive_object_locked(const Digest& object);
    // The charge an accounted orphan object already carries, or 0 when it is not queued for deletion.
    uint64_t pending_object_charge_locked(const Digest& object) const;
    // Pins one object a publication will reference without creating, for as long as the transaction is in flight.
    void lease_object_locked(const std::shared_ptr<PublishJob>& job, const Digest& object, uint64_t charge);
    // Drops every lease one settled publication holds.  An object left unreferenced and not already queued becomes
    // reclaimable again (the file really does exist); a faulted store queues nothing, because no file may be deleted.
    void release_object_leases(const std::shared_ptr<PublishJob>& job);
    uint64_t retire_head_locked(uint64_t head_id);
    void touch_recency_locked(uint64_t head_id);
    uint64_t recency_of_locked(const CatalogHead& head) const;
    uint64_t choose_victim_locked(const Catalog& candidate, const std::set<uint64_t>& excluded, bool want_protected);
    void cap_protected_locked(Catalog& candidate);
    bool obtain_head_slot_locked(Catalog& candidate, uint64_t keep_head, std::vector<uint64_t>& retiring);
    bool head_manifest_locked(uint64_t head_id, Digest& manifest_id, uint64_t& bytes) const;
    // How one stored manifest was read.  `invalid` is a defect in the record and the only outcome that may retire it;
    // `unreadable` means this run's own preparation ceiling or published allowance could not hold the read and decode,
    // which says nothing about the record itself.
    enum class ManifestRead : uint8_t { loaded = 0, invalid = 1, unreadable = 2 };
    // Reads and decodes one retained head's manifest under the retention figures.  The caller must have charged the
    // decode's workspace already (manifest_read_workspace_bound of manifest_read_cap(head charge, limits)): a read
    // charges its whole peak, a publication charges its donors, and recovery and the stream-table diagnostic charge it
    // with reserve_decode_locked() around this call.
    ManifestRead load_manifest_of(uint64_t head_id, uint64_t epoch_now, ManifestMeta& out, std::string& why);
    // These helpers lock internally. The lease must outlive the raw and decoded buffers,
    // including early returns and exceptions.
    bool reserve_decode_locked(uint64_t workspace, std::string& why);
    void release_decode_locked(uint64_t workspace);
    struct DecodeLease {
        Impl& store;
        uint64_t bytes = 0;
        explicit DecodeLease(Impl& store) : store(store) {}
        DecodeLease(const DecodeLease&) = delete;
        DecodeLease& operator=(const DecodeLease&) = delete;
        ~DecodeLease() { if (bytes) store.release_decode_locked(bytes); }
        bool acquire(uint64_t requested, std::string& why) {
            if (!store.reserve_decode_locked(requested, why)) return false;
            bytes = requested;
            return true;
        }
    };
    bool stream_identity_of(const std::vector<ContainerMeta>& containers, const ConversationStoreStreamId& id,
                            Digest& scope, uint64_t& ordinal, std::string& why) const;

    // These do I/O, so they are never called with `mutex` held.
    bool remove_pending_file(const PendingFile& file, std::string& why);
    void discard_files(std::vector<PendingFile>&& files);
    void reclaim_pending();
    void queue_orphans(std::vector<PendingFile>& files);
    void copy_catalog_locked(CommitPlan& plan);
    CommitOutcome commit_plan(CommitPlan& plan, std::string& why);
    bool evict_head_durably(uint64_t head_id, std::string& why, const char* retirement_reason);
    bool make_room(uint64_t incoming_bytes, bool& floor_limited, std::string& why);
    CommitOutcome write_catalog_or_refuse(CommitPlan& plan, std::string& why);
    bool take_fault(const char* step);
    bool write_object_file(const std::string& temp, const Digest& id, const uint8_t* data, size_t size,
                           PendingFile& orphan, std::string& why);
    // Revalidate a leased orphan's directory entries outside the catalog mutex before naming it in a manifest.
    bool sync_revived_object(const Digest& id, std::string& why);

    // ---- catalog copy headroom, writer workspace and the post-rename apply ----
    // The physical charge of the catalog's *next* copy: the current catalog plus the most one commit can add to it.
    uint64_t catalog_next_bound_locked(const Catalog& candidate) const;
    // The quota headroom ordinary admissions never spend: two catalog copies plus the growth of the one after them.
    uint64_t catalog_reserve_locked() const;
    // The commit's own copy, admitted against the quota and the floor exactly like every other physical byte.
    bool admit_catalog_copy_locked(uint64_t copy_charge, const Catalog& candidate, std::string& why);
    // The counted preparation of the named parent/donor heads: catalog metadata only, no I/O, no allocation.
    DonorPrep donor_prep_locked(const ConversationStorePublish& request) const;
    // The workspace of one admitted publication, counted before it allocates any of it.
    WriterWorkspace writer_workspace_locked(const ConversationStorePublish& request,
                                            const conversation_store_detail::ImageShape& shape) const;
    // The apply structures one commit may copy: the candidate catalog, the index nodes and the queues' room.
    uint64_t catalog_apply_bytes_locked(uint64_t chunks) const;
    // Reserves the *actual* prospective chunk peak once the layout and the donors are decoded: never more than the
    // publication was admitted with, and only upwards if that counted figure grew.
    bool reserve_writer_workspace(const std::shared_ptr<PublishJob>& job, uint64_t workspace, std::string& why);
    // Everything the post-rename apply needs, allocated while the old authority is still the durable one.
    bool stage_apply_locked(CommitPlan& plan, std::string& why);
    void unstage_apply_locked(CommitPlan& plan);
    void apply_staged_locked(CommitPlan& plan);   // allocation-free: the staged nodes are moved, never built
    StoreReadTrace trace_head_locked(const CatalogHead& head, const char* event, const char* reason) const;
    void trace_retirement_locked(StoreReadTraceBatch& trace, const CommitPlan& plan, const char* event) const;
    void trace_publication_locked(StoreReadTraceBatch& trace, const CommitPlan& plan) const;

    // Lookup: catalog metadata only, under the mutex, no payload read and no GPU pointer exposed.
    void collect_matches(const ConversationStoreSelector& selector, const int32_t* ids, size_t count,
                         const std::vector<ConversationImageKey>& images,
                         std::vector<ConversationStoreMatch>& out, StoreReadTraceBatch* trace,
                         const ConversationStoreQuery* query, bool demand) const;

    // Read pipeline.  Settlement is idempotent and every path funnels through it.
    bool admit_read_locked(const ConversationStoreQuery& query, bool demand, std::string& why,
                          StoreReadTraceBatch& trace, ConversationStoreReadRefusal* refusal);
    bool refresh_read_locked(const std::shared_ptr<ReadJob>& job, StoreReadTraceBatch& trace);
    void retry_budget_deferred_locked(StoreReadTraceBatch& trace);
    void settle_job_locked(const std::shared_ptr<ReadJob>& job, ConversationStoreReadStatus status, std::string detail,
                           StoreReadTraceBatch& trace);
    void retire_job_locked(const std::shared_ptr<ReadJob>& job, ConversationStoreReadStatus status,
                           const std::string& detail, StoreReadTraceBatch& trace, const char* reason);
    void retire_reads_locked(ConversationStoreReadStatus status, const std::string& detail,
                             StoreReadTraceBatch& trace, const char* reason);
    void revoke_speculation_locked(StoreReadTraceBatch& trace, bool because_of_demand = true);
    void expire_reads_locked(StoreReadTraceBatch& trace);
    StoreReadTrace trace_state_locked(const char* event, const char* reason, ConversationStoreReadStatus status) const;
    uint64_t trace_wait_locked(const ReadJob& job) const;
    void trace_job_locked(StoreReadTraceBatch& trace, const ReadJob& job, const char* event, const char* reason,
                          ConversationStoreReadStatus status = ConversationStoreReadStatus::none) const;
    void trace_refusal_locked(StoreReadTraceBatch& trace, const ConversationStoreQuery& query, bool demand,
                              ConversationStoreReadStatus status, const char* reason, uint64_t previous = 0,
                              uint64_t head = 0, uint64_t cut = 0, uint64_t peak = 0) const;
    void trace_begin_locked(ReadJob& job, uint64_t previous = 0);
    void trace_waiters_locked(StoreReadTraceBatch& trace);
    void release_reservation_locked(const std::shared_ptr<ReadJob>& job);
    void release_lease_locked(const std::shared_ptr<ReadJob>& job);
    void release_slot_locked(const std::shared_ptr<ReadJob>& job);
    void release_head_lease_locked(const std::shared_ptr<ReadJob>& job);
    void erase_job_locked(const std::shared_ptr<ReadJob>& job);
    void free_job_prep_locked(const std::shared_ptr<ReadJob>& job);
    void release_job_preparation_locked(const std::shared_ptr<ReadJob>& job);
    bool deliver_locked(ConversationStoreRead& out, uint64_t request_id, uint64_t generation, bool specific,
                        StoreReadTraceBatch& trace);
    void push_publish_result_locked(ConversationStorePublishResult&& result);

    void writer_body();
    void reader_body();
    void publish_body(const std::shared_ptr<PublishJob>& job, ConversationStorePublishResult& out);
    // One read's own verdict.  The body allocates its raw manifest bytes, its decoded metadata, its restore targets,
    // its I/O slab and its restored image in its own frame, and settlement is what refunds the reservation those
    // bytes were charged under - so the body never settles itself.  It records what it decided here and reader_body
    // settles the job, and releases its charge, only after that frame and every buffer in it are gone.
    struct ReadOutcome {
        ConversationStoreReadStatus status = ConversationStoreReadStatus::none;
        std::string detail;
        std::shared_ptr<SavedConversation> image;   // the restored image, on a `ready` verdict
        uint64_t payload_bytes = 0;
        uint32_t stages = 0;
        double materialize_ms = 0;
    };
    void read_body(const std::shared_ptr<ReadJob>& job, ReadOutcome& out, StoreReadTraceBatch& trace);

    bool open_root(std::string& why);
    // Reads and decodes the durable catalog and decides this domain's epoch.  It may not write anything: the first
    // catalog transaction happens only once the whole physical ledger has been accounted (see reconcile()).
    bool read_catalog(std::string& why);
    bool ensure_certified_reader_floor(std::string& why);
    bool reconcile(std::string& why);
    bool write_intent(bool present, std::string& why);
};

// ---------------------------------------------------------------------------------------------------------------
// Authority: root, lock, issuer key, catalog, run intent, recovery
// ---------------------------------------------------------------------------------------------------------------

uint64_t ConversationStore::Impl::sample_free_locked() {
    free_sample = filesystem_free_bytes(dirs.root);
    return free_sample;
}

std::string ConversationStore::Impl::temp_name() {
    const uint64_t n = ++name_counter;
    return "tm-" + std::to_string(::getpid()) + "-" + std::to_string(n);
}

std::string ConversationStore::Impl::intent_name() const {
    std::array<uint8_t, 96> material{};
    std::memcpy(material.data(), options.domain.data(), 32);
    std::memcpy(material.data() + 32, options.model.data(), 32);
    std::memcpy(material.data() + 64, options.engine.data(), 32);
    return "intent-" + sha256_hex(sha256(material.data(), material.size())).substr(0, 16);
}

// A genuinely new store: nothing but the directories and the lock file this module has just created.  Only such a
// store may receive a fresh issuer key; an existing one whose key is missing stays disabled.
bool ConversationStore::Impl::store_is_empty(std::string& why) {
    std::vector<DirEntry> entries;
    if (!list_entries(dirs.root, entries, 64, why)) return false;
    for (const auto& entry : entries) {
        if (entry.name == "objects" || entry.name == "manifests" || entry.name == "tmp" ||
            entry.name == "owner.lock")
            continue;
        why.clear();
        return false;
    }
    for (int fd : {dirs.objects, dirs.manifests, dirs.tmp}) {
        std::vector<DirEntry> inner;
        if (!list_entries(fd, inner, 1, why)) return false;
        if (!inner.empty()) {
            why.clear();
            return false;
        }
    }
    return true;
}

void ConversationStore::Impl::latch_fault_locked(const std::string& why) {
    if (!faulted) {
        faulted = true;
        fault_reason = why;
    }
    promote_pending = false; // uncertain authority can never schedule a policy retry
    cv_ready.notify_all();
    cv_publish.notify_all();
    cv_work.notify_all();
}

bool ConversationStore::Impl::take_fault(const char* step) {
    std::lock_guard<std::mutex> guard(mutex);
    if (!step) return false;
    const auto found = fault_steps.find(step);
    if (found == fault_steps.end()) return false;
    fault_steps.erase(found);
    return true;
}

// An object is durable before any manifest names it. Default writes flush here; the opt-in caller marks every
// renamed object's shard and drains its shards and objects/ before encoding or creating a manifest.
bool ConversationStore::Impl::write_object_file(const std::string& temp, const Digest& id, const uint8_t* data,
                                               size_t size, PendingFile& orphan, std::string& why) {
    const std::string hex = sha256_hex(id);
    const char shard[] = {hex[0], hex[1], '\0'};
    bool created = false;
    if (!make_child_dir(dirs.objects, shard, created, why)) return false;
    // An existing shard can be the residue of mkdir followed by a failed parent flush. Existence alone is not a
    // durability proof, so even that retry must flush objects/ before the new object can be published.
    if (!dir_fsync_batch) {
        if (take_fault("object-parent-sync")) {
            why = "injected fault: the objects directory flush";
            return false;
        }
        if (!fsync_dir(dirs.objects, "objects", why)) return false;
    }
    const ShardFd shard_fd(open_child_dir(dirs.objects, shard, why));
    if (shard_fd.fd < 0) return false;
    // The scratch name is reported to the caller before it is created, so a failure here can always account for the
    // bytes of a file that may exist.
    orphan.where = Where::tmp;
    orphan.name = temp;
    if (!write_new_file(dirs.tmp, temp, data, size, why)) return false;
    if (::renameat(dirs.tmp, temp.c_str(), shard_fd.fd, hex.c_str()) != 0) {
        why = std::string("cannot publish an object: ") + std::strerror(errno);
        return false;
    }
    orphan = PendingFile{};
    if (dir_fsync_batch) return true; // The caller marks this shard before any subsequent payload is emitted.
    if (take_fault("object-shard-sync")) {
        why = "injected fault: an object shard flush";
        return false;
    }
    return fsync_dir(shard_fd.fd, "an object shard", why);
}

bool ConversationStore::Impl::sync_revived_object(const Digest& id, std::string& why) {
    // The file was flushed before its rename, but the prior publication may have failed either directory flush.
    // Its lease pins the inode while both directory entries are revalidated, without holding the catalog mutex.
    static constexpr char hex[] = "0123456789abcdef";
    const char shard[] = {hex[id[0] >> 4], hex[id[0] & 15], '\0'};
    const ShardFd shard_fd(open_child_dir(dirs.objects, shard, why));
    if (shard_fd.fd < 0) return false;
    if (take_fault("object-shard-sync")) {
        why = "injected fault: a revived object shard flush";
        return false;
    }
    if (!fsync_dir(shard_fd.fd, "a revived object shard", why)) return false;
    if (take_fault("object-parent-sync")) {
        why = "injected fault: the revived objects directory flush";
        return false;
    }
    return fsync_dir(dirs.objects, "objects", why);
}

bool ConversationStore::Impl::open_root(std::string& why) {
    // Pin every component without following symlinks. Shared system ancestors may be root-owned; the cache
    // root itself and every directory below it must be private to this account.
    if (root.empty() || root[0] != '/') {
        why = "the store root must be an absolute path";
        return false;
    }
    std::vector<std::string> parts;
    for (size_t at = 1; at <= root.size();) {
        size_t slash = root.find('/', at);
        if (slash == std::string::npos) slash = root.size();
        if (slash > at) {
            const auto part = root.substr(at, slash - at);
            if (part == "." || part == "..") {
                why = "the store root must not contain relative path components";
                return false;
            }
            parts.push_back(part);
        }
        at = slash + 1;
    }
    if (parts.empty()) {
        why = "the filesystem root is not a private cache directory";
        return false;
    }
    int parent = ::open("/", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (parent < 0) {
        why = std::string("cannot open the filesystem root: ") + std::strerror(errno);
        return false;
    }
    for (size_t i = 0; i < parts.size(); ++i) {
        const std::string& part = parts[i];
        std::string detail;
        const int existing = open_child_dir(parent, part.c_str(), detail, i + 1 == parts.size());
        if (existing >= 0) {
            ::close(parent);
            parent = existing;
            continue;
        }
        if (errno != ENOENT) {
            why = detail;
            ::close(parent);
            return false;
        }
        if (::mkdirat(parent, part.c_str(), 0700) != 0) {
            why = std::string("cannot create ") + part + ": " + std::strerror(errno);
            ::close(parent);
            return false;
        }
        if (::fsync(parent) != 0) {
            why = std::string("the parent of ") + part + " could not be flushed: " + std::strerror(errno);
            ::close(parent);
            return false;
        }
        const int made = open_child_dir(parent, part.c_str(), why);
        if (made < 0) {
            ::close(parent);
            return false;
        }
        ::close(parent);
        parent = made;
    }
    dirs.root = parent;

    for (int* slot : {&dirs.objects, &dirs.manifests, &dirs.tmp}) {
        const char* name = slot == &dirs.objects ? "objects" : (slot == &dirs.manifests ? "manifests" : "tmp");
        bool created = false;
        if (!make_child_dir(dirs.root, name, created, why)) return false;
        *slot = open_child_dir(dirs.root, name, why);
        if (*slot < 0) return false;
    }
    if (!fsync_dir(dirs.root, "the store root", why)) return false;

    // The lock is taken before the key or the catalog is even looked at, so a rejected second owner can neither read
    // nor write any authority state, and close() on it can do nothing at all.
    lock_fd = ::openat(dirs.root, "owner.lock", O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (lock_fd < 0) {
        why = std::string("cannot open the store's lock file: ") + std::strerror(errno);
        return false;
    }
    struct stat info {};
    if (::fstat(lock_fd, &info) != 0 || !private_regular(info)) {
        why = "the store's lock file is not a regular file this account owns";
        return false;
    }
    if (::flock(lock_fd, LOCK_EX | LOCK_NB) != 0) {
        counters.lock_conflicts++;
        why = "another process owns this store's root";
        return false;
    }
    lock_owned = true;

    // The issuer key: read it if it exists, create it only for a new empty store, refuse anything else.
    if (::fstatat(dirs.root, "issuer.key", &info, AT_SYMLINK_NOFOLLOW) == 0) {
        std::vector<uint8_t> key;
        if (!read_file(dirs.root, "issuer.key", 64, key, why)) return false;
        if (key.size() == issuer_key.size()) {
            std::memcpy(issuer_key.data(), key.data(), issuer_key.size());
        } else {
            std::array<uint8_t, 32> original{};
            if (key.size() == kIssuerEnvelopeBytes)
                std::memcpy(original.data(), key.data() + 16, original.size());
            const auto recognized = issuer_envelope(original);
            if (key.size() != recognized.size() ||
                std::memcmp(key.data(), recognized.data(), recognized.size()) != 0) {
                why = "unsupported or malformed issuer-key minimum-reader envelope";
                return false;
            }
            issuer_key = original;
            certified_reader_floor = true;
        }
        return true;
    }
    if (errno != ENOENT) {
        why = std::string("cannot stat the issuer key: ") + std::strerror(errno);
        return false;
    }
    std::string empty_why;
    if (!store_is_empty(empty_why)) {
        why = "the store holds records but has no issuer key: refusing to mint a new authorization identity";
        return false;
    }
    uint8_t fresh[32] = {};
    if (!fill_random(fresh, sizeof fresh)) {
        why = "no cryptographic entropy is available for the store's issuer key";
        return false;
    }
    if (!write_new_file(dirs.root, "issuer.key", fresh, sizeof fresh, why)) return false;
    if (!fsync_dir(dirs.root, "the store root", why)) return false;
    std::memcpy(issuer_key.data(), fresh, sizeof fresh);
    return true;
}

bool ConversationStore::Impl::ensure_certified_reader_floor(std::string& why) {
    // Everything that can allocate is prepared before the reservation is taken: a failure here must not leave the
    // ledger charged for a temporary that was never created.
    const auto envelope = issuer_envelope(issuer_key);
    std::string temp = temp_name();
    uint64_t reservation = 0;
    {
        std::lock_guard<std::mutex> guard(mutex);
        if (certified_reader_floor) return true;
        if (faulted || !lock_owned) {
            why = "the minimum-reader floor cannot change without healthy authority ownership";
            return false;
        }
        // A floor transition never evicts histories to pay for itself. Include two future catalog copies.
        sample_free_locked();
        const uint64_t copy = allocated_charge(kIssuerEnvelopeBytes);
        const uint64_t reserve = sat_mul(catalog_next_bound_locked(catalog), 2);
        if (sat_add(charged, sat_add(copy, reserve)) > options.quota_bytes ||
            copy > free_sample || free_sample - copy < options.min_free_bytes) {
            why = "the issuer minimum-reader envelope does not fit the shared quota or filesystem floor";
            return false;
        }
        charged = sat_add(charged, copy);
        reservation = copy;
    }
    bool written = false;
    bool own_temp = false;
    bool renamed = false;
    bool reservation_held = true;
    // Releases the temporary's charge exactly once, whichever pre-rename path ends the attempt.
    const auto release_reservation = [&] {
        if (!reservation_held) return;
        reservation_held = false;
        std::lock_guard<std::mutex> guard(mutex);
        charged -= std::min(charged, reservation);
    };
    try {
        for (uint32_t attempt = 0; attempt < kTempNameAttempts; ++attempt) {
            if (take_fault("reader-floor-write")) { why = "injected reader-floor write failure"; break; }
            if (write_new_file(dirs.tmp, temp, envelope.data(), envelope.size(), why)) {
                written = own_temp = true;
                break;
            }
            if (errno != EEXIST) { own_temp = true; break; }
            temp = temp_name();
        }
        if (written && !take_fault("reader-floor-rename")) {
            if (::renameat(dirs.tmp, temp.c_str(), dirs.root, "issuer.key") == 0) renamed = true;
            else why = std::string("cannot replace the issuer authority: ") + std::strerror(errno);
        }
        if (!renamed) {
            if (why.empty()) why = "the issuer minimum-reader envelope was not renamed";
            if (!own_temp) {
                release_reservation();
                return false;
            }
            PendingFile orphan;
            orphan.where = Where::tmp;
            orphan.name = std::move(temp);
            orphan.charge = reservation;
            std::vector<PendingFile> files;
            files.push_back(std::move(orphan));
            // The queue owns both the charge and the name from here; discard_files() credits what it removes.
            own_temp = false;
            reservation_held = false;
            discard_files(std::move(files));
            return false;
        }
    } catch (...) {
        // A failed allocation before the rename leaves a temporary behind: remove it by design, so the ledger
        // never keeps a charge for bytes this attempt no longer owns.
        if (own_temp) {
            own_temp = false;
            if (::unlinkat(dirs.tmp, temp.c_str(), 0) == 0 || errno == ENOENT) release_reservation();
        } else {
            release_reservation();
        }
        throw;
    }
    // The renamed file is now the authority: its charge is real, not a reservation for a temporary.
    reservation_held = false;
    // Set the non-allocating safety latch before constructing diagnostics. Even an exception from fsync's
    // error formatting must stop writer_body() from treating this renamed authority as an ordinary failed job.
    const auto uncertain = [&] {
        std::lock_guard<std::mutex> guard(mutex);
        faulted = true;
        promote_pending = false;
        cv_ready.notify_all();
        cv_publish.notify_all();
        cv_work.notify_all();
    };
    try {
        if (take_fault("reader-floor-allocate")) throw std::bad_alloc();
        // Mandatory regardless of object-directory fsync batching. No MAN2 payload has been staged yet.
        if (take_fault("reader-floor-sync") || !fsync_dir(dirs.root, "the issuer authority root", why) ||
            take_fault("reader-floor-tmp-sync") || !fsync_dir(dirs.tmp, "the issuer authority temporary directory", why)) {
            uncertain(); // no deletion, publication, or credit until restart
            why = "the issuer minimum-reader floor rename has uncertain durability";
            std::lock_guard<std::mutex> guard(mutex);
            fault_reason = why;
            return false;
        }
        std::lock_guard<std::mutex> guard(mutex);
        certified_reader_floor = true;
        // Only now is the old key's replaced inode durably gone.
        charged -= std::min(charged, allocated_charge(issuer_key.size()));
    } catch (...) {
        uncertain();
        throw;
    }
    return true;
}

bool ConversationStore::Impl::write_intent(bool present, std::string& why) {
    const std::string name = intent_name();
    const uint8_t marker[8] = {'S', 'T', 'C', 'S', 'I', 'N', 'T', '1'};
    if (present) {
        struct stat info {};
        if (::fstatat(dirs.root, name.c_str(), &info, AT_SYMLINK_NOFOLLOW) == 0) return true;
        // This marker is safety critical: it is created even when the quota or the floor has no room for it, and its
        // bytes are charged before it is written so the ledger never pretends the store owns less than it does.
        {
            std::lock_guard<std::mutex> guard(mutex);
            charged = sat_add(charged, allocated_charge(sizeof marker));
        }
        if (!write_new_file(dirs.root, name, marker, sizeof marker, why)) return false;
        return fsync_dir(dirs.root, "the store root", why);
    }
    if (!remove_file(dirs.root, name, why)) return false;
    {
        std::lock_guard<std::mutex> guard(mutex);
        charged -= std::min(charged, allocated_charge(sizeof marker));
    }
    std::string flush;
    const bool gone_durable = fsync_dir(dirs.root, "the store root", flush) && !take_fault("intent-unlink-sync");
    if (!gone_durable) {
        // The marker is gone from this process's view but not durably, so it is re-armed here and its own flush is
        // attempted again.  The close is still reported as failed either way: a re-arm this call could not prove is
        // not a durable one, and the next start must distrust this domain.
        why = flush.empty() ? std::string("the store root could not be flushed after removing the run-intent marker")
                            : flush;
        std::string rearm;
        if (!write_new_file(dirs.root, name, marker, sizeof marker, rearm)) {
            why += "; the run-intent marker could not be re-armed: " + rearm;
            return false;
        }
        {
            std::lock_guard<std::mutex> guard(mutex);
            charged = sat_add(charged, allocated_charge(sizeof marker));
        }
        std::string refush;
        const bool rearmed_durable =
            fsync_dir(dirs.root, "the store root", refush) && !take_fault("intent-rearm-sync");
        if (rearmed_durable)
            why += "; the run-intent marker was re-armed and flushed, so this domain stays distrusted";
        else
            why += "; the re-armed run-intent marker is not durably flushed: " +
                   (refush.empty() ? std::string("the store root could not be flushed") : refush);
        return false;
    }
    return true;
}

bool ConversationStore::Impl::read_catalog(std::string& why) {
    Catalog loaded;
    std::string detail;
    struct stat catalog_info {};
    if (::fstatat(dirs.root, "catalog", &catalog_info, AT_SYMLINK_NOFOLLOW) == 0) {
        // Whatever the file turns out to be, its bytes exist and are charged before anything is trusted.
        catalog_charge = allocated_charge(uint64_t(catalog_info.st_size > 0 ? catalog_info.st_size : 0));
        charged += catalog_charge;
        std::vector<uint8_t> bytes;
        // Unknown versions must stop before reconcile() can delete a single file.
        std::array<char, 8> tag{};
        const int fd = ::openat(dirs.root, "catalog", O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
        const ssize_t got = fd >= 0 ? ::pread(fd, tag.data(), tag.size(), 0) : -1;
        if (fd >= 0) ::close(fd);
        if (got < 0 || (got == ssize_t(tag.size()) && std::memcmp(tag.data(), "STCSCAT1", 8) != 0)) {
            why = "the catalog requires an unsupported reader version";
            return false;
        }
        if (!read_file(dirs.root, "catalog", limits.max_manifest_bytes, bytes, detail) ||
            !conversation_store_detail::catalog_decode(bytes.data(), bytes.size(), loaded, detail) ||
            !(loaded.issuer_key_digest == sha256(issuer_key.data(), issuer_key.size()))) {
            // A catalog that cannot be trusted is cold caching: never a directory-based resurrection.  The reason is
            // recorded for the open sequence's diagnostics; serving from an empty catalog is still a success.
            counters.corrupt_records++;
            loaded = Catalog{};
            if (!detail.empty()) why = "the stored catalog was not trusted: " + detail;
        }
    }
    catalog = std::move(loaded);
    catalog.issuer_key_digest = sha256(issuer_key.data(), issuer_key.size());

    domain_index = uint32_t(catalog.domains.size());
    for (uint32_t i = 0; i < catalog.domains.size(); ++i)
        if (catalog.domains[i].digest == options.domain) {
            domain_index = i;
            break;
        }
    if (domain_index == catalog.domains.size()) {
        CatalogDomain domain;
        domain.digest = options.domain;
        domain.epoch = 1;
        catalog.domains.push_back(domain);
        epoch = 1;
    } else {
        epoch = catalog.domains[domain_index].epoch;
    }

    // A run-intent marker for exactly this domain means its previous run did not close cleanly: its records and every
    // outstanding completion of its epoch are invalid.  Otherwise records of a retired epoch are unreachable.
    struct stat intent_info {};
    unclean_start = ::fstatat(dirs.root, intent_name().c_str(), &intent_info, AT_SYMLINK_NOFOLLOW) == 0;
    if (unclean_start) {
        // The epoch is advanced in memory here; retire_head_locked() for this domain's records waits for
        // reconcile(), which retires them once every one of their bytes is in the physical ledger.
        epoch += 1;
        catalog.domains[domain_index].epoch = epoch;
        counters.epoch_advances++;
    }
    for (const auto& head : catalog.heads) head_recency[head.head_id] = head.published_order;
    return true;
}

// Read only the bounded schema header before recovery performs any cleanup.
// Unsupported manifests are not corrupt histories and may belong to a different profile.
static bool manifest_versions_supported(int directory, const Catalog& catalog, bool floor, std::string& why) {
    for (const auto& head : catalog.heads) {
        const std::string name = sha256_hex(head.manifest_id);
        const int fd = ::openat(directory, name.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
        if (fd < 0) {
            if (errno == ENOENT || errno == ENOTDIR) continue;
            why = "cannot inspect a manifest's reader version";
            return false;
        }
        std::array<char, 8> tag{};
        const ssize_t got = ::pread(fd, tag.data(), tag.size(), 0);
        ::close(fd);
        // Only the ordering invariant is decided here: a certified record must never be reachable through legacy
        // authority, because the legacy reader would reclaim it. A damaged tag stays an ordinary corrupt record.
        if (got == ssize_t(tag.size()) && std::memcmp(tag.data(), "STCSMAN2", 8) == 0 && !floor) {
            why = "a certified record exists without the established minimum reader";
            return false;
        }
    }
    return true;
}

bool ConversationStore::Impl::reconcile(std::string& why) {
    StoreReadTraceBatch recovery_trace;
    if (!manifest_versions_supported(dirs.manifests, catalog, certified_reader_floor, why)) return false;
    recovery_trace.io_allowed = false; // snapshots are not removals until the startup catalog commits
    const auto retiring_trace = [&](const CatalogHead& head, const char* reason) {
        if (read_timing) recovery_trace.add(trace_head_locked(head, "retired", reason));
    };
    // Every scratch file is garbage by construction: a durable write completes with a rename out of tmp/.  Anything
    // that is not a regular file this account owns is left in place (and is still accounted for below).
    std::vector<DirEntry> temporaries;
    if (!list_entries(dirs.tmp, temporaries, kMaxScanEntries, why)) return false;
    for (const auto& entry : temporaries) {
        if (!entry.regular || !entry.ours) continue;
        PendingFile file;
        file.where = Where::tmp;
        file.name = entry.name;
        file.charge = allocated_charge(entry.size);
        charged += file.charge;
        reclaimable += file.charge;
        counters.recovered_orphans++;
        pending_files.push_back(std::move(file));
    }
    // The control files are real bytes on the device and belong to the shared charge as well.  The catalog is
    // charged by load_catalog() from the file it read, so it is not counted twice here.
    std::vector<DirEntry> root_entries;
    if (!list_entries(dirs.root, root_entries, kMaxScanEntries, why)) return false;
    for (const auto& entry : root_entries)
        if (entry.regular && entry.ours && entry.name != "catalog") charged += allocated_charge(entry.size);

    // Count what is physically present before the catalog is trusted: only the catalog authorizes a manifest, and
    // only a manifest authorizes an object.
    std::map<Digest, uint64_t> present_objects;
    std::map<Digest, uint64_t> present_manifests;
    uint64_t scanned = 0;
    std::vector<DirEntry> shards;
    if (!list_entries(dirs.objects, shards, kMaxScanEntries, why)) return false;
    for (const auto& shard : shards) {
        if (!shard.directory || !shard.ours || shard.name.size() != 2 || !is_hex_name(shard.name)) {
            // Not a shard this store could have created: account nothing for it and delete nothing from it.
            continue;
        }
        const int fd = open_child_dir(dirs.objects, shard.name.c_str(), why);
        if (fd < 0) {
            why.clear();
            continue;
        }
        std::vector<DirEntry> files;
        std::string nested;
        if (!list_entries(fd, files, kMaxScanEntries, nested)) {
            ::close(fd);
            continue;
        }
        for (const auto& file : files) {
            const uint64_t charge = allocated_charge(file.size);
            charged += charge;
            if (++scanned > kMaxScanEntries) {
                why = "the store holds more objects than it can account at opening";
                ::close(fd);
                return false;
            }
            Digest digest{};
            if (file.regular && file.ours && file.name.size() == 64 && is_hex_name(file.name) &&
                hex_to_digest(file.name, digest) && sha256_hex(digest).compare(0, 2, shard.name) == 0) {
                present_objects.emplace(digest, charge);
                continue;
            }
            if (file.regular && file.ours) {
                reclaimable += charge;
                PendingFile unknown;
                unknown.where = Where::objects;
                unknown.shard = shard.name;
                unknown.name = file.name;
                unknown.charge = charge;
                pending_files.push_back(std::move(unknown));
            }
        }
        ::close(fd);
    }
    std::vector<DirEntry> manifests;
    if (!list_entries(dirs.manifests, manifests, kMaxScanEntries, why)) return false;
    for (const auto& file : manifests) {
        const uint64_t charge = allocated_charge(file.size);
        charged += charge;
        if (++scanned > kMaxScanEntries) {
            why = "the store holds more manifests than it can account at opening";
            return false;
        }
        Digest digest{};
        if (file.regular && file.ours && file.name.size() == 64 && is_hex_name(file.name) &&
            hex_to_digest(file.name, digest)) {
            present_manifests.emplace(digest, charge);
            continue;
        }
        if (file.regular && file.ours) {
            reclaimable += charge;
            PendingFile unknown;
            unknown.where = Where::manifests;
            unknown.name = file.name;
            unknown.charge = charge;
            pending_files.push_back(std::move(unknown));
        }
    }

    std::vector<uint64_t> dropped;
    for (const auto& head : catalog.heads) {
        recount_metadata_locked();
        // A record of this domain that this run must not reuse - an unclean start or a retired epoch - is dropped
        // here, after its manifest and its objects have been accounted above, so its bytes are charged into the
        // ledger before the first catalog transaction forgets it.
        if (head.domain == domain_index && (unclean_start || head.epoch != epoch)) {
            dropped.push_back(head.head_id);
            retiring_trace(head, unclean_start ? "unclean_domain" : "retired_epoch");
            continue;
        }
        const auto manifest = present_manifests.find(head.manifest_id);
        DecodeLease lease(*this);
        ManifestMeta meta;
        std::string detail;
        ManifestRead read = ManifestRead::invalid;
        if (manifest != present_manifests.end()) {
            // The decode's own workspace is charged before the read allocates any of it: the wider retention figures
            // this run decodes another profile's record under can describe a manifest whose decoded metadata and
            // validation tables are several times the file, and nothing here may allocate past the preparation
            // ceiling (see manifest_read_workspace_bound).
            const uint64_t workspace = conversation_store_detail::manifest_read_workspace_bound(
                manifest_read_cap(head.manifest_bytes, limits));
            if (!lease.acquire(workspace, detail)) {
                read = ManifestRead::unreadable;
            } else {
                read = load_manifest_of(head.head_id, head.epoch, meta, detail);
            }
        }
        if (read == ManifestRead::unreadable) {
            // This run cannot read the record within its own figures.  That is not evidence about the record, so it is
            // not corruption and nothing is dropped, retired or reclaimed: the open fails, persistence stays disabled
            // for this run and the caller falls back to its cold path, leaving the catalog, every manifest, every
            // object and every charge exactly where they are for a run that can read them.
            why = "the store cannot read every record its catalog authorizes within this run's figures: " + detail;
            return false;
        }
        if (read == ManifestRead::invalid) {
            counters.corrupt_records++;
            dropped.push_back(head.head_id);
            retiring_trace(head, "invalid_manifest");
            continue;
        }
        bool complete = true;
        for (const auto& object : meta.objects)
            if (present_objects.find(object.first) == present_objects.end()) {
                complete = false;
                break;
            }
        if (!complete) {
            counters.corrupt_records++;
            dropped.push_back(head.head_id);
            retiring_trace(head, "missing_object");
            continue;
        }
        head_manifest_bytes[head.head_id] = manifest->second;
        head_recency[head.head_id] = head.published_order;
        recency_counter = std::max(recency_counter, head.published_order);
        std::vector<Digest>& list = head_objects[head.head_id];
        list.reserve(meta.objects.size());
        for (const auto& object : meta.objects) {
            ObjectState& state = objects[object.first];
            const auto present = present_objects.find(object.first);
            if (state.bytes == 0) state.bytes = present == present_objects.end() ? 0 : present->second;
            state.refs += 1;
            list.push_back(object.first);
        }
    }
    for (uint64_t head_id : dropped) retire_head_locked(head_id);

    // Queue unreferenced files, but retain them until the startup catalog is durable.
    // A refused transaction must not delete bytes still named by the old catalog.
    for (const auto& manifest : present_manifests) {
        bool referenced = false;
        for (const auto& head : catalog.heads)
            if (head.manifest_id == manifest.first) {
                referenced = true;
                break;
            }
        if (referenced) continue;
        reclaimable += manifest.second;
        const std::string name = sha256_hex(manifest.first);
        PendingFile file;
        file.where = Where::manifests;
        file.name = name;
        file.charge = manifest.second;
        pending_files.push_back(std::move(file));
        counters.deferred_reclaims++;
    }
    for (const auto& object : present_objects) {
        if (objects.count(object.first)) continue;
        reclaimable += object.second;
        PendingFile file;
        file.where = Where::objects;
        file.object = object.first;
        file.shard = sha256_hex(object.first).substr(0, 2);
        file.name = sha256_hex(object.first);
        file.charge = object.second;
        pending_files.push_back(std::move(file));
        counters.deferred_reclaims++;
    }

    // Enforce the head and index ceilings before serving anything.
    std::vector<uint64_t> retiring;
    while (catalog.heads.size() > options.head_limit) {
        const uint64_t victim = choose_victim_locked(catalog, {}, false);
        if (victim == 0) break;
        for (size_t i = 0; i < catalog.heads.size(); ++i)
            if (catalog.heads[i].head_id == victim) {
                retiring.push_back(victim);
                retiring_trace(catalog.heads[i], "startup_head_limit");
                catalog.heads.erase(catalog.heads.begin() + std::ptrdiff_t(i));
                break;
            }
    }
    recount_metadata_locked();
    while (metadata > options.metadata_bytes && catalog.heads.size() > 1) {
        const uint64_t victim = choose_victim_locked(catalog, {}, false);
        if (victim == 0) break;
        for (size_t i = 0; i < catalog.heads.size(); ++i)
            if (catalog.heads[i].head_id == victim) {
                retiring.push_back(victim);
                retiring_trace(catalog.heads[i], "startup_metadata_limit");
                catalog.heads.erase(catalog.heads.begin() + std::ptrdiff_t(i));
                break;
            }
        recount_metadata_locked();
    }
    if (metadata > options.metadata_bytes) {
        why = "the store's index cannot be held under ConversationStoreOptions::metadata_bytes";
        return false;
    }
    for (uint64_t head_id : retiring) retire_head_locked(head_id);
    // The marker is armed before anything is served, and it is armed whether or not a catalog transaction follows.
    if (!write_intent(true, why)) return false;
    intent_active = true;
    // The first catalog transaction happens only when this run's view of the authority really differs from the
    // durable one.  An unchanged catalog *is* the durable authority (the decoded file, re-derived deterministically
    // where a domain was added), so rewriting it would be a durable write nothing needs - and, on a filesystem or a
    // quota that cannot pay for it, the difference between a store that serves reads and one that cannot open.  A
    // retired epoch, a dropped record or a queued retirement all change the authority, and those are committed here
    // now that the whole physical ledger above has been accounted.
    if (!unclean_start && dropped.empty() && retiring.empty()) {
        recount_metadata_locked();
        return true;
    }
    CommitPlan plan;
    plan.kind = CommitPlan::Kind::promote;
    if (commit_plan(plan, why) != CommitOutcome::durable_applied) return false;
    counters.evictions += retiring.size();
    if (read_timing && (recovery_trace.count || recovery_trace.dropped)) {
        const uint64_t applied = plan.applied_ns;
        for (size_t i = 0; i < recovery_trace.count; ++i) {
            recovery_trace.records[i].finish_ns = applied;
            recovery_trace.records[i].catalog_version = catalog_version;
        }
        recovery_trace.io_allowed = true;
    }
    recount_metadata_locked();
    return true;
}

bool ConversationStore::Impl::revive_object_locked(const Digest& object) {
    for (auto it = pending_files.begin(); it != pending_files.end(); ++it) {
        if (it->where != Where::objects || !(it->object == object)) continue;
        reclaimable -= std::min(reclaimable, it->charge);
        pending_files.erase(it);
        return true;
    }
    return false;
}

uint64_t ConversationStore::Impl::pending_object_charge_locked(const Digest& object) const {
    for (const auto& file : pending_files)
        if (file.where == Where::objects && file.object == object) return file.charge;
    return 0;
}

void ConversationStore::Impl::lease_object_locked(const std::shared_ptr<PublishJob>& job, const Digest& object,
                                                 uint64_t charge) {
    // There is one writer. Repeated chunks in its image need one lease, not one unreleasable count per use.
    if (object_leases.count(object)) return;
    job->leased_objects.push_back(object);
    ObjectLease lease;
    lease.charge = charge;
    object_leases.emplace(object, lease);
}

// The attempt's leases end here, after the transaction has settled.  Every object it referenced without creating is
// dropped from the lease table; one that no retained head and no other publication still names is queued for deletion
// exactly once, at the charge its own file carries, so the credit follows the file and nothing is credited twice.
void ConversationStore::Impl::release_object_leases(const std::shared_ptr<PublishJob>& job) {
    std::lock_guard<std::mutex> guard(mutex);
    for (const Digest& object : job->leased_objects) {
        const auto lease = object_leases.find(object);
        if (lease == object_leases.end()) continue;
        const uint64_t charge = lease->second.charge;
        object_leases.erase(lease);
        // A faulted or uncertain store may not delete or credit anything: the catalog that would say which file is
        // authorized is exactly the one this process cannot read.  The lease goes; the file and its charge stay.
        if (faulted) continue;
        const auto state = objects.find(object);
        if (state != objects.end() && state->second.refs != 0) continue;   // still owned by a retained head
        if (pending_object_charge_locked(object) != 0) continue;           // already queued for deletion
        // A record that cannot even be built stays unqueued and charged: the next start accounts the file from the
        // directory listing, and nothing is ever credited for a file that was not confirmed gone.
        try {
            PendingFile file;
            file.where = Where::objects;
            file.object = object;
            file.shard = sha256_hex(object).substr(0, 2);
            file.name = sha256_hex(object);
            file.charge = state != objects.end() ? state->second.bytes : charge;
            pending_files.push_back(std::move(file));
            reclaimable += pending_files.back().charge;
        } catch (...) {
            continue;
        }
    }
    job->leased_objects.clear();
}

uint64_t ConversationStore::Impl::retire_head_locked(uint64_t head_id) {
    PendingFile manifest;
    bool have_manifest = false;
    size_t head_at = catalog.heads.size();
    for (size_t i = 0; i < catalog.heads.size(); ++i) {
        if (catalog.heads[i].head_id != head_id) continue;
        head_at = i;
        const Digest id = catalog.heads[i].manifest_id;
        bool shared = false;
        for (size_t j = 0; j < catalog.heads.size(); ++j)
            if (j != i && catalog.heads[j].manifest_id == id) shared = true;
        if (digest_nonzero(id) && !shared) {
            manifest.where = Where::manifests;
            manifest.name = sha256_hex(id);
            const auto bytes = head_manifest_bytes.find(head_id);
            manifest.charge = bytes == head_manifest_bytes.end() ? 0 : bytes->second;
            manifest.head_id = head_id;
            have_manifest = true;
        }
        break;
    }
    const bool held = head_leases.count(head_id) != 0;
    const auto list = head_objects.find(head_id);
    const size_t before = pending_files.size();
    // Prepare every allocating filename and queue slot before changing catalog
    // authority or any object reference. A final-lease retry must never decrement
    // a shared object twice after an allocation failure partway through a head.
    try {
        const size_t room = (have_manifest ? 1 : 0) + (!held && list != head_objects.end() ? list->second.size() : 0);
        pending_files.reserve(before + room);
        if (have_manifest) pending_files.push_back(std::move(manifest));
        if (!held && list != head_objects.end()) {
            for (const Digest& object : list->second) {
                const auto state = objects.find(object);
                if (state == objects.end() || state->second.refs != 1) continue;
                if (!fault_steps.empty()) {
                    const auto fault = fault_steps.find("retire-object-prepare");
                    if (fault != fault_steps.end()) {
                        fault_steps.erase(fault);
                        throw std::bad_alloc();
                    }
                }
                PendingFile file;
                file.where = Where::objects;
                file.object = object;
                file.name = sha256_hex(object);
                file.shard = file.name.substr(0, 2);
                file.charge = state->second.bytes;
                file.head_id = head_id;
                pending_files.push_back(std::move(file));
            }
        }
    } catch (...) {
        pending_files.resize(before);
        throw;
    }
    if (head_at != catalog.heads.size())
        catalog.heads.erase(catalog.heads.begin() + std::ptrdiff_t(head_at));
    uint64_t freed = 0;
    for (size_t i = before; i < pending_files.size(); ++i) {
        reclaimable += pending_files[i].charge;
        freed += pending_files[i].charge;
    }
    // A retired selected head retains all object references until its final
    // request lease ends. Its pending manifest cannot be reclaimed while held.
    if (held) return freed;
    if (list != head_objects.end()) {
        for (const Digest& object : list->second) {
            auto state = objects.find(object);
            if (state != objects.end() && state->second.refs > 0) --state->second.refs;
        }
        head_objects.erase(list);
    }
    head_manifest_bytes.erase(head_id);
    head_recency.erase(head_id);
    eviction_excluded.erase(head_id);
    return freed;
}

void ConversationStore::Impl::touch_recency_locked(uint64_t head_id) {
    // Recovery and publication prepare this retained head's node; a confirmed restore must never allocate one.
    head_recency.at(head_id) = ++recency_counter;
}

uint64_t ConversationStore::Impl::recency_of_locked(const CatalogHead& head) const {
    const auto found = head_recency.find(head.head_id);
    return found == head_recency.end() ? head.published_order : found->second;
}

uint64_t ConversationStore::Impl::choose_victim_locked(const Catalog& candidate, const std::set<uint64_t>& excluded,
                                                       bool want_protected) {
    uint64_t victim = 0;
    uint64_t oldest = 0;
    for (const auto& head : candidate.heads) {
        if (head.pinned) continue;
        if (excluded.count(head.head_id)) continue;
        if (head_leases.count(head.head_id)) continue;
        if (bool(head.protected_seg) != want_protected) continue;
        const uint64_t recency = recency_of_locked(head);
        if (victim == 0 || recency < oldest) {
            victim = head.head_id;
            oldest = recency;
        }
    }
    return victim;
}

void ConversationStore::Impl::cap_protected_locked(Catalog& candidate) {
    const uint64_t limit = uint64_t(options.head_limit) * 3 / 4;
    for (;;) {
        uint64_t count = 0;
        uint64_t demote = 0;
        uint64_t oldest = 0;
        for (const auto& head : candidate.heads) {
            if (!head.protected_seg) continue;
            ++count;
            const uint64_t recency = recency_of_locked(head);
            if (demote == 0 || recency < oldest) {
                demote = head.head_id;
                oldest = recency;
            }
        }
        if (count <= limit || demote == 0) return;
        for (auto& head : candidate.heads)
            if (head.head_id == demote) head.protected_seg = false;
    }
}

bool ConversationStore::Impl::obtain_head_slot_locked(Catalog& candidate, uint64_t keep_head,
                                                      std::vector<uint64_t>& retiring) {
    while (candidate.heads.size() > options.head_limit) {
        std::set<uint64_t> excluded = eviction_excluded;
        if (keep_head) excluded.insert(keep_head);
        uint64_t victim = choose_victim_locked(candidate, excluded, false);
        if (victim == 0) victim = choose_victim_locked(candidate, excluded, true);
        if (victim == 0) return false;   // every head is pinned or leased: refuse rather than overrun the ceiling
        for (size_t i = 0; i < candidate.heads.size(); ++i)
            if (candidate.heads[i].head_id == victim) {
                retiring.push_back(victim);
                candidate.heads.erase(candidate.heads.begin() + std::ptrdiff_t(i));
                break;
            }
    }
    return true;
}

bool ConversationStore::Impl::head_manifest_locked(uint64_t head_id, Digest& manifest_id, uint64_t& bytes) const {
    for (const auto& head : catalog.heads) {
        if (head.head_id != head_id) continue;
        manifest_id = head.manifest_id;
        bytes = head.manifest_bytes;
        return true;
    }
    return false;
}

ConversationStore::Impl::ManifestRead ConversationStore::Impl::load_manifest_of(uint64_t head_id, uint64_t epoch_now,
                                                                                ManifestMeta& out, std::string& why) {
    Digest manifest_id{};
    uint64_t bytes = 0;
    {
        std::lock_guard<std::mutex> guard(mutex);
        if (!head_manifest_locked(head_id, manifest_id, bytes)) {
            why = "no retained head has that identity";
            return ManifestRead::invalid;
        }
    }
    const std::string name = sha256_hex(manifest_id);
    std::vector<uint8_t> raw;
    // The read is capped by the head's own recorded charge as well as the configured ceiling, so the reservations
    // built from those same recorded charges - a read's peak, a publication's donors - can never be smaller than it.
    const uint64_t capped = manifest_read_cap(bytes, limits);
    if (!read_file(dirs.manifests, name, capped, raw, why)) {
        // Missing or contradictory files are invalid. A resource, permission or I/O
        // failure does not establish corruption and must preserve catalog authority.
        struct stat info {};
        if (::fstatat(dirs.manifests, name.c_str(), &info, AT_SYMLINK_NOFOLLOW) != 0)
            return errno == ENOENT || errno == ENOTDIR ? ManifestRead::invalid : ManifestRead::unreadable;
        if (!S_ISREG(info.st_mode) || info.st_uid != ::geteuid() || info.st_size < 0 ||
            (bytes != 0 && uint64_t(info.st_size) > bytes))
            return ManifestRead::invalid;
        return ManifestRead::unreadable;
    }
    {
        std::lock_guard<std::mutex> guard(mutex);
        counters.bytes_read += raw.size();
    }
    // Decoded under the *retention* figures rather than this run's admitted ones: this is where the soundness of a
    // record this run did not write is decided - for recovery, for the shared physical ledger and for the publication
    // planner - and a record another profile captured is not corrupt merely because this run could not have created
    // it (a different stage count, a different checkpoint policy, a different image budget).  Whether this run may
    // *serve* the record is a separate decision, made against `limits` on the read and publication paths themselves.
    // Under those figures a decode fails only on the record's own structure: the manifest ceiling is the configured
    // one (never relaxed) and every count is still bounded by the bytes the file actually holds, so no capacity figure
    // is left for a decode to fail on - it is classified with manifest_within_limits() instead, where the caller can
    // refuse the record without deleting it.  The workspace this decode needs was charged by the caller.
    if (!conversation_store_detail::manifest_decode(raw.data(), raw.size(), retained_limits, out, why))
        return ManifestRead::invalid;
    if (!(out.manifest_id == manifest_id) || out.epoch != epoch_now) {
        why = "that manifest does not belong to this execution-domain epoch";
        return ManifestRead::invalid;
    }
    return ManifestRead::loaded;
}

bool ConversationStore::Impl::stream_identity_of(const std::vector<ContainerMeta>& containers,
                                                 const ConversationStoreStreamId& id, Digest& scope, uint64_t& ordinal,
                                                 std::string& why) const {
    return conversation_store_detail::stream_object_identity(containers, id, options.model, scope, ordinal, why);
}

// The bounded preparation the named parent and reuse donors need: the raw manifest bytes this attempt will read and
// the decoded metadata it will retain, from the heads' own recorded charges.  Catalog metadata only: no I/O, no
// allocation, and no figure taken from a configured maximum.  A head this store does not retain contributes nothing,
// because the writer's own lookup of it fails before it allocates anything.
DonorPrep ConversationStore::Impl::donor_prep_locked(const ConversationStorePublish& request) const {
    DonorPrep prep;
    // The writer retains one decoded copy per distinct donor, even when many
    // streams name it. Its predecessor is a separate retained copy: `supersedes` is charged here whenever it is
    // named, and again through the donor loop when it is also a donor of its own (the writer really does load both).
    for (const auto& head : catalog.heads) {
        const uint64_t raw = manifest_read_cap(head.manifest_bytes, limits);
        if (head.head_id == request.supersedes)
            prep.decoded = sat_add(prep.decoded, conversation_store_detail::manifest_read_workspace_bound(raw));
        bool donor_counted = false;
        for (const auto& entry : request.reuse) {
            if (entry.stream.kind == kDraftBoundaryKind || entry.head_id != head.head_id) continue;
            donor_counted = true;
            break;
        }
        if (!donor_counted) continue;
        prep.decoded = sat_add(prep.decoded, conversation_store_detail::manifest_read_workspace_bound(raw));
        // Repeated intervals contribute repeated chunk references, but do not cause another manifest decode.  Every
        // chunk of a donor belongs to exactly one stream and one interval can reference at most every chunk of the
        // stream it names, so the donor's whole-manifest chunk bound is multiplied by the most intervals any *single*
        // stream of that donor carries - independent streams stay one manifest wide, while a stream named repeatedly
        // is charged for every time it can name the same chunk again.  One count per donor would let repeats
        // reference chunks nothing reserved; summing per donor manifest is what this replaces.
        uint64_t heaviest = 0;
        for (size_t i = 0; i < request.reuse.size(); ++i) {
            const auto& entry = request.reuse[i];
            if (entry.stream.kind == kDraftBoundaryKind || entry.head_id != head.head_id) continue;
            uint64_t named = 1;
            for (size_t j = 0; j < i; ++j) {
                const auto& earlier = request.reuse[j];
                if (earlier.head_id == head.head_id && earlier.stream == entry.stream) ++named;
            }
            heaviest = std::max(heaviest, named);
        }
        prep.chunks = sat_add(prep.chunks, sat_mul(sat_add(raw / 44, 2), heaviest));
    }
    return prep;
}

// The workspace of one admission, counted before the job that would need it exists.  `shape` is the allocation-free
// count of the image's own structures; the donor figure is the bounded one above, so the first stage never depends on
// a decoded manifest to know what the decode will cost.
WriterWorkspace ConversationStore::Impl::writer_workspace_locked(const ConversationStorePublish& request,
                                                                 const conversation_store_detail::ImageShape& shape)
    const {
    WriterWorkspace work;
    work.shard_batch = dir_fsync_batch ? kShardBatchWorkspace : 0;
    work.request_copy = sat_add(sat_mul(uint64_t(request.reuse.size()), sizeof(ConversationStoreReuse)),
                                kJobFixedOverhead);
    work.layout = conversation_store_detail::image_layout_workspace(shape);
    if (shape.certified) work.layout = sat_add(work.layout, kReaderFloorWorkspace);
    const DonorPrep prep = donor_prep_locked(request);
    work.donor_decoded = prep.decoded;
    const uint64_t chunk_limit = std::max<uint64_t>(4096, options.chunk_bytes);
    // The image's own chunk grid: every stream contributes at most one partial chunk beside its full ones, and the
    // smallest chunk an element-aligned stream uses is the chunk size minus its 16-byte alignment.  Then the most
    // chunks the named donors could contribute through their intervals.
    const uint64_t aligned = std::max<uint64_t>(1, chunk_limit - 16);
    const uint64_t grid = sat_add(sat_add(sat_mul(shape.payload_bytes / aligned, 1), shape.streams), 1);
    const uint64_t chunks = sat_add(grid, prep.chunks);
    work.chunks = chunks;
    work.buffer = std::min(chunk_limit, shape.max_stream_bytes);
    const auto manifest_shape = conversation_store_detail::ManifestShape{
        shape.containers, shape.records - shape.containers, shape.kv_layers, shape.streams,
        chunks, chunks, shape.cuts, shape.certified};
    work.manifest = conversation_store_detail::manifest_encoded_size_of(manifest_shape);
    work.catalog_apply = catalog_apply_bytes_locked(chunks);
    return work;
}

// The apply structures one commit of this publication may copy: the candidate catalog (every head and every cut),
// the object-index nodes, the head's own list and the room the retire and orphan queues need.  Catalog metadata and
// counts only.
uint64_t ConversationStore::Impl::catalog_apply_bytes_locked(uint64_t chunks) const {
    uint64_t bytes = sat_mul(uint64_t(catalog.heads.size()) + 1,
                             sizeof(CatalogHead) + 3 * kAllocSlackBytes);
    for (const auto& head : catalog.heads)
        bytes = sat_add(bytes, sat_mul(uint64_t(head.cuts.size()) + 1, sizeof(CutRecord) + kAllocSlackBytes));
    // The retire queue (one manifest entry per retired head plus every object that head alone named) and the orphan
    // queue this attempt fills before its commit.
    bytes = sat_add(bytes, sat_mul(sat_add(sat_add(uint64_t(catalog.heads.size()), uint64_t(objects.size())), chunks),
                                   sizeof(PendingFile) + 2 * kAllocSlackBytes));
    return bytes;
}

// Reserves the *actual* prospective peak once the layout and the donors are decoded: never below what is already
// live, never above what the publication was admitted with unless the counted figure really grew (which is then
// admitted or refused here, before one chunk is read or written).
bool ConversationStore::Impl::reserve_writer_workspace(const std::shared_ptr<PublishJob>& job, uint64_t workspace,
                                                       std::string& why) {
    std::lock_guard<std::mutex> guard(mutex);
    if (workspace <= job->workspace_bytes) {
        const uint64_t drop = job->workspace_bytes - workspace;
        ram_held -= std::min(ram_held, drop);
        job->workspace_bytes = workspace;
        return true;
    }
    const uint64_t delta = workspace - job->workspace_bytes;
    if (workspace > options.metadata_bytes || preparation_locked() > options.metadata_bytes - delta) {
        why = "the writer's workspace does not fit the store's preparation ceiling";
        return false;
    }
    // A later absolute allowance already accounts for the caller's released image lease.
    // resident_locked() includes this job's image and existing workspace; charge only growth.
    const uint64_t envelope = sat_add(ram_budget, job->external_credit_bytes);
    if (delta > envelope || resident_locked() > envelope - delta) {
        why = "the writer's workspace does not fit the published RAM allowance";
        return false;
    }
    ram_held += delta;
    job->workspace_bytes = workspace;
    counters.peak_bytes = std::max(counters.peak_bytes, resident_locked());
    return true;
}

// Keep transient metadata charged until its buffers are destroyed. The preparation ceiling
// bounds an operation's workspace alongside the resident index; the shared RAM allowance
// bounds concurrent reads, publication and diagnostic work together. A decode allocates
// metadata and validation tables before checking consistency, so its workspace is charged
// against both before it starts.  A read's and a writer's own decodes are charged to their reservations instead (a
// read's whole peak, a writer's donor figures), so this is only ever a path that holds no reservation of its own.
bool ConversationStore::Impl::reserve_decode_locked(uint64_t workspace, std::string& why) {
    std::lock_guard<std::mutex> guard(mutex);
    if (shutdown != ConversationStoreShutdownState::none) {
        why = "the store is shutting down";
        return false;
    }
    if (!workspace) return true;
    if (workspace > options.metadata_bytes || preparation_locked() > options.metadata_bytes - workspace) {
        why = "the workspace a manifest read and decode needs does not fit the store's preparation ceiling";
        return false;
    }
    if (ram_budget != 0 && sat_add(resident_locked(), workspace) > ram_budget) {
        why = "the workspace a manifest read and decode needs does not fit the published RAM allowance";
        return false;
    }
    decode += workspace;
    counters.peak_bytes = std::max(counters.peak_bytes, resident_locked());
    return true;
}

void ConversationStore::Impl::release_decode_locked(uint64_t workspace) {
    std::lock_guard<std::mutex> guard(mutex);
    decode -= std::min(decode, workspace);
    cv_shutdown.notify_all();
}

// ---------------------------------------------------------------------------------------------------------------
// Deleting files, and crediting only what is really gone
// ---------------------------------------------------------------------------------------------------------------

bool ConversationStore::Impl::remove_pending_file(const PendingFile& file, std::string& why) {
    if (file.where != Where::objects) return remove_file(dirs.of(file.where), file.name, why);
    const int shard = open_shard_fd(dirs, file.shard, why);
    if (shard < 0) {
        const bool missing = errno == ENOENT;
        why.clear();
        return missing;   // a missing shard means the whole subtree is already gone
    }
    const bool ok = remove_file(shard, file.name, why);
    ::close(shard);
    return ok;
}

void ConversationStore::Impl::discard_files(std::vector<PendingFile>&& files) {
    std::vector<PendingFile> keep;
    for (PendingFile& file : files) {
        std::string why;
        if (remove_pending_file(file, why)) {
            std::lock_guard<std::mutex> guard(mutex);
            charged -= std::min(charged, file.charge);
            reclaimable -= std::min(reclaimable, file.charge);
            continue;
        }
        std::lock_guard<std::mutex> guard(mutex);
        counters.deferred_reclaims++;
        keep.push_back(std::move(file));
    }
    if (!keep.empty()) {
        std::lock_guard<std::mutex> guard(mutex);
        for (auto& file : keep) pending_files.push_back(std::move(file));
    }
}

void ConversationStore::Impl::queue_orphans(std::vector<PendingFile>& files) {
    std::lock_guard<std::mutex> guard(mutex);
    for (auto& file : files) {
        reclaimable += file.charge;
        pending_files.push_back(file);
    }
    files.clear();
}

void ConversationStore::Impl::reclaim_pending() {
    std::vector<PendingFile> batch;
    std::vector<PendingFile> retry;
    {
        std::lock_guard<std::mutex> guard(mutex);
        // After an uncertain catalog transaction this process cannot know which catalog is durable, so neither
        // catalog's files may be deleted: every pending file is retained, stays charged, and awaits a restart that
        // re-reads the authority.
        if (faulted) return;
        // Keep the hint armed if preparing the pass throws; only a prepared batch consumes this snapshot of work.
        reclaim_pending_work = true;
        // A final-lease release may have lacked room for its retire queue. Retry before reclaiming files.
        for (auto it = head_objects.begin(); it != head_objects.end();) {
            const uint64_t head = it->first;
            ++it;
            Digest manifest{};
            uint64_t bytes = 0;
            if (!head_leases.count(head) && !head_manifest_locked(head, manifest, bytes))
                retire_head_locked(head);
        }
        std::vector<PendingFile> keep;
        keep.reserve(pending_files.size());
        // Every move below stays inside the room reserved here, so a failed allocation can neither lose a pending
        // record between the queue and the batch nor leave the two lists describing the same file twice.
        batch.reserve(std::min<size_t>(pending_files.size(), kMaxReclaimPerPass));
        retry.reserve(batch.capacity());
        bool more = false;
        for (auto& file : pending_files) {
            // A file whose head still has an open hold or a handed-out image is not deleted yet: a reader's
            // descriptors may still name it, and its bytes are still in use.
            if (file.head_id != 0 && head_leases.count(file.head_id)) {
                keep.push_back(std::move(file));
                continue;
            }
            if (batch.size() >= kMaxReclaimPerPass) {
                more = true;
                keep.push_back(std::move(file));
                continue;
            }
            batch.push_back(std::move(file));
        }
        pending_files = std::move(keep);
        // Leased files wait for their final release, not an idle retry loop. Only a truncated eligible batch re-arms.
        reclaim_pending_work = more;
    }
    for (PendingFile& file : batch) {
        if (file.where == Where::objects) {
            bool referenced = false;
            {
                std::lock_guard<std::mutex> guard(mutex);
                const auto state = objects.find(file.object);
                // A lease is a reference: an in-flight publication has already named this object in the manifest it
                // is writing, so its file may not be deleted and credited even though no *committed* head names it
                // any more.  Keeping the file is what a later apply (or the lease's release) settles.
                referenced = (state != objects.end() && state->second.refs != 0) ||
                             object_leases.count(file.object) != 0;
            }
            if (referenced) {
                // Referenced again: the file is needed, so the pending record goes away and its charge stays.
                std::lock_guard<std::mutex> guard(mutex);
                reclaimable -= std::min(reclaimable, file.charge);
                continue;
            }
        }
        bool removed = false;
        try {
            std::string why;
            removed = remove_pending_file(file, why);
        } catch (const std::bad_alloc&) {
            // Even constructing the failure detail may allocate. Keep the preallocated retry record.
        }
        if (removed) {
            std::lock_guard<std::mutex> guard(mutex);
            charged -= std::min(charged, file.charge);
            reclaimable -= std::min(reclaimable, file.charge);
            if (file.where == Where::objects) {
                const auto state = objects.find(file.object);
                if (state != objects.end() && state->second.refs == 0) objects.erase(state);
                counters.reclaimed_objects++;
            }
            counters.reclaimed_bytes += file.charge;
            continue;
        }
        std::lock_guard<std::mutex> guard(mutex);
        counters.deferred_reclaims++;
        retry.push_back(std::move(file));
    }
    {
        std::lock_guard<std::mutex> guard(mutex);
        for (auto& file : retry) pending_files.push_back(std::move(file));
        if (!retry.empty()) reclaim_pending_work = true;
        recount_metadata_locked();
    }
}

void ConversationStore::Impl::recount_metadata_locked() {
    // Counted, never encoded: this runs after a rename, where an allocation for the sake of measuring one would be
    // exactly the escape the transaction must not have.
    uint64_t bytes = conversation_store_detail::catalog_encoded_size(catalog);
    bytes += uint64_t(objects.size()) * kObjectBytesEstimate;
    bytes = sat_add(bytes, sat_mul(uint64_t(pending_files.size()), kPendingMetadataBytes));
    for (const auto& entry : head_objects)
        bytes = sat_add(bytes, head_hold_metadata_bytes(entry.second.size()));
    metadata = bytes;
}

// ---------------------------------------------------------------------------------------------------------------
// Durable catalog transactions
// ---------------------------------------------------------------------------------------------------------------

// The physical charge of the catalog's *next* copy: the current catalog's encoding plus the most one commit can add
// to it, charged at the conservative allocated size the quota is kept in.
uint64_t ConversationStore::Impl::catalog_next_bound_locked(const Catalog& candidate) const {
    return allocated_charge(sat_add(conversation_store_detail::catalog_encoded_size(candidate),
                                    catalog_growth_bytes(limits)));
}

// The headroom ordinary admissions never spend.  Every commit needs its own copy to coexist with the live catalog
// (which `charged` already includes) and the copy after that to remain possible, so the reserve is
// max(live, next) + next.  Payloads, manifests and objects are admitted only while this much of the quota is still
// free, which is what makes an eviction - whose whole purpose is to make room - always writable.
uint64_t ConversationStore::Impl::catalog_reserve_locked() const {
    const uint64_t next = catalog_next_bound_locked(catalog);
    return sat_add(std::max(catalog_charge, next), next);
}

// The commit's own copy, admitted against the quota and the floor exactly like every other physical byte.  It is
// deliberately NOT routed through make_room(): a catalog commit never evicts recursively, and an admission that
// cannot be paid for is refused without deleting a single victim.
bool ConversationStore::Impl::admit_catalog_copy_locked(uint64_t copy_charge, const Catalog& candidate,
                                                        std::string& why) {
    sample_free_locked();
    if (free_sample == 0) {
        counters.refusals_floor++;
        why = "the filesystem could not be sampled, so the catalog's own copy cannot be admitted";
        return false;
    }
    const uint64_t next = catalog_next_bound_locked(candidate);
    // After this transaction the live catalog *is* this copy, so the committed figure is what is left of the quota
    // once the old copy is released, plus the headroom the transaction after this one needs.
    const uint64_t after = sat_add(charged - std::min(charged, catalog_charge), copy_charge);
    const uint64_t committed = sat_add(after, sat_add(std::max(copy_charge, next), next));
    if (committed > options.quota_bytes) {
        counters.refusals_quota++;
        why = "the catalog's own copy does not fit the shared quota (committed " + std::to_string(committed) +
              ", quota " + std::to_string(options.quota_bytes) + ")";
        return false;
    }
    if (copy_charge > free_sample || free_sample - copy_charge < options.min_free_bytes) {
        counters.refusals_floor++;
        why = "the catalog's own copy would cross the filesystem floor";
        return false;
    }
    return true;
}

// Everything the apply after the rename needs, allocated while the old authority is still the durable one.  The
// object-index nodes, the head's own index entries and the room the retire queue needs are built here; a commit that
// cannot build them changes nothing at all (its caller treats a false return as `unchanged`).
bool ConversationStore::Impl::stage_apply_locked(CommitPlan& plan, std::string& why) {
    if (!plan.new_head) return true;
    plan.staged_objects.clear();
    for (const Digest& object : plan.new_objects) {
        ObjectState state;
        const auto charge = plan.new_object_charges.find(object);
        state.bytes = charge == plan.new_object_charges.end() ? 0 : charge->second;
        if (charge == plan.new_object_charges.end()) {
            // Revival already carries a global physical charge. Transfer its value into the new index node,
            // not into charged again; lease release then leaves the live node responsible for eventual credit.
            const auto lease = object_leases.find(object);
            if (lease != object_leases.end()) state.bytes = lease->second.charge;
        }
        state.refs = 1;
        plan.staged_objects.emplace(object, state);
    }
    if (head_objects.find(plan.new_head) == head_objects.end()) {
        head_objects.emplace(plan.new_head, std::vector<Digest>{});
        plan.staged_head_objects = true;
    }
    if (head_manifest_bytes.find(plan.new_head) == head_manifest_bytes.end()) {
        head_manifest_bytes.emplace(plan.new_head, 0);
        plan.staged_head_manifest = true;
    }
    if (head_recency.find(plan.new_head) == head_recency.end()) {
        head_recency.emplace(plan.new_head, 0);
        plan.staged_head_recency = true;
    }
    // The retire path queues one manifest and every object it alone named, then hands the new head its own list: all
    // of it must fit without reallocating after the rename.
    uint64_t retiring_entries = 1;
    for (uint64_t victim : plan.retiring) {
        const auto list = head_objects.find(victim);
        retiring_entries = sat_add(retiring_entries, sat_add(1, list == head_objects.end() ? 0 : list->second.size()));
    }
    pending_files.reserve(size_t(sat_add(uint64_t(pending_files.size()), sat_add(retiring_entries, plan.new_objects.size()))));
    if (plan.staged_objects.size() != plan.new_objects.size()) {
        why = "the object index cannot hold this publication";
        unstage_apply_locked(plan);
        return false;
    }
    return true;
}

void ConversationStore::Impl::unstage_apply_locked(CommitPlan& plan) {
    if (plan.staged_head_objects) head_objects.erase(plan.new_head);
    if (plan.staged_head_manifest) head_manifest_bytes.erase(plan.new_head);
    if (plan.staged_head_recency) head_recency.erase(plan.new_head);
    plan.staged_head_objects = plan.staged_head_manifest = plan.staged_head_recency = false;
    plan.staged_objects.clear();
}

// The apply after the rename: node merges, moves and erases only.  Nothing here allocates, so nothing here can fail
// after the point of no return; the exception guard around it exists to keep even a bug from deleting files.
void ConversationStore::Impl::apply_staged_locked(CommitPlan& plan) {
    const bool newer_policy = recency_counter != plan.recency_at_copy;
    charged -= std::min(charged, catalog_charge);   // the previous catalog file is gone
    catalog_charge = plan.new_catalog_charge;
    if (plan.new_head) {
        for (const Digest& object : plan.new_objects) revive_object_locked(object);
        // The fresh nodes were built before the rename; merging them cannot allocate.  A key that already existed
        // keeps its own node (and its charge) and only its reference count changes.
        objects.merge(plan.staged_objects);
        for (const auto& entry : plan.staged_objects) {
            const auto state = objects.find(entry.first);
            if (state == objects.end()) continue;
            if (state->second.bytes == 0) state->second.bytes = entry.second.bytes;
            state->second.refs += 1;
        }
        plan.staged_objects.clear();
        // The list and the two per-head figures belong to nodes this transaction created before the rename.
        head_objects[plan.new_head] = std::move(plan.new_objects);
        head_manifest_bytes[plan.new_head] = plan.new_manifest_bytes;
        head_recency[plan.new_head] = ++recency_counter;
        plan.staged_head_objects = plan.staged_head_manifest = plan.staged_head_recency = false;
    }
    if (newer_policy) {
        // The encoded candidate stays immutable. Only RAM policy is carried forward: a restore after the copy has
        // already re-armed promote_pending (or close deferred it to its final transaction), and must not be lost when
        // this older authority snapshot applies. Covered successors inherit their predecessor's latest policy.
        for (auto& head : plan.candidate.heads) {
            const uint64_t source = head.head_id == plan.new_head && plan.replaced_head
                ? plan.replaced_head : head.head_id;
            for (const auto& live : catalog.heads)
                if (live.head_id == source) {
                    head.protected_seg = live.protected_seg;
                    break;
                }
        }
        cap_protected_locked(plan.candidate);
    }
    for (uint64_t victim : plan.retiring) retire_head_locked(victim);
    catalog = std::move(plan.candidate);
    catalog_version++;
    recount_metadata_locked();
    counters.heads = uint64_t(catalog.heads.size());
    counters.objects = uint64_t(objects.size());
    counters.manifests = uint64_t(head_manifest_bytes.size());
}

StoreReadTrace ConversationStore::Impl::trace_head_locked(const CatalogHead& head, const char* event,
                                                         const char* reason) const {
    auto record = trace_state_locked(event, reason, ConversationStoreReadStatus::none);
    record.head_event = true;
    record.head = head.head_id;
    record.epoch = head.epoch;
    record.catalog_version = catalog_version;
    const auto lease = head_leases.find(head.head_id);
    record.head_leases = lease == head_leases.end() ? 0 : lease->second;
    record.pinned = head.pinned;
    record.protected_seg = head.protected_seg;
    record.recency = recency_of_locked(head);
    record.stages = head.stages;
    record.charged_bytes = charged;
    record.free_bytes = free_sample;
    record.quota_bytes = options.quota_bytes;
    record.floor_bytes = options.min_free_bytes;
    for (const auto& cut : head.cuts) {
        record.cut = std::max(record.cut, uint64_t(cut.tokens));
        if (cut.live) record.live_tokens = std::max(record.live_tokens, uint64_t(cut.tokens));
    }
    return record;
}

void ConversationStore::Impl::trace_retirement_locked(StoreReadTraceBatch& trace, const CommitPlan& plan,
                                                      const char* event) const {
    if (!read_timing) return;
    for (uint64_t victim : plan.retiring) {
        for (const auto& head : catalog.heads) {
            if (head.head_id != victim) continue;
            const char* reason = "head_limit";
            if (plan.kind == CommitPlan::Kind::eviction) reason = plan.eviction_reason;
            else if (plan.kind == CommitPlan::Kind::invalidate) reason = "domain_invalidated";
            else if (victim == plan.replaced_head) reason = "covered_replacement";
            auto record = trace_head_locked(head, event, reason);
            record.request_id = plan.request_id;
            record.generation = plan.generation;
            record.plan_version = plan.version;
            record.replacement_head = plan.new_head;
            trace.add(record);
            break;
        }
    }
}

void ConversationStore::Impl::trace_publication_locked(StoreReadTraceBatch& trace, const CommitPlan& plan) const {
    if (!read_timing || plan.kind != CommitPlan::Kind::publish || !plan.new_head) return;
    for (const auto& head : catalog.heads) {
        if (head.head_id != plan.new_head) continue;
        StoreReadTrace record{};
        record.head_event = true;
        record.event = "published_cut";
        record.reason = "all_stage";
        record.ts_ns = store_timing_ns();
        record.request_id = plan.request_id;
        record.generation = plan.generation;
        record.head = head.head_id;
        record.epoch = head.epoch;
        record.live_tokens = plan.live_tokens;
        record.stages = head.stages;
        record.plan_version = plan.version;
        record.catalog_version = catalog_version;
        const auto lease = head_leases.find(head.head_id);
        record.head_leases = lease == head_leases.end() ? 0 : lease->second;
        record.pinned = head.pinned;
        record.protected_seg = head.protected_seg;
        record.recency = recency_of_locked(head);
        record.charged_bytes = charged;
        record.free_bytes = free_sample;
        record.quota_bytes = options.quota_bytes;
        record.floor_bytes = options.min_free_bytes;
        for (size_t i = 0; i < head.cuts.size(); ++i) {
            if (trace.count == sizeof trace.records / sizeof trace.records[0]) {
                trace.dropped += head.cuts.size() - i;
                break;
            }
            record.cut = head.cuts[i].tokens;
            record.live_cut = head.cuts[i].live;
            trace.add(record);
        }
        return;
    }
}

void ConversationStore::Impl::copy_catalog_locked(CommitPlan& plan) {
    // Consumption and the copy share one mutex scope. Clear even if allocation refuses the copy: the live policy
    // remains available to the next real transaction, but an optional failure must not schedule its own retry.
    if (promote_pending) {
        promote_pending = false;
        if (plan.kind != CommitPlan::Kind::promote) counters.promotions_absorbed++;
    }
    plan.recency_at_copy = recency_counter;
    plan.candidate = catalog;
}

CommitOutcome ConversationStore::Impl::write_catalog_or_refuse(CommitPlan& plan, std::string& why) {
    StoreReadTraceBatch trace;
    std::vector<uint8_t> encoded;
#ifdef STRATA_STORE_TEST_HOOKS
    void (*gate)(void*, bool) = nullptr;
    void* gate_context = nullptr;
    {
        std::lock_guard<std::mutex> guard(mutex);
        gate = catalog_gate;
        gate_context = catalog_gate_context;
    }
    if (gate) gate(gate_context, false);
#endif
    // ---- plan, count, admit and stage, under the lock, before any durable byte changes.  Every failure below
    // leaves the previously committed catalog as the authority ----
    try {
        std::lock_guard<std::mutex> guard(mutex);
        // The guard lives here rather than only at the public entry points: a fault that must stop reuse stops
        // queued durable work too, and a queued transaction must never be the first thing to notice it.
        if (faulted) {
            why = fault_reason.empty() ? "the store has faulted" : fault_reason;
            return CommitOutcome::unchanged;
        }
        if (!lock_owned) {
            why = "this instance does not own the store's authority";
            return CommitOutcome::unchanged;
        }
        switch (plan.kind) {
            case CommitPlan::Kind::invalidate: {
                // Re-derived at commit time, so an intervening commit can never be overwritten by a stale snapshot.
                copy_catalog_locked(plan);
                if (plan.domain_index < plan.candidate.domains.size())
                    plan.candidate.domains[plan.domain_index].epoch = plan.epoch;
                std::vector<CatalogHead> keep;
                keep.reserve(plan.candidate.heads.size());
                for (const auto& head : plan.candidate.heads)
                    if (head.domain == plan.domain_index)
                        plan.retiring.push_back(head.head_id);
                    else
                        keep.push_back(head);
                plan.candidate.heads = std::move(keep);
                break;
            }
            case CommitPlan::Kind::promote:
                // "Write the live catalog durably": a promotion is already reflected in its policy flags.
                copy_catalog_locked(plan);
                break;
            default:
                if (plan.version != catalog_version) {
                    why = "the catalog changed while this plan was being built";
                    return CommitOutcome::unchanged;
                }
                break;
        }
        // Selection and retirement planning use the same mutex. A lease acquired after victim choice but before
        // staging wins; a lease acquired after staging owns its captured manifest and files through durable apply.
        // Covered replacement is allowed: its old request-owned authority remains usable, without pinning a slot.
        if (plan.kind != CommitPlan::Kind::invalidate)
            for (uint64_t victim : plan.retiring)
                if (victim != plan.replaced_head && head_leases.count(victim)) {
                    why = "a selected-head lease was acquired before retirement staging";
                    return CommitOutcome::unchanged;
                }
        encoded = conversation_store_detail::catalog_encode(plan.candidate);
        plan.new_catalog_charge = allocated_charge(encoded.size());
        if (!admit_catalog_copy_locked(plan.new_catalog_charge, plan.candidate, why))
            return CommitOutcome::unchanged;
        if (!stage_apply_locked(plan, why)) return CommitOutcome::unchanged;
        charged += plan.new_catalog_charge;
        trace_retirement_locked(trace, plan, "planned");
    } catch (const std::exception& error) {
        why = std::string("the catalog transaction could not be prepared: ") + error.what();
        return CommitOutcome::unchanged;
    } catch (...) {
        why = "the catalog transaction could not be prepared";
        return CommitOutcome::unchanged;
    }
    trace.emit(); // planned is not retired; logging never holds the catalog mutex
#ifdef STRATA_STORE_TEST_HOOKS
    if (gate) gate(gate_context, true);
#endif

    std::string temp = temp_name();
    bool written = false;
    for (uint32_t attempt = 0; attempt < kTempNameAttempts && !written; ++attempt) {
        if (write_new_file(dirs.tmp, temp, encoded.data(), encoded.size(), why)) {
            written = true;
            break;
        }
        if (errno != EEXIST) break;
        temp = temp_name();
        why.clear();
    }
    bool renamed = false;
    if (written) {
        if (take_fault("catalog-rename")) {
            why = "injected fault: the catalog rename";
        } else if (::renameat(dirs.tmp, temp.c_str(), dirs.root, "catalog") != 0) {
            why = std::string("cannot replace the catalog: ") + std::strerror(errno);
        } else {
            renamed = true;
        }
    }
    if (!renamed) {
        // The catalog itself is untouched (a rename either happened or did not), so the authority is unchanged and
        // only the temporary this attempt created may remain.  It is cleaned here and its charge is credited only
        // once the file is confirmed gone; a cleanup that cannot confirm that leaves the bytes charged.
        {
            std::lock_guard<std::mutex> guard(mutex);
            unstage_apply_locked(plan);
        }
        std::vector<PendingFile> orphans;
        PendingFile orphan;
        orphan.where = Where::tmp;
        orphan.name = temp;
        orphan.charge = plan.new_catalog_charge;
        orphans.push_back(std::move(orphan));
        discard_files(std::move(orphans));
        return CommitOutcome::unchanged;
    }
    // From here the new catalog exists under the live name.  Recording that immediately - before the first fallible
    // step after it - is what separates "nothing happened" from "something happened and durability is unknown".
    bool durable = true;
    if (take_fault("catalog-sync")) {
        why = "injected fault: the store root could not be flushed";
        durable = false;
    } else if (!fsync_dir(dirs.root, "the store root", why)) {
        durable = false;
    }
    if (!durable) {
        // Uncertainty, not failure: the durable catalog is the new one *or* the old one, and this process cannot
        // tell which.  No file may be deleted or credited, reuse stops, further transactions stop, and the run
        // intent stays armed so the next start re-reads the authority instead of trusting this view.
        const std::string detail =
            "the new catalog was renamed into place but its durability is not confirmed (" + why +
            "); every file either catalog may name is retained and this store stops serving";
        {
            std::lock_guard<std::mutex> guard(mutex);
            unstage_apply_locked(plan);
            latch_fault_locked(detail);
        }
        why = detail;
        return CommitOutcome::uncertain;
    }
    try {
        // A failure that only appears while the apply is allocating is the same class of uncertainty as a failed
        // flush: the renamed file is in place and only a restart can say which catalog is durable.  The injected
        // fault exists so that path is exercised rather than merely reachable.
        if (take_fault("catalog-apply")) throw std::bad_alloc();
        std::lock_guard<std::mutex> guard(mutex);
        trace_retirement_locked(trace, plan, "retired");
        apply_staged_locked(plan);
        plan.applied_catalog_version = catalog_version;
        // Count removals only at the same successful authority boundary as the catalog receipt.
        // A rejected/stale/uncertain plan is not an eviction; a covered replacement is not capacity pressure.
        if (plan.kind == CommitPlan::Kind::publish || plan.kind == CommitPlan::Kind::eviction)
            for (uint64_t victim : plan.retiring)
                if (victim != plan.replaced_head) counters.evictions++;
        for (size_t i = 0; i < trace.count; ++i)
            trace.records[i].catalog_version = catalog_version;
        trace_publication_locked(trace, plan);
        const uint64_t applied = options.publication_timing || read_timing ? store_timing_ns() : 0;
        if (options.publication_timing || read_timing) plan.applied_ns = applied;
        if (trace.count)
            for (size_t i = 0; i < trace.count; ++i) trace.records[i].finish_ns = applied;
    } catch (const std::exception& error) {
        trace.count = 0;
        trace.dropped = 0; // an uncertain apply must not emit a confirmed retirement
        const std::string detail = std::string("the catalog apply failed after the rename: ") + error.what();
        std::lock_guard<std::mutex> guard(mutex);
        latch_fault_locked(detail);
        why = detail;
        return CommitOutcome::uncertain;
    } catch (...) {
        trace.count = 0;
        trace.dropped = 0;
        const std::string detail = "the catalog apply failed after the rename";
        std::lock_guard<std::mutex> guard(mutex);
        latch_fault_locked(detail);
        why = detail;
        return CommitOutcome::uncertain;
    }
    trace.emit(); // only a confirmed durable apply emits retired
    if (plan.defer_reclaim) return CommitOutcome::durable_applied;
    // Deletions happen here: the authority that named these files is already durable, and the mutex is released.
    // The transaction is committed at this point, so a reclamation that cannot allocate its own batch must never be
    // reported as a transaction failure: the caller would then treat the manifest and the objects this commit just
    // authorized as orphans and queue them for deletion.  A failure here leaves every pending file pending and
    // charged - the safe direction - and the next pass, or the next start, reclaims them.
    try {
        if (take_fault("catalog-reclaim")) throw std::bad_alloc();
        reclaim_pending();
    } catch (...) {
        // Nothing durable changes: the committed catalog is already the authority and no file was credited for a
        // deletion that did not happen.
    }
    return CommitOutcome::durable_applied;
}

CommitOutcome ConversationStore::Impl::commit_plan(CommitPlan& plan, std::string& why) {
    return write_catalog_or_refuse(plan, why);
}

bool ConversationStore::Impl::evict_head_durably(uint64_t head_id, std::string& why,
                                                const char* retirement_reason) {
    CommitPlan plan;
    plan.kind = CommitPlan::Kind::eviction;
    plan.eviction_reason = retirement_reason;
    {
        std::lock_guard<std::mutex> guard(mutex);
        plan.version = catalog_version;
        if (publishing && publishing->started) {
            plan.request_id = publishing->request.request_id;
            plan.generation = publishing->request.generation;
        }
        copy_catalog_locked(plan);
        bool found = false;
        for (size_t i = 0; i < plan.candidate.heads.size(); ++i) {
            if (plan.candidate.heads[i].head_id != head_id) continue;
            plan.candidate.heads.erase(plan.candidate.heads.begin() + std::ptrdiff_t(i));
            found = true;
            break;
        }
        if (!found) {
            why = "no retained head has that identity";
            return false;
        }
        plan.retiring.push_back(head_id);
    }
    return commit_plan(plan, why) == CommitOutcome::durable_applied;
}

// Frees `incoming_bytes` before they are written.  Only physically present, unreferenced charge is counted, and an
// eviction is *durable* (catalog committed, files deleted) before the new bytes are charged: no promise of a later
// deletion ever authorizes a new file.
bool ConversationStore::Impl::make_room(uint64_t incoming_bytes, bool& floor_limited, std::string& why) {
    floor_limited = false;
    for (;;) {
        uint64_t victim = 0;
        const char* retirement_reason = "space";
        {
            std::lock_guard<std::mutex> guard(mutex);
            sample_free_locked();
            if (free_sample == 0) {
                // An unreadable filesystem is never treated as an empty one.
                counters.refusals_floor++;
                floor_limited = true;
                why = "the filesystem could not be sampled, so no reservation can be admitted";
                return false;
            }
            // The catalog's own next copy is never spent by a payload, a manifest or an object: ordinary admissions
            // keep that headroom free, so the commit that follows them - and the eviction commit that makes room for
            // them - can always be written, and no deletion is ever credited before it happens.
            const uint64_t reserve = catalog_reserve_locked();
            const uint64_t committed = sat_add(charged, sat_add(incoming_bytes, reserve));
            const bool over_quota = committed > options.quota_bytes;
            const bool over_floor = incoming_bytes > free_sample ||
                free_sample - incoming_bytes < options.min_free_bytes;
            floor_limited = over_floor;
            if (!over_quota && !over_floor) return true;
            if (read_timing)
                retirement_reason = over_quota ? (over_floor ? "quota_and_floor" : "byte_quota") : "free_space_floor";
            std::set<uint64_t> blocked = eviction_excluded;
            victim = choose_victim_locked(catalog, blocked, false);
            if (victim == 0) victim = choose_victim_locked(catalog, blocked, true);
            if (victim == 0) {
                if (over_quota) counters.refusals_quota++;
                if (over_floor) counters.refusals_floor++;
                why = over_quota
                          ? "the shared quota is full and no unpinned, unleased head can be evicted"
                          : "the filesystem floor would be crossed and no unpinned, unleased head can be evicted";
                why += " (charged " + std::to_string(charged) + ", reserved for the catalog " +
                    std::to_string(reserve) + ", incoming " + std::to_string(incoming_bytes) + ", quota " +
                    std::to_string(options.quota_bytes) + ")";
                return false;
            }
        }
        // An eviction that itself cannot be committed - a quota or a floor that even a catalog copy cannot be paid
        // from - is a refusal: it deletes nothing, so it never turns a promise of space into an admitted byte.
        if (!evict_head_durably(victim, why, retirement_reason)) return false;
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Matching: catalog metadata only, no payload is read and no GPU pointer is exposed
// ---------------------------------------------------------------------------------------------------------------


// Opt-in: name every input of the bucket digest so a cross-process miss can be attributed to one
// component instead of the whole key.  Never printed without STRATA_STORE_TIMING.
static void trace_bucket(const char* where, const ConversationStoreOptions& options,
                         const ConversationStoreSelector& selector, const Digest& bucket) {
    if (!std::getenv("STRATA_STORE_TIMING")) return;
    uint64_t d0 = 0, m0 = 0, e0 = 0, b0 = 0, s0 = 0;
    std::memcpy(&d0, options.domain.data(), sizeof d0);
    std::memcpy(&m0, options.model.data(), sizeof m0);
    std::memcpy(&e0, options.engine.data(), sizeof e0);
    std::memcpy(&b0, bucket.data(), sizeof b0);
    std::memcpy(&s0, selector.scope.value.data(), sizeof s0);
    std::fprintf(stderr, "strata serve: conversation store: bucket %s=%016llx domain=%016llx model=%016llx "
                         "engine=%016llx scope=%016llx cvec=%d geometry=%d layer=%lld:%lld\n",
                 where, (unsigned long long) b0, (unsigned long long) d0, (unsigned long long) m0,
                 (unsigned long long) e0, (unsigned long long) s0, int(selector.cvec),
                 int(selector.geometry.size()), (long long) selector.layer_lo, (long long) selector.layer_hi);
}

void ConversationStore::Impl::collect_matches(const ConversationStoreSelector& selector, const int32_t* ids,
                                             size_t count, const std::vector<ConversationImageKey>& images,
                                             std::vector<ConversationStoreMatch>& out, StoreReadTraceBatch* trace,
                                             const ConversationStoreQuery* query, bool demand) const {
    out.clear();
    if (!on || faulted || shutdown != ConversationStoreShutdownState::none) return;
    const Digest bucket = conversation_store_detail::store_bucket_digest(options, selector);
    trace_bucket("lookup", options, selector, bucket);
    conversation_store_detail::PrefixChain chain(bucket, ids, count, images);
    size_t rejections = 0;
    uint64_t omitted = 0;
    auto reject = [&](const CatalogHead& head, const auto& cut, const char* reason) {
        if (!read_timing || !trace || !query) return;
        // Leave room for the selection/adoption ownership records in this fixed batch. Candidate fanout must
        // not silently displace the very transition the diagnostics are intended to explain.
        if (rejections == 8 || trace->count >= 48) {
            ++omitted;
            return;
        }
        ++rejections;
        auto record = trace_state_locked("candidate_rejected", reason, ConversationStoreReadStatus::none);
        record.request_id = query->request_id;
        record.generation = query->generation;
        record.demand = demand;
        record.prefetched = !demand;
        record.head = head.head_id;
        record.cut = cut.tokens;
        record.epoch = head.epoch;
        trace->add(record);
    };
    for (const auto& head : catalog.heads) {
        if (!(head.bucket == bucket)) continue; // unrelated namespaces never enter diagnostics
        if (head.epoch != epoch) {
            if (read_timing && trace)
                for (const auto& cut : head.cuts) reject(head, cut, "epoch_mismatch");
            continue;
        }
        for (const auto& cut : head.cuts) {
            // The last prompt token always starts the next verify window, so only a strictly shorter prefix is usable.
            if (cut.tokens == 0 || cut.tokens >= count) {
                if (read_timing && trace) reject(head, cut, "not_strict_prefix");
                continue;
            }
            Digest digest{};
            if (!chain.digest(cut.tokens, digest)) {
                if (read_timing && trace) reject(head, cut, "prefix_unavailable");
                continue;
            }
            if (!(digest == cut.digest)) {
                if (read_timing && trace) reject(head, cut, "prefix_mismatch");
                continue;
            }
            ConversationStoreMatch match;
            match.head_id = head.head_id;
            match.tokens = int64_t(cut.tokens);
            match.live = cut.live;
            match.pinned = cut.pinned;
            match.stages = head.stages;
            match.payload_bytes = head.payload_bytes;
            match.manifest_bytes = head.manifest_bytes;
            match.published_order = head.published_order;
            out.push_back(match);
        }
    }
    if (omitted) {
        auto record = trace_state_locked("candidate_rejections_omitted", "trace_bound",
                                         ConversationStoreReadStatus::none);
        record.request_id = query->request_id;
        record.generation = query->generation;
        record.demand = demand;
        record.prefetched = !demand;
        record.omitted_rejections = omitted;
        trace->add(record);
    }
    std::stable_sort(out.begin(), out.end(), [](const ConversationStoreMatch& a, const ConversationStoreMatch& b) {
        if (a.tokens != b.tokens) return a.tokens > b.tokens;
        if (a.published_order != b.published_order) return a.published_order > b.published_order;
        if (a.live != b.live) return a.live;
        return a.pinned && !b.pinned;
    });
}

// ---------------------------------------------------------------------------------------------------------------
// Reads: a bounded, cancellable pipeline.  A demand is admitted as soon as its whole peak fits the allowance and
// never loses its slot to a prefetch; a prefetch is speculative and is refused or revoked instead.
// ---------------------------------------------------------------------------------------------------------------

StoreReadTrace ConversationStore::Impl::trace_state_locked(const char* event, const char* why,
                                                           ConversationStoreReadStatus status) const {
    StoreReadTrace record{};
    record.event = event;
    record.reason = why;
    record.status = status;
    record.ts_ns = store_timing_ns();
    record.epoch = epoch;
    record.resident_bytes = resident_locked();
    record.budget_bytes = ram_budget;
    record.preparation_bytes = preparation_locked();
    record.metadata_limit = options.metadata_bytes;
    record.incoming = incoming;
    record.max_reads = options.max_reads;
    record.queued = queue.size();
    record.descriptors = jobs.size();
    return record;
}

uint64_t ConversationStore::Impl::trace_wait_locked(const ReadJob& job) const {
    if (job.active || job.status != ConversationStoreReadStatus::none) return 0;
    // Independent blockers, not a guessed cause or a sum of durations: 1 = no published allowance,
    // 2 = suspended, 4 = incoming slots held, 8 = resident ledger + peak cannot fit, 16 = reader workers occupied.
    // Count real active owners, including retiring workers; ready/handed leases occupy slots but not workers.
    uint64_t active_readers = 0;
    for (const auto& candidate : jobs)
        if (candidate->active) ++active_readers;
    return uint64_t(ram_budget == 0) | (uint64_t(admission_suspended) << 1) |
           (uint64_t(incoming >= options.max_reads) << 2) |
           (uint64_t(ram_budget != 0 && sat_add(resident_locked(), job.peak_bytes) > ram_budget) << 3) |
           (uint64_t(active_readers >= reader_threads) << 4);
}

void ConversationStore::Impl::trace_job_locked(StoreReadTraceBatch& trace, const ReadJob& job, const char* event,
                                               const char* why, ConversationStoreReadStatus status) const {
    if (!read_timing) return;
    auto record = trace_state_locked(event, why,
        status == ConversationStoreReadStatus::none ? job.status : status);
    record.request_id = job.query.request_id;
    record.generation = job.query.generation;
    record.attempt_id = job.trace_attempt;
    record.previous_attempt_id = job.trace_previous;
    record.head = job.head_id;
    record.cut = job.tokens;
    record.epoch = job.epoch_at_admission;
    record.enqueue_ns = job.trace_enqueue;
    record.admit_ns = job.trace_admit;
    record.start_ns = job.trace_start;
    record.finish_ns = job.trace_finish;
    record.ready_ns = job.trace_ready;
    record.ready_end_ns = job.trace_ready_end;
    record.body_lock_wait_ns = job.trace_body_lock_wait;
    record.completion_lock_wait_ns = job.trace_completion_lock_wait;
    record.peak_bytes = job.peak_bytes;
    record.reserved_bytes = job.reserved_bytes;
    record.lease_bytes = job.lease_bytes;
    record.prepared_bytes = job.prepared_bytes;
    record.wait_mask = trace_wait_locked(job);
    record.demand = job.demand;
    record.prefetched = job.prefetched;
    record.active = job.active;
    record.holds_slot = job.holds_slot;
    record.cancelled = job.cancelled.load();
    record.io_attempted = job.trace_io_attempted;
    record.bytes_read = job.trace_bytes_read;
    trace.add(record);
}

void ConversationStore::Impl::trace_refusal_locked(StoreReadTraceBatch& trace, const ConversationStoreQuery& query,
                                                   bool demand, ConversationStoreReadStatus status, const char* why,
                                                   uint64_t previous, uint64_t head, uint64_t cut, uint64_t peak) const {
    if (!read_timing) return;
    auto record = trace_state_locked("admission_refused", why, status);
    record.request_id = query.request_id;
    record.generation = query.generation;
    record.attempt_id = store_read_attempt.fetch_add(1, std::memory_order_relaxed) + 1;
    record.previous_attempt_id = previous;
    record.demand = demand;
    record.prefetched = !demand;
    record.head = head;
    record.cut = cut;
    record.peak_bytes = peak;
    trace.add(record);
}

// The real attempt boundary also owns always-on byte progress; opt-in timing uses that same identity.
void ConversationStore::Impl::trace_begin_locked(ReadJob& job, uint64_t previous) {
    const bool prior_unmeasured = job.progress.prior_materialize_unmeasured || job.progress.materialize_started;
    job.progress = {};
    job.prefetched = !job.demand;
    job.progress.prior_materialize_unmeasured = prior_unmeasured;
    job.progress.attempt_id = store_read_attempt.fetch_add(1, std::memory_order_relaxed) + 1;
    if (!read_timing) return;
    job.trace_attempt = job.progress.attempt_id;
    job.trace_previous = previous;
    job.trace_enqueue = store_timing_ns();
    job.trace_admit = job.trace_start = job.trace_finish = job.trace_ready = 0;
    job.trace_ready_end = 0;
    job.trace_body_lock_wait = job.trace_completion_lock_wait = 0;
    job.trace_io_attempted = false;
    job.trace_bytes_read = 0;
    job.trace_retire_reason = "none";
    job.trace_wait_mask = trace_wait_locked(job);
}

void ConversationStore::Impl::trace_waiters_locked(StoreReadTraceBatch& trace) {
    if (!read_timing) return;
    for (const auto& job : queue) {
        if (job->cancelled || job->released) continue;
        const uint64_t mask = trace_wait_locked(*job);
        if (mask == job->trace_wait_mask) continue;
        job->trace_wait_mask = mask;
        trace_job_locked(trace, *job, "wait", "resources");
    }
}

void ConversationStore::Impl::release_reservation_locked(const std::shared_ptr<ReadJob>& job) {
    if (!job->reserved_bytes) return;
    ram_held -= std::min(ram_held, job->reserved_bytes);
    job->reserved_bytes = 0;
}

void ConversationStore::Impl::release_lease_locked(const std::shared_ptr<ReadJob>& job) {
    if (!job->lease_bytes) return;
    ram_held -= std::min(ram_held, job->lease_bytes);
    ram_lease -= std::min(ram_lease, job->lease_bytes);
    job->lease_bytes = 0;
}

void ConversationStore::Impl::release_slot_locked(const std::shared_ptr<ReadJob>& job) {
    if (!job->holds_slot) return;
    job->holds_slot = false;
    if (incoming) --incoming;
}

// The hold that keeps a retired head's files on disk while a reader still names them.  It ends exactly where the
// job's ownership of the head ends - which is after the lease for a handed-out image.
void ConversationStore::Impl::release_head_lease_locked(const std::shared_ptr<ReadJob>& job) {
    if (!job->leased) return;
    job->leased = false;
    if (!job->head_id) return;
    const auto lease = head_leases.find(job->head_id);
    if (lease == head_leases.end()) return;
    if (lease->second > 0) --lease->second;
    if (lease->second == 0) {
        head_leases.erase(lease);
        Digest manifest{};
        uint64_t bytes = 0;
        if (head_manifest_bytes.count(job->head_id) && !head_manifest_locked(job->head_id, manifest, bytes)) {
            reclaim_pending_work = true;
            cv_work.notify_all();
            try {
                retire_head_locked(job->head_id);
                recount_metadata_locked();
            } catch (const std::bad_alloc&) {
                // Keep refs and their charge if the retire queue cannot grow; reclaim_pending retries it.
            }
        }
    }
}

// The read's own preparation beyond its query copy - the manifest read and decode it reserved and the scaffolding it
// restored into - stops being the store's preparation the moment its image is handed out or its descriptor is dropped:
// from the hand-out on, those bytes are the caller's image, charged as its lease.  Released on the ready transition
// and whenever the descriptor is erased, so `read_prep` never keeps a view of bytes that are no longer live.
void ConversationStore::Impl::release_job_preparation_locked(const std::shared_ptr<ReadJob>& job) {
    read_prep -= std::min(read_prep, job->prepared_bytes);
    job->prepared_bytes = 0;
}

void ConversationStore::Impl::free_job_prep_locked(const std::shared_ptr<ReadJob>& job) {
    release_job_preparation_locked(job);
    prep -= std::min(prep, job->prep_bytes);
    job->prep_bytes = 0;
    job->query.tokens.clear();
    job->query.tokens.shrink_to_fit();
    job->query.images.clear();
    job->query.images.shrink_to_fit();
}

void ConversationStore::Impl::erase_job_locked(const std::shared_ptr<ReadJob>& job) {
    release_head_lease_locked(job);
    free_job_prep_locked(job);
    for (auto it = jobs.begin(); it != jobs.end(); ++it)
        if (*it == job) {
            jobs.erase(it);
            break;
        }
}

void ConversationStore::Impl::settle_job_locked(const std::shared_ptr<ReadJob>& job,
                                                ConversationStoreReadStatus status, std::string detail,
                                                StoreReadTraceBatch& trace) {
    if (job->status != ConversationStoreReadStatus::none) return;   // exactly one settlement per job
    job->active = false;
    job->status = status;
    job->detail = std::move(detail);
    switch (status) {
        case ConversationStoreReadStatus::ready: counters.reads_served++; break;
        case ConversationStoreReadStatus::cancelled: counters.reads_cancelled++; break;
        case ConversationStoreReadStatus::stale: counters.reads_stale++; break;
        case ConversationStoreReadStatus::corrupt: counters.reads_corrupt++; break;
        case ConversationStoreReadStatus::failed: counters.reads_failed++; break;
        case ConversationStoreReadStatus::refused_budget: counters.refused_budget++; break;
        case ConversationStoreReadStatus::refused_quota: counters.refused_quota++; break;
        case ConversationStoreReadStatus::refused_limits: counters.refused_limits++; break;
        default: break;
    }
    if (job->refresh_on_completion) {
        job->refresh_on_completion = false;
        if (refresh_read_locked(job, trace)) return;
    }
    if (status == ConversationStoreReadStatus::ready && job->image) {
        // The reservation is refunded and the image's own bytes are charged as its lease; the read's preparation view
        // ends here too, because from now on those bytes are the image the caller receives.
        release_reservation_locked(job);
        release_job_preparation_locked(job);
        const uint64_t bytes = job->image->bytes();
        ram_held += bytes;
        ram_lease += bytes;
        job->lease_bytes = bytes;
        counters.peak_bytes = std::max(counters.peak_bytes, resident_locked());
        if (job->released) {
            // The caller already released this request: the image is dropped and its charge is not kept.
            release_lease_locked(job);
            release_slot_locked(job);
            release_head_lease_locked(job);
            job->image.reset();
            erase_job_locked(job);
            trace_job_locked(trace, *job, "settled", "released");
            return;
        }
        ready.push_back(job);
        if (read_timing) job->trace_ready = store_timing_ns();
        trace_job_locked(trace, *job, "settled", "ready");
        cv_ready.notify_all();
        return;
    }
    release_reservation_locked(job);
    release_lease_locked(job);
    release_slot_locked(job);
    job->budget_deferred = status == ConversationStoreReadStatus::refused_budget && !job->demand &&
        !job->cancelled && !job->released && job->epoch_at_admission == epoch && job->head_id && job->leased;
    if (job->budget_deferred) {
        const uint64_t held = resident_locked();
        job->deferred_available = admission_suspended || incoming >= options.max_reads || ram_budget <= held ?
            0 : ram_budget - held;
    }
    // An accepted budget-deferred selection keeps only its already-counted query/decode and head hold. No
    // image, incoming slot or peak reservation survives; explicit retirement still ends the hold.
    if ((!job->speculation_revoked && !job->budget_deferred) || job->released) release_head_lease_locked(job);
    job->image.reset();
    if (job->released) {
        erase_job_locked(job);
        trace_job_locked(trace, *job, "settled", "released");
        return;
    }
    if (!job->cancelled) {
        ready.push_back(job);
        if (read_timing) job->trace_ready = store_timing_ns();
        trace_job_locked(trace, *job, "settled", job->trace_start ? "read_result" : "no_io");
        cv_ready.notify_all();
        return;
    }
    // A cancelled request reports its terminal status but is never delivered; its descriptor is spent.
    if (job->speculation_revoked && !job->released)
        release_job_preparation_locked(job);
    else
        free_job_prep_locked(job);
    trace_job_locked(trace, *job, "settled", job->trace_retire_reason);
    cv_ready.notify_all();
}

void ConversationStore::Impl::retire_job_locked(const std::shared_ptr<ReadJob>& job,
                                                ConversationStoreReadStatus status, const std::string& detail,
                                                StoreReadTraceBatch& trace, const char* why) {
    if (job->handed) return;                       // already the caller's image: release_read() is its way out
    if (!job->speculation_revoked) job->budget_deferred = false;
    if (read_timing && !job->cancelled) job->trace_retire_reason = why;
    if (read_timing && job->trace_ready && !job->trace_ready_end) job->trace_ready_end = store_timing_ns();
    job->cancelled = true;
    if (job->active) {
        trace_job_locked(trace, *job, "retire", why, status);
        return;                                   // the reader drops it at its next chunk boundary
    }
    for (auto q = queue.begin(); q != queue.end(); ++q)
        if (*q == job) {
            queue.erase(q);
            break;
        }
    for (auto r = ready.begin(); r != ready.end(); ++r)
        if (*r == job) {
            ready.erase(r);
            break;
        }
    release_reservation_locked(job);
    release_lease_locked(job);
    release_slot_locked(job);
    if (!job->speculation_revoked || job->released) release_head_lease_locked(job);
    job->image.reset();
    // A job that already holds a *refusal* keeps its own verdict; a queued or ready completion takes the authority's
    // terminal status, because a completion that can no longer be handed out must never keep reporting `ready`.
    if (job->status != ConversationStoreReadStatus::none &&
        job->status != ConversationStoreReadStatus::ready) {
        if (job->speculation_revoked && !job->released) release_job_preparation_locked(job);
        else free_job_prep_locked(job);
        trace_job_locked(trace, *job, "retire", why);
        return;
    }
    job->status = status;
    job->detail = detail;
    switch (status) {
        case ConversationStoreReadStatus::stale: counters.reads_stale++; break;
        case ConversationStoreReadStatus::cancelled: counters.reads_cancelled++; break;
        default: break;
    }
    if (job->speculation_revoked && !job->released) release_job_preparation_locked(job);
    else free_job_prep_locked(job);
    trace_job_locked(trace, *job, "retire", why);
}

void ConversationStore::Impl::retire_reads_locked(ConversationStoreReadStatus status, const std::string& detail,
                                                  StoreReadTraceBatch& trace, const char* why) {
    for (const auto& job : jobs) {
        job->speculation_revoked = false;
        retire_job_locked(job, status, detail, trace, why);
    }
    cv_ready.notify_all();
}

void ConversationStore::Impl::revoke_speculation_locked(StoreReadTraceBatch& trace, bool because_of_demand) {
    if (because_of_demand) {
        bool waiting_demand = false;
        for (const auto& job : jobs)
            if (job->demand && !job->handed && !job->cancelled) waiting_demand = true;
        if (!waiting_demand) return;
    }
    for (const auto& job : jobs) {
        if (job->demand || job->handed || job->released || job->cancelled) continue;
        // Revoking speculative I/O does not cancel the request that may later need this prefix.
        job->speculation_revoked = true;
        retire_job_locked(job, ConversationStoreReadStatus::cancelled, "revoked in favour of a required operation",
                          trace, because_of_demand ? "demand_revoke" : "publication_revoke");
    }
    cv_work.notify_all();
}

void ConversationStore::Impl::expire_reads_locked(StoreReadTraceBatch& trace) {
    uint64_t now = 0;
    for (const auto& job : jobs) {
        if (!job->query.deadline_ns || job->released || job->handed ||
            (job->cancelled && !job->speculation_revoked)) continue;
        if (!now) now = store_timing_ns();
        if (now < job->query.deadline_ns) continue;
        job->speculation_revoked = false;
        retire_job_locked(job, ConversationStoreReadStatus::cancelled, "the request deadline expired", trace, "deadline");
    }
}

// Reuse the descriptor and its original authorized query/deadline. An active reader reaches this only after its
// allocation frame has unwound, so replacing its image never overlaps a second materialization or refunds live bytes.
bool ConversationStore::Impl::refresh_read_locked(const std::shared_ptr<ReadJob>& job, StoreReadTraceBatch& trace) {
    if (shutdown != ConversationStoreShutdownState::none) return false;
    if (job->active || job->cancelled || job->released || job->handed || job->epoch_at_admission != epoch) return false;
    if (job->status != ConversationStoreReadStatus::none && job->status != ConversationStoreReadStatus::ready &&
        job->status != ConversationStoreReadStatus::missing && job->status != ConversationStoreReadStatus::refused_budget)
        return false; // integrity/resource failures are terminal, not speculative refresh candidates
    std::vector<ConversationStoreMatch> found;
    try {
        collect_matches(job->query.selector, job->query.tokens.data(), job->query.tokens.size(), job->query.images,
                        found, &trace, &job->query, job->demand);
    } catch (const std::bad_alloc&) {
        return false; // keep useful prepared work when optional revalidation cannot allocate
    }
    if (found.empty() || uint64_t(found.front().tokens) < job->tokens ||
        (uint64_t(found.front().tokens) == job->tokens &&
         (found.front().published_order <= job->published_order || job->image))) return false;
    const ConversationStoreMatch& best = found.front();
    // A ready shorter cut is still useful if the longer image cannot fit. Check the replacement's own peak and
    // preparation before discarding anything; its decode replaces this descriptor's old decode, not its query.
    const uint64_t decode_workspace =
        conversation_store_detail::manifest_read_workspace_bound(manifest_read_cap(best.manifest_bytes, limits));
    const uint64_t prepared = preparation_locked();
    if (sat_add(prepared - std::min(prepared, job->prepared_bytes), decode_workspace) > options.metadata_bytes)
        return false;
    const uint64_t peak = uint64_t(best.payload_bytes) + uint64_t(best.payload_bytes) / 4 + decode_workspace +
                          job->prep_bytes + kReadSlabBytes + kReadFixedOverhead +
                          uint64_t(best.stages) * kDraftCertificationBytes;
    if ((ram_budget != 0 && peak > ram_budget) || (ram_budget == 0 && job->image)) return false;
    // Secure the replacement's queue position before releasing the old image or its leases.
    // Reordering an existing descriptor needs no allocation; a failed new slot leaves it ready.
    const auto queued = std::find(queue.begin(), queue.end(), job);
    if (queued != queue.end()) {
        std::rotate(queue.begin(), queued, std::next(queued));
    } else {
        try {
            queue.push_front(job);
        } catch (const std::bad_alloc&) {
            return false;
        }
    }
    for (auto it = ready.begin(); it != ready.end(); ++it)
        if (*it == job) {
            ready.erase(it);
            break;
        }
    // Acquire the new immutable authority before dropping the old one, including an equal-cut covered successor.
    head_leases[best.head_id] += 1;
    job->image.reset();
    release_reservation_locked(job);
    release_lease_locked(job);
    release_slot_locked(job);
    release_head_lease_locked(job);
    release_job_preparation_locked(job);
    job->status = ConversationStoreReadStatus::none;
    job->detail.clear();
    job->budget_deferred = false;
    job->prepared_bytes = decode_workspace;
    read_prep = sat_add(read_prep, decode_workspace);
    job->head_id = best.head_id;
    job->tokens = uint64_t(best.tokens);
    job->published_order = best.published_order;
    head_manifest_locked(best.head_id, job->manifest_id, job->manifest_bytes);
    // The reissue's new selection renews the admission lease: a demand that revalidated to a newer head keeps that
    // head protected for its queued life, exactly like a fresh admission.
    job->leased = true;
    job->live = best.live;
    job->stages = best.stages;
    job->payload_bytes = best.payload_bytes;
    job->peak_bytes = peak;
    counters.matches++;
    trace_begin_locked(*job, job->trace_attempt);
    trace_job_locked(trace, *job, "reissue", "catalog_refresh");
    cv_work.notify_all();
    return true;
}

// Only the caller's allowance-publication seam retries accepted budget deferrals. The reader's idle poll and
// catalog sweeps never do: unchanged low capacity cannot manufacture attempts or repeatedly perform I/O.
void ConversationStore::Impl::retry_budget_deferred_locked(StoreReadTraceBatch& trace) {
    expire_reads_locked(trace);
    if (!on || faulted || shutdown != ConversationStoreShutdownState::none ||
        admission_suspended || !ram_budget || incoming >= options.max_reads) return;
    for (const auto& job : jobs)
        if (job->demand && !job->cancelled && !job->released && !job->handed) return;
    uint64_t slots = options.max_reads - incoming;
    uint64_t available = ram_budget > resident_locked() ? ram_budget - resident_locked() : 0;
    for (const auto& job : jobs) {
        if (!slots || !available) break;
        if (!job->budget_deferred || job->status != ConversationStoreReadStatus::refused_budget ||
            job->active || job->demand || job->handed || job->released ||
            (job->cancelled && !job->speculation_revoked) || job->epoch_at_admission != epoch ||
            !job->head_id || !job->leased || !digest_nonzero(job->manifest_id) || !job->manifest_bytes) continue;
        const uint64_t workspace = conversation_store_detail::manifest_read_workspace_bound(
            manifest_read_cap(job->manifest_bytes, limits));
        const uint64_t prepared = preparation_locked();
        if (sat_add(prepared - std::min(prepared, job->prepared_bytes), workspace) > options.metadata_bytes) continue;
        const uint64_t needed = job->peak_bytes; // already includes the eventual manifest decode workspace
        if (available <= job->deferred_available || needed > available) continue;
        // Keep the captured immutable authority, even if a covered publication has retired its catalog head.
        // Enqueue first so an allocation failure leaves the refusal, leases and accounting intact.
        try {
            queue.push_back(job);
        } catch (const std::bad_alloc&) {
            continue;
        }
        for (auto it = ready.begin(); it != ready.end(); ++it)
            if (*it == job) {
                ready.erase(it);
                break;
            }
        release_job_preparation_locked(job);
        job->prepared_bytes = workspace;
        read_prep = sat_add(read_prep, workspace);
        job->cancelled = false;
        job->speculation_revoked = false;
        job->budget_deferred = false;
        job->status = ConversationStoreReadStatus::none;
        job->detail.clear();
        trace_begin_locked(*job, job->trace_attempt);
        trace_job_locked(trace, *job, "reissue", "allowance_recovery");
        available -= needed;
        --slots;
    }
}

bool ConversationStore::Impl::admit_read_locked(const ConversationStoreQuery& query, bool demand, std::string& why,
                                                StoreReadTraceBatch& trace, ConversationStoreReadRefusal* refusal) {
    uint64_t previous = 0;
    const char* reissue_reason = "new";
    std::shared_ptr<ReadJob> previous_job;
    expire_reads_locked(trace);
    // Dedupe by (request_id, generation); demand revalidates speculation without replacing its request ownership.
    for (auto it = jobs.begin(); it != jobs.end(); ++it) {
        const std::shared_ptr<ReadJob> job = *it;
        if (job->released) continue; // an old worker may still own a retired descriptor with this identity
        if (job->query.request_id != query.request_id || job->query.generation != query.generation) continue;
        const bool retryable = job->speculation_revoked || job->budget_deferred;
        if (demand && retryable && !job->handed && !job->released) {
            // Keep active retired workers inside the descriptor/byte/slot bounds until they settle.
            // Retirement removes any queued result before the fresh query can reuse this identity.
            if (read_timing) {
                previous = job->trace_attempt;
                reissue_reason = job->speculation_revoked ? "revoked_demand" : "refused_budget";
            }
            previous_job = job; // keep the bounded descriptor, query and selected immutable authority
            break;
        }
        if (job->cancelled) {
            why = "that request was cancelled";
            trace_job_locked(trace, *job, "admission_refused", "cancelled", ConversationStoreReadStatus::cancelled);
            return false;
        }
        if (job->handed || job->released) {
            why = "that request has already been handed out";
            trace_job_locked(trace, *job, "admission_refused", "already_handed");
            return false;
        }
        if (demand && !job->demand) {
            const char* adoption = job->active ? "active" :
                (job->status == ConversationStoreReadStatus::ready ? "ready_hit" : "queued");
            const uint64_t old_attempt = job->trace_attempt;
            job->demand = true;
            if (job->active) {
                job->refresh_on_completion = true;
            } else if (!refresh_read_locked(job, trace)) {
                for (auto it = queue.begin(); it != queue.end(); ++it)
                    if (*it == job) {
                        std::rotate(queue.begin(), it, std::next(it));
                        break;
                    }
            }
            trace_job_locked(trace, *job, "demand", old_attempt == job->trace_attempt ? adoption : "catalog_refresh");
            revoke_speculation_locked(trace);
        }
        return true;
    }
    // Reuse a settled revoked descriptor even at max_read_requests. Its query and selected-head hold survived the
    // RAM revocation, but its future decode must still fit beside all other preparation.
    if (previous_job && !previous_job->active) {
        if (!previous_job->head_id) {
            // A revoked miss has no manifest to price. Re-select using its original authorized query before
            // manifest_read_cap(0) can turn a bounded miss into a maximum-sized decode reservation.
            previous_job->cancelled = false;
            previous_job->speculation_revoked = false;
            previous_job->demand = true;
            if (refresh_read_locked(previous_job, trace)) {
                revoke_speculation_locked(trace);
                return true;
            }
            std::vector<ConversationStoreMatch> found;
            collect_matches(previous_job->query.selector, previous_job->query.tokens.data(),
                            previous_job->query.tokens.size(), previous_job->query.images, found,
                            &trace, &previous_job->query, true);
            if (!found.empty()) {
                // A real selection exists but did not fit the unchanged preparation/allowance checks.
                previous_job->cancelled = true;
                previous_job->speculation_revoked = true;
                previous_job->demand = false;
                why = "the newly selected head cannot fit the current read allowance or preparation ceiling";
                return false;
            }
            previous_job->status = ConversationStoreReadStatus::missing;
            previous_job->detail = "no eligible stored prefix";
            ready.push_back(previous_job);
            trace_begin_locked(*previous_job, previous);
            counters.misses++;
            trace_job_locked(trace, *previous_job, "reissue", "revoked_miss");
            trace_job_locked(trace, *previous_job, "settled", "no_io");
            cv_ready.notify_all();
            revoke_speculation_locked(trace);
            return true;
        }
        const uint64_t workspace = conversation_store_detail::manifest_read_workspace_bound(
            manifest_read_cap(previous_job->manifest_bytes, limits));
        // This settled selection owns no active reservation or ready image. Zero allowance cannot start it, and
        // the sole serving owner cannot republish capacity while waiting on an accepted demand. Refuse before
        // removing its terminal result or requeueing; its descriptor and selected-head hold stay owned until release.
        // Active workers and ready images never enter this branch; positive-budget suspension still queues normally.
        if (sat_add(preparation_locked() - std::min(read_prep, previous_job->prepared_bytes), workspace) >
            options.metadata_bytes || previous_job->peak_bytes > ram_budget) {
            why = "the retained selection cannot fit the current read allowance or preparation ceiling";
            return false;
        }
        queue.push_front(previous_job); // allocate before changing ownership
        for (auto it = ready.begin(); it != ready.end(); ++it)
            if (*it == previous_job) {
                ready.erase(it);
                break;
            }
        previous_job->cancelled = false;
        previous_job->speculation_revoked = false;
        previous_job->budget_deferred = false;
        previous_job->demand = true;
        previous_job->status = ConversationStoreReadStatus::none;
        previous_job->detail.clear();
        release_job_preparation_locked(previous_job);
        previous_job->prepared_bytes = workspace;
        read_prep = sat_add(read_prep, workspace);
        if (!previous_job->leased && previous_job->head_id) {
            head_leases[previous_job->head_id] += 1;
            previous_job->leased = true;
        }
        trace_begin_locked(*previous_job, previous);
        if (!refresh_read_locked(previous_job, trace))
            trace_job_locked(trace, *previous_job, "reissue", reissue_reason);
        revoke_speculation_locked(trace);
        cv_work.notify_all();
        return true;
    }
    // Everything is bounded *before* anything is allocated, including a miss and including the caller's own tokens.
    // The request's own prefix chain is charged here too: every image key it carries can become one chain state, and
    // a state is its EVP context as well as its bytes (see kChainBytesPerImageKey).
    const uint64_t own_prep =
        uint64_t(query.tokens.size()) * sizeof(int32_t) +
        uint64_t(query.images.size()) * sizeof(ConversationImageKey) +
        conversation_store_detail::prefix_chain_workspace(uint64_t(query.images.size())) + kJobFixedOverhead;
    if (jobs.size() >= options.max_read_requests) {
        counters.refused_budget++;
        why = "the store is already holding the maximum number of read requests";
        trace_refusal_locked(trace, query, demand, ConversationStoreReadStatus::refused_budget, "descriptor_limit", previous);
        return false;
    }
    if (query.tokens.size() > size_t(limits.max_tokens) || query.images.size() > size_t(limits.max_images) ||
        sat_add(preparation_locked(), own_prep) > options.metadata_bytes) {
        counters.refused_limits++;
        why = "the store's preparation ceiling cannot hold another read descriptor";
        trace_refusal_locked(trace, query, demand, ConversationStoreReadStatus::refused_limits, "descriptor_preparation",
                             previous);
        return false;
    }
    std::vector<ConversationStoreMatch> found;
    collect_matches(query.selector, query.tokens.data(), query.tokens.size(), query.images, found,
                    &trace, &query, demand);
    if (previous_job && previous_job->head_id &&
        (found.empty() || uint64_t(found.front().tokens) < previous_job->tokens ||
         (uint64_t(found.front().tokens) == previous_job->tokens &&
          found.front().published_order <= previous_job->published_order))) {
        ConversationStoreMatch held;
        held.head_id = previous_job->head_id;
        held.tokens = int64_t(previous_job->tokens);
        held.live = previous_job->live;
        held.stages = previous_job->stages;
        held.payload_bytes = previous_job->payload_bytes;
        held.manifest_bytes = previous_job->manifest_bytes;
        held.published_order = previous_job->published_order;
        found.insert(found.begin(), held);
    }
    if (found.empty()) {
        auto job = std::make_shared<ReadJob>();
        job->query = query;
        job->demand = demand;
        job->prep_bytes = own_prep;
        job->epoch_at_admission = epoch;
        prep += own_prep;
        job->status = ConversationStoreReadStatus::missing;
        job->detail = "no eligible stored prefix";
        jobs.push_back(job);
        ready.push_back(job);
        counters.misses++;
        trace_begin_locked(*job, previous);
        if (read_timing) {
            job->trace_enqueue = 0; // a catalog miss never joined the materialization queue
            job->trace_ready = store_timing_ns();
        }
        trace_job_locked(trace, *job, "admission", "missing");
        if (previous) trace_job_locked(trace, *job, "reissue", reissue_reason);
        trace_job_locked(trace, *job, "settled", "no_io");
        cv_ready.notify_all();
        return true;
    }
    const ConversationStoreMatch best = found.front();
    // The whole peak this read will allocate, including the manifest it has to read and decode - its raw bytes, its
    // decoded metadata and the tables the consistency pass builds (see manifest_read_workspace_bound) - and its final
    // image's scaffolding.  A demand that can never fit the published allowance is refused here rather than given a
    // reservation it can never use, and the reservation this one does get covers the *widened* decode as well: whether
    // a record this run did not write may be served is decided after the decode, so the decode has to be admitted
    // before it runs, never charged after the fact.
    const uint64_t decode_workspace =
        conversation_store_detail::manifest_read_workspace_bound(manifest_read_cap(best.manifest_bytes, limits));
    if (sat_add(sat_add(preparation_locked(), own_prep), decode_workspace) > options.metadata_bytes) {
        counters.refused_limits++;
        why = "the store's preparation ceiling cannot hold this read and the manifest decode it needs";
        trace_refusal_locked(trace, query, demand, ConversationStoreReadStatus::refused_limits, "manifest_preparation",
                             previous, best.head_id, uint64_t(best.tokens));
        return false;
    }
    const uint64_t peak = uint64_t(best.payload_bytes) + uint64_t(best.payload_bytes) / 4 + decode_workspace +
                          own_prep + kReadSlabBytes + kReadFixedOverhead +
                          uint64_t(best.stages) * kDraftCertificationBytes;
    if ((demand && ram_budget == 0) || (ram_budget != 0 && peak > ram_budget)) {
        counters.refused_budget++;
        why = "the read's peak does not fit the published RAM allowance";
        // read_body also holds restore scaffolding beside its decoded manifest.
        // The existing manifest workspace bound covers that scaffolding too;
        // quote it separately, without reserving a second decode in the reader.
        // An accepted revoked worker still owns its reservation; it is not a
        // fresh demand and cannot authorize eviction of another RAM history.
        if (refusal && !previous_job)
            refusal->additional_ram_bytes = sat_add(sat_add(peak, own_prep), decode_workspace);
        trace_refusal_locked(trace, query, demand, ConversationStoreReadStatus::refused_budget, "peak_budget",
                             previous, best.head_id, uint64_t(best.tokens), peak);
        return false;
    }
    auto job = std::make_shared<ReadJob>();
    job->query = query;
    job->demand = demand;
    job->prep_bytes = own_prep;
    job->epoch_at_admission = epoch;
    prep += own_prep;
    // The decode this read will do is preparation the store holds beside every other read, the writer and the
    // diagnostic decodes, so it is tracked from here on: it is already inside the peak this read reserved as RAM (a
    // view of those bytes, never a second charge), and it is released with the descriptor or the hand-out.
    job->prepared_bytes = decode_workspace;
    read_prep = sat_add(read_prep, decode_workspace);
    job->head_id = best.head_id;
    job->tokens = uint64_t(best.tokens);
    job->published_order = best.published_order;
    if (!head_manifest_locked(best.head_id, job->manifest_id, job->manifest_bytes) && previous_job) {
        job->manifest_id = previous_job->manifest_id;
        job->manifest_bytes = previous_job->manifest_bytes;
    }
    // The selection itself is the protection (#243): a queued read holds the same head lease a started reader
    // would, so ceiling pressure cannot retire the newest cut it just re-validated. The lease follows the job's
    // whole life and is released wherever its slot is; a catalog reissue re-grants it for the new head.
    if (job->head_id) {
        head_leases[job->head_id] += 1;
        job->leased = true;
    }
    if (previous_job) previous_job->released = true;
    job->live = best.live;
    job->stages = best.stages;
    job->payload_bytes = best.payload_bytes;
    job->peak_bytes = peak;
    jobs.push_back(job);
    queue.push_back(job);
    counters.matches++;
    trace_begin_locked(*job, previous);
    if (previous_job)
        job->progress.prior_materialize_unmeasured = previous_job->progress.prior_materialize_unmeasured ||
                                                   previous_job->progress.materialize_started;
    trace_job_locked(trace, *job, "admission", "queued");
    if (previous) trace_job_locked(trace, *job, "reissue", reissue_reason);
    if (demand) revoke_speculation_locked(trace);
    cv_work.notify_all();
    return true;
}

bool ConversationStore::Impl::deliver_locked(ConversationStoreRead& out, uint64_t request_id, uint64_t generation,
                                             bool specific, StoreReadTraceBatch& trace) {
    if (shutdown != ConversationStoreShutdownState::none) return false;
    expire_reads_locked(trace);
    for (auto it = ready.begin(); it != ready.end(); ++it) {
        const std::shared_ptr<ReadJob> job = *it;   // a copy: the ready entry is erased before the last uses
        if (job->cancelled || job->released) continue;
        if (specific && (job->query.request_id != request_id || job->query.generation != generation)) continue;
        if (!on || faulted) continue;   // a stopped or faulted store hands nothing out
        if (job->status == ConversationStoreReadStatus::ready && job->epoch_at_admission != epoch)
            continue;                   // a completion of the previous epoch is never relabelled into this one
        out.status = job->status;
        out.request_id = job->query.request_id;
        out.generation = job->query.generation;
        out.head_id = job->head_id;
        out.attempt_id = job->progress.attempt_id;
        out.prefetched = job->prefetched;
        out.epoch = job->epoch_at_admission;
        out.tokens = int64_t(job->tokens);
        out.live = job->live;
        out.stages = job->stages;
        out.payload_bytes = job->payload_bytes;
        out.materialize_ms = job->status == ConversationStoreReadStatus::ready ? job->materialize_ms : 0;
        out.image = job->image;
        out.detail = job->detail;
        job->handed = true;
        if (read_timing) job->trace_ready_end = store_timing_ns();
        counters.reads_handed_out++;
        ready.erase(it);
        trace_job_locked(trace, *job, "adopt", job->status == ConversationStoreReadStatus::ready ? "ready" : "terminal");
        if (job->lease_bytes) {
            // The store drops its own alias: the caller holds the only remaining one, and the image's RAM stays
            // charged until release_read(), because an alias derived from it may outlive it.
            job->image.reset();
            return true;
        }
        // Nothing is charged for a terminal result: its descriptor is spent.
        erase_job_locked(job);
        return true;
    }
    return false;
}

void ConversationStore::Impl::read_body(const std::shared_ptr<ReadJob>& job, ReadOutcome& out,
                                        StoreReadTraceBatch& trace) {
    const uint64_t started = read_timing ? store_timing_ns() : 0;
    const ConversationStoreQuery& query = job->query;
    const uint64_t head_id = job->head_id;
    const uint64_t tokens = job->tokens;
    const uint64_t epoch_at_admission = job->epoch_at_admission;
    // This records a verdict; it never settles.  Settlement refunds this read's reservation, and the reservation is
    // what every buffer allocated below is charged under, so reader_body settles only once this frame - and every
    // local in it - is gone.
    const auto stop_with = [&](ConversationStoreReadStatus status, const std::string& detail) {
        out.status = status;
        out.detail = detail;
    };
    Digest manifest_id{};
    uint64_t manifest_bytes = 0;
#ifdef STRATA_STORE_TEST_HOOKS
    void (*progress_gate)(void*) = nullptr;
    void* progress_gate_context = nullptr;
#endif
    {
        StoreReadTimedLock guard(mutex, read_timing, job->trace_body_lock_wait);
        if (read_timing) job->trace_start = started;
        trace_job_locked(trace, *job, "read_start", "materialization");
        if (!on || shutdown != ConversationStoreShutdownState::none) {
            stop_with(ConversationStoreReadStatus::disabled,
                      reason.empty() ? std::string("the store stopped serving") : reason);
            return;
        }
        if (faulted) {
            stop_with(ConversationStoreReadStatus::disabled, fault_reason);
            return;
        }
        if (epoch != epoch_at_admission) {
            stop_with(ConversationStoreReadStatus::stale, "the execution domain advanced before this read started");
            return;
        }
        if (!job->leased || !digest_nonzero(job->manifest_id)) {
            stop_with(ConversationStoreReadStatus::stale, "the read no longer owns its selected manifest");
            return;
        }
        manifest_id = job->manifest_id;
        manifest_bytes = job->manifest_bytes;
#ifdef STRATA_STORE_TEST_HOOKS
        progress_gate = read_progress_gate;
        progress_gate_context = read_progress_gate_context;
#endif
        if (read_timing) job->trace_io_attempted = true;
    }
    const uint64_t materialize_started = store_timing_ns();
    std::vector<uint8_t> raw;
    std::string why;
    // The same cap the admission counted the read and decode workspace from, so this read can never want more bytes
    // than its own reservation already holds.
    if (!read_file(dirs.manifests, sha256_hex(manifest_id), manifest_read_cap(manifest_bytes, limits), raw, why)) {
        stop_with(ConversationStoreReadStatus::failed, why);
        return;
    }
    {
        StoreReadTimedLock guard(mutex, read_timing, job->trace_body_lock_wait);
        counters.bytes_read += raw.size();
        if (read_timing) job->trace_bytes_read += raw.size();
    }
    conversation_store_detail::ManifestMeta meta;
    // One decode, under the *retention* figures: they are the widest set that still describes what a sound record is
    // (the manifest ceiling is this run's configured one and every count is still bounded by the bytes the file really
    // holds), so this decides whether the record exists at all.  A record another profile captured is not corrupt
    // merely because this run could not have created it, and a decode that fails here is a defect in the record - the
    // workspace this decode allocated was charged before it started, by the admission above.
    if (!conversation_store_detail::manifest_decode(raw.data(), raw.size(), retained_limits, meta, why) ||
        !(meta.manifest_id == manifest_id)) {
        stop_with(ConversationStoreReadStatus::corrupt, why.empty() ? "the manifest does not match its identity" : why);
        return;
    }
    // Whether this run may *serve* the record is a separate decision, made against the configured figures with the
    // same structural and capacity predicate the decoder applies - no second decode of the same bytes.  A record whose
    // figures this run's admitted capacity does not cover (more stages, more retained checkpoints, a longer image or
    // state stream, more tokens than this profile's context) is *refused*: it stays on disk, stays indexed and stays
    // readable by the profile that wrote it, because the figures it exceeds are this run's capacity, not a defect in
    // the record.
    if (!conversation_store_detail::manifest_within_limits(meta, limits, why)) {
        stop_with(ConversationStoreReadStatus::refused_limits,
                  "the stored record exceeds this run's admitted figures: " + why);
        return;
    }
    {
        StoreReadTimedLock guard(mutex, read_timing, job->trace_body_lock_wait);
        if (job->cancelled.load()) return;
        if (epoch != epoch_at_admission) {
            stop_with(ConversationStoreReadStatus::stale,
                      "the execution domain advanced before this read started");
            return;
        }
    }
    // The store - not the caller - is responsible for the compatibility namespace and the exact prefix: the manifest
    // must belong to this request's own bucket, domain, scope, steering mode and layer carve.
    if (!(meta.bucket == conversation_store_detail::store_bucket_digest(options, query.selector)) ||
        !(meta.domain == options.domain)) {
        stop_with(ConversationStoreReadStatus::missing,
                  "the stored image belongs to a different compatibility namespace");
        return;
    }
    const conversation_store_detail::ContainerMeta& top = meta.containers[0];
    if (!(top.scope == query.selector.scope) || top.cvec != query.selector.cvec ||
        top.geometry != query.selector.geometry || top.layer_lo != query.selector.layer_lo ||
        top.layer_hi != query.selector.layer_hi) {
        stop_with(ConversationStoreReadStatus::missing,
                  "the stored image does not match the requested scope, geometry or layer carve");
        return;
    }
    conversation_store_detail::PrefixChain chain(meta.bucket, query.tokens.data(), query.tokens.size(), query.images);
    Digest digest{};
    if (!chain.digest(tokens, digest)) {
        stop_with(ConversationStoreReadStatus::missing, "the request is shorter than the stored cut");
        return;
    }
    bool advertised = false;
    for (const auto& cut : meta.cuts)
        if (cut.tokens == tokens && cut.digest == digest) advertised = true;
    if (!advertised) {
        stop_with(ConversationStoreReadStatus::missing, "the stored image no longer holds that cut");
        return;
    }
    ConversationStoreReadProgress decoded_progress;
    if (!conversation_store_detail::restore_progress_totals(meta, raw.size(), decoded_progress, why)) {
        stop_with(ConversationStoreReadStatus::corrupt, why);
        return;
    }
    // The exact figure the restore will hold, computed from the decoded metadata before any of it is allocated: a
    // read whose restored image would not fit the published allowance fails here and is never charged past it.
    const uint64_t scaffolding = uint64_t(meta.streams.size()) *
                                     (sizeof(conversation_store_detail::RestoreTarget) + 64) +
                                 uint64_t(meta.containers.size()) * sizeof(ContainerMeta) +
                                 uint64_t(meta.objects.size()) * kObjectBytesEstimate +
                                 uint64_t(meta.stages) * kDraftCertificationBytes;
    const uint64_t needed = uint64_t(meta.payload_bytes) + uint64_t(meta.payload_bytes) / 4 + job->prepared_bytes +
                            job->prep_bytes + kReadSlabBytes + kReadFixedOverhead + scaffolding;
    {
        StoreReadTimedLock guard(mutex, read_timing, job->trace_body_lock_wait);
        if (job->cancelled.load()) return;
        // The manifest is byte-verified, decoded and bound to this query. Publish its exact component totals and
        // acknowledge only that validated metadata; none of the payload targets is visible to the owner.
        job->progress.components = decoded_progress.components;
        job->progress.totals_known = true;
        // Retain the decoded peak even on refusal, so a smaller allowance
        // increase cannot reissue the same manifest read and fail again.
        job->peak_bytes = std::max(job->peak_bytes, needed);
        if (needed > job->reserved_bytes) {
            if (needed > ram_budget || resident_locked() - job->reserved_bytes > ram_budget - needed) {
                stop_with(ConversationStoreReadStatus::refused_budget,
                          "the restored image does not fit the published RAM allowance");
                return;
            }
            ram_held += needed - job->reserved_bytes;
            job->reserved_bytes = needed;
            counters.peak_bytes = std::max(counters.peak_bytes, resident_locked());
        }
        // The decoded manifest remains live beside the restore scaffolding. Its admission bound already
        // includes the raw manifest; neither can be refunded until read_body has returned.
        const uint64_t prepared = sat_add(job->prepared_bytes, scaffolding);
        if (prepared > job->prepared_bytes) {
            const uint64_t growth = prepared - job->prepared_bytes;
            if (sat_add(preparation_locked(), growth) > options.metadata_bytes) {
                stop_with(ConversationStoreReadStatus::refused_limits,
                          "the restored image does not fit the store's preparation ceiling");
                return;
            }
            read_prep = sat_add(read_prep, growth);
            job->prepared_bytes = prepared;
        }
    }
#ifdef STRATA_STORE_TEST_HOOKS
    if (progress_gate) progress_gate(progress_gate_context);
#endif
    std::vector<conversation_store_detail::RestoreTarget> targets;
    SavedConversation image;
    if (!conversation_store_detail::restore_prepare(meta, limits, image, targets, why)) {
        stop_with(ConversationStoreReadStatus::corrupt, why);
        return;
    }
    // Every object is verified against its own content address - derived from the same stream identity the writer
    // used - and copied into its final buffer in one pass over one descriptor: the bytes that are hashed are exactly
    // the bytes that are restored, and no buffer is trusted before every digest has matched.
    std::vector<uint8_t> slab(size_t(std::min<uint64_t>(kReadSlabBytes, std::max<uint64_t>(options.chunk_bytes, 4096))));
    for (size_t i = 0; i < meta.streams.size(); ++i) {
        const auto& id = meta.streams[i].id;
        Digest identity_scope{};
        uint64_t identity_ordinal = 0;
        if (!stream_identity_of(meta.containers, id, identity_scope, identity_ordinal, why)) {
            stop_with(ConversationStoreReadStatus::corrupt, why);
            return;
        }
        const size_t component = size_t(conversation_store_detail::stream_read_component(id.kind, identity_ordinal));
        for (const auto& chunk : meta.streams[i].chunks) {
            if (job->cancelled.load()) return;
            {
                StoreReadTimedLock guard(mutex, read_timing, job->trace_body_lock_wait);
                if (epoch != epoch_at_admission) {
                    stop_with(ConversationStoreReadStatus::stale,
                              "the execution domain advanced before this read completed");
                    return;
                }
            }
            ReadObject current;
            if (!open_read_object(dirs, chunk.object, chunk.length, current, why)) {
                stop_with(ConversationStoreReadStatus::corrupt, why);
                return;
            }
            Digest actual{};
            auto hash = conversation_store_detail::object_hasher(identity_scope, id.kind, identity_ordinal,
                                                                chunk.length);
            const bool ok = hash_and_copy_object(current.fd, chunk.length, hash, targets[i].write, chunk.offset,
                                                 slab, actual, why);
            if (!ok) {
                stop_with(ConversationStoreReadStatus::corrupt, why);
                return;
            }
            if (!(actual == chunk.object)) {
                stop_with(ConversationStoreReadStatus::corrupt,
                          "an object's bytes do not reproduce its content address");
                return;
            }
            {
                StoreReadTimedLock guard(mutex, read_timing, job->trace_body_lock_wait);
                counters.bytes_read += chunk.length;
                if (read_timing) job->trace_bytes_read += chunk.length;
                // hash_and_copy_object and the content-address comparison both succeeded. A wake, read attempt,
                // short read or failed digest never acknowledges even a partially copied chunk.
                job->progress.components[component].acknowledged += chunk.length;
            }
            source_bytes_read += chunk.length;
#ifdef STRATA_STORE_TEST_HOOKS
            if (progress_gate) progress_gate(progress_gate_context);
#endif
        }
    }
    conversation_store_detail::restore_finish(image);
    if (!restored_cut_matches(image, meta, query, tokens, job->live, digest, why)) {
        stop_with(ConversationStoreReadStatus::corrupt, why);
        return;
    }
    image.freeze();
    // The restored image is the one buffer that must outlive this frame: reader_body takes it from the verdict and
    // settles the job - which is what charges it as this read's lease - only after everything else here is gone.
    out.image = std::make_shared<SavedConversation>(std::move(image));
    out.payload_bytes = meta.payload_bytes;
    out.stages = meta.stages;
    const double materialize_ms = double(store_timing_ns() - materialize_started) / 1e6;
    StoreReadTimedLock guard(mutex, read_timing, job->trace_body_lock_wait);
    if (job->cancelled.load()) {
        out.image.reset();
        return;
    }
    if (out.image->bytes() > job->reserved_bytes) {
        out.image.reset();
        stop_with(ConversationStoreReadStatus::failed,
                  "the restored image is larger than the peak it was admitted under");
        return;
    }
    out.status = ConversationStoreReadStatus::ready;
    out.materialize_ms = materialize_ms;
}

void ConversationStore::Impl::reader_body() {
    StoreReadTraceBatch trace;
    std::unique_lock<std::mutex> lock(mutex);
    while (!stop.load()) {
        trace.flush(lock);
        expire_reads_locked(trace);
        revoke_speculation_locked(trace);
        trace_waiters_locked(trace);
        for (auto it = queue.begin(); it != queue.end();) {
            const std::shared_ptr<ReadJob> job = *it;   // a copy: the entry itself is about to be erased
            if (job->cancelled) {
                it = queue.erase(it);
                continue;
            }
            if (ram_budget != 0 && job->peak_bytes > ram_budget) {
                // A read whose whole peak can never fit the allowance is refused instead of waiting forever.
                it = queue.erase(it);
                settle_job_locked(job, ConversationStoreReadStatus::refused_budget,
                                  "the read's peak does not fit the published allowance", trace);
                continue;
            }
            if (!job->demand && (ram_budget == 0 || admission_suspended || incoming >= options.max_reads)) {
                // Speculative work never waits: it is refused when it cannot start now.
                it = queue.erase(it);
                settle_job_locked(job, ConversationStoreReadStatus::refused_budget,
                                  "the read allowance has no spare room for a prefetch", trace);
                continue;
            }
            ++it;
        }
        std::shared_ptr<ReadJob> chosen;
        const bool can_start = !admission_suspended && ram_budget != 0 && incoming < options.max_reads;
        if (can_start) {
            for (int pass = 0; pass < 2 && !chosen; ++pass) {
                for (auto it = queue.begin(); it != queue.end(); ++it) {
                    const std::shared_ptr<ReadJob> job = *it;   // a copy: the entry may be erased below
                    if (job->cancelled) continue;
                    if (job->demand != (pass == 0)) continue;          // asks for a demand, then a prefetch
                    // The complete resident ledger, not ram_held alone: the index, every live descriptor's own copy,
                    // the admitted reads' decode preparation and any diagnostic decode are resident too, and only
                    // this read's peak is about to be added to them.
                    if (ram_budget != 0 && sat_add(resident_locked(), job->peak_bytes) > ram_budget) {
                        if (job->demand) continue;                     // a demand waits for room
                        queue.erase(it);
                        settle_job_locked(job, ConversationStoreReadStatus::refused_budget,
                                          "the read allowance has no spare room for a prefetch", trace);
                        break;
                    }
                    chosen = job;
                    queue.erase(it);
                    break;
                }
            }
        }
        if (!chosen) {
            if (trace.count || trace.dropped) {
                trace.flush(lock);
                continue; // recheck state after unlocked logging; never sleep through a wake delivered during it
            }
            cv_work.wait_for(lock, std::chrono::milliseconds(kIdlePollMs));
            continue;
        }
        // The peak is charged here, under the same lock that guards `admission_suspended`, so a later suspension
        // always sees every byte this read can still allocate.
        chosen->active = true;
        chosen->progress.materialize_started = true;
        chosen->reserved_bytes = chosen->peak_bytes;
        ram_held += chosen->peak_bytes;
        counters.peak_bytes = std::max(counters.peak_bytes, resident_locked());
        chosen->holds_slot = true;
        ++incoming;
        if (chosen->head_id && !chosen->leased) {
            head_leases[chosen->head_id] += 1;
            chosen->leased = true;
        }
        if (read_timing) chosen->trace_admit = store_timing_ns();
        trace_job_locked(trace, *chosen, "read_admitted", "slot_reserved");
#ifdef STRATA_STORE_TEST_HOOKS
        const auto gate = read_gate;
        void* const gate_context = read_gate_context;
#endif
        lock.unlock();
        trace.emit();
        bool frame_recorded = false;
        const auto frame_unwound = [&](ConversationStoreReadStatus status, uint64_t finished) {
            if (!read_timing || frame_recorded) return;
            chosen->trace_finish = finished;
            chosen->trace_completion_lock_wait = store_timing_ns() - finished;
            trace_job_locked(trace, *chosen, "read_finish",
                             chosen->cancelled ? chosen->trace_retire_reason : "frame_unwound", status);
            frame_recorded = true;
        };
        try {
#ifdef STRATA_STORE_TEST_HOOKS
            if (gate) gate(gate_context);
#endif
            ReadOutcome outcome;
            read_body(chosen, outcome, trace);
            const uint64_t finished = read_timing ? store_timing_ns() : 0;
            lock.lock();
            frame_unwound(outcome.status != ConversationStoreReadStatus::none ? outcome.status :
                (chosen->cancelled ? ConversationStoreReadStatus::cancelled : ConversationStoreReadStatus::failed),
                finished);
            // Every buffer the body allocated is gone now, so this is the first moment its reservation may be
            // refunded: settlement is what releases it, and the verdict carries the one buffer that had to outlive
            // the frame.
            if (outcome.status != ConversationStoreReadStatus::none) {
                chosen->payload_bytes = outcome.payload_bytes;
                chosen->stages = outcome.stages;
                chosen->materialize_ms = outcome.materialize_ms;
                chosen->image = std::move(outcome.image);
                settle_job_locked(chosen, outcome.status, std::move(outcome.detail), trace);
            }
        } catch (const std::bad_alloc&) {
            const uint64_t finished = read_timing && !frame_recorded ? store_timing_ns() : 0;
            if (!lock.owns_lock()) lock.lock();
            frame_unwound(ConversationStoreReadStatus::failed, finished);
            settle_job_locked(chosen, ConversationStoreReadStatus::failed,
                              "the read could not allocate within its admitted peak", trace);
            cv_work.notify_all();
            continue;
        } catch (...) {
            const uint64_t finished = read_timing && !frame_recorded ? store_timing_ns() : 0;
            if (!lock.owns_lock()) lock.lock();
            frame_unwound(ConversationStoreReadStatus::failed, finished);
            settle_job_locked(chosen, ConversationStoreReadStatus::failed, "the read aborted unexpectedly", trace);
            cv_work.notify_all();
            continue;
        }
        if (chosen->active) {
            // No verdict settled this frame. A demand refresh may instead have settled and requeued it as `none`.
            settle_job_locked(chosen,
                              chosen->cancelled.load() ? ConversationStoreReadStatus::cancelled
                                                       : ConversationStoreReadStatus::failed,
                              chosen->cancelled.load() ? "the request was cancelled"
                                                       : "the read ended without a verdict", trace);
        }
        cv_work.notify_all();
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Publication
// ---------------------------------------------------------------------------------------------------------------

void ConversationStore::Impl::push_publish_result_locked(ConversationStorePublishResult&& result) {
    // Admission already reserved this slot. Never evict another caller's terminal verdict.
    publish_done.push_back(std::move(result));
    cv_publish.notify_all();
}

void ConversationStore::Impl::writer_body() {
    std::unique_lock<std::mutex> lock(mutex);
    for (;;) {
        if (!durable.empty()) {
            // Urgent invalidations always outrank starting a publication or optional policy work.
            const std::shared_ptr<DurableWork> work = durable.front();
            durable.pop_front();
            lock.unlock();
            std::string why;
            const CommitOutcome outcome = commit_plan(work->plan, why);
            lock.lock();
            // A refusal that changed nothing durable is an ordinary failure and does not fault the store: the
            // previously committed authority is still the authority.  Uncertainty is latched by the transaction
            // itself, and an invalidation that could not be committed faults through its caller either way.
            if (outcome != CommitOutcome::durable_applied && !why.empty()) work->detail = why;
            work->ok = outcome == CommitOutcome::durable_applied;
            work->done = true;
            cv_durable.notify_all();
            cv_work.notify_all();
            continue;
        }
        if (publishing) {
            const std::shared_ptr<PublishJob> job = publishing;
            // Once the worker has begun reading the image it may never be replaced: publish() refuses instead.
            job->started = true;
            ConversationStorePublishResult result;
            lock.unlock();
            bool escaped = false;
            std::string escape_detail;
            try {
                publish_body(job, result);
            } catch (const std::bad_alloc&) {
                result = ConversationStorePublishResult{};
                result.status = ConversationStorePublishStatus::failed;
                result.detail = "the publication could not allocate within the published allowance";
                escaped = false;
            } catch (...) {
                escaped = true;
                escape_detail = "the publication aborted unexpectedly; the authority is not trusted";
            }
            // The attempt's in-flight object leases end here, whatever it settled as: an object it named but did not
            // create becomes reclaimable again only if no retained head and no other publication still names it (and
            // a faulted store queues nothing at all, because no file may be deleted against an unreadable catalog).
            // Released before the pass below, so a file this attempt abandoned is reclaimed in the same pass.
            release_object_leases(job);
            // This helper acquires the coordinator lock itself.  When the catalog transaction could not confirm its
            // own durability, every file this attempt created may be named by either catalog and must be retained:
            // nothing is queued, nothing is deleted, nothing is credited, and the charge stays where it is.
            if (!job->retain_files) {
                queue_orphans(job->orphans);
                reclaim_pending();
            }
            lock.lock();
            if (escaped) {
                result = ConversationStorePublishResult{};
                result.status = ConversationStorePublishStatus::failed;
                result.detail = escape_detail;
                // An escape could have interrupted the write path anywhere, so this store stops serving rather than
                // guessing what reached the device.  The durable catalog is still consistent (every catalog write is
                // one rename), and the run intent stays armed.
                latch_fault_locked(escape_detail);
            }
            eviction_excluded.clear();
            if (job->ram_bytes) {
                ram_held -= std::min(ram_held, job->ram_bytes);
                ram_lease -= std::min(ram_lease, job->ram_bytes);
                job->ram_bytes = 0;
            }
            // The workspace charge is released here, after the writer's own frame - and with it every chunk, chunk
            // buffer, map and orphan record it counted for - is really gone.  What the commit retained into the
            // index has already been counted as metadata by the apply itself.
            if (job->workspace_bytes) {
                ram_held -= std::min(ram_held, job->workspace_bytes);
                job->workspace_bytes = 0;
            }
            // One place counts what a publication attempt cost: anything that did not end in a committed head.
            if (result.status != ConversationStorePublishStatus::stored &&
                result.status != ConversationStorePublishStatus::replaced &&
                result.status != ConversationStorePublishStatus::retained_branch &&
                result.status != ConversationStorePublishStatus::coalesced)
                counters.aborted_publications++;
            result.released = true;
            if (job->coalesced && result.status == ConversationStorePublishStatus::stored) {
                result.status = ConversationStorePublishStatus::coalesced;
                result.detail = "replaced by a newer publication of the same lineage";
                counters.coalesced_publishes++;
            } else if (job->cancelled && result.status == ConversationStorePublishStatus::stored) {
                result.status = ConversationStorePublishStatus::stale;
                result.detail = "the publication ticket was cancelled";
                counters.refusals_stale++;
            }
            result.request_id = job->request.request_id;
            result.generation = job->request.generation;
            if (result.epoch == 0) result.epoch = job->epoch;
            push_publish_result_locked(std::move(result));
            if (publishing == job) publishing.reset();
            cv_work.notify_all();
            continue;
        }
        if (stop.load()) break;
        if (promote_pending && shutdown == ConversationStoreShutdownState::none && !faulted) {
            // Policy persistence uses only an idle seam. The transaction consumes the hint with its catalog copy,
            // not on completion, so any later restore re-arms work that this snapshot cannot make durable.
            CommitPlan plan;
            plan.kind = CommitPlan::Kind::promote;
            lock.unlock();
            std::string why;
            const CommitOutcome outcome = commit_plan(plan, why);
            lock.lock();
            if (outcome == CommitOutcome::durable_applied) counters.promotions_applied++;
            else counters.promotions_refused++;
            continue;
        }
        if (reclaim_pending_work && shutdown == ConversationStoreShutdownState::none && !faulted) {
            // Final reader releases schedule storage work, never filesystem I/O on the releasing thread.
            // Reclamation consumes the hint under the mutex and re-arms it for truncated passes or failed unlinks.
            lock.unlock();
            try {
                reclaim_pending();
            } catch (const std::bad_alloc&) {
                // The pass leaves its hint armed and retains every file it could not prepare.
            }
            lock.lock();
            // Pace retries (including persistent unlink failures), while urgent work and shutdown wake immediately.
            cv_work.wait_for(lock, std::chrono::milliseconds(kIdlePollMs),
                             [&] { return stop.load() || !durable.empty() || publishing; });
            continue;
        }
        cv_work.wait_for(lock, std::chrono::milliseconds(kIdlePollMs));
    }
}

void ConversationStore::Impl::publish_body(const std::shared_ptr<PublishJob>& job,
                                           ConversationStorePublishResult& out) {
    const SavedConversation& image = *job->image;
    const ConversationStorePublish& request = job->request;
    out.status = ConversationStorePublishStatus::failed;
    const auto fail = [&out](ConversationStorePublishStatus status, const std::string& detail) {
        out.status = status;
        out.detail = detail;
    };
    // Every file this attempt creates is recorded here first, so an abandoned (or escaped) attempt can never lose
    // track of a charged file.  The index is returned so a write that fails *before* the file reaches its own name
    // can move that one record to the temporary it really left behind, instead of charging both names.
    const auto note_file = [&job](Where where, const std::string& name, uint64_t charge, const Digest& object,
                                  const std::string& shard) -> size_t {
        PendingFile file;
        file.where = where;
        file.name = name;
        file.charge = charge;
        file.object = object;
        file.shard = shard;
        job->orphans.push_back(std::move(file));
        return job->orphans.size() - 1;
    };

    Digest bucket{};
    uint64_t epoch_now = 0;
    {
        std::lock_guard<std::mutex> guard(mutex);
        if (!on || shutdown != ConversationStoreShutdownState::none || faulted) {
            fail(ConversationStorePublishStatus::disabled,
                 faulted ? fault_reason : (reason.empty() ? std::string("the store is not enabled") : reason));
            return;
        }
        epoch_now = job->epoch;
        if (epoch != epoch_now) {
            counters.refusals_stale++;
            fail(ConversationStorePublishStatus::stale,
                 "the execution domain advanced after this image was captured");
            return;
        }
        ConversationStoreSelector selector;
        selector.scope = image.scope;
        selector.cvec = image.cvec;
        selector.geometry = image.geometry;
        selector.layer_lo = image.layer_lo;
        selector.layer_hi = image.layer_hi;
        bucket = conversation_store_detail::store_bucket_digest(options, selector);
        trace_bucket("publish", options, selector, bucket);
        // Every head this publication reads from - its predecessor and every reuse donor - is protected from eviction
        // (including this publication's own quota evictions) until it has committed.
        if (request.supersedes) eviction_excluded.insert(request.supersedes);
        for (const auto& entry : request.reuse)
            if (entry.stream.kind != kDraftBoundaryKind && entry.head_id) eviction_excluded.insert(entry.head_id);
    }
    if (take_fault("layout")) {
        fail(ConversationStorePublishStatus::failed, "injected fault: layout");
        return;
    }
    conversation_store_detail::ImageLayout layout;
    std::string why;
    if (!conversation_store_detail::image_layout(image, bucket, limits, layout, why)) {
        fail(ConversationStorePublishStatus::refused_limits, why);
        return;
    }
    if (job->shape.certified && !ensure_certified_reader_floor(why)) {
        fail(ConversationStorePublishStatus::failed, why);
        return;
    }
    // The predecessor (for the replacement predicate) and each reuse donor are read from their own manifests, and a
    // donor is only ever the head the caller named: `supersedes` never stands in for it.
    conversation_store_detail::ManifestMeta source;
    bool have_source = false;
    if (request.supersedes != 0) {
        std::string detail;
        // The workspace this decode needs was counted into this publication's reservation before it was admitted
        // (donor_prep_locked), so the read and decode here allocate only what was already charged.
        have_source = load_manifest_of(request.supersedes, epoch_now, source, detail) == ManifestRead::loaded &&
                      source.bucket == bucket;
    }
    std::map<uint64_t, conversation_store_detail::ManifestMeta> donors;
    for (const auto& entry : request.reuse) {
        if (entry.stream.kind == kDraftBoundaryKind || !entry.head_id || donors.count(entry.head_id)) continue;
        conversation_store_detail::ManifestMeta donor;
        std::string detail;
        if (load_manifest_of(entry.head_id, epoch_now, donor, detail) == ManifestRead::loaded &&
            donor.bucket == bucket)
            donors.emplace(entry.head_id, std::move(donor));
        else
            donors.emplace(entry.head_id, conversation_store_detail::ManifestMeta{});   // recorded as unusable
    }

    // ---- the actual prospective peak, counted once the layout and the donors are decoded ----
    //
    // This is the second half of the two-stage reservation: the admission counted what the layout and the decode
    // would need, and now that both exist the chunk, map, orphan, manifest, catalog and apply peak is counted from
    // the real objects - the actual layout, the adopted counts of the decoded donors and the uncovered gaps - before
    // one payload byte is read or written.  A publication whose counted peak cannot fit settles as refused_limits;
    // the reservation is never raised above what it was admitted with unless the counted figure really grew.
    const uint64_t chunk_limit = std::max<uint64_t>(4096, options.chunk_bytes);
    std::vector<uint64_t> stream_chunks(layout.streams.size(), 0);
    uint64_t peak_chunks = 0;
    uint64_t chunk_buffer = 0;
    for (size_t index = 0; index < layout.streams.size(); ++index) {
        const uint64_t bytes = layout.streams[index].info.bytes;
        const uint32_t element = layout.streams[index].element;
        const ConversationStoreStreamId id = layout.streams[index].info.id;
        // The grid this stream chunks into, plus every chunk a decoded donor's own interval could contribute.
        uint64_t count = conversation_store_detail::stream_chunk_count(bytes, element, chunk_limit);
        for (const auto& entry : request.reuse) {
            if (id.kind == kDraftBoundaryKind || !(entry.stream == id)) continue;
            const auto donor = donors.find(entry.head_id);
            if (donor == donors.end()) continue;
            for (const auto& candidate : donor->second.streams)
                if (candidate.id == id) {
                    count = sat_add(count, conversation_store_detail::donor_chunks_in_interval(
                                               candidate, entry.source_offset, entry.bytes));
                    break;
                }
        }
        stream_chunks[index] = count;
        peak_chunks = sat_add(peak_chunks, count);
        uint64_t size = chunk_limit;
        if (element > 1) size -= size % element;
        if (bytes) chunk_buffer = std::max(chunk_buffer, std::min(size, bytes));
    }
    uint64_t retained_donors = 0;
    if (have_source)
        retained_donors = sat_add(retained_donors,
                                  conversation_store_detail::manifest_read_workspace_bound(
                                      conversation_store_detail::manifest_encoded_size(source)));
    for (const auto& donor : donors)
        retained_donors = sat_add(retained_donors,
                                  conversation_store_detail::manifest_read_workspace_bound(
                                      conversation_store_detail::manifest_encoded_size(donor.second)));
    WriterWorkspace actual;
    actual.shard_batch = dir_fsync_batch ? kShardBatchWorkspace : 0;
    actual.request_copy = sat_add(sat_mul(uint64_t(request.reuse.size()), sizeof(ConversationStoreReuse)),
                                  kJobFixedOverhead);
    actual.layout = conversation_store_detail::image_layout_workspace(job->shape);
    if (job->shape.certified) actual.layout = sat_add(actual.layout, kReaderFloorWorkspace);
    actual.donor_decoded = retained_donors;
    actual.chunks = peak_chunks;
    actual.buffer = chunk_buffer;
    const auto manifest_counts = conversation_store_detail::ManifestShape{
        job->shape.containers, job->shape.records - job->shape.containers, job->shape.kv_layers,
        job->shape.streams, peak_chunks, peak_chunks, job->shape.cuts, job->shape.certified};
    actual.manifest = conversation_store_detail::manifest_encoded_size_of(manifest_counts);
    actual.catalog_apply = job->catalog_apply_bytes;
    if (!reserve_writer_workspace(job, actual.peak(), why)) {
        fail(ConversationStorePublishStatus::refused_limits, why);
        return;
    }

    // ---- payload: adopt proven intervals, chunk everything else, never write bytes that already exist ----
    std::vector<StreamRecord> streams;
    streams.reserve(layout.streams.size());
    std::map<Digest, uint64_t> referenced;
    std::map<Digest, uint64_t> charged_objects;      // objects this attempt created, and their charge
    uint64_t new_bytes = 0, reused_bytes = 0;
    uint64_t dirty_shards[4]; // Disabled mode never initializes or reads the opt-in tracker.
    if (dir_fsync_batch) std::fill_n(dirty_shards, 4, uint64_t(0));
    size_t adopted_chunks = 0;
    bool aborted = false;
    bool floor_limited = false;
    ConversationStorePublishStatus abort_status = ConversationStorePublishStatus::failed;
    const auto abandon = [&](ConversationStorePublishStatus status, const std::string& detail) {
        aborted = true;
        abort_status = status;
        why = detail;
    };
    // Every structure that grows once per chunk is reserved to the count this attempt was charged for: nothing below
    // may reallocate past the reserved workspace, and a chunk vector that grew geometrically would do exactly that.
    // (The two per-chunk maps have no reserve: their node growth is what the reservation counted.)
    job->orphans.reserve(size_t(peak_chunks));
    job->leased_objects.reserve(size_t(peak_chunks));

    for (const auto& stream : layout.streams) {
        const ConversationStoreStreamId id = stream.info.id;
        StreamRecord record;
        record.id = id;
        record.bytes = stream.info.bytes;
        record.chunks.reserve(size_t(stream_chunks[stream.info.index]));
        // The canonical physical identity of this stream - authorized scope, interpretation and global layer ordinal
        // - computed once and used for every object of the stream, in the writer and in the reader alike.
        Digest identity_scope{};
        uint64_t identity_ordinal = 0;
        if (!stream_identity_of(layout.containers, id, identity_scope, identity_ordinal, why)) {
            abandon(ConversationStorePublishStatus::refused_limits, why);
            break;
        }
        auto emit = [&](uint64_t from, uint64_t to) {
            while (from < to) {
                const uint64_t element = std::max<uint32_t>(1, stream.element);
                uint64_t size = chunk_limit;
                if (element > 1) size -= size % element;
                size = std::min(size, to - from);
                std::vector<uint8_t> buffer(static_cast<size_t>(size));
                if (!stream.read(from, buffer.data(), size_t(size), why)) return false;
                source_bytes_read += size;
                const Digest object = conversation_store_detail::object_digest(identity_scope, id.kind,
                                                                              identity_ordinal, size, buffer.data());
                bool known = false, revived = false;
                {
                    std::lock_guard<std::mutex> guard(mutex);
                    if (job->coalesced || job->cancelled) {
                        abandon(ConversationStorePublishStatus::coalesced, "the publication ticket was cancelled");
                        return false;
                    }
                    if (epoch != epoch_now) {
                        counters.refusals_stale++;
                        abandon(ConversationStorePublishStatus::stale,
                                "the execution domain advanced while the image was written");
                        return false;
                    }
                    const bool ours = charged_objects.count(object) != 0;
                    const auto state = objects.find(object);
                    const auto lease = object_leases.find(object);
                    // Only this single writer acquires object leases, and writer_body releases all of a job's
                    // leases before starting the next job. Thus a lease here belongs to this publication, including
                    // an orphan revived by an earlier identical chunk. Reclamation treats it as a reference.
                    known = ours || state != objects.end() || lease != object_leases.end();
                    if (known) counters.dedup_chunks++;
                    if (!ours && lease == object_leases.end()) {
                        const uint64_t charge = state == objects.end() || state->second.refs == 0
                            ? pending_object_charge_locked(object) : 0;
                        if (charge != 0) {
                            // Allocate the lease before removing the pending record: allocation failure must leave
                            // the orphan's original ownership and charge available to reclamation. An indexed
                            // object with zero refs can also be pending after an earlier failed deletion.
                            lease_object_locked(job, object, charge);
                            known = revived = revive_object_locked(object);
                        } else if (state != objects.end()) {
                            // An indexed object may lose its last retained head during make_room(); pin it until apply.
                            lease_object_locked(job, object, state->second.bytes);
                        }
                    }
                }
                if (revived) {
                    if (dir_fsync_batch) {
                        dirty_shards[object[0] / 64] |= uint64_t(1) << (object[0] % 64);
                    } else if (!sync_revived_object(object, why)) {
                        abandon(ConversationStorePublishStatus::failed, why);
                        return false;
                    }
                }
                if (!known) {
                    const uint64_t charge = allocated_charge(size);
                    if (!make_room(charge, floor_limited, why)) {
                        abandon(floor_limited ? ConversationStorePublishStatus::refused_floor
                                              : ConversationStorePublishStatus::refused_quota,
                                why);
                        return false;
                    }
                    if (take_fault("objects")) {
                        abandon(ConversationStorePublishStatus::failed, "injected fault: objects");
                        return false;
                    }
                    {
                        std::lock_guard<std::mutex> guard(mutex);
                        charged += charge;
                    }
                    const std::string object_name = sha256_hex(object);
                    const size_t object_slot =
                        note_file(Where::objects, object_name, charge, object, object_name.substr(0, 2));
                    const std::string scratch_name = "ob-" + temp_name();
                    PendingFile scratch;
                    if (!write_object_file(scratch_name, object, buffer.data(), size_t(size), scratch, why)) {
                        // The object is charged once, so the attempt carries exactly one record for the file that
                        // may really exist.  The scratch is reported whenever the bytes never reached their shard
                        // (a failed write, or a rename that did not happen), and it is empty when they did: there
                        // the destination record already names the one file, whatever the shard's flush then did.
                        if (!scratch.name.empty()) {
                            scratch.charge = charge;
                            job->orphans[object_slot] = std::move(scratch);
                        }
                        abandon(ConversationStorePublishStatus::failed, why);
                        return false;
                    }
                    if (dir_fsync_batch)
                        dirty_shards[object[0] / 64] |= uint64_t(1) << (object[0] % 64);
                    charged_objects[object] = charge;
                    new_bytes += size;
                    {
                        std::lock_guard<std::mutex> guard(mutex);
                        counters.bytes_written += size;
                        counters.published_chunks++;
                    }
                }
                conversation_store_detail::ChunkRecord chunk;
                chunk.offset = from;
                chunk.length = uint32_t(size);
                chunk.object = object;
                record.chunks.push_back(chunk);
                referenced[object] = size;
                from += size;
            }
            return true;
        };

        // Reuse intervals, placed by their own offset; every byte they do not cover is a gap that is chunked.
        std::vector<const ConversationStoreReuse*> intervals;
        for (const auto& entry : request.reuse)
            if (id.kind != kDraftBoundaryKind && entry.stream == id) intervals.push_back(&entry);
        std::stable_sort(intervals.begin(), intervals.end(),
                         [](const ConversationStoreReuse* a, const ConversationStoreReuse* b) {
                             return a->offset < b->offset;
                         });
        uint64_t covered_until = 0;
        for (const ConversationStoreReuse* entry : intervals) {
            if (aborted) break;
            const auto donor = donors.find(entry->head_id);
            if (donor == donors.end()) continue;   // only the head the caller named is ever a donor
            const StreamRecord* source_stream = nullptr;
            for (const auto& candidate : donor->second.streams)
                if (candidate.id == id) {
                    source_stream = &candidate;
                    break;
                }
            bool usable = entry->bytes > 0 && source_stream != nullptr && entry->offset >= covered_until &&
                          entry->offset % chunk_limit == 0 && entry->offset + entry->bytes <= stream.info.bytes &&
                          entry->source_offset + entry->bytes <= source_stream->bytes;
            if (usable) {
                // The donor must map this stream to the same physical identity as the outgoing image, or an adopted
                // object could not verify against the manifest it is written into.
                Digest donor_scope{};
                uint64_t donor_ordinal = 0;
                std::string detail;
                usable = stream_identity_of(donor->second.containers, id, donor_scope, donor_ordinal, detail) &&
                         donor_scope == identity_scope && donor_ordinal == identity_ordinal;
            }
            if (usable) {
                // The interval must start and end exactly on the donor's own chunk boundaries, and end either on a
                // chunk boundary of the outgoing stream or at its end: an immutable chunk is never rewritten.
                const bool ends_clean = (entry->offset + entry->bytes) % chunk_limit == 0 ||
                                        entry->offset + entry->bytes == stream.info.bytes;
                size_t first = source_stream->chunks.size(), last = 0;
                for (size_t i = 0; i < source_stream->chunks.size(); ++i) {
                    const auto& chunk = source_stream->chunks[i];
                    if (chunk.offset + chunk.length <= entry->source_offset ||
                        chunk.offset >= entry->source_offset + entry->bytes)
                        continue;
                    first = std::min(first, i);
                    last = std::max(last, i);
                }
                usable = ends_clean && first <= last &&
                         source_stream->chunks[first].offset == entry->source_offset &&
                         source_stream->chunks[last].offset + source_stream->chunks[last].length ==
                             entry->source_offset + entry->bytes;
                if (usable) {
                    // Every adopted object must still be present and authorized, or the interval is not usable.
                    std::lock_guard<std::mutex> guard(mutex);
                    bool present = true;
                    for (size_t i = first; i <= last; ++i)
                        if (objects.count(source_stream->chunks[i].object) == 0) {
                            present = false;
                            break;
                        }
                    usable = present;
                    // An adopted object is named by the new manifest without being created here, so it takes the
                    // same in-flight lease a deduplicated chunk does: a later make_room() may otherwise evict the
                    // head that owns it and reclaim the file this manifest is about to authorize.
                    if (usable)
                        for (size_t i = first; i <= last; ++i) {
                            const auto state = objects.find(source_stream->chunks[i].object);
                            if (state != objects.end()) lease_object_locked(job, state->first, state->second.bytes);
                        }
                }
                if (usable) {
                    if (!emit(covered_until, entry->offset)) break;
                    for (size_t i = first; i <= last; ++i) {
                        const auto& chunk = source_stream->chunks[i];
                        conversation_store_detail::ChunkRecord adopted = chunk;
                        adopted.offset = entry->offset + (chunk.offset - entry->source_offset);
                        record.chunks.push_back(adopted);
                        referenced[chunk.object] = chunk.length;
                        reused_bytes += chunk.length;
                        ++adopted_chunks;
                    }
                    covered_until = entry->offset + entry->bytes;
                    continue;
                }
            }
        }
        if (aborted) break;
        if (!emit(covered_until, stream.info.bytes)) break;
        streams.push_back(std::move(record));
    }
    if (!aborted) {
        std::lock_guard<std::mutex> guard(mutex);
        counters.reused_chunks += adopted_chunks;
        counters.reused_bytes += reused_bytes;
    }
    if (aborted || streams.size() != layout.streams.size()) {
        fail(abort_status, why.empty() ? "the publication was abandoned" : why);
        return;
    }

    conversation_store_detail::ManifestMeta meta;
    uint64_t published_order = 0;
    {
        std::lock_guard<std::mutex> guard(mutex);
        if (!on || shutdown != ConversationStoreShutdownState::none || faulted) {
            fail(ConversationStorePublishStatus::disabled, "the store stopped serving while the image was written");
            return;
        }
        if (epoch != epoch_now) {
            counters.refusals_stale++;
            fail(ConversationStorePublishStatus::stale,
                 "the execution domain advanced while the image was written");
            return;
        }
        if (job->cancelled || job->coalesced) {
            fail(ConversationStorePublishStatus::stale, "the publication ticket was cancelled");
            return;
        }
        published_order = ++catalog.published_order_counter;
    }
    // All storage work stays outside the catalog mutex. No manifest may name an object until this drain completes.
    if (dir_fsync_batch && (dirty_shards[0] || dirty_shards[1] || dirty_shards[2] || dirty_shards[3])) {
#ifdef STRATA_STORE_TEST_HOOKS
        void (*gate)(void*) = nullptr;
        void* context = nullptr;
        {
            std::lock_guard<std::mutex> guard(mutex);
            gate = object_batch_gate;
            context = object_batch_gate_context;
        }
        if (gate) gate(context);
#endif
#ifdef STRATA_STORE_TEST_HOOKS
        bool flushed = false;
#endif
        for (unsigned byte = 0; byte < 256; ++byte) {
            if (!(dirty_shards[byte / 64] & (uint64_t(1) << (byte % 64)))) continue;
#ifdef STRATA_STORE_TEST_HOOKS
            if ((flushed && take_fault("batch-shard-partial")) || take_fault("batch-shard-open")) {
                fail(ConversationStorePublishStatus::failed, "injected fault: object shard batch open");
                return;
            }
#endif
            static constexpr char hex[] = "0123456789abcdef";
            const char name[] = {hex[byte / 16], hex[byte % 16], '\0'};
            const ShardFd shard(open_child_dir(dirs.objects, name, why));
            if (shard.fd < 0) {
                fail(ConversationStorePublishStatus::failed, why);
                return;
            }
#ifdef STRATA_STORE_TEST_HOOKS
            // Exercise stack unwinding while the transient descriptor is owned, without changing authority.
            if (take_fault("batch-shard-allocate")) throw std::bad_alloc();
            if (take_fault("batch-shard-sync")) {
                fail(ConversationStorePublishStatus::failed, "injected fault: object shard batch flush");
                return;
            }
#endif
            // Generic flush faults retain their meaning in both modes: refuse the actual directory boundary.
            if (take_fault("object-shard-sync")) {
                fail(ConversationStorePublishStatus::failed, "injected fault: an object shard flush");
                return;
            }
            if (!fsync_dir(shard.fd, "an object shard", why)) {
                fail(ConversationStorePublishStatus::failed, why);
                return;
            }
#ifdef STRATA_STORE_TEST_HOOKS
            flushed = true;
#endif
        }
        // Always flush the parent for a nonempty batch, including shards left behind by failed attempts.
#ifdef STRATA_STORE_TEST_HOOKS
        if (take_fault("batch-parent-sync")) {
            fail(ConversationStorePublishStatus::failed, "injected fault: objects parent batch flush");
            return;
        }
#endif
        if (take_fault("object-parent-sync")) {
            fail(ConversationStorePublishStatus::failed, "injected fault: the objects directory flush");
            return;
        }
        if (!fsync_dir(dirs.objects, "objects", why)) {
            fail(ConversationStorePublishStatus::failed, why);
            return;
        }
    }
    meta.bucket = bucket;
    meta.domain = options.domain;
    meta.epoch = epoch_now;
    meta.generation = request.generation;
    meta.parent = request.supersedes;
    meta.published_order = published_order;
    meta.stages = layout.stages;
    // Moved, not copied: the layout is finished with (every stream has been emitted) and a second copy of the
    // container metadata and the cut list would be an allocation the reservation did not need to make.
    meta.containers = std::move(layout.containers);
    meta.streams = std::move(streams);
    meta.payload_bytes = layout.payload_bytes;
    meta.cuts = std::move(layout.cuts);
    meta.objects.assign(referenced.begin(), referenced.end());
    const std::vector<uint8_t> encoded = conversation_store_detail::manifest_encode(meta);
    if (encoded.size() > limits.max_manifest_bytes) {
        fail(ConversationStorePublishStatus::refused_limits,
             "the encoded manifest exceeds ConversationStoreLimits::max_manifest_bytes");
        return;
    }
    meta.manifest_id = conversation_store_detail::manifest_id_of(encoded.data(), encoded.size());
    const uint64_t head_id = std::max<uint64_t>(1, digest_u64(meta.manifest_id));
    if (take_fault("manifest")) {
        fail(ConversationStorePublishStatus::failed, "injected fault: manifest");
        return;
    }
    // A durable published_order makes two manifests identical only when they describe the same publication, so an
    // existing manifest file is reused rather than rewritten; anything else is written, flushed and renamed.
    const std::string manifest_name = sha256_hex(meta.manifest_id);
    struct stat info {};
    const bool manifest_exists = ::fstatat(dirs.manifests, manifest_name.c_str(), &info, AT_SYMLINK_NOFOLLOW) == 0;
    if (!manifest_exists) {
        const uint64_t charge = allocated_charge(encoded.size());
        if (!make_room(charge, floor_limited, why)) {
            fail(floor_limited ? ConversationStorePublishStatus::refused_floor
                               : ConversationStorePublishStatus::refused_quota,
                 why);
            return;
        }
        {
            std::lock_guard<std::mutex> guard(mutex);
            charged += charge;
        }
        const size_t manifest_slot = note_file(Where::manifests, manifest_name, charge, Digest{}, std::string());
        std::string temp = temp_name();
        bool written = false;
        for (uint32_t attempt = 0; attempt < kTempNameAttempts && !written; ++attempt) {
            if (write_new_file(dirs.tmp, temp, encoded.data(), encoded.size(), why)) {
                written = true;
                break;
            }
            if (errno != EEXIST) break;
            temp = temp_name();
            why.clear();
        }
        bool renamed = false;
        if (written) {
            if (::renameat(dirs.tmp, temp.c_str(), dirs.manifests, manifest_name.c_str()) != 0) {
                why = std::string("cannot publish the manifest: ") + std::strerror(errno);
            } else {
                renamed = true;
            }
        }
        // The rename is what moves the file, so a flush the rename already outran leaves the manifest under its own
        // name: the destination record stands and the temporary it left behind is not charged a second time.
        if (renamed && !fsync_dir(dirs.manifests, "manifests", why)) {
            fail(ConversationStorePublishStatus::failed,
                 why.empty() ? "the manifest was published but its directory could not be flushed" : why);
            return;
        }
        if (!renamed) {
            // The bytes never reached the manifest's own name (a failed write, or a rename that did not happen), so
            // the one charge this attempt took follows the temporary name instead of the name it never reached.
            job->orphans[manifest_slot].where = Where::tmp;
            job->orphans[manifest_slot].name = temp;
            fail(ConversationStorePublishStatus::failed, why.empty() ? "the manifest could not be written" : why);
            return;
        }
        {
            std::lock_guard<std::mutex> guard(mutex);
            counters.bytes_written += encoded.size();
        }
    }

    // ---- the commit: plan under the lock, write the catalog without it, then apply ----
    CommitPlan plan;
    ConversationStorePublishStatus status = ConversationStorePublishStatus::stored;
    uint64_t superseded = 0;
    {
        std::lock_guard<std::mutex> guard(mutex);
        plan.kind = CommitPlan::Kind::publish;
        plan.request_id = request.request_id;
        plan.generation = request.generation;
        plan.live_tokens = image.live.ids.size();
        plan.version = catalog_version;
        copy_catalog_locked(plan);
        CatalogHead head;
        head.head_id = head_id;
        head.manifest_id = meta.manifest_id;
        head.bucket = bucket;
        head.domain = domain_index;
        head.epoch = epoch_now;
        head.generation = request.generation;
        head.parent = request.supersedes;
        head.payload_bytes = layout.payload_bytes;
        head.manifest_bytes = allocated_charge(encoded.size());   // the charged size, rounded like every other file
        head.published_order = published_order;
        head.stages = layout.stages;
        head.pinned = image.pinned();
        head.protected_seg = false;
        head.cuts = meta.cuts;   // the moved list: the layout no longer owns one

        // A covered replacement must be a proven cover of *this* predecessor, still authorized at commit time, in
        // this branch and epoch, by a newer capture, with every pinned cut of the predecessor preserved as a pin.
        const uint64_t predecessor = request.supersedes;
        bool predecessor_present = false;
        uint64_t predecessor_generation = 0;
        if (predecessor != 0) {
            for (const auto& candidate : plan.candidate.heads) {
                if (candidate.head_id != predecessor) continue;
                if (candidate.domain == domain_index && candidate.epoch == epoch_now &&
                    candidate.manifest_id == source.manifest_id) {
                    predecessor_present = true;
                    predecessor_generation = candidate.generation;
                    head.protected_seg = candidate.protected_seg;   // a covered replacement inherits its recency
                }
                break;
            }
        }
        bool covered = request.cover && predecessor_present && have_source &&
                       request.generation > predecessor_generation;
        if (request.cover && predecessor != 0 && (!predecessor_present || !have_source ||
                                                 request.generation <= predecessor_generation)) {
            // A cover naming a predecessor that is not retained here, or one captured at an older or equal generation
            // than the descendant already stored, is a stale completion: it must never overwrite or duplicate it.
            counters.refusals_stale++;
            fail(ConversationStorePublishStatus::stale,
                 "that image is not a newer cover of the head it names");
            return;
        }
        if (covered) {
            for (const auto& cut : source.cuts) {
                if (!cut.pinned) continue;
                bool kept = false;
                for (const auto& ours : meta.cuts)
                    if (ours.tokens == cut.tokens && ours.digest == cut.digest && ours.pinned) kept = true;
                if (!kept) covered = false;
            }
        }
        if (predecessor != 0 && covered) {
            superseded = predecessor;
            plan.replaced_head = predecessor;
            for (size_t i = 0; i < plan.candidate.heads.size(); ++i)
                if (plan.candidate.heads[i].head_id == predecessor) {
                    plan.candidate.heads.erase(plan.candidate.heads.begin() + std::ptrdiff_t(i));
                    break;
                }
            plan.retiring.push_back(predecessor);
            status = ConversationStorePublishStatus::replaced;
        } else if (predecessor != 0 && have_source) {
            status = ConversationStorePublishStatus::retained_branch;
        } else if (predecessor != 0) {
            // A predecessor that is gone, retired or re-captured must never be silently turned into a new branch.
            counters.refusals_stale++;
            fail(ConversationStorePublishStatus::stale,
                 "the head this image was captured from is no longer retained");
            return;
        }
        // Count a proven replacement in the candidate catalog before enforcing its head ceiling.
        // The predecessor stays durable until commit; byte/floor admission still includes its files.
        plan.candidate.heads.push_back(head);
        plan.new_head = head_id;
        plan.new_objects.clear();
        plan.new_objects.reserve(referenced.size());
        for (const auto& entry : referenced) plan.new_objects.push_back(entry.first);
        for (const auto& entry : charged_objects) plan.new_object_charges[entry.first] = entry.second;
        plan.new_manifest_bytes = allocated_charge(encoded.size());
        bool room = obtain_head_slot_locked(plan.candidate, head_id, plan.retiring);
        if (room) {
            cap_protected_locked(plan.candidate);
            // Counted, not encoded: this is a measurement, and a measurement must never be the reason bytes are
            // allocated inside an admission that has already decided what it can afford.
            uint64_t estimate = conversation_store_detail::catalog_encoded_size(plan.candidate);
            estimate += uint64_t(objects.size() + plan.new_objects.size()) * kObjectBytesEstimate;
            estimate = sat_add(estimate, sat_mul(uint64_t(pending_files.size()), kPendingMetadataBytes));
            for (const auto& entry : head_objects)
                estimate = sat_add(estimate, head_hold_metadata_bytes(entry.second.size()));
            estimate = sat_add(estimate, head_hold_metadata_bytes(plan.new_objects.size()));
            if (estimate > options.metadata_bytes) {
                plan.candidate.heads.pop_back();
                plan.retiring.clear();
                room = false;
                counters.refused_limits++;
                fail(ConversationStorePublishStatus::refused_limits,
                     "the store's index ceiling cannot hold another head");
            }
        } else {
            plan.candidate.heads.pop_back();
            counters.refused_limits++;
            fail(ConversationStorePublishStatus::refused_limits,
                 "every retained head is pinned or leased, so no head slot can be obtained");
        }
        if (!room) return;
    }
    if (take_fault("catalog")) {
        fail(ConversationStorePublishStatus::failed, "injected fault: catalog");
        return;
    }
    std::string commit_why;
    const CommitOutcome outcome = commit_plan(plan, commit_why);
    if (outcome == CommitOutcome::unchanged) {
        // The durable catalog is unchanged, so nothing was authorized.  Files this attempt wrote stay accounted
        // orphans whose bytes stay charged until they are confirmed gone, and the predecessor's authority is intact.
        // This is an ordinary failure of an optional write: it does not fault the store.
        fail(ConversationStorePublishStatus::failed,
             commit_why.empty() ? "the catalog commit was refused" : commit_why);
        return;
    }
    if (outcome == CommitOutcome::uncertain) {
        // The new catalog may be the durable one, so every file this attempt created may be named by it.  Nothing is
        // handed to the reclaimer, nothing is credited, and the store has already latched its fault: the next start
        // re-reads the authority instead of trusting this view.
        job->retain_files = true;
        fail(ConversationStorePublishStatus::failed,
             commit_why.empty() ? "the catalog commit is not confirmed" : commit_why);
        return;
    }
    job->orphans.clear();   // every file this attempt created is now referenced by the committed manifest
    {
        std::lock_guard<std::mutex> guard(mutex);
        counters.published++;
        if (status == ConversationStorePublishStatus::replaced) counters.replaced++;
        if (status == ConversationStorePublishStatus::retained_branch) counters.retained_branches++;
        counters.heads = uint64_t(catalog.heads.size());
    }
    out.status = status;
    out.head_id = head_id;
    out.epoch = epoch_now;
    out.superseded = superseded;
    out.stages = layout.stages;
    out.payload_bytes = layout.payload_bytes;
    out.new_bytes = new_bytes;
    out.reused_bytes = reused_bytes;
    out.applied_ns = plan.applied_ns;
    out.catalog_version = plan.applied_catalog_version;
    out.detail = status == ConversationStorePublishStatus::retained_branch
                     ? "the predecessor keeps a divergent or pinned cut, so both branches are retained"
                     : std::string();
}

// ---------------------------------------------------------------------------------------------------------------
// The public seams
// ---------------------------------------------------------------------------------------------------------------

ConversationStore::ConversationStore(ConversationStoreOptions options, ConversationStoreLimits limits)
    : impl_(std::make_unique<Impl>()) {
    Impl& self = *impl_;
    const char* timing = std::getenv("STRATA_STORE_TIMING");
    self.read_timing = timing && std::strcmp(timing, "1") == 0;
    const char* batch = std::getenv("STRATA_STORE_DIR_FSYNC_BATCH");
    self.dir_fsync_batch = batch && std::strcmp(batch, "1") == 0;
    self.options = std::move(options);
    self.limits = limits;
    // The retention figures are derived here, from the format's own representability bounds and the configured
    // ceilings, and are the only ones the recovery and index paths use: an owner cannot widen or narrow what may be
    // kept by accident, and a root shared by profiles with different admitted figures never loses a record to the
    // difference, nor to a change of the physical quota (see conversation_store_detail::retention_limits).
    self.retained_limits = conversation_store_detail::retention_limits(limits);
    self.root = self.options.root;
    self.ram_budget = self.options.read_budget_bytes;
    const uint32_t wanted = self.options.max_reads ? self.options.max_reads : 1;
    self.reader_threads = std::max<uint32_t>(1, std::min<uint32_t>(wanted, kMaxReaderThreads));
    if (self.options.root.empty()) {
        self.reason = "no store root was configured";
        return;
    }
    if (!digest_nonzero(self.options.domain) || !digest_nonzero(self.options.model) ||
        !digest_nonzero(self.options.engine)) {
        self.reason = "the execution-domain, model or engine identity is missing";
        return;
    }
    if (!self.limits.max_stages || !self.limits.max_images || !self.limits.max_streams ||
        !self.limits.max_manifest_bytes) {
        self.reason = "a ConversationStoreLimits count is zero";
        return;
    }
    if (self.options.chunk_bytes < 4096 || self.options.chunk_bytes > (uint64_t(1) << 30)) {
        self.reason = "the immutable chunk size is outside the supported range";
        return;
    }
    if (self.options.max_read_requests == 0 || self.options.max_pending_publishes == 0) {
        self.reason = "a read or publication bound is zero";
        return;
    }
    std::string why;
    // Construction is single-threaded: no worker exists yet, so the authority steps take no lock, and any failure
    // here disables persistence without leaving the lock or a half-armed marker behind.
    bool opened = false;
    try {
        opened = self.open_root(why) && self.read_catalog(why) && self.reconcile(why);
    } catch (const std::exception& error) {
        why = std::string("the store's open sequence failed: ") + error.what();
        opened = false;
    } catch (...) {
        why = "the store's open sequence failed";
        opened = false;
    }
    if (!opened) {
        self.reason = why.empty() ? "the store could not be opened" : why;
        if (self.lock_fd >= 0) {
            ::flock(self.lock_fd, LOCK_UN);
            ::close(self.lock_fd);
            self.lock_fd = -1;
        }
        self.lock_owned = false;
        self.dirs.close_all();
        return;
    }
    self.sample_free_locked();
    self.on = true;
    self.initialized = true;
    self.writer = std::thread([&self] { self.writer_body(); });
    self.readers.reserve(self.reader_threads);
    for (uint32_t i = 0; i < self.reader_threads; ++i)
        self.readers.emplace_back([&self] { self.reader_body(); });
}

ConversationStore::~ConversationStore() {
    // Never upgrade an unfinished request into a clean exit merely because its owner is being destroyed.
    request_shutdown(0);
    finish_shutdown();
}

bool ConversationStore::enabled() const {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    return impl_->on;
}

const char* ConversationStore::disabled_reason() const { return impl_->reason.c_str(); }

bool ConversationStore::healthy() const {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    return impl_->on && !impl_->faulted;
}

void ConversationStore::set_unhealthy(const char* reason) {
    StoreReadTraceBatch trace;
    std::lock_guard<std::mutex> guard(impl_->mutex);
    // A latched fault is permanent: the engine observed something that must stop reuse, and only a restart (which
    // re-reads the run-intent marker) may trust this domain again.
    impl_->retire_reads_locked(ConversationStoreReadStatus::failed,
                               "the store faulted before this read completed", trace, "store_fault");
    impl_->latch_fault_locked(reason ? reason : "the owner observed a failure");
    impl_->cv_work.notify_all();
}

uint64_t ConversationStore::epoch() const {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    return impl_->epoch;
}

uint64_t ConversationStore::publication_generation_floor() const {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    // Every head the catalog holds for this domain - including one waiting to be retired for an unclean start or a
    // retired epoch: a figure the catalog has already recorded is never minted again, whether or not the head it
    // belongs to is still reachable, so a cover can never name a generation this domain has already used.
    uint64_t floor = 0;
    for (const auto& head : impl_->catalog.heads)
        if (head.domain == impl_->domain_index) floor = std::max(floor, head.generation);
    return floor;
}

const std::array<uint8_t, 32>& ConversationStore::issuer_key() const { return impl_->issuer_key; }

bool ConversationStore::run_intent_active() const {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    return impl_->intent_active;
}

uint64_t ConversationStore::domain_digest_truncated() const {
    std::array<uint8_t, 96> material{};
    std::memcpy(material.data(), impl_->options.domain.data(), 32);
    std::memcpy(material.data() + 32, impl_->options.model.data(), 32);
    std::memcpy(material.data() + 64, impl_->options.engine.data(), 32);
    return digest_u64(sha256(material.data(), material.size()));
}

void ConversationStore::for_each_match(const ConversationStoreSelector& selector, const int32_t* ids, size_t count,
                                       const ConversationImageKey* images, size_t image_count,
                                       const std::function<void(const ConversationStoreMatch&)>& visit,
                                       ConversationStoreLookupFrontier* frontier) const {
    std::vector<ConversationImageKey> keys;
    if (images && image_count) keys.assign(images, images + image_count);
    std::vector<ConversationStoreMatch> found;
    {
        std::lock_guard<std::mutex> guard(impl_->mutex);
        impl_->collect_matches(selector, ids, count, keys, found, nullptr, nullptr, false);
        if (frontier) {
            frontier->observed_ns = store_timing_ns();
            frontier->catalog_version = impl_->catalog_version;
        }
    }
    for (const auto& match : found) visit(match);
}

std::vector<ConversationStoreMatch> ConversationStore::matches(const ConversationStoreSelector& selector,
                                                              const std::vector<int32_t>& ids,
                                                              const std::vector<ConversationImageKey>& images) const {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    std::vector<ConversationStoreMatch> found;
    impl_->collect_matches(selector, ids.data(), ids.size(), images, found, nullptr, nullptr, false);
    return found;
}

bool ConversationStore::retains_head(uint64_t head_id) const {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    for (const auto& head : impl_->catalog.heads)
        if (head.head_id == head_id) return true;
    return false;
}

bool ConversationStore::manifest_info(uint64_t head_id, ConversationStoreManifestInfo& out) const {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    for (const auto& head : impl_->catalog.heads) {
        if (head.head_id != head_id) continue;
        out = ConversationStoreManifestInfo{};
        out.head_id = head.head_id;
        out.epoch = head.epoch;
        out.generation = head.generation;
        out.parent_head_id = head.parent;
        out.payload_bytes = head.payload_bytes;
        out.manifest_bytes = head.manifest_bytes;
        out.published_order = head.published_order;
        out.stages = head.stages;
        out.pinned = head.pinned;
        out.cuts.clear();
        out.cuts.reserve(head.cuts.size());
        for (const auto& cut : head.cuts) out.cuts.push_back(int64_t(cut.tokens));
        return true;
    }
    return false;
}

bool ConversationStore::manifest_streams(uint64_t head_id, std::vector<ConversationStoreStreamInfo>& out) const {
    Impl& self = *impl_;
    Digest manifest_id{};
    uint64_t bytes = 0;
    {
        std::lock_guard<std::mutex> guard(self.mutex);
        if (!self.on || self.shutdown != ConversationStoreShutdownState::none ||
            !self.head_manifest_locked(head_id, manifest_id, bytes)) return false;
    }
    // This diagnostic holds no reservation of its own, so its read and decode are charged before either allocates:
    // the retention figures this decodes under are wider than this run's admitted ones, and the decoded metadata and
    // the tables the consistency pass builds are several times the file (see manifest_read_workspace_bound).
    const uint64_t capped = manifest_read_cap(bytes, self.limits);
    const uint64_t workspace = conversation_store_detail::manifest_read_workspace_bound(capped);
    std::string why;
    Impl::DecodeLease lease(self);
    if (!lease.acquire(workspace, why)) return false;
    std::vector<uint8_t> raw;
    const bool read = read_file(self.dirs.manifests, sha256_hex(manifest_id), capped, raw, why);
    conversation_store_detail::ManifestMeta meta;
    const bool decoded = read &&
                         conversation_store_detail::manifest_decode(raw.data(), raw.size(), self.retained_limits, meta,
                                                                   why);
    if (!decoded) return false;
    out = conversation_store_detail::manifest_stream_infos(meta);
    return true;
}

bool ConversationStore::demand(const ConversationStoreQuery& query, std::string& why,
                               ConversationStoreReadRefusal* refusal) {
    if (refusal) *refusal = {};
    StoreReadTraceBatch trace;
    if (impl_->read_timing) trace.call_ns = store_timing_ns();
    std::lock_guard<std::mutex> guard(impl_->mutex);
    if (impl_->read_timing) trace.call_lock_wait_ns = store_timing_ns() - trace.call_ns;
    if (impl_->shutdown != ConversationStoreShutdownState::none) {
        why = "the store is shutting down";
        impl_->trace_refusal_locked(trace, query, true, ConversationStoreReadStatus::disabled, "shutdown");
        return false;
    }
    if (!impl_->on) {
        why = impl_->reason.empty() ? "the store is not enabled" : impl_->reason;
        impl_->trace_refusal_locked(trace, query, true, ConversationStoreReadStatus::disabled, "disabled");
        return false;
    }
    if (impl_->faulted) {
        why = impl_->fault_reason;
        impl_->trace_refusal_locked(trace, query, true, ConversationStoreReadStatus::disabled, "store_fault");
        return false;
    }
    return impl_->admit_read_locked(query, true, why, trace, refusal);
}

bool ConversationStore::prefetch(const ConversationStoreQuery& query, std::string& why) {
    StoreReadTraceBatch trace;
    if (impl_->read_timing) trace.call_ns = store_timing_ns();
    std::lock_guard<std::mutex> guard(impl_->mutex);
    if (impl_->read_timing) trace.call_lock_wait_ns = store_timing_ns() - trace.call_ns;
    if (impl_->shutdown != ConversationStoreShutdownState::none) {
        why = "the store is shutting down";
        impl_->trace_refusal_locked(trace, query, false, ConversationStoreReadStatus::disabled, "shutdown");
        return false;
    }
    if (!impl_->on) {
        why = impl_->reason.empty() ? "the store is not enabled" : impl_->reason;
        impl_->trace_refusal_locked(trace, query, false, ConversationStoreReadStatus::disabled, "disabled");
        return false;
    }
    if (impl_->faulted) {
        why = impl_->fault_reason;
        impl_->trace_refusal_locked(trace, query, false, ConversationStoreReadStatus::disabled, "store_fault");
        return false;
    }
    return impl_->admit_read_locked(query, false, why, trace, nullptr);
}

void ConversationStore::cancel_publication(uint64_t request_id, uint64_t generation) {
    Impl& self = *impl_;
    std::lock_guard<std::mutex> guard(self.mutex);
    if (self.publishing && !self.publishing->started &&
        self.publishing->request.request_id == request_id && self.publishing->request.generation == generation) {
        // A ticket the writer has not begun reading is cancelled here, with its alias and charge released at once.
        const std::shared_ptr<Impl::PublishJob>& job = self.publishing;
        job->cancelled = true;
        if (job->ram_bytes) {
            self.ram_held -= std::min(self.ram_held, job->ram_bytes);
            self.ram_lease -= std::min(self.ram_lease, job->ram_bytes);
            job->ram_bytes = 0;
        }
        if (job->workspace_bytes) {
            self.ram_held -= std::min(self.ram_held, job->workspace_bytes);
            job->workspace_bytes = 0;
        }
        job->image.reset();
        ConversationStorePublishResult result;
        result.status = ConversationStorePublishStatus::stale;
        result.request_id = request_id;
        result.generation = generation;
        result.epoch = job->epoch;
        result.released = true;
        result.detail = "the publication ticket was cancelled before it began";
        self.counters.refusals_stale++;
        self.push_publish_result_locked(std::move(result));
        self.publishing.reset();
        self.cv_work.notify_all();
        return;
    }
}

void ConversationStore::cancel_read(uint64_t request_id, uint64_t generation) {
    Impl& self = *impl_;
    StoreReadTraceBatch trace;
    std::lock_guard<std::mutex> guard(self.mutex);
    for (const auto& job : self.jobs) {
        if (job->released) continue;
        if (job->query.request_id != request_id || job->query.generation != generation) continue;
        job->speculation_revoked = false; // explicit cancellation wins over a previous internal revocation
        self.retire_job_locked(job, ConversationStoreReadStatus::cancelled, "the caller cancelled this read",
                               trace, "caller_cancel");
        self.cv_work.notify_all();
        return;
    }
}

void ConversationStore::withdraw_prefetch(uint64_t request_id, uint64_t generation) {
    Impl& self = *impl_;
    StoreReadTraceBatch trace;
    std::lock_guard<std::mutex> guard(self.mutex);
    for (const auto& entry : self.jobs) {
        const std::shared_ptr<Impl::ReadJob> job = entry;
        if (job->released || job->query.request_id != request_id || job->query.generation != generation) continue;
        if (job->demand || job->handed) return; // the demand owner, not the advisory channel, now owns it
        job->speculation_revoked = false;
        self.retire_job_locked(job, ConversationStoreReadStatus::cancelled, "the advisory prefetch was withdrawn",
                               trace, "withdraw");
        job->released = true;
        if (!job->active) self.erase_job_locked(job);
        self.cv_work.notify_all();
        return;
    }
}

bool ConversationStore::refresh_speculation(uint64_t request_id, uint64_t generation) {
    Impl& self = *impl_;
    StoreReadTraceBatch trace;
    std::unique_lock<std::mutex> lock(self.mutex);
    if (self.shutdown != ConversationStoreShutdownState::none) return false;
    self.expire_reads_locked(trace);
    for (const auto& entry : self.jobs) {
        const std::shared_ptr<Impl::ReadJob> job = entry;
        if (job->released || job->cancelled || job->handed || job->demand ||
            job->query.request_id != request_id || job->query.generation != generation) continue;
        if (job->active || job->status == ConversationStoreReadStatus::none) return false;
        // The selection is rebuilt against the catalog as it stands now; the job keeps its descriptor,
        // its authorized query and its queue identity. Refusals behave like the original admission.
        if (!self.refresh_read_locked(job, trace)) return false;
        lock.unlock();
        self.cv_work.notify_all();
        return true;
    }
    return false;
}

bool ConversationStore::take_ready(ConversationStoreRead& out) {
    StoreReadTraceBatch trace;
    std::lock_guard<std::mutex> guard(impl_->mutex);
    return impl_->deliver_locked(out, 0, 0, false, trace);
}

bool ConversationStore::take_ready(uint64_t request_id, uint64_t generation, ConversationStoreRead& out) {
    StoreReadTraceBatch trace;
    std::lock_guard<std::mutex> guard(impl_->mutex);
    return impl_->deliver_locked(out, request_id, generation, true, trace);
}

bool ConversationStore::wait_ready(ConversationStoreRead& out, uint64_t timeout_ms) {
    StoreReadTraceBatch trace;
    std::unique_lock<std::mutex> lock(impl_->mutex);
    if (impl_->deliver_locked(out, 0, 0, false, trace)) return true;
    if (timeout_ms == 0) return false;
    impl_->cv_ready.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                             [&] { return !impl_->ready.empty() ||
                                          impl_->shutdown != ConversationStoreShutdownState::none; });
    return impl_->deliver_locked(out, 0, 0, false, trace);
}

bool ConversationStore::wait_ready(uint64_t request_id, uint64_t generation, ConversationStoreRead& out,
                                   uint64_t timeout_ms) {
    Impl& self = *impl_;
    StoreReadTraceBatch trace;
    std::unique_lock<std::mutex> lock(self.mutex);
    const auto settled = [&]() {
        if (self.shutdown != ConversationStoreShutdownState::none) return true;
        for (const auto& job : self.jobs)
            if (!job->released && job->query.request_id == request_id && job->query.generation == generation)
                return job->handed ||
                       (job->status != ConversationStoreReadStatus::none && !job->active);
        return true;   // an unknown request has settled (as nothing)
    };
    if (!settled() && timeout_ms)
        self.cv_ready.wait_for(lock, std::chrono::milliseconds(timeout_ms), settled);
    if (!settled()) return false;
    return self.deliver_locked(out, request_id, generation, true, trace);
}

void ConversationStore::release_read(uint64_t request_id, uint64_t generation) {
    Impl& self = *impl_;
    StoreReadTraceBatch trace;
    std::lock_guard<std::mutex> guard(self.mutex);
    for (const auto& job : self.jobs) {
        if (job->released) continue;
        if (job->query.request_id != request_id || job->query.generation != generation) continue;
        if (!job->released) {
            job->released = true;
            self.counters.reads_released++;
        }
        job->cancelled = true;                       // a released request is never delivered
        if (self.read_timing) {
            job->trace_retire_reason = "release";
            if (job->trace_ready && !job->trace_ready_end) job->trace_ready_end = store_timing_ns();
        }
        for (auto q = self.queue.begin(); q != self.queue.end(); ++q)
            if (*q == job) {
                self.queue.erase(q);
                break;
            }
        for (auto r = self.ready.begin(); r != self.ready.end(); ++r)
            if (*r == job) {
                self.ready.erase(r);
                break;
            }
        if (!job->active) {
            // The reader owns the charge until its body returns; a settled job gives everything back here.
            self.release_reservation_locked(job);
            self.release_lease_locked(job);
            self.release_slot_locked(job);
            self.release_head_lease_locked(job);
            job->image.reset();
            self.trace_job_locked(trace, *job, "release", "caller_release");
            self.erase_job_locked(job);
        }
        else self.trace_job_locked(trace, *job, "release", "active_release");
        self.cv_work.notify_all();
        return;
    }
}

ConversationStoreReadStatus ConversationStore::request_status(uint64_t request_id, uint64_t generation) const {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    for (const auto& job : impl_->jobs)
        if (!job->released && job->query.request_id == request_id && job->query.generation == generation)
            return job->status;
    return ConversationStoreReadStatus::none;
}

ConversationStoreReadProgress ConversationStore::request_progress(uint64_t request_id, uint64_t generation) const {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    const Impl::ReadJob* found = nullptr;
    for (const auto& job : impl_->jobs) {
        if (job->query.request_id != request_id || job->query.generation != generation) continue;
        found = job.get();
        if (!job->released) break; // a new owner can reuse an identity while its old released worker unwinds
    }
    if (!found) return {};
    auto out = found->progress;
    out.status = found->status;
    out.head_id = found->head_id;
    out.state = found->released ? ConversationStoreReadState::released :
        found->handed ? ConversationStoreReadState::handed_out :
        found->active ? ConversationStoreReadState::materializing :
        found->status == ConversationStoreReadStatus::none ? ConversationStoreReadState::queued :
        ConversationStoreReadState::settled;
    return out;
}

void ConversationStore::confirmed_restore(uint64_t head_id) {
    Impl& self = *impl_;
    std::lock_guard<std::mutex> guard(self.mutex);
    if (!self.on || self.faulted || !self.initialized ||
        self.shutdown != ConversationStoreShutdownState::none || self.stop.load()) return;
    bool found = false;
    for (auto& head : self.catalog.heads) {
        if (head.head_id != head_id) continue;
        found = true;
        self.touch_recency_locked(head_id);
        if (!head.protected_seg) {
            head.protected_seg = true;
            self.cap_protected_locked(self.catalog);
        }
        break;
    }
    if (!found) return;
    // No per-restore allocation or durable queue entry: every hint before a candidate copy shares this one bit.
    if (self.promote_pending) self.counters.promotions_coalesced++;
    self.promote_pending = true;
    self.cv_work.notify_all();
}

ConversationStorePublishAdmission ConversationStore::publish(std::shared_ptr<const SavedConversation> image,
                                                             const ConversationStorePublish& request,
                                                             std::string& why) {
    Impl& self = *impl_;
    StoreReadTraceBatch trace;
    std::lock_guard<std::mutex> guard(self.mutex);
    if (self.shutdown != ConversationStoreShutdownState::none) {
        why = "the store is shutting down";
        return ConversationStorePublishAdmission::refused;
    }
    self.expire_reads_locked(trace);
    if (!self.on) {
        why = self.reason.empty() ? "the store is not enabled" : self.reason;
        return ConversationStorePublishAdmission::refused;
    }
    if (self.faulted) {
        why = self.fault_reason;
        return ConversationStorePublishAdmission::refused;
    }
    if (!image) {
        why = "no image was supplied";
        return ConversationStorePublishAdmission::refused;
    }
    if (request.epoch != 0 && request.epoch != self.epoch) {
        self.counters.refusals_stale++;
        why = "the image was captured in a previous execution-domain epoch";
        return ConversationStorePublishAdmission::refused;
    }
    const uint64_t image_bytes = image->bytes();
    // The image is not the workspace.  Count what this publication will really allocate - the copied request, the
    // layout skeleton, its per-record prefix chains and the parent/donor manifest decode - before one byte of it is
    // allocated, and count the image itself separately: a shape whose own counted peak cannot fit is refused as such,
    // never deferred, because no revoked speculation and no settled lease can make room for it.
    conversation_store_detail::ImageShape shape;
    if (!conversation_store_detail::image_shape(*image, self.limits, shape, why)) {
        self.counters.refused_limits++;
        why = "the publication image is not representable: " + why;
        return ConversationStorePublishAdmission::refused;
    }
    const WriterWorkspace workspace = self.writer_workspace_locked(request, shape);
    const uint64_t needed = workspace.peak();
    const uint64_t required = sat_add(image_bytes, needed);
    // A transferred lease: the caller's own cache still holds exactly this entry's bytes outside the store's
    // allowance, so this one publication is fitted against the allowance *plus* that credit.  The credit is clamped
    // to the image's own size, is granted to no read, and widens nothing else: the store still charges the full
    // image and the whole counted workspace into ram_held_bytes().
    const uint64_t credit = std::min(request.external_lease_bytes, image_bytes);
    const uint64_t envelope = sat_add(self.ram_budget, credit);
    if (!self.ram_budget || sat_add(self.metadata, needed) > self.options.metadata_bytes ||
        sat_add(self.metadata, required) > envelope) {
        // The persistent index cannot be refunded by settling another operation. Include it in the
        // isolated publication floor; only transient reservations belong to the deferral path.
        self.counters.refused_limits++;
        why = credit
                  ? "the publication image and its writer workspace do not fit the published RAM allowance plus the "
                    "lease the caller transferred"
                  : "the publication image and its writer workspace do not fit the published RAM allowance";
        return ConversationStorePublishAdmission::refused;
    }
    // Every accepted request needs its own terminal result, even when the pending writer is coalesced, cancelled or
    // closed before starting. Count that writer's reserved slot as well as completed, unreaped results before taking
    // another image. Harvesting is the only way to make room; revoking speculative reads cannot recover these slots.
    const uint64_t result_bound = uint64_t(self.options.max_pending_publishes) * 4 + 4;
    if (uint64_t(self.publish_done.size()) + uint64_t(bool(self.publishing)) >= result_bound) {
        self.counters.deferred_publishes++;
        why = "the publication result queue is full; harvest completed publications before retrying";
        return ConversationStorePublishAdmission::deferred;
    }
    // What is left is *occupancy*: another operation's live preparation, or a read's reservation and lease, may be
    // holding the ceiling or the allowance at this instant.  Speculative reads are revocable and an outgoing capture
    // is required work, so they give way first - never an active worker, a demand or a handed-out image, whose charges
    // are released only when they settle (see revoke_speculation_locked) - and the admission is then retried under
    // this same lock.  A publication that still cannot fit is deferred: nothing was allocated, charged or transferred,
    // so the caller keeps its image and its lease and may offer it again at a later seam.
    if (self.preparation_locked() > self.options.metadata_bytes - needed ||
        self.resident_locked() > envelope - required) {
        self.revoke_speculation_locked(trace, false);
        if (self.preparation_locked() > self.options.metadata_bytes - needed ||
            self.resident_locked() > envelope - required) {
            self.counters.deferred_publishes++;
            why = "the store's RAM allowance or preparation ceiling is occupied by work it cannot free right now";
            return ConversationStorePublishAdmission::deferred;
        }
    }
    if (self.publishing) {
        const std::shared_ptr<Impl::PublishJob>& current = self.publishing;
        // Only a *proven cover of the same nonzero predecessor* by a newer generation may replace the job in flight:
        // two unrelated new branches (supersedes == 0) are never coalesced.
        const bool same_lineage = request.supersedes != 0 && request.supersedes == current->request.supersedes &&
                                  request.cover && current->request.cover &&
                                  current->epoch == (request.epoch != 0 ? request.epoch : self.epoch);
        if (current->started || !same_lineage || request.generation <= current->request.generation) {
            self.counters.refusals_busy++;
            why = current->started ? "the writer is already serializing another image"
                                   : "another branch is already admitted";
            return ConversationStorePublishAdmission::refused;
        }
        ConversationStorePublishResult coalesced;
        coalesced.status = ConversationStorePublishStatus::coalesced;
        coalesced.request_id = current->request.request_id;
        coalesced.generation = current->request.generation;
        coalesced.epoch = current->epoch;
        coalesced.detail = "replaced by a newer publication of the same lineage";
        coalesced.released = true;
        if (current->ram_bytes) {
            self.ram_held -= std::min(self.ram_held, current->ram_bytes);
            self.ram_lease -= std::min(self.ram_lease, current->ram_bytes);
            current->ram_bytes = 0;
        }
        if (current->workspace_bytes) {
            self.ram_held -= std::min(self.ram_held, current->workspace_bytes);
            current->workspace_bytes = 0;
        }
        current->image.reset();
        self.counters.coalesced_publishes++;
        self.push_publish_result_locked(std::move(coalesced));
        self.publishing.reset();
    }
    auto job = std::make_shared<Impl::PublishJob>();
    job->image = std::move(image);
    // The request copy is allocated here, after its own bytes were counted and admitted above.
    job->request = request;
    job->epoch = request.epoch != 0 ? request.epoch : self.epoch;
    job->ram_bytes = job->image->bytes();
    job->workspace_bytes = needed;
    job->catalog_apply_bytes = workspace.catalog_apply;
    job->shape = shape;
    job->external_credit_bytes = credit;
    // The image alias is the only part of this that is also a lease: the workspace belongs to the writer alone.
    self.ram_held += job->ram_bytes + needed;
    self.ram_lease += job->ram_bytes;
    self.counters.peak_bytes = std::max(self.counters.peak_bytes, self.resident_locked());
    self.publishing = job;
    self.cv_work.notify_all();
    self.cv_publish.notify_all();
    return ConversationStorePublishAdmission::accepted;
}

bool ConversationStore::take_publish(ConversationStorePublishResult& out) {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    if (impl_->publish_done.empty()) return false;
    out = impl_->publish_done.front();
    impl_->publish_done.pop_front();
    return true;
}

bool ConversationStore::wait_publish(ConversationStorePublishResult& out, uint64_t timeout_ms) {
    std::unique_lock<std::mutex> lock(impl_->mutex);
    if (impl_->publish_done.empty() && timeout_ms)
        impl_->cv_publish.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                                   [&] { return !impl_->publish_done.empty() ||
                                                impl_->shutdown != ConversationStoreShutdownState::none; });
    if (impl_->publish_done.empty()) return false;
    out = impl_->publish_done.front();
    impl_->publish_done.pop_front();
    return true;
}

bool ConversationStore::writer_busy() const {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    return bool(impl_->publishing);
}

uint64_t ConversationStore::ram_held_bytes() const {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    return impl_->resident_locked();
}

uint64_t ConversationStore::ram_lease_bytes() const {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    return impl_->ram_lease;
}

uint64_t ConversationStore::ram_read_budget() const {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    return impl_->ram_budget;
}

uint64_t ConversationStore::ram_available_bytes() const {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    const uint64_t held = impl_->resident_locked();
    return impl_->ram_budget > held ? impl_->ram_budget - held : 0;
}

void ConversationStore::suspend_read_admission() {
    StoreReadTraceBatch trace;
    std::lock_guard<std::mutex> guard(impl_->mutex);
    impl_->admission_suspended = true;
    impl_->trace_waiters_locked(trace);
}

uint64_t ConversationStore::publish_read_allowance(uint64_t bytes) {
    StoreReadTraceBatch trace;
    std::lock_guard<std::mutex> guard(impl_->mutex);
    if (impl_->shutdown != ConversationStoreShutdownState::none) return impl_->ram_budget;
    // Absolute, and never retroactive: a lower bound admits no new work but revokes nothing already admitted.
    impl_->ram_budget = bytes;
    // The caller has now published a complete absolute envelope, including any
    // lease it released after publish(). Do not add that transfer a second time.
    if (impl_->publishing) impl_->publishing->external_credit_bytes = 0;
    impl_->admission_suspended = false;
    impl_->retry_budget_deferred_locked(trace);
    impl_->trace_waiters_locked(trace);
    impl_->cv_work.notify_all();
    return impl_->ram_budget;
}

uint64_t ConversationStore::disk_charged_bytes() const {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    return impl_->charged;
}

uint64_t ConversationStore::disk_quota_bytes() const {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    return impl_->options.quota_bytes;
}

uint64_t ConversationStore::disk_available_bytes() const {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    // The outstanding catalog-copy reservation is part of what is committed: bytes this store may still have to
    // write for its own authority are never reported as available for payload.
    const uint64_t committed = sat_add(impl_->charged, impl_->catalog_reserve_locked());
    return impl_->options.quota_bytes > committed ? impl_->options.quota_bytes - committed : 0;
}

uint64_t ConversationStore::disk_free_bytes() const {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    return impl_->free_sample;
}

bool ConversationStore::invalidate_domain(const char* reason) {
    Impl& self = *impl_;
    std::shared_ptr<Impl::DurableWork> work;
    {
        StoreReadTraceBatch trace;
        std::lock_guard<std::mutex> guard(self.mutex);
        if (!self.on || !self.initialized || self.shutdown != ConversationStoreShutdownState::none) return false;
        if (self.faulted) return false;
        self.epoch += 1;
        if (self.domain_index < self.catalog.domains.size())
            self.catalog.domains[self.domain_index].epoch = self.epoch;
        self.counters.epoch_advances++;
        // Every unhanded completion of the previous epoch is retired here, never relabelled, and nothing of the
        // previous epoch is delivered again.
        self.retire_reads_locked(ConversationStoreReadStatus::stale,
                                 "the execution domain was invalidated before this read completed",
                                 trace, "domain_invalidated");
        work = std::make_shared<Impl::DurableWork>();
        work->plan.kind = Impl::CommitPlan::Kind::invalidate;
        work->plan.epoch = self.epoch;
        work->plan.domain_index = self.domain_index;
        self.durable.push_back(work);
        self.cv_work.notify_all();
    }
    (void)reason;
    // The durable commit happens on the storage thread; this caller waits for it without ever performing or holding a
    // storage wait itself, and reports a commit it could not confirm as a latched fault.  The run-intent marker is
    // still armed in that case, so the next start distrusts this domain.
    std::unique_lock<std::mutex> lock(self.mutex);
    const bool done =
        self.cv_durable.wait_for(lock, std::chrono::milliseconds(kDurableWaitMs), [&] { return work->done; });
    if (!done) {
        self.latch_fault_locked("the domain invalidation could not be confirmed in time");
        return false;
    }
    if (!work->ok) {
        // A safety invalidation that did not commit durably must fault: the marker stays armed either way, so the
        // next start distrusts this domain rather than trusting an epoch advance this run could not record.
        self.latch_fault_locked(work->detail.empty()
                                    ? "the domain invalidation could not be committed"
                                    : "the domain invalidation could not be committed: " + work->detail);
        return false;
    }
    return true;
}

void ConversationStore::request_shutdown(uint64_t finalize_before_ms) {
    Impl& self = *impl_;
    // This entry point must never emit tracing I/O, even with STRATA_STORE_TIMING enabled.
    StoreReadTraceBatch trace;
    trace.io_allowed = false;
    {
        std::lock_guard<std::mutex> guard(self.mutex);
        if (self.shutdown != ConversationStoreShutdownState::none) {
            if (self.shutdown == ConversationStoreShutdownState::pending)
                self.finalize_before_ms = std::min(self.finalize_before_ms, finalize_before_ms);
            return;
        }
        self.finalize_before_ms = finalize_before_ms;
        self.shutdown = ConversationStoreShutdownState::pending;
        self.promote_pending = false; // the final catalog, if allowed, persists the latest live policy
        self.admission_suspended = true;
        self.retire_reads_locked(ConversationStoreReadStatus::disabled, "the store closed before this read completed",
                                 trace, "closed");
        self.queue.clear();
        if (self.publishing && !self.publishing->started) {
            // Unstarted work owns no worker frame: its alias and charge can be released immediately.
            const std::shared_ptr<Impl::PublishJob>& job = self.publishing;
            job->cancelled = true;
            if (job->ram_bytes) {
                self.ram_held -= std::min(self.ram_held, job->ram_bytes);
                self.ram_lease -= std::min(self.ram_lease, job->ram_bytes);
                job->ram_bytes = 0;
            }
            if (job->workspace_bytes) {
                self.ram_held -= std::min(self.ram_held, job->workspace_bytes);
                job->workspace_bytes = 0;
            }
            job->image.reset();
            ConversationStorePublishResult result;
            result.status = ConversationStorePublishStatus::disabled;
            result.request_id = job->request.request_id;
            result.generation = job->request.generation;
            result.epoch = job->epoch;
            result.released = true;
            result.detail = "the store closed before the publication began";
            self.push_publish_result_locked(std::move(result));
            self.publishing.reset();
        }
    }
    self.cv_work.notify_all();
    self.cv_ready.notify_all();
    self.cv_publish.notify_all();
    self.cv_durable.notify_all();
}

ConversationStoreShutdownState ConversationStore::shutdown_state() const {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    return impl_->shutdown;
}

ConversationStoreShutdownState ConversationStore::finish_shutdown() {
    Impl& self = *impl_;
    if (shutdown_state() == ConversationStoreShutdownState::none) request_shutdown(0);
    {
        std::unique_lock<std::mutex> lock(self.mutex);
        self.cv_shutdown.wait(lock, [&] { return !self.finishing; });
        if (self.shutdown != ConversationStoreShutdownState::pending) return self.shutdown;
        self.finishing = true;
        self.stop.store(true);
    }
    self.cv_work.notify_all();
    self.cv_ready.notify_all();
    self.cv_publish.notify_all();
    self.cv_durable.notify_all();
    // A blocked filesystem call cannot be interrupted by this deadline. Keep all live ownership and the root lock
    // until every worker really exits; concurrent finish callers wait without holding the catalog mutex.
    if (self.writer.joinable()) self.writer.join();
    for (auto& thread : self.readers)
        if (thread.joinable()) thread.join();
    self.readers.clear();

    bool commit_final = false;
    {
        std::unique_lock<std::mutex> lock(self.mutex);
        // A diagnostic decode admitted before the fence is caller-thread I/O, not a worker. It too must drain
        // before another owner can acquire this root, while handed-out in-memory images stay releasable afterwards.
        self.cv_shutdown.wait(lock, [&] { return self.decode == 0; });
        for (const auto& job : self.jobs) {
            if (job->status != ConversationStoreReadStatus::none || job->active) continue;
            job->status = ConversationStoreReadStatus::disabled;
            job->detail = "the store closed before this read completed";
            self.release_reservation_locked(job);
            self.release_lease_locked(job);
            self.release_slot_locked(job);
            self.release_head_lease_locked(job);
            job->image.reset();
            self.free_job_prep_locked(job);
        }
        self.queue.clear();
        const uint64_t now_ms = uint64_t(std::chrono::duration_cast<std::chrono::milliseconds>(
                                            std::chrono::steady_clock::now().time_since_epoch()).count());
        // This mutex scope is the finalization start decision. Once begun, its real result is reported even if
        // filesystem I/O crosses the deadline or a later request tightens it. A faulted/non-owning store never starts.
        commit_final = self.initialized && self.lock_owned && !self.faulted &&
                       self.finalize_before_ms != 0 && now_ms < self.finalize_before_ms;
    }
    bool durable_clean = false;
    try {
        if (commit_final) {
            Impl::CommitPlan plan;
            plan.kind = Impl::CommitPlan::Kind::promote;
            plan.defer_reclaim = true; // drain -> durable catalog -> proven intent removal -> reclaim -> lock release
            std::string why;
            const CommitOutcome outcome = self.commit_plan(plan, why);
            if (outcome == CommitOutcome::durable_applied) {
                std::string intent_why;
                const bool cleared = self.write_intent(false, intent_why);
                std::lock_guard<std::mutex> guard(self.mutex);
                self.intent_active = !cleared;
                durable_clean = cleared;
                if (!cleared) self.latch_fault_locked("the run-intent marker could not be cleared: " + intent_why);
            } else {
                // A refusal or uncertain final commit is never a clean exit; intent remains armed for restart distrust.
                std::lock_guard<std::mutex> guard(self.mutex);
                self.intent_active = true;
                if (outcome == CommitOutcome::uncertain)
                    self.latch_fault_locked(why.empty() ? "the final catalog commit is not confirmed" : why);
            }
        }
    } catch (...) {
        // No escaped finalization may strand concurrent finish callers or turn missing durability proof into clean.
        std::lock_guard<std::mutex> guard(self.mutex);
        self.intent_active = true;
        self.latch_fault_locked("the shutdown finalization could not be confirmed");
    }
    bool reclaim = false;
    {
        std::lock_guard<std::mutex> guard(self.mutex);
        reclaim = durable_clean && self.initialized && self.lock_owned && !self.faulted;
        if (reclaim) {
            // No storage reader remains. A handed-out image owns RAM only, not a filesystem operation.
            for (auto& file : self.pending_files) file.head_id = 0;
        }
    }
    if (reclaim) {
        try {
            self.reclaim_pending();
        } catch (...) {
            // Reclamation is optional: unreclaimed files stay charged, and cannot undo already proven durability.
        }
    }
    int lock_fd = -1;
    {
        std::lock_guard<std::mutex> guard(self.mutex);
        lock_fd = self.lock_fd;
        self.lock_fd = -1;
    }
    // Even lock-release syscalls stay outside the mutex so another thread can keep observing pending.
    if (lock_fd >= 0) {
        ::flock(lock_fd, LOCK_UN);
        ::close(lock_fd);
    }
    {
        std::lock_guard<std::mutex> guard(self.mutex);
        self.lock_owned = false;
        self.on = false;
        self.shutdown = durable_clean ? ConversationStoreShutdownState::durable_clean
                                      : ConversationStoreShutdownState::quiescent_unclean;
        self.finishing = false;
    }
    self.cv_shutdown.notify_all();
    return shutdown_state();
}

void ConversationStore::close() {
    request_shutdown(UINT64_MAX);
    finish_shutdown();
}

ConversationStoreCounters ConversationStore::counters() const {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    ConversationStoreCounters out = impl_->counters;
    out.heads = uint64_t(impl_->catalog.heads.size());
    out.objects = uint64_t(impl_->objects.size());
    out.manifests = uint64_t(impl_->head_manifest_bytes.size());
    out.orphan_bytes = impl_->reclaimable;
    uint64_t pending = 0;
    for (const auto& file : impl_->pending_files) pending += file.charge;
    out.pending_delete_bytes = pending;
    uint64_t pinned = 0;
    for (const auto& head : impl_->catalog.heads) pinned += head.pinned ? 1 : 0;
    out.pinned_heads = pinned;
    uint64_t shared = 0, object_bytes = 0;
    for (const auto& object : impl_->objects) {
        shared += object.second.refs > 1 ? 1 : 0;
        object_bytes += object.second.bytes;   // the ledger's own allocated charge, never a second accounting
    }
    out.shared_objects = shared;
    out.object_bytes = object_bytes;
    uint64_t manifest_bytes = 0;
    for (const auto& manifest : impl_->head_manifest_bytes) manifest_bytes += manifest.second;
    out.manifest_bytes = manifest_bytes;
    return out;
}

std::string ConversationStore::describe() const {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    std::string text;
    text += "root: " + (impl_->root.empty() ? std::string("(none)") : impl_->root) + "\n";
    text += std::string("enabled: ") + (impl_->on ? "yes" : "no") +
            (impl_->reason.empty() ? "" : " (" + impl_->reason + ")") + "\n";
    text += std::string("healthy: ") + ((impl_->on && !impl_->faulted) ? "yes" : "no") +
            (impl_->fault_reason.empty() ? "" : " (" + impl_->fault_reason + ")") + "\n";
    text += std::string("shutdown: ") + conversation_store_shutdown_state_name(impl_->shutdown) + "\n";
    text += "owner lock: " + std::string(impl_->lock_owned ? "held" : "not held") + "\n";
    text += "issuer key digest: " +
            sha256_hex(sha256(impl_->issuer_key.data(), impl_->issuer_key.size())).substr(0, 16) + "\n";
    text += "run intent: " + std::string(impl_->intent_active ? "armed" : "clear") + "\n";
    text += "domain index: " + std::to_string(impl_->domain_index) + " of " +
            std::to_string(impl_->catalog.domains.size()) + ", epoch " + std::to_string(impl_->epoch) +
            ", catalog version " + std::to_string(impl_->catalog_version) + "\n";
    text += "heads: " + std::to_string(impl_->catalog.heads.size()) + " of " +
            std::to_string(impl_->options.head_limit) + ", objects " + std::to_string(impl_->objects.size()) +
            ", pub order " + std::to_string(impl_->catalog.published_order_counter) + "\n";
    text += "disk: charged " + std::to_string(impl_->charged) + " of quota " +
            std::to_string(impl_->options.quota_bytes) + ", reserved for the catalog's own copy " +
            std::to_string(impl_->catalog_reserve_locked()) + ", reclaimable " + std::to_string(impl_->reclaimable) +
            ", pending files " + std::to_string(impl_->pending_files.size()) + ", free sampled " +
            std::to_string(impl_->free_sample) + "\n";
    text += "index: " + std::to_string(impl_->metadata) + " of " + std::to_string(impl_->options.metadata_bytes) +
            " bytes estimated, preparation " + std::to_string(impl_->prep) + " bytes, read preparation " +
            std::to_string(impl_->read_prep) + " bytes, diagnostic decode " + std::to_string(impl_->decode) +
            " bytes\n";
    text += "ram: held " + std::to_string(impl_->resident_locked()) + ", leased " +
            std::to_string(impl_->ram_lease) + ", budget " + std::to_string(impl_->ram_budget) +
            (impl_->admission_suspended ? " (admission suspended)" : "") + "\n";
    text += "reads: queued " + std::to_string(impl_->queue.size()) + ", incoming " +
            std::to_string(impl_->incoming) + " of " + std::to_string(impl_->options.max_reads) + ", ready " +
            std::to_string(impl_->ready.size()) + ", held descriptors " + std::to_string(impl_->jobs.size()) + "\n";
    text += std::string("writer: ") + (impl_->publishing ? "busy" : "idle") + ", durable queue " +
            std::to_string(impl_->durable.size()) + ", promotion pending " +
            (impl_->promote_pending ? "yes" : "no") + "\n";
    text += "promotions: applied " + std::to_string(impl_->counters.promotions_applied) + ", coalesced " +
            std::to_string(impl_->counters.promotions_coalesced) + ", absorbed " +
            std::to_string(impl_->counters.promotions_absorbed) + ", refused " +
            std::to_string(impl_->counters.promotions_refused) + "\n";
    return text;
}

void ConversationStore::set_publish_fault_for_test(const char* step) {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    if (!step || !*step) {
        impl_->fault_steps.clear();
        return;
    }
    // Named steps accumulate and each one fires once, so a test can arm the whole sequence of a transaction (for
    // example a failed rename flush followed by a failed re-arm flush) before it starts.
    impl_->fault_steps.insert(step);
}

uint64_t ConversationStore::source_bytes_read_for_test() const { return impl_->source_bytes_read.load(); }

#ifdef STRATA_STORE_TEST_HOOKS
void ConversationStore::set_read_gate_for_test(void (*gate)(void*), void* context) {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    impl_->read_gate = gate;
    impl_->read_gate_context = context;
}

void ConversationStore::set_read_progress_gate_for_test(void (*gate)(void*), void* context) {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    impl_->read_progress_gate = gate;
    impl_->read_progress_gate_context = context;
}

void ConversationStore::set_catalog_gate_for_test(void (*gate)(void*, bool), void* context) {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    impl_->catalog_gate = gate;
    impl_->catalog_gate_context = context;
}

void ConversationStore::set_object_batch_gate_for_test(void (*gate)(void*), void* context) {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    impl_->object_batch_gate = gate;
    impl_->object_batch_gate_context = context;
}
#endif

const char* conversation_store_shutdown_state_name(ConversationStoreShutdownState state) {
    switch (state) {
        case ConversationStoreShutdownState::none: return "none";
        case ConversationStoreShutdownState::pending: return "pending";
        case ConversationStoreShutdownState::quiescent_unclean: return "quiescent_unclean";
        case ConversationStoreShutdownState::durable_clean: return "durable_clean";
    }
    return "unknown";
}

const char* conversation_store_read_status_name(ConversationStoreReadStatus status) {
    switch (status) {
        case ConversationStoreReadStatus::none: return "none";
        case ConversationStoreReadStatus::ready: return "ready";
        case ConversationStoreReadStatus::missing: return "missing";
        case ConversationStoreReadStatus::corrupt: return "corrupt";
        case ConversationStoreReadStatus::cancelled: return "cancelled";
        case ConversationStoreReadStatus::stale: return "stale";
        case ConversationStoreReadStatus::refused_budget: return "refused_budget";
        case ConversationStoreReadStatus::refused_quota: return "refused_quota";
        case ConversationStoreReadStatus::refused_limits: return "refused_limits";
        case ConversationStoreReadStatus::failed: return "failed";
        case ConversationStoreReadStatus::disabled: return "disabled";
    }
    return "unknown";
}

const char* conversation_store_read_state_name(ConversationStoreReadState state) {
    switch (state) {
        case ConversationStoreReadState::unknown: return "unknown";
        case ConversationStoreReadState::queued: return "queued";
        case ConversationStoreReadState::materializing: return "materializing";
        case ConversationStoreReadState::settled: return "settled";
        case ConversationStoreReadState::handed_out: return "handed_out";
        case ConversationStoreReadState::released: return "released";
    }
    return "unknown";
}

const char* conversation_store_publish_status_name(ConversationStorePublishStatus status) {
    switch (status) {
        case ConversationStorePublishStatus::stored: return "stored";
        case ConversationStorePublishStatus::replaced: return "replaced";
        case ConversationStorePublishStatus::retained_branch: return "retained_branch";
        case ConversationStorePublishStatus::stale: return "stale";
        case ConversationStorePublishStatus::refused_quota: return "refused_quota";
        case ConversationStorePublishStatus::refused_floor: return "refused_floor";
        case ConversationStorePublishStatus::refused_limits: return "refused_limits";
        case ConversationStorePublishStatus::refused_busy: return "refused_busy";
        case ConversationStorePublishStatus::coalesced: return "coalesced";
        case ConversationStorePublishStatus::failed: return "failed";
        case ConversationStorePublishStatus::disabled: return "disabled";
    }
    return "unknown";
}

const char* conversation_store_publish_admission_name(ConversationStorePublishAdmission admission) {
    switch (admission) {
        case ConversationStorePublishAdmission::accepted: return "accepted";
        case ConversationStorePublishAdmission::deferred: return "deferred";
        case ConversationStorePublishAdmission::refused: return "refused";
    }
    return "unknown";
}

}  // namespace strata::core
