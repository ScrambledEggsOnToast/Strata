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
#include <chrono>
#include <condition_variable>
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
constexpr uint64_t kMaxScanEntries = uint64_t(1) << 20;
constexpr uint64_t kMaxReclaimPerPass = 65536;
constexpr uint64_t kTempNameAttempts = 32;
constexpr uint32_t kDurableWaitMs = 30000;
constexpr uint64_t kIdlePollMs = 20;

uint64_t sat_add(uint64_t a, uint64_t b) { return a > UINT64_MAX - b ? UINT64_MAX : a + b; }
uint64_t sat_mul(uint64_t a, uint64_t b) { return a != 0 && b > UINT64_MAX / a ? UINT64_MAX : a * b; }
uint64_t sat_max3(uint64_t a, uint64_t b, uint64_t c) { return std::max(a, std::max(b, c)); }

// The most one committed change can add to the catalog: a single head's own encoding (its fixed fields plus every cut
// one image can advertise), the publication counter and one allocation's rounding.  A commit adds at most one head,
// and an eviction, promotion or invalidation only shrinks or repeats one, so this is what makes the next catalog
// copy's cost predictable before it is written.
uint64_t catalog_growth_bytes(const ConversationStoreLimits& limits) {
    return 512 + 130 + 4 + (uint64_t(limits.max_checkpoints) + 1) * 42;
}

// The outcome of one catalog transaction.  `unchanged` means not one durable byte changed and the previously
// committed authority is still the authority; `durable_applied` means the new catalog was renamed into place and
// every directory that changed was flushed; `uncertain` means the rename happened but its durability could not be
// confirmed, so nothing may be deleted, credited or reused until a restart re-reads the authority.
enum class CommitOutcome : uint8_t { unchanged = 0, durable_applied = 1, uncertain = 2 };

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

// Every host byte one admitted publication can hold at once, counted before any of it is allocated.  Each field is an
// upper bound on one phase's structures; `peak` is the largest sum of them that can be live together, so the writer
// never allocates beyond what it reserved (see publish() and publish_body()).
struct WriterWorkspace {
    uint64_t request_copy = 0;    // the copied ConversationStorePublish and its reuse intervals
    uint64_t layout = 0;          // the layout skeleton, its per-record prefix chains and the identity set
    uint64_t donor_decoded = 0;   // every decoded manifest this attempt retains (its parent and its donors)
    uint64_t donor_raw_peak = 0;  // the largest single manifest file it reads at once
    uint64_t chunks = 0;          // the most immutable chunks it can emit; every one is assumed unique
    uint64_t buffer = 0;          // the largest single chunk it reads into one buffer
    uint64_t manifest = 0;        // its own encoded manifest
    uint64_t catalog_apply = 0;   // the candidate catalog copy, the apply staging and the retire queue's room

    // I + J + max(W_layout/decode, W_chunks/maps/orphans + B, W_manifest/catalog/apply)
    uint64_t peak() const {
        uint64_t chunk_phase = sat_mul(chunks, sizeof(conversation_store_detail::ChunkRecord) + kAllocSlackBytes);
        // Every prospective chunk may be a distinct object: its node in the referenced/charged object maps, its node
        // in the new-object map, its entry in the persistent index and its slot in the head's own object list.
        chunk_phase = sat_add(chunk_phase,
                              sat_mul(chunks, 4 * conversation_store_detail::kMapNodeBytes + 2 * sizeof(Digest) + 40 +
                                                 2 * kAllocSlackBytes));
        // Every object this attempt writes is accounted before it is written, by name and by shard.
        chunk_phase = sat_add(chunk_phase, sat_mul(chunks, sizeof(PendingFile) + 2 * kAllocSlackBytes));
        chunk_phase = sat_add(chunk_phase, 2 * kAllocSlackBytes);   // the per-stream chunk reservations
        chunk_phase = sat_add(chunk_phase, buffer);
        const uint64_t base = sat_add(sat_add(request_copy, layout), donor_decoded);
        return sat_add(base, sat_max3(donor_raw_peak, chunk_phase, sat_add(manifest, catalog_apply)));
    }
};

// The bounded preparation one publication's parent and reuse donors need.  Built from the named heads' own recorded
// charges - never from head_limit x max_manifest_bytes and never from a default maximum - and computed from catalog
// metadata alone, with no I/O and no allocation.
struct DonorPrep {
    uint64_t decoded = 0;     // every decoded manifest this attempt retains
    uint64_t raw_peak = 0;    // the largest single raw manifest it reads at once
    uint64_t chunks = 0;      // the most chunks those donors can contribute through reuse intervals
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

// An object is durable before any manifest names it: a scratch file, a flush, a rename into its shard, then the
// shard's own flush - and the shard entry itself is flushed in `objects/` before anything can name it.
bool write_object_file(RootDirs& dirs, const std::string& temp, const Digest& id, const uint8_t* data, size_t size,
                       PendingFile& orphan, std::string& why) {
    const std::string hex = sha256_hex(id);
    const std::string shard = hex.substr(0, 2);
    bool created = false;
    if (!make_child_dir(dirs.objects, shard.c_str(), created, why)) return false;
    if (created && !fsync_dir(dirs.objects, "objects", why)) return false;
    const int shard_fd = open_shard_fd(dirs, shard, why);
    if (shard_fd < 0) return false;
    // The scratch name is reported to the caller before it is created, so a failure here can always account for the
    // bytes of a file that may exist.
    orphan.where = Where::tmp;
    orphan.name = temp;
    if (!write_new_file(dirs.tmp, temp, data, size, why)) {
        ::close(shard_fd);
        return false;
    }
    if (::renameat(dirs.tmp, temp.c_str(), shard_fd, hex.c_str()) != 0) {
        why = std::string("cannot publish an object: ") + std::strerror(errno);
        ::close(shard_fd);
        return false;
    }
    orphan = PendingFile{};
    const bool ok = fsync_dir(shard_fd, "an object shard", why);
    ::close(shard_fd);
    return ok;
}

// The mandatory verification: an object is trusted only after its bytes have reproduced its own content address.  The
// bytes are hashed and written from the *same* descriptor in one pass, so verification and restore can never disagree
// about which bytes were checked, and no buffer is trusted before every digest has matched.
bool hash_and_copy_object(int shard, const std::string& name, uint64_t size, conversation_store_detail::Sha256& hash,
                          StreamWrite& write, uint64_t offset, std::vector<uint8_t>& slab, Digest& out,
                          std::string& why) {
    const int fd = ::openat(shard, name.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) {
        why = std::string("cannot open an object: ") + std::strerror(errno);
        return false;
    }
    struct stat info {};
    if (::fstat(fd, &info) != 0 || !S_ISREG(info.st_mode) || uint64_t(info.st_size) != size) {
        why = "an object's size does not match the manifest";
        ::close(fd);
        return false;
    }
    uint64_t left = size;
    while (left) {
        const ssize_t n = ::read(fd, slab.data(), size_t(std::min<uint64_t>(slab.size(), left)));
        if (n < 0) {
            if (errno == EINTR) continue;
            why = std::string("cannot read an object: ") + std::strerror(errno);
            ::close(fd);
            return false;
        }
        if (n == 0) {
            why = "an object is shorter than the manifest says";
            ::close(fd);
            return false;
        }
        hash.update(slab.data(), size_t(n));
        if (!write(offset + size - left, slab.data(), size_t(n), why)) {
            ::close(fd);
            return false;
        }
        left -= uint64_t(n);
    }
    ::close(fd);
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
    if (live ? !selected->stage_parts.empty() : selected->stage_parts.size() + 1 != meta.stages) {
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
    std::string root;                 // diagnostics only: every access goes through `dirs`
    RootDirs dirs;
    uint32_t reader_threads = 1;

    // ---- authority ----
    std::mutex mutex;
    std::condition_variable cv_ready;
    std::condition_variable cv_publish;
    std::condition_variable cv_work;
    std::condition_variable cv_durable;
    std::array<uint8_t, 32> issuer_key{};
    Catalog catalog;
    uint64_t catalog_charge = 0;
    uint64_t epoch = 0;
    uint64_t catalog_version = 0;      // bumped by every applied commit
    uint32_t domain_index = 0;
    bool intent_active = false;
    bool on = false;
    bool closing = false;
    bool closed = false;
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
    std::map<Digest, ObjectState> objects;                  // every object the catalog authorizes
    std::map<uint64_t, std::vector<Digest>> head_objects;   // head id -> the objects its manifest names
    std::map<uint64_t, uint64_t> head_manifest_bytes;       // head id -> its manifest's allocated charge
    std::map<uint64_t, uint32_t> head_leases;               // head id -> images still using it
    std::map<uint64_t, uint64_t> head_recency;              // RAM-only eviction recency (published order is fallback)
    uint64_t recency_counter = 0;
    std::set<uint64_t> eviction_excluded;                   // heads protected by the in-flight publication
    std::vector<PendingFile> pending_files;                 // orphans and retired files awaiting confirmed deletion
    uint64_t charged = 0;         // allocated bytes of every file this module owns that still exists
    uint64_t reclaimable = 0;     // the part of `charged` no longer referenced but still on disk
    uint64_t free_sample = 0;     // last sampled filesystem space under the root
    uint64_t metadata = 0;        // estimated RAM of the in-RAM index
    uint64_t prep = 0;            // the live read descriptors' own query copies

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
        bool active = false;              // a reader thread is materializing it right now
        std::atomic<bool> cancelled{false};
        bool released = false;
        bool handed = false;
        bool leased = false;              // this job holds a slot in head_leases
        bool holds_slot = false;          // one of Options::max_reads incoming-image slots
        uint64_t reserved_bytes = 0;      // the peak charged to ram_held while it is not yet a lease
        uint64_t lease_bytes = 0;         // the finished image's charge, refunded by release_read()
        uint64_t peak_bytes = 0;          // the peak its admission reserved
        uint64_t prep_bytes = 0;          // its own query copy, charged to `prep`
        uint64_t epoch_at_admission = 0;
        uint64_t head_id = 0;
        uint64_t tokens = 0;
        bool live = false;
        uint32_t stages = 0;
        uint64_t payload_bytes = 0;
        std::shared_ptr<const SavedConversation> image;
        std::string detail;
    };
    std::vector<std::shared_ptr<ReadJob>> jobs;      // queued, active, ready or handed out (bounded)
    std::deque<std::shared_ptr<ReadJob>> queue;      // waiting for the allowance and a slot (demand before prefetch)
    std::deque<std::shared_ptr<ReadJob>> ready;      // settled, waiting for take_ready()
    uint64_t incoming = 0;                           // jobs holding one of Options::max_reads slots

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
    };
    std::shared_ptr<PublishJob> publishing;
    std::deque<ConversationStorePublishResult> publish_done;

    struct CommitPlan {
        enum class Kind : uint8_t { publish, eviction, invalidate, promote };
        Kind kind = Kind::promote;
        Catalog candidate;                        // exact for publish/eviction; re-derived at commit otherwise
        uint64_t version = 0;                     // the catalog version an exact candidate was planned against
        std::vector<uint64_t> retiring;           // heads the applied catalog drops
        uint64_t new_head = 0;                    // publish only
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
    std::deque<std::shared_ptr<DurableWork>> durable;
    std::atomic<bool> stop{false};

    // ---- counters and test seams ----
    ConversationStoreCounters counters;
    std::set<std::string> fault_steps;   // armed failure steps; each one fires once and is then disarmed
    std::atomic<uint64_t> source_bytes_read{0};
    std::atomic<uint64_t> name_counter{0};

    // ------------------------------ helpers ------------------------------
    // Every one of these assumes the caller holds `mutex` unless its name says otherwise; not one of them performs a
    // storage wait while the lock is held.
    uint64_t sample_free_locked();
    uint64_t resident_locked() const { return ram_held + metadata + prep; }
    std::string temp_name();
    std::string intent_name() const;
    bool store_is_empty(std::string& why);
    void latch_fault_locked(const std::string& why);
    void recount_metadata_locked();
    bool revive_object_locked(const Digest& object);
    uint64_t retire_head_locked(uint64_t head_id);
    void touch_recency_locked(uint64_t head_id);
    uint64_t recency_of_locked(const CatalogHead& head) const;
    uint64_t choose_victim_locked(const Catalog& candidate, const std::set<uint64_t>& excluded, bool want_protected);
    void cap_protected_locked(Catalog& candidate);
    bool obtain_head_slot_locked(Catalog& candidate, uint64_t keep_head, std::vector<uint64_t>& retiring);
    bool head_manifest_locked(uint64_t head_id, Digest& manifest_id, uint64_t& bytes) const;
    bool load_manifest_of(uint64_t head_id, uint64_t epoch_now, ManifestMeta& out, std::string& why);
    bool stream_identity_of(const std::vector<ContainerMeta>& containers, const ConversationStoreStreamId& id,
                            Digest& scope, uint64_t& ordinal, std::string& why) const;

    // These do I/O, so they are never called with `mutex` held.
    bool remove_pending_file(const PendingFile& file, std::string& why);
    void discard_files(std::vector<PendingFile>&& files);
    void reclaim_pending();
    void queue_orphans(std::vector<PendingFile>& files);
    CommitOutcome commit_plan(CommitPlan& plan, std::string& why);
    bool evict_head_durably(uint64_t head_id, std::string& why);
    bool make_room(uint64_t incoming_bytes, bool& floor_limited, std::string& why);
    CommitOutcome write_catalog_or_refuse(CommitPlan& plan, std::string& why);
    bool take_fault(const char* step);

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

    // Lookup: catalog metadata only, under the mutex, no payload read and no GPU pointer exposed.
    void collect_matches(const ConversationStoreSelector& selector, const int32_t* ids, size_t count,
                         const std::vector<ConversationImageKey>& images,
                         std::vector<ConversationStoreMatch>& out) const;

    // Read pipeline.  Settlement is idempotent and every path funnels through it.
    bool admit_read_locked(const ConversationStoreQuery& query, bool demand, std::string& why);
    void settle_job_locked(const std::shared_ptr<ReadJob>& job, ConversationStoreReadStatus status, std::string detail);
    void retire_job_locked(const std::shared_ptr<ReadJob>& job, ConversationStoreReadStatus status,
                           const std::string& detail);
    void retire_reads_locked(ConversationStoreReadStatus status, const std::string& detail);
    void revoke_speculation_locked(bool because_of_demand = true);
    void release_reservation_locked(const std::shared_ptr<ReadJob>& job);
    void release_lease_locked(const std::shared_ptr<ReadJob>& job);
    void release_slot_locked(const std::shared_ptr<ReadJob>& job);
    void release_head_lease_locked(const std::shared_ptr<ReadJob>& job);
    void erase_job_locked(const std::shared_ptr<ReadJob>& job);
    void free_job_prep_locked(const std::shared_ptr<ReadJob>& job);
    bool deliver_locked(ConversationStoreRead& out, uint64_t request_id, uint64_t generation, bool specific);
    void push_publish_result_locked(ConversationStorePublishResult&& result);

    void writer_body();
    void reader_body();
    void publish_body(const std::shared_ptr<PublishJob>& job, ConversationStorePublishResult& out);
    void read_body(const std::shared_ptr<ReadJob>& job);

    bool open_root(std::string& why);
    // Reads and decodes the durable catalog and decides this domain's epoch.  It may not write anything: the first
    // catalog transaction happens only once the whole physical ledger has been accounted (see reconcile()).
    bool read_catalog(std::string& why);
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
        if (key.size() != 32) {
            why = "the issuer key file is not 32 bytes";
            return false;
        }
        std::memcpy(issuer_key.data(), key.data(), key.size());
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

bool ConversationStore::Impl::reconcile(std::string& why) {
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
        // A record of this domain that this run must not reuse - an unclean start or a retired epoch - is dropped
        // here, after its manifest and its objects have been accounted above, so its bytes are charged into the
        // ledger before the first catalog transaction forgets it.
        if (head.domain == domain_index && (unclean_start || head.epoch != epoch)) {
            dropped.push_back(head.head_id);
            continue;
        }
        const auto manifest = present_manifests.find(head.manifest_id);
        ManifestMeta meta;
        std::string detail;
        if (manifest == present_manifests.end() || !load_manifest_of(head.head_id, head.epoch, meta, detail)) {
            counters.corrupt_records++;
            dropped.push_back(head.head_id);
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
                catalog.heads.erase(catalog.heads.begin() + std::ptrdiff_t(i));
                break;
            }
        counters.evictions++;
    }
    recount_metadata_locked();
    while (metadata > options.metadata_bytes && catalog.heads.size() > 1) {
        const uint64_t victim = choose_victim_locked(catalog, {}, false);
        if (victim == 0) break;
        for (size_t i = 0; i < catalog.heads.size(); ++i)
            if (catalog.heads[i].head_id == victim) {
                retiring.push_back(victim);
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

uint64_t ConversationStore::Impl::retire_head_locked(uint64_t head_id) {
    PendingFile manifest;
    bool have_manifest = false;
    for (size_t i = 0; i < catalog.heads.size(); ++i) {
        if (catalog.heads[i].head_id != head_id) continue;
        const Digest id = catalog.heads[i].manifest_id;
        bool shared = false;   // defensive: never queue a manifest another retained head still names
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
        catalog.heads.erase(catalog.heads.begin() + std::ptrdiff_t(i));
        break;
    }
    uint64_t freed = manifest.charge;
    if (have_manifest) {
        reclaimable += manifest.charge;
        pending_files.push_back(std::move(manifest));
    }
    const auto list = head_objects.find(head_id);
    if (list != head_objects.end()) {
        for (const Digest& object : list->second) {
            auto state = objects.find(object);
            if (state == objects.end()) continue;
            if (state->second.refs > 0) --state->second.refs;
            if (state->second.refs != 0) continue;
            PendingFile file;
            file.where = Where::objects;
            file.object = object;
            file.shard = sha256_hex(object).substr(0, 2);
            file.name = sha256_hex(object);
            file.charge = state->second.bytes;
            file.head_id = head_id;
            reclaimable += file.charge;
            freed += file.charge;
            pending_files.push_back(std::move(file));
        }
        head_objects.erase(list);
    }
    head_manifest_bytes.erase(head_id);
    // The lease count is NOT erased here: an open hold or a handed-out image keeps it until it finishes, and only
    // then may the files it still reads be deleted and credited.
    head_recency.erase(head_id);
    eviction_excluded.erase(head_id);
    return freed;
}

void ConversationStore::Impl::touch_recency_locked(uint64_t head_id) { head_recency[head_id] = ++recency_counter; }

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
        counters.evictions++;
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

bool ConversationStore::Impl::load_manifest_of(uint64_t head_id, uint64_t epoch_now, ManifestMeta& out,
                                               std::string& why) {
    Digest manifest_id{};
    uint64_t bytes = 0;
    {
        std::lock_guard<std::mutex> guard(mutex);
        if (!head_manifest_locked(head_id, manifest_id, bytes)) {
            why = "no retained head has that identity";
            return false;
        }
    }
    std::vector<uint8_t> raw;
    // The read is capped by the head's own recorded charge as well as the configured ceiling, so the writer's
    // reservation - which is built from those same recorded charges - can never be smaller than this read.
    const uint64_t capped = bytes != 0 && bytes < limits.max_manifest_bytes ? bytes : limits.max_manifest_bytes;
    if (!read_file(dirs.manifests, sha256_hex(manifest_id), capped, raw, why)) return false;
    {
        std::lock_guard<std::mutex> guard(mutex);
        counters.bytes_read += raw.size();
    }
    if (!conversation_store_detail::manifest_decode(raw.data(), raw.size(), limits, out, why)) return false;
    if (!(out.manifest_id == manifest_id) || out.epoch != epoch_now) {
        why = "that manifest does not belong to this execution-domain epoch";
        return false;
    }
    return true;
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
    const auto charge_of = [&](uint64_t head_id, uint64_t& bytes) {
        for (const auto& head : catalog.heads)
            if (head.head_id == head_id) {
                bytes = head.manifest_bytes;
                return true;
            }
        return false;
    };
    uint64_t bytes = 0;
    if (request.supersedes != 0 && charge_of(request.supersedes, bytes)) {
        const uint64_t raw = bytes != 0 && bytes < limits.max_manifest_bytes ? bytes : limits.max_manifest_bytes;
        prep.raw_peak = std::max(prep.raw_peak, raw);
        prep.decoded = sat_add(prep.decoded, conversation_store_detail::manifest_workspace_bound(raw));
    }
    for (const auto& entry : request.reuse) {
        if (!entry.head_id || !charge_of(entry.head_id, bytes)) continue;
        const uint64_t raw = bytes != 0 && bytes < limits.max_manifest_bytes ? bytes : limits.max_manifest_bytes;
        prep.raw_peak = std::max(prep.raw_peak, raw);
        prep.decoded = sat_add(prep.decoded, conversation_store_detail::manifest_workspace_bound(raw));
        // Every donor chunk costs 44 encoded bytes, so the donor's own encoding bounds how many chunks one interval
        // can adopt.  This is deliberately generous: the writer counts the interval's real donor chunks once the
        // manifest is decoded, and only ever lowers what was reserved.
        prep.chunks = sat_add(prep.chunks, sat_add(raw / 44, 2));
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
    work.request_copy = sat_add(sat_mul(uint64_t(request.reuse.size()), sizeof(ConversationStoreReuse)),
                                kJobFixedOverhead);
    work.layout = conversation_store_detail::image_layout_workspace(shape);
    const DonorPrep prep = donor_prep_locked(request);
    work.donor_decoded = prep.decoded;
    work.donor_raw_peak = prep.raw_peak;
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
        chunks, chunks, shape.cuts};
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
    if (workspace > options.metadata_bytes || metadata + prep > options.metadata_bytes - workspace) {
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
    counters.peak_bytes = std::max(counters.peak_bytes, ram_held);
    return true;
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
    {
        std::lock_guard<std::mutex> guard(mutex);
        // After an uncertain catalog transaction this process cannot know which catalog is durable, so neither
        // catalog's files may be deleted: every pending file is retained, stays charged, and awaits a restart that
        // re-reads the authority.
        if (faulted) return;
        std::vector<PendingFile> keep;
        keep.reserve(pending_files.size());
        for (auto& file : pending_files) {
            // A file whose head still has an open hold or a handed-out image is not deleted yet: a reader's
            // descriptors may still name it, and its bytes are still in use.
            if (file.head_id != 0 && head_leases.count(file.head_id)) {
                keep.push_back(std::move(file));
                continue;
            }
            if (batch.size() >= kMaxReclaimPerPass) {
                keep.push_back(std::move(file));
                continue;
            }
            batch.push_back(std::move(file));
        }
        pending_files = std::move(keep);
    }
    std::vector<PendingFile> retry;
    for (PendingFile& file : batch) {
        if (file.where == Where::objects) {
            bool referenced = false;
            {
                std::lock_guard<std::mutex> guard(mutex);
                const auto state = objects.find(file.object);
                referenced = state != objects.end() && state->second.refs != 0;
            }
            if (referenced) {
                // Referenced again: the file is needed, so the pending record goes away and its charge stays.
                std::lock_guard<std::mutex> guard(mutex);
                revive_object_locked(file.object);
                continue;
            }
        }
        std::string why;
        if (remove_pending_file(file, why)) {
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
        recount_metadata_locked();
    }
}

void ConversationStore::Impl::recount_metadata_locked() {
    // Counted, never encoded: this runs after a rename, where an allocation for the sake of measuring one would be
    // exactly the escape the transaction must not have.
    uint64_t bytes = conversation_store_detail::catalog_encoded_size(catalog);
    bytes += uint64_t(objects.size()) * kObjectBytesEstimate;
    for (const auto& entry : head_objects) bytes += uint64_t(entry.second.size()) * sizeof(Digest);
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
    for (uint64_t victim : plan.retiring) retire_head_locked(victim);
    catalog = std::move(plan.candidate);
    catalog_version++;
    recount_metadata_locked();
    counters.heads = uint64_t(catalog.heads.size());
    counters.objects = uint64_t(objects.size());
    counters.manifests = uint64_t(head_manifest_bytes.size());
}

CommitOutcome ConversationStore::Impl::write_catalog_or_refuse(CommitPlan& plan, std::string& why) {
    std::vector<uint8_t> encoded;
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
                plan.candidate = catalog;
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
                plan.candidate = catalog;
                break;
            default:
                if (plan.version != catalog_version) {
                    why = "the catalog changed while this plan was being built";
                    return CommitOutcome::unchanged;
                }
                break;
        }
        encoded = conversation_store_detail::catalog_encode(plan.candidate);
        plan.new_catalog_charge = allocated_charge(encoded.size());
        if (!admit_catalog_copy_locked(plan.new_catalog_charge, plan.candidate, why))
            return CommitOutcome::unchanged;
        if (!stage_apply_locked(plan, why)) return CommitOutcome::unchanged;
        charged += plan.new_catalog_charge;
    } catch (const std::exception& error) {
        why = std::string("the catalog transaction could not be prepared: ") + error.what();
        return CommitOutcome::unchanged;
    } catch (...) {
        why = "the catalog transaction could not be prepared";
        return CommitOutcome::unchanged;
    }

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
        apply_staged_locked(plan);
    } catch (const std::exception& error) {
        const std::string detail = std::string("the catalog apply failed after the rename: ") + error.what();
        std::lock_guard<std::mutex> guard(mutex);
        latch_fault_locked(detail);
        why = detail;
        return CommitOutcome::uncertain;
    } catch (...) {
        const std::string detail = "the catalog apply failed after the rename";
        std::lock_guard<std::mutex> guard(mutex);
        latch_fault_locked(detail);
        why = detail;
        return CommitOutcome::uncertain;
    }
    // Deletions happen here: the authority that named these files is already durable, and the mutex is released.
    reclaim_pending();
    return CommitOutcome::durable_applied;
}

CommitOutcome ConversationStore::Impl::commit_plan(CommitPlan& plan, std::string& why) {
    return write_catalog_or_refuse(plan, why);
}

bool ConversationStore::Impl::evict_head_durably(uint64_t head_id, std::string& why) {
    CommitPlan plan;
    plan.kind = CommitPlan::Kind::eviction;
    {
        std::lock_guard<std::mutex> guard(mutex);
        plan.version = catalog_version;
        plan.candidate = catalog;
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
        counters.evictions++;
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
        if (!evict_head_durably(victim, why)) return false;
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Matching: catalog metadata only, no payload is read and no GPU pointer is exposed
// ---------------------------------------------------------------------------------------------------------------

void ConversationStore::Impl::collect_matches(const ConversationStoreSelector& selector, const int32_t* ids,
                                             size_t count, const std::vector<ConversationImageKey>& images,
                                             std::vector<ConversationStoreMatch>& out) const {
    out.clear();
    if (!on || faulted) return;
    const Digest bucket = conversation_store_detail::store_bucket_digest(options, selector);
    conversation_store_detail::PrefixChain chain(bucket, ids, count, images);
    for (const auto& head : catalog.heads) {
        if (!(head.bucket == bucket) || head.epoch != epoch) continue;
        for (const auto& cut : head.cuts) {
            // The last prompt token always starts the next verify window, so only a strictly shorter prefix is usable.
            if (cut.tokens == 0 || cut.tokens >= count) continue;
            Digest digest{};
            if (!chain.digest(cut.tokens, digest) || !(digest == cut.digest)) continue;
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
    if (lease->second == 0) head_leases.erase(lease);
}

void ConversationStore::Impl::free_job_prep_locked(const std::shared_ptr<ReadJob>& job) {
    prep -= std::min(prep, job->prep_bytes);
    job->prep_bytes = 0;
    job->query.tokens.clear();
    job->query.tokens.shrink_to_fit();
    job->query.images.clear();
    job->query.images.shrink_to_fit();
}

void ConversationStore::Impl::erase_job_locked(const std::shared_ptr<ReadJob>& job) {
    free_job_prep_locked(job);
    for (auto it = jobs.begin(); it != jobs.end(); ++it)
        if (*it == job) {
            jobs.erase(it);
            break;
        }
}

void ConversationStore::Impl::settle_job_locked(const std::shared_ptr<ReadJob>& job,
                                                ConversationStoreReadStatus status, std::string detail) {
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
    if (status == ConversationStoreReadStatus::ready && job->image) {
        release_reservation_locked(job);
        const uint64_t bytes = job->image->bytes();
        ram_held += bytes;
        ram_lease += bytes;
        job->lease_bytes = bytes;
        counters.peak_bytes = std::max(counters.peak_bytes, ram_held);
        if (job->released) {
            // The caller already released this request: the image is dropped and its charge is not kept.
            release_lease_locked(job);
            release_slot_locked(job);
            release_head_lease_locked(job);
            job->image.reset();
            erase_job_locked(job);
            return;
        }
        ready.push_back(job);
        cv_ready.notify_all();
        return;
    }
    release_reservation_locked(job);
    release_lease_locked(job);
    release_slot_locked(job);
    release_head_lease_locked(job);
    job->image.reset();
    if (job->released) {
        erase_job_locked(job);
        return;
    }
    if (!job->cancelled) {
        ready.push_back(job);
        cv_ready.notify_all();
        return;
    }
    // A cancelled request reports its terminal status but is never delivered; its descriptor is spent.
    free_job_prep_locked(job);
    cv_ready.notify_all();
}

void ConversationStore::Impl::retire_job_locked(const std::shared_ptr<ReadJob>& job,
                                                ConversationStoreReadStatus status, const std::string& detail) {
    if (job->handed) return;                       // already the caller's image: release_read() is its way out
    job->cancelled = true;
    if (job->active) return;                       // the reader drops it at its next chunk boundary
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
    release_head_lease_locked(job);
    job->image.reset();
    // A job that already holds a *refusal* keeps its own verdict; a queued or ready completion takes the authority's
    // terminal status, because a completion that can no longer be handed out must never keep reporting `ready`.
    if (job->status != ConversationStoreReadStatus::none &&
        job->status != ConversationStoreReadStatus::ready) {
        free_job_prep_locked(job);
        return;
    }
    job->status = status;
    job->detail = detail;
    switch (status) {
        case ConversationStoreReadStatus::stale: counters.reads_stale++; break;
        case ConversationStoreReadStatus::cancelled: counters.reads_cancelled++; break;
        default: break;
    }
    free_job_prep_locked(job);
}

void ConversationStore::Impl::retire_reads_locked(ConversationStoreReadStatus status, const std::string& detail) {
    for (const auto& job : jobs) retire_job_locked(job, status, detail);
    cv_ready.notify_all();
}

void ConversationStore::Impl::revoke_speculation_locked(bool because_of_demand) {
    if (because_of_demand) {
        bool waiting_demand = false;
        for (const auto& job : jobs)
            if (job->demand && !job->handed && !job->cancelled) waiting_demand = true;
        if (!waiting_demand) return;
    }
    for (const auto& job : jobs) {
        if (job->demand || job->handed || job->released) continue;
        // Speculative work and speculation that is already materialized both give way to required work: a ready image
        // is dropped here, so its RAM and its incoming-image slot are back before the demand or capture needs them.
        retire_job_locked(job, ConversationStoreReadStatus::cancelled, "revoked in favour of a required operation");
    }
    cv_work.notify_all();
}

bool ConversationStore::Impl::admit_read_locked(const ConversationStoreQuery& query, bool demand, std::string& why) {
    // Dedupe by (request_id, generation): the demand for the same request upgrades the prefetch in place.
    for (const auto& job : jobs) {
        if (job->query.request_id != query.request_id || job->query.generation != query.generation) continue;
        if (job->handed || job->released) {
            why = "that request has already been handed out";
            return false;
        }
        if (demand && !job->demand) {
            job->demand = true;
            if (!job->active)
                for (auto it = queue.begin(); it != queue.end(); ++it)
                    if (*it == job) {
                        queue.erase(it);
                        queue.push_front(job);
                        break;
                    }
            revoke_speculation_locked();
        }
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
        return false;
    }
    if (query.tokens.size() > size_t(limits.max_tokens) || query.images.size() > size_t(limits.max_images) ||
        metadata + prep + own_prep > options.metadata_bytes) {
        counters.refused_limits++;
        why = "the store's preparation ceiling cannot hold another read descriptor";
        return false;
    }
    std::vector<ConversationStoreMatch> found;
    collect_matches(query.selector, query.tokens.data(), query.tokens.size(), query.images, found);
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
        cv_ready.notify_all();
        return true;
    }
    const ConversationStoreMatch best = found.front();
    // The whole peak this read will allocate, including its final image's scaffolding.  A demand that can never fit
    // the published allowance is refused here rather than given a reservation it can never use.
    const uint64_t peak = uint64_t(best.payload_bytes) + uint64_t(best.payload_bytes) / 4 +
                          2 * uint64_t(best.manifest_bytes) + own_prep + kReadSlabBytes + kReadFixedOverhead;
    if (ram_budget != 0 && peak > ram_budget) {
        counters.refused_budget++;
        why = "the read's peak does not fit the published RAM allowance";
        return false;
    }
    auto job = std::make_shared<ReadJob>();
    job->query = query;
    job->demand = demand;
    job->prep_bytes = own_prep;
    job->epoch_at_admission = epoch;
    prep += own_prep;
    job->head_id = best.head_id;
    job->tokens = uint64_t(best.tokens);
    job->live = best.live;
    job->stages = best.stages;
    job->payload_bytes = best.payload_bytes;
    job->peak_bytes = peak;
    jobs.push_back(job);
    queue.push_back(job);
    counters.matches++;
    if (demand) revoke_speculation_locked();
    cv_work.notify_all();
    return true;
}

bool ConversationStore::Impl::deliver_locked(ConversationStoreRead& out, uint64_t request_id, uint64_t generation,
                                             bool specific) {
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
        out.epoch = job->epoch_at_admission;
        out.tokens = int64_t(job->tokens);
        out.live = job->live;
        out.stages = job->stages;
        out.payload_bytes = job->payload_bytes;
        out.image = job->image;
        out.detail = job->detail;
        job->handed = true;
        counters.reads_handed_out++;
        ready.erase(it);
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

void ConversationStore::Impl::read_body(const std::shared_ptr<ReadJob>& job) {
    const ConversationStoreQuery& query = job->query;
    const uint64_t head_id = job->head_id;
    const uint64_t tokens = job->tokens;
    const uint64_t epoch_at_admission = job->epoch_at_admission;
    const auto stop_with = [&](ConversationStoreReadStatus status, const std::string& detail) {
        std::lock_guard<std::mutex> guard(mutex);
        settle_job_locked(job, status, detail);
    };
    Digest manifest_id{};
    uint64_t manifest_bytes = 0;
    {
        std::lock_guard<std::mutex> guard(mutex);
        if (!on || closing) {
            settle_job_locked(job, ConversationStoreReadStatus::disabled,
                      reason.empty() ? std::string("the store stopped serving") : reason);
            return;
        }
        if (faulted) {
            settle_job_locked(job, ConversationStoreReadStatus::disabled, fault_reason);
            return;
        }
        if (epoch != epoch_at_admission) {
            settle_job_locked(job, ConversationStoreReadStatus::stale, "the execution domain advanced before this read started");
            return;
        }
        if (!head_manifest_locked(head_id, manifest_id, manifest_bytes)) {
            settle_job_locked(job, ConversationStoreReadStatus::stale, "the head this read selected is no longer retained");
            return;
        }
    }
    std::vector<uint8_t> raw;
    std::string why;
    if (!read_file(dirs.manifests, sha256_hex(manifest_id), limits.max_manifest_bytes, raw, why)) {
        stop_with(ConversationStoreReadStatus::failed, why);
        return;
    }
    {
        std::lock_guard<std::mutex> guard(mutex);
        counters.bytes_read += raw.size();
    }
    conversation_store_detail::ManifestMeta meta;
    if (!conversation_store_detail::manifest_decode(raw.data(), raw.size(), limits, meta, why) ||
        !(meta.manifest_id == manifest_id)) {
        stop_with(ConversationStoreReadStatus::corrupt, why.empty() ? "the manifest does not match its identity" : why);
        return;
    }
    {
        std::lock_guard<std::mutex> guard(mutex);
        if (job->cancelled.load()) return;
        if (epoch != epoch_at_admission) {
            settle_job_locked(job, ConversationStoreReadStatus::stale,
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
    // The exact figure the restore will hold, computed from the decoded metadata before any of it is allocated: a
    // read whose restored image would not fit the published allowance fails here and is never charged past it.
    const uint64_t scaffolding = uint64_t(meta.streams.size()) *
                                     (sizeof(conversation_store_detail::RestoreTarget) + 64) +
                                 uint64_t(meta.containers.size()) * sizeof(ContainerMeta) +
                                 uint64_t(meta.objects.size()) * kObjectBytesEstimate;
    const uint64_t needed = uint64_t(meta.payload_bytes) + uint64_t(meta.payload_bytes) / 4 + raw.size() +
                            job->prep_bytes + kReadSlabBytes + kReadFixedOverhead + scaffolding;
    {
        std::lock_guard<std::mutex> guard(mutex);
        if (job->cancelled.load()) return;
        if (needed > job->reserved_bytes) {
            if (needed > ram_budget || resident_locked() - job->reserved_bytes > ram_budget - needed) {
                settle_job_locked(job, ConversationStoreReadStatus::refused_budget,
                                  "the restored image does not fit the published RAM allowance");
                return;
            }
            ram_held += needed - job->reserved_bytes;
            job->reserved_bytes = needed;
            counters.peak_bytes = std::max(counters.peak_bytes, ram_held);
        }
    }
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
        for (const auto& chunk : meta.streams[i].chunks) {
            if (job->cancelled.load()) return;
            {
                std::lock_guard<std::mutex> guard(mutex);
                if (epoch != epoch_at_admission) {
                    settle_job_locked(job, ConversationStoreReadStatus::stale,
                                      "the execution domain advanced before this read completed");
                    return;
                }
            }
            const std::string name = sha256_hex(chunk.object);
            const int shard = open_shard_fd(dirs, name.substr(0, 2), why);
            if (shard < 0) {
                stop_with(ConversationStoreReadStatus::corrupt, why);
                return;
            }
            Digest actual{};
            auto hash = conversation_store_detail::object_hasher(identity_scope, id.kind, identity_ordinal,
                                                                chunk.length);
            const bool ok = hash_and_copy_object(shard, name, chunk.length, hash, targets[i].write, chunk.offset,
                                                 slab, actual, why);
            ::close(shard);
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
                std::lock_guard<std::mutex> guard(mutex);
                counters.bytes_read += chunk.length;
            }
            source_bytes_read += chunk.length;
        }
    }
    conversation_store_detail::restore_finish(image);
    if (!restored_cut_matches(image, meta, query, tokens, job->live, digest, why)) {
        stop_with(ConversationStoreReadStatus::corrupt, why);
        return;
    }
    image.freeze();
    auto shared = std::make_shared<SavedConversation>(std::move(image));
    std::lock_guard<std::mutex> guard(mutex);
    if (job->cancelled.load()) return;
    if (shared->bytes() > job->reserved_bytes) {
        settle_job_locked(job, ConversationStoreReadStatus::failed,
                          "the restored image is larger than the peak it was admitted under");
        return;
    }
    job->image = shared;
    job->payload_bytes = meta.payload_bytes;
    job->stages = meta.stages;
    settle_job_locked(job, ConversationStoreReadStatus::ready, std::string());
}

void ConversationStore::Impl::reader_body() {
    std::unique_lock<std::mutex> lock(mutex);
    while (!stop.load()) {
        revoke_speculation_locked();
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
                                  "the read's peak does not fit the published allowance");
                continue;
            }
            if (!job->demand && (ram_budget == 0 || admission_suspended || incoming >= options.max_reads)) {
                // Speculative work never waits: it is refused when it cannot start now.
                it = queue.erase(it);
                settle_job_locked(job, ConversationStoreReadStatus::refused_budget,
                                  "the read allowance has no spare room for a prefetch");
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
                    if (ram_held + job->peak_bytes > ram_budget) {
                        if (job->demand) continue;                     // a demand waits for room
                        queue.erase(it);
                        settle_job_locked(job, ConversationStoreReadStatus::refused_budget,
                                          "the read allowance has no spare room for a prefetch");
                        break;
                    }
                    chosen = job;
                    queue.erase(it);
                    break;
                }
            }
        }
        if (!chosen) {
            cv_work.wait_for(lock, std::chrono::milliseconds(kIdlePollMs));
            continue;
        }
        // The peak is charged here, under the same lock that guards `admission_suspended`, so a later suspension
        // always sees every byte this read can still allocate.
        chosen->active = true;
        chosen->reserved_bytes = chosen->peak_bytes;
        ram_held += chosen->peak_bytes;
        counters.peak_bytes = std::max(counters.peak_bytes, ram_held);
        chosen->holds_slot = true;
        ++incoming;
        if (chosen->head_id) {
            head_leases[chosen->head_id] += 1;
            chosen->leased = true;
        }
        lock.unlock();
        try {
            read_body(chosen);
            lock.lock();
        } catch (const std::bad_alloc&) {
            if (!lock.owns_lock()) lock.lock();
            settle_job_locked(chosen, ConversationStoreReadStatus::failed,
                              "the read could not allocate within its admitted peak");
            cv_work.notify_all();
            continue;
        } catch (...) {
            if (!lock.owns_lock()) lock.lock();
            settle_job_locked(chosen, ConversationStoreReadStatus::failed, "the read aborted unexpectedly");
            cv_work.notify_all();
            continue;
        }
        if (chosen->status == ConversationStoreReadStatus::none) {
            // The body returned without a verdict because the request was cancelled or the store is closing.
            settle_job_locked(chosen,
                              chosen->cancelled.load() ? ConversationStoreReadStatus::cancelled
                                                       : ConversationStoreReadStatus::failed,
                              chosen->cancelled.load() ? "the request was cancelled"
                                                       : "the read ended without a verdict");
        }
        cv_work.notify_all();
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Publication
// ---------------------------------------------------------------------------------------------------------------

void ConversationStore::Impl::push_publish_result_locked(ConversationStorePublishResult&& result) {
    publish_done.push_back(std::move(result));
    const size_t keep = size_t(options.max_pending_publishes) * 4 + 4;
    while (publish_done.size() > keep) publish_done.pop_front();
    cv_publish.notify_all();
}

void ConversationStore::Impl::writer_body() {
    std::unique_lock<std::mutex> lock(mutex);
    for (;;) {
        if (!durable.empty()) {
            // Safety work first: an invalidation or a promotion outranks starting another publication.
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
    // track of a charged file.
    const auto note_file = [&job](Where where, const std::string& name, uint64_t charge, const Digest& object,
                                  const std::string& shard) {
        PendingFile file;
        file.where = where;
        file.name = name;
        file.charge = charge;
        file.object = object;
        file.shard = shard;
        job->orphans.push_back(std::move(file));
    };

    Digest bucket{};
    uint64_t epoch_now = 0;
    {
        std::lock_guard<std::mutex> guard(mutex);
        if (!on || closing || faulted) {
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
        // Every head this publication reads from - its predecessor and every reuse donor - is protected from eviction
        // (including this publication's own quota evictions) until it has committed.
        if (request.supersedes) eviction_excluded.insert(request.supersedes);
        for (const auto& entry : request.reuse)
            if (entry.head_id) eviction_excluded.insert(entry.head_id);
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
    // The predecessor (for the replacement predicate) and each reuse donor are read from their own manifests, and a
    // donor is only ever the head the caller named: `supersedes` never stands in for it.
    conversation_store_detail::ManifestMeta source;
    bool have_source = false;
    if (request.supersedes != 0) {
        std::string detail;
        have_source = load_manifest_of(request.supersedes, epoch_now, source, detail) && source.bucket == bucket;
    }
    std::map<uint64_t, conversation_store_detail::ManifestMeta> donors;
    for (const auto& entry : request.reuse) {
        if (!entry.head_id || donors.count(entry.head_id)) continue;
        conversation_store_detail::ManifestMeta donor;
        std::string detail;
        if (load_manifest_of(entry.head_id, epoch_now, donor, detail) && donor.bucket == bucket)
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
            if (!(entry.stream == id)) continue;
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
                                  conversation_store_detail::manifest_workspace_bound(
                                      conversation_store_detail::manifest_encoded_size(source)));
    for (const auto& donor : donors)
        retained_donors = sat_add(retained_donors,
                                  conversation_store_detail::manifest_workspace_bound(
                                      conversation_store_detail::manifest_encoded_size(donor.second)));
    WriterWorkspace actual;
    actual.request_copy = sat_add(sat_mul(uint64_t(request.reuse.size()), sizeof(ConversationStoreReuse)),
                                  kJobFixedOverhead);
    actual.layout = conversation_store_detail::image_layout_workspace(job->shape);
    actual.donor_decoded = retained_donors;
    actual.donor_raw_peak = 0;   // every manifest file this attempt read is already released
    actual.chunks = peak_chunks;
    actual.buffer = chunk_buffer;
    const auto manifest_counts = conversation_store_detail::ManifestShape{
        job->shape.containers, job->shape.records - job->shape.containers, job->shape.kv_layers,
        job->shape.streams, peak_chunks, peak_chunks, job->shape.cuts};
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
                bool known = false;
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
                    known = charged_objects.count(object) != 0 || objects.count(object) != 0;
                    if (known) counters.dedup_chunks++;
                    if (!known) {
                        // A complete file for this content address may exist as an accounted orphan: adopting it
                        // rather than rewriting it keeps its charge honest and keeps it out of the reclaimer's way.
                        known = revive_object_locked(object);
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
                    note_file(Where::objects, sha256_hex(object), charge, object, sha256_hex(object).substr(0, 2));
                    const std::string scratch_name = "ob-" + temp_name();
                    PendingFile scratch;
                    if (!write_object_file(dirs, scratch_name, object, buffer.data(), size_t(size), scratch, why)) {
                        if (!scratch.name.empty()) {
                            scratch.charge = charge;
                            job->orphans.push_back(std::move(scratch));
                        }
                        abandon(ConversationStorePublishStatus::failed, why);
                        return false;
                    }
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
            if (entry.stream == id) intervals.push_back(&entry);
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
                    for (size_t i = first; i <= last; ++i)
                        if (objects.count(source_stream->chunks[i].object) == 0) usable = false;
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
        if (!on || closing || faulted) {
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
        note_file(Where::manifests, manifest_name, charge, Digest{}, std::string());
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
        if (written && ::renameat(dirs.tmp, temp.c_str(), dirs.manifests, manifest_name.c_str()) != 0) {
            why = std::string("cannot publish the manifest: ") + std::strerror(errno);
            written = false;
        }
        if (written) written = fsync_dir(dirs.manifests, "manifests", why);
        if (!written) {
            PendingFile scratch;
            scratch.where = Where::tmp;
            scratch.name = temp;
            scratch.charge = charge;
            job->orphans.push_back(std::move(scratch));
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
        plan.version = catalog_version;
        plan.candidate = catalog;
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
            for (const auto& entry : head_objects) estimate += uint64_t(entry.second.size()) * sizeof(Digest);
            estimate += uint64_t(plan.new_objects.size()) * sizeof(Digest);
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
        if (predecessor != 0 && covered) {
            superseded = predecessor;
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
    self.options = std::move(options);
    self.limits = limits;
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
    Impl& self = *impl_;
    // The destructor deliberately abandons the run intent: an exit that never called close() must not look healthy.
    self.stop.store(true);
    self.cv_work.notify_all();
    self.cv_ready.notify_all();
    self.cv_publish.notify_all();
    self.cv_durable.notify_all();
    if (self.writer.joinable()) self.writer.join();
    for (auto& thread : self.readers)
        if (thread.joinable()) thread.join();
    self.readers.clear();
    std::lock_guard<std::mutex> guard(self.mutex);
    if (self.lock_fd >= 0) {
        ::flock(self.lock_fd, LOCK_UN);
        ::close(self.lock_fd);
        self.lock_fd = -1;
    }
    self.on = false;
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
    std::lock_guard<std::mutex> guard(impl_->mutex);
    // A latched fault is permanent: the engine observed something that must stop reuse, and only a restart (which
    // re-reads the run-intent marker) may trust this domain again.
    impl_->retire_reads_locked(ConversationStoreReadStatus::failed,
                               "the store faulted before this read completed");
    impl_->latch_fault_locked(reason ? reason : "the owner observed a failure");
    impl_->cv_work.notify_all();
}

uint64_t ConversationStore::epoch() const {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    return impl_->epoch;
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
                                       const std::function<void(const ConversationStoreMatch&)>& visit) const {
    std::vector<ConversationImageKey> keys;
    if (images && image_count) keys.assign(images, images + image_count);
    std::vector<ConversationStoreMatch> found;
    {
        std::lock_guard<std::mutex> guard(impl_->mutex);
        impl_->collect_matches(selector, ids, count, keys, found);
    }
    for (const auto& match : found) visit(match);
}

std::vector<ConversationStoreMatch> ConversationStore::matches(const ConversationStoreSelector& selector,
                                                              const std::vector<int32_t>& ids,
                                                              const std::vector<ConversationImageKey>& images) const {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    std::vector<ConversationStoreMatch> found;
    impl_->collect_matches(selector, ids.data(), ids.size(), images, found);
    return found;
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
        if (!self.on || !self.head_manifest_locked(head_id, manifest_id, bytes)) return false;
    }
    std::vector<uint8_t> raw;
    std::string why;
    if (!read_file(self.dirs.manifests, sha256_hex(manifest_id), self.limits.max_manifest_bytes, raw, why))
        return false;
    conversation_store_detail::ManifestMeta meta;
    if (!conversation_store_detail::manifest_decode(raw.data(), raw.size(), self.limits, meta, why)) return false;
    out = conversation_store_detail::manifest_stream_infos(meta);
    return true;
}

bool ConversationStore::demand(const ConversationStoreQuery& query, std::string& why) {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    if (!impl_->on) {
        why = impl_->reason.empty() ? "the store is not enabled" : impl_->reason;
        return false;
    }
    if (impl_->faulted) {
        why = impl_->fault_reason;
        return false;
    }
    return impl_->admit_read_locked(query, true, why);
}

bool ConversationStore::prefetch(const ConversationStoreQuery& query, std::string& why) {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    if (!impl_->on) {
        why = impl_->reason.empty() ? "the store is not enabled" : impl_->reason;
        return false;
    }
    if (impl_->faulted) {
        why = impl_->fault_reason;
        return false;
    }
    return impl_->admit_read_locked(query, false, why);
}

void ConversationStore::cancel(uint64_t request_id, uint64_t generation) {
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
    for (const auto& job : self.jobs) {
        if (job->query.request_id != request_id || job->query.generation != generation) continue;
        self.retire_job_locked(job, ConversationStoreReadStatus::cancelled, "the caller cancelled this read");
        self.cv_work.notify_all();
        return;
    }
}

bool ConversationStore::take_ready(ConversationStoreRead& out) {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    return impl_->deliver_locked(out, 0, 0, false);
}

bool ConversationStore::take_ready(uint64_t request_id, uint64_t generation, ConversationStoreRead& out) {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    return impl_->deliver_locked(out, request_id, generation, true);
}

bool ConversationStore::wait_ready(ConversationStoreRead& out, uint64_t timeout_ms) {
    std::unique_lock<std::mutex> lock(impl_->mutex);
    if (impl_->deliver_locked(out, 0, 0, false)) return true;
    if (timeout_ms == 0) return false;
    impl_->cv_ready.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                             [&] { return !impl_->ready.empty() || impl_->closed; });
    return impl_->deliver_locked(out, 0, 0, false);
}

bool ConversationStore::wait_ready(uint64_t request_id, uint64_t generation, ConversationStoreRead& out,
                                   uint64_t timeout_ms) {
    Impl& self = *impl_;
    std::unique_lock<std::mutex> lock(self.mutex);
    const auto settled = [&]() {
        for (const auto& job : self.jobs)
            if (job->query.request_id == request_id && job->query.generation == generation)
                return job->handed ||
                       (job->status != ConversationStoreReadStatus::none && !job->active);
        return true;   // an unknown request has settled (as nothing)
    };
    if (!settled() && timeout_ms)
        self.cv_ready.wait_for(lock, std::chrono::milliseconds(timeout_ms), settled);
    if (!settled()) return false;
    return self.deliver_locked(out, request_id, generation, true);
}

void ConversationStore::release_read(uint64_t request_id, uint64_t generation) {
    Impl& self = *impl_;
    std::lock_guard<std::mutex> guard(self.mutex);
    for (const auto& job : self.jobs) {
        if (job->query.request_id != request_id || job->query.generation != generation) continue;
        if (!job->released) {
            job->released = true;
            self.counters.reads_released++;
        }
        job->cancelled = true;                       // a released request is never delivered
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
            self.erase_job_locked(job);
        } else {
            self.free_job_prep_locked(job);
        }
        self.cv_work.notify_all();
        return;
    }
}

ConversationStoreReadStatus ConversationStore::request_status(uint64_t request_id, uint64_t generation) const {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    for (const auto& job : impl_->jobs)
        if (job->query.request_id == request_id && job->query.generation == generation) return job->status;
    return ConversationStoreReadStatus::none;
}

void ConversationStore::confirmed_restore(uint64_t head_id) {
    Impl& self = *impl_;
    std::lock_guard<std::mutex> guard(self.mutex);
    if (!self.on || self.faulted || !self.initialized) return;
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
    // The durable write is the storage thread's job: this call performs no storage wait, and a failure is persisted
    // by the next commit (a promotion is a policy hint, never authority).
    auto work = std::make_shared<Impl::DurableWork>();
    work->plan.kind = Impl::CommitPlan::Kind::promote;
    self.durable.push_back(std::move(work));
    self.cv_work.notify_all();
}

bool ConversationStore::publish(std::shared_ptr<const SavedConversation> image, const ConversationStorePublish& request,
                                std::string& why) {
    Impl& self = *impl_;
    std::lock_guard<std::mutex> guard(self.mutex);
    if (!self.on || self.closed) {
        why = self.reason.empty() ? "the store is not enabled" : self.reason;
        return false;
    }
    if (self.faulted) {
        why = self.fault_reason;
        return false;
    }
    if (self.closing) {
        why = "the store is closing";
        return false;
    }
    if (!image) {
        why = "no image was supplied";
        return false;
    }
    if (request.epoch != 0 && request.epoch != self.epoch) {
        self.counters.refusals_stale++;
        why = "the image was captured in a previous execution-domain epoch";
        return false;
    }
    const uint64_t image_bytes = image->bytes();
    // The image is not the workspace.  Count what this publication will really allocate - the copied request, the
    // layout skeleton, its per-record prefix chains and the parent/donor manifest decode - before one byte of it is
    // allocated, and count the image itself separately: a ticket whose workspace cannot fit settles as refused_limits
    // instead of overrunning either the allowance or the preparation ceiling.
    conversation_store_detail::ImageShape shape;
    if (!conversation_store_detail::image_shape(*image, self.limits, shape, why)) {
        self.counters.refused_limits++;
        why = "the publication image is not representable: " + why;
        return false;
    }
    const WriterWorkspace workspace = self.writer_workspace_locked(request, shape);
    const uint64_t needed = workspace.peak();
    const uint64_t required = sat_add(image_bytes, needed);
    // The preparation sub-cap covers the index and the writer alike, so the whole counted peak is checked against
    // it: the writer's workspace is preparing an index entry, not a payload.
    if (needed > self.options.metadata_bytes ||
        self.metadata + self.prep > self.options.metadata_bytes - needed) {
        self.counters.refused_limits++;
        why = "the publication's writer workspace does not fit the store's preparation ceiling";
        return false;
    }
    // A transferred lease: the caller's own cache still holds exactly this entry's bytes outside the store's
    // allowance, so this one publication is fitted against the allowance *plus* that credit.  The credit is clamped
    // to the image's own size, is granted to no read, and widens nothing else: the store still charges the full
    // image and the whole counted workspace into ram_held_bytes().
    const uint64_t credit = std::min(request.external_lease_bytes, image_bytes);
    const uint64_t envelope = sat_add(self.ram_budget, credit);
    if (self.ram_budget == 0 || required > envelope) {
        // Speculative read resources are revocable, and an outgoing capture is required work: outstanding prefetch
        // gives way to it before the store declines to persist the image at all.
        self.revoke_speculation_locked(false);
    }
    if (self.ram_budget == 0 || required > envelope || self.resident_locked() > envelope - required) {
        self.counters.refused_limits++;
        why = credit
                  ? "the publication image and its writer workspace do not fit the published RAM allowance plus the "
                    "lease the caller transferred"
                  : "the publication image and its writer workspace do not fit the published RAM allowance";
        return false;
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
            return false;
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
    self.counters.peak_bytes = std::max(self.counters.peak_bytes, self.ram_held);
    self.publishing = job;
    self.cv_work.notify_all();
    self.cv_publish.notify_all();
    return true;
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
                                   [&] { return !impl_->publish_done.empty() || impl_->closed; });
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
    return impl_->ram_held + impl_->metadata + impl_->prep;
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
    const uint64_t held = impl_->ram_held + impl_->metadata + impl_->prep;
    return impl_->ram_budget > held ? impl_->ram_budget - held : 0;
}

void ConversationStore::suspend_read_admission() {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    impl_->admission_suspended = true;
}

uint64_t ConversationStore::publish_read_allowance(uint64_t bytes) {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    // Absolute, and never retroactive: a lower bound admits no new work but revokes nothing already admitted.
    impl_->ram_budget = bytes;
    // The caller has now published a complete absolute envelope, including any
    // lease it released after publish(). Do not add that transfer a second time.
    if (impl_->publishing) impl_->publishing->external_credit_bytes = 0;
    impl_->admission_suspended = false;
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
        std::lock_guard<std::mutex> guard(self.mutex);
        if (!self.on || !self.initialized) return false;
        if (self.faulted) return false;
        self.epoch += 1;
        if (self.domain_index < self.catalog.domains.size())
            self.catalog.domains[self.domain_index].epoch = self.epoch;
        self.counters.epoch_advances++;
        // Every unhanded completion of the previous epoch is retired here, never relabelled, and nothing of the
        // previous epoch is delivered again.
        self.retire_reads_locked(ConversationStoreReadStatus::stale,
                                 "the execution domain was invalidated before this read completed");
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

void ConversationStore::close() {
    Impl& self = *impl_;
    {
        std::lock_guard<std::mutex> guard(self.mutex);
        if (self.closed) return;
        self.closed = true;
        if (!self.initialized || !self.lock_owned) {
            // This instance never owned the store's authority (a rejected second owner, or a root that could not be
            // opened): it must not write the catalog, retire anything, or touch the run-intent marker.
            self.on = false;
            if (self.lock_fd >= 0) {
                ::flock(self.lock_fd, LOCK_UN);
                ::close(self.lock_fd);
                self.lock_fd = -1;
            }
            self.lock_owned = false;
            return;
        }
        self.closing = true;
        self.admission_suspended = true;
        self.retire_reads_locked(ConversationStoreReadStatus::disabled, "the store closed before this read completed");
        self.queue.clear();
        if (self.publishing && !self.publishing->started) {
            // A publication the writer never began is settled here, with its alias and its charge released.
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
    self.stop.store(true);
    self.cv_work.notify_all();
    self.cv_ready.notify_all();
    self.cv_publish.notify_all();
    self.cv_durable.notify_all();
    if (self.writer.joinable()) self.writer.join();
    for (auto& thread : self.readers)
        if (thread.joinable()) thread.join();
    self.readers.clear();

    bool commit_final = false;
    {
        std::lock_guard<std::mutex> guard(self.mutex);
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
        // A faulted store never presents itself as a clean close: the marker stays armed so the next start distrusts
        // this domain, and no catalog write is attempted on authority already known to be untrustworthy.
        commit_final = self.initialized && self.lock_owned && !self.faulted;
    }
    if (commit_final) {
        Impl::CommitPlan plan;
        plan.kind = Impl::CommitPlan::Kind::promote;   // commit the live catalog durably, first
        std::string why;
        const CommitOutcome outcome = self.commit_plan(plan, why);
        if (outcome == CommitOutcome::durable_applied) {
            // Only now may this run be called clean: the catalog is durable, so the marker can go.
            std::string intent_why;
            const bool cleared = self.write_intent(false, intent_why);
            std::lock_guard<std::mutex> guard(self.mutex);
            self.intent_active = !cleared;
            // A close that could not prove the marker gone is not a clean close, and the reason is reported as the
            // fault it is rather than as a successful drain.
            if (!cleared) self.latch_fault_locked("the run-intent marker could not be cleared: " + intent_why);
        } else {
            // The final commit either changed nothing durable (its own copy could not be admitted, so the previously
            // committed catalog is still the authority) or its durability is uncertain.  Neither is a clean close:
            // the marker stays armed so the next start distrusts this domain.  Uncertainty already latched a fault;
            // an ordinary refusal is not a fault, it just is not a promise.
            std::lock_guard<std::mutex> guard(self.mutex);
            self.intent_active = true;
            if (outcome == CommitOutcome::uncertain)
                self.latch_fault_locked(why.empty() ? "the final catalog commit is not confirmed" : why);
        }
    }
    {
        std::lock_guard<std::mutex> guard(self.mutex);
        if (!self.faulted) {
            // No reader exists any more, so retired files can be removed and credited regardless of past leases.
            for (auto& file : self.pending_files) file.head_id = 0;
        }
    }
    if (!self.faulted) self.reclaim_pending();
    {
        std::lock_guard<std::mutex> guard(self.mutex);
        if (self.lock_fd >= 0) {
            ::flock(self.lock_fd, LOCK_UN);
            ::close(self.lock_fd);
            self.lock_fd = -1;
        }
        self.lock_owned = false;
        self.on = false;
    }
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
    uint64_t shared = 0;
    for (const auto& object : impl_->objects) shared += object.second.refs > 1 ? 1 : 0;
    out.shared_objects = shared;
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
            " bytes estimated, preparation " + std::to_string(impl_->prep) + " bytes\n";
    text += "ram: held " + std::to_string(impl_->ram_held + impl_->metadata + impl_->prep) + ", leased " +
            std::to_string(impl_->ram_lease) + ", budget " + std::to_string(impl_->ram_budget) +
            (impl_->admission_suspended ? " (admission suspended)" : "") + "\n";
    text += "reads: queued " + std::to_string(impl_->queue.size()) + ", incoming " +
            std::to_string(impl_->incoming) + " of " + std::to_string(impl_->options.max_reads) + ", ready " +
            std::to_string(impl_->ready.size()) + ", held descriptors " + std::to_string(impl_->jobs.size()) + "\n";
    text += std::string("writer: ") + (impl_->publishing ? "busy" : "idle") + ", durable queue " +
            std::to_string(impl_->durable.size()) + "\n";
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

}  // namespace strata::core
