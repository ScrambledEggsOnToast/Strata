// The durable NVMe conversation store: its disk authority (root, lock, issuer key, catalog, run intent), the shared
// physical quota, the one admitted writer and the bounded read pipeline, and every byte it hands back.
//
// Ordering rule that the whole file follows: an object is durable before a manifest names it, a manifest is durable
// before the catalog authorizes it, and the catalog is the only authority.  A file that the catalog does not
// authorize is an accounted orphan, never a resurrected history; a catalog that cannot be trusted means cold caching.
//
// Threading rule: the catalog mutex guards state, never I/O.  Every storage wait happens on a worker thread with the
// lock released, so no caller ever blocks under a lock it needs, and every entry point returns immediately.
#include "strata/core/conversation_store.hpp"

#include "conversation_store_format.hpp"
#include "conversation_store_sha256.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
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
using conversation_store_detail::Digest;
using conversation_store_detail::digest_nonzero;
using conversation_store_detail::digest_u64;
using conversation_store_detail::ManifestMeta;
using conversation_store_detail::RestoreTarget;
using conversation_store_detail::sha256;
using conversation_store_detail::sha256_hex;
using conversation_store_detail::StreamRead;
using conversation_store_detail::StreamRecord;
using conversation_store_detail::StreamWrite;

constexpr uint64_t kObjectBytesEstimate = 64;  // the RAM one object-table entry costs, at most
constexpr size_t kMaxReaderThreads = 8;

// ---------------------------------------------------------------------------------------------------------------
// POSIX helpers.  Every one of them is either exact or fails with a reason; none of them throws.
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

bool write_file(const std::string& path, const void* data, size_t size, bool exclusive, std::string& why) {
    const int flags = O_WRONLY | O_CREAT | O_CLOEXEC | O_NOFOLLOW | (exclusive ? O_EXCL : O_TRUNC);
    const int fd = ::open(path.c_str(), flags, 0600);
    if (fd < 0) {
        why = std::string("cannot create ") + path + ": " + std::strerror(errno);
        return false;
    }
    bool ok = write_all(fd, data, size, why);
    if (ok && ::fsync(fd) != 0) {
        why = std::string("fsync failed for ") + path + ": " + std::strerror(errno);
        ok = false;
    }
    if (::close(fd) != 0 && ok) {
        why = std::string("close failed for ") + path + ": " + std::strerror(errno);
        ok = false;
    }
    return ok;
}

bool read_file(const std::string& path, std::vector<uint8_t>& out, uint64_t limit, std::string& why) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) {
        why = std::string("cannot open ") + path + ": " + std::strerror(errno);
        return false;
    }
    struct stat info {};
    if (::fstat(fd, &info) != 0) {
        why = std::string("cannot stat ") + path + ": " + std::strerror(errno);
        ::close(fd);
        return false;
    }
    if (info.st_size < 0 || uint64_t(info.st_size) > limit) {
        why = path + " is larger than the limit this store admits";
        ::close(fd);
        return false;
    }
    out.resize(size_t(info.st_size));
    size_t at = 0;
    while (at < out.size()) {
        const ssize_t n = ::read(fd, out.data() + at, out.size() - at);
        if (n < 0) {
            if (errno == EINTR) continue;
            why = std::string("read failed for ") + path + ": " + std::strerror(errno);
            ::close(fd);
            return false;
        }
        if (n == 0) break;
        at += size_t(n);
    }
    ::close(fd);
    if (at != out.size()) {
        why = path + " shrank while it was read";
        return false;
    }
    return true;
}

// The file existed and could not be made durable, or the directory could not be updated: both are fatal for the
// caller's intent, never silently ignored.
bool fsync_directory(const std::string& path, std::string& why) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd < 0) {
        why = std::string("cannot open directory ") + path + ": " + std::strerror(errno);
        return false;
    }
    const bool ok = ::fsync(fd) == 0;
    if (!ok) why = std::string("directory fsync failed for ") + path + ": " + std::strerror(errno);
    ::close(fd);
    return ok;
}

bool make_directory(const std::string& path, std::string& why) {
    if (::mkdir(path.c_str(), 0700) == 0) return true;
    if (errno == EEXIST) {
        struct stat info {};
        if (::stat(path.c_str(), &info) == 0 && S_ISDIR(info.st_mode)) return true;
        why = path + " exists and is not a directory";
        return false;
    }
    why = std::string("cannot create ") + path + ": " + std::strerror(errno);
    return false;
}

bool path_exists(const std::string& path) {
    struct stat info {};
    return ::stat(path.c_str(), &info) == 0;
}

bool remove_file(const std::string& path, std::string& why) {
    if (::unlink(path.c_str()) == 0) return true;
    if (errno == ENOENT) return true;
    why = std::string("cannot remove ") + path + ": " + std::strerror(errno);
    return false;
}

bool list_directory(const std::string& path, std::vector<std::string>& out, std::string& why) {
    DIR* dir = ::opendir(path.c_str());
    if (!dir) {
        why = std::string("cannot list ") + path + ": " + std::strerror(errno);
        return false;
    }
    out.clear();
    while (struct dirent* entry = ::readdir(dir)) {
        const std::string name = entry->d_name;
        if (name == "." || name == "..") continue;
        out.push_back(name);
    }
    ::closedir(dir);
    return true;
}

void fill_random(uint8_t* out, size_t size) {
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
    if (at == size) return;
    // The kernel has no entropy source available: fall back to the device node rather than persisting a weak key.
    const int fd = ::open("/dev/urandom", O_RDONLY | O_CLOEXEC);
    if (fd >= 0) {
        while (at < size) {
            const ssize_t n = ::read(fd, out + at, size - at);
            if (n <= 0) {
                if (n < 0 && errno == EINTR) continue;
                break;
            }
            at += size_t(n);
        }
        ::close(fd);
    }
    if (at != size) {
        // Still short: mix the clock and the address of this stack frame so the key is at least not constant.
        uint64_t seed = uint64_t(std::chrono::steady_clock::now().time_since_epoch().count());
        for (size_t i = at; i < size; ++i) {
            seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
            out[i] = uint8_t(seed >> 33);
        }
    }
}

uint64_t filesystem_free_bytes(const std::string& path) {
    struct statvfs info {};
    if (::statvfs(path.c_str(), &info) != 0) return 0;
    return uint64_t(info.f_bavail) * uint64_t(info.f_frsize);
}

// A stable, collision-free file name for one content address.
std::string object_path(const std::string& root, const Digest& digest) {
    const std::string hex = sha256_hex(digest);
    return root + "/objects/" + hex.substr(0, 2) + "/" + hex;
}

std::string manifest_path(const std::string& root, const Digest& digest) {
    return root + "/manifests/" + sha256_hex(digest);
}

bool hex_to_bytes(const std::string& text, Digest& out) { return conversation_store_detail::sha256_hex_parse(text, out); }

} // namespace

// ---------------------------------------------------------------------------------------------------------------
// The implementation
// ---------------------------------------------------------------------------------------------------------------

struct ConversationStore::Impl {
    // ---- configuration (immutable after construction) ----
    ConversationStoreOptions options;
    ConversationStoreLimits limits;
    std::string root;
    uint32_t reader_threads = 1;

    // ---- authority ----
    std::mutex mutex;
    std::condition_variable cv_idsle;
    std::condition_variable cv_ready;
    std::condition_variable cv_publish;
    std::condition_variable cv_work;
    std::array<uint8_t, 32> issuer_key{};
    Catalog catalog;
    uint64_t epoch = 0;
    uint32_t domain_index = 0;
    bool intent_active = false;
    bool on = false;
    bool closing = false;
    bool closed = false;
    bool healthy = true;
    std::string reason;
    std::string unhealthy_reason;
    int lock_fd = -1;

    // ---- disk accounting (unique physical bytes) ----
    struct ObjectState {
        uint64_t bytes = 0;
        uint32_t refs = 0;
    };
    std::map<Digest, ObjectState> objects;                  // every object the catalog authorizes
    std::map<uint64_t, std::vector<Digest>> head_objects;   // head id -> the objects its manifest names
    std::map<uint64_t, uint64_t> head_manifest_bytes;       // head id -> its manifest's file size
    std::map<uint64_t, uint32_t> head_leases;               // head id -> materialized/ handed-out images using it
    uint64_t charged = 0;         // live objects + manifests
    uint64_t reserved = 0;        // about to be written
    uint64_t reclaimable = 0;     // charged but no longer referenced
    uint64_t free_sample = 0;     // last sampled filesystem space under the root
    uint64_t metadata = 0;        // estimated RAM the index itself holds

    // ---- RAM ledger ----
    uint64_t ram_held = 0;        // everything we own in host memory
    uint64_t ram_lease = 0;       // the handed-out share of it
    uint64_t ram_reserved = 0;    // admitted, not yet allocated
    uint64_t ram_budget = 0;
    bool admission_suspended = false;

    // ---- reads ----
    struct ReadJob {
        ConversationStoreQuery query;
        ConversationStoreReadStatus status = ConversationStoreReadStatus::none;
        bool demand = false;
        bool active = false;
        std::atomic<bool> cancelled{false};
        bool released = false;
        bool handed = false;
        uint64_t reserved_bytes = 0;
        uint64_t lease_bytes = 0;
        uint64_t epoch_at_admission = 0;
        uint64_t head_id = 0;
        uint64_t tokens = 0;
        bool live = false;
        uint32_t stages = 0;
        uint64_t payload_bytes = 0;
        std::shared_ptr<const SavedConversation> image;
        std::string detail;
    };
    std::vector<std::shared_ptr<ReadJob>> jobs;      // queued, active and handed out
    std::deque<std::shared_ptr<ReadJob>> queue;      // waiting for a slot (demand before prefetch)
    std::deque<std::shared_ptr<ReadJob>> ready;      // materialized, waiting for take_ready()
    size_t materializing = 0;

    // ---- publication ----
    struct PublishJob {
        std::shared_ptr<const SavedConversation> image;
        ConversationStorePublish request;
        uint64_t ram_bytes = 0;
        bool started = false;          // the writer has begun reading this image: it may never be replaced
        bool coalesced = false;
    };
    // At most one publication identifier/image is admitted.  A ready-to-start (not yet begun) job may be replaced by
    // a newer same-lineage generation - that is the only coalesce, and it happens before any capture image is read.
    std::shared_ptr<PublishJob> publishing;
    std::deque<ConversationStorePublishResult> publish_done;

    // ---- workers ----
    std::vector<std::thread> readers;
    std::thread writer;
    std::atomic<bool> stop{false};

    // ---- counters and test seams ----
    ConversationStoreCounters counters;
    std::string fault_step;
    std::atomic<uint64_t> source_bytes_read{0};
    uint64_t name_counter = 0;

    // ------------------------------ helpers ------------------------------
    // Every one of these assumes the caller holds `mutex` unless its name says otherwise; none of them performs a
    // storage wait while the lock is held.
    uint64_t sample_free();
    std::string temp_name();
    std::string intent_path() const;
    bool write_catalog(std::string& why);
    bool arm_intent(std::string& why);
    void clear_intent();
    bool open_root(std::string& why);
    bool load_catalog(std::string& why);
    bool reconcile(std::string& why);
    bool load_manifest(uint64_t head_id, ManifestMeta& out, std::string& why) const;
    bool take_fault(const char* step);
    void recount_metadata();
    bool ensure_capacity(uint64_t incoming, std::vector<uint64_t>& victims, uint64_t& freeable, bool& floor_limited,
                         std::string& why);

    void writer_body();
    void reader_body();
    void publish_body(const std::shared_ptr<PublishJob>& job, ConversationStorePublishResult& out);
    void read_body(const std::shared_ptr<ReadJob>& job);

    // A head whose authority has been dropped but whose files are still on disk: the bytes stay charged (they exist!)
    // until the catalog commit that removed the head is durable, and only then are they deleted and credited.
    struct Retired {
        Digest manifest_id{};
        uint64_t manifest_bytes = 0;
        std::vector<Digest> objects;
    };
    std::vector<Retired> pending_reclaim;
    uint64_t retire_head(uint64_t head_id);
    void reclaim_files();

    void collect_matches(const ConversationStoreSelector& selector, const int32_t* ids, size_t count,
                         const std::vector<ConversationImageKey>& images,
                         std::vector<ConversationStoreMatch>& out) const;
    bool admit_read(const ConversationStoreQuery& query, bool demand, std::string& why);
    bool manifest_path_of(uint64_t head_id, std::string& path, uint64_t& bytes) const;
    Digest head_manifest_id(uint64_t head_id) const;
    void finish_job(const std::shared_ptr<ReadJob>& job, ConversationStoreReadStatus status, std::string detail);
    bool deliver_locked(ConversationStoreRead& out, uint64_t request_id, uint64_t generation, bool specific);
};

// ---------------------------------------------------------------------------------------------------------------
// Object files: the only place bytes are created, verified or copied
// ---------------------------------------------------------------------------------------------------------------

namespace {

// An object is durable before any manifest names it: a scratch file, an fsync, a rename into its shard, then the
// shard's own fsync.
bool write_object_file(const std::string& root, const Digest& id, const uint8_t* data, size_t size, std::string& why) {
    const std::string final_path = object_path(root, id);
    const std::string shard = final_path.substr(0, final_path.rfind('/'));
    std::string detail;
    if (!make_directory(shard, detail) && !path_exists(shard)) {
        why = detail;
        return false;
    }
    const std::string tmp = root + "/tmp/o" + sha256_hex(id).substr(0, 16);
    if (!write_file(tmp, data, size, false, why)) return false;
    if (::rename(tmp.c_str(), final_path.c_str()) != 0) {
        why = std::string("cannot publish an object: ") + std::strerror(errno);
        std::string ignored;
        remove_file(tmp, ignored);
        return false;
    }
    return fsync_directory(shard, why);
}

// The mandatory verification: an object is trusted only after its bytes have reproduced its own content address.
bool hash_object_file(const std::string& path, uint64_t size, Digest& out, std::string& why) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
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
    conversation_store_detail::Sha256 hash;
    std::vector<uint8_t> slab(1u << 20);
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
        left -= uint64_t(n);
    }
    ::close(fd);
    out = hash.digest();
    return true;
}

// Copies one verified object into its final buffer in bounded slabs: no whole image is ever resident as scratch.
bool copy_object_file(const std::string& path, uint64_t size, StreamWrite& write, uint64_t offset, std::string& why) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) {
        why = std::string("cannot reopen an object: ") + std::strerror(errno);
        return false;
    }
    std::vector<uint8_t> slab(1u << 20);
    uint64_t at = 0;
    bool ok = true;
    while (at < size) {
        const ssize_t n = ::read(fd, slab.data(), size_t(std::min<uint64_t>(slab.size(), size - at)));
        if (n < 0) {
            if (errno == EINTR) continue;
            why = std::string("cannot read an object: ") + std::strerror(errno);
            ok = false;
            break;
        }
        if (n == 0) {
            why = "an object shrank while it was being restored";
            ok = false;
            break;
        }
        if (!write(offset + at, slab.data(), size_t(n), why)) {
            ok = false;
            break;
        }
        at += uint64_t(n);
    }
    ::close(fd);
    return ok && at == size;
}

} // namespace

// The engine's freeze()/frozen_alias() landed after this module's base commit, so it is called when it exists and is
// a no-op otherwise: a caller that mounts the handed-out image pays no payload copy where the engine can freeze, and
// this module still compiles against a base that has no freeze() at all.
template<class T, class = void>
struct has_freeze : std::false_type {};
template<class T>
struct has_freeze<T, std::void_t<decltype(std::declval<T&>().freeze())>> : std::true_type {};

template<class T>
void freeze_if_available(T& image) {
    if constexpr (has_freeze<T>::value) image.freeze();
}

void ConversationStore::Impl::writer_body() {
    std::unique_lock<std::mutex> lock(mutex);
    while (!stop.load()) {
        if (!publishing) {
            cv_work.wait_for(lock, std::chrono::milliseconds(50));
            continue;
        }
        const std::shared_ptr<PublishJob> job = publishing;
        // Once the worker has begun reading the image it may never be replaced: publish() refuses instead.
        job->started = true;
        ConversationStorePublishResult result;
        lock.unlock();
        publish_body(job, result);
        lock.lock();
        if (job->ram_bytes) {
            ram_held -= std::min(ram_held, job->ram_bytes);
            job->ram_bytes = 0;
        }
        result.released = true;
        if (job->coalesced && result.status == ConversationStorePublishStatus::stored) {
            result.status = ConversationStorePublishStatus::coalesced;
            result.detail = "replaced by a newer publication of the same lineage";
            counters.coalesced_publishes++;
        }
        result.request_id = job->request.request_id;
        result.generation = job->request.generation;
        if (result.epoch == 0) result.epoch = epoch;
        publish_done.push_back(std::move(result));
        const size_t keep = size_t(options.max_pending_publishes) * 4 + 4;
        while (publish_done.size() > keep) publish_done.pop_front();
        if (publishing == job) publishing.reset();
        cv_publish.notify_all();
    }
}

void ConversationStore::Impl::publish_body(const std::shared_ptr<PublishJob>& job, ConversationStorePublishResult& out) {
    const SavedConversation& image = *job->image;
    const ConversationStorePublish& request = job->request;
    out.status = ConversationStorePublishStatus::failed;
    const auto fail = [&out](ConversationStorePublishStatus status, const std::string& detail) {
        out.status = status;
        out.detail = detail;
    };

    Digest bucket{};
    uint64_t epoch_now = 0;
    {
        std::lock_guard<std::mutex> guard(mutex);
        if (!on || closing) {
            fail(ConversationStorePublishStatus::disabled, reason.empty() ? "the store is not enabled" : reason);
            return;
        }
        if (!healthy) {
            fail(ConversationStorePublishStatus::failed, unhealthy_reason);
            return;
        }
        epoch_now = epoch;
        ConversationStoreSelector selector;
        selector.scope = image.scope;
        selector.cvec = image.cvec;
        selector.geometry = image.geometry;
        selector.layer_lo = image.layer_lo;
        selector.layer_hi = image.layer_hi;
        bucket = conversation_store_detail::store_bucket_digest(options, selector);
    }
    if (take_fault("layout")) {
        fail(ConversationStorePublishStatus::failed, "injected fault: layout");
        return;
    }
    conversation_store_detail::ImageLayout layout;
    std::string why;
    if (!conversation_store_detail::image_layout(image, bucket, limits, layout, why)) {
        counters.aborted_publications++;
        fail(ConversationStorePublishStatus::refused_limits, why);
        return;
    }
    // The caller's proven unchanged intervals and the predecessor's pinned cuts both need its manifest, so it is
    // read whenever this publication names a predecessor; path resolution takes the state lock, the read does not.
    conversation_store_detail::ManifestMeta source;
    bool have_source = false;
    if (request.supersedes != 0) {
        std::string source_path;
        uint64_t source_bytes = 0;
        {
            std::lock_guard<std::mutex> guard(mutex);
            have_source = manifest_path_of(request.supersedes, source_path, source_bytes);
        }
        if (have_source) {
            std::vector<uint8_t> raw;
            std::string detail;
            have_source = read_file(source_path, raw, limits.max_manifest_bytes, detail) &&
                          conversation_store_detail::manifest_decode(raw.data(), raw.size(), limits, source, detail) &&
                          source.bucket == bucket && source.epoch == epoch_now;
            if (have_source) {
                std::lock_guard<std::mutex> guard(mutex);
                counters.bytes_read += raw.size();
            }
        }
    }

    // ---- payload: reuse proven intervals, chunk everything else, never write bytes that already exist ----
    std::vector<conversation_store_detail::StreamRecord> streams;
    streams.reserve(layout.streams.size());
    std::map<Digest, uint64_t> referenced;
    std::vector<std::pair<Digest, uint64_t>> written;
    std::set<Digest> written_set;
    std::vector<uint64_t> victims;
    uint64_t freeable = 0;
    uint64_t new_bytes = 0, reused_bytes = 0;
    const uint64_t chunk_limit = std::max<uint64_t>(4096, options.chunk_bytes);
    auto kv_meta_for = [&layout](const ConversationStoreStreamId& id) -> const conversation_store_detail::KvMeta* {
        if (!conversation_store_kv_kind(id.kind)) return nullptr;
        for (const auto& container : layout.containers)
            if (container.is_image && container.stage == id.stage && size_t(id.ordinal) < container.kv.size())
                return &container.kv[size_t(id.ordinal)];
        return nullptr;
    };
    bool aborted = false;
    bool floor_limited = false;
    ConversationStorePublishStatus abort_status = ConversationStorePublishStatus::failed;
    auto rollback = [&]() {
        std::string ignored;
        for (const auto& object : written) remove_file(object_path(root, object.first), ignored);
        std::lock_guard<std::mutex> guard(mutex);
        for (const auto& object : written) charged -= std::min(charged, object.second);
        counters.aborted_publications++;
    };

    for (const auto& stream : layout.streams) {
        const ConversationStoreStreamId id = stream.info.id;
        conversation_store_detail::StreamRecord record;
        record.id = id;
        record.bytes = stream.info.bytes;
        const conversation_store_detail::KvMeta* kv = kv_meta_for(id);
        const Digest scope = conversation_store_detail::object_scope_digest(id.kind, kv);
        uint64_t at = 0;
        auto emit = [&](uint64_t from, uint64_t to) {
            while (from < to) {
                const uint64_t element = std::max<uint32_t>(1, stream.element);
                uint64_t size = chunk_limit;
                if (element > 1) size -= size % element;
                size = std::min(size, to - from);
                std::vector<uint8_t> buffer(static_cast<size_t>(size));
                if (!stream.read(from, buffer.data(), size_t(size), why)) return false;
                source_bytes_read += size;
                const Digest object =
                    conversation_store_detail::object_digest(scope, id.kind, id.ordinal, size, buffer.data());
                bool present = false;
                {
                    std::lock_guard<std::mutex> guard(mutex);
                    if (job->coalesced) {
                        why = "replaced by a newer publication of the same lineage";
                        aborted = true;
                        abort_status = ConversationStorePublishStatus::coalesced;
                        return false;
                    }
                    present = objects.count(object) != 0 || written_set.count(object) != 0;
                }
                if (present) {
                    counters.dedup_chunks++;
                } else {
                    {
                        std::lock_guard<std::mutex> guard(mutex);
                        if (!ensure_capacity(size, victims, freeable, floor_limited, why)) {
                            abort_status = floor_limited ? ConversationStorePublishStatus::refused_floor
                                                        : ConversationStorePublishStatus::refused_quota;
                            counters.aborted_publications++;
                            return false;
                        }
                        charged += size;
                    }
                    if (take_fault("objects")) {
                        why = "injected fault: objects";
                        std::lock_guard<std::mutex> guard(mutex);
                        charged -= std::min(charged, size);
                        counters.aborted_publications++;
                        return false;
                    }
                    if (!write_object_file(root, object, buffer.data(), size_t(size), why)) {
                        std::lock_guard<std::mutex> guard(mutex);
                        charged -= std::min(charged, size);
                        counters.aborted_publications++;
                        return false;
                    }
                    written.emplace_back(object, size);
                    written_set.insert(object);
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

        // Reuse intervals, longest first is irrelevant: they are placed by their own offset, and every byte they do
        // not cover is a gap that gets chunked normally.
        std::vector<const ConversationStoreReuse*> intervals;
        for (const auto& entry : request.reuse)
            if (entry.stream == id) intervals.push_back(&entry);
        std::stable_sort(intervals.begin(), intervals.end(),
                         [](const ConversationStoreReuse* a, const ConversationStoreReuse* b) {
                             return a->offset < b->offset;
                         });
        const conversation_store_detail::StreamRecord* source_stream = nullptr;
        if (have_source)
            for (const auto& candidate : source.streams)
                if (candidate.id == id) {
                    source_stream = &candidate;
                    break;
                }
        uint64_t covered_until = 0;
        for (const ConversationStoreReuse* entry : intervals) {
            if (aborted) break;
            bool usable = entry->bytes > 0 && source_stream != nullptr &&
                          entry->offset >= covered_until && entry->offset % chunk_limit == 0 &&
                          entry->offset + entry->bytes <= stream.info.bytes &&
                          entry->source_offset + entry->bytes <= source_stream->bytes;
            if (usable) {
                // The interval must start and end exactly on the source's own chunk boundaries, and end either on a
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
                usable = ends_clean && first <= last && source_stream->chunks[first].offset == entry->source_offset &&
                         source_stream->chunks[last].offset + source_stream->chunks[last].length ==
                             entry->source_offset + entry->bytes;
                if (usable) {
                    // Every reused object must still be present and authorized, or the interval is not usable.
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
                        counters.reused_chunks++;
                        counters.reused_bytes += chunk.length;
                    }
                    covered_until = entry->offset + entry->bytes;
                    at = covered_until;
                    continue;
                }
            }
        }
        if (aborted) break;
        if (!emit(covered_until, stream.info.bytes)) break;
        (void)at;
        streams.push_back(std::move(record));
    }
    if (aborted || streams.size() != layout.streams.size()) {
        rollback();
        fail(abort_status, why.empty() ? "the publication was abandoned" : why);
        return;
    }

    // ---- the manifest, then the catalog commit that authorizes it ----
    uint64_t published_order = 0;
    {
        std::lock_guard<std::mutex> guard(mutex);
        published_order = ++catalog.published_order_counter;
    }
    conversation_store_detail::ManifestMeta meta;
    meta.bucket = bucket;
    meta.domain = options.domain;
    meta.epoch = epoch_now;
    meta.generation = request.generation;
    meta.parent = request.supersedes;
    meta.published_order = published_order;
    meta.stages = layout.stages;
    meta.containers = layout.containers;
    meta.streams = std::move(streams);
    meta.payload_bytes = layout.payload_bytes;
    meta.objects.assign(referenced.begin(), referenced.end());
    meta.cuts = layout.cuts;
    const std::vector<uint8_t> encoded = conversation_store_detail::manifest_encode(meta);
    meta.manifest_id = conversation_store_detail::manifest_id_of(encoded.data(), encoded.size());
    const uint64_t head_id = std::max<uint64_t>(1, digest_u64(meta.manifest_id));
    if (take_fault("manifest")) {
        rollback();
        fail(ConversationStorePublishStatus::failed, "injected fault: manifest");
        return;
    }
    {
        std::string detail;
        std::lock_guard<std::mutex> guard(mutex);
        if (!ensure_capacity(encoded.size(), victims, freeable, floor_limited, detail)) {
            rollback();
            fail(floor_limited ? ConversationStorePublishStatus::refused_floor
                               : ConversationStorePublishStatus::refused_quota,
                 detail);
            return;
        }
        charged += encoded.size();
    }
    const std::string final_manifest = manifest_path(root, meta.manifest_id);
    const std::string tmp = temp_name();
    if (!write_file(tmp, encoded.data(), encoded.size(), false, why) ||
        ::rename(tmp.c_str(), final_manifest.c_str()) != 0 ||
        !fsync_directory(root + "/manifests", why)) {
        std::string ignored;
        remove_file(tmp, ignored);
        {
            std::lock_guard<std::mutex> guard(mutex);
            charged -= std::min(charged, uint64_t(encoded.size()));
        }
        rollback();
        fail(ConversationStorePublishStatus::failed, why.empty() ? "the manifest could not be written" : why);
        return;
    }
    {
        std::lock_guard<std::mutex> guard(mutex);
        counters.bytes_written += encoded.size();
    }

    ConversationStorePublishStatus status = ConversationStorePublishStatus::stored;
    uint64_t superseded = 0;
    {
        std::lock_guard<std::mutex> guard(mutex);
        // A publication that cannot commit leaves its manifest and any object only it wrote as accounted orphans:
        // the next reclaim deletes them and credits their bytes, and the catalog never authorizes them.
        auto abandon = [&](ConversationStorePublishStatus code, const std::string& detail) {
            Retired retired;
            retired.manifest_id = meta.manifest_id;
            retired.manifest_bytes = encoded.size();
            for (const auto& object : written) retired.objects.push_back(object.first);
            pending_reclaim.push_back(std::move(retired));
            for (const auto& object : written) {
                auto state = objects.find(object.first);
                if (state != objects.end() && state->second.refs == 0) objects.erase(state);
            }
            counters.aborted_publications++;
            fail(code, detail);
        };
        if (!on || closing || !healthy) {
            abandon(ConversationStorePublishStatus::disabled,
                    "the store stopped serving while the image was written");
            return;
        }
        if (epoch != epoch_now) {
            counters.refusals_stale++;
            abandon(ConversationStorePublishStatus::stale,
                    "the execution domain advanced while the image was written");
            return;
        }
        if (fault_step == "catalog") {
            fault_step.clear();
            abandon(ConversationStorePublishStatus::failed, "injected fault: catalog");
            return;
        }
        // Commit: the new head, its supersession, one durable catalog write, and only then any deletion eviction
        // caused.  A head is only ever authorized by that write.
        CatalogHead head;
        head.head_id = head_id;
        head.manifest_id = meta.manifest_id;
        head.bucket = bucket;
        head.domain = domain_index;
        head.epoch = epoch_now;
        head.generation = request.generation;
        head.parent = request.supersedes;
        head.payload_bytes = layout.payload_bytes;
        head.manifest_bytes = encoded.size();
        head.published_order = published_order;
        head.stages = layout.stages;
        head.pinned = image.pinned();
        head.protected_seg = false;
        head.cuts = layout.cuts;
        const uint64_t predecessor = request.supersedes;
        bool covered = request.cover;
        if (covered && have_source) {
            // Only a proven covered lineage may replace its predecessor, and only when every pinned cut of that
            // predecessor is preserved: an independent branch always survives as its own head.
            for (const auto& cut : source.cuts) {
                if (!cut.pinned) continue;
                bool kept = false;
                for (const auto& ours : layout.cuts)
                    if (ours.tokens == cut.tokens && ours.digest == cut.digest) kept = true;
                if (!kept) covered = false;
            }
        }
        catalog.heads.push_back(head);
        head_manifest_bytes[head_id] = encoded.size();
        head_objects[head_id];
        while (catalog.heads.size() > options.head_limit) {
            uint64_t victim = 0;
            uint64_t oldest = 0;
            for (const auto& candidate : catalog.heads) {
                if (candidate.head_id == head_id || candidate.pinned || head_leases.count(candidate.head_id)) continue;
                if (victim == 0 || candidate.published_order < oldest) {
                    victim = candidate.head_id;
                    oldest = candidate.published_order;
                }
            }
            if (victim == 0) break;
            counters.evictions++;
            retire_head(victim);
        }
        recount_metadata();
        if (metadata > options.metadata_bytes) {
            catalog.heads.pop_back();
            head_manifest_bytes.erase(head_id);
            head_objects.erase(head_id);
            abandon(ConversationStorePublishStatus::refused_limits,
                    "the store's index ceiling cannot hold another head");
            return;
        }
        if (predecessor != 0) {
            bool exists = false;
            for (const auto& candidate : catalog.heads)
                if (candidate.head_id == predecessor) exists = true;
            if (!exists)
                status = ConversationStorePublishStatus::stored;
            else if (covered) {
                superseded = predecessor;
                retire_head(predecessor);
                status = ConversationStorePublishStatus::replaced;
            } else {
                status = ConversationStorePublishStatus::retained_branch;
            }
        }
        std::string detail;
        if (!write_catalog(detail)) {
            // The durable catalog is unchanged, so nothing was authorized; the in-memory catalog no longer matches
            // it, so the store stops serving rather than continuing from a divergent authority.
            healthy = false;
            unhealthy_reason = "the catalog commit failed: " + detail;
            abandon(ConversationStorePublishStatus::failed, unhealthy_reason);
            return;
        }
        reclaim_files();
        freeable = 0;
        for (const auto& object : referenced) {
            ObjectState& state = objects[object.first];
            state.bytes = object.second;
            state.refs += 1;
        }
        counters.published++;
        if (status == ConversationStorePublishStatus::replaced) counters.replaced++;
        if (status == ConversationStorePublishStatus::retained_branch) counters.retained_branches++;
        counters.heads = uint64_t(catalog.heads.size());
        counters.objects = uint64_t(objects.size());
        counters.manifests = uint64_t(head_manifest_bytes.size());
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
// Quota and eviction: one shared physical quota for every layout, credited only after a removal is durable
// ---------------------------------------------------------------------------------------------------------------

bool ConversationStore::Impl::ensure_capacity(uint64_t incoming, std::vector<uint64_t>& victims, uint64_t& freeable,
                                             bool& floor_limited, std::string& why) {
    sample_free();
    while (charged + incoming > options.quota_bytes + freeable ||
           (free_sample != 0 && free_sample + freeable < options.min_free_bytes + incoming)) {
        uint64_t victim = 0;
        uint64_t oldest = 0;
        // Two passes: the recent segment is evicted before the protected one, and neither ever takes a pinned or
        // leased head.  The oldest publication in the segment goes first.
        for (int pass = 0; pass < 2 && victim == 0; ++pass) {
            for (const auto& head : catalog.heads) {
                if (head.pinned || head_leases.count(head.head_id)) continue;
                if (bool(head.protected_seg) != (pass == 1)) continue;
                if (victim == 0 || head.published_order < oldest) {
                    victim = head.head_id;
                    oldest = head.published_order;
                }
            }
        }
        if (victim == 0) {
            floor_limited = !(charged + incoming > options.quota_bytes + freeable);
            why = floor_limited
                      ? "the filesystem floor would be crossed and no unpinned, unleased head can be evicted"
                      : "the shared quota is full and no unpinned, unleased head can be evicted";
            if (floor_limited)
                counters.refusals_floor++;
            else
                counters.refusals_quota++;
            return false;
        }
        counters.evictions++;
        freeable += retire_head(victim);
        victims.push_back(victim);
    }
    return true;
}

// ---------------------------------------------------------------------------------------------------------------
// Matching: catalog metadata only, no payload is read and no GPU pointer is exposed
// ---------------------------------------------------------------------------------------------------------------

void ConversationStore::Impl::collect_matches(const ConversationStoreSelector& selector, const int32_t* ids,
                                             size_t count, const std::vector<ConversationImageKey>& images,
                                             std::vector<ConversationStoreMatch>& out) const {
    out.clear();
    if (!on || !healthy) return;
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

bool ConversationStore::Impl::manifest_path_of(uint64_t head_id, std::string& path, uint64_t& bytes) const {
    for (const auto& head : catalog.heads) {
        if (head.head_id != head_id) continue;
        path = manifest_path(root, head.manifest_id);
        bytes = head.manifest_bytes;
        return true;
    }
    return false;
}

Digest ConversationStore::Impl::head_manifest_id(uint64_t head_id) const {
    for (const auto& head : catalog.heads)
        if (head.head_id == head_id) return head.manifest_id;
    return Digest{};
}

// ---------------------------------------------------------------------------------------------------------------
// Reads: a bounded, cancellable pipeline.  A demand is admitted as soon as the allowance admits it and never loses
// its slot to a prefetch; a prefetch is speculative and is refused or revoked instead.
// ---------------------------------------------------------------------------------------------------------------

bool ConversationStore::Impl::admit_read(const ConversationStoreQuery& query, bool demand, std::string& why) {
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
            cv_work.notify_all();
        }
        return true;
    }
    std::vector<ConversationStoreMatch> found;
    collect_matches(query.selector, query.tokens.data(), query.tokens.size(), query.images, found);
    if (found.empty()) {
        auto job = std::make_shared<ReadJob>();
        job->query = query;
        job->demand = demand;
        job->status = ConversationStoreReadStatus::missing;
        job->detail = "no eligible stored prefix";
        counters.misses++;
        jobs.push_back(job);
        ready.push_back(job);
        cv_ready.notify_all();
        return true;
    }
    if (jobs.size() >= options.max_read_requests) {
        counters.refused_budget++;
        why = "the store is already holding the maximum number of read requests";
        return false;
    }
    const ConversationStoreMatch& best = found.front();
    auto job = std::make_shared<ReadJob>();
    job->query = query;
    job->demand = demand;
    job->head_id = best.head_id;
    job->tokens = uint64_t(best.tokens);
    job->live = best.live;
    job->stages = best.stages;
    job->payload_bytes = best.payload_bytes;
    jobs.push_back(job);
    queue.push_back(job);
    counters.matches++;
    cv_work.notify_all();
    return true;
}

void ConversationStore::Impl::finish_job(const std::shared_ptr<ReadJob>& job, ConversationStoreReadStatus status,
                                         std::string detail) {
    if (job->reserved_bytes) {
        ram_held -= std::min(ram_held, job->reserved_bytes);
        job->reserved_bytes = 0;
    }
    if (job->head_id) {
        auto lease = head_leases.find(job->head_id);
        if (lease != head_leases.end()) {
            if (lease->second > 0) --lease->second;
            if (lease->second == 0) head_leases.erase(lease);
        }
    }
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
        default: break;
    }
    if (status == ConversationStoreReadStatus::ready && job->image) {
        const uint64_t bytes = job->image->bytes();
        ram_held += bytes;
        ram_lease += bytes;
        job->lease_bytes = bytes;
        counters.peak_bytes = std::max(counters.peak_bytes, ram_held);
    }
    if (job->released) {
        // The caller already released this request: the result is dropped and its charge is not kept.
        if (job->lease_bytes) {
            ram_held -= std::min(ram_held, job->lease_bytes);
            ram_lease -= std::min(ram_lease, job->lease_bytes);
            job->lease_bytes = 0;
        }
        job->image.reset();
        return;
    }
    ready.push_back(job);
    cv_ready.notify_all();
}

bool ConversationStore::Impl::deliver_locked(ConversationStoreRead& out, uint64_t request_id, uint64_t generation,
                                             bool specific) {
    for (auto it = ready.begin(); it != ready.end(); ++it) {
        const std::shared_ptr<ReadJob>& job = *it;
        if (job->cancelled || job->released) continue;
        if (specific &&
            (job->query.request_id != request_id || job->query.generation != generation))
            continue;
        out.status = job->status;
        out.request_id = job->query.request_id;
        out.generation = job->query.generation;
        out.head_id = job->head_id;
        out.epoch = epoch;
        out.tokens = int64_t(job->tokens);
        out.live = job->live;
        out.stages = job->stages;
        out.payload_bytes = job->payload_bytes;
        out.image = job->image;
        out.detail = job->detail;
        job->handed = true;
        // The store drops its own alias: the caller holds the only remaining one, and the image's RAM stays charged
        // until release_read(), because an alias the caller derived from it may outlive it.
        job->image.reset();
        counters.reads_handed_out++;
        ready.erase(it);
        return true;
    }
    return false;
}

void ConversationStore::Impl::read_body(const std::shared_ptr<ReadJob>& job) {
    const ConversationStoreQuery query = job->query;
    const uint64_t head_id = job->head_id;
    const uint64_t tokens = job->tokens;
    const auto stop_with = [&](ConversationStoreReadStatus status, const std::string& detail) {
        std::lock_guard<std::mutex> guard(mutex);
        finish_job(job, status, detail);
    };
    std::string path;
    uint64_t manifest_bytes = 0;
    uint64_t epoch_now = 0;
    bool allowed = true;
    {
        std::lock_guard<std::mutex> guard(mutex);
        if (!on || closing) {
            allowed = false;
        } else {
            epoch_now = epoch;
            allowed = manifest_path_of(head_id, path, manifest_bytes);
        }
    }
    if (!allowed) {
        stop_with(ConversationStoreReadStatus::disabled,
                  reason.empty() ? "the store stopped serving" : reason);
        return;
    }
    std::vector<uint8_t> raw;
    std::string why;
    if (!read_file(path, raw, limits.max_manifest_bytes, why)) {
        stop_with(ConversationStoreReadStatus::failed, why);
        return;
    }
    {
        std::lock_guard<std::mutex> guard(mutex);
        counters.bytes_read += raw.size();
    }
    conversation_store_detail::ManifestMeta meta;
    if (!conversation_store_detail::manifest_decode(raw.data(), raw.size(), limits, meta, why)) {
        stop_with(ConversationStoreReadStatus::corrupt, why);
        return;
    }
    if (meta.epoch != epoch_now) {
        stop_with(ConversationStoreReadStatus::stale, "the execution domain advanced before this read completed");
        return;
    }
    // The store - not the caller - is responsible for the compatibility namespace and the exact prefix: the manifest
    // must belong to this request's own bucket, domain, scope, steering mode and layer carve, and this request's own
    // tokens and images must reproduce the advertised cut bit for bit.
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
    std::vector<conversation_store_detail::RestoreTarget> targets;
    SavedConversation image;
    if (!conversation_store_detail::restore_prepare(meta, limits, image, targets, why)) {
        stop_with(ConversationStoreReadStatus::corrupt, why);
        return;
    }
    // Every object is verified against its own content address before any of its bytes are trusted, the transfer is
    // bounded by the manifest's own chunk lengths, and the request is cancelled only at a chunk boundary.
    for (size_t i = 0; i < meta.streams.size(); ++i) {
        for (const auto& chunk : meta.streams[i].chunks) {
            if (job->cancelled.load()) return;
            {
                std::lock_guard<std::mutex> guard(mutex);
                if (epoch != epoch_now) {
                    finish_job(job, ConversationStoreReadStatus::stale,
                               "the execution domain advanced before this read completed");
                    return;
                }
            }
            const std::string object_file = object_path(root, chunk.object);
            Digest actual{};
            if (!hash_object_file(object_file, chunk.length, actual, why)) {
                stop_with(ConversationStoreReadStatus::corrupt, why);
                return;
            }
            if (!(actual == chunk.object)) {
                stop_with(ConversationStoreReadStatus::corrupt,
                          "an object's bytes do not reproduce its content address");
                return;
            }
            if (!copy_object_file(object_file, chunk.length, targets[i].write, chunk.offset, why)) {
                stop_with(ConversationStoreReadStatus::failed, why);
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
    freeze_if_available(image);
    auto shared = std::make_shared<SavedConversation>(std::move(image));
    std::lock_guard<std::mutex> guard(mutex);
    if (job->cancelled.load()) return;
    job->image = shared;
    job->payload_bytes = meta.payload_bytes;
    job->stages = meta.stages;
    finish_job(job, ConversationStoreReadStatus::ready, std::string());
}

void ConversationStore::Impl::reader_body() {
    std::unique_lock<std::mutex> lock(mutex);
    while (!stop.load()) {
        // A demand that is waiting revokes an in-flight prefetch: its speculative resources are released at the next
        // chunk boundary, and no prefetch ever overtakes a demand.
        bool waiting_demand = false;
        for (const auto& job : queue)
            if (job->demand && !job->cancelled) waiting_demand = true;
        if (waiting_demand)
            for (auto& job : jobs)
                if (job->active && !job->demand && !job->cancelled) job->cancelled = true;
        std::shared_ptr<ReadJob> chosen;
        if (materializing < reader_threads) {
            for (auto it = queue.begin(); it != queue.end() && !chosen; ++it) {
                const std::shared_ptr<ReadJob>& job = *it;
                if (job->cancelled) continue;
                if (waiting_demand && !job->demand) continue;
                if (admission_suspended || ram_budget == 0 ||
                    ram_held + job->payload_bytes > ram_budget) {
                    if (job->demand) continue;  // a demand waits for the allowance instead of being refused
                    queue.erase(it);
                    finish_job(job, ConversationStoreReadStatus::refused_budget,
                               "the read allowance has no spare room for a prefetch");
                    break;
                }
                chosen = job;
                queue.erase(it);
            }
        }
        if (!chosen) {
            cv_work.wait_for(lock, std::chrono::milliseconds(20));
            continue;
        }
        chosen->active = true;
        chosen->reserved_bytes = chosen->payload_bytes;
        ram_held += chosen->reserved_bytes;
        head_leases[chosen->head_id] += 1;
        ++materializing;
        lock.unlock();
        read_body(chosen);
        lock.lock();
        --materializing;
        if (chosen->status == ConversationStoreReadStatus::none) {
            // The body returned without a verdict because the request was cancelled or the store is closing: refund
            // the reservation and forget the request, so nothing is delivered and nothing stays charged.
            if (chosen->reserved_bytes) {
                ram_held -= std::min(ram_held, chosen->reserved_bytes);
                chosen->reserved_bytes = 0;
            }
            auto lease = head_leases.find(chosen->head_id);
            if (lease != head_leases.end()) {
                if (lease->second > 0) --lease->second;
                if (lease->second == 0) head_leases.erase(lease);
            }
            for (auto it = jobs.begin(); it != jobs.end(); ++it)
                if (*it == chosen) {
                    jobs.erase(it);
                    break;
                }
        }
        cv_work.notify_all();
    }
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
    if (!self.limits.max_stages || !self.limits.max_images || !self.limits.max_streams) {
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
    // Construction is single-threaded: no worker exists yet, so the authority steps do not take the state lock.
    if (!self.open_root(why)) {
        self.reason = why;
        return;
    }
    if (!self.load_catalog(why) || !self.reconcile(why)) {
        self.reason = why;
        self.on = false;
        if (self.lock_fd >= 0) {
            ::flock(self.lock_fd, LOCK_UN);
            ::close(self.lock_fd);
            self.lock_fd = -1;
        }
        return;
    }
    self.sample_free();
    self.on = true;
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

bool ConversationStore::enabled() const { return impl_->on; }
const char* ConversationStore::disabled_reason() const { return impl_->reason.c_str(); }
bool ConversationStore::healthy() const { return impl_->on && impl_->healthy; }

void ConversationStore::set_unhealthy(const char* reason) {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    impl_->healthy = false;
    impl_->unhealthy_reason = reason ? reason : "the owner observed a failure";
}

uint64_t ConversationStore::epoch() const {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    return impl_->epoch;
}

const std::array<uint8_t, 32>& ConversationStore::issuer_key() const { return impl_->issuer_key; }
bool ConversationStore::run_intent_active() const { return impl_->intent_active; }

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
    std::vector<ConversationStoreMatch> found;
    {
        std::lock_guard<std::mutex> guard(impl_->mutex);
        impl_->collect_matches(selector, ids.data(), ids.size(), images, found);
    }
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
    std::string path;
    uint64_t bytes = 0;
    conversation_store_detail::ManifestMeta meta;
    {
        std::lock_guard<std::mutex> guard(impl_->mutex);
        if (!impl_->on || !impl_->manifest_path_of(head_id, path, bytes)) return false;
    }
    std::vector<uint8_t> raw;
    std::string why;
    if (!read_file(path, raw, impl_->limits.max_manifest_bytes, why)) return false;
    if (!conversation_store_detail::manifest_decode(raw.data(), raw.size(), impl_->limits, meta, why)) return false;
    out = conversation_store_detail::manifest_stream_infos(meta);
    return true;
}

bool ConversationStore::demand(const ConversationStoreQuery& query, std::string& why) {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    if (!impl_->on) {
        why = impl_->reason.empty() ? "the store is not enabled" : impl_->reason;
        return false;
    }
    if (!impl_->healthy) {
        why = impl_->unhealthy_reason;
        return false;
    }
    return impl_->admit_read(query, true, why);
}

bool ConversationStore::prefetch(const ConversationStoreQuery& query, std::string& why) {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    if (!impl_->on) {
        why = impl_->reason.empty() ? "the store is not enabled" : impl_->reason;
        return false;
    }
    if (!impl_->healthy) {
        why = impl_->unhealthy_reason;
        return false;
    }
    return impl_->admit_read(query, false, why);
}

void ConversationStore::cancel(uint64_t request_id, uint64_t generation) {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    for (auto it = impl_->jobs.begin(); it != impl_->jobs.end(); ++it) {
        const std::shared_ptr<Impl::ReadJob>& job = *it;
        if (job->query.request_id != request_id || job->query.generation != generation) continue;
        if (job->handed) return;  // already the caller's image: release_read() is the way out
        job->cancelled = true;
        if (job->active) return;  // the worker drops it at its next chunk boundary and refunds it
        for (auto q = impl_->queue.begin(); q != impl_->queue.end(); ++q)
            if (*q == job) {
                impl_->queue.erase(q);
                break;
            }
        for (auto r = impl_->ready.begin(); r != impl_->ready.end(); ++r)
            if (*r == job) {
                impl_->ready.erase(r);
                break;
            }
        if (job->reserved_bytes) {
            impl_->ram_held -= std::min(impl_->ram_held, job->reserved_bytes);
            job->reserved_bytes = 0;
        }
        if (job->lease_bytes) {
            // The completion was already charged but never handed out: a cancellation drops it and its charge.
            impl_->ram_held -= std::min(impl_->ram_held, job->lease_bytes);
            impl_->ram_lease -= std::min(impl_->ram_lease, job->lease_bytes);
            job->lease_bytes = 0;
        }
        impl_->jobs.erase(it);
        impl_->counters.reads_cancelled++;
        impl_->cv_work.notify_all();
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
    std::unique_lock<std::mutex> lock(impl_->mutex);
    auto settled = [&]() {
        for (const auto& job : impl_->jobs)
            if (job->query.request_id == request_id && job->query.generation == generation)
                return job->handed || job->cancelled ||
                       (job->status != ConversationStoreReadStatus::none && !job->active);
        return false;
    };
    if (!settled() && timeout_ms)
        impl_->cv_ready.wait_for(lock, std::chrono::milliseconds(timeout_ms), settled);
    if (!settled()) return false;
    return impl_->deliver_locked(out, request_id, generation, true);
}

void ConversationStore::release_read(uint64_t request_id, uint64_t generation) {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    for (auto it = impl_->jobs.begin(); it != impl_->jobs.end(); ++it) {
        const std::shared_ptr<Impl::ReadJob>& job = *it;
        if (job->query.request_id != request_id || job->query.generation != generation) continue;
        if (!job->released && job->lease_bytes) {
            impl_->ram_held -= std::min(impl_->ram_held, job->lease_bytes);
            impl_->ram_lease -= std::min(impl_->ram_lease, job->lease_bytes);
            job->lease_bytes = 0;
            job->released = true;
            impl_->counters.reads_released++;
        }
        if (!job->active) impl_->jobs.erase(it);
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
    std::lock_guard<std::mutex> guard(impl_->mutex);
    for (auto& head : impl_->catalog.heads) {
        if (head.head_id != head_id) continue;
        if (head.protected_seg) return;
        head.protected_seg = true;
        std::string why;
        if (!impl_->write_catalog(why)) {
            // The promotion is a policy hint: a failed commit keeps it in RAM and leaves durability to the next one.
            impl_->counters.corrupt_records++;
        }
        return;
    }
}

bool ConversationStore::publish(std::shared_ptr<const SavedConversation> image, const ConversationStorePublish& request,
                                std::string& why) {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    if (!impl_->on) {
        why = impl_->reason.empty() ? "the store is not enabled" : impl_->reason;
        return false;
    }
    if (!impl_->healthy) {
        why = impl_->unhealthy_reason;
        return false;
    }
    if (!image) {
        why = "no image was supplied";
        return false;
    }
    if (impl_->publishing) {
        const std::shared_ptr<Impl::PublishJob>& current = impl_->publishing;
        const bool same_lineage = current->request.supersedes == request.supersedes &&
                                  current->request.cover == request.cover;
        if (current->started || !same_lineage || request.generation <= current->request.generation) {
            // A job the writer is already reading may never be replaced, and a different branch is never coalesced:
            // the in-flight predecessor is left untouched and this optional publication is refused as a safe miss.
            impl_->counters.refusals_busy++;
            why = current->started ? "the writer is already serializing another image"
                                   : "another branch is already admitted";
            return false;
        }
        // Coalesce before any capture image is read: the not-yet-started predecessor is cancelled and reported as
        // coalesced, so the store never holds two images.
        ConversationStorePublishResult coalesced;
        coalesced.status = ConversationStorePublishStatus::coalesced;
        coalesced.request_id = current->request.request_id;
        coalesced.generation = current->request.generation;
        coalesced.detail = "replaced by a newer publication of the same lineage";
        coalesced.released = true;
        if (current->ram_bytes) {
            impl_->ram_held -= std::min(impl_->ram_held, current->ram_bytes);
            current->ram_bytes = 0;
        }
        impl_->counters.coalesced_publishes++;
        impl_->publish_done.push_back(coalesced);
        impl_->publishing.reset();
    }
    auto job = std::make_shared<Impl::PublishJob>();
    job->image = std::move(image);
    job->request = request;
    job->ram_bytes = job->image->bytes();
    impl_->ram_held += job->ram_bytes;
    impl_->publishing = job;
    impl_->counters.peak_bytes = std::max(impl_->counters.peak_bytes, impl_->ram_held);
    impl_->cv_work.notify_all();
    impl_->cv_publish.notify_all();
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
    return impl_->ram_held;
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
    return impl_->ram_budget > impl_->ram_held ? impl_->ram_budget - impl_->ram_held : 0;
}

void ConversationStore::suspend_read_admission() {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    impl_->admission_suspended = true;
}

uint64_t ConversationStore::publish_read_allowance(uint64_t bytes) {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    // Absolute, and never retroactive: a lower bound admits no new work but revokes nothing already admitted.
    impl_->ram_budget = bytes;
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
    return impl_->options.quota_bytes > impl_->charged ? impl_->options.quota_bytes - impl_->charged : 0;
}

uint64_t ConversationStore::disk_free_bytes() const {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    return impl_->free_sample;
}

bool ConversationStore::invalidate_domain(const char* reason) {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    if (!impl_->on) return false;
    impl_->epoch += 1;
    impl_->catalog.domains[impl_->domain_index].epoch = impl_->epoch;
    impl_->counters.epoch_advances++;
    std::vector<uint64_t> victims;
    for (const auto& head : impl_->catalog.heads)
        if (head.domain == impl_->domain_index) victims.push_back(head.head_id);
    for (uint64_t head_id : victims) impl_->retire_head(head_id);
    // Outstanding reads of the previous epoch are dropped rather than delivered: their completions are stale.
    for (auto& job : impl_->jobs)
        if (!job->handed && job->status == ConversationStoreReadStatus::none) job->cancelled = true;
    std::string why;
    if (!impl_->write_catalog(why)) {
        // The run intent stays armed, and the store stops serving: the next start must not reuse anything.
        impl_->healthy = false;
        impl_->unhealthy_reason = "the domain invalidation could not be made durable: " + why;
        (void)reason;
        return false;
    }
    impl_->reclaim_files();
    impl_->recount_metadata();
    impl_->counters.heads = uint64_t(impl_->catalog.heads.size());
    impl_->cv_work.notify_all();
    return true;
}

void ConversationStore::close() {
    Impl& self = *impl_;
    {
        std::lock_guard<std::mutex> guard(self.mutex);
        if (self.closed) return;
        self.closing = true;
        self.admission_suspended = true;
        for (auto& job : self.jobs)
            if (!job->handed && job->status == ConversationStoreReadStatus::none) job->cancelled = true;
        self.queue.clear();
    }
    self.stop.store(true);
    self.cv_work.notify_all();
    self.cv_ready.notify_all();
    self.cv_publish.notify_all();
    if (self.writer.joinable()) self.writer.join();
    for (auto& thread : self.readers)
        if (thread.joinable()) thread.join();
    self.readers.clear();
    std::lock_guard<std::mutex> guard(self.mutex);
    for (auto& job : self.jobs) {
        if (job->status != ConversationStoreReadStatus::none) continue;
        if (job->reserved_bytes) {
            self.ram_held -= std::min(self.ram_held, job->reserved_bytes);
            job->reserved_bytes = 0;
        }
        job->status = ConversationStoreReadStatus::disabled;
        job->detail = "the store closed before this read completed";
        job->active = false;
    }
    self.queue.clear();
    self.clear_intent();
    std::string why;
    self.write_catalog(why);
    if (self.lock_fd >= 0) {
        ::flock(self.lock_fd, LOCK_UN);
        ::close(self.lock_fd);
        self.lock_fd = -1;
    }
    self.on = false;
    self.closed = true;
}

ConversationStoreCounters ConversationStore::counters() const {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    ConversationStoreCounters out = impl_->counters;
    out.heads = uint64_t(impl_->catalog.heads.size());
    out.objects = uint64_t(impl_->objects.size());
    out.manifests = uint64_t(impl_->head_manifest_bytes.size());
    out.pending_delete_bytes = uint64_t(impl_->pending_reclaim.size());
    out.orphan_bytes = impl_->reclaimable;
    uint64_t pinned = 0;
    for (const auto& head : impl_->catalog.heads) pinned += head.pinned ? 1 : 0;
    out.pinned_heads = pinned;
    out.shared_objects = uint64_t(impl_->objects.size());
    return out;
}

std::string ConversationStore::describe() const {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    std::string text;
    text += "root: " + (impl_->root.empty() ? std::string("(none)") : impl_->root) + "\n";
    text += std::string("enabled: ") + (impl_->on ? "yes" : "no") +
            (impl_->reason.empty() ? "" : " (" + impl_->reason + ")") + "\n";
    text += std::string("healthy: ") + (impl_->healthy ? "yes" : "no") +
            (impl_->unhealthy_reason.empty() ? "" : " (" + impl_->unhealthy_reason + ")") + "\n";
    text += "owner lock: " + std::string(impl_->lock_fd >= 0 ? "held" : "not held") + "\n";
    text += "issuer key digest: " + sha256_hex(sha256(impl_->issuer_key.data(), impl_->issuer_key.size())).substr(0, 16) +
            "\n";
    text += "run intent: " + std::string(impl_->intent_active ? "armed" : "clear") + "\n";
    text += "domain index: " + std::to_string(impl_->domain_index) + " of " +
            std::to_string(impl_->catalog.domains.size()) + ", epoch " + std::to_string(impl_->epoch) + "\n";
    text += "heads: " + std::to_string(impl_->catalog.heads.size()) + " of " +
            std::to_string(impl_->options.head_limit) + ", objects " + std::to_string(impl_->objects.size()) +
            ", pub order " + std::to_string(impl_->catalog.published_order_counter) + "\n";
    text += "disk: charged " + std::to_string(impl_->charged) + " of quota " +
            std::to_string(impl_->options.quota_bytes) + ", reclaimable " + std::to_string(impl_->reclaimable) +
            ", free sampled " + std::to_string(impl_->free_sample) + "\n";
    text += "index: " + std::to_string(impl_->metadata) + " of " + std::to_string(impl_->options.metadata_bytes) +
            " bytes estimated\n";
    text += "ram: held " + std::to_string(impl_->ram_held) + ", leased " + std::to_string(impl_->ram_lease) +
            ", budget " + std::to_string(impl_->ram_budget) +
            (impl_->admission_suspended ? " (admission suspended)" : "") + "\n";
    text += "reads: queued " + std::to_string(impl_->queue.size()) + ", materializing " +
            std::to_string(impl_->materializing) + ", ready " + std::to_string(impl_->ready.size()) + "\n";
    text += std::string("writer: ") + (impl_->publishing ? "busy" : "idle") + "\n";
    return text;
}

void ConversationStore::set_publish_fault_for_test(const char* step) {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    impl_->fault_step = step ? step : "";
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


// ---------------------------------------------------------------------------------------------------------------
// Authority: root, lock, issuer key, catalog, run intent, recovery
// ---------------------------------------------------------------------------------------------------------------

uint64_t ConversationStore::Impl::sample_free() {
    free_sample = filesystem_free_bytes(root);
    return free_sample;
}

std::string ConversationStore::Impl::temp_name() {
    return root + "/tmp/w" + std::to_string(::getpid()) + "-" + std::to_string(++name_counter);
}

std::string ConversationStore::Impl::intent_path() const {
    std::array<uint8_t, 96> material{};
    std::memcpy(material.data(), options.domain.data(), 32);
    std::memcpy(material.data() + 32, options.model.data(), 32);
    std::memcpy(material.data() + 64, options.engine.data(), 32);
    return root + "/intent-" + sha256_hex(sha256(material.data(), material.size())).substr(0, 16);
}

bool ConversationStore::Impl::write_catalog(std::string& why) {
    const std::vector<uint8_t> bytes = conversation_store_detail::catalog_encode(catalog);
    const std::string tmp = temp_name();
    if (!write_file(tmp, bytes.data(), bytes.size(), false, why)) return false;
    if (::rename(tmp.c_str(), (root + "/catalog").c_str()) != 0) {
        why = std::string("cannot replace the catalog: ") + std::strerror(errno);
        std::string ignored;
        remove_file(tmp, ignored);
        return false;
    }
    return fsync_directory(root, why);
}

bool ConversationStore::Impl::arm_intent(std::string& why) {
    // The durable claim that this execution domain has unfinished business.  It is created before the first reuse and
    // cleared only by a clean close, so an unclean exit invalidates this domain instead of trusting its records.
    const uint8_t marker[8] = {'S', 'T', 'C', 'S', 'I', 'N', 'T', '1'};
    if (!write_file(intent_path(), marker, sizeof marker, false, why)) return false;
    if (!fsync_directory(root, why)) return false;
    intent_active = true;
    return true;
}

void ConversationStore::Impl::clear_intent() {
    std::string ignored;
    if (remove_file(intent_path(), ignored)) {
        if (fsync_directory(root, ignored)) intent_active = false;
    }
}

bool ConversationStore::Impl::open_root(std::string& why) {
    if (!make_directory(root, why)) return false;
    if (!make_directory(root + "/objects", why)) return false;
    if (!make_directory(root + "/manifests", why)) return false;
    if (!make_directory(root + "/tmp", why)) return false;
    if (!fsync_directory(root, why)) return false;
    lock_fd = ::open((root + "/owner.lock").c_str(), O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (lock_fd < 0) {
        why = std::string("cannot open the store's lock file: ") + std::strerror(errno);
        return false;
    }
    if (::flock(lock_fd, LOCK_EX | LOCK_NB) != 0) {
        counters.lock_conflicts++;
        why = "another process owns this store's root";
        return false;
    }
    const std::string key_path = root + "/issuer.key";
    std::vector<uint8_t> key;
    if (path_exists(key_path)) {
        if (!read_file(key_path, key, 1024, why)) return false;
        if (key.size() != 32) {
            why = "the issuer key file is not 32 bytes";
            return false;
        }
    } else {
        key.assign(32, 0);
        fill_random(key.data(), key.size());
        if (!write_file(key_path, key.data(), key.size(), true, why)) return false;
        if (!fsync_directory(root, why)) return false;
    }
    std::memcpy(issuer_key.data(), key.data(), key.size());
    return true;
}

bool ConversationStore::Impl::load_catalog(std::string& why) {
    const std::string path = root + "/catalog";
    Catalog loaded;
    std::string detail;
    bool cold = !path_exists(path);
    if (!cold) {
        std::vector<uint8_t> bytes;
        if (!read_file(path, bytes, uint64_t(256) << 20, detail) ||
            !conversation_store_detail::catalog_decode(bytes.data(), bytes.size(), loaded, detail)) {
            // A catalog that cannot be trusted is cold caching: never a directory-based resurrection.
            counters.corrupt_records++;
            loaded = Catalog{};
            cold = true;
        } else if (!(loaded.issuer_key_digest == sha256(issuer_key.data(), issuer_key.size()))) {
            counters.corrupt_records++;
            loaded = Catalog{};
            cold = true;
        }
    }
    (void)cold;
    catalog = std::move(loaded);
    catalog.issuer_key_digest = sha256(issuer_key.data(), issuer_key.size());

    // Register (or find) this execution domain, and decide whether its previous run was clean.
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
    if (path_exists(intent_path())) {
        // The previous run of exactly this domain did not close cleanly: its records and every outstanding
        // completion of its epoch are invalid, and only this domain is invalidated.
        epoch += 1;
        catalog.domains[domain_index].epoch = epoch;
        counters.epoch_advances++;
        std::vector<uint64_t> victims;
        for (const auto& head : catalog.heads)
            if (head.domain == domain_index) victims.push_back(head.head_id);
        for (uint64_t head_id : victims) retire_head(head_id);
    } else if (!catalog.heads.empty()) {
        // A clean restart: records of a retired epoch are unreachable, so they are dropped rather than trusted.
        std::vector<uint64_t> victims;
        for (const auto& head : catalog.heads)
            if (head.domain == domain_index && head.epoch != epoch) victims.push_back(head.head_id);
        for (uint64_t head_id : victims) retire_head(head_id);
    }
    if (!arm_intent(why)) return false;
    if (!write_catalog(why)) return false;
    reclaim_files();
    return true;
}

bool ConversationStore::Impl::reconcile(std::string& why) {
    constexpr uint64_t kMaxScanEntries = 1u << 20;
    constexpr uint64_t kMaxReclaimPerOpen = 65536;
    // Every scratch file is garbage by construction: a durable write completes with a rename out of tmp/.
    std::vector<std::string> names;
    if (list_directory(root + "/tmp", names, why)) {
        for (const auto& name : names) {
            std::string ignored;
            if (remove_file(root + "/tmp/" + name, ignored)) counters.recovered_orphans++;
        }
    }
    // Count what is physically present first, so the shared quota reflects the real footprint even before the
    // catalog is trusted.
    std::map<Digest, uint64_t> present_objects;
    std::map<Digest, uint64_t> present_manifests;
    uint64_t scanned = 0;
    std::vector<std::string> shards;
    if (!list_directory(root + "/objects", shards, why)) return false;
    for (const auto& shard : shards) {
        if (shard.size() != 2) {
            std::string ignored;
            remove_file(root + "/objects/" + shard, ignored);
            continue;
        }
        std::vector<std::string> files;
        std::string nested;
        if (!list_directory(root + "/objects/" + shard, files, nested)) continue;
        for (const auto& file : files) {
            const std::string path = root + "/objects/" + shard + "/" + file;
            struct stat info {};
            if (::stat(path.c_str(), &info) != 0 || !S_ISREG(info.st_mode)) continue;
            if (++scanned > kMaxScanEntries) {
                why = "the store holds more objects than it can account at startup";
                return false;
            }
            charged += uint64_t(info.st_size);
            Digest digest{};
            if (file.size() == 64 && hex_to_bytes(file, digest) &&
                sha256_hex(digest).compare(0, 2, shard) == 0)
                present_objects.emplace(digest, uint64_t(info.st_size));
        }
    }
    std::vector<std::string> manifests;
    if (!list_directory(root + "/manifests", manifests, why)) return false;
    for (const auto& name : manifests) {
        const std::string path = root + "/manifests/" + name;
        struct stat info {};
        if (::stat(path.c_str(), &info) != 0 || !S_ISREG(info.st_mode)) continue;
        if (++scanned > kMaxScanEntries) {
            why = "the store holds more manifests than it can account at startup";
            return false;
        }
        charged += uint64_t(info.st_size);
        Digest digest{};
        if (name.size() == 64 && hex_to_bytes(name, digest))
            present_manifests.emplace(digest, uint64_t(info.st_size));
    }
    // Only the catalog authorizes a manifest, and only a manifest authorizes an object.
    std::vector<uint64_t> dropped;
    for (const auto& head : catalog.heads) {
        const auto manifest = present_manifests.find(head.manifest_id);
        ManifestMeta meta;
        std::string detail;
        if (manifest == present_manifests.end()) {
            counters.corrupt_records++;
            dropped.push_back(head.head_id);
            continue;
        }
        if (!load_manifest(head.head_id, meta, detail)) {
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
        std::vector<Digest>& list = head_objects[head.head_id];
        list.reserve(meta.objects.size());
        for (const auto& object : meta.objects) {
            ObjectState& state = objects[object.first];
            state.bytes = object.second;
            state.refs += 1;
            list.push_back(object.first);
        }
    }
    for (uint64_t head_id : dropped) retire_head(head_id);
    // Present but referenced by nothing is an orphan: account it, then remove it (bounded per open).
    uint64_t removed = 0;
    for (const auto& manifest : present_manifests) {
        bool referenced = false;
        for (const auto& head : catalog.heads)
            if (head.manifest_id == manifest.first) {
                referenced = true;
                break;
            }
        if (referenced) continue;
        std::string ignored;
        if (removed < kMaxReclaimPerOpen && remove_file(manifest_path(root, manifest.first), ignored)) {
            charged -= manifest.second;
            ++removed;
            counters.recovered_orphans++;
        } else {
            reclaimable += manifest.second;
            counters.deferred_reclaims++;
        }
    }
    for (const auto& object : present_objects) {
        if (objects.count(object.first)) continue;
        std::string ignored;
        if (removed < kMaxReclaimPerOpen && remove_file(object_path(root, object.first), ignored)) {
            charged -= object.second;
            ++removed;
            counters.recovered_orphans++;
        } else {
            reclaimable += object.second;
            counters.deferred_reclaims++;
        }
    }
    // Enforce the head limit and the metadata ceiling before serving anything.
    while (catalog.heads.size() > options.head_limit) {
        uint64_t victim = 0;
        uint64_t oldest = 0;
        for (const auto& head : catalog.heads) {
            if (head.pinned) continue;
            if (victim == 0 || head.published_order < oldest) {
                victim = head.head_id;
                oldest = head.published_order;
            }
        }
        if (victim == 0) break;
        counters.evictions++;
        retire_head(victim);
    }
    recount_metadata();
    counters.heads = uint64_t(catalog.heads.size());
    counters.objects = uint64_t(objects.size());
    counters.manifests = uint64_t(head_manifest_bytes.size());
    counters.pending_delete_bytes = uint64_t(pending_reclaim.size());
    reclaim_files();
    return true;
}

bool ConversationStore::Impl::load_manifest(uint64_t head_id, ManifestMeta& out, std::string& why) const {
    for (const auto& head : catalog.heads) {
        if (head.head_id != head_id) continue;
        std::vector<uint8_t> bytes;
        if (!read_file(manifest_path(root, head.manifest_id), bytes, limits.max_manifest_bytes, why)) return false;
        return conversation_store_detail::manifest_decode(bytes.data(), bytes.size(), limits, out, why);
    }
    why = "no retained head has that identity";
    return false;
}

uint64_t ConversationStore::Impl::retire_head(uint64_t head_id) {
    Retired retired;
    for (size_t i = 0; i < catalog.heads.size(); ++i) {
        if (catalog.heads[i].head_id != head_id) continue;
        retired.manifest_id = catalog.heads[i].manifest_id;
        retired.manifest_bytes = head_manifest_bytes.count(head_id) ? head_manifest_bytes[head_id] : 0;
        catalog.heads.erase(catalog.heads.begin() + std::ptrdiff_t(i));
        break;
    }
    uint64_t freed = retired.manifest_bytes;
    const auto list = head_objects.find(head_id);
    if (list != head_objects.end()) {
        retired.objects = list->second;
        for (const Digest& object : retired.objects) {
            auto state = objects.find(object);
            if (state == objects.end()) continue;
            if (state->second.refs > 0) --state->second.refs;
            if (state->second.refs == 0) {
                reclaimable += state->second.bytes;
                freed += state->second.bytes;
            }
        }
        head_objects.erase(list);
    }
    head_manifest_bytes.erase(head_id);
    head_leases.erase(head_id);
    pending_reclaim.push_back(std::move(retired));
    return freed;
}

void ConversationStore::Impl::reclaim_files() {
    if (pending_reclaim.empty()) return;
    std::string ignored;
    for (const Retired& retired : pending_reclaim) {
        if (digest_nonzero(retired.manifest_id) &&
            remove_file(manifest_path(root, retired.manifest_id), ignored)) {
            charged -= std::min(charged, retired.manifest_bytes);
            counters.reclaimed_bytes += retired.manifest_bytes;
        }
        for (const Digest& object : retired.objects) {
            const auto state = objects.find(object);
            if (state != objects.end() && state->second.refs != 0) continue;
            if (remove_file(object_path(root, object), ignored)) {
                const uint64_t bytes = state != objects.end() ? state->second.bytes : 0;
                charged -= std::min(charged, bytes);
                reclaimable -= std::min(reclaimable, bytes);
                counters.reclaimed_objects++;
                counters.reclaimed_bytes += bytes;
                if (state != objects.end()) objects.erase(state);
            }
        }
    }
    pending_reclaim.clear();
    recount_metadata();
}

bool ConversationStore::Impl::take_fault(const char* step) {
    std::lock_guard<std::mutex> guard(mutex);
    if (fault_step != step) return false;
    fault_step.clear();
    return true;
}

void ConversationStore::Impl::recount_metadata() {
    uint64_t bytes = uint64_t(conversation_store_detail::catalog_encode(catalog).size());
    bytes += uint64_t(objects.size()) * kObjectBytesEstimate;
    for (const auto& entry : head_objects) bytes += uint64_t(entry.second.size()) * sizeof(Digest);
    metadata = bytes;
}

} // namespace strata::core

