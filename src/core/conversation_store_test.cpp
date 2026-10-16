// Deterministic CPU tests for the durable conversation store.  No GPU, no model, no network: a temporary root under
// the system temp directory, fixtures built by hand, and checks that read the module's own observable contract -
// matches, statuses, leases, epochs and byte counts.  Every check is a consumer-visible behaviour, not an internal.
#include "strata/core/conversation_store.hpp"

#include "strata/core/conversation_reuse.hpp"

#include "conversation_store_format.hpp"
#include "conversation_store_sha256.hpp"

#include <array>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <functional>
#include <memory>
#include <map>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {

using namespace strata::core;
using conversation_store_detail::image_payload_bytes;
using conversation_store_detail::images_equal;
using conversation_store_detail::Digest;
using conversation_store_detail::sha256;
using conversation_store_detail::sha256_hex;

int failures = 0;
int checks = 0;

void check(bool ok, const char* what) {
    ++checks;
    if (!ok) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s\n", what);
    }
}

void check_eq(uint64_t got, uint64_t want, const char* what) {
    ++checks;
    if (got != want) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s (got %llu, want %llu)\n", what,
                     (unsigned long long) got, (unsigned long long) want);
    }
}

Digest digest_of(const std::string& text) { return sha256(text.data(), text.size()); }

std::array<uint8_t, 32> bytes_from(const std::string& text) { return digest_of(text); }

// The three admission classes a publication can come back with (see ConversationStorePublishAdmission), spelled once
// so every call site states its own expectation: only `kAccepted` means the store took the image and its lease.
constexpr auto kAccepted = ConversationStorePublishAdmission::accepted;
constexpr auto kDeferred = ConversationStorePublishAdmission::deferred;
constexpr auto kRefused = ConversationStorePublishAdmission::refused;

// The digest of a whole file, for the tests that must show a store wrote nothing to the authority.
std::string file_digest(const std::string& path) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return std::string("(missing)");
    std::vector<uint8_t> bytes;
    uint8_t block[4096];
    for (;;) {
        const ssize_t n = ::read(fd, block, sizeof block);
        if (n <= 0) break;
        bytes.insert(bytes.end(), block, block + n);
    }
    ::close(fd);
    return sha256_hex(sha256(bytes.data(), bytes.size()));
}

// The physical headroom one catalog copy needs, at the store's own conservative allocated size (block rounded plus
// one block for the directory entry).  A store must be able to write the copy a commit creates *and* the copy after
// that, so a quota is only "tight" - not impossible - when it leaves these few copies free: anything less and the
// store has to refuse rather than spend a promise, and the eviction that would make room could not be written either.
uint64_t catalog_copy_headroom(const std::string& root) {
    struct stat info {};
    if (::stat((root + "/catalog").c_str(), &info) != 0) return uint64_t(1) << 20;
    const uint64_t charge = (uint64_t(info.st_size) + 4095) / 4096 * 4096 + 4096;
    // Six copies: the live catalog, the copy a transaction creates, their next-copy room and margin for the growth a
    // single commit adds (one head and its cuts).  Deliberately generous: the fixture must not drift with the exact
    // encoding size, and it stays far below the payload a publication needs, so the eviction path is still exercised.
    return charge * 6;
}

// A bounded poll for a status the storage threads have to reach.  Deterministic: it only waits for the store's own
// worker, and it reports the real status.
bool await_status(ConversationStore& store, uint64_t request_id, uint64_t generation,
                  ConversationStoreReadStatus want, int milliseconds = 20000) {
    for (int waited = 0; waited < milliseconds; ++waited) {
        if (store.request_status(request_id, generation) == want) return true;
        ::usleep(1000);
    }
    return store.request_status(request_id, generation) == want;
}

// Capture the real opt-in diagnostic, not a test-only mirror of the read state. The store samples the environment at
// construction; each capture outlives its stores so all worker writes have drained before stderr is restored.
struct StoreReadTraceRecord {
    std::map<std::string, std::string> fields;
    std::string text(const char* key) const {
        const auto found = fields.find(key);
        return found == fields.end() ? std::string() : found->second;
    }
    uint64_t number(const char* key) const { return std::strtoull(text(key).c_str(), nullptr, 10); }
};

struct StoreReadTraceCapture {
    bool had_environment = false;
    std::string old_environment;
    FILE* file = nullptr;
    int saved_stderr = -1;

    explicit StoreReadTraceCapture(bool enabled = true) {
        if (const char* previous = std::getenv("STRATA_STORE_TIMING")) {
            had_environment = true;
            old_environment = previous;
        }
        ::setenv("STRATA_STORE_TIMING", enabled ? "1" : "0", 1);
        file = std::tmpfile();
        check(file != nullptr, "the native read diagnostic capture opens");
        if (!file) return;
        std::fflush(stderr);
        saved_stderr = ::dup(STDERR_FILENO);
        check(saved_stderr >= 0 && ::dup2(::fileno(file), STDERR_FILENO) >= 0,
              "the native read diagnostic capture redirects stderr");
    }

    std::string contents() const {
        if (!file) return {};
        std::string result;
        char block[4096];
        off_t offset = 0;
        for (;;) {
            const ssize_t count = ::pread(::fileno(file), block, sizeof block, offset);
            if (count <= 0) return result;
            result.append(block, size_t(count));
            offset += count;
        }
    }

    std::vector<StoreReadTraceRecord> records() const {
        std::vector<StoreReadTraceRecord> result;
        std::istringstream lines(contents());
        std::string line;
        while (std::getline(lines, line)) {
            if (line.compare(0, 11, "store_read ") != 0) continue;
            StoreReadTraceRecord record;
            std::istringstream fields(line.substr(11));
            std::string field;
            while (fields >> field) {
                const size_t equals = field.find('=');
                if (equals != std::string::npos)
                    record.fields.emplace(field.substr(0, equals), field.substr(equals + 1));
            }
            result.push_back(std::move(record));
        }
        return result;
    }

    ~StoreReadTraceCapture() {
        std::fflush(stderr);
        if (saved_stderr >= 0) {
            ::dup2(saved_stderr, STDERR_FILENO);
            ::close(saved_stderr);
        }
        // Keep ordinary test failures visible even when a diagnostic assertion fails inside this capture.
        std::istringstream lines(contents());
        std::string line;
        while (std::getline(lines, line))
            if (line.compare(0, 11, "store_read ") != 0) std::fprintf(stderr, "%s\n", line.c_str());
        if (file) std::fclose(file);
        if (had_environment)
            ::setenv("STRATA_STORE_TIMING", old_environment.c_str(), 1);
        else
            ::unsetenv("STRATA_STORE_TIMING");
    }
};

const StoreReadTraceRecord* read_trace(const std::vector<StoreReadTraceRecord>& records, const char* event,
                                       uint64_t request_id, const char* reason = nullptr) {
    for (const auto& record : records)
        if (record.text("event") == event && record.number("request_id") == request_id &&
            (!reason || record.text("reason") == reason))
            return &record;
    return nullptr;
}

uint64_t read_trace_count(const std::vector<StoreReadTraceRecord>& records, const char* event, uint64_t request_id) {
    uint64_t count = 0;
    for (const auto& record : records)
        if (record.text("event") == event && record.number("request_id") == request_id) ++count;
    return count;
}

// The store's own physical identity of one stream: the authorized scope digest and the global ordinal that the
// writer and the reader both derive.  Equal pairs plus equal bytes are one physical object.
bool identity_of(const SavedConversation& image, const ConversationStoreStreamId& id,
                 std::pair<Digest, uint64_t>& out, std::string& why) {
    conversation_store_detail::ImageLayout layout;
    if (!conversation_store_detail::image_layout(image, Digest{}, conversation_store_detail::unconstrained_limits(),
                                                layout, why))
        return false;
    Digest scope{};
    uint64_t ordinal = 0;
    if (!conversation_store_detail::stream_object_identity(layout.containers, id, bytes_from("model:test"), scope,
                                                          ordinal, why))
        return false;
    out = {scope, ordinal};
    return true;
}

// ---------------------------------------------------------------------------------------------------------------
// Fixtures and filesystem helpers
// ---------------------------------------------------------------------------------------------------------------

void remove_tree(const std::string& root) {
    if (root.empty()) return;
    DIR* dir = ::opendir(root.c_str());
    if (dir) {
        while (struct dirent* entry = ::readdir(dir)) {
            const std::string name = entry->d_name;
            if (name == "." || name == "..") continue;
            const std::string child = root + "/" + name;
            struct stat info {};
            if (::stat(child.c_str(), &info) == 0 && S_ISDIR(info.st_mode))
                remove_tree(child);
            else
                ::unlink(child.c_str());
        }
        ::closedir(dir);
    }
    ::rmdir(root.c_str());
}

struct TempRoot {
    std::string path;
    TempRoot() {
        char pattern[] = "/tmp/strata-store-test-XXXXXX";
        const char* made = ::mkdtemp(pattern);
        if (made) path = made;
    }
    ~TempRoot() { remove_tree(path); }
};

ConversationStoreSelector selector_of(const SavedConversation& image) {
    ConversationStoreSelector selector;
    selector.scope = image.scope;
    selector.cvec = image.cvec;
    selector.geometry = image.geometry;
    selector.layer_lo = image.layer_lo;
    selector.layer_hi = image.layer_hi;
    return selector;
}

ConversationStoreOptions options_for(const std::string& root, uint64_t quota, uint64_t free_floor) {
    ConversationStoreOptions options;
    options.root = root;
    options.domain = bytes_from("domain:test");
    options.model = bytes_from("model:test");
    options.engine = bytes_from("engine:test");
    options.quota_bytes = quota;
    options.min_free_bytes = free_floor;
    options.metadata_bytes = uint64_t(8) << 20;
    options.head_limit = 8;
    options.chunk_bytes = 4096;
    options.max_reads = 1;
    options.max_read_requests = 8;
    options.max_pending_publishes = 1;
    options.read_budget_bytes = uint64_t(64) << 20;
    return options;
}

// One serving profile's admitted figures: the stage count and the checkpoint policy it captures and restores under,
// and the image budget it serves from.  Two profiles that share one store root differ exactly here, and the same
// root must hold records from both, so this is what a cross-profile recovery test varies.
ConversationStoreLimits profile_limits(uint32_t stages, uint32_t checkpoints, uint64_t budget) {
    ConversationStoreLimits limits;
    limits.max_tokens = uint64_t(1) << 20;
    limits.max_kv_layers = 4;
    limits.max_stages = stages;
    limits.max_checkpoints = checkpoints;
    limits.max_image_bytes = budget;
    limits.max_stream_bytes = budget;
    limits.max_state_bytes = budget;
    return limits;
}

// Opens the shared root under one profile's admitted figures and asserts the invariants every switch must satisfy:
// the store comes up enabled and healthy, every retained head is still there, and not one record was called corrupt
// or evicted by the difference in figures.  The caller runs its own checks with the returned store open.
std::unique_ptr<ConversationStore> open_profile(const std::string& root, uint64_t quota,
                                               const ConversationStoreLimits& limits, uint64_t heads) {
    auto store = std::make_unique<ConversationStore>(options_for(root, quota, 0), limits);
    check(store->enabled(), "the shared root opens under this profile's admitted figures");
    check(store->healthy(), "the profile switch leaves the store healthy");
    check_eq(store->counters().heads, heads, "every retained head survives the profile switch");
    check_eq(store->counters().corrupt_records, 0, "a retained head is not called corrupt by a different profile");
    check_eq(store->counters().evictions, 0, "a profile switch evicts nothing");
    return store;
}

// One image with a live record, one pinned checkpoint, two K/V layers and no stage split: enough to exercise every
// stream family, the pin rule and the K/V layout digest.
SavedConversation fixture(uint64_t seed, int64_t tokens, int32_t extra_kv = 0) {
    SavedConversation image;
    std::memset(image.scope.value.data(), uint8_t((seed | 1) & 0xff), image.scope.value.size());
    for (int i = 0; i < 18; ++i) image.geometry[size_t(i)] = int64_t(seed) * 1000 + i;
    image.geometry[1] = 8;   // two global attention layers, at blocks 3 and 7
    image.geometry[2] = 4;
    image.layer_lo = 0;
    image.layer_hi = 8;
    image.cvec = true;
    for (int64_t t = 0; t < tokens; ++t) image.live.ids.push_back(int32_t(seed * 7 + uint64_t(t)));
    image.live.used = seed;
    image.live.gdn.assign(size_t(tokens), uint8_t(seed));
    image.live.ple.assign(size_t(tokens) / 2, uint8_t(seed + 1));
    image.live.tails.assign(size_t(tokens) / 4, uint8_t(seed + 2));
    image.live.dead.assign(3, uint8_t(seed + 3));
    image.live.block_pos.assign(size_t(tokens), uint8_t(seed + 4));
    ConversationCheckpoint checkpoint;
    checkpoint.ids.assign(image.live.ids.begin(), image.live.ids.begin() + long(tokens / 2));
    checkpoint.used = seed + 1;
    checkpoint.pinned = true;
    checkpoint.gdn.assign(size_t(tokens / 2), uint8_t(seed + 5));
    checkpoint.tails.assign(2, uint8_t(seed + 6));
    image.checkpoints.push_back(std::move(checkpoint));
    for (int layer = 0; layer < 2 + extra_kv; ++layer) {
        ConversationKv kv;
        kv.format = 0;
        kv.cells = tokens;
        kv.heads = 2;
        kv.head_dim = 8;
        kv.page_size = 16;
        kv.pooled_rows = 1;
        kv.idx_dim = 3;
        kv.k.resize(size_t(tokens) * 16, uint8_t(seed + uint64_t(layer)));
        kv.v.resize(size_t(tokens) * 16, uint8_t(seed + 10 + uint64_t(layer)));
        kv.k_scale.resize(8, uint8_t(seed + 20));
        kv.v_scale.resize(8, uint8_t(seed + 21));
        image.kv.push_back(std::move(kv));
    }
    return image;
}

ConversationStoreQuery query_of(const SavedConversation& image, uint64_t request_id, uint64_t generation) {
    ConversationStoreQuery query;
    query.selector = selector_of(image);
    query.tokens = image.live.ids;
    // A stored cut must be strictly shorter than the request: the last prompt token starts the next verify window.
    query.tokens.push_back(12345);
    query.request_id = request_id;
    query.generation = generation;
    return query;
}

// An appended turn: the live record grows and the K/V extents with it, so every unchanged completed prefix byte keeps
// its physical identity (see check_reuse).  `seed` is the base fixture's own seed, so the grown turns continue exactly
// the pattern the fixture wrote.
SavedConversation appended_turn(const SavedConversation& base, int64_t tokens, int seed) {
    SavedConversation grown = base;
    for (int64_t t = int64_t(base.live.ids.size()); t < tokens; ++t)
        grown.live.ids.push_back(int32_t(uint64_t(seed) * 7 + uint64_t(t)));
    grown.live.gdn.resize(grown.live.ids.size(), uint8_t(seed));
    grown.live.ple.resize(grown.live.ids.size() / 2, uint8_t(seed + 1));
    grown.live.tails.resize(grown.live.ids.size() / 4, uint8_t(seed + 2));
    grown.live.block_pos.resize(grown.live.ids.size(), uint8_t(seed + 4));
    for (size_t layer = 0; layer < grown.kv.size(); ++layer) {
        grown.kv[layer].cells = grown.live.ids.size();
        grown.kv[layer].k.resize(size_t(grown.live.ids.size()) * 16, uint8_t(seed + int(layer)));
        grown.kv[layer].v.resize(size_t(grown.live.ids.size()) * 16, uint8_t(seed + 10 + int(layer)));
    }
    return grown;
}

// An image with hundreds of streams and almost no payload: every record is empty and every K/V part is empty, so a
// reservation that scaled with a configured maximum instead of with the actual object would show up immediately.
SavedConversation hollow_fixture(uint64_t seed, int checkpoints, int kv_layers) {
    SavedConversation image;
    std::memset(image.scope.value.data(), uint8_t((seed | 1) & 0xff), image.scope.value.size());
    for (int i = 0; i < 18; ++i) image.geometry[size_t(i)] = int64_t(seed) * 1000 + i;
    image.geometry[1] = 8;   // two global attention layers at a QSA interval of four
    image.geometry[2] = 4;
    image.layer_lo = 0;
    image.layer_hi = 8;
    image.cvec = true;
    image.live.used = seed;
    for (int c = 0; c < checkpoints; ++c) {
        ConversationCheckpoint checkpoint;
        checkpoint.used = seed + uint64_t(c);
        image.checkpoints.push_back(std::move(checkpoint));
    }
    for (int layer = 0; layer < kv_layers; ++layer) {
        ConversationKv kv;
        kv.format = 0;
        kv.cells = 1;
        kv.heads = 1;
        kv.head_dim = 1;
        kv.page_size = 1;
        kv.pooled_rows = 0;
        kv.idx_dim = 1;
        image.kv.push_back(std::move(kv));
    }
    return image;
}

// A record carrying `whole`'s own ids and image keys and one stage tag's state bytes: what a split moves into a
// later stage's record, so the same cut can be told apart byte for byte across stages.
ConversationCheckpoint split_record(const ConversationCheckpoint& whole, uint8_t tag) {
    ConversationCheckpoint out;
    out.ids = whole.ids;
    out.imgs = whole.imgs;
    out.used = whole.used;
    out.pinned = whole.pinned;
    out.gdn.assign(whole.ids.size(), tag);
    out.ple.assign(whole.ids.size() / 2, uint8_t(tag + 1));
    out.tails.assign(2, uint8_t(tag + 2));
    out.dead.assign(3, uint8_t(tag + 3));
    out.block_pos.assign(whole.ids.size(), uint8_t(tag + 4));
    return out;
}

// One later stage of a native parked split: its own carve and K/V layers, its own running record and, as its own
// checkpoints, one part per primary checkpoint.  `stage_parts` is deliberately never set anywhere here: this is the
// shape `strata serve` captures (SavedConversation::stage_images), where a primary record stays part-free.
SavedConversation split_stage(const SavedConversation& primary, uint8_t tag, int64_t layer_lo, int64_t layer_hi) {
    SavedConversation stage;
    stage.scope = primary.scope;
    stage.geometry = primary.geometry;
    stage.cvec = primary.cvec;
    stage.layer_lo = layer_lo;
    stage.layer_hi = layer_hi;
    stage.live = split_record(primary.live, tag);
    stage.checkpoints.reserve(primary.checkpoints.size());
    for (const auto& checkpoint : primary.checkpoints) stage.checkpoints.push_back(split_record(checkpoint, tag));
    stage.kv.reserve(primary.kv.size());
    for (const auto& layer : primary.kv) {
        ConversationKv copy = layer;
        const uint8_t fill_k = uint8_t(tag + 8), fill_v = uint8_t(tag + 9);
        copy.k.visit(0, copy.k.size(), [&](uint8_t* bytes, size_t count, size_t) {
            std::memset(bytes, fill_k, count);
            return true;
        });
        copy.v.visit(0, copy.v.size(), [&](uint8_t* bytes, size_t count, size_t) {
            std::memset(bytes, fill_v, count);
            return true;
        });
        stage.kv.push_back(std::move(copy));
    }
    return stage;
}

// A native parked split image of `stages` stages with two cuts - a root at a quarter of the prompt and the newest at
// half - and one saved image per later stage holding both, as that stage's own records.
SavedConversation split_fixture(uint64_t seed, int64_t tokens, size_t stages) {
    SavedConversation image = fixture(seed, tokens);
    image.geometry[1] = int64_t(4 * stages);
    image.kv.resize(1);  // One attention layer per four-layer stage carve.
    image.layer_hi = 4;
    image.checkpoints.clear();
    for (int64_t at : {tokens / 4, tokens / 2}) {
        ConversationCheckpoint cut;
        cut.ids.assign(image.live.ids.begin(), image.live.ids.begin() + long(at));
        cut.used = seed + uint64_t(at);
        cut.gdn.assign(size_t(at), uint8_t(seed + 5));
        cut.tails.assign(2, uint8_t(seed + 6));
        cut.block_pos.assign(size_t(at), uint8_t(seed + 7));
        image.checkpoints.push_back(std::move(cut));
    }
    for (size_t k = 1; k < stages; ++k) {
        const uint8_t tag = uint8_t(0x20 * k);
        image.stage_images.push_back(split_stage(image, tag, int64_t(4 * k), int64_t(4 * k + 4)));
    }
    return image;
}

// Every regular file under `root`, by relative path: what a transaction may not delete, and what it must not lose.
std::vector<std::string> file_inventory(const std::string& root) {
    std::vector<std::string> out;
    const std::function<void(const std::string&)> walk = [&](const std::string& path) {
        DIR* dir = ::opendir(path.c_str());
        if (!dir) return;
        while (struct dirent* entry = ::readdir(dir)) {
            const std::string name = entry->d_name;
            if (name == "." || name == "..") continue;
            const std::string child = path + "/" + name;
            struct stat info {};
            if (::stat(child.c_str(), &info) != 0) continue;
            if (S_ISDIR(info.st_mode))
                walk(child);
            else if (S_ISREG(info.st_mode))
                out.push_back(child.substr(root.size()));
        }
        ::closedir(dir);
    };
    walk(root);
    std::sort(out.begin(), out.end());
    return out;
}

bool inventory_contains(const std::vector<std::string>& inventory, const std::string& entry) {
    return std::binary_search(inventory.begin(), inventory.end(), entry);
}

// The entry names of one directory, sorted: used to identify the file a publication just wrote.
std::vector<std::string> directory_names(const std::string& path) {
    std::vector<std::string> out;
    DIR* dir = ::opendir(path.c_str());
    if (!dir) return out;
    while (struct dirent* entry = ::readdir(dir)) {
        const std::string name = entry->d_name;
        if (name == "." || name == "..") continue;
        out.push_back(name);
    }
    ::closedir(dir);
    std::sort(out.begin(), out.end());
    return out;
}

// One probe of a publication admission against one published allowance.  Every probe uses its own root and its own
// store: `attempt` seeds whatever state the publication needs (a donor, a predecessor) and returns the request, so no
// probe can drift with another one's commits.  The result is the admission class the store chose, which is what the
// boundary checks assert on.
ConversationStorePublishAdmission publication_admission_probe(
    const ConversationStoreOptions& shape, const SavedConversation& image,
    const std::function<ConversationStorePublish(ConversationStore&)>& attempt, uint64_t allowance) {
    TempRoot root;
    ConversationStoreOptions options = shape;
    options.root = root.path;
    ConversationStore store(options);
    if (!store.enabled()) return kRefused;
    store.publish_read_allowance(allowance);
    const ConversationStorePublish request = attempt(store);
    auto shared = std::make_shared<SavedConversation>(image);
    std::string why;
    const ConversationStorePublishAdmission admission = store.publish(shared, request, why);
    if (admission == kAccepted) {
        ConversationStorePublishResult result;
        store.wait_publish(result, 5000);
    }
    store.close();
    return admission;
}

// Whether this shape, image and allowance admit the publication at all: only an `accepted` admission transfers the
// image's accounting to the store.
bool publication_is_admitted(const ConversationStoreOptions& shape, const SavedConversation& image,
                             const std::function<ConversationStorePublish(ConversationStore&)>& attempt,
                             uint64_t allowance) {
    return publication_admission_probe(shape, image, attempt, allowance) == kAccepted;
}

// The smallest published allowance that admits one publication: the admission figure is counted and deterministic, so
// its exact boundary is observable by bisection.
uint64_t publication_admission_floor(const ConversationStoreOptions& shape, const SavedConversation& image,
                                     const std::function<ConversationStorePublish(ConversationStore&)>& attempt,
                                     uint64_t high) {
    uint64_t low = 0;
    while (low + 1 < high) {
        const uint64_t mid = low + (high - low) / 2;
        if (publication_is_admitted(shape, image, attempt, mid))
            high = mid;
        else
            low = mid;
    }
    return high;
}

// ---------------------------------------------------------------------------------------------------------------
// Checks
// ---------------------------------------------------------------------------------------------------------------

void check_crypto();
void check_layout();
void check_identity();
void check_manifest();
void check_round_trip();
void check_profile_recovery();
void check_full_catalog_profile_restart();
void check_split_pinned_restart();
void check_dedup_beyond_the_quota();
void check_dense_manifest_workspace_refusal();
void check_damaged_manifest_is_still_reclaimed();
void check_multi_stage_restore();
void check_native_split_restore();
void check_request_specific();
void check_incoming_lifetime();
void check_publication_rules();
void check_restart_generation_ordering();
void check_pinned_branch();
void check_reuse();
void check_leases_and_budget();
void check_tight_ram();
void check_cancel_and_release();
void check_read_timing();
void check_revoked_prefetch_demand();
void check_active_prefetch_replacement();
void check_prefetch_catalog_refresh();
void check_prefetch_catalog_reuse_and_cancel();
void check_epoch_and_intent();
void check_dirty_close();
void check_ownership_and_close();
void check_stale_ready();
void check_capacity_floor();
void check_quota_credit();
void check_stream_identity();
void check_corruption();
void check_writer_workspace();
void check_reuse_reservation();
void check_many_stream_reuse_admission();
void check_reuse_multiplicity_reservation();
void check_ready_speculation_yields();
void check_active_reader_defers_publication();
void check_permanent_admission_boundaries();
void check_catalog_headroom();
void check_commit_outcomes();
void check_bootstrap_and_intent();
void check_inflight_object_lease();
void check_failed_write_charges_once();
void check_postcommit_reclamation_cannot_undo_commit();

} // namespace

int main() {
#ifndef __linux__
    std::printf("conversation_store_test: the store is POSIX-only, skipped here\n");
    return 77;
#else
    check_crypto();
    check_layout();
    check_identity();
    check_manifest();
    check_round_trip();
    check_profile_recovery();
    check_full_catalog_profile_restart();
    check_split_pinned_restart();
    check_dedup_beyond_the_quota();
    check_dense_manifest_workspace_refusal();
    check_damaged_manifest_is_still_reclaimed();
    check_multi_stage_restore();
    check_native_split_restore();
    check_request_specific();
    check_incoming_lifetime();
    check_publication_rules();
    check_restart_generation_ordering();
    check_pinned_branch();
    check_reuse();
    check_leases_and_budget();
    check_tight_ram();
    check_cancel_and_release();
    check_revoked_prefetch_demand();
    check_read_timing();
    check_active_prefetch_replacement();
    check_prefetch_catalog_refresh();
    check_prefetch_catalog_reuse_and_cancel();
    check_epoch_and_intent();
    check_dirty_close();
    check_ownership_and_close();
    check_stale_ready();
    check_capacity_floor();
    check_quota_credit();
    check_stream_identity();
    check_corruption();
    check_writer_workspace();
    check_reuse_reservation();
    check_many_stream_reuse_admission();
    check_reuse_multiplicity_reservation();
    check_ready_speculation_yields();
    check_active_reader_defers_publication();
    check_permanent_admission_boundaries();
    check_catalog_headroom();
    check_commit_outcomes();
    check_bootstrap_and_intent();
    check_inflight_object_lease();
    check_failed_write_charges_once();
    check_postcommit_reclamation_cannot_undo_commit();
    std::printf("conversation_store_test: %d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
#endif
}

namespace {

void check_crypto() {
    check(sha256_hex(sha256("abc", 3)) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
          "sha256(abc) matches the published vector");
    check(sha256_hex(sha256("", 0)) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
          "sha256(empty) matches the published vector");
    Digest parsed{};
    check(conversation_store_detail::sha256_hex_parse(sha256_hex(sha256("abc", 3)), parsed) &&
              parsed == sha256("abc", 3),
          "the hex form round-trips");
    check(!conversation_store_detail::sha256_hex_parse("zz", parsed), "a malformed hex string is rejected");
    check(conversation_store_detail::digest_nonzero(sha256("abc", 3)), "a real digest is non-zero");

    // The canonical builder is what a caller uses to digest its resolved execution configuration.
    ConversationStoreDigestBuilder builder;
    builder.add_u64(7);
    builder.add_str("stage");
    builder.add_bytes(nullptr, 0);
    const auto built = builder.finish();
    ConversationStoreDigestBuilder other;
    other.add_u64(7);
    other.add_str("stage");
    other.add_bytes(nullptr, 0);
    check(built == other.finish(), "the canonical builder is deterministic");
    ConversationStoreDigestBuilder split;
    split.add_u64(7);
    split.add_str("sta");
    split.add_str("ge");
    check(built != split.finish(), "a different field split cannot collide");
    check(conversation_store_domain_digest(nullptr, 0) == sha256("", 0),
          "the raw digest entry point accepts an empty sequence");
}

void check_layout() {
    const SavedConversation image = fixture(11, 64);
    const std::vector<ConversationStoreStreamInfo> streams = conversation_store_streams(image);
    check_eq(streams.size(), 7 + 7 + 2 * 5, "one image with one checkpoint has the expected stream count");
    check(conversation_store_streams(image).size() == streams.size(), "the enumeration is deterministic");
    check(!streams.empty() && streams.front().id.kind == uint16_t(ConversationStoreStreamKind::ids) &&
              streams.front().index == 0,
          "the first stream is the live token ids");
    uint64_t total = 0;
    for (const auto& stream : streams) total += stream.bytes;
    check_eq(total, image_payload_bytes(image), "the stream lengths add up to the payload byte count");
    check(image_payload_bytes(image) > 0, "the payload is not empty");

    conversation_store_detail::ImageLayout layout;
    std::string why;
    SavedConversation deeper = fixture(12, 16);
    deeper.stage_images.push_back(fixture(13, 16));
    deeper.stage_images[0].stage_images.push_back(fixture(14, 16));
    check(!conversation_store_detail::image_layout(deeper, Digest{},
                                                  conversation_store_detail::unconstrained_limits(), layout, why),
          "a second level of stage images is refused");
    check(!why.empty(), "the refusal carries a reason");

    SavedConversation part_of_part = fixture(15, 16);
    ConversationCheckpoint part;
    part.ids.assign(4, 9);
    part.stage_parts.push_back(part);
    part_of_part.live.stage_parts.push_back(part);
    check(!conversation_store_detail::image_layout(part_of_part, Digest{},
                                                  conversation_store_detail::unconstrained_limits(), layout, why),
          "a stage part that owns further parts is refused");

    // A scope-less image is refused: an absent identity never authorizes prefix reuse.
    SavedConversation anonymous = fixture(16, 16);
    anonymous.scope = ConversationScope{};
    check(!conversation_store_detail::image_layout(anonymous, Digest{},
                                                  conversation_store_detail::unconstrained_limits(), layout, why),
          "an image without a valid scope is refused");
}

void check_identity() {
    const SavedConversation left = fixture(21, 32);
    SavedConversation other_layout = fixture(21, 32, 1);
    other_layout.geometry[1] = 12;
    other_layout.layer_hi = 12;
    const TempRoot root;
    const ConversationStoreOptions options = options_for(root.path, 1, 0);
    const Digest bucket = conversation_store_detail::store_bucket_digest(options, selector_of(left));
    check(bucket != conversation_store_detail::store_bucket_digest(options, selector_of(other_layout)),
          "a different K/V layer count changes the bucket");
    ConversationStoreSelector other_scope = selector_of(left);
    other_scope.scope.value[0] ^= 0xff;
    check(bucket != conversation_store_detail::store_bucket_digest(options, other_scope),
          "a different authorized scope changes the bucket");

    conversation_store_detail::KvMeta kv;
    kv.format = 0;
    kv.cells = 32;
    kv.heads = 2;
    kv.head_dim = 8;
    const Digest scope =
        conversation_store_detail::object_scope_digest(
            uint16_t(ConversationStoreStreamKind::kv_k), &kv, left.scope, options.model);
    const uint8_t payload[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    const Digest first = conversation_store_detail::object_digest(
        scope, uint16_t(ConversationStoreStreamKind::kv_k), 0, sizeof payload, payload);
    conversation_store_detail::KvMeta other = kv;
    other.heads = 3;
    const Digest different = conversation_store_detail::object_digest(
        conversation_store_detail::object_scope_digest(
            uint16_t(ConversationStoreStreamKind::kv_k), &other, left.scope, options.model),
        uint16_t(ConversationStoreStreamKind::kv_k), 0, sizeof payload, payload);
    check(first != different, "a different K/V interpretation is a different object");
    check(first != conversation_store_detail::object_digest(
                         scope, uint16_t(ConversationStoreStreamKind::kv_k), 1, sizeof payload, payload),
          "a different layer ordinal is a different object");
    check(first != conversation_store_detail::object_digest(
                         scope, uint16_t(ConversationStoreStreamKind::kv_v), 0, sizeof payload, payload),
          "a different state kind is a different object");
    other = kv;
    other.cells *= 2;
    other.pooled_rows += 16;
    check(scope == conversation_store_detail::object_scope_digest(
                       uint16_t(ConversationStoreStreamKind::kv_k), &other, left.scope, options.model),
          "growing a snapshot does not rename an unchanged prefix chunk");
    check(scope != conversation_store_detail::object_scope_digest(
                       uint16_t(ConversationStoreStreamKind::kv_k), &kv, other_scope.scope, options.model),
          "different authorized conversations cannot share physical objects");
    check(scope != conversation_store_detail::object_scope_digest(
                       uint16_t(ConversationStoreStreamKind::kv_k), &kv, left.scope, bytes_from("other-model")),
          "different models cannot share physical objects");
}

void check_manifest() {
    conversation_store_detail::ManifestMeta meta;
    meta.bucket = bytes_from("bucket");
    meta.domain = bytes_from("domain");
    meta.epoch = 3;
    meta.generation = 9;
    meta.stages = 1;
    conversation_store_detail::ContainerMeta container;
    container.scope = fixture(31, 8).scope;
    container.record.tokens = 8;
    meta.containers.push_back(container);
    conversation_store_detail::StreamRecord stream;
    stream.id.stage = 0;
    stream.id.kind = uint16_t(ConversationStoreStreamKind::ids);
    stream.bytes = 32;
    conversation_store_detail::ChunkRecord chunk;
    chunk.offset = 0;
    chunk.length = 32;
    chunk.object = bytes_from("object");
    stream.chunks.push_back(chunk);
    meta.streams.push_back(stream);
    for (uint16_t kind = uint16_t(ConversationStoreStreamKind::imgs);
         kind <= uint16_t(ConversationStoreStreamKind::live_block_pos); ++kind) {
        conversation_store_detail::StreamRecord empty;
        empty.id.kind = kind;
        meta.streams.push_back(std::move(empty));
    }
    meta.objects.emplace_back(chunk.object, 32);
    meta.payload_bytes = 32;
    const std::vector<uint8_t> encoded = conversation_store_detail::manifest_encode(meta);
    // The writer reserves a container's capacity from the counted size without encoding it, and the store's whole
    // catalog-copy headroom is derived from the same kind of figure: a counted size must equal the encoded one, or a
    // reservation would be built on a lie.
    check_eq(uint64_t(conversation_store_detail::manifest_encoded_size(meta)), uint64_t(encoded.size()),
             "the counted manifest size is exactly the encoded size");
    for (int variant = 0; variant < 3; ++variant) {
        conversation_store_detail::ManifestMeta other = meta;
        if (variant == 1) {
            other.containers.clear();
            other.streams.clear();
            other.objects.clear();
            other.cuts.clear();
        }
        if (variant == 2) {
            for (int i = 0; i < 3; ++i) {
                conversation_store_detail::CutRecord cut;
                cut.tokens = uint64_t(i + 1);
                cut.digest = bytes_from("cut");
                other.cuts.push_back(cut);
            }
            for (int i = 0; i < 4; ++i) {
                conversation_store_detail::StreamRecord copy = stream;
                copy.chunks.push_back(chunk);
                other.streams.push_back(std::move(copy));
            }
            for (int i = 0; i < 5; ++i) other.objects.emplace_back(bytes_from("object" + std::to_string(i)), 32);
        }
        const std::vector<uint8_t> variant_bytes = conversation_store_detail::manifest_encode(other);
        check_eq(uint64_t(conversation_store_detail::manifest_encoded_size(other)), uint64_t(variant_bytes.size()),
                 "the counted manifest size is exact for every layout");
    }
    conversation_store_detail::ManifestMeta decoded;
    std::string why;
    check(conversation_store_detail::manifest_decode(encoded.data(), encoded.size(),
                                                    conversation_store_detail::unconstrained_limits(), decoded, why),
          "a manifest round-trips");
    check_eq(decoded.epoch, 3, "the epoch survives the round trip");
    check_eq(decoded.generation, 9, "the generation survives the round trip");
    check_eq(decoded.payload_bytes, 32, "the payload byte count survives the round trip");
    check_eq(decoded.objects.size(), 1, "the object table survives the round trip");
    check(decoded.manifest_id == conversation_store_detail::manifest_id_of(encoded.data(), encoded.size()),
          "the decoded manifest carries its own content address");
    std::vector<uint8_t> damaged = encoded;
    damaged[damaged.size() / 2] ^= 0x40;
    check(!conversation_store_detail::manifest_decode(damaged.data(), damaged.size(),
                                                     conversation_store_detail::unconstrained_limits(), decoded, why),
          "a flipped byte fails the manifest checksum");
    // A stream that does not cover itself contiguously is refused even with a valid checksum.
    conversation_store_detail::ManifestMeta broken = meta;
    broken.streams[0].chunks[0].offset = 1;
    const std::vector<uint8_t> broken_bytes = conversation_store_detail::manifest_encode(broken);
    check(!conversation_store_detail::manifest_decode(broken_bytes.data(), broken_bytes.size(),
                                                     conversation_store_detail::unconstrained_limits(), decoded, why),
          "a non-contiguous chunk list is refused");
    broken = meta;
    broken.containers[0].record.parts = UINT32_MAX;
    const auto excessive_parts = conversation_store_detail::manifest_encode(broken);
    check(!conversation_store_detail::manifest_decode(excessive_parts.data(), excessive_parts.size(),
                                                     conversation_store_detail::unconstrained_limits(), decoded, why),
          "a checksummed manifest cannot allocate undeclared stage parts");

    // The catalog is the authority and round-trips the same way.
    conversation_store_detail::Catalog catalog;
    catalog.issuer_key_digest = bytes_from("key");
    catalog.published_order_counter = 4;
    catalog.domains.push_back(conversation_store_detail::CatalogDomain{bytes_from("domain"), 2});
    conversation_store_detail::CatalogHead head;
    head.head_id = 77;
    head.manifest_id = bytes_from("manifest");
    head.bucket = bytes_from("bucket");
    head.domain = 0;
    head.epoch = 2;
    head.payload_bytes = 128;
    head.published_order = 4;
    head.stages = 1;
    conversation_store_detail::CutRecord cut;
    cut.tokens = 12;
    cut.digest = bytes_from("cut");
    cut.live = true;
    head.cuts.push_back(cut);
    catalog.heads.push_back(head);
    const std::vector<uint8_t> catalog_bytes = conversation_store_detail::catalog_encode(catalog);
    check_eq(uint64_t(conversation_store_detail::catalog_encoded_size(catalog)), uint64_t(catalog_bytes.size()),
             "the counted catalog size is exactly the encoded size");
    {
        conversation_store_detail::Catalog other = catalog;
        other.domains.push_back(conversation_store_detail::CatalogDomain{bytes_from("other"), 7});
        for (int i = 0; i < 4; ++i) {
            conversation_store_detail::CatalogHead extra = head;
            extra.head_id = 100 + uint64_t(i);
            extra.cuts.clear();
            for (int c = 0; c < 3; ++c) {
                conversation_store_detail::CutRecord extra_cut;
                extra_cut.tokens = uint64_t(c + 1);
                extra_cut.digest = bytes_from("cut");
                extra.cuts.push_back(extra_cut);
            }
            other.heads.push_back(std::move(extra));
        }
        const std::vector<uint8_t> other_bytes = conversation_store_detail::catalog_encode(other);
        check_eq(uint64_t(conversation_store_detail::catalog_encoded_size(other)), uint64_t(other_bytes.size()),
                 "the counted catalog size is exact as the catalog grows");
        conversation_store_detail::Catalog empty;
        empty.issuer_key_digest = bytes_from("key");
        const std::vector<uint8_t> empty_bytes = conversation_store_detail::catalog_encode(empty);
        check_eq(uint64_t(conversation_store_detail::catalog_encoded_size(empty)), uint64_t(empty_bytes.size()),
                 "the counted catalog size is exact for an empty catalog");
    }
    conversation_store_detail::Catalog restored;
    check(conversation_store_detail::catalog_decode(catalog_bytes.data(), catalog_bytes.size(), restored, why),
          "a catalog round-trips");
    check_eq(restored.heads.size(), 1, "the catalog keeps its head");
    check(restored.heads[0].manifest_id == head.manifest_id, "the catalog keeps the manifest identity");
    std::vector<uint8_t> catalog_damaged = catalog_bytes;
    catalog_damaged[catalog_damaged.size() - 1] ^= 0x01;
    check(!conversation_store_detail::catalog_decode(catalog_damaged.data(), catalog_damaged.size(), restored, why),
          "a flipped byte fails the catalog checksum");
}

// ---------------------------------------------------------------------------------------------------------------
// The store itself
// ---------------------------------------------------------------------------------------------------------------

bool publish_and_wait(ConversationStore& store, const SavedConversation& image, uint64_t request_id,
                      uint64_t generation, ConversationStorePublishResult& out) {
    auto shared = std::make_shared<SavedConversation>(image);
    ConversationStorePublish request;
    request.request_id = request_id;
    request.generation = generation;
    std::string why;
    const ConversationStorePublishAdmission admission = store.publish(shared, request, why);
    if (admission != kAccepted) {
        std::fprintf(stderr, "Publication admission (%s): %s\n",
                     conversation_store_publish_admission_name(admission), why.c_str());
        return false;
    }
    const bool settled = store.wait_publish(out, 5000);
    if (settled && !out.detail.empty()) std::fprintf(stderr, "Publication result: %s\n", out.detail.c_str());
    return settled;
}

void check_multi_stage_restore() {
    for (int scenario = 0; scenario < 5; ++scenario) {
        TempRoot root;
        SavedConversation image = fixture(151, 128);
        SavedConversation later = image;
        later.layer_lo = 4;
        later.kv.erase(later.kv.begin());
        image.layer_hi = 4;
        image.kv.resize(1);
        image.checkpoints[0].stage_parts.push_back(later.checkpoints[0]);
        image.stage_images.push_back(std::move(later));
        if (scenario == 2) image.checkpoints[0].stage_parts[0].ids[0] += 1;
        if (scenario == 3) image.stage_images[0].scope.value[0] ^= 1;
        if (scenario == 4) image.stage_images[0].cvec = !image.cvec;

        ConversationStore store(options_for(root.path, uint64_t(1) << 30, 0));
        ConversationStorePublishResult result;
        check(publish_and_wait(store, image, 1, 1, result), "a split-image publication settles");
        if (scenario >= 2 && result.status == ConversationStorePublishStatus::refused_limits) {
            check_eq(store.counters().heads, 0, "an incoherent split image authorizes no head");
            store.close();
            continue;
        }
        check(result.status == ConversationStorePublishStatus::stored, "the split image is stored");
        if (result.status != ConversationStorePublishStatus::stored) {
            store.close();
            continue;
        }
        ConversationStoreQuery query = query_of(image, 2, 1);
        if (scenario == 1 || scenario == 2) {
            query.tokens = image.checkpoints[0].ids;
            query.tokens.push_back(12345);
        }
        std::string why;
        check(store.demand(query, why), "the split-image demand is accepted");
        ConversationStoreRead read;
        check(store.wait_ready(2, 1, read, 5000), "the split-image demand settles");
        if (scenario < 2) {
            check(read.status == ConversationStoreReadStatus::ready, "a coherent all-stage cut restores");
            check(read.live == (scenario == 0), "the requested live or checkpoint cut is selected");
            check(read.image && images_equal(*read.image, image, why),
                  "all stages and checkpoint parts survive serialization");
        } else {
            check(read.status == ConversationStoreReadStatus::corrupt,
                  "an incoherent stage or selected checkpoint part is refused before handoff");
            check(!read.image, "a refused split image cannot reach GPU restore");
        }
        read.image.reset();
        store.release_read(2, 1);
        store.close();
    }
}

// The native parked split - the shape `strata serve` writes - restores an earlier checkpoint cut: that cut's part
// lives in each later stage's own image, never in the primary record's `stage_parts`.  A later stage that cannot
// hold the cut is either never advertised or refused before handoff, and the live cut stays independent of it.
void check_native_split_restore() {
    for (size_t stages : {size_t(2), size_t(3)}) {
        const SavedConversation image = split_fixture(151, 128, stages);

        // A coherent native split: every stage's own image holds both cuts.
        {
            TempRoot root;
            ConversationStore store(options_for(root.path, uint64_t(1) << 30, 0));
            ConversationStorePublishResult result;
            check(publish_and_wait(store, image, 1, 1, result) &&
                      result.status == ConversationStorePublishStatus::stored,
                  "a native split image is stored");
            check_eq(result.stages, stages, "the stored image keeps every stage");
            check_eq(result.payload_bytes, image_payload_bytes(image), "the stored payload is every stage's own bytes");
            uint64_t request = 10;
            for (const auto& cut : image.checkpoints) {
                ConversationStoreQuery query = query_of(image, request, 1);
                query.tokens = cut.ids;
                query.tokens.push_back(12345);
                std::string why;
                check(store.demand(query, why), "the earlier-cut demand is admitted");
                ConversationStoreRead read;
                check(store.wait_ready(request, 1, read, 5000), "the earlier-cut demand settles");
                check(read.status == ConversationStoreReadStatus::ready,
                      "a checkpoint cut of a native split restores");
                check(!read.live, "the restore reports the selected checkpoint cut, not the live cut");
                check_eq(read.stages, stages, "the restore reports every stage");
                check(read.image && images_equal(*read.image, image, why),
                      "every stage's own image survives the round trip byte for byte");
                store.release_read(request, 1);
                ++request;
            }
            store.close();
        }

        // A torn native split: the last stage has lost the earlier cut and holds another prefix at the newest one.
        {
            TempRoot root;
            ConversationStore store(options_for(root.path, uint64_t(1) << 30, 0));
            SavedConversation torn = split_fixture(151, 128, stages);
            torn.stage_images.back().checkpoints.erase(torn.stage_images.back().checkpoints.begin());
            torn.stage_images.back().checkpoints.back().ids[0] += 1;
            ConversationStorePublishResult result;
            check(publish_and_wait(store, torn, 1, 1, result) &&
                      result.status == ConversationStorePublishStatus::stored,
                  "a torn native split image is storable");
            std::string why;
            uint64_t request = 20;
            // The earlier cut: the last stage holds no record of that length at all, so no lookup may offer it.
            ConversationStoreQuery absent = query_of(torn, request, 1);
            absent.tokens = torn.checkpoints[0].ids;
            absent.tokens.push_back(12345);
            check(store.matches(absent.selector, absent.tokens, absent.images).empty(),
                  "a cut a later stage no longer holds is not advertised");
            check(store.demand(absent, why), "the request no advertisement covers is still admitted");
            ConversationStoreRead absent_read;
            check(store.wait_ready(request, 1, absent_read, 5000), "the unadvertised request settles");
            check(absent_read.status == ConversationStoreReadStatus::missing && !absent_read.image,
                  "a cut a later stage no longer holds is a miss, never a wrong restore");
            store.release_read(request, 1);
            ++request;
            // The newest cut: the last stage holds that length as another prefix, so it is advertised, then refused.
            ConversationStoreQuery other = query_of(torn, request, 1);
            other.tokens = torn.checkpoints.back().ids;
            other.tokens.push_back(12345);
            check(!store.matches(other.selector, other.tokens, other.images).empty(),
                  "a same-length part still advertises the cut it is not part of");
            check(store.demand(other, why), "the mismatched demand is admitted");
            ConversationStoreRead other_read;
            check(store.wait_ready(request, 1, other_read, 5000), "the mismatched demand settles");
            check(other_read.status == ConversationStoreReadStatus::corrupt && !other_read.image,
                  "a later stage holding another prefix is refused before handoff");
            store.release_read(request, 1);
            ++request;
            // The live cut does not depend on any checkpoint part.
            ConversationStoreQuery live = query_of(torn, request, 1);
            check(store.demand(live, why), "the live demand is admitted");
            ConversationStoreRead live_read;
            check(store.wait_ready(request, 1, live_read, 5000), "the live demand settles");
            check(live_read.status == ConversationStoreReadStatus::ready && live_read.live && live_read.image,
                  "the live cut of a torn split still restores");
            check(live_read.image && images_equal(*live_read.image, torn, why),
                  "the live restore is byte-identical to the torn image it was stored from");
            store.release_read(request, 1);
            store.close();
        }
    }
}

void check_round_trip() {
    TempRoot root;
    const SavedConversation image = fixture(41, 256);
    uint64_t head_id = 0;
    {
        ConversationStore store(options_for(root.path, uint64_t(1) << 30, 0));
        check(store.enabled(), "the store opens its root");
        if (!store.enabled()) {
            std::fprintf(stderr, "Store initialization: %s\n", store.disabled_reason());
            return;
        }
        check(store.healthy(), "a freshly opened store is healthy");
        check(store.run_intent_active(), "the store arms its run intent before serving");
        check_eq(store.counters().heads, 0, "a fresh store holds no heads");
        check(store.disk_free_bytes() > 0, "the store samples the filesystem it was given");

        ConversationStorePublishResult result;
        check(publish_and_wait(store, image, 1, 1, result), "the first publication settles");
        check(result.status == ConversationStorePublishStatus::stored, "a new branch is stored");
        check(result.released, "the store released its publication alias");
        check(result.new_bytes > 0, "the publication wrote object bytes");
        const ConversationStoreCounters counted = store.counters();
        check(counted.objects > 0 && counted.object_bytes > 0,
              "the counters report the bytes of the objects the store holds");
        check(counted.object_bytes >= result.new_bytes,
              "the counted object bytes cover the objects this publication wrote");
        check(counted.object_bytes <= store.disk_charged_bytes(),
              "the counted object bytes stay inside every physical byte the store owns");
        check(counted.manifests > 0 && counted.manifest_bytes > 0,
              "the counters report the bytes of the manifests the store holds");
        head_id = result.head_id;
        check(head_id != 0, "the publication has a head identity");
        check_eq(store.ram_lease_bytes(), 0, "no read is leased yet");
        check(store.disk_charged_bytes() > 0, "the store charges what it wrote");
        check(store.disk_available_bytes() < store.disk_quota_bytes(), "the charge is against the shared quota");

        const ConversationStoreQuery query = query_of(image, 7, 1);
        const std::vector<ConversationStoreMatch> found = store.matches(query.selector, query.tokens, query.images);
        check(!found.empty(), "the published cut is matchable");
        if (found.empty()) return;
        check(found[0].head_id == head_id, "the match names the published head");
        check_eq(uint64_t(found[0].tokens), image.live.ids.size(), "the match is the live cut's length");
        check(found[0].live, "the match reports a live cut");
        check(found[0].pinned == false, "the live cut is not pinned");

        std::string why;
        check(store.demand(query, why), "a demand is admitted");
        ConversationStoreRead read;
        check(store.wait_ready(7, 1, read, 5000), "the request-specific wait returns this request's completion");
        check(read.status == ConversationStoreReadStatus::ready, "the read is ready");
        check_eq(read.request_id, 7, "the delivered read is the requested one");
        check(read.live, "the read reports the live cut");
        check(read.image != nullptr, "the read carries an image");
        if (read.image) {
            std::string why2;
            check(images_equal(*read.image, image, why2), "the restored image is byte-identical to the fixture");
        }
        check(store.ram_lease_bytes() > 0, "a handed-out image stays charged");
        const uint64_t held = store.ram_held_bytes();
        check(held >= store.ram_lease_bytes(), "the lease is part of what the store holds");
        store.release_read(7, 1);
        check_eq(store.ram_lease_bytes(), 0, "the release clears the lease");
        check(store.ram_held_bytes() <= held, "the release lowers the store's resident total");

        ConversationStoreQuery miss = query_of(image, 8, 1);
        miss.tokens.assign(16, 3);
        check(store.demand(miss, why), "a request that matches nothing is still admitted");
        ConversationStoreRead miss_read;
        check(store.wait_ready(8, 1, miss_read, 5000), "the miss settles");
        check(miss_read.status == ConversationStoreReadStatus::missing, "an unmatched request misses");

        ConversationStoreManifestInfo info;
        check(store.manifest_info(head_id, info), "the head's catalog metadata is readable");
        check_eq(info.payload_bytes, image_payload_bytes(image), "the catalog records the payload size");
        check(!info.cuts.empty(), "the catalog records the cuts");
        std::vector<ConversationStoreStreamInfo> streams;
        check(store.manifest_streams(head_id, streams), "the head's stream table is readable");
        check_eq(streams.size(), conversation_store_streams(image).size(), "the manifest stream table is complete");
        check(!store.describe().empty(), "describe() reports the open sequence");
        store.close();
        check(!store.enabled(), "close() stops serving");
    }
    {
        ConversationStore store(options_for(root.path, uint64_t(1) << 30, 0));
        check(store.enabled(), "the root reopens after a clean close");
        check(store.run_intent_active(), "a reopened store arms its own intent");
        const ConversationStoreCounters reopened = store.counters();
        check(reopened.objects > 0 && reopened.object_bytes > 0 && reopened.manifest_bytes > 0,
              "a reopened store counts the bytes it inherited from disk");
        const ConversationStoreQuery query = query_of(image, 9, 1);
        const std::vector<ConversationStoreMatch> found = store.matches(query.selector, query.tokens, query.images);
        check(!found.empty(), "the retained head answers a lookup after a clean restart");
        if (found.empty()) check(false, "the retained head survived the close");
        ConversationStoreRead read;
        std::string why;
        check(store.demand(query, why), "a demand is admitted after the restart");
        check(store.wait_ready(9, 1, read, 5000), "the read settles after the restart");
        check(read.status == ConversationStoreReadStatus::ready, "the retained image is restorable");
        check(read.image != nullptr && read.head_id == head_id, "the restart serves the retained head's image");
        store.release_read(9, 1);
        store.close();
    }
}

// One store root, shared by profiles whose *admitted* figures differ the way two serving profiles' do: the same
// execution domain and the same physical quota, but a different stage count, a different checkpoint policy and a
// different image budget.  A record one profile captured must survive every other profile's clean open - retained,
// indexed, still on disk - because those figures describe what a run may serve, not whether a record exists at all.
// A profile that cannot serve one refuses the read and refuses to *create* one of that shape; it deletes nothing,
// reports nothing corrupt, and the profile that wrote the record reads it back byte for byte.  This is the
// regression for the cross-profile data loss where a two-stage profile, opening a root a three-stage profile had
// published into, counted every one of those heads as corrupt and discarded them at startup.
void check_profile_recovery() {
    TempRoot root;
    const uint64_t quota = uint64_t(1) << 30;
    const SavedConversation triple = split_fixture(401, 128, 3);   // the three-stage profile's own image
    const SavedConversation pair = split_fixture(402, 256, 2);     // the two-stage profile's own image
    const uint64_t payload = std::max(image_payload_bytes(triple), image_payload_bytes(pair));
    check(payload > 0, "both fixtures have a payload");
    if (payload == 0) return;
    const uint64_t retained_heads = 2;

    // The profile that writes both records: it serves from a budget that covers them, keeps several checkpoints and
    // splits across the deeper stage count.
    const ConversationStoreLimits wide = profile_limits(3, 6, payload * 2);
    // Three ways a later profile differs, each of which used to be read as corruption:
    //   fewer_stages        a shallower split: the three-stage record is not representable in this run at all;
    //   fewer_checkpoints   a shorter checkpoint policy: the record keeps more checkpoints than this run admits;
    //   smaller_budget      a smaller image budget: the record's image is larger than this run's admitted image.
    const ConversationStoreLimits fewer_stages = profile_limits(2, 6, payload * 2);
    const ConversationStoreLimits fewer_checkpoints = profile_limits(3, 1, payload * 2);
    const ConversationStoreLimits smaller_budget = profile_limits(3, 6, payload / 2);

    std::vector<std::string> inventory;
    std::string catalog_digest;
    uint64_t triple_head = 0, pair_head = 0;
    {
        std::unique_ptr<ConversationStore> store = open_profile(root.path, quota, wide, 0);
        if (!store->enabled()) return;
        ConversationStorePublishResult stored;
        check(publish_and_wait(*store, triple, 1, 1, stored) &&
                  stored.status == ConversationStorePublishStatus::stored,
              "the three-stage profile stores its own image");
        triple_head = stored.head_id;
        check(publish_and_wait(*store, pair, 2, 1, stored) &&
                  stored.status == ConversationStorePublishStatus::stored,
              "the three-stage profile stores the two-stage image as well");
        pair_head = stored.head_id;
        check(triple_head != 0 && pair_head != 0 && triple_head != pair_head, "both records have their own head");
        check_eq(store->counters().heads, retained_heads, "the root holds both records");
        // A clean close: the run intent is cleared and the epoch does not advance, so both records stay eligible for
        // the profiles that follow.
        store->close();
    }
    inventory = file_inventory(root.path);
    catalog_digest = file_digest(root.path + "/catalog");
    check(!inventory.empty(), "the published root holds files");

    // A profile that splits across fewer stages: the three-stage record is foreign to it, and it must be retained
    // rather than retired as corrupt.  Its own two-stage record still restores, so the switch is not merely refusing
    // everything.
    {
        std::unique_ptr<ConversationStore> store = open_profile(root.path, quota, fewer_stages, retained_heads);
        if (!store->enabled()) return;
        const ConversationStoreQuery query = query_of(pair, 21, 1);
        check(!store->matches(query.selector, query.tokens, query.images).empty(),
              "the two-stage record is still indexed under the shallower split");
        ConversationStoreManifestInfo info;
        check(store->manifest_info(triple_head, info) && info.stages == 3,
              "the foreign three-stage head is still catalog-authorized here");
        std::string why;
        check(store->demand(query, why), "the two-stage demand is admitted under the shallower split");
        ConversationStoreRead read;
        check(store->wait_ready(21, 1, read, 5000), "the two-stage demand settles");
        check(read.status == ConversationStoreReadStatus::ready, "a record this profile can serve is served");
        check(read.image && images_equal(*read.image, pair, why), "the served image is byte-identical to its fixture");
        store->release_read(21, 1);
        // The three-stage record is the one this profile cannot serve: it is *refused* as beyond this run's stages -
        // never corrupt, never deleted - and it stays indexed and catalog-authorized above.
        const ConversationStoreQuery foreign = query_of(triple, 22, 1);
        check(store->demand(foreign, why), "the demand for the foreign three-stage record is admitted");
        ConversationStoreRead refused;
        check(store->wait_ready(22, 1, refused, 5000), "the foreign demand settles");
        check(refused.status == ConversationStoreReadStatus::refused_limits,
              "a record this profile cannot represent is refused as such");
        check(refused.image == nullptr, "the refused foreign record hands out no image");
        check_eq(store->counters().reads_corrupt, 0, "a stage-count difference is never reported as corruption");
        check_eq(store->counters().heads, retained_heads, "the refused foreign record is still retained");
        store->close();
    }
    check(file_inventory(root.path) == inventory, "the shallower split deleted and wrote nothing");
    check(file_digest(root.path + "/catalog") == catalog_digest, "the shallower split rewrote no authority");

    // A profile with a shorter checkpoint policy: the record is still indexed and matched, and the read is *refused*
    // as beyond this run's figures - never called corrupt, never deleted.
    {
        std::unique_ptr<ConversationStore> store = open_profile(root.path, quota, fewer_checkpoints, retained_heads);
        if (!store->enabled()) return;
        const ConversationStoreQuery query = query_of(pair, 31, 1);
        check(!store->matches(query.selector, query.tokens, query.images).empty(),
              "a record beyond this run's checkpoint policy stays matchable");
        std::string why;
        check(store->demand(query, why), "the demand for a record beyond this profile's policy is admitted");
        ConversationStoreRead read;
        check(store->wait_ready(31, 1, read, 5000), "the refused demand settles");
        check(read.status == ConversationStoreReadStatus::refused_limits,
              "a record beyond this run's admitted figures is refused as such");
        check(read.image == nullptr, "a refused record hands out no image");
        check_eq(store->counters().reads_corrupt, 0, "a capacity refusal is never reported as corruption");
        check_eq(store->counters().heads, retained_heads, "the refused record is still retained");
        // Publication admission stays strict as well: this run never creates a record of a shape it cannot serve.
        auto image = std::make_shared<SavedConversation>(triple);
        ConversationStorePublish request;
        request.request_id = 32;
        request.generation = 1;
        check(store->publish(image, request, why) == kRefused, "this profile refuses to publish a record it cannot serve");
        check(store->counters().refused_limits > 0, "the refused publication is counted as a limit refusal");
        check_eq(store->counters().heads, retained_heads, "the refused publication authorizes no head");
        store->close();
    }
    check(file_inventory(root.path) == inventory, "the shorter checkpoint policy deleted and wrote nothing");
    check(file_digest(root.path + "/catalog") == catalog_digest, "the shorter checkpoint policy rewrote no authority");

    // A profile with a smaller image budget: the record's image is larger than this run admits, so the read is
    // refused for that reason alone - and the record stays on disk, charged to the shared quota, for a profile that
    // can pay for it.
    {
        std::unique_ptr<ConversationStore> store = open_profile(root.path, quota, smaller_budget, retained_heads);
        if (!store->enabled()) return;
        const ConversationStoreQuery query = query_of(pair, 41, 1);
        std::string why;
        check(store->demand(query, why), "the demand for an oversized record is admitted");
        ConversationStoreRead read;
        check(store->wait_ready(41, 1, read, 5000), "the oversized demand settles");
        check(read.status == ConversationStoreReadStatus::refused_limits,
              "a record larger than this run's image budget is refused as such");
        check_eq(store->counters().reads_corrupt, 0, "an oversized record is never reported as corruption");
        check_eq(store->counters().heads, retained_heads, "the oversized record is still retained");
        check(store->disk_charged_bytes() > 0, "the retained record still holds its charge in the shared quota");
        store->close();
    }
    check(file_inventory(root.path) == inventory, "the smaller budget deleted and wrote nothing");
    check(file_digest(root.path + "/catalog") == catalog_digest, "the smaller budget rewrote no authority");

    {
        auto state_limited = wide;
        state_limited.max_state_bytes = 1;
        auto store = open_profile(root.path, quota, state_limited, retained_heads);
        if (!store->enabled()) return;
        std::string why;
        const auto query = query_of(pair, 45, 1);
        check(store->demand(query, why), "a state-limited demand reaches capacity classification");
        ConversationStoreRead read;
        check(store->wait_ready(45, 1, read, 5000) &&
                  read.status == ConversationStoreReadStatus::refused_limits,
              "a state-only capacity reduction refuses without corruption");
        ConversationStorePublish request;
        request.request_id = 46;
        request.generation = 1;
        check(store->publish(std::make_shared<SavedConversation>(pair), request, why) == kRefused,
              "publication cannot create state this profile refuses to restore");
        check_eq(store->counters().reads_corrupt, 0, "state capacity is not corruption");
        store->close();
    }
    check(file_inventory(root.path) == inventory, "state capacity refusal preserves retained files");
    check(file_digest(root.path + "/catalog") == catalog_digest, "state capacity refusal preserves authority");

    // Back under the profile that wrote both records: they are still there, still matchable, and both restore byte
    // for byte - including the three-stage one every other profile above could not serve.
    {
        std::unique_ptr<ConversationStore> store = open_profile(root.path, quota, wide, retained_heads);
        if (!store->enabled()) return;
        ConversationStoreQuery live = query_of(triple, 51, 1);
        std::vector<ConversationStoreStreamInfo> streams;
        check(store->manifest_streams(triple_head, streams) && !streams.empty(),
              "the three-stage record's stream table is still readable");
        check(!store->matches(live.selector, live.tokens, live.images).empty(),
              "the three-stage record is still indexed by its own profile");
        std::string why;
        check(store->demand(live, why), "the three-stage demand is admitted again");
        ConversationStoreRead read;
        check(store->wait_ready(51, 1, read, 5000), "the three-stage demand settles");
        check(read.status == ConversationStoreReadStatus::ready, "the retained three-stage record restores again");
        check(read.image && images_equal(*read.image, triple, why),
              "the restored three-stage image is byte-identical to the one that was published");
        store->release_read(51, 1);
        ConversationStoreQuery second = query_of(pair, 52, 1);
        check(store->demand(second, why), "the two-stage demand is admitted again");
        check(store->wait_ready(52, 1, read, 5000), "the two-stage demand settles");
        check(read.status == ConversationStoreReadStatus::ready, "the retained two-stage record restores again");
        check(read.image && images_equal(*read.image, pair, why),
              "the restored two-stage image is byte-identical to the one that was published");
        store->release_read(52, 1);
        ConversationStoreManifestInfo info;
        check(store->manifest_info(triple_head, info) && info.stages == 3 &&
                  info.cuts.size() == 1 + triple.checkpoints.size(),
              "the three-stage head is still catalog-authorized with every cut it advertised");
        store->close();
    }
    check(file_inventory(root.path) == inventory, "the original profile's own open changed nothing");
    check(file_digest(root.path + "/catalog") == catalog_digest, "the original profile's own open rewrote no authority");
}

// ---------------------------------------------------------------------------------------------------------------
// Retention: what a record's own bytes say about it, as opposed to what this run admitted or what the device holds
// ---------------------------------------------------------------------------------------------------------------

// An image whose K/V state is long runs of identical bytes.  Under a 1 MiB chunk grid every chunk of such a run has the
// same content, so it has the same content address and the publication writes one object for all of them: the record's
// *logical* bytes are then far larger than the bytes it costs on the device, which is exactly why a retention ceiling
// derived from the physical quota would be wrong (see conversation_store_detail::retention_limits).
SavedConversation repeated_state_fixture(uint64_t seed, int64_t tokens, size_t state_bytes) {
    SavedConversation image = fixture(seed, tokens);
    image.kv.resize(1);
    image.geometry[1] = image.layer_hi = 4;  // one global attention layer
    ConversationKv& layer = image.kv[0];
    // Two heads of eight values per cell, so both extents below are coherent with the layout the metadata declares.
    layer.cells = int64_t(state_bytes / 16);
    layer.heads = 2;
    layer.head_dim = 8;
    layer.k.resize(state_bytes, uint8_t(seed));
    layer.v.resize(state_bytes, uint8_t(seed + 1));
    layer.k_scale.resize(8, uint8_t(seed + 2));
    layer.v_scale.resize(8, uint8_t(seed + 3));
    return image;
}

// A record whose logical payload exceeds the whole physical quota is still a sound record, and the run that opens the
// root next must keep it whatever its own admitted budget happens to be: a smaller budget refuses the read, it never
// deletes the record.
void check_dedup_beyond_the_quota() {
    TempRoot root;
    const uint64_t quota = uint64_t(4) << 20;        // the whole device allowance
    const size_t state_bytes = size_t(3) << 20;      // three identical 1 MiB chunks in each state extent
    const SavedConversation image = repeated_state_fixture(451, 4096, state_bytes);
    const uint64_t payload = image_payload_bytes(image);
    check(payload > quota, "the fixture's logical payload exceeds the whole physical quota");
    if (payload <= quota) return;

    ConversationStoreOptions options = options_for(root.path, quota, 0);
    options.chunk_bytes = uint64_t(1) << 20;
    // The profile that writes it serves from a budget that covers what the record declares.
    const ConversationStoreLimits wide = profile_limits(1, 2, payload * 2);
    uint64_t head = 0;
    {
        ConversationStore store(options, wide);
        if (!store.enabled()) return;
        ConversationStorePublishResult stored;
        check(publish_and_wait(store, image, 1, 1, stored) &&
                  stored.status == ConversationStorePublishStatus::stored,
              "a record whose logical bytes exceed the quota is published inside it");
        head = stored.head_id;
        check(store.disk_charged_bytes() < quota, "the record's physical charge stays inside the shared quota");
        check(store.counters().dedup_chunks > 0, "its repeated chunks were deduplicated onto one object");
        store.close();
    }
    const std::vector<std::string> inventory = file_inventory(root.path);
    const std::string catalog_digest = file_digest(root.path + "/catalog");
    check(head != 0 && !inventory.empty(), "the published root holds the record");

    // A profile whose admitted budget is a third of the record's own: it cannot serve it, and that is all it may
    // conclude.  A retention ceiling taken from the quota would call this record corrupt and retire it here, taking
    // its manifest and every object only it names.
    const ConversationStoreLimits narrow = profile_limits(1, 2, uint64_t(2) << 20);
    check(narrow.max_image_bytes < payload, "the narrower profile's budget is below the record's payload");
    {
        std::unique_ptr<ConversationStore> store = open_profile(root.path, quota, narrow, 1);
        if (!store->enabled()) return;
        ConversationStoreQuery query = query_of(image, 21, 1);
        std::string why;
        check(store->demand(query, why), "the demand for a record beyond this profile's budget is admitted");
        ConversationStoreRead read;
        check(store->wait_ready(21, 1, read, 5000), "the oversized-state demand settles");
        check(read.status == ConversationStoreReadStatus::refused_limits,
              "a record whose logical state exceeds this run's budget is refused, never served");
        check(read.image == nullptr, "the refused record hands out no image");
        check_eq(store->counters().reads_corrupt, 0, "a quota-independent record is never reported as corruption");
        check_eq(store->counters().heads, 1, "the record is still retained under the smaller budget");
        store->close();
    }
    check(file_inventory(root.path) == inventory, "the smaller budget deleted and wrote nothing");
    check(file_digest(root.path + "/catalog") == catalog_digest, "the smaller budget rewrote no authority");

    // Back under the profile that wrote it: the record is still there and still restores byte for byte.
    {
        std::unique_ptr<ConversationStore> store = open_profile(root.path, quota, wide, 1);
        if (!store->enabled()) return;
        ConversationStoreQuery query = query_of(image, 31, 1);
        std::string why;
        check(store->demand(query, why), "the writing profile's demand is admitted again");
        ConversationStoreRead read;
        check(store->wait_ready(31, 1, read, 5000), "the retained demand settles");
        check(read.status == ConversationStoreReadStatus::ready,
              "the record published inside the physical quota is served again");
        check(read.image && images_equal(*read.image, image, why),
              "the restored repeated-chunk image is byte-identical to the one that was published");
        store->release_read(31, 1);
        store->close();
    }
    check(file_inventory(root.path) == inventory, "the writing profile's own open changed nothing");
    check(file_digest(root.path + "/catalog") == catalog_digest, "the writing profile's own open rewrote no authority");
}

// A record's decoded metadata can be several times its file: a record with thousands of K/V layers and checkpoints
// makes a manifest dominated by its per-element entries, and the tables the consistency pass builds are charged on top
// of them.  This run's preparation ceiling must cover all of it before the decode allocates any of it, and a record
// that does not fit is refused - never deleted, and never the reason a root is called corrupt.
void check_dense_manifest_workspace_refusal() {
    TempRoot root;
    const uint64_t quota = uint64_t(1) << 30;
    // A dense record with a couple of live tokens, so it advertises a cut a read can ask for.
    SavedConversation dense = hollow_fixture(461, 64, 4096);
    dense.geometry[1] = dense.layer_hi = 4 * int64_t(dense.kv.size());
    for (auto& layer : dense.kv) {
        layer.cells = 4;
        layer.k.resize(4, 12);
        layer.v.resize(4, 13);
    }
    dense.live.ids.assign(4, 11);
    dense.live.gdn.resize(4, 12);
    dense.live.block_pos.resize(4, 13);
    dense.live.used = 461;

    ConversationStoreOptions roomy = options_for(root.path, quota, 0);
    roomy.metadata_bytes = uint64_t(64) << 20;
    uint64_t head = 0;
    {
        ConversationStore store(roomy);
        if (!store.enabled()) return;
        ConversationStorePublishResult stored;
        check(publish_and_wait(store, dense, 1, 1, stored) &&
                  stored.status == ConversationStorePublishStatus::stored,
              "the dense record is published under a preparation ceiling that covers its writer");
        head = stored.head_id;
        store.close();
    }
    const std::vector<std::string> inventory = file_inventory(root.path);
    const std::string catalog_digest = file_digest(root.path + "/catalog");
    check(head != 0 && !inventory.empty(), "the published root holds the dense record");

    // A ceiling that holds the store's index and the record's manifest but not its read and decode: the run declines
    // to serve, keeps every byte and every charge where they are, and never reports the record as corrupt.
    {
        ConversationStoreOptions tight = roomy;
        tight.metadata_bytes = uint64_t(1) << 20;
        ConversationStore store(tight);
        check(!store.enabled(), "a run whose preparation ceiling cannot hold the decode refuses to serve the root");
        check_eq(store.counters().corrupt_records, 0, "an unreadable record is never called corrupt");
    }
    check(file_inventory(root.path) == inventory, "the refused open deleted nothing");
    check(file_digest(root.path + "/catalog") == catalog_digest, "the refused open rewrote no authority");

    // The run that can pay for the decode reads the same record back: its stream table and its image both come out.
    {
        ConversationStore store(roomy);
        check(store.enabled(), "a run whose ceiling covers the decode serves the same root");
        check_eq(store.counters().heads, 1, "the dense record is still catalog-authorized");
        std::vector<ConversationStoreStreamInfo> streams;
        check(store.manifest_streams(head, streams) && !streams.empty(),
              "the dense record's stream table is readable under a ceiling that covers the decode");
        ConversationStoreQuery query = query_of(dense, 41, 1);
        std::string why;
        check(store.demand(query, why), "the dense record's demand is admitted");
        ConversationStoreRead read;
        check(store.wait_ready(41, 1, read, 5000), "the dense demand settles");
        check(read.status == ConversationStoreReadStatus::ready, "the dense record is served");
        check(read.image && images_equal(*read.image, dense, why),
              "the restored dense image is byte-identical to the one that was published");
        store.release_read(41, 1);
        store.close();
    }
    check(file_inventory(root.path) == inventory, "the reading run deleted and wrote nothing");
}

// The classification that preserves a record this run cannot read must never excuse one that is really damaged: a
// manifest whose bytes no longer match its checksum is still corrupt, is dropped, and its bytes are reclaimed, while
// every record around it survives unchanged.
void check_damaged_manifest_is_still_reclaimed() {
    TempRoot root;
    const uint64_t quota = uint64_t(1) << 30;
    const SavedConversation survivor = fixture(471, 1024);
    const SavedConversation damaged = fixture(472, 1024);
    std::string damaged_manifest;
    uint64_t damaged_head = 0;
    {
        ConversationStore store(options_for(root.path, quota, 0));
        if (!store.enabled()) return;
        ConversationStorePublishResult stored;
        check(publish_and_wait(store, survivor, 1, 1, stored) &&
                  stored.status == ConversationStorePublishStatus::stored,
              "the record that must survive is stored");
        const std::vector<std::string> before = directory_names(root.path + "/manifests");
        check(publish_and_wait(store, damaged, 2, 1, stored) &&
                  stored.status == ConversationStorePublishStatus::stored,
              "the record that will be damaged is stored");
        damaged_head = stored.head_id;
        const std::vector<std::string> after = directory_names(root.path + "/manifests");
        for (const auto& candidate : after)
            if (!inventory_contains(before, candidate)) damaged_manifest = candidate;
        check(!damaged_manifest.empty(), "the second publication wrote its own manifest");
        store.close();
    }
    const std::string path = root.path + "/manifests/" + damaged_manifest;
    const std::string catalog_before = file_digest(root.path + "/catalog");
    const auto inventory_before = file_inventory(root.path);
    if (::geteuid() != 0) {
        check(::chmod(path.c_str(), 0000) == 0, "the owned manifest can be made temporarily unreadable");
        {
            ConversationStore store(options_for(root.path, quota, 0));
            check(!store.enabled(), "an unreadable manifest disables persistence instead of guessing corruption");
            check_eq(store.counters().corrupt_records, 0, "a permission failure is not corruption");
        }
        check(::chmod(path.c_str(), 0600) == 0, "manifest readability is restored");
        check(file_digest(root.path + "/catalog") == catalog_before, "a read failure preserves catalog authority");
        check(file_inventory(root.path) == inventory_before, "a read failure preserves every file");
        {
            ConversationStore store(options_for(root.path, quota, 0));
            check(store.enabled(), "persistence recovers when the manifest is readable again");
            check_eq(store.counters().heads, 2, "both histories survived the temporary read failure");
            store.close();
        }
    }
    {
        // One flipped byte in the file.  It still reads - it is far inside this run's manifest ceiling - so nothing
        // about it may be excused as capacity this run lacks: it is a defect in the record.
        const int fd = ::open(path.c_str(), O_RDWR | O_CLOEXEC);
        check(fd >= 0, "the manifest opens for damage");
        if (fd >= 0) {
            uint8_t byte = 0;
            check(::read(fd, &byte, 1) == 1, "the manifest is readable");
            byte ^= 0x40;
            check(::lseek(fd, 0, SEEK_SET) == 0, "the manifest seeks to its first byte");
            check(::write(fd, &byte, 1) == 1, "the manifest is damaged");
            ::fsync(fd);
            ::close(fd);
        }
    }
    {
        ConversationStore store(options_for(root.path, quota, 0));
        check(store.enabled(), "a damaged manifest does not disable the store");
        check_eq(store.counters().corrupt_records, 1, "exactly the damaged record is called corrupt");
        check_eq(store.counters().heads, 1, "only the damaged record is dropped");
        ConversationStoreManifestInfo info;
        check(!store.manifest_info(damaged_head, info), "the damaged record is no longer catalog-authorized");
        ConversationStoreQuery query = query_of(survivor, 51, 1);
        std::string why;
        check(store.demand(query, why), "the surviving record's demand is admitted");
        ConversationStoreRead read;
        check(store.wait_ready(51, 1, read, 5000), "the surviving demand settles");
        check(read.status == ConversationStoreReadStatus::ready, "the surviving record is still served");
        check(read.image && images_equal(*read.image, survivor, why),
              "the restored surviving image is byte-identical to the one that was published");
        store.release_read(51, 1);
        store.close();
    }
}

void check_request_specific() {
    TempRoot root;
    // This test exercises the request-specific delivery API with two incoming images at once, so it configures two
    // slots explicitly: the production default is one image through its whole lease (check_incoming_lifetime below).
    ConversationStoreOptions options = options_for(root.path, uint64_t(1) << 30, 0);
    options.max_reads = 2;
    ConversationStore store(options);
    const SavedConversation image = fixture(51, 128);
    ConversationStorePublishResult result;
    check(publish_and_wait(store, image, 1, 1, result) &&
              result.status == ConversationStorePublishStatus::stored,
          "the publication is stored");
    std::string why;
    check(store.demand(query_of(image, 100, 1), why), "the first demand is admitted");
    check(store.demand(query_of(image, 200, 1), why), "the second demand is admitted");
    ConversationStoreRead second;
    check(store.wait_ready(200, 1, second, 5000), "the second request settles");
    check_eq(second.request_id, 200, "the second request receives only its own completion");
    ConversationStoreRead first;
    check(store.wait_ready(100, 1, first, 5000), "the first request settles afterwards");
    check_eq(first.request_id, 100, "the first request receives only its own completion");
    check(first.image != nullptr, "the first request carries an image");
    ConversationStoreRead stranger;
    check(!store.take_ready(999, 1, stranger), "an unknown request takes nothing");
    check(!store.take_ready(200, 1, stranger), "a request that was already delivered takes nothing again");
    store.release_read(100, 1);
    store.release_read(200, 1);
    check_eq(store.ram_lease_bytes(), 0, "both leases are released");
    check(store.request_status(100, 1) == ConversationStoreReadStatus::none,
          "a released read is forgotten rather than reported as live");
    // Unrelated tokens never consume another request's completion.
    ConversationStoreQuery miss = query_of(image, 300, 1);
    miss.tokens.assign(4, 7);
    check(store.demand(miss, why), "an unrelated request is admitted");
    ConversationStoreRead miss_read;
    check(store.wait_ready(300, 1, miss_read, 5000), "the unrelated request settles");
    check(miss_read.status == ConversationStoreReadStatus::missing, "the unrelated request misses");
    check(!store.take_ready(100, 1, stranger), "a settled miss does not satisfy a different request");
    store.close();
}

void check_publication_rules() {
    TempRoot root;
    ConversationStore store(options_for(root.path, uint64_t(1) << 30, 0));
    const SavedConversation image = fixture(61, 192);
    ConversationStorePublishResult first;
    check(publish_and_wait(store, image, 1, 1, first) &&
              first.status == ConversationStorePublishStatus::stored,
          "the first branch is stored");

    // A covered publication of the same lineage replaces its predecessor.
    auto newer = std::make_shared<SavedConversation>(image);
    ConversationStorePublish covered;
    covered.request_id = 2;
    covered.generation = 2;
    covered.supersedes = first.head_id;
    covered.cover = true;
    std::string why;
    check(store.publish(newer, covered, why) == kAccepted, "a covered publication of the same lineage is admitted");
    ConversationStorePublishResult replaced;
    check(store.wait_publish(replaced, 5000), "the covered publication settles");
    check(replaced.status == ConversationStorePublishStatus::replaced, "a covered predecessor is replaced");
    check_eq(replaced.superseded, first.head_id, "the replaced predecessor is reported");
    check_eq(uint64_t(store.counters().heads), 1, "replacement leaves one head");

    // An independent branch always survives.
    const SavedConversation branch = fixture(62, 192);
    ConversationStorePublishResult branched;
    check(publish_and_wait(store, branch, 3, 3, branched) &&
              branched.status == ConversationStorePublishStatus::stored,
          "an independent branch is stored");
    check_eq(uint64_t(store.counters().heads), 2, "both branches are retained");
    const std::vector<ConversationStoreMatch> both =
        store.matches(selector_of(image), query_of(image, 0, 0).tokens, {});
    check(!both.empty(), "the retained branch still matches");

    // A second image is never admitted while the writer is still serializing the first.
    const SavedConversation big = fixture(71, 8192);
    const SavedConversation other = fixture(72, 8192);
    auto first_image = std::make_shared<SavedConversation>(big);
    ConversationStorePublish request;
    request.request_id = 10;
    request.generation = 1;
    check(store.publish(first_image, request, why) == kAccepted, "the first of two back-to-back publications is admitted");
    auto second_image = std::make_shared<SavedConversation>(other);
    ConversationStorePublish busy;
    busy.request_id = 11;
    busy.generation = 1;
    check(store.publish(second_image, busy, why) == kRefused, "a second image is refused while the writer is busy");
    check(!why.empty(), "the refusal carries a reason");
    check(store.counters().refusals_busy > 0, "the refusal is counted");
    ConversationStorePublishResult drain;
    while (store.wait_publish(drain, 2000)) {
    }
    store.close();
}

// The ordering a restart must respect.  A generation is the store's chronology rather than one process's own count:
// the head a cover names was captured by whichever process wrote it, so a restarted engine that minted from zero
// again would name a figure this domain has already used and the cover of the head it just restored would be refused
// as a stale completion.  The floor it must start above is this domain's own persisted history - and the floor
// relaxes nothing: the numbering that ignores it is still refused, and the same-run ordering is unchanged.
void check_restart_generation_ordering() {
    TempRoot root;
    const SavedConversation restored = fixture(91, 128);
    const SavedConversation sibling = fixture(92, 160);
    uint64_t predecessor = 0;
    {
        ConversationStore store(options_for(root.path, uint64_t(1) << 30, 0));
        ConversationStorePublishResult stored;
        check(publish_and_wait(store, restored, 1, 1, stored) &&
                  stored.status == ConversationStorePublishStatus::stored,
              "the pre-restart head is stored");
        predecessor = stored.head_id;
        ConversationStorePublishResult branch;
        check(publish_and_wait(store, sibling, 2, 2, branch) &&
                  branch.status == ConversationStorePublishStatus::stored,
              "a second pre-restart head is stored");
        check_eq(store.publication_generation_floor(), 2,
                 "the floor is the domain's greatest retained generation");
        store.close();
    }

    {
        ConversationStore store(options_for(root.path, uint64_t(1) << 30, 0));
        check(store.enabled() && store.healthy(), "the domain reopens cleanly after the close");
        check_eq(uint64_t(store.counters().heads), 2, "both pre-restart heads survive the clean close");
        check_eq(store.publication_generation_floor(), 2,
                 "the reopened domain reports the generations it has already written");

        // The numbering a process that restarted from zero would mint: the predecessor's own generation is not newer
        // than itself, so this cover is a stale completion and can never overwrite the head it names.
        auto restart_numbering = std::make_shared<SavedConversation>(appended_turn(restored, 192, 91));
        ConversationStorePublish stale;
        stale.request_id = 1;
        stale.generation = 1;
        stale.supersedes = predecessor;
        stale.cover = true;
        std::string why;
        check(store.publish(restart_numbering, stale, why) == kAccepted, "the stale cover is admitted as a ticket");
        ConversationStorePublishResult refused;
        check(store.wait_publish(refused, 5000), "the stale cover settles");
        check(refused.status == ConversationStorePublishStatus::stale,
              "a cover numbered from zero is still refused as a stale completion");
        check(store.counters().refusals_stale > 0, "the stale refusal is counted");
        check_eq(store.counters().replaced, 0, "a stale completion replaces nothing");

        // Minting strictly above the floor covers the head the reopen restored: the first capture after a restart is
        // admitted, exactly as the engine's own numbering now is.
        auto restarted_cover = std::make_shared<SavedConversation>(appended_turn(restored, 192, 91));
        const uint64_t generation = store.publication_generation_floor() + 1;
        ConversationStorePublish cover;
        cover.request_id = generation;
        cover.generation = generation;
        cover.supersedes = predecessor;
        cover.cover = true;
        check(store.publish(restarted_cover, cover, why) == kAccepted, "a cover minted above the floor is admitted");
        ConversationStorePublishResult replaced;
        check(store.wait_publish(replaced, 5000), "the restarted cover settles");
        check(replaced.status == ConversationStorePublishStatus::replaced,
              "a cover minted above the floor retires the head the restart restored");
        check_eq(replaced.superseded, predecessor, "the retired predecessor is the restored head");
        store.close();
    }

    // The floor is this domain's chronology, never the catalog's: a head another execution domain wrote orders
    // nothing here, so one profile's numbering can never inflate or reorder another's covers.
    {
        ConversationStoreOptions other_options = options_for(root.path, uint64_t(1) << 30, 0);
        other_options.domain = bytes_from("domain:other");
        ConversationStore other(other_options);
        check(other.enabled(), "a second execution domain opens on the same root");
        ConversationStorePublishResult stored;
        check(publish_and_wait(other, fixture(93, 96), 1, 40, stored) &&
                  stored.status == ConversationStorePublishStatus::stored,
              "the other domain writes a far newer generation");
        check_eq(other.publication_generation_floor(), 40, "the other domain's floor is its own");
        other.close();
    }
    {
        ConversationStore again(options_for(root.path, uint64_t(1) << 30, 0));
        check_eq(again.publication_generation_floor(), 3,
                 "another domain's generations never order this domain's covers");
        again.close();
    }
}

void check_reuse() {
    TempRoot root;
    ConversationStore store(options_for(root.path, uint64_t(1) << 30, 0));
    const SavedConversation base = fixture(81, 2048);
    ConversationStorePublishResult stored;
    check(publish_and_wait(store, base, 1, 1, stored) &&
              stored.status == ConversationStorePublishStatus::stored,
          "the base publication is stored");

    // The appended turn grows the real K/V extents as well as the token count.  Unchanged completed prefix bytes must
    // keep their physical identity, so the first immutable chunk of the live ids stream *and* of a K/V stream are
    // adopted instead of rewritten, and the grown image must still restore exactly.
    SavedConversation grown = base;
    for (int64_t t = 2048; t < 3072; ++t) grown.live.ids.push_back(int32_t(81 * 7 + uint64_t(t)));
    grown.live.gdn.resize(grown.live.ids.size(), uint8_t(81));
    grown.live.ple.resize(grown.live.ids.size() / 2, uint8_t(81 + 1));
    grown.live.tails.resize(grown.live.ids.size() / 4, uint8_t(81 + 2));
    grown.live.block_pos.resize(grown.live.ids.size(), uint8_t(81 + 4));
    for (size_t layer = 0; layer < grown.kv.size(); ++layer) {
        grown.kv[layer].cells = grown.live.ids.size();
        grown.kv[layer].k.resize(size_t(grown.live.ids.size()) * 16, uint8_t(81 + layer));
        grown.kv[layer].v.resize(size_t(grown.live.ids.size()) * 16, uint8_t(81 + 10 + layer));
    }
    ConversationStoreReuse ids_interval;
    ids_interval.head_id = stored.head_id;   // the donor is the head the bytes came from, named on its own
    ids_interval.stream = ConversationStoreStreamId{0, uint16_t(ConversationStoreStreamKind::ids), 0, 0};
    ids_interval.bytes = 4096;               // exactly the base stream's first immutable chunk
    ConversationStoreReuse kv_interval;
    kv_interval.head_id = stored.head_id;
    kv_interval.stream = ConversationStoreStreamId{0, uint16_t(ConversationStoreStreamKind::kv_k), 0, 0};
    kv_interval.bytes = 4096;

    auto image = std::make_shared<SavedConversation>(grown);
    ConversationStorePublish request;
    request.request_id = 2;
    request.generation = 2;
    request.reuse.push_back(ids_interval);
    request.reuse.push_back(kv_interval);
    std::string why;
    check(store.publish(image, request, why) == kAccepted, "a publication carrying proven intervals is admitted");
    ConversationStorePublishResult reused;
    check(store.wait_publish(reused, 5000), "the reuse publication settles");
    check(reused.status == ConversationStorePublishStatus::stored, "the reuse publication is stored");
    check_eq(reused.reused_bytes, 8192, "both proven intervals were adopted instead of re-read");
    check(store.counters().reused_chunks >= 2, "the adopted chunks are counted");

    // The restored head must be the one that adopted the proven bytes: take it before any other publication can
    // become the newest candidate for the same tokens.
    ConversationStoreQuery query = query_of(grown, 4, 4);
    check(store.demand(query, why), "the grown image can be demanded");
    ConversationStoreRead read;
    check(store.wait_ready(4, 4, read, 5000), "the grown read settles");
    check(read.status == ConversationStoreReadStatus::ready, "the grown image is restorable");
    check_eq(read.head_id, reused.head_id, "the restored image is the head that adopted the proven bytes");
    if (read.image) {
        std::string why2;
        check(images_equal(*read.image, grown, why2), "the restored grown image matches its fixture");
    }
    store.release_read(4, 4);

    // An interval that is not chunk-aligned is ignored, never guessed at: the store has to read and hash the source
    // bytes itself, and an interval whose donor is unknown is ignored the same way.
    const uint64_t source_before = store.source_bytes_read_for_test();
    auto second = std::make_shared<SavedConversation>(grown);
    ConversationStorePublish unaligned;
    unaligned.request_id = 3;
    unaligned.generation = 3;
    ConversationStoreReuse bad = ids_interval;
    bad.offset = 8;
    unaligned.reuse.push_back(bad);
    ConversationStoreReuse stranger = ids_interval;
    stranger.head_id = 0xdeadbeef;
    unaligned.reuse.push_back(stranger);
    check(store.publish(second, unaligned, why) == kAccepted, "an unusable interval still admits the publication");
    ConversationStorePublishResult ignored;
    check(store.wait_publish(ignored, 5000), "the unaligned publication settles");
    check_eq(ignored.reused_bytes, 0, "an unusable interval is ignored");
    check_eq(ignored.new_bytes, 0, "already deduplicated objects are not written again");
    check(store.source_bytes_read_for_test() > source_before,
          "an ignored interval makes the store read and hash the source bytes");
    store.close();
}

void check_leases_and_budget() {
    TempRoot root;
    ConversationStoreOptions options = options_for(root.path, uint64_t(1) << 30, 0);
    options.read_budget_bytes = 0;  // no read is admitted until the caller publishes an allowance
    ConversationStore store(options);
    const SavedConversation image = fixture(91, 128);
    ConversationStorePublishResult result;
    std::string why;
    ConversationStorePublish blocked;
    blocked.request_id = 1;
    blocked.generation = 1;
    check(store.publish(std::make_shared<SavedConversation>(image), blocked, why) == kRefused,
          "zero shared RAM allowance refuses a publication");
    store.publish_read_allowance(uint64_t(64) << 20);
    check(publish_and_wait(store, image, 1, 1, result) &&
              result.status == ConversationStorePublishStatus::stored,
          "the publication is stored");
    store.publish_read_allowance(0);

    // A prefetch never blocks: with no allowance it is refused rather than left pending forever.
    check(store.prefetch(query_of(image, 301, 1), why), "a prefetch descriptor is accepted");
    ConversationStoreRead prefetched;
    check(store.wait_ready(301, 1, prefetched, 30000), "the prefetch settles");
    check(prefetched.status == ConversationStoreReadStatus::refused_budget,
          "a prefetch without spare allowance is refused");
    check(store.counters().refused_budget > 0, "the budget refusal is counted");

    // A demand waits for the allowance instead of being refused.
    check(store.demand(query_of(image, 300, 1), why), "a demand is admitted without an allowance");
    ConversationStoreRead not_yet;
    check(!store.wait_ready(300, 1, not_yet, 200), "a demand is not admitted without an allowance");
    check_eq(store.publish_read_allowance(uint64_t(64) << 20), uint64_t(64) << 20, "the allowance is absolute");
    ConversationStoreRead read;
    check(store.wait_ready(300, 1, read, 5000), "the demand is admitted once the allowance exists");
    check(read.status == ConversationStoreReadStatus::ready, "the demand completes");
    check(store.ram_lease_bytes() > 0, "the handed-out image is leased");
    check(store.ram_held_bytes() >= store.ram_lease_bytes(), "the lease counts inside the held total");
    store.release_read(300, 1);
    check_eq(store.ram_lease_bytes(), 0, "the lease is released");

    store.suspend_read_admission();
    check(store.ram_available_bytes() <= store.ram_read_budget(), "availability never exceeds the bound");
    check_eq(store.publish_read_allowance(uint64_t(64) << 20), uint64_t(64) << 20, "the allowance stays absolute");

    // A cancelled request is refunded and never delivered.
    const uint64_t before = store.ram_held_bytes();
    check(store.demand(query_of(image, 400, 1), why), "a demand is admitted for cancellation");
    store.cancel_read(400, 1);
    ConversationStoreRead cancelled;
    check(!store.wait_ready(400, 1, cancelled, 200), "a cancelled request is not delivered");
    check(store.ram_held_bytes() <= before, "a cancelled request does not leave a charge behind");
    store.cancel_read(4242, 99);  // unknown identifiers are safe
    store.close();
}

void check_epoch_and_intent() {
    TempRoot root;
    const SavedConversation image = fixture(101, 128);
    uint64_t first_epoch = 0;
    {
        ConversationStore store(options_for(root.path, uint64_t(1) << 30, 0));
        ConversationStorePublishResult result;
        check(publish_and_wait(store, image, 1, 1, result) &&
                  result.status == ConversationStorePublishStatus::stored,
              "the first run stores its branch");
        first_epoch = store.epoch();
        check_eq(first_epoch, 1, "the first run's epoch is one");
        check(store.invalidate_domain("non-finite logits"), "the domain can be invalidated");
        check_eq(store.epoch(), first_epoch + 1, "invalidation advances the epoch");
        check_eq(store.counters().heads, 0, "invalidation retires every record of the old epoch");
        const ConversationStoreQuery query = query_of(image, 2, 1);
        check(store.matches(query.selector, query.tokens, query.images).empty(),
              "an invalidated domain matches nothing");
        // The destructor abandons the intent: the next start of this domain must invalidate it again.
    }
    {
        ConversationStore store(options_for(root.path, uint64_t(1) << 30, 0));
        check(store.enabled(), "the root reopens after an unclean exit");
        check_eq(store.epoch(), first_epoch + 2, "an unclean restart invalidates exactly this domain");
        const ConversationStoreQuery query = query_of(image, 3, 1);
        check(store.matches(query.selector, query.tokens, query.images).empty(),
              "an unclean restart reuses nothing");
        store.close();
        check(!store.run_intent_active(), "a clean close clears the run intent");
    }
    {
        ConversationStore store(options_for(root.path, uint64_t(1) << 30, 0));
        check_eq(store.epoch(), first_epoch + 2, "a clean restart keeps the epoch");
        check(store.run_intent_active(), "the reopened store armed its own intent");
        const SavedConversation second = fixture(102, 128);
        ConversationStorePublishResult result;
        check(publish_and_wait(store, second, 4, 4, result) &&
                  result.status == ConversationStorePublishStatus::stored,
              "the clean run stores its branch");
        store.close();
    }
    {
        ConversationStore store(options_for(root.path, uint64_t(1) << 30, 0));
        check_eq(store.epoch(), first_epoch + 2, "the clean close kept the epoch");
        const SavedConversation second = fixture(102, 128);
        const ConversationStoreQuery query = query_of(second, 5, 5);
        check(!store.matches(query.selector, query.tokens, query.images).empty(),
              "a cleanly closed run's branch survives the restart");
        store.close();
    }
}

void check_capacity_floor() {
    TempRoot root;
    // A floor the device cannot satisfy refuses the publication instead of filling it.
    ConversationStore store(options_for(root.path, uint64_t(1) << 20, uint64_t(1) << 62));
    const SavedConversation image = fixture(111, 256);
    // The shared charge already includes the control files this store created (catalog, issuer key, lock, marker).
    const uint64_t charged_before = store.disk_charged_bytes();
    check(charged_before > 0, "the store charges the control files it owns");
    check(charged_before < store.disk_quota_bytes(), "the control files fit the quota");
    ConversationStorePublishResult result;
    check(publish_and_wait(store, image, 1, 1, result), "the publication is admitted");
    check(result.status == ConversationStorePublishStatus::refused_floor,
          "an impossible filesystem floor refuses the publication");
    check_eq(uint64_t(store.counters().heads), 0, "a refused publication authorizes no head");
    check_eq(store.disk_charged_bytes(), charged_before, "a refused publication leaves nothing charged");
    check(store.counters().refusals_floor > 0, "the floor refusal is counted");
    const ConversationStoreQuery query = query_of(image, 2, 1);
    check(store.matches(query.selector, query.tokens, query.images).empty(),
          "a refused publication stored nothing");
    store.close();
}

void check_corruption() {
    StoreReadTraceCapture trace;
    TempRoot root;
    ConversationStore store(options_for(root.path, uint64_t(1) << 30, 0));
    const SavedConversation image = fixture(121, 128);
    ConversationStorePublishResult result;
    check(publish_and_wait(store, image, 1, 1, result) &&
              result.status == ConversationStorePublishStatus::stored,
          "the publication is stored");

    // Damage one object: the store must refuse the image rather than serve altered bytes.
    std::string victim;
    DIR* shards = ::opendir((root.path + "/objects").c_str());
    check(shards != nullptr, "the object directory exists");
    if (shards) {
        while (struct dirent* shard = ::readdir(shards)) {
            const std::string name = shard->d_name;
            if (name == "." || name == "..") continue;
            DIR* files = ::opendir((root.path + "/objects/" + name).c_str());
            if (!files) continue;
            while (struct dirent* file = ::readdir(files)) {
                const std::string object = file->d_name;
                if (object == "." || object == "..") continue;
                victim = root.path + "/objects/" + name + "/" + object;
                break;
            }
            ::closedir(files);
            if (!victim.empty()) break;
        }
        ::closedir(shards);
    }
    check(!victim.empty(), "at least one object exists to damage");
    if (!victim.empty()) {
        const int fd = ::open(victim.c_str(), O_RDWR | O_CLOEXEC);
        check(fd >= 0, "the object opens for damage");
        if (fd >= 0) {
            uint8_t byte = 0;
            check(::read(fd, &byte, 1) == 1, "the object is readable");
            byte ^= 0x80;
            ::lseek(fd, 0, SEEK_SET);
            check(::write(fd, &byte, 1) == 1, "the object is damaged");
            ::fsync(fd);
            ::close(fd);
        }
    }
    std::string why;
    check(store.demand(query_of(image, 1, 1), why), "a demand is admitted after the damage");
    ConversationStoreRead read;
    check(store.wait_ready(1, 1, read, 5000), "the damaged read settles");
    check(read.status == ConversationStoreReadStatus::corrupt,
          "a damaged object is reported as corrupt, never served");
    check(read.image == nullptr, "a corrupt read hands out no image");
    check(store.counters().reads_corrupt > 0, "the corruption is counted");
    store.close();
    const auto records = trace.records();
    const auto* finish = read_trace(records, "read_finish", 1);
    const auto* settled = read_trace(records, "settled", 1);
    check(finish && finish->text("status") == "corrupt" && finish->number("io_attempted") &&
              finish->number("start_ns") && finish->number("finish_ns") && finish->number("reserved_bytes") &&
              finish->number("holds_slot"), "corruption reports real materialization before refunding its holds");
    check(settled && settled->text("status") == "corrupt" && !settled->number("reserved_bytes") &&
              !settled->number("lease_bytes") && !settled->number("holds_slot"),
          "the corrupt completion releases its reservation and slot without inventing a ready image");
}

// ---------------------------------------------------------------------------------------------------------------
// Boundaries: one incoming image, branch pins, a tight RAM allowance, lease lifetimes, ownership, faulted close,
// stale completions and the physical identity a layer carve maps to
// ---------------------------------------------------------------------------------------------------------------

void check_incoming_lifetime() {
    TempRoot root;
    // The production geometry: one incoming image, held through its whole lease (Options::max_reads defaults to 1).
    ConversationStore store(options_for(root.path, uint64_t(1) << 30, 0));
    const SavedConversation image = fixture(131, 256);
    ConversationStorePublishResult result;
    check(publish_and_wait(store, image, 1, 1, result) &&
              result.status == ConversationStorePublishStatus::stored,
          "the incoming-image test stores its branch");
    std::string why;
    check(store.demand(query_of(image, 11, 1), why), "the first demand is admitted");
    check(await_status(store, 11, 1, ConversationStoreReadStatus::ready),
          "the first request materializes and waits to be taken");
    check(store.ram_lease_bytes() > 0, "an untaken image is charged while it waits");
    check_eq(store.counters().reads_served, 1, "exactly one image has been materialized");

    check(store.demand(query_of(image, 12, 1), why), "a second demand is admitted");
    ::usleep(50000);
    check(store.request_status(12, 1) == ConversationStoreReadStatus::none,
          "the incoming-image slot is held through the lease, so no second image starts");
    check_eq(store.counters().reads_served, 1, "no second image materialized beside the first");

    store.release_read(11, 1);
    check_eq(store.ram_lease_bytes(), 0, "releasing the lease refunds the image");
    ConversationStoreRead second;
    check(store.wait_ready(12, 1, second, 5000), "the second demand starts once the first lease ends");
    check(second.status == ConversationStoreReadStatus::ready, "the second image is ready");
    check(second.image != nullptr, "the second image carries bytes");
    store.release_read(12, 1);
    check_eq(store.ram_lease_bytes(), 0, "the second lease is released");
    store.close();
}

void check_pinned_branch() {
    TempRoot root;
    auto options = options_for(root.path, uint64_t(1) << 30, 0);
    options.head_limit = 2;
    ConversationStore store(options);
    const SavedConversation pinned = fixture(141, 128);
    check(pinned.pinned(), "the fixture carries a pinned cut");
    ConversationStorePublishResult first;
    check(publish_and_wait(store, pinned, 1, 1, first) &&
              first.status == ConversationStorePublishStatus::stored,
          "the pinned branch is stored");

    // A covered descendant that drops the pin must not supersede the pinned predecessor.
    SavedConversation unpinned = pinned;
    for (auto& checkpoint : unpinned.checkpoints) checkpoint.pinned = false;
    auto dropped = std::make_shared<SavedConversation>(unpinned);
    ConversationStorePublish request;
    request.request_id = 2;
    request.generation = 2;
    request.supersedes = first.head_id;
    request.cover = true;
    std::string why;
    check(store.publish(dropped, request, why) == kAccepted, "a covered descendant is admitted");
    ConversationStorePublishResult retained;
    check(store.wait_publish(retained, 5000), "the covered descendant settles");
    check(retained.status == ConversationStorePublishStatus::retained_branch,
          "a cover that loses a pinned cut retains both branches");
    check_eq(store.counters().heads, 2, "the pinned predecessor survives");
    ConversationStoreManifestInfo info;
    check(store.manifest_info(first.head_id, info), "the pinned predecessor is still authorized");

    // A cover that keeps every pinned cut replaces its predecessor and inherits its retained membership.
    auto kept = std::make_shared<SavedConversation>(pinned);
    ConversationStorePublish cover;
    cover.request_id = 3;
    cover.generation = 3;
    cover.supersedes = first.head_id;
    cover.cover = true;
    check(store.publish(kept, cover, why) == kAccepted, "a pin-preserving cover is admitted");
    ConversationStorePublishResult replaced;
    check(store.wait_publish(replaced, 5000), "the pin-preserving cover settles");
    check(replaced.status == ConversationStorePublishStatus::replaced,
          "a cover that preserves every pinned cut replaces its predecessor");
    check_eq(replaced.superseded, first.head_id, "the replaced predecessor is reported");
    check_eq(store.counters().heads, 2, "the retained divergent branch survives the replacement");
    check(store.manifest_info(retained.head_id, info),
          "replacement at the head ceiling does not evict the independent branch");

    // An older capture that claims to cover a newer descendant is a stale completion, never an overwrite.
    auto older = std::make_shared<SavedConversation>(pinned);
    ConversationStorePublish stale;
    stale.request_id = 4;
    stale.generation = 2;
    stale.supersedes = replaced.head_id;
    stale.cover = true;
    check(store.publish(older, stale, why) == kAccepted, "the older cover is admitted as a ticket");
    ConversationStorePublishResult refused;
    check(store.wait_publish(refused, 5000), "the older cover settles");
    check(refused.status == ConversationStorePublishStatus::stale,
          "an older capture can never cover a newer descendant");
    check(store.counters().refusals_stale > 0, "the stale refusal is counted");
    store.close();
}

void check_full_catalog_profile_restart() {
    TempRoot root;
    auto options = options_for(root.path, uint64_t(1) << 30, 0);
    options.head_limit = 2;
    const auto limits = profile_limits(3, 2, uint64_t(64) << 20);
    const auto pair = split_fixture(171, 128, 2);
    const auto triple = split_fixture(173, 128, 3);
    ConversationStorePublishResult pair_head, triple_head, replaced;
    std::string why;
    {
        ConversationStore store(options, limits);
        check(publish_and_wait(store, pair, 1, 1, pair_head), "store the two-stage history");
        check(publish_and_wait(store, triple, 2, 2, triple_head), "store the three-stage history");
        auto image = std::make_shared<SavedConversation>(pair);
        ConversationStorePublish cover;
        cover.request_id = 3;
        cover.generation = 3;
        cover.supersedes = pair_head.head_id;
        cover.cover = true;
        check(store.publish(image, cover, why) == kAccepted, "admit replacement in the full shared catalog");
        check(store.wait_publish(replaced, 5000) &&
                  replaced.status == ConversationStorePublishStatus::replaced,
              "replace the pair without borrowing a third head slot");
        store.close();
    }
    const auto restore = [&](ConversationStore& store, const SavedConversation& expected, uint64_t request) {
        ConversationStoreRead read;
        check(store.demand(query_of(expected, request, 1), why), "admit retained layout after restart");
        check(store.wait_ready(request, 1, read, 5000) && read.image &&
                  images_equal(*read.image, expected, why),
              "restart restores every stage byte of the retained history");
        if (read.image) store.confirmed_restore(read.head_id);
        read.image.reset();
        store.release_read(request, 1);
    };
    {
        ConversationStore store(options, limits);
        restore(store, pair, 10);
        restore(store, triple, 11);
        // A genuinely new branch must still obey the same two-head bound. The pair is older.
        const auto incoming = split_fixture(175, 128, 2);
        ConversationStorePublishResult added;
        check(publish_and_wait(store, incoming, 12, 4, added), "admit a new branch under bounded eviction");
        ConversationStoreManifestInfo info;
        check(!store.manifest_info(replaced.head_id, info), "real head pressure evicts the older unpinned pair");
        check(store.manifest_info(triple_head.head_id, info), "real head pressure preserves the newer triple");
        check_eq(store.counters().heads, 2, "shared catalog stays at its head bound");
        store.close();
    }
    {
        ConversationStore store(options, limits);
        restore(store, triple, 20);
        restore(store, split_fixture(175, 128, 2), 21);
        ConversationStoreManifestInfo info;
        check(!store.manifest_info(replaced.head_id, info), "restart never resurrects the evicted pair");
        store.close();
    }
}

void check_split_pinned_restart() {
    for (size_t stages : {size_t(2), size_t(3)}) {
        TempRoot root;
        const auto options = options_for(root.path, uint64_t(1) << 30, 0);
        const auto limits = profile_limits(3, 3, uint64_t(64) << 20);
        const auto make_pinned = [&](uint64_t seed) {
            auto image = split_fixture(seed, 128, stages);
            image.checkpoints[1].pinned = true;  // root32, pin64, latest96
            auto latest = split_record(image.live, uint8_t(seed + 17));
            latest.ids.resize(96);
            latest.pinned = false;
            image.checkpoints.push_back(std::move(latest));
            image.stage_images.clear();
            for (size_t stage = 1; stage < stages; ++stage)
                image.stage_images.push_back(split_stage(
                    image, uint8_t(seed + 0x20 * stage), int64_t(4 * stage), int64_t(4 * stage + 4)));
            return image;
        };
        const auto first = make_pinned(211);
        auto second = make_pinned(213);
        second.scope = first.scope;
        second.geometry = first.geometry;
        for (auto& stage : second.stage_images) {
            stage.scope = second.scope;
            stage.geometry = second.geometry;
        }
        {
            ConversationStore store(options, limits);
            ConversationStorePublishResult a, b;
            check(publish_and_wait(store, first, 1, 1, a) &&
                      a.status == ConversationStorePublishStatus::stored,
                  "persist the first all-stage pinned history");
            check(publish_and_wait(store, second, 2, 2, b) &&
                      b.status == ConversationStorePublishStatus::stored,
                  "persist an independent pin in the same scope");
            store.close();
        }
        {
            ConversationStore store(options, limits);
            uint64_t request = 10;
            for (const SavedConversation* expected :
                 std::array<const SavedConversation*, 2>{&first, &second}) {
                auto query = query_of(*expected, request, 1);
                query.tokens.resize(64);
                query.tokens.push_back(12345);  // diverges immediately after the pin, before latest/live
                ConversationStoreRead restored;
                std::string why;
                check(store.demand(query, why) && store.wait_ready(request, 1, restored, 5000),
                      "a divergent request can demand its persisted pin after reopening");
                check(restored.status == ConversationStoreReadStatus::ready &&
                          restored.tokens == 64 && !restored.live && restored.stages == stages,
                      "disk lookup selects the exact pinned cut, not the root or latest cut");
                check(restored.image && images_equal(*restored.image, *expected, why),
                      "both layouts preserve every stage's bytes and root/pin/latest metadata across restart");
                restored.image.reset();
                store.release_read(request++, 1);
            }
            store.close();
        }
    }
}

void check_tight_ram() {
    TempRoot root;
    const SavedConversation small = fixture(151, 4096);
    const SavedConversation large = fixture(152, 131072);
    const uint64_t small_payload = image_payload_bytes(small);
    const uint64_t large_payload = image_payload_bytes(large);
    ConversationStoreOptions options = options_for(root.path, uint64_t(1) << 30, 0);
    options.max_reads = 2;   // the allowance, not the incoming-image slot, is what bounds this test
    ConversationStore store(options);
    ConversationStorePublishResult small_result, large_result;
    check(publish_and_wait(store, small, 1, 1, small_result) &&
              small_result.status == ConversationStorePublishStatus::stored,
          "the small image is stored");
    check(publish_and_wait(store, large, 2, 2, large_result) &&
              large_result.status == ConversationStorePublishStatus::stored,
          "the large image is stored");
    ConversationStoreManifestInfo small_info;
    check(store.manifest_info(small_result.head_id, small_info), "the small head's metadata is readable");

    // An allowance that fits one small read (payload, manifest, the job's own query copy, one I/O slab and the fixed
    // scaffolding) *and* everything the store already holds resident - its index, this read's own descriptor copy and
    // any diagnostic decode - and nothing of the large one: a read is admitted against the complete ledger, so the
    // allowance that admits it has to cover the ledger too.
    const uint64_t prep = uint64_t(small.live.ids.size() + 1) * sizeof(int32_t) + 1024;
    const uint64_t resident_before = store.ram_held_bytes();
    const uint64_t allowance = small_payload + small_payload / 4 + 2 * small_info.manifest_bytes + prep +
                               (uint64_t(1) << 20) + (uint64_t(320) << 10) + resident_before;
    check(large_payload > allowance + (uint64_t(1) << 20),
          "the large image is larger than the allowance by construction");
    check_eq(store.publish_read_allowance(allowance), allowance, "the allowance is absolute");

    // A publication image that does not fit is refused instead of overrunning the envelope.
    auto big = std::make_shared<SavedConversation>(large);
    ConversationStorePublish big_request;
    big_request.request_id = 3;
    big_request.generation = 3;
    std::string why;
    check(store.publish(big, big_request, why) == kRefused, "a publication image larger than the allowance is refused");
    check(!why.empty(), "the refusal carries a reason");
    check(store.counters().refused_limits > 0, "the publication refusal is counted");

    // A read whose whole peak cannot fit is refused as a budget miss, so the caller falls back to its own prefill.
    check(!store.demand(query_of(large, 4, 4), why), "a demand that can never fit the allowance is refused");
    check(store.counters().refused_budget > 0, "the refusal is counted as a budget refusal");
    check(store.ram_held_bytes() <= store.ram_read_budget(), "the ledger stays inside the allowance");

    // The small read fits, and the ledger never exceeds what was published.
    check(store.demand(query_of(small, 5, 5), why), "a demand that fits is admitted");
    ConversationStoreRead read;
    check(store.wait_ready(5, 5, read, 5000), "the small image materializes");
    check(read.status == ConversationStoreReadStatus::ready, "the small image is ready");
    check(store.ram_held_bytes() <= store.ram_read_budget(), "held bytes never exceed the published allowance");
    store.release_read(5, 5);
    check(store.ram_held_bytes() <= store.ram_read_budget(), "the released ledger stays inside the allowance");

    // Suspension is serialized with the start of allocation: a queued demand can allocate nothing while admission is
    // suspended, so a caller may read the ledger and publish a new allowance without racing an allocation.
    store.suspend_read_admission();
    check(store.demand(query_of(small, 6, 6), why), "a demand is admitted while admission is suspended");
    const uint64_t held = store.ram_held_bytes();
    ::usleep(50000);
    check_eq(store.ram_held_bytes(), held, "nothing is allocated while admission is suspended");
    check_eq(store.publish_read_allowance(allowance), allowance, "the allowance stays absolute");
    ConversationStoreRead after;
    check(store.wait_ready(6, 6, after, 5000), "the queued demand materializes once admission resumes");
    check(after.status == ConversationStoreReadStatus::ready, "the resumed read is ready");
    store.release_read(6, 6);
    store.close();
}

void check_read_timing() {
    for (bool enabled : {false, true}) {
        StoreReadTraceCapture trace(enabled);
        TempRoot root;
        auto options = options_for(root.path, uint64_t(1) << 30, 0);
        options.max_reads = 2;
        ConversationStore store(options);
        const SavedConversation image = fixture(161, 4096);
        ConversationStorePublishResult published;
        check(publish_and_wait(store, image, 1, 1, published) &&
                  published.status == ConversationStorePublishStatus::stored, "the timed-read fixture is stored");
        std::string why;
        auto missing = query_of(image, 7001, 8);
        missing.selector.scope = bytes_from("private:never-log-this-scope");
        check(store.demand(missing, why), "a timed miss is admitted without materialization");
        ConversationStoreRead miss;
        check(store.wait_ready(7001, 8, miss, 5000) && miss.status == ConversationStoreReadStatus::missing,
              "the timed miss reports its actual no-I/O result");
        store.publish_read_allowance(1);
        check(!store.demand(query_of(image, 7002, 8), why), "an impossible timed read is refused at admission");
        store.publish_read_allowance(uint64_t(64) << 20);
        store.suspend_read_admission();
        check(store.demand(query_of(image, 7003, 8), why), "a suspended timed demand is queued");
        check(store.request_status(7003, 8) == ConversationStoreReadStatus::none,
              "a suspended timed demand has not materialized");
        store.publish_read_allowance(uint64_t(64) << 20);
        check(await_status(store, 7003, 8, ConversationStoreReadStatus::ready), "the resumed timed demand materializes");

        if (enabled) {
            const auto queued_records = trace.records();
            const auto* admission = read_trace(queued_records, "admission", 7003);
            check(admission && (admission->number("wait_mask") & 2) && !admission->number("start_ns"),
                  "the enqueue record identifies suspension before the read starts");
            check(admission && admission->number("call_ns") &&
                      admission->number("call_ns") + admission->number("call_lock_wait_ns") <=
                          admission->number("enqueue_ns"),
                  "native API lock acquisition precedes enqueue and is not charged as queue residence");
            const uint64_t lease = store.ram_lease_bytes();
            check(lease > 32768, "the timed fixture provides meaningful memory pressure");
            const uint64_t peak = admission ? admission->number("peak_bytes") : 0;
            store.publish_read_allowance(peak + store.ram_held_bytes() - lease + lease / 2);
            check(store.demand(query_of(image, 7004, 8), why), "a timed demand can wait for memory with a spare slot");
            const auto memory_records = trace.records();
            const auto* memory = read_trace(memory_records, "admission", 7004);
            check(memory && (memory->number("wait_mask") & 8) && !(memory->number("wait_mask") & 4),
                  "the waiting record separates memory pressure from slot pressure");
            store.publish_read_allowance(uint64_t(64) << 20);
            check(await_status(store, 7004, 8, ConversationStoreReadStatus::ready),
                  "the memory-blocked read runs after its allowance grows");
            check(store.demand(query_of(image, 7005, 8), why), "a third timed demand waits behind two ready leases");
            const auto slot_records = trace.records();
            const auto* slots = read_trace(slot_records, "admission", 7005);
            check(slots && (slots->number("wait_mask") & 4) && !(slots->number("wait_mask") & 16) &&
                      slots->number("incoming") == 2,
                  "ready leases occupy slots without pretending the materialization workers are busy");
        }

        ConversationStoreRead read;
        check(store.wait_ready(7003, 8, read, 5000) && read.image && images_equal(*read.image, image, why),
              "timing does not change the restored bytes");
        read.image.reset();
        store.release_read(7003, 8);
        if (enabled) {
            ConversationStoreRead second, third;
            check(store.wait_ready(7004, 8, second, 5000) && second.image, "the memory waiter remains deliverable");
            check(store.wait_ready(7005, 8, third, 5000) && third.image, "releasing a lease admits the slot waiter");
            second.image.reset();
            third.image.reset();
            store.release_read(7004, 8);
            store.release_read(7005, 8);
            store.set_read_gate_for_test([](void*) { throw std::bad_alloc(); }, nullptr);
            check(store.demand(query_of(image, 7006, 8), why), "an exception-path timed demand is admitted");
            ConversationStoreRead failed;
            check(store.wait_ready(7006, 8, failed, 5000) && failed.status == ConversationStoreReadStatus::failed &&
                      !failed.image, "a real reader exception settles as a failed cache read");
            store.set_read_gate_for_test(nullptr, nullptr);
        }
        store.close();
        const auto records = trace.records();
        if (!enabled) {
            check(records.empty(), "disabled timing produces no native read records");
            continue;
        }
        const auto* miss_record = read_trace(records, "settled", 7001);
        const auto* refusal = read_trace(records, "admission_refused", 7002);
        check(miss_record && miss_record->text("status") == "missing" && !miss_record->number("io_attempted") &&
                  !miss_record->number("start_ns") && !miss_record->number("finish_ns"),
              "a catalog miss has no invented queue or materialization interval");
        check(refusal && refusal->text("status") == "refused_budget" && !refusal->number("io_attempted") &&
                  !refusal->number("start_ns") && !read_trace_count(records, "read_start", 7002),
              "an admission refusal records no materialization or filesystem attempt");
        const auto* failed_finish = read_trace(records, "read_finish", 7006);
        const auto* failed_settled = read_trace(records, "settled", 7006);
        check(failed_finish && failed_finish->text("status") == "failed" && !failed_finish->number("io_attempted") &&
                  !failed_finish->number("start_ns") && !failed_finish->number("read_ns") &&
                  failed_finish->number("admit_ns") && failed_finish->number("dispatch_ns") &&
                  failed_finish->number("reserved_bytes") && failed_finish->number("holds_slot"),
              "a pre-I/O worker exception reports unwind with its real holds, not fictitious filesystem work");
        check(failed_settled && !failed_settled->number("reserved_bytes") && !failed_settled->number("holds_slot"),
              "exception settlement refunds the worker only after its unwind record");
        const auto* admitted = read_trace(records, "read_admitted", 7003);
        const auto* start = read_trace(records, "read_start", 7003);
        const auto* finish = read_trace(records, "read_finish", 7003);
        const auto* settled = read_trace(records, "settled", 7003);
        const auto* adopted = read_trace(records, "adopt", 7003);
        const auto* released = read_trace(records, "release", 7003);
        check(admitted && start && finish && settled && adopted, "the real timed read records each ownership boundary");
        if (admitted && start && finish && settled && adopted) {
            check(start->number("attempt_id") && start->number("attempt_id") == finish->number("attempt_id") &&
                      finish->number("attempt_id") == adopted->number("attempt_id"),
                  "one materialization retains one unique attempt identity through adoption");
            check_eq(start->number("head"), published.head_id, "the trace names the selected native head");
            check_eq(start->number("cut"), 4096, "the trace names only the selected cut length");
            check_eq(start->number("generation"), 8, "the trace retains the request generation");
            check(start->number("enqueue_ns") <= admitted->number("admit_ns") &&
                      admitted->number("admit_ns") <= start->number("start_ns") &&
                      start->number("start_ns") <= finish->number("finish_ns") &&
                      finish->number("finish_ns") <= settled->number("ready_ns") &&
                      settled->number("ready_ns") <= adopted->number("ts_ns"),
                  "timestamps order queueing, worker dispatch, frame unwind, ready residence and adoption");
            check_eq(finish->number("queue_ns"), admitted->number("admit_ns") - start->number("enqueue_ns"),
                     "queue residence ends where the real worker reserves the slot");
            check_eq(finish->number("dispatch_ns"), start->number("start_ns") - admitted->number("admit_ns"),
                     "worker dispatch is not counted as materialization");
            check_eq(adopted->number("read_ns"), finish->number("finish_ns") - start->number("start_ns"),
                     "ready residence does not inflate materialization time");
            check_eq(finish->number("read_ns"), finish->number("materialize_ns") + finish->number("body_lock_wait_ns"),
                     "body catalog-lock acquisition waits are distinct from logical materialization");
            check(finish->number("finish_ns") + finish->number("completion_lock_wait_ns") <= finish->number("ts_ns"),
                  "post-unwind settlement lock wait is measured after materialization has ended");
            check_eq(adopted->number("ready_residence_ns"), adopted->number("ready_end_ns") - settled->number("ready_ns"),
                     "ready residence ends separately at adoption");
            check(released && released->number("ready_end_ns") == adopted->number("ready_end_ns") &&
                      released->number("ready_residence_ns") == adopted->number("ready_residence_ns"),
                  "the caller's handed-out lease lifetime does not inflate ready residence");
            check(finish->number("io_attempted") && finish->number("bytes_read") &&
                      finish->number("reserved_bytes") && finish->number("holds_slot"),
                  "frame-unwound records precede settlement refunds of real byte and slot holds");
            check(!settled->number("reserved_bytes") && settled->number("lease_bytes") &&
                      settled->number("holds_slot"), "settlement replaces the peak with a ready lease, not read work");
        }
        check(!read_trace(records, "trace_overflow", 0), "the bounded native trace did not lose ownership records");
        check(trace.contents().find(root.path) == std::string::npos &&
                  trace.contents().find("private:never-log-this-scope") == std::string::npos &&
                  trace.contents().find(sha256_hex(missing.selector.scope)) == std::string::npos,
              "read diagnostics contain neither paths nor private scope material");
    }
}

void check_revoked_prefetch_demand() {
    StoreReadTraceCapture trace;
    TempRoot root;
    ConversationStore store(options_for(root.path, uint64_t(1) << 30, 0));
    const SavedConversation image = fixture(162, 256);
    ConversationStorePublishResult published;
    check(publish_and_wait(store, image, 1, 1, published) &&
              published.status == ConversationStorePublishStatus::stored, "the prefetch branch is stored");
    std::string why;
    check(store.prefetch(query_of(image, 101, 1), why), "a future request is prefetched");
    check(await_status(store, 101, 1, ConversationStoreReadStatus::ready), "the future image is materialized");
    check(store.demand(query_of(image, 102, 1), why), "an earlier demand displaces speculative work");
    ConversationStoreRead earlier;
    check(store.wait_ready(102, 1, earlier, 5000) &&
              earlier.status == ConversationStoreReadStatus::ready, "the earlier demand receives its image");
    check(store.request_status(101, 1) == ConversationStoreReadStatus::cancelled,
          "the unused prefetch was revoked");
    earlier.image.reset();
    store.release_read(102, 1);
    check(store.demand(query_of(image, 101, 1), why), "the future request later becomes demand");
    ConversationStoreRead later;
    check(store.wait_ready(101, 1, later, 5000) &&
              later.status == ConversationStoreReadStatus::ready, "a revoked prefetch becomes a deliverable demand");
    check(later.image && images_equal(*later.image, image, why),
          "the reissued demand restores the original complete image");
    later.image.reset();
    store.release_read(101, 1);
    check_eq(store.ram_lease_bytes(), 0, "both demand images release their leases");

    store.publish_read_allowance(0);
    check(store.prefetch(query_of(image, 104, 1), why), "a budget-limited prefetch is registered");
    check(await_status(store, 104, 1, ConversationStoreReadStatus::refused_budget), "the speculative read is refused");
    store.publish_read_allowance(uint64_t(64) << 20);
    check(store.demand(query_of(image, 104, 1), why), "demand retries after its allowance becomes available");
    ConversationStoreRead budget_retry;
    check(store.wait_ready(104, 1, budget_retry, 5000) && budget_retry.image &&
              images_equal(*budget_retry.image, image, why), "a prior speculative budget refusal does not force cold prefill");
    budget_retry.image.reset();
    store.release_read(104, 1);

    check(store.prefetch(query_of(image, 103, 1), why), "another request is prefetched");
    check(await_status(store, 103, 1, ConversationStoreReadStatus::ready), "its image is materialized");
    store.cancel_read(103, 1);
    check(!store.demand(query_of(image, 103, 1), why), "explicit caller cancellation cannot be revived as demand");
    store.release_read(103, 1);
    store.close();
    const auto records = trace.records();
    const auto* revoked = read_trace(records, "retire", 101, "demand_revoke");
    const auto* replacement = read_trace(records, "reissue", 101, "revoked_demand");
    check(revoked && replacement && replacement->number("previous_attempt_id") == revoked->number("attempt_id") &&
              replacement->number("attempt_id") != revoked->number("attempt_id"),
          "a revoked ready image and its demand redo have linked, distinct attempt identities");
    const auto* budget = read_trace(records, "settled", 104);
    const auto* retry = read_trace(records, "reissue", 104, "refused_budget");
    check(budget && budget->text("status") == "refused_budget" && !budget->number("io_attempted") &&
              !budget->number("start_ns") && retry &&
              retry->number("previous_attempt_id") == budget->number("attempt_id"),
          "a speculative no-I/O refusal is distinct from its later demand materialization");
    check(read_trace(records, "retire", 103, "caller_cancel") &&
              !read_trace(records, "reissue", 103), "explicit cancellation is logged without a fabricated retry");
}

void check_active_prefetch_replacement() {
    StoreReadTraceCapture trace;
    struct Gate {
        std::mutex mutex;
        std::condition_variable cv;
        bool entered = false, open = false, timed_out = false;
    } gate;
    TempRoot root;
    auto options = options_for(root.path, uint64_t(1) << 30, 0);
    options.max_read_requests = 3;
    ConversationStore store(options);
    const SavedConversation image = fixture(163, 256);
    ConversationStorePublishResult published;
    check(publish_and_wait(store, image, 1, 1, published) &&
              published.status == ConversationStorePublishStatus::stored, "the active-reader branch is stored");
    store.set_read_gate_for_test([](void* context) {
        auto& state = *static_cast<Gate*>(context);
        std::unique_lock<std::mutex> lock(state.mutex);
        state.entered = true;
        state.cv.notify_all();
        if (!state.cv.wait_for(lock, std::chrono::seconds(5), [&] { return state.open; })) state.timed_out = true;
    }, &gate);
    std::string why;
    check(store.prefetch(query_of(image, 201, 1), why), "the old speculative worker starts");
    {
        std::unique_lock<std::mutex> lock(gate.mutex);
        check(gate.cv.wait_for(lock, std::chrono::seconds(5), [&] { return gate.entered; }),
              "the real reader holds its reservation before revocation");
    }
    check(store.demand(query_of(image, 202, 1), why), "an earlier demand revokes the active prefetch");
    check(store.demand(query_of(image, 201, 1), why), "a replacement owns the revoked request identity");
    check(!store.demand(query_of(image, 203, 1), why), "the retiring worker still counts against the descriptor bound");
    store.withdraw_prefetch(201, 1);
    store.cancel_publication(201, 1);
    {
        std::lock_guard<std::mutex> lock(gate.mutex);
        gate.open = true;
    }
    gate.cv.notify_all();
    ConversationStoreRead earlier;
    check(store.wait_ready(202, 1, earlier, 5000) && earlier.image &&
              images_equal(*earlier.image, image, why), "the earlier demand survives active-worker retirement");
    earlier.image.reset();
    store.release_read(202, 1);
    ConversationStoreRead replacement;
    check(store.wait_ready(201, 1, replacement, 5000) && replacement.image &&
              images_equal(*replacement.image, image, why),
          "the replacement survives old-worker cleanup, advisory withdrawal and publication cancellation");
    replacement.image.reset();
    store.release_read(201, 1);
    check_eq(store.ram_lease_bytes(), 0, "active-worker replacement releases every image lease");
    store.set_read_gate_for_test(nullptr, nullptr);
    // Cross the descriptor ceiling with withdrawn offers: none may consume a later request's slot.
    for (uint64_t request = 210; request < 214; ++request) {
        check(store.prefetch(query_of(image, request, 1), why), "a withdrawn offer leaves room for the next one");
        check(await_status(store, request, 1, ConversationStoreReadStatus::ready), "the next offered image materializes");
        store.withdraw_prefetch(request, 1);
        check(store.request_status(request, 1) == ConversationStoreReadStatus::none,
              "withdrawal retires an unadopted descriptor");
    }
    store.close();
    check(!gate.timed_out, "the worker gate was released explicitly, not by timeout");
    const auto records = trace.records();
    const auto* revoked = read_trace(records, "retire", 201, "demand_revoke");
    const auto* old_finish = read_trace(records, "read_finish", 201);
    const auto* retry = read_trace(records, "reissue", 201, "revoked_demand");
    const auto* old_admitted = read_trace(records, "read_admitted", 201);
    const auto* old_start = read_trace(records, "read_start", 201);
    const auto* waiting = read_trace(records, "admission", 202);
    check(waiting && (waiting->number("wait_mask") & 16), "a real gated reader is reported as an occupied worker");
    check(old_admitted && revoked && old_start &&
              old_admitted->number("admit_ns") <= revoked->number("ts_ns") &&
              revoked->number("ts_ns") <= old_start->number("start_ns") && !revoked->number("start_ns"),
          "the real gate exposes dispatch residence without falsely reporting an entered materialization frame");
    check(revoked && revoked->number("active") && revoked->number("reserved_bytes") &&
              revoked->number("holds_slot") && !revoked->number("finish_ns"),
          "active retirement reports the real byte and slot holds before unwind");
    check(old_finish && revoked && old_finish->number("attempt_id") == revoked->number("attempt_id") &&
              old_finish->number("reserved_bytes") && old_finish->number("holds_slot") &&
              old_finish->text("status") == "cancelled",
          "the retiring reader keeps its holds through its own frame-unwound event");
    check(retry && revoked && retry->number("previous_attempt_id") == revoked->number("attempt_id") &&
              retry->number("attempt_id") != revoked->number("attempt_id") &&
              read_trace_count(records, "read_start", 201) == 2,
          "active retirement never conflates the old and replacement materializations");
}

void check_prefetch_catalog_refresh() {
    // Ready and active speculation must both follow a longer cut published after the offer. A ready image remains
    // useful when the replacement cannot fit the current allowance; demand must not turn that image into a miss.
    for (int scenario = 0; scenario < 3; ++scenario) {
        StoreReadTraceCapture trace;
        struct Gate {
            std::mutex mutex;
            std::condition_variable cv;
            bool entered = false, open = false, timed_out = false;
        } gate;
        const bool active = scenario == 1;
        const bool limited = scenario == 2;
        TempRoot root;
        auto options = options_for(root.path, uint64_t(1) << 30, 0);
        options.max_read_requests = 1; // refreshing cannot require a second descriptor
        ConversationStore store(options);
        const SavedConversation old = fixture(164, 256);
        const SavedConversation grown = appended_turn(old, 512, 164);
        const ConversationStoreQuery query = query_of(grown, 301, 7);
        ConversationStorePublishResult first;
        check(publish_and_wait(store, old, 1, 1, first) &&
                  first.status == ConversationStorePublishStatus::stored, "the old prefetch cut is stored");
        if (active)
            store.set_read_gate_for_test([](void* context) {
                auto& state = *static_cast<Gate*>(context);
                std::unique_lock<std::mutex> lock(state.mutex);
                state.entered = true;
                state.cv.notify_all();
                if (!state.cv.wait_for(lock, std::chrono::seconds(5), [&] { return state.open; }))
                    state.timed_out = true;
            }, &gate);
        std::string why;
        check(store.prefetch(query, why), "the future longer request prefetches the old cut");
        if (active) {
            std::unique_lock<std::mutex> lock(gate.mutex);
            check(gate.cv.wait_for(lock, std::chrono::seconds(5), [&] { return gate.entered; }),
                  "the real prefetch reader holds the old reservation");
        } else {
            check(await_status(store, 301, 7, ConversationStoreReadStatus::ready),
                  "the old prefetched image is ready before publication");
        }
        ConversationStorePublishResult newer;
        check(publish_and_wait(store, grown, 2, 2, newer) &&
                  newer.status == ConversationStorePublishStatus::stored, "a longer compatible cut is published");
        const auto found = store.matches(query.selector, query.tokens, query.images);
        check(!found.empty() && found.front().tokens == 512 && found.front().head_id == newer.head_id,
              "the current catalog selects the newly published longer cut");
        const uint64_t reads = store.counters().reads_served;
        const uint64_t bytes = store.counters().bytes_read;
        if (limited)
            store.publish_read_allowance(uint64_t(256) << 10);
        else
            store.suspend_read_admission();
        check(store.demand(query, why), "demand adopts the prepared request after the longer publication");
        check(store.demand(query, why), "repeated demand does not start another refresh");
        store.withdraw_prefetch(301, 7); // the demand owner now controls cancellation
        if (!limited) {
            ConversationStoreRead premature;
            check(!store.take_ready(301, 7, premature), "the old cut is not handed out ahead of the longer demand");
        }
        {
            std::lock_guard<std::mutex> lock(gate.mutex);
            gate.open = true;
        }
        gate.cv.notify_all();
        if (!limited) store.publish_read_allowance(uint64_t(64) << 20);
        ConversationStoreRead read;
        check(store.wait_ready(301, 7, read, 5000) && read.status == ConversationStoreReadStatus::ready,
              "the upgraded demand produces a verified image");
        check_eq(uint64_t(read.tokens), limited ? 256 : 512, "demand uses the longest cut its allowance can restore");
        check_eq(read.head_id, limited ? first.head_id : newer.head_id, "the delivered cut names its actual image");
        check(read.live && read.image && images_equal(*read.image, limited ? old : grown, why),
              "the delivered image contains the exact selected live, checkpoint and K/V bytes");
        check_eq(store.counters().reads_served - reads, limited ? 0 : (active ? 2 : 1),
                 "one demand refresh reads at most the old in-flight image and one replacement");
        if (limited)
            check_eq(store.counters().bytes_read, bytes, "an unaffordable refresh reuses the valid ready bytes");
        check(store.ram_lease_bytes() > 0, "the selected image remains leased until its demand owner releases it");
        read.image.reset();
        store.release_read(301, 7);
        check_eq(store.ram_lease_bytes(), 0, "refreshing leaves no old or new image lease behind");
        store.close();
        check(!gate.timed_out, "the prefetch gate was explicitly released");
        const auto records = trace.records();
        const auto* original = read_trace(records, "admission", 301);
        const auto* reissue = read_trace(records, "reissue", 301, "catalog_refresh");
        const auto* adopted = read_trace(records, "adopt", 301);
        check(original && adopted, "catalog selection diagnostics cover the real offer and adoption");
        if (limited) {
            check(!reissue && read_trace(records, "demand", 301, "ready_hit") &&
                      read_trace_count(records, "read_start", 301) == 1,
                  "an unaffordable refresh is a ready hit, not a fabricated redo");
        } else {
            check(reissue && original && reissue->number("previous_attempt_id") == original->number("attempt_id") &&
                      reissue->number("attempt_id") != original->number("attempt_id") &&
                      reissue->number("head") == newer.head_id && reissue->number("cut") == 512 &&
                      read_trace_count(records, "read_start", 301) == 2,
                  "ready and active catalog refreshes each identify their one actual replacement read");
        }
    }
}

void check_prefetch_catalog_reuse_and_cancel() {
    TempRoot root;
    ConversationStore store(options_for(root.path, uint64_t(1) << 30, 0));
    const SavedConversation old = fixture(165, 256);
    const SavedConversation grown = appended_turn(old, 512, 165);
    ConversationStorePublishResult first;
    check(publish_and_wait(store, old, 1, 1, first) &&
              first.status == ConversationStorePublishStatus::stored, "the unchanged prefetch cut is stored");
    const ConversationStoreQuery same = query_of(old, 302, 1);
    std::string why;
    check(store.prefetch(same, why) && await_status(store, 302, 1, ConversationStoreReadStatus::ready),
          "the unchanged cut is materialized before demand");
    ConversationStorePublishResult duplicate;
    check(publish_and_wait(store, old, 2, 2, duplicate) &&
              duplicate.status == ConversationStorePublishStatus::stored, "a newer head publishes the same cut");
    const uint64_t bytes = store.counters().bytes_read;
    const uint64_t leased = store.ram_lease_bytes();
    check(store.demand(same, why), "demand reuses a ready cut despite a newer equal-length head");
    ConversationStoreRead reused;
    check(store.take_ready(302, 1, reused) && reused.status == ConversationStoreReadStatus::ready &&
              reused.head_id == first.head_id && reused.tokens == 256 && reused.image &&
              images_equal(*reused.image, old, why), "unchanged ready work is immediately handed out byte-identically");
    check_eq(store.counters().bytes_read, bytes, "an equal-length publication causes no second payload read");
    check_eq(store.ram_lease_bytes(), leased, "adoption keeps the original image lease");
    reused.image.reset();
    store.release_read(302, 1);

    const ConversationStoreQuery cancelled = query_of(grown, 303, 1);
    check(store.prefetch(cancelled, why) && await_status(store, 303, 1, ConversationStoreReadStatus::ready),
          "the cancellable generation prefetches the shorter cut");
    ConversationStorePublishResult newer;
    check(publish_and_wait(store, grown, 3, 3, newer) &&
              newer.status == ConversationStorePublishStatus::stored, "the cancellation case publishes a longer cut");
    store.cancel_read(303, 1);
    check(!store.demand(cancelled, why), "a longer publication cannot revive an explicitly cancelled generation");
    ConversationStoreRead absent;
    check(!store.take_ready(303, 1, absent), "the cancelled old image is never handed out");
    check(store.demand(query_of(grown, 303, 2), why), "a new cancellation generation can demand the longer cut");
    store.cancel_read(303, 1);
    ConversationStoreRead next;
    check(store.wait_ready(303, 2, next, 5000) && next.status == ConversationStoreReadStatus::ready &&
              next.generation == 2 && next.tokens == 512 && next.image && images_equal(*next.image, grown, why),
          "cancelling the old generation cannot cancel the new longer demand");
    next.image.reset();
    store.release_read(303, 2);
    store.release_read(303, 1);
    check_eq(store.ram_lease_bytes(), 0, "reuse and cancellation release every image lease");
    store.close();
}

void check_cancel_and_release() {
    TempRoot root;
    ConversationStore store(options_for(root.path, uint64_t(1) << 30, 0));
    const SavedConversation image = fixture(161, 256);
    ConversationStorePublishResult result;
    check(publish_and_wait(store, image, 1, 1, result) &&
              result.status == ConversationStorePublishStatus::stored,
          "the lease test stores its branch");
    std::string why;

    // A queued demand that is cancelled is refunded and never delivered.
    const uint64_t before = store.ram_held_bytes();
    check(store.demand(query_of(image, 21, 1), why), "a demand is admitted for cancellation");
    store.cancel_read(21, 1);
    ConversationStoreRead cancelled;
    check(!store.wait_ready(21, 1, cancelled, 300), "a cancelled request is never delivered");
    check(store.request_status(21, 1) == ConversationStoreReadStatus::cancelled,
          "a cancelled request reports cancellation");
    check(store.ram_held_bytes() <= before, "cancelling a request leaves no charge behind");

    // A request released before its read is ready is forgotten, not delivered later.
    check(store.demand(query_of(image, 22, 1), why), "a demand is admitted for release");
    store.release_read(22, 1);
    ConversationStoreRead released;
    check(!store.wait_ready(22, 1, released, 300), "a released request is never delivered");
    check(store.request_status(22, 1) == ConversationStoreReadStatus::none,
          "a released request is forgotten rather than reported as live");

    // A materialized, untaken image that is released gives its charge back without ever being handed out.
    check(store.demand(query_of(image, 23, 1), why), "a demand is admitted for an untaken release");
    check(await_status(store, 23, 1, ConversationStoreReadStatus::ready), "the image materializes");
    const uint64_t leased = store.ram_lease_bytes();
    check(leased > 0, "the untaken image is charged");
    store.release_read(23, 1);
    check_eq(store.ram_lease_bytes(), 0, "releasing an untaken image refunds it");

    // A handed-out image keeps its charge until the release, and a repeated release is safe.
    check(store.demand(query_of(image, 24, 1), why), "a demand is admitted for a handed-out release");
    ConversationStoreRead read;
    check(store.wait_ready(24, 1, read, 5000), "the request is handed out");
    check(read.image != nullptr, "the handed-out read carries an image");
    check(store.ram_lease_bytes() > 0, "a handed-out image stays charged");
    check(!store.take_ready(24, 1, read), "a handed-out request takes nothing twice");
    store.release_read(24, 1);
    store.release_read(24, 1);
    check_eq(store.ram_lease_bytes(), 0, "a repeated release is safe and idempotent");
    store.cancel_read(4242, 99);   // unknown identifiers are safe
    store.cancel_read(0, 0);
    store.close();
}

void check_dirty_close() {
    TempRoot root;
    const SavedConversation image = fixture(171, 128);
    uint64_t epoch = 0;
    {
        ConversationStore store(options_for(root.path, uint64_t(1) << 30, 0));
        ConversationStorePublishResult result;
        check(publish_and_wait(store, image, 1, 1, result) &&
                  result.status == ConversationStorePublishStatus::stored,
              "the faulted run stores its branch");
        epoch = store.epoch();
        store.set_unhealthy("non-finite logits");   // the engine observed a safety fault
        check(!store.healthy(), "an observed fault latches");
        store.close();                              // ... and then closes
        check(store.run_intent_active(), "a faulted close leaves the run intent armed");
        check(!store.healthy(), "a closed store is not healthy");
    }
    {
        ConversationStore store(options_for(root.path, uint64_t(1) << 30, 0));
        check(store.enabled(), "the root reopens after a faulted close");
        check_eq(store.epoch(), epoch + 1, "the next start invalidates the faulted run");
        check(store.run_intent_active(), "the reopened store arms its own intent");
        const ConversationStoreQuery query = query_of(image, 2, 1);
        check(store.matches(query.selector, query.tokens, query.images).empty(),
              "a faulted run's records are never reused");
        store.close();
        check(!store.run_intent_active(), "a healthy close clears the run intent");
    }
}

void check_ownership_and_close() {
    TempRoot root;
    const SavedConversation image = fixture(181, 128);
    ConversationStore first(options_for(root.path, uint64_t(1) << 30, 0));
    check(first.enabled(), "the first owner opens the root");
    ConversationStorePublishResult result;
    check(publish_and_wait(first, image, 1, 1, result) &&
              result.status == ConversationStorePublishStatus::stored,
          "the owning store stores its branch");
    const uint64_t heads = first.counters().heads;
    const std::string catalog_before = file_digest(root.path + "/catalog");

    // A second owner must serve without persistent caching: it holds no lock and writes no authority state.
    ConversationStore second(options_for(root.path, uint64_t(1) << 30, 0));
    check(!second.enabled(), "a second owner is refused the root");
    check(!second.healthy(), "a refused owner is not healthy");
    check(second.counters().lock_conflicts > 0, "the lock conflict is counted");
    std::string why;
    check(second.publish(std::make_shared<SavedConversation>(image), ConversationStorePublish{}, why) == kRefused,
          "a refused owner cannot publish");
    check(second.matches(selector_of(image), query_of(image, 9, 9).tokens, {}).empty(),
          "a refused owner matches nothing");
    second.close();   // must not clear the owner's marker or write its own catalog
    check(file_digest(root.path + "/catalog") == catalog_before, "the refused owner wrote no catalog");
    check(first.enabled() && first.healthy(), "the owner keeps serving");
    check(first.run_intent_active(), "the refused owner did not clear the run intent");
    check_eq(first.counters().heads, heads, "the refused owner retired nothing");
    first.close();
}

void check_stale_ready() {
    TempRoot root;
    ConversationStore store(options_for(root.path, uint64_t(1) << 30, 0));
    const SavedConversation image = fixture(191, 256);
    ConversationStorePublishResult result;
    check(publish_and_wait(store, image, 1, 1, result) &&
              result.status == ConversationStorePublishStatus::stored,
          "the stale-completion test stores its branch");
    std::string why;
    check(store.demand(query_of(image, 31, 1), why), "a demand is admitted before the invalidation");
    check(await_status(store, 31, 1, ConversationStoreReadStatus::ready),
          "the image is materialized and not yet taken");
    check(store.ram_lease_bytes() > 0, "an untaken completion is charged");

    // A completed image of the previous epoch is retired, never relabelled into the new one.
    check(store.invalidate_domain("non-finite logits"), "the domain invalidation commits durably");
    check_eq(store.counters().heads, 0, "the invalidated domain retains no head");
    check_eq(store.ram_lease_bytes(), 0, "an unhanded completion is retired instead of kept");
    ConversationStoreRead read;
    check(!store.wait_ready(31, 1, read, 300), "a completion of the retired epoch is never delivered");
    check(store.request_status(31, 1) == ConversationStoreReadStatus::stale,
          "the retired completion reports staleness");
    store.release_read(31, 1);

    // A read admitted before the invalidation and still queued is stale as well.
    const SavedConversation second = fixture(192, 256);
    ConversationStorePublishResult stored;
    check(publish_and_wait(store, second, 2, 2, stored) &&
              stored.status == ConversationStorePublishStatus::stored,
          "a later branch is stored in the new epoch");
    check(store.demand(query_of(second, 32, 1), why), "a demand is admitted in the new epoch");
    check(store.invalidate_domain("non-finite logits again"), "the domain invalidates a second time");
    ConversationStoreRead queued;
    check(!store.wait_ready(32, 1, queued, 300), "a queued read of the retired epoch is never delivered");
    check(store.request_status(32, 1) == ConversationStoreReadStatus::stale,
          "the queued read of the retired epoch reports staleness");
    store.release_read(32, 1);
    check(store.ram_held_bytes() <= store.ram_read_budget(), "no charge survives the invalidation");
    store.close();
}

void check_quota_credit() {
    TempRoot root;
    // Two independent stores, each sized from the charge it actually made, so the arithmetic is exact and the test
    // cannot drift with a filesystem's block size.

    // (1) A store holding one pinned branch has no evictable victim: a full quota must refuse, never spend the bytes
    // it merely promises to delete.
    {
        const std::string path = root.path + "/pinned";
        const SavedConversation pinned = fixture(201, 4096);
        uint64_t charged = 0;
        {
            ConversationStore store(options_for(path, uint64_t(1) << 30, 0));
            ConversationStorePublishResult result;
            check(publish_and_wait(store, pinned, 1, 1, result) &&
                      result.status == ConversationStorePublishStatus::stored,
                      "the pinned branch is stored");
            charged = store.disk_charged_bytes();
            check(charged > 0, "the pinned branch is charged");
            store.close();
        }
        // A quota that leaves exactly the physical headroom the store's own catalog copies need - not an exempt
        // write, and not a promise of a later deletion: every payload admission still has to pay from what is left.
        ConversationStore store(options_for(path, charged + catalog_copy_headroom(path), 0));
        check(store.enabled(), "the pinned store opens inside its quota");
        const uint64_t before = store.disk_charged_bytes();
        const SavedConversation other_pinned = fixture(203, 4096);
        auto other = std::make_shared<SavedConversation>(other_pinned);
        ConversationStorePublish request;
        request.request_id = 2;
        request.generation = 2;
        std::string why;
        check(store.publish(other, request, why) == kAccepted, "a second branch is admitted as a ticket");
        ConversationStorePublishResult refused;
        check(store.wait_publish(refused, 5000), "the second branch settles");
        check(refused.status == ConversationStorePublishStatus::refused_quota,
              "a pinned-only store cannot make room by promising to delete");
        check_eq(store.disk_charged_bytes(), before, "a refused publication charges nothing");
        check(store.counters().refusals_quota > 0, "the quota refusal is counted");
        check_eq(store.counters().heads, 1, "the pinned branch survived the refusal");
        const ConversationStoreQuery still_there = query_of(pinned, 3, 3);
        check(!store.matches(still_there.selector, still_there.tokens, still_there.images).empty(),
              "the pinned branch is still authorized");
        store.close();
    }

    // (2) With an evictable branch present, the store makes room by evicting it *durably* before it charges the new
    // bytes, and never touches the pinned branch.
    {
        const std::string path = root.path + "/evict";
        SavedConversation big = fixture(211, 4096);
        for (auto& checkpoint : big.checkpoints) checkpoint.pinned = false;
        const SavedConversation small_pinned = fixture(212, 256);
        check(!big.pinned(), "the evictable fixture carries no pin");
        check(small_pinned.pinned(), "the small fixture is pinned");
        uint64_t charged = 0;
        {
            ConversationStore store(options_for(path, uint64_t(1) << 30, 0));
            ConversationStorePublishResult first, second;
            check(publish_and_wait(store, big, 1, 1, first) &&
                      first.status == ConversationStorePublishStatus::stored,
                  "the evictable branch is stored");
            check(publish_and_wait(store, small_pinned, 2, 2, second) &&
                      second.status == ConversationStorePublishStatus::stored,
                  "the pinned branch is stored beside it");
            charged = store.disk_charged_bytes();
            store.close();
        }
        // A quota that leaves exactly the physical headroom the store's own catalog copies need - not an exempt
        // write, and not a promise of a later deletion: every payload admission still has to pay from what is left.
        ConversationStore store(options_for(path, charged + catalog_copy_headroom(path), 0));
        check(store.enabled(), "the eviction store opens inside its quota");
        SavedConversation replacement_image = fixture(213, 4096);
        auto replacement = std::make_shared<SavedConversation>(replacement_image);
        ConversationStorePublish request;
        request.request_id = 3;
        request.generation = 3;
        std::string why;
        check(store.publish(replacement, request, why) == kAccepted, "an evictable publication is admitted");
        ConversationStorePublishResult stored;
        check(store.wait_publish(stored, 5000), "the evicting publication settles");
        if (stored.status != ConversationStorePublishStatus::stored)
            std::fprintf(stderr, "Eviction publication: %s\n%s\n", stored.detail.c_str(), store.describe().c_str());
        check(stored.status == ConversationStorePublishStatus::stored, "the evicting publication is stored");
        check(store.disk_charged_bytes() <= store.disk_quota_bytes(), "the store stays inside its quota");
        check_eq(store.counters().heads, 2, "the pinned branch and the new branch are retained");
        const ConversationStoreQuery evicted = query_of(big, 4, 4);
        check(store.matches(evicted.selector, evicted.tokens, evicted.images).empty(),
              "the evicted branch is no longer authorized");
        const ConversationStoreQuery kept = query_of(small_pinned, 5, 5);
        check(!store.matches(kept.selector, kept.tokens, kept.images).empty(),
              "the pinned branch survived the eviction");
        store.close();
    }
}

// ---------------------------------------------------------------------------------------------------------------
// The writer's own workspace: counted before it is allocated, exact at its own boundary, and always given back
// ---------------------------------------------------------------------------------------------------------------

void check_writer_workspace() {
    const ConversationStoreOptions shape = options_for("/unused", uint64_t(1) << 30, 0);
    const auto fresh_branch = [](ConversationStore&) {
        ConversationStorePublish request;
        request.request_id = 1;
        request.generation = 1;
        return request;
    };

    // Hundreds of empty streams and a nearly empty payload: a reservation that scaled with a configured maximum rather
    // than with the actual object would be visible immediately, and the counted peak is exact at its own boundary.
    const SavedConversation hollow = hollow_fixture(301, 64, 2);
    const uint64_t hollow_bytes = uint64_t(hollow.bytes());
    const uint64_t hollow_floor = publication_admission_floor(shape, hollow, fresh_branch, uint64_t(16) << 20);
    check(hollow_floor > hollow_bytes, "the writer's workspace is charged on top of the image itself");
    check(hollow_floor - hollow_bytes < (uint64_t(2) << 20),
          "a hollow image's workspace is counted from its own structures, not a per-stream or per-head maximum");
    check(publication_is_admitted(shape, hollow, fresh_branch, hollow_floor),
          "a publication is admitted at exactly its counted peak");
    check(!publication_is_admitted(shape, hollow, fresh_branch, hollow_floor - 1),
          "one byte below the counted peak is refused");

    // The preparation ceiling covers the writer as well as the index, and a refusal allocates nothing at all.
    {
        TempRoot root;
        ConversationStoreOptions options = options_for(root.path, uint64_t(1) << 30, 0);
        options.metadata_bytes = uint64_t(4) << 10;
        ConversationStore store(options);
        check(store.enabled(), "a store opens with a four-kilobyte preparation ceiling");
        const uint64_t held = store.ram_held_bytes();
        auto image = std::make_shared<SavedConversation>(hollow);
        ConversationStorePublish request;
        request.request_id = 2;
        request.generation = 1;
        std::string why;
        check(store.publish(image, request, why) == kRefused,
              "a workspace larger than the store's preparation ceiling is refused");
        check(!why.empty(), "the refusal carries a reason");
        check_eq(store.ram_held_bytes(), held, "a refused workspace allocates nothing");
        check(store.counters().refused_limits > 0, "the refusal is counted as a limit refusal");
        store.close();
    }

    // A failed attempt gives the image and the whole counted workspace back, exactly.
    {
        TempRoot root;
        ConversationStore store(options_for(root.path, uint64_t(1) << 30, 0));
        store.publish_read_allowance(uint64_t(64) << 20);
        const uint64_t before = store.ram_held_bytes();
        store.set_publish_fault_for_test("layout");
        auto image = std::make_shared<SavedConversation>(fixture(302, 4096));
        ConversationStorePublish request;
        request.request_id = 3;
        request.generation = 1;
        std::string why;
        check(store.publish(image, request, why) == kAccepted, "a publication with a fault ahead of it is admitted");
        ConversationStorePublishResult result;
        check(store.wait_publish(result, 5000), "the failed publication settles");
        check(result.status == ConversationStorePublishStatus::failed,
              "the injected layout fault fails the publication");
        check_eq(store.ram_held_bytes(), before, "a failed publication releases its image and its counted workspace");
        check_eq(store.ram_lease_bytes(), 0, "a failed publication releases its lease");
        check(store.healthy(), "an ordinary publication failure does not fault the store");
        store.close();
    }

    // A stored publication keeps only its retained index entry: what is left charged must be far smaller than the
    // workspace that was released, or the counted peak would still be held alive.
    {
        TempRoot root;
        ConversationStore store(options_for(root.path, uint64_t(1) << 30, 0));
        store.publish_read_allowance(uint64_t(64) << 20);
        const SavedConversation tiny = fixture(303, 256);
        const uint64_t workspace =
            publication_admission_floor(shape, tiny, fresh_branch, uint64_t(4) << 20) - uint64_t(tiny.bytes());
        check(workspace > 0, "the tiny fixture's workspace is a real charge");
        const uint64_t before = store.ram_held_bytes();
        ConversationStorePublishResult result;
        check(publish_and_wait(store, tiny, 4, 4, result) &&
                  result.status == ConversationStorePublishStatus::stored,
              "the tiny publication is stored");
        check_eq(store.ram_lease_bytes(), 0, "a stored publication releases its alias");
        check(store.ram_held_bytes() - before < workspace,
              "a stored publication retains its index entry and releases its counted workspace");
        store.close();
    }

    // A ticket the writer has not begun is refunded at once when it is cancelled.  If the writer won the race the
    // publication settles normally instead and its charge is released at its end, so the ledger is conserved either
    // way and only the cancellation branch asserts the stronger, immediate refund.
    {
        TempRoot root;
        ConversationStore store(options_for(root.path, uint64_t(1) << 30, 0));
        store.publish_read_allowance(uint64_t(1) << 30);
        const uint64_t before = store.ram_held_bytes();
        auto image = std::make_shared<SavedConversation>(fixture(304, 65536));
        ConversationStorePublish request;
        request.request_id = 5;
        request.generation = 1;
        std::string why;
        check(store.publish(image, request, why) == kAccepted, "a large publication is admitted for cancellation");
        store.cancel_publication(5, 1);
        ConversationStorePublishResult settled;
        bool got = false;
        for (int waited = 0; waited < 5000 && !got; waited += 10) got = store.wait_publish(settled, 10);
        check(got, "the cancelled ticket settles");
        check(settled.status == ConversationStorePublishStatus::stale ||
                  settled.status == ConversationStorePublishStatus::stored,
              "a cancelled ticket either never started or settled normally");
        check_eq(store.ram_lease_bytes(), 0, "no publication alias survives its attempt");
        check(store.ram_held_bytes() >= before, "the ledger never drops below what the store still owns");
        if (settled.status == ConversationStorePublishStatus::stale)
            check_eq(store.ram_held_bytes(), before,
                     "a ticket cancelled before it starts gives its workspace back at once");
        store.close();
    }

    // A transferred lease: a shared cache hands the entry over while still holding a lease for exactly its bytes, so
    // this one publication is fitted against the published allowance *plus* that credit.  The credit is never added
    // to the allowance itself, is never granted to a read, and cannot claim more than the image being handed over.
    {
        const ConversationStoreOptions shape = options_for("/unused", uint64_t(1) << 30, 0);
        const SavedConversation leased = fixture(305, 4096);
        const uint64_t floor = publication_admission_floor(shape, leased, fresh_branch, uint64_t(16) << 20);
        const auto without_credit = [](ConversationStore&) {
            ConversationStorePublish request;
            request.request_id = 1;
            request.generation = 1;
            return request;
        };
        const auto with_credit = [&](ConversationStore&) {
            ConversationStorePublish request;
            request.request_id = 1;
            request.generation = 1;
            request.external_lease_bytes = uint64_t(leased.bytes());
            return request;
        };
        check(!publication_is_admitted(shape, leased, without_credit, floor - 1),
              "one byte below the floor is refused without a transferred lease");
        check(publication_is_admitted(shape, leased, with_credit, floor - 1),
              "the entry's own lease credit admits the same publication at that allowance");
        check(!publication_is_admitted(shape, leased, with_credit, 1),
              "a credit larger than the image is clamped to the image it hands over");

        TempRoot root;
        ConversationStore store(options_for(root.path, uint64_t(1) << 30, 0));
        store.publish_read_allowance(floor - 1);
        auto image = std::make_shared<SavedConversation>(leased);
        ConversationStorePublish request;
        request.request_id = 2;
        request.generation = 1;
        request.external_lease_bytes = uint64_t(leased.bytes()) * 4;   // a claim far beyond the entry itself
        std::string why;
        check(store.publish(image, request, why) == kAccepted, "the publication is admitted on its own lease's credit");
        check_eq(store.ram_read_budget(), floor - 1,
                 "a transferred lease is never added to the published allowance");
        check_eq(store.ram_available_bytes(), 0,
                 "the credit leaves no room for a read: the full image and workspace are still charged");
        ConversationStorePublishResult result;
        check(store.wait_publish(result, 5000), "the credited publication settles");
        check_eq(store.ram_lease_bytes(), 0, "the credited publication releases its alias");
        check(store.ram_held_bytes() >= store.ram_lease_bytes(), "the ledger is conserved after the credit");
        store.close();
    }
}

// The reused bytes are counted before they are adopted: a donor's own finer chunks, a repeated interval and a
// predecessor that doubles as a donor are all charged from the donor's actual manifest, never from a head-limit maximum.
void check_reuse_reservation() {
    const SavedConversation base = fixture(311, 2048);
    const SavedConversation grown = appended_turn(base, 3072, 311);
    ConversationStoreOptions shape = options_for("/unused", uint64_t(1) << 30, 0);
    shape.chunk_bytes = uint64_t(1) << 20;   // 256x coarser than the donor's own grid
    const uint64_t donor_stream_bytes = uint64_t(base.live.ids.size()) * sizeof(int32_t);
    const auto idle = [](ConversationStore&) {
        ConversationStorePublish request;
        request.request_id = 1;
        request.generation = 1;
        return request;
    };
    const uint64_t plain_floor = publication_admission_floor(shape, grown, idle, uint64_t(8) << 20);
    check(publication_is_admitted(shape, grown, idle, plain_floor), "the plain publication is admitted at its floor");
    check(!publication_is_admitted(shape, grown, idle, plain_floor - 1),
          "one byte below the plain publication's floor is refused");

    // A donor written with 4 KiB chunks, adopted by a writer whose own grid is 1 MiB: the whole-stream interval ends
    // where both streams end, so the donor's own chunk boundaries are what the destination adopts.
    const auto with_donor = [&](ConversationStore& store) {
        ConversationStorePublishResult stored;
        check(publish_and_wait(store, base, 1, 1, stored) &&
                  stored.status == ConversationStorePublishStatus::stored,
              "the finer-chunk donor is stored");
        ConversationStorePublish request;
        request.request_id = 2;
        request.generation = 2;
        ConversationStoreReuse interval;
        interval.head_id = stored.head_id;
        interval.stream = ConversationStoreStreamId{0, uint16_t(ConversationStoreStreamKind::ids), 0, 0};
        interval.bytes = donor_stream_bytes;
        request.reuse.push_back(interval);
        return request;
    };
    const uint64_t donor_floor = publication_admission_floor(shape, grown, with_donor, uint64_t(8) << 20);
    check(donor_floor > plain_floor, "a named donor's preparation is counted on top of the plain publication");
    check(publication_is_admitted(shape, grown, with_donor, donor_floor),
          "the donor publication is admitted at its floor");
    check(!publication_is_admitted(shape, grown, with_donor, donor_floor - 1),
          "one byte below the donor publication's floor is refused");

    // The same counts with many named donors, including one this store does not retain: an unknown name is a lookup
    // that fails before it allocates, so it cannot inflate the reservation.
    const auto with_many_donors = [&](ConversationStore& store) {
        ConversationStorePublishResult stored;
        check(publish_and_wait(store, base, 1, 1, stored), "the many-donor publication's donor is stored");
        ConversationStorePublish request;
        request.request_id = 3;
        request.generation = 2;
        for (int i = 0; i < 8; ++i) {
            ConversationStoreReuse interval;
            interval.head_id = i < 4 ? stored.head_id : uint64_t(0xdead0000 + i);
            interval.stream = ConversationStoreStreamId{0, uint16_t(ConversationStoreStreamKind::ids), 0, 0};
            interval.bytes = donor_stream_bytes;
            request.reuse.push_back(interval);
        }
        return request;
    };
    const uint64_t many_floor = publication_admission_floor(shape, grown, with_many_donors, uint64_t(8) << 20);
    check(many_floor < uint64_t(grown.bytes()) + (uint64_t(4) << 20),
          "many named donors cost their own manifests, never a configured maximum each");
    check(publication_is_admitted(shape, grown, with_many_donors, many_floor),
          "the many-donor publication is admitted at its floor");

    // Behaviour through a real store: the finer donor's chunks are really adopted, once per interval even when the
    // caller repeats one, and a predecessor that is also a donor is decoded (and charged) twice.
    {
        TempRoot root;
        const std::string path = root.path + "/reuse";
        uint64_t donor_head = 0;
        {
            ConversationStoreOptions donor_options = options_for(path, uint64_t(1) << 30, 0);
            donor_options.chunk_bytes = 4096;
            ConversationStore donor_store(donor_options);
            ConversationStorePublishResult stored;
            check(publish_and_wait(donor_store, base, 1, 1, stored) &&
                      stored.status == ConversationStorePublishStatus::stored,
                  "the finer-chunk donor is stored under its own grid");
            donor_head = stored.head_id;
            donor_store.close();
        }
        ConversationStoreOptions options = options_for(path, uint64_t(1) << 30, 0);
        options.chunk_bytes = uint64_t(1) << 20;
        ConversationStore store(options);
        const auto interval_for = [&](uint64_t head) {
            ConversationStoreReuse interval;
            interval.head_id = head;
            interval.stream = ConversationStoreStreamId{0, uint16_t(ConversationStoreStreamKind::ids), 0, 0};
            interval.bytes = donor_stream_bytes;
            return interval;
        };
        auto image = std::make_shared<SavedConversation>(base);
        ConversationStorePublish request;
        request.request_id = 4;
        request.generation = 2;
        request.reuse.push_back(interval_for(donor_head));
        request.reuse.push_back(interval_for(donor_head));   // repeated: it must not be adopted twice
        std::string why;
        check(store.publish(image, request, why) == kAccepted, "the repeated-donor publication is admitted");
        ConversationStorePublishResult result;
        check(store.wait_publish(result, 5000), "the repeated-donor publication settles");
        check(result.status == ConversationStorePublishStatus::stored, "the repeated-donor publication is stored");
        check_eq(result.reused_bytes, donor_stream_bytes, "a repeated donor interval is adopted exactly once");
        // Every payload byte is adopted, written, or already present as an identical object (the donor's finer grid
        // shares the short streams with this one, byte for byte), and never covered twice.
        check(result.new_bytes + result.reused_bytes <= result.payload_bytes,
              "adopted and newly written chunks cover the payload at most once");
        check(store.counters().dedup_chunks > 0, "the already present chunks are counted as deduplicated");
        check(store.counters().reused_chunks >= 2, "the adopted chunks are counted");

        auto cover_image = std::make_shared<SavedConversation>(base);
        ConversationStorePublish cover;
        cover.request_id = 5;
        cover.generation = 3;
        cover.supersedes = donor_head;
        cover.cover = true;
        cover.reuse.push_back(interval_for(donor_head));
        check(store.publish(cover_image, cover, why) == kAccepted,
              "a cover that also names its own predecessor as a donor is admitted");
        ConversationStorePublishResult covered;
        check(store.wait_publish(covered, 5000), "the cover settles");
        check(covered.status == ConversationStorePublishStatus::replaced,
              "the cover replaces the predecessor it also read as a donor");
        store.close();
    }
}

void check_stream_identity() {
    // n_layers 8 with a QSA interval of 4: the carve [0,8) holds two primary K/V layers and the carve [4,8) holds one
    // primary layer and the draft.  One global layer must have one physical identity in both carves, because that
    // identity is what decides whether an adopted chunk can verify in the manifest it is written into.
    const SavedConversation wide = fixture(211, 32);
    SavedConversation narrow = fixture(211, 32);
    narrow.layer_lo = 4;
    narrow.layer_hi = 8;
    std::pair<Digest, uint64_t> wide_first{}, wide_second{}, narrow_primary{}, narrow_draft{};
    std::string why;
    const uint16_t kv_k = uint16_t(ConversationStoreStreamKind::kv_k);
    check(identity_of(wide, ConversationStoreStreamId{0, kv_k, 0, 0}, wide_first, why),
          "the wide carve maps its first primary layer");
    check(identity_of(wide, ConversationStoreStreamId{0, kv_k, 0, 1}, wide_second, why),
          "the wide carve maps its second primary layer");
    check(identity_of(narrow, ConversationStoreStreamId{0, kv_k, 0, 0}, narrow_primary, why),
          "the narrow carve maps its primary layer");
    check(identity_of(narrow, ConversationStoreStreamId{0, kv_k, 0, 1}, narrow_draft, why),
          "the narrow carve maps its draft entry");
    check(wide_second == narrow_primary,
          "one global K/V layer has one identity across two layer carves");
    check(!(wide_first == wide_second), "two global K/V layers are two identities");
    check(!(narrow_draft == narrow_primary), "the draft entry never shares a primary layer's identity");

    // The same identity and the same bytes are one content address, in either carve; a different global layer or the
    // draft is a different object.
    const uint8_t payload[8] = {9, 8, 7, 6, 5, 4, 3, 2};
    check(conversation_store_detail::object_digest(wide_second.first, kv_k, wide_second.second, sizeof payload,
                                                  payload) ==
              conversation_store_detail::object_digest(narrow_primary.first, kv_k, narrow_primary.second,
                                                       sizeof payload, payload),
          "one global layer's bytes are one content address in both carves");
    check(conversation_store_detail::object_digest(narrow_draft.first, kv_k, narrow_draft.second, sizeof payload,
                                                  payload) !=
              conversation_store_detail::object_digest(narrow_primary.first, kv_k, narrow_primary.second,
                                                       sizeof payload, payload),
          "the draft's bytes are never the primary layer's object");

    // Growing the extent counts of a conversation must not rename an unchanged prefix chunk.
    SavedConversation grown = wide;
    grown.kv[0].cells = grown.kv[0].cells * 2;
    grown.kv[0].pooled_rows += 16;
    std::pair<Digest, uint64_t> grown_first{};
    check(identity_of(grown, ConversationStoreStreamId{0, kv_k, 0, 0}, grown_first, why),
          "the grown carve maps its first primary layer");
    check(grown_first == wide_first, "growing a snapshot does not rename an unchanged prefix chunk");

    // The interpretation and the authorization are part of the identity.
    SavedConversation other_heads = wide;
    other_heads.kv[0].heads = 3;
    std::pair<Digest, uint64_t> other{};
    check(identity_of(other_heads, ConversationStoreStreamId{0, kv_k, 0, 0}, other, why),
          "the reinterpreted carve maps its first primary layer");
    check(!(other == wide_first), "a different K/V interpretation is a different object identity");
    SavedConversation other_scope = wide;
    other_scope.scope.value[0] ^= 0xff;
    std::pair<Digest, uint64_t> scoped{};
    check(identity_of(other_scope, ConversationStoreStreamId{0, kv_k, 0, 0}, scoped, why),
          "the other-scope carve maps its first primary layer");
    check(!(scoped == wide_first), "another authorized conversation cannot share physical objects");
}

// ---------------------------------------------------------------------------------------------------------------
// The catalog's own copy: headroom reserved, headroom never spent by payload, and credit only after removal
// ---------------------------------------------------------------------------------------------------------------

void check_catalog_headroom() {
    TempRoot root;
    const std::string path = root.path + "/tight";
    const SavedConversation pinned = fixture(321, 4096);
    uint64_t charged = 0;
    {
        ConversationStore store(options_for(path, uint64_t(1) << 30, 0));
        ConversationStorePublishResult result;
        check(publish_and_wait(store, pinned, 1, 1, result) &&
                  result.status == ConversationStorePublishStatus::stored,
              "the tight store publishes its pinned branch");
        charged = store.disk_charged_bytes();
        check(charged > 0, "the pinned branch is charged");
        store.close();
    }
    // The headroom the store keeps for its own catalog copy is observable as quota - charged - available.
    uint64_t reserve = 0;
    {
        ConversationStore probe(options_for(path, charged + (uint64_t(1) << 20), 0));
        check(probe.enabled(), "the probe store opens inside a permissive quota");
        check_eq(probe.disk_charged_bytes(), charged, "reopening the root charges the same physical bytes");
        reserve = probe.disk_quota_bytes() - probe.disk_charged_bytes() - probe.disk_available_bytes();
        check(reserve >= (uint64_t(2) << 13), "at least two catalog copies are held back");
        check(reserve <= (uint64_t(1) << 20), "the held-back headroom is proportional to the catalog, not a default");
        probe.close();
    }
    // A quota that leaves exactly that headroom and nothing else: no payload may be admitted, the durable catalog is
    // untouched, and the store can still write the catalog its own close needs.
    {
        ConversationStoreOptions options = options_for(path, 0, 0);
        options.quota_bytes = charged + reserve;
        ConversationStore store(options);
        check(store.enabled(), "the store opens on a quota that is exactly its own catalog headroom");
        check_eq(store.disk_available_bytes(), 0, "exactly the catalog's headroom is held back, and nothing is spare");
        const std::string catalog_before = file_digest(path + "/catalog");
        auto image = std::make_shared<SavedConversation>(fixture(322, 4096));
        ConversationStorePublish request;
        request.request_id = 2;
        request.generation = 2;
        std::string why;
        check(store.publish(image, request, why) == kAccepted, "the publication is admitted as a ticket");
        ConversationStorePublishResult refused;
        check(store.wait_publish(refused, 5000), "the publication settles");
        check(refused.status == ConversationStorePublishStatus::refused_quota,
              "a payload cannot spend the headroom held back for the catalog");
        check(store.healthy(), "an ordinary quota refusal does not fault the store");
        check(store.counters().refusals_quota > 0, "the refusal is counted");
        check_eq(store.disk_charged_bytes(), charged, "a refused publication leaves nothing charged");
        check(file_digest(path + "/catalog") == catalog_before, "a refused publication wrote no catalog");
        store.close();
        check(!store.run_intent_active(), "the close committed its catalog inside the headroom it held back");
    }

    // The headroom tracks the catalog's growth, block by block, and a store that just crossed an allocation block
    // still commits and closes cleanly.
    {
        const std::string growth_path = root.path + "/growth";
        ConversationStoreOptions options = options_for(growth_path, uint64_t(1) << 30, 0);
        options.head_limit = 64;
        ConversationStore store(options);
        check(store.enabled(), "the growth store opens");
        const uint64_t reserve_before =
            store.disk_quota_bytes() - store.disk_charged_bytes() - store.disk_available_bytes();
        bool all_stored = true;
        bool crossed = false;
        for (int i = 0; i < 24 && !crossed; ++i) {
            const SavedConversation image = fixture(330 + uint64_t(i), 256);
            ConversationStorePublishResult result;
            if (!publish_and_wait(store, image, uint64_t(100 + i), 1, result) ||
                result.status != ConversationStorePublishStatus::stored)
                all_stored = false;
            struct stat info {};
            if (::stat((growth_path + "/catalog").c_str(), &info) == 0 && uint64_t(info.st_size) > 4096) crossed = true;
        }
        check(all_stored, "every head of the growth store is stored");
        check(crossed, "the catalog grows past one allocation block");
        const uint64_t reserve_after =
            store.disk_quota_bytes() - store.disk_charged_bytes() - store.disk_available_bytes();
        check(reserve_after >= reserve_before, "the held-back headroom grows with the catalog");
        check(store.healthy(), "the growth store is healthy after crossing the block boundary");
        store.close();
        check(!store.run_intent_active(), "the growth store closes cleanly after crossing the block boundary");
    }

    // A deletion that cannot be confirmed stays charged and stays queued: the credit happens only once the file is
    // really gone, and it is never spent as room for the bytes that displaced it.
    {
        const std::string credit_path = root.path + "/credit";
        SavedConversation victim = fixture(340, 2048);
        for (auto& checkpoint : victim.checkpoints) checkpoint.pinned = false;
        check(!victim.pinned(), "the credit fixture carries no pin, so it can be evicted");
        std::vector<std::string> before_manifests = directory_names(credit_path + "/manifests");
        uint64_t charged_now = 0;
        {
            ConversationStore store(options_for(credit_path, uint64_t(1) << 30, 0));
            ConversationStorePublishResult result;
            check(publish_and_wait(store, victim, 1, 1, result) &&
                      result.status == ConversationStorePublishStatus::stored,
                  "the evictable victim is stored");
            charged_now = store.disk_charged_bytes();
            store.close();
        }
        std::string victim_manifest;
        for (const auto& name : directory_names(credit_path + "/manifests"))
            if (!inventory_contains(before_manifests, name)) victim_manifest = name;
        check(!victim_manifest.empty(), "the victim's manifest file is identified");
        const std::string manifest_path = credit_path + "/manifests/" + victim_manifest;
        ConversationStoreOptions tight = options_for(credit_path, 0, 0);
        tight.quota_bytes = charged_now + catalog_copy_headroom(credit_path) + 8192;
        ConversationStore store(tight);
        check(store.enabled(), "the credit store opens on a tight quota");
        // A directory where a file used to be cannot be unlinked, so an eviction's removal of that manifest cannot be
        // confirmed.  The sabotage lands only now, with the victim already authorized: the store has to evict it for
        // room, and the removal it cannot confirm must stay charged rather than be credited as if it were gone.
        check(::unlink(manifest_path.c_str()) == 0, "the victim's manifest is replaced by a directory");
        check(::mkdir(manifest_path.c_str(), 0700) == 0, "the directory takes the manifest's place");
        const int keep = ::open((manifest_path + "/keep").c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
        if (keep >= 0) ::close(keep);
        auto image = std::make_shared<SavedConversation>(fixture(341, 2048));
        ConversationStorePublish request;
        request.request_id = 2;
        request.generation = 2;
        std::string why;
        check(store.publish(image, request, why) == kAccepted, "the displacing publication is admitted");
        ConversationStorePublishResult stored;
        check(store.wait_publish(stored, 5000), "the displacing publication settles");
        check(stored.status == ConversationStorePublishStatus::stored,
              "the displacing publication is stored after evicting the victim");
        check(store.counters().pending_delete_bytes > 0,
              "a deletion that could not be confirmed stays charged");
        check(store.counters().deferred_reclaims > 0, "the failed deletion is counted as deferred");
        check(store.counters().reclaimed_objects > 0, "the victim's objects were reclaimed and credited");
        // Repair the sabotage: the next commit's retry finds the file gone and credits it.
        ::unlink((manifest_path + "/keep").c_str());
        ::rmdir(manifest_path.c_str());
        store.close();
        check_eq(store.counters().pending_delete_bytes, 0,
                 "the retried deletion is credited once the file is confirmed gone");
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Catalog transaction outcomes: an ordinary refusal changes nothing, an unconfirmed rename changes everything
// ---------------------------------------------------------------------------------------------------------------

void check_commit_outcomes() {
    // (1) A rename that never happened is an ordinary refusal: the previous authority is intact, the store keeps
    // serving, and the bytes this attempt wrote come back out of the ledger once they are confirmed gone.
    {
        TempRoot root;
        const std::string path = root.path + "/unchanged";
        ConversationStore store(options_for(path, uint64_t(1) << 30, 0));
        const SavedConversation head_image = fixture(361, 2048);
        ConversationStorePublishResult head;
        check(publish_and_wait(store, head_image, 1, 1, head) &&
                  head.status == ConversationStorePublishStatus::stored,
              "the predecessor of the refused commit is stored");
        const uint64_t charged_before = store.disk_charged_bytes();
        const std::string catalog_before = file_digest(path + "/catalog");
        store.set_publish_fault_for_test("catalog-rename");
        auto image = std::make_shared<SavedConversation>(head_image);
        ConversationStorePublish cover;
        cover.request_id = 2;
        cover.generation = 2;
        cover.supersedes = head.head_id;
        cover.cover = true;
        std::string why;
        check(store.publish(image, cover, why) == kAccepted, "the covering publication is admitted");
        ConversationStorePublishResult refused;
        check(store.wait_publish(refused, 5000), "the refused commit settles");
        check(refused.status == ConversationStorePublishStatus::failed, "a refused catalog commit fails the publication");
        check(!refused.detail.empty(), "the failure carries a reason");
        check(store.healthy(), "a catalog commit that changed nothing does not fault the store");
        check_eq(store.counters().heads, 1, "a refused commit authorizes no head and retires none");
        ConversationStoreManifestInfo info;
        check(store.manifest_info(head.head_id, info), "the predecessor keeps its authority");
        const ConversationStoreQuery query = query_of(head_image, 3, 3);
        check(!store.matches(query.selector, query.tokens, query.images).empty(),
              "the predecessor still answers a lookup");
        check(file_digest(path + "/catalog") == catalog_before, "a refused commit left the durable catalog alone");
        check_eq(store.disk_charged_bytes(), charged_before,
                 "a refused commit credits back everything its attempt wrote");
        // The store is still a working store: the same publication succeeds once the fault is gone.
        ConversationStorePublishResult retry;
        auto retry_image = std::make_shared<SavedConversation>(head_image);
        check(store.publish(retry_image, cover, why) == kAccepted, "the retry is admitted");
        check(store.wait_publish(retry, 5000), "the retry settles");
        check(retry.status == ConversationStorePublishStatus::replaced, "the retry replaces the predecessor");
        store.close();
    }

    // (2) A rename whose durability cannot be confirmed is uncertainty: every file this attempt wrote is retained,
    // nothing is deleted or credited, no further transaction or reuse happens, and the close is not a clean one.
    for (const char* injected : {"catalog-sync", "catalog-apply"}) {
        TempRoot root;
        const std::string path = std::string(root.path) + "/uncertain";
        uint64_t first_epoch = 0;
        uint64_t head_id = 0;
        {
            ConversationStore store(options_for(path, uint64_t(1) << 30, 0));
            const SavedConversation image = fixture(362, 2048);
            ConversationStorePublishResult result;
            check(publish_and_wait(store, image, 1, 1, result) &&
                      result.status == ConversationStorePublishStatus::stored,
                  "the uncertain-commit store stores its branch");
            first_epoch = store.epoch();
            head_id = result.head_id;
            const std::vector<std::string> before = file_inventory(path);
            const std::string catalog_before = file_digest(path + "/catalog");
            const uint64_t charged_before_attempt = store.disk_charged_bytes();
            store.set_publish_fault_for_test(injected);
            auto image2 = std::make_shared<SavedConversation>(fixture(363, 2048));
            ConversationStorePublish request;
            request.request_id = 2;
            request.generation = 2;
            std::string why;
            check(store.publish(image2, request, why) == kAccepted, "the publication that will be uncertain is admitted");
            ConversationStorePublishResult uncertain;
            check(store.wait_publish(uncertain, 5000), "the uncertain publication settles");
            check(uncertain.status == ConversationStorePublishStatus::failed,
                  "an unconfirmed catalog rename fails the publication");
            check(!store.healthy(), "an unconfirmed catalog rename latches a fault");
            // Every file this attempt created - and every file that existed before it - is still there.
            const std::vector<std::string> after = file_inventory(path);
            bool retained = true;
            for (const auto& entry : before)
                if (!inventory_contains(after, entry)) retained = false;
            check(retained, "an uncertain commit retains every file that existed before it");
            check(after.size() > before.size(), "the files this attempt wrote are retained, not cleaned up");
            // The rename really happened, so the live catalog file *is* the new one - and because that cannot be
            // confirmed, its bytes and every file it may name stay charged: nothing is credited, and no further
            // transaction may be written over it.
            const std::string catalog_after_attempt = file_digest(path + "/catalog");
            check(catalog_after_attempt != catalog_before,
                  "an uncertain commit does not pretend the temporary's absence means nothing happened");
            check(store.disk_charged_bytes() >= charged_before_attempt,
                  "an uncertain commit credits nothing and keeps every byte it wrote charged");
            // No further transaction, no reclamation and no reuse: the authority cannot be trusted.
            check(!store.invalidate_domain("non-finite logits"),
                  "a safety invalidation cannot commit after an uncertain transaction");
            store.confirmed_restore(head_id);   // a queued promotion must not be written either
            ::usleep(50000);
            check(file_digest(path + "/catalog") == catalog_after_attempt,
                  "no queued commit is written after an uncertain transaction");
            check_eq(file_inventory(path).size(), after.size(),
                     "no queued commit collects garbage after an uncertain transaction");
            auto third = std::make_shared<SavedConversation>(fixture(364, 2048));
            ConversationStorePublish blocked;
            blocked.request_id = 3;
            blocked.generation = 3;
            check(store.publish(third, blocked, why) == kRefused, "a faulted store admits no further publication");
            const ConversationStoreQuery query = query_of(fixture(362, 2048), 4, 4);
            check(store.matches(query.selector, query.tokens, query.images).empty(),
                  "a faulted store answers no lookup");
            store.close();
            check(store.run_intent_active(), "an uncertain transaction keeps the run intent armed");
            check(file_inventory(path) == after, "closing an uncertain store deletes nothing");
        }
        {
            // The next start re-reads the authority instead of trusting the uncertain view.
            ConversationStore store(options_for(path, uint64_t(1) << 30, 0));
            check(store.enabled(), "the root reopens after an uncertain transaction");
            check_eq(store.epoch(), first_epoch + 1, "the armed intent makes the next start distrust this domain");
            const ConversationStoreQuery query = query_of(fixture(362, 2048), 5, 5);
            check(store.matches(query.selector, query.tokens, query.images).empty(),
                  "an uncertain run's records are never reused");
            store.close();
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// The bootstrap ledger and the run-intent marker: nothing is admitted against a partial ledger, and a re-arm is
// reported for what it is
// ---------------------------------------------------------------------------------------------------------------

void check_bootstrap_and_intent() {
    // (1) A dirty start whose first transaction cannot be paid for does not open at all: the epoch advance it needs
    // is not recorded, so serving from its own view of the authority would be serving an uncommitted one.
    {
        TempRoot root;
        const std::string path = root.path + "/unclean";
        uint64_t charged = 0;
        {
            // A run that arms its intent and never closes it: the next start must distrust this domain.  Nothing is
            // published, so there is no record whose removal could free the room the boot transaction needs, and a
            // brand-new store writes no catalog at all (its domain is re-derived identically on the next start).
            ConversationStore store(options_for(path, uint64_t(1) << 30, 0));
            check(store.enabled() && store.run_intent_active(), "the first run arms its run intent");
            charged = store.disk_charged_bytes();
            check(charged > 0, "the first run charges the control files it created");
            // No close(): the destructor abandons the run intent.
        }
        check(file_digest(path + "/catalog") == "(missing)", "a new store's first open writes no catalog");
        {
            ConversationStoreOptions options = options_for(path, 0, 0);
            options.quota_bytes = charged + (uint64_t(4) << 10);
            ConversationStore store(options);
            check(!store.enabled(), "a dirty start without catalog headroom does not open");
            check(!store.healthy(), "a store that could not record its epoch is not healthy");
            check(store.run_intent_active(), "the run intent stays armed when the boot transaction is refused");
            check(store.counters().refusals_quota > 0, "the refused boot transaction is counted");
            check(store.disk_charged_bytes() < store.disk_quota_bytes(),
                  "the refused boot transaction admitted nothing");
            const ConversationStoreQuery query = query_of(fixture(351, 2048), 2, 1);
            check(store.matches(query.selector, query.tokens, query.images).empty(),
                  "a store that never opened serves nothing");
            store.close();
            check(file_digest(path + "/catalog") == "(missing)",
                  "closing a store that never opened writes no catalog");
        }
        {
            // With the headroom the store holds back for its own copy, the same dirty start opens, records the epoch
            // advance and distrusts the previous run's records.
            ConversationStoreOptions options = options_for(path, 0, 0);
            options.quota_bytes = charged + catalog_copy_headroom(path);
            ConversationStore store(options);
            check(store.enabled(), "the dirty start opens once its own headroom exists");
            check(store.counters().epoch_advances > 0, "the unclean start advanced the epoch");
            const ConversationStoreQuery query = query_of(fixture(351, 2048), 3, 1);
            check(store.matches(query.selector, query.tokens, query.images).empty(), "an unclean start reuses nothing");
            store.close();
            check(!store.run_intent_active(), "the repaired run closes cleanly");
        }
    }

    // (2) A close that cannot prove its marker removal durably re-arms it and says so; the next start then distrusts
    // this domain, which is exactly what the re-arm was for.
    for (const char* extra : {"", "intent-rearm-sync"}) {
        TempRoot root;
        const std::string path = std::string(root.path) + "/intent";
        uint64_t epoch = 0;
        {
            ConversationStore store(options_for(path, uint64_t(1) << 30, 0));
            const SavedConversation image = fixture(352, 2048);
            ConversationStorePublishResult result;
            check(publish_and_wait(store, image, 1, 1, result) &&
                      result.status == ConversationStorePublishStatus::stored,
                  "the intent store stores its branch");
            epoch = store.epoch();
            store.set_publish_fault_for_test("intent-unlink-sync");
            if (*extra) store.set_publish_fault_for_test(extra);
            store.close();
            check(!store.healthy(), "a close that cannot prove its marker gone is not a healthy close");
            check(store.run_intent_active(), "a failed marker removal leaves the run intent armed");
            const std::string described = store.describe();
            check(described.find("re-armed") != std::string::npos,
                  "the re-armed marker is reported rather than hidden");
            if (*extra)
                check(described.find("not durably flushed") != std::string::npos,
                      "a re-arm that could not be flushed is reported as such");
            else
                check(described.find("flushed") != std::string::npos,
                      "a re-arm that was flushed is reported honestly");
        }
        {
            ConversationStore store(options_for(path, uint64_t(1) << 30, 0));
            check(store.enabled(), "the root reopens after a failed marker removal");
            check_eq(store.epoch(), epoch + 1, "the re-armed marker makes the next start distrust this domain");
            check(store.run_intent_active(), "the reopened store arms its own intent");
            store.close();
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Admission boundaries: occupied resources against an over-cap shape
// ---------------------------------------------------------------------------------------------------------------

// The warm-publication estimate charges a donor's whole-manifest chunk bound once per *repeated* stream, never once
// per interval: an alias that reuses many independent K/V streams at once must fit a ceiling that is not widened for
// it.  Before the tightening, every interval added a whole-donor-manifest chunk bound, so 64 independent K/V
// intervals reserved 64 manifests' worth of chunks and this admission was refused.
void check_many_stream_reuse_admission() {
    TempRoot root;
    ConversationStoreOptions options = options_for(root.path, uint64_t(1) << 30, 0);
    options.metadata_bytes = uint64_t(8) << 20;   // unchanged: the ceiling is not widened for this record
    options.chunk_bytes = 4096;
    ConversationStore store(options);
    if (!store.enabled()) return;

    // A dense carve with a few live tokens: every K/V part is a real four-byte stream, and the geometry says the
    // carve really holds one stream per (layer, kind) - the correction the dense-manifest tests use.
    SavedConversation dense = hollow_fixture(4711, 0, 32);
    dense.geometry[1] = dense.layer_hi = 4 * int64_t(dense.kv.size());
    for (auto& layer : dense.kv) {
        layer.cells = 4;
        layer.k.resize(4, 12);
        layer.v.resize(4, 13);
    }
    dense.live.ids.assign(4, 11);
    dense.live.gdn.resize(4, 12);
    dense.live.block_pos.resize(4, 13);
    dense.live.used = 4711;

    ConversationStorePublishResult seed;
    check(publish_and_wait(store, dense, 1, 1, seed) && seed.status == ConversationStorePublishStatus::stored,
          "the many-stream donor is stored under the unchanged ceiling");
    if (seed.head_id == 0) {
        store.close();
        return;
    }

    // The outgoing image's own canonical stream table, plus one whole-stream retained prefix per K/V part: exactly the
    // intervals a capture that kept the whole K/V head of the image it retained would claim.
    const std::vector<ConversationStoreStreamInfo> streams = conversation_store_streams(dense);
    std::vector<ConversationRetainedPrefix> retained;
    uint64_t kv_streams = 0, kv_bytes = 0;
    for (const auto& info : streams) {
        if (!conversation_store_kv_kind(info.id.kind) || info.bytes == 0) continue;
        ConversationRetainedPrefix part;
        part.stage = info.id.stage;
        part.layer = uint32_t(info.id.ordinal);
        part.part = uint32_t(info.id.kind - uint16_t(ConversationStoreStreamKind::kv_k));
        part.bytes = info.bytes;
        part.donor_bytes = info.bytes;
        retained.push_back(part);
        ++kv_streams;
        kv_bytes += info.bytes;
    }
    const std::vector<ConversationStoreReuse> intervals = conversation_store_reuse_intervals(
        streams, retained, seed.head_id, store.epoch(), store.epoch(), uint64_t(4096));
    check(kv_streams >= 64, "the dense carve carries many independent K/V streams");
    check_eq(uint64_t(intervals.size()), kv_streams,
             "every independent K/V stream is offered as one whole-stream interval");
    check(intervals.size() > 1, "the reuse spans independent streams rather than repeats of a single one");
    {
        std::vector<ConversationStoreStreamId> distinct;
        for (const auto& interval : intervals) {
            bool seen = false;
            for (const auto& id : distinct) seen = seen || (id == interval.stream);
            if (!seen) distinct.push_back(interval.stream);
        }
        check_eq(uint64_t(distinct.size()), uint64_t(intervals.size()),
                 "no interval repeats another interval's own stream");
    }

    // A newer immutable alias of the same bytes: it covers the head it came from, adopts every K/V stream, and must be
    // admitted under the ceiling that was never widened for it.
    auto image = std::make_shared<SavedConversation>(dense);
    ConversationStorePublish request;
    request.request_id = 2;
    request.generation = 2;
    request.supersedes = seed.head_id;
    request.cover = true;
    request.reuse = intervals;
    std::string why;
    check(store.publish(image, request, why) == kAccepted,
          "many independent K/V reuse streams fit the unchanged preparation ceiling");
    ConversationStorePublishResult replaced;
    check(store.wait_publish(replaced, 5000), "the many-stream alias settles");
    check(replaced.status == ConversationStorePublishStatus::replaced, "the alias replaces the head it covers");
    check_eq(replaced.reused_bytes, kv_bytes, "every independent K/V stream was adopted instead of re-read");
    check(store.counters().reused_chunks >= intervals.size(), "the adopted chunks are counted");
    store.close();

    // Byte equality across a clean reopen: what the alias wrote is the donor's own bytes, restorable from scratch.
    {
        ConversationStore again(options_for(root.path, uint64_t(1) << 30, 0));
        check(again.enabled(), "the many-stream root reopens");
        const ConversationStoreQuery query = query_of(dense, 3, 1);
        std::string reopened_why;
        check(again.demand(query, reopened_why), "the many-stream record is demanded after a reopen");
        ConversationStoreRead read;
        check(again.wait_ready(3, 1, read, 5000), "the many-stream demand settles");
        check(read.status == ConversationStoreReadStatus::ready, "the many-stream record restores");
        check(read.image != nullptr, "the restored many-stream image is handed out");
        if (read.image) {
            std::string diff;
            check(images_equal(*read.image, dense, diff),
                  "the restored many-stream image is byte-identical to the fixture");
        }
        again.release_read(3, 1);
        again.close();
    }
}

// A donor's chunk bound is charged for every time a stream can name the same chunk again, and never multiplied by
// streams that are independent: the tightening may not be so deep that a repeated interval references chunks nothing
// reserved, and it may not charge one manifest per stream either.
void check_reuse_multiplicity_reservation() {
    const SavedConversation base = fixture(321, 2048);
    const SavedConversation grown = appended_turn(base, 3072, 321);
    ConversationStoreOptions shape = options_for("/unused", uint64_t(1) << 30, 0);
    shape.chunk_bytes = uint64_t(1) << 20;   // far coarser than the donor's own grid
    const uint64_t donor_stream_bytes = uint64_t(base.live.ids.size()) * sizeof(int32_t);
    const uint16_t ids_kind = uint16_t(ConversationStoreStreamKind::ids);
    const uint16_t kv_k_kind = uint16_t(ConversationStoreStreamKind::kv_k);
    // `streams` names one interval per entry: repeating a kind repeats the stream, while one entry per K/V ordinal
    // names that many independent streams of the same family.
    const auto intervals_on = [&](const std::vector<std::pair<uint16_t, uint64_t>>& streams) {
        return [&, streams](ConversationStore& store) {
            ConversationStorePublishResult stored;
            check(publish_and_wait(store, base, 1, 1, stored) && stored.status == ConversationStorePublishStatus::stored,
                  "the multiplicity donor is stored");
            ConversationStorePublish request;
            request.request_id = 2;
            request.generation = 2;
            for (const auto& named : streams) {
                ConversationStoreReuse interval;
                interval.head_id = stored.head_id;
                interval.stream = ConversationStoreStreamId{0, named.first, 0, named.second};
                interval.bytes = donor_stream_bytes;
                request.reuse.push_back(interval);
            }
            return request;
        };
    };
    const auto one = intervals_on({{ids_kind, 0}});
    const auto four_repeats = intervals_on({{ids_kind, 0}, {ids_kind, 0}, {ids_kind, 0}, {ids_kind, 0}});
    const auto four_streams = intervals_on({{kv_k_kind, 0}, {kv_k_kind, 1}, {kv_k_kind, 2}, {kv_k_kind, 3}});
    const uint64_t high = uint64_t(8) << 20;
    const uint64_t single_floor = publication_admission_floor(shape, grown, one, high);
    const uint64_t repeats_floor = publication_admission_floor(shape, grown, four_repeats, high);
    const uint64_t streams_floor = publication_admission_floor(shape, grown, four_streams, high);

    check(repeats_floor > single_floor, "a repeated same-stream interval is charged again, not counted once");
    check(repeats_floor > streams_floor, "independent streams of one donor are not multiplied by the donor's manifest");
    check(publication_is_admitted(shape, grown, four_repeats, repeats_floor),
          "the repeated-interval publication is admitted at exactly its counted peak");
    check(!publication_is_admitted(shape, grown, four_repeats, repeats_floor - 1),
          "one byte below the repeated-interval peak is refused");
    check(publication_is_admitted(shape, grown, four_streams, streams_floor),
          "the independent-stream publication is admitted at exactly its counted peak");
    check(!publication_is_admitted(shape, grown, four_streams, streams_floor - 1),
          "one byte below the independent-stream peak is refused");
}

// A ready but unhanded speculative image yields its bytes to a required publication; a demanded, handed-out image is
// the caller's and is never refunded to make room for one.
void check_ready_speculation_yields() {
    StoreReadTraceCapture trace;
    TempRoot root;
    ConversationStore store(options_for(root.path, uint64_t(1) << 30, 0));
    const SavedConversation large = fixture(331, 131072);
    const SavedConversation small = fixture(332, 4096);
    ConversationStorePublishResult stored;
    check(publish_and_wait(store, large, 1, 1, stored) && stored.status == ConversationStorePublishStatus::stored,
          "the speculation-versus-publication branch is stored");
    std::string why;

    // A materialized speculation holds a real lease; the allowance is then lowered *under* it (which never revokes
    // admitted work), so a publication now has to fit beside bytes this store may free only by revoking speculation.
    check(store.prefetch(query_of(large, 401, 1), why), "the speculative read is admitted under a generous allowance");
    check(await_status(store, 401, 1, ConversationStoreReadStatus::ready), "the speculative image materializes");
    check(store.ram_lease_bytes() > 0, "the unhanded speculation holds a lease");
    store.publish_read_allowance(uint64_t(8) << 20);
    check(store.ram_lease_bytes() > 0, "a lowered allowance never revokes work the store already admitted");

    auto image = std::make_shared<SavedConversation>(small);
    ConversationStorePublish request;
    request.request_id = 402;
    request.generation = 1;
    check(store.publish(image, request, why) == kAccepted,
          "a ready speculation yields its bytes to a required publication");
    check(store.request_status(401, 1) == ConversationStoreReadStatus::cancelled,
          "the unhanded speculation is revoked rather than served");
    ConversationStorePublishResult settled;
    check(store.wait_publish(settled, 5000), "the publication that displaced speculation settles");
    check(settled.status == ConversationStorePublishStatus::stored, "the displacing publication is stored");
    check_eq(store.ram_lease_bytes(), 0, "both revoked speculation and completed publication release their leases");

    // The same completion, adopted as demand and handed out, is not speculation: its lease survives the attempt and
    // the publication is deferred instead of being paid for with the caller's bytes.
    check_eq(store.publish_read_allowance(uint64_t(64) << 20), uint64_t(64) << 20,
             "the allowance is restored for the second probe");
    check(store.prefetch(query_of(large, 403, 1), why), "a second speculative read starts under the allowance again");
    check(await_status(store, 403, 1, ConversationStoreReadStatus::ready), "the second speculative image materializes");
    check(store.demand(query_of(large, 403, 1), why), "the demand adopts the completed speculation");
    ConversationStoreRead read;
    check(store.wait_ready(403, 1, read, 5000) && read.image != nullptr, "the adopted demand is handed out");
    const uint64_t leased = store.ram_lease_bytes();
    check(leased > 0, "the handed-out image is charged to its caller");
    store.publish_read_allowance(uint64_t(8) << 20);
    auto second = std::make_shared<SavedConversation>(small);
    ConversationStorePublish blocked;
    blocked.request_id = 404;
    blocked.generation = 1;
    check(store.publish(second, blocked, why) == kDeferred,
          "a handed-out image defers a publication instead of being refunded for it");
    check(store.counters().deferred_publishes > 0, "the deferral is counted");
    check_eq(store.ram_lease_bytes(), leased, "the handed-out lease survives the deferred publication");
    check(store.request_status(403, 1) == ConversationStoreReadStatus::ready,
          "the handed-out read keeps its verdict and its image");
    read.image.reset();
    store.release_read(403, 1);
    check_eq(store.ram_lease_bytes(), 0, "only the caller's release ends the lease");

    auto third = std::make_shared<SavedConversation>(small);
    ConversationStorePublish after;
    after.request_id = 405;
    after.generation = 2;
    check(store.publish(third, after, why) == kAccepted,
          "the same publication is admitted once the caller releases the lease");
    ConversationStorePublishResult released;
    check(store.wait_publish(released, 5000), "the deferred-then-admitted publication settles");
    check(released.status == ConversationStorePublishStatus::stored, "the deferred-then-admitted publication is stored");
    store.close();
    const auto records = trace.records();
    check(read_trace(records, "retire", 401, "publication_revoke") &&
              !read_trace(records, "retire", 403, "publication_revoke"),
          "publication revocation is distinguished from demand revocation without revoking the adopted owner");
}

// A worker that is still materializing keeps every byte it reserved - its RAM peak and the preparation it holds -
// until it settles, and a publication that cannot fit beside it is deferred rather than served by refunding it.
void check_active_reader_defers_publication() {
    struct Gate {
        std::mutex mutex;
        std::condition_variable cv;
        bool entered = false, open = false, timed_out = false;
    } gate;
    TempRoot root;
    auto options = options_for(root.path, uint64_t(1) << 30, 0);
    options.max_read_requests = 3;
    ConversationStore store(options);
    const SavedConversation large = fixture(341, 131072);
    const SavedConversation small = fixture(342, 4096);
    ConversationStorePublishResult stored;
    check(publish_and_wait(store, large, 1, 1, stored) && stored.status == ConversationStorePublishStatus::stored,
          "the active-reader branch is stored");
    if (stored.head_id == 0) {
        store.close();
        return;
    }
    store.set_read_gate_for_test([](void* context) {
        auto& state = *static_cast<Gate*>(context);
        std::unique_lock<std::mutex> lock(state.mutex);
        state.entered = true;
        state.cv.notify_all();
        if (!state.cv.wait_for(lock, std::chrono::seconds(5), [&] { return state.open; })) state.timed_out = true;
    }, &gate);
    std::string why;
    check(store.demand(query_of(large, 501, 1), why), "the demand starts and reserves its whole peak");
    {
        std::unique_lock<std::mutex> lock(gate.mutex);
        check(gate.cv.wait_for(lock, std::chrono::seconds(5), [&] { return gate.entered; }),
              "the real reader holds its reservation before any publication is offered");
    }
    const uint64_t resident = store.ram_held_bytes();
    check(resident > 0, "the running worker's reservation is charged");

    // The allowance is lowered *under* the running worker, which keeps every byte it reserved and keeps working.
    store.publish_read_allowance(uint64_t(8) << 20);
    check_eq(store.ram_held_bytes(), resident, "a lowered allowance never revokes a running worker");
    auto image = std::make_shared<SavedConversation>(small);
    ConversationStorePublish request;
    request.request_id = 502;
    request.generation = 1;
    check(store.publish(image, request, why) == kDeferred,
          "a running worker defers a publication until it settles");
    check(store.counters().deferred_publishes > 0, "the deferral is counted");
    check_eq(store.ram_held_bytes(), resident, "the deferred publication leaves the worker's reservation charged");
    check(store.request_status(501, 1) == ConversationStoreReadStatus::none,
          "the deferred publication neither cancels the worker nor gives it a verdict");
    check(!store.writer_busy(), "a deferred publication never becomes the writer's work");
    // A lower allowance preserves the existing reservation, not permission for later growth.
    store.publish_read_allowance(uint64_t(64) << 20);
    {
        std::lock_guard<std::mutex> lock(gate.mutex);
        gate.open = true;
    }
    gate.cv.notify_all();

    ConversationStoreRead read;
    check(store.wait_ready(501, 1, read, 5000) && read.image != nullptr, "the worker completes and hands out its image");
    check(store.ram_lease_bytes() > 0, "the completed image is the caller's lease");
    store.publish_read_allowance(uint64_t(8) << 20);
    auto second = std::make_shared<SavedConversation>(small);
    ConversationStorePublish again;
    again.request_id = 503;
    again.generation = 2;
    check(store.publish(second, again, why) == kDeferred,
          "a handed-out lease defers the publication until the caller releases it");
    check(store.ram_lease_bytes() > 0, "the handed-out lease survives the deferral");
    read.image.reset();
    store.release_read(501, 1);

    auto third = std::make_shared<SavedConversation>(small);
    ConversationStorePublish admitted;
    admitted.request_id = 504;
    admitted.generation = 3;
    check(store.publish(third, admitted, why) == kAccepted, "the publication is admitted once the reader is released");
    ConversationStorePublishResult result;
    check(store.wait_publish(result, 5000), "the deferred-then-admitted publication settles");
    check(result.status == ConversationStorePublishStatus::stored, "it is stored");
    store.set_read_gate_for_test(nullptr, nullptr);
    store.close();
    check(!gate.timed_out, "the worker gate was released explicitly, not by timeout");
}

// The other half of the classification: a shape that can never fit is refused - at exactly one byte below its counted
// peak, with no allowance published, and under the preparation ceiling - and never reported as a deferral.
void check_permanent_admission_boundaries() {
    const ConversationStoreOptions shape = options_for("/unused", uint64_t(1) << 30, 0);
    const auto fresh_branch = [](ConversationStore&) {
        ConversationStorePublish request;
        request.request_id = 1;
        request.generation = 1;
        return request;
    };
    const SavedConversation image = fixture(351, 4096);
    const uint64_t floor = publication_admission_floor(shape, image, fresh_branch, uint64_t(16) << 20);
    check(publication_admission_probe(shape, image, fresh_branch, floor) == kAccepted,
          "the publication is admitted at exactly its counted peak");
    check(publication_admission_probe(shape, image, fresh_branch, floor - 1) == kRefused,
          "one byte below the counted peak is a permanent refusal, never a deferral");
    check(publication_admission_probe(shape, image, fresh_branch, 0) == kRefused,
          "no published allowance is a permanent refusal");

    // A transferred lease widens one ticket's own fit and nothing else: it admits the same publication at one byte
    // below the plain floor, and a claim larger than the image is clamped so a one-byte allowance still refuses.
    const auto with_credit = [&](ConversationStore&) {
        ConversationStorePublish request;
        request.request_id = 1;
        request.generation = 1;
        request.external_lease_bytes = uint64_t(image.bytes());
        return request;
    };
    check(publication_admission_probe(shape, image, with_credit, floor - 1) == kAccepted,
          "the entry's own lease credit admits the publication at one byte below the plain floor");
    check(publication_admission_probe(shape, image, with_credit, 1) == kRefused,
          "a credit larger than the image is clamped, so a one-byte allowance still refuses");

    // The preparation ceiling is the other permanent bound: a workspace it cannot hold is refused, not deferred.
    {
        TempRoot root;
        ConversationStoreOptions options = options_for(root.path, uint64_t(1) << 30, 0);
        options.metadata_bytes = uint64_t(4) << 10;
        ConversationStore store(options);
        check(store.enabled(), "a store opens with a four-kilobyte preparation ceiling");
        auto candidate = std::make_shared<SavedConversation>(hollow_fixture(352, 64, 2));
        ConversationStorePublish request;
        request.request_id = 1;
        request.generation = 1;
        std::string why;
        check(store.publish(candidate, request, why) == kRefused,
              "a workspace larger than the preparation ceiling is refused, never deferred");
        check(store.counters().refused_limits > 0, "the permanent refusal is counted as a limit refusal");
        check_eq(store.counters().deferred_publishes, 0, "a permanent refusal is never counted as a deferral");
        store.close();
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Failure paths at the operation boundary: the in-flight object lease, one charge per file, and a commit whose
// reclamation cannot allocate
// ---------------------------------------------------------------------------------------------------------------

// A publication that names a chunk it did not create must keep that chunk's file alive until it commits.  Its only
// owner may be evicted for room *while the manifest that will name the chunk is still being written*: a store that
// reclaimed those bytes then would commit a manifest describing an object it had already deleted.  The fixture's
// shared streams are exact multiples of the chunk size where the grown branch is longer, so the prefix really is the
// same physical object: 1024 token ids are one full 4 KiB chunk and each K/V layer is four.
void check_inflight_object_lease() {
    SavedConversation owner = fixture(281, 1024);
    for (auto& checkpoint : owner.checkpoints) checkpoint.pinned = false;
    check(!owner.pinned(), "the shared owner carries no pin, so it can be evicted for room");
    const SavedConversation grown = appended_turn(owner, 8192, 281);

    // (1) The figures the tight root has to meet, measured in their own roots: what the store holds once only the
    // grown branch remains (its predecessor reclaimed, every chunk the two share kept), and the headroom the store
    // holds back for its own catalog copy.
    uint64_t charged_owner = 0;
    {
        TempRoot root;
        ConversationStore store(options_for(root.path, uint64_t(1) << 30, 0));
        ConversationStorePublishResult result;
        check(publish_and_wait(store, owner, 1, 1, result) &&
                  result.status == ConversationStorePublishStatus::stored,
              "the shared owner is stored under a permissive quota");
        charged_owner = store.disk_charged_bytes();
        store.close();
    }
    uint64_t charged_replaced = 0;
    uint64_t catalog_reserve = 0;
    TempRoot probe;
    {
        ConversationStore store(options_for(probe.path, uint64_t(1) << 30, 0));
        check(store.enabled(), "the probe root opens");
        ConversationStorePublishResult first;
        check(publish_and_wait(store, owner, 1, 1, first) &&
                  first.status == ConversationStorePublishStatus::stored,
              "the probe root stores the shared owner");
        auto image = std::make_shared<SavedConversation>(grown);
        ConversationStorePublish cover;
        cover.request_id = 2;
        cover.generation = 2;
        cover.supersedes = first.head_id;
        cover.cover = true;
        std::string why;
        check(store.publish(image, cover, why) == kAccepted, "the covering branch is admitted");
        ConversationStorePublishResult second;
        check(store.wait_publish(second, 5000), "the covering branch settles");
        check(second.status == ConversationStorePublishStatus::replaced,
              "the covering branch replaces the head it was captured from");
        charged_replaced = store.disk_charged_bytes();
        catalog_reserve = store.disk_quota_bytes() - store.disk_charged_bytes() - store.disk_available_bytes();
        check(catalog_reserve > 0, "the store holds back the headroom its own catalog copy needs");
        check_eq(store.counters().heads, 1, "only the grown branch remains in the probe root");
        store.close();
    }

    // (2) The real store.  Its quota is exactly what the grown branch leaves behind once it has reclaimed the owner,
    // plus that headroom and one block: high enough that keeping every chunk the branch shares with its predecessor
    // still fits, low enough that the branch cannot finish without evicting the only head that names them.
    TempRoot tight;
    ConversationStoreOptions options = options_for(tight.path, 0, 0);
    options.quota_bytes = charged_replaced + catalog_reserve + 8192;
    check(options.quota_bytes > charged_owner + catalog_reserve,
          "the tight quota still admits the shared owner it starts from");
    uint64_t grown_head = 0;
    {
        ConversationStore store(options);
        check(store.enabled(), "the tight root opens");
        ConversationStorePublishResult first;
        check(publish_and_wait(store, owner, 1, 1, first) &&
                  first.status == ConversationStorePublishStatus::stored,
              "the shared owner is stored in the tight root");
        auto image = std::make_shared<SavedConversation>(grown);
        ConversationStorePublish request;
        request.request_id = 2;
        request.generation = 2;
        std::string why;
        check(store.publish(image, request, why) == kAccepted, "the grown branch is admitted");
        ConversationStorePublishResult stored;
        check(store.wait_publish(stored, 5000), "the grown branch settles");
        check(stored.status == ConversationStorePublishStatus::stored,
              "the grown branch commits even though it had to evict the only head that owned a chunk it names");
        check(store.counters().evictions > 0, "the grown branch really did evict the shared owner for room");
        check(store.healthy(), "evicting for room under a publication is not a fault");
        grown_head = stored.head_id;
        store.close();
    }

    // (3) The consumer-visible consequence: the committed record must still restore.  A store that reclaimed a chunk
    // its own committed manifest names has no such record left, and the next start says so.
    {
        ConversationStore store(options_for(tight.path, uint64_t(1) << 30, 0));
        check(store.enabled(), "the tight root reopens");
        check_eq(store.counters().corrupt_records, 0,
                 "no stored record names an object the store deleted for room");
        check_eq(store.counters().heads, 1, "the grown branch is still authorized");
        ConversationStoreManifestInfo info;
        check(store.manifest_info(grown_head, info), "the grown branch keeps its manifest");
        ConversationStoreQuery query = query_of(grown, 3, 3);
        std::string why;
        check(store.demand(query, why), "the grown record's demand is admitted");
        ConversationStoreRead read;
        check(store.wait_ready(3, 3, read, 5000), "the grown demand settles");
        check(read.status == ConversationStoreReadStatus::ready,
              "the grown record is served, not reported as a miss or a corrupt record");
        check(read.image && images_equal(*read.image, grown, why), "the grown record restores byte for byte");
        if (read.image) store.release_read(3, 3);
        store.close();
    }
}

// One failed write, one charge.  A write that fails before its file reaches its own name leaves bytes only in the
// temporary, and the store must credit that one file exactly once - never a temporary *and* the name it never
// reached.  The failure is injected at the operation itself (the directory refuses the write), not ahead of it.
void check_failed_write_charges_once() {
    // (1) The object write: its temporary directory refuses the write, so the shard never receives the object.
    {
        TempRoot root;
        const std::string path = root.path + "/objects";
        ConversationStore store(options_for(path, uint64_t(1) << 30, 0));
        check(store.enabled(), "the object-write root opens");
        ConversationStorePublishResult head;
        check(publish_and_wait(store, fixture(291, 2048), 1, 1, head) &&
                  head.status == ConversationStorePublishStatus::stored,
              "the object-write root stores a first branch");
        const uint64_t charged = store.disk_charged_bytes();
        const uint64_t reclaimed = store.counters().reclaimed_objects;
        check(::chmod((path + "/tmp").c_str(), 0500) == 0, "the store's temporary directory can be made unwritable");
        auto image = std::make_shared<SavedConversation>(fixture(292, 2048));
        ConversationStorePublish request;
        request.request_id = 2;
        request.generation = 2;
        std::string why;
        check(store.publish(image, request, why) == kAccepted, "the publication that will fail is admitted");
        ConversationStorePublishResult failed;
        check(store.wait_publish(failed, 5000), "the failing publication settles");
        check(failed.status == ConversationStorePublishStatus::failed, "a refused object write fails the publication");
        check(!failed.detail.empty(), "the failure carries a reason");
        check(store.healthy(), "a failed object write is an ordinary failure, not a fault");
        check_eq(store.counters().reclaimed_objects, reclaimed,
                 "an object that was never written is never counted as reclaimed");
        check_eq(store.disk_charged_bytes(), charged,
                 "a failed object creation refunds only its reservation, not existing files");
        check(::chmod((path + "/tmp").c_str(), 0700) == 0, "the temporary directory is made writable again");
        ConversationStoreRead restored;
        const auto prior = fixture(291, 2048);
        check(store.demand(query_of(prior, 90, 1), why) &&
                  store.wait_ready(90, 1, restored, 5000) && restored.image &&
                  images_equal(*restored.image, prior, why),
              "existing history remains readable after an ordinary object write failure");
        restored.image.reset();
        store.release_read(90, 1);
        store.close();
        check_eq(store.counters().pending_delete_bytes, 0, "the failed write leaves nothing queued for deletion");
    }

    // (2) The manifest write: its own directory refuses the rename, so the bytes stay in the temporary.  Its objects
    // were written and are reclaimed normally; only the manifest's single charge must follow the one file there is.
    {
        TempRoot root;
        const std::string path = root.path + "/manifests";
        ConversationStore store(options_for(path, uint64_t(1) << 30, 0));
        check(store.enabled(), "the manifest-write root opens");
        ConversationStorePublishResult head;
        check(publish_and_wait(store, fixture(293, 2048), 1, 1, head) &&
                  head.status == ConversationStorePublishStatus::stored,
              "the manifest-write root stores a first branch");
        const uint64_t charged = store.disk_charged_bytes();
        check(::chmod((path + "/manifests").c_str(), 0500) == 0,
              "the store's manifest directory can be made unwritable");
        auto image = std::make_shared<SavedConversation>(fixture(294, 2048));
        ConversationStorePublish request;
        request.request_id = 2;
        request.generation = 2;
        std::string why;
        check(store.publish(image, request, why) == kAccepted, "the failing manifest publication is admitted");
        ConversationStorePublishResult failed;
        check(store.wait_publish(failed, 5000), "the failing manifest publication settles");
        check(failed.status == ConversationStorePublishStatus::failed,
              "a refused manifest rename fails the publication");
        check(store.healthy(), "a failed manifest rename is an ordinary failure, not a fault");
        check_eq(store.disk_charged_bytes(), charged,
                 "the manifest's one charge is credited exactly once");
        check_eq(store.counters().pending_delete_bytes, 0,
                 "one file, one removal: the name the manifest never reached is not left queued for deletion");
        check(::chmod((path + "/manifests").c_str(), 0700) == 0,
              "the manifest directory is made writable again");
        ConversationStoreRead restored;
        const auto prior = fixture(293, 2048);
        check(store.demand(query_of(prior, 91, 1), why) &&
                  store.wait_ready(91, 1, restored, 5000) && restored.image &&
                  images_equal(*restored.image, prior, why),
              "existing history remains readable after an ordinary manifest write failure");
        restored.image.reset();
        store.release_read(91, 1);
        store.close();
        check_eq(store.counters().pending_delete_bytes, 0, "the failed manifest write leaves nothing queued");
    }
}

// A transaction that has already committed stays committed even when the reclamation that follows it cannot allocate
// its own batch.  The publication has to learn `stored`, the manifest and every object it authorized have to survive,
// and the files the reclamation never reached have to stay queued and charged rather than be credited or deleted.
void check_postcommit_reclamation_cannot_undo_commit() {
    TempRoot root;
    const std::string path = root.path + "/commit";
    SavedConversation first_image = fixture(295, 2048);
    for (auto& checkpoint : first_image.checkpoints) checkpoint.pinned = false;
    check(!first_image.pinned(), "the head the ceiling retires carries no pin");
    ConversationStoreOptions options = options_for(path, uint64_t(1) << 30, 0);
    options.head_limit = 1;   // the second branch's commit has to retire the first, so reclamation has real work
    uint64_t second_head = 0;
    {
        ConversationStore store(options);
        check(store.enabled(), "the one-head root opens");
        ConversationStorePublishResult first;
        check(publish_and_wait(store, first_image, 1, 1, first) &&
                  first.status == ConversationStorePublishStatus::stored,
              "the first branch is stored");
        check_eq(store.counters().heads, 1, "the one-head root holds exactly one head");
        store.set_publish_fault_for_test("catalog-reclaim");
        auto image = std::make_shared<SavedConversation>(fixture(296, 2048));
        ConversationStorePublish request;
        request.request_id = 2;
        request.generation = 2;
        std::string why;
        check(store.publish(image, request, why) == kAccepted, "the second branch is admitted");
        ConversationStorePublishResult stored;
        check(store.wait_publish(stored, 5000), "the second branch settles");
        check(stored.status == ConversationStorePublishStatus::stored,
              "a reclamation that cannot allocate does not undo the commit it follows");
        check(store.healthy(), "a failed reclamation is not a fault: the authority was never in doubt");
        second_head = stored.head_id;
        ConversationStoreManifestInfo info;
        check(store.manifest_info(second_head, info), "the committed head keeps its authority");
        check_eq(store.counters().heads, 1, "the ceiling retired the first branch and kept the second");
        check(store.counters().reclaimed_objects > 0,
              "the retired branch's objects were still reclaimed, on the pass that follows the commit");
        // The same record must still be servable in this very instance: what the commit authorized may not have been
        // handed to the reclaimer as an orphan.
        const SavedConversation second_image = fixture(296, 2048);
        ConversationStoreQuery query = query_of(second_image, 3, 3);
        check(store.demand(query, why), "the committed record's demand is admitted");
        ConversationStoreRead read;
        check(store.wait_ready(3, 3, read, 5000), "the committed demand settles");
        check(read.status == ConversationStoreReadStatus::ready,
              "the committed record is served, not reported as corrupt");
        check(read.image && images_equal(*read.image, second_image, why),
              "the committed record restores byte for byte");
        if (read.image) store.release_read(3, 3);
        store.close();
        check_eq(store.counters().pending_delete_bytes, 0,
                 "every file the reclamation owed is reclaimed once the pass can run");
    }
    {
        ConversationStore store(options_for(path, uint64_t(1) << 30, 0));
        check(store.enabled(), "the root reopens after the committed transaction");
        check_eq(store.counters().corrupt_records, 0,
                 "the committed manifest and every object it names are still there");
        check_eq(store.counters().heads, 1, "the committed branch is still authorized");
        const SavedConversation second_image = fixture(296, 2048);
        ConversationStoreQuery query = query_of(second_image, 4, 4);
        std::string why;
        check(store.demand(query, why), "the committed record's demand is admitted after a restart");
        ConversationStoreRead read;
        check(store.wait_ready(4, 4, read, 5000), "the committed demand settles after a restart");
        check(read.status == ConversationStoreReadStatus::ready, "the committed record is served after a restart");
        check(read.image && images_equal(*read.image, second_image, why),
              "the committed record restores byte for byte after a restart");
        if (read.image) store.release_read(4, 4);
        store.close();
    }
}

} // namespace