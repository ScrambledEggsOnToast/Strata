// Deterministic CPU tests for the durable conversation store.  No GPU, no model, no network: a temporary root under
// the system temp directory, fixtures built by hand, and checks that read the module's own observable contract -
// matches, statuses, leases, epochs and byte counts.  Every check is a consumer-visible behaviour, not an internal.
#include "strata/core/conversation_store.hpp"
#include "strata/core/dflash2_bridge.hpp"

#include "strata/core/conversation_reuse.hpp"
#include "strata/program/conversation_store_admission.hpp"

#include "conversation_store_format.hpp"
#include "conversation_store_sha256.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
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
#include <sys/wait.h>
#include <stdexcept>
#include <unistd.h>
#include <thread>
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

    std::vector<StoreReadTraceRecord> records(const std::string& channel = "store_read ") const {
        std::vector<StoreReadTraceRecord> result;
        std::istringstream lines(contents());
        std::string line;
        while (std::getline(lines, line)) {
            if (line.compare(0, channel.size(), channel) != 0) continue;
            StoreReadTraceRecord record;
            std::istringstream fields(line.substr(channel.size()));
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
            if (line.compare(0, 11, "store_read ") != 0 && line.compare(0, 11, "store_head ") != 0)
                std::fprintf(stderr, "%s\n", line.c_str());
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

std::vector<std::string> intent_names(const std::string& path) {
    std::vector<std::string> out;
    for (const auto& name : directory_names(path))
        if (name.compare(0, 7, "intent-") == 0) out.push_back(name);
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
void check_native_draft_store();
void check_draft_certification_codec();
void check_draft_certification_store();
void check_draft_reader_floor_failures();
void check_profile_recovery();
void check_full_catalog_profile_restart();
void check_per_domain_head_limit();
void check_protected_cap_per_domain();
void check_retirement_receipts();
void check_bounded_candidate_rejections();
void check_split_pinned_restart();
void check_dedup_beyond_the_quota();
void check_dense_manifest_workspace_refusal();
void check_damaged_manifest_is_still_reclaimed();
void check_multi_stage_restore();
void check_native_split_restore();
void check_request_specific();
void check_incoming_lifetime();
void check_publication_rules();
void check_publication_result_backpressure();
void check_coalesced_publication_result_backpressure();
void check_restart_generation_ordering();
void check_pinned_branch();
void check_queued_read_protects_frontier();
void check_speculation_follows_publication();
void check_selected_hold_revocation();
void check_selected_hold_commit_interleaving();
void check_selected_hold_terminal_release();
void check_selected_retirement_allocation_retry();
void check_selected_equal_cut_order();
void check_budget_deferred_prefetch();
void check_budget_deferred_expanded_peak();
void check_budget_deferred_terminals();
void check_budget_deferred_head_hold();
void check_revoked_headless_demand();
void check_reuse();
void check_leases_and_budget();
void check_tight_ram();
void check_demand_reclaim_quote();
void check_demand_reclaim_owner();
void check_demand_reclaim_accepted_reader();
void check_zero_allowance_demand_adoption();
void check_zero_allowance_adoption_controls();
void check_demand_reclaim_interleavings();
void check_cancel_and_release();
void check_read_timing();
void check_read_progress();
void check_read_progress_failures();
void check_read_progress_reissue();
void check_revoked_prefetch_demand();
void check_active_prefetch_replacement();
void check_prefetch_catalog_refresh();
void check_prefetch_catalog_reuse_and_cancel();
void check_epoch_and_intent();
void check_dirty_close();
void check_ownership_and_close();
void check_shutdown_admission();
void check_shutdown_pending();
void check_shutdown_deadlines();
void check_shutdown_failures();
void check_shutdown_late_finalization();
void check_shutdown_request_policy();
void check_stale_ready();
void check_idle_final_lease_reclaim();
void check_capacity_floor();
void check_quota_credit();
void check_stream_identity();
void check_shared_scope_payload_separation();
void check_corruption(bool truncate = false);
void check_writer_workspace();
void check_reuse_reservation();
void check_many_stream_reuse_admission();
void check_reuse_multiplicity_reservation();
void check_ready_speculation_yields();
void check_active_reader_defers_publication();
void check_permanent_admission_boundaries();
void check_catalog_headroom();
void check_commit_outcomes();
void check_promotion_burst_coalescing();
void check_promotion_publication_progress();
void check_promotion_invalidation_order();
void check_promotion_rearm();
void check_promotion_eviction_absorption();
void check_promotion_commit_faults();
void check_promotion_clean_close();
void check_promotion_recency_eviction();
void check_bootstrap_and_intent();
void check_inflight_object_lease();
void check_failed_write_charges_once();
void check_orphan_revival();
void check_object_directory_retry();
void check_object_directory_batch();
void check_object_data_batch();
void check_postcommit_reclamation_cannot_undo_commit();
void check_read_object_boundaries();

} // namespace

int main() {
#ifndef __linux__
    std::printf("conversation_store_test: the store is POSIX-only, skipped here\n");
    return 77;
#else
    if (::getenv("STRATA_STORE_OBJECT_BATCH_ONLY")) {
        check_object_data_batch();
        check_object_directory_batch();
        std::printf("conversation_store_test object batch: %d checks, %d failures\n", checks, failures);
        return failures == 0 ? 0 : 1;
    }
    if (::getenv("STRATA_STORE_NATIVE_DRAFT_ONLY")) {
        check_native_draft_store();
        std::printf("conversation_store_test native draft: %d checks, %d failures\n", checks, failures);
        return failures == 0 ? 0 : 1;
    }
    if (::getenv("STRATA_STORE_DRAFT_CERTIFICATION_ONLY")) {
        check_draft_certification_codec();
        check_draft_certification_store();
        check_draft_reader_floor_failures();
        std::printf("conversation_store_test draft certification: %d checks, %d failures\n", checks, failures);
        return failures == 0 ? 0 : 1;
    }
    if (::getenv("STRATA_STORE_RETENTION_ONLY")) {
        check_per_domain_head_limit();
        check_protected_cap_per_domain();
        check_retirement_receipts();
        check_bounded_candidate_rejections();
        check_quota_credit();
        check_selected_hold_commit_interleaving();
        std::printf("conversation_store_test retention: %d checks, %d failures\n", checks, failures);
        return failures == 0 ? 0 : 1;
    }
    if (::getenv("STRATA_STORE_SHUTDOWN_ONLY")) {
        check_shutdown_admission();
        check_shutdown_pending();
        check_shutdown_deadlines();
        check_shutdown_failures();
        check_shutdown_late_finalization();
        check_shutdown_request_policy();
        check_dirty_close();
        check_ownership_and_close();
        std::printf("conversation_store_test shutdown: %d checks, %d failures\n", checks, failures);
        return failures == 0 ? 0 : 1;
    }
    if (::getenv("STRATA_STORE_PROMOTION_ONLY")) {
        check_promotion_burst_coalescing();
        check_promotion_publication_progress();
        check_promotion_invalidation_order();
        check_promotion_rearm();
        check_promotion_eviction_absorption();
        check_promotion_commit_faults();
        check_promotion_clean_close();
        check_promotion_recency_eviction();
        std::printf("conversation_store_test promotion: %d checks, %d failures\n", checks, failures);
        return failures == 0 ? 0 : 1;
    }
    if (::getenv("STRATA_STORE_REVIEW_ONLY")) {
        check_selected_retirement_allocation_retry();
        check_selected_equal_cut_order();
        std::printf("conversation_store_test review: %d checks, %d failures\n", checks, failures);
        return failures == 0 ? 0 : 1;
    }
    if (::getenv("STRATA_STORE_DEFERRED_ONLY")) {
        check_budget_deferred_prefetch();
        check_budget_deferred_terminals();
        check_budget_deferred_head_hold();
        check_revoked_headless_demand();
        check_revoked_prefetch_demand();
        check_zero_allowance_demand_adoption();
        check_zero_allowance_adoption_controls();
        check_corruption();
        std::printf("conversation_store_test deferred: %d checks, %d failures\n", checks, failures);
        return failures == 0 ? 0 : 1;
    }
    if (::getenv("STRATA_STORE_LIFETIME_ONLY")) {
        check_selected_hold_revocation();
        check_selected_hold_commit_interleaving();
        check_selected_hold_terminal_release();
        std::printf("conversation_store_test lifetime: %d checks, %d failures\n", checks, failures);
        return failures == 0 ? 0 : 1;
    }
    if (::getenv("STRATA_STORE_PROGRESS_ONLY")) {
        check_read_progress();
        check_read_progress_failures();
        check_read_progress_reissue();
        check_budget_deferred_expanded_peak();
        std::printf("conversation_store_test progress: %d checks, %d failures\n", checks, failures);
        return failures == 0 ? 0 : 1;
    }
    check_crypto();
    check_layout();
    check_identity();
    check_manifest();
    check_round_trip();
    check_native_draft_store();
    check_draft_certification_codec();
    check_draft_certification_store();
    check_draft_reader_floor_failures();
    check_profile_recovery();
    check_full_catalog_profile_restart();
    check_per_domain_head_limit();
    check_protected_cap_per_domain();
    check_retirement_receipts();
    check_bounded_candidate_rejections();
    check_split_pinned_restart();
    check_dedup_beyond_the_quota();
    check_dense_manifest_workspace_refusal();
    check_damaged_manifest_is_still_reclaimed();
    check_multi_stage_restore();
    check_native_split_restore();
    check_request_specific();
    check_incoming_lifetime();
    check_publication_rules();
    check_publication_result_backpressure();
    check_coalesced_publication_result_backpressure();
    check_restart_generation_ordering();
    check_pinned_branch();
    check_queued_read_protects_frontier();
    check_speculation_follows_publication();
    check_reuse();
    check_leases_and_budget();
    check_tight_ram();
    check_demand_reclaim_quote();
    check_demand_reclaim_owner();
    check_demand_reclaim_accepted_reader();
    check_zero_allowance_demand_adoption();
    check_zero_allowance_adoption_controls();
    check_demand_reclaim_interleavings();
    check_cancel_and_release();
    check_revoked_prefetch_demand();
    check_read_timing();
    check_read_progress();
    check_read_progress_failures();
    check_read_progress_reissue();
    check_active_prefetch_replacement();
    check_prefetch_catalog_refresh();
    check_prefetch_catalog_reuse_and_cancel();
    check_selected_hold_revocation();
    check_selected_hold_commit_interleaving();
    check_selected_hold_terminal_release();
    check_selected_retirement_allocation_retry();
    check_idle_final_lease_reclaim();
    check_selected_equal_cut_order();
    check_budget_deferred_prefetch();
    check_budget_deferred_expanded_peak();
    check_budget_deferred_terminals();
    check_budget_deferred_head_hold();
    check_revoked_headless_demand();
    check_epoch_and_intent();
    check_dirty_close();
    check_ownership_and_close();
    check_shutdown_admission();
    check_shutdown_pending();
    check_shutdown_deadlines();
    check_shutdown_failures();
    check_shutdown_late_finalization();
    check_shutdown_request_policy();
    check_stale_ready();
    check_capacity_floor();
    check_quota_credit();
    check_stream_identity();
    check_corruption();
    check_corruption(true);
    check_read_object_boundaries();
    check_writer_workspace();
    check_reuse_reservation();
    check_many_stream_reuse_admission();
    check_reuse_multiplicity_reservation();
    check_ready_speculation_yields();
    check_active_reader_defers_publication();
    check_permanent_admission_boundaries();
    check_catalog_headroom();
    check_commit_outcomes();
    check_promotion_burst_coalescing();
    check_promotion_publication_progress();
    check_promotion_invalidation_order();
    check_promotion_rearm();
    check_promotion_eviction_absorption();
    check_promotion_commit_faults();
    check_promotion_clean_close();
    check_promotion_recency_eviction();
    check_bootstrap_and_intent();
    check_inflight_object_lease();
    check_failed_write_charges_once();
    check_orphan_revival();
    check_object_directory_retry();
    check_object_directory_batch();
    check_object_data_batch();
    check_postcommit_reclamation_cannot_undo_commit();
    check_shared_scope_payload_separation();
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

void check_shared_scope_payload_separation() {
    TempRoot root;
    ConversationStore store(options_for(root.path, uint64_t(1) << 30, 0));
    SavedConversation private_image = fixture(431, 64);
    private_image.checkpoints.clear();
    SavedConversation shared = private_image;
    shared.scope = ConversationScope{bytes_from("explicit-shared-prefix")};
    // Persist the exact-draft extent using the existing cells field, including
    // after reopening. This fixture does not claim numerical runtime validity.
    shared.kv.back().cells = int64_t(shared.live.ids.size()) - 1;
    ConversationStorePublishResult first;
    check(publish_and_wait(store, private_image, 8801, 1, first) &&
          first.status == ConversationStorePublishStatus::stored, "a private record is stored before shared export");
    const auto before = store.counters();
    std::string why;
    const ConversationStoreStreamId stream{0, uint16_t(ConversationStoreStreamKind::kv_k), 0, 0};
    std::pair<Digest, uint64_t> private_identity, shared_identity;
    check(identity_of(private_image, stream, private_identity, why) &&
          identity_of(shared, stream, shared_identity, why) && private_identity != shared_identity,
          "byte-identical private and shared K/V have scope-bound physical identities");
    ConversationStorePublish request;
    request.request_id = 8802;
    request.generation = 2;
    request.epoch = store.epoch();
    request.reuse.push_back({first.head_id, stream, 0, 0, private_image.kv.front().k.size()});
    check(store.publish(std::make_shared<SavedConversation>(shared), request, why) == kAccepted,
          "a shared-scope export with an unauthorized private donor is admitted without adoption");
    ConversationStorePublishResult second;
    check(store.wait_publish(second, 5000) && second.status == ConversationStorePublishStatus::stored &&
          second.reused_bytes == 0 && store.counters().objects == 2 * before.objects,
          "cross-scope bytes are written separately rather than adopted or deduplicated");
    const auto shared_query = query_of(shared, 8803, 3);
    const auto private_query = query_of(private_image, 8804, 4);
    const auto shared_matches = store.matches(shared_query.selector, shared_query.tokens, {});
    const auto private_matches = store.matches(private_query.selector, private_query.tokens, {});
    check(shared_matches.size() == 1 && shared_matches.front().head_id == second.head_id &&
          private_matches.size() == 1 && private_matches.front().head_id == first.head_id,
          "private and shared selectors cannot combine equal token histories");
    auto foreign = shared_query;
    foreign.selector.scope.value[0] ^= 0x80;
    check(store.matches(foreign.selector, foreign.tokens, {}).empty(),
          "a different credential-bound shared scope cannot discover an equal prefix");
    store.close();
    ConversationStore reopened(options_for(root.path, uint64_t(1) << 30, 0));
    check(reopened.demand(shared_query, why), "the shared record is demandable after a clean restart");
    ConversationStoreRead read;
    check(reopened.wait_ready(8803, 3, read, 5000) && read.status == ConversationStoreReadStatus::ready &&
          read.image && read.image->kv.back().cells == 63 && images_equal(*read.image, shared, why),
          "the unchanged disk fields preserve the exact shorter draft claim and scoped payload");
    read.image.reset();
    reopened.release_read(8803, 3);
}

void check_read_object_boundaries() {
    // The final object, empty streams, cross-stream transitions and a partial/full/partial ids chunk grid
    // must all preserve bytes with synchronous verified reads.
    for (const size_t tokens : {size_t(1023), size_t(1024), size_t(1025)}) {
        TempRoot root;
        ConversationStore store(options_for(root.path, uint64_t(1) << 30, 0));
        const SavedConversation image = fixture(401, tokens);
        ConversationStorePublishResult published;
        check(publish_and_wait(store, image, 1, 1, published) &&
                  published.status == ConversationStorePublishStatus::stored,
              "object boundary fixture is stored");
        std::string why;
        check(store.demand(query_of(image, 2, 1), why), "object boundary read is admitted");
        ConversationStoreRead restored;
        check(store.wait_ready(2, 1, restored, 5000) &&
                  restored.status == ConversationStoreReadStatus::ready && restored.image &&
                  images_equal(*restored.image, image, why),
              "synchronous reads preserve full and partial object boundaries byte for byte");
        restored.image.reset();
        store.release_read(2, 1);
        store.close();
    }
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
        auto options = options_for(root.path, uint64_t(1) << 30, 0);
        options.publication_timing = true;
        ConversationStore store(options);
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
        check(result.applied_ns != 0 && result.catalog_version != 0,
              "a stored publication reports its exact catalog visibility boundary");
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
        ConversationStoreLookupFrontier frontier;
        std::vector<ConversationStoreMatch> observed;
        store.for_each_match(query.selector, query.tokens, query.images,
            [&](const ConversationStoreMatch& match) { observed.push_back(match); }, &frontier);
        check(frontier.observed_ns >= result.applied_ns && frontier.catalog_version == result.catalog_version,
              "a later lookup reports the exact catalog version that contains the publication");
        check(!observed.empty() && observed.front().head_id == result.head_id,
              "the frontier-bearing lookup returns the published head");
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

// Completed verdicts still belong to their admitted callers. Fill the bound without harvesting, offer more than
// another bound's worth, then free exactly one slot and exercise normal, cancelled and close-time settlement.
void check_publication_result_backpressure() {
    for (int settlement = 0; settlement < 3; ++settlement) {
        TempRoot root;
        ConversationStoreOptions options = options_for(root.path, uint64_t(1) << 30, 0);
        options.max_pending_publishes = settlement == 0 ? 2 : 1;
        const uint64_t bound = uint64_t(options.max_pending_publishes) * 4 + 4;
        options.head_limit = size_t(bound + 2);
        ConversationStore store(options);
        auto image = std::make_shared<SavedConversation>(fixture(63, 32));
        const auto wait_idle = [&] {
            for (int waited = 0; waited < 5000 && store.writer_busy(); ++waited) ::usleep(1000);
            check(!store.writer_busy(), "a publication settles without harvesting its verdict");
        };
        const auto request_for = [&](uint64_t id) {
            ConversationStorePublish request;
            request.request_id = id;
            request.generation = 1000 + id;
            request.epoch = store.epoch();
            return request;
        };
        std::string why;
        for (uint64_t id = 1; id <= bound; ++id) {
            check(store.publish(image, request_for(id), why) == kAccepted,
                  "every publication up to the unreaped-result bound is admitted");
            wait_idle();
        }
        check_eq(store.counters().published, bound, "every admitted image finishes before the result queue is reaped");
        const uint64_t held = store.ram_held_bytes();
        const uint64_t deferred_before = store.counters().deferred_publishes;
        const uint64_t overflow_attempts = bound + 2;
        for (uint64_t attempt = 0; attempt < overflow_attempts; ++attempt) {
            check(store.publish(image, request_for(bound + 1), why) == kDeferred,
                  "a full result queue defers admission instead of losing an older verdict");
            check(!why.empty(), "result backpressure explains the deferral");
            wait_idle();
        }
        check_eq(store.counters().deferred_publishes, deferred_before + overflow_attempts,
                 "every completion-capacity deferral is counted exactly once");
        check_eq(store.counters().published, bound, "deferred images never become untracked publications");
        check_eq(store.ram_held_bytes(), held, "completion-capacity deferrals allocate no image or writer lease");
        check_eq(store.ram_lease_bytes(), 0, "the caller retains ownership during completion backpressure");

        ConversationStorePublishResult result;
        check(store.take_publish(result), "harvesting one verdict releases one completion slot");
        check_eq(result.request_id, 1, "backpressure preserves the oldest result's request identity");
        check_eq(result.generation, 1001, "backpressure preserves the oldest result's generation");
        check(result.status == ConversationStorePublishStatus::stored && result.released,
              "the oldest caller receives its original terminal outcome");
        const ConversationStorePublish retry = request_for(bound + 1);
        check(store.publish(image, retry, why) == kAccepted, "one harvested verdict permits exactly one new admission");
        if (settlement == 1) store.cancel_publication(retry.request_id, retry.generation);
        if (settlement == 2) store.close();
        else wait_idle();

        uint64_t received = 0;
        uint64_t stored = 1;   // the first result was harvested above
        while (store.take_publish(result)) {
            ++received;
            check_eq(result.request_id, received + 1, "each remaining result keeps its FIFO request identity");
            check_eq(result.generation, 1001 + received, "each remaining result keeps its capture generation");
            const bool made = result.status == ConversationStorePublishStatus::stored;
            if (made) ++stored;
            const bool terminal = made ||
                (received == bound && settlement == 1 && result.status == ConversationStorePublishStatus::stale) ||
                (received == bound && settlement == 2 && result.status == ConversationStorePublishStatus::disabled);
            check(terminal && result.released, "every admitted publication retains exactly one terminal verdict");
        }
        check_eq(received, bound, "no admitted caller is stranded and no deferred attempt emits a verdict");
        check_eq(store.counters().published, stored, "harvesting counts each completed publication exactly once");
        check(!store.wait_publish(result, 0), "an exhausted completion queue cannot report an outcome twice");
        check_eq(store.ram_lease_bytes(), 0, "all settled publications release their image lease");
        store.close();
    }
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

    // A demand with no published allowance must refuse synchronously. The production owner can then
    // reclaim a parked history from the refusal's quote; accepting here would leave the only owner
    // thread waiting for an allowance it cannot republish.
    const uint64_t before_demand = store.ram_held_bytes();
    ConversationStoreReadRefusal refusal;
    check(!store.demand(query_of(image, 300, 1), why, &refusal),
          "a demand without an allowance is refused");
    check(refusal.additional_ram_bytes > 0, "the zero-allowance refusal quotes recoverable RAM");
    check_eq(store.ram_held_bytes(), before_demand,
             "the refused demand acquires no descriptor or image charge");
    check_eq(store.publish_read_allowance(uint64_t(64) << 20), uint64_t(64) << 20, "the allowance is absolute");
    check(store.demand(query_of(image, 300, 1), why), "the demand is admitted once the allowance exists");
    ConversationStoreRead read;
    check(store.wait_ready(300, 1, read, 5000), "the admitted demand settles");
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
    check(result.applied_ns == 0 && result.catalog_version == 0,
          "a refused publication reports no catalog visibility boundary");
    check_eq(uint64_t(store.counters().heads), 0, "a refused publication authorizes no head");
    check_eq(store.disk_charged_bytes(), charged_before, "a refused publication leaves nothing charged");
    check(store.counters().refusals_floor > 0, "the floor refusal is counted");
    const ConversationStoreQuery query = query_of(image, 2, 1);
    check(store.matches(query.selector, query.tokens, query.images).empty(),
          "a refused publication stored nothing");
    store.close();
}

void check_corruption(bool truncate) {
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
            if (truncate) {
                check(::ftruncate(fd, 0) == 0, "the object's size is damaged");
            } else {
                uint8_t byte = 0;
                check(::read(fd, &byte, 1) == 1, "the object is readable");
                byte ^= 0x80;
                ::lseek(fd, 0, SEEK_SET);
                check(::write(fd, &byte, 1) == 1, "the object is damaged");
            }
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
    check(store.prefetch(query_of(image, 2, 1), why), "a speculative corrupt-read descriptor is accepted");
    check(await_status(store, 2, 1, ConversationStoreReadStatus::corrupt), "speculative integrity failure settles");
    const uint64_t corrupt_reads = store.counters().bytes_read;
    store.publish_read_allowance(uint64_t(64) << 20);
    store.publish_read_allowance(uint64_t(128) << 20);
    check(!store.refresh_speculation(2, 1), "catalog refresh cannot retry a corrupt speculative result");
    check_eq(store.counters().bytes_read, corrupt_reads, "allowance recovery never retries corrupt work");
    store.release_read(2, 1);
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
    check(!read_trace(records, "reissue", 2), "integrity failures have no fabricated recovery attempt");
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

void check_queued_read_protects_frontier() {
    TempRoot root;
    auto options = options_for(root.path, uint64_t(1) << 30, 0);
    options.head_limit = 3;
    ConversationStore store(options);
    const SavedConversation latest = fixture(601, 256);
    for (auto& checkpoint : const_cast<SavedConversation&>(latest).checkpoints) checkpoint.pinned = false;
    check(!latest.pinned(), "the frontier fixture carries no pin");
    ConversationStorePublishResult latest_result;
    check(publish_and_wait(store, latest, 1, 1, latest_result) &&
              latest_result.status == ConversationStorePublishStatus::stored,
          "the protected frontier is stored");
    const SavedConversation neighbour = fixture(602, 128);
    ConversationStorePublishResult neighbour_result;
    check(publish_and_wait(store, neighbour, 2, 2, neighbour_result) &&
              neighbour_result.status == ConversationStorePublishStatus::stored,
          "the neighbour head fills the ceiling");

    // A queued (admitted, not started) read has selected the frontier cut: until it starts or is retired, that
    // selection must hold the same protection a started reader's lease has. Otherwise publications under head
    // pressure retire the newest checkpoint out from under the queued demand, and its later run finds only an
    // older cut (the #243 latest-history loss).
    store.suspend_read_admission();
    std::string why;
    check(store.demand(query_of(latest, 801, 8), why), "the queued demand is admitted while admission is suspended");
    for (uint64_t id = 3; id <= 4; ++id) {
        const SavedConversation pressure = fixture(602 + id, 128);
        auto pressure_image = std::make_shared<SavedConversation>(pressure);
        ConversationStorePublish request;
        request.request_id = id;
        request.generation = id;
        check(store.publish(pressure_image, request, why) == kAccepted, "the ceiling-pressure publication is admitted");
        ConversationStorePublishResult stored;
        check(store.wait_publish(stored, 5000), "the pressure publication settles");
    }
    store.publish_read_allowance(uint64_t(64) << 20);   // resume admission
    ConversationStoreRead read;
    check(store.wait_ready(801, 8, read, 5000) && read.status == ConversationStoreReadStatus::ready,
          "the queued demand's selected frontier survived the pressure publications");
    check(read.image != nullptr && read.tokens == 256, "the demand mounts its full latest cut, not an older one");
    if (read.image) store.release_read(801, 8);
    ConversationStoreManifestInfo info;
    check(store.manifest_info(latest_result.head_id, info), "the frontier head is still authorized after the read");
    store.close();
}

// Reuse the real reader/transaction seams, never sleeps to guess whether an interleaving occurred.
struct SelectedHoldGate {
    std::mutex mutex;
    std::condition_variable cv;
    bool entered = false, open = false, timed_out = false;
    void pause() {
        std::unique_lock<std::mutex> lock(mutex);
        entered = true;
        cv.notify_all();
        if (!cv.wait_for(lock, std::chrono::seconds(5), [&] { return open; })) timed_out = true;
    }
    void await() {
        std::unique_lock<std::mutex> lock(mutex);
        check(cv.wait_for(lock, std::chrono::seconds(5), [&] { return entered; }),
              "the selected-head regression reached its real worker seam");
    }
    void resume() {
        std::lock_guard<std::mutex> lock(mutex);
        open = true;
        cv.notify_all();
    }
};

// Each permit releases one actual acknowledgement, not a timer. The metadata notification is first, followed by
// exactly one notification per verified/copied chunk; no callback exists in production.
struct ReadProgressGate {
    std::mutex mutex;
    std::condition_variable cv;
    uint64_t reached = 0, permitted = 0;
    bool open = false, timed_out = false;
    void pause() {
        std::unique_lock<std::mutex> lock(mutex);
        const uint64_t step = ++reached;
        cv.notify_all();
        if (!cv.wait_for(lock, std::chrono::seconds(5), [&] { return open || permitted >= step; })) timed_out = true;
    }
    bool await(uint64_t step) {
        std::unique_lock<std::mutex> lock(mutex);
        const bool entered = cv.wait_for(lock, std::chrono::seconds(5), [&] { return reached >= step; });
        check(entered, "the progress test reaches its actual acknowledgement seam");
        return entered;
    }
    void permit(uint64_t step) {
        std::lock_guard<std::mutex> lock(mutex);
        permitted = step;
        cv.notify_all();
    }
    void resume() {
        std::lock_guard<std::mutex> lock(mutex);
        open = true;
        cv.notify_all();
    }
};

void check_read_progress() {
    using State = ConversationStoreReadState;
    using Component = ConversationStoreReadComponent;
    TempRoot root;
    ConversationStore store(options_for(root.path, uint64_t(1) << 30, 0));
    auto image = split_fixture(971, 257, 3);
    image.stage_images.back().kv.push_back(image.stage_images.back().kv.front()); // extra pool = draft
    for (auto* stage : {&image, &image.stage_images[0], &image.stage_images[1]})
        for (auto& kv : stage->kv) kv.pooled.resize(9, 0x4a);
    ConversationStorePublishResult stored;
    check(publish_and_wait(store, image, 1, 1, stored), "the component-progress split image is stored");
    const auto manifests = directory_names(root.path + "/manifests");
    struct stat info {};
    check(manifests.size() == 1 && ::stat((root.path + "/manifests/" + manifests.front()).c_str(), &info) == 0,
          "the progress fixture has one actual encoded manifest");
    const uint64_t manifest_bytes = uint64_t(info.st_size);
    std::array<uint64_t, size_t(Component::count)> expected{};
    expected[size_t(Component::metadata)] = manifest_bytes;
    const auto add_record = [&](const ConversationCheckpoint& record) {
        expected[size_t(Component::metadata)] += record.ids.size() * sizeof(int32_t) +
                                                record.imgs.size() * sizeof(ConversationImageKey);
        expected[size_t(Component::recurrent)] += record.gdn.size();
        expected[size_t(Component::ple)] += record.ple.size();
        expected[size_t(Component::indexer)] += record.tails.size() + record.dead.size() + record.block_pos.size();
    };
    for (const auto* stage : {&image, &image.stage_images[0], &image.stage_images[1]}) {
        add_record(stage->live);
        for (const auto& checkpoint : stage->checkpoints) add_record(checkpoint);
        for (size_t i = 0; i < stage->kv.size(); ++i) {
            const auto& kv = stage->kv[i];
            const bool draft = stage == &image.stage_images.back() && i == 1;
            expected[size_t(draft ? Component::draft : Component::main_kv)] +=
                kv.k.size() + kv.v.size() + kv.k_scale.size() + kv.v_scale.size();
            expected[size_t(draft ? Component::draft : Component::indexer)] += kv.pooled.size();
        }
    }
    uint64_t notifications = 1;
    for (const auto& stream : conversation_store_streams(image))
        notifications += (stream.bytes + 4095) / 4096;

    const auto unknown = store.request_progress(2, 1);
    ConversationStoreRead read;
    check(unknown.state == State::unknown && !unknown.attempt_id &&
              !store.wait_ready(2, 1, read, 0), "an unknown request is not a successful wait");
    store.suspend_read_admission();
    std::string why;
    check(store.demand(query_of(image, 2, 1), why), "the component-progress demand owns a queued descriptor");
    const auto queued = store.request_progress(2, 1);
    check(queued.state == State::queued && queued.status == ConversationStoreReadStatus::none &&
              queued.attempt_id && !queued.totals_known, "queued ownership is distinct from an unknown request");
    check(!queued.materialize_started && !queued.prior_materialize_unmeasured,
          "a queued descriptor is not a materialization observation");
    check(!store.wait_ready(2, 1, read, 1) &&
              !store.request_progress(2, 1).acknowledged_after(queued),
          "a queue timer cannot report delivery or acknowledged progress");
    SelectedHoldGate start;
    ReadProgressGate gate;
    store.set_read_gate_for_test([](void* p) { static_cast<SelectedHoldGate*>(p)->pause(); }, &start);
    store.set_read_progress_gate_for_test([](void* p) { static_cast<ReadProgressGate*>(p)->pause(); }, &gate);
    store.publish_read_allowance(uint64_t(64) << 20);
    start.await();
    const auto active = store.request_progress(2, 1);
    check(active.state == State::materializing && active.attempt_id == queued.attempt_id &&
              !active.totals_known && !active.acknowledged_after(queued),
          "a reader reservation is materializing, not acknowledged I/O");
    check(active.materialize_started && !active.prior_materialize_unmeasured,
          "reader dispatch records an attempt without claiming acknowledged I/O");
    check(!store.wait_ready(2, 1, read, 1) &&
              !store.request_progress(2, 1).acknowledged_after(active) && !read.image,
          "a no-I/O worker stall stays invisible to the engine's byte-only heartbeat predicate");
    start.resume();
    auto previous = active;
    for (uint64_t step = 1; step <= notifications; ++step) {
        if (!gate.await(step)) break;
        const auto current = store.request_progress(2, 1);
        check(current.state == State::materializing && current.totals_known &&
                  current.attempt_id == active.attempt_id, "partial progress cannot settle or replace the image");
        check(current.acknowledged_after(previous), "each verification seam advances real acknowledged bytes");
        for (size_t i = 0; i < current.components.size(); ++i) {
            check_eq(current.components[i].total, expected[i], "component totals match the actual all-stage payload");
            check(current.components[i].acknowledged >= previous.components[i].acknowledged &&
                      current.components[i].acknowledged <= current.components[i].total,
                  "each component's acknowledged count is monotonic and bounded within the attempt");
        }
        if (step == 1)
            check_eq(current.components[size_t(Component::metadata)].acknowledged, manifest_bytes,
                     "only the validated manifest is acknowledged before the first object");
        check(!store.wait_ready(2, 1, read, 1) && !read.image &&
                  !store.request_progress(2, 1).acknowledged_after(current),
              "repeated polls/timers cannot mount a partial image or manufacture progress");
        previous = current;
        gate.permit(step);
    }
    gate.resume();
    check(await_status(store, 2, 1, ConversationStoreReadStatus::ready), "the complete materialization settles");
    const auto settled = store.request_progress(2, 1);
    check(settled.state == State::settled && settled.status == ConversationStoreReadStatus::ready,
          "successful settlement precedes whole-image hand-over");
    check(settled.materialize_started && !settled.prior_materialize_unmeasured,
          "successful settlement retains the current attempt's diagnostic dispatch bit");
    for (size_t i = 0; i < settled.components.size(); ++i) {
        check(expected[i] != 0, "the fixture exercises every restore component");
        check_eq(settled.components[i].acknowledged, expected[i], "terminal success acknowledges each entire component");
    }
    check(store.take_ready(2, 1, read) && read.image && images_equal(*read.image, image, why),
          "only the complete verified all-stage image is handed out");
    const auto handed = store.request_progress(2, 1);
    check(handed.state == State::handed_out && handed.components == settled.components &&
              !handed.acknowledged_after(settled), "hand-over changes ownership, not acknowledged byte progress");
    check(!store.wait_ready(2, 1, read, 0), "a handed-out image cannot be delivered twice");
    read.image.reset();
    store.release_read(2, 1);
    check(store.request_progress(2, 1).state == State::unknown,
          "a settled release erases the descriptor instead of retaining a tombstone");
    store.set_read_gate_for_test(nullptr, nullptr);
    store.set_read_progress_gate_for_test(nullptr, nullptr);
    store.close();
    check(!start.timed_out && !gate.timed_out, "all real progress gates were released explicitly");
}

void check_read_progress_failures() {
    using State = ConversationStoreReadState;
    using Component = ConversationStoreReadComponent;
    for (int damage = 0; damage < 3; ++damage) {
        TempRoot root;
        ConversationStore store(options_for(root.path, uint64_t(1) << 30, 0));
        const auto image = fixture(973, 1025, 1);
        ConversationStorePublishResult stored;
        check(publish_and_wait(store, image, 1, 1, stored), "the progress-corruption fixture is stored");
        const auto manifests = directory_names(root.path + "/manifests");
        if (manifests.empty()) { check(false, "the damage fixture needs its manifest"); continue; }
        const std::string manifest = root.path + "/manifests/" + manifests.front();
        struct stat info {};
        check(::stat(manifest.c_str(), &info) == 0, "the actual manifest byte count is available");
        std::string victim = manifest;
        if (damage != 2) {
            std::pair<Digest, uint64_t> identity;
            const ConversationStoreStreamId ids{0, uint16_t(ConversationStoreStreamKind::ids), 0, 0};
            std::string why;
            check(identity_of(image, ids, identity, why), "the first token chunk's physical identity is known");
            const auto hash = conversation_store_detail::object_digest(
                identity.first, ids.kind, identity.second, 4096,
                reinterpret_cast<const uint8_t*>(image.live.ids.data()));
            const std::string name = sha256_hex(hash);
            victim = root.path + "/objects/" + name.substr(0, 2) + "/" + name;
        }
        const int fd = ::open(victim.c_str(), O_RDWR | O_CLOEXEC);
        check(fd >= 0, "the exact first object or manifest opens for deterministic damage");
        if (fd >= 0) {
            if (damage == 1) check(::ftruncate(fd, 0) == 0, "a short object is introduced");
            else {
                uint8_t byte = 0;
                check(::pread(fd, &byte, 1, 0) == 1, "the damaged byte is read");
                byte ^= 0x80;
                check(::pwrite(fd, &byte, 1, 0) == 1, "a same-length digest mismatch is introduced");
            }
            ::close(fd);
        }
        std::string why;
        check(store.demand(query_of(image, 2, 1), why), "the damaged image's demand owns its descriptor");
        check(await_status(store, 2, 1, ConversationStoreReadStatus::corrupt), "the failed verification settles");
        const auto failed = store.request_progress(2, 1);
        check(failed.state == State::settled && failed.status == ConversationStoreReadStatus::corrupt,
              "settled is not synonymous with ready");
        check(failed.materialize_started && !failed.prior_materialize_unmeasured,
              "even a corrupt manifest records attempted materialization before byte progress exists");
        check(failed.totals_known == (damage != 2), "a corrupt manifest never publishes trusted component totals");
        for (size_t i = 0; i < failed.components.size(); ++i) {
            const uint64_t expected = damage != 2 && i == size_t(Component::metadata) ? uint64_t(info.st_size) : 0;
            check_eq(failed.components[i].acknowledged, expected,
                     "a failed copy/digest acknowledges no byte of its chunk");
            check(failed.components[i].acknowledged <= failed.components[i].total,
                  "failure keeps all component counters bounded");
        }
        ConversationStoreRead read;
        check(store.wait_ready(2, 1, read, 0) && read.status == ConversationStoreReadStatus::corrupt && !read.image,
              "a delivered terminal failure never exposes its partially written target");
        store.release_read(2, 1);
        store.close();
    }

    TempRoot root;
    ConversationStore store(options_for(root.path, uint64_t(1) << 30, 0));
    const auto image = fixture(975, 64);
    ConversationStorePublishResult stored;
    check(publish_and_wait(store, image, 1, 1, stored), "the cancellation-progress image is stored");
    store.suspend_read_admission();
    std::string why;
    check(store.demand(query_of(image, 2, 1), why), "a queued cancellation owns a descriptor");
    const auto queued = store.request_progress(2, 1);
    store.cancel_read(2, 1);
    ConversationStoreRead read;
    const auto cancelled = store.request_progress(2, 1);
    check(!store.wait_ready(2, 1, read, 0) && cancelled.state == State::settled &&
              cancelled.status == ConversationStoreReadStatus::cancelled &&
              !cancelled.acknowledged_after(queued), "false also describes settled undeliverable cancellation");
    check(!cancelled.materialize_started && !cancelled.prior_materialize_unmeasured,
          "queued cancellation never invents attempted materialization");
    store.release_read(2, 1);
    SelectedHoldGate gate;
    store.set_read_gate_for_test([](void* p) { static_cast<SelectedHoldGate*>(p)->pause(); }, &gate);
    store.publish_read_allowance(uint64_t(64) << 20);
    check(store.demand(query_of(image, 3, 1), why), "an active release owns its allocation frame");
    gate.await();
    const auto active = store.request_progress(3, 1);
    store.release_read(3, 1);
    const auto released = store.request_progress(3, 1);
    check(released.state == State::released && !released.acknowledged_after(active) &&
              !store.wait_ready(3, 1, read, 0), "released ownership is observable only while the active frame unwinds");
    check(store.demand(query_of(image, 3, 1), why), "a fresh owner can reuse an actively released identifier");
    const auto replacement = store.request_progress(3, 1);
    check(replacement.state == State::queued && replacement.attempt_id != released.attempt_id &&
              !replacement.acknowledged_after(released), "polling prefers the fresh owner, not its old released worker");
    store.cancel_read(3, 1);
    store.release_read(3, 1);
    gate.resume();
    store.set_read_gate_for_test(nullptr, nullptr);
    store.close();
    check(store.request_progress(3, 1).state == State::unknown && !gate.timed_out,
          "unwinding the released frame erases its bounded diagnostic state");
}

void check_read_progress_reissue() {
    TempRoot root;
    ConversationStore store(options_for(root.path, uint64_t(1) << 30, 0));
    const auto old_image = fixture(977, 128);
    const auto new_image = appended_turn(old_image, 256, 977);
    ConversationStorePublishResult old_head, new_head;
    check(publish_and_wait(store, old_image, 1, 1, old_head), "the earlier progress cut is stored");
    const auto query = query_of(new_image, 3, 1);
    std::string why;
    check(store.prefetch(query, why) && await_status(store, 3, 1, ConversationStoreReadStatus::ready),
          "the earlier selected attempt fully materializes");
    const auto previous = store.request_progress(3, 1);
    check(previous.totals_known && previous.components[0].acknowledged > 0, "the old attempt had real progress");
    check(publish_and_wait(store, new_image, 2, 2, new_head), "a longer eligible image is published");
    SelectedHoldGate gate;
    store.set_read_gate_for_test([](void* p) { static_cast<SelectedHoldGate*>(p)->pause(); }, &gate);
    check(store.demand(query, why), "demand reissues the revoked earlier selection");
    gate.await();
    const auto retry = store.request_progress(3, 1);
    check(retry.state == ConversationStoreReadState::materializing && retry.attempt_id != previous.attempt_id &&
              retry.head_id == new_head.head_id && !retry.totals_known && !retry.acknowledged_after(previous),
          "a real reissue binds a new attempt/manifest without crediting discarded bytes");
    check(retry.materialize_started && retry.prior_materialize_unmeasured,
          "reissuing a completed image preserves the discarded attempt's unavailable duration");
    for (const auto& component : retry.components)
        check(component.total == 0 && component.acknowledged == 0, "reissue resets every component together");
    gate.resume();
    ConversationStoreRead read;
    check(store.wait_ready(3, 1, read, 5000) && read.image && images_equal(*read.image, new_image, why),
          "the reissued read still delivers exactly one complete image");
    const auto ready = store.request_progress(3, 1);
    check(ready.attempt_id == retry.attempt_id && ready.acknowledged_after(retry),
          "the new attempt's verified bytes, not its wakeup, count as progress");
    check(ready.materialize_started && ready.prior_materialize_unmeasured,
          "a later ready sample cannot erase prior unsampled materialization");
    for (const auto& component : ready.components)
        check_eq(component.acknowledged, component.total, "the new successful attempt completes all components");
    read.image.reset();
    store.release_read(3, 1);
    store.set_read_gate_for_test(nullptr, nullptr);
    store.close();
    check(!gate.timed_out, "the replacement attempt's real worker was explicitly released");
}

// Each permit releases one real transaction seam. A timeout is only a deadlock guard; assertions below observe
// reached seams, committed results and joined workers, never the absence of progress during a guessed delay.
struct PromotionCatalogGate {
    std::mutex mutex;
    std::condition_variable cv;
    uint64_t before = 0, staged = 0, permit_before = 0, permit_staged = 0;
    bool open = false, timed_out = false;

    void pause(bool after) {
        std::unique_lock<std::mutex> lock(mutex);
        const uint64_t visit = after ? ++staged : ++before;
        cv.notify_all();
        if (!cv.wait_for(lock, std::chrono::seconds(5), [&] {
                return open || (after ? permit_staged : permit_before) >= visit;
            }))
            timed_out = true;
    }
    void await(bool after, uint64_t visit) {
        std::unique_lock<std::mutex> lock(mutex);
        check(cv.wait_for(lock, std::chrono::seconds(5), [&] { return (after ? staged : before) >= visit; }),
              "the promotion regression reaches its real catalog seam");
    }
    void resume(bool after, uint64_t visit) {
        std::lock_guard<std::mutex> guard(mutex);
        (after ? permit_staged : permit_before) = visit;
        cv.notify_all();
    }
    void release() {
        std::lock_guard<std::mutex> guard(mutex);
        open = true;
        cv.notify_all();
    }
    void check_visits(uint64_t expected) {
        std::lock_guard<std::mutex> guard(mutex);
        check(!timed_out, "every promotion catalog seam was explicitly released");
        check_eq(before, expected, "only the expected catalog transactions started");
        check_eq(staged, expected, "only the expected catalog transactions were staged");
    }
};

template <class Predicate>
bool await_store_progress(Predicate progress) {
    for (int waited = 0; waited < 5000; ++waited) {
        if (progress()) return true;
        ::usleep(1000);
    }
    return progress();
}

uint64_t shutdown_now_ms() {
    return uint64_t(std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now().time_since_epoch()).count());
}

void check_shutdown_admission() {
    TempRoot root;
    SelectedHoldGate gate;
    ConversationStore store(options_for(root.path, uint64_t(1) << 30, 0));
    const auto image = fixture(750, 128);
    ConversationStorePublishResult head;
    check(publish_and_wait(store, image, 1, 1, head), "the shutdown fence has a durable history");
    check(store.shutdown_state() == ConversationStoreShutdownState::none, "an open store has no shutdown request");
    check(std::strcmp(conversation_store_shutdown_state_name(store.shutdown_state()), "none") == 0,
          "the initial shutdown state has a stable name");

    std::string why;
    check(store.demand(query_of(image, 2, 1), why), "the shutdown fence first acquires an image lease");
    ConversationStoreRead held;
    check(store.wait_ready(2, 1, held, 5000) && held.image, "the image lease is handed out before shutdown");
    const uint64_t lease = store.ram_lease_bytes();
    check(lease > 0, "the handed-out image has a live charge");
    store.suspend_read_admission();
    check(store.demand(query_of(image, 3, 1), why), "a demand queues behind the handed-out lease");
    check(store.request_status(3, 1) == ConversationStoreReadStatus::none, "the second demand has not started");

    store.set_catalog_gate_for_test([](void* p, bool staged) {
        if (staged) static_cast<SelectedHoldGate*>(p)->pause();
    }, &gate);
    store.confirmed_restore(head.head_id);
    gate.await(); // the real writer cannot start the publication offered below
    auto outgoing = std::make_shared<SavedConversation>(fixture(751, 128));
    ConversationStorePublish publication;
    publication.request_id = 4;
    publication.generation = 4;
    check(store.publish(outgoing, publication, why) == kAccepted, "an unstarted publication owns its image");
    const std::string catalog_before = file_digest(root.path + "/catalog");
    store.request_shutdown(UINT64_MAX);
    check(store.shutdown_state() == ConversationStoreShutdownState::pending, "request_shutdown publishes pending");
    check(store.describe().find("shutdown: pending") != std::string::npos, "describe exposes the pending state");
    check(store.run_intent_active(), "request_shutdown cannot speculatively clear the run intent");
    check(file_digest(root.path + "/catalog") == catalog_before, "request_shutdown performs no catalog transaction");
    check(store.request_status(3, 1) == ConversationStoreReadStatus::disabled, "shutdown retires the queued demand");
    ConversationStorePublishResult cancelled;
    check(store.take_publish(cancelled) && cancelled.request_id == 4 &&
              cancelled.status == ConversationStorePublishStatus::disabled && cancelled.released,
          "shutdown settles and releases the unstarted publication");
    check(!store.writer_busy() && outgoing.use_count() == 1, "an unstarted publication keeps no image alias");
    check_eq(store.ram_lease_bytes(), lease, "shutdown never refunds a handed-out image lease");

    const auto before = store.counters();
    const uint64_t ram = store.ram_held_bytes(), disk = store.disk_charged_bytes(), epoch = store.epoch();
    const uint64_t allowance = store.ram_read_budget();
    ConversationStoreReadRefusal refusal{123};
    check(!store.demand(query_of(image, 2, 1), why), "shutdown fences read adoption even for an existing lease");
    check(!store.demand(query_of(image, 5, 1), why, &refusal) && refusal.additional_ram_bytes == 0,
          "shutdown fences a fresh demand before admission accounting");
    check(!store.prefetch(query_of(image, 6, 1), why), "shutdown fences a fresh prefetch");
    check(!store.prefetch(query_of(fixture(752, 128), 7, 1), why), "shutdown fences even a catalog-miss descriptor");
    check(!store.refresh_speculation(3, 1), "shutdown cannot refresh speculation");
    check(!store.invalidate_domain("after shutdown request"), "shutdown fences durable invalidation");
    check(store.publish(outgoing, publication, why) == kRefused, "shutdown fences new publications");
    store.confirmed_restore(head.head_id);
    std::vector<ConversationStoreStreamInfo> streams;
    check(!store.manifest_streams(head.head_id, streams), "shutdown fences new diagnostic filesystem reads");
    check(store.matches(selector_of(image), query_of(image, 8, 1).tokens, {}).empty(),
          "shutdown exposes no new restorable match");
    check_eq(store.publish_read_allowance(allowance + 4096), allowance, "an allowance update cannot reopen shutdown");
    const auto after = store.counters();
    check_eq(after.matches, before.matches, "fenced reads never count a match");
    check_eq(after.misses, before.misses, "fenced reads never count a miss");
    check_eq(after.refused_budget, before.refused_budget, "shutdown is not a budget refusal");
    check_eq(after.refused_limits, before.refused_limits, "shutdown is not a limits refusal");
    check_eq(after.epoch_advances, before.epoch_advances, "fenced invalidation changes no epoch accounting");
    check_eq(store.epoch(), epoch, "fenced invalidation changes no epoch");
    check_eq(store.ram_held_bytes(), ram, "fenced admissions allocate no store-held RAM");
    check_eq(store.disk_charged_bytes(), disk, "fenced admissions alter no disk charge");
    check(store.describe().find("promotion pending no") != std::string::npos,
          "a confirmed restore cannot re-arm policy work during shutdown");
    ConversationStoreRead blocked;
    check(!store.wait_ready(blocked, 60000) && !store.wait_ready(3, 1, blocked, 60000),
          "shutdown wakes both forms of read waiter without waiting for the worker");
    check(!store.wait_publish(cancelled, 60000), "shutdown wakes publication waiters without waiting for the worker");

    gate.resume();
    check(store.finish_shutdown() == ConversationStoreShutdownState::durable_clean,
          "the fenced owner finishes cleanly after the real writer drains");
    check(!gate.timed_out, "the gated writer was explicitly released");
    check_eq(store.ram_lease_bytes(), lease, "finishing shutdown still owns the caller's live lease");
    check(held.image && images_equal(*held.image, image, why), "a previously acquired image stays valid after shutdown");
    held.image.reset();
    store.release_read(2, 1);
    store.release_read(3, 1);
    check_eq(store.ram_lease_bytes(), 0, "lease release remains legal after terminal shutdown");
}

void check_shutdown_pending() {
    TempRoot root;
    SelectedHoldGate gate;
    const auto options = options_for(root.path, uint64_t(1) << 30, 0);
    ConversationStore store(options);
    const auto image = fixture(753, 128);
    ConversationStorePublishResult head;
    check(publish_and_wait(store, image, 1, 1, head), "pending observation has a real stored image");
    store.set_read_gate_for_test([](void* p) { static_cast<SelectedHoldGate*>(p)->pause(); }, &gate);
    std::string why;
    check(store.demand(query_of(image, 2, 1), why), "the reader reserves its real materialization");
    gate.await();
    const uint64_t epoch = store.epoch(), deadline = shutdown_now_ms() + 1000;
    store.request_shutdown(deadline);
    const uint64_t reserved = store.ram_held_bytes();
    ConversationStoreShutdownState results[2]{}, observed = ConversationStoreShutdownState::none;
    std::thread first([&] { results[0] = store.finish_shutdown(); });
    std::thread second([&] { results[1] = store.finish_shutdown(); });
    std::thread observer([&] { observed = store.shutdown_state(); });
    observer.join(); // finish must not hold the store mutex while joining the gated reader
    check(observed == ConversationStoreShutdownState::pending, "another thread can cheaply observe a blocked shutdown");
    check(store.run_intent_active(), "a blocked worker cannot be reported as a clean exit");
    check_eq(store.ram_held_bytes(), reserved, "a live reader keeps its reservation until its frame unwinds");
    {
        ConversationStore contender(options);
        check(!contender.enabled(), "pending shutdown keeps the root lock while the reader is live");
        contender.request_shutdown(UINT64_MAX);
        check(contender.finish_shutdown() == ConversationStoreShutdownState::quiescent_unclean,
              "a non-owner never reports a durable-clean shutdown");
    }
    std::this_thread::sleep_until(std::chrono::steady_clock::time_point(std::chrono::milliseconds(deadline + 1)));
    gate.resume();
    first.join();
    second.join();
    check(!gate.timed_out, "pending observation never depended on the worker gate timing out");
    check(results[0] == ConversationStoreShutdownState::quiescent_unclean && results[1] == results[0],
          "concurrent finish calls join once and report the same actual terminal state");
    check(store.shutdown_state() == results[0] && store.finish_shutdown() == results[0],
          "terminal shutdown observation and repeated finish are idempotent");
    check(store.describe().find("shutdown: quiescent_unclean") != std::string::npos,
          "describe distinguishes quiescent unclean from pending");
    check(store.run_intent_active(), "a deadline that expired while joining leaves the intent armed");
    check_eq(store.ram_lease_bytes(), 0, "the retired worker delivers and leases no image");
    store.release_read(2, 1);
    ConversationStore reopened(options);
    check(reopened.enabled(), "an expired cooperative drain eventually releases its root lock");
    check_eq(reopened.epoch(), epoch + 1, "deadline expiry during worker drain forces restart distrust");
    check(reopened.matches(selector_of(image), query_of(image, 3, 1).tokens, {}).empty(),
          "the expired worker drain never authorizes reuse on restart");
    reopened.close();
}

void check_shutdown_deadlines() {
    for (int mode = 0; mode != 3; ++mode) {
        TempRoot root;
        PromotionCatalogGate gate;
        gate.release();
        const auto options = options_for(root.path, uint64_t(1) << 30, 0);
        const auto image = fixture(754, 128);
        ConversationStore store(options);
        ConversationStorePublishResult head;
        check(publish_and_wait(store, image, 1, 1, head), "deadline shutdown starts with a durable history");
        const uint64_t epoch = store.epoch();
        const std::string catalog = file_digest(root.path + "/catalog");
        store.set_catalog_gate_for_test([](void* p, bool staged) {
            static_cast<PromotionCatalogGate*>(p)->pause(staged);
        }, &gate);
        const uint64_t deadline = mode == 0 ? 0 : (mode == 1 ? shutdown_now_ms() - 1 : shutdown_now_ms() + 60000);
        store.request_shutdown(deadline);
        store.request_shutdown(deadline);
        check(store.shutdown_state() == ConversationStoreShutdownState::pending && store.run_intent_active(),
              "requesting any deadline is pending, not a durable-clean promise");
        const auto expected = mode == 2 ? ConversationStoreShutdownState::durable_clean
                                         : ConversationStoreShutdownState::quiescent_unclean;
        check(store.finish_shutdown() == expected, "the deadline gates finalization only after drain");
        check(!store.enabled() && !store.writer_busy(), "a terminal shutdown is quiescent and releases serving authority");
        check(store.run_intent_active() == (mode != 2), "only a proven clean finalization removes intent");
        check(std::strcmp(conversation_store_shutdown_state_name(expected),
                          mode == 2 ? "durable_clean" : "quiescent_unclean") == 0,
              "terminal shutdown names are stable");
        gate.check_visits(mode == 2 ? 1 : 0);
        if (mode != 2)
            check(file_digest(root.path + "/catalog") == catalog, "a zero or expired deadline starts no final catalog");
        const std::string after = file_digest(root.path + "/catalog");
        store.request_shutdown(UINT64_MAX);
        store.close();
        check(store.finish_shutdown() == expected && file_digest(root.path + "/catalog") == after,
              "repeated request, close and finish cannot change a terminal outcome");

        // Reopening while the old object still exists also proves finish released the actual owner lock.
        ConversationStore reopened(options);
        check(reopened.enabled(), "terminal shutdown released the root for another owner");
        if (mode == 2) {
            check_eq(reopened.epoch(), epoch, "future-deadline durable clean preserves the epoch");
            std::string why;
            check(reopened.demand(query_of(image, 2, 1), why), "the cleanly finalized history remains selectable");
            ConversationStoreRead read;
            check(reopened.wait_ready(2, 1, read, 5000) && read.image && images_equal(*read.image, image, why),
                  "future-deadline finalization preserves the exact stored image on restart");
            read.image.reset();
            reopened.release_read(2, 1);
        } else {
            check_eq(reopened.epoch(), epoch + 1, "zero or expired deadline forces restart distrust");
            const auto query = query_of(image, 2, 1);
            check(reopened.matches(query.selector, query.tokens, query.images).empty(),
                  "a quiescent unclean history cannot be reused after restart");
        }
        reopened.close();
    }
}

void check_shutdown_failures() {
    for (const char* step : {"catalog-rename", "catalog-sync", "catalog-apply",
                             "intent-unlink-sync", "intent-rearm-sync"}) {
        TempRoot root;
        const auto options = options_for(root.path, uint64_t(1) << 30, 0);
        const auto image = fixture(755, 128);
        ConversationStore store(options);
        ConversationStorePublishResult head;
        check(publish_and_wait(store, image, 1, 1, head), "failed finalization starts with a durable history");
        const uint64_t epoch = store.epoch();
        if (std::strcmp(step, "intent-rearm-sync") == 0) store.set_publish_fault_for_test("intent-unlink-sync");
        store.set_publish_fault_for_test(step);
        store.request_shutdown(shutdown_now_ms() + 60000);
        check(store.finish_shutdown() == ConversationStoreShutdownState::quiescent_unclean,
              "catalog or intent failure is never reported as durable clean");
        check(store.run_intent_active(), "unproven catalog or intent durability keeps intent armed");
        store.close();
        check(store.shutdown_state() == ConversationStoreShutdownState::quiescent_unclean,
              "close cannot retry a failed finalization into a clean result");
        ConversationStore reopened(options);
        check(reopened.enabled(), "a failed finalization still releases its drained root lock");
        check_eq(reopened.epoch(), epoch + 1, "failed finalization causes restart distrust");
        const auto query = query_of(image, 2, 1);
        check(reopened.matches(query.selector, query.tokens, query.images).empty(),
              "a failed finalization authorizes no reuse on restart");
        reopened.close();
    }
}

void check_shutdown_late_finalization() {
    for (bool fail : {false, true}) {
        TempRoot root;
        PromotionCatalogGate gate;
        const auto options = options_for(root.path, uint64_t(1) << 30, 0);
        const auto image = fixture(756, 128);
        ConversationStore store(options);
        ConversationStorePublishResult head;
        check(publish_and_wait(store, image, 1, 1, head), "late finalization has a durable history");
        const uint64_t epoch = store.epoch();
        store.set_catalog_gate_for_test([](void* p, bool staged) {
            static_cast<PromotionCatalogGate*>(p)->pause(staged);
        }, &gate);
        const uint64_t deadline = shutdown_now_ms() + 1000;
        store.request_shutdown(deadline);
        ConversationStoreShutdownState result = ConversationStoreShutdownState::none;
        std::thread finisher([&] { result = store.finish_shutdown(); });
        gate.await(false, 1);
        check(shutdown_now_ms() < deadline, "real finalization began before its deadline");
        check(store.shutdown_state() == ConversationStoreShutdownState::pending && store.run_intent_active(),
              "starting finalization proves neither durability nor intent removal");
        store.request_shutdown(0); // cannot interrupt a finalization already inside the real transaction
        gate.resume(false, 1);
        gate.await(true, 1);
        std::this_thread::sleep_until(std::chrono::steady_clock::time_point(std::chrono::milliseconds(deadline + 1)));
        check(shutdown_now_ms() > deadline, "the real final transaction is held until after its deadline");
        check(store.shutdown_state() == ConversationStoreShutdownState::pending && store.run_intent_active(),
              "passing the deadline cannot fabricate a terminal result or clear intent");
        if (fail) store.set_publish_fault_for_test("catalog-sync");
        gate.release();
        finisher.join();
        const auto expected = fail ? ConversationStoreShutdownState::quiescent_unclean
                                   : ConversationStoreShutdownState::durable_clean;
        check(result == expected && store.shutdown_state() == expected,
              "finalization completed after its deadline reports its actual durable outcome");
        check(store.run_intent_active() == fail, "late success alone proves durable intent removal");
        gate.check_visits(1);
        ConversationStore reopened(options);
        check(reopened.enabled(), "late finalization eventually releases the root lock");
        check_eq(reopened.epoch(), epoch + (fail ? 1 : 0), "restart trusts only the real late finalization outcome");
        const auto query = query_of(image, 2, 1);
        check(reopened.matches(query.selector, query.tokens, query.images).empty() == fail,
              "a deadline crossing does not substitute for the final transaction's result");
        reopened.close();
    }
}

void check_shutdown_request_policy() {
    TempRoot root;
    const auto options = options_for(root.path, uint64_t(1) << 30, 0);
    const auto image = fixture(757, 128);
    uint64_t epoch = 0;
    {
        ConversationStore store(options);
        ConversationStorePublishResult head;
        check(publish_and_wait(store, image, 1, 1, head), "deadline tightening has a durable history");
        epoch = store.epoch();
        store.request_shutdown(shutdown_now_ms() + 60000);
        store.request_shutdown(shutdown_now_ms() - 1);
        store.request_shutdown(UINT64_MAX);
        store.close();
        check(store.shutdown_state() == ConversationStoreShutdownState::quiescent_unclean && store.run_intent_active(),
              "a later request or legacy close cannot extend a stricter deadline");
    }
    {
        ConversationStore store(options);
        check_eq(store.epoch(), epoch + 1, "a tightened shutdown deadline leaves restart distrust intact");
        ConversationStorePublishResult head;
        check(publish_and_wait(store, image, 2, 2, head), "destructor policy has a newly durable history");
        epoch = store.epoch();
        store.request_shutdown(shutdown_now_ms() + 60000);
        // No finish: the destructor must tighten this request to zero, not silently finalize it cleanly.
    }
    {
        ConversationStore store(options);
        check_eq(store.epoch(), epoch + 1, "destruction never upgrades a pending future-deadline request into clean");
        check_eq(store.counters().heads, 0, "destruction leaves previously published histories distrusted");
        store.request_shutdown(0);
        store.request_shutdown(UINT64_MAX);
        store.close();
        check(store.shutdown_state() == ConversationStoreShutdownState::quiescent_unclean && store.run_intent_active(),
              "zero is the strictest deadline and legacy close cannot regain permission to finalize");
    }
}

SavedConversation promotion_fixture(uint64_t seed, int64_t tokens = 128) {
    auto image = fixture(seed, tokens);
    for (auto& checkpoint : image.checkpoints) checkpoint.pinned = false;
    return image;
}

void check_coalesced_publication_result_backpressure() {
    TempRoot root;
    PromotionCatalogGate gate;
    const auto options = options_for(root.path, uint64_t(1) << 30, 0);
    const uint64_t bound = uint64_t(options.max_pending_publishes) * 4 + 4;
    ConversationStore store(options);
    const SavedConversation image = promotion_fixture(707, 32);
    ConversationStorePublishResult predecessor;
    check(publish_and_wait(store, image, 1, 1, predecessor),
          "coalescing backpressure starts from a retained predecessor");
    store.set_catalog_gate_for_test([](void* p, bool staged) {
        static_cast<PromotionCatalogGate*>(p)->pause(staged);
    }, &gate);
    store.confirmed_restore(predecessor.head_id);
    gate.await(false, 1);   // hold the worker on durable work: every offered publication remains not-yet-started
    auto shared = std::make_shared<SavedConversation>(image);
    const auto request_for = [&](uint64_t id) {
        ConversationStorePublish request;
        request.request_id = 100 + id;
        request.generation = 200 + id;
        request.epoch = store.epoch();
        request.supersedes = predecessor.head_id;
        request.cover = true;
        return request;
    };
    std::string why;
    for (uint64_t id = 1; id <= bound; ++id)
        check(store.publish(shared, request_for(id), why) == kAccepted,
              "same-lineage replacements each reserve their own result slot");
    check_eq(store.counters().coalesced_publishes, bound - 1,
             "all but the active writer have one counted coalesced outcome");
    for (uint64_t attempt = 0; attempt < bound + 1; ++attempt)
        check(store.publish(shared, request_for(bound + 1), why) == kDeferred,
              "the active writer's reserved verdict prevents coalescing past the result bound");
    check_eq(store.counters().coalesced_publishes, bound - 1,
             "a completion-capacity deferral cannot displace the admitted writer");
    ConversationStorePublishResult result;
    check(store.take_publish(result) && result.request_id == 101 && result.generation == 201 &&
              result.status == ConversationStorePublishStatus::coalesced && result.released,
          "the oldest coalesced caller receives its own terminal verdict");
    check(store.publish(shared, request_for(bound + 1), why) == kAccepted,
          "harvesting one coalesced verdict permits one additional same-lineage handoff");
    check(store.publish(shared, request_for(bound + 2), why) == kDeferred,
          "the newly admitted writer retains its own completion reservation");
    check_eq(store.counters().deferred_publishes, bound + 2, "coalescing backpressure counts every deferred offer once");
    gate.release();
    check(await_store_progress([&] { return !store.writer_busy(); }), "the final admitted cover settles");
    uint64_t received = 0;
    while (store.take_publish(result)) {
        ++received;
        check_eq(result.request_id, 101 + received, "coalesced pressure preserves every request identity");
        check_eq(result.generation, 201 + received, "coalesced pressure preserves every generation");
        check(result.released && result.status == (received == bound ? ConversationStorePublishStatus::replaced
                                                                    : ConversationStorePublishStatus::coalesced),
              "each admitted cover gets exactly its own coalesced or committed verdict");
    }
    check_eq(received, bound, "all coalesced and active reservations become terminal results without overflow");
    check_eq(store.counters().coalesced_publishes, bound, "each displaced image is counted exactly once");
    store.close();
    gate.check_visits(3); // held promotion, final cover, clean close
}

void check_promotion_burst_coalescing() {
    TempRoot root;
    PromotionCatalogGate gate;
    ConversationStore store(options_for(root.path, uint64_t(1) << 30, 0));
    ConversationStorePublishResult head;
    check(publish_and_wait(store, promotion_fixture(701), 1, 1, head), "the promotion burst has a retained head");
    store.set_catalog_gate_for_test([](void* p, bool staged) {
        static_cast<PromotionCatalogGate*>(p)->pause(staged);
    }, &gate);
    store.confirmed_restore(head.head_id);
    gate.await(false, 1);
    constexpr uint64_t restores = 512;
    for (uint64_t i = 1; i < restores; ++i) store.confirmed_restore(head.head_id);
    check_eq(store.counters().promotions_coalesced, restores - 1, "a restore burst shares one pending policy hint");
    check_eq(store.counters().promotions_applied, 0, "no promotion is reported applied before its durable transaction");
    check(store.describe().find("durable queue 0") != std::string::npos,
          "confirmed restores allocate no durable work descriptors");
    gate.resume(false, 1);
    gate.await(true, 1);
    check(store.describe().find("promotion pending no") != std::string::npos,
          "copying the policy candidate consumes its pending hint atomically");
    gate.release();
    store.close();
    check_eq(store.counters().promotions_applied, 1, "one idle transaction applies the whole restore burst");
    check_eq(store.counters().promotions_absorbed, 0, "a pure burst needed no authority transaction absorption");
    check_eq(store.counters().promotions_refused, 0, "the coalesced promotion succeeds");
    check(!store.run_intent_active(), "the burst still closes only after its final durable catalog");
    gate.check_visits(2); // one idle promotion and the existing final close transaction
}

void check_promotion_publication_progress() {
    TempRoot root;
    PromotionCatalogGate gate;
    ConversationStore store(options_for(root.path, uint64_t(1) << 30, 0));
    ConversationStorePublishResult head;
    check(publish_and_wait(store, promotion_fixture(702), 1, 1, head), "publication progress has an existing history");
    store.set_catalog_gate_for_test([](void* p, bool staged) {
        static_cast<PromotionCatalogGate*>(p)->pause(staged);
    }, &gate);
    store.confirmed_restore(head.head_id);
    gate.await(false, 1);
    gate.resume(false, 1);
    gate.await(true, 1);
    constexpr uint64_t restores = 512;
    for (uint64_t i = 0; i < restores; ++i) store.confirmed_restore(head.head_id);
    const uint64_t source_before = store.source_bytes_read_for_test();
    ConversationStorePublish request;
    request.request_id = 2;
    request.generation = 2;
    std::string why;
    check(store.publish(std::make_shared<SavedConversation>(promotion_fixture(703)), request, why) == kAccepted,
          "a publication is runnable while a promotion has another pending burst");
    gate.resume(true, 1);
    gate.await(false, 2);
    check(store.source_bytes_read_for_test() > source_before, "the next transaction belongs to the real publication");
    check_eq(store.counters().promotions_applied, 1, "a pending promotion cannot jump ahead of a runnable publication");
    check_eq(store.counters().promotions_absorbed, 1, "the publication snapshot absorbs the earlier pending burst");
    check_eq(store.counters().promotions_coalesced, restores - 1, "the waiting burst still uses just one hint");
    check(store.describe().find("promotion pending no") != std::string::npos,
          "the publication consumes only the hints preceding its own candidate copy");
    gate.release();
    ConversationStorePublishResult stored;
    check(store.wait_publish(stored, 5000) && stored.status == ConversationStorePublishStatus::stored,
          "publication completes without draining one transaction per confirmed restore");
    store.close();
    check_eq(store.counters().promotions_applied, 1, "an absorbed burst schedules no redundant idle transaction");
    gate.check_visits(3); // in-flight promotion, publication, final close
}

void check_promotion_invalidation_order() {
    TempRoot root;
    PromotionCatalogGate gate;
    ConversationStore store(options_for(root.path, uint64_t(1) << 30, 0));
    ConversationStorePublishResult head;
    check(publish_and_wait(store, promotion_fixture(704), 1, 1, head), "invalidation ordering has an existing history");
    store.set_catalog_gate_for_test([](void* p, bool staged) {
        static_cast<PromotionCatalogGate*>(p)->pause(staged);
    }, &gate);
    store.confirmed_restore(head.head_id);
    gate.await(false, 1);
    gate.resume(false, 1);
    gate.await(true, 1);
    store.confirmed_restore(head.head_id);
    const uint64_t epoch = store.epoch();
    bool first_ok = false, second_ok = false;
    std::thread first([&] { first_ok = store.invalidate_domain("first urgent invalidation"); });
    check(await_store_progress([&] { return store.epoch() == epoch + 1; }),
          "the first invalidation is enqueued while catalog I/O is gated");
    std::thread second([&] { second_ok = store.invalidate_domain("second urgent invalidation"); });
    check(await_store_progress([&] { return store.epoch() == epoch + 2; }),
          "the second invalidation owns its distinct advanced epoch");
    check(store.describe().find("durable queue 2") != std::string::npos,
          "urgent invalidations remain distinct durable work items");
    const uint64_t source_before = store.source_bytes_read_for_test();
    ConversationStorePublish request;
    request.request_id = 2;
    request.generation = 2;
    request.epoch = epoch + 2;
    std::string why;
    check(store.publish(std::make_shared<SavedConversation>(promotion_fixture(705)), request, why) == kAccepted,
          "a current-epoch publication waits behind both urgent invalidations");
    gate.resume(true, 1);
    gate.await(false, 2);
    check_eq(store.source_bytes_read_for_test(), source_before, "the first urgent transaction precedes publication");
    check(store.describe().find("durable queue 1") != std::string::npos,
          "taking one invalidation leaves the other queued");
    gate.resume(false, 2);
    gate.await(true, 2);
    check_eq(store.counters().promotions_absorbed, 1, "an invalidation snapshot absorbs the older promotion hint");
    store.confirmed_restore(head.head_id); // after the copy: the next invalidation must consume a fresh hint
    gate.resume(true, 2);
    gate.await(false, 3);
    check_eq(store.source_bytes_read_for_test(), source_before, "the second urgent transaction also precedes publication");
    check_eq(store.counters().heads, 0, "the first invalidation durably retired the old-epoch history");
    gate.resume(false, 3);
    gate.await(true, 3);
    check_eq(store.counters().promotions_absorbed, 2, "a restore after the first copy re-arms a separate absorption");
    gate.resume(true, 3);
    gate.await(false, 4);
    check(store.source_bytes_read_for_test() > source_before, "publication starts only after both invalidations apply");
    check_eq(store.counters().promotions_applied, 1, "policy work cannot interpose between urgent work and publication");
    gate.release();
    first.join();
    second.join();
    check(first_ok && second_ok, "each urgent invalidation receives its own durable success");
    ConversationStorePublishResult stored;
    check(store.wait_publish(stored, 5000) && stored.status == ConversationStorePublishStatus::stored &&
              stored.epoch == epoch + 2,
          "the runnable publication commits in the twice-advanced epoch");
    store.close();
    check(!store.run_intent_active(), "ordered invalidations preserve a clean final close");
    gate.check_visits(5); // promotion, two invalidations, publication, final close
}

void check_promotion_rearm() {
    for (bool publication : {false, true}) {
        TempRoot root;
        PromotionCatalogGate gate;
        auto options = options_for(root.path, uint64_t(1) << 30, 0);
        options.head_limit = 2; // only one protected head: losing the later hint changes the next real victim
        ConversationStore store(options);
        const auto older_image = promotion_fixture(706);
        ConversationStorePublishResult older, later;
        check(publish_and_wait(store, older_image, 1, 1, older) &&
                  publish_and_wait(store, promotion_fixture(707), 2, 2, later),
              "the re-arm race has two unpinned histories");
        store.set_catalog_gate_for_test([](void* p, bool staged) {
            static_cast<PromotionCatalogGate*>(p)->pause(staged);
        }, &gate);
        if (publication) {
            ConversationStorePublish cover;
            cover.request_id = 3;
            cover.generation = 3;
            cover.supersedes = older.head_id;
            cover.cover = true;
            std::string why;
            check(store.publish(std::make_shared<SavedConversation>(older_image), cover, why) == kAccepted,
                  "a covered publication provides the older immutable snapshot");
        } else {
            store.confirmed_restore(older.head_id);
        }
        gate.await(false, 1);
        gate.resume(false, 1);
        gate.await(true, 1);
        store.confirmed_restore(later.head_id);
        check(store.describe().find("promotion pending yes") != std::string::npos,
              "a restore after a staged candidate copy re-arms policy persistence");
        gate.resume(true, 1);
        gate.await(false, 2);
        check_eq(store.counters().promotions_applied, publication ? 0 : 1,
                 "the older transaction settles before the re-armed policy transaction");
        check_eq(store.counters().promotions_absorbed, 0, "a late hint is not incorrectly absorbed by the older copy");
        uint64_t replaced = older.head_id;
        if (publication) {
            ConversationStorePublishResult cover;
            check(store.wait_publish(cover, 5000) && cover.status == ConversationStorePublishStatus::replaced,
                  "the older covering publication still succeeds with exact predecessor authority");
            replaced = cover.head_id;
        }
        gate.resume(false, 2);
        gate.await(true, 2);
        check(store.describe().find("promotion pending no") != std::string::npos,
              "the follow-up snapshot consumes the re-armed hint");
        ConversationStorePublish pressure;
        pressure.request_id = 4;
        pressure.generation = 4;
        std::string why;
        check(store.publish(std::make_shared<SavedConversation>(promotion_fixture(708)), pressure, why) == kAccepted,
              "real head pressure waits for the staged follow-up");
        gate.release();
        ConversationStorePublishResult stored;
        check(store.wait_publish(stored, 5000) && stored.status == ConversationStorePublishStatus::stored,
              "head pressure commits after the follow-up");
        ConversationStoreManifestInfo info;
        check(store.manifest_info(later.head_id, info) && !store.manifest_info(replaced, info),
              "the late restore survives the older apply and protects its own history from eviction");
        store.close();
        check_eq(store.counters().promotions_applied, publication ? 1 : 2,
                 "a post-copy restore causes exactly one follow-up policy transaction");
        gate.check_visits(4);
    }
}

void check_promotion_eviction_absorption() {
    TempRoot root;
    auto options = options_for(root.path, uint64_t(1) << 30, 0);
    options.head_limit = 3;
    const auto large = promotion_fixture(710, 4096);
    const auto middle = promotion_fixture(711);
    const auto newest = promotion_fixture(712);
    ConversationStorePublishResult oldest_head, middle_head, newest_head, incoming;
    uint64_t charged = 0;
    {
        ConversationStore store(options);
        check(publish_and_wait(store, large, 1, 1, oldest_head) &&
                  publish_and_wait(store, middle, 2, 2, middle_head) &&
                  publish_and_wait(store, newest, 3, 3, newest_head),
              "quota pressure has one large and two small unpinned histories");
        charged = store.disk_charged_bytes();
        store.close();
    }
    {
        PromotionCatalogGate gate;
        options.quota_bytes = charged + catalog_copy_headroom(root.path);
        ConversationStore store(options);
        check(store.enabled(), "the eviction absorption store opens with its real catalog-copy headroom");
        store.set_catalog_gate_for_test([](void* p, bool staged) {
            static_cast<PromotionCatalogGate*>(p)->pause(staged);
        }, &gate);
        store.confirmed_restore(newest_head.head_id);
        gate.await(false, 1);
        gate.resume(false, 1);
        gate.await(true, 1);
        store.confirmed_restore(newest_head.head_id);
        ConversationStorePublish request;
        request.request_id = 4;
        request.generation = 4;
        std::string why;
        check(store.publish(std::make_shared<SavedConversation>(promotion_fixture(713, 4096)), request, why) == kAccepted,
              "the next runnable publication must evict real bytes before writing its replacement");
        gate.resume(true, 1);
        gate.await(false, 2);
        check(store.retains_head(oldest_head.head_id),
              "selecting a quota victim does not remove its authoritative head before commit");
        check_eq(store.counters().promotions_absorbed, 1, "the eviction candidate absorbs the earlier pending hint");
        store.confirmed_restore(middle_head.head_id);
        check(store.describe().find("promotion pending yes") != std::string::npos,
              "a restore after the eviction copy re-arms policy work");
        gate.resume(false, 2);
        gate.await(true, 2);
        gate.resume(true, 2);
        gate.await(false, 3);
        check_eq(store.counters().evictions, 1, "one large victim is sufficient; no extra reclamation was promised");
        check_eq(store.counters().promotions_absorbed, 2, "the publication absorbs the post-eviction-copy hint");
        check_eq(store.counters().promotions_applied, 1, "publication and eviction avoid redundant policy transactions");
        gate.release();
        check(store.wait_publish(incoming, 5000) && incoming.status == ConversationStorePublishStatus::stored,
              "the quota eviction and following publication both commit");
        ConversationStoreManifestInfo info;
        check(!store.manifest_info(oldest_head.head_id, info) && store.manifest_info(middle_head.head_id, info) &&
                  store.manifest_info(newest_head.head_id, info),
              "only the oldest unprotected history was retired for physical space");
        check(store.disk_charged_bytes() <= store.disk_quota_bytes(), "promotion absorption cannot spend quota headroom");
        store.close();
        gate.check_visits(4); // idle promotion, quota eviction, publication, final close
    }
    {
        options.quota_bytes = uint64_t(1) << 30;
        ConversationStore store(options);
        std::string why;
        check(store.demand(query_of(middle, 5, 1), why), "the post-copy restored history remains selectable after restart");
        ConversationStoreRead read;
        check(store.wait_ready(5, 1, read, 5000) && read.image && images_equal(*read.image, middle, why),
              "absorbed policy changes never alter the restored model state");
        read.image.reset();
        store.release_read(5, 1);
        ConversationStorePublishResult pressure;
        check(publish_and_wait(store, promotion_fixture(714), 6, 6, pressure), "a later head pressure publication commits");
        ConversationStoreManifestInfo info;
        check(store.manifest_info(middle_head.head_id, info) && store.manifest_info(newest_head.head_id, info) &&
                  !store.manifest_info(incoming.head_id, info),
              "the post-eviction-copy protection survived apply, absorption and clean restart");
        store.close();
    }
}

void check_promotion_recency_eviction() {
    TempRoot root;
    PromotionCatalogGate gate;
    auto options = options_for(root.path, uint64_t(1) << 30, 0);
    options.head_limit = 4;
    ConversationStore store(options);
    ConversationStorePublishResult heads[4];
    for (uint64_t i = 0; i < 4; ++i)
        check(publish_and_wait(store, promotion_fixture(720 + i), i + 1, i + 1, heads[i]),
              "the recency policy starts with four unpinned histories");
    store.set_catalog_gate_for_test([](void* p, bool staged) {
        static_cast<PromotionCatalogGate*>(p)->pause(staged);
    }, &gate);
    store.confirmed_restore(heads[0].head_id);
    gate.await(false, 1);
    for (uint64_t i : {1, 2, 3, 0, 2}) store.confirmed_restore(heads[i].head_id);
    check_eq(store.counters().promotions_coalesced, 5, "coalescing preserves every restore's immediate recency touch");
    gate.resume(false, 1);
    gate.await(true, 1);
    ConversationStorePublish request;
    request.request_id = 5;
    request.generation = 5;
    std::string why;
    check(store.publish(std::make_shared<SavedConversation>(promotion_fixture(724)), request, why) == kAccepted,
          "head pressure is admitted after the coalesced recency sequence");
    gate.release();
    ConversationStorePublishResult stored;
    check(store.wait_publish(stored, 5000) && stored.status == ConversationStorePublishStatus::stored,
          "head pressure commits under the unchanged protected-segment policy");
    ConversationStoreManifestInfo info;
    check(!store.manifest_info(heads[1].head_id, info), "the oldest demoted protected head is the eviction victim");
    for (uint64_t i : {0, 2, 3})
        check(store.manifest_info(heads[i].head_id, info), "the newest three confirmed histories retain protection");
    check_eq(store.counters().heads, 4, "coalesced policy persistence preserves the head ceiling");
    store.close();
    check_eq(store.counters().promotions_applied, 1, "unchanged recency decisions need only one policy transaction");
    gate.check_visits(3);
}

void check_promotion_commit_faults() {
    for (const char* step : {"catalog-rename", "catalog-sync", "catalog-apply"}) {
        TempRoot root;
        const auto options = [&] {
            auto out = options_for(root.path, uint64_t(1) << 30, 0);
            out.head_limit = 2;
            return out;
        }();
        const bool uncertain = std::strcmp(step, "catalog-rename") != 0;
        ConversationStorePublishResult older, later;
        uint64_t epoch = 0;
        {
            PromotionCatalogGate gate;
            ConversationStore store(options);
            check(publish_and_wait(store, promotion_fixture(730), 1, 1, older) &&
                      publish_and_wait(store, promotion_fixture(731), 2, 2, later),
                  "promotion fault injection starts with two durable unpinned histories");
            epoch = store.epoch();
            const uint64_t charged = store.disk_charged_bytes();
            const std::string before = file_digest(root.path + "/catalog");
            const auto files = file_inventory(root.path);
            store.set_catalog_gate_for_test([](void* p, bool staged) {
                static_cast<PromotionCatalogGate*>(p)->pause(staged);
            }, &gate);
            store.confirmed_restore(older.head_id);
            gate.await(false, 1);
            gate.resume(false, 1);
            gate.await(true, 1);
            if (uncertain) store.confirmed_restore(later.head_id); // an uncertainty must discard even a re-armed hint
            store.set_publish_fault_for_test(step);
            gate.release();
            check(await_store_progress([&] { return store.counters().promotions_refused == 1; }),
                  "the injected idle promotion reaches its refused outcome");
            check_eq(store.counters().promotions_applied, 0, "a refused or uncertain promotion is never counted applied");
            check(store.describe().find("promotion pending no") != std::string::npos,
                  "a refused promotion cannot reschedule itself in a busy loop");
            const std::string after = file_digest(root.path + "/catalog");
            if (!uncertain) {
                check(store.healthy(), "a pre-rename promotion refusal leaves the committed authority usable");
                check(after == before, "a failed promotion rename changes no durable catalog bytes");
                check_eq(store.disk_charged_bytes(), charged, "a pre-rename promotion refusal refunds only its removed copy");
                check(file_inventory(root.path) == files, "a pre-rename refusal retains every authorized file");
                store.close();
                check(!store.run_intent_active(), "the ordinary refusal can still finish its existing final close");
                check(file_digest(root.path + "/catalog") != before,
                      "the final transaction persists live policy left by the refused hint");
                gate.check_visits(2); // refused idle transaction and final close, never a self-retry
            } else {
                check(!store.healthy(), "post-rename promotion uncertainty latches the store fault");
                check(after != before, "an uncertain promotion records that its catalog rename really happened");
                check(store.disk_charged_bytes() >= charged, "uncertain policy persistence credits no unconfirmed bytes");
                const auto retained = file_inventory(root.path);
                store.confirmed_restore(older.head_id);
                check(!store.invalidate_domain("after uncertain promotion"), "invalidation cannot write uncertain authority");
                std::string why;
                ConversationStorePublish blocked;
                blocked.request_id = 3;
                blocked.generation = 3;
                check(store.publish(std::make_shared<SavedConversation>(promotion_fixture(732)), blocked, why) == kRefused,
                      "an uncertain promotion refuses further publication");
                const auto query = query_of(promotion_fixture(730), 4, 1);
                check(store.matches(query.selector, query.tokens, query.images).empty(),
                      "an uncertain promotion prevents reuse");
                store.close();
                check(store.run_intent_active(), "an uncertain promotion leaves the run intent armed");
                check(file_digest(root.path + "/catalog") == after && file_inventory(root.path) == retained,
                      "joined workers and close write and reclaim nothing after promotion uncertainty");
                gate.check_visits(1);
            }
            check_eq(store.counters().promotions_refused, 1, "the refused promotion is attempted only once");
        }
        ConversationStore reopened(options);
        check(reopened.enabled(), "the promotion-fault root can be re-opened from its real authority");
        if (uncertain) {
            check_eq(reopened.epoch(), epoch + 1, "an uncertain promotion's armed intent forces a new epoch");
            check_eq(reopened.counters().heads, 0, "no history from the uncertain run is reused");
        } else {
            ConversationStorePublishResult pressure;
            check(publish_and_wait(reopened, promotion_fixture(733), 5, 5, pressure),
                  "publication remains usable after a refused promotion and clean close");
            ConversationStoreManifestInfo info;
            check(reopened.manifest_info(older.head_id, info) && !reopened.manifest_info(later.head_id, info),
                  "the refused hint's live protection was retained by the final durable transaction");
        }
        reopened.close();
    }
}

void check_promotion_clean_close() {
    // The direct restore/close seam has no delay: either the writer already owns one attempt or the final
    // transaction owns the hint. In both schedules close is the drain barrier, not a timing assertion.
    {
        TempRoot root;
        const auto image = promotion_fixture(740);
        auto options = options_for(root.path, uint64_t(1) << 30, 0);
        PromotionCatalogGate gate;
        gate.release();
        {
            ConversationStore store(options);
            ConversationStorePublishResult head;
            check(publish_and_wait(store, image, 1, 1, head), "immediate close starts with a durable history");
            store.set_catalog_gate_for_test([](void* p, bool staged) {
                static_cast<PromotionCatalogGate*>(p)->pause(staged);
            }, &gate);
            store.confirmed_restore(head.head_id);
            store.close();
            const auto counters = store.counters();
            check(counters.promotions_applied <= 1 && counters.promotions_refused == 0,
                  "immediate close never drains a per-restore policy queue");
            check(!store.run_intent_active(), "immediate close clears intent only after its final catalog");
            store.confirmed_restore(head.head_id);
            check(store.describe().find("promotion pending no") != std::string::npos,
                  "confirmed_restore cannot re-arm work after close");
            gate.check_visits(counters.promotions_applied + 1);
        }
        ConversationStore reopened(options);
        std::string why;
        check(reopened.demand(query_of(image, 2, 1), why), "an immediately closed restore remains selectable");
        ConversationStoreRead read;
        check(reopened.wait_ready(2, 1, read, 5000) && read.image && images_equal(*read.image, image, why),
              "immediate close preserves exact model-state bytes");
        read.image.reset();
        reopened.release_read(2, 1);
        reopened.close();
    }
    for (const char* final_fault : {"", "catalog-rename", "catalog-sync", "catalog-apply"}) {
        TempRoot root;
        auto options = options_for(root.path, uint64_t(1) << 30, 0);
        options.head_limit = 2;
        ConversationStorePublishResult older, later;
        uint64_t epoch = 0;
        {
            PromotionCatalogGate gate;
            ConversationStore store(options);
            check(publish_and_wait(store, promotion_fixture(741), 1, 1, older) &&
                      publish_and_wait(store, promotion_fixture(742), 2, 2, later),
                  "the close race has an older snapshot and a distinct late-restored history");
            epoch = store.epoch();
            store.set_catalog_gate_for_test([](void* p, bool staged) {
                static_cast<PromotionCatalogGate*>(p)->pause(staged);
            }, &gate);
            store.confirmed_restore(older.head_id);
            gate.await(false, 1);
            gate.resume(false, 1);
            gate.await(true, 1);
            store.confirmed_restore(later.head_id);
            std::thread closer([&] { store.close(); });
            check(await_store_progress([&] {
                return store.describe().find("(admission suspended)") != std::string::npos;
            }), "close enters its drain while the older catalog I/O is gated");
            check(store.describe().find("promotion pending no") != std::string::npos,
                  "close defers the re-armed hint to its existing final transaction");
            store.confirmed_restore(older.head_id);
            gate.resume(true, 1);
            gate.await(false, 2);
            check(store.run_intent_active(), "the final transaction still owns an armed intent before staging");
            gate.resume(false, 2);
            gate.await(true, 2);
            check(store.run_intent_active(), "staging the final catalog cannot clear intent before durable success");
            const std::string before_final = file_digest(root.path + "/catalog");
            if (*final_fault) store.set_publish_fault_for_test(final_fault);
            gate.release();
            closer.join();
            check(store.run_intent_active() == bool(*final_fault),
                  "only a durably successful final catalog clears the deferred promotion's intent");
            if (std::strcmp(final_fault, "catalog-rename") == 0)
                check(file_digest(root.path + "/catalog") == before_final,
                      "a refused final rename leaves the earlier durable authority untouched");
            else
                check(file_digest(root.path + "/catalog") != before_final,
                      "a successful or uncertain final rename records the late policy snapshot");
            check_eq(store.counters().promotions_applied, 1, "close runs no extra policy attempt for the deferred flag");
            gate.check_visits(2);
        }
        ConversationStore reopened(options);
        if (*final_fault) {
            check_eq(reopened.epoch(), epoch + 1, "a failed final promotion leaves intent armed for the next epoch");
            check_eq(reopened.counters().heads, 0, "a refused or uncertain final transaction promises no reusable history");
        } else {
            ConversationStorePublishResult pressure;
            check(publish_and_wait(reopened, promotion_fixture(743), 3, 3, pressure),
                  "the clean-close race retains usable publication authority");
            ConversationStoreManifestInfo info;
            check(reopened.manifest_info(later.head_id, info) && !reopened.manifest_info(older.head_id, info),
                  "the final close preserved a late policy update even after dropping its scheduling flag");
        }
        reopened.close();
    }
}

void check_selected_retirement_allocation_retry() {
    TempRoot root;
    auto options = options_for(root.path, uint64_t(1) << 30, 0);
    options.head_limit = 1;
    ConversationStore store(options);
    const auto original = fixture(631, 256);
    auto successor = original;
    successor.live.gdn.assign(successor.live.gdn.size(), 92);
    ConversationStorePublishResult first;
    check(publish_and_wait(store, original, 1, 1, first), "allocation retry starts with a real stored predecessor");
    std::string why;
    check(store.demand(query_of(original, 831, 1), why), "the predecessor receives a request-owned hold");
    ConversationStoreRead held;
    check(store.wait_ready(831, 1, held, 5000) && held.image && images_equal(*held.image, original, why),
          "the held predecessor has verified original bytes");
    ConversationStorePublish cover;
    cover.request_id = 2;
    cover.generation = 2;
    cover.supersedes = first.head_id;
    cover.cover = true;
    check(store.publish(std::make_shared<SavedConversation>(successor), cover, why) == kAccepted,
          "a covered successor shares objects while changing one state stream");
    ConversationStorePublishResult replacement;
    check(store.wait_publish(replacement, 5000) && replacement.status == ConversationStorePublishStatus::replaced,
          "the successor durably replaces the held catalog head");
    store.set_publish_fault_for_test("retire-object-prepare");
    held.image.reset();
    store.release_read(831, 1);
    check_eq(store.ram_lease_bytes(), 0, "the failed retirement releases the caller's image lease");
    store.close(); // the final writer/reclamation pass retries retirement before any survivor is verified
    ConversationStore reopened(options);
    reopened.publish_read_allowance(uint64_t(64) << 20);
    check(reopened.demand(query_of(successor, 832, 1), why), "a fresh store selects the retained successor");
    ConversationStoreRead restored;
    check(reopened.wait_ready(832, 1, restored, 5000) && restored.image &&
              restored.head_id == replacement.head_id && images_equal(*restored.image, successor, why),
          "allocation failure and retirement retry cannot delete the successor's shared object bytes");
    restored.image.reset();
    reopened.release_read(832, 1);
    check_eq(reopened.ram_lease_bytes(), 0, "the survivor's image lease is released once after retirement retry");
    reopened.close();
}

void check_selected_equal_cut_order() {
    TempRoot root;
    auto options = options_for(root.path, uint64_t(1) << 30, 0);
    options.head_limit = 3;
    ConversationStore store(options);
    const auto older = fixture(641, 256);
    auto newest = older;
    newest.live.gdn.assign(newest.live.gdn.size(), 117);
    const auto neighbour = fixture(642, 128);
    ConversationStorePublishResult older_head, newest_head, neighbour_head;
    check(publish_and_wait(store, older, 1, 1, older_head), "an older equal-length history is stored");
    check(publish_and_wait(store, newest, 2, 2, newest_head), "a newer equal-length history is stored");
    check(publish_and_wait(store, neighbour, 3, 3, neighbour_head), "an independent revoking demand is stored");
    SelectedHoldGate gate;
    store.set_catalog_gate_for_test([](void* context, bool staged) {
        if (staged) static_cast<SelectedHoldGate*>(context)->pause();
    }, &gate);
    ConversationStorePublish pressure;
    pressure.request_id = 4;
    pressure.generation = 4;
    pressure.supersedes = newest_head.head_id;
    pressure.cover = true;
    std::string why;
    check(store.publish(std::make_shared<SavedConversation>(appended_turn(newest, 384, 641)), pressure, why) == kAccepted,
          "a covered successor stages retirement of the newer equal-cut head");
    gate.await();
    store.suspend_read_admission();
    const auto future = query_of(newest, 841, 1);
    check(store.prefetch(future, why), "the request selects the newer head after retirement staging");
    check(store.demand(query_of(neighbour, 842, 1), why), "a different demand revokes its speculative RAM");
    store.release_read(842, 1);
    gate.resume();
    ConversationStorePublishResult published;
    check(store.wait_publish(published, 5000) && published.status == ConversationStorePublishStatus::replaced,
          "staged retirement completes while the selected immutable authority remains held");
    store.set_catalog_gate_for_test(nullptr, nullptr);
    ConversationStoreManifestInfo info;
    check(!store.manifest_info(newest_head.head_id, info) && store.manifest_info(older_head.head_id, info),
          "only the older equal-cut head remains catalog-visible");
    check(store.demand(future, why), "the revoked request reuses its original selected authority");
    store.publish_read_allowance(uint64_t(64) << 20);
    ConversationStoreRead read;
    check(store.wait_ready(841, 1, read, 5000) && read.image && read.head_id == newest_head.head_id &&
              images_equal(*read.image, newest, why),
          "equal-cut catalog lookup cannot downgrade a retained newer selection");
    read.image.reset();
    store.release_read(841, 1);
    store.close();
    check(!gate.timed_out, "equal-cut ordering is established by the real commit gate");
}

void check_budget_deferred_prefetch() {
    // Exercise accepted descriptors, not admission refusals: zero allowance, aggregate resident pressure (the
    // production diagnostic), suspension, and an occupied incoming slot all settle without doing object I/O.
    for (int blocked = 0; blocked < 4; ++blocked) {
        StoreReadTraceCapture trace;
        TempRoot root;
        auto options = options_for(root.path, uint64_t(1) << 30, 0);
        options.max_read_requests = blocked == 3 ? 2 : 1;
        ConversationStore store(options);
        const SavedConversation image = fixture(621, 256);
        ConversationStorePublishResult published;
        check(publish_and_wait(store, image, 1, 1, published), "the deferred-offer head is stored");
        const uint64_t baseline = store.ram_held_bytes();
        std::string why;
        // Learn the actual charged peak from a no-I/O accepted descriptor, without duplicating admission math.
        store.publish_read_allowance(0);
        check(store.prefetch(query_of(image, 930, 1), why), "the peak probe is accepted");
        check(await_status(store, 930, 1, ConversationStoreReadStatus::refused_budget), "the peak probe settles");
        store.release_read(930, 1);
        const auto probe_records = trace.records();
        const auto* probe = read_trace(probe_records, "admission", 930);
        const uint64_t peak = probe ? probe->number("peak_bytes") : 0;
        check(peak > baseline && peak > 0, "the probe exposes the real bounded peak");
        uint64_t low = 0;
        ConversationStoreRead occupied;
        if (blocked == 1) {
            low = peak + baseline / 2;
            store.publish_read_allowance(low);
        } else if (blocked >= 2) {
            low = uint64_t(64) << 20;
            store.publish_read_allowance(low);
            if (blocked == 2) store.suspend_read_admission();
            else {
                check(store.demand(query_of(image, 931, 1), why), "an earlier request occupies the only image slot");
                check(store.wait_ready(931, 1, occupied, 5000) && occupied.image, "the earlier request owns its lease");
            }
        }
        const uint64_t io_before = store.counters().bytes_read;
        check(store.prefetch(query_of(image, 932, 1), why), "the real selected offer descriptor is accepted");
        check(await_status(store, 932, 1, ConversationStoreReadStatus::refused_budget),
              "an accepted offer settles as budget-deferred");
        check_eq(store.counters().bytes_read, io_before, "budget-deferred work performs zero payload I/O");
        check(!store.refresh_speculation(932, 1), "a catalog sweep does not poll an unchanged deferred head");
        if (blocked != 2) {
            const uint64_t refusals = store.counters().refused_budget;
            store.publish_read_allowance(low);
            store.publish_read_allowance(low);
            check(store.request_status(932, 1) == ConversationStoreReadStatus::refused_budget,
                  "unchanged unavailable capacity does not requeue a deferred offer");
            check_eq(store.counters().refused_budget, refusals, "unchanged capacity does not create refusal attempts");
        }
        if (blocked == 3) {
            occupied.image.reset();
            store.release_read(931, 1);
        }
        SelectedHoldGate gate;
        store.set_read_gate_for_test([](void* p) { static_cast<SelectedHoldGate*>(p)->pause(); }, &gate);
        store.publish_read_allowance(uint64_t(64) << 20);
        gate.await();
        const uint64_t active_bytes = store.ram_held_bytes();
        store.publish_read_allowance(uint64_t(64) << 20);
        check(!store.refresh_speculation(932, 1), "an active recovered offer cannot be refreshed");
        check_eq(store.ram_held_bytes(), active_bytes, "allowance publication never refunds an active reservation");
        gate.resume();
        check(await_status(store, 932, 1, ConversationStoreReadStatus::ready),
              "budget recovery makes the same head background-ready BEFORE any demand");
        store.set_read_gate_for_test(nullptr, nullptr);
        check(!gate.timed_out, "the recovered reader was released explicitly");
        const uint64_t ready_bytes = store.ram_held_bytes();
        check(ready_bytes <= store.ram_read_budget(), "recovery stays within the unchanged RAM envelope");
        check(store.demand(query_of(image, 932, 1), why), "demand adopts the already-ready recovered descriptor");
        ConversationStoreRead read;
        check(store.wait_ready(932, 1, read, 5000) && read.image && read.head_id == published.head_id &&
                  read.tokens == 256 && images_equal(*read.image, image, why),
              "recovery restores the original selected head and cut byte for byte");
        read.image.reset();
        store.release_read(932, 1);
        check_eq(store.ram_lease_bytes(), 0, "recovery releases every incoming image lease");
        check_eq(store.ram_held_bytes(), baseline, "recovery returns exactly to the original resident accounting");
        store.close();
        const auto records = trace.records();
        unsigned reissues = 0, starts = 0;
        for (const auto& record : records) {
            if (record.number("request_id") != 932) continue;
            if (record.text("event") == "reissue") ++reissues;
            if (record.text("event") == "read_start") ++starts;
        }
        const auto* deferred = read_trace(records, "settled", 932, "no_io");
        const auto* retry = read_trace(records, "reissue", 932, "allowance_recovery");
        check(deferred && retry && deferred->text("status") == "refused_budget" &&
                  !deferred->number("io_attempted") &&
                  retry->number("previous_attempt_id") == deferred->number("attempt_id"),
              "the accepted no-I/O refusal and its recovery have linked attempt identities");
        check_eq(reissues, 1, "one meaningful capacity recovery causes exactly one retry");
        check_eq(starts, 1, "the recovered offer materializes once, not again at demand adoption");
    }
}

void check_budget_deferred_expanded_peak() {
    StoreReadTraceCapture trace;
    TempRoot root;
    ConversationStore store(options_for(root.path, uint64_t(1) << 30, 0));
    const SavedConversation image = fixture(624, 256);
    ConversationStorePublishResult published;
    check(publish_and_wait(store, image, 1, 1, published), "the expanded-peak head is stored");
    SelectedHoldGate gate;
    store.set_read_gate_for_test([](void* p) { static_cast<SelectedHoldGate*>(p)->pause(); }, &gate);
    std::string why;
    check(store.prefetch(query_of(image, 935, 1), why), "the expanded-peak offer is accepted");
    gate.await();
    // The active reader already owns its original reservation. Leave exactly
    // that much room, so decoded restore scaffolding discovers the larger peak.
    const uint64_t admitted_bytes = store.ram_held_bytes();
    store.publish_read_allowance(admitted_bytes);
    gate.resume();
    check(await_status(store, 935, 1, ConversationStoreReadStatus::refused_budget),
          "decoded restore scaffolding exceeds the original read reservation");
    store.set_read_gate_for_test(nullptr, nullptr);
    check(!gate.timed_out, "the expanded-peak reader was released explicitly");
    const auto refused_progress = store.request_progress(935, 1);
    check(refused_progress.state == ConversationStoreReadState::settled && refused_progress.totals_known &&
              refused_progress.components[0].acknowledged > 0,
          "a decoded budget refusal keeps its attempt's validated manifest progress, not a successful image");
    check(refused_progress.materialize_started && !refused_progress.prior_materialize_unmeasured,
          "a decoded budget refusal is attempted work, unlike an unstarted admission refusal");
    const auto first = trace.records();
    check_eq(read_trace_count(first, "read_start", 935), 1,
             "the first refusal follows one real manifest read");
    store.publish_read_allowance(admitted_bytes + 1);
    const auto insufficient = trace.records();
    check_eq(read_trace_count(insufficient, "reissue", 935), 0,
             "capacity below the discovered restore peak does not repeat materialization");
    const auto unchanged_progress = store.request_progress(935, 1);
    check(unchanged_progress.attempt_id == refused_progress.attempt_id &&
              unchanged_progress.components == refused_progress.components,
          "an insufficient allowance wake cannot create an attempt or reset acknowledged bytes");
    SelectedHoldGate retry_gate;
    store.set_read_gate_for_test([](void* p) { static_cast<SelectedHoldGate*>(p)->pause(); }, &retry_gate);
    store.publish_read_allowance(uint64_t(64) << 20);
    retry_gate.await();
    const auto retry_progress = store.request_progress(935, 1);
    check(retry_progress.attempt_id != refused_progress.attempt_id && !retry_progress.totals_known &&
              !retry_progress.acknowledged_after(refused_progress),
          "a real allowance retry resets its fixed component counters under a new attempt identity");
    check(retry_progress.materialize_started && retry_progress.prior_materialize_unmeasured,
          "allowance recovery preserves unknown failed work across the attempt reset");
    for (const auto& component : retry_progress.components)
        check(component.total == 0 && component.acknowledged == 0, "a retried decode starts with no credited bytes");
    retry_gate.resume();
    check(await_status(store, 935, 1, ConversationStoreReadStatus::ready),
          "sufficient capacity rearms the same offer before demand");
    store.set_read_gate_for_test(nullptr, nullptr);
    check(!retry_gate.timed_out, "the allowance retry's actual allocation frame was explicitly released");
    check(store.demand(query_of(image, 935, 1), why), "demand adopts the recovered expanded-peak offer");
    ConversationStoreRead restored;
    check(store.wait_ready(935, 1, restored, 5000) && restored.image &&
              images_equal(*restored.image, image, why),
          "expanded-peak recovery preserves every restored byte");
    check(store.request_progress(935, 1).prior_materialize_unmeasured,
          "ready delivery retains the failed attempt's unknown aggregate contribution");
    restored.image.reset();
    store.release_read(935, 1);
    store.close();
    const auto finished = trace.records();
    check_eq(read_trace_count(finished, "reissue", 935), 1,
             "only sufficient capacity creates one replacement attempt");
    check_eq(read_trace_count(finished, "read_start", 935), 2,
             "the original and recovered materializations each execute once");
}

void check_budget_deferred_terminals() {
    for (int terminal = 0; terminal < 5; ++terminal) {
        StoreReadTraceCapture trace;
        TempRoot root;
        auto options = options_for(root.path, uint64_t(1) << 30, 0);
        options.max_read_requests = 1;
        ConversationStore store(options);
        const SavedConversation image = fixture(623, 128);
        ConversationStorePublishResult published;
        check(publish_and_wait(store, image, 1, 1, published), "the terminal deferred-offer fixture is stored");
        std::string why;
        store.publish_read_allowance(1);
        check(!store.prefetch(query_of(image, 940, 1), why),
              "an initial impossible-peak refusal is not an accepted deferred offer");
        store.publish_read_allowance(0);
        auto query = query_of(image, 941, 1);
        if (terminal == 4) query.selector.scope = ConversationScope{bytes_from("no-stored-head")};
        if (terminal == 2)
            query.deadline_ns = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count()) + uint64_t(1000000000);
        check(store.prefetch(query, why), "a terminal-case descriptor is accepted");
        check(await_status(store, 941, 1, terminal == 4 ? ConversationStoreReadStatus::missing :
                           ConversationStoreReadStatus::refused_budget), "the terminal-case descriptor settles");
        if (terminal == 0) store.withdraw_prefetch(941, 1);
        if (terminal == 1) store.cancel_read(941, 1);
        if (terminal == 2)
            std::this_thread::sleep_until(std::chrono::steady_clock::time_point(
                std::chrono::nanoseconds(query.deadline_ns)));
        if (terminal == 3) check(store.invalidate_domain("deferred test"), "domain retirement ends deferred authority");
        const uint64_t io_before = store.counters().bytes_read;
        store.publish_read_allowance(uint64_t(64) << 20);
        store.release_read(941, 1);
        check_eq(store.counters().bytes_read, io_before, "terminal and missing-head work never performs recovery I/O");
        store.close();
        const auto records = trace.records();
        check(!read_trace(records, "reissue", 940) && !read_trace(records, "reissue", 941) &&
                  !read_trace(records, "read_start", 941),
              "initial refusal, withdrawal, cancellation, expiry, stale epoch and missing heads never re-arm");
    }
}

void check_revoked_headless_demand() {
    // A store-level miss is not a production accepted offer. It nevertheless owns a bounded descriptor and must
    // reselect before pricing its manifest if an unrelated demand internally revokes it.
    for (bool publish_head : {false, true}) {
        TempRoot root;
        auto options = options_for(root.path, uint64_t(1) << 30, 0);
        options.max_read_requests = 2;
        ConversationStore store(options);
        const SavedConversation wanted = fixture(625, 256), neighbour = fixture(626, 128);
        std::string why;
        check(store.prefetch(query_of(wanted, 950, 1), why), "the headless store-level descriptor is accepted");
        check(store.request_status(950, 1) == ConversationStoreReadStatus::missing, "the headless descriptor misses");
        check(store.demand(query_of(neighbour, 951, 1), why), "an unrelated demand revokes the headless descriptor");
        store.release_read(951, 1);
        ConversationStorePublishResult published;
        if (publish_head)
            check(publish_and_wait(store, wanted, 1, 1, published), "the wanted head appears after revocation");
        check(store.demand(query_of(wanted, 950, 1), why),
              "the original demand reselects before charging manifest workspace");
        ConversationStoreRead read;
        check(store.wait_ready(950, 1, read, 5000) &&
                  read.status == (publish_head ? ConversationStoreReadStatus::ready : ConversationStoreReadStatus::missing),
              "the revoked headless descriptor restores a newly available head or settles another bounded miss");
        if (publish_head)
            check(read.image && images_equal(*read.image, wanted, why), "the headless retry restores every byte");
        read.image.reset();
        store.release_read(950, 1);
        check_eq(store.ram_lease_bytes(), 0, "the headless retry releases all leases");
        store.close();
    }
}

void check_budget_deferred_head_hold() {
    for (bool resident_recovery : {false, true}) {
        StoreReadTraceCapture trace;
        TempRoot root;
        auto options = options_for(root.path, uint64_t(1) << 30, 0);
        options.head_limit = 1;
        options.max_read_requests = resident_recovery ? 2 : 1;
        ConversationStore store(options);
        SavedConversation selected = fixture(627, 256);
        for (auto& checkpoint : selected.checkpoints) checkpoint.pinned = false;
        ConversationStorePublishResult head;
        check(publish_and_wait(store, selected, 1, 1, head), "the budget-deferred protected head is stored");
        store.suspend_read_admission();
        std::string why;
        check(store.prefetch(query_of(selected, 960, 1), why), "the protected deferred descriptor is accepted");
        check(await_status(store, 960, 1, ConversationStoreReadStatus::refused_budget),
              "the protected descriptor defers without an image reservation");
        const auto records = trace.records();
        const auto* admission = read_trace(records, "admission", 960);
        const uint64_t peak = admission ? admission->number("peak_bytes") : 0;
        check(peak > 0, "the recovery boundary uses the reader's actual charged peak");
        ConversationStorePublishResult pressure;
        check(publish_and_wait(store, fixture(628, 128), 2, 2, pressure) &&
                  pressure.status == ConversationStorePublishStatus::refused_limits,
              "head-limit pressure cannot evict an accepted budget-deferred selection");
        check_eq(store.ram_lease_bytes(), 0, "a deferred selected-head hold never retains an image lease");
        const auto query = query_of(selected, 960, 1);
        const auto matches = store.matches(query.selector, query.tokens, query.images);
        check(!matches.empty() && matches.front().head_id == head.head_id,
              "publication revocation preserves the deferred selection's head protection");
        const uint64_t tight_allowance = store.ram_held_bytes() + peak + 1;
        ConversationStoreRead occupied;
        if (resident_recovery) {
            check(store.demand(query_of(selected, 961, 1), why), "another demand acquires resident capacity first");
            store.publish_read_allowance(uint64_t(64) << 20);
            check(store.wait_ready(961, 1, occupied, 5000) && occupied.image,
                  "the competing demand owns a real materialized image");
            store.publish_read_allowance(tight_allowance);
            check(store.request_status(960, 1) == ConversationStoreReadStatus::refused_budget,
                  "unchanged tight allowance cannot bypass an occupied slot");
        }
        SelectedHoldGate gate;
        store.set_read_gate_for_test([](void* p) { static_cast<SelectedHoldGate*>(p)->pause(); }, &gate);
        if (resident_recovery) {
            occupied.image.reset();
            store.release_read(961, 1);
        }
        store.publish_read_allowance(tight_allowance);
        gate.await();
        check(store.ram_held_bytes() <= store.ram_read_budget(),
              "recovered reservation fits with one manifest workspace, not two");
        // The gate proves initial admission. Full decoded restore scaffolding is separately admitted afterwards.
        store.publish_read_allowance(uint64_t(64) << 20);
        gate.resume();
        check(await_status(store, 960, 1, ConversationStoreReadStatus::ready),
              "a revoked deferral admits at its original peak and completes in the background before demand");
        store.set_read_gate_for_test(nullptr, nullptr);
        check(!gate.timed_out, "tight-capacity recovery reaches the real allocation gate");
        ConversationStoreRead read;
        check(store.take_ready(960, 1, read) && read.image && read.head_id == head.head_id &&
                  images_equal(*read.image, selected, why), "the protected deferred head remains fully restorable");
        read.image.reset();
        store.release_read(960, 1);
        check_eq(store.ram_lease_bytes(), 0, "the protected deferred recovery releases its image");
        store.close();
    }
}

void check_selected_hold_revocation() {
    for (bool active : {false, true}) {
        TempRoot root;
        auto options = options_for(root.path, uint64_t(1) << 30, 0);
        options.head_limit = 3;
        options.max_read_requests = 2;
        ConversationStore store(options);
        const SavedConversation older = fixture(601, 128);
        SavedConversation latest = appended_turn(older, 256, 601);
        SavedConversation neighbour = fixture(602, 128);
        for (auto& checkpoint : latest.checkpoints) checkpoint.pinned = false;
        for (auto& checkpoint : neighbour.checkpoints) checkpoint.pinned = false;
        ConversationStorePublishResult old_head, latest_head, neighbour_head;
        check(publish_and_wait(store, older, 1, 1, old_head), "the original older failed-history cut is retained");
        check(publish_and_wait(store, latest, 2, 2, latest_head), "the selected newest history is stored");
        check(publish_and_wait(store, neighbour, 3, 3, neighbour_head), "the earlier demand's separate history is stored");
        SelectedHoldGate gate;
        if (active)
            store.set_read_gate_for_test([](void* context) { static_cast<SelectedHoldGate*>(context)->pause(); }, &gate);
        std::string why;
        const auto future = query_of(latest, 801, 8);
        check(store.prefetch(future, why), "the queued future request owns a selected newest cut");
        if (active) gate.await();
        else check(await_status(store, 801, 8, ConversationStoreReadStatus::ready), "the speculative image is ready");
        store.suspend_read_admission();
        check(store.demand(query_of(neighbour, 802, 8), why), "a different demand revokes only speculative RAM");
        gate.resume();
        if (active) {
            // The cancelled active body must unwind before its RAM can be refunded.
            check(await_status(store, 801, 8, ConversationStoreReadStatus::cancelled),
                  "the revoked active body unwinds without losing request ownership");
            store.set_read_gate_for_test(nullptr, nullptr);
        }
        store.release_read(802, 8);
        check_eq(store.ram_lease_bytes(), 0, "revocation releases materialized bytes but not selected metadata");
        // At a full head ceiling, pressure must consume another history, not the request's selected head.
        for (uint64_t id = 4; id <= 5; ++id) {
            ConversationStorePublishResult pressure;
            auto image = fixture(602 + id, 128);
            for (auto& checkpoint : image.checkpoints) checkpoint.pinned = false;
            check(publish_and_wait(store, image, id, id, pressure) &&
                      pressure.status == ConversationStorePublishStatus::stored,
                  "useful independent pressure publications still obtain head slots");
        }
        ConversationStoreManifestInfo info;
        check(store.manifest_info(latest_head.head_id, info), "RAM revocation preserved the selected newest head");
        check(store.manifest_info(old_head.head_id, info), "the original older failed-history fallback is still present");
        // A covered equal-cut replacement can retire the catalog head, but must not erase the queued selection.
        ConversationStorePublish cover;
        cover.request_id = 6;
        cover.generation = 6;
        cover.supersedes = latest_head.head_id;
        cover.cover = true;
        check(store.publish(std::make_shared<SavedConversation>(latest), cover, why) == kAccepted,
              "a selected-head hold does not refuse useful covered publication");
        ConversationStorePublishResult replacement;
        check(store.wait_publish(replacement, 5000) && replacement.status == ConversationStorePublishStatus::replaced,
              "the equal-cut cover durably replaces the selected catalog head");
        check(store.demand(future, why), "the original descriptor becomes demand at the descriptor ceiling");
        check(store.demand(query_of(neighbour, 803, 8), why), "the last available read descriptor is admitted");
        check(!store.demand(query_of(neighbour, 804, 8), why), "retained selections cannot exceed max_read_requests");
        store.release_read(803, 8);
        store.publish_read_allowance(uint64_t(64) << 20);
        ConversationStoreRead read;
        check(store.wait_ready(801, 8, read, 5000) && read.image && read.tokens == 256 &&
                  read.head_id == replacement.head_id && images_equal(*read.image, latest, why),
              "the revoked selection rebinds the equal-cut new head, never the older failed history");
        read.image.reset();
        store.release_read(801, 8);
        check_eq(store.ram_lease_bytes(), 0, "all revoked/rebound image leases end exactly once");
        check(store.ram_held_bytes() <= store.ram_read_budget(), "metadata and byte holds stay inside the allowance");
        check(store.disk_charged_bytes() <= store.disk_quota_bytes(), "retired selected files stay quota-accounted");
        store.close();
        check(!gate.timed_out, "the active revocation gate was explicitly released");
    }
}

void check_selected_hold_commit_interleaving() {
    // A demand racing victim selection must either win before staging, or retain immutable authority after staging.
    // Also cover a selected predecessor absent from the catalog and a second retirement of shared object bytes.
    for (int scenario = 0; scenario < 4; ++scenario) {
        StoreReadTraceCapture trace;
        const bool after_staging = scenario != 0;
        const bool cover = scenario >= 2;
        const bool rollback = scenario == 3;
        TempRoot root;
        auto options = options_for(root.path, uint64_t(1) << 30, 0);
        options.head_limit = 1;
        ConversationStore store(options);
        SavedConversation image = fixture(611, 256);
        for (auto& checkpoint : image.checkpoints) checkpoint.pinned = false;
        ConversationStorePublishResult first;
        check(publish_and_wait(store, image, 1, 1, first), "the retirement race starts from a stored head");
        SelectedHoldGate gate;
        struct Context { SelectedHoldGate* gate; bool after; } context{&gate, after_staging};
        store.set_catalog_gate_for_test([](void* opaque, bool staged) {
            auto& state = *static_cast<Context*>(opaque);
            if (staged == state.after) state.gate->pause();
        }, &context);
        ConversationStorePublish request;
        request.request_id = 2;
        request.generation = 2;
        if (cover) {
            request.cover = true;
            request.supersedes = first.head_id;
        }
        std::string why;
        // Same payload for an independent head too: this exercises shared-object ownership after two retirements.
        check(store.publish(std::make_shared<SavedConversation>(image), request, why) == kAccepted,
              "the racing publication is admitted before any selected hold exists");
        gate.await();
        store.suspend_read_admission();
        check(store.demand(query_of(image, 811, 1), why), "the demand selects while catalog I/O is gated");
        if (rollback) store.set_publish_fault_for_test("catalog-rename");
        gate.resume();
        ConversationStorePublishResult result;
        check(store.wait_publish(result, 5000), "the racing catalog transaction settles");
        store.set_catalog_gate_for_test(nullptr, nullptr);
        const auto records = trace.records("store_head ");
        const auto* retired = read_trace(records, "retired", request.request_id,
                                          cover ? "covered_replacement" : "head_limit");
        if (after_staging && !rollback) {
            check(retired && retired->number("head") == first.head_id && retired->number("head_leases") == 1 &&
                      retired->number("catalog_version") == result.catalog_version,
                  "a staged retirement receipt reports the lease actually present at committed removal");
        } else {
            check(!retired, "a winning pre-stage lease or failed rename emits no committed removal");
        }
        if (!after_staging || rollback) {
            ConversationStoreManifestInfo retained;
            check(store.manifest_info(first.head_id, retained), "a pre-stage lease or failed rename preserves authority");
        } else {
            check(result.status == (cover ? ConversationStorePublishStatus::replaced : ConversationStorePublishStatus::stored),
                  "a post-stage lease does not block the already staged durable publication");
            ConversationStorePublishResult pressure;
            check(publish_and_wait(store, fixture(612, 128), 3, 3, pressure) &&
                      pressure.status == ConversationStorePublishStatus::stored,
                  "a second retirement cannot reclaim the first selected head's shared objects");
        }
        store.publish_read_allowance(uint64_t(64) << 20);
        ConversationStoreRead read;
        check(store.wait_ready(811, 1, read, 5000) && read.image && read.head_id == first.head_id &&
                  read.tokens == 256 && images_equal(*read.image, image, why),
              "a staged/applying/rolled-back retirement cannot stale the captured selected manifest");
        read.image.reset();
        store.release_read(811, 1);
        check_eq(store.ram_lease_bytes(), 0, "the interleaved request releases its sole image charge");
        check(store.disk_charged_bytes() <= store.disk_quota_bytes(), "retired shared objects remain physically charged");
        store.close();
        check(!gate.timed_out, "the catalog gate was explicitly released without holding the store mutex");
    }
}

void check_selected_hold_terminal_release() {
    // A revoked request may pin metadata, never unlimited descriptors, images or an immortal eviction exemption.
    for (int terminal = 0; terminal < 4; ++terminal) {
        TempRoot root;
        auto options = options_for(root.path, uint64_t(1) << 30, 0);
        options.head_limit = 1;
        options.max_read_requests = 2;
        ConversationStore store(options);
        SavedConversation image = fixture(621, 128);
        for (auto& checkpoint : image.checkpoints) checkpoint.pinned = false;
        ConversationStorePublishResult first;
        check(publish_and_wait(store, image, 1, 1, first), "the terminal hold fixture is stored");
        const uint64_t idle_bytes = store.ram_held_bytes();
        std::string why;
        auto query = query_of(image, 821, 1);
        SelectedHoldGate gate;
        if (terminal == 2) {
            query.deadline_ns = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count()) + uint64_t(1000000000);
            store.set_read_gate_for_test([](void* opaque) { static_cast<SelectedHoldGate*>(opaque)->pause(); }, &gate);
        }
        check(store.prefetch(query, why), "the terminal test admits a selected speculation");
        if (terminal == 2) gate.await();
        else check(await_status(store, 821, 1, ConversationStoreReadStatus::ready), "the terminal speculation is ready");
        check(store.demand(query_of(image, 822, 1), why), "an earlier demand revokes the terminal speculation");
        store.release_read(822, 1);
        if (terminal == 0) store.cancel_read(821, 1);
        if (terminal == 1) store.withdraw_prefetch(821, 1);
        if (terminal == 2) {
            // The gate, not a guessed reader sleep, owns the active lifetime. Wait only for the original deadline.
            std::this_thread::sleep_until(std::chrono::steady_clock::time_point(std::chrono::nanoseconds(query.deadline_ns)));
            gate.resume();
            check(await_status(store, 821, 1, ConversationStoreReadStatus::cancelled), "the expired worker settles");
            store.set_read_gate_for_test(nullptr, nullptr);
        }
        if (terminal == 3) {
            store.close();
            check_eq(store.ram_lease_bytes(), 0, "close drops revoked speculative bytes and metadata holds");
            check(store.ram_held_bytes() <= idle_bytes, "close releases the request-owned query/metadata hold");
        } else {
            // Admission synchronously sweeps expiry; cancellation/withdrawal must already have released the hold.
            ConversationStorePublishResult pressure;
            check(publish_and_wait(store, fixture(622, 128), 2, 2, pressure) &&
                      pressure.status == ConversationStorePublishStatus::stored,
                  "cancelled, withdrawn or expired selected metadata no longer blocks the head ceiling");
            check(!store.demand(query, why) || terminal == 1,
                  "caller cancellation and expiry cannot resurrect the retired request");
            store.release_read(821, 1);
            check_eq(store.ram_lease_bytes(), 0, "terminal release leaves no materialized image charge");
            store.close();
        }
        check(!gate.timed_out, "terminal cleanup did not depend on a gate timeout");
    }
}

void check_speculation_follows_publication() {
    TempRoot root;
    ConversationStore store(options_for(root.path, uint64_t(1) << 30, 0));
    const SavedConversation final_image = fixture(651, 512);
    // Direct store-API coverage only: production's offer channel declines missing heads before prefetch.
    // A direct speculative query can still settle as a no-I/O miss retaining its descriptor; explicit
    // catalog refresh can reselect it, but allowance recovery must not treat it as a budget-deferred offer.
    std::string why;
    check(store.prefetch(query_of(final_image, 901, 4), why), "the too-early prefetch is accepted");
    check(store.request_status(901, 4) == ConversationStoreReadStatus::missing,
          "the too-early prefetch misses without materializing");
    check(!store.refresh_speculation(901, 4), "revalidating before any head exists changes nothing");
    check(store.request_status(901, 4) == ConversationStoreReadStatus::missing,
          "the miss keeps its descriptor for the publication that will name it");

    const SavedConversation first = fixture(651, 256);
    ConversationStorePublishResult first_result;
    check(publish_and_wait(store, first, 1, 1, first_result) &&
              first_result.status == ConversationStorePublishStatus::stored,
          "a first, shorter cut of the conversation is stored");
    check(store.refresh_speculation(901, 4), "the publication revalidates the outstanding speculation");
    check(await_status(store, 901, 4, ConversationStoreReadStatus::ready), "the refreshed prefetch materializes");

    ConversationStorePublishResult final_result;
    check(publish_and_wait(store, final_image, 2, 2, final_result) &&
              final_result.status == ConversationStorePublishStatus::stored,
          "a strictly longer cut of the same conversation is stored");
    check(store.refresh_speculation(901, 4), "the speculation follows the longer cut");
    check(await_status(store, 901, 4, ConversationStoreReadStatus::ready), "the second refresh materializes");
    check(!store.refresh_speculation(901, 4), "revalidating an up-to-date ready speculation is refused");
    check(store.request_status(901, 4) == ConversationStoreReadStatus::ready,
          "the ready image survives a redundant revalidation attempt");
    check(!store.refresh_speculation(901, 5), "an unknown generation is refused");
    ConversationStoreRead read;
    check(store.wait_ready(901, 4, read, 5000) && read.image && read.tokens == 512,
          "the refreshed speculation mounts the longest cut its query can use");
    if (read.image) {
        check(images_equal(*read.image, final_image, why), "the revalidated speculation restores byte for byte");
        store.release_read(901, 4);
    }
    check(store.demand(query_of(first, 902, 4), why), "the demand path is admitted");
    check(await_status(store, 902, 4, ConversationStoreReadStatus::ready), "the demand materializes");
    check(!store.refresh_speculation(902, 4), "a demand owner is never revalidated as speculation");
    store.release_read(902, 4);
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

void check_demand_reclaim_quote() {
    StoreReadTraceCapture trace;
    TempRoot root;
    const SavedConversation image = fixture(153, 4096);
    ConversationStore store(options_for(root.path, uint64_t(1) << 30, 0));
    ConversationStorePublishResult published;
    check(publish_and_wait(store, image, 1, 1, published) &&
              published.status == ConversationStorePublishStatus::stored, "the reclaim-quote image is durable");
    std::string why;
    const auto query = query_of(image, 2, 2);
    const uint64_t before = store.ram_held_bytes();
    store.publish_read_allowance(1);
    ConversationStoreReadRefusal refusal{UINT64_MAX};
    check(!store.demand(query, why, &refusal), "fresh demand returns a recoverable peak refusal");
    check_eq(store.ram_held_bytes(), before, "a refused fresh demand acquires no descriptor or image charge");
    const uint64_t enough = before + refusal.additional_ram_bytes;
    store.publish_read_allowance(enough);
    check(store.demand(query, why, &refusal) && refusal.additional_ram_bytes == 0,
          "accepted demand clears the advisory quote instead of inviting a second image reservation");
    ConversationStoreRead read;
    check(store.wait_ready(2, 2, read, 5000) && read.status == ConversationStoreReadStatus::ready &&
              read.image && images_equal(*read.image, image, why),
          "the quoted capacity covers descriptor, decode and restore scaffolding without another allowance bump");
    check(store.ram_held_bytes() <= store.ram_read_budget(), "quoted recovery never exceeds the published envelope");
    read.image.reset();
    store.release_read(2, 2);

    store.publish_read_allowance(uint64_t(64) << 20);
    const auto prefetched = query_of(image, 3, 3);
    check(store.prefetch(prefetched, why) && await_status(store, 3, 3, ConversationStoreReadStatus::ready),
          "a pre-existing ready image can be adopted independently of fresh-demand reclamation");
    const uint64_t ready_charge = store.ram_held_bytes();
    refusal.additional_ram_bytes = UINT64_MAX;
    check(store.demand(prefetched, why, &refusal) && refusal.additional_ram_bytes == 0 &&
              store.ram_held_bytes() == ready_charge,
          "adopting a ready image neither quotes nor charges a second materialization");
    check(store.wait_ready(3, 3, read, 5000) && read.image && images_equal(*read.image, image, why),
          "the adopted image still belongs to its exact request");
    refusal.additional_ram_bytes = UINT64_MAX;
    check(!store.demand(prefetched, why, &refusal) && refusal.additional_ram_bytes == 0,
          "an already-handed refusal cannot reuse a stale RAM-reclamation quote");
    read.image.reset();
    store.release_read(3, 3);
    store.close();
}

void check_demand_reclaim_owner() {
    enum class Case { restored, view_allocation, cancel_before, expire_before,
                      cancel_after, expire_after, missing_objects };
    enum class Selection { private_demand, shared_ram, shared_private_disk };
    struct Candidate {
        bool disk = false;
        uint64_t head = 0;
        ConversationCache::Match parked{};
    };
    for (const auto selection : {Selection::private_demand, Selection::shared_ram, Selection::shared_private_disk})
    for (const auto scenario : {Case::restored, Case::view_allocation, Case::cancel_before,
                               Case::expire_before, Case::cancel_after, Case::expire_after,
                               Case::missing_objects}) {
        TempRoot root;
        ConversationStore store(options_for(root.path, uint64_t(1) << 30, 0));
        SavedConversation disk = fixture(155, 4096);
        SavedConversation fallback = fixture(155, 32), victim = fixture(159, 16), newer = fixture(161, 32);
        const bool keep_newer = scenario == Case::missing_objects;
        newer.checkpoints.clear();
        const SavedConversation private_disk = fixture(155, 256);
        const bool shared = selection != Selection::private_demand;
        if (shared) disk.scope.value[0] ^= 0x80;
        fallback.checkpoints.clear();
        victim.checkpoints.clear();
        ConversationStorePublishResult published;
        if (!publish_and_wait(store, disk, 1, 1, published) ||
            published.status != ConversationStorePublishStatus::stored) {
            check(false, "the owner-reclamation image is durable");
            continue;
        }
        uint64_t private_head = 0;
        if (selection == Selection::shared_private_disk) {
            ConversationStorePublishResult private_published;
            check(publish_and_wait(store, private_disk, 11, 11, private_published) &&
                  private_published.status == ConversationStorePublishStatus::stored,
                  "a separate private disk fallback is durably available behind the shared candidate");
            private_head = private_published.head_id;
        }
        std::string why;
        auto query = query_of(disk, 2, 2);
        uint64_t now = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
        query.deadline_ns = now + uint64_t(60) * 1000000000;
        const uint64_t baseline = store.ram_held_bytes();
        store.publish_read_allowance(baseline + 1);
        ConversationStoreReadRefusal quote;
        if (store.demand(query, why, &quote) || !quote.additional_ram_bytes) {
            check(false, "the owner fixture starts at a fresh peak refusal without a descriptor");
            continue;
        }
        victim.live.gdn.resize(size_t(quote.additional_ram_bytes + baseline + (1u << 20)), 9);
        const size_t lease = 2048, preparation = 8192, outside = 4096;
        const size_t envelope = victim.bytes() + fallback.bytes() + (keep_newer ? newer.bytes() : 0) +
                                lease + preparation + outside + baseline + 1;
        const size_t entries = keep_newer ? 3 : 2;
        ConversationCache cache(envelope, entries);
        check(cache.put(std::move(victim)) && cache.put(SavedConversation(fallback)) &&
                  (!keep_newer || cache.put(SavedConversation(newer))) && cache.try_reserve_lease(lease),
              "the owner holds an older victim, private fallback, optional following owner and separate lease");
        const size_t before = cache.envelope_bytes();
        strata::program::ConversationStoreDemandFallback<Candidate> backups;
        backups.ram.parked = cache.best(query.tokens, {}, fallback.cvec, fallback.scope);
        check(backups.ram.parked.index == 1 && backups.ram.parked.tokens == 32,
              "the authoritative fallback is selected by the private prompt and scope after an older victim");
        if (private_head) backups.private_disk = Candidate{true, private_head, {}};
        bool cancelled = scenario == Case::cancel_before;
        if (scenario == Case::expire_before) now = query.deadline_ns;
        // This is the optional published coverage, not a stand-in for the store.
        // The production function calls the real store and real cache below.
        auto coverage = std::make_shared<size_t>(cache.size());
        bool dirty = false;
        const auto result = strata::program::admit_store_demand(
            store, cache, query, why, {envelope, preparation, outside}, &backups.ram.parked.index,
            [&] { return now; }, [&] { return cancelled; },
            [&] {
                dirty = true;
                if (scenario == Case::view_allocation) throw std::bad_alloc();
                coverage = std::make_shared<size_t>(cache.size());
                dirty = false;
                if (scenario == Case::cancel_after) cancelled = true;
                if (scenario == Case::expire_after) now = query.deadline_ns;
                if (scenario == Case::missing_objects) remove_tree(root.path + "/objects");
            },
            [&]() noexcept { coverage.reset(); dirty = true; });
        const bool early = scenario == Case::cancel_before || scenario == Case::expire_before;
        const bool late = scenario == Case::cancel_after || scenario == Case::expire_after;
        bool disk_missed = early || late;
        if (early) {
            check(!result.accepted && !result.recovery_attempted && !result.retried &&
                      cache.envelope_bytes() == before && cache.size() == entries && backups.ram.parked.index == 1,
                  "cancelled or expired work mutates no cache ownership before admission");
        } else {
            check(result.recovery_attempted && cache.size() == entries - 1 && backups.ram.parked.index == 0 &&
                      cache.evictions() == 1 && result.released_bytes == before - cache.envelope_bytes(),
                  "the actual owner reclaims once and rebases its original fallback");
            check(store.ram_read_budget() == envelope - preparation - outside - cache.envelope_bytes(),
                  "the actual owner republishes its exact remaining allowance");
            if (late) {
                check(!result.accepted && !result.retried && store.ram_held_bytes() == baseline,
                      "cancellation or expiry after reclamation starts no disk read and leaves no descriptor");
            } else {
                check(result.accepted && result.retried, "a fresh peak refusal retries after actual ownership release");
                ConversationStoreRead read;
                check(store.wait_ready(2, 2, read, 5000), "the reclaimed demand settles through the real reader");
                if (scenario == Case::missing_objects) {
                    disk_missed = true;
                    check(read.status == ConversationStoreReadStatus::corrupt && !read.image,
                          "a real missing-object failure after reclamation delivers no disk image");
                } else {
                    check(read.status == ConversationStoreReadStatus::ready && read.image &&
                              images_equal(*read.image, disk, why),
                          "the owner sequence restores the authorized disk image byte for byte");
                }
                check(store.ram_held_bytes() <= store.ram_read_budget(),
                      "the owner sequence preserves the live store envelope through delivery");
                read.image.reset();
                store.release_read(2, 2);
            }
        }
        check(cache.leased_bytes() == lease, "reclamation never refunds a separate live lease");
        if (scenario == Case::view_allocation)
            check(!coverage && dirty, "failed optional view allocation removes stale coverage and leaves it dirty");
        Candidate selected = backups.after_failure(shared);
        if (selection == Selection::shared_private_disk) {
            check(selected.disk && selected.head == private_head,
                  "shared failure preserves a separate private disk choice, never a duplicate RAM index");
            if (disk_missed) {
                auto private_query = query_of(private_disk, 3, 3);
                private_query.deadline_ns = query.deadline_ns;
                const auto private_admission = strata::program::admit_store_demand(
                    store, cache, private_query, why, {envelope, preparation, outside}, &backups.ram.parked.index,
                    [&] { return now; }, [&] { return cancelled; }, [] {}, []() noexcept {});
                if (private_admission.accepted) {
                    ConversationStoreRead read;
                    check(store.wait_ready(3, 3, read, 5000) &&
                          read.status == ConversationStoreReadStatus::corrupt && !read.image,
                          "the separate private disk attempt also settles its real missing-object failure");
                    read.image.reset();
                    store.release_read(3, 3);
                } else {
                    check(early || late, "only real cancellation/expiry refuses the private fallback before its read");
                }
            }
            selected = backups.after_failure(false);
        }
        check(!selected.disk && selected.parked.index == backups.ram.parked.index,
              "failure selects exactly the RAM object whose index admission protected and rebased");
        if (keep_newer) {
            const auto newer_query = query_of(newer, 5, 5);
            check(cache.best(newer_query.tokens, {}, newer.cvec, newer.scope).index == 1,
                  "the obsolete pre-compaction index is in range but names a different private scope");
        }
        try {
            bool retained = true;
            auto restored = cache.take(selected.parked.index, retained);
            check(!retained && restored.scope.matches(fallback.scope) &&
                  (!shared || !restored.scope.matches(disk.scope)) && images_equal(restored, fallback, why),
                  "after failure/refusal compaction returns the authorized private fallback scope and exact content");
        } catch (const std::out_of_range&) {
            check(false, "a shared/private demand chain must not retain an out_of_range pre-compaction RAM index");
        }
        cache.release_lease(lease);
        store.close();
    }
}

void check_zero_allowance_demand_adoption() {
    for (bool revoked : {false, true}) for (bool zero_allowance : {true, false}) {
        StoreReadTraceCapture trace;
        TempRoot root;
        auto options = options_for(root.path, uint64_t(1) << 30, 0);
        options.max_read_requests = 2;
        ConversationStore store(options);
        const SavedConversation disk = fixture(179, 256), fallback = fixture(179, 32);
        ConversationStorePublishResult published;
        check(publish_and_wait(store, disk, 1, 1, published) &&
                  published.status == ConversationStorePublishStatus::stored,
              "the demand-adoption fixture is durable");
        const uint64_t baseline = store.ram_held_bytes();
        auto query = query_of(disk, 2, 2);
        const uint64_t now = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
        query.deadline_ns = now + uint64_t(60) * 1000000000;
        std::string why;
        if (!revoked) store.suspend_read_admission();
        check(store.prefetch(query, why), "the selected prefetch is accepted before demand");
        check(await_status(store, 2, 2, revoked ? ConversationStoreReadStatus::ready :
                           ConversationStoreReadStatus::refused_budget),
              "the prefetch reaches its asserted state before demand adoption");
        if (revoked) {
            // Retiring an already-ready offer completes revocation synchronously, with no worker left to refund.
            store.suspend_read_admission();
            check(store.demand(query_of(disk, 3, 2), why), "a suspended peer internally revokes the ready offer");
            store.release_read(3, 2);
            check(store.request_status(2, 2) == ConversationStoreReadStatus::cancelled,
                  "internal revocation has settled before the original demand returns");
        }
        const auto settled_status = store.request_status(2, 2);
        const uint64_t held = store.ram_held_bytes(), io = store.counters().bytes_read;
        check_eq(store.ram_lease_bytes(), 0, "the settled offer has no image lease to reclaim");
        if (zero_allowance) {
            // The real serving-owner seam publishes zero once. A false admission must return to the RAM fallback;
            // this test never waits for a refused adoption or supplies a later allowance to rescue it.
            const size_t lease = 512;
            ConversationCache cache(fallback.bytes() + lease, 1);
            check(cache.put(SavedConversation(fallback)) && cache.try_reserve_lease(lease),
                  "the owner retains its fallback and an unrelated live RAM lease");
            const size_t cache_charge = cache.envelope_bytes();
            size_t protected_index = 0;
            const auto result = strata::program::admit_store_demand(
                store, cache, query, why, {cache_charge, 0, 0}, &protected_index,
                [&] { return now; }, [] { return false; },
                [] { check(false, "refused adoption must not refresh a reclaimed cache view"); },
                []() noexcept { check(false, "refused adoption must not invalidate the fallback view"); });
            check(!result.accepted && !result.recovery_attempted && !result.retried &&
                      !result.required_increment_bytes && !why.empty(),
                  "a settled accepted offer refuses zero allowance synchronously without a fresh-demand reclaim");
            check_eq(store.ram_read_budget(), 0, "the sole owner published no materialization allowance");
            check(store.request_status(2, 2) == settled_status,
                  "refusal preserves the settled descriptor instead of requeueing it pending");
            check_eq(store.ram_held_bytes(), held, "refusal preserves the selected metadata without reserving an image");
            check_eq(store.counters().bytes_read, io, "refused adoption performs no new payload read");
            check(cache.size() == 1 && cache.evictions() == 0 && cache.envelope_bytes() == cache_charge &&
                      cache.leased_bytes() == lease && protected_index == 0,
                  "refused adoption evicts no history and refunds no unrelated RAM lease");
            bool retained = true;
            const auto restored = cache.take(protected_index, retained);
            check(!retained && images_equal(restored, fallback, why),
                  "the owner consumes its original byte-identical fallback without waiting on the store");
            cache.release_lease(lease);
            check_eq(cache.leased_bytes(), 0, "only the fallback owner releases its separate RAM lease");
        } else {
            ConversationStoreReadRefusal refusal{UINT64_MAX};
            check(store.demand(query, why, &refusal) && refusal.additional_ram_bytes == 0,
                  "positive-budget suspension still permits adoption of a settled selection");
            ConversationStoreRead pending;
            check(store.request_status(2, 2) == ConversationStoreReadStatus::none &&
                      !store.take_ready(2, 2, pending) && !store.ram_lease_bytes(),
                  "positive-budget adoption remains queued without allocating while admission is suspended");
            store.publish_read_allowance(uint64_t(64) << 20);
            check(store.wait_ready(2, 2, pending, 5000) && pending.image &&
                      pending.head_id == published.head_id && images_equal(*pending.image, disk, why),
                  "resuming positive-budget admission restores the retained selection byte for byte");
            pending.image.reset();
        }
        const uint64_t releases = store.counters().reads_released;
        store.release_read(2, 2);
        check(store.request_status(2, 2) == ConversationStoreReadStatus::none,
              "normal owner release forgets the retained descriptor");
        check_eq(store.ram_lease_bytes(), 0, "normal release leaves no incoming image lease");
        check_eq(store.ram_held_bytes(), baseline, "normal release balances every query, decode and image charge");
        check_eq(store.counters().reads_released, releases + 1, "the accepted descriptor is released exactly once");
        store.release_read(2, 2);
        check_eq(store.counters().reads_released, releases + 1, "repeated owner release cannot refund the descriptor twice");
        store.close();
        if (zero_allowance) {
            const auto records = trace.records();
            check_eq(read_trace_count(records, "reissue", 2), 0, "zero-allowance refusal never creates a pending attempt");
            check_eq(read_trace_count(records, "read_start", 2), revoked ? 1 : 0,
                     "only the original ready prefetch, when present, enters materialization");
            const auto* released = read_trace(records, "release", 2);
            check(released && !released->number("reserved_bytes") && !released->number("lease_bytes") &&
                      !released->number("holds_slot"),
                  "the refused adoption releases with no outstanding worker reservation or incoming slot");
        }
    }
}

void check_zero_allowance_adoption_controls() {
    for (bool active : {false, true}) {
        TempRoot root;
        SelectedHoldGate gate;
        ConversationStore store(options_for(root.path, uint64_t(1) << 30, 0));
        const SavedConversation image = fixture(181, 256);
        ConversationStorePublishResult published;
        check(publish_and_wait(store, image, 1, 1, published), "the zero-allowance control image is durable");
        const uint64_t baseline = store.ram_held_bytes();
        if (active)
            store.set_read_gate_for_test([](void* p) { static_cast<SelectedHoldGate*>(p)->pause(); }, &gate);
        const auto query = query_of(image, 2, 2);
        std::string why;
        check(store.prefetch(query, why), "the control prefetch is accepted");
        if (active) gate.await();
        else check(await_status(store, 2, 2, ConversationStoreReadStatus::ready), "the control image is already ready");
        const uint64_t held = store.ram_held_bytes(), leased = store.ram_lease_bytes();
        check(active ? held > baseline && !leased : leased > 0,
              "the control owns a real active reservation or a materialized image lease");
        store.publish_read_allowance(0);
        ConversationStoreReadRefusal refusal{UINT64_MAX};
        check(store.demand(query, why, &refusal) && refusal.additional_ram_bytes == 0,
              "zero allowance still permits adoption of ready images and genuinely active reads");
        check_eq(store.ram_held_bytes(), held, "control adoption never refunds an active reservation or ready image");
        check_eq(store.ram_lease_bytes(), leased, "control adoption preserves the existing image lease");
        if (active) {
            store.release_read(2, 2);
            check_eq(store.ram_held_bytes(), held, "even explicit release cannot refund a gated active worker");
            gate.resume();
        } else {
            ConversationStoreRead read;
            check(store.wait_ready(2, 2, read, 5000) && read.image && read.head_id == published.head_id &&
                      images_equal(*read.image, image, why),
                  "the pre-existing ready image remains deliverable without another allowance publication");
            read.image.reset();
            store.release_read(2, 2);
            check_eq(store.ram_held_bytes(), baseline, "ready-image release balances its pre-existing full charge");
        }
        store.set_read_gate_for_test(nullptr, nullptr);
        store.close();
        check_eq(store.ram_lease_bytes(), 0, "the control owner releases its image lease normally");
        check(store.ram_held_bytes() <= baseline, "worker unwind and normal release leave no read-owned charge");
        check(!gate.timed_out, "the active control is released by its explicit gate, never a timing-dependent sleep");
    }
}

void check_demand_reclaim_accepted_reader() {
    TempRoot root;
    ConversationStore store(options_for(root.path, uint64_t(1) << 30, 0));
    const SavedConversation disk = fixture(173, 4096);
    ConversationStorePublishResult published;
    check(publish_and_wait(store, disk, 1, 1, published), "the accepted-reader image is durable");
    const uint64_t baseline = store.ram_held_bytes();
    auto query = query_of(disk, 2, 2);
    const auto now = [] { return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count()); };
    query.deadline_ns = now() + uint64_t(60) * 1000000000;
    SelectedHoldGate gate;
    store.set_read_gate_for_test([](void* p) { static_cast<SelectedHoldGate*>(p)->pause(); }, &gate);
    std::string why;
    check(store.prefetch(query, why), "the speculative reader owns its reservation before revocation");
    gate.await();
    store.suspend_read_admission();
    check(store.demand(query_of(disk, 3, 2), why),
          "a distinct demand revokes the gated speculative reader");
    store.release_read(3, 2);
    const uint64_t held = store.ram_held_bytes();
    SavedConversation victim = fixture(175, 16), fallback = fixture(177, 32);
    victim.checkpoints.clear();
    fallback.checkpoints.clear();
    victim.live.gdn.resize(size_t(16) << 20, 17);
    const size_t envelope = victim.bytes() + fallback.bytes() + baseline + 1;
    ConversationCache cache(envelope, 2);
    check(cache.put(std::move(victim)) && cache.put(SavedConversation(fallback)),
          "the owner has reclaimable RAM beside its protected rollback");
    const size_t before = cache.envelope_bytes();
    size_t protected_index = 1;
    // The competing demand revoked this worker, but its original reservation
    // remains charged. A lower allowance must not turn its retry into a fresh demand.
    const auto result = strata::program::admit_store_demand(
        store, cache, query, why, {envelope, 0, 0}, &protected_index,
        now, [] { return false; }, [] {}, []() noexcept {});
    check(!result.accepted && !result.recovery_attempted && !result.retried,
          "an accepted revoked reader cannot be mistaken for a fresh reclamation demand");
    check(cache.size() == 2 && cache.evictions() == 0 &&
              cache.envelope_bytes() == before && protected_index == 1,
          "retrying an accepted reader does not evict another history or rebase rollback");
    check(store.ram_held_bytes() == held,
          "refusing the accepted retry preserves its active worker's existing charge");
    gate.resume();
    store.release_read(query.request_id, query.generation);
    store.close();
    check(!gate.timed_out, "the accepted revoked worker is explicitly released");
}

void check_demand_reclaim_interleavings() {
    enum class Case { active_reader, rearmed_speculation, larger_head };
    for (const auto scenario : {Case::active_reader, Case::rearmed_speculation, Case::larger_head}) {
        StoreReadTraceCapture trace;
        TempRoot root;
        SelectedHoldGate gate;
        auto options = options_for(root.path, uint64_t(1) << 30, 0);
        options.chunk_bytes = 1u << 20; // serving object size: a peak race, not a writer-workspace refusal
        options.max_reads = 2;
        ConversationStore store(options);
        const auto disk = fixture(165, 4096), peer = fixture(167, 8);
        ConversationStorePublishResult published;
        check(publish_and_wait(store, disk, 1, 1, published) &&
                  publish_and_wait(store, peer, 3, 1, published), "interleaved demand fixtures are durable");
        std::string why;
        auto query = query_of(disk, 2, 2);
        query.deadline_ns = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count()) + uint64_t(60) * 1000000000;
        ConversationStoreReadRefusal quote;
        const uint64_t idle_charge = store.ram_held_bytes();
        store.publish_read_allowance(idle_charge + 1);
        if (store.demand(query, why, &quote) || !quote.additional_ram_bytes) {
            check(false, "the interleaved fixture starts with a fresh peak refusal");
            continue;
        }
        if (scenario != Case::larger_head) {
            store.set_read_gate_for_test([](void* p) { static_cast<SelectedHoldGate*>(p)->pause(); }, &gate);
            store.publish_read_allowance(scenario == Case::active_reader ? uint64_t(64) << 20 : 0);
            check(store.prefetch(query_of(peer, 4, 2), why), "another request retains its own speculative ownership");
            if (scenario == Case::active_reader) gate.await();
            else check(await_status(store, 4, 2, ConversationStoreReadStatus::refused_budget),
                       "speculation waits for actual capacity before the owner reclaims");
        }
        const uint64_t held = store.ram_held_bytes();
        SavedConversation victim = fixture(169, 16), fallback = fixture(171, 32);
        victim.checkpoints.clear();
        fallback.checkpoints.clear();
        victim.live.gdn.resize(scenario == Case::larger_head ? size_t(24) << 20 :
                              size_t(quote.additional_ram_bytes + held + (1u << 20)), 11);
        const size_t envelope = victim.bytes() + fallback.bytes() + held + 1;
        ConversationCache cache(envelope, 2);
        check(cache.put(std::move(victim)) && cache.put(SavedConversation(fallback)),
              "interleaved owner has an evictable image and protected rollback");
        size_t protected_index = 1;
        bool reclaimed = false, interleaved = false;
        const auto result = strata::program::admit_store_demand(
            store, cache, query, why, {envelope, 0, 0}, &protected_index,
            [] { return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                     std::chrono::steady_clock::now().time_since_epoch()).count()); },
            [&] {
                // This callback gates real competing work at the post-publication
                // cancellation check, before the production function's sole retry.
                if (reclaimed && !interleaved) {
                    interleaved = true;
                    if (scenario == Case::rearmed_speculation) gate.await();
                    if (scenario == Case::larger_head) {
                        auto grown = disk;
                        grown.live.gdn.resize(size_t(21) << 20, 13);
                        ConversationStorePublishResult newer;
                        check(publish_and_wait(store, grown, 5, 3, newer) &&
                                  newer.status == ConversationStorePublishStatus::stored,
                              "a real larger equal-cut head commits between allowance publication and retry");
                    }
                }
                return false;
            },
            [&] {
                reclaimed = true;
                if (scenario == Case::active_reader)
                    check(store.ram_held_bytes() == held,
                          "suspension and cache reclamation do not refund an already active reader");
            }, []() noexcept {});
        check(result.recovery_attempted && result.retried && interleaved &&
                  cache.evictions() == 1 && protected_index == 0,
              "competing work cannot cause a second reclaim or invalidate the original fallback");
        gate.resume();
        if (scenario == Case::larger_head) {
            check(!result.accepted && store.counters().refused_budget >= 3,
                  "the sole retry revalidates a larger current head and refuses its new peak");
        } else {
            check(result.accepted, "the demand retains a descriptor beside independent speculative work");
            ConversationStoreRead read;
            check(store.wait_ready(2, 2, read, 5000) && read.image && images_equal(*read.image, disk, why),
                  "interleaved demand restores only its own authorized image");
            read.image.reset();
            store.release_read(2, 2);
            check(store.demand(query_of(peer, 4, 2), why) && store.wait_ready(4, 2, read, 5000) &&
                      read.image && images_equal(*read.image, peer, why),
                  "the speculative request retains its own body and completion");
            read.image.reset();
            store.release_read(4, 2);
            check(!gate.timed_out, "the actual worker interleaving ends by explicit release, not timeout");
            store.set_read_gate_for_test(nullptr, nullptr);
        }
        check(store.ram_held_bytes() <= store.ram_read_budget(),
              "competing reads and publication remain within the authoritative allowance");
        bool retained = true;
        const auto restored = cache.take(protected_index, retained);
        check(!retained && images_equal(restored, fallback, why), "a refused retry still has its original RAM rollback");
        store.close();
        const auto records = trace.records();
        check(read_trace_count(records, "admission", 2) <= 1,
              "the owner retry admits at most one descriptor for the demand identity");
    }
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
        missing.selector.scope = ConversationScope{bytes_from("private:never-log-this-scope")};
        check(store.demand(missing, why), "a timed miss is admitted without materialization");
        ConversationStoreRead miss;
        check(store.wait_ready(7001, 8, miss, 5000) && miss.status == ConversationStoreReadStatus::missing,
              "the timed miss reports its actual no-I/O result");
        check(miss.materialize_ms == 0, "a no-I/O miss cannot train materialization cost");
        store.release_read(7001, 8);
        store.publish_read_allowance(1);
        check(!store.demand(query_of(image, 7002, 8), why), "an impossible timed read is refused at admission");
        store.publish_read_allowance(uint64_t(64) << 20);
        store.release_read(7002, 8);
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
        check(std::isfinite(read.materialize_ms) && read.materialize_ms > 0,
              "successful full body timing remains available even when tracing is off");
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
            check(failed.materialize_ms == 0, "a failed body cannot contribute a successful timing");
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
                  trace.contents().find(sha256_hex(conversation_store_detail::sha256(
                      missing.selector.scope.value.data(), missing.selector.scope.value.size()))) ==
                      std::string::npos,
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
    check(!store.demand(query_of(image, 104, 1), why),
          "demand refuses a settled budget deferral while the published allowance is zero");
    check(store.request_status(104, 1) == ConversationStoreReadStatus::refused_budget,
          "zero-allowance refusal preserves the deferred selection for a later retry");
    ConversationStoreRead premature;
    SelectedHoldGate gate;
    store.set_read_gate_for_test([](void* p) { static_cast<SelectedHoldGate*>(p)->pause(); }, &gate);
    store.publish_read_allowance(uint64_t(64) << 20);
    check(store.demand(query_of(image, 104, 1), why),
          "the owner can retry the preserved selection after publishing positive allowance");
    gate.await();
    check(!store.take_ready(104, 1, premature), "an active adopted reader cannot deliver an unfinished descriptor");
    gate.resume();
    ConversationStoreRead budget_retry;
    check(store.wait_ready(104, 1, budget_retry, 5000) && budget_retry.image &&
              images_equal(*budget_retry.image, image, why), "a prior speculative budget refusal does not force cold prefill");
    store.set_read_gate_for_test(nullptr, nullptr);
    check(!store.take_ready(104, 1, premature), "an adopted deferred image is delivered exactly once before release");
    check(!gate.timed_out, "the adopted reader's pending interval was exercised deterministically");
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
    const auto* retry = read_trace(records, "reissue", 104, "allowance_recovery");
    check(budget && budget->text("status") == "refused_budget" && !budget->number("io_attempted") &&
              !budget->number("start_ns") && retry &&
              retry->number("previous_attempt_id") == budget->number("attempt_id"),
          "a speculative no-I/O refusal is retried only by the later positive-allowance publication seam");
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
        check(store.shutdown_state() == ConversationStoreShutdownState::quiescent_unclean,
              "legacy close truthfully reports a faulted owner as quiescent unclean");
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
    check(second.shutdown_state() == ConversationStoreShutdownState::quiescent_unclean,
          "legacy close never calls a refused non-owner durable clean");
    check(file_digest(root.path + "/catalog") == catalog_before, "the refused owner wrote no catalog");
    check(first.enabled() && first.healthy(), "the owner keeps serving");
    check(first.run_intent_active(), "the refused owner did not clear the run intent");
    check_eq(first.counters().heads, heads, "the refused owner retired nothing");
    first.close();
    check(first.shutdown_state() == ConversationStoreShutdownState::durable_clean && !first.run_intent_active(),
          "legacy close without an earlier deadline preserves healthy durable-clean behavior");
    first.close();
    check(first.finish_shutdown() == ConversationStoreShutdownState::durable_clean,
          "legacy close remains idempotent after proven success");
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
    store.suspend_read_admission(); // Exercise a queued descriptor, not a racing active worker's retirement.
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

// A retired head's final reader must wake the idle writer: no later publication or close may be needed to return
// its physical charge. Two handed-out images distinguish the last lease from an earlier release of the same head.
void check_idle_final_lease_reclaim() {
    for (bool fail_retire : {false, true}) {
        TempRoot root;
        auto options = options_for(root.path, uint64_t(1) << 30, 0);
        options.max_reads = 2;
        ConversationStore store(options);
        const SavedConversation image = fixture(193, 256);
        ConversationStorePublishResult published;
        check(publish_and_wait(store, image, 1, 1, published) &&
                  published.status == ConversationStorePublishStatus::stored,
              "the idle-reclaim branch is stored");
        std::string why;
        ConversationStoreRead first, last;
        check(store.demand(query_of(image, 41, 1), why) && store.wait_ready(41, 1, first, 5000) && first.image,
              "the first reader holds a handed-out image");
        check(store.demand(query_of(image, 42, 1), why) && store.wait_ready(42, 1, last, 5000) && last.image,
              "the final reader holds a second image of the same head");
        check(store.invalidate_domain("idle reclaim regression"), "the leased head is retired durably");
        check_eq(store.counters().heads, 0, "the retired head is no longer authorized");
        const auto held = store.counters();
        const uint64_t charged = store.disk_charged_bytes();
        const uint64_t available = store.disk_available_bytes();
        check(held.pending_delete_bytes > 0, "the leased retired manifest awaits confirmed deletion");
        first.image.reset();
        store.release_read(41, 1);
        ::usleep(50000);
        check_eq(store.disk_charged_bytes(), charged, "an earlier lease release cannot credit files still held");
        check_eq(store.counters().pending_delete_bytes, held.pending_delete_bytes,
                 "the manifest remains pending until the last reader releases it");
        if (fail_retire) store.set_publish_fault_for_test("retire-object-prepare");
        last.image.reset();
        store.release_read(42, 1);
        // Read-only observation only: deliberately no publication, promotion or close to trigger another pass.
        bool reclaimed = false;
        for (int waited = 0; waited < 5000; ++waited) {
            const auto remaining = store.counters();
            if (remaining.pending_delete_bytes == 0 && remaining.orphan_bytes == 0 &&
                store.disk_charged_bytes() < charged) {
                reclaimed = true;
                break;
            }
            ::usleep(1000);
        }
        check(reclaimed, "the idle writer reclaims the final lease, including a failed retire-queue retry");
        check_eq(store.counters().pending_delete_bytes, 0, "no pending-delete charge survives the idle release");
        check_eq(store.counters().orphan_bytes, 0, "no reclaimable charge survives the idle release");
        check(store.disk_available_bytes() > available, "idle deletion returns quota headroom without a publication");
        check(store.counters().reclaimed_bytes > held.reclaimed_bytes, "idle deletion is confirmed and counted");
        check(store.healthy() && store.run_intent_active(), "idle reclamation leaves the live store healthy");
        store.close();
        check(store.shutdown_state() == ConversationStoreShutdownState::durable_clean &&
                  !store.run_intent_active(),
              "the subsequent shutdown still closes durably and clears the intent");
    }
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
        StoreReadTraceCapture trace;
        const std::string path = root.path + "/evict";
        SavedConversation big = fixture(211, 4096);
        for (auto& checkpoint : big.checkpoints) checkpoint.pinned = false;
        const SavedConversation small_pinned = fixture(212, 256);
        check(!big.pinned(), "the evictable fixture carries no pin");
        check(small_pinned.pinned(), "the small fixture is pinned");
        uint64_t charged = 0;
        uint64_t victim_head = 0;
        {
            ConversationStore store(options_for(path, uint64_t(1) << 30, 0));
            ConversationStorePublishResult first, second;
            check(publish_and_wait(store, big, 1, 1, first) &&
                      first.status == ConversationStorePublishStatus::stored,
                  "the evictable branch is stored");
            victim_head = first.head_id;
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
        const auto records = trace.records("store_head ");
        const auto* retired = read_trace(records, "retired", request.request_id, "byte_quota");
        check(retired && retired->number("head") == victim_head && !retired->number("pinned") &&
                  !retired->number("head_leases") && !retired->number("protected") &&
                  retired->number("replacement_head") == 0 && retired->number("applied_ns") &&
                  retired->number("catalog_version") < stored.catalog_version,
              "physical byte eviction names the committed victim and requesting publication, not a head-limit event");
        check_eq(store.counters().evictions, 1, "one committed physical quota victim counts exactly once");
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
        StoreReadTraceCapture trace;
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
        const auto inventory_before = file_inventory(path);
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
        check(refused.released, "a failed catalog rename reports its publication image released");
        check_eq(store.ram_lease_bytes(), 0, "a failed catalog rename releases its publication lease");
        check(directory_names(path + "/tmp").empty(), "a failed catalog rename removes its catalog temporary");
        check(file_inventory(path) == inventory_before, "a failed catalog rename removes every abandoned file");
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
        const auto refused_records = trace.records("store_head ");
        const auto* planned = read_trace(refused_records, "planned", cover.request_id, "covered_replacement");
        check(planned && planned->number("head") == head.head_id && planned->number("pinned"),
              "a planned replacement records its predecessor and pin without claiming retirement");
        check_eq(read_trace_count(refused_records, "retired", cover.request_id), 0,
                 "a failed catalog rename must not report a committed head retirement");
        check_eq(read_trace_count(refused_records, "published_cut", cover.request_id), 0,
                 "a failed catalog rename reports no confirmed reusable cut");
        // The store is still a working store: the same publication succeeds once the fault is gone.
        ConversationStorePublishResult retry;
        auto retry_image = std::make_shared<SavedConversation>(head_image);
        check(store.publish(retry_image, cover, why) == kAccepted, "the retry is admitted");
        check(store.wait_publish(retry, 5000), "the retry settles");
        check(retry.status == ConversationStorePublishStatus::replaced, "the retry replaces the predecessor");
        check(!store.retains_head(head.head_id) && store.retains_head(retry.head_id),
              "successful replacement removes only predecessor authority and retains its successor");
        store.close();
        const auto committed_records = trace.records("store_head ");
        const auto* retired = read_trace(committed_records, "retired", cover.request_id, "covered_replacement");
        check_eq(read_trace_count(committed_records, "retired", cover.request_id), 1,
                 "only the successful retry emits a confirmed retirement");
        check(retired && retired->number("head") == head.head_id &&
                  retired->number("replacement_head") == retry.head_id &&
                  retired->number("applied_ns") >= retired->number("state_ns"),
              "the committed record binds the real replacement after its durable apply");
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
            const std::weak_ptr<SavedConversation> uncertain_alias = image2;
            ConversationStorePublish request;
            request.request_id = 2;
            request.generation = 2;
            std::string why;
            check(store.publish(image2, request, why) == kAccepted, "the publication that will be uncertain is admitted");
            image2.reset();
            ConversationStorePublishResult uncertain;
            check(store.wait_publish(uncertain, 5000), "the uncertain publication settles");
            check(uncertain.status == ConversationStorePublishStatus::failed,
                  "an unconfirmed catalog rename fails the publication");
            check(!store.healthy(), "an unconfirmed catalog rename latches a fault");
            check(uncertain.released && uncertain_alias.expired(),
                  "uncertain disk authority does not retain the publication image");
            check_eq(store.ram_lease_bytes(), 0, "an uncertain publication releases its RAM lease");
            check(directory_names(path + "/tmp").empty(), "the uncertain catalog really left its temporary name");
            check_eq(intent_names(path).size(), 1, "an uncertain publication retains a real run-intent marker");
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
            check_eq(intent_names(path).size(), 1, "a faulted close leaves its distrust marker on disk");
            // A process can also die with incomplete object/catalog scratch files. Recovery must use neither as
            // authority, and must retire them along with the real payloads retained by the uncertain transaction.
            for (const char* name : {"interrupted-object", "interrupted-catalog"}) {
                const std::string scratch = path + "/tmp/" + name;
                const int fd = ::open(scratch.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
                check(fd >= 0, "an interrupted-write scratch file is created");
                if (fd >= 0) {
                    const uint8_t partial[] = {1, 2, 3};
                    check(::write(fd, partial, sizeof partial) == ssize_t(sizeof partial),
                          "the scratch file holds a real incomplete payload");
                    check(::close(fd) == 0, "the interrupted-write scratch fixture closes");
                }
            }
            // The next start re-reads the authority instead of trusting the uncertain view.
            ConversationStore store(options_for(path, uint64_t(1) << 30, 0));
            check(store.enabled(), "the root reopens after an uncertain transaction");
            check_eq(store.epoch(), first_epoch + 1, "the armed intent makes the next start distrust this domain");
            const ConversationStoreQuery query = query_of(fixture(362, 2048), 5, 5);
            check(store.matches(query.selector, query.tokens, query.images).empty(),
                  "an uncertain run's records are never reused");
            const ConversationStoreQuery successor = query_of(fixture(363, 2048), 6, 6);
            check(store.matches(successor.selector, successor.tokens, successor.images).empty(),
                  "restart also distrusts the head in the unconfirmed replacement catalog");
            check_eq(store.counters().heads, 0, "restart retires every head of the uncertain domain");
            check_eq(store.counters().objects, 0, "restart keeps no object authority from the uncertain domain");
            check_eq(store.counters().manifests, 0, "restart keeps no manifest authority from the uncertain domain");
            check(file_inventory(path + "/objects").empty() && directory_names(path + "/manifests").empty() &&
                      directory_names(path + "/tmp").empty(),
                  "restart reclaims both catalogs' abandoned payloads and scratch files");
            check_eq(store.counters().pending_delete_bytes, 0, "uncertain-run recovery leaves no orphan deletion lease");
            check_eq(store.counters().recovered_orphans, 2, "restart accounts both interrupted scratch writes");
            store.close();
            check(!store.run_intent_active() && intent_names(path).empty(),
                  "the recovered store clears the real marker only after its healthy close");
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
        uint64_t head_id = 0;
        const SavedConversation image = fixture(352, 2048);
        {
            ConversationStore store(options_for(path, uint64_t(1) << 30, 0));
            ConversationStorePublishResult result;
            check(publish_and_wait(store, image, 1, 1, result) &&
                      result.status == ConversationStorePublishStatus::stored,
                  "the intent store stores its branch");
            epoch = store.epoch();
            head_id = result.head_id;
            const auto markers = intent_names(path);
            check_eq(markers.size(), 1, "the intent store has one real marker before close");
            const std::string marker_digest = markers.empty() ? "" : file_digest(path + "/" + markers[0]);
            const uint64_t charged = store.disk_charged_bytes();
            store.set_publish_fault_for_test("intent-unlink-sync");
            if (*extra) store.set_publish_fault_for_test(extra);
            store.close();
            check(!store.healthy(), "a close that cannot prove its marker gone is not a healthy close");
            check(store.run_intent_active(), "a failed marker removal leaves the run intent armed");
            check(intent_names(path) == markers, "a failed marker removal actually re-creates the same marker");
            check(!markers.empty() && file_digest(path + "/" + markers[0]) == marker_digest,
                  "the re-armed marker preserves the original distrust payload");
            check_eq(store.disk_charged_bytes(), charged, "a failed marker clear charges its re-arm exactly once");
            check_eq(store.ram_lease_bytes(), 0, "a failed marker clear retains no image lease");
            const auto after = file_inventory(path);
            store.close();
            check(file_inventory(path) == after, "repeating a failed close neither removes nor duplicates its marker");
            check_eq(store.disk_charged_bytes(), charged, "repeating a failed close does not double-charge its marker");
        }
        {
            ConversationStore store(options_for(path, uint64_t(1) << 30, 0));
            check(store.enabled(), "the root reopens after a failed marker removal");
            check_eq(store.epoch(), epoch + 1, "the re-armed marker makes the next start distrust this domain");
            check(store.run_intent_active(), "the reopened store arms its own intent");
            const auto query = query_of(image, 7, 1);
            check(store.matches(query.selector, query.tokens, query.images).empty(),
                  "a re-armed intent prevents reuse of the previous run's record");
            ConversationStoreManifestInfo info;
            check(!store.manifest_info(head_id, info), "restart retires the record distrusted by the re-armed intent");
            check_eq(store.counters().heads, 0, "the re-armed domain has no authorized record after recovery");
            check(file_inventory(path + "/objects").empty() && directory_names(path + "/manifests").empty(),
                  "intent recovery reclaims the retired record's files");
            check_eq(store.counters().pending_delete_bytes, 0, "intent recovery leaves no deletion lease");
            store.close();
            check(!store.run_intent_active() && intent_names(path).empty(),
                  "healthy recovery removes the re-armed marker from disk");
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
    check(std::isfinite(read.materialize_ms) && read.materialize_ms > 0,
          "ready adoption preserves full prefetch body timing instead of near-zero demand wait");
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

// Only ids are initially authorized; the repeated state adds exactly one unique physical object.
SavedConversation revival_fixture(size_t copies) {
    SavedConversation image = fixture(381, 64);
    image.checkpoints.clear();
    image.kv.clear();
    image.live.gdn.assign(copies * 4096, 0x5a);
    image.live.ple.resize(0);
    image.live.tails.resize(0);
    image.live.dead.resize(0);
    image.live.block_pos.resize(0);
    return image;
}

void check_orphan_revival() {
    if (::geteuid() == 0) return; // Real unlink/rename permission failures require an unprivileged owner.
    for (size_t copies : {size_t(1), size_t(2)}) {
        TempRoot root;
        ConversationStore store(options_for(root.path, uint64_t(1) << 30, 0));
        check(store.invalidate_domain("empty durable catalog baseline") && store.counters().heads == 0,
              "establish the same durable empty catalog before and after retirement");
        const uint64_t empty_charge = store.disk_charged_bytes();
        const SavedConversation base = revival_fixture(0), image = revival_fixture(copies);
        ConversationStorePublishResult result;
        check(publish_and_wait(store, base, 1, 1, result) &&
                  result.status == ConversationStorePublishStatus::stored,
              "revival starts with only the token object authorized");
        const uint64_t base_charge = store.disk_charged_bytes();
        const std::string catalog = file_digest(root.path + "/catalog");
        const ConversationStoreStreamId gdn{0, uint16_t(ConversationStoreStreamKind::live_gdn), 0, 0};
        std::pair<Digest, uint64_t> identity;
        std::string why;
        check(identity_of(image, gdn, identity, why), "the revival object's physical identity is known");
        const Digest object = conversation_store_detail::object_digest(
            identity.first, gdn.kind, identity.second, 4096, image.live.gdn.data());
        const std::string name = sha256_hex(object);
        const std::string shard = root.path + "/objects/" + name.substr(0, 2);
        const std::string file = shard + "/" + name;
        SelectedHoldGate gate;
        store.set_catalog_gate_for_test([](void* p, bool staged) {
            if (staged) static_cast<SelectedHoldGate*>(p)->pause();
        }, &gate);
        store.set_publish_fault_for_test("catalog-rename");
        ConversationStorePublish request;
        request.request_id = request.generation = 2;
        check(store.publish(std::make_shared<SavedConversation>(image), request, why) == kAccepted,
              "the orphan-producing publication is admitted");
        gate.await();
        struct stat before {};
        check(::stat(file.c_str(), &before) == 0, "the orphan's real inode exists before catalog refusal");
        check(::chmod(shard.c_str(), 0500) == 0, "a real permission failure prevents orphan deletion and rewriting");
        gate.resume();
        check(store.wait_publish(result, 5000) && result.status == ConversationStorePublishStatus::failed,
              "catalog refusal leaves the new object without authority");
        store.set_catalog_gate_for_test(nullptr, nullptr);
        check(!gate.timed_out, "orphan retention uses the reached catalog seam, not a timeout");
        const uint64_t retained_charge = store.disk_charged_bytes();
        check(retained_charge > base_charge, "the undeletable orphan retains its original charge");
        check_eq(store.counters().orphan_bytes, retained_charge - base_charge,
                 "the stable orphan ledger accounts for the retained physical file exactly once");
        check(file_digest(root.path + "/catalog") == catalog, "orphan retention changes no catalog authority");
        const uint64_t written = store.counters().bytes_written;
        // A failed revival flush must put the same lease charge back into orphan ownership. Repeat each boundary
        // so that a retry cannot accidentally treat mere file/directory existence as proof of durability.
        uint64_t attempt = 3;
        for (const char* boundary : {"object-shard-sync", "object-parent-sync"}) {
            for (unsigned retry = 0; retry != 2; ++retry) {
                store.set_publish_fault_for_test(boundary);
                check(publish_and_wait(store, image, attempt, attempt, result) &&
                          result.status == ConversationStorePublishStatus::failed && result.released,
                      "each revival retry must traverse both real directory durability boundaries");
                ++attempt;
                check_eq(store.disk_charged_bytes(), retained_charge, "failed revival neither loses nor doubles its charge");
                check_eq(store.counters().orphan_bytes, retained_charge - base_charge,
                         "failed revival returns its charge to the stable orphan ledger");
                check_eq(store.counters().bytes_written, written, "a failed revival flush rewrites no payload");
                check_eq(store.counters().heads, 1, "a failed revival adds no catalog authority");
                check(file_digest(root.path + "/catalog") == catalog &&
                          directory_names(root.path + "/manifests").size() == 1 &&
                          directory_names(root.path + "/tmp").empty(),
                      "revival durability refusal precedes manifest creation and preserves prior authority");
                check(store.healthy() && !store.writer_busy() && store.ram_lease_bytes() == 0,
                      "directory refusal settles the publication without poisoning catalog authority");
            }
        }
        check(publish_and_wait(store, image, attempt, attempt, result) &&
                  result.status == ConversationStorePublishStatus::stored && result.new_bytes == 0,
              "revival and a duplicate in the same publication reuse the pinned object without rewriting");
        struct stat after {};
        check(::stat(file.c_str(), &after) == 0 && after.st_ino == before.st_ino,
              "revival keeps the original inode");
        check_eq(store.counters().orphan_bytes, 0, "successful revival transfers ownership out of the orphan ledger");
        // All ledger transitions must be in this process: reopening reconstructs the ledger and hides a lost charge.
        ConversationStoreManifestInfo revived_info;
        check(store.manifest_info(result.head_id, revived_info), "the revived manifest's allocated charge is observable");
        check_eq(store.disk_charged_bytes(), retained_charge + revived_info.manifest_bytes,
                 "publication adds one manifest charge, not another charge for the revived object");
        ConversationStoreRead read;
        check(store.demand(query_of(image, 91, 1), why) && store.wait_ready(91, 1, read, 5000) &&
                  read.head_id == result.head_id && read.image && images_equal(*read.image, image, why),
              "the revived repeated state is restored byte for byte");
        read.image.reset();
        store.release_read(91, 1);
        check(::chmod(shard.c_str(), 0700) == 0, "normal shard permissions are restored before retirement");
        check(store.invalidate_domain("revival retirement fixture"), "the revived head can be durably retired");
        check_eq(store.counters().heads, 0, "retirement removes every authority");
        check_eq(store.disk_charged_bytes(), empty_charge,
                 "retirement and confirmed deletion refund the revived object's original charge");
        check(file_inventory(root.path + "/objects").empty(), "retirement deletes all payload objects");
        check(store.healthy() && store.ram_lease_bytes() == 0, "revival and retirement preserve health and image ownership");
        store.close();
    }
}

void check_object_directory_retry() {
    TempRoot root;
    ConversationStore store(options_for(root.path, uint64_t(1) << 30, 0));
    check(store.invalidate_domain("empty durable catalog baseline") && store.counters().heads == 0,
          "establish the same durable empty catalog before and after retirement");
    const uint64_t empty_charge = store.disk_charged_bytes();
    const std::string catalog = file_digest(root.path + "/catalog");
    const SavedConversation image = revival_fixture(0);
    ConversationStorePublishResult result;
    // mkdir succeeds before the flush fault. Default mode refuses before creating an object; batch mode refuses
    // after renames and reclaims them. In either mode the shard survives, so existence cannot prove parent durability.
    for (uint64_t attempt = 1; attempt <= 2; ++attempt) {
        store.set_publish_fault_for_test("object-parent-sync");
        check(publish_and_wait(store, image, attempt, attempt, result) &&
                  result.status == ConversationStorePublishStatus::failed && result.released,
              "both newly created and existing shards must retry a failed objects-parent flush");
        check_eq(directory_names(root.path + "/objects").size(), 1,
                 "the retry really encounters the shard left by the earlier mkdir");
        check(file_inventory(root.path + "/objects").empty() &&
                  directory_names(root.path + "/tmp").empty() && directory_names(root.path + "/manifests").empty(),
              "parent flush refusal leaves no object or manifest authority after cleanup");
        check_eq(store.disk_charged_bytes(), empty_charge, "a parent refusal refunds only confirmed absent files");
        check_eq(store.counters().orphan_bytes, 0, "a parent refusal leaves no physical orphan charge");
        check(file_digest(root.path + "/catalog") == catalog && store.healthy(),
              "parent durability refusal preserves catalog authority and health");
    }
    store.set_publish_fault_for_test("object-shard-sync");
    check(publish_and_wait(store, image, 3, 3, result) &&
              result.status == ConversationStorePublishStatus::failed,
          "a new object's post-rename shard flush can refuse publication");
    check(file_inventory(root.path + "/objects").empty() && directory_names(root.path + "/tmp").empty() &&
              directory_names(root.path + "/manifests").empty(),
          "ordinary post-rename refusal reclaims the real object without publishing a manifest");
    check_eq(store.disk_charged_bytes(), empty_charge, "confirmed deletion refunds the failed post-rename write once");
    check(publish_and_wait(store, image, 4, 4, result) && result.status == ConversationStorePublishStatus::stored,
          "a fully flushed same-process retry becomes catalog authority");
    check(store.invalidate_domain("directory retry fixture"), "the retried object can be retired normally");
    check_eq(store.disk_charged_bytes(), empty_charge, "retirement returns all physical payload charges");
    store.close();
}

// One failed write, one charge.  A write that fails before its file reaches its own name leaves bytes only in the
// temporary, and the store must credit that one file exactly once - never a temporary *and* the name it never
// reached.  The failure is injected at the operation itself (the directory refuses the write), not ahead of it.
void check_failed_write_charges_once() {
    // The phase hooks are distinct from the operation-level permission failures below: objects refuses before any
    // new file, manifest after object creation, and catalog after both objects and a manifest have been written.
    // Each failure must release the image and its object leases without retiring the covered predecessor.
    const char* old_batch = std::getenv("STRATA_STORE_DIR_FSYNC_BATCH");
    const bool had_batch = old_batch != nullptr;
    const std::string saved_batch = old_batch ? old_batch : "";
    for (const char* step : {"objects", "batch-shard-open", "batch-shard-sync", "batch-shard-partial",
                             "batch-parent-sync", "batch-data-sync", "batch-shard-allocate", "manifest", "catalog"}) {
        // Preserve the caller's mode for legacy phases; only the new fault cases force the candidate on.
        ::setenv("STRATA_STORE_DIR_FSYNC_BATCH",
                 std::strncmp(step, "batch-", 6) == 0 ? "1" : (had_batch ? saved_batch.c_str() : "0"), 1);
        TempRoot root;
        const auto options = options_for(root.path, uint64_t(1) << 30, 0);
        const SavedConversation prior = fixture(371, 1024);
        const SavedConversation grown = appended_turn(prior, 2048, 371);
        uint64_t head_id = 0;
        uint64_t epoch = 0;
        uint64_t successor_id = 0;
        {
            ConversationStore store(options);
            ConversationStorePublishResult head;
            check(publish_and_wait(store, prior, 1, 1, head) &&
                      head.status == ConversationStorePublishStatus::stored,
                  "the phase-failure root stores its predecessor");
            head_id = head.head_id;
            epoch = store.epoch();
            const uint64_t charged = store.disk_charged_bytes();
            const size_t descriptors = directory_names("/proc/self/fd").size();
            const auto inventory = file_inventory(root.path);
            const std::string catalog = file_digest(root.path + "/catalog");
            const auto markers = intent_names(root.path);
            check_eq(markers.size(), 1, "the phase-failure root has one real run-intent marker");
            store.set_publish_fault_for_test(step);
            auto image = std::make_shared<SavedConversation>(grown);
            const std::weak_ptr<SavedConversation> alias = image;
            ConversationStorePublish request;
            request.request_id = 2;
            request.generation = 2;
            request.supersedes = head_id;
            request.cover = true;
            std::string why;
            check(store.publish(image, request, why) == kAccepted, "the phase-failure cover is admitted");
            image.reset();
            ConversationStorePublishResult failed;
            check(store.wait_publish(failed, 5000), "the phase-failure cover settles");
            check(failed.status == ConversationStorePublishStatus::failed && !failed.detail.empty(),
                  "a refused publication phase reports failure");
            check(failed.released && alias.expired(), "a failed phase really drops its publication image");
            check(!store.writer_busy(), "a failed phase releases its writer slot");
            check_eq(store.ram_lease_bytes(), 0, "a failed phase leaves no publication lease");
            check(store.healthy(), "a failed phase before catalog publication leaves authority trustworthy");
            check_eq(store.counters().heads, 1, "a failed phase neither authorizes a successor nor retires its predecessor");
            ConversationStoreManifestInfo info;
            check(store.manifest_info(head_id, info), "a failed phase preserves the predecessor's manifest authority");
            check(file_digest(root.path + "/catalog") == catalog, "a failed phase preserves the durable catalog");
            check_eq(directory_names("/proc/self/fd").size(), descriptors,
                     "a failed publication phase leaks no transient descriptor, including exception unwinding");
            check(file_inventory(root.path) == inventory, "a failed phase removes every file its attempt created");
            check(directory_names(root.path + "/tmp").empty(), "a failed phase leaves no scratch file");
            check(intent_names(root.path) == markers && store.run_intent_active(),
                  "an ordinary publication failure retains the real run intent");
            check_eq(store.disk_charged_bytes(), charged, "a failed phase refunds only its abandoned files");
            check_eq(store.counters().pending_delete_bytes, 0, "a failed phase leaves no unpaid deletion");
            ConversationStoreRead read;
            check(store.demand(query_of(prior, 3, 1), why) && store.wait_ready(3, 1, read, 5000) &&
                      read.status == ConversationStoreReadStatus::ready && read.image &&
                      images_equal(*read.image, prior, why),
                  "a failed phase leaves the predecessor restorable byte for byte");
            read.image.reset();
            store.release_read(3, 1);
            check(store.publish(std::make_shared<SavedConversation>(grown), request, why) == kAccepted,
                  "the one-shot phase fault permits an immediate retry");
            ConversationStorePublishResult retried;
            check(store.wait_publish(retried, 5000) && retried.status == ConversationStorePublishStatus::replaced,
                  "the same-run retry replaces the original authority");
            successor_id = retried.head_id;
            check(!store.manifest_info(head_id, info), "the retry retires the predecessor's manifest authority");
            check_eq(store.counters().pending_delete_bytes, 0, "a failed attempt leaves no object lease blocking retirement");
            store.close();
            check(!store.run_intent_active() && intent_names(root.path).empty(),
                  "an ordinary phase failure still permits a genuinely clean close");
            check_eq(store.ram_lease_bytes(), 0, "a clean close after a failed phase leaves no image lease");
        }
        {
            ConversationStore store(options);
            check(store.enabled() && store.healthy(), "the root reopens after an ordinary phase failure");
            check_eq(store.epoch(), epoch, "a clean close after a phase failure preserves the epoch");
            check_eq(store.counters().heads, 1, "restart retains only the successfully retried authority");
            check_eq(store.counters().corrupt_records, 0, "a failed phase leaves no corrupt record for restart");
            ConversationStoreRead read;
            std::string why;
            check(store.demand(query_of(grown, 4, 1), why) && store.wait_ready(4, 1, read, 5000) &&
                      read.status == ConversationStoreReadStatus::ready && read.head_id == successor_id && read.image &&
                      images_equal(*read.image, grown, why),
                  "restart restores the successful retry byte for byte");
            read.image.reset();
            store.release_read(4, 1);
            check_eq(store.ram_lease_bytes(), 0, "the restarted read releases its image lease");
            store.close();
        }
    }
    if (had_batch)
        ::setenv("STRATA_STORE_DIR_FSYNC_BATCH", saved_batch.c_str(), 1);
    else
        ::unsetenv("STRATA_STORE_DIR_FSYNC_BATCH");

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
        check(failed.released && store.ram_lease_bytes() == 0,
              "an object creation failure drops its publication lease");
        check(directory_names(path + "/tmp").empty(), "a failed object creation leaves no temporary file");
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
        check(failed.released && store.ram_lease_bytes() == 0,
              "a manifest rename failure drops its publication lease");
        check(directory_names(path + "/tmp").empty(), "a failed manifest rename removes its real temporary file");
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
    StoreReadTraceCapture trace;
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
        check(!store.retains_head(first.head_id) && store.retains_head(second_head),
              "head pressure removes the actual victim while retaining the committed successor");
        check(store.counters().reclaimed_objects > 0,
              "the retired branch's objects were still reclaimed, on the pass that follows the commit");
        const auto retired_records = trace.records("store_head ");
        const auto* retired = read_trace(retired_records, "retired", request.request_id, "head_limit");
        check(retired && retired->number("head") == first.head_id &&
                  !retired->number("pinned") && !retired->number("head_leases") &&
                  retired->number("replacement_head") == second_head,
              "a committed head-ceiling eviction records its actual victim and protection state");
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

// The batch pause is only a scheduling seam. Assertions use real files, public authority/restore APIs and charges.
void check_object_directory_batch() {
    const char* old = std::getenv("STRATA_STORE_DIR_FSYNC_BATCH");
    const bool had = old != nullptr;
    const std::string saved = old ? old : "";
    ::setenv("STRATA_STORE_DIR_FSYNC_BATCH", "1", 1);

    // The same image must need the fixed tracker/name/descriptor workspace at admission, not only at execution.
    const auto shape = options_for("/unused", uint64_t(1) << 30, 0);
    const SavedConversation hollow = revival_fixture(0);
    const auto fresh = [](ConversationStore&) {
        ConversationStorePublish request;
        request.request_id = request.generation = 1;
        return request;
    };
    ::setenv("STRATA_STORE_DIR_FSYNC_BATCH", "0", 1);
    const uint64_t ordinary_floor = publication_admission_floor(shape, hollow, fresh, uint64_t(16) << 20);
    ::setenv("STRATA_STORE_DIR_FSYNC_BATCH", "1", 1);
    const uint64_t batch_floor = publication_admission_floor(shape, hollow, fresh, uint64_t(16) << 20);
    check_eq(batch_floor, ordinary_floor + 4 * sizeof(uint64_t) + 3 * sizeof(char) + sizeof(int),
             "the enabled tracker, stack shard name and transient descriptor are included in writer admission");
    check(publication_is_admitted(shape, hollow, fresh, batch_floor) &&
              !publication_is_admitted(shape, hollow, fresh, batch_floor - 1),
          "batch publication fits exactly its counted allowance and refuses one byte less");

    // Real unlink permission failures retain renamed objects. Revival must prove data/directory durability again,
    // without rewriting or double-charging them. The third attempt succeeds using those very same files.
    for (const char* boundary : {"batch-shard-sync", "batch-parent-sync"}) {
        if (::geteuid() == 0) continue; // Retained-orphan assertions require real unlink permission denial.
        TempRoot root;
        const auto options = options_for(root.path, uint64_t(1) << 30, 0);
        const SavedConversation image = fixture(381, 1024);
        uint64_t epoch = 0, successful_head = 0;
        {
            ConversationStore store(options);
            epoch = store.epoch();
            const uint64_t base_charge = store.disk_charged_bytes();
            const std::string catalog = file_digest(root.path + "/catalog");
            uint64_t written = 0, retained_charge = 0;
            std::vector<std::string> object_files;
            std::vector<ino_t> object_inodes;
            for (unsigned attempt = 0; attempt < 3; ++attempt) {
                SelectedHoldGate gate;
                store.set_object_batch_gate_for_test([](void* p) {
                    static_cast<SelectedHoldGate*>(p)->pause();
                }, &gate);
                if (attempt < 2) store.set_publish_fault_for_test(boundary);
                ConversationStorePublish request;
                request.request_id = attempt + 1;
                request.generation = attempt + 1;
                std::string why;
                check(store.publish(std::make_shared<SavedConversation>(image), request, why) == kAccepted,
                      "the orphan/revival publication is admitted");
                gate.await();
                check(directory_names(root.path + "/manifests").empty(),
                      "the batch pause precedes every manifest file creation");
                const auto query = query_of(image, 90, 1);
                check(store.matches(query.selector, query.tokens, query.images).empty(),
                      "renamed objects and a pending batch confer no catalog authority");
                if (attempt == 0) {
                    object_files = file_inventory(root.path + "/objects");
                    check(!object_files.empty(), "the batch really has renamed object files");
                    for (const auto& file : object_files) {
                        struct stat info {};
                        check(::stat((root.path + "/objects" + file).c_str(), &info) == 0,
                              "the staged object's real inode can be recorded");
                        object_inodes.push_back(info.st_ino);
                    }
                    written = store.counters().bytes_written;
                    for (const auto& shard : directory_names(root.path + "/objects"))
                        check(::chmod((root.path + "/objects/" + shard).c_str(), 0500) == 0,
                              "renamed objects become genuinely undeletable without blocking reads or fsync");
                } else {
                    if (attempt == 1)
                        check_eq(store.counters().bytes_written, written, "failed revival writes no payload a second time");
                    check(file_inventory(root.path + "/objects") == object_files,
                          "revival retains the exact same set of physical files");
                    for (size_t i = 0; i < object_files.size(); ++i) {
                        struct stat info {};
                        check(::stat((root.path + "/objects" + object_files[i]).c_str(), &info) == 0 &&
                                  info.st_ino == object_inodes[i],
                              "revival reuses the same inode instead of replacing its content-addressed name");
                    }
                }
                if (attempt == 2)
                    for (const auto& shard : directory_names(root.path + "/objects"))
                        check(::chmod((root.path + "/objects/" + shard).c_str(), 0700) == 0,
                              "the successful retry restores normal shard permissions");
                gate.resume();
                ConversationStorePublishResult result;
                check(store.wait_publish(result, 5000), "the batch/revival attempt settles");
                store.set_object_batch_gate_for_test(nullptr, nullptr);
                check(!gate.timed_out, "the deterministic batch gate never relies on its timeout");
                check(result.released && store.ram_lease_bytes() == 0 && !store.writer_busy(),
                      "every batch/revival outcome releases its image, RAM lease and writer slot");
                check(store.healthy(), "object durability refusal never latches uncertain catalog authority");
                if (attempt < 2) {
                    check(result.status == ConversationStorePublishStatus::failed,
                          "the revival itself must traverse the armed directory durability boundary");
                    check_eq(store.counters().heads, 0, "failed batch/revival authorizes no head");
                    check(file_digest(root.path + "/catalog") == catalog,
                          "a roomy failed batch/revival leaves the catalog unchanged");
                    check(directory_names(root.path + "/tmp").empty() &&
                              directory_names(root.path + "/manifests").empty(),
                          "an object durability failure creates neither manifest nor lingering scratch");
                    const uint64_t charge = store.disk_charged_bytes();
                    check(charge > base_charge, "failed deletion retains a real physical object charge");
                    check_eq(store.counters().orphan_bytes, charge - base_charge,
                             "each abandoned physical object retains exactly one stable orphan charge");
                    if (attempt == 0) retained_charge = charge;
                    else check_eq(charge, retained_charge, "failed revival adds no second object charge");
                } else {
                    check(result.status == ConversationStorePublishStatus::stored && result.new_bytes == 0,
                          "a fully flushed revival becomes durable authority without rewriting any payload");
                    successful_head = result.head_id;
                    for (size_t i = 0; i < object_files.size(); ++i) {
                        struct stat info {};
                        check(::stat((root.path + "/objects" + object_files[i]).c_str(), &info) == 0 &&
                                  info.st_ino == object_inodes[i],
                              "successful revival leaves the retained payload inode unchanged");
                    }
                    check_eq(store.counters().orphan_bytes, 0,
                             "successful revival transfers the stable orphan ledger into live object ownership");
                }
            }
            store.close();
            check(!store.run_intent_active(), "successful revival permits durable clean shutdown");
        }
        {
            ConversationStore store(options);
            check_eq(store.epoch(), epoch, "clean reopening preserves the successfully retried epoch");
            check_eq(store.counters().heads, 1, "only the successful revival survives reopening");
            ConversationStoreRead read;
            std::string why;
            check(store.demand(query_of(image, 91, 1), why) && store.wait_ready(91, 1, read, 5000) &&
                      read.head_id == successful_head && read.image && images_equal(*read.image, image, why),
                  "the revived history restores byte for byte after clean reopening");
            read.image.reset();
            store.release_read(91, 1);
            store.close();
        }
    }

    // Two of 257 distinct chunks must share a shard. Authorize ids first, so only the colliding objects (and a
    // duplicate chunk) need durability in the successor. A partial-batch fault cannot fire on that single shard.
    {
        TempRoot root;
        SavedConversation base = fixture(383, 1024);
        base.checkpoints.clear();
        base.kv.clear();
        base.live.gdn.resize(0);
        base.live.ple.resize(0);
        base.live.tails.resize(0);
        base.live.dead.resize(0);
        base.live.block_pos.resize(0);
        SavedConversation shared = base;
        shared.live.gdn.resize(4096);
        const ConversationStoreStreamId gdn{0, uint16_t(ConversationStoreStreamKind::live_gdn), 0, 0};
        std::pair<Digest, uint64_t> identity;
        std::string why;
        check(identity_of(shared, gdn, identity, why), "the shared-shard fixture has a valid stream identity");
        std::array<int, 256> first;
        first.fill(-1);
        int left = -1, right = -1;
        for (int seed = 0; seed <= 256; ++seed) {
            std::vector<uint8_t> bytes(4096, 0);
            bytes[0] = uint8_t(seed);
            bytes[1] = uint8_t(seed >> 8);
            const auto object = conversation_store_detail::object_digest(identity.first, gdn.kind, identity.second,
                                                                          bytes.size(), bytes.data());
            if (first[object[0]] >= 0) {
                left = first[object[0]];
                right = seed;
                break;
            }
            first[object[0]] = seed;
        }
        check(left >= 0 && right >= 0, "two distinct chunks share a shard by the pigeonhole principle");
        {
            auto& bytes = shared.live.gdn.write();
            bytes.assign(3 * 4096, 0);
            for (size_t index = 0; index < 3; ++index) {
                const int seed = index == 1 ? right : left;
                bytes[index * 4096] = uint8_t(seed);
                bytes[index * 4096 + 1] = uint8_t(seed >> 8);
            }
        }
        ConversationStore store(options_for(root.path, uint64_t(1) << 30, 0));
        ConversationStorePublishResult prior, result;
        check(publish_and_wait(store, base, 1, 1, prior), "the single-shard fixture first authorizes its ids");
        const uint64_t chunks = store.counters().published_chunks;
        const uint64_t dedup = store.counters().dedup_chunks;
        store.set_publish_fault_for_test("batch-shard-partial");
        check(publish_and_wait(store, shared, 2, 2, result) &&
                  result.status == ConversationStorePublishStatus::stored,
              "distinct objects in one shard and a repeated chunk need no second shard flush");
        const uint64_t shared_head = result.head_id;
        check_eq(store.counters().published_chunks - chunks, 2, "the shared-shard publication creates only two objects");
        check(store.counters().dedup_chunks > dedup, "the repeated new chunk is deduplicated");
        check(publish_and_wait(store, fixture(385, 1024), 3, 3, result) &&
                  result.status == ConversationStorePublishStatus::failed,
              "the partial fault remains armed until a second touched shard exists");

        // A wholly authorized donor/dedup publication must leave a first-shard fault armed for new objects.
        store.set_publish_fault_for_test("batch-shard-sync");
        ConversationStorePublish donor_request;
        donor_request.request_id = 4;
        donor_request.generation = 4;
        donor_request.reuse.push_back({shared_head, gdn, 0, 0, 4096});
        check(store.publish(std::make_shared<SavedConversation>(shared), donor_request, why) == kAccepted,
              "the wholly authorized donor/dedup publication is admitted");
        check(store.wait_publish(result, 5000) && result.status == ConversationStorePublishStatus::stored &&
                  result.reused_bytes == 4096 && result.new_bytes == 0,
              "authorized donor and deduplicated objects publish without a directory batch");
        check(publish_and_wait(store, fixture(387, 1024), 5, 5, result) &&
                  result.status == ConversationStorePublishStatus::failed,
              "authorized reuse leaves the durability fault armed for genuinely new objects");
        store.close();
    }

    // A selected immutable reader remains usable during the batch; renames alone never authorize the successor.
    {
        TempRoot root;
        ConversationStore store(options_for(root.path, uint64_t(1) << 30, 0));
        const SavedConversation prior = fixture(389, 1024), next = fixture(391, 1024);
        ConversationStorePublishResult result;
        check(publish_and_wait(store, prior, 1, 1, result), "the concurrent-reader fixture stores its predecessor");
        SelectedHoldGate read_gate, batch_gate;
        store.set_read_gate_for_test([](void* p) { static_cast<SelectedHoldGate*>(p)->pause(); }, &read_gate);
        store.set_object_batch_gate_for_test([](void* p) { static_cast<SelectedHoldGate*>(p)->pause(); }, &batch_gate);
        std::string why;
        check(store.demand(query_of(prior, 96, 1), why), "a reader selects the already-authorized predecessor");
        read_gate.await();
        ConversationStorePublish request;
        request.request_id = 2;
        request.generation = 2;
        check(store.publish(std::make_shared<SavedConversation>(next), request, why) == kAccepted,
              "a successor can emit objects beside the selected immutable reader");
        batch_gate.await();
        const auto query = query_of(next, 97, 1);
        check(store.matches(query.selector, query.tokens, query.images).empty(),
              "the successor is unselectable while its object-directory batch is paused");
        check_eq(directory_names(root.path + "/manifests").size(), 1,
                 "the paused successor has not created a manifest");
        read_gate.resume();
        ConversationStoreRead read;
        check(store.wait_ready(96, 1, read, 5000) && read.image && images_equal(*read.image, prior, why),
              "the selected reader restores unchanged while the writer remains paused at object durability");
        read.image.reset();
        store.release_read(96, 1);
        store.cancel_publication(2, 2);
        check(store.writer_busy(), "cancelling an already-started batch cannot retire its active writer");
        batch_gate.resume();
        check(store.wait_publish(result, 5000) && result.status == ConversationStorePublishStatus::stored,
              "cancelling a started publication is a no-op: the successor stores after its durability drain");
        check(!store.matches(query.selector, query.tokens, query.images).empty(),
              "catalog publication, not the pause or renames, makes the successor selectable");
        check(!read_gate.timed_out && !batch_gate.timed_out, "reader/batch interleaving uses no timeout as evidence");
        store.set_read_gate_for_test(nullptr, nullptr);
        store.set_object_batch_gate_for_test(nullptr, nullptr);
        store.close();
    }

    // Destruction after refusal, unlike close(), keeps intent armed and preserves existing epoch distrust.
    {
        TempRoot root;
        const auto options = options_for(root.path, uint64_t(1) << 30, 0);
        uint64_t epoch = 0;
        {
            ConversationStore store(options);
            epoch = store.epoch();
            store.set_publish_fault_for_test("batch-parent-sync");
            ConversationStorePublishResult result;
            check(publish_and_wait(store, fixture(393, 1024), 1, 1, result) &&
                      result.status == ConversationStorePublishStatus::failed,
                  "the interrupted-run fixture fails before authorizing its publication");
        }
        ConversationStore reopened(options);
        check_eq(reopened.epoch(), epoch + 1, "unclean restart after a batch refusal distrusts the previous epoch");
        check_eq(reopened.counters().heads, 0, "interrupted object publication never resurrects a history");
        check(directory_names(root.path + "/manifests").empty(),
              "unclean recovery finds no manifest authority from an interrupted batch");
        reopened.close();
    }
    if (had) ::setenv("STRATA_STORE_DIR_FSYNC_BATCH", saved.c_str(), 1);
    else ::unsetenv("STRATA_STORE_DIR_FSYNC_BATCH");
}

void check_object_data_batch() {
    const char* old = ::getenv("STRATA_STORE_DIR_FSYNC_BATCH");
    const bool had = old != nullptr;
    const std::string saved = old ? old : "";
    const SavedConversation image = fixture(397, 1025);
    ConversationStorePublishTrace traces[2];
    ConversationStoreCounters counts[2];
    uint64_t charges[2]{};
    std::map<std::string, std::string> payloads[2];
    for (unsigned batching = 0; batching < 2; ++batching) {
        ::setenv("STRATA_STORE_DIR_FSYNC_BATCH", batching ? "1" : "0", 1);
        TempRoot root;
        auto options = options_for(root.path, uint64_t(1) << 30, 0);
        options.publication_timing = true;
        {
            ConversationStore store(options);
            store.set_publish_trace_for_test(
                +[](void* context, const ConversationStorePublishTrace& trace) noexcept {
                    *static_cast<ConversationStorePublishTrace*>(context) = trace;
                }, &traces[batching]);
            ConversationStorePublishResult result;
            check(publish_and_wait(store, image, 1, 1, result) &&
                      result.status == ConversationStorePublishStatus::stored,
                  "both durability modes publish the same complete image");
            counts[batching] = store.counters();
            charges[batching] = store.disk_charged_bytes();
            for (const auto& file : file_inventory(root.path + "/objects"))
                payloads[batching][file] = file_digest(root.path + "/objects" + file);
            store.close(); // Join the trace observer before reading its summary.
            check(store.shutdown_state() == ConversationStoreShutdownState::durable_clean,
                  "both durability modes complete a clean durable close");
        }
        {
            ConversationStore reopened(options);
            ConversationStoreRead read;
            std::string why;
            check(reopened.demand(query_of(image, 2, 1), why) &&
                      reopened.wait_ready(2, 1, read, 5000) && read.image &&
                      images_equal(*read.image, image, why),
                  "the stored image restores byte for byte after reopening in either mode");
            read.image.reset();
            reopened.release_read(2, 1);
            reopened.close();
        }
    }
    check(counts[0].published_chunks > 2, "the durability fixture writes multiple distinct immutable objects");
    check_eq(traces[0].file_fsync_calls, counts[0].published_chunks + 2,
             "the default writer fsyncs each object, manifest and catalog");
    check_eq(traces[1].file_fsync_calls, 2,
             "the batched writer retains only the manifest and catalog file fsyncs");
    check_eq(traces[0].data_sync_calls, 0, "the default writer does not sweep the filesystem");
    check_eq(traces[1].data_sync_calls, 1, "one data sweep replaces all per-object data fsyncs");
    check(traces[1].payload_finished_ns <= traces[1].data_sync_started_ns &&
              traces[1].data_sync_started_ns <= traces[1].data_sync_finished_ns &&
              traces[1].data_sync_finished_ns <= traces[1].manifest_started_ns &&
              traces[1].manifest_started_ns < traces[1].manifest_finished_ns &&
              traces[1].manifest_finished_ns <= traces[1].catalog_started_ns &&
              traces[1].catalog_started_ns <= traces[1].catalog_applied_ns &&
              traces[1].catalog_applied_ns <= traces[1].catalog_finished_ns,
          "payload data is durable before manifest creation, manifest before catalog write and apply");
    check(payloads[0] == payloads[1], "both modes create byte-identical content-addressed payload files");
    check_eq(counts[1].objects, counts[0].objects, "batching preserves the exact physical object count");
    check_eq(counts[1].object_bytes, counts[0].object_bytes, "batching preserves every rounded object charge");
    check_eq(counts[1].published_chunks, counts[0].published_chunks, "batching preserves per-object publication counts");
    check_eq(counts[1].bytes_written, counts[0].bytes_written, "batching preserves the exact completed byte count");
    check_eq(traces[1].file_write_bytes, traces[0].file_write_bytes, "batching changes no encoded payload bytes");
    check_eq(charges[1], charges[0], "batching changes no total disk accounting");

    ::setenv("STRATA_STORE_DIR_FSYNC_BATCH", "1", 1);
    for (const char* boundary : {"objects", "batch-data-sync", "batch-shard-sync", "batch-parent-sync", "manifest", "catalog"}) {
        TempRoot root;
        ConversationStorePublishTrace trace;
        ConversationStore store(options_for(root.path, uint64_t(1) << 30, 0));
        const uint64_t charge = store.disk_charged_bytes();
        const std::string catalog = file_digest(root.path + "/catalog");
        store.set_publish_trace_for_test(
            +[](void* context, const ConversationStorePublishTrace& value) noexcept {
                *static_cast<ConversationStorePublishTrace*>(context) = value;
            }, &trace);
        store.set_publish_fault_for_test(boundary);
        ConversationStorePublishResult result;
        check(publish_and_wait(store, image, 1, 1, result) && result.status == ConversationStorePublishStatus::failed,
              "a refused pre/post-data-sweep boundary fails the publication");
        check(file_digest(root.path + "/catalog") == catalog && store.counters().heads == 0,
              "a refused sweep or later staging step cannot authorize the image");
        check(file_inventory(root.path + "/objects").empty() && directory_names(root.path + "/manifests").empty() &&
                  directory_names(root.path + "/tmp").empty(),
              "a refused boundary reclaims its actual files without creating a retained manifest");
        check_eq(store.disk_charged_bytes(), charge, "failed sweep/staging refunds exactly the files confirmed gone");
        check(store.healthy() && result.released && store.ram_lease_bytes() == 0 && !store.writer_busy(),
              "pre-catalog failure preserves trust and releases the single writer and image lease");
        store.close();
        const bool before_data = std::strcmp(boundary, "objects") == 0;
        const bool after_data = std::strcmp(boundary, "manifest") == 0 || std::strcmp(boundary, "catalog") == 0;
        check_eq(trace.data_sync_calls, before_data ? 0 : 1,
                 "new objects fail before any sweep, otherwise data durability is attempted exactly once");
        check((trace.manifest_started_ns != 0) == after_data,
              "only a successful data sweep can reach manifest staging");
        if (std::strcmp(boundary, "batch-data-sync") == 0)
            check(result.detail.find("syncfs failed:") != std::string::npos,
                  "a filesystem EIO is propagated through the real data-sweep error path");
    }

    // Authorized dedup requires no data sweep; the one-shot fault must remain armed until genuinely new data.
    {
        TempRoot root;
        std::atomic<uint64_t> sweeps{UINT64_MAX};
        ConversationStore store(options_for(root.path, uint64_t(1) << 30, 0));
        ConversationStorePublishResult result;
        check(publish_and_wait(store, image, 1, 1, result), "dedup starts with a durably authorized image");
        store.set_publish_trace_for_test(
            +[](void* context, const ConversationStorePublishTrace& value) noexcept {
                static_cast<std::atomic<uint64_t>*>(context)->store(value.data_sync_calls);
            }, &sweeps);
        store.set_publish_fault_for_test("batch-data-sync");
        check(publish_and_wait(store, image, 2, 2, result) &&
                  result.status == ConversationStorePublishStatus::stored && result.new_bytes == 0,
              "wholly authorized dedup publishes without touching the data-sweep fault");
        check(await_store_progress([&] { return sweeps.load() != UINT64_MAX; }),
              "the bounded observer delivers the dedup syscall summary");
        check_eq(sweeps.load(), 0, "authorized dedup needs no filesystem-wide data sweep");
        check(publish_and_wait(store, fixture(399, 1025), 3, 3, result) &&
                  result.status == ConversationStorePublishStatus::failed,
              "new payload, unlike authorized dedup, must pass the armed data-sweep boundary");
        store.close();
    }

    // The existing shutdown fence is still before the filesystem sweep, not after durability work.
    {
        TempRoot root;
        SelectedHoldGate gate;
        ConversationStorePublishTrace trace;
        ConversationStore store(options_for(root.path, uint64_t(1) << 30, 0));
        store.set_publish_trace_for_test(
            +[](void* context, const ConversationStorePublishTrace& value) noexcept {
                *static_cast<ConversationStorePublishTrace*>(context) = value;
            }, &trace);
        store.set_publish_payload_gate_for_test(
            +[](void* context) { static_cast<SelectedHoldGate*>(context)->pause(); }, &gate);
        ConversationStorePublish request;
        request.request_id = request.generation = 1;
        std::string why;
        check(store.publish(std::make_shared<SavedConversation>(image), request, why) == kAccepted,
              "the fence fixture starts the real payload writer");
        gate.await();
        const uint64_t deadline = shutdown_now_ms() + 8000;
        store.request_shutdown(deadline);
        gate.resume();
        store.finish_shutdown(); // Join the observer; do not upgrade the selected deadline.
        ConversationStorePublishResult result;
        check(store.take_publish(result) && result.status == ConversationStorePublishStatus::disabled,
              "the unchanged pre-manifest fence refuses payload completed after shutdown was requested");
        check_eq(trace.data_sync_calls, 0, "a fenced publication never enters the deferred data sweep");
        check_eq(trace.manifest_started_ns, 0, "the shutdown fence still precedes every manifest byte");
        check(directory_names(root.path + "/manifests").empty() && !gate.timed_out,
              "the fence leaves no manifest and the scheduling seam was explicitly released");
    }

    // Process exits on both sides of the data sweep leave no manifest or authority, without finalization.
    for (const bool before_data : {true, false}) {
        TempRoot root;
        const auto options = options_for(root.path, uint64_t(1) << 30, 0);
        uint64_t epoch = 0;
        {
            ConversationStore seed(options);
            epoch = seed.epoch();
            seed.close();
        }
        const std::string catalog = file_digest(root.path + "/catalog");
        const pid_t child = ::fork(); // no live store threads are inherited
        check(child >= 0, "the pre-manifest crash fixture creates its child process");
        if (child == 0) {
            ::alarm(15);
            ConversationStore store(options);
            if (before_data)
                store.set_publish_payload_gate_for_test(+[](void*) { ::_exit(81); }, nullptr);
            else
                store.set_object_batch_gate_for_test(+[](void*) { ::_exit(81); }, nullptr);
            ConversationStorePublishResult result;
            publish_and_wait(store, image, 1, 1, result);
            ::_exit(82); // Reaching settlement would miss the intended process-crash boundary.
        }
        if (child > 0) {
            int status = 0;
            check(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 81,
                  "the child exits from the real writer seam, without running shutdown/destructors");
            check(directory_names(root.path + "/manifests").empty() &&
                      (before_data ? !directory_names(root.path + "/tmp").empty() &&
                                         file_inventory(root.path + "/objects").empty()
                                   : directory_names(root.path + "/tmp").empty() &&
                                         !file_inventory(root.path + "/objects").empty()),
                  "a crash leaves only tmp before data durability, only named objects after it, and never a manifest");
            check(file_digest(root.path + "/catalog") == catalog && intent_names(root.path).size() == 1,
                  "the crash leaves the prior catalog authority unchanged and the actual intent armed");
            ConversationStore reopened(options);
            check_eq(reopened.epoch(), epoch + 1, "the armed intent invalidates the interrupted epoch");
            check_eq(reopened.counters().heads, 0, "restart never authorizes the crash's unswept payload");
            check_eq(reopened.counters().object_bytes, 0, "restart credits crash orphans only after reclaiming them");
            reopened.close();
        }
    }
    if (::geteuid() != 0) {
        TempRoot root;
        SelectedHoldGate gate;
        ConversationStore store(options_for(root.path, uint64_t(1) << 30, 0));
        const uint64_t base = store.disk_charged_bytes();
        store.set_publish_payload_gate_for_test(
            +[](void* context) { static_cast<SelectedHoldGate*>(context)->pause(); }, &gate);
        store.set_publish_fault_for_test("batch-data-sync");
        ConversationStorePublish request;
        request.request_id = request.generation = 1;
        std::string why;
        check(store.publish(std::make_shared<SavedConversation>(image), request, why) == kAccepted,
              "the failed-data fixture starts its sole writer");
        gate.await();
        check(::chmod((root.path + "/tmp").c_str(), 0500) == 0,
              "data-sweep failure uses real permission denial to retain its unrenamed files");
        gate.resume();
        ConversationStorePublishResult result;
        check(store.wait_publish(result, 5000) && result.status == ConversationStorePublishStatus::failed,
              "the data EIO settles before any content-addressed object rename");
        const uint64_t written = store.counters().bytes_written;
        check(file_inventory(root.path + "/objects").empty() &&
                  !directory_names(root.path + "/tmp").empty() && directory_names(root.path + "/manifests").empty(),
              "a failed data sweep cannot leave digest-named objects eligible for later revival");
        check_eq(store.disk_charged_bytes() - base, store.counters().orphan_bytes,
                 "every undeletable data-sweep temporary stays charged exactly once");
        check(::chmod((root.path + "/tmp").c_str(), 0700) == 0,
              "normal permissions are restored before retrying the actual source");
        store.set_publish_payload_gate_for_test(nullptr, nullptr);
        check(publish_and_wait(store, image, 2, 2, result) &&
                  result.status == ConversationStorePublishStatus::stored && result.new_bytes == written &&
                  result.new_bytes > 0,
              "retry rewrites failed-data payload instead of reviving an unproved inode");
        store.close();
        check(store.counters().orphan_bytes == 0 && !gate.timed_out,
              "confirmed cleanup refunds the failed-data temporaries once and the gate never times out");
    }
    if (had) ::setenv("STRATA_STORE_DIR_FSYNC_BATCH", saved.c_str(), 1);
    else ::unsetenv("STRATA_STORE_DIR_FSYNC_BATCH");
}

// #243: the retained-head ceiling belongs to one execution domain, never to the shared catalog.  A pair profile and a
// triple profile publish into the same root in both directions.  The pair's heads are older than every triple head, so
// a global ceiling would retire them first; the per-domain ceiling must keep each profile's completed head while each
// still retires its own oldest history at its own allowance.  A queued selected lease keeps its own head through a
// publication, both directions restore their retained head byte-identically, and startup trims by the same rule.  The
// byte quota, filesystem floor and metadata ceiling stay global.
void check_per_domain_head_limit() {
    const auto history = [](uint64_t seed, bool pinned) {
        auto image = fixture(seed, 128);
        for (auto& checkpoint : image.checkpoints) checkpoint.pinned = pinned;
        return image;
    };
    TempRoot root;
    auto pair_options = options_for(root.path, uint64_t(1) << 30, 0);
    pair_options.domain = bytes_from("domain:pair");
    pair_options.head_limit = 2;
    pair_options.publication_timing = true;
    auto triple_options = pair_options;
    triple_options.domain = bytes_from("domain:triple");
    const auto fallback = history(901, true), latest = history(903, false);
    const auto one = history(911, false), two = history(913, false), three = history(915, false);
    check(!latest.pinned(), "the latest pair history is an ordinary unpinned branch");
    ConversationStorePublishResult pair_fallback, pair_latest, triple_one, triple_two, triple_three, pair_fresh;
    uint64_t pair_epoch = 0;
    std::string why;
    {
        ConversationStore pair(pair_options);
        pair_epoch = pair.epoch();
        check(publish_and_wait(pair, fallback, 1, 1, pair_fallback) &&
                  pair_fallback.status == ConversationStorePublishStatus::stored &&
                  publish_and_wait(pair, latest, 2, 2, pair_latest) &&
                  pair_latest.status == ConversationStorePublishStatus::stored,
              "the pair profile fills its own two-head allowance");
        check_eq(pair.counters().evictions, 0, "filling one domain's allowance evicts nothing");
        pair.close();
    }
    {
        StoreReadTraceCapture trace;
        ConversationStore triple(triple_options);
        check(triple.enabled(), "the triple profile opens the same root as its own execution domain");
        check(triple.retains_head(pair_fallback.head_id) && triple.retains_head(pair_latest.head_id),
              "a clean profile switch retains both completed pair heads");
        check(publish_and_wait(triple, one, 11, 1, triple_one) &&
                  triple_one.status == ConversationStorePublishStatus::stored &&
                  publish_and_wait(triple, two, 12, 2, triple_two) &&
                  triple_two.status == ConversationStorePublishStatus::stored,
              "the triple profile fills its own allowance beside the pair's");
        const auto selected = query_of(two, 20, 2);
        triple.suspend_read_admission();
        check(triple.prefetch(selected, why),
              "a returning request selects its own retained head before materialization");
        check(publish_and_wait(triple, three, 13, 3, triple_three) &&
                  triple_three.status == ConversationStorePublishStatus::stored &&
                  triple.retains_head(pair_fallback.head_id) && triple.retains_head(pair_latest.head_id),
              "the publication past the triple allowance takes its own slot, never the older pair's");
        check(!triple.retains_head(triple_one.head_id) && triple.retains_head(triple_two.head_id),
              "the victim is this domain's own oldest history, spared for the queued selected lease");
        check_eq(triple.counters().evictions, 1, "only the one over-allowance triple head is retired");
        check_eq(triple.counters().refused_quota, 0, "per-domain head pressure is not a byte-quota refusal");
        check_eq(triple.counters().corrupt_records, 0, "per-domain head pressure is not checkpoint corruption");
        check_eq(triple.counters().heads, 4, "the catalog holds one allowance per execution domain");
        triple.publish_read_allowance(uint64_t(64) << 20);
        check(await_status(triple, 20, 2, ConversationStoreReadStatus::ready), "the queued selection materializes");
        check(triple.demand(selected, why), "the returning demand adopts its materialized prefetch");
        ConversationStoreRead read;
        check(triple.wait_ready(20, 2, read, 5000) && read.image && read.head_id == triple_two.head_id &&
                  read.tokens == 128 && images_equal(*read.image, two, why),
              "the leased triple head restores byte-identically after the publication beside it");
        read.image.reset();
        triple.release_read(20, 2);
        triple.close();
        const auto heads = trace.records("store_head ");
        const auto* retired = read_trace(heads, "retired", 13, "head_limit");
        check(retired && retired->number("head") == triple_one.head_id &&
                  retired->number("pinned") == 0 && retired->number("protected") == 0 &&
                  retired->number("replacement_head") == triple_three.head_id &&
                  retired->number("catalog_version") == triple_three.catalog_version &&
                  retired->number("applied_ns") == triple_three.applied_ns,
              "the retirement receipt names the triple domain's own victim and its real apply");
    }
    {
        ConversationStore pair(pair_options);
        check_eq(pair.epoch(), pair_epoch, "a clean domain switch preserves this domain's own epoch");
        check(pair.demand(query_of(latest, 30, 1), why), "the retained pair head is demandable after the switch");
        ConversationStoreRead read;
        check(pair.wait_ready(30, 1, read, 5000) && read.image && read.head_id == pair_latest.head_id &&
                  read.tokens == 128 && images_equal(*read.image, latest, why),
              "the pair's completed head still restores after the triple profile's head pressure");
        read.image.reset();
        pair.release_read(30, 1);
        check(publish_and_wait(pair, history(917, false), 31, 3, pair_fresh) &&
                  pair_fresh.status == ConversationStorePublishStatus::stored,
              "the returning pair publishes past its own allowance");
        check(pair.retains_head(triple_two.head_id) && pair.retains_head(triple_three.head_id),
              "the pair profile's own pressure retires none of the triple profile's completed heads");
        check(pair.retains_head(pair_fallback.head_id) && !pair.retains_head(pair_latest.head_id),
              "the pair's victim is its own unpinned history, not the pinned fallback or another domain");
        check_eq(pair.counters().evictions, 1, "one over-allowance pair publication retires one pair head");
        check_eq(pair.counters().heads, 4, "both domains keep their full allowance after the pair's pressure");
        pair.close();
    }
    {
        ConversationStore triple(triple_options);
        check(triple.demand(query_of(three, 40, 3), why), "the triple profile returns to its retained newest head");
        ConversationStoreRead read;
        check(triple.wait_ready(40, 3, read, 5000) && read.image && read.head_id == triple_three.head_id &&
                  read.tokens == 128 && images_equal(*read.image, three, why),
              "the triple's completed head still restores after the pair profile's head pressure");
        read.image.reset();
        triple.release_read(40, 3);
        triple.close();
    }
    {
        auto reduced = pair_options;
        reduced.head_limit = 1;
        StoreReadTraceCapture trace;
        ConversationStore reopened(reduced);
        check_eq(reopened.epoch(), pair_epoch, "the reduced ceiling is still a clean switch");
        check(reopened.retains_head(pair_fallback.head_id) && reopened.retains_head(triple_three.head_id),
              "startup trims each domain to its own reduced allowance");
        check(!reopened.retains_head(pair_fresh.head_id) && !reopened.retains_head(triple_two.head_id),
              "each domain loses only its own unpinned history at the reduced ceiling");
        check_eq(reopened.counters().evictions, 2, "both over-allowance domains are trimmed exactly once");
        check_eq(reopened.counters().heads, 2, "one retained head per execution domain");
        reopened.close();
        const auto heads = trace.records("store_head ");
        size_t trimmed = 0;
        for (const auto& record : heads) {
            if (record.text("event") != "retired" || record.text("reason") != "startup_head_limit") continue;
            ++trimmed;
            check(record.number("head") == pair_fresh.head_id || record.number("head") == triple_two.head_id,
                  "each startup retirement is one domain's own over-allowance head");
        }
        check_eq(trimmed, 2, "startup reports one authoritative retirement per trimmed domain");
    }
}

// The protected segment is three quarters of *one* domain's allowance.  A promoted head in a later domain must be
// demoted once that domain exceeds its own share, even while an earlier domain sits exactly at its share: protection
// enforcement cannot stop at the first domain, or a later profile's restored heads stay protected and global byte
// pressure then evicts its newer unprotected history instead of the older restored head that should have been demoted.
void check_protected_cap_per_domain() {
    const auto history = [](uint64_t seed) {
        auto image = fixture(seed, 128);
        for (auto& checkpoint : image.checkpoints) checkpoint.pinned = false;
        return image;
    };
    TempRoot root;
    auto pair_options = options_for(root.path, uint64_t(1) << 30, 0);
    pair_options.domain = bytes_from("domain:pair");
    pair_options.head_limit = 2;   // one protected head per domain
    pair_options.publication_timing = true;
    auto triple_options = pair_options;
    triple_options.domain = bytes_from("domain:triple");
    ConversationStorePublishResult pair_one, pair_two, triple_one, triple_two, triple_three;
    {
        ConversationStore pair(pair_options);
        check(publish_and_wait(pair, history(921), 1, 1, pair_one) &&
                  publish_and_wait(pair, history(923), 2, 2, pair_two),
              "the first domain stores two ordinary histories");
        pair.confirmed_restore(pair_one.head_id);   // this domain now sits exactly at its own protected share
        pair.close();
    }
    {
        StoreReadTraceCapture trace;
        ConversationStore triple(triple_options);
        check(publish_and_wait(triple, history(931), 11, 1, triple_one) &&
                  publish_and_wait(triple, history(933), 12, 2, triple_two),
              "the second domain stores two ordinary histories");
        triple.confirmed_restore(triple_one.head_id);
        triple.confirmed_restore(triple_two.head_id);   // over this domain's share: its older promotion is demoted
        check(publish_and_wait(triple, history(935), 13, 3, triple_three) &&
                  triple_three.status == ConversationStorePublishStatus::stored,
              "the over-allowance publication still obtains this domain's own head slot");
        check(!triple.retains_head(triple_one.head_id) && triple.retains_head(triple_two.head_id),
              "the demoted older promotion is the ordinary victim, so the newer promotion survives");
        check(triple.retains_head(pair_one.head_id) && triple.retains_head(pair_two.head_id),
              "the first domain's protected head and its sibling are untouched by the second domain's cap");
        check_eq(triple.counters().evictions, 1, "one committed removal serves the second domain's own pressure");
        triple.close();
        const auto heads = trace.records("store_head ");
        const auto* retired = read_trace(heads, "retired", 13, "head_limit");
        check(retired && retired->number("head") == triple_one.head_id && retired->number("protected") == 0,
              "the per-domain protected cap demoted in the later domain before the victim was chosen");
    }
}

void check_retirement_receipts() {
    for (const char* fault : {"catalog-rename", "catalog-sync", "catalog-apply"}) {
        StoreReadTraceCapture trace;
        TempRoot root;
        auto options = options_for(root.path, uint64_t(1) << 30, 0);
        options.head_limit = 1;
        options.publication_timing = true;
        ConversationStore store(options);
        auto first = fixture(821, 128);
        for (auto& checkpoint : first.checkpoints) checkpoint.pinned = false;
        ConversationStorePublishResult prior, failed, committed;
        check(publish_and_wait(store, first, 71, 1, prior), "the retirement fault fixture starts with a real head");
        store.set_publish_fault_for_test(fault);
        check(publish_and_wait(store, fixture(823, 128), 72, 2, failed) &&
                  failed.status == ConversationStorePublishStatus::failed,
              "the injected catalog boundary rejects the capacity publication");
        check_eq(store.counters().evictions, 0, "planned or uncertain removals never increment committed evictions");
        const auto refused = trace.records("store_head ");
        check(read_trace(refused, "planned", 72, "head_limit") &&
                  !read_trace(refused, "retired", 72),
              "a planned victim is not reported as an authoritative retirement on failure");
        if (std::strcmp(fault, "catalog-rename") == 0) {
            check(store.retains_head(prior.head_id), "failed rename preserves the original authority");
            check(publish_and_wait(store, fixture(823, 128), 73, 3, committed) &&
                      committed.status == ConversationStorePublishStatus::stored,
                  "the same real capacity publication succeeds once the rename fault is gone");
            check_eq(store.counters().evictions, 1, "successful retry counts one removal, not two attempts");
            const auto records = trace.records("store_head ");
            const auto* retired = read_trace(records, "retired", 73, "head_limit");
            check(retired && retired->number("head") == prior.head_id &&
                      retired->number("catalog_version") == committed.catalog_version &&
                      retired->number("plan_version") < retired->number("catalog_version") &&
                      retired->number("applied_ns") == committed.applied_ns,
                  "retirement and publication share the actual successful apply version and timestamp");
        }
        store.close();
    }
    StoreReadTraceCapture trace;
    TempRoot root;
    auto options = options_for(root.path, uint64_t(1) << 30, 0);
    options.head_limit = 2;
    ConversationStorePublishResult older, newer;
    {
        ConversationStore store(options);
        auto image = fixture(825, 128);
        for (auto& checkpoint : image.checkpoints) checkpoint.pinned = false;
        check(publish_and_wait(store, image, 81, 1, older) &&
                  publish_and_wait(store, fixture(827, 128), 82, 2, newer), "startup pressure has two durable heads");
        store.close();
    }
    options.head_limit = 1;
    ConversationStore reopened(options);
    check(reopened.enabled() && !reopened.retains_head(older.head_id) && reopened.retains_head(newer.head_id),
          "startup head pressure removes the unpinned actual victim");
    check_eq(reopened.counters().evictions, 1, "recovery also counts only its committed capacity removal");
    const auto records = trace.records("store_head ");
    const auto* retired = read_trace(records, "retired", 0, "startup_head_limit");
    check(retired && retired->number("head") == older.head_id && retired->number("applied_ns") &&
              retired->number("catalog_version"), "startup retirement has a durable-apply receipt, not silent loss");
    reopened.close();
}

void check_bounded_candidate_rejections() {
    StoreReadTraceCapture trace;
    TempRoot root;
    auto options = options_for(root.path, uint64_t(1) << 30, 0);
    options.head_limit = 16;
    ConversationStore store(options);
    for (uint64_t i = 0; i < 12; ++i) {
        auto image = fixture(841, 128);
        for (auto& token : image.live.ids) token += int32_t(i);
        for (auto& checkpoint : image.checkpoints)
            for (auto& token : checkpoint.ids) token += int32_t(i);
        ConversationStorePublishResult result;
        check(publish_and_wait(store, image, 100 + i, i + 1, result) &&
                  result.status == ConversationStorePublishStatus::stored, "rejection fanout stores real candidate heads");
    }
    auto query = query_of(fixture(841, 128), 120, 1);
    query.tokens.front() = -1;
    std::string why;
    check(store.prefetch(query, why), "a missing prefix traverses the real candidate selection");
    ConversationStoreRead read;
    check(store.wait_ready(120, 1, read, 5000) && read.status == ConversationStoreReadStatus::missing,
          "bounded diagnostics leave the real no-I/O miss unchanged");
    store.close();
    const auto records = trace.records();
    check_eq(read_trace_count(records, "candidate_rejected", 120), 8, "rejection reasons have a fixed per-selection cap");
    const auto* omitted = read_trace(records, "candidate_rejections_omitted", 120, "trace_bound");
    check(omitted && omitted->number("omitted_rejections") == 16 && omitted->text("origin") == "prefetch",
          "the receipt truthfully counts the rest of the twenty-four rejected cuts");
    check(read_trace(records, "settled", 120, "no_io") && read_trace(records, "adopt", 120) &&
              !read_trace(records, "trace_overflow", 0), "candidate fanout cannot displace ownership or settlement records");

    // Each rejection names the head and cut it refused, with the origin of the selection that refused it, and a
    // delivered read names the same head and cut it adopted. Nothing here carries prompt text, token ids or keys.
    {
        StoreReadTraceCapture reason_trace;
        TempRoot reason_root;
        auto reason_options = options_for(reason_root.path, uint64_t(1) << 30, 0);
        reason_options.head_limit = 8;
        ConversationStore scoped(reason_options);
        SavedConversation matched = fixture(861, 128), diverging = fixture(861, 128), longer = fixture(861, 512);
        // One token past the shared prefix: both 128-token histories keep the same scope and cut lengths, so only the
        // digest distinguishes them, and the 512-token history can only be refused for its length.
        diverging.live.ids[5] += 1;
        diverging.checkpoints[0].ids[5] += 1;
        ConversationStorePublishResult matched_pub, diverging_pub, longer_pub;
        check(publish_and_wait(scoped, matched, 300, 1, matched_pub) &&
                  publish_and_wait(scoped, diverging, 301, 1, diverging_pub) &&
                  publish_and_wait(scoped, longer, 302, 1, longer_pub),
              "the reason fixture stores a matching, a diverging and a longer history");
        const auto records_by_selection = [&] { return reason_trace.records(); };
        const auto refusal = [](const std::vector<StoreReadTraceRecord>& records, uint64_t request_id,
                                const char* reason, uint64_t head) -> const StoreReadTraceRecord* {
            for (const auto& record : records)
                if (record.text("event") == "candidate_rejected" && record.number("request_id") == request_id &&
                    record.text("reason") == reason && record.number("head") == head) return &record;
            return nullptr;
        };
        ConversationStoreQuery query = query_of(matched, 310, 1);
        query.tokens.resize(100);
        std::string reason_why;
        check(scoped.prefetch(query, reason_why), "the reason fixture prefetches a real candidate set");
        ConversationStoreRead delivered;
        check(scoped.wait_ready(310, 1, delivered, 5000) &&
                  delivered.status == ConversationStoreReadStatus::ready,
              "the matching checkpoint is delivered beside the refused candidates");
        check(delivered.head_id == matched_pub.head_id && delivered.tokens == 64 && !delivered.live,
              "the delivered read names its own head and checkpoint cut");
        scoped.release_read(310, 1);
        query = query_of(matched, 311, 1);
        query.tokens.resize(100);
        ConversationStoreRead demanded;
        check(scoped.demand(query, reason_why) && scoped.wait_ready(311, 1, demanded, 5000) &&
                  demanded.status == ConversationStoreReadStatus::ready &&
                  demanded.head_id == matched_pub.head_id && demanded.tokens == 64,
              "the same selection is delivered for a demand");
        scoped.release_read(311, 1);
        scoped.close();
        const auto selection = records_by_selection();
        const auto* prefetch_longer = refusal(selection, 310, "not_strict_prefix", longer_pub.head_id);
        check(prefetch_longer && prefetch_longer->number("cut") == 512 &&
                  prefetch_longer->text("origin") == "prefetch",
              "a longer candidate is refused by name, head, cut and origin");
        const auto* prefetch_diverging = refusal(selection, 310, "prefix_mismatch", diverging_pub.head_id);
        check(prefetch_diverging && prefetch_diverging->number("cut") == 64 &&
                  prefetch_diverging->text("origin") == "prefetch",
              "a diverging prefix is refused at the cut it failed, not at the request length");
        const auto* demand_diverging = refusal(selection, 311, "prefix_mismatch", diverging_pub.head_id);
        check(demand_diverging && demand_diverging->number("cut") == 64 &&
                  demand_diverging->text("origin") == "demand",
              "the same refusal is attributed to the demand that made it");
        const auto* matched_adopt = read_trace(selection, "adopt", 310, "ready");
        check(matched_adopt && matched_adopt->number("head") == matched_pub.head_id &&
                  matched_adopt->number("cut") == 64 && matched_adopt->text("status") == "ready",
              "the delivered attempt carries the same head and cut as the read it produced");
        const auto scope_digest = sha256_hex(conversation_store_detail::sha256(
            matched.scope.value.data(), matched.scope.value.size()));
        check(reason_trace.contents().find(reason_root.path) == std::string::npos &&
                  reason_trace.contents().find(scope_digest) == std::string::npos,
              "refusal diagnostics carry neither the store path nor the authorized scope");
    }
}

// The dimensions are intentionally small; these are storage/authority proofs, not inference proofs.
SavedConversation certified_fixture(bool shared, bool split = false) {
    auto image = split ? split_fixture(991, 64, 2) : fixture(991, 64, 1);
    image.geometry[0] = 8;
    image.geometry[14] = 2;
    for (auto& stage : image.stage_images) stage.geometry = image.geometry;
    auto& owner = split ? image.stage_images.back() : image;
    if (split) owner.kv.push_back(owner.kv.back());
    owner.draft_begin = 3;
    owner.draft_cells = shared ? 63 : 64;
    owner.draft_next_token = shared ? -1 : 731;
    if (shared) owner.draft_boundary_residual.resize(64, 0x5a);
    if (shared) {
        image.scope = ConversationScope{bytes_from("certified-shared-prefix")};
        for (auto& stage : image.stage_images) stage.scope = image.scope;
    }
    return image;
}

std::vector<uint8_t> fixture_file_bytes(const std::string& path) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) { check(false, "the fixture authority can be read"); return {}; }
    struct stat info {};
    if (::fstat(fd, &info) != 0 || info.st_size < 0) { ::close(fd); return {}; }
    std::vector<uint8_t> bytes(size_t(info.st_size), 0);
    size_t at = 0;
    while (at < bytes.size()) {
        const ssize_t n = ::read(fd, bytes.data() + at, bytes.size() - at);
        if (n <= 0) break;
        at += size_t(n);
    }
    ::close(fd);
    check_eq(at, bytes.size(), "the fixture reads every authority byte");
    return bytes;
}

bool fixture_replace_file(const std::string& path, const std::vector<uint8_t>& bytes) {
    const int fd = ::open(path.c_str(), O_WRONLY | O_TRUNC | O_CLOEXEC);
    if (fd < 0) return false;
    size_t at = 0;
    while (at < bytes.size()) {
        const ssize_t n = ::write(fd, bytes.data() + at, bytes.size() - at);
        if (n <= 0) break;
        at += size_t(n);
    }
    const bool flushed = ::fsync(fd) == 0;
    const bool closed = ::close(fd) == 0;
    return at == bytes.size() && flushed && closed;
}

std::map<std::string, std::string> fixture_contents(const std::string& root) {
    std::map<std::string, std::string> out;
    for (const auto& path : file_inventory(root)) out.emplace(path, file_digest(root + path));
    return out;
}

conversation_store_detail::ManifestMeta fixture_manifest(
    const conversation_store_detail::ImageLayout& layout) {
    conversation_store_detail::ManifestMeta meta;
    meta.stages = layout.stages;
    meta.containers = layout.containers;
    meta.payload_bytes = layout.payload_bytes;
    meta.cuts = layout.cuts;
    std::map<Digest, uint64_t> objects;
    for (const auto& source : layout.streams) {
        conversation_store_detail::StreamRecord stream;
        stream.id = source.info.id;
        stream.bytes = source.info.bytes;
        if (stream.bytes) {
            std::vector<uint8_t> bytes(size_t(stream.bytes));
            std::string why;
            check(source.read(0, bytes.data(), bytes.size(), why), "the real codec fixture stream reads");
            const auto object = sha256(bytes.data(), bytes.size());
            stream.chunks.push_back({0, uint32_t(bytes.size()), object});
            objects.emplace(object, bytes.size());
        }
        meta.streams.push_back(std::move(stream));
    }
    meta.objects.assign(objects.begin(), objects.end());
    return meta;
}

// Exercise the real asynchronous writer and reopened reader, including the object
// identities that pure snapshot/manifest roundtrips do not visit.
void check_native_draft_store() {
    using namespace conversation_store_detail;
    for (int variant = 0; variant < 3; ++variant) {
        const bool native = variant != 2;
        const bool split = variant == 1;
        auto image = native ? (split ? split_fixture(997, 8, 3) : fixture(997, 8))
                            : certified_fixture(true);
        auto& owner = split ? image.stage_images.back() : image;
        size_t main = owner.kv.size() - (native ? 0 : 1);
        if (native) {
            image.checkpoints.clear();
            for (auto& stage : image.stage_images) stage.checkpoints.clear();
            owner.draft_identity.kind = DraftKind::dflash2;
            owner.draft_identity.format = 1;
            owner.draft_identity.weights = bytes_from("native weights");
            owner.draft_identity.configuration = bytes_from("native configuration");
            owner.draft_identity.borrowed_target = bytes_from("borrowed target");
            owner.draft_identity.geometry = dflash2::native_geometry;
            owner.draft_origin = owner.draft_begin = 0;
            owner.draft_cells = int64_t(owner.live.ids.size());
            owner.draft_next_token = -1;
            for (int layer = 0; layer < 5; ++layer) {
                ConversationKv kv;
                kv.format = 0; kv.cells = owner.draft_cells;
                kv.heads = 8; kv.head_dim = 128; kv.page_size = 1;
                kv.pooled_rows = kv.idx_dim = 0;
                const size_t bytes = size_t(kv.cells) * 8 * 128 * sizeof(uint16_t);
                kv.k.resize(bytes, uint8_t(0x30 + layer));
                kv.v.resize(bytes, uint8_t(0x50 + layer));
                owner.kv.push_back(std::move(kv));
            }
        }
        std::string why;
        ImageLayout layout;
        if (!image_layout(image, Digest{}, unconstrained_limits(), layout, why)) {
            check(false, "native/MTP store fixture has a valid certified image layout");
            continue;
        }
        const uint32_t stage = split ? 2 : 0;
        const size_t layers = native ? 5 : 1;
        uint64_t draft_bytes = owner.draft_boundary_residual.size();
        for (size_t layer = 0; layer < layers; ++layer) {
            const auto& kv = owner.kv[main + layer];
            draft_bytes += kv.k.size() + kv.v.size() + kv.k_scale.size() + kv.v_scale.size() + kv.pooled.size();
            for (auto kind : {ConversationStoreStreamKind::kv_k, ConversationStoreStreamKind::kv_v}) {
                const ConversationStoreStreamId id{stage, uint16_t(kind), 0, uint64_t(main + layer)};
                std::pair<Digest, uint64_t> identity;
                check(identity_of(image, id, identity, why), "every certified draft layer has a physical object identity");
                check_eq(identity.second, (uint64_t{1} << 63) | layer,
                         "native layers are distinct; the historical MTP layer-zero ordinal is unchanged");
                check(stream_read_component(id.kind, identity.second) == ConversationStoreReadComponent::draft,
                      "all five native layer streams are charged to draft progress");
            }
        }
        ConversationStoreReadProgress progress;
        const auto meta = fixture_manifest(layout);
        check(restore_progress_totals(meta, manifest_encoded_size(meta), progress, why),
              "native/MTP restore computes bounded component totals");
        check_eq(progress.components[size_t(ConversationStoreReadComponent::draft)].total, draft_bytes,
                 "draft progress includes all layers and preserves MTP residual accounting");
        if (native) {
            for (int delta : {-1, 1}) {
                auto bad = image;
                auto& bad_owner = split ? bad.stage_images.back() : bad;
                if (delta < 0) bad_owner.kv.pop_back();
                else bad_owner.kv.push_back(bad_owner.kv.back());
                ImageLayout refused;
                check(!image_layout(bad, Digest{}, unconstrained_limits(), refused, why),
                      "native four/six-layer captures remain refused");
            }
        }
        TempRoot root;
        const auto options = options_for(root.path, uint64_t(1) << 30, 0);
        uint64_t head = 0;
        {
            ConversationStore store(options);
            check(store.enabled(), "native/MTP writer opens the owned temporary store");
            ConversationStorePublishResult published;
            const bool settled = publish_and_wait(store, image, 1, 1, published);
            check(settled && published.status == ConversationStorePublishStatus::stored && published.head_id != 0,
                  "actual native/MTP writer publishes a durable head instead of refusing K/V object layout");
            if (!settled || published.status != ConversationStorePublishStatus::stored) {
                store.close();
                continue;
            }
            head = published.head_id;
            store.close();
        }
        {
            ConversationStore store(options);
            check(store.enabled(), "native/MTP reader reopens the durable store");
            const auto query = query_of(image, 2, 1);
            const auto matches = store.matches(query.selector, query.tokens, query.images);
            check(!matches.empty() && matches.front().head_id == head,
                  "reopened native/MTP lookup identifies the actual published head");
            store.publish_read_allowance(uint64_t(64) << 20);
            const bool admitted = store.demand(query, why);
            check(admitted, "the actual native/MTP demand is admitted");
            ConversationStoreRead read;
            const bool ready = admitted && store.wait_ready(2, 1, read, 5000);
            check(ready && read.status == ConversationStoreReadStatus::ready && read.image &&
                      images_equal(*read.image, image, why),
                  "reopened reader verifies object hashes and restores every native layer and identity exactly");
            read.image.reset();
            if (admitted) store.release_read(2, 1);
            check_eq(store.ram_lease_bytes(), 0, "native/MTP read leases retire after release");
            store.close();
        }
    }
}

void check_draft_certification_codec() {
    using namespace conversation_store_detail;
    const auto limits = unconstrained_limits();
    std::string why;
    for (int variant = 0; variant < 4; ++variant) {
        const auto image = variant == 0 ? fixture(991, 64, 1) :
                           certified_fixture(variant >= 2, variant == 3);
        ImageLayout layout;
        ImageShape shape;
        check(image_shape(image, limits, shape, why) && image_layout(image, Digest{}, limits, layout, why),
              "legacy, private, shared and split certified images have real layouts");
        check_eq(shape.payload_bytes, layout.payload_bytes, "allocation-free admission counts every residual byte");
        check_eq(shape.streams, layout.streams.size(), "allocation-free admission counts the explicit residual stream");
        auto meta = fixture_manifest(layout);
        const auto encoded = manifest_encode(meta);
        check_eq(encoded.size(), manifest_encoded_size(meta), "MAN1 and MAN2 exact manifest charges agree");
        check(encoded.size() >= 8 && encoded[7] == (variant == 0 ? '1' : '2'),
              "uncertified captures retain MAN1; certification uses explicit MAN2");
        ManifestMeta decoded;
        check(manifest_decode(encoded.data(), encoded.size(), limits, decoded, why), "certification metadata decodes");
        SavedConversation restored;
        std::vector<RestoreTarget> targets;
        check(restore_prepare(decoded, limits, restored, targets, why), "certified restore binds real final buffers");
        if (targets.size() != layout.streams.size()) { check(false, "every restore target exists"); continue; }
        for (size_t i = 0; i < targets.size(); ++i) {
            std::vector<uint8_t> bytes(size_t(layout.streams[i].info.bytes));
            if (!bytes.empty())
                check(layout.streams[i].read(0, bytes.data(), bytes.size(), why) &&
                          targets[i].write(0, bytes.data(), bytes.size(), why),
                      "certified codec roundtrip transfers actual payload bytes");
        }
        restore_finish(restored);
        check(images_equal(image, restored, why), "the exact interval, private token and residual bytes survive");
        if (variant == 0)
            check(restored.draft_begin == -1 && restored.draft_cells == -1 && restored.draft_next_token == -1,
                  "legacy draft allocation never becomes inferred certification");
        if (variant != 2) continue;
        ConversationStoreReadProgress progress;
        check(restore_progress_totals(decoded, encoded.size(), progress, why),
              "certified progress includes the validated residual stream");
        const auto& draft = image.kv.back();
        const uint64_t draft_bytes = image.draft_boundary_residual.size() + draft.k.size() + draft.v.size() +
                                     draft.k_scale.size() + draft.v_scale.size() + draft.pooled.size();
        check_eq(progress.components[size_t(ConversationStoreReadComponent::draft)].total, draft_bytes,
                 "the boundary is charged to draft, not main K/V or metadata");
        check(!conversation_store_kv_kind(uint16_t(ConversationStoreStreamKind::draft_boundary_residual)),
              "the residual identity cannot be misclassified as a K/V donor kind");
        for (int damage = 0; damage < 12; ++damage) {
            auto bad = meta;
            auto& c = bad.containers.front();
            switch (damage) {
                case 0: c.draft_begin = -1; break;
                case 1: c.draft_cells = -2; break;
                case 2: c.draft_begin = c.draft_cells + 1; break;
                case 3: c.draft_cells = 65; break;
                case 4: c.draft_next_token = 12; break;
                case 5: c.draft_cells = 64; c.draft_boundary_bytes = 0; break;
                case 6: c.kv.back().cells = 62; break;
                case 7: ++c.draft_boundary_bytes; break;
                case 8: c.geometry[0] = INT64_MAX; c.geometry[14] = INT64_MAX; break;
                case 9: c.kv.pop_back(); break;
                case 10: c.layer_hi = 4; break;
                default: bad.streams.push_back(bad.streams[7]); break;
            }
            const auto malformed = manifest_encode(bad);
            check(!manifest_decode(malformed.data(), malformed.size(), limits, decoded, why),
                  "checksummed malformed certification is rejected before payload transfer");
            SavedConversation untouched;
            untouched.live.ids.push_back(777);
            std::vector<RestoreTarget> unbound(1);
            unbound[0].bytes = 123;
            check(!restore_prepare(bad, limits, untouched, unbound, why) &&
                      untouched.live.ids == std::vector<int32_t>{777} && unbound.size() == 1 && unbound[0].bytes == 123,
                  "malformed certification cannot replace the destination or bind transfer targets");
        }
    }
    auto boundary = certified_fixture(true);
    boundary.geometry[0] = 512;
    boundary.draft_boundary_residual.resize(4096, 0x5a);
    ImageShape counted;
    check(image_shape(boundary, limits, counted, why), "a boundary larger than every other stream is representable");
    auto bounded = limits;
    bounded.max_image_bytes = counted.payload_bytes;
    bounded.max_state_bytes = bounded.max_stream_bytes = 4096;
    check(image_shape(boundary, bounded, counted, why), "the exact payload and residual budgets admit");
    --bounded.max_image_bytes;
    check(!image_shape(boundary, bounded, counted, why), "one payload byte below the boundary refuses");
    bounded = limits;
    bounded.max_stream_bytes = 4095;
    check(!image_shape(boundary, bounded, counted, why), "the residual spends the stream ceiling");
    bounded = limits;
    bounded.max_state_bytes = 4095;
    check(!image_shape(boundary, bounded, counted, why), "the residual spends the state ceiling");
    auto invalid = boundary;
    invalid.draft_begin = -1;
    check(!image_shape(invalid, limits, counted, why), "malformed certification refuses allocation-free admission");
    ImageLayout boundary_layout;
    check(image_layout(boundary, Digest{}, limits, boundary_layout, why), "the bounded image has a canonical layout");
    const auto boundary_meta = fixture_manifest(boundary_layout);
    const auto encoded = manifest_encode(boundary_meta);
    auto manifest_limit = limits;
    manifest_limit.max_manifest_bytes = encoded.size();
    ManifestMeta decoded;
    check(manifest_decode(encoded.data(), encoded.size(), manifest_limit, decoded, why),
          "the exact MAN2 metadata ceiling admits");
    --manifest_limit.max_manifest_bytes;
    check(!manifest_decode(encoded.data(), encoded.size(), manifest_limit, decoded, why),
          "one manifest byte below refuses before payload binding");
    check(manifest_decode(encoded.data(), encoded.size(), retention_limits(bounded), decoded, why) &&
              !manifest_within_limits(decoded, bounded, why),
          "a residual too large for this profile is retained as capacity refusal, not corruption");
    const auto fresh = [](ConversationStore&) {
        ConversationStorePublish request;
        request.request_id = request.generation = 1;
        return request;
    };
    const auto options = options_for("", uint64_t(1) << 30, 0);
    const uint64_t floor = publication_admission_floor(options, boundary, fresh, uint64_t(16) << 20);
    check(publication_is_admitted(options, boundary, fresh, floor) &&
              !publication_is_admitted(options, boundary, fresh, floor - 1),
          "real certified publication agrees at its one-byte RAM admission boundary");
    auto no_boundary = boundary;
    no_boundary.draft_boundary_residual = ConversationBuffer{};
    const uint64_t smaller = publication_admission_floor(options, no_boundary, fresh, uint64_t(16) << 20);
    check(floor > smaller, "residual payload, stream, objects and manifest/workspace are not free at admission");
}

void check_draft_certification_store() {
    for (bool split : {false, true}) {
        TempRoot root;
        const auto options = options_for(root.path, uint64_t(1) << 30, 0);
        auto legacy = fixture(993, 64, 1);
        auto image = certified_fixture(true, split);
        auto changed = image;
        std::array<uint8_t, 32> secret{};
        uint64_t charged = 0;
        {
            ConversationStore store(options);
            secret = store.issuer_key();
            ConversationStorePublishResult old, certified;
            check(publish_and_wait(store, legacy, 1, 1, old), "legacy history is durably retained before activation");
            check_eq(fixture_file_bytes(root.path + "/issuer.key").size(), 32,
                     "default-off capture does not activate a reader floor");
            const uint64_t prior_epoch = store.epoch();
            const auto prior_catalog_bytes = fixture_file_bytes(root.path + "/catalog");
            conversation_store_detail::Catalog prior_catalog;
            std::string identity_why;
            check(conversation_store_detail::catalog_decode(prior_catalog_bytes.data(), prior_catalog_bytes.size(),
                                                             prior_catalog, identity_why),
                  "the pre-activation catalog authority decodes");
            check(publish_and_wait(store, image, 2, 2, certified) &&
                      certified.status == ConversationStorePublishStatus::stored,
                  "certified publication establishes the existing authority floor and commits MAN2");
            const auto envelope = fixture_file_bytes(root.path + "/issuer.key");
            check_eq(envelope.size(), 64, "activation versions the existing issuer authority");
            check(store.issuer_key() == secret && envelope.size() == 64 &&
                      std::equal(secret.begin(), secret.end(), envelope.begin() + 16),
                  "the original secret and every frontend authorization identity are preserved");
            check_eq(store.counters().heads, 2, "activation preserves the legacy history beside the certified one");
            const auto activated_catalog_bytes = fixture_file_bytes(root.path + "/catalog");
            conversation_store_detail::Catalog activated_catalog;
            check(conversation_store_detail::catalog_decode(activated_catalog_bytes.data(), activated_catalog_bytes.size(),
                                                             activated_catalog, identity_why) &&
                      activated_catalog.issuer_key_digest == prior_catalog.issuer_key_digest &&
                      activated_catalog.issuer_key_digest == sha256(secret.data(), secret.size()) &&
                      store.epoch() == prior_epoch,
                  "activation preserves catalog issuer digest and the domain epoch");
            for (const auto& head : activated_catalog.heads)
                if (head.head_id == old.head_id)
                    check(head.generation == 1 && head.epoch == prior_epoch,
                          "activation preserves the existing history's generation and epoch");
            // A caller cannot turn a residual into a K/V donor interval, even with a valid donor head.
            auto& owner = split ? changed.stage_images.back() : changed;
            owner.draft_boundary_residual.resize(64, 0x6b);
            owner.draft_boundary_residual.visit(0, 64, [](uint8_t* p, size_t n, size_t) {
                std::memset(p, 0x6b, n); return true;
            });
            ConversationStorePublish request;
            request.request_id = 3;
            request.generation = 3;
            ConversationStoreReuse reuse;
            reuse.head_id = certified.head_id;
            reuse.stream = {uint32_t(split ? 1 : 0), uint16_t(ConversationStoreStreamKind::draft_boundary_residual), 0, 0};
            reuse.bytes = 64;
            request.reuse.push_back(reuse);
            std::string why;
            check(store.publish(std::make_shared<SavedConversation>(changed), request, why) == kAccepted,
                  "the residual donor challenge is admitted");
            ConversationStorePublishResult result;
            check(store.wait_publish(result, 5000) && result.status == ConversationStorePublishStatus::stored &&
                      result.reused_bytes == 0, "residual bytes cannot be adopted from an explicit donor interval");
            store.publish_read_allowance(1);
            const uint64_t read_before = store.counters().bytes_read;
            check(!store.demand(query_of(changed, 40, 1), why) && store.counters().bytes_read == read_before,
                  "an insufficient readpeak refuses before manifest or residual transfer");
            store.publish_read_allowance(uint64_t(64) << 20);
            check(store.demand(query_of(changed, 41, 1), why), "the charged certified demand admits");
            ConversationStoreRead read;
            check(store.wait_ready(41, 1, read, 5000) && read.status == ConversationStoreReadStatus::ready &&
                      read.image && images_equal(*read.image, changed, why),
                  "the destination reads its own changed residual, never the donor's row");
            check(!read.image || store.ram_lease_bytes() >= read.image->bytes(),
                  "the handed-out certified image and residual remain fully charged until release");
            read.image.reset();
            store.release_read(41, 1);
            store.close();
            charged = store.disk_charged_bytes();
        }
        const auto floor = fixture_file_bytes(root.path + "/issuer.key");
        {
            ConversationStore store(options); // no producer feature flag is needed to read or roll back safely
            check(store.enabled() && store.issuer_key() == secret, "a compatible feature-off reader reopens the pool");
            check_eq(store.disk_charged_bytes(), charged + 8192, "reopen charges only its real run-intent marker");
            check(!store.matches(selector_of(legacy), query_of(legacy, 50, 1).tokens, {}).empty(),
                  "the old history remains matchable after activation and reopen");
            uint64_t request_id = 51;
            for (const SavedConversation* expected : {&legacy, &changed}) {
                std::string why;
                check(store.demand(query_of(*expected, request_id, 1), why),
                      "a compatible feature-off reader admits both legacy and certified histories");
                ConversationStoreRead read;
                check(store.wait_ready(request_id, 1, read, 5000) &&
                          read.status == ConversationStoreReadStatus::ready && read.image &&
                          images_equal(*read.image, *expected, why),
                      "both legacy and exact certified residual bytes roundtrip after reopen");
                read.image.reset();
                store.release_read(request_id++, 1);
            }
            ConversationStorePublishResult result;
            check(publish_and_wait(store, legacy, 4, 4, result), "feature-off legacy publication still works after activation");
            store.close();
        }
        check(fixture_file_bytes(root.path + "/issuer.key") == floor,
              "feature-off publication and clean close never downgrade the floor");
        for (int malformed = 0; malformed < 5; ++malformed) {
            auto bytes = floor;
            if (malformed == 0) bytes[12] = 3; // a validly checksummed future minimum reader
            if (malformed == 1) bytes[8] = 3;  // a future envelope version
            if (malformed < 2) {
                const auto digest = sha256(bytes.data(), 48);
                std::copy_n(digest.begin(), 16, bytes.begin() + 48);
            } else if (malformed == 2) bytes.back() ^= 1;
            else if (malformed == 3) bytes.resize(63);
            else bytes.resize(65);
            check(fixture_replace_file(root.path + "/issuer.key", bytes), "the envelope refusal fixture is durable");
            const auto before = fixture_contents(root.path);
            { ConversationStore store(options); check(!store.enabled(), "unsupported or malformed authority refuses open"); }
            check(fixture_contents(root.path) == before, "authority refusal preserves every history and file byte");
            check(fixture_replace_file(root.path + "/issuer.key", floor), "the exact recognized authority is restored");
        }
        {
            // The catalog is the root's authority: its schema cannot be quarantined record by record, so an
            // unrecognized tag refuses the open with nothing read, rewritten or deleted.
            const auto original = fixture_file_bytes(root.path + "/catalog");
            auto future = original;
            future[7] = '9';
            check(fixture_replace_file(root.path + "/catalog", future), "the unsupported catalog fixture is written");
            const auto before = fixture_contents(root.path);
            { ConversationStore store(options); check(!store.enabled(), "an unknown catalog schema refuses before recovery"); }
            check(fixture_contents(root.path) == before, "catalog refusal cannot reconcile or delete histories");
            check(fixture_replace_file(root.path + "/catalog", original), "the original catalog bytes are restored");
        }
        {
            // A manifest's own tag is that record's defect, not the root's: it is quarantined exactly as before and
            // every other history stays authorized. A record written by a newer reader never reaches this point,
            // because the issuer envelope refuses such a root first.
            TempRoot damaged;
            const uint64_t damaged_quota = uint64_t(1) << 30;
            const SavedConversation survivor = fixture(961, 64), victim = fixture(963, 64);
            std::string victim_manifest;
            {
                ConversationStore store(options_for(damaged.path, damaged_quota, 0));
                check(store.enabled(), "the quarantine fixture root opens");
                const auto before = directory_names(damaged.path + "/manifests");
                ConversationStorePublishResult stored;
                check(publish_and_wait(store, survivor, 1, 1, stored), "the surviving history is stored");
                check(publish_and_wait(store, victim, 2, 1, stored), "the history that will be damaged is stored");
                const auto after = directory_names(damaged.path + "/manifests");
                for (const auto& name : after)
                    if (!inventory_contains(before, name)) victim_manifest = name;
                check(!victim_manifest.empty(), "the damaged publication wrote its own manifest");
                store.close();
            }
            const std::string path = damaged.path + "/manifests/" + victim_manifest;
            const auto original = fixture_file_bytes(path);
            auto future = original;
            future[7] = '9';
            check(fixture_replace_file(path, future), "the unknown manifest tag is written");
            {
                ConversationStore store(options_for(damaged.path, damaged_quota, 0));
                check(store.enabled(), "an unknown manifest tag does not disable the root");
                check_eq(store.counters().corrupt_records, 1, "exactly the record with the unknown tag is corrupt");
                check_eq(store.counters().heads, 1, "only the damaged record is dropped");
                std::string why;
                check(store.demand(query_of(survivor, 11, 1), why), "the surviving history is still demandable");
                ConversationStoreRead read;
                check(store.wait_ready(11, 1, read, 5000) && read.status == ConversationStoreReadStatus::ready &&
                      read.image && images_equal(*read.image, survivor, why),
                      "the surviving history restores byte-identically beside a quarantined record");
                store.release_read(11, 1);
                store.close();
            }
            // The quarantined record's manifest and objects are reclaimed with it, so nothing is left to restore.
        }
    }
}

void check_draft_reader_floor_failures() {
    const char* prior_batch = ::getenv("STRATA_STORE_DIR_FSYNC_BATCH");
    const bool had_batch = prior_batch != nullptr;
    const std::string saved_batch = prior_batch ? prior_batch : "";
    for (bool batching : {false, true}) {
        ::setenv("STRATA_STORE_DIR_FSYNC_BATCH", batching ? "1" : "0", 1);
    for (const char* step : {"reader-floor-write", "reader-floor-rename", "reader-floor-sync", "reader-floor-allocate",
                             "reader-floor-tmp-sync",
                             "objects", "manifest", "catalog", "catalog-rename", "catalog-sync", "catalog-apply"}) {
        TempRoot root;
        auto options = options_for(root.path, uint64_t(1) << 30, 0);
        auto foreign = options;
        foreign.domain = bytes_from("other-profile-retained-domain");
        const auto history = fixture(995, 64);
        {
            ConversationStore store(foreign);
            ConversationStorePublishResult result;
            check(publish_and_wait(store, history, 1, 1, result), "the foreign history predates the floor transaction");
            store.close();
        }
        const auto secret = fixture_file_bytes(root.path + "/issuer.key");
        const auto old_files = fixture_contents(root.path);
        const bool before_rename = std::strcmp(step, "reader-floor-write") == 0 ||
                                   std::strcmp(step, "reader-floor-rename") == 0;
        const bool uncertain = std::strcmp(step, "reader-floor-sync") == 0 ||
                               std::strcmp(step, "reader-floor-allocate") == 0 ||
                               std::strcmp(step, "reader-floor-tmp-sync") == 0 ||
                               std::strcmp(step, "catalog-sync") == 0 || std::strcmp(step, "catalog-apply") == 0;
        {
            ConversationStore store(options);
            const uint64_t charged = store.disk_charged_bytes();
            store.set_publish_fault_for_test(step);
            ConversationStorePublishResult result;
            check(publish_and_wait(store, certified_fixture(true), 2, 2, result) &&
                      result.status != ConversationStorePublishStatus::stored &&
                      result.status != ConversationStorePublishStatus::replaced,
                  "the injected transition/publication failure never claims a committed head");
            check(store.healthy() != uncertain, "only actual authority uncertainty latches the store fault");
            const auto after = fixture_file_bytes(root.path + "/issuer.key");
            check(after.size() == (before_rename ? 32 : 64),
                  "no new-format write precedes the floor; every later failure keeps the sticky floor");
            if (before_rename)
                check(after == secret && store.disk_charged_bytes() == charged,
                      "pre-rename failure preserves the key and refunds only confirmed absent scratch");
            if (uncertain) {
                const auto retained = fixture_contents(root.path);
                check(store.disk_charged_bytes() >= charged, "uncertain authority credits no replaced inode");
                std::string why;
                ConversationStorePublish request;
                request.request_id = 3;
                request.generation = 3;
                check(store.publish(std::make_shared<SavedConversation>(history), request, why) == kRefused,
                      "authority uncertainty also refuses subsequent publication work");
                store.close();
                check(fixture_contents(root.path) == retained, "faulted close neither downgrades nor reclaims");
            } else store.close();
        }
        {
            ConversationStore store(options);
            check(store.enabled(), "reopen recovers the real authority after every transition failure");
            check(store.matches(selector_of(history), query_of(history, 9, 1).tokens, {}).empty(),
                  "foreign-domain histories are not served by the current domain");
            store.close();
        }
        {
            ConversationStore store(foreign);
            check(store.enabled() && !store.matches(selector_of(history), query_of(history, 10, 1).tokens, {}).empty(),
                  "the original profile still reads its preserved history after failure/reopen");
            store.close();
        }
        const auto after = fixture_contents(root.path);
        for (const auto& entry : old_files)
            if (entry.first != "/catalog" && entry.first != "/issuer.key")
                check(after.count(entry.first) && after.at(entry.first) == entry.second,
                      "every pre-existing history object and manifest survives the floor failure");
    }
    }
    for (bool floor_limited : {false, true}) {
        TempRoot root;
        auto options = options_for(root.path, uint64_t(1) << 30, 0);
        auto legacy = certified_fixture(false);
        legacy.draft_begin = legacy.draft_cells = legacy.draft_next_token = -1;
        for (auto& checkpoint : legacy.checkpoints) checkpoint.pinned = false;
        uint64_t baseline = 0;
        {
            ConversationStore store(options);
            ConversationStorePublishResult result;
            check(publish_and_wait(store, legacy, 1, 1, result), "the non-evicting floor admission has a real history");
            store.close();
            baseline = store.disk_charged_bytes();
        }
        if (floor_limited) options.min_free_bytes = uint64_t(1) << 62;
        else options.quota_bytes = baseline + 8192; // only the mandatory run-intent fits; no replacement headroom
        ConversationStore store(options);
        check(store.enabled(), "the retained history opens before an unaffordable activation");
        const auto before = fixture_contents(root.path);
        const uint64_t charge = store.disk_charged_bytes();
        ConversationStorePublishResult result;
        check(publish_and_wait(store, certified_fixture(false), 2, 2, result) &&
                  result.status != ConversationStorePublishStatus::stored,
              "insufficient replacement headroom refuses activation even with evictable histories");
        check(store.disk_charged_bytes() == charge && store.counters().evictions == 0 &&
                  fixture_contents(root.path) == before && fixture_file_bytes(root.path + "/issuer.key").size() == 32,
              "quota/floor refusal neither evicts nor rewrites any authority or history file");
        store.close();
    }
    for (bool batching : {false, true}) {
        ::setenv("STRATA_STORE_DIR_FSYNC_BATCH", batching ? "1" : "0", 1);
        TempRoot root;
        const auto options = options_for(root.path, uint64_t(1) << 30, 0);
        auto legacy = certified_fixture(false);
        legacy.draft_begin = legacy.draft_cells = legacy.draft_next_token = -1;
        ConversationStore store(options);
        ConversationStorePublishResult result;
        check(publish_and_wait(store, legacy, 1, 1, result), "the all-deduplicated activation has real donor objects");
        const auto objects = fixture_contents(root.path + "/objects");
        const auto manifests = fixture_contents(root.path + "/manifests");
        const uint64_t written = store.counters().bytes_written;
        store.set_publish_fault_for_test("reader-floor-sync");
        check(publish_and_wait(store, certified_fixture(false), 2, 2, result) &&
                  result.status != ConversationStorePublishStatus::stored && !store.healthy(),
              "an all-deduplicated publication still waits for the mandatory floor barrier in both batching modes");
        check(store.counters().bytes_written == written &&
                  fixture_contents(root.path + "/objects") == objects &&
                  fixture_contents(root.path + "/manifests") == manifests,
              "failed floor durability stages neither objects nor a MAN2 manifest, even with zero dirty shards");
        store.close();
    }
    if (had_batch) ::setenv("STRATA_STORE_DIR_FSYNC_BATCH", saved_batch.c_str(), 1);
    else ::unsetenv("STRATA_STORE_DIR_FSYNC_BATCH");
}

} // namespace