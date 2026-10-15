// Deterministic CPU tests for the durable conversation store.  No GPU, no model, no network: a temporary root under
// the system temp directory, fixtures built by hand, and checks that read the module's own observable contract -
// matches, statuses, leases, epochs and byte counts.  Every check is a consumer-visible behaviour, not an internal.
#include "strata/core/conversation_store.hpp"

#include "conversation_store_format.hpp"
#include "conversation_store_sha256.hpp"

#include <array>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <functional>
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
// probe can drift with another one's commits.
bool publication_is_admitted(const ConversationStoreOptions& shape, const SavedConversation& image,
                             const std::function<ConversationStorePublish(ConversationStore&)>& attempt,
                             uint64_t allowance) {
    TempRoot root;
    ConversationStoreOptions options = shape;
    options.root = root.path;
    ConversationStore store(options);
    if (!store.enabled()) return false;
    store.publish_read_allowance(allowance);
    const ConversationStorePublish request = attempt(store);
    auto shared = std::make_shared<SavedConversation>(image);
    std::string why;
    const bool admitted = store.publish(shared, request, why);
    if (admitted) {
        ConversationStorePublishResult result;
        store.wait_publish(result, 5000);
    }
    store.close();
    return admitted;
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
void check_multi_stage_restore();
void check_native_split_restore();
void check_request_specific();
void check_incoming_lifetime();
void check_publication_rules();
void check_pinned_branch();
void check_reuse();
void check_leases_and_budget();
void check_tight_ram();
void check_cancel_and_release();
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
void check_catalog_headroom();
void check_commit_outcomes();
void check_bootstrap_and_intent();

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
    check_multi_stage_restore();
    check_native_split_restore();
    check_request_specific();
    check_incoming_lifetime();
    check_publication_rules();
    check_pinned_branch();
    check_reuse();
    check_leases_and_budget();
    check_tight_ram();
    check_cancel_and_release();
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
    check_catalog_headroom();
    check_commit_outcomes();
    check_bootstrap_and_intent();
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
    if (!store.publish(shared, request, why)) {
        std::fprintf(stderr, "Publication admission: %s\n", why.c_str());
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
    check(store.publish(newer, covered, why), "a covered publication of the same lineage is admitted");
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
    check(store.publish(first_image, request, why), "the first of two back-to-back publications is admitted");
    auto second_image = std::make_shared<SavedConversation>(other);
    ConversationStorePublish busy;
    busy.request_id = 11;
    busy.generation = 1;
    check(!store.publish(second_image, busy, why), "a second image is refused while the writer is busy");
    check(!why.empty(), "the refusal carries a reason");
    check(store.counters().refusals_busy > 0, "the refusal is counted");
    ConversationStorePublishResult drain;
    while (store.wait_publish(drain, 2000)) {
    }
    store.close();
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
    check(store.publish(image, request, why), "a publication carrying proven intervals is admitted");
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
    check(store.publish(second, unaligned, why), "an unusable interval still admits the publication");
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
    check(!store.publish(std::make_shared<SavedConversation>(image), blocked, why),
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
    store.cancel(400, 1);
    ConversationStoreRead cancelled;
    check(!store.wait_ready(400, 1, cancelled, 200), "a cancelled request is not delivered");
    check(store.ram_held_bytes() <= before, "a cancelled request does not leave a charge behind");
    store.cancel(4242, 99);  // unknown identifiers are safe
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
    ConversationStore store(options_for(root.path, uint64_t(1) << 30, 0));
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
    check(store.publish(dropped, request, why), "a covered descendant is admitted");
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
    check(store.publish(kept, cover, why), "a pin-preserving cover is admitted");
    ConversationStorePublishResult replaced;
    check(store.wait_publish(replaced, 5000), "the pin-preserving cover settles");
    check(replaced.status == ConversationStorePublishStatus::replaced,
          "a cover that preserves every pinned cut replaces its predecessor");
    check_eq(replaced.superseded, first.head_id, "the replaced predecessor is reported");
    check_eq(store.counters().heads, 2, "the retained divergent branch survives the replacement");

    // An older capture that claims to cover a newer descendant is a stale completion, never an overwrite.
    auto older = std::make_shared<SavedConversation>(pinned);
    ConversationStorePublish stale;
    stale.request_id = 4;
    stale.generation = 2;
    stale.supersedes = replaced.head_id;
    stale.cover = true;
    check(store.publish(older, stale, why), "the older cover is admitted as a ticket");
    ConversationStorePublishResult refused;
    check(store.wait_publish(refused, 5000), "the older cover settles");
    check(refused.status == ConversationStorePublishStatus::stale,
          "an older capture can never cover a newer descendant");
    check(store.counters().refusals_stale > 0, "the stale refusal is counted");
    store.close();
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
    // scaffolding) and nothing of the large one.
    const uint64_t prep = uint64_t(small.live.ids.size() + 1) * sizeof(int32_t) + 1024;
    const uint64_t allowance = small_payload + small_payload / 4 + 2 * small_info.manifest_bytes + prep +
                               (uint64_t(1) << 20) + (uint64_t(320) << 10);
    check(large_payload > allowance + (uint64_t(1) << 20),
          "the large image is larger than the allowance by construction");
    check_eq(store.publish_read_allowance(allowance), allowance, "the allowance is absolute");

    // A publication image that does not fit is refused instead of overrunning the envelope.
    auto big = std::make_shared<SavedConversation>(large);
    ConversationStorePublish big_request;
    big_request.request_id = 3;
    big_request.generation = 3;
    std::string why;
    check(!store.publish(big, big_request, why), "a publication image larger than the allowance is refused");
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
    store.cancel(21, 1);
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
    store.cancel(4242, 99);   // unknown identifiers are safe
    store.cancel(0, 0);
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
    check(!second.publish(std::make_shared<SavedConversation>(image), ConversationStorePublish{}, why),
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
        check(store.publish(other, request, why), "a second branch is admitted as a ticket");
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
        check(store.publish(replacement, request, why), "an evictable publication is admitted");
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
        check(!store.publish(image, request, why),
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
        check(store.publish(image, request, why), "a publication with a fault ahead of it is admitted");
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
        check(store.publish(image, request, why), "a large publication is admitted for cancellation");
        store.cancel(5, 1);
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
        check(store.publish(image, request, why), "the publication is admitted on its own lease's credit");
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
    check(donor_floor - uint64_t(grown.bytes()) < (uint64_t(256) << 10),
          "the donor's preparation is bounded by its own manifest charge, not by a head-limit maximum");

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
        check(store.publish(image, request, why), "the repeated-donor publication is admitted");
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
        check(store.publish(cover_image, cover, why),
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
        check(store.publish(image, request, why), "the publication is admitted as a ticket");
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
        check(store.publish(image, request, why), "the displacing publication is admitted");
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
        check(store.publish(image, cover, why), "the covering publication is admitted");
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
        check(store.publish(retry_image, cover, why), "the retry is admitted");
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
            check(store.publish(image2, request, why), "the publication that will be uncertain is admitted");
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
            check(!store.publish(third, blocked, why), "a faulted store admits no further publication");
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

} // namespace