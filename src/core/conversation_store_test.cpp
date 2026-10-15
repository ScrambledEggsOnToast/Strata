// Deterministic CPU tests for the durable conversation store.  No GPU, no model, no network: a temporary root under
// the system temp directory, fixtures built by hand, and checks that read the module's own observable contract -
// matches, statuses, leases, epochs and byte counts.  Every check is a consumer-visible behaviour, not an internal.
#include "strata/core/conversation_store.hpp"

#include "conversation_store_format.hpp"
#include "conversation_store_sha256.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
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

// ---------------------------------------------------------------------------------------------------------------
// Checks
// ---------------------------------------------------------------------------------------------------------------

void check_crypto();
void check_layout();
void check_identity();
void check_manifest();
void check_round_trip();
void check_multi_stage_restore();
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
        ConversationStore store(options_for(path, charged + (uint64_t(4) << 10), 0));
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
        ConversationStore store(options_for(path, charged + (uint64_t(4) << 10), 0));
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


} // namespace