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
        std::printf("FAIL: %s\n", what);
    }
}

void check_eq(uint64_t got, uint64_t want, const char* what) {
    ++checks;
    if (got != want) {
        ++failures;
        std::printf("FAIL: %s (got %llu, want %llu)\n", what, (unsigned long long)got, (unsigned long long)want);
    }
}

Digest digest_of(const std::string& text) { return sha256(text.data(), text.size()); }

std::array<uint8_t, 32> bytes_from(const std::string& text) { return digest_of(text); }

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
    image.layer_lo = 0;
    image.layer_hi = 4;
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
void check_request_specific();
void check_publication_rules();
void check_reuse();
void check_leases_and_budget();
void check_epoch_and_intent();
void check_capacity_floor();
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
    check_request_specific();
    check_publication_rules();
    check_reuse();
    check_leases_and_budget();
    check_epoch_and_intent();
    check_capacity_floor();
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
    const SavedConversation other_layout = fixture(21, 32, 1);
    const TempRoot root;
    const ConversationStoreOptions options = options_for(root.path, 1, 0);
    const Digest bucket = conversation_store_detail::store_bucket_digest(options, selector_of(left));
    check_eq(conversation_store_detail::digest_u64(bucket),
             conversation_store_detail::digest_u64(
                 conversation_store_detail::store_bucket_digest(options, selector_of(left))),
             "the same selector produces the same bucket");
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
        conversation_store_detail::object_scope_digest(uint16_t(ConversationStoreStreamKind::kv_k), &kv);
    const uint8_t payload[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    const Digest first = conversation_store_detail::object_digest(
        scope, uint16_t(ConversationStoreStreamKind::kv_k), 0, sizeof payload, payload);
    const Digest again = conversation_store_detail::object_digest(
        scope, uint16_t(ConversationStoreStreamKind::kv_k), 0, sizeof payload, payload);
    check(first == again, "the same interpretation, ordinal and bytes give one object");
    conversation_store_detail::KvMeta other = kv;
    other.heads = 3;
    const Digest different = conversation_store_detail::object_digest(
        conversation_store_detail::object_scope_digest(uint16_t(ConversationStoreStreamKind::kv_k), &other),
        uint16_t(ConversationStoreStreamKind::kv_k), 0, sizeof payload, payload);
    check(first != different, "a different K/V interpretation is a different object");
    check(first != conversation_store_detail::object_digest(
                         scope, uint16_t(ConversationStoreStreamKind::kv_k), 1, sizeof payload, payload),
          "a different layer ordinal is a different object");
    check(first != conversation_store_detail::object_digest(
                         scope, uint16_t(ConversationStoreStreamKind::kv_v), 0, sizeof payload, payload),
          "a different state kind is a different object");
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
    stream.bytes = 4;
    conversation_store_detail::ChunkRecord chunk;
    chunk.offset = 0;
    chunk.length = 4;
    chunk.object = bytes_from("object");
    stream.chunks.push_back(chunk);
    meta.streams.push_back(stream);
    meta.objects.emplace_back(chunk.object, 4);
    meta.payload_bytes = 4;
    const std::vector<uint8_t> encoded = conversation_store_detail::manifest_encode(meta);
    conversation_store_detail::ManifestMeta decoded;
    std::string why;
    check(conversation_store_detail::manifest_decode(encoded.data(), encoded.size(),
                                                    conversation_store_detail::unconstrained_limits(), decoded, why),
          "a manifest round-trips");
    check_eq(decoded.epoch, 3, "the epoch survives the round trip");
    check_eq(decoded.generation, 9, "the generation survives the round trip");
    check_eq(decoded.payload_bytes, 4, "the payload byte count survives the round trip");
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
    if (!store.publish(shared, request, why)) return false;
    return store.wait_publish(out, 120000);
}

void check_round_trip() {
    TempRoot root;
    const SavedConversation image = fixture(41, 256);
    uint64_t head_id = 0;
    {
        ConversationStore store(options_for(root.path, uint64_t(1) << 30, 0));
        check(store.enabled(), "the store opens its root");
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
        check(found[0].head_id == head_id, "the match names the published head");
        check_eq(uint64_t(found[0].tokens), image.live.ids.size(), "the match is the live cut's length");
        check(found[0].live, "the match reports a live cut");
        check(found[0].pinned == false, "the live cut is not pinned");

        std::string why;
        check(store.demand(query, why), "a demand is admitted");
        ConversationStoreRead read;
        check(store.wait_ready(7, 1, read, 120000), "the request-specific wait returns this request's completion");
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
        check(store.wait_ready(8, 1, miss_read, 120000), "the miss settles");
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
        check(store.wait_ready(9, 1, read, 120000), "the read settles after the restart");
        check(read.status == ConversationStoreReadStatus::ready, "the retained image is restorable");
        check(read.image != nullptr && read.head_id == head_id, "the restart serves the retained head's image");
        store.release_read(9, 1);
        store.close();
    }
}

void check_request_specific() {
    TempRoot root;
    ConversationStore store(options_for(root.path, uint64_t(1) << 30, 0));
    const SavedConversation image = fixture(51, 128);
    ConversationStorePublishResult result;
    check(publish_and_wait(store, image, 1, 1, result) &&
              result.status == ConversationStorePublishStatus::stored,
          "the publication is stored");
    std::string why;
    check(store.demand(query_of(image, 100, 1), why), "the first demand is admitted");
    check(store.demand(query_of(image, 200, 1), why), "the second demand is admitted");
    ConversationStoreRead second;
    check(store.wait_ready(200, 1, second, 120000), "the second request settles");
    check_eq(second.request_id, 200, "the second request receives only its own completion");
    ConversationStoreRead first;
    check(store.wait_ready(100, 1, first, 120000), "the first request settles afterwards");
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
    check(store.wait_ready(300, 1, miss_read, 120000), "the unrelated request settles");
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
    check(store.wait_publish(replaced, 120000), "the covered publication settles");
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

    SavedConversation grown = base;
    for (int64_t t = 2048; t < 3072; ++t) grown.live.ids.push_back(int32_t(81 * 7 + uint64_t(t)));
    grown.live.gdn.resize(grown.live.ids.size(), uint8_t(81));
    grown.live.block_pos.resize(grown.live.ids.size(), uint8_t(81 + 4));
    ConversationStoreReuse reuse;
    reuse.head_id = stored.head_id;
    reuse.stream = ConversationStoreStreamId{0, uint16_t(ConversationStoreStreamKind::ids), 0, 0};
    reuse.source_offset = 0;
    reuse.offset = 0;
    reuse.bytes = 4096;  // exactly the base stream's first immutable chunk (1024 int32 tokens)

    auto image = std::make_shared<SavedConversation>(grown);
    ConversationStorePublish request;
    request.request_id = 2;
    request.generation = 2;
    request.reuse.push_back(reuse);
    std::string why;
    check(store.publish(image, request, why), "a publication carrying a proven interval is admitted");
    ConversationStorePublishResult reused;
    check(store.wait_publish(reused, 120000), "the reuse publication settles");
    check(reused.status == ConversationStorePublishStatus::stored, "the reuse publication is stored");
    check_eq(reused.reused_bytes, 4096, "the proven interval was adopted instead of re-read");
    check(store.counters().reused_chunks >= 1, "the adopted chunk is counted");

    // An interval that is not chunk-aligned is ignored, never guessed at.
    auto second = std::make_shared<SavedConversation>(grown);
    ConversationStorePublish unaligned;
    unaligned.request_id = 3;
    unaligned.generation = 3;
    ConversationStoreReuse bad = reuse;
    bad.offset = 8;
    unaligned.reuse.push_back(bad);
    check(store.publish(second, unaligned, why), "an unusable interval still admits the publication");
    ConversationStorePublishResult ignored;
    check(store.wait_publish(ignored, 120000), "the unaligned publication settles");
    check_eq(ignored.reused_bytes, 0, "an unusable interval is ignored");
    check(ignored.new_bytes > 0, "the bytes were written normally instead");

    // The reused image restores to exactly the outgoing fixture.
    ConversationStoreQuery query = query_of(grown, 4, 4);
    check(store.demand(query, why), "the grown image can be demanded");
    ConversationStoreRead read;
    check(store.wait_ready(4, 4, read, 120000), "the grown read settles");
    check(read.status == ConversationStoreReadStatus::ready, "the grown image is restorable");
    if (read.image) {
        std::string why2;
        check(images_equal(*read.image, grown, why2), "the restored grown image matches its fixture");
    }
    store.release_read(4, 4);
    store.close();
}

void check_leases_and_budget() {
    TempRoot root;
    ConversationStoreOptions options = options_for(root.path, uint64_t(1) << 30, 0);
    options.read_budget_bytes = 0;  // no read is admitted until the caller publishes an allowance
    ConversationStore store(options);
    const SavedConversation image = fixture(91, 128);
    ConversationStorePublishResult result;
    check(publish_and_wait(store, image, 1, 1, result) &&
              result.status == ConversationStorePublishStatus::stored,
          "the publication is stored");
    std::string why;

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
    check(store.wait_ready(300, 1, read, 120000), "the demand is admitted once the allowance exists");
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
    ConversationStorePublishResult result;
    check(publish_and_wait(store, image, 1, 1, result), "the publication is admitted");
    check(result.status == ConversationStorePublishStatus::refused_floor,
          "an impossible filesystem floor refuses the publication");
    check_eq(uint64_t(store.counters().heads), 0, "a refused publication authorizes no head");
    check_eq(store.disk_charged_bytes(), 0, "a refused publication leaves nothing charged");
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
    check(store.wait_ready(1, 1, read, 120000), "the damaged read settles");
    check(read.status == ConversationStoreReadStatus::corrupt,
          "a damaged object is reported as corrupt, never served");
    check(read.image == nullptr, "a corrupt read hands out no image");
    check(store.counters().reads_corrupt > 0, "the corruption is counted");
    store.close();
}

} // namespace