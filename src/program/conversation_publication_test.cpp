// Deterministic CPU regressions for the native publication handover (include/strata/program/conversation_publication.hpp).
// The store is the real one, on a temporary root: the handover's verdicts, the retirement a completed replacement
// performs on the references to the head it removed, and the branch a divergent capture is published as are all
// checked against actual admissions, actual catalog commits and a fresh open of the same root - never against a
// stand-in that would agree with whatever the handover happened to decide.
#include "strata/program/conversation_publication.hpp"
#include "strata/program/conversation_store_admission.hpp"
#include "strata/program/prefix_selection.hpp"

#include "conversation_store_format.hpp"
#include "conversation_store_sha256.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <dirent.h>
#include <memory>
#include <limits>
#include <new>
#include <optional>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/wait.h>
#include <utility>
#include <vector>

namespace {

using namespace strata::core;
using strata::program::ConversationPublicationCapture;
using strata::program::ConversationPublicationOwner;
using strata::program::ConversationPublicationPending;
using strata::program::ConversationPublicationQueue;
using strata::program::ConversationPublicationSink;
using strata::program::ContinuationPublisher;
using strata::program::PublicationHandle;
using strata::program::PublicationReceipt;
using strata::program::PublicationStatus;
using strata::program::OptionalSharedExportInput;
using strata::program::optional_shared_export;
using conversation_store_detail::images_equal;

int failures = 0;
int checks = 0;
std::vector<std::string> phase_lines;
void phase_report(const char* text) { phase_lines.emplace_back(text); }

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
        std::fprintf(stderr, "FAIL: %s (got %llu, want %llu)\n", what, (unsigned long long) got,
                     (unsigned long long) want);
    }
}

void check_optional_shared_export_predicate() {
    const OptionalSharedExportInput eligible{
        .enabled = true, .shared_prefix = true, .continues = false, .private_pin = true, .tokens = 64,
        .restore_lower_bound_ms = 4.0, .recompute_ms = 2.0};
    const auto expect = [](const OptionalSharedExportInput& input, bool publish, const char* reason) {
        const auto result = optional_shared_export(input);
        check(result.publish == publish && std::strcmp(result.reason, reason) == 0,
              reason);
    };
    auto input = eligible;
    input.enabled = false; expect(input, true, "gate_off");
    input = eligible; input.shared_prefix = false; expect(input, true, "not_optional_shared_export");
    input = eligible; input.continues = true; expect(input, true, "not_optional_shared_export");
    input = eligible; input.has_checkpoints = true; expect(input, true, "not_optional_shared_export");
    input = eligible; input.has_images = true; expect(input, true, "not_optional_shared_export");
    input = eligible; input.tokens = 0; expect(input, true, "not_optional_shared_export");
    input = eligible; input.tokens = -1; expect(input, true, "not_optional_shared_export");
    input = eligible; input.private_pin = false; expect(input, true, "private_pin_not_proven");
    input = eligible; input.restore_lower_bound_ms.reset(); expect(input, true, "unknown_restore_cost");
    input = eligible; input.recompute_ms.reset(); expect(input, true, "unknown_recompute_cost");
    for (const double invalid : {-1.0, std::numeric_limits<double>::infinity(),
                                 std::numeric_limits<double>::quiet_NaN()}) {
        input = eligible; input.restore_lower_bound_ms = invalid; expect(input, true, "unknown_restore_cost");
        input = eligible; input.recompute_ms = invalid; expect(input, true, "unknown_recompute_cost");
    }
    input = eligible; input.recompute_ms = 0.0; expect(input, true, "unknown_recompute_cost");
    input = eligible; input.restore_lower_bound_ms = 0.0; expect(input, true, "reuse_pays");
    input = eligible; input.restore_lower_bound_ms = 1.0; expect(input, true, "reuse_pays");
    input = eligible; input.restore_lower_bound_ms = 2.0; expect(input, false, "restore_not_cheaper");
    expect(eligible, false, "restore_not_cheaper");

    using strata::program::PrefixCosts;
    using strata::program::PrefixTransfer;
    PrefixCosts costs;
    const strata::program::PrefixWork work{1024, 7};
    check(!costs.transfer_lower_bound_ms(PrefixTransfer::parked, work), "absent restore observation keeps unknown");
    costs.transfer(PrefixTransfer::parked, work, 8.0);
    costs.transfer(PrefixTransfer::parked, work, 2.0);
    check(costs.transfer_lower_bound_ms(PrefixTransfer::parked, work) == 2.0,
          "restore lower envelope is the best exact observation, not the average");
    check(!costs.transfer_lower_bound_ms(PrefixTransfer::parked, {1024, 8}),
          "restore route mismatch stays unknown");
    check(!costs.transfer_lower_bound_ms(PrefixTransfer::parked, {2048, 7}),
          "a scaled transfer estimate is not an observed restore lower envelope");
}

// ---------------------------------------------------------------------------------------------------------------
// Fixtures, filesystem helpers and the store's own options
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
    TempRoot() : TempRoot("/tmp") {}
    explicit TempRoot(const char* parent) {
        std::string pattern = std::string(parent) + "/strata-publication-test-XXXXXX";
        const char* made = ::mkdtemp(pattern.data());
        if (made) path = made;
    }
    ~TempRoot() { remove_tree(path); }
};

std::array<uint8_t, 32> bytes_from(const std::string& text) {
    return conversation_store_detail::sha256(text.data(), text.size());
}

ConversationStoreOptions options_for(const std::string& root) {
    ConversationStoreOptions options;
    options.root = root;
    options.domain = bytes_from("domain:native-publication-test");
    options.publication_timing = true;
    options.model = bytes_from("model:native-publication-test");
    options.engine = bytes_from("engine:native-publication-test");
    options.quota_bytes = uint64_t(1) << 30;
    options.min_free_bytes = 0;
    options.metadata_bytes = uint64_t(8) << 20;
    options.head_limit = 8;
    options.chunk_bytes = 4096;
    options.max_reads = 1;
    options.max_read_requests = 8;
    options.max_pending_publishes = 1;
    options.read_budget_bytes = uint64_t(64) << 20;
    return options;
}

// One stored record with a live prefix, one pinned checkpoint and two K/V layers: enough for the store to embed it,
// advertise the cut a later client can reuse and prove a cover of it.  Deliberately small - these regressions are
// about the handover's decisions, never about the writer's arithmetic.
SavedConversation stored_record(uint64_t seed, int64_t tokens) {
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
    for (int layer = 0; layer < 2; ++layer) {
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

// The native shape: every later stage owns its own image and checkpoint records, never primary stage_parts.
SavedConversation all_stage_record(uint64_t seed, int64_t tokens) {
    SavedConversation image = stored_record(seed, tokens);
    image.geometry[1] = 24;
    image.checkpoints.clear();
    for (size_t index = 1; index < 3; ++index) {
        SavedConversation stage = stored_record(seed + index * 20, tokens);
        stage.scope = image.scope;
        stage.geometry = image.geometry;
        stage.layer_lo = int64_t(index * 8);
        stage.layer_hi = int64_t((index + 1) * 8);
        stage.live.ids = image.live.ids;
        stage.checkpoints.clear();
        image.stage_images.push_back(std::move(stage));
    }
    return image;
}

// One turn later on the same branch: the live record and the K/V extents grow, so every unchanged byte of the prefix
// keeps the physical identity the store deduplicates on.  `seed` chooses the appended bytes, so two captures taken
// from the same head with different seeds are two different states that share only that head's prefix.
void extend_record(SavedConversation& grown, int64_t tokens, int seed) {
    for (int64_t t = int64_t(grown.live.ids.size()); t < tokens; ++t)
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
    for (SavedConversation& stage : grown.stage_images) extend_record(stage, tokens, seed);
}

SavedConversation turn_on(const SavedConversation& base, int64_t tokens, int seed) {
    SavedConversation grown = base;
    extend_record(grown, tokens, seed);
    return grown;
}

ConversationStoreQuery query_of(const SavedConversation& image, uint64_t request_id, uint64_t generation) {
    ConversationStoreQuery query;
    query.selector.scope = image.scope;
    query.selector.cvec = image.cvec;
    query.selector.geometry = image.geometry;
    query.selector.layer_lo = image.layer_lo;
    query.selector.layer_hi = image.layer_hi;
    query.tokens = image.live.ids;
    // A stored cut must be strictly shorter than the request: the last prompt token starts the next verify window.
    query.tokens.push_back(12345);
    query.request_id = request_id;
    query.generation = generation;
    return query;
}

// The record a head was written from must still be demandable, and restore byte for byte, in a fresh process.
void check_record_restores(ConversationStore& store, SavedConversation expected, uint64_t request_id,
                           bool root_cut = false) {
    expected.live.pinned = root_cut;
    ConversationStoreQuery query = query_of(expected, request_id, 1);
    std::string why;
    check(store.demand(query, why), "a persisted record is demandable after a fresh open");
    ConversationStoreRead read;
    check(store.wait_ready(request_id, 1, read, 5000), "the demand settles after a fresh open");
    check(read.status == ConversationStoreReadStatus::ready, "the persisted record restores after a fresh open");
    check(read.live && read.tokens == int64_t(expected.live.ids.size()), "the restored cut is the exact latest live cut");
    check_eq(read.stages, expected.stage_images.size() + 1, "the restored frontier includes every stage");
    check(read.image && images_equal(*read.image, expected, why), "the restored record is byte-identical");
    store.release_read(request_id, 1);
}

// ---------------------------------------------------------------------------------------------------------------
// The serving engine as this module sees it
// ---------------------------------------------------------------------------------------------------------------

struct Sink {
    ConversationPublicationSink out;
    ConversationStore& store;
    ConversationCache cache;
    const size_t budget;
    uint64_t next_id = 0;
    uint64_t next_ticket = 0;
    size_t acquisitions = 0, lease_releases = 0, ticket_releases = 0, retentions = 0;
    bool fail_acquire = false;
    std::string reports;

    explicit Sink(ConversationStore& store, size_t budget = size_t(64) << 20,
                  size_t slots = ConversationPublicationQueue::kMaxQueued, size_t headroom = size_t(8) << 20)
        : store(store), cache(budget, slots), budget(budget) {
        out.publication_headroom_bytes = headroom;
        out.report = [this](const char* line) { reports += line; };
        out.acquire_image = [this](uint64_t ticket) -> std::shared_ptr<const SavedConversation> {
            ++acquisitions;
            if (fail_acquire) throw std::bad_alloc();
            // This is the production ordering: allocate the shared owner before the allocation-free cache move.
            auto root = std::make_shared<SavedConversation>();
            std::optional<SavedConversation> image = cache.take_publication(ticket);
            if (!image) return {};
            *root = std::move(*image);
            return root;
        };
        out.release_publication = [this](uint64_t ticket) {
            ++ticket_releases;
            cache.release_publication(ticket);
        };
        out.release_lease = [this](size_t bytes) {
            ++lease_releases;
            cache.release_lease(bytes);
        };
        out.retain_image = [this](std::shared_ptr<const SavedConversation> image) {
            const size_t held = size_t(this->store.ram_held_bytes());
            if (!cache.can_fit(image->bytes(), held)) return;
            SavedConversation alias = image->frozen_alias();
            if (cache.put(std::move(alias), held)) ++retentions;
        };
        out.refresh_allowance = [this] { refresh_allowance(); };
        out.next_publication_id = [this](uint64_t& id) {
            id = ++next_id;
            return true;
        };
        refresh_allowance();
    }

    void refresh_allowance() {
        const size_t occupied = cache.envelope_bytes();
        store.publish_read_allowance(occupied < budget ? budget - occupied : 0);
    }
};

// One metadata-only capture, exactly as the serving engine builds it. The public transition, not this fixture,
// protects and parks the image, and the head's sink callback later transfers its one charge into a lease.
ConversationPublicationCapture capture_of(uint64_t owner, uint64_t generation, uint64_t supersedes_hint,
                                          bool continues, const SavedConversation& image, uint64_t epoch,
                                          uint64_t ticket) {
    ConversationPublicationCapture entry;
    entry.owner = owner;
    entry.owner_generation = generation;
    entry.capture_epoch = epoch;
    entry.supersedes_hint = supersedes_hint;
    entry.continues = continues;
    entry.scope = image.scope;
    entry.cvec = image.cvec;
    entry.live = image.live.ids;
    entry.image_ticket = ticket;
    return entry;
}

bool capture_image(ConversationPublicationQueue& publications, Sink& sink, uint64_t owner, uint64_t generation,
                   uint64_t supersedes_hint, bool continues, const SavedConversation& image, uint64_t epoch) {
    SavedConversation captured = image;
    ConversationPublicationCapture entry = capture_of(owner, generation, supersedes_hint, continues, image, epoch,
                                                       ++sink.next_ticket);
    const auto result = publications.capture(sink.cache, std::move(captured), std::move(entry),
                                             size_t(sink.store.ram_held_bytes()), 0, sink.out);
    sink.refresh_allowance();
    return result.queued;
}

const ConversationPublicationPending* pending_of(const ConversationPublicationQueue& publications, uint64_t owner) {
    for (const ConversationPublicationPending& pending : publications.pending())
        if (pending.owner == owner) return &pending;
    return nullptr;
}

bool any_record_names(const ConversationPublicationQueue& publications, uint64_t head) {
    for (const ConversationPublicationCapture& entry : publications.queued())
        if (entry.supersedes_hint == head) return true;
    for (const ConversationPublicationPending& pending : publications.pending())
        if (pending.supersedes == head) return true;
    return false;
}

// Hand over and harvest until nothing is queued and nothing is outstanding, at the seam's own cadence.
void settle(ConversationPublicationQueue& publications, ConversationStore& store, Sink& sink) {
    for (int waited = 0; waited < 20000 && publications.holds_work(); ++waited) {
        publications.flush(store, sink.out, 20);
        ::usleep(1000);
    }
}

// Produce a real unpinned latest through the same capture/handoff as every other fixture. Its first checkpoint is
// the pinned root, and the whole record re-carries that root promise as a checkpoint, so the root can be replaced.
uint64_t mount_latest(ConversationStore& store, ConversationPublicationQueue& publications, Sink& sink,
                      const SavedConversation& record) {
    if (record.checkpoints.empty()) {
        check(false, "an unpinned-latest fixture needs a reusable root checkpoint");
        return 0;
    }
    SavedConversation root = record;
    root.live = record.checkpoints.front();
    root.checkpoints.clear();
    for (ConversationKv& kv : root.kv) {
        kv.cells = int64_t(root.live.ids.size());
        kv.k.resize(root.live.ids.size() * 16);
        kv.v.resize(root.live.ids.size() * 16);
    }
    check(capture_image(publications, sink, 0, publications.owner(0).generation + 1, 0, true, root, store.epoch()),
          "the mount fixture's reusable root is captured");
    settle(publications, store, sink);
    const uint64_t root_head = publications.owner(0).lineage.head;
    check(capture_image(publications, sink, 0, publications.owner(0).generation + 1, root_head, true, record, store.epoch()),
          "the mount fixture's whole latest record is captured");
    settle(publications, store, sink);
    const uint64_t head = publications.owner(0).lineage.head;
    check(head != 0 && head != root_head && store.counters().heads == 1,
          "the whole latest replaces its root while preserving the pinned checkpoint");
    publications.owner(0).mount(head, store.epoch(), record.live.ids, int64_t(record.live.ids.size()),
                                record.scope, record.cvec);
    return head;
}

// Demand a real checkpoint image, then run the native mount handoff before shortening the execution record.
std::vector<int32_t> mount_checkpoint(ConversationStore& store, ConversationPublicationQueue& publications,
                                      const SavedConversation& record, uint64_t request_id) {
    ConversationStoreQuery query = query_of(record, request_id, 1);
    query.tokens = record.checkpoints.front().ids;
    query.tokens.push_back(98765);   // diverges immediately after the stored checkpoint, not after the live cut
    std::string why;
    check(store.demand(query, why), "a checkpoint-only prompt is demanded from the real store");
    ConversationStoreRead read;
    check(store.wait_ready(request_id, 1, read, 5000) &&
              read.status == ConversationStoreReadStatus::ready && read.image,
          "the checkpoint demand returns its complete mounted image");
    if (!read.image) {
        store.release_read(request_id, 1);
        return {};
    }
    check(!read.live && read.tokens == int64_t(record.checkpoints.front().ids.size()),
          "the mount resumes a short checkpoint, not the full live record");
    check(images_equal(*read.image, record, why), "the short mount still carries the whole persisted record");
    std::vector<int32_t> live = read.image->live.ids;
    ConversationPublicationOwner& owner = publications.owner(0);
    const uint64_t previous_generation = owner.generation;
    owner.mount(read.head_id, read.epoch, live, read.tokens, query.selector.scope, query.selector.cvec);
    check_eq(owner.generation, previous_generation + 1, "a disk mount moves the owner generation");
    check(owner.lineage.valid && owner.lineage.head == read.head_id && owner.lineage.epoch == read.epoch &&
              owner.lineage.scope.matches(query.selector.scope) && owner.lineage.cvec == query.selector.cvec,
          "a disk mount preserves its head, epoch, scope and steering");
    live.resize(size_t(read.tokens));   // the same ordering as the native direct-checkpoint restore
    read.image.reset();
    store.release_read(request_id, 1);
    return live;
}

// ---------------------------------------------------------------------------------------------------------------
// The regressions
// ---------------------------------------------------------------------------------------------------------------

// A latest head's live cut is deliberately unpinned. Resuming one of its shorter checkpoints and diverging must not
// make that checkpoint stand in for the whole predecessor in the coverage proof: all its actual pins may still be
// present in the new branch, so the store cannot recover the missing ancestry check on the native owner's behalf.
void check_checkpoint_mount_preserves_the_longer_branch() {
    TempRoot root;
    ConversationStore store(options_for(root.path));
    if (!store.enabled()) return;
    Sink sink(store);
    ConversationPublicationQueue publications;
    SavedConversation record = stored_record(141, 64);
    record.cvec = false;
    const uint64_t head = mount_latest(store, publications, sink, record);
    const uint64_t replaced_before = store.counters().replaced;
    std::vector<int32_t> live = mount_checkpoint(store, publications, record, 201);
    if (live.empty()) return;

    SavedConversation resumed = record;
    resumed.live = record.checkpoints.front();
    resumed.live.ids = std::move(live);
    resumed.live.pinned = false;
    for (ConversationKv& kv : resumed.kv) {
        kv.cells = int64_t(resumed.live.ids.size());
        kv.k.resize(resumed.live.ids.size() * 16);
        kv.v.resize(resumed.live.ids.size() * 16);
    }
    const SavedConversation divergent = turn_on(resumed, 96, 900);
    ConversationPublicationOwner& owner = publications.owner(0);
    check(capture_image(publications, sink, 0, owner.generation + 1, head, true, divergent, store.epoch()),
          "the branch generated after a short disk mount is queued");
    publications.flush(store, sink.out, 0);
    const ConversationPublicationPending* pending = pending_of(publications, 0);
    check(pending && pending->supersedes == 0, "a short-mounted divergence claims no cover of the longer head");
    settle(publications, store, sink);
    check_eq(publications.counters().refused, 0, "the short-mounted branch publishes without refusal");
    check_eq(store.counters().replaced, replaced_before, "the longer predecessor was not retired by the divergent branch");
    check_eq(store.counters().heads, 2, "the original longer record and the checkpoint branch both persist");

    store.close();
    ConversationStore reopened(options_for(root.path));
    check(reopened.enabled(), "the short-mount branch store reopens");
    if (!reopened.enabled()) return;
    check_record_restores(reopened, record, 202);
    check_record_restores(reopened, divergent, 203, true);
}

// A short mount is not itself a reason to forbid replacement. If execution re-creates the predecessor's whole live
// prefix before extending it, the existing full-record/pin predicate must still retire that unpinned latest head.
void check_checkpoint_mount_allows_a_whole_prefix_replacement() {
    TempRoot root;
    ConversationStore store(options_for(root.path));
    if (!store.enabled()) return;
    Sink sink(store);
    ConversationPublicationQueue publications;
    const SavedConversation record = stored_record(151, 64);
    const uint64_t head = mount_latest(store, publications, sink, record);
    const uint64_t replaced_before = store.counters().replaced;
    std::vector<int32_t> live = mount_checkpoint(store, publications, record, 211);
    if (live.empty()) return;
    live.insert(live.end(), record.live.ids.begin() + long(live.size()), record.live.ids.end());
    check(live == record.live.ids, "execution re-creates the whole predecessor prefix after its short mount");
    SavedConversation continued = record;
    continued.live.ids = std::move(live);
    const SavedConversation covered = turn_on(continued, 96, 151);
    ConversationPublicationOwner& owner = publications.owner(0);
    check(capture_image(publications, sink, 0, owner.generation + 1, head, true, covered, store.epoch()),
          "the full-prefix capture after a short disk mount is queued");
    publications.flush(store, sink.out, 0);
    const ConversationPublicationPending* pending = pending_of(publications, 0);
    check(pending && pending->supersedes == head, "the full-prefix capture still covers the mounted head");
    settle(publications, store, sink);
    check_eq(publications.counters().refused, 0, "the full-prefix capture publishes without refusal");
    check_eq(store.counters().replaced, replaced_before + 1, "a valid full-prefix continuation replaces its unpinned latest head");
    check_eq(store.counters().heads, 1, "a valid replacement does not retain an unnecessary predecessor branch");

    store.close();
    ConversationStore reopened(options_for(root.path));
    check(reopened.enabled(), "the short-mount replacement store reopens");
    if (!reopened.enabled()) return;
    check_record_restores(reopened, covered, 212);
}

// A capture that diverges from the head its owner's state was taken from must be published as the branch it is - and
// a completion that *replaces* a head must invalidate every reference to it, for every owner and every queued capture,
// whether or not that completion may attach to the owner that produced it.  Without the retirement bookkeeping the
// second capture keeps naming a head the store has removed, the store refuses it as stale, and eight identical
// refusals abandon the only snapshot of that divergence the engine paid for.
void check_divergent_capture_is_published_as_a_branch() {
    TempRoot root;
    ConversationStore store(options_for(root.path));
    if (!store.enabled()) return;
    const uint64_t epoch = store.epoch();
    Sink sink(store);
    ConversationPublicationQueue publications;

    const SavedConversation record = stored_record(77, 64);
    const SavedConversation covered = turn_on(record, 96, 77);      // the same branch, one turn later
    const SavedConversation divergent = turn_on(record, 96, 900);   // the same prefix, another answer
    const SavedConversation mounted = turn_on(record, 96, 901);     // a second owner's capture of the same prefix

    const uint64_t root_head = mount_latest(store, publications, sink, record);
    const uint64_t replaced_before = store.counters().replaced;
    const uint64_t capture_generation = publications.owner(0).generation + 1;

    // A second owner mounts that very head - an ordinary client whose prefix is the stored record.
    ConversationPublicationOwner& other = publications.owner(1);
    other.generation = 5;
    other.lineage.head = root_head;
    other.lineage.epoch = epoch;
    other.lineage.live = record.live.ids;
    other.lineage.scope = record.scope;
    other.lineage.cvec = record.cvec;
    other.lineage.valid = true;

    // The covering capture is handed over first; the divergent one and the mounted owner's are captured while its
    // completion is still in flight, so both name the head its completion is about to remove.
    check(capture_image(publications, sink, 0, capture_generation, root_head, true, covered, epoch),
          "the covering capture is queued");
    publications.flush(store, sink.out, 0);
    check_eq(publications.pending().size(), 1, "the covering capture is in flight");
    check_eq(publications.pending().front().supersedes, root_head,
             "the cover names the head whose whole record it still contains");

    check(capture_image(publications, sink, 0, capture_generation + 1, root_head, true, divergent, epoch),
          "the divergent capture is queued while the cover is in flight");
    check(capture_image(publications, sink, 1, 5, root_head, true, mounted, epoch),
          "the mounted owner's capture is queued");

    publications.flush(store, sink.out, 5000);
    check(!any_record_names(publications, root_head), "no queued or outstanding capture still names the retired head");
    const ConversationPublicationPending* divergent_pending = pending_of(publications, 0);
    check(divergent_pending != nullptr && divergent_pending->supersedes == 0,
          "the divergent capture is handed over as a branch of its own");
    const ConversationPublicationPending* mounted_pending = pending_of(publications, 1);
    check(mounted_pending == nullptr || mounted_pending->supersedes == 0,
          "the mounted owner's capture claims no predecessor either");
    check(!publications.owner(0).lineage.valid,
          "the later capture of the same owner never adopts the completion it cannot describe");
    check(!publications.owner(1).lineage.valid, "the owner that mounted the retired head forgets it");

    settle(publications, store, sink);
    check(!any_record_names(publications, root_head), "no capture ever named the retired head again");
    check_eq(publications.counters().refused, 0, "all queued captures publish without refusal");
    check_eq(store.counters().replaced, replaced_before + 1, "exactly one publication replaced a head after mounting");
    check_eq(store.counters().heads, 3, "the covered head, the divergent capture and the other owner's are all heads");

    // A fresh open of the same root: the two divergent captures and the covered head are all durable.
    store.close();
    {
        ConversationStore reopened(options_for(root.path));
        if (!reopened.enabled()) {
            check(false, "the root reopens");
            return;
        }
        check_eq(reopened.counters().heads, 3, "the reopened store keeps all three heads");
        check_record_restores(reopened, covered, 21);
        check_record_restores(reopened, divergent, 22, true);
        check_record_restores(reopened, mounted, 23, true);
    }
}

// The writer queues its result and only then clears its busy flag, so a completion can land between the seam's
// harvest and its observation of the idle writer.  The seam must reconcile again before it derives the next capture's
// predecessor: a completion missed in that window leaves a lineage naming a head the store has already removed, and
// the capture derived from it is refused as stale.
void check_completion_landing_after_the_harvest() {
    TempRoot root;
    ConversationStore store(options_for(root.path));
    if (!store.enabled()) return;
    const uint64_t epoch = store.epoch();
    Sink sink(store);
    ConversationPublicationQueue publications;

    const SavedConversation record = stored_record(88, 64);
    const SavedConversation covered = turn_on(record, 96, 88);
    const SavedConversation divergent = turn_on(record, 96, 902);

    const uint64_t root_head = mount_latest(store, publications, sink, record);
    const uint64_t capture_generation = publications.owner(0).generation + 1;

    check(capture_image(publications, sink, 0, capture_generation, root_head, true, covered, epoch),
          "the covering capture is queued");
    publications.flush(store, sink.out, 0);   // handed over; its completion is deliberately not harvested yet
    check_eq(publications.pending().size(), 1, "the covering capture is in flight and unsettled");

    check(capture_image(publications, sink, 0, capture_generation + 1, root_head, true, divergent, epoch),
          "the divergent capture is queued");

    // Land the covering capture's completion inside that window: this seam's harvest has already run, and the writer
    // goes idle (with the replacement settled and unharvested) before the seam observes it - exactly the state the
    // second reconcile exists for.  If the harvest happens to win the race, the assertions below still hold: they
    // state the guarantee, not the interleaving.
    sink.out.between_harvest_and_idle_for_test = [&] {
        for (int waited = 0; waited < 20000 && store.writer_busy(); ++waited) ::usleep(1000);
        check(!store.writer_busy(), "the covering capture settled while the seam observed the writer");
    };
    publications.flush(store, sink.out, 0);
    sink.out.between_harvest_and_idle_for_test = nullptr;

    const ConversationPublicationPending* divergent_pending = pending_of(publications, 0);
    check(divergent_pending != nullptr, "the divergent capture was handed over");
    check(divergent_pending == nullptr || divergent_pending->supersedes == 0,
          "it is handed over as a branch of its own, never as a cover of the removed head");
    settle(publications, store, sink);
    check_eq(store.counters().heads, 2, "the covered head and the divergent branch both exist");
}

// An owner slot can be handed to another conversation (a remount moves its generation and drops its lineage), and a
// one-shot borrow establishes no lineage at all.  Neither may adopt a completion of the state that was there before.
void check_a_moved_on_owner_never_receives_an_old_completion() {
    TempRoot root;
    ConversationStore store(options_for(root.path));
    if (!store.enabled()) return;
    const uint64_t epoch = store.epoch();
    Sink sink(store);
    ConversationPublicationQueue publications;

    const SavedConversation first_conversation = stored_record(99, 64);
    const SavedConversation second_conversation = stored_record(101, 64);

    check(capture_image(publications, sink, 0, 1, 0, true, first_conversation, epoch),
          "the first conversation's capture is queued");
    publications.flush(store, sink.out, 0);
    check_eq(publications.pending().size(), 1, "the first conversation's capture is in flight");

    // The remount: another conversation now runs in this slot, and the completion in flight describes the old one.
    ++publications.owner(0).generation;
    publications.owner(0).lineage = {};
    settle(publications, store, sink);
    check(!publications.owner(0).lineage.valid, "the old completion attached to nothing");
    check_eq(publications.owner(0).lineage.head, 0, "the slot keeps the mount's own empty lineage");
    check_eq(store.counters().heads, 1, "the old conversation's head is still the store's only head");

    // The conversation that now runs in the slot publishes from its own mount, as a branch of its own.
    const uint64_t generation = publications.owner(0).generation;
    const uint64_t hint = publications.owner(0).lineage.valid ? publications.owner(0).lineage.head : 0;
    check_eq(hint, 0, "the moved-on owner names no predecessor");
    check(capture_image(publications, sink, 0, generation, hint, true, second_conversation, epoch),
          "the second conversation's capture is queued");
    publications.flush(store, sink.out, 0);
    const ConversationPublicationPending* second = pending_of(publications, 0);
    check(second != nullptr && second->supersedes == 0, "it supersedes nothing");
    settle(publications, store, sink);
    check_eq(store.counters().heads, 2, "each conversation keeps its own head");

    // A one-shot borrow is published as a branch of its own and never becomes the borrowing owner's lineage.
    check(capture_image(publications, sink, 1, 1, 0, false, second_conversation, epoch),
          "the one-shot borrow is queued");
    settle(publications, store, sink);
    check(!publications.owner(1).lineage.valid, "a one-shot borrow establishes no lineage");
    check_eq(store.counters().heads, 3, "the borrow is a head of its own");
}

// A branch's own first record is the cut a later request comes back for, and the store retires a covered head only
// when the descendant re-carries every *pinned* cut it promised.  The handover therefore publishes a fresh branch
// with its live cut pinned: the cover retains the root instead of removing it, and the record - with every cut under
// it - keeps being served after the branch has moved on.  Retiring the root is what makes a returning turn, a
// divergent sibling or a pinned prefix resolve no cut at all: warm in RAM, nothing on disk.
void check_a_covered_root_keeps_its_record() {
    TempRoot root;
    ConversationStore store(options_for(root.path));
    if (!store.enabled()) return;
    const uint64_t epoch = store.epoch();
    Sink sink(store);
    ConversationPublicationQueue publications;

    const SavedConversation record = stored_record(121, 64);
    const SavedConversation later = turn_on(record, 96, 121);

    check(capture_image(publications, sink, 0, 1, 0, true, record, epoch), "the root capture is queued");
    publications.flush(store, sink.out, 0);
    settle(publications, store, sink);
    const uint64_t root_head = publications.owner(0).lineage.head;
    check(publications.owner(0).lineage.valid && root_head != 0, "the root becomes the owner's head");
    check_eq(store.counters().heads, 1, "the root is the store's only head");

    // The branch moves on: the next turn's capture still contains the root's whole record.
    check(capture_image(publications, sink, 0, 2, root_head, true, later, epoch),
          "the later turn is queued");
    publications.flush(store, sink.out, 0);
    const ConversationPublicationPending* cover = pending_of(publications, 0);
    check(cover != nullptr && cover->supersedes == root_head, "the later turn covers the root's record");
    settle(publications, store, sink);

    check_eq(store.counters().retained_branches, 1, "the cover retained the root instead of retiring it");
    check_eq(store.counters().replaced, 0, "no head was retired");
    check_eq(store.counters().heads, 2, "the root and the branch's latest are both heads");

    // The root's record is still served, and the latest turn's too, after a fresh open of the same root.
    store.close();
    {
        ConversationStore reopened(options_for(root.path));
        if (!reopened.enabled()) {
            check(false, "the root reopens");
            return;
        }
        check_eq(reopened.counters().heads, 2, "the reopened store keeps the root and the latest turn");
        check_record_restores(reopened, record, 31, true);
        check_record_restores(reopened, later, 32);
    }
}

// A deferred handover is the store saying "not now" about a resource that is occupied at this instant: it must not
// spend one of the bounded attempts, or a capture that was only ever early would be dropped after eight deferrals.
void check_deferral_does_not_spend_an_attempt() {
    TempRoot root;
    ConversationStore store(options_for(root.path));
    Sink sink(store);
    ConversationPublicationQueue publications;
    const SavedConversation large = stored_record(131, 131072);
    const SavedConversation small = stored_record(132, 64);
    mount_latest(store, publications, sink, large);
    std::string why;
    check(store.demand(query_of(large, 100, 1), why), "the blocking image is demanded");
    ConversationStoreRead read;
    check(store.wait_ready(100, 1, read, 5000) && read.image, "the blocking image is handed out");
    check(capture_image(publications, sink, 1, 1, 0, true, small, store.epoch()),
          "the blocked capture is queued");
    store.publish_read_allowance(uint64_t(8) << 20);
    for (int i = 0; i < 16; ++i) publications.flush(store, sink.out, 0);
    check_eq(publications.queued().size(), 1, "temporary pressure retains the captured image past the retry limit");
    check_eq(publications.counters().dropped, 0, "temporary pressure never drops the capture");
    check_eq(store.counters().deferred_publishes, 16, "every handover reached real occupied-resource admission");
    read.image.reset();
    store.release_read(100, 1);
    settle(publications, store, sink);
    check(!publications.holds_work(), "the retained capture publishes after its blocker releases");
    check_eq(store.counters().heads, 2, "the deferred capture becomes a durable head");
    store.close();
    {
        ConversationStore reopened(options_for(root.path));
        check_record_restores(reopened, small, 101, true);
    }
}

// Waiting publications protect the only parked owner. Restore, supersession and pressure cannot consume it; an
// allocation failure leaves the ticket parked, and acquiring only the head transfers rather than duplicates charge.
void check_waiting_ownership_and_release() {
    TempRoot root;
    ConversationStore store(options_for(root.path));
    Sink sink(store, size_t(64) << 20, 2);
    ConversationPublicationQueue publications;
    const size_t fixed = size_t(16) << 20;
    check(sink.cache.try_reserve_lease(fixed), "the resident chain reserves its startup charge");
    sink.refresh_allowance();
    SavedConversation first = stored_record(161, 64);
    first.checkpoints.clear();   // protection here is the publication ticket, not a pinned checkpoint
    const SavedConversation later = turn_on(first, 96, 161);
    check(capture_image(publications, sink, 0, 1, 0, true, first, store.epoch()), "the first protected owner is captured");
    check(capture_image(publications, sink, 0, 2, 0, true, later, store.epoch()), "a newer waiting capture is protected too");
    check_eq(publications.queued().size(), 2, "both metadata records are queued");
    if (publications.queued().size() != 2) return;
    const uint64_t first_ticket = publications.queued()[0].image_ticket;
    const uint64_t later_ticket = publications.queued()[1].image_ticket;
    for (const auto& entry : publications.queued())
        check(!entry.image && entry.lease_bytes == 0, "waiting queue entries own neither image aliases nor leases");
    check_eq(sink.cache.size(), 2, "supersession does not remove an older pending owner");
    check_eq(sink.cache.superseded(), 0, "no protected predecessor is counted as superseded");
    check_eq(sink.cache.leased_bytes(), fixed, "queued metadata adds nothing to the fixed reservation");
    const size_t envelope = sink.cache.envelope_bytes();
    const size_t held = size_t(store.ram_held_bytes());
    check_eq(store.ram_available_bytes(), sink.budget - envelope - held,
             "store reads share the reduced envelope after fixed and parked ownership");
    bool retained_in_cache = false;
    SavedConversation borrowed = sink.cache.take(0, retained_in_cache);
    check(retained_in_cache && borrowed.publication_ticket == 0, "ordinary restore explicitly reports a borrowed pending alias");
    check(sink.cache.has_publication(first_ticket), "ordinary restore leaves the sole publication owner protected");
    check_eq(sink.cache.envelope_bytes(), envelope, "borrowing does not subtract the protected parked charge");
    borrowed = {};
    check(!sink.cache.evict_oldest(), "ordinary RAM eviction cannot remove either pending image");

    SavedConversation refused = stored_record(163, 64);
    const std::vector<int32_t> refused_live = refused.live.ids;
    const uint64_t refused_ticket = ++sink.next_ticket;
    auto refused_entry = capture_of(0, 3, 0, true, refused, store.epoch(), refused_ticket);
    const auto refusal = publications.capture(sink.cache, std::move(refused), std::move(refused_entry),
                                               size_t(store.ram_held_bytes()), 0, sink.out);
    check(!refusal.parked && !refusal.queued, "genuine protected-slot exhaustion refuses the capture without overcommit");
    check(refused.live.ids == refused_live && refused.publication_ticket == 0, "a refused capture retains its ordinary image");
    check(!sink.cache.has_publication(refused_ticket), "capacity refusal creates no orphan protected ticket");
    check_eq(publications.owner(0).generation, 2, "a failed capture does not advance its owner generation");
    check_eq(publications.queued().size(), 2, "capacity refusal leaves existing metadata intact");

    const uint64_t refused_before = publications.counters().refused;
    sink.fail_acquire = true;
    publications.flush(store, sink.out, 0);
    check(sink.cache.has_publication(first_ticket) && !publications.queued().front().image,
          "shared-root allocation failure retains the protected parked owner");
    check_eq(sink.cache.leased_bytes(), fixed, "failed acquisition preserves only the fixed reservation");
    check_eq(publications.queued().front().attempts, 0, "allocation failure spends no store-refusal attempt");
    check_eq(publications.counters().refused, refused_before, "allocation retry is not a fabricated store refusal");
    sink.fail_acquire = false;
    store.publish_read_allowance(1);   // a real permanent admission refusal keeps the acquired head for retry
    publications.flush(store, sink.out, 0);
    check(publications.queued().front().image && !publications.queued()[1].image,
          "only the head acquires an image while a refused handoff retries");
    check(!sink.cache.has_publication(first_ticket) && sink.cache.has_publication(later_ticket),
          "acquisition consumes only the head's protected ticket");
    check_eq(sink.cache.envelope_bytes(), envelope, "parked-to-lease transfer leaves the complete envelope identical");
    check_eq(sink.cache.leased_bytes(), fixed + publications.queued().front().lease_bytes,
             "the acquired head adds exactly its transferred image charge to the fixed reservation");
    check_eq(sink.acquisitions, 2, "one failed allocation and one head acquisition never acquire the waiter");
    publications.drop(0, "explicit acquired-head drop", sink.out);
    check_eq(sink.cache.leased_bytes(), fixed, "dropping an acquired head preserves the fixed reservation");
    check_eq(sink.lease_releases, 1, "acquired-head drop uses the lease-release path");
    publications.abandon("execution state abandoned", sink.out);
    check(!sink.cache.has_publication(later_ticket), "abandon clears every still-parked publication protection");
    check_eq(sink.ticket_releases, 1, "a waiting drop releases protection instead of subtracting a lease");
    check(sink.cache.evict_oldest(), "an abandoned unpinned owner becomes ordinarily evictable");

    check(capture_image(publications, sink, 0, 3, 0, true, first, store.epoch()),
          "a poison-clear retry head is captured");
    check(capture_image(publications, sink, 0, 4, 0, true, later, store.epoch()),
          "a protected waiter coexists with the retry head");
    store.publish_read_allowance(1);
    publications.flush(store, sink.out, 0);
    check(publications.queued().size() == 2 && publications.queued().front().image &&
              !publications.queued()[1].image,
          "real admission pressure leaves an acquired retry head beside a waiting ticket");
    if (publications.queued().size() != 2) return;
    const size_t retry_charge = publications.queued().front().lease_bytes;
    const size_t releases_before_clear = sink.lease_releases;
    const size_t tickets_before_clear = sink.ticket_releases;
    sink.cache.clear();
    check_eq(sink.cache.leased_bytes(), fixed + retry_charge,
             "poison clearing parked owners cannot refund the acquired retry image or fixed chain");
    publications.abandon("poisoned execution state abandoned", sink.out);
    check_eq(sink.cache.leased_bytes(), fixed,
             "abandon refunds the acquired image but not the fixed chain after poison clear");
    check_eq(sink.lease_releases, releases_before_clear + 1, "the acquired path refunds exactly one image");
    check_eq(sink.ticket_releases, tickets_before_clear + 1, "the cleared waiter uses only ticket release");
    publications.abandon("repeated poison abandonment", sink.out);
    publications.reconcile(store, sink.out, 0);
    check_eq(sink.cache.leased_bytes(), fixed, "repeated cleanup cannot refund the fixed chain");
    check_eq(sink.lease_releases, releases_before_clear + 1, "repeated cleanup cannot refund the image twice");
    check_eq(sink.ticket_releases, tickets_before_clear + 1, "repeated cleanup cannot release the waiter twice");
    const size_t parking_room = sink.budget - fixed - size_t(store.ram_held_bytes());
    check(sink.cache.can_fit(parking_room, size_t(store.ram_held_bytes())),
          "the real parking budget admits exactly the remaining nonresident bytes");
    check(!sink.cache.can_fit(parking_room + 1, size_t(store.ram_held_bytes())),
          "parking cannot borrow a byte from the fixed resident chain");

    check(capture_image(publications, sink, 0, 5, 0, true, first, store.epoch()), "an epoch-bound image is captured");
    const uint64_t epoch_ticket = publications.queued().front().image_ticket;
    const size_t acquisitions_before = sink.acquisitions;
    check(store.invalidate_domain("publication epoch regression"), "the real store advances its execution epoch");
    publications.flush(store, sink.out, 0);
    check(publications.queued().empty() && !sink.cache.has_publication(epoch_ticket), "epoch drop releases parked protection");
    check_eq(sink.acquisitions, acquisitions_before, "a retired-epoch image is never acquired");

    check(capture_image(publications, sink, 0, 6, 0, true, first, store.epoch()), "a missing-ticket case is captured");
    sink.cache.clear();   // the safety invalidation seam is allowed to discard the protected physical owner
    publications.flush(store, sink.out, 0);
    check(publications.queued().empty(), "a missing protected owner is diagnosed and dropped");
    check(sink.reports.find("protected publication ticket has no image") != std::string::npos,
          "missing-ticket drop is visible rather than silently counted as publication");
    check_eq(sink.cache.leased_bytes(), fixed, "a missing ticket invents no lease and preserves the fixed chain");
}

// Metadata exhaustion is an honest publication refusal, not lost RAM caching. No compatibility path may accept an
// entry that already owns a full alias/lease, and neither kind of failed capture may move the owner's generation.
void check_capture_refusal_is_atomic() {
    TempRoot root;
    ConversationStore store(options_for(root.path));
    Sink sink(store, size_t(64) << 20, ConversationPublicationQueue::kMaxQueued + 1);
    ConversationPublicationQueue publications;
    for (size_t index = 0; index < ConversationPublicationQueue::kMaxQueued; ++index) {
        const SavedConversation image = stored_record(191 + index * 2, 64);
        check(capture_image(publications, sink, 0, index + 1, 0, true, image, store.epoch()),
              "the bounded metadata queue accepts a protected capture");
    }
    SavedConversation overflow = stored_record(251, 64);
    const uint64_t overflow_ticket = ++sink.next_ticket;
    auto overflow_entry = capture_of(0, 99, 0, true, overflow, store.epoch(), overflow_ticket);
    const auto result = publications.capture(sink.cache, std::move(overflow), std::move(overflow_entry),
                                              size_t(store.ram_held_bytes()), 0, sink.out);
    check(result.parked && !result.queued, "queue exhaustion still permits ordinary parking without publication protection");
    check(!sink.cache.has_publication(overflow_ticket), "queue exhaustion leaves no protected ticket without metadata");
    check_eq(publications.owner(0).generation, ConversationPublicationQueue::kMaxQueued,
             "queue exhaustion leaves the successful capture generation unchanged");

    SavedConversation invalid = stored_record(253, 64);
    const uint64_t invalid_ticket = ++sink.next_ticket;
    auto invalid_entry = capture_of(0, 100, 0, true, invalid, store.epoch(), invalid_ticket);
    invalid_entry.image = std::make_shared<SavedConversation>(invalid);
    invalid_entry.lease_bytes = invalid.bytes();
    const auto rejected = publications.capture(sink.cache, std::move(invalid), std::move(invalid_entry),
                                                size_t(store.ram_held_bytes()), 0, sink.out);
    check(!rejected.queued && !sink.cache.has_publication(invalid_ticket),
          "an already-aliased entry is rejected rather than accepted through a legacy handoff mode");
    check_eq(publications.counters().refused, 2, "both pre-enqueue refusals are counted honestly");
    check_eq(sink.cache.leased_bytes(), 0, "neither refused metadata capture reserves a publication lease");
    publications.abandon("bounded queue regression complete", sink.out);
    check_eq(publications.counters().dropped, ConversationPublicationQueue::kMaxQueued,
             "only the captures actually queued are counted as dropped");
}

void check_optional_retention_failure_preserves_publication() {
    TempRoot root;
    ConversationStore store(options_for(root.path));
    Sink sink(store);
    ConversationPublicationQueue publications;
    const SavedConversation image = stored_record(181, 64);
    sink.out.retain_image = [](std::shared_ptr<const SavedConversation>) { throw std::bad_alloc(); };
    check(capture_image(publications, sink, 0, 1, 0, true, image, store.epoch()), "the allocation-failure image is captured");
    settle(publications, store, sink);
    check(!publications.holds_work(), "optional retention failure cannot undo an accepted handoff");
    check_eq(publications.counters().published, 1, "the accepted publication is harvested after retention failure");
    check_eq(publications.counters().refused, 0, "an optional RAM miss is not reported as a publication refusal");
    check_eq(sink.cache.leased_bytes(), 0, "the accepted handoff releases its head lease before optional retention");
    store.close();
    ConversationStore reopened(options_for(root.path));
    check_record_restores(reopened, image, 301, true);
}

// The original failure happens before enqueue: a second full image reservation cannot fit even though the captured
// image itself does. Drive that exact refusal with the real bounded cache, then capture the newer three-stage cut
// through the production transition, pressure RAM before/after the writer seam, and demand the newest cut on reopen.
void check_bounded_capture_persists_the_newest_all_stage_frontier() {
    TempRoot root;
    auto options = options_for(root.path);
    options.chunk_bytes = uint64_t(1) << 20;   // production chunking; keep the existing 8 MiB metadata ceiling
    SavedConversation record = all_stage_record(171, 65536);
    SavedConversation newest = turn_on(record, 131072, 171);
    SavedConversation pressure = all_stage_record(173, 131072);
    record.freeze();
    newest.freeze();
    pressure.freeze();
    // One metadata ceiling covers the existing store/index, the other preserves its real writer workspace. The
    // all-stage image is larger than both together, so this still cannot admit the old second full-image charge.
    const size_t budget = newest.bytes() + 2 * size_t(options.metadata_bytes);
    options.read_budget_bytes = budget;
    ConversationStore store(options);
    Sink sink(store, budget, 1, size_t(options.metadata_bytes));
    ConversationPublicationQueue publications;
    check(capture_image(publications, sink, 0, 1, 0, true, record, store.epoch()), "the older all-stage root is captured");
    settle(publications, store, sink);
    const uint64_t root_head = publications.owner(0).lineage.head;
    check(root_head != 0 && publications.owner(0).lineage.valid, "the older all-stage root becomes a durable predecessor");

    ConversationCache old_admission(budget, 1);
    SavedConversation old_parked = record.frozen_alias();
    check(old_admission.put(std::move(old_parked), size_t(store.ram_held_bytes())), "the old admission seam has a parked predecessor");
    check(!old_admission.reserve_snapshot(newest.bytes() * 2, size_t(store.ram_held_bytes())),
          "the old pre-enqueue full-image-plus-lease reservation really fails in this envelope");
    SavedConversation skipped = newest.frozen_alias();
    check(old_admission.put(std::move(skipped), size_t(store.ram_held_bytes())), "the skipped publication can still park normally");
    SavedConversation old_pressure = pressure.frozen_alias();
    check(old_admission.put(std::move(old_pressure), size_t(store.ram_held_bytes())), "unrelated RAM pressure evicts that unprotected newest image");
    const auto newest_query = query_of(newest, 310, 1);
    check_eq(old_admission.best(newest_query.tokens, newest_query.images, newest.cvec, newest.scope).tokens, 0,
             "the old refusal path loses the newest RAM-only cut under real cache pressure");
    check(old_admission.evictions() != 0, "the reproduction exercised actual cache eviction, not only an arithmetic helper");

    check(optional_shared_export({.enabled=true, .continues=true, .private_pin=true,
          .tokens=int64_t(newest.live.ids.size()), .restore_lower_bound_ms=4.0, .recompute_ms=2.0}).publish,
          "value policy never withholds the continued private newest/root/pin publication");
    check(capture_image(publications, sink, 0, 2, root_head, true, newest, store.epoch()),
          "the repaired transition queues the newer complete frontier without the second image reservation");
    check_eq(publications.queued().size(), 1, "the newer frontier has a bounded metadata owner");
    if (publications.queued().empty()) return;
    const uint64_t newest_ticket = publications.queued().front().image_ticket;
    check(!publications.queued().front().image && sink.cache.leased_bytes() == 0,
          "the newer frontier waits in its one already-budgeted parked owner");
    SavedConversation blocked_pressure = pressure.frozen_alias();
    check(!sink.cache.put(std::move(blocked_pressure), size_t(store.ram_held_bytes())),
          "unrelated pressure cannot evict the protected frontier before the writer seam");
    check(sink.cache.has_publication(newest_ticket), "pre-handoff pressure leaves the newest publication ticket intact");
    check(sink.cache.envelope_bytes() + store.ram_held_bytes() <= budget,
          "capturing the newer frontier preserves bounded cache plus store accounting");
    settle(publications, store, sink);
    check(!publications.holds_work(), "the newer frontier drains through the real safe-seam handoff and harvest");
    check_eq(publications.counters().published, 2, "both older and newest all-stage captures really persist");
    check_eq(publications.counters().refused, 0, "the repaired capture does not hide a pre-enqueue or store refusal");
    check_eq(publications.counters().dropped, 0, "RAM pressure does not drop the newer frontier");
    check_eq(store.counters().retained_branches, 1, "the pinned root promise remains a retained predecessor");
    check_eq(sink.cache.leased_bytes(), 0, "the accepted frontier leaves no caller-side lease");
    SavedConversation after_pressure = pressure.frozen_alias();
    check(sink.cache.put(std::move(after_pressure), size_t(store.ram_held_bytes())), "unrelated pressure is admitted after publication ownership transfers");
    check_eq(sink.cache.best(newest_query.tokens, newest_query.images, newest.cvec, newest.scope).tokens, 0,
             "the newest cut is no longer resident when persistence is exercised");
    check(sink.cache.envelope_bytes() + store.ram_held_bytes() <= budget,
          "the same bounded envelope still holds after pressure and writer drain");
    store.close();
    ConversationStore reopened(options);
    check(reopened.enabled(), "the bounded all-stage store reopens cleanly");
    if (!reopened.enabled()) return;
    check_record_restores(reopened, newest, 311);
    check_record_restores(reopened, record, 312, true);
}


// Storage/authority policy only: these are explicit small store fixtures, not
// claimed CUDA writes or a substitute for the real GEN-to-resident regression.
void check_ram_certification_is_independent_of_disk_export() {
    for (bool split : {false, true}) for (bool export_certification : {false, true})
    for (bool partial : {false, true}) {
        TempRoot root;
        const auto options = options_for(root.path);
        ConversationStore store(options);
        Sink sink(store);
        sink.out.export_draft_certification = export_certification;
        ConversationPublicationQueue publications;
        auto record = split ? all_stage_record(241, 64) : stored_record(241, 64);
        auto& draft_owner = split ? record.stage_images.back() : record;
        draft_owner.kv.push_back(draft_owner.kv.back());
        draft_owner.draft_begin = 0;
        draft_owner.draft_cells = partial ? 63 : 64;
        draft_owner.draft_next_token = partial ? -1 : 731;
        if (partial) {
            record.geometry[0] = 8;
            record.geometry[14] = 2;
            for (auto& stage : record.stage_images) stage.geometry = record.geometry;
            draft_owner.draft_boundary_residual.resize(64, 0x5a);
        }
        check(capture_image(publications, sink, 0, 1, 0, true, record, store.epoch()),
              "private RAM certification enters the real protected publication queue");
        settle(publications, store, sink);
        check(!publications.holds_work() && publications.owner(0).lineage.valid &&
              sink.cache.size() == 1, "publication commits and retains its private read-through RAM image");
        if (sink.cache.size() == 1) {
            bool retained = false;
            const auto ram = sink.cache.take(0, retained);
            const auto& owner = split ? ram.stage_images.back() : ram;
            check(owner.draft_begin == 0 && owner.draft_cells == (partial ? 63 : 64) &&
                  owner.draft_next_token == (partial ? -1 : 731),
                  "disk export policy never strips the RAM image's completed interval or boundary token");
            std::string why;
            check(images_equal(ram, record, why),
                  "both export policies preserve the RAM boundary row, checkpoints, pins and stage metadata");
        }
        check(draft_owner.draft_begin == 0 && draft_owner.draft_cells == (partial ? 63 : 64) &&
              draft_owner.draft_next_token == (partial ? -1 : 731) &&
              draft_owner.draft_boundary_residual.size() == (partial ? 64 : 0),
              "publication leaves the captured source interval and boundary intact");
        struct stat issuer {};
        check(::stat((root.path + "/issuer.key").c_str(), &issuer) == 0 &&
              issuer.st_size == (export_certification ? 64 : 32),
              "only certified disk export activates the minimum-reader issuer envelope");
        size_t manifests = 0;
        DIR* directory = ::opendir((root.path + "/manifests").c_str());
        check(directory != nullptr, "the committed manifest directory is readable");
        if (directory) {
            while (const auto* entry = ::readdir(directory)) {
                if (entry->d_name[0] == '.') continue;
                ++manifests;
                const auto path = root.path + "/manifests/" + entry->d_name;
                FILE* file = std::fopen(path.c_str(), "rb");
                char tag[8] = {};
                const bool read = file && std::fread(tag, 1, sizeof(tag), file) == sizeof(tag);
                if (file) std::fclose(file);
                check(read && std::memcmp(tag, export_certification ? "STCSMAN2" : "STCSMAN1", sizeof(tag)) == 0,
                      "the actual persisted manifest uses the explicitly selected certificate format");
            }
            ::closedir(directory);
        }
        check_eq(manifests, 1, "the publication wrote exactly one complete all-stage manifest");
        store.close();
        check(store.shutdown_state() == ConversationStoreShutdownState::durable_clean,
              "the publication root closes cleanly before a disk-only restore");
        ConversationStore reopened(options);
        auto query = query_of(record, 1201, 1);
        std::string why;
        check(reopened.demand(query, why), "the reopened store admits the persisted private image");
        ConversationStoreRead read;
        check(reopened.wait_ready(1201, 1, read, 5000) &&
              read.status == ConversationStoreReadStatus::ready && read.image &&
              read.image->stage_images.size() == (split ? 2 : 0),
              "disk-only restore returns the complete physical stages");
        if (read.image && read.image->stage_images.size() == (split ? 2 : 0)) {
            check(strata::program::conversation_store_restore_compatible(*read.image, false),
                  "a non-MTP engine accepts both MAN1 and MAN2 disk candidates");
            check(strata::program::conversation_store_restore_compatible(*read.image, true) == export_certification,
                  "an MTP engine declines MAN1 disk candidates and accepts certified MAN2");
            const auto& owner = split ? read.image->stage_images.back() : *read.image;
            check(owner.draft_begin == (export_certification ? 0 : -1) &&
                  owner.draft_cells == (export_certification ? partial ? 63 : 64 : -1) &&
                  owner.draft_next_token == (export_certification && !partial ? 731 : -1),
                  "ON restores the exported certificate; OFF disk restore remains explicitly unknown");
            check(export_certification
                      ? owner.draft_boundary_residual == draft_owner.draft_boundary_residual
                      : owner.draft_boundary_residual.empty(),
                  "certified disk export keeps the actual private N-1 boundary; the legacy alias omits it");
        }
        read.image.reset();
        reopened.release_read(1201, 1);
        reopened.close();
        check(reopened.shutdown_state() == ConversationStoreShutdownState::durable_clean,
              "the disk-only certificate policy check closes cleanly");
    }
}

void check_shared_borrow_and_owner_neutral_publication() {
    TempRoot root;
    ConversationStore store(options_for(root.path));
    Sink sink(store);
    ConversationPublicationQueue publications;
    SavedConversation private_image = stored_record(201, 64);
    check(capture_image(publications, sink, 0, 1, 0, true, private_image, store.epoch()),
          "the private predecessor is captured before shared publication");
    settle(publications, store, sink);
    const auto private_owner = publications.owner(0);
    const uint64_t private_head = private_owner.lineage.head;
    check(private_head != 0, "the private predecessor establishes its own lineage");

    SavedConversation shared = private_image;
    shared.scope.value[0] ^= 0x80;
    shared.checkpoints.clear();
    shared.live.pinned = true; // local retention must not pin the persistent shared head
    const auto enqueue_shared = [&](uint64_t owner) {
        SavedConversation image = shared;
        auto entry = capture_of(owner, 9999, private_head, true, image, store.epoch(), ++sink.next_ticket);
        entry.shared_prefix = true;
        entry.reuse.push_back({private_head, {0, uint16_t(ConversationStoreStreamKind::kv_k), 0, 0}, 0, 0, 64});
        return publications.capture(sink.cache, std::move(image), std::move(entry),
                                    size_t(store.ram_held_bytes()), 0, sink.out);
    };
    check(enqueue_shared(UINT64_MAX).queued, "a shared capture needs no physical owner slot");
    check(publications.owners().size() == 1 && publications.owner(0).generation == private_owner.generation &&
          publications.owner(0).lineage.head == private_head,
          "shared capture neither allocates an owner nor invalidates an unrelated private completion");
    check(!publications.queued().front().continues && !publications.queued().front().supersedes_hint &&
          publications.queued().front().reuse.empty(), "shared metadata cannot combine private donor lineage");
    check(enqueue_shared(0).queued, "a duplicate may wait before the first shared capture has committed");
    sink.refresh_allowance();
    settle(publications, store, sink);
    check(!publications.holds_work() && store.counters().published == 2 && store.counters().heads == 2,
          "the handoff publishes only one shared head and suppresses the queued duplicate");
    check(publications.owner(0).generation == private_owner.generation &&
          publications.owner(0).lineage.head == private_head, "shared completion is owner-neutral too");
    auto query = query_of(shared, 1001, 17);
    const auto matches = store.matches(query.selector, query.tokens, {});
    check(matches.size() == 1 && !matches.front().pinned && matches.front().live,
          "local checkpoint pinning never pins the persistent shared head");
    const uint64_t shared_head = matches.empty() ? 0 : matches.front().head_id;
    check(ConversationPublicationQueue::shared_prefix_exists(store, query.selector, query.tokens, 64, 1),
          "an exact shared selector suppresses duplicate capture without a payload read");
    auto changed = query;
    changed.tokens[5] ^= 1;
    check(!ConversationPublicationQueue::shared_prefix_exists(store, changed.selector, changed.tokens, 64, 1),
          "token divergence cannot suppress a different prefix");
    changed = query;
    changed.selector.scope.value[1] ^= 0x40;
    check(!ConversationPublicationQueue::shared_prefix_exists(store, changed.selector, changed.tokens, 64, 1),
          "another shared authorization cannot observe the stored selector");
    changed = query;
    changed.selector = query_of(private_image, 1002, 18).selector;
    const auto private_matches = store.matches(changed.selector, changed.tokens, {});
    check(!private_matches.empty() && private_matches.front().head_id == private_head,
          "the private selector stays independent of the equal shared record");

    std::string why;
    check(store.demand(query, why), "shared demand is admitted with its own IDs and no preceding offer");
    ConversationStoreRead read;
    check(store.wait_ready(1001, 17, read, 5000) && read.status == ConversationStoreReadStatus::ready &&
          read.image && read.head_id == shared_head, "direct shared demand restores its exact record");
    check(read.image && !read.image->live.pinned && read.image->checkpoints.empty(),
          "the restored shared record carries no private checkpoints or pin promise");
    read.image.reset();
    store.release_read(1001, 17);

    publications.owner(0).borrow();
    check(!publications.owner(0).lineage.valid && publications.owner(0).force_fresh,
          "shared borrowing forgets private ancestry and owes a fresh one-shot publication");
    ConversationPublicationPending old;
    old.owner = 0;
    old.owner_generation = private_owner.generation;
    old.scope = private_image.scope;
    old.live = private_image.live.ids;
    ConversationStorePublishResult old_result;
    old_result.status = ConversationStorePublishStatus::stored;
    old_result.head_id = private_head;
    old_result.epoch = store.epoch();
    check(!publications.adopt(old, old_result) && !publications.owner(0).lineage.valid,
          "a pre-borrow completion cannot reattach private lineage");
    SavedConversation after = turn_on(private_image, 80, 203);
    auto after_entry = capture_of(0, publications.owner(0).generation + 1, private_head, true,
                                  after, store.epoch(), ++sink.next_ticket);
    after_entry.reuse.push_back({private_head, {0, uint16_t(ConversationStoreStreamKind::kv_k), 0, 0}, 0, 0, 64});
    check(optional_shared_export({.enabled=true, .continues=false, .private_pin=true,
          .tokens=int64_t(after.live.ids.size()), .restore_lower_bound_ms=4.0, .recompute_ms=2.0}).publish,
          "post-shared force_fresh one-shot is private, never an optional shared export");
    check(publications.capture(sink.cache, std::move(after), std::move(after_entry),
                               size_t(store.ram_held_bytes()), 0, sink.out).queued,
          "the post-shared private state is captured even with a stale private donor hint");
    sink.refresh_allowance();
    check(!publications.queued().front().continues && !publications.queued().front().supersedes_hint &&
          publications.queued().front().reuse.empty(), "the next private publication is fresh, one-shot and donor-free");
    settle(publications, store, sink);
    check(!publications.owner(0).lineage.valid && !publications.owner(0).force_fresh,
          "the one-shot private result never adopts a lineage and settles its fresh-publication obligation");
    ConversationStoreManifestInfo info;
    check(store.manifest_info(private_head, info) && store.manifest_info(shared_head, info) &&
          store.counters().replaced == 0, "post-shared private publication supersedes neither private nor shared donors");
}

void check_shared_head_is_evictable() {
    TempRoot root;
    auto options = options_for(root.path);
    options.head_limit = 1;
    ConversationStore store(options);
    Sink sink(store);
    ConversationPublicationQueue publications;
    SavedConversation shared = stored_record(211, 64);
    shared.checkpoints.clear();
    shared.live.pinned = true;
    auto entry = capture_of(UINT64_MAX, 1, 0, true, shared, store.epoch(), ++sink.next_ticket);
    entry.shared_prefix = true;
    check(publications.capture(sink.cache, std::move(shared), std::move(entry),
                               size_t(store.ram_held_bytes()), 0, sink.out).queued,
          "an explicitly locally pinned shared cut is captured at the head ceiling");
    sink.refresh_allowance();
    settle(publications, store, sink);
    check(store.counters().heads == 1 && store.counters().pinned_heads == 0,
          "shared publication leaves its persistent head unpinned");
    const SavedConversation private_image = stored_record(213, 64);
    check(capture_image(publications, sink, 0, 1, 0, true, private_image, store.epoch()),
          "a private publication is admitted behind the shared head");
    settle(publications, store, sink);
    check(store.counters().heads == 1 && store.counters().evictions == 1 && publications.owner(0).lineage.valid,
          "real head pressure evicts the shared head rather than pinning capacity forever");
}

// Ordinary head pressure can remove an owner's latest without a replacement result. The first later capture still
// receives one accepted-then-stale verdict; that must not poison the queued capture or every future capture too.
void check_evicted_latest_does_not_strand_publications() {
    TempRoot root;
    auto options = options_for(root.path);
    options.head_limit = 2;
    ConversationStore store(options);
    if (!store.enabled()) return;
    Sink sink(store);
    ConversationPublicationQueue publications;
    const uint64_t epoch = store.epoch();
    SavedConversation initial = stored_record(231, 32);
    initial.checkpoints.clear();
    const SavedConversation record = turn_on(initial, 64, 231);
    check(capture_image(publications, sink, 0, 1, 0, true, initial, epoch), "the owner's initial cut is captured");
    settle(publications, store, sink);
    const uint64_t first = publications.owner(0).lineage.head;
    check(capture_image(publications, sink, 0, 2, first, true, record, epoch), "the owner's latest cut is captured");
    settle(publications, store, sink);
    const uint64_t evicted = publications.owner(0).lineage.head;
    check(evicted != 0 && evicted != first && store.counters().pinned_heads == 0,
          "a real completion adopts an unpinned latest while retaining the initial live-cut promise");
    const auto publish_pressure = [&](const SavedConversation& image) {
        const uint64_t ticket = ++sink.next_ticket;
        auto entry = capture_of(UINT64_MAX, ticket, 0, false, image, epoch, ticket);
        entry.shared_prefix = true;
        SavedConversation captured = image;
        check(publications.capture(sink.cache, std::move(captured), std::move(entry),
                                   size_t(store.ram_held_bytes()), 0, sink.out).queued,
              "an unpinned pressure cut is captured through the real queue");
        sink.refresh_allowance();
        settle(publications, store, sink);
    };
    SavedConversation older = stored_record(233, 64);
    older.checkpoints.clear();
    SavedConversation retained = stored_record(235, 64);
    retained.checkpoints.clear();
    publish_pressure(older);
    publish_pressure(retained);
    ConversationStoreManifestInfo info;
    check(!store.manifest_info(evicted, info) && store.counters().evictions == 2,
          "normal small-head-limit eviction removes the unpinned latest without a replacement result");
    check(publications.owner(0).lineage.valid && publications.owner(0).lineage.head == evicted,
          "the evicted latest initially remains in the owner's lineage");
    auto demand = query_of(retained, 810, 1);
    std::string why;
    check(store.demand(demand, why), "a distinct retained head has a real demand lease during recovery");
    ConversationStoreRead read;
    check(store.wait_ready(810, 1, read, 5000) && read.status == ConversationStoreReadStatus::ready && read.image,
          "the distinct head's object lease is handed out");
    publications.owner(1).mount(read.head_id, epoch, retained.live.ids, int64_t(retained.live.ids.size()),
                                retained.scope, retained.cvec);
    const uint64_t other_head = publications.owner(1).lineage.head;
    const uint64_t other_generation = publications.owner(1).generation;

    const SavedConversation failed = turn_on(record, 96, 231);
    const SavedConversation queued = turn_on(record, 128, 231);
    check(capture_image(publications, sink, 0, publications.owner(0).generation + 1, evicted, true, failed, epoch),
          "the first post-eviction capture is queued with its dangling hint");
    publications.flush(store, sink.out, 0);
    check(publications.queued().empty() && publications.pending().size() == 1 &&
              publications.pending().front().supersedes == evicted,
          "the first post-eviction capture is accepted once, not refused eight times at admission");
    check(capture_image(publications, sink, 0, publications.owner(0).generation + 1, evicted, true, queued, epoch),
          "a newer capture waits with the same dangling predecessor");
    publications.reconcile(store, sink.out, 5000);
    check_eq(publications.counters().refused, 1, "the accepted capture keeps its terminal stale refusal");
    check_eq(store.counters().refusals_stale, 1, "the writer reports exactly one stale publication");
    check(publications.pending().empty() && publications.queued().size() == 1 &&
              publications.queued().front().attempts == 0,
          "terminal staleness does not spend the queued capture's admission allowance");
    check(!publications.owner(0).lineage.valid && publications.queued().front().supersedes_hint == 0,
          "missing-predecessor recovery clears the dangling owner and queued hint");
    check(publications.owner(1).lineage.valid && publications.owner(1).lineage.head == other_head &&
              publications.owner(1).generation == other_generation && store.manifest_info(other_head, info),
          "recovery leaves the distinct owner's retained predecessor and generation alone");
    check(read.image && images_equal(*read.image, retained, why),
          "recovery does not revoke or mutate an active read/object lease");
    settle(publications, store, sink);
    check_eq(publications.counters().refused, 1, "the queued capture does not repeat the persistence failure");
    check(publications.owner(0).lineage.valid && publications.owner(0).lineage.head != evicted,
          "the queued valid capture establishes its own fresh lineage after recovery");
    read.image.reset();
    store.release_read(810, 1);
    const uint64_t recovered = publications.owner(0).lineage.head;
    const SavedConversation future = turn_on(queued, 160, 231);
    check(capture_image(publications, sink, 0, publications.owner(0).generation + 1,
                        publications.owner(0).lineage.head, true, future, epoch),
          "another future capture uses only the recovered lineage");
    settle(publications, store, sink);
    check_eq(publications.counters().refused, 1, "future captures no longer repeat the missing-head stale failure");
    check(publications.owner(0).lineage.head != recovered,
          "a subsequent valid capture also publishes rather than remaining stranded");
    store.close();
    ConversationStore reopened(options);
    check(reopened.enabled(), "the eviction-recovery catalog reopens");
    if (reopened.enabled()) check_record_restores(reopened, future, 811);
}

// Stale is not synonymous with evicted: an older/equal publication generation must keep naming the still-retained
// predecessor and must not silently become a fresh branch. Once numbering is valid, the original cover still works.
void check_retained_stale_predecessor_is_not_rebranched() {
    TempRoot root;
    ConversationStore store(options_for(root.path));
    if (!store.enabled()) return;
    Sink sink(store);
    ConversationPublicationQueue publications;
    const SavedConversation record = stored_record(237, 64);
    const uint64_t head = mount_latest(store, publications, sink, record);
    const SavedConversation continued = turn_on(record, 96, 237);
    const uint64_t published = publications.counters().published;
    sink.next_id = 0;   // two deliberately stale generations, both at or below the retained head's generation
    for (int i = 0; i < 2; ++i) {
        check(capture_image(publications, sink, 0, publications.owner(0).generation + 1,
                            head, true, continued, store.epoch()), "an older-generation cover is captured");
        settle(publications, store, sink);
        ConversationStoreManifestInfo info;
        check(store.manifest_info(head, info) && publications.owner(0).lineage.valid &&
                  publications.owner(0).lineage.head == head,
              "a retained stale predecessor is not cleared or rebranched");
    }
    check_eq(publications.counters().refused, 2, "both retained-predecessor stale refusals are preserved");
    check_eq(publications.counters().published, published, "neither stale capture creates an unauthorized branch");
    sink.next_id = store.publication_generation_floor();
    check(capture_image(publications, sink, 0, publications.owner(0).generation + 1,
                        head, true, continued, store.epoch()), "a correctly numbered cover is captured");
    settle(publications, store, sink);
    check_eq(publications.counters().published, published + 1, "a valid later capture still publishes");
    check_eq(store.counters().heads, 1, "the valid cover replaces the retained predecessor, not a new branch");
}

// Keep a real production pending entry unreconciled while other callers finish more publications than the result
// queue can hold. Backpressure must preserve that first completion, not quietly leave its owner pending forever.
void check_unreaped_results_preserve_production_pending() {
    TempRoot root;
    ConversationStoreOptions options = options_for(root.path);
    const size_t bound = size_t(options.max_pending_publishes) * 4 + 4;
    options.head_limit = bound + 2;
    ConversationStore store(options);
    Sink sink(store);
    const size_t fixed = size_t(16) << 20;
    check(sink.cache.try_reserve_lease(fixed), "the completion fixture reserves its resident chain at startup");
    sink.refresh_allowance();
    ConversationPublicationQueue publications;
    const SavedConversation record = stored_record(143, 32);
    check(capture_image(publications, sink, 0, 1, 0, true, record, store.epoch()),
          "a production capture is protected before the completion flood");
    publications.flush(store, sink.out, 0);
    check_eq(publications.pending().size(), 1, "the production capture owns a real pending publication");
    check_eq(sink.cache.leased_bytes(), fixed, "accepted handoff refunds only the transferred publication image");
    check_eq(sink.lease_releases, 1, "accepted handoff transfers the refundable image exactly once");
    const auto wait_idle = [&] {
        for (int waited = 0; waited < 5000 && store.writer_busy(); ++waited) ::usleep(1000);
        check(!store.writer_busy(), "the completion flood settles without reaping production's result");
    };
    wait_idle();
    auto foreign_image = std::make_shared<SavedConversation>(stored_record(145, 32));
    std::string why;
    const uint64_t deferred_before = store.counters().deferred_publishes;
    for (size_t i = 0; i < bound * 2; ++i) {
        ConversationStorePublish request;
        request.request_id = 1000 + i;
        request.generation = 2000 + i;
        request.epoch = store.epoch();
        const auto admission = store.publish(foreign_image, request, why);
        check(admission == (i < bound - 1 ? ConversationStorePublishAdmission::accepted
                                        : ConversationStorePublishAdmission::deferred),
              "foreign publications stop at the bounded outstanding-verdict capacity");
        wait_idle();
    }
    check_eq(store.counters().published, bound, "unreaped results impose a hard admission bound");
    check_eq(store.counters().deferred_publishes, deferred_before + bound + 1,
             "the overflow is explicit and counted rather than silently evicted");
    check_eq(publications.pending().size(), 1, "the original owner still awaits its terminal outcome");
    // Reconcile all batches, including foreign results: only this queue's matching pending entry may be counted.
    for (size_t i = 0; i < bound; ++i) publications.reconcile(store, sink.out, 0);
    check(!publications.holds_work(), "completion pressure cannot strand an admitted production pending entry");
    check_eq(publications.counters().published, 1, "the production result is accounted exactly once");
    check_eq(publications.counters().refused, 0, "foreign backpressure does not refuse the production publication");
    check_eq(publications.counters().dropped, 0, "completion pressure never drops the production capture");
    check(publications.owner(0).lineage.valid && publications.owner(0).lineage.head != 0,
          "the preserved completion installs its exact production owner's lineage");
    check_eq(publications.reconcile(store, sink.out, 0), 0, "all outcomes have been harvested without duplicates");
    check_eq(publications.counters().published, 1, "a second reconciliation cannot account the outcome twice");
    check_eq(sink.lease_releases, 1, "the production image charge transfers only once");
    check_eq(store.ram_lease_bytes(), 0, "completed flood publications retain no image lease");
    check_eq(sink.cache.leased_bytes(), fixed, "terminal reconciliation cannot consume the fixed resident chain");
    publications.abandon("completed queue abandoned", sink.out);
    publications.reconcile(store, sink.out, 0);
    check_eq(sink.cache.leased_bytes(), fixed, "cleanup after completion preserves the fixed resident chain");
    check_eq(sink.lease_releases, 1, "cleanup after completion cannot refund the transferred image again");
}

void check_phase_producer_survives_owner_reuse() {
    namespace phase = strata::program::phase;
    TempRoot root;
    ConversationStore store(options_for(root.path));
    Sink sink(store);
    ConversationPublicationQueue publications;
    std::vector<phase::PublicationEvent> events;
    sink.out.phase_event = [&](const phase::PublicationEvent& event) { events.push_back(event); };
    const auto producer = phase::parse_input("GEN 1 phase_request=producer phase_leg=1 "
        "phase_incarnation=0123456789abcdef0123456789abcdef 1,2").identity;
    const auto successor = phase::parse_input("GEN 1 phase_request=successor phase_leg=2 "
        "phase_incarnation=0123456789abcdef0123456789abcdef 1,2").identity;
    auto record = stored_record(191, 64);
    auto captured = record;
    auto entry = capture_of(0, 7, 0, true, record, store.epoch(), ++sink.next_ticket);
    entry.phase.producer = producer;
    check(publications.capture(sink.cache, std::move(captured), std::move(entry),
          size_t(store.ram_held_bytes()), 0, sink.out).queued, "the phase fixture queues one real capture");
    sink.refresh_allowance();
    publications.flush(store, sink.out, 0);
    check(!publications.pending().empty(), "the phase fixture retains the admitted result owner");
    publications.owner(0).borrow(); // completion must not attach to this successor
    publications.owner(0).phase_producer = successor;
    settle(publications, store, sink);
    check_eq(events.size(), 3, "capture, admission and terminal each have one phase observation");
    uint64_t publication = 0;
    for (const auto& event : events) {
        check(std::strcmp(event.publication.producer.request, "producer") == 0 &&
              event.publication.producer.leg == 1, "late publication keeps its producer, never the reused owner");
        check_eq(event.publication.capture_ticket, 1, "every publication phase retains the exact capture ticket");
        check_eq(event.publication.owner_generation, 7, "every publication phase retains the captured generation");
        if (std::strcmp(event.event, "admitted") == 0) {
            publication = event.publication.publication;
            check(publication != 0 && event.publication.admitted_ns >= event.publication.captured_ns,
                  "admission carries its own publication identity and chronological timestamps");
        }
        if (std::strcmp(event.event, "terminal") == 0)
            check(publication != 0 && event.publication.publication == publication && event.head != 0 &&
                  event.publication.applied_ns != 0 && event.publication.catalog_version != 0,
                  "only real reconciliation supplies the matching terminal head and catalog visibility boundary");
    }
    check(!publications.owner(0).lineage.valid, "diagnostic correlation never bypasses the ownership generation guard");
    check_eq(publications.reconcile(store, sink.out, 0), 0, "the phase observer did not consume or duplicate a result");

    captured = record;
    entry = capture_of(0, 8, 0, true, record, store.epoch(), ++sink.next_ticket);
    entry.phase.producer = successor;
    check(publications.capture(sink.cache, std::move(captured), std::move(entry),
          size_t(store.ram_held_bytes()), 0, sink.out).queued, "a successor capture queues independently");
    publications.drop(0, "cancelled before handover", sink.out);
    check(std::strcmp(events.back().event, "terminal") == 0 &&
          std::strcmp(events.back().status, "dropped") == 0 &&
          std::strcmp(events.back().publication.producer.request, "successor") == 0,
          "cancel/drop emits one terminal for the capture it actually released");
    // A failed diagnostic destination must not change persistence or ownership.
    sink.out.phase_event = [](const phase::PublicationEvent&) { throw std::bad_alloc(); };
    const auto independent = stored_record(193, 64);
    check(capture_image(publications, sink, 0, 9, 0, true, independent, store.epoch()),
          "an observer failure cannot refuse a real capture");
    settle(publications, store, sink);
    check(!publications.holds_work(), "observer failure cannot strand publication ownership");
    check_eq(publications.counters().published, 2, "observer failure does not change successful publication results");
}

// The production queue/store, not synthetic publication outcomes: sixteen
// protected captures each spend eight real synchronous refusals before drop.
void check_phase_terminals_survive_trace_saturation() {
    namespace phase = strata::program::phase;
    static_assert(phase::Receipts::max_publication_terminals >=
                  ConversationPublicationQueue::kMaxQueued + ConversationPublicationQueue::kMaxPending);
    TempRoot root;
    ConversationStore store(options_for(root.path));
    Sink sink(store);
    ConversationPublicationQueue publications;
    auto receipts = std::make_unique<phase::Receipts>(&phase_report);
    phase_lines.clear();
    const auto executor = phase::parse_input("GEN 1 phase_request=executor phase_leg=2 "
        "phase_incarnation=0123456789abcdef0123456789abcdef 1,2").identity;
    sink.out.phase_event = [&](const phase::PublicationEvent& event) { receipts->publication(event, executor); };
    const auto drain_terminals = [&] {
        if (receipts->terminal_size()) receipts->drain(); // same post-operation seam as the engine
    };
    for (size_t i = 0; i < ConversationPublicationQueue::kMaxQueued; ++i) {
        auto image = stored_record(191 + i * 2, 64);
        auto entry = capture_of(i, 100 + i, 0, true, image, store.epoch(), ++sink.next_ticket);
        entry.phase.producer = phase::parse_input("GEN 1 phase_request=publisher-" + std::to_string(i) +
            " phase_leg=1 phase_incarnation=0123456789abcdef0123456789abcdef 1,2").identity;
        check(publications.capture(sink.cache, std::move(image), std::move(entry),
              size_t(store.ram_held_bytes()), 0, sink.out).queued, "saturation fixture queues a real protected capture");
        drain_terminals();
    }
    for (size_t i = 0; i < phase::Receipts::max_trace; ++i) receipts->trace("expendable trace\n");
    store.publish_read_allowance(1); // deterministic genuine admission refusal, never a fake result
    for (size_t i = 0; i <= ConversationPublicationQueue::kMaxQueued * ConversationPublicationQueue::kMaxAttempts &&
                       publications.holds_work(); ++i) {
        const size_t before = phase_lines.size();
        publications.flush(store, sink.out, 0);
        check_eq(phase_lines.size(), before, "the publication observer never performs receipt I/O");
        drain_terminals();
    }
    check(!publications.holds_work(), "all permanently refused captures reach actual terminal drop");
    check_eq(publications.counters().refused,
             ConversationPublicationQueue::kMaxQueued * ConversationPublicationQueue::kMaxAttempts,
             "all eight attempts of each real capture were refused");
    check_eq(publications.counters().dropped, ConversationPublicationQueue::kMaxQueued,
             "every protected capture retires exactly once");
    check(receipts->dropped() > 0, "the expendable trace really saturated");
    check_eq(receipts->terminal_overflow(), 0, "the per-operation terminal bound was sufficient");
    for (size_t i = 0; i < ConversationPublicationQueue::kMaxQueued; ++i) {
        const std::string producer = "kind=publication request=publisher-" + std::to_string(i) + " leg=1 ";
        size_t terminals = 0;
        for (const auto& line : phase_lines) {
            if (line.find(producer) == std::string::npos ||
                line.find("event=terminal status=dropped ") == std::string::npos) continue;
            ++terminals;
            check(line.find("capture_ticket=" + std::to_string(i + 1) + " ") != std::string::npos &&
                  line.find("publication=na") == std::string::npos,
                  "saturated terminal retains its exact capture and real publication identity");
            check(line.find("owner=" + std::to_string(i) + " owner_generation=" + std::to_string(100 + i) + " ") !=
                  std::string::npos, "saturated terminal retains its captured owner generation");
            check(line.find("head=na attempts=8 ") != std::string::npos &&
                  line.find("executor_request=executor executor_leg=2 ") != std::string::npos,
                  "saturated terminal reports the actual exhausted verdict and current executor");
        }
        check_eq(terminals, 1, "trace saturation loses or duplicates no publication terminal");
    }
}


// Smallest real ownership/store fixture for the mutation evidence. The optional
// file is a diagnostic CPU capture, never a model/client latency benchmark.
void check_cache_mutation_publication(const char* path = nullptr) {
    TempRoot root;
    ConversationStore store(options_for(root.path));
    Sink sink(store, size_t(64) << 20, 2);
    FILE* file = path ? std::fopen(path, "w+x") : std::tmpfile();
    check(file != nullptr, "create the publication mutation trace without replacing evidence");
    if (!file) return;
    {
        ConversationCacheTrace trace(file);
        trace.request_sequence = 1;
        std::strcpy(trace.request, "cpu-publication");
        std::strcpy(trace.incarnation, "0123456789abcdef0123456789abcdef");
        trace.leg = 1;
        const size_t fixed = size_t(1) << 20;
        trace.set_sample(&store, +[](void* context, ConversationCacheTrace::External& e) {
            e.checkpoint_lease = size_t(1) << 20;
            e.store = static_cast<ConversationStore*>(context)->ram_trace();
            check(e.store.held == e.store.payload + e.store.metadata + e.store.descriptors + e.store.decode,
                  "the trace samples one coherent store ledger");
        });
        sink.cache.set_mutation_trace(&trace);
        check(sink.cache.try_reserve_lease(fixed), "the trace fixture charges fixed resident checkpoints");
        sink.refresh_allowance();
        ConversationPublicationQueue publications;
        std::vector<SavedConversation> records;
        for (int i = 0; i < 4; ++i) {
            records.push_back(stored_record(241 + 2 * i, 64 + 32 * i));
            records.back().checkpoints.clear();
        }
        for (size_t i = 0; i < records.size(); ++i) {
            trace.request_sequence = i + 1;
            const auto start = std::chrono::steady_clock::now();
            check(capture_image(publications, sink, i, 1, 0, true, records[i], store.epoch()),
                  "the bounded fixture captures through the production publication queue");
            settle(publications, store, sink);
            sink.cache.trace_observation({.site="fixture", .action="cpu_capture_publication_complete",
                .measured_ms=std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start).count(), .end_tokens=(int64_t)records[i].live.ids.size()});
            if (i == 0) {
                const auto match = sink.cache.best(query_of(records[0], 1, 1).tokens, {}, true, records[0].scope);
                check(match.tokens == int64_t(records[0].live.ids.size()), "the same first image has a real RAM return");
                const auto ram_start = std::chrono::steady_clock::now();
                bool retained = false;
                auto image = sink.cache.take(match.index, retained);
                std::string why;
                check(!retained && images_equal(image, records[0], why), "RAM return preserves the exact first image");
                check(sink.cache.put(std::move(image)), "RAM return re-admits the same first image");
                sink.cache.trace_observation({.site="ram_return", .action="cpu_owner_roundtrip_complete",
                    .subject_serial=match.serial,
                    .measured_ms=std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - ram_start).count(), .end_tokens=match.tokens});
            }
        }
        check_eq(sink.cache.evictions(), 2, "four real images force two RAM evictions at the unchanged two-slot budget");
        check_eq(sink.acquisitions, 4, "every publication transfers one protected owner");
        check_eq(sink.retentions, 4, "every publication exercises its optional same-image RAM return");
        check_eq(sink.lease_releases, 4, "each acquired publication charge is refunded once");
        // Both images displaced by the finite pressure suffix actually return.
        // Materialization/I/O is real; GPU mount, fresh prefill and HTTP are absent.
        std::vector<size_t> materialized_charges;
        for (size_t i = 0; i < 2; ++i) {
            trace.request_sequence = 5 + i;
            auto query = query_of(records[i], 1001 + i, 1);
            const auto before = store.counters();
            const auto start = std::chrono::steady_clock::now();
            std::string why;
            check(store.demand(query, why), "a named displaced image is demandable from the real store");
            ConversationStoreRead read;
            check(store.wait_ready(query.request_id, query.generation, read, 5000) &&
                  read.status == ConversationStoreReadStatus::ready && read.image,
                  "the displaced store return materializes within its finite deadline");
            if (!read.image) continue;
            auto expected = records[i];
            expected.live.pinned = true; // the queue's mandatory fresh-branch root promise, not a RAM pin
            check(images_equal(*read.image, expected, why), "the store return preserves the displaced image and root pin");
            std::optional<SavedConversation> incoming;
            strata::program::conversation_store_mount_alias(read, incoming, &sink.cache);
            check(!read.image && incoming && images_equal(*incoming, expected, why),
                  "production handover preserves the incoming image after resetting the store root");
            materialized_charges.push_back(incoming->bytes());
            sink.cache.trace_observation({.site="store_return", .action="cpu_materialized",
                .head=read.head_id, .measured_ms=read.materialize_ms,
                .work_bytes=store.counters().bytes_read - before.bytes_read, .end_tokens=read.tokens}, &*incoming);
            sink.cache.trace_observation({.site="store_return", .action="cpu_demand_roundtrip_complete",
                .head=read.head_id, .measured_ms=std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start).count(), .end_tokens=read.tokens});
            incoming.reset();
            store.release_read(query.request_id, query.generation);
            sink.cache.trace_observation({.site="store_return", .action="read_released"});
        }
        sink.cache.set_mutation_trace(nullptr);
        trace.finish();
        std::fflush(file);
        const long bytes = std::ftell(file);
        std::rewind(file);
        std::string text(size_t(std::max(bytes, 0l)), '\0');
        check(std::fread(text.data(), 1, text.size(), file) == text.size(), "read the complete publication mutation trace");
        const auto take = text.find("\"site\":\"publication\",\"action\":\"take\"");
        const auto take_end = text.find('\n', take);
        const auto returned = text.find("\"site\":\"put\",\"action\":\"plan\"", take_end);
        check(take != std::string::npos && returned != std::string::npos &&
              text.find("\"action\":\"release\"", take_end) < returned,
              "publication take transfers its lease before optional put/return");
        check(text.find("\"subject_serial\":1", take) < take_end &&
              text.find("\"checkpoint_lease\":1048576", take) < take_end &&
              text.find("\"request\":\"cpu-publication\",\"leg\":1") != std::string::npos &&
              text.find("\"publication_ticket\":1") != std::string::npos &&
              text.find("\"history_complete\":true") != std::string::npos &&
              text.find("\"read_image\":0") != std::string::npos,
              "publication take/return retain serial, request, ticket, identity and exact charge fields");
        size_t materialized_at = 0;
        for (size_t charge : materialized_charges) {
            materialized_at = text.find("\"site\":\"store_return\",\"action\":\"materialized\"", materialized_at);
            check(materialized_at != std::string::npos, "the production handover emits its materialization receipt");
            if (materialized_at == std::string::npos) break;
            const auto end = text.find("{\"kind\":\"cache_event_end\"", materialized_at);
            const auto event = text.substr(materialized_at, end - materialized_at);
            check(event.find("\"subject_present\":true,\"subject_complete\":true") != std::string::npos &&
                  event.find("\"role\":\"subject\"") != std::string::npos &&
                  event.find("\"bytes\":" + std::to_string(charge) + ",") != std::string::npos &&
                  event.find("\"scope_hash\":") != std::string::npos &&
                  event.find("\"history_complete\":true") != std::string::npos &&
                  event.find("\"checkpoint_bytes\":") != std::string::npos,
                  "materialization retains scope/history/image/checkpoint charges after the real alias/reset ordering");
            materialized_at = end;
        }
        check(text.find("\"limited\":false,\"hash_limited\":false") != std::string::npos,
              "the forced-eviction evidence has explicit complete closure");
        check_eq(sink.cache.leased_bytes(), fixed, "all returns leave the mandatory fixed checkpoint lease intact");
        check_eq(store.ram_lease_bytes(), 0, "all displaced reads release their actual store leases");
    }
    std::fclose(file);
}

strata::program::PublicationSubmission facade_capture(ContinuationPublisher& publisher, Sink& sink,
    uint64_t owner, uint64_t generation, const SavedConversation& record, uint64_t supersedes = 0,
    std::vector<ConversationStoreReuse> reuse = {}) {
    auto image = record;
    auto entry = capture_of(owner, generation, supersedes, true, record, sink.store.epoch(), ++sink.next_ticket);
    entry.reuse = std::move(reuse);
    const auto submission = publisher.capture(sink.cache, std::move(image), std::move(entry),
                                              size_t(sink.store.ram_held_bytes()), 0);
    sink.refresh_allowance();
    return submission;
}

void settle(ContinuationPublisher& publisher) {
    for (int waited = 0; waited < 1000 && publisher.holds_work(); ++waited)
        publisher.flush(20);
    check(!publisher.holds_work(), "facade work settles within the bounded CPU fixture");
}
void check_optional_shared_export_parks_without_capture() {
    TempRoot root;
    auto image = stored_record(321, 64);
    image.checkpoints.clear();
    image.freeze();
    const size_t image_bytes = image.bytes();
    ConversationStore store(options_for(root.path));
    const size_t held = size_t(store.ram_held_bytes());
    // Enough for the ordinary immutable image, deliberately not the production
    // 64 MiB protected-publication headroom.
    Sink sink(store, held + image_bytes + 4096, 2, size_t(64) << 20);
    ContinuationPublisher publisher;
    publisher.configure(&store, std::move(sink.out));
    const auto before = store.counters();
    const auto decision = optional_shared_export({.enabled=true, .shared_prefix=true, .continues=false,
        .private_pin=true, .tokens=64, .restore_lower_bound_ms=4.0, .recompute_ms=2.0});
    bool parked = false;
    if (decision.publish) {
        auto entry = capture_of(UINT64_MAX, 1, 0, false, image, store.epoch(), ++sink.next_ticket);
        entry.shared_prefix = true;
        parked = publisher.capture(sink.cache, std::move(image), std::move(entry), held, 0).parked;
    } else {
        ++publisher.counters().optional_skips;
        sink.refresh_allowance();
        parked = sink.cache.put(std::move(image), held);
    }
    check(parked && sink.cache.size() == 1, "skipped durable export still takes the ordinary RAM park path");
    check_eq(sink.cache.envelope_bytes(), image_bytes, "ordinary park charges only its image, not 64 MiB headroom");
    check_eq(sink.cache.evictions(), 0, "ordinary optional park spends no other conversation");
    check_eq(sink.next_id, 0, "skip never mints a publication id");
    check_eq(sink.next_ticket, 0, "skip precedes capture ticket allocation");
    check_eq(sink.acquisitions, 0, "skip never acquires an image lease");
    check_eq(sink.cache.leased_bytes(), 0, "ordinary parked bytes are not leased publication bytes");
    sink.cache.for_each_image([&](size_t, const auto& saved) {
        check(saved.publication_ticket == 0, "RAM export has no protected publication ticket");
        auto prompt = saved.live.ids;
        prompt.push_back(98765);
        const auto selected = sink.cache.best(prompt, {}, saved.cvec, saved.scope);
        check(selected.tokens == 64, "shared selector still finds the RAM-only exact prefix");
    });
    const auto ready = publisher.readiness();
    check(!publisher.holds_work() && !ready.outstanding && !ready.receipts,
          "skip reserves neither facade ownership nor a terminal receipt");
    const auto after = store.counters();
    check(after.heads == before.heads && after.objects == before.objects && after.manifests == before.manifests &&
          after.bytes_written == before.bytes_written, "skip creates no head, object, manifest or durable bytes");
    check_eq(publisher.counters().optional_skips, 1, "only the explicit optional skip counter advances");
    check(!publisher.counters().published && !publisher.counters().refused &&
          !publisher.counters().coalesced && !publisher.counters().dropped,
          "skip is neither a refusal, coalescing nor a durability drop");
    store.close();
    ConversationStore reopened(options_for(root.path));
    check(reopened.enabled() && reopened.counters().heads == 0, "clean reopen invents no skipped shared head");
    reopened.close();
}


void check_facade_receipts_and_identity() {
    TempRoot root;
    ConversationStore store(options_for(root.path));
    Sink sink(store);
    ContinuationPublisher publisher;
    publisher.configure(&store, std::move(sink.out));
    const auto ready = publisher.readiness();
    check(ready.enabled && ready.healthy && ready.run_intent && ready.epoch == store.epoch() &&
          !ready.outstanding && !ready.last_error[0], "persistence readiness observes only real store state");
    auto first = stored_record(301, 512);
    first.checkpoints.clear();
    const auto a = facade_capture(publisher, sink, 0, 11, first);
    check(a.queued && a.handle && a.status == PublicationStatus::queued,
          "certified-image handoff returns a stable generation-bound handle");
    check(a.handle.owner == 0 && a.handle.owner_generation == 11 &&
          a.handle.capture_epoch == store.epoch(), "the handle retains the capture's own owner and epoch");
    PublicationReceipt receipt;
    check(!publisher.take_receipt(a.handle, receipt), "queued ownership is not a completed publication");
    publisher.flush(0);
    publisher.owner(0).borrow(); // an unrelated successor must not receive this completion's lineage
    settle(publisher);
    auto stale = a.handle;
    ++stale.owner_generation;
    check(!publisher.take_receipt(stale, receipt), "a mismatched generation cannot consume a terminal receipt");
    stale = a.handle;
    ++stale.owner;
    check(!publisher.take_receipt(stale, receipt), "a mismatched physical owner cannot consume a terminal receipt");
    stale = a.handle;
    ++stale.capture_epoch;
    check(!publisher.take_receipt(stale, receipt), "a mismatched epoch cannot consume a terminal receipt");
    check(publisher.take_receipt(a.handle, receipt) && receipt.handle == a.handle &&
          receipt.status == PublicationStatus::stored && receipt.head && receipt.epoch == a.handle.capture_epoch &&
          receipt.new_bytes > 0 && !receipt.reused_bytes && receipt.released,
          "the receipt names this publication's actual head, bytes and ownership release");
    const uint64_t first_head = receipt.head;
    check(!publisher.owner(0).lineage.valid, "receipt correlation never bypasses authoritative generation adoption");
    check(!publisher.take_receipt(a.handle, receipt), "terminal receipts are consumed exactly once");

    publisher.owner(0).mount(first_head, store.epoch(), first.live.ids, 512, first.scope, first.cvec);
    const auto grown = turn_on(first, 544, 301);
    // A whole aligned chunk of unchanged K is proven by this fixture's append-only construction.
    ConversationStoreReuse reused{first_head, {0, uint16_t(ConversationStoreStreamKind::kv_k), 0, 0}, 0, 0, 4096};
    const auto b = facade_capture(publisher, sink, 0, 20, grown, first_head, {reused});
    const auto peer = facade_capture(publisher, sink, 1, 21, stored_record(303, 64));
    settle(publisher);
    check(publisher.take_receipt(peer.handle, receipt) && receipt.handle == peer.handle &&
          receipt.status == PublicationStatus::stored && receipt.head != first_head,
          "a peer result is selected by its identity, not by a global latest head");
    check(publisher.take_receipt(b.handle, receipt) && receipt.handle == b.handle &&
          receipt.status == PublicationStatus::retained_branch && receipt.reused_bytes == 4096,
          "the facade preserves root retention and reports the real donor bytes");
    check_eq(publisher.owner(0).lineage.head, receipt.head, "existing authoritative adoption still owns lineage");
    check_eq(store.counters().heads, 3, "the unchanged pinned-root/latest rule retains root, latest and peer");
    check_eq(publisher.readiness().outstanding, 0, "consuming every receipt reopens all bounded capacity");
    store.close();
    const auto closed = publisher.readiness();
    check(!closed.enabled && !closed.healthy && !closed.run_intent,
          "clean shutdown is not confused with live persistence readiness");
    ConversationStore reopened(options_for(root.path));
    check_record_restores(reopened, first, 3001, true);
    check_record_restores(reopened, grown, 3002);
    reopened.close();
}

void check_facade_cancel_with_consumers() {
    TempRoot root;
    ConversationStore store(options_for(root.path));
    Sink sink(store, size_t(64) << 20, 3);
    ContinuationPublisher publisher;
    publisher.configure(&store, std::move(sink.out));
    const size_t fixed = size_t(1) << 20;
    check(sink.cache.try_reserve_lease(fixed), "a peer retains its existing fixed consumer reservation");
    auto record = stored_record(307, 64);
    record.checkpoints.clear();
    const auto cancelled = facade_capture(publisher, sink, 0, 1, record);
    bool retained = false;
    auto consumer = sink.cache.take(0, retained); // a real frozen alias of the protected owner
    check(retained && sink.cache.has_publication(sink.next_ticket), "the consumer borrows without taking capture ownership");
    auto wrong = cancelled.handle;
    ++wrong.owner_generation;
    check(!publisher.cancel(wrong), "stale cancellation does not release another generation's ticket");
    check(publisher.cancel(cancelled.handle), "exact cancellation retires a not-yet-admitted capture");
    PublicationReceipt receipt;
    check(publisher.take_receipt(cancelled.handle, receipt) && receipt.status == PublicationStatus::dropped &&
          receipt.released && !receipt.head, "queued cancellation has one ownership-complete receipt");
    std::string why;
    check(images_equal(consumer, record, why), "cancellation leaves outstanding immutable consumers intact");
    check_eq(sink.cache.leased_bytes(), fixed, "cancellation refunds neither the peer nor a consumer lease");
    check_eq(sink.ticket_releases, 1, "queued cancellation releases protection exactly once");
    check(!publisher.cancel(cancelled.handle), "repeated cancellation releases nothing twice");
    const auto peer = facade_capture(publisher, sink, 1, 2, stored_record(309, 64));
    publisher.flush(0);
    check(publisher.cancel(peer.handle), "admitted cancellation requests no premature buffer release");
    check(!publisher.take_receipt(peer.handle, receipt), "admitted work remains outstanding until real reconciliation");
    settle(publisher);
    check(publisher.take_receipt(peer.handle, receipt) && receipt.released &&
          (receipt.status == PublicationStatus::stored || receipt.status == PublicationStatus::stale),
          "admitted cancellation reports the real writer outcome, not a fabricated interrupted write");
    check(images_equal(consumer, record, why), "the outstanding consumer remains valid after peer settlement");
    check_eq(sink.cache.leased_bytes(), fixed, "the fixed consumer reservation survives all publication terminals");
    store.close();
}

void check_facade_optional_failure_and_peer() {
    TempRoot root;
    ConversationStore store(options_for(root.path));
    Sink sink(store);
    ContinuationPublisher publisher;
    publisher.configure(&store, std::move(sink.out));
    store.set_publish_fault_for_test("objects");
    const auto failed = facade_capture(publisher, sink, 0, 1, stored_record(311, 64));
    const auto peer = facade_capture(publisher, sink, 1, 2, stored_record(313, 64));
    settle(publisher);
    PublicationReceipt receipt;
    check(publisher.take_receipt(failed.handle, receipt) && receipt.status == PublicationStatus::failed &&
          !receipt.head && receipt.released, "a genuine optional object-write failure drops only its capture");
    check(!publisher.owner(0).lineage.valid, "failed publication establishes no private lineage");
    check(publisher.take_receipt(peer.handle, receipt) && receipt.status == PublicationStatus::stored && receipt.head &&
          receipt.released, "the independently queued peer survives the failed optional write");
    const auto ready = publisher.readiness();
    check(ready.enabled && ready.healthy && ready.run_intent && ready.last_error[0] && !ready.outstanding,
          "last optional error is visible without falsely faulting a healthy persistence module");
    check_eq(sink.cache.leased_bytes(), 0, "ordinary failure and surviving peer release both transferred leases");
    check_eq(publisher.counters().refused, 1, "the failed optional write has exactly one terminal verdict");
    check_eq(publisher.counters().published, 1, "only the surviving peer counts as published");
    store.close();
    ConversationStore reopened(options_for(root.path));
    check_record_restores(reopened, stored_record(313, 64), 3101, true);
    reopened.close();
}

void check_facade_bounded_receipts() {
    TempRoot root;
    ConversationStore store(options_for(root.path));
    Sink sink(store);
    ContinuationPublisher publisher;
    publisher.configure(&store, std::move(sink.out));
    std::vector<PublicationHandle> handles;
    for (size_t i = 0; i < ContinuationPublisher::kMaxOutstanding; ++i) {
        const auto submission = facade_capture(publisher, sink, 0, i + 1, stored_record(315, 64));
        check(submission.queued, "the receipt bound accepts one protected capture at a time");
        handles.push_back(submission.handle);
        check(publisher.cancel(submission.handle), "cancellation fills one pre-reserved terminal slot");
    }
    const auto full = publisher.readiness();
    check_eq(full.receipts, ContinuationPublisher::kMaxOutstanding, "unconsumed terminals count toward the outstanding bound");
    const auto refused = facade_capture(publisher, sink, 0, 99, stored_record(317, 64));
    check(!refused.handle && !refused.queued && refused.parked && refused.status == PublicationStatus::backpressure,
          "completion pressure refuses ownership synchronously and preserves ordinary RAM parking");
    PublicationReceipt receipt;
    for (const auto& handle : handles)
        check(publisher.take_receipt(handle, receipt) && receipt.handle == handle &&
              receipt.status == PublicationStatus::dropped, "backpressure loses no existing terminal identity");
    check_eq(publisher.readiness().outstanding, 0, "draining terminals releases the entire fixed outcome bound");

    // Local interface overhead only: no disk I/O or model work inside these diagnostic loops.
    constexpr size_t iterations = 10000;
    uint64_t observed = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < iterations; ++i) observed += store.readiness().epoch;
    const auto t1 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < iterations; ++i) observed += publisher.readiness().epoch;
    const auto t2 = std::chrono::steady_clock::now();
    check(observed == 2 * iterations * store.epoch(), "readiness overhead loop consumes every real observation");
    std::fprintf(stderr, "publisher interface: queue_bytes=%zu facade_bytes=%zu fixed_delta=%zu outcomes=%zu "
        "store_readiness_ns=%.1f facade_readiness_ns=%.1f dynamic_receipt_allocations=0 payload_copies=0\n",
        sizeof(ConversationPublicationQueue), sizeof(ContinuationPublisher),
        sizeof(ContinuationPublisher) - sizeof(ConversationPublicationQueue), ContinuationPublisher::kMaxOutstanding,
        std::chrono::duration<double, std::nano>(t1 - t0).count() / iterations,
        std::chrono::duration<double, std::nano>(t2 - t1).count() / iterations);
    store.close();
}

void check_store_readiness_fault_reason() {
    TempRoot root;
    ConversationStore store(options_for(root.path));
    Sink sink(store);
    ContinuationPublisher publisher;
    publisher.configure(&store, std::move(sink.out));
    store.set_unhealthy("the owner observed poisoned shared state");
    const auto faulted = store.readiness();
    check(faulted.enabled && !faulted.healthy && faulted.run_intent,
          "an explicitly latched fault remains distinct from a disabled or cleanly closed store");
    check(std::strcmp(faulted.last_error.data(), "the owner observed poisoned shared state") == 0,
          "store readiness exposes the authoritative set_unhealthy cause");
    check(std::strcmp(publisher.readiness().last_error.data(), "the owner observed poisoned shared state") == 0,
          "the facade forwards a latched owner fault without needing a failed publication receipt");
    store.set_unhealthy("a later unrelated failure");
    check(std::strcmp(store.readiness().last_error.data(), "the owner observed poisoned shared state") == 0,
          "later observations cannot overwrite the first latched store fault");
    store.close();
}

void check_facade_uncertain_publication() {
    TempRoot root;
    uint64_t epoch = 0;
    const auto record = stored_record(319, 64);
    {
        ConversationStore store(options_for(root.path));
        Sink sink(store);
        ContinuationPublisher publisher;
        publisher.configure(&store, std::move(sink.out));
        epoch = store.epoch();
        store.set_publish_fault_for_test("catalog-sync");
        const auto submission = facade_capture(publisher, sink, 0, 1, record);
        settle(publisher);
        PublicationReceipt receipt;
        check(publisher.take_receipt(submission.handle, receipt) && receipt.status == PublicationStatus::failed &&
              !receipt.head && receipt.released, "uncertain catalog durability produces no successful receipt");
        const auto ready = publisher.readiness();
        check(!ready.healthy && ready.run_intent && ready.last_error[0],
              "authority uncertainty is exposed as faulted persistence with armed intent");
        const auto store_ready = store.readiness();
        check(!store_ready.healthy && store_ready.run_intent &&
              std::strstr(store_ready.last_error.data(), "durability is not confirmed") != nullptr,
              "store readiness itself exposes catalog uncertainty, without the facade's last-error fallback");
        store.close();
        check(store.shutdown_state() == ConversationStoreShutdownState::quiescent_unclean,
              "the facade cannot turn an uncertain publication into a clean shutdown");
    }
    ConversationStore reopened(options_for(root.path));
    check_eq(reopened.epoch(), epoch + 1, "existing interrupted-run recovery advances the execution epoch");
    const auto query = query_of(record, 3201, 1);
    check(reopened.matches(query.selector, query.tokens, query.images).empty(),
          "an uncertain publication cannot become an authorized continuation on restart");
    reopened.close();
}

#ifdef STRATA_STORE_TEST_HOOKS
namespace phase = strata::program::phase;

// Explicit diagnostic only, never part of the default test run. The nine inputs approximate the retained
// recurrent-state sizes, not its model state, exact K/V shape, alias topology or arrival schedule.
struct WriterDiagnostic {
    struct Row {
        phase::Publication publication;
        ConversationStorePublishTrace writer;
        PublicationReceipt receipt;
        uint64_t capture_started_ns = 0, capture_finished_ns = 0, admitted_observed_ns = 0;
        uint64_t terminal_ns = 0, receipt_ns = 0;
        bool writer_seen = false, receipt_seen = false;
    };
    std::mutex mutex;
    std::array<Row, 9> rows;
    size_t omitted = 0;

    void observe(const phase::PublicationEvent& event) noexcept {
        std::lock_guard<std::mutex> guard(mutex);
        const uint64_t id = event.publication.publication;
        if (!id || id > rows.size()) { ++omitted; return; }
        auto& row = rows[size_t(id - 1)];
        row.publication = event.publication;
        if (std::strcmp(event.event, "admitted") == 0) row.admitted_observed_ns = event.observed_ns;
        if (std::strcmp(event.event, "terminal") == 0) row.terminal_ns = event.observed_ns;
    }
    void observe(const ConversationStorePublishTrace& trace) noexcept {
        std::lock_guard<std::mutex> guard(mutex);
        if (!trace.request_id || trace.request_id > rows.size()) { ++omitted; return; }
        auto& row = rows[size_t(trace.request_id - 1)];
        row.writer = trace;
        row.writer_seen = true;
    }
    void drain(ContinuationPublisher& publisher) {
        PublicationReceipt receipt;
        while (publisher.take_receipt(receipt)) {
            std::lock_guard<std::mutex> guard(mutex);
            const uint64_t id = receipt.handle.publication_id;
            if (!id || id > rows.size()) { ++omitted; continue; }
            auto& row = rows[size_t(id - 1)];
            if (row.receipt_seen) { ++omitted; continue; }
            row.receipt = receipt;
            row.receipt_seen = true;
            row.receipt_ns = phase::now_ns();
        }
    }
};

void diagnostic_recurrent(CheckpointBytes& target, uint64_t salt) {
    constexpr size_t bytes = 39223296; // one third of the retained 117669888-byte live GDN total
    auto& payload = target.write();
    payload.resize(bytes, uint8_t(salt));
    for (size_t at = 0; at < payload.size(); at += size_t(1) << 20) {
        // Distinct chunks avoid the unrealistic one-object dedup of a uniform synthetic recurrent stream.
        const uint64_t marker = salt * 0x9e3779b97f4a7c15ull ^ uint64_t(at);
        std::memcpy(payload.data() + at, &marker, std::min(sizeof marker, payload.size() - at));
    }
}

SavedConversation writer_diagnostic_record(size_t index) {
    constexpr int64_t tokens[] = {374, 389, 442, 395, 457, 463, 389, 395, 457};
    constexpr size_t checkpoints[] = {0, 2, 0, 2, 2, 2, 1, 1, 1};
    SavedConversation image = all_stage_record(41 + 2 * index, tokens[index]);
    const auto fill = [&](SavedConversation& stage, size_t ordinal) {
        const uint64_t salt = (index + 1) * 100 + ordinal * 8;
        diagnostic_recurrent(stage.live.gdn, salt);
        for (size_t cut = 0; cut < checkpoints[index]; ++cut) {
            ConversationCheckpoint checkpoint;
            checkpoint.ids.assign(stage.live.ids.begin(), stage.live.ids.end() - long(7 * (cut + 1)));
            checkpoint.used = cut + 1;
            checkpoint.pinned = true;
            diagnostic_recurrent(checkpoint.gdn, salt + cut + 1);
            checkpoint.ple.assign(checkpoint.ids.size() / 2, uint8_t(salt + cut));
            checkpoint.tails.assign(2, uint8_t(salt + cut));
            stage.checkpoints.push_back(std::move(checkpoint));
        }
    };
    fill(image, 0);
    for (size_t stage = 0; stage < image.stage_images.size(); ++stage)
        fill(image.stage_images[stage], stage + 1);
    image.freeze();
    return image;
}

int writer_phase_diagnostic(const char* parent, const char* output_path) {
    // Create an exclusive raw record, and a fresh private child of the requested disposable-store parent.
    // No preserved store is opened. Preparation precedes acceptance; no publication can run before the fixed clock.
    FILE* output = std::fopen(output_path, "wx");
    if (!output) {
        std::fprintf(stderr, "writer diagnostic: cannot exclusively create the output file\n");
        return 1;
    }
    TempRoot root(parent);
    const char* batching = std::getenv("STRATA_STORE_DIR_FSYNC_BATCH");
    if (root.path.empty() || !immutable_checkpoints_enabled() || !batching || std::strcmp(batching, "1") != 0) {
        std::fputs("{\"kind\":\"not_started\",\"reason\":\"private_root_immutable_or_batching_prerequisite\"}\n", output);
        std::fclose(output);
        return 1;
    }
    // mkdtemp's final component contains only the fixed prefix and its generated safe suffix.
    const char* leaf = std::strrchr(root.path.c_str(), '/') + 1;
    std::fprintf(output, "{\"kind\":\"configuration\",\"issue\":309,\"publications\":9,\"drain_ms\":8000,"
               "\"stages\":3,\"chunk_bytes\":1048576,\"quota_bytes\":4294967296,"
               "\"ram_envelope_bytes\":12884901888,\"dir_fsync_batch\":true,\"store_leaf\":\"%s\","
               "\"original_workload_reproduction\":false}\n", leaf);
    std::fflush(output);
    const uint64_t prepare_started = phase::now_ns();
    std::array<SavedConversation, 9> inputs;
    uint64_t input_bytes = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        inputs[i] = writer_diagnostic_record(i);
        input_bytes += inputs[i].bytes();
    }
    const uint64_t prepare_ns = phase::now_ns() - prepare_started;
    check(input_bytes <= (uint64_t(3) << 30), "the prepared nine-publication inputs stay within their frozen 3 GiB cap");
    if (input_bytes > (uint64_t(3) << 30)) { std::fclose(output); return 1; }
    WriterDiagnostic diagnostic; // observer context outlives the store's real writer
    auto options = options_for(root.path);
    options.chunk_bytes = uint64_t(1) << 20;
    options.quota_bytes = uint64_t(4) << 30;
    options.min_free_bytes = uint64_t(1) << 30;
    options.metadata_bytes = uint64_t(64) << 20;
    options.head_limit = 128;
    ConversationStore store(options);
    if (!store.enabled() || !store.healthy()) {
        std::fputs("{\"kind\":\"not_started\",\"reason\":\"disposable_store_unavailable\"}\n", output);
        std::fclose(output);
        return 1;
    }
    Sink sink(store, size_t(12) << 30, 17, size_t(64) << 20);
    sink.out.phase_event = [&](const phase::PublicationEvent& event) { diagnostic.observe(event); };
    ContinuationPublisher publisher;
    publisher.configure(&store, std::move(sink.out));
    store.set_publish_trace_for_test(+[](void* context, const ConversationStorePublishTrace& trace) noexcept {
        static_cast<WriterDiagnostic*>(context)->observe(trace);
    }, &diagnostic);
    const uint64_t started_ns = phase::now_ns();
    const uint64_t deadline = started_ns / 1000000 + 8000; // fixed before the first capture/handoff
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (phase::now_ns() / 1000000 >= deadline) {
            check(false, "all nine captures must fit the original fixed admission deadline");
            break;
        }
        diagnostic.rows[i].capture_started_ns = phase::now_ns();
        const auto submission = facade_capture(publisher, sink, i % 2, i + 1, inputs[i]);
        diagnostic.rows[i].capture_finished_ns = phase::now_ns();
        check(submission.queued && submission.handle.publication_id == i + 1,
              "the diagnostic accepts each separately identified publication obligation");
        inputs[i] = {}; // drop the preparation alias, never retain an unaccounted independent payload
        if (phase::now_ns() / 1000000 < deadline) publisher.flush(0);
        diagnostic.drain(publisher);
    }
    while (publisher.holds_work()) {
        const uint64_t now = phase::now_ns() / 1000000;
        if (now >= deadline) break;
        sink.refresh_allowance();
        publisher.flush(std::min<uint64_t>(250, deadline - now));
        diagnostic.drain(publisher);
    }
    const uint64_t drain_finished_ns = phase::now_ns();
    const size_t queued_at_fence = publisher.queued().size(), pending_at_fence = publisher.pending().size();
    if (!publisher.queued().empty()) publisher.abandon("the bounded shutdown capture/drain ended");
    // No output, wait or deadline adjustment between drain and this fence; disabled remains a terminal consequence.
    store.request_shutdown(deadline);
    const auto terminal = store.finish_shutdown();
    publisher.reconcile(0);
    diagnostic.drain(publisher);
    store.set_publish_trace_for_test(nullptr, nullptr);
    check(diagnostic.omitted == 0, "the bounded phase collector loses no publication identities");
    check(!publisher.holds_work(), "quiescence settles the diagnostic's publication ownership");
    std::fprintf(output, "{\"kind\":\"shutdown\",\"started_ns\":%llu,\"finalize_before_ms\":%llu,"
        "\"drain_finished_ns\":%llu,\"finished_ns\":%llu,\"queued_at_fence\":%zu,\"pending_at_fence\":%zu,"
        "\"state\":\"%s\",\"run_intent\":%s,\"fixture_prepare_ns\":%llu,\"input_accounted_bytes\":%llu,"
        "\"omitted\":%zu,\"disabled_is_latency_cause\":false}\n",
        (unsigned long long) started_ns, (unsigned long long) deadline, (unsigned long long) drain_finished_ns,
        (unsigned long long) phase::now_ns(), queued_at_fence, pending_at_fence,
        conversation_store_shutdown_state_name(terminal), store.run_intent_active() ? "true" : "false",
        (unsigned long long) prepare_ns, (unsigned long long) input_bytes, diagnostic.omitted);
    for (size_t i = 0; i < diagnostic.rows.size(); ++i) {
        const auto& row = diagnostic.rows[i];
        check(row.receipt_seen && row.receipt.released, "each accepted diagnostic handle has one ownership-complete receipt");
        if (row.writer_seen)
            check(row.writer.request_id == i + 1 && row.writer.generation == i + 1,
                  "writer summaries bind to the exact publication id and store generation");
        std::fprintf(output, "{\"kind\":\"publication\",\"publication_id\":%zu,\"owner_generation\":%llu,"
            "\"status\":\"%s\",\"writer_seen\":%s", i + 1,
            (unsigned long long) row.receipt.handle.owner_generation,
            row.receipt_seen ? publication_status_name(row.receipt.status) : "unavailable",
            row.writer_seen ? "true" : "false");
        const auto number = [&](const char* name, uint64_t value, bool known) {
            std::fprintf(output, ",\"%s\":", name);
            if (known) std::fprintf(output, "%llu", (unsigned long long) value);
            else std::fputs("null", output);
        };
        const auto clock = [&](const char* name, uint64_t value) { number(name, value, value != 0); };
        const auto span = [&](const char* name, uint64_t begin, uint64_t end) {
            number(name, end >= begin ? end - begin : 0, begin && end && end >= begin);
        };
        clock("capture_started_ns", row.capture_started_ns);
        clock("capture_finished_ns", row.capture_finished_ns);
        clock("handoff_started_ns", row.publication.admitted_ns);
        clock("admitted_observed_ns", row.admitted_observed_ns);
        clock("store_admitted_ns", row.writer.admitted_ns);
        clock("writer_started_ns", row.writer.writer_started_ns);
        clock("payload_started_ns", row.writer.payload_started_ns);
        clock("payload_finished_ns", row.writer.payload_finished_ns);
        clock("manifest_started_ns", row.writer.manifest_started_ns);
        clock("manifest_finished_ns", row.writer.manifest_finished_ns);
        clock("catalog_started_ns", row.writer.catalog_started_ns);
        clock("catalog_finished_ns", row.writer.catalog_finished_ns);
        clock("catalog_applied_ns", row.writer.catalog_applied_ns);
        clock("body_finished_ns", row.writer.body_finished_ns);
        clock("result_ready_ns", row.writer.result_ready_ns);
        clock("terminal_observed_ns", row.terminal_ns);
        clock("receipt_consumed_ns", row.receipt_ns);
        span("capture_ns", row.capture_started_ns, row.capture_finished_ns);
        span("capture_to_admission_ns", row.capture_finished_ns, row.writer.admitted_ns);
        span("handoff_ns", row.publication.admitted_ns, row.admitted_observed_ns);
        span("admission_to_writer_ns", row.writer.admitted_ns, row.writer.writer_started_ns);
        span("payload_ns", row.writer.payload_started_ns, row.writer.payload_finished_ns);
        span("manifest_ns", row.writer.manifest_started_ns, row.writer.manifest_finished_ns);
        span("catalog_ns", row.writer.catalog_started_ns, row.writer.catalog_finished_ns);
        span("cleanup_and_settlement_ns", row.writer.body_finished_ns, row.writer.result_ready_ns);
        span("result_to_receipt_ns", row.writer.result_ready_ns, row.receipt_ns);
        number("store_generation", row.writer.generation, row.writer_seen);
        number("epoch", row.receipt.epoch, row.receipt_seen);
        number("logical_payload_bytes", row.writer.logical_payload_bytes, row.writer.payload_started_ns != 0);
        number("source_bytes", row.writer.source_bytes, row.writer_seen);
        number("source_chunks", row.writer.source_chunks, row.writer_seen);
        number("hashed_bytes", row.writer.hashed_bytes, row.writer_seen);
        std::fprintf(output, ",\"timing_calls_complete\":%s",
            row.writer_seen ? (row.writer.calls_complete ? "true" : "false") : "null");
        const bool calls_known = row.writer_seen && row.writer.calls_complete;
        number("source_read_ns", row.writer.source_read_ns, calls_known);
        number("hash_ns", row.writer.hash_ns, calls_known);
        number("file_write_ns", row.writer.file_write_ns, calls_known);
        number("completed_file_write_bytes", row.writer.file_write_bytes, row.writer_seen);
        number("file_fsync_ns", row.writer.file_fsync_ns, calls_known);
        number("file_fsync_calls", row.writer.file_fsync_calls, row.writer_seen);
        number("dir_fsync_ns", row.writer.dir_fsync_ns, calls_known);
        number("dir_fsync_calls", row.writer.dir_fsync_calls, row.writer_seen);
        number("writer_thread_cpu_ns", row.writer.thread_cpu_ns, row.writer.thread_cpu_known);
        std::fputs("}\n", output);
    }
    std::fputs("{\"kind\":\"limits\",\"timings\":\"inclusive; fsync/write counters are nested, not additive\","
        "\"missing_phases\":null,\"shape\":\"synthetic three-stage recurrent workload, not original MAN2/model/KV topology\","
        "\"claim\":\"CPU/store diagnostic only; disabled is shutdown fencing, not an attributed latency cause\"}\n", output);
    const bool output_ok = !std::ferror(output);
    const bool closed = std::fclose(output) == 0;
    check(output_ok && closed, "the exclusive diagnostic record is written completely");
    std::fprintf(stderr, "conversation_publication_test: writer diagnostic %d checks, %d failures; shutdown=%s\n",
                 checks, failures, conversation_store_shutdown_state_name(terminal));
    return failures == 0 ? 0 : 1;
}

// Diagnostic contract fixture only: a forced late writer reproduces the retained terminal shape, not its latency
// cause. The real publisher, payload writer, shutdown fence, receipt ownership and restart recovery all run.
struct PublicationPayloadGate {
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
              "the backlog fixture reaches the real post-payload, pre-manifest shutdown fence");
    }
    void release() {
        std::lock_guard<std::mutex> guard(mutex);
        open = true;
        cv.notify_all();
    }
};

void check_facade_shutdown_backlog_deadline() {
    for (const bool expired : {false, true}) {
        TempRoot root;
        PublicationPayloadGate gate; // outlives the store and its real worker frame
        auto options = options_for(root.path);
        options.head_limit = 16; // all nine promised fresh roots fit in both cases
        ConversationStore store(options);
        const uint64_t epoch = store.epoch();
        Sink sink(store);
        ContinuationPublisher publisher;
        publisher.configure(&store, std::move(sink.out));
        store.set_publish_payload_gate_for_test(+[](void* p) {
            static_cast<PublicationPayloadGate*>(p)->pause();
        }, &gate);
        std::array<PublicationHandle, 9> handles;
        const auto now_ms = [] {
            return uint64_t(std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
        };
        // Fixed once, before work; no retry, legacy close or finalization may move this deadline.
        const uint64_t deadline = now_ms() + (expired ? 20 : 10000);
        for (size_t i = 0; i < handles.size(); ++i) {
            auto record = stored_record(341 + 2 * i, 64);
            record.checkpoints.clear();
            const auto submission = facade_capture(publisher, sink, i % 2, i + 1, record);
            check(submission.queued && submission.handle, "the backlog fixture accepts every publication obligation");
            handles[i] = submission.handle;
            publisher.flush(0);
            if (i == 0) gate.await();
        }
        const auto blocked = publisher.readiness();
        check(blocked.queued == 8 && blocked.pending == 1 && blocked.outstanding == 9 && !blocked.receipts,
              "one real writer holds eight accepted FIFO captures without hiding terminal receipts");
        check(store.writer_busy() && store.ram_lease_bytes() > 0,
              "the started payload owns the real writer slot and its RAM lease");
        if (expired) {
            std::this_thread::sleep_until(std::chrono::steady_clock::time_point(
                std::chrono::milliseconds(deadline + 1)));
            publisher.abandon("the bounded shutdown capture/drain ended");
            check(publisher.queued().empty() && publisher.pending().size() == 1,
                  "deadline abandonment returns only queued ownership, never the active store lease");
            check(store.ram_lease_bytes() > 0, "queued abandonment cannot refund the active writer image");
            store.request_shutdown(deadline);
        }
        gate.release();
        if (!expired) {
            settle(publisher);
            store.request_shutdown(deadline);
        }
        // This is not a deadline relaxation: a repeated request is required to retain the original stricter value.
        store.request_shutdown(UINT64_MAX);
        const auto terminal = store.finish_shutdown();
        store.set_publish_payload_gate_for_test(nullptr, nullptr);
        publisher.reconcile(0);
        check(!gate.timed_out, "the backlog fixture releases the real writer explicitly");
        check(terminal == (expired ? ConversationStoreShutdownState::quiescent_unclean
                                  : ConversationStoreShutdownState::durable_clean),
              "only work settled before the fixed finalization decision can end durable clean");
        check(store.run_intent_active() == expired, "a late publication keeps the real run-intent armed");
        check_eq(store.ram_lease_bytes(), 0, "writer settlement, not queued abandonment, releases its final lease");
        check(!publisher.holds_work(), "the final reconcile settles every accepted publication exactly once");
        for (size_t i = 0; i < handles.size(); ++i) {
            PublicationReceipt receipt;
            const auto expected = !expired ? PublicationStatus::stored :
                (i == 0 ? PublicationStatus::disabled : PublicationStatus::abandoned);
            check(publisher.take_receipt(handles[i], receipt) && receipt.status == expected && receipt.released,
                  "each generation-bound handle has the actual stored, abandoned or disabled receipt");
            check(bool(receipt.head) == !expired, "no abandoned or disabled receipt invents durable authority");
            check(!publisher.take_receipt(handles[i], receipt), "no publication obligation is refunded twice");
        }
        check_eq(publisher.readiness().outstanding, 0, "all bounded receipt slots reopen only after consumption");
        check_eq(store.counters().published, expired ? 0 : 9, "abandonment and disabled work never count as published");
        ConversationStore reopened(options);
        check_eq(reopened.epoch(), epoch + (expired ? 1 : 0), "the unchanged finalization verdict controls restart trust");
        check_eq(reopened.counters().heads, expired ? 0 : 9,
                 "only the nine completed obligations can remain reusable after restart");
        reopened.close();
    }
}

void check_facade_interrupted_process() {
    TempRoot root;
    const auto record = stored_record(321, 64);
    uint64_t epoch = 0;
    {
        ConversationStore seed(options_for(root.path));
        epoch = seed.epoch();
        seed.close();
    }
    const pid_t child = ::fork(); // no live store threads are inherited
    check(child >= 0, "the deterministic interrupted-publication fixture creates its child");
    if (child == 0) {
        ConversationStore store(options_for(root.path));
        Sink sink(store);
        ContinuationPublisher publisher;
        publisher.configure(&store, std::move(sink.out));
        // The real writer reaches this only after payload/manifest emission. Exit while the catalog transaction
        // is staged, before it can durably authorize the new head; no destructor or finalization may run.
        store.set_catalog_gate_for_test(+[](void*, bool staged) { if (staged) ::_exit(73); }, nullptr);
        const auto submission = facade_capture(publisher, sink, 0, 1, record);
        if (!submission.queued) ::_exit(74);
        settle(publisher);
        ::_exit(75);
    }
    if (child < 0) return;
    int status = 0;
    check(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 73,
          "the process really exits inside an outstanding publication, not before capture");
    ConversationStore reopened(options_for(root.path));
    check(reopened.enabled() && reopened.healthy(), "interrupted publication recovers through the existing store");
    check_eq(reopened.epoch(), epoch + 1, "abrupt process death preserves the existing run-intent recovery rule");
    const auto query = query_of(record, 3301, 1);
    check(reopened.matches(query.selector, query.tokens, query.images).empty(),
          "a manifest abandoned before catalog publication is never promoted into a stored history");
    reopened.close();
}
#endif

} // namespace

int main(int argc, char** argv) {
    // These regressions need the real store: where it is not implemented, the test is skipped, never failed.
    {
        TempRoot root;
        ConversationStore probe(options_for(root.path));
        if (!probe.enabled()) {
            std::fprintf(stderr, "conversation_publication_test: the durable store is not implemented here; skipped\n");
            return 77;
        }
    }
    if (argc == 3 && std::string_view(argv[1]) == "--cache-mutation-trace") {
        check_cache_mutation_publication(argv[2]);
        std::fprintf(stderr, "conversation_publication_test: mutation fixture %d checks, %d failures\n", checks, failures);
        return failures == 0 ? 0 : 1;
    }
    if (argc == 2 && std::string_view(argv[1]) == "--persistence-readiness") {
        check_store_readiness_fault_reason();
        check_facade_uncertain_publication();
        std::fprintf(stderr, "conversation_publication_test: readiness %d checks, %d failures\n", checks, failures);
        return failures == 0 ? 0 : 1;
    }
#ifdef STRATA_STORE_TEST_HOOKS
    if (argc == 4 && std::string_view(argv[1]) == "--writer-phase-diagnostic")
        return writer_phase_diagnostic(argv[2], argv[3]);
    if (argc == 2 && std::string_view(argv[1]) == "--shutdown-backlog") {
        check_facade_shutdown_backlog_deadline();
        std::fprintf(stderr, "conversation_publication_test: shutdown backlog %d checks, %d failures\n", checks, failures);
        return failures == 0 ? 0 : 1;
    }
#endif
    check_optional_shared_export_predicate();
    check_optional_shared_export_parks_without_capture();
    check_store_readiness_fault_reason();
    check_facade_receipts_and_identity();
    check_facade_cancel_with_consumers();
    check_facade_optional_failure_and_peer();
    check_facade_bounded_receipts();
    check_facade_uncertain_publication();
#ifdef STRATA_STORE_TEST_HOOKS
    check_facade_shutdown_backlog_deadline();
    check_facade_interrupted_process();
#endif
    check_cache_mutation_publication();
    check_checkpoint_mount_preserves_the_longer_branch();
    check_checkpoint_mount_allows_a_whole_prefix_replacement();
    check_divergent_capture_is_published_as_a_branch();
    check_completion_landing_after_the_harvest();
    check_a_covered_root_keeps_its_record();
    check_a_moved_on_owner_never_receives_an_old_completion();
    check_deferral_does_not_spend_an_attempt();
    check_waiting_ownership_and_release();
    check_capture_refusal_is_atomic();
    check_optional_retention_failure_preserves_publication();
    check_bounded_capture_persists_the_newest_all_stage_frontier();
    check_ram_certification_is_independent_of_disk_export();
    check_shared_borrow_and_owner_neutral_publication();
    check_shared_head_is_evictable();
    check_evicted_latest_does_not_strand_publications();
    check_retained_stale_predecessor_is_not_rebranched();
    check_unreaped_results_preserve_production_pending();
    check_phase_producer_survives_owner_reuse();
    check_phase_terminals_survive_trace_saturation();
    std::fprintf(stderr, "conversation_publication_test: %d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
