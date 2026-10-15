// Deterministic CPU regressions for the native publication handover (include/strata/program/conversation_publication.hpp).
// The store is the real one, on a temporary root: the handover's verdicts, the retirement a completed replacement
// performs on the references to the head it removed, and the branch a divergent capture is published as are all
// checked against actual admissions, actual catalog commits and a fresh open of the same root - never against a
// stand-in that would agree with whatever the handover happened to decide.
#include "strata/program/conversation_publication.hpp"

#include "conversation_store_format.hpp"
#include "conversation_store_sha256.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <memory>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

using namespace strata::core;
using strata::program::ConversationPublicationCapture;
using strata::program::ConversationPublicationOwner;
using strata::program::ConversationPublicationPending;
using strata::program::ConversationPublicationQueue;
using strata::program::ConversationPublicationSink;
using conversation_store_detail::images_equal;

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
        std::fprintf(stderr, "FAIL: %s (got %llu, want %llu)\n", what, (unsigned long long) got,
                     (unsigned long long) want);
    }
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
    TempRoot() {
        char pattern[] = "/tmp/strata-publication-test-XXXXXX";
        const char* made = ::mkdtemp(pattern);
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

// One turn later on the same branch: the live record and the K/V extents grow, so every unchanged byte of the prefix
// keeps the physical identity the store deduplicates on.  `seed` chooses the appended bytes, so two captures taken
// from the same head with different seeds are two different states that share only that head's prefix.
SavedConversation turn_on(const SavedConversation& base, int64_t tokens, int seed) {
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
    check(read.image && images_equal(*read.image, expected, why), "the restored record is byte-identical");
    store.release_read(request_id, 1);
}

// ---------------------------------------------------------------------------------------------------------------
// The serving engine as this module sees it
// ---------------------------------------------------------------------------------------------------------------

struct Sink {
    ConversationPublicationSink out;
    uint64_t next_id = 0;

    Sink() {
        out.next_publication_id = [this](uint64_t& id) {
            id = ++next_id;
            return true;
        };
    }

};

// One capture, exactly as the serving engine builds it: the record it froze, the head its owner's lineage named when
// it was taken, and the export cache's lease on the parked image.
ConversationPublicationCapture capture_of(uint64_t owner, uint64_t generation, uint64_t supersedes_hint,
                                          bool continues, const SavedConversation& image, uint64_t epoch) {
    ConversationPublicationCapture entry;
    entry.owner = owner;
    entry.owner_generation = generation;
    entry.capture_epoch = epoch;
    entry.supersedes_hint = supersedes_hint;
    entry.continues = continues;
    entry.scope = image.scope;
    entry.cvec = image.cvec;
    entry.live = image.live.ids;
    entry.image = std::make_shared<SavedConversation>(image);
    entry.lease_bytes = image.bytes();
    return entry;
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

// A retained, unpinned latest head, as mounted by the native engine. Fresh branch roots are tested separately.
uint64_t mount_latest(ConversationStore& store, ConversationPublicationQueue& publications, Sink& sink,
                      const SavedConversation& record) {
    ConversationStorePublish request;
    request.request_id = request.generation = ++sink.next_id;
    std::string why;
    check(store.publish(std::make_shared<SavedConversation>(record), request, why) ==
              ConversationStorePublishAdmission::accepted, "the prior latest head is admitted");
    ConversationStorePublishResult result;
    check(store.wait_publish(result, 5000) && result.status == ConversationStorePublishStatus::stored,
          "the prior latest head is persisted");
    auto& owner = publications.owner(0);
    owner.generation = 1;
    owner.lineage = {result.head_id, store.epoch(), record.live.ids, record.scope, record.cvec, true};
    return result.head_id;
}

// ---------------------------------------------------------------------------------------------------------------
// The regressions
// ---------------------------------------------------------------------------------------------------------------

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
    Sink sink;
    ConversationPublicationQueue publications;

    const SavedConversation record = stored_record(77, 64);
    const SavedConversation covered = turn_on(record, 96, 77);      // the same branch, one turn later
    const SavedConversation divergent = turn_on(record, 96, 900);   // the same prefix, another answer
    const SavedConversation mounted = turn_on(record, 96, 901);     // a second owner's capture of the same prefix

    const uint64_t root_head = mount_latest(store, publications, sink, record);

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
    publications.owner(0).generation = 2;
    check(publications.enqueue(capture_of(0, 2, root_head, true, covered, epoch), sink.out),
          "the covering capture is queued");
    publications.flush(store, sink.out, 0);
    check_eq(publications.pending().size(), 1, "the covering capture is in flight");
    check_eq(publications.pending().front().supersedes, root_head,
             "the cover names the head whose whole record it still contains");

    publications.owner(0).generation = 3;
    check(publications.enqueue(capture_of(0, 3, root_head, true, divergent, epoch), sink.out),
          "the divergent capture is queued while the cover is in flight");
    check(publications.enqueue(capture_of(1, 5, root_head, true, mounted, epoch), sink.out),
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
    check_eq(store.counters().replaced, 1, "exactly one publication replaced a head");
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
    Sink sink;
    ConversationPublicationQueue publications;

    const SavedConversation record = stored_record(88, 64);
    const SavedConversation covered = turn_on(record, 96, 88);
    const SavedConversation divergent = turn_on(record, 96, 902);

    const uint64_t root_head = mount_latest(store, publications, sink, record);

    publications.owner(0).generation = 2;
    check(publications.enqueue(capture_of(0, 2, root_head, true, covered, epoch), sink.out),
          "the covering capture is queued");
    publications.flush(store, sink.out, 0);   // handed over; its completion is deliberately not harvested yet
    check_eq(publications.pending().size(), 1, "the covering capture is in flight and unsettled");

    publications.owner(0).generation = 3;
    check(publications.enqueue(capture_of(0, 3, root_head, true, divergent, epoch), sink.out),
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
    Sink sink;
    ConversationPublicationQueue publications;

    const SavedConversation first_conversation = stored_record(99, 64);
    const SavedConversation second_conversation = stored_record(101, 64);

    check(publications.enqueue(capture_of(0, 1, 0, true, first_conversation, epoch), sink.out),
          "the first conversation's capture is queued");
    publications.owner(0).generation = 1;
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
    check(publications.enqueue(capture_of(0, generation, hint, true, second_conversation, epoch), sink.out),
          "the second conversation's capture is queued");
    publications.flush(store, sink.out, 0);
    const ConversationPublicationPending* second = pending_of(publications, 0);
    check(second != nullptr && second->supersedes == 0, "it supersedes nothing");
    settle(publications, store, sink);
    check_eq(store.counters().heads, 2, "each conversation keeps its own head");

    // A one-shot borrow is published as a branch of its own and never becomes the borrowing owner's lineage.
    check(publications.enqueue(capture_of(1, 1, 0, false, second_conversation, epoch), sink.out),
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
    Sink sink;
    ConversationPublicationQueue publications;

    const SavedConversation record = stored_record(121, 64);
    const SavedConversation later = turn_on(record, 96, 121);

    check(publications.enqueue(capture_of(0, 1, 0, true, record, epoch), sink.out), "the root capture is queued");
    publications.owner(0).generation = 1;
    publications.flush(store, sink.out, 0);
    settle(publications, store, sink);
    const uint64_t root_head = publications.owner(0).lineage.head;
    check(publications.owner(0).lineage.valid && root_head != 0, "the root becomes the owner's head");
    check_eq(store.counters().heads, 1, "the root is the store's only head");

    // The branch moves on: the next turn's capture still contains the root's whole record.
    publications.owner(0).generation = 2;
    check(publications.enqueue(capture_of(0, 2, root_head, true, later, epoch), sink.out),
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
    Sink sink;
    ConversationPublicationQueue publications;
    const SavedConversation large = stored_record(131, 131072);
    const SavedConversation small = stored_record(132, 64);
    mount_latest(store, publications, sink, large);
    std::string why;
    check(store.demand(query_of(large, 100, 1), why), "the blocking image is demanded");
    ConversationStoreRead read;
    check(store.wait_ready(100, 1, read, 5000) && read.image, "the blocking image is handed out");
    store.publish_read_allowance(uint64_t(8) << 20);
    publications.owner(1).generation = 1;
    check(publications.enqueue(capture_of(1, 1, 0, true, small, store.epoch()), sink.out),
          "the blocked capture is queued");
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

} // namespace

int main() {
    // These regressions need the real store: where it is not implemented, the test is skipped, never failed.
    {
        TempRoot root;
        ConversationStore probe(options_for(root.path));
        if (!probe.enabled()) {
            std::fprintf(stderr, "conversation_publication_test: the durable store is not implemented here; skipped\n");
            return 77;
        }
    }
    check_divergent_capture_is_published_as_a_branch();
    check_completion_landing_after_the_harvest();
    check_a_covered_root_keeps_its_record();
    check_a_moved_on_owner_never_receives_an_old_completion();
    check_deferral_does_not_spend_an_attempt();
    std::fprintf(stderr, "conversation_publication_test: %d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
