#pragma once

// The native half of the durable store's publication handover: the bounded queue between the engine's capture seam
// and the store's single writer, the owner-lineage bookkeeping that turns a settled completion into the next
// capture's predecessor, and the retry policy for a capture the store will not take yet.
//
// It is a module of its own because its *decisions* are the ones a wrong answer costs a stored history: which head a
// capture may supersede, what a completed replacement retires, whether a completion may attach to an owner, whether a
// handover the store turns back spends one of the bounded attempts, and which cut a fresh branch promises so that a
// branch keeps its reusable root as well as its latest.  Keeping them here lets a test drive them against the real
// store instead of re-deriving them beside it.
//
// What deliberately stays in the serving engine: the images themselves, the export cache's lease on them, the
// published read allowance and every diagnostic destination.  Nothing here allocates beyond the queue it owns, and
// every collection it owns is bounded by a compile-time ceiling.

#include "strata/core/conversation_store.hpp"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace strata::program {

// The stored branch one owner's state currently continues from: the head the store created for it and the *whole
// record that head holds*, as this engine captured it.  It is only ever the reference a later capture's coverage
// proof is built from - what the engine holds in host memory is the state, not this.
struct ConversationPublicationLineage {
    uint64_t head = 0;
    uint64_t epoch = 0;
    std::vector<int32_t> live;   // the record this head holds, as this engine continues it
    strata::core::ConversationScope scope;
    bool cvec = true;
    bool valid = false;
};

// `generation` is the owner's current capture generation: a completion moves the lineage only when it belongs to
// exactly that generation, so a completion that arrives after its owner was retargeted, remounted or recaptured is
// dropped instead of attaching itself to a state it does not describe.
struct ConversationPublicationOwner {
    uint64_t generation = 0;
    ConversationPublicationLineage lineage;
};

// A capture the engine already paid for (GPU copies included), waiting for the store's single bounded writer.  The
// image is frozen in place and its RAM stays charged through the export cache while it waits, so losing the parked
// copy can never silently drop the only snapshot that is still to be written.  Owner slots can be reused by unrelated
// conversations: the queue keeps their publications distinct, and the image leases and its fixed entry ceiling bound
// it independently of slot identity.
struct ConversationPublicationCapture {
    uint64_t owner = 0;
    uint64_t owner_generation = 0;
    uint64_t capture_epoch = 0;
    uint64_t publication_id = 0;       // assigned when it is handed to the store, so results can be matched
    uint64_t supersedes_hint = 0;      // the owner's lineage head at capture (0: this capture starts a branch)
    bool continues = true;             // false: a one-shot borrow; never supersedes and never sets a lineage
    strata::core::ConversationScope scope;
    bool cvec = true;
    std::vector<int32_t> live;         // the captured record, for the coverage proof and the lineage
    std::vector<strata::core::ConversationStoreReuse> reuse;   // the proven intervals, donor already resolved
    std::shared_ptr<const strata::core::SavedConversation> image;   // frozen alias: payloads are shared
    size_t lease_bytes = 0;            // its charge in the export cache's envelope while it waits
    uint32_t attempts = 0;             // bounded retries; a permanently refused entry is dropped and counted
    bool deferred_reported = false;    // one diagnostic per entry: a deferral repeats by design
};

// One record per publication the store has admitted and not yet reported, so its result can move the exact owner
// generation it belongs to and its lease can be reported when the store releases it.
struct ConversationPublicationPending {
    uint64_t request_id = 0;
    uint64_t generation = 0;
    uint64_t supersedes = 0;
    uint64_t owner = 0;
    uint64_t owner_generation = 0;
    uint64_t capture_epoch = 0;
    bool continues = true;
    strata::core::ConversationScope scope;
    bool cvec = true;
    std::vector<int32_t> live;
    size_t lease_bytes = 0;
};

// The figures `strata serve` reports at shutdown for this module.  A deferral is deliberately not one of them: the
// store counts the verdicts it returned (`ConversationStoreCounters::deferred_publishes`), and a second copy of that
// count here could only drift from the answer it is supposed to describe.
struct ConversationPublicationCounters {
    uint64_t published = 0;   // completions that created a head
    uint64_t refused = 0;     // synchronous refusals, plus settled results that created no head
    uint64_t coalesced = 0;   // results a newer capture of the same lineage displaced while it was in flight
    uint64_t dropped = 0;     // queued captures abandoned before the store ever took them
};

// The one-shot verdict for a capture the store has just handed back.  `accepted` means the store owns the bytes from
// here on.  `exhausted` means this refusal was the last of the bounded attempts.  A deferral is neither: the capture
// keeps its place, its image and its whole attempt allowance, because the store is saying "not now", not "not ever"
// - and a capture that is dropped for a resource that was busy at the wrong moment is a stored history the engine
// paid for and then threw away.
struct ConversationPublicationRetry {
    bool accepted = false;
    bool exhausted = false;
};

// What a handover needs from the serving engine.
struct ConversationPublicationSink {
    // One bounded line per event, already formatted.  The engine owns the destination and the file descriptor.
    std::function<void(const char* line)> report;
    // Give up the export cache's lease on exactly these bytes: the store charges them from here on.
    std::function<void(size_t lease_bytes)> release_lease;
    // Republish the read allowance from live accounting: an accepted handover changes what the store itself holds.
    std::function<void()> refresh_allowance;
    // The store chronology the engine's own capture counter mints publication identifiers from, so a completion can
    // still be matched after a clean reopen.  False means the identifier namespace is exhausted.
    std::function<bool(uint64_t& out)> next_publication_id;
    // A test seam, never set by the serving engine: called between the harvest that opens a seam and the observation
    // of the writer that follows it, so a regression can land a completion exactly in the window that is the reason
    // the flush reconciles again once it has seen the writer idle.
    std::function<void()> between_harvest_and_idle_for_test;
};

class ConversationPublicationQueue {
public:
    // Bounds, all compile-time: the queue is a handover, never an accumulation.
    static constexpr size_t kMaxQueued = 16;
    static constexpr size_t kMaxPending = 8;   // the store's own writer plus its pending slots, plus slack
    static constexpr uint32_t kMaxAttempts = 8;   // bounded retries of an entry the store refuses to admit yet

    // Owner slots can be reused by unrelated conversations, so an owner is a *slot* identity here and its lineage is
    // the only thing that establishes ancestry.  Entries are only ever appended, so a reference stays valid.
    ConversationPublicationOwner& owner(uint64_t owner);
    std::deque<ConversationPublicationOwner>& owners() noexcept { return owners_; }
    std::deque<ConversationPublicationCapture>& queued() noexcept { return queued_; }
    std::deque<ConversationPublicationPending>& pending() noexcept { return pending_; }
    const std::deque<ConversationPublicationCapture>& queued() const noexcept { return queued_; }
    const std::deque<ConversationPublicationPending>& pending() const noexcept { return pending_; }
    ConversationPublicationCounters& counters() noexcept { return counters_; }
    const ConversationPublicationCounters& counters() const noexcept { return counters_; }
    bool holds_work() const noexcept { return !queued_.empty() || !pending_.empty(); }
    size_t size() const noexcept { return queued_.size() + pending_.size(); }

    // Take a capture into the queue.  False means the queue is full (bounded by kMaxQueued): the caller keeps
    // ownership of everything it passed and the capture is not persisted.
    bool enqueue(ConversationPublicationCapture&& entry, const ConversationPublicationSink& sink);
    // Drop one queued capture, returning its lease to the caller.
    void drop(size_t index, const char* why, const ConversationPublicationSink& sink);
    // The execution domain advanced: no head of the retired epoch may ever be superseded, so every queued capture is
    // abandoned and every owner lineage is forgotten.  Storage work the store already admitted is the store's own.
    void abandon(const char* why, const ConversationPublicationSink& sink);
    void forget_lineages() noexcept;

    // A result that created a head, as opposed to one that established nothing.
    static bool created(const strata::core::ConversationStorePublishResult& result);
    // The coverage proof: fill `out.supersedes`/`out.cover` only when the entry's own captured record still contains
    // the whole record the owner's current head holds.  Returns whether it did.
    static bool covers(const ConversationPublicationCapture& entry, const ConversationPublicationOwner& owner,
                       strata::core::ConversationStorePublish& out);
    // The retry policy: `refused` spends one of the bounded attempts, `deferred` spends none, `accepted` transfers
    // ownership.  `attempts` is the entry's own counter.
    static ConversationPublicationRetry retry(strata::core::ConversationStorePublishAdmission admission,
                                             uint32_t& attempts);

    // The cut promise a publication carries.  The store retires a covered head only when the descendant re-carries
    // every *pinned* cut the predecessor promised; a predecessor whose pinned cut the descendant drops is kept as a
    // retained branch instead (its own cover rule).  A fresh branch is the root of a history - the record a later
    // request comes back for, by a returning turn, a divergent sibling or a pinned prefix - so its live cut is
    // published pinned: the cover then retains the root and its record, with every cut under it, keeps being served
    // after the branch has moved on.  A capture that supersedes a head is a later turn of an existing branch and is
    // deliberately left unpinned: its own descendant replaces it, so a branch keeps its root and its latest, never
    // every turn, and a root's promise never grows with the number of turns.
    static bool pins_live_cut(const strata::core::ConversationStorePublish& publish) {
        return publish.supersedes == 0;
    }
    // The publication alias of `image` with that promise set.  Metadata only: every payload stays shared, and the
    // caller's image - the entry's own retry image and the export cache's copy - is never mutated for it.
    static std::shared_ptr<const strata::core::SavedConversation> pinned_live_cut(
        const std::shared_ptr<const strata::core::SavedConversation>& image);

    // Retirement bookkeeping, deliberately independent of the generation-guarded adoption below: a completed
    // *replacement* removed its predecessor from the store's catalog, so every reference to that head - the lineage
    // of each owner that mounted or continued it, and every queued capture that was captured from it, whichever
    // owner it belongs to - now names a head the store no longer has.  A capture recorded after the completion
    // cannot adopt the new head (its generation is newer), and without this its own predecessor hint would stay
    // pointed at the removed head, which the store refuses as stale, for every one of its bounded attempts: the
    // history the capture holds would be dropped with an "it was stale" line even though its bytes are perfectly
    // representable as a branch of its own.  Returns how many references were invalidated.
    //
    // Rebase is deliberately not an option here: turning the capture into a cover of the *new* head requires the
    // new head's whole record to be an exact prefix of this capture, which `covers()` proves and this bookkeeping
    // never assumes.  Forgetting the hint makes the capture a fresh branch, which is always representable.
    size_t retire(uint64_t retired_head);
    // Adopt a completion into the owner lineage it belongs to.  Precondition: `created(result)`.  Returns whether
    // the lineage moved - it does not when the capture does not continue an owner branch or when the owner's
    // generation has already moved past the one that captured it.
    bool adopt(const ConversationPublicationPending& pending,
               const strata::core::ConversationStorePublishResult& result);

    // Harvest settled publications: their lease leaves the RAM envelope, and the lineage of the *exact owner
    // generation* that produced the image moves to the head the store created.  A result whose owner was captured
    // again (or remounted, or taken over by another conversation) since it was queued is dropped instead of
    // attaching itself to a state it does not describe, and a cancelled, stale or refused publication never
    // establishes a lineage at all.  A replacement always retires the references to the head it superseded, adopted
    // or not.  Returns how many results were harvested.
    size_t reconcile(strata::core::ConversationStore& store, const ConversationPublicationSink& sink,
                     uint64_t wait_ms);

    // Hand the queued captures to the store's writer, oldest first.  A busy or momentarily-full writer leaves an
    // entry queued and the next safe seam retries it - never a misleading "deferred" line over work the engine has
    // already paid for.  The coverage proof is re-derived here from the *entry's own* captured record and its
    // owner's current lineage, so a capture queued before an earlier publication reported its head is still able to
    // supersede the head that publication created.
    //
    // The store admits the image against the read allowance in force, which every flush site publishes from live
    // accounting right before calling this (and which a suspended capture keeps from racing an allocation): this
    // function deliberately never publishes it, and it only asks the sink to refresh it after an accepted handover.
    void flush(strata::core::ConversationStore& store, const ConversationPublicationSink& sink, uint64_t wait_ms);

private:
    static void report(const ConversationPublicationSink& sink, const char* format, ...);

    std::deque<ConversationPublicationOwner> owners_;
    std::deque<ConversationPublicationCapture> queued_;
    std::deque<ConversationPublicationPending> pending_;
    ConversationPublicationCounters counters_;
};

} // namespace strata::program
