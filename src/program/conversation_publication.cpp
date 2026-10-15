#include "strata/program/conversation_publication.hpp"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <utility>

namespace strata::program {

void ConversationPublicationQueue::report(const ConversationPublicationSink& sink, const char* format, ...) {
    if (!sink.report) return;
    // Bounded by construction: the only variable-length pieces are the store's own short reasons.  A truncation
    // loses the tail of a diagnostic line and never a record, and no allocation happens here.
    char line[1024];
    va_list args;
    va_start(args, format);
    const int written = std::vsnprintf(line, sizeof line, format, args);
    va_end(args);
    if (written < 0) return;
    sink.report(std::string(line));
}

ConversationPublicationOwner& ConversationPublicationQueue::owner(uint64_t id) {
    while (owners_.size() <= id) owners_.emplace_back();
    return owners_[size_t(id)];
}

bool ConversationPublicationQueue::enqueue(ConversationPublicationCapture&& entry,
                                           const ConversationPublicationSink& sink) {
    if (queued_.size() >= kMaxQueued) {
        report(sink, "strata serve: conversation store: %zu publications are already waiting; this capture is not "
                     "queued for persistence\n", queued_.size());
        return false;
    }
    queued_.push_back(std::move(entry));
    return true;
}

void ConversationPublicationQueue::drop(size_t index, const char* why, const ConversationPublicationSink& sink) {
    ConversationPublicationCapture& entry = queued_[index];
    if (sink.release_lease) sink.release_lease(entry.lease_bytes);
    ++counters_.dropped;
    report(sink, "strata serve: conversation store: queued publication of owner %llu (capture epoch %llu) dropped "
                 "(%s, %zu MiB)\n", (unsigned long long) entry.owner, (unsigned long long) entry.capture_epoch, why,
           entry.lease_bytes >> 20);
    queued_.erase(queued_.begin() + (std::ptrdiff_t) index);
}

void ConversationPublicationQueue::abandon(const char* why, const ConversationPublicationSink& sink) {
    for (ConversationPublicationCapture& entry : queued_) {
        if (sink.release_lease) sink.release_lease(entry.lease_bytes);
        ++counters_.dropped;
        report(sink, "strata serve: conversation store: queued publication of owner %llu dropped (%s)\n",
               (unsigned long long) entry.owner, why);
    }
    queued_.clear();
    forget_lineages();
}

void ConversationPublicationQueue::forget_lineages() noexcept {
    for (ConversationPublicationOwner& owner : owners_) owner.lineage = {};
}

bool ConversationPublicationQueue::created(const strata::core::ConversationStorePublishResult& result) {
    return result.status == strata::core::ConversationStorePublishStatus::stored ||
           result.status == strata::core::ConversationStorePublishStatus::replaced ||
           result.status == strata::core::ConversationStorePublishStatus::retained_branch;
}

bool ConversationPublicationQueue::covers(const ConversationPublicationCapture& entry,
                                          const ConversationPublicationOwner& owner,
                                          strata::core::ConversationStorePublish& out) {
    const ConversationPublicationLineage& lineage = owner.lineage;
    // The whole record the predecessor head holds is still this capture's prefix (equal or longer), so the store may
    // retire it; the store re-validates the predecessor, every pin it carries and its epoch itself, and keeps the
    // predecessor as a retained branch whenever that proof does not hold.  A capture whose owner lineage has moved
    // on - or whose own hint was retired with the head it named - makes no claim at all and is published as the
    // branch it is.
    if (!(entry.continues && owner.generation == entry.owner_generation && lineage.valid &&
          entry.supersedes_hint != 0 && lineage.head == entry.supersedes_hint &&
          lineage.epoch == entry.capture_epoch && lineage.scope.matches(entry.scope) && lineage.cvec == entry.cvec &&
          lineage.live.size() <= entry.live.size() &&
          std::equal(lineage.live.begin(), lineage.live.end(), entry.live.begin()))) {
        return false;
    }
    out.supersedes = lineage.head;
    out.cover = true;
    return true;
}

ConversationPublicationRetry ConversationPublicationQueue::retry(
    strata::core::ConversationStorePublishAdmission admission, uint32_t& attempts) {
    ConversationPublicationRetry out;
    if (admission == strata::core::ConversationStorePublishAdmission::accepted) {
        out.accepted = true;
        return out;
    }
    // The store is saying "not now" about a resource that is occupied at this instant - the writer's queue, or room
    // it is about to have again.  That is exactly the case the seam's own handover already retries without spending
    // an attempt, so a deferral must not consume the allowance either: a capture dropped after eight *deferrals*
    // would be a stored history the engine paid for and threw away while nothing was ever wrong with it.
    if (admission == strata::core::ConversationStorePublishAdmission::deferred) return out;
    out.exhausted = ++attempts >= kMaxAttempts;
    return out;
}

std::shared_ptr<const strata::core::SavedConversation> ConversationPublicationQueue::pinned_live_cut(
    const std::shared_ptr<const strata::core::SavedConversation>& image) {
    if (!image || image->live.pinned) return image;
    // The alias shares the frozen payload segments the publication would have offered anyway; only the one bit that
    // becomes a cut promise in the manifest differs.
    strata::core::SavedConversation published = image->frozen_alias();
    published.live.pinned = true;
    return std::make_shared<strata::core::SavedConversation>(std::move(published));
}

size_t ConversationPublicationQueue::retire(uint64_t retired_head) {
    if (retired_head == 0) return 0;
    size_t references = 0;
    for (ConversationPublicationCapture& entry : queued_) {
        if (entry.supersedes_hint != retired_head) continue;
        entry.supersedes_hint = 0;
        ++references;
    }
    for (ConversationPublicationOwner& owner : owners_) {
        if (!owner.lineage.valid || owner.lineage.head != retired_head) continue;
        owner.lineage = {};
        ++references;
    }
    return references;
}

bool ConversationPublicationQueue::adopt(const ConversationPublicationPending& pending,
                                        const strata::core::ConversationStorePublishResult& result) {
    // A one-shot borrow published as a fresh branch never becomes the owner's lineage, and a completion that arrives
    // after its owner's generation moved on describes a state the owner no longer runs.
    ConversationPublicationOwner& target = owner(pending.owner);
    if (!pending.continues || target.generation != pending.owner_generation) return false;
    target.lineage.head = result.head_id;
    target.lineage.epoch = result.epoch;
    target.lineage.live = pending.live;
    target.lineage.scope = pending.scope;
    target.lineage.cvec = pending.cvec;
    target.lineage.valid = !pending.live.empty() && pending.scope.valid();
    return true;
}

size_t ConversationPublicationQueue::reconcile(strata::core::ConversationStore& store,
                                               const ConversationPublicationSink& sink, uint64_t wait_ms) {
    size_t reaped = 0;
    for (;;) {
        strata::core::ConversationStorePublishResult result;
        const bool got = reaped == 0 && wait_ms > 0 ? store.wait_publish(result, wait_ms)
                                                   : store.take_publish(result);
        if (!got) break;
        ++reaped;
        size_t at = pending_.size();
        for (size_t i = 0; i < pending_.size(); ++i)
            if (pending_[i].request_id == result.request_id) { at = i; break; }
        if (at == pending_.size()) continue;   // already reaped (or a foreign result)
        const ConversationPublicationPending pending = pending_[at];
        pending_.erase(pending_.begin() + (std::ptrdiff_t) at);
        const bool made = created(result);
        if (made) {
            adopt(pending, result);
            // Independently of the adoption above: a replacement removed the head it superseded from the store's
            // catalog, so every reference to it is now dangling - including the references of owners whose
            // generation guard just refused the adoption.
            retire(result.superseded);
        }
        if (result.status == strata::core::ConversationStorePublishStatus::coalesced) {
            // A newer capture of the same owner and lineage replaced this one before the writer started it.  The
            // retired lease is the store's own accounting; no lineage moves here.
            ++counters_.coalesced;
            report(sink, "strata serve: conversation store: publication of owner %llu was coalesced by a newer "
                         "capture of the same lineage\n", (unsigned long long) pending.owner);
        } else if (!made) {
            ++counters_.refused;
            report(sink, "strata serve: conversation store: publication %s for owner %llu generation %llu (%s, %llu "
                         "new bytes)\n", strata::core::conversation_store_publish_status_name(result.status),
                   (unsigned long long) pending.owner, (unsigned long long) pending.generation,
                   result.detail.c_str(), (unsigned long long) result.new_bytes);
        } else {
            ++counters_.published;
            report(sink, "strata serve: conversation store: %s head=%llu owner=%llu supersedes=%llu "
                         "payload_bytes=%llu new=%llu reused=%llu released=%d\n",
                   strata::core::conversation_store_publish_status_name(result.status),
                   (unsigned long long) result.head_id, (unsigned long long) pending.owner,
                   (unsigned long long) pending.supersedes, (unsigned long long) result.payload_bytes,
                   (unsigned long long) result.new_bytes, (unsigned long long) result.reused_bytes,
                   int(result.released));
        }
        if (reaped >= pending_.size() + 4) break;
    }
    return reaped;
}

void ConversationPublicationQueue::flush(strata::core::ConversationStore& store,
                                        const ConversationPublicationSink& sink, uint64_t wait_ms) {
    reconcile(store, sink, wait_ms);
    while (!queued_.empty()) {
        const uint64_t epoch_now = store.epoch();
        if (queued_.front().capture_epoch != epoch_now) {
            // The execution domain advanced after this capture: the store would refuse it, and no head of the retired
            // epoch may be superseded.
            drop(0, "captured in a retired execution-domain epoch", sink);
            continue;
        }
        if (pending_.size() >= kMaxPending) break;   // the writer's own queue is full: retry
        // A test seam, never set by the serving engine: called between this seam's harvest above and the writer
        // observation below, so a regression can land a completion in exactly that window and prove the seam still
        // reconciles it before deriving the predecessor proof.
        if (sink.between_harvest_and_idle_for_test) sink.between_harvest_and_idle_for_test();
        if (store.writer_busy()) break;              // retry at the next safe seam
        // The writer queues its result and *then* clears its busy flag, so a completion can land between the harvest
        // above and this observation of idle.  Harvest it here, before the predecessor proof below reads a lineage
        // that completion may have just retired: a completion that arrived in that window and was not reconciled
        // would leave this capture naming a head the store has already removed, and it would be refused as stale on
        // every one of its attempts.
        if (reconcile(store, sink, 0) != 0) continue;
        uint64_t publication_id = 0;
        if (!sink.next_publication_id || !sink.next_publication_id(publication_id)) {
            drop(0, "the publication identifier namespace is exhausted", sink);
            continue;
        }
        ConversationPublicationCapture& entry = queued_.front();
        entry.publication_id = publication_id;
        strata::core::ConversationStorePublish publish;
        publish.request_id = publication_id;
        publish.generation = publication_id;
        publish.epoch = entry.capture_epoch;
        publish.external_lease_bytes = entry.lease_bytes;
        publish.reuse = entry.reuse;
        covers(entry, owner(entry.owner), publish);
        ConversationPublicationPending pending;
        pending.request_id = publish.request_id;
        pending.generation = publish.generation;
        pending.supersedes = publish.supersedes;
        pending.owner = entry.owner;
        pending.owner_generation = entry.owner_generation;
        pending.capture_epoch = entry.capture_epoch;
        pending.continues = entry.continues;
        pending.scope = entry.scope;
        pending.cvec = entry.cvec;
        pending.live = entry.live;
        pending.lease_bytes = entry.lease_bytes;
        pending_.push_back(std::move(pending));
        std::string why;
        // The entry keeps its own reference across the call: a refused or deferred publication leaves the image
        // exactly where it was, so the retry at the next seam hands over the same immutable bytes.  What the store is
        // *offered* may carry one extra metadata promise - a fresh branch's pinned root cut - and the entry's image is
        // left untouched for it.
        std::shared_ptr<const strata::core::SavedConversation> image = std::move(entry.image);
        const std::shared_ptr<const strata::core::SavedConversation> offered =
            pins_live_cut(publish) ? pinned_live_cut(image) : image;
        const strata::core::ConversationStorePublishAdmission admission = store.publish(offered, publish, why);
        if (admission == strata::core::ConversationStorePublishAdmission::accepted) {
            // The store charges the alias itself from here on; the export cache's lease on it ends.
            if (sink.release_lease) sink.release_lease(entry.lease_bytes);
            queued_.pop_front();
            if (sink.refresh_allowance) sink.refresh_allowance();
            continue;
        }
        entry.image = std::move(image);
        pending_.pop_back();
        if (admission == strata::core::ConversationStorePublishAdmission::deferred) {
            // A resource is occupied for the moment.  The entry keeps its image, its lease and its whole attempt
            // allowance; one line records that it happened, because it repeats by design at every seam until the
            // resource frees (the store counts the verdicts it returned).
            if (!entry.deferred_reported) {
                entry.deferred_reported = true;
                report(sink, "strata serve: conversation store: publication of owner %llu is deferred (%s); it stays "
                             "queued with its lease and its whole attempt allowance (id %llu)\n",
                       (unsigned long long) entry.owner, why.c_str(),
                       (unsigned long long) entry.publication_id);
            }
            break;
        }
        ++counters_.refused;
        if (retry(admission, entry.attempts).exhausted) {
            drop(0, why.empty() ? "the store refused the publication" : why.c_str(), sink);
            continue;
        }
        report(sink, "strata serve: conversation store: publication of owner %llu held back (%s); it stays queued for "
                     "the next safe seam (attempt %u/%u, id %llu)\n", (unsigned long long) entry.owner, why.c_str(),
               entry.attempts + 1, kMaxAttempts, (unsigned long long) entry.publication_id);
        break;
    }
}

} // namespace strata::program
