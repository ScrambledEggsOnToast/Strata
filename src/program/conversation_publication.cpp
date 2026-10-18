#include "strata/program/conversation_publication.hpp"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <limits>
#include <new>
#include <utility>

namespace strata::program {

namespace {
void phase_notify(const ConversationPublicationSink& sink, const phase::Publication& publication,
                  const char* event, const char* status, uint32_t attempts = 0, uint64_t head = 0) {
    if (!sink.phase_event) return;
    // Diagnostic failure cannot affect ownership, retirement or reconciliation.
    try { sink.phase_event({publication, event, status, phase::now_ns(), head, attempts}); }
    catch (...) {}
}
}

void ConversationPublicationOwner::borrow() {
    ++generation;
    lineage = {};
    force_fresh = true;
}

void ConversationPublicationOwner::mount(uint64_t head, uint64_t epoch, const std::vector<int32_t>& whole_live,
                                         int64_t resume, const strata::core::ConversationScope& scope, bool cvec) {
    ++generation;
    lineage = {};
    force_fresh = false;
    if (resume > 0 && (int64_t) whole_live.size() >= resume) {
        lineage.head = head;
        lineage.epoch = epoch;
        // The checkpoint selects execution state, not the extent of the predecessor this lineage may retire.
        lineage.live = whole_live;
        lineage.scope = scope;
        lineage.cvec = cvec;
        lineage.valid = true;
    }
}

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
    sink.report(line);
}

ConversationPublicationOwner& ConversationPublicationQueue::owner(uint64_t id) {
    while (owners_.size() <= id) owners_.emplace_back();
    return owners_[size_t(id)];
}

ConversationPublicationCaptureResult ConversationPublicationQueue::capture(
    strata::core::ConversationCache& cache, strata::core::SavedConversation&& image,
    ConversationPublicationCapture&& entry, size_t held, size_t held_entries,
    const ConversationPublicationSink& sink) {
    ConversationPublicationCaptureResult out;
    if (sink.phase_event) {
        entry.phase.capture_ticket = entry.image_ticket;
        entry.phase.owner = entry.owner;
        entry.phase.owner_generation = entry.owner_generation;
        entry.phase.epoch = entry.capture_epoch;
        entry.phase.shared_prefix_tokens = entry.shared_prefix ? uint64_t(image.live.ids.size()) : 0;
        if (!entry.phase.captured_ns) entry.phase.captured_ns = phase::now_ns();
    }
    const auto ordinary_park = [&] {
        image.publication_ticket = 0;
        try {
            out.parked = cache.put(std::move(image), held, held_entries);
        } catch (const std::bad_alloc&) {
            // put allocates before committing: the caller still owns the image.
        }
        return out;
    };
    const auto refuse = [&](const char* why) {
        ++counters_.refused;
        report(sink, "strata serve: conversation store: capture not queued for persistence (%s)\n", why);
        phase_notify(sink, entry.phase, "terminal", "capture_refused");
    };
    if (image.publication_ticket != 0) {
        refuse("the candidate already owns a publication ticket");
        return out;
    }
    if (entry.image_ticket == 0 || entry.image || entry.lease_bytes != 0 ||
        cache.has_publication(entry.image_ticket) ||
        std::any_of(queued_.begin(), queued_.end(), [&](const ConversationPublicationCapture& queued) {
            return queued.image_ticket == entry.image_ticket;
        })) {
        refuse("the capture needs an unused nonzero ticket and metadata without an image or lease");
        return ordinary_park();
    }
    if (queued_.size() >= kMaxQueued) {
        refuse("the bounded publication queue is full");
        return ordinary_park();
    }
    try {
        // Both containers may allocate. Do so before put may move the candidate or retire an old cache entry.
        if (!entry.shared_prefix) owner(entry.owner);
        if (entry.shared_prefix || owner(entry.owner).force_fresh) {
            entry.continues = false;
            entry.supersedes_hint = 0;
            entry.reuse.clear();
        }
        queued_.push_back(std::move(entry));
    } catch (const std::bad_alloc&) {
        refuse("queue metadata allocation failed; the captured image is still owned");
        return ordinary_park();
    }
    image.publication_ticket = queued_.back().image_ticket;
    try {
        if (sink.publication_headroom_bytes <= std::numeric_limits<size_t>::max() - held)
            out.parked = cache.put(std::move(image), held + sink.publication_headroom_bytes, held_entries);
    } catch (const std::bad_alloc&) {
        queued_.pop_back();
        refuse("protected image allocation failed; the captured image is still owned");
        return ordinary_park();
    }
    if (!out.parked) {
        queued_.pop_back();
        refuse("the protected image does not fit the bounded cache");
        return ordinary_park();
    }
    if (!queued_.back().shared_prefix) {
        auto& target = owners_[size_t(queued_.back().owner)];
        target.generation = queued_.back().owner_generation;
        target.force_fresh = false;
    }
    out.queued = true;
    phase_notify(sink, queued_.back().phase, "captured", "queued");
    return out;
}

void ConversationPublicationQueue::drop(size_t index, const char* why, const ConversationPublicationSink& sink) {
    ConversationPublicationCapture& entry = queued_[index];
    if (entry.image) {
        if (sink.release_lease) sink.release_lease(entry.lease_bytes);
    } else if (sink.release_publication) {
        sink.release_publication(entry.image_ticket);
    }
    ++counters_.dropped;
    report(sink, "strata serve: conversation store: queued publication of owner %llu (capture epoch %llu, ticket "
                 "%llu) dropped (%s, %zu MiB acquired)\n", (unsigned long long) entry.owner,
           (unsigned long long) entry.capture_epoch, (unsigned long long) entry.image_ticket, why,
           entry.lease_bytes >> 20);
    phase_notify(sink, entry.phase, "terminal", "dropped", entry.attempts);
    queued_.erase(queued_.begin() + (std::ptrdiff_t) index);
}

void ConversationPublicationQueue::abandon(const char* why, const ConversationPublicationSink& sink) {
    for (ConversationPublicationCapture& entry : queued_) {
        if (entry.image) {
            if (sink.release_lease) sink.release_lease(entry.lease_bytes);
        } else if (sink.release_publication) {
            sink.release_publication(entry.image_ticket);
        }
        ++counters_.dropped;
        report(sink, "strata serve: conversation store: queued publication of owner %llu dropped (%s)\n",
               (unsigned long long) entry.owner, why);
        phase_notify(sink, entry.phase, "terminal", "abandoned", entry.attempts);
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
    if (!(entry.continues && !entry.shared_prefix && !owner.force_fresh &&
          owner.generation == entry.owner_generation && lineage.valid &&
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

std::shared_ptr<const strata::core::SavedConversation> ConversationPublicationQueue::unpinned_shared_cut(
    const std::shared_ptr<const strata::core::SavedConversation>& image) {
    if (!image) return {};
    strata::core::SavedConversation published = image->frozen_alias();
    const auto clear_pins = [](auto& stage) {
        stage.live.pinned = false;
        for (auto& cut : stage.checkpoints) cut.pinned = false;
    };
    clear_pins(published);
    for (auto& stage : published.stage_images) clear_pins(stage);
    return std::make_shared<strata::core::SavedConversation>(std::move(published));
}

bool ConversationPublicationQueue::shared_prefix_exists(strata::core::ConversationStore& store,
    const strata::core::ConversationStoreSelector& selector, const std::vector<int32_t>& prompt,
    int64_t tokens, uint32_t stages) {
    if (!selector.scope.valid() || tokens <= 0 || tokens >= int64_t(prompt.size())) return false;
    bool exists = false;
    store.for_each_match(selector, prompt, {}, [&](const strata::core::ConversationStoreMatch& match) {
        if (match.stages == stages && match.tokens >= tokens) exists = true;
    });
    return exists;
}

size_t ConversationPublicationQueue::retire(uint64_t retired_head, uint64_t epoch) {
    if (retired_head == 0) return 0;
    size_t references = 0;
    for (ConversationPublicationCapture& entry : queued_) {
        if (entry.supersedes_hint != retired_head || entry.capture_epoch != epoch) continue;
        entry.supersedes_hint = 0;
        ++references;
    }
    for (ConversationPublicationOwner& owner : owners_) {
        if (!owner.lineage.valid || owner.lineage.head != retired_head || owner.lineage.epoch != epoch) continue;
        owner.lineage = {};
        ++references;
    }
    return references;
}

bool ConversationPublicationQueue::adopt(const ConversationPublicationPending& pending,
                                        const strata::core::ConversationStorePublishResult& result) {
    // A one-shot borrow published as a fresh branch never becomes the owner's lineage, and a completion that arrives
    // after its owner's generation moved on describes a state the owner no longer runs.
    if (pending.shared_prefix || !pending.continues) return false;
    ConversationPublicationOwner& target = owner(pending.owner);
    if (target.force_fresh || target.generation != pending.owner_generation) return false;
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
        ConversationPublicationPending pending = pending_[at];
        pending_.erase(pending_.begin() + (std::ptrdiff_t) at);
        pending.phase.applied_ns = result.applied_ns;
        pending.phase.catalog_version = result.catalog_version;
        const bool made = created(result);
        if (made) {
            adopt(pending, result);
            // Independently of the adoption above: a replacement removed the head it superseded from the store's
            // catalog, so every reference to it is now dangling - including the references of owners whose
            // generation guard just refused the adoption.
            retire(result.superseded, result.epoch);
        }
        if (result.status == strata::core::ConversationStorePublishStatus::stale && pending.supersedes != 0 &&
            result.generation == pending.generation && result.epoch == pending.capture_epoch &&
            store.healthy() && store.epoch() == pending.capture_epoch) {
            // Ordinary eviction has no replacement completion. Only a catalog-proven missing predecessor in this
            // capture's still-current epoch permits forgetting its references: stale generation/cancellation alone
            // must not turn a retained predecessor into a new branch. No lookup or allocation on successful results.
            if (!store.retains_head(pending.supersedes) && store.epoch() == pending.capture_epoch)
                retire(pending.supersedes, pending.capture_epoch);
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
        phase_notify(sink, pending.phase, "terminal",
                     strata::core::conversation_store_publish_status_name(result.status), 0, result.head_id);
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
        // would needlessly lose this accepted capture to a terminal stale verdict.
        if (reconcile(store, sink, 0) != 0) continue;
        uint64_t publication_id = 0;
        if (!sink.next_publication_id || !sink.next_publication_id(publication_id)) {
            drop(0, "the publication identifier namespace is exhausted", sink);
            continue;
        }
        ConversationPublicationCapture& entry = queued_.front();
        entry.publication_id = publication_id;
        if (sink.phase_event) entry.phase.publication = publication_id;
        std::string why;
        strata::core::ConversationStorePublishAdmission admission;
        const size_t pending_before = pending_.size();
        try {
            if (!entry.image) {
                // acquire_image allocates its shared root before consuming the ticket. A failed allocation leaves
                // the sole protected owner parked; only this head can acquire it at a later safe seam.
                entry.image = sink.acquire_image ? sink.acquire_image(entry.image_ticket) : nullptr;
                if (!entry.image) {
                    drop(0, "the protected publication ticket has no image", sink);
                    continue;
                }
                entry.lease_bytes = entry.image->bytes();
            }
            if (entry.shared_prefix) {
                strata::core::ConversationStoreSelector selector;
                selector.scope = entry.scope;
                selector.cvec = entry.cvec;
                selector.geometry = entry.image->geometry;
                selector.layer_lo = entry.image->layer_lo;
                selector.layer_hi = entry.image->layer_hi;
                // The selector requires a cut strictly shorter than the demand.
                // This sentinel is metadata only, never part of the captured cut.
                auto prompt = entry.live;
                prompt.push_back(0);
                if (shared_prefix_exists(store, selector, prompt, int64_t(entry.live.size()),
                                         uint32_t(entry.image->stage_images.size() + 1))) {
                    drop(0, "an equal or longer exact shared prefix is already stored", sink);
                    continue;
                }
            }
            strata::core::ConversationStorePublish publish;
            publish.request_id = publication_id;
            publish.generation = publication_id;
            publish.epoch = entry.capture_epoch;
            publish.external_lease_bytes = entry.lease_bytes;
            if (!entry.shared_prefix) {
                publish.reuse = entry.reuse;
                covers(entry, owner(entry.owner), publish);
            }
            ConversationPublicationPending pending;
            pending.request_id = publish.request_id;
            pending.generation = publish.generation;
            pending.supersedes = publish.supersedes;
            pending.owner = entry.owner;
            pending.owner_generation = entry.owner_generation;
            pending.capture_epoch = entry.capture_epoch;
            pending.continues = entry.continues;
            pending.shared_prefix = entry.shared_prefix;
            pending.scope = entry.scope;
            pending.cvec = entry.cvec;
            pending.live = entry.live;
            pending.lease_bytes = entry.lease_bytes;
            if (sink.phase_event) {
                pending.phase = entry.phase;
                pending.phase.admitted_ns = phase::now_ns();
            }
            // Every allocation the queue needs to remember the result precedes the store's destructive handoff.
            pending_.push_back(std::move(pending));
            // Keep the original head image throughout refusal, deferral and allocation failure. A fresh branch's
            // extra root-cut promise is offered through a metadata alias, never by mutating the retry image.
            std::shared_ptr<const strata::core::SavedConversation> offered;
            const auto has_draft_certification = [](const auto& stage) {
                return stage.draft_begin != -1 || stage.draft_cells != -1 || stage.draft_next_token != -1 ||
                    !stage.draft_boundary_residual.empty();
            };
            const bool legacy_view_needed = !sink.export_draft_certification &&
                (has_draft_certification(*entry.image) ||
                 std::any_of(entry.image->stage_images.begin(), entry.image->stage_images.end(),
                             has_draft_certification));
            if (legacy_view_needed) {
                // Keep the certified retry/read-through image intact. This is
                // one metadata alias, like the pin view below, not a payload copy.
                auto published = entry.image->frozen_alias();
                const auto legacy_view = [&](auto& stage) {
                    stage.draft_begin = stage.draft_cells = -1;
                    stage.draft_next_token = -1;
                    stage.draft_boundary_residual = {};
                    if (entry.shared_prefix) {
                        stage.live.pinned = false;
                        for (auto& cut : stage.checkpoints) cut.pinned = false;
                    }
                };
                legacy_view(published);
                for (auto& stage : published.stage_images) legacy_view(stage);
                if (!entry.shared_prefix && pins_live_cut(publish)) published.live.pinned = true;
                offered = std::make_shared<strata::core::SavedConversation>(std::move(published));
            } else {
                offered = entry.shared_prefix ? unpinned_shared_cut(entry.image)
                    : pins_live_cut(publish) ? pinned_live_cut(entry.image) : entry.image;
            }
            admission = store.publish(offered, publish, why);
        } catch (const std::bad_alloc&) {
            if (pending_.size() != pending_before) pending_.pop_back();
            if (!entry.deferred_reported) {
                entry.deferred_reported = true;
                report(sink, "strata serve: conversation store: publication of owner %llu waits for allocation; "
                             "its protected image or acquired lease is retained (ticket %llu)\n",
                       (unsigned long long) entry.owner, (unsigned long long) entry.image_ticket);
            }
            break;
        }
        if (admission == strata::core::ConversationStorePublishAdmission::accepted) {
            if (sink.phase_event) entry.phase = pending_.back().phase;
            phase_notify(sink, entry.phase, "admitted", "accepted", entry.attempts);
            // The store owns its full charge now. Optional RAM retention gets only spare capacity after that exact
            // transfer; failing to retain is not a failed publication and must never undo the admitted result.
            if (sink.release_lease) sink.release_lease(entry.lease_bytes);
            std::shared_ptr<const strata::core::SavedConversation> image = std::move(entry.image);
            queued_.pop_front();
            if (sink.retain_image) {
                try {
                    sink.retain_image(std::move(image));
                } catch (const std::bad_alloc&) {
                    report(sink, "strata serve: conversation store: publication accepted; optional RAM retention "
                                 "was skipped after allocation failure\n");
                }
            }
            if (sink.refresh_allowance) sink.refresh_allowance();
            continue;
        }
        pending_.pop_back();
        if (admission == strata::core::ConversationStorePublishAdmission::deferred) {
            // A resource is occupied for the moment.  The entry keeps its image, its lease and its whole attempt
            // allowance; one line records that it happened, because it repeats by design at every seam until the
            // resource frees (the store counts the verdicts it returned).
            if (!entry.deferred_reported) {
                entry.deferred_reported = true;
                phase_notify(sink, entry.phase, "handoff", "deferred", entry.attempts);
                report(sink, "strata serve: conversation store: publication of owner %llu is deferred (%s); it stays "
                             "queued with its lease and its whole attempt allowance (id %llu)\n",
                       (unsigned long long) entry.owner, why.c_str(),
                       (unsigned long long) entry.publication_id);
            }
            break;
        }
        ++counters_.refused;
        phase_notify(sink, entry.phase, "handoff", "refused", entry.attempts);
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
