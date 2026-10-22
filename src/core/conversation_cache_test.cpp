#include "strata/core/conversation_cache.hpp"
#include "strata/core/conversation_reuse.hpp"
#include "strata/core/prefix_extent_probe.hpp"
#include "strata/program/prefix_selection.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <limits>
#include <string>
#include <cstring>
#include <new>
#include <atomic>
#if defined(__linux__)
#include <unistd.h>
#endif

namespace test_allocation_failure {
thread_local long after = -1;
std::atomic<long> live{0};
}
void* operator new(std::size_t bytes) {
    if (test_allocation_failure::after == 0) {
        test_allocation_failure::after = -1;
        throw std::bad_alloc();
    }
    if (test_allocation_failure::after > 0) --test_allocation_failure::after;
    if (void* memory = std::malloc(bytes ? bytes : 1)) {
        test_allocation_failure::live.fetch_add(1, std::memory_order_relaxed);
        return memory;
    }
    throw std::bad_alloc();
}
void operator delete(void* memory) noexcept {
    if (memory) test_allocation_failure::live.fetch_sub(1, std::memory_order_relaxed);
    std::free(memory);
}
void operator delete(void* memory, std::size_t) noexcept { ::operator delete(memory); }

using namespace strata::core;

namespace {
const ConversationScope test_scope{{1}};
int checks = 0;
void check(bool value, const char* description) {
    ++checks;
    if (!value) { std::fprintf(stderr, "FAIL: %s\n", description); std::exit(1); }
}
SavedConversation image(std::initializer_list<int32_t> ids, bool cvec = true) {
    SavedConversation s;
    s.scope = test_scope;
    s.live.ids = ids;
    s.live.gdn.resize(64, 7);
    s.cvec = cvec;
    return s;
}
bool same_checkpoint(const ConversationCheckpoint& a, const ConversationCheckpoint& b) {
    if (a.ids != b.ids || a.imgs != b.imgs || a.used != b.used || a.pinned != b.pinned || a.plan != b.plan ||
        a.gdn != b.gdn || a.ple != b.ple || a.tails != b.tails || a.dead != b.dead ||
        a.block_pos != b.block_pos || a.stage_parts.size() != b.stage_parts.size()) return false;
    for (size_t i = 0; i < a.stage_parts.size(); ++i)
        if (!same_checkpoint(a.stage_parts[i], b.stage_parts[i])) return false;
    return true;
}
// Publication tickets are ownership metadata, not part of the saved image.
bool same_image(const SavedConversation& a, const SavedConversation& b) {
    if (a.scope != b.scope || a.geometry != b.geometry || a.layer_lo != b.layer_lo ||
        a.layer_hi != b.layer_hi || a.cvec != b.cvec || !same_checkpoint(a.live, b.live) ||
        a.checkpoints.size() != b.checkpoints.size() || a.kv.size() != b.kv.size() ||
        a.stage_images.size() != b.stage_images.size()) return false;
    for (size_t i = 0; i < a.checkpoints.size(); ++i)
        if (!same_checkpoint(a.checkpoints[i], b.checkpoints[i])) return false;
    for (size_t i = 0; i < a.kv.size(); ++i) {
        const auto& x = a.kv[i];
        const auto& y = b.kv[i];
        if (x.format != y.format || x.cells != y.cells || x.heads != y.heads || x.head_dim != y.head_dim ||
            x.page_size != y.page_size || x.pooled_rows != y.pooled_rows || x.idx_dim != y.idx_dim ||
            !(x.k == y.k) || !(x.v == y.v) || !(x.k_scale == y.k_scale) ||
            !(x.v_scale == y.v_scale) || !(x.pooled == y.pooled)) return false;
    }
    for (size_t i = 0; i < a.stage_images.size(); ++i)
        if (!same_image(a.stage_images[i], b.stage_images[i])) return false;
    return true;
}

std::string trace_text(FILE* file) {
    std::fflush(file);
    const long n = std::ftell(file);
    std::rewind(file);
    std::string out(size_t(std::max(n, 0l)), '\0');
    check(std::fread(out.data(), 1, out.size(), file) == out.size(), "read the complete bounded trace");
    std::fseek(file, 0, SEEK_END);
    return out;
}
std::string trace_event(const std::string& text, const char* site, const char* action) {
    const std::string key = "\"site\":\"" + std::string(site) + "\",\"action\":\"" + action + "\"";
    auto at = text.find(key);
    while (at != std::string::npos && std::string_view(action) == "plan") {
        const auto end = text.find('\n', at);
        if (text.substr(at, end - at).find("\"victims\":0,") == std::string::npos) break;
        at = text.find(key, end);
    }
    check(at != std::string::npos, "the real mutation site emitted its named event");
    const auto begin = text.rfind('{', at);
    const auto end = text.find("{\"kind\":\"cache_event_end\"", at);
    check(end != std::string::npos, "every mutation event closes");
    return text.substr(begin, end - begin);
}
std::string victim_identity_text(const ConversationVictimIdentity& value) {
    char row[192];
    std::snprintf(row, sizeof(row), "%llu %016llx %016llx %llu %llu",
        (unsigned long long)value.serial, (unsigned long long)value.scope,
        (unsigned long long)value.history, (unsigned long long)value.tokens,
        (unsigned long long)value.bytes);
    return row;
}
std::string victim_contract(ConversationCache& cache, ConversationCacheTrace& trace,
                           const char* mode, const char* site, std::initializer_list<uint64_t> victims,
                           const SavedConversation* subject = nullptr, uint64_t subject_serial = 0,
                           uint64_t preceding_events = 0) {
    ConversationVictimIdentity identity;
    if (subject) check(ConversationVictimReplay::identify(subject_serial, *subject, identity),
                       "the nominated incoming identity is bounded");
    std::string text = "strata-cache-victim-replay-v1 mode " + std::string(mode) +
        " event " + std::to_string(trace.next_event() + preceding_events) +
        " request_sequence " + std::to_string(trace.request_sequence) +
        " leg " + std::to_string(trace.leg) + " site " + site +
        " subject " + victim_identity_text(identity) + " victims " + std::to_string(victims.size());
    for (uint64_t serial : victims) {
        const auto* image = cache.find(serial);
        check(image && ConversationVictimReplay::identify(serial, *image, identity),
              "the test nominates an exact retained allocation generation");
        text += " victim " + victim_identity_text(identity);
    }
    return text + "\n";
}
void replay_phase(ConversationCacheTrace& trace) {
    trace.request_sequence = 9; trace.leg = 2;
    std::strcpy(trace.request, "victim-fixture");
    std::strcpy(trace.incarnation, "0123456789abcdef0123456789abcdef");
    trace.set_sample(nullptr, +[](void*, ConversationCacheTrace::External&) {});
}
void check_victim_process_contract() {
    check(std::string_view(ConversationVictimReplay::from_path(nullptr).status) == "off" &&
          std::string_view(ConversationVictimReplay::from_path("/nonexistent/strata-victim-contract").status) == "missing",
          "absent/missing diagnostic contracts retain ordinary LRU");
    check(std::string_view(ConversationVictimReplay::parse(
              "strata-cache-victim-replay-v1 mode keep\n").status) == "keep",
          "an explicit un-nominated keep arm is default-preserving");
    // Descriptor pathnames under procfs are a Linux-only fixture mechanism.
#if defined(__linux__)
    FILE* contract = std::tmpfile();
    FILE* output = std::tmpfile();
    check(contract && output, "open parse-once contract and trace fixtures");
    {
        ConversationCacheTrace trace(output);
        replay_phase(trace);
        ConversationCache seed(1u << 20, 2, 0, nullptr);
        check(seed.put(image({1,2,3})) && seed.put(image({4,5,6})), "seed the process contract identities");
        const auto text = victim_contract(seed, trace, "alternate", "evict_oldest", {2}, nullptr, 0, 4);
        check(std::fwrite(text.data(), 1, text.size(), contract) == text.size() && std::fflush(contract) == 0,
              "write the exact bounded process contract");
        const std::string path = "/proc/self/fd/" + std::to_string(fileno(contract));
        const char* previous = std::getenv("STRATA_CACHE_VICTIM_REPLAY");
        const bool had = previous != nullptr;
        const std::string saved = previous ? previous : "";
        check(::setenv("STRATA_CACHE_VICTIM_REPLAY", path.c_str(), 1) == 0, "set the diagnostic contract pathname");
        auto& replay = ConversationVictimReplay::process();
        check(replay.configured && replay.count == 1, "the process parses its contract before victim work");
        ConversationCache cache(1u << 20, 2);
        cache.set_mutation_trace(&trace);
        check(cache.put(image({1,2,3})) && cache.put(image({4,5,6})), "seed the actual process-owned replay cache");
        check(cache.evict_oldest() && cache.find(1) && !cache.find(2) && replay.consumed,
              "the process contract consumes its exact generation at the real victim site");
        const auto hash = replay.contract;
        std::rewind(contract);
        check(std::fputs("malformed", contract) >= 0 && std::fflush(contract) == 0 &&
              ::setenv("STRATA_CACHE_VICTIM_REPLAY", "/nonexistent/changed-contract", 1) == 0,
              "change both contract contents and pathname after consumption");
        check(&ConversationVictimReplay::process() == &replay && replay.contract == hash && replay.consumed,
              "contents/environment changes cannot reload or re-arm the process-lifetime contract");
        ConversationCache successor(1u << 20, 2);
        check(successor.put(image({1,2,3})) && successor.put(image({4,5,6})) &&
              successor.evict_oldest() && !successor.find(1) && successor.find(2),
              "a second cache cannot reuse a consumed process nomination with coincident local serials");
        if (had) check(::setenv("STRATA_CACHE_VICTIM_REPLAY", saved.c_str(), 1) == 0, "restore the prior contract environment");
        else check(::unsetenv("STRATA_CACHE_VICTIM_REPLAY") == 0, "restore the absent contract environment");
    }
    std::fclose(output); std::fclose(contract);
#endif
}
void check_victim_replay() {
    // Explicitly injected contracts exercise the production parser and real
    // planners/compactors, without changing the process-lifetime configuration.
    for (const char* site : {"put", "reserve_snapshot", "evict_oldest"}) {
        for (const char* mode : {"default", "keep", "alternate", "malformed"}) {
            FILE* file = std::tmpfile();
            check(file != nullptr, "open the victim-plan fixture trace");
            {
                ConversationCacheTrace trace(file);
                replay_phase(trace);
                ConversationVictimReplay replay;
                constexpr size_t budget = 1u << 20, fixed = 128;
                ConversationCache cache(budget, 3, 0, &replay);
                cache.set_mutation_trace(&trace);
                auto large = image({1,2,3});
                large.live.gdn.resize(4096, 7);
                auto peer = image({7,8,9});
                peer.publication_ticket = 77;
                check(cache.try_reserve_lease(fixed) && cache.put(std::move(large)) &&
                      cache.put(image({4,5,6})) && cache.put(std::move(peer)), "seed two eligible owners and a peer writer");
                cache.guard(3);
                const size_t before = cache.bytes(), small_bytes = cache.find(2)->bytes();
                auto incoming = image({10,11,12});
                const bool put = std::string_view(site) == "put";
                std::string text = victim_contract(cache, trace, std::string_view(mode) == "keep" ? "keep" : "alternate",
                                                  site, {2}, put ? &incoming : nullptr, put ? 4 : 0);
                if (std::string_view(mode) == "malformed") text += "trailing-field";
                if (std::string_view(mode) != "default") replay = ConversationVictimReplay::parse(text);
                cache.set_mutation_trace(&trace);
                size_t protected_index = 2;
                const bool accepted = put ? cache.put(std::move(incoming)) :
                    std::string_view(site) == "reserve_snapshot" ?
                        cache.reserve_snapshot(budget - fixed - (before - small_bytes), 0, &protected_index) :
                        cache.evict_oldest();
                const bool alternate = std::string_view(mode) == "alternate";
                check(accepted && cache.find(alternate ? 1 : 2) && !cache.find(alternate ? 2 : 1) &&
                      cache.find(3) && cache.evictions() == 1 && cache.leased_bytes() == fixed &&
                      cache.envelope_bytes() <= budget, "each real victim site commits its exact plan within the same envelope");
                if (std::string_view(site) == "reserve_snapshot")
                    check(protected_index == cache.index_of(3), "protected fallback index follows alternate compaction");
                if (alternate || std::string_view(mode) == "keep")
                    check(replay.consumed && std::string_view(replay.status) == (alternate ? "forced" : "keep"),
                          "a valid keep/alternate nomination is consumed once at its specified plan");
                if (alternate) {
                    check(cache.evict_oldest() && !cache.find(1) && cache.find(3),
                          "a consumed nomination cannot intercept the next convenient victim");
                    const auto event = trace_event(trace_text(file), site, "plan");
                    check(event.find("\"replay_decision\":\"forced\"") != std::string::npos &&
                          event.find("\"kind\":\"cache_victim_nomination\"") != std::string::npos &&
                          event.find("\"serial\":2,") != std::string::npos,
                          "the live plan records the nomination and actual alternate choice");
                }
                bool retained = false;
                auto alias = cache.take(cache.index_of(3), retained);
                check(retained && cache.has_publication(77) && alias.live.ids == std::vector<int32_t>({7,8,9}) &&
                      !cache.take_publication(77), "a peer read alias and guarded publication remain owned");
                cache.unguard();
                const size_t peer_bytes = cache.find(3)->bytes(), envelope = cache.envelope_bytes();
                auto writer = cache.take_publication(77);
                check(writer && cache.envelope_bytes() == envelope && cache.leased_bytes() == fixed + peer_bytes,
                      "writer acquisition transfers, rather than refunds, the peer image charge");
                cache.release_publication(77);
                cache.release_publication(77);
                check(cache.leased_bytes() == fixed + peer_bytes,
                      "repeated late/cancel cleanup cannot refund an acquired peer writer");
                alias = {}; writer.reset();
                cache.release_lease(peer_bytes);
                check(cache.leased_bytes() == fixed, "peer consumer drain preserves the permanent checkpoint lease");
                trace.finish();
                const auto terminal = trace_text(file);
                check(terminal.find(alternate ? "\"replay_forced\":1" : "\"replay_forced\":0") != std::string::npos,
                      "terminal receipt bounds the intervention to exactly one plan");
            }
            std::fclose(file);
        }
    }
    for (const char* failure : {"pin", "ticket", "guard", "protected", "generation", "history", "scope",
                                "bytes", "subject", "event", "sequence", "leg", "identity", "insufficient", "trace"}) {
        FILE* file = std::tmpfile();
        check(file != nullptr, "open the rejected-nomination trace");
        {
            ConversationCacheTrace trace(file);
            replay_phase(trace);
            ConversationVictimReplay replay;
            constexpr size_t budget = 1u << 20;
            ConversationCache cache(budget, 6, 0, &replay);
            cache.set_mutation_trace(&trace);
            auto large = image({1,2,3});
            large.live.gdn.resize(4096, 7);
            auto pin = image({7,8,9});
            pin.checkpoints.push_back(pin.live); pin.checkpoints.back().pinned = true;
            auto pending = image({10,11,12}); pending.publication_ticket = 77;
            check(cache.put(std::move(large)) && cache.put(image({4,5,6})) &&
                  cache.put(std::move(pin)) && cache.put(std::move(pending)) &&
                  cache.put(image({13,14,15})) && cache.put(image({16,17,18})), "seed every exclusion class");
            cache.guard(5);
            const std::string_view why = failure;
            const uint64_t nominated = why == "pin" ? 3 : why == "ticket" ? 4 :
                why == "guard" ? 5 : why == "protected" ? 6 : 2;
            replay = ConversationVictimReplay::parse(victim_contract(cache, trace, "alternate", "reserve_snapshot", {nominated}));
            check(replay.configured, "negative fixture starts from a fully parsed nomination");
            if (why == "generation") replay.victims[0].serial = 999;
            if (why == "history") ++replay.victims[0].history;
            if (why == "scope") ++replay.victims[0].scope;
            if (why == "bytes") ++replay.victims[0].bytes;
            if (why == "subject") replay.subject.serial = 1;
            if (why == "event") --replay.event;
            if (why == "sequence") ++replay.request_sequence;
            if (why == "leg") ++replay.leg;
            if (why == "identity") trace.incarnation[0] = '!';
            cache.set_mutation_trace(why == "trace" ? nullptr : &trace);
            size_t protected_index = cache.index_of(6);
            const size_t need = why == "insufficient" ? cache.find(2)->bytes() + 1 : 1;
            check(cache.reserve_snapshot(budget - cache.bytes() + need, 0, &protected_index) &&
                  !cache.find(1) && cache.find(2) && cache.find(3) && cache.has_publication(77) &&
                  cache.find(5) && cache.find(6) && protected_index == cache.index_of(6),
                  "malformed/stale/ineligible/underfunded nominations leave ordinary LRU and all peers intact");
            check(replay.consumed && std::string_view(replay.status) != "forced",
                  "a refused exact nomination cannot retry against later eligibility");
            cache.unguard(); cache.release_publication(77);
            const size_t after = cache.size();
            check(cache.evict_oldest() && cache.size() + 1 == after && !cache.find(2) && cache.find(3),
                  "later cancellation/unprotection never re-arms an ineligible nomination or spends the pin");
        }
        std::fclose(file);
    }
    for (bool pressure : {false, true}) {
        FILE* file = std::tmpfile();
        check(file != nullptr, "open the exact victim-set trace");
        {
            ConversationCacheTrace trace(file);
            replay_phase(trace);
            ConversationVictimReplay replay;
            constexpr size_t budget = 1u << 20;
            ConversationCache cache(budget, 4, 0, &replay);
            cache.set_mutation_trace(&trace);
            auto large = image({1,2,3}); large.live.gdn.resize(4096, 7);
            check(cache.put(std::move(large)) && cache.put(image({4,5,6})) && cache.put(image({7,8,9})),
                  "seed a large LRU owner and a two-member alternate set");
            auto incoming = image({10,11,12});
            const auto text = pressure ?
                victim_contract(cache, trace, "alternate", "reserve_snapshot", {2,3}) :
                victim_contract(cache, trace, "alternate", "put", {2}, &incoming, 4);
            replay = ConversationVictimReplay::parse(text);
            cache.set_mutation_trace(&trace);
            if (pressure) {
                const size_t reclaim = cache.find(2)->bytes() + cache.find(3)->bytes();
                check(cache.reserve_snapshot(budget - cache.bytes() + reclaim) && cache.size() == 1 &&
                      cache.find(1) && !cache.find(2) && !cache.find(3) && cache.evictions() == 2,
                      "the entire nominated set, not an LRU prefix or a proxy, is validated and committed");
            } else {
                check(cache.put(std::move(incoming)) && cache.size() == 4 && cache.find(2) &&
                      cache.evictions() == 0 && replay.consumed && std::string_view(replay.status) == "no_pressure",
                      "an exact nomination cannot manufacture pressure or evict an otherwise retained owner");
            }
        }
        std::fclose(file);
    }
    FILE* file = std::tmpfile();
    check(file != nullptr, "open the grammar fixture trace");
    {
        ConversationCacheTrace trace(file);
        replay_phase(trace);
        ConversationCache cache(1u << 20, 2, 0, nullptr);
        check(cache.put(image({1,2,3})) && cache.put(image({4,5,6})), "seed strict parser fixtures");
        const auto valid = victim_contract(cache, trace, "alternate", "evict_oldest", {2});
        for (const auto& malformed : {std::string{}, valid + "extra", valid.substr(1),
                std::string(ConversationVictimReplay::max_contract_bytes + 1, 'x')})
            check(!ConversationVictimReplay::parse(malformed).configured, "unknown/truncated/oversize grammar is default-off");
        const auto duplicate = victim_contract(cache, trace, "alternate", "reserve_snapshot", {2,2});
        check(!ConversationVictimReplay::parse(duplicate).configured, "duplicate allocation generations cannot be nominated twice");
        auto replay = ConversationVictimReplay::parse(valid);
        check(replay.configured && replay.site == "evict_oldest" && replay.count == 1,
              "parsed site names do not borrow the temporary contract buffer");
        check(!ConversationVictimReplay::parse(
            "strata-cache-victim-replay-v1 mode omit").configured, "a publication-omission mode is not a victim contract");
    }
    std::fclose(file);
}
void check_mutation_work_trace() {
    FILE* file = std::tmpfile();
    check(file != nullptr, "open the transfer-work trace");
    {
        ConversationCacheTrace trace(file);
        trace.set_sample(nullptr, +[](void*, ConversationCacheTrace::External&) {});
        ConversationCache cache(1u << 20, 2);
        cache.set_mutation_trace(&trace);
        cache.trace_transfer({.site="capture", .action="work_unavailable", .measured_ms=3.25}, 0, 0);
        const strata::program::PrefixWork checkpoint_work{4096, 17};
        cache.trace_transfer({.site="checkpoint", .action="restore_complete", .measured_ms=1.25},
                             checkpoint_work.bytes, checkpoint_work.route);
        cache.trace_transfer({.site="checkpoint", .action="zero_known", .measured_ms=0.5}, 0, 17);
        cache.trace_transfer({.site="checkpoint", .action="overflow_unavailable", .measured_ms=2.5},
                             UINT64_MAX, 17);
        const auto text = trace_text(file);
        const auto unavailable = trace_event(text, "capture", "work_unavailable");
        const auto checkpoint = trace_event(text, "checkpoint", "restore_complete");
        check(unavailable.find("\"work_bytes\":0,\"route\":0,\"work_known\":false") != std::string::npos,
              "unavailable transfer work is explicit, never a measured zero-byte/zero-route transfer");
        check(checkpoint.find("\"work_bytes\":4096,\"route\":17,\"work_known\":true") != std::string::npos &&
              checkpoint.find("\"measured_ms\":1.25") != std::string::npos,
              "checkpoint receipts retain the computed transfer extent, route and measured duration");
        check(trace_event(text, "checkpoint", "zero_known").find(
                  "\"work_bytes\":0,\"route\":17,\"work_known\":true") != std::string::npos,
              "a known zero-byte transfer with a real route is distinct from unavailable work");
        check(trace_event(text, "checkpoint", "overflow_unavailable").find(
                  "\"work_known\":false") != std::string::npos,
              "overflowed extents are unavailable even when the route is known");
    }
    std::fclose(file);
}

void check_mutation_trace() {
    FILE* file = std::tmpfile();
    check(file != nullptr, "open a bounded metadata trace");
    {
        ConversationCacheTrace trace(file);
        trace.request_sequence = 9;
        std::strcpy(trace.request, "trace-fixture");
        std::strcpy(trace.incarnation, "0123456789abcdef0123456789abcdef");
        trace.leg = 2;
        size_t samples = 0;
        trace.set_sample(&samples, +[](void* context, ConversationCacheTrace::External& e) {
            ++*static_cast<size_t*>(context);
            e.checkpoint_lease = 128; e.preparation = 29; e.outside = 17;
            e.store.held = 41; e.store.metadata = 11; e.store.descriptors = 13;
            e.store.decode = 17; e.store.allowance = 65536;
        });
        constexpr size_t budget = 1u << 20;
        ConversationCache cache(budget, 4);
        check(cache.try_reserve_lease(128), "trace fixture reserves fixed checkpoints");
        auto pin = image({1,2,3});
        pin.checkpoints.push_back(pin.live);
        pin.checkpoints.back().pinned = true;
        auto pending = image({4,5,6});
        pending.publication_ticket = 77;
        check(cache.put(std::move(pin)) && cache.put(std::move(pending)) &&
              cache.put(image({7,8,9})) && cache.put(image({10,11,12})), "fill all four eligibility classes");
        const auto guarded = cache.best(std::vector<int32_t>{7,8,9,99}, {}, true, test_scope);
        cache.guard(guarded.serial);
        cache.set_mutation_trace(&trace);
        check(cache.evict_oldest() && cache.find(4) == nullptr, "explicit eviction still selects serial four");
        check(cache.put(image({13,14,15})) && cache.put(image({16,17,18})),
              "put pressure still selects the only ordinary oldest owner");
        std::vector<ConversationKv> kv(1);
        kv[0].k.resize(32, 1);
        cache.retain(std::move(kv), 3);
        size_t protected_bytes = 0;
        cache.for_each_image([&](size_t index, const auto& e) { if (index < 3) protected_bytes += e.bytes(); });
        size_t protected_index = 1;
        check(cache.reserve_snapshot(budget - 128 - 17 - protected_bytes, 17, &protected_index) &&
              protected_index == 1 && cache.size() == 3, "snapshot pressure reclaims only serial six");
        const std::string text = trace_text(file);
        for (const char* site : {"evict_oldest", "put", "reserve_snapshot"}) {
            const auto event = trace_event(text, site, "plan");
            check(event.find("\"request_sequence\":9,\"request\":\"trace-fixture\",\"leg\":2") != std::string::npos &&
                  event.find("\"incarnation\":\"0123456789abcdef0123456789abcdef\"") != std::string::npos,
                  "each victim site binds the actual command and request leg");
            check(event.find("\"candidate_count\":4,\"candidates_complete\":true,\"external_complete\":true") !=
                      std::string::npos && event.find("\"leased_bytes\":128") != std::string::npos &&
                  event.find("\"checkpoint_lease\":128,\"preparation\":29,\"outside\":17") != std::string::npos,
                  "each victim site records its full candidates and held ledger");
            check(event.find("\"scope_hash\":") != std::string::npos && event.find("\"history_complete\":true") !=
                      std::string::npos && event.find("\"checkpoint_bytes\":") != std::string::npos &&
                  event.find("\"pinned\":true") != std::string::npos &&
                  event.find("\"publication_ticket\":77") != std::string::npos &&
                  event.find("\"guarded\":true") != std::string::npos &&
                  event.find("\"eligible\":true,\"chosen\":true") != std::string::npos,
                  "each victim site exposes actual identities, byte classes, exclusions and choice");
            const auto chosen = event.find("\"eligible\":true,\"chosen\":true");
            const auto chosen_begin = event.rfind('{', chosen);
            const auto chosen_end = event.find('\n', chosen);
            const unsigned expected_serial = std::string_view(site) == "evict_oldest" ? 4 :
                std::string_view(site) == "put" ? 5 : 6;
            check(event.substr(chosen_begin, chosen_end - chosen_begin).find(
                      "\"serial\":" + std::to_string(expected_serial) + ",") != std::string::npos,
                  "each real victim plan names exactly the serial the unchanged LRU path removes");
        }
        const auto reserve = trace_event(text, "reserve_snapshot", "plan");
        check(reserve.find("\"held\":17") != std::string::npos &&
              reserve.find("\"protected_serial\":2") != std::string::npos &&
              reserve.find("\"reuse_bytes\":0") == std::string::npos &&
              reserve.find("\"serial\":6,") != std::string::npos,
              "snapshot plan includes transient held, retained KV, protected fallback and exact victim serial");
        const size_t before_samples = samples;
        cache.set_mutation_trace(nullptr);
        const long before_bytes = std::ftell(file);
        const size_t before_envelope = cache.envelope_bytes();
        check(!cache.evict_oldest() && !cache.reserve_snapshot(budget) &&
              !cache.put(image({40,41,42}), budget) && cache.envelope_bytes() == before_envelope,
              "disabled observer preserves protected refusal and the exact envelope");
        check(samples == before_samples && std::ftell(file) == before_bytes,
              "disabled observer performs no callback, hashing or output");
        cache.unguard();
        bool retained = false;
        auto taken = cache.take(cache.index_of(3), retained);
        check(!retained && taken.live.ids == std::vector<int32_t>({7,8,9}) &&
              cache.put(std::move(taken)), "disabled take/return preserves the real payload and policy");
        trace.finish();
        check(trace_text(file).find("\"limited\":false,\"hash_limited\":false") != std::string::npos,
              "the focused capture ends without silent truncation");
    }
    std::fclose(file);
    file = std::tmpfile();
    check(file != nullptr, "open the saturation trace");
    {
        ConversationCacheTrace trace(file);
        ConversationCache cache(1u << 20, 1);
        cache.set_mutation_trace(&trace);
        trace.set_sample(nullptr, +[](void*, ConversationCacheTrace::External&) { throw std::bad_alloc(); });
        check(cache.put(image({40,41,42})) && cache.size() == 1,
              "a failed observer cannot refuse or consume a real cache admission");
        cache.clear();
        for (size_t i = 0; i < ConversationCacheTrace::max_records; ++i)
            cache.trace_observation({.site="bounded", .action="observe"});
        const auto text = trace_text(file);
        check(text.find("\"limited\":true") != std::string::npos &&
              text.find("\"external_complete\":false") != std::string::npos &&
              size_t(std::count(text.begin(), text.end(), '\n')) <= ConversationCacheTrace::max_records &&
              text.size() < ConversationCacheTrace::max_records * ConversationCacheTrace::max_line,
              "record pressure closes explicitly inside the fixed record and byte bounds");
        check(cache.put(image({1,2,3})) && cache.size() == 1, "trace saturation never refuses cache admission");
    }
    std::fclose(file);
}
void check_restore_charge_traces(const char* prefix) {
    // Two actual allocation failures: after one owner seals, and after all
    // owners seal and the first alias metadata allocation succeeds.
    for (int failure : {-1, 1, 5}) {
        const char* label = failure < 0 ? "success" : failure == 1 ? "partial" : "alias";
        const std::string path = prefix ? std::string(prefix) + "-" + label + ".jsonl" : "";
        FILE* file = prefix ? std::fopen(path.c_str(), "w+x") : std::tmpfile();
        check(file != nullptr, "open bounded restore charge trace");
        {
            ConversationCacheTrace trace(file);
            trace.set_sample(nullptr, +[](void*, ConversationCacheTrace::External& external) {
                external.checkpoint_lease = 128;
            });
            ConversationCache cache(1 << 20, 2, 0, nullptr);
            cache.set_mutation_trace(&trace);
            check(cache.try_reserve_lease(128), "trace fixture owns its fixed checkpoint lease");
            auto donor = image({1, 2, 3, 4, 5, 6});
            donor.live.ple.assign(32, 11);
            for (const auto& ids : {std::vector<int32_t>{1, 2}, std::vector<int32_t>{1, 2, 3, 4}}) {
                ConversationCheckpoint cut;
                cut.ids = ids;
                cut.gdn.assign(32, 12);
                donor.checkpoints.push_back(std::move(cut));
            }
            const size_t before = donor.bytes(), growth = donor.freeze_growth_bytes();
            check(growth == 4 * CheckpointBytes::owner_bytes, "fixture has exactly four unsealed payload owners");
            check(cache.put(std::move(donor)), "park traced shallow-restore donor");
            const auto match = cache.best(std::vector<int32_t>{1, 2, 9}, {}, true, test_scope);
            bool failed = false;
            const long allocations_before = test_allocation_failure::live.load(std::memory_order_relaxed);
            {
                ConversationCache::Restore restore(cache);
                test_allocation_failure::after = failure;
                try {
                    check(restore.select(match), "traced shallow restore selects its validated image");
                } catch (const std::bad_alloc&) {
                    failed = true;
                }
                test_allocation_failure::after = -1;
                const size_t charged = failure == 1 ? CheckpointBytes::owner_bytes : growth;
                const auto* retained = cache.find(match.serial);
                check(failed == (failure >= 0) && retained &&
                      retained->live.ids == std::vector<int32_t>({1, 2, 3, 4, 5, 6}) &&
                      cache.bytes() == before + charged,
                      "successful and partially failed freezes preserve the donor and charge every new owner");
                check(retained && retained->live.gdn.sealed() &&
                      retained->live.ple.sealed() == (failure != 1),
                      "injection reaches partial freezing or the post-freeze alias allocation");
                check(cache.leased_bytes() == 128 + (restore.image() ? restore.image()->bytes() : 0) &&
                      cache.envelope_bytes() <= (1 << 20),
                      "alias failure gives back only its own reservation, never frozen donor or fixed ownership");
            }
            check(test_allocation_failure::live.load(std::memory_order_relaxed) == allocations_before +
                      long((failure == 1 ? CheckpointBytes::owner_bytes : growth) / CheckpointBytes::owner_bytes),
                  "failed partial aliases free their metadata; only the donor's completed freeze owners remain");
            check(cache.leased_bytes() == 128 && cache.find(match.serial) != nullptr &&
                  cache.envelope_bytes() == cache.bytes() + 128,
                  "restore completion and failure both preserve fixed ownership without undercharge");
            cache.clear();
            check(cache.leased_bytes() == 128, "poison clear cannot refund the fixed checkpoint lease");
            trace.finish();
            const auto text = trace_text(file);
            const auto transfer = trace_event(text, "restore", "freeze_charge");
            check(transfer.find("\"subject_serial\":1,") != std::string::npos &&
                  transfer.find("\"incoming\":" + std::to_string(failure == 1 ?
                      CheckpointBytes::owner_bytes : growth) + ",") != std::string::npos,
                  "freeze growth is an explicit same-owner lease-to-parked charge transfer");
            check(text.find("\"limited\":false,\"hash_limited\":false") != std::string::npos,
                  "all three bounded native restore traces close without truncation");
        }
        std::fclose(file);
    }
}
}

int main(int argc, char** argv) {
    if (argc != 1 && (argc != 3 || std::strcmp(argv[1], "--restore-traces") != 0)) return 2;
    check_restore_charge_traces(argc == 3 ? argv[2] : nullptr);
    check_victim_process_contract();
    check_mutation_work_trace();
    check_mutation_trace();
    check_victim_replay();
    {
        auto first = image({11, 12, 13});
        auto second = image({21, 22, 23});
        auto third = image({31, 32, 33});
        first.freeze();
        second.freeze();
        third.freeze();
        const size_t one = first.bytes(), fixed = 3 * one;
        ConversationCache cache(fixed + 2 * one, 4);
        check(cache.try_reserve_lease(fixed), "empty cache reserves lifetime resident checkpoint ownership");
        auto pending = SavedConversation(first);
        pending.publication_ticket = 91;
        check(cache.put(std::move(pending)) && cache.put(SavedConversation(second)),
              "parking uses the remainder after fixed checkpoint ownership");
        auto publication = cache.take_publication(91);
        check(publication.has_value() && same_image(*publication, first) &&
                  cache.leased_bytes() == fixed + publication->bytes(),
              "ordinary publication transfers its own charge alongside the fixed lease");
        if (!publication) return 1;
        const size_t publication_bytes = publication->bytes();
        publication.reset();
        cache.release_lease(publication_bytes);
        check(cache.leased_bytes() == fixed && cache.can_fit(one),
              "publication refund restores parking room without refunding fixed ownership");
        check(cache.put(SavedConversation(third)), "refunded publication room accepts another parked image");
        const size_t before = cache.envelope_bytes();
        check(!cache.reserve_snapshot(2 * one + 1) && cache.size() == 2 &&
                  cache.envelope_bytes() == before && cache.evictions() == 0,
              "impossible reservation cannot spend fixed ownership or destroy parked entries");
        cache.for_each_image([&](size_t index, const auto& current) {
            check(same_image(current, index == 0 ? second : third),
                  "impossible reservation preserves actual parked payloads");
        });
        check(cache.reserve_snapshot(one) && cache.size() == 1 && cache.evictions() == 1 &&
                  cache.leased_bytes() == fixed,
              "feasible reservation evicts an ordinary image but never the fixed lease");
        check(cache.best(std::vector<int32_t>{31, 32, 33, 34}, {}, true, test_scope).tokens == 3 &&
                  cache.best(std::vector<int32_t>{21, 22, 23, 24}, {}, true, test_scope).tokens == 0,
              "reservation evicts the oldest payload and leaves the newer consumer match");
        check(cache.clear() == 1 && cache.bytes() == 0 && cache.leased_bytes() == fixed,
              "poison-style clear removes images without releasing resident checkpoint ownership");
        check(cache.put(SavedConversation(first)) && cache.put(SavedConversation(second)) &&
                  !cache.try_reserve_lease(1),
              "clear leaves exactly the reduced parking envelope available");
    }
    {
        auto make = [] {
            auto s = image({1,2,3,4,5,6,7,8,9});
            auto cut = s.live; cut.ids.resize(5); s.checkpoints.push_back(cut);
            for (int fmt : {3,1}) { // current profile: K8V4 main, INT8 draft
                const bool index = fmt == 3;
                auto shape = strata::kernels::qsa_real_shapes();
                shape.n_head_kv=1; shape.head_dim=64; shape.idx_dim=8;
                strata::plan::QsaImageBytes bytes;
                check(strata::plan::qsa_image_bytes(shape,9,fmt,index,bytes), "probe fixture layout");
                ConversationKv kv;
                kv.format=fmt; kv.cells=bytes.cells; kv.page_size=shape.page_size;
                kv.heads=1; kv.head_dim=64; kv.idx_dim=8; kv.pooled_rows=bytes.pooled_rows;
                kv.k.resize(bytes.k,7); kv.v.resize(bytes.v,7);
                kv.k_scale.resize(bytes.k_scale,7); kv.v_scale.resize(bytes.v_scale,7);
                kv.pooled.resize(bytes.pooled,7);
                s.kv.push_back(std::move(kv));
            }
            return s;
        };
        auto a=make(), b=make();
        b.live.ids[6]=17; // sibling branch after an actual common checkpoint
        auto r=probe_prefix_extents(a,b);
        check(std::string_view(r.status)=="equal" && r.cut==5 && r.eligible_bytes==968 &&
              r.duplicate_bytes==968 && r.aliased_bytes==0 && r.compared_bytes==968,
              "independent backing counted for complete K8V4 and draft pages plus completed index row");
        auto change = [](ConversationBuffer& data, size_t at) {
            check(data.visit(at,1,[](uint8_t* p,size_t,size_t) { *p=23; return true; }), "probe fixture mutation");
        };
        change(b.kv[0].k,4*64); change(b.kv[0].pooled,32); change(b.kv[1].k,4*64);
        r=probe_prefix_extents(a,b);
        check(std::string_view(r.status)=="equal" && r.duplicate_bytes==968,
              "partial pages, moving index row and draft tail excluded from duplicate witness");
        change(b.kv[0].k,0);
        r=probe_prefix_extents(a,b);
        check(std::string_view(r.status)=="different" && r.duplicate_bytes==0 && r.compared_bytes>0,
              "equal token prefixes cannot hide unequal authoritative KV");
        b=make(); b.scope=ConversationScope{{2}};
        r=probe_prefix_extents(a,b);
        check(std::string_view(r.status)=="scope" && r.compared_bytes==0, "foreign scope never reads KV");
        b=make(); b.cvec=!a.cvec;
        r=probe_prefix_extents(a,b);
        check(std::string_view(r.status)=="scope" && r.compared_bytes==0, "steering mismatch never reads KV");
        b=make(); b.checkpoints.clear();
        r=probe_prefix_extents(a,b);
        check(std::string_view(r.status)=="no-cut" && r.compared_bytes==0, "equal tokens do not manufacture recurrent cut");
        b=make(); b.kv[1].v.pop_back();
        r=probe_prefix_extents(a,b);
        check(std::string_view(r.status)!="equal" && r.compared_bytes==0, "malformed final draft payload rejected before comparisons");
        b=make();
        r=probe_prefix_extents(a,b,1);
        check(std::string_view(r.status)=="limit" && r.compared_bytes==0, "budget refusal is not a zero-opportunity finding");
        r=probe_prefix_extents(a,a);
        check(std::string_view(r.status)=="equal" && r.duplicate_bytes==0 && r.aliased_bytes==968,
              "already shared backing never counted as duplicated allocation");
        a.checkpoints[0].ids.resize(4); b.checkpoints[0].ids.resize(4);
        r=probe_prefix_extents(a,b);
        check(std::string_view(r.status)=="equal" && r.duplicate_bytes==440,
              "draft final page stays private even when recurrent cut aligns to page boundary");
        a.stage_images.push_back(make()); b.stage_images.push_back(make());
        r=probe_prefix_extents(a,b);
        check(std::string_view(r.status)=="no-cut", "every stage must hold the same recurrent cut");
        a.checkpoints[0].ids.push_back(5); b.checkpoints[0].ids.push_back(5);
        r=probe_prefix_extents(a,b);
        check(std::string_view(r.status)=="equal" && r.duplicate_bytes==1936,
              "complete staged siblings include each stage exactly once");
        const size_t charged=a.bytes();
        const auto* payload=a.live.gdn.data();
        (void) probe_prefix_extents(a,b);
        check(a.bytes()==charged && a.live.gdn.data()==payload && a.kv[0].k==b.kv[0].k,
              "observer does not change accounting, ownership or contents");
    }
    {
        auto make_segmented = [](bool shifted) {
            auto s = image({});
            s.live.ids.assign(4097, 1);
            s.checkpoints.push_back(s.live);
            s.checkpoints.back().ids.resize(4096);
            auto shape = strata::kernels::qsa_real_shapes();
            shape.n_head_kv = 129; shape.head_dim = 64; shape.idx_dim = 8;
            strata::plan::QsaImageBytes bytes;
            check(strata::plan::qsa_image_bytes(shape, 4097, 3, true, bytes), "segmented probe layout");
            ConversationKv kv;
            kv.format = 3; kv.cells = bytes.cells; kv.page_size = shape.page_size;
            kv.heads = 129; kv.head_dim = 64; kv.idx_dim = 8; kv.pooled_rows = bytes.pooled_rows;
            if (shifted) kv.k.resize(17, 7);
            kv.k.resize(bytes.k, 7); kv.v.resize(bytes.v, 7);
            kv.k_scale.resize(bytes.k_scale, 7); kv.v_scale.resize(bytes.v_scale, 7);
            kv.pooled.resize(bytes.pooled, 7);
            s.kv.push_back(std::move(kv));
            return s;
        };
        auto a = make_segmented(false), b = make_segmented(true);
        auto r = probe_prefix_extents(a, b);
        check(std::string_view(r.status) == "equal" && r.cut == 4096,
              "equal eligible bytes survive different segment boundaries");
        check(b.kv[0].k.visit(2 * ConversationBuffer::segment_bytes + 7, 1,
                             [](uint8_t* p, size_t, size_t) { *p = 23; return true; }),
              "third segment mutation lies inside the eligible prefix");
        r = probe_prefix_extents(a, b);
        check(std::string_view(r.status) == "different" && r.duplicate_bytes == 0,
              "a third-segment mismatch invalidates the complete-cut witness");
    }
    {
        // Explicit buffer policy keeps the rest of this executable on its normal
        // default path; no environment-dependent expectations in these fixtures.
        auto make = [](int branch) {
            auto s = image({});
            s.live.ids.assign(2305, 1);
            s.live.ids[2100] = branch;
            auto root = s.live; root.ids.resize(2048); s.checkpoints.push_back(root);
            auto deep = s.live; deep.ids.resize(2304); s.checkpoints.push_back(deep);
            for (int fmt : {3, 1}) {
                const bool index = fmt == 3;
                auto shape = strata::kernels::qsa_real_shapes();
                shape.n_head_kv = 1; shape.head_dim = 64; shape.idx_dim = 8;
                strata::plan::QsaImageBytes bytes;
                check(strata::plan::qsa_image_bytes(shape, 2305, fmt, index, bytes), "sharing fixture layout");
                ConversationKv kv;
                kv.format = fmt; kv.cells = bytes.cells; kv.page_size = shape.page_size;
                kv.heads = 1; kv.head_dim = 64; kv.idx_dim = 8; kv.pooled_rows = bytes.pooled_rows;
                const std::array<ConversationBuffer*, 5> parts{&kv.k,&kv.v,&kv.k_scale,&kv.v_scale,&kv.pooled};
                const std::array<uint64_t, 5> sizes{bytes.k,bytes.v,bytes.k_scale,bytes.v_scale,bytes.pooled};
                for (size_t i = 0; i < parts.size(); ++i) {
                    *parts[i] = ConversationBuffer(true);
                    parts[i]->resize(size_t(sizes[i]), 7);
                }
                s.kv.push_back(std::move(kv));
            }
            return s;
        };
        auto pointer = [](const ConversationBuffer& b, size_t at) {
            const uint8_t* out = nullptr;
            b.visit(at, 1, [&](const uint8_t* p, size_t, size_t) { out = p; return true; });
            return out;
        };
        auto a = make(2), b = make(3), c = make(4);
        PrefixBackingLedger before; before.add(a); before.add(b); before.add(c);
        const auto* tail = pointer(b.kv[0].k, 2048 * 64);
        auto r = share_prefix_extents(b, a);
        check(std::string_view(r.status) == "equal", "authorized completed common cut accepted");
        share_prefix_extents(c, a);
        auto aliases = probe_prefix_extents(a, b);
        check(aliases.aliased_bytes > 0 && aliases.duplicate_bytes > 0,
              "complete allocations share; subextent scales and private tails remain separate");
        PrefixBackingLedger after; after.add(a); after.add(b); after.add(c);
        check(!after.truncated && after.unique_payload < before.unique_payload &&
              before.unique_payload - after.unique_payload == 2 * aliases.aliased_bytes &&
              after.logical_payload == before.logical_payload,
              "unique allocation ledger counts three siblings once, not three repeated pairs");
        check(pointer(b.kv[0].k, 2048 * 64) == tail,
              "sharing frees full prefix allocations without relocating the private tail");
        check(pointer(a.kv[1].k, 2044 * 64) != pointer(b.kv[1].k, 2044 * 64),
              "draft last possible uncomputed page stays private at aligned cut");
        const size_t peak = b.kv[0].k.allocation_peak(b.kv[0].k.size());
        check(peak >= b.kv[0].k.bytes() + 2 * ConversationBuffer::prefix_extent_bytes,
              "same-size rewrite admission includes every shared extent replacement");
        const auto* original = pointer(a.kv[0].k, 0);
        check(b.kv[0].k.visit(0, 1, [](uint8_t* p, size_t, size_t) { *p = 29; return true; }),
              "mutable visitor can rewrite a shared prefix after detaching");
        check(*pointer(a.kv[0].k, 0) == 7 && *pointer(c.kv[0].k, 0) == 7 &&
              pointer(b.kv[0].k, 0) != original && *pointer(b.kv[0].k, 0) == 29,
              "prefix mutation leaves both other branches and their backing intact");
        check(peak >= b.kv[0].k.bytes(), "detachment stays inside admitted buffer peak");
        check(c.kv[0].k.allocation_peak(17) >= c.kv[0].k.bytes() + ConversationBuffer::prefix_extent_bytes,
              "partial shrink prices detachment before changing segment length");
        c.kv[0].k.resize(17);
        check(a.kv[0].k.size() > 17 && *pointer(a.kv[0].k, 0) == 7,
              "shrinking shared prefix cannot truncate the donor");
        for (int mismatch = 0; mismatch < 6; ++mismatch) {
            auto isolated = make(5);
            if (mismatch == 0) isolated.scope = ConversationScope{{2}};
            if (mismatch == 1) isolated.cvec = !a.cvec;
            if (mismatch == 2) isolated.geometry[0] = 1;
            if (mismatch == 3) isolated.layer_hi = 1;
            if (mismatch == 4) isolated.checkpoints.erase(isolated.checkpoints.begin());
            if (mismatch == 5) isolated.kv.back().v.pop_back();
            const auto* own = pointer(isolated.kv[0].k, 0);
            auto refused = share_prefix_extents(isolated, a);
            check(std::string_view(refused.status) != "equal" && refused.compared_bytes == 0 &&
                  pointer(isolated.kv[0].k, 0) == own,
                  "scope, steering, geometry, carve, checkpoint and format-size refusals preserve ownership");
        }
        auto unequal = make(6);
        unequal.kv.back().v.visit(0, 1, [](uint8_t* p, size_t, size_t) { *p = 99; return true; });
        const auto* unchanged = pointer(unequal.kv[0].k, 0);
        r = share_prefix_extents(unequal, a);
        check(std::string_view(r.status) == "different" && pointer(unequal.kv[0].k, 0) == unchanged,
              "late-layer mismatch rejects entire cut before sharing earlier layers");
        auto staged_a = make(2), staged_b = make(3);
        staged_a.stage_images.push_back(make(2)); staged_b.stage_images.push_back(make(3));
        staged_b.stage_images[0].checkpoints.erase(staged_b.stage_images[0].checkpoints.begin());
        r = share_prefix_extents(staged_b, staged_a);
        check(std::string_view(r.status) != "equal" && probe_prefix_extents(staged_a, staged_b).aliased_bytes == 0,
              "missing checkpoint on one stage prevents sharing every stage");
        staged_b.stage_images[0] = make(3);
        r = share_prefix_extents(staged_b, staged_a);
        check(std::string_view(r.status) == "equal" &&
              probe_prefix_extents(staged_a, staged_b).aliased_bytes == 2 * aliases.aliased_bytes,
              "all-stage identity shares each stage exactly once");
        auto owner = make(2), survivor = make(3);
        share_prefix_extents(survivor, owner);
        const auto* surviving = pointer(survivor.kv[0].k, 0);
        ConversationCache cache(32u << 20, 4);
        check(cache.put(std::move(owner)) && cache.put(std::move(survivor)), "shared siblings occupy original cache slots");
        check(cache.evict_oldest(), "donor can be evicted with live sibling references");
        bool retained_in_cache = true;
        auto remaining = cache.take(0, retained_in_cache);
        check(!retained_in_cache && pointer(remaining.kv[0].k, 0) == surviving && *surviving == 7,
              "eviction and take preserve exactly the surviving allocation");
        cache.retain(std::move(remaining.kv), 2305);
        auto active = cache.take_reuse();
        check(pointer(active.kv[0].k, 0) == surviving, "active retained reuse keeps the same backing");
        const auto* same = pointer(active.kv[0].k, 0);
        active.kv[0].k.visit(0, 1, [](uint8_t* p, size_t, size_t) { *p = 31; return true; });
        check(pointer(active.kv[0].k, 0) == same && *same == 31,
              "last surviving owner writes without a redundant detach allocation");
        auto limited = make(7);
        r = share_prefix_extents(limited, a, 1);
        check(std::string_view(r.status) == "limit" && r.compared_bytes == 0,
              "comparison limit refuses sharing without a partial publication");
        PrefixBackingLedger bounded;
        for (size_t i = 0; i <= PrefixBackingLedger::max_backings; ++i) bounded.add(a.kv[0].k);
        check(bounded.truncated, "diagnostic reference bound reports incomplete coverage explicitly");
    }
    {
        auto source = image({1, 2, 3});
        source.live.stage_parts.resize(1);
        source.live.stage_parts[0].gdn.assign(32, 9);
        source.live.seal();
        const auto* root = source.live.gdn.data();
        const auto* stage = source.live.stage_parts[0].gdn.data();
        auto reader = source.live;
        check(reader.gdn.data()==root && reader.stage_parts[0].gdn.data()==stage,
              "sealed checkpoint copies retain exact root and stage allocations");
        reader.used = 17; reader.pinned = true;
        check(source.live.used!=reader.used && !source.live.pinned,
              "shared payload does not share LRU or pin metadata");
        reader.gdn.write()[0] = 42;
        check(reader.gdn.data()!=root && source.live.gdn[0]==7 && reader.gdn[0]==42,
              "mutating a sealed checkpoint detaches without changing another branch");
        auto survivor = source.live;
        source = {};
        check(survivor.gdn.data()==root && survivor.stage_parts[0].gdn.data()==stage,
              "evicting source preserves held root and stage readers");
        const size_t payload_charge = survivor.gdn.bytes();
        auto another = survivor;
        check(another.gdn.bytes()==payload_charge && payload_charge>=survivor.gdn.size(),
              "each shared reference retains its conservative payload charge");
    }
    {
        CheckpointBytes source({7, 11, 13});
        source.resize(5);
        check(source == std::vector<uint8_t>({7, 11, 13, 0, 0}),
              "ordinary checkpoint growth remains zero-filled");
        source.seal();
        auto writer = source;
        writer.resize_for_overwrite(8);
        std::fill(writer.write().begin() + 5, writer.write().end(), uint8_t(29));
        writer.write()[0] = 31;
        writer.seal();
        check(source == std::vector<uint8_t>({7, 11, 13, 0, 0}) &&
              writer == std::vector<uint8_t>({31, 11, 13, 0, 0, 29, 29, 29}) &&
              source.data() != writer.data(),
              "overwrite growth preserves existing bytes and cannot mutate sealed aliases");
        writer.resize(3);
        writer.resize(6);
        check(writer == std::vector<uint8_t>({31, 11, 13, 0, 0, 0}),
              "ordinary growth after overwrite and shrink does not expose stale tail bytes");
    }

    {
        ConversationScope scope;
        check(conversation_scope_parse("0100000000000000000000000000000000000000000000000000000000000000", scope) &&
              scope == test_scope, "exact-width scope round trip");
        for (const auto text : {"", "01", "010000000000000000000000000000000000000000000000000000000000000",
                               "0000000000000000000000000000000000000000000000000000000000000000",
                               "z100000000000000000000000000000000000000000000000000000000000000"})
            check(!conversation_scope_parse(text, scope) && scope == test_scope,
                  "invalid scope is refused without overwriting a populated owner");
        ConversationCache cache(4096, 2);
        auto a = image({1, 2, 3});
        a.checkpoints.emplace_back();
        a.checkpoints.back().ids = {1, 2};
        check(cache.put(std::move(a)), "park conversation A with its scope");
        const ConversationScope other{{2}};
        const std::vector<int64_t> second_turn{1, 2, 3, 4, 5};
        check(cache.best(second_turn, {}, true, test_scope).tokens == 3,
              "parked then resumed turn two reads only its suffix");
        check(cache.best(second_turn, {}, true, other).tokens == 0 &&
              cache.best(second_turn, {}, true, {}).tokens == 0,
              "identical cold conversation and absent namespace cannot recover A");
        auto b = image({1, 2, 3});
        b.scope = other;
        b.checkpoints.emplace_back();
        b.checkpoints.back().ids = {1, 2};
        check(cache.put(std::move(b)) && cache.size() == 2 && cache.superseded() == 0,
              "identical authorized branches remain separate, not superseded across scopes");
        check(cache.bytes() == 2 * (image({1, 2, 3}).bytes() +
              sizeof(ConversationCheckpoint) + 2 * sizeof(int32_t)),
              "both 32-byte namespaces are charged inside the bounded parking budget");
        check(cache.evict_oldest() && cache.best(second_turn, {}, true, test_scope).tokens == 0 &&
              cache.best(second_turn, {}, true, other).tokens == 3,
              "eviction cannot resurrect the retired conversation's namespace");
        auto unscoped = image({1, 2, 3});
        unscoped.scope = {};
        check(!cache.put(std::move(unscoped)), "unscoped snapshot is not parkable");
    }
    {
        ConversationBuffer bytes;
        const size_t first = ConversationBuffer::segment_bytes + 17;
        const size_t peak = bytes.allocation_peak(first);
        bytes.resize(first, 7);
        check(bytes.bytes() <= peak, "segmented payload and directory fit admitted bytes");
        uint8_t* original = nullptr;
        bytes.visit(0, 1, [&](uint8_t* p, size_t, size_t) { original = p; return true; });
        const size_t grown_peak = bytes.allocation_peak(first + 99);
        bytes.resize(first + 99, 9);
        bytes.visit(0, 1, [&](uint8_t* p, size_t, size_t) {
            check(p == original, "appending preserves existing payload addresses"); return true;
        });
        check(bytes.bytes() <= grown_peak, "growth including transient directory fits admission");
        std::vector<uint8_t> tail(116);
        check(bytes.read(tail.data(), ConversationBuffer::segment_bytes, tail.size()), "read spans segment boundaries");
        check(std::all_of(tail.begin(), tail.begin()+17, [](auto v){return v==7;}) &&
              std::all_of(tail.begin()+17, tail.end(), [](auto v){return v==9;}), "growth preserves prefix and initializes only suffix");
        auto copy = bytes;
        bytes.resize(ConversationBuffer::segment_bytes + 10);
        bytes.resize(first, 7); bytes.resize(first + 99, 9);
        check(bytes == copy, "equality ignores differing segmentation after rewind and regrowth");
        check(!bytes.read(tail.data(), bytes.size()-1, 2), "range check rejects a truncated payload");
        check(bytes.allocation_peak(SIZE_MAX) == SIZE_MAX, "allocation estimate rejects overflow");
    }
    {
        ConversationBuffer bytes;
        for (size_t n=1;n<=4096;++n) {
            const size_t peak = bytes.allocation_peak(n*17);
            bytes.resize(n*17,7);
            check(bytes.bytes() <= peak,"small append stays within the predicted allocation peak");
        }
        size_t segments = 0;
        bytes.visit(0,bytes.size(),[&](const uint8_t*,size_t,size_t){++segments;return true;});
        check(segments <= 4,"thousands of small turns do not create thousands of restore transfers");
    }
    {
        // a parked conversation's K/V buffer (16 MiB, as captured) grows by one turn: the reserve for later turns is
        // an eighth of the buffer at most, not another 16 MiB segment
        ConversationBuffer bytes;
        bytes.resize(ConversationBuffer::segment_bytes, 3);
        const size_t captured = bytes.bytes();
        check(captured - bytes.size() < 1024, "a fresh capture reserves nothing");
        const size_t turn = 1 << 20;
        const size_t peak = bytes.allocation_peak(bytes.size() + turn);
        bytes.resize(bytes.size() + turn, 4);
        check(bytes.bytes() <= peak, "a turn's growth stays within the predicted allocation peak");
        check(bytes.bytes() - captured <= turn + ConversationBuffer::segment_bytes / 8 + 1024,
              "the reserve for later turns is at most an eighth of the buffer");
        const size_t reserved = bytes.bytes();
        bytes.resize(bytes.size() + turn / 2, 5);
        check(bytes.bytes() == reserved, "the next small turn fills the reserve instead of allocating");
        size_t segments = 0;
        bytes.visit(0, bytes.size(), [&](const uint8_t*, size_t, size_t) { ++segments; return true; });
        check(segments == 2, "the reserve extends the turn's segment");
    }
    {
        // many turns of 1 MiB onto a 16 MiB buffer: the reserve grows with the buffer, so the segments do not
        // multiply one per turn, and the reserve never exceeds an eighth of the buffer
        ConversationBuffer bytes;
        bytes.resize(ConversationBuffer::segment_bytes, 1);
        for (int k = 0; k < 256; ++k) {
            const size_t before = bytes.size();
            const size_t peak = bytes.allocation_peak(before + (1 << 20));
            bytes.resize(before + (1 << 20), 2);
            check(bytes.bytes() <= peak, "each turn stays within the predicted allocation peak");
            check(bytes.bytes() - bytes.size() <= before / 8 + 4096, "the reserve stays within an eighth");
        }
        size_t segments = 0;
        bytes.visit(0, bytes.size(), [&](const uint8_t*, size_t, size_t) { ++segments; return true; });
        check(segments <= 64, "256 turns of 1 MiB take far fewer than 256 segments");
    }
    {
        ConversationKv layer;
        layer.k.resize(400);
        std::vector<ConversationKv> layers;
        layers.push_back(std::move(layer));
        const size_t retained = layers.capacity()*sizeof(ConversationKv) + layers[0].bytes();
        const size_t parked = image({1,2,3}).bytes();
        ConversationCache cache(retained + parked, 4);
        check(cache.put(image({1,2,3})), "park before retaining active storage");
        cache.retain(std::move(layers), 16);
        check(cache.bytes() == retained + parked && cache.size() == 1, "active retained storage consumes bytes but no parked slot");
        cache.limit_reuse(9); cache.limit_reuse(12);
        auto reuse = cache.take_reuse();
        check(reuse.unchanged_tokens == 9, "a later continuation cannot undo a rewind's dirty boundary");
        check(cache.bytes() == parked, "taking retained buffers releases their budget accounting");
        cache.retain(std::move(reuse.kv), 16);
        check(!cache.can_fit(parked),"retained storage is counted when checking a capture without eviction");
        check(cache.retained_bytes() == retained && cache.size() == 1 && cache.evictions() == 0,
              "optional reuse admission does not evict or release anything");
        check(!cache.can_fit(SIZE_MAX) && !cache.can_fit(1,SIZE_MAX),"non-mutating reservation rejects overflow");
        check(cache.reserve_snapshot(parked), "reservation can discard optional active buffers");
        check(cache.retained_bytes() == 0 && cache.size() == 1 && cache.evictions() == 0,
              "pressure drops retained storage before evicting parked conversations");
    }
    {
        const auto fallback = image({21, 22, 23});
        const size_t one = fallback.bytes(), lease = 128, outside = 256;
        ConversationCache cache(3 * one + lease + outside + 64, 3);
        check(cache.put(image({11, 12, 13})) && cache.put(SavedConversation(fallback)) &&
                  cache.put(image({31, 32, 33})),
              "reclamation starts with an older and newer victim around the fallback");
        check(cache.try_reserve_lease(lease), "an outstanding lease remains separately charged");
        auto match = cache.best(std::vector<int32_t>{21, 22, 23, 24}, {}, true, test_scope);
        check(cache.reserve_snapshot(64, outside, &match.index) && cache.size() == 3,
              "a byte reservation needs no spare parked-entry slot");
        check(cache.reserve_snapshot(2 * one + 64, outside, &match.index),
              "reclamation can remove eligible entries on both sides of a protected fallback");
        check(cache.size() == 1 && cache.evictions() == 2 && match.index == 0 &&
                  cache.bytes() == one && cache.leased_bytes() == lease,
              "compaction rebases the fallback without refunding held ownership");
        bool retained = true;
        auto restored = cache.take(match.index, retained);
        check(!retained && same_image(restored, fallback),
              "compaction preserves the original parked fallback for later consumption");
    }
    {
        auto victim = image({11, 12, 13}), fallback = image({21, 22, 23}),
             pinned = image({31, 32, 33}), pending = image({41, 42, 43});
        pinned.checkpoints.push_back(pinned.live);
        pinned.checkpoints.back().pinned = true;
        pending.publication_ticket = 17;
        pending.freeze();
        std::vector<ConversationKv> layers(1);
        layers[0].k.resize(400, 9);
        const size_t reuse = layers.capacity() * sizeof(ConversationKv) + layers[0].bytes();
        const size_t payload = victim.bytes() + fallback.bytes() + pinned.bytes() + pending.bytes();
        const size_t lease = 128, outside = 256;
        ConversationCache cache(payload + reuse + lease + outside, 4);
        check(cache.put(SavedConversation(victim)) && cache.put(SavedConversation(fallback)) &&
                  cache.put(SavedConversation(pinned)) && cache.put(SavedConversation(pending)),
              "mixed reclamation fixture keeps ordinary, protected, pinned and pending ownership");
        cache.retain(std::move(layers), 16);
        check(cache.try_reserve_lease(lease), "mixed fixture holds a publication lease");
        auto match = cache.best(std::vector<int32_t>{21, 22, 23, 24}, {}, true, test_scope);
        const size_t before = cache.envelope_bytes(), original_index = match.index;
        check(!cache.reserve_snapshot(victim.bytes() + reuse + 1, outside, &match.index),
              "one byte beyond all eligible reclamation refuses atomically");
        check(cache.envelope_bytes() == before && cache.retained_bytes() == reuse &&
                  cache.evictions() == 0 && cache.size() == 4 && match.index == original_index &&
                  cache.has_publication(17) && cache.leased_bytes() == lease,
              "failed reclamation changes no payload, reuse, lease, publication or fallback position");
        const SavedConversation* expected[] = {&victim, &fallback, &pinned, &pending};
        cache.for_each_image([&](size_t index, const auto& current) {
            check(same_image(current, *expected[index]), "impossible reclamation preserves every original image");
        });
    }
    {
        // Layer-split accounting: each directory capacity slot owns the inline
        // stage metadata, including scope; only external payload is added recursively.
        SavedConversation split;
        const size_t solo = split.bytes();
        split.stage_images.reserve(1);
        check(split.bytes() == solo + sizeof(SavedConversation),
              "one reserved stage-image directory slot is charged exactly once");
        const SavedConversation stage = image({1, 2, 3});
        split.stage_images.push_back(stage);
        check(split.bytes() == solo + sizeof(SavedConversation) + stage.bytes() - sizeof(stage.scope) -
                                    SavedConversation::draft_metadata_bytes,
              "one stage image charges its inline scope and certification only through its directory slot");
    }
    {
        // An empty stage owns no external allocation: its entire contribution
        // is already in the preallocated directory, including unused slots.
        SavedConversation split;
        split.scope = test_scope;
        split.stage_images.reserve(3);
        split.stage_images.emplace_back();
        const size_t expected = sizeof(ConversationScope) + SavedConversation::draft_metadata_bytes +
                                split.stage_images.capacity() * sizeof(SavedConversation);
        check(split.bytes() == expected,
              "empty stage adds no payload beyond its full directory allocation");
        ConversationCache short_cache(expected - 1, 1);
        check(!short_cache.put(std::move(split)) && short_cache.bytes() == 0 && short_cache.size() == 0,
              "empty-stage insertion refuses a budget one byte below allocation ownership");
        ConversationCache exact_cache(expected, 1);
        check(exact_cache.put(std::move(split)) && exact_cache.bytes() == expected && exact_cache.size() == 1,
              "empty-stage insertion fits the exact independently derived allocation boundary");
    }
    const std::vector<int64_t> a = {1, 2, 3, 4}, b = {9, 8, 7, 6};
    {
        ConversationCache cache(1024, 2);
        check(cache.put(image({1, 2, 3})), "park A");
        check(cache.put(image({9, 8, 7})), "park B");
        auto match = cache.best(a, {}, true, test_scope);
        check(match.tokens == 3 && match.live, "A/B/A: recover A");
        bool retained_in_cache = true;
        auto restored = cache.take(match.index, retained_in_cache);
        check(!retained_in_cache && restored.live.ids == std::vector<int32_t>({1, 2, 3}), "taking selected A preserves identity");
        check(cache.size() == 1 && cache.best(b, {}, true, test_scope).tokens == 3, "B remains parked");
        check(cache.bytes() == image({9,8,7}).bytes(), "byte accounting after take");
        check(cache.put(std::move(restored)), "park returned A as newest");
        check(cache.put(image({5, 6})), "evict oldest by slot limit");
        check(cache.best(b, {}, true, test_scope).tokens == 0 && cache.best(a, {}, true, test_scope).tokens == 3, "B evicted before A");
        check(cache.evictions() == 1, "eviction counter");
    }
    {
        // evict_oldest(): the physical-RAM admission gate in generate.cpp frees parked conversations
        // one at a time and re-checks the host's free memory, so this has to be exactly the oldest-first
        // step make_room() takes - same accounting, same counters - and it has to stop at empty.
        ConversationCache cache(4096, 4);
        cache.put(image({1, 2, 3}));
        cache.put(image({9, 8, 7}));
        check(cache.size() == 2 && cache.slots() == 4, "slots() reports the configured limit");
        const size_t both = cache.bytes();
        check(cache.evict_oldest(), "evict_oldest reports an eviction");
        check(cache.size() == 1 && cache.evictions() == 1, "evict_oldest drops exactly one, oldest first");
        check(cache.best(a, {}, true, test_scope).tokens == 0 && cache.best(b, {}, true, test_scope).tokens == 3,
              "the oldest conversation went, the newest stayed");
        check(cache.bytes() == both - image({1, 2, 3}).bytes(), "evict_oldest releases the entry's bytes");
        cache.evict_oldest();
        check(cache.size() == 0 && cache.bytes() == 0, "evicting the last parked conversation empties the cache");
        check(!cache.evict_oldest(), "the gate's loop can reach an empty cache: false");
        check(cache.size() == 0 && cache.evictions() == 2, "evict_oldest on an empty cache is a no-op");
    }
    {   // the RAM gate's eviction skips a conversation that holds a pinned shared prefix, and says so (false)
        ConversationCache cache(1 << 20, 4);
        auto pinned = image({1, 2, 3});
        pinned.checkpoints.push_back(ConversationCheckpoint{});
        pinned.checkpoints.back().ids = {1, 2};
        pinned.checkpoints.back().pinned = true;
        pinned.checkpoints.back().gdn.resize(64, 1);
        cache.put(std::move(pinned));
        cache.put(image({9, 8, 7}));
        check(cache.evict_oldest() && cache.size() == 1, "the oldest unpinned conversation went (the pinned one is older)");
        check(!cache.evict_oldest() && cache.size() == 1, "only a pinned one is left: nothing more can go");
        // #879: after non-finite logits every parked conversation goes, the pinned one too
        cache.put(image({4, 5, 6}));
        std::vector<ConversationKv> kv(1);
        kv[0].cells = 3;
        kv[0].k.resize(64, 1);
        cache.retain(std::move(kv), 3);
        check(cache.retained_bytes() > 0, "a retained K/V before clear()");
        check(cache.clear() == 2, "clear() reports how many were parked");
        check(cache.size() == 0 && cache.bytes() == 0 && cache.retained_bytes() == 0, "clear() leaves nothing parked");
        check(cache.best(std::vector<int64_t>{1, 2, 3, 4}, {}, true, test_scope).tokens == 0, "a cleared cache matches nothing");
        check(cache.clear() == 0, "clear() on an empty cache");
    }
    {
        auto s = image({1, 2, 3});
        ConversationCheckpoint cp;
        cp.ids = {1, 2};
        s.checkpoints.push_back(cp);
        ConversationCache cache(4096, 4);
        cache.put(std::move(s));
        auto match = cache.best(std::vector<int64_t>{1, 2, 9, 4}, {}, true, test_scope);
        check(match.tokens == 2 && !match.live, "edited suffix falls back to parked checkpoint");
        check(cache.best(a, {}, true, test_scope).tokens == 3, "live prefix beats shorter checkpoint");
        check(cache.best(a, {}, false, test_scope).tokens == 0, "steering mode is isolated");
        check(cache.best(std::vector<int64_t>{1, 2}, {}, true, test_scope).tokens == 0, "equal-length checkpoint cannot consume last token");
        check(cache.best(std::vector<int64_t>{1}, {}, true, test_scope).tokens == 0, "short prompt cannot match");
        check(cache.best(std::vector<int64_t>{}, {}, true, test_scope).tokens == 0, "empty prompt cannot match");
    }
    {
        auto s = image({1, 2, 3});
        s.live.imgs = {{1, 123}};
        ConversationCache cache(1024, 3);
        cache.put(std::move(s));
        check(cache.best(a, {{1,123}}, true, test_scope).tokens == 3, "same image can resume");
        check(cache.best(a, {{1,124}}, true, test_scope).tokens == 0, "different image pixels invalidate same pad tokens");
        check(cache.best(a, {}, true, test_scope).tokens == 0, "missing image invalidates prefix");
        check(cache.best(a, {{1,123},{2,45}}, true, test_scope).tokens == 0, "additional image in cached prefix invalidates");
        check(cache.best(a, {{1,123},{3,45}}, true, test_scope).tokens == 3, "image after cached prefix does not invalidate");
    }
    {
        const size_t one = image({1,2,3}).bytes();
        ConversationCache cache(one*2, 8);
        cache.put(image({1,2,3})); cache.put(image({9,8,7}));
        check(cache.bytes() == one*2, "budget holds two exact-sized images");
        bool retained_in_cache = true;
        auto held = cache.take(cache.best(a, {}, true, test_scope).index, retained_in_cache);
        check(!retained_in_cache && cache.reserve_snapshot(one, held.bytes()), "count in-flight image when reserving outgoing snapshot");
        check(cache.size() == 0, "in-flight reservation evicts otherwise fitting B");
        check(cache.put(image({5,6,7}), held.bytes()), "insert with in-flight accounting");
        check(cache.bytes()+held.bytes() <= one*2, "exchange obeys byte budget");
        check(!cache.reserve_snapshot(one+1, one), "oversized exchange rejected");
        check(cache.size() == 1, "oversized snapshot does not evict useful entries");
        auto huge = image({1}); huge.live.gdn.resize(one*3);
        check(!cache.put(std::move(huge)), "oversized image rejected");
        check(cache.size() == 1, "oversized put leaves cache unchanged");
        check(!cache.reserve_snapshot(0, one*2+1), "held larger than budget cannot underflow");
    }
    for (bool pinned : {false, true}) {
        ConversationCache cache(1 << 20, 2);
        auto donor = image({1, 2, 3, 4}), other = image({9, 8, 7});
        if (pinned) {
            donor.checkpoints = {donor.live}; donor.checkpoints.back().pinned = true;
        }
        other.checkpoints = {other.live}; other.checkpoints.back().pinned = true;
        check(cache.put(std::move(donor)) && cache.put(std::move(other)), "park a donor beside a pinned history");
        bool retained_in_cache = true;
        auto held = cache.take(cache.best(std::vector<int32_t>{1, 2, 3, 4, 5}, {}, true, test_scope).index, retained_in_cache);
        check(!retained_in_cache && !cache.put(image({6, 5, 4}), held.bytes(), 1),
              "outgoing capture cannot spend a non-consuming donor's reserved entry");
        check(cache.best(std::vector<int32_t>{9, 8, 7, 6}, {}, true, test_scope).tokens == 3,
              "refused outgoing capture preserves the existing pinned history");
        check(cache.put(std::move(held)) &&
              cache.best(std::vector<int32_t>{1, 2, 3, 4, 5}, {}, true, test_scope).tokens == 4,
              "one-shot donor returns with its complete original prefix");
        check(cache.evictions() == 0 && cache.size() == 2, "borrow and return do not evict either history");
    }
    {
        ConversationCache cache(1 << 20, 2);
        check(cache.put(image({1, 2, 3})) && cache.put(image({9, 8, 7})), "fill ordinary donor table");
        bool retained_in_cache = true;
        auto held = cache.take(0, retained_in_cache);
        check(!retained_in_cache && cache.put(image({6, 5, 4}), held.bytes(), 1),
              "reserved-entry capture may evict an unrelated unpinned history");
        check(cache.put(std::move(held)) && cache.size() == 2 && cache.evictions() == 1 &&
              cache.best(std::vector<int32_t>{1, 2, 3, 4}, {}, true, test_scope).tokens == 3 &&
              cache.best(std::vector<int32_t>{6, 5, 4, 3}, {}, true, test_scope).tokens == 3,
              "capture plus donor return obey the entry limit without consuming the donor");
        check(!cache.put(image({7, 6, 5}), 0, std::numeric_limits<size_t>::max()),
              "invalid held-entry reservations cannot underflow the publication limit");
    }
    {
        ConversationCache cache(1 << 20, 4);
        auto first = image({1, 2, 3, 8});
        auto root = first.live;
        root.ids = {1, 2};
        first.checkpoints = {root};
        auto second = image({1, 2, 4, 9});
        second.checkpoints = {root};
        check(cache.put(std::move(first)) && cache.put(std::move(second)), "park no-ID root siblings");
        check(cache.best(std::vector<int32_t>{1, 2, 3, 8, 10}, {}, true, test_scope).tokens == 4 &&
              cache.best(std::vector<int32_t>{1, 2, 4, 9, 10}, {}, true, test_scope).tokens == 4,
              "unrelated histories sharing their sole root retain their own complete prefixes");
    }
    {
        ConversationCache cache(1 << 20, 1);
        auto first = image({1, 2, 3, 8});
        auto pin = first.live;
        pin.ids = {1, 2, 3};
        pin.pinned = true;
        first.checkpoints = {pin};
        check(cache.put(std::move(first)), "park explicitly pinned history");
        auto second = image({1, 2, 3, 8, 9});
        pin.pinned = false;
        second.checkpoints = {pin};
        check(!cache.put(std::move(second)), "a matching but unpinned cut cannot displace the only pinned history");
        bool preserved = false;
        cache.for_each_image([&](size_t, const SavedConversation& saved) { preserved = saved.pinned(); });
        check(preserved && cache.best(std::vector<int32_t>{1, 2, 3, 7}, {}, true, test_scope).tokens == 3,
              "refused replacement leaves the explicitly pinned prefix available");
    }
    {
        ConversationCache cache(1 << 20, 2);
        check(cache.put(image({9, 8, 7})) && cache.put(image({1, 2, 3})), "fill the published entry table");
        auto next = image({1, 2, 3, 4});
        check(cache.reserve_snapshot(next.bytes()), "admit capture bytes at a full entry limit");
        check(cache.best(std::vector<int32_t>{9, 8, 7, 6}, {}, true, test_scope).tokens == 3 &&
              cache.best(std::vector<int32_t>{1, 2, 3, 5}, {}, true, test_scope).tokens == 3,
              "a failed or abandoned capture has not spent a published history entry");
        check(cache.put(std::move(next)), "publish a checkpoint-free prefix extension");
        check(cache.best(std::vector<int32_t>{9, 8, 7, 6}, {}, true, test_scope).tokens == 3 &&
              cache.best(std::vector<int32_t>{1, 2, 3, 4, 5}, {}, true, test_scope).tokens == 4,
              "replacement retains the unrelated waiting history and advances its own prefix");
        check(cache.best(std::vector<int32_t>{1, 2, 3, 99}, {}, true, test_scope).tokens == 0,
              "an old unpinned fork cold-reads rather than fabricating a recurrent checkpoint");
        check(cache.put(image({1, 2, 3, 4})), "publish an equal live snapshot");
        check(cache.best(std::vector<int32_t>{9, 8, 7, 6}, {}, true, test_scope).tokens == 3,
              "equal live snapshots do not consume the unrelated history's entry");
    }
    {
        ConversationCache cache(1 << 20, 1);
        auto first = image({1, 2, 3, 8});
        auto pin = first.live;
        pin.ids = {1, 2, 3}; pin.pinned = true;
        first.checkpoints = {pin};
        check(cache.put(std::move(first)), "fill a pinned entry table");
        auto next = image({1, 2, 3, 8, 9});
        next.checkpoints = {pin};
        check(cache.put(std::move(next)), "a full pinned table permits a pin-preserving lineage replacement");
        bool kept_pin = false;
        cache.for_each_image([&](size_t, const SavedConversation& saved) { kept_pin = saved.pinned(); });
        check(kept_pin && cache.best(std::vector<int32_t>{1, 2, 3, 99}, {}, true, test_scope).tokens == 3 &&
              cache.best(std::vector<int32_t>{1, 2, 3, 8, 9, 99}, {}, true, test_scope).tokens == 5,
              "replacement retains the real pinned cut and the latest complete history");
    }
    for (int change = 0; change < 10; ++change) {
        ConversationCache cache(1 << 20, 4);
        auto old = image({1, 2, 3, 4}), next = image({1, 2, 3, 4, 5});
        std::vector<ConversationImageKey> query_images;
        switch (change) {
        case 0: next.scope = ConversationScope{{2}}; break;
        case 1: next.cvec = false; break;
        case 2:
            old.live.imgs = {{1, 42}}; next.live.imgs = {{1, 43}};
            query_images = old.live.imgs; break;
        case 3: next.live.imgs = {{1, 42}}; break;
        case 4: next.geometry[0] = 1; break;
        case 5: next.layer_hi = 1; break;
        case 6:
            old.kv.resize(1); next.kv.resize(1);
            old.kv[0].format = 1; next.kv[0].format = 2; break;
        case 7: next.stage_images.push_back(image({1, 2, 3, 4, 5})); break;
        case 8:
            old.stage_images = {image({1, 2, 3, 4}), image({1, 2, 3, 4})};
            next.stage_images = {image({1, 2, 3, 4, 5}), image({1, 2, 3, 4, 5})};
            old.stage_images[0].layer_hi = next.stage_images[1].layer_hi = 1;
            old.stage_images[1].layer_hi = next.stage_images[0].layer_hi = 2; break;
        case 9: next.live.ids.back() = 99; next.live.ids[3] = 8; break;
        }
        check(cache.put(std::move(old)) && cache.put(std::move(next)), "park non-equivalent replacement candidates");
        check(cache.best(std::vector<int32_t>{1, 2, 3, 4, 99}, query_images, true, test_scope).tokens == 4,
              "scope, mode, image, shape, stage and branch mismatches preserve the earlier history");
    }
    {
        ConversationCache cache(1 << 20, 1);
        auto old = image({1, 2, 3, 4}), next = image({1, 2, 3, 4, 5});
        next.live.imgs = {{4, 42}}; // an image after the old cut does not change that prefix
        check(cache.put(std::move(old)) && cache.put(std::move(next)), "later images permit a complete prefix extension");
        check(cache.best(std::vector<int32_t>{1, 2, 3, 4, 5, 99}, {{4, 42}}, true, test_scope).tokens == 5,
              "the newly extended image history restores its actual image identity");
    }
    for (int missing = 0; missing < 4; ++missing) {
        ConversationCache cache(1 << 20, 1);
        auto old = image({1, 2, 3, 4}), next = image({1, 2, 3, 4, 5});
        auto pin = old.live; pin.ids = {1, 2, 3}; pin.pinned = true;
        old.checkpoints = next.checkpoints = {pin};
        old.stage_images.push_back(image({1, 2, 3, 4}));
        next.stage_images.push_back(image({1, 2, 3, 4, 5}));
        old.stage_images[0].scope = next.stage_images[0].scope = {};
        old.stage_images[0].checkpoints = next.stage_images[0].checkpoints = {pin};
        if (missing == 1) next.checkpoints.clear(); // live prefix alone is not a retained pin
        if (missing == 2) next.stage_images[0].checkpoints.clear();
        if (missing == 3) next.stage_images[0].checkpoints[0].pinned = false;
        check(cache.put(std::move(old)), "park a pinned layered history");
        check(cache.put(std::move(next)) == (missing == 0), "every stage must preserve the explicit pinned cut");
        check(cache.best(std::vector<int32_t>{1, 2, 3, 99}, {}, true, test_scope).tokens == 3,
              "layered replacement or refusal leaves a real pinned prefix available");
    }
    {
        // A genuinely appended history replaces its older live-prefix snapshots,
        // leaving the unrelated parent reusable even after many child turns.
        auto cp = [](std::vector<int32_t> ids) { ConversationCheckpoint c; c.ids = std::move(ids); return c; };
        auto with = [](std::vector<int32_t> prefix, std::initializer_list<int32_t> more) {
            prefix.insert(prefix.end(), more); return prefix;
        };
        const std::vector<int32_t> root = {1, 2, 3, 4};
        SavedConversation main = image({});
        main.live.ids = with(with(root, {10, 11, 12}), {13, 14});
        main.checkpoints = {cp(root), cp(with(root, {10, 11, 12}))};
        const size_t big = 1 << 20;
        ConversationCache cache(big, 4);
        check(cache.put(std::move(main)), "park MAIN");
        std::vector<int32_t> history = with(root, {20});
        std::vector<ConversationCheckpoint> chain = {cp(root)};
        for (int turn = 1; turn <= 9; ++turn) {
            chain.push_back(cp(history));                        // the turn boundary the next request resumes from
            SavedConversation sub = image({});
            sub.live.ids = with(history, {900, (int32_t) turn});
            sub.checkpoints = chain;
            check(cache.put(std::move(sub)), "park SUB turn");
            history = with(history, {900, (int32_t) turn});
        }
        const auto m = cache.best(with(with(root, {10, 11, 12}), {13, 14, 15}), {}, true, test_scope);
        check(m.tokens == 9 && m.live, "MAIN still restores in full");
        const auto s = cache.best(with(history, {40}), {}, true, test_scope);
        check(s.tokens == (int64_t) history.size() && s.live, "SUB resumes its complete appended history");
    }
    {
        // Shared roots and shorter rewind cuts do not replace complete divergent histories.
        auto cp = [](std::vector<int32_t> ids) { ConversationCheckpoint c; c.ids = std::move(ids); return c; };
        ConversationCache cache(1 << 20, 8);
        SavedConversation other = image({1, 2, 3, 4, 50, 51, 52});
        other.checkpoints = {cp({1, 2, 3, 4}), cp({1, 2, 3, 4, 50, 51})};
        cache.put(std::move(other));
        cache.put(image({1, 2, 3, 4, 7, 7}));                    // no checkpoints
        SavedConversation steered = image({1, 2, 3, 4, 60, 61}, false);
        steered.checkpoints = {cp({1, 2, 3, 4, 60})};
        cache.put(std::move(steered));
        SavedConversation branch = image({1, 2, 3, 4, 60, 70, 71});
        branch.checkpoints = {cp({1, 2, 3, 4}), cp({1, 2, 3, 4, 60, 70})};
        cache.put(std::move(branch));
        SavedConversation incoming = image({1, 2, 3, 4, 60, 80, 81});
        incoming.checkpoints = {cp({1, 2, 3, 4}), cp({1, 2, 3, 4, 60})};
        check(cache.put(std::move(incoming)), "park a conversation sharing roots with all of them");
        check(cache.size() == 5 && cache.superseded() == 0, "none of them is superseded");
        SavedConversation same_state = image({1, 2, 3, 4, 60, 70});   // live equal to the branch's deepest point
        same_state.checkpoints = {cp({1, 2, 3, 4})};
        check(cache.put(std::move(same_state)), "park a shorter rewind branch");
        check(cache.best(std::vector<int32_t>{1, 2, 3, 4, 60, 70, 71, 99}, {}, true, test_scope).tokens == 7,
              "a rewind matching only an older checkpoint preserves the longer branch");
        SavedConversation huge = image({1, 2, 3, 4, 60, 80});
        huge.checkpoints = {cp({1, 2, 3, 4}), cp({1, 2, 3, 4, 60})};
        huge.live.gdn.resize(2 << 20);
        check(!cache.put(std::move(huge)) &&
              cache.best(std::vector<int32_t>{1, 2, 3, 4, 60, 80, 81, 99}, {}, true, test_scope).tokens == 7,
              "oversized replacement preserves the useful complete branch");
    }
    {
        // --conversation-cache-park-gain N: parked_prefix_tokens reports what a park now would supersede, so a
        // conversation that advanced fewer than N tokens since that copy keeps it instead of writing the whole
        // snapshot again. Planning uses the same scoped live-extension rule as transactional put().
        auto cp = [](std::vector<int32_t> ids) { ConversationCheckpoint c; c.ids = std::move(ids); return c; };
        ConversationCache cache(1 << 20, 8);
        check(cache.parked_prefix_tokens({1, 2, 3}, {}, {}, true, test_scope) == 0, "an empty cache supersedes nothing");

        const std::vector<int32_t> live_ids = {1, 2, 3, 4, 60, 70};
        const std::vector<ConversationCheckpoint> chain = {cp({1, 2, 3, 4}), cp({1, 2, 3, 4, 60})};
        SavedConversation same = image({1, 2, 3, 4, 60, 70});
        same.checkpoints = chain;
        check(cache.put(std::move(same)), "park a conversation with a checkpoint chain");
        check(cache.parked_prefix_tokens(live_ids, {}, chain, true, test_scope) == 6,
              "park gain reports the complete parked live prefix transactional put would replace");

        const std::vector<ConversationCheckpoint> other_chain = {cp({1, 2, 3})};
        check(cache.parked_prefix_tokens({1, 2, 3, 40, 50}, {}, other_chain, true, test_scope) == 0,
              "a conversation sharing only the root supersedes nothing");
        check(cache.parked_prefix_tokens(live_ids, {}, chain, false, test_scope) == 0,
              "the other steering mode is a different entry");
        check(cache.parked_prefix_tokens(live_ids, {}, chain, true, {}) == 0 &&
              cache.parked_prefix_tokens(live_ids, {}, chain, true, ConversationScope{{2}}) == 0,
              "absent and foreign scopes cannot authorize park gain for identical tokens");
        check(cache.parked_prefix_tokens({1, 2, 3, 4, 60, 70, 71}, {}, chain, true, test_scope) == 6,
              "an appended tail keeps the gain boundary at the complete parked live state");
        SavedConversation no_cp = image({1, 2, 3, 9});
        check(cache.put(std::move(no_cp)) &&
              cache.parked_prefix_tokens({1, 2, 3, 9}, {}, {}, true, test_scope) == 4,
              "checkpoint-free live extensions use the same automatic replacement rule");
        ConversationCache pinned(1 << 20, 2);
        auto document = image({1, 2, 3, 4});
        auto pin = cp({1, 2}); pin.pinned = true; pin.plan.chunk = 256;
        document.checkpoints = {pin};
        check(pinned.put(std::move(document)), "park an explicitly pinned document");
        check(pinned.parked_prefix_tokens({1, 2, 3, 4, 5}, {}, {}, true, test_scope) == 0 &&
              pinned.parked_prefix_tokens({1, 2, 3, 4, 5}, {}, {pin}, true, test_scope) == 4,
              "park gain requires the outgoing chain to preserve every explicit pin");
        pin.plan.chunk = 512;
        check(pinned.parked_prefix_tokens({1, 2, 3, 4, 5}, {}, {pin}, true, test_scope) == 0,
              "a changed prompt continuation plan does not preserve the parked pin");
    }
    {
        // borrowing: a NEW conversation that starts with a checkpoint inside a parked one (a sibling subagent with
        // the same system prompt and tools) restores from it and leaves it parked; the same conversation going on
        // (its live state, or its deepest checkpoint after the client re-rendered the last reply) is still taken
        auto cp = [](std::vector<int32_t> ids) { ConversationCheckpoint c; c.ids = std::move(ids); return c; };
        SavedConversation a = image({});
        a.live.ids = {1, 2, 3, 4, 10, 11, 12, 13};
        a.checkpoints = {cp({1, 2, 3, 4}), cp({1, 2, 3, 4, 10, 11})};
        ConversationCache cache(1 << 20, 2);
        check(cache.put(std::move(a)), "park subagent A");
        const auto sib = cache.best(std::vector<int64_t>{1, 2, 3, 4, 20, 21}, {}, true, test_scope);
        check(sib.tokens == 4 && !sib.live && sib.serial != 0 && cache.borrows(sib), "a sibling on A's root borrows");
        const auto again = cache.best(std::vector<int64_t>{1, 2, 3, 4, 10, 11, 30}, {}, true, test_scope);
        check(again.tokens == 6 && !again.live && !cache.borrows(again),
              "A re-rendered at its deepest checkpoint is taken");
        const auto cont = cache.best(std::vector<int64_t>{1, 2, 3, 4, 10, 11, 12, 13, 14}, {}, true, test_scope);
        check(cont.live && !cache.borrows(cont), "A going on is taken");
        check(!cache.borrows(cache.best(std::vector<int64_t>{5, 6, 7}, {}, true, test_scope)), "no match, nothing to borrow");
        // while the outgoing conversations are parked, the guarded donor is neither evicted nor dropped
        cache.guard(sib.serial);
        check(cache.put(image({7, 7, 7})) && cache.put(image({8, 8, 8})), "two outgoing conversations park");
        check(cache.find(sib.serial) != nullptr && cache.size() == 2 && cache.evictions() == 1 && !cache.guard_blocked(),
              "the slot limit evicts the other entry, not the guarded donor");
        check(cache.best(std::vector<int64_t>{7, 7, 7, 1}, {}, true, test_scope).tokens == 0, "the other one went");
        SavedConversation rewritten = image({1, 2, 3, 4, 10, 11, 12, 13, 40}); // a complete live extension
        rewritten.checkpoints = {cp({1, 2, 3, 4}), cp({1, 2, 3, 4, 10, 11})};
        check(cache.put(std::move(rewritten)) && cache.find(sib.serial) != nullptr && cache.superseded() == 0,
              "a guarded donor is not dropped as superseded");
        cache.unguard();
        // identity survives index shifts: take an older entry, the donor's serial still finds it
        const auto donor_index = cache.index_of(sib.serial);
        check(donor_index != SIZE_MAX, "donor indexed by its serial");
        check(cache.best(std::vector<int64_t>{1, 2, 3, 4, 10, 11, 12, 13, 14}, {}, true, test_scope).serial == sib.serial,
              "matches report the entry's serial");
        auto refused = image({1, 2, 3, 4, 10, 11, 12, 13, 41});
        refused.live.gdn.resize(2 << 20);
        const auto before = cache.envelope_bytes();
        check(!cache.put(std::move(refused)) && cache.find(sib.serial) != nullptr &&
              cache.envelope_bytes() == before && cache.superseded() == 0,
              "a failed replacement leaves the exact fallback identity and envelope unchanged");
        auto accepted = image({1, 2, 3, 4, 10, 11, 12, 13, 42});
        check(cache.put(std::move(accepted)) && cache.find(sib.serial) == nullptr && cache.superseded() == 1,
              "only a committed complete live extension retires the fallback");
    }
    {
        // the outgoing conversation fits only in the guarded donor's room: make_room refuses before evicting anything
        auto cp = [](std::vector<int32_t> ids) { ConversationCheckpoint c; c.ids = std::move(ids); return c; };
        SavedConversation a = image({1, 2, 3, 4, 10, 11});
        a.checkpoints = {cp({1, 2, 3, 4}), cp({1, 2, 3, 4, 10})};
        const size_t a_bytes = a.bytes();
        const size_t out_bytes = image({9, 9, 9}).bytes();
        ConversationCache tight(a_bytes + out_bytes - 1, 4);
        check(tight.put(std::move(a)), "park the donor");
        const auto m = tight.best(std::vector<int64_t>{1, 2, 3, 4, 50}, {}, true, test_scope);
        check(tight.borrows(m), "a sibling would borrow");
        tight.guard(m.serial);
        check(!tight.put(image({9, 9, 9})) && tight.guard_blocked(), "no room beside the guarded donor: refused");
        check(tight.size() == 1 && tight.evictions() == 0, "nothing evicted by the refusal");
        tight.unguard();
        bool retained = true;
        auto taken = tight.take(tight.index_of(m.serial), retained);
        check(!retained, "taking an ordinary donor transfers its sole owner");
        check(taken.live.ids.size() == 6 && taken.checkpoints.size() == 2 && tight.size() == 0,
              "the taken donor is whole");
        ConversationCache one_slot(1 << 20, 1);
        SavedConversation b = image({1, 2, 3, 4, 10, 11});
        b.checkpoints = {cp({1, 2, 3, 4}), cp({1, 2, 3, 4, 10})};
        check(one_slot.put(std::move(b)), "park in a single slot");
        const auto mb = one_slot.best(std::vector<int64_t>{1, 2, 3, 4, 50}, {}, true, test_scope);
        one_slot.guard(mb.serial);
        check(!one_slot.put(image({9, 9, 9})) && one_slot.guard_blocked() && one_slot.size() == 1,
              "one slot: the guarded donor blocks, nothing evicted");
        one_slot.unguard();
    }
    {
        // pin=N beside a borrow: the guarded donor and a conversation holding a pinned shared prefix both stay; the
        // oldest other entry is the one evicted (the case midhatn tested on #1164)
        auto cp = [](std::vector<int32_t> ids, bool pinned = false) {
            ConversationCheckpoint c; c.ids = std::move(ids); c.pinned = pinned; return c;
        };
        SavedConversation doc = image({1, 2, 3, 4, 5, 6, 30, 31});
        doc.checkpoints = {cp({1, 2, 3, 4, 5, 6}, true), cp({1, 2, 3, 4, 5, 6, 30})};
        SavedConversation donor = image({1, 2, 10, 11, 12});
        donor.checkpoints = {cp({1, 2}), cp({1, 2, 10, 11})};
        ConversationCache cache(1 << 20, 3);
        check(cache.put(std::move(doc)) && cache.put(std::move(donor)) && cache.put(image({7, 7, 7})),
              "pinned document, donor and an ordinary conversation parked");
        const auto m = cache.best(std::vector<int64_t>{1, 2, 40, 41}, {}, true, test_scope);
        check(cache.borrows(m), "a sibling on the donor's root borrows");
        cache.guard(m.serial);
        check(cache.put(image({8, 8, 8})) && !cache.guard_blocked() && cache.size() == 3 && cache.evictions() == 1,
              "the outgoing conversation parks");
        check(cache.find(m.serial) != nullptr &&
              cache.best(std::vector<int64_t>{1, 2, 3, 4, 5, 6, 9}, {}, true, test_scope).tokens == 6 &&
              cache.best(std::vector<int64_t>{7, 7, 7, 1}, {}, true, test_scope).tokens == 0,
              "only the ordinary entry was evicted: the donor and the pinned document stay");
        check(cache.put(image({9, 9, 9})) && cache.size() == 3 && cache.evictions() == 2 &&
              cache.find(m.serial) != nullptr && cache.best(std::vector<int64_t>{8, 8, 8, 1}, {}, true, test_scope).tokens == 0,
              "the next outgoing one evicts the previous outgoing one, still not the donor or the document");
        cache.unguard();
        ConversationCache two(1 << 20, 2);
        SavedConversation doc2 = image({1, 2, 3, 4, 5, 6, 30});
        doc2.checkpoints = {cp({1, 2, 3, 4, 5, 6}, true)};
        SavedConversation donor2 = image({1, 2, 10, 11, 12});
        donor2.checkpoints = {cp({1, 2}), cp({1, 2, 10, 11})};
        check(two.put(std::move(doc2)) && two.put(std::move(donor2)), "pinned document and donor fill two slots");
        const auto m2 = two.best(std::vector<int64_t>{1, 2, 40}, {}, true, test_scope);
        two.guard(m2.serial);
        check(!two.put(image({9, 9, 9})) && two.guard_blocked() && two.size() == 2 && two.evictions() == 0,
              "no slot beside the donor and the pinned document: refused before evicting anything");
        two.unguard();
        // a match that reaches the pinned prefix is taken, not borrowed: one conversation keeps holding the pin
        const auto on_pin = two.best(std::vector<int64_t>{1, 2, 3, 4, 5, 6, 77}, {}, true, test_scope);
        check(on_pin.tokens == 6 && !on_pin.live && !two.borrows(on_pin), "a pinned prefix is taken, not borrowed");
        ConversationCache three(1 << 20, 3);
        SavedConversation doc3 = image({1, 2, 3, 4, 5, 6, 30, 31, 32});
        doc3.checkpoints = {cp({1, 2, 3, 4, 5, 6}, true), cp({1, 2, 3, 4, 5, 6, 30}), cp({1, 2, 3, 4, 5, 6, 30, 31})};
        check(three.put(std::move(doc3)), "a conversation past its pinned prefix is parked");
        const auto past_pin = three.best(std::vector<int64_t>{1, 2, 3, 4, 5, 6, 30, 99}, {}, true, test_scope);
        check(past_pin.tokens == 7 && !three.borrows(past_pin),
              "a match past the pinned prefix (the copy would carry the pin) is taken, not borrowed");
    }
    {
        ConversationCache disabled(0,4), no_slots(1024,0);
        check(!disabled.enabled() && !no_slots.enabled(), "both disable switches");
        check(!disabled.put(image({1,2,3})) && !no_slots.put(image({1,2,3})), "disabled cache stores nothing");
        check(disabled.best(a, {}, true, test_scope).tokens == 0, "disabled cache has no matches");
    }
    {
        // --conversation-cache-min-tokens: a conversation shorter than the minimum is not worth one of the slots.
        // Parking is a switch-time decision, so without this a client's short side requests park themselves and,
        // once the cache is full, evict the long conversation they were interleaved with (oldest first).
        ConversationCache any(1024, 4, 0), floor(1024, 4, 12288);
        check(any.enabled() && floor.enabled(), "the minimum does not disable the cache");
        check(any.wants(0) && any.wants(1) && any.wants(1 << 20), "0 parks every length, as before the flag");
        check(!floor.wants(0) && !floor.wants(1) && !floor.wants(12287), "below the minimum is not parked");
        check(floor.wants(12288) && floor.wants(12289), "the minimum itself and above are parked");
        // the CLI rejects a negative, but the predicate must not invert for one that reaches it another way
        ConversationCache negative(1024, 4, -1);
        check(negative.wants(0) && negative.wants(1), "a negative minimum parks everything, like 0");
        // wants() answers for the length alone; the gate is enabled() && wants(), so a disabled cache parks
        // nothing whatever the length - keep the two independent rather than folding one into the other
        ConversationCache disabled_min(0, 4, 12288);
        check(!disabled_min.enabled() && disabled_min.wants(1 << 20), "a disabled cache still answers for length");
    }
    {
        // layer-split parking: checkpoints moved apart into stage parts and put back, no running state copied
        auto running = [](std::initializer_list<int32_t> ids, uint8_t salt, size_t parts) {
            ConversationCheckpoint c;
            c.ids = ids;
            c.gdn.assign(256, salt);
            c.tails.assign(32, uint8_t(salt + 1));
            c.used = salt;
            for (size_t k = 0; k < parts; ++k) {
                ConversationCheckpoint p;
                p.gdn.assign(128, uint8_t(salt + 10 + k));
                c.stage_parts.push_back(std::move(p));
            }
            return c;
        };
        std::vector<ConversationCheckpoint> checks;
        checks.push_back(running({1, 2}, 3, 2));
        checks.push_back(running({1, 2, 3}, 5, 1));       // an incomplete set: kept aside, put back in place
        checks.push_back(running({1, 2, 3, 4}, 7, 2));
        const auto original = checks;
        const uint8_t* gdn0 = checks[0].gdn.data();
        const uint8_t* part1 = checks[2].stage_parts[1].gdn.data();
        auto split = conversation_checkpoints_split(std::move(checks), 2, ConversationCheckpointRetention::all);
        check(checks.empty() && split.stage0.size() == 2 && split.rest.size() == 1 && split.parts.size() == 2 &&
              split.parts[0].size() == 2 && split.parts[1].size() == 2, "split: complete checkpoints apart, the other aside");
        check(split.stage0[0].stage_parts.empty() && split.parts[1][1].ids == original[2].ids &&
              split.parts[1][1].used == original[2].used && split.parts[0][0].gdn == original[0].stage_parts[0].gdn,
              "split: every stage part carries its checkpoint's ids and LRU stamp");
        check(split.stage0[0].gdn.data() == gdn0 && split.parts[1][1].gdn.data() == part1,
              "split moves the running state (same buffers, no copy)");
        std::vector<ConversationCheckpoint> back;
        check(conversation_checkpoints_merge(std::move(split), back) && back.size() == 3, "merge puts them back");
        bool same = true;
        for (size_t j = 0; j < back.size(); ++j) {
            same = same && back[j].ids == original[j].ids && back[j].gdn == original[j].gdn &&
                   back[j].tails == original[j].tails && back[j].used == original[j].used &&
                   back[j].stage_parts.size() == original[j].stage_parts.size();
            for (size_t k = 0; same && k < back[j].stage_parts.size(); ++k)
                same = back[j].stage_parts[k].gdn == original[j].stage_parts[k].gdn;
        }
        check(same, "merge(split(x)) == x, in the original order");
        check(back[0].gdn.data() == gdn0 && back[2].stage_parts[1].gdn.data() == part1, "merge moves them back too");

        auto torn = conversation_checkpoints_split(std::vector<ConversationCheckpoint>(original), 2, ConversationCheckpointRetention::all);
        torn.parts[1].pop_back();
        check(!conversation_checkpoints_merge(std::move(torn), back) && back.empty(), "a stage short of a part is refused");
        auto mixed = conversation_checkpoints_split(std::vector<ConversationCheckpoint>(original), 2, ConversationCheckpointRetention::all);
        mixed.parts[0][0].ids = {9, 9};
        check(!conversation_checkpoints_merge(std::move(mixed), back) && back.empty(), "a part of another checkpoint is refused");
        // a restored image: stage 0's checkpoints and each stage's, with no record of an incomplete one
        ConversationCheckpointSplit restored;
        restored.stage0.push_back(running({4, 5}, 11, 0));
        restored.parts.resize(2);
        for (auto& p : restored.parts) { auto c = running({4, 5}, 12, 0); p.push_back(std::move(c)); }
        check(conversation_checkpoints_merge(std::move(restored), back) && back.size() == 1 &&
              back[0].stage_parts.size() == 2, "a restored image's parts are all whole");
        auto none = conversation_checkpoints_split({}, 3, ConversationCheckpointRetention::all);
        check(conversation_checkpoints_merge(std::move(none), back) && back.empty(), "no checkpoints: nothing to do");
    }
    for (int pins = 0; pins <= 2; ++pins) {
        std::vector<ConversationCheckpoint> working(6);
        for (size_t i = 0; i < working.size(); ++i) {
            auto& cut = working[i];
            cut.ids.resize(i + 2);
            for (size_t j = 0; j < cut.ids.size(); ++j) cut.ids[j] = (int32_t)j + 1;
            cut.gdn = {(uint8_t)i}; cut.used = i;
            cut.pinned = (pins >= 1 && i == 1) || (pins == 2 && i == 3);
            cut.stage_parts.resize(2);
            for (auto& part : cut.stage_parts) {
                part.ids = cut.ids; part.used = cut.used; part.pinned = cut.pinned;
                part.gdn = {(uint8_t)(i + 10)};
            }
        }
        const auto original = working;
        auto selected = conversation_checkpoints_split(std::move(working), 2,
                                                       ConversationCheckpointRetention::parked_history);
        size_t at = 0;
        for (size_t i = 0; i < original.size(); ++i) {
            if (i != 0 && i + 1 != original.size() && !original[i].pinned) continue;
            check(at < selected.stage0.size() && selected.stage0[at].ids == original[i].ids &&
                  selected.stage0[at].gdn == original[i].gdn && selected.stage0[at].pinned == original[i].pinned,
                  "parked history contains its actual root, latest cut and every explicit pin");
            for (size_t stage = 0; stage < 2; ++stage)
                check(selected.parts[stage][at].ids == original[i].ids &&
                      selected.parts[stage][at].pinned == original[i].pinned &&
                      selected.parts[stage][at].gdn == original[i].stage_parts[stage].gdn,
                      "every retained cut keeps matching stage state and pin metadata");
            ++at;
        }
        check(at == selected.stage0.size(), "older unpinned cuts do not multiply across parked histories");
        check(conversation_checkpoints_merge(std::move(selected), working) && working.size() == original.size(),
              "capture restores the complete working checkpoint chain");
        for (size_t i = 0; i < original.size(); ++i) {
            check(working[i].ids == original[i].ids && working[i].gdn == original[i].gdn &&
                  working[i].pinned == original[i].pinned, "working rewind cuts and pins survive selective parking");
            for (size_t stage = 0; stage < 2; ++stage)
                check(working[i].stage_parts[stage].gdn == original[i].stage_parts[stage].gdn &&
                      working[i].stage_parts[stage].pinned == original[i].pinned,
                      "omitted and retained stage payloads both return to their working owners");
        }
    }
    {
        // the later stages' retained K/V, kept with the first stage's and limited together
        auto kv = [](size_t n) {
            std::vector<ConversationKv> v(2);
            for (auto& l : v) l.k.resize(n, 1);
            return v;
        };
        ConversationCache cache(1 << 20, 4);
        std::vector<std::vector<ConversationKv>> stage_kv;
        stage_kv.push_back(kv(1000));
        stage_kv.push_back(kv(2000));
        cache.retain(kv(500), 40, std::move(stage_kv));
        check(cache.retained_bytes() >= 2 * (500 + 1000 + 2000), "retained bytes count every stage");
        cache.limit_reuse(30);
        auto r = cache.take_reuse();
        check(r.kv.size() == 2 && r.stages.size() == 2 && r.unchanged_tokens == 30 &&
              r.stages[0].unchanged_tokens == 30 && r.stages[1].unchanged_tokens == 30 &&
              r.stages[1].captured_tokens == 40, "every stage's reuse is limited to the first rewrite");
        check(cache.retained_bytes() == 0, "taken once");
        std::vector<std::vector<ConversationKv>> big;
        big.push_back(kv(1 << 20));
        cache.retain(kv(500), 40, std::move(big));
        check(cache.retained_bytes() == 0, "a stage's K/V over the budget drops the whole reuse");
        std::vector<std::vector<ConversationKv>> one;
        one.push_back(kv(100));
        cache.retain(kv(500), 40, std::move(one));
        cache.limit_reuse(0);
        check(cache.retained_bytes() == 0, "namespace/steering changes and resident-owner mounts discard every stage's reuse");
        cache.retain(kv(500), 40);
        r = cache.take_reuse();
        check(r.kv.size() == 2 && r.stages.empty(), "no layer split: no stage reuse, as before");
    }
    {   // pin=N: a parked conversation that holds a pinned shared prefix is not the one evicted
        const size_t one = image({1,2,3}).bytes();
        auto pinned = image({1,2,3});
        pinned.checkpoints.push_back(ConversationCheckpoint{});
        pinned.checkpoints.back().ids = {1,2};
        pinned.checkpoints.back().pinned = true;
        pinned.checkpoints.back().gdn.resize(64, 1);
        const size_t pinned_bytes = pinned.bytes();
        ConversationCache big(pinned_bytes + one * 2 + 64, 3);
        check(pinned.pinned() && !image({1,2,3}).pinned(), "a conversation is pinned when one of its checkpoints is");
        check(big.put(std::move(pinned)), "the pinned conversation is parked first (the oldest)");
        check(big.put(image({7,8,9})) && big.put(image({4,5,6})) && big.size() == 3, "two more fill the slots");
        check(big.put(image({10,11,12})) && big.size() == 3 && big.evictions() == 1,
              "a fourth evicts the oldest unpinned one");
        check(big.best(std::vector<int32_t>{1,2,9,9}, {}, true, test_scope).tokens == 2, "the pinned prefix is still found");
        ConversationCache tiny(pinned_bytes + 8, 2);
        auto p2 = image({1,2,3});
        p2.checkpoints.push_back(ConversationCheckpoint{});
        p2.checkpoints.back().ids = {1,2};
        p2.checkpoints.back().pinned = true;
        p2.checkpoints.back().gdn.resize(64, 1);
        check(tiny.put(std::move(p2)), "a pinned one fits an empty cache");
        check(!tiny.put(image({7,8,9})) && tiny.size() == 1, "nothing else fits beside it: parking is refused, the pin stays");
    }
    {
        // #250: every parked entry holds a pin, so there is no admissible victim and the cache refuses instead of
        // choosing one.  The live byte-pressure path (reserve_snapshot), the RAM gate's step (evict_oldest) and the
        // park itself (put) all refuse atomically; both pins stay restorable, which is the documented refusal.
        auto first = image({1, 2, 3}), second = image({9, 8, 7});
        for (auto* s : {&first, &second}) {
            s->checkpoints.push_back(ConversationCheckpoint{});
            s->checkpoints.back().ids = s->live.ids;
            s->checkpoints.back().pinned = true;
            s->checkpoints.back().gdn.resize(64, 1);
        }
        const size_t both = first.bytes() + second.bytes();
        ConversationCache cache(both, 2);
        check(cache.put(SavedConversation(first)) && cache.put(SavedConversation(second)),
              "both pinned conversations fill the cache exactly");
        const size_t before_reserve = cache.envelope_bytes();
        check(!cache.reserve_snapshot(1) && cache.size() == 2 && cache.bytes() == both && cache.evictions() == 0 &&
                  cache.envelope_bytes() == before_reserve,
              "all-pinned byte pressure refuses the reservation atomically, spending no pin");
        check(!cache.evict_oldest() && cache.size() == 2 && cache.bytes() == both,
              "the RAM gate's step refuses too: nothing admissible is left");
        check(!cache.put(image({4, 5, 6})) && cache.size() == 2 && cache.bytes() == both,
              "a new conversation is not parked beside them");
        check(cache.best(std::vector<int32_t>{1, 2, 3, 4}, {}, true, test_scope).tokens == 3 &&
                  cache.best(std::vector<int32_t>{9, 8, 7, 6}, {}, true, test_scope).tokens == 3,
              "the refusal keeps both pinned histories restorable");
    }
    {
        using namespace strata::program;
        PrefixCosts costs;
        PrefixWork resident{1000, 11}, parked{10000, 22}, park{5000, 33};
        check(!costs.transfer_ms(PrefixTransfer::resident_live, resident), "unobserved transfer is unknown");
        check(costs.transfer_ms(PrefixTransfer::park, {}) == 0.0, "no outgoing capture has zero cost");
        costs.transfer(PrefixTransfer::resident_live, resident, 10);
        costs.transfer(PrefixTransfer::parked, parked, 400);
        costs.transfer(PrefixTransfer::park, park, 5);
        costs.prefill(0, 2048, false, 409.6);
        const auto tail = costs.prefill_ms(1024, 2048, false);
        check(tail && std::abs(*tail - 204.8) < 1e-9, "reuse gain comes from observed fresh-prefill work");
        costs.prefill(1536, 2048, false, 102.4); // observed suffix geometry, not an extrapolated tiny batch
        auto choose = [&](const PrefixCosts& model, bool measured, double margin = 0) {
            PrefixChoice selection(measured, margin);
            selection.consider(0, 0, model.prefill_ms(0, 2048, false).value() +
                model.transfer_ms(PrefixTransfer::park, park).value());
            selection.consider(1, 1024, model.transfer_ms(PrefixTransfer::resident_live, resident).value() +
                model.prefill_ms(1024, 2048, false).value() + 5);
            selection.consider(2, 1792, model.transfer_ms(PrefixTransfer::parked, parked).value() +
                model.prefill_ms(1792, 2048, false).value() + 5);
            return selection;
        };
        check(choose(costs, true).selected() == 1 && choose(costs, true).using_cost(),
              "cheaper shallower resident source beats a deeper parked source");
        check(choose(costs, false).selected() == 2, "longest policy is the original rollback");
        PrefixCosts fast = costs;
        for (int i = 0; i < 8; ++i) fast.transfer(PrefixTransfer::parked, parked, 1);
        check(choose(fast, true).selected() == 2, "deeper parked image wins when the measured copy is worthwhile");
        PrefixChoice overpriced(true);
        overpriced.consider(0, 0, 10);
        overpriced.consider(1, 1000, 100);
        check(overpriced.selected() == 0, "full recomputation beats an expensive restore");
        check(choose(costs, true, 300).selected() == 2, "preregistered savings margin keeps longest");
        PrefixChoice tied(true);
        tied.consider(0, 0, 200);
        tied.consider(1, 1024, 100);
        tied.consider(2, 1792, 100);
        tied.consider(3, 1792, 100);
        check(tied.selected() == 2, "cost ties preserve longest and its original stable tie order");
        PrefixChoice unknown(true);
        unknown.consider(0, 0, 100);
        unknown.consider(1, 1024, 20);
        unknown.consider(2, 1792, {});
        check(unknown.selected() == 2 && unknown.unknown(), "unpriced longest uses deterministic longest fallback");
        unknown.consider(3, 2048, std::numeric_limits<double>::quiet_NaN());
        check(unknown.selected() == 3, "nonfinite costs cannot become a spurious cheapest source");
        PrefixChoice missing_alternative(true);
        missing_alternative.consider(0, 0, {});
        missing_alternative.consider(1, 1024, 20);
        missing_alternative.consider(2, 1792, 100);
        check(missing_alternative.selected() == 1,
              "unobserved alternatives cannot veto improvement over a priced longest prefix");
        check(!costs.transfer_ms(PrefixTransfer::parked, {10000, 23}),
              "observations never cross resolved device or residency paths");
        check(!costs.transfer_ms(PrefixTransfer::checkpoint, parked),
              "capture, restore and resident bounce costs are separate observations");
        check(!costs.transfer_ms(PrefixTransfer::parked, {40001, 22}) &&
              costs.transfer_ms(PrefixTransfer::parked, {40000, 22}) == 1600.0,
              "transfer prediction stops at its declared four-times size bound");
        check(!costs.prefill_ms(0, 65536, false) && !costs.prefill_ms(1024, 2048, true),
              "unknown context scale and window geometry remain unknown");
        check(!costs.prefill_ms(2000, 2048, false), "tiny batched parts cannot inherit a long-prefill rate");
        costs.prefill(1024, 2048, true, 512, 7);
        check(costs.prefill_ms(1536, 2048, true, 7) == 256.0 &&
              !costs.prefill_ms(1536, 2048, true, 8), "request arithmetic paths calibrate separately");
        costs.transfer(PrefixTransfer::parked, parked, std::numeric_limits<double>::infinity());
        check(costs.transfer_ms(PrefixTransfer::parked, parked) == 400.0, "failed/nonfinite timing is not learned");
        PrefixWork overflow{UINT64_MAX - 2, 1};
        overflow.add(3);
        check(!costs.transfer_ms(PrefixTransfer::parked, overflow), "byte overflow is unknown, never a cheap copy");
    }
    {
        // Price every compatible parked branch without taking, copying or evicting it.
        ConversationCache cache(4096, 4);
        auto shallow = image({1,2});
        auto deep = image({1,2,3,4});
        deep.checkpoints.push_back(ConversationCheckpoint{});
        deep.checkpoints.back().ids = {1,2,3};
        check(cache.put(std::move(shallow)) && cache.put(std::move(deep)), "park cost-selection branches");
        const size_t bytes = cache.bytes();
        const size_t size_before = cache.size(), evictions_before = cache.evictions();
        int visits = 0;
        auto denied = [&](auto, const auto&, const auto&) { ++visits; };
        cache.for_each_match(std::vector<int32_t>{1,2,3,4,5}, {}, true, ConversationScope{{2}}, denied);
        cache.for_each_match(std::vector<int32_t>{1,2,3,4,5}, {}, true, {}, denied);
        cache.for_each_match(std::vector<int32_t>{1,2,3,4,5}, {}, false, test_scope, denied);
        check(visits == 0, "identity and steering refusals happen before any economic comparison");
        auto economics = [&](const ConversationScope& scope) {
            strata::program::PrefixChoice selection(true);
            selection.consider(0, 0, 1000);
            size_t candidate = 0;
            cache.for_each_match(std::vector<int32_t>{1,2,3,4,5}, {}, true, scope,
                [&](auto match, const auto&, const auto&) {
                    selection.consider(++candidate, match.tokens, 1 + 100 * (4 - match.tokens));
                });
            return selection.selected();
        };
        check(economics(test_scope) == 1, "same-conversation long beneficial prefix stays reusable with cost on");
        check(economics(ConversationScope{{2}}) == 0 && economics({}) == 0,
              "even an attractive measured reuse gain cannot authorize a different or absent conversation");
        check(cache.size() == size_before && cache.bytes() == bytes && cache.evictions() == evictions_before &&
              cache.best(std::vector<int32_t>{1,2,3,4,5}, {}, true, test_scope).tokens == 4,
              "economic ranking leaves parked history, byte accounting and lifecycle state unchanged");
    }
    {
        // The real restore seam must preserve a shallow sibling's deeper donor
        // for the next economic choice, held-out continuation and rollback.
        auto deep = image({1, 2, 3, 4, 5, 6, 7, 8});
        for (const auto& ids : {std::vector<int32_t>{1, 2, 3, 4},
                                std::vector<int32_t>{1, 2, 3, 4, 5, 6}}) {
            ConversationCheckpoint cut;
            cut.ids = ids;
            cut.gdn.resize(32, 7);
            deep.checkpoints.push_back(std::move(cut));
        }
        deep.live.gdn.assign(64, 9);
        deep.kv.emplace_back();
        deep.kv.back().k.resize(128, 11);
        deep.stage_images.push_back(image({1, 2, 3, 4, 5, 6, 7, 8}));
        deep.stage_images.back().live.gdn.assign(32, 12);
        const size_t frozen_bytes = deep.bytes() + deep.freeze_growth_bytes();
        ConversationCache cache(1 << 20, 2);
        check(cache.try_reserve_lease(128), "reserve the fixed checkpoint ownership before restore");
        check(cache.put(std::move(deep)), "seed divergent-choice donor");
        const std::vector<int32_t> shallow{1, 2, 3, 4, 20};
        const auto match = cache.best(shallow, {}, true, test_scope);
        const auto serial = match.serial;
        {
            ConversationCache::Restore restore(cache);
            check(restore.select(match) && restore.image() &&
                  restore.image()->live.ids.size() == 8,
                  "restore acquires the validated shallow checkpoint's whole image");
            check(restore.retained_in_cache() && cache.find(serial) != nullptr,
                  "a shallow restore preserves its deeper parked candidate");
            const auto* donor = cache.find(serial);
            check(donor && donor->bytes() == frozen_bytes &&
                  cache.leased_bytes() == 128 + restore.image()->bytes() &&
                  restore.outside_bytes() == 0 && cache.borrowed() == 1,
                  "freeze owners and the full alias are charged exactly once inside the cache envelope");
            check(donor && donor->live.gdn.data() == restore.image()->live.gdn.data() &&
                  donor->kv.front().k.frozen() && restore.image()->kv.front().k.frozen() &&
                  donor->stage_images.front().live.gdn.data() ==
                      restore.image()->stage_images.front().live.gdn.data(),
                  "ordinary borrowing shares immutable all-stage payloads without copying them");
            restore.image()->live.gdn.write()[0] = 13;
            check(donor && donor->live.gdn.read()[0] == 9,
                  "a sibling write detaches and cannot mutate the parked continuation");
            check(cache.put(image({9, 9, 9})) && cache.put(image({8, 8, 8})) &&
                  cache.find(serial) != nullptr,
                  "outgoing slot pressure cannot spend a borrowed donor");
        }
        check(cache.leased_bytes() == 128 && cache.envelope_bytes() <= (1 << 20),
              "completed or cancelled restore releases only its alias lease, never the fixed checkpoint lease");
        const auto choose = [&](bool cost, const std::vector<int32_t>& continuation, double parked_ms) {
            strata::program::PrefixChoice choice(cost);
            choice.consider(0, 0, 1000);
            choice.consider(1, 4, 10); // already-certified shallow resident source
            cache.for_each_match(continuation, {}, true, test_scope, [&](auto candidate, const auto&, const auto&) {
                choice.consider(2, candidate.tokens, parked_ms);
            });
            return choice.selected();
        };
        const std::vector<int32_t> continuation{1, 2, 3, 4, 5, 6, 7, 8, 30};
        const std::vector<int32_t> held_out{1, 2, 3, 4, 5, 6, 7, 8, 40};
        check(cache.best(continuation, {}, true, test_scope).tokens == 8 &&
              choose(true, continuation, 100) == 1 && choose(true, held_out, 100) == 1,
              "public and held-out choices still compare resident shallow against parked deep");
        check(choose(true, continuation, 1) == 2 && choose(false, held_out, 100) == 2,
              "worthwhile deeper restore and longest rollback still select the retained donor");
        const auto rollback_match = cache.best(held_out, {}, true, test_scope);
        {
            ConversationCache::Restore rollback(cache);
            check(rollback.select(rollback_match) && !rollback.retained_in_cache() &&
                  rollback.image()->live.ids.size() == 8 && !cache.find(serial) &&
                  rollback.outside_bytes() == rollback.image()->bytes(),
                  "longest rollback consumes the whole valid live donor with explicit outside ownership");
        }
        check(cache.leased_bytes() == 128, "rollback cannot refund fixed checkpoint storage");
    }
    {
        auto donor = image({1, 2, 3, 4, 5, 6});
        ConversationCheckpoint shallow, deepest;
        shallow.ids = {1, 2}; deepest.ids = {1, 2, 3, 4};
        donor.checkpoints = {shallow, deepest};
        const size_t bytes = donor.bytes();
        ConversationCache tight(bytes, 1);
        check(tight.put(std::move(donor)), "seed a donor with no extra alias headroom");
        const auto match = tight.best(std::vector<int32_t>{1, 2, 9}, {}, true, test_scope);
        tight.guard(match.serial);
        {
            ConversationCache::Restore guarded(tight);
            check(!guarded.select(match) && !guarded.image() && tight.find(match.serial) != nullptr &&
                  tight.leased_bytes() == 0,
                  "low-headroom restore cannot consume a donor guarded by another owner");
        }
        tight.unguard();
        {
            ConversationCache::Restore restore(tight);
            check(restore.select(match) && !restore.retained_in_cache() && tight.size() == 0 &&
                  restore.outside_bytes() == bytes && tight.leased_bytes() == 0,
                  "insufficient alias headroom preserves the bounded original consuming restore");
        }
        ConversationCache::Restore stale(tight);
        check(!stale.select(match) && !stale.image() && tight.leased_bytes() == 0,
              "a stale serial cannot acquire a different image or a reservation");
    }
    {
        auto pending = image({1, 2, 3});
        pending.publication_ticket = 77;
        ConversationCache cache(1 << 20, 2);
        check(cache.put(std::move(pending)), "seed a protected pending publication for restore");
        const auto match = cache.best(std::vector<int32_t>{1, 2, 3, 4}, {}, true, test_scope);
        {
            ConversationCache::Restore restore(cache);
            check(restore.select(match) && restore.retained_in_cache() &&
                  restore.image()->publication_ticket == 0 && cache.has_publication(77) &&
                  cache.leased_bytes() == 0,
                  "restore preserves pending publication ownership and its existing alias accounting");
        }
        check(cache.has_publication(77) && cache.size() == 1,
              "ending a read alias neither cancels nor consumes a pending writer");
    }
    {
        auto donor = image({1, 2, 3, 4, 5, 6});
        ConversationCheckpoint shallow, deepest;
        shallow.ids = {1, 2}; deepest.ids = {1, 2, 3, 4};
        donor.checkpoints = {shallow, deepest};
        ConversationCache cache(1 << 20, 2);
        check(cache.try_reserve_lease(128) && cache.put(std::move(donor)),
              "seed inherited guard and poison-clear ownership");
        const auto match = cache.best(std::vector<int32_t>{1, 2, 9}, {}, true, test_scope);
        cache.guard(match.serial);
        {
            ConversationCache::Restore restore(cache);
            check(restore.select(match) && restore.retained_in_cache(),
                  "a restore may share an existing same-donor guard");
        }
        check(!cache.evict_oldest() && cache.find(match.serial) != nullptr && cache.leased_bytes() == 128,
              "restore cleanup cannot release another owner's guard or fixed lease");
        cache.unguard();
        {
            ConversationCache::Restore restore(cache);
            check(restore.select(match) && cache.clear() == 1 &&
                  restore.image()->live.ids == std::vector<int32_t>({1, 2, 3, 4, 5, 6}) &&
                  cache.leased_bytes() == 128 + restore.image()->bytes(),
                  "poison clear removes parked protection but not an acquired immutable image or its charge");
        }
        check(cache.size() == 0 && cache.leased_bytes() == 128,
              "cancelled or cleared restore releases its own bytes without resurrecting a donor");
    }
    {
        // Freezing for an immutable publication lease: payloads are shared, not copied, and the alias never observes
        // a byte written through the image afterwards (copy-on-write detach).
        auto original = image({1, 2, 3, 4});
        original.live.ple.resize(1024, 11);
        original.checkpoints.push_back(ConversationCheckpoint{});
        original.checkpoints.back().ids = {1, 2};
        original.checkpoints.back().gdn.resize(64, 12);
        original.kv.push_back(ConversationKv{});
        original.kv.back().k.resize(4096, 13);
        const auto before_gdn = original.live.gdn.read();
        std::vector<uint8_t> before_k(original.kv.back().k.size());
        original.kv.back().k.read(before_k.data(), 0, before_k.size());
        auto lease = conversation_image_freeze_share(original);
        std::vector<uint8_t> leased_k(lease->kv.back().k.size());
        lease->kv.back().k.read(leased_k.data(), 0, leased_k.size());
        check(lease != nullptr && lease->scope == original.scope && lease->live.ids == original.live.ids &&
              lease->checkpoints.size() == original.checkpoints.size() &&
              lease->checkpoints.back().ids == original.checkpoints.back().ids &&
              lease->live.gdn.read() == before_gdn && lease->kv.size() == 1 &&
              leased_k == before_k,
              "a frozen alias carries the image's metadata and bytes");
        check(original.live.gdn.sealed() && original.live.ple.sealed() &&
              original.checkpoints.back().gdn.sealed() && original.kv.back().k.frozen(),
              "freezing seals every state array and freezes every buffer");
        check(original.live.gdn.data() == lease->live.gdn.data(),
              "the alias shares the frozen state instead of copying the payload");
        // A write through the image after the lease exists can only reach the lease's bytes if it did not detach.
        auto* gdn_bytes = original.live.gdn.write().data();
        gdn_bytes[0] = 0xEE;
        check(lease->live.gdn.read() == before_gdn, "a post-freeze write through the image detaches and cannot reach the lease");
        check(original.live.gdn.read() != before_gdn, "the detaching write is visible on the image's own side");
        check(original.live.gdn.write().data() != lease->live.gdn.data(),
              "the image and its lease hold different allocations after the detach");
        // Freezing reaches the nested owners too: a stage image's payloads and a checkpoint's stage parts.
        auto nested = image({8});
        nested.stage_images.push_back(image({8}));
        nested.stage_images.back().live.dead.resize(128, 4);
        nested.checkpoints.push_back(ConversationCheckpoint{});
        nested.checkpoints.back().stage_parts.push_back(ConversationCheckpoint{});
        nested.checkpoints.back().stage_parts.back().ple.resize(64, 5);
        auto nested_alias = conversation_image_freeze_share(nested);
        check(nested.stage_images.back().live.dead.sealed() &&
              nested.checkpoints.back().stage_parts.back().ple.sealed() &&
              nested_alias->stage_images.back().live.dead.data() ==
                  nested.stage_images.back().live.dead.data() &&
              nested_alias->checkpoints.back().stage_parts.back().ple.data() ==
                  nested.checkpoints.back().stage_parts.back().ple.data(),
              "freezing reaches stage images and checkpoint stage parts, sharing their allocations");
    }
    {
        // The unfrozen fallback is still correct: an alias of a buffer that was never frozen copies, so a later write
        // to the source cannot reach it either.
        auto source = image({5, 6, 7});
        source.live.tails.resize(256, 21);
        auto copy = source.frozen_alias();
        check(!source.live.tails.sealed(), "a capture that was never frozen is not reported as frozen");
        check(copy.live.tails.read() == source.live.tails.read(), "the fallback alias starts equal");
        check(copy.live.tails.data() != source.live.tails.data(),
              "the fallback alias owns its own bytes, so the source is free to move on");
        source.live.tails.write()[0] = 0x5A;
        check(copy.live.tails.read()[0] == 21, "a write to the source cannot reach an unfrozen alias");
    }
    {
        // Publication leases are charged to the same envelope as the parked payloads, and releasing them restores
        // admission exactly.  Every reference is charged in full, so the peak can never over-commit.
        auto one = image({1, 2, 3});
        one.live.gdn.resize(4096, 9);
        const size_t unit = one.bytes();
        ConversationCache cache(unit * 3, 4);
        check(cache.try_reserve_lease(unit * 3), "a lease may use the whole envelope when nothing is parked");
        check(cache.envelope_bytes() == unit * 3 && cache.leased_bytes() == unit * 3,
              "the lease is visible in the envelope and its own accounting");
        check(!cache.try_reserve_lease(1), "a lease cannot exceed the budget");
        check(!cache.can_fit(unit) && !cache.reserve_snapshot(unit) && !cache.put(image({9, 8, 7})),
              "parking and capture admission both stop while the envelope is fully leased");
        std::vector<ConversationKv> retained(1);
        retained[0].k.resize(64);
        cache.retain(std::move(retained), 3);
        check(cache.retained_bytes() == 0, "retained K/V cannot spend a leased envelope again");
        cache.release_lease(unit);
        check(cache.leased_bytes() == unit * 2 && cache.can_fit(unit),
              "releasing part of a lease restores exactly that much admission");
        cache.release_lease(unit * 4);
        check(cache.leased_bytes() == 0, "releasing more than was reserved never underflows");
        check(cache.put(image({9, 8, 7})), "parking works again once the leases are released");
        // A frozen, published image is charged in full even though its bytes are shared with the parked entry: the
        // bookkeeping stays conservative and the physical floor remains the real gate.
        auto parked = image({4, 5, 6});
        parked.live.gdn.resize(4096, 3);
        check(cache.put(std::move(parked)), "park the image that will be published");
        const size_t parked_bytes = cache.bytes();
        check(!cache.try_reserve_lease(unit * 3 - parked_bytes + 1),
              "a lease cannot spend bytes already occupied by parked histories");
        auto published = image({4, 5, 6, 7});
        published.live.gdn.resize(4096, 3);
        auto alias = conversation_image_freeze_share(published);
        check(cache.try_reserve_lease(alias->bytes()), "the publication lease is admitted beside the parked entry");
        check(cache.envelope_bytes() > parked_bytes && cache.bytes() == parked_bytes,
              "the lease changes the envelope, not the parked payload accounting");
        cache.release_lease(alias->bytes());
        check(cache.envelope_bytes() == parked_bytes,
              "releasing the publication lease returns the envelope to the parked payloads alone");
        check(published.live.gdn.data() == alias->live.gdn.data() &&
              published.live.gdn.read() == alias->live.gdn.read(),
              "the publication alias shares the frozen image's allocations with no payload copy");
    }
    {
        // Ordinary restore may borrow a pending image, but cannot consume its publication owner.
        auto pending = image({1, 2, 3, 4});
        pending.publication_ticket = 101;
        pending.geometry[0] = 7;
        pending.layer_hi = 2;
        pending.live.imgs = {{1, 42}};
        pending.live.ple.resize(32, 11);
        pending.live.tails.resize(48, 12);
        pending.live.dead.resize(16, 13);
        pending.live.block_pos.resize(8, 14);
        pending.live.stage_parts.push_back(image({1, 2, 3, 4}).live);
        pending.live.stage_parts.back().tails.resize(24, 15);
        auto cut = pending.live;
        cut.ids.resize(2);
        cut.pinned = true;
        pending.checkpoints.push_back(std::move(cut));
        ConversationKv layer;
        layer.format = 3; layer.cells = 4; layer.heads = 1; layer.head_dim = 64;
        layer.page_size = 16; layer.pooled_rows = 1; layer.idx_dim = 8;
        layer.k.resize(256, 21); layer.v.resize(128, 22);
        layer.k_scale.resize(16, 23); layer.v_scale.resize(16, 24); layer.pooled.resize(8, 25);
        pending.kv.push_back(std::move(layer));
        auto stage = image({1, 2, 3, 4});
        stage.live.dead.resize(32, 31);
        stage.kv = pending.kv;
        pending.stage_images.push_back(std::move(stage));
        const auto expected = pending;
        const auto* original_gdn = pending.live.gdn.data();
        auto data = [](const ConversationBuffer& bytes) {
            const uint8_t* result = nullptr;
            bytes.visit(0, 1, [&](const uint8_t* p, size_t, size_t) { result = p; return false; });
            return result;
        };
        const auto* original_k = data(pending.kv.front().k);
        ConversationCache cache(1u << 20, 4);
        check(cache.put(std::move(pending)), "park an unfrozen ticketed image");
        bool frozen = false;
        cache.for_each_image([&](size_t, const SavedConversation& saved) {
            frozen = saved.publication_ticket == 101 && saved.live.gdn.sealed() && saved.live.ple.sealed() &&
                saved.live.tails.sealed() && saved.live.dead.sealed() && saved.live.block_pos.sealed() &&
                saved.live.stage_parts.front().gdn.sealed() && saved.checkpoints.front().gdn.sealed() &&
                saved.checkpoints.front().stage_parts.front().tails.sealed() &&
                saved.kv.front().k.frozen() && saved.kv.front().v.frozen() &&
                saved.kv.front().k_scale.frozen() && saved.kv.front().v_scale.frozen() &&
                saved.kv.front().pooled.frozen() && saved.stage_images.front().live.dead.sealed() &&
                saved.stage_images.front().kv.front().v.frozen() &&
                saved.live.gdn.data() == original_gdn && data(saved.kv.front().k) == original_k;
        });
        check(frozen, "pending visibility already has every nested payload frozen without copying bytes");
        const size_t parked = cache.bytes(), envelope = cache.envelope_bytes();
        bool retained_in_cache = false;
        auto restored = cache.take(0, retained_in_cache);
        check(retained_in_cache && restored.publication_ticket == 0 &&
              restored.stage_images.front().publication_ticket == 0 && same_image(restored, expected),
              "pending take reports a ticket-free alias with exactly the original all-stage image");
        check(cache.has_publication(101) && cache.size() == 1 && cache.bytes() == parked &&
              cache.envelope_bytes() == envelope && cache.leased_bytes() == 0,
              "pending take leaves the sole owner and its exact charge parked");
        check(restored.live.gdn.data() == original_gdn && data(restored.kv.front().k) == original_k,
              "pending take aliases frozen payloads instead of copying them");
        restored.live.ids.front() = 99;
        restored.live.gdn.write().front() = 99;
        restored.checkpoints.front().stage_parts.front().tails.write().front() = 99;
        restored.kv.front().k.visit(0, 1, [](uint8_t* p, size_t, size_t) { *p = 99; return true; });
        restored.stage_images.front().live.dead.write().front() = 99;
        restored.stage_images.front().kv.front().v.visit(0, 1, [](uint8_t* p, size_t, size_t) { *p = 99; return true; });
        auto publication = cache.take_publication(101);
        check(publication && publication->publication_ticket == 101 && same_image(*publication, expected) &&
              publication->live.gdn.data() == original_gdn && data(publication->kv.front().k) == original_k,
              "restore-side writes cannot change any original publication bytes or ownership");
        check(cache.bytes() == 0 && cache.size() == 0 && cache.leased_bytes() == parked &&
              cache.envelope_bytes() == envelope && !cache.has_publication(101),
              "acquiring the owner transfers its exact parked charge without growing the envelope");
        cache.release_lease(publication->bytes());
        check(cache.envelope_bytes() == 0, "releasing the acquired pending owner drains its exact charge");
    }
    {
        // Victim planning and compaction must skip the same protected entry.
        auto pending = image({1, 2, 3});
        pending.publication_ticket = 201;
        pending.freeze();
        const size_t protected_bytes = pending.bytes(), budget = 1u << 20;
        ConversationCache cache(budget, 3);
        check(cache.put(std::move(pending)) && cache.put(image({9, 8, 7})) && cache.put(image({6, 5, 4})),
              "fill a table with its pending owner oldest");
        auto next = image({1, 2, 3, 4});
        const size_t next_bytes = next.bytes();
        check(cache.put(std::move(next)) && cache.has_publication(201) &&
              cache.evictions() == 1 && cache.superseded() == 0 && cache.size() == 3 &&
              cache.bytes() == protected_bytes + image({6, 5, 4}).bytes() + next_bytes,
              "put neither supersedes nor compacts away a pending owner when evicting an ordinary victim");
        check(cache.best(std::vector<int32_t>{9, 8, 7, 6}, {}, true, test_scope).tokens == 0 &&
              cache.best(std::vector<int32_t>{6, 5, 4, 3}, {}, true, test_scope).tokens == 3,
              "protected compaction removes the planned oldest ordinary entry, not its successor");
        check(cache.evict_oldest() && cache.evict_oldest() && !cache.evict_oldest() &&
              cache.size() == 1 && cache.bytes() == protected_bytes && cache.has_publication(201),
              "physical pressure evicts every ordinary image but preserves the pending sole owner");
        auto incoming = image({8, 7, 6});
        incoming.live.gdn.resize(budget - protected_bytes + 1, 5);
        check(incoming.bytes() <= budget && !cache.reserve_snapshot(incoming.bytes()) &&
              !cache.put(std::move(incoming)) && cache.bytes() == protected_bytes && cache.has_publication(201),
              "snapshot and put honestly refuse byte pressure rather than spending a pending owner");
        cache.release_publication(0);
        cache.release_publication(999);
        check(cache.has_publication(201), "zero and unknown releases cannot unprotect another entry");
        cache.release_publication(201);
        cache.release_publication(201);
        check(!cache.has_publication(201) && cache.leased_bytes() == 0 && cache.evict_oldest() && cache.bytes() == 0,
              "releasing pending protection is safe to repeat and restores ordinary eviction");
    }
    {
        ConversationCache cache(1u << 20, 1);
        auto pending = image({1, 2, 3});
        pending.publication_ticket = 211;
        check(cache.put(std::move(pending)), "fill the only slot with a pending publication");
        const size_t parked = cache.bytes();
        auto duplicate = image({9, 8, 7});
        duplicate.publication_ticket = 211;
        check(!cache.put(std::move(duplicate)) && duplicate.publication_ticket == 211 &&
              duplicate.live.ids == std::vector<int32_t>({9, 8, 7}) && !duplicate.live.gdn.sealed(),
              "a duplicate ticket cannot acquire a second parked owner");
        auto next = image({1, 2, 3, 4});
        next.publication_ticket = 212;
        const auto expected = next;
        check(!cache.put(std::move(next)) && next.publication_ticket == 212 && same_image(next, expected) &&
              next.live.gdn.sealed() && cache.bytes() == parked && cache.has_publication(211) &&
              !cache.has_publication(212) && cache.evictions() == 0 && cache.superseded() == 0,
              "a full pending slot refuses even its extension without losing either candidate or owner");
        check(!cache.put(image({1, 2, 3, 4})), "an ordinary extension cannot supersede a pending owner either");
        cache.release_publication(211);
        check(cache.put(std::move(next)) && cache.size() == 1 && cache.has_publication(212) &&
              cache.superseded() == 1 && cache.evictions() == 0,
              "unprotection restores normal compatible supersession without relaxing the slot limit");
    }
    {
        auto pinned = image({1, 2, 3});
        pinned.publication_ticket = 221;
        pinned.checkpoints = {pinned.live};
        pinned.checkpoints.front().ids.resize(2);
        pinned.checkpoints.front().pinned = true;
        ConversationCache cache(1u << 20, 1);
        check(cache.put(std::move(pinned)) && cache.try_reserve_lease(17), "park a pending explicit pin beside a separate lease");
        cache.release_publication(221);
        check(!cache.has_publication(221) && cache.leased_bytes() == 17 && !cache.evict_oldest() &&
              !cache.put(image({1, 2, 3, 4})),
              "unprotection clears no explicit pin, spends no separate lease and changes no ordinary pin policy");
        cache.release_lease(17);
    }
    {
        // The shared root is allocated before the allocation-free lease-transfer commit.
        auto first = image({2, 4, 6});
        first.publication_ticket = 301;
        first.freeze();
        const size_t first_bytes = first.bytes(), separate_lease = 17;
        const auto* payload = first.live.gdn.data();
        const auto expected = first;
        auto second = image({3, 6, 9});
        second.publication_ticket = 302;
        ConversationCache cache(1u << 20, 4);
        check(cache.try_reserve_lease(separate_lease) && cache.put(std::move(first)) && cache.put(std::move(second)),
              "park two protected owners beside an existing acquired lease");
        std::vector<ConversationKv> reuse(1);
        reuse.front().k.resize(128, 8);
        cache.retain(std::move(reuse), 3);
        check(cache.retained_bytes() > 0, "retain optional K/V beside the waiting owners");
        const size_t parked = cache.bytes(), envelope = cache.envelope_bytes();
        auto root = std::make_shared<SavedConversation>();
        static_assert(noexcept(cache.take_publication(301)));
        auto acquired = cache.take_publication(301);
        check(acquired.has_value(), "acquire a waiting owner after root allocation");
        *root = std::move(*acquired);
        check(root->publication_ticket == 301 && root->bytes() == first_bytes && root->live.gdn.data() == payload &&
              same_image(*root, expected) && cache.bytes() == parked - first_bytes &&
              cache.leased_bytes() == separate_lease + first_bytes && cache.envelope_bytes() == envelope &&
              !cache.has_publication(301) && cache.has_publication(302),
              "allocation-free transfer preserves payload identity and the complete envelope including other leases and reuse");
        check(!cache.has_publication(0) && !cache.take_publication(0) && !cache.take_publication(301) &&
              !cache.take_publication(999) && cache.envelope_bytes() == envelope,
              "zero, missing and already acquired tickets cannot charge another lease");
        cache.release_publication(301);
        check(cache.leased_bytes() == separate_lease + first_bytes,
              "releasing an already acquired ticket never subtracts its outstanding lease");
        check(cache.clear() == 1 && cache.bytes() == 0 && cache.retained_bytes() == 0 &&
              !cache.has_publication(302) && cache.leased_bytes() == separate_lease + first_bytes &&
              same_image(*root, expected),
              "clear invalidates parked publications and reuse without touching separately acquired leases or bytes");
        cache.release_publication(302);
        cache.release_lease(root->bytes());
        check(cache.leased_bytes() == separate_lease && cache.envelope_bytes() == separate_lease,
              "the acquired owner's exact lease release leaves the unrelated reservation intact");
        cache.release_lease(separate_lease);
        check(cache.envelope_bytes() == 0, "all separately acquired leases drain independently after clear");
    }
    {
        auto pending = image({4, 5, 6});
        pending.publication_ticket = 401;
        const size_t unfrozen_bytes = pending.bytes();
        const auto expected = pending;
        ConversationCache cache(unfrozen_bytes, 1);
        check(!cache.put(std::move(pending)) && pending.publication_ticket == 401 &&
              pending.live.gdn.sealed() && pending.bytes() > unfrozen_bytes && same_image(pending, expected) &&
              cache.size() == 0 && cache.envelope_bytes() == 0,
              "ticketed admission prices frozen owner metadata before commit and refuses a genuinely too-small envelope");
    }
    {
        // The stored candidate's contract inside the existing selector (the demand path feeds its disk candidate
        // through the same PrefixChoice, never a second selector): an equal cut never outranks the RAM/live/resident
        // source enumerated first, and an unpriced longer candidate keeps the longest-prefix rollback rather than
        // inventing an SSD rate.
        strata::program::PrefixChoice choice(false);
        choice.consider(0, 1000, 5.0);          // the RAM/live cut, priced by real observations
        choice.consider(1, 1000, std::nullopt); // an equal stored cut, unpriced
        check(choice.selected() == 0, "an equal stored cut does not displace the RAM source enumerated first");
        strata::program::PrefixChoice cost(true);
        cost.consider(0, 1000, 5.0);
        cost.consider(1, 2000, std::nullopt);
        check(cost.longest() == 1 && cost.selected() == 1 && !cost.using_cost(),
              "an unpriced longer stored cut keeps the longest rollback instead of a measured choice");
        strata::program::PrefixChoice priced(true);
        priced.consider(0, 1000, 5.0);
        priced.consider(1, 2000, 9.0);
        check(priced.using_cost() && priced.selected() == 0,
              "a priced stored candidate competes in the existing cost policy like any other source");
    }
    {
        ConversationScope scope{{7}}, parsed;
        const std::string lower(64, 'a'), upper(64, 'A'), zero(64, '0');
        check(conversation_shared_scope_parse(lower, parsed) && parsed.valid(),
              "native shared scope accepts an exact lowercase authorization digest");
        const auto prior = parsed;
        check(!conversation_shared_scope_parse(upper, parsed) &&
              !conversation_shared_scope_parse(zero, parsed) &&
              !conversation_shared_scope_parse(lower + "0", parsed) && parsed == prior,
              "invalid shared scope is refused atomically without changing the accepted scope");
        for (bool measured : {false, true}) for (size_t private_source = 0; private_source < 5; ++private_source) {
            strata::program::PrefixChoice choice(measured);
            // Each private class is enumerated before the shared store candidate:
            // live, checkpoint, resident, parked and private persistent store.
            choice.consider(private_source, 512, 7.0);
            choice.consider(5, 512, std::nullopt);
            check(choice.selected() == private_source, "every equal-length private source beats the shared store");
            choice.consider(6, 1024, std::nullopt);
            check(choice.selected() == 6, "a strictly longer authorized shared cut remains eligible");
        }
        ConversationCache cache(1u << 20, 2);
        auto shared = image({1, 2, 3});
        shared.scope = scope;
        check(cache.put(std::move(shared)), "the shared RAM publication owner is retained separately");
        check(cache.best(std::vector<int32_t>{1,2,3,4}, {}, true, test_scope).tokens == 0 &&
              cache.best(std::vector<int32_t>{1,2,3,4}, {}, true, scope).tokens == 3,
              "shared publication payload cannot become an implicit private RAM source");
    }
    {
        // The reuse intervals a capture may offer: each one names the donor head the retained bytes came from, ends on
        // that donor's own object grid, and claims nothing the capture did not keep.  The store re-validates all of it;
        // this is the caller-side half of the proof.
        const uint64_t chunk = 1u << 20;
        std::vector<ConversationStoreStreamInfo> streams;
        auto add_stream = [&](uint32_t stage, ConversationStoreStreamKind kind, uint64_t ordinal, uint64_t bytes) {
            ConversationStoreStreamInfo info;
            info.id = ConversationStoreStreamId{stage, uint16_t(kind), 0, ordinal};
            info.index = uint32_t(streams.size());
            info.bytes = bytes;
            streams.push_back(info);
        };
        add_stream(0, ConversationStoreStreamKind::kv_k, 0, 4 * chunk);
        add_stream(0, ConversationStoreStreamKind::kv_v, 0, 4 * chunk);
        add_stream(0, ConversationStoreStreamKind::kv_pooled, 0, chunk / 2);
        add_stream(1, ConversationStoreStreamKind::kv_k, 0, 2 * chunk);
        auto kept = [](uint32_t stage, uint32_t layer, uint32_t part, uint64_t bytes, uint64_t donor_bytes) {
            ConversationRetainedPrefix prefix;
            prefix.stage = stage;
            prefix.layer = layer;
            prefix.part = part;
            prefix.bytes = bytes;
            prefix.donor_bytes = donor_bytes;
            return prefix;
        };
        const std::vector<ConversationRetainedPrefix> reported = {
            kept(0, 0, 0, 3 * chunk + 4096, 4 * chunk),   // an unaligned prefix of this part's objects
            kept(0, 0, 4, chunk / 2, chunk / 2),          // the whole (short) pooled page: it ends both streams
            kept(1, 0, 0, 2 * chunk, 2 * chunk),          // a stage image's own stream, not the top image's
        };
        const auto reuse = conversation_store_reuse_intervals(streams, reported, 7, 5, 5, chunk);
        check(reuse.size() == 3, "every retained part may offer exactly one interval");
        check(reuse[0].head_id == 7 && reuse[0].offset == 0 && reuse[0].source_offset == 0 &&
              reuse[0].stream.stage == 0 && reuse[0].stream.ordinal == 0 &&
              reuse[0].stream.kind == uint16_t(ConversationStoreStreamKind::kv_k) &&
              reuse[0].bytes == 3 * chunk,
              "an unaligned retained prefix is offered down to the donor's own object grid");
        check(reuse[1].bytes == chunk / 2 && reuse[1].stream.kind == uint16_t(ConversationStoreStreamKind::kv_pooled),
              "a retained part that ends both streams needs no object grid");
        check(reuse[2].stream.stage == 1 && reuse[2].bytes == 2 * chunk,
              "a stage image's interval names the stage's own stream identity");
        check(conversation_store_reuse_intervals(streams, reported, 7, 4, 5, chunk).empty(),
              "a donor captured in another execution-domain epoch authorizes nothing");
        check(conversation_store_reuse_intervals(streams, reported, 0, 5, 5, chunk).empty(),
              "a capture with no named donor authorizes nothing");
        check(conversation_store_reuse_intervals(streams, reported, 7, 5, 0, chunk).empty(),
              "a capture outside any execution epoch authorizes nothing");
        const auto clamped = conversation_store_reuse_intervals(streams,
            {kept(0, 0, 0, 5 * chunk, 3 * chunk)}, 7, 5, 5, chunk);
        check(clamped.size() == 1 && clamped[0].bytes == 3 * chunk,
              "an interval is bounded by what the donor actually holds");
        const auto unknown = conversation_store_reuse_intervals(streams, {kept(0, 0, 1, chunk, chunk),
                                                                         kept(0, 3, 0, chunk, chunk)}, 7, 5, 5, chunk);
        check(unknown.size() == 1 && unknown[0].bytes == chunk,
              "a part whose stream identity is absent is skipped, never guessed at");
        check(conversation_store_reuse_intervals(streams, {kept(0, 0, 1, 4096, 4 * chunk)}, 7, 5, 5, chunk).empty(),
              "a retained prefix shorter than one object offers nothing");
        check(conversation_store_reuse_intervals(streams, {kept(0, 0, 1, 9 * chunk, 9 * chunk)}, 7, 5, 5, chunk).empty(),
              "a retained length longer than the outgoing stream offers nothing");
    }
    {
        // The retained K/V carries the store head it was restored from, so a later capture can name the donor; a
        // RAM-only restore carries none, and a capture then offers nothing.
        ConversationCache cache(1u << 20, 2);
        std::vector<ConversationKv> layers(1);
        layers[0].k.resize(512);
        cache.retain(std::move(layers), 8, {}, 91, 4);
        const ConversationKvReuse named = cache.take_reuse();
        check(named.donor_head == 91 && named.donor_epoch == 4 && named.captured_tokens == 8,
              "retained K/V keeps the head and epoch it was restored from");
        std::vector<ConversationKv> anonymous(1);
        anonymous[0].k.resize(512);
        cache.retain(std::move(anonymous), 8);
        check(cache.take_reuse().donor_head == 0, "a RAM-only restore retains its bytes without naming a donor");
    }
    std::printf("conversation_cache_test: %d checks passed\n", checks);
}
