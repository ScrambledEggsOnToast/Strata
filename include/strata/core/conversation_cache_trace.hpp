// Opt-in metadata receipts for the existing conversation-cache ownership seams.
#pragma once

#include "strata/core/conversation_cache_victim_replay.hpp"

#include <algorithm>
#include <chrono>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace strata::core {

// One catalog-mutex snapshot. lease/read_preparation are overlapping views, not
// additional charges. held = payload + metadata + descriptors + decode.
struct ConversationStoreRamTrace {
    uint64_t held = 0, payload = 0, lease = 0, metadata = 0, descriptors = 0, decode = 0;
    uint64_t publication_image = 0, publication_workspace = 0, read_preparation = 0, allowance = 0;
    uint64_t read_reserved = 0, read_image = 0;
    bool admission_suspended = false;
};

class ConversationCacheTrace {
public:
    static constexpr size_t max_records = 4096, max_candidates = 64, max_line = 4096;
    static constexpr uint64_t max_hash_words = 64ull << 20;
    struct External {
        ConversationStoreRamTrace store;
        uint64_t checkpoint_lease = 0, preparation = 0, outside = 0;
        bool complete = true;
    };
    struct State {
        const char* site = "observation";
        const char* action = "state";
        bool accepted = true;
        bool subject_required = false;
        size_t budget = 0, slots = 0, parked = 0, parked_bytes = 0, reuse_bytes = 0, leased_bytes = 0;
        size_t held = 0, held_entries = 0, incoming = 0, limit = SIZE_MAX, available_entries = 0;
        size_t victims = 0, superseded = 0, evictions = 0;
        uint64_t guarded = 0, protected_serial = 0, subject_serial = 0, head = 0;
        double measured_ms = -1;
        uint64_t work_bytes = 0, route = 0;
        bool work_known = false;
        int64_t begin_tokens = -1, end_tokens = -1;
        uint64_t replay_contract = 0;
        const char* replay_mode = "off";
        const char* replay_decision = "off";
        size_t replay_count = 0;
    };

    // The destination is caller-owned, create-only in serving. No trace object,
    // clock read, hash, callback, I/O or allocation exists on the disabled path.
    explicit ConversationCacheTrace(FILE* file) : file_(file) {}
    ConversationCacheTrace(const ConversationCacheTrace&) = delete;
    ConversationCacheTrace& operator=(const ConversationCacheTrace&) = delete;
    ~ConversationCacheTrace() { finish(); }
    using Sampler = void (*)(void*, External&);
    void set_sample(void* context, Sampler sample) noexcept {
        sample_context_ = context; sample_ = sample;
    }
    uint64_t request_sequence = 0, leg = 0;
    char request[97]{}, incarnation[33]{}; // validated phase identity, otherwise empty
    uint64_t next_event() const noexcept { return event_ + 1; }
    bool recording() const noexcept { return file_ && !stopped_; }
    bool accepts_victim_plan(size_t candidates, bool subject, size_t nominees) const noexcept {
        return recording() && candidates <= max_candidates &&
            records_ + candidates + size_t(subject) + nominees + 2 < max_records;
    }
    bool valid_phase_identity() const noexcept {
        size_t n = 0;
        for (; n < sizeof(request) && request[n]; ++n) {
            const char c = request[n];
            if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                  (c >= '0' && c <= '9') || c == '_' || c == '-')) return false;
        }
        if (!n || n == sizeof(request) || !leg || incarnation[32]) return false;
        for (size_t i = 0; i < 32; ++i)
            if (!((incarnation[i] >= '0' && incarnation[i] <= '9') ||
                  (incarnation[i] >= 'a' && incarnation[i] <= 'f'))) return false;
        return true;
    }
    void set_victim_replay(const ConversationVictimReplay* replay) noexcept {
        if (!replay) return;
        replay_contract_ = replay->contract;
        replay_status_ = replay->status;
    }
    void victim_replay_decision(const char* decision, bool forced) noexcept {
        replay_status_ = decision;
        ++replay_decisions_;
        if (forced) ++replay_forced_;
    }

    bool begin(const State& s, size_t candidates, bool subject) noexcept {
        if (!file_ || stopped_) return false;
        const size_t count = std::min(candidates, max_candidates);
        if (records_ + count + size_t(subject) + s.replay_count + 2 >= max_records) {
            limited_ = true;
            finish();
            return false;
        }
        External e;
        e.complete = sample_ != nullptr; // an absent ledger is unknown, never an all-zero observation
        if (sample_) {
            try { sample_(sample_context_, e); }
            catch (...) { e.complete = false; }
        }
        ++event_;
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        record("{\"kind\":\"cache_mutation\",\"version\":1,\"event\":%llu,\"ns\":%llu,"
            "\"site\":\"%s\",\"action\":\"%s\",\"accepted\":%s,\"request_sequence\":%llu,"
            "\"request\":\"%s\",\"leg\":%llu,\"incarnation\":\"%s\","
            "\"budget\":%zu,\"slots\":%zu,\"parked\":%zu,\"parked_bytes\":%zu,\"reuse_bytes\":%zu,"
            "\"leased_bytes\":%zu,\"held\":%zu,\"held_entries\":%zu,\"incoming\":%zu,\"limit\":%zu,"
            "\"available_entries\":%zu,\"victims\":%zu,\"superseded\":%zu,\"evictions\":%zu,"
            "\"guarded\":%llu,\"protected_serial\":%llu,\"subject_serial\":%llu,\"head\":%llu,"
            "\"measured_ms\":%.9g,\"work_bytes\":%llu,\"route\":%llu,\"work_known\":%s,"
            "\"begin_tokens\":%lld,\"end_tokens\":%lld,"
            "\"candidate_count\":%zu,\"candidates_complete\":%s,\"external_complete\":%s,"
            "\"subject_present\":%s,\"subject_complete\":%s,"
            "\"checkpoint_lease\":%llu,\"preparation\":%llu,\"outside\":%llu,"
            "\"store_held\":%llu,\"store_payload\":%llu,\"store_lease\":%llu,\"store_metadata\":%llu,"
            "\"store_descriptors\":%llu,\"store_decode\":%llu,\"publication_image\":%llu,"
            "\"publication_workspace\":%llu,\"read_preparation\":%llu,\"read_allowance\":%llu,"
            "\"read_reserved\":%llu,\"read_image\":%llu,"
            "\"read_admission_suspended\":%s,\"replay_contract\":\"%016llx\","
            "\"replay_mode\":\"%s\",\"replay_decision\":\"%s\",\"replay_count\":%zu}",
            ull(event_), ull(ns), s.site, s.action, boolean(s.accepted), ull(request_sequence),
            request, ull(leg), incarnation, s.budget, s.slots, s.parked, s.parked_bytes, s.reuse_bytes,
            s.leased_bytes, s.held, s.held_entries, s.incoming, s.limit, s.available_entries,
            s.victims, s.superseded, s.evictions, ull(s.guarded), ull(s.protected_serial), ull(s.subject_serial),
            ull(s.head), s.measured_ms, ull(s.work_bytes), ull(s.route), boolean(s.work_known), (long long)s.begin_tokens,
            (long long)s.end_tokens, candidates, boolean(candidates <= max_candidates), boolean(e.complete),
            boolean(subject), boolean(!s.subject_required || subject),
            ull(e.checkpoint_lease), ull(e.preparation), ull(e.outside), ull(e.store.held), ull(e.store.payload),
            ull(e.store.lease), ull(e.store.metadata), ull(e.store.descriptors), ull(e.store.decode),
            ull(e.store.publication_image), ull(e.store.publication_workspace), ull(e.store.read_preparation),
            ull(e.store.allowance), ull(e.store.read_reserved), ull(e.store.read_image),
            boolean(e.store.admission_suspended), ull(s.replay_contract), s.replay_mode,
            s.replay_decision, s.replay_count);
        return !stopped_;
    }

    void nomination(const ConversationVictimIdentity& image) noexcept {
        record("{\"kind\":\"cache_victim_nomination\",\"event\":%llu,\"serial\":%llu,"
            "\"scope_hash\":\"%016llx\",\"history_hash\":\"%016llx\","
            "\"tokens\":%llu,\"bytes\":%llu}", ull(event_), ull(image.serial), ull(image.scope),
            ull(image.history), ull(image.tokens), ull(image.bytes));
    }

    template<class Image>
    void image(const char* role, size_t index, uint64_t serial, const Image& image,
               bool eligible, bool guarded, bool protected_entry, bool chosen, bool superseded) noexcept {
        if (stopped_) return;
        uint64_t history = 1469598103934665603ull;
        const uint64_t words = image.live.ids.size() + 2 * image.live.imgs.size();
        const bool complete = words <= max_hash_words - hash_words_;
        if (complete) {
            hash_words_ += words;
            for (auto id : image.live.ids) history = (history ^ uint32_t(id)) * 1099511628211ull;
            for (const auto& key : image.live.imgs) {
                history = (history ^ uint64_t(key.start)) * 1099511628211ull;
                history = (history ^ key.hash) * 1099511628211ull;
            }
        } else { history = 0; hash_limited_ = true; }
        uint64_t scope = 1469598103934665603ull;
        for (auto b : image.scope.value) scope = (scope ^ b) * 1099511628211ull;
        size_t checkpoints = image.live.bytes() + image.checkpoints.capacity() * sizeof(image.live);
        for (const auto& c : image.checkpoints) checkpoints += c.bytes();
        record("{\"kind\":\"cache_image\",\"event\":%llu,\"role\":\"%s\",\"index\":%zu,\"serial\":%llu,"
            "\"scope_hash\":\"%016llx\",\"history_hash\":\"%016llx\",\"history_complete\":%s,"
            "\"tokens\":%zu,\"bytes\":%zu,\"checkpoint_bytes\":%zu,\"checkpoints\":%zu,\"stages\":%zu,"
            "\"cvec\":%s,\"pinned\":%s,\"publication_ticket\":%llu,\"guarded\":%s,\"protected\":%s,"
            "\"eligible\":%s,\"chosen\":%s,\"superseded\":%s}", ull(event_), role, index, ull(serial),
            ull(scope), ull(history), boolean(complete), image.live.ids.size(), image.bytes(), checkpoints,
            image.checkpoints.size(), image.stage_images.size(), boolean(image.cvec), boolean(image.pinned()),
            ull(image.publication_ticket), boolean(guarded), boolean(protected_entry), boolean(eligible),
            boolean(chosen), boolean(superseded));
    }
    void end() noexcept { record("{\"kind\":\"cache_event_end\",\"event\":%llu}", ull(event_)); }
    void finish() noexcept {
        if (!file_ || stopped_) return;
        record("{\"kind\":\"cache_trace_end\",\"events\":%llu,\"records\":%zu,\"record_limit\":%zu,"
            "\"limited\":%s,\"hash_limited\":%s,\"hash_words\":%llu,"
            "\"replay_contract\":\"%016llx\",\"replay_status\":\"%s\","
            "\"replay_decisions\":%zu,\"replay_forced\":%zu}", ull(event_), records_, max_records,
            boolean(limited_), boolean(hash_limited_), ull(hash_words_), ull(replay_contract_),
            replay_status_, replay_decisions_, replay_forced_);
        std::fflush(file_);
        stopped_ = true;
    }
private:
    static unsigned long long ull(uint64_t n) { return (unsigned long long)n; }
    static const char* boolean(bool b) { return b ? "true" : "false"; }
    void record(const char* format, ...) noexcept {
        if (stopped_ || !file_) return;
        char line[max_line];
        va_list args;
        va_start(args, format);
        const int n = std::vsnprintf(line, sizeof(line), format, args);
        va_end(args);
        if (n < 0 || size_t(n) >= sizeof(line) || std::fputs(line, file_) < 0 || std::fputc('\n', file_) < 0) {
            stopped_ = true; // missing closure explicitly disqualifies an incomplete capture
            return;
        }
        ++records_;
    }
    FILE* file_;
    void* sample_context_ = nullptr;
    Sampler sample_ = nullptr;
    size_t records_ = 0;
    uint64_t event_ = 0, hash_words_ = 0;
    uint64_t replay_contract_ = 0;
    const char* replay_status_ = "off";
    size_t replay_decisions_ = 0, replay_forced_ = 0;
    bool stopped_ = false, limited_ = false, hash_limited_ = false;
};
} // namespace strata::core
