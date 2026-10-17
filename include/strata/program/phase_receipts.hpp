#pragma once

// Optional diagnostic receipts, not scheduling or latency accounting. Host spans
// are inclusive; materialization and publication may overlap other requests.
// Producers only accumulate fixed-size metadata. Output is drained at request,
// idle or rare publication-maintenance terminals, never on each token/window.
#include <array>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string_view>

namespace strata::program::phase {

using Clock = std::chrono::steady_clock;
inline bool enabled() {
    static const bool on = [] {
        const char* value = std::getenv("STRATA_PHASE_TIMING");
        return value && std::strcmp(value, "1") == 0;
    }();
    return on;
}
inline uint64_t now_ns() {
    return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count());
}
inline void increment(uint64_t& value) { if (value != UINT64_MAX) ++value; }
inline bool next_sequence(uint64_t& value, uint64_t& out) {
    if (value == UINT64_MAX) return false;
    out = ++value;
    return true;
}

struct Identity {
    static constexpr size_t max_request = 96;
    char request[max_request + 1]{};
    char incarnation[33]{};
    uint64_t leg = 0;
    enum class Priority : uint8_t { unknown, foreground, background } priority = Priority::unknown;
    bool valid() const { return request[0] && incarnation[0] && leg; }
    const char* priority_name() const {
        return priority == Priority::foreground ? "foreground" : priority == Priority::background ? "background" : "unknown";
    }
};
inline bool valid_request(std::string_view value) {
    if (value.empty() || value.size() > Identity::max_request) return false;
    for (char c : value)
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
              (c >= '0' && c <= '9') || c == '_' || c == '-')) return false;
    return true;
}
inline bool positive_u64(std::string_view value, uint64_t& out) {
    if (value.empty() || value.front() < '1' || value.front() > '9') return false;
    uint64_t n = 0;
    for (char c : value) {
        if (c < '0' || c > '9' || n > (UINT64_MAX - uint64_t(c - '0')) / 10) return false;
        n = n * 10 + uint64_t(c - '0');
    }
    out = n;
    return true;
}
struct Input {
    Identity identity;
    uint8_t seen = 0;
    bool invalid = false;
    // Unknown/legacy keys remain optional. A partial, duplicate or malformed
    // identity is uncorrelated, never attached to the prior slot's request.
    void key(std::string_view name, std::string_view value) {
        uint8_t bit = name == "phase_request" ? 1 : name == "phase_leg" ? 2 :
                      name == "phase_incarnation" ? 4 : name == "phase_priority" ? 8 : 0;
        if (!bit) return;
        if (seen & bit) invalid = true;
        seen |= bit;
        if (bit == 1) {
            if (!valid_request(value)) invalid = true;
            else { std::memcpy(identity.request, value.data(), value.size()); identity.request[value.size()] = 0; }
        } else if (bit == 2) {
            if (!positive_u64(value, identity.leg)) invalid = true;
        } else if (bit == 4) {
            bool valid = value.size() == 32;
            for (char c : value) valid = valid && ((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'));
            if (!valid) invalid = true;
            else { std::memcpy(identity.incarnation, value.data(), 32); identity.incarnation[32] = 0; }
        } else {
            if (value == "foreground") identity.priority = Identity::Priority::foreground;
            else if (value == "background") identity.priority = Identity::Priority::background;
            else if (value != "unknown") invalid = true;
        }
    }
    void finish() {
        invalid = invalid || (seen && (seen & 7) != 7);
        if (invalid) identity = {};
    }
};
inline bool generation_command(std::string_view line) {
    return line.starts_with("GEN ") || line.starts_with("GENI ") ||
           line.starts_with("BGEN ") || line.starts_with("BGENI ");
}
inline Input parse_input(std::string_view line) {
    Input out;
    const bool batched = line.starts_with("BGEN ") || line.starts_with("BGENI ");
    size_t at = 0;
    auto token = [&]() {
        while (at < line.size() && line[at] == ' ') ++at;
        const size_t start = at;
        while (at < line.size() && line[at] != ' ') ++at;
        return line.substr(start, at - start);
    };
    token(); // command
    if (batched) token(); // slot is not an identity
    token(); // max_new
    for (;;) {
        const auto part = token();
        const size_t equal = part.find('=');
        if (equal == std::string_view::npos) break; // image path or IDs: no prompt scan
        out.key(part.substr(0, equal), part.substr(equal + 1));
    }
    out.finish();
    return out;
}

enum class Stage : uint8_t {
    demand_identity, demand_wait, demand_materialize, resident_giveback,
    resident_copyin, gpu_mount, prefill_run, store_seam, decode_interleave,
    park_capture, decode_residence, count
};
inline constexpr const char* stage_names[] = {
    "demand_identity", "demand_wait", "demand_materialize", "resident_giveback",
    "resident_copyin", "gpu_mount", "prefill_run", "store_seam", "decode_interleave",
    "park_capture", "decode_residence"
};
struct Total { double ms = 0; uint64_t count = 0; bool known = true, overflow = false, unavailable = false; };
struct Leg {
    Input input;
    uint64_t sequence = 0, started_ns = 0;
    std::array<Total, size_t(Stage::count)> totals{};
    std::array<uint32_t, size_t(Stage::count)> depth{};
    std::array<uint64_t, size_t(Stage::count)> stage_started{};
    uint64_t aggregate_overflow = 0;
    uint64_t demand_request = 0, demand_generation = 0, demand_attempt = 0, demand_head = 0;
    uint64_t shared_prefix_tokens = 0, shared_prefix_restored_tokens = 0;
    uint64_t shared_lookup_ns = 0, shared_lookup_catalog_version = 0;
    uint64_t shared_lookup_matches = 0, shared_lookup_eligible = 0, shared_lookup_head = 0;
    uint64_t periodic_checkpoint_count = 0, periodic_checkpoint_last_tokens = 0, periodic_checkpoint_last_ns = 0;
    const char* demand_status = "unavailable";
    bool demand_attempt_known = false, terminal = false;
    bool demand_started = false, demand_measured = false;
    const char* status = "error";
    int slot = -1;
    explicit Leg(Input value, uint64_t sequence_value, uint64_t started)
        : input(value), sequence(sequence_value), started_ns(started) {
        totals[size_t(Stage::demand_materialize)].known = false;
        totals[size_t(Stage::decode_residence)].known = false;
    }
    void add(Stage stage, double ms, uint64_t count = 1) {
        auto& total = totals[size_t(stage)];
        const bool count_overflow = count > UINT64_MAX - total.count;
        const bool time_overflow = ms < 0 || !std::isfinite(ms) || !std::isfinite(total.ms + ms);
        if (!total.overflow && (count_overflow || time_overflow)) increment(aggregate_overflow);
        total.overflow = total.overflow || count_overflow || time_overflow;
        total.known = !total.overflow && !total.unavailable;
        if (!time_overflow) total.ms += ms;
        total.count = count_overflow ? UINT64_MAX : total.count + count;
    }
    void unavailable(Stage stage) {
        auto& total = totals[size_t(stage)];
        total.unavailable = true;
        total.known = false;
    }
    // Only the latest demand's identity is printed; totals still span all
    // fallbacks. Clear before even disabled/deadline/admission refusals.
    void begin_demand(uint64_t request = 0, uint64_t generation = 0) {
        demand_request = request;
        demand_generation = generation;
        demand_attempt = demand_head = 0;
        demand_attempt_known = demand_started = demand_measured = false;
        demand_status = "unavailable";
    }
    void observe_demand(uint64_t attempt, uint64_t head, bool started, bool prior_unmeasured) {
        if (prior_unmeasured) unavailable(Stage::demand_materialize);
        if (!attempt) return;
        if (demand_attempt_known && demand_attempt != attempt) {
            if (demand_started && !demand_measured) unavailable(Stage::demand_materialize);
            demand_started = demand_measured = false;
        }
        demand_attempt = attempt;
        demand_attempt_known = true;
        demand_head = head;
        demand_started = demand_started || started;
    }
    void materialized_demand(double ms, uint64_t head) {
        add(Stage::demand_materialize, ms);
        demand_measured = true;
        demand_head = head;
    }
    void finish_demand(const char* value) {
        demand_status = value;
        if (demand_started && !demand_measured) unavailable(Stage::demand_materialize);
    }
    bool finish(const char* value, uint64_t ended) {
        if (terminal) return false;
        for (size_t i = 0; i < depth.size(); ++i) {
            if (!depth[i]) continue;
            add(Stage(i), ended >= stage_started[i] ? double(ended - stage_started[i]) / 1e6 : -1);
            depth[i] = 0;
            stage_started[i] = 0;
        }
        terminal = true;
        status = value;
        return true;
    }
};
class Span {
    Leg* leg_;
    Stage stage_;
public:
    Span(Leg* leg, Stage stage) : leg_(leg && !leg->terminal ? leg : nullptr), stage_(stage) {
        if (leg_ && leg_->depth[size_t(stage_)]++ == 0)
            leg_->stage_started[size_t(stage_)] = now_ns();
    }
    Span(Leg* leg, Stage stage, Clock::time_point start)
        : leg_(leg && !leg->terminal ? leg : nullptr), stage_(stage) {
        if (leg_ && leg_->depth[size_t(stage_)]++ == 0)
            leg_->stage_started[size_t(stage_)] =
                uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(start.time_since_epoch()).count());
    }
    Span(const Span&) = delete;
    ~Span() { stop(); }
    void stop(uint64_t ended = 0) {
        if (!leg_) return;
        const size_t i = size_t(stage_);
        if (!leg_->terminal && --leg_->depth[i] == 0) {
            if (!ended) ended = now_ns();
            leg_->add(stage_, ended >= leg_->stage_started[i] ? double(ended - leg_->stage_started[i]) / 1e6 : -1);
            leg_->stage_started[i] = 0;
        }
        leg_ = nullptr;
    }
};

struct Publication {
    Identity producer;
    uint64_t capture_ticket = 0, publication = 0, owner = 0, owner_generation = 0, epoch = 0;
    uint64_t captured_ns = 0, admitted_ns = 0;
    uint64_t applied_ns = 0, catalog_version = 0;
    uint64_t shared_prefix_tokens = 0;
};
struct PublicationEvent {
    Publication publication;
    const char* event = "terminal";
    const char* status = "unknown";
    uint64_t observed_ns = 0, head = 0;
    uint32_t attempts = 0;
};

// Only the native owner thread writes or drains receipts. Intermediate trace is
// expendable; publication terminals use independent storage drained after each
// publication maintenance operation, outside its ownership/store locks.
class Receipts {
public:
    static constexpr size_t max_trace = 128;
    static constexpr size_t max_line = 2048;
    // One flush can retire at most its initial 16 queued + 8 pending captures.
    // Capture (one refusal) and abandon (16 drops) are also followed by a drain.
    // The engine statically checks these bounds against the publication queue.
    static constexpr size_t max_publication_terminals = 24;
    using Report = void (*)(const char*);
private:
    std::array<std::array<char, max_line>, max_trace> trace_{};
    struct Terminal { PublicationEvent event; Identity executor; };
    std::array<Terminal, max_publication_terminals> terminals_{};
    size_t terminal_used_ = 0;
    uint64_t terminal_overflow_ = 0; // caller violated the per-operation drain contract
    size_t used_ = 0;
    uint64_t dropped_ = 0, truncated_ = 0;
    Report report_;
    struct Line {
        char bytes[max_line]{};
        size_t used = 0;
        bool truncated = false;
        void append(const char* format, ...) {
            if (truncated) return;
            va_list args;
            va_start(args, format);
            const int n = std::vsnprintf(bytes + used, sizeof bytes - used, format, args);
            va_end(args);
            if (n < 0 || size_t(n) >= sizeof bytes - used) { truncated = true; return; }
            used += size_t(n);
        }
        void identity(const Identity& id, const char* prefix = "") {
            if (id.valid()) append(" %srequest=%s %sleg=%llu %sincarnation=%s", prefix, id.request, prefix,
                                   (unsigned long long)id.leg, prefix, id.incarnation);
            else append(" %srequest=na %sleg=na %sincarnation=na", prefix, prefix, prefix);
        }
    };
    void emit(Line& line) {
        if (line.truncated) { increment(truncated_); return; }
        report_(line.bytes);
    }
public:
    explicit Receipts(Report report) : report_(report) {}
    uint64_t dropped() const { return dropped_; }
    uint64_t truncated() const { return truncated_; }
    size_t size() const { return used_ + terminal_used_; }
    size_t terminal_size() const { return terminal_used_; }
    uint64_t terminal_overflow() const { return terminal_overflow_; }
    void trace(std::string_view line) {
        if (line.size() >= max_line) { increment(truncated_); return; }
        if (used_ == max_trace) { increment(dropped_); return; }
        std::memcpy(trace_[used_].data(), line.data(), line.size());
        trace_[used_++][line.size()] = 0;
    }
    void drain() {
        for (size_t i = 0; i < used_; ++i) report_(trace_[i].data());
        used_ = 0;
        for (size_t i = 0; i < terminal_used_; ++i)
            publication_line(terminals_[i].event, terminals_[i].executor, true);
        terminal_used_ = 0;
    }
    void publication(const PublicationEvent& event, const Identity& executor) {
        if (std::strcmp(event.event, "terminal") == 0) {
            // Never use trace capacity (or trace_dropped) for terminal closure.
            if (terminal_used_ == terminals_.size()) { increment(terminal_overflow_); return; }
            terminals_[terminal_used_++] = {event, executor};
        } else publication_line(event, executor, false);
    }
private:
    void publication_line(const PublicationEvent& event, const Identity& executor, bool terminal) {
        const auto& p = event.publication;
        Line line;
        line.append("STRATA_PHASE v=1 kind=publication");
        line.identity(p.producer);
        line.append(" priority=%s event=%s status=%s semantics=asynchronous_host capture_ticket=%llu",
                    p.producer.priority_name(), event.event, event.status, (unsigned long long)p.capture_ticket);
        if (p.publication) line.append(" publication=%llu", (unsigned long long)p.publication);
        else line.append(" publication=na");
        line.append(" owner=%llu owner_generation=%llu epoch=%llu", (unsigned long long)p.owner,
                    (unsigned long long)p.owner_generation, (unsigned long long)p.epoch);
        line.append(" shared_prefix_tokens=%llu", (unsigned long long)p.shared_prefix_tokens);
        if (p.catalog_version) line.append(" catalog_version=%llu", (unsigned long long)p.catalog_version);
        else line.append(" catalog_version=na");
        if (event.head) line.append(" head=%llu", (unsigned long long)event.head);
        else line.append(" head=na");
        line.append(" attempts=%u observed_ns=%llu", event.attempts, (unsigned long long)event.observed_ns);
        line.identity(executor, "executor_");
        if (p.captured_ns && event.observed_ns >= p.captured_ns)
            line.append(" since_capture_ms=%.6f", double(event.observed_ns - p.captured_ns) / 1e6);
        if (p.admitted_ns && p.admitted_ns >= p.captured_ns)
            line.append(" queue_ms=%.6f", double(p.admitted_ns - p.captured_ns) / 1e6);
        if (p.admitted_ns && event.observed_ns >= p.admitted_ns)
            line.append(" admitted_to_observed_ms=%.6f", double(event.observed_ns - p.admitted_ns) / 1e6);
        // The store supplies the actual catalog-visibility boundary. Never
        // derive commit timing from terminal reconciliation or observation.
        if (p.admitted_ns && p.applied_ns >= p.admitted_ns)
            line.append(" writer_ms=%.6f", double(p.applied_ns - p.admitted_ns) / 1e6);
        else line.append(" writer_ms=na");
        if (p.applied_ns) line.append(" durable_ns=%llu", (unsigned long long)p.applied_ns);
        else line.append(" durable_ns=na");
        line.append(" trace_dropped=%llu trace_truncated=%llu terminal_overflow=%llu\n",
                    (unsigned long long)dropped_, (unsigned long long)truncated_,
                    (unsigned long long)terminal_overflow_);
        if (terminal) emit(line);
        else if (line.truncated) increment(truncated_);
        else trace(std::string_view(line.bytes, line.used));
    }
public:
    void request(Leg& leg, const char* status, const char* continuation = "none", uint64_t ended = 0) {
        if (leg.terminal) return;
        if (!ended) ended = now_ns();
        leg.finish(status, ended);
        drain();
        Line line;
        line.append("STRATA_PHASE v=1 kind=request event=terminal");
        line.identity(leg.input.identity);
        line.append(" identity=%s priority=%s",
                    leg.input.invalid ? "invalid" : leg.input.identity.valid() ? "bound" : "unavailable",
                    leg.input.identity.priority_name());
        if (leg.sequence) line.append(" sequence=%llu", (unsigned long long)leg.sequence);
        else line.append(" sequence=na");
        line.append(" slot=%d status=%s continuation=%s semantics=inclusive_host "
                    "materialize_semantics=asynchronous_store prefill_semantics=inclusive_run "
                    "started_ns=%llu ended_ns=%llu total_ms=%.6f aggregate_overflow=%llu",
                    leg.slot, status, continuation, (unsigned long long)leg.started_ns, (unsigned long long)ended,
                    double(ended - leg.started_ns) / 1e6, (unsigned long long)leg.aggregate_overflow);
        line.append(" shared_prefix_tokens=%llu shared_prefix_restored_tokens=%llu",
                    (unsigned long long)leg.shared_prefix_tokens,
                    (unsigned long long)leg.shared_prefix_restored_tokens);
        if (leg.shared_lookup_ns) {
            line.append(" shared_lookup_ns=%llu shared_lookup_catalog_version=%llu shared_lookup_matches=%llu "
                        "shared_lookup_eligible=%llu",
                        (unsigned long long)leg.shared_lookup_ns,
                        (unsigned long long)leg.shared_lookup_catalog_version,
                        (unsigned long long)leg.shared_lookup_matches,
                        (unsigned long long)leg.shared_lookup_eligible);
            if (leg.shared_lookup_head)
                line.append(" shared_lookup_head=%llu", (unsigned long long)leg.shared_lookup_head);
            else line.append(" shared_lookup_head=na");
        } else
            line.append(" shared_lookup_ns=na shared_lookup_catalog_version=na shared_lookup_matches=na "
                        "shared_lookup_eligible=na shared_lookup_head=na");
        line.append(" periodic_checkpoint_count=%llu periodic_checkpoint_last_tokens=%llu "
                    "periodic_checkpoint_last_ns=%llu", (unsigned long long)leg.periodic_checkpoint_count,
                    (unsigned long long)leg.periodic_checkpoint_last_tokens,
                    (unsigned long long)leg.periodic_checkpoint_last_ns);
        for (size_t i = 0; i < leg.totals.size(); ++i) {
            const auto& total = leg.totals[i];
            if (total.known) line.append(" %s_ms=%.6f %s_count=%llu", stage_names[i], total.ms,
                                        stage_names[i], (unsigned long long)total.count);
            else line.append(" %s_ms=na %s_count=na", stage_names[i], stage_names[i]);
        }
        if (leg.demand_request)
            line.append(" demand_request=%llu demand_generation=%llu", (unsigned long long)leg.demand_request,
                        (unsigned long long)leg.demand_generation);
        else line.append(" demand_request=na demand_generation=na");
        if (leg.demand_head) line.append(" demand_head=%llu", (unsigned long long)leg.demand_head);
        else line.append(" demand_head=na");
        line.append(" demand_status=%s", leg.demand_status);
        if (leg.demand_attempt_known) line.append(" demand_attempt=%llu", (unsigned long long)leg.demand_attempt);
        else line.append(" demand_attempt=na");
        line.append(" decode_run_ms=na trace_dropped=%llu trace_truncated=%llu\n",
                    (unsigned long long)dropped_, (unsigned long long)truncated_);
        emit(line);
    }
    void resident(const Identity& id, uint64_t slot_generation, int slot, const char* status,
                  uint64_t started_ns, uint64_t ended_ns) {
        drain();
        Line line;
        line.append("STRATA_PHASE v=1 kind=resident event=terminal");
        line.identity(id);
        line.append(" priority=%s slot=%d slot_generation=%llu status=%s semantics=inclusive_host "
                    "started_ns=%llu ended_ns=%llu decode_residence_ms=%.6f "
                    "decode_residence_count=1 decode_run_ms=na "
                    "trace_dropped=%llu trace_truncated=%llu\n", id.priority_name(), slot,
                    (unsigned long long)slot_generation, status, (unsigned long long)started_ns,
                    (unsigned long long)ended_ns, double(ended_ns - started_ns) / 1e6,
                    (unsigned long long)dropped_, (unsigned long long)truncated_);
        emit(line);
    }
};
} // namespace strata::program::phase
