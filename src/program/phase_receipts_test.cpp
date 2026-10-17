// Deterministic CPU-only framing/identity/terminal regressions; no model or GPU.
#include "strata/program/phase_receipts.hpp"

#include <algorithm>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace phase = strata::program::phase;
namespace {
int failures = 0;
std::vector<std::string> lines;
void report(const char* text) { lines.emplace_back(text); }
#define CHECK(condition) do { if (!(condition)) { ++failures; std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition); } } while (0)
const char* incarnation = "0123456789abcdef0123456789abcdef";
std::string keys(const std::string& request, const std::string& leg = "1") {
    return " phase_request=" + request + " phase_leg=" + leg + " phase_incarnation=" + incarnation;
}
}

int main() {
    const auto first = phase::parse_input("BGEN 0 8" + keys("http-one") + " phase_priority=foreground 1,2,3");
    CHECK(!first.invalid && first.identity.valid());
    CHECK(std::string(first.identity.request) == "http-one" && first.identity.leg == 1);
    CHECK(first.identity.priority == phase::Identity::Priority::foreground);
    const auto successor = phase::parse_input("BGEN 0 8" + keys("http-two", "2") + " 1,2,3");
    CHECK(successor.identity.valid() && successor.identity.leg == 2);
    CHECK(std::string(first.identity.request) == "http-one"); // slot reuse cannot mutate the former producer
    const auto image = phase::parse_input("GENI 8" + keys("image") + " /tmp/input 1,2,3");
    CHECK(image.identity.valid());
    const auto legacy = phase::parse_input("GEN 8 cache_offer=9 cache_generation=3 temperature=0 1,2,3");
    CHECK(!legacy.invalid && !legacy.identity.valid() && legacy.seen == 0);
    const auto future = phase::parse_input("GEN 8 future_optional=99" + keys("future") + " 1,2,3");
    CHECK(!future.invalid && future.identity.valid());
    for (const std::string& invalid : std::vector<std::string>{
            "GEN 8 phase_request=partial 1,2", "GEN 8 phase_priority=foreground 1,2",
            "GEN 8" + keys("bad/id") + " 1,2", "GEN 8" + keys(std::string(97, 'x')) + " 1,2",
            "GEN 8" + keys("x", "0") + " 1,2", "GEN 8" + keys("x", "01") + " 1,2",
            "GEN 8" + keys("x", "-1") + " 1,2", "GEN 8" + keys("x", "+1") + " 1,2",
            "GEN 8" + keys("x", "18446744073709551616") + " 1,2",
            "GEN 8" + keys("x", "1x") + " 1,2", "GEN 8" + keys("x") + " phase_request=x 1,2",
            "GEN 8" + keys("x") + " phase_leg=1 1,2", "GEN 8" + keys("x") + " phase_priority=urgent 1,2",
            std::string("GEN 8 phase_request=x phase_leg=1 phase_incarnation=ABCDEF0123456789abcdef0123456789 1,2")}) {
        const auto parsed = phase::parse_input(invalid);
        CHECK(parsed.invalid && !parsed.identity.valid());
    }
    uint64_t number = 0;
    CHECK(phase::positive_u64("18446744073709551615", number) && number == UINT64_MAX);
    uint64_t sequence = UINT64_MAX - 1, next = 0;
    CHECK(phase::next_sequence(sequence, next) && next == UINT64_MAX);
    next = 0;
    CHECK(!phase::next_sequence(sequence, next) && next == 0 && sequence == UINT64_MAX);

    auto receipts = std::make_unique<phase::Receipts>(&report);
    phase::Leg cancelled(first, 1, 1000000);
    cancelled.add(phase::Stage::prefill_run, 12.5, 3);
    cancelled.add(phase::Stage::store_seam, 0.125, 2);
    cancelled.shared_lookup_ns = 4000000;
    cancelled.shared_lookup_catalog_version = 7;
    cancelled.shared_lookup_matches = 0;
    cancelled.shared_lookup_eligible = 0;
    for (size_t i = 0; i < phase::Receipts::max_trace + 7; ++i) receipts->trace("bounded trace\n");
    receipts->trace(std::string(phase::Receipts::max_line, 'x'));
    CHECK(receipts->size() == phase::Receipts::max_trace);
    CHECK(receipts->dropped() == 7 && receipts->truncated() == 1);
    receipts->request(cancelled, "cancelled", "none", 26000000);
    CHECK(lines.size() == phase::Receipts::max_trace + 1 && receipts->size() == 0);
    const auto terminal = lines.back();
    CHECK(terminal.find("kind=request event=terminal request=http-one leg=1") != terminal.npos);
    CHECK(terminal.find("status=cancelled") != terminal.npos && terminal.find("total_ms=25.000000") != terminal.npos);
    CHECK(terminal.find("prefill_run_ms=12.500000 prefill_run_count=3") != terminal.npos);
    CHECK(terminal.find("demand_materialize_ms=na demand_materialize_count=na") != terminal.npos);
    CHECK(terminal.find("demand_request=na demand_generation=na demand_head=na") != terminal.npos);
    CHECK(terminal.find("shared_lookup_ns=4000000 shared_lookup_catalog_version=7 shared_lookup_matches=0 "
                        "shared_lookup_eligible=0 shared_lookup_head=na") != terminal.npos);
    CHECK(terminal.find("trace_dropped=7 trace_truncated=1") != terminal.npos);
    CHECK(terminal.size() < phase::Receipts::max_line && std::count(terminal.begin(), terminal.end(), '\n') == 1);
    const size_t emitted = lines.size();
    receipts->request(cancelled, "completed", "none", 27000000);
    CHECK(lines.size() == emitted); // cancellation/error/completion converge on one terminal
    phase::Leg error(successor, 2, 1000000);
    receipts->request(error, "error", "none", 2000000);
    CHECK(lines.back().find("request=http-two leg=2") != lines.back().npos);
    CHECK(lines.back().find("status=error") != lines.back().npos);
    CHECK(lines.back().find("shared_lookup_ns=na shared_lookup_catalog_version=na shared_lookup_matches=na "
                            "shared_lookup_eligible=na shared_lookup_head=na") != lines.back().npos);
    phase::Leg old(legacy, 3, 1000000);
    receipts->request(old, "rejected", "none", 2000000);
    CHECK(lines.back().find("request=na leg=na incarnation=na") != lines.back().npos);
    CHECK(lines.back().find("identity=unavailable") != lines.back().npos);
    phase::Leg count_bound(first, 4, 1000000);
    count_bound.add(phase::Stage::prefill_run, 1, UINT64_MAX);
    count_bound.add(phase::Stage::prefill_run, 1, 1);
    CHECK(count_bound.totals[size_t(phase::Stage::prefill_run)].count == UINT64_MAX);
    CHECK(count_bound.aggregate_overflow == 1);
    CHECK(!count_bound.totals[size_t(phase::Stage::prefill_run)].known);
    count_bound.add(phase::Stage::prefill_run, 1);
    CHECK(count_bound.aggregate_overflow == 1);
    {
        // Nested maintenance contributes one inclusive span, not two durations.
        phase::Span outer(&count_bound, phase::Stage::store_seam, phase::Clock::time_point(std::chrono::nanoseconds(100)));
        phase::Span inner(&count_bound, phase::Stage::store_seam, phase::Clock::time_point(std::chrono::nanoseconds(200)));
        inner.stop(300);
        outer.stop(500);
    }
    CHECK(count_bound.totals[size_t(phase::Stage::store_seam)].count == 1);
    CHECK(count_bound.totals[size_t(phase::Stage::store_seam)].ms == 0.0004);
    { phase::Span disabled(nullptr, phase::Stage::prefill_run); } // gate-off span does not dereference or read a clock
    count_bound.unavailable(phase::Stage::decode_interleave);
    count_bound.add(phase::Stage::decode_interleave, 10);
    CHECK(!count_bound.totals[size_t(phase::Stage::decode_interleave)].known);
    receipts->request(count_bound, "completed", "none", 2000000);
    CHECK(lines.back().find("aggregate_overflow=1") != lines.back().npos);
    CHECK(lines.back().find("prefill_run_ms=na prefill_run_count=na") != lines.back().npos);

    // ERR must carry work still inside live scopes, with one terminal cutoff.
    // Unwinding or explicitly stopping those scopes cannot mutate the closed leg.
    phase::Leg mounted_error(first, 10, 1000000);
    mounted_error.unavailable(phase::Stage::decode_interleave);
    {
        phase::Span mount(&mounted_error, phase::Stage::gpu_mount,
                          phase::Clock::time_point(std::chrono::nanoseconds(1000000)));
        phase::Span nested_mount(&mounted_error, phase::Stage::gpu_mount,
                                 phase::Clock::time_point(std::chrono::nanoseconds(2000000)));
        phase::Span copy(&mounted_error, phase::Stage::resident_copyin,
                         phase::Clock::time_point(std::chrono::nanoseconds(3000000)));
        phase::Span decode(&mounted_error, phase::Stage::decode_interleave,
                           phase::Clock::time_point(std::chrono::nanoseconds(4000000)));
        receipts->request(mounted_error, "error", "none", 6000000);
        const auto& closed = lines.back();
        CHECK(closed.find("total_ms=5.000000") != closed.npos);
        CHECK(closed.find("gpu_mount_ms=5.000000 gpu_mount_count=1") != closed.npos);
        CHECK(closed.find("resident_copyin_ms=3.000000 resident_copyin_count=1") != closed.npos);
        CHECK(closed.find("decode_interleave_ms=na decode_interleave_count=na") != closed.npos);
        mount.stop(7000000); // explicit outer stop before its still-live nested scope
        copy.stop(8000000);
    }
    CHECK(mounted_error.totals[size_t(phase::Stage::gpu_mount)].ms == 5.0);
    CHECK(mounted_error.totals[size_t(phase::Stage::gpu_mount)].count == 1);
    CHECK(mounted_error.totals[size_t(phase::Stage::resident_copyin)].ms == 3.0);
    CHECK(mounted_error.totals[size_t(phase::Stage::decode_interleave)].ms == 2.0);
    CHECK(!mounted_error.totals[size_t(phase::Stage::decode_interleave)].known);
    const auto closed_lines = lines.size();
    receipts->request(mounted_error, "completed", "none", 9000000);
    CHECK(lines.size() == closed_lines);
    {
        phase::Span after_terminal(&mounted_error, phase::Stage::gpu_mount,
                                   phase::Clock::time_point(std::chrono::nanoseconds(9000000)));
        after_terminal.stop(10000000);
    }
    CHECK(mounted_error.totals[size_t(phase::Stage::gpu_mount)].ms == 5.0);
    CHECK(mounted_error.totals[size_t(phase::Stage::gpu_mount)].count == 1);

    // Shared -> private -> RAM fallback must never pair a new request/generation
    // with an old attempt/head, including an early refusal before admission.
    phase::Leg fallback(first, 5, 1000000);
    fallback.begin_demand(41, 7);
    fallback.observe_demand(101, 201, true, false);
    fallback.finish_demand("corrupt");
    fallback.begin_demand(42, 8);
    CHECK(fallback.demand_request == 42 && fallback.demand_generation == 8);
    CHECK(!fallback.demand_attempt_known && !fallback.demand_attempt && !fallback.demand_head);
    fallback.finish_demand("refused_budget");
    receipts->request(fallback, "completed", "none", 2000000);
    CHECK(lines.back().find("demand_request=42 demand_generation=8 demand_head=na") != lines.back().npos);
    CHECK(lines.back().find("demand_status=refused_budget demand_attempt=na") != lines.back().npos);
    phase::Leg early(first, 6, 1000000);
    early.begin_demand(43, 9);
    early.observe_demand(102, 202, true, false);
    early.materialized_demand(3.0, 202);
    early.finish_demand("ready");
    early.begin_demand(); // disabled/deadline/cancellation before a query exists
    early.finish_demand("cancelled");
    receipts->request(early, "completed", "none", 2000000);
    CHECK(lines.back().find("demand_request=na demand_generation=na demand_head=na") != lines.back().npos);
    CHECK(lines.back().find("demand_status=cancelled demand_attempt=na") != lines.back().npos);

    // A measured successor cannot turn a partial aggregate back into known.
    phase::Leg mixed_fallback(first, 9, 1000000);
    mixed_fallback.begin_demand(41, 7);
    mixed_fallback.observe_demand(101, 201, true, false);
    mixed_fallback.finish_demand("corrupt");
    mixed_fallback.begin_demand(44, 10);
    mixed_fallback.observe_demand(103, 203, true, false);
    mixed_fallback.materialized_demand(4.0, 203);
    mixed_fallback.finish_demand("ready");
    const auto& mixed = mixed_fallback.totals[size_t(phase::Stage::demand_materialize)];
    CHECK(mixed.ms == 4.0 && mixed.count == 1 && mixed.unavailable && !mixed.known);
    receipts->request(mixed_fallback, "completed", "none", 2000000);
    CHECK(lines.back().find("demand_materialize_ms=na demand_materialize_count=na") != lines.back().npos);
    CHECK(lines.back().find("demand_request=44 demand_generation=10 demand_head=203") != lines.back().npos);
    CHECK(lines.back().find("demand_status=ready demand_attempt=103") != lines.back().npos);
    phase::Leg retry(first, 7, 1000000);
    retry.begin_demand(45, 11);
    // A failed prior attempt may have finished/reissued entirely between polls.
    retry.observe_demand(105, 205, true, true);
    retry.materialized_demand(5.0, 205);
    retry.finish_demand("ready");
    CHECK(!retry.totals[size_t(phase::Stage::demand_materialize)].known);
    phase::Leg unstarted(first, 8, 1000000);
    unstarted.begin_demand(46, 12);
    unstarted.observe_demand(106, 206, false, false); // queued then refused: no materialization
    unstarted.finish_demand("refused_budget");
    unstarted.begin_demand(47, 13);
    unstarted.observe_demand(107, 207, true, false);
    unstarted.materialized_demand(6.0, 207);
    unstarted.finish_demand("ready");
    const auto& complete = unstarted.totals[size_t(phase::Stage::demand_materialize)];
    CHECK(complete.known && !complete.unavailable && complete.ms == 6.0 && complete.count == 1);

    phase::Publication publication;
    publication.producer = first.identity;
    publication.capture_ticket = 19;
    publication.owner = 1;
    publication.owner_generation = 23;
    publication.epoch = 2;
    publication.captured_ns = 1000000;
    publication.publication = 29;
    publication.admitted_ns = 3000000;
    receipts->publication({publication, "terminal", "stored", 8000000, 31, 0}, successor.identity);
    CHECK(receipts->size() == 1 && lines.back().find("kind=publication") == lines.back().npos);
    receipts->drain();
    const auto delayed = lines.back();
    CHECK(delayed.find("kind=publication request=http-one leg=1") != delayed.npos);
    CHECK(delayed.find("executor_request=http-two executor_leg=2") != delayed.npos);
    CHECK(delayed.find("queue_ms=2.000000") != delayed.npos);
    CHECK(delayed.find("admitted_to_observed_ms=5.000000") != delayed.npos);
    CHECK(delayed.find("writer_ms=na durable_ns=na") != delayed.npos);
    publication.capture_ticket = 20;
    publication.publication = 30;
    publication.applied_ns = 6000000;
    publication.catalog_version = 4;
    receipts->publication({publication, "terminal", "stored", 8000000, 32, 0}, successor.identity);
    receipts->drain();
    CHECK(lines.back().find("writer_ms=3.000000 durable_ns=6000000") != lines.back().npos);
    CHECK(lines.back().find("catalog_version=4") != lines.back().npos);

    // A full, lossy trace cannot consume a publication terminal reservation.
    lines.clear();
    for (size_t i = 0; i < phase::Receipts::max_trace; ++i) receipts->trace("expendable\n");
    const auto dropped_before = receipts->dropped();
    for (size_t i = 0; i < phase::Receipts::max_publication_terminals; ++i) {
        publication.capture_ticket = 100 + i;
        publication.publication = 200 + i;
        publication.producer = phase::parse_input("GEN 1" + keys("publisher-" + std::to_string(i)) + " 1,2").identity;
        receipts->publication({publication, "handoff", "refused", 8000000, 0, 7}, successor.identity);
        receipts->publication({publication, "terminal", "dropped", 9000000, 0, 8}, successor.identity);
    }
    CHECK(lines.empty()); // observer performs no I/O, including on saturation
    CHECK(receipts->terminal_size() == phase::Receipts::max_publication_terminals);
    CHECK(receipts->terminal_overflow() == 0);
    CHECK(receipts->dropped() == dropped_before + phase::Receipts::max_publication_terminals);
    receipts->drain();
    CHECK(receipts->size() == 0);
    CHECK(lines.size() == phase::Receipts::max_trace + phase::Receipts::max_publication_terminals);
    for (size_t i = 0; i < phase::Receipts::max_publication_terminals; ++i) {
        const auto& closed = lines[phase::Receipts::max_trace + i];
        CHECK(closed.find("kind=publication request=publisher-" + std::to_string(i) + " leg=1 ") != closed.npos);
        CHECK(closed.find("event=terminal status=dropped") != closed.npos);
        CHECK(closed.find("capture_ticket=" + std::to_string(100 + i) + " publication=" + std::to_string(200 + i) + " ") != closed.npos);
        CHECK(closed.find("attempts=8 ") != closed.npos);
        CHECK(closed.find("executor_request=http-two executor_leg=2 ") != closed.npos);
        CHECK(closed.find("terminal_overflow=0") != closed.npos);
    }

    if (!failures) std::puts("phase_receipts_test: ok (validated optional keys, identity reuse, terminal closure, overflow, nested and asynchronous spans)");
    return failures ? 1 : 0;
}
