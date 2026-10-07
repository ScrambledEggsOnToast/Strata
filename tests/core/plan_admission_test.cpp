// tests/core/plan_admission_test.cpp - HET-017 boundary tests for the admission arithmetic.
//
// Host-only: include/strata/plan/admission.hpp must decide with no GPU attached, because the decision the
// engine enforces pre-allocation is exactly the one these tests replay.  Each test is an acceptance
// boundary named in the ticket:
//
//   AC-1  one overfull device refuses even when the aggregate has slack  (no aggregate test exists)
//   AC-2  the guest allocation is a real limit; the reserves cannot be flagged away
//   AC-4  file-backed demand is counted; only the bounded clean-cache credit discounts it
//   +     unknown classes refuse; missing telemetry refuses; hypothetical devices never admit;
//         the explain-or-shrink rule blocks enlargement on an unexplained overshoot.
#include "strata/plan/admission.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace strata::plan;

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

constexpr uint64_t GiB = 1ull << 30;

DeviceTelemetry measured(const std::string& name, uint64_t total) {
    DeviceTelemetry t;
    t.name = name;
    t.measured = true;
    t.total_bytes = total;
    t.free_bytes = total;
    return t;
}

GuestHostPolicy both_scopes_measured(uint64_t guest_total, uint64_t host_available) {
    GuestHostPolicy p;
    p.guest_total_measured = true;
    p.guest_total_bytes = guest_total;
    p.host_available_measured = true;
    p.host_available_bytes = host_available;
    return p;
}

// AC-1, the headline case: CUDA1 alone is overfull while the AGGREGATE of the two cards has room.  The
// planner must refuse, and name the device and the class - a warning would be a plan the engine cannot
// honour.
void test_single_overfull_device_refuses_despite_aggregate() {
    DeviceCost a;
    a.name = "CUDA0";
    a.vram.add("weights_canonical", 1 * GiB, "test");
    a.headroom_bytes = 1 * GiB;
    DeviceCost b;
    b.name = "CUDA1";
    b.vram.add("weights_canonical", 20 * GiB, "test");
    b.vram.add("expert_cache", 10 * GiB, "test");
    b.headroom_bytes = 1 * GiB;
    // aggregate demand 32 GiB + 2 GiB headroom against an aggregate ceiling of 48 GiB: plenty.  CUDA1
    // alone holds 31 GiB on a 24 GiB card: refused.
    std::vector<DeviceTelemetry> tel = {measured("CUDA0", 24 * GiB), measured("CUDA1", 24 * GiB)};
    const AdmissionDecision d = admit_configuration({a, b}, tel, {}, {}, both_scopes_measured(0, 0));
    require(!d.admitted, "an overfull CUDA1 was admitted because the aggregate fit");
    require(d.limiting_term.rfind("CUDA1:", 0) == 0, "limiting term did not name CUDA1: " + d.limiting_term);
    require(d.devices[(size_t) 0].fits, "CUDA0 was dragged down with CUDA1");
    require(!d.devices[(size_t) 1].fits, "CUDA1 overfull was not reported");
    require(d.devices[(size_t) 1].limiting_class == "expert_cache",
            "limiting class was not the class that crossed the ceiling: " + d.devices[(size_t) 1].limiting_class);

    // The headroom is not decoration: without it CUDA1 would close.  Shrink its demand back under the
    // ceiling-minus-headroom and the same configuration admits.
    b.vram.classes.clear();
    b.vram.add("weights_canonical", 20 * GiB, "test");
    const AdmissionDecision ok = admit_configuration({a, b}, tel, {}, {}, both_scopes_measured(0, 0));
    require(ok.admitted, "a fitting configuration was refused: " +
                             (ok.reasons.empty() ? "" : ok.reasons.front()));
}

// AC-2: the guest allocation is a real limit, and the design-default reserve survives any policy input:
// clamp_reserves floors the guest at 4 GiB and the host at 8 GiB, so a caller cannot plan all physical
// RAM or the whole VM allocation.
void test_guest_is_a_real_limit_and_reserves_cannot_be_planned_away() {
    DeviceCost a;
    a.name = "CUDA0";
    a.vram.add("weights_canonical", 2 * GiB, "test");
    Scope guest;
    guest.add("conversation_parked", 6 * GiB, "test");   // anon demand, no file credit applies
    // 8 GiB guest, 4 GiB floor reserve -> 4 GiB usable < 6 GiB demanded: refuse, limiting term "guest".
    const AdmissionDecision refused =
        admit_configuration({a}, {measured("CUDA0", 24 * GiB)}, guest, {}, both_scopes_measured(8 * GiB, 0));
    require(!refused.admitted, "guest demand above the reserved envelope was admitted");
    require(refused.limiting_term == "guest", "limiting term was not the guest: " + refused.limiting_term);

    // The same demand fits once the guest really has the room (16 GiB - 4 GiB reserve = 12 GiB usable).
    const AdmissionDecision fits =
        admit_configuration({a}, {measured("CUDA0", 24 * GiB)}, guest, {}, both_scopes_measured(16 * GiB, 0));
    require(fits.admitted && fits.guest_fits, "guest demand under the reserved envelope was refused");

    // A policy that hands in zero reserves is floored back to the design defaults (AC-2's "default policy
    // does not allocate all physical host RAM").
    GuestHostPolicy greedy = both_scopes_measured(8 * GiB, 0);
    greedy.guest_reserve_bytes = 0;
    greedy.host_reserve_bytes = 0;
    const AdmissionDecision clamped =
        admit_configuration({a}, {measured("CUDA0", 24 * GiB)}, guest, {}, greedy);
    require(!clamped.admitted, "a zero-reserve policy planned the whole guest");
    require(clamped.applied_guest_reserve_bytes == 4 * GiB, "guest reserve was not floored at 4 GiB");
    require(clamped.applied_host_reserve_bytes == 8 * GiB, "host reserve was not floored at 8 GiB");
}

// AC-2, telemetry half: an unmeasured guest (or host, or device) refuses.  Unknown is not zero, and a
// guard that never fires because telemetry is absent is exactly the false-pass the convention forbids.
void test_missing_telemetry_fails_closed() {
    DeviceCost a;
    a.name = "CUDA0";
    a.vram.add("weights_canonical", 2 * GiB, "test");
    Scope guest;
    guest.add("conversation_parked", 1 * GiB, "test");
    GuestHostPolicy unmeasured_guest = both_scopes_measured(0, 64 * GiB);
    unmeasured_guest.guest_total_measured = false;
    const AdmissionDecision d = admit_configuration({a}, {measured("CUDA0", 24 * GiB)}, guest, {}, unmeasured_guest);
    require(!d.admitted, "admission succeeded with an unmeasured guest allocation");
    require(!d.verified, "decision claims verification without guest telemetry");
    require(d.limiting_term == "telemetry:guest", "limiting term was not telemetry:guest: " + d.limiting_term);

    const AdmissionDecision no_device =
        admit_configuration({a}, {}, {}, {}, both_scopes_measured(0, 0));
    require(!no_device.admitted && no_device.limiting_term == "telemetry:CUDA0",
            "admission succeeded with no device telemetry");
}



// Unknown allocations are explicit admission blockers, never zeros: an `unknown` class refuses whatever
// the arithmetic says, and the refusal names the class.
void test_unknown_class_blocks_admission() {
    DeviceCost a;
    a.name = "CUDA0";
    a.vram.add("weights_canonical", 1 * GiB, "test");
    a.vram.add_unknown("vision_buffers", "unmeasured new path");
    const AdmissionDecision d =
        admit_configuration({a}, {measured("CUDA0", 24 * GiB)}, {}, {}, both_scopes_measured(64 * GiB, 64 * GiB));
    require(!d.admitted, "an unaccounted allocation was admitted");
    require(!d.unknown_classes.empty() && d.unknown_classes.front() == "CUDA0:vision_buffers",
            "the unknown class was not named");
    require(!d.verified, "unknown classes must unverify the decision");
}

// AC-4: file-backed is not free.  File classes count in full by default; the ONLY discount is the bounded
// clean-cache credit, capped at the file bytes themselves - shared pages are never credited.
void test_file_backed_demand_is_counted_and_credit_is_bounded() {
    DeviceCost a;
    a.name = "CUDA0";
    a.vram.add("weights_canonical", 2 * GiB, "test");
    Scope guest;
    guest.add("file:ple_table_pages", 20 * GiB, "mmap of the PLE shard", ByteMeaning::clean_file);
    guest.add("conversation_parked", 3 * GiB, "test");
    // Usable guest = 24 GiB - 4 GiB reserve = 20 GiB.  No credit: 23 GiB demanded -> refuse, with the
    // file pages counted in full.
    GuestHostPolicy no_credit = both_scopes_measured(24 * GiB, 0);
    const AdmissionDecision counted =
        admit_configuration({a}, {measured("CUDA0", 24 * GiB)}, guest, {}, no_credit);
    require(!counted.admitted, "file-backed guest demand was treated as free");
    require(counted.guest_demand_bytes == 23 * GiB, "file-backed bytes were not counted in full");

    // Bounded credit: 2 GiB of measured clean cache discounts exactly 2 GiB - still over.
    GuestHostPolicy some = no_credit;
    some.file_cache_credit_bytes = 2 * GiB;
    const AdmissionDecision partial =
        admit_configuration({a}, {measured("CUDA0", 24 * GiB)}, guest, {}, some);
    require(!partial.admitted && partial.guest_demand_bytes == 21 * GiB,
            "clean-cache credit did not apply boundedly");

    // Credit above the file bytes caps at them: 40 GiB claimed credit discounts the 20 GiB of files
    // exactly once - never the anon bytes, and never into a negative.
    GuestHostPolicy greedy = no_credit;
    greedy.file_cache_credit_bytes = 40 * GiB;
    const AdmissionDecision capped =
        admit_configuration({a}, {measured("CUDA0", 24 * GiB)}, guest, {}, greedy);
    require(capped.guest_demand_bytes == 3 * GiB, "credit was not capped at the file bytes");

    // With the whole file set measured clean (20 GiB), the same configuration closes.
    GuestHostPolicy full = no_credit;
    full.file_cache_credit_bytes = 20 * GiB;
    const AdmissionDecision closes =
        admit_configuration({a}, {measured("CUDA0", 24 * GiB)}, guest, {}, full);
    require(closes.admitted && closes.guest_demand_bytes == 3 * GiB,
            "a fully-clean file set still refused: " +
                (closes.reasons.empty() ? "" : closes.reasons.front()));
}

// A MIXED plan - a real, measured device plus labelled future rows - is decided by the real device
// alone; the hypothetical rows travel as labels, not as refusals.
void test_mixed_real_and_hypothetical_is_decided_by_the_real_device() {
    DeviceCost real;
    real.name = "CUDA0";
    real.vram.add("weights_canonical", 2 * GiB, "test");
    DeviceCost future;
    future.name = "V100-32G";
    future.hypothetical = true;
    future.vram.add("weights_canonical", 30 * GiB, "test");
    const AdmissionDecision d =
        admit_configuration({real, future}, {measured("CUDA0", 24 * GiB)}, {}, {}, both_scopes_measured(0, 0));
    require(d.admitted, "a fitting real device was refused because a labelled future row was present");
    require(d.devices.size() == 2 && d.devices[(size_t) 1].hypothetical,
            "the hypothetical row lost its label in a mixed plan");
    require(!d.devices[(size_t) 1].fits, "a hypothetical row reported fits");
}

// Hypothetical devices (the future 24/32/32 GiB rows) are labelled, reported, never admitted, and never
// lend slack to a real device.
void test_hypothetical_devices_are_labelled_and_never_admitted() {
    DeviceCost future;
    future.name = "V100-24G";
    future.hypothetical = true;
    future.vram.add("weights_canonical", 2 * GiB, "test");
    const AdmissionDecision alone = admit_configuration({future}, {}, {}, {}, both_scopes_measured(0, 0));
    require(!alone.admitted, "a hypothetical-only plan was admitted");
    require(!alone.verified, "a hypothetical-only plan claims verification");
    require(alone.devices.size() == 1 && alone.devices[(size_t) 0].hypothetical,
            "the hypothetical device lost its label");
}

// Per-request admission goes by the request's total state demand and context length, not a slot count.
void test_request_admission_by_state_demand() {
    Geometry g;
    RequestDemand req;
    req.prompt_tokens = 3900;
    req.max_new_tokens = 90;
    req.kv_bytes_per_token = kv_bytes_per_token(g);
    req.state_bytes_per_sequence = state_bytes(g);
    const RequestDecision ok = admit_request(req, 4096);
    require(ok.admitted, "a fitting request was refused");
    require(ok.state_demand_bytes ==
                state_bytes(g) + 3990ull * kv_bytes_per_token(g),
            "request state demand arithmetic is wrong");
    require(admit_request(req, 3990).reason.find("exceeds the context") != std::string::npos,
            "an over-context request did not name the context in its refusal");
}

// AC-3: the explain-or-shrink rule.  An overshoot within tolerance is fine; beyond tolerance it blocks
// enlargement until an explanation is recorded.
void test_explain_or_shrink_comparison() {
    GuestHostPolicy p;
    p.comparison_tolerance_bytes = 256ull << 20;
    const PeakComparison within = compare_peaks(10 * GiB, 10 * GiB + (200ull << 20), p);
    require(within.within_tolerance && within.enlargement_allowed,
            "an in-tolerance overshoot blocked enlargement");
    const PeakComparison unexplained = compare_peaks(10 * GiB, 11 * GiB, p);
    require(!unexplained.explained && !unexplained.enlargement_allowed,
            "an unexplained overshoot allowed enlargement");
    require(unexplained.explanation.find("resolve") != std::string::npos,
            "the unexplained verdict does not say what to do");
    const PeakComparison explained = compare_peaks(10 * GiB, 11 * GiB, p,
                                                   "fragmentation measured at 1.0 GiB");
    require(explained.explained && explained.enlargement_allowed,
            "an explained overshoot still blocked enlargement");
}

void test_mps_physical_and_residual_gates_are_independent() {
    DeviceCost device;
    device.vram.add("payload", 80, "boundary");
    auto t = measured("CUDA0", 100);
    t.free_bytes = 80;
    t.cap_declared = true;
    t.enforced_cap_bytes = 90;
    t.physical_free_measured = true;
    t.physical_free_bytes = 100;
    t.outside_client_allowance_bytes = 10;
    const auto decide = [&]() { return admit_configuration({device}, {t}, {}, {}, {}); };
    auto d = decide();
    require(d.admitted && d.verified && d.devices[0].ceiling_bytes == 80,
            "MPS equality boundary must admit against residual, not physical capacity");
    t.physical_free_bytes = 200;
    d = decide();
    require(d.admitted && d.devices[0].ceiling_bytes == 80 && d.devices[0].slack_bytes == 0,
            "extra physical headroom changed client budget");
    t.physical_free_bytes = 99;
    d = decide();
    require(!d.admitted && d.verified && !d.devices[0].fits,
            "one-byte physical overflow must be a measured refusal");
    t.free_bytes = 79;
    d = decide();
    require(d.reasons.size() == 2 && d.limiting_term == "CUDA0:payload",
            "physical refusal hid independent client-demand refusal");
    t.physical_free_bytes = 100;
    require(!decide().admitted, "physical capacity rescued an overfull client");
    t.free_bytes = 80;
    t.physical_free_measured = false;
    d = decide();
    require(!d.admitted && !d.verified && d.devices[0].cap_declared &&
            d.devices[0].enforced_cap_bytes == 90 && d.devices[0].physical_free_bytes == 100 &&
            d.devices[0].outside_client_allowance_bytes == 10,
            "missing physical measurement was verified or lost declaration identity");
    t.physical_free_measured = true;
    t.outside_client_allowance_bytes = 0;
    require(!decide().admitted && !decide().verified, "missing outside allowance admitted");
    t.outside_client_allowance_bytes = 10;
    t.enforced_cap_bytes = 0;
    require(!decide().admitted && !decide().verified, "declared zero cap admitted");
    t.enforced_cap_bytes = 79;
    require(!decide().admitted && !decide().verified, "residual above cap admitted");
    t.enforced_cap_bytes = 90;
    t.measured = false;
    d = decide();
    require(!d.verified && d.devices[0].cap_declared && d.devices[0].physical_free_measured,
            "unmeasured client telemetry lost physical identity");
    t.measured = true;
    t.total_bytes = UINT64_MAX;
    t.enforced_cap_bytes = UINT64_MAX;
    t.physical_free_bytes = UINT64_MAX;
    t.outside_client_allowance_bytes = 1;
    require(!decide().admitted, "overflowing full cap plus allowance admitted");
    t.enforced_cap_bytes = UINT64_MAX - 1;
    require(decide().admitted, "UINT64 equality physical envelope refused");
    t.cap_declared = false;
    t.physical_free_measured = false;
    t.physical_free_bytes = 0;
    t.outside_client_allowance_bytes = 0;
    require(decide().admitted, "non-MPS unexpectedly required physical telemetry");
}

void test_overflow_refuses_in_every_scope_and_request() {
    const uint64_t largest = UINT64_MAX;
    DeviceCost device;
    device.vram.add("first", largest, "boundary");
    device.vram.add("second", 1, "boundary");
    const auto telemetry = measured("CUDA0", largest);
    const auto policy = both_scopes_measured(largest, largest);
    require(!admit_configuration({device}, {telemetry}, {}, {}, policy).admitted,
            "wrapped device total admitted");
    device.vram.classes.pop_back();
    device.headroom_bytes = 1;
    require(!admit_configuration({device}, {telemetry}, {}, {}, policy).admitted,
            "wrapped device demand plus headroom admitted");
    device.vram.classes.clear();
    device.headroom_bytes = 0;
    Scope ram;
    ram.add("file:first", largest, "boundary", ByteMeaning::clean_file);
    ram.add("file:second", 1, "boundary", ByteMeaning::clean_file);
    auto credited = policy;
    credited.file_cache_credit_bytes = largest;
    require(!admit_configuration({device}, {telemetry}, ram, {}, credited).admitted,
            "overflowing guest total admitted after file credit");
    require(!admit_configuration({device}, {telemetry}, {}, ram, credited).admitted,
            "overflowing host total admitted after file credit");
    RequestDemand request;
    request.prompt_tokens = INT64_MAX;
    request.max_new_tokens = 1;
    require(!admit_request(request, INT64_MAX).admitted, "wrapped request context admitted");
    request.prompt_tokens = 1;
    request.max_new_tokens = 1;
    request.kv_bytes_per_token = largest;
    require(!admit_request(request, 100).admitted, "wrapped request state admitted");
    request.prompt_tokens = -1;
    require(!admit_request(request, 100).admitted, "negative prompt admitted");
}

void test_declared_meaning_projects_unknowns_and_bounds_credit() {
    Scope guest, host;
    guest.add("renamed-clean", 100, "fixture", ByteMeaning::clean_file);
    guest.add("file:locked", 40, "fixture", ByteMeaning::locked_file);
    guest.add("file:anonymous", 20, "fixture");
    guest.add("measured-guest", 13, "fixture", ByteMeaning::measured);
    guest.mirror_into(host);
    host.add("measured-host", 7, "independent fixture", ByteMeaning::measured);
    auto policy = both_scopes_measured(4 * GiB + 73, 8 * GiB + 47);
    policy.file_cache_credit_bytes = 1000;
    DeviceCost device;
    const auto telemetry = measured("CUDA0", 1024);
    auto decide = [&] { return admit_configuration({device}, {telemetry}, guest, host, policy); };
    auto result = decide();
    require(result.admitted && result.guest_demand_bytes == 73 && result.host_demand_bytes == 47,
            "declared demand lost independent measurement or credited locked/anonymous bytes");
    --policy.guest_total_bytes;
    require(!decide().admitted, "one-byte guest overflow admitted");
    ++policy.guest_total_bytes;
    --policy.host_available_bytes;
    require(!decide().admitted, "one-byte host overflow admitted");
    ++policy.host_available_bytes;
    Scope unknown;
    unknown.add_unknown("renamed-unknown", "fixture", ByteMeaning::clean_file);
    unknown.mirror_into(host);
    result = decide();
    require(!result.admitted && !result.verified && host.first_unknown() &&
            result.unknown_classes == std::vector<std::string>{"host:renamed-unknown"},
            "unknown source demand became an admitted zero during projection");
}

}  // namespace

int main() {
    try {
        test_single_overfull_device_refuses_despite_aggregate();
        test_guest_is_a_real_limit_and_reserves_cannot_be_planned_away();
        test_missing_telemetry_fails_closed();
        test_unknown_class_blocks_admission();
        test_file_backed_demand_is_counted_and_credit_is_bounded();
        test_hypothetical_devices_are_labelled_and_never_admitted();
        test_mixed_real_and_hypothetical_is_decided_by_the_real_device();
        test_request_admission_by_state_demand();
        test_explain_or_shrink_comparison();
        test_overflow_refuses_in_every_scope_and_request();
        test_mps_physical_and_residual_gates_are_independent();
        test_declared_meaning_projects_unknowns_and_bounds_credit();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "plan_admission_test: %s\n", e.what());
        return 1;
    }
    return 0;
}
