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
#include "strata/plan/resident.hpp"
#include "strata/plan/qsa_memory.hpp"
#include "strata/program/time_budget.hpp"

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
    // 8 GiB guest - 4 GiB reserve cannot hold 6 GiB: name the scope and crossing class.
    const AdmissionDecision refused =
        admit_configuration({a}, {measured("CUDA0", 24 * GiB)}, guest, {}, both_scopes_measured(8 * GiB, 0));
    require(!refused.admitted, "guest demand above the reserved envelope was admitted");
    require(refused.limiting_term == "guest:conversation_parked", "guest refusal omitted its class: " + refused.limiting_term);

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

void test_controller_host_reserve_does_not_lower_other_bounds() {
    GuestHostPolicy p = both_scopes_measured(8 * GiB, 3 * GiB);
    p.host_reserve_bytes = 0;
    Scope host;
    host.add("source-mirror", 2 * GiB, "fixture");
    DeviceCost device;
    device.name = "CUDA0";
    const std::vector<DeviceCost> devices{device};
    const std::vector<DeviceTelemetry> telemetry{measured("CUDA0", 24 * GiB)};
    require(!admit_configuration(devices, telemetry, {}, host, p).admitted,
            "unapproved zero host reserve bypassed the default floor");
    p.host_reserve_controller_approved = true;
    auto allowed = admit_configuration(devices, telemetry, {}, host, p);
    require(allowed.admitted && allowed.applied_host_reserve_bytes == 0,
            "authenticated controller reserve was clamped back to the default");
    require(allowed.applied_guest_reserve_bytes == 4 * GiB, "guest floor changed");
    p.host_available_bytes = 2 * GiB - 1;
    require(!admit_configuration(devices, telemetry, {}, host, p).admitted,
            "controller exception ignored actual host demand");
    p.host_available_measured = false;
    require(!admit_configuration(devices, telemetry, {}, host, p).admitted,
            "controller exception admitted missing host telemetry");
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
    // A resident prefill executes GEN 1, but admission must price the original
    // completion budget, with the same eight-cell margin as the solo request.
    req.prompt_tokens = 23;
    req.max_new_tokens = 4065;
    const auto near = admit_request(req, 4096);
    require(near.admitted && near.cells == 4088 &&
            near.state_demand_bytes == state_bytes(g) + 4088ull * kv_bytes_per_token(g),
            "near-limit resident budget lost cells or state demand");
    ++req.max_new_tokens;
    require(!admit_request(req, 4096).admitted,
            "one cell beyond the unchanged resident context margin admitted");
    req.prompt_tokens = 2;
    req.max_new_tokens = INT64_MAX;
    require(!admit_request(req, 4096).admitted,
            "resident output-budget overflow reached state pricing");
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

// HET-015: per-device MPS ceilings are per-device facts. Two devices with unequal caps must
// each close on their own numbers: one device's envelope violation refuses while the other's
// slack cannot rescue it, and both fitting admits with the receipts carried per line.
void test_unequal_per_device_caps_and_envelopes_refuse_independently() {
    DeviceCost primary;
    primary.name = "CUDA0";
    primary.vram.add("weights_canonical", 6 * GiB, "test");
    primary.headroom_bytes = 1 * GiB;
    DeviceCost stage;
    stage.name = "CUDA1";
    stage.vram.add("weights_canonical", 4 * GiB, "test");
    stage.vram.add("split_expert_caches", 2 * GiB, "admitted per-stage budget");
    stage.headroom_bytes = 1 * GiB;
    auto telemetry = [&](uint64_t cap0, uint64_t free0, uint64_t physical0,
                         uint64_t cap1, uint64_t free1, uint64_t physical1) {
        DeviceTelemetry a;
        a.name = "CUDA0";
        a.measured = true;
        a.total_bytes = 24 * GiB;
        a.free_bytes = free0;
        a.cap_declared = true;
        a.enforced_cap_bytes = cap0;
        a.physical_free_measured = true;
        a.physical_free_bytes = physical0;
        a.outside_client_allowance_bytes = 1 * GiB;
        DeviceTelemetry b = a;
        b.name = "CUDA1";
        b.free_bytes = free1;
        b.enforced_cap_bytes = cap1;
        b.physical_free_bytes = physical1;
        return std::vector<DeviceTelemetry>{a, b};
    };
    const auto both_fit = admit_configuration({primary, stage},
        telemetry(10 * GiB, 9 * GiB, 12 * GiB, 8 * GiB, 7 * GiB, 20 * GiB), {}, {}, both_scopes_measured(64 * GiB, 64 * GiB));
    require(both_fit.admitted && both_fit.verified, "unequal caps that each fit must admit");
    require(both_fit.devices[0].enforced_cap_bytes == 10 * GiB &&
            both_fit.devices[1].enforced_cap_bytes == 8 * GiB &&
            both_fit.devices[0].physical_free_bytes == 12 * GiB &&
            both_fit.devices[1].physical_free_bytes == 20 * GiB,
            "per-device receipts must not be cross-contaminated");
    // The second device's physical envelope is its own: 8 + 1 > 8.5 refuses even though the
    // first device has 5 GiB of slack its ceiling never shares.
    const auto second_overfull = admit_configuration({primary, stage},
        telemetry(10 * GiB, 9 * GiB, 12 * GiB, 8 * GiB, 7 * GiB, (8 * GiB + 1 * GiB) / 2), {}, {},
        both_scopes_measured(64 * GiB, 64 * GiB));
    require(!second_overfull.admitted && second_overfull.devices[0].fits &&
            !second_overfull.devices[1].fits,
            "one device's physical envelope violation must refuse without touching the other");
    require(second_overfull.limiting_term == "mps:CUDA1",
            "the limiting term must name the offending device: " + second_overfull.limiting_term);
    // A residual above ITS OWN cap on the first device refuses with the second still fitting.
    const auto first_residual = admit_configuration({primary, stage},
        telemetry(10 * GiB, 11 * GiB, 12 * GiB, 8 * GiB, 7 * GiB, 20 * GiB), {}, {},
        both_scopes_measured(64 * GiB, 64 * GiB));
    require(!first_residual.admitted && first_residual.limiting_term == "mps:CUDA0" &&
            first_residual.devices[1].fits,
            "a residual above its own declared cap refuses on that device alone");
}

// The explicit-split per-stage expert-cache budget: the host arithmetic the runtime cache is
// held to. Rejection, the one-blob minimum, blob-granularity flooring and the profiled-pair
// bound are the consumer-visible risks of the admission seam.
void test_split_stage_cache_budget_boundaries() {
    const uint64_t blob = 2 * GiB;
    // Nothing after fixed demand plus headroom: no budget, never a negative or a guess.
    require(split_stage_cache_budget(8 * GiB, 8 * GiB, 1 * GiB, blob, 64 * GiB) == 0,
            "an over-committed stage priced a cache from nothing");
    require(split_stage_cache_budget(8 * GiB, 9 * GiB, 0, blob, 64 * GiB) == 0,
            "fixed demand above the residual priced a cache");
    // The minimum: residual - fixed - headroom funds exactly one blob.
    require(split_stage_cache_budget(11 * GiB, 8 * GiB, 1 * GiB, blob, 64 * GiB) == blob,
            "the one-blob minimum was not the floor");
    // One byte short of one blob floors to zero: the granularity is never rounded up.
    require(split_stage_cache_budget(11 * GiB, 8 * GiB + 1, 1 * GiB, blob, 64 * GiB) == 0,
            "a sub-granularity remainder was rounded up into a blob");
    // The floor drops the remainder: 2.5 blobs of room prices 2 blobs.
    require(split_stage_cache_budget(14 * GiB, 8 * GiB, 1 * GiB, blob, 64 * GiB) == 2 * blob,
            "the budget was not floored to the blob granularity");
    // The stage's profiled pairs bound the budget: unbounded VRAM cannot buy unprofiled slots.
    require(split_stage_cache_budget(64 * GiB, 8 * GiB, 1 * GiB, blob, 3 * blob) == 3 * blob,
            "the profile bound did not cap the budget");
    // Zero blob geometry refuses everything rather than dividing by zero.
    require(split_stage_cache_budget(64 * GiB, 0, 0, 0, 64 * GiB) == 0,
            "a zero blob geometry produced a budget");
}

void test_remote_device_mapping_excludes_every_stage_and_helper() {
    std::array<int, 3> devices{-1, -1, -1};
    std::string error;
    require(remote_expert_devices(3, {1}, {512, 0, 0}, devices, error) && devices[0] == 2,
            "CUDA1 stage did not put tier one on CUDA2");
    require(remote_expert_devices(3, {2}, {512, 0, 0}, devices, error) && devices[0] == 1,
            "CUDA2 stage did not put tier one on CUDA1");
    require(remote_expert_devices(6, {2}, {512, 256, 128}, devices, error) &&
            devices == std::array<int, 3>{1, 3, 4}, "multiple helpers shared a stage or helper device");
    const auto retained = devices;
    for (const auto& stages : {std::vector<int>{1, 2}, std::vector<int>{1, 1},
                              std::vector<int>{0}, std::vector<int>{3}}) {
        require(!remote_expert_devices(3, stages, {512, 0, 0}, devices, error) &&
                !error.empty() && devices == retained,
                "unavailable/overlapping stage layout changed the admitted helper mapping");
    }
    require(!remote_expert_devices(3, {}, {0, 512, 0}, devices, error) && devices == retained,
            "a missing earlier helper tier was accepted");
    require(!remote_expert_devices(3, {}, {-1, 0, 0}, devices, error) && devices == retained,
            "a negative helper budget was accepted");
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

void test_streamed_slot_host_budget_refuses_before_allocation() {
    using namespace strata::kernels;
    const auto s = qsa_real_shapes();
    const auto roles = qsa_resolve_roles(s, 262144, kKvHybrid, 0, 20480, 32768, 8);
    const uint64_t pinned = 2 * 12 * qsa_pinned_bytes(s, roles.slot_main.kv, roles.slot_main.format);
    require(pinned == 5133855456ull, "streamed hybrid slots do not charge full logical backing");
    Scope guest, host;
    guest.add("resident_slot_main_pinned", pinned, "resolved requested slot backing");
    guest.mirror_all_into(host); // direct host operation, not VM backing duplication
    DeviceCost device;
    device.vram.add("resident_slot_main_device", 2 * 12 * qsa_device_bytes(
        s, 262144, roles.slot_main.kv, roles.slot_main.format, false), "resolved slot windows");
    auto policy = both_scopes_measured(64 * GiB, 8 * GiB + pinned);
    policy.file_cache_credit_bytes = UINT64_MAX; // pinned memory is not clean cache
    const auto tel = measured("CUDA0", 24 * GiB);
    const auto admitted = admit_configuration({device}, {tel}, guest, host, policy);
    require(admitted.admitted && admitted.host_demand_bytes == pinned,
            "exact-fit streamed slots refused or received clean-cache credit");
    --policy.host_available_bytes;
    const auto refused = admit_configuration({device}, {tel}, guest, host, policy);
    require(!refused.admitted && refused.devices[0].fits && refused.guest_fits && !refused.host_fits,
            "device/guest slack rescued one-byte streamed host overflow");
    require(refused.limiting_term == "host:resident_slot_main_pinned",
            "streamed host refusal did not name its byte class: " + refused.limiting_term);
    require(refused.reasons.front().find("limiting class resident_slot_main_pinned") != std::string::npos,
            "streamed host refusal message omitted its byte class");
    Scope vm_host;
    guest.mirror_into(vm_host);
    require(vm_host.known_bytes() == 0, "pinned guest backing counted twice in VM host projection");
}

void test_role_aware_priceability_keeps_unknown_combinations_closed() {
    using namespace strata::kernels;
    const auto s = qsa_real_shapes();
    for (int fmt : {kKvF16, kKvInt8, kKvQ4, kKvHybrid}) {
        for (int64_t window : {int64_t{-1}, int64_t{0}, int64_t{65536}}) {
            const auto roles = qsa_resolve_roles(s, 262144, fmt, 0, window, 32768, 8);
            require(!resident_role_unpriced(s, 262144, roles, true),
                    "quantized resident or streamed roles are still refused");
        }
    }
    auto roles = qsa_resolve_roles(s, 262144, kKvHybrid, 20480, -1, 32768, 8);
    require(!resident_role_unpriced(s, 262144, roles, true), "inherited global stream refused");
    roles.slot_main.format = kKvF16;
    require(std::string(resident_role_unpriced(s, 262144, roles, true)) == "resident_slot_main_device",
            "per-role main format drift admitted");
    roles = qsa_resolve_roles(s, 262144, kKvHybrid, 0, 65536, 32768, 8);
    roles.slot_draft.format = kKvHybrid;
    require(std::string(resident_role_unpriced(s, 262144, roles, true)) == "resident_slot_draft_device",
            "hybrid slot draft silently priced despite whole-INT8 allocation");
    roles = qsa_resolve_roles(s, 262144, kKvHybrid, 0, 65536, 32768, 8);
    roles.slot_main.kv.pages += 1;
    require(resident_role_unpriced(s, 262144, roles, false) != nullptr,
            "unpriceable frozen layout admitted");
    roles = qsa_resolve_roles(s, 262144, kKvHybrid, 0, 65536, 32768, 8);
    roles.working_main.elastic = true;
    require(std::string(resident_role_unpriced(s, 262144, roles, true)) == "elastic_kv_growth",
            "elastic batch placement widened into a known zero");
    DeviceCost device;
    device.vram.add_unknown("elastic_kv_growth", "unpriced VMM growth with requested batch slots");
    const auto refused = admit_configuration({device}, {measured("CUDA0", 24 * GiB)},
                                            {}, {}, both_scopes_measured(64 * GiB, 64 * GiB));
    require(!refused.admitted && !refused.verified && refused.limiting_term == "CUDA0:elastic_kv_growth",
            "elastic batch demand admitted or refusal lost its scope/class");
}

void test_measured_prefill_quanta_and_bounded_decode_share() {
    using strata::program::TimeBudget;
    require(TimeBudget::valid(1000, .25) && TimeBudget::valid(0, .25), "valid scheduling budget refused");
    require(!TimeBudget::valid(std::nan(""), .25) && !TimeBudget::valid(1000, 0) &&
            !TimeBudget::valid(1000, 1) && !TimeBudget::valid(60001, .25), "invalid scheduling budget accepted");
    TimeBudget policy(1000, .25);
    require(policy.prefill_tokens(10000, 2048, 0) == 2048, "cold estimate must use one admitted chunk");
    require(policy.prefill_tokens(10000, 0, 0) == 0, "unknown memory headroom was treated as a chunk");
    policy.prefilled(250, 1, 2048);
    require(policy.prefill_tokens(10000, 2048, 0) == 8192, "measured time did not choose a multi-chunk quantum");
    require(policy.prefill_tokens(10000, 2048, 700) == 2048, "due decoder did not shorten the quantum");
    require(policy.prefill_tokens(123, 2048, 0) == 123, "tail partition changed");
    policy.prefilled(1200, 1, 2048);
    require(policy.prefill_tokens(INT64_MAX, 2048, 0) == 2048, "over-budget chunk was silently enlarged");
    policy.prefilled(1, 1, 256);
    require(policy.prefill_tokens(10000, 2048, 0) == 2048 && policy.decode_due(0, 2048),
            "a short request poisoned a larger memory-admitted chunk's estimate");
    TimeBudget overflow(60000, .25);
    overflow.prefilled(1e-18, 1, 1);
    require(overflow.prefill_tokens(INT64_MAX, 1, 0) == INT64_MAX &&
            overflow.prefill_tokens(INT64_MAX, INT64_MAX, 0) == INT64_MAX, "quantum arithmetic overflow");

    // Persistent foreground load: each round serves three decoders. The fourth
    // decoder arrives during prefill; every indivisible round drains before the
    // next background chunk and its overrun remains debt (not free credit).
    TimeBudget busy(100, .25);
    double now = 0, last = 0, prefill = 0, decode = 0;
    int commits[4] = {}, background_chunks = 0;
    for (int chunk = 0; chunk < 200; ++chunk) {
        const int64_t tokens = busy.prefill_tokens(2048 * 201, 2048, now - last);
        require(tokens > 0 && tokens % 2048 == 0, "scheduling changed the arithmetic partition");
        const double cost = 50 * (double) (tokens / 2048);
        now += cost; prefill += cost; ++background_chunks;
        busy.prefilled(cost, tokens / 2048, 2048);
        int rounds = 0;
        while (busy.decode_due(now - last, 2048) && busy.decode_allowed(0)) {
            now += 40; decode += 40; last = now;
            busy.decoded(40);
            for (int owner = 0; owner < (chunk >= 1 ? 4 : 3); ++owner) ++commits[owner];
            require(++rounds <= 3, "foreground rounds starved background progress");
        }
        require(now - last <= 100, "foreground arrival trace exceeded its supported quantum envelope");
    }
    require(background_chunks == 200 && commits[0] > 0 && commits[3] > 0, "an owner starved");
    require(decode <= prefill * 3 + 40, "indivisible-round debt was forgiven on every boundary");
    busy.decoded(10000, false);
    busy.prefilled(50, 1, 2048);
    require(busy.decode_allowed(0), "decode-only time accumulated background debt");

    TimeBudget warm(100, .25);
    double warm_service = 0, resident_service = 0;
    int last_round = 0;
    for (int admission = 1; admission <= 100; ++admission) {
        warm.background_work(10); // restoration/admission only: no sp.run
        warm_service += 10;
        if (warm.decode_due(0, 2048) && warm.decode_allowed(0)) {
            warm.decoded(40);
            resident_service += 40;
            last_round = admission;
        }
        require(admission - last_round <= 1, "warm-only admissions starved resident decoding");
    }
    require(resident_service <= warm_service * 3 + 40, "warm service forgave indivisible decode debt");
    require(warm.prefill_tokens(10000, 2048, 0) == 2048,
            "restoration time invented a trained bulk-prefill chunk cost");
    warm.decoded(1000);
    warm.background_work(-100);
    warm.background_work(std::numeric_limits<double>::quiet_NaN());
    warm.background_work(std::numeric_limits<double>::infinity());
    require(!warm.decode_allowed(0), "invalid service samples forgave outstanding decode debt");

    TimeBudget overlap(1000, .25);
    overlap.prefilled(20, 1, 2048);
    overlap.decoded(10);
    overlap.overlapped(100, 40); // 50 + 3*(100-40) - 40 = 190, not 310
    require(overlap.decode_allowed(189) && !overlap.decode_allowed(190),
            "concurrent decode was credited as prompt work or charged twice");
    require(overlap.chunk_ms() == 20 && overlap.decode_ms() == 10,
            "overlap trained the sequential prefill/decode estimate");
    overlap.overlapped(10, 200); // no negative background credit; debt = -10
    require(!overlap.decode_allowed(0), "decode overrun did not carry debt");
    for (const auto invalid : {std::nan(""), std::numeric_limits<double>::infinity(), -1.0}) {
        overlap.overlapped(invalid, 0);
        overlap.overlapped(100, invalid);
    }
    require(!overlap.decode_allowed(0), "invalid overlap samples forgave debt");
    overlap.overlapped(4, 0); // debt cleared by non-decode time, without decode_due(age)
    require(overlap.decode_allowed(1) && !overlap.decode_allowed(2) &&
            !overlap.decode_due(0, 2048), "concurrent start gate still waits for the foreground age");
    overlap.overlapped(100000, 0);
    require(overlap.decode_allowed(999) && !overlap.decode_allowed(1000), "overlap exceeded bounded credit");
    TimeBudget disabled(0, .25);
    disabled.overlapped(100, 20);
    require(!disabled.decode_allowed(0) && disabled.decode_ms() == 0 && disabled.chunk_ms() == 0,
            "disabled overlap mutated scheduling state");

    // Carried debt remains blocked exactly at zero; uncharged decode cannot spend it.
    TimeBudget retained(100, .25);
    retained.prefilled(60, 1, 2048);
    retained.decoded(250);
    require(!retained.decode_allowed(0) && retained.decode_ms() == 250,
            "retained decode discarded its indivisible-round debt");
    retained.background_work(40);
    require(!retained.decode_allowed(0), "partial background work cleared retained debt");
    retained.background_work(10);
    require(!retained.decode_allowed(0), "zero retained credit admitted decoding");
    retained.background_work(1);
    require(retained.decode_allowed(0) && !retained.decode_allowed(3),
            "retained credit did not preserve its exact start boundary");
    retained.decoded(1000, false);
    require(retained.decode_allowed(0), "uncharged decode spent retained foreground credit");

    // Live read intervals replenish carried debt. Settling only at the part's
    // end would either starve every cold start or cap all decode at 1000 ms.
    TimeBudget live(100, .25);
    double overlapped_decode = 0;
    for (int interval = 0; interval < 10000; ++interval) {
        const double decoding = live.decode_allowed(0) ? 4.0 : 0.0;
        live.overlapped(4, decoding);
        overlapped_decode += decoding;
    }
    require(std::abs(overlapped_decode / 40000.0 - .75) < .001,
            "live overlap violated the background share or stopped replenishing");
}

}  // namespace

int main() {
    try {
        test_single_overfull_device_refuses_despite_aggregate();
        test_guest_is_a_real_limit_and_reserves_cannot_be_planned_away();
        test_controller_host_reserve_does_not_lower_other_bounds();
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
        test_unequal_per_device_caps_and_envelopes_refuse_independently();
        test_split_stage_cache_budget_boundaries();
        test_remote_device_mapping_excludes_every_stage_and_helper();
        test_streamed_slot_host_budget_refuses_before_allocation();
        test_role_aware_priceability_keeps_unknown_combinations_closed();
        test_measured_prefill_quanta_and_bounded_decode_share();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "plan_admission_test: %s\n", e.what());
        return 1;
    }
    return 0;
}
