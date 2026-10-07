// include/strata/plan/admission.hpp - HET-017: per-device + guest/host memory admission.
//
// `plan.hpp` answers "how many expert slots does one VRAM pool leave".  That arithmetic has no notion of
// WHICH device holds what, of the guest allocation, or of the host reserve - which is exactly the gap
// HET-017 closes.  The memory equation (docs/02_HARDWARE_AND_MEMORY.md) is enforced here, per device,
// separately:
//
//     M_d = W_d + sum_r S_{r,d} + B_d + G_d + H_d  <=  M_{d,usable}
//
// with W weights, S per-request state, B in-flight buffers, G graph/workspace allocations and H explicit
// headroom.  Aggregate VRAM cannot rescue one overfull GPU: there is no aggregate step to rescue with.
//
// RULES THIS ENCODES (all four acceptance criteria, structurally):
//   * Every byte is an `Accounted` entry with a source; a class the caller cannot quantify is added with
//     `unknown = true` and REFUSES the configuration - a new allocation is an admission blocker, never a
//     silent zero.
//   * Guest and host are real scopes with design-default reserves (4 GiB in-guest, 8 GiB on the host,
//     docs/02): demand above the reserve refuses.  File-backed demand is COUNTED, never called free; the
//     only credit ever taken is the bounded clean-file-cache credit the caller measured through the
//     cgroup-v2 counters (fail-closed when they are missing), and it caps at the file bytes themselves.
//   * Missing telemetry (device ceiling, guest allocation, host available) refuses: unknown is not zero.
//   * Devices marked `hypothetical` (the future 24/32/32 GiB capacities) are reported and never admitted:
//     a plan whose only devices are hypothetical cannot admit, because nothing was measured.
//
// This header is host-only arithmetic (no CUDA), so the same decision the runtime enforces is the one
// `strata-plan --json` prints and the tests replay.
#pragma once

#include "strata/plan/plan.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace strata::plan {

inline uint64_t saturating_add(uint64_t a, uint64_t b) {
    return b > UINT64_MAX - a ? UINT64_MAX : a + b;
}

inline uint64_t saturating_multiply(uint64_t a, uint64_t b) {
    return a != 0 && b > UINT64_MAX / a ? UINT64_MAX : a * b;
}

// ---- one accounted byte class -------------------------------------------------------------------------------
//
// `unknown` is the load-bearing flag: `bytes == 0 && !unknown` is a measured/derived zero, while
// `unknown == true` means "this allocation exists and I cannot size it", which refuses admission.
enum class ByteMeaning { anonymous, clean_file, locked_file, measured, measured_clean };

inline const char* byte_meaning_name(ByteMeaning meaning) {
    switch (meaning) {
    case ByteMeaning::anonymous: return "anonymous";
    case ByteMeaning::clean_file: return "clean-file";
    case ByteMeaning::locked_file: return "locked-file";
    case ByteMeaning::measured: return "measured";
    case ByteMeaning::measured_clean: return "measured-clean";
    }
    return "invalid";
}

inline bool parse_byte_meaning(const std::string& text, ByteMeaning& meaning) {
    for (auto candidate : {ByteMeaning::anonymous, ByteMeaning::clean_file, ByteMeaning::locked_file,
                           ByteMeaning::measured, ByteMeaning::measured_clean})
        if (text == byte_meaning_name(candidate)) { meaning = candidate; return true; }
    return false;
}

struct Accounted {
    std::string name;
    uint64_t bytes = 0;
    bool unknown = false;
    std::string source;   // where the number came from, e.g. "index.txt pool header" - printed in the plan
    ByteMeaning meaning = ByteMeaning::anonymous;
    bool mirrors_host() const { return meaning == ByteMeaning::clean_file || meaning == ByteMeaning::locked_file; }
    bool clean_credit() const { return meaning == ByteMeaning::clean_file || meaning == ByteMeaning::measured_clean; }
};

// A scope is one ceiling's worth of classes: one device's VRAM, the guest's RAM, or the host's.
struct Scope {
    std::vector<Accounted> classes;

    void add(std::string name, uint64_t bytes, std::string source, ByteMeaning meaning = ByteMeaning::anonymous) {
        classes.push_back(Accounted{std::move(name), bytes, false, std::move(source), meaning});
    }
    void add_unknown(std::string name, std::string source, ByteMeaning meaning = ByteMeaning::anonymous) {
        classes.push_back(Accounted{std::move(name), 0, true, std::move(source), meaning});
    }
    void mirror_into(Scope& host) const {
        for (const auto& c : classes) if (c.mirrors_host()) {
            auto mirrored = c;
            mirrored.source = "physical host cache mirror upper bound (no sharing credit): " + c.source;
            host.classes.push_back(std::move(mirrored));
        }
    }
    uint64_t credit_eligible_bytes() const {
        uint64_t bytes = 0;
        for (const auto& c : classes)
            if (!c.unknown && c.clean_credit()) bytes = saturating_add(bytes, c.bytes);
        return bytes;
    }
    const Accounted* find(const std::string& name) const {
        for (const Accounted& c : classes)
            if (c.name == name) return &c;
        return nullptr;
    }
    uint64_t known_bytes() const {
        uint64_t n = 0;
        for (const Accounted& c : classes) n = saturating_add(n, c.bytes);
        return n;
    }
    bool overflowed() const {
        uint64_t n = 0;
        for (const Accounted& c : classes) {
            if (c.bytes > UINT64_MAX - n) return true;
            n += c.bytes;
        }
        return false;
    }
    const Accounted* first_unknown() const {
        for (const Accounted& c : classes)
            if (c.unknown) return &c;
        return nullptr;
    }
};

// ---- inputs -----------------------------------------------------------------

struct DeviceCost {
    std::string name = "CUDA0";   // driver-visible device name, e.g. "CUDA0"
    Scope vram;                   // everything this device must hold at peak
    uint64_t headroom_bytes = 0;  // the explicit H in the memory equation
    bool hypothetical = false;    // a labelled future capacity; never admitted, never aggregated
};

struct DeviceTelemetry {
    std::string name;
    bool measured = false;
    uint64_t total_bytes = 0;   // driver-visible total
    uint64_t free_bytes = 0;    // driver-visible free before planned allocations; zero admits no bytes
    /// Creation-time MPS client budget; free_bytes is its residual, not physical free.
    /// The independent physical envelope uses the sample taken before this context.
    bool cap_declared = false;
    uint64_t enforced_cap_bytes = 0;
    bool physical_free_measured = false;
    uint64_t physical_free_bytes = 0;
    uint64_t outside_client_allowance_bytes = 0;
};

/// Budget policy.  The reserves carry the docs/02 design defaults; an operator may RAISE them through the
/// engine's flags but the defaults here are the floor - the admission code never accepts a reserve below
/// them (see `clamp_reserves`), so no flag can plan away the host or guest.
struct GuestHostPolicy {
    bool guest_total_measured = false;
    uint64_t guest_total_bytes = 0;        // the VM allocation, driver/hypervisor-visible
    uint64_t guest_reserve_bytes = 4ull << 30;
    bool host_available_measured = false;
    uint64_t host_available_bytes = 0;     // physical-host MemAvailable; incremental host demand
    uint64_t host_reserve_bytes = 8ull << 30;
    /// Bounded clean-file-cache credit (AC-4): the ONLY way file-backed bytes are ever discounted, capped
    /// at the file bytes and at what the caller measured as clean/reclaimable.  Zero credits nothing.
    uint64_t file_cache_credit_bytes = 0;
    /// Predicted-vs-observed comparison tolerance (AC-3): observed peaks above predicted+tolerance must be
    /// explained before any configuration is enlarged.  Explicit, not magic: 256 MiB covers allocator and
    /// driver-visible rounding on one card.
    uint64_t comparison_tolerance_bytes = 256ull << 20;

    void clamp_reserves() {
        const uint64_t kMinGuest = 4ull << 30, kMinHost = 8ull << 30;
        guest_reserve_bytes = std::max(guest_reserve_bytes, kMinGuest);
        host_reserve_bytes = std::max(host_reserve_bytes, kMinHost);
    }
};

// ---- the decision ------------------------------------------------------------

struct DeviceLine {
    std::string name;
    bool hypothetical = false;
    uint64_t ceiling_bytes = 0;    // measured driver-visible total (0 for hypothetical)
    bool cap_declared = false;     // an MPS client ceiling was in force for this device
    uint64_t enforced_cap_bytes = 0;
    bool physical_free_measured = false;
    uint64_t physical_free_bytes = 0;
    uint64_t outside_client_allowance_bytes = 0;
    uint64_t demand_bytes = 0;     // known classes summed
    uint64_t headroom_bytes = 0;
    uint64_t slack_bytes = 0;      // ceiling - demand - headroom (saturating)
    bool fits = false;
    bool has_ceiling = false;
    std::string limiting_class;    // the class that pushed demand over, when it does not fit
};

struct AdmissionDecision {
    bool admitted = false;
    bool verified = false;         // every required ceiling was measured
    std::string variant = "cold";  // cold | warm | resident - a LABEL on how file-backed pages were treated
    std::vector<std::string> reasons;
    std::vector<std::string> unknown_classes;
    std::string limiting_term;     // "CUDA0:expert_cache" / "guest" / "host" / "telemetry:CUDA1" / ""
    std::vector<DeviceLine> devices;

    bool guest_known = false, guest_fits = false;
    uint64_t guest_demand_bytes = 0, guest_ceiling_bytes = 0;
    bool host_known = false, host_fits = false;
    uint64_t host_demand_bytes = 0, host_ceiling_bytes = 0;
    // The reserves AFTER the design-default floor - what the decision actually enforced, so a plan
    // document cannot claim a smaller reserve was honoured.
    uint64_t applied_guest_reserve_bytes = 0, applied_host_reserve_bytes = 0;

    uint64_t total_known_demand_bytes() const {
        uint64_t n = 0;
        for (const DeviceLine& d : devices) n = saturating_add(n, d.demand_bytes);
        return n;
    }
};

/// THE ADMISSION DECISION.  Refuses (returns admitted=false) when:
///   * any class is unaccounted (`unknown`) - named in `unknown_classes` and `reasons`;
///   * a real device's ceiling was not measured - `telemetry:<name>` (hypothetical devices need none);
///   * any single real device exceeds its own ceiling even when every other device has slack - the
///     limiting term names `device:class`;
///   * guest or host demand exceeds the measured ceiling less the (floored) reserve, or either ceiling
///     is unmeasured while that scope carries demand;
///   * every device is hypothetical - nothing was measured, so nothing is admitted.
AdmissionDecision admit_configuration(const std::vector<DeviceCost>& devices,
                                      const std::vector<DeviceTelemetry>& telemetry,
                                      const Scope& guest_demand, const Scope& host_demand,
                                      const GuestHostPolicy& policy,
                                      const std::string& variant = "cold") {
    AdmissionDecision d;
    d.variant = variant;
    if (devices.empty()) {
        d.reasons.push_back("no devices in the configuration - nothing was measured, nothing is admitted");
        d.limiting_term = "telemetry:none";
        return d;
    }
    GuestHostPolicy p = policy;
    p.clamp_reserves();
    d.applied_guest_reserve_bytes = p.guest_reserve_bytes;
    d.applied_host_reserve_bytes = p.host_reserve_bytes;

    // 1. Unknown classes refuse, by name, whatever else is true.
    for (const DeviceCost& dev : devices)
        if (const Accounted* u = dev.vram.first_unknown()) {
            d.unknown_classes.push_back(dev.name + ":" + u->name);
            d.reasons.push_back("unaccounted allocation: " + dev.name + ":" + u->name +
                                " (" + u->source + ") - size it before admitting");
        }
    if (const Accounted* u = guest_demand.first_unknown()) {
        d.unknown_classes.push_back("guest:" + u->name);
        d.reasons.push_back("unaccounted allocation: guest:" + u->name + " (" + u->source + ")");
    }
    if (const Accounted* u = host_demand.first_unknown()) {
        d.unknown_classes.push_back("host:" + u->name);
        d.reasons.push_back("unaccounted allocation: host:" + u->name + " (" + u->source + ")");
    }
    auto reject_overflow = [&](const Scope& scope, const std::string& name) {
        if (scope.overflowed()) {
            d.unknown_classes.push_back(name + ":byte_overflow");
            d.reasons.push_back(name + ": byte total overflow - refuse before allocation");
            if (d.limiting_term.empty()) d.limiting_term = name + ":byte_overflow";
        }
    };
    for (const DeviceCost& dev : devices) reject_overflow(dev.vram, dev.name);
    reject_overflow(guest_demand, "guest");
    reject_overflow(host_demand, "host");

    auto telemetry_of = [&](const std::string& name) -> const DeviceTelemetry* {
        for (const DeviceTelemetry& t : telemetry)
            if (t.name == name) return &t;
        return nullptr;
    };

    // 2. Per-device accounting - the memory equation, one device at a time.  There is deliberately no
    //    aggregate test anywhere in this loop: AC-1 is the absence of one.
    bool all_hypothetical = !devices.empty();
    for (const DeviceCost& dev : devices) {
        DeviceLine line;
        line.name = dev.name;
        line.hypothetical = dev.hypothetical;
        line.headroom_bytes = dev.headroom_bytes;
        line.demand_bytes = dev.vram.known_bytes();
        const DeviceTelemetry* t = telemetry_of(dev.name);
        if (dev.hypothetical) {
            // A labelled future capacity: reported for the plan document, never admitted, never a
            // source of slack for a real device.
            line.fits = false;
            line.has_ceiling = false;
            line.limiting_class = "hypothetical: not installed - plan is labelled, not validated";
            // NOTE: deliberately no `reasons` entry here - a mixed plan (a real device plus labelled
            // future rows) is decided by the REAL devices alone; only an all-hypothetical plan refuses
            // (the check after this loop).  The label travels on the DeviceLine instead.
        } else {
            all_hypothetical = false;
            if (t != nullptr) {
                line.cap_declared = t->cap_declared;
                line.enforced_cap_bytes = t->enforced_cap_bytes;
                line.physical_free_measured = t->physical_free_measured;
                line.physical_free_bytes = t->physical_free_bytes;
                line.outside_client_allowance_bytes = t->outside_client_allowance_bytes;
            }
            if (t == nullptr || !t->measured) {
                line.fits = false;
                line.has_ceiling = false;
                line.limiting_class = "telemetry missing";
                d.reasons.push_back("no measured ceiling for " + dev.name +
                                    " - refuse (fail-closed; unknown telemetry is not zero)");
                if (d.limiting_term.empty()) d.limiting_term = "telemetry:" + dev.name;
            } else {
                line.has_ceiling = true;
                // With a declared MPS ceiling, `free_bytes` is the client's remaining budget:
                // taking min(total, free) is still the tightest honest ceiling because the
                // client may not exceed its budget, and the budget is already below the cap
                // by whatever the context has charged. Without a ceiling this is unchanged.
                line.ceiling_bytes = std::min(t->total_bytes, t->free_bytes);
                line.fits = !dev.vram.overflowed() && line.headroom_bytes <= line.ceiling_bytes &&
                            line.demand_bytes <= line.ceiling_bytes - line.headroom_bytes;
                line.slack_bytes = line.fits ? line.ceiling_bytes - line.demand_bytes - line.headroom_bytes : 0;
                if (!line.fits) {
                    // name the class that crossed the line: walk the classes and find the one whose
                    // addition first exceeded the ceiling.
                    uint64_t running = 0;
                    const Accounted* crossed = nullptr;
                    for (const Accounted& c : dev.vram.classes) {
                        running = saturating_add(running, c.bytes);
                        if (line.headroom_bytes > line.ceiling_bytes ||
                            running > line.ceiling_bytes - line.headroom_bytes) { crossed = &c; break; }
                    }
                    line.limiting_class = crossed ? crossed->name : "headroom";
                    d.reasons.push_back(dev.name + " overfull: demand " +
                                        std::to_string(line.demand_bytes) + " B + headroom " +
                                        std::to_string(line.headroom_bytes) + " B > ceiling " +
                                        std::to_string(line.ceiling_bytes) + " B (limiting class " +
                                        line.limiting_class + ") - aggregate VRAM cannot rescue it");
                    if (d.limiting_term.empty()) d.limiting_term = dev.name + ":" + line.limiting_class;
                }
            }
            // This envelope is independent of the residual client-demand gate above.
            // A measured physical refusal remains verified; absent/invalid evidence does not.
            if (t != nullptr && t->cap_declared) {
                std::string failure;
                std::string limiting;
                bool valid = false;
                if (t->enforced_cap_bytes == 0) {
                    limiting = "declared MPS ceiling is zero";
                    failure = "telemetry declares an MPS ceiling of zero bytes - refuse";
                } else if (t->free_bytes > t->enforced_cap_bytes) {
                    limiting = "MPS residual exceeds declared ceiling";
                    failure = "MPS residual " + std::to_string(t->free_bytes) +
                              " B exceeds declared ceiling " + std::to_string(t->enforced_cap_bytes) +
                              " B - refuse";
                } else if (!t->physical_free_measured) {
                    limiting = "physical telemetry missing";
                    failure = "pre-context physical free memory unmeasured - refuse (fail-closed)";
                } else if (t->outside_client_allowance_bytes == 0) {
                    limiting = "outside-client allowance missing";
                    failure = "declared MPS ceiling requires a positive outside-client allowance - refuse";
                } else {
                    valid = true;
                    if (t->enforced_cap_bytes > t->physical_free_bytes ||
                        t->outside_client_allowance_bytes > t->physical_free_bytes - t->enforced_cap_bytes) {
                        limiting = "physical envelope";
                        failure = "MPS physical envelope overfull: cap " + std::to_string(t->enforced_cap_bytes) +
                                  " B + outside-client allowance " + std::to_string(t->outside_client_allowance_bytes) +
                                  " B > pre-context physical free " + std::to_string(t->physical_free_bytes) + " B";
                    }
                }
                if (!failure.empty()) {
                    line.fits = false;
                    line.slack_bytes = 0;
                    if (!valid) line.has_ceiling = false;
                    if (line.limiting_class.empty()) line.limiting_class = limiting;
                    d.reasons.push_back(dev.name + ": " + failure);
                    if (d.limiting_term.empty()) d.limiting_term = "mps:" + dev.name;
                }
            }
        }
        d.devices.push_back(line);
    }
    if (all_hypothetical && !devices.empty()) {
        d.reasons.push_back("every device is hypothetical - the plan is a labelled projection, not an "
                            "admission");
        if (d.limiting_term.empty()) d.limiting_term = "hypothetical-only";
    }

    // 3. Guest scope: a real limit (AC-2).  File-backed demand is counted; the bounded clean-cache credit
    //    is the only discount, and it can never exceed the counted bytes.
    const uint64_t guest_file = guest_demand.credit_eligible_bytes();
    const uint64_t guest_known = guest_demand.known_bytes();
    const uint64_t guest_credit = std::min(p.file_cache_credit_bytes, guest_file);
    d.guest_known = p.guest_total_measured;
    d.guest_demand_bytes = guest_known - guest_credit;
    d.guest_ceiling_bytes = p.guest_total_measured
                                ? (p.guest_total_bytes >= p.guest_reserve_bytes
                                       ? p.guest_total_bytes - p.guest_reserve_bytes
                                       : 0)
                                : 0;
    if (guest_known > 0) {
        if (!p.guest_total_measured) {
            d.guest_fits = false;
            d.reasons.push_back("guest allocation unmeasured - refuse (fail-closed)");
            if (d.limiting_term.empty()) d.limiting_term = "telemetry:guest";
        } else if (d.guest_demand_bytes > d.guest_ceiling_bytes) {
            d.guest_fits = false;
            d.reasons.push_back("guest overfull: demand " + std::to_string(d.guest_demand_bytes) +
                                " B (after clean-cache credit " + std::to_string(guest_credit) +
                                " B) > guest " + std::to_string(p.guest_total_bytes) + " B - reserve " +
                                std::to_string(p.guest_reserve_bytes) + " B");
            if (d.limiting_term.empty()) d.limiting_term = "guest";
        } else {
            d.guest_fits = true;
        }
    } else {
        d.guest_fits = true;   // nothing demanded of the guest: nothing to refuse
    }

    // 4. Host scope, same shape.  The 8 GiB host reserve is the floor; telemetry missing refuses.
    const uint64_t host_file = host_demand.credit_eligible_bytes();
    const uint64_t host_known = host_demand.known_bytes();
    const uint64_t host_credit = std::min(p.file_cache_credit_bytes, host_file);
    d.host_known = p.host_available_measured;
    d.host_demand_bytes = host_known - host_credit;
    d.host_ceiling_bytes = p.host_available_measured
                               ? (p.host_available_bytes >= p.host_reserve_bytes
                                      ? p.host_available_bytes - p.host_reserve_bytes
                                      : 0)
                               : 0;
    if (host_known > 0) {
        if (!p.host_available_measured) {
            d.host_fits = false;
            d.reasons.push_back("host available memory unmeasured - refuse (fail-closed)");
            if (d.limiting_term.empty()) d.limiting_term = "telemetry:host";
        } else if (d.host_demand_bytes > d.host_ceiling_bytes) {
            d.host_fits = false;
            d.reasons.push_back("host overfull: demand " + std::to_string(d.host_demand_bytes) +
                                " B (after clean-cache credit " + std::to_string(host_credit) +
                                " B) > available " + std::to_string(p.host_available_bytes) +
                                " B - reserve " + std::to_string(p.host_reserve_bytes) + " B");
            if (d.limiting_term.empty()) d.limiting_term = "host";
        } else {
            d.host_fits = true;
        }
    } else {
        d.host_fits = true;
    }

    d.verified = d.unknown_classes.empty() && !all_hypothetical;
    for (const DeviceLine& line : d.devices)
        if (!line.hypothetical && !line.has_ceiling) d.verified = false;
    if (guest_known > 0 && !d.guest_known) d.verified = false;
    if (host_known > 0 && !d.host_known) d.verified = false;

    d.admitted = d.verified && d.reasons.empty() && !all_hypothetical && d.guest_fits && d.host_fits;
    for (const DeviceLine& line : d.devices)
        if (!line.hypothetical && !line.fits) d.admitted = false;
    if (d.admitted) {
        if (d.limiting_term.empty()) d.limiting_term = "none";
        d.reasons.push_back("admitted: every device, the guest and the host close with explicit headroom");
    }
    return d;
}

// ---- per-request admission --------------------------------------------------

/// One active request's state demand, by context length rather than by slot count: the state is fixed per
/// sequence (GDN recurrence + conv + indexer tails), the KV grows with the cells this request can reach.
struct RequestDemand {
    int64_t prompt_tokens = 0;
    int64_t max_new_tokens = 0;
    int64_t cells() const {
        if (prompt_tokens < 0 || max_new_tokens < 0 || max_new_tokens > INT64_MAX - prompt_tokens) return -1;
        return prompt_tokens + max_new_tokens;
    }
    uint64_t kv_bytes_per_token = 0;          // from kv_bytes_per_token(Geometry)
    uint64_t state_bytes_per_sequence = 0;    // from state_bytes(Geometry)
    uint64_t total_state_bytes() const {
        if (cells() < 0) return 0;
        return saturating_add(state_bytes_per_sequence,
                              saturating_multiply((uint64_t) cells(), kv_bytes_per_token));
    }
};

struct RequestDecision {
    bool admitted = false;
    int64_t cells = 0;
    int64_t context_limit = 0;
    uint64_t state_demand_bytes = 0;
    std::string reason;
};

/// The runtime check: a request whose reachable cells do not fit the admitted context is refused BEFORE
/// any prompt work, with the arithmetic in the refusal.  `margin` is the engine's stop-token allowance
/// (the serving path uses 8; the caller passes its own convention).
inline RequestDecision admit_request(const RequestDemand& req, int64_t context_limit, int64_t margin = 8) {
    RequestDecision r;
    r.cells = req.cells();
    r.context_limit = context_limit;
    r.state_demand_bytes = req.total_state_bytes();
    if (r.cells < 0 || margin < 0 || context_limit < 0 ||
        (req.kv_bytes_per_token != 0 && (uint64_t) r.cells >
         (UINT64_MAX - req.state_bytes_per_sequence) / req.kv_bytes_per_token)) {
        r.reason = "invalid or overflowing request state demand";
        return r;
    }
    if (margin > context_limit || r.cells > context_limit - margin) {
        r.reason = "prompt (" + std::to_string(req.prompt_tokens) + " tokens) + max_new (" +
                   std::to_string(req.max_new_tokens) + ") exceeds the context (" +
                   std::to_string(context_limit) + ")";
        return r;
    }
    r.admitted = true;
    return r;
}

// ---- predicted vs observed (AC-3) -------------------------------------------

struct PeakComparison {
    uint64_t predicted_bytes = 0;
    uint64_t observed_bytes = 0;
    uint64_t delta_bytes = 0;          // observed - predicted (saturating; 0 when observed is smaller)
    uint64_t tolerance_bytes = 0;
    bool within_tolerance = false;
    bool explained = false;
    std::string explanation;
    bool enlargement_allowed = false;
};

/// The explain-or-shrink rule: an observed peak above prediction + tolerance BLOCKS enlargement until an
/// explanation is recorded.  Nothing here edits a configuration; it produces the verdict the operator
/// (or the harness) must honour before raising any size.
inline PeakComparison compare_peaks(uint64_t predicted_bytes, uint64_t observed_bytes,
                                    const GuestHostPolicy& policy, std::string explanation = "") {
    PeakComparison c;
    c.predicted_bytes = predicted_bytes;
    c.observed_bytes = observed_bytes;
    c.tolerance_bytes = policy.comparison_tolerance_bytes;
    c.delta_bytes = observed_bytes > predicted_bytes ? observed_bytes - predicted_bytes : 0;
    c.within_tolerance = c.delta_bytes <= c.tolerance_bytes;
    c.explained = c.within_tolerance || !explanation.empty();
    c.explanation = explanation;
    c.enlargement_allowed = c.explained;
    if (!c.explained)
        c.explanation = "observed peak exceeds prediction by " + std::to_string(c.delta_bytes) +
                        " B with no explanation - resolve (or shrink the admitted sizes) before enlarging";
    return c;
}

// ---- JSON -------------------------------------------------------------------
// A tiny deterministic emitter (no dependencies): fixed key order, bytes AND GiB for every figure, so a
// plan diff is a text diff.  `schemas/memory-plan.schema.json` describes the document the CLI assembles
// around this decision block.

inline std::string json_escape(const std::string& s) {
    std::string out;
    for (char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if ((unsigned char) c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof buf, "\\u%04x", (unsigned) (unsigned char) c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

inline double gib(uint64_t bytes) { return (double) bytes / 1073741824.0; }

// `fmt_gib`: two decimals, no locale - the plan is machine-diffed.
inline std::string num2(double v) {
    std::ostringstream o;
    o.precision(2);
    o << std::fixed << v;
    return o.str();
}

inline void emit_accounted(std::ostringstream& o, const Accounted& c, const std::string& pad) {
    o << pad << "{\"name\": \"" << json_escape(c.name) << "\", \"bytes\": " << c.bytes
      << ", \"gib\": " << num2(gib(c.bytes)) << ", \"unknown\": " << (c.unknown ? "true" : "false")
      << ", \"source\": \"" << json_escape(c.source) << "\", \"meaning\": \"" << byte_meaning_name(c.meaning)
      << "\", \"host_mirror\": " << (c.mirrors_host() ? "true" : "false")
      << ", \"clean_credit\": " << (c.clean_credit() ? "true" : "false") << "}";
}

inline void emit_scope(std::ostringstream& o, const Scope& s, const std::string& pad) {
    o << pad << "\"classes\": [\n";
    for (size_t i = 0; i < s.classes.size(); ++i) {
        emit_accounted(o, s.classes[i], pad + "  ");
        if (i + 1 < s.classes.size()) o << ",";
        o << "\n";
    }
    o << pad << "],\n"
      << pad << "\"known_bytes\": " << s.known_bytes() << ",\n"
      << pad << "\"known_gib\": " << num2(gib(s.known_bytes())) << "\n";
}

inline std::string admission_json(const AdmissionDecision& d, const std::vector<DeviceCost>& costs,
                                  const Scope& guest_demand, const Scope& host_demand,
                                  const GuestHostPolicy& policy, const std::string& indent = "  ") {
    std::ostringstream o;
    o << indent << "\"decision\": {\n"
      << indent << "  \"admitted\": " << (d.admitted ? "true" : "false") << ",\n"
      << indent << "  \"verified\": " << (d.verified ? "true" : "false") << ",\n"
      << indent << "  \"variant\": \"" << json_escape(d.variant) << "\",\n"
      << indent << "  \"limiting_term\": \"" << json_escape(d.limiting_term) << "\",\n";
    if (d.unknown_classes.empty()) o << indent << "  \"unknown_classes\": [],\n";
    else {
        o << indent << "  \"unknown_classes\": [";
        for (size_t i = 0; i < d.unknown_classes.size(); ++i)
            o << "\"" << json_escape(d.unknown_classes[i]) << (i + 1 < d.unknown_classes.size() ? "\", " : "\"");
        o << "],\n";
    }
    o << indent << "  \"reasons\": [";
    for (size_t i = 0; i < d.reasons.size(); ++i)
        o << "\"" << json_escape(d.reasons[i]) << (i + 1 < d.reasons.size() ? "\", " : "\"");
    o << "],\n";
    // per-device: the accounting CLASSES come from `costs` (the caller's composition), the VERDICT from `d`
    o << indent << "  \"devices\": [\n";
    for (size_t i = 0; i < d.devices.size(); ++i) {
        const DeviceLine& line = d.devices[i];
        o << indent << "    {\n"
          << indent << "      \"name\": \"" << json_escape(line.name) << "\",\n"
          << indent << "      \"hypothetical\": " << (line.hypothetical ? "true" : "false") << ",\n"
          << indent << "      \"has_ceiling\": " << (line.has_ceiling ? "true" : "false") << ",\n"
          << indent << "      \"ceiling_bytes\": " << line.ceiling_bytes << ",\n"
          << indent << "      \"ceiling_gib\": " << num2(gib(line.ceiling_bytes)) << ",\n"
          // the ceiling the telemetry was taken under: a fixed MPS client budget or plain
          // device free memory. Emitted so the document and the Python mirror agree.
          << indent << "      \"cap_declared\": " << (line.cap_declared ? "true" : "false") << ",\n"
          << indent << "      \"enforced_cap_bytes\": " << line.enforced_cap_bytes << ",\n"
          << indent << "      \"physical_free_measured\": " << (line.physical_free_measured ? "true" : "false") << ",\n"
          << indent << "      \"physical_free_bytes\": " << line.physical_free_bytes << ",\n"
          << indent << "      \"outside_client_allowance_bytes\": " << line.outside_client_allowance_bytes << ",\n"
          << indent << "      \"demand_bytes\": " << line.demand_bytes << ",\n"
          << indent << "      \"demand_gib\": " << num2(gib(line.demand_bytes)) << ",\n"
          << indent << "      \"headroom_bytes\": " << line.headroom_bytes << ",\n"
          << indent << "      \"slack_bytes\": " << line.slack_bytes << ",\n"
          << indent << "      \"fits\": " << (line.fits ? "true" : "false") << ",\n"
          << indent << "      \"limiting_class\": \"" << json_escape(line.limiting_class) << "\",\n";
        // classes of the matching DeviceCost
        const DeviceCost* cost = nullptr;
        for (const DeviceCost& c : costs)
            if (c.name == line.name) { cost = &c; break; }
        if (cost != nullptr) emit_scope(o, cost->vram, indent + "      ");
        else o << indent << "      \"classes\": [],\n" << indent << "      \"known_bytes\": 0,\n"
               << indent << "      \"known_gib\": 0.00\n";
        o << indent << "    }" << (i + 1 < d.devices.size() ? "," : "") << "\n";
    }
    o << indent << "  ],\n"
      << indent << "  \"guest\": {\n"
      << indent << "    \"known\": " << (d.guest_known ? "true" : "false") << ",\n"
      << indent << "    \"fits\": " << (d.guest_fits ? "true" : "false") << ",\n"
      << indent << "    \"demand_bytes\": " << d.guest_demand_bytes << ",\n"
      << indent << "    \"demand_gib\": " << num2(gib(d.guest_demand_bytes)) << ",\n"
      << indent << "    \"usable_bytes\": " << d.guest_ceiling_bytes << ",\n"
      << indent << "    \"usable_gib\": " << num2(gib(d.guest_ceiling_bytes)) << ",\n";
    emit_scope(o, guest_demand, indent + "    ");
    o << indent << "  },\n"
      << indent << "  \"host\": {\n"
      << indent << "    \"known\": " << (d.host_known ? "true" : "false") << ",\n"
      << indent << "    \"fits\": " << (d.host_fits ? "true" : "false") << ",\n"
      << indent << "    \"demand_bytes\": " << d.host_demand_bytes << ",\n"
      << indent << "    \"demand_gib\": " << num2(gib(d.host_demand_bytes)) << ",\n"
      << indent << "    \"usable_bytes\": " << d.host_ceiling_bytes << ",\n"
      << indent << "    \"usable_gib\": " << num2(gib(d.host_ceiling_bytes)) << ",\n";
    emit_scope(o, host_demand, indent + "    ");
    o << indent << "  },\n"
      << indent << "  \"policy\": {\n"
      << indent << "    \"guest_reserve_bytes\": " << d.applied_guest_reserve_bytes << ",\n"
      << indent << "    \"guest_reserve_gib\": " << num2(gib(d.applied_guest_reserve_bytes)) << ",\n"
      << indent << "    \"host_reserve_bytes\": " << d.applied_host_reserve_bytes << ",\n"
      << indent << "    \"host_reserve_gib\": " << num2(gib(d.applied_host_reserve_bytes)) << ",\n"
      << indent << "    \"file_cache_credit_bytes\": " << policy.file_cache_credit_bytes << ",\n"
      << indent << "    \"comparison_tolerance_bytes\": " << policy.comparison_tolerance_bytes << "\n"
      << indent << "  }\n"
      << indent << "}";
    return o.str();
}

} // namespace strata::plan
