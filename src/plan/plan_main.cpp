// src/plan/plan_main.cpp - the `strata-plan` CLI: print the memory plan, or say why it does not close.
//
// The engine must adapt to the GPU it is running on. This CLI accepts explicit measured byte inputs;
// --device NAME:TOTAL without FREE is a labelled capacity-only dry run, not a live free-memory probe.
// Host-only arithmetic permits boundary testing without a GPU.
//
// HET-017: the same arithmetic now also admits a whole configuration - per device (the memory equation of
// docs/02, enforced per card with explicit headroom), and against the guest allocation and the host reserve.
// The costs come in as caller-supplied figures exactly as before (`--dense`, `--device-class`, ...); the
// project's `memory_planner/` derives them from the pinned artifact.  `--json` emits the plan document that
// `schemas/memory-plan.schema.json` describes, so a harness can diff plans and feed the decision onward.
// The legacy single-pool invocation is unchanged: same flags, same stdout, same exit codes.
#include "strata/plan/admission.hpp"
#include "strata/plan/plan.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

using strata::plan::DeviceCost;
using strata::plan::DeviceTelemetry;

namespace {

struct Cli {
    uint64_t ctx = 20480;
    uint64_t free_vram = 12ull * 1000 * 1000 * 1000; // RTX 5070 nominal; overridden by --vram
    strata::plan::Costs costs;
    bool state_given = false;
    bool json = false;
    std::string variant = "cold";
    std::vector<strata::plan::DeviceCost> devices;
    std::vector<strata::plan::DeviceTelemetry> telemetry;
    strata::plan::Scope guest, host;
    strata::plan::GuestHostPolicy policy;
    // per-request admission (all zero -> not requested)
    int64_t request_prompt = -1, request_max_new = -1;

    strata::plan::DeviceCost* device(const std::string& name, bool create) {
        for (auto& d : devices)
            if (d.name == name) return &d;
        if (!create) return nullptr;
        devices.push_back(strata::plan::DeviceCost{});
        devices.back().name = name;
        return &devices.back();
    }
    // `NAME` or `NAME:TOTAL_BYTES` or `NAME:TOTAL_BYTES:FREE_BYTES`
    bool add_device(const std::string& spec, bool hypothetical) {
        std::vector<std::string> parts;
        size_t at = 0;
        for (;;) {
            const size_t colon = spec.find(':', at);
            parts.push_back(spec.substr(at, colon == std::string::npos ? std::string::npos : colon - at));
            if (colon == std::string::npos) break;
            at = colon + 1;
        }
        if (parts.empty() || parts[0].empty() || parts.size() > 3) return false;
        DeviceCost* d = device(parts[0], true);
        d->hypothetical = hypothetical;
        DeviceTelemetry t;
        t.name = parts[0];
        if (hypothetical) { telemetry.push_back(t); return true; }
        if (parts.size() >= 2) {
            char* end = nullptr;
            t.total_bytes = std::strtoull(parts[1].c_str(), &end, 10);
            if (end == nullptr || *end != '\0' || t.total_bytes == 0) return false;
            t.measured = true;
            t.free_bytes = t.total_bytes; // capacity-only dry run assumes an otherwise idle device
        }
        if (parts.size() == 3) {
            char* end = nullptr;
            t.free_bytes = std::strtoull(parts[2].c_str(), &end, 10);
            if (end == nullptr || *end != '\0') return false;
        }
        for (const auto& seen : telemetry)
            if (seen.name == t.name) return false;
        telemetry.push_back(t);
        return true;
    }
    // `DEV:NAME=BYTES` for devices, `NAME=BYTES` for guest/host scopes
    bool add_class(const std::string& spec, const std::string& prefix) {
        const size_t eq = spec.rfind('=');
        if (eq == std::string::npos || eq + 1 >= spec.size()) return false;
        const std::string left = spec.substr(0, eq);
        char* end = nullptr;
        const uint64_t bytes = std::strtoull(spec.c_str() + eq + 1, &end, 10);
        if (end == nullptr || *end != '\0') return false;
        if (prefix == "device") {
            const size_t colon = left.find(':');
            if (colon == std::string::npos) return false;
            DeviceCost* d = device(left.substr(0, colon), true);
            d->vram.add(left.substr(colon + 1), bytes, "CLI --device-class");
        } else if (prefix == "guest") {
            guest.add(left, bytes, "CLI --guest-class");
        } else {
            host.add(left, bytes, "CLI --host-class");
        }
        return true;
    }
};

int usage() {
    std::printf(
        "usage: strata-plan [--max-context N] [--vram BYTES] [--dense B] [--embd B]\n"
        "                   [--workspace B] [--state B] [--pool B]\n"
        "  --state defaults to the geometry's own GDN recurrence + conv history, NOT to 0:\n"
        "  the flag used to default to zero and nothing passed it, so the planner silently\n"
        "  planned without 117.7 MB (85 expert slots) of non-evictable state.\n"
        "\n"
        "HET-017 admission (all optional; any of these switches on the per-device decision):\n"
        "  --json                          emit the memory-plan document (schemas/memory-plan.schema.json)\n"
        "  --device NAME:TOTAL[:FREE]      a measured device ceiling in driver-visible bytes (repeatable)\n"
        "  --hypothetical NAME             a labelled future capacity: reported, never admitted\n"
        "  --device-class DEV:NAME=B       a byte class one device must hold (repeatable)\n"
        "  --headroom [DEV:]B              explicit headroom H in the memory equation (per device or all)\n"
        "  --guest-total B                 the measured guest allocation\n"
        "  --guest-class NAME=B            guest RAM demand (file-backed: prefix the name with \"file:\")\n"
        "  --guest-reserve B               >= 4 GiB; smaller values are floored to the design default\n"
        "  --host-available B              measured physical-host MemAvailable (incremental host scope)\n"
        "  --host-class NAME=B             host RAM demand (same \"file:\" convention)\n"
        "  --host-reserve B                >= 8 GiB; smaller values are floored to the design default\n"
        "  --file-cache-credit B           the ONLY discount for file-backed demand: the clean-cache bytes\n"
        "                                  measured through the cgroup-v2 counters (never shared/dirty)\n"
        "  --comparison-tolerance B        observed-above-predicted slack before explain-or-shrink (256 MiB)\n"
        "  --variant cold|warm|resident    label only: the caller composes the classes each variant holds\n"
        "  --request-prompt N --request-max-new N\n"
        "                                  admit one active request by its total state demand, not a slot count\n");
    return 2;
}

std::string plan_json(const strata::plan::Plan* p, const Cli& cli,
                      const strata::plan::AdmissionDecision* d, const strata::plan::RequestDecision* r,
                      bool does_not_close, const std::string& why) {
    std::ostringstream o;
    o << "{\n"
      << "  \"schema_version\": 1,\n"
      << "  \"execution_status\": \"executed\",\n"
      << "  \"generated_by\": \"strata-plan\",\n"
      << "  \"variant\": \"" << strata::plan::json_escape(cli.variant) << "\",\n"
      << "  \"max_context\": " << cli.ctx << ",\n";
    if (p != nullptr) {
        o << "  \"plan\": {\n"
          << "    \"kv_bytes\": " << p->kv_bytes << ", \"kv_gib\": " << strata::plan::num2(strata::plan::gib(p->kv_bytes)) << ",\n"
          << "    \"state_bytes\": " << p->state_bytes << ", \"state_gib\": " << strata::plan::num2(strata::plan::gib(p->state_bytes)) << ",\n"
          << "    \"cache_bytes\": " << p->cache_bytes << ", \"cache_gib\": " << strata::plan::num2(strata::plan::gib(p->cache_bytes)) << ",\n"
          << "    \"cache_slots\": " << p->cache_slots << ",\n"
          << "    \"vram_budget_bytes\": " << p->vram_budget << ", \"vram_used_bytes\": " << p->vram_used << ",\n"
          << "    \"dram_experts_bytes\": " << p->dram_experts << "\n"
          << "  },\n";
    } else {
        o << "  \"plan\": null,\n";
    }
    if (r != nullptr) {
        o << "  \"request\": {\n"
          << "    \"prompt_tokens\": " << cli.request_prompt << ",\n"
          << "    \"max_new_tokens\": " << cli.request_max_new << ",\n"
          << "    \"cells\": " << r->cells << ",\n"
          << "    \"context_limit\": " << r->context_limit << ",\n"
          << "    \"state_demand_bytes\": " << r->state_demand_bytes << ",\n"
          << "    \"state_demand_gib\": " << strata::plan::num2(strata::plan::gib(r->state_demand_bytes)) << ",\n"
          << "    \"admitted\": " << (r->admitted ? "true" : "false") << ",\n"
          << "    \"reason\": \"" << strata::plan::json_escape(r->reason) << "\"\n"
          << "  },\n";
    }
    if (d != nullptr) {
        o << strata::plan::admission_json(*d, cli.devices, cli.guest, cli.host, cli.policy) << ",\n";
    } else {
        o << "  \"decision\": null,\n";
    }
    o << "  \"does_not_close\": " << (does_not_close ? "true" : "false") << ",\n"
      << "  \"notes\": [";
    if (!why.empty()) o << "\"" << strata::plan::json_escape(why) << "\"";
    o << "]\n"
      << "}\n";
    return o.str();
}

}  // namespace

int main(int argc, char** argv) {
    Cli cli;
    bool state_given = false;
    bool admission_given = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](uint64_t& out) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs a value\n", a.c_str());
                std::exit(2);
            }
            out = std::strtoull(argv[++i], nullptr, 10);
        };
        if (a == "--max-context") next(cli.ctx);
        else if (a == "--vram") next(cli.free_vram);
        else if (a == "--dense") next(cli.costs.dense_bytes);
        else if (a == "--embd") next(cli.costs.embd_bytes);
        else if (a == "--workspace") next(cli.costs.workspace_bytes);
        else if (a == "--state") { next(cli.costs.state_bytes); state_given = true; }
        else if (a == "--pool") { uint64_t v = 0; next(v); cli.free_vram = v; }
        else if (a == "--json") { cli.json = true; admission_given = true; }
        else if (a == "--variant") {
            if (i + 1 >= argc) return usage();
            cli.variant = argv[++i];
            if (cli.variant != "cold" && cli.variant != "warm" && cli.variant != "resident") return usage();
            admission_given = true;
        } else if (a == "--device") {
            if (i + 1 >= argc) return usage();
            if (!cli.add_device(argv[++i], /*hypothetical=*/false)) return usage();
            admission_given = true;
        } else if (a == "--hypothetical") {
            if (i + 1 >= argc) return usage();
            if (!cli.add_device(argv[++i], /*hypothetical=*/true)) return usage();
            admission_given = true;
        } else if (a == "--device-class") {
            if (i + 1 >= argc || !cli.add_class(argv[++i], "device")) return usage();
            admission_given = true;
        } else if (a == "--guest-class") {
            if (i + 1 >= argc || !cli.add_class(argv[++i], "guest")) return usage();
            admission_given = true;
        } else if (a == "--host-class") {
            if (i + 1 >= argc || !cli.add_class(argv[++i], "host")) return usage();
            admission_given = true;
        } else if (a == "--headroom") {
            if (i + 1 >= argc) return usage();
            const std::string spec = argv[++i];
            const size_t colon = spec.find(':');
            const uint64_t bytes = std::strtoull(spec.c_str() + (colon == std::string::npos ? 0 : colon + 1),
                                                 nullptr, 10);
            if (colon == std::string::npos) {
                for (auto& d : cli.devices) d.headroom_bytes = bytes;
            } else {
                DeviceCost* d = cli.device(spec.substr(0, colon), true);
                d->headroom_bytes = bytes;
            }
            admission_given = true;
        } else if (a == "--guest-total") { next(cli.policy.guest_total_bytes); cli.policy.guest_total_measured = true; admission_given = true; }
        else if (a == "--guest-reserve") { next(cli.policy.guest_reserve_bytes); admission_given = true; }
        else if (a == "--host-available") { next(cli.policy.host_available_bytes); cli.policy.host_available_measured = true; admission_given = true; }
        else if (a == "--host-reserve") { next(cli.policy.host_reserve_bytes); admission_given = true; }
        else if (a == "--file-cache-credit") { next(cli.policy.file_cache_credit_bytes); admission_given = true; }
        else if (a == "--comparison-tolerance") { next(cli.policy.comparison_tolerance_bytes); }
        else if (a == "--request-prompt") { uint64_t v = 0; next(v); cli.request_prompt = (int64_t) v; admission_given = true; }
        else if (a == "--request-max-new") { uint64_t v = 0; next(v); cli.request_max_new = (int64_t) v; admission_given = true; }
        else if (a == "--help" || a == "-h") { usage(); return 0; }
        else { std::fprintf(stderr, "unknown argument: %s\n", a.c_str()); return 2; }
    }
    // DEFAULT THE STATE FROM THE GEOMETRY.  A `--state` that defaults to 0 and is never passed is not a
    // default, it is an omission - and this one was worth 85 cache slots of overcommit.
    if (!state_given) cli.costs.state_bytes = strata::plan::state_bytes(strata::plan::Geometry{});

    // ---- the legacy single-pool plan (unchanged arithmetic) ------------------
    const uint64_t pool = cli.free_vram > cli.costs.dense_bytes + cli.costs.embd_bytes
                              ? cli.free_vram - cli.costs.dense_bytes - cli.costs.embd_bytes
                              : 0;
    const uint64_t pool_used = pool ? pool : strata::plan::vram_pool_bytes();

    strata::plan::RequestDecision request;
    bool have_request = cli.request_prompt >= 0 && cli.request_max_new >= 0;
    if (have_request) {
        strata::plan::RequestDemand req;
        req.prompt_tokens = cli.request_prompt;
        req.max_new_tokens = cli.request_max_new;
        req.kv_bytes_per_token = strata::plan::kv_bytes_per_token(strata::plan::Geometry{});
        req.state_bytes_per_sequence = cli.costs.state_bytes;
        request = strata::plan::admit_request(req, (int64_t) cli.ctx);
    }

    // ---- admission -----------------------------------------------------------
    // Every device named via --device/--hypothetical/--device-class takes part; a configuration with
    // devices on the command line is decided per device, the legacy --vram/--pool figure keeps feeding
    // the single-pool plan exactly as before.
    strata::plan::AdmissionDecision decision;
    // A request admission on its own is a complete decision (against the geometry), so it does not need
    // device or scope flags to be reachable.
    const bool have_scope = !cli.devices.empty() || !cli.guest.classes.empty() || !cli.host.classes.empty();
    const bool have_admission = (admission_given && have_scope) || have_request;
    if (have_scope && (admission_given || have_request)) {
        // When the caller supplied no explicit per-device classes, the single-pool plan's fixed costs
        // ARE the primary device's accounting - one source of bytes, not two.
        bool any_device_class = false;
        for (const auto& d : cli.devices)
            if (!d.vram.classes.empty()) any_device_class = true;
        if (!any_device_class && !cli.devices.empty()) {
            strata::plan::Scope vram;
            if (cli.costs.dense_bytes) vram.add("weights_canonical", cli.costs.dense_bytes, "--dense");
            if (cli.costs.embd_bytes) vram.add("weights_embd", cli.costs.embd_bytes, "--embd");
            if (cli.costs.mtp_bytes) vram.add("mtp_weights", cli.costs.mtp_bytes, "--mtp");
            if (cli.costs.workspace_bytes) vram.add("graph_workspace", cli.costs.workspace_bytes, "--workspace");
            if (cli.costs.state_bytes) vram.add("recurrent_state", cli.costs.state_bytes, "--state / geometry");
            vram.add("kv_at_max_context", strata::plan::kv_bytes_per_token(strata::plan::Geometry{}) * cli.ctx,
                     "kv_bytes_per_token(geometry) * --max-context");
            for (auto& d : cli.devices)
                if (!d.hypothetical) { d.vram = vram; break; }   // the first real device carries the pool
        }
        decision = strata::plan::admit_configuration(cli.devices, cli.telemetry, cli.guest, cli.host,
                                                     cli.policy, cli.variant);
    }
    if (have_request && !request.admitted) {
        decision.admitted = false;
        decision.reasons.push_back("request refused: " + request.reason);
        if (decision.limiting_term.empty()) decision.limiting_term = "request:context";
    }

    if (cli.json) {
        strata::plan::Plan legacy;
        bool closed = true;
        std::string why;
        try {
            legacy = strata::plan::make_plan(cli.ctx, strata::plan::Geometry{}, cli.costs, pool_used);
        } catch (const strata::plan::DoesNotClose& e) {
            closed = false;
            why = e.what();
        }
        const std::string doc = plan_json(closed ? &legacy : nullptr, cli,
                                          have_admission ? &decision : nullptr,
                                          have_request ? &request : nullptr, !closed, why);
        std::printf("%s", doc.c_str());
        const bool ok = closed && (!have_admission || decision.admitted) && (!have_request || request.admitted);
        return ok ? 0 : 1;
    }

    if (have_admission) {
        bool closed = true;
        std::string why;
        try {
            const strata::plan::Plan legacy =
                strata::plan::make_plan(cli.ctx, strata::plan::Geometry{}, cli.costs, pool_used);
            std::printf("%s", strata::plan::to_string(legacy).c_str());
        } catch (const strata::plan::DoesNotClose& e) {
            closed = false;
            why = e.what();
            std::fprintf(stderr, "strata-plan: %s\n", e.what());
        }
        std::printf("admission: %s (variant %s, verified %s), limiting term: %s\n",
                    decision.admitted ? "ADMIT" : "REFUSE", decision.variant.c_str(),
                    decision.verified ? "yes" : "no", decision.limiting_term.c_str());
        for (const auto& line : decision.devices)
            std::printf("  %-12s %s ceiling %.2f GiB demand %.2f GiB headroom %.2f GiB slack %.2f GiB%s%s\n",
                        line.name.c_str(), line.hypothetical ? "hypothetical" : (line.fits ? "fits" : "OVERFULL"),
                        strata::plan::gib(line.ceiling_bytes), strata::plan::gib(line.demand_bytes),
                        strata::plan::gib(line.headroom_bytes), strata::plan::gib(line.slack_bytes),
                        line.limiting_class.empty() ? "" : "  limiting: ", line.limiting_class.c_str());
        std::printf("  guest %s: demand %.2f GiB of %.2f GiB usable;  host %s: demand %.2f GiB of %.2f GiB usable\n",
                    decision.guest_known ? (decision.guest_fits ? "fits" : "OVERFULL") : "unmeasured",
                    strata::plan::gib(decision.guest_demand_bytes), strata::plan::gib(decision.guest_ceiling_bytes),
                    decision.host_known ? (decision.host_fits ? "fits" : "OVERFULL") : "unmeasured",
                    strata::plan::gib(decision.host_demand_bytes), strata::plan::gib(decision.host_ceiling_bytes));
        for (const auto& r : decision.reasons) std::printf("  reason: %s\n", r.c_str());
        if (have_request)
            std::printf("  request %lld+%lld cells: %s (%.3f GiB state demand)\n",
                        (long long) cli.request_prompt, (long long) cli.request_max_new,
                        request.admitted ? "ADMIT" : "REFUSE", strata::plan::gib(request.state_demand_bytes));
        return closed && decision.admitted && (!have_request || request.admitted) ? 0 : 1;
    }

    try {
        const strata::plan::Plan p = strata::plan::make_plan(cli.ctx, strata::plan::Geometry{}, cli.costs,
                                                             pool_used);
        std::printf("%s", strata::plan::to_string(p).c_str());
        return 0;
    } catch (const strata::plan::DoesNotClose& e) {
        std::fprintf(stderr, "strata-plan: %s\n", e.what());
        return 1;
    }
}
