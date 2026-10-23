// Compile production predicates/width selection, not a second policy or a GPU owner.
// Host records supply only the fields those fragments read; no kernels or model run.
#include "strata/core/dflash2_bridge.hpp"
#include "strata/core/draft_state.hpp"
#include "strata/kernels/dflash2_selector.hpp"
#include "strata/spec/draft_policy.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>

namespace {
namespace core = strata::core;
namespace df = core::dflash2;
namespace kernels = strata::kernels;
using namespace df;
using Clock = std::chrono::steady_clock;
void require(bool okay, const char* message) {
    if (!okay) throw std::runtime_error(message);
}
bool refuse(std::string& error, const char* reason) { error = reason; return false; }
struct Sampling { bool greedy = true; float temperature = 1; };
struct NativeContext { int frontier; int end() const { return frontier; } };
struct NativeRequest {
    NativeContext* context;
    core::DraftOwner uniform_owner;
    int uniform_frontier = -1;
    std::array<float, 3> uniforms{};
    bool uniforms_valid = false;
};
bool install_uniforms(NativeRequest* request, const core::DraftOwner& expected,
        const core::DraftOwner& current, int frontier, const std::array<float, 3>& uniforms,
        std::string& error, int slot = 0) {
    const auto owns = [&](const core::DraftOwner& owner) { return owner == current; };
#include "dflash2_uniform_guard.inc"
    return true;
}
bool native_preflight(NativeRequest* request, const core::DraftOwner& expected,
        const core::DraftCoverage& coverage, const Sampling& sampling, int64_t cell,
        int32_t anchor, std::string& error) {
    const int frontier = int(cell + 1);
#include "dflash2_proposal_guard.inc"
    return true;
}
auto constructed_ids(int32_t anchor) {
#include "dflash2_constructed_ids.inc"
    return std::pair{rows + 1, ids};
}

// The real forward preflight runs against committed-context metadata and spans.
struct DeviceContext {
    struct Impl {
        uint64_t owner = 1;
        int device = 0, origin = 0, begin = 0, end = 0;
        bool poisoned = false, initialized = true;
        std::array<char, 128> request{};
        size_t request_size = 0;
        int64_t incarnation = 0, last_generation = 0;
    };
};
#include "dflash2_request_id.inc"
struct ForwardPreflight {
    uint64_t identity = 1;
    int device = 0;
#include "dflash2_forward_owner.inc"
#include "dflash2_forward_features.inc"
    bool accepts(const DeviceContext::Impl& c, const FeatureView& features, int frontier,
                 int proposal_rows, const void* embeddings, std::string& error) const {
        struct { void complete() {} } guard;
#include "dflash2_forward_extent.inc"
        return true;
    }
};
bool public_rows(int rows, const void* embeddings, const void* output, std::string& error) {
#include "dflash2_forward_public_guard.inc"
    return true;
}

struct Proposal {
    core::DraftOwner owner;
    int64_t frontier = 0;
    int count = 0;
    std::array<int32_t, 7> tokens{};
    std::array<float, 7> confidence{};
};
struct BSlot {
    Proposal proposal;
    Sampling sampling;
    int64_t p = 1, max_new = 16, produced = 0;
    bool stop = false;
    struct { double spec_min_probability = 0.5; } input;
};
struct Options {
    int dflash2_max_proposals = 1, spec_corrupt = 0, suffix_draft = 1;
    int64_t max_context = max_position;
};
bool resident_attempt(const BSlot& state, const Options& o, int S_mtp, int& calls) {
#include "dflash2_resident_gate.inc"
    ++calls;
    return true;
}
bool resident_final_attempt(const BSlot& state, int64_t context_, int& calls) {
    const struct { core::DraftKind kind() const { return core::DraftKind::dflash2; } } draft_;
#include "dflash2_resident_final_gate.inc"
    ++calls;
    return true;
}
struct SerialAdmission {
    Options o;
    bool first_window = true, native_first_block = true, admit_first_block = false;
    bool dflash = true, pipe = false, terminal = false, admission_yield = false;
    int admit_slot = -1, merge_T = 0, S_mtp = 8, keep = 1;
    int64_t p = 1, max_new = 16, produced_n = 0;
    struct { bool images = false; } input;
    struct { bool stopped = false; bool load() const { return stopped; } } stop_req;
    void initial(int& calls) const {
#include "dflash2_initial_gate.inc"
        ++calls;
        }
    }
    void continuation(int& calls) const {
#include "dflash2_continuation_gate.inc"
        ++calls;
        }
    }
};
int resident_width(const BSlot& slot, int cap, int S_mtp = 8, int64_t remaining = 16) {
    const bool use_dflash = true;
    Options o; o.dflash2_max_proposals = cap; o.max_context = slot.p + remaining;
#include "dflash2_resident_width.inc"
    return neural_row_width(slot);
}
struct Widths { int selected, total, prefix, competition; bool lookup; };
Widths serial_width(int count, int cap, bool sampled = false, float confidence = 1,
        bool replace = false, int chain_n = 0, int S_mtp = 8, int64_t remaining = 16,
        int32_t first_neural = 11) {
    const bool use_dflash = true, use_neural = true, sampled_request = sampled;
    const bool first_window = false, initial_native_proposal = false, lookup_trace = false;
    const int merge_T = 0, S = 8;
    const double req_spec_min_p = 0.5;
    const int64_t p = 1, max_new = remaining, produced_n = 0;
    Options o; o.dflash2_max_proposals = cap;
    Proposal serial_proposal; serial_proposal.count = count;
    std::array<float, 7> dprob; dprob.fill(confidence);
    std::array<int32_t, 7> sbuf{}, drafts{}; drafts[0] = first_neural;
    struct {
        int propose(int, int32_t* out) const { out[0] = 11; return 7; }
        int last_match() const { return 20; }
    } sfx;
    struct {
        bool replace;
        int seen = -1;
        strata::spec::DraftPolicy::Pick choose(int ceiling, int, int) {
            seen = ceiling; return {replace, 7};
        }
    } policy{replace};
    do {
#include "dflash2_serial_width.inc"
        const int selected = T;
#include "dflash2_mixed_width.inc"
        return {selected, T, T_mtp, policy.seen, from_sfx};
    } while (false);
    throw std::runtime_error("no target rows remain");
}

namespace io {
void require(bool okay, const char* message) { ::require(okay, message); }
}
struct Capture {
    core::DraftOwner owner{0, 1, 0, 1};
    Proposal proposal;
    int origin = 0, frontier = 0, anchor = 1, device = 0;
    const void *raw_embeddings_device = this, *hidden_device = this, *logits_device = this;
};
void observer_range(const Capture& c) {
#include "dflash2_observer_range.inc"
}
int bounded(int value, int lower, int upper, const char* message) {
    require(value >= lower && value <= upper, message); return value;
}
auto replay_range(const std::map<std::string, int>& bundle) {
#include "dflash2_replay_range.inc"
    return std::pair{origin, frontier};
}
template<class F> bool refuses(F action) {
    try { action(); return false; } catch (const std::runtime_error&) { return true; }
}

void frontier_cases() {
    const core::DraftOwner owner{0, 1, 0, 2};
    core::DraftCoverage coverage; coverage.present = true;
    std::string error;
    for (int positions = 1; positions <= 7; ++positions) {
        require(set_draft_positions(positions), "startup construction width");
        const auto [rows, ids] = constructed_ids(17);
        require(rows == positions + 1 && ids[0] == 17, "actual anchor plus D construction");
        for (int row = 1; row < rows; ++row) require(ids[size_t(row)] == mask_token, "actual mask rows");
        for (int beyond = 0; beyond <= 1; ++beyond) {
            const int frontier = max_position - rows + beyond;
            NativeContext context{frontier}; NativeRequest request{&context};
            require(install_uniforms(&request, owner, owner, frontier, {0, 0.5f, 0.99f}, error) == !beyond,
                    "uniform endpoint uses constructed D+1");
            require(request.uniforms_valid == !beyond, "refused uniforms are not published");
            require(native_preflight(&request, owner, coverage, {}, frontier - 1, 17, error) == !beyond,
                    "native proposal endpoint uses constructed D+1");
            BSlot state; state.p = frontier;
            Options o; // verification cap 1 cannot reduce the constructed reservation
            int resident = 0, final = 0, initial = 0, continuation = 0;
            require(resident_attempt(state, o, 8, resident), "resident gate status");
            require(resident_final_attempt(state, max_position, final), "resident final gate status");
            SerialAdmission serial; serial.p = frontier; serial.initial(initial);
            serial.p = frontier - serial.keep; serial.continuation(continuation);
            require(resident == !beyond && final == !beyond && initial == !beyond && continuation == !beyond,
                    "all changed caller reservations agree at endpoint and one beyond");
        }
    }
    BSlot state; state.p = max_position - 2; state.proposal.count = 7; state.proposal.confidence.fill(1);
    int calls = 0;
    require(resident_attempt(state, {}, 8, calls) && calls == 0 && resident_width(state, 1, 8, 2) == 2,
            "constructed7/cap1 still needs eight absolute rows, even when only two are verified");
    set_draft_positions(3); state.p = 10; state.max_new = 1;
    resident_attempt(state, {}, 8, calls);
    require(calls == 0, "one output row remains anchor-only");
    state.max_new = 16; Options one_context; one_context.max_context = state.p + 1;
    resident_attempt(state, one_context, 8, calls); resident_attempt(state, {}, 1, calls);
    require(calls == 0, "one context row or spec1 does not construct");
    state.stop = true; resident_final_attempt(state, max_position, calls);
    require(calls == 0, "resident terminal guard preserved");
    SerialAdmission serial; serial.native_first_block = false; serial.admit_first_block = true;
    serial.initial(calls); require(calls == 1, "BGEN first-block gate remains supported");
    serial.stop_req.stopped = true; serial.initial(calls);
    serial.terminal = true; serial.continuation(calls);
    serial.terminal = false; serial.admission_yield = true; serial.continuation(calls);
    require(calls == 1, "serial cancellation, terminal and admission-yield gates preserved");
}
void native_guard_cases() {
    set_draft_positions(3);
    const int frontier = max_position - 4;
    NativeContext context{frontier}; NativeRequest request{&context};
    const core::DraftOwner owner{0, 1, 0, 2};
    auto stale = owner; ++stale.generation;
    std::string error;
    require(!install_uniforms(&request, stale, owner, frontier, {0, 0, 0}, error) &&
            error == "stale proposal random owner/frontier" && !request.uniforms_valid, "foreign/stale owner refusal");
    require(!install_uniforms(&request, owner, owner, frontier - 1, {0, 0, 0}, error), "wrong uniform frontier");
    for (float invalid : {1.0f, std::numeric_limits<float>::quiet_NaN()})
        require(!install_uniforms(&request, owner, owner, frontier, {0, invalid, 0}, error) &&
                error == "proposal uniforms must be engine-owned [0,1) draws", "finite [0,1) guard");
    core::DraftCoverage coverage; coverage.present = true;
    const Sampling sampled{false, 1};
    require(!native_preflight(&request, owner, coverage, sampled, frontier - 1, 17, error), "sampled uniforms required");
    require(install_uniforms(&request, owner, owner, frontier, {0, 0, 0}, error) &&
            native_preflight(&request, owner, coverage, sampled, frontier - 1, 17, error), "sampled endpoint remains valid");
    request.uniform_owner = stale;
    require(!native_preflight(&request, owner, coverage, sampled, frontier - 1, 17, error), "exact random owner preserved");
    request.uniform_owner = owner; --request.uniform_frontier;
    require(!native_preflight(&request, owner, coverage, sampled, frontier - 1, 17, error), "exact random frontier preserved");
    coverage.coherent = false;
    require(!native_preflight(&request, owner, coverage, {}, frontier - 1, 17, error), "coverage coherence preserved");
    coverage.coherent = true;
    require(!native_preflight(&request, owner, coverage, {}, frontier - 1, -1, error), "anchor range preserved");
    for (int positions : {4, 7}) {
        set_draft_positions(positions);
        require(!native_preflight(&request, owner, coverage, sampled, frontier - 1, 17, error) &&
                error == "sampled proposals support at most three independently drawn rows",
                "small verification cap cannot authorize unsupported sampled construction");
    }
}
void forward_cases() {
    set_draft_positions(7); // explicit forward width must not consult this value
    ForwardPreflight forward;
    std::string error;
    const int token = 0;
    for (int rows : {1, 4, 8}) for (int beyond = 0; beyond <= 1; ++beyond) {
        const int frontier = max_position - rows + beyond;
        DeviceContext::Impl c; c.begin = c.end = frontier;
        FeatureView f; f.start = frontier; f.request_id = "request";
        require(forward.accepts(c, f, frontier, rows, &token, error) == !beyond,
                "forward reserves its explicit row count");
    }
    DeviceContext::Impl c; c.begin = c.end = max_position;
    FeatureView f; f.start = max_position; f.request_id = "request";
    require(forward.accepts(c, f, max_position, 8, nullptr, error), "ingest reserves zero rows at absolute endpoint");
    require(!public_rows(0, &token, &token, error) && !public_rows(9, &token, &token, error) &&
            !public_rows(4, nullptr, &token, error), "public forward row/pointer guards preserved");
    c.owner = 2;
    require(!forward.accepts(c, f, max_position, 0, nullptr, error) && error == "foreign or poisoned context",
            "foreign forward owner still refused");
    c.owner = 1; c.poisoned = true;
    require(!forward.accepts(c, f, max_position, 0, nullptr, error), "poisoned forward context still refused");
    c.poisoned = false; --c.end;
    require(!forward.accepts(c, f, max_position, 0, nullptr, error), "incoherent committed endpoint still refused");
    c.begin = c.end; c.last_generation = 3;
    std::array<uint16_t, feature_width> values{}; const std::array<int64_t, 1> tags{2};
    --f.start; f.count = 1; f.segments[0] = values; f.generations[0] = tags;
    require(!forward.accepts(c, f, max_position, 0, nullptr, error) &&
            error == "uncommitted or stale feature generation" && c.end == max_position - 1,
            "real feature-generation predicate refuses stale tags without publishing context");
}
void width_cases() {
    set_draft_positions(7);
    for (const auto [count, cap, expected] : {std::array{3, 1, 2}, std::array{7, 3, 4},
                                            std::array{1, 7, 2}, std::array{3, 3, 4}}) {
        BSlot state; state.proposal.count = count; state.proposal.confidence.fill(1);
        require(resident_width(state, cap) == expected && serial_width(count, cap).selected == expected,
                "resident and serial cap availability before confidence selection");
    }
    BSlot state; state.proposal.count = 3; state.proposal.confidence.fill(1);
    require(constructed_ids(17).first == 8, "width selection did not shrink startup construction");
    set_draft_positions(3);
    require(constructed_ids(17).first == 4 && resident_width(state, 1, 8, 2) == 2 &&
            serial_width(3, 1, false, 1, false, 0, 8, 2).total == 2, "constructed3/cap1 with two rows left");
    require(resident_width(state, 3, 8, 1) == 1 && resident_width(state, 7, 3) == 3 &&
            serial_width(7, 7, false, 1, false, 0, 3).selected == 3, "remaining rows and S_mtp still clip independently");
    require(serial_width(3, 3, false, 0.49f).selected == 1 && serial_width(3, 3, false, 0.5f).selected == 4,
            "greedy below/equal confidence threshold");
    const auto low = serial_width(3, 3, true, 0.49f, false, 0, 8, 16, 12);
    const auto equal = serial_width(3, 3, true, 0.5f);
    require(low.selected == 2 && equal.selected == 4 && low.competition == 4 && equal.competition == 4,
            "sampled competition ceiling ignores current confidence-selected T and token agreement");
    require(serial_width(7, 1, true, 0).competition == 2 &&
            serial_width(7, 1, true, 0).selected == 2, "sampled cap1 ceiling and unsuppressible first draw");
    require(serial_width(3, 3, false, 0.49f).competition == 1, "greedy competition still uses selected T");
    const auto lookup = serial_width(7, 1, true, 1, true);
    const auto chain = serial_width(7, 1, false, 1, false, 3);
    require(lookup.lookup && lookup.total == 7 && chain.total == 5 && chain.prefix == 2,
            "neural cap does not cap lookup replacement or chained final width");
}
void reference_cases() {
    set_draft_positions(7); // fixed reference extent is independent of native startup width
    Capture c; c.frontier = max_position - reference_block_size;
    c.proposal.owner = c.owner; c.proposal.frontier = c.frontier; c.proposal.count = 3;
    observer_range(c);
    require(replay_range({{"origin", c.frontier}, {"frontier", c.frontier}}).second == c.frontier,
            "fixed-four replay accepts exact endpoint");
    ++c.frontier; c.proposal.frontier = c.frontier;
    require(refuses([&] { observer_range(c); }) &&
            refuses([&] { replay_range({{"origin", 0}, {"frontier", c.frontier}}); }) &&
            refuses([&] { replay_range({{"origin", c.frontier}, {"frontier", c.frontier}}); }),
            "observer and replay refuse one beyond fixed-four extent");
    --c.frontier; c.proposal.frontier = c.frontier; c.proposal.count = 4;
    require(refuses([&] { observer_range(c); }), "observer retains exactly three proposals");
}
} // namespace
int main() {
    try {
        frontier_cases(); native_guard_cases(); forward_cases(); width_cases(); reference_cases();
        set_draft_positions(3);
        std::puts("PASS dflash2_frontier_cap_test (source predicates only; no device/model execution)");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL dflash2_frontier_cap_test: %s\n", error.what());
        return 1;
    }
}
