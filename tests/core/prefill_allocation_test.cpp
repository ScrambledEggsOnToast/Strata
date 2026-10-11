// tests/core/prefill_allocation_test.cpp - the prefill allocator's CONSUMER contracts, checked without a
// device, a session, or weights.  `allocation_needed` is the admission-grade bound: it must run before
// SessionState exists, fail closed on overflow and invalid configurations, and report the loan/owned split
// that generate.cpp's lending (bytes_needed) and init's loan validation rely on.  These tests pin those
// contracts; they deliberately do NOT transcribe the internal take sequence - a copy of the implementation
// would pass even when the estimator and the carver drifted apart, which is exactly what it must catch.
#include "strata/prefill/prefill.hpp"
#include "strata/core/layout.hpp"
#include "strata/core/session.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/prefill/moe_mmq.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <cstdlib>
#include <filesystem>
#include <fstream>

namespace {
// The allocator's one rounding rule, documented on AllocationBytes: every payload costs
// ceil(payload / 256) * 256 + 256 (alignment plus guard), so consumers can price single buffers.
uint64_t allocation(uint64_t payload) { return (payload + 511) & ~UINT64_C(255); }
uint64_t owned(uint64_t payload) { return (payload + (2ull << 20) - 1) & ~((2ull << 20) - 1); }
int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "%s\n", what); ++failures; }
}

// Run in its own process: the prompt MMQ plan freezes the loaded artifact on first use.
int native_bounds_regression() {
    using namespace strata;
    if (!prefill::mmq::built() || !kernels::cpu::native_experts_available()) return 77;
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto directory = std::filesystem::temp_directory_path() /
                           ("strata-native-bounds-" + std::to_string(stamp));
    if (!std::filesystem::create_directory(directory)) return 1;
    struct Fixture {
        std::filesystem::path path;
        ~Fixture() {
            std::error_code ignored;
            std::filesystem::remove_all(path, ignored);
        }
    } fixture{directory};
    const std::string path = directory.string();
    core::ModelGeometry g;
    kernels::cpu::NativeFmt fmt;
    std::string err;
    if (!kernels::cpu::native_fmt(21, 20, 2560, 640, fmt, err)) return 1;
    {
        std::ofstream out(path + "/native_experts.txt");
        for (int64_t layer = 0; layer < g.n_layers; ++layer)
            out << layer << " 21 20 " << (uint64_t) layer * g.n_expert * fmt.bytes << ' ' << fmt.bytes << '\n';
        if (!out) return 1;
    }
    if (!kernels::cpu::expert_layout_load(path, g.n_layers, g.n_expert, err)) return 1;
#if defined(_WIN32)
    if (_putenv_s("STRATA_PREFILL_MMQ", "1") != 0) return 1;
#else
    if (setenv("STRATA_PREFILL_MMQ", "1", 1) != 0) return 1;
#endif
    prefill::PrefillFacts facts;
    facts.max_cells = 4097;
    facts.n_pages = (facts.max_cells + 3) / 4;
    core::ModelGeometry small = g;
    small.n_expert = 16;
    // At this small chunk the attention workspace dominates both router sizes, isolating the fixed bounds.
    const uint64_t full_price = prefill::Prefill::bytes_needed_from(g, facts, 64, 8, fmt.bytes, 0);
    const uint64_t small_price = prefill::Prefill::bytes_needed_from(small, facts, 64, 8, fmt.bytes, 0);
    const uint64_t full_bounds = ((uint64_t) g.n_expert + 1 + (uint64_t) g.n_expert * 17) * 4;
    const uint64_t small_bounds = ((uint64_t) small.n_expert + 1 + (uint64_t) small.n_expert * 17) * 4;
    check(full_price != UINT64_MAX && small_price != UINT64_MAX && full_price >= small_price &&
          full_price - small_price >= allocation(full_bounds) - allocation(small_bounds),
          "native MMQ admission covers one row-limited compact group per expert");
    return failures ? 1 : 0;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--native-bounds") return native_bounds_regression();
    using namespace strata;
    core::ModelGeometry g;                       // the artifact's fixed geometry (validated inside)
    kernels::cpu::ExpertLayout experts;          // canonical pack: default max_blob, no native tables
    experts.n_layers = g.n_layers;
    experts.n_expert = g.n_expert;
    const kernels::QsaShapes s = kernels::qsa_real_shapes();

    prefill::AllocationConfig c;
    c.max_cells = 4097;
    c.n_pages = (c.max_cells + 3) / 4;
    c.chunk = 64;
    c.kv_mode = 0;
    c.ring = 8;
    c.mmq = false;
    c.device_cc = 860; c.device_sms = 82; c.device_shared_bytes = 101376;

    prefill::AllocationBytes b, other;
    std::string err;
    auto estimate = [&](prefill::AllocationBytes& out) {
        return prefill::Prefill::allocation_needed(g, c, experts, out, err);
    };

    // ---- works before SessionState allocation, and is a pure function of its inputs.
    check(estimate(b), "standalone sizing without initialized session/weights");
    check(b.standalone_device == prefill::Prefill::bytes_needed_from(g, c.facts(), c.chunk, c.ring,
          experts.max_blob, 0, true) + b.owned_device,
          "standalone prices owned facts sequence plus raw independent allocations");
    check(b.loanable_device > 0 && b.owned_device > 0, "both regions are populated");

    // Fixed host payload includes the mapped bounds tail even with MMQ off; dynamic jobs are separate.
    const uint64_t bounds = (uint64_t) (g.n_expert + 1 + (g.n_expert + 15) / 16 * 17) * 4;
    check(b.host_payload == (uint64_t) c.chunk * 4 + (2 * (uint64_t) g.n_expert + 1) * 4 + 16 + bounds,
          "non-MMQ host payload includes the mapped bounds tail");

    // ---- a loan never covers the token ids: they are owned even in the plainest configuration.
    check(b.owned_device == owned((uint64_t) c.chunk * 4), "raw token allocation uses 2 MiB granules");

    // ---- each streaming ring slot is one aligned expert-blob slot (the ring contract admission lends by).
    c.ring = 16;
    check(estimate(other), "expanded ring sizing");
    check(other.loanable_device - b.loanable_device == 8 * allocation(experts.max_blob),
          "each ring slot includes its own guard/alignment");
    check(other.standalone_device == prefill::Prefill::bytes_needed_from(g, c.facts(), c.chunk, c.ring,
          experts.max_blob, 0, true) + other.owned_device,
          "owned ring is a single guarded take");
    c.ring = 8;
    const uint64_t ring_base = prefill::Prefill::bytes_needed_from(g, c.facts(), c.chunk, 0, 1, 0, true);
    const uint64_t ring_one = prefill::Prefill::bytes_needed_from(g, c.facts(), c.chunk, 8, 1, 0, true);
    check(ring_one - ring_base == owned(allocation(8)), "eight tiny owned ring slots consume one granule, not eight");

    // ---- a loan sized for the largest chunk covers every smaller relayout of the same configuration.
    c.chunk = 8192;
    check(estimate(other), "large chunk sizing");
    const uint64_t big = other.loanable_device;
    c.chunk = 64;
    check(estimate(other) && big >= other.loanable_device, "loan bound is monotone in chunk");
    const uint64_t small = other.loanable_device;
    c.ring = 512;
    check(estimate(other), "widest ring sizing");
    check(other.loanable_device >= small, "loan bound is monotone in ring at fixed chunk");
    c.ring = 8;

    // ---- KV STREAMING: the stage pool is sized by LOGICAL pages (ceil(max_cells / 4)), never by the
    // resident slot count, and the staging identity table is owned, not lent.
    const uint64_t rows = (uint64_t) c.n_pages * (uint64_t) g.n_head_kv * (uint64_t) s.page_size;
    c.kv_mode = 1;
    check(estimate(other), "streamed FP16 stage sizing");
    const uint64_t fp16_stage = 2 * allocation(rows * (uint64_t) g.head_dim * 2);
    check(other.loanable_device - b.loanable_device == fp16_stage,
          "logical pages, not resident KV slots, size streamed stage");
    check(other.owned_device - b.owned_device == owned((uint64_t) c.n_pages * 4),
          "identity table always owned");
    check(other.standalone_device == prefill::Prefill::bytes_needed_from(g, c.facts(), c.chunk, c.ring,
          experts.max_blob, 0, true) + other.owned_device,
          "streamed stage uses owned facts sequence");
    const prefill::AllocationBytes streamed = other;

    // STRATA_KV_STAGE_OWN moves stage buffers from guarded borrowing to independent 2 MiB pages.
    c.kv_stage_own = true;
    check(estimate(other), "separate stage sizing");
    check(other.standalone_device == streamed.standalone_device,
          "separate stage is never omitted from standalone peak");
    check(other.loanable_device + fp16_stage == streamed.loanable_device,
          "separate stage cannot be deducted as a loan");
    const uint64_t owned_fp16_stage = 2 * owned(allocation(rows * (uint64_t) g.head_dim * 2));
    check(other.owned_device == streamed.owned_device + owned_fp16_stage,
          "stage allocation moves to independent owned page demand");
    c.kv_stage_own = false;

    // ---- KV formats price their documented per-head bytes (INT8: codes + one FP16 scale per 64).
    c.kv_int8 = true;
    check(estimate(other), "INT8 stage sizing");
    check(other.loanable_device - b.loanable_device ==
          2 * allocation(rows * (uint64_t) g.head_dim) +
          2 * allocation(rows * (uint64_t) (g.head_dim / 64) * 2), "INT8 codes and scales counted");
    c.kv_int8 = false;
    c.kv_q4 = true;
    check(estimate(other), "Q4 stage sizing");
    check(other.loanable_device - b.loanable_device ==
          2 * allocation(rows * kernels::kv_q4_bytes_per_head((int) g.head_dim)),
          "Q4 stage block bytes counted");
    c.kv_q4 = false;
    c.kv_hybrid = true;
    check(estimate(other), "K8V4 stage sizing");
    check(other.loanable_device - b.loanable_device ==
          allocation(rows * (uint64_t) g.head_dim) + allocation(rows * (uint64_t) (g.head_dim / 64) * 2) +
          allocation(rows * kernels::kv_q4_bytes_per_head((int) g.head_dim)),
          "K8V4 counts K codes/scales and V Q4 blocks separately");
    c.kv_hybrid = false;
    c.kv_q4 = false;
    c.kv_mode = 0;

    // ---- DYNAMIC HOST PAYLOAD (host_dynamic): the counted fixed storage init allocates once and the path
    // refills in place.  Pinned by properties - what moves the row and what must not - never by
    // transcribing the term list.
    const uint64_t dyn0 = b.host_dynamic;
    check(dyn0 > 0, "routed staging still carries counted dynamic host storage");
    c.ring = 512;   // Estimator ring independence; this does not exercise runtime stream_all.
    check(estimate(other) && other.host_dynamic == dyn0,
          "the device ring choice moves no dynamic host storage");
    check(other.host_payload == b.host_payload, "fixed host payload stays ring-independent");
    c.ring = 8;
    c.kv_mode = 0;
    c.layer_begin = g.n_layers / 2;   // a narrower stage window shrinks the shared job/plan bound
    check(estimate(other) && other.host_dynamic < dyn0,
          "a narrower stage window shrinks the dynamic row");
    c.layer_begin = 0;
    c.chunk = 8192;   // the bound grows with the chunk, and a relayout to smaller never exceeds it
    check(estimate(other) && other.host_dynamic > dyn0,
          "the dynamic row grows with the chunk's draft staging bound");
    c.chunk = 64;
    check(estimate(other) && other.host_dynamic == dyn0, "restoring the configuration restores the row");

    // The shared job bound: one pure value sizes init's storage and prices this row, and the stager's
    // 16-bit claim word must be able to encode everything valid geometry can produce.
    int64_t jobs = -1;
    {
        core::ModelGeometry wide = g;
        wide.n_expert = 0x10000;
        int64_t refused = -1;
        check(!prefill::Prefill::stager_max_jobs(wide, c, refused) && refused == 0,
              "the job bound refuses, never clamps, past the claim word's range");
    }

    // ---- overflow and invalid configuration refuse, and leave the output zeroed (fail closed).
    c.chunk = std::numeric_limits<int64_t>::max();
    check(!estimate(other) && other.standalone_device == 0 && other.host_dynamic == 0,
          "chunk overflow fails closed");
    c.chunk = 64;
    experts.max_blob = std::numeric_limits<uint64_t>::max() - 511;
    check(!estimate(other) && other.standalone_device == 0, "ring accumulation overflow fails closed");
    check(prefill::Prefill::bytes_needed_from(g, c.facts(), c.chunk, 8, experts.max_blob, 0, true) == UINT64_MAX,
          "public owned facts counter independently refuses ring multiplication overflow");
    experts.max_blob = kernels::cpu::BLOB;
    c.n_pages -= 1;
    check(!estimate(other), "undersized logical page configuration rejected");
    c.n_pages += 1;
    c.ring = 7;
    check(!estimate(other), "ring below the routed staging minimum rejected");
    c.ring = 8;
    c.stager_ring = 1;
    check(!estimate(other), "out-of-range stager ring rejected");
    c.stager_ring = 16;
    c.kv_mode = 3;
    check(!estimate(other), "unknown KV mode rejected");
    c.kv_mode = 1; c.kv_int8 = true; c.kv_q4 = true;
    check(!estimate(other), "INT8 and Q4 together rejected");
    c.kv_int8 = c.kv_q4 = false;
    c.kv_mode = 0;

    // ---- the full-model IQ3_S/IQ4_NL geometry qualifies without a session, and the MMQ bound is the
    // configuration-derived stream-k maximum: the bound is what the external pool launch can never exceed.
    experts.native = true;
    experts.fmt.resize((size_t) g.n_layers);
    experts.bytes.assign((size_t) g.n_layers, 2000000);
    experts.max_blob = 2000000;
    for (auto& f : experts.fmt) {
        f.gu_type = 21; f.d_type = 20;   // GGML IQ3_S and IQ4_NL artifact type ids
        f.n_embd = 2560; f.n_ff = 640; f.bytes = 2000000;
    }
    c.mmq = true;
    check(estimate(b), "full-model IQ3_S geometry qualifies without session allocation");
    check(b.standalone_device == prefill::Prefill::bytes_needed_from(g, c.facts(), c.chunk, c.ring,
          experts.max_blob, 0, true) + b.owned_device,
          "MMQ uses owned facts sequence plus raw independent allocations");
    if (prefill::mmq::built()) {
        uint64_t workspace = 0;
        check(prefill::mmq::workspace_bytes(c.device_cc, c.device_sms, c.device_shared_bytes, workspace) &&
              b.mmq_workspace == workspace && prefill::mmq::supported(21) && prefill::mmq::supported(20) &&
              !prefill::mmq::supported(24),
              "MMQ scratch is explicitly included in estimator; IQ1_M stays on the FP16 path");
        c.device_sms = 0;
        check(!estimate(other), "missing MMQ hardware input fails closed");
        c.device_sms = 82;
    } else {
        check(b.mmq_workspace == 0, "no MMQ support: no workspace charged");
    }

    // Only the staging-vector row depends on MMQ support; the mapped tail is always priced.
    c.mmq = false;
    const uint64_t mmq_bounds = prefill::mmq::built() ? bounds : 0;
    check(estimate(other) && other.host_payload == b.host_payload - mmq_bounds,
          "MMQ off removes only supported bounds staging payload");

    // ---- native per-layer tables are validated before any arithmetic.
    experts.fmt[0].n_ff = 641;
    check(!estimate(other), "native tensor geometry mismatch rejected before allocation");

    // ---- restore the canonical layout: the lending entry point reads the process-wide one.
    experts = kernels::cpu::ExpertLayout{};
    experts.n_layers = g.n_layers;
    experts.n_expert = g.n_expert;
    c.mmq = false;

    // ---- a session whose logical page count contradicts its max_cells is refused: never under-lend.
    {
        core::QsaState q;
        q.max_cells = c.max_cells;
        q.n_pages = c.n_pages - 1;           // not ceil(max_cells / 4): refuse
        core::SessionState ss;
        ss.qsa_states = &q;
        check(prefill::Prefill::bytes_needed(g, ss, c.chunk) == UINT64_MAX,
              "bytes_needed refuses an inconsistent session");
    }
    return failures ? 1 : 0;
}
