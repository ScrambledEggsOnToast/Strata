// #198: native upload pages cover the deduplicated union of the stages' planned sources.
// Synthetic GGUF headers/payloads only: no device discovery, allocation, engine or model run.
#include "gguf_fixture.hpp"
#include "strata/core/native_dense.hpp"
#include "strata/plan/admission.hpp"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using strata::core::NativeDense;
using namespace strata::plan;
namespace fs = std::filesystem;

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

struct TempDir {
    fs::path path = fs::temp_directory_path() /
        ("strata-native-source-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    TempDir() { fs::create_directories(path); }
    ~TempDir() {
        NativeDense::set_layer_range(-1, -1);
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

constexpr uint64_t GiB = 1ull << 30;
constexpr uint64_t page_size = 4096;
const char* key = "blk.1.ple_key.weight";
const char* first = "blk.0.attn_qkv.weight";
const char* later = "blk.25.attn_qkv.weight";

AdmissionDecision admission(const std::vector<std::string>& shards, const NativeDense::SourceSpans& sources,
                            const std::vector<uint64_t>& device_bytes, uint64_t& pages, std::string& err) {
    Scope guest, host;
    if (!NativeDense::source_pages(shards, sources, page_size, pages, err))
        guest.add_unknown("file:native_dense_upload", err, ByteMeaning::clean_file);
    guest.add("file:native_dense_upload", pages, "checked native source union", ByteMeaning::clean_file);
    guest.mirror_all_into(host);
    std::vector<DeviceCost> devices;
    std::vector<DeviceTelemetry> telemetry;
    for (size_t i = 0; i < device_bytes.size(); ++i) {
        const std::string name = "CUDA" + std::to_string(i);
        devices.emplace_back();
        devices.back().name = name;
        devices.back().vram.add("weights_native_dense", device_bytes[i], "post-trim plan");
        DeviceTelemetry measured;
        measured.name = name;
        measured.measured = true;
        measured.total_bytes = measured.free_bytes = 32 * GiB;
        telemetry.push_back(measured);
    }
    GuestHostPolicy policy;
    policy.guest_total_measured = policy.host_available_measured = true;
    policy.guest_total_bytes = policy.host_available_bytes = 64 * GiB;
    return admit_configuration(devices, telemetry, guest, host, policy);
}

uint64_t price(const std::vector<std::string>& shards, int lo, int hi, NativeDense::SourceSpans& sources) {
    NativeDense::set_layer_range(lo, hi);
    uint64_t bytes = 0;
    std::string err;
    require(NativeDense::planned_bytes(shards, true, bytes, err, &sources), "stage pricing: " + err);
    return bytes;
}
} // namespace

int main() {
    try {
        TempDir temp;
        const std::vector<std::string> shards{(temp.path / "a.gguf").string(), (temp.path / "b.gguf").string()};
        fixture::write(shards[0], {}, {
            {first, {256, 4}, 8, 1},
            {"blk.24.attn_gate.weight", {256, 4}, 8, 2},
            {key, {256, 4}, 23, 3},
            {"blk.0.attn_output.weight", {256, 4}, 0, 4}, // unsupported type: canonical fallback
            {"blk.0.ffn_gate_inp.weight", {256, 4}, 8, 5}, // router, not a native projection
            {"blk.0.attn_q.weight", {256}, 8, 6}}); // not a 2-D matrix
        fixture::write(shards[1], {}, {
            {later, {256, 4}, 8, 7}, {"blk.47.ffn_down_shexp.weight", {256, 4}, 8, 8}});

        NativeDense::SourceSpans trimmed;
        const uint64_t primary = price(shards, 0, 25, trimmed);
        require(primary == 3008 && trimmed.size() == 3 && !trimmed.count(later),
                "CUDA0 must price only its two projections, the PLE key and Q8_1 scratch");
        const uint64_t secondary = price(shards, 25, 48, trimmed);
        NativeDense::set_layer_range(0, 25); // generate restores the pending primary load range
        require(secondary == 3008 && trimmed.size() == 5 && trimmed.at(key) == 544 && trimmed.at(later) == 1088,
                "later-stage sources must join the union; the replicated PLE key must not be counted twice");
        uint64_t pages = 0;
        std::string err;
        auto decision = admission(shards, trimmed, {primary, secondary}, pages, err);
        require(decision.admitted && decision.unknown_classes.empty() && pages == 40960 &&
                decision.guest_demand_bytes == pages && decision.host_demand_bytes == pages,
                "trimmed two-stage configuration must have priced guest and physical-host native upload pages: " + err);
        std::printf("trimmed split25: stage bytes=3008,3008; source union=5 tensors; guest/host pages=40960; admitted\n");

        NativeDense::SourceSpans untrimmed;
        const uint64_t full = price(shards, -1, -1, untrimmed);
        require(full == 5184 && trimmed == untrimmed,
                "trim changes per-device weights, not the full set of sources uploaded across all stages");
        decision = admission(shards, untrimmed, {full, full}, pages, err);
        require(decision.admitted && pages == 40960, "untrimmed upload accounting must be unchanged");
        std::printf("untrimmed: stage bytes=5184,5184; same source union/pages; admitted\n");

        NativeDense::SourceSpans three_stages;
        const std::vector<uint64_t> three_prices{
            price(shards, 0, 10, three_stages), price(shards, 10, 25, three_stages),
            price(shards, 25, 48, three_stages)};
        require(three_stages == trimmed, "three-stage union must also deduplicate the PLE key");
        decision = admission(shards, three_stages, three_prices, pages, err);
        require(decision.admitted && pages == 40960, "three-stage native source price must cover every stage");
        std::printf("trimmed three-stage union: same source union/pages; admitted\n");

        // A real disagreement between the kernel's planned byte count and the independent GGUF block geometry.
        auto mismatched = trimmed;
        ++mismatched.at(later);
        decision = admission(shards, mismatched, {primary, secondary}, pages, err);
        require(!decision.admitted && decision.limiting_term == "guest:file:native_dense_upload" &&
                decision.unknown_classes.size() == 2 &&
                err == "native source spans differ from NativeDense::planned_bytes",
                "a deliberately wrong planned source extent must still refuse admission: " + err);
        std::printf("deliberate +1-byte planned/source mismatch: refused, limiting=%s; guest and host unknown\n",
                    decision.limiting_term.c_str());

        mismatched = trimmed;
        mismatched.emplace("blk.99.attn_q.weight", 1088);
        decision = admission(shards, mismatched, {primary, secondary}, pages, err);
        require(!decision.admitted && err == "native source spans differ from NativeDense::planned_bytes",
                "a planned source missing from the GGUF must refuse, not silently disappear");
        std::printf("deliberate missing planned source: refused\n");
        std::printf("native_dense_source_test: all passed\n");
    } catch (const std::exception& error) {
        std::fprintf(stderr, "native_dense_source_test: %s\n", error.what());
        return 1;
    }
    return 0;
}
