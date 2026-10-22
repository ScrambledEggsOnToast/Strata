// Queued real-context fixture. Actual pinned weights; synthetic features
// and embeddings. No target inference, sampled verification or resident claim.
#include "strata/program/dflash2_parity_capture.hpp"
#include "strata/core/dflash2_forward.hpp"
#include "strata/core/dflash2_weights.hpp"
#include "strata/core/family_layout.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/dflash2_context_blas.hpp"
#include "strata/kernels/dflash2_forward.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <vector>

namespace {
namespace io = strata::program::dflash2_parity::io;
namespace df = strata::core::dflash2;
using namespace strata::core;
constexpr size_t guard = 16;
constexpr size_t elements = size_t(df::block_size) * df::hidden_size;
struct Options {
    std::string runtime, manifest, pack, tokenizer, shard0, shard1;
    int device = -1;
};
Options options(int argc, char** argv) {
    Options out;
    std::map<std::string, std::string*> strings{{"--runtime", &out.runtime}, {"--manifest-sha256", &out.manifest},
        {"--target-pack", &out.pack}, {"--tokenizer", &out.tokenizer}, {"--shard0", &out.shard0}, {"--shard1", &out.shard1}};
    std::set<std::string> seen;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        io::require(seen.insert(key).second && i + 1 < argc, "duplicate argument or missing value");
        const std::string value = argv[++i];
        if (key == "--device") {
            const auto parsed = std::from_chars(value.data(), value.data() + value.size(), out.device);
            io::require(parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size() && out.device >= 0,
                        "invalid device ordinal");
        } else {
            const auto at = strings.find(key);
            io::require(at != strings.end() && !value.empty(), "unknown/empty option");
            *at->second = value;
        }
    }
    io::require(!out.runtime.empty() && io::sha256_valid(out.manifest) && !out.pack.empty() &&
        !out.tokenizer.empty() && !out.shard0.empty() && !out.shard1.empty() && out.device >= 0,
        "missing required options; use --help");
    return out;
}
class ContextTrace final : public df::ForwardTraceSink {
public:
    int fc = 0, kv = 0;
    bool capture(const df::ForwardTrace& trace, void*, std::string&) override {
        if (std::strcmp(trace.operation, "context.fc") == 0) ++fc;
        if (std::strcmp(trace.operation, "context.k_proj") == 0 || std::strcmp(trace.operation, "context.v_proj") == 0) ++kv;
        return true;
    }
};
struct Features {
    std::vector<uint16_t> values;
    std::vector<int64_t> tags;
    df::FeatureView view;
    Features(int start, int count, std::string_view request, int seed)
        : values(size_t(count) * df::feature_width), tags(size_t(count), start + 1) {
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = strata::kernels::bf16_from_f32(float(int((i * 31 + size_t(seed)) % 127) - 63) / 128.f);
        view.start = start; view.count = count; view.request_id = request; view.incarnation = 1;
        view.segments[0] = values; view.generations[0] = tags;
    }
};
bool same(const df::DeviceContextSnapshot& a, const df::DeviceContextSnapshot& b) {
    const auto& x = a.metadata(); const auto& y = b.metadata();
    return a.valid() && b.valid() && x.origin == y.origin && x.begin == y.begin && x.end == y.end &&
        x.initialized == y.initialized && x.target == y.target && x.manifest_sha256 == y.manifest_sha256 &&
        x.projection_mode == y.projection_mode &&
        a.payload().size() == b.payload().size() && std::equal(a.payload().begin(), a.payload().end(), b.payload().begin());
}
struct Buffers {
    cudaStream_t stream = nullptr;
    uint16_t *input = nullptr, *output = nullptr;
    std::vector<uint16_t> host_input = std::vector<uint16_t>(elements);
    std::vector<uint16_t> result = std::vector<uint16_t>(elements + 2 * guard);
    Buffers() {
        try {
            io::cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "fixture stream");
            io::cuda(cudaMalloc(&input, elements * sizeof(uint16_t)), "fixture embeddings");
            io::cuda(cudaMalloc(&output, result.size() * sizeof(uint16_t)), "fixture guarded output");
        } catch (...) { release(); throw; }
    }
    void release() noexcept {
        if (stream && cudaStreamSynchronize(stream) != cudaSuccess) std::terminate();
        if (input && cudaFree(input) != cudaSuccess) std::terminate();
        if (output && cudaFree(output) != cudaSuccess) std::terminate();
        if (stream && cudaStreamDestroy(stream) != cudaSuccess) std::terminate();
    }
    ~Buffers() { release(); }
    void upload(int seed, bool nonfinite = false) {
        for (size_t i = 0; i < elements; ++i)
            host_input[i] = strata::kernels::bf16_from_f32(float(int((i * 13 + size_t(seed)) % 97) - 48) / 64.f);
        if (nonfinite) host_input[0] = 0x7fc0;
        io::cuda(cudaMemcpyAsync(input, host_input.data(), elements * sizeof(uint16_t), cudaMemcpyHostToDevice, stream), "upload embeddings");
    }
    void poison() { io::cuda(cudaMemsetAsync(output, 0xa5, result.size() * sizeof(uint16_t), stream), "poison output"); }
    std::vector<uint16_t> read(int rows) {
        io::cuda(cudaMemcpyAsync(result.data(), output, result.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost, stream), "read output");
        io::cuda(cudaStreamSynchronize(stream), "complete fixture read");
        for (size_t i = 0; i < result.size(); ++i) {
            if (i < guard || i >= guard + size_t(rows) * df::hidden_size)
                io::require(result[i] == 0xa5a5, "output suffix/guard changed");
            else io::require((result[i] & 0x7f80) != 0x7f80, "nonfinite healthy output");
        }
        return {result.begin() + guard, result.begin() + guard + size_t(rows) * df::hidden_size};
    }
};
int run(const Options& o) {
    io::require(!strata::kernels::dflash2::context_blas_enabled() && strata::kernels::dflash2::attention_value_split_enabled(),
                "fixture requires split-only mode");
    io::DeviceScope device(o.device);
    cudaDeviceProp properties{}; io::cuda(cudaGetDeviceProperties(&properties, o.device), "read device capability");
    io::require(properties.major == 7 && properties.minor == 0, "split lifecycle fixture requires SM70");
    ModelGeometry geometry; int64_t top_k = 0; std::string error;
    {
        strata::GgufFile metadata(o.shard0);
        io::require(geometry_from_gguf(metadata, geometry, top_k, error), error);
        io::require(compiled_geometry_mismatch(geometry).empty(), "target compiled geometry mismatch");
    }
    NativeEmbed embedding; NativeHead head;
    const std::vector<std::string> shards{o.shard0, o.shard1};
    io::require(embedding.load(shards, df::hidden_size, df::vocabulary_size, error, true), error);
    io::require(head.load(shards, df::hidden_size, df::vocabulary_size, error, true), error);
    const auto target = df::VerifiedDailyTarget::identify(o.pack, o.tokenizer, {o.shard0, o.shard1}, geometry,
        embedding.identity(), head.identity(), std::chrono::seconds(900));
    df::RuntimeWeights weights(o.runtime, o.manifest, target.binding().identity);
    Buffers buffers;
    df::DeviceForward forward(weights, o.device);
    io::require(forward.device_bytes() == weights.dense().size_bytes() + df::DeviceForward::scratch_bytes(), "split mode device admission is absent");
    io::require(df::DeviceForward::scratch_bytes() == size_t(76718592) + strata::kernels::dflash2::attention_value_split_bytes,
                "split workspace is not fully priced before allocation");
    io::require(df::DeviceForward::numerical_mode() == "legacy-bf16-v1:value-split128-context1024-fp32-guard-v1",
                "split-only mode identity is not explicit");
    auto a = forward.create_context(2039), b = forward.create_context(73);
    Features initial_a(2039, 1024, "101", 1), initial_b(73, 1, "202", 9);
    initial_a.values[0] = 0x0001; // Finite BF16 subnormal retained by the baseline projection.
    ContextTrace trace;
    io::require(forward.ingest(*a, initial_a.view, 3063, buffers.stream, error, &trace), error);
    io::require(trace.fc == 1 && trace.kv == 10, "real FC and all five K/V pairs not visited");
    io::require(a->initialized() && a->origin() == 2039 && a->begin() == 2039 && a->end() == 3063,
                "real context did not publish the wrapped committed interval");
    io::require(forward.ingest(*b, initial_b.view, 74, buffers.stream, error), error);
    auto saved = forward.create_snapshot(), peer = forward.create_snapshot(), check = forward.create_snapshot();
    io::require(forward.snapshot(*a, *saved, buffers.stream, error), error);
    io::require(forward.snapshot(*b, *peer, buffers.stream, error), error);
    io::require(saved->metadata().projection_mode == df::DeviceForward::numerical_mode() &&
        saved->metadata().projection_mode != "legacy-bf16-v1", "snapshot lacks actual numerical mode");
    Features empty(3063, 0, "101", 0);
    buffers.upload(3); buffers.poison();
    io::require(forward.forward(*a, empty.view, buffers.input, 4, 3063, buffers.output + guard, buffers.stream, error), error);
    (void)buffers.read(4);
    io::require(forward.snapshot(*a, *check, buffers.stream, error) && same(*saved, *check), "proposal changed committed context");
    Features advance(3063, 1, "101", 31);
    io::require(forward.ingest(*a, advance.view, 3064, buffers.stream, error), error);
    io::require(forward.rewind(*a, 3063, error), error);
    io::require(forward.restore(*a, *saved, buffers.stream, error), error);
    io::require(forward.snapshot(*a, *check, buffers.stream, error) && same(*saved, *check), "same-mode restore changed context");

    auto imported = forward.create_snapshot();
    io::require(imported->assign(saved->metadata(), saved->payload(), error), error);
    for (const char* rejected : {"legacy-bf16-v1", "foreign-numerical-mode"}) {
        auto metadata = saved->metadata(); metadata.projection_mode = rejected;
        io::require(!imported->assign(metadata, saved->payload(), error) && same(*saved, *imported),
                    "cross-mode import changed the valid image before refusing");
    }
    // Refused imports must not poison the healthy owner or block a valid import.
    io::require(imported->assign(saved->metadata(), saved->payload(), error), error);
    io::require(forward.reset(*a, 2039, error), error);
    io::require(forward.validate_restore(*a, *imported, error), error);
    io::require(forward.restore(*a, *imported, buffers.stream, error), error);
    io::require(forward.snapshot(*a, *check, buffers.stream, error) && same(*saved, *check), "import/reset/restore changed context");
    io::require(forward.snapshot(*b, *check, buffers.stream, error) && same(*peer, *check), "peer changed during advance/restore/import");
    std::puts("PASS split_value_runtime_real_weights context=1024 proposal_rows=4 snapshot_restore_import=1 wrong_mode_refused_before_mutation=1 peer_isolation=1");

    // One final failing call exercises original nonfinite data through the
    // retained numerical-status checks and the shared-owner poison boundary.
    io::require(forward.reset(*a, 0, error), error);
    Features poison(0, 129, "303", 3); poison.values[0] = 0x7fc0;
    io::require(!forward.ingest(*a, poison.view, 129, buffers.stream, error) && a->poisoned() &&
                error.find("nonfinite") != std::string::npos, "nonfinite context did not poison its owner");
    Features peer_empty(74, 0, "202", 0);
    io::require(!forward.ingest(*b, peer_empty.view, 74, buffers.stream, error) &&
                error.find("poisoned") != std::string::npos, "shared poisoned scratch admitted another request");
    std::puts("PASS dflash2_value_split_runtime_test real_weights=1 synthetic_inputs=1 actual_q_not_exercised=1 resident_qualification=0");
    return 0;
}
} // namespace
int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--help") == 0) {
        std::puts("Queued real-context fixture: --runtime DIR --manifest-sha256 SHA --target-pack DIR --tokenizer DIR --shard0 FILE --shard1 FILE --device N. Synthetic features/embeddings; no target inference or throughput claim.");
        return 0;
    }
    try {
        const auto o = options(argc, argv);
        io::require(setenv("STRATA_DFLASH2_CONTEXT_CUBLAS", "0", 1) == 0 &&
                    setenv("STRATA_DFLASH2_CONTEXT_CUBLAS_TRACE", "0", 1) == 0 &&
                    setenv("STRATA_DFLASH2_VALUE_SPLIT", "1", 1) == 0 &&
                    setenv("STRATA_DFLASH2_VALUE_SPLIT_TRACE", "1", 1) == 0, "cannot enable split-only fixture");
        return run(o);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL dflash2_value_split_runtime_test: %s\n", error.what()); return 1;
    }
}
