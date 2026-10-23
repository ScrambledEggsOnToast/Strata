// Finite functional diagnostics: full native five-layer replay and actual borrowed
// quantized head. This executable is not a benchmark, verifier or acceptance gate.
#include "strata/program/dflash2_parity_capture.hpp"
#include "strata/core/dflash2_forward.hpp"
#include "strata/core/dflash2_selector.hpp"
#include "strata/core/dflash2_weights.hpp"
#include "strata/core/family_layout.hpp"
#include "strata/kernels/dflash2_forward.hpp"

#include <charconv>
#include <cstdio>
#include <iostream>
#include <map>
#include <set>

namespace {
namespace parity = strata::program::dflash2_parity;
namespace io = parity::io;
namespace df = strata::core::dflash2;
namespace gpu = strata::kernels::dflash2;
using io::Json;
namespace fs = std::filesystem;

void help() {
    std::cout <<
        "Usage: strata-dflash2-parity --runtime DIR --manifest-sha256 SHA256\n"
        "  --bundle DIR --output NEW_DIR --device N --target-pack DIR --tokenizer DIR\n"
        "  --shard0 FILE --shard1 FILE [--reference-hidden FILE]\n\n"
        "Replay exactly ONE real proposal bundle.json (four block positions; zero to\n"
        "2047 real committed context rows). Independently hash actual target artifacts\n"
        "and verified native embedding/head loads, then strictly load the pinned runtime.\n"
        "--reference-hidden is exactly 20480 little-endian BF16 bytes [4,2560] from\n"
        "the pinned CPU forward; project rows 1..3 through this SAME actual native head\n"
        "with NO extra target norm, and run the native selector on them.\n"
        "Every exposed forward operation is synchronously read on its submitting stream\n"
        "before scratch reuse. At most 1024 trace entries and 1 GiB output are permitted.\n"
        "No target forward, MTP, generated reference math, verifier or quality threshold\n"
        "is substituted. Replay/captured differences are numerical diagnostics only.\n";
}
struct Options {
    std::string runtime, manifest, bundle, output, pack, tokenizer, shard0, shard1, reference_hidden;
    int device = -1;
};
int number(std::string_view text) {
    int result = -1;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), result);
    io::require(parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size() && result >= 0,
                "device must be a nonnegative integer");
    return result;
}
Options options(int argc, char** argv) {
    Options out;
    std::map<std::string, std::string*> strings{{"--runtime", &out.runtime}, {"--manifest-sha256", &out.manifest},
        {"--bundle", &out.bundle}, {"--output", &out.output}, {"--target-pack", &out.pack}, {"--tokenizer", &out.tokenizer},
        {"--shard0", &out.shard0}, {"--shard1", &out.shard1}, {"--reference-hidden", &out.reference_hidden}};
    std::set<std::string> seen;
    for (int i = 1; i < argc; ++i) {
        const std::string key(argv[i]);
        io::require(seen.insert(key).second && i + 1 < argc, "duplicate option or absent value: " + key);
        const std::string value(argv[++i]);
        if (key == "--device") out.device = number(value);
        else {
            const auto at = strings.find(key);
            io::require(at != strings.end() && !value.empty(), "unknown/empty option: " + key);
            *at->second = value;
        }
    }
    io::require(!out.runtime.empty() && io::sha256_valid(out.manifest) && !out.bundle.empty() && !out.output.empty() &&
        !out.pack.empty() && !out.tokenizer.empty() && !out.shard0.empty() && !out.shard1.empty() && out.device >= 0,
        "all required finite input options must be supplied; use --help");
    return out;
}
int bounded(const Json& value, int minimum, int maximum, const char* name) {
    io::require(value.is_number_integer(), std::string(name) + " is not an integer");
    if (value.is_number_unsigned()) io::require(value.get<uint64_t>() <= uint64_t(maximum), std::string(name) + " exceeds bound");
    const int64_t integer = value.get<int64_t>();
    io::require(integer >= minimum && integer <= maximum, std::string(name) + " is out of range");
    return int(integer);
}
uint64_t owner_number(const Json& value, bool positive) {
    io::require(value.is_number_unsigned() || (value.is_number_integer() && value.get<int64_t>() >= 0), "invalid owner integer");
    const auto result = value.get<uint64_t>();
    io::require(result <= uint64_t(std::numeric_limits<int64_t>::max()) && (!positive || result > 0), "owner integer exceeds bound");
    return result;
}
class Stream {
public:
    Stream() { io::cuda(cudaStreamCreateWithFlags(&value_, cudaStreamNonBlocking), "create replay stream"); }
    ~Stream() {
        if (cudaStreamSynchronize(value_) != cudaSuccess || cudaStreamDestroy(value_) != cudaSuccess) std::terminate();
    }
    cudaStream_t get() const { return value_; }
    void drain() const { io::cuda(cudaStreamSynchronize(value_), "drain replay stream"); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
private:
    cudaStream_t value_ = nullptr;
};
class Buffer {
public:
    Buffer(size_t bytes, const Stream& stream) : stream_(stream), size_(bytes) {
        io::cuda(cudaMalloc(&data_, bytes), "allocate replay device buffer");
    }
    ~Buffer() {
        if (cudaStreamSynchronize(stream_.get()) != cudaSuccess || cudaFree(data_) != cudaSuccess) std::terminate();
    }
    template<class T> T* as() const { return static_cast<T*>(data_); }
    void upload(std::span<const std::byte> bytes) const {
        io::require(bytes.size() == size_, "replay upload extent mismatch");
        io::cuda(cudaMemcpyAsync(data_, bytes.data(), size_, cudaMemcpyHostToDevice, stream_.get()), "upload captured replay input");
        stream_.drain();
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
private:
    const Stream& stream_;
    void* data_ = nullptr;
    size_t size_;
};
class TraceWriter final : public df::ForwardTraceSink {
public:
    TraceWriter(fs::path root, uint64_t& written) : root_(std::move(root)), written_(written), host_(df::max_feature_bytes) {}
    bool capture(const df::ForwardTrace& trace, void* stream, std::string& error) override {
        try {
            io::require(entries_.size() < 1024 && trace.operation && trace.rank > 0 && trace.rank <= 4 && trace.data,
                        "invalid/overlong native trace");
            std::vector<int> shape(trace.shape.begin(), trace.shape.begin() + trace.rank);
            const size_t bytes = trace.bytes();
            io::require(bytes && bytes <= host_.size(), "trace readback exceeds bounded host arena");
            try {
                io::cuda(cudaMemcpyAsync(host_.data(), trace.data, bytes, cudaMemcpyDeviceToHost,
                                         static_cast<cudaStream_t>(stream)), "read actual forward operation");
                io::cuda(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)), "complete trace readback before scratch overwrite");
            } catch (...) {
                if (cudaStreamSynchronize(static_cast<cudaStream_t>(stream)) != cudaSuccess) std::terminate();
                throw;
            }
            const std::string operation(trace.operation);
            const auto dtype = trace.type == df::ForwardTrace::Type::bf16 ? "bf16" : "f32";
            auto descriptor = io::write_tensor(root_, "trace-" + std::to_string(entries_.size()) + "." + dtype,
                dtype, shape, {host_.data(), bytes}, written_);
            descriptor["operation"] = operation;
            descriptor["layer"] = trace.layer; descriptor["stage"] = trace.stage; descriptor["offset"] = trace.offset;
            descriptor["first_position"] = trace.first_position;
            descriptor["source"] = "actual_native_forward";
            descriptor["rms_subject"] = operation == "rms.mean_square_and_inverse" ? previous_ : "";
            entries_.push_back(std::move(descriptor)); previous_ = operation;
            return true;
        } catch (const std::exception& failure) { error = failure.what(); return false; }
    }
    Json download(const std::string& file, const std::string& dtype, const std::vector<int>& shape,
                  const void* device, size_t bytes, const Stream& stream) {
        io::require(bytes <= host_.size(), "output readback exceeds bounded arena");
        try {
            io::cuda(cudaMemcpyAsync(host_.data(), device, bytes, cudaMemcpyDeviceToHost, stream.get()), "read native replay output");
            stream.drain();
        } catch (...) {
            if (cudaStreamSynchronize(stream.get()) != cudaSuccess) std::terminate();
            throw;
        }
        io::require_finite({host_.data(), bytes}, dtype == "bf16");
        return io::write_tensor(root_, file, dtype, shape, {host_.data(), bytes}, written_);
    }
    const Json& entries() const { return entries_; }
private:
    fs::path root_;
    uint64_t& written_;
    io::PinnedBuffer host_;
    Json entries_ = Json::array();
    std::string previous_;
};
Json selection_document(const df::Selection& selection) {
    return {{"tokens", selection.tokens}, {"candidates", selection.candidates},
        {"conditional_probabilities", selection.conditional_probabilities},
        {"distribution", selection.sampled ? "conditional_q" : "point_mass"},
        {"greedy_tie_rule", "first strongest candidate slot; pinned CPU topk(sorted=False) order"}};
}
Json receipt_document(const strata::core::NativeTensorIdentity& receipt) {
    return {{"sha256", receipt.sha256}, {"bytes", receipt.bytes}, {"generation", receipt.generation},
        {"input", receipt.input}, {"output", receipt.output}, {"ggml_type", receipt.type}, {"device", receipt.device},
        {"source_path", receipt.source.path}, {"source_stamp", receipt.source.stamp}};
}
void project(const strata::core::NativeHead& head, const Buffer& hidden, Buffer& widened, Buffer& logits,
             Buffer& scratch, const Stream& stream, std::string& error) {
    for (int row = 0; row < 3; ++row) {
        io::cuda(cudaError_t(gpu::widen_bf16(hidden.as<uint16_t>() + size_t(row + 1) * df::hidden_size,
                         widened.as<float>(), df::hidden_size, stream.get())), "widen normalized BF16 for actual native head");
        io::require(head.run_with_scratch(widened.as<float>(), logits.as<float>() + size_t(row) * df::vocabulary_size,
                    scratch.as<void>(), head.projection_scratch_bytes(), stream.get(), error), "native head projection: " + error);
    }
    stream.drain();
}
int run(const Options& o) {
    io::require(std::endian::native == std::endian::little, "replay requires little-endian host");
    io::DeviceScope device(o.device);
    strata::core::ModelGeometry geometry;
    int64_t top_k = 0;
    std::string error;
    {
        strata::GgufFile metadata(o.shard0);
        io::require(strata::core::geometry_from_gguf(metadata, geometry, top_k, error), error);
        io::require(strata::core::compiled_geometry_mismatch(geometry).empty(), "target compiled geometry mismatch");
    }
    // Actual native objects, unchanged quantized bytes and opt-in load receipts.
    strata::core::NativeEmbed embedding;
    strata::core::NativeHead head;
    const std::vector<std::string> shards{o.shard0, o.shard1};
    io::require(embedding.load(shards, df::hidden_size, df::vocabulary_size, error, true), "verified embedding load: " + error);
    io::require(head.load(shards, df::hidden_size, df::vocabulary_size, error, true), "verified head load: " + error);
    const auto verified = df::VerifiedDailyTarget::identify(o.pack, o.tokenizer, {o.shard0, o.shard1}, geometry,
        embedding.identity(), head.identity(), std::chrono::seconds(900));
    df::RuntimeWeights weights(o.runtime, o.manifest, verified.binding().identity);
    const fs::path bundle_root(o.bundle);
    const auto bundle_bytes = io::read_file(bundle_root / "bundle.json", io::json_limit);
    const Json bundle = io::parse_json(bundle_bytes);
    io::require(bundle.at("schema") == 1 && bundle.at("kind") == "strata-dflash2-real-proposal" &&
        bundle.at("target") == io::target_document(verified.binding().identity) && bundle.at("runtime_manifest_sha256") == o.manifest,
        "bundle is not bound to independently loaded actual target/runtime bytes");
    const int origin = bounded(bundle.at("origin"), 0, df::max_position - df::reference_block_size, "origin");
    const int frontier = bounded(bundle.at("frontier"), origin, df::max_position - df::reference_block_size, "frontier");
    const int anchor = bounded(bundle.at("anchor"), 0, df::vocabulary_size - 1, "anchor");
    const auto& context_metadata = bundle.at("context");
    const int begin = bounded(context_metadata.at("begin"), origin, frontier, "context begin");
    const int count = bounded(context_metadata.at("count"), 0, df::context_rows, "context count");
    io::require(begin == std::max(origin, frontier - df::context_rows) && begin + count == frontier,
                "bundle does not contain whole current retained context");
    const auto& owner = bundle.at("owner");
    bounded(owner.at("seat"), 0, 8, "seat");
    const uint64_t request = owner_number(owner.at("request"), true);
    const uint64_t incarnation = owner_number(owner.at("incarnation"), false);
    owner_number(owner.at("generation"), true);
    const std::string request_id = context_metadata.at("request_id").get<std::string>();
    io::require(request_id == std::to_string(request), "bundle context/proposal request mismatch");
    const float temperature = bundle.at("temperature").get<float>();
    const auto uniforms = bundle.at("uniforms").get<std::array<float, 3>>();
    io::require(std::isfinite(temperature) && temperature >= 0, "invalid captured selector temperature");
    for (float value : uniforms) io::require(std::isfinite(value) && value >= 0 && value < 1, "invalid captured selector uniform");
    const auto& files = bundle.at("files");
    auto features = io::read_tensor(bundle_root, files.at("context"), "bf16", {count, df::feature_width});
    auto generations_bytes = io::read_tensor(bundle_root, files.at("generations"), "i64", {count});
    auto embeddings = io::read_tensor(bundle_root, files.at("raw_embeddings"), "bf16", {4, df::hidden_size});
    const auto engine_hidden = io::read_tensor(bundle_root, files.at("engine_hidden"), "bf16", {4, df::hidden_size});
    const auto engine_logits = io::read_tensor(bundle_root, files.at("engine_logits"), "f32", {3, df::vocabulary_size});
    io::require_finite(features, true);
    io::require_finite(embeddings, true);
    io::require_finite(engine_hidden, true);
    io::require_finite(engine_logits, false);
    std::vector<int64_t> generations(count);
    if (count) std::memcpy(generations.data(), generations_bytes.data(), generations_bytes.size());
    int64_t last = 0;
    for (int64_t generation : generations) {
        io::require(generation > 0 && generation >= last, "invalid chronological context generations"); last = generation;
    }
    const auto& actual_proposal = bundle.at("native_proposal");
    io::require(actual_proposal.at("count") == 3 && actual_proposal.at("tokens").size() == 3 &&
        actual_proposal.at("candidates").size() == 3 && actual_proposal.at("conditional_probabilities").size() == 3 &&
        actual_proposal.at("distribution") == (temperature > 0 ? "conditional_q" : "point_mass"), "invalid engine proposal descriptor");
    for (int row = 0; row < 3; ++row) {
        const int token = bounded(actual_proposal.at("tokens").at(row), 0, df::vocabulary_size - 1, "proposal token");
        const auto candidates = actual_proposal.at("candidates").at(row).get<std::array<int32_t, 16>>();
        const auto probabilities = actual_proposal.at("conditional_probabilities").at(row).get<std::array<float, 16>>();
        bool present = false; float sum = 0;
        std::set<int32_t> distinct;
        for (size_t slot = 0; slot < 16; ++slot) {
            io::require(candidates[slot] >= 0 && candidates[slot] < df::vocabulary_size && distinct.insert(candidates[slot]).second &&
                std::isfinite(probabilities[slot]) && probabilities[slot] >= 0 && (temperature > 0 || probabilities[slot] == 0),
                "invalid actual-engine candidate support");
            present |= candidates[slot] == token; sum += probabilities[slot];
        }
        io::require(present && (temperature == 0 || std::abs(sum - 1) <= 16 * std::numeric_limits<float>::epsilon()),
                    "actual-engine proposal absent from normalized support");
    }
    std::vector<std::byte> reference_hidden;
    if (!o.reference_hidden.empty()) {
        reference_hidden = io::read_file(o.reference_hidden, 4 * df::hidden_size * sizeof(uint16_t));
        io::require(reference_hidden.size() == 4 * df::hidden_size * sizeof(uint16_t), "CPU reference hidden must be exact BF16 [4,2560]");
        io::require_finite(reference_hidden, true);
    }
    const fs::path output = fs::weakly_canonical(o.output);
    for (const auto& input : {o.runtime, o.bundle, o.pack, o.tokenizer}) {
        const auto parent = fs::canonical(input);
        const auto compared = std::mismatch(parent.begin(), parent.end(), output.begin(), output.end());
        io::require(compared.first != parent.end(), "native output must remain outside immutable input directories");
    }
    io::require(fs::create_directory(output), "native output must be a new directory");
    io::fsync_directory(output.parent_path().empty() ? fs::path(".") : output.parent_path());
    uint64_t written = 0;
    Stream stream;
    df::DeviceForward forward(weights, o.device);
    auto context = forward.create_context(origin);
    df::DeviceSelector selector(o.device, forward.weight("candidate_selector.hidden_projection.weight"),
        forward.weight("candidate_selector.predecessor_codebook"), forward.weight("candidate_selector.successor_codebook"));
    Buffer raw(4 * df::hidden_size * sizeof(uint16_t), stream), hidden(4 * df::hidden_size * sizeof(uint16_t), stream);
    constexpr size_t replay_logits_bytes = 3 * df::vocabulary_size * sizeof(float);
    Buffer widened(df::hidden_size * sizeof(float), stream), logits(replay_logits_bytes, stream);
    Buffer head_scratch(head.projection_scratch_bytes(), stream);
    raw.upload(embeddings);
    df::FeatureView view;
    view.start = begin; view.count = count; view.request_id = request_id; view.incarnation = int64_t(incarnation);
    view.segments[0] = {reinterpret_cast<const uint16_t*>(features.data()), size_t(count) * df::feature_width};
    view.generations[0] = generations;
    TraceWriter trace(output, written);
    io::require(forward.forward(*context, view, raw.as<uint16_t>(), 4, frontier, hidden.as<uint16_t>(), stream.get(), error, &trace), error);
    Json outputs;
    outputs["replay_hidden"] = trace.download("replay-hidden.bf16", "bf16", {4, df::hidden_size}, hidden.as<void>(),
        4 * df::hidden_size * sizeof(uint16_t), stream);
    project(head, hidden, widened, logits, head_scratch, stream, error);
    outputs["replay_logits"] = trace.download("replay-logits.f32", "f32", {3, df::vocabulary_size}, logits.as<void>(), replay_logits_bytes, stream);
    df::Selection selected;
    io::require(selector.select(3, hidden.as<uint16_t>() + df::hidden_size, logits.as<float>(), anchor, temperature, uniforms,
        selected, stream.get(), error), error);
    Json result = {{"schema", 1}, {"kind", "strata-dflash2-native-parity-replay"}, {"status", "DIAGNOSTIC_ONLY"},
        {"runtime_manifest_sha256", o.manifest}, {"target", io::target_document(verified.binding().identity)},
        {"bundle_sha256", io::sha256(bundle_bytes)}, {"owner", owner}, {"origin", origin}, {"frontier", frontier},
        {"embedding_receipt", receipt_document(embedding.identity())}, {"head_receipt", receipt_document(head.identity())},
        {"replay_selection", selection_document(selected)}, {"trace", trace.entries()},
        {"context", {{"begin", context->begin()}, {"end", context->end()}, {"origin", context->origin()}}},
        {"replay_caveat", "fresh full-ring replay may differ from actual engine incremental-ingest arithmetic; never an identity quality gate"},
        {"verifier_acceptance", "NOT EXERCISED; engine proposal capture and real verifier window logs remain authoritative"}};
    if (!reference_hidden.empty()) {
        hidden.upload(reference_hidden);
        outputs["reference_hidden_input"] = io::write_tensor(output, "reference-hidden-input.bf16", "bf16", {4, df::hidden_size}, reference_hidden, written);
        project(head, hidden, widened, logits, head_scratch, stream, error);
        outputs["reference_head_logits"] = trace.download("reference-head-logits.f32", "f32", {3, df::vocabulary_size},
            logits.as<void>(), replay_logits_bytes, stream);
        df::Selection reference_selection;
        io::require(selector.select(3, hidden.as<uint16_t>() + df::hidden_size, logits.as<float>(), anchor, temperature, uniforms,
            reference_selection, stream.get(), error), error);
        result["reference_hidden_native_selection"] = selection_document(reference_selection);
        result["reference_head_source"] = "same verified loaded quantized NativeHead::run_with_scratch; BF16 widening only, no target norm";
    }
    stream.drain();
    result["outputs"] = std::move(outputs);
    result["trace_entries"] = trace.entries().size();
    io::write_json(output / "replay.json", result, written);
    std::cout << "Native parity diagnostics written to " << (output / "replay.json").string() << '\n';
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--help") { help(); return 0; }
    try { return run(options(argc, argv)); }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
