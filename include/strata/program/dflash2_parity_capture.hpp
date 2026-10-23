// Opt-in, synchronous diagnostics only. Never install this observer in a benchmark.
#pragma once

#include "strata/core/dflash2_drafter.hpp"

#include <cuda_runtime_api.h>
#include <nlohmann/json.hpp>
#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

namespace strata::program::dflash2_parity {
namespace io {
using Json = nlohmann::json;
namespace fs = std::filesystem;
inline constexpr uint64_t artifact_limit = uint64_t(1) << 30;
inline constexpr size_t json_limit = size_t(1) << 20;
inline void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error("dflash2 parity: " + message);
}
inline void cuda(cudaError_t result, const char* operation) {
    require(result == cudaSuccess, std::string(operation) + ": " + cudaGetErrorString(result));
}
class DeviceScope {
public:
    explicit DeviceScope(int device) {
        cuda(cudaGetDevice(&previous_), "get previous device");
        cuda(cudaSetDevice(device), "select diagnostic device");
    }
    ~DeviceScope() { if (cudaSetDevice(previous_) != cudaSuccess) std::terminate(); }
    DeviceScope(const DeviceScope&) = delete;
    DeviceScope& operator=(const DeviceScope&) = delete;
private:
    int previous_ = -1;
};
class PinnedBuffer {
public:
    explicit PinnedBuffer(size_t bytes) : bytes_(bytes) {
        cuda(cudaHostAlloc(&data_, bytes, cudaHostAllocPortable), "allocate diagnostic host readback");
    }
    ~PinnedBuffer() { if (data_ && cudaFreeHost(data_) != cudaSuccess) std::terminate(); }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    std::byte* data() { return static_cast<std::byte*>(data_); }
    size_t size() const { return bytes_; }
private:
    void* data_ = nullptr;
    size_t bytes_;
};
inline bool sha256_valid(const std::string& value) {
    return value.size() == 64 && value != std::string(64, '0') &&
        std::all_of(value.begin(), value.end(), [](char c) {
            return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
        });
}
inline std::string sha256(std::span<const std::byte> bytes) {
    std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
    unsigned size = 0;
    require(EVP_Digest(bytes.data(), bytes.size(), digest.data(), &size, EVP_sha256(), nullptr) == 1 && size == 32,
            "SHA-256 failed");
    constexpr char digits[] = "0123456789abcdef";
    std::string result(64, '0');
    for (size_t i = 0; i < 32; ++i) {
        result[2 * i] = digits[digest[i] >> 4]; result[2 * i + 1] = digits[digest[i] & 15];
    }
    return result;
}
inline Json target_document(const core::dflash2::TargetIdentity& identity) {
    for (const auto* value : {&identity.pack_sha256, &identity.embedding_sha256,
                              &identity.head_sha256, &identity.tokenizer_sha256})
        require(sha256_valid(*value), "invalid independently loaded target identity");
    return {{"pack_sha256", identity.pack_sha256}, {"embedding_sha256", identity.embedding_sha256},
            {"head_sha256", identity.head_sha256}, {"tokenizer_sha256", identity.tokenizer_sha256}};
}
inline void fsync_directory(const fs::path& path) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
    require(fd >= 0, "open artifact directory: " + path.string());
    const int status = ::fsync(fd), saved = errno;
    const int closed = ::close(fd);
    require(status == 0 && closed == 0, "sync artifact directory: " + std::string(std::strerror(saved)));
}
inline void write_file(const fs::path& path, std::span<const std::byte> bytes) {
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
    require(fd >= 0, "create artifact: " + path.string());
    size_t cursor = 0;
    bool good = true;
    while (cursor < bytes.size()) {
        const ssize_t n = ::write(fd, bytes.data() + cursor, bytes.size() - cursor);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) { good = false; break; }
        cursor += size_t(n);
    }
    if (good) good = ::fsync(fd) == 0;
    const int saved = errno;
    if (::close(fd) != 0) good = false;
    require(good, "write/sync artifact " + path.string() + ": " + std::strerror(saved));
}
inline std::string json_bytes(const Json& document) { return document.dump(2) + "\n"; }
inline void write_json(const fs::path& path, const Json& document, uint64_t& written) {
    const std::string bytes = json_bytes(document);
    require(bytes.size() <= json_limit && written <= artifact_limit - bytes.size(), "artifact JSON/budget exceeded");
    const auto partial = fs::path(path.string() + ".partial");
    write_file(partial, std::as_bytes(std::span(bytes.data(), bytes.size())));
    fs::rename(partial, path);
    fsync_directory(path.parent_path());
    written += bytes.size();
}
inline Json write_tensor(const fs::path& root, const std::string& name, std::string_view dtype,
                         const std::vector<int>& shape, std::span<const std::byte> bytes, uint64_t& written) {
    require(written <= artifact_limit && bytes.size() <= artifact_limit - written, "1-GiB artifact budget exceeded");
    require(fs::path(name).filename() == name && !name.empty(), "invalid tensor filename");
    size_t elements = 1;
    for (int extent : shape) {
        require(extent >= 0 && (extent == 0 || elements <= artifact_limit / size_t(extent)), "tensor extent overflow");
        elements *= size_t(extent);
    }
    const size_t width = dtype == "bf16" ? 2 : dtype == "f32" || dtype == "i32" ? 4 : dtype == "i64" ? 8 : 0;
    require(width && elements <= artifact_limit / width && elements * width == bytes.size(), "tensor dtype/extent mismatch");
    const std::string digest = sha256(bytes);
    write_file(root / name, bytes); written += bytes.size();
    return {{"file", name}, {"dtype", dtype}, {"shape", shape}, {"bytes", bytes.size()}, {"sha256", digest},
            {"byte_order", "little"}, {"layout", "row_major"}};
}
inline std::vector<std::byte> read_file(const fs::path& path, size_t limit) {
    require(!fs::is_symlink(path) && fs::is_regular_file(path), "artifact is not a regular file: " + path.string());
    const auto size = fs::file_size(path);
    require(size <= limit, "artifact exceeds finite read bound: " + path.string());
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    require(fd >= 0, "open artifact: " + path.string());
    std::vector<std::byte> result(size);
    size_t cursor = 0;
    bool good = true;
    while (cursor < size) {
        const ssize_t n = ::read(fd, result.data() + cursor, size - cursor);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) { good = false; break; }
        cursor += size_t(n);
    }
    std::byte extra{};
    ssize_t end;
    do { end = ::read(fd, &extra, 1); } while (end < 0 && errno == EINTR);
    if (::close(fd) != 0) good = false;
    require(good && end == 0, "truncated/changed artifact: " + path.string());
    return result;
}
inline Json parse_json(std::span<const std::byte> bytes) {
    // nlohmann's parser callback refuses duplicate object keys at every depth.
    std::vector<std::vector<std::string>> keys;
    require(!bytes.empty(), "empty artifact JSON");
    const auto* begin = reinterpret_cast<const char*>(bytes.data());
    return Json::parse(begin, begin + bytes.size(), [&keys](int depth, Json::parse_event_t event, Json& value) {
        if (event == Json::parse_event_t::object_start) {
            if (keys.size() <= size_t(depth)) keys.resize(size_t(depth) + 1);
            keys[size_t(depth)].clear();
        } else if (event == Json::parse_event_t::key) {
            require(depth > 0 && keys.size() >= size_t(depth), "invalid JSON object depth");
            auto& seen = keys[size_t(depth - 1)];
            const auto key = value.get<std::string>();
            require(std::find(seen.begin(), seen.end(), key) == seen.end(), "duplicate artifact JSON key");
            seen.push_back(key);
        }
        return true;
    });
}
inline Json read_json(const fs::path& path) {
    const auto bytes = read_file(path, json_limit);
    return parse_json(bytes);
}
inline std::vector<std::byte> read_tensor(const fs::path& root, const Json& descriptor,
                                        std::string_view dtype, const std::vector<int>& shape) {
    const auto name = descriptor.at("file").get<std::string>();
    require(!name.empty() && fs::path(name).filename() == name && descriptor.at("dtype") == dtype &&
            descriptor.at("shape") == shape && descriptor.at("byte_order") == "little" &&
            descriptor.at("layout") == "row_major", "tensor descriptor mismatch");
    size_t elements = 1;
    for (const auto extent : shape) {
        require(extent >= 0 && (extent == 0 || elements <= artifact_limit / size_t(extent)), "tensor shape overflow");
        elements *= size_t(extent);
    }
    const size_t width = dtype == "bf16" ? 2 : dtype == "f32" || dtype == "i32" ? 4 : dtype == "i64" ? 8 : 0;
    require(width && elements <= artifact_limit / width && descriptor.at("bytes") == elements * width,
            "tensor byte extent mismatch");
    auto bytes = read_file(root / name, elements * width);
    require(bytes.size() == elements * width && descriptor.at("sha256") == sha256(bytes), "tensor size/SHA-256 mismatch");
    return bytes;
}
inline void require_finite(std::span<const std::byte> bytes, bool bf16) {
    const size_t width = bf16 ? 2 : 4;
    require(bytes.size() % width == 0, "unaligned floating tensor");
    for (size_t i = 0; i < bytes.size(); i += width) {
        uint32_t bits = 0;
        if (bf16) { uint16_t word; std::memcpy(&word, bytes.data() + i, 2); bits = uint32_t(word) << 16; }
        else std::memcpy(&bits, bytes.data() + i, 4);
        require(std::isfinite(std::bit_cast<float>(bits)), "nonfinite captured floating tensor");
    }
}
} // namespace io

// Own one bounded readback allocation only when DFLASH2_CAPTURE_DIR is enabled.
// Attach with drafter.set_proposal_observer(&ProposalBundleWriter::observe, &writer).
// Writer MUST outlive the drafter's final operation. This is a finite ONE-request,
// eight-output diagnostic: any ninth proposal or second owner refuses, never skips.
class ProposalBundleWriter {
public:
    using Capture = core::dflash2::ProposalCapture;
    static constexpr size_t planned_host_bytes() {
        return core::dflash2::max_feature_bytes + core::dflash2::context_rows * sizeof(int64_t) +
               2 * core::dflash2::block_size * core::dflash2::hidden_size * sizeof(uint16_t) + core::dflash2::logits_bytes;
    }
    ProposalBundleWriter(const std::string& directory, core::dflash2::TargetIdentity actual_target,
                         std::string independent_manifest_sha256)
        : root_(directory), target_(std::move(actual_target)), manifest_(std::move(independent_manifest_sha256)),
          host_(planned_host_bytes()) {
        io::require(std::endian::native == std::endian::little, "capture requires little-endian host");
        io::target_document(target_);
        io::require(io::sha256_valid(manifest_) && !root_.empty(), "capture requires independent runtime digest and directory");
        io::require(io::fs::create_directory(root_), "capture output must be a new directory");
        io::fsync_directory(root_.parent_path().empty() ? io::fs::path(".") : root_.parent_path());
        index_ = {{"schema", 1}, {"kind", "strata-dflash2-real-proposal-capture"}, {"runtime_manifest_sha256", manifest_},
                  {"target", io::target_document(target_)}, {"output_request_bound", 8}, {"byte_limit", io::artifact_limit},
                  {"windows", io::Json::array()}, {"verifier_acceptance", "not captured by this observer"}};
        io::write_json(root_ / "capture.json", index_, written_);
    }
    size_t allocated_host_bytes() const noexcept { return host_.size(); }
    static bool observe(void* user, const Capture& capture, std::string& error) noexcept {
        if (!user) { error = "dflash2 parity: absent proposal writer"; return false; }
        return static_cast<ProposalBundleWriter*>(user)->capture(capture, error);
    }
private:
    bool capture(const Capture& c, std::string& error) noexcept {
        try {
            using namespace core::dflash2;
            io::require(!poisoned_ && windows_ < 8, "capture poisoned or eight-proposal bound exceeded");
            io::require(c.owner.request && c.owner.generation && c.owner.seat >= 0 && c.owner.seat <= 8 &&
                        c.proposal.owner == c.owner && c.proposal.frontier == c.frontier && c.proposal.count == 3,
                        "capture proposal ownership/extent mismatch");
            io::require(c.owner.request <= uint64_t(std::numeric_limits<int64_t>::max()) &&
                        c.owner.incarnation <= uint64_t(std::numeric_limits<int64_t>::max()) &&
                        c.owner.generation <= uint64_t(std::numeric_limits<int64_t>::max()),
                        "capture owner exceeds signed reference operation bounds");
            io::require(c.origin >= 0 && c.frontier >= c.origin && c.frontier <= max_position - reference_block_size &&
                        c.anchor >= 0 && c.anchor < vocabulary_size && c.device >= 0 &&
                        c.raw_embeddings_device && c.hidden_device && c.logits_device,
                        "invalid real proposal buffers/range");
            io::require(c.context.start == std::max(c.origin, c.frontier - context_rows) &&
                        c.context.count == c.frontier - c.context.start &&
                        c.context.request_id == std::to_string(c.owner.request) &&
                        c.context.incarnation >= 0 && uint64_t(c.context.incarnation) == c.owner.incarnation,
                        "full chronological current feature ring unavailable (restored historical features cannot be invented)");
            io::require(std::isfinite(c.temperature) && c.temperature >= 0 &&
                        ((c.temperature > 0) == (c.proposal.distribution == core::DraftProposal::Distribution::conditional_q)),
                        "capture temperature/distribution mismatch");
            for (float value : c.uniforms) io::require(std::isfinite(value) && value >= 0 && value < 1, "invalid actual selector uniform");
            io::require(c.proposal.validate(vocabulary_size, error, core::DraftKind::dflash2), error);
            if (windows_) io::require(c.owner.request == owner_.request && c.owner.incarnation == owner_.incarnation &&
                        c.owner.seat == owner_.seat && c.owner.generation > owner_.generation && c.frontier > frontier_,
                        "capture changed owner or repeated/stale proposal");
            const int proposal_rows = c.proposal.count;
            const int forward_rows = proposal_rows + 1;
            const size_t forward_bytes = size_t(forward_rows) * hidden_size * sizeof(uint16_t);
            const size_t live_logits_bytes = size_t(proposal_rows) * vocabulary_size * sizeof(float);
            auto* features = host_.data();
            auto* tags = features + max_feature_bytes;
            auto* embeddings = tags + context_rows * sizeof(int64_t);
            auto* hidden = embeddings + block_size * hidden_size * sizeof(uint16_t);
            auto* logits = hidden + block_size * hidden_size * sizeof(uint16_t);
            size_t feature_bytes = 0, tag_bytes = 0;
            int64_t last_generation = 0;
            for (size_t i = 0; i < c.context.segments.size(); ++i) {
                const auto part = c.context.segments[i]; const auto generations = c.context.generations[i];
                io::require(part.size() % feature_width == 0 && generations.size() == part.size() / feature_width &&
                            part.size_bytes() <= max_feature_bytes - feature_bytes, "feature/tag segments mismatch");
                if (!part.empty()) std::memcpy(features + feature_bytes, part.data(), part.size_bytes());
                for (auto generation : generations) {
                    io::require(generation > 0 && generation >= last_generation, "invalid chronological feature generation");
                    last_generation = generation;
                }
                if (!generations.empty()) std::memcpy(tags + tag_bytes, generations.data(), generations.size_bytes());
                feature_bytes += part.size_bytes(); tag_bytes += generations.size_bytes();
            }
            io::require(feature_bytes == size_t(c.context.count) * feature_row_bytes &&
                        tag_bytes == size_t(c.context.count) * sizeof(int64_t), "incomplete captured context");
            {
                io::DeviceScope device(c.device);
                const auto stream = static_cast<cudaStream_t>(c.stream);
                try {
                    io::cuda(cudaMemcpyAsync(embeddings, c.raw_embeddings_device, forward_bytes,
                                             cudaMemcpyDeviceToHost, stream), "read actual raw embeddings");
                    io::cuda(cudaMemcpyAsync(hidden, c.hidden_device, forward_bytes,
                                             cudaMemcpyDeviceToHost, stream), "read actual engine hidden");
                    io::cuda(cudaMemcpyAsync(logits, c.logits_device, live_logits_bytes, cudaMemcpyDeviceToHost, stream),
                             "read actual engine logits");
                    io::cuda(cudaStreamSynchronize(stream), "complete real proposal readback");
                } catch (...) {
                    // Returning host storage while any submitted readback is live is unsafe.
                    if (cudaStreamSynchronize(stream) != cudaSuccess) std::terminate();
                    throw;
                }
            }
            const auto floats = [&](const std::byte* data, size_t bytes, bool bf16) { io::require_finite({data, bytes}, bf16); };
            floats(features, feature_bytes, true);
            floats(embeddings, forward_bytes, true);
            floats(hidden, forward_bytes, true);
            floats(logits, live_logits_bytes, false);
            io::Json candidates = io::Json::array(), probabilities = io::Json::array(), q = io::Json::array();
            for (size_t row = 0; row < 3; ++row) {
                float sum = 0; bool chosen = false;
                for (size_t j = 0; j < 16; ++j) {
                    const auto token = c.candidates[row][j]; const float probability = c.conditional_probabilities[row][j];
                    io::require(token >= 0 && token < vocabulary_size && std::isfinite(probability) && probability >= 0,
                                "invalid captured selector support");
                    for (size_t k = 0; k < j; ++k) io::require(c.candidates[row][k] != token, "duplicate captured candidate");
                    chosen |= token == c.proposal.tokens[row]; sum += probability;
                    if (c.temperature > 0) {
                        const auto base = row * core::kSpecQStride;
                        io::require(c.proposal.q[base + j] == token &&
                            std::bit_cast<float>(c.proposal.q[base + core::kSpecQEntries + j]) == probability,
                            "proposal q differs from captured selector q");
                    } else io::require(probability == 0, "greedy selector unexpectedly exposed q");
                }
                io::require(chosen && (c.temperature == 0 || std::abs(sum - 1.0f) <= 16 * std::numeric_limits<float>::epsilon()),
                            "captured chosen token/q normalization invalid");
                candidates.push_back(c.candidates[row]); probabilities.push_back(c.conditional_probabilities[row]);
                if (c.temperature > 0) q.push_back({{"ids", c.candidates[row]}, {"probabilities", c.conditional_probabilities[row]}});
            }
            const auto name = std::string("proposal-") + std::to_string(windows_);
            const auto directory = root_ / name;
            io::require(io::fs::create_directory(directory), "proposal directory already exists");
            io::Json files = io::Json::object();
            files["context"] = io::write_tensor(directory, "context.bf16", "bf16", {c.context.count, feature_width}, {features, feature_bytes}, written_);
            files["generations"] = io::write_tensor(directory, "generations.i64", "i64", {c.context.count}, {tags, tag_bytes}, written_);
            files["raw_embeddings"] = io::write_tensor(directory, "embeddings.bf16", "bf16", {forward_rows, hidden_size},
                {embeddings, forward_bytes}, written_);
            files["engine_hidden"] = io::write_tensor(directory, "engine-hidden.bf16", "bf16", {forward_rows, hidden_size},
                {hidden, forward_bytes}, written_);
            files["engine_logits"] = io::write_tensor(directory, "engine-logits.f32", "f32", {proposal_rows, vocabulary_size},
                {logits, live_logits_bytes}, written_);
            io::Json bundle = {{"schema", 1}, {"kind", "strata-dflash2-real-proposal"}, {"runtime_manifest_sha256", manifest_},
                {"target", io::target_document(target_)}, {"owner", {{"seat", c.owner.seat}, {"request", c.owner.request},
                    {"incarnation", c.owner.incarnation}, {"generation", c.owner.generation}}},
                {"origin", c.origin}, {"frontier", c.frontier}, {"anchor", c.anchor},
                {"context", {{"begin", c.context.start}, {"count", c.context.count}, {"request_id", std::string(c.context.request_id)}}},
                {"temperature", c.temperature}, {"uniforms", c.uniforms}, {"files", std::move(files)},
                {"native_proposal", {{"count", 3}, {"tokens", {c.proposal.tokens[0], c.proposal.tokens[1], c.proposal.tokens[2]}},
                    {"distribution", c.temperature > 0 ? "conditional_q" : "point_mass"}, {"candidates", std::move(candidates)},
                    {"conditional_probabilities", std::move(probabilities)}, {"q", std::move(q)}}},
                {"verifier_acceptance", "not observed; join real verifier window log by owner and frontier"}};
            io::write_json(directory / "bundle.json", bundle, written_);
            const std::string bundle_bytes = io::json_bytes(bundle);
            index_["windows"].push_back({{"directory", name}, {"owner", bundle["owner"]}, {"frontier", c.frontier},
                {"bundle_sha256", io::sha256(std::as_bytes(std::span(bundle_bytes.data(), bundle_bytes.size())))}});
            io::write_json(root_ / "capture.json", index_, written_);
            owner_ = c.owner; frontier_ = c.frontier; ++windows_; error.clear(); return true;
        } catch (const std::exception& failure) { poisoned_ = true; error = failure.what(); return false; }
          catch (...) { poisoned_ = true; error = "dflash2 parity: unexpected capture failure"; return false; }
    }
    io::fs::path root_;
    core::dflash2::TargetIdentity target_;
    std::string manifest_;
    io::PinnedBuffer host_;
    io::Json index_;
    uint64_t written_ = 0;
    size_t windows_ = 0;
    core::DraftOwner owner_;
    int frontier_ = -1;
    bool poisoned_ = false;
};
} // namespace strata::program::dflash2_parity
