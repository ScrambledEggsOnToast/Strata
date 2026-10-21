#include "strata/core/dflash2_weights.hpp"
#include "conversation_store_sha256.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cmath>
#include <dirent.h>
#include <fcntl.h>
#include <functional>
#include <initializer_list>
#include <limits>
#include <map>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

namespace strata::core::dflash2 {
namespace {
using Json = nlohmann::json;
using conversation_store_detail::Sha256;
using conversation_store_detail::sha256;
using conversation_store_detail::sha256_hex;
using TensorMap = std::map<std::string, RuntimeTensor, std::less<>>;

static_assert(std::endian::native == std::endian::little,
              "DFlash2 runtime weights require a little-endian host");
static_assert(sizeof(size_t) >= sizeof(uint64_t), "DFlash2 runtime mapping requires a 64-bit host");
constexpr size_t alignment = 64;
constexpr size_t runtime_header_bytes = 1u << 20;
constexpr size_t tensor_count = 81;
constexpr uint64_t checkpoint_bytes = 1206140008;
constexpr char config_sha256[] = "2409575986f9ea24d455343eb469c0e6940def5edab1d1be503bb8eb4cc3b1b9";
constexpr char checkpoint_sha256[] = "d6723f2a359eaf33ec0c95f22ae8edda896774020c552296a85e1913b4d3186f";
constexpr char reference_sha256[] = "f55b7fe0a4c0b3073e0f9cdce547cce29f4b8e2168c4d2818760007c43b7651e";
constexpr char drafter_revision[] = "a22a1744e85c17364dcc54a6a420af7135d4c7f6";
constexpr std::array<std::string_view, 3> filenames{"manifest.json", "config.json", "dense.bin"};

void require(bool condition, const char* message) {
    if (!condition) throw Refusal(message);
}
void validate_digest(std::string_view value) {
    require(value.size() == 64 && std::all_of(value.begin(), value.end(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    }), "invalid DFlash2 runtime lowercase SHA-256");
}
const std::string& digest_field(const Json& value) {
    require(value.is_string(), "invalid DFlash2 runtime SHA-256 field");
    const auto& result = value.get_ref<const std::string&>();
    validate_digest(result);
    return result;
}
void fields(const Json& value, std::initializer_list<const char*> names, const char* message) {
    require(value.is_object() && value.size() == names.size(), message);
    for (const char* name : names) require(value.contains(name), message);
}
uint64_t integer(const Json& value, uint64_t low, uint64_t high, const char* message) {
    require(value.is_number_integer(), message);
    uint64_t result;
    if (value.is_number_unsigned()) {
        result = value.get<uint64_t>();
    } else {
        const auto signed_value = value.get<int64_t>();
        require(signed_value >= 0, message);
        result = uint64_t(signed_value);
    }
    require(result >= low && result <= high, message);
    return result;
}

// Match Python's canonical-document comparison: integer 1, float 1.0 and
// boolean true are distinct. Signed/unsigned storage of the same JSON integer
// is not a format distinction. Only rms_norm_eps is a floating contract field.
bool same_document(const Json& actual, const Json& expected) {
    if (expected.is_number_integer()) return actual.is_number_integer() && actual == expected;
    if (actual.type() != expected.type()) return false;
    if (expected.is_object()) {
        if (actual.size() != expected.size()) return false;
        for (auto it = expected.begin(); it != expected.end(); ++it) {
            const auto found = actual.find(it.key());
            if (found == actual.end() || !same_document(*found, it.value())) return false;
        }
        return true;
    }
    if (expected.is_array()) {
        if (actual.size() != expected.size()) return false;
        for (size_t i = 0; i < expected.size(); ++i)
            if (!same_document(actual[i], expected[i])) return false;
        return true;
    }
    return actual == expected;
}
Json strict_manifest(std::string_view wire) {
    // Reuse the bridge's duplicate-key callback convention. Bound recursion
    // before descending; every valid v1 document is much shallower than this.
    std::array<std::vector<std::string>, 32> keys;
    size_t objects = 0;
    auto callback = [&](int depth, Json::parse_event_t event, Json& parsed) {
        require(depth >= 0 && size_t(depth) < keys.size(), "DFlash2 runtime manifest nesting limit exceeded");
        if (event == Json::parse_event_t::object_start) {
            require(objects < keys.size(), "DFlash2 runtime object nesting limit exceeded");
            keys[objects++].clear();
        } else if (event == Json::parse_event_t::key) {
            require(objects != 0, "unowned DFlash2 runtime JSON key");
            auto& names = keys[objects - 1];
            const auto& key = parsed.get_ref<const std::string&>();
            require(std::find(names.begin(), names.end(), key) == names.end(), "duplicate DFlash2 runtime JSON key");
            names.push_back(key);
        } else if (event == Json::parse_event_t::object_end) {
            require(objects != 0, "invalid DFlash2 runtime JSON object");
            --objects;
        } else if (event == Json::parse_event_t::value && parsed.is_number_float()) {
            require(std::isfinite(parsed.get<double>()), "nonfinite DFlash2 runtime JSON number");
        }
        return true;
    };
    try {
        return Json::parse(wire.begin(), wire.end(), callback);
    } catch (const Json::exception&) {
        throw Refusal("malformed DFlash2 runtime JSON manifest");
    }
}

using Stamp = std::array<uint64_t, 7>;
Stamp stamp(const struct stat& value) {
    // Same inode/size/mtime/ctime identity as native_weight_identity.hpp's
    // NativeSourceIdentity, but taken from the actual opened descriptor.
    return {uint64_t(value.st_dev), uint64_t(value.st_ino), uint64_t(value.st_size),
            uint64_t(value.st_mtim.tv_sec), uint64_t(value.st_mtim.tv_nsec),
            uint64_t(value.st_ctim.tv_sec), uint64_t(value.st_ctim.tv_nsec)};
}
struct OwnedFd {
    explicit OwnedFd(int value) : value(value) {}
    ~OwnedFd() { if (value >= 0) ::close(value); }
    OwnedFd(const OwnedFd&) = delete;
    OwnedFd& operator=(const OwnedFd&) = delete;
    int release() noexcept { return std::exchange(value, -1); }
    int value;
};
struct DirectoryCloser {
    void operator()(DIR* directory) const noexcept { ::closedir(directory); }
};
class RuntimeDirectory {
public:
    explicit RuntimeDirectory(const std::string& path)
        : path_(path), fd_(::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC)) {
        require(fd_.value >= 0 && ::fstat(fd_.value, &initial_) == 0 && S_ISDIR(initial_.st_mode),
                "DFlash2 runtime directory is unavailable");
        exact_files();
    }
    int fd() const { return fd_.value; }
    void exact_files() const {
        // A separate open description avoids sharing a readdir offset between
        // the initial and final exact-directory checks.
        OwnedFd listing(::openat(fd_.value, ".", O_RDONLY | O_DIRECTORY | O_CLOEXEC));
        require(listing.value >= 0, "cannot list DFlash2 runtime directory");
        DIR* raw = ::fdopendir(listing.value);
        require(raw != nullptr, "cannot read DFlash2 runtime directory");
        listing.release();
        std::unique_ptr<DIR, DirectoryCloser> directory(raw);
        std::array<bool, filenames.size()> seen{};
        for (;;) {
            errno = 0;
            const auto* entry = ::readdir(directory.get());
            if (!entry) {
                require(errno == 0, "DFlash2 runtime directory read failed");
                break;
            }
            const std::string_view name(entry->d_name);
            if (name == "." || name == "..") continue;
            const auto found = std::find(filenames.begin(), filenames.end(), name);
            require(found != filenames.end(), "unexpected DFlash2 runtime directory file");
            const size_t index = size_t(found - filenames.begin());
            require(!seen[index], "duplicate DFlash2 runtime directory entry");
            seen[index] = true;
        }
        require(std::all_of(seen.begin(), seen.end(), [](bool value) { return value; }),
                "missing DFlash2 runtime directory file");
    }
    void unchanged() const {
        struct stat opened{}, named{};
        require(::fstat(fd_.value, &opened) == 0 && ::stat(path_.c_str(), &named) == 0 &&
                S_ISDIR(named.st_mode) && stamp(opened) == stamp(initial_) && stamp(named) == stamp(initial_),
                "DFlash2 runtime directory changed during loading");
    }
private:
    const std::string& path_;
    OwnedFd fd_;
    struct stat initial_{};
};
class RuntimeFile {
public:
    RuntimeFile(int directory, const char* name)
        : directory_(directory), name_(name),
          fd_(::openat(directory, name, O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK)) {
        require(fd_.value >= 0 && ::fstat(fd_.value, &initial_) == 0 &&
                S_ISREG(initial_.st_mode) && initial_.st_size >= 0,
                "DFlash2 runtime file is not regular non-symlink storage");
        unchanged();
    }
    int fd() const { return fd_.value; }
    uint64_t size() const { return uint64_t(initial_.st_size); }
    std::string manifest_bytes() const {
        require(size() <= runtime_header_bytes, "oversized DFlash2 runtime manifest");
        std::string result(size_t(size()), '\0');
        size_t position = 0;
        while (position < result.size())
            position += read_at(result.data() + position, result.size() - position, position);
        return result;
    }
    std::string file_digest() const {
        Sha256 hash;
        std::array<uint8_t, 64u << 10> buffer;
        uint64_t position = 0;
        while (position < size()) {
            const size_t count = read_at(buffer.data(), size_t(std::min<uint64_t>(buffer.size(), size() - position)), position);
            hash.update(buffer.data(), count);
            position += count;
        }
        return sha256_hex(hash.digest());
    }
    void unchanged() const {
        struct stat opened{}, named{};
        require(::fstat(fd_.value, &opened) == 0 &&
                ::fstatat(directory_, name_, &named, AT_SYMLINK_NOFOLLOW) == 0 && S_ISREG(named.st_mode) &&
                stamp(opened) == stamp(initial_) && stamp(named) == stamp(initial_),
                "DFlash2 runtime file changed or was replaced during loading");
    }
private:
    size_t read_at(void* output, size_t count, uint64_t position) const {
        ssize_t result;
        do {
            result = ::pread(fd_.value, output, count, off_t(position));
        } while (result < 0 && errno == EINTR);
        require(result > 0, "truncated/unreadable DFlash2 runtime file");
        return size_t(result);
    }
    int directory_;
    const char* name_;
    OwnedFd fd_;
    struct stat initial_{};
};
class ReadOnlyMapping {
public:
    ReadOnlyMapping() = default;
    ~ReadOnlyMapping() { if (size_) ::munmap(const_cast<uint8_t*>(data_), size_); }
    ReadOnlyMapping(const ReadOnlyMapping&) = delete;
    ReadOnlyMapping& operator=(const ReadOnlyMapping&) = delete;
    void map(int fd, size_t size) {
        require(size != 0 && size_ == 0, "invalid DFlash2 runtime mapping extent");
        const void* address = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        require(address != MAP_FAILED, "cannot mmap DFlash2 runtime dense.bin");
        data_ = static_cast<const uint8_t*>(address);
        size_ = size;
    }
    const uint8_t* data() const { return data_; }
    size_t size() const { return size_; }
private:
    const uint8_t* data_ = nullptr;
    size_t size_ = 0;
};

Json expected_source() {
    return {{"repository", "incoai/Qwen3.8-Flash-Next-DFlash2"}, {"revision", drafter_revision},
            {"checkpoint_sha256", checkpoint_sha256}, {"checkpoint_bytes", checkpoint_bytes},
            {"config_sha256", config_sha256}, {"reference_commit", reference_commit},
            {"reference_sha256", reference_sha256}};
}
Json expected_contract() {
    return {
        {"layers", 5}, {"hidden_size", hidden_size}, {"intermediate_size", 7680},
        {"query_heads", 32}, {"kv_heads", 8}, {"head_dim", 128},
        {"kv_layout", "batch,kv_head,absolute_context_row,head_dim"},
        {"kv_bytes_per_context_row", 20480}, {"sliding_window", 2048}, {"is_causal", false},
        {"published_block_size", 8}, {"reference_block_positions", block_size},
        {"features", {{"tap_layers", tap_layers}, {"tap_reduction", "arithmetic_mean_of_four_unmixed_streams"},
                      {"width", feature_width}, {"projection", "hidden_norm(fc(concat(tap_means)))"},
                      {"normalization", "ordinary_rmsnorm"}, {"rms_norm_eps", 1e-6},
                      {"norm_weight_shift", 0}, {"stream_mixer", nullptr}}},
        {"convolution", {{"kernel_size", 2}, {"group_size", 16}}},
        {"selector", {{"rank", 256}, {"top_k", 16}}},
        {"borrowed", {{"embedding", "raw_input_rows"}, {"head", "no_target_final_norm"}}},
        {"dtype", "BF16"}, {"byte_order", "little"}, {"layout", "row_major"}, {"alignment", alignment}
    };
}
Json expected_target(const TargetIdentity& target) {
    return {
        {"identity", {{"pack_sha256", target.pack_sha256}, {"embedding_sha256", target.embedding_sha256},
                      {"head_sha256", target.head_sha256}, {"tokenizer_sha256", target.tokenizer_sha256}}},
        {"vocabulary_size", vocabulary_size}, {"embedding_shape", {vocabulary_size, hidden_size}},
        {"head_shape", {vocabulary_size, hidden_size}}
    };
}
TensorMap expected_tensors() {
    TensorMap result;
    auto add = [&](std::string name, std::initializer_list<int64_t> shape) {
        RuntimeTensor tensor;
        tensor.shape.assign(shape);
        result.emplace(std::move(name), std::move(tensor));
    };
    add("fc.weight", {2560, 12800});
    add("hidden_norm.weight", {2560});
    add("norm.weight", {2560});
    add("candidate_selector.hidden_projection.weight", {256, 2560});
    add("candidate_selector.predecessor_codebook", {248320, 256});
    add("candidate_selector.successor_codebook", {248320, 256});
    for (int layer = 0; layer < 5; ++layer) {
        const std::string prefix = "layers." + std::to_string(layer) + ".";
        add(prefix + "self_attn.q_proj.weight", {4096, 2560});
        add(prefix + "self_attn.k_proj.weight", {1024, 2560});
        add(prefix + "self_attn.v_proj.weight", {1024, 2560});
        add(prefix + "self_attn.o_proj.weight", {2560, 4096});
        add(prefix + "self_attn.q_norm.weight", {128});
        add(prefix + "self_attn.k_norm.weight", {128});
        add(prefix + "input_layernorm.weight", {2560});
        add(prefix + "post_attention_layernorm.weight", {2560});
        add(prefix + "mlp.gate_proj.weight", {7680, 2560});
        add(prefix + "mlp.up_proj.weight", {7680, 2560});
        add(prefix + "mlp.down_proj.weight", {2560, 7680});
        for (const char* convolution : {"attention_conv", "mlp_conv"}) {
            add(prefix + convolution + ".base_kernel", {2, 2, 2560});
            add(prefix + convolution + ".kernel_projection.weight", {640, 2560});
        }
    }
    return result;
}
size_t tensor_bytes(const RuntimeTensor& tensor) {
    size_t result = sizeof(uint16_t);
    for (const int64_t dimension : tensor.shape) result *= size_t(dimension);
    return result;
}
} // namespace

struct RuntimeWeights::Impl {
    // The map is destroyed after all its borrowed tensor spans.
    ReadOnlyMapping mapping;
    TensorMap tensors;
    TargetIdentity target;
    std::string manifest_digest;

    Impl(const std::string& path, const std::string& independent_manifest_sha256,
         const TargetIdentity& borrowed_target)
        : target(borrowed_target), manifest_digest(independent_manifest_sha256) {
        validate_digest(manifest_digest);
        for (const auto* value : {&target.pack_sha256, &target.embedding_sha256,
                                  &target.head_sha256, &target.tokenizer_sha256}) validate_digest(*value);
        RuntimeDirectory directory(path);
        RuntimeFile manifest_file(directory.fd(), "manifest.json");
        RuntimeFile config_file(directory.fd(), "config.json");
        RuntimeFile dense_file(directory.fd(), "dense.bin");
        const std::string wire = manifest_file.manifest_bytes();
        require(sha256_hex(sha256(wire.data(), wire.size())) == manifest_digest,
                "DFlash2 runtime manifest SHA-256 differs from independent identity");
        const Json manifest = strict_manifest(wire);
        fields(manifest, {"format", "source", "contract", "target", "source_payload_offset", "data", "tensors", "roundtrip"},
               "invalid DFlash2 runtime manifest fields");
        require(manifest.at("format") == "strata-dflash2-bf16-v1", "unsupported DFlash2 runtime format");
        require(same_document(manifest.at("source"), expected_source()), "DFlash2 runtime checkpoint/reference identity mismatch");
        require(same_document(manifest.at("contract"), expected_contract()), "DFlash2 runtime feature/KV geometry contract mismatch");
        require(same_document(manifest.at("target"), expected_target(target)), "DFlash2 runtime borrowed target identity/shape mismatch");
        require(config_file.file_digest() == config_sha256, "DFlash2 runtime config.json SHA-256 mismatch");

        tensors = expected_tensors();
        const auto& entries = manifest.at("tensors");
        require(entries.is_object() && entries.size() == tensor_count && tensors.size() == tensor_count,
                "missing/unexpected DFlash2 runtime tensors");
        const uint64_t source_base = integer(manifest.at("source_payload_offset"), 10, runtime_header_bytes + 8,
                                             "invalid DFlash2 source payload offset");
        std::array<std::pair<uint64_t, uint64_t>, tensor_count> source_intervals;
        size_t cursor = 0, total = 0, index = 0;
        for (auto& [name, tensor] : tensors) {
            const auto found = entries.find(name);
            if (found == entries.end()) throw Refusal("missing DFlash2 runtime tensor: " + name);
            const auto& entry = *found;
            fields(entry, {"reference_key", "shape", "dtype", "source_offsets", "offset", "nbytes", "sha256"},
                   "invalid DFlash2 runtime tensor fields");
            if (!same_document(entry.at("shape"), Json(tensor.shape)))
                throw Refusal("DFlash2 runtime tensor shape mismatch: " + name);
            const bool codebook = name == "candidate_selector.predecessor_codebook" ||
                                  name == "candidate_selector.successor_codebook";
            if (entry.at("dtype") != "BF16" || entry.at("reference_key") != (codebook ? name + ".weight" : name))
                throw Refusal("DFlash2 runtime tensor dtype/reference key mismatch: " + name);
            const size_t size = tensor_bytes(tensor);
            integer(entry.at("nbytes"), size, size, "invalid DFlash2 runtime tensor bytes");
            tensor.offset_bytes = (cursor + alignment - 1) / alignment * alignment;
            integer(entry.at("offset"), tensor.offset_bytes, tensor.offset_bytes, "invalid DFlash2 runtime tensor offset");
            const auto& offsets = entry.at("source_offsets");
            require(offsets.is_array() && offsets.size() == 2, "invalid DFlash2 source tensor offsets");
            const uint64_t begin = integer(offsets[0], 0, checkpoint_bytes - source_base, "invalid DFlash2 source tensor start");
            const uint64_t end = integer(offsets[1], begin + size, begin + size, "invalid DFlash2 source tensor end");
            source_intervals[index++] = {begin, end};
            digest_field(entry.at("sha256"));
            cursor = tensor.offset_bytes + size;
            total += size;
        }
        std::sort(source_intervals.begin(), source_intervals.end());
        uint64_t source_cursor = 0;
        for (const auto& [begin, end] : source_intervals) {
            require(begin == source_cursor, "DFlash2 source tensor extents overlap or contain gaps");
            source_cursor = end;
        }
        require(source_cursor + source_base == checkpoint_bytes, "DFlash2 source tensor extents do not cover the exact checkpoint");
        require(same_document(manifest.at("roundtrip"), Json{{"method", "byte_equal_every_tensor"},
                             {"tensors", tensor_count}, {"bytes", total}}), "DFlash2 runtime round-trip receipt mismatch");
        const auto& data = manifest.at("data");
        fields(data, {"file", "bytes", "sha256"}, "invalid DFlash2 runtime data fields");
        require(data.at("file") == "dense.bin", "unexpected DFlash2 runtime tensor file");
        integer(data.at("bytes"), cursor, cursor, "invalid DFlash2 runtime data bytes");
        const std::string& data_digest = digest_field(data.at("sha256"));
        require(dense_file.size() == cursor, "DFlash2 runtime dense.bin is truncated or has unclaimed trailing bytes");
        mapping.map(dense_file.fd(), cursor);

        // Hash the same sole read-only map exposed to GPU upload. There is no
        // intermediate tensor allocation, dtype conversion or second file read.
        Sha256 overall;
        cursor = 0;
        for (auto& [name, tensor] : tensors) {
            const uint8_t* padding = mapping.data() + cursor;
            const size_t padding_bytes = tensor.offset_bytes - cursor;
            require(std::all_of(padding, padding + padding_bytes, [](uint8_t byte) { return byte == 0; }),
                    "nonzero DFlash2 runtime alignment padding");
            overall.update(padding, padding_bytes);
            const size_t size = tensor_bytes(tensor);
            const uint8_t* bytes = mapping.data() + tensor.offset_bytes;
            Sha256 tensor_hash;
            constexpr size_t chunk_bytes = 1u << 20;
            for (size_t position = 0; position < size;) {
                const size_t count = std::min(chunk_bytes, size - position);
                tensor_hash.update(bytes + position, count);
                overall.update(bytes + position, count);
                position += count;
            }
            if (sha256_hex(tensor_hash.digest()) != entries.at(name).at("sha256").get_ref<const std::string&>())
                throw Refusal("DFlash2 runtime tensor SHA-256 mismatch: " + name);
            tensor.values = {reinterpret_cast<const uint16_t*>(bytes), size / sizeof(uint16_t)};
            cursor = tensor.offset_bytes + size;
        }
        require(sha256_hex(overall.digest()) == data_digest, "DFlash2 runtime dense.bin SHA-256 mismatch");
        manifest_file.unchanged();
        config_file.unchanged();
        dense_file.unchanged();
        directory.exact_files();
        directory.unchanged();
    }
};

RuntimeWeights::RuntimeWeights(const std::string& directory, const std::string& independent_manifest_sha256,
                               const TargetIdentity& borrowed_target) {
    try {
        impl_ = std::make_unique<Impl>(directory, independent_manifest_sha256, borrowed_target);
    } catch (const Refusal&) {
        throw;
    } catch (const std::exception& error) {
        throw Refusal(std::string("DFlash2 runtime loading failed: ") + error.what());
    }
}
RuntimeWeights::~RuntimeWeights() = default;
const RuntimeTensor& RuntimeWeights::tensor(std::string_view name) const {
    const auto found = impl_->tensors.find(name);
    if (found == impl_->tensors.end()) throw Refusal("unknown DFlash2 runtime tensor: " + std::string(name));
    return found->second;
}
std::span<const uint16_t> RuntimeWeights::dense() const {
    return {reinterpret_cast<const uint16_t*>(impl_->mapping.data()), impl_->mapping.size() / sizeof(uint16_t)};
}
const TargetIdentity& RuntimeWeights::target_identity() const { return impl_->target; }
const std::string& RuntimeWeights::manifest_sha256() const { return impl_->manifest_digest; }
} // namespace strata::core::dflash2
