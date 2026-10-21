#include "strata/core/dflash2_target.hpp"
#include "conversation_store_sha256.hpp"
#include "strata/artifact/gguf_reader.hpp"

#include <array>
#include <filesystem>
#include <fstream>
#include <optional>
#include <set>
#include <sstream>
#include <vector>

#ifndef _WIN32
#include <fcntl.h>
#include <linux/fsverity.h>
#include <sys/ioctl.h>
#include <unistd.h>
#endif

namespace strata::core::dflash2 {
namespace {
using conversation_store_detail::Sha256;
using conversation_store_detail::sha256_hex;
using Clock = std::chrono::steady_clock;
void require(bool ok, const char* message) { if (!ok) throw Refusal(message); }
void digest(const std::string& value) {
    require(value.size() == 64 && value != std::string(64, '0') &&
            std::all_of(value.begin(), value.end(), [](char c) {
                return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
            }), "native target requires nonzero lowercase SHA-256 identities");
}
void geometry(const ModelGeometry& g) {
    constexpr auto expected = kCompiledGeometry;
    constexpr std::array fields{
        &ModelGeometry::n_embd, &ModelGeometry::n_layers, &ModelGeometry::qsa_interval, &ModelGeometry::n_vocab,
        &ModelGeometry::ssm_state_size, &ModelGeometry::ssm_k_heads, &ModelGeometry::ssm_v_heads,
        &ModelGeometry::ssm_d_conv, &ModelGeometry::ssm_conv_channels, &ModelGeometry::ssm_value_dim,
        &ModelGeometry::n_head, &ModelGeometry::n_head_kv, &ModelGeometry::head_dim, &ModelGeometry::n_rot,
        &ModelGeometry::idx_q_heads, &ModelGeometry::idx_key_dim, &ModelGeometry::idx_block, &ModelGeometry::idx_top_k,
        &ModelGeometry::hc, &ModelGeometry::hc_lr, &ModelGeometry::n_expert, &ModelGeometry::n_expert_used,
        &ModelGeometry::n_ff, &ModelGeometry::n_ff_shexp};
    for (auto field : fields) require(g.*field == expected.*field, "daily target geometry mismatch");
    require(g.family == expected.family && g.rms_eps == expected.rms_eps, "daily target family/norm geometry mismatch");
}
void receipt(const NativeTensorIdentity& value, const std::string& expected) {
    digest(value.sha256); digest(expected);
    int block = 0, block_bytes = 0;
    require(value.type >= 0 && strata::block_geometry(uint32_t(value.type), block, block_bytes) && block > 0 &&
            hidden_size % block == 0 &&
            value.bytes == uint64_t(hidden_size / block) * uint64_t(block_bytes) * vocabulary_size,
            "native tensor payload byte geometry mismatch");
    require(value.sha256 == expected && value.input == hidden_size && value.output == vocabulary_size &&
            value.bytes > 0 && value.generation > 0 && value.device >= 0 && value.type >= 0 &&
            !value.source.path.empty(), "unverified/mismatched native tensor identity/geometry");
}
std::optional<std::string> measure_verity(const std::string& path) {
#ifdef _WIN32
    (void) path;
    return std::nullopt;
#else
    // digest_size is an input as well as an output: a zeroed request is EOVERFLOW
    // and every sealed file then reads as "no tree" (#207's review finding).
    // fsverity_digest ends in a flexible array, so the request and its room are one
    // contiguous buffer and digest_size declares that room before the call.
    std::array<uint8_t, sizeof(struct fsverity_digest) + 64> storage{};
    auto* request = reinterpret_cast<struct fsverity_digest*>(storage.data());
    request->digest_size = 64;
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return std::nullopt;
    const int status = ::ioctl(fd, FS_IOC_MEASURE_VERITY, request);
    ::close(fd);
    if (status != 0 || request->digest_size == 0 || request->digest_size > 64) return std::nullopt;
    static constexpr char hex[] = "0123456789abcdef";
    std::string text;
    text.reserve(size_t(request->digest_size) * 2);
    for (uint32_t index = 0; index < request->digest_size; ++index) {
        text.push_back(hex[request->digest[index] >> 4]);
        text.push_back(hex[request->digest[index] & 15]);
    }
    return text;
#endif
}
const VerityPin* pin_for(const std::map<std::string, VerityPin>& roots, const std::string& path) {
    if (roots.empty()) return nullptr;
    std::error_code ec;
    const auto canonical = std::filesystem::canonical(path, ec);
    const auto at = roots.find(ec ? path : canonical.string());
    return at == roots.end() ? nullptr : &at->second;
}
struct Verification {
    Clock::time_point deadline;
    std::vector<char> buffer = std::vector<char>(4u << 20);
    std::map<std::string, NativeSourceIdentity> checked;
    explicit Verification(Clock::time_point deadline) : deadline(deadline) {}
    void remaining() const { require(Clock::now() < deadline, "daily target artifact verification timed out"); }
    /// `pin`: the release's content digest and fs-verity root for this path. A sealed
    /// file whose measured root matches is the pinned bytes without being read: the
    /// kernel verifies every page the engine later reads, so a mismatching page fails
    /// its read instead of entering a cache. Unsealed files and paths without a pin
    /// take the content read unchanged.
    std::string file(const TargetFile& value, bool check_digest, const VerityPin* pin = nullptr) {
        remaining();
        if (check_digest) digest(value.sha256);
        if (pin != nullptr) {
            const auto before = native_source_identity(value.path);
            const auto measured = measure_verity(before.path);
            if (measured.has_value()) {
                // Sealed: the kernel owns the bytes. A different root is a different
                // artifact; the same root is the pinned content without reading it.
                require(*measured == pin->root,
                        ("daily target artifact is not the release's sealed bytes: " + before.path).c_str());
                const auto after = native_source_identity(value.path);
                require(before == after, "daily target artifact changed during verification");
                digest(pin->sha256);
                require(!check_digest || pin->sha256 == value.sha256,
                        "pinned daily target digest contradicts the expected artifact");
                const auto [at, inserted] = checked.emplace(before.path, before);
                require(inserted || at->second == before, "daily target artifact was replaced");
                return pin->sha256;
            }
            // Unsealed (a copy, or a tree the release never sealed): take the content
            // read below, exactly as an artifact with no pin does.
        }
        const auto before = native_source_identity(value.path);
        std::ifstream input(before.path, std::ios::binary);
        require(bool(input), "daily target artifact is unreadable");
        Sha256 hash;
        uint64_t count = 0;
        while (input) {
            remaining();
            input.read(buffer.data(), std::streamsize(buffer.size()));
            const auto n = input.gcount();
            hash.update(buffer.data(), size_t(n)); count += uint64_t(n);
        }
        require(input.eof() && !input.bad(), "daily target artifact read failed");
        const auto after = native_source_identity(value.path);
        require(before == after && count == before.stamp[2], "daily target artifact changed during verification");
        const std::string actual = sha256_hex(hash.digest());
        require(!check_digest || actual == value.sha256, "daily target artifact SHA-256 mismatch");
        const auto [at, inserted] = checked.emplace(before.path, before);
        require(inserted || at->second == before, "daily target artifact was replaced");
        return actual;
    }
    void unchanged() const {
        for (const auto& [path, identity] : checked) {
            remaining();
            require(native_source_identity(path) == identity, "daily target artifact changed before binding");
        }
    }
};
std::string group(const char* domain, const std::map<std::string, std::string>& roles) {
    Sha256 hash;
    hash.update(domain, std::char_traits<char>::length(domain) + 1);
    for (const auto& [role, value] : roles) {
        hash.update(role.data(), role.size() + 1);
        hash.update(value.data(), value.size());
    }
    return sha256_hex(hash.digest());
}
void native_sources(const std::filesystem::path& pack, const std::set<std::string>& shards) {
    require(!std::filesystem::exists(pack / "experts.bin"), "packed experts override is not the daily native closure");
    std::ifstream input(pack / "native_experts.txt");
    require(bool(input), "native expert index is unreadable");
    std::set<int> layers;
    for (std::array<char, 8192> line{}; input.getline(line.data(), std::streamsize(line.size()));) {
        std::istringstream row(line.data());
        row >> std::ws;
        if (row.eof() || row.peek() == '#') continue;
        int layer = -1;
        uint64_t values[7];
        require(bool(row >> layer) && layer >= 0 && layer < 48 && layers.insert(layer).second,
                "native expert index has invalid/duplicate layers");
        for (auto& value : values) require(bool(row >> value), "native expert index format mismatch");
        std::string names, extra;
        if (row >> names) {
            require(!(row >> extra), "native expert index has unexpected columns");
            size_t begin = 0, roles = 0;
            for (;;) {
                const auto end = names.find(',', begin);
                const auto name = names.substr(begin, end - begin);
                // Empty v4 roles use the selected --native shard, already verified above.
                if (!name.empty()) {
                    require(std::filesystem::path(name).filename() == name,
                            "native expert source must be a resolved shard filename");
                    const auto path = std::filesystem::canonical(std::filesystem::path(*shards.begin()).parent_path() / name);
                    require(shards.contains(path.string()), "native expert backing shard is outside verified daily target");
                }
                ++roles;
                if (end == std::string::npos) break;
                begin = end + 1;
            }
            require(roles == 1 || roles == 3, "native expert source role count mismatch");
        }
    }
    require(input.eof() && layers.size() == 48, "native expert index does not cover the complete daily target");
}
} // namespace

bool load_verity_roots(const std::string& path, std::map<std::string, VerityPin>& out, std::string& error) {
    try {
        std::ifstream input(path);
        if (!input) { error = "verity roots: unreadable file " + path; return false; }
        std::map<std::string, VerityPin> parsed;
        size_t line_number = 0;
        for (std::string line; std::getline(input, line);) {
            ++line_number;
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty() || line[0] == '#') continue;
            const auto first = line.find('\t');
            const auto second = first == std::string::npos ? std::string::npos : line.find('\t', first + 1);
            require(first != std::string::npos && second != std::string::npos,
                    ("verity roots: malformed line " + std::to_string(line_number)).c_str());
            VerityPin pin{line.substr(0, first), line.substr(first + 1, second - first - 1)};
            const std::string file = line.substr(second + 1);
            digest(pin.sha256);
            digest(pin.root);
            require(!file.empty() && file.front() == '/', "verity roots: path must be absolute");
            std::error_code ec;
            const auto canonical = std::filesystem::canonical(file, ec);
            require(!ec, ("verity roots: unresolvable path " + file).c_str());
            require(parsed.emplace(canonical.string(), std::move(pin)).second,
                    ("verity roots: duplicate path " + file).c_str());
        }
        require(!input.bad(), "verity roots: read failed");
        out = std::move(parsed);
        return true;
    } catch (const Refusal& refusal) {
        error = refusal.what();
        return false;
    }
}

VerifiedDailyTarget VerifiedDailyTarget::verify(const DailyTargetFiles& files, const ModelGeometry& g,
                                               const NativeTensorIdentity& embedding, const NativeTensorIdentity& head,
                                               std::chrono::milliseconds timeout) {
    return verify_impl(files, g, embedding, head, timeout, true, {});
}
VerifiedDailyTarget VerifiedDailyTarget::identify(const std::string& pack, const std::string& tokenizer,
                                                 const std::array<std::string, 2>& shards, const ModelGeometry& g,
                                                 const NativeTensorIdentity& embedding, const NativeTensorIdentity& head,
                                                 std::chrono::milliseconds timeout,
                                                 const std::map<std::string, VerityPin>& verity_roots) {
    DailyTargetFiles files;
    files.profile_name = daily_profile; files.pack_path = pack; files.tokenizer_path = tokenizer;
    for (size_t i = 0; i < shards.size(); ++i) files.shards[i].path = shards[i];
    for (const char* name : {"index.txt", "conversions.json", "native_experts.txt", "dense.bin"})
        files.pack_sha256.emplace(name, "");
    for (const char* name : {"chat_template.jinja", "vocab.json", "merges.txt", "token_type.json", "tokenizer.json"})
        files.tokenizer_sha256.emplace(name, "");
    files.embedding_sha256 = embedding.sha256; files.head_sha256 = head.sha256;
    return verify_impl(files, g, embedding, head, timeout, false, verity_roots);
}
VerifiedDailyTarget VerifiedDailyTarget::verify_impl(const DailyTargetFiles& files, const ModelGeometry& g,
                                                    const NativeTensorIdentity& embedding, const NativeTensorIdentity& head,
                                                    std::chrono::milliseconds timeout, bool check_digests,
                                                    const std::map<std::string, VerityPin>& verity_roots) {
    try {
        require(files.profile_name == daily_profile && timeout.count() > 0 && timeout <= std::chrono::seconds(1800),
                "unselected daily target profile or invalid verification deadline");
        geometry(g); receipt(embedding, files.embedding_sha256); receipt(head, files.head_sha256);
        constexpr std::array pack_names{"index.txt", "conversions.json", "native_experts.txt", "dense.bin"};
        constexpr std::array tokenizer_names{"chat_template.jinja", "vocab.json", "merges.txt", "token_type.json", "tokenizer.json"};
        require(files.pack_sha256.size() == pack_names.size() && files.tokenizer_sha256.size() == tokenizer_names.size(),
                "daily target artifact closure is incomplete/unexpected");
        Verification verification{Clock::now() + timeout};
        std::map<std::string, std::string> pack_roles, tokenizer_roles;
        std::set<std::string> shards;
        for (size_t index = 0; index < files.shards.size(); ++index) {
            pack_roles.emplace("native-shard:" + std::to_string(index), verification.file(files.shards[index], check_digests,
                                      pin_for(verity_roots, files.shards[index].path)));
            require(shards.insert(std::filesystem::canonical(files.shards[index].path).string()).second,
                    "duplicate daily target shard");
        }
        require(std::filesystem::path(*shards.begin()).parent_path() == std::filesystem::path(*shards.rbegin()).parent_path(),
                "daily native shards must resolve beside one another");
        for (const char* name : pack_names) {
            require(files.pack_sha256.contains(name), "missing daily target pack artifact");
            pack_roles.emplace(std::string("pack:") + name,
                verification.file({(std::filesystem::path(files.pack_path) / name).string(), files.pack_sha256.at(name)},
                                  check_digests, pin_for(verity_roots, (std::filesystem::path(files.pack_path) / name).string())));
        }
        for (const char* name : tokenizer_names) {
            require(files.tokenizer_sha256.contains(name), "missing daily target tokenizer artifact");
            const auto path = (std::filesystem::path(files.tokenizer_path) / name).string();
            tokenizer_roles.emplace(name, verification.file({path, files.tokenizer_sha256.at(name)},
                                                           check_digests, pin_for(verity_roots, path)));
        }
        native_sources(files.pack_path, shards);
        for (const auto* tensor : {&embedding, &head}) {
            const auto source = verification.checked.find(tensor->source.path);
            require(shards.contains(tensor->source.path) && source != verification.checked.end() &&
                    source->second == tensor->source, "native receipt does not name the verified loaded shard bytes");
        }
        verification.unchanged();
        TargetIdentity identity{group("strata-dflash2-daily-pack-v1", pack_roles), embedding.sha256, head.sha256,
                                group("strata-dflash2-daily-tokenizer-v1", tokenizer_roles)};
        return VerifiedDailyTarget({std::move(identity), hidden_size, vocabulary_size, head.generation}, embedding, head);
    } catch (const Refusal&) { throw; }
      catch (const std::exception& error) { throw Refusal(std::string("daily target verification: ") + error.what()); }
}
} // namespace strata::core::dflash2
