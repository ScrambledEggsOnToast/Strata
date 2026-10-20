#include "strata/core/dflash2_target.hpp"
#include "conversation_store_sha256.hpp"
#include "strata/artifact/gguf_reader.hpp"

#include <array>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <vector>

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
struct Verification {
    Clock::time_point deadline;
    std::vector<char> buffer = std::vector<char>(4u << 20);
    std::map<std::string, NativeSourceIdentity> checked;
    explicit Verification(Clock::time_point deadline) : deadline(deadline) {}
    void remaining() const { require(Clock::now() < deadline, "daily target artifact verification timed out"); }
    std::string file(const TargetFile& value) {
        remaining(); digest(value.sha256);
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
        require(actual == value.sha256, "daily target artifact SHA-256 mismatch");
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
                require(!name.empty() && std::filesystem::path(name).filename() == name,
                        "native expert source must be a resolved shard filename");
                const auto path = std::filesystem::canonical(std::filesystem::path(*shards.begin()).parent_path() / name);
                require(shards.contains(path.string()), "native expert backing shard is outside verified daily target");
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

VerifiedDailyTarget VerifiedDailyTarget::verify(const DailyTargetFiles& files, const ModelGeometry& g,
                                               const NativeTensorIdentity& embedding, const NativeTensorIdentity& head,
                                               std::chrono::milliseconds timeout) {
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
            pack_roles.emplace("native-shard:" + std::to_string(index), verification.file(files.shards[index]));
            require(shards.insert(std::filesystem::canonical(files.shards[index].path).string()).second,
                    "duplicate daily target shard");
        }
        require(std::filesystem::path(*shards.begin()).parent_path() == std::filesystem::path(*shards.rbegin()).parent_path(),
                "daily native shards must resolve beside one another");
        for (const char* name : pack_names) {
            require(files.pack_sha256.contains(name), "missing daily target pack artifact");
            pack_roles.emplace(std::string("pack:") + name,
                verification.file({(std::filesystem::path(files.pack_path) / name).string(), files.pack_sha256.at(name)}));
        }
        for (const char* name : tokenizer_names) {
            require(files.tokenizer_sha256.contains(name), "missing daily target tokenizer artifact");
            tokenizer_roles.emplace(name, verification.file({(std::filesystem::path(files.tokenizer_path) / name).string(),
                                                             files.tokenizer_sha256.at(name)}));
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
