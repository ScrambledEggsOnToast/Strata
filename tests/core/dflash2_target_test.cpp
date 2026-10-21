// CPU-only synthetic native weights/device sinks. No target/checkpoint load or
// GPU qualification: exercise the SAME adapter, upload hashing and file verifier.
#include "strata/core/dflash2_target.hpp"
#include "conversation_store_sha256.hpp"
#include "native_verified_upload.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>
#include <sys/socket.h>
#include <unistd.h>

namespace d = strata::core::dflash2;
namespace c = strata::core;
namespace fs = std::filesystem;
using c::conversation_store_detail::sha256;
using c::conversation_store_detail::sha256_hex;
namespace {
void check(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
template<class F> void refuses(F&& action) {
    try { action(); } catch (const d::Refusal&) { return; }
    throw std::runtime_error("expected DFlash2 refusal");
}
std::string hash(const std::string& value) { return sha256_hex(sha256(value.data(), value.size())); }
void file(const fs::path& path, const std::string& value) {
    std::ofstream out(path, std::ios::binary); out << value;
    check(bool(out), "fixture file write failed");
}
struct Files {
    fs::path root = fs::temp_directory_path() /
        ("dflash2-target-" + std::to_string(getpid()) + "-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    d::DailyTargetFiles files;
    c::NativeTensorIdentity embedding, head;
    explicit Files(const std::string& expert_roles = "shard0.gguf,shard1.gguf,shard0.gguf") {
        fs::create_directories(root / "pack"); fs::create_directories(root / "tokenizer");
        files.profile_name = d::daily_profile;
        files.pack_path = (root / "pack").string(); files.tokenizer_path = (root / "tokenizer").string();
        for (size_t index = 0; index < 2; ++index) {
            const auto path = root / ("shard" + std::to_string(index) + ".gguf");
            const auto data = "SYNTHETIC native source " + std::to_string(index);
            file(path, data); files.shards[index] = {path.string(), hash(data)};
        }
        std::string expert_index;
        for (int layer = 0; layer < 48; ++layer)
            expert_index += std::to_string(layer) + " 8 8 0 1 0 0 0 " + expert_roles + "\n";
        for (const char* name : {"index.txt", "conversions.json", "native_experts.txt", "dense.bin"}) {
            const std::string data = std::string(name) == "native_experts.txt" ? expert_index : "fixture " + std::string(name);
            file(root / "pack" / name, data); files.pack_sha256[name] = hash(data);
        }
        for (const char* name : {"chat_template.jinja", "vocab.json", "merges.txt", "token_type.json", "tokenizer.json"}) {
            const std::string data = "fixture " + std::string(name);
            file(root / "tokenizer" / name, data); files.tokenizer_sha256[name] = hash(data);
        }
        // Deliberately synthetic loader receipts: the fixture does NOT pretend
        // these tiny files contain the real daily weight matrices.
        embedding = {hash("embedding bytes"), c::native_source_identity(files.shards[0].path),
                     uint64_t(d::hidden_size / 32 * 34) * d::vocabulary_size, 11,
                     d::hidden_size, d::vocabulary_size, 8, 0};
        head = {hash("head bytes"), c::native_source_identity(files.shards[1].path),
                uint64_t(d::hidden_size / 32 * 34) * d::vocabulary_size, 12,
                d::hidden_size, d::vocabulary_size, 8, 2};
        files.embedding_sha256 = embedding.sha256; files.head_sha256 = head.sha256;
    }
    ~Files() { std::error_code ignored; fs::remove_all(root, ignored); }
    d::VerifiedDailyTarget verify() const { return d::VerifiedDailyTarget::verify(files, {}, embedding, head); }
};
struct Io {
    inline static int constructions = 0, drains = 0;
    inline static bool finish_failure = false;
    std::array<int32_t, 4> ids;
    std::array<float, 4 * d::hidden_size> emb;
    std::array<float, d::hidden_size> hidden;
    std::vector<float> output = std::vector<float>(d::vocabulary_size);
    std::array<std::byte, d::hidden_size / 32 * 36> scratch;
    int state = 0;
    Io(int embedding_device, int head_device, size_t bytes) {
        check(embedding_device == 0 && head_device == 2 && bytes == scratch.size(), "wrong dedicated I/O placement");
        ++constructions;
    }
    void begin_embeddings(std::span<const int32_t> input) {
        check(state == 0, "overlapping embedding I/O"); state = 1;
        std::copy(input.begin(), input.end(), ids.begin());
    }
    void finish_embeddings(std::span<float> out) {
        check(state == 1, "embedding completion missing");
        if (finish_failure) throw d::Refusal("synthetic embedding synchronization failure");
        std::copy(emb.begin(), emb.end(), out.begin()); state = 0;
    }
    void begin_head(std::span<const float> input) {
        check(state == 0, "overlapping projection I/O"); state = 2;
        std::copy(input.begin(), input.end(), hidden.begin());
    }
    void finish_head(std::span<float> out) {
        check(state == 2, "head completion missing");
        if (finish_failure) throw d::Refusal("synthetic head synchronization failure");
        std::copy(output.begin(), output.end(), out.begin()); state = 0;
    }
    const int32_t* tokens() const { return ids.data(); }
    float* embedding_rows() { return emb.data(); }
    float* hidden_row() { return hidden.data(); }
    float* logits() { return output.data(); }
    void* embedding_stream() { return this; }
    void* head_stream() { return this; }
    void* head_scratch() { return scratch.data(); }
    size_t head_scratch_bytes() const { return scratch.size(); }
    void drain() noexcept { state = 0; ++drains; }
};
struct Embed {
    c::NativeTensorIdentity info;
    bool live = true;
    mutable int calls = 0;
    const c::NativeTensorIdentity& identity() const { return info; }
    bool loaded() const { return live; }
    void gather_dev(const int32_t* ids, int64_t rows, float* out, void* stream) const {
        check(stream != nullptr && rows == 4, "native gather shape/stream"); ++calls;
        for (int row = 0; row < rows; ++row) for (int column = 0; column < d::hidden_size; ++column)
            out[size_t(row) * d::hidden_size + column] = float(ids[row] % 128) * .125f + float(column % 7) * .00001f;
    }
};
struct Head {
    c::NativeTensorIdentity info;
    bool live = true, failure = false, throws = false;
    mutable int calls = 0;
    mutable std::array<float, d::hidden_size> seen{};
    mutable const void* used_scratch = nullptr;
    const c::NativeTensorIdentity& identity() const { return info; }
    bool loaded() const { return live; }
    size_t projection_scratch_bytes() const { return d::hidden_size / 32 * 36; }
    bool run_with_scratch(const float* in, float* out, void* scratch, size_t bytes, void* stream, std::string& err) const {
        check(scratch != nullptr && bytes == projection_scratch_bytes() && stream != nullptr, "projection ownership");
        auto& io = *static_cast<Io*>(stream);
        check(scratch == io.scratch.data() && in == io.hidden.data() && out == io.output.data(), "private scratch/I/O not used");
        used_scratch = scratch; ++calls; std::copy_n(in, seen.size(), seen.begin());
        if (throws) throw std::runtime_error("synthetic projection exception");
        if (failure) { err = "synthetic projection failure"; return false; }
        for (int token = 0; token < d::vocabulary_size; ++token) out[token] = in[0] + in[1] * float(token % 17);
        return true;
    }
};
using Adapter = d::NativeTargetAdapter<Embed, Head, Io>;
d::Operation op(int generation = 1) { return {"request-A", 7, generation, 37}; }
constexpr std::array<int32_t, 4> ids{17, d::mask_token, d::mask_token, d::mask_token};
std::array<float, d::hidden_size> hidden_row(int row) {
    std::array<float, d::hidden_size> result;
    for (size_t column = 0; column < result.size(); ++column) result[column] = float(int(column % 11) - 5) * .25f * float(row + 1);
    return result;
}
void upload_tests() {
    std::vector<std::byte> source(c::native_identity_staging_bytes + 17), uploaded(source.size());
    for (size_t i = 0; i < source.size(); ++i) source[i] = std::byte(i % 251);
    const auto expected = sha256_hex(sha256(source.data(), source.size()));
    std::string digest;
    size_t chunks = 0;
    check(c::native_verified_upload(source.data(), source.size(), digest,
        [&](size_t offset, std::span<const std::byte> chunk) {
            check(chunk.size() <= c::native_identity_staging_bytes && chunk.data() != source.data() + offset,
                  "verified upload did not own bounded immutable staging");
            // Mutating the original source cannot change the bytes uploaded or
            // the digest of this already staged chunk.
            std::fill(source.begin() + offset, source.begin() + offset + chunk.size(), std::byte{0});
            std::copy(chunk.begin(), chunk.end(), uploaded.begin() + offset); ++chunks; return true;
        }), "verified upload failed");
    check(chunks == 2 && digest == expected && digest == sha256_hex(sha256(uploaded.data(), uploaded.size())),
          "receipt differs from exact uploaded bytes");
    check(!c::native_verified_upload(source.data(), source.size(), digest,
                                    [](size_t, std::span<const std::byte>) { return false; }) && digest.empty(),
          "failed upload published a digest");
    check(!c::native_verified_upload(nullptr, 0, digest, [](size_t, std::span<const std::byte>) { return true; }),
          "empty/null verified upload accepted");
}
void artifact_tests() {
    Files fixture;
    const auto verified = fixture.verify();
    check(verified.binding().hidden == d::hidden_size && verified.binding().vocabulary == d::vocabulary_size &&
          verified.binding().load_generation == fixture.head.generation &&
          verified.binding().identity.embedding_sha256 == fixture.embedding.sha256 &&
          verified.binding().identity.head_sha256 == fixture.head.sha256,
          "verified target binding identity/geometry");
    {
        Files split(",shard1.gguf,");
        const auto split_target = split.verify();
        const auto& binding = split_target.binding();
        check(binding.identity.embedding_sha256 == split.embedding.sha256 &&
              binding.identity.head_sha256 == split.head.sha256,
              "empty v4 expert roles retain the verified primary shard across a split layer");
    }
    for (const std::string roles : {",shard1.gguf", ",shard1.gguf,,", "../shard0.gguf,,",
                                    ",foreign.gguf,"}) {
        Files changed(roles);
        file(changed.root / "foreign.gguf", "unverified expert source");
        refuses([&] { changed.verify(); });
    }
    for (int scenario = 0; scenario < 18; ++scenario) {
        Files changed;
        c::ModelGeometry geometry;
        if (scenario == 0) changed.files.profile_name = "original-Qwen";
        if (scenario == 1) changed.embedding.sha256.clear();
        if (scenario == 2) changed.head.generation = 0;
        if (scenario == 3) changed.head.input--;
        if (scenario == 4) changed.embedding.output--;
        if (scenario == 5) geometry.hc--;
        if (scenario == 6) geometry.n_expert--;
        if (scenario == 7) geometry.n_head_kv++;
        if (scenario == 8) changed.files.pack_sha256.erase("dense.bin");
        if (scenario == 9) changed.files.tokenizer_sha256.erase("chat_template.jinja");
        if (scenario == 10) changed.files.head_sha256 = hash("another head");
        if (scenario == 11) file(changed.root / "pack" / "dense.bin", "mutated pack");
        if (scenario == 12) file(changed.root / "tokenizer" / "vocab.json", "mutated tokenizer");
        if (scenario == 13) changed.files.shards[1].sha256 = hash("different daily shards");
        if (scenario == 14) file(changed.root / "pack" / "experts.bin", "unrecorded override");
        if (scenario == 15) changed.head.device = -1;
        if (scenario == 16) changed.head.bytes--;
        if (scenario == 17) changed.embedding.type = 999;
        refuses([&] { d::VerifiedDailyTarget::verify(changed.files, geometry, changed.embedding, changed.head); });
    }
    {
        Files changed;
        // Even a newly matching full-file hash cannot launder an OLD loaded
        // receipt after its source changed or was replaced.
        file(changed.files.shards[0].path, "new source bytes");
        changed.files.shards[0].sha256 = hash("new source bytes");
        refuses([&] { changed.verify(); });
    }
    for (const std::string index : {"0 8 8 0 1 0 0 0\n", "0 8 8 0 1 0 0 0 unknown.gguf\n",
                                    "0 8 8 0 1 0 0 0\n0 8 8 0 1 0 0 0\n"}) {
        Files changed; file(changed.root / "pack" / "native_experts.txt", index);
        changed.files.pack_sha256["native_experts.txt"] = hash(index);
        refuses([&] { changed.verify(); });
    }
    refuses([&] { d::VerifiedDailyTarget::verify(fixture.files, {}, fixture.embedding, fixture.head, std::chrono::milliseconds(0)); });
    check(c::next_native_weight_generation() != c::next_native_weight_generation(), "reload generation repeated");
}
void adapter_tests() {
    Files fixture;
    const auto verified = fixture.verify();
    Embed embed{fixture.embedding}; Head head{fixture.head};
    Adapter target(verified, embed, head);
    std::array<float, 4 * d::hidden_size> raw;
    std::vector<float> logits(d::vocabulary_size);
    for (int generation = 1; generation <= 2; ++generation) {
        c::OperationLease::Guard lease(target.scratch_lease(), std::try_to_lock);
        check(bool(lease), "could not acquire real private projection lease");
        target.raw_embeddings(op(generation), ids, raw);
        check(raw[1] == 17 * .125f + .00001f, "raw embeddings were BF16 rounded or normalized twice");
        for (int row = 0; row < 3; ++row) {
            const auto hidden = hidden_row(row);
            target.native_head(op(generation), hidden, logits);
            check(head.seen == hidden, "draft hidden was mixed/normalized/converted before native projection");
            check(logits.front() == hidden[0] && logits.back() == hidden[0] + hidden[1] * float((d::vocabulary_size - 1) % 17),
                  "full native vocabulary was not returned host-complete");
        }
        lease.complete();
    }
    check(embed.calls == 2 && head.calls == 6, "wrong native invocation count");
    const void* scratch = head.used_scratch;
    Adapter second(verified, embed, head);
    second.raw_embeddings(op(3), ids, raw); second.native_head(op(3), hidden_row(0), logits);
    check(scratch != head.used_scratch, "independent adapters shared activation scratch");
    second.native_head(op(3), hidden_row(1), logits); second.native_head(op(3), hidden_row(2), logits);
    // Constructor fencing happens before allocating any I/O.
    const int allocations = Io::constructions;
    ++head.info.generation;
    refuses([&] { Adapter stale(verified, embed, head); });
    check(Io::constructions == allocations, "stale receipt allocated before refusal");
    refuses([&] { target.binding(); });
}
void adapter_refusals() {
    for (int scenario = 0; scenario < 15; ++scenario) {
        Files fixture; Embed embed{fixture.embedding}; Head head{fixture.head};
        Adapter target(fixture.verify(), embed, head);
        std::array<float, 4 * d::hidden_size> raw;
        std::vector<float> logits(d::vocabulary_size);
        const int drains = Io::drains;
        {
            c::OperationLease::Guard lease(target.scratch_lease());
            check(bool(lease), "fixture lease acquisition failed");
            refuses([&] {
                auto operation = op(); auto tokens = ids;
                if (scenario == 0) { target.native_head(operation, hidden_row(0), logits); return; }
                if (scenario == 1) tokens[1] = 4;
                if (scenario == 2) tokens[0] = d::vocabulary_size;
                if (scenario == 3) operation.generation = 0;
                if (scenario == 4) Io::finish_failure = true;
                target.raw_embeddings(operation, tokens, raw);
                if (scenario == 5) { target.raw_embeddings(operation, tokens, raw); return; }
                if (scenario == 6) ++head.info.generation;
                if (scenario == 7) ++embed.info.generation;
                if (scenario == 8) head.live = false;
                if (scenario == 9) head.failure = true;
                if (scenario == 10) head.throws = true;
                if (scenario == 11) ++operation.incarnation;
                auto hidden = hidden_row(0);
                if (scenario == 12) hidden[0] = .1234567f;
                if (scenario == 13) hidden[0] = std::numeric_limits<float>::infinity();
                if (scenario == 14) Io::finish_failure = true;
                try { target.native_head(operation, hidden, logits); }
                catch (const std::runtime_error&) { throw d::Refusal("native adapter surfaced fixture failure"); }
            });
            // Deliberately no complete(): failed projection ownership is poisoned.
        }
        Io::finish_failure = false;
        check(Io::drains > drains, "failed asynchronous consumers were not drained");
        refuses([&] { target.binding(); });
        c::OperationLease::Guard poisoned(target.scratch_lease(), std::try_to_lock);
        check(!poisoned && poisoned.failure() == c::OperationLease::Failure::poisoned,
              "failure repaired native projection scratch");
    }
}
void bridge_binding_tests() {
    Files fixture; Embed embed{fixture.embedding}; Head head{fixture.head};
    Adapter target(fixture.verify(), embed, head);
    int sockets[2]; check(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0, "socketpair failed");
    d::CaptureStore store("request-A", 7, 37);
    {
        d::Bridge bridge(store, target, target.binding().identity, sockets[0]);
        // Native adapter satisfies the actual Bridge interface; no listener or
        // real draft forward is fabricated by this constructor-only fixture.
        check(!bridge.poisoned(), "native adapter binding refused by actual bridge");
    }
    close(sockets[1]);
    check(d::NativeTargetDeviceIo::embedding_device_bytes() == 40976 &&
          d::NativeTargetDeviceIo::head_device_bytes(d::hidden_size / 32 * 36) == 1006400,
          "adapter device storage budget changed");
}
} // namespace
int main() {
    try {
        upload_tests(); artifact_tests(); adapter_tests(); adapter_refusals(); bridge_binding_tests();
        std::cout << "dflash2_target_test: all passed (synthetic CPU contracts; no GPU qualification)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "dflash2_target_test: " << error.what() << '\n'; return 1;
    }
}
