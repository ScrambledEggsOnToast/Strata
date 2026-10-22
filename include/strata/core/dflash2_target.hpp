#pragma once

#include "strata/core/dflash2_bridge.hpp"
#include "strata/core/geometry.hpp"
#include "strata/core/native_head.hpp"
#include "strata/core/native_weight_identity.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <map>
#include <utility>

namespace strata::core::dflash2 {
inline constexpr char daily_profile[] = "swift-1.5-flash-next-256k-kv-concurrency-daily";
struct TargetFile {
    std::string path, sha256;
};
/// One release artifact's content digest and its fs-verity root. A sealed file
/// whose measured root equals this root is the pinned bytes without being read:
/// enabling verity is one-way and the kernel verifies every page the engine
/// then reads. Keyed by absolute path in `identify`'s optional roots argument.
struct VerityPin {
    std::string sha256, root;
};
/// Reads `<sha256>\t<verity-root>\t<absolute-path>` lines, `#` comments allowed.
/// A malformed line refuses the whole file rather than dropping a pin.
bool load_verity_roots(const std::string& path, std::map<std::string, VerityPin>& out, std::string& error);
struct DailyTargetFiles {
    std::string profile_name;
    // Exactly the two resolved native shards, in load order. Expected digests
    // come from the pinned daily release, never shipped SHA256SUMS alone.
    std::array<TargetFile, 2> shards;
    std::string pack_path, tokenizer_path;
    std::map<std::string, std::string> pack_sha256, tokenizer_sha256;
    std::string embedding_sha256, head_sha256;
};

// Constructible only by verification of actual file bytes plus native receipts.
// The loaded geometry and receipts must come from the very target being borrowed.
// No global service selection, model load, download or device work is performed.
class VerifiedDailyTarget {
public:
    static VerifiedDailyTarget verify(const DailyTargetFiles&, const ModelGeometry&,
                                      const NativeTensorIdentity& embedding, const NativeTensorIdentity& head,
                                      std::chrono::milliseconds timeout = std::chrono::seconds(300));
    // Identify the actual selected native closure once, before comparing it with
    // an independently pinned drafter manifest. No declared digest substitutes
    // for the bytes or loaded embedding/head receipts.
    static VerifiedDailyTarget identify(const std::string& pack, const std::string& tokenizer,
                                        const std::array<std::string, 2>& shards, const ModelGeometry&,
                                        const NativeTensorIdentity& embedding, const NativeTensorIdentity& head,
                                        std::chrono::milliseconds timeout = std::chrono::seconds(300),
                                        const std::map<std::string, VerityPin>& verity_roots = {});
    const BorrowedBinding& binding() const { return binding_; }
    const NativeTensorIdentity& embedding() const { return embedding_; }
    const NativeTensorIdentity& head() const { return head_; }
private:
    static VerifiedDailyTarget verify_impl(const DailyTargetFiles&, const ModelGeometry&,
                                           const NativeTensorIdentity&, const NativeTensorIdentity&,
                                           std::chrono::milliseconds, bool check_digests,
                                           const std::map<std::string, VerityPin>& verity_roots);
    VerifiedDailyTarget(BorrowedBinding binding, NativeTensorIdentity embedding, NativeTensorIdentity head)
        : binding_(std::move(binding)), embedding_(std::move(embedding)), head_(std::move(head)) {}
    BorrowedBinding binding_;
    NativeTensorIdentity embedding_, head_;
};

// Actual device implementation is linked only with the opt-in GPU engine build.
// CPU tests substitute this device seam, not the adapter's contract/ordering.
class NativeTargetDeviceIo {
public:
    NativeTargetDeviceIo(int embedding_device, int head_device, size_t head_scratch_bytes);
    ~NativeTargetDeviceIo();
    NativeTargetDeviceIo(const NativeTargetDeviceIo&) = delete;
    NativeTargetDeviceIo& operator=(const NativeTargetDeviceIo&) = delete;
    static constexpr size_t embedding_device_bytes() { return reference_block_size * (sizeof(int32_t) + hidden_size * sizeof(float)); }
    static constexpr size_t head_device_bytes(size_t scratch) { return (hidden_size + vocabulary_size) * sizeof(float) + scratch; }
    void begin_embeddings(std::span<const int32_t>);
    void finish_embeddings(std::span<float>);
    void begin_head(std::span<const float>);
    void finish_head(std::span<float>);
    const int32_t* tokens() const;
    float* embedding_rows() const;
    float* hidden_row() const;
    float* logits() const;
    void* embedding_stream() const;
    void* head_stream() const;
    void* head_scratch() const;
    size_t head_scratch_bytes() const;
    void drain() noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// A compile-time device seam keeps the same adapter executable in CPU fixtures.
// The native objects and their immutable allocations must outlive this adapter;
// all methods, unload and reload have one serialized host owner. Reload fences
// even identical bytes. No captured target graph is changed or replayed here.
// Device buffers/streams are allocated ONCE, after receipt validation, and must
// be priced before target residency admission. The lease owns the actual private
// projection scratch; NativeHead::run and its internal scratch are NEVER used.
template<class Embed = NativeEmbed, class Head = NativeHead, class Io = NativeTargetDeviceIo>
class NativeTargetAdapter final : public BorrowedTarget {
public:
    NativeTargetAdapter(VerifiedDailyTarget verified, const Embed& embedding, const Head& head)
        : verified_(checked(std::move(verified), embedding, head)), embedding_(embedding), head_(head),
          io_(verified_.embedding().device, verified_.head().device, head.projection_scratch_bytes()) {}
    const BorrowedBinding& binding() const override {
        check_loaded();
        return verified_.binding();
    }
    OperationLease& scratch_lease() override { return lease_; }
    void raw_embeddings(const Operation& op, std::span<const int32_t> ids, std::span<float> out) override {
        try {
            check_loaded();
            require(!active_ && valid_operation(op) &&
                    (op.request_id != last_operation_.request_id || op.incarnation != last_operation_.incarnation ||
                     (op.generation > last_operation_.generation && op.frontier >= last_operation_.frontier)),
                    "stale/outstanding native target operation");
            require(ids.size() == reference_block_size && out.size() == reference_block_size * hidden_size &&
                    ids[0] >= 0 && ids[0] < vocabulary_size &&
                    std::all_of(ids.begin() + 1, ids.end(), [](int32_t token) { return token == mask_token; }),
                    "native embedding requires anchor/mask/mask/mask and four raw rows");
            operation_ = op; active_ = true; head_rows_ = 0;
            io_.begin_embeddings(ids);
            embedding_.gather_dev(io_.tokens(), reference_block_size, io_.embedding_rows(), io_.embedding_stream());
            io_.finish_embeddings(out);
            check_loaded();
        } catch (...) { fail(); throw; }
    }
    void native_head(const Operation& op, std::span<const float> hidden, std::span<float> out) override {
        try {
            check_loaded();
            require(active_ && op == operation_ && head_rows_ < reference_block_size - 1 &&
                    hidden.size() == hidden_size && out.size() == vocabulary_size,
                    "native head requires three serial owned hidden rows and full vocabulary");
            // This is a widening check, NOT a norm: the upstream draft has already
            // applied its own final norm. No target mixer/norm or +1 conversion.
            for (float value : hidden)
                require(std::isfinite(value) && (std::bit_cast<uint32_t>(value) & 0xffffu) == 0,
                        "native head input is not widened finite BF16");
            io_.begin_head(hidden);
            error_.clear();
            if (!head_.run_with_scratch(io_.hidden_row(), io_.logits(), io_.head_scratch(),
                                        io_.head_scratch_bytes(), io_.head_stream(), error_))
                throw Refusal("borrowed native projection failed: " + error_);
            io_.finish_head(out);
            check_loaded();
            if (++head_rows_ == reference_block_size - 1) {
                active_ = false; last_operation_ = operation_;
            }
        } catch (...) { fail(); throw; }
    }
private:
    static void require(bool ok, const char* message) { if (!ok) throw Refusal(message); }
    static bool valid_operation(const Operation& op) {
        return !op.request_id.empty() && op.request_id.size() <= 128 && op.incarnation >= 0 &&
            op.generation > 0 && op.frontier >= 0 && op.frontier <= max_position - reference_block_size &&
            std::all_of(op.request_id.begin(), op.request_id.end(), [](unsigned char c) {
                return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                       (c >= '0' && c <= '9') || c == '_' || c == '.' || c == ':' || c == '-';
            });
    }
    static VerifiedDailyTarget checked(VerifiedDailyTarget value, const Embed& embedding, const Head& head) {
        require(embedding.loaded() && head.loaded() && embedding.identity() == value.embedding() &&
                head.identity() == value.head() && head.projection_scratch_bytes() == size_t(hidden_size / 32 * 36),
                "unverified/reloaded native weights or unexpected projection scratch");
        return value;
    }
    void check_loaded() const {
        require(!poisoned_ && embedding_.loaded() && head_.loaded() &&
                embedding_.identity() == verified_.embedding() && head_.identity() == verified_.head(),
                "borrowed native weights are poisoned/unloaded/replaced");
    }
    void fail() noexcept { poisoned_ = true; active_ = false; io_.drain(); }
    VerifiedDailyTarget verified_;
    const Embed& embedding_;
    const Head& head_;
    Io io_;
    OperationLease lease_;
    Operation operation_, last_operation_;
    std::string error_;
    int head_rows_ = 0;
    bool active_ = false, poisoned_ = false;
};
} // namespace strata::core::dflash2
