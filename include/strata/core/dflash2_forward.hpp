// Native five-layer DFlash2. Immutable weights/shared scratch; request-local KV.
#pragma once

#include "strata/core/dflash2_bridge.hpp"
#include "strata/core/dflash2_weights.hpp"

#include <array>
#include <memory>
#include <span>
#include <string>
#include <string_view>

namespace strata::core { class ConversationBuffer; }
namespace strata::program { struct ResidentExecutorFixture; }

namespace strata::core::dflash2 {
class DeviceForward;
class CaptureProducer;
class DflashNativeDrafter;

// A trace is borrowed only during capture(). A sink may enqueue a readback into
// its OWN preallocated host storage on stream; subsequent writes are ordered on
// that same stream. It must not retain the device view, change the stream, run a
// nested draft, or allocate device memory. forward() drains those readbacks too.
struct ForwardTrace {
    enum class Type { bf16, fp32 };
    const char* operation = nullptr;
    int layer = -1;
    int stage = -1, offset = -1; // grouped convolution's prepare/finish and tap
    Type type = Type::bf16;
    const void* data = nullptr;
    std::array<int, 4> shape{};
    int rank = 0;
    int first_position = 0;
    size_t bytes() const noexcept;
};
class ForwardTraceSink {
public:
    virtual ~ForwardTraceSink() = default;
    virtual bool capture(const ForwardTrace&, void* stream, std::string& error) = 0;
};

class DeviceContext {
public:
    ~DeviceContext();
    DeviceContext(const DeviceContext&) = delete;
    DeviceContext& operator=(const DeviceContext&) = delete;
    int origin() const noexcept;
    int begin() const noexcept;
    int end() const noexcept;
    bool initialized() const noexcept;
    bool poisoned() const noexcept;
    int device() const noexcept;
private:
    friend class DeviceForward;
    struct Impl;
    explicit DeviceContext(std::unique_ptr<Impl>);
    std::unique_ptr<Impl> impl_;
};

// Public, persistable metadata. No device ordinal or request incarnation is
// serialized: an independently bound replacement GPU/request may restore the
// same immutable model. The common lifecycle owner certifies request ownership.
struct ContextSnapshotMetadata {
    std::string kind = "dflash2-bf16-kv";
    uint32_t version = 1;
    int layers = 5, kv_heads = 8, head_dim = 128, capacity = context_rows;
    int origin = 0, begin = 0, end = 0;
    bool initialized = false;
    std::string manifest_sha256;
    TargetIdentity target;
    // Historical field name covers all forward arithmetic modes, including attention.
    // Legacy default preserves existing import compatibility; numerical variants
    // must match even when immutable checkpoint/target identities are equal.
    std::string projection_mode = "legacy-bf16-v1";
};
class DeviceContextSnapshot {
public:
    ~DeviceContextSnapshot();
    DeviceContextSnapshot(const DeviceContextSnapshot&) = delete;
    DeviceContextSnapshot& operator=(const DeviceContextSnapshot&) = delete;
    const ContextSnapshotMetadata& metadata() const noexcept;
    // Compact chronological BF16 [layer,kind(K/V),head,end-begin,128].
    // Unused capacity is never exposed or serialized.
    std::span<const uint16_t> payload() const noexcept;
    std::span<const uint16_t> part(int layer, bool values) const noexcept;
    bool valid() const noexcept;
    // Import an existing ConversationBuffer image into this preallocated owner.
    // Identity/geometry/range/payload are checked before any snapshot mutation.
    bool assign(const ContextSnapshotMetadata&, std::span<const uint16_t>, std::string& error);
    bool assign(const ContextSnapshotMetadata&,
                const std::array<const ConversationBuffer*, 10>& parts, std::string& error);
    static size_t storage_bytes() noexcept;
private:
    friend class DeviceForward;
    struct Impl;
    explicit DeviceContextSnapshot(std::unique_ptr<Impl>);
    std::unique_ptr<Impl> impl_;
};

class DeviceForward {
public:
    DeviceForward(const RuntimeWeights&, int device);
    ~DeviceForward();
    DeviceForward(const DeviceForward&) = delete;
    DeviceForward& operator=(const DeviceForward&) = delete;

    std::unique_ptr<DeviceContext> create_context(int origin) const;
    std::unique_ptr<DeviceContextSnapshot> create_snapshot() const;
    // All methods are serialized by the SAME shared OperationLease. Completion
    // fences the caller stream before metadata publication. Once submitted work
    // fails, both the touched context and shared scratch remain poisoned.
    bool forward(DeviceContext&, const FeatureView& newly_committed,
                 const uint16_t* device_raw_embeddings, int rows, int frontier,
                 uint16_t* device_hidden_rows, void* stream, std::string& error,
                 ForwardTraceSink* trace = nullptr);
    // Commit context without constructing a proposal (prefill/catch-up).
    bool ingest(DeviceContext&, const FeatureView&, int frontier, void* stream, std::string& error,
                ForwardTraceSink* trace = nullptr);
    bool reset(DeviceContext&, int origin, std::string& error) const;
    bool rewind(DeviceContext&, int end, std::string& error) const;
    bool copy_context(const DeviceContext& from, DeviceContext& to, void* stream,
                      std::string& error) const;
    bool snapshot(const DeviceContext&, DeviceContextSnapshot&, void* stream, std::string& error) const;
    bool validate_restore(const DeviceContext&, const DeviceContextSnapshot&, std::string& error) const;
    bool restore(DeviceContext&, const DeviceContextSnapshot&, void* stream, std::string& error) const;

    const uint16_t* weight(std::string_view name) const;
    int device() const noexcept;
    size_t device_bytes() const noexcept;
    static size_t context_bytes() noexcept;
    static size_t scratch_bytes() noexcept;
    // Startup only, before device telemetry/admission and expert-cache pricing.
    // Unsupported physical peers retain an explicitly counted compact-host route.
    static bool prepare_feature_peers(std::span<const int> sources, int destination, std::string& error);
    // Off-mode identities stay byte-for-byte compatible with prior snapshots.
    static const std::string& numerical_mode();
private:
    friend class DflashNativeDrafter;
    friend struct strata::program::ResidentExecutorFixture;
    bool ingest_producer(DeviceContext&, const FeatureView&, CaptureProducer&,
                         const OperationLease::Guard&, int frontier, void* stream, std::string& error);
    static bool gather_features(CaptureProducer&, const FeatureView&, const OperationLease::Guard&,
        uint16_t* destination, int device, void* stream, uint64_t& local_bytes,
        uint64_t& peer_bytes, uint64_t& host_bytes, std::string& error, ForwardTraceSink* trace = nullptr);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strata::core::dflash2
