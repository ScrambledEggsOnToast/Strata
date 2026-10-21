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

namespace strata::core::dflash2 {
class DeviceForward;

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
                 const uint16_t* device_raw_embeddings, int frontier,
                 uint16_t* device_hidden_four_rows, void* stream, std::string& error,
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
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strata::core::dflash2
