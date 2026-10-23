#include "strata/core/dflash2_forward.hpp"
#include "strata/core/dflash2_capture.hpp"
#include "strata/core/conversation_buffer.hpp"
#include "strata/kernels/dflash2_forward.hpp"
#include "strata/kernels/dflash2_context_blas.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <exception>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <stdexcept>
#include <utility>

namespace strata::core::dflash2 {
namespace {
namespace gpu = strata::kernels::dflash2;
constexpr size_t alignment = 256;
constexpr size_t row_bytes = gpu::head_dim * sizeof(uint16_t);
constexpr size_t full_part_elements = size_t(gpu::kv_heads) * context_rows * gpu::head_dim;

bool refuse(std::string& error, const char* reason) {
    error = "dflash2 forward: "; error += reason; return false;
}
void checked(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("dflash2 forward: ") + operation + ": " + cudaGetErrorString(status));
}
class DeviceScope {
public:
    explicit DeviceScope(int device) {
        checked(cudaGetDevice(&previous_), "get current device");
        if (previous_ != device) checked(cudaSetDevice(device), "select owning device");
    }
    ~DeviceScope() {
        if (previous_ >= 0 && cudaSetDevice(previous_) != cudaSuccess) std::terminate();
    }
    DeviceScope(const DeviceScope&) = delete;
    DeviceScope& operator=(const DeviceScope&) = delete;
private:
    int previous_ = -1;
};
struct DeviceAllocation {
    void* pointer = nullptr;
    int device = -1;
    void allocate(size_t bytes, int ordinal) {
        device = ordinal;
        checked(cudaMalloc(&pointer, bytes), "allocate bounded storage");
    }
    ~DeviceAllocation() {
        if (!pointer) return;
        int old = -1;
        if (cudaGetDevice(&old) != cudaSuccess || (old != device && cudaSetDevice(device) != cudaSuccess))
            std::terminate();
        // Public calls fence every submitted operation, including their failure
        // paths, before dropping their lease. No asynchronous view escapes.
        if (cudaFree(pointer) != cudaSuccess) std::terminate();
        if (old != device && cudaSetDevice(old) != cudaSuccess) std::terminate();
    }
    DeviceAllocation() = default;
    DeviceAllocation(const DeviceAllocation&) = delete;
    DeviceAllocation& operator=(const DeviceAllocation&) = delete;
};
struct PhaseEvents {
    int device = -1;
    std::array<cudaEvent_t, 3> events{};
    void initialize(int ordinal) {
        if (!feature_phases_enabled()) return;
        device = ordinal;
        for (auto& event : events) checked(cudaEventCreate(&event), "allocate feature phase event");
    }
    ~PhaseEvents() {
        if (device < 0) return;
        DeviceScope on(device);
        for (auto event : events)
            if (event && cudaEventDestroy(event) != cudaSuccess) std::terminate();
    }
};
struct FeaturePeerTopology {
    int destination = -1, count = 0;
    std::array<int, 4> sources{};
    std::array<bool, 4> peer{};
};
FeaturePeerTopology& feature_peer_topology() {
    static FeaturePeerTopology topology;
    return topology;
}
struct Scratch {
    uint16_t *features = nullptr, *target = nullptr, *context_k = nullptr, *context_v = nullptr;
    uint16_t *hidden = nullptr, *normalized = nullptr, *convolved = nullptr, *product = nullptr;
    uint16_t *dynamic = nullptr, *query = nullptr, *block_k = nullptr, *block_v = nullptr;
    uint16_t *attention = nullptr, *gate = nullptr, *up = nullptr, *cosine = nullptr, *sine = nullptr;
    float *scores = nullptr, *probabilities = nullptr, *statistics = nullptr, *inverse = nullptr;
    uint32_t* status = nullptr;
    gpu::AttentionValueWorkspace value_workspace;
    size_t bytes = 0;
};
struct Arena {
    unsigned char* base;
    size_t used = 0;
    template<class T> T* take(size_t n) {
        auto* result = base ? reinterpret_cast<T*>(base + used) : nullptr;
        used += (n * sizeof(T) + alignment - 1) / alignment * alignment;
        return result;
    }
};
Scratch scratch_layout(void* allocation) {
    Arena a{static_cast<unsigned char*>(allocation)};
    Scratch s;
    s.features = a.take<uint16_t>(size_t(context_rows) * feature_width);
    s.target = a.take<uint16_t>(size_t(context_rows) * hidden_size);
    s.context_k = a.take<uint16_t>(size_t(context_rows) * gpu::kv_width);
    s.context_v = a.take<uint16_t>(size_t(context_rows) * gpu::kv_width);
    s.hidden = a.take<uint16_t>(block_size * hidden_size);
    s.normalized = a.take<uint16_t>(block_size * hidden_size);
    s.convolved = a.take<uint16_t>(block_size * hidden_size);
    s.product = a.take<uint16_t>(block_size * hidden_size);
    s.dynamic = a.take<uint16_t>(block_size * gpu::dynamic_width);
    s.query = a.take<uint16_t>(block_size * gpu::query_width);
    s.block_k = a.take<uint16_t>(block_size * gpu::kv_width);
    s.block_v = a.take<uint16_t>(block_size * gpu::kv_width);
    s.attention = a.take<uint16_t>(block_size * gpu::query_width);
    s.gate = a.take<uint16_t>(block_size * gpu::intermediate);
    s.up = a.take<uint16_t>(block_size * gpu::intermediate);
    s.cosine = a.take<uint16_t>(gpu::max_keys * 64);
    s.sine = a.take<uint16_t>(gpu::max_keys * 64);
    s.scores = a.take<float>(size_t(gpu::query_heads) * block_size * gpu::max_keys);
    s.probabilities = a.take<float>(size_t(gpu::query_heads) * block_size * gpu::max_keys);
    s.statistics = a.take<float>(size_t(context_rows) * gpu::kv_heads * 2);
    s.inverse = a.take<float>(64);
    s.status = a.take<uint32_t>(1);
    if (gpu::attention_value_split_enabled()) {
        s.value_workspace.partial = a.take<float>(gpu::attention_value_split_partial_count);
        s.value_workspace.unsafe = a.take<uint8_t>(gpu::attention_value_split_flag_count);
        s.value_workspace.partial_count = gpu::attention_value_split_partial_count;
        s.value_workspace.flag_count = gpu::attention_value_split_flag_count;
    }
    s.bytes = a.used;
    return s;
}
size_t payload_elements(int begin, int end) {
    return size_t(gpu::layers) * 2 * gpu::kv_heads * size_t(end - begin) * gpu::head_dim;
}
bool metadata_range(const ContextSnapshotMetadata& m, std::string& error) {
    if (m.kind != "dflash2-bf16-kv" || m.version != 1 || m.layers != gpu::layers ||
        m.kv_heads != gpu::kv_heads || m.head_dim != gpu::head_dim || m.capacity != context_rows)
        return refuse(error, "snapshot kind/version/geometry mismatch");
    if (m.origin < 0 || m.origin > max_position || m.begin < m.origin || m.end < m.begin ||
        m.end > max_position || m.end - m.begin > context_rows ||
        (m.initialized ? m.begin != std::max(m.origin, m.end - context_rows) :
                         (m.begin != m.origin || m.end != m.origin)))
        return refuse(error, "snapshot absolute context extent mismatch");
    return true;
}
bool identity_matches(const ContextSnapshotMetadata& m, const std::string& digest,
                       const TargetIdentity& target, const std::string& mode, std::string& error) {
    return (m.manifest_sha256 == digest && m.target == target && m.projection_mode == mode) ||
        refuse(error, "snapshot immutable runtime/borrowed-target identity mismatch");
}
bool legal_request(std::string_view id) {
    if (id.empty() || id.size() > 128) return false;
    for (char c : id)
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
              (c >= '0' && c <= '9') || c == '_' || c == '.' || c == ':' || c == '-')) return false;
    return true;
}
bool device_pointer(const void* pointer, int device, std::string& error) {
    if (!pointer || reinterpret_cast<uintptr_t>(pointer) % alignof(uint16_t))
        return refuse(error, "null/misaligned device row pointer");
    cudaPointerAttributes attributes{};
    const cudaError_t status = cudaPointerGetAttributes(&attributes, pointer);
    if (status != cudaSuccess || attributes.type != cudaMemoryTypeDevice || attributes.device != device)
        return refuse(error, "rows do not reside on the explicitly owning device");
    return true;
}
bool overlaps(const void* p, size_t n, const void* q, size_t m) {
    const uintptr_t a = reinterpret_cast<uintptr_t>(p), b = reinterpret_cast<uintptr_t>(q);
    return a <= b ? b - a < n : a - b < m;
}
bool stream_preflight(void* stream, std::string& error) {
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    const cudaError_t status = cudaStreamIsCapturing(static_cast<cudaStream_t>(stream), &capture);
    if (status != cudaSuccess || capture != cudaStreamCaptureStatusNone)
        return refuse(error, "stream is invalid or graph capture forbids stream-complete publication");
    return true;
}
struct SubmittedFailure {};
struct Execution {
    void* stream;
    std::string& error;
    ForwardTraceSink* trace = nullptr;
    bool split_value_submitted = false;
    bool grouped_score_submitted = false;
    void must(int status, const char* operation) {
        if (status == int(cudaSuccess)) return;
        error = "dflash2 forward: "; error += operation; error += ": ";
        error += cudaGetErrorString(static_cast<cudaError_t>(status));
        throw SubmittedFailure{};
    }
    void must(cudaError_t status, const char* operation) { must(int(status), operation); }
    void emit(const char* operation, int layer, const void* data, ForwardTrace::Type type,
              std::initializer_list<int> shape, int first, int stage = -1, int offset = -1) {
        if (!trace) return;
        ForwardTrace record;
        record.operation = operation; record.layer = layer; record.stage = stage; record.offset = offset;
        record.data = data; record.type = type; record.rank = int(shape.size()); record.first_position = first;
        std::copy(shape.begin(), shape.end(), record.shape.begin());
        if (!trace->capture(record, stream, error)) {
            if (error.empty()) error = "dflash2 forward: trace readback refused";
            throw SubmittedFailure{};
        }
    }
    void drain_or_terminate(int device) noexcept {
        int previous = -1;
        // An unsuccessful drain is not permission to release borrowed buffers.
        // Stop without unwinding their owners when completion is unknowable.
        if (cudaGetDevice(&previous) != cudaSuccess ||
            (previous != device && cudaSetDevice(device) != cudaSuccess) ||
            cudaStreamSynchronize(static_cast<cudaStream_t>(stream)) != cudaSuccess)
            std::terminate();
        if (previous != device && cudaSetDevice(previous) != cudaSuccess) std::terminate();
    }
};
} // namespace

struct DeviceContext::Impl {
    DeviceAllocation storage;
    std::shared_ptr<const uint8_t> owner;
    int device = -1, origin = 0, begin = 0, end = 0;
    bool initialized = false, poisoned = false;
    std::array<char, 128> request{};
    size_t request_size = 0;
    int64_t incarnation = 0, last_generation = 0;
    uint16_t* ring() const { return static_cast<uint16_t*>(storage.pointer); }
    void clear_owner() { request_size = 0; incarnation = last_generation = 0; }
};
struct DeviceContextSnapshot::Impl {
    ContextSnapshotMetadata metadata;
    std::unique_ptr<uint16_t[]> storage;
    bool valid = false;
    Impl(const std::string& digest, const TargetIdentity& target, const std::string& mode)
        : storage(std::make_unique_for_overwrite<uint16_t[]>(gpu::kv_elements)) {
        metadata.manifest_sha256 = digest; metadata.target = target; metadata.projection_mode = mode;
    }
};
struct DeviceForward::Impl {
    struct Layer {
        const uint16_t *q, *k, *v, *o, *q_norm, *k_norm, *input_norm, *post_norm;
        const uint16_t *gate, *up, *down, *attention_base, *attention_projection, *mlp_base, *mlp_projection;
    };
    struct Entry { std::string name; const uint16_t* device = nullptr; };
    // Kept alive by contexts, so allocator address reuse cannot admit a stale
    // context after this DeviceForward has been destroyed and reconstructed.
    std::shared_ptr<const uint8_t> identity = std::make_shared<const uint8_t>(0);
    int device;
    size_t weight_bytes;
    std::string manifest;
    TargetIdentity target_identity;
    DeviceAllocation weights, scratch;
    std::unique_ptr<gpu::ContextBlas> context_blas;
    const std::string projection_mode = DeviceForward::numerical_mode();
    const bool value_trace = [] { const char* v = std::getenv("STRATA_DFLASH2_VALUE_SPLIT_TRACE"); return v && std::strcmp(v, "1") == 0; }();
    bool value_reported = false;
    const bool grouped_score_trace = [] { const char* v = std::getenv("STRATA_DFLASH2_SCORE_GROUPED_TRACE"); return v && std::strcmp(v, "1") == 0; }();
    bool grouped_score_reported = false;
    bool device_features_reported = false;
    PhaseEvents feature_timing;
    Scratch s;
    std::array<Layer, gpu::layers> layer{};
    std::array<Entry, 81> entries;
    size_t entry_count = 0;
    const uint16_t *fc = nullptr, *hidden_norm = nullptr, *final_norm = nullptr;
    mutable OperationLease lease;

    Impl(const RuntimeWeights& runtime, int ordinal)
        : device(ordinal), weight_bytes(runtime.dense().size_bytes()),
          manifest(runtime.manifest_sha256()), target_identity(runtime.target_identity()) {
        std::string flags_error;
        if (!validate_feature_flags(flags_error)) throw Refusal(flags_error);
        if (device < 0 || !weight_bytes) throw Refusal("dflash2 forward: invalid device/empty dense runtime");
        DeviceScope on(device);
        cudaDeviceProp properties{};
        checked(cudaGetDeviceProperties(&properties, device), "read device capability");
        if (properties.major < 7) throw Refusal("dflash2 forward: requires CUDA SM70 or later");
        if (gpu::attention_value_split_enabled() && (properties.major != 7 || properties.minor != 0))
            throw Refusal("dflash2 split value mode requires exercised CUDA SM70");
        if (gpu::attention_grouped_scores_enabled() &&
            !((properties.major == 7 && properties.minor == 0) || (properties.major == 8 && properties.minor == 6)))
            throw Refusal("dflash2 grouped scores require CUDA SM70 or SM86");
        if (gpu::context_blas_enabled()) context_blas = std::make_unique<gpu::ContextBlas>(device);
        weights.allocate(weight_bytes, device);
        scratch.allocate(DeviceForward::scratch_bytes(), device);
        s = scratch_layout(scratch.pointer);
        feature_timing.initialize(device);
        auto bind = [&](std::string name, std::initializer_list<int64_t> shape) {
            const auto& tensor = runtime.tensor(name);
            size_t elements = 1;
            for (int64_t n : shape) elements *= size_t(n);
            if (tensor.shape.size() != shape.size() || !std::equal(shape.begin(), shape.end(), tensor.shape.begin()) ||
                tensor.values.size() != elements || tensor.offset_bytes % 64 || tensor.offset_bytes > weight_bytes ||
                tensor.values.size_bytes() > weight_bytes - tensor.offset_bytes ||
                tensor.values.data() != runtime.dense().data() + tensor.offset_bytes / sizeof(uint16_t) ||
                entry_count == entries.size()) throw Refusal("dflash2 forward: runtime tensor extent mismatch");
            const auto* pointer = reinterpret_cast<const uint16_t*>(
                static_cast<const unsigned char*>(weights.pointer) + tensor.offset_bytes);
            entries[entry_count++] = {std::move(name), pointer};
            return pointer;
        };
        fc = bind("fc.weight", {hidden_size, feature_width});
        hidden_norm = bind("hidden_norm.weight", {hidden_size});
        final_norm = bind("norm.weight", {hidden_size});
        bind("candidate_selector.hidden_projection.weight", {256, hidden_size});
        bind("candidate_selector.predecessor_codebook", {vocabulary_size, 256});
        bind("candidate_selector.successor_codebook", {vocabulary_size, 256});
        for (int i = 0; i < gpu::layers; ++i) {
            const std::string prefix = "layers." + std::to_string(i) + ".";
            auto& l = layer[size_t(i)];
            l.q = bind(prefix + "self_attn.q_proj.weight", {gpu::query_width, hidden_size});
            l.k = bind(prefix + "self_attn.k_proj.weight", {gpu::kv_width, hidden_size});
            l.v = bind(prefix + "self_attn.v_proj.weight", {gpu::kv_width, hidden_size});
            l.o = bind(prefix + "self_attn.o_proj.weight", {hidden_size, gpu::query_width});
            l.q_norm = bind(prefix + "self_attn.q_norm.weight", {gpu::head_dim});
            l.k_norm = bind(prefix + "self_attn.k_norm.weight", {gpu::head_dim});
            l.input_norm = bind(prefix + "input_layernorm.weight", {hidden_size});
            l.post_norm = bind(prefix + "post_attention_layernorm.weight", {hidden_size});
            l.gate = bind(prefix + "mlp.gate_proj.weight", {gpu::intermediate, hidden_size});
            l.up = bind(prefix + "mlp.up_proj.weight", {gpu::intermediate, hidden_size});
            l.down = bind(prefix + "mlp.down_proj.weight", {hidden_size, gpu::intermediate});
            l.attention_base = bind(prefix + "attention_conv.base_kernel", {2, 2, hidden_size});
            l.attention_projection = bind(prefix + "attention_conv.kernel_projection.weight", {gpu::dynamic_width, hidden_size});
            l.mlp_base = bind(prefix + "mlp_conv.base_kernel", {2, 2, hidden_size});
            l.mlp_projection = bind(prefix + "mlp_conv.kernel_projection.weight", {gpu::dynamic_width, hidden_size});
        }
        // No borrowed mmap pointer escapes construction. The immutable full
        // dense copy and device-generated RoPE frequencies are complete here.
        checked(cudaMemcpy(weights.pointer, runtime.dense().data(), weight_bytes, cudaMemcpyHostToDevice), "upload immutable BF16 runtime");
        checked(static_cast<cudaError_t>(gpu::inverse_frequencies(s.inverse, nullptr)), "initialize inverse frequencies");
        checked(cudaStreamSynchronize(nullptr), "complete immutable device initialization");
    }
    bool owns(const DeviceContext::Impl& c, std::string& error) const {
        return (c.owner == identity && c.device == device && !c.poisoned) ||
            refuse(error, "foreign or poisoned context");
    }
    bool acquire(OperationLease::Guard& guard, std::string& error) const {
        if (guard) return true;
        return refuse(error, guard.failure() == OperationLease::Failure::poisoned
            ? "shared scratch is poisoned" : "shared operation lease timed out");
    }
    bool feature_preflight(const DeviceContext::Impl& c, const FeatureView& f,
                           int frontier, int forward_rows, int& begin, std::string& error) const {
        if (!owns(c, error)) return false;
        if (frontier < c.origin || !absolute_range_fits(frontier, forward_rows) ||
            f.count < 0 || f.count > context_rows || f.start < c.origin ||
            f.start > frontier || f.count != frontier - f.start ||
            (c.initialized ? f.start != c.end : f.start != std::max(c.origin, frontier - context_rows)))
            return refuse(error, "features are not the contiguous committed absolute context");
        begin = c.initialized ? std::max(c.begin, frontier - context_rows) : f.start;
        if (frontier - begin > context_rows || !legal_request(f.request_id) || f.incarnation < 0 ||
            (c.request_size && (std::string_view(c.request.data(), c.request_size) != f.request_id || c.incarnation != f.incarnation)))
            return refuse(error, "context retention or request incarnation mismatch");
        size_t rows = 0;
        int64_t generation = c.last_generation;
        for (size_t part = 0; part < f.segments.size(); ++part) {
            const auto values = f.segments[part];
            if (values.size() % feature_width || (values.size() && !values.data()))
                return refuse(error, "misaligned/wrong-width feature span");
            const size_t n = values.size() / feature_width;
            if (n > size_t(context_rows) - rows || f.generations[part].size() != n)
                return refuse(error, "feature/generation span geometry mismatch");
            rows += n;
            for (int64_t g : f.generations[part]) {
                if (g < 1 || g < generation) return refuse(error, "uncommitted or stale feature generation");
                generation = g;
            }
        }
        if (rows != size_t(f.count)) return refuse(error, "feature span extent mismatch");
        return true;
    }
    void publish(DeviceContext::Impl& c, const FeatureView& f, int begin, int frontier) const {
        c.begin = begin; c.end = frontier; c.initialized = true;
        if (!c.request_size) {
            std::copy(f.request_id.begin(), f.request_id.end(), c.request.begin());
            c.request_size = f.request_id.size(); c.incarnation = f.incarnation;
        }
        for (const auto& part : f.generations) if (!part.empty()) c.last_generation = part.back();
    }
    void normalize(Execution& e, const char* unweighted, const char* weighted, int index,
                   const uint16_t* input, uint16_t* output, const uint16_t* weight,
                   int rows, int width, int position) const {
        e.must(gpu::normalize_bf16(input, output, s.statistics, rows, width, s.status, e.stream), unweighted);
        e.emit(unweighted, index, output, ForwardTrace::Type::bf16, {rows, width}, position);
        e.emit("rms.mean_square_and_inverse", index, s.statistics, ForwardTrace::Type::fp32, {rows, 2}, position);
        e.must(gpu::multiply_norm_bf16(output, weight, rows * width, width, s.status, e.stream), weighted);
        e.emit(weighted, index, output, ForwardTrace::Type::bf16, {rows, width}, position);
    }
    void project(Execution& e, const char* name, int index, const uint16_t* input,
                 const uint16_t* weight, uint16_t* output, int rows, int ni, int no, int position) const {
        e.must(gpu::linear_bf16(input, weight, output, rows, ni, no, e.stream, s.status), name);
        e.emit(name, index, output, ForwardTrace::Type::bf16, {rows, no}, position);
    }
    void project_context(Execution& e, const char* name, int index, const uint16_t* input,
                         const uint16_t* weight, uint16_t* output, int rows, int ni, int no, int position) const {
        if (!context_blas || rows < 128) { project(e, name, index, input, weight, output, rows, ni, no, position); return; }
        if (!context_blas->project(input, weight, output, rows, ni, no, s.status, e.stream, e.error)) throw SubmittedFailure{};
        e.emit(name, index, output, ForwardTrace::Type::bf16, {rows, no}, position);
    }
    void context_layer(Execution& e, DeviceContext::Impl& c, const FeatureView& f, int index) const {
        if (!f.count) return;
        const auto& l = layer[size_t(index)];
        project_context(e, "context.k_proj", index, s.target, l.k, s.context_k, f.count, hidden_size, gpu::kv_width, f.start);
        project_context(e, "context.v_proj", index, s.target, l.v, s.context_v, f.count, hidden_size, gpu::kv_width, f.start);
        normalize(e, "context.k_norm.normalized", "context.k_norm.weighted", index,
                  s.context_k, s.context_k, l.k_norm, f.count * gpu::kv_heads, gpu::head_dim, f.start);
        e.must(gpu::rotary_bf16(s.context_k, s.cosine, s.sine, f.count, gpu::kv_heads, s.status, e.stream), "context RoPE");
        e.emit("context.k_rope", index, s.context_k, ForwardTrace::Type::bf16,
               {f.count, gpu::kv_heads, gpu::head_dim}, f.start);
        e.must(gpu::store_context(s.context_k, s.context_v, c.ring(), index, f.start, f.count, s.status, e.stream), "write committed context KV");
    }
    void convolve(Execution& e, const uint16_t* input, const uint16_t* base,
                  bool attention, int stage, int index, int frontier, int block) const {
        const char* product = attention ? "attention_conv.base_product" : "mlp_conv.base_product";
        const char* addition = attention ? "attention_conv.base_add" : "mlp_conv.base_add";
        const char* dynamic = attention ? "attention_conv.addcmul" : "mlp_conv.addcmul";
        e.must(cudaMemsetAsync(s.convolved, 0, size_t(block) * hidden_size * sizeof(uint16_t), static_cast<cudaStream_t>(e.stream)), "reset block-local convolution");
        for (int offset = 0; offset < 2; ++offset) {
            e.must(gpu::convolution_product(input, base, s.product, block, stage, offset, s.status, e.stream), product);
            e.emit(product, index, s.product, ForwardTrace::Type::bf16, {block, hidden_size}, frontier, stage, offset);
            e.must(gpu::add_bf16(s.convolved, s.product, s.convolved, size_t(block) * hidden_size, s.status, e.stream), addition);
            e.emit(addition, index, s.convolved, ForwardTrace::Type::bf16, {block, hidden_size}, frontier, stage, offset);
            e.must(gpu::convolution_addcmul(s.convolved, s.dynamic, input, block, stage, offset, s.status, e.stream), dynamic);
            e.emit(dynamic, index, s.convolved, ForwardTrace::Type::bf16, {block, hidden_size}, frontier, stage, offset);
        }
    }
    void block_layer(Execution& e, DeviceContext::Impl& c, int begin, int frontier,
                     int new_rows, int index, int block) const {
        const auto& l = layer[size_t(index)];
        normalize(e, "input_layernorm.normalized", "input_layernorm.weighted", index,
                  s.hidden, s.normalized, l.input_norm, block, hidden_size, frontier);
        project(e, "attention_conv.kernel_projection", index, s.normalized, l.attention_projection,
                s.dynamic, block, hidden_size, gpu::dynamic_width, frontier);
        convolve(e, s.normalized, l.attention_base, true, 0, index, frontier, block);
        project(e, "block.q_proj", index, s.convolved, l.q, s.query, block, hidden_size, gpu::query_width, frontier);
        normalize(e, "block.q_norm.normalized", "block.q_norm.weighted", index, s.query, s.query,
                  l.q_norm, block * gpu::query_heads, gpu::head_dim, frontier);
        project(e, "block.k_proj", index, s.convolved, l.k, s.block_k, block, hidden_size, gpu::kv_width, frontier);
        project(e, "block.v_proj", index, s.convolved, l.v, s.block_v, block, hidden_size, gpu::kv_width, frontier);
        normalize(e, "block.k_norm.normalized", "block.k_norm.weighted", index, s.block_k, s.block_k,
                  l.k_norm, block * gpu::kv_heads, gpu::head_dim, frontier);
        const auto* cos = s.cosine + size_t(new_rows) * 64;
        const auto* sin = s.sine + size_t(new_rows) * 64;
        e.must(gpu::rotary_bf16(s.query, cos, sin, block, gpu::query_heads, s.status, e.stream), "query RoPE");
        e.emit("block.q_rope", index, s.query, ForwardTrace::Type::bf16, {block, gpu::query_heads, gpu::head_dim}, frontier);
        e.must(gpu::rotary_bf16(s.block_k, cos, sin, block, gpu::kv_heads, s.status, e.stream), "block key RoPE");
        e.emit("block.k_rope", index, s.block_k, ForwardTrace::Type::bf16, {block, gpu::kv_heads, gpu::head_dim}, frontier);
        const int keys = frontier - begin + block;
        e.must(gpu::attention_scores(s.query, s.block_k, c.ring(), s.scores, index, begin, frontier, block, s.status, e.stream), "absolute-window attention scores");
        e.grouped_score_submitted |= gpu::attention_grouped_scores_enabled();
        e.emit("attention.scaled_masked_scores", index, s.scores, ForwardTrace::Type::fp32, {gpu::query_heads, block, keys}, begin);
        e.must(gpu::attention_softmax(s.scores, s.probabilities, s.statistics, block, keys, s.status, e.stream), "FP32 attention softmax");
        e.emit("attention.softmax", index, s.probabilities, ForwardTrace::Type::fp32, {gpu::query_heads, block, keys}, begin);
        e.emit("attention.max_and_sum", index, s.statistics, ForwardTrace::Type::fp32, {gpu::query_heads, block, 2}, frontier);
        e.must(gpu::attention_values_planned(s.probabilities, s.block_v, c.ring(), s.attention, index, begin, frontier, block, s.status, e.stream, s.value_workspace), "FP32 attention value product");
        e.split_value_submitted |= gpu::attention_value_split_enabled() && frontier - begin >= gpu::attention_value_split_min_context;
        e.emit("attention.value_product", index, s.attention, ForwardTrace::Type::bf16, {block, gpu::query_width}, frontier);
        project(e, "attention.o_proj", index, s.attention, l.o, s.normalized, block, gpu::query_width, hidden_size, frontier);
        convolve(e, s.normalized, l.attention_base, true, 1, index, frontier, block);
        e.must(gpu::add_bf16(s.hidden, s.convolved, s.hidden, block * hidden_size, s.status, e.stream), "attention residual add");
        e.emit("attention.residual", index, s.hidden, ForwardTrace::Type::bf16, {block, hidden_size}, frontier);
        normalize(e, "post_attention_layernorm.normalized", "post_attention_layernorm.weighted", index,
                  s.hidden, s.normalized, l.post_norm, block, hidden_size, frontier);
        project(e, "mlp_conv.kernel_projection", index, s.normalized, l.mlp_projection,
                s.dynamic, block, hidden_size, gpu::dynamic_width, frontier);
        convolve(e, s.normalized, l.mlp_base, false, 0, index, frontier, block);
        project(e, "mlp.gate_proj", index, s.convolved, l.gate, s.gate, block, hidden_size, gpu::intermediate, frontier);
        project(e, "mlp.up_proj", index, s.convolved, l.up, s.up, block, hidden_size, gpu::intermediate, frontier);
        e.must(gpu::silu_bf16(s.gate, block * gpu::intermediate, s.status, e.stream), "SiLU gate");
        e.emit("mlp.silu", index, s.gate, ForwardTrace::Type::bf16, {block, gpu::intermediate}, frontier);
        e.must(gpu::multiply_bf16(s.gate, s.up, s.gate, block * gpu::intermediate, s.status, e.stream), "SiLU/up BF16 product");
        e.emit("mlp.gate_up_product", index, s.gate, ForwardTrace::Type::bf16, {block, gpu::intermediate}, frontier);
        project(e, "mlp.down_proj", index, s.gate, l.down, s.normalized, block, gpu::intermediate, hidden_size, frontier);
        convolve(e, s.normalized, l.mlp_base, false, 1, index, frontier, block);
        e.must(gpu::add_bf16(s.hidden, s.convolved, s.hidden, block * hidden_size, s.status, e.stream), "MLP residual add");
        e.emit("mlp.residual", index, s.hidden, ForwardTrace::Type::bf16, {block, hidden_size}, frontier);
    }
    bool run(DeviceContext::Impl& c, const FeatureView& features, const uint16_t* embeddings, int proposal_rows,
             int frontier, uint16_t* output, void* stream, std::string& error, ForwardTraceSink* trace,
             CaptureProducer* producer = nullptr, const OperationLease::Guard* producer_lease = nullptr) {
        OperationLease::Guard guard(lease);
        if (!acquire(guard, error)) return false;
        const bool proposal = embeddings != nullptr;
        int begin = 0;
        if (!feature_preflight(c, features, frontier, proposal ? proposal_rows : 0, begin, error)) { guard.complete(); return false; }
        bool submitted = false;
        uint32_t numeric = 0;
        Execution e{stream, error, trace};
        try {
            DeviceScope on(device);
            if (!stream_preflight(stream, error)) { guard.complete(); return false; }
            if (proposal && (!device_pointer(embeddings, device, error) || !device_pointer(output, device, error))) {
                guard.complete(); return false;
            }
            if (proposal && (overlaps(output, block_size * hidden_size * sizeof(uint16_t), weights.pointer, weight_bytes) ||
                             overlaps(output, block_size * hidden_size * sizeof(uint16_t), scratch.pointer, s.bytes) ||
                             overlaps(output, block_size * hidden_size * sizeof(uint16_t), c.ring(), DeviceForward::context_bytes()))) {
                guard.complete(); return refuse(error, "hidden output overlaps immutable weights or private owner storage");
            }
            if (producer && (!producer_lease || !producer->device_preflight(features, *producer_lease, error))) {
                guard.complete(); return false;
            }
            submitted = true;
            e.must(cudaMemsetAsync(s.status, 0, sizeof(uint32_t), static_cast<cudaStream_t>(stream)), "reset numerical failure flag");
            const bool phases = feature_phases_enabled() && features.count != 0;
            auto& events = feature_timing.events;
            if (phases) e.must(cudaEventRecord(events[0], static_cast<cudaStream_t>(stream)), "record feature input start");
            const size_t uploaded = size_t(features.count) * feature_width;
            uint64_t local_bytes = 0, peer_bytes = 0, host_bytes = 0;
            if (producer) {
                if (!DeviceForward::gather_features(*producer, features, *producer_lease, s.features, device,
                        stream, local_bytes, peer_bytes, host_bytes, error, trace)) throw SubmittedFailure{};
            } else {
                size_t offset = 0;
                for (const auto part : features.segments) {
                    if (!part.empty()) {
                        e.must(cudaMemcpyAsync(s.features + offset, part.data(), part.size_bytes(), cudaMemcpyHostToDevice,
                                               static_cast<cudaStream_t>(stream)), "upload committed feature span exactly once");
                        offset += part.size(); host_bytes += part.size_bytes();
                    }
                }
            }
            if (phases) e.must(cudaEventRecord(events[1], static_cast<cudaStream_t>(stream)), "record completed feature input boundary");
            if (features.count) {
                e.must(gpu::finite_bf16(s.features, uploaded, s.status, stream), "validate committed feature finiteness");
                e.emit("context.features", -1, s.features, ForwardTrace::Type::bf16, {features.count, feature_width}, features.start);
                project_context(e, "context.fc", -1, s.features, fc, s.target, features.count, feature_width, hidden_size, features.start);
                normalize(e, "hidden_norm.normalized", "hidden_norm.weighted", -1,
                          s.target, s.target, hidden_norm, features.count, hidden_size, features.start);
            }
            const int positions = features.count + (proposal ? proposal_rows : 0);
            if (positions) {
                e.must(gpu::rotary_tables(s.inverse, s.cosine, s.sine, features.start, positions, s.status, stream), "absolute RoPE tables");
                e.emit("rope.inverse_frequencies", -1, s.inverse, ForwardTrace::Type::fp32, {64}, 0);
                e.emit("rope.cosine", -1, s.cosine, ForwardTrace::Type::bf16, {positions, 64}, features.start);
                e.emit("rope.sine", -1, s.sine, ForwardTrace::Type::bf16, {positions, 64}, features.start);
            }
            if (proposal) {
                e.must(cudaMemcpyAsync(s.hidden, embeddings, size_t(proposal_rows) * hidden_size * sizeof(uint16_t), cudaMemcpyDeviceToDevice,
                                       static_cast<cudaStream_t>(stream)), "stage four raw borrowed embedding rows");
                e.must(gpu::finite_bf16(s.hidden, size_t(proposal_rows) * hidden_size, s.status, stream), "validate raw embedding finiteness");
                e.emit("raw_embeddings", -1, s.hidden, ForwardTrace::Type::bf16, {proposal_rows, hidden_size}, frontier);
            }
            for (int index = 0; index < gpu::layers; ++index) {
                context_layer(e, c, features, index);
                if (proposal) block_layer(e, c, begin, frontier, features.count, index, proposal_rows);
            }
            if (proposal) {
                normalize(e, "norm.normalized", "norm.weighted", -1, s.hidden, s.hidden,
                          final_norm, proposal_rows, hidden_size, frontier);
                e.must(cudaMemcpyAsync(output, s.hidden, size_t(proposal_rows) * hidden_size * sizeof(uint16_t), cudaMemcpyDeviceToDevice,
                                       static_cast<cudaStream_t>(stream)), "publish four normalized BF16 hidden rows");
            }
            if (phases) e.must(cudaEventRecord(events[2], static_cast<cudaStream_t>(stream)), "record context projection completion");
            e.must(cudaMemcpyAsync(&numeric, s.status, sizeof(numeric), cudaMemcpyDeviceToHost,
                                   static_cast<cudaStream_t>(stream)), "read numerical failure flag");
            e.must(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)), "complete all five layers and trace readbacks");
            if (numeric) { error = "dflash2 forward: nonfinite BF16/FP32 operation poisoned context and scratch"; throw SubmittedFailure{}; }
            if (context_blas) context_blas->report_completed();
            if (phases) {
                float input_ms = 0, projection_ms = 0;
                e.must(cudaEventElapsedTime(&input_ms, events[0], events[1]), "read completed feature input duration");
                e.must(cudaEventElapsedTime(&projection_ms, events[1], events[2]), "read completed context projection duration");
                std::fprintf(stderr, "DFLASH2_PHASE feature_input completed=1 device=%d start=%d rows=%d "
                    "route=%s input_gpu_ms=%.6f context_gpu_ms=%.6f local_device_bytes=%llu "
                    "peer_device_bytes=%llu host_upload_bytes=%llu\n", device, features.start, features.count,
                    producer ? "authenticated_verifier" : "host_ring", input_ms, projection_ms,
                    (unsigned long long)local_bytes, (unsigned long long)peer_bytes, (unsigned long long)host_bytes);
            }
            if (producer && !device_features_reported &&
                    (feature_phases_enabled() || std::getenv("STRATA_STATE_HASH") || std::getenv("DFLASH2_CAPTURE_DIR"))) {
                std::fprintf(stderr, "DFLASH2_DEVICE_FEATURES completed=1 device=%d start=%d rows=%d "
                    "producer_generation=%llu local_device_bytes=%llu peer_device_bytes=%llu host_upload_bytes=%llu "
                    "wholly_device_resident=%d\n", device, features.start, features.count,
                    (unsigned long long)producer->owner_.generation, (unsigned long long)local_bytes,
                    (unsigned long long)peer_bytes, (unsigned long long)host_bytes, int(host_bytes == 0));
                device_features_reported = true;
            }
            publish(c, features, begin, frontier);
            if (e.split_value_submitted && value_trace && !value_reported) {
                std::fprintf(stderr, "DFLASH2_VALUE_SPLIT completed=1 scope=DeviceForward device=%d rows=%d context=%d chunk=%d workspace=%zu mode=%s\n",
                    device, proposal_rows, frontier - begin, gpu::attention_value_split_chunk,
                    gpu::attention_value_split_bytes, projection_mode.c_str());
                value_reported = true;
            }
            if (e.grouped_score_submitted && grouped_score_trace && !grouped_score_reported) {
                std::fprintf(stderr, "DFLASH2_GROUPED_SCORE completed=1 scope=DeviceForward device=%d rows=%d context=%d heads_per_kv=4 workspace=0\n",
                    device, proposal_rows, frontier - begin);
                grouped_score_reported = true;
            }
            guard.complete(); error.clear(); return true;
        } catch (const SubmittedFailure&) {
            e.drain_or_terminate(device);
        } catch (const std::exception& failure) {
            if (submitted) e.drain_or_terminate(device);
            error = failure.what();
        } catch (...) {
            if (submitted) e.drain_or_terminate(device);
            error = "dflash2 forward: unexpected execution/trace failure";
        }
        if (submitted) c.poisoned = true;
        else guard.complete();
        return false;
    }
    bool same_snapshot(const DeviceContextSnapshot::Impl& image, std::string& error) const {
        return metadata_range(image.metadata, error) && identity_matches(image.metadata, manifest, target_identity, projection_mode, error);
    }
    // Device ring <-> chronological host image. Two 2D copies cover a wrap;
    // each 2D height is eight heads, not a per-row upload/readback loop.
    void image_copies(Execution& e, uint16_t* ring, uint16_t* image, int begin, int end,
                      cudaMemcpyKind kind) const {
        const int count = end - begin;
        if (!count) return;
        const int slot = begin % context_rows, first = std::min(count, context_rows - slot);
        const size_t compact_pitch = size_t(count) * row_bytes, ring_pitch = size_t(context_rows) * row_bytes;
        for (int part = 0; part < gpu::layers * 2; ++part) {
            auto* d = ring + size_t(part) * full_part_elements;
            auto* h = image + size_t(part) * gpu::kv_heads * count * gpu::head_dim;
            if (kind == cudaMemcpyDeviceToHost) {
                e.must(cudaMemcpy2DAsync(h, compact_pitch, d + size_t(slot) * gpu::head_dim, ring_pitch,
                    size_t(first) * row_bytes, gpu::kv_heads, kind, static_cast<cudaStream_t>(e.stream)), "snapshot chronological context KV");
                if (first < count) e.must(cudaMemcpy2DAsync(h + size_t(first) * gpu::head_dim, compact_pitch, d, ring_pitch,
                    size_t(count - first) * row_bytes, gpu::kv_heads, kind, static_cast<cudaStream_t>(e.stream)), "snapshot wrapped context KV");
            } else {
                e.must(cudaMemcpy2DAsync(d + size_t(slot) * gpu::head_dim, ring_pitch, h, compact_pitch,
                    size_t(first) * row_bytes, gpu::kv_heads, kind, static_cast<cudaStream_t>(e.stream)), "restore chronological context KV");
                if (first < count) e.must(cudaMemcpy2DAsync(d, ring_pitch, h + size_t(first) * gpu::head_dim, compact_pitch,
                    size_t(count - first) * row_bytes, gpu::kv_heads, kind, static_cast<cudaStream_t>(e.stream)), "restore wrapped context KV");
            }
        }
    }
};

size_t ForwardTrace::bytes() const noexcept {
    size_t n = 1;
    for (int i = 0; i < rank; ++i) n *= size_t(shape[size_t(i)]);
    return n * (type == Type::bf16 ? sizeof(uint16_t) : sizeof(float));
}
DeviceContext::DeviceContext(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
DeviceContext::~DeviceContext() = default;
int DeviceContext::origin() const noexcept { return impl_->origin; }
int DeviceContext::begin() const noexcept { return impl_->begin; }
int DeviceContext::end() const noexcept { return impl_->end; }
bool DeviceContext::initialized() const noexcept { return impl_->initialized; }
bool DeviceContext::poisoned() const noexcept { return impl_->poisoned; }
int DeviceContext::device() const noexcept { return impl_->device; }
DeviceContextSnapshot::DeviceContextSnapshot(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
DeviceContextSnapshot::~DeviceContextSnapshot() = default;
const ContextSnapshotMetadata& DeviceContextSnapshot::metadata() const noexcept { return impl_->metadata; }
bool DeviceContextSnapshot::valid() const noexcept { return impl_->valid; }
std::span<const uint16_t> DeviceContextSnapshot::payload() const noexcept {
    return {impl_->storage.get(), impl_->valid ? payload_elements(impl_->metadata.begin, impl_->metadata.end) : 0};
}
std::span<const uint16_t> DeviceContextSnapshot::part(int layer, bool values) const noexcept {
    if (!impl_->valid || layer < 0 || layer >= gpu::layers) return {};
    const size_t elements = size_t(gpu::kv_heads) * (impl_->metadata.end - impl_->metadata.begin) * gpu::head_dim;
    return {impl_->storage.get() + (size_t(layer) * 2 + size_t(values)) * elements, elements};
}
size_t DeviceContextSnapshot::storage_bytes() noexcept { return DeviceForward::context_bytes(); }
bool DeviceContextSnapshot::assign(const ContextSnapshotMetadata& metadata, std::span<const uint16_t> payload,
                                    std::string& error) {
    if (!metadata_range(metadata, error) ||
        !identity_matches(metadata, impl_->metadata.manifest_sha256, impl_->metadata.target, impl_->metadata.projection_mode, error)) return false;
    if (payload.size() != payload_elements(metadata.begin, metadata.end) || (payload.size() && !payload.data()))
        return refuse(error, "snapshot payload does not own the declared retained KV");
    for (uint16_t bits : payload)
        if ((bits & 0x7f80u) == 0x7f80u) return refuse(error, "snapshot contains nonfinite BF16 KV");
    if (!payload.empty()) std::memmove(impl_->storage.get(), payload.data(), payload.size_bytes());
    impl_->metadata.origin = metadata.origin; impl_->metadata.begin = metadata.begin; impl_->metadata.end = metadata.end;
    impl_->metadata.initialized = metadata.initialized; impl_->valid = true;
    error.clear(); return true;
}
bool DeviceContextSnapshot::assign(const ContextSnapshotMetadata& metadata,
                                    const std::array<const ConversationBuffer*, 10>& parts,
                                    std::string& error) {
    if (!metadata_range(metadata, error) ||
        !identity_matches(metadata, impl_->metadata.manifest_sha256, impl_->metadata.target, impl_->metadata.projection_mode, error)) return false;
    const size_t elements = size_t(gpu::kv_heads) * (metadata.end - metadata.begin) * gpu::head_dim;
    const size_t bytes = elements * sizeof(uint16_t);
    for (const auto* part : parts)
        if (!part || part->size() != bytes) return refuse(error, "segmented snapshot part extent mismatch");
    // Validate all immutable input parts before changing this image. Carry the
    // low byte across arbitrary ConversationBuffer segment cuts.
    for (const auto* part : parts) {
        uint8_t low = 0;
        if (!part->visit(0, bytes, [&](const uint8_t* data, size_t n, size_t at) {
                for (size_t i = 0; i < n; ++i) {
                    if (!((at + i) & 1)) low = data[i];
                    else if ((low & 0x80u) && (data[i] & 0x7fu) == 0x7fu) return false;
                }
                return true;
            })) return refuse(error, "segmented snapshot contains invalid/nonfinite BF16 KV");
    }
    impl_->valid = false;
    for (size_t i = 0; i < parts.size(); ++i)
        if (!parts[i]->read(impl_->storage.get() + i * elements, 0, bytes))
            return refuse(error, "segmented snapshot read failed");
    impl_->metadata.origin = metadata.origin; impl_->metadata.begin = metadata.begin; impl_->metadata.end = metadata.end;
    impl_->metadata.initialized = metadata.initialized; impl_->valid = true;
    error.clear(); return true;
}

DeviceForward::DeviceForward(const RuntimeWeights& runtime, int device) : impl_(std::make_unique<Impl>(runtime, device)) {}
DeviceForward::~DeviceForward() = default;
int DeviceForward::device() const noexcept { return impl_->device; }
size_t DeviceForward::device_bytes() const noexcept { return impl_->weight_bytes + scratch_bytes() + (impl_->context_blas ? gpu::ContextBlas::device_budget : 0); }
size_t DeviceForward::context_bytes() noexcept { return gpu::kv_elements * sizeof(uint16_t); }
size_t DeviceForward::scratch_bytes() noexcept { return scratch_layout(nullptr).bytes; }
const std::string& DeviceForward::numerical_mode() {
    static const std::string mode = [] {
        std::string result = gpu::context_projection_identity();
        if (gpu::attention_value_split_enabled()) { result += ":"; result += gpu::attention_value_split_identity(); }
        return result;
    }();
    return mode;
}
const uint16_t* DeviceForward::weight(std::string_view name) const {
    for (size_t i = 0; i < impl_->entry_count; ++i) if (impl_->entries[i].name == name) return impl_->entries[i].device;
    throw Refusal("dflash2 forward: unknown immutable device tensor");
}
std::unique_ptr<DeviceContext> DeviceForward::create_context(int origin) const {
    if (origin < 0 || origin > max_position) throw Refusal("dflash2 forward: invalid context origin");
    DeviceScope on(impl_->device);
    auto c = std::make_unique<DeviceContext::Impl>();
    c->owner = impl_->identity; c->device = impl_->device; c->origin = c->begin = c->end = origin;
    c->storage.allocate(context_bytes(), impl_->device);
    return std::unique_ptr<DeviceContext>(new DeviceContext(std::move(c)));
}
std::unique_ptr<DeviceContextSnapshot> DeviceForward::create_snapshot() const {
    return std::unique_ptr<DeviceContextSnapshot>(new DeviceContextSnapshot(
        std::make_unique<DeviceContextSnapshot::Impl>(impl_->manifest, impl_->target_identity, impl_->projection_mode)));
}
bool DeviceForward::forward(DeviceContext& context, const FeatureView& features, const uint16_t* embeddings,
                            int rows, int frontier, uint16_t* output, void* stream, std::string& error,
                            ForwardTraceSink* trace) {
    if (!embeddings || !output) return refuse(error, "proposal requires embedding and output rows");
    if (rows < 1 || rows > block_size) return refuse(error, "proposal row count is outside the block capacity");
    return impl_->run(*context.impl_, features, embeddings, rows, frontier, output, stream, error, trace);
}
bool DeviceForward::ingest(DeviceContext& context, const FeatureView& features, int frontier,
                           void* stream, std::string& error, ForwardTraceSink* trace) {
    return impl_->run(*context.impl_, features, nullptr, 0, frontier, nullptr, stream, error, trace);
}
bool DeviceForward::ingest_producer(DeviceContext& context, const FeatureView& features, CaptureProducer& producer,
        const OperationLease::Guard& lease, int frontier, void* stream, std::string& error) {
    if (!device_features_enabled() || producer.phase_ != CapturePhase::verify)
        return ingest(context, features, frontier, stream, error);
    return impl_->run(*context.impl_, features, nullptr, 0, frontier, nullptr, stream, error, nullptr, &producer, &lease);
}
bool DeviceForward::prepare_feature_peers(std::span<const int> sources, int destination, std::string& error) {
    if (!validate_feature_flags(error)) return false;
    if (!device_features_enabled()) return true;
    int devices = 0;
    const auto status = cudaGetDeviceCount(&devices);
    if (status != cudaSuccess || sources.empty() || sources.size() > 4 || destination < 0 || destination >= devices)
        return refuse(error, "invalid startup device-feature topology");
    for (size_t i = 0; i < sources.size(); ++i)
        if (sources[i] < 0 || sources[i] >= devices ||
            std::find(sources.begin(), sources.begin() + ptrdiff_t(i), sources[i]) != sources.begin() + ptrdiff_t(i))
            return refuse(error, "device-feature sources must be distinct real device ordinals");
    auto& topology = feature_peer_topology();
    if (topology.count) {
        return (topology.destination == destination && topology.count == int(sources.size()) &&
                std::equal(sources.begin(), sources.end(), topology.sources.begin())) ||
            refuse(error, "startup device-feature topology cannot be rebound");
    }
    try {
        DeviceScope on(destination);
        FeaturePeerTopology prepared;
        prepared.destination = destination; prepared.count = int(sources.size());
        std::copy(sources.begin(), sources.end(), prepared.sources.begin());
        for (size_t i = 0; i < sources.size(); ++i) {
            const int source = sources[i];
            if (source != destination) {
                int access = 0;
                checked(cudaDeviceCanAccessPeer(&access, destination, source), "check physical feature peer access");
                if (access) {
                    const auto enabled = cudaDeviceEnablePeerAccess(source, 0);
                    if (enabled == cudaErrorPeerAccessAlreadyEnabled) (void)cudaGetLastError();
                    else checked(enabled, "enable physical feature peer access before admission");
                    prepared.peer[i] = true;
                }
            }
            std::fprintf(stderr, "DFLASH2_DEVICE_FEATURE_TOPOLOGY source=%d destination=%d "
                "route=%s admission_pending=1\n", source, destination,
                source == destination ? "local_device" : prepared.peer[i] ? "peer_device" : "compact_host_staging");
        }
        topology = prepared; error.clear(); return true;
    } catch (const std::exception& failure) { error = failure.what(); return false; }
}
bool DeviceForward::gather_features(CaptureProducer& producer, const FeatureView& features,
        const OperationLease::Guard& lease, uint16_t* destination, int device, void* stream,
        uint64_t& local_bytes, uint64_t& peer_bytes, uint64_t& host_bytes, std::string& error,
        ForwardTraceSink* trace) {
    if (!producer.device_preflight(features, lease, error)) return false;
    const auto& topology = feature_peer_topology();
    if (!destination || topology.destination != device || !topology.count)
        return refuse(error, "device features lack their admitted physical topology");
    std::array<bool, tap_layers.size()> peers{};
    for (size_t tap = 0; tap < producer.device_taps_.size(); ++tap) {
        const int source = producer.device_taps_[tap].device_;
        const auto end = topology.sources.begin() + topology.count;
        const auto found = std::find(topology.sources.begin(), end, source);
        if (found == end) return refuse(error, "device tap source was not prepared before admission");
        peers[tap] = topology.peer[size_t(found - topology.sources.begin())];
    }
    bool submitted = false;
    Execution execution{stream, error, trace};
    try {
        DeviceScope on(device);
        if (!stream_preflight(stream, error)) return false;
        local_bytes = peer_bytes = host_bytes = 0;
        const size_t bytes = size_t(features.count) * reduced_tap_row_bytes;
        for (size_t tap = 0; tap < producer.device_taps_.size(); ++tap) {
            const auto& source = producer.device_taps_[tap];
            auto* output = destination + tap * hidden_size;
            submitted = true;
            if (source.device_ == device) {
                execution.must(cudaMemcpy2DAsync(output, feature_row_bytes, source.data_, reduced_tap_row_bytes,
                    reduced_tap_row_bytes, size_t(features.count), cudaMemcpyDeviceToDevice,
                    static_cast<cudaStream_t>(stream)), "gather accepted local verifier tap");
                local_bytes += bytes;
            } else if (peers[tap]) {
                cudaMemcpy3DPeerParms copy{};
                copy.srcDevice = source.device_; copy.dstDevice = device;
                copy.srcPtr = make_cudaPitchedPtr(const_cast<uint16_t*>(source.data_),
                    reduced_tap_row_bytes, reduced_tap_row_bytes, size_t(features.count));
                copy.dstPtr = make_cudaPitchedPtr(output, feature_row_bytes, feature_row_bytes, size_t(features.count));
                copy.extent = make_cudaExtent(reduced_tap_row_bytes, size_t(features.count), 1);
                execution.must(cudaMemcpy3DPeerAsync(&copy, static_cast<cudaStream_t>(stream)),
                               "gather accepted peer verifier tap");
                peer_bytes += bytes;
            } else {
                // This mirror was already required and host-complete at target
                // commit. No extra staging allocation or concealed GPU-only claim.
                execution.must(cudaMemcpy2DAsync(output, feature_row_bytes, source.mirror_, reduced_tap_row_bytes,
                    reduced_tap_row_bytes, size_t(features.count), cudaMemcpyHostToDevice,
                    static_cast<cudaStream_t>(stream)), "gather accepted compact host-staged verifier tap");
                host_bytes += bytes;
            }
            // Existing diagnostic seam can reject after real partial submission;
            // failure must drain before the verifier storage or lease can release.
            execution.emit("context.feature_gather.tap", int(tap), destination, ForwardTrace::Type::bf16,
                           {features.count, feature_width}, features.start);
        }
        return true;
    } catch (const SubmittedFailure&) {
        if (submitted) execution.drain_or_terminate(device);
    } catch (const std::exception& failure) {
        if (submitted) execution.drain_or_terminate(device);
        error = failure.what();
    } catch (...) {
        if (submitted) execution.drain_or_terminate(device);
        error = "dflash2 forward: unexpected accepted-feature transfer failure";
    }
    if (submitted) producer.abort();
    return false;
}
bool DeviceForward::reset(DeviceContext& context, int origin, std::string& error) const {
    OperationLease::Guard guard(impl_->lease);
    if (!impl_->acquire(guard, error)) return false;
    auto& c = *context.impl_;
    if (!impl_->owns(c, error)) { guard.complete(); return false; }
    if (origin < 0 || origin > max_position) { guard.complete(); return refuse(error, "invalid reset origin"); }
    c.origin = c.begin = c.end = origin; c.initialized = false; c.clear_owner();
    guard.complete(); error.clear(); return true;
}
bool DeviceForward::rewind(DeviceContext& context, int end, std::string& error) const {
    OperationLease::Guard guard(impl_->lease);
    if (!impl_->acquire(guard, error)) return false;
    auto& c = *context.impl_;
    if (!impl_->owns(c, error)) { guard.complete(); return false; }
    // A metadata-only rewind must NEVER pretend already evicted KV still exists.
    if (end < c.begin || end > c.end || std::max(c.origin, end - context_rows) != c.begin) {
        guard.complete(); return refuse(error, "rewind requires evicted KV; restore an owning snapshot instead");
    }
    c.end = end; c.clear_owner();
    guard.complete(); error.clear(); return true;
}
bool DeviceForward::copy_context(const DeviceContext& from, DeviceContext& to, void* stream, std::string& error) const {
    OperationLease::Guard guard(impl_->lease);
    if (!impl_->acquire(guard, error)) return false;
    auto& src = *from.impl_; auto& dst = *to.impl_;
    if (!impl_->owns(src, error) || !impl_->owns(dst, error) || (&src != &dst && dst.initialized && dst.origin != src.origin)) {
        guard.complete(); return refuse(error, "context copy identity/origin mismatch");
    }
    if (&src == &dst) { guard.complete(); error.clear(); return true; }
    bool submitted = false;
    Execution e{stream, error};
    try {
        DeviceScope on(impl_->device);
        if (!stream_preflight(stream, error)) { guard.complete(); return false; }
        const int count = src.end - src.begin, slot = src.begin % context_rows, first = std::min(count, context_rows - slot);
        const size_t pitch = size_t(context_rows) * row_bytes;
        submitted = true;
        if (count) {
            e.must(cudaMemcpy2DAsync(dst.ring() + size_t(slot) * gpu::head_dim, pitch,
                src.ring() + size_t(slot) * gpu::head_dim, pitch, size_t(first) * row_bytes,
                gpu::layers * 2 * gpu::kv_heads, cudaMemcpyDeviceToDevice, static_cast<cudaStream_t>(stream)), "copy retained context KV");
            if (first < count) e.must(cudaMemcpy2DAsync(dst.ring(), pitch, src.ring(), pitch, size_t(count - first) * row_bytes,
                gpu::layers * 2 * gpu::kv_heads, cudaMemcpyDeviceToDevice, static_cast<cudaStream_t>(stream)), "copy wrapped retained context KV");
        }
        e.must(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)), "complete context copy");
        dst.origin = src.origin; dst.begin = src.begin; dst.end = src.end; dst.initialized = src.initialized;
        dst.clear_owner(); guard.complete(); error.clear(); return true;
    } catch (const SubmittedFailure&) { e.drain_or_terminate(impl_->device); }
      catch (const std::exception& failure) {
        if (submitted) e.drain_or_terminate(impl_->device);
        error = failure.what();
      } catch (...) {
        if (submitted) e.drain_or_terminate(impl_->device);
        error = "dflash2 forward: unexpected context copy failure";
      }
    if (submitted) dst.poisoned = true; else guard.complete();
    return false;
}
bool DeviceForward::snapshot(const DeviceContext& context, DeviceContextSnapshot& snapshot,
                             void* stream, std::string& error) const {
    OperationLease::Guard guard(impl_->lease);
    if (!impl_->acquire(guard, error)) return false;
    auto& c = *context.impl_; auto& image = *snapshot.impl_;
    if (!impl_->owns(c, error) || !impl_->same_snapshot(image, error)) { guard.complete(); return false; }
    bool submitted = false;
    Execution e{stream, error};
    try {
        DeviceScope on(impl_->device);
        if (!stream_preflight(stream, error)) { guard.complete(); return false; }
        submitted = true; image.valid = false;
        impl_->image_copies(e, c.ring(), image.storage.get(), c.begin, c.end, cudaMemcpyDeviceToHost);
        e.must(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)), "complete context snapshot");
        image.metadata.origin = c.origin; image.metadata.begin = c.begin; image.metadata.end = c.end;
        image.metadata.initialized = c.initialized; image.valid = true;
        guard.complete(); error.clear(); return true;
    } catch (const SubmittedFailure&) { e.drain_or_terminate(impl_->device); }
      catch (const std::exception& failure) {
        if (submitted) e.drain_or_terminate(impl_->device);
        error = failure.what();
      } catch (...) {
        if (submitted) e.drain_or_terminate(impl_->device);
        error = "dflash2 forward: unexpected snapshot failure";
      }
    if (submitted) c.poisoned = true; else guard.complete();
    return false;
}
bool DeviceForward::validate_restore(const DeviceContext& context, const DeviceContextSnapshot& snapshot,
                                     std::string& error) const {
    const auto& c = *context.impl_; const auto& image = *snapshot.impl_;
    if (!impl_->owns(c, error) || !image.valid || !impl_->same_snapshot(image, error))
        return refuse(error, "foreign/invalid context snapshot");
    if (c.initialized && c.origin != image.metadata.origin)
        return refuse(error, "nonempty destination origin differs from snapshot");
    error.clear(); return true;
}
bool DeviceForward::restore(DeviceContext& context, const DeviceContextSnapshot& snapshot,
                            void* stream, std::string& error) const {
    OperationLease::Guard guard(impl_->lease);
    if (!impl_->acquire(guard, error)) return false;
    if (!validate_restore(context, snapshot, error)) { guard.complete(); return false; }
    auto& c = *context.impl_; const auto& image = *snapshot.impl_;
    bool submitted = false;
    Execution e{stream, error};
    try {
        DeviceScope on(impl_->device);
        if (!stream_preflight(stream, error)) { guard.complete(); return false; }
        submitted = true;
        impl_->image_copies(e, c.ring(), image.storage.get(), image.metadata.begin, image.metadata.end, cudaMemcpyHostToDevice);
        e.must(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)), "complete owning context restore");
        c.origin = image.metadata.origin; c.begin = image.metadata.begin; c.end = image.metadata.end;
        c.initialized = image.metadata.initialized; c.clear_owner();
        guard.complete(); error.clear(); return true;
    } catch (const SubmittedFailure&) { e.drain_or_terminate(impl_->device); }
      catch (const std::exception& failure) {
        if (submitted) e.drain_or_terminate(impl_->device);
        error = failure.what();
      } catch (...) {
        if (submitted) e.drain_or_terminate(impl_->device);
        error = "dflash2 forward: unexpected restore failure";
      }
    if (submitted) c.poisoned = true; else guard.complete();
    return false;
}
} // namespace strata::core::dflash2
