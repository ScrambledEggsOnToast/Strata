#include "strata/core/dflash2_target.hpp"

#include <cuda_runtime.h>

namespace strata::core::dflash2 {
namespace {
void cuda_require(cudaError_t status, const char* action) {
    if (status != cudaSuccess) throw Refusal(std::string(action) + ": " + cudaGetErrorString(status));
}
struct DeviceScope {
    int previous = -1;
    explicit DeviceScope(int device) {
        cuda_require(cudaGetDevice(&previous), "native target current device");
        cuda_require(cudaSetDevice(device), "native target owning device");
    }
    ~DeviceScope() { if (previous >= 0) cudaSetDevice(previous); }
};
} // namespace
struct NativeTargetDeviceIo::Impl {
    int embedding_device, head_device, previous = -1;
    cudaStream_t embedding_stream = nullptr, head_stream = nullptr;
    int32_t* tokens = nullptr;
    float *embedding = nullptr, *hidden = nullptr, *logits = nullptr;
    void* scratch = nullptr;
    size_t scratch_bytes;
    Impl(int embedding_device, int head_device, size_t scratch_bytes)
        : embedding_device(embedding_device), head_device(head_device), scratch_bytes(scratch_bytes) {}
    void restore() noexcept {
        if (previous >= 0) { cudaSetDevice(previous); previous = -1; }
    }
    void begin(int device) {
        if (previous >= 0) throw Refusal("native target device I/O is outstanding");
        cuda_require(cudaGetDevice(&previous), "native target current device");
        cuda_require(cudaSetDevice(device), "native target owning device");
    }
    void drain() noexcept {
        int saved = -1;
        cudaGetDevice(&saved);
        if (embedding_stream && cudaSetDevice(embedding_device) == cudaSuccess) cudaStreamSynchronize(embedding_stream);
        if (head_stream && cudaSetDevice(head_device) == cudaSuccess) cudaStreamSynchronize(head_stream);
        if (previous >= 0) restore(); else if (saved >= 0) cudaSetDevice(saved);
    }
    ~Impl() {
        drain();
        int saved = -1;
        cudaGetDevice(&saved);
        if (cudaSetDevice(embedding_device) == cudaSuccess) {
            if (embedding_stream) cudaStreamDestroy(embedding_stream);
            if (embedding) cudaFree(embedding);
            if (tokens) cudaFree(tokens);
        }
        if (cudaSetDevice(head_device) == cudaSuccess) {
            if (head_stream) cudaStreamDestroy(head_stream);
            if (hidden) cudaFree(hidden);
            if (logits) cudaFree(logits);
            if (scratch) cudaFree(scratch);
        }
        if (saved >= 0) cudaSetDevice(saved);
    }
};
NativeTargetDeviceIo::NativeTargetDeviceIo(int embedding_device, int head_device, size_t scratch_bytes) {
    if (embedding_device < 0 || head_device < 0 || scratch_bytes != size_t(hidden_size / 32 * 36))
        throw Refusal("native target device/scratch geometry mismatch");
    auto state = std::make_unique<Impl>(embedding_device, head_device, scratch_bytes);
    {
        DeviceScope device(embedding_device);
        cuda_require(cudaStreamCreateWithFlags(&state->embedding_stream, cudaStreamNonBlocking), "embedding stream");
        cuda_require(cudaMalloc(reinterpret_cast<void**>(&state->tokens), reference_block_size * sizeof(int32_t)), "embedding tokens");
        cuda_require(cudaMalloc(reinterpret_cast<void**>(&state->embedding), reference_block_size * hidden_size * sizeof(float)),
                     "embedding output");
    }
    {
        DeviceScope device(head_device);
        cuda_require(cudaStreamCreateWithFlags(&state->head_stream, cudaStreamNonBlocking), "head stream");
        cuda_require(cudaMalloc(reinterpret_cast<void**>(&state->hidden), hidden_size * sizeof(float)), "head input");
        cuda_require(cudaMalloc(reinterpret_cast<void**>(&state->logits), vocabulary_size * sizeof(float)), "head logits");
        cuda_require(cudaMalloc(&state->scratch, scratch_bytes), "dedicated head activation scratch");
    }
    impl_ = std::move(state);
}
NativeTargetDeviceIo::~NativeTargetDeviceIo() = default;
void NativeTargetDeviceIo::begin_embeddings(std::span<const int32_t> ids) {
    if (ids.size() != reference_block_size) throw Refusal("native target embedding input shape");
    impl_->begin(impl_->embedding_device);
    cuda_require(cudaMemcpyAsync(impl_->tokens, ids.data(), ids.size_bytes(), cudaMemcpyHostToDevice,
                                 impl_->embedding_stream), "embedding token upload");
}
void NativeTargetDeviceIo::finish_embeddings(std::span<float> out) {
    if (out.size() != reference_block_size * hidden_size) throw Refusal("native target embedding output shape");
    cuda_require(cudaPeekAtLastError(), "embedding gather launch");
    cuda_require(cudaMemcpyAsync(out.data(), impl_->embedding, out.size_bytes(), cudaMemcpyDeviceToHost,
                                 impl_->embedding_stream), "embedding readback");
    cuda_require(cudaStreamSynchronize(impl_->embedding_stream), "embedding completion");
    impl_->restore();
}
void NativeTargetDeviceIo::begin_head(std::span<const float> hidden) {
    if (hidden.size() != hidden_size) throw Refusal("native target head input shape");
    impl_->begin(impl_->head_device);
    cuda_require(cudaMemcpyAsync(impl_->hidden, hidden.data(), hidden.size_bytes(), cudaMemcpyHostToDevice,
                                 impl_->head_stream), "draft hidden upload");
}
void NativeTargetDeviceIo::finish_head(std::span<float> out) {
    if (out.size() != vocabulary_size) throw Refusal("native target head vocabulary shape");
    cuda_require(cudaMemcpyAsync(out.data(), impl_->logits, out.size_bytes(), cudaMemcpyDeviceToHost,
                                 impl_->head_stream), "native logits readback");
    cuda_require(cudaStreamSynchronize(impl_->head_stream), "native head completion");
    impl_->restore();
}
const int32_t* NativeTargetDeviceIo::tokens() const { return impl_->tokens; }
float* NativeTargetDeviceIo::embedding_rows() const { return impl_->embedding; }
float* NativeTargetDeviceIo::hidden_row() const { return impl_->hidden; }
float* NativeTargetDeviceIo::logits() const { return impl_->logits; }
void* NativeTargetDeviceIo::embedding_stream() const { return impl_->embedding_stream; }
void* NativeTargetDeviceIo::head_stream() const { return impl_->head_stream; }
void* NativeTargetDeviceIo::head_scratch() const { return impl_->scratch; }
size_t NativeTargetDeviceIo::head_scratch_bytes() const { return impl_->scratch_bytes; }
void NativeTargetDeviceIo::drain() noexcept { impl_->drain(); }

// Compile the production adapter against the real native interfaces. There is no
// serving activation or template instantiation that can select the CPU fixtures.
template class NativeTargetAdapter<>;
} // namespace strata::core::dflash2
