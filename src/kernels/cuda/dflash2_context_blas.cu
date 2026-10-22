#include "strata/kernels/dflash2_context_blas.hpp"
#if !defined(__linux__) || defined(STRATA_HIP_GFX906) || defined(__HIPCC__)
#include <cstdlib>
#include <cstring>
#include <stdexcept>
namespace strata::kernels::dflash2 {
bool context_blas_enabled() noexcept {
    static const bool enabled = [] { const char* v = std::getenv("STRATA_DFLASH2_CONTEXT_CUBLAS"); return v && std::strcmp(v, "1") == 0; }();
    return enabled;
}
const char* context_projection_identity() noexcept {
    return context_blas_enabled() ? "context-cublas-fp16-exact-fp32-n1024-m128-v1" : "legacy-bf16-v1";
}
struct ContextBlas::Impl {};
ContextBlas::ContextBlas(int) { throw std::runtime_error("dflash2 context cuBLAS requires Linux CUDA SM70 or SM86; this platform is unsupported"); }
ContextBlas::~ContextBlas() = default;
bool ContextBlas::project(const uint16_t*, const uint16_t*, uint16_t*, int, int, int, uint32_t*, void*, std::string& error) {
    error = "dflash2 context cuBLAS requires Linux CUDA SM70 or SM86"; return false;
}
ContextBlas::Statistics ContextBlas::statistics() const { return {}; }
void ContextBlas::report_completed() {}
} // namespace strata::kernels::dflash2
#else
#include "strata/kernels/bf16_bits.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <stdexcept>
#include <unistd.h>

namespace strata::kernels::dflash2 {
namespace {
constexpr int max_rows = 2047, max_input = 12800, max_output = 2560, chunk = 1024;
constexpr int16_t raw = -32768;
constexpr size_t workspace_bytes = size_t(4) << 20;
bool option(const char* name) { const char* v = std::getenv(name); return v && std::strcmp(v, "1") == 0; }
void checked(cudaError_t s, const char* op) {
    if (s != cudaSuccess) throw std::runtime_error(std::string("dflash2 context cuBLAS: ") + op + ": " + cudaGetErrorString(s));
}
void checked_blas(cublasStatus_t s, const char* op) {
    if (s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string("dflash2 context cuBLAS: ") + op + " status=" + std::to_string(int(s)));
}
size_t resident_bytes() {
    std::ifstream file("/proc/self/statm"); size_t total = 0, resident = 0;
    const long page = sysconf(_SC_PAGESIZE);
    if (!(file >> total >> resident) || page <= 0) throw std::runtime_error("dflash2 context cuBLAS: cannot sample host RSS");
    return resident * size_t(page);
}
struct DeviceScope {
    int previous = -1;
    explicit DeviceScope(int device) {
        checked(cudaGetDevice(&previous), "current device");
        if (previous != device) checked(cudaSetDevice(device), "select device");
    }
    ~DeviceScope() { if (previous >= 0 && cudaSetDevice(previous) != cudaSuccess) std::terminate(); }
};
struct Arena {
    unsigned char* base; size_t used = 0;
    template<class T> T* take(size_t count) {
        auto* p = base ? reinterpret_cast<T*>(base + used) : nullptr;
        used += (count * sizeof(T) + 255) / 256 * 256; return p;
    }
};
struct Scratch {
    uint16_t *activation, *weight; float* result; void* workspace;
    int16_t *activation_scale, *weight_scale; ContextBlas::Statistics* counts;
    size_t bytes;
};
Scratch layout(void* pointer) {
    Arena arena{static_cast<unsigned char*>(pointer)}; Scratch s{};
    s.activation = arena.take<uint16_t>(size_t(max_rows) * max_input);
    s.weight = arena.take<uint16_t>(size_t(chunk) * max_input);
    s.result = arena.take<float>(size_t(max_rows) * chunk);
    s.workspace = arena.take<unsigned char>(workspace_bytes);
    s.activation_scale = arena.take<int16_t>(max_rows);
    s.weight_scale = arena.take<int16_t>(chunk);
    s.counts = arena.take<ContextBlas::Statistics>(1); s.bytes = arena.used; return s;
}
__device__ float power(int e) { return __uint_as_float(unsigned(e + 127) << 23); }
__device__ unsigned max_warp(unsigned v) {
    for (int d = 16; d; d >>= 1) v = max(v, __shfl_down_sync(0xffffffffu, v, d)); return v;
}
__device__ bool lossless(uint16_t bits, int exponent) {
    const float original = f32_from_bf16(bits), scaled = __fmul_rn(original, power(exponent));
    const __half half = __float2half_rn(scaled); const unsigned magnitude = __half_as_ushort(half) & 0x7fffU;
    return (!magnitude || (magnitude >= 0x400U && magnitude < 0x7c00U)) &&
        __float_as_uint(__half2float(half)) == __float_as_uint(scaled) &&
        __float_as_uint(__fmul_rn(__half2float(half), power(-exponent))) == __float_as_uint(original);
}
// Rejected rows become HALF zero, never raw BF16 masquerading as half. The
// originals remain unchanged for the per-output exact fallback below.
__global__ void prepare(const uint16_t* original, uint16_t* half, int16_t* scale,
                        int width, bool weights, ContextBlas::Statistics* counts) {
    __shared__ unsigned maxima[8];
    const int row = blockIdx.x, tid = threadIdx.x;
    unsigned top = 0;
    for (int k = tid; k < width; k += 256) top = max(top, unsigned(original[size_t(row) * width + k] & 0x7fffU));
    top = max_warp(top); if (!(tid & 31)) maxima[tid >> 5] = top;
    __syncthreads(); top = 0; for (int i = 0; i < 8; ++i) top = max(top, maxima[i]);
    const int exponent = top ? 14 - (int(top >> 7) - 127) : 0;
    bool bad = top >= 0x7f80U || exponent < -60 || exponent > 60;
    if (!bad) for (int k = tid; k < width; k += 256) bad |= !lossless(original[size_t(row) * width + k], exponent);
    const bool rejected = __syncthreads_or(bad);
    if (!tid) {
        scale[row] = rejected ? raw : int16_t(exponent);
        if (counts) {
            atomicAdd(weights ? &counts->weight_rows : &counts->activation_rows, 1ULL);
            if (rejected) atomicAdd(weights ? &counts->raw_weight_rows : &counts->raw_activation_rows, 1ULL);
            if (!weights && !row) atomicAdd(&counts->projection_calls, 1ULL);
        }
    }
    for (int k = tid; k < width; k += 256) half[size_t(row) * width + k] = rejected ? 0 :
        __half_as_ushort(__float2half_rn(__fmul_rn(f32_from_bf16(original[size_t(row) * width + k]), power(exponent))));
}
__global__ void epilogue(const uint16_t* input, const uint16_t* weight, const float* result,
                         const int16_t* xs, const int16_t* ws, uint16_t* output,
                         int rows, int ni, int no, int offset, int width,
                         uint32_t* status, ContextBlas::Statistics* counts) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(rows) * width) return;
    const int r = int(i / width), o = int(i % width);
    bool fallback = xs[r] == raw || ws[o] == raw;
    float value = 0.0f;
    if (!fallback) {
        value = __fmul_rn(result[i], power(-int(xs[r]) - int(ws[o])));
        fallback = !isfinite(value) || (bf16_from_f32(value) & 0x7f80U) == 0x7f80U;
    }
    if (fallback) {
        value = 0.0f;
        // Same increasing-K FP32 FMA and zero padding as both context kernels.
        for (int k = 0; k < ((ni + 31) / 32) * 32; ++k) {
            const float x = k < ni ? f32_from_bf16(input[size_t(r) * ni + k]) : 0.0f;
            const float w = k < ni ? f32_from_bf16(weight[size_t(offset + o) * ni + k]) : 0.0f;
            value = fmaf(x, w, value);
        }
        if (counts) atomicAdd(&counts->fallback_outputs, 1ULL);
    }
    const uint16_t bits = bf16_from_f32(value);
    if ((bits & 0x7f80U) == 0x7f80U) atomicOr(status, 1U);
    output[size_t(r) * no + offset + o] = bits;
}
#ifdef STRATA_DFLASH2_CONTEXT_BLAS_TEST
int construction_failure_stage = 0; // Compiled only into the standalone fixture.
#endif
} // namespace

bool context_blas_enabled() noexcept { static const bool enabled = option("STRATA_DFLASH2_CONTEXT_CUBLAS"); return enabled; }
const char* context_projection_identity() noexcept {
    return context_blas_enabled() ? "context-cublas-fp16-exact-fp32-n1024-m128-v1" : "legacy-bf16-v1";
}

struct ContextBlas::Impl {
    int device;
    void* allocation = nullptr;
    Scratch s{};
    cublasHandle_t handle = nullptr;
    cudaStream_t initialization_stream = nullptr;
    bool diagnostic = false, reported = false;
    explicit Impl(int ordinal) : device(ordinal) {}
    ~Impl() {
        int previous = -1;
        if (cudaGetDevice(&previous) != cudaSuccess || cudaSetDevice(device) != cudaSuccess) std::terminate();
        // The owning forward drains caller streams. Only our initialization
        // stream can remain pending on a constructor failure.
        if (initialization_stream && cudaStreamSynchronize(initialization_stream) != cudaSuccess) std::terminate();
        if (handle && cublasDestroy(handle) != CUBLAS_STATUS_SUCCESS) std::terminate();
        if (allocation && cudaFree(allocation) != cudaSuccess) std::terminate();
        if (initialization_stream && cudaStreamDestroy(initialization_stream) != cudaSuccess) std::terminate();
        if (cudaSetDevice(previous) != cudaSuccess) std::terminate();
    }
    void bind(cudaStream_t stream) {
        checked_blas(cublasSetStream(handle, stream), "bind stream");
        checked_blas(cublasSetWorkspace(handle, s.workspace, workspace_bytes), "bind bounded workspace");
    }
    void gemm(int rows, int ni, int width) {
        const float alpha = 1.0f, beta = 0.0f;
        checked_blas(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, width, rows, ni,
            &alpha, s.weight, CUDA_R_16F, ni, s.activation, CUDA_R_16F, ni,
            &beta, s.result, CUDA_R_32F, width, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT), "GemmEx");
    }
    void initialize() {
        DeviceScope on(device);
        cudaDeviceProp properties{}; checked(cudaGetDeviceProperties(&properties, device), "device capability");
        if (!((properties.major == 7 && properties.minor == 0) || (properties.major == 8 && properties.minor == 6)))
            throw std::runtime_error("dflash2 context cuBLAS: only SM70 or SM86 are eligible for this opt-in mode");
        const size_t rss_before = resident_bytes(); size_t free_before, total;
        checked(cudaMemGetInfo(&free_before, &total), "initial free memory");
        if (layout(nullptr).bytes != explicit_device_bytes) throw std::runtime_error("dflash2 context cuBLAS: workspace accounting mismatch");
        checked(cudaMalloc(&allocation, explicit_device_bytes), "allocate bounded scratch"); s = layout(allocation);
        checked(cudaStreamCreateWithFlags(&initialization_stream, cudaStreamNonBlocking), "initialization stream");
        checked_blas(cublasCreate(&handle), "create handle");
#ifdef STRATA_DFLASH2_CONTEXT_BLAS_TEST
        if (construction_failure_stage == 1) throw std::runtime_error("injected context cuBLAS construction failure");
#endif
        bind(initialization_stream);
        checked_blas(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST), "host scalars");
        checked_blas(cublasSetMathMode(handle, static_cast<cublasMath_t>(
            CUBLAS_DEFAULT_MATH | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)), "FP32 reductions");
        checked_blas(cublasSetAtomicsMode(handle, CUBLAS_ATOMICS_NOT_ALLOWED), "deterministic library policy");
        checked(cudaMemsetAsync(s.activation, 0, size_t(max_rows) * max_input * 2, initialization_stream), "warmup activation");
        checked(cudaMemsetAsync(s.weight, 0, size_t(chunk) * max_input * 2, initialization_stream), "warmup weights");
        // Exercise actual largest/smallest context and FC/KV/tail GEMM dimensions
        // using exclusively owned zero scratch before publishing this owner.
        for (int rows : {128, max_rows}) for (int ni : {2560, max_input}) for (int width : {512, chunk}) gemm(rows, ni, width);
        checked(cudaMemsetAsync(s.counts, 0, sizeof(Statistics), initialization_stream), "reset diagnostics");
        checked(cudaStreamSynchronize(initialization_stream), "complete bounded initialization");
#ifdef STRATA_DFLASH2_CONTEXT_BLAS_TEST
        if (construction_failure_stage == 2) throw std::runtime_error("injected context cuBLAS warmup failure");
#endif
        size_t free_after; checked(cudaMemGetInfo(&free_after, &total), "initialized free memory");
        const size_t rss_after = resident_bytes();
        const size_t device_growth = free_before > free_after ? free_before - free_after : 0;
        const size_t host_growth = rss_after > rss_before ? rss_after - rss_before : 0;
        std::fprintf(stderr, "dflash2 context cuBLAS admission device_budget=%zu explicit=%zu sampled_device_growth=%zu host_budget=%zu sampled_rss_growth=%zu endpoint_samples_only=1\n",
            device_budget, explicit_device_bytes, device_growth, host_budget, host_growth);
        if (device_growth > device_budget || host_growth > host_budget)
            throw std::runtime_error("dflash2 context cuBLAS: observed construction growth exceeds admitted allowance");
        diagnostic = option("STRATA_DFLASH2_CONTEXT_CUBLAS_TRACE");
    }
};

ContextBlas::ContextBlas(int device) {
    if (device < 0) throw std::invalid_argument("dflash2 context cuBLAS: invalid device");
    auto state = std::make_unique<Impl>(device); state->initialize(); impl_ = std::move(state);
}
ContextBlas::~ContextBlas() = default;
bool ContextBlas::project(const uint16_t* input, const uint16_t* weight, uint16_t* output,
                         int rows, int ni, int no, uint32_t* status, void* stream, std::string& error) {
    if (!input || !weight || !output || !status || rows < 128 || rows > max_rows || ni < 1 || ni > max_input || no < 1 || no > max_output) {
        error = "dflash2 context cuBLAS: invalid projection extent"; return false;
    }
    try {
        DeviceScope on(impl_->device); auto cs = static_cast<cudaStream_t>(stream);
        impl_->bind(cs); auto& s = impl_->s; auto* counts = impl_->diagnostic ? s.counts : nullptr;
        prepare<<<rows, 256, 0, cs>>>(input, s.activation, s.activation_scale, ni, false, counts);
        checked(cudaGetLastError(), "prepare activation rows");
        for (int offset = 0; offset < no; offset += chunk) {
            const int width = std::min(chunk, no - offset);
            prepare<<<width, 256, 0, cs>>>(weight + size_t(offset) * ni, s.weight, s.weight_scale, ni, true, counts);
            checked(cudaGetLastError(), "prepare weight chunk"); impl_->gemm(rows, ni, width);
            epilogue<<<unsigned((size_t(rows) * width + 255) / 256), 256, 0, cs>>>(input, weight, s.result,
                s.activation_scale, s.weight_scale, output, rows, ni, no, offset, width, status, counts);
            checked(cudaGetLastError(), "publish context projection");
        }
        error.clear(); return true;
    } catch (const std::exception& failure) { error = failure.what(); return false; }
}
ContextBlas::Statistics ContextBlas::statistics() const {
    DeviceScope on(impl_->device); Statistics result{};
    checked(cudaMemcpy(&result, impl_->s.counts, sizeof(result), cudaMemcpyDeviceToHost), "completed diagnostic readback"); return result;
}
void ContextBlas::report_completed() {
    if (!impl_->diagnostic || impl_->reported) return;
    const auto s = statistics(); if (!s.projection_calls) return;
    std::fprintf(stderr, "dflash2 context cuBLAS completed calls=%llu activation_rows=%llu raw_activation_rows=%llu weight_rows=%llu raw_weight_rows=%llu fallback_outputs=%llu\n",
        s.projection_calls, s.activation_rows, s.raw_activation_rows, s.weight_rows, s.raw_weight_rows, s.fallback_outputs);
    impl_->reported = true;
}
} // namespace strata::kernels::dflash2

#endif // Linux CUDA implementation / explicit unsupported platform stub
