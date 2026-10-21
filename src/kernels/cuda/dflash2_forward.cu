#include "strata/kernels/dflash2_forward.hpp"
#include "strata/kernels/bf16_bits.hpp"

#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>

namespace strata::kernels::dflash2 {
namespace {
constexpr int threads = 256;
constexpr float attention_scale_root = 0.29730177875068026668f; // sqrt(128**-0.5)

__device__ inline void nonfinite(uint32_t* status) {
    if (status) atomicOr(status, 1u);
}
__device__ inline uint16_t rounded(float value, uint32_t* status) {
    const uint16_t bits = bf16_from_f32(value);
    if ((bits & 0x7f80u) == 0x7f80u) nonfinite(status);
    return bits;
}
__device__ inline float warp_sum(float value) {
    for (int offset = 16; offset; offset >>= 1)
        value = __fadd_rn(value, __shfl_down_sync(0xffffffffu, value, offset));
    return value;
}
__device__ inline float block_sum(float value, float* shared) {
    value = warp_sum(value);
    if ((threadIdx.x & 31) == 0) shared[threadIdx.x >> 5] = value;
    __syncthreads();
    value = threadIdx.x < 8 ? shared[threadIdx.x] : 0.0f;
    if (threadIdx.x < 32) value = warp_sum(value);
    if (threadIdx.x == 0) shared[0] = value;
    __syncthreads();
    return shared[0];
}

// Same BF16 input / FP32 FMA and shfl_down reduction as bf16_gemv.cu,
// with an explicit BF16 output and recoverable launch error for the owner.
__global__ void linear_small(const uint16_t* input, const uint16_t* weight,
                              uint16_t* output, int ni, int no, uint32_t* status) {
    const int o = int(blockIdx.x) * 8 + int(threadIdx.x >> 5);
    if (o >= no) return;
    const int row = int(blockIdx.y), lane = int(threadIdx.x & 31);
    float sum = 0.0f;
    for (int i = lane; i < ni; i += 32)
        sum = fmaf(f32_from_bf16(input[size_t(row) * ni + i]),
                   f32_from_bf16(weight[size_t(o) * ni + i]), sum);
    sum = warp_sum(sum);
    if (lane == 0) output[size_t(row) * no + o] = rounded(sum, status);
}

// Large context projection: each BF16 weight/input tile is widened once and
// reused over 16 rows/outputs. FP32 scalar arithmetic works unchanged on SM70.
__global__ void linear_context(const uint16_t* input, const uint16_t* weight,
                                uint16_t* output, int rows, int ni, int no, uint32_t* status) {
    __shared__ float x[16][33];
    __shared__ float w[16][33];
    const int tx = int(threadIdx.x), ty = int(threadIdx.y), linear = ty * 16 + tx;
    const int row = int(blockIdx.y) * 16 + ty;
    const int o = int(blockIdx.x) * 16 + tx;
    float sum = 0.0f;
    for (int start = 0; start < ni; start += 32) {
        for (int t = linear; t < 16 * 32; t += 256) {
            const int a = t / 32, k = t % 32;
            const int xr = int(blockIdx.y) * 16 + a;
            const int wo = int(blockIdx.x) * 16 + a;
            x[a][k] = xr < rows && start + k < ni
                ? f32_from_bf16(input[size_t(xr) * ni + start + k]) : 0.0f;
            w[a][k] = wo < no && start + k < ni
                ? f32_from_bf16(weight[size_t(wo) * ni + start + k]) : 0.0f;
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < 32; ++k) sum = fmaf(x[ty][k], w[tx][k], sum);
        __syncthreads();
    }
    if (row < rows && o < no) output[size_t(row) * no + o] = rounded(sum, status);
}

__global__ void finite_kernel(const uint16_t* input, size_t count, uint32_t* status) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count && (input[i] & 0x7f80u) == 0x7f80u) nonfinite(status);
}
__global__ void normalize_kernel(const uint16_t* input, uint16_t* output,
                                  float* statistics, int width, uint32_t* status) {
    __shared__ float shared[8];
    const size_t base = size_t(blockIdx.x) * width;
    float squares = 0.0f;
    for (int d = int(threadIdx.x); d < width; d += threads) {
        const float x = f32_from_bf16(input[base + d]);
        squares = __fadd_rn(squares, __fmul_rn(x, x));
    }
    const float mean = __fdiv_rn(block_sum(squares, shared), float(width));
    const float inverse = rsqrtf(__fadd_rn(mean, 1e-6f));
    if (!isfinite(mean) || !isfinite(inverse)) nonfinite(status);
    if (threadIdx.x == 0 && statistics) {
        statistics[size_t(blockIdx.x) * 2] = mean;
        statistics[size_t(blockIdx.x) * 2 + 1] = inverse;
    }
    // Every input used by the reduction has been read before this write, so
    // input==output is valid. Learned weight multiplication is a later launch.
    for (int d = int(threadIdx.x); d < width; d += threads)
        output[base + d] = rounded(__fmul_rn(f32_from_bf16(input[base + d]), inverse), status);
}
__global__ void widen_kernel(const uint16_t* input, float* output, size_t count) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) output[i] = f32_from_bf16(input[i]);
}
__global__ void norm_weight_kernel(uint16_t* input, const uint16_t* weight,
                                    int count, int width, uint32_t* status) {
    const int i = int(blockIdx.x) * threads + int(threadIdx.x);
    if (i < count) input[i] = rounded(__fmul_rn(f32_from_bf16(input[i]),
                                                  f32_from_bf16(weight[i % width])), status);
}
template<bool multiply>
__global__ void binary_kernel(const uint16_t* left, const uint16_t* right,
                               uint16_t* output, int count, uint32_t* status) {
    const int i = int(blockIdx.x) * threads + int(threadIdx.x);
    if (i >= count) return;
    const float a = f32_from_bf16(left[i]), b = f32_from_bf16(right[i]);
    output[i] = rounded(multiply ? __fmul_rn(a, b) : __fadd_rn(a, b), status);
}
__global__ void silu_kernel(uint16_t* values, int count, uint32_t* status) {
    const int i = int(blockIdx.x) * threads + int(threadIdx.x);
    if (i < count) {
        const float x = f32_from_bf16(values[i]);
        values[i] = rounded(__fdiv_rn(x, __fadd_rn(1.0f, expf(-x))), status);
    }
}
__global__ void conv_product_kernel(const uint16_t* input, const uint16_t* base,
                                     uint16_t* output, int stage, int offset, uint32_t* status) {
    const int i = int(blockIdx.x) * threads + int(threadIdx.x);
    if (i >= block * hidden) return;
    const int row = i / hidden, d = i % hidden;
    const float x = row >= offset ? f32_from_bf16(input[(row - offset) * hidden + d]) : 0.0f;
    output[i] = rounded(__fmul_rn(f32_from_bf16(base[(stage * 2 + offset) * hidden + d]), x), status);
}
__global__ void conv_addcmul_kernel(uint16_t* accumulator, const uint16_t* dynamic,
                                     const uint16_t* input, int stage, int offset, uint32_t* status) {
    const int i = int(blockIdx.x) * threads + int(threadIdx.x);
    if (i >= block * hidden) return;
    const int row = i / hidden, d = i % hidden;
    const float x = row >= offset ? f32_from_bf16(input[(row - offset) * hidden + d]) : 0.0f;
    const float coefficient = f32_from_bf16(dynamic[row * dynamic_width + (stage * 2 + offset) * groups + d / 16]);
    accumulator[i] = rounded(fmaf(coefficient, x, f32_from_bf16(accumulator[i])), status);
}

__global__ void inverse_kernel(float* inverse) {
    const int d = int(threadIdx.x);
    if (d < 64) inverse[d] = __fdiv_rn(1.0f, powf(10000.0f, __fdiv_rn(float(2 * d), 128.0f)));
}
__global__ void rotary_table_kernel(const float* inverse, uint16_t* cosine, uint16_t* sine,
                                     int first, int rows, uint32_t* status) {
    const int i = int(blockIdx.x) * threads + int(threadIdx.x);
    if (i >= rows * 64) return;
    const float angle = __fmul_rn(float(first + i / 64), inverse[i % 64]);
    cosine[i] = rounded(cosf(angle), status);
    sine[i] = rounded(sinf(angle), status);
}
__global__ void rotary_kernel(uint16_t* input, const uint16_t* cosine, const uint16_t* sine,
                               int rows, int heads, uint32_t* status) {
    const int i = int(blockIdx.x) * threads + int(threadIdx.x);
    if (i >= rows * heads * 64) return;
    const int pair = i % 64, row = i / (heads * 64);
    const int base = (i / 64) * 128 + pair;
    const float a = f32_from_bf16(input[base]), b = f32_from_bf16(input[base + 64]);
    const float c = f32_from_bf16(cosine[row * 64 + pair]);
    const float s = f32_from_bf16(sine[row * 64 + pair]);
    // Torch's two multiply tensors round BEFORE the out-of-place addition.
    const float ac = f32_from_bf16(rounded(__fmul_rn(a, c), status));
    const float bs = f32_from_bf16(rounded(__fmul_rn(-b, s), status));
    const float bc = f32_from_bf16(rounded(__fmul_rn(b, c), status));
    const float as = f32_from_bf16(rounded(__fmul_rn(a, s), status));
    input[base] = rounded(__fadd_rn(ac, bs), status);
    input[base + 64] = rounded(__fadd_rn(bc, as), status);
}
__device__ inline size_t ring_index(int layer, int kind, int head, int position, int d) {
    return (((size_t(layer) * 2 + kind) * kv_heads + head) * capacity + position % capacity) * head_dim + d;
}
__global__ void store_kernel(const uint16_t* keys, const uint16_t* values, uint16_t* ring,
                              int layer, int first, int rows, uint32_t* status) {
    const int i = int(blockIdx.x) * threads + int(threadIdx.x);
    if (i >= rows * kv_width) return;
    const int row = i / kv_width, head = i % kv_width / head_dim, d = i % head_dim;
    const uint16_t k = keys[i], v = values[i];
    if ((k & 0x7f80u) == 0x7f80u || (v & 0x7f80u) == 0x7f80u) nonfinite(status);
    ring[ring_index(layer, 0, head, first + row, d)] = k;
    ring[ring_index(layer, 1, head, first + row, d)] = v;
}
__global__ void score_kernel(const uint16_t* query, const uint16_t* block_keys,
                              const uint16_t* ring, float* scores, int layer,
                              int begin, int frontier, uint32_t* status) {
    const int key = int(blockIdx.x) * 8 + int(threadIdx.x >> 5);
    const int keys = frontier - begin + block;
    if (key >= keys) return;
    const int hq = int(blockIdx.y) / block, row = int(blockIdx.y) % block;
    const int hk = hq / (query_heads / kv_heads), lane = int(threadIdx.x & 31);
    const int context = frontier - begin, position = key < context ? begin + key : frontier + key - context;
    const int distance = frontier + row - position;
    const size_t out = (size_t(hq) * block + row) * keys + key;
    if (distance <= -2048 || distance >= 2048) {
        if (lane == 0) scores[out] = -CUDART_INF_F;
        return;
    }
    float sum = 0.0f;
    for (int d = lane; d < head_dim; d += 32) {
        const float q = __fmul_rn(f32_from_bf16(query[(row * query_heads + hq) * head_dim + d]), attention_scale_root);
        const uint16_t k = key < context
            ? ring[ring_index(layer, 0, hk, position, d)]
            : block_keys[((key - context) * kv_heads + hk) * head_dim + d];
        sum = fmaf(q, __fmul_rn(f32_from_bf16(k), attention_scale_root), sum);
    }
    sum = warp_sum(sum);
    if (lane == 0) {
        if (!isfinite(sum)) nonfinite(status);
        scores[out] = sum;
    }
}
__global__ void softmax_kernel(const float* scores, float* probabilities, float* statistics,
                                int keys, uint32_t* status) {
    __shared__ float shared[8];
    const size_t base = size_t(blockIdx.x) * keys;
    float maximum = -CUDART_INF_F;
    for (int k = int(threadIdx.x); k < keys; k += threads) maximum = fmaxf(maximum, scores[base + k]);
    for (int offset = 16; offset; offset >>= 1)
        maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffu, maximum, offset));
    if ((threadIdx.x & 31) == 0) shared[threadIdx.x >> 5] = maximum;
    __syncthreads();
    maximum = threadIdx.x < 8 ? shared[threadIdx.x] : -CUDART_INF_F;
    if (threadIdx.x < 32)
        for (int offset = 16; offset; offset >>= 1)
            maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffu, maximum, offset));
    if (threadIdx.x == 0) shared[0] = maximum;
    __syncthreads();
    maximum = shared[0];
    // Shared reduction storage may not be reused until EVERY warp has read
    // the row maximum (independent thread scheduling on both supported GPUs).
    __syncthreads();
    if (!isfinite(maximum)) nonfinite(status);
    float sum = 0.0f;
    for (int k = int(threadIdx.x); k < keys; k += threads) {
        const float p = expf(__fadd_rn(scores[base + k], -maximum));
        probabilities[base + k] = p;
        sum = __fadd_rn(sum, p);
    }
    // The maximum's read barrier above permits block_sum to reuse shared.
    sum = block_sum(sum, shared);
    if (!(sum > 0.0f) || !isfinite(sum)) nonfinite(status);
    if (threadIdx.x == 0 && statistics) {
        statistics[size_t(blockIdx.x) * 2] = maximum;
        statistics[size_t(blockIdx.x) * 2 + 1] = sum;
    }
    const float inverse = __fdiv_rn(1.0f, sum);
    for (int k = int(threadIdx.x); k < keys; k += threads)
        probabilities[base + k] = __fmul_rn(probabilities[base + k], inverse);
}
__global__ void value_kernel(const float* probabilities, const uint16_t* block_values,
                              const uint16_t* ring, uint16_t* output, int layer,
                              int begin, int frontier, uint32_t* status) {
    const int hq = int(blockIdx.x) / block, row = int(blockIdx.x) % block;
    const int hk = hq / (query_heads / kv_heads), d = int(threadIdx.x);
    const int context = frontier - begin, keys = context + block;
    const size_t base = (size_t(hq) * block + row) * keys;
    float sum = 0.0f;
    for (int k = 0; k < keys; ++k) {
        const uint16_t v = k < context
            ? ring[ring_index(layer, 1, hk, begin + k, d)]
            : block_values[((k - context) * kv_heads + hk) * head_dim + d];
        sum = fmaf(probabilities[base + k], f32_from_bf16(v), sum);
    }
    output[(row * query_heads + hq) * head_dim + d] = rounded(sum, status);
}
inline int launched() { return int(cudaGetLastError()); }
inline unsigned blocks(size_t count) { return unsigned((count + threads - 1) / threads); }
} // namespace

int linear_bf16(const uint16_t* input, const uint16_t* weight, uint16_t* output,
                int rows, int ni, int no, void* stream, uint32_t* status) {
    if (!input || !weight || !output || rows < 1 || rows > capacity || ni < 1 || no < 1)
        return int(cudaErrorInvalidValue);
    if (rows <= block)
        linear_small<<<dim3(unsigned((no + 7) / 8), unsigned(rows)), threads, 0, (cudaStream_t)stream>>>(input, weight, output, ni, no, status);
    else
        linear_context<<<dim3(unsigned((no + 15) / 16), unsigned((rows + 15) / 16)), dim3(16, 16), 0, (cudaStream_t)stream>>>(input, weight, output, rows, ni, no, status);
    return launched();
}
int finite_bf16(const uint16_t* input, size_t count, uint32_t* status, void* stream) {
    if (!input || !count || !status) return int(cudaErrorInvalidValue);
    finite_kernel<<<blocks(count), threads, 0, (cudaStream_t)stream>>>(input, count, status);
    return launched();
}
int widen_bf16(const uint16_t* input, float* output, size_t count, void* stream) {
    if (!input || !output || !count || count > 0x7fffffffu) return int(cudaErrorInvalidValue);
    widen_kernel<<<blocks(count), threads, 0, (cudaStream_t)stream>>>(input, output, count);
    return launched();
}
int normalize_bf16(const uint16_t* input, uint16_t* output, float* statistics,
                   int rows, int width, uint32_t* status, void* stream) {
    if (!input || !output || rows < 1 || width < 1) return int(cudaErrorInvalidValue);
    normalize_kernel<<<unsigned(rows), threads, 0, (cudaStream_t)stream>>>(input, output, statistics, width, status);
    return launched();
}
int multiply_norm_bf16(uint16_t* input, const uint16_t* weight, int count,
                       int width, uint32_t* status, void* stream) {
    if (!input || !weight || count < 1 || width < 1) return int(cudaErrorInvalidValue);
    norm_weight_kernel<<<blocks(count), threads, 0, (cudaStream_t)stream>>>(input, weight, count, width, status);
    return launched();
}
int add_bf16(const uint16_t* left, const uint16_t* right, uint16_t* output,
             int count, uint32_t* status, void* stream) {
    if (!left || !right || !output || count < 1) return int(cudaErrorInvalidValue);
    binary_kernel<false><<<blocks(count), threads, 0, (cudaStream_t)stream>>>(left, right, output, count, status);
    return launched();
}
int multiply_bf16(const uint16_t* left, const uint16_t* right, uint16_t* output,
                  int count, uint32_t* status, void* stream) {
    if (!left || !right || !output || count < 1) return int(cudaErrorInvalidValue);
    binary_kernel<true><<<blocks(count), threads, 0, (cudaStream_t)stream>>>(left, right, output, count, status);
    return launched();
}
int silu_bf16(uint16_t* values, int count, uint32_t* status, void* stream) {
    if (!values || count < 1) return int(cudaErrorInvalidValue);
    silu_kernel<<<blocks(count), threads, 0, (cudaStream_t)stream>>>(values, count, status);
    return launched();
}
int convolution_product(const uint16_t* input, const uint16_t* base, uint16_t* product,
                         int stage, int offset, uint32_t* status, void* stream) {
    if (!input || !base || !product || stage < 0 || stage > 1 || offset < 0 || offset > 1)
        return int(cudaErrorInvalidValue);
    conv_product_kernel<<<blocks(block * hidden), threads, 0, (cudaStream_t)stream>>>(input, base, product, stage, offset, status);
    return launched();
}
int convolution_addcmul(uint16_t* accumulator, const uint16_t* dynamic, const uint16_t* input,
                        int stage, int offset, uint32_t* status, void* stream) {
    if (!accumulator || !dynamic || !input || stage < 0 || stage > 1 || offset < 0 || offset > 1)
        return int(cudaErrorInvalidValue);
    conv_addcmul_kernel<<<blocks(block * hidden), threads, 0, (cudaStream_t)stream>>>(accumulator, dynamic, input, stage, offset, status);
    return launched();
}
int inverse_frequencies(float* output, void* stream) {
    if (!output) return int(cudaErrorInvalidValue);
    inverse_kernel<<<1, 64, 0, (cudaStream_t)stream>>>(output);
    return launched();
}
int rotary_tables(const float* inverse, uint16_t* cosine, uint16_t* sine,
                   int first, int rows, uint32_t* status, void* stream) {
    if (!inverse || !cosine || !sine || first < 0 || rows < 1 || rows > max_keys || first > 262144 - rows)
        return int(cudaErrorInvalidValue);
    rotary_table_kernel<<<blocks(rows * 64), threads, 0, (cudaStream_t)stream>>>(inverse, cosine, sine, first, rows, status);
    return launched();
}
int rotary_bf16(uint16_t* input, const uint16_t* cosine, const uint16_t* sine,
                int rows, int heads, uint32_t* status, void* stream) {
    if (!input || !cosine || !sine || rows < 1 || heads < 1) return int(cudaErrorInvalidValue);
    rotary_kernel<<<blocks(size_t(rows) * heads * 64), threads, 0, (cudaStream_t)stream>>>(input, cosine, sine, rows, heads, status);
    return launched();
}
int store_context(const uint16_t* keys, const uint16_t* values, uint16_t* ring,
                   int layer, int first, int rows, uint32_t* status, void* stream) {
    if (!keys || !values || !ring || layer < 0 || layer >= layers || first < 0 ||
        rows < 1 || rows > capacity || first > 262144 - rows) return int(cudaErrorInvalidValue);
    store_kernel<<<blocks(size_t(rows) * kv_width), threads, 0, (cudaStream_t)stream>>>(keys, values, ring, layer, first, rows, status);
    return launched();
}
int attention_scores(const uint16_t* query, const uint16_t* block_keys, const uint16_t* ring,
                      float* scores, int layer, int begin, int frontier, uint32_t* status, void* stream) {
    if (!query || !block_keys || !ring || !scores || layer < 0 || layer >= layers ||
        begin < 0 || frontier < begin || frontier - begin > capacity || frontier > 262144 - block)
        return int(cudaErrorInvalidValue);
    score_kernel<<<dim3(unsigned((frontier - begin + block + 7) / 8), query_heads * block), threads, 0, (cudaStream_t)stream>>>(query, block_keys, ring, scores, layer, begin, frontier, status);
    return launched();
}
int attention_softmax(const float* scores, float* probabilities, float* statistics,
                      int keys, uint32_t* status, void* stream) {
    if (!scores || !probabilities || keys < block || keys > max_keys) return int(cudaErrorInvalidValue);
    softmax_kernel<<<query_heads * block, threads, 0, (cudaStream_t)stream>>>(scores, probabilities, statistics, keys, status);
    return launched();
}
int attention_values(const float* probabilities, const uint16_t* block_values,
                     const uint16_t* ring, uint16_t* output, int layer,
                     int begin, int frontier, uint32_t* status, void* stream) {
    if (!probabilities || !block_values || !ring || !output || layer < 0 || layer >= layers ||
        begin < 0 || frontier < begin || frontier - begin > capacity || frontier > 262144 - block)
        return int(cudaErrorInvalidValue);
    value_kernel<<<query_heads * block, head_dim, 0, (cudaStream_t)stream>>>(probabilities, block_values, ring, output, layer, begin, frontier, status);
    return launched();
}
} // namespace strata::kernels::dflash2
