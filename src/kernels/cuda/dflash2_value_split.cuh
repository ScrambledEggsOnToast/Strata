#pragma once
#include "strata/kernels/dflash2_forward.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include <cuda_runtime.h>
#include <cmath>

// Guarded grouped-head split-K value accumulation; chunk128 production variant.
namespace strata::kernels::dflash2::value_split_detail {
using namespace strata::kernels;
using namespace strata::kernels::dflash2;
constexpr int grouped_heads = query_heads / kv_heads;
static_assert(grouped_heads == 4 && head_dim == 128);

inline size_t partial_count(int rows, int chunks) {
    return size_t(query_heads) * rows * chunks * head_dim;
}
inline size_t flag_count(int rows, int chunks) {
    return size_t(kv_heads) * rows * chunks * head_dim;
}
__device__ inline float value_at(const uint16_t* block_values, const uint16_t* ring,
    int layer, int begin, int context, int hk, int k, int d) {
    const size_t at = k < context
        ? (((size_t(layer) * 2 + 1) * kv_heads + hk) * capacity + (begin + k) % capacity) * head_dim + d
        : (size_t(k - context) * kv_heads + hk) * head_dim + d;
    return f32_from_bf16(k < context ? ring[at] : block_values[at]);
}

// All four query heads share each BF16 V load. Keys are never skipped for p=0.
// An unsafe flag applies to all four heads of this one output dimension.
// Guarded values ensure every partial, original prefix, and merged sum is
// finite: <=2055 products, |p|<=1, |v|<=2^112 => absolute sum <2^124.
// Thus this changed ordering cannot hide an overflow of the retained chain.
template<int Chunk>
__global__ void partial_kernel(const float* probabilities, const uint16_t* values,
    const uint16_t* ring, float* partial, uint8_t* unsafe, int layer,
    int begin, int frontier, int rows) {
    const int c = blockIdx.x, hk = blockIdx.y, row = blockIdx.z, d = threadIdx.x;
    const int context = frontier - begin, keys = context + rows;
    const int chunks = (keys + Chunk - 1) / Chunk;
    const int end = min(keys, (c + 1) * Chunk);
    float sum[grouped_heads]{};
    bool bad = false;
    for (int k = c * Chunk; k < end; ++k) {
        const float v = value_at(values, ring, layer, begin, context, hk, k, d);
        bad |= !isfinite(v) || fabsf(v) > 0x1p112f;
#pragma unroll
        for (int h = 0; h < grouped_heads; ++h) {
            const float p = probabilities[(size_t(hk * grouped_heads + h) * rows + row) * keys + k];
            bad |= !isfinite(p) || fabsf(p) > 1.0f;
            sum[h] = fmaf(p, v, sum[h]);
        }
    }
#pragma unroll
    for (int h = 0; h < grouped_heads; ++h)
        partial[((size_t(hk * grouped_heads + h) * rows + row) * chunks + c) * head_dim + d] = sum[h];
    unsafe[((size_t(hk) * rows + row) * chunks + c) * head_dim + d] = uint8_t(bad);
}

template<int Chunk>
__global__ void merge_kernel(const float* probabilities, const uint16_t* values,
    const uint16_t* ring, const float* partial, const uint8_t* unsafe,
    uint16_t* output, int layer, int begin, int frontier, int rows,
    uint32_t* status, uint32_t* fallback_dimensions) {
    const int hk = blockIdx.x, row = blockIdx.y, d = threadIdx.x;
    const int context = frontier - begin, keys = context + rows;
    const int chunks = (keys + Chunk - 1) / Chunk;
    float sum[grouped_heads]{};
    bool bad = false;
    for (int c = 0; c < chunks; ++c) {
        bad |= unsafe[((size_t(hk) * rows + row) * chunks + c) * head_dim + d] != 0;
#pragma unroll
        for (int h = 0; h < grouped_heads; ++h)
            sum[h] = __fadd_rn(sum[h], partial[((size_t(hk * grouped_heads + h) * rows + row) * chunks + c) * head_dim + d]);
    }
    if (bad) {
        if (fallback_dimensions) atomicAdd(fallback_dimensions, 1u);
#pragma unroll
        for (int h = 0; h < grouped_heads; ++h) sum[h] = 0.0f;
        // Retained increasing-key FMA ordering, including masked zero times NaN.
        for (int k = 0; k < keys; ++k) {
            const float v = value_at(values, ring, layer, begin, context, hk, k, d);
#pragma unroll
            for (int h = 0; h < grouped_heads; ++h) {
                const float p = probabilities[(size_t(hk * grouped_heads + h) * rows + row) * keys + k];
                sum[h] = fmaf(p, v, sum[h]);
            }
        }
    }
#pragma unroll
    for (int h = 0; h < grouped_heads; ++h) {
        const uint16_t bits = bf16_from_f32(sum[h]);
        if ((bits & 0x7f80u) == 0x7f80u && status) atomicOr(status, 1u);
        output[(size_t(row) * query_heads + hk * grouped_heads + h) * head_dim + d] = bits;
    }
}

template<int Chunk>
inline cudaError_t launch(const float* probabilities, const uint16_t* values,
    const uint16_t* ring, float* partial, uint8_t* unsafe, uint16_t* output,
    int layer, int begin, int frontier, int rows, uint32_t* status,
    cudaStream_t stream, uint32_t* fallback_dimensions = nullptr) {
    static_assert(Chunk == 64 || Chunk == 128 || Chunk == 256);
    if (!probabilities || !values || !ring || !partial || !unsafe || !output ||
        layer < 0 || layer >= layers || rows < 1 || rows > block || begin < 0 ||
        frontier < begin || frontier - begin > capacity || frontier > 262144 - rows)
        return cudaErrorInvalidValue;
    const int chunks = (frontier - begin + rows + Chunk - 1) / Chunk;
    partial_kernel<Chunk><<<dim3(chunks, kv_heads, rows), head_dim, 0, stream>>>(
        probabilities, values, ring, partial, unsafe, layer, begin, frontier, rows);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return error;
    merge_kernel<Chunk><<<dim3(kv_heads, rows), head_dim, 0, stream>>>(
        probabilities, values, ring, partial, unsafe, output, layer, begin, frontier,
        rows, status, fallback_dimensions);
    return cudaGetLastError();
}
} // namespace strata::kernels::dflash2::value_split_detail
