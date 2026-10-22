#pragma once
#include "strata/kernels/dflash2_forward.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>
namespace strata::kernels::dflash2::score_grouped_detail {
using namespace strata::kernels;
using namespace strata::kernels::dflash2;
constexpr int group = query_heads / kv_heads;
static_assert(group == 4 && head_dim == 128);
constexpr float scale_root = 0.29730177875068026668f;
__device__ inline float reduce(float sum) {
    for (int offset = 16; offset; offset >>= 1)
        sum = __fadd_rn(sum, __shfl_down_sync(0xffffffffu, sum, offset));
    return sum;
}
// A warp owns one key/query-row and all four query heads sharing that KV head.
// Each head retains d=lane,lane+32,lane+64,lane+96 FMAs then shuffles16/8/4/2/1.
__global__ void kernel(const uint16_t* query, const uint16_t* block_keys,
    const uint16_t* ring, float* scores, int layer, int begin, int frontier,
    int rows, uint32_t* status) {
    const int key = int(blockIdx.x) * 8 + int(threadIdx.x >> 5);
    const int context = frontier - begin, keys = context + rows;
    if (key >= keys) return; // Whole warp only; no cross-warp barrier.
    const int hk = int(blockIdx.y) / rows, row = int(blockIdx.y) % rows;
    const int lane = int(threadIdx.x & 31);
    const int position = key < context ? begin + key : frontier + key - context;
    const int distance = frontier + row - position;
    if (distance <= -2048 || distance >= 2048) {
        if (lane == 0) {
#pragma unroll
            for (int h = 0; h < group; ++h)
                scores[(size_t(hk * group + h) * rows + row) * keys + key] = -CUDART_INF_F;
        }
        return; // Do not read Q or K at masked positions, including NaN/Inf.
    }
    float sum[group]{};
    for (int d = lane; d < head_dim; d += 32) {
        const uint16_t bits = key < context
            ? ring[(((size_t(layer) * 2) * kv_heads + hk) * capacity + position % capacity) * head_dim + d]
            : block_keys[((key - context) * kv_heads + hk) * head_dim + d];
        const float k = __fmul_rn(f32_from_bf16(bits), scale_root);
#pragma unroll
        for (int h = 0; h < group; ++h) {
            const float q = __fmul_rn(f32_from_bf16(query[(size_t(row) * query_heads + hk * group + h) * head_dim + d]), scale_root);
            sum[h] = fmaf(q, k, sum[h]);
        }
    }
#pragma unroll
    for (int h = 0; h < group; ++h) {
        const float value = reduce(sum[h]);
        if (lane == 0) {
            if (!isfinite(value) && status) atomicOr(status, 1u);
            scores[(size_t(hk * group + h) * rows + row) * keys + key] = value;
        }
    }
}
inline cudaError_t launch(const uint16_t* query, const uint16_t* block_keys,
    const uint16_t* ring, float* scores, int layer, int begin, int frontier,
    int rows, uint32_t* status, cudaStream_t stream) {
    if (!query || !block_keys || !ring || !scores || layer < 0 || layer >= layers ||
        rows < 1 || rows > block || begin < 0 || frontier < begin ||
        frontier > 262144 - rows || frontier - begin > capacity)
        return cudaErrorInvalidValue;
    kernel<<<dim3((frontier - begin + rows + 7) / 8, kv_heads * rows), 256, 0, stream>>>(
        query, block_keys, ring, scores, layer, begin, frontier, rows, status);
    return cudaGetLastError();
}
} // namespace strata::kernels::dflash2::score_grouped_detail
