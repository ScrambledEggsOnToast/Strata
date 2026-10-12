// #195: additive WMMA port from jmnargi/Strata-V100 277a707b58d08b7a089b9c41c4c9610a05a27405.
// Included at iq_kernels.cu EOF; existing dequantizers and SM86 kernels are deliberately untouched.
// Packed GGML layouts/codebooks retain third_party/ggml/LICENSE (MIT). This changes arithmetic.
#pragma once

#if !defined(__HIPCC__)
#include <mma.h>
namespace strata::kernels {
namespace {

template<typename dst_t>
__device__ void wmma_dq_iq2_s(const void* vx, int64_t ibs, dst_t* yy, int tid, const uint64_t* codebook) {
    const block_iq2_s* x = (const block_iq2_s*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint64_t grid = codebook[x[ibs].qs[4 * ib + il] | ((x[ibs].qh[ib] << (8 - 2 * il)) & 0x300)];
    const float d = (float) x[ibs].d * (0.5f + ((x[ibs].scales[ib] >> 4 * (il / 2)) & 0xf)) * 0.25f;
    const uint8_t signs = x[ibs].qs[QK_K / 8 + 4 * ib + il];
    for (int j = 0; j < 8; ++j)
        y[j] = cvt<dst_t>(d * (uint8_t) (grid >> (8 * j)) * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
}
template<typename dst_t>
__device__ void wmma_dq_iq3_xxs(const void* vx, int64_t ibs, dst_t* yy, int tid, const uint32_t* codebook) {
    const block_iq3_xxs* x = (const block_iq3_xxs*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint8_t* q3 = x[ibs].qs + 8 * ib;
    const uint16_t* gas = (const uint16_t*) (x[ibs].qs + QK_K / 4) + 2 * ib;
    const uint32_t grid1 = codebook[q3[2 * il + 0]];
    const uint32_t grid2 = codebook[q3[2 * il + 1]];
    const uint32_t aux32 = gas[0] | (gas[1] << 16);
    const float d = (float) x[ibs].d * (0.5f + (aux32 >> 28)) * 0.5f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * il) & 127];
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = cvt<dst_t>(d * (uint8_t) (grid1 >> (8 * j)) * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f));
        y[j + 4] = cvt<dst_t>(d * (uint8_t) (grid2 >> (8 * j)) * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f));
    }
}
template<typename dst_t>
__device__ void wmma_dq_iq3_s(const void* vx, int64_t ibs, dst_t* yy, int tid, const uint32_t* codebook) {
    const block_iq3_s* x = (const block_iq3_s*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint8_t* qs = x[ibs].qs + 8 * ib;
    const uint32_t grid1 = codebook[qs[2 * il + 0] | ((x[ibs].qh[ib] << (8 - 2 * il)) & 256)];
    const uint32_t grid2 = codebook[qs[2 * il + 1] | ((x[ibs].qh[ib] << (7 - 2 * il)) & 256)];
    const float d = (float) x[ibs].d * (1 + 2 * ((x[ibs].scales[ib / 2] >> 4 * (ib % 2)) & 0xf));
    const uint8_t signs = x[ibs].signs[4 * ib + il];
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = cvt<dst_t>(d * (uint8_t) (grid1 >> (8 * j)) * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f));
        y[j + 4] = cvt<dst_t>(d * (uint8_t) (grid2 >> (8 * j)) * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f));
    }
}
template<typename dst_t>
__device__ void wmma_dq_iq4_xs(const void* vx, int64_t ibs, dst_t* yy, int tid, const int8_t* codebook) {
    const block_iq4_xs* x = (const block_iq4_xs*) vx + ibs;
    const int il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 4 * il;
    const uint8_t* q4 = x->qs + 16 * ib + 4 * il;
    const float d = (float) x->d * ((((x->scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((x->scales_h >> 2 * ib) & 3) << 4)) - 32);
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = cvt<dst_t>(d * codebook[q4[j] & 0xf]);
        y[j + 16] = cvt<dst_t>(d * codebook[q4[j] >> 4]);
    }
}

constexpr int WG_BM = 64, WG_BN = 128, WG_THREADS = 256, WG_XK = 128, WG_MT = 4;
__host__ __device__ constexpr size_t gemm_iq_f16_grouped_smem(int wk, int xk = WG_XK) {
    return (size_t) WG_BN * (wk + 8) * 2 + (size_t) WG_BM * (xk + 8) * 2;
}
// A block covers 128 weight rows of one expert and up to WG_MT * WG_BM token rows.
// Each weight superblock is decoded once per chunk into shared memory; activations use a BM x 128 tile.
// Eight warps use 32 x 32 m-tiles and FP32 accumulators for four m-tiles.
// DEC selects the decoder: 0 dq_dispatch (256 values), 1 Q2_0 (64 values), 2 IQ4_NL (32 values).
// Optional ids gather input rows. With indirect weights, W is a device array of expert pointers;
// otherwise experts are contiguous with expert_bytes stride.
template<int WK, int DEC, int TY = 0, int XK = WG_XK, int MT = WG_MT>
__global__ void __launch_bounds__(WG_THREADS, DEC == 1 ? 2 : 1) gemm_iq_f16_grouped_kernel(
        int ty, const uint8_t* __restrict__ W, size_t expert_bytes, size_t row_bytes, int K,
        const __half* __restrict__ X, int ldx, const int32_t* __restrict__ bounds, const int32_t* __restrict__ ids,
        int ch_per_e, float* __restrict__ Y, int ldy, int indirect) {
    static_assert(gemm_iq_f16_grouped_smem(WK, XK) >= (size_t) WG_BM * (WG_BN + 4) * sizeof(float));
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700
    using namespace nvcuda;
    constexpr int LDW = WK + 8, LDX = XK + 8;
    extern __shared__ __align__(32) unsigned char wg_smem[];
    __half* sw = reinterpret_cast<__half*>(wg_smem);            // [BN][LDW]
    __half* sx = sw + WG_BN * LDW;                              // [BM][LDX]
    const int e = blockIdx.y / ch_per_e, ch = blockIdx.y % ch_per_e;
    const int c0 = bounds[e] + ch * MT * WG_BM, r1 = bounds[e + 1];
    if (c0 >= r1) return;
    const int crow = min(MT * WG_BM, r1 - c0);
    const int nmt = (crow + WG_BM - 1) / WG_BM;
    const int n0 = blockIdx.x * WG_BN;
    const uint8_t* we = indirect ? reinterpret_cast<const uint8_t* const*>(W)[e]
                                 : W + (size_t) e * expert_bytes;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp >> 2, wn = warp & 3;                     // rows wm*32, cols wn*32 of an m-tile
    __shared__ uint64_t codebook[TY == 22 ? 1024 : TY == 21 ? 256 : TY == 18 ? 128 : 2];
    auto* grid32 = reinterpret_cast<uint32_t*>(codebook);
    auto* grid8 = reinterpret_cast<int8_t*>(codebook);
    if constexpr (DEC == 0 && TY != 0) {
        if constexpr (TY == 22) {
            for (int i = tid; i < 1024; i += WG_THREADS) codebook[i] = iq2s_grid[i];
        } else if constexpr (TY == 21) {
            for (int i = tid; i < 512; i += WG_THREADS) grid32[i] = iq3s_grid[i];
        } else if constexpr (TY == 18) {
            for (int i = tid; i < 256; i += WG_THREADS) grid32[i] = iq3xxs_grid[i];
        } else if (tid < 16) grid8[tid] = kvalues_iq4nl[tid];
        __syncthreads();
    } else if constexpr (DEC == 2) {
        if (tid < 16) grid8[tid] = kvalues_iq4nl[tid];
        __syncthreads();
    }
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[MT][2][2];
#pragma unroll
    for (int t = 0; t < MT; ++t)
#pragma unroll
        for (int a = 0; a < 2; ++a)
#pragma unroll
            for (int b = 0; b < 2; ++b) wmma::fill_fragment(acc[t][a][b], 0.0f);
    // the activation tiles in order (k0, m-tile, half): the next one's 4 uint4 per thread are loaded into registers
    // while the current one is multiplied (the first before the weights' dequantization)
    constexpr int XQ = WG_BM * (XK / 8) / WG_THREADS;
    const int nx = WK / XK, per_k = nmt * nx, total = (K / WK) * per_k;
    uint4 xr[XQ];
    auto load_x = [&](int it) {
        const int kb = it / per_k, rem = it % per_k, t = rem / nx, xh = rem % nx;
        const int rows = min(WG_BM, crow - t * WG_BM);
#pragma unroll
        for (int q = 0; q < XQ; ++q) {
            const int i = tid + q * WG_THREADS, r = i / (XK / 8), c = (i % (XK / 8)) * 8;
            const int row = c0 + t * WG_BM + r;                  // the output row; ids gathers its input row
            xr[q] = r < rows ? *reinterpret_cast<const uint4*>(X + (size_t) (ids ? ids[row] : row) * ldx + kb * WK +
                                                              xh * XK + c)
                             : make_uint4(0, 0, 0, 0);
        }
    };
    load_x(0);
    int it = 0;
    for (int k0 = 0; k0 < K; k0 += WK) {
        if constexpr (DEC == 0) {
            for (int r = warp; r < WG_BN; r += WG_THREADS / 32) {
                const auto* wr = we + (size_t) (n0 + r) * row_bytes;
                auto* out = sw + r * LDW;
                if constexpr (TY == 22) wmma_dq_iq2_s(wr, k0 / 256, out, lane, codebook);
                else if constexpr (TY == 21) wmma_dq_iq3_s(wr, k0 / 256, out, lane, grid32);
                else if constexpr (TY == 18) wmma_dq_iq3_xxs(wr, k0 / 256, out, lane, grid32);
                else if constexpr (TY == 23) wmma_dq_iq4_xs(wr, k0 / 256, out, lane, grid8);
                else dq_dispatch<__half>(ty, wr, k0 / 256, out, lane);
            }
        } else if constexpr (DEC == 1) {
            const int r = tid >> 1, blk = tid & 1;               // one thread per (row, 64-value block)
            const block_q2_0* b = reinterpret_cast<const block_q2_0*>(we + (size_t) (n0 + r) * row_bytes) + k0 / 64 + blk;
            // Codes select {-d, signed zero, d, 2*d}. Pack two exact FP16 weights with
            // a byte permutation instead of repeated integer-to-float conversions.
            const uint32_t d = __half_as_ushort(b->d);
            const uint32_t z = __half_as_ushort(__hmul(b->d, __float2half(0.0f)));
            const uint32_t twice = __half_as_ushort(__hadd(b->d, b->d));
            const uint32_t table01 = (d ^ 0x8000u) | (z << 16);
            const uint32_t table23 = d | (twice << 16);
            uint4 q4;
            __builtin_memcpy(&q4, b->qs, 16);
            const uint8_t* qb = reinterpret_cast<const uint8_t*>(&q4);
            __half* o = sw + r * LDW + blk * 64;
#pragma unroll
            for (int c = 0; c < 16; ++c) {
#pragma unroll
                for (int pair = 0; pair < 2; ++pair) {
                    const unsigned q = qb[c] >> (4 * pair);
                    const unsigned a = (q & 3u) << 1, b = ((q >> 2) & 3u) << 1;
                    const unsigned selector = 0x1010u | a | (a << 4) | (b << 8) | (b << 12);
                    *reinterpret_cast<uint32_t*>(o + 4 * c + 2 * pair) = __byte_perm(table01, table23, selector);
                }
            }
        } else {
            const int r = tid >> 1, h = tid & 1;                 // one thread per (row, 64-value half of the chunk)
            const block_iq4_nl* b = reinterpret_cast<const block_iq4_nl*>(we + (size_t) (n0 + r) * row_bytes) + k0 / 32 + 2 * h;
            __half* o = sw + r * LDW + h * 64;
#pragma unroll
            for (int blk = 0; blk < 2; ++blk) {
                const __half2 d = __halves2half2(b[blk].d, b[blk].d);
                const uint8_t* qs = b[blk].qs;
#pragma unroll
                for (int j = 0; j < 16; j += 2) {
                    uint16_t q;
                    __builtin_memcpy(&q, qs + j, sizeof(q));
                    const __half2 lo = __floats2half2_rn((float) grid8[q & 0xf], (float) grid8[(q >> 8) & 0xf]);
                    const __half2 hi = __floats2half2_rn((float) grid8[(q >> 4) & 0xf], (float) grid8[q >> 12]);
                    *reinterpret_cast<__half2*>(o + 32 * blk + j) = __hmul2(d, lo);
                    *reinterpret_cast<__half2*>(o + 32 * blk + 16 + j) = __hmul2(d, hi);
                }
            }
        }
#pragma unroll
        for (int t = 0; t < MT; ++t) {
            if (t >= nmt) break;
#pragma unroll 1
            for (int xh = 0; xh < WK / XK; ++xh, ++it) {
                __syncthreads();                                 // the previous X tile is consumed (and W is written)
#pragma unroll
                for (int q = 0; q < XQ; ++q) {
                    const int i = tid + q * WG_THREADS, r = i / (XK / 8), c = (i % (XK / 8)) * 8;
                    *reinterpret_cast<uint4*>(sx + r * LDX + c) = xr[q];
                }
                __syncthreads();
                if (it + 1 < total) load_x(it + 1);
#pragma unroll
                for (int kk = 0; kk < XK; kk += 16) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a[2];
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b[2];
#pragma unroll
                    for (int f = 0; f < 2; ++f) {
                        wmma::load_matrix_sync(a[f], sx + (wm * 32 + f * 16) * LDX + kk, LDX);
                        wmma::load_matrix_sync(b[f], sw + (wn * 32 + f * 16) * LDW + xh * XK + kk, LDW);
                    }
#pragma unroll
                    for (int fa = 0; fa < 2; ++fa)
#pragma unroll
                        for (int fb = 0; fb < 2; ++fb) wmma::mma_sync(acc[t][fa][fb], a[fa], b[fb], acc[t][fa][fb]);
                }
            }
        }
        __syncthreads();                                         // W is consumed before the next superblock
    }
    float* so = reinterpret_cast<float*>(wg_smem);               // [BM][BN + 4] per m-tile
    constexpr int LDO = WG_BN + 4;
#pragma unroll
    for (int t = 0; t < MT; ++t) {
        if (t >= nmt) break;
        const int rows = min(WG_BM, crow - t * WG_BM);
#pragma unroll
        for (int fa = 0; fa < 2; ++fa)
#pragma unroll
            for (int fb = 0; fb < 2; ++fb)
                wmma::store_matrix_sync(so + (wm * 32 + fa * 16) * LDO + wn * 32 + fb * 16, acc[t][fa][fb], LDO, wmma::mem_row_major);
        __syncthreads();
        for (int i = tid; i < rows * (WG_BN / 4); i += WG_THREADS) {
            const int r = i / (WG_BN / 4), c = (i % (WG_BN / 4)) * 4;
            *reinterpret_cast<float4*>(Y + (size_t) (c0 + t * WG_BM + r) * ldy + n0 + c) = *reinterpret_cast<const float4*>(so + r * LDO + c);
        }
        __syncthreads();
    }
#endif
}

// Attribute setup is device-local; a refusal happens before any product launch.
unsigned gemm_iq_f16_grouped_caps(int dev) {
    static thread_local unsigned per_dev[64]{};
    if (dev < 0 || dev >= 64) return 0;
    if (!(per_dev[dev] & 1u)) {
        unsigned caps = 1u;
        int major = 0, minor = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess &&
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) == cudaSuccess &&
            major == 7 && minor == 0) {
#define STRATA_WMMA_CAP(BIT, ...) \
            if (cudaFuncSetAttribute(gemm_iq_f16_grouped_kernel<__VA_ARGS__>, \
                    cudaFuncAttributeMaxDynamicSharedMemorySize, \
                    (int) gemm_iq_f16_grouped_smem(256)) == cudaSuccess) caps |= BIT; \
            else (void) cudaGetLastError();
            STRATA_WMMA_CAP(2u, 256, 0, 18)
            STRATA_WMMA_CAP(4u, 256, 0, 21)
            STRATA_WMMA_CAP(8u, 256, 0, 22)
            STRATA_WMMA_CAP(16u, 256, 0, 23)
#undef STRATA_WMMA_CAP
            if (cudaFuncSetAttribute(gemm_iq_f16_grouped_kernel<128, 1, 0, 64, 2>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    (int) gemm_iq_f16_grouped_smem(128, 64)) == cudaSuccess) caps |= 32u;
            else (void) cudaGetLastError();
            if (cudaFuncSetAttribute(gemm_iq_f16_grouped_kernel<128, 2>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    (int) gemm_iq_f16_grouped_smem(128)) == cudaSuccess) caps |= 64u;
            else (void) cudaGetLastError();
        }
        per_dev[dev] = caps;
    }
    return per_dev[dev];
}
unsigned gemm_iq_f16_grouped_bit(int ty) {
    switch (ty) {
        case 18: return 2u;
        case 21: return 4u;
        case 22: return 8u;
        case 23: return 16u;
        case 42: return 32u;
        case 20: return 64u;
        default: return 0;
    }
}
}  // namespace

bool gemm_iq_f16_grouped_supported(int ty, int n_out, int K) noexcept {
    const unsigned bit = gemm_iq_f16_grouped_bit(ty);
    if (!bit || n_out <= 0 || n_out % WG_BN || K <= 0 || K % (ty == 20 || ty == 42 ? 128 : 256))
        return false;
    int dev = 0;
    return cudaGetDevice(&dev) == cudaSuccess && (gemm_iq_f16_grouped_caps(dev) & bit) != 0;
}

bool gemm_iq_f16_grouped(int ty, const void* W, size_t expert_bytes, int n_out, int K, const void* X, int ldx,
                         const int32_t* bounds, int n_experts, int max_rows, float* Y, int ldy, void* stream,
                         const int32_t* ids, bool indirect_weights) {
    if (n_out <= 0 || K <= 0 || n_out % WG_BN != 0 || n_experts <= 0 || max_rows <= 0) return false;
    if (ldx < K || ldx % 8 != 0 || ldy < n_out || ldy % 4 != 0) return false;   // uint4 loads / float4 stores
    const int dec = ty == 42 ? 1 : ty == 20 ? 2 : 0;             // Q2_0 and IQ4_NL at WK 128, everything else WK 256
    if (!gemm_iq_f16_grouped_supported(ty, n_out, K)) return false;
    if (!W || !X || !Y || !bounds || (uintptr_t) X % 16 || (uintptr_t) Y % 16 ||
        (uintptr_t) W % (indirect_weights ? alignof(void*) : 2)) return false;
    const size_t row_bytes = iq_row_bytes(ty, K);
    const int mt = dec == 1 ? 2 : WG_MT;
    const int chunks = (int) (((int64_t) max_rows + mt * WG_BM - 1) / (mt * WG_BM));
    if ((int64_t) n_experts * chunks > 65535) return false;
    const dim3 grid((unsigned) (n_out / WG_BN), (unsigned) (n_experts * chunks));
    cudaStream_t s = (cudaStream_t) stream;
    const uint8_t* w = (const uint8_t*) W;
    const __half* x = (const __half*) X;
    const int ind = indirect_weights ? 1 : 0;
    if (dec == 0 && ty == 18)
        gemm_iq_f16_grouped_kernel<256, 0, 18><<<grid, WG_THREADS, gemm_iq_f16_grouped_smem(256), s>>>(
            ty, w, expert_bytes, row_bytes, K, x, ldx, bounds, ids, chunks, Y, ldy, ind);
    else if (dec == 0 && ty == 21)
        gemm_iq_f16_grouped_kernel<256, 0, 21><<<grid, WG_THREADS, gemm_iq_f16_grouped_smem(256), s>>>(
            ty, w, expert_bytes, row_bytes, K, x, ldx, bounds, ids, chunks, Y, ldy, ind);
    else if (dec == 0 && ty == 22)
        gemm_iq_f16_grouped_kernel<256, 0, 22><<<grid, WG_THREADS, gemm_iq_f16_grouped_smem(256), s>>>(
            ty, w, expert_bytes, row_bytes, K, x, ldx, bounds, ids, chunks, Y, ldy, ind);
    else if (dec == 0 && ty == 23)
        gemm_iq_f16_grouped_kernel<256, 0, 23><<<grid, WG_THREADS, gemm_iq_f16_grouped_smem(256), s>>>(
            ty, w, expert_bytes, row_bytes, K, x, ldx, bounds, ids, chunks, Y, ldy, ind);
    else if (dec == 1)
        gemm_iq_f16_grouped_kernel<128, 1, 0, 64, 2><<<grid, WG_THREADS, gemm_iq_f16_grouped_smem(128, 64), s>>>(
            ty, w, expert_bytes, row_bytes, K, x, ldx, bounds, ids, chunks, Y, ldy, ind);
    else
        gemm_iq_f16_grouped_kernel<128, 2><<<grid, WG_THREADS, gemm_iq_f16_grouped_smem(128), s>>>(
            ty, w, expert_bytes, row_bytes, K, x, ldx, bounds, ids, chunks, Y, ldy, ind);
    check("gemm_iq_f16_grouped");   // Launch errors are failures, never permission to fall back after a launch.
    return true;
}
}  // namespace strata::kernels
#else
namespace strata::kernels {
bool gemm_iq_f16_grouped_supported(int, int, int) noexcept { return false; }
bool gemm_iq_f16_grouped(int, const void*, size_t, int, int, const void*, int, const int32_t*, int, int, float*, int, void*,
                         const int32_t*, bool) { return false; }
}  // namespace strata::kernels
#endif
