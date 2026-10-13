// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Adapted from 1Cat-vLLM/csrc/qsa_lexicographic_topk.cuh at
// be3722bc095b9792bf9833068595f0b1ef70ad37 (PR #1171).
// Changes for Strata: weighted four-cell blocks, canonical NaNs/signed zero,
// exact cutoff ties via the existing ascending cell emitter, and scratch reuse.
// The donor license is in third_party/1cat-qsa/LICENSE.
#pragma once


constexpr int QSA_NARROW_CAP = 2304;
constexpr int QSA_NARROW_MIN = 32768;

struct QsaNarrowScratch {
    int hist[256];
    uint32_t keys[QSA_NARROW_CAP];
    int weights[QSA_NARROW_CAP];
    int remaining, bin, bucket_cells, count, invalid, above;
    uint32_t prefix;
};
static_assert(sizeof(QsaNarrowScratch) <= TK_T / 32 * 256 * sizeof(int));

// Half conversion is used only for a monotone coarse bucket, never as the score
// compared at the final cutoff. Match order_key's equality of both signed zeros.
__device__ __forceinline__ uint32_t qsa_narrow_coarse_key(float value) {
    value += 0.0f;
    const uint16_t bits = __half_as_ushort(__float2half_rn(value));
    return static_cast<uint16_t>((bits & 0x8000u) ? ~bits : (bits | 0x8000u)) >> 8;
}

__device__ __forceinline__ void qsa_narrow_choose_bin(QsaNarrowScratch& s) {
    if (threadIdx.x == 0) {
        for (int bin = 255; bin >= 0; --bin) {
            const int count = s.hist[bin];
            if (s.remaining > count) {
                s.remaining -= count;
                s.above += count;
            } else {
                s.bin = bin;
                s.bucket_cells = count;
                break;
            }
        }
    }
    __syncthreads();
}

// All decisions use live device steps, so graph replay can cross the live-length
// threshold or change the partial tail. False leaves the ordinary wide selector
// to resolve the entire row; no truncation, host readback or extra launch.
__device__ bool qsa_narrow_pivot(const float* scores, int64_t n_kv, int64_t n_bid, int width,
                                QsaNarrowScratch& s, uint32_t& pivot, int& above) {
    const int64_t nb = n_bid + (n_kv % R != 0);
    if (nb < QSA_NARROW_MIN) return false;
    const int tx = threadIdx.x, lane = tx & 31;
    const float probe = scores[(tx >> 5) * (nb >> 5) + lane];
    const uint32_t coarse = qsa_narrow_coarse_key(probe);
    const bool concentrated = __all_sync(0xffffffffu, coarse == __shfl_sync(0xffffffffu, coarse, 0));
    if (__syncthreads_or(!(probe == probe) || concentrated)) return false;
    if (tx < 256) s.hist[tx] = 0;
    if (tx == 0) {
        s.remaining = width;
        s.prefix = 0;
        s.above = 0;
        s.invalid = 0;
    }
    __syncthreads();
    bool invalid = false;
    for (int64_t b = tx; b < nb; b += TK_T) {
        const float score = scores[b];
        invalid |= !(score == score);
        const int weight = b < n_bid ? R : int(n_kv - n_bid * R);
        atomicAdd(&s.hist[qsa_narrow_coarse_key(score)], weight);
    }
    if (invalid) atomicExch(&s.invalid, 1);
    __syncthreads();
    if (s.invalid) return false;  // NaNs use exactly the existing key-zero rule.
    qsa_narrow_choose_bin(s);
    // There is at most one partial block: ceil(cells / 4) is the exact block
    // count in this bucket. Never gather a bucket larger than the scratch.
    if ((s.bucket_cells + R - 1) / R > QSA_NARROW_CAP) return false;
    const int coarse_bin = s.bin;
    if (tx == 0) s.count = 0;
    __syncthreads();
    for (int64_t b = tx; b < nb; b += TK_T) {
        const float score = scores[b];
        if (qsa_narrow_coarse_key(score) == uint32_t(coarse_bin)) {
            const int slot = atomicAdd(&s.count, 1);
            s.keys[slot] = order_key(score);
            s.weights[slot] = b < n_bid ? R : int(n_kv - n_bid * R);
        }
    }
    __syncthreads();
    // These are full FP32 order keys. Keeping the initial (bounded) candidate
    // set avoids donor ping-pong buffers; later bytes filter by the fixed prefix.
    for (int shift = 24; shift >= 0; shift -= 8) {
        if (tx < 256) s.hist[tx] = 0;
        __syncthreads();
        const uint32_t mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
        for (int item = tx; item < s.count; item += TK_T) {
            const uint32_t key = s.keys[item];
            if ((key & mask) == s.prefix) atomicAdd(&s.hist[(key >> shift) & 255u], s.weights[item]);
        }
        __syncthreads();
        qsa_narrow_choose_bin(s);
        if (tx == 0) s.prefix |= uint32_t(s.bin) << shift;
        __syncthreads();
    }
    pivot = s.prefix;
    above = s.above;
    __syncthreads();  // all scratch readers finish before its wide-emitter reuse
    return true;
}
