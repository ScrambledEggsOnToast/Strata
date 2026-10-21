#include "strata/kernels/dflash2_selector.hpp"

#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/dflash2_forward.hpp"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <cmath>

namespace strata::kernels::dflash2 {
namespace {
constexpr unsigned warp_mask = 0xffffffffu;
constexpr int tile_threads = 256;

__device__ SelectorCandidate warp_max(SelectorCandidate value) {
    const int lane = int(threadIdx.x) & 31;
    for (int distance = 16; distance != 0; distance >>= 1) {
        const float other_value = __shfl_down_sync(warp_mask, value.value, distance);
        const int32_t other_token = __shfl_down_sync(warp_mask, value.token, distance);
        if (lane + distance < 32 && other_value > value.value) value = {other_value, other_token};
    }
    return value;
}

// Observable semantics of GCC 13.3's __adjust_heap with Torch's value-only greater
// comparator. Sift down through the smaller child (RIGHT on equality), then insert
// the saved value upwards with strict comparison. Do not replace with an index key.
// Authority: gcc-mirror/gcc releases/gcc-13.3.0,
// libstdc++-v3/include/bits/{stl_heap.h,stl_algo.h}; pinned Torch TopKImpl.h lines 44-58.
__device__ void heap_adjust(SelectorCandidate* heap, int position, int count, SelectorCandidate value) {
    const int start = position;
    while (position * 2 + 2 < count) {
        int child = position * 2 + 2;
        if (heap[child].value > heap[child - 1].value) --child;
        heap[position] = heap[child];
        position = child;
    }
    const int left = position * 2 + 1;
    if (left < count) {
        heap[position] = heap[left];
        position = left;
    }
    while (position > start) {
        const int parent = (position - 1) / 2;
        if (!(heap[parent].value > value.value)) break;
        heap[position] = heap[parent];
        position = parent;
    }
    heap[position] = value;
}

__device__ void heap_make(SelectorCandidate* heap, int count) {
    for (int parent = (count - 2) / 2; parent >= 0; --parent)
        heap_adjust(heap, parent, count, heap[parent]);
}

__device__ void heap_sort(SelectorCandidate* heap, int count) {
    for (int remaining = count - 1; remaining > 0; --remaining) {
        const SelectorCandidate saved = heap[remaining];
        heap[remaining] = heap[0];
        heap_adjust(heap, 0, remaining, saved);
    }
}

// A parallel shortlist is sufficient when all global top-17 VALUES are distinct.
// Tie choices inside this private shortlist are unobservable: any repeated selected
// value (also the 16th/17th cutoff) forces a complete reference-order device replay.
__global__ void tile_top_kernel(const float* logits, SelectorWorkspace* work) {
    const int tile = int(blockIdx.x);
    const int row = int(blockIdx.y);
    const int t = int(threadIdx.x);
    const int lane = t & 31;
    const int warp = t >> 5;
    __shared__ SelectorCandidate warp_best[tile_threads / 32];
    __shared__ SelectorCandidate chosen;
    float values[selector_tile_width / tile_threads];
#pragma unroll
    for (int item = 0; item < selector_tile_width / tile_threads; ++item) {
        const int token = tile * selector_tile_width + item * tile_threads + t;
        const float value = token < selector_vocabulary ? logits[size_t(row) * selector_vocabulary + token]
                                                       : -CUDART_INF_F;
        values[item] = value;
        if (token < selector_vocabulary && !isfinite(value))
            atomicOr(&work->result.status, int(selector_nonfinite_logits));
    }
    for (int slot = 0; slot < selector_tile_keep; ++slot) {
        SelectorCandidate best{-CUDART_INF_F, -1};
#pragma unroll
        for (int item = 0; item < selector_tile_width / tile_threads; ++item) {
            if (values[item] > best.value)
                best = {values[item], tile * selector_tile_width + item * tile_threads + t};
        }
        best = warp_max(best);
        if (lane == 0) warp_best[warp] = best;
        __syncthreads();
        if (warp == 0) {
            best = lane < tile_threads / 32 ? warp_best[lane] : SelectorCandidate{-CUDART_INF_F, -1};
            best = warp_max(best);
            if (lane == 0) {
                chosen = best;
                work->tiles[row][tile][slot] = best;
            }
        }
        __syncthreads();
#pragma unroll
        for (int item = 0; item < selector_tile_width / tile_threads; ++item)
            if (tile * selector_tile_width + item * tile_threads + t == chosen.token)
                values[item] = -CUDART_INF_F;
    }
}

__global__ void merge_top_kernel(SelectorWorkspace* work) {
    if (threadIdx.x != 0) return;
    const int row = int(blockIdx.x);
    work->replay[row] = 0;
    if (work->result.status != 0) return;
    __shared__ SelectorCandidate heap[selector_tile_keep];
    for (int slot = 0; slot < selector_tile_keep; ++slot) heap[slot] = work->tiles[row][0][slot];
    heap_make(heap, selector_tile_keep);
    for (int tile = 1; tile < selector_tiles; ++tile)
        for (int slot = 0; slot < selector_tile_keep; ++slot) {
            const SelectorCandidate next = work->tiles[row][tile][slot];
            if (next.value > heap[0].value) heap_adjust(heap, 0, selector_tile_keep, next);
        }
    heap_sort(heap, selector_tile_keep);
    for (int slot = 0; slot < selector_candidates; ++slot)
        if (heap[slot].value == heap[slot + 1].value) {
            work->replay[row] = 1;
            return;
        }
    for (int slot = 0; slot < selector_candidates; ++slot) {
        work->result.candidates[row][slot] = heap[slot].token;
        work->unary[row][slot] = heap[slot].value;
    }
}

// Replay the FULL input sequence, not a shortlist of tied tokens: earlier, later-
// discarded values can change equal-value heap order. Warp-coalesced input batches
// preserve the exact ascending-index scan while the owner lane updates the heap.
__global__ void replay_ties_kernel(const float* logits, SelectorWorkspace* work) {
    const int row = int(blockIdx.x);
    if (work->result.status != 0 || work->replay[row] == 0) return;
    const int lane = int(threadIdx.x);
    __shared__ SelectorCandidate heap[selector_candidates];
    __shared__ float values[32];
    const float* input = logits + size_t(row) * selector_vocabulary;
    if (lane < selector_candidates) heap[lane] = {input[lane], lane};
    __syncwarp(warp_mask);
    if (lane == 0) heap_make(heap, selector_candidates);
    for (int base = selector_candidates; base < selector_vocabulary; base += 32) {
        values[lane] = base + lane < selector_vocabulary ? input[base + lane] : -CUDART_INF_F;
        __syncwarp(warp_mask);
        if (lane == 0)
            for (int offset = 0; offset < 32 && base + offset < selector_vocabulary; ++offset) {
                const SelectorCandidate next{values[offset], base + offset};
                if (next.value > heap[0].value) heap_adjust(heap, 0, selector_candidates, next);
            }
        __syncwarp(warp_mask);
    }
    if (lane == 0) {
        heap_sort(heap, selector_candidates);
        for (int slot = 0; slot < selector_candidates; ++slot) {
            work->result.candidates[row][slot] = heap[slot].token;
            work->unary[row][slot] = heap[slot].value;
        }
    }
}

// Every return before a later block barrier must use one shared status snapshot.
// Reading the global atomic flag independently can strand slower warps at a barrier.
__device__ bool stage_succeeded(SelectorWorkspace* work, int32_t* stage_status) {
    __syncthreads();
    if (threadIdx.x == 0) *stage_status = work->result.status;
    __syncthreads();
    return *stage_status == 0;
}

__global__ void walk_kernel(const uint16_t* predecessor_codebook, const uint16_t* successor_codebook,
                            int32_t anchor, float temperature, float uniform0, float uniform1, float uniform2,
                            SelectorWorkspace* work) {
    const int t = int(threadIdx.x);
    const int lane = t & 31;
    const int candidate = t >> 5;
    __shared__ int32_t predecessor;
    __shared__ float scores[selector_candidates];
    __shared__ float probabilities[selector_candidates];
    __shared__ int32_t stage_status;
    if (t == 0) {
        predecessor = anchor;
        stage_status = work->result.status;
    }
    __syncthreads();
    if (stage_status != 0) return;
    for (int row = 0; row < selector_rows; ++row) {
        const int32_t token = work->result.candidates[row][candidate];
        float dot = 0.0f;
        for (int rank = lane; rank < selector_rank; rank += 32) {
            const float hidden = f32_from_bf16(work->projected[row][rank]);
            const float before = f32_from_bf16(predecessor_codebook[size_t(predecessor) * selector_rank + rank]);
            const float after = f32_from_bf16(successor_codebook[size_t(token) * selector_rank + rank]);
            if (!isfinite(hidden)) atomicOr(&work->result.status, int(selector_nonfinite_projection));
            if (!isfinite(before) || !isfinite(after)) atomicOr(&work->result.status, int(selector_nonfinite_codebook));
            // Upstream elementwise multiplication is a distinct BF16 tensor operation.
            const float product = f32_from_bf16(bf16_from_f32(__fmul_rn(before, hidden)));
            dot = fmaf(product, after, dot);
        }
        for (int distance = 16; distance != 0; distance >>= 1)
            dot = __fadd_rn(dot, __shfl_down_sync(warp_mask, dot, distance));
        if (lane == 0) {
            // Einsum returns BF16 BEFORE addition to the FP32 unary logits.
            const float affinity = f32_from_bf16(bf16_from_f32(dot));
            scores[candidate] = __fadd_rn(work->unary[row][candidate], affinity);
            if (!isfinite(scores[candidate])) atomicOr(&work->result.status, int(selector_nonfinite_score));
        }
        if (!stage_succeeded(work, &stage_status)) return;
        {
            if (t < selector_candidates && temperature > 0.0f) {
                scores[t] = __fdiv_rn(scores[t], temperature);
                if (!isfinite(scores[t])) atomicOr(&work->result.status, int(selector_nonfinite_probability));
            }
            if (!stage_succeeded(work, &stage_status)) return;
            if (candidate == 0) {
                float maximum = lane < selector_candidates ? scores[lane] : -CUDART_INF_F;
                for (int distance = 16; distance != 0; distance >>= 1)
                    maximum = fmaxf(maximum, __shfl_down_sync(warp_mask, maximum, distance));
                maximum = __shfl_sync(warp_mask, maximum, 0);
                const float weight = lane < selector_candidates ? expf(__fsub_rn(scores[lane], maximum)) : 0.0f;
                float sum = weight;
                for (int distance = 16; distance != 0; distance >>= 1)
                    sum = __fadd_rn(sum, __shfl_down_sync(warp_mask, sum, distance));
                sum = __shfl_sync(warp_mask, sum, 0);
                if (lane < selector_candidates) {
                    const float q = __fdiv_rn(weight, sum);
                    probabilities[lane] = q;
                    if (temperature > 0.0f) work->result.conditional_probabilities[row][lane] = q;
                    if (!isfinite(q) || q < 0.0f)
                        atomicOr(&work->result.status, int(selector_nonfinite_probability));
                }
            }
            if (!stage_succeeded(work, &stage_status)) return;
        }
        if (t == 0) {
            double mass = 0.0;
            for (int slot = 0; slot < selector_candidates; ++slot) mass += double(probabilities[slot]);
            const bool valid_mass = mass > 0.0 && fabs(mass - 1.0) <= selector_probability_mass_tolerance;
            if (!valid_mass) atomicOr(&work->result.status, int(selector_nonfinite_probability));
            int chosen = 0;
            if (temperature > 0.0f && valid_mass) {
                const float uniform = row == 0 ? uniform0 : row == 1 ? uniform1 : uniform2;
                double cumulative = 0.0;
                // Same strict inverse-CDF boundary as core::spec_pick. No hidden random draws.
                // No sampling-only renormalization; use the exact FP32 values exported above.
                for (int slot = 0; slot < selector_candidates; ++slot) {
                    if (!(probabilities[slot] > 0.0f)) continue;
                    chosen = slot; // endpoint fallback is always a strictly positive q
                    cumulative += double(probabilities[slot]);
                    if (double(uniform) < cumulative) break;
                }
            } else if (temperature <= 0.0f) {
                for (int slot = 1; slot < selector_candidates; ++slot)
                    if (scores[slot] > scores[chosen]) chosen = slot;
            }
            if (!(probabilities[chosen] > 0.0f))
                atomicOr(&work->result.status, int(selector_nonfinite_probability));
            if (work->result.status == 0) {
                predecessor = work->result.candidates[row][chosen];
                work->result.tokens[row] = predecessor;
                work->result.confidence[row] = probabilities[chosen];
            }
        }
        if (!stage_succeeded(work, &stage_status)) return;
    }
}
} // namespace

int selector_select(const uint16_t* hidden, const float* logits, const uint16_t* hidden_projection,
                    const uint16_t* predecessor_codebook, const uint16_t* successor_codebook,
                    int32_t anchor, float temperature, float uniform0, float uniform1, float uniform2,
                    SelectorWorkspace* workspace, void* stream) {
    const auto cuda_stream = static_cast<cudaStream_t>(stream);
    cudaError_t status = cudaMemsetAsync(&workspace->result, 0, sizeof(SelectorResult), cuda_stream);
    if (status != cudaSuccess) return int(status);
    const int projection = linear_bf16(hidden, hidden_projection, &workspace->projected[0][0], selector_rows,
                                       selector_hidden, selector_rank, stream);
    if (projection != int(cudaSuccess)) return projection;
    tile_top_kernel<<<dim3(selector_tiles, selector_rows), tile_threads, 0, cuda_stream>>>(logits, workspace);
    status = cudaGetLastError();
    if (status != cudaSuccess) return int(status);
    merge_top_kernel<<<selector_rows, 32, 0, cuda_stream>>>(workspace);
    status = cudaGetLastError();
    if (status != cudaSuccess) return int(status);
    replay_ties_kernel<<<selector_rows, 32, 0, cuda_stream>>>(logits, workspace);
    status = cudaGetLastError();
    if (status != cudaSuccess) return int(status);
    walk_kernel<<<1, selector_candidates * 32, 0, cuda_stream>>>(predecessor_codebook, successor_codebook,
        anchor, temperature, uniform0, uniform1, uniform2, workspace);
    return int(cudaGetLastError());
}

} // namespace strata::kernels::dflash2
