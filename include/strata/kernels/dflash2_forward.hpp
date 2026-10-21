// DFlash2's BF16 operation boundaries, also used by the device selector.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::dflash2 {
inline constexpr int layers = 5;
inline constexpr int hidden = 2560;
inline constexpr int intermediate = 7680;
inline constexpr int query_heads = 32;
inline constexpr int kv_heads = 8;
inline constexpr int head_dim = 128;
inline constexpr int query_width = query_heads * head_dim;
inline constexpr int kv_width = kv_heads * head_dim;
inline constexpr int block = 4;   // rows at or below this use the reference-validated small path
inline constexpr int capacity = 2047;
inline constexpr int groups = hidden / 16;
inline constexpr int dynamic_width = 2 * 2 * groups;
inline constexpr int max_keys = capacity + block;
inline constexpr size_t kv_elements = size_t(layers) * 2 * kv_heads * capacity * head_dim;

// All operations return a CUDA runtime status, enqueue on exactly the supplied
// stream (including nullptr), and never allocate, synchronize, throw or exit.
// status, when supplied, is a device uint32_t: any nonfinite result ORs bit 0.
// Matrices are unchanged row-major BF16 checkpoint bytes, [output,input].
// The <=4-row path uses the existing bf16_gemv warp accumulation/reduction;
// the context path tiles rows/outputs with FP32 FMAs, never FP16/native BF16.
int linear_bf16(const uint16_t* input, const uint16_t* weight, uint16_t* output,
                int rows, int input_width, int output_width, void* stream,
                uint32_t* status = nullptr);
int finite_bf16(const uint16_t* input, size_t count, uint32_t* status, void* stream);
// Exact BF16 -> FP32 widening for the borrowed native head, no normalization.
int widen_bf16(const uint16_t* input, float* output, size_t count, void* stream);
// statistics is [rows,2]: FP32 mean square, reciprocal RMS. Normalization rounds
// to BF16 BEFORE the separate learned-weight multiplication, as Qwen3RMSNorm.
int normalize_bf16(const uint16_t* input, uint16_t* output, float* statistics,
                   int rows, int width, uint32_t* status, void* stream);
int multiply_norm_bf16(uint16_t* rows, const uint16_t* weight, int count,
                       int width, uint32_t* status, void* stream);
int add_bf16(const uint16_t* left, const uint16_t* right, uint16_t* output,
             int count, uint32_t* status, void* stream);
int multiply_bf16(const uint16_t* left, const uint16_t* right, uint16_t* output,
                  int count, uint32_t* status, void* stream);
int silu_bf16(uint16_t* values, int count, uint32_t* status, void* stream);

// One pair of independent stage kernels per convolution. Dynamic storage is
// [4,2 stages,2 offsets,160 groups]. Each offset has THREE BF16 boundaries:
// base*shifted input, accumulator+that product, addcmul(acc,dynamic,input).
// No input is ever read outside this four-row block.
int convolution_product(const uint16_t* input, const uint16_t* base,
                         uint16_t* product, int stage, int offset,
                         uint32_t* status, void* stream);
int convolution_addcmul(uint16_t* accumulator, const uint16_t* dynamic,
                        const uint16_t* input, int stage, int offset,
                        uint32_t* status, void* stream);

int inverse_frequencies(float* output, void* stream);
// cos/sin are [rows,64], shared by both rotary halves. Positions are absolute.
int rotary_tables(const float* inverse, uint16_t* cosine, uint16_t* sine,
                   int first_position, int rows, uint32_t* status, void* stream);
// Input/output [rows,heads,128]; each BF16 multiply and final add rounds.
int rotary_bf16(uint16_t* rows, const uint16_t* cosine, const uint16_t* sine,
                int count, int heads, uint32_t* status, void* stream);
// The committed ring is [layer,kind(K/V),head,absolute_position%2047,dim].
int store_context(const uint16_t* keys, const uint16_t* values, uint16_t* ring,
                   int layer, int first_position, int rows,
                   uint32_t* status, void* stream);

// Explicit abs(query_position-key_position)<2048, including ALL four block
// keys, not a causal block mask. Scores/probabilities are [32,4,context+4].
// The pinned torch MATH path widens BF16 Q/K/V, scales Q and K separately by
// sqrt(128**-0.5), and keeps scores, softmax and PV accumulation in FP32.
int attention_scores(const uint16_t* query, const uint16_t* block_keys,
                      const uint16_t* ring, float* scores, int layer,
                      int context_begin, int frontier, uint32_t* status, void* stream);
int attention_softmax(const float* scores, float* probabilities, float* statistics,
                      int keys, uint32_t* status, void* stream);
int attention_values(const float* probabilities, const uint16_t* block_values,
                     const uint16_t* ring, uint16_t* output, int layer,
                     int context_begin, int frontier, uint32_t* status, void* stream);
} // namespace strata::kernels::dflash2
