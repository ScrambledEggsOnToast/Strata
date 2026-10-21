// Fixed-geometry DFlash2 selector. All pointers below are on the caller's current CUDA device.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::dflash2 {

inline constexpr int selector_rows = 3;
inline constexpr int selector_hidden = 2560;
inline constexpr int selector_vocabulary = 248320;
inline constexpr int selector_rank = 256;
inline constexpr int selector_candidates = 16;
inline constexpr int selector_tile_width = 1024;
inline constexpr int selector_tiles = (selector_vocabulary + selector_tile_width - 1) / selector_tile_width;
// The extra maximum proves whether any selected value has a tie, including at the cutoff.
inline constexpr int selector_tile_keep = selector_candidates + 1;
// Prospective FP32 normalization bound: 16 * FLT_EPSILON, checked without renormalizing the CDF.
inline constexpr double selector_probability_mass_tolerance = 0x1p-19;

struct SelectorCandidate {
    float value;
    int32_t token;
};

// Read back only this bounded result, never hidden states, logits, weights, or intermediate arithmetic.
struct SelectorResult {
    int32_t tokens[selector_rows];
    int32_t candidates[selector_rows][selector_candidates];
    float conditional_probabilities[selector_rows][selector_candidates];
    float confidence[selector_rows];
    int32_t status;
};

enum SelectorStatus : int32_t {
    selector_nonfinite_logits = 1 << 0,
    selector_nonfinite_projection = 1 << 1,
    selector_nonfinite_codebook = 1 << 2,
    selector_nonfinite_score = 1 << 3,
    selector_nonfinite_probability = 1 << 4,
};

struct SelectorWorkspace {
    SelectorCandidate tiles[selector_rows][selector_tiles][selector_tile_keep];
    uint16_t projected[selector_rows][selector_rank];
    float unary[selector_rows][selector_candidates];
    int32_t replay[selector_rows];
    SelectorResult result;
};
static_assert(sizeof(SelectorResult) == 412);
static_assert(sizeof(SelectorWorkspace) == 101296);

inline constexpr size_t selector_workspace_bytes() noexcept { return sizeof(SelectorWorkspace); }
inline constexpr size_t selector_result_bytes() noexcept { return sizeof(SelectorResult); }

// Returns a cudaError_t encoded as int. No allocation, synchronization, RNG, acceptance loop,
// exceptions, or process exit. The caller owns serialization, stream completion, and poisoning.
// Input: BF16 [3,2560], FP32 [3,248320], BF16 weights [256,2560]/[248320,256].
// The supplied uniforms are used only for temperature > 0 and must be finite in [0,1).
int selector_select(const uint16_t* hidden, const float* logits, const uint16_t* hidden_projection,
                    const uint16_t* predecessor_codebook, const uint16_t* successor_codebook,
                    int32_t anchor, float temperature, float uniform0, float uniform1, float uniform2,
                    SelectorWorkspace* workspace, void* stream);

} // namespace strata::kernels::dflash2
