#pragma once

#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define STRATA_PHILOX_INLINE __host__ __device__ __forceinline__
#else
#define STRATA_PHILOX_INLINE inline
#endif

namespace strata::kernels {
namespace philox_detail {
// Preserve the engine's existing round constants and seed/counter mapping.
// This is the serving sampler's stream, not a new drafter-owned RNG.
STRATA_PHILOX_INLINE void round(uint32_t& c0, uint32_t& c1, uint32_t& c2, uint32_t& c3,
                                uint32_t k0, uint32_t k1) {
#if defined(__CUDA_ARCH__) || (defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__)
    const uint32_t hi0 = __umulhi(0x9E3779B9u, c0);
    const uint32_t hi1 = __umulhi(0xBB67AE85u, c2);
#else
    const uint32_t hi0 = uint32_t((uint64_t(0x9E3779B9u) * c0) >> 32);
    const uint32_t hi1 = uint32_t((uint64_t(0xBB67AE85u) * c2) >> 32);
#endif
    const uint32_t lo0 = 0x9E3779B9u * c0;
    const uint32_t lo1 = 0xBB67AE85u * c2;
    const uint32_t n0 = hi1 ^ c1 ^ k0;
    const uint32_t n2 = hi0 ^ c3 ^ k1;
    c0 = n0; c1 = lo1; c2 = n2; c3 = lo0;
}
} // namespace philox_detail

// Exactly 24 random mantissa bits, uniform in [0,1), never rounded to 1.
STRATA_PHILOX_INLINE float philox_uniform(uint64_t seed, uint64_t counter) {
    uint32_t c0 = uint32_t(counter), c1 = uint32_t(counter >> 32);
    uint32_t c2 = uint32_t(seed), c3 = uint32_t(seed >> 32);
    for (int i = 0; i < 10; ++i)
        philox_detail::round(c0, c1, c2, c3, uint32_t(i), 0u);
    return float(c0 >> 8) * (1.0f / 16777216.0f);
}
} // namespace strata::kernels

#undef STRATA_PHILOX_INLINE
