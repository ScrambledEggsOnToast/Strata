// CPU-share host payload: CUDA-free sizing and the exact heap arrays used by prefill.
#pragma once

#include "strata/kernels/cpu/pool.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <utility>

namespace strata::prefill {

inline constexpr int64_t kCpuShareRoutes = 10;

inline double cpu_share_value(const char* value) {
    if (value == nullptr) return -2.0;  // unset: the caller's resolved default
    if (std::strcmp(value, "auto") == 0) return -1.0;
    return std::clamp(std::atof(value), 0.0, 1.0);
}

inline uint64_t cpu_share_max_rows(int64_t tokens, int64_t experts) {
    // Only experts with at most MAXT routed tokens are eligible for the CPU pool.
    return std::min((uint64_t) tokens * kCpuShareRoutes,
                    (uint64_t) experts * kernels::cpu::MAXT);
}

inline uint64_t cpu_share_host_bytes(int64_t tokens, int64_t experts) {
    if (tokens <= 0 || tokens > INT32_MAX / kCpuShareRoutes || experts <= 0 || experts > kernels::cpu::NE)
        return UINT64_MAX;
    const uint64_t t = (uint64_t) tokens, e = (uint64_t) experts;
    // 0.1.44's CPU-share rows carry their own row-scale (`cpu_ones`, all ones: the CPU's rows carry s_down
    // already): `rows * 2` floats, grown with the rows' headroom and never counted here before the merge.
    return (t + 2 * cpu_share_max_rows(tokens, experts)) * kernels::cpu::H * sizeof(float)
        + 2 * cpu_share_max_rows(tokens, experts) * sizeof(float)
        + t * (kernels::cpu::kNativeActBytes + sizeof(kernels::cpu::ActQ) + sizeof(char))
        + e * (sizeof(kernels::cpu::ExpertJobMulti) + sizeof(std::pair<int32_t, int32_t>)
               + sizeof(char) + sizeof(const uint8_t*));
}

// One sizing rule drives cudaHostAlloc and the hermetic partner's allocation backend.
// Free before growing: the old and new pinned payloads never coexist.
template<class Allocate, class Release>
bool cpu_share_grow_pinned(float*& data, size_t& capacity, size_t want, size_t headroom,
                           Allocate allocate, Release release) {
    if (want <= capacity) return true;
    if (!headroom || want > SIZE_MAX / headroom / sizeof(float)) return false;
    if (data) release(data);
    data = nullptr;
    capacity = 0;
    const size_t count = want * headroom;
    if (!allocate(data, count * sizeof(float))) return false;
    capacity = count;
    return true;
}

// Lazy exact-capacity arrays replace vector growth/copying. Both activation formats
// can remain live across different layers; admission charges both, once per stage.
struct CpuShareHeap {
    std::unique_ptr<uint8_t[]> nact;
    std::unique_ptr<kernels::cpu::ActQ[]> actq;
    std::unique_ptr<kernels::cpu::ExpertJobMulti[]> jobs;
    std::unique_ptr<char[]> need, on_cpu;
    std::unique_ptr<const uint8_t*[]> blobs;
    std::unique_ptr<std::pair<int32_t, int32_t>[]> candidates;
    size_t nact_tokens = 0, actq_tokens = 0, work_tokens = 0, job_experts = 0, route_experts = 0;

    void ensure_nact(size_t tokens) {
        if (tokens <= nact_tokens) return;
        nact.reset();
        nact.reset(new uint8_t[tokens * kernels::cpu::kNativeActBytes]);
        nact_tokens = tokens;
    }
    void ensure_actq(size_t tokens) {
        if (tokens <= actq_tokens) return;
        actq.reset();
        actq.reset(new kernels::cpu::ActQ[tokens]);
        actq_tokens = tokens;
    }
    void ensure_work(size_t tokens, size_t experts) {
        if (tokens > work_tokens) {
            need.reset();
            need.reset(new char[tokens]);
            work_tokens = tokens;
        }
        if (experts > job_experts) {
            jobs.reset();
            jobs.reset(new kernels::cpu::ExpertJobMulti[experts]);
            job_experts = experts;
        }
    }
    void ensure_routes(size_t experts) {
        if (experts <= route_experts) return;
        on_cpu.reset(); blobs.reset(); candidates.reset();
        on_cpu.reset(new char[experts]);
        blobs.reset(new const uint8_t*[experts]);
        candidates.reset(new std::pair<int32_t, int32_t>[experts]);
        route_experts = experts;
    }
    uint64_t storage_bytes() const {
        return nact_tokens * kernels::cpu::kNativeActBytes + actq_tokens * sizeof(kernels::cpu::ActQ)
            + work_tokens * sizeof(char) + job_experts * sizeof(kernels::cpu::ExpertJobMulti)
            + route_experts * (sizeof(char) + sizeof(const uint8_t*) + sizeof(std::pair<int32_t, int32_t>));
    }
};

}  // namespace strata::prefill
