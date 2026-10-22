#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace strata::kernels::dflash2 {
// Frozen before admission and construction. The default keeps the legacy mode.
bool context_blas_enabled() noexcept;
const char* context_projection_identity() noexcept;

// One serialized owner's context-only workspace; all model weights stay BF16.
// The caller drains its stream on success AND failure before releasing its lease.
class ContextBlas {
public:
    static constexpr size_t device_budget = size_t(128) << 20;
    static constexpr size_t host_budget = size_t(256) << 20;
    static constexpr size_t explicit_device_bytes = 91202816;
    struct Statistics {
        unsigned long long activation_rows = 0, raw_activation_rows = 0;
        unsigned long long weight_rows = 0, raw_weight_rows = 0;
        unsigned long long fallback_outputs = 0, projection_calls = 0;
    };
    explicit ContextBlas(int device);
    ~ContextBlas();
    ContextBlas(const ContextBlas&) = delete;
    ContextBlas& operator=(const ContextBlas&) = delete;
    // Enqueues only. Finite and exceptional results use the owner's OR status.
    // Supported shape: 128..2047 rows, K<=12800, N<=2560; no overlapping operands.
    bool project(const uint16_t* input, const uint16_t* weight, uint16_t* output,
                 int rows, int input_width, int output_width, uint32_t* status,
                 void* stream, std::string& error);
    // Diagnostic only, after the caller's existing completion fence. Counters
    // are enabled only by STRATA_DFLASH2_CONTEXT_CUBLAS_TRACE=1 at construction.
    Statistics statistics() const;
    void report_completed();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strata::kernels::dflash2
