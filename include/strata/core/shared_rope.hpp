// Checked non-owning views of an immutable process-owned RoPE allocation. No runtime dependency.
#pragma once
#include "strata/kernels/rope_scaling.hpp"
#include <cstddef>
#include <cstdint>
#include <limits>

namespace strata::core {
struct SharedRopeView {
    float* cosine = nullptr;
    float* sine = nullptr;
    int64_t cells = 0;  // usable rows, NOT the process allocation's capacity
};

// The registry publishes this descriptor only after both uploads succeed. Neither
// descriptor nor view releases storage; the runtime owns it until process exit.
struct SharedRopeTable {
    float* data = nullptr;
    int64_t cells = 0, n_rot = 0;
    kernels::RopeScaling scaling;

    static const char* invalid_geometry(int64_t rows, int64_t width) noexcept {
        if (rows < 1 || rows > INT32_MAX) return "shared RoPE: requested rows must be in 1..INT32_MAX";
        if (width < 2 || width > INT32_MAX || width % 2)
            return "shared RoPE: rotary width must be positive, even and at most INT32_MAX";
        if ((uint64_t) rows > std::numeric_limits<size_t>::max() / sizeof(float) / 2 / (uint64_t)(width / 2))
            return "shared RoPE: table byte extent is not representable";
        return nullptr;
    }

    // Failure changes neither the published entry nor the caller's existing view.
    const char* borrow(int table_device, int device, int64_t rows, int64_t width,
                       const kernels::RopeScaling& requested, SharedRopeView& out) const noexcept {
        if (const char* why = invalid_geometry(rows, width)) return why;
        if (!data) return "shared RoPE: no published process-owned table on this device";
        if (table_device != device) return "shared RoPE: allocation device differs";
        if (width != n_rot) return "shared RoPE: rotary width differs";
        if (requested.type != scaling.type) return "shared RoPE: scaling type differs";
        if (requested.freq_base != scaling.freq_base) return "shared RoPE: frequency base differs";
        if (requested.factor != scaling.factor) return "shared RoPE: scaling factor differs";
        if (requested.freq_scale_in != scaling.freq_scale_in) return "shared RoPE: frequency scale differs";
        if (requested.orig_ctx != scaling.orig_ctx) return "shared RoPE: original context differs";
        if (requested.ext_factor != scaling.ext_factor) return "shared RoPE: YaRN extension factor differs";
        if (requested.attn_factor != scaling.attn_factor) return "shared RoPE: attention factor differs";
        if (requested.beta_fast != scaling.beta_fast) return "shared RoPE: YaRN fast beta differs";
        if (requested.beta_slow != scaling.beta_slow) return "shared RoPE: YaRN slow beta differs";
        if (rows > cells) return "shared RoPE: requested rows exceed immutable table capacity";
        // Cosine and sine are separate capacity-sized planes. A short borrow must
        // NEVER use its requested row count to locate the start of the sine plane.
        out = {data, data + (size_t) cells * (size_t)(n_rot / 2), rows};
        return nullptr;
    }
};
} // namespace strata::core
