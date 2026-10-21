// One physical verifier frame: request-owned rows in contiguous slot groups.
#pragma once

#include "strata/core/dflash2_capture.hpp"

#include <cstdint>

namespace strata::core::dflash2 {
struct BatchCaptureRow {
    int slot = -1;
    CaptureOwner owner;
    int64_t position = 0;
};
} // namespace strata::core::dflash2
