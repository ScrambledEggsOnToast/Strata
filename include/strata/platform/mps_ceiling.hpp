// include/strata/platform/mps_ceiling.hpp - the enforced MPS client ceiling, if one was declared.
//
// A protected supervisor can launch this process inside an NVIDIA MPS client whose
// total device-memory budget is fixed at creation (`CUDA_MPS_PINNED_DEVICE_MEM_LIMIT`)
// and whose control pipe is `CUDA_MPS_PIPE_DIRECTORY`. Qualification on the pinned stack
// (`evidence/HET-017/clientcap-qualification.json`) established the semantics this
// module relies on:
//
//   * the limit is a TOTAL per-client device ceiling and it counts CUDA-internal
//     allocations, so `cudaMemoryAllocation` fails at the boundary rather than growing;
//   * the ceiling is fixed when the client is created and a later server-limit change
//     affects only clients created afterwards;
//   * inside an MPS client `cudaMemGetInfo` reports the REMAINING CLIENT BUDGET, not
//     device free memory. That is why `DeviceTelemetry` must be told which of the two
//     numbers it is looking at: using the residual budget as a ceiling double-counts
//     and refuses configurations the ceiling would actually admit.
//
// The declaration is the environment pair and nothing else. Both variables absent means
// no ceiling was declared and this module stays out of the way; exactly one present, or
// any malformed value, is refused rather than guessed at.
#pragma once

#include <cstdint>
#include <string>

namespace strata::platform {

/// One parsed, unverified declaration. `declared` is false when neither variable is set.
struct MpsCeiling {
    bool declared = false;
    int device = -1;              ///< device ordinal the ceiling applies to
    uint64_t cap_bytes = 0;       ///< total device bytes this client may allocate
    std::string pipe_directory;   ///< the daemon's control pipe directory
    bool limit_from_environment = false;  ///< true when the limit came from the env, not a default
};

/// Parse the declaration from the environment. Never touches the filesystem or CUDA.
/// Returns false with `err` set when the declaration is present but unusable.
bool mps_ceiling_from_environment(MpsCeiling& out, std::string& err);

/// Parse one `CUDA_MPS_PINNED_DEVICE_MEM_LIMIT`-style value: `<device>=<value>[,<device>=<value>...]`
/// with an optional K/M/G/T suffix (decimal or `i`-binary), compared case-insensitively.
/// `wanted_device` selects the entry. Returns false when no entry matches or the value is malformed.
bool mps_parse_limit(const std::string& text, int wanted_device, uint64_t& bytes, std::string& err);

/// Verify the declared ceiling against the filesystem: the pipe directory must exist, be a
/// directory, not a symbolic link, and be owned by the effective uid (or root), because a
/// ceiling the process cannot reach is not in force. Pure stat/no CUDA.
bool mps_ceiling_pipe_identity(const MpsCeiling& ceiling, std::string& err);

/// The runtime half: given the numbers `cudaMemGetInfo` just returned and the ceiling the
/// supervisor declared, decide whether this process is really inside that budget.
///
/// Under MPS the reported free value is the remaining client budget, so it can never exceed
/// the ceiling; a value above it means the process is NOT constrained by it - which is the
/// measured failure mode of a client that carries the variable with no daemon running
/// (it silently uses the whole device). That case must refuse.
bool mps_ceiling_holds(uint64_t reported_free_bytes, uint64_t declared_cap_bytes,
                       std::string& err);

/// The device ceiling this process may plan against: the smaller of the driver-visible
/// total and the declared cap. Returns `total_bytes` unchanged when no cap is declared.
uint64_t mps_device_ceiling(uint64_t total_bytes, uint64_t declared_cap_bytes, bool declared);

}  // namespace strata::platform
