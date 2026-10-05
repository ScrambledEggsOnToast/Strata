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
//   * inside an MPS client `cudaMemGetInfo` reports the REMAINING CLIENT BUDGET,
//     not physical device free memory. Planned demand must fit that residual; the
//     separate physical envelope is the full cap plus an explicit outside allowance.
//
// A declaration requires the limit, control pipe and a complete single GPU UUID in
// CUDA_VISIBLE_DEVICES. Neither limit nor pipe present means no declaration. Readback
// verifies attachment, but daemon limits describe future clients, not an immutable
// numeric query of this client's creation-time cap. The physical sample is no reservation.
#pragma once

#include <cstdint>
#include <string>

namespace strata::platform {

/// One parsed, unverified declaration. `declared` is false when limit and pipe are absent.
struct MpsCeiling {
    bool declared = false;
    int device = -1;              ///< device ordinal the ceiling applies to
    uint64_t cap_bytes = 0;       ///< total device bytes this client may allocate
    std::string pipe_directory;   ///< the daemon's control pipe directory
    std::string gpu_uuid;         ///< complete CUDA_VISIBLE_DEVICES identity, GPU-...
    bool limit_from_environment = false;  ///< true when the limit came from the env, not a default
};

/// Parse the declaration from the environment. Never touches the filesystem or CUDA.
/// Returns false with `err` set when the declaration is present but unusable.
bool mps_ceiling_from_environment(MpsCeiling& out, std::string& err);

/// Parse the supported single-device declaration `0=<positive value>` in its entirety.
/// K/M/G/T, optionally followed by i and/or B, are case-insensitive binary multipliers.
/// Only `wanted_device == 0` is supported; bare values and multi-device lists refuse.
bool mps_parse_limit(const std::string& text, int wanted_device, uint64_t& bytes, std::string& err);

/// Verify the declared ceiling against the filesystem: the pipe directory must exist, be a
/// directory, not a symbolic link, and be owned by the effective uid (or root), because a
/// ceiling the process cannot reach is not in force. Pure stat/no CUDA.
bool mps_ceiling_pipe_identity(const MpsCeiling& ceiling, std::string& err);

/// Necessary residual-budget check, NOT proof of MPS attachment. A value above the cap
/// refuses; a smaller value alone cannot distinguish an attached client from a busy GPU.
bool mps_ceiling_holds(uint64_t reported_free_bytes, uint64_t declared_cap_bytes,
                       std::string& err);

/// Readback receipt. The physical sample precedes this client's CUDA context. Default
/// and server caps are current future-client settings, not a query of its immutable cap.
struct MpsVerification {
    uint64_t physical_free_bytes = 0;
    uint64_t outside_client_allowance_bytes = 0;
    bool physical_free_measured = false;
    int server_pid = 0;
    int client_pid = 0;
    uint64_t default_cap_bytes = 0;
    uint64_t server_cap_bytes = 0;
};

/// Before ANY CUDA call: verify the pipe and daemon default, then measure the exact
/// GPU's physical free memory and require full cap + positive outside allowance to fit.
/// Linux queries are read-only, fixed-executable, 3-second phase-deadline bounded, with
/// at most 64 KiB per reply, 32 servers and 4096 clients per server.
bool mps_verify_before_cuda(const MpsCeiling& ceiling, uint64_t outside_client_allowance_bytes,
                            MpsVerification& verification, std::string& err);

/// After CUDA0 context creation, BEFORE allocations/probes: verify actual UUID, residual,
/// this PID's membership in exactly one MPS server, and that server's device-0 limit.
/// Retains the pre-context physical sample. No declaration skips verification and queries.
bool mps_verify_after_cuda(const MpsCeiling& ceiling, const std::string& actual_gpu_uuid,
                           uint64_t reported_free_bytes, MpsVerification& verification,
                           std::string& err);

/// The device ceiling this process may plan against: the smaller of the driver-visible
/// total and the declared cap. Returns `total_bytes` unchanged when no cap is declared.
uint64_t mps_device_ceiling(uint64_t total_bytes, uint64_t declared_cap_bytes, bool declared);

}  // namespace strata::platform
