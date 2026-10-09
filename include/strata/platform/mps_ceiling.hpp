// include/strata/platform/mps_ceiling.hpp - the enforced MPS client ceilings, if one was declared.
//
// A protected supervisor can launch this process inside an NVIDIA MPS client whose total
// device-memory budget is fixed at creation (`CUDA_MPS_PINNED_DEVICE_MEM_LIMIT`) and whose
// control pipe is `CUDA_MPS_PIPE_DIRECTORY`. Qualification on the pinned stack
// (`evidence/HET-017/clientcap-qualification.json`) established the semantics this module
// relies on:
//
//   * the limit is a TOTAL per-client per-device ceiling and it counts CUDA-internal
//     allocations, so `cudaMemoryAllocation` fails at the boundary rather than growing;
//   * the ceiling is fixed when the client is created and a later server-limit change
//     affects only clients created afterwards;
//   * inside an MPS client `cudaMemGetInfo` reports the REMAINING CLIENT BUDGET,
//     not physical device free memory. Planned demand must fit that residual; the
//     separate physical envelope is the full cap plus an explicit outside allowance.
//
// A declaration requires the limit, the control pipe, and a CUDA_VISIBLE_DEVICES whose
// entries are complete GPU UUIDs - one, or an ordered list with exactly one cap per entry
// (`CUDA_MPS_PINNED_DEVICE_MEM_LIMIT` is the full contiguous `ordinal=limit` list, e.g.
// `0=10240M,1=10240M`: no missing, duplicate or extra identifiers). One pipe serves every
// device, and one client PID attaches to exactly one server for all of them. The supervisor
// pins daemon and client to the SAME UUID order; driver550 control commands use ordinals
// within that order. Neither limit nor pipe present means no declaration.
// Readback verifies attachment, but daemon limits describe future clients, not an immutable
// numeric query of this client's creation-time caps. The physical sample is no reservation.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace strata::platform {

/// One device's parsed, unverified ceiling. `device` is the ordinal in CUDA_VISIBLE_DEVICES
/// order (the client's CUDA ordinal), and `gpu_uuid` is that entry's complete identity.
struct MpsDeviceCeiling {
    int device = -1;              ///< ordinal in CUDA_VISIBLE_DEVICES order, 0..N-1
    uint64_t cap_bytes = 0;       ///< total device bytes this client may allocate there
    std::string gpu_uuid;         ///< complete CUDA_VISIBLE_DEVICES identity, GPU-...
};

/// The parsed, unverified declaration for every visible device. `declared` is false when
/// limit and pipe are absent. `devices` is ordered by CUDA_VISIBLE_DEVICES ordinal.
struct MpsCeiling {
    bool declared = false;
    std::string pipe_directory;   ///< the daemon's control pipe directory (shared by all devices)
    bool limit_from_environment = false;  ///< true when the limit came from the env, not a default
    std::vector<MpsDeviceCeiling> devices;

    size_t size() const { return devices.size(); }
    const MpsDeviceCeiling* find_device(int ordinal) const {
        for (const MpsDeviceCeiling& d : devices)
            if (d.device == ordinal) return &d;
        return nullptr;
    }
};

/// Parse the declaration from the environment. Never touches the filesystem or CUDA.
/// Returns false with `err` set when the declaration is present but unusable.
bool mps_ceiling_from_environment(MpsCeiling& out, std::string& err);

/// Parse the supported declaration list `0=<v0>[,1=<v1>...]` in its entirety: ordinals must
/// be contiguous from 0 in ascending order, each exactly once, and every value a positive
/// byte count. K/M/G/T, optionally followed by i and/or B, are case-insensitive binary
/// multipliers. A bare value, a gap, a duplicate, a descending or non-zero-based ordinal refuse.
bool mps_parse_limit(const std::string& text, std::vector<uint64_t>& caps, std::string& err);

/// Verify the declared ceiling against the filesystem: the pipe directory must exist, be a
/// directory, not a symbolic link, and be owned by the effective uid (or root), because a
/// ceiling the process cannot reach is not in force. Pure stat/no CUDA.
bool mps_ceiling_pipe_identity(const MpsCeiling& ceiling, std::string& err);

/// Necessary residual-budget check, NOT proof of MPS attachment. A value above the cap
/// refuses; a smaller value alone cannot distinguish an attached client from a busy GPU.
bool mps_ceiling_holds(uint64_t reported_free_bytes, uint64_t declared_cap_bytes,
                       std::string& err);

/// One device's readback receipt. The physical sample precedes this client's CUDA contexts.
/// Default and server caps are current future-client settings, not a query of the client's
/// immutable creation-time cap.
struct MpsDeviceVerification {
    int device = -1;                      ///< ordinal in CUDA_VISIBLE_DEVICES order
    std::string gpu_uuid;                 ///< the declared identity verified after context
    uint64_t cap_bytes = 0;               ///< this device's declared cap
    uint64_t physical_free_bytes = 0;
    uint64_t outside_client_allowance_bytes = 0;
    bool physical_free_measured = false;
    uint64_t default_cap_bytes = 0;
    uint64_t server_cap_bytes = 0;
    uint64_t residual_free_bytes = 0;     ///< cudaMemGetInfo residual, when its context existed
};

/// Readback receipt for the whole declaration. The single-device fields are retained as
/// single-specific output because retained inspectors consume them; they mirror `devices`
/// front row, which for a single-device declaration is the only row. `devices` is ordered
/// by CUDA_VISIBLE_DEVICES ordinal.
struct MpsVerification {
    uint64_t physical_free_bytes = 0;
    uint64_t outside_client_allowance_bytes = 0;
    bool physical_free_measured = false;
    int server_pid = 0;
    int client_pid = 0;
    uint64_t default_cap_bytes = 0;
    uint64_t server_cap_bytes = 0;
    std::vector<MpsDeviceVerification> devices;
};

/// Before ANY CUDA call: verify the pipe and each device's daemon default, then measure each
/// exact GPU's physical free memory and require full cap + positive outside allowance to fit
/// per device. Linux queries are read-only, fixed-executable, and share one finite phase
/// deadline, with at most 64 KiB per reply, 32 servers and 4096 clients per server.
bool mps_verify_before_cuda(const MpsCeiling& ceiling, uint64_t outside_client_allowance_bytes,
                            MpsVerification& verification, std::string& err);

/// After CUDA0 context creation, BEFORE allocations/probes: verify the ACTUAL CUDA UUID of
/// every declared device, device 0's residual, this PID's membership in exactly one MPS
/// server, and that server's per-device cap - each readback uses the declared
/// device ordinal within the supervisor-pinned common UUID order. Retains the pre-context
/// physical sample. `actual_uuids` must carry one complete UUID per declared device, in order.
bool mps_verify_after_cuda(const MpsCeiling& ceiling, const std::vector<std::string>& actual_uuids,
                           uint64_t primary_residual_free_bytes, MpsVerification& verification,
                           std::string& err);

/// The device ceiling this process may plan against: the smaller of the driver-visible
/// total and the declared cap. Returns `total_bytes` unchanged when no cap is declared.
uint64_t mps_device_ceiling(uint64_t total_bytes, uint64_t declared_cap_bytes, bool declared);

}  // namespace strata::platform
