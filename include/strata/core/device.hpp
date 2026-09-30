// include/strata/core/device.hpp - P2.S1: the device arena and the runtime's device facts.
//
// `DeviceArena` is ONE cudaMalloc per planner region with bump sub-allocation below it and no frees.  That is
// not a simplification for the first version: the memory plan from P1.S9 is fixed at startup, so the set of
// regions and their sizes is known before anything is allocated, and an allocator that can free would be
// solving a problem the engine does not have while adding fragmentation and failure modes it does.
//
// The reason to get this in early is that the VRAM budget is the binding constraint of the whole design
// (5.95 GB pooled between KV and the expert cache, 33.97 GB of experts in DRAM).  A runtime that discovers at
// token 4000 that it has overcommitted has already lost; the plan is printed against `cudaMemGetInfo` at
// startup so the discrepancy is visible immediately.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::core {

struct DeviceInfo {
    int ordinal = -1;
    std::string name;
    int cc_major = 0, cc_minor = 0;
    uint64_t total_bytes = 0;      // as reported by cudaMemGetInfo at query time
    uint64_t free_bytes = 0;
    int driver_version = 0, runtime_version = 0;
    int multi_processor_count = 0;
    std::string arch;              // HIP: gcnArchName without its feature suffix ("gfx1201"); empty on CUDA
};

// HIP builds: whether GPU `ordinal` can run this binary - its architecture must be one the binary was COMPILED
// for (STRATA_HIP_ARCHS, set by cmake/hip_backend.cmake) and it must run wave32.  "" when it can (or when there
// is no such device: the caller's own device errors apply), else the reason in a sentence.  A binary carried to
// another card would otherwise fail later with "invalid device function".  CUDA builds: always "".
std::string gpu_arch_problem(int ordinal);

// The GPU architectures this binary was compiled for ("gfx1100,gfx1201"); "" on CUDA builds.
const char* compiled_gpu_archs();

// The compute-capability floor this BINARY admits at run time.  A release build is sm_75 (Turing): the QSA
// scorer's tf32 mma has a portable fp32-FMA fallback below sm_80 and the tensor-core prompt kernels refuse, but
// nothing in the arithmetic needs more than that.  -DSTRATA_EXPERIMENTAL_SM60=ON lowers the floor to the Pascal
// generation that flag exists for; Volta (7.0) sits between that floor and the release one, so the same switch
// admits it.  The compile-time half of this policy is in CMakeLists.txt.
#if defined(STRATA_EXPERIMENTAL_SM60)
inline constexpr int kMinComputeCapability = 60;
#else
inline constexpr int kMinComputeCapability = 75;
#endif

// Why a card reporting `cc_major`.`cc_minor` cannot run this binary, or "" when it can - the sentence
// `device_info` throws with the device's name in front of it.  Header-only and CUDA-free on purpose: this policy
// decides whether an sm_70;sm_86 binary runs, and on a host with no GPU the only way to check it is to call it
// directly.  Keeping it here rather than inline in device.cu is what makes that check possible.
inline std::string compute_capability_problem(int cc_major, int cc_minor) {
    if (cc_major * 10 + cc_minor >= kMinComputeCapability) return "";
    return "reports compute capability " + std::to_string(cc_major) + "." + std::to_string(cc_minor) +
           "; Strata needs compute capability " + std::to_string(kMinComputeCapability / 10) + "." +
           std::to_string(kMinComputeCapability % 10) +
           (kMinComputeCapability < 75 ? " or newer in an experimental SM60 build"
                                       : " or newer (RTX 20 / 30 / 40 / 50 series)");
}

// Throws when there is no CUDA device.  The engine targets sm_120 specifically and must say so rather than
// run slowly on something else: `CMakeLists.txt` already refuses to COMPILE for another architecture, and
// this is the matching check at run time (a binary can be carried to a different machine).
DeviceInfo device_info(int ordinal = 0);

class CudaError : public std::runtime_error {
public:
    CudaError(const std::string& what, int code) : std::runtime_error(what), code_(code) {}
    int code() const { return code_; }

private:
    int code_;
};

// One cudaMalloc, bump-allocated below.  `poison` fills new allocations with a NaN-ish pattern in a debug
// build so that reading uninitialised VRAM gives a NaN rather than a plausible number - the same reasoning as
// the harness work in Phase 1: a wrong value that looks right is the expensive kind.
class DeviceArena {
public:
    explicit DeviceArena(uint64_t bytes, int ordinal = 0, bool poison = false);
    ~DeviceArena();
    DeviceArena(const DeviceArena&) = delete;
    DeviceArena& operator=(const DeviceArena&) = delete;

    // `align` must be a power of two; 256 keeps every sub-allocation at a sector boundary.
    void* alloc(uint64_t bytes, uint64_t align = 256);

    uint64_t capacity() const { return capacity_; }
    uint64_t used() const { return used_; }
    uint64_t peak() const { return used_; }        // no frees, so used IS the peak
    int ordinal() const { return ordinal_; }
    void* base() const { return base_; }

private:
    void* base_ = nullptr;
    uint64_t capacity_ = 0, used_ = 0;
    int ordinal_ = 0;
    bool poison_ = false;
};

}  // namespace strata::core
