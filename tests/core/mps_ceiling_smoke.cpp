// Protected-supervisor-only smoke of the production MPS verification module.
// This is deliberately not a CTest: a context is opened only after daemon/physical checks.
#include "strata/platform/mps_ceiling.hpp"
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {
void format_uuid(char (&out)[41], const cudaDeviceProp& prop) {
    std::memcpy(out, "GPU-", 4);
    size_t at = 4;
    constexpr char hex[] = "0123456789abcdef";
    for (int i = 0; i < 16; ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) out[at++] = '-';
        const auto byte = static_cast<unsigned char>(prop.uuid.bytes[i]);
        out[at++] = hex[byte >> 4];
        out[at++] = hex[byte & 15];
    }
    out[at] = '\0';
}
}  // namespace

int main() {
    const char* supervised = std::getenv("STRATA_SUPERVISED");
    if (!supervised || std::strcmp(supervised, "1") != 0) {
        std::fprintf(stderr, "mps_ceiling_smoke requires the protected supervisor\n");
        return 2;
    }
    using namespace strata::platform;
    constexpr uint64_t outside = 64ull << 20; // explicit commitment, not a measured maximum
    MpsCeiling ceiling;
    MpsVerification receipt;
    std::string error;
    if (!mps_ceiling_from_environment(ceiling, error) || !ceiling.declared) {
        std::fprintf(stderr, "declaration: %s\n", error.c_str());
        return 1;
    }
    MpsCeiling mismatch = ceiling;
    for (auto& d : mismatch.devices) --d.cap_bytes;
    if (mps_verify_before_cuda(mismatch, outside, receipt, error)) {
        std::fprintf(stderr, "mismatched daemon cap was accepted\n");
        return 1;
    }
    std::printf("NEGATIVE daemon_mismatch refused=%s\n", error.c_str());
    error.clear();
    if (mps_verify_before_cuda(ceiling, UINT64_MAX, receipt, error)) {
        std::fprintf(stderr, "overflowing physical envelope was accepted\n");
        return 1;
    }
    std::printf("NEGATIVE physical_envelope refused=%s\n", error.c_str());
    error.clear();
    if (!mps_verify_before_cuda(ceiling, outside, receipt, error)) {
        std::fprintf(stderr, "before CUDA: %s\n", error.c_str());
        return 1;
    }
    {
        std::vector<std::string> declared_uuids;
        for (const auto& d : ceiling.devices) declared_uuids.push_back(d.gpu_uuid);
        if (mps_verify_after_cuda(ceiling, declared_uuids, 0, receipt, error)) {
            std::fprintf(stderr, "unattached own PID was accepted\n");
            return 1;
        }
    }
    std::printf("NEGATIVE unattached_own_pid refused=%s\n", error.c_str());
    error.clear();
    // Refresh the pre-context physical sample immediately before CUDA creates this client.
    if (!mps_verify_before_cuda(ceiling, outside, receipt, error)) {
        std::fprintf(stderr, "refreshed precontext: %s\n", error.c_str());
        return 1;
    }
    cudaDeviceProp properties{};
    size_t free_bytes = 0, total_bytes = 0;
    cudaError_t status = cudaSetDevice(0);
    if (status == cudaSuccess) status = cudaFree(nullptr);
    if (status == cudaSuccess) status = cudaGetDeviceProperties(&properties, 0);
    if (status == cudaSuccess) status = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "context: %s\n", cudaGetErrorString(status));
        return 1;
    }
    // Every declared device's actual CUDA identity, before the first allocation.
    std::vector<std::string> actual_uuids;
    for (size_t i = 0; i < ceiling.devices.size(); ++i) {
        cudaDeviceProp device_properties{};
        if (cudaGetDeviceProperties(&device_properties, (int) i) != cudaSuccess) {
            std::fprintf(stderr, "device %zu properties: %s\n", i,
                         cudaGetErrorString(cudaGetLastError()));
            return 1;
        }
        char uuid[41];
        format_uuid(uuid, device_properties);
        actual_uuids.emplace_back(uuid);
    }
    if (!mps_verify_after_cuda(ceiling, actual_uuids, free_bytes, receipt, error)) {
        std::fprintf(stderr, "after CUDA: %s\n", error.c_str());
        return 1;
    }
    const MpsVerification verified = receipt;
    {
        std::vector<std::string> foreign(ceiling.devices.size(),
                                         "GPU-00000000-0000-0000-0000-000000000000");
        if (mps_verify_after_cuda(ceiling, foreign, free_bytes, receipt, error)) {
            std::fprintf(stderr, "wrong CUDA UUID was accepted\n");
            return 1;
        }
    }
    std::printf("NEGATIVE wrong_cuda_uuid refused=%s\n", error.c_str());
    receipt = verified;
    error.clear();
    if (mps_verify_after_cuda(ceiling, actual_uuids, ceiling.devices.front().cap_bytes + 1, receipt, error)) {
        std::fprintf(stderr, "residual above declared cap was accepted\n");
        return 1;
    }
    std::printf("NEGATIVE residual_above_cap refused=%s\n", error.c_str());
    void* buffer = nullptr;
    status = cudaMalloc(&buffer, 64ull << 20);
    if (status == cudaSuccess) status = cudaMemset(buffer, 0x5a, 64ull << 20);
    if (status == cudaSuccess) status = cudaDeviceSynchronize();
    const cudaError_t released = buffer ? cudaFree(buffer) : cudaSuccess;
    if (status != cudaSuccess || released != cudaSuccess) {
        std::fprintf(stderr, "bounded allocation/release failed: %s / %s\n",
                     cudaGetErrorString(status), cudaGetErrorString(released));
        return 1;
    }
    std::printf("MPS_MODULE_SMOKE uuid=%s client_pid=%d server_pid=%d cap_bytes=%llu "
                "default_cap_bytes=%llu server_cap_bytes=%llu physical_free_bytes=%llu "
                "outside_client_allowance_bytes=%llu residual_free_bytes=%llu devices=%zu "
                "negative_controls=5 allocated_bytes=67108864 failures=0\n",
                actual_uuids.front().c_str(), verified.client_pid, verified.server_pid,
                (unsigned long long)ceiling.devices.front().cap_bytes,
                (unsigned long long)verified.default_cap_bytes, (unsigned long long)verified.server_cap_bytes,
                (unsigned long long)verified.physical_free_bytes,
                (unsigned long long)verified.outside_client_allowance_bytes, (unsigned long long)free_bytes,
                ceiling.devices.size());
    std::fflush(stdout);
    std::this_thread::sleep_for(std::chrono::seconds(6)); // supervisor observes the real own-PID attachment
    return 0;
}
