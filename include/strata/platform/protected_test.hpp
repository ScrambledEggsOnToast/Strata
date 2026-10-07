#pragma once

#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#if defined(__linux__)
#include <unistd.h>
#endif

namespace strata::platform {

// Short-lived test clients must remain alive until the supervisor has pinned
// their actual CUDA client PID. Each protected run owns exactly one client.
inline bool acknowledge_protected_test() {
#if defined(__linux__)
    const char* supervised = std::getenv("STRATA_SUPERVISED");
    if (!supervised || std::strcmp(supervised, "1") != 0) return true;
    const char* snapshot = std::getenv("STRATA_ADMISSION_SNAPSHOT");
    if (!snapshot) {
        std::fprintf(stderr, "missing protected admission path\n");
        return false;
    }
    const std::string path(snapshot);
    const auto slash = path.find_last_of('/');
    if (slash == std::string::npos) return false;
    const std::string receipt = path.substr(0, slash + 1) + "mps-client";
    const auto status = cudaFree(nullptr);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "protected context initialization: %s\n", cudaGetErrorString(status));
        return false;
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (std::chrono::steady_clock::now() < deadline) {
        if (FILE* file = std::fopen(receipt.c_str(), "r")) {
            long pid = 0;
            const bool own = std::fscanf(file, "%ld", &pid) == 1 && pid == static_cast<long>(getpid());
            std::fclose(file);
            if (own) {
                std::printf("protected MPS client acknowledged: %ld\n", static_cast<long>(getpid()));
                std::fflush(stdout);
                return true;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    std::fprintf(stderr, "protected MPS acknowledgement timed out\n");
    return false;
#else
    return true;
#endif
}

} // namespace strata::platform
