// #217 diagnostic only: retain the last QSA layer's real score/step inputs after
// the existing solo-verification drain. No extra device allocation or graph node.
#pragma once
#include "strata/kernels/qsa.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

namespace strata::core {
inline bool qsa_score_snapshot(int device, int64_t position, int rows, int64_t stride,
                               const float* scores, const int32_t* steps, std::string& error) {
    static const char* prefix = std::getenv("STRATA_QSA_SCORE_SNAPSHOT");
    static bool done[16] = {};
    if (!prefix || position < 131071 || rows <= 0 || rows > 32 || device < 0 || device >= 16 || done[device]) return true;
    if (!std::filesystem::path(prefix).is_absolute()) {
        error = "QSA score snapshot needs an absolute create-only prefix";
        return false;
    }
    int major = 0, minor = 0;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess) {
        error = "QSA score snapshot cannot read device architecture";
        return false;
    }
    if (major != 7 || minor != 0) { done[device] = true; return true; }
    std::vector<int32_t> st((size_t) rows * kernels::kStepCount);
    std::vector<float> sc((size_t) rows * stride);
    if (cudaMemcpy(st.data(), steps, st.size() * sizeof(int32_t), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(sc.data(), scores, sc.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        error = "QSA score snapshot readback failed";
        return false;
    }
    const std::string path = std::string(prefix) + ".cuda" + std::to_string(device) + ".bin";
    std::FILE* file = std::fopen(path.c_str(), "wbx");
    if (!file) { error = "QSA score snapshot cannot create " + path; return false; }
    const int64_t header[3] = {rows, stride, kernels::kStepCount};
    bool ok = std::fwrite(header, sizeof(header), 1, file) == 1 &&
              std::fwrite(st.data(), sizeof(int32_t), st.size(), file) == st.size() &&
              std::fwrite(sc.data(), sizeof(float), sc.size(), file) == sc.size();
    ok = std::fclose(file) == 0 && ok;
    if (!ok) { error = "QSA score snapshot write failed"; return false; }
    done[device] = true;
    std::fprintf(stderr, "QSA_SCORE_SNAPSHOT device=%d rows=%d stride=%lld path=%s\n",
                 device, rows, (long long) stride, path.c_str());
    return true;
}
}  // namespace strata::core
