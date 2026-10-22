// Run only in a guarded measurement window. Tests the actual mapped publication kernel.
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <bit>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

void ck(cudaError_t status) { if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status)); }
int main() {
    try {
        constexpr int hidden = 2560, capacity = 2047;
        constexpr size_t n = size_t(capacity) * hidden;
        float* source;
        uint16_t *host, *mapped;
        uint32_t *enabled, *device_enabled;
        cudaStream_t stream;
        ck(cudaStreamCreate(&stream));
        ck(cudaMalloc(&source, n * 4 * sizeof(float)));
        ck(cudaHostAlloc(&host, (n + 16) * sizeof(uint16_t), cudaHostAllocMapped));
        ck(cudaHostGetDevicePointer(&mapped, host, 0));
        ck(cudaHostAlloc(&enabled, sizeof(uint32_t), cudaHostAllocMapped));
        ck(cudaHostGetDevicePointer(&device_enabled, enabled, 0));
        std::vector<float> input(n * 4);
        size_t checked = 0;
        for (int rows : {1, 3, 257, capacity}) {
            for (int row = 0; row < rows; ++row) for (int col = 0; col < hidden; ++col) {
                float a = float(row + col % 31) / 16, b = -a / 2, c = 2, d = -1;
                switch (col % 17) {
                case 0: a = 16777216; b = 1; c = -16777216; d = 4; break;
                case 1: a = b = c = d = 1.00390625f; break;
                case 2: a = b = c = d = 1.01171875f; break;
                case 3: a = b = c = d = -0.0f; break;
                case 4: a = b = c = d = std::bit_cast<float>(uint32_t(0x00008001)); break;
                case 5: a = b = std::numeric_limits<float>::max(); break;
                case 6: a = std::numeric_limits<float>::quiet_NaN(); break;
                case 7: a = std::numeric_limits<float>::infinity(); break;
                }
                size_t i = size_t(row) * 4 * hidden + col;
                input[i] = a; input[i + hidden] = b; input[i + 2 * hidden] = c; input[i + 3 * hidden] = d;
            }
            ck(cudaMemcpy(source, input.data(), size_t(rows) * hidden * 4 * sizeof(float), cudaMemcpyHostToDevice));
            cudaGraph_t graph;
            cudaGraphExec_t executable;
            ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
            strata::kernels::tap_reduce_publish(source, mapped + 8, rows, hidden, device_enabled, stream);
            ck(cudaStreamEndCapture(stream, &graph));
            ck(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
            for (int replay = 0; replay < 3; ++replay) {
                std::fill_n(host, n + 16, uint16_t(0x55aa));
                *enabled = replay != 0;
                if (replay == 2) {
                    input[0] = 4; input[hidden] = 4; input[2 * hidden] = 4; input[3 * hidden] = 4;
                    ck(cudaMemcpy(source, input.data(), size_t(rows) * hidden * 4 * sizeof(float), cudaMemcpyHostToDevice));
                }
                ck(cudaGraphLaunch(executable, stream));
                ck(cudaStreamSynchronize(stream));
                for (size_t i = 0; i < n + 16; ++i) {
                    uint16_t expected = 0x55aa;
                    if (replay && i >= 8 && i < 8 + size_t(rows) * hidden) {
                        size_t j = i - 8, base = (j / hidden) * 4 * hidden + j % hidden;
                        volatile float ab = input[base] + input[base + hidden];
                        volatile float abc = ab + input[base + 2 * hidden];
                        volatile float abcd = abc + input[base + 3 * hidden];
                        expected = strata::kernels::bf16_from_f32(abcd * 0.25f);
                        // NaN payload is immaterial: both paths must preserve nonfinite rejection.
                        if ((expected & 0x7f80) == 0x7f80 && (host[i] & 0x7f80) == 0x7f80) continue;
                    }
                    if (host[i] != expected) throw std::runtime_error("mapped reduction/replay/extent mismatch");
                    ++checked;
                }
            }
            ck(cudaGraphExecDestroy(executable)); ck(cudaGraphDestroy(graph));
        }
        // Two adjacent chunk publications with distinct source/destination offsets.
        std::fill_n(host, n + 16, uint16_t(0x55aa));
        *enabled = 1;
        strata::kernels::tap_reduce_publish(source + 4 * hidden, mapped + 8 + hidden, 1, hidden, device_enabled, stream);
        strata::kernels::tap_reduce_publish(source + 8 * hidden, mapped + 8 + 2 * hidden, 1, hidden, device_enabled, stream);
        ck(cudaStreamSynchronize(stream));
        for (size_t i = 0; i < n + 16; ++i) {
            uint16_t expected = 0x55aa;
            if (i >= 8 + hidden && i < 8 + 3 * hidden) {
                size_t j = i - 8, base = (j / hidden) * 4 * hidden + j % hidden;
                volatile float ab = input[base] + input[base + hidden];
                volatile float abc = ab + input[base + 2 * hidden];
                volatile float abcd = abc + input[base + 3 * hidden];
                expected = strata::kernels::bf16_from_f32(abcd * 0.25f);
                if ((expected & 0x7f80) == 0x7f80 && (host[i] & 0x7f80) == 0x7f80) continue;
            }
            if (host[i] != expected) throw std::runtime_error("chunk source/destination offset mismatch");
            ++checked;
        }
        ck(cudaFree(source)); ck(cudaFreeHost(host)); ck(cudaFreeHost(enabled)); ck(cudaStreamDestroy(stream));
        std::printf("capture reduction: %zu exact comparisons passed, including disabled/repeated graph replay\n", checked);
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
