// Explicit queued GPU fixture. Expose the retained kernels in this translation
// unit, without a test-only production API or a second implementation oracle.
#include "cuda/dflash2_forward.cu"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

using namespace strata::kernels;
using namespace strata::kernels::dflash2;
namespace {
constexpr size_t guard = 16;
constexpr size_t output_count = size_t(block) * query_width;
constexpr uint32_t status_sentinel = 0x20000000u;
void check(cudaError_t code) {
    if (code != cudaSuccess) {
        std::fprintf(stderr, "CUDA failure: %s\n", cudaGetErrorString(code));
        std::exit(1); // Do not unwind borrowed buffers after an unknown drain.
    }
}
void require(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr, "FAIL %s\n", message); std::exit(1); }
}
uint16_t finite_value(size_t i) {
    constexpr uint16_t values[]{0x3f80, 0xbf80, 0x4101, 0xc101, 0x3b81, 0xbb81,
                                0x0001, 0x8001, 0x3f7f, 0xbf7f, 0, 0x8000};
    return values[((i * 1664525u + (i >> 4) * 1013904223u) >> 9) % std::size(values)];
}
struct Fixture {
    std::vector<uint16_t> ring, values, a, b;
    std::vector<float> probabilities;
    uint16_t *dring = nullptr, *dvalues = nullptr, *da = nullptr, *db = nullptr;
    float* dp = nullptr;
    uint32_t *sa = nullptr, *sb = nullptr;
    cudaStream_t stream = nullptr;
    Fixture() : ring(kv_elements), values(size_t(block) * kv_width),
                a(output_count + 2 * guard), b(output_count + 2 * guard),
                probabilities(size_t(query_heads) * block * max_keys) {
        for (size_t i = 0; i < ring.size(); ++i) ring[i] = finite_value(i + 103);
        check(cudaMalloc(&dring, ring.size() * sizeof(uint16_t)));
        check(cudaMalloc(&dvalues, values.size() * sizeof(uint16_t)));
        check(cudaMalloc(&dp, probabilities.size() * sizeof(float)));
        check(cudaMalloc(&da, a.size() * sizeof(uint16_t)));
        check(cudaMalloc(&db, b.size() * sizeof(uint16_t)));
        check(cudaMalloc(&sa, sizeof(uint32_t))); check(cudaMalloc(&sb, sizeof(uint32_t)));
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        check(cudaMemcpyAsync(dring, ring.data(), ring.size() * sizeof(uint16_t), cudaMemcpyHostToDevice, stream));
        check(cudaStreamSynchronize(stream));
    }
    ~Fixture() {
        check(cudaStreamSynchronize(stream));
        check(cudaStreamDestroy(stream));
        check(cudaFree(dring)); check(cudaFree(dvalues)); check(cudaFree(dp));
        check(cudaFree(da)); check(cudaFree(db)); check(cudaFree(sa)); check(cudaFree(sb));
    }
    void reset() {
        check(cudaMemsetAsync(da, 0xa5, a.size() * sizeof(uint16_t), stream));
        check(cudaMemsetAsync(db, 0xa5, b.size() * sizeof(uint16_t), stream));
        check(cudaMemcpyAsync(sa, &status_sentinel, sizeof(status_sentinel), cudaMemcpyHostToDevice, stream));
        check(cudaMemcpyAsync(sb, &status_sentinel, sizeof(status_sentinel), cudaMemcpyHostToDevice, stream));
    }
    void compare(int rows, bool exceptional) {
        check(cudaStreamSynchronize(stream));
        check(cudaMemcpy(a.data(), da, a.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost));
        check(cudaMemcpy(b.data(), db, b.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost));
        uint32_t ra = 0, rb = 0;
        check(cudaMemcpy(&ra, sa, sizeof(ra), cudaMemcpyDeviceToHost));
        check(cudaMemcpy(&rb, sb, sizeof(rb), cudaMemcpyDeviceToHost));
        require(ra == rb && ra == (status_sentinel | (exceptional ? 1u : 0u)), "numeric status differs");
        const size_t active = size_t(rows) * query_width;
        for (size_t i = 0; i < a.size(); ++i) {
            const bool outside = i < guard || i >= guard + active;
            const bool both_nan = (a[i] & 0x7f80u) == 0x7f80u && (a[i] & 0x7fu) &&
                                  (b[i] & 0x7f80u) == 0x7f80u && (b[i] & 0x7fu);
            if ((outside && (a[i] != 0xa5a5u || b[i] != 0xa5a5u)) ||
                (a[i] != b[i] && !both_nan)) {
                std::fprintf(stderr, "value mismatch rows=%d index=%zu %04x %04x\n", rows, i, a[i], b[i]);
                std::exit(1);
            }
        }
    }
    void run(int rows, int context, int begin, int layer, int exceptional = 0) {
        const int frontier = begin + context, keys = context + rows;
        for (size_t i = 0; i < values.size(); ++i) values[i] = finite_value(i + 51);
        constexpr float weights[]{0.0f, 1.0f, 0.125f, 0.03125f, 1e-20f, 1e-30f};
        std::fill(probabilities.begin(), probabilities.end(), std::numeric_limits<float>::quiet_NaN());
        const size_t used = size_t(query_heads) * rows * keys;
        for (size_t i = 0; i < used; ++i) probabilities[i] = weights[(i * 13 + (i >> 4)) % std::size(weights)];
        probabilities[size_t(context)] = 1.0f; // A changed block value must affect output [0,0].
        const size_t ring_at = (((size_t(layer) * 2 + 1) * kv_heads) * capacity + begin % capacity) * head_dim;
        const uint16_t saved = ring[ring_at];
        if (exceptional == 1) {
            require(context > 0, "nonfinite context test needs context");
            ring[ring_at] = 0x7fc0u;
            probabilities[0] = 0.0f; // Zero weight must not skip a nonfinite V read.
        }
        if (exceptional == 2) values[17] = 0x7f80u;
        if (exceptional == 3) probabilities[0] = std::numeric_limits<float>::quiet_NaN();
        check(cudaMemcpyAsync(dring + ring_at, ring.data() + ring_at, sizeof(uint16_t), cudaMemcpyHostToDevice, stream));
        check(cudaMemcpyAsync(dvalues, values.data(), values.size() * sizeof(uint16_t), cudaMemcpyHostToDevice, stream));
        check(cudaMemcpyAsync(dp, probabilities.data(), probabilities.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
        auto launch = [&](bool candidate) {
            if (candidate)
                value_spans_kernel<<<query_heads * rows, head_dim, 0, stream>>>(dp, dvalues, dring, db + guard,
                    layer, begin, frontier, rows, sb);
            else
                value_kernel<<<query_heads * rows, head_dim, 0, stream>>>(dp, dvalues, dring, da + guard,
                    layer, begin, frontier, rows, sa);
            check(cudaGetLastError());
        };
        auto production = [&] {
            check(static_cast<cudaError_t>(attention_values(dp, dvalues, dring, db + guard,
                layer, begin, frontier, rows, sb, stream)));
        };
        reset(); launch(false); launch(true); compare(rows, exceptional != 0);
        reset(); launch(false); production(); compare(rows, exceptional != 0);
        cudaGraph_t graph = nullptr; cudaGraphExec_t executable = nullptr;
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        launch(false); production();
        check(cudaStreamEndCapture(stream, &graph));
        check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        uint16_t previous = a[guard];
        for (float changed : {64.0f, -64.0f}) {
            values[0] = bf16_from_f32(changed);
            check(cudaMemcpyAsync(dvalues, values.data(), sizeof(uint16_t), cudaMemcpyHostToDevice, stream));
            reset(); check(cudaGraphLaunch(executable, stream)); compare(rows, exceptional != 0);
            if (!exceptional) require(a[guard] != previous, "graph replay ignored changed device input");
            previous = a[guard];
        }
        // Balanced reference/candidate order, finite bounded repetitions. These
        // kernel timings are screening evidence, not full-model throughput.
        if (rows == 4 && (context == 220 || context == capacity) && !exceptional) {
            cudaEvent_t start, end; check(cudaEventCreate(&start)); check(cudaEventCreate(&end));
            float times[2]{};
            for (int variant : {0, 1, 1, 0}) {
                check(cudaEventRecord(start, stream));
                for (int repeat = 0; repeat < 5; ++repeat) launch(variant != 0);
                check(cudaEventRecord(end, stream)); check(cudaEventSynchronize(end));
                float elapsed = 0; check(cudaEventElapsedTime(&elapsed, start, end));
                times[variant] += elapsed / 10.0f;
            }
            std::printf("{\"rows\":%d,\"context\":%d,\"begin\":%d,\"reference_ms\":%.6f,\"candidate_ms\":%.6f}\n",
                        rows, context, begin, times[0], times[1]);
            check(cudaEventDestroy(start)); check(cudaEventDestroy(end));
        }
        check(cudaGraphExecDestroy(executable)); check(cudaGraphDestroy(graph));
        ring[ring_at] = saved;
        check(cudaMemcpyAsync(dring + ring_at, ring.data() + ring_at, sizeof(uint16_t), cudaMemcpyHostToDevice, stream));
        check(cudaStreamSynchronize(stream));
        std::printf("{\"rows\":%d,\"context\":%d,\"begin\":%d,\"layer\":%d,\"nonfinite_case\":%d,\"exact\":true,\"graph_replay\":true,\"guards\":true}\n",
                    rows, context, begin, layer, exceptional);
    }
};
} // namespace
int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--help") == 0) {
        std::puts("Queued CUDA attention-value exactness and bounded timing fixture. Select the GPU with CUDA_VISIBLE_DEVICES; no arguments.");
        return 0;
    }
    require(argc == 1, "unsupported arguments");
    require(setenv("STRATA_DFLASH2_VALUE_SPANS", "1", 1) == 0, "could not enable candidate dispatch");
    Fixture fixture;
    for (int rows : {1, 4, 8}) {
        fixture.run(rows, 220, 0, 0);
        fixture.run(rows, 220, capacity - 3, 4);
        fixture.run(rows, capacity, 0, 0);
        fixture.run(rows, capacity, capacity - 3, 4);
    }
    fixture.run(1, 0, 0, 4);
    fixture.run(4, 1, capacity - 1, 0);
    for (int kind : {1, 2, 3}) fixture.run(4, 220, capacity - 3, 4, kind);
    return 0;
}
