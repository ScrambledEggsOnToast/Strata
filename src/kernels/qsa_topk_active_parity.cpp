// Capacity 262144, live prefill up to the register boundary, and the legacy fallbacks.
// Compare selected IDs only: unused output padding is not part of the API contract.
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_select.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace k = strata::kernels;
namespace {
void ck(cudaError_t e) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e));
        std::exit(2);
    }
}
struct Case { int64_t ctx, queries; int mode; };
float score_value(int mode, int64_t j, float random) {
    if (mode == 1) return 0.f;
    if (mode == 2) return float(j % 7);
    if (mode == 5) {
        const float edge[] = {-0.f, 0.f, -std::numeric_limits<float>::infinity(),
                              std::numeric_limits<float>::infinity(),
                              std::numeric_limits<float>::quiet_NaN(), -1.f, 1.f};
        return edge[j % 7];
    }
    if (mode == 6) return 100.f + float(j % 31) * 0.00001f;  // concentrated probe
    if (mode == 7) return j % 2 ? 100.f + float(j % 31) * 0.00001f : 0.1f;  // overflow, diverse probe
    if (mode == 8) return std::exp2(random * (12.f / 200.f));
    if (mode == 9) return j % 53 == 0 ? (j % 3 ? 4000.f : 4096.f) : std::exp2(random * (11.f / 200.f));
    return random;
}
bool check(Case c) {
    constexpr int64_t stride = 262144 / 4 + 2;
    const auto s = k::qsa_real_shapes();
    const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, s);
    // >32 rows forces the existing optimized selector, not the original reference,
    // as the control for a narrowing-eligible call. Extra rows are valid empty rows.
    const int64_t control_queries = std::max<int64_t>(33, c.queries);
    std::vector<int32_t> steps((size_t)(control_queries * k::kStepCount));
    std::vector<float> scores((size_t)(control_queries * stride), -999.f);
    std::mt19937 rng(742913);
    std::uniform_real_distribution<float> random(0.f, 200.f);
    for (int64_t i = 0; i < c.queries; ++i) {
        auto* st = steps.data() + i * k::kStepCount;
        const int64_t pos = c.ctx - c.queries + i;
        st[k::kStepPos] = (int32_t)pos;
        st[k::kStepNKv] = (int32_t)(pos + 1);
        st[k::kStepNBid] = (int32_t)((pos + 1) / s.idx_block);
        st[k::kStepWidth] = (int32_t)k::qsa_selection_width(pos + 1, s);
        for (int64_t j = 0; j <= st[k::kStepNBid]; ++j)
            scores[(size_t)(i * stride + j)] = score_value(c.mode, j, random(rng));
        if ((pos + 1) % s.idx_block)
            scores[(size_t)(i * stride + st[k::kStepNBid])] += 1e9f;
    }
    int64_t active = steps[(size_t)((c.queries - 1) * k::kStepCount + k::kStepNBid)] + 1;
    if (c.mode == 3) active = 0;
    if (c.mode == 4) active = stride + 1;
    float* ds = nullptr;
    int32_t *dt = nullptr, *da = nullptr, *db = nullptr;
    ck(cudaMalloc(&ds, scores.size() * sizeof(float)));
    ck(cudaMalloc(&dt, steps.size() * sizeof(int32_t)));
    ck(cudaMalloc(&da, (size_t)(control_queries * cap) * sizeof(int32_t)));
    ck(cudaMalloc(&db, (size_t)(control_queries * cap) * sizeof(int32_t)));
    ck(cudaMemcpy(ds, scores.data(), scores.size() * sizeof(float), cudaMemcpyHostToDevice));
    ck(cudaMemcpy(dt, steps.data(), steps.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    // Same capacity-based optimized wide path as captured decode. The padded
    // query count deliberately excludes narrowing for this control launch.
    k::qsa_block_topk(ds, dt, control_queries, stride, cap, s, da, nullptr);
    k::qsa_block_topk(ds, dt, c.queries, stride, cap, s, db, nullptr, active);
    ck(cudaDeviceSynchronize());
    std::vector<int32_t> a((size_t)(c.queries * cap)), b(a.size());
    ck(cudaMemcpy(a.data(), da, a.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    ck(cudaMemcpy(b.data(), db, b.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    bool ok = true;
    for (int64_t i = 0; i < c.queries; ++i) {
        const int64_t w = steps[(size_t)(i * k::kStepCount + k::kStepWidth)];
        if (!std::equal(a.begin() + i * cap, a.begin() + i * cap + w, b.begin() + i * cap)) {
            std::fprintf(stderr, "FAIL ctx=%lld queries=%lld mode=%d row=%lld\n",
                         (long long)c.ctx, (long long)c.queries, c.mode, (long long)i);
            ok = false;
            break;
        }
        for (int64_t j = 0; j < w; ++j) {
            const int32_t id = b[(size_t)(i * cap + j)];
            if (id < 0 || id >= steps[(size_t)(i * k::kStepCount + k::kStepNKv)] ||
                (j && id <= b[(size_t)(i * cap + j - 1)])) ok = false;
        }
    }
    ck(cudaFree(ds)); ck(cudaFree(dt)); ck(cudaFree(da)); ck(cudaFree(db));
    return ok;
}

bool graph_check() {
    constexpr int64_t stride = 262144 / 4 + 2, nq = 4, control_nq = 33;
    const auto s = k::qsa_real_shapes();
    const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, s);
    std::vector<int32_t> steps((size_t)(control_nq * k::kStepCount));
    std::vector<float> scores((size_t)(control_nq * stride), -999.f);
    float* ds = nullptr;
    int32_t *dt = nullptr, *da = nullptr, *db = nullptr;
    ck(cudaMalloc(&ds, scores.size() * sizeof(float)));
    ck(cudaMalloc(&dt, steps.size() * sizeof(int32_t)));
    ck(cudaMalloc(&da, (size_t)(control_nq * cap) * sizeof(int32_t)));
    ck(cudaMalloc(&db, (size_t)(nq * cap) * sizeof(int32_t)));
    ck(cudaMemcpy(dt, steps.data(), steps.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    cudaStream_t stream;
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    ck(cudaStreamCreate(&stream));
    // Warm the host's per-device dispatch caches before capture.
    k::qsa_block_topk(ds, dt, control_nq, stride, cap, s, da, stream);
    k::qsa_block_topk(ds, dt, nq, stride, cap, s, db, stream);
    ck(cudaStreamSynchronize(stream));
    ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    k::qsa_block_topk(ds, dt, control_nq, stride, cap, s, da, stream);
    k::qsa_block_topk(ds, dt, nq, stride, cap, s, db, stream);
    ck(cudaStreamEndCapture(stream, &graph));
    ck(cudaGraphInstantiate(&exec, graph, 0));
    const int64_t lengths[][nq] = {
        {0, 1, 2, 3}, {2048, 2051, 2052, 2061}, {131069, 131072, 131073, 131075},
        {262140, 262141, 262142, 262143}, {7, 63, 4097, 65535},
        {131073, 131074, 131075, 131076}
    };
    const int modes[] = {8, 8, 8, 9, 5, 7};
    std::mt19937 rng(217);
    std::uniform_real_distribution<float> random(0.f, 200.f);
    bool ok = true;
    for (size_t replay = 0; replay < sizeof(modes) / sizeof(modes[0]); ++replay) {
        for (int64_t row = 0; row < nq; ++row) {
            const int64_t n = lengths[replay][row];
            auto* st = steps.data() + row * k::kStepCount;
            k::qsa_step_fill(st, n - 1, s);
            for (int64_t b = 0; b <= n / 4; ++b)
                scores[(size_t)(row * stride + b)] = score_value(modes[replay], b, random(rng));
            if (n % 4) scores[(size_t)(row * stride + n / 4)] += 1e9f;
        }
        ck(cudaMemcpy(ds, scores.data(), scores.size() * sizeof(float), cudaMemcpyHostToDevice));
        ck(cudaMemcpy(dt, steps.data(), steps.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
        ck(cudaGraphLaunch(exec, stream));
        ck(cudaStreamSynchronize(stream));
        std::vector<int32_t> a((size_t)(nq * cap)), b(a.size());
        ck(cudaMemcpy(a.data(), da, a.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
        ck(cudaMemcpy(b.data(), db, b.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
        for (int64_t row = 0; row < nq; ++row) {
            const int64_t width = k::qsa_selection_width(lengths[replay][row], s);
            if (!std::equal(a.begin() + row * cap, a.begin() + row * cap + width, b.begin() + row * cap)) {
                std::fprintf(stderr, "FAIL graph replay=%zu row=%lld mode=%d n=%lld\n",
                             replay, (long long) row, modes[replay], (long long) lengths[replay][row]);
                ok = false;
            }
        }
    }
    ck(cudaGraphExecDestroy(exec)); ck(cudaGraphDestroy(graph)); ck(cudaStreamDestroy(stream));
    ck(cudaFree(ds)); ck(cudaFree(dt)); ck(cudaFree(da)); ck(cudaFree(db));
    return ok;
}

bool snapshot_check(const char* path) {
    std::FILE* file = std::fopen(path, "rb");
    int64_t header[3] = {};
    if (!file || std::fread(header, sizeof(header), 1, file) != 1 ||
        header[0] < 1 || header[0] > 32 || header[1] < 32768 ||
        header[1] > 262144 / 4 + 2 || header[2] != k::kStepCount) {
        if (file) std::fclose(file);
        std::fprintf(stderr, "invalid QSA score snapshot: %s\n", path);
        return false;
    }
    const int64_t nq = header[0], stride = header[1], control_nq = 33;
    const auto s = k::qsa_real_shapes();
    const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, s);
    std::vector<int32_t> steps((size_t)(control_nq * k::kStepCount));
    std::vector<float> scores((size_t)(control_nq * stride));
    const bool read_ok = std::fread(steps.data(), sizeof(int32_t), (size_t)(nq * k::kStepCount), file) ==
                            (size_t)(nq * k::kStepCount) &&
                         std::fread(scores.data(), sizeof(float), (size_t)(nq * stride), file) ==
                            (size_t)(nq * stride) && std::fgetc(file) == EOF;
    std::fclose(file);
    if (!read_ok) { std::fprintf(stderr, "truncated or oversized snapshot: %s\n", path); return false; }
    for (int64_t row = 0; row < nq; ++row) {
        const auto* st = steps.data() + row * k::kStepCount;
        if (st[k::kStepNKv] < 0 || st[k::kStepNKv] > 262144 ||
            st[k::kStepNBid] != st[k::kStepNKv] / 4 || st[k::kStepNBid] >= stride ||
            st[k::kStepWidth] != k::qsa_selection_width(st[k::kStepNKv], s)) {
            std::fprintf(stderr, "invalid live steps in snapshot: %s\n", path);
            return false;
        }
    }
    float* ds = nullptr;
    int32_t *dt = nullptr, *da = nullptr, *db = nullptr;
    ck(cudaMalloc(&ds, scores.size() * sizeof(float)));
    ck(cudaMalloc(&dt, steps.size() * sizeof(int32_t)));
    ck(cudaMalloc(&da, (size_t)(control_nq * cap) * sizeof(int32_t)));
    ck(cudaMalloc(&db, (size_t)(nq * cap) * sizeof(int32_t)));
    ck(cudaMemcpy(ds, scores.data(), scores.size() * sizeof(float), cudaMemcpyHostToDevice));
    ck(cudaMemcpy(dt, steps.data(), steps.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    // Compare the candidate with the actual optimized capacity-wide selector.
    k::qsa_block_topk(ds, dt, control_nq, stride, cap, s, da, nullptr);
    k::qsa_block_topk(ds, dt, nq, stride, cap, s, db, nullptr);
    ck(cudaDeviceSynchronize());
    std::vector<int32_t> a((size_t)(nq * cap)), b(a.size());
    ck(cudaMemcpy(a.data(), da, a.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    ck(cudaMemcpy(b.data(), db, b.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    bool ok = true;
    for (int64_t row = 0; row < nq; ++row) {
        const int64_t width = steps[(size_t)(row * k::kStepCount + k::kStepWidth)];
        ok &= std::equal(a.begin() + row * cap, a.begin() + row * cap + width, b.begin() + row * cap);
    }
    ck(cudaFree(ds)); ck(cudaFree(dt)); ck(cudaFree(da)); ck(cudaFree(db));
    std::printf("%s: real-score optimized-control parity, %lld rows from %s\n",
                ok ? "PASS" : "FAIL", (long long) nq, path);
    return ok;
}
}

int main(int argc, char** argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::puts("SKIP: no CUDA device");
        return 77;
    }
    // the active-bound dispatch is Turing's in the engine; forced here so every CUDA card checks it
#if defined(_WIN32)
    _putenv_s("STRATA_TOPK_ACTIVE_ANY", "1");
#else
    setenv("STRATA_TOPK_ACTIVE_ANY", "1", 1);
#endif
    if (argc == 3 && std::strcmp(argv[1], "--snapshot") == 0) return snapshot_check(argv[2]) ? 0 : 1;
    if (argc != 1) {
        std::fprintf(stderr, "usage: qsa_topk_active_parity [--snapshot FILE]\n");
        return 2;
    }
    const Case cases[] = {
        {1111, 256, 0}, {8192, 256, 0}, {32768, 256, 0}, {131072, 256, 0},
        {131075, 255, 0}, {135164, 256, 0}, {135168, 256, 0}, {250022, 256, 0},
        {262144, 256, 0}, {131072, 1, 0}, {131072, 8, 0}, {131072, 257, 0},
        {131072, 256, 1}, {131072, 256, 2}, {131072, 256, 3}, {131072, 256, 4},
        // either side of the Turing wide-kernel bound (22,528 blocks), and ties above the register limit
        {90108, 256, 0}, {90112, 256, 0}, {200000, 256, 2}, {262144, 256, 1},
        // #217: device-live threshold, every partial tail, exact cutoff ties,
        // signed zero/NaNs/infinities, concentrated probes and gather overflow.
        {0, 1, 8}, {1, 1, 8}, {3, 1, 8}, {2051, 1, 8}, {2052, 1, 8},
        {131068, 1, 8}, {131072, 1, 8}, {131073, 1, 8}, {131074, 1, 8}, {131075, 1, 8},
        {131072, 4, 8}, {131104, 32, 8}, {131105, 33, 8},
        {131075, 1, 9}, {262143, 4, 9}, {131075, 1, 5}, {262143, 4, 5},
        {131075, 1, 6}, {131075, 1, 7}, {262143, 4, 7},
    };
    for (const auto c : cases) if (!check(c)) return 1;
    if (!graph_check()) return 1;
    std::printf("PASS: %zu optimized-control top-k cases and 24 changed-length graph rows, selected IDs bitwise identical\n",
                sizeof(cases) / sizeof(cases[0]));
    return 0;
}
