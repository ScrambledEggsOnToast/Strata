// Explicit queued GPU fixture. No runtime weights or service launch.
#define STRATA_DFLASH2_CONTEXT_BLAS_TEST 1
#include "cuda/dflash2_forward.cu"
#include "cuda/dflash2_context_blas.cu"
#include <vector>

namespace gpu = strata::kernels::dflash2;
using strata::kernels::bf16_from_f32;
using strata::kernels::f32_from_bf16;
namespace {
constexpr uint32_t sentinel = 0x20000000U;
void need(bool okay, const char* why) { if (!okay) { std::fprintf(stderr, "FAIL: %s\n", why); std::exit(1); } }
void ck(cudaError_t e) { if (e != cudaSuccess) { std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e)); std::exit(2); } }
uint16_t datum(size_t i) {
    unsigned v = unsigned(i); v ^= v >> 16; v *= 0x7feb352dU; v ^= v >> 15; v *= 0x846ca68bU; v ^= v >> 16;
    return bf16_from_f32((int(v % 65535) - 32767) * 0.000013f);
}
void construction() {
    // Prime process-wide library caches before observing per-owner cleanup.
    { gpu::ContextBlas good(0); }
    size_t initial, total; ck(cudaMemGetInfo(&initial, &total));
    for (int stage : {1, 2}) {
        gpu::construction_failure_stage = stage;
        bool refused = false;
        try { gpu::ContextBlas broken(0); }
        catch (const std::exception& e) { refused = std::strstr(e.what(), "injected context cuBLAS") != nullptr; }
        gpu::construction_failure_stage = 0;
        need(refused, "injected partial construction refusal");
        size_t after; ck(cudaMemGetInfo(&after, &total));
        // A leaked explicit arena is87MiB. Allow16MiB for library cache behavior,
        // not enough to hide a missed owner-arena release.
        need(after + (size_t(16) << 20) >= initial, "failed construction releases explicit arena");
    }
    { gpu::ContextBlas reconstructed(0); }
    std::puts("PASS context_cublas_constructor_partial_cleanup_and_reconstruction");
}
void run(gpu::ContextBlas& owner, int m, int k, int n, int mode) {
    const size_t count = size_t(m) * n;
    std::vector<uint16_t> input(size_t(m) * k), weight(size_t(n) * k), reference(count + 32), candidate(count + 32);
    for (size_t i = 0; i < input.size(); ++i) input[i] = datum(i + 13);
    for (size_t i = 0; i < weight.size(); ++i) weight[i] = datum(i + 4519);
    input.back() = 0x8000;
    if (mode == 1 || mode == 3) input[0] = 0x0001;
    if (mode == 2 || mode == 3) weight[0] = 0x0001;
    if (mode == 4) input[0] = 0x7fc0;
    if (mode == 5) weight[0] = 0x7f80;
    if (mode == 6) {
        std::fill(input.begin(), input.end(), uint16_t(0)); std::fill(weight.begin(), weight.end(), uint16_t(0));
        for (int r = 0; r < m; ++r) { input[size_t(r) * k] = bf16_from_f32(std::ldexp(1.f, 64)); input[size_t(r) * k + 1] = bf16_from_f32(std::ldexp(1.f, 60)); }
        for (int o = 0; o < n; ++o) { weight[size_t(o) * k] = bf16_from_f32(std::ldexp(2.f - std::ldexp(1.f, -7), 63)); weight[size_t(o) * k + 1] = bf16_from_f32(std::ldexp(1.f, 59)); }
    }
    if (mode == 7) {
        std::fill(input.begin(), input.end(), bf16_from_f32(std::ldexp(1.f, 70)));
        std::fill(weight.begin(), weight.end(), bf16_from_f32(std::ldexp(1.f, 70)));
    }
    if (mode == 8) {
        need(k == 128, "cancellation shape"); std::fill(input.begin(), input.end(), bf16_from_f32(std::ldexp(1.f, 60)));
        for (int o = 0; o < n; ++o) for (int j = 0; j < k; ++j)
            weight[size_t(o) * k + j] = bf16_from_f32(std::ldexp(j < 64 ? -1.f : 1.f, j < 64 ? 60 : 62));
    }
    uint16_t *x, *w, *a, *b; uint32_t *sa, *sb;
    ck(cudaMalloc(&x, input.size() * 2)); ck(cudaMalloc(&w, weight.size() * 2));
    ck(cudaMalloc(&a, reference.size() * 2)); ck(cudaMalloc(&b, candidate.size() * 2));
    ck(cudaMalloc(&sa, 4)); ck(cudaMalloc(&sb, 4));
    cudaStream_t stream; ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    ck(cudaMemcpyAsync(x, input.data(), input.size() * 2, cudaMemcpyHostToDevice, stream));
    ck(cudaMemcpyAsync(w, weight.data(), weight.size() * 2, cudaMemcpyHostToDevice, stream));
    auto reset = [&] {
        ck(cudaMemsetAsync(a, 0xa5, reference.size() * 2, stream)); ck(cudaMemsetAsync(b, 0xa5, candidate.size() * 2, stream));
        ck(cudaMemcpyAsync(sa, &sentinel, 4, cudaMemcpyHostToDevice, stream)); ck(cudaMemcpyAsync(sb, &sentinel, 4, cudaMemcpyHostToDevice, stream));
    };
    auto launch = [&] {
        gpu::linear_context_reuse<<<dim3((n + 63) / 64, (m + 63) / 64), dim3(16, 16), 0, stream>>>(x, w, a + 16, m, k, n, sa);
        ck(cudaGetLastError()); std::string error;
        const bool okay = owner.project(x, w, b + 16, m, k, n, sb, stream, error); need(okay, error.c_str());
    };
    bool changed = false;
    auto compare = [&](gpu::ContextBlas::Statistics before) {
        ck(cudaStreamSynchronize(stream));
        ck(cudaMemcpy(reference.data(), a, reference.size() * 2, cudaMemcpyDeviceToHost));
        ck(cudaMemcpy(candidate.data(), b, candidate.size() * 2, cudaMemcpyDeviceToHost));
        uint32_t aa, bb; ck(cudaMemcpy(&aa, sa, 4, cudaMemcpyDeviceToHost)); ck(cudaMemcpy(&bb, sb, 4, cudaMemcpyDeviceToHost));
        need(aa == bb && aa == (sentinel | ((mode >= 4 && mode <= 7) ? 1U : 0U)), "exact numeric status/sentinel");
        auto stats = owner.statistics();
        need(stats.projection_calls == before.projection_calls + 1 && stats.activation_rows == before.activation_rows + m && stats.weight_rows == before.weight_rows + n,
             "actual production projection/row exposure");
        const bool raw_x = mode == 1 || mode == 3 || mode == 4, raw_w = mode == 2 || mode == 3 || mode == 5;
        need(stats.raw_activation_rows == before.raw_activation_rows + unsigned(raw_x) && stats.raw_weight_rows == before.raw_weight_rows + unsigned(raw_w), "raw row exposure");
        const size_t fallback = mode == 6 || mode == 7 ? count : (raw_x ? n : 0) + (raw_w ? m : 0) - (raw_x && raw_w ? 1 : 0);
        need(stats.fallback_outputs == before.fallback_outputs + fallback, "exact fallback output exposure");
        double e2 = 0, r2 = 0;
        for (size_t i = 0; i < reference.size(); ++i) {
            if (i < 16 || i >= count + 16) { need(reference[i] == 0xa5a5 && candidate[i] == 0xa5a5, "output guards"); continue; }
            const size_t row = (i - 16) / n, col = (i - 16) % n;
            const float av = f32_from_bf16(reference[i]), bv = f32_from_bf16(candidate[i]);
            if ((raw_x && !row) || (raw_w && !col) || mode == 6 || mode == 7)
                need(reference[i] == candidate[i] || (std::isnan(av) && std::isnan(bv)), "fallback retains increasing-K output");
            else { need(std::isfinite(av) && std::isfinite(bv), "finite candidate output"); const double d = double(av) - bv; e2 += d * d; r2 += double(av) * av; }
            if (mode == 8) need(reference[i] == (changed && row == 0 ? 0x7f41 : 0x7f40) && candidate[i] == reference[i], "finite cancellation exact result");
            if (mode == 0 && changed && row == size_t(m - 1)) need(av == 0.f && bv == 0.f, "changed input is consumed");
        }
        need(std::sqrt(e2 / std::max(r2, 1e-30)) <= 0.01, "fixed 1 percent normalized RMSE screen");
    };
    // API refusal happens before submission; the core dispatch keeps these rows
    // on its unchanged retained path.
    std::string error; need(!owner.project(x, w, b + 16, 127, k, n, sb, stream, error), "below-threshold API refuses");
    reset(); auto before = owner.statistics(); launch(); compare(before);
    bool witness_a = false, witness_b = false;
    for (size_t i = 16 + size_t(m - 1) * n; i < count + 16; ++i) {
        witness_a |= f32_from_bf16(reference[i]) != 0.f; witness_b |= f32_from_bf16(candidate[i]) != 0.f;
    }
    cudaGraph_t graph; cudaGraphExec_t executable;
    ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal)); launch(); ck(cudaStreamEndCapture(stream, &graph));
    ck(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    if (mode == 0) { need(witness_a && witness_b, "finite replay witness starts nonzero"); ck(cudaMemsetAsync(x + size_t(m - 1) * k, 0, size_t(k) * 2, stream)); }
    if (mode == 8) ck(cudaMemsetAsync(x, 0, 2, stream));
    changed = true; reset(); before = owner.statistics(); ck(cudaGraphLaunch(executable, stream)); compare(before);
    ck(cudaGraphExecDestroy(executable)); ck(cudaGraphDestroy(graph));
    // Original immutable weights must remain byte-for-byte BF16.
    std::vector<uint16_t> after(weight.size()); ck(cudaMemcpy(after.data(), w, weight.size() * 2, cudaMemcpyDeviceToHost)); need(after == weight, "weight bytes unchanged");
    ck(cudaStreamDestroy(stream)); ck(cudaFree(x)); ck(cudaFree(w)); ck(cudaFree(a)); ck(cudaFree(b)); ck(cudaFree(sa)); ck(cudaFree(sb));
    std::printf("PASS context_cublas rows=%d k=%d n=%d mode=%d production=1 replay=1 guards=1 fallback=1\n", m, k, n, mode);
}
} // namespace
int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--help") == 0) { std::puts("Queued SM70/SM86 context cuBLAS fixture, no arguments; CUDA_VISIBLE_DEVICES selects card."); return 0; }
    need(argc == 1, "unsupported arguments");
    need(setenv("STRATA_DFLASH2_CONTEXT_CUBLAS", "1", 1) == 0 &&
         setenv("STRATA_DFLASH2_CONTEXT_CUBLAS_TRACE", "1", 1) == 0, "opt-in mode and diagnostic counters");
    ck(cudaSetDevice(0));
    cudaDeviceProp properties{}; ck(cudaGetDeviceProperties(&properties, 0));
    const int sm = properties.major * 10 + properties.minor;
    need(sm == 70 || sm == 86, "fixture requires an actual SM70 or SM86 device");
    need(gpu::context_blas_enabled() &&
         std::strcmp(gpu::context_projection_identity(), "context-cublas-fp16-exact-fp32-n1024-m128-v1") == 0,
         "fixture requires the actual unchanged opt-in numerical mode");
    construction(); gpu::ContextBlas owner(0);
    for (int mode = 0; mode <= 8; ++mode) run(owner, 129, 128, 71, mode);
    run(owner, 128, 67, 1057, 3); // K padding plus chunk-tail offset on both raw axes.
    run(owner, 128, 128, 1057, 0);
    std::printf("PASS context_cublas_functional numerical_variant=1 real_model_not_exercised=1 sm=%d projection_mode=%s\n",
        sm, gpu::context_projection_identity());
}
