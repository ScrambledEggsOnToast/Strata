// Session allocation/pricing seam. Default mode is host-only; --gpu requires the protected supervisor.
// Linux CUDA target: link strata_engine and --wrap=cudaMalloc,cudaFree,cudaHostAlloc,cudaFreeHost,
// cudaHostGetDevicePointer,cudaMemcpy,cudaDeviceSynchronize (one -Wl,--wrap=NAME per symbol), and define
// STRATA_SESSION_TEST_WRAP. Wrappers forward to the REAL runtime; they count and inject allocation failures.
// --gpu also requires STRATA_ROPE_TABLE=1 (set before process start) to observe registration ownership.
#include "strata/core/session.hpp"
#include "strata/kernels/mrope.hpp"
#include "strata/kernels/rope.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/platform/protected_test.hpp"

#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>
#include <cstdlib>
#include <algorithm>
#include <unordered_map>

#include <memory>
#include <vector>
namespace {
int failures = 0;
void check(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); ++failures; }
}
static_assert(!std::is_copy_constructible_v<strata::core::SessionOwner>);
static_assert(!std::is_move_constructible_v<strata::core::SessionOwner>);

#ifdef STRATA_SESSION_TEST_WRAP
std::unordered_map<void*, size_t> device_live, host_live;
int fail_device = -1, fail_host = -1, fail_map = -1, fail_copy = -1;
uint64_t device_allocations = 0, host_allocations = 0, device_releases = 0, host_releases = 0;
size_t peak_device = 0, peak_host = 0;
bool need_drain = false;
bool inject(int& countdown) { return countdown >= 0 && countdown-- == 0; }
size_t bytes(const std::unordered_map<void*, size_t>& allocations) {
    size_t n = 0;
    for (const auto& allocation : allocations) n += allocation.second;
    return n;
}
#endif
}

#ifdef STRATA_SESSION_TEST_WRAP
extern "C" {
cudaError_t __real_cudaMalloc(void**, size_t);
cudaError_t __real_cudaFree(void*);
cudaError_t __real_cudaHostAlloc(void**, size_t, unsigned);
cudaError_t __real_cudaFreeHost(void*);
cudaError_t __real_cudaHostGetDevicePointer(void**, void*, unsigned);
cudaError_t __real_cudaMemcpy(void*, const void*, size_t, cudaMemcpyKind);
cudaError_t __real_cudaDeviceSynchronize();
cudaError_t __wrap_cudaMalloc(void** p, size_t n) {
    if (inject(fail_device)) { *p = nullptr; return cudaErrorMemoryAllocation; }
    const auto result = __real_cudaMalloc(p, n);
    if (result == cudaSuccess) {
        device_live.emplace(*p, n); ++device_allocations;
        peak_device = std::max(peak_device, bytes(device_live));
    }
    return result;
}
cudaError_t __wrap_cudaFree(void* p) {
    check(!need_drain, "device allocation freed before draining pending session work");
    const auto result = __real_cudaFree(p);
    if (result == cudaSuccess && p) { check(device_live.erase(p) == 1, "arena freed exactly once"); ++device_releases; }
    return result;
}
cudaError_t __wrap_cudaHostAlloc(void** p, size_t n, unsigned flags) {
    if (inject(fail_host)) { *p = nullptr; return cudaErrorMemoryAllocation; }
    const auto result = __real_cudaHostAlloc(p, n, flags);
    if (result == cudaSuccess) {
        host_live.emplace(*p, n); ++host_allocations;
        peak_host = std::max(peak_host, bytes(host_live));
    }
    return result;
}
cudaError_t __wrap_cudaFreeHost(void* p) {
    check(!need_drain, "pinned allocation freed before draining pending session work");
    const auto result = __real_cudaFreeHost(p);
    if (result == cudaSuccess && p) { check(host_live.erase(p) == 1, "original pinned handle freed exactly once"); ++host_releases; }
    return result;
}
cudaError_t __wrap_cudaHostGetDevicePointer(void** d, void* h, unsigned flags) {
    if (inject(fail_map)) return cudaErrorMemoryAllocation;
    return __real_cudaHostGetDevicePointer(d, h, flags);
}
cudaError_t __wrap_cudaMemcpy(void* d, const void* s, size_t n, cudaMemcpyKind kind) {
    if (inject(fail_copy)) return cudaErrorMemoryAllocation;
    return __real_cudaMemcpy(d, s, n, kind);
}
cudaError_t __wrap_cudaDeviceSynchronize() {
    const auto result = __real_cudaDeviceSynchronize();
    if (result == cudaSuccess) need_drain = false;
    return result;
}
}
#endif

int main(int argc, char** argv) {
    using namespace strata::core;
    ModelGeometry g;
    g.n_layers = 9;
    qsa_set_kv_resident(0);
    qsa_set_kv_q4(false);
    qsa_set_kv_hybrid(false);
    qsa_set_kv_int8(false);
    SessionAllocationBytes whole, range;
    check(session_allocation_bytes(g, 65, 10, whole), "price without CUDA initialization");
    check(whole.device == session_bytes(g, 65, 10), "owner and external arena use identical pricing");
    check(session_allocation_bytes(g, 65, 10, range, 4, 8), "nonzero ordinal range prices");
    check(range.device < whole.device && range.host_pinned * 2 == whole.host_pinned,
          "range pricing counts only its allocated QSA state");
    check(session_allocation_bytes(g, 65, 10, range, 8, 9) && range.host_pinned > 0,
          "suffix without QSA retains one bounded primary");
    check(!session_allocation_bytes(g, 65, 10, range, 9, 9) && range.device == 0,
          "empty range rejected and output cleared");
    check(!session_allocation_bytes(g, std::numeric_limits<int64_t>::max(), 10, range), "context overflow refused");
    check(!session_allocation_bytes(g, 65, 0, range), "invalid expert count refused");
    g.qsa_interval = 0;
    check(!session_allocation_bytes(g, 65, 10, range), "zero QSA interval refused before division");
    g.qsa_interval = 4;
    g.n_embd = std::numeric_limits<int64_t>::max();
    check(!session_allocation_bytes(g, 65, 10, range), "geometry overflow refused before layer arithmetic");
    g.n_embd = 2560;
    check(session_allocation_bytes(g, INT32_MAX, 64, range), "largest supported context remains representable");
    SessionState manual;
    session_release(manual);   // no CUDA calls for an empty/manual view
    if (argc == 1) return failures ? 1 : 0;
    if (argc != 2 || std::strcmp(argv[1], "--gpu") != 0) return 2;
#ifndef STRATA_SESSION_TEST_WRAP
    std::fprintf(stderr, "--gpu requires the real-runtime allocation wrapper target\n");
    return 2;
#else
    if (!std::getenv("STRATA_SUPERVISED") || std::strcmp(std::getenv("STRATA_SUPERVISED"), "1") != 0) {
        std::fprintf(stderr, "--gpu requires the protected supervisor\n"); return 2;
    }
    if (!strata::platform::acknowledge_protected_test()) return 2;
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) return 1;
    static float rope_canary[1];
    strata::kernels::rope_table_set(rope_canary, rope_canary, 1, strata::kernels::rope_scaling());
    if (strata::kernels::rope_table_for(strata::kernels::rope_scaling()).cos != rope_canary) {
        std::fprintf(stderr, "--gpu requires STRATA_ROPE_TABLE=1\n");
        strata::kernels::rope_table_release(rope_canary);
        return 2;
    }
    strata::kernels::rope_table_release(rope_canary);
    std::string error;
    const uint64_t host_kv_before = qsa_kv_host_bytes();
    auto empty = [&] {
        check(device_live.empty(), "no leaked device arena");
        check(host_live.empty(), "no leaked pinned staging/host KV");
        check(qsa_kv_host_bytes() == host_kv_before, "streamed KV live-byte counter returns to baseline");
        check(strata::kernels::rope_table_for(strata::kernels::rope_scaling()).cos == nullptr,
              "no dangling RoPE registration after release/failure");
        int current = -1;
        check(cudaGetDevice(&current) == cudaSuccess && current == device, "owner restores calling device");
    };
    for (int iteration = 0; iteration < 32; ++iteration) {
        SessionOwner owner;
        const int64_t lo = iteration % 2 ? 8 : 4;
        const int64_t hi = iteration % 2 ? 9 : 8;
        if (!owner.init(g, 65, 10, error, lo, hi)) {
            std::fprintf(stderr, "%s\n", error.c_str()); return 1;
        }
        auto& s = owner.state();
        check(s.qsa_primary() == 1 && s.qsa_alloc == 1, "global QSA ordinal, including trailing non-QSA range");
        check(s.qsa_states[0].host_step == nullptr && s.qsa_states[1].host_step != nullptr,
              "only the owned ordinal is initialized");
        check(bytes(device_live) == owner.memory().device, "charged arena equals actual cudaMalloc payload");
        check(bytes(host_live) == owner.memory().host_pinned, "charged pinned payload equals actual allocations");
        check(!owner.init(g, 65, 10, error), "live owner cannot silently replace graph addresses");
        session_zero(s, g, nullptr, nullptr);
        need_drain = true;  // cancellation/early release with submitted work still pending
        if (iteration % 2) { owner.reset(); owner.reset(); }
        // Other iterations exercise the destructor, not just explicit reset.
    }
    empty();
    {
        SessionOwner owner;
        fail_device = 0;
        check(!owner.init(g, 65, 10, error), "arena allocation failure is propagated");
        fail_device = -1;
        check(owner.device() == -1 && owner.memory().device == 0, "failed arena has no owner/charge");
    }
    empty();

    // Legacy carve: secondary ownership ends at session_release, but the caller still owns its arena.
    void* external = nullptr;
    const auto external_bytes = session_bytes(g, 65, 10, 4, 8);
    check(cudaMalloc(&external, external_bytes) == cudaSuccess, "external arena allocation");
    SessionState legacy;
    check(session_init(g, 65, 10, external, legacy, 4, 8) > 0, "legacy carve succeeds");
    session_zero(legacy, g, nullptr, nullptr);
    need_drain = true;
    session_release(legacy);
    session_release(legacy);
    check(host_live.empty() && device_live.count(external) == 1, "release leaves external arena intact");
    check(cudaFree(external) == cudaSuccess, "external owner frees its arena");
    empty();

    // A failure in the second layer must unwind the already completed first one, including host KV.
    g.n_layers = 8;
    qsa_set_kv_resident(qsa_kv_resident_min());
    for (int fail = 0; fail < 6; ++fail) {
        SessionOwner owner;
        fail_host = fail;
        check(!owner.init(g, qsa_kv_resident_min() + 4, 10, error), "every pinned allocation failure is propagated");
        fail_host = -1;
        check(owner.memory().device == 0, "failed allocation has zero live charge");
        empty();
    }
    for (int fail = 0; fail < 2; ++fail) {
        SessionOwner owner;
        fail_map = fail;
        check(!owner.init(g, qsa_kv_resident_min() + 4, 10, error), "mapped alias failure is propagated");
        fail_map = -1;
        empty();
    }
    for (int iteration = 0; iteration < 4; ++iteration) {
        {
            SessionOwner owner;
            check(owner.init(g, qsa_kv_resident_min() + 4, 10, error), "streamed create after allocation failure");
            check(bytes(host_live) == owner.memory().host_pinned, "streamed pinned allocations exactly priced");
        }
        empty();
    }
    qsa_set_kv_resident(0);
    for (int fail = 0; fail < 4; ++fail) {
        SessionOwner owner;
        fail_copy = fail;  // first RoPE cos/sin, first page table, second page table
        check(!owner.init(g, 65, 10, error), "initial upload failure is propagated");
        fail_copy = -1;
        empty();
    }
    // Borrowed RoPE is not unregistered when its own secondary state is released (MTP contract).
    {
        SessionOwner owner;
        check(owner.init(g, 65, 10, error), "RoPE owner initialized");
        auto& primary = owner.state().qsa_states[owner.state().qsa_primary()];
        strata::kernels::rope_table_set(primary.cos_tab, primary.sin_tab, 65, strata::kernels::rope_scaling());
        const auto registered = strata::kernels::rope_table_for(strata::kernels::rope_scaling());
        QsaState borrowed;
        void* arena = nullptr;
        check(cudaMalloc(&arena, qsa_state_bytes(g, 65, false)) == cudaSuccess, "borrower arena");
        check(qsa_state_init(g, 65, arena, borrowed, &primary) > 0, "borrowed RoPE initialization");
        check(cudaDeviceSynchronize() == cudaSuccess, "drain borrower");
        qsa_state_release(borrowed);
        qsa_state_release(borrowed);
        check(strata::kernels::rope_table_for(strata::kernels::rope_scaling()).cos == registered.cos,
              "borrower release preserves owner's registration");
        check(cudaFree(arena) == cudaSuccess, "borrower arena release");
        owner.reset();
        check(strata::kernels::rope_table_for(strata::kernels::rope_scaling()).cos == nullptr,
              "owner release removes its global registration");
    }
    empty();
    // Resident setup must restore the long-lived working table after every unwind,
    // including failures after registration and trimming with >=2 surviving slots.
    {
        SessionOwner working;
        check(working.init(g, 65, 10, error), "working RoPE session initialized");
        auto& primary = working.state().qsa_states[working.state().qsa_primary()];
        auto cuda_ok = [&](cudaError_t status) {
            if (status != cudaSuccess) {
                std::fprintf(stderr, "RoPE regression CUDA failure: %s\n", cudaGetErrorString(status));
                std::exit(1);
            }
        };
        constexpr int rows = 4, width = 128;
        float input[rows * width], expected[rows * width], observed[rows * width];
        for (int i = 0; i < rows * width; ++i) input[i] = (float) (i % 37 - 18) / 16.0f;
        const int positions[rows] = {1, 13, 32, 64};
        float *device_input = nullptr, *device_output = nullptr;
        int* device_positions = nullptr;
        cudaStream_t stream = nullptr;
        cuda_ok(cudaMalloc(&device_input, sizeof(input)));
        cuda_ok(cudaMalloc(&device_output, sizeof(observed)));
        cuda_ok(cudaMalloc(&device_positions, sizeof(positions)));
        cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        cuda_ok(cudaMemcpy(device_input, input, sizeof(input), cudaMemcpyHostToDevice));
        cuda_ok(cudaMemcpy(device_positions, positions, sizeof(positions), cudaMemcpyHostToDevice));
        auto launch = [&] {
            strata::kernels::native_rope_apply(device_input, device_output, rows, width, 64,
                                              strata::kernels::rope_scaling(), device_positions, stream);
        };
        auto copy_result = [&](float* destination) {
            cuda_ok(cudaStreamSynchronize(stream));
            cuda_ok(cudaMemcpy(destination, device_output, sizeof(observed), cudaMemcpyDeviceToHost));
        };
        launch();
        copy_result(expected);
        int negative_controls = 0;
        for (int scenario = 0; scenario < 4; ++scenario) {
            std::vector<std::unique_ptr<SessionOwner>> slots;
            const int count = scenario == 0 ? 1 : scenario == 1 ? 0 : scenario == 2 ? 2 : 3;
            for (int slot = 0; slot < count; ++slot) {
                auto owner = std::make_unique<SessionOwner>();
                check(owner->init(g, 65, 10, error), "resident RoPE session initialized");
                slots.push_back(std::move(owner));
            }
            if (scenario != 3) {
                SessionOwner failed;
                if (scenario == 0) fail_device = 0;
                if (scenario == 1) fail_copy = 2; // primary registered, page-table upload fails
                if (scenario == 2) fail_host = 2; // primary complete, second QSA staging fails
                check(!failed.init(g, 65, 10, error), "resident failure is propagated");
                fail_device = fail_copy = fail_host = -1;
                if (scenario == 0) slots.clear(); // one slot cannot sustain batching
            } else {
                slots.resize(2); // another stage fitted fewer slots
            }
            check(strata::kernels::rope_table_for(strata::kernels::rope_scaling()).cos == nullptr,
                  "failing-before control loses the latest registry on unwind");
            launch();
            copy_result(observed);
            if (std::memcmp(expected, observed, sizeof(expected)) != 0) ++negative_controls;
            else check(false, "lost RoPE registration must expose different actual arithmetic");
            working.bind_rope(); // the same post-resident-setup operation used by generate
            check(strata::kernels::rope_table_for(strata::kernels::rope_scaling()).cos == primary.cos_tab,
                  "fallback rebinds the surviving working table");
            cudaGraph_t graph = nullptr;
            cudaGraphExec_t executable = nullptr;
            cuda_ok(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
            launch();
            cuda_ok(cudaStreamEndCapture(stream, &graph));
            cuda_ok(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
            cuda_ok(cudaGraphDestroy(graph));
            slots.clear(); // captured table belongs to working, not a retired resident
            cuda_ok(cudaGraphLaunch(executable, stream));
            copy_result(observed);
            check(std::memcmp(expected, observed, sizeof(expected)) == 0,
                  "captured fallback rotation is byte-identical to isolated working rotation");
            cuda_ok(cudaGraphExecDestroy(executable));
        }
        check(negative_controls == 4, "all four fallback failures change arithmetic before repair");
        cuda_ok(cudaStreamDestroy(stream));
        cuda_ok(cudaFree(device_positions));
        cuda_ok(cudaFree(device_output));
        cuda_ok(cudaFree(device_input));
        working.reset();
        empty();
        std::printf("ROPE_FALLBACK cases=4 negative_controls=%d captured_exact=4 failures=%d\n",
                    negative_controls, failures);
    }
    // The same cleanup owns MTP ring host storage even though its device KV uses fewer physical pages.
    qsa_set_kv_resident(qsa_kv_resident_min());
    for (int format = 0; format < 3; ++format) {
        qsa_set_kv_int8(format == 1);
        qsa_set_kv_q4(format == 2);
        void* arena = nullptr;
        QsaState ring;
        check(cudaMalloc(&arena, qsa_state_bytes(g, 65, true, 16)) == cudaSuccess, "ring arena");
        check(qsa_state_init(g, 65, arena, ring, nullptr, 16) > 0, "ring QSA initialization");
        check(ring.kv_mode == 2 && bytes(host_live) == qsa_state_host_bytes(g, 65, 16),
              "ring pinned payload matches pricing in FP16/INT8/Q4");
        check(cudaDeviceSynchronize() == cudaSuccess, "drain ring");
        qsa_state_release(ring);
        check(cudaFree(arena) == cudaSuccess, "ring arena release");
        empty();
    }
    std::printf("SESSION_LIFECYCLE device_allocations=%llu device_releases=%llu host_allocations=%llu host_releases=%llu "
                "device_live=%zu host_live=%zu peak_device=%zu peak_host=%zu pending_release_cycles=32 failures=%d\n",
                (unsigned long long) device_allocations, (unsigned long long) device_releases,
                (unsigned long long) host_allocations, (unsigned long long) host_releases,
                bytes(device_live), bytes(host_live), peak_device, peak_host, failures);
    return failures ? 1 : 0;
#endif
}
