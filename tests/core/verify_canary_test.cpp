// Synthetic GPU seam only. Run exclusively through the protected supervisor.
// Uses production kernels: no weights/model, no substitute host echo.
#include "strata/platform/protected_test.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/cvec.hpp"
#include "strata/core/operation_canary.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include <cuda_runtime.h>

#include <atomic>
#include <array>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

using namespace strata::core;
using namespace strata::kernels;
#define CUDA_OK(x) do { const auto e = (x); if (e != cudaSuccess) throw std::runtime_error(std::string(#x) + ": " + cudaGetErrorString(e)); } while (0)
#define REQUIRE(x) do { if (!(x)) throw std::runtime_error(std::string("line ") + std::to_string(__LINE__) + ": " + #x); } while (0)

struct Staging {
    VerifyCanaryInput input[2];
    VerifyCanaryOwner owners[2];
    VerifyCanaryOutput output[2], committed[2];
    int32_t token[2], position[2], steering[2], picked[2];
};
struct Device {
    VerifyCanaryOwner owners[2];
    VerifyCanaryOutput scratch[2];
    int32_t token[2], position[2], steering[2];
    float residual[2 * 256];
};
struct Fixture {
    Staging *h = nullptr, *m = nullptr;
    Device* d = nullptr;
    cudaStream_t stream = nullptr;
    cudaGraphExec_t exec[4] = {};
    ~Fixture() {
        if (stream) cudaStreamSynchronize(stream);
        for (auto e : exec) if (e) cudaGraphExecDestroy(e);
        if (d) cudaFree(d);
        if (h) cudaFreeHost(h);
        if (stream) cudaStreamDestroy(stream);
    }
    void init() {
        CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUDA_OK(cudaHostAlloc((void**) &h, sizeof(Staging), cudaHostAllocMapped));
        CUDA_OK(cudaHostGetDevicePointer((void**) &m, h, 0));
        CUDA_OK(cudaMalloc((void**) &d, sizeof(Device)));
        *h = {};
        std::string err;
        std::vector<float> direction(2 * 256, 0.0f), scales{0.0f, 1.0f};
        for (int j = 256; j < 512; ++j) direction[(size_t) j] = 1.0f;
        REQUIRE(cvec_upload(direction, scales, 1, 1, 1, 256, 1, err));
        cvec_set_enabled(false); // row flags must override even the disabled global flag
        // Correct mapping, swapped window, swapped commit, and two consecutive rows owned by slot zero.
        for (int variant = 0; variant < 4; ++variant) {
            CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
            copy_i32_from_mapped(d->token, m->token, 2, stream);
            copy_i32_from_mapped(d->position, m->position, 2, stream);
            // Steering is deliberately uploaded independently before replay, as in Verifier.
            for (int t = 0; t < 2; ++t) {
                const int slot = variant == 3 ? 0 : variant == 1 ? 1 - t : t;
                verify_canary_begin(m->input + t, d->owners + slot, slot, d->token + t,
                    d->position + t, d->steering + t, d->scratch + t, stream);
            }
            cvec_apply(d->residual, 1, 2, 256, nullptr, 0, nullptr, 0, false, stream, d->steering);
            for (int t = 0; t < 2; ++t) {
                const int slot = variant == 3 ? 0 : variant == 1 ? 1 - t : t;
                verify_canary_finish(m->input + t, d->owners + slot, slot, d->token + t,
                    d->position + t, d->steering + t, d->scratch + t, m->picked + t,
                    m->output + t, 1, stream);
            }
            for (int t = 0; t < 2; ++t) {
                const int slot = variant == 3 ? 0 : variant == 2 ? 1 - t : t;
                verify_canary_finish(m->input + t, d->owners + slot, slot, d->token + t,
                    d->position + t, d->steering + t, d->scratch + t, m->picked + t,
                    m->committed + t, 2, stream);
            }
            cudaGraph_t graph = nullptr;
            CUDA_OK(cudaStreamEndCapture(stream, &graph));
            const auto status = cudaGraphInstantiate(&exec[variant], graph, nullptr, nullptr, 0);
            cudaGraphDestroy(graph);
            CUDA_OK(status);
        }
    }
    void stage(uint64_t epoch) {
        for (int t = 0; t < 2; ++t) {
            h->input[t] = {uint64_t(101 + t * 101), epoch, t, 31 + t, 7 + t, t};
            h->owners[t] = {h->input[t].request_id, t, 0};
            h->token[t] = h->input[t].token;
            h->position[t] = h->input[t].position;
            h->steering[t] = t;
            h->picked[t] = 71 + t;
            h->output[t] = h->committed[t] = {};
        }
    }
    void upload_controls() {
        CUDA_OK(cudaMemcpyAsync(d->owners, h->owners, sizeof(h->owners), cudaMemcpyHostToDevice, stream));
        CUDA_OK(cudaMemcpyAsync(d->steering, h->steering, sizeof(h->steering), cudaMemcpyHostToDevice, stream));
        CUDA_OK(cudaStreamSynchronize(stream));
    }
    void replay(int variant) {
        std::atomic_thread_fence(std::memory_order_seq_cst);
        CUDA_OK(cudaMemsetAsync(d->residual, 0, sizeof(d->residual), stream));
        CUDA_OK(cudaGraphLaunch(exec[variant], stream));
        CUDA_OK(cudaStreamSynchronize(stream));
    }
};

void indexer_rollback_test(cudaStream_t stream) {
    constexpr int D = 128, CELLS = 96;
    struct State {
        float tail[3 * D], dead[D], pooled[(CELLS / 4 + 2) * D];
        int32_t block_pos;
    };
    struct Storage {
        State reference, candidate;
        float raw[CELLS * D], gamma[D], saved_tail[3 * D];
        int32_t positions[CELLS], first, count;
    };
    Storage host{};
    for (int i = 0; i < CELLS; ++i) {
        host.positions[i] = i;
        for (int d = 0; d < D; ++d) host.raw[i * D + d] = float((i * 17 + d * 7) % 113 - 56) / 32.0f;
    }
    for (float& value : host.gamma) value = 1.0f;
    Storage* device = nullptr;
    CUDA_OK(cudaMalloc((void**) &device, sizeof(Storage)));
    struct Release { Storage* p; ~Release() { cudaFree(p); } } release{device};
    CUDA_OK(cudaMemcpy(device, &host, sizeof(host), cudaMemcpyHostToDevice));
    const auto shape = qsa_real_shapes();
    const RopeScaling scaling;
    const auto buffers = [](State* s) { return QsaIndexerBuffers{s->tail, s->dead, s->pooled, &s->block_pos}; };
    const auto append = [&](State* s, int first, int end, bool steps = false) {
        if (steps) {
            native_qsa_indexer_append_steps(device->raw + first * D, device->positions + first, 1, end - first,
                                           0, device->gamma, 1e-6f, buffers(s), shape, CELLS, scaling, stream);
        } else for (int i = first; i < end; ++i)
            native_qsa_indexer_append(device->raw + i * D, device->positions + i, 0,
                                      device->gamma, 1e-6f, buffers(s), shape, CELLS, scaling, stream);
    };
    const auto equal = [](const State& a, const State& b, int cells) {
        return std::memcmp(a.tail, b.tail, sizeof(a.tail)) == 0 &&
            std::memcmp(a.dead, b.dead, sizeof(a.dead)) == 0 && a.block_pos == b.block_pos &&
            std::memcmp(a.pooled, b.pooled, (cells / 4 + 1) * D * sizeof(float)) == 0;
    };
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t commit = nullptr;
    CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    native_qsa_indexer_commit(&device->first, &device->count, buffers(&device->candidate), stream);
    CUDA_OK(cudaStreamEndCapture(stream, &graph));
    CUDA_OK(cudaGraphInstantiate(&commit, graph, nullptr, nullptr, 0));
    CUDA_OK(cudaGraphDestroy(graph));
    struct GraphRelease { cudaGraphExec_t g; ~GraphRelease() { cudaGraphExecDestroy(g); } } graph_release{commit};
    int negative_controls = 0;
    for (const auto& c : {std::array<int, 3>{65, 4, 2}, {85, 4, 2}, {77, 4, 3},
                         {3, 4, 1}, {0, 4, 1}, {1, 4, 1}, {64, 4, 4}}) {
        const int first = c[0], width = c[1], keep = c[2], cells = first + keep;
        CUDA_OK(cudaMemsetAsync(&device->reference, 0, sizeof(State), stream));
        CUDA_OK(cudaMemsetAsync(&device->candidate, 0, sizeof(State), stream));
        append(&device->reference, 0, cells);
        append(&device->candidate, 0, first);
        CUDA_OK(cudaMemcpyAsync(device->saved_tail, device->candidate.tail, sizeof(host.saved_tail), cudaMemcpyDeviceToDevice, stream));
        append(&device->candidate, first, first + width, true);
        CUDA_OK(cudaMemcpyAsync(device->candidate.tail, device->saved_tail, sizeof(host.saved_tail), cudaMemcpyDeviceToDevice, stream));
        append(&device->candidate, first, cells, true);
        CUDA_OK(cudaStreamSynchronize(stream));
        CUDA_OK(cudaMemcpy(&host.reference, &device->reference, sizeof(State), cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(&host.candidate, &device->candidate, sizeof(State), cudaMemcpyDeviceToHost));
        if (first == 65 || first == 85) {
            REQUIRE(!equal(host.reference, host.candidate, cells)); // exact r5 rejected-block shape
            ++negative_controls;
        }
        CUDA_OK(cudaMemcpy(&device->first, &first, sizeof(first), cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(&device->count, &keep, sizeof(keep), cudaMemcpyHostToDevice));
        CUDA_OK(cudaGraphLaunch(commit, stream));
        CUDA_OK(cudaStreamSynchronize(stream));
        CUDA_OK(cudaMemcpy(&host.candidate, &device->candidate, sizeof(State), cudaMemcpyDeviceToHost));
        REQUIRE(equal(host.reference, host.candidate, cells));
    }
    REQUIRE(negative_controls == 2);
    std::puts("indexer rollback: batched append versus independent per-cell states, 2 failing-before controls, captured commit passed");
}

int main() {
    try {
        int devices = 0;
        const auto status = cudaGetDeviceCount(&devices);
        if (status != cudaSuccess || devices == 0) { std::puts("GPU unavailable: canary seam not run"); return 77; }
        if (!strata::platform::acknowledge_protected_test()) return 2;
        Fixture f;
        f.init();
        indexer_rollback_test(f.stream);
        f.stage(1); f.upload_controls(); f.replay(0);
        for (int t = 0; t < 2; ++t) {
            REQUIRE(verify_canary_matches(f.h->output[t], f.h->input[t], 1));
            REQUIRE(verify_canary_matches(f.h->committed[t], f.h->input[t], 2));
            REQUIRE(f.h->output[t].output_token == f.h->picked[t]);
        }
        float residual[512] = {};
        CUDA_OK(cudaMemcpy(residual, f.d->residual, sizeof(residual), cudaMemcpyDeviceToHost));
        for (int j = 0; j < 256; ++j) REQUIRE(residual[j] == 0.0f && residual[256 + j] == 1.0f);
        // Same graph, opposite flags; both owners and steering must follow the
        // new context rather than the capture-time registry/process switch.
        f.stage(2);
        for (int t = 0; t < 2; ++t)
            f.h->input[t].steering = f.h->owners[t].steering = f.h->steering[t] = 1 - t;
        f.upload_controls(); f.replay(0);
        CUDA_OK(cudaMemcpy(residual, f.d->residual, sizeof(residual), cudaMemcpyDeviceToHost));
        for (int j = 0; j < 256; ++j) REQUIRE(residual[j] == 1.0f && residual[256 + j] == 0.0f);
        REQUIRE(verify_canary_matches(f.h->output[0], f.h->input[0], 1));
        // Reuse the exact captured graph with new owners: addresses stay fixed,
        // request identity must NOT be baked into graph nodes.
        f.stage(2);
        f.h->input[0].request_id = f.h->owners[0].request_id = 303;
        f.upload_controls(); f.replay(0);
        REQUIRE(verify_canary_matches(f.h->output[0], f.h->input[0], 1));
        f.stage(3);
        f.h->input[1].request_id = f.h->input[0].request_id;
        f.h->input[1].slot = 0;
        f.h->input[1].steering = f.h->steering[1] = 0;
        f.upload_controls(); f.replay(3);
        for (int t = 0; t < 2; ++t) {
            REQUIRE(verify_canary_matches(f.h->output[t], f.h->input[t], 1));
            REQUIRE(verify_canary_matches(f.h->committed[t], f.h->input[t], 2));
        }
        ++f.h->token[1]; f.replay(3);
        REQUIRE(verify_canary_matches(f.h->output[0], f.h->input[0], 1));
        REQUIRE(!verify_canary_matches(f.h->output[1], f.h->input[1], 1));
        f.stage(3); f.upload_controls(); f.replay(1);
        REQUIRE(!verify_canary_matches(f.h->output[0], f.h->input[0], 1));
        f.replay(2);
        REQUIRE(verify_canary_matches(f.h->output[0], f.h->input[0], 1));
        REQUIRE(!verify_canary_matches(f.h->committed[0], f.h->input[0], 2));
        // Stale shared token staging despite correct self-reported input ID.
        ++f.h->token[0]; f.replay(0);
        REQUIRE(!verify_canary_matches(f.h->output[0], f.h->input[0], 1));
        f.stage(4); f.upload_controls();
        // Stale/wrong actual resident steering row; declared owner/input agree.
        f.h->steering[0] = 1; f.upload_controls(); f.replay(0);
        REQUIRE(!verify_canary_matches(f.h->output[0], f.h->input[0], 1));
        f.stage(5); f.upload_controls();
        f.h->input[0].request_id = 404; // independent device owner still 101
        f.replay(0);
        REQUIRE(!verify_canary_matches(f.h->output[0], f.h->input[0], 1));
        f.stage(6); f.upload_controls(); f.replay(0);
        auto next = f.h->input[0]; ++next.epoch;
        REQUIRE(!verify_canary_matches(f.h->output[0], next, 1));
        // Deterministic overwrite between begin/end, same production seam.
        verify_canary_begin(f.m->input, f.d->owners, 0, f.d->token, f.d->position,
                            f.d->steering, f.d->scratch, f.stream);
        CUDA_OK(cudaStreamSynchronize(f.stream));
        ++f.h->input[0].epoch;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        verify_canary_finish(f.m->input, f.d->owners, 0, f.d->token, f.d->position,
            f.d->steering, f.d->scratch, f.m->picked, f.m->output, 1, f.stream);
        CUDA_OK(cudaStreamSynchronize(f.stream));
        REQUIRE((f.h->output[0].errors & 16u) != 0);
        {
            OperationCanary canary;
            std::string error;
            REQUIRE(canary.init(error));
            REQUIRE(canary.bind(0, 101, error));
            REQUIRE(canary.bind(1, 202, error));
            cudaGraph_t graph = nullptr;
            cudaGraphExec_t wrong = nullptr;
            CUDA_OK(cudaStreamBeginCapture(f.stream, cudaStreamCaptureModeThreadLocal));
            canary.check(1, f.stream); // wrong resident graph, independent owner table
            canary.check(0, f.stream); // a later correct check must not erase the error
            CUDA_OK(cudaStreamEndCapture(f.stream, &graph));
            CUDA_OK(cudaGraphInstantiate(&wrong, graph, nullptr, nullptr, 0));
            CUDA_OK(cudaGraphDestroy(graph));
            canary.begin(0, f.stream);
            canary.check(0, f.stream);
            CUDA_OK(cudaStreamSynchronize(f.stream));
            REQUIRE(canary.validate("operation-test", error));
            canary.begin(0, f.stream);
            CUDA_OK(cudaGraphLaunch(wrong, f.stream));
            CUDA_OK(cudaStreamSynchronize(f.stream));
            REQUIRE(!canary.validate("operation-test", error));
            canary.begin(0, f.stream);
            CUDA_OK(cudaStreamSynchronize(f.stream));
            REQUIRE(canary.bind(0, 303, error)); // overwrite the independent owner while leased
            canary.check(0, f.stream);
            CUDA_OK(cudaStreamSynchronize(f.stream));
            REQUIRE(!canary.validate("operation-test", error));
            f.h->token[0] = 71;
            f.h->position[0] = 23;
            copy_i32_from_mapped(f.d->token, f.m->token, 1, f.stream);
            copy_i32_from_mapped(f.d->position, f.m->position, 1, f.stream);
            canary.begin(0, f.stream, 71, 23);
            canary.check(0, f.stream, f.d->token, f.d->position);
            CUDA_OK(cudaStreamSynchronize(f.stream));
            REQUIRE(canary.validate("operation-input", error));
            for (int corrupted = 0; corrupted < 2; ++corrupted) {
                canary.begin(0, f.stream, 71, 23);
                CUDA_OK(cudaMemsetAsync(corrupted ? f.d->position : f.d->token, 0, sizeof(int32_t), f.stream));
                canary.check(0, f.stream, f.d->token, f.d->position);
                canary.check(0, f.stream); // final owner-only check cannot hide corrupted real staging
                CUDA_OK(cudaStreamSynchronize(f.stream));
                REQUIRE(!canary.validate("operation-input", error));
                copy_i32_from_mapped(f.d->token, f.m->token, 1, f.stream);
                copy_i32_from_mapped(f.d->position, f.m->position, 1, f.stream);
            }
            CUDA_OK(cudaGraphExecDestroy(wrong));
        }
        std::puts("verify GPU canary seams: passed (including negative controls)");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
