// Independent exact-dyadic CPU oracle for native helper weighting and combination.
// No model, MMQ reference, timing claim or router changes; real H=2560, K=10.
#include "strata/platform/protected_test.hpp"
#include "strata/prefill/kernels.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int T = 3, H = 2560, K = 10, E = 13;
void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
struct Buffer {
    void* data = nullptr;
    explicit Buffer(size_t bytes) { check(cudaMalloc(&data, bytes)); }
    ~Buffer() { cudaFree(data); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    template<class V> V* as() { return static_cast<V*>(data); }
    template<class V> void upload(const std::vector<V>& values) {
        check(cudaMemcpy(data, values.data(), values.size() * sizeof(V), cudaMemcpyHostToDevice));
    }
};
void equal(const Buffer& buffer, const std::vector<float>& expected, const char* operation) {
    std::vector<float> actual(expected.size());
    check(cudaMemcpy(actual.data(), buffer.data, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < actual.size(); ++i)
        if (actual[i] != expected[i])
            throw std::runtime_error(std::string(operation) + " differs from exact CPU oracle at " + std::to_string(i));
}
void run() {
    std::vector<int32_t> ids(T * K), slot(T * K), pair;
    std::vector<uint8_t> resident(E);
    std::vector<float> weights(T * K), shared(T * H), gate(T, 0.0f);
    for (int e = 0; e < E; ++e) resident[e] = e % 3 == 0;
    for (int p = 0; p < T * K; ++p) {
        ids[p] = (p / K * 7 + p % K * 3) % E;
        weights[p] = float(p % 7 + 1) / 64.0f;
    }
    // Independent mathematical oracle: every product is an integer /2048;
    // all sums and the shared sigmoid(0)=1/2 term are exactly representable.
    std::vector<float> expected_peer(T * H), expected_combined(T * H);
    for (int t = 0; t < T; ++t)
        for (int d = 0; d < H; ++d) {
            int total = 0, remote = 0;
            for (int k = 0; k < K; ++k) {
                const int p = t * K + k;
                const int term = (p % 7 + 1) * (p + 1) * (d % 13 - 6);
                total += term;
                if (resident[ids[p]]) remote += term;
            }
            shared[t * H + d] = float(d % 5 - 2) / 8.0f;
            expected_peer[t * H + d] = float(remote) / 2048.0f;
            expected_combined[t * H + d] = float(total + 128 * (d % 5 - 2)) / 2048.0f;
        }
    std::vector<int> order;
    for (int remote = 0; remote <= 1; ++remote)
        for (int e = E - 1; e >= 0; --e)
            for (int p = T * K - 1; p >= 0; --p)
                if (resident[ids[p]] == remote && ids[p] == e) order.push_back(p);
    const auto local_rows = std::count_if(order.begin(), order.end(), [&](int p) { return !resident[ids[p]]; });
    std::vector<float> all(T * K * H), peer;
    for (size_t row = 0; row < order.size(); ++row) {
        const int p = order[row];
        slot[p] = int32_t(row);
        for (int d = 0; d < H; ++d) all[row * H + d] = float((p + 1) * (d % 13 - 6)) / 32.0f;
        if (resident[ids[p]]) {
            pair.push_back(p);
            peer.insert(peer.end(), all.begin() + row * H, all.begin() + (row + 1) * H);
        }
    }
    Buffer d_all(all.size() * 4), d_peer(peer.size() * 4), d_ids(ids.size() * 4), d_slot(slot.size() * 4);
    Buffer d_pair(pair.size() * 4), d_mask(resident.size()), d_weights(weights.size() * 4);
    Buffer d_shared(shared.size() * 4), d_gate(gate.size() * 4), d_inverse(T * K * 4);
    Buffer d_sum(T * H * 4), d_combined(T * H * 4);
    d_all.upload(all); d_peer.upload(peer); d_ids.upload(ids); d_slot.upload(slot);
    d_pair.upload(pair); d_mask.upload(resident); d_weights.upload(weights);
    d_shared.upload(shared); d_gate.upload(gate);
    using namespace strata::prefill;
    peer_reduce_k_order(d_peer.as<float>(), d_pair.as<int32_t>(), d_weights.as<float>(), pair.size(),
                        d_inverse.as<int32_t>(), d_sum.as<float>(), T, nullptr);
    check(cudaDeviceSynchronize());
    equal(d_sum, expected_peer, "native peer reduction");
    moe_combine_native_reference(d_all.as<float>(), d_slot.as<int32_t>(), d_ids.as<int32_t>(), d_mask.as<uint8_t>(),
                                 d_weights.as<float>(), d_shared.as<float>(), d_gate.as<float>(), d_combined.as<float>(), T, nullptr);
    check(cudaDeviceSynchronize());
    equal(d_combined, expected_combined, "unassisted native combination");
    moe_combine_peer(d_all.as<float>(), d_slot.as<int32_t>(), d_weights.as<float>(), d_shared.as<float>(),
                     d_gate.as<float>(), d_sum.as<float>(), local_rows, d_combined.as<float>(), T, nullptr);
    check(cudaDeviceSynchronize());
    equal(d_combined, expected_combined, "assisted native combination");
    // Reusing scratch for an empty helper group must not retain the previous inverse/output.
    peer_reduce_k_order(d_peer.as<float>(), d_pair.as<int32_t>(), d_weights.as<float>(), 0,
                        d_inverse.as<int32_t>(), d_sum.as<float>(), T, nullptr);
    check(cudaDeviceSynchronize());
    equal(d_sum, std::vector<float>(T * H, 0.0f), "empty helper scratch recovery");
}
}
int main() {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
        if (!strata::platform::acknowledge_protected_test()) return 1;
        run();
        std::puts("native prefill reduction: independent exact CPU oracle PASS (3x2560, shuffled partial routes, empty scratch recovery)");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "native prefill reduction FAIL: %s\n", error.what());
        return 1;
    }
}
