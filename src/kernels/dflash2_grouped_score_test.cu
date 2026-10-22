// Explicit queued GPU fixture; retained kernels are the direct control.
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
constexpr size_t score_count = size_t(query_heads) * block * max_keys;
constexpr uint32_t status_sentinel = 0x20000000u;
void check(cudaError_t code) {
    if (code != cudaSuccess) {
        std::fprintf(stderr, "CUDA failure: %s\n", cudaGetErrorString(code));
        std::exit(1); // Never unwind buffers after an unknown device drain.
    }
}
void require(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr, "FAIL %s\n", message); std::exit(1); }
}
uint32_t bits(float value) {
    uint32_t result; std::memcpy(&result, &value, sizeof(result)); return result;
}
uint16_t finite_value(size_t i) {
    constexpr uint16_t values[]{0x3f80, 0xbf80, 0x4101, 0xc101, 0x3b81, 0xbb81,
                                0x0001, 0x8001, 0x3f7f, 0xbf7f, 0, 0x8000};
    return values[((i * 1664525u + (i >> 4) * 1013904223u) >> 9) % std::size(values)];
}
struct Fixture {
    std::vector<uint16_t> ring, query, keys;
    std::vector<float> a, b, probabilities, probabilities_b;
    std::vector<uint16_t> output_a, output_b;
    uint16_t *dring = nullptr, *dq = nullptr, *dk = nullptr, *dvout = nullptr, *dvout_b = nullptr;
    float *da = nullptr, *db = nullptr, *dp = nullptr, *dstats = nullptr, *dp_b = nullptr, *partial = nullptr;
    uint8_t* unsafe = nullptr;
    uint32_t *sa = nullptr, *sb = nullptr;
    cudaStream_t stream = nullptr;
    Fixture() : ring(kv_elements), query(size_t(block) * query_width),
                keys(size_t(block) * kv_width), a(score_count + 2 * guard),
                b(score_count + 2 * guard), probabilities(score_count + 2 * guard), probabilities_b(probabilities.size()),
                output_a(size_t(block) * query_width + 2 * guard), output_b(output_a.size()) {
        for (size_t i = 0; i < ring.size(); ++i) ring[i] = finite_value(i + 103);
        check(cudaMalloc(&dring, ring.size() * sizeof(uint16_t)));
        check(cudaMalloc(&dq, query.size() * sizeof(uint16_t)));
        check(cudaMalloc(&dk, keys.size() * sizeof(uint16_t)));
        check(cudaMalloc(&da, a.size() * sizeof(float)));
        check(cudaMalloc(&db, b.size() * sizeof(float)));
        check(cudaMalloc(&dp, probabilities.size() * sizeof(float)));
        check(cudaMalloc(&dp_b, probabilities.size() * sizeof(float)));
        check(cudaMalloc(&partial, attention_value_split_partial_count * sizeof(float)));
        check(cudaMalloc(&unsafe, attention_value_split_flag_count));
        check(cudaMalloc(&dstats, size_t(query_heads) * block * 2 * sizeof(float)));
        check(cudaMalloc(&dvout, output_a.size() * sizeof(uint16_t)));
        check(cudaMalloc(&dvout_b, output_b.size() * sizeof(uint16_t)));
        check(cudaMalloc(&sa, sizeof(uint32_t))); check(cudaMalloc(&sb, sizeof(uint32_t)));
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        check(cudaMemcpyAsync(dring, ring.data(), ring.size() * sizeof(uint16_t), cudaMemcpyHostToDevice, stream));
        check(cudaStreamSynchronize(stream));
    }
    ~Fixture() {
        check(cudaStreamSynchronize(stream)); check(cudaStreamDestroy(stream));
        check(cudaFree(dring)); check(cudaFree(dq)); check(cudaFree(dk));
        check(cudaFree(da)); check(cudaFree(db)); check(cudaFree(dp));
        check(cudaFree(dp_b)); check(cudaFree(partial)); check(cudaFree(unsafe)); check(cudaFree(dvout_b));
        check(cudaFree(dstats)); check(cudaFree(dvout)); check(cudaFree(sa)); check(cudaFree(sb));
    }
    void reset() {
        check(cudaMemsetAsync(da, 0xa5, a.size() * sizeof(float), stream));
        check(cudaMemsetAsync(db, 0xa5, b.size() * sizeof(float), stream));
        check(cudaMemsetAsync(dp, 0xa5, probabilities.size() * sizeof(float), stream));
        check(cudaMemsetAsync(dp_b, 0xa5, probabilities.size() * sizeof(float), stream));
        check(cudaMemsetAsync(dvout, 0xa5, output_a.size() * sizeof(uint16_t), stream));
        check(cudaMemsetAsync(dvout_b, 0xa5, output_b.size() * sizeof(uint16_t), stream));
        check(cudaMemcpyAsync(sa, &status_sentinel, sizeof(status_sentinel), cudaMemcpyHostToDevice, stream));
        check(cudaMemcpyAsync(sb, &status_sentinel, sizeof(status_sentinel), cudaMemcpyHostToDevice, stream));
    }
    void compare(int rows, int context, int exceptional) {
        check(cudaStreamSynchronize(stream));
        check(cudaMemcpy(a.data(), da, a.size() * sizeof(float), cudaMemcpyDeviceToHost));
        check(cudaMemcpy(b.data(), db, b.size() * sizeof(float), cudaMemcpyDeviceToHost));
        uint32_t ra = 0, rb = 0;
        check(cudaMemcpy(&ra, sa, sizeof(ra), cudaMemcpyDeviceToHost));
        check(cudaMemcpy(&rb, sb, sizeof(rb), cudaMemcpyDeviceToHost));
        require(ra == rb && ra == (status_sentinel | (exceptional ? 1u : 0u)), "numeric status differs");
        const int count = context + rows;
        const size_t active = size_t(query_heads) * rows * count;
        for (size_t i = 0; i < a.size(); ++i) {
            const bool outside = i < guard || i >= guard + active;
            const bool both_nan = std::isnan(a[i]) && std::isnan(b[i]);
            if ((outside && (bits(a[i]) != 0xa5a5a5a5u || bits(b[i]) != 0xa5a5a5a5u)) ||
                (bits(a[i]) != bits(b[i]) && !both_nan)) {
                std::fprintf(stderr, "score mismatch rows=%d context=%d index=%zu %08x %08x\n",
                             rows, context, i, bits(a[i]), bits(b[i]));
                std::exit(1);
            }
            if (!outside) {
                const size_t at = i - guard;
                const int row = int(at / count) % rows, key = int(at % count);
                const int distance = key < context ? context + row - key : row - (key - context);
                if (distance <= -2048 || distance >= 2048)
                    require(bits(a[i]) == 0xff800000u && bits(b[i]) == 0xff800000u,
                            "masked score must remain negative infinity even with nonfinite input");
            }
        }
    }
    void compare_pipeline(int rows, int context, int exceptional) {
        compare(rows, context, exceptional);
        check(cudaMemcpy(probabilities.data(), dp, probabilities.size() * sizeof(float), cudaMemcpyDeviceToHost));
        check(cudaMemcpy(probabilities_b.data(), dp_b, probabilities_b.size() * sizeof(float), cudaMemcpyDeviceToHost));
        const size_t active = size_t(query_heads) * rows * (context + rows);
        for (size_t i = 0; i < probabilities.size(); ++i) {
            if (i < guard || i >= guard + active)
                require(bits(probabilities[i]) == 0xa5a5a5a5u && bits(probabilities_b[i]) == 0xa5a5a5a5u, "probability guard");
            else require(bits(probabilities[i]) == bits(probabilities_b[i]) ||
                         (std::isnan(probabilities[i]) && std::isnan(probabilities_b[i])), "probabilities differ");
        }
        check(cudaMemcpy(output_a.data(), dvout, output_a.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost));
        check(cudaMemcpy(output_b.data(), dvout_b, output_b.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < output_a.size(); ++i) {
            if (i < guard || i >= guard + size_t(rows) * query_width)
                require(output_a[i] == 0xa5a5 && output_b[i] == 0xa5a5, "value output guard");
            else require(output_a[i] == output_b[i] ||
                         (std::isnan(f32_from_bf16(output_a[i])) && std::isnan(f32_from_bf16(output_b[i]))), "pipeline BF16 outputs differ");
        }
    }
    void run(int rows, int context, int begin, int layer, int exceptional = 0) {
        const int frontier = begin + context, count = context + rows;
        std::fill(query.begin(), query.end(), uint16_t(0x7fc0));
        std::fill(keys.begin(), keys.end(), uint16_t(0x7fc0));
        for (size_t i = 0; i < size_t(rows) * query_width; ++i) query[i] = finite_value(i + 51);
        for (size_t i = 0; i < size_t(rows) * kv_width; ++i) keys[i] = finite_value(i + 79);
        std::fill(query.begin(), query.begin() + head_dim, uint16_t(0));
        query[0] = 0x3f80; // Changed block key dimension zero must affect score [head0,row0,context].
        const size_t ring_at = ((size_t(layer) * 2 * kv_heads) * capacity + begin % capacity) * head_dim;
        const uint16_t saved = ring[ring_at];
        if (exceptional == 1) {
            require(context > 0, "nonfinite context test needs context");
            ring[ring_at] = 0x7fc0;
        }
        if (exceptional == 2) keys[17] = 0x7f80;
        if (exceptional == 3) query[0] = 0x7fc0;
        if (exceptional == 4) {
            require(rows == 8 && context == capacity, "masked nonfinite query geometry");
            query[size_t(rows - 1) * query_width] = 0x7fc0;
        }
        if (exceptional == 5) { query[0] = 0x7f7f; keys[0] = 0x7f7f; }
        check(cudaMemcpyAsync(dring + ring_at, ring.data() + ring_at, sizeof(uint16_t), cudaMemcpyHostToDevice, stream));
        check(cudaMemcpyAsync(dq, query.data(), query.size() * sizeof(uint16_t), cudaMemcpyHostToDevice, stream));
        check(cudaMemcpyAsync(dk, keys.data(), keys.size() * sizeof(uint16_t), cudaMemcpyHostToDevice, stream));
        auto launch = [&](bool candidate) {
            if (candidate) check(static_cast<cudaError_t>(attention_scores(dq, dk, dring, db + guard, layer, begin, frontier, rows, sb, stream)));
            else {
                score_kernel<<<dim3(unsigned((count + 7) / 8), query_heads * rows), threads, 0, stream>>>(
                    dq, dk, dring, da + guard, layer, begin, frontier, rows, sa);
                check(cudaGetLastError());
            }
        };
        auto pipeline = [&](bool candidate) {
            launch(candidate);
            float* probability = (candidate ? dp_b : dp) + guard;
            uint32_t* status = candidate ? sb : sa;
            check(static_cast<cudaError_t>(attention_softmax((candidate ? db : da) + guard, probability, dstats, rows, count, status, stream)));
            check(static_cast<cudaError_t>(attention_values_planned(probability, dk, dring,
                (candidate ? dvout_b : dvout) + guard, layer, begin, frontier, rows, status, stream,
                {partial, unsafe, attention_value_split_partial_count, attention_value_split_flag_count})));
        };
        reset(); launch(false); launch(true); compare(rows, context, exceptional);
        reset(); pipeline(false); pipeline(true); compare_pipeline(rows, context, exceptional);
        cudaGraph_t graph = nullptr; cudaGraphExec_t executable = nullptr;
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        pipeline(false); pipeline(true);
        check(cudaStreamEndCapture(stream, &graph));
        check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        uint32_t previous = bits(a[guard + size_t(context)]), previous_candidate = bits(b[guard + size_t(context)]);
        uint16_t previous_value = output_a[guard], previous_candidate_value = output_b[guard];
        for (float changed : {64.0f, -64.0f}) {
            keys[0] = bf16_from_f32(changed);
            check(cudaMemcpyAsync(dk, keys.data(), sizeof(uint16_t), cudaMemcpyHostToDevice, stream));
            reset(); check(cudaGraphLaunch(executable, stream)); compare_pipeline(rows, context, exceptional);
            const uint32_t current = bits(a[guard + size_t(context)]);
            if (!exceptional) require(current != previous && bits(b[guard + size_t(context)]) != previous_candidate &&
                                      output_a[guard] != previous_value && output_b[guard] != previous_candidate_value &&
                                      std::isfinite(a[guard + size_t(context)]) && std::isfinite(f32_from_bf16(output_b[guard])),
                                      "graph replay ignored changed device input");
            previous = current; previous_candidate = bits(b[guard + size_t(context)]);
            previous_value = output_a[guard]; previous_candidate_value = output_b[guard];
        }
        if ((context == 220 || context == capacity) && begin == capacity - 3 && !exceptional) {
            cudaEvent_t start, end; check(cudaEventCreate(&start)); check(cudaEventCreate(&end));
            for (bool full_pipeline : {false, true}) {
                float times[2]{};
                for (int arm : {0, 1, 1, 0}) {
                    check(cudaEventRecord(start, stream));
                    for (int repeat = 0; repeat < 5; ++repeat) {
                        if (full_pipeline) pipeline(arm != 0); else launch(arm != 0);
                    }
                    check(cudaEventRecord(end, stream)); check(cudaEventSynchronize(end));
                    float elapsed = 0; check(cudaEventElapsedTime(&elapsed, start, end)); times[arm] += elapsed / 10.f;
                }
                std::printf("{\"timing\":\"%s\",\"rows\":%d,\"context\":%d,\"reference_ms\":%.6f,\"candidate_ms\":%.6f}\n",
                    full_pipeline ? "scores_softmax_same_values" : "scores", rows, context, times[0], times[1]);
            }
            check(cudaEventDestroy(start)); check(cudaEventDestroy(end));
        }
        check(cudaGraphExecDestroy(executable)); check(cudaGraphDestroy(graph));
        ring[ring_at] = saved;
        check(cudaMemcpyAsync(dring + ring_at, ring.data() + ring_at, sizeof(uint16_t), cudaMemcpyHostToDevice, stream));
        check(cudaStreamSynchronize(stream));
        std::printf("{\"rows\":%d,\"context\":%d,\"begin\":%d,\"layer\":%d,\"nonfinite_case\":%d,"
                    "\"exact\":true,\"graph_replay\":true,\"guards\":true,\"masks\":true}\n",
                    rows, context, begin, layer, exceptional);
    }
};
} // namespace
int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--help") == 0) {
        std::puts("Queued CUDA attention-score exactness and score/softmax/value timing fixture. Select GPU with CUDA_VISIBLE_DEVICES; optional --off.");
        return 0;
    }
    const bool off = argc == 2 && std::strcmp(argv[1], "--off") == 0;
    require(argc == 1 || off, "only optional --off supported");
    require(setenv("STRATA_DFLASH2_SCORE_GROUPED_HEADS", off ? "0" : "1", 1) == 0, "freeze score dispatch");
    require(setenv("STRATA_DFLASH2_VALUE_SPLIT", "0", 1) == 0 && setenv("STRATA_DFLASH2_VALUE_SPANS", "1", 1) == 0,
            "retained exact value spans in both arms");
    int device = 0; cudaDeviceProp properties{}; check(cudaGetDevice(&device)); check(cudaGetDeviceProperties(&properties, device));
    require((properties.major == 7 && properties.minor == 0) || (properties.major == 8 && properties.minor == 6),
            "queued SM70 or SM86 required");
    std::printf("{\"device\":\"%s\",\"major\":%d,\"minor\":%d}\n", properties.name, properties.major, properties.minor);
    {
    Fixture fixture;
    for (int rows : {1, 4, 8}) {
        fixture.run(rows, 220, 0, 0);
        fixture.run(rows, 220, capacity - 3, 4);
        fixture.run(rows, capacity, 0, 0);
        fixture.run(rows, capacity, capacity - 3, 4);
    }
    fixture.run(1, 0, 0, 4);
    fixture.run(4, 1, capacity - 1, 0);
    for (int context : {3, 4, 5, 27, 28, 29}) fixture.run(4, context, capacity - 3, 4);
    for (int kind : {1, 2, 3, 5}) fixture.run(4, capacity, capacity - 3, 4, kind);
    fixture.run(8, capacity, capacity - 3, 4, 4);
    require(attention_scores(fixture.dq, fixture.dk, fixture.dring, fixture.db, 0, 0, 0, 0, fixture.sb, fixture.stream) == int(cudaErrorInvalidValue), "row guard");
    }
    std::puts("PASS grouped_score exact_scores=1 exact_pipeline=1 graph_changed=1 masks=1 status=1 guards=1");
    return 0;
}
