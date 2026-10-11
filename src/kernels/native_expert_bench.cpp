// src/kernels/native_expert_bench.cpp - the grouped native experts on real GGUF rows: one AMD layout against
// another, bit for bit, and each one's time.
//
//     build/native_expert_bench <shard1.gguf> <layer[,layer...]> <groups> <tokens per group> <ref mode> <mode> [iters]
//
// Modes are STRATA_EXP_MODE values.  A group is one expert (distinct blobs, ~2 MB each, so G >= 8 does not fit
// in L2); its entries read distinct tokens of an 8-token window.  The output must be bitwise equal to the
// reference mode's: the verify window's text depends on it.
//
// HET-033, exact captured subset (no model session or scheduling changes):
//   native_expert_bench --capture-group SHARD ROUTING INPUTS LAYER EXPERTS ROLE REPEATS
//       DEVICE_BUDGET_BYTES HOST_BUDGET_BYTES RUNTIME_ALLOWANCE_BYTES [--plan]
// EXPERTS is `all` or actual routed IDs separated by commas; ROLE is local/helper.
// REPEATS is 1..16. --plan reads headers/capture only, and never initializes CUDA.
// Real input/metadata/output copies and a WHOLE quantize/grouped/weighted-subset
// operation are timed separately, as is the whole packet. No kernel-time sums,
// scaling by token count, or packet-as-compute aliases. JSON is emitted to stdout.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <limits>
#include <memory>
#include <stdexcept>

namespace cpu = strata::kernels::cpu;
namespace K = strata::kernels;

namespace {
uint64_t integer(const char* text, uint64_t maximum, bool positive = true) {
    uint64_t value = 0;
    const auto p = std::from_chars(text, text + std::strlen(text), value);
    if (p.ec != std::errc{} || *p.ptr || value > maximum || (positive && !value))
        throw std::runtime_error("capture-group: invalid finite integer argument");
    return value;
}
using File = std::unique_ptr<FILE, int (*)(FILE*)>;
File open_input(const char* path) {
    File f(std::fopen(path, "rb"), &std::fclose);
    if (!f) throw std::runtime_error("capture-group: cannot read capture");
    return f;
}
void bytes(FILE* f, void* dst, size_t count) {
    if (std::fread(dst, 1, count, f) != count)
        throw std::runtime_error("capture-group: truncated actual capture");
}
void eof(FILE* f) {
    if (std::fgetc(f) != EOF || std::ferror(f))
        throw std::runtime_error("capture-group: trailing or unreadable actual capture");
}
int64_t host_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("capture-group: ") + what + ": " + cudaGetErrorString(e));
}
void json_text(const char* text) {
    std::putchar('"');
    for (const unsigned char* p = (const unsigned char*) text; *p; ++p) {
        if (*p == '"' || *p == '\\') std::putchar('\\');
        if (*p < 32) std::printf("\\u%04x", *p);
        else std::putchar(*p);
    }
    std::putchar('"');
}
uint64_t meta_payload(const strata::MetaValue& v) {
    uint64_t n = v.s.capacity() + 1 + v.items.capacity() * sizeof(strata::MetaValue);
    for (const auto& child : v.items) n += meta_payload(child);
    return n;
}
struct GroupResources {
    cudaStream_t stream = nullptr;
    cudaEvent_t event[8] = {};
    void *blob = nullptr, *input = nullptr, *q8 = nullptr, *metadata = nullptr, *scratch = nullptr;
    void *parts = nullptr, *output = nullptr, *h_input = nullptr, *h_metadata = nullptr, *h_output = nullptr;
    ~GroupResources() {
        if (stream) cudaStreamSynchronize(stream);
        for (auto e : event) if (e) cudaEventDestroy(e);
        for (void* p : {blob, input, q8, metadata, scratch, parts, output}) if (p) cudaFree(p);
        for (void* p : {h_input, h_metadata, h_output}) if (p) cudaFreeHost(p);
        if (stream) cudaStreamDestroy(stream);
    }
};

int capture_group(int argc, char** argv) {
    if ((argc != 12 && argc != 13) || (argc == 13 && std::strcmp(argv[12], "--plan") != 0))
        throw std::runtime_error("usage: native_expert_bench --capture-group SHARD ROUTING INPUTS LAYER "
                                 "EXPERTS(all|id,id) ROLE(local|helper) REPEATS(1..16) "
                                 "DEVICE_BUDGET_BYTES HOST_BUDGET_BYTES RUNTIME_ALLOWANCE_BYTES [--plan]");
#if defined(STRATA_USE_HIP) || defined(STRATA_USE_SYCL) || defined(__HIPCC__)
    throw std::runtime_error("capture-group: this causal capture/packet surface requires CUDA");
#else
    const int layer = (int) integer(argv[5], 47, false);
    const bool local = std::strcmp(argv[7], "local") == 0;
    if (!local && std::strcmp(argv[7], "helper") != 0) throw std::runtime_error("capture-group: role must be local/helper");
    const int repeats = (int) integer(argv[8], 16);
    const uint64_t device_budget = integer(argv[9], INT64_MAX), host_budget = integer(argv[10], INT64_MAX);
    const uint64_t runtime_allowance = integer(argv[11], INT64_MAX);
    const bool plan = argc == 13;
    auto input_file = open_input(argv[4]);
    char magic[8]; bytes(input_file.get(), magic, sizeof magic);
    if (std::memcmp(magic, "SCH03301", 8) != 0) throw std::runtime_error("capture-group: unsupported actual-input capture");
    uint64_t request = 0, window = 0; int64_t position = 0;
    int32_t h[5]; // layer begin/end, actual T, K, H; little-endian Linux/CUDA capture
    bytes(input_file.get(), &request, 8); bytes(input_file.get(), &window, 8);
    bytes(input_file.get(), &position, 8); bytes(input_file.get(), h, sizeof h);
    const int lb = h[0], le = h[1], T = h[2], routed = h[3], H = h[4];
    if (!request || !window || position < 0 || lb < 0 || le > 48 || lb >= le || layer < lb || layer >= le ||
        T < 1 || T > K::kVerifyMaxT || routed != 10 || H <= 0 || H > 8192 || H % 32)
        throw std::runtime_error("capture-group: invalid request/window/native geometry");
    const int cap = T * routed;
    std::vector<float> x((size_t) T * H);
    if (x.capacity() != x.size()) throw std::runtime_error("capture-group: input allocation exceeds priced capacity");
    int32_t shape[4] = {}; uint64_t expert_bytes = 0;
    for (int l = lb; l < le; ++l) {
        int32_t row[4]; uint64_t blob_bytes = 0;
        bytes(input_file.get(), row, sizeof row); bytes(input_file.get(), &blob_bytes, 8);
        if (row[0] != l || row[1] <= 0 || row[1] > 8192 || row[2] < 0 || row[3] < 0 || !blob_bytes)
            throw std::runtime_error("capture-group: missing actual per-layer geometry/format");
        if (l == layer) {
            std::copy(row, row + 4, shape); expert_bytes = blob_bytes;
            bytes(input_file.get(), x.data(), x.size() * sizeof(float));
            for (float value : x) if (!std::isfinite(value)) throw std::runtime_error("capture-group: nonfinite actual input");
        } else {
            float block[1024];
            for (size_t left = (size_t) T * H; left;) {
                const size_t n = std::min(left, (size_t) 1024);
                bytes(input_file.get(), block, n * sizeof(float));
                for (size_t i = 0; i < n; ++i) if (!std::isfinite(block[i]))
                    throw std::runtime_error("capture-group: incomplete actual input transcript");
                left -= n;
            }
        }
    }
    eof(input_file.get());
    auto routing_file = open_input(argv[3]);
    std::array<int32_t, 80> ids{}; std::array<float, 80> weights{};
    for (int l = lb; l < le; ++l) for (int t = 0; t < T; ++t) {
        int32_t header[2], row_ids[10]; float row_weights[10];
        bytes(routing_file.get(), header, sizeof header);
        if (header[0] != l || header[1] != routed) throw std::runtime_error("capture-group: routing/window binding differs");
        bytes(routing_file.get(), row_ids, sizeof row_ids); bytes(routing_file.get(), row_weights, sizeof row_weights);
        float total = 0;
        for (int k = 0; k < routed; ++k) {
            if (row_ids[k] < 0 || row_ids[k] >= 1024 || !std::isfinite(row_weights[k]) || row_weights[k] < 0)
                throw std::runtime_error("capture-group: actual routing is missing or malformed");
            for (int j = 0; j < k; ++j) if (row_ids[k] == row_ids[j])
                throw std::runtime_error("capture-group: duplicate actual routed ID");
            total += row_weights[k];
        }
        if (!(total > 0) || !std::isfinite(total)) throw std::runtime_error("capture-group: placeholder/missing route weights");
        if (l == layer) for (int k = 0; k < routed; ++k) {
            ids[(size_t) t * routed + k] = row_ids[k]; weights[(size_t) t * routed + k] = row_weights[k];
        }
    }
    eof(routing_file.get());
    bool selected[1024] = {};
    if (std::strcmp(argv[6], "all") == 0) for (int i = 0; i < cap; ++i) selected[ids[i]] = true;
    else {
        const char* p = argv[6];
        if (!*p) throw std::runtime_error("capture-group: empty expert subset");
        while (*p) {
            const char* end = std::strchr(p, ','); if (!end) end = p + std::strlen(p);
            uint64_t id = 0; const auto parsed = std::from_chars(p, end, id);
            if (parsed.ec != std::errc{} || parsed.ptr != end || id >= 1024 || selected[id] ||
                std::find(ids.begin(), ids.begin() + cap, (int32_t) id) == ids.begin() + cap)
                throw std::runtime_error("capture-group: subset IDs must be distinct actually routed experts");
            selected[id] = true;
            if (!*end) break;
            p = end + 1; if (!*p) throw std::runtime_error("capture-group: trailing expert separator");
        }
    }
    std::array<int32_t, 80> group_ids{}, counts{}, rank{}, original{};
    int G = 0, E = 0;
    for (int i = 0; i < cap; ++i) if (selected[ids[i]]) {
        rank[i] = E; original[E++] = i;
        if (std::find(group_ids.begin(), group_ids.begin() + G, ids[i]) == group_ids.begin() + G)
            group_ids[G++] = ids[i];
    }
    if (!G || !E) throw std::runtime_error("capture-group: no actual routed group");
    strata::GgufFile gguf(argv[2]);
    const strata::TensorInfo* tensor[3] = {};
    const char* role[3] = {"gate", "up", "down"};
    for (const auto& ti : gguf.tensors()) for (int r = 0; r < 3; ++r)
        if (ti.name == "blk." + std::to_string(layer) + ".ffn_" + role[r] + "_exps.weight") tensor[r] = &ti;
    for (int r = 0; r < 3; ++r) if (!tensor[r] || tensor[r]->shape.size() != 3)
        throw std::runtime_error("capture-group: real expert tensors are missing from the selected shard");
    const int FF = shape[1];
    const uint64_t experts = tensor[0]->shape[2];
    if (tensor[0]->shape[0] != (uint64_t) H || tensor[0]->shape[1] != (uint64_t) FF ||
        tensor[1]->shape != tensor[0]->shape || tensor[1]->type != tensor[0]->type ||
        tensor[2]->shape[0] != (uint64_t) FF || tensor[2]->shape[1] != (uint64_t) H ||
        tensor[2]->shape[2] != experts || (int) tensor[0]->type != shape[2] || (int) tensor[2]->type != shape[3])
        throw std::runtime_error("capture-group: shard geometry/formats differ from the actual captured operation");
    cpu::NativeFmt fmt; std::string err;
    if (!cpu::native_fmt(shape[2], shape[3], H, FF, fmt, err) || fmt.bytes != expert_bytes ||
        !K::native_expert_supported(shape[2], shape[3], H, FF))
        throw std::runtime_error("capture-group: captured native format is unsupported or differs: " + err);
    for (int g = 0; g < G; ++g) if ((uint64_t) group_ids[g] >= experts)
        throw std::runtime_error("capture-group: actual routed expert is outside the shard");
    const uint64_t per_tensor[] = {fmt.up_off, fmt.down_off - fmt.up_off, fmt.bytes - fmt.down_off};
    for (int r = 0; r < 3; ++r) {
        const uint64_t available = gguf.file_size() - gguf.data_start();
        if (tensor[r]->offset > available || experts > (available - tensor[r]->offset) / per_tensor[r])
            throw std::runtime_error("capture-group: real expert tensor payload is truncated");
    }
    const int cap_groups = local ? cap : G, cap_entries = local ? cap : E;
    const uint64_t input_bytes = (uint64_t) T * H * sizeof(float), output_bytes = input_bytes;
    const uint64_t metadata_bytes = (uint64_t) E * 40;
    const uint64_t blob_bytes = (uint64_t) G * fmt.bytes, scratch_bytes = K::native_expert_scratch_bytes(cap_entries, FF);
    const uint64_t q8_bytes = (uint64_t) T * (H / 32) * 36, parts_bytes = (uint64_t) cap_entries * H * sizeof(float);
    const uint64_t device_payload = blob_bytes + input_bytes + output_bytes + metadata_bytes + q8_bytes + parts_bytes + scratch_bytes;
    uint64_t reader_payload = gguf.tensors().capacity() * sizeof(strata::TensorInfo);
    for (const auto& t : gguf.tensors()) reader_payload += t.name.capacity() + 1 + t.shape.capacity() * sizeof(uint64_t);
    for (const auto& kv : gguf.metadata())
        reader_payload += sizeof(kv) + 64 + kv.first.capacity() + 1 + meta_payload(kv.second);
    const uint64_t host_pinned = input_bytes + output_bytes + metadata_bytes;
    const uint64_t host_payload = 2 * blob_bytes + (uint64_t) G * 3 * 4096 + 2 * input_bytes +
                                 host_pinned + reader_payload + gguf.data_start() + (64ull << 10);
    if (device_payload > device_budget || runtime_allowance > device_budget - device_payload ||
        host_payload > host_budget || runtime_allowance > host_budget - host_payload)
        throw std::runtime_error("capture-group: exact payload plus explicit opaque runtime allowance exceeds budget");
    auto identity = [&] {
        std::printf("\"version\":1,\"request\":\"%llu\",\"window\":%llu,\"position\":%lld,"
                    "\"phase\":\"verify\",\"batch_width\":%d,\"verify_width\":%d,\"layer\":%d,"
                    "\"n_embd\":%d,\"n_ff\":%d,\"k\":%d,\"role\":\"%s\",\"groups\":%d,\"entries\":%d,"
                    "\"cap_groups\":%d,\"cap_entries\":%d,\"gate_up_format\":%d,\"down_format\":%d,"
                    "\"input_format\":\"F32\",\"activation_format\":\"Q8_1\",\"output_format\":\"F32\","
                    "\"operator\":\"native_grouped-quantize-weighted-subset-fma-route-order\","
                    "\"grouped_operator\":\"native-grouped-gate-up-swiglu-down\","
                    "\"input_bytes\":%llu,\"metadata_bytes\":%llu,\"metadata_bytes_per_route\":40,"
                    "\"output_bytes\":%llu,\"device_payload_bytes\":%llu,\"host_payload_bytes\":%llu,"
                    "\"host_pinned_bytes\":%llu,\"runtime_storage_allowance_bytes\":%llu,"
                    "\"device_budget_bytes\":%llu,\"host_budget_bytes\":%llu,\"source_shard\":",
                    (unsigned long long) request, (unsigned long long) window, (long long) position,
                    T, T, layer, H, FF, routed, local ? "local" : "helper", G, E, cap_groups, cap_entries,
                    fmt.gu_type, fmt.d_type, (unsigned long long) input_bytes, (unsigned long long) metadata_bytes,
                    (unsigned long long) output_bytes, (unsigned long long) device_payload,
                    (unsigned long long) host_payload, (unsigned long long) host_pinned,
                    (unsigned long long) runtime_allowance, (unsigned long long) device_budget,
                    (unsigned long long) host_budget);
        json_text(argv[2]);
        std::printf(",\"expert_ids\":[");
        for (int g = 0; g < G; ++g) std::printf("%s%d", g ? "," : "", group_ids[g]);
        for (int g = 0; g < G; ++g) for (int i = 0; i < cap; ++i) if (selected[ids[i]] && ids[i] == group_ids[g]) ++counts[g];
        std::sort(counts.begin(), counts.begin() + G);
        std::printf("],\"expert_token_counts\":[");
        for (int g = 0; g < G; ++g) std::printf("%s%d", g ? "," : "", counts[g]);
        std::printf("]");
    };
    if (plan) {
        std::printf("{"); identity();
        std::printf(",\"kind\":\"allocation-plan\",\"measurement\":false,"
                    "\"note\":\"opaque runtime/event/stream/allocator storage is an operator allowance, "
                    "not a measured allocation; no CUDA initialization or timed execution\"}\n");
        return std::ferror(stdout) ? 1 : 0;
    }
    const char* visibility = std::getenv("CUDA_VISIBLE_DEVICES");
    if (!visibility || std::strlen(visibility) != 40 || std::strncmp(visibility, "GPU-", 4) != 0)
        throw std::runtime_error("capture-group: pin exactly one actual GPU UUID in CUDA_VISIBLE_DEVICES");
    int visible = 0; cuda_check(cudaGetDeviceCount(&visible), "device count");
    if (visible != 1) throw std::runtime_error("capture-group: exactly one GPU is required");
    cudaDeviceProp prop{}; cuda_check(cudaGetDeviceProperties(&prop, 0), "device properties");
    char actual_uuid[41]; char* out = actual_uuid;
    std::memcpy(out, "GPU-", 4); out += 4;
    for (int i = 0; i < 16; ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) *out++ = '-';
        std::snprintf(out, 3, "%02x", (unsigned char) prop.uuid.bytes[i]); out += 2;
    }
    *out = '\0';
    if (std::strcmp(actual_uuid, visibility) != 0) throw std::runtime_error("capture-group: observed UUID differs from the pin");
    size_t free_bytes = 0, total_bytes = 0; cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes), "available memory");
    if (device_payload + runtime_allowance > free_bytes)
        throw std::runtime_error("capture-group: available GPU memory cannot admit the finite packet diagnostic");
    GroupResources r;
    cuda_check(cudaStreamCreateWithFlags(&r.stream, cudaStreamNonBlocking), "stream");
    for (auto& e : r.event) cuda_check(cudaEventCreate(&e), "packet event");
    auto alloc = [&](void** dst, uint64_t n) { cuda_check(cudaMalloc(dst, (size_t) n), "priced device allocation"); };
    alloc(&r.blob, blob_bytes); alloc(&r.input, input_bytes); alloc(&r.output, output_bytes);
    alloc(&r.q8, q8_bytes); alloc(&r.metadata, metadata_bytes); alloc(&r.parts, parts_bytes); alloc(&r.scratch, scratch_bytes);
    cuda_check(cudaHostAlloc(&r.h_input, input_bytes, cudaHostAllocDefault), "priced input staging");
    cuda_check(cudaHostAlloc(&r.h_output, output_bytes, cudaHostAllocDefault), "priced output staging");
    cuda_check(cudaHostAlloc(&r.h_metadata, metadata_bytes, cudaHostAllocDefault), "priced metadata staging");
    std::vector<uint8_t> blobs((size_t) blob_bytes);
    if (blobs.capacity() != blobs.size()) throw std::runtime_error("capture-group: weight allocation exceeds priced capacity");
    for (int g = 0; g < G; ++g) {
        uint8_t* dst = blobs.data() + (size_t) g * fmt.bytes;
        for (int part = 0; part < 3; ++part) {
            const size_t offset = part == 0 ? 0 : part == 1 ? fmt.up_off : fmt.down_off;
            std::memcpy(dst + offset, gguf.tensor_data(*tensor[part]) + (size_t) group_ids[g] * per_tensor[part], per_tensor[part]);
        }
    }
    cuda_check(cudaMemcpyAsync(r.blob, blobs.data(), blob_bytes, cudaMemcpyHostToDevice, r.stream), "resident expert upload");
    alignas(8) std::array<uint8_t, 80 * 40> meta{};
    auto* pointers = (unsigned long long*) meta.data();
    auto* start = (int32_t*) (pointers + G);
    auto* dst = start + G + 1;
    auto* tok = dst + E;
    auto* original_meta = tok + E;
    auto* weight = (float*) (original_meta + E);
    auto* n_groups = (int32_t*) (weight + E);
    if ((const uint8_t*) (n_groups + 1) > meta.data() + metadata_bytes)
        throw std::runtime_error("capture-group: metadata exceeds its 40-byte routed-token packet");
    int entry = 0;
    for (int g = 0; g < G; ++g) {
        pointers[g] = (unsigned long long) ((uint8_t*) r.blob + (size_t) g * fmt.bytes);
        start[g] = entry;
        for (int i = 0; i < cap; ++i) if (selected[ids[i]] && ids[i] == group_ids[g]) {
            dst[entry] = rank[i]; tok[entry++] = i / routed;
        }
    }
    start[G] = E; *n_groups = G;
    for (int i = 0; i < E; ++i) { original_meta[i] = original[i]; weight[i] = weights[original[i]]; }
    auto device_pointer = [&](const void* p) {
        return (uint8_t*) r.metadata + ((const uint8_t*) p - meta.data());
    };
    const auto layout = K::native_expert_layout(fmt.gu_type, fmt.d_type, H, FF);
    struct Sample {
        int64_t input = 0, metadata = 0, compute = 0, output = 0, packet = 0, host = 0, staging = 0, grouped = 0;
    };
    std::array<Sample, 16> samples{};
    std::vector<float> reference((size_t) T * H);
    if (reference.capacity() != reference.size()) throw std::runtime_error("capture-group: output allocation exceeds priced capacity");
    K::native_expert_set_mode(-1, 0); // entire real operator, never GU/down-only timing modes
    for (int round = -1; round < repeats; ++round) {
        const int64_t began = host_ns();
        std::memcpy(r.h_input, x.data(), input_bytes);
        std::memcpy(r.h_metadata, meta.data(), metadata_bytes);
        const int64_t staged = host_ns();
        cuda_check(cudaEventRecord(r.event[0], r.stream), "packet begin");
        cuda_check(cudaMemcpyAsync(r.input, r.h_input, input_bytes, cudaMemcpyHostToDevice, r.stream), "actual input copy");
        cuda_check(cudaEventRecord(r.event[1], r.stream), "input end");
        cuda_check(cudaMemcpyAsync(r.metadata, r.h_metadata, metadata_bytes, cudaMemcpyHostToDevice, r.stream), "actual metadata copy");
        cuda_check(cudaEventRecord(r.event[2], r.stream), "metadata end");
        K::quantize_q8_1_rows((const float*) r.input, T, H, r.q8, r.stream);
        cuda_check(cudaEventRecord(r.event[6], r.stream), "whole grouped operator begin");
        K::native_expert_grouped(layout, (const unsigned long long*) device_pointer(pointers),
            (const int32_t*) device_pointer(start), (const int32_t*) device_pointer(n_groups),
            (const int32_t*) device_pointer(dst), (const int32_t*) device_pointer(tok),
            cap_groups, cap_entries, r.q8, r.scratch, (float*) r.parts, r.stream);
        cuda_check(cudaEventRecord(r.event[7], r.stream), "whole grouped operator end");
        K::expert_weighted_subset((const float*) r.parts, (const int32_t*) device_pointer(original_meta),
            (const float*) device_pointer(weight), (float*) r.output, H, routed, T, E, r.stream);
        cuda_check(cudaGetLastError(), "whole grouped compute");
        cuda_check(cudaEventRecord(r.event[3], r.stream), "whole compute end");
        cuda_check(cudaMemcpyAsync(r.h_output, r.output, output_bytes, cudaMemcpyDeviceToHost, r.stream), "actual weighted output copy");
        cuda_check(cudaEventRecord(r.event[4], r.stream), "output end");
        cuda_check(cudaEventRecord(r.event[5], r.stream), "whole packet end");
        cuda_check(cudaEventSynchronize(r.event[5]), "packet drain");
        const int64_t ended = host_ns();
        const float* values = (const float*) r.h_output;
        for (size_t i = 0; i < reference.size(); ++i) if (!std::isfinite(values[i]))
            throw std::runtime_error("capture-group: whole grouped weighted output is nonfinite");
        if (round == -1) { std::memcpy(reference.data(), values, output_bytes); continue; }
        if (std::memcmp(reference.data(), values, output_bytes) != 0)
            throw std::runtime_error("capture-group: repeated exact operation differs bitwise");
        auto duration = [&](int from, int to) {
            float ms = 0; cuda_check(cudaEventElapsedTime(&ms, r.event[from], r.event[to]), "measured duration");
            if (!(ms > 0) || !std::isfinite(ms)) throw std::runtime_error("capture-group: missing/nonpositive packet measurement");
            const int64_t ns = (int64_t) std::llround((double) ms * 1000000.0);
            if (ns <= 0) throw std::runtime_error("capture-group: measured duration rounds to zero nanoseconds");
            return ns;
        };
        samples[round] = {duration(0, 1), duration(1, 2), duration(2, 3), duration(3, 4),
                          duration(0, 5), ended - began, staged - began, duration(6, 7)};
    }
    std::printf("{"); identity();
    std::printf(",\"kind\":\"measured-whole-group-packet\",\"measurement\":true,\"device\":\"%s\","
                "\"backend\":\"CUDA-native-grouped\",\"compute_capability\":\"%d.%d\","
                "\"clock\":\"cuda-event-duration-ns\",\"host_clock\":\"host-monotonic-ns\","
                "\"warmup_packets\":1,\"repeat_bitwise_equal\":true,\"kernel_settings\":{",
                actual_uuid, prop.major, prop.minor);
    bool first_setting = true;
    for (const char* name : {"STRATA_EXP_MODE", "STRATA_GROUPED_V1", "STRATA_OLD_IQ_MMVQ",
                            "STRATA_NO_SUB16_GU", "STRATA_IQ_STAGE_GRID", "STRATA_EXPERT_V2",
                            "STRATA_EXPERT_V2K", "STRATA_TSUM", "STRATA_IQ_MT_MIN"}) {
        if (!first_setting) std::putchar(',');
        first_setting = false;
        json_text(name); std::putchar(':');
        if (const char* value = std::getenv(name)) json_text(value); else std::printf("null");
    }
    std::printf("},\"samples\":[");
    for (int i = 0; i < repeats; ++i) {
        const auto& s = samples[i];
        std::printf("%s{\"input_copy_ns\":%lld,\"metadata_copy_ns\":%lld,\"compute_ns\":%lld,"
                    "\"output_copy_ns\":%lld,\"packet_ns\":%lld,\"host_packet_ns\":%lld,"
                    "\"host_staging_ns\":%lld,\"grouped_ns\":%lld}",
                    i ? "," : "", (long long) s.input, (long long) s.metadata, (long long) s.compute,
                    (long long) s.output, (long long) s.packet, (long long) s.host,
                    (long long) s.staging, (long long) s.grouped);
    }
    std::printf("],\"note\":\"finite exact captured shape; separate activation/metadata transfers; resident selected "
                "weights uploaded before timing; no CPU-kernel equivalence, pipelining, concurrent-serving or "
                "cross-help adoption claim; opaque runtime storage remains an explicit allowance\"}\n");
    return std::ferror(stdout) ? 1 : 0;
#endif
}
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc > 1 && std::strcmp(argv[1], "--capture-group") == 0) {
        try { return capture_group(argc, argv); }
        catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
    }
    if (argc < 7) {
        std::fprintf(stderr, "usage: native_expert_bench <shard1.gguf> <layer[,layer]> <groups> <tokens> <ref mode> <mode> [iters]\n");
        return 2;
    }
    strata::GgufFile gguf(argv[1]);
    std::vector<int> layers;
    {
        std::stringstream ss(argv[2]);
        std::string t;
        while (std::getline(ss, t, ',')) layers.push_back(std::atoi(t.c_str()));
    }
    const int G = std::atoi(argv[3]), T = std::atoi(argv[4]), mref = std::atoi(argv[5]), mode = std::atoi(argv[6]);
    const int iters = argc > 7 ? std::atoi(argv[7]) : 200;
    const int NTOK = 8;
    if (T < 1 || T > NTOK || G < 1) { std::fprintf(stderr, "tokens 1..8, groups >= 1\n"); return 2; }
    const int64_t H = 2560, FF_FULL = 640, FF = FF_FULL;   // Flash-Next's expert geometry
    int failures = 0;
    cudaStream_t s;
    cudaStreamCreate(&s);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    for (int l : layers) {
        const strata::TensorInfo* t[3] = {};
        const char* roles[3] = {"gate", "up", "down"};
        for (const auto& ti : gguf.tensors())
            for (int r = 0; r < 3; ++r)
                if (ti.name == "blk." + std::to_string(l) + ".ffn_" + roles[r] + "_exps.weight") t[r] = &ti;
        if (!t[0] || !t[1] || !t[2]) { std::printf("layer %d: no expert tensors\n", l); ++failures; continue; }
        cpu::NativeFmt f;
        std::string err;
        if (!cpu::native_fmt((int) t[0]->type, (int) t[2]->type, H, FF, f, err)) {
            std::printf("layer %d: %s\n", l, err.c_str()); ++failures; continue;
        }
        cpu::NativeFmt ff;    // the full expert's geometry, for strides into the GGUF
        if (!cpu::native_fmt((int) t[0]->type, (int) t[2]->type, H, FF_FULL, ff, err)) {
            std::printf("layer %d: %s\n", l, err.c_str()); ++failures; continue;
        }
        const size_t dsz = f.bytes - f.down_off, dsz_full = ff.bytes - ff.down_off;
        const size_t drow = dsz / H, drow_full = dsz_full / H;
        std::vector<uint8_t> blobs((size_t) G * f.bytes);
        for (int g = 0; g < G; ++g) {
            const size_t E = (size_t) ((g * 37 + 5) % 256);
            uint8_t* b = blobs.data() + (size_t) g * f.bytes;
            std::memcpy(b, gguf.tensor_data(*t[0]) + E * ff.up_off, f.up_off);
            std::memcpy(b + f.up_off, gguf.tensor_data(*t[1]) + E * ff.up_off, f.up_off);
            for (int64_t r = 0; r < H; ++r)
                std::memcpy(b + f.down_off + r * drow, gguf.tensor_data(*t[2]) + E * dsz_full + r * drow_full, drow);
        }
        std::mt19937 rng(17 + l);
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> x((size_t) NTOK * H);
        for (auto& v : x) v = nd(rng);
        const int NE = G * T;
        std::vector<unsigned long long> ptr(G);
        std::vector<int32_t> start(G + 1), dst(NE), tok(NE);
        const auto L = K::native_expert_layout(f.gu_type, f.d_type, H, FF);
        uint8_t* dblob;
        void *dx, *dxq, *dscr;
        float* dout;
        unsigned long long* dptr;
        int32_t *dstart, *dn, *ddst, *dtok;
        cudaMalloc((void**) &dblob, blobs.size());
        cudaMalloc(&dx, x.size() * 4);
        cudaMalloc(&dxq, (size_t) NTOK * H / 32 * 36);
        cudaMalloc(&dscr, K::native_expert_scratch_bytes(NE, FF));
        cudaMalloc((void**) &dout, (size_t) NE * H * 4);
        cudaMalloc((void**) &dptr, (size_t) G * 8);
        cudaMalloc((void**) &dstart, (size_t) (G + 1) * 4);
        cudaMalloc((void**) &dn, 4);
        cudaMalloc((void**) &ddst, (size_t) NE * 4);
        cudaMalloc((void**) &dtok, (size_t) NE * 4);
        for (int g = 0; g < G; ++g) {
            ptr[g] = (unsigned long long) (dblob + (size_t) g * f.bytes);
            start[g] = g * T;
            for (int j = 0; j < T; ++j) {
                dst[g * T + j] = g * T + j;
                tok[g * T + j] = (g * 3 + j) % NTOK;
            }
        }
        start[G] = NE;
        cudaMemcpy(dblob, blobs.data(), blobs.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dptr, ptr.data(), (size_t) G * 8, cudaMemcpyHostToDevice);
        cudaMemcpy(dstart, start.data(), (size_t) (G + 1) * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dn, &G, 4, cudaMemcpyHostToDevice);
        cudaMemcpy(ddst, dst.data(), (size_t) NE * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dtok, tok.data(), (size_t) NE * 4, cudaMemcpyHostToDevice);
        K::quantize_q8_1_rows((const float*) dx, NTOK, H, dxq, s);
        auto run = [&](int m, int phase) {
            K::native_expert_set_mode(m, phase);
            K::native_expert_grouped(L, dptr, dstart, dn, ddst, dtok, G, NE, dxq, dscr, dout, s);
        };
        std::vector<float> ref((size_t) NE * H), got((size_t) NE * H);
        cudaMemsetAsync(dout, 0xff, got.size() * 4, s);
        run(mref, 0);
        cudaMemcpyAsync(ref.data(), dout, ref.size() * 4, cudaMemcpyDeviceToHost, s);
        cudaMemsetAsync(dout, 0xff, got.size() * 4, s);
        run(mode, 0);
        cudaMemcpyAsync(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost, s);
        if (cudaStreamSynchronize(s) != cudaSuccess) { std::printf("layer %d: CUDA error\n", l); return 1; }
        size_t ndiff = 0;
        double worst = 0;
        for (size_t i = 0; i < ref.size(); ++i)
            if (std::memcmp(&ref[i], &got[i], 4) != 0) {
                ++ndiff;
                worst = std::fmax(worst, std::fabs((double) ref[i] - got[i]) / (std::fabs((double) ref[i]) + 1e-6));
            }
        auto time = [&](int m, int phase) {
            for (int i = 0; i < 10; ++i) run(m, phase);
            cudaEventRecord(e0, s);
            for (int i = 0; i < iters; ++i) run(m, phase);
            cudaEventRecord(e1, s);
            cudaEventSynchronize(e1);
            float ms = 0;
            cudaEventElapsedTime(&ms, e0, e1);
            return 1e3 * ms / iters;
        };
        const double gu_b = (double) G * 2 * f.up_off, d_b = (double) G * dsz;
        double us[2][3] = {{1e30, 1e30, 1e30}, {1e30, 1e30, 1e30}};
        for (int round = 0; round < 3; ++round)   // interleaved, the minimum: the clocks wander between runs
            for (int k = 0; k < 2; ++k)
                for (int p = 0; p < 3; ++p) us[k][p] = std::fmin(us[k][p], time(k ? mode : mref, p));
        std::printf("layer %2d %-7s/%-6s G %d T %d | mode %d: all %7.1f us  gu %7.1f us (%3.0f GB/s)  down %6.1f us (%3.0f GB/s)"
                    " | mode %d: all %7.1f us  gu %7.1f us (%3.0f GB/s)  down %6.1f us (%3.0f GB/s) | %s\n",
                    l, ggml_type_name((ggml_type) f.gu_type), ggml_type_name((ggml_type) f.d_type), G, T,
                    mref, us[0][0], us[0][1], gu_b / us[0][1] / 1e3, us[0][2], d_b / us[0][2] / 1e3,
                    mode, us[1][0], us[1][1], gu_b / us[1][1] / 1e3, us[1][2], d_b / us[1][2] / 1e3,
                    ndiff ? "DIFFERS" : "bitwise equal");
        if (ndiff) {
            std::printf("          %zu of %zu outputs differ, worst rel %.2e\n", ndiff, ref.size(), worst);
            ++failures;
        }
        K::native_expert_set_mode(-1, 0);
        cudaFree(dblob); cudaFree(dx); cudaFree(dxq); cudaFree(dscr); cudaFree(dout); cudaFree(dptr);
        cudaFree(dstart); cudaFree(dn); cudaFree(ddst); cudaFree(dtok);
    }
    std::printf("native_expert_bench: %d differing\n", failures);
    return failures ? 1 : 0;
}
