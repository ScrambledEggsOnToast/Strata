// Actual NativeEmbed staging only; not an inference, service or quality benchmark.
// The queue binds this driver, native libraries, CUDA runtime and actual target identity.
// Isolate placement: both arms retain the existing eight-row conversion, allocation
// capacity and embedding completion fence. The control also copies GPU0 -> GPU2.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/native_head.hpp"
#include "strata/core/on_device.hpp"
#include "strata/kernels/elementwise.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
using strata::core::OnDevice;
using Clock = std::chrono::steady_clock;
constexpr int source_device = 0, destination_device = 2;
constexpr int hidden = 2560, vocabulary = 248320, capacity = 8;
constexpr int anchor = 271, mask = 248077, repetitions = 128;
constexpr size_t elements = size_t(capacity) * hidden;
constexpr size_t bf16_bytes = elements * sizeof(uint16_t);
using Output = std::array<uint16_t, elements>;

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
void checked(cudaError_t result, const char* operation) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}
void require_current_device(int expected) {
    int actual = -1;
    checked(cudaGetDevice(&actual), "identify current CUDA device");
    require(actual == expected, "component resource is on the wrong CUDA device");
}
struct Buffer {
    int device;
    void* pointer = nullptr;
    Buffer(int owner, size_t bytes) : device(owner) {
        const OnDevice on(device);
        require_current_device(device);
        checked(cudaMalloc(&pointer, bytes), "allocate component buffer");
    }
    ~Buffer() noexcept {
        const OnDevice on(device);
        require_current_device(device);
        if (pointer) checked(cudaFree(pointer), "free component buffer");
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    template<class T> T* as() const { return static_cast<T*>(pointer); }
};
struct Stream {
    int device;
    cudaStream_t value = nullptr;
    explicit Stream(int owner) : device(owner) {
        const OnDevice on(device);
        require_current_device(device);
        checked(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking), "create component stream");
    }
    ~Stream() noexcept {
        const OnDevice on(device);
        require_current_device(device);
        if (value) {
            // An exception here terminates: never release buffers after an unproved drain.
            checked(cudaStreamSynchronize(value), "drain component stream before teardown");
            checked(cudaStreamDestroy(value), "destroy component stream");
        }
    }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
};
struct Endpoint {
    int device;
    Buffer tokens, raw, bf16;
    Stream stream; // destroyed and drained before its buffers
    explicit Endpoint(int owner) : device(owner), tokens(owner, capacity * sizeof(int32_t)),
        raw(owner, elements * sizeof(float)), bf16(owner, bf16_bytes), stream(owner) {}
};

// The destination join is a component boundary, not a proposed production fence.
// It is present in both arms: enqueue-only peer-copy timing is not completion.
double stage(const strata::core::NativeEmbed& embedding, Endpoint& from, Endpoint& destination,
             const std::array<int32_t, capacity>& ids, int live, bool active_nonfinite = false) {
    const auto start = Clock::now();
    {
        const OnDevice on(from.device);
        checked(cudaMemcpyAsync(from.tokens.pointer, ids.data(), size_t(live) * sizeof(int32_t),
                                cudaMemcpyHostToDevice, from.stream.value), "upload actual token ids");
        embedding.gather_dev(from.tokens.as<int32_t>(), live, from.raw.as<float>(), from.stream.value);
        if (active_nonfinite) {
            static constexpr float infinity = std::numeric_limits<float>::infinity();
            checked(cudaMemcpyAsync(from.raw.pointer, &infinity, sizeof(infinity),
                                    cudaMemcpyHostToDevice, from.stream.value), "inject active nonfinite");
        }
        strata::kernels::f32_to_bf16_bulk(from.raw.as<float>(), from.bf16.as<uint16_t>(),
                                         elements, from.stream.value);
        checked(cudaStreamSynchronize(from.stream.value), "existing embedding completion");
    }
    {
        const OnDevice on(destination.device);
        if (from.device != destination.device)
            checked(cudaMemcpyPeerAsync(destination.bf16.pointer, destination.device,
                                        from.bf16.pointer, from.device, bf16_bytes,
                                        destination.stream.value), "unchanged capacity peer transfer");
        checked(cudaStreamSynchronize(destination.stream.value), "destination-complete staging boundary");
    }
    return std::chrono::duration<double, std::micro>(Clock::now() - start).count();
}
void read_output(const Endpoint& destination, Output& result) {
    const OnDevice on(destination.device);
    checked(cudaMemcpy(result.data(), destination.bf16.pointer, bf16_bytes, cudaMemcpyDeviceToHost),
            "read completed output");
}
bool bf16_is_finite(uint16_t value) { return (value & 0x7f80u) != 0x7f80u; }
void sample(const char* name, const strata::core::NativeEmbed& embedding, Endpoint& from,
            Endpoint& destination, const std::array<int32_t, capacity>& ids) {
    for (int i = 0; i < 4; ++i) stage(embedding, from, destination, ids, 4);
    std::array<double, repetitions> times{};
    double total = 0;
    for (double& value : times) { value = stage(embedding, from, destination, ids, 4); total += value; }
    std::sort(times.begin(), times.end());
    std::printf("{\"kind\":\"sample\",\"arm\":\"%s\",\"calls\":%d,\"live_rows\":4,"
                "\"mean_us\":%.9f,\"median_us\":%.9f,\"p95_us\":%.9f,\"max_us\":%.9f}\n",
                name, repetitions, total / repetitions,
                (times[repetitions / 2 - 1] + times[repetitions / 2]) / 2,
                times[(repetitions * 95 + 99) / 100 - 1], times.back());
}
} // namespace

static int run_component(int argc, char** argv) {
    try {
        require(argc == 7 && std::strcmp(argv[1], "--native") == 0 &&
                std::strcmp(argv[3], "--embedding-sha256") == 0 &&
                std::strcmp(argv[5], "--captured-embeddings") == 0, "invalid fixed component arguments; use --help");
        const std::string expected = argv[4];
        require(expected.size() == 64 && expected.find_first_not_of("0123456789abcdef") == std::string::npos,
                "expected embedding digest must be canonical SHA256");
        static_assert(std::endian::native == std::endian::little);
        std::array<uint16_t, 4 * hidden> captured{};
        std::ifstream capture(argv[6], std::ios::binary | std::ios::ate);
        require(bool(capture) && capture.tellg() == std::streamoff(sizeof(captured)),
                "retained embedding reference has the wrong extent");
        capture.seekg(0);
        capture.read(reinterpret_cast<char*>(captured.data()), sizeof(captured));
        require(bool(capture) && capture.gcount() == std::streamsize(sizeof(captured)),
                "retained embedding reference is truncated");
        int count = 0;
        checked(cudaGetDeviceCount(&count), "enumerate bound CUDA devices");
        require(count == 3, "component requires the three explicitly bound visible devices");
        for (const int device : {source_device, destination_device}) {
            cudaDeviceProp properties{};
            checked(cudaGetDeviceProperties(&properties, device), "query CUDA device properties");
            require(properties.unifiedAddressing && properties.canMapHostMemory,
                    "cross-device host embedding requires mapped host memory and unified addressing");
            std::printf("{\"kind\":\"device\",\"ordinal\":%d,\"major\":%d,\"minor\":%d,"
                        "\"unified_addressing\":true,\"can_map_host\":true}\n",
                        device, properties.major, properties.minor);
        }
        checked(cudaSetDevice(source_device), "select original embedding loader device");
        strata::core::NativeEmbed embedding;
        std::string error;
        const auto shards = strata::gguf_split_paths(argv[2]);
        if (!embedding.load(shards, hidden, vocabulary, error, true)) throw std::runtime_error(error);
        require(embedding.loaded() && embedding.mapped_host(), "VRAM fallback is ineligible for direct local gathering");
        require(embedding.device() == source_device && embedding.identity().sha256 == expected,
                "actual embedding load identity differs from the frozen target");
        const auto identity = embedding.identity();
        std::printf("{\"kind\":\"embedding\",\"sha256\":\"%s\",\"bytes\":%llu,\"type\":%d,"
                    "\"mapped_host\":true,\"loader_device\":0,\"anchor\":271,\"mask\":248077}\n",
                    identity.sha256.c_str(), static_cast<unsigned long long>(embedding.bytes()), embedding.type());
        const auto source_shard = std::find_if(shards.begin(), shards.end(), [&](const std::string& path) {
            return strata::core::native_source_identity(path) == identity.source;
        });
        require(source_shard != shards.end(), "fresh source receipt is outside the selected native shards");
        std::printf("{\"kind\":\"source_receipt\",\"shard_index\":%zu,\"load_generation\":%llu,\"stamp\":[",
                    size_t(source_shard - shards.begin()), static_cast<unsigned long long>(identity.generation));
        for (size_t i = 0; i < identity.source.stamp.size(); ++i)
            std::printf("%s%llu", i ? "," : "", static_cast<unsigned long long>(identity.source.stamp[i]));
        std::puts("]}");
        std::array<int32_t, capacity> ids{};
        ids.fill(mask); ids[0] = anchor;
        std::array<float, elements> poison{};
        poison.fill(std::bit_cast<float>(uint32_t{0x7fc00001}));
        Endpoint control(source_device), local(destination_device);
        for (Endpoint* endpoint : {&control, &local}) {
            const OnDevice on(endpoint->device);
            checked(cudaMemcpy(endpoint->raw.pointer, poison.data(), sizeof(poison), cudaMemcpyHostToDevice),
                    "poison inactive capacity before real gathers");
        }
        checked(cudaSetDevice(destination_device), "enter the real drafter device context");
        Output reference{}, candidate{};
        // Descending live extents would leave earlier valid rows in the suffix.
        // Ascending2/4/8 preserves the initially poisoned inactive suffix.
        for (const int live : {2, 4, 8}) {
            stage(embedding, control, local, ids, live); read_output(local, reference);
            stage(embedding, local, local, ids, live); read_output(local, candidate);
            const size_t active = size_t(live) * hidden;
            require(std::equal(reference.begin(), reference.begin() + active, candidate.begin()),
                    "moving the unchanged embedding lookup changed active BF16 values");
            require(std::all_of(candidate.begin(), candidate.begin() + active, bf16_is_finite),
                    "actual active embeddings contain nonfinite data");
            require(std::none_of(candidate.begin() + active, candidate.end(), bf16_is_finite),
                    "inactive poison was unexpectedly replaced");
            if (live == 4)
                require(std::equal(captured.begin(), captured.end(), candidate.begin()),
                        "actual gathered embeddings differ from the retained caf83187 live-input reference");
            std::printf("{\"kind\":\"functional\",\"live_rows\":%d,\"active_values_equal\":true,"
                        "\"active_finite\":true,\"inactive_poison_preserved\":true,\"inactive_elements\":%zu}\n",
                        live, elements - active);
        }
        for (Endpoint* endpoint : {&control, &local}) {
            stage(embedding, *endpoint, local, ids, 4, true); read_output(local, candidate);
            require(candidate[0] == 0x7f80u && std::all_of(candidate.begin() + 1,
                    candidate.begin() + 4 * hidden, bf16_is_finite), "active nonfinite was hidden or spread");
        }
        std::puts("{\"kind\":\"functional\",\"active_nonfinite_preserved\":true,\"arms\":2}");
        sample("control_noise_a", embedding, control, local, ids);
        sample("control_noise_b", embedding, control, local, ids);
        sample("control_ab", embedding, control, local, ids);
        sample("local_ab", embedding, local, local, ids);
        sample("local_ba", embedding, local, local, ids);
        sample("control_ba", embedding, control, local, ids);
        require(embedding.identity() == identity, "borrowed target identity changed during component work");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "embedding staging component refused: %s\n", error.what());
        return 1;
    }
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    if (argc == 2 && std::strcmp(argv[1], "--help") == 0) {
        std::puts("Usage: dflash2_embedding_staging_micro --native MODEL_SHARD --embedding-sha256 HEX --captured-embeddings BF16_FILE\n"
                  "Fixed CUDA0->CUDA2 versus local CUDA2; actual anchor271/mask248077; live2/4/8 checks;\n"
                  "eight-row capacity in both arms; live4 timing,128 calls,control/control and AB/BA.\n"
                  "Requires the external queued owner, frozen identities and resource/thermal guards.");
        return 0;
    }
    const int result = run_component(argc, argv);
    // run_component's streams and buffers have already been destroyed successfully.
    if (result == 0)
        std::puts("{\"kind\":\"complete\",\"identity_unchanged\":true,\"endpoint_teardown_checked\":true,"
                  "\"production_path_changed\":false,\"scope\":\"completed embedding staging only; no service or quality claim\"}");
    return result;
}
