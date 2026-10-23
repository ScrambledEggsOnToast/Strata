// Explicit queued CUDA fixture, not CTest. Inject host-side API refusals only;
// never submit an invalid release or manufacture a device fault.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/dflash2_drafter.hpp"
#include "strata/core/dflash2_weights.hpp"
#include "strata/core/family_layout.hpp"
#include "strata/core/native_head.hpp"
#include <cuda_runtime.h>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

extern "C" {
cudaError_t __real_cudaMalloc(void**, size_t);
cudaError_t __real_cudaFree(void*);
cudaError_t __real_cudaHostAlloc(void**, size_t, unsigned);
cudaError_t __real_cudaFreeHost(void*);
cudaError_t __real_cudaStreamCreateWithFlags(cudaStream_t*, unsigned);
cudaError_t __real_cudaStreamDestroy(cudaStream_t);
cudaError_t __real_cudaStreamSynchronize(cudaStream_t);
cudaError_t __real_cudaSetDevice(int);
}
namespace {
namespace d = strata::core::dflash2;
enum class Kind { device, pinned, stream };
enum class Fault { selection, stream, transfer, late };
struct Resource { void* pointer = nullptr; Kind kind{}; int device = -1; bool baseline = false; };
std::array<Resource, 1024> resources{};
Fault fault{};
bool armed = false;
unsigned injections = 0, created = 0, released = 0;
[[noreturn]] void invalid(const char* message) {
    std::fprintf(stderr, "ownership probe prevents invalid CUDA operation: %s\n", message);
    std::_Exit(90);
}
void check(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
void cuda_check(cudaError_t status) { check(status == cudaSuccess, cudaGetErrorString(status)); }
int current() {
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) invalid("cannot establish current device");
    return device;
}
Resource& spare() {
    for (auto& resource : resources) if (!resource.pointer) return resource;
    invalid("bounded resource tracker exhausted before allocation");
}
void remember(Resource& slot, void* pointer, Kind kind, int device) noexcept {
    slot = {pointer, kind, device, false};
    if (armed) ++created;
}
Resource& owned(void* pointer, Kind kind) {
    for (auto& resource : resources) {
        if (resource.pointer != pointer || resource.kind != kind) continue;
        if (kind != Kind::pinned && resource.device != current()) invalid("wrong-device resource operation");
        return resource;
    }
    invalid("unknown or already released resource");
}
Resource& releasable(void* pointer, Kind kind) {
    auto& resource = owned(pointer, kind);
    if (armed && resource.baseline) invalid("constructor released a borrowed fixture resource");
    return resource;
}
void forgotten(Resource& resource, cudaError_t status) noexcept {
    if (status == cudaSuccess) {
        resource = {};
        if (armed) ++released;
    }
}
bool inject(Fault selected) {
    if (!armed || injections || fault != selected) return false;
    ++injections;
    return true;
}
void pricing() {
    constexpr size_t raw = d::block_size * d::hidden_size * sizeof(float);
    constexpr size_t bf16 = d::block_size * d::hidden_size * sizeof(uint16_t);
    constexpr size_t tokens = d::block_size * sizeof(int32_t);
    const auto remote = d::DflashNativeDrafter::planned_allocation(2, 0, 0, 2, 0);
    const auto local = d::DflashNativeDrafter::planned_allocation(2, 0, 2, 2, 0);
    const auto source = d::DflashNativeDrafter::planned_target_allocation(0, 2, 0, 2, 0);
    const auto moved = d::DflashNativeDrafter::planned_target_allocation(0, 2, 2, 2, 0);
    check(source.shared_device == tokens + raw + bf16 && moved.shared_device == 0,
          "source GPU must release the entire staging allocation");
    check(local.shared_device == remote.shared_device + tokens + raw && local.shared_host == remote.shared_host,
          "destination must charge tokens/raw and reuse its existing BF16 allocation");
    std::puts("embedding placement pricing passed: source -122912 bytes; destination +81952 bytes");
}
void run(const std::map<std::string, std::string>& args) {
    const auto& name = args.at("--case");
    const char* refusal = nullptr;
    bool local = false;
    if (name == "selection") { fault = Fault::selection; local = true; refusal = "dflash2 embedding execution device selection failed"; }
    else if (name == "stream") { fault = Fault::stream; refusal = "native embedding stream"; }
    else if (name == "transfer") { fault = Fault::transfer; refusal = "native embedding transfer rows"; }
    else if (name == "late") { fault = Fault::late; local = true; refusal = "native head float input"; }
    else throw std::runtime_error("unknown injected case");
    const char* flag = std::getenv("STRATA_DFLASH2_LOCAL_EMBEDDING");
    check(flag && std::strcmp(flag, local ? "1" : "0") == 0, "case needs its explicit process-start placement mode");
    const auto shards = strata::gguf_split_paths(args.at("--native"));
    check(shards.size() == 2, "fixture requires the verified two-shard daily target");
    strata::core::ModelGeometry geometry;
    const strata::GgufFile model(shards.front());
    std::string error;
    int64_t top_k = 0;
    const bool geometry_ok = strata::core::geometry_from_gguf(model, geometry, top_k, error);
    check(geometry_ok, error.c_str());
    error = strata::core::compiled_geometry_mismatch(geometry);
    check(error.empty(), error.c_str());
    strata::core::NativeEmbed embedding;
    cuda_check(__real_cudaSetDevice(0));
    if (!embedding.load(shards, geometry.n_embd, geometry.n_vocab, error, true)) throw std::runtime_error(error);
    check(embedding.mapped_host(), "actual host-mapped embedding required");
    strata::core::NativeHead head;
    struct HeadCleanup {
        strata::core::NativeHead& head;
        ~HeadCleanup() {
            if (__real_cudaSetDevice(2) != cudaSuccess) invalid("fixture head cleanup device selection");
            head.unload();
        }
    } head_cleanup{head};
    cuda_check(__real_cudaSetDevice(2));
    if (!head.load(shards, geometry.n_embd, geometry.n_vocab, error, true)) throw std::runtime_error(error);
    std::map<std::string, d::VerityPin> roots;
    if (!d::load_verity_roots(args.at("--verity-roots"), roots, error)) throw std::runtime_error(error);
    auto target = d::VerifiedDailyTarget::identify(args.at("--pack"), args.at("--tokenizer"),
        {shards[0], shards[1]}, geometry, embedding.identity(), head.identity(), std::chrono::seconds(120), roots);
    d::RuntimeWeights weights(args.at("--runtime"), args.at("--manifest-sha256"), target.binding().identity);
    cuda_check(__real_cudaSetDevice(0));
    unsigned baseline = 0;
    for (auto& resource : resources) if (resource.pointer) { resource.baseline = true; ++baseline; }
    std::string observed;
    {
        struct Disarm { ~Disarm() { armed = false; } } disarm;
        armed = true;
        try {
            d::DflashNativeDrafter drafter(weights, std::move(target), embedding, head, 2);
        } catch (const std::exception& failure) {
            observed = failure.what();
        }
    }
    check(injections == 1 && observed == refusal, "one intended injection and its exact constructor refusal required");
    unsigned live = 0;
    for (const auto& resource : resources) if (resource.pointer) {
        check(resource.baseline, "constructor leaked a newly owned resource");
        ++live;
    }
    check(live == baseline && created == released, "fixture baseline and exactly-once cleanup must survive refusal");
    if (fault == Fault::selection) check(created == 0, "selection refusal must precede drafter resource creation");
    std::printf("{\"case\":\"%s\",\"local\":%s,\"injections\":%u,\"refusal\":\"%s\",\"created\":%u,\"released\":%u,\"baseline_live\":%u}\n",
                name.c_str(), local ? "true" : "false", injections, observed.c_str(), created, released, baseline);
}
}
extern "C" cudaError_t __wrap_cudaSetDevice(int device) {
    if (device == 2 && inject(Fault::selection)) return cudaErrorInvalidDevice;
    return __real_cudaSetDevice(device);
}
extern "C" cudaError_t __wrap_cudaMalloc(void** out, size_t size) {
    const int device = current();
    if ((device == 0 && size == 40960 && inject(Fault::transfer)) ||
        (device == 2 && size == 71680 && inject(Fault::late))) return cudaErrorMemoryAllocation;
    auto& slot = spare();
    const auto result = __real_cudaMalloc(out, size);
    if (result == cudaSuccess) remember(slot, *out, Kind::device, device);
    return result;
}
extern "C" cudaError_t __wrap_cudaFree(void* pointer) {
    if (!pointer) return __real_cudaFree(pointer);
    auto& resource = releasable(pointer, Kind::device);
    const auto result = __real_cudaFree(pointer);
    forgotten(resource, result);
    return result;
}
extern "C" cudaError_t __wrap_cudaHostAlloc(void** out, size_t size, unsigned flags) {
    auto& slot = spare();
    const auto result = __real_cudaHostAlloc(out, size, flags);
    if (result == cudaSuccess) remember(slot, *out, Kind::pinned, -1);
    return result;
}
extern "C" cudaError_t __wrap_cudaFreeHost(void* pointer) {
    auto& resource = releasable(pointer, Kind::pinned);
    const auto result = __real_cudaFreeHost(pointer);
    forgotten(resource, result);
    return result;
}
extern "C" cudaError_t __wrap_cudaStreamCreateWithFlags(cudaStream_t* out, unsigned flags) {
    const int device = current();
    if (device == 0 && inject(Fault::stream)) return cudaErrorMemoryAllocation;
    auto& slot = spare();
    const auto result = __real_cudaStreamCreateWithFlags(out, flags);
    if (result == cudaSuccess) remember(slot, *out, Kind::stream, device);
    return result;
}
extern "C" cudaError_t __wrap_cudaStreamDestroy(cudaStream_t stream) {
    auto& resource = releasable(stream, Kind::stream);
    const auto result = __real_cudaStreamDestroy(stream);
    forgotten(resource, result);
    return result;
}
extern "C" cudaError_t __wrap_cudaStreamSynchronize(cudaStream_t stream) {
    if (stream) (void)owned(stream, Kind::stream);
    return __real_cudaStreamSynchronize(stream);
}
int main(int argc, char** argv) {
    std::set_terminate([] { std::fputs("ownership probe: unexpected fail-stop\n", stderr); std::_Exit(91); });
    try {
        if (argc == 2 && std::strcmp(argv[1], "--help") == 0) {
            std::puts("dflash2_embedding_owner_test --pricing | --case selection|stream|transfer|late --native SHARD --pack PACK --tokenizer DIR --runtime DIR --manifest-sha256 SHA256 --verity-roots FILE\nQueued CUDA execution only; process-start STRATA_DFLASH2_LOCAL_EMBEDDING must be 0 for stream/transfer, 1 for selection/late.");
            return 0;
        }
        if (argc == 2 && std::strcmp(argv[1], "--pricing") == 0) { pricing(); return 0; }
        check(argc == 15, "expected exactly seven option/value pairs; see --help");
        std::map<std::string, std::string> args;
        for (int i = 1; i < argc; i += 2) check(args.emplace(argv[i], argv[i + 1]).second, "duplicate option");
        for (const char* key : {"--case", "--native", "--pack", "--tokenizer", "--runtime", "--manifest-sha256", "--verity-roots"})
            check(args.contains(key), "missing or unknown option");
        run(args);
        for (const auto& resource : resources) check(!resource.pointer, "fixture resource cleanup failed");
        std::puts("{\"complete\":true,\"live_resources\":0,\"scope\":\"constructor refusal only; no proposal or service qualification\"}");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ownership probe failed: %s\n", error.what());
        return 1;
    }
}
