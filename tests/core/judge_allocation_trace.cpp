// Link-wrap explicit engine CUDA allocations, not shared-library/driver internals.
// Driver commitments are independently sampled by SerialJudgeLane and the launcher.
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <malloc.h>

extern "C" {
cudaError_t __real_cudaMalloc(void**, size_t);
cudaError_t __real_cudaFree(void*);
cudaError_t __real_cudaHostAlloc(void**, size_t, unsigned);
cudaError_t __real_cudaFreeHost(void*);
cudaError_t __real_cudaHostRegister(void*, size_t, unsigned);
cudaError_t __real_cudaHostUnregister(void*);
}
namespace {
struct Count { uint64_t live = 0, peak = 0, allocations = 0; };
struct Allocation { uint64_t bytes; int device, kind; };
struct Trace {
    std::mutex mutex;
    std::map<void*, Allocation> allocations;
    std::array<Count, 16> device;
    Count pinned, registered;
};
Trace& trace() { static Trace* value = new Trace; return *value; }
Count& count(Trace& t, const Allocation& a) {
    if (a.kind == 1) return t.pinned;
    if (a.kind == 2) return t.registered;
    if (a.device < 0 || a.device >= (int)t.device.size()) std::abort();
    return t.device[(size_t)a.device];
}
void add(void* pointer, size_t bytes, int kind) {
    if (!pointer || !bytes) return;
    int device = -1;
    if (kind == 0 && cudaGetDevice(&device) != cudaSuccess) std::abort();
    auto& t = trace();
    std::lock_guard<std::mutex> lock(t.mutex);
    if (!t.allocations.emplace(pointer, Allocation{bytes, device, kind}).second) std::abort();
    auto& c = count(t, t.allocations.find(pointer)->second);
    c.live += bytes; c.peak = std::max(c.live, c.peak); ++c.allocations;
}
void remove(void* pointer) {
    if (!pointer) return;
    auto& t = trace();
    std::lock_guard<std::mutex> lock(t.mutex);
    const auto found = t.allocations.find(pointer);
    if (found == t.allocations.end()) return; // an allocation internal to an unwrapped shared library
    auto& c = count(t, found->second);
    c.live -= found->second.bytes;
    t.allocations.erase(found);
}
uint64_t proc_bytes(const char* field) {
    FILE* input = std::fopen("/proc/self/status", "r");
    if (!input) std::abort();
    char name[128], line[512]; unsigned long long kb = 0;
    uint64_t bytes = 0;
    while (std::fgets(line, sizeof(line), input))
        if (std::sscanf(line, "%127s %llu kB", name, &kb) == 2 && std::strcmp(name, field) == 0) { bytes = kb * 1024; break; }
    std::fclose(input);
    return bytes;
}
}
extern "C" cudaError_t __wrap_cudaMalloc(void** pointer, size_t bytes) {
    const auto status = __real_cudaMalloc(pointer, bytes);
    if (status == cudaSuccess) add(*pointer, bytes, 0);
    return status;
}
extern "C" cudaError_t __wrap_cudaFree(void* pointer) {
    const auto status = __real_cudaFree(pointer);
    if (status == cudaSuccess) remove(pointer);
    return status;
}
extern "C" cudaError_t __wrap_cudaHostAlloc(void** pointer, size_t bytes, unsigned flags) {
    const auto status = __real_cudaHostAlloc(pointer, bytes, flags);
    if (status == cudaSuccess) add(*pointer, bytes, 1);
    return status;
}
extern "C" cudaError_t __wrap_cudaFreeHost(void* pointer) {
    const auto status = __real_cudaFreeHost(pointer);
    if (status == cudaSuccess) remove(pointer);
    return status;
}
extern "C" cudaError_t __wrap_cudaHostRegister(void* pointer, size_t bytes, unsigned flags) {
    const auto status = __real_cudaHostRegister(pointer, bytes, flags);
    if (status == cudaSuccess) add(pointer, bytes, 2);
    return status;
}
extern "C" cudaError_t __wrap_cudaHostUnregister(void* pointer) {
    const auto status = __real_cudaHostUnregister(pointer);
    if (status == cudaSuccess) remove(pointer);
    return status;
}
extern "C" void strata_judge_trace_phase(const char* phase) {
    auto& t = trace();
    std::lock_guard<std::mutex> lock(t.mutex);
    for (size_t d = 0; d < t.device.size(); ++d) if (t.device[d].allocations)
        std::fprintf(stderr, "JUDGE_ALLOC phase=%s device=%zu explicit_live_bytes=%llu explicit_high_water_bytes=%llu allocations=%llu\n",
                     phase, d, (unsigned long long)t.device[d].live, (unsigned long long)t.device[d].peak,
                     (unsigned long long)t.device[d].allocations);
    const auto heap = mallinfo2();
    std::fprintf(stderr, "JUDGE_HOST phase=%s pinned_live_bytes=%llu pinned_high_water_bytes=%llu registered_live_bytes=%llu "
                         "registered_high_water_bytes=%llu heap_live_bytes=%llu rss_bytes=%llu rss_high_water_bytes=%llu\n", phase,
                 (unsigned long long)t.pinned.live, (unsigned long long)t.pinned.peak,
                 (unsigned long long)t.registered.live, (unsigned long long)t.registered.peak,
                 (unsigned long long)(heap.uordblks + heap.hblkhd), (unsigned long long)proc_bytes("VmRSS:"),
                 (unsigned long long)proc_bytes("VmHWM:"));
}
