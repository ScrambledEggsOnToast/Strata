// The real SessionOwner/QSA init and teardown with a host-memory CUDA allocation
// adapter. No kernels, model, CUDA context or runtime-forwarding wrappers.
#include "strata/core/session.hpp"
#include "strata/kernels/mrope.hpp"
#include "strata/kernels/rope.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
int failures = 0, current_device = 0, fail_copy = -1;
uint64_t allocations = 0, host_allocations = 0, drains = 0;
std::unordered_map<void*, size_t> device_live, host_live;
void check(bool value, const char* name) {
    if (!value) { std::fprintf(stderr, "FAIL %s\n", name); ++failures; }
}
size_t live_bytes() {
    size_t bytes = 0;
    for (const auto& entry : device_live) bytes += entry.second;
    return bytes;
}
cudaError_t allocate(void** out, size_t bytes, std::unordered_map<void*, size_t>& live) {
    *out = std::malloc(bytes); // untouched session pages consume address space, not full-context RAM
    if (!*out) return cudaErrorMemoryAllocation;
    live.emplace(*out, bytes);
    return cudaSuccess;
}
}
extern "C" {
cudaError_t __wrap_cudaGetDevice(int* device) { *device = current_device; return cudaSuccess; }
cudaError_t __wrap_cudaSetDevice(int device) { current_device = device; return cudaSuccess; }
cudaError_t __wrap_cudaMalloc(void** out, size_t bytes) {
    ++allocations;
    return allocate(out, bytes, device_live);
}
cudaError_t __wrap_cudaFree(void* pointer) {
    check(!pointer || device_live.erase(pointer) == 1, "device allocation released exactly once");
    std::free(pointer);
    return cudaSuccess;
}
cudaError_t __wrap_cudaHostAlloc(void** out, size_t bytes, unsigned) {
    ++host_allocations;
    return allocate(out, bytes, host_live);
}
cudaError_t __wrap_cudaFreeHost(void* pointer) {
    check(!pointer || host_live.erase(pointer) == 1, "pinned allocation released exactly once");
    std::free(pointer);
    return cudaSuccess;
}
cudaError_t __wrap_cudaHostGetDevicePointer(void** out, void* pointer, unsigned) {
    *out = pointer; return cudaSuccess;
}
cudaError_t __wrap_cudaMemcpy(void* destination, const void* source, size_t bytes, cudaMemcpyKind) {
    if (fail_copy >= 0 && fail_copy-- == 0) return cudaErrorMemoryAllocation;
    std::memcpy(destination, source, bytes);
    return cudaSuccess;
}
cudaError_t __wrap_cudaDeviceSynchronize() { ++drains; return cudaSuccess; }
const char* __wrap_cudaGetErrorString(cudaError_t) { return "no-load allocation adapter error"; }
}

int main() {
    using namespace strata;
    core::ModelGeometry geometry;
    geometry.n_layers = 4; // one actual QSA primary; full-profile stage prices have their own fixture
    core::qsa_set_kv_resident(0);
    core::qsa_set_kv_int8(false);
    core::qsa_set_kv_q4(false);
    core::qsa_set_kv_hybrid(true);
    core::qsa_set_kv_elastic(false, 0);
    std::string error;
    const auto scaling = kernels::rope_scaling();
    core::SessionOwner generation;
    const bool shared = core::shared_rope_enabled();
    const int64_t capacity = shared ? 262144 : 65, cells = shared ? 4096 : 32;
    if (!generation.init(geometry, capacity, 10, error)) {
        std::fprintf(stderr, "generation no-load init: %s\n", error.c_str()); return 1;
    }
    const auto& primary = generation.state().qsa_states[generation.state().qsa_primary()];
    float* const cosine = primary.cos_tab;
    float* const sine = primary.sin_tab;
    const uint64_t table_bytes = shared ? core::shared_rope_device_bytes(geometry, capacity) : 0;
    check(primary.owns_rope != shared, "dedicated/shared table ownership remains distinct");
    check(live_bytes() == generation.memory().device + table_bytes,
          "only one generation arena and one separately admitted shared table");
    generation.bind_rope();
    check(kernels::rope_table_for(scaling).max_pos == capacity,
          "generation registration initially exposes its full capacity");
    const uint64_t generation_bytes = generation.memory().device;
    std::vector<float> expected((size_t)cells * 32 * 2);
    kernels::build_rope_table(64, scaling, (int)cells, expected.data(), expected.data() + cells * 32);
    for (int iteration = 0; iteration < 2; ++iteration) {
        const auto before = allocations;
        core::SessionAllocationBytes priced;
        check(core::session_allocation_bytes(geometry, cells, 10, priced), "short owner price available");
        core::SessionOwner short_state;
        if (!short_state.init(geometry, cells, 10, error)) {
            std::fprintf(stderr, "short no-load init: %s\n", error.c_str()); return 1;
        }
        const auto& qsa = short_state.state().qsa_states[short_state.state().qsa_primary()];
        check(allocations == before + 1 && short_state.memory().device == priced.device &&
              short_state.memory().device < generation_bytes && short_state.state().max_cells == cells && qsa.max_cells == cells,
              "judge initialization makes exactly one short arena, never a full-context arena or extra table");
        check(!shared || (!qsa.owns_rope && qsa.cos_tab == cosine && qsa.sin_tab == sine &&
              sine == cosine + capacity * 32), "short owner borrows both original capacity-sized planes");
        check(std::memcmp(qsa.cos_tab, expected.data(), (size_t)cells * 32 * sizeof(float)) == 0 &&
              std::memcmp(qsa.sin_tab, expected.data() + cells * 32, (size_t)cells * 32 * sizeof(float)) == 0,
              "actual initialized short owner reads the CPU-built first-N rows exactly");
        check(kernels::rope_table_for(scaling).max_pos == cells,
              "short registration exposes only the admitted short capacity");
        short_state.reset();
        check(live_bytes() == generation_bytes + table_bytes && generation.state().max_cells == capacity &&
              generation.state().qsa_states[generation.state().qsa_primary()].cos_tab == cosine,
              "short SessionOwner reset preserves generation state and allocation");
        if (shared) check(kernels::rope_table_for(scaling).cos == cosine,
                          "short teardown never releases the process registration");
        else check(kernels::rope_table_for(scaling).cos == nullptr,
                   "dedicated short teardown still releases only its own registration");
        generation.bind_rope();
        check(kernels::rope_table_for(scaling).cos == cosine && kernels::rope_table_for(scaling).sin == sine &&
              kernels::rope_table_for(scaling).max_pos == capacity,
              "generation binding restores its unchanged long capacity and pointers");
    }
    if (shared) {
        auto refused = [&](const core::ModelGeometry& shape, int64_t rows, const char* expected_reason) {
            const auto before = allocations, host_before = host_allocations;
            core::SessionOwner invalid;
            check(!invalid.init(shape, rows, 10, error) && error == expected_reason,
                  "incompatible SessionOwner returns the precise borrow refusal");
            check(allocations == before && host_allocations == host_before && invalid.state().allocation_device == -1 &&
                  invalid.state().qsa_states == nullptr && invalid.memory().device == 0,
                  "incompatibility refuses before arena allocation or session mutation");
        };
        refused(geometry, capacity + 1, "shared RoPE: requested rows exceed immutable table capacity");
        auto changed_geometry = geometry; changed_geometry.n_rot = 32;
        refused(changed_geometry, cells, "shared RoPE: rotary width differs");
        auto changed_scaling = scaling; changed_scaling.freq_base += 1;
        kernels::rope_scaling_set(changed_scaling); // deliberate invalid test input; no captured graph or inference
        refused(geometry, cells, "shared RoPE: frequency base differs");
        kernels::rope_scaling_set(scaling);
        const char* why = core::shared_rope_session_invalid(geometry, cells, 1, true);
        check(why && std::strcmp(why, "shared RoPE: no published process-owned table on this device") == 0,
              "judge preflight requires an existing table on each actual generation device");
        {
            core::SessionOwner failed;
            fail_copy = 0; // reused table needs no upload; fail the short page-table initialization
            check(!failed.init(geometry, cells, 10, error), "partial short-session initialization failure propagated");
            fail_copy = -1;
            check(live_bytes() == generation_bytes + table_bytes && host_live.size() == 2,
                  "failed short owner releases only its arena and pinned controls");
        }
        generation.bind_rope();
        check(kernels::rope_table_for(scaling).cos == cosine && kernels::rope_table_for(scaling).max_pos == capacity,
              "generation owner survives incompatible and failed short creations");
    }
    generation.reset();
    check(host_live.empty(), "every session pinned allocation retired");
    check(live_bytes() == table_bytes && device_live.size() == (shared ? 1u : 0u),
          "shared table remains process-owned; dedicated arenas fully retire");
    if (shared) {
        core::SessionOwner later;
        check(later.init(geometry, cells, 10, error), "process table remains borrowable after every original owner retires");
        later.reset();
        check(live_bytes() == table_bytes && device_live.size() == 1 && host_live.empty(),
              "later short teardown neither replaces nor frees the retained table");
    }
    check(drains > 0, "real session init/teardown traverses the drain calls");
    std::printf("SESSION_OWNER_NO_LOAD shared=%d generation_cells=%lld short_cells=%lld retained_table_bytes=%llu failures=%d\n",
                shared, (long long)capacity, (long long)cells, (unsigned long long)table_bytes, failures);
    return failures ? 1 : 0;
}
