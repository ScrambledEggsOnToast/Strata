// Staged-pointer handout, DMA enqueue and fence retirement are distinct ownership boundaries.
// Wrap CUDA as in split_lend_complement_test so every interleaving is deterministic on a CPU-only host.
#include "strata/core/expert_source.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "gguf_fixture.hpp"

#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace {

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

struct Stream {
    void* dst = nullptr;
    const void* src = nullptr;
    size_t bytes = 0;
    bool pending = false;
};
struct Event {
    Stream* stream = nullptr;
    bool used = false;
};
std::array<Stream, 2> streams;
std::array<Event, 8> events;
bool fail_host_alloc = false, fail_create = false, fail_record = false;
bool fail_enqueue = false, fail_stream_sync = false, fail_query = false, fail_event_sync = false;
bool fail_device_sync = false;
int records = 0, event_waits = 0, stream_waits = 0, device_waits = 0;

void drain(Stream& stream) {
    if (stream.pending) std::memcpy(stream.dst, stream.src, stream.bytes);
    stream.pending = false;
}

cudaStream_t stream_at(size_t i) {
    return i == 0 ? nullptr : reinterpret_cast<cudaStream_t>(&streams[i]);
}

Stream& stream_of(cudaStream_t handle) {
    return handle ? *reinterpret_cast<Stream*>(handle) : streams[0];
}

void reset_cuda() {
    for (const Stream& stream : streams) require(!stream.pending, "a test left DMA queued");
    for (const Event& event : events) require(!event.used, "a test leaked an event");
    streams = {};
    fail_host_alloc = fail_create = fail_record = false;
    fail_enqueue = fail_stream_sync = fail_query = fail_event_sync = false;
    fail_device_sync = false;
    records = event_waits = stream_waits = device_waits = 0;
}

}  // namespace

extern "C" cudaError_t __wrap_cudaHostAlloc(void** p, size_t n, unsigned) {
    *p = fail_host_alloc ? nullptr : std::malloc(n);
    return *p ? cudaSuccess : cudaErrorMemoryAllocation;
}
extern "C" cudaError_t __wrap_cudaFreeHost(void* p) { std::free(p); return cudaSuccess; }
extern "C" cudaError_t __wrap_cudaGetLastError(void) { return cudaSuccess; }
extern "C" cudaError_t __wrap_cudaMemcpyAsync(void* dst, const void* src, size_t n, cudaMemcpyKind,
                                              cudaStream_t handle) {
    if (fail_enqueue) return cudaErrorUnknown;
    Stream& stream = stream_of(handle);
    require(!stream.pending, "the test stream already has a queued copy");
    stream = {dst, src, n, true};
    return cudaSuccess;
}
extern "C" cudaError_t __wrap_cudaDeviceSynchronize(void) {
    ++device_waits;
    if (fail_device_sync) return cudaErrorNoDevice;
    for (Stream& stream : streams) drain(stream);
    return cudaSuccess;
}
extern "C" cudaError_t __wrap_cudaStreamSynchronize(cudaStream_t handle) {
    ++stream_waits;
    if (fail_stream_sync) return cudaErrorUnknown;
    drain(stream_of(handle));
    return cudaSuccess;
}
extern "C" cudaError_t __wrap_cudaEventCreateWithFlags(cudaEvent_t* handle, unsigned) {
    if (fail_create) return cudaErrorUnknown;
    for (Event& event : events) {
        if (event.used) continue;
        event = {nullptr, true};
        *handle = reinterpret_cast<cudaEvent_t>(&event);
        return cudaSuccess;
    }
    return cudaErrorMemoryAllocation;
}
extern "C" cudaError_t __wrap_cudaEventRecord(cudaEvent_t handle, cudaStream_t stream) {
    if (fail_record) return cudaErrorUnknown;
    ++records;
    reinterpret_cast<Event*>(handle)->stream = &stream_of(stream);
    return cudaSuccess;
}
extern "C" cudaError_t __wrap_cudaEventQuery(cudaEvent_t handle) {
    if (fail_query) return cudaErrorUnknown;
    const Event& event = *reinterpret_cast<Event*>(handle);
    return event.stream && event.stream->pending ? cudaErrorNotReady : cudaSuccess;
}
extern "C" cudaError_t __wrap_cudaEventSynchronize(cudaEvent_t handle) {
    ++event_waits;
    if (fail_event_sync) return cudaErrorUnknown;
    Event& event = *reinterpret_cast<Event*>(handle);
    if (event.stream) drain(*event.stream);
    return cudaSuccess;
}
extern "C" cudaError_t __wrap_cudaEventDestroy(cudaEvent_t handle) {
    *reinterpret_cast<Event*>(handle) = {};
    return cudaSuccess;
}

namespace fs = std::filesystem;

namespace {

struct Fixture {
    fs::path path;
    strata::core::FileExpertSource source;
    size_t bytes = 0;

    Fixture() {
        using namespace strata::kernels::cpu;
        path = fs::temp_directory_path() / ("strata-stage-lease-" +
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(path);
        constexpr uint32_t q4_k = 12, q5_1 = 7;
        NativeFmt fmt;
        std::string err;
        require(native_fmt(q4_k, q5_1, H, FF, fmt, err), err);
        bytes = (size_t) fmt.bytes;
        std::vector<fixture::Tensor> tensors;
        for (int layer = 0; layer < 2; ++layer) {
            const std::string prefix = "blk." + std::to_string(layer) + ".ffn_";
            tensors.push_back({prefix + "gate_exps.weight", {H, FF, 2}, q4_k, (uint8_t) (11 + 3 * layer)});
            tensors.push_back({prefix + "up_exps.weight", {H, FF, 2}, q4_k, (uint8_t) (12 + 3 * layer)});
            tensors.push_back({prefix + "down_exps.weight", {FF, H, 2}, q5_1, (uint8_t) (13 + 3 * layer)});
        }
        const fs::path shard = path / "experts.gguf";
        const auto written = fixture::write(shard, {}, tensors);
        {
            std::ofstream manifest(path / "native_experts.txt");
            manifest << "# strata native experts v2: staged reader fixture\n";
            for (int layer = 0; layer < 2; ++layer) {
                manifest << layer << ' ' << q4_k << ' ' << q5_1 << ' ' << layer * 2 * bytes << ' ' << bytes;
                for (int role = 0; role < 3; ++role)
                    manifest << ' ' << written.data_start + written.offsets[(size_t) (3 * layer + role)];
                manifest << '\n';
            }
            require((bool) manifest, "could not write staged reader manifest");
        }
        require(expert_layout_load(path.string(), 2, 2, err), err);
        source.set_gguf(shard.string());
        require(source.open(path.string(), 2, 2, err), err);
    }

    ~Fixture() {
        source.close();
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }

    std::vector<uint8_t> truth(int64_t layer, int64_t expert) {
        std::vector<uint8_t> result(bytes);
        require(source.copy_blob(layer, expert, result.data()), "could not copy fixture truth");
        return result;
    }

    void age() {
        // Four alternating boundaries age every previously handed-out pointer past kStageAge.
        for (int layer : {0, 1, 0, 1}) source.begin_layer(layer, nullptr, 0);
    }
};

void queue_copy(const uint8_t* ptr, std::vector<uint8_t>& copy, size_t stream = 0) {
    require(cudaMemcpyAsync(copy.data(), ptr, copy.size(), cudaMemcpyHostToDevice, stream_at(stream)) == cudaSuccess,
            "could not queue fixture DMA");
}

void test_pageable_cpu_close_without_device() {
    reset_cuda();
    fail_host_alloc = true;  // a device-free host falls back to ordinary pageable storage
    Fixture rig;
    const auto original = rig.truth(0, 0);
    const uint8_t* ptr = rig.source.blob(0, 0);
    require(ptr && std::memcmp(ptr, original.data(), rig.bytes) == 0, "CPU blob has wrong bytes");
    fail_device_sync = true;  // CUDA initialization/drain would fail on this host
    rig.source.close();
    require(device_waits == 0, "closing a CPU-only blob attempted a CUDA drain");
}

void test_close_drains_live_fence() {
    reset_cuda();
    std::vector<uint8_t> copied;
    Fixture rig;
    const auto original = rig.truth(0, 0);
    copied.resize(rig.bytes);
    const uint8_t* ptr = rig.source.blob_async(0, 0);
    require(ptr != nullptr, "async reader failed before close");
    queue_copy(ptr, copied);
    rig.source.note_async_read(ptr, stream_at(0));
    rig.source.close();
    require(device_waits == 1 && copied == original && !streams[0].pending,
            "teardown did not drain a live completion fence");
}

void test_handout_to_enqueue(bool pageable) {
    reset_cuda();
    fail_host_alloc = pageable;
    std::vector<uint8_t> copied;   // outlives the source's failure-path teardown drain
    Fixture rig;
    const auto original = rig.truth(0, 0);
    copied.resize(rig.bytes);
    const uint8_t* ptr = rig.source.blob_async(0, 0);
    require(ptr && std::memcmp(ptr, original.data(), rig.bytes) == 0, "async handout has wrong bytes");
    require(!rig.source.blob_async(0, 0), "two unarmed readers acquired the same stage buffer");
    rig.age();
    const uint8_t* other = rig.source.blob(1, 1);
    require(other && other != ptr, "an outstanding reader's stage buffer was repurposed before DMA enqueue");
    require(std::memcmp(ptr, original.data(), rig.bytes) == 0, "handout bytes changed before DMA enqueue");
    // The elastic-KV refill's raw copy uses the legacy stream too; the source claim spans its pre-enqueue gap.
    if (pageable) std::memcpy(copied.data(), ptr, rig.bytes);  // the driver's synchronous pageable staging
    else queue_copy(ptr, copied);
    rig.source.note_async_read(ptr, stream_at(0));
    // An unpaired notification must not replace the first reader's fence with another stream's event.
    rig.source.note_async_read(ptr, stream_at(1));
    require(records == (pageable ? 0 : 1), "an unclaimed reader replaced a live DMA fence");
    rig.age();
    const uint8_t* reused = rig.source.blob(1, 0);
    require(reused == ptr, "a drained stage buffer did not become reusable");
    require(copied == original, "refill overwrote bytes before the queued DMA completed");
    require(event_waits == (pageable ? 0 : 1), "refill did not retire the outstanding DMA fence");
    rig.source.close();
    require(device_waits == 0, "teardown drained an already retired fence");
}

void test_next_reader_retires_prior_event() {
    reset_cuda();
    std::vector<uint8_t> first, second;
    Fixture rig;
    const auto original = rig.truth(0, 0);
    first.resize(rig.bytes);
    second.resize(rig.bytes);
    const uint8_t* ptr = rig.source.blob_async(0, 0);
    require(ptr != nullptr, "first async reader failed");
    queue_copy(ptr, first);
    rig.source.note_async_read(ptr, stream_at(0));
    require(rig.source.blob_async(0, 0) == ptr, "next reader could not acquire the unchanged blob");
    require(event_waits == 1 && first == original && !streams[0].pending,
            "next reader acquired a buffer before the prior stream's fence retired");
    require(!rig.source.blob_async(0, 0), "next reader's outstanding claim was overwritten");
    queue_copy(ptr, second, 1);
    rig.source.note_async_read(ptr, stream_at(1));
    require(records == 2, "the second reader did not arm its own fence");
    rig.age();
    require(rig.source.blob(1, 1) == ptr && second == original && event_waits == 2,
            "a later stream's DMA fence did not protect refill");
    rig.source.close();
}

void test_enqueue_failure_retains_claim() {
    reset_cuda();
    std::vector<uint8_t> copied;
    Fixture rig;
    copied.resize(rig.bytes);
    const auto original = rig.truth(0, 0);
    const uint8_t* ptr = rig.source.blob_async(0, 0);
    require(ptr != nullptr, "async reader failed before enqueue fault injection");
    fail_enqueue = true;
    require(cudaMemcpyAsync(copied.data(), ptr, copied.size(), cudaMemcpyHostToDevice, nullptr) != cudaSuccess,
            "enqueue fault did not reach the legacy-stream refill copy");
    // Just as in the refill caller, failed enqueue must not be followed by note_async_read.
    require(!rig.source.blob_async(0, 0), "failed enqueue released its outstanding reader claim");
    rig.age();
    const uint8_t* next = rig.source.blob(1, 1);
    require(next && next != ptr && std::memcmp(ptr, original.data(), rig.bytes) == 0,
            "a failed enqueue's stage buffer was refilled without a proven drain");
    require(records == 0, "a failed enqueue armed a completion fence");
    fail_enqueue = false;
    rig.source.close();
    require(device_waits == 1, "teardown did not drain an outstanding async reader claim");
}

void test_arm_failure(bool creating, bool failed_drain) {
    reset_cuda();
    std::vector<uint8_t> copied;
    Fixture rig;
    const auto original = rig.truth(0, 0);
    copied.resize(rig.bytes);
    const uint8_t* ptr = rig.source.blob_async(0, 0);
    require(ptr != nullptr, "async reader failed before fault injection");
    queue_copy(ptr, copied);
    fail_create = creating;
    fail_record = !creating;
    fail_stream_sync = failed_drain;
    rig.source.note_async_read(ptr, stream_at(0));
    require(stream_waits == 1, "failed event arming did not attempt to drain its own stream");
    rig.age();
    const uint8_t* next = rig.source.blob(1, 1);
    if (failed_drain) {
        require(next && next != ptr && !rig.source.blob_async(0, 0),
                "an event/stream failure released the outstanding reader claim");
        require(streams[0].pending && std::memcmp(ptr, original.data(), rig.bytes) == 0,
                "failed DMA proof permitted the source bytes to change");
    } else {
        require(next == ptr && copied == original, "a successful fallback drain did not release the claim safely");
    }
    fail_create = fail_record = fail_stream_sync = false;
    rig.source.close();
    require(copied == original, "teardown failed to drain the retained DMA");
}

void test_retirement_failure(bool query) {
    reset_cuda();
    std::vector<uint8_t> copied;
    Fixture rig;
    const auto original = rig.truth(0, 0);
    copied.resize(rig.bytes);
    const uint8_t* ptr = rig.source.blob_async(0, 0);
    require(ptr != nullptr, "async reader failed before retirement fault injection");
    queue_copy(ptr, copied);
    rig.source.note_async_read(ptr, stream_at(0));
    fail_query = query;
    fail_event_sync = !query;
    require(!rig.source.blob_async(0, 0), "failed event retirement allowed another reader");
    rig.age();
    require(!rig.source.blob(1, 1), "failed event retirement allowed a stage refill");
    require(streams[0].pending && std::memcmp(ptr, original.data(), rig.bytes) == 0,
            "failed event retirement discarded the live DMA proof");
    fail_query = fail_event_sync = false;
    require(rig.source.blob(1, 1) == ptr && copied == original,
            "a retained fence could not be retired after the fault cleared");
    rig.source.close();
}

}  // namespace

int main() {
    try {
        setenv("STRATA_STAGE_PIN", "1", 1);
        test_pageable_cpu_close_without_device();
        test_close_drains_live_fence();
        test_handout_to_enqueue(false);
        test_handout_to_enqueue(true);
        test_next_reader_retires_prior_event();
        test_enqueue_failure_retains_claim();
        for (bool creating : {false, true})
            for (bool failed_drain : {false, true}) test_arm_failure(creating, failed_drain);
        test_retirement_failure(false);
        test_retirement_failure(true);
        std::cout << "expert_stage_lease_test: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "expert_stage_lease_test: " << error.what() << '\n';
        return 1;
    }
}
