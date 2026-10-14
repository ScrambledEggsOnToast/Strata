#pragma once

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

namespace strata::prefill {

// Persistent, owner-local storage. Events are reused only after the existing
// chunk join has completed and copied their durations into plain host records.
struct NativeProjectionProfile {
    static constexpr size_t event_capacity = 512;
    static constexpr size_t record_capacity = 8192;
    struct Record {
        uint64_t request = 0;
        int64_t position = 0, rows = 0, columns = 0, inner = 0, row_start = 0, row_count = 0;
        int64_t input_stride = 0, output_stride = 0;
        int64_t dequant_begin_ns = 0, dequant_end_ns = 0, gemm_begin_ns = 0, gemm_end_ns = 0;
        float beta = 0, dequant_ms = 0, gemm_ms = 0;
        int type = 0;
        char name[64] = {};
    };
    struct Pending {
        cudaEvent_t events[3] = {};
        Record record;
        bool valid = true;
    };
    std::array<Pending, event_capacity> pending;
    std::array<Record, record_capacity> records;
    size_t used = 0, retained = 0, high_water = 0;
    uint64_t dropped = 0, invalid = 0, request = 0;
    int64_t position = 0;
    int device = -1, sm = 0;
    const char* label = nullptr;

    static int64_t now_ns() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    bool init(std::string& error) {
        int major = 0, minor = 0;
        if (cudaGetDevice(&device) != cudaSuccess ||
            cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess ||
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess) {
            error = "native projection profile: device identity unavailable"; return false;
        }
        sm = major * 10 + minor;
        for (auto& slot : pending) for (auto& event : slot.events)
            if (cudaEventCreate(&event) != cudaSuccess) {
                error = "native projection profile: bounded event allocation failed"; return false;
            }
        std::fprintf(stderr, "NATIVE_PROFILE_OWNER version=1 owner=%p device=%d sm=%d events=%zu records=%zu host_bytes=%zu "
                             "scope=native_f16 opaque=existing_prefill_and_host_allowances clock=device_event_duration\n",
                     (void*)this, device, sm, event_capacity * 3, record_capacity, sizeof(*this));
        return true;
    }
    ~NativeProjectionProfile() {
        // Event destruction is deferred by CUDA for any outstanding record. The
        // Prefill owner retains/drains execution before releasing this object.
        for (auto& slot : pending) for (auto event : slot.events) if (event) cudaEventDestroy(event);
    }
    Pending* begin(int type, int64_t t, int64_t n, int64_t k, int64_t r0, int64_t rn,
                   int64_t ldx, int64_t ldy, float beta, cudaStream_t stream) {
        if (used == pending.size() || retained + used == records.size()) { ++dropped; return nullptr; }
        auto& slot = pending[used++];
        high_water = std::max(high_water, used);
        slot.record = {};
        auto& r = slot.record;
        r.request = request; r.position = position; r.type = type;
        r.rows = t; r.columns = n; r.inner = k; r.row_start = r0; r.row_count = rn;
        r.input_stride = ldx > 0 ? ldx : k; r.output_stride = ldy > 0 ? ldy : n; r.beta = beta;
        const char* name = label ? label : "unlabelled_native";
        slot.valid = std::strlen(name) < sizeof(r.name);
        if (slot.valid) std::strcpy(r.name, name);
        slot.valid = cudaEventRecord(slot.events[0], stream) == cudaSuccess && slot.valid;
        return &slot;
    }
    bool collect(std::string& error) {
        // Called only at the execution owner's existing successful stream join.
        // No synchronization, allocation or output is introduced here.
        bool okay = true;
        for (size_t i = 0; i < used; ++i) {
            auto& slot = pending[i];
            auto& r = slot.record;
            if (!slot.valid || cudaEventQuery(slot.events[2]) != cudaSuccess ||
                cudaEventElapsedTime(&r.dequant_ms, slot.events[0], slot.events[1]) != cudaSuccess ||
                cudaEventElapsedTime(&r.gemm_ms, slot.events[1], slot.events[2]) != cudaSuccess ||
                !std::isfinite(r.dequant_ms) || !std::isfinite(r.gemm_ms) ||
                r.dequant_ms < 0 || r.gemm_ms < 0 || retained == records.size()) {
                ++invalid; okay = false;
            } else records[retained++] = r;
        }
        used = 0;
        if (!okay) error = "native projection profile: incomplete or invalid drained interval";
        return okay;
    }
    void flush(bool complete) {
        const int64_t begin = now_ns();
        for (size_t i = 0; i < retained; ++i) {
            const auto& r = records[i];
            std::fprintf(stderr, "NATIVE_PROJECTION version=1 owner=%p device=%d sm=%d request=%llu pos=%lld name=%s type=%d "
                                 "T=%lld N=%lld K=%lld r0=%lld n=%lld ldx=%lld ldy=%lld beta=%.9g "
                                 "dequant_envelope_ms=%.9g gemm_envelope_ms=%.9g "
                                 "dequant_begin_ns=%lld dequant_end_ns=%lld gemm_begin_ns=%lld gemm_end_ns=%lld\n",
                         (void*)this, device, sm, (unsigned long long)r.request, (long long)r.position, r.name, r.type,
                         (long long)r.rows, (long long)r.columns, (long long)r.inner,
                         (long long)r.row_start, (long long)r.row_count,
                         (long long)r.input_stride, (long long)r.output_stride, (double)r.beta,
                         (double)r.dequant_ms, (double)r.gemm_ms, (long long)r.dequant_begin_ns,
                         (long long)r.dequant_end_ns, (long long)r.gemm_begin_ns, (long long)r.gemm_end_ns);
        }
        std::fprintf(stderr, "NATIVE_PROFILE_END version=1 owner=%p device=%d records=%zu event_high_water=%zu "
                             "dropped=%llu invalid=%llu unclosed=%zu complete=%d write_us=%lld\n",
                     (void*)this, device, retained, high_water, (unsigned long long)dropped, (unsigned long long)invalid,
                     used, complete && !dropped && !invalid && !used ? 1 : 0,
                     (long long)((now_ns() - begin) / 1000));
        retained = 0; high_water = 0; dropped = invalid = 0;
        // Unclosed events are never recycled after a failed operation.
    }
};

} // namespace strata::prefill
