#pragma once

#include "strata/core/on_device.hpp"
#include "strata/core/verify_ownership.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include <cuda_runtime.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>

namespace strata::core {

// One serialized operation over shared MTP/prefill scratch. Owner IDs are uploaded
// at request binding, independently of mapped per-operation staging. Captured
// checks bake the same resident slot as the graph's state pointers. No allocations
// on the request path; owners, epoch and error word survive every graph in a lease.

class OperationCanary {
public:
    OperationCanary() = default;
    OperationCanary(const OperationCanary&) = delete;
    OperationCanary& operator=(const OperationCanary&) = delete;
    ~OperationCanary() {
        if (device_ < 0) return;
        const OnDevice on(device_);
        if (cudaDeviceSynchronize() != cudaSuccess) std::terminate();
        if (scratch_ && cudaFree(scratch_) != cudaSuccess) std::terminate();
        if (host_ && cudaFreeHost(host_) != cudaSuccess) std::terminate();
    }
    bool init(std::string& err) {
        if (scratch_) { err = "operation canary already initialized"; return false; }
        if (cudaGetDevice(&device_) != cudaSuccess ||
            cudaHostAlloc((void**)&host_, kOperationCanaryHostBytes, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess ||
            cudaHostGetDevicePointer((void**)&mapped_, host_, 0) != cudaSuccess ||
            cudaMalloc((void**)&scratch_, kOperationCanaryDeviceBytes) != cudaSuccess ||
            cudaMemset(scratch_, 0, kOperationCanaryDeviceBytes) != cudaSuccess) {
            err = "operation canary allocation failed"; return false;
        }
        *host_ = {};
        *output(host_) = {};
        *input_checks(host_) = 0;
        return true;
    }
    bool bind(int slot, uint64_t request, std::string& err) {
        if (!scratch_) return true; // unloaded optional MTP/prefill
        if (slot < 0 || slot > 8) { err = "operation owner slot out of range"; return false; }
        const OnDevice on(device_);
        const VerifyCanaryOwner owner{request, 0, 0};
        if (cudaMemcpy(owners() + slot, &owner, sizeof owner, cudaMemcpyHostToDevice) != cudaSuccess) {
            err = "operation owner upload failed"; return false;
        }
        ids_[slot] = request;
        return true;
    }
    void begin(int slot, cudaStream_t stream, int32_t token = 0, int32_t position = 0) {
        if (!scratch_) return;
        expected_ = {ids_[slot], ++epoch_, slot, token, position, 0};
        *host_ = expected_;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        kernels::verify_canary_begin(mapped_, owners() + slot, slot, &mapped_->token, &mapped_->position,
                                     nullptr, scratch_, stream);
        cudaMemsetAsync(&inputs()->checks, 0, sizeof(uint32_t), stream);
    }
    // Capturable; a wrong graph/slot sets a sticky error rather than replacing the
    // operation's independently staged identity. Also checks pinned staging reuse.
    void check(int slot, cudaStream_t stream, const int32_t* token = nullptr, const int32_t* position = nullptr) {
        if (!scratch_) return;
        kernels::verify_canary_finish(mapped_, owners() + slot, slot,
                                      token ? token : &mapped_->token, position ? position : &mapped_->position,
                                      nullptr, scratch_, nullptr, output(mapped_), 1, stream);
        cudaMemcpyAsync(input_checks(host_), &inputs()->checks, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
    }
    void stage_inputs(const int32_t* tokens, int rows, int32_t position, int accepted, cudaStream_t stream) {
        if (!scratch_) return;
        OperationInputs value;
        value.request_id = expected_.request_id; value.epoch = expected_.epoch;
        value.position = position; value.accepted = accepted;
        for (int t = 0; t < rows; ++t) value.tokens[t] = tokens[t];
        kernels::operation_inputs_stage(value, inputs(), stream);
    }
    void check_inputs(const int32_t* tokens, const int32_t* steps, const int32_t* positions,
                      int rows, int heads, cudaStream_t stream, int draft_step = -1, const int32_t* previous = nullptr) {
        if (!scratch_) return;
        kernels::operation_inputs_check(inputs(), scratch_, tokens, steps, positions,
                                         rows, heads, draft_step, previous, stream);
    }
    bool validate(const char* resource, std::string& err) const {
        if (!scratch_) return true;
        if (!verify_canary_matches(*output(host_), expected_, 1)) {
            err = std::string(resource) + ": shared operation canary mismatch"; return false;
        }
        if (std::getenv("STRATA_STATE_HASH"))
            std::fprintf(stderr, "STATE_SCRATCH resource=%s request=%llu slot=%d epoch=%llu checked=1 inputs=%u\n",
                         resource, (unsigned long long) expected_.request_id, expected_.slot,
                         (unsigned long long) expected_.epoch, *input_checks(host_));
        return true;
    }
private:
    static VerifyCanaryOutput* output(VerifyCanaryInput* p) {
        return reinterpret_cast<VerifyCanaryOutput*>(p + 1);
    }
    VerifyCanaryOwner* owners() const { return reinterpret_cast<VerifyCanaryOwner*>(scratch_ + 1); }
    OperationInputs* inputs() const { return reinterpret_cast<OperationInputs*>(owners() + 9); }
    static uint32_t* input_checks(VerifyCanaryInput* p) { return reinterpret_cast<uint32_t*>(output(p) + 1); }
    int device_ = -1;
    uint64_t ids_[9] = {}, epoch_ = 0;
    VerifyCanaryInput expected_;
    VerifyCanaryInput *host_ = nullptr, *mapped_ = nullptr;
    VerifyCanaryOutput* scratch_ = nullptr;
};

static_assert(kOperationCanaryDeviceBytes == 280 && kOperationCanaryHostBytes == 108);
} // namespace strata::core
