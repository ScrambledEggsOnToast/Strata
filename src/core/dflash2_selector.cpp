#include "strata/core/dflash2_selector.hpp"

#include "strata/core/on_device.hpp"
#include "strata/core/operation_lease.hpp"
#include "strata/kernels/dflash2_selector.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <exception>
#include <stdexcept>

namespace strata::core::dflash2 {
namespace {
namespace selector = strata::kernels::dflash2;

std::string cuda_failure(const char* operation, cudaError_t status) {
    return std::string("dflash2 selector: ") + operation + ": " + cudaGetErrorString(status);
}

bool current_device(int device, std::string& error) {
    int current = -1;
    const auto status = cudaGetDevice(&current);
    if (status != cudaSuccess) { error = cuda_failure("get current device", status); return false; }
    if (current != device) { error = "dflash2 selector: could not enter the owning device"; return false; }
    return true;
}

bool device_pointer(const void* pointer, int device, const char* name, std::string& error) {
    if (pointer == nullptr) { error = std::string("dflash2 selector: null ") + name; return false; }
    cudaPointerAttributes attributes{};
    const auto status = cudaPointerGetAttributes(&attributes, pointer);
    if (status != cudaSuccess) {
        error = cuda_failure(name, status);
        (void) cudaGetLastError(); // discard only this refused pointer query's error
        return false;
    }
    if (attributes.type != cudaMemoryTypeDevice || attributes.device != device) {
        error = std::string("dflash2 selector: ") + name + " is not storage on the owning device";
        return false;
    }
    return true;
}
} // namespace

void Selection::fill_probability_rows(int32_t* destination) const noexcept {
    for (int row = 0; row < kernels::dflash2::selector_rows; ++row) {
        int32_t* output = destination + size_t(row) * kSpecQStride;
        std::fill(output, output + kSpecQEntries, int32_t(-1));
        std::fill(output + kSpecQEntries, output + kSpecQStride, int32_t(0));
        if (!sampled) continue;
        for (int slot = 0; slot < 16; ++slot) {
            output[slot] = candidates[size_t(row)][size_t(slot)];
            std::memcpy(output + kSpecQEntries + slot,
                        &conditional_probabilities[size_t(row)][size_t(slot)], sizeof(float));
        }
    }
}

struct DeviceSelector::Impl {
    const int device;
    const uint16_t* const hidden_projection;
    const uint16_t* const predecessor_codebook;
    const uint16_t* const successor_codebook;
    OperationLease lease;
    std::atomic<bool> poisoned{false};
    selector::SelectorWorkspace* scratch = nullptr;
    selector::SelectorResult* host = nullptr;
    cudaStream_t pending_stream = nullptr; // borrowed, only while in_flight
    bool in_flight = false;

    Impl(int ordinal, const uint16_t* projection, const uint16_t* predecessor, const uint16_t* successor)
        : device(ordinal), hidden_projection(projection), predecessor_codebook(predecessor),
          successor_codebook(successor) {
        if (device < 0) throw std::invalid_argument("dflash2 selector: negative device");
        const OnDevice on(device);
        std::string error;
        if (!current_device(device, error) ||
            !device_pointer(hidden_projection, device, "hidden projection", error) ||
            !device_pointer(predecessor_codebook, device, "predecessor codebook", error) ||
            !device_pointer(successor_codebook, device, "successor codebook", error))
            throw std::invalid_argument(error);
        cudaDeviceProp properties{};
        auto status = cudaGetDeviceProperties(&properties, device);
        if (status != cudaSuccess) throw std::runtime_error(cuda_failure("device properties", status));
        if (properties.major < 7) throw std::invalid_argument("dflash2 selector: CUDA SM70 or newer is required");
        status = cudaMalloc(reinterpret_cast<void**>(&scratch), selector::selector_workspace_bytes());
        if (status != cudaSuccess) throw std::runtime_error(cuda_failure("scratch allocation", status));
        status = cudaHostAlloc(reinterpret_cast<void**>(&host), selector::selector_result_bytes(), cudaHostAllocPortable);
        if (status != cudaSuccess) {
            // No consumers have been launched; this is the only partial-construction allocation.
            if (cudaFree(scratch) != cudaSuccess) std::terminate();
            scratch = nullptr;
            throw std::runtime_error(cuda_failure("pinned result allocation", status));
        }
    }

    ~Impl() {
        // Object lifetime excludes concurrent host calls. Never return/refund buffers with
        // an undrained consumer; this is the same fail-stop rule as the native MTP owner.
        const OnDevice on(device);
        int current = -1;
        if (cudaGetDevice(&current) != cudaSuccess || current != device) std::terminate();
        if (in_flight && cudaStreamSynchronize(pending_stream) != cudaSuccess) std::terminate();
        if (host != nullptr && cudaFreeHost(host) != cudaSuccess) std::terminate();
        if (scratch != nullptr && cudaFree(scratch) != cudaSuccess) std::terminate();
    }

    // The guard is destroyed before the operation lease, including on C++ exceptions.
    struct PendingDrain {
        Impl& owner;
        std::string& error;
        bool attempted = false;
        PendingDrain(Impl& state, cudaStream_t stream, std::string& message) : owner(state), error(message) {
            owner.pending_stream = stream;
            owner.in_flight = true;
        }
        ~PendingDrain() {
            if (!owner.in_flight) return;
            owner.poisoned.store(true, std::memory_order_release);
            if (attempted) return;
            const auto status = cudaStreamSynchronize(owner.pending_stream);
            if (status == cudaSuccess) owner.in_flight = false;
            else {
                try { error += "; " + cuda_failure("failure drain", status); }
                catch (...) {} // preserve the original exception and retain poisoned ownership
            }
        }
        cudaError_t finish() {
            attempted = true;
            const auto status = cudaStreamSynchronize(owner.pending_stream);
            if (status == cudaSuccess) owner.in_flight = false;
            return status;
        }
    };
};

DeviceSelector::DeviceSelector(int device, const uint16_t* hidden_projection, const uint16_t* predecessor_codebook,
                               const uint16_t* successor_codebook)
    : impl_(std::make_unique<Impl>(device, hidden_projection, predecessor_codebook, successor_codebook)) {}
DeviceSelector::~DeviceSelector() = default;

bool DeviceSelector::select(int rows, const uint16_t* hidden, const float* logits, int32_t anchor, float temperature,
                            const std::array<float, 3>& uniforms, Selection& output, void* stream, std::string& error) {
    auto& state = *impl_;
    error.clear();
    OperationLease::Guard lease(state.lease);
    if (!lease) {
        error = lease.failure() == OperationLease::Failure::timeout
            ? "dflash2 selector: scratch lease wait exceeded 2000 ms" : "dflash2 selector: scratch is poisoned";
        return false;
    }
    // Argument refusals occur before any scratch mutation and do not poison a healthy owner.
    if (rows < 1 || rows > selector::selector_rows) {
        error = "dflash2 selector: proposal row count is outside 1..selector_rows";
        lease.complete(); return false;
    }
    if (temperature > 0.0f && rows > selector::selector_sampled_rows) {
        error = "dflash2 selector: sampled selection supports at most three proposal rows";
        lease.complete(); return false;
    }
    if (anchor < 0 || anchor >= selector::selector_vocabulary || !std::isfinite(temperature) || temperature < 0.0f) {
        error = "dflash2 selector: anchor or temperature is outside the reference contract";
        lease.complete(); return false;
    }
    if (temperature > 0.0f)
        for (float uniform : uniforms)
            if (!std::isfinite(uniform) || !(uniform >= 0.0f && uniform < 1.0f)) {
                error = "dflash2 selector: sampled uniforms must be finite and in [0,1)";
                lease.complete(); return false;
            }
    const OnDevice on(state.device);
    if (!current_device(state.device, error) || !device_pointer(hidden, state.device, "hidden input", error) ||
        !device_pointer(logits, state.device, "logit input", error)) {
        lease.complete(); return false;
    }
    const auto cuda_stream = static_cast<cudaStream_t>(stream);
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    auto status = cudaStreamIsCapturing(cuda_stream, &capture);
    if (status != cudaSuccess || capture != cudaStreamCaptureStatusNone) {
        error = status != cudaSuccess ? cuda_failure("stream ownership/capture query", status)
                                     : "dflash2 selector: host-returning selection cannot run during graph capture";
        lease.complete(); return false;
    }
    Impl::PendingDrain pending(state, cuda_stream, error);
    status = static_cast<cudaError_t>(selector::selector_select(rows, hidden, logits, state.hidden_projection,
        state.predecessor_codebook, state.successor_codebook, anchor, temperature, uniforms[0], uniforms[1], uniforms[2],
        state.scratch, stream));
    if (status != cudaSuccess) { error = cuda_failure("kernel submission", status); return false; }
    status = cudaMemcpyAsync(state.host, &state.scratch->result, sizeof(selector::SelectorResult),
                             cudaMemcpyDeviceToHost, cuda_stream);
    if (status != cudaSuccess) { error = cuda_failure("bounded result readback", status); return false; }
    status = pending.finish();
    if (status != cudaSuccess) { error = cuda_failure("selection completion", status); return false; }
    if (state.host->status != 0) {
        state.poisoned.store(true, std::memory_order_release);
        error = "dflash2 selector: nonfinite device arithmetic (status " + std::to_string(state.host->status) + ")";
        return false;
    }
    Selection selected;
    selected.sampled = temperature > 0.0f;
    std::memcpy(selected.tokens.data(), state.host->tokens, sizeof(state.host->tokens));
    std::memcpy(selected.confidence.data(), state.host->confidence, sizeof(state.host->confidence));
    for (size_t row = 0; row < selected.candidates.size(); ++row) {
        std::memcpy(selected.candidates[row].data(), state.host->candidates[row], sizeof(state.host->candidates[row]));
        std::memcpy(selected.conditional_probabilities[row].data(), state.host->conditional_probabilities[row],
                    sizeof(state.host->conditional_probabilities[row]));
    }
    output = selected;
    lease.complete();
    return true;
}

bool DeviceSelector::drain(std::string& error) {
    auto& state = *impl_;
    error.clear();
    OperationLease::Guard lease(state.lease);
    // A poisoned Guard still owns the mutex (and cannot unpoison it). Drain may
    // join retained consumers, but never bless that allocation for another select.
    if (!lease && lease.failure() != OperationLease::Failure::poisoned) {
        error = "dflash2 selector: scratch lease wait exceeded 2000 ms while draining";
        return false;
    }
    const OnDevice on(state.device);
    if (!current_device(state.device, error)) {
        state.poisoned.store(true, std::memory_order_release);
        return false;
    }
    if (state.in_flight) {
        const auto status = cudaStreamSynchronize(state.pending_stream);
        if (status != cudaSuccess) {
            state.poisoned.store(true, std::memory_order_release);
            error = cuda_failure("explicit drain", status);
            return false;
        }
        state.in_flight = false;
    }
    lease.complete();
    return true;
}

bool DeviceSelector::poisoned() const noexcept { return impl_->poisoned.load(std::memory_order_acquire); }
int DeviceSelector::device() const noexcept { return impl_->device; }
size_t DeviceSelector::device_bytes() const noexcept { return scratch_bytes(); }
size_t DeviceSelector::scratch_bytes() noexcept { return selector::selector_workspace_bytes(); }
size_t DeviceSelector::pinned_bytes() noexcept { return selector::selector_result_bytes(); }

} // namespace strata::core::dflash2
