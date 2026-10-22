// Throwaway actual-input screen, NOT an engine path or service/quality benchmark.
// Input: layer 0, final mlp_conv.addcmul, stage 0 / offset 1, proposal rows 4.
// Output: mlp.gate_proj before SiLU. No synthetic data, shape sweep, or CLI tuning.
// Parent queue MUST bind SHA-256 of this source, retained kernel/header, input,
// replay gate output, original dense.bin, compiler and libraries.
// FNV below is only a diagnostic, not that SHA-256 identity check.
// dense.bin: 1206131200 bytes, SHA-256
// 75c78461ce6be468012b3c415b86ef738153b47494db102a9ef37540a84592d6.
// Unchanged gate slice SHA-256:
// a52da862c8a00dde400ac019a51df0574c92cc923abed596fcfd9f02b4a960f0.
// Compile this CUDA TU and the actual retained dflash2_forward.cu together;
// include its existing headers and link CUDA 12.9 cuBLAS/cuBLASLt/cudart.
#include "strata/kernels/dflash2_forward.hpp"
#include "strata/kernels/bf16_bits.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

namespace gpu = strata::kernels::dflash2;
namespace {
constexpr int rows = 4, ni = 2560, no = 7680;
constexpr int warmup_calls = 4, timed_calls = 128;
constexpr size_t input_count = size_t(rows) * ni;
constexpr size_t weight_count = size_t(no) * ni;
constexpr size_t output_count = size_t(rows) * no;
constexpr size_t input_bytes = input_count * sizeof(uint16_t);
constexpr size_t weight_bytes = weight_count * sizeof(uint16_t);
constexpr size_t output_bytes = output_count * sizeof(uint16_t);
constexpr size_t dense_bytes = 1206131200;
constexpr size_t weight_offset = 363755520;
constexpr size_t workspace_bytes = size_t(4) << 20;
constexpr size_t common_device_bytes = input_bytes + weight_bytes + output_bytes + sizeof(uint32_t);
constexpr size_t candidate_extra_bytes = workspace_bytes + output_count * sizeof(float) + output_bytes + sizeof(uint32_t);
static_assert(ni == gpu::hidden && no == gpu::intermediate);
static_assert(weight_bytes == 39321600 && weight_offset + weight_bytes == 403077120);
static_assert(weight_offset + weight_bytes <= dense_bytes);
static_assert(sizeof(float) == 4 && sizeof(uint16_t) == 2);
static_assert(std::endian::native == std::endian::little, "actual raw BF16 files are little-endian");

using Clock = std::chrono::steady_clock;
double microseconds(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::micro>(end - begin).count();
}
void require(bool ok, const char* why) {
    if (!ok) throw std::runtime_error(why);
}
void checked(cudaError_t status, const char* op) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(op) + ": " + cudaGetErrorString(status));
}
void checked_blas(cublasStatus_t status, const char* op) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(op) + ": cuBLAS status=" + std::to_string(int(status)));
}
struct Buffer {
    void* pointer = nullptr;
    explicit Buffer(size_t bytes) { checked(cudaMalloc(&pointer, bytes), "cudaMalloc"); }
    ~Buffer() { if (pointer) cudaFree(pointer); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    template<class T> T* as() const { return static_cast<T*>(pointer); }
};
struct PinnedStatus {
    uint32_t* pointer = nullptr;
    PinnedStatus() { checked(cudaMallocHost(reinterpret_cast<void**>(&pointer), sizeof(uint32_t)), "cudaMallocHost status"); }
    ~PinnedStatus() { if (pointer) cudaFreeHost(pointer); }
    PinnedStatus(const PinnedStatus&) = delete;
    PinnedStatus& operator=(const PinnedStatus&) = delete;
};
struct Stream {
    cudaStream_t value = nullptr;
    Stream() { checked(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking), "create stream"); }
    ~Stream() { if (value) { cudaStreamSynchronize(value); cudaStreamDestroy(value); } }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
};
struct Event {
    cudaEvent_t value = nullptr;
    Event() { checked(cudaEventCreate(&value), "create event"); }
    ~Event() { if (value) cudaEventDestroy(value); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
};
struct Blas {
    cublasHandle_t value = nullptr;
    Blas() { checked_blas(cublasCreate(&value), "create cuBLAS handle"); }
    ~Blas() { if (value) cublasDestroy(value); }
    Blas(const Blas&) = delete;
    Blas& operator=(const Blas&) = delete;
};
size_t resident_bytes() {
    std::ifstream file("/proc/self/statm");
    size_t total = 0, resident = 0;
    const long page = sysconf(_SC_PAGESIZE);
    require(bool(file >> total >> resident) && page > 0, "cannot sample host RSS");
    return resident * size_t(page);
}
std::vector<uint16_t> read_actual(const char* path, size_t expected_file_bytes,
                                  size_t offset, size_t count) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    require(bool(file), "cannot open actual BF16 file");
    const std::streamoff extent = file.tellg();
    require(extent >= 0 && size_t(extent) == expected_file_bytes, "actual file extent differs from pinned contract");
    require(offset <= expected_file_bytes && count * sizeof(uint16_t) <= expected_file_bytes - offset,
            "actual BF16 slice exceeds file extent");
    file.seekg(static_cast<std::streamoff>(offset));
    require(bool(file), "cannot seek actual BF16 slice");
    std::vector<uint16_t> values(count);
    file.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(count * sizeof(uint16_t)));
    require(bool(file) && size_t(file.gcount()) == count * sizeof(uint16_t), "short actual BF16 read");
    return values;
}
// Hash raw little-endian BF16 bytes while performing the required finite scan.
uint64_t finite_hash(const std::vector<uint16_t>& values) {
    uint64_t hash = 14695981039346656037ULL;
    for (uint16_t value : values) {
        require((value & 0x7f80u) != 0x7f80u, "nonfinite BF16 data");
        hash = (hash ^ (value & 0xffu)) * 1099511628211ULL;
        hash = (hash ^ (value >> 8)) * 1099511628211ULL;
    }
    return hash;
}
int parse_device(const char* text) {
    require(text && *text >= '0' && *text <= '9', "device must be a nonnegative visible ordinal");
    errno = 0;
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    require(errno == 0 && end && *end == '\0' && value >= 0 && value <= INT_MAX, "invalid device ordinal");
    return int(value);
}
// Only the candidate epilogue is task-local. Retained projection is linked,
// never copied. Use the existing BF16 round-to-nearest-even implementation.
__global__ void convert_and_check(const float* input, uint16_t* output, uint32_t* status) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= output_count) return;
    const float value = input[index];
    const uint16_t bits = strata::kernels::bf16_from_f32(value);
    output[index] = bits;
    if ((__float_as_uint(value) & 0x7f800000u) == 0x7f800000u || (bits & 0x7f80u) == 0x7f80u)
        atomicOr(status, 1u);
}
struct Sample { double gpu_us = 0, host_us = 0; };
void output_statistics(const char* arm, const std::vector<uint16_t>& values) {
    const uint64_t hash = finite_hash(values);
    double minimum = strata::kernels::f32_from_bf16(values.front());
    double maximum = minimum, sum = 0, squares = 0;
    for (uint16_t bits : values) {
        const double value = strata::kernels::f32_from_bf16(bits);
        minimum = std::min(minimum, value); maximum = std::max(maximum, value);
        sum += value; squares += value * value;
    }
    std::printf("RESULT kind=output arm=%s count=%zu fnv1a64_le=%016llx min=%.17g max=%.17g mean=%.17g rms=%.17g first8_bf16_hex=",
                arm, values.size(), static_cast<unsigned long long>(hash), minimum, maximum,
                sum / values.size(), std::sqrt(squares / values.size()));
    for (size_t i = 0; i < 8; ++i) std::printf("%s%04x", i ? "," : "", unsigned(values[i]));
    std::printf("\n");
}
} // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    try {
        require(argc == 9, "usage: actual_gate_micro --device N --input ACTUAL_LAYER0_STAGE0_OFFSET1_ADDCMUL_BF16 --dense ORIGINAL_DENSE_BIN --expected-control-output ACTUAL_REPLAY_GATE_BF16; requires STRATA_DFLASH2_LINEAR_REUSE=1");
        const char *input_path = nullptr, *dense_path = nullptr, *expected_path = nullptr;
        int device = -1;
        for (int arg = 1; arg < argc; arg += 2) {
            if (std::strcmp(argv[arg], "--device") == 0) {
                require(device == -1, "duplicate --device"); device = parse_device(argv[arg + 1]);
            } else if (std::strcmp(argv[arg], "--input") == 0) {
                require(input_path == nullptr, "duplicate --input"); input_path = argv[arg + 1];
            } else if (std::strcmp(argv[arg], "--dense") == 0) {
                require(dense_path == nullptr, "duplicate --dense"); dense_path = argv[arg + 1];
            } else if (std::strcmp(argv[arg], "--expected-control-output") == 0) {
                require(expected_path == nullptr, "duplicate --expected-control-output"); expected_path = argv[arg + 1];
            } else throw std::runtime_error("unknown CLI option; shape and iteration counts are fixed");
        }
        require(device >= 0 && input_path && dense_path && expected_path, "all four CLI options are required");
        const char* reuse = std::getenv("STRATA_DFLASH2_LINEAR_REUSE");
        require(reuse && std::strcmp(reuse, "1") == 0, "retained control requires STRATA_DFLASH2_LINEAR_REUSE=1");
        const auto setup_begin = Clock::now();
        auto input = read_actual(input_path, input_bytes, 0, input_count);
        auto weights = read_actual(dense_path, dense_bytes, weight_offset, weight_count);
        auto expected = read_actual(expected_path, output_bytes, 0, output_count);
        const uint64_t input_hash = finite_hash(input), weight_hash = finite_hash(weights), expected_hash = finite_hash(expected);
        std::array<std::vector<uint16_t>, 2> outputs{
            std::vector<uint16_t>(output_count), std::vector<uint16_t>(output_count)};
        const double file_load_validate_us = microseconds(setup_begin, Clock::now());
        const size_t rss_before = resident_bytes();
        const auto cuda_setup_begin = Clock::now();
        int devices = 0;
        checked(cudaGetDeviceCount(&devices), "device count");
        require(device < devices, "requested visible device is absent");
        checked(cudaSetDevice(device), "select device");
        checked(cudaFree(nullptr), "initialize selected CUDA context");
        cudaDeviceProp properties{};
        checked(cudaGetDeviceProperties(&properties, device), "device properties");
        require(properties.major == 8 && properties.minor == 6, "actual gate screen is restricted to SM86");
        int runtime_version = 0, driver_version = 0;
        checked(cudaRuntimeGetVersion(&runtime_version), "runtime version");
        checked(cudaDriverGetVersion(&driver_version), "driver version");
        size_t free_before = 0, total_device = 0;
        checked(cudaMemGetInfo(&free_before, &total_device), "memory before explicit setup");
        Buffer x(input_bytes), weight(weight_bytes), retained_output(output_bytes), candidate_output(output_bytes);
        Buffer retained_status(sizeof(uint32_t)), candidate_status(sizeof(uint32_t));
        Buffer result_fp32(output_count * sizeof(float)), workspace(workspace_bytes);
        PinnedStatus host_status;
        Stream stream;
        Event start, stop;
        const auto blas_begin = Clock::now();
        Blas blas;
        checked_blas(cublasSetStream(blas.value, stream.value), "set cuBLAS stream");
        // SetStream resets workspace, so bind the fixed 4 MiB bound afterwards.
        checked_blas(cublasSetWorkspace(blas.value, workspace.pointer, workspace_bytes), "set bounded cuBLAS workspace");
        checked_blas(cublasSetPointerMode(blas.value, CUBLAS_POINTER_MODE_HOST), "host alpha/beta");
        checked_blas(cublasSetMathMode(blas.value, static_cast<cublasMath_t>(
            CUBLAS_DEFAULT_MATH | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)), "FP32 reduction policy");
        checked_blas(cublasSetAtomicsMode(blas.value, CUBLAS_ATOMICS_NOT_ALLOWED), "cuBLAS atomics policy");
        int blas_version = 0;
        checked_blas(cublasGetVersion(blas.value, &blas_version), "cuBLAS version");
        const double blas_setup_us = microseconds(blas_begin, Clock::now());
        checked(cudaMemcpyAsync(x.pointer, input.data(), input_bytes, cudaMemcpyHostToDevice, stream.value), "upload actual input");
        checked(cudaMemcpyAsync(weight.pointer, weights.data(), weight_bytes, cudaMemcpyHostToDevice, stream.value), "upload unchanged gate weights");
        checked(cudaStreamSynchronize(stream.value), "complete actual data upload");
        const double cuda_setup_us = microseconds(cuda_setup_begin, Clock::now());
        std::printf("RESULT kind=contract version=actual_gate_micro_v1 rows=%d ni=%d no=%d layer=0 input_op=mlp_conv.addcmul stage=0 offset=1 output_op=mlp.gate_proj input_bytes=%zu dense_bytes=%zu weight_offset=%zu weight_bytes=%zu expected_control_bytes=%zu input_fnv1a64_le=%016llx weight_fnv1a64_le=%016llx expected_control_fnv1a64_le=%016llx parent_sha256_binding_required=1 linear_reuse=1 warmup_calls_per_arm=%d cold_calls_per_arm=1 timing_pairs=2 timed_calls_per_measurement=%d\n",
                    rows, ni, no, input_bytes, dense_bytes, weight_offset, weight_bytes, output_bytes,
                    static_cast<unsigned long long>(input_hash), static_cast<unsigned long long>(weight_hash),
                    static_cast<unsigned long long>(expected_hash), warmup_calls, timed_calls);
        std::printf("RESULT kind=device ordinal=%d cc=86 pci_domain=%d pci_bus=%d pci_device=%d uuid_hex=",
                    device, properties.pciDomainID, properties.pciBusID, properties.pciDeviceID);
        for (int i = 0; i < 16; ++i) std::printf("%02x", static_cast<unsigned char>(properties.uuid.bytes[i]));
        std::printf(" cuda_compile=%d cuda_runtime=%d cuda_driver=%d cublas_version=%d\n",
                    CUDART_VERSION, runtime_version, driver_version, blas_version);
        std::printf("RESULT kind=arithmetic control=retained_linear_bf16_reuse candidate=GemmEx_BF16_BF16_FP32_then_BF16_RNE compute=CUBLAS_COMPUTE_32F algorithm=CUBLAS_GEMM_DEFAULT reduced_precision_reduction=disallowed atomics=disallowed conversion_per_call=1 finite_or_per_call=1 status_clear_and_pinned_readback_per_batch=1 graphs=0\n");
        auto launch = [&](int arm) {
            if (arm == 0) {
                checked(static_cast<cudaError_t>(gpu::linear_bf16(x.as<uint16_t>(), weight.as<uint16_t>(),
                    retained_output.as<uint16_t>(), rows, ni, no, stream.value, retained_status.as<uint32_t>())), "retained linear_bf16");
            } else {
                const float alpha = 1.0f, beta = 0.0f;
                // Row-major Y = X W^T is column-major Y^T = W X^T.
                // Original BF16 weights and inputs are passed without conversion.
                checked_blas(cublasGemmEx(blas.value, CUBLAS_OP_T, CUBLAS_OP_N, no, rows, ni,
                    &alpha, weight.pointer, CUDA_R_16BF, ni, x.pointer, CUDA_R_16BF, ni,
                    &beta, result_fp32.pointer, CUDA_R_32F, no, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT), "native BF16 GemmEx");
                convert_and_check<<<unsigned((output_count + 255) / 256), 256, 0, stream.value>>>(
                    result_fp32.as<float>(), candidate_output.as<uint16_t>(), candidate_status.as<uint32_t>());
                checked(cudaGetLastError(), "candidate conversion and finite status");
            }
        };
        auto measure = [&](int arm, int calls, const char* phase, int pair, int order) {
            checked(cudaStreamSynchronize(stream.value), "drain before batch");
            uint32_t* status = arm == 0 ? retained_status.as<uint32_t>() : candidate_status.as<uint32_t>();
            const auto begin = Clock::now();
            checked(cudaEventRecord(start.value, stream.value), "record start");
            checked(cudaMemsetAsync(status, 0, sizeof(uint32_t), stream.value), "reset batch finite status");
            for (int call = 0; call < calls; ++call) launch(arm);
            // OR is sticky across every call in this batch; no earlier failure
            // can be erased by a later iteration. Reset/readback cost is symmetric.
            checked(cudaMemcpyAsync(host_status.pointer, status, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream.value), "read batch finite status");
            checked(cudaEventRecord(stop.value, stream.value), "record stop");
            checked(cudaEventSynchronize(stop.value), "complete batch");
            const uint32_t finite_status = *host_status.pointer;
            const auto end = Clock::now();
            float milliseconds = 0;
            checked(cudaEventElapsedTime(&milliseconds, start.value, stop.value), "elapsed CUDA events");
            const Sample sample{double(milliseconds) * 1000.0 / calls, microseconds(begin, end) / calls};
            std::printf("RESULT kind=timing phase=%s pair=%d order=%d arm=%s calls=%d gpu_us_per_call=%.9g host_wall_us_per_call=%.9g finite_status=%u\n",
                        phase, pair, order, arm == 0 ? "retained" : "candidate", calls, sample.gpu_us, sample.host_us, finite_status);
            require(finite_status == 0, "nonfinite projection output; screen refused");
            return sample;
        };
        // Cold calls are separate setup evidence, never mixed into paired timing.
        measure(0, 1, "cold", -1, 0);
        checked(cudaMemcpy(outputs[0].data(), retained_output.pointer, output_bytes, cudaMemcpyDeviceToHost), "snapshot cold retained output");
        const bool cold_control_matches_replay = std::memcmp(outputs[0].data(), expected.data(), output_bytes) == 0;
        std::printf("RESULT kind=control_provenance phase=cold replay_gate_bytes=%zu byte_equal=%d candidate_quality_gate=0\n",
                    output_bytes, int(cold_control_matches_replay));
        require(cold_control_matches_replay, "retained output differs from actual replay gate output; provenance refused");
        measure(1, 1, "cold", -1, 1);
        measure(0, warmup_calls, "warmup", -1, 0); measure(1, warmup_calls, "warmup", -1, 1);
        size_t free_after_setup = 0;
        checked(cudaMemGetInfo(&free_after_setup, &total_device), "memory after bounded warmup");
        const size_t rss_after_setup = resident_bytes();
        std::printf("RESULT kind=setup file_load_validate_host_us=%.9g cuda_setup_host_us=%.9g cublas_handle_setup_host_us=%.9g initial_setup_through_warmup_host_us=%.9g common_device_requested_bytes=%zu candidate_extra_device_requested_bytes=%zu total_device_requested_bytes=%zu cublas_workspace_bytes=%zu fp32_result_bytes=%zu pinned_status_bytes=%zu host_bf16_payload_and_snapshot_bytes=%zu device_free_before_bytes=%zu device_free_after_bytes=%zu device_total_bytes=%zu host_rss_before_bytes=%zu host_rss_after_bytes=%zu endpoint_samples_only=1 cuda_context_excluded_from_device_baseline=1\n",
                    file_load_validate_us, cuda_setup_us, blas_setup_us, microseconds(setup_begin, Clock::now()),
                    common_device_bytes, candidate_extra_bytes, common_device_bytes + candidate_extra_bytes,
                    workspace_bytes, output_count * sizeof(float), sizeof(uint32_t), input_bytes + weight_bytes + 3 * output_bytes,
                    free_before, free_after_setup, total_device, rss_before, rss_after_setup);
        std::array<std::array<Sample, 2>, 2> timings{};
        constexpr int orders[2][2] = {{0, 1}, {1, 0}};
        for (int pair = 0; pair < 2; ++pair) {
            for (int step = 0; step < 2; ++step) {
                const int arm = orders[pair][step];
                timings[pair][arm] = measure(arm, timed_calls, "paired", pair, step);
            }
            std::printf("RESULT kind=pair pair=%d arm_order=%s gpu_saved_us_per_call=%.9g host_wall_saved_us_per_call=%.9g\n",
                        pair, pair == 0 ? "retained,candidate" : "candidate,retained",
                        timings[pair][0].gpu_us - timings[pair][1].gpu_us,
                        timings[pair][0].host_us - timings[pair][1].host_us);
        }
        checked(cudaMemcpy(outputs[0].data(), retained_output.pointer, output_bytes, cudaMemcpyDeviceToHost), "snapshot retained output");
        checked(cudaMemcpy(outputs[1].data(), candidate_output.pointer, output_bytes, cudaMemcpyDeviceToHost), "snapshot candidate output");
        const bool final_control_matches_replay = std::memcmp(outputs[0].data(), expected.data(), output_bytes) == 0;
        std::printf("RESULT kind=control_provenance phase=paired replay_gate_bytes=%zu byte_equal=%d candidate_quality_gate=0\n",
                    output_bytes, int(final_control_matches_replay));
        require(final_control_matches_replay, "timed retained output differs from actual replay gate output; provenance refused");
        output_statistics("retained", outputs[0]); output_statistics("candidate", outputs[1]);
        size_t bit_differences = 0;
        double max_abs = 0, squared_error = 0;
        for (size_t i = 0; i < output_count; ++i) {
            bit_differences += outputs[0][i] != outputs[1][i];
            const double delta = double(strata::kernels::f32_from_bf16(outputs[1][i])) - strata::kernels::f32_from_bf16(outputs[0][i]);
            max_abs = std::max(max_abs, std::abs(delta)); squared_error += delta * delta;
        }
        std::printf("RESULT kind=numerical count=%zu bf16_bit_differences=%zu max_abs=%.17g rmse=%.17g diagnostic_only=1 finite_output_required=1 token_identity_gate=0 quality_evaluation_performed=0\n",
                    output_count, bit_differences, max_abs, std::sqrt(squared_error / output_count));
        std::printf("RESULT kind=complete status=PASS component_only=1 service_gain_claim=0 parent_decides_screen=1 required_saving_us_per_call=10 both_orders_must_improve=1 timing_pairs=2 timed_calls_per_measurement=%d\n", timed_calls);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL actual_gate_micro: %s\n", error.what());
        std::printf("RESULT kind=complete status=FAIL\n");
        return 1;
    }
}
