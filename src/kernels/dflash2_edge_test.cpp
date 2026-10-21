// Synthetic DFlash2 CUDA boundaries only: no model, CPU fallback, or performance claim.
// Run explicitly in the guarded measurement window, never as an unguarded CTest.
// The external 64-byte tie order is generated before CUDA by the pinned CPU oracle.
#include "strata/core/dflash2_selector.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/dflash2_forward.hpp"
#include "strata/kernels/dflash2_selector.hpp"
#include "strata/kernels/sampler.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
namespace gpu = strata::kernels::dflash2;
using strata::core::dflash2::DeviceSelector;
using strata::core::dflash2::Selection;
using strata::kernels::bf16_from_f32;
using strata::kernels::f32_from_bf16;

constexpr size_t context_elements = size_t(gpu::capacity) * gpu::kv_width;
constexpr size_t score_elements = size_t(gpu::query_heads) * gpu::block * gpu::max_keys;
constexpr size_t convolution_elements = size_t(gpu::block) * gpu::hidden;
constexpr size_t logit_elements = size_t(gpu::selector_rows) * gpu::selector_vocabulary;
constexpr double attention_absolute_tolerance = 1e-3;
constexpr double probability_absolute_tolerance = 2e-6;
constexpr double probability_mass_tolerance = 16 * double(std::numeric_limits<float>::epsilon());
static_assert(gpu::capacity == 2047 && gpu::block == 4 && gpu::layers == 5);
static_assert(gpu::query_heads == 32 && gpu::kv_heads == 8 && gpu::head_dim == 128);
static_assert(gpu::hidden == 2560 && gpu::groups == 160 && gpu::dynamic_width == 640);
static_assert(gpu::selector_rows == 3 && gpu::selector_candidates == 16);
static_assert(gpu::selector_hidden == gpu::hidden && gpu::selector_rank == 256);
static_assert(gpu::selector_vocabulary == 248320);

// sample_tokens_spec reserves at least 16 rows of 64 candidates per split block.
// This standalone process uses one device/stream and one vocabulary, so it never
// grows or retires sampler scratch. The exported query gives the per-row bytes.
constexpr size_t sampler_scratch_bound = size_t(16) * 64 * strata::core::kSpecQEntries * sizeof(int2);
constexpr uint64_t consumer_counter = 37;
constexpr std::array<int32_t, 3> known_drafts{1, 0, 1};

struct TargetRow {
    std::array<int32_t, 3> ids{};
    std::array<double, 3> probabilities{};
    std::array<float, 3> logits{};
    int count = 2;

    strata::core::SpecP distribution() const noexcept {
        return {ids.data(), probabilities.data(), count};
    }
};

struct ConsumerCase {
    const char* name = nullptr;
    int rows = 1;
    int drafts = 1;
    std::array<TargetRow, 3> target{};
    uint64_t seed = 0;
    std::array<int32_t, 3> expected{};
    std::array<int, 2> accepted{};
};

struct ConsumerPlan {
    std::array<int32_t, 16> ids{};
    std::array<std::array<float, 16>, 3> conditional{};
    std::array<float, 16> base_head{};
    std::array<float, 16> row_one_marginal{};
    std::array<ConsumerCase, 5> cases{};
};

// One fixture allocation. No full codebook mirror is ever allocated on the host.
struct DevicePayload {
    uint16_t ring[gpu::kv_elements];
    uint16_t context_keys[context_elements];
    uint16_t context_values[context_elements];
    uint16_t query[gpu::block * gpu::query_width];
    uint16_t block_keys[gpu::block * gpu::kv_width];
    float scores[score_elements];
    uint16_t convolution_input[convolution_elements];
    uint16_t convolution_base[2 * 2 * gpu::hidden];
    uint16_t convolution_dynamic[gpu::block * gpu::dynamic_width];
    // Product becomes the accumulator in place, so poisoning cannot be hidden by a host reset.
    uint16_t convolution_output[convolution_elements];
    uint16_t projection[gpu::selector_rank * gpu::selector_hidden];
    uint16_t predecessor[size_t(gpu::selector_vocabulary) * gpu::selector_rank];
    uint16_t successor[size_t(gpu::selector_vocabulary) * gpu::selector_rank];
    uint16_t hidden[gpu::selector_rows * gpu::selector_hidden];
    float logits[logit_elements];
    int32_t consumer_q[Selection::probability_row_slots];
    int32_t consumer_drafts[gpu::selector_rows];
    int consumer_output[gpu::selector_rows];
    uint32_t status;
};

// One host allocation, reused for every case, plus DeviceSelector's bounded pinned result.
struct HostPayload {
    uint16_t staging[context_elements];
    float scores[score_elements];
    float logits[logit_elements];
    Selection selection{};
    std::array<int32_t, Selection::probability_row_slots> consumer_q{};
    std::array<int, gpu::selector_rows> consumer_output{};
    uint32_t status = 0;
};
static_assert(sizeof(DevicePayload) + sizeof(gpu::SelectorWorkspace) + sampler_scratch_bound < size_t(400) * 1024 * 1024);
static_assert(sizeof(HostPayload) + sizeof(gpu::SelectorResult) + sizeof(ConsumerPlan) < size_t(16) * 1024 * 1024);

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

[[noreturn]] void mismatch(const char* test, int case_index, int row, int column,
                          double actual, double expected) {
    char message[384];
    std::snprintf(message, sizeof(message),
                  "%s: case=%d row=%d column=%d actual=%.17g expected=%.17g",
                  test, case_index, row, column, actual, expected);
    throw std::runtime_error(message);
}

void cuda_check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return;
    char message[512];
    std::snprintf(message, sizeof(message), "%s: CUDA status=%d (%s): %s", operation,
                  int(status), cudaGetErrorName(status), cudaGetErrorString(status));
    throw std::runtime_error(message);
}

void kernel_check(int status, const char* operation) {
    cuda_check(static_cast<cudaError_t>(status), operation);
}

[[noreturn]] void fail_stop(cudaError_t status, const char* operation) noexcept {
    std::fprintf(stderr, "FAIL-STOP %s: CUDA status=%d (%s): %s; borrowed buffers retained until process exit\n",
                 operation, int(status), cudaGetErrorName(status), cudaGetErrorString(status));
    std::fflush(stderr);
    std::_Exit(2);
}

class DeviceBuffer {
public:
    DeviceBuffer() {
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&data_), sizeof(DevicePayload)), "fixture allocation");
    }
    ~DeviceBuffer() {
        const auto status = cudaFree(data_);
        if (status != cudaSuccess) fail_stop(status, "fixture release");
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DevicePayload* get() const noexcept { return data_; }
private:
    DevicePayload* data_ = nullptr;
};

// Declared after the selector and all borrowed owners. It fences both success and
// exception unwinding BEFORE either owner can release memory. An unfenced fault
// ends the process without unwinding/freeing buffers; there is no retry or reset.
struct DrainBeforeOwners {
    ~DrainBeforeOwners() {
        const auto status = cudaStreamSynchronize(nullptr);
        if (status != cudaSuccess) fail_stop(status, "default-stream completion before owner destruction");
    }
};

struct Options {
    int device = -1;
    const char* tie_reference = nullptr;
};

void usage(const char* executable) {
    std::printf(
        "Usage: %s --device N --tie-reference FILE\n"
        "       %s --help\n\n"
        "Explicit CUDA kernel/selector fixture; --help does not initialize CUDA.\n"
        "FILE must contain exactly 16 little-endian int32 IDs (64 bytes), generated\n"
        "by pinned torch 2.13.0+cpu / GCC 13.3: topk(k=16, sorted=False) of 248320\n"
        "FP32 logits with first 20 equal to 1 and the remainder equal to -20.\n"
        "No invented tie order or CPU fallback. Run only in the guarded GPU window.\n"
        "Covers two context rings, two block-local convolutions, pinned ties/CDF,\n"
        "and native conditional-16 q consumed by production rejection sampling.\n"
        "CPU Philox/reference cases are fixed before CUDA; no whole-model claim.\n",
        executable, executable);
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view argument(argv[i]);
        if (argument == "--device") {
            require(options.device == -1 && i + 1 < argc, "--device requires one nonnegative integer, once");
            const std::string_view value(argv[++i]);
            const auto result = std::from_chars(value.data(), value.data() + value.size(), options.device);
            require(result.ec == std::errc{} && result.ptr == value.data() + value.size() && options.device >= 0,
                    "--device requires one nonnegative integer");
        } else if (argument == "--tie-reference") {
            require(options.tie_reference == nullptr && i + 1 < argc, "--tie-reference requires one file, once");
            options.tie_reference = argv[++i];
            require(options.tie_reference[0] != '\0', "--tie-reference must not be empty");
        } else {
            throw std::runtime_error("unknown argument: " + std::string(argument));
        }
    }
    require(options.device >= 0 && options.tie_reference != nullptr,
            "both --device N and --tie-reference FILE are required (see --help)");
    return options;
}

std::array<int32_t, 16> read_tie_reference(const char* path) {
    std::ifstream input(path, std::ios::binary);
    require(input.is_open(), "cannot open --tie-reference file");
    std::array<unsigned char, 65> bytes{};
    input.read(reinterpret_cast<char*>(bytes.data()), std::streamsize(bytes.size()));
    require(!input.bad() && input.gcount() == 64, "--tie-reference must be exactly 64 bytes");
    std::array<int32_t, 16> ids{};
    for (size_t slot = 0; slot < ids.size(); ++slot) {
        const size_t offset = slot * 4;
        const uint32_t id = uint32_t(bytes[offset]) | (uint32_t(bytes[offset + 1]) << 8) |
                            (uint32_t(bytes[offset + 2]) << 16) | (uint32_t(bytes[offset + 3]) << 24);
        require(id < 20, "--tie-reference contains an ID outside the tied top-20 support");
        ids[slot] = int32_t(id);
        for (size_t previous = 0; previous < slot; ++previous)
            require(ids[previous] != ids[slot], "--tie-reference contains a duplicate ID");
    }
    return ids;
}

// Independent CPU oracle from spec_verify_parity.cpp. This is the production
// Philox counter mapping, not another acceptance algorithm or a selector RNG.
float host_philox_uniform(uint64_t seed, uint64_t counter) {
    uint32_t c0 = uint32_t(counter), c1 = uint32_t(counter >> 32);
    uint32_t c2 = uint32_t(seed), c3 = uint32_t(seed >> 32);
    for (int i = 0; i < 10; ++i) {
        const uint32_t hi0 = uint32_t((uint64_t(0x9E3779B9u) * c0) >> 32);
        const uint32_t hi1 = uint32_t((uint64_t(0xBB67AE85u) * c2) >> 32);
        const uint32_t lo0 = 0x9E3779B9u * c0;
        const uint32_t lo1 = 0xBB67AE85u * c2;
        const uint32_t n0 = hi1 ^ c1 ^ uint32_t(i);
        const uint32_t n2 = hi0 ^ c3;
        c0 = n0; c1 = lo1; c2 = n2; c3 = lo0;
    }
    return float(c0 >> 8) * (1.0f / 16777216.0f);
}

std::array<double, 16> known_conditional_q(double before) {
    // With raw logits (1,1,-20,...), the pinned partial_sort keeps IDs 0..15:
    // later tied -20 entries never strictly improve its initial heap. Their
    // order is immaterial to SpecQ's ID lookup; no GPU candidate list is read.
    std::array<double, 16> q{};
    double mass = 0.0;
    for (size_t token = 0; token < q.size(); ++token) {
        const double unary = token < 2 ? 1.0 : -20.0;
        const double after = token == 0 ? 1.0 : token == 1 ? -1.0 : 0.0;
        q[token] = std::exp(unary + before * after - 5.0);
        mass += q[token];
    }
    for (double& probability : q) probability /= mass;
    return q;
}

TargetRow binary_target(int32_t favored) {
    TargetRow row;
    row.ids = {favored, 1 - favored, -1};
    row.logits = {std::log(3.0f), 0.0f, 0.0f};
    const double odds = std::exp(double(row.logits[0]));
    row.probabilities = {odds / (odds + 1.0), 1.0 / (odds + 1.0), 0.0};
    return row;
}

void prepare_consumer_plan(ConsumerPlan& plan) {
    const auto after_a = known_conditional_q(-4.0);
    const auto after_b = known_conditional_q(4.0);
    const auto base = known_conditional_q(0.0);
    double other_predecessors = 0.0;
    for (size_t token = 3; token < after_a.size(); ++token)
        other_predecessors += after_a[token];
    for (size_t token = 0; token < plan.ids.size(); ++token) {
        plan.ids[token] = int32_t(token);
        plan.conditional[0][token] = plan.conditional[2][token] = float(after_a[token]);
        plan.conditional[1][token] = float(after_b[token]);
        plan.base_head[token] = float(base[token]);
        // True row-1 marginal before observing B: sum over row-0 predecessor.
        // A and the anchor ID 2 have coordinate -4, B has +4, others zero.
        plan.row_one_marginal[token] = float(after_a[1] * after_b[token] +
            (after_a[0] + after_a[2]) * after_a[token] + other_predecessors * base[token]);
    }
    const TargetRow favors_a = binary_target(0);
    const TargetRow favors_b = binary_target(1);
    plan.cases[0].name = "predictor-accept-not-pointmass-or-plain";
    plan.cases[0].target[0] = favors_b;
    plan.cases[1].name = "predictor-reject-not-base-head";
    plan.cases[1].target[0] = favors_a;
    plan.cases[2].name = "predictor-residual-subtracts-nonselected-q";
    TargetRow& residual = plan.cases[2].target[0];
    residual.count = 3;
    residual.ids = {0, 2, 1};
    residual.logits = {std::log(1.5f), std::log(1.5f), 0.0f};
    const double weight = std::exp(double(residual.logits[0]));
    residual.probabilities = {weight / (2.0 * weight + 1.0),
                              weight / (2.0 * weight + 1.0), 1.0 / (2.0 * weight + 1.0)};
    plan.cases[3].name = "window-all-accept-then-bonus";
    plan.cases[4].name = "window-row1-reject-not-marginal";
    for (size_t index = 3; index < plan.cases.size(); ++index) {
        auto& test = plan.cases[index];
        test.rows = 3;
        test.drafts = 2;
        test.target = {favors_b, index == 3 ? favors_a : favors_b, favors_a};
    }

    // Fix seeds before CUDA/selection. Narrow, prospective uniform regions put
    // the near-point-mass and marginal cases safely away from FP32 boundaries.
    // No search sees a GPU value, exported q, or an observed consumer outcome.
    const auto between = [](float value, float low, float high) {
        return value > low && value < high;
    };
    for (size_t index = 0; index < plan.cases.size(); ++index) {
        auto& test = plan.cases[index];
        for (uint64_t seed = 1; seed <= (uint64_t(1) << 22); ++seed) {
            const float accept0 = host_philox_uniform(seed, consumer_counter | strata::core::kSpecCtrAccept);
            bool chosen = false;
            if (index == 0) {
                chosen = between(accept0, 0.75005f, 0.75020f) &&
                    between(host_philox_uniform(seed, consumer_counter), 0.80f, 0.95f);
            } else if (index == 1) {
                chosen = between(accept0, 0.30f, 0.45f) &&
                    between(host_philox_uniform(seed, consumer_counter), 0.80f, 0.95f);
            } else if (index == 2) {
                chosen = between(accept0, 0.60f, 0.90f) &&
                    between(host_philox_uniform(seed, consumer_counter | strata::core::kSpecCtrResid),
                            0.49982f, 0.49994f) &&
                    host_philox_uniform(seed, consumer_counter) < 0.30f;
            } else if (between(accept0, 0.10f, 0.60f)) {
                const float accept1 = host_philox_uniform(seed, (consumer_counter + 1) | strata::core::kSpecCtrAccept);
                const float pick2 = host_philox_uniform(seed, consumer_counter + 2);
                chosen = index == 3 ?
                    between(accept1, 0.10f, 0.60f) && pick2 < 0.70f &&
                        host_philox_uniform(seed, (consumer_counter + 2) | strata::core::kSpecCtrAccept) < 0.20f :
                    between(accept1, 0.25011f, 0.25014f) && between(pick2, 0.80f, 0.95f);
            }
            if (chosen) { test.seed = seed; break; }
        }
        require(test.seed != 0, "CPU-only prospective Philox seed search exhausted its fixed bound");
        for (int row = 0; row < test.rows; ++row) {
            const auto p = test.target[size_t(row)].distribution();
            const uint64_t pos = consumer_counter + uint64_t(row);
            if (row < test.drafts) {
                const strata::core::SpecQ q{plan.ids.data(), plan.conditional[size_t(row)].data(), 16};
                const auto result = strata::core::spec_verify_row(p, known_drafts[size_t(row)], q,
                    host_philox_uniform(test.seed, pos | strata::core::kSpecCtrAccept),
                    host_philox_uniform(test.seed, pos | strata::core::kSpecCtrResid));
                test.expected[size_t(row)] = result.token;
                test.accepted[size_t(row)] = result.accepted;
            } else {
                test.expected[size_t(row)] = strata::core::spec_sample(p, host_philox_uniform(test.seed, pos));
            }
        }
        constexpr std::array<std::array<int32_t, 3>, 5> known_outcomes{{
            {1, 0, 0}, {0, 0, 0}, {2, 0, 0}, {1, 0, 0}, {1, 1, 1}}};
        constexpr std::array<std::array<int, 2>, 5> known_accepts{{
            {1, 0}, {0, 0}, {0, 0}, {1, 1}, {1, 0}}};
        require(test.expected == known_outcomes[index] && test.accepted == known_accepts[index],
                "independent conditional-q/Philox reference disagrees with the designed boundary case");

        const int row = index == 4 ? 1 : 0;
        const uint64_t pos = consumer_counter + uint64_t(row);
        const auto p = test.target[size_t(row)].distribution();
        const float accept = host_philox_uniform(test.seed, pos | strata::core::kSpecCtrAccept);
        const float resid = host_philox_uniform(test.seed, pos | strata::core::kSpecCtrResid);
        const strata::core::SpecQ point{nullptr, nullptr, -1};
        const strata::core::SpecQ base_q{plan.ids.data(), plan.base_head.data(), 16};
        const strata::core::SpecQ marginal_q{plan.ids.data(), plan.row_one_marginal.data(), 16};
        if (index == 0 || index == 2)
            require(strata::core::spec_verify_row(p, known_drafts[size_t(row)], point, accept, resid).token !=
                        test.expected[size_t(row)], "consumer case does not distinguish dropped q/point mass");
        if (index == 1)
            require(strata::core::spec_verify_row(p, known_drafts[size_t(row)], base_q, accept, resid).token !=
                        test.expected[size_t(row)], "consumer case does not distinguish base-head q");
        if (index == 4)
            require(strata::core::spec_verify_row(p, known_drafts[size_t(row)], marginal_q, accept, resid).token !=
                        test.expected[size_t(row)], "consumer case does not distinguish marginal q");
        if (index < 3)
            require(strata::core::spec_sample(p, host_philox_uniform(test.seed, pos)) != test.expected[size_t(row)],
                    "consumer case does not distinguish ordinary-sampler fallback");
        if (index == 3) {
            const uint64_t bonus_pos = consumer_counter + 2;
            const strata::core::SpecQ third_q{plan.ids.data(), plan.conditional[2].data(), 16};
            require(strata::core::spec_verify_row(test.target[2].distribution(), known_drafts[2], third_q,
                        host_philox_uniform(test.seed, bonus_pos | strata::core::kSpecCtrAccept),
                        host_philox_uniform(test.seed, bonus_pos | strata::core::kSpecCtrResid)).token != test.expected[2],
                    "consumer case does not distinguish a bonus row from a third draft");
        }
    }
}

void check_status(DevicePayload& device, HostPayload& host, const char* test, int case_index) {
    cuda_check(cudaMemcpy(&host.status, &device.status, sizeof(host.status), cudaMemcpyDeviceToHost),
               "kernel status readback");
    if (host.status != 0) mismatch(test, case_index, 0, 0, host.status, 0);
}

void encode_key(uint16_t* destination, int key) {
    destination[0] = bf16_from_f32(float(key & 15));
    destination[1] = bf16_from_f32(float((key >> 4) & 15));
    destination[2] = bf16_from_f32(float((key >> 8) & 15));
}

void attention_edges(DevicePayload& device, HostPayload& host) {
    // Row-major input keys encode every key exactly in BF16, without collisions.
    std::fill_n(host.staging, context_elements, uint16_t(0));
    for (int key = 0; key < gpu::capacity; ++key)
        for (int head = 0; head < gpu::kv_heads; ++head)
            encode_key(host.staging + (size_t(key) * gpu::kv_heads + head) * gpu::head_dim, key);
    cuda_check(cudaMemcpy(device.context_keys, host.staging, sizeof(device.context_keys), cudaMemcpyHostToDevice),
               "distinct context keys upload");

    std::fill_n(host.staging, gpu::block * gpu::kv_width, uint16_t(0));
    for (int row = 0; row < gpu::block; ++row)
        for (int head = 0; head < gpu::kv_heads; ++head)
            encode_key(host.staging + (row * gpu::kv_heads + head) * gpu::head_dim, gpu::capacity + row);
    cuda_check(cudaMemcpy(device.block_keys, host.staging, sizeof(device.block_keys), cudaMemcpyHostToDevice),
               "distinct anchor/future block keys upload");

    std::fill_n(host.staging, gpu::block * gpu::query_width, uint16_t(0));
    for (int row = 0; row < gpu::block; ++row)
        for (int head = 0; head < gpu::query_heads; ++head) {
            uint16_t* query = host.staging + (row * gpu::query_heads + head) * gpu::head_dim;
            query[0] = bf16_from_f32(1.0f);
            query[1] = bf16_from_f32(16.0f);
            query[2] = bf16_from_f32(256.0f);
        }
    cuda_check(cudaMemcpy(device.query, host.staging, sizeof(device.query), cudaMemcpyHostToDevice),
               "base-16 decoding queries upload");

    constexpr std::array<int, 2> origins{0, 37};
    constexpr std::array<int, 2> layers{0, 4};
    for (int case_index = 0; case_index < 2; ++case_index) {
        const int origin = origins[size_t(case_index)], layer = layers[size_t(case_index)];
        const int frontier = origin + gpu::capacity;
        // Untouched layers/slots and unwritten output are NaNs, not plausible zeros.
        cuda_check(cudaMemset(device.ring, 0xff, sizeof(device.ring)), "context ring poison");
        cuda_check(cudaMemset(device.scores, 0xff, sizeof(device.scores)), "attention score poison");
        cuda_check(cudaMemset(&device.status, 0, sizeof(device.status)), "attention status reset");
        kernel_check(gpu::store_context(device.context_keys, device.context_values, device.ring,
                                       layer, origin, gpu::capacity, &device.status, nullptr),
                     "store_context");
        kernel_check(gpu::attention_scores(device.query, device.block_keys, device.ring, device.scores,
                                          layer, origin, frontier, &device.status, nullptr),
                     "attention_scores");
        cuda_check(cudaMemcpy(host.scores, device.scores, sizeof(device.scores), cudaMemcpyDeviceToHost),
                   "attention scores readback");
        check_status(device, host, "attention status", case_index);
        for (int head = 0; head < gpu::query_heads; ++head)
            for (int row = 0; row < gpu::block; ++row)
                for (int key = 0; key < gpu::max_keys; ++key) {
                    const float actual = host.scores[(size_t(head) * gpu::block + row) * gpu::max_keys + key];
                    const int distance = (frontier + row) - (origin + key);
                    if (std::abs(distance) >= 2048) {
                        if (!std::isinf(actual) || !std::signbit(actual))
                            mismatch("attention masked score", case_index, head * gpu::block + row, key,
                                     actual, -std::numeric_limits<double>::infinity());
                    } else {
                        const double expected = double(key) / std::sqrt(128.0);
                        if (!std::isfinite(actual) || std::abs(double(actual) - expected) > attention_absolute_tolerance)
                            mismatch("attention visible score", case_index, head * gpu::block + row, key,
                                     actual, expected);
                    }
                }
        std::printf("PASS attention/distinct-keys-left-window-noncausal-anchor origin=%d layer=%d scores=%zu status=0\n",
                    origin, layer, score_elements);
    }
}

void convolution_edges(DevicePayload& device, HostPayload& host) {
    constexpr int stage = 1, offset = 1;
    std::fill_n(host.staging, 2 * 2 * gpu::hidden, uint16_t(0));
    std::fill_n(host.staging + (stage * 2 + offset) * gpu::hidden, gpu::hidden, bf16_from_f32(2.0f));
    cuda_check(cudaMemcpy(device.convolution_base, host.staging, sizeof(device.convolution_base), cudaMemcpyHostToDevice),
               "convolution base upload");
    std::fill_n(host.staging, gpu::block * gpu::dynamic_width, uint16_t(0));
    for (int row = 0; row < gpu::block; ++row)
        for (int group = 0; group < gpu::groups; ++group)
            host.staging[row * gpu::dynamic_width + (stage * 2 + offset) * gpu::groups + group] =
                bf16_from_f32(float(group % 3 + 1));
    cuda_check(cudaMemcpy(device.convolution_dynamic, host.staging, sizeof(device.convolution_dynamic),
                          cudaMemcpyHostToDevice), "convolution dynamic upload");

    constexpr std::array<int, 4> input_rows{1, 2, 4, 8};
    for (int block_index = 0; block_index < 2; ++block_index) {
        const int scale = block_index == 0 ? 16 : 1;
        for (int row = 0; row < gpu::block; ++row)
            std::fill_n(host.staging + row * gpu::hidden, gpu::hidden,
                        bf16_from_f32(float(scale * input_rows[size_t(row)])));
        cuda_check(cudaMemcpy(device.convolution_input, host.staging, sizeof(device.convolution_input),
                              cudaMemcpyHostToDevice), "convolution block upload");
        std::fill_n(host.staging, convolution_elements, bf16_from_f32(99.0f));
        cuda_check(cudaMemcpy(device.convolution_output, host.staging, sizeof(device.convolution_output),
                              cudaMemcpyHostToDevice), "convolution product/accumulator poison");
        cuda_check(cudaMemset(&device.status, 0, sizeof(device.status)), "convolution status reset");
        kernel_check(gpu::convolution_product(device.convolution_input, device.convolution_base,
                                             device.convolution_output, stage, offset, &device.status, nullptr),
                     "convolution_product");
        cuda_check(cudaMemcpy(host.staging, device.convolution_output, sizeof(device.convolution_output),
                              cudaMemcpyDeviceToHost), "convolution product readback");
        check_status(device, host, "convolution product status", block_index);
        for (int row = 0; row < gpu::block; ++row)
            for (int channel = 0; channel < gpu::hidden; ++channel) {
                const float shifted = row == 0 ? 0.0f : float(scale * input_rows[size_t(row - 1)]);
                const float actual = f32_from_bf16(host.staging[row * gpu::hidden + channel]);
                if (actual != 2.0f * shifted)
                    mismatch("convolution product", block_index, row, channel, actual, 2.0f * shifted);
            }
        // The first pass has replaced every poisoned element, including row zero.
        // The second pass adds only this block's shifted input; no history is supplied.
        kernel_check(gpu::convolution_addcmul(device.convolution_output, device.convolution_dynamic,
                                             device.convolution_input, stage, offset, &device.status, nullptr),
                     "convolution_addcmul");
        cuda_check(cudaMemcpy(host.staging, device.convolution_output, sizeof(device.convolution_output),
                              cudaMemcpyDeviceToHost), "convolution addcmul readback");
        check_status(device, host, "convolution addcmul status", block_index);
        for (int row = 0; row < gpu::block; ++row)
            for (int channel = 0; channel < gpu::hidden; ++channel) {
                const float shifted = row == 0 ? 0.0f : float(scale * input_rows[size_t(row - 1)]);
                const float expected = float(2 + (channel / 16) % 3 + 1) * shifted;
                const float actual = f32_from_bf16(host.staging[row * gpu::hidden + channel]);
                if (actual != expected)
                    mismatch("convolution addcmul", block_index, row, channel, actual, expected);
            }
        std::printf("PASS convolution/kernel-block-local-two-pass block=%d input-scale=%d stage=1 offset=1 channels=%d status=0\n",
                    block_index, scale, gpu::hidden);
    }
    // This is a real kernel boundary fixture, not whole-DeviceForward request isolation.
}

void select(DeviceSelector& selector, DevicePayload& device, HostPayload& host, float temperature,
            const std::array<float, 3>& uniforms) {
    std::string error;
    if (!selector.select(3, device.hidden, device.logits, 2, temperature, uniforms, host.selection, nullptr, error))
        throw std::runtime_error("DeviceSelector.select: " + error);
    require(!selector.poisoned(), "successful selector is unexpectedly poisoned");
    require(host.selection.sampled == (temperature > 0.0f), "selector sampled mode disagrees with temperature");
    // A successful DeviceSelector call also establishes its internal device status == 0.
}


void check_tie_selection(HostPayload& host, const std::array<int32_t, 16>& reference,
                         const std::array<int, 3>& chosen_slots) {
    for (int row = 0; row < gpu::selector_rows; ++row) {
        const int32_t expected_token = reference[size_t(chosen_slots[size_t(row)])];
        if (host.selection.tokens[size_t(row)] != expected_token)
            mismatch("pinned tie token", int(host.selection.sampled), row, 0,
                     host.selection.tokens[size_t(row)], expected_token);
        if (host.selection.confidence[size_t(row)] != 0.0625f)
            mismatch("pinned tie confidence", int(host.selection.sampled), row, 0,
                     host.selection.confidence[size_t(row)], 0.0625);
        for (int slot = 0; slot < gpu::selector_candidates; ++slot) {
            if (host.selection.candidates[size_t(row)][size_t(slot)] != reference[size_t(slot)])
                mismatch("pinned CPU top16 order", int(host.selection.sampled), row, slot,
                         host.selection.candidates[size_t(row)][size_t(slot)], reference[size_t(slot)]);
            if (host.selection.sampled && host.selection.conditional_probabilities[size_t(row)][size_t(slot)] != 0.0625f)
                mismatch("pinned tie conditional q", 1, row, slot,
                         host.selection.conditional_probabilities[size_t(row)][size_t(slot)], 0.0625);
        }
    }
}

void check_conditioned_selection(HostPayload& host) {
    constexpr std::array<int32_t, 3> selected_prefix{1, 0, 1}; // B, A, B, independent of GPU probabilities.
    for (int row = 0; row < gpu::selector_rows; ++row)
        if (host.selection.tokens[size_t(row)] != selected_prefix[size_t(row)])
            mismatch("predecessor-conditioned token", int(host.selection.sampled), row, 0,
                     host.selection.tokens[size_t(row)], selected_prefix[size_t(row)]);

    const double denominator = 1.0 + std::exp(-8.0) + 14.0 * std::exp(-25.0);
    const double expected_confidence = 1.0 / denominator;
    for (int row = 0; row < gpu::selector_rows; ++row) {
        const float confidence = host.selection.confidence[size_t(row)];
        if (!std::isfinite(confidence) || std::abs(double(confidence) - expected_confidence) > probability_absolute_tolerance)
            mismatch("predecessor-conditioned confidence", int(host.selection.sampled), row, 0,
                     confidence, expected_confidence);
        // Reconstruct from the independently asserted prefix, never from returned q.
        const int32_t predecessor = row == 0 ? 2 : selected_prefix[size_t(row - 1)];
        const double before = predecessor == 1 ? 4.0 : -4.0;
        bool found_a = false, found_b = false;
        double mass = 0.0;
        for (int slot = 0; slot < gpu::selector_candidates; ++slot) {
            const int32_t token = host.selection.candidates[size_t(row)][size_t(slot)];
            require(token >= 0 && token < gpu::selector_vocabulary, "conditioned candidate outside vocabulary");
            for (int previous = 0; previous < slot; ++previous)
                require(token != host.selection.candidates[size_t(row)][size_t(previous)], "duplicate conditioned candidate");
            if (slot < 2) require(token == 0 || token == 1, "conditioned top16 omitted a maximal raw logit");
            found_a = found_a || token == 0;
            found_b = found_b || token == 1;
            if (!host.selection.sampled) continue;
            const double unary = token == 0 || token == 1 ? 1.0 : -20.0;
            const double after = token == 0 ? 1.0 : token == 1 ? -1.0 : 0.0;
            const double expected_q = std::exp(unary + before * after - 5.0) / denominator;
            const float actual_q = host.selection.conditional_probabilities[size_t(row)][size_t(slot)];
            if (!std::isfinite(actual_q) || actual_q < 0.0f ||
                std::abs(double(actual_q) - expected_q) > probability_absolute_tolerance)
                mismatch("predecessor-conditioned q, not base head/marginal", 1, row, slot, actual_q, expected_q);
            mass += double(actual_q);
        }
        require(found_a && found_b, "conditioned top16 must contain both A and B");
        if (host.selection.sampled && std::abs(mass - 1.0) > probability_mass_tolerance)
            mismatch("predecessor-conditioned q mass", 1, row, 0, mass, 1.0);
    }
}

void selector_edges(DeviceSelector& selector, DevicePayload& device, HostPayload& host,
                    const std::array<int32_t, 16>& tie_reference) {
    std::fill_n(host.logits, logit_elements, -20.0f);
    for (int row = 0; row < gpu::selector_rows; ++row)
        std::fill_n(host.logits + size_t(row) * gpu::selector_vocabulary, 20, 1.0f);
    cuda_check(cudaMemcpy(device.logits, host.logits, sizeof(device.logits), cudaMemcpyHostToDevice),
               "tied selector logits upload");
    select(selector, device, host, 0.0f, {0.0f, 0.0f, 0.0f});
    check_tie_selection(host, tie_reference, {0, 0, 0});
    std::printf("PASS selector/pinned-CPU-top16-ties-greedy-pointmass status=0\n");
    select(selector, device, host, 1.0f, {0.0f, 0.25f, std::nextafter(1.0f, 0.0f)});
    check_tie_selection(host, tie_reference, {0, 4, 15});
    std::printf("PASS selector/pinned-ties-strict-CDF-zero-quarter-last status=0\n");

    // select() is synchronous: no borrowed weights are changed under a consumer.
    // Projection rank 0 and hidden coordinate 0 are exactly one, everything else zero.
    host.staging[0] = bf16_from_f32(1.0f);
    cuda_check(cudaMemcpy(device.projection, host.staging, sizeof(uint16_t), cudaMemcpyHostToDevice),
               "conditioned projection upload");
    std::fill_n(host.staging, gpu::selector_rows * gpu::selector_hidden, uint16_t(0));
    for (int row = 0; row < gpu::selector_rows; ++row)
        host.staging[row * gpu::selector_hidden] = bf16_from_f32(1.0f);
    cuda_check(cudaMemcpy(device.hidden, host.staging, sizeof(device.hidden), cudaMemcpyHostToDevice),
               "conditioned hidden upload");
    std::fill_n(host.staging, 3 * gpu::selector_rank, uint16_t(0));
    host.staging[0] = bf16_from_f32(-4.0f);                       // A
    host.staging[gpu::selector_rank] = bf16_from_f32(4.0f);      // B
    host.staging[2 * gpu::selector_rank] = bf16_from_f32(-4.0f);  // anchor
    cuda_check(cudaMemcpy(device.predecessor, host.staging, 3 * gpu::selector_rank * sizeof(uint16_t),
                          cudaMemcpyHostToDevice), "conditioned predecessor codebook upload");
    std::fill_n(host.staging, 2 * gpu::selector_rank, uint16_t(0));
    host.staging[0] = bf16_from_f32(1.0f);                       // A
    host.staging[gpu::selector_rank] = bf16_from_f32(-1.0f);     // B
    cuda_check(cudaMemcpy(device.successor, host.staging, 2 * gpu::selector_rank * sizeof(uint16_t),
                          cudaMemcpyHostToDevice), "conditioned successor codebook upload");
    std::fill_n(host.logits, logit_elements, -20.0f);
    for (int row = 0; row < gpu::selector_rows; ++row)
        std::fill_n(host.logits + size_t(row) * gpu::selector_vocabulary, 2, 1.0f);
    cuda_check(cudaMemcpy(device.logits, host.logits, sizeof(device.logits), cudaMemcpyHostToDevice),
               "conditioned selector logits upload");
    select(selector, device, host, 0.0f, {0.0f, 0.0f, 0.0f});
    check_conditioned_selection(host);
    std::printf("PASS selector/predecessor-conditioned-greedy-B-A-B-no-q status=0\n");
    select(selector, device, host, 1.0f, {0.5f, 0.5f, 0.5f});
    check_conditioned_selection(host);
    std::printf("PASS selector/predecessor-conditioned-sampled-B-A-B-q-not-base-head status=0\n");
}

void conditional_consumer_edges(DevicePayload& device, HostPayload& host, const ConsumerPlan& plan) {
    // Consume the actual synchronous DeviceSelector result through its public
    // production wire export. Expected values above never inspect this buffer
    // or copy probabilities/candidates back out of Selection.
    host.selection.fill_probability_rows(host.consumer_q.data());
    cuda_check(cudaMemcpy(device.consumer_q, host.consumer_q.data(), sizeof(device.consumer_q),
                          cudaMemcpyHostToDevice), "native conditional16 q wire upload");
    cuda_check(cudaMemcpy(device.consumer_drafts, host.selection.tokens.data(), sizeof(device.consumer_drafts),
                          cudaMemcpyHostToDevice), "native conditional16 proposals upload");
    for (size_t index = 0; index < plan.cases.size(); ++index) {
        const auto& test = plan.cases[index];
        // The selector has drained. Reuse its three full-vocabulary logit rows
        // for finite, independently specified target distributions.
        std::fill_n(host.logits, logit_elements, -40.0f);
        for (int row = 0; row < test.rows; ++row) {
            const auto& target = test.target[size_t(row)];
            for (int entry = 0; entry < target.count; ++entry)
                host.logits[size_t(row) * gpu::selector_vocabulary + size_t(target.ids[size_t(entry)])] =
                    target.logits[size_t(entry)];
        }
        cuda_check(cudaMemcpy(device.logits, host.logits, sizeof(device.logits), cudaMemcpyHostToDevice),
                   "independent target logits upload");
        cuda_check(cudaMemset(device.consumer_output, 0xff, sizeof(device.consumer_output)),
                   "conditional16 consumer output poison");
        strata::kernels::SamplerParams parameters;
        parameters.top_k = test.target[0].count;
        parameters.top_p = 1.0f;
        parameters.min_p = 0.0f;
        parameters.temperature = 1.0f;
        parameters.seed = test.seed;
        parameters.counter = consumer_counter;
        require(strata::kernels::sample_tokens_spec(device.logits, test.rows, gpu::selector_vocabulary,
                    nullptr, 0, parameters, device.consumer_drafts, device.consumer_q, test.drafts,
                    test.drafts, device.consumer_output, nullptr),
                "production conditional16 rejection sampler refused; ordinary fallback is forbidden");
        cuda_check(cudaStreamSynchronize(nullptr), "conditional16 consumer completion");
        cuda_check(cudaMemcpy(host.consumer_output.data(), device.consumer_output, size_t(test.rows) * sizeof(int),
                              cudaMemcpyDeviceToHost), "conditional16 consumer output readback");
        for (int row = 0; row < test.rows; ++row)
            if (host.consumer_output[size_t(row)] != test.expected[size_t(row)])
                mismatch(test.name, int(index), row, 0, host.consumer_output[size_t(row)], test.expected[size_t(row)]);
        int prefix = 0, expected_prefix = 0;
        while (prefix < test.drafts && host.consumer_output[size_t(prefix)] == known_drafts[size_t(prefix)]) ++prefix;
        while (expected_prefix < test.drafts && test.accepted[size_t(expected_prefix)]) ++expected_prefix;
        if (prefix != expected_prefix)
            mismatch("conditional16 consumer accepted prefix", int(index), 0, 0, prefix, expected_prefix);
        std::printf("PASS consumer/native-conditional16/%s seed=%llu counter=%llu rows=%d drafts=%d q=%d prefix=%d",
                    test.name, static_cast<unsigned long long>(test.seed),
                    static_cast<unsigned long long>(consumer_counter), test.rows, test.drafts, test.drafts, prefix);
        for (int row = 0; row < test.rows; ++row)
            std::printf(" row%d=%s:%d", row, row < test.drafts ?
                (test.accepted[size_t(row)] ? "accept" : "reject-residual") : "bonus", host.consumer_output[size_t(row)]);
        std::printf("\n");
    }
}

void run(const Options& options, const std::array<int32_t, 16>& tie_reference, const ConsumerPlan& consumer) {
    const size_t sampler_scratch = 16 * strata::kernels::coupled_draft_scratch_bytes(gpu::selector_vocabulary);
    require(sampler_scratch != 0 && sampler_scratch <= sampler_scratch_bound,
            "production sampler scratch exceeds the declared fixture bound");
    cuda_check(cudaSetDevice(options.device), "select requested device");
    cudaDeviceProp properties{};
    cuda_check(cudaGetDeviceProperties(&properties, options.device), "requested device properties");
    require(properties.major >= 7, "DFlash2 fixture requires SM70 or newer");
    std::printf("device=%d name=%s cc=%d.%d\n", options.device, properties.name, properties.major, properties.minor);
    std::printf("tolerances attention_abs=%.9g conditional_q_abs=%.9g q_mass_abs=%.9g\n",
                attention_absolute_tolerance, probability_absolute_tolerance, probability_mass_tolerance);

    auto host = std::make_unique_for_overwrite<HostPayload>();
    DeviceBuffer buffer;
    DevicePayload& device = *buffer.get();
    // Borrowed inputs/weights outlive selector destruction, including every exception path.
    DeviceSelector selector(options.device, device.projection, device.predecessor, device.successor);
    DrainBeforeOwners drain;
    std::printf("declared_peak device_payload=%zu selector_device_scratch=%zu sampler_device_scratch=%zu device_total=%zu "
                "host_payload=%zu selector_pinned=%zu consumer_reference=%zu host_total=%zu bytes "
                "(sizeof/requested payloads, not measured residency; excludes CUDA runtime/driver and small owners/diagnostics)\n",
                sizeof(DevicePayload), selector.device_bytes(), sampler_scratch,
                sizeof(DevicePayload) + selector.device_bytes() + sampler_scratch,
                sizeof(HostPayload), DeviceSelector::pinned_bytes(), sizeof(ConsumerPlan),
                sizeof(HostPayload) + DeviceSelector::pinned_bytes() + sizeof(ConsumerPlan));
    // Initializes values and all selector weights/hidden once, without host codebook copies.
    cuda_check(cudaMemset(device.context_values, 0, sizeof(device.context_values)), "context values initialization");
    cuda_check(cudaMemset(device.projection, 0, sizeof(device.projection)), "selector projection initialization");
    cuda_check(cudaMemset(device.predecessor, 0, sizeof(device.predecessor)), "selector predecessor initialization");
    cuda_check(cudaMemset(device.successor, 0, sizeof(device.successor)), "selector successor initialization");
    cuda_check(cudaMemset(device.hidden, 0, sizeof(device.hidden)), "selector hidden initialization");
    attention_edges(device, *host);
    convolution_edges(device, *host);
    selector_edges(selector, device, *host, tie_reference);
    conditional_consumer_edges(device, *host, consumer);
    std::string error;
    if (!selector.drain(error)) throw std::runtime_error("DeviceSelector.drain: " + error);
}
} // namespace

int main(int argc, char** argv) {
    // Help is recognized before argument validation, file I/O, and every CUDA call.
    for (int i = 1; i < argc; ++i)
        if (std::string_view(argv[i]) == "--help") {
            usage(argv[0]);
            return 0;
        }
    try {
        const Options options = parse_options(argc, argv);
        const auto tie_reference = read_tie_reference(options.tie_reference);
        ConsumerPlan consumer;
        prepare_consumer_plan(consumer); // CPU-only, before any device selection or GPU observation.
        run(options, tie_reference, consumer);
        std::printf("PASS dflash2_edge_test: synthetic GPU kernel/selector/conditional-q consumer boundaries only; no real-model or performance claim\n");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL dflash2_edge_test: %s\n", error.what());
        return 1;
    }
}
