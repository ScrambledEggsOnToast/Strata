// Run only in a guarded measurement window. Tests the actual mapped publication kernel.
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/core/dflash2_forward.hpp"
#include "strata/core/verify.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <bit>
#include <charconv>
#include <cstdlib>
#include <memory>
#include <string>
#include <string_view>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

void ck(cudaError_t status) { if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status)); }

// The existing verifier friend constructs only bounded capture storage, not a
// model or a fabricated target result. Public arm/view/release, the real reducer,
// real CUDA copies, producer certification and the production gather are tested.
namespace strata::program {
struct ResidentExecutorFixture {
    static void features(std::span<const int> devices) {
        namespace d = core::dflash2;
        std::string error;
        auto require = [&](bool ok, const char* why) {
            if (!ok) throw std::runtime_error(std::string(why) + ": " + error);
        };
        require(!devices.empty() && devices.size() <= 3, "fixture admits one to three explicit devices");
        const int destination = devices.back();
        require(d::DeviceForward::prepare_feature_peers(devices, destination, error), "prepare actual peer topology");
        constexpr int physical_rows = 7, capacity = d::max_verify_rows;
        constexpr size_t source_values = d::tap_layers.size() * capacity * d::residual_width;
        struct Storage {
            core::ModelGeometry geometry;
            std::array<core::SessionState, 2> sessions;
            std::array<std::unique_ptr<core::Verifier>, 3> verifiers;
            std::array<float*, 3> raw{};
            std::array<cudaGraphExec_t, 3> graphs{};
            uint16_t* output = nullptr;
            int destination = -1;
            ~Storage() {
                int previous = -1;
                if (cudaGetDevice(&previous) != cudaSuccess) std::terminate();
                for (size_t stage = 0; stage < verifiers.size(); ++stage) {
                    if (!verifiers[stage]) continue;
                    if (cudaSetDevice(verifiers[stage]->device_) != cudaSuccess ||
                        cudaStreamSynchronize(verifiers[stage]->cs_) != cudaSuccess) std::terminate();
                    if (graphs[stage] && cudaGraphExecDestroy(graphs[stage]) != cudaSuccess) std::terminate();
                    if (raw[stage] && cudaFree(raw[stage]) != cudaSuccess) std::terminate();
                }
                if (output && (cudaSetDevice(destination) != cudaSuccess || cudaFree(output) != cudaSuccess))
                    std::terminate();
                if (cudaSetDevice(previous) != cudaSuccess) std::terminate();
            }
        } storage;
        storage.destination = destination;
        for (auto& session : storage.sessions) session.max_cells = d::max_position;
        for (size_t stage = 0; stage < devices.size(); ++stage) {
            ck(cudaSetDevice(devices[stage]));
            auto& verifier = storage.verifiers[stage] = std::make_unique<core::Verifier>();
            require(verifier->enable_dflash2_capture(error), "enable real CUDA capture contract");
            verifier->device_ = devices[stage]; verifier->max_t_ = capacity;
            verifier->g_ = &storage.geometry; verifier->ss_ = &storage.sessions[0];
            verifier->slots_ = {&storage.sessions[0], &storage.sessions[1]};
            verifier->dflash2_raw_rows_ = capacity;
            // This bounded synthetic fixture does not claim target phase timing.
            verifier->dflash2_phases_ = false;
            const int lower = stage == 0 ? 0 : stage == 1 ? 20 : 39;
            const int upper = stage + 1 == devices.size() ? 48 : stage == 0 ? 20 : 39;
            require(verifier->dflash2_capture_.configure(d::CapturePhase::verify, capacity,
                    lower, upper, 2560, 4, 48, error), "configure physical tap ownership");
            for (auto& seat : verifier->dflash2_seats_)
                require(seat.state.configure(d::CapturePhase::verify, capacity,
                    lower, upper, 2560, 4, 48, error), "configure bounded resident capture states");
            for (int slot = 0; slot < 2; ++slot) {
                verifier->slot_request_id_[slot] = verifier->validated_slot_id_[slot] = uint64_t(7 + slot);
                verifier->slot_incarnation_[size_t(slot)] = 9;
            }
            ck(cudaStreamCreateWithFlags(&verifier->cs_, cudaStreamNonBlocking));
            ck(cudaHostAlloc(reinterpret_cast<void**>(&verifier->dflash2_on_h_), sizeof(uint32_t),
                             cudaHostAllocMapped | cudaHostAllocPortable));
            ck(cudaHostGetDevicePointer(reinterpret_cast<void**>(&verifier->dflash2_on_m_),
                                        verifier->dflash2_on_h_, 0));
            *verifier->dflash2_on_h_ = 0;
            for (size_t tap = 0; tap < d::tap_layers.size(); ++tap) {
                if (!verifier->dflash2_capture_.owns_tap(int(tap))) continue;
                const size_t bytes = size_t(capacity) * d::reduced_tap_row_bytes;
                ck(cudaHostAlloc(reinterpret_cast<void**>(&verifier->dflash2_h_[tap]), bytes,
                                 cudaHostAllocMapped | cudaHostAllocPortable));
                ck(cudaHostGetDevicePointer(reinterpret_cast<void**>(&verifier->dflash2_m_[tap]),
                                            verifier->dflash2_h_[tap], 0));
                ck(cudaMalloc(reinterpret_cast<void**>(&verifier->dflash2_device_[tap]), bytes));
            }
            ck(cudaMalloc(reinterpret_cast<void**>(&storage.raw[stage]), source_values * sizeof(float)));
        }
        ck(cudaSetDevice(destination));
        constexpr size_t output_values = size_t(capacity) * d::feature_width + 16;
        ck(cudaMalloc(reinterpret_cast<void**>(&storage.output), output_values * sizeof(uint16_t)));
        std::vector<uint16_t> tentative(size_t(d::context_rows) * d::feature_width);
        core::OperationLease common;
        std::array<std::unique_ptr<d::CaptureProducer>, 2> producers;
        for (int slot = 0; slot < 2; ++slot) {
            producers[size_t(slot)] = std::make_unique<d::CaptureProducer>(tentative, common);
            require(producers[size_t(slot)]->reset(uint64_t(7 + slot), 9, slot == 0 ? 0 : 71, error),
                    "bind independent actual producer rings");
        }
        {
            // Prime the actual host ring so the next accepted two-row verifier
            // interval crosses slot 2046 -> 0. The device gather stays linear.
            core::OperationLease::Guard lease(common);
            auto& producer = *producers[0];
            constexpr int retained = d::context_rows - 1;
            const std::vector<uint16_t> compact(size_t(retained) * d::hidden_size, uint16_t(0x3f80));
            require(bool(lease) && producer.begin(d::CapturePhase::prefill, 0, retained, error, &lease),
                    "prime the real committed feature ring immediately before wrap");
            for (int tap : d::tap_layers) {
                d::TapView view{0, retained, retained, tap, tap, d::CapturePhase::prefill,
                    d::CapturePoint::completed_second_residual_write, {}, producer.owner(), compact};
                require(producer.consume(view, error, &lease), "prime all five actual retained host tap slices");
            }
            require(producer.publish(error), "publish the bounded prefill that positions the wrap");
            lease.complete();
        }
        std::vector<float> input(source_values);
        std::vector<uint16_t> output(output_values);
        uint64_t total_local = 0, total_peer = 0, total_host = 0;
        auto value = [](int position, int tap) { return float(position % 97 + tap * 128 + 3); };
        for (int frame = 0; frame < 2; ++frame) {
            std::array<d::BatchCaptureRow, physical_rows> rows{};
            for (int slot = 0; slot < 2; ++slot) {
                auto& producer = *producers[size_t(slot)];
                const int count = slot == 0 ? 3 : 4;
                const int first = slot == 0 ? 0 : 3;
                require(producer.begin(d::CapturePhase::verify, producer.frontier(), count, error),
                        "begin actual contiguous owner capture generation");
                for (int row = 0; row < count; ++row)
                    rows[size_t(first + row)] = {slot, producer.owner(), producer.frontier() + row};
            }
            for (size_t stage = 0; stage < devices.size(); ++stage) {
                auto& verifier = *storage.verifiers[stage];
                ck(cudaSetDevice(devices[stage]));
                auto malformed = rows; ++malformed[4].position;
                require(!verifier.arm_dflash2_batch(malformed, error) &&
                        verifier.dflash2_batch_phase_ == core::Verifier::Dflash2BatchPhase::idle,
                        "nonconsecutive packed owner is refused before live mutation");
                require(verifier.arm_dflash2_batch(rows, error), "arm both packed owner intervals");
                d::TapView premature;
                const int tap = verifier.dflash2_capture_.owns_tap(0) ? 5 :
                    verifier.dflash2_capture_.owns_tap(2) ? 25 : 45;
                require(!verifier.dflash2_batch_tap(producers[0]->owner(), 0, tap, premature, error),
                        "armed device tap is not a completed target prefix");
                for (size_t index = 0; index < d::tap_layers.size(); ++index)
                    for (int row = 0; row < capacity; ++row)
                        for (int stream = 0; stream < 4; ++stream) {
                            const int slot = row < 3 ? 0 : 1;
                            const int relative = slot == 0 ? row : row - 3;
                            const bool accepted = row < physical_rows && relative < 2;
                            const float number = accepted
                                ? value(producers[size_t(slot)]->frontier() + relative, int(index)) - 3.f + 2.f * stream
                                : std::numeric_limits<float>::quiet_NaN();
                            std::fill_n(input.data() + (index * capacity + size_t(row)) * d::residual_width +
                                size_t(stream) * d::hidden_size, d::hidden_size, number);
                        }
                ck(cudaMemcpy(storage.raw[stage], input.data(), source_values * sizeof(float), cudaMemcpyHostToDevice));
                require(verifier.begin_dflash2_batch(error), "real stage marks its packed capture in flight");
                if (!storage.graphs[stage]) {
                    cudaGraph_t graph = nullptr;
                    ck(cudaStreamBeginCapture(verifier.cs_, cudaStreamCaptureModeGlobal));
                    for (size_t index = 0; index < d::tap_layers.size(); ++index) {
                        if (!verifier.dflash2_capture_.owns_tap(int(index))) continue;
                        strata::kernels::tap_reduce_publish(storage.raw[stage] + index * capacity * d::residual_width,
                            verifier.dflash2_device_[index], physical_rows, d::hidden_size, verifier.dflash2_on_m_, verifier.cs_);
                        ck(cudaMemcpyAsync(verifier.dflash2_h_[index], verifier.dflash2_device_[index],
                            size_t(physical_rows) * d::reduced_tap_row_bytes, cudaMemcpyDeviceToHost, verifier.cs_));
                    }
                    ck(cudaStreamEndCapture(verifier.cs_, &graph));
                    ck(cudaGraphInstantiate(&storage.graphs[stage], graph, nullptr, nullptr, 0));
                    ck(cudaGraphDestroy(graph));
                }
                ck(cudaGraphLaunch(storage.graphs[stage], verifier.cs_));
                ck(cudaStreamSynchronize(verifier.cs_));
                require(verifier.finish_dflash2_batch(error), "actual reduced payload/mirror completes before target acknowledgement");
                for (int slot = 0; slot < 2; ++slot)
                    require(verifier.dflash2_seats_[size_t(slot)].state.committed(2, error) &&
                            verifier.dflash2_seats_[size_t(slot)].state.waited(error),
                            "synthetic target acknowledges its accepted prefix only after CUDA completion");
                verifier.dflash2_batch_phase_ = core::Verifier::Dflash2BatchPhase::ready;
            }
            for (int slot = 0; slot < 2; ++slot) {
                auto& producer = *producers[size_t(slot)];
                if (frame == 0 && slot == 1) {
                    require(producer.cancel(error) && producer.frontier() == 71 && !producer.store().poisoned(),
                            "cancel a completed target owner before any feature consumer, without publishing rows");
                } else {
                    core::OperationLease::Guard lease(common);
                    require(bool(lease), "serialize shared feature staging with the actual common lease");
                    for (size_t tap = 0; tap < d::tap_layers.size(); ++tap) {
                        size_t stage = 0;
                        while (!storage.verifiers[stage]->dflash2_capture_.owns_tap(int(tap))) ++stage;
                        d::TapView view;
                        require(storage.verifiers[stage]->dflash2_batch_tap(producer.owner(), slot,
                            d::tap_layers[tap], view, error) && view.device.present(), "borrow authenticated real packed tap");
                        auto stale = view; ++stale.owner.generation;
                        require(!producer.consume(stale, error, &lease) && !producer.store().poisoned(),
                                "foreign completion cannot poison the active producer");
                        if (frame == 0 && slot == 0 && tap == 0) {
                            d::CaptureProducer forged(tentative, common);
                            require(forged.reset(7, 9, producer.frontier(), error), "prepare mirror-substitution refusal");
                            while (forged.owner().generation + 1 < view.owner.generation)
                                require(forged.begin(d::CapturePhase::verify, producer.frontier(), 3, error) &&
                                        forged.cancel(error), "align a real cancelled producer generation");
                            require(forged.begin(d::CapturePhase::verify, producer.frontier(), 3, error),
                                    "bind the same actual device capability generation");
                            auto substituted = view;
                            const std::vector<uint16_t> alien(view.reduced.begin(), view.reduced.end());
                            substituted.reduced = alien;
                            require(!forged.consume(substituted, error, &lease) && forged.store().poisoned(),
                                    "a copied device capability cannot authenticate an arbitrary host mirror");
                        }
                        require(producer.consume(view, error, &lease), "interleave only accepted compact tap rows");
                    }
                    require(producer.publish(error), "all five real capabilities join the committed producer generation");
                    const int start = producer.frontier() - 2;
                    const auto features = producer.store().committed_since(start);
                    if (frame == 0 && slot == 0)
                        require(features.segments[0].size() == d::feature_width &&
                                features.segments[1].size() == d::feature_width,
                                "authenticated fresh accepted interval crosses the real committed-ring wrap");
                    auto foreign = features; foreign.segments[0] = std::span<const uint16_t>{tentative.data(), features.segments[0].size()};
                    require(!producer.device_preflight(foreign, lease, error), "arbitrary feature-ring pointers cannot skip upload");
                    ck(cudaSetDevice(destination));
                    std::fill(output.begin(), output.end(), uint16_t(0x55aa));
                    ck(cudaMemcpy(storage.output, output.data(), output.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
                    uint64_t local = 0, peer = 0, host = 0;
                    if (frame == 1 && slot == 1) {
                        struct RefusePartial final : d::ForwardTraceSink {
                            bool capture(const d::ForwardTrace&, void*, std::string& why) override {
                                why = "fixture refuses after a real first-tap transfer"; return false;
                            }
                        } refusal;
                        require(!d::DeviceForward::gather_features(producer, features, lease, storage.output, destination,
                            storage.verifiers[devices.size() - 1]->cs_, local, peer, host, error, &refusal) &&
                            producer.store().poisoned() && local + peer + host == size_t(features.count) * d::reduced_tap_row_bytes,
                            "partial transfer failure drains before poisoning producer and common lease");
                    } else {
                        require(d::DeviceForward::gather_features(producer, features, lease, storage.output, destination,
                            storage.verifiers[devices.size() - 1]->cs_, local, peer, host, error), "gather accepted owner device features");
                        ck(cudaStreamSynchronize(storage.verifiers[devices.size() - 1]->cs_));
                        ck(cudaMemcpy(output.data(), storage.output, output.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost));
                        require(local + peer + host == size_t(features.count) * d::feature_row_bytes,
                                "every accepted byte has one honest local/peer/host route");
                        uint64_t expected_local = 0, expected_peer = 0, expected_host = 0;
                        for (size_t stage = 0; stage < devices.size(); ++stage) {
                            size_t taps = 0;
                            for (int tap = 0; tap < int(d::tap_layers.size()); ++tap)
                                taps += storage.verifiers[stage]->dflash2_capture_.owns_tap(tap);
                            const auto bytes = uint64_t(features.count) * taps * d::reduced_tap_row_bytes;
                            if (devices[stage] == destination) expected_local += bytes;
                            else {
                                int access = 0;
                                ck(cudaDeviceCanAccessPeer(&access, destination, devices[stage]));
                                if (access) expected_peer += bytes; else expected_host += bytes;
                            }
                        }
                        require(local == expected_local && peer == expected_peer && host == expected_host,
                                "unsupported peers are honestly counted compact-host uploads, never GPU-resident claims");
                        for (size_t at = 0; at < output.size(); ++at) {
                            const bool kept = at < size_t(features.count) * d::feature_width;
                            const auto expected = kept ? strata::kernels::bf16_from_f32(
                                value(start + int(at / d::feature_width), int(at % d::feature_width / d::hidden_size)))
                                : uint16_t(0x55aa);
                            if (output[at] != expected)
                                std::fprintf(stderr, "DFLASH2_DEVICE_FEATURE_MISMATCH frame=%d slot=%d index=%zu "
                                    "actual=0x%04x expected=0x%04x\n", frame, slot, at,
                                    unsigned(output[at]), unsigned(expected));
                            require(output[at] == expected, "packed first-row offset, exact arithmetic, and rejected/sentinel suffix");
                        }
                        total_local += local; total_peer += peer; total_host += host;
                        lease.complete();
                    }
                }
                for (size_t stage = 0; stage < devices.size(); ++stage) {
                    require(!storage.verifiers[stage]->release_dflash2_batch(
                        {producer.owner().request, 9, producer.owner().generation + 1}, slot, error),
                        "a stale release cannot recycle physical device staging");
                    require(storage.verifiers[stage]->release_dflash2_batch(producer.owner(), slot, error),
                            "release only the drained accepted/cancelled/failed owner slice");
                }
                if (slot == 0) {
                    core::OperationLease::Guard lease(common);
                    const auto previous = producer.store().committed_since(producer.frontier() - 2);
                    require(bool(lease) && !producer.device_preflight(previous, lease, error),
                            "released device capability cannot authorize a later gather");
                    lease.complete();
                }
            }
            const auto peer_ring = producers[0]->store().committed_since(producers[0]->frontier() - 1);
            size_t offset = 0;
            for (const auto part : peer_ring.segments)
                for (const uint16_t bits : part) {
                    const auto expected = strata::kernels::bf16_from_f32(value(
                        producers[0]->frontier() - 1, int(offset % d::feature_width / d::hidden_size)));
                    require(bits == expected, "cancelled or failed owner cannot change the independent committed feature ring");
                    ++offset;
                }
        }
        core::OperationLease::Guard refused(common, std::try_to_lock);
        require(!refused && refused.failure() == core::OperationLease::Failure::poisoned,
                "a drained partial transfer never repairs its common operation lease");
        std::printf("PASS device-features devices=%zu frames=2 packed_first_row=3 rejected_nonfinite=1 "
            "cancelled_owner=1 stale_capability=1 foreign_mirror=1 foreign_ring=1 wrap=1 partial_transfer_poison=1 "
            "local_device_bytes=%llu peer_device_bytes=%llu host_upload_bytes=%llu target_inference=0\n",
            devices.size(), (unsigned long long)total_local, (unsigned long long)total_peer, (unsigned long long)total_host);
    }
};
} // namespace strata::program

int main(int argc, char** argv) {
    try {
        std::string flags_error;
        if (!strata::core::dflash2::validate_feature_flags(flags_error)) throw std::runtime_error(flags_error);
        std::vector<int> devices;
        if (argc != 1 && (argc != 3 || std::string_view(argv[1]) != "--feature-devices"))
            throw std::runtime_error("usage: dflash2_capture_reduce_test [--feature-devices 0,1,2]");
        if (argc == 3) {
            std::string_view remaining(argv[2]);
            while (!remaining.empty()) {
                const size_t comma = remaining.find(',');
                const auto part = remaining.substr(0, comma);
                int ordinal = -1;
                const auto parsed = std::from_chars(part.data(), part.data() + part.size(), ordinal);
                if (parsed.ec != std::errc{} || parsed.ptr != part.data() + part.size() || ordinal < 0)
                    throw std::runtime_error("feature devices require explicit nonnegative CUDA ordinals");
                devices.push_back(ordinal);
                if (comma == std::string_view::npos) break;
                remaining.remove_prefix(comma + 1);
                if (remaining.empty()) throw std::runtime_error("empty trailing feature device");
            }
        }
        if (strata::core::dflash2::device_features_enabled()) {
            if (devices.empty()) { int current = -1; ck(cudaGetDevice(&current)); devices.push_back(current); }
            strata::program::ResidentExecutorFixture::features(devices);
        } else if (!devices.empty()) throw std::runtime_error("--feature-devices requires STRATA_DFLASH2_DEVICE_FEATURES=1");
        constexpr int hidden = 2560, capacity = 2047;
        constexpr size_t n = size_t(capacity) * hidden;
        float* source;
        uint16_t *host, *mapped;
        uint32_t *enabled, *device_enabled;
        cudaStream_t stream;
        ck(cudaStreamCreate(&stream));
        ck(cudaMalloc(&source, n * 4 * sizeof(float)));
        ck(cudaHostAlloc(&host, (n + 16) * sizeof(uint16_t), cudaHostAllocMapped));
        ck(cudaHostGetDevicePointer(&mapped, host, 0));
        ck(cudaHostAlloc(&enabled, sizeof(uint32_t), cudaHostAllocMapped));
        ck(cudaHostGetDevicePointer(&device_enabled, enabled, 0));
        std::vector<float> input(n * 4);
        size_t checked = 0;
        for (int rows : {1, 3, 8, 257, capacity}) {
            for (int row = 0; row < rows; ++row) for (int col = 0; col < hidden; ++col) {
                float a = float(row + col % 31) / 16, b = -a / 2, c = 2, d = -1;
                switch (col % 17) {
                case 0: a = 16777216; b = 1; c = -16777216; d = 4; break;
                case 1: a = b = c = d = 1.00390625f; break;
                case 2: a = b = c = d = 1.01171875f; break;
                case 3: a = b = c = d = -0.0f; break;
                case 4: a = b = c = d = std::bit_cast<float>(uint32_t(0x00008001)); break;
                case 5: a = b = std::numeric_limits<float>::max(); break;
                case 6: a = std::numeric_limits<float>::quiet_NaN(); break;
                case 7: a = std::numeric_limits<float>::infinity(); break;
                }
                size_t i = size_t(row) * 4 * hidden + col;
                input[i] = a; input[i + hidden] = b; input[i + 2 * hidden] = c; input[i + 3 * hidden] = d;
            }
            ck(cudaMemcpy(source, input.data(), size_t(rows) * hidden * 4 * sizeof(float), cudaMemcpyHostToDevice));
            cudaGraph_t graph;
            cudaGraphExec_t executable;
            ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
            strata::kernels::tap_reduce_publish(source, mapped + 8, rows, hidden, device_enabled, stream);
            ck(cudaStreamEndCapture(stream, &graph));
            ck(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
            for (int replay = 0; replay < 3; ++replay) {
                std::fill_n(host, n + 16, uint16_t(0x55aa));
                *enabled = replay != 0;
                if (replay == 2) {
                    input[0] = 4; input[hidden] = 4; input[2 * hidden] = 4; input[3 * hidden] = 4;
                    ck(cudaMemcpy(source, input.data(), size_t(rows) * hidden * 4 * sizeof(float), cudaMemcpyHostToDevice));
                }
                ck(cudaGraphLaunch(executable, stream));
                ck(cudaStreamSynchronize(stream));
                for (size_t i = 0; i < n + 16; ++i) {
                    uint16_t expected = 0x55aa;
                    if (replay && i >= 8 && i < 8 + size_t(rows) * hidden) {
                        size_t j = i - 8, base = (j / hidden) * 4 * hidden + j % hidden;
                        volatile float ab = input[base] + input[base + hidden];
                        volatile float abc = ab + input[base + 2 * hidden];
                        volatile float abcd = abc + input[base + 3 * hidden];
                        expected = strata::kernels::bf16_from_f32(abcd * 0.25f);
                        // NaN payload is immaterial: both paths must preserve nonfinite rejection.
                        if ((expected & 0x7f80) == 0x7f80 && (host[i] & 0x7f80) == 0x7f80) continue;
                    }
                    if (host[i] != expected) throw std::runtime_error("mapped reduction/replay/extent mismatch");
                    ++checked;
                }
            }
            ck(cudaGraphExecDestroy(executable)); ck(cudaGraphDestroy(graph));
        }
        // Two adjacent chunk publications with distinct source/destination offsets.
        std::fill_n(host, n + 16, uint16_t(0x55aa));
        *enabled = 1;
        strata::kernels::tap_reduce_publish(source + 4 * hidden, mapped + 8 + hidden, 1, hidden, device_enabled, stream);
        strata::kernels::tap_reduce_publish(source + 8 * hidden, mapped + 8 + 2 * hidden, 1, hidden, device_enabled, stream);
        ck(cudaStreamSynchronize(stream));
        for (size_t i = 0; i < n + 16; ++i) {
            uint16_t expected = 0x55aa;
            if (i >= 8 + hidden && i < 8 + 3 * hidden) {
                size_t j = i - 8, base = (j / hidden) * 4 * hidden + j % hidden;
                volatile float ab = input[base] + input[base + hidden];
                volatile float abc = ab + input[base + 2 * hidden];
                volatile float abcd = abc + input[base + 3 * hidden];
                expected = strata::kernels::bf16_from_f32(abcd * 0.25f);
                if ((expected & 0x7f80) == 0x7f80 && (host[i] & 0x7f80) == 0x7f80) continue;
            }
            if (host[i] != expected) throw std::runtime_error("chunk source/destination offset mismatch");
            ++checked;
        }
        ck(cudaFree(source)); ck(cudaFreeHost(host)); ck(cudaFreeHost(enabled)); ck(cudaStreamDestroy(stream));
        std::printf("capture reduction: %zu exact comparisons passed, including disabled/repeated graph replay\n", checked);
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
