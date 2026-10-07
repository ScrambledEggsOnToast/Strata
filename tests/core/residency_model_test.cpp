// Protected real-model operator regression, not a pruned-model generation result.
// Reuses the admitted expert cache and full resident sessions. Each Verifier lives
// alone and executes the production layer [0,1) stage, with real CPU/GPU experts.
#include "strata/core/expert_cache.hpp"
#include "strata/core/verify.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/program/session_fingerprint.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
using namespace strata;

struct PoolProbe {
    core::PoolMultiFn forward = nullptr;
    void* user = nullptr;
    core::ExpertDispatch* dispatch = nullptr;
    int64_t width = 0;
    int missing = -1;
    int layers = 0, completions = 0, missed_rows = 0;
    std::array<int32_t, 20> routed{};
    std::string error;

    static bool call(void* user, const float* x, const int32_t* ids, int64_t count, int64_t k,
                     float* out, int64_t layer, const int64_t* positions, const int* slots,
                     int group, int lane) {
        auto& probe = *static_cast<PoolProbe*>(user);
        if (layer >= 0) {
            if (layer != 0 || probe.layers != 0 || count != 2 || k != 10 || !x || !ids || !out ||
                !positions || !slots || slots[0] != 1 || slots[1] != 0 || group != 0 || lane != 0) {
                probe.error = "unexpected real pool layer/row contract";
                return false;
            }
            ++probe.layers;
            for (int i = 0; i < 20; ++i) {
                if (ids[i] < 0 || ids[i] >= probe.width) {
                    probe.error = "production router selected an invalid expert";
                    return false;
                }
                probe.routed[i] = ids[i];
                if (ids[i] == probe.missing) ++probe.missed_rows;
            }
        } else if (layer == -1) {
            if (count != 0 || group != -1 || ++probe.completions != 1) {
                probe.error = "unexpected final consumer acknowledgement";
                return false;
            }
            if (probe.layers && !probe.dispatch->completion[0].active()) {
                probe.error = "missing actual expert-consumer ownership before acknowledgement";
                return false;
            }
        }
        // No replacement plan or output: instrumentation always forwards to generate's
        // real adapter, including its graph-consumer completion and poisoned-drain calls.
        return probe.forward(probe.user, x, ids, count, k, out, layer, positions, slots, group, lane);
    }
};

struct Handoff {
    float* host = nullptr;
    float* device = nullptr;
    ~Handoff() { if (host) cudaFreeHost(host); }
};

struct Observation {
    std::array<program::SessionFingerprint, 2> state;
    std::vector<float> handoff;
};

bool same_committed(const program::SessionFingerprint& a, const program::SessionFingerprint& b) {
    return a.gdn == b.gdn && a.ple == b.ple && a.tail == b.tail && a.pooled == b.pooled &&
           a.kv == b.kv && a.draft == b.draft && a.dead == b.dead && a.pooled_full == b.pooled_full &&
           a.residual == b.residual;
}
} // namespace

int residency_model_checks(const strata::core::WeightTable& weights, const strata::core::ModelGeometry& geometry,
                           strata::core::SessionState& main, strata::core::SessionState& a,
                           strata::core::SessionState& b, strata::core::ExpertCache& cache,
                           std::vector<int32_t>& residency, int32_t* device_residency,
                           strata::core::ExpertDispatch& dispatch, strata::core::PoolMultiFn pool, void* user) {
    using namespace strata;
    std::string error;
    int checks = 0;
    auto require = [&](bool ok, const char* message) {
        ++checks;
        if (!ok) throw std::runtime_error(std::string(message) + ": " + error);
    };
    auto cuda = [&](cudaError_t status, const char* message) {
        if (status != cudaSuccess) error = cudaGetErrorString(status);
        require(status == cudaSuccess, message);
    };
    try {
        const char* all_resident = std::getenv("STRATA_VERIFY_ALL_RESIDENT");
        require(!all_resident || std::atoi(all_resident) != 0, "all-resident graph must be enabled");
        for (const char* name : {"STRATA_VERIFY_DEVICE_PLAN", "STRATA_LFUSE", "STRATA_QFUSE"}) {
            const char* value = std::getenv(name);
            if (value && std::atoi(value) != 0) error = std::string(name) + " must be unset or zero";
            require(!value || std::atoi(value) == 0, "neutral graph configuration for the residency seam");
        }
        require(geometry.n_layers > 1 && geometry.n_expert == 512 && !core::is_qsa_layer(geometry, 0),
                "fixture requires the real Flash-Next layer-zero GDN geometry");
        require(main.layer_lo == 0 && main.layer_hi == geometry.n_layers && main.max_cells >= 2 &&
                a.gdn_state && b.gdn_state && pool && dispatch.pool && dispatch.src &&
                dispatch.worker_contract && dispatch.remote.empty() && dispatch.workers.empty() && !dispatch.peer && !dispatch.failed &&
                device_residency && residency.size() == (size_t) (geometry.n_layers * geometry.n_expert),
                "initialized single-device real-model/session/pool prerequisites");
        for (const auto& completion : dispatch.completion)
            require(!completion.active(), "no expert consumer at cache repurpose boundary");
        cuda(cudaDeviceSynchronize(), "drain startup before repurposing cache");

        // Find enough existing slots for every actual expert of layer zero. Never
        // allocate another expert arena or pretend a nonresident expert is present.
        const auto& layout = kernels::cpu::expert_layout();
        const uint64_t blob_bytes = layout.blob_bytes(0);
        std::vector<int32_t> slots;
        slots.reserve((size_t) geometry.n_expert);
        for (int64_t slot = 0; slot < cache.slots() && slots.size() < (size_t) geometry.n_expert; ++slot)
            if (cache.slot_offset(slot + 1) - cache.slot_offset(slot) >= blob_bytes)
                slots.push_back((int32_t) slot);
        if (slots.size() != (size_t) geometry.n_expert)
            error = "need 512 existing cache slots each at least " + std::to_string(blob_bytes) +
                    " bytes; found " + std::to_string(slots.size()) + "; no extra GPU arena is permitted";
        require(slots.size() == (size_t) geometry.n_expert, "bounded real expert fixture fits admitted cache");
        std::fill(residency.begin(), residency.end(), core::kNotResident);
        for (int64_t expert = 0; expert < geometry.n_expert; ++expert) {
            const uint8_t* blob = dispatch.src->blob(0, expert);
            require(blob != nullptr && cache.fill_slot_blocking(slots[(size_t) expert], blob, error,
                                                               (int64_t) blob_bytes), "fill actual model expert");
            require(cache.verify_slot(slots[(size_t) expert], blob, error, (int64_t) blob_bytes),
                    "verify actual expert cache bytes");
            residency[(size_t) expert] = slots[(size_t) expert];
        }
        auto publish = [&](int expert, bool resident) {
            residency[(size_t) expert] = resident ? slots[(size_t) expert] : core::kNotResident;
            cuda(cudaMemcpy(device_residency, residency.data(), residency.size() * sizeof(int32_t),
                            cudaMemcpyHostToDevice), "publish host/device residency together");
            cuda(cudaStreamSynchronize(nullptr), "residency upload finishes before next graph");
        };
        publish(0, true);
        dispatch.host_res = residency.data();
        dispatch.cache_base = cache.device_slot(0);
        dispatch.cache_blob = (int64_t) layout.max_blob;
        dispatch.cache_slot_off = cache.slot_offsets();
        dispatch.pcie_num = 0; // the selected miss must execute through the real CPU pool

        const size_t handoff_floats = (size_t) core::Verifier::handoff_floats(geometry) * 2;
        Handoff handoff;
        cuda(cudaHostAlloc((void**) &handoff.host, handoff_floats * sizeof(float), cudaHostAllocMapped),
             "allocate priced stage handoff");
        cuda(cudaHostGetDevicePointer((void**) &handoff.device, handoff.host, 0), "map stage handoff");
        core::VerifyHits hits;
        hits.h_res = residency.data(); hits.d_res = device_residency;
        hits.cache_base = cache.device_slot(0); hits.blob = (int64_t) layout.max_blob;
        hits.slot_off = cache.slot_offsets(); hits.n_slots = cache.slots();
        const int rows[2] = {1, 0};
        const int32_t first[2] = {11, 22}, second[2] = {33, 44};
        const std::vector<int32_t> history;
        kernels::SamplerParams sampling;
        sampling.greedy = true; sampling.temperature = 0; sampling.penalty_last_n = 0;
        const core::QsaState no_draft{};
        auto reset = [&] {
            core::session_zero(a, geometry, nullptr, nullptr);
            core::session_zero(b, geometry, nullptr, nullptr);
            cuda(cudaDeviceSynchronize(), "reset real slot state");
            std::memset(handoff.host, 0, handoff_floats * sizeof(float));
        };
        auto init = [&](core::Verifier& verifier) {
            verifier.set_stage(0, 1, nullptr, handoff.device);
            verifier.set_head_sampling(false); // no head/logit observation in this intermediate stage
            require(verifier.init(weights, geometry, main, hits, nullptr, 2, error), "initialize production Verifier");
            require(verifier.init_slots({&a, &b}, error), "bind actual resident sessions");
            for (int slot = 0; slot < 2; ++slot) {
                dispatch.slot_requests[slot] = 100 + slot;
                require(verifier.set_slot_context(slot, dispatch.slot_requests[slot], sampling, &history, true, error),
                        "bind distinct request rows");
            }
            dispatch.request_generation = 100;
            dispatch.plan = verifier.plan_sink();
        };
        auto window = [&](core::Verifier& verifier, const char* phase, int64_t position, int missing,
                          int expected_layers, Observation* observation) {
            PoolProbe probe;
            probe.forward = pool; probe.user = user; probe.dispatch = &dispatch;
            probe.width = geometry.n_expert; probe.missing = missing;
            const int64_t positions[2] = {position, position};
            const int64_t cpu_before = dispatch.multi_misses;
            auto fail_drained = [&](const char* message) {
                verifier.diag(stderr);
                if (!verifier.release_gpu_waits(5000)) {
                    std::fprintf(stderr, "RESIDENCY_MODEL_CHECKS failures=1 phase=%s undrained=1 error=%s\n",
                                 phase, error.c_str());
                    std::fflush(stderr);
                    std::_Exit(1);
                }
                pool(user, nullptr, nullptr, 0, main.k, nullptr, -2, nullptr, nullptr, -1, 0);
                require(false, message);
            };
            if (!verifier.batch_launch(0, rows, 2, position == 0 ? first : second, positions, error))
                fail_drained("launch real batch stage and commit");
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            int result = 0;
            do {
                result = verifier.batch_poll(&PoolProbe::call, &probe, error);
                if (result == 0 && std::chrono::steady_clock::now() < deadline)
                    std::this_thread::sleep_for(std::chrono::microseconds(50));
                else break;
            } while (true);
            if (result != 1) {
                if (result == 0) error = "five-second batch_poll deadline expired (no forced completion accepted)";
                fail_drained("bounded production batch completes");
            }
            if (!probe.error.empty()) error = probe.error;
            require(probe.layers == expected_layers && probe.completions == 1 && probe.error.empty(),
                    "exact layer callbacks and one real final consumer acknowledgement");
            require(!verifier.batch_busy() && verifier.context_idle(error), "quiescent stage after commit");
            for (const auto& completion : dispatch.completion)
                require(!completion.active(), "real expert consumer retired after batch");
            require(verifier.validated_slot_request_id(0) == 100 && verifier.validated_slot_request_id(1) == 101,
                    "production commit canaries validate row owners");
            if (missing >= 0) require(probe.missed_rows > 0 && dispatch.multi_misses > cpu_before,
                                      "lost expert is routed and computed by the actual pool");
            if (observation) {
                require(program::session_fingerprint(a, no_draft, geometry, position + 1, observation->state[0], error,
                                                     verifier.final_R(1)) &&
                        program::session_fingerprint(b, no_draft, geometry, position + 1, observation->state[1], error,
                                                     verifier.final_R(0)), "fingerprint committed real slots");
                observation->handoff.assign(handoff.host, handoff.host + handoff_floats);
            }
            std::printf("RESIDENCY_WINDOW phase=%s layers=%d completions=%d missed_rows=%d committed=2\n",
                        phase, probe.layers, probe.completions, probe.missed_rows);
            return probe.routed[0];
        };
        int missing = -1;
        reset();
        publish(0, false);
        {
            core::Verifier calibration;
            init(calibration); // born nonresident: unaffected by the batch_launch regression
            missing = window(calibration, "calibration", 0, -1, 1, nullptr);
        }
        dispatch.plan = nullptr;
        publish(0, true);
        std::array<Observation, 2> control;
        reset();
        publish(missing, false);
        {
            core::Verifier verifier;
            init(verifier); // same miss placement, but all_resident_ was never true
            window(verifier, "control-loss", 0, missing, 1, &control[0]);
            publish(missing, true);
            window(verifier, "control-restored", 1, -1, 1, &control[1]);
        }
        dispatch.plan = nullptr;
        reset();
        {
            core::Verifier verifier;
            init(verifier); // every layer-zero expert genuinely resident at initialization
            window(verifier, "candidate-initial", 0, -1, 0, nullptr);
            reset(); // after context_idle; candidate starts the identical control prefix
            publish(missing, false);
            for (int phase = 0; phase < 2; ++phase) {
                Observation candidate;
                window(verifier, phase == 0 ? "candidate-loss" : "candidate-restored", phase,
                       phase == 0 ? missing : -1, phase == 0 ? 1 : 0, &candidate);
                for (int slot = 0; slot < 2; ++slot)
                    require(same_committed(control[phase].state[slot], candidate.state[slot]),
                            "exact committed fingerprint against same-placement control");
                require(candidate.handoff.size() == control[phase].handoff.size() &&
                        std::memcmp(candidate.handoff.data(), control[phase].handoff.data(),
                                    handoff_floats * sizeof(float)) == 0,
                        "exact complete stage handoff against same-placement control");
                std::printf("RESIDENCY_COMPARE phase=%s fingerprints=2 handoff_exact=1\n",
                            phase == 0 ? "loss" : "restored");
                if (phase == 0) publish(missing, true);
            }
        }
        dispatch.plan = nullptr;
        std::printf("RESIDENCY_MODEL_CHECKS checks=%d layers=0:1 slots=2 lost_expert=%d windows=6 "
                    "loss_callbacks=1 restored_callbacks=0 fingerprint_exact=1 handoff_exact=1 failures=0\n",
                    checks, missing);
        return 0;
    } catch (const std::exception& exception) {
        dispatch.plan = nullptr;
        std::fprintf(stderr, "RESIDENCY_MODEL_CHECKS checks=%d failures=1 error=%s\n", checks, exception.what());
        return 1;
    }
}
