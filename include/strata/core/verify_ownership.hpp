// Verifier ownership/canary contract. No CUDA dependency: also used by CPU seam tests.
#pragma once

#include <atomic>
#include <cstdint>

namespace strata::core {

// One host submitter owns all verifier APIs. The atomic phase additionally lets
// the watchdog poison ownership without making concurrent host submissions safe.
class VerifyScratchOwnership {
public:
    enum class Phase { idle, running, ready, committing, poisoned };
    Phase phase() const { return phase_.load(); }
    bool begin() { return transition(Phase::idle, Phase::running); }
    bool ready() { return transition(Phase::running, Phase::ready); }
    bool commit() { return transition(Phase::ready, Phase::committing); }
    bool finish() { return transition(Phase::committing, Phase::idle); }
    bool finish_batch() { return transition(Phase::running, Phase::idle); }
    bool abandon_unlaunched() { return transition(Phase::running, Phase::idle); }
    void poison() { phase_.store(Phase::poisoned); }
private:
    bool transition(Phase from, Phase to) { return phase_.compare_exchange_strong(from, to); }
    std::atomic<Phase> phase_{Phase::idle};
};

inline constexpr int kVerifyCanaryRows = 8;
inline constexpr int kVerifySoloOwner = 8;
inline constexpr unsigned kVerifyBatchGraphKeys = 32;

// Bound at admission, independently of the per-operation mapped row staging.
struct VerifyCanaryOwner {
    uint64_t request_id = 0;
    int32_t steering = 0;
    uint32_t reserved = 0;
};
struct VerifyCanaryInput {
    uint64_t request_id = 0;
    uint64_t epoch = 0;
    int32_t slot = 0;
    int32_t token = 0;
    int32_t position = 0;
    int32_t steering = 0;
};
struct VerifyCanaryOutput {
    VerifyCanaryInput input;
    uint64_t owner_id = 0;
    int32_t graph_slot = 0;
    int32_t owner_steering = 0;
    int32_t token = 0;
    int32_t position = 0;
    int32_t steering = 0;
    int32_t output_token = 0;
    uint32_t errors = 0;
    uint32_t phase = 0; // 1: window output; 2: committed state
};
inline bool verify_canary_matches(const VerifyCanaryOutput& out, const VerifyCanaryInput& expected,
                                  uint32_t phase) {
    return out.errors == 0 && out.phase == phase &&
           out.input.request_id == expected.request_id && out.input.epoch == expected.epoch &&
           out.input.slot == expected.slot && out.input.token == expected.token &&
           out.input.position == expected.position && out.input.steering == expected.steering &&
           out.owner_id == expected.request_id && out.graph_slot == expected.slot &&
           out.owner_steering == expected.steering && out.token == expected.token &&
           out.position == expected.position && out.steering == expected.steering;
}

// Operation input witness staged independently of the model's mapped/token buffers.
// A stream-ordered upload permits queued prefill groups without host synchronization.
struct OperationInputs {
    uint64_t request_id = 0, epoch = 0;
    int32_t position = 0, accepted = 0;
    int32_t tokens[8] = {};
    uint32_t checks = 0, reserved = 0;
};
static_assert(sizeof(OperationInputs) == 64);

// Exact allocations, excluding allocator overhead: fixed device scratch and
// independent device owner table; mapped input/owner upload/window/commit output.
static_assert(sizeof(VerifyCanaryInput) == 32 && sizeof(VerifyCanaryOwner) == 16 &&
              sizeof(VerifyCanaryOutput) == 72, "canary admission arithmetic must match device layout");
inline constexpr uint64_t kVerifyCanaryOutputBytes = sizeof(VerifyCanaryOutput) * kVerifyCanaryRows;
inline constexpr uint64_t kVerifyCanaryDeviceBytes = kVerifyCanaryOutputBytes +
    sizeof(VerifyCanaryOwner) * (kVerifyCanaryRows + 1);
inline constexpr uint64_t kVerifyCanaryHostBytes = sizeof(VerifyCanaryInput) * kVerifyCanaryRows +
    sizeof(VerifyCanaryOwner) * (kVerifyCanaryRows + 1) + 2 * kVerifyCanaryOutputBytes;
inline constexpr uint64_t kOperationCanaryDeviceBytes = sizeof(VerifyCanaryOutput) + 9 * sizeof(VerifyCanaryOwner) + sizeof(OperationInputs);
inline constexpr uint64_t kOperationCanaryHostBytes = sizeof(VerifyCanaryInput) + sizeof(VerifyCanaryOutput) + sizeof(uint32_t);

} // namespace strata::core
