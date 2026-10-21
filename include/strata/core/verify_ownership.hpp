// Verifier ownership/canary contract. No CUDA dependency: also used by CPU seam tests.
#pragma once

#include "strata/core/batch_rows.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <map>

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

// Native DFlash2's complete row-schedule superset over four physical slots:
// dflash2_batch_rows_supported enforces this union; other modes keep the general row validator.
// decode: 4*4 singleton shapes + 4*3*4*4 ordered two-owner shapes = 208;
// retained known-input target reads: singleton depths 5..8 add 4*4 = 16;
// final prompt rows: every distinct-owner order of length 3 or 4 adds 4P3+4P4 = 48.
// Length-one/two final prompts and cancellation/partial-ready subsets already occur above.
// Thus 272 layouts * bases {0,8} * residency variants {0,1} = 1088 keys. A physical
// verifier still holds at most eight rows; the optional third native replay owner is unsupported.
// Failed capture reservations remain keys in BOTH maps until idle graph release.
inline constexpr unsigned kDflash2BatchGraphKeys = 1088;

// Prospective Linux64 release libstdc++/glibc host-node ACCOUNTING ALLOWANCE, not a
// measured allocator footprint or CUDA graph executable/capture-storage bound.
// Each key reserves a window node and a commit node, including failed reservations.
// The 256-byte/node allowance includes allocator/header/alignment slack, with one
// additional insertion temporary charged at peak. The embedded map objects are already
// part of sizeof(Verifier). Charge this FULL allowance per physical native verifier/stage.
// The process host ceiling and unchanged opaque CUDA allowances remain independently enforced;
// neither the node-size assertion below nor this finite allowance proves runtime graph fit.
inline constexpr uint64_t kDflash2BatchGraphNodeBytes = 256;
inline constexpr uint64_t kDflash2BatchGraphHostBytes =
    (2ull * kDflash2BatchGraphKeys + 1) * kDflash2BatchGraphNodeBytes;

#if defined(__linux__) && defined(__LP64__) && defined(__GLIBC__) && defined(__GLIBCXX__) && !defined(_GLIBCXX_DEBUG)
inline constexpr bool kDflash2BatchGraphHostPricingSupported = sizeof(int) == 4 && sizeof(void*) == 8;
#else
inline constexpr bool kDflash2BatchGraphHostPricingSupported = false;
#endif

// Native admission must refuse unsupported pricing ABIs, not break unrelated legacy builds.
// Only the opt-in native build references libstdc++'s implementation-specific node type.
#if defined(STRATA_DFLASH2_CAPTURE) && defined(__linux__) && defined(__LP64__) && \
    defined(__GLIBC__) && defined(__GLIBCXX__) && !defined(_GLIBCXX_DEBUG)
namespace detail {
using VerifyBatchGraphMapValue = std::map<VerifyBatchKey, void*>::value_type;
inline constexpr size_t kVerifyBatchGraphAllocatorSlack = 2 * sizeof(size_t) + alignof(std::max_align_t);
// This checks the actual key/pointer-handle/tree-link node shape, not allocator RSS.
static_assert(sizeof(std::_Rb_tree_node<VerifyBatchGraphMapValue>) + kVerifyBatchGraphAllocatorSlack <=
                  kDflash2BatchGraphNodeBytes,
              "Linux64 std::map node no longer fits the prospective batch graph host-node allowance");
} // namespace detail
#endif

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
    // Zero: dense draft steps. Nonzero: prompt QSA width=min(position+1,prompt_cap).
    uint32_t checks = 0, prompt_cap = 0;
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
