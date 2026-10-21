// Native, fixed-shape DFlash2 candidate selection; borrowed weights are owned by DeviceForward.
#pragma once

#include "strata/core/spec_prob.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include "strata/kernels/dflash2_selector.hpp"

#include <string>

namespace strata::core::dflash2 {

struct Selection {
    static constexpr size_t probability_row_slots = kernels::dflash2::selector_rows * kSpecQStride;
    std::array<int32_t, kernels::dflash2::selector_rows> tokens{};
    std::array<std::array<int32_t, 16>, kernels::dflash2::selector_rows> candidates{};
    // Conditional on anchor and the already drawn proposal prefix, in candidates' original order.
    // Greedy selection has no q distribution: all entries are zero and sampled is false.
    std::array<std::array<float, 16>, kernels::dflash2::selector_rows> conditional_probabilities{};
    // Draft-family confidence, not calibrated target acceptance: chosen-token
    // probability over these predecessor-conditioned candidates. Greedy uses
    // unit temperature without exporting a sampled verifier q distribution.
    std::array<float, kernels::dflash2::selector_rows> confidence{};
    bool sampled = false;

    // Writes exactly 3*kSpecQStride int32 slots: ids[0..15], then -1 through id[63];
    // FP32 probability bits at slots [64..79], then zero. Greedy rows are all -1/zero,
    // the existing verifier's no-list/point-mass representation. Destination is host memory.
    void fill_probability_rows(int32_t* destination) const noexcept;
};

// CPU ordering authority: unchanged model.py at 07ebd93db9f472af339b644bb70221ad8428328a,
// torch 2.13.0+cpu (cf30153c4c131c8164ee7798e5022d810682e2cb), built with GCC 13.3.
// For k=16,n=248320, its CPU topk(sorted=False) takes std::partial_sort and therefore
// returns descending values. Equal values have GNU heap order, NOT a token-index tie break:
// heapify from parent 7 down to 0; prefer the right equal child while sifting down;
// scan vocabulary in ascending index and replace only a strictly worse root; sort that heap.
// The CUDA path replays precisely that rule whenever any top-16 value is repeated.
// Greedy score ties select the first candidate slot, as torch.argmax does. This pins the
// installed CPU implementation, not an API guarantee across other Torch/stdlib builds.
class DeviceSelector {
public:
    // Borrowed immutable, row-major BF16 device pointers: [256,2560], [248320,256],
    // [248320,256]. They must remain alive until this object has drained and is destroyed.
    // Exactly one device scratch allocation and one pinned result allocation, at construction.
    DeviceSelector(int device, const uint16_t* hidden_projection, const uint16_t* predecessor_codebook,
                   const uint16_t* successor_codebook);
    ~DeviceSelector();
    DeviceSelector(const DeviceSelector&) = delete;
    DeviceSelector& operator=(const DeviceSelector&) = delete;
    DeviceSelector(DeviceSelector&&) = delete;
    DeviceSelector& operator=(DeviceSelector&&) = delete;

    // Inputs are borrowed device buffers on device(), ready in the caller-owned stream.
    // All numerical work is on that stream; completion precedes result publication. No graph
    // capture: this API returns host proposals. Concurrent calls share OperationLease's bounded
    // wait; no private RNG. Successful calls allocate nothing. Failure leaves output unchanged.
    // Positive temperature consumes exactly the supplied finite uniforms in [0,1), including zero.
    // FP32 q normalization has a proposed absolute validity bound of 16*FLT_EPSILON;
    // this is not a measured CPU-reference drift bound.
    // Invalid arguments refuse before launch; any failed launched work permanently poisons
    // scratch, even if it drains. Destruction requires no concurrent host callers; inability
    // to drain is fail-stop rather than freeing storage still owned by a GPU consumer.
    // If a failed consumer cannot drain, retain borrowed inputs and its stream as well as
    // weights until explicit drain succeeds or destruction takes the fail-stop path.
    bool select(int rows, const uint16_t* device_hidden_rows, const float* device_logits_rows,
                int32_t anchor, float temperature, const std::array<float, 3>& uniforms,
                Selection& output, void* stream, std::string& error);

    // Selection is synchronous, including its failure drain. This explicitly joins any retained
    // failed consumer; successful draining never repairs poison or authorizes further selection.
    bool drain(std::string& error);
    bool poisoned() const noexcept;
    int device() const noexcept;
    size_t device_bytes() const noexcept;
    // Exact requested payloads come from the fixed workspace/result layouts;
    // privately owned weights/streams/events and allocator-bookkeeping charges are absent.
    static size_t scratch_bytes() noexcept;
    static size_t pinned_bytes() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace strata::core::dflash2
