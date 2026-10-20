// Host-only DFlash2 capture/reference bridge. No target kernels or serving activation.
#pragma once

#include "strata/core/operation_lease.hpp"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>

namespace strata::core::dflash2 {
inline constexpr int hidden_size = 2560;
inline constexpr int vocabulary_size = 248320;
inline constexpr int mask_token = 248077;
inline constexpr int block_size = 4;
inline constexpr int context_rows = 2047;
inline constexpr int max_position = 262144;
inline constexpr std::array<int, 5> tap_layers{5, 15, 25, 35, 45};
inline constexpr int feature_width = hidden_size * int(tap_layers.size());
inline constexpr size_t feature_row_bytes = feature_width * sizeof(uint16_t);
inline constexpr size_t max_feature_bytes = context_rows * feature_row_bytes;
inline constexpr size_t max_header_bytes = 8192;
inline constexpr size_t logits_bytes = (block_size - 1) * vocabulary_size * sizeof(float);
inline constexpr char reference_commit[] = "07ebd93db9f472af339b644bb70221ad8428328a";

class Refusal : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct TargetIdentity {
    std::string pack_sha256, embedding_sha256, head_sha256, tokenizer_sha256;
    bool operator==(const TargetIdentity&) const = default;
};
struct Operation {
    std::string request_id;
    int64_t incarnation = 0, generation = 0;
    int frontier = 0;
    bool operator==(const Operation&) const = default;
};
enum class CapturePhase { prefill, verify };
enum class CapturePoint { completed_second_residual_write, next_layer_first_gr_read_group };

// Borrowed ring views, valid only until the next successful capture commit. Both
// segments are in increasing absolute-position order, never stage arrival order.
struct FeatureView {
    int start = 0, count = 0;
    std::array<std::span<const uint16_t>, 2> segments;
};

// One serial request/incarnation. Allocates a 2047-row committed ring and exactly
// capture_capacity tentative rows once, before target graph/cache preparation.
// The default tentative capacity accepts a full context-sized prefill chunk.
// All methods require a single host owner; stage completions must be serialized.
// Capture reads immutable host-complete FP32 [row, four streams, 2560] residuals.
// It performs ((s0+s1)+s2)+s3 in FP32, *0.25, then one BF16 ties-to-even round.
// No mixed activation, learned final mixer, normalization or target state write.
class CaptureStore {
public:
    CaptureStore(std::string request_id, int64_t incarnation, int origin = 0,
                 int capture_capacity = context_rows);
    ~CaptureStore();
    CaptureStore(const CaptureStore&) = delete;
    CaptureStore& operator=(const CaptureStore&) = delete;

    void begin(const Operation&, int start, int count, CapturePhase);
    // materialized_layer is the tap itself in prefill, tap+1 in verification.
    // Verification must read the previous layer's completed R, not mixed, after
    // the next layer's FIRST gr_read_group and before its attention writes R.
    void capture(const Operation&, int tap_layer, int materialized_layer,
                 std::span<const int> positions, CapturePoint,
                 std::span<const float> residuals);
    // Caller acknowledgement of an already successful target commit. Prefill
    // keeps its whole chunk; verification keeps 1..4 rows, rejecting the suffix.
    void commit(const Operation&, int keep);
    void discard(const Operation&);
    void abort() noexcept;

    FeatureView committed_since(int start) const;
    int frontier() const noexcept { return end_; }
    int retained_start() const noexcept { return end_ - retained_; }
    int origin() const noexcept { return origin_; }
    bool poisoned() const noexcept { return poisoned_; }
    bool capturing() const noexcept { return active_; }
    size_t storage_bytes() const noexcept;

private:
    friend class Bridge;
    [[noreturn]] void refuse(const char*);
    void check_commit(const Operation&, int keep);
    bool owns(const Operation&) const;
    std::string request_id_;
    int64_t incarnation_, generation_ = 0;
    int origin_, capacity_, end_, retained_ = 0, head_ = 0;
    std::unique_ptr<uint16_t[]> ring_, tentative_;
    Operation operation_;
    int start_ = 0, count_ = 0;
    CapturePhase phase_ = CapturePhase::prefill;
    unsigned seen_ = 0;
    bool active_ = false, poisoned_ = false, bridge_attached_ = false;
};

struct BorrowedBinding {
    TargetIdentity identity;
    int hidden = 0, vocabulary = 0;
    uint64_t load_generation = 0; // changes even on reload of identical bytes
    bool operator==(const BorrowedBinding&) const = default;
};

// A real adapter must borrow the already loaded daily target, not load a second
// embedding/head. These calls return host-complete results: explicit device,
// stream, dedicated I/O buffers and synchronization belong to the adapter.
// scratch_lease MUST be the lease shared with every user of that native head's
// scratch, not a bridge-private replacement. No target final norm may be added.
// The adapter and its loaded weights must outlive Bridge; no asynchronous views
// may escape either call. CPU fixtures implement this same interface.
class BorrowedTarget {
public:
    virtual ~BorrowedTarget() = default;
    virtual const BorrowedBinding& binding() const = 0;
    virtual OperationLease& scratch_lease() = 0;
    virtual void raw_embeddings(const Operation&, std::span<const int32_t> tokens,
                                std::span<float> output) = 0;
    // One of three SERIAL native-head calls, on widened BF16 draft hidden rows.
    virtual void native_head(const Operation&, std::span<const float> hidden,
                             std::span<float> full_vocabulary_logits) = 0;
};

struct Proposals {
    std::array<int64_t, block_size - 1> tokens;
    std::array<std::array<int64_t, 16>, block_size - 1> candidates;
};

// Engine side of tools/dflash2_cpu/exchange.py version 1. Adopts and closes an
// inherited, connected, anonymous AF_UNIX STREAM socketpair fd, including on
// construction failure. No bind/listen/HTTP/TCP, process launch or fallback.
// The launcher must execute the pinned reference and enforce combined process
// memory and a hard wall deadline (a socket cannot interrupt CPU model work).
// Exactly one propose -> target verification/capture -> commit_verified may be
// outstanding. Its <=300s deadline includes the bounded scratch wait, embeddings,
// CPU work, head, verifier and ack. Calls are serial; one Bridge may own a store.
// Constructor refusal leaves the store unchanged. Scratch contention before any
// work refuses without poisoning, so the same operation may be retried. Other
// operation failures poison bridge/store; failure while borrowed scratch is held
// also poisons that lease. No partial repair.
class Bridge {
public:
    Bridge(CaptureStore&, BorrowedTarget&, const TargetIdentity& expected_daily_target,
           int inherited_socket_fd, std::chrono::milliseconds timeout = std::chrono::seconds(300));
    ~Bridge();
    Bridge(const Bridge&) = delete;
    Bridge& operator=(const Bridge&) = delete;

    Proposals propose(const Operation&, int32_t anchor);
    // Call ONLY after the real verifier's commit(keep) succeeds and all five
    // tentative target taps are complete. Publication waits for the CPU ack.
    void commit_verified(const Operation&, int keep);
    void abort() noexcept;
    bool poisoned() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strata::core::dflash2
