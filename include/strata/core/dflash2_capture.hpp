// Owned DFlash2 capture admission and operation-scoped target tap staging.
#pragma once

#include "strata/core/dflash2_bridge.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>
#include <string>

namespace strata::core::dflash2 {
inline constexpr int residual_width = 4 * hidden_size;
inline constexpr size_t residual_row_bytes = residual_width * sizeof(float);

// Numeric engine identity, translated to Operation by the eventual host sequencer.
struct CaptureOwner {
    uint64_t request = 0, incarnation = 0, generation = 0;
    bool operator==(const CaptureOwner&) const = default;
};
struct TapView {
    int start = 0, count = 0, keep = 0, tap_layer = 0, materialized_layer = 0;
    CapturePhase phase = CapturePhase::prefill;
    CapturePoint point = CapturePoint::completed_second_residual_write;
    std::span<const float> residuals; // immutable [row, four streams, 2560]
    CaptureOwner owner;
};
struct CaptureSlice {
    int source_row = 0, staging_row = 0, count = 0;
};

// One host submitter, one fixed stage, no allocation. The device-owning verifier /
// prefill owns the mapped payload; this state fences its reuse and host exposure.
// Views remain borrowed until release(owner), reset or destruction. All stages of
// a split must be armed separately with the SAME owner and absolute run interval.
class TapCaptureState {
public:
    bool configure(CapturePhase phase, int capacity, int64_t lb, int64_t le,
                   int64_t hidden, int64_t streams, int64_t layers, std::string& err) {
        if (configured_ || hidden != hidden_size || streams != 4 || layers != 48 ||
            lb < 0 || lb >= le || le > layers || capacity < 1 ||
            capacity > (phase == CapturePhase::verify ? max_verify_rows : context_rows))
            return refuse(err, "invalid or already configured capture geometry");
        unsigned mask = 0;
        for (size_t i = 0; i < tap_layers.size(); ++i) {
            const int tap = tap_layers[i];
            if (tap < lb || tap >= le) continue;
            if (phase == CapturePhase::verify && tap + 1 >= le)
                return refuse(err, "a verifier tap crosses a stage hand-off");
            mask |= 1u << i;
        }
        phase_ = phase; capacity_ = capacity; mask_ = mask; configured_ = true;
        return true;
    }

    // Prefill retains the final <=2047 rows of this exact run, even across chunks.
    // Verification starts at the committed INPUT frontier, including bootstrap.
    // A prompt-length frontier would silently omit the unread final prompt row.
    bool arm(CaptureOwner owner, int64_t frontier, int64_t start, int64_t count, std::string& err) {
        if (!configured_ || status_ != Status::idle || !owner.request || !owner.generation ||
            (owner.request == owner_.request && owner.incarnation == owner_.incarnation &&
             owner.generation <= owner_.generation) || start < 0 || count < 1 ||
            start > max_position - count || frontier < 0 || frontier > max_position)
            return refuse(err, "invalid, stale or overlapping capture owner/range");
        if (phase_ == CapturePhase::verify ?
                (count > capacity_ || start != frontier) : start + count != frontier)
            return refuse(err, "capture does not match the committed frontier");
        owner_ = owner; run_start_ = int(start); run_end_ = int(start + count); next_run_ = run_start_;
        count_ = int(std::min<int64_t>(count, capacity_));
        start_ = phase_ == CapturePhase::prefill ? run_end_ - count_ : run_start_;
        written_.fill(0); keep_ = 0; status_ = Status::armed;
        return true;
    }

    // A split prefill successor sees successive sub-runs, not the whole prompt.
    bool begin_run(uint64_t request, int64_t start, int64_t count, std::string& err) {
        if (status_ == Status::idle) return true; // enabled, but not armed: observer off
        if (status_ != Status::armed || request != owner_.request || start != next_run_ ||
            count < 1 || count > run_end_ - start ||
            (phase_ == CapturePhase::verify && start + count != run_end_))
            return refuse(err, "run does not own the armed capture interval");
        active_end_ = int(start + count); status_ = Status::running;
        return true;
    }
    CaptureSlice slice(int64_t start, int64_t count) const {
        if (!running()) return {};
        const int64_t first = std::max<int64_t>(start, start_);
        const int64_t end = std::min<int64_t>(start + count, start_ + count_);
        if (end <= first) return {};
        return {int(first - start), int(first - start_), int(end - first)};
    }
    // Prefill's actual, ordered copy submissions, recorded after each tap write.
    bool record(int tap, CaptureSlice part, std::string& err) {
        if (!running() || !owns_tap(tap) || part.count < 1 || part.staging_row != written_[size_t(tap)] ||
            part.count > count_ - part.staging_row)
            return refuse(err, "duplicate, missing or unordered prefill tap rows");
        written_[size_t(tap)] += part.count;
        return true;
    }
    bool finish_run(std::string& err) {
        if (status_ == Status::idle) return true;
        if (!running()) return refuse(err, "capture run is not in flight");
        next_run_ = active_end_;
        // Verifier taps are fixed graph nodes for every owned tap/group. They are
        // not published here: target commit + wait_commit must succeed first.
        if (phase_ == CapturePhase::verify) { status_ = Status::verified; return true; }
        if (next_run_ < run_end_) { status_ = Status::armed; return true; }
        for (size_t i = 0; i < written_.size(); ++i)
            if (owns_tap(int(i)) && written_[i] != count_)
                return refuse(err, "prefill finished without every retained tap row");
        keep_ = count_; status_ = Status::ready;
        return true;
    }
    bool committed(int keep, std::string& err) {
        if (status_ == Status::idle) return true;
        if (phase_ != CapturePhase::verify || status_ != Status::verified || keep < 1 ||
            keep > run_end_ - run_start_)
            return refuse(err, "capture commit does not own a verified prefix");
        keep_ = keep; status_ = Status::committing;
        return true;
    }
    bool waited(std::string& err) {
        if (status_ == Status::committing) status_ = Status::ready;
        else if (status_ != Status::idle && status_ != Status::armed && status_ != Status::ready)
            return refuse(err, "capture requires a successful target commit before publication");
        return true;
    }
    bool view(CaptureOwner owner, int tap, const float* host, TapView& out, std::string& err) const {
        if (status_ != Status::ready || owner != owner_ || !owns_tap(tap) || count_ == 0 || !host)
            return refuse(err, "tap is absent, stale or not host-complete");
        const int layer = tap_layers[size_t(tap)];
        out = {start_, count_, keep_, layer, layer + (phase_ == CapturePhase::verify ? 1 : 0), phase_,
               phase_ == CapturePhase::verify ? CapturePoint::next_layer_first_gr_read_group :
                                               CapturePoint::completed_second_residual_write,
               {host, size_t(count_) * residual_width}, owner_};
        return true;
    }
    bool release(CaptureOwner owner, std::string& err) {
        if (status_ != Status::ready || owner != owner_)
            return refuse(err, "capture release requires its completed owner");
        status_ = Status::idle;
        return true;
    }
    void poison() { if (status_ != Status::idle) status_ = Status::poisoned; }
    bool idle() const { return status_ == Status::idle; }
    bool running() const { return status_ == Status::running; }
    bool copying() const { return running() && count_ != 0; }
    bool owns_tap(int tap) const { return tap >= 0 && tap < int(tap_layers.size()) && (mask_ & (1u << tap)); }
    size_t staging_bytes() const {
        size_t taps = 0;
        for (size_t i = 0; i < tap_layers.size(); ++i) taps += owns_tap(int(i));
        return taps * size_t(capacity_) * residual_row_bytes;
    }
private:
    static bool refuse(std::string& err, const char* why) { err = std::string("dflash2 capture: ") + why; return false; }
    enum class Status { idle, armed, running, verified, committing, ready, poisoned };
    Status status_ = Status::idle;
    bool configured_ = false;
    CapturePhase phase_ = CapturePhase::prefill;
    int capacity_ = 0, start_ = 0, count_ = 0, keep_ = 0;
    int run_start_ = 0, run_end_ = 0, next_run_ = 0, active_end_ = 0;
    unsigned mask_ = 0;
    CaptureOwner owner_;
    std::array<int, tap_layers.size()> written_{};
};

// Mapped raw staging is NOT the BF16 feature bound. Quote each owning layer range
// once; CUDA host allocations are charged at page granularity, including flags.
struct CaptureAllocation {
    size_t prefill = 0, verify = 0, flags = 0;
    size_t mapped_bytes() const { return prefill + verify + flags; }
    static bool stage(int64_t lb, int64_t le, CaptureAllocation& out, std::string& err,
                       int verify_rows = block_size) {
        if (verify_rows < block_size || verify_rows > max_verify_rows) {
            err = "dflash2 capture: physical verifier rows must be in [4,8]"; return false;
        }
        TapCaptureState pf, vr;
        if (!pf.configure(CapturePhase::prefill, context_rows, lb, le, hidden_size, 4, 48, err) ||
            !vr.configure(CapturePhase::verify, verify_rows, lb, le, hidden_size, 4, 48, err)) return false;
        auto pages = [](size_t n) { return (n + 4095) / 4096 * 4096; };
        out = {};
        for (size_t i = 0; i < tap_layers.size(); ++i) {
            if (pf.owns_tap(int(i))) out.prefill += pages(size_t(context_rows) * residual_row_bytes);
            if (vr.owns_tap(int(i))) out.verify += pages(size_t(verify_rows) * residual_row_bytes);
        }
        out.flags = 2 * pages(sizeof(uint32_t));
        return true;
    }
};

// One serial committed-input frontier, not prompt length or emitted-token count.
// begin() fences the target operation; consume() copies/reduces only host-complete
// committed prefixes and NEVER retains a TapView/pointer. publish() has no caller
// keep argument: the target's completed views are the authoritative commit count.
class CaptureProducer {
public:
    CaptureProducer();
    CaptureProducer(std::span<uint16_t> shared_tentative, OperationLease& shared_lease);
    bool resume(uint64_t request, uint64_t incarnation, int origin, int frontier, std::string& error);
    bool reset(uint64_t request, uint64_t incarnation, int origin, std::string& err);
    bool begin(CapturePhase, int64_t start, int64_t count, std::string& err,
               const OperationLease::Guard* lease = nullptr);
    bool consume(const TapView&, std::string& err, const OperationLease::Guard* lease = nullptr);
    bool publish(std::string& err);
    CaptureOwner owner() const { return owner_; }
    bool cancel(std::string& error);
    const CaptureStore& store() const { return store_; }
    int frontier() const { return frontier_; }
    bool active() const { return active_; }
    int64_t committed_rows() const { return committed_rows_; }
    int64_t rejected_rows() const { return rejected_rows_; }
    bool cancellable() const { return !poisoned_ && active_ && !seen_ && !store_.capturing(); }
    void abort() noexcept { poisoned_ = true; active_ = false; lease_ = nullptr; store_.abort(); }
private:
    CaptureStore store_;
    CaptureOwner owner_;
    Operation operation_;
    CapturePhase phase_ = CapturePhase::prefill;
    int frontier_ = 0, run_start_ = 0, run_count_ = 0, start_ = 0, count_ = 0, keep_ = 0;
    int64_t committed_rows_ = 0, rejected_rows_ = 0;
    unsigned seen_ = 0;
    bool active_ = false, poisoned_ = false;
    std::array<int, context_rows> positions_{};
    const OperationLease::Guard* lease_ = nullptr;
};
} // namespace strata::core::dflash2
