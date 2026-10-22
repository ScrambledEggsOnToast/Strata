#pragma once

#include "strata/core/dflash2_drafter.hpp"
#include "strata/core/verify.hpp"
#include "strata/prefill/prefill.hpp"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <span>

namespace strata::program::dflash2_serving {
inline bool diagnostic_logging() {
    static const bool enabled = std::getenv("STRATA_STATE_HASH") != nullptr ||
                                std::getenv("DFLASH2_CAPTURE_DIR") != nullptr;
    return enabled;
}

// A physical target frame owns raw capture storage, but never holds the neural
// scratch lease while target graphs run. Each seat is projected serially only
// after every physical stage has committed and drained.
class CaptureFrame {
public:
    void bind(core::dflash2::DflashNativeDrafter* draft,
              std::span<core::Verifier* const> verifiers,
              std::span<prefill::Prefill* const> prompts,
              const std::array<int, 5>& tap_stages) {
        draft_ = draft;
        stages_ = int(verifiers.size());
        for (int stage = 0; stage < stages_; ++stage) {
            verifiers_[size_t(stage)] = verifiers[size_t(stage)];
            prompts_[size_t(stage)] = prompts[size_t(stage)];
        }
        tap_stages_ = tap_stages;
    }
    bool begin(core::dflash2::CapturePhase phase, int slot, int64_t start,
               int64_t count, std::string& error) {
        if (!draft_) return true;
        if (count_ || stages_ < 1 || stages_ > 4) return refuse(error, "overlapping physical frame");
        phase_ = phase; batch_ = false;
        if (!reserve(slot, start, count, error)) return false;
        const auto owner = seats_[0].capture;
        const int64_t frontier = phase == core::dflash2::CapturePhase::prefill ? start + count : start;
        for (int stage = 0; stage < stages_; ++stage) {
            const bool armed = phase == core::dflash2::CapturePhase::prefill
                ? prompts_[size_t(stage)]->arm_dflash2_capture(owner, frontier, start, count, error, slot)
                : verifiers_[size_t(stage)]->arm_dflash2_capture(owner, frontier, start, int(count), error);
            if (!armed) return false; // partial arm is fatal, never reused as a healthy frame
        }
        return true;
    }
    bool begin_batch(std::span<const core::BatchRow> rows, std::string& error) {
        if (!draft_) return true;
        if (count_ || stages_ < 1 || stages_ > 4 || rows.empty() || rows.size() > kernels::kVerifyMaxT)
            return refuse(error, "invalid packed physical frame");
        phase_ = core::dflash2::CapturePhase::verify; batch_ = true;
        std::array<core::dflash2::BatchCaptureRow, kernels::kVerifyMaxT> captures;
        for (size_t first = 0; first < rows.size();) {
            const auto& row = rows[first];
            size_t end = first + 1;
            while (end < rows.size() && rows[end].slot == row.slot) ++end;
            if (row.mask != core::BatchRowMask::active || end - first > kernels::kVerifyMaxT ||
                row.request_id != draft_->owner(row.slot).request ||
                !reserve(row.slot, row.position, int64_t(end - first), error)) return false;
            const auto capture = seats_[size_t(count_ - 1)].capture;
            for (size_t at = first; at < end; ++at) {
                if (rows[at].mask != core::BatchRowMask::active || rows[at].request_id != row.request_id ||
                    rows[at].position != row.position + int64_t(at - first))
                    return refuse(error, "noncontiguous or foreign packed owner rows");
                captures[at] = {row.slot, capture, rows[at].position};
            }
            first = end;
        }
        for (int stage = 0; stage < stages_; ++stage)
            if (!verifiers_[size_t(stage)]->arm_dflash2_batch(
                    {captures.data(), rows.size()}, error)) return false;
        return true;
    }
    // Cancellation may discard an entire drained owner before any tap is
    // consumed. Once projection starts, failure poisons the real common lease.
    bool finish(std::string& error, uint32_t discard_seats = 0) {
        if (!draft_ || !count_) return true;
        using Clock = std::chrono::steady_clock;
        const bool phases = core::dflash2::feature_phases_enabled();
        const auto wait_started = phases ? Clock::now() : Clock::time_point{};
        if (phase_ == core::dflash2::CapturePhase::verify)
            for (int stage = 0; stage < stages_; ++stage)
                if (!verifiers_[size_t(stage)]->wait_commit(error)) return false;
        const auto waited = phases ? Clock::now() : Clock::time_point{};
        if (phases) std::fprintf(stderr, "DFLASH2_PHASE capture_wait completed=1 phase=%s owners=%d target_wait_ms=%.6f\n",
            phase_ == core::dflash2::CapturePhase::prefill ? "prefill" : "verify", count_,
            std::chrono::duration<double, std::milli>(waited - wait_started).count());
        for (int index = 0; index < count_; ++index) {
            const auto& seat = seats_[size_t(index)];
            if (!draft_->owns(seat.ticket)) return refuse(error, "stale target completion ticket");
            if (!(discard_seats & (uint32_t(1) << seat.ticket.seat))) {
                auto& producer = draft_->capture_producer(seat.ticket.seat);
                core::NeuralDrafter::Operation operation(*draft_, error, seat.ticket.seat,
                    nullptr, 0, int32_t(seat.start), &seat.ticket);
                if (!operation) return false;
                struct AbortUnpublished {
                    core::dflash2::CaptureProducer& producer;
                    bool published = false;
                    ~AbortUnpublished() { if (!published) producer.abort(); }
                } completion{producer};
                const auto assembly_started = phases ? Clock::now() : Clock::time_point{};
                for (size_t tap = 0; tap < core::dflash2::tap_layers.size(); ++tap) {
                    const int stage = tap_stages_[tap];
                    core::dflash2::TapView view;
                    const bool ready = phase_ == core::dflash2::CapturePhase::prefill
                        ? prompts_[size_t(stage)]->dflash2_tap(seat.capture, core::dflash2::tap_layers[tap], view, error)
                        : batch_
                            ? verifiers_[size_t(stage)]->dflash2_batch_tap(seat.capture, seat.ticket.seat,
                                core::dflash2::tap_layers[tap], view, error)
                            : verifiers_[size_t(stage)]->dflash2_tap(seat.capture,
                                core::dflash2::tap_layers[tap], view, error);
                    if (!ready || !producer.consume(view, error, &operation.guard())) return false;
                }
                if (!producer.publish(error)) return false;
                completion.published = true;
                const auto assembled = phases ? Clock::now() : Clock::time_point{};
                const int64_t from = std::max<int64_t>(producer.store().retained_start(), seat.project_from);
                const auto features = producer.store().committed_since(int(from));
                if (!draft_->ingest_committed(operation, features, error) || !operation.finish(error)) return false;
                if (phases) {
                    const auto completed = Clock::now();
                    auto ms = [](auto a, auto b) {
                        return std::chrono::duration<double, std::milli>(b - a).count();
                    };
                    std::fprintf(stderr, "DFLASH2_PHASE capture_assembly completed=1 request=%llu incarnation=%llu "
                        "generation=%llu slot=%d phase=%s start=%d rows=%d "
                        "host_assembly_ms=%.6f ingest_completed_ms=%.6f\n",
                        (unsigned long long)seat.capture.request, (unsigned long long)seat.capture.incarnation,
                        (unsigned long long)seat.capture.generation, seat.ticket.seat,
                        phase_ == core::dflash2::CapturePhase::prefill ? "prefill" : "verify", features.start, features.count,
                        ms(assembly_started, assembled), ms(assembled, completed));
                }
                if (diagnostic_logging()) std::fprintf(stderr, "DFLASH2_COMMITTED request=%llu incarnation=%llu generation=%llu slot=%d "
                    "phase=%s frontier=%d retained_start=%d committed_rows=%lld rejected_rows=%lld taps=5\n",
                    (unsigned long long)seat.capture.request, (unsigned long long)seat.capture.incarnation,
                    (unsigned long long)seat.capture.generation, seat.ticket.seat,
                    phase_ == core::dflash2::CapturePhase::prefill ? "prefill" : "verify", producer.frontier(),
                    producer.store().retained_start(), (long long)producer.committed_rows(),
                    (long long)producer.rejected_rows());
            }
            for (int stage = 0; stage < stages_; ++stage) {
                const bool released = phase_ == core::dflash2::CapturePhase::prefill
                    ? prompts_[size_t(stage)]->release_dflash2_capture(seat.capture, error)
                    : batch_
                        ? verifiers_[size_t(stage)]->release_dflash2_batch(seat.capture, seat.ticket.seat, error)
                        : verifiers_[size_t(stage)]->release_dflash2_capture(seat.capture, error);
                if (!released) return false;
            }
            if (discard_seats & (uint32_t(1) << seat.ticket.seat)) {
                if (!draft_->discard_inputs(seat.ticket, error)) return false;
                if (diagnostic_logging()) std::fprintf(stderr, "DFLASH2_DISCARDED request=%llu incarnation=%llu "
                    "owner_generation=%llu capture_generation=%llu slot=%d phase=%s frontier=%lld rows=%lld released=1\n",
                    (unsigned long long)seat.ticket.request, (unsigned long long)seat.ticket.incarnation,
                    (unsigned long long)seat.ticket.generation, (unsigned long long)seat.capture.generation,
                    seat.ticket.seat, phase_ == core::dflash2::CapturePhase::prefill ? "prefill" : "verify",
                    (long long)seat.start, (long long)seat.count);
            }
        }
        count_ = 0;
        return true;
    }
private:
    struct Seat {
        core::DraftOwner ticket;
        core::dflash2::CaptureOwner capture;
        int64_t start = 0, count = 0, project_from = 0;
    };
    bool reserve(int slot, int64_t start, int64_t count, std::string& error) {
        if (slot < 0 || slot > 8 || count_ == int(seats_.size())) return refuse(error, "invalid capture seat");
        auto& seat = seats_[size_t(count_)];
        const auto expected = draft_->owner(slot);
        const auto coverage = draft_->slot_coverage(slot);
        seat.start = start;
        seat.count = count;
        seat.project_from = coverage.present && coverage.coherent ? coverage.certified_end : draft_->origin(slot);
        if (!draft_->reserve_inputs(slot, expected, start, count, seat.ticket, error)) return false;
        auto& producer = draft_->capture_producer(slot);
        if (!producer.begin(phase_, start, count, error)) return false;
        seat.capture = producer.owner(); // capture generation is not the common ticket counter
        if (seat.capture.request != seat.ticket.request || seat.capture.incarnation != seat.ticket.incarnation)
            return refuse(error, "capture producer differs from actual request incarnation");
        ++count_;
        return true;
    }
    static bool refuse(std::string& error, const char* reason) {
        error = std::string("dflash2 serving capture: ") + reason; return false;
    }
    core::dflash2::DflashNativeDrafter* draft_ = nullptr;
    std::array<core::Verifier*, 4> verifiers_{};
    std::array<prefill::Prefill*, 4> prompts_{};
    std::array<int, 5> tap_stages_{};
    std::array<Seat, kernels::kVerifyMaxT> seats_{};
    int stages_ = 0, count_ = 0;
    bool batch_ = false;
    core::dflash2::CapturePhase phase_ = core::dflash2::CapturePhase::verify;
};

inline void log_window(const core::DraftOwner& owner, int64_t frontier,
        std::span<const int32_t> tokens, int accepted, int keep, int64_t remaining,
        const kernels::SamplerParams& sampling, uint64_t history_cells, uint64_t bias_hash,
        std::span<const int64_t> eos, const char* source = "neural",
        int neural_drafts = -1, const char* phase = "decode") {
    if (!diagnostic_logging()) return;
    std::fprintf(stderr, "DFLASH2_WINDOW request=%llu incarnation=%llu generation=%llu slot=%d frontier=%lld "
        "T=%zu a=%d keep=%d remaining=%lld greedy=%d temperature=%.9g seed=%llu top_k=%d top_p=%.9g "
        "min_p=%.9g min_keep=%d penalty_last_n=%d penalty_repeat=%.9g penalty_freq=%.9g penalty_present=%.9g "
        "gumbel=%d history_cells=%llu bias_hash=%016llx source=%s neural_drafts=%d phase=%s tokens=",
        (unsigned long long)owner.request, (unsigned long long)owner.incarnation,
        (unsigned long long)owner.generation, owner.seat, (long long)frontier, tokens.size(), accepted, keep,
        (long long)remaining, int(sampling.greedy), sampling.temperature, (unsigned long long)sampling.seed,
        sampling.top_k, sampling.top_p, sampling.min_p, sampling.min_keep, sampling.penalty_last_n,
        sampling.penalty_repeat, sampling.penalty_freq, sampling.penalty_present, int(sampling.gumbel),
        (unsigned long long)history_cells, (unsigned long long)bias_hash, source,
        neural_drafts >= 0 ? neural_drafts : int(tokens.size()) - 1, phase);
    for (size_t i = 0; i < tokens.size(); ++i) std::fprintf(stderr, "%s%d", i ? "," : "", tokens[i]);
    std::fprintf(stderr, " eos=");
    for (size_t i = 0; i < eos.size(); ++i) std::fprintf(stderr, "%s%lld", i ? "," : "", (long long)eos[i]);
    std::fputc('\n', stderr);
}
} // namespace strata::program::dflash2_serving
