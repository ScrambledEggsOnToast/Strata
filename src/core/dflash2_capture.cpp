#include "strata/core/dflash2_capture.hpp"

#include <numeric>
#include <charconv>

namespace strata::core::dflash2 {
namespace {
void require_capture(bool ok, const char* message) {
    if (!ok) throw Refusal(std::string("dflash2 producer: ") + message);
}
}
CaptureProducer::CaptureProducer() : store_("unbound", 0) { operation_.request_id.reserve(128); }
CaptureProducer::CaptureProducer(std::span<uint16_t> staging, OperationLease& lease)
    : store_("unbound", 0, 0, staging, lease) { operation_.request_id.reserve(128); }
bool CaptureProducer::resume(uint64_t request, uint64_t incarnation, int origin, int frontier,
                             std::string& error) {
    if (!request || incarnation < owner_.incarnation ||
        (incarnation == owner_.incarnation && request != owner_.request)) {
        error = "dflash2 producer: stale capture resume identity"; return false;
    }
    try {
        require_capture(!poisoned_ && !active_ && request && incarnation <= uint64_t(INT64_MAX) &&
                        incarnation >= owner_.incarnation, "stale/busy capture resume");
        std::array<char, 21> id{};
        const auto end = std::to_chars(id.data(), id.data() + id.size(), request);
        const std::string_view text{id.data(), size_t(end.ptr - id.data())};
        store_.resume(text, int64_t(incarnation), origin, frontier);
        if (incarnation > owner_.incarnation) owner_.generation = 0;
        owner_.request = request; owner_.incarnation = incarnation; frontier_ = frontier;
        operation_.request_id.assign(text); operation_.incarnation = int64_t(incarnation);
        lease_ = nullptr;
        return true;
    } catch (const std::exception& failure) { error = failure.what(); abort(); return false; }
}
bool CaptureProducer::reset(uint64_t request, uint64_t incarnation, int origin, std::string& err) {
    try {
        require_capture(!poisoned_ && !active_ && request && incarnation <= uint64_t(INT64_MAX) &&
                        incarnation > owner_.incarnation,
                        "stale/busy request binding");
        store_.reset(std::to_string(request), int64_t(incarnation), origin);
        owner_ = {request, incarnation, 0}; frontier_ = origin;
        operation_.request_id = std::to_string(request);
        operation_.incarnation = int64_t(incarnation);
        committed_rows_ = 0; rejected_rows_ = 0;
        return true;
    } catch (const std::exception& error) { err = error.what(); abort(); return false; }
}
bool CaptureProducer::begin(CapturePhase phase, int64_t start, int64_t count, std::string& err,
                            const OperationLease::Guard* lease) {
    try {
        require_capture(!poisoned_ && !active_ && owner_.request && owner_.generation < uint64_t(INT64_MAX) &&
                        (phase == CapturePhase::prefill || phase == CapturePhase::verify) &&
                        start == frontier_ && count > 0 && count <= max_position - start &&
                        (phase == CapturePhase::prefill || count <= max_verify_rows),
                        "noncontiguous, stale or overlapping target operation");
        phase_ = phase; run_start_ = int(start); run_count_ = int(count);
        count_ = phase == CapturePhase::prefill ? std::min(run_count_, context_rows) : run_count_;
        start_ = phase == CapturePhase::prefill ? run_start_ + run_count_ - count_ : run_start_;
        ++owner_.generation;
        operation_.generation = int64_t(owner_.generation);
        operation_.frontier = phase == CapturePhase::prefill ? run_start_ + run_count_ : run_start_;
        keep_ = 0; seen_ = 0; active_ = true;
        lease_ = lease;
        return true;
    } catch (const std::exception& error) { err = error.what(); abort(); return false; }
}

bool CaptureProducer::consume(const TapView& view, std::string& err, const OperationLease::Guard* lease) {
    if (view.owner != owner_) {
        err = "dflash2 producer: stale target completion owner"; return false;
    }
    try {
        if (lease) lease_ = lease;
        require_capture(!poisoned_ && active_ && view.owner == owner_ && view.phase == phase_ &&
                        view.start == start_ && view.count == count_ && view.keep >= 1 && view.keep <= count_ &&
                        (phase_ == CapturePhase::verify || view.keep == count_) &&
                        view.residuals.size() == size_t(count_) * residual_width,
                        "stale/uncommitted or mismatched target tap");
        const auto tap = std::find(tap_layers.begin(), tap_layers.end(), view.tap_layer);
        require_capture(tap != tap_layers.end() && !(seen_ & (1u << (tap - tap_layers.begin()))),
                        "missing/duplicate global layer ownership");
        if (!seen_) {
            keep_ = view.keep;
            // Rejected rows are neither reduced nor stored, even tentatively. In
            // particular, a nonfinite rejected suffix cannot poison good context.
            store_.begin(operation_, start_, keep_, phase_, lease_);
            std::iota(positions_.begin(), positions_.begin() + keep_, start_);
        }
        require_capture(view.keep == keep_, "stage commit counts disagree");
        store_.capture(operation_, view.tap_layer, view.materialized_layer,
                       {positions_.data(), size_t(keep_)}, view.point,
                       view.residuals.first(size_t(keep_) * residual_width));
        seen_ |= 1u << (tap - tap_layers.begin());
        return true;
    } catch (const std::exception& error) { err = error.what(); abort(); return false; }
}
bool CaptureProducer::publish(std::string& err) {
    try {
        require_capture(!poisoned_ && active_ && seen_ == (1u << tap_layers.size()) - 1,
                        "cannot publish without every completed global tap");
        store_.commit(operation_, keep_);
        const int target_keep = phase_ == CapturePhase::prefill ? run_count_ : keep_;
        frontier_ = run_start_ + target_keep;
        committed_rows_ += target_keep;
        rejected_rows_ += run_count_ - target_keep;
        active_ = false; lease_ = nullptr;
        return true;
    } catch (const std::exception& error) { err = error.what(); abort(); return false; }
}
bool CaptureProducer::cancel(std::string& error) {
    if (poisoned_ || !active_ || seen_ || store_.capturing()) {
        error = "dflash2 producer: cannot discard partially consumed/poisoned capture"; return false;
    }
    active_ = false; lease_ = nullptr;
    return true;
}
} // namespace strata::core::dflash2
