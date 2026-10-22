#include "strata/core/dflash2_capture.hpp"

#include <numeric>
#include <charconv>

namespace strata::core::dflash2 {
namespace {
void require_capture(bool ok, const char* message) {
    if (!ok) throw Refusal(std::string("dflash2 producer: ") + message);
}
}
DeviceTapView::DeviceTapView(const TapCaptureState* state, CaptureOwner owner, int tap,
        const TapView& view, const uint16_t* data, int device)
    : state_(state), owner_(owner), tap_(tap), start_(view.start), count_(view.count),
      keep_(view.keep), device_(device), data_(data), mirror_(view.reduced.data()) {}
bool DeviceTapView::matches(const TapView& view, std::string& error) const {
    if (!state_ || !data_ || !mirror_ || device_ < 0 || tap_ < 0 || tap_ >= int(tap_layers.size()) ||
        view.owner != owner_ || view.start != start_ || view.count != count_ || view.keep != keep_ ||
        view.tap_layer != tap_layers[size_t(tap_)] || view.materialized_layer != view.tap_layer + 1 ||
        view.phase != CapturePhase::verify || view.point != CapturePoint::next_layer_first_gr_read_group ||
        !view.residuals.empty() || view.reduced.data() != mirror_ ||
        view.reduced.size() != size_t(count_) * hidden_size) {
        error = "dflash2 producer: device capability differs from its physical verifier tap"; return false;
    }
    return current(error);
}
bool DeviceTapView::current(std::string& error) const {
    TapView current_view;
    if (!state_ || !state_->view_reduced(owner_, tap_, reinterpret_cast<const float*>(mirror_),
                                         current_view, error)) return false;
    if (current_view.phase != CapturePhase::verify || current_view.start != start_ ||
        current_view.count != count_ || current_view.keep != keep_) {
        error = "dflash2 producer: physical verifier capability was released or rebound"; return false;
    }
    return true;
}
CaptureProducer::CaptureProducer() : store_("unbound", 0) { operation_.request_id.reserve(128); }
CaptureProducer::CaptureProducer(std::span<uint16_t> staging, OperationLease& lease)
    : common_lease_(&lease), store_("unbound", 0, 0, staging, lease) { operation_.request_id.reserve(128); }
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
        device_seen_ = 0; device_taps_ = {}; device_lease_ = nullptr;
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
        device_seen_ = 0; device_taps_ = {}; device_lease_ = nullptr;
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
        device_seen_ = 0; device_taps_ = {}; device_lease_ = nullptr;
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
                        ((view.reduced.empty() && view.residuals.size() == size_t(count_) * residual_width) ||
                         (view.residuals.empty() &&
                          view.reduced.size() == size_t(count_) * hidden_size)),
                        "stale/uncommitted or mismatched target tap");
        const auto tap = std::find(tap_layers.begin(), tap_layers.end(), view.tap_layer);
        require_capture(tap != tap_layers.end() && !(seen_ & (1u << (tap - tap_layers.begin()))),
                        "missing/duplicate global layer ownership");
        if (view.device.present()) {
            require_capture(common_lease_ && lease_ && lease_->holds(*common_lease_),
                            "device tap requires its actual common operation lease");
            if (!view.device.matches(view, err)) { abort(); return false; }
        }
        if (!seen_) {
            keep_ = view.keep;
            // Rejected rows are never validated or stored, even tentatively.
            // Device reduction may evaluate them, but no gather can transfer them.
            store_.begin(operation_, start_, keep_, phase_, lease_);
            std::iota(positions_.begin(), positions_.begin() + keep_, start_);
        }
        require_capture(view.keep == keep_, "stage commit counts disagree");
        store_.capture(operation_, view.tap_layer, view.materialized_layer,
                       {positions_.data(), size_t(keep_)}, view.point,
                       view.residuals.empty() ? std::span<const float>{} :
                           view.residuals.first(size_t(keep_) * residual_width),
                       view.reduced.empty() ? std::span<const uint16_t>{} :
                           view.reduced.first(size_t(keep_) * hidden_size));
        seen_ |= 1u << (tap - tap_layers.begin());
        if (view.device.present()) {
            const size_t index = size_t(tap - tap_layers.begin());
            device_taps_[index] = view.device;
            device_seen_ |= 1u << index;
        }
        return true;
    } catch (const std::exception& error) { err = error.what(); abort(); return false; }
}
bool CaptureProducer::publish(std::string& err) {
    try {
        require_capture(!poisoned_ && active_ && seen_ == (1u << tap_layers.size()) - 1,
                        "cannot publish without every completed global tap");
        require_capture(!device_seen_ || device_seen_ == seen_, "mixed host/device verifier tap frame");
        store_.commit(operation_, keep_);
        const int target_keep = phase_ == CapturePhase::prefill ? run_count_ : keep_;
        frontier_ = run_start_ + target_keep;
        committed_rows_ += target_keep;
        rejected_rows_ += run_count_ - target_keep;
        device_lease_ = device_seen_ ? lease_ : nullptr;
        active_ = false; lease_ = nullptr;
        return true;
    } catch (const std::exception& error) { err = error.what(); abort(); return false; }
}
bool CaptureProducer::cancel(std::string& error) {
    if (poisoned_ || !active_ || seen_ || store_.capturing()) {
        error = "dflash2 producer: cannot discard partially consumed/poisoned capture"; return false;
    }
    active_ = false; lease_ = nullptr;
    device_seen_ = 0; device_taps_ = {}; device_lease_ = nullptr;
    return true;
}
bool CaptureProducer::device_preflight(const FeatureView& features, const OperationLease::Guard& lease,
                                        std::string& error) const {
    try {
        require_capture(!poisoned_ && !active_ && phase_ == CapturePhase::verify &&
                        device_seen_ == (1u << tap_layers.size()) - 1 &&
                        common_lease_ && device_lease_ == &lease && lease.holds(*common_lease_) &&
                        features.start == start_ && features.count == keep_ &&
                        frontier_ == start_ + keep_, "device features lack their completed producer/lease/interval");
        const auto actual = store_.committed_since(start_);
        require_capture(actual.start == features.start && actual.count == features.count &&
                        actual.request_id == features.request_id && actual.incarnation == features.incarnation,
                        "device features differ from the actual committed producer");
        for (size_t part = 0; part < actual.segments.size(); ++part) {
            require_capture(features.segments[part].data() == actual.segments[part].data() &&
                            features.segments[part].size() == actual.segments[part].size() &&
                            features.generations[part].data() == actual.generations[part].data() &&
                            features.generations[part].size() == actual.generations[part].size(),
                            "device features borrow a foreign host mirror or generation ring");
            for (const int64_t generation : actual.generations[part])
                require_capture(generation == int64_t(owner_.generation), "device capture generation differs from mirror");
        }
        for (size_t tap = 0; tap < device_taps_.size(); ++tap) {
            const auto& view = device_taps_[tap];
            require_capture(view.owner_ == owner_ && view.tap_ == int(tap) && view.start_ == start_ &&
                            view.count_ == count_ && view.keep_ == keep_, "device tap has a foreign owner or accepted range");
            if (!view.current(error)) return false;
        }
        return true;
    } catch (const std::exception& failure) { error = failure.what(); return false; }
}
} // namespace strata::core::dflash2
