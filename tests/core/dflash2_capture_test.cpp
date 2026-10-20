#include "strata/core/dflash2_capture.hpp"

#include <array>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace d = strata::core::dflash2;
namespace {
int checks = 0;
void check(bool ok, const char* why) { ++checks; if (!ok) throw std::runtime_error(why); }
d::CaptureOwner owner(uint64_t generation = 1) { return {7, 9, generation}; }
d::TapCaptureState state(d::CapturePhase phase, int lb = 0, int le = 48) {
    d::TapCaptureState result;
    std::string err;
    check(result.configure(phase, phase == d::CapturePhase::verify ? 4 : 2047,
                           lb, le, 2560, 4, 48, err), "configure valid capture");
    return result;
}

void verification() {
    std::array<float, 4 * d::residual_width> residuals{};
    for (int keep = 1; keep <= 4; ++keep) {
        auto capture = state(d::CapturePhase::verify);
        std::string err;
        d::TapView view;
        check(capture.staging_bytes() == 819200, "bounded five-tap verifier staging");
        check(capture.arm(owner(), 37, 37, 4, err), "arm at frontier");
        check(!capture.view(owner(), 0, residuals.data(), view, err), "no armed host view");
        check(!capture.begin_run(8, 37, 4, err), "reject wrong request");
        check(!capture.begin_run(7, 38, 4, err), "reject shifted run");
        check(capture.begin_run(7, 37, 4, err) && capture.copying(), "exact run enables replay copy");
        check(!capture.arm(owner(2), 37, 37, 4, err), "no overlapping owner");
        check(capture.finish_run(err), "verified graph completion");
        check(!capture.view(owner(), 0, residuals.data(), view, err), "verification is not target commit");
        check(!capture.waited(err), "wait before commit is refused");
        check(!capture.committed(0, err) && !capture.committed(5, err), "commit only verified prefix");
        check(capture.committed(keep, err), "record accepted prefix");
        check(!capture.view(owner(), 0, residuals.data(), view, err), "async commit is not host-complete");
        check(capture.waited(err), "successful wait publishes staging");
        for (int tap = 0; tap < 5; ++tap) {
            check(capture.view(owner(), tap, residuals.data(), view, err), "completed owned tap view");
            check(view.start == 37 && view.count == 4 && view.keep == keep &&
                  view.tap_layer == d::tap_layers[size_t(tap)] && view.materialized_layer == view.tap_layer + 1 &&
                  view.point == d::CapturePoint::next_layer_first_gr_read_group &&
                  view.phase == d::CapturePhase::verify && view.residuals.data() == residuals.data() &&
                  view.residuals.size() == residuals.size(), "exact verifier tap convention and accepted prefix");
        }
        auto stale = owner(); ++stale.incarnation;
        check(!capture.view(stale, 0, residuals.data(), view, err), "incarnation fenced");
        check(!capture.release(stale, err), "stale release cannot free staging");
        check(!capture.begin_run(7, 41, 4, err), "completed views block scratch reuse");
        check(capture.release(owner(), err) && capture.idle(), "owner releases views");
        check(!capture.view(owner(), 0, residuals.data(), view, err), "released view cannot be acquired again");
        check(!capture.arm(owner(), 37, 37, 4, err), "generation cannot replay");
        check(capture.arm(owner(2), 37 + keep, 37 + keep, 4, err), "next window at advanced frontier");
        check(capture.begin_run(7, 37 + keep, 4, err), "next generation owns staging");
        capture.poison();
        check(!capture.finish_run(err) && !capture.waited(err) && !capture.release(owner(2), err), "failure never publishes");
    }
}

void frontier() {
    auto capture = state(d::CapturePhase::verify);
    std::string err;
    d::TapView view;
    float unused = 0;
    check(!capture.arm(owner(), 100, 99, 2, err), "bootstrap cannot offer a proposal suffix");
    check(!capture.arm(owner(), 100, 98, 1, err), "no capture before bootstrap");
    check(!capture.arm(owner(), 100, 101, 1, err), "no capture beyond frontier");
    check(!capture.arm(owner(), 100, 100, 5, err), "no lookup/merged window width");
    check(!capture.arm({0, 9, 1}, 100, 100, 4, err), "unbound identity refused");
    check(!capture.arm({7, 9, 0}, 100, 100, 4, err), "zero generation refused");
    check(!capture.arm(owner(), d::max_position, d::max_position, 1, err), "position overflow refused");
    check(capture.arm(owner(), 100, 99, 1, err), "p=n-1 bootstrap is legal");
    check(capture.begin_run(7, 99, 1, err) && !capture.copying(), "bootstrap disables all copy nodes");
    check(capture.slice(99, 1).count == 0, "bootstrap has no tentative tap range");
    check(capture.finish_run(err) && capture.committed(1, err) && capture.waited(err), "bootstrap target commit");
    check(!capture.view(owner(), 0, &unused, view, err), "prompt row cannot be recaptured at bootstrap");
    check(capture.release(owner(), err), "release bootstrap ticket");
    check(capture.arm(owner(2), 100, 100, 4, err) && capture.begin_run(7, 100, 4, err) && capture.copying(),
          "first proposal capture begins at p=n");
}

void prefill_tail() {
    for (int count : {1, 2047, 2048, 4097}) for (int chunk : {1024, 2048, 4096}) {
        auto capture = state(d::CapturePhase::prefill);
        std::string err;
        check(capture.staging_bytes() == 419225600, "bounded raw prefill staging, not reduced feature bytes");
        check(!capture.arm(owner(), 37 + count - 1, 37, count, err), "prefill frontier is end of run");
        check(capture.arm(owner(), 37 + count, 37, count, err), "prefill arm whole absolute interval");
        check(capture.begin_run(7, 37, count, err), "prefill exact run");
        int copied = 0;
        for (int row = 0; row < count; row += chunk) {
            const int n = std::min(chunk, count - row);
            const auto part = capture.slice(37 + row, n);
            const int origin = 37 + count - std::min(count, 2047);
            const int first = std::max(37 + row, origin), end = std::min(37 + row + n, 37 + count);
            check(part.count == std::max(0, end - first), "chunk intersects only retained tail");
            if (!part.count) continue;
            check(part.source_row == first - 37 - row && part.staging_row == copied, "absolute row staging order");
            for (int tap = 0; tap < 5; ++tap) check(capture.record(tap, part, err), "prefill copied owned tap rows");
            copied += part.count;
        }
        check(copied == std::min(count, 2047), "2048-row chunks retain 2047 rows, no extra row");
        check(capture.finish_run(err) && capture.release(owner(), err), "complete prefill tail releases once");
    }
    auto capture = state(d::CapturePhase::prefill);
    std::string err;
    std::array<float, 3 * d::residual_width> rows{};
    d::TapView view;
    check(capture.arm(owner(), 40, 37, 3, err) && capture.begin_run(7, 37, 3, err), "small prefill");
    const auto part = capture.slice(37, 3);
    for (int tap = 0; tap < 5; ++tap) check(capture.record(tap, part, err), "all small prefill taps");
    check(!capture.record(0, part, err), "duplicate tap rows refused");
    check(capture.finish_run(err) && capture.view(owner(), 0, rows.data(), view, err), "prefill host completion");
    check(view.start == 37 && view.count == 3 && view.keep == 3 && view.tap_layer == 5 &&
          view.materialized_layer == 5 && view.point == d::CapturePoint::completed_second_residual_write &&
          view.phase == d::CapturePhase::prefill, "prefill materializes the tap itself");
    auto missing = state(d::CapturePhase::prefill);
    check(missing.arm(owner(), 40, 37, 3, err) && missing.begin_run(7, 37, 3, err), "missing tap fixture");
    check(missing.record(0, missing.slice(37, 3), err) && !missing.finish_run(err), "missing taps cannot publish");
    missing.poison();
    check(!missing.view(owner(), 0, rows.data(), view, err), "partial failures expose no tap");
}

void split_stages() {
    std::string err;
    size_t verify_bytes = 0, prefill_bytes = 0;
    for (const auto bounds : {std::array{0, 20}, std::array{20, 39}, std::array{39, 48}}) {
        auto verify = state(d::CapturePhase::verify, bounds[0], bounds[1]);
        auto capture = state(d::CapturePhase::prefill, bounds[0], bounds[1]);
        verify_bytes += verify.staging_bytes(); prefill_bytes += capture.staging_bytes();
        check(capture.arm(owner(), 4134, 37, 4097, err), "split stage shares whole prompt ticket");
        for (int row = 0; row < 4097; row += 2048) {
            const int n = std::min(2048, 4097 - row);
            check(capture.begin_run(7, 37 + row, n, err), "successor owns consecutive sub-run");
            // A successor may re-chunk the upstream hand-off even further.
            for (int offset = 0; offset < n; offset += 512) {
                const auto part = capture.slice(37 + row + offset, std::min(512, n - offset));
                if (!part.count) continue;
                for (int tap = 0; tap < 5; ++tap) {
                    check(capture.owns_tap(tap) == (d::tap_layers[size_t(tap)] >= bounds[0] &&
                          d::tap_layers[size_t(tap)] < bounds[1]), "global tap attribution by owning stage");
                    if (capture.owns_tap(tap)) check(capture.record(tap, part, err), "stage-local rows in absolute order");
                }
            }
            check(capture.finish_run(err), "sub-run stream completion");
            if (row + n < 4097) check(!capture.release(owner(), err), "intermediate chunk cannot publish/release whole tail");
        }
        check(capture.release(owner(), err), "last split sub-run completes retained tail");
    }
    check(verify_bytes == 819200 && prefill_bytes == 419225600, "split ownership allocates each tap once");
    for (int tap : d::tap_layers) {
        d::TapCaptureState bad;
        check(!bad.configure(d::CapturePhase::verify, 4, 0, tap + 1, 2560, 4, 48, err), "boundary tap is refused");
        auto prefill = state(d::CapturePhase::prefill, 0, tap + 1);
        (void) prefill; // prefill applied the write already, even at the stage's end
    }
    d::TapCaptureState wrong;
    check(!wrong.configure(d::CapturePhase::verify, 4, 0, 48, 2560, 3, 48, err), "four-stream residual geometry required");
    auto unarmed = state(d::CapturePhase::verify);
    check(unarmed.begin_run(7, 99, 8, err) && !unarmed.copying() && unarmed.finish_run(err), "unarmed observer stays off");
}
} // namespace

int main() {
    try {
        verification(); frontier(); prefill_tail(); split_stages();
        std::printf("dflash2 capture: %d CPU checks passed\n", checks);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "dflash2 capture: %s\n", error.what());
        return 1;
    }
}
