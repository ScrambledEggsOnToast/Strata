#include "strata/core/dflash2_capture.hpp"
#include "strata/kernels/bf16_bits.hpp"

#include <array>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

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
    std::array<float, d::residual_width> residuals{};
    check(!capture.arm(owner(), 100, 99, 2, err), "bootstrap cannot offer a proposal suffix");
    check(!capture.arm(owner(), 100, 98, 1, err), "no capture before bootstrap");
    check(!capture.arm(owner(), 100, 101, 1, err), "no capture beyond frontier");
    check(!capture.arm(owner(), 100, 100, 5, err), "no lookup/merged window width");
    check(!capture.arm({0, 9, 1}, 100, 100, 4, err), "unbound identity refused");
    check(!capture.arm({7, 9, 0}, 100, 100, 4, err), "zero generation refused");
    check(!capture.arm(owner(), d::max_position, d::max_position, 1, err), "position overflow refused");
    check(!capture.arm(owner(), 100, 99, 1, err), "reject prompt-length frontier that would drop bootstrap");
    check(capture.arm(owner(), 99, 99, 1, err), "bootstrap owns the next unread input row");
    check(capture.begin_run(7, 99, 1, err) && capture.copying(), "bootstrap enables all owned tap copy nodes");
    check(capture.slice(99, 1).count == 1, "bootstrap has exactly one tentative row");
    check(capture.finish_run(err) && capture.committed(1, err) && capture.waited(err), "bootstrap target commit");
    check(capture.view(owner(), 0, residuals.data(), view, err) && view.start == 99 && view.keep == 1,
          "bootstrap is captured exactly once after target commit");
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

float feature_value(int position, int tap) { return float(position % 97 + tap * 128 + 3); }
void producer_split_tail() {
    std::string err;
    d::CaptureProducer producer;
    check(producer.reset(7, 9, 0, err), "bind reusable owned feature storage");
    const size_t storage = producer.store().storage_bytes();
    check(storage == 2 * d::max_feature_bytes + d::context_rows * sizeof(int64_t),
          "feature payload is exactly 25600 bytes/token plus separately priced generation tags");
    std::array<d::TapCaptureState, 3> pf, vr;
    constexpr std::array<std::array<int, 2>, 3> bounds{{{0, 20}, {20, 39}, {39, 48}}};
    for (size_t stage = 0; stage < bounds.size(); ++stage) {
        pf[stage] = state(d::CapturePhase::prefill, bounds[stage][0], bounds[stage][1]);
        vr[stage] = state(d::CapturePhase::verify, bounds[stage][0], bounds[stage][1]);
    }
    std::vector<int64_t> generations;
    auto complete = [&](d::CapturePhase phase, int start, int rows, int keep) {
        check(producer.begin(phase, start, rows, err), "begin contiguous actual target operation");
        const auto ticket = producer.owner();
        auto& stages = phase == d::CapturePhase::prefill ? pf : vr;
        const int frontier = phase == d::CapturePhase::prefill ? start + rows : start;
        for (auto& stage : stages) {
            check(stage.arm(ticket, frontier, start, rows, err), "same owner on every split stage");
            for (int offset = 0; offset < rows;) {
                const int n = phase == d::CapturePhase::prefill ? std::min(rows - offset, 2048) : rows;
                check(stage.begin_run(ticket.request, start + offset, n, err), "actual split sub-run");
                if (phase == d::CapturePhase::prefill) {
                    const auto slice = stage.slice(start + offset, n);
                    if (slice.count) for (int tap = 0; tap < 5; ++tap)
                        if (stage.owns_tap(tap)) check(stage.record(tap, slice, err), "copy each retained absolute row once");
                }
                check(stage.finish_run(err), "sub-run completes on owning stage");
                offset += n;
            }
            if (phase == d::CapturePhase::verify)
                check(stage.committed(keep, err) && stage.waited(err), "real target keep/drain determines publication");
        }
        const int captured = phase == d::CapturePhase::prefill ? std::min(rows, d::context_rows) : rows;
        const int first = phase == d::CapturePhase::prefill ? start + rows - captured : start;
        for (int tap : {4, 0, 3, 1, 2}) { // stage completions intentionally out of order
            int stage = 0;
            while (d::tap_layers[size_t(tap)] >= bounds[size_t(stage)][1]) ++stage;
            std::vector<float> residuals(size_t(captured) * d::residual_width);
            for (int row = 0; row < captured; ++row) for (int stream = 0; stream < 4; ++stream) {
                const float value = phase == d::CapturePhase::verify && row >= keep
                    ? std::numeric_limits<float>::quiet_NaN() // rejected rows must NEVER be reduced
                    : feature_value(first + row, tap) - 3.0f + float(2 * stream);
                std::fill_n(residuals.data() + size_t(row * 4 + stream) * d::hidden_size, d::hidden_size, value);
            }
            const auto before = residuals;
            d::TapView view;
            check(stages[size_t(stage)].view(ticket, tap, residuals.data(), view, err), "completed stage-owned tap");
            check(view.owner == ticket && producer.consume(view, err), "assemble by global layer/absolute position");
            check(std::memcmp(residuals.data(), before.data(), residuals.size() * sizeof(float)) == 0,
                  "observer preserves target local state including rejected rows");
            // Reuse/destroy the operation's raw buffer immediately. Publication
            // below must depend only on its owned BF16 copy, never this pointer.
            std::fill(residuals.begin(), residuals.end(), -1000.0f);
        }
        check(producer.publish(err), "publish all and only committed target input rows");
        for (auto& stage : stages) check(stage.release(ticket, err), "release every raw stage lease after owned copy");
        const int committed = phase == d::CapturePhase::prefill ? rows : keep;
        generations.insert(generations.end(), size_t(committed), int64_t(ticket.generation));
        check(producer.frontier() == start + committed && producer.store().frontier() == producer.frontier(),
              "committed frontier does not count evaluated/rejected rows");
        const auto features = producer.store().committed_since(producer.store().retained_start());
        check(features.request_id == "7" && features.incarnation == 9, "persistent row request/incarnation tags");
        int position = features.start;
        for (size_t segment = 0; segment < features.segments.size(); ++segment) {
            check(features.generations[segment].size() * d::feature_width == features.segments[segment].size(),
                  "generation tags follow exactly the same wrapped row extents");
            for (size_t row = 0; row < features.generations[segment].size(); ++row, ++position) {
                check(features.generations[segment][row] == generations[size_t(position)], "row generation survives ring wrap");
                for (int tap = 0; tap < 5; ++tap) {
                    const auto expected = strata::kernels::bf16_from_f32(feature_value(position, tap));
                    const auto values = features.segments[segment].subspan(row * d::feature_width +
                        size_t(tap) * d::hidden_size, d::hidden_size);
                    check(std::all_of(values.begin(), values.end(), [=](uint16_t bits) { return bits == expected; }),
                          "every retained position has exactly one correct feature for every global tap");
                }
            }
        }
        check(position == producer.frontier(), "no missing or duplicated committed positions");
        check(producer.store().storage_bytes() == storage, "no per-window store growth");
    };
    // A 4102-token prompt: batched tail, verifier prompt tail, and its previously
    // UNREAD bootstrap at n-1. The old prompt-length frontier skipped that row.
    complete(d::CapturePhase::prefill, 0, 4097, 4097);
    complete(d::CapturePhase::verify, 4097, 4, 4);
    complete(d::CapturePhase::verify, 4101, 1, 1);
    int start = 4102, rejected = 0;
    for (int keep : {2, 1, 3, 4}) {
        complete(d::CapturePhase::verify, start, 4, keep);
        start += keep; rejected += 4 - keep;
    }
    check(producer.committed_rows() == start && producer.rejected_rows() == rejected,
          "bootstrap and every accepted decode row counted once, rejected suffixes counted separately");
    // A later long segment replaces all old retained context, not a persistent gap.
    complete(d::CapturePhase::prefill, start, 2048, 2048);
    check(producer.reset(8, 10, 37, err) && producer.store().storage_bytes() == storage &&
          producer.store().committed_since(37).count == 0, "drained newer incarnation reuses allocations with no old rows");
    check(!producer.reset(7, 9, 0, err), "stale request completion cannot rebind recycled storage");
}

void producer_refusals() {
    for (int fault = 0; fault < 9; ++fault) {
        d::CaptureProducer producer;
        std::string err;
        check(producer.reset(7, 9, 37, err) && producer.begin(d::CapturePhase::verify, 37, 4, err),
              "owned refusal fixture");
        std::vector<float> residuals(4 * d::residual_width, 1.0f);
        d::TapView view{37, 4, 2, 5, 6, d::CapturePhase::verify,
            d::CapturePoint::next_layer_first_gr_read_group, residuals, producer.owner()};
        if (fault == 0) { check(!producer.publish(err), "missing taps refuse publication"); continue; }
        if (fault == 1) ++view.owner.request;
        if (fault == 2) ++view.owner.incarnation;
        if (fault == 3) ++view.owner.generation;
        if (fault == 4) ++view.start;
        if (fault == 5) view.materialized_layer = 5;
        if (fault == 6) view.keep = 0;
        if (fault <= 6) check(!producer.consume(view, err), "wrong tags/position/seam/commit refuse");
        if (fault >= 7) {
            check(producer.consume(view, err), "first correct global tap");
            if (fault == 8) { view.tap_layer = 15; view.materialized_layer = 16; view.keep = 3; }
            check(!producer.consume(view, err), "duplicate tap or disagreeing stage commit count refuses");
        }
        check(producer.store().poisoned() && producer.frontier() == 37, "failure publishes no partial persistent context");
    }
    d::CaptureProducer gap;
    std::string err;
    check(gap.reset(7, 9, 0, err) && !gap.begin(d::CapturePhase::verify, 1, 1, err),
          "cannot skip a committed input or use prompt length for bootstrap frontier");
    size_t pf = 0, vr = 0, flags = 0;
    for (const auto bounds : {std::array{0, 20}, std::array{20, 39}, std::array{39, 48}}) {
        d::CaptureAllocation allocation;
        check(d::CaptureAllocation::stage(bounds[0], bounds[1], allocation, err), "allocator-derived stage quote");
        pf += allocation.prefill; vr += allocation.verify; flags += allocation.flags;
    }
    check(d::feature_row_bytes == 25600 && pf == 419225600 && vr == 819200 && flags == 24576,
          "admission separates 50MiB BF16 context, 400MiB raw taps and per-stage mapped flags");
}
} // namespace

int main() {
    try {
        verification(); frontier(); prefill_tail(); split_stages(); producer_split_tail(); producer_refusals();
        std::printf("dflash2 capture: %d CPU checks passed\n", checks);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "dflash2 capture: %s\n", error.what());
        return 1;
    }
}
