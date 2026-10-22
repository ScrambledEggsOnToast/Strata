#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace strata::program {

// Optional, fixed-size host observation. No logging occurs while the coordinator
// is serving; destruction emits the retained records and an explicit completeness
// result. An overflow or unfinished window is not a representative sample.
class BatchReadyTrace {
public:
    static constexpr size_t capacity = 1024;
    static constexpr size_t max_stages = 4;
    static constexpr size_t max_rows = 8;
    static constexpr uint64_t output_budget_bytes = capacity * 4096;

    struct Stage {
        uintptr_t owner = 0;
        int64_t launch_begin_ns = 0, launch_end_ns = 0, polled_done_ns = 0;
        int64_t eligible_ns = 0;
        uint64_t retirement_witness = 0;
    };
    struct Window {
        uint64_t sequence = 0;
        int group = -1, count = 0;
        std::array<int, max_rows> rows{};
        std::array<int64_t, max_rows> positions{};
        std::array<uint64_t, max_rows> requests{}, incarnations{};
        std::array<Stage, max_stages> stages{};
        uint64_t active_mask = 0, selected_mask = 0;
        int64_t retirement_begin_ns = 0, retirement_end_ns = 0;
        uint64_t retirement_inflight_mask = 0, retirement_active_mask = 0, retirement_leased_mask = 0;
        int retirement_blocked_stage = -1;
        bool retirement_reader = false;
    };

    explicit BatchReadyTrace(int stages) : stage_count_(stages) {}
    BatchReadyTrace(const BatchReadyTrace&) = delete;
    BatchReadyTrace& operator=(const BatchReadyTrace&) = delete;

    Window* begin(int group, int count) {
        ++seen_;
        if (used_ == capacity) return nullptr;
        auto& result = windows_[used_++];
        result.sequence = seen_;
        result.group = group;
        result.count = count;
        return &result;
    }

    bool complete() const {
        if (invalid || seen_ != used_ || stage_count_ < 1 || stage_count_ > int(max_stages)) return false;
        for (size_t i = 0; i < used_; ++i) {
            const auto& window = windows_[i];
            if (window.count < 1 || window.count > int(max_rows) || !window.retirement_begin_ns ||
                window.retirement_end_ns < window.retirement_begin_ns) return false;
            for (int k = 0; k < stage_count_; ++k) {
                const auto& stage = window.stages[size_t(k)];
                if (!stage.owner || !stage.launch_begin_ns || stage.launch_end_ns < stage.launch_begin_ns ||
                    stage.polled_done_ns < stage.launch_end_ns) return false;
            }
        }
        return true;
    }

    ~BatchReadyTrace() {
        for (size_t i = 0; i < used_; ++i) {
            const auto& w = windows_[i];
            std::fprintf(stderr, "BATCH_READY_TRACE {\"kind\":\"window\",\"sequence\":%llu,\"group\":%d,"
                "\"rows\":%d,\"active_mask\":%llu,\"selected_mask\":%llu,\"retirement_begin_ns\":%lld,"
                "\"retirement_end_ns\":%lld,\"retirement_inflight_mask\":%llu,\"retirement_active_mask\":%llu,"
                "\"retirement_leased_mask\":%llu,\"retirement_blocked_stage\":%d,\"retirement_reader\":%d,\"seats\":[",
                (unsigned long long)w.sequence, w.group, w.count, (unsigned long long)w.active_mask,
                (unsigned long long)w.selected_mask, (long long)w.retirement_begin_ns,
                (long long)w.retirement_end_ns, (unsigned long long)w.retirement_inflight_mask,
                (unsigned long long)w.retirement_active_mask, (unsigned long long)w.retirement_leased_mask,
                w.retirement_blocked_stage, int(w.retirement_reader));
            for (int t = 0; t < w.count && t < int(max_rows); ++t)
                std::fprintf(stderr, "%s{\"slot\":%d,\"position\":%lld,\"request\":%llu,\"incarnation\":%llu}",
                    t ? "," : "", w.rows[size_t(t)], (long long)w.positions[size_t(t)],
                    (unsigned long long)w.requests[size_t(t)], (unsigned long long)w.incarnations[size_t(t)]);
            std::fprintf(stderr, "],\"stages\":[");
            for (int k = 0; k < stage_count_ && k < int(max_stages); ++k) {
                const auto& stage = w.stages[size_t(k)];
                std::fprintf(stderr, "%s{\"stage\":%d,\"owner\":%llu,\"launch_begin_ns\":%lld,"
                    "\"launch_end_ns\":%lld,\"polled_done_ns\":%lld,\"eligible_ns\":%lld,\"retirement_witness\":%llu}",
                    k ? "," : "", k, (unsigned long long)stage.owner, (long long)stage.launch_begin_ns,
                    (long long)stage.launch_end_ns, (long long)stage.polled_done_ns,
                    (long long)stage.eligible_ns, (unsigned long long)stage.retirement_witness);
            }
            std::fprintf(stderr, "]}\n");
        }
        const bool output_ok = std::ferror(stderr) == 0;
        std::fprintf(stderr, "BATCH_READY_TRACE {\"kind\":\"summary\",\"seen\":%llu,\"retained\":%zu,"
            "\"capacity\":%zu,\"host_bytes\":%zu,\"output_budget_bytes\":%llu,\"complete\":%d}\n",
            (unsigned long long)seen_, used_, capacity, sizeof(*this),
            (unsigned long long)output_budget_bytes, int(output_ok && complete()));
    }

    bool invalid = false;

private:
    int stage_count_;
    uint64_t seen_ = 0;
    size_t used_ = 0;
    std::array<Window, capacity> windows_{};
};

} // namespace strata::program
