// Weighted decode service with bounded background progress. Pure: no GPU, session or wire state.
//
// Units: one ticket is one decode window of service for one logical request. Every ready request appears
// foreground_weight times (3) or background_weight times (1) in a service cycle, so a request receives at most one
// row per window and overlapping work is never double-counted. The cursor advances past the examined part of the
// cycle, which bounds a background request's wait by (cycle length - 1) windows when capacity is one; with capacity
// covering every ready request, every request is served every window.
//
// The scheduler selects *requests*, never physical slots: a seat reused by a new request is a different participant.
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::program {

struct DecodeParticipant {
    uint64_t request = 0;      ///< stable logical request identity
    bool foreground = true;
};

// The rows one decode window admits. A resident background request cannot enlarge the interactive window; when no
// interactive request is ready, backgrounds are not artificially serialized.
inline int decode_window_capacity(const std::vector<DecodeParticipant>& ready, int max_rows) {
    int foreground = 0;
    for (const auto& p : ready) foreground += p.request && p.foreground ? 1 : 0;
    if (foreground == 0) return std::min(max_rows, (int) ready.size());
    return std::max(1, std::min(max_rows, foreground));
}

class DecodeService {
public:
    DecodeService(int capacity, unsigned foreground_weight = 3, unsigned background_weight = 1)
        : capacity_(std::max(0, capacity)),
          foreground_weight_(std::max(1u, foreground_weight)),
          background_weight_(std::max(1u, background_weight)) {}

    void set_weights(unsigned foreground_weight, unsigned background_weight) {
        foreground_weight_ = std::max(1u, foreground_weight);
        background_weight_ = std::max(1u, background_weight);
        schedule_.clear();   // rebuilt from the next ready set
        cursor_ = 0;
    }

    // Choose up to `capacity()` distinct ready requests for this window, in service-cycle order. `ready` must be in a
    // stable order; priority changes and membership changes rebuild the cycle without resetting the phase when the
    // request at the cursor still exists. A request is returned at most once per call.
    std::vector<uint64_t> select(const std::vector<DecodeParticipant>& ready) {
        std::vector<uint64_t> chosen;
        if (capacity_ <= 0) return chosen;
        if (!same_members(ready)) rebuild(ready);
        if (schedule_.empty()) return chosen;
        const uint64_t resume = cursor_ < schedule_.size() ? schedule_[cursor_] : schedule_.front();
        size_t examined = 0;
        while (examined < schedule_.size() && (int) chosen.size() < capacity_) {
            const uint64_t request = schedule_[(cursor_ + examined) % schedule_.size()];
            ++examined;
            if (std::find(chosen.begin(), chosen.end(), request) != chosen.end()) continue;
            if (!is_ready(ready, request)) continue;   // membership changed between calls: skip without a ticket
            chosen.push_back(request);
        }
        cursor_ = (cursor_ + examined) % schedule_.size();
        (void) resume;   // only used by rebuild() to keep the phase across a membership change
        return chosen;
    }

    // Interactive requests are never displaced by resident background work. With any interactive request ready, the
    // window carries at most one row per interactive request (the cycle still gives backgrounds their weighted turn);
    // with only backgrounds, every ready request shares the window normally.
    void set_capacity(int capacity) { capacity_ = std::max(0, capacity); }
    int capacity() const { return capacity_; }
    unsigned foreground_weight() const { return foreground_weight_; }
    unsigned background_weight() const { return background_weight_; }
    size_t cycle_length() const { return schedule_.size(); }
    // The longest a background request can wait between turns when capacity is one, in windows.
    size_t background_age_bound() const { return schedule_.empty() ? 0 : schedule_.size(); }

private:
    static bool is_ready(const std::vector<DecodeParticipant>& ready, uint64_t request) {
        return std::any_of(ready.begin(), ready.end(),
                           [&](const DecodeParticipant& p) { return p.request == request; });
    }
    bool same_members(const std::vector<DecodeParticipant>& ready) const {
        if (schedule_.empty()) return ready.empty();
        // The cycle repeats each request by weight; compare the distinct set and each weight.
        std::vector<uint64_t> members;
        for (const auto& p : ready) {
            if (p.request && std::find(members.begin(), members.end(), p.request) == members.end())
                members.push_back(p.request);
        }
        std::vector<uint64_t> cycle;
        for (const uint64_t r : schedule_)
            if (std::find(cycle.begin(), cycle.end(), r) == cycle.end()) cycle.push_back(r);
        if (members.size() != cycle.size()) return false;
        for (size_t i = 0; i < members.size(); ++i)
            if (members[i] != cycle[i]) return false;
        for (const auto& p : ready) {
            if (!p.request) continue;
            const unsigned want = p.foreground ? foreground_weight_ : background_weight_;
            unsigned have = 0;
            for (const uint64_t r : schedule_) have += r == p.request ? 1u : 0u;
            if (have != want) return false;
        }
        return true;
    }
    void rebuild(const std::vector<DecodeParticipant>& ready) {
        const uint64_t resume = cursor_ < schedule_.size() ? schedule_[cursor_] : 0;
        // Preserve the position *within* that request's ticket block, not merely its first ticket: a
        // membership change must not reset a background request's turn to the start of the foreground block.
        size_t offset = 0;
        if (resume)
            for (size_t i = 0; i < cursor_ && i < schedule_.size(); ++i)
                offset += schedule_[i] == resume ? 1 : 0;
        schedule_.clear();
        for (const auto& p : ready) {
            if (!p.request) continue;
            const unsigned weight = p.foreground ? foreground_weight_ : background_weight_;
            for (unsigned i = 0; i < weight; ++i) schedule_.push_back(p.request);
        }
        cursor_ = 0;
        if (resume) {
            const auto it = std::find(schedule_.begin(), schedule_.end(), resume);
            if (it != schedule_.end()) {
                unsigned weight = 0;
                for (const uint64_t r : schedule_) weight += r == resume ? 1u : 0u;
                cursor_ = (size_t) std::distance(schedule_.begin(), it) + std::min(offset, (size_t) weight - 1);
            }
        }
    }

    int capacity_;
    unsigned foreground_weight_, background_weight_;
    std::vector<uint64_t> schedule_;
    size_t cursor_ = 0;
};

} // namespace strata::program
