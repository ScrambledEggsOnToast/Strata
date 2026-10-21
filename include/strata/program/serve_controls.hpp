#pragma once

#include <deque>
#include <string>
#include <string_view>
#include <utility>

namespace strata::program {
// Caller holds the input mutex. Slot-number controls must never overtake an
// admission: the same number can refer to a different resident incarnation.
inline bool pop_control_prefix(std::deque<std::string>& pending, std::string& command, bool consume_yield) {
    if (pending.empty()) return false;
    const auto& front = pending.front();
    if (front.rfind("BSTOP ", 0) != 0 && front.rfind("BPRIORITY ", 0) != 0 &&
        (!consume_yield || front.rfind("BYIELD", 0) != 0)) return false;
    command = std::move(pending.front());
    pending.pop_front();
    return true;
}

// Caller holds the input mutex and has classified this as a native control.
// A stale seat/incarnation/cursor must not erase a valid pending intention.
inline void coalesce_native_control(std::deque<std::string>& pending, std::string_view text) {
    const bool reprioritize = text.starts_with("BQPRIORITY ");
    size_t fence_end = text.find(' ', text.find(' ') + 1);
    for (int field = 0; reprioritize && field < 3; ++field) {
        const size_t begin = text.find_first_not_of(" \t", fence_end);
        fence_end = begin == text.npos ? text.npos : text.find_first_of(" \t", begin);
        if (fence_end == text.npos) break;
    }
    const auto prefix = text.substr(0, fence_end);
    for (auto it = pending.begin(); it != pending.end(); ++it) {
        const bool same = reprioritize ?
            it->compare(0, prefix.size(), prefix) == 0 && it->size() > prefix.size() &&
                ((*it)[prefix.size()] == ' ' || (*it)[prefix.size()] == '\t') :
            *it == text;
        if (same) { pending.erase(it); break; }
    }
}
} // namespace strata::program
