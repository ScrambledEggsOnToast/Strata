#pragma once

#include <deque>
#include <string>
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
} // namespace strata::program
