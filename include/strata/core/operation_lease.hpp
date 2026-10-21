// A host-only scratch owner. Failed work stays poisoned even after its consumers drain.
#pragma once

#include <chrono>
#include <mutex>

namespace strata::core {
class OperationLease {
public:
    static constexpr auto wait_limit = std::chrono::milliseconds(2000);
    enum class Failure { none, busy, timeout, poisoned };
    class Guard {
    public:
        explicit Guard(OperationLease& owner, std::chrono::milliseconds wait = wait_limit)
            : owner_(owner), lock_(owner.mutex_, std::defer_lock) {
            if (!lock_.try_lock_for(wait)) failure_ = Failure::timeout;
            else check_poison();
        }
        Guard(OperationLease& owner, std::try_to_lock_t)
            : owner_(owner), lock_(owner.mutex_, std::try_to_lock) {
            if (!lock_) failure_ = Failure::busy;
            else check_poison();
        }
        ~Guard() {
            if (lock_ && !complete_) owner_.poisoned_ = true;
        }
        explicit operator bool() const { return failure_ == Failure::none; }
        Failure failure() const { return failure_; }
        bool holds(const OperationLease& owner) const {
            return &owner_ == &owner && lock_.owns_lock() && failure_ == Failure::none;
        }
        // Keep ownership through the caller's final scratch reads. Destruction
        // unlocks; draining or a timeout alone never repairs poisoned state.
        void complete() { complete_ = true; }
    private:
        void check_poison() {
            if (owner_.poisoned_) { failure_ = Failure::poisoned; complete_ = true; }
        }
        OperationLease& owner_;
        std::unique_lock<std::timed_mutex> lock_;
        Failure failure_ = Failure::none;
        bool complete_ = false;
    };
private:
    std::timed_mutex mutex_;
    bool poisoned_ = false;
};
} // namespace strata::core
