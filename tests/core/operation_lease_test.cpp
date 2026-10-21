#include "strata/core/operation_lease.hpp"

#include <atomic>
#include <cstdio>
#include <future>
#include <stdexcept>
#include <thread>

using strata::core::OperationLease;
using namespace std::chrono_literals;

static void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}


int main() {
    try {
        OperationLease lease;
        std::future<bool> waiter;
        std::promise<void> entered;
        std::atomic<bool> acquired{false};
        {
            OperationLease::Guard first(lease);
            require(bool(first), "initial lease refused");
            waiter = std::async(std::launch::async, [&] {
                entered.set_value();
                OperationLease::Guard second(lease);
                acquired.store(bool(second));
                second.complete();
                return bool(second);
            });
            entered.get_future().wait();
            auto probe = std::async(std::launch::async, [&] {
                OperationLease::Guard busy(lease, std::try_to_lock);
                require(!busy && busy.failure() == OperationLease::Failure::busy, "lifecycle preflight did not refuse contention");
                const auto start = std::chrono::steady_clock::now();
                OperationLease::Guard timeout(lease, 20ms);
                require(!timeout && timeout.failure() == OperationLease::Failure::timeout, "bounded contention did not time out");
                require(std::chrono::steady_clock::now() - start >= 20ms, "wait bound was not honored");
            });
            probe.get();
            require(!acquired.load(), "scratch was concurrently acquired");
            first.complete();
            require(!acquired.load(), "completion unlocked before final scratch reads");
        }
        require(waiter.get() && acquired.load(), "bounded owner did not hand off to waiter");
        {
            OperationLease::Guard successor(lease, std::try_to_lock);
            require(bool(successor), "a timeout poisoned someone else's healthy lease");
            successor.complete();
        }

        // Failure unlocks so contenders can observe the poison, not wait forever.
        OperationLease poisoned;
        std::future<OperationLease::Failure> after_failure;
        std::promise<void> waiting;
        {
            OperationLease::Guard failed(poisoned);
            require(bool(failed), "failed operation did not acquire");
            after_failure = std::async(std::launch::async, [&] {
                waiting.set_value();
                OperationLease::Guard next(poisoned);
                return next.failure();
            });
            waiting.get_future().wait();
        }
        require(after_failure.get() == OperationLease::Failure::poisoned, "waiting owner lost failure poison");
        for (int i = 0; i < 2; ++i) {
            OperationLease::Guard refused(poisoned, std::try_to_lock);
            require(!refused && refused.failure() == OperationLease::Failure::poisoned, "poison disappeared after refusal/drain");
        }

        OperationLease unwound;
        try {
            OperationLease::Guard operation(unwound);
            throw std::runtime_error("injected consumer failure");
        } catch (const std::runtime_error&) {}
        OperationLease::Guard exception_poison(unwound);
        require(!exception_poison && exception_poison.failure() == OperationLease::Failure::poisoned,
                "exception released scratch without preserving poison");
        std::puts("operation_lease_test: contention, timeout, final reads, poison and unwind passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "operation_lease_test: %s\n", error.what());
        return 1;
    }
}
