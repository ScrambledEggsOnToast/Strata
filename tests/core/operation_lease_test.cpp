#include "strata/core/operation_lease.hpp"
#include <algorithm>

#include <atomic>
#include <cstdio>
#include <future>
#include <optional>
#include <string>
#include <stdexcept>
#include <thread>
#include <vector>

using strata::core::OperationLease;
using namespace std::chrono_literals;

static void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

namespace strata::core {
struct OnDevice { explicit OnDevice(int) {} }; // only device selection is outside the CPU ownership seam
}
namespace core = strata::core;

static void test_owned_boundaries() {
    struct E9Drafter {
        bool owned = true;
        int calls = 0;
        bool idle(std::string&) { return false; } // public lifecycle preflight must not be used by this owner
        bool idle_owned(std::string&) { ++calls; return owned; }
    } mtp;
    std::string err;
    using Clock = std::chrono::steady_clock;
    const auto ms_since = [](Clock::time_point from) {
        return std::chrono::duration<double, std::milli>(Clock::now() - from).count();
    };
    const auto e9 = [&] {
#include "draft_owned_drain.inc"
        (void)ms_idle;
        return true;
    };
    require(e9() && mtp.calls == 1, "E9's already-owned stream boundary reacquired its lease");

    struct Target {
        int64_t cells = 0, completed_cells = 0, begin = 0, prompt_len = 0, proposal_cell = 0;
        int32_t tail_token = -1, proposal = -1, proposal_token = -1;
        bool certified = false;
    };
    struct Drafter {
        OperationLease scratch;
        Target slots_[1];
        int calls = 0;
        int device() const { return 0; }
        bool catchup_slot(int slot, int, int32_t token, int64_t cell, std::string&) {
            OperationLease::Guard catchup(scratch, 20ms);
            if (!catchup) return false;
            ++calls;
            slots_[slot].cells = cell + 1;
            slots_[slot].tail_token = token;
            catchup.complete();
            return true;
        }
    } draft_;
    struct RestoreOperation {
        OperationLease::Guard owner;
        explicit RestoreOperation(Drafter& draft) : owner(draft.scratch) {}
        bool finish(std::string&) { owner.complete(); return bool(owner); }
    };
    struct Image { int64_t draft_cells = 2, draft_begin = 0; int draft_boundary_residual = 123; } image;
    struct Source { Image* image; } source{&image};
    struct Lease { int slot = 0; } lease;
    struct State { std::vector<int32_t> prompt{1, 2, 3, 4}; } state;
    const std::vector<int> stages_{0};
    const auto preparation_stage = [](const Image& value, size_t) -> const Image& { return value; };
    const auto fail = [](int) { return false; };
    std::string error;
    const int64_t position = 3;
    const auto restore = [&] {
        const bool mtp = true;
        std::optional<RestoreOperation> operation;
        operation.emplace(draft_);
#include "draft_restore_handoff.inc"
        return true;
    };
    require(restore() && draft_.calls == 1 && draft_.slots_[0].cells == 3 && draft_.slots_[0].tail_token == 4,
            "exact-cut restore held its finished operation across boundary catchup");
}

int main() {
    try {
        test_owned_boundaries();
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
