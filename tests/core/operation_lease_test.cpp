#include "strata/core/operation_lease.hpp"

#include <atomic>
#include <cstdio>
#include <future>
#include <algorithm>
#include <string>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <thread>

using strata::core::OperationLease;
using namespace std::chrono_literals;

static void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}


namespace strata::core {
struct OnDevice { explicit OnDevice(int) {} }; // CPU seam: device selection is the only omitted adapter.
}
using strata::core::OnDevice;

static void test_owned_boundaries() {
    struct E9Drafter {
        int calls = 0;
        bool idle(std::string&) { return false; } // Reacquiring public lifecycle path is invalid here.
        bool idle_owned(std::string&) { ++calls; return true; }
    } mtp;
    std::string err;
    using Clock = std::chrono::steady_clock;
    const auto ms_since = [](Clock::time_point from) {
        return std::chrono::duration<double, std::milli>(Clock::now() - from).count();
    };
    const auto e9 = [&] {
#include "draft_owned_drain.inc"
        (void) ms_idle;
        return true;
    };
    require(e9() && mtp.calls == 1, "E9's already-owned drain used its public lease acquisition path");

    // The modern common restore keeps its Operation and performs MTP boundary
    // catchup through catchup_owned; it must not reacquire the same scratch lease.
    OperationLease scratch;
    OperationLease::Guard owner(scratch);
    require(bool(owner), "restore owner acquisition");
    struct Operation {
        OperationLease::Guard& owner;
        int64_t begin = -1, end = -1;
        int32_t token = -1;
        void writes(int64_t b, int64_t e, int32_t t) { begin = b; end = e; token = t; }
    } operation{owner};
    struct Image {
        std::vector<int> kv{1};
        int64_t draft_cells = 2, draft_begin = 0;
        std::vector<float> draft_boundary_residual{7.f};
        struct { std::vector<int32_t> ids{1, 2, 3}; } live;
    } image;
    struct Target { int64_t prompt_len = 0, proposal_cell = 0; int proposal = 0, proposal_token = 0; };
    std::vector<Target> slots_(1);
    int state_value = 1, geometry_value = 1, device_ = 0;
    const auto* g_ = &geometry_value;
    int slot = 0, expectation = 0, restored_calls = 0, catchup_calls = 0;
    int64_t cells = 3, prompt_length = 4, prompt_len_ = 0;
    int32_t next_token = 4;
    std::string error;
    const auto legacy_state = [&](int selected) { return selected == 0 ? &state_value : nullptr; };
    const auto conversation_kv_restore = [&](int, int, int, int64_t restored, bool,
                                               int, std::string&, int64_t live) {
        ++restored_calls;
        require(owner.holds(scratch) && restored == 2 && live == 3, "restore lost owner or wrong cut");
        return true;
    };
    const auto restore_ring = [](int64_t, std::string&) { return true; };
    const auto set_boundary_capture = [](int) {};
    const auto catchup_owned = [&](Operation& bound, int selected, const std::vector<float>& residual,
                                   int32_t token, int64_t cell, std::string&) {
        ++catchup_calls;
        require(&bound == &operation && owner.holds(scratch) && selected == slot &&
                residual == std::vector<float>{7.f} && token == next_token && cell == 2,
                "restore catchup did not use the already-owned operation");
        bound.writes(0, cell + 1, token);
        return true;
    };
    const auto restore = [&] {
#include "draft_restore_owned.inc"
    };
    require(restore() && restored_calls == 1 && catchup_calls == 1 && owner.holds(scratch),
            "restore did not retain one owner across boundary catchup");
    require(operation.end == 3 && operation.token == 4 && slots_[0].prompt_len == 4 &&
            slots_[0].proposal == -1 && slots_[0].proposal_token == -1 && slots_[0].proposal_cell == -1,
            "restore catchup did not invalidate proposal and advance the boundary");
    cells = 2;
    require(restore() && restored_calls == 2 && catchup_calls == 1 && operation.end == 2 &&
            operation.token == 3, "exact restored cut unexpectedly repeated catchup");
    owner.complete(); // The common caller, not the adapter, drains/publishes and releases.
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
