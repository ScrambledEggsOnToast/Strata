#include "strata/core/dflash2_bridge.hpp"
#include "strata/kernels/bf16_bits.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cfenv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <iostream>
#include <latch>
#include <limits>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace d = strata::core::dflash2;
using strata::kernels::bf16_from_f32;
void check(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
template<class F> void refuses(F&& action) {
    bool refused = false;
    try { action(); } catch (const d::Refusal&) { refused = true; }
    check(refused, "expected a DFlash2 refusal");
}
d::TargetIdentity identity() {
    return {std::string(64, '1'), std::string(64, '2'), std::string(64, '3'), std::string(64, '4')};
}
d::Operation operation(int64_t generation = 1, int frontier = 37) { return {"request-A", 7, generation, frontier}; }
struct Pair {
    int engine = -1, reference = -1;
    explicit Pair(int type = SOCK_STREAM) {
        int fds[2];
        check(socketpair(AF_UNIX, type, 0, fds) == 0, "socketpair failed");
        engine = fds[0]; reference = fds[1];
    }
    int take() { const int fd = engine; engine = -1; return fd; }
    ~Pair() { if (engine >= 0) close(engine); if (reference >= 0) close(reference); }
};

struct HeldScratch {
    std::latch acquired{1}, release{1};
    std::jthread owner;
    explicit HeldScratch(strata::core::OperationLease& lease) : owner([this, &lease] {
        strata::core::OperationLease::Guard guard(lease);
        check(bool(guard), "fixture could not own shared scratch");
        acquired.count_down();
        release.wait();
        guard.complete();
    }) { acquired.wait(); }
    ~HeldScratch() { release.count_down(); }
};

// Synthetic weights ONLY. This exercises the borrowed interface; it does not
// claim to borrow the loaded daily IQ3_S target or qualify any native GPU head.
struct FixtureTarget : d::BorrowedTarget {
    d::BorrowedBinding info{identity(), d::hidden_size, d::vocabulary_size, 1};
    strata::core::OperationLease lease;
    std::string fault;
    int head_calls = 0;
    std::jthread lease_probe;
    const d::BorrowedBinding& binding() const override { return info; }
    strata::core::OperationLease& scratch_lease() override { return lease; }
    void raw_embeddings(const d::Operation&, std::span<const int32_t> ids, std::span<float> out) override {
        check(ids.size() == 4 && ids[1] == d::mask_token && ids[2] == d::mask_token && ids[3] == d::mask_token,
              "borrowed block must be anchor/mask/mask/mask");
        check(out.size() == 4 * d::hidden_size, "raw embedding shape");
        for (int row = 0; row < 4; ++row) for (int column = 0; column < d::hidden_size; ++column)
            out[size_t(row) * d::hidden_size + column] = float(ids[size_t(row)] % 128) * 0.125f +
                                                         float(column % 8) * 0.00390625f;
        if (fault == "embedding") out.back() = std::numeric_limits<float>::quiet_NaN();
        if (fault == "reload") ++info.load_generation;
    }
    void native_head(const d::Operation&, std::span<const float> hidden, std::span<float> logits) override {
        check(hidden.size() == d::hidden_size && logits.size() == d::vocabulary_size, "native head shape");
        ++head_calls;
        for (int token = 0; token < d::vocabulary_size; ++token)
            logits[size_t(token)] = hidden[0] + hidden[1] * (float(token % 17) * 0.25f);
        if (fault == "head") logits.back() = std::numeric_limits<float>::infinity();
        if (fault == "head-throw") throw std::runtime_error("fixture native-head failure");
        if (fault == "head-reload") ++info.load_generation;
        if (fault == "lease-probe" && head_calls == 3) {
            lease_probe = std::jthread([this] {
                strata::core::OperationLease::Guard guard(lease, std::chrono::milliseconds(250));
                const bool available = bool(guard);
                if (guard) guard.complete();
                std::cout << "{\"scratch_available\":" << (available ? "true" : "false") << "}\n" << std::flush;
            });
        }
    }
};

std::vector<float> residual_rows(int start, int rows, int tap) {
    std::vector<float> values(size_t(rows) * 4 * d::hidden_size);
    for (int row = 0; row < rows; ++row)
        std::fill_n(values.data() + size_t(row) * 4 * d::hidden_size, 4 * d::hidden_size,
                    float(tap * 64 + start + row));
    return values;
}
void capture_all(d::CaptureStore& store, const d::Operation& op, int start, int rows, d::CapturePhase phase) {
    std::vector<int> positions(size_t(rows), 0);
    std::iota(positions.begin(), positions.end(), start);
    for (int tap : {4, 0, 3, 1, 2}) { // deliberately not stage/layer order
        const auto values = residual_rows(start, rows, tap);
        const int layer = d::tap_layers[size_t(tap)];
        store.capture(op, layer, layer + (phase == d::CapturePhase::verify ? 1 : 0), positions,
                      phase == d::CapturePhase::verify ? d::CapturePoint::next_layer_first_gr_read_group :
                                                        d::CapturePoint::completed_second_residual_write, values);
    }
}
void prefill(d::CaptureStore& store, int start, int rows, int64_t generation) {
    const auto op = operation(generation, start + rows);
    store.begin(op, start, rows, d::CapturePhase::prefill);
    capture_all(store, op, start, rows, d::CapturePhase::prefill);
    store.commit(op, rows); // explicit successful-target-commit fixture
}
void check_features(const d::FeatureView& features) {
    int row = 0;
    for (const auto segment : features.segments) {
        check(segment.size() % d::feature_width == 0, "ring split cut a feature row");
        for (size_t offset = 0; offset < segment.size(); offset += d::feature_width, ++row)
            for (int tap = 0; tap < 5; ++tap) for (int column = 0; column < d::hidden_size; ++column)
                check(segment[offset + size_t(tap) * d::hidden_size + column] ==
                      bf16_from_f32(float(tap * 64 + features.start + row)), "wrong tap/absolute row ordering");
    }
    check(row == features.count, "feature coverage mismatch");
}

void reduction_and_capture_tests() {
    for (int keep = 1; keep <= 4; ++keep) {
        d::CaptureStore store("request-A", 7, 37, d::block_size);
        const auto op = operation();
        store.begin(op, 37, 4, d::CapturePhase::verify);
        capture_all(store, op, 37, 4, d::CapturePhase::verify);
        refuses([&] { store.committed_since(37); }); // tentative is not public
        store.commit(op, keep);
        check(store.frontier() == 37 + keep, "rejected suffix published");
        check_features(store.committed_since(37));
        check(store.storage_bytes() == d::CaptureStore::planned_bytes(4),
              "unexpected observer allocation bound");
    }
    d::CaptureStore store("request-A", 7, 37, 1);
    auto op = operation(1, 38);
    store.begin(op, 37, 1, d::CapturePhase::prefill);
    std::vector<float> values(4 * d::hidden_size, 1.0f);
    values[0] = 16777216.0f; values[d::hidden_size] = 1.0f;
    values[2 * d::hidden_size] = -16777216.0f; values[3 * d::hidden_size] = 4.0f;
    for (int stream = 0; stream < 4; ++stream) {
        values[size_t(stream) * d::hidden_size + 1] = 1.00390625f; // retained mantissa even
        values[size_t(stream) * d::hidden_size + 2] = 1.01171875f; // retained mantissa odd
    }
    const auto before = values;
    const std::array<int, 1> positions{37};
    for (int layer : d::tap_layers)
        store.capture(op, layer, layer, positions, d::CapturePoint::completed_second_residual_write, values);
    store.commit(op, 1);
    const auto first = store.committed_since(37).segments[0];
    for (int tap = 0; tap < 5; ++tap) {
        check(first[size_t(tap) * d::hidden_size] == 0x3f80, "FP32 stream addition order changed");
        check(first[size_t(tap) * d::hidden_size + 1] == 0x3f80, "BF16 even tie changed");
        check(first[size_t(tap) * d::hidden_size + 2] == 0x3f82, "BF16 odd tie changed");
    }
    check(values == before, "observer mutated target residuals");
}
void ring_and_refusal_tests() {
    d::CaptureStore ring("request-A", 7, 37);
    check(ring.storage_bytes() == d::CaptureStore::planned_bytes(),
          "default capacity must support a full prefill chunk");
    prefill(ring, 37, 8, 1); // prefill chunks are not limited to the four-row draft block
    prefill(ring, 45, d::context_rows - 9, 2);
    const int start = ring.frontier();
    const auto wrapped = operation(3, start);
    ring.begin(wrapped, start, 4, d::CapturePhase::verify);
    capture_all(ring, wrapped, start, 4, d::CapturePhase::verify);
    ring.commit(wrapped, 3); // partial verification crosses the 2047-row ring boundary
    check(ring.frontier() == start + 3 && ring.retained_start() == 39,
          "partial wrapped commit retained a rejected suffix or obsolete prefix");
    const auto view = ring.committed_since(39);
    check(!view.segments[1].empty() && view.count == d::context_rows, "ring wrap not exposed as ordered segments");
    check_features(view);
    check_features(ring.committed_since(start));
    refuses([&] { ring.committed_since(38); });
    refuses([] { d::CaptureStore invalid("request-A", 7, 37, 2048); });
    for (int scenario = 0; scenario < 12; ++scenario) {
        d::CaptureStore store("request-A", 7, 37);
        const auto op = operation();
        store.begin(op, 37, 4, d::CapturePhase::verify);
        std::array<int, 4> positions{37, 38, 39, 40};
        auto values = residual_rows(37, 4, 0);
        refuses([&] {
            auto completion = op;
            int layer = 5, materialized = 6;
            auto point = d::CapturePoint::next_layer_first_gr_read_group;
            if (scenario == 0) completion.incarnation++;
            if (scenario == 1) completion.request_id = "request-B";
            if (scenario == 2) completion.generation++;
            if (scenario == 3) positions[2] = 38;
            if (scenario == 4) std::swap(positions[0], positions[1]);
            if (scenario == 5) layer = 6;
            if (scenario == 6) materialized = 5; // incomplete deferred write
            if (scenario == 7) point = d::CapturePoint::completed_second_residual_write;
            if (scenario == 8) values.back() = std::numeric_limits<float>::quiet_NaN();
            if (scenario == 9) {
                values[0] = values[d::hidden_size] = std::numeric_limits<float>::max();
            }
            if (scenario == 10) values.pop_back();
            store.capture(completion, layer, materialized, positions, point, values);
            if (scenario == 11) store.capture(completion, layer, materialized, positions, point, values);
        });
        check(store.poisoned() && store.frontier() == 37, "capture refusal published/repaired state");
    }
    for (int scenario = 0; scenario < 5; ++scenario) {
        d::CaptureStore store("request-A", 7, 37);
        const auto op = operation(1, scenario == 2 ? 40 : 37);
        refuses([&] {
            if (scenario == 0) store.begin(op, 38, 1, d::CapturePhase::verify);
            if (scenario == 1) store.begin(op, 37, 5, d::CapturePhase::verify);
            if (scenario == 2) {
                store.begin(op, 37, 3, d::CapturePhase::prefill);
                capture_all(store, op, 37, 3, d::CapturePhase::prefill);
                store.commit(op, 2);
            }
            if (scenario == 3) { store.begin(op, 37, 4, d::CapturePhase::verify); store.commit(op, 1); }
            if (scenario == 4) {
                store.begin(op, 37, 4, d::CapturePhase::verify);
                store.discard(op);
                store.begin(op, 37, 4, d::CapturePhase::verify);
            }
        });
        check(store.poisoned(), "invalid extent/publication/generation did not poison");
    }
    {
        d::CaptureStore small("request-A", 7, 37, 4);
        refuses([&] { small.begin(operation(1, 42), 37, 5, d::CapturePhase::prefill); });
        check(small.poisoned() && small.frontier() == 37, "oversized prefill was silently truncated");
    }
    {
        d::CaptureStore discarded("request-A", 7, 37, 4);
        const auto first = operation(), next = operation(2);
        discarded.begin(first, 37, 4, d::CapturePhase::verify);
        capture_all(discarded, first, 37, 4, d::CapturePhase::verify);
        discarded.discard(first);
        check(!discarded.poisoned() && discarded.frontier() == 37, "discard published or poisoned tentative rows");
        discarded.begin(next, 37, 4, d::CapturePhase::verify);
        capture_all(discarded, next, 37, 4, d::CapturePhase::verify);
        discarded.commit(next, 2);
        check_features(discarded.committed_since(37));
        refuses([&] { discarded.commit(next, 2); });
        check(discarded.poisoned() && discarded.frontier() == 39, "duplicate commit advanced the frontier");
    }
    {
        d::CaptureStore aborted("request-A", 7, 37, 4);
        aborted.begin(operation(), 37, 4, d::CapturePhase::verify);
        aborted.abort();
        aborted.abort();
        check(aborted.poisoned() && !aborted.capturing() && aborted.frontier() == 37, "abort published tentative rows");
        refuses([&] { aborted.begin(operation(2), 37, 4, d::CapturePhase::verify); });
    }
    d::CaptureStore rounding("request-A", 7, 37);
    const auto op = operation();
    rounding.begin(op, 37, 4, d::CapturePhase::verify);
    const int saved = std::fegetround();
    check(std::fesetround(FE_DOWNWARD) == 0, "cannot set rounding fixture");
    try { refuses([&] { capture_all(rounding, op, 37, 4, d::CapturePhase::verify); }); }
    catch (...) { std::fesetround(saved); throw; }
    std::fesetround(saved);
}
void ownership_and_borrowing_tests() {
    for (int scenario = 0; scenario < 5; ++scenario) {
        Pair pair;
        const int owned_fd = pair.engine;
        d::CaptureStore store("request-A", 7, 37);
        prefill(store, 37, 3, 1);
        FixtureTarget target;
        if (scenario == 0) target.info.vocabulary--;
        if (scenario == 1) target.info.hidden--;
        if (scenario == 2) target.info.identity.head_sha256 = std::string(64, '5');
        if (scenario == 3) target.info.load_generation = 0;
        refuses([&] { d::Bridge bridge(store, target, identity(), pair.take(),
                                     std::chrono::milliseconds(scenario == 4 ? 0 : 1000)); });
        check(fcntl(owned_fd, F_GETFD) == -1 && !store.poisoned() && store.frontier() == 40,
              "constructor leaked fd or poisoned previously committed rows");
        check_features(store.committed_since(37));
    }
    {
        Pair pair(SOCK_SEQPACKET);
        d::CaptureStore store("request-A", 7, 37);
        FixtureTarget target;
        refuses([&] { d::Bridge bridge(store, target, identity(), pair.take()); });
    }
    for (bool committed : {false, true}) {
        d::CaptureStore store("request-A", 7, 37, 4);
        FixtureTarget target;
        if (committed) prefill(store, 37, 3, 1);
        {
            Pair first, second;
            d::Bridge owner(store, target, identity(), first.take());
            const int rejected_fd = second.engine;
            refuses([&] { d::Bridge duplicate(store, target, identity(), second.take()); });
            check(!owner.poisoned() && !store.poisoned() && fcntl(rejected_fd, F_GETFD) == -1,
                  "second bridge stole/poisoned the store or leaked its fd");
        }
        check(!store.poisoned(), "idle bridge destruction poisoned its store");
        Pair replacement;
        d::Bridge owner(store, target, identity(), replacement.take());
    }
    {
        Pair pair;
        d::CaptureStore store("request-A", 7, 37, 4);
        FixtureTarget target;
        d::Bridge bridge(store, target, identity(), pair.take(), std::chrono::milliseconds(40));
        {
            HeldScratch owner(target.lease);
            const auto started = std::chrono::steady_clock::now();
            refuses([&] { bridge.propose(operation(), 17); });
            const auto elapsed = std::chrono::steady_clock::now() - started;
            check(elapsed >= std::chrono::milliseconds(20) && elapsed < std::chrono::seconds(1),
                  "shared scratch wait was not bounded");
            check(!bridge.poisoned() && !store.poisoned() && target.head_calls == 0,
                  "scratch contention poisoned or started an operation");
            char byte;
            check(recv(pair.reference, &byte, 1, MSG_DONTWAIT) < 0 && (errno == EAGAIN || errno == EWOULDBLOCK),
                  "scratch contention wrote exchange bytes");
        }
        strata::core::OperationLease::Guard lease(target.lease, std::try_to_lock);
        check(bool(lease), "contention poisoned healthy scratch");
        lease.complete();
        bridge.abort();
        bridge.abort();
        check(bridge.poisoned() && store.poisoned(), "explicit abort did not fence its owner");
    }
    for (const auto& fault : {"embedding", "reload", "poisoned"}) {
        Pair pair;
        d::CaptureStore store("request-A", 7, 37);
        FixtureTarget target;
        target.fault = fault;
        d::Bridge bridge(store, target, identity(), pair.take(), std::chrono::seconds(1));
        if (target.fault == "poisoned") { strata::core::OperationLease::Guard failed(target.lease); }
        refuses([&] { bridge.propose(operation(), 17); });
        strata::core::OperationLease::Guard lease(target.lease, std::try_to_lock);
        check(!lease && lease.failure() == strata::core::OperationLease::Failure::poisoned,
              "failed borrowed target operation repaired shared scratch");
        check(bridge.poisoned() && store.poisoned() && store.frontier() == 37,
              "bridge failure did not fence its owner");
        refuses([&] { bridge.propose(operation(2), 17); });
    }
}

// Invoked by the Python exchange tests over pass_fds. No listener, model launch
// or reference replacement: Python drives the real version-1 Channel, and may
// also use the explicitly admitted pinned CPU reference for both transactions.
int exchange_fixture(int fd, int keep, const std::string& fault) {
    d::CaptureStore store("request-A", 7, 37);
    prefill(store, 37, 3, 1);
    FixtureTarget target;
    target.fault = fault;
    std::optional<d::Bridge> bridge(std::in_place, store, target, identity(), fd,
                                  std::chrono::milliseconds(fault == "timeout" ? 250 : 5000));
    try {
        if (fault == "retry") {
            HeldScratch owner(target.lease);
            refuses([&] { bridge->propose(operation(2, store.frontier()), 17); });
            check(!bridge->poisoned() && !store.poisoned(), "contended operation could not be retried");
        }
        std::array<d::Proposals, 2> results;
        for (int step = 0; step < 2; ++step) {
            const auto op = operation(2 + step, store.frontier());
            results[size_t(step)] = bridge->propose(op, step ? 19 : 17);
            check(store.frontier() == op.frontier, "proposals published tentative target features");
            if (fault == "abort") {
                bridge->abort();
                bridge->abort();
                bridge->commit_verified(op, keep);
            }
            if (fault == "destroy") {
                bridge.reset();
                throw std::runtime_error("fixture destroyed an outstanding bridge");
            }
            store.begin(op, op.frontier, 4, d::CapturePhase::verify);
            capture_all(store, op, op.frontier, 4, d::CapturePhase::verify);
            // This CPU fixture models a successful real target commit, not an
            // acceptance computation or evidence of engine verifier integration.
            bridge->commit_verified(op, step ? 1 : keep);
            if (fault == "duplicate-commit") bridge->commit_verified(op, keep);
            check_features(store.committed_since(store.retained_start()));
        }
        check(target.head_calls == 6 && !bridge->poisoned(), "head did not run three serial rows per step");
        std::cout << "{\"tokens\":[";
        for (int step = 0; step < 2; ++step) {
            if (step) std::cout << ',';
            const auto& tokens = results[size_t(step)].tokens;
            std::cout << '[' << tokens[0] << ',' << tokens[1] << ',' << tokens[2] << ']';
        }
        std::cout << "],\"frontier\":" << store.frontier() << ",\"head_calls\":" << target.head_calls << "}\n";
        return 0;
    } catch (const std::exception& error) {
        check((!bridge || bridge->poisoned()) && store.poisoned(), "exchange failure did not poison its request");
        strata::core::OperationLease::Guard lease(target.lease, std::try_to_lock);
        const bool scratch_poisoned = lease.failure() == strata::core::OperationLease::Failure::poisoned;
        if (lease) lease.complete();
        std::cout << "{\"poisoned\":true,\"scratch_poisoned\":" << (scratch_poisoned ? "true" : "false")
                  << ",\"frontier\":" << store.frontier() << "}\n";
        std::cerr << "dflash2_bridge_test: " << error.what() << '\n';
        return 2;
    }
}
int main(int argc, char** argv) {
    try {
        if (argc >= 5 && std::string(argv[1]) == "--exchange-fd")
            return exchange_fixture(std::stoi(argv[2]), std::stoi(argv[4]), argc > 5 ? argv[5] : "");
        reduction_and_capture_tests();
        ring_and_refusal_tests();
        ownership_and_borrowing_tests();
        std::puts("dflash2_bridge_test: reduction, taps, ring, publication, borrowing and ownership passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "dflash2_bridge_test: %s\n", error.what());
        return 1;
    }
}
