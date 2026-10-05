#include "strata/core/verify_ownership.hpp"

#include <cstdio>
#include <cstdlib>

using namespace strata::core;
#define REQUIRE(x) do { if (!(x)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); std::exit(1); } } while (0)

int main() {
    VerifyScratchOwnership owner;
    REQUIRE(!owner.commit());
    REQUIRE(!owner.finish());
    REQUIRE(owner.begin());
    REQUIRE(!owner.begin()); // overlapping submission cannot touch staging
    REQUIRE(!owner.commit());
    REQUIRE(owner.abandon_unlaunched()); // failed capture/setup remains retryable
    REQUIRE(owner.begin());
    REQUIRE(owner.ready());
    REQUIRE(!owner.begin()); // completed but uncommitted recurrence inputs still owned
    REQUIRE(!owner.abandon_unlaunched());
    REQUIRE(owner.commit());
    REQUIRE(!owner.commit());
    REQUIRE(!owner.begin()); // pending async commit owns mapped commit/staging
    REQUIRE(owner.finish());
    REQUIRE(owner.begin());
    REQUIRE(owner.finish_batch()); // pipeline release only after window AND commit
    REQUIRE(owner.begin());
    owner.poison(); // watchdog cannot be undone by late success or setup unwind
    REQUIRE(!owner.ready());
    REQUIRE(!owner.finish_batch());
    REQUIRE(!owner.abandon_unlaunched());
    REQUIRE(!owner.begin());
    REQUIRE(owner.phase() == VerifyScratchOwnership::Phase::poisoned);
    VerifyScratchOwnership committing;
    REQUIRE(committing.begin() && committing.ready() && committing.commit());
    committing.poison();
    REQUIRE(!committing.finish());
    REQUIRE(!committing.begin());

    VerifyCanaryInput expected{1001, 7, 1, 42, 19, 1};
    VerifyCanaryOutput observed{};
    observed.input = expected;
    observed.owner_id = expected.request_id;
    observed.graph_slot = expected.slot;
    observed.owner_steering = observed.steering = expected.steering;
    observed.token = expected.token;
    observed.position = expected.position;
    observed.phase = 1;
    REQUIRE(verify_canary_matches(observed, expected, 1));
    for (int which = 0; which < 14; ++which) {
        auto wrong = observed;
        switch (which) {
        case 0: ++wrong.input.request_id; break;
        case 1: ++wrong.input.epoch; break;
        case 2: ++wrong.input.slot; break;
        case 3: ++wrong.input.token; break;
        case 4: ++wrong.input.position; break;
        case 5: wrong.input.steering = 0; break;
        case 6: ++wrong.owner_id; break;
        case 7: ++wrong.graph_slot; break;
        case 8: wrong.owner_steering = 0; break;
        case 9: ++wrong.token; break;
        case 10: ++wrong.position; break;
        case 11: wrong.steering = 0; break;
        case 12: wrong.errors = 16; break;
        case 13: wrong.phase = 2; break;
        }
        REQUIRE(!verify_canary_matches(wrong, expected, 1));
    }
    REQUIRE(!verify_canary_matches(observed, expected, 2));
    ++expected.epoch; // identical request/token/position, stale previous output
    REQUIRE(!verify_canary_matches(observed, expected, 1));
    static_assert(kVerifyCanaryDeviceBytes == 720 && kVerifyCanaryHostBytes == 1552);
    static_assert(kVerifyBatchGraphKeys == 32);
    std::puts("verify ownership CPU seams: passed");
}
