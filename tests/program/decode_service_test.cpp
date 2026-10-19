// tests/program/decode_service_test.cpp - weighted decode service (header-only, CPU).
//
//   1. one foreground and one background request with capacity 1 serve 3:1 in a fixed cycle;
//   2. a background request's wait is bounded by the cycle length, and capacity covering every request serves all;
//   3. the scheduler names logical requests, not seats: removal/addition rebuilds the cycle without serving a ghost;
//   4. a priority change takes effect on the next window and preserves the phase where possible;
//   5. a request is returned at most once per window and zero capacity selects nothing.
#include "strata/program/decode_service.hpp"

#include <cstdio>
#include <vector>

namespace ds = strata::program;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-78s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
std::vector<ds::DecodeParticipant> participants(std::initializer_list<std::pair<uint64_t, bool>> in) {
    std::vector<ds::DecodeParticipant> out;
    for (const auto& [request, foreground] : in) out.push_back({request, foreground});
    return out;
}
}  // namespace

int main() {
    std::printf("decode_service_test\n");
    {
        ds::DecodeService service(1);
        const auto ready = participants({{1, true}, {2, false}});
        std::vector<uint64_t> served;
        for (int window = 0; window < 8; ++window) {
            const auto chosen = service.select(ready);
            check(chosen.size() == 1, "capacity one serves exactly one request per window");
            served.push_back(chosen.front());
        }
        check(served[0] == 1 && served[1] == 1 && served[2] == 1 && served[3] == 2 &&
              served[4] == 1 && served[5] == 1 && served[6] == 1 && served[7] == 2,
              "foreground and background alternate 3:1 over a fixed cycle");
        check(service.cycle_length() == 4 && service.background_age_bound() == 4,
              "the cycle length is the weight sum and bounds a background wait");
    }
    {
        ds::DecodeService service(2);
        const auto ready = participants({{1, true}, {2, false}});
        for (int window = 0; window < 4; ++window) {
            const auto chosen = service.select(ready);
            check(chosen.size() == 2 && chosen[0] == 1 && chosen[1] == 2,
                  "capacity covering every request serves both every window");
        }
    }
    {
        ds::DecodeService service(1);
        const auto both = participants({{1, true}, {2, false}});
        service.select(both);
        const auto only_background = participants({{2, false}});
        for (int window = 0; window < 3; ++window) {
            const auto chosen = service.select(only_background);
            check(chosen.size() == 1 && chosen.front() == 2,
                  "a removed foreground request leaves its seat out of the cycle");
        }
        const auto joined = participants({{2, false}, {3, true}});
        bool saw_three = false;
        for (int window = 0; window < 8; ++window)
            for (const uint64_t request : service.select(joined)) saw_three |= request == 3;
        check(saw_three, "a newly admitted request enters the service cycle");
    }
    {
        ds::DecodeService service(1);
        auto ready = participants({{1, true}, {2, false}});
        std::vector<uint64_t> before;
        for (int window = 0; window < 4; ++window) before.push_back(service.select(ready).front());
        ready[1].foreground = true;   // the background request is promoted
        int served_two = 0;
        for (int window = 0; window < 6; ++window)
            for (const uint64_t request : service.select(ready)) served_two += request == 2 ? 1 : 0;
        check(served_two >= 3, "a promoted request receives the foreground share on the next windows");
    }
    {
        ds::DecodeService service(4);
        const auto ready = participants({{1, true}, {2, true}, {3, false}, {4, false}});
        for (int window = 0; window < 3; ++window) {
            const auto chosen = service.select(ready);
            check(chosen.size() == 4, "capacity four serves four distinct requests");
            bool distinct = true;
            for (size_t i = 0; i < chosen.size(); ++i)
                for (size_t j = i + 1; j < chosen.size(); ++j) distinct &= chosen[i] != chosen[j];
            check(distinct, "a request is selected at most once per window");
        }
    }
    {
        const auto one_each = participants({{1, true}, {2, false}});
        check(ds::decode_window_capacity(one_each, 8) == 1,
              "one interactive request and one background request admit one row per window");
        const auto two_interactive = participants({{1, true}, {2, true}, {3, false}});
        check(ds::decode_window_capacity(two_interactive, 8) == 2,
              "background work cannot enlarge a two-interactive window");
        const auto only_background = participants({{2, false}, {3, false}});
        check(ds::decode_window_capacity(only_background, 8) == 2,
              "with no interactive work, ready backgrounds share the window normally");
        check(ds::decode_window_capacity({}, 8) == 0, "an empty ready set admits no rows");
        check(ds::decode_window_capacity(participants({{1, true}}), 1) == 1, "the verify row cap is respected");
    }
    {
        ds::DecodeService service(0);
        check(service.select(participants({{1, true}})).empty(), "zero capacity selects nothing");
    }
    {
        ds::DecodeService service(2);
        check(service.select(participants({{0, true}})).empty(), "an unbound request id is not a participant");
    }
    std::printf("%s\n", g_fail == 0 ? "decode_service_test: all passed" : "decode_service_test: FAILED");
    return g_fail == 0 ? 0 : 1;
}
