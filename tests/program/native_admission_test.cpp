// Exercise the production native-control coalescer, not copied generate.cpp statements.
#include "strata/program/serve_controls.hpp"
#include <algorithm>
#include <cstdio>
#include <deque>
#include <string>

namespace {
int failures = 0;
void check(bool okay, const char* message) {
    if (!okay) { std::fprintf(stderr, "FAIL: %s\n", message); ++failures; }
}
void control_case() {
    std::deque<std::string> pending;
    const auto enqueue = [&](const std::string& incoming) {
        strata::program::coalesce_native_control(pending, incoming);
        pending.push_back(incoming);
    };
    const std::string cancel = "BQCANCEL victim 0 17 3";
    const std::string priority = "BQPRIORITY victim 0 17 3 foreground";
    const std::string resume = "BQRESUME victim 0 17 3 32 1 1 0 128 32 0";
    for (const auto& valid : {cancel, priority, resume}) {
        pending = {valid, "BQREGISTER peer foreground eligible 100000 4"};
        auto stale = valid;
        stale.replace(stale.find(" 17 3"), 5, " 17 2");
        enqueue(stale);
        check(std::find(pending.begin(), pending.end(), valid) != pending.end(),
              "a late incarnation cannot erase an unconsumed valid control");
    }
    pending = {cancel};
    enqueue("BQCANCEL victim 1 17 3");
    check(std::find(pending.begin(), pending.end(), cancel) != pending.end(),
          "a stale physical hint cannot overwrite valid cancellation");
    pending = {resume};
    enqueue("BQRESUME victim 0 17 3 64 1 1 0 128 32 0");
    check(std::find(pending.begin(), pending.end(), resume) != pending.end(),
          "a stale retained cursor cannot overwrite a valid resume");
    pending = {priority};
    enqueue("BQPRIORITY victim 0 17 3 background");
    check(pending.size() == 1 && pending.front().ends_with("background"),
          "same-fence priorities retain only the latest intention");
    pending = {cancel};
    enqueue(cancel);
    check(pending.size() == 1 && pending.front() == cancel,
          "identical cancellation is coalesced without dropping its intention");
}
}

int main() {
    control_case();
    if (!failures) std::puts("native_admission_test: production control fences passed");
    return failures ? 1 : 0;
}
