// #233 production SCORE framing, bounded ownership and FP32 readout: no model, no GPU.
#include "strata/program/score_protocol.hpp"
#include <algorithm>
#include <cstdio>

namespace score = strata::program::score;
static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)
int main() {
    score::Request request;
    std::string error;
    CHECK(score::parse("SCORE pass-233 v1 prompt=5,6 selected=3,1,0,3", 4096, 258, request, error));
    CHECK(request.id == "pass-233");
    CHECK(request.prompt == std::vector<int64_t>({5, 6}));
    CHECK(request.selected == std::vector<int64_t>({3, 1, 0, 3}));
    const float row[] = {.125f, -17.5f, 2.25f, 1.0f / 3.0f};
    std::vector<float> selected;
    CHECK(score::select_logits(row, 4, request.selected, selected, error));
    CHECK(selected == std::vector<float>({0.3333333432674408f, -17.5f, .125f, 0.3333333432674408f}));
    const std::string reply = score::terminal(request.id, "OK", &request, &selected);
    CHECK(reply.find("SDONE pass-233 OK ") == 0);
    CHECK(reply.find("\"request_id\":\"pass-233\"") != reply.npos);
    CHECK(reply.find("\"generated_tokens\":0") != reply.npos);
    CHECK(reply.find("\"logit_passes\":1,\"dtype\":\"float32\",\"scale\":1") != reply.npos);
    CHECK(reply.find("\"logits\":[0.333333343,-17.5,0.125,0.333333343]") != reply.npos);
    CHECK(std::count(reply.begin(), reply.end(), '\n') == 1);
    selected.pop_back();
    const std::string partial = score::terminal(request.id, "OK", &request, &selected);
    CHECK(partial.find("SDONE pass-233 ERROR ") == 0);
    CHECK(partial.find("\"logits\"") == partial.npos);
    const auto rejected = score::terminal(request.id, "REJECT", nullptr, nullptr, "bound\n\"ary");
    CHECK(rejected.find("SDONE pass-233 REJECT ") == 0);
    CHECK(std::count(rejected.begin(), rejected.end(), '\n') == 1);
    CHECK(rejected.find("bound\\u000a\\\"ary") != rejected.npos);

    for (const auto line : {"SCORE x v2 prompt=5 selected=0", "SCORE x v1 prompt= selected=0",
                           "SCORE x v1 prompt=5 selected=", "SCORE x v1 prompt=5, selected=0",
                           "SCORE x v1 prompt=-1 selected=0", "SCORE x v1 prompt=258 selected=0",
                           "SCORE x v1 prompt=5 selected=258", "SCORE x v1 prompt=5 selected=0 trailing",
                           "SCORE x v1 prompt=+5 selected=0", "SCORE bad/id v1 prompt=5 selected=0"})
        CHECK(!score::parse(line, 4096, 258, request, error));
    CHECK(!score::parse("SCORE x v1 prompt=5,6 selected=0", 1, 258, request, error));
    CHECK(!score::parse("SCORE x " + std::string(score::max_wire_bytes, 'x'), 4096, 258, request, error));
    std::string work = "SCORE x v1 prompt=5";
    for (size_t token = 1; token < score::max_prompt_tokens; ++token) work += ",5";
    CHECK(score::parse(work + " selected=0", 4096, 258, request, error));
    CHECK(!score::parse(work + ",5 selected=0", 5000, 258, request, error));
    work = "SCORE x v1 prompt=5 selected=0";
    for (size_t token = 1; token < score::max_selected_tokens; ++token) work += ",0";
    CHECK(score::parse(work, 4096, 258, request, error));
    CHECK(!score::parse(work + ",0", 4096, 258, request, error));
    CHECK(!score::select_logits(row, 4, {0, 4}, selected, error));
    CHECK(selected.empty());
    const float poisoned[] = {1, std::numeric_limits<float>::quiet_NaN()};
    CHECK(!score::select_logits(poisoned, 2, {0}, selected, error));
    CHECK(selected.empty());                  // even an unselected poison is an error, never a partial result

    score::Pending pending;
    using Admission = score::Pending::Admission;
    for (size_t i = 0; i < score::max_pending; ++i)
        CHECK(pending.admit(std::to_string(i)) == Admission::accepted);
    CHECK(pending.admit("overflow") == Admission::full);
    CHECK(pending.admit("0") == Admission::duplicate);
    pending.cancel("0");
    CHECK(pending.cancelled("0"));
    CHECK(!pending.cancelled("1"));
    CHECK(pending.admit("overflow") == Admission::full); // cancellation alone cannot release ownership
    pending.complete("0");
    CHECK(pending.admit("successor") == Admission::accepted);
    CHECK(!pending.cancelled("successor"));
    pending.cancel("missing");               // late SCANCEL has no effect on a successor
    CHECK(!pending.cancelled("successor"));
    if (!fails) std::puts("score_protocol_test: ok (identity, ordered FP32, all-or-error, bounds, cancellation ownership)");
    return fails ? 1 : 0;
}
