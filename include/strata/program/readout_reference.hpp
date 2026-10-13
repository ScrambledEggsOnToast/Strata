#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace strata::program::readout_reference {

struct BranchReadout {
    int64_t prefix_tokens = 0, branch_index = 0;
    bool restoration_verified = false;
    double checkpoint_ms = 0, restore_ms = 0, call_elapsed_ms = 0;
};

// No top-k truncation and no skipped targets: validate the entire submitted set
// before writing any values. The caller owns the final-position full-vocab row.
inline bool valid_ids(const std::vector<int64_t>& ids, int64_t vocab, std::string& error) {
    if (ids.empty() || vocab <= 0) { error = "reference: empty token set or vocabulary"; return false; }
    for (int64_t id : ids) {
        if (id < 0 || id >= vocab) {
            error = "reference: token outside vocabulary: " + std::to_string(id);
            return false;
        }
    }
    return true;
}

inline bool write(const std::string& path, const float* row, int64_t vocab, int64_t consumed,
                  const std::vector<int64_t>& selected, std::string& error, BranchReadout branch = {}) {
    if (!row || consumed < 1 || !valid_ids(selected, vocab, error)) {
        if (error.empty()) error = "reference: missing final-position row";
        return false;
    }
    if (branch.prefix_tokens < 0 || branch.prefix_tokens >= consumed || branch.branch_index < 0 ||
        (branch.prefix_tokens > 0 && (!branch.restoration_verified || !std::isfinite(branch.checkpoint_ms) ||
         !std::isfinite(branch.restore_ms) || !std::isfinite(branch.call_elapsed_ms) ||
         branch.checkpoint_ms < 0 || branch.restore_ms < 0 ||
         branch.call_elapsed_ms < branch.checkpoint_ms + branch.restore_ms))) {
        error = "reference: unverified prefix restoration or invalid branch timing";
        return false;
    }
    double maximum = -INFINITY;
    for (int64_t v = 0; v < vocab; ++v) {
        if (!std::isfinite(row[v])) { error = "reference: nonfinite full-vocabulary row"; return false; }
        maximum = std::max(maximum, (double) row[v]);
    }
    double sum = 0;
    for (int64_t v = 0; v < vocab; ++v) sum += std::exp((double) row[v] - maximum);
    const double lse = maximum + std::log(sum);
    std::FILE* file = std::fopen(path.c_str(), "wx");
    if (!file) { error = "reference: cannot create output: " + path; return false; }
    std::fprintf(file, "{\"schema_version\":1,\"mode\":\"%s\",\"engine_mode\":\"serial_solo\","
                       "\"position\":%lld,\"consumed_prompt_tokens\":%lld,\"row_position\":%lld,"
                       "\"prefix_tokens\":%lld,\"generated_tokens\":0,\"sampling\":false,\"mtp\":false,"
                       "\"n_vocab\":%lld,\"vocab_logsumexp\":%.17g,\"selected_token_ids\":[",
                 branch.prefix_tokens > 0 ? "reused" : "fresh", (long long) consumed, (long long) consumed,
                 (long long) (consumed - 1), (long long) branch.prefix_tokens, (long long) vocab, lse);
    for (size_t i = 0; i < selected.size(); ++i)
        std::fprintf(file, "%s%lld", i ? "," : "", (long long) selected[i]);
    std::fprintf(file, "],\"logits\":[");
    for (size_t i = 0; i < selected.size(); ++i)
        std::fprintf(file, "%s%.9g", i ? "," : "", (double) row[selected[i]]);
    std::fprintf(file, "],\"log_probs\":[");
    for (size_t i = 0; i < selected.size(); ++i)
        std::fprintf(file, "%s%.17g", i ? "," : "", (double) row[selected[i]] - lse);
    if (branch.prefix_tokens > 0)
        std::fprintf(file, "],\"branch_index\":%lld,\"restoration_verified\":true,\"checkpoint_ms\":%.9g,"
                           "\"restore_ms\":%.9g,\"call_elapsed_ms\":%.9g}\n", (long long) branch.branch_index,
                     branch.checkpoint_ms, branch.restore_ms, branch.call_elapsed_ms);
    else std::fprintf(file, "]}\n");
    const bool ok = std::ferror(file) == 0;
    const bool closed = std::fclose(file) == 0;
    if (!ok || !closed) {
        std::remove(path.c_str());
        error = "reference: output write failed";
        return false;
    }
    return true;
}

} // namespace strata::program::readout_reference
