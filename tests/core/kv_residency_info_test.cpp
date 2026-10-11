// Engine INFO serialization only: supplied initialized-state fields, no CUDA allocation or model execution.
#include "strata/program/kv_residency_info.hpp"
#include <cstdio>
#include <cstring>

namespace {
struct State {
    int kv_mode;
    int64_t n_slots;
};
bool report_is(const char* role, const State* main, const State* draft, const char* expected) {
    FILE* out = std::tmpfile();
    if (!out) {
        std::perror("kv_residency_info_test: tmpfile");
        return false;
    }
    strata::program::write_kv_residency_info(out, role, main, draft, 4);
    std::rewind(out);
    char actual[512] = {};
    const size_t n = std::fread(actual, 1, sizeof(actual) - 1, out);
    const bool ok = !std::ferror(out) && n == std::strlen(expected) && std::strcmp(actual, expected) == 0;
    std::fclose(out);
    if (!ok) std::fprintf(stderr, "kv_residency_info_test: expected %sreceived %s", expected, actual);
    return ok;
}
} // namespace

int main() {
    const State resident{0, 65536}, streamed{1, 16384}, ring{2, 8208}, fits{0, 1024};
    if (!report_is("working", &resident, &ring,
                   "INFO working_kv_mode=resident working_kv_window=262144 "
                   "working_draft_kv_mode=streamed working_draft_kv_window=32832\n")) return 1;
    if (!report_is("slot_0", &streamed, &streamed,
                   "INFO slot_0_kv_mode=streamed slot_0_kv_window=65536 "
                   "slot_0_draft_kv_mode=streamed slot_0_draft_kv_window=65536\n")) return 1;
    // A fully resident initialized draft must report resident even if a streamed draft was requested.
    if (!report_is("slot_1", &streamed, &resident,
                   "INFO slot_1_kv_mode=streamed slot_1_kv_window=65536 "
                   "slot_1_draft_kv_mode=resident slot_1_draft_kv_window=262144\n")) return 1;
    // A context that fits the window initializes in mode 0, not in the requested streaming mode.
    if (!report_is("slot_0", &fits, &fits,
                   "INFO slot_0_kv_mode=resident slot_0_kv_window=4096 "
                   "slot_0_draft_kv_mode=resident slot_0_draft_kv_window=4096\n")) return 1;
    if (!report_is("slot_0", &resident, nullptr,
                   "INFO slot_0_kv_mode=resident slot_0_kv_window=262144 "
                   "slot_0_draft_kv_mode=off slot_0_draft_kv_window=0\n")) return 1;
    std::puts("kv_residency_info_test: 5 serialization cases passed");
    return 0;
}
