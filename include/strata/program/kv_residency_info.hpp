#pragma once
#include <cstdint>
#include <cstdio>

namespace strata::program {

// QsaState's initialized mode and pool extent, never the requested placement.
// Template keeps this serialization seam CUDA-free. Both a sparse stream (1)
// and a draft ring (2) have host-backed K/V; an absent draft is explicitly off.
template<class State>
inline void write_kv_residency_info(FILE* out, const char* role, const State* main,
                                    const State* draft, int64_t page_size) {
    std::fprintf(out, "INFO %s_kv_mode=%s %s_kv_window=%lld %s_draft_kv_mode=%s %s_draft_kv_window=%lld\n",
                 role, main ? (main->kv_mode == 0 ? "resident" : "streamed") : "off",
                 role, (long long) (main ? main->n_slots * page_size : 0),
                 role, draft ? (draft->kv_mode == 0 ? "resident" : "streamed") : "off",
                 role, (long long) (draft ? draft->n_slots * page_size : 0));
}

} // namespace strata::program
