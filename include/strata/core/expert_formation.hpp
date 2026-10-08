#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::core {

// Append one CPU-owned routed row to the actual dispatch's bounded job tables.
// The caller validates the expert and table capacities before entry. Fetch is
// called only when a new expert job is required; null leaves the tables intact.
// Rows remain unweighted and retain their router-index output address.
template<class Job, class Act, class Fetch>
bool append_expert_row(Job* jobs, int16_t& expert_job, int& job_count, int max_rows,
                       const Act* activation, const void* native_activation, float* output,
                       Fetch&& fetch, bool& created) {
    created = expert_job < 0 || jobs[static_cast<size_t>(expert_job)].nt == max_rows;
    if (created) {
        const uint8_t* blob = fetch();
        if (blob == nullptr) return false;
        expert_job = static_cast<int16_t>(job_count++);
        jobs[static_cast<size_t>(expert_job)].blob = blob;
        jobs[static_cast<size_t>(expert_job)].nt = 0;
    }
    Job& job = jobs[static_cast<size_t>(expert_job)];
    job.act[job.nt] = activation;
    job.nact[job.nt] = native_activation;
    job.out[job.nt] = output;
    ++job.nt;
    return true;
}

} // namespace strata::core
