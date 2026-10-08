#include "strata/core/expert_formation.hpp"
#include <array>
#include <stdexcept>

namespace {
constexpr int rows = 5, width = 2, hidden = 4;
struct Job {
    const uint8_t* blob = nullptr;
    int nt = 0;
    const float* act[rows] = {};
    const void* nact[rows] = {};
    float* out[rows] = {};
};
void require(bool ok) { if (!ok) throw std::runtime_error("expert formation mismatch"); }
void run(bool separate) {
    const int experts[rows * width] = {7,11,7,13,3,13,7,21,5,21};
    const uint8_t weights[32] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
                                 16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31};
    float input[rows][hidden], output[rows * width][hidden] = {};
    for (int t=0; t<rows; ++t) for (int c=0; c<hidden; ++c) input[t][c] = float(t*10+c+1);
    std::array<int16_t,32> lookup; lookup.fill(-1);
    Job jobs[rows * width];
    int count=0, fetches=0;
    for (int t=0; t<rows; ++t) {
        if (separate && t==3) lookup.fill(-1); // two unequal request spans
        for (int j=0; j<width; ++j) {
            const int i=t*width+j, e=experts[i];
            bool created=false;
            require(strata::core::append_expert_row(jobs,lookup[e],count,rows,input[t],
                input[t],output[i],[&] { ++fetches; return &weights[e]; },created));
        }
    }
    require(count==(separate?7:6) && fetches==count);
    std::array<int,rows * width> writes{};
    // The job consumer uses exactly the production's stored activation/result addresses.
    for (int j=0; j<count; ++j) for (int r=0; r<jobs[j].nt; ++r) {
        require(jobs[j].nact[r]==jobs[j].act[r]);
        for (int c=0; c<hidden; ++c) jobs[j].out[r][c]=*jobs[j].blob*jobs[j].act[r][c];
        for (int i=0; i<rows*width; ++i) if (jobs[j].out[r]==output[i]) ++writes[i];
    }
    for (int i=0; i<rows*width; ++i) {
        require(writes[i]==1);
        for (int c=0; c<hidden; ++c) require(output[i][c]==experts[i]*input[i/width][c]);
    }
    const int prior=count;
    int16_t missing=-1; bool created=false;
    require(!strata::core::append_expert_row(jobs,missing,count,rows,input[0],nullptr,output[0],
                                           []() -> const uint8_t* { return nullptr; },created));
    require(missing==-1 && count==prior); // failed source never consumes a job slot
}
void full_job_splits_without_losing_rows() {
    Job jobs[2]; int16_t index=-1; int count=0; uint8_t blob=3;
    float input[2]={2,4}, out[2]={0,0};
    for (int i=0;i<2;++i) {
        bool created=false;
        require(strata::core::append_expert_row(jobs,index,count,1,&input[i],nullptr,&out[i],
                                               [&] { return &blob; },created));
        require(created);
    }
    require(count==2 && jobs[0].out[0]==&out[0] && jobs[1].out[0]==&out[1]);
}
}
int main() { run(false); run(true); full_job_splits_without_losing_rows(); }
