// Queued GPU fixture. Including the implementation exposes the retained reference
// kernels without adding a test-only entry point to the engine API.
#include "cuda/dflash2_forward.cu"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <array>

using namespace strata::kernels;
using namespace strata::kernels::dflash2;
static void check(cudaError_t code) {
    if (code != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(code)); std::exit(1); }
}
static uint16_t value(size_t i) {
    // Distinct rows/columns, cancellation, small and large finite magnitudes.
    static const uint16_t data[] = {0x3f80,0xbf80,0x3b81,0xbb81,0x4101,0xc101,0x0001,0x8001,0x3f01,0x3f7f,0xbf7f};
    return data[((i * 1664525u + (i >> 3) * 1013904223u) >> 7) % 11];
}
static void run(int rows, int ni, int no, bool exceptional) {
    const size_t count = size_t(rows) * no;
    std::vector<uint16_t> x(size_t(rows)*ni), w(size_t(no)*ni), a(count+32), b(count+32);
    for (size_t i=0;i<x.size();++i) x[i]=value(i+37);
    for (size_t i=0;i<w.size();++i) w[i]=value(i+101);
    if(exceptional) { x[ni/2]=0x7f80; w[ni-1]=0x7fc0; }
    uint16_t *dx,*dw,*da,*db; uint32_t *sa,*sb;
    check(cudaMalloc(&dx,x.size()*2));check(cudaMalloc(&dw,w.size()*2));
    check(cudaMalloc(&da,a.size()*2));check(cudaMalloc(&db,b.size()*2));
    check(cudaMalloc(&sa,4));check(cudaMalloc(&sb,4));
    check(cudaMemcpy(dx,x.data(),x.size()*2,cudaMemcpyHostToDevice));
    check(cudaMemcpy(dw,w.data(),w.size()*2,cudaMemcpyHostToDevice));
    cudaStream_t stream;check(cudaStreamCreate(&stream));
    auto launch=[&](bool candidate) {
        uint16_t* out=(candidate?db:da)+16;uint32_t* status=candidate?sb:sa;
        if(rows<=block) {
            if(candidate) linear_small_reuse<<<dim3((no+7)/8,(rows+3)/4),256,0,stream>>>(dx,dw,out,rows,ni,no,status);
            else linear_small<<<dim3((no+7)/8,rows),256,0,stream>>>(dx,dw,out,ni,no,status);
        } else {
            if(candidate) linear_context_reuse<<<dim3((no+63)/64,(rows+63)/64),dim3(16,16),0,stream>>>(dx,dw,out,rows,ni,no,status);
            else linear_context<<<dim3((no+15)/16,(rows+15)/16),dim3(16,16),0,stream>>>(dx,dw,out,rows,ni,no,status);
        }
        check(cudaGetLastError());
    };
    auto compare=[&] {
        check(cudaStreamSynchronize(stream));
        check(cudaMemcpy(a.data(),da,a.size()*2,cudaMemcpyDeviceToHost));
        check(cudaMemcpy(b.data(),db,b.size()*2,cudaMemcpyDeviceToHost));
        uint32_t ra,rb;check(cudaMemcpy(&ra,sa,4,cudaMemcpyDeviceToHost));check(cudaMemcpy(&rb,sb,4,cudaMemcpyDeviceToHost));
        if(ra!=rb || ra != (exceptional ? 1u : 0u)) std::abort();
        for(size_t i=0;i<a.size();++i) {
            const bool guard=i<16||i>=count+16;
            const bool nan=(a[i]&0x7f80)==0x7f80&&(a[i]&0x7f)&&(b[i]&0x7f80)==0x7f80&&(b[i]&0x7f);
            if((guard&&(a[i]!=0xa5a5||b[i]!=0xa5a5)) || (a[i]!=b[i]&&!nan)) {
                std::fprintf(stderr,"mismatch rows=%d ni=%d no=%d i=%zu %04x %04x\n",rows,ni,no,i,a[i],b[i]);std::abort();
            }
        }
    };
    check(cudaMemsetAsync(da,0xa5,a.size()*2,stream));check(cudaMemsetAsync(db,0xa5,b.size()*2,stream));
    check(cudaMemsetAsync(sa,0,4,stream));check(cudaMemsetAsync(sb,0,4,stream));
    launch(false);launch(true);compare();
    // Production dispatch uses independently frozen environment toggles.
    check(cudaMemsetAsync(db,0xa5,b.size()*2,stream));check(cudaMemsetAsync(sb,0,4,stream));
    check(static_cast<cudaError_t>(linear_bf16(dx,dw,db+16,rows,ni,no,stream,sb)));compare();
    cudaGraph_t graph;cudaGraphExec_t exec;
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));launch(false);launch(true);
    check(cudaStreamEndCapture(stream,&graph));check(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    // Replays must read current input rather than stale captured host contents.
    x.back()=0x3e81;check(cudaMemcpyAsync(dx,x.data(),x.size()*2,cudaMemcpyHostToDevice,stream));
    for(int i=0;i<2;++i) {
        check(cudaMemsetAsync(da,0xa5,a.size()*2,stream));check(cudaMemsetAsync(db,0xa5,b.size()*2,stream));
        check(cudaMemsetAsync(sa,0,4,stream));check(cudaMemsetAsync(sb,0,4,stream));
        check(cudaGraphLaunch(exec,stream));compare();
    }
    float times[2]{};cudaEvent_t start,end;check(cudaEventCreate(&start));check(cudaEventCreate(&end));
    for(int mode=0;mode<2;++mode) {
        check(cudaEventRecord(start,stream));for(int i=0;i<3;++i)launch(mode);
        check(cudaEventRecord(end,stream));check(cudaEventSynchronize(end));check(cudaEventElapsedTime(times+mode,start,end));times[mode]/=3;
    }
    std::printf("{\"rows\":%d,\"ni\":%d,\"no\":%d,\"nonfinite\":%s,\"reference_ms\":%.6f,\"candidate_ms\":%.6f,\"exact\":true}\n",rows,ni,no,exceptional?"true":"false",times[0],times[1]);
    check(cudaEventDestroy(start));check(cudaEventDestroy(end));check(cudaGraphExecDestroy(exec));check(cudaGraphDestroy(graph));
    check(cudaStreamDestroy(stream));check(cudaFree(dx));check(cudaFree(dw));check(cudaFree(da));check(cudaFree(db));check(cudaFree(sa));check(cudaFree(sb));
}
int main() {
    setenv("STRATA_DFLASH2_LINEAR_REUSE","1",1);setenv("STRATA_DFLASH2_LINEAR_CONTEXT","1",1);
    for(int rows:{1,3,4,7,8,9,17,63,64,65})run(rows,67,71,false);
    for(auto shape:std::vector<std::array<int,3>>{{4,12800,2560},{4,2560,7680},{4,7680,2560},{64,12800,2560},{128,12800,2560},{220,2560,1024},{257,12800,2560},{2047,2560,2560}})run(shape[0],shape[1],shape[2],false);
    run(3,67,71,true);run(65,67,71,true);
    return 0;
}
