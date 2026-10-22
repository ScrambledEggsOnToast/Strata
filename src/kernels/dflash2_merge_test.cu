// Explicit queued GPU fixture: retained serial merge/replay is the reference.
#include "cuda/dflash2_selector.cu"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <limits>
using namespace strata::kernels::dflash2;
static void check(int code){if(code!=int(cudaSuccess)){std::fprintf(stderr,"%s\n",cudaGetErrorString(cudaError_t(code)));std::exit(1);}}
static void run(int rows) {
    std::vector<float> logits(size_t(rows)*selector_vocabulary);
    float* device_logits;SelectorWorkspace *a,*b;
    check(cudaMalloc(&device_logits,logits.size()*4));
    uint8_t *raw_a,*raw_b;const size_t bytes=sizeof(SelectorWorkspace)+32;
    check(cudaMalloc(&raw_a,bytes));check(cudaMalloc(&raw_b,bytes));check(cudaMemset(raw_a,0xa5,bytes));check(cudaMemset(raw_b,0xa5,bytes));
    a=reinterpret_cast<SelectorWorkspace*>(raw_a+16);b=reinterpret_cast<SelectorWorkspace*>(raw_b+16);
    check(cudaMemset(a,0,sizeof(*a)));check(cudaMemset(b,0,sizeof(*b)));
    cudaStream_t stream;check(cudaStreamCreate(&stream));
    auto launch=[&] {merge_top_kernel<<<rows,32,0,stream>>>(a);merge_top_parallel_kernel<<<rows,256,0,stream>>>(b);replay_ties_kernel<<<rows,32,0,stream>>>(device_logits,a);replay_ties_kernel<<<rows,32,0,stream>>>(device_logits,b);check(cudaGetLastError());};
    cudaGraph_t graph;cudaGraphExec_t exec;check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));launch();check(cudaStreamEndCapture(stream,&graph));check(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    std::vector<uint8_t> host_a(bytes),host_b(bytes);
    auto compare=[&](int mode) {
        check(cudaStreamSynchronize(stream));check(cudaMemcpy(host_a.data(),raw_a,bytes,cudaMemcpyDeviceToHost));check(cudaMemcpy(host_b.data(),raw_b,bytes,cudaMemcpyDeviceToHost));
        const auto* x=reinterpret_cast<const SelectorWorkspace*>(host_a.data()+16);const auto* y=reinterpret_cast<const SelectorWorkspace*>(host_b.data()+16);
        if(x->result.status!=y->result.status||x->result.status!=(mode==5?selector_nonfinite_logits:0))std::abort();
        for(int row=0;row<rows;++row) {
            if(x->replay[row]!=y->replay[row]||x->replay[row]!=(mode>0&&mode<5?1:0))std::abort();
            if(mode!=5 && (std::memcmp(x->result.candidates[row],y->result.candidates[row],sizeof(x->result.candidates[row]))||std::memcmp(x->unary[row],y->unary[row],sizeof(x->unary[row])))) {
                std::fprintf(stderr,"merge mismatch rows%d mode%d row%d\n",rows,mode,row);std::abort();
            }
        }
        for(size_t i=0;i<bytes;++i)if((i<16||i>=bytes-16)&&(host_a[i]!=0xa5||host_b[i]!=0xa5))std::abort();
    };
    // Reuse workspace across unique -> tied -> nonfinite -> unique calls.
    for(int mode:{0,1,2,3,4,5,0}) {
        for(int row=0;row<rows;++row) {
            float* p=logits.data()+size_t(row)*selector_vocabulary;
            for(int token=0;token<selector_vocabulary;++token)p[token]=float((int64_t(token)*7919+row*37)%selector_vocabulary)-300000.f;
            if(mode==1)for(int i=0;i<20;++i)p[i]=1000000.f;
            if(mode==2)for(int i=0;i<20;++i)p[i*selector_tile_width+3]=1000000.f;
            if(mode==3){for(int i=0;i<15;++i)p[i*selector_tile_width+2]=2000000.f+i;p[20*selector_tile_width]=1000000.f;p[30*selector_tile_width]=1000000.f;}
            if(mode==4)for(int i=0;i<20;++i)p[i*selector_tile_width+1]=i%2?-0.f:0.f;
            if(mode==5)p[17]=std::numeric_limits<float>::quiet_NaN();
        }
        check(cudaMemcpyAsync(device_logits,logits.data(),logits.size()*4,cudaMemcpyHostToDevice,stream));
        auto prepare=[&] {
            // Match public selector: reset only result, not stale replay/unary.
            check(cudaMemsetAsync(&a->result,0,sizeof(SelectorResult),stream));
            tile_top_kernel<<<dim3(selector_tiles,rows),256,0,stream>>>(device_logits,a);check(cudaGetLastError());
            check(cudaMemcpyAsync(b,a,sizeof(*a),cudaMemcpyDeviceToDevice,stream));
        };
        prepare();launch();compare(mode);
        prepare();check(cudaGraphLaunch(exec,stream));compare(mode);
        if(mode==0) {
            cudaEvent_t start,end;check(cudaEventCreate(&start));check(cudaEventCreate(&end));float times[2];
            for(int variant=0;variant<2;++variant){check(cudaEventRecord(start,stream));for(int i=0;i<5;++i){if(variant)merge_top_parallel_kernel<<<rows,256,0,stream>>>(b);else merge_top_kernel<<<rows,32,0,stream>>>(a);}check(cudaEventRecord(end,stream));check(cudaEventSynchronize(end));check(cudaEventElapsedTime(times+variant,start,end));times[variant]/=5;}
            std::printf("{\"rows\":%d,\"reference_ms\":%.6f,\"candidate_ms\":%.6f}\n",rows,times[0],times[1]);check(cudaEventDestroy(start));check(cudaEventDestroy(end));
        }
        std::printf("{\"rows\":%d,\"mode\":%d,\"exact\":true}\n",rows,mode);
    }
    check(cudaGraphExecDestroy(exec));check(cudaGraphDestroy(graph));check(cudaStreamDestroy(stream));check(cudaFree(raw_a));check(cudaFree(raw_b));check(cudaFree(device_logits));
}
int main(){for(int rows:{1,3,7})run(rows);return 0;}
