#include "cuda/iq_kernels.cu"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

namespace K = strata::kernels;
namespace {
void ck(cudaError_t e) { if(e!=cudaSuccess){std::fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e));std::exit(1);} }
void req(bool x,const char* why){if(!x){std::fprintf(stderr,"FAIL %s\n",why);std::exit(1);}}
bool reuse_enabled=true;
constexpr size_t guard=128;
constexpr int max_groups=40,max_entries=64,max_tokens=8;
uint32_t random_word(uint32_t& s){s^=s<<13;s^=s>>17;s^=s<<5;return s;}
template<class T> T* alloc(size_t n){T* p=nullptr;ck(cudaMalloc(&p,n*sizeof(T)));return p;}

// The part witness exercises all512 grid indices and all256 sign bytes, including full-range q8 input.
__global__ void part_witness(const block_iq3_s* w,const block_q8_1* x,float* a,float* b,int count){
    __shared__ uint32_t grid[512];
    for(int i=threadIdx.x;i<512;i+=blockDim.x)grid[i]=iq3s_grid[i];
    __syncthreads();
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count*8)return;
    int part=i%8,block=i/8;
    a[i]=K::DotG<21,1>::f(w,x+block*8,block,part*2,grid);
    auto decoded=K::Split<21>::load<true>(w,block,part*2,grid);
    b[i]=K::Split<21>::apply(decoded,x+block*8,part*2);
}
void parts(){
    constexpr int count=512;std::vector<block_iq3_s>w(count);std::vector<block_q8_1>x(count*8);
    for(int b=0;b<count;++b){
        auto& v=w[b];v.d=__float2half((b&1)?-0.03125f:0.03125f);
        std::memset(v.qs,b&255,sizeof(v.qs));std::memset(v.qh,b>=256?255:0,sizeof(v.qh));
        std::memset(v.signs,b&255,sizeof(v.signs));std::memset(v.scales,(b*17)&255,sizeof(v.scales));
        for(int p=0;p<8;++p){auto& q=x[b*8+p];q.ds=__floats2half2_rn((p&1)?-0.125f:0.125f,0);
            for(int j=0;j<32;++j)q.qs[j]=int8_t(((b*13+p*7+j*29)&255)-128);}
    }
    auto dw=alloc<block_iq3_s>(w.size());auto dx=alloc<block_q8_1>(x.size());auto da=alloc<float>(count*8);auto db=alloc<float>(count*8);
    ck(cudaMemcpy(dw,w.data(),w.size()*sizeof(w[0]),cudaMemcpyHostToDevice));ck(cudaMemcpy(dx,x.data(),x.size()*sizeof(x[0]),cudaMemcpyHostToDevice));
    part_witness<<<16,256>>>(dw,dx,da,db,count);ck(cudaGetLastError());ck(cudaDeviceSynchronize());
    std::vector<float>a(count*8),b(count*8);ck(cudaMemcpy(a.data(),da,a.size()*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(b.data(),db,b.size()*4,cudaMemcpyDeviceToHost));
    req(std::memcmp(a.data(),b.data(),a.size()*4)==0,"IQ3_S part exactness");
    ck(cudaFree(dw));ck(cudaFree(dx));ck(cudaFree(da));ck(cudaFree(db));std::puts("PASS part grid512 signs256 q8fullrange exact=1");
}

struct Fixture {
    K::NativeExpertLayout L;
    std::vector<uint8_t> weights;
    std::vector<block_q8_1> input;
    std::vector<int> starts,tokens,destinations;
    std::vector<unsigned long long> pointers;
    uint8_t *dw,*ha,*hb,*oa,*ob,*public_scratch;
    block_q8_1* dx;
    int *ds,*dt,*dd,*dg;
    unsigned long long* dp;
    unsigned* counts;
    cudaStream_t stream;
    int groups=0,entries=0,cap_groups=0;
    Fixture(int k,int n):L(K::native_expert_layout(21,20,k,n)),weights(max_groups*L.bytes),input(max_tokens*k/32),
        starts(max_groups+1),tokens(max_entries),destinations(max_entries),pointers(max_groups){
        uint32_t seed=0x7821543u;
        for(auto& v:weights)v=uint8_t(random_word(seed));
        for(int g=0;g<max_groups;++g){
            auto blob=weights.data()+g*L.bytes;
            for(int row=0;row<2*n;++row)for(int b=0;b<k/256;++b){
                auto* v=reinterpret_cast<block_iq3_s*>(blob+row*L.gu_row)+b;
                v->d=__float2half((row&1)?-0.00048828125f:0.00048828125f);
            }
            for(int row=0;row<k;++row)for(int b=0;b<n/32;++b){
                auto* v=reinterpret_cast<block_iq4_nl*>(blob+L.down_off+row*L.d_row)+b;
                v->d=__float2half((row&1)?-0.0078125f:0.0078125f);
            }
        }
        dw=alloc<uint8_t>(weights.size());dx=alloc<block_q8_1>(input.size());
        ha=alloc<uint8_t>(hq_bytes()+2*guard);
        public_scratch=alloc<uint8_t>(K::native_expert_scratch_bytes(max_entries,L.n_ff)+guard);
        const size_t f=size_t(max_entries)*L.n_ff*4,fa=(f+255)&~size_t(255);
        hb=public_scratch+3*fa-guard; // guarded view of public API's actual q8 scratch output
        oa=alloc<uint8_t>(out_bytes()+2*guard);ob=alloc<uint8_t>(out_bytes()+2*guard);
        ds=alloc<int>(starts.size());dt=alloc<int>(tokens.size());dd=alloc<int>(destinations.size());dg=alloc<int>(1);
        dp=alloc<unsigned long long>(pointers.size());counts=alloc<unsigned>(8);
        ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        for(int g=0;g<max_groups;++g)pointers[g]=(unsigned long long)(dw+g*L.bytes);
        ck(cudaMemcpyAsync(dw,weights.data(),weights.size(),cudaMemcpyHostToDevice,stream));
        ck(cudaMemcpyAsync(dp,pointers.data(),pointers.size()*8,cudaMemcpyHostToDevice,stream));
        ck(cudaStreamSynchronize(stream));
    }
    ~Fixture(){ck(cudaStreamSynchronize(stream));ck(cudaStreamDestroy(stream));
        for(void*p:{(void*)dw,(void*)dx,(void*)ha,(void*)public_scratch,(void*)oa,(void*)ob,(void*)ds,(void*)dt,(void*)dd,(void*)dg,(void*)dp,(void*)counts})ck(cudaFree(p));}
    size_t hq_bytes()const{return size_t(max_entries)*(L.n_ff/32)*sizeof(block_q8_1);}
    size_t out_bytes()const{return size_t(max_entries)*L.n_embd*sizeof(float);}
    void set(const std::vector<int>& sizes,int mode=0,int token_pool=max_tokens){
        groups=int(sizes.size());entries=std::accumulate(sizes.begin(),sizes.end(),0);cap_groups=std::min(max_groups,groups+2);
        req(groups<=max_groups&&entries<=max_entries,"fixture bounds");starts.assign(max_groups+1,entries);starts[0]=0;
        for(int g=0;g<groups;++g)starts[g+1]=starts[g]+sizes[g];
        for(int e=0;e<max_entries;++e){tokens[e]=(e*3+1)%token_pool;destinations[e]=(e*17+5)%max_entries;}
        for(size_t b=0;b<input.size();++b){auto&q=input[b];
            q.ds=__floats2half2_rn((b&1)?-0.03125f:0.03125f,0);
            for(int j=0;j<32;++j)q.qs[j]=int8_t(int((b*19+j*31)%255)-127);
            if(mode==1)q.ds=__floats2half2_rn(0,0);
            if(mode==2)q.ds=__floats2half2_rn(65504.0f,65504.0f);
            if(mode==3&&b%13==0)q.ds=__floats2half2_rn(INFINITY,0);
            if(mode==4&&b%13==0)q.ds=__floats2half2_rn(NAN,0);
            if(mode==5)q.ds=__floats2half2_rn((b&1)?-0x1p-24f:0x1p-24f,0);
        }
        ck(cudaMemcpyAsync(ds,starts.data(),starts.size()*4,cudaMemcpyHostToDevice,stream));
        ck(cudaMemcpyAsync(dt,tokens.data(),tokens.size()*4,cudaMemcpyHostToDevice,stream));
        ck(cudaMemcpyAsync(dd,destinations.data(),destinations.size()*4,cudaMemcpyHostToDevice,stream));
        ck(cudaMemcpyAsync(dg,&groups,4,cudaMemcpyHostToDevice,stream));
        ck(cudaMemcpyAsync(dx,input.data(),input.size()*sizeof(input[0]),cudaMemcpyHostToDevice,stream));
        reset();ck(cudaStreamSynchronize(stream));
    }
    void reset(){ck(cudaMemsetAsync(public_scratch,0xa5,K::native_expert_scratch_bytes(max_entries,L.n_ff)+guard,stream));
        ck(cudaMemsetAsync(ha,0xa5,hq_bytes()+2*guard,stream));
        for(auto p:{oa,ob})ck(cudaMemsetAsync(p,0xa5,out_bytes()+2*guard,stream));ck(cudaMemsetAsync(counts,0,8*4,stream));}
    void launch(bool candidate,bool down=true,bool count=false){
        auto hq=reinterpret_cast<block_q8_1*>((candidate?hb:ha)+guard);auto out=reinterpret_cast<float*>((candidate?ob:oa)+guard);
        dim3 gu(unsigned(L.n_ff/32),unsigned(cap_groups));size_t shared=4*(L.n_embd/32)*sizeof(block_q8_1);
        if(candidate){
            req(down,"public fixture measures only whole two-launch operation");
            K::native_expert_grouped(L,dp,ds,dg,dd,dt,cap_groups,max_entries,dx,public_scratch,out,stream,0,
                                    count?counts:nullptr);
        }else {
            K::native_gu_fused_kernel<21><<<gu,256,shared,stream>>>(dp,ds,dg,dt,dx,L,hq);
            if(down)K::native_down_lds_kernel<20><<<dim3(unsigned((L.n_embd+63)/64),unsigned(cap_groups)),256,4*(L.n_ff/32)*sizeof(block_q8_1),stream>>>(dp,ds,dg,dd,hq,L,out);
        }
        ck(cudaGetLastError());
    }
    std::vector<uint8_t> compare(const std::vector<int>& sizes,bool counter=true){
        ck(cudaStreamSynchronize(stream));std::vector<uint8_t>a(hq_bytes()+2*guard),b(a.size()),c(out_bytes()+2*guard),d(c.size());
        ck(cudaMemcpy(a.data(),ha,a.size(),cudaMemcpyDeviceToHost));ck(cudaMemcpy(b.data(),hb,b.size(),cudaMemcpyDeviceToHost));
        ck(cudaMemcpy(c.data(),oa,c.size(),cudaMemcpyDeviceToHost));ck(cudaMemcpy(d.data(),ob,d.size(),cudaMemcpyDeviceToHost));
        req(a==b,"fused q8_1 bytes differ");req(c==d,"whole expert float output differs");
        for(size_t i=0;i<a.size();++i)if(i<guard||i>=guard+size_t(entries)*(L.n_ff/32)*sizeof(block_q8_1))req(a[i]==0xa5,"hq guard/tail modified");
        std::array<bool,max_entries> active{};for(int e=0;e<entries;++e)active[destinations[e]]=true;
        for(size_t i=0;i<c.size();++i)if(i<guard||i>=guard+out_bytes()||!active[(i-guard)/(L.n_embd*4)])req(c[i]==0xa5,"output guard/unrouted slot modified");
        if(counter){unsigned got[8],want[8]{};ck(cudaMemcpy(got,counts,32,cudaMemcpyDeviceToHost));
            for(int n:sizes)if(n&&reuse_enabled){++want[std::min(n,5)-1];want[7]+=n/4;if(n%4>=2)++want[3+n%4];}
            req(std::memcmp(got,want,32)==0,"completed group/chunk counter mismatch");}
        return d;
    }
    void non_mode8_fallback(){std::vector<int> sizes{1,2,3,4};set(sizes);
        K::native_expert_set_mode(7,0);launch(true,true,true);K::native_expert_set_mode(8,0);
        ck(cudaStreamSynchronize(stream));
        unsigned got[8]{};ck(cudaMemcpy(got,counts,sizeof(got),cudaMemcpyDeviceToHost));
        req(std::all_of(std::begin(got),std::end(got),[](unsigned v){return v==0;}),"mode7 unexpectedly reused candidate");
        std::puts("PASS public_mode7_counters_untouched=1");
    }
    void exact(const std::vector<int>& sizes,int mode=0){set(sizes,mode);launch(false);launch(true,true,true);compare(sizes);
        std::printf("EXACT k=%lld ff=%lld groups=%d entries=%d mode=%d\n",(long long)L.n_embd,(long long)L.n_ff,groups,entries,mode);}
    void graph(){std::vector<int> sizes{2,3,4,1};set(sizes);cudaGraph_t graph;cudaGraphExec_t exec;
        ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));launch(false);launch(true);ck(cudaStreamEndCapture(stream,&graph));
        size_t nodes=0;ck(cudaGraphGetNodes(graph,nullptr,&nodes));req(nodes==4,"captured pair must preserve exactly two launches per arm");
        ck(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));ck(cudaGraphLaunch(exec,stream));auto before=compare(sizes,false);
        // Change the already-bound input allocation, then replay with unchanged kernel/node addresses.
        for(auto&q:input){q.ds=__floats2half2_rn(0,0);std::memset(q.qs,0,32);}
        ck(cudaMemcpyAsync(dx,input.data(),input.size()*sizeof(input[0]),cudaMemcpyHostToDevice,stream));
        reset();ck(cudaGraphLaunch(exec,stream));auto after=compare(sizes,false);req(before!=after,"finite changed-input graph witness missing");
        // Same graph also observes changed device group count (all writes must disappear).
        groups=0;entries=0;ck(cudaMemcpyAsync(dg,&groups,4,cudaMemcpyHostToDevice,stream));reset();ck(cudaGraphLaunch(exec,stream));compare({},false);
        ck(cudaGraphExecDestroy(exec));ck(cudaGraphDestroy(graph));std::puts("PASS graph nodes4 changed_input=1 changed_groups=1 guards=1");
    }
};
}
int main(int argc,char**argv){req(argc==1||(argc==2&&std::strcmp(argv[1],"--off")==0),"usage: native_expert_reuse_test [--off]");
    reuse_enabled=argc==1;
#ifdef _WIN32
    req(_putenv_s("STRATA_EXPERT_REUSE",reuse_enabled?"1":"0")==0,"set reuse environment");
#else
    req(setenv("STRATA_EXPERT_REUSE",reuse_enabled?"1":"0",1)==0,"set reuse environment");
#endif
    req(K::expert_reuse_arch(7,0)&&!K::expert_reuse_arch(7,5)&&!K::expert_reuse_arch(8,6)&&!K::expert_reuse_arch(6,0),"SM70-only predicate");
    K::native_expert_set_mode(8,0);cudaDeviceProp p{};ck(cudaGetDeviceProperties(&p,0));req(p.major==7&&p.minor==0,"fixture requires SM70");
    std::printf("DEVICE name=%s sm=%d%d\n",p.name,p.major,p.minor);parts();
    {Fixture f(2560,640);f.non_mode8_fallback();for(int n:{1,2,3,4,5,7,8})f.exact({n});f.exact({0,2,0,3,4,1});f.exact({});
     for(int m=1;m<=5;++m)f.exact({2,3,4,1},m);f.graph();
    }
    {Fixture f(256,32);f.exact({1,2,3,4});}
    std::printf("PASS native_expert_reuse_test enabled=%d public_dispatch=1 exact_hq=1 exact_output=1 graph_changed=1 guards=1 two_launch=1 optional_counters=1 sm70_predicate=1 mode7_fallback=1\n",int(reuse_enabled));return 0;}
