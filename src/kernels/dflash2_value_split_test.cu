#include "cuda/dflash2_forward.cu"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
using namespace strata::kernels;
using namespace strata::kernels::dflash2;
namespace {
constexpr size_t guard=16, output_count=size_t(block)*query_width;
constexpr int max_chunks=attention_value_split_max_chunks;
constexpr uint32_t sentinel=0x20000000u;
void check(cudaError_t c) { if(c!=cudaSuccess) { std::fprintf(stderr,"CUDA %s\n",cudaGetErrorString(c));std::exit(1); } }
void require(bool b,const char* s) { if(!b) {std::fprintf(stderr,"FAIL %s\n",s);std::exit(1);} }
uint16_t finite_value(size_t i) { return bf16_from_f32(float(int((i*1664525u+(i>>4)*1013904223u)%257)-128)/128.f); }
struct Fixture {
 std::vector<uint16_t> ring,values,query,keys,a,b;
 std::vector<float> probabilities,partials;
 std::vector<uint8_t> flags;
 uint16_t *dring,*dv,*dq,*dk,*da,*db;
 float *dp,*ds,*partial;
 uint8_t* unsafe;
 uint32_t *sa,*sb,*fallback;
 cudaStream_t stream;
 Fixture():ring(kv_elements),values(size_t(block)*kv_width),query(output_count),keys(values.size()),a(output_count+2*guard),b(a.size()),probabilities(size_t(query_heads)*block*max_keys),partials(value_split_detail::partial_count(block,max_chunks)+2*guard),flags(value_split_detail::flag_count(block,max_chunks)+2*guard) {
  for(size_t i=0;i<ring.size();++i)ring[i]=finite_value(i+103);
  for(size_t i=0;i<query.size();++i)query[i]=finite_value(i+17);
  for(size_t i=0;i<keys.size();++i)keys[i]=finite_value(i+29);
  check(cudaMalloc(&dring,ring.size()*2));check(cudaMalloc(&dv,values.size()*2));check(cudaMalloc(&dq,query.size()*2));check(cudaMalloc(&dk,keys.size()*2));
  check(cudaMalloc(&da,a.size()*2));check(cudaMalloc(&db,b.size()*2));check(cudaMalloc(&dp,probabilities.size()*4));check(cudaMalloc(&ds,probabilities.size()*4));
  check(cudaMalloc(&partial,partials.size()*4));check(cudaMalloc(&unsafe,flags.size()));
  check(cudaMalloc(&sa,4));check(cudaMalloc(&sb,4));check(cudaMalloc(&fallback,4));check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
  check(cudaMemcpyAsync(dring,ring.data(),ring.size()*2,cudaMemcpyHostToDevice,stream));
  check(cudaMemcpyAsync(dq,query.data(),query.size()*2,cudaMemcpyHostToDevice,stream));
  check(cudaMemcpyAsync(dk,keys.data(),keys.size()*2,cudaMemcpyHostToDevice,stream));check(cudaStreamSynchronize(stream));
 }
 ~Fixture(){check(cudaStreamSynchronize(stream));check(cudaStreamDestroy(stream));for(void* p:{(void*)dring,(void*)dv,(void*)dq,(void*)dk,(void*)da,(void*)db,(void*)dp,(void*)ds,(void*)partial,(void*)unsafe,(void*)sa,(void*)sb,(void*)fallback})check(cudaFree(p));}
 void reset(){
  check(cudaMemsetAsync(da,0xa5,a.size()*2,stream));check(cudaMemsetAsync(db,0xa5,b.size()*2,stream));
  check(cudaMemsetAsync(partial,0xa5,partials.size()*4,stream));check(cudaMemsetAsync(unsafe,0xa5,flags.size(),stream));
  check(cudaMemcpyAsync(sa,&sentinel,4,cudaMemcpyHostToDevice,stream));check(cudaMemcpyAsync(sb,&sentinel,4,cudaMemcpyHostToDevice,stream));check(cudaMemsetAsync(fallback,0,4,stream));
 }
 double compare(int rows,int chunks,bool exceptional,bool expect_fallback,int exact_dimension=-1){
  check(cudaStreamSynchronize(stream));check(cudaMemcpy(a.data(),da,a.size()*2,cudaMemcpyDeviceToHost));check(cudaMemcpy(b.data(),db,b.size()*2,cudaMemcpyDeviceToHost));
  uint32_t x,y,f;check(cudaMemcpy(&x,sa,4,cudaMemcpyDeviceToHost));check(cudaMemcpy(&y,sb,4,cudaMemcpyDeviceToHost));check(cudaMemcpy(flags.data(),unsafe,flags.size(),cudaMemcpyDeviceToHost));f=0;
  for(size_t i=guard;i<guard+value_split_detail::flag_count(rows,chunks);++i)f+=flags[i]!=0;
  require(x==y&&x==(sentinel|uint32_t(exceptional)),"numeric status");require(expect_fallback?f>0:f==0,"fallback exposure");
  double error=0,scale=0;
  for(size_t i=0;i<a.size();++i){
   if(i<guard||i>=guard+size_t(rows)*query_width){require(a[i]==0xa5a5&&b[i]==0xa5a5,"output guard");continue;}
   float av=f32_from_bf16(a[i]),bv=f32_from_bf16(b[i]);
   if(!std::isfinite(av)||!std::isfinite(bv))require((std::isnan(av)&&std::isnan(bv))||a[i]==b[i],"exceptional output differs");
   else {double delta=double(av)-bv;error+=delta*delta;scale+=double(av)*av;
    if(exact_dimension>=0&&int((i-guard)%head_dim)==exact_dimension&&((i-guard)%query_width)/head_dim<4)require(a[i]==b[i],"retained fallback result differs");}
  }
  double rmse=std::sqrt(error/std::max(scale,1e-80));require(rmse<=.01,"relative RMSE > 1% gross screen");if(chunks==0)require(error==0,"short/off path must retain exact results");
  check(cudaMemcpy(partials.data(),partial,partials.size()*4,cudaMemcpyDeviceToHost));check(cudaMemcpy(flags.data(),unsafe,flags.size(),cudaMemcpyDeviceToHost));
  const size_t pn=value_split_detail::partial_count(rows,chunks),fn=value_split_detail::flag_count(rows,chunks);
  for(size_t i=0;i<partials.size();++i)if(i<guard||i>=guard+pn){uint32_t bits;std::memcpy(&bits,&partials[i],4);require(bits==0xa5a5a5a5u,"partial guard");}
  for(size_t i=0;i<flags.size();++i)if(i<guard||i>=guard+fn)require(flags[i]==0xa5,"flag guard");
  return rmse;
 }
 template<int Chunk> void run(int rows,int context,int begin,int layer,int mode=0,bool pipeline=false){
  int frontier=begin+context,nkeys=context+rows;
  const bool selected=attention_value_split_enabled()&&context>=attention_value_split_min_context;
  const int chunks=selected?(nkeys+Chunk-1)/Chunk:0;
  for(size_t i=0;i<values.size();++i)values[i]=finite_value(i+51);
  std::fill(probabilities.begin(),probabilities.end(),std::numeric_limits<float>::quiet_NaN());
  for(int h=0;h<query_heads;++h)for(int r=0;r<rows;++r){double sum=0;size_t base=(size_t(h)*rows+r)*nkeys;
   for(int k=0;k<nkeys;++k){float p=float(1+(k*13+h+r)%17);probabilities[base+k]=p;sum+=p;}
   for(int k=0;k<nkeys;++k)probabilities[base+k]/=float(sum);
  }
  size_t at=(((size_t(layer)*2+1)*kv_heads)*capacity+begin%capacity)*head_dim;
  uint16_t saved=ring[at];
  if(mode==1){require(context>0,"need context");ring[at]=0x7fc0;probabilities[0]=0;}
  if(mode==2)values[17]=0x7f80;
  if(mode==3)probabilities[0]=std::numeric_limits<float>::quiet_NaN();
  if(mode==4)values[0]=0x7b80; // 2^120: finite but conservatively falls back.
  if(mode==5){require(context>=256,"need overflow sequence");
   for(int k=0;k<nkeys;++k)probabilities[k]=1.f;
   for(int k=0;k<context;++k)ring[(((size_t(layer)*2+1)*kv_heads)*capacity+(begin+k)%capacity)*head_dim]=k<128?0x7f7f:0xff7f;
  }
  if(mode==6){ // The first context V is masked for row 7, but 0*NaN must propagate.
   require(pipeline&&rows==8&&context==capacity,"masked pipeline fixture geometry");ring[at]=0x7fc0;
  }
  if(mode==7){constexpr uint16_t tiny[]{0x0001,0x8001,0,0x8000};
   for(size_t i=0;i<ring.size();++i)ring[i]=tiny[i%4];for(size_t i=0;i<values.size();++i)values[i]=tiny[i%4];}
  check(cudaMemcpyAsync(dring,ring.data(),ring.size()*2,cudaMemcpyHostToDevice,stream));
  check(cudaMemcpyAsync(dv,values.data(),values.size()*2,cudaMemcpyHostToDevice,stream));check(cudaMemcpyAsync(dp,probabilities.data(),probabilities.size()*4,cudaMemcpyHostToDevice,stream));
  auto launch=[&](bool candidate,bool =true){
   uint32_t* status=candidate?sb:sa;
   if(pipeline){score_kernel<<<dim3((nkeys+7)/8,query_heads*rows),256,0,stream>>>(dq,dk,dring,ds,layer,begin,frontier,rows,status);
    softmax_kernel<<<query_heads*rows,256,0,stream>>>(ds,dp,nullptr,nkeys,status);check(cudaGetLastError());}
   if(candidate)check(static_cast<cudaError_t>(attention_values_planned(dp,dv,dring,db+guard,layer,begin,frontier,rows,sb,stream,
       {partial+guard,unsafe+guard,value_split_detail::partial_count(block,max_chunks),value_split_detail::flag_count(block,max_chunks)})));
   else {value_spans_kernel<<<query_heads*rows,head_dim,0,stream>>>(dp,dv,dring,da+guard,layer,begin,frontier,rows,sa);check(cudaGetLastError());}
  };
  bool exceptional=mode==1||mode==2||mode==3||mode==5||mode==6;
  reset();launch(false);launch(true);double rmse=compare(rows,chunks,exceptional,selected&&mode!=0&&mode!=7,mode==4||mode==5?0:-1);
  if(mode==5)require(a[guard]==0x7f80u&&b[guard]==0x7f80u,"retained overflow witness");
  if(mode==6){float masked=-1;check(cudaMemcpy(&masked,dp+size_t(7)*nkeys,4,cudaMemcpyDeviceToHost));require(masked==0.f&&std::isnan(f32_from_bf16(a[guard+size_t(7)*query_width])),"masked zero-times-NaN witness");}
  cudaGraph_t graph;cudaGraphExec_t executable;
  check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));launch(false);launch(true);check(cudaStreamEndCapture(stream,&graph));check(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
  uint16_t previous=a[guard],previous_candidate=b[guard];
  for(float changed:{64.f,-64.f}){
   if(mode!=4&&mode!=5)values[0]=bf16_from_f32(changed);
   check(cudaMemcpyAsync(dv,values.data(),2,cudaMemcpyHostToDevice,stream));reset();check(cudaGraphLaunch(executable,stream));compare(rows,chunks,exceptional,selected&&mode!=0&&mode!=7,mode==4||mode==5?0:-1);
   if(mode==0||mode==7){require(a[guard]!=previous&&b[guard]!=previous_candidate&&std::isfinite(f32_from_bf16(b[guard])),"finite changed-input graph witness");previous=a[guard];previous_candidate=b[guard];}
  }
  if(pipeline&&mode==0){
   cudaEvent_t start,end;check(cudaEventCreate(&start));check(cudaEventCreate(&end));for(bool value_only:{false,true}){pipeline=!value_only;float times[2]{};
   for(int arm:{0,1,1,0}){check(cudaEventRecord(start,stream));for(int i=0;i<5;++i)launch(arm,false);check(cudaEventRecord(end,stream));check(cudaEventSynchronize(end));float elapsed;check(cudaEventElapsedTime(&elapsed,start,end));times[arm]+=elapsed/10.f;}
   std::printf("{\"timing\":\"%s\",\"chunk\":%d,\"rows\":%d,\"context\":%d,\"reference_ms\":%.6f,\"candidate_ms\":%.6f}\n",value_only?"values":"scores_softmax_values",Chunk,rows,context,times[0],times[1]);}pipeline=true;check(cudaEventDestroy(start));check(cudaEventDestroy(end));
  }
  check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));
  if(mode==5||mode==7)for(size_t i=0;i<ring.size();++i)ring[i]=finite_value(i+103);else ring[at]=saved;
  check(cudaStreamSynchronize(stream));
  std::printf("{\"chunk\":%d,\"rows\":%d,\"context\":%d,\"begin\":%d,\"mode\":%d,\"pipeline\":%s,\"relative_rmse\":%.9g,\"status\":true,\"graph_changed_finite\":%s,\"guards\":true,\"workspace_bytes\":%zu}\n",Chunk,rows,context,begin,mode,pipeline?"true":"false",rmse,(mode==0||mode==7)?"true":"false",4*value_split_detail::partial_count(rows,chunks)+value_split_detail::flag_count(rows,chunks));
 }
 template<int Chunk> void suite(){
  for(int rows:{1,4,8})for(int context:{220,capacity})run<Chunk>(rows,context,capacity-3,4,0,true);
  run<Chunk>(4,1023,capacity-3,4);run<Chunk>(4,1024,capacity-3,4);
  run<Chunk>(1,0,0,0);run<Chunk>(4,1,capacity-1,0);run<Chunk>(4,capacity,0,0);
  for(int mode:{1,2,3,4,7})run<Chunk>(4,1024,capacity-3,4,mode);
  run<Chunk>(4,1024,capacity-3,4,5);run<Chunk>(8,capacity,capacity-3,4,6,true);
 }
};
}
int main(int argc,char**argv){
 if(argc==2&&std::strcmp(argv[1],"--help")==0){std::puts("Public planned split-K value fixture. Select GPU with CUDA_VISIBLE_DEVICES; optional --off starts retained mode.");return 0;}
 const bool off=argc==2&&std::strcmp(argv[1],"--off")==0;require(argc==1||off,"only --off or --help supported");require(setenv("STRATA_DFLASH2_VALUE_SPLIT",off?"0":"1",1)==0,"set variant before frozen planning");int device=0;cudaDeviceProp prop{};check(cudaGetDevice(&device));check(cudaGetDeviceProperties(&prop,device));require(prop.major==7&&prop.minor==0,"fixture requires queued SM70 V100");std::printf("{\"device\":\"%s\",\"compute_major\":%d,\"compute_minor\":%d}\n",prop.name,prop.major,prop.minor);{Fixture f;f.suite<128>();
 require(attention_values_planned(f.dp,f.dv,f.dring,f.db,0,0,0,0,f.sb,f.stream,{})==int(cudaErrorInvalidValue),"row guard");
 require(attention_values_planned(f.dp,f.dv,f.dring,f.db,0,0,capacity+1,4,f.sb,f.stream,{})==int(cudaErrorInvalidValue),"context guard");
 if(!off){
  AttentionValueWorkspace w{f.partial,f.unsafe,attention_value_split_partial_count,attention_value_split_flag_count};
  --w.partial_count;require(attention_values_planned(f.dp,f.dv,f.dring,f.db,0,0,capacity,block,f.sb,f.stream,w)==int(cudaErrorInvalidValue),"undersized partial refuses");
  ++w.partial_count;--w.flag_count;require(attention_values_planned(f.dp,f.dv,f.dring,f.db,0,0,capacity,block,f.sb,f.stream,w)==int(cudaErrorInvalidValue),"undersized flags refuse");
 }
 }
 std::puts("{\"result\":\"PASS\",\"chunks\":[128],\"relative_rmse_limit\":0.01,\"quality_gate\":false}");
 return 0;
}
