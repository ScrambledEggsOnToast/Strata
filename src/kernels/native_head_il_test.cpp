// Explicit queued SM70 fixture: synthetic GGUF only, actual public NativeHead API.
#include "gguf_fixture.hpp"
#include "strata/core/native_head.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
namespace k = strata::kernels;
using strata::core::NativeHead;
void need(bool yes, const char* why) { if (!yes) throw std::runtime_error(why); }
void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
struct DeviceBytes {
    unsigned char* base = nullptr; size_t bytes;
    explicit DeviceBytes(size_t n) : bytes(n) { ck(cudaMalloc(&base, n + 512)); ck(cudaMemset(base, 0xa5, n + 512)); }
    ~DeviceBytes() { if (base) cudaFree(base); }
    void* data() const { return base + 256; }
    template<class T> T* as() const { return static_cast<T*>(data()); }
    void guards() const {
        unsigned char a[256], b[256]; ck(cudaMemcpy(a, base, 256, cudaMemcpyDeviceToHost));
        ck(cudaMemcpy(b, base + 256 + bytes, 256, cudaMemcpyDeviceToHost));
        for (int i = 0; i < 256; ++i) need(a[i] == 0xa5 && b[i] == 0xa5, "device buffer guard");
    }
};
struct Temp {
    std::filesystem::path path;
    Temp() : path(std::filesystem::temp_directory_path() / ("strata-head-il-" +
           std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()))) { std::filesystem::create_directory(path); }
    ~Temp() { std::error_code error; std::filesystem::remove_all(path, error); }
};
struct Format { int type, bytes, scale0, scale1; };
const Format formats[] = {{23,136,0,-1},{12,144,0,2},{13,176,0,2},{14,210,208,-1},{8,34,0,-1}};

std::filesystem::path write_head(const Temp& temp, Format format, int ni, int no) {
    const auto& g = strata::core::kCompiledGeometry;
    std::vector<fixture::Kv> meta{fixture::str("general.architecture", "qwen4exp")};
    for (const auto& pair : std::vector<std::pair<std::string,int64_t>>{
            {"block_count",g.n_layers},{"embedding_length",g.n_embd},
            {"expert_count",g.n_expert},{"expert_used_count",g.n_expert_used},
            {"attention.head_count",g.n_head},{"attention.head_count_kv",g.n_head_kv}})
        meta.push_back(fixture::u32("qwen4exp." + pair.first, pair.second));
    const auto path = temp.path / "head.gguf";
    const auto written = fixture::write(path, meta, {{"output.weight", {uint64_t(ni),uint64_t(no)}, uint32_t(format.type), 13}});
    // Rewrite scales in bounded host chunks. All quantized bytes remain synthetic;
    // normal half scales ensure finite outputs and a non-vacuous exact comparison.
    std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
    const int block_values = format.type == 8 ? 32 : 256;
    const size_t blocks = size_t(ni / block_values) * no;
    std::vector<unsigned char> buffer(size_t(format.bytes) * 1024);
    for (size_t first = 0; first < blocks; first += 1024) {
        const size_t count = std::min(size_t(1024), blocks - first), bytes = count * format.bytes;
        file.seekg(std::streamoff(written.data_start + first * format.bytes)); file.read(reinterpret_cast<char*>(buffer.data()), bytes);
        need(bool(file), "read generated GGUF payload");
        for (size_t b = 0; b < count; ++b) for (int at : {format.scale0, format.scale1}) if (at >= 0) {
            const uint16_t scale = uint16_t(0x2400U + ((first + b) & 0x1ffU));
            std::memcpy(buffer.data() + b * format.bytes + at, &scale, 2);
        }
        file.seekp(std::streamoff(written.data_start + first * format.bytes)); file.write(reinterpret_cast<char*>(buffer.data()), bytes);
        need(bool(file), "write generated GGUF payload");
    }
    return path;
}

void run(Format format, int ni, int no, bool enabled) {
    Temp temp; const auto path = write_head(temp, format, ni, no);
    NativeHead head; std::string error;
    need(head.load({path.string()}, ni, no, error, true), error.c_str());
    need(head.projection_scratch_bytes() == k::native_q8_1_bytes(ni, 1), "single-row public allocation unchanged");
    need(head.projection_scratch_bytes(7) >= head.projection_scratch_bytes(3) &&
         head.projection_scratch_bytes(7) >= head.projection_scratch_bytes(4) &&
         head.projection_scratch_bytes(8) >= head.projection_scratch_bytes(4), "capacity query covers smaller IL windows");
    cudaStream_t stream; ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    for (int rows : no == 248320 ? std::vector<int>{1,2,3} : std::vector<int>{1,2,3,4}) {
        const size_t qbytes = k::native_q8_1_bytes(ni, rows), required = head.projection_scratch_bytes(rows);
        need(required == NativeHead::planned_projection_scratch_bytes(ni, rows), "public plan equals caller allocation");
        const bool il = enabled && rows >= 2 && k::native_mmvq_il_supported(format.type, rows, no);
        need(il == (enabled && rows >= 2 && format.type != 8), "actual SM70 table route eligibility");
        need(required == (enabled && rows >= 2 ? (qbytes + 255) / 256 * 256 + k::native_q8_1_il_bytes(ni, rows) : qbytes), "exact scratch accounting");
        DeviceBytes x(size_t(ni)*rows*4), q(qbytes), scratch(required), ref(size_t(no)*rows*4), got(size_t(no)*rows*4);
        // Poison writes use the default stream, while test work uses a
        // nonblocking stream. Finish initialization before publishing buffers
        // to that stream. This fence is outside capture and all timing loops.
        ck(cudaDeviceSynchronize());
        std::vector<float> input(size_t(ni)*rows), a(size_t(no)*rows), b(a.size());
        for (size_t i=0;i<input.size();++i) input[i] = float(int((i*7919+123)%8191)-4095)*0.0001f;
        ck(cudaMemcpyAsync(x.data(),input.data(),input.size()*4,cudaMemcpyHostToDevice,stream));
        auto reference = [&] {
            k::native_quantize_q8_1(x.as<float>(),q.data(),ni,rows,stream);
            k::native_mmvq(format.type,head.weights(),q.data(),ref.as<float>(),ni,no,rows,stream);
        };
        auto candidate = [&] { need(head.run_rows_with_scratch(x.as<float>(),got.as<float>(),rows,scratch.data(),required,stream,error),error.c_str()); };
        need(!head.run_rows_with_scratch(x.as<float>(),got.as<float>(),rows,scratch.data(),required-1,stream,error), "undersized scratch refused before launch");
        const char* phase = "initial-public";
        auto exact_logits = [&](const char* label) {
            for (size_t i=0;i<a.size();++i) {
                uint32_t av=0,bv=0;std::memcpy(&av,&a[i],4);std::memcpy(&bv,&b[i],4);
                if(av!=bv){std::fprintf(stderr,"MISMATCH phase=%s check=%s type=%d ni=%d vocabulary=%d rows=%d il=%d index=%zu row=%zu token=%zu reference_bits=%08x candidate_bits=%08x reference=%.9g candidate=%.9g\n",phase,label,format.type,ni,no,rows,int(il),i,i/size_t(no),i%size_t(no),unsigned(av),unsigned(bv),double(a[i]),double(b[i]));
                    throw std::runtime_error(label);}
            }
        };
        auto compare = [&] {
            ck(cudaStreamSynchronize(stream)); head.report_rows_completed(rows);
            ck(cudaMemcpy(a.data(),ref.data(),a.size()*4,cudaMemcpyDeviceToHost)); ck(cudaMemcpy(b.data(),got.data(),b.size()*4,cudaMemcpyDeviceToHost));
            std::vector<unsigned char> qa(qbytes),qb(qbytes);ck(cudaMemcpy(qa.data(),q.data(),qbytes,cudaMemcpyDeviceToHost));
            ck(cudaMemcpy(qb.data(),scratch.data(),qbytes,cudaMemcpyDeviceToHost));
            if(qa!=qb){for(size_t i=0;i<qbytes;++i)if(qa[i]!=qb[i]){std::fprintf(stderr,"MISMATCH phase=%s check=q8 type=%d ni=%d rows=%d il=%d byte=%zu reference=%02x candidate=%02x\n",phase,format.type,ni,rows,int(il),i,unsigned(qa[i]),unsigned(qb[i]));break;}}
            need(qa==qb,"unchanged q8_1 activation bytes");
            exact_logits("actual public head outputs bit-exact over full vocabulary");
            for (float value:b) need(std::isfinite(value),"finite output exactness witness");
            for (const auto* buffer:{&x,&q,&scratch,&ref,&got}) buffer->guards();
        };
        // Diagnostic only: replay the retained MMVQ with exactly the same q8
        // buffer, before exercising the public candidate. This distinguishes an
        // inherited small-K/odd-vocabulary issue from activation/IL dispatch.
        std::fprintf(stderr,"CASE type=%d ni=%d vocabulary=%d rows=%d il=%d\n",format.type,ni,no,rows,int(il));
        phase="baseline-repeat";reference();
        k::native_mmvq(format.type,head.weights(),q.data(),got.as<float>(),ni,no,rows,stream);
        ck(cudaStreamSynchronize(stream));
        ck(cudaMemcpy(a.data(),ref.data(),a.size()*4,cudaMemcpyDeviceToHost));
        ck(cudaMemcpy(b.data(),got.data(),b.size()*4,cudaMemcpyDeviceToHost));
        exact_logits("retained baseline repeat with shared q8 bytes");
        phase="initial-public";reference();candidate();compare(); const auto initial=b;
        cudaGraph_t graph;cudaGraphExec_t exec;
        ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));candidate();ck(cudaStreamEndCapture(stream,&graph));
        // Assert actual kernel dispatch in the captured public call, not just eligibility.
        size_t count=0;ck(cudaGraphGetNodes(graph,nullptr,&count));std::vector<cudaGraphNode_t> nodes(count);ck(cudaGraphGetNodes(graph,nodes.data(),&count));
        bool interleaved_kernel=false;
        for(auto node:nodes){cudaGraphNodeType type;ck(cudaGraphNodeGetType(node,&type));if(type!=cudaGraphNodeTypeKernel)continue;
            cudaKernelNodeParams params{};ck(cudaGraphKernelNodeGetParams(node,&params));const char* name=nullptr;
            ck(cudaFuncGetName(&name,params.func)); if(name && std::strstr(name,"native_quantize_q8_1_il_kernel"))interleaved_kernel=true;
        }
        need(interleaved_kernel==il,"actual captured interleaved dispatch");
        ck(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
        std::fill(input.end()-ni,input.end(),0.f);ck(cudaMemcpyAsync(x.data(),input.data(),input.size()*4,cudaMemcpyHostToDevice,stream));
        phase="graph-changed-input";reference();ck(cudaGraphLaunch(exec,stream));compare();bool changed=false;
        for(size_t i=size_t(rows-1)*no;i<b.size();++i){changed|=initial[i]!=b[i];need(b[i]==0.f,"graph consumes changed final input row");}
        need(changed,"finite changed-input output witness");ck(cudaGraphExecDestroy(exec));ck(cudaGraphDestroy(graph));
        // Useful head-shape screen only; both paths include their unchanged quantization.
        if(no==248320 && rows==3){for(size_t i=0;i<input.size();++i)input[i]=float(int((i*7919+123)%8191)-4095)*0.0001f;
            ck(cudaMemcpyAsync(x.data(),input.data(),input.size()*4,cudaMemcpyHostToDevice,stream));
            cudaEvent_t begin,end;ck(cudaEventCreate(&begin));ck(cudaEventCreate(&end));float times[3]{};
            for(int pass=0;pass<3;++pass){for(int warm=0;warm<2;++warm){if(pass==1)candidate();else reference();}
                ck(cudaEventRecord(begin,stream));for(int rep=0;rep<10;++rep){if(pass==1)candidate();else reference();}
                ck(cudaEventRecord(end,stream));ck(cudaEventSynchronize(end));ck(cudaEventElapsedTime(&times[pass],begin,end));times[pass]/=10;}
            ck(cudaEventDestroy(begin));ck(cudaEventDestroy(end));
            std::printf("TIMING type=%d ni=%d vocabulary=%d rows=%d baseline_ms=%.6f candidate_ms=%.6f baseline_after_ms=%.6f\n",format.type,ni,no,rows,times[0],times[1],times[2]);}
        std::printf("PASS head_il type=%d ni=%d vocabulary=%d rows=%d il=%d exact_q8=1 exact_logits=1 graph_changed=1 guards=1 scratch=%zu\n",format.type,ni,no,rows,int(il),required);
    }
    ck(cudaStreamDestroy(stream));
}
} // namespace
int main(int argc,char**argv){
    try {
        const bool off=argc==2 && std::strcmp(argv[1],"--off")==0;
        need(argc==1||off,"only optional --off supported");
        std::fprintf(stderr,"DIAGNOSTIC head_il=%d ordered_init=1 baseline_repeat=1\n",int(!off));
        need(setenv("STRATA_NATIVE_HEAD_IL",off?"0":"1",1)==0,"set candidate flag");
        need(setenv("STRATA_NATIVE_HEAD_IL_TRACE","1",1)==0,"set diagnostic flag");
        need(setenv("STRATA_SM70_TABLE","1",1)==0,"set SM70 table");
        need(!std::getenv("STRATA_MMVQ_IL_ROWS") && !std::getenv("STRATA_Q6_PACKED"),"independent overrides must be absent");
        k::native_mmvq_set_multi_exact(true);cudaDeviceProp properties{};ck(cudaGetDeviceProperties(&properties,0));
        need(properties.major==7&&properties.minor==0,"fixture requires queued SM70");
        for(auto format:formats)run(format,256,2051,!off);
        run(formats[3],2560,248320,!off);
        std::puts("PASS native_head_il_public_api synthetic_only=1 service_not_exercised=1");return 0;
    }catch(const std::exception& error){std::fprintf(stderr,"FAIL: %s\n",error.what());return 1;}
}
