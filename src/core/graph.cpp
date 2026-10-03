// src/core/graph.cpp - P2.S5: the GraphRegistry implementation.
#include "strata/core/graph.hpp"

#include <immintrin.h>

#include <chrono>
#include <cstdio>
#include <string>
#if defined(__linux__)
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace strata::core {
namespace {

thread_local GraphCaptureHook capture_hook = nullptr;
thread_local void* capture_user = nullptr;

/// Fills `err` from the CUDA runtime, naming the call that failed.  A bare "invalid argument" with no call
/// site is the least useful error this API can produce and the easiest to avoid.
bool fail(std::string& err, const char* what, cudaError_t e) {
    err = std::string(what) + ": " + cudaGetErrorString(e);
    return false;
}

}  // namespace

GraphCaptureObservationScope::GraphCaptureObservationScope(GraphCaptureHook hook, void* user)
    : previous_hook_(capture_hook), previous_user_(capture_user) {
    capture_hook = hook;
    capture_user = user;
}

GraphCaptureObservationScope::~GraphCaptureObservationScope() {
    capture_hook = previous_hook_;
    capture_user = previous_user_;
}

bool GraphCaptureProbe::check(GraphCapturePhase phase, std::string& err) const {
    if (!capture_hook) return true;
    GraphCaptureSample sample;
    cudaError_t e = cudaGetDevice(&sample.device);
    if (e != cudaSuccess) return fail(err, "graph observation cudaGetDevice", e);
    size_t free_bytes = 0, total_bytes = 0;
    e = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (e != cudaSuccess) return fail(err, "graph observation cudaMemGetInfo", e);
    sample.free_device_bytes = free_bytes;
    sample.total_device_bytes = total_bytes;
#if defined(__linux__)
    struct rusage usage{};
    const long page = sysconf(_SC_PAGESIZE);
    FILE* f = std::fopen("/proc/self/statm", "r");
    unsigned long long pages = 0, resident = 0;
    const bool rss_ok = f && std::fscanf(f, "%llu %llu", &pages, &resident) == 2;
    if (f) std::fclose(f);
    if (rss_ok && page > 0 && getrusage(RUSAGE_SELF, &usage) == 0) {
        sample.host_rss_available = true;
        sample.host_rss_bytes = resident * (uint64_t) page;
        sample.host_peak_rss_bytes = (uint64_t) usage.ru_maxrss * 1024;
    }
#endif
    if (capture_hook(capture_user, key_, phase, sample, err)) return true;
    if (err.empty()) err = "graph qualification observer refused capture";
    return false;
}

bool graph_capture_instantiate(const GraphCaptureProbe& probe, cudaGraph_t graph,
                               cudaGraphExec_t& exec, cudaStream_t stream, bool upload,
                               std::string& err) {
    if (!probe.check(GraphCapturePhase::AfterCapture, err)) return false;
    cudaError_t e = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    if (e != cudaSuccess) {
        if (exec) { cudaGraphExecDestroy(exec); exec = nullptr; }
        return fail(err, "cudaGraphInstantiate", e);
    }
    bool ok = probe.check(GraphCapturePhase::AfterInstantiate, err);
    if (ok && (upload || capture_hook)) {
        e = cudaGraphUpload(exec, stream);
        // Even on upload failure, drain before destruction; no kernels are launched here.
        const cudaError_t sync = cudaStreamSynchronize(stream);
        if (e != cudaSuccess) ok = fail(err, "cudaGraphUpload", e);
        else if (sync != cudaSuccess) ok = fail(err, "graph upload cudaStreamSynchronize", sync);
        else ok = probe.check(GraphCapturePhase::AfterUpload, err);
    }
    if (!ok) { cudaGraphExecDestroy(exec); exec = nullptr; }
    return ok;
}

CapturedGraph& CapturedGraph::operator=(CapturedGraph&& o) noexcept {
    if (this != &o) {
        reset();
        graph_ = o.graph_;
        exec_ = o.exec_;
        done_ = o.done_;
        nodes_ = o.nodes_;
        key_ = o.key_;
        o.graph_ = nullptr;
        o.exec_ = nullptr;
        o.done_ = nullptr;
        o.nodes_ = 0;
    }
    return *this;
}

void CapturedGraph::reset() {
    // ORDER MATTERS: destroying a graph exec BLOCKS until the launch it is destroying has completed, so
    // destroying it before the event is harmless but destroying it after a spin that was supposed to observe
    // completion would silently become the thing that caused it.  `bench/micro/graph_capture.cu` was fooled by
    // exactly this once - it looked like captured graphs made a doorbell visible and direct launches did not.
    if (exec_) { cudaGraphExecDestroy(exec_); exec_ = nullptr; }
    if (graph_) { cudaGraphDestroy(graph_); graph_ = nullptr; }
    if (done_) { cudaEventDestroy(done_); done_ = nullptr; }
    nodes_ = 0;
}

bool CapturedGraph::begin(void* stream, std::string& err, GraphCaptureKey key) {
    if (graph_ || exec_) { err = "begin: this CapturedGraph is already recorded"; return false; }
    key_ = key;
    if (!GraphCaptureProbe(key_).check(GraphCapturePhase::BeforeCapture, err)) return false;
    const cudaError_t e = cudaStreamBeginCapture((cudaStream_t) stream, cudaStreamCaptureModeThreadLocal);
    if (e != cudaSuccess) return fail(err, "cudaStreamBeginCapture", e);
    return true;
}

bool CapturedGraph::end(void* stream, std::string& err) {
    cudaError_t e = cudaStreamEndCapture((cudaStream_t) stream, &graph_);
    if (e != cudaSuccess) { reset(); return fail(err, "cudaStreamEndCapture", e); }

    nodes_ = 0;
    e = cudaGraphGetNodes(graph_, nullptr, &nodes_);
    if (e != cudaSuccess) { reset(); return fail(err, "cudaGraphGetNodes", e); }
    // A capture that recorded NOTHING is a wiring mistake, and a graph that replays nothing produces no error
    // and no output - the silent kind of failure this project keeps paying for.
    if (nodes_ == 0) { err = "end: the capture recorded ZERO nodes - the body launched nothing"; reset(); return false; }

    const GraphCaptureProbe probe(key_);
    if (!graph_capture_instantiate(probe, graph_, exec_, (cudaStream_t) stream, false, err)) {
        reset();
        return false;
    }

    // The completion event is recorded ONCE and reused: `launch` records it again after each replay, which is
    // what makes `wait_ms` a query rather than a sync.
    e = cudaEventCreateWithFlags(&done_, cudaEventDisableTiming);
    if (e != cudaSuccess) { reset(); return fail(err, "cudaEventCreateWithFlags", e); }
    if (!probe.check(GraphCapturePhase::Retained, err)) { reset(); return false; }
    return true;
}

bool CapturedGraph::launch(void* stream, std::string& err) const {
    if (!exec_) { err = "launch: not recorded"; return false; }
    cudaError_t e = cudaGraphLaunch(exec_, (cudaStream_t) stream);
    if (e != cudaSuccess) return fail(err, "cudaGraphLaunch", e);
    e = cudaEventRecord(done_, (cudaStream_t) stream);
    if (e != cudaSuccess) return fail(err, "cudaEventRecord", e);
    return true;
}

bool CapturedGraph::wait_ms(int timeout_ms) const {
    if (!done_) return false;
    // A BOUNDED wait, and the bound is the point: an unbounded spin turns a protocol bug into a hung run, and
    // a hung run says nothing about which side is stuck.  `cudaEventQuery` is a QUERY - it does not block and
    // it does not synchronise - so this satisfies P2.X3 while still giving the driver the call it needs to
    // flush the submission.  See NOTE 1 in the header: without a driver call here the work never starts.
    //
    // The deadline is a REAL CLOCK, not a count of pause instructions: `_mm_pause` is a few cycles, so
    // counting pauses as microseconds would make the timeout tens of times longer than the caller asked for -
    // a timeout that does not time out is worse than none, because it reports a hang as a pass.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    for (;;) {
        const cudaError_t q = cudaEventQuery(done_);
        if (q == cudaSuccess) return true;
        if (q != cudaErrorNotReady) return false;   // a real error, not "not finished"
        if (std::chrono::steady_clock::now() >= deadline) return false;
        for (int i = 0; i < 64; ++i) _mm_pause();
    }
}

bool GraphRegistry::record(LayerType type, int n_tokens, const std::function<void()>& body, std::string& err) {
    const Key k{(int) type, n_tokens};
    if (graphs_.count(k)) return true;              // already recorded: the point of a registry

    CapturedGraph g;
    if (!g.begin(stream_, err, {"registry", to_string(type), 0, n_tokens, 0})) return false;
    body();                                        // the caller launches into the captured stream
    // The body must not have failed silently.  An error state left on the stream would make EndCapture
    // succeed with a broken graph, so it is checked and cleared first.
    const cudaError_t body_err = cudaGetLastError();
    if (body_err != cudaSuccess) {
        // Abandon the capture without instantiating anything.
        cudaGraph_t junk = nullptr;
        cudaStreamEndCapture((cudaStream_t) stream_, &junk);
        if (junk) cudaGraphDestroy(junk);
        return fail(err, "the capture body left a CUDA error", body_err);
    }
    if (!g.end(stream_, err)) return false;

    graphs_.emplace(k, std::move(g));
    ++captures_;
    return true;
}

const CapturedGraph* GraphRegistry::find(LayerType type, int n_tokens) const {
    const auto it = graphs_.find(Key{(int) type, n_tokens});
    return it == graphs_.end() ? nullptr : &it->second;
}

bool GraphRegistry::launch(LayerType type, int n_tokens, int timeout_ms, std::string& err) const {
    const CapturedGraph* g = find(type, n_tokens);
    if (!g) {
        char buf[128];
        std::snprintf(buf, sizeof buf, "launch: no graph recorded for %s n=%d", to_string(type), n_tokens);
        err = buf;
        return false;
    }
    if (!g->launch(stream_, err)) return false;
    if (!g->wait_ms(timeout_ms)) { err = "launch: timed out waiting for the graph to complete"; return false; }
    return true;
}

}  // namespace strata::core
