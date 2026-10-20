// Device feasibility gate, deliberately separate from full-model/client acceptance.
#include "strata/core/kv_shared_pool.hpp"
#include "strata/core/conversation_snapshot.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace {
using namespace strata;
void require(bool good, const char* why) { if (!good) throw std::runtime_error(why); }
void check(cudaError_t result, const char* why) {
    if (result != cudaSuccess) throw std::runtime_error(std::string(why) + ": " + cudaGetErrorString(result));
}
template<class T> struct Device {
    T* p = nullptr;
    explicit Device(uint64_t count) { check(cudaMalloc(&p, (size_t) count * sizeof(T)), "allocate fixture"); }
    ~Device() { if (p && cudaFree(p) != cudaSuccess) std::terminate(); }
    Device(const Device&) = delete;
};
program::ContinuationCertificate certificate(core::KvOwnerKey key, int64_t cells) {
    program::ContinuationInput in;
    in.request = key.request; in.incarnation = key.incarnation; in.committed_tokens = cells;
    in.needs_qsa = in.needs_gdn = in.needs_ple = in.needs_stages = true;
    return program::certify_continuation(in);
}
std::array<uint8_t*, 3> arrays(const kernels::QsaAttnPools& p) {
    return {reinterpret_cast<uint8_t*>(const_cast<int8_t*>(p.k_q)),
        reinterpret_cast<uint8_t*>(const_cast<uint16_t*>(p.k_scale)), const_cast<uint8_t*>(p.v_q4)};
}
constexpr uint64_t widths[] = {2048, 64, 1152};
void initialize(const kernels::QsaAttnPools& p, uint64_t first, uint64_t pages) {
    auto a = arrays(p);
    for (size_t part = 0; part < 3; ++part) {
        std::vector<uint8_t> data((size_t) pages * widths[part]);
        for (size_t i = 0; i < data.size(); ++i) data[i] = uint8_t((i * 17 + i / 127) % 251);
        if (part == 1) for (size_t i = 0; i < data.size(); i += 2) { data[i] = 0; data[i + 1] = 0x20; }
        if (part == 2) for (size_t i = 0; i < data.size(); i += 18) { data[i] = 0; data[i + 1] = 0x28; }
        check(cudaMemcpy(a[part] + first * widths[part], data.data(), data.size(), cudaMemcpyHostToDevice), "initialize K8V4");
    }
}
std::vector<uint8_t> page(const kernels::QsaAttnPools& p, uint64_t physical) {
    std::vector<uint8_t> out(3264);
    auto a = arrays(p);
    size_t at = 0;
    for (size_t part = 0; part < 3; ++part) {
        check(cudaMemcpy(out.data() + at, a[part] + physical * widths[part], (size_t) widths[part],
                         cudaMemcpyDeviceToHost), "page readback");
        at += (size_t) widths[part];
    }
    return out;
}
struct PairStats {
    uint64_t layers = 0, release_chunks = 0, cow_chunks = 0, host_bytes = 0, live_bytes = 0;
    uint64_t terminal_chunks = 0, reused_images = 0;
    double handoff_ms = 0;
    double kernel_private_ms = 0, kernel_candidate_ms = 0;
};
uint64_t output_hash(const std::vector<float>& values) {
    uint64_t hash = 14695981039346656037ull;
    for (float value : values) {
        uint32_t bits = 0; std::memcpy(&bits, &value, sizeof(bits));
        for (int shift = 0; shift < 32; shift += 8) { hash ^= (bits >> shift) & 255; hash *= 1099511628211ull; }
    }
    return hash;
}
void screen(int device, int layer, bool chain, bool aliases, PairStats& stats) {
    check(cudaSetDevice(device), "select stage device");
    const auto s = kernels::qsa_real_shapes();
    plan::KvShareBudget budget{aliases ? plan::KvShareMode::alias : plan::KvShareMode::pool, 262144, 4, 12, 3,
        plan::kKvShareChunk, 12ull << 30, 3299132688};
    core::KvSharedPool pool(budget);
    std::string error;
    require(pool.init(error), error.c_str());
    const auto quote = plan::kv_share_quote(s, kernels::kKvHybrid, budget, 65539);
    require(pool.host_bytes() * budget.layers <= quote.host_metadata + quote.cow_host,
        "measured adapter metadata exceeds admitted host envelope");
    const core::KvOwnerKey a{101, 1, uint64_t(11 + 2 * layer)}, b{102, 1, uint64_t(12 + 2 * layer)};
    Device<int32_t> ta(65536), tb(65536);
    require(pool.begin(0, a, ta.p, error) && pool.begin(1, b, tb.p, error) && pool.ready(0, a), error.c_str());
    const auto pa = pool.pools(0), pb = pool.pools(1);
    initialize(pa, uint64_t(pool.table(0)[0]), 16385);
    const auto aa = arrays(pa), ab = arrays(pb);
    auto private_copy = [&] {
        for (size_t part = 0; part < 3; ++part)
            check(cudaMemcpy(ab[part] + uint64_t(pool.table(1)[0]) * widths[part],
                aa[part] + uint64_t(pool.table(0)[0]) * widths[part], 16385 * widths[part],
                cudaMemcpyDeviceToDevice), "private prefix control");
        check(cudaDeviceSynchronize(), "private control drain");
    };
    private_copy();
    constexpr int cap = 2048;
    Device<float> q(uint64_t(s.n_head) * s.head_dim), out(uint64_t(s.n_head) * s.head_dim);
    Device<float> scratch(kernels::qsa_decode_attn_scratch_floats(cap, s));
    Device<float> key(uint64_t(s.n_head_kv) * s.head_dim), value(uint64_t(s.n_head_kv) * s.head_dim);
    Device<int32_t> ids(cap), step(kernels::kStepCount);
    std::vector<float> query(size_t(s.n_head) * s.head_dim, 0.03125f), control(query.size()), shared(query.size());
    std::vector<float> current(size_t(s.n_head_kv) * s.head_dim, 0.25f);
    std::vector<int32_t> selected(cap), readback(65536), source_table(65536);
    for (int i = 0; i < cap; ++i) selected[size_t(i)] = i == cap - 1 ? 65538 : i * 31;
    int32_t steps[kernels::kStepCount];
    kernels::qsa_step_fill(steps, 65538, s); steps[kernels::kStepWidth] = cap;
    check(cudaMemcpy(q.p, query.data(), query.size() * sizeof(float), cudaMemcpyHostToDevice), "query upload");
    check(cudaMemcpy(ids.p, selected.data(), selected.size() * sizeof(int32_t), cudaMemcpyHostToDevice), "selection upload");
    check(cudaMemcpy(step.p, steps, sizeof(steps), cudaMemcpyHostToDevice), "step upload");
    auto record_append = [&](const kernels::QsaAttnPools& p, void* on) {
        kernels::kv_append_q8_step(const_cast<int8_t*>(p.k_q), const_cast<int8_t*>(p.k_q),
            const_cast<uint16_t*>(p.k_scale), const_cast<uint16_t*>(p.k_scale), p.page_table, step.p,
            key.p, key.p, s, on);
        kernels::kv_append_q4_step(const_cast<uint8_t*>(p.v_q4), const_cast<uint8_t*>(p.v_q4), p.page_table,
            step.p, value.p, value.p, s, on);
    };
    // Warm the append modules at the first uncommitted private cell, before
    // capture. The 65,539-cell committed-prefix attention never selects it.
    check(cudaMemcpy(key.p, current.data(), current.size() * sizeof(float), cudaMemcpyHostToDevice), "warm K");
    std::fill(current.begin(), current.end(), 0.125f);
    check(cudaMemcpy(value.p, current.data(), current.size() * sizeof(float), cudaMemcpyHostToDevice), "warm V");
    kernels::fwht256_inplace_cuda(value.p, s.n_head_kv, nullptr);
    int32_t warm_step[kernels::kStepCount];
    kernels::qsa_step_fill(warm_step, 65539, s);
    check(cudaMemcpy(step.p, warm_step, sizeof(warm_step), cudaMemcpyHostToDevice), "warm append position");
    require(pool.writable(1, b, 16384, error), error.c_str());
    record_append(pb, nullptr);
    check(cudaDeviceSynchronize(), "append module warm drain");
    check(cudaMemcpy(step.p, steps, sizeof(steps), cudaMemcpyHostToDevice), "restore attention position");
    kernels::qsa_decode_attn_step(q.p, pb, ids.p, step.p, cap, s, scratch.p, out.p, nullptr);
    check(cudaDeviceSynchronize(), "attention warm");
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr, peer_graph = nullptr;
    cudaGraphExec_t exec = nullptr, peer_exec = nullptr;
    cudaStream_t producer_stream = nullptr;
    check(cudaStreamCreateWithFlags(&producer_stream, cudaStreamNonBlocking), "source producer stream");
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "capture stream");
    auto capture = [&](const kernels::QsaAttnPools& p, cudaGraph_t& g, cudaGraphExec_t& e) {
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin preserved capture");
        kernels::qsa_decode_attn_step(q.p, p, ids.p, step.p, cap, s, scratch.p, out.p, stream);
        check(cudaStreamEndCapture(stream, &g), "end capture");
        check(cudaGraphInstantiate(&e, g, 0), "instantiate once");
    };
    capture(pb, graph, exec); capture(pa, peer_graph, peer_exec);
    std::array<cudaGraph_t, 2> append_graphs{};
    std::array<cudaGraphExec_t, 2> append_execs{};
    for (size_t owner = 0; owner < append_graphs.size(); ++owner) {
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin append capture");
        record_append(owner ? pb : pa, stream);
        check(cudaStreamEndCapture(stream, &append_graphs[owner]), "end append capture");
        check(cudaGraphInstantiate(&append_execs[owner], append_graphs[owner], 0), "instantiate append once");
    }
    cudaEvent_t kernel_start = nullptr, kernel_stop = nullptr;
    check(cudaEventCreate(&kernel_start), "kernel start event"); check(cudaEventCreate(&kernel_stop), "kernel stop event");
    auto kernel_time = [&] {
        check(cudaEventRecord(kernel_start, stream), "kernel start");
        for (int repeat = 0; repeat < 100; ++repeat) check(cudaGraphLaunch(exec, stream), "kernel timing replay");
        check(cudaEventRecord(kernel_stop, stream), "kernel stop"); check(cudaEventSynchronize(kernel_stop), "kernel timing drain");
        float ms = 0; check(cudaEventElapsedTime(&ms, kernel_start, kernel_stop), "kernel elapsed");
        return double(ms) / 100;
    };
    const double private_kernel_ms = kernel_time();
    check(cudaGraphLaunch(exec, stream), "private graph replay");
    check(cudaStreamSynchronize(stream), "private replay drain");
    check(cudaMemcpy(control.data(), out.p, control.size() * sizeof(float), cudaMemcpyDeviceToHost), "control output");
    require(std::all_of(control.begin(), control.end(), [](float x) { return std::isfinite(x); }), "nonfinite control output");
    const auto boundary = pool.table(1)[16384];
    const uint64_t before_bytes = pool.mapped_bytes(), before_chunks = pool.released_chunks();
    size_t free_before = 0, total = 0, free_after = 0;
    check(cudaMemGetInfo(&free_before, &total), "before handoff memory");
    if (aliases) {
        require(!pool.alias(1, b, 0, a, 32768, certificate(a, 32768), error), "32K must refuse");
        require(pool.mapped_bytes() == before_bytes && pool.released_chunks() == before_chunks,
            "32K refusal mutated physical storage");
    }
    auto start = std::chrono::steady_clock::now();
    core::KvAliasReceipt trace;
    double boundary_us = 0;
    if (aliases) {
        const auto boundary_start = std::chrono::steady_clock::now();
        require(pool.writable(1, b, 16384, error), error.c_str());
        for (size_t part = 0; part < 3; ++part)
            check(cudaMemcpy(ab[part] + uint64_t(boundary) * widths[part],
                aa[part] + uint64_t(pool.table(0)[16384]) * widths[part],
                widths[part], cudaMemcpyDeviceToDevice), "unaligned private boundary");
        boundary_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - boundary_start).count();
        require(pool.alias(1, b, 0, a, 65539, certificate(a, 65539), error, &trace,
            layer % 2 ? producer_stream : stream, stream), error.c_str());
    }
    else private_copy();
    const double handoff_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    check(cudaMemGetInfo(&free_after, &total), "after handoff memory");
    check(cudaMemcpy(readback.data(), tb.p, readback.size() * sizeof(int32_t), cudaMemcpyDeviceToHost), "target identities");
    check(cudaMemcpy(source_table.data(), ta.p, source_table.size() * sizeof(int32_t), cudaMemcpyDeviceToHost), "source identities");
    require(readback == pool.table(1) && source_table == pool.table(0) &&
        readback[16384] == boundary && boundary != source_table[16384], "physical identities or unaligned boundary violated");
    if (aliases) require(std::equal(readback.begin(), readback.begin() + 16384, source_table.begin()), "prefix not physically aliased");
    const uint64_t released = pool.released_chunks() - before_chunks;
    require(released == (aliases ? 25u : 0u) && before_bytes - pool.mapped_bytes() == released * plan::kKvShareChunk &&
        int64_t(free_after) - int64_t(free_before) == int64_t(released * plan::kKvShareChunk),
        "handoff release not independently measured in successful 2MiB chunks");
    const double candidate_kernel_ms = kernel_time();
    check(cudaGraphLaunch(exec, stream), "handoff graph replay");
    check(cudaStreamSynchronize(stream), "handoff graph drain");
    check(cudaMemcpy(shared.data(), out.p, shared.size() * sizeof(float), cudaMemcpyDeviceToHost), "handoff output");
    require(std::memcmp(control.data(), shared.data(), shared.size() * sizeof(float)) == 0, "captured attention differs from private copy");
    auto select = [&](int64_t position) {
        selected[0] = int32_t(position);
        kernels::qsa_step_fill(steps, position, s); steps[kernels::kStepWidth] = 1;
        check(cudaMemcpy(ids.p, selected.data(), selected.size() * sizeof(int32_t), cudaMemcpyHostToDevice), "tail selection");
        check(cudaMemcpy(step.p, steps, sizeof(steps), cudaMemcpyHostToDevice), "tail counts");
    };
    auto commit_output = [&](cudaGraphExec_t replay, int64_t position, float expected, const char* phase) {
        check(cudaGraphLaunch(replay, stream), "committed attention replay");
        kernels::fwht256_inplace_cuda(out.p, s.n_head, stream);
        check(cudaStreamSynchronize(stream), "committed output drain");
        check(cudaMemcpy(shared.data(), out.p, shared.size() * sizeof(float), cudaMemcpyDeviceToHost), "committed output");
        require(std::all_of(shared.begin(), shared.end(), [=](float x) { return std::isfinite(x) && x == expected; }),
            "committed QSA output differs from independent one-cell constant-value oracle");
        std::printf("KV_SHARE_COMMITTED arm=%s layer=%d phase=%s position=%lld float_count=%zu value=%.6f fnv64=%llu scope=qsa_activation_not_model_tokens\n",
            aliases ? "alias" : "private", layer, phase, (long long) position, shared.size(), expected,
            (unsigned long long) output_hash(shared));
    };
    auto append = [&](uint64_t owner, const core::KvOwnerKey& owner_key,
                      cudaGraphExec_t replay, int64_t position, float expected, const char* phase) {
        select(position);
        std::fill(current.begin(), current.end(), 0.25f);
        check(cudaMemcpy(key.p, current.data(), current.size() * sizeof(float), cudaMemcpyHostToDevice), "tail K");
        std::fill(current.begin(), current.end(), expected);
        check(cudaMemcpy(value.p, current.data(), current.size() * sizeof(float), cudaMemcpyHostToDevice), "tail V");
        kernels::fwht256_inplace_cuda(value.p, s.n_head_kv, stream);
        const auto permission_start = std::chrono::steady_clock::now();
        require(pool.writable(owner, owner_key, uint64_t(position) / 4, error), error.c_str());
        const double permission_us = std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - permission_start).count();
        check(cudaEventRecord(kernel_start, stream), "append timing start");
        check(cudaGraphLaunch(append_execs[(size_t) owner], stream), "captured writable append");
        check(cudaEventRecord(kernel_stop, stream), "append timing stop");
        commit_output(replay, position, expected, phase);
        float append_ms = 0;
        check(cudaEventElapsedTime(&append_ms, kernel_start, kernel_stop), "append kernel time");
        std::printf("KV_SHARE_APPEND arm=%s layer=%d phase=%s captured=1 writable_before_graph=1 permission_us=%.3f kernel_ms=%.6f\n",
            aliases ? "alias" : "private", layer, phase, permission_us, double(append_ms));
    };
    append(0, a, peer_exec, 65539, 0.5f, "source_tail");
    append(1, b, exec, 65539, -0.25f, "divergent_tail");
    const auto source_page = page(pa, uint64_t(pool.table(0)[0]));
    const auto old_binding = pool.table(1)[0];
    const uint64_t cow_before = pool.mapped_bytes();
    size_t cow_free_before = 0, cow_free_after = 0;
    check(cudaMemGetInfo(&cow_free_before, &total), "before COW memory");
    const auto cow_start = std::chrono::steady_clock::now();
    require(pool.writable(1, b, 0, error), error.c_str());
    const double cow_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - cow_start).count();
    check(cudaMemGetInfo(&cow_free_after, &total), "after COW memory");
    const uint64_t cow_bytes = pool.mapped_bytes() - cow_before;
    require(cow_bytes == (aliases ? 3 * plan::kKvShareChunk : 0) &&
        int64_t(cow_free_before) - int64_t(cow_free_after) == int64_t(cow_bytes) &&
        (aliases ? pool.table(1)[0] != old_binding : pool.table(1)[0] == old_binding) &&
        page(pb, uint64_t(pool.table(1)[0])) == source_page, "COW identity/copy/physical peak differs");
    append(1, b, exec, 0, 0.125f, "cow_divergence");
    require(page(pa, uint64_t(pool.table(0)[0])) == source_page &&
        page(pb, uint64_t(pool.table(1)[0])) != source_page, "COW changed peer or failed divergence");
    Device<float> pooled(65538ull * 128);
    check(cudaMemset(pooled.p, 0, 65538ull * 128 * sizeof(float)), "private indexer image");
    core::QsaState state;
    require(pool.mount(state, 1, b, error), error.c_str());
    state.idx_pooled_rows = 65538; state.idx_pooled = pooled.p;
    core::ConversationKv image;
    const core::ModelGeometry geometry;
    require(core::conversation_kv_save(image, state, geometry, 65540, true, error), error.c_str());
    require(pool.ready(1, b), "target ready");
    ++stats.layers; stats.release_chunks += released; stats.cow_chunks += cow_bytes / plan::kKvShareChunk;
    stats.host_bytes += pool.host_bytes(); stats.live_bytes += pool.mapped_bytes(); stats.handoff_ms += handoff_ms;
    stats.kernel_private_ms += private_kernel_ms; stats.kernel_candidate_ms += candidate_kernel_ms;
    const double ready_us = trace.quote_us + trace.entry_drain_us + trace.acquire_us +
        trace.table_copy_us + trace.publication_drain_us + trace.bind_us;
    std::printf("KV_SHARE_PHASES arm=%s layer=%d device=%d quote_us=%.3f entry_drain_us=%.3f acquire_us=%.3f table_copy_us=%.3f publication_drain_us=%.3f bind_us=%.3f alias_ready_us=%.3f directory_us=%.3f unmap_us=%.3f physical_release_us=%.3f alias_total_us=%.3f boundary_us=%.3f cow_us=%.3f table_bytes=%llu unmaps=%llu releases=%llu kernel_private_ms=%.6f kernel_candidate_ms=%.6f\n",
        aliases ? "alias" : "private", layer, device, trace.quote_us, trace.entry_drain_us, trace.acquire_us,
        trace.table_copy_us, trace.publication_drain_us, trace.bind_us, ready_us, trace.directory_us, trace.unmap_us, trace.free_us,
        trace.total_us, boundary_us, cow_us, (unsigned long long) trace.table_bytes,
        (unsigned long long) trace.unmaps, (unsigned long long) trace.releases, private_kernel_ms, candidate_kernel_ms);
    std::printf("KV_SHARE_PUBLICATION layer=%d arm=%s table_kind=device_to_device table_calls=%llu incoming_events=%llu publication_events=%llu\n",
        layer, aliases ? "alias" : "private", (unsigned long long) trace.table_calls,
        (unsigned long long) trace.incoming_events, (unsigned long long) trace.publication_events);
    std::printf("KV_SHARE_IDENTITIES arm=%s layer=%d device=%d request_a=101 request_b=102 common_k=%p common_v=%p source_first=%d target_first_before_cow=%d target_first_after_cow=%d shared_last=%d boundary_a=%d boundary_b=%d measured_release_chunks=%llu measured_cow_chunks=%llu handoff_ms=%.6f\n",
        aliases ? "alias" : "private", layer, device, (const void*) pa.k_q, (const void*) pa.v_q4,
        source_table[0], readback[0], pool.table(1)[0], readback[16383], source_table[16384], boundary,
        (unsigned long long) released, (unsigned long long) (cow_bytes / plan::kKvShareChunk), handoff_ms);
    if (chain && layer + 1 < 12) screen((layer + 1) / 4, layer + 1, true, aliases, stats);
    check(cudaSetDevice(device), "restore recursive stage device");
    const auto held = pool.ownership().binding(1, 1);
    const uint32_t references = pool.ownership().extent(held.extent).references;
    const uint64_t cancelled_bytes = pool.mapped_bytes(), cancelled_chunks = pool.released_chunks();
    require(pool.cancel(1, b) && pool.mapped_bytes() == cancelled_bytes &&
        pool.ownership().extent(held.extent).references == references, "cancel released before drain");
    require(pool.terminal(1, b, error) && !pool.terminal(1, b, error) &&
        pool.ownership().extent(held.extent).references == references - 1, "terminal not exactly once");
    const uint64_t retired = pool.released_chunks() - cancelled_chunks;
    stats.terminal_chunks += retired;
    append(0, a, peer_exec, 65540, 0.75f, "survivor_after_cancel");
    const core::KvOwnerKey replacement{103, 2, b.allocation};
    uint64_t fingerprint = 0;
    require(pool.begin(1, replacement, tb.p, error) && pool.mount(state, 1, replacement, error) &&
        core::conversation_kv_restore(image, state, geometry, 65540, true, error) &&
        core::conversation_kv_verify(image, state, geometry, 65540, true, fingerprint, error), error.c_str());
    select(0); commit_output(exec, 0, 0.125f, "image_reuse_new_generation");
    require(!pool.writable(1, b, 0, error) && !pool.terminal(1, b, error), "stale generation affected image reuse");
    ++stats.reused_images;
    require(pool.terminal(1, replacement, error), error.c_str());
    check(cudaGraphExecDestroy(exec), "destroy target graph"); check(cudaGraphDestroy(graph), "destroy target graph definition");
    check(cudaGraphExecDestroy(peer_exec), "destroy source graph"); check(cudaGraphDestroy(peer_graph), "destroy source graph definition");
    for (size_t owner = 0; owner < append_graphs.size(); ++owner) {
        check(cudaGraphExecDestroy(append_execs[owner]), "destroy append executable");
        check(cudaGraphDestroy(append_graphs[owner]), "destroy append definition");
    }
    check(cudaStreamDestroy(stream), "destroy stream");
    check(cudaStreamDestroy(producer_stream), "destroy producer stream");
    check(cudaEventDestroy(kernel_start), "destroy kernel start event"); check(cudaEventDestroy(kernel_stop), "destroy kernel stop event");
    require(pool.terminal(0, a, error) && pool.mapped_bytes() == 0, error.c_str());
    std::printf("KV_SHARE_DEVICE arm=%s layer=%d device=%d captured=4 instantiated=4 captured_append=1 writable_before_graph=1 attention_exact=1 divergent_tails=1 cow_exact=1 cancel_no_release=1 terminal_once=1 survivor=1 image_reuse=1 measured_release_chunks=%llu measured_release_bytes=%llu measured_cow_bytes=%llu cancel_terminal_chunks=%llu host_metadata_bytes=%llu\n",
        aliases ? "alias" : "private", layer, device, (unsigned long long) released,
        (unsigned long long) (released * plan::kKvShareChunk), (unsigned long long) cow_bytes,
        (unsigned long long) retired, (unsigned long long) pool.host_bytes());
}
void image_contract() {
    check(cudaSetDevice(0), "select image-contract device");
    plan::KvShareBudget budget{plan::KvShareMode::alias, 262144, 4, 12, 3,
        plan::kKvShareChunk, 12ull << 30, 3299132688};
    core::KvSharedPool pool(budget);
    std::string error;
    if (!pool.init(error)) throw std::runtime_error(error);
    const core::KvOwnerKey a{201, 1, 111}, b{202, 1, 222};
    Device<int32_t> ta(65536), tb(65536);
    if (!pool.begin(0, a, ta.p, error) || !pool.begin(1, b, tb.p, error) || !pool.ready(0, a))
        throw std::runtime_error(error);
    const auto pa = pool.pools(0), pb = pool.pools(1);
    initialize(pa, uint64_t(pool.table(0)[0]), 16385);
    const auto src = arrays(pa), dst = arrays(pb);
    for (size_t part = 0; part < 3; ++part)
        check(cudaMemcpy(dst[part] + uint64_t(pool.table(1)[0]) * widths[part],
                         src[part] + uint64_t(pool.table(0)[0]) * widths[part], 16385 * widths[part],
                         cudaMemcpyDeviceToDevice), "image private boundary copy");
    // Independent ordinary backing is populated BEFORE alias/COW. It does not
    // consult either shared table when forming the canonical expected bytes.
    Device<int8_t> private_k(16385 * widths[0]);
    Device<uint16_t> private_scale(16385 * widths[1] / sizeof(uint16_t));
    Device<uint8_t> private_v(16385 * widths[2]);
    Device<float> private_index(16387ull * 128);
    core::QsaState ordinary;
    ordinary.kv_hybrid = true; ordinary.max_cells = 65540; ordinary.n_pages = ordinary.n_slots = 16385;
    ordinary.k_q = private_k.p; ordinary.k_scale = private_scale.p; ordinary.v_q4 = private_v.p;
    ordinary.idx_pooled_rows = 16387; ordinary.idx_pooled = private_index.p;
    const std::array<void*, 3> private_parts{private_k.p, private_scale.p, private_v.p};
    for (size_t part = 0; part < 3; ++part)
        check(cudaMemcpy(private_parts[part], src[part], 16385 * widths[part], cudaMemcpyDeviceToDevice),
            "independent ordinary prefix");
    check(cudaMemset(private_k.p, 7, widths[0]), "independent ordinary COW oracle");
    check(cudaMemset(private_index.p, 0, 16387ull * 128 * sizeof(float)), "independent private indexer");
    if (!pool.alias(1, b, 0, a, 65539, certificate(a, 65539), error) || !pool.writable(1, b, 0, error))
        throw std::runtime_error(error);
    const uint64_t physical = uint64_t(pool.table(1)[0]);
    check(cudaMemset(dst[0] + physical * widths[0], 7, (size_t) widths[0]), "image COW divergence");
    const auto expected = page(pb, physical);
    Device<float> pooled(65538ull * 128);
    check(cudaMemset(pooled.p, 0, 65538ull * 128 * sizeof(float)), "private image indexer");
    core::QsaState state;
    if (!pool.mount(state, 1, b, error)) throw std::runtime_error(error);
    state.idx_pooled_rows = 65538; state.idx_pooled = pooled.p;
    Device<float> tail(384), dead(128);
    Device<int32_t> block_pos(1);
    state.idx_tail = tail.p; state.idx_dead = dead.p; state.idx_block_pos = block_pos.p;
    core::ConversationKv image;
    const core::ModelGeometry geometry;
    if (!core::conversation_kv_save(image, state, geometry, 65539, true, error))
        throw std::runtime_error(error);
    const bool logical_match = image.k.visit(0, 2048, [&](const uint8_t* data, size_t bytes, size_t at) {
        return std::memcmp(data, expected.data() + at, bytes) == 0;
    });
    uint8_t captured_first = 0;
    require(image.k.visit(0, 1, [&](const uint8_t* data, size_t, size_t) {
        captured_first = *data; return true;
    }), "image first-byte read");
    uint64_t fingerprint = 0;
    const bool self_verify = core::conversation_kv_verify(image, state, geometry, 65539, true, fingerprint, error);
    require(logical_match && self_verify, "COW-bound logical capture/verify mismatch");
    require(core::conversation_kv_verify(image, ordinary, geometry, 65539, true, fingerprint, error),
        "logical capture differs from independent ordinary K/scales/V/private-indexer oracle");
    core::ConversationKv wrong = image;
    require(wrong.k.visit(0, 1, [](uint8_t* data, size_t, size_t) { *data ^= 1; return true; }), "mutate raw image");
    require(!core::conversation_kv_verify(wrong, state, geometry, 65539, true, fingerprint, error),
        "logical verifier accepted a corrupt COW-bound image");
    wrong = image; wrong.k = {};
    require(!core::conversation_kv_validate(wrong, state, geometry, 65539, true, error), "truncated image accepted");
    wrong = image; ++wrong.head_dim;
    require(!core::conversation_kv_validate(wrong, state, geometry, 65539, true, error), "wrong geometry accepted");
    wrong = image; wrong.format = kernels::kKvInt8;
    require(!core::conversation_kv_validate(wrong, state, geometry, 65539, true, error), "wrong format accepted");
    auto& mapped = const_cast<std::vector<int32_t>&>(pool.table(1));
    const int32_t real_page = mapped[0];
    mapped[0] = int32_t(pool.ownership().extent_count());
    require(!core::conversation_kv_verify(image, state, geometry, 65539, true, fingerprint, error) &&
        !core::conversation_kv_restore(image, state, geometry, 65539, true, error), "invalid mapping accepted");
    mapped[0] = real_page;
    core::SessionKvSource source;
    if (!core::conversation_kv_source(source, state, geometry, 65539, true, error)) throw std::runtime_error(error);
    std::vector<uint8_t> streamed(2048);
    require(source.read(0, 0, streamed.data(), streamed.size()) &&
        std::equal(streamed.begin(), streamed.end(), expected.begin()), "streamed logical image source mismatch");
    const auto borrowed_bytes = pool.mapped_bytes();
    require(!core::kv_shared_reset(state, error) && !pool.writable(1, b, 0, error) &&
        !pool.terminal(1, b, error) && !pool.mount(state, 1, b, error) && !pool.mount(state, 0, a, error) &&
        pool.mapped_bytes() == borrowed_bytes,
        "a streamed source allowed same-key reset/write/terminal or descriptor remount");
    ++state.shared_request;
    require(!source.read(0, 0, streamed.data(), streamed.size()), "stale mounted source generation accepted");
    state.shared_request = b.request;
    auto duplicate_source = source;
    source = {};
    require(!core::kv_shared_reset(state, error), "copying a stream source lost its borrow");
    duplicate_source = {};
    require(core::conversation_kv_copy(state, ordinary, geometry, 65539, true, error) &&
        core::conversation_kv_verify(image, ordinary, geometry, 65539, true, fingerprint, error),
        "mapped-to-ordinary direct copy mismatch");
    const auto peer_before = page(pa, uint64_t(pool.table(0)[0]));
    core::qsa_state_zero(state, geometry, nullptr);
    require(page(pa, uint64_t(pool.table(0)[0])) == peer_before, "logical reset changed peer");
    if (!core::conversation_kv_restore(image, state, geometry, 65539, true, error) ||
        !core::conversation_kv_verify(image, state, geometry, 65539, true, fingerprint, error))
        throw std::runtime_error(error);
    require(page(pb, uint64_t(pool.table(1)[0])) == expected &&
        page(pa, uint64_t(pool.table(0)[0])) == peer_before, "logical roundtrip lost COW bytes or changed peer");
    Device<int32_t> tc(65536);
    const core::KvOwnerKey c{203, 1, 333};
    core::QsaState copy;
    Device<float> copy_pooled(65538ull * 128);
    if (!pool.begin(2, c, tc.p, error) || !pool.mount(copy, 2, c, error)) throw std::runtime_error(error);
    copy.idx_pooled_rows = 65538; copy.idx_pooled = copy_pooled.p;
    if (!core::conversation_kv_copy(state, copy, geometry, 65539, true, error) ||
        !core::conversation_kv_verify(image, copy, geometry, 65539, true, fingerprint, error))
        throw std::runtime_error(error);
    require(core::conversation_kv_copy(ordinary, copy, geometry, 65539, true, error) &&
        core::conversation_kv_verify(image, copy, geometry, 65539, true, fingerprint, error),
        "ordinary-to-mapped direct copy mismatch");
    require(pool.ready(1, b) && pool.cancel(1, b), "cancel restored owner");
    if (!pool.terminal(1, b, error)) throw std::runtime_error(error);
    require(core::conversation_kv_verify(image, copy, geometry, 65539, true, fingerprint, error),
        "copied survivor failed after cancellation");
    const core::KvOwnerKey replacement{204, 2, 222};
    if (!pool.begin(1, replacement, tb.p, error) || !pool.mount(state, 1, replacement, error) ||
        !core::conversation_kv_restore(image, state, geometry, 65539, true, error) ||
        !core::conversation_kv_verify(image, state, geometry, 65539, true, fingerprint, error))
        throw std::runtime_error(error);
    require(!pool.writable(1, b, 0, error) && !pool.reset(1, b, error), "old generation mutated replacement");
    std::puts("KV_SHARE_IMAGE_REUSE capture=exact independent_private=exact corrupt_image=refused truncated=refused geometry=refused format=refused invalid_mapping=refused streamed=exact borrow_reset_write_terminal=refused copied_borrow=held stale_source=refused reset_peer=unchanged roundtrip=exact mapped_ordinary_copy=exact direct_copy=exact cancelled_peer=survives replacement_generation=exact format_bytes=unchanged");
    std::printf("KV_SHARE_IMAGE_CONTRACT device=0 prefix=65539 logical_page=0 physical_page=%llu expected_first_byte=%u captured_first_byte=%u logical_capture_match=%d production_self_verify=%d kv_data_bytes=%llu private_indexer_bytes=%llu\n",
        (unsigned long long) physical, unsigned(expected[0]), unsigned(captured_first), int(logical_match),
        int(self_verify), (unsigned long long) (image.k.size() + image.v.size() + image.k_scale.size()),
        (unsigned long long) image.pooled.size());
    std::fflush(stdout);
    if (!pool.terminal(1, replacement, error) || !pool.terminal(2, c, error) || !pool.terminal(0, a, error))
        throw std::runtime_error(error);
}
}
int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--help") == 0) {
        std::puts("kv_shared_pool_test [--preflight|--image-contract|--pair-contract]: selected devices; no model loaded; exit78=required VMM unavailable");
        return 0;
    }
    int count = 0;
    try {
        require(argc <= 2 && (argc == 1 || std::strcmp(argv[1], "--preflight") == 0 ||
            std::strcmp(argv[1], "--image-contract") == 0 || std::strcmp(argv[1], "--pair-contract") == 0),
            "unknown fixture option");
        const char* window_ticket = std::getenv("STRATA_KV_SHARE_TEST_WINDOW");
        if (!window_ticket || !*window_ticket) {
            std::puts("KV_SHARE_BLOCKED exclusive protected measure wrapper required; no CUDA calls issued");
            return 78;
        }
        check(cudaGetDeviceCount(&count), "selected device inventory");
        require(count == 3, "expected the three frozen profile UUIDs, no implicit device substitution");
        bool available = true;
        for (int device = 0; device < count; ++device) {
            check(cudaSetDevice(device), "preflight device");
            check(cudaFree(nullptr), "initialize selected CUDA context");
            const auto receipt = core::vmm_status();
            const bool vmm = receipt.available;
            const uint64_t granularity = receipt.granularity;
            std::printf("KV_SHARE_PREFLIGHT device=%d vmm=%d granularity=%llu required_granularity=%llu phase=%s result=%d capability=%d allocation_device=%d\n",
                device, int(vmm), (unsigned long long) granularity, (unsigned long long) plan::kKvShareChunk,
                receipt.phase, receipt.result, receipt.capability, receipt.device);
            std::fflush(stdout);
            available = available && vmm && granularity == plan::kKvShareChunk;
        }
        if (!available) { std::puts("KV_SHARE_BLOCKED required stage-local VMM/2MiB backing unavailable; no aliases issued"); return 78; }
        if (argc > 1 && std::strcmp(argv[1], "--preflight") == 0) return 0;
        if (argc > 1 && std::strcmp(argv[1], "--image-contract") == 0) { image_contract(); return 0; }
        if (argc > 1 && std::strcmp(argv[1], "--pair-contract") == 0) {
            for (int device = 0; device < count; ++device) {
                PairStats warm; screen(device, device * 4, false, false, warm);
            }
            for (bool aliases : {false, true}) {
                PairStats stats;
                const auto started = std::chrono::steady_clock::now();
                screen(0, 0, true, aliases, stats);
                const double complete_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - started).count();
                require(stats.layers == 12 && stats.reused_images == 12 &&
                    stats.host_bytes <= plan::kKvShareHostLimit &&
                    stats.release_chunks == (aliases ? 300u : 0u) && stats.cow_chunks == (aliases ? 36u : 0u),
                    "12-layer measured pair contract incomplete");
                std::printf("KV_SHARE_PAIR_SUMMARY arm=%s layers=12 stages=3 simultaneous_owner_pairs=12 release_chunks=%llu measured_release_bytes=%llu cow_chunks=%llu measured_cow_bytes=%llu host_metadata_bytes=%llu live_pool_bytes_at_coexistence=%llu cancelled_terminal_chunks=%llu images_reused=%llu handoff_ms=%.6f kernel_private_sum_ms=%.6f kernel_candidate_sum_ms=%.6f complete_fixture_call_ms=%.6f scope=target_kv_fixture_not_full_model_client\n",
                    aliases ? "alias" : "private", (unsigned long long) stats.release_chunks,
                    (unsigned long long) (stats.release_chunks * plan::kKvShareChunk), (unsigned long long) stats.cow_chunks,
                    (unsigned long long) (stats.cow_chunks * plan::kKvShareChunk), (unsigned long long) stats.host_bytes,
                    (unsigned long long) stats.live_bytes, (unsigned long long) stats.terminal_chunks,
                    (unsigned long long) stats.reused_images, stats.handoff_ms, stats.kernel_private_ms,
                    stats.kernel_candidate_ms, complete_ms);
                std::fflush(stdout);
            }
            image_contract();
        } else {
            for (int device = 0; device < count; ++device) {
                PairStats stats; screen(device, device * 4, false, true, stats);
            }
        }
    } catch (const std::exception& e) { std::fprintf(stderr, "kv_shared_pool_test: %s\n", e.what()); return 1; }
    return 0;
}
