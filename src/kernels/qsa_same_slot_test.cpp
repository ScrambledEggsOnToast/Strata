// Bounded synthetic probe for verify.cpp's same-owner QSA branch. No model files.
// Exercise the actual SM70/pre75 and SM86/default K8V4 routes; a fallback is not a pass.
#include "strata/core/batch_rows.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_select.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

namespace k = strata::kernels;
namespace c = strata::core;
namespace {
constexpr int kRows = 8, kCells = 4096;
constexpr size_t kGuard = 16;
bool expect_pre75 = true;
void require(bool value, const char* message) {
    if (!value) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(1); }
}
void ck(cudaError_t error) {
    if (error != cudaSuccess) { std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(error)); std::exit(2); }
}

template<class T> struct Buffer {
    T* allocation = nullptr;
    T* data = nullptr;
    size_t size = 0;
    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    ~Buffer() { if (allocation) cudaFree(allocation); }
    void init(size_t count) {
        size = count;
        ck(cudaMalloc(&allocation, (size + 2 * kGuard) * sizeof(T)));
        data = allocation + kGuard;
    }
    void poison(cudaStream_t stream) { ck(cudaMemsetAsync(allocation, 0xa5, (size + 2 * kGuard) * sizeof(T), stream)); }
    void upload(const std::vector<T>& host, cudaStream_t stream) {
        require(host.size() == size, "input size");
        ck(cudaMemcpyAsync(data, host.data(), size * sizeof(T), cudaMemcpyHostToDevice, stream));
        ck(cudaStreamSynchronize(stream)); // pageable host vector remains alive through completion
    }
    std::vector<T> read() const {
        std::vector<T> host(size);
        ck(cudaMemcpy(host.data(), data, size * sizeof(T), cudaMemcpyDeviceToHost));
        return host;
    }
    void guard(size_t active) const {
        require(active <= size, "guard active bounds");
        std::vector<uint8_t> host((size + 2 * kGuard) * sizeof(T));
        ck(cudaMemcpy(host.data(), allocation, host.size(), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < kGuard * sizeof(T); ++i) require(host[i] == 0xa5, "prefix guard changed");
        for (size_t i = (kGuard + active) * sizeof(T); i < host.size(); ++i)
            require(host[i] == 0xa5, "inactive row or suffix guard changed");
    }
};

struct Pool {
    Buffer<float> pooled, dead;
    Buffer<int32_t> pages;
    Buffer<int8_t> keys;
    Buffer<uint16_t> scales;
    Buffer<uint8_t> values;
    k::QsaAttnPools view{};
    void init(const k::QsaShapes& s, int capacity, int blocks, unsigned seed, cudaStream_t stream) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> nd(-0.5f, 0.5f);
        pooled.init(size_t(blocks * s.idx_dim)); dead.init(size_t(s.idx_dim));
        pages.init(size_t(capacity / s.page_size));
        const size_t cells = size_t(capacity * s.n_head_kv), hd = size_t(s.head_dim);
        keys.init(cells * hd); scales.init(cells * hd / k::KV_Q8_GROUP);
        values.init(cells * size_t(k::kv_q4_bytes_per_head(int(hd))));
        std::vector<float> hp(pooled.size), hdv(dead.size);
        for (auto& v : hp) v = nd(rng);
        for (auto& v : hdv) v = nd(rng);
        pooled.upload(hp, stream); dead.upload(hdv, stream);
        std::vector<int32_t> pt(pages.size);
        std::iota(pt.begin(), pt.end(), 0);
        std::shuffle(pt.begin(), pt.end(), rng);
        pt[3] = -1; // masked page inside every exercised context
        pages.upload(pt, stream);
        std::vector<int8_t> hk(keys.size);
        for (auto& v : hk) v = int8_t(int(rng() % 255) - 127);
        keys.upload(hk, stream);
        scales.upload(std::vector<uint16_t>(scales.size, 0x2000), stream); // finite positive fp16 2^-7
        std::vector<uint8_t> hv(values.size);
        auto* q4 = reinterpret_cast<k::block_q4_0*>(hv.data());
        for (size_t i = 0; i < hv.size() / sizeof(*q4); ++i) {
            q4[i].d = 0x2000;
            for (auto& q : q4[i].qs) q = uint8_t(rng());
        }
        values.upload(hv, stream);
        view.page_table = pages.data; view.k_q = keys.data; view.k_scale = scales.data; view.v_q4 = values.data;
    }
};

struct Graph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    int pre75_nodes = 0;
    int default_nodes = 0, wide_topk_nodes = 0;
    ~Graph() { if (exec) cudaGraphExecDestroy(exec); if (graph) cudaGraphDestroy(graph); }
};

struct Probe {
    k::QsaShapes shapes = k::qsa_real_shapes();
    const int cells, blocks, cap;
    const int qi = int(shapes.idx_n_head * shapes.idx_dim), qo = int(shapes.n_head * shapes.head_dim);
    const size_t scratch_stride = k::qsa_decode_attn_scratch_floats(cap, shapes);
    cudaStream_t stream = nullptr;
    Pool pools[2];
    Buffer<float> query_index, query, scores[2], scratch[2], output[2];
    Buffer<int32_t> steps, selection[2];
    std::vector<int32_t> host_steps;
    int count, context;
    bool mixed_mode = false;
    std::map<c::VerifyBatchKey, std::unique_ptr<Graph>> cache;
    explicit Probe(int n, int ctx, int capacity = kCells)
        : cells(capacity), blocks(capacity / int(shapes.idx_block) + 2),
          cap(int(k::qsa_selection_width(capacity, shapes))), count(n), context(ctx) {
        ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        for (int i = 0; i < 2; ++i) pools[i].init(shapes, cells, blocks, 912 + i, stream);
        query_index.init(size_t(kRows * qi)); query.init(size_t(kRows * qo)); steps.init(size_t(kRows * k::kStepCount));
        for (int arm = 0; arm < 2; ++arm) {
            scores[arm].init(size_t(kRows * blocks)); selection[arm].init(size_t(kRows * cap));
            scratch[arm].init(size_t(kRows) * scratch_stride); output[arm].init(size_t(kRows * qo));
        }
    }
    ~Probe() { ck(cudaStreamSynchronize(stream)); cache.clear(); ck(cudaStreamDestroy(stream)); }
    void input(int revision) {
        std::mt19937 rng(91);
        std::uniform_real_distribution<float> nd(-0.25f, 0.25f);
        std::vector<float> hqi(query_index.size, std::numeric_limits<float>::quiet_NaN());
        std::vector<float> hq(query.size, std::numeric_limits<float>::quiet_NaN());
        for (int i = 0; i < count * qi; ++i)
            hqi[size_t(i)] = nd(rng) + ((!mixed_mode || i / qi < count / 2) ? revision * 0.1f : 0.0f);
        for (int i = 0; i < count * qo; ++i)
            hq[size_t(i)] = nd(rng) + ((!mixed_mode || i / qo < count / 2) ? revision * 0.1f : 0.0f);
        host_steps.assign(steps.size, -123456);
        for (int row = 0; row < count; ++row) {
            const int depth = mixed_mode && row < count / 2 ? 2040 : context;
            const int shift = !mixed_mode || row < count / 2 ? revision * 4 : 0;
            k::qsa_step_fill(host_steps.data() + row * k::kStepCount, depth - 1 + row + shift, shapes);
        }
        if (mixed_mode)
            require(host_steps[k::kStepWidth] != host_steps[size_t((count - 1) * k::kStepCount + k::kStepWidth)],
                    "mixed owners must exercise different attention widths");
        query_index.upload(hqi, stream); query.upload(hq, stream); steps.upload(host_steps, stream);
    }
    void poison() {
        for (int arm = 0; arm < 2; ++arm) {
            scores[arm].poison(stream); selection[arm].poison(stream);
            scratch[arm].poison(stream); output[arm].poison(stream);
        }
        ck(cudaStreamSynchronize(stream));
    }
    bool mixed_supported(const std::vector<int>& owners, bool candidate, bool resident,
                         k::QsaAttnPoolsRows& rows) {
        if (!mixed_mode || !candidate || !resident || count < 2 || count > k::kAttnRowsMax) return false;
        bool different = false;
        rows.n = count;
        for (int row = 0; row < count; ++row) {
            rows.p[row] = pools[owners[size_t(row)]].view;
            different = different || owners[size_t(row)] != owners[0];
        }
        return different && k::qsa_decode_attn_mixed_pre75_supported(rows);
    }
    // part=0 full pipeline, 1 scores, 2 top-k, 3 chunk+merge. Same eligibility helper as verify.cpp.
    void launch(const std::vector<int>& owners, bool candidate, bool resident = true, int part = 0) {
        require(int(owners.size()) == count, "owner extent");
        const int arm = candidate ? 1 : 0;
        const bool grouped = !mixed_mode &&
            c::qsa_same_slot_rows_eligible(candidate, true, owners.data(), count, resident);
        k::QsaAttnPoolsRows mixed_pools{};
        const bool mixed = mixed_supported(owners, candidate, resident, mixed_pools);
        const int stride = grouped ? count : 1;
        for (int row = 0; row < count; row += stride) {
            const Pool& pool = pools[owners[size_t(row)]];
            if (part == 0 || part == 1)
                k::qsa_block_scores(pool.pooled.data, pool.dead.data, query_index.data + row * qi,
                    steps.data + row * k::kStepCount, stride, blocks, shapes, scores[arm].data + row * blocks, stream);
            if (part == 0 || part == 2)
                k::qsa_block_topk(scores[arm].data + row * blocks, steps.data + row * k::kStepCount,
                    stride, blocks, cap, shapes, selection[arm].data + row * cap, stream);
            if (!mixed && (part == 0 || part == 3))
                k::qsa_decode_attn_batch(query.data + row * qo, pool.view, selection[arm].data + row * cap,
                    steps.data + row * k::kStepCount, cap, shapes, scratch[arm].data + size_t(row) * scratch_stride,
                    output[arm].data + row * qo, stride, stream);
        }
        if (mixed && (part == 0 || part == 3))
            require(k::qsa_decode_attn_mixed_pre75(query.data, mixed_pools, selection[arm].data,
                steps.data, cap, shapes, scratch[arm].data, output[arm].data, count, stream),
                "mixed launch must retain its preflight eligibility");
    }
    std::unique_ptr<Graph> capture(const std::vector<int>& owners, bool candidate, bool resident = true, int part = 0) {
        auto graph = std::make_unique<Graph>();
        ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        launch(owners, candidate, resident, part);
        ck(cudaStreamEndCapture(stream, &graph->graph));
        size_t count_nodes = 0;
        ck(cudaGraphGetNodes(graph->graph, nullptr, &count_nodes));
        std::vector<cudaGraphNode_t> nodes(count_nodes);
        ck(cudaGraphGetNodes(graph->graph, nodes.data(), &count_nodes));
        const bool grouped = !mixed_mode &&
            c::qsa_same_slot_rows_eligible(candidate, true, owners.data(), count, resident);
        k::QsaAttnPoolsRows mixed_pools{};
        const bool mixed = mixed_supported(owners, candidate, resident, mixed_pools);
        for (auto node : nodes) {
            cudaGraphNodeType type;
            ck(cudaGraphNodeGetType(node, &type));
            if (type != cudaGraphNodeTypeKernel) continue;
            cudaKernelNodeParams params{};
            ck(cudaGraphKernelNodeGetParams(node, &params));
#if CUDART_VERSION >= 12030
            const char* name = nullptr;
            ck(cudaFuncGetName(&name, params.func));
            if (name && std::strstr(name, "attn_chunk_kernel")) {
                if (std::strstr(name, "attn_chunk_kernel_pre75")) ++graph->pre75_nodes;
                else ++graph->default_nodes;
                require(params.gridDim.z == unsigned(grouped || mixed ? count : 1), "attention live query grid");
            }
            if (name && std::strstr(name, "block_topk_wide_kernel")) {
                ++graph->wide_topk_nodes;
                require(params.gridDim.x == unsigned(grouped ? count : 1), "wide top-k live query grid");
            }
#else
            require(false, "CUDA >=12.3 required to establish actual kernel exposure");
#endif
        }
        const int attention_nodes = (part == 0 || part == 3) ? (grouped || mixed ? 1 : count) : 0;
        require(graph->pre75_nodes == (expect_pre75 ? attention_nodes : 0) &&
                graph->default_nodes == (expect_pre75 ? 0 : attention_nodes),
                "actual device-specific attention route must execute; fallback is not acceptance");
        if (cells == 262144)
            require(graph->wide_topk_nodes == ((part == 0 || part == 2) ? (grouped ? 1 : count) : 0),
                    "actual production-capacity wide top-k must execute");
        ck(cudaGraphInstantiate(&graph->exec, graph->graph, nullptr, nullptr, 0));
        return graph;
    }
    Graph& cached(const std::vector<int>& owners) {
        // This is the engine's exact cache-key helper, not a fixture-only width key.
        const auto key = c::verify_batch_graph_key(owners.data(), count, 0, false);
        auto it = cache.find(key);
        if (it == cache.end()) it = cache.emplace(key, capture(owners, true)).first;
        return *it->second;
    }
    std::vector<float> compare() {
        for (int arm = 0; arm < 2; ++arm) {
            scores[arm].guard(size_t(count * blocks)); selection[arm].guard(size_t(count * cap));
            scratch[arm].guard(size_t(count) * scratch_stride); output[arm].guard(size_t(count * qo));
        }
        const auto sa = scores[0].read(), sb = scores[1].read();
        const auto ia = selection[0].read(), ib = selection[1].read();
        const auto oa = output[0].read(), ob = output[1].read();
        require(std::memcmp(sa.data(), sb.data(), sa.size() * sizeof(float)) == 0, "score bits and untouched extents");
        require(std::memcmp(ia.data(), ib.data(), ia.size() * sizeof(int32_t)) == 0, "selection bits and untouched extents");
        require(std::memcmp(oa.data(), ob.data(), oa.size() * sizeof(float)) == 0, "attention bits and untouched extents");
        double magnitude = 0;
        for (int row = 0; row < count; ++row) {
            const int nb = host_steps[size_t(row * k::kStepCount + k::kStepNBid)] + 1;
            for (int block = 0; block < nb; ++block)
                require(std::isfinite(sa[size_t(row * blocks + block)]), "finite live score");
            for (int i = 0; i < qo; ++i) {
                const float a = oa[size_t(row * qo + i)], b = ob[size_t(row * qo + i)];
                require(std::isfinite(a) && std::isfinite(b), "finite live attention");
                magnitude += std::abs(double(a));
            }
        }
        require(magnitude > 0, "nontrivial attention witness");
        return std::vector<float>(ob.begin(), ob.begin() + count * qo);
    }
    std::vector<float> replay(const std::vector<int>& owners, Graph& reference, Graph& candidate) {
        poison();
        ck(cudaGraphLaunch(reference.exec, stream)); ck(cudaGraphLaunch(candidate.exec, stream));
        ck(cudaStreamSynchronize(stream));
        auto values = compare();
        std::printf("completed=1 rows=%d owners=", count);
        for (int owner : owners) std::printf("%d", owner);
        std::printf(" pre75_reference_nodes=%d pre75_candidate_nodes=%d exact=1 guards=1\n",
                    reference.pre75_nodes, candidate.pre75_nodes);
        return values;
    }
    double time(Graph& graph) {
        for (int i = 0; i < 3; ++i) ck(cudaGraphLaunch(graph.exec, stream));
        cudaEvent_t begin, end;
        ck(cudaEventCreate(&begin)); ck(cudaEventCreate(&end));
        ck(cudaEventRecord(begin, stream));
        constexpr int iterations = 30;
        for (int i = 0; i < iterations; ++i) ck(cudaGraphLaunch(graph.exec, stream));
        ck(cudaEventRecord(end, stream)); ck(cudaEventSynchronize(end));
        float milliseconds = 0;
        ck(cudaEventElapsedTime(&milliseconds, begin, end));
        ck(cudaEventDestroy(begin)); ck(cudaEventDestroy(end));
        return milliseconds / iterations;
    }
    void micro(const std::vector<int>& owners) {
        static const char* names[] = {"pipeline", "scores", "topk", "attention"};
        for (int part = 0; part < 4; ++part) {
            auto reference = capture(owners, false, true, part), candidate = capture(owners, true, true, part);
            const double before = time(*reference), changed = time(*candidate), after = time(*reference);
            std::printf("micro rows=%d first_context=%d part=%s ref_before_ms=%.6f candidate_ms=%.6f ref_after_ms=%.6f\n",
                        count, context, names[part], before, changed, after);
        }
    }
};

void policy_checks() {
    const int aa[] = {0, 0}, aaaa[] = {0, 0, 0, 0}, bbbb[] = {1, 1, 1, 1};
    const int aabb[] = {0, 0, 1, 1}, bad[] = {-1, -1}, five[] = {0, 0, 0, 0, 0};
    require(c::qsa_same_slot_rows_eligible(true, true, aa, 2, true), "eligible two rows");
    require(c::qsa_same_slot_rows_eligible(true, true, aaaa, 3, true), "eligible three rows");
    require(c::qsa_same_slot_rows_eligible(true, true, aaaa, 4, true), "eligible four rows");
    require(!c::qsa_same_slot_rows_eligible(false, true, aa, 2, true), "flag-off fallback");
    require(!c::qsa_same_slot_rows_eligible(true, false, aa, 2, true), "non-batch fallback");
    require(!c::qsa_same_slot_rows_eligible(true, true, aa, 2, false), "streamed fallback");
    require(!c::qsa_same_slot_rows_eligible(true, true, aabb, 4, true), "mixed-owner fallback");
    require(!c::qsa_same_slot_rows_eligible(true, true, aa, 1, true), "single-row fallback");
    require(!c::qsa_same_slot_rows_eligible(true, true, five, 5, true), "width-five fallback");
    require(!c::qsa_same_slot_rows_eligible(true, true, bad, 2, true), "invalid-owner fallback");
    const auto a = c::verify_batch_graph_key(aaaa, 4, 0, false);
    require(a != c::verify_batch_graph_key(aabb, 4, 0, false), "mixed layout has separate cache key");
    require(a != c::verify_batch_graph_key(bbbb, 4, 0, false), "other owner has separate cache key");
    require(a != c::verify_batch_graph_key(aaaa, 4, 8, false), "handoff offset has separate cache key");
    require(a != c::verify_batch_graph_key(aaaa, 4, 0, true), "doorbell variant has separate cache key");
}
} // namespace

int main(int argc, char** argv) {
    policy_checks();
    require(argc <= 3, "usage: qsa_same_slot_test [CUDA-device-ordinal] [mixed]");
    const bool mixed_mode = argc == 3;
    require(!mixed_mode || std::strcmp(argv[2], "mixed") == 0, "unknown fixture mode");
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    int device = 0;
    if (argc >= 2) {
        char* end = nullptr;
        const long parsed = std::strtol(argv[1], &end, 10);
        require(end != argv[1] && *end == '\0' && parsed >= 0 && parsed < devices, "CUDA device ordinal");
        device = int(parsed);
    }
    ck(cudaSetDevice(device));
    cudaDeviceProp properties{}; ck(cudaGetDeviceProperties(&properties, device));
    const int sm = properties.major * 10 + properties.minor;
    require(sm == 70 || sm == 86, "this bounded probe supports only qualified SM70/SM86");
    expect_pre75 = sm == 70;
    std::printf("DEVICE ordinal=%d sm=%d attention=%s\n", device, sm, expect_pre75 ? "pre75" : "default");
    // The actual device-specific K8V4 path must be selected; conflicting ambient selectors refuse.
    for (const char* name : {"STRATA_ATTN_PRE75", "STRATA_ATTN_DAILY", "STRATA_ATTN_LANECELL",
                             "STRATA_ATTN_V3", "STRATA_QSA_ROWS", "STRATA_SCORES_MULTI",
                             "STRATA_SCORES_HALVE", "STRATA_TOPK_OLD", "STRATA_TOPK_SPLIT",
                             "STRATA_EMULATE_CC"})
        require(std::getenv(name) == nullptr, "clear independent attention overrides before probe");
    if (mixed_mode) {
        for (int rows : {5, 8}) {
            Probe p(rows, 262132, 262144);
            p.mixed_mode = true;
            std::vector<int> owners(size_t(rows), 1);
            for (int row = 0; row < rows / 2; ++row) owners[size_t(row)] = 0;
            p.input(0); p.poison();
            auto reference = p.capture(owners, false);
            Graph& candidate = p.cached(owners);
            require(candidate.pre75_nodes == (expect_pre75 ? 1 : 0) &&
                    candidate.default_nodes == (expect_pre75 ? 0 : rows),
                    "SM70 must execute mixed pre75; SM86 must retain scalar fallback");
            k::QsaAttnPoolsRows incomplete{};
            incomplete.n = rows;
            for (int row = 0; row < rows; ++row) incomplete.p[row] = p.pools[owners[size_t(row)]].view;
            incomplete.p[0].k_scale = nullptr;
            require(!k::qsa_decode_attn_mixed_pre75_supported(incomplete), "missing K scale must refuse before launch");
            const auto first = p.replay(owners, *reference, candidate);
            p.input(1);
            const auto changed = p.replay(owners, *reference, candidate);
            require(first != changed, "mixed independent-pool replay must observe changed inputs");
            const size_t unchanged = size_t(rows / 2 * p.qo);
            require(std::memcmp(first.data() + unchanged, changed.data() + unchanged,
                                (first.size() - unchanged) * sizeof(float)) == 0,
                    "changing one owner must not change the other owner's attention");
            auto reversed = owners;
            for (int& owner : reversed) owner = 1 - owner;
            auto reversed_reference = p.capture(reversed, false);
            Graph& reversed_candidate = p.cached(reversed);
            p.replay(reversed, *reversed_reference, reversed_candidate);
            require(p.cache.size() == 2 && &p.cached(owners) == &candidate &&
                    &candidate != &reversed_candidate, "mixed owner layouts must retain distinct graph executables");
            require(changed == p.replay(owners, *reference, p.cached(owners)),
                    "revisiting owner layout must preserve its own pools");
            auto fallback = p.capture(owners, true, false);
            p.replay(owners, *reference, *fallback);
            std::fill(owners.begin(), owners.end(), 0);
            auto same_reference = p.capture(owners, false);
            auto same_candidate = p.capture(owners, true);
            p.replay(owners, *same_reference, *same_candidate);
        }
        std::puts("PASS mixed-pool QSA: full capacity, unequal depths, independent pools, changed replay, guards and fallbacks");
        return 0;
    }
    for (int rows : {2, 3, 4}) for (int context : {244, 2047}) {
        Probe p(rows, context);
        const std::vector<int> owners(size_t(rows), 0);
        p.input(0); p.poison();
        auto reference = p.capture(owners, false);
        Graph& candidate = p.cached(owners);
        const auto first = p.replay(owners, *reference, candidate);
        p.input(1);
        const auto changed = p.replay(owners, *reference, candidate);
        require(first != changed, "same captured graph must observe changed queries/steps");
        require(&p.cached(owners) == &candidate, "replay must reuse original cached executable");
        p.input(0); p.replay(owners, *reference, candidate);
        p.micro(owners);
    }
    // A single full-capacity case reaches the wide top-k kernel, not just its register counterpart.
    // Leave room for all four positions and the changed-input replay inside the allocated pool.
    {
        Probe wide(4, 262136, 262144);
        const std::vector<int> owners(4, 0);
        wide.input(0); wide.poison();
        auto reference = wide.capture(owners, false);
        Graph& candidate = wide.cached(owners);
        const auto first = wide.replay(owners, *reference, candidate);
        wide.input(1);
        require(first != wide.replay(owners, *reference, candidate),
                "wide top-k cached graph must observe changed input");
        wide.input(0); wide.replay(owners, *reference, candidate);
        std::puts("PASS capacity=262144 rows=4 wide_topk=executed exact_scores_ids_attention=1");
    }
    // Same width, three distinct owner layouts; different pools must never share an executable.
    Probe p(4, 2050); // crosses the 2051-cell selection-width boundary and 4-cell block/page boundary
    const std::vector<std::vector<int>> layouts{{0, 0, 0, 0}, {1, 1, 1, 1}, {0, 0, 1, 1}};
    std::vector<std::unique_ptr<Graph>> references;
    p.input(0); p.poison();
    std::vector<std::vector<float>> first;
    for (const auto& layout : layouts) {
        references.push_back(p.capture(layout, false));
        first.push_back(p.replay(layout, *references.back(), p.cached(layout)));
    }
    require(p.cache.size() == 3, "owner-layout cache must retain three distinct executables");
    require(first[0] != first[1], "separate pool witness must differ");
    p.input(1);
    for (int i : {2, 0, 1}) {
        const auto next = p.replay(layouts[size_t(i)], *references[size_t(i)], p.cached(layouts[size_t(i)]));
        require(first[size_t(i)] != next, "cached owner layout must observe changed input");
    }
    // Exercise the streamed eligibility fallback with synthetic resident backing; this does not
    // claim real streamed-KV resolution coverage, whose production path is left untouched.
    auto fallback = p.capture(layouts[0], true, false);
    p.replay(layouts[0], *references[0], *fallback);
    std::puts("PASS same-slot QSA: exact scores/IDs/attention, device-specific attention and wide top-k, cache-layout isolation, guards and changed-input replay");
    return 0;
}
