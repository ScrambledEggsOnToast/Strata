// prefill_fused_iq_test - #136: the fused int8 expert kernels for the native packs (moe_fused_iq.hpp, STRATA_PF_FUSED=1)
// on random native expert blobs ([gate rows | up rows | down rows], GGUF blocks) and random routing, per format pair,
// against
//   - a double-precision reference: the weights dequantized by ggml's own to_float, times the FP32 activations
//     (gate/up, SwiGLU, down), and
//   - the MMQ path in prefill.cpp's sequence (gather_native into 16-expert groups, q8_1 of the slots' activations,
//     gate/up, SwiGLU, q8_1 of H, down).
// Both round the activations and H to int8 per 32 values, differently, so this is not a bitwise test:
// the fused path's error against the reference has to be comparable to MMQ's own (at most 1.5x its RMS and 2x its worst row).
// The pairs cover every format the kernels take: gate/up IQ2_XXS IQ2_XS IQ2_S IQ3_XXS IQ3_S IQ4_XS, down Q2_0 IQ4_NL.
// --mmq-only runs the pairs' MMQ expert path ALONE (no fused launch at all, so it runs below sm_80 - V100's sm_70
// included - where MMQ's dp4a path works but the fused int8 kernels do not exist), against an independent
// double-precision reference at the engine's FF/H (640/2560) on a compact deterministic routing; see mmq_only_part.
// Part 1 (reference), per pair: 256 tokens over 64 experts, skewed routing - experts with 0 rows, with one, with
// several 64-row tiles - an all-zero token, per-expert blobs at unrelated (2-byte aligned) addresses, three launches of
// different expert ranges.  Part 2 (timing): one layer at the engine's chunks (2048, 3584, 8192 tokens; 512 experts,
// top 10) on both paths and on MMQ's products alone (no gathers), for the IQ2_XS, IQ3_XXS and IQ3_S packs' most
// common layers, with the grouping checked and a sample of rows against the reference.  --no-ref, --no-timing,
// --only=NAME, --chunks=A,B.  Exit 77 without a CUDA device of sm_80 or newer.
#include "strata/platform/protected_test.hpp"
#include "strata/prefill/moe_fused_iq.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "swiglu_reference.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {
namespace mmq = strata::prefill::mmq;
namespace fused = strata::prefill::fused;
using swiglu_reference::checked_swiglu;
using swiglu_reference::kUnitRoundoff;
using swiglu_reference::normal_or_zero;
using swiglu_reference::SwigluReferenceDomain;

constexpr int N = 2560, FF = 640, K = 10, GROUP = 16;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

struct Dev {
    void* p = nullptr;
    explicit Dev(size_t n) { ck(cudaMalloc(&p, n), "cudaMalloc"); }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    template <typename T> T* as() const { return (T*) p; }
};

struct Pair {
    ggml_type gu, d;
    const char* name;
};
// MMQ covers this pair (the HIP build has no Q4_K / Q5_K / Q5_1 MMQ: those pairs are checked against the reference only)
bool mmq_ok(const Pair& p) { return mmq::supported((int) p.gu) && mmq::supported((int) p.d); }

// the pairs of both modes: the fused kernels' formats, plus (HIP only) UD-Q4_K_XL's Q4_K / Q5_K / Q5_1 / Q8_0
const std::vector<Pair>& native_pairs() {
    static const std::vector<Pair> pairs = {
        {GGML_TYPE_IQ2_S, GGML_TYPE_Q2_0, "IQ2_S / Q2_0"},       {GGML_TYPE_IQ2_XXS, GGML_TYPE_Q2_0, "IQ2_XXS / Q2_0"},
        {GGML_TYPE_IQ2_XS, GGML_TYPE_IQ4_NL, "IQ2_XS / IQ4_NL"}, {GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_NL, "IQ3_XXS / IQ4_NL"},
        {GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_NL, "IQ3_S / IQ4_NL"},   {GGML_TYPE_IQ4_XS, GGML_TYPE_Q2_0, "IQ4_XS / Q2_0"},
        {GGML_TYPE_Q4_K, GGML_TYPE_Q5_1, "Q4_K / Q5_1"},         {GGML_TYPE_Q4_K, GGML_TYPE_Q8_0, "Q4_K / Q8_0"},
        {GGML_TYPE_Q5_K, GGML_TYPE_Q5_1, "Q5_K / Q5_1"},         {GGML_TYPE_Q5_K, GGML_TYPE_Q8_0, "Q5_K / Q8_0"}};
    return pairs;
}

struct Geo {
    size_t gu_row, d_row, up_off, down_off, bytes;
    explicit Geo(const Pair& p) {
        gu_row = ggml_row_size(p.gu, N);
        d_row = ggml_row_size(p.d, FF);
        up_off = gu_row * FF;
        down_off = 2 * up_off;
        bytes = down_off + d_row * N;
    }
    fused::NativeGeom native(const Pair& p) const {
        fused::NativeGeom g;
        g.gu_type = (int) p.gu; g.d_type = (int) p.d; g.gu_row = gu_row; g.d_row = d_row; g.up_off = up_off;
        g.down_off = down_off;
        return g;
    }
};

// random blocks with a sane fp16 scale `d` at each block's start (every format here begins with it): scales chosen so
// that the weights' typical size is that of a 2-3 bit expert (a few hundredths)
void fill_matrix(uint8_t* m, ggml_type t, int64_t rows, int64_t cols, std::mt19937& rng) {
    const size_t bs = ggml_type_size(t), nb = (size_t) rows * (size_t) (cols / ggml_blck_size(t));
    std::uniform_int_distribution<int> byte(0, 255);
    const float lo = t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q5_K ? 0.00004f : t == GGML_TYPE_Q5_1 ? 0.0006f
                   : t == GGML_TYPE_Q8_0 ? 0.0001f : t == GGML_TYPE_IQ3_S ? 0.0004f : t == GGML_TYPE_IQ4_XS ? 0.00005f
                   : t == GGML_TYPE_IQ4_NL ? 0.0002f : t == GGML_TYPE_Q2_0 ? 0.004f : 0.0008f;
    std::uniform_real_distribution<float> sc(lo, 6.0f * lo);
    for (size_t i = 0; i < nb * bs; ++i) m[i] = (uint8_t) byte(rng);
    for (size_t b = 0; b < nb; ++b) {
        const ggml_fp16_t h = ggml_fp32_to_fp16(sc(rng));
        std::memcpy(m + b * bs, &h, 2);
        if (t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q5_K || t == GGML_TYPE_Q5_1) {   // the minimum's scale, either sign
            const ggml_fp16_t h2 = ggml_fp32_to_fp16(sc(rng) * ((rng() & 1) ? 1.0f : -1.0f) * (t == GGML_TYPE_Q5_1 ? 4.0f : 1.0f));
            std::memcpy(m + b * bs + 2, &h2, 2);
        }
    }
}
std::vector<uint8_t> make_blob(const Pair& p, const Geo& g, std::mt19937& rng) {
    std::vector<uint8_t> b(g.bytes);
    fill_matrix(b.data(), p.gu, 2 * FF, N, rng);
    fill_matrix(b.data() + g.down_off, p.d, N, FF, rng);
    return b;
}

// ggml's dequantized weights: gate [640][2560], up [640][2560], down [2560][640]
void dequant(const Pair& p, const Geo& g, const std::vector<uint8_t>& b, std::vector<float>& gu, std::vector<float>& dn) {
    const auto* tg = ggml_get_type_traits(p.gu);
    const auto* td = ggml_get_type_traits(p.d);
    gu.resize((size_t) 2 * FF * N);
    dn.resize((size_t) N * FF);
    for (int r = 0; r < 2 * FF; ++r) tg->to_float(b.data() + (size_t) r * g.gu_row, gu.data() + (size_t) r * N, N);
    for (int r = 0; r < N; ++r) td->to_float(b.data() + g.down_off + (size_t) r * g.d_row, dn.data() + (size_t) r * FF, FF);
}

// activations: N(0, 1) with a few large values (the hidden state has outliers); token `zero` all zero
std::vector<float> make_x(int T, std::mt19937& rng, int zero) {
    std::vector<float> x((size_t) T * N);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_int_distribution<int> pick(0, 199);
    for (float& v : x) v = nd(rng) * (pick(rng) == 0 ? 12.0f : 1.0f);
    if (zero >= 0) std::fill(x.begin() + (size_t) zero * N, x.begin() + (size_t) (zero + 1) * N, 0.0f);
    return x;
}

// K distinct experts per token; `hot` > 0 skews the first choices towards the lowest ids
std::vector<int32_t> make_ids(int T, int E, std::mt19937& rng, int hot, int unused) {
    std::vector<int32_t> ids((size_t) T * K);
    std::uniform_int_distribution<int> any(0, E - 1 - unused);
    for (int t = 0; t < T; ++t)
        for (int k = 0; k < K; ++k) {
            int e;
            bool dup;
            do {
                e = (hot > 0 && k < 3 && (int) (rng() % 4) != 0) ? (int) (rng() % (unsigned) hot) : any(rng);
                dup = false;
                for (int j = 0; j < k; ++j) dup |= ids[(size_t) t * K + j] == e;
            } while (dup);
            ids[(size_t) t * K + k] = e;
        }
    return ids;
}

struct Routing {
    std::vector<int32_t> cnt, off, src;   // per expert; sorted row -> token (rows in pair order within an expert)
    std::vector<int32_t> row_of;          // pair -> sorted row
};
Routing sort_rows(const std::vector<int32_t>& ids, int E) {
    Routing r;
    r.cnt.assign((size_t) E, 0);
    for (int32_t e : ids) ++r.cnt[(size_t) e];
    r.off.assign((size_t) E + 1, 0);
    for (int e = 0; e < E; ++e) r.off[(size_t) e + 1] = r.off[(size_t) e] + r.cnt[(size_t) e];
    std::vector<int32_t> fill(r.off.begin(), r.off.end() - 1);
    r.src.resize(ids.size());
    r.row_of.resize(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) {
        const int32_t p = fill[(size_t) ids[i]]++;
        r.row_of[i] = p;
        r.src[(size_t) p] = (int32_t) (i / K);
    }
    return r;
}

// MMQ, as prefill.cpp runs a native layer: per group of 16 routed experts a gather, gate/up, SwiGLU (gate rows, then
// up rows), q8_1 of H, down.  Returns Dm [rows][N] in sorted-row order.
struct MmqBufs {
    Dev xq, gu, h, hq, dm, ident, bounds, grp_gu, grp_d;
    MmqBufs(const Pair& p, int64_t rows, int E)
        : xq(mmq::q8_bytes(rows, N)), gu((size_t) rows * 1280 * 4), h((size_t) rows * FF * 4),
          hq(mmq::q8_bytes(rows, FF)), dm((size_t) rows * N * 4), ident((size_t) rows * 4),
          bounds((size_t) (2 * (E + E / GROUP + 2)) * 4), grp_gu(GROUP * mmq::matrix_bytes(p.gu, 1280, N) + 4096),
          grp_d(GROUP * mmq::matrix_bytes(p.d, N, FF) + 4096) {}
};
// `direct`: no gathers - the products read the blobs where they are (they must lie at a fixed stride, every expert
// with rows: the timing part's layout), the cost of MMQ's own kernels alone
void run_mmq(const Pair& p, const Geo& geo, mmq::Context& ctx, MmqBufs& b, const Routing& r, const float* x_dev,
             const int32_t* src_dev, const std::vector<const uint8_t*>& blob, int64_t rows, cudaStream_t s,
             bool direct = false, size_t stride = 0) {
    const size_t gub = mmq::matrix_bytes(p.gu, 1280, N), db = mmq::matrix_bytes(p.d, N, FF);
    const int E = (int) r.cnt.size();
    std::vector<int32_t> order;
    for (int e = 0; e < E; ++e) if (r.cnt[(size_t) e] > 0) order.push_back(e);
    const size_t n = order.size(), ng = (n + GROUP - 1) / GROUP;
    std::vector<int32_t> bh(n + 1 + ng * (GROUP + 1));
    for (size_t j = 0; j < n; ++j) bh[j] = r.off[(size_t) order[j]];
    bh[n] = (int32_t) rows;
    for (size_t g = 0; g < ng; ++g)
        for (size_t i = 0; i <= GROUP; ++i)
            bh[n + 1 + g * (GROUP + 1) + i] = bh[std::min(n, g * GROUP + i)] - bh[g * GROUP];
    ck(cudaMemcpyAsync(b.bounds.p, bh.data(), bh.size() * 4, cudaMemcpyHostToDevice, s), "bounds");
    mmq::iota(b.ident.as<int32_t>(), rows, s);
    mmq::quantize(x_dev, src_dev, b.xq.p, (int) p.gu, N, N, rows, s);
    for (size_t j = 0; j < n; ++j) {
        const size_t q = j % GROUP;
        const uint8_t* bl = blob[(size_t) order[j]];
        if (!direct)
            mmq::gather_native(bl, bl + geo.up_off, gub / 2, bl + geo.down_off, db, b.grp_gu.as<uint8_t>() + q * gub,
                               b.grp_d.as<uint8_t>() + q * db, s);
        if (q + 1 < GROUP && j + 1 < n) continue;
        const size_t j0 = j - q, g = j0 / GROUP;
        const int ngx = (int) (q + 1);
        const int64_t r0 = bh[j0], nr = bh[j + 1] - r0;
        int64_t maxr = 0;
        for (size_t i = j0; i <= j; ++i) maxr = std::max<int64_t>(maxr, r.cnt[(size_t) order[i]]);
        if (!direct) {
            ck(cudaMemsetAsync(b.grp_gu.as<uint8_t>() + (size_t) ngx * gub, 0, 4096, s), "tail");
            ck(cudaMemsetAsync(b.grp_d.as<uint8_t>() + (size_t) ngx * db, 0, 4096, s), "tail");
        }
        const uint8_t* b0 = blob[(size_t) order[j0]];
        mmq::Product gu;
        gu.w = direct ? (const void*) b0 : b.grp_gu.p; gu.type = (int) p.gu; gu.w_rows = 1280; gu.w_cols = N;
        gu.expert_bytes = direct ? stride : gub;
        gu.n = ngx; gu.xq = b.xq.p; gu.bounds = b.bounds.as<int32_t>() + j0; gu.ids = b.ident.as<int32_t>();
        gu.total_rows = rows; gu.max_rows = maxr; gu.dst = b.gu.as<float>(); gu.ld_dst = 1280;
        ctx.run(gu, s);
        mmq::swiglu(b.gu.as<float>() + r0 * 1280, b.h.as<float>() + r0 * FF, nr, FF, false, s);
        mmq::quantize(b.h.as<float>() + r0 * FF, nullptr, b.hq.p, (int) p.d, FF, FF, nr, s);
        mmq::Product dn;
        dn.w = direct ? (const void*) (b0 + geo.down_off) : b.grp_d.p; dn.type = (int) p.d; dn.w_rows = N; dn.w_cols = FF;
        dn.expert_bytes = direct ? stride : db;
        dn.n = ngx; dn.xq = b.hq.p; dn.bounds = b.bounds.as<int32_t>() + n + 1 + g * (GROUP + 1);
        dn.ids = b.ident.as<int32_t>(); dn.total_rows = nr; dn.max_rows = maxr; dn.dst = b.dm.as<float>() + r0 * N;
        dn.ld_dst = N;
        ctx.run(dn, s);
    }
}

// the fused path: quantize, group on the device, then launches over expert ranges `cuts` (e.g. {0, 20, 64})
struct FusedBufs {
    Dev xa, ha, dm, scratch, slot, src, ids;
    FusedBufs(int T, int64_t rows, int E)
        : xa(fused::act_bytes(T, N)), ha(fused::act_bytes(rows, FF)), dm((size_t) rows * N * 4),
          scratch(fused::group_bytes(rows, E)), slot((size_t) rows * 4), src((size_t) rows * 4),
          ids((size_t) rows * 4) {}
};
void run_fused(const Pair& p, const Geo& geo, FusedBufs& b, const float* x_dev, int T, int E,
               const std::vector<int>& cuts, const std::vector<const uint8_t*>& blob, cudaStream_t s) {
    const int64_t rows = (int64_t) T * K;
    const fused::NativeGeom ng = geo.native(p);
    fused::quantize_act_native(x_dev, T, N, b.xa.p, s);
    fused::group(b.ids.as<int32_t>(), rows, K, E, b.scratch.p, b.slot.as<int32_t>(), b.src.as<int32_t>(), s);
    for (size_t c = 0; c + 1 < cuts.size(); ++c) {
        for (int e0 = cuts[c]; e0 < cuts[c + 1]; e0 += fused::kMaxBatch) {
            fused::Batch bt;
            bt.e0 = e0;
            bt.e1 = std::min(cuts[c + 1], e0 + fused::kMaxBatch);
            for (int e = bt.e0; e < bt.e1; ++e) bt.blob[e - bt.e0] = blob[(size_t) e];
            fused::experts_native(bt, ng, E, rows, b.scratch.p, b.xa.p, b.src.as<int32_t>(), b.ha.p, b.dm.as<float>(), s);
        }
    }
}

struct Err {
    double rms = 0, worst = 0;   // RMS error / RMS of the reference; the worst row's max |error| / its max |ref|
};
// y_a[row_a(i)] against y_b[row_b(i)] for every pair i (rows of N values)
template <typename RA, typename RB>
Err compare(const std::vector<float>& a, RA row_a, const std::vector<float>& b, RB row_b, size_t pairs) {
    double e2 = 0, r2 = 0;
    Err out;
    for (size_t i = 0; i < pairs; ++i) {
        const float* ya = a.data() + (size_t) row_a(i) * N;
        const float* yb = b.data() + (size_t) row_b(i) * N;
        double me = 0, mr = 0;
        for (int o = 0; o < N; ++o) {
            if (!std::isfinite(ya[o])) throw std::runtime_error("a non-finite output");
            const double d = (double) ya[o] - yb[o];
            e2 += d * d;
            r2 += (double) yb[o] * yb[o];
            me = std::max(me, std::fabs(d));
            mr = std::max(mr, (double) std::fabs(yb[o]));
        }
        if (mr > 0) out.worst = std::max(out.worst, me / mr);
        else if (me > 0) out.worst = 1e30;   // a zero row must stay exactly zero
    }
    out.rms = r2 > 0 ? std::sqrt(e2 / r2) : 0;
    return out;
}

// the kernels' int8 rounding per 32 values (quant_act_nat_kernel / the H epilogue), as floats: codes * (amax / 127)
void quant32(const float* in, float* out, int n) {
    for (int b = 0; b < n; b += 32) {
        float am = 0.0f;
        for (int j = 0; j < 32; ++j) am = std::max(am, std::fabs(in[b + j]));
        const float inv = am > 0.0f ? 127.0f / am : 0.0f, d = am / 127.0f;
        for (int j = 0; j < 32; ++j) out[b + j] = (float) std::nearbyint(in[b + j] * inv) * d;
    }
}

std::vector<float> download(const Dev& d, size_t n) {
    std::vector<float> h(n);
    ck(cudaMemcpy(h.data(), d.p, n * 4, cudaMemcpyDeviceToHost), "download");
    return h;
}
std::vector<int32_t> download_i(const Dev& d, size_t n) {
    std::vector<int32_t> h(n);
    ck(cudaMemcpy(h.data(), d.p, n * 4, cudaMemcpyDeviceToHost), "download");
    return h;
}

// part 1: against the double-precision reference
void reference_part(const Pair& p, cudaStream_t s) {
    constexpr int T = 256, E = 64, ZERO = 77;
    const Geo geo(p);
    std::mt19937 rng(136 + (unsigned) p.gu * 7 + (unsigned) p.d);
    std::vector<std::vector<uint8_t>> host((size_t) E);
    std::vector<std::unique_ptr<Dev>> dev;
    std::vector<const uint8_t*> blob((size_t) E);
    for (int e = 0; e < E; ++e) {
        host[(size_t) e] = make_blob(p, geo, rng);
        dev.push_back(std::make_unique<Dev>(geo.bytes + 4096 * (size_t) (e % 3) + 64));   // unrelated addresses
        uint8_t* q = dev.back()->as<uint8_t>() + 256 * (size_t) (e % 3) + 2 * (size_t) (e % 2);   // 2-byte aligned
        ck(cudaMemcpy(q, host[(size_t) e].data(), geo.bytes, cudaMemcpyHostToDevice), "blob");
        blob[(size_t) e] = q;
    }
    const std::vector<float> x = make_x(T, rng, ZERO);
    const std::vector<int32_t> ids = make_ids(T, E, rng, 2, 4);   // experts 0, 1 hot (several tiles), 60-63 unused
    const int64_t rows = (int64_t) T * K;
    const Routing r = sort_rows(ids, E);
    std::printf("%s - reference part: %d tokens, %d experts, rows per expert: max %d, experts without rows %d\n", p.name,
                T, E, *std::max_element(r.cnt.begin(), r.cnt.end()), (int) std::count(r.cnt.begin(), r.cnt.end(), 0));

    Dev x_dev(x.size() * 4), src_dev((size_t) rows * 4);
    ck(cudaMemcpy(x_dev.p, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
    ck(cudaMemcpy(src_dev.p, r.src.data(), r.src.size() * 4, cudaMemcpyHostToDevice), "src");
    // (the uploads above are on the legacy stream and the work below on a non-blocking one: a cudaMemcpy from pageable
    // memory may return before its DMA lands - on Windows it can wait in the queue until the next synchronous call)
    ck(cudaDeviceSynchronize(), "uploads");
    const bool mq = mmq_ok(p);
    mmq::Context ctx;
    MmqBufs* mbp = mq ? new MmqBufs(p, rows, E) : nullptr;
    if (mq) run_mmq(p, geo, ctx, *mbp, r, x_dev.as<float>(), src_dev.as<int32_t>(), blob, rows, s);
    FusedBufs fb(T, rows, E);
    ck(cudaMemcpy(fb.ids.p, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice), "ids");
    ck(cudaMemset(fb.dm.p, 0xff, (size_t) rows * N * 4), "sentinel");   // an unwritten row is NaN
    ck(cudaDeviceSynchronize(), "uploads");
    run_fused(p, geo, fb, x_dev.as<float>(), T, E, {0, 1, 20, E}, blob, s);
    ck(cudaStreamSynchronize(s), "sync");
    const std::vector<float> y_mmq = mq ? download(mbp->dm, (size_t) rows * N) : std::vector<float>((size_t) rows * N, 0.0f),
                             y_f = download(fb.dm, (size_t) rows * N);
    delete mbp;
    const std::vector<int32_t> slot = download_i(fb.slot, (size_t) rows), fsrc = download_i(fb.src, (size_t) rows);
    for (int64_t i = 0; i < rows; ++i)
        if (fsrc[(size_t) slot[(size_t) i]] != (int32_t) (i / K)) throw std::runtime_error("group: slot/src disagree");

    // the reference, an expert at a time on all cores
    std::vector<float> ref((size_t) rows * N), ref2((size_t) rows * N);   // ref2: the same with the kernels' int8 rounding
    std::vector<std::thread> th;
    const int nth = std::max(1, std::min(16, (int) std::thread::hardware_concurrency()));
    for (int w = 0; w < nth; ++w)
        th.emplace_back([&, w] {
            std::vector<float> gu, dn, h(FF), h2(FF), hq(FF), xq(N);
            for (int e = w; e < E; e += nth) {
                if (r.cnt[(size_t) e] == 0) continue;
                dequant(p, geo, host[(size_t) e], gu, dn);
                for (int64_t i = 0; i < rows; ++i) {
                    if (ids[(size_t) i] != e) continue;
                    const float* xr = x.data() + (size_t) (i / K) * N;
                    for (int f = 0; f < FF; ++f) {
                        double gt = 0, up = 0;
                        const float *wg = gu.data() + (size_t) f * N, *wu = gu.data() + (size_t) (FF + f) * N;
                        for (int k = 0; k < N; ++k) { gt += (double) wg[k] * xr[k]; up += (double) wu[k] * xr[k]; }
                        h[(size_t) f] = (float) (gt / (1.0 + std::exp(-gt)) * up);
                    }
                    for (int o = 0; o < N; ++o) {
                        double a = 0;
                        const float* wd = dn.data() + (size_t) o * FF;
                        for (int f = 0; f < FF; ++f) a += (double) wd[f] * h[(size_t) f];
                        ref[(size_t) i * N + o] = (float) a;
                    }
                    quant32(xr, xq.data(), N);
                    for (int f = 0; f < FF; ++f) {
                        double gt = 0, up = 0;
                        const float *wg = gu.data() + (size_t) f * N, *wu = gu.data() + (size_t) (FF + f) * N;
                        for (int k = 0; k < N; ++k) { gt += (double) wg[k] * xq[(size_t) k]; up += (double) wu[k] * xq[(size_t) k]; }
                        h2[(size_t) f] = (float) (gt / (1.0 + std::exp(-gt)) * up);
                    }
                    quant32(h2.data(), hq.data(), FF);
                    for (int o = 0; o < N; ++o) {
                        double a = 0;
                        const float* wd = dn.data() + (size_t) o * FF;
                        for (int f = 0; f < FF; ++f) a += (double) wd[f] * hq[(size_t) f];
                        ref2[(size_t) i * N + o] = (float) a;
                    }
                }
            }
        });
    for (auto& t : th) t.join();

    auto pair = [](size_t i) { return (int64_t) i; };
    auto mrow = [&](size_t i) { return (int64_t) r.row_of[i]; };
    auto frow = [&](size_t i) { return (int64_t) slot[i]; };
    const Err em = mq ? compare(y_mmq, mrow, ref, pair, (size_t) rows) : Err{};
    const Err ef = compare(y_f, frow, ref, pair, (size_t) rows);
    const Err efm = mq ? compare(y_f, frow, y_mmq, mrow, (size_t) rows) : Err{};
    const Err e2 = compare(y_f, frow, ref2, pair, (size_t) rows);
    std::printf("  fused vs the int8-rounded model (same activation / H rounding, double sums): rel RMS %.3e  worst %.3e\n",
                e2.rms, e2.worst);
    if (e2.rms > 2e-3 || e2.worst > 3e-2)
        throw std::runtime_error(std::string(p.name) + ": the fused path differs from its own arithmetic's model");
    if (mq) std::printf("  MMQ   vs FP32 reference: rel RMS %.3e  worst row max rel %.3e\n", em.rms, em.worst);
    std::printf("  fused vs FP32 reference: rel RMS %.3e  worst row max rel %.3e\n", ef.rms, ef.worst);
    if (mq) std::printf("  fused vs MMQ           : rel RMS %.3e  worst row max rel %.3e\n", efm.rms, efm.worst);
    double zmax = 0;
    for (int k = 0; k < K; ++k)
        for (int o = 0; o < N; ++o)
            zmax = std::max(zmax, (double) std::fabs(y_f[(size_t) slot[(size_t) (ZERO * K + k)] * N + o]));
    if (zmax != 0) throw std::runtime_error(std::string(p.name) + ": the all-zero token's outputs are not zero");
    if (!mq) {   // no MMQ here: the int8 activations' own error (the IQ pairs' fused path sits at ~1e-2 rel RMS)
        if (ef.rms > 0.03 || ef.worst > 0.25)
            throw std::runtime_error(std::string(p.name) + ": the fused path's error against the reference is too large");
        return;
    }
    if (ef.rms > 1.5 * em.rms || ef.worst > 2.0 * em.worst)
        throw std::runtime_error(std::string(p.name) + ": the fused path's error is not comparable to MMQ's");
}

// part 2: one layer at a real chunk, timed
void timing_part(const Pair& p, int T, cudaStream_t s) {
    constexpr int E = 512;
    const Geo geo(p);
    std::mt19937 rng(1360);
    const int64_t rows = (int64_t) T * K;
    // 512 distinct blobs - the products read each once per tile, as the engine does
    // at a stride of whole blocks of both formats (MMQ's expert stride counts blocks), + MMQ's read past the end
    size_t unit = ggml_type_size(p.gu);
    while (unit % ggml_type_size(p.d) != 0 || unit % 16 != 0) unit += ggml_type_size(p.gu);
    const size_t stride = (geo.bytes + unit - 1) / unit * unit;
    Dev blobs((size_t) E * stride + 4096);
    ck(cudaMemset(blobs.p, 0, (size_t) E * stride + 4096), "zero");
    {
        std::vector<uint8_t> b = make_blob(p, geo, rng);
        for (int e = 0; e < E; ++e) {
            b[4 + (size_t) (e % 16)] ^= (uint8_t) e;   // not all equal (inside the first block, not its scale)
            ck(cudaMemcpy(blobs.as<uint8_t>() + (size_t) e * stride, b.data(), geo.bytes, cudaMemcpyHostToDevice),
               "blob");
        }
    }
    std::vector<const uint8_t*> blob((size_t) E);
    for (int e = 0; e < E; ++e) blob[(size_t) e] = blobs.as<uint8_t>() + (size_t) e * stride;
    const std::vector<float> x = make_x(T, rng, -1);
    const std::vector<int32_t> ids = make_ids(T, E, rng, 0, 0);
    const Routing r = sort_rows(ids, E);
    Dev x_dev(x.size() * 4), src_dev((size_t) rows * 4);
    ck(cudaMemcpy(x_dev.p, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
    ck(cudaMemcpy(src_dev.p, r.src.data(), r.src.size() * 4, cudaMemcpyHostToDevice), "src");
    const bool mq = mmq_ok(p);
    mmq::Context ctx;
    MmqBufs* mbp = mq ? new MmqBufs(p, rows, E) : nullptr;
    MmqBufs& mb = *mbp;   // (only used when mq)
    FusedBufs fb(T, rows, E);
    ck(cudaMemcpy(fb.ids.p, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice), "ids");
    ck(cudaDeviceSynchronize(), "uploads");   // (see reference_part)
    cudaEvent_t a, b;
    ck(cudaEventCreate(&a), "event");
    ck(cudaEventCreate(&b), "event");
    auto once = [&](auto&& f) {
        ck(cudaEventRecord(a, s), "record");
        f();
        ck(cudaEventRecord(b, s), "record");
        ck(cudaEventSynchronize(b), "sync");
        float t = 0;
        ck(cudaEventElapsedTime(&t, a, b), "elapsed");
        return t;
    };
    auto mmq_run = [&] { run_mmq(p, geo, ctx, mb, r, x_dev.as<float>(), src_dev.as<int32_t>(), blob, rows, s); };
    auto fused_run = [&] { run_fused(p, geo, fb, x_dev.as<float>(), T, E, {0, E}, blob, s); };
    const bool all_routed = mq && std::count(r.cnt.begin(), r.cnt.end(), 0) == 0;
    auto direct_run = [&] {
        run_mmq(p, geo, ctx, mb, r, x_dev.as<float>(), src_dev.as<int32_t>(), blob, rows, s, true, stride);
    };
    if (std::getenv("S20_IDSDIAG")) {   // which step overwrites the routing ids on the device
        auto check = [&](const char* after) {
            ck(cudaStreamSynchronize(s), "sync");
            const std::vector<int32_t> d = download_i(fb.ids, (size_t) rows);
            int64_t bad = 0;
            for (size_t i = 0; i < (size_t) rows; ++i) bad += d[i] != ids[i];
            std::printf("    ids on the device after %s: %lld of %lld differ (fb.ids %p, mb.dm %p..%p, fb.dm %p..%p)\n",
                        after, (long long) bad, (long long) rows, fb.ids.p, mb.dm.p,
                        (void*) (mb.dm.as<uint8_t>() + (size_t) rows * N * 4), fb.dm.p,
                        (void*) (fb.dm.as<uint8_t>() + (size_t) rows * N * 4));
        };
        check("the upload");
        mmq_run();
        check("an MMQ run");
        fused_run();
        check("a fused run");
        if (all_routed) {
            direct_run();
            check("an MMQ run without gathers");
        }
    }
    // ~1 s of both first (the clocks ramp up), then alternating rounds; the medians
    const bool no_mmq = !mq || std::getenv("S20_NOMMQ") != nullptr;   // (debug: the fused runs alone)
    for (float spent = 0; spent < 1000.0f;) spent += (no_mmq ? 0.0f : once(mmq_run)) + once(fused_run);
    std::vector<float> tm, tf, td;
    for (int rep = 0; rep < 8; ++rep) {
        tm.push_back(no_mmq ? 1.0f : once(mmq_run));
        tf.push_back(once(fused_run));
        if (all_routed && !no_mmq) td.push_back(once(direct_run));
    }
    if (td.empty()) td.push_back(0.0f);
    std::sort(tm.begin(), tm.end());
    std::sort(tf.begin(), tf.end());
    std::sort(td.begin(), td.end());
    const float t_mmq = tm[tm.size() / 2], t_f = tf[tf.size() / 2];
    if (all_routed)
        std::printf("  (MMQ's products alone, no gathers: %.3f ms - fused %.2fx)\n", td[td.size() / 2],
                    td[td.size() / 2] / t_f);
    std::vector<float> tf2;   // back to back
    for (int rep = 0; rep < 8; ++rep) tf2.push_back(once(fused_run));
    std::sort(tf2.begin(), tf2.end());
    std::printf("  (fused back to back: median %.3f ms, min %.3f; alternating min %.3f; MMQ min %.3f)\n", tf2[4],
                tf2[0], tf[0], tm[0]);
    const std::vector<float> y_mmq = mq ? download(mb.dm, (size_t) rows * N) : std::vector<float>((size_t) rows * N, 0.0f),
                             y_f = download(fb.dm, (size_t) rows * N);
    const std::vector<int32_t> slot = download_i(fb.slot, (size_t) rows);
    {   // (Aurora S23) a hash of the fused output's bits, to compare two builds bit for bit
        uint64_t hsh = 1469598103934665603ull;
        for (size_t i = 0; i < (size_t) rows; ++i)   // in pair order: the rows of an expert are placed in any order
            for (size_t c = 0; c < (size_t) N; ++c) { uint32_t u; std::memcpy(&u, &y_f[(size_t) slot[i] * N + c], 4); hsh = (hsh ^ u) * 1099511628211ull; }
        std::printf("  fused output bits hash %016llx\n", (unsigned long long) hsh);
    }
    const Err efm = mq ? compare(y_f, [&](size_t i) { return (int64_t) slot[i]; }, y_mmq,
                                 [&](size_t i) { return (int64_t) r.row_of[i]; }, (size_t) rows) : Err{};
    std::printf("%s - timing part: %d tokens x top %d over %d experts, one layer: MMQ path %.3f ms, fused %.3f ms "
                "(%.2fx); fused vs MMQ rel RMS %.3e worst row %.3e\n",
                p.name, T, K, E, t_mmq, t_f, t_mmq / t_f, efm.rms, efm.worst);
    {   // the grouping: every pair in its expert's rows (the near-identical blobs here would hide a wrong expert)
        const std::vector<int32_t> fsrc = download_i(fb.src, (size_t) rows);
        int64_t bad = 0, bad_src = 0;
        for (size_t i = 0; i < (size_t) rows; ++i) {
            const int e = ids[i];
            bad += slot[i] < r.off[(size_t) e] || slot[i] >= r.off[(size_t) e + 1];
            bad_src += fsrc[(size_t) slot[i]] != (int32_t) (i / K);
        }
        if (bad || bad_src) {
            std::vector<int32_t> tbl((size_t) 4 * E + 2);
            ck(cudaMemcpy(tbl.data(), fb.scratch.p, tbl.size() * 4, cudaMemcpyDeviceToHost), "tables");
            int64_t bad_cnt = 0, bad_off = 0, bad_ids = 0;
            const std::vector<int32_t> d_ids = download_i(fb.ids, (size_t) rows);
            for (size_t i = 0; i < (size_t) rows; ++i) bad_ids += d_ids[i] != ids[i];
            std::printf("  the routing ids on the device: %lld of %lld differ (first %d %d %d)\n", (long long) bad_ids,
                        (long long) rows, d_ids[0], d_ids[1], d_ids[2]);
            const int32_t smin = *std::min_element(slot.begin(), slot.end()), smax = *std::max_element(slot.begin(), slot.end());
            std::printf("  slots %d..%d; off[0..3] %d %d %d %d off[E] %d; fill[0..3] %d %d %d %d; ts[0..3] %d %d %d %d "
                        "ts[E] %d\n", smin, smax, tbl[E], tbl[E + 1], tbl[E + 2], tbl[E + 3], tbl[2 * E], tbl[2 * E + 1],
                        tbl[2 * E + 2], tbl[2 * E + 3], tbl[2 * E + 4], tbl[3 * E + 1], tbl[3 * E + 2], tbl[3 * E + 3],
                        tbl[3 * E + 4], tbl[4 * E + 1]);
            // the group again on these ids, synchronously
            fused::group(fb.ids.as<int32_t>(), rows, K, E, fb.scratch.p, fb.slot.as<int32_t>(), fb.src.as<int32_t>(), s);
            ck(cudaStreamSynchronize(s), "sync");
            std::vector<int32_t> t2((size_t) 4 * E + 2);
            ck(cudaMemcpy(t2.data(), fb.scratch.p, t2.size() * 4, cudaMemcpyDeviceToHost), "tables");
            std::printf("  group() again, synchronized: cnt[0..3] %d %d %d %d off[E] %d\n", t2[0], t2[1], t2[2], t2[3],
                        t2[(size_t) 2 * E]);
            for (int e = 0; e < E; ++e) {
                bad_cnt += tbl[(size_t) e] != r.cnt[(size_t) e];
                bad_off += tbl[(size_t) E + e] != r.off[(size_t) e];
            }
            std::printf("  GROUPING WRONG: %lld pairs outside their expert's rows, %lld src mismatches; tables: %lld counts "
                        "and %lld offsets differ from the host's (cnt[0..3] %d %d %d %d vs %d %d %d %d)\n",
                        (long long) bad, (long long) bad_src, (long long) bad_cnt, (long long) bad_off, tbl[0], tbl[1],
                        tbl[2], tbl[3], r.cnt[0], r.cnt[1], r.cnt[2], r.cnt[3]);
            throw std::runtime_error(std::string(p.name) + ": the grouping is wrong");
        }
    }
    if (std::getenv("S20_ROWDIAG")) {   // the pairs whose rows differ most: expert, its rows, the row's place
        const std::vector<int32_t> fsrc = download_i(fb.src, (size_t) rows);
        std::vector<std::pair<double, size_t>> worst;
        for (size_t i = 0; i < (size_t) rows; ++i) {
            const float* a = y_f.data() + (size_t) slot[i] * N;
            const float* m = y_mmq.data() + (size_t) r.row_of[i] * N;
            double me = 0, mr = 0;
            for (int o = 0; o < N; ++o) { me = std::max(me, (double) std::fabs(a[o] - m[o])); mr = std::max(mr, (double) std::fabs(m[o])); }
            worst.push_back({mr > 0 ? me / mr : me, i});
        }
        std::sort(worst.rbegin(), worst.rend());
        for (int k = 0; k < 12; ++k) {
            const size_t i = worst[(size_t) k].second;
            const int e = ids[i];
            // the expert's rows start at its first fused row: the smallest slot of its pairs
            int32_t first = INT32_MAX;
            for (size_t j = 0; j < (size_t) rows; ++j) if (ids[j] == e) first = std::min(first, slot[j]);
            std::printf("    pair %zu token %zu expert %d (%d rows) fused row %d (+%d in the expert) max rel %.3e\n", i,
                        i / K, e, r.cnt[(size_t) e], slot[i], slot[i] - first, worst[(size_t) k].first);
        }
        (void) fsrc;
    }
    // both against the double-precision reference on a sample of 96 pairs
    {
        std::vector<size_t> pick;
        for (size_t i = 0; i < 96; ++i) pick.push_back((i * 2654435761u) % (size_t) rows);
        std::vector<float> ref(pick.size() * N), ym(pick.size() * N), yf(pick.size() * N);
        std::vector<uint8_t> hb(geo.bytes);
        std::vector<float> gu, dn, h(FF);
        for (size_t j = 0; j < pick.size(); ++j) {
            const size_t i = pick[j];
            const int e = ids[i];
            ck(cudaMemcpy(hb.data(), blob[(size_t) e], geo.bytes, cudaMemcpyDeviceToHost), "blob back");
            dequant(p, geo, hb, gu, dn);
            const float* xr = x.data() + (i / K) * N;
            for (int f = 0; f < FF; ++f) {
                double gt = 0, up = 0;
                const float *wg = gu.data() + (size_t) f * N, *wu = gu.data() + (size_t) (FF + f) * N;
                for (int k = 0; k < N; ++k) { gt += (double) wg[k] * xr[k]; up += (double) wu[k] * xr[k]; }
                h[(size_t) f] = (float) (gt / (1.0 + std::exp(-gt)) * up);
            }
            for (int o = 0; o < N; ++o) {
                double a = 0;
                const float* wd = dn.data() + (size_t) o * FF;
                for (int f = 0; f < FF; ++f) a += (double) wd[f] * h[(size_t) f];
                ref[j * N + o] = (float) a;
            }
            if (mq) std::copy_n(y_mmq.begin() + (ptrdiff_t) ((size_t) r.row_of[i] * N), N, ym.begin() + (ptrdiff_t) (j * N));
            std::copy_n(y_f.begin() + (ptrdiff_t) ((size_t) slot[i] * N), N, yf.begin() + (ptrdiff_t) (j * N));
        }
        auto id = [](size_t j) { return (int64_t) j; };
        const Err em = mq ? compare(ym, id, ref, id, pick.size()) : Err{}, ef = compare(yf, id, ref, id, pick.size());
        std::printf("  sample of %zu pairs vs FP32 reference: MMQ rel RMS %.3e worst %.3e, fused rel RMS %.3e worst "
                    "%.3e\n", pick.size(), em.rms, em.worst, ef.rms, ef.worst);
        if (!mq ? (ef.rms > 0.03 || ef.worst > 0.25) : (ef.rms > 1.5 * em.rms || ef.worst > 2.0 * em.worst))
            throw std::runtime_error(std::string(p.name) + ": the fused path's error is not comparable to MMQ's (timing)");
    }
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    delete mbp;
    if (efm.rms > 0.05) throw std::runtime_error(std::string(p.name) + ": the paths disagree at the real shape");
}

// ---- --mmq-only: the MMQ expert path alone, on any sm_70+ device (the fused comparison needs sm_80) ----

// Independent D4 layout decoder and quantizer check. The pinned quantize.cu stores
// four FP32 scales followed by 128 signed codes; blocks are transposed over rows.
// CUDA 12.4 fast division's <=2 ULP error plus multiplication rounding is covered
// by the 8u multiplier interval and 12u two-reciprocal scale interval (u=2^-24).
// Half-away endpoint rounding admits only codes possible inside that interval.
std::vector<double> checked_d4(const Dev& device, const std::vector<float>& input,
                               const std::vector<int32_t>& sources, int cols, int rows) {
    const int padded = (cols + 511) / 512 * 512;
    std::vector<uint8_t> raw((size_t) rows * padded / 128 * 144);
    ck(cudaMemcpy(raw.data(), device.p, raw.size(), cudaMemcpyDeviceToHost), "D4 download");
    std::vector<double> decoded((size_t) rows * cols);
    for (int row = 0; row < rows; ++row) {
        const int src = sources.empty() ? row : sources[(size_t) row];
        for (int b = 0; b < padded; b += 32) {
            double am = 0;
            for (int j = 0; j < 32 && b + j < cols; ++j) {
                const double x = input[(size_t) src * cols + b + j];
                normal_or_zero(x, "quantizer input");
                am = std::max(am, std::fabs(x));
            }
            const size_t offset = ((size_t) (b / 128) * rows + row) * 144;
            float scale;
            std::memcpy(&scale, raw.data() + offset + (size_t) ((b % 128) / 32) * 4, 4);
            normal_or_zero(scale, "quantizer scale");
            const double ideal_scale = am / 127.0;
            if (am > 0) {
                normal_or_zero(ideal_scale, "reference scale");
                normal_or_zero(127.0 / am, "reference reciprocal");
            }
            if (scale < 0 || std::fabs((double) scale - ideal_scale) > 12 * kUnitRoundoff * ideal_scale)
                throw std::runtime_error("D4 scale differs from independent quantizer");
            for (int j = 0; j < 32; ++j) {
                int8_t code;
                std::memcpy(&code, raw.data() + offset + 16 + b % 128 + j, 1);
                const double x = b + j < cols ? input[(size_t) src * cols + b + j] : 0.0;
                const double ideal = am > 0 ? x * (127.0 / am) : 0;
                const double radius = 8 * kUnitRoundoff * std::fabs(ideal);
                const int low = (int) std::round(ideal - radius), high = (int) std::round(ideal + radius);
                if (code < -127 || code < low || code > high || (am == 0 && code != 0))
                    throw std::runtime_error("D4 code differs from half-away quantizer interval");
                if (b + j < cols) {
                    const double value = (double) code * scale;
                    normal_or_zero(value, "decoded activation");
                    decoded[(size_t) row * cols + b + j] = value;
                }
            }
        }
    }
    return decoded;
}

// At most four FP32 roundings per scalar contribution cover scale formation,
// multiplication, accumulation and stream-K fixups; integer partial sums are exact.
// Use the full scalar K, not block count. The absolute term-mass bound remains
// meaningful under cancellation. Host summation/products use FP64.
double checked_dot(const float* weights, const double* input, int count, float got, const char* stage) {
    double sum = 0, mass = 0;
    for (int k = 0; k < count; ++k) {
        normal_or_zero(weights[k], "decoded weight");
        const double product = (double) weights[k] * input[k];
        normal_or_zero(product, "reference dot product");
        sum += product;
        mass += std::fabs(product);
    }
    normal_or_zero(sum, "reference dot output");
    normal_or_zero(got, stage);
    const double nu = (4 * count + 16) * kUnitRoundoff;
    const double bound = nu / (1 - nu) * mass;
    const double error = std::fabs((double) got - sum);
    if (!(error <= bound)) {
        std::printf("%s: got %.9g reference %.17g abs error %.3e bound %.3e\n", stage, got, sum, error, bound);
        throw std::runtime_error(std::string(stage) + ": term-mass bound exceeded");
    }
    return bound > 0 ? error / bound : 0;
}


// the compact routing: K distinct experts per token, fully deterministic; experts 12..15 unrouted (equal
// bounds), expert 11 exactly one row (token 0 alone reaches it), the rest several; tokens 2..5 route randomly
// over 0..10
std::vector<int32_t> make_ids_small(int T, int E, std::mt19937& rng) {
    std::vector<int32_t> ids((size_t) T * K);
    std::uniform_int_distribution<int> any(0, E - 6);
    for (int t = 0; t < T; ++t)
        for (int k = 0; k < K; ++k) {
            int e;
            bool dup;
            do {
                e = t == 0 ? 2 + k : t == 1 ? k : any(rng);
                dup = false;
                for (int j = 0; j < k; ++j) dup |= ids[(size_t) t * K + j] == e;
            } while (dup);
            ids[(size_t) t * K + k] = e;
        }
    return ids;
}

struct SwigluCensus {
    std::unique_ptr<FILE, decltype(&std::fclose)> file{nullptr, &std::fclose};
    size_t tuples = 0, refused = 0;
    explicit SwigluCensus(const char* path) {
        if (!path) return;
        file.reset(std::fopen(path, "wx"));
        if (!file) throw std::runtime_error("Cannot create new SwiGLU census");
        if (std::fprintf(file.get(), "gate_type,down_type,expert,row,feature,g_bits,up_bits,h_bits,reference_domain\n") < 0)
            throw std::runtime_error("Cannot write SwiGLU census header");
    }
    void record(const Pair& p, int expert, int row, int feature, float g, float up, float h,
                const char* domain) {
        uint32_t gb, ub, hb;
        std::memcpy(&gb, &g, sizeof(gb));
        std::memcpy(&ub, &up, sizeof(ub));
        std::memcpy(&hb, &h, sizeof(hb));
        if (std::fprintf(file.get(), "%d,%d,%d,%d,%d,%08x,%08x,%08x,%s\n",
                         (int) p.gu, (int) p.d, expert, row, feature, gb, ub, hb, domain) < 0)
            throw std::runtime_error("Cannot write SwiGLU census row");
        ++tuples;
    }
};

void mmq_only_part(const Pair& p, cudaStream_t s, SwigluCensus& census) {
    constexpr int T = 6, E = 16, ZERO = 0;
    const Geo geo(p);
    std::mt19937 rng(236 + (unsigned) p.gu * 7 + (unsigned) p.d);
    std::vector<std::vector<uint8_t>> host((size_t) E);
    std::vector<std::unique_ptr<Dev>> dev;
    std::vector<const uint8_t*> blob((size_t) E);
    for (int e = 0; e < E; ++e) {
        host[(size_t) e] = make_blob(p, geo, rng);
        dev.push_back(std::make_unique<Dev>(geo.bytes + 4096 * (size_t) (e % 3) + 64));
        uint8_t* q = dev.back()->as<uint8_t>() + 256 * (size_t) (e % 3) + 2 * (size_t) (e % 2);   // 2-byte aligned
        ck(cudaMemcpy(q, host[(size_t) e].data(), geo.bytes, cudaMemcpyHostToDevice), "blob");
        blob[(size_t) e] = q;
    }
    const std::vector<float> x = make_x(T, rng, ZERO);   // outliers + the all-zero token
    const std::vector<int32_t> ids = make_ids_small(T, E, rng);
    const int64_t rows = (int64_t) T * K;
    const Routing r = sort_rows(ids, E);
    std::printf("%s - mmq-only: %d tokens, top %d over %d experts, %lld rows; rows per expert: max %d, experts "
                "without rows %d, with one row %d\n", p.name, T, K, E, (long long) rows,
                *std::max_element(r.cnt.begin(), r.cnt.end()), (int) std::count(r.cnt.begin(), r.cnt.end(), 0),
                (int) std::count(r.cnt.begin(), r.cnt.end(), 1));

    Dev x_dev(x.size() * 4), src_dev((size_t) rows * 4);
    ck(cudaMemcpy(x_dev.p, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
    ck(cudaMemcpy(src_dev.p, r.src.data(), r.src.size() * 4, cudaMemcpyHostToDevice), "src");
    ck(cudaDeviceSynchronize(), "uploads");
    mmq::Context ctx;
    MmqBufs mb(p, rows, E);
    ck(cudaMemset(mb.dm.p, 0xff, (size_t) rows * N * 4), "sentinel");   // an unwritten row is NaN
    run_mmq(p, geo, ctx, mb, r, x_dev.as<float>(), src_dev.as<int32_t>(), blob, rows, s);
    ck(cudaStreamSynchronize(s), "sync");
    const std::vector<float> y = download(mb.dm, (size_t) rows * N);

    // With E=16 this fixture has one gathered group, so H's packed row base is
    // zero. Check every boundary independently before consuming it downstream.
    const auto xq = checked_d4(mb.xq, x, r.src, N, (int) rows);
    const auto got_gu = download(mb.gu, (size_t) rows * 2 * FF);
    const auto got_h = download(mb.h, (size_t) rows * FF);
    double max_gu = 0, max_epilogue = 0, max_down = 0;
    std::vector<float> gu, dn;
    for (int e = 0; e < E; ++e) {
        if (r.cnt[(size_t) e] == 0) continue;
        dequant(p, geo, host[(size_t) e], gu, dn);
        for (int row = r.off[(size_t) e]; row < r.off[(size_t) e] + r.cnt[(size_t) e]; ++row) {
            for (int f = 0; f < 2 * FF; ++f)
                max_gu = std::max(max_gu, checked_dot(gu.data() + (size_t) f * N,
                    xq.data() + (size_t) row * N, N, got_gu[(size_t) row * 2 * FF + f], "gate/up"));
            for (int f = 0; f < FF; ++f) {
                const float g = got_gu[(size_t) row * 2 * FF + f];
                const float up = got_gu[(size_t) row * 2 * FF + FF + f];
                const float h = got_h[(size_t) row * FF + f];
                try {
                    max_epilogue = std::max(max_epilogue, checked_swiglu(g, up, h));
                    if (census.file) census.record(p, e, row, f, g, up, h, "supported");
                } catch (const SwigluReferenceDomain& ex) {
                    if (!census.file) throw;
                    // Diagnostic refusal, never a pass. All other checks remain fatal.
                    census.record(p, e, row, f, g, up, h, ex.what());
                    if (census.refused++ == 0)
                        std::fprintf(stderr, "First original reference refusal: %s expert=%d row=%d feature=%d g=%a up=%a h=%a: %s\n",
                                     p.name, e, row, f, (double) g, (double) up, (double) h, ex.what());
                }
            }
        }
    }
    const auto hq = checked_d4(mb.hq, got_h, {}, FF, (int) rows);
    for (int e = 0; e < E; ++e) {
        if (r.cnt[(size_t) e] == 0) continue;
        dequant(p, geo, host[(size_t) e], gu, dn);
        for (int row = r.off[(size_t) e]; row < r.off[(size_t) e] + r.cnt[(size_t) e]; ++row)
            for (int o = 0; o < N; ++o)
                max_down = std::max(max_down, checked_dot(dn.data() + (size_t) o * FF,
                    hq.data() + (size_t) row * FF, FF, y[(size_t) row * N + o], "down"));
    }
    std::printf("  independent quantizers exact/interval-valid; supported-domain error/bound: gate/up %.3e SwiGLU %.3e down %.3e\n",
                max_gu, max_epilogue, max_down);
    for (int k = 0; k < K; ++k) {
        const float* row = y.data() + (size_t) r.row_of[(size_t) ZERO * K + k] * N;
        for (int o = 0; o < N; ++o)
            if (row[o] != 0.0f) throw std::runtime_error(std::string(p.name) + ": the all-zero token's MMQ outputs are not zero");
    }
}

// --mmq-only's main: no fused kernel is touched (below sm_80 there is none - that is the point), the pairs run
// wherever this build's MMQ covers them and the GPU has a tile (mmq::fits, the engine's own gate).  Returns the
// process exit code.
int mmq_only_main(int argc, char** argv) {
    if (!strata::platform::acknowledge_protected_test()) return 2;
    // These weight formats all use D4 activations in the pinned ggml MMQ.
    try {
        cudaDeviceProp prop{};
        int dev = 0;
        ck(cudaGetDevice(&dev), "device");
        ck(cudaGetDeviceProperties(&prop, dev), "props");
        std::printf("prefill fused MoE (native formats) parity --mmq-only: %s sm_%d%d (%d SMs)\n", prop.name,
                    prop.major, prop.minor, prop.multiProcessorCount);
        const char* only = nullptr;
        bool pack_pairs = false;
        const char* census_path = nullptr;
        for (int i = 1; i < argc; ++i) {
            if (std::strncmp(argv[i], "--only=", 7) == 0) only = argv[i] + 7;
            if (std::strcmp(argv[i], "--pack-pairs") == 0) pack_pairs = true;
            if (std::string_view(argv[i]).starts_with("--swiglu-census="))
                census_path = argv[i] + sizeof("--swiglu-census=") - 1;
        }
        if (pack_pairs && only) throw std::runtime_error("--pack-pairs must run the complete seven-pair inventory");
        if (census_path && !pack_pairs) throw std::runtime_error("SwiGLU census requires unchanged complete --pack-pairs fixture");
        SwigluCensus census(census_path);
        constexpr Pair single[] = {{GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_NL, "IQ3_S / IQ4_NL"}};
        constexpr Pair mixed[] = {
            {GGML_TYPE_IQ2_S, GGML_TYPE_Q2_0, "IQ2_S / Q2_0"},
            {GGML_TYPE_IQ2_S, GGML_TYPE_IQ4_NL, "IQ2_S / IQ4_NL"},
            {GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_NL, "IQ3_XXS / IQ4_NL"},
            {GGML_TYPE_IQ3_XXS, GGML_TYPE_Q2_0, "IQ3_XXS / Q2_0"},
            {GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_NL, "IQ3_S / IQ4_NL"},
            {GGML_TYPE_IQ3_S, GGML_TYPE_Q2_0, "IQ3_S / Q2_0"},
            {GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_NL, "IQ4_XS / IQ4_NL"}};
        const std::span<const Pair> pairs = pack_pairs ? std::span<const Pair>(mixed) : std::span<const Pair>(single);
        cudaStream_t s = nullptr;
        ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
        int run = 0;
        for (const Pair& p : pairs) {
            if (only && std::string(p.name).find(only) == std::string::npos) continue;
            if (!mmq_ok(p))
                throw std::runtime_error(std::string(p.name) + ": no MMQ for a selected pair");
            if (!mmq::fits((int) p.gu, 1280) || !mmq::fits((int) p.d, N))
                throw std::runtime_error(std::string(p.name) + ": selected pair needs independent non-MMQ fallback coverage");
            mmq_only_part(p, s, census);
            ++run;
        }
        ck(cudaStreamDestroy(s), "destroy");
        if (run == 0) throw std::runtime_error("--mmq-only: no pair ran (a silent skip would be a false pass)");
        if (pack_pairs && run != 7) throw std::runtime_error("--pack-pairs: incomplete inventory");
        if (census.file) {
            if (std::fflush(census.file.get()) != 0) throw std::runtime_error("Cannot flush SwiGLU census");
            if (std::fclose(census.file.release()) != 0) throw std::runtime_error("Cannot close SwiGLU census");
            if (census.tuples != 7 * 60 * 640) throw std::runtime_error("SwiGLU census incomplete");
            std::printf("SwiGLU diagnostic census: %zu tuples, %zu reference-domain refusals; NOT QUALIFIED\n",
                        census.tuples, census.refused);
            return 2;  // A diagnostic never qualifies the operator, even with no refusals.
        }
        std::printf("prefill MMQ (native formats) parity passed (--mmq-only, %d pairs)\n", run);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "prefill MMQ (native formats) parity failed (--mmq-only): %s\n", e.what());
        return 1;
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
#ifdef _WIN32
        _putenv_s("STRATA_PF_FUSED", "1");
#else
        setenv("STRATA_PF_FUSED", "1", 1);
        setenv("STRATA_PF_FUSED_KQ", "1", 1);   // (the HIP build's Q4_K / Q5_K / Q5_1 / Q8_0 pairs)
#endif
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) { std::printf("no CUDA device: skipped\n"); return 77; }
        bool mmq_only = false;   // --mmq-only: the MMQ expert path alone (no fused comparison: any sm_70+ device)
        bool census_requested = false;
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "--mmq-only") == 0) mmq_only = true;
            if (std::string_view(argv[i]).starts_with("--swiglu-census=")) census_requested = true;
        }
        if (census_requested && !mmq_only)
            throw std::runtime_error("SwiGLU census requires --mmq-only");
        if (!mmq_only && !fused::available()) { std::printf("the fused kernels need sm_80 or newer: skipped\n"); return 77; }
        if (!mmq::built()) { std::printf("no MMQ in this build\n"); return 1; }
        if (mmq_only) return mmq_only_main(argc, argv);
        const std::vector<Pair>& pairs = native_pairs();
        const char* only = nullptr;   // --only=NAME: the pairs whose name contains NAME
        for (int i = 1; i < argc; ++i)
            if (std::strncmp(argv[i], "--only=", 7) == 0) only = argv[i] + 7;
        bool timing = true, ref = true;   // --no-timing, --no-ref: skip a part
        std::vector<int> chunks = {2048, 3584, 8192};   // --chunks=A,B: the timing part's chunk sizes
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "--no-timing") == 0) timing = false;
            if (std::strcmp(argv[i], "--no-ref") == 0) ref = false;
            if (std::strncmp(argv[i], "--chunks=", 9) == 0) {
                chunks.clear();
                for (const char* c = argv[i] + 9; *c;) {
                    chunks.push_back(std::atoi(c));
                    while (*c && *c != ',') ++c;
                    if (*c == ',') ++c;
                }
            }
        }
        cudaStream_t s = nullptr;
        ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
        for (const Pair& p : pairs) {
            if (only ? std::string(p.name).find(only) == std::string::npos : p.gu == GGML_TYPE_Q4_K || p.gu == GGML_TYPE_Q5_K) continue;
            if (!fused::native_supported((int) p.gu, (int) p.d))
                throw std::runtime_error(std::string(p.name) + ": not covered by the native kernels");
            if (ref) reference_part(p, s);
        }
        if (timing) {
            // the IQ2_XS pack's layers (34 of 48 IQ2_S / Q2_0, 11 IQ2_XXS / Q2_0), the IQ3_S pack's most common, IQ3_XXS's
            for (const Pair& p : only ? pairs : std::vector<Pair>{pairs[0], pairs[1], pairs[4], pairs[3]}) {
                if (only && std::string(p.name).find(only) == std::string::npos) continue;
                for (int T : chunks) timing_part(p, T, s);
            }
        }
        ck(cudaStreamDestroy(s), "destroy");
        std::printf("prefill fused MoE (native formats) parity passed\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "prefill fused MoE (native formats) parity failed: %s\n", e.what());
        return 1;
    }
}
