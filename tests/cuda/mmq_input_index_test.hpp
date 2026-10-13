// #219: address-only MMQ regression, using this fixture's native weights/routing/FFN sequence.
// Included inside prefill_fused_iq_test.cpp's anonymous namespace; no fused/native/WMMA products.

void index_equal(const void* a, const void* b, size_t bytes, const char* label) {
    std::vector<uint8_t> x(bytes), y(bytes);
    ck(cudaMemcpy(x.data(), a, bytes, cudaMemcpyDeviceToHost), label);
    ck(cudaMemcpy(y.data(), b, bytes, cudaMemcpyDeviceToHost), label);
    if (x != y) {
        size_t i = 0;
        while (i < bytes && x[i] == y[i]) ++i;
        throw std::runtime_error(std::string(label) + ": exact mismatch at byte " + std::to_string(i));
    }
}

size_t index_workspace(mmq::Context& ctx, std::unique_ptr<Dev>& workspace) {
    int cc = 0, sms = 0;
    uint64_t shared = 0, bytes = 0;
    if (!mmq::device_config(cc, sms, shared) || !mmq::workspace_bytes(cc, sms, shared, bytes))
        throw std::runtime_error("cannot price MMQ workspace");
    workspace = std::make_unique<Dev>(std::max<uint64_t>(bytes, 256));
    ctx.set_workspace(workspace->p, std::max<uint64_t>(bytes, 256));
    return bytes;
}

void index_payload(const Pair& p, const float* x, const int32_t* src, const Routing& r,
                   int T, cudaStream_t s) {
    // The pinned quantizer transposes 144-byte blocks of 128 values across physical rows.
    // Compare codes, scale/sum bits AND padded values, not just the logical activation values.
    constexpr size_t block_bytes = 144;
    const int rows = T * K;
    for (int type : {(int) p.gu, (int) p.d}) for (int cols : {N, FF}) {
        const size_t blocks = ((size_t) cols + 511) / 512 * 4;
        if (mmq::q8_layout(type) == mmq::Q8Layout::Unsupported ||
            mmq::q8_bytes(1, cols) - mmq::q8_bytes(0, cols) != blocks * block_bytes)
            throw std::runtime_error("unrecognized quantizer representation");
        Dev expanded(mmq::q8_bytes(rows, cols)), unique(mmq::q8_bytes(T, cols));
        ck(cudaMemsetAsync(expanded.p, 0x5a, mmq::q8_bytes(rows, cols), s), "payload sentinel");
        ck(cudaMemsetAsync(unique.p, 0x5a, mmq::q8_bytes(T, cols), s), "payload sentinel");
        mmq::quantize(x, src, expanded.p, type, cols, N, rows, s);
        mmq::quantize(x, nullptr, unique.p, type, cols, N, T, s);
        ck(cudaStreamSynchronize(s), "payload sync");
        std::vector<uint8_t> a(blocks * rows * block_bytes), b(blocks * T * block_bytes);
        ck(cudaMemcpy(a.data(), expanded.p, a.size(), cudaMemcpyDeviceToHost), "expanded payload");
        ck(cudaMemcpy(b.data(), unique.p, b.size(), cudaMemcpyDeviceToHost), "unique payload");
        for (size_t k = 0; k < blocks; ++k) for (int row = 0; row < rows; ++row)
            if (std::memcmp(a.data() + (k * rows + row) * block_bytes,
                            b.data() + (k * T + r.src[(size_t) row]) * block_bytes, block_bytes))
                throw std::runtime_error("quantized row payload mismatch");
    }
}

void index_empty_groups(const Pair& p, const float* x, int T, cudaStream_t s, mmq::Context& ctx) {
    // Unlike production's compact walk, deliberately retain empty experts inside a product.
    const std::vector<int> counts{0, 1, 3, 0, 7, 2, 0, 9, 1, 0, 17, 2, 0, 3, 1, 0};
    const int rows = std::accumulate(counts.begin(), counts.end(), 0);
    std::vector<int32_t> bounds(17, 0), src((size_t) rows + 128, 0), dst(src.size(), 0);
    for (int e = 0; e < 16; ++e) bounds[e + 1] = bounds[e] + counts[e];
    for (int r = 0; r < rows; ++r) { src[r] = (r * 7 + 3) % T; dst[r] = rows - 1 - r; }
    const size_t eb = mmq::matrix_bytes(p.gu, 1280, N);
    std::mt19937 rng(219);
    std::vector<uint8_t> weights(16 * eb + 4096, 0);
    for (int e = 0; e < 16; ++e) fill_matrix(weights.data() + e * eb, p.gu, 1280, N, rng);
    Dev w(weights.size()), ds(src.size() * 4), dd(dst.size() * 4), db(bounds.size() * 4),
        q0(mmq::q8_bytes(rows, N)), q1(mmq::q8_bytes(T, N)),
        y0((size_t) rows * 1280 * 4), y1((size_t) rows * 1280 * 4);
    ck(cudaMemcpyAsync(w.p, weights.data(), weights.size(), cudaMemcpyHostToDevice, s), "empty weights");
    ck(cudaMemcpyAsync(ds.p, src.data(), src.size() * 4, cudaMemcpyHostToDevice, s), "empty src");
    ck(cudaMemcpyAsync(dd.p, dst.data(), dst.size() * 4, cudaMemcpyHostToDevice, s), "empty dst");
    ck(cudaMemcpyAsync(db.p, bounds.data(), bounds.size() * 4, cudaMemcpyHostToDevice, s), "empty bounds");
    mmq::quantize(x, ds.as<int32_t>(), q0.p, p.gu, N, N, rows, s);
    mmq::quantize(x, nullptr, q1.p, p.gu, N, N, T, s);
    mmq::Product prod;
    prod.w = w.p; prod.type = p.gu; prod.w_rows = 1280; prod.w_cols = N; prod.expert_bytes = eb;
    prod.n = 16; prod.bounds = db.as<int32_t>(); prod.ids = dd.as<int32_t>();
    prod.total_rows = rows; prod.max_rows = 17; prod.ld_dst = 1280;
    prod.xq = q0.p; prod.dst = y0.as<float>(); ctx.run(prod, s);
    prod.xq = q1.p; prod.dst = y1.as<float>(); prod.input_ids = ds.as<int32_t>(); prod.input_rows = T;
    ctx.run(prod, s);
    ck(cudaStreamSynchronize(s), "empty group sync");
    index_equal(y0.p, y1.p, (size_t) rows * 1280 * 4, "empty groups/output maps");
}

void index_pair(const Pair& p, cudaStream_t s, bool graph) {
    constexpr int T = 73, E = 37;
    const int rows = T * K;
    const Geo geo(p);
    std::mt19937 rng(219 + (int) p.gu * 31 + (int) p.d);
    const auto ids = make_ids(T, E, rng, 4, 7);
    const Routing r = sort_rows(ids, E);
    auto x = make_x(T, rng, 0);
    std::vector<uint8_t> weights((size_t) E * geo.bytes + 4096, 0);
    for (int e = 0; e < E; ++e) {
        const auto blob = make_blob(p, geo, rng);
        std::memcpy(weights.data() + (size_t) e * geo.bytes, blob.data(), geo.bytes);
    }
    std::vector<int32_t> src = r.src, output((size_t) rows + 128, 0), slots((size_t) rows);
    src.resize((size_t) rows + 128, 0); // existing MMQ's output-map tile-tail reads remain in bounds
    for (int row = 0; row < rows; ++row) output[row] = rows - 1 - row;
    for (int i = 0; i < rows; ++i) slots[i] = output[r.row_of[i]];
    std::vector<float> wk((size_t) rows), sg(T, 0.0f);
    for (int i = 0; i < rows; ++i) wk[i] = (float) (1 + i % K) / 64.0f;
    Dev dx(x.size() * 4), dw(weights.size()), ds(src.size() * 4), dout(output.size() * 4),
        dslots(slots.size() * 4), dweights(wk.size() * 4), dsg(sg.size() * 4),
        bo0((size_t) T * N * 4), bo1((size_t) T * N * 4);
    ck(cudaMemcpyAsync(dx.p, x.data(), x.size() * 4, cudaMemcpyHostToDevice, s), "input");
    ck(cudaMemcpyAsync(dw.p, weights.data(), weights.size(), cudaMemcpyHostToDevice, s), "weights");
    ck(cudaMemcpyAsync(ds.p, src.data(), src.size() * 4, cudaMemcpyHostToDevice, s), "src");
    ck(cudaMemcpyAsync(dout.p, output.data(), output.size() * 4, cudaMemcpyHostToDevice, s), "output map");
    ck(cudaMemcpyAsync(dslots.p, slots.data(), slots.size() * 4, cudaMemcpyHostToDevice, s), "slots");
    ck(cudaMemcpyAsync(dweights.p, wk.data(), wk.size() * 4, cudaMemcpyHostToDevice, s), "route weights");
    ck(cudaMemcpyAsync(dsg.p, sg.data(), sg.size() * 4, cudaMemcpyHostToDevice, s), "shared gate");
    std::vector<const uint8_t*> blobs((size_t) E);
    for (int e = 0; e < E; ++e) blobs[e] = dw.as<uint8_t>() + (size_t) e * geo.bytes;
    MmqBufs old(p, rows + 128, E), indexed(p, rows + 128, E, T);
    mmq::Context c0, c1;
    std::unique_ptr<Dev> ws0, ws1;
    index_workspace(c0, ws0); index_workspace(c1, ws1);
    auto run = [&](bool on, bool tables_ready) {
        auto& b = on ? indexed : old;
        auto& ctx = on ? c1 : c0;
        run_mmq(p, geo, ctx, b, r, dx.as<float>(), ds.as<int32_t>(), blobs, rows, s,
                 false, 0, on ? T : 0, dout.as<int32_t>(), tables_ready);
        strata::prefill::moe_combine(b.dm.as<float>(), dslots.as<int32_t>(), dweights.as<float>(),
                                    dx.as<float>(), dsg.as<float>(), on ? bo1.as<float>() : bo0.as<float>(), T, s);
    };
    auto equal = [&] {
        ck(cudaStreamSynchronize(s), "FFN sync");
        index_equal(old.gu.p, indexed.gu.p, (size_t) rows * 1280 * 4, "gate/up");
        index_equal(old.h.p, indexed.h.p, (size_t) rows * FF * 4, "FP32 H");
        index_equal(old.dm.p, indexed.dm.p, (size_t) rows * N * 4, "complete routed FFN/output map");
        index_equal(bo0.p, bo1.p, (size_t) T * N * 4, "complete weighted FFN/k-order reduction");
        std::vector<float> finite((size_t) rows * N);
        ck(cudaMemcpy(finite.data(), indexed.dm.p, finite.size() * 4, cudaMemcpyDeviceToHost), "finite FFN");
        for (float f : finite) if (!std::isfinite(f)) throw std::runtime_error("nonfinite exact result");
    };
    index_payload(p, dx.as<float>(), ds.as<int32_t>(), r, T, s);
    index_empty_groups(p, dx.as<float>(), T, s, c1);
    run(false, false); run(true, false); equal();
    if (graph) {
        cudaGraph_t g = nullptr;
        cudaGraphExec_t exec = nullptr;
        ck(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), "capture");
        run(true, true); // tables already device-resident; no captured ephemeral host pointers
        ck(cudaStreamEndCapture(s, &g), "capture end");
        ck(cudaGraphInstantiate(&exec, g, nullptr, nullptr, 0), "instantiate");
        cudaStream_t other_stream = nullptr;
        ck(cudaStreamCreateWithFlags(&other_stream, cudaStreamNonBlocking), "other request stream");
        Dev other_x(x.size() * 4);
        MmqBufs other(p, rows + 128, E, T);
        mmq::Context other_ctx;
        std::unique_ptr<Dev> other_ws;
        index_workspace(other_ctx, other_ws);
        for (int request = 1; request <= 3; ++request) {
            x = make_x(T, rng, request);
            ck(cudaMemcpyAsync(dx.p, x.data(), x.size() * 4, cudaMemcpyHostToDevice, s), "next request");
            run(false, true);
            ck(cudaGraphLaunch(exec, s), "replay");
            const auto other_input = make_x(T, rng, request + 4);
            ck(cudaMemcpyAsync(other_x.p, other_input.data(), other_input.size() * 4,
                               cudaMemcpyHostToDevice, other_stream), "isolated request input");
            run_mmq(p, geo, other_ctx, other, r, other_x.as<float>(), ds.as<int32_t>(), blobs, rows,
                     other_stream, false, 0, T, dout.as<int32_t>());
            equal(); // B must not overwrite A's graph/workspace/quantized rows
            ck(cudaStreamSynchronize(other_stream), "other request drain");
            run(true, true); equal(); // uncaptured reuse after replay must also stay exact
        }
        ck(cudaStreamDestroy(other_stream), "other stream destroy");
        ck(cudaGraphExecDestroy(exec), "graph exec destroy");
        ck(cudaGraphDestroy(g), "graph destroy");
    }
    std::printf("indexed exact %-19s tokens=%d routes=%d empty=7 group_tail=14 graph=%d Xq=%zu->%zu B\n",
                p.name, T, rows, (int) graph, mmq::q8_bytes(rows, N), mmq::q8_bytes(T, N));
}

int mmq_input_index_main() {
    if (!mmq::input_index_supported(GGML_TYPE_IQ3_S, N)) {
        std::printf("MMQ input indexing is not built: skipped\n"); return 77;
    }
    cudaStream_t s = nullptr;
    ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "index stream");
    int pairs = 0;
    for (const Pair& p : native_pairs()) if (mmq_ok(p)) {
        index_pair(p, s, p.gu == GGML_TYPE_IQ3_S || p.gu == GGML_TYPE_IQ2_S);
        ++pairs;
    }
    ck(cudaStreamDestroy(s), "index stream destroy");
    std::printf("indexed DEFAULT MMQ exact payload/FFN/graph/request-isolation passed: %d pairs\n", pairs);
    return 0;
}

int mmq_input_index_timing() {
    if (!mmq::input_index_supported(GGML_TYPE_IQ3_S, N)) return 77;
    constexpr int T = 2048, E = 512, rows = T * K;
    const Pair p{GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_NL, "IQ3_S / IQ4_NL"};
    const Geo geo(p);
    std::mt19937 rng(219);
    auto x = make_x(T, rng, 0);
    std::vector<int32_t> ids(rows);
    for (int t = 0; t < T; ++t) for (int k = 0; k < K; ++k) ids[t * K + k] = (t * 7 + k * 47) % E;
    const Routing r = sort_rows(ids, E);
    std::vector<uint8_t> weights((size_t) E * geo.bytes + 4096, 0);
    for (int e = 0; e < E; ++e) {
        const auto blob = make_blob(p, geo, rng);
        std::memcpy(weights.data() + (size_t) e * geo.bytes, blob.data(), geo.bytes);
    }
    Dev dx(x.size() * 4), ds((size_t) (rows + 128) * 4), dw(weights.size());
    cudaStream_t s = nullptr;
    ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "timing stream");
    ck(cudaMemcpyAsync(dx.p, x.data(), x.size() * 4, cudaMemcpyHostToDevice, s), "timing x");
    ck(cudaMemsetAsync(ds.p, 0, (size_t) (rows + 128) * 4, s), "timing src padding");
    ck(cudaMemcpyAsync(ds.p, r.src.data(), (size_t) rows * 4, cudaMemcpyHostToDevice, s), "timing src");
    ck(cudaMemcpyAsync(dw.p, weights.data(), weights.size(), cudaMemcpyHostToDevice, s), "timing weights");
    std::vector<const uint8_t*> blobs(E);
    for (int e = 0; e < E; ++e) blobs[e] = dw.as<uint8_t>() + (size_t) e * geo.bytes;
    MmqBufs b0(p, rows + 128, E), b1(p, rows + 128, E, T);
    mmq::Context c0, c1;
    std::unique_ptr<Dev> ws0, ws1;
    index_workspace(c0, ws0); index_workspace(c1, ws1);
    auto run = [&](bool on, bool direct) {
        run_mmq(p, geo, on ? c1 : c0, on ? b1 : b0, r, dx.as<float>(), ds.as<int32_t>(), blobs,
                 rows, s, direct, geo.bytes, on ? T : 0);
    };
    run(false, false); run(true, false); ck(cudaStreamSynchronize(s), "timing warmup");
    index_equal(b0.dm.p, b1.dm.p, (size_t) rows * N * 4, "real routed shape FFN");
    cudaEvent_t a = nullptr, b = nullptr;
    ck(cudaEventCreate(&a), "timing event"); ck(cudaEventCreate(&b), "timing event");
    for (bool direct : {false, true}) {
        std::vector<float> times[2];
        for (int repeat = 0; repeat < 7; ++repeat) for (int arm = 0; arm < 2; ++arm) {
            const bool on = ((repeat & 1) ? 1 - arm : arm) != 0;
            ck(cudaEventRecord(a, s), "timing begin"); run(on, direct);
            ck(cudaEventRecord(b, s), "timing end"); ck(cudaEventSynchronize(b), "timing join");
            float ms = 0; ck(cudaEventElapsedTime(&ms, a, b), "timing duration");
            times[on ? 1 : 0].push_back(ms);
        }
        for (auto& t : times) std::sort(t.begin(), t.end());
        std::printf("indexed matched FFN T=2048 K=10 E=512 %s off=%.6f on=%.6f ms gain=%.3f%% Xq=%zu->%zu B\n",
                    direct ? "input-prep+products" : "input-prep+gather+products",
                    times[0][3], times[1][3], 100.0 * (1.0 - times[1][3] / times[0][3]),
                    mmq::q8_bytes(rows, N), mmq::q8_bytes(T, N));
    }
    ck(cudaEventDestroy(a), "timing event destroy"); ck(cudaEventDestroy(b), "timing event destroy");
    ck(cudaStreamDestroy(s), "timing stream destroy");
    return 0;
}
