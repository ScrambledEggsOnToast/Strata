// Included by prefill_fused_iq_test.cpp inside its test namespace, reusing GGML fixture/decoder helpers.
// #195 limits are fixed in the parent project's preregistration, not fitted to a GPU output.
#pragma once

float wmma_half(float x) { return ggml_fp16_to_fp32(ggml_fp32_to_fp16(x)); }
double wmma_silu(double x) {
    if (x >= 0) return x / (1.0 + std::exp(-x));
    const double e = std::exp(x);
    return x * e / (1.0 + e);
}

void wmma_swiglu_boundaries(cudaStream_t stream) {
    std::vector<float> gu(1280, 0.0f);
    gu[0] = 1000; gu[640] = 1000;
    gu[1] = 1000; gu[641] = -1000;
    gu[2] = std::numeric_limits<float>::quiet_NaN(); gu[642] = 1;
    gu[3] = -1000; gu[643] = 1000;
    Dev input(gu.size() * 4), output(640 * 2);
    ck(cudaMemcpyAsync(input.p, gu.data(), gu.size() * 4, cudaMemcpyHostToDevice, stream), "SwiGLU fixture");
    strata::prefill::swiglu_split(input.as<float>(), output.as<uint16_t>(), 1, stream);
    std::vector<ggml_fp16_t> h(640);
    ck(cudaMemcpyAsync(h.data(), output.p, h.size() * 2, cudaMemcpyDeviceToHost, stream), "SwiGLU boundaries");
    ck(cudaStreamSynchronize(stream), "SwiGLU sync");
    if (ggml_fp16_to_fp32(h[0]) != 65504 || ggml_fp16_to_fp32(h[1]) != -65504 ||
        !std::isnan(ggml_fp16_to_fp32(h[2])) || ggml_fp16_to_fp32(h[3]) != 0)
        throw std::runtime_error("FP16 SwiGLU saturation/NaN/negative-tail contract failed");
}

void wmma_ffn_part(const Pair& pair, cudaStream_t stream) {
    constexpr int E = 4, T = 67, START = 4, LDX = N + 8, LDGU = 1280 + 4, LDY = N + 4;
    const int32_t relative[E + 1] = {0, 0, 17, 82, 339}; // empty, 17, 65, 257: partial WMMA/chunk tiles
    constexpr int ROWS = 339;
    const int allocation_rows = START + ROWS + 1;
    std::vector<int32_t> absolute(E + 1), ids(allocation_rows);
    for (int e = 0; e <= E; ++e) absolute[e] = START + relative[e];
    for (int r = 0; r < allocation_rows; ++r) ids[r] = (r * 37 + 5) % T;
    const Geo geometry(pair);
    std::mt19937 rng(195 + (unsigned) pair.gu * 7 + (unsigned) pair.d);
    std::vector<std::vector<uint8_t>> host(E);
    std::vector<std::unique_ptr<Dev>> weights;
    std::vector<const uint8_t*> blobs(E);
    for (int e = 0; e < E; ++e) {
        host[e] = make_blob(pair, geometry, rng);
        // Native Q2_0/IQ4_NL block scales may be negative. Exercise both
        // signs rather than only the positive-scale synthetic fixture.
        for (size_t off = geometry.down_off; off < geometry.bytes; off += 7 * ggml_type_size((ggml_type) pair.d)) {
            uint16_t scale;
            std::memcpy(&scale, host[e].data() + off, sizeof(scale));
            scale ^= 0x8000;
            std::memcpy(host[e].data() + off, &scale, sizeof(scale));
        }
        weights.push_back(std::make_unique<Dev>(geometry.bytes + 64));
        blobs[e] = weights.back()->as<uint8_t>() + 2 * e; // unrelated, merely 2-byte aligned blobs
        ck(cudaMemcpyAsync((void*) blobs[e], host[e].data(), geometry.bytes, cudaMemcpyHostToDevice, stream), "WMMA blob");
    }
    const auto x = make_x(T, rng, ids[START]); // repeated/permuted inputs; first active row is exactly zero
    std::vector<ggml_fp16_t> hx((size_t) T * LDX, ggml_fp32_to_fp16(0));
    for (int r = 0; r < T; ++r)
        for (int k = 0; k < N; ++k) hx[(size_t) r * LDX + k] = ggml_fp32_to_fp16(x[(size_t) r * N + k]);
    Dev x16(hx.size() * 2), source(ids.size() * 4), bounds(2 * (E + 1) * 4);
    Dev gu((size_t) allocation_rows * LDGU * 4), h((size_t) allocation_rows * FF * 2);
    Dev out((size_t) allocation_rows * LDY * 4), tables(2 * E * sizeof(void*));
    Dev gathered_gu((size_t) E * geometry.down_off), gathered_dn((size_t) E * geometry.d_row * N);
    ck(cudaMemcpyAsync(x16.p, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice, stream), "WMMA FP16 input");
    ck(cudaMemcpyAsync(source.p, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice, stream), "WMMA ids");
    ck(cudaMemcpyAsync(bounds.p, absolute.data(), absolute.size() * 4, cudaMemcpyHostToDevice, stream), "absolute bounds");
    ck(cudaMemcpyAsync(bounds.as<int32_t>() + E + 1, relative, sizeof(relative), cudaMemcpyHostToDevice, stream), "relative bounds");
    // Gathered and indirect modes use exactly the same bytes, decoder and independent references.
    for (int e = 0; e < E; ++e)
        strata::prefill::mmq::gather_native(blobs[e], blobs[e] + geometry.up_off, geometry.up_off,
            blobs[e] + geometry.down_off, geometry.d_row * N,
            gathered_gu.as<uint8_t>() + (size_t) e * geometry.down_off,
            gathered_dn.as<uint8_t>() + (size_t) e * geometry.d_row * N, stream);
    std::vector<float> first;
    for (bool indirect : {false, true}) {
        const void* gu_weight = gathered_gu.p;
        const void* down_weight = gathered_dn.p;
        if (indirect) {
            const void* gate_up[E]; const void* down[E];
            for (int e = 0; e < E; ++e) { gate_up[e] = blobs[e]; down[e] = blobs[e] + geometry.down_off; }
            strata::prefill::expert_ptr_table(gate_up, down, E, tables.as<const void*>(), tables.as<const void*>() + E, stream);
            // Prove the table kernel captured values, not asynchronous
            // pointers into these short-lived host arrays.
            for (int e = 0; e < E; ++e) { gate_up[e] = nullptr; down[e] = nullptr; }
            gu_weight = tables.p; down_weight = tables.as<const void*>() + E;
        }
        ck(cudaMemsetAsync(gu.p, 0xff, (size_t) allocation_rows * LDGU * 4, stream), "GU guards");
        ck(cudaMemsetAsync(out.p, 0xff, (size_t) allocation_rows * LDY * 4, stream), "FFN guards");
        if (!strata::kernels::gemm_iq_f16_grouped(pair.gu, gu_weight, geometry.down_off, 1280, N,
                x16.p, LDX, bounds.as<int32_t>(), E, 257, gu.as<float>(), LDGU, stream, source.as<int32_t>(), indirect))
            throw std::runtime_error("covered WMMA gate/up refused");
        // swiglu_split is the production contiguous-GU epilogue. The padded product above tests ldy;
        // copying only payloads into a contiguous view also makes the untouched guards observable.
        Dev contiguous_gu((size_t) ROWS * 1280 * 4);
        ck(cudaMemcpy2DAsync(contiguous_gu.p, 1280 * 4, gu.as<float>() + START * LDGU, LDGU * 4,
                            1280 * 4, ROWS, cudaMemcpyDeviceToDevice, stream), "contiguous GU view");
        strata::prefill::swiglu_split(contiguous_gu.as<float>(), h.as<uint16_t>(), ROWS, stream);
        if (!strata::kernels::gemm_iq_f16_grouped(pair.d, down_weight, geometry.d_row * N, N, FF,
                h.p, FF, bounds.as<int32_t>() + E + 1, E, 257,
                out.as<float>() + START * LDY, LDY, stream, nullptr, indirect))
            throw std::runtime_error("covered WMMA down refused");
        ck(cudaStreamSynchronize(stream), "WMMA FFN sync");
        const auto got_gu = download(gu, (size_t) allocation_rows * LDGU);
        const auto got = download(out, (size_t) allocation_rows * LDY);
        for (int r = 0; r < allocation_rows; ++r)
            for (int c = 0; c < LDY; ++c) {
                const bool payload = r >= START && r < START + ROWS && c < N;
                if (payload ? !std::isfinite(got[(size_t) r * LDY + c]) : !std::isnan(got[(size_t) r * LDY + c]))
                    throw std::runtime_error("FFN output unwritten, non-finite or guard overwritten");
            }
        for (int r = 0; r < allocation_rows; ++r)
            for (int c = 0; c < LDGU; ++c) {
                const bool payload = r >= START && r < START + ROWS && c < 1280;
                if (payload ? !std::isfinite(got_gu[(size_t) r * LDGU + c]) : !std::isnan(got_gu[(size_t) r * LDGU + c]))
                    throw std::runtime_error("GU output unwritten, non-finite or guard overwritten");
            }
        for (int r = START; r < START + ROWS; ++r)
            if (ids[r] == ids[START])
                for (int o = 0; o < N; ++o)
                    if (got[(size_t) r * LDY + o] != 0) throw std::runtime_error("zero input changed by FFN");
        if (indirect && std::memcmp(first.data(), got.data(), got.size() * 4) != 0)
            throw std::runtime_error("indirect/gathered FFN differs for identical packed bytes");
        if (!indirect) first = got;
    }
    // Independent complete FFN at each expert's start/end and each internal tile/chunk boundary.
    double error2[2]{}, ref2[2]{}, max_error[2]{};
    size_t values = 0;
    for (int e = 1; e < E; ++e) {
        std::vector<float> wgu, wdn;
        dequant(pair, geometry, host[e], wgu, wdn);
        auto half_gu = wgu, half_dn = wdn;
        for (float& value : half_gu) value = wmma_half(value);
        for (float& value : half_dn) value = wmma_half(value);
        const int begin = START + relative[e], end = START + relative[e + 1];
        for (int offset : {0, 15, 16, 63, 64, 127, 128, 255, 256}) {
            const int row = begin + offset;
            if (row >= end) continue;
            std::vector<double> ref_h[2] = {std::vector<double>(FF), std::vector<double>(FF)};
            for (int r = 0; r < FF; ++r) {
                double gate[2]{}, up[2]{};
                for (int k = 0; k < N; ++k) {
                    const double input = x[(size_t) ids[row] * N + k];
                    const double rounded = ggml_fp16_to_fp32(hx[(size_t) ids[row] * LDX + k]);
                    gate[0] += (double) half_gu[(size_t) r * N + k] * rounded;
                    up[0] += (double) half_gu[(size_t) (FF + r) * N + k] * rounded;
                    gate[1] += (double) wgu[(size_t) r * N + k] * input;
                    up[1] += (double) wgu[(size_t) (FF + r) * N + k] * input;
                }
                const double nonlinear = wmma_silu((float) gate[0]) * (float) up[0];
                ref_h[0][r] = wmma_half((float) std::clamp(nonlinear, -65504.0, 65504.0));
                ref_h[1][r] = wmma_silu(gate[1]) * up[1];
            }
            for (int o = 0; o < N; ++o) {
                double ref[2]{};
                for (int r = 0; r < FF; ++r) {
                    ref[0] += (double) half_dn[(size_t) o * FF + r] * ref_h[0][r];
                    ref[1] += (double) wdn[(size_t) o * FF + r] * ref_h[1][r];
                }
                for (int contract = 0; contract < 2; ++contract) {
                    const double delta = first[(size_t) row * LDY + o] - ref[contract];
                    error2[contract] += delta * delta; ref2[contract] += ref[contract] * ref[contract];
                    max_error[contract] = std::max(max_error[contract], std::fabs(delta));
                }
                ++values;
            }
        }
    }
    for (int contract = 0; contract < 2; ++contract) {
        const double rms = std::sqrt(ref2[contract] / values);
        const double relative_error = ref2[contract] ? std::sqrt(error2[contract] / ref2[contract]) : std::sqrt(error2[contract]);
        const double normalized_max = rms ? max_error[contract] / rms : max_error[contract];
        std::printf("WMMA %s reference=%s sampled_values=%zu relL2=%.6g max/RMS=%.6g\n", pair.name,
                    contract ? "unrounded" : "FP16-contract", values, relative_error, normalized_max);
        if (relative_error > (contract ? 0.01 : 0.001) || normalized_max > (contract ? 0.1 : 0.01))
            throw std::runtime_error("preregistered WMMA complete-FFN limit exceeded");
    }
}

int wmma_only_main() {
    cudaDeviceProp device{};
    ck(cudaGetDeviceProperties(&device, 0), "WMMA device");
    if (device.major != 7 || device.minor != 0) {
        if (strata::kernels::gemm_iq_f16_grouped_supported(21, 1280, N))
            throw std::runtime_error("WMMA was enabled off SM70");
        std::printf("SM70 WMMA numerical test skipped; non-SM70 refusal checked\n");
        return 77;
    }
    if (strata::kernels::gemm_iq_f16_grouped_supported(17, 1280, N) ||
        strata::kernels::gemm_iq_f16_grouped_supported(21, 1279, N) ||
        strata::kernels::gemm_iq_f16_grouped_supported(21, 1280, N - 1))
        throw std::runtime_error("unsupported WMMA type/shape accepted");
    if (!strata::platform::acknowledge_protected_test()) return 2;
    cudaStream_t stream = nullptr;
    ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "WMMA stream");
    wmma_swiglu_boundaries(stream);
    const std::vector<Pair> pairs = {
        {GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_NL, "IQ3_XXS/IQ4_NL"},
        {GGML_TYPE_IQ3_XXS, GGML_TYPE_Q2_0, "IQ3_XXS/Q2_0"},
        {GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_NL, "IQ3_S/IQ4_NL"},
        {GGML_TYPE_IQ3_S, GGML_TYPE_Q2_0, "IQ3_S/Q2_0"},
        {GGML_TYPE_IQ2_S, GGML_TYPE_IQ4_NL, "IQ2_S/IQ4_NL"},
        {GGML_TYPE_IQ2_S, GGML_TYPE_Q2_0, "IQ2_S/Q2_0"},
        {GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_NL, "IQ4_XS/IQ4_NL"}
    };
    for (const Pair& pair : pairs) wmma_ffn_part(pair, stream);
    ck(cudaStreamDestroy(stream), "WMMA stream destroy");
    std::printf("Independent SM70 IQ WMMA complete-FFN checks passed (bounded synthetic fixtures)\n");
    return 0;
}
