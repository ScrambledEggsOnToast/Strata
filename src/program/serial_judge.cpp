#include "strata/program/serial_judge.hpp"
#include "strata/core/conversation_snapshot.hpp"
#include "strata/core/native_head.hpp"
#include "strata/core/operation_canary.hpp"
#include "strata/core/mtp.hpp"
#include "strata/core/verify.hpp"
#include "strata/prefill/prefill.hpp"
#include "strata/plan/resident.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <exception>

#if defined(__linux__) && !defined(STRATA_USE_HIP)
extern "C" void strata_judge_trace_phase(const char*) __attribute__((weak));
#endif

namespace strata::program {
namespace {
constexpr uint64_t granule = 2ull << 20;
uint64_t rounded(uint64_t bytes) { return bytes ? (bytes + granule - 1) / granule * granule : 0; }
void observe(const char* phase, int device) {
    const core::OnDevice on(device);
    size_t free = 0, total = 0;
    if (cudaMemGetInfo(&free, &total) != cudaSuccess) std::terminate();
    std::fprintf(stderr, "JUDGE_MEMORY phase=%s device=%d driver_used_bytes=%llu driver_free_bytes=%llu\n",
                 phase, device, (unsigned long long)(total - free), (unsigned long long)free);
#if defined(__linux__) && !defined(STRATA_USE_HIP)
    if (strata_judge_trace_phase) strata_judge_trace_phase(phase);
#endif
}
}

uint64_t JudgeStageBytes::device_bytes() const {
    return session.device + device_canary + device_head_output + device_allocator_padding;
}
uint64_t JudgeStageBytes::host_bytes() const {
    return session.host_pinned + session.host_heap + host_canary + host_checkpoint + host_readback + host_metadata;
}

struct SerialJudgeLane::Impl {
    struct Stage {
        JudgeStage binding;
        core::SessionOwner session;
        core::OperationCanary canary;
        core::ConversationCheckpoint branch;
        JudgeStageBytes price;
        uint64_t saved_request = 0;
        const void* saved_gemm = nullptr;
        bool rebound = false;
        std::function<bool(const float*, int64_t, int64_t, std::string&)> saved_chunk;
        std::function<bool(int64_t, std::string&)> saved_stage_chunk;
        std::function<bool()> saved_stop;
        const float* const* saved_embeddings = nullptr;
    };
    core::ModelGeometry geometry;
    int64_t cells = 0, vocabulary = 0, request_tokens = 0;
    std::vector<std::unique_ptr<Stage>> stages;
    const core::NativeHead* head = nullptr;
    core::MtpDrafter* draft = nullptr;
    float* device_logits = nullptr;
    std::vector<float> readback;
    std::atomic<bool> running{false};
    bool initialized = false, poisoned = false, row_ready = false;
    ~Impl() {
        if (running.load()) std::terminate();
        if (device_logits) {
            const core::OnDevice on(stages.back()->session.device());
            if (cudaDeviceSynchronize() != cudaSuccess || cudaFree(device_logits) != cudaSuccess) std::terminate();
        }
    }
    bool idle(std::string& error) {
        if (poisoned) { error = "judge: ownership lease poisoned"; return false; }
        if (draft && !draft->context_idle(error)) return false;
        for (const auto& stage : stages) {
            const auto& binding = stage->binding;
            if ((binding.verifier && !binding.verifier->context_idle(error)) ||
                (binding.batch_peer && !binding.batch_peer->context_idle(error)) ||
                !binding.prompt->context_idle(error)) return false;
            const core::OnDevice on(stage->session.device());
            if (cudaStreamQuery((cudaStream_t)binding.stream) != cudaSuccess) {
                error = "judge: generation stream still in flight or failed"; return false;
            }
        }
        return true;
    }
    bool restore(std::string& error) {
        bool okay = true;
        for (auto& stage : stages) {
            auto& prompt = *stage->binding.prompt;
            if (stage->rebound) {
                const core::OnDevice on(stage->session.device());
                if (!prompt.rebind_session(stage->binding.generation->state(), stage->binding.stream,
                                           stage->saved_request, error)) okay = false;
                stage->binding.generation->bind_rope();
                prompt.on_chunk = std::move(stage->saved_chunk);
                prompt.on_stage_chunk = std::move(stage->saved_stage_chunk);
                prompt.should_stop = std::move(stage->saved_stop);
                prompt.embd_rows = stage->saved_embeddings;
                if (prompt.gemm_owner() != stage->saved_gemm ||
                    prompt.bound_session() != &stage->binding.generation->state() ||
                    prompt.bound_request() != stage->saved_request) okay = false;
                stage->rebound = false;
            }
        }
        if (!okay) { poisoned = true; error = "judge: failed restoring generation ownership; no resume permitted"; }
        return okay;
    }
};

bool judge_stage_bytes(const core::ModelGeometry& g, int64_t cells, int64_t k,
                       int64_t lo, int64_t hi, int64_t vocab, bool head_stage,
                       const plan::QsaPlacement& placement, JudgeStageBytes& out) {
    out = {};
    if (cells < 1 || cells > 4096 || vocab < 1 || vocab > INT32_MAX || placement.kv.mode != 0 ||
        lo < 0 || hi <= lo || hi > g.n_layers) return false;
    JudgeStageBytes bytes;
    if (!core::session_allocation_bytes(g, cells, k, bytes.session, lo, hi, &placement)) return false;
    auto shape = kernels::qsa_real_shapes();
    shape.idx_dim = g.idx_key_dim;
    const auto layers = plan::session_layer_counts(g.n_layers, g.qsa_interval, lo, hi);
    plan::CheckpointStateBytes state;
    plan::ResidentCheckpointPayload checkpoint;
    if (!plan::checkpoint_state_bytes(layers.gdn, g.ssm_state_size, g.ssm_v_heads,
                                      g.ssm_conv_channels, g.ssm_d_conv, shape, state) ||
        !plan::resident_checkpoint_stage(cells, layers.qsa, state, checkpoint)) return false;
    bytes.device_canary = core::kOperationCanaryDeviceBytes;
    bytes.host_canary = core::kOperationCanaryHostBytes;
    bytes.device_head_output = head_stage ? (uint64_t)vocab * sizeof(float) : 0;
    bytes.host_readback = bytes.device_head_output;
    bytes.device_allocator_padding = rounded(bytes.session.device) - bytes.session.device +
        rounded(bytes.device_canary) - bytes.device_canary +
        rounded(bytes.device_head_output) - bytes.device_head_output;
    bytes.host_checkpoint = checkpoint.bytes + (checkpoint.bytes + 7) / 8;
    bytes.host_metadata = sizeof(SerialJudgeLane::Impl::Stage) + sizeof(void*) + sizeof(SerialJudgeLane::Impl);
    // SessionOwner is included by session.host_heap; do not charge it twice.
    bytes.host_metadata -= sizeof(core::SessionOwner);
    out = bytes;
    return true;
}

SerialJudgeLane::SerialJudgeLane() : impl_(std::make_unique<Impl>()) {}
SerialJudgeLane::~SerialJudgeLane() = default;
bool SerialJudgeLane::ready() const { return impl_->initialized && !impl_->poisoned && !impl_->running.load(); }
int64_t SerialJudgeLane::vocab() const { return impl_->vocabulary; }
const float* SerialJudgeLane::logits() const { return impl_->row_ready ? impl_->readback.data() : nullptr; }

bool SerialJudgeLane::init(const core::ModelGeometry& g, int64_t cells, int64_t k, int64_t vocab,
                           const plan::QsaPlacement& placement, const std::vector<JudgeStage>& bindings,
                           const core::NativeHead& head, core::MtpDrafter* draft,
                           const plan::AdmissionDecision& admission, std::string& error) {
    auto& m = *impl_;
    if (m.initialized || !m.stages.empty() || bindings.empty() || !head.loaded() ||
        !admission.admitted || !admission.verified || !admission.unknown_classes.empty()) {
        error = "judge: requires verified complete admission and existing native execution owners"; return false;
    }
    m.geometry = g; m.cells = cells; m.vocabulary = vocab; m.head = &head; m.draft = draft;
    m.stages.reserve(bindings.size());
    int64_t lo = 0;
    for (const auto& binding : bindings) {
        if (!binding.generation || !binding.prompt || !binding.weights ||
            binding.prompt->bound_session() != &binding.generation->state() ||
            binding.prompt->stream() != binding.stream || binding.generation->state().layer_lo != lo) {
            error = "judge: actual generation execution binding does not match descriptor"; return false;
        }
        auto stage = std::make_unique<Impl::Stage>();
        stage->binding = binding;
        const auto& generation = binding.generation->state();
        lo = generation.layer_hi;
        if (!judge_stage_bytes(g, cells, k, generation.layer_lo, lo, vocab, lo == g.n_layers, placement, stage->price)) {
            error = "judge: unpriced short-state geometry"; return false;
        }
        const core::OnDevice on(binding.generation->device());
        observe("before-state-init", binding.generation->device());
        if (!stage->session.init(g, cells, k, error, generation.layer_lo, lo, &placement)) return false;
        binding.generation->bind_rope();
        if (stage->session.memory().device != stage->price.session.device || !stage->canary.init(error) ||
            !binding.prompt->validate_rebind(stage->session.state(), binding.stream, UINT64_MAX, error)) return false;
        auto& state = stage->session.state();
        if (std::getenv("STRATA_JUDGE_OWNERSHIP_PROBE")) {
            auto refuses = [&](const core::SessionState& candidate, void* stream, uint64_t request) {
                std::string why;
                return !binding.prompt->validate_rebind(candidate, stream, request, why);
            };
            auto invalid = state; // borrowed read-only view, never released or bound
            ++invalid.allocation_geometry.n_embd;
            bool checked = refuses(invalid, binding.stream, UINT64_MAX);
            invalid = state; ++invalid.allocation_device;
            checked = checked && refuses(invalid, binding.stream, UINT64_MAX);
            invalid = state; invalid.max_cells = INT32_MAX;
            checked = checked && refuses(invalid, binding.stream, UINT64_MAX);
            checked = checked && refuses(state, nullptr, UINT64_MAX) && refuses(state, binding.stream, 0);
            if (!checked) { error = "judge: rebind rejection guard failed"; return false; }
            std::fprintf(stderr, "JUDGE_REBIND_GUARDS device=%d geometry=1 device_owner=1 max_blocks=1 stream_owner=1 request_canary=1\n",
                         stage->session.device());
        }
        state.ple = generation.ple; // weights/table/staging are serially shared; history and IDs are private
        state.ple.hist = state.ple_hist; state.ple.token = &state.ple_token; state.ple.prev = state.ple_prev;
        core::ConversationStateSizes sizes;
        if (!core::conversation_session_sizes(g, state, sizes, error)) return false;
        stage->branch.ids.reserve((size_t)cells);
        stage->branch.gdn.resize(sizes.gdn); stage->branch.ple.resize(sizes.ple);
        stage->branch.tails.resize((size_t)state.qsa_alloc * sizes.tail);
        stage->branch.dead.resize((size_t)state.qsa_alloc * sizes.dead);
        stage->branch.block_pos.resize((size_t)state.qsa_alloc * sizes.block_pos);
        std::fprintf(stderr, "JUDGE_OWNER device=%d cells=%lld session_bytes=%llu device_charge_bytes=%llu "
                             "host_charge_bytes=%llu host_init_transient_bytes=%llu prefill=%p gemm=%p head=%p "
                             "generation_state=%p judge_state=%p stream=%p\n", stage->session.device(), (long long)cells,
                     (unsigned long long)stage->price.session.device, (unsigned long long)stage->price.device_bytes(),
                     (unsigned long long)stage->price.host_bytes(), (unsigned long long)stage->price.session.host_transient,
                     (void*)binding.prompt, binding.prompt->gemm_owner(), (const void*)&head,
                     (const void*)&generation, (void*)&state, binding.stream);
        observe("after-state-init", binding.generation->device());
        m.stages.push_back(std::move(stage));
    }
    if (lo != g.n_layers) { error = "judge: execution stages do not cover the whole model"; return false; }
    const core::OnDevice on(m.stages.back()->session.device());
    cudaPointerAttributes head_attribute{};
    if (cudaPointerGetAttributes(&head_attribute, head.weights()) != cudaSuccess ||
        head_attribute.type != cudaMemoryTypeDevice || head_attribute.device != m.stages.back()->session.device() ||
        cudaMalloc((void**)&m.device_logits, (size_t)vocab * sizeof(float)) != cudaSuccess) {
        error = "judge: head ownership or output allocation failed"; return false;
    }
    m.readback.resize((size_t)vocab);
    observe("lane-ready", m.stages.back()->session.device());
    m.initialized = true;
    return true;
}

bool SerialJudgeLane::run(uint64_t request, const int64_t* tokens, int64_t count, std::string& error) {
    auto& m = *impl_;
    if (!m.initialized || !request || !tokens || count < 1 || count > m.cells || m.running.exchange(true)) {
        error = "judge: invalid, unadmitted or concurrent scoring request"; return false;
    }
    struct Running { std::atomic<bool>& flag; ~Running() { flag.store(false); } } running{m.running};
    m.row_ready = false;
    m.request_tokens = count;
    for (int64_t t = 0; t < count; ++t) if (tokens[t] < 0 || tokens[t] >= m.vocabulary) {
        error = "judge: token outside model vocabulary"; return false;
    }
    if (!m.idle(error)) return false;
    for (const auto& stage : m.stages) {
        if (request == stage->binding.prompt->bound_request() ||
            !stage->binding.prompt->validate_rebind(stage->session.state(), stage->binding.stream, request, error)) {
            if (error.empty()) error = "judge: scoring and generation request identities must differ";
            return false;
        }
    }
    struct Restore {
        Impl& owner;
        bool restored = false;
        ~Restore() { if (!restored) { std::string error; if (!owner.restore(error)) std::terminate(); } }
    } restore{m};
    for (auto& stage : m.stages) {
        auto& prompt = *stage->binding.prompt;
        const core::OnDevice on(stage->session.device());
        stage->saved_request = prompt.bound_request(); stage->saved_gemm = prompt.gemm_owner();
        stage->saved_chunk = std::move(prompt.on_chunk); stage->saved_stage_chunk = std::move(prompt.on_stage_chunk);
        stage->saved_stop = std::move(prompt.should_stop); stage->saved_embeddings = prompt.embd_rows;
        stage->rebound = true; prompt.embd_rows = nullptr;
        if (!prompt.rebind_session(stage->session.state(), stage->binding.stream, request, error) ||
            !stage->canary.bind(8, request, error)) { m.poisoned = true; return false; }
        stage->session.bind_rope();
        auto& state = stage->session.state();
        stage->canary.begin(8, (cudaStream_t)stage->binding.stream, (int32_t)tokens[0], 0);
        core::session_zero(state, m.geometry, nullptr, stage->binding.stream);
        if (cudaStreamSynchronize((cudaStream_t)stage->binding.stream) != cudaSuccess) {
            m.poisoned = true; error = "judge: short-state initialization failed"; return false;
        }
    }
    m.stages.back()->binding.prompt->on_chunk = [&m](const float* residual, int64_t rows, int64_t position, std::string& why) {
        if (position + rows != m.request_tokens) return true;
        auto& last = *m.stages.back();
        core::BlockBuffers block = last.session.state().block;
        block.R = const_cast<float*>(residual) + (rows - 1) * m.geometry.hc_dim();
        if (!core::lm_head_mix(*last.binding.weights, m.geometry, block, last.binding.stream, why) ||
            !m.head->run(block.mixed, m.device_logits, last.binding.stream, why) ||
            cudaMemcpyAsync(m.readback.data(), m.device_logits, (size_t)m.vocabulary * sizeof(float),
                            cudaMemcpyDeviceToHost, (cudaStream_t)last.binding.stream) != cudaSuccess ||
            cudaStreamSynchronize((cudaStream_t)last.binding.stream) != cudaSuccess) {
            if (why.empty()) why = "judge: native head/readback failed";
            return false;
        }
        for (float value : m.readback) if (!std::isfinite(value)) { why = "judge: nonfinite head output"; return false; }
        m.row_ready = true;
        return true;
    };
    if (!m.stages.front()->binding.prompt->run(tokens, count, 0, error)) { m.poisoned = true; return false; }
    for (auto& stage : m.stages) {
        const core::OnDevice on(stage->session.device());
        stage->canary.check(8, (cudaStream_t)stage->binding.stream);
        if (cudaStreamSynchronize((cudaStream_t)stage->binding.stream) != cudaSuccess ||
            !stage->canary.validate("judge", error)) { m.poisoned = true; return false; }
        stage->branch.ids.resize((size_t)count);
        for (int64_t t = 0; t < count; ++t) stage->branch.ids[(size_t)t] = (int32_t)tokens[t];
        if (!core::conversation_checkpoint_save(stage->branch, stage->session.state(), m.geometry, error)) {
            m.poisoned = true; return false;
        }
        observe("first-use-complete", stage->session.device());
    }
    if (!m.row_ready) { m.poisoned = true; error = "judge: final head row not produced"; return false; }
    restore.restored = true;
    if (!m.restore(error)) return false;
    std::fprintf(stderr, "JUDGE_RESTORE request=%llu cells=%lld stages=%zu checked=1\n",
                 (unsigned long long)request, (long long)count, m.stages.size());
    return true;
}
} // namespace strata::program
