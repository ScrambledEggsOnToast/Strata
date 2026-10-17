// Host-only validation fixtures. CUDA is linked but must never be initialized:
// invalid restores must return before the first CUDA call or destination write.
#include "strata/core/conversation_snapshot.hpp"
#include "strata/core/resident_memory.hpp"
#include "strata/kernels/kv_q4.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <cstring>
#include <vector>

#if defined(CONVERSATION_TEST_TRANSFERS)
#include "strata/core/native_head.hpp"
#include "strata/core/mtp.hpp"
#include "strata/program/draft_kv_plan.hpp"
#include <memory>

// Execute the production horizon/dispatch/pruning statements. Only the GPU
// arithmetic is replaced: each selected draft row writes a nonzero host marker.
namespace {
namespace shared_draft_fixture {
struct DraftRows {
    int64_t prompt_len_ = 0, window_ = 32768, max_t_ = 8;
    bool per_group_sync = false;
    int token_calls = 0;
    strata::core::QsaState state;
    std::function<void(int64_t, int64_t)> write;
    strata::core::ConversationDraftCoverage coverage;
    int64_t boundary_position = -1;
    void set_boundary_capture(int64_t position) { boundary_position = position; }
    void set_prompt_len(int64_t n) { prompt_len_ = n; }
    int64_t first_needed(int = 8) const {
#include "shared_draft_first_needed.inc"
        return first_needed;
    }
    const strata::core::QsaState& kv_state() const { return state; }
    bool prefill(const float*, const int32_t* tokens, int64_t n, int64_t cell0, std::string&) {
        ++token_calls;
        const int64_t first_needed = this->first_needed();
        if (per_group_sync) {
#include "shared_draft_sync_range.inc"
            write(cell0 + c, T);
            coverage.completed(cell0 + c, cell0 + c + T, tokens[c + T - 1]);
            }
        } else {
#include "shared_draft_async_range.inc"
            write(cell0 + c, T);
            coverage.completed(cell0 + c, cell0 + c + T, tokens[c + T - 1]);
            }
        }
        return true;
    }
};

namespace strata {
namespace core = ::strata::core;
namespace program = ::strata::program;
namespace prefill {
class Prefill {
public:
    bool accept_batched = false;
    int batched_calls = 0;
    std::function<bool(const float*, int64_t, int64_t, std::string&)> on_chunk;
    bool draft_kv(DraftRows& mtp, const float*, const int32_t* tokens, int64_t n, int64_t cell0,
                  std::string&, const char**) {
        ++batched_calls;
        if (!accept_batched) return false;
        constexpr int slot = 8;
#include "shared_draft_batched_range.inc"
        mtp.write(cell0 + r0, n - r0);
        mtp.coverage.completed(cell0 + r0, cell0 + n, r0 < n ? tokens[n - 1] : -1);
        return true;
    }
};
}
}
}
}
#endif
#if defined(CONVERSATION_TEST_TRANSFERS)
// GNU/ELF link wrapping exercises the actual restore control flow without
// initializing CUDA or deliberately poisoning a real device context. This is
// host-backend fault injection, not a claim of hardware-failure recovery.
namespace {
int copy_calls = 0, sync_calls = 0, stream_sync_calls = 0, fail_copy = 0, fail_sync = 0;
int residency_sync_calls = 0;
bool defer_residency_copies = false, fail_residency_sync = false;
bool fail_mtp_stream = false;
struct PendingCopy { void* dst; const void* src; size_t bytes; };
std::vector<PendingCopy> pending_residency_copies;
size_t copied_bytes = 0;
}
extern "C" cudaError_t __wrap_cudaMemcpy(void* dst, const void* src, size_t n, cudaMemcpyKind) {
    if (++copy_calls == fail_copy) return cudaErrorInvalidValue;
    copied_bytes += n;
    std::memcpy(dst, src, n);
    return cudaSuccess;
}
// the snapshot/checkpoint copies go through a private non-blocking stream (conversation_copy.hpp)
extern "C" cudaError_t __wrap_cudaGetDevice(int* dev) { *dev = 0; return cudaSuccess; }
extern "C" cudaError_t __wrap_cudaStreamCreateWithFlags(cudaStream_t* s, unsigned int) {
    *s = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(1)); return cudaSuccess;
}
extern "C" cudaError_t __wrap_cudaMemcpyAsync(void* dst, const void* src, size_t n, cudaMemcpyKind, cudaStream_t stream) {
    if (!stream && defer_residency_copies) {
        if (++copy_calls == fail_copy) return cudaErrorInvalidValue;
        copied_bytes += n;
        pending_residency_copies.push_back({dst, src, n});
        return cudaSuccess;
    }
    return __wrap_cudaMemcpy(dst, src, n, cudaMemcpyDefault);
}
extern "C" cudaError_t __wrap_cudaStreamSynchronize(cudaStream_t stream) {
    ++stream_sync_calls;
    if (stream == reinterpret_cast<cudaStream_t>(uintptr_t(2)) && fail_mtp_stream) return cudaErrorUnknown;
    if (!stream) {
        ++residency_sync_calls;
        if (fail_residency_sync) return cudaErrorUnknown;
        for (const auto& copy : pending_residency_copies) std::memcpy(copy.dst, copy.src, copy.bytes);
        pending_residency_copies.clear();
    }
    return cudaSuccess;
}
extern "C" cudaError_t __wrap_cudaDeviceSynchronize() {
    return ++sync_calls == fail_sync ? cudaErrorUnknown : cudaSuccess;
}
extern "C" cudaError_t __wrap_cudaGetLastError() { return cudaSuccess; }
#endif

#if defined(CONVERSATION_TEST_TRANSFERS)
namespace strata::core {
// Host-transfer backend: exercise the real Operation/detach/idle state machine,
// without model arithmetic, a CUDA context, or claims of device fault recovery.
struct MtpDrafterCertificationFixture {
    static void run(void (*check)(bool, const char*)) {
        MtpDrafter draft;
        std::string error;
        draft.cs_ = reinterpret_cast<cudaStream_t>(uintptr_t(2));
        draft.working_request_ = 17;
        {
            MtpDrafter::Operation operation(draft, error);
            check(bool(operation), "acquire actual draft operation in host backend");
            operation.writes(4, 8, 11);
            check(!draft.completed_coverage().known(), "submitted writes are not completed certification");
            operation.detach();
        }
        check(!draft.completed_coverage().known(), "detached async prefill cannot certify on return");
        check(draft.idle(error) && draft.completed_coverage().begin == 4 &&
              draft.completed_coverage().cells == 8,
              "logical drain certifies detached writes before any subsequent operation/capture");
        {
            MtpDrafter::Operation operation(draft, error);
            check(bool(operation), "reacquire drained async scratch");
            operation.writes(12, 16, 23);
            operation.detach();
        }
        check(draft.idle(error) && draft.completed_coverage().begin == 12 &&
              draft.completed_coverage().cells == 16, "async completion never unions a hole");
        draft.kv_restore(16, 23);
        check(draft.completed_coverage().cells == 16, "matching private boundary keeps actual completed tail");
        draft.kv_restore(16, 24);
        check(draft.completed_coverage().cells == 15 && draft.completed_coverage().next_token == -1,
              "mismatching private boundary drops dependent tail instead of pretending full reuse");
        SavedConversation unknown;
        draft.restored_snapshot(unknown, 16);
        check(!draft.completed_coverage().known(), "unknown restored image replaces old ambient certification");
        {
            MtpDrafter::Operation operation(draft, error);
            operation.writes(0, 4, 19);
            operation.detach();
        }
        fail_mtp_stream = true;
        check(!draft.idle(error) && !draft.completed_coverage().known() && !draft.context_idle(error),
              "failed async drain clears evidence and retains poisoned ownership");
        fail_mtp_stream = false;
        draft.cs_ = nullptr; // a host fake stream is never passed to CUDA destruction
        MtpDrafter cancelled;
        cancelled.cs_ = reinterpret_cast<cudaStream_t>(uintptr_t(2));
        cancelled.coverage_ = {0, 8, 11};
        {
            MtpDrafter::Operation operation(cancelled, error);
            operation.writes(4, 6, 13);
            // Early return/cancellation without finish is the actual failure path.
        }
        check(!cancelled.completed_coverage().known() && !cancelled.context_idle(error),
              "abandoned operation drains but cannot retain stale prior certification");
        cancelled.cs_ = nullptr;
        // Coverage belongs to the request that wrote it: a later bind without its own completed
        // writes must not advertise the earlier request's rows as this request's own.
        MtpDrafter rebound;
        rebound.coverage_ = {0, 8, 11};
        rebound.coverage_owner_ = 41;
        rebound.working_request_ = 42;
        {
            const std::vector<int32_t> ids(size_t(16), 7);
            const std::vector<ConversationImageKey> no_images;
            const std::vector<ConversationCheckpoint> no_checks;
            ConversationView view{ids, no_images, no_checks, true};
            rebound.certify_view(view);
            check(view.draft_owner == 0 && view.draft_begin == -1 && view.draft_tokens == -1,
                  "a later request's bind leaves the earlier request's coverage unadvertised");
            rebound.working_request_ = 41;
            rebound.certify_view(view);
            check(view.draft_owner == 41 && view.draft_begin == 0 && view.draft_tokens == 8,
                  "the request that wrote the coverage can still certify it");
        }
    }
};
}
#endif

using namespace strata::core;
namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
struct Pools {
    QsaState st;
    std::array<std::vector<uint8_t>, 8> data;
    Pools(const ModelGeometry& g, int format, int64_t cells = 96) {
        st.max_cells = cells; st.n_pages = st.n_slots = cells / 4; st.idx_pooled_rows = cells / 4 + 2;
        st.kv_int8 = format == 1; st.kv_q4 = format == 2;
        st.kv_hybrid = format == 3;
        const size_t per = format == 2 ? strata::kernels::kv_q4_bytes_per_head((int) g.head_dim)
                                      : g.head_dim * (format == 1 || format == 3 ? 1 : 2);
        data[0].resize(cells * g.n_head_kv * per, 0xa5);
        data[1].resize(format == 3 ? cells * g.n_head_kv *
                       strata::kernels::kv_q4_bytes_per_head((int) g.head_dim) : data[0].size(), 0xa5);
        data[2].resize(format == 1 || format == 3 ? cells * g.n_head_kv * (g.head_dim / 64) * 2 : 0, 0xa5);
        if (format == 1) data[3] = data[2];
        data[4].resize(st.idx_pooled_rows * g.idx_key_dim * 4, 0xa5);
        data[5].resize(3 * g.idx_key_dim * 4, 0xa5);
        data[6].resize(g.idx_key_dim * 4, 0xa5); data[7].resize(4, 0xa5);
        if (format == 3) {
            st.k_q = (int8_t*) data[0].data(); st.v_q4 = data[1].data();
            st.k_scale = (uint16_t*) data[2].data();
        } else if (format == 2) { st.k_q4 = data[0].data(); st.v_q4 = data[1].data(); }
        else if (format == 1) {
            st.k_q = (int8_t*) data[0].data(); st.v_q = (int8_t*) data[1].data();
            st.k_scale = (uint16_t*) data[2].data(); st.v_scale = (uint16_t*) data[3].data();
        } else { st.k_pool = (uint16_t*) data[0].data(); st.v_pool = (uint16_t*) data[1].data(); }
        st.idx_pooled = (float*) data[4].data(); st.idx_tail = (float*) data[5].data();
        st.idx_dead = (float*) data[6].data(); st.idx_block_pos = (int32_t*) data[7].data();
    }
    ConversationKv image(const ModelGeometry& g, bool index) const {
        ConversationKv k;
        k.format = qsa_kv_format(st); k.cells = 12; k.page_size = 4;
        k.heads = g.n_head_kv; k.head_dim = g.head_dim; k.idx_dim = g.idx_key_dim;
        k.pooled_rows = index ? 3 : 0;
        k.k.resize(data[0].size() / 8, 13); k.v = k.k;
        k.k_scale.resize(data[2].size() / 8, 13); k.v_scale = k.k_scale;
        k.pooled.resize(k.pooled_rows * g.idx_key_dim * 4, 13);
        return k;
    }
};

void fixture(int format, int experts, bool zero_qsa, bool ple) {
    ModelGeometry g;
    g.n_layers = zero_qsa ? 3 : 8; g.n_expert = experts;
    g.ssm_state_size = 2; g.ssm_v_heads = 2; g.ssm_conv_channels = 8;
    g.n_head_kv = 1; g.head_dim = 64; g.idx_key_dim = 8;
    std::string error;
    ConversationStateSizes z;
    check(conversation_state_sizes(g, z, error), "checked canonical/pruned/zero-QSA sizing");
    Pools first(g, format), last(g, format), draft(g, format);
    std::array<QsaState, 2> layers{first.st, last.st};
    std::vector<uint8_t> gdn(z.gdn, 0xa5), history(ple ? z.ple : 0, 0xa5);
    SessionState ss;
    ss.max_cells = 96; ss.gdn_state = (float*) gdn.data();
    // the whole-model carve, as session_init leaves a one-GPU session
    ss.layer_hi = g.n_layers; ss.gdn_alloc = g.n_gdn_layers(); ss.qsa_alloc = g.n_qsa_layers();
    ss.ple_hist = ple ? (float*) history.data() : nullptr;
    ss.qsa_states = zero_qsa ? nullptr : layers.data();
    SavedConversation image;
    image.geometry = {g.n_embd,g.n_layers,g.qsa_interval,g.ssm_state_size,g.ssm_k_heads,g.ssm_v_heads,
                      g.ssm_d_conv,g.ssm_conv_channels,g.ssm_value_dim,g.n_head,g.n_head_kv,g.head_dim,
                      g.idx_q_heads,g.idx_key_dim,g.hc,g.hc_lr,g.n_expert,g.n_ff};
    image.layer_hi = g.n_layers;
    auto checkpoint = [&](size_t tokens) {
        ConversationCheckpoint c;
        c.ids.resize(tokens);
        for (size_t i = 0; i < tokens; ++i) c.ids[i] = (int32_t) i + 1;
        c.imgs = {{1, 44}};
        c.gdn.resize(z.gdn, 13); c.ple.resize(ple ? z.ple : 0, 13);
        c.tails.resize(g.n_qsa_layers() * z.tail, 13); c.dead.resize(g.n_qsa_layers() * z.dead, 13);
        c.block_pos.resize(g.n_qsa_layers() * z.block_pos, 13);
        return c;
    };
    image.live = checkpoint(9); image.checkpoints.push_back(checkpoint(5));
    image.checkpoints[0].gdn.assign(z.gdn, 17);
    image.checkpoints[0].dead.assign(g.n_qsa_layers() * z.dead, 19);
    if (immutable_checkpoints_enabled()) { image.live.seal(); image.checkpoints[0].seal(); }
    image.kv.reserve((size_t) g.n_qsa_layers() + 1);
    if (!zero_qsa) { image.kv.push_back(first.image(g, true)); image.kv.push_back(last.image(g, true)); }
    image.kv.push_back(draft.image(g, false));
    check(conversation_snapshot_validate(image, ss, g, draft.st, error), "complete image validates without CUDA");
    size_t estimate = 0;
    check(conversation_snapshot_bytes({image.live.ids,image.live.imgs,image.checkpoints,true},ss,g,draft.st,estimate,error),
          "capture estimate works without CUDA");
    check(estimate == image.bytes(), "estimate covers checkpoint/indexer/spare payloads");
    auto unchanged = [&] {
        auto pristine = [](const auto& bytes) { return std::all_of(bytes.begin(),bytes.end(),[](uint8_t b){return b==0xa5;}); };
        if (!pristine(gdn) || !pristine(history) || ss.ple_prev[0] != -1 || ss.ple_prev[1] != -1) return false;
        for (const auto* p : {&first,&last,&draft}) for (const auto& bytes : p->data) if (!pristine(bytes)) return false;
        return true;
    };
    auto reject = [&](const std::function<void(SavedConversation&)>& mutate, const char* label) {
        auto bad = image; mutate(bad);
        check(conversation_snapshot_restore(bad,ss,g,draft.st,error) == ConversationRestore::invalid,label);
        check(unchanged(), "invalid restore did not touch any layer or running-state buffer");
    };
    reject([](auto& s){s.kv.back().k.pop_back();}, "late draft corruption rejected before first layer write");
    reject([](auto& s){s.geometry[16] = 128;}, "different expert geometry rejected");
    reject([](auto& s){s.live.stage_parts.emplace_back();}, "layer-split live state rejected before writes");
    reject([](auto& s){s.checkpoints[0].stage_parts.emplace_back();}, "layer-split checkpoint rejected before writes");
    reject([](auto& s){s.draft_cells = 0;}, "partially unknown certification is rejected");
    reject([](auto& s){s.draft_begin = 4; s.draft_cells = 3;}, "inverted completed interval is rejected");
    reject([](auto& s){s.draft_begin = 0; s.draft_cells = 10;}, "coverage beyond live main extent is rejected");
    reject([](auto& s){s.draft_begin = 0; s.draft_cells = 9;},
           "full private extent without its actual next token is rejected");
    reject([](auto& s){s.draft_begin = 0; s.draft_cells = 8; s.draft_next_token = 11;},
           "partial interval cannot persist a private boundary token");
    reject([](auto& s){s.draft_boundary_residual.resize(4);},
           "an unknown image cannot attach an arbitrary residual");
    reject([](auto& s){s.kv.pop_back();}, "missing KV layer rejected");
    reject([](auto& s){s.live.gdn.pop_back();}, "bad live recurrence rejected");
    reject([](auto& s){s.checkpoints.back().gdn.pop_back();}, "bad retained checkpoint rejected");
    reject([](auto& s){s.checkpoints.back().ids[0] = 99;}, "foreign checkpoint prefix rejected");
    reject([](auto& s){s.checkpoints.back().imgs[0].hash++;}, "foreign checkpoint image rejected");
    reject([](auto& s){s.live.imgs[0].start = -1;}, "negative image position rejected");
    reject([](auto& s){s.live.ids[0] = -1;}, "negative live token rejected");
    reject([](auto& s){s.live.ple.push_back(0);}, "PLE size/enabled mismatch rejected");
    reject([](auto& s){s.live.tails.push_back(0);}, "tail size/zero-QSA mismatch rejected");
    if (!zero_qsa) {
        reject([](auto& s){s.kv[1].pooled.pop_back();}, "late indexer spare row corruption rejected");
        reject([](auto& s){s.live.dead.pop_back();}, "missing indexer spare key rejected");
        reject([](auto& s){s.live.block_pos.pop_back();}, "missing indexer metadata rejected");
        ss.qsa_states = nullptr;
        check(!conversation_snapshot_validate(image,ss,g,draft.st,error), "missing QSA targets rejected");
        ss.qsa_states = layers.data();
        reject([](auto& s){s.layer_lo = 4;}, "snapshot from another layer range rejected");
        // #216's carve: a split stage owning layers [4, 8) holds GDN rows 3..5 and QSA ordinal 1 only
        SessionState stage = ss;
        stage.layer_lo = 4; stage.gdn_alloc = 3; stage.qsa_ord0 = 1; stage.qsa_alloc = 1;
        ConversationStateSizes zs;
        check(conversation_session_sizes(g,stage,zs,error) && zs.gdn == z.gdn / 6 * 3 && zs.tail == z.tail,
              "carved session sizes cover the owned rows only");
        check(!conversation_checkpoint_validate(image.live,stage,g,error), "whole-model checkpoint rejected by a carve");
        ConversationCheckpoint part = image.live;
        part.gdn.resize(zs.gdn); part.tails.resize(zs.tail); part.dead.resize(zs.dead);
        part.block_pos.resize(zs.block_pos);
        check(conversation_checkpoint_validate(part,stage,g,error), "carve-sized checkpoint validates");
        check(!conversation_snapshot_validate(image,stage,g,draft.st,error), "whole-model snapshot rejected by a carve");
        stage.qsa_alloc = 2;
        check(!conversation_session_sizes(g,stage,zs,error), "carve past the last QSA ordinal rejected");
        stage.qsa_alloc = 1; stage.gdn_alloc = 7;
        check(!conversation_session_sizes(g,stage,zs,error), "carve past the GDN rows rejected");
    }
    auto bad_geometry = g; bad_geometry.ssm_state_size = std::numeric_limits<int64_t>::max();
    check(!conversation_state_sizes(bad_geometry,z,error), "running-state arithmetic overflow rejected");
    bad_geometry = g; bad_geometry.qsa_interval = 0;
    check(!conversation_state_sizes(bad_geometry,z,error), "zero layer interval rejected before division");
    bad_geometry = g; bad_geometry.n_head_kv = std::numeric_limits<int64_t>::max();
    check(conversation_kv_bytes(draft.st,bad_geometry,9,false)==0, "KV byte overflow rejected");
    {
        auto foreign=ss;
        foreign.gdn_ord0++;
        check(!conversation_session_copy(ss,foreign,g,image.live.ids,nullptr,error) && unchanged(),
              "direct session refuses a foreign carve before any CUDA call or write");
        auto short_target=draft.st;
        short_target.max_cells=8;
        check(!conversation_kv_copy(first.st,short_target,g,9,false,error) && unchanged(),
              "direct K/V refuses an undersized destination without fallback");
        if (!zero_qsa) {
            Pools source_first(g,format), source_last(g,format);
            for (auto* p : {&source_first,&source_last})
                for (auto& bytes : p->data) std::fill(bytes.begin(),bytes.end(),0x3c);
            std::array<QsaState,2> source_layers{source_first.st,source_last.st};
            std::vector<uint8_t> source_gdn(gdn.size(),0x3c), source_history(history.size(),0x3c);
            auto source=ss;
            source.qsa_states=source_layers.data(); source.gdn_state=(float*)source_gdn.data();
            source.ple_hist=ple?(float*)source_history.data():nullptr;
            source.ple_prev[0]=7; source.ple_prev[1]=8;
            for (bool invalid_target : {false,true}) {
                auto& late=invalid_target?layers.back():source_layers.back();
                const auto saved=late;
                late.kv_q4=true; late.kv_hybrid=true;
#if defined(CONVERSATION_TEST_TRANSFERS)
                copy_calls=sync_calls=stream_sync_calls=0;
#endif
                check(!conversation_session_copy(source,ss,g,image.live.ids,nullptr,error) && unchanged(),
                      "late direct source/target rejection leaves distinct destination bytes unchanged");
#if defined(CONVERSATION_TEST_TRANSFERS)
                check(!copy_calls && !sync_calls && !stream_sync_calls,
                      "late direct source/target rejection performs zero transfers or synchronizations");
#endif
                late=saved;
            }
        }
    }
#if defined(CONVERSATION_TEST_TRANSFERS)
    auto reset = [&] {
        for (auto* p : {&first,&last,&draft}) for (auto& bytes : p->data) std::fill(bytes.begin(),bytes.end(),0xa5);
        std::fill(gdn.begin(),gdn.end(),0xa5); std::fill(history.begin(),history.end(),0xa5);
        ss.ple_prev[0] = ss.ple_prev[1] = -1;
        copy_calls = sync_calls = stream_sync_calls = fail_copy = fail_sync = 0;
        residency_sync_calls = 0; fail_residency_sync = false;
    };
    reset(); fail_sync = 1;
    check(conversation_snapshot_restore(image,ss,g,draft.st,error)==ConversationRestore::transfer_failed,
          "pre-transfer CUDA synchronization failure is fatal");
    check(copy_calls==0 && unchanged(), "pre-sync failure did not mutate state");
    reset(); fail_copy = 1;
    check(conversation_snapshot_restore(image,ss,g,draft.st,error)==ConversationRestore::transfer_failed,
          "first CUDA transfer failure is fatal");
    check(unchanged(), "first-copy failure did not mutate state");
    reset(); fail_copy = 2;
    check(conversation_snapshot_restore(image,ss,g,draft.st,error)==ConversationRestore::transfer_failed,
          "partial CUDA transfer failure is fatal, not an invalid-image fallback");
    check(!unchanged(), "fault fixture genuinely produced partial state");
    reset(); fail_sync = 2;
    check(conversation_snapshot_restore(image,ss,g,draft.st,error)==ConversationRestore::transfer_failed,
          "post-transfer CUDA synchronization failure is fatal");
    check(copy_calls>0 && !unchanged(), "post-sync failure occurred after state writes");
    reset();
    check(conversation_snapshot_restore(image,ss,g,draft.st,error)==ConversationRestore::restored,
          "injected host transfer backend completes a valid restore");
    check(!residency_sync_calls, "fully resident snapshot restore adds no legacy-stream fence");
    check(gdn==image.live.gdn && history==image.live.ple && ss.ple_prev[0]==8 && ss.ple_prev[1]==9,
          "successful restore publishes correct running state and PLE window");
    uint64_t fingerprint = 0;
    check(conversation_kv_verify(image.kv.back(),draft.st,g,9,false,fingerprint,error), "read-back verifies complete draft payload");
    draft.data[0][0] ^= 1;
    check(!conversation_kv_verify(image.kv.back(),draft.st,g,9,false,fingerprint,error), "read-back detects corrupted draft byte");
    draft.data[0][0] ^= 1;
    if (experts==256 && !zero_qsa && ple) {
        // The main layers remain resident; only this draft needs a stream-0
        // ring repair. A failure must stop before publishing running state.
        const auto saved = draft.st;
        Pools authoritative(g,format);
        std::array<int32_t,24> table{};
        draft.st.kv_mode=2; draft.st.n_slots=2; draft.st.page_table=table.data();
        draft.st.host.k_pool=authoritative.st.k_pool; draft.st.host.v_pool=authoritative.st.v_pool;
        draft.st.host.k_q=authoritative.st.k_q; draft.st.host.v_q=authoritative.st.v_q;
        draft.st.host.k_scale=authoritative.st.k_scale; draft.st.host.v_scale=authoritative.st.v_scale;
        draft.st.host.k_q4=authoritative.st.k_q4; draft.st.host.v_q4=authoritative.st.v_q4;
        reset(); fail_residency_sync=true;
        check(conversation_snapshot_restore(image,ss,g,draft.st,error)==ConversationRestore::transfer_failed &&
              residency_sync_calls==1 && sync_calls==1 && ss.ple_prev[0]==-1 &&
              error.find("residency synchronize")!=std::string::npos,
              "snapshot propagates the draft residency fence failure before restoring running state");
        reset();
        check(conversation_snapshot_restore(image,ss,g,draft.st,error)==ConversationRestore::restored &&
              residency_sync_calls==1,
              "snapshot completes its draft repair with exactly one legacy-stream fence");
        draft.st=saved;
    }
    fail_copy = copy_calls + 1;
    check(!conversation_kv_verify(image.kv.back(),draft.st,g,9,false,fingerprint,error), "read-back transfer failure is not a successful fingerprint");
    reset(); copied_bytes = 0;
    check(conversation_snapshot_restore(image,ss,g,&draft.st,error,&image.checkpoints[0])==ConversationRestore::restored,
          "selected checkpoint restores directly from its longer parent");
    const size_t cut_copies = copied_bytes;
    check(gdn==image.checkpoints[0].gdn && history==image.checkpoints[0].ple &&
          ss.ple_prev[0]==4 && ss.ple_prev[1]==5, "direct cut installs recurrent and PLE state at selected position");
    check(conversation_kv_verify(image.kv.back(),draft.st,g,5,false,fingerprint,error,9),
          "direct cut restores only the selected draft extent");
    const size_t draft_cut_bytes = draft.data[0].size() / 96 * 8;
    check(draft.data[0][draft_cut_bytes]==0xa5, "direct cut leaves suffix past rounded page untouched");
    if (!zero_qsa) {
        const auto& c = image.checkpoints[0];
        check(std::memcmp(first.data[4].data()+g.idx_key_dim*4,c.dead.data(),z.dead)==0,
              "selected checkpoint repairs moving pooled row");
    }
    reset(); copied_bytes = 0;
    check(conversation_snapshot_restore(image,ss,g,draft.st,error)==ConversationRestore::restored &&
          conversation_checkpoint_restore(image.checkpoints[0],ss,g,error), "baseline whole image then rewind");
    check(cut_copies < copied_bytes, "direct cut removes transfer bytes versus full image plus rewind");
    check(gdn==image.checkpoints[0].gdn && history==image.checkpoints[0].ple &&
          ss.ple_prev[0]==4 && ss.ple_prev[1]==5, "baseline rewind reaches the same selected running state");
    reset();
    auto foreign_cut = image.checkpoints[0];
    check(conversation_snapshot_restore(image,ss,g,&draft.st,error,&foreign_cut)==ConversationRestore::invalid &&
          !copy_calls && !sync_calls && unchanged(), "foreign checkpoint rejected before any CUDA or writes");
    reset(); fail_copy = 2;
    check(conversation_snapshot_restore(image,ss,g,&draft.st,error,&image.checkpoints[0])==ConversationRestore::transfer_failed,
          "direct cut partial transfer failure is fatal");
    reset(); copied_bytes = 0;
    const ConversationView view{image.live.ids,image.live.imgs,image.checkpoints,true};
    SavedConversation full,incremental;
    check(conversation_snapshot_save(full,view,ss,g,draft.st,error), "full capture through host transfer backend");
    const size_t full_copies = copied_bytes;
    check((full.checkpoints[0].gdn.data()==image.checkpoints[0].gdn.data()) == immutable_checkpoints_enabled(),
          "capture shares immutable checkpoint payload only in the enabled arm");
    check(full.checkpoints[0].gdn==image.checkpoints[0].gdn,
          "capture preserves checkpoint payload in either ownership arm");
    ConversationKvReuse reuse{full.kv,9,9};
    size_t peak = 0, reused = 0;
    check(conversation_snapshot_capture_bytes(reuse,view,ss,g,draft.st,peak,error), "incremental capture admission without transfers");
    copied_bytes = 0;
    const bool saved_incrementally = conversation_snapshot_save(incremental,view,ss,g,draft.st,error,std::move(reuse),&reused);
    if (!saved_incrementally) std::fprintf(stderr,"capture error: %s\n",error.c_str());
    check(saved_incrementally, "incremental capture through host backend");
    check(reused > 0 && copied_bytes + reused == full_copies, "reused byte count measures transfers actually omitted");
    check(incremental.bytes() <= peak && incremental.live.gdn == full.live.gdn, "running state refreshed within admitted allocation");
    for (size_t i=0;i<full.kv.size();++i)
        check(incremental.kv[i].k==full.kv[i].k && incremental.kv[i].v==full.kv[i].v &&
              incremental.kv[i].k_scale==full.kv[i].k_scale && incremental.kv[i].v_scale==full.kv[i].v_scale &&
              incremental.kv[i].pooled==full.kv[i].pooled, "incremental and full capture agree across every payload");
    reuse = {full.kv,9,9};
    reuse.kv.back().k.pop_back();
    copy_calls = sync_calls = 0;
    check(!conversation_snapshot_capture_bytes(reuse,view,ss,g,draft.st,peak,error), "malformed retained draft rejected before admission");
    check(!copy_calls && !sync_calls, "invalid retained storage performs no CUDA calls");
    reuse = {full.kv,9,9};
    fail_copy = copy_calls + 1;
    SavedConversation unpublished;
    check(!conversation_snapshot_save(unpublished,view,ss,g,draft.st,error,std::move(reuse)), "incremental capture copy failure is reported");
    check(unpublished.kv.empty() && unpublished.live.ids.empty(), "failed incremental capture cannot publish a partial image");
    reset();
    Pools direct_source(g,format);
    for (auto& bytes : direct_source.data) std::fill(bytes.begin(),bytes.end(),0x3c);
    fail_copy=2;
    check(!conversation_kv_copy(direct_source.st,draft.st,g,9,false,error),
          "direct copy reports failure after a successful key transfer");
    check(copy_calls==2 && draft.data[0][0]==0x3c && draft.data[1][0]==0xa5,
          "failed direct transfer leaves partial destination and never falls back through a host image");
    reset(); copied_bytes=0;
    std::array<uint64_t,5> parts{};
    check(conversation_kv_part_sizes(direct_source.st,g,9,false,parts,error),"price direct prefix payload");
    uint64_t expected_bytes=0;
    for (uint64_t n : parts) expected_bytes+=n;
    check(conversation_kv_copy(direct_source.st,draft.st,g,9,false,error) && copied_bytes==expected_bytes,
          "direct handoff copies each payload once, including rounded pages and scales");
    reset(); fail_sync=1;
    check(!conversation_session_copy(ss,ss,g,image.live.ids,nullptr,error) && !copy_calls && unchanged(),
          "direct session pre-copy drain failure leaves all destination state unchanged");
#endif
}

#if defined(CONVERSATION_TEST_TRANSFERS)
void ring_refusal(int format) {
    ModelGeometry g;
    g.n_head_kv=1; g.head_dim=64; g.idx_key_dim=8;
    for (bool direct : {false,true}) {
        Pools source(g,format), authoritative(g,format), ring(g,format);
        for (auto& bytes : source.data) std::fill(bytes.begin(),bytes.end(),0x3c);
        ring.st.kv_mode=2; ring.st.n_slots=2;
        std::array<int32_t,24> table{};
        ring.st.page_table=table.data();
        ring.st.host.k_pool=authoritative.st.k_pool; ring.st.host.v_pool=authoritative.st.v_pool;
        ring.st.host.k_q=authoritative.st.k_q; ring.st.host.v_q=authoritative.st.v_q;
        ring.st.host.k_scale=authoritative.st.k_scale; ring.st.host.v_scale=authoritative.st.v_scale;
        ring.st.host.k_q4=authoritative.st.k_q4; ring.st.host.v_q4=authoritative.st.v_q4;
        std::string error;
        ConversationKv image;
        check(conversation_kv_save(image,source.st,g,9,false,error),"capture ring refusal source");
        const int authoritative_calls=format==1?4:2;
        copy_calls=sync_calls=stream_sync_calls=fail_sync=0;
        // With three pages and two slots, page 1 lands at slot 1 before wrapping.
        // Refuse its V copy after its K copy succeeds. The injected error never
        // enters CUDA's last-error slot (__wrap_cudaGetLastError returns success).
        fail_copy=authoritative_calls+2;
        const bool ok=direct?conversation_kv_copy(source.st,ring.st,g,9,false,error)
                            :conversation_kv_restore(image,ring.st,g,9,false,error);
        check(!ok && error.find(cudaGetErrorString(cudaErrorInvalidValue))!=std::string::npos,
              "ring refusal preserves the actual copy error with a clear last-error slot");
        const size_t page_bytes=source.data[0].size()/24;
        check(copy_calls==fail_copy && !sync_calls &&
              authoritative.data[0][0]==0x3c && authoritative.data[1][0]==0x3c &&
              ring.data[0][page_bytes]==0x3c && ring.data[1][page_bytes]==0xa5 && ring.data[0][0]==0xa5,
              "ring failure stops after a partial repair and never submits the wrapped page");
        fail_copy=0;
    }
    Pools source(g,format), authoritative(g,format), ring(g,format);
    for (auto& bytes : source.data) std::fill(bytes.begin(),bytes.end(),0x3c);
    ring.st.kv_mode=2; ring.st.n_slots=2;
    std::array<int32_t,24> table{};
    ring.st.page_table=table.data();
    ring.st.host.k_pool=authoritative.st.k_pool; ring.st.host.v_pool=authoritative.st.v_pool;
    ring.st.host.k_q=authoritative.st.k_q; ring.st.host.v_q=authoritative.st.v_q;
    ring.st.host.k_scale=authoritative.st.k_scale; ring.st.host.v_scale=authoritative.st.v_scale;
    ring.st.host.k_q4=authoritative.st.k_q4; ring.st.host.v_q4=authoritative.st.v_q4;
    std::string error;
    ConversationKv image;
    check(conversation_kv_save(image,source.st,g,9,false,error),"capture ring completion source");
    // Model a nonblocking consumer: private-stream copy fences cannot complete
    // stream-0 repair submissions. Only the residency fence drains this queue.
    defer_residency_copies=true;
    for (bool direct : {false,true}) for (bool fail : {false,true}) {
        copy_calls=sync_calls=stream_sync_calls=residency_sync_calls=fail_copy=fail_sync=0;
        fail_residency_sync=fail;
        for (auto& bytes : ring.data) std::fill(bytes.begin(),bytes.end(),0xa5);
        const bool ok=direct?conversation_kv_copy(source.st,ring.st,g,9,false,error)
                            :conversation_kv_restore(image,ring.st,g,9,false,error);
        check(ok!=fail && residency_sync_calls==1 && !sync_calls,
              "restore and direct copy each own exactly one stream-0 repair fence");
        if (fail) {
            check(!pending_residency_copies.empty() && ring.data[0][0]==0xa5 &&
                  error.find("residency synchronize")!=std::string::npos &&
                  error.find(cudaGetErrorString(cudaErrorUnknown))!=std::string::npos,
                  "failed repair completion cannot return success or lose the synchronization error");
            pending_residency_copies.clear();
        } else {
            uint64_t fingerprint=0;
            check(pending_residency_copies.empty() &&
                  conversation_kv_verify(image,ring.st,g,9,false,fingerprint,error),
                  "successful restore completes wrapped ring codes and scales before a nonblocking consumer");
        }
    }
    defer_residency_copies=fail_residency_sync=false;
    copy_calls=sync_calls=stream_sync_calls=residency_sync_calls=fail_copy=fail_sync=0;
}
#endif

// The conversation cache's host price is every saving stage's part, not one whole-model state:
// conversation_checkpoint_save runs on the primary session and on each split stage's session, so every
// stage allocates its own GDN rows and indexer layers, while the whole PLE history (session.cpp
// allocates it in every session) and the checkpoint's token-ID vector are written once per stage.  No
// checkpoint carries draft/MTP state, so the cache prices none.  The carve and the per-session extents
// come from the production helpers (session_layer_counts, conversation_session_sizes); the
// wrapped-transfer arm also captures a real checkpoint per stage and measures what that allocated.
void checkpoint_payload_stages() {
    ModelGeometry g;
    g.n_layers = 8; g.ssm_state_size = 2; g.ssm_v_heads = 2; g.ssm_conv_channels = 8;
    g.n_head_kv = 1; g.head_dim = 64; g.idx_key_dim = 8;
    std::string error;
    const int64_t cells = 96;   // the fixture's max_cells: the token bound one checkpoint can reach
    const uint64_t ids = (uint64_t) cells * sizeof(int32_t);
    ConversationStateSizes whole;
    check(conversation_state_sizes(g, whole, error), "whole-model running-state sizes");
    const uint64_t per_layer = whole.tail + whole.dead + whole.block_pos;
    const uint64_t whole_model_one = whole.gdn + whole.ple + (uint64_t) g.n_qsa_layers() * per_layer + ids;
    strata::plan::ResidentCheckpointPayload price;
    check(strata::core::conversation_checkpoint_payload(g, cells, 1, nullptr, price) && price.stages == 1 &&
          price.bytes == whole_model_one, "one stage keeps the whole-model price exactly");
    // The carve session.cpp leaves the session owning [lo, hi): its own GDN rows and QSA ordinals.
    auto stage_bytes = [&](int64_t lo, int64_t hi) -> uint64_t {
        const auto layers = strata::plan::session_layer_counts(g.n_layers, g.qsa_interval, lo, hi);
        SessionState stage;
        stage.layer_lo = lo; stage.layer_hi = hi; stage.max_cells = cells;
        stage.gdn_ord0 = lo - lo / g.qsa_interval; stage.gdn_alloc = layers.gdn;
        stage.qsa_ord0 = std::min(lo / g.qsa_interval, g.n_qsa_layers() - 1); stage.qsa_alloc = layers.qsa;
        ConversationStateSizes z;
        check(conversation_session_sizes(g, stage, z, error), "a real stage carve has session sizes");
        return z.gdn + z.ple + (uint64_t) layers.qsa * (z.tail + z.dead + z.block_pos) + ids;
    };
    const int64_t two_stages[1] = {4};   // the primary session owns [0,4), the split stage [4,8)
    check(strata::core::conversation_checkpoint_payload(g, cells, 2, two_stages, price) && price.stages == 2 &&
          price.bytes == stage_bytes(0, 4) + stage_bytes(4, 8),
          "two stages price every saving stage's carve, PLE history and token IDs");
    check(price.bytes - whole_model_one == whole.ple + ids,
          "an attention-aligned split adds one PLE history and token-ID vector");
    const int64_t three_stages[2] = {2, 6};   // [0,2) owns no attention layer of its own
    check(strata::core::conversation_checkpoint_payload(g, cells, 3, three_stages, price) && price.stages == 3 &&
          price.bytes == stage_bytes(0, 2) + stage_bytes(2, 6) + stage_bytes(6, 8),
          "three stages price every saving stage, including an attention-free one");
    check(price.stage_peak == stage_bytes(2, 6) && price.bytes > whole_model_one,
          "the largest stage sets the peak and the whole-model price is insufficient");
    check(price.bytes - whole_model_one == 2 * (whole.ple + ids) + per_layer,
          "an attention-free stage also adds its primary indexer state");
    const int64_t descending[2] = {6, 2};
    check(!strata::core::conversation_checkpoint_payload(g, cells, 3, descending, price) && !price.stages,
          "a non-ascending split is refused");
    const int64_t past_last[1] = {g.n_layers + 1};
    check(!strata::core::conversation_checkpoint_payload(g, cells, 2, past_last, price) && !price.stages,
          "a boundary past the last layer is refused");
    const int64_t empty_stage[1] = {0};
    check(!strata::core::conversation_checkpoint_payload(g, cells, 2, empty_stage, price) && !price.stages,
          "an empty stage is refused");
    check(!strata::core::conversation_checkpoint_payload(g, cells, 0, nullptr, price) && !price.stages &&
          !strata::core::conversation_checkpoint_payload(g, cells, (uint64_t) g.n_layers + 1, nullptr, price) &&
          !price.stages, "no stage, or more stages than layers, is refused");
    check(!strata::core::conversation_checkpoint_payload(g, 0, 2, two_stages, price) && !price.stages &&
          !strata::core::conversation_checkpoint_payload(g, (int64_t) INT32_MAX + 1, 2, two_stages, price) &&
          !price.stages, "a zero or over-int32 cell count is refused");
    auto overflow = g;
    overflow.ssm_state_size = std::numeric_limits<int64_t>::max();
    check(!strata::core::conversation_checkpoint_payload(overflow, cells, 2, two_stages, price) && !price.stages,
          "checkpoint payload arithmetic overflow is refused");
#if defined(CONVERSATION_TEST_TRANSFERS)
    copy_calls = sync_calls = stream_sync_calls = fail_copy = fail_sync = 0;
    auto saved_bytes = [&](int64_t lo, int64_t hi) -> uint64_t {
        const auto layers = strata::plan::session_layer_counts(g.n_layers, g.qsa_interval, lo, hi);
        SessionState stage;
        stage.layer_lo = lo; stage.layer_hi = hi; stage.max_cells = cells;
        stage.gdn_ord0 = lo - lo / g.qsa_interval; stage.gdn_alloc = layers.gdn;
        stage.qsa_ord0 = std::min(lo / g.qsa_interval, g.n_qsa_layers() - 1); stage.qsa_alloc = layers.qsa;
        ConversationStateSizes z;
        check(conversation_session_sizes(g, stage, z, error), "a real stage carve has session sizes");
        Pools pool(g, 0);
        std::array<QsaState, 2> owned{pool.st, pool.st};   // global ordinals: qsa_states[ord0 + j]
        std::vector<uint8_t> gdn(z.gdn, 0xa5), history(z.ple, 0xa5);
        stage.gdn_state = (float*) gdn.data();
        stage.ple_hist = (float*) history.data();
        stage.qsa_states = owned.data();
        ConversationCheckpoint part;
        part.ids.assign((size_t) cells, 1);
        check(conversation_checkpoint_save(part, stage, g, error),
              "a stage checkpoint captures on the host transfer backend");
        return (uint64_t) part.ids.size() * sizeof(int32_t) + part.gdn.size() + part.ple.size() +
               part.tails.size() + part.dead.size() + part.block_pos.size();
    };
    check(saved_bytes(0, 2) == stage_bytes(0, 2) && saved_bytes(2, 6) == stage_bytes(2, 6) &&
          saved_bytes(6, 8) == stage_bytes(6, 8),
          "a real three-stage capture allocates exactly the priced stage payloads");
#endif
}

void resident_transfer_envelopes() {
    ModelGeometry g;
    g.n_layers = 8; g.ssm_state_size = 2; g.ssm_v_heads = 2; g.ssm_conv_channels = 8;
    g.n_head_kv = 1; g.head_dim = 64; g.idx_key_dim = 8;
    const int64_t split[] = {2, 6};
    auto shape = strata::kernels::qsa_real_shapes();
    shape.n_head_kv = g.n_head_kv; shape.head_dim = g.head_dim; shape.idx_dim = g.idx_key_dim;
    for (int fmt : {0, 1, 2, 3}) for (int64_t cells : {1, 5, 96}) {
        auto roles = strata::plan::qsa_resolve_roles(shape, cells, fmt, -1, -1, -1, 8);
        strata::plan::ResidentHostBytes legacy, direct;
        check(resident_host_payload(g, cells, true, roles, 3, split, false, 1, legacy) &&
              resident_host_payload(g, cells, true, roles, 3, split, true, 1, direct),
              "both transfer envelopes price supported formats and partial pages");
        check(direct.histories == legacy.histories && direct.checkpoints == legacy.checkpoints &&
              direct.transfer_transient > strata::plan::kResidentCheckpointMetadataBytes &&
              direct.transfer_transient < legacy.transfer_transient,
              "direct removes only serial images, retaining checkpoint/repair storage and margin");
        const auto old_format = roles.working_main.format;
        roles.working_main.format = 999;
        for (bool mode : {false, true})
            check(!resident_host_payload(g, cells, true, roles, 3, split, mode, 1, direct) &&
                  !direct.histories && !direct.checkpoints && !direct.transfer_transient,
                  "unknown main format cannot become a zero-image optimistic admission");
        roles.working_main.format = old_format;
        roles.working_draft.format = 999;
        for (bool mode : {false, true})
            check(!resident_host_payload(g, cells, true, roles, 3, split, mode, 1, direct),
                  "unknown draft format fails closed in both modes");
    }
    const auto roles = strata::plan::qsa_resolve_roles(shape, 96, 0, -1, -1, -1, 8);
    for (bool mode : {false, true}) {
        strata::plan::ResidentHostBytes host;
        check(!resident_host_payload(g, 0, true, roles, 3, split, mode, 1, host) &&
              !host.transfer_transient, "invalid context has no usable transfer price");
        const int64_t bad_split[] = {6, 2};
        check(!resident_host_payload(g, 96, true, roles, 3, bad_split, mode, 1, host),
              "invalid split cannot shrink the transfer envelope");
        auto overflow = g;
        overflow.head_dim = INT64_MAX;
        check(!resident_host_payload(overflow, 96, true, roles, 3, split, mode, 1, host) &&
              !host.transfer_transient, "direct still validates image geometry arithmetic before omitting storage");
        strata::plan::ResidentCheckpointPayload checkpoint{UINT64_MAX, UINT64_MAX, 3};
        check(!strata::plan::resident_host_bytes(96, checkpoint, 0, 0, 1, host) &&
              !host.transfer_transient, "checkpoint plus margin overflow fails closed even without images");
        auto unknown = roles;
        unknown.slot_main.kv.mode = 999;
        strata::plan::ResidentBytes bytes;
        const char* unpriced = nullptr;
        check(!resident_allocation_bytes(g, 96, 1, 2, true, unknown, mode, 1, bytes, unpriced, 3, split) &&
              unpriced && !bytes.transfer_transient,
              "unknown actual-owner placement refuses before either transfer envelope is admitted");
    }
    {
        using namespace strata::plan;
        ResidentCheckpointPayload checkpoint{64, 64, 3};
        ResidentHostBytes one, many, legacy;
        check(resident_host_bytes(96, checkpoint, 0, 0, 1, one) &&
              resident_host_bytes(96, checkpoint, 0, 0, 7, many),
              "price independently owned retained chains");
        check(many.checkpoints == 7 * one.checkpoints && many.histories == one.histories,
              "every retained root/pin/latest owns a full checkpoint without duplicating seat histories");
        const uint64_t aliases = 7 * 3 * (96 * sizeof(int32_t) + kResidentCheckpointMetadataBytes);
        check(many.transfer_transient == aliases + aliases / 8,
              "bind keeps old owners alive and prices new alias IDs, stage objects and controls");
        check(resident_host_bytes(96, checkpoint, 1000000, 0, 7, legacy) &&
              legacy.transfer_transient == 1000064 + 1000064 / 8 + kResidentCheckpointMetadataBytes,
              "legacy transfer and bind alias peaks compose by maximum, not sum");
        // Nonintegral eighths and the exact max crossover: these constants
        // come from a one-stage 65-byte payload, one cell (8 history bytes),
        // and the independently specified 4096-byte metadata envelope.
        ResidentCheckpointPayload odd{65, 65, 1};
        ResidentHostBytes below, tie, above;
        check(kResidentCheckpointMetadataBytes == 4096 &&
              resident_host_bytes(1, odd, 393, 0, 1, below) &&
              resident_host_bytes(1, odd, 394, 0, 1, tie) &&
              resident_host_bytes(1, odd, 395, 0, 1, above) &&
              below.checkpoints == 4171 && below.histories == 8 &&
              below.transfer_transient == 4613 && tie.transfer_transient == 4613 &&
              above.transfer_transient == 4614,
              "ceil(73/8)=10 and ceil(4100/8)=513; serial maximum crosses at 459 bytes");
        for (uint64_t limit : {uint64_t(0), uint64_t(INT32_MAX) + 1, UINT64_MAX})
            check(!resident_host_bytes(96, checkpoint, 0, 0, limit, many) &&
                  !many.checkpoints && !many.transfer_transient, "invalid chain limit fails closed");
        checkpoint.bytes = UINT64_MAX / 2;
        check(!resident_host_bytes(96, checkpoint, 0, 0, 7, many) && !many.checkpoints,
              "retained chain multiplication overflow fails closed");
    }
    // The production caller resolves this before pricing. Later environment
    // mutation must not switch ResidentSlots/MTP to an allocating legacy branch.
    const char* value = std::getenv("STRATA_RESIDENT_DIRECT_COPY");
    const bool had_value = value != nullptr;
    const bool expected = value && std::strcmp(value, "1") == 0;
    const std::string original = value ? value : "";
    check(resident_direct_copy_enabled() == expected, "only exact one selects direct handoff at process resolution");
#ifdef _WIN32
    _putenv_s("STRATA_RESIDENT_DIRECT_COPY", expected ? "0" : "1");
#else
    setenv("STRATA_RESIDENT_DIRECT_COPY", expected ? "0" : "1", 1);
#endif
    check(resident_direct_copy_enabled() == expected, "pricing and later runtime share one immutable selection");
#ifdef _WIN32
    _putenv_s("STRATA_RESIDENT_DIRECT_COPY", original.c_str());
#else
    if (had_value) setenv("STRATA_RESIDENT_DIRECT_COPY", original.c_str(), 1);
    else unsetenv("STRATA_RESIDENT_DIRECT_COPY");
#endif
}
#if defined(CONVERSATION_TEST_TRANSFERS)
namespace shared_draft_fixture {
int64_t draft_horizon(DraftRows& mtp, bool shared, int64_t resume, int64_t reread_to,
                      int64_t req_pin = 8192, int64_t n = 65536) {
    const bool use_mtp = true, packed_prefill = true;
    const ConversationScope req_shared_scope = shared ? ConversationScope{{1}} : ConversationScope{};
#include "shared_draft_horizon.inc"
    return read_from;
}

void diagnostic_shared_draft(bool batched, bool per_group_sync, int64_t window) {
    constexpr int64_t pin = 8192, resume = 16384, prompt = 65536;
    ModelGeometry g;
    g.n_layers = 4; g.ssm_state_size = 2; g.ssm_v_heads = 2; g.ssm_conv_channels = 8;
    g.n_head_kv = 1; g.head_dim = 64; g.idx_key_dim = 8;
    Pools main(g, 0, resume), draft(g, 0, resume);
    // A diagnostic reread starts with no valid draft rows, not a restored ring.
    std::fill(draft.data[0].begin(), draft.data[0].end(), 0);
    std::fill(draft.data[1].begin(), draft.data[1].end(), 0);
    DraftRows mtp;
    mtp.window_ = window;
    mtp.per_group_sync = per_group_sync;
    mtp.write = [&](int64_t at, int64_t count) {
        const size_t row = size_t(g.n_head_kv * g.head_dim * 2);
        for (size_t part : {size_t(0), size_t(1)})
            std::fill(draft.data[part].begin() + size_t(at) * row,
                      draft.data[part].begin() + size_t(at + count) * row, 0x6d);
    };
    // Private serving keeps the full-prompt horizon, including diagnostic reads;
    // a normal private checkpoint at/past the pin does not create a shared cut.
    check(draft_horizon(mtp, false, resume, resume) == 0 && mtp.prompt_len_ == prompt,
          "private diagnostic reread keeps its ordinary full-prompt horizon");
    check(draft_horizon(mtp, true, resume, -1) == resume && mtp.prompt_len_ == prompt,
          "private resume past the shared pin keeps the ordinary horizon");
    check(draft_horizon(mtp, true, pin, -1) == pin && mtp.prompt_len_ == prompt,
          "a resume exactly at the pin does not recompute or capture that cut");
    const int64_t read_from = draft_horizon(mtp, true, resume, resume);
    const int64_t required_begin = std::max<int64_t>(0, pin + 1 - window - 64);
    check(read_from == 0, "a diagnostic reread starts at zero despite selecting a deeper private checkpoint");

    strata::prefill::Prefill sp;
    sp.accept_batched = batched;
    struct Stage { strata::prefill::Prefill sp; int dev = 0; };
    const std::vector<std::unique_ptr<Stage>> stages;
    const bool use_mtp = true, multi_gpu = false, split_mtp_batch = false;
    const std::vector<int64_t> cur(size_t(prompt), 11);
#include "shared_draft_callback.inc"
        return true;
    };
    std::string error;
    for (int64_t at = read_from; at < pin; at += 2048)
        check(sp.on_chunk(nullptr, 2048, at, error) && error.empty(),
              "the production prompt callback populates draft state up to the pin");
    check(sp.batched_calls == 4 && mtp.token_calls == (batched ? 0 : 4),
          "both the E-9 batched callback and its per-group fallback are exercised");
    const size_t row = size_t(g.n_head_kv * g.head_dim * 2);

    ConversationStateSizes z;
    check(conversation_state_sizes(g, z, error), "diagnostic fixture prices recurrent state");
    std::vector<uint8_t> gdn(z.gdn, 0x45), ple(z.ple, 0x46);
    SessionState ss;
    ss.max_cells = resume; ss.layer_hi = g.n_layers; ss.gdn_alloc = g.n_gdn_layers(); ss.qsa_alloc = 1;
    ss.gdn_state = reinterpret_cast<float*>(gdn.data());
    ss.ple_hist = reinterpret_cast<float*>(ple.data());
    ss.qsa_states = &main.st;
    const std::vector<int32_t> ids(size_t(pin), 11);
    const std::vector<ConversationImageKey> no_images;
    const std::vector<ConversationCheckpoint> no_checks;
    ConversationBuffer boundary;
    boundary.resize(size_t(g.n_embd * g.hc) * sizeof(float), 0x3d);
    boundary.freeze();
    check(mtp.coverage.known() && mtp.coverage.cells == pin,
          "host writers expose their actually completed contiguous interval");
    const ConversationView exact{ids, no_images, no_checks, true, true, pin - 1, mtp.coverage.begin,
                                  -1, 17, &boundary, pin - 1, 17};
    SavedConversation image;
    copy_calls = sync_calls = stream_sync_calls = fail_copy = fail_sync = 0;
    check(conversation_snapshot_save(image, exact, ss, g, &draft.st, error) &&
          conversation_snapshot_validate(image, ss, g, &draft.st, ConversationSnapshotExpectation::exact_cut, error) &&
          image.kv.back().cells == pin - 1,
          "diagnostic reread captures a valid exact N-1 draft extent");
    for (const auto* part : {&image.kv.back().k, &image.kv.back().v}) {
        bool populated = true;
        check(part->visit(size_t(required_begin) * row, size_t(pin - 1 - required_begin) * row,
              [&](const uint8_t* bytes, size_t n, size_t) {
                  populated = populated && std::all_of(bytes, bytes + n, [](uint8_t byte) { return byte == 0x6d; });
                  return true;
              }) && populated, "the accepted exact capture contains required draft rows, not valid-looking zeros");
    }
    for (size_t part : {size_t(0), size_t(1)})
        check(std::all_of(draft.data[part].begin() + size_t(required_begin) * row,
                          draft.data[part].begin() + size_t(pin - 1) * row,
                          [](uint8_t byte) { return byte == 0x6d; }),
              "every required prefix draft K/V row is recomputed, not skipped by the private suffix");
    check(mtp.prompt_len_ == pin + 1 && mtp.first_needed() == pin + 1 - window - 64,
          "diagnostic reread of a deeper private checkpoint uses the exact shared-cut horizon");
}
}

void exact_shared_cut(int format, int64_t cut) {
    ModelGeometry g;
    g.n_layers = 4; g.ssm_state_size = 2; g.ssm_v_heads = 2; g.ssm_conv_channels = 8;
    g.n_head_kv = 2; g.head_dim = 64; g.idx_key_dim = 8;
    Pools main(g, format), draft(g, format == 3 ? 1 : format);
    ConversationStateSizes z;
    std::string error;
    check(conversation_state_sizes(g, z, error), "exact cut fixture prices running state");
    std::vector<uint8_t> gdn(z.gdn, 0x45), ple(z.ple, 0x46);
    SessionState ss;
    ss.max_cells = 96; ss.layer_hi = g.n_layers; ss.gdn_alloc = g.n_gdn_layers(); ss.qsa_alloc = 1;
    ss.gdn_state = reinterpret_cast<float*>(gdn.data());
    ss.ple_hist = reinterpret_cast<float*>(ple.data());
    ss.qsa_states = &main.st;
    std::vector<int32_t> ids(size_t(cut), 11);
    const std::vector<ConversationImageKey> no_images;
    const std::vector<ConversationCheckpoint> no_checks;
    const int64_t begin = cut == 80 ? 9 : std::min<int64_t>(1, cut - 1);
    ConversationView exact{ids, no_images, no_checks, true, true, cut - 1, begin};
    ConversationBuffer boundary;
    boundary.resize(size_t(g.n_embd * g.hc) * sizeof(float), 0x3d);
    boundary.freeze();
    exact.draft_owner = 17; exact.draft_boundary_owner = 17;
    exact.draft_boundary_position = cut - 1; exact.draft_boundary_residual = &boundary;
    size_t estimate = 0;
    copy_calls = sync_calls = stream_sync_calls = fail_copy = fail_sync = 0;
    const ConversationView invalid_beyond{ids, no_images, no_checks, true, true, cut + 1, begin};
    check(!conversation_snapshot_bytes(invalid_beyond, ss, g, &draft.st, estimate, error) &&
          copy_calls == 0 && sync_calls == 0, "draft extent beyond an exact cut is refused before device access");
    const ConversationView invalid_consuming_suffix{ids, no_images, no_checks, true, true, cut, begin};
    check(!conversation_snapshot_bytes(invalid_consuming_suffix, ss, g, &draft.st, estimate, error),
          "the draft row consuming the first private suffix token cannot be claimed");
    if (cut > 1) {
        const ConversationView invalid_short{ids, no_images, no_checks, true, true, cut - 2, begin};
        check(!conversation_snapshot_bytes(invalid_short, ss, g, &draft.st, estimate, error),
              "an exact capture must own all N-1 draft rows, not an arbitrary shorter extent");
    }
    auto invalid_read = exact;
    invalid_read.draft_boundary_position = cut;
    SavedConversation refused;
    check(!conversation_snapshot_save(refused, invalid_read, ss, g, &draft.st, error) &&
          refused.live.ids.empty() && refused.kv.empty() && copy_calls == 0 && sync_calls == 0,
          "wrong-position residual produces no shared image or device access");
    invalid_read = exact; invalid_read.draft_boundary_owner = 18;
    check(!conversation_snapshot_save(refused, invalid_read, ss, g, &draft.st, error) &&
          refused.kv.empty() && copy_calls == 0 && sync_calls == 0,
          "a residual from a different request generation is rejected before writes");
    invalid_read = exact; invalid_read.draft_boundary_residual = nullptr;
    invalid_read.draft_boundary_owner = 0; invalid_read.draft_boundary_position = -1;
    check(!conversation_snapshot_bytes(invalid_read, ss, g, &draft.st, estimate, error),
          "completed KV without the final residual cannot claim a catchup-capable exact image");
    SavedConversation unsafe_meta;
    std::vector<SessionKvSource> unsafe_sources;
    check(!conversation_snapshot_sources(unsafe_meta, unsafe_sources, exact, ss, g, draft.st, error) &&
          copy_calls == 0 && sync_calls == 0, "streamed exports cannot bypass exact-cut sanitization");
    SavedConversation first, second, ordinary;
    check(conversation_snapshot_save(first, exact, ss, g, &draft.st, error),
          "capture an exact shared cut with poisoned inactive storage");
    check(first.live.ids == ids && first.checkpoints.empty() && first.kv.back().cells == cut - 1 &&
          first.live.gdn == gdn && first.live.ple == ple,
          "exact capture keeps recurrent/PLE state at N and explicitly records draft extent N-1");
    auto legacy_view = exact;
    legacy_view.draft_owner = 0; legacy_view.draft_boundary_residual = nullptr;
    legacy_view.draft_boundary_owner = 0; legacy_view.draft_boundary_position = -1;
    SavedConversation legacy;
    check(conversation_snapshot_save(legacy, legacy_view, ss, g, &draft.st, error) &&
          legacy.draft_begin == -1 && legacy.draft_cells == -1 && legacy.draft_next_token == -1 &&
          legacy.draft_boundary_residual.empty() &&
          conversation_snapshot_validate(legacy, ss, g, &draft.st, ConversationSnapshotExpectation::exact_cut, error),
          "feature-off shared capture preserves legacy unknown metadata without a new residual allocation/export");
    check(first.draft_begin == begin && first.draft_cells == cut - 1 && first.draft_next_token == -1 &&
          first.draft_boundary_residual == boundary,
          "certified exact capture carries only actual interval and immutable final main residual");
    const int64_t rounded = (cut + 3) / 4 * 4;
    const auto rows_equal = [&](const ConversationKv& kv, int64_t lo, int64_t hi) {
        for (const auto* part : {&kv.k, &kv.v, &kv.k_scale, &kv.v_scale}) {
            if (part->empty()) continue;
            const size_t width = part->size() / size_t(rounded * g.n_head_kv);
            for (int64_t cell = 0; cell < rounded; ++cell) for (int64_t head = 0; head < g.n_head_kv; ++head) {
                const size_t at = size_t((cell / 4 * g.n_head_kv + head) * 4 + cell % 4) * width;
                const uint8_t want = cell >= lo && cell < hi ? 0xa5 : 0;
                bool equal = true;
                part->visit(at, width, [&](const uint8_t* p, size_t n, size_t) {
                    equal = std::all_of(p, p + n, [&](uint8_t byte) { return byte == want; });
                    return true;
                });
                if (!equal) return false;
            }
        }
        return true;
    };
    check(rows_equal(first.kv.front(), 0, cut) && rows_equal(first.kv.back(), begin, cut - 1),
          "all heads, formats, scale runs, skipped draft cells and page tails obey the exact cut");
    const size_t active_tail = size_t(cut % 4) * size_t(g.idx_key_dim) * 4;
    check(std::all_of(first.live.tails.read().begin() + active_tail, first.live.tails.read().end(),
                      [](uint8_t b) { return b == 0; }) &&
          std::all_of(first.live.block_pos.read().begin(), first.live.block_pos.read().end(),
                      [](uint8_t b) { return b == 0; }),
          "inactive indexer tails and position scratch cannot retain another conversation");

    const auto poison_rows = [&](Pools& pools, int64_t lo, int64_t hi) {
        for (size_t part = 0; part < 4; ++part) {
            auto& bytes = pools.data[part];
            if (bytes.empty()) continue;
            const size_t width = bytes.size() / size_t(96 * g.n_head_kv);
            for (int64_t cell = 0; cell < 96; ++cell) if (cell < lo || cell >= hi)
                for (int64_t head = 0; head < g.n_head_kv; ++head) {
                    const size_t at = size_t((cell / 4 * g.n_head_kv + head) * 4 + cell % 4) * width;
                    std::fill(bytes.begin() + at, bytes.begin() + at + width, 0x9b);
                }
        }
    };
    poison_rows(main, 0, cut);
    poison_rows(draft, begin, cut - 1);
    std::fill(main.data[5].begin() + active_tail, main.data[5].end(), 0x9b);
    std::fill(main.data[7].begin(), main.data[7].end(), 0x9b);
    const size_t spare = size_t(cut / 4) * size_t(g.idx_key_dim) * 4;
    std::fill(main.data[4].begin() + spare, main.data[4].end(), 0x9b);
    check(conversation_snapshot_save(second, exact, ss, g, &draft.st, error),
          "recapture the same exact prefix after poisoning private suffix and inactive bytes differently");
    bool same = first.live.gdn == second.live.gdn && first.live.ple == second.live.ple &&
                first.live.tails == second.live.tails && first.live.dead == second.live.dead &&
                first.live.block_pos == second.live.block_pos;
    for (size_t layer = 0; layer < first.kv.size(); ++layer) {
        const auto& a = first.kv[layer]; const auto& b = second.kv[layer];
        same = same && a.k == b.k && a.v == b.v && a.k_scale == b.k_scale &&
               a.v_scale == b.v_scale && a.pooled == b.pooled;
    }
    check(same, "poisoned post-cut storage cannot influence any published shared payload byte");
    constexpr auto expectation = ConversationSnapshotExpectation::exact_cut;
    check(conversation_snapshot_validate(second, ss, g, &draft.st, expectation, error),
          "the explicit shared expectation accepts the sanitized exact cut");
    auto rounded_claim = second;
    rounded_claim.kv.back().cells = rounded;
    copy_calls = sync_calls = 0;
    uint64_t fingerprint = 0;
    check(!conversation_snapshot_validate(rounded_claim, ss, g, &draft.st, expectation, error) &&
          conversation_snapshot_restore(rounded_claim, ss, g, &draft.st, expectation, error) ==
              ConversationRestore::invalid &&
          !conversation_kv_restore(rounded_claim.kv.back(), draft.st, g, cut, false, expectation, error) &&
          !conversation_kv_verify(rounded_claim.kv.back(), draft.st, g, cut, false,
                                  expectation, fingerprint, error) &&
          copy_calls == 0 && sync_calls == 0,
          "ordinary rounded metadata cannot downgrade a shared validate, restore or diagnostic verify");
    check(conversation_snapshot_validate(rounded_claim, ss, g, &draft.st, error),
          "ordinary private validation still accepts its rounded-cell contract");
    check(rounded_claim.kv.back().k.visit(rounded_claim.kv.back().k.size() - 1, 1,
        [](uint8_t* p, size_t, size_t) { *p = 0xee; return true; }),
        "a malicious rounded record also carries a nonzero excluded byte");
    check(conversation_snapshot_validate(rounded_claim, ss, g, &draft.st, error) &&
          !conversation_snapshot_validate(rounded_claim, ss, g, &draft.st, expectation, error),
          "shared authorization, not metadata or payload shape, selects the stronger validation contract");
    if (cut > 1) {
        auto shorter_claim = second;
        shorter_claim.kv.back().cells = rounded;
        check(conversation_kv_clear_inactive(shorter_claim.kv.back(), cut, std::min(begin, cut - 2), cut - 2, error),
              "zero every row the malicious shorter claim would exclude");
        shorter_claim.kv.back().cells = cut - 2;
        check(!conversation_snapshot_validate(shorter_claim, ss, g, &draft.st, expectation, error),
              "shared restore refuses a shorter-than-N-1 draft claim even when excluded bytes are zero");
    }
    // Every excluded row, head and populated part is checked, not just the last
    // byte or a metadata-selected extent. The N-1 row consumes the private suffix.
    for (size_t layer = 0; layer < second.kv.size(); ++layer) {
        const int64_t end = layer + 1 == second.kv.size() ? cut - 1 : cut;
        for (size_t part = 0; part < 4; ++part) {
            for (int64_t cell = end; cell < rounded; ++cell) {
                for (int64_t head = 0; head < g.n_head_kv; ++head) {
                    auto corrupt = second;
                    auto& kv = corrupt.kv[layer];
                    const std::array<ConversationBuffer*, 4> buffers{&kv.k, &kv.v, &kv.k_scale, &kv.v_scale};
                    if (buffers[part]->empty()) continue;
                    const size_t width = buffers[part]->size() / size_t(rounded * g.n_head_kv);
                    const size_t at = size_t((cell / 4 * g.n_head_kv + head) * 4 + cell % 4) * width;
                    check(buffers[part]->visit(at, 1, [](uint8_t* p, size_t, size_t) {
                        *p = 0xee; return true;
                    }), "poison an excluded shared row");
                    copy_calls = sync_calls = 0;
                    check(conversation_snapshot_restore(corrupt, ss, g, &draft.st, expectation, error) ==
                              ConversationRestore::invalid && copy_calls == 0 && sync_calls == 0,
                          "nonzero excluded main/draft bytes are refused before any device call or write");
                }
            }
        }
    }
    if (cut == 9 || cut == 80) {
        // Execute the actual ordinary private restore -> divergent prefill ->
        // shared-capture chain with the host transfer backend. Only the model
        // arithmetic is synthetic: it rewrites rows [R,N-1), never R-1 or the
        // earlier retained rows skipped by the old longer-window prompt.
        const int64_t read_from = cut - 2;
        std::vector<int32_t> old_ids = ids;
        old_ids.resize(size_t(cut + 16), 31);
        old_ids[size_t(read_from)] = 37;
        std::vector<ConversationCheckpoint> old_checks(1);
        old_checks[0].ids.assign(old_ids.begin(), old_ids.begin() + read_from);
        check(conversation_checkpoint_save(old_checks[0], ss, g, error),
              "save a real private resume checkpoint before the divergent suffix");
        const ConversationView old_view{old_ids, no_images, old_checks, true};
        SavedConversation old_private;
        check(conversation_snapshot_save(old_private, old_view, ss, g, &draft.st, error) &&
              conversation_snapshot_restore(old_private, ss, g, &draft.st, error, &old_private.checkpoints[0]) ==
                  ConversationRestore::restored,
              "restore the old longer private snapshot at R without a shared expectation");
        for (size_t part = 0; part < 4; ++part) {
            auto& bytes = draft.data[part];
            if (bytes.empty()) continue;
            const size_t width = bytes.size() / size_t(96 * g.n_head_kv);
            for (int64_t cell = read_from; cell < cut - 1; ++cell)
                for (int64_t head = 0; head < g.n_head_kv; ++head) {
                    const size_t at = size_t((cell / 4 * g.n_head_kv + head) * 4 + cell % 4) * width;
                    std::fill(bytes.begin() + at, bytes.begin() + at + width, 0x6a);
                }
        }
        auto resumed_view = exact;
        resumed_view.draft_boundary_residual = nullptr;
        resumed_view.draft_boundary_owner = 0; resumed_view.draft_boundary_position = -1;
        SavedConversation suppressed;
        copy_calls = sync_calls = 0;
        check(!conversation_snapshot_save(suppressed, resumed_view, ss, g, &draft.st, error) &&
              suppressed.live.ids.empty() && suppressed.kv.empty() && !copy_calls && !sync_calls,
              "legacy partial resume without a captured final residual cannot emit a certified exact cut");
        const ConversationView private_continuation{ids, no_images, no_checks, true};
        check(conversation_snapshot_save(ordinary, private_continuation, ss, g, &draft.st, error),
              "suppressing shared export does not suppress ordinary private capture or serving state");
    }
    for (auto& bytes : draft.data) std::fill(bytes.begin(), bytes.end(), 0xee);
    check(conversation_snapshot_restore(second, ss, g, &draft.st, expectation, error) == ConversationRestore::restored,
          "explicit shared restore understands the required N-1 draft extent");
    check(conversation_kv_verify(second.kv.back(), draft.st, g, cut, false, expectation, fingerprint, error),
          "shared verify checks the zeroed excluded draft row and page tail after restore");
    const ConversationView private_view{ids, no_images, no_checks, true};
    check(conversation_snapshot_save(ordinary, private_view, ss, g, &draft.st, error) &&
          ordinary.kv.back().cells == rounded, "post-shared private captures preserve the ordinary rounded-cell contract");
    if (format == 0 && cut == 5) {
        auto unused = ss;
        unused.layer_hi = 2; unused.gdn_alloc = 2; // allocated QSA primary lends only RoPE
        const ConversationView cut_view{ids, no_images, no_checks, true, true};
        SavedConversation stage;
        check(conversation_snapshot_save(stage, cut_view, unused, g, nullptr, error),
              "capture a carved stage whose primary QSA owns no attention layer");
        const auto zeros = [](const CheckpointBytes& bytes) {
            const auto& source = bytes.read();
            return std::all_of(source.begin(), source.end(), [](uint8_t b) { return b == 0; });
        };
        const auto zero_buffer = [](const ConversationBuffer& bytes) {
            return bytes.visit(0, bytes.size(), [](const uint8_t* p, size_t n, size_t) {
                return std::all_of(p, p + n, [](uint8_t b) { return b == 0; });
            });
        };
        check(zero_buffer(stage.kv.front().k) && zero_buffer(stage.kv.front().v) && zero_buffer(stage.kv.front().pooled) &&
              zeros(stage.live.tails) && zeros(stage.live.dead),
              "inactive primary QSA payload and index state cannot carry bytes from another owner");
        check(conversation_snapshot_validate(stage, unused, g, nullptr, expectation, error),
              "an exact stage without the draft owner uses the same explicit shared expectation");
        unused = ss;
        unused.layer_lo = 2; unused.gdn_ord0 = 2; unused.gdn_alloc = 1;
        check(conversation_snapshot_save(stage, cut_view, unused, g, nullptr, error) && zeros(stage.live.ple),
              "PLE history on a stage that never runs the first layer is excluded");
        check(conversation_snapshot_validate(stage, unused, g, nullptr, expectation, error),
              "later exact stages validate without inventing a draft owner or PLE history");
    }
}
#endif

}

int main() {
    ConversationDraftCoverage coverage;
    check(!coverage.known(), "fresh draft coverage is unknown");
    coverage.completed(4, 8, 11);
    coverage.completed(8, 12, 13);
    check(coverage.begin == 4 && coverage.cells == 12 && coverage.next_token == 13,
          "adjacent completed batches form one real interval");
    coverage.completed(16, 20, 17);
    check(coverage.begin == 16 && coverage.cells == 20,
          "completed writes after a gap never union the missing cells");
    coverage.completed(18, 19, 23);
    check(coverage.begin == 16 && coverage.cells == 19 && coverage.next_token == 23,
          "rewriting an accepted prefix abandons old speculative tail coverage");
    coverage.rewind(17);
    check(coverage.begin == 16 && coverage.cells == 17 && coverage.next_token == -1,
          "rewind keeps actual prefix but cannot invent its boundary token");
    coverage.rewind(15);
    check(!coverage.known(), "rewind before completed interval loses certification");
    SavedConversation owned;
    owned.draft_boundary_residual = {1, 2, 3, 4};
    owned.freeze();
    const auto alias = owned.frozen_alias();
    check(owned.draft_boundary_residual.visit(0, 4, [](uint8_t* row, size_t n, size_t) {
              std::fill(row, row + n, 9); return true;
          }) && alias.draft_boundary_residual == ConversationBuffer({1, 2, 3, 4}) &&
          owned.draft_boundary_residual != alias.draft_boundary_residual,
          "replacement of a held boundary row cannot mutate a frozen publication alias");
    for (int format : {0,1,2}) for (int experts : {256,512})
        for (bool zero_qsa : {false,true}) for (bool ple : {false,true}) fixture(format,experts,zero_qsa,ple);
    checkpoint_payload_stages();
    resident_transfer_envelopes();
#if defined(CONVERSATION_TEST_TRANSFERS)
    strata::core::MtpDrafterCertificationFixture::run(check);
    for (int format : {0,1,2}) ring_refusal(format);
    for (int format : {0,1,2,3}) for (int64_t cut : {1,4,5,7,8,9,80}) exact_shared_cut(format, cut);
    for (bool batched : {false, true}) for (bool sync : {false, true})
        for (int64_t window : {2048, 32768}) shared_draft_fixture::diagnostic_shared_draft(batched, sync, window);
#endif
    std::printf("conversation_validation_test: %d host-only checks passed\n", checks);
}
