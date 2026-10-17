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

#if defined(CONVERSATION_TEST_TRANSFERS)
// GNU/ELF link wrapping exercises the actual restore control flow without
// initializing CUDA or deliberately poisoning a real device context. This is
// host-backend fault injection, not a claim of hardware-failure recovery.
namespace {
int copy_calls = 0, sync_calls = 0, stream_sync_calls = 0, fail_copy = 0, fail_sync = 0;
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
extern "C" cudaError_t __wrap_cudaMemcpyAsync(void* dst, const void* src, size_t n, cudaMemcpyKind, cudaStream_t) {
    return __wrap_cudaMemcpy(dst, src, n, cudaMemcpyDefault);
}
extern "C" cudaError_t __wrap_cudaStreamSynchronize(cudaStream_t) { ++stream_sync_calls; return cudaSuccess; }
extern "C" cudaError_t __wrap_cudaDeviceSynchronize() {
    return ++sync_calls == fail_sync ? cudaErrorUnknown : cudaSuccess;
}
extern "C" cudaError_t __wrap_cudaGetLastError() { return cudaSuccess; }
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
    Pools(const ModelGeometry& g, int format) {
        st.max_cells = 96; st.n_pages = st.n_slots = 24; st.idx_pooled_rows = 26;
        st.kv_int8 = format == 1; st.kv_q4 = format == 2;
        st.kv_hybrid = format == 3;
        const size_t per = format == 2 ? strata::kernels::kv_q4_bytes_per_head((int) g.head_dim)
                                      : g.head_dim * (format == 1 || format == 3 ? 1 : 2);
        data[0].resize(96 * g.n_head_kv * per, 0xa5);
        data[1].resize(format == 3 ? 96 * g.n_head_kv *
                       strata::kernels::kv_q4_bytes_per_head((int) g.head_dim) : data[0].size(), 0xa5);
        data[2].resize(format == 1 || format == 3 ? 96 * g.n_head_kv * (g.head_dim / 64) * 2 : 0, 0xa5);
        if (format == 1) data[3] = data[2];
        data[4].resize(26 * g.idx_key_dim * 4, 0xa5);
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
    check(gdn==image.live.gdn && history==image.live.ple && ss.ple_prev[0]==8 && ss.ple_prev[1]==9,
          "successful restore publishes correct running state and PLE window");
    uint64_t fingerprint = 0;
    check(conversation_kv_verify(image.kv.back(),draft.st,g,9,false,fingerprint,error), "read-back verifies complete draft payload");
    draft.data[0][0] ^= 1;
    check(!conversation_kv_verify(image.kv.back(),draft.st,g,9,false,fingerprint,error), "read-back detects corrupted draft byte");
    draft.data[0][0] ^= 1;
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
#if defined(CONVERSATION_TEST_TRANSFERS)
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
    const int64_t begin = std::min<int64_t>(1, cut - 1);
    ConversationView exact{ids, no_images, no_checks, true, true, cut - 1, begin};
    size_t estimate = 0;
    copy_calls = sync_calls = stream_sync_calls = fail_copy = fail_sync = 0;
    auto invalid = exact;
    invalid.draft_tokens = cut + 1;
    check(!conversation_snapshot_bytes(invalid, ss, g, &draft.st, estimate, error) &&
          copy_calls == 0 && sync_calls == 0, "draft extent beyond an exact cut is refused before device access");
    invalid.draft_tokens = cut;
    check(!conversation_snapshot_bytes(invalid, ss, g, &draft.st, estimate, error),
          "the draft row consuming the first private suffix token cannot be claimed");
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
    check(std::all_of(first.live.tails.begin() + active_tail, first.live.tails.end(),
                      [](uint8_t b) { return b == 0; }) &&
          std::all_of(first.live.block_pos.begin(), first.live.block_pos.end(), [](uint8_t b) { return b == 0; }),
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
    auto corrupt = second;
    corrupt.kv.back().k.visit(corrupt.kv.back().k.size() - 1, 1,
        [](uint8_t* p, size_t, size_t) { *p = 0xee; return true; });
    copy_calls = sync_calls = 0;
    check(!conversation_snapshot_validate(corrupt, ss, g, &draft.st, error) &&
          copy_calls == 0 && sync_calls == 0, "restore rejects a shorter draft claim that conceals nonzero post-cut bytes");
    for (auto& bytes : draft.data) std::fill(bytes.begin(), bytes.end(), 0xee);
    check(conversation_snapshot_restore(second, ss, g, &draft.st, error) == ConversationRestore::restored,
          "restore understands the intentionally shorter draft extent");
    uint64_t fingerprint = 0;
    check(conversation_kv_verify(second.kv.back(), draft.st, g, cut, false, fingerprint, error),
          "restore overwrites the excluded draft row and page tail with zeros instead of reviving old suffix state");
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
        const auto zeros = [](const auto& bytes) {
            return std::all_of(bytes.begin(), bytes.end(), [](uint8_t b) { return b == 0; });
        };
        const auto zero_buffer = [](const ConversationBuffer& bytes) {
            return bytes.visit(0, bytes.size(), [](const uint8_t* p, size_t n, size_t) {
                return std::all_of(p, p + n, [](uint8_t b) { return b == 0; });
            });
        };
        check(zero_buffer(stage.kv.front().k) && zero_buffer(stage.kv.front().v) && zero_buffer(stage.kv.front().pooled) &&
              zeros(stage.live.tails) && zeros(stage.live.dead),
              "inactive primary QSA payload and index state cannot carry bytes from another owner");
        unused = ss;
        unused.layer_lo = 2; unused.gdn_ord0 = 2; unused.gdn_alloc = 1;
        check(conversation_snapshot_save(stage, cut_view, unused, g, nullptr, error) && zeros(stage.live.ple),
              "PLE history on a stage that never runs the first layer is excluded");
    }
}
#endif

}

int main() {
    for (int format : {0,1,2}) for (int experts : {256,512})
        for (bool zero_qsa : {false,true}) for (bool ple : {false,true}) fixture(format,experts,zero_qsa,ple);
    checkpoint_payload_stages();
#if defined(CONVERSATION_TEST_TRANSFERS)
    for (int format : {0,1,2}) ring_refusal(format);
    for (int format : {0,1,2,3}) for (int64_t cut : {1,4,5,7,8,9}) exact_shared_cut(format, cut);
#endif
    std::printf("conversation_validation_test: %d host-only checks passed\n", checks);
}
