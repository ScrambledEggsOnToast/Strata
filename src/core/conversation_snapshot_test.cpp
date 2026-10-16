#include "strata/core/conversation_snapshot.hpp"
#include "strata/core/conversation_file.hpp"
#include "strata/core/prefix_extent_probe.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/program/session_fingerprint.hpp"
#include "../../tests/core/conversation_snapshot_equivalence.hpp"
#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <cstdlib>
#include <limits>
#include <vector>

using namespace strata::core;
using namespace strata::kernels;
using strata::test::equal;
using strata::test::same_snapshot;

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
void cuda_check(cudaError_t e) {
    if (e != cudaSuccess) { std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e)); std::exit(1); }
}
struct Fixture {
    ModelGeometry g;
    QsaState state;
    std::vector<void*> device, host;
    std::array<void*,5> sources{};
    std::array<size_t,5> sizes{};

    template<class T> void alloc(T*& p, size_t n, bool pinned = false) {
        if (!n) return;
        void* raw = nullptr;
        if (pinned) {
            cuda_check(cudaHostAlloc(&raw, n, cudaHostAllocMapped));
            host.push_back(raw);
            void* mapped = nullptr;
            cuda_check(cudaHostGetDevicePointer(&mapped, raw, 0));
            p = static_cast<T*>(mapped);
        } else {
            cuda_check(cudaMalloc(&raw, n)); device.push_back(raw); p = static_cast<T*>(raw);
        }
    }
    Fixture(int fmt, int mode) {
        auto& st = state;
        st.kv_mode = mode; st.kv_int8 = fmt==kKvInt8; st.kv_q4 = fmt==kKvQ4; st.kv_hybrid = fmt==3;
        st.n_pages = 24; st.n_slots = mode ? 4 : st.n_pages;
        st.max_cells = st.n_pages * 4; st.idx_pooled_rows = mode==2 ? 2 : st.max_cells/4+2;
        const size_t per = fmt==kKvQ4 ? kv_q4_bytes_per_head((int)g.head_dim) : g.head_dim*((st.kv_int8 || st.kv_hybrid) ? 1:2);
        const size_t rows = st.max_cells*g.n_head_kv, slot_rows = st.n_slots*4*g.n_head_kv;
        sizes = {rows*per, rows*per, fmt==kKvInt8 ? rows*(g.head_dim/64)*2:0,
                 fmt==kKvInt8 ? rows*(g.head_dim/64)*2:0, (size_t)st.idx_pooled_rows*g.idx_key_dim*4};
        if (fmt==3) {
            sizes = {rows*per, rows*kv_q4_bytes_per_head((int)g.head_dim), rows*(g.head_dim/64)*2, 0,
                     (size_t)st.idx_pooled_rows*g.idx_key_dim*4};
            alloc(st.k_q,slot_rows*per); alloc(st.v_q4,slot_rows*kv_q4_bytes_per_head((int)g.head_dim));
            alloc(st.k_scale,slot_rows*(g.head_dim/64)*2);
            if (mode) {
                alloc(st.host.k_q,sizes[0],true); alloc(st.host.v_q4,sizes[1],true);
                alloc(st.host.k_scale,sizes[2],true);
            }
            sources[0]=mode?st.host.k_q:st.k_q; sources[1]=mode?st.host.v_q4:st.v_q4;
            sources[2]=mode?st.host.k_scale:st.k_scale;
        } else if (fmt==kKvQ4) {
            alloc(st.k_q4,slot_rows*per); alloc(st.v_q4,slot_rows*per);
            if (mode) {alloc(st.host.k_q4,sizes[0],true); alloc(st.host.v_q4,sizes[1],true);}
            sources[0]=mode?st.host.k_q4:st.k_q4; sources[1]=mode?st.host.v_q4:st.v_q4;
        } else if (fmt==kKvInt8) {
            alloc(st.k_q,slot_rows*per); alloc(st.v_q,slot_rows*per);
            alloc(st.k_scale,slot_rows*(g.head_dim/64)*2); alloc(st.v_scale,slot_rows*(g.head_dim/64)*2);
            if (mode) {
                alloc(st.host.k_q,sizes[0],true); alloc(st.host.v_q,sizes[1],true);
                alloc(st.host.k_scale,sizes[2],true); alloc(st.host.v_scale,sizes[3],true);
            }
            sources[0]=mode?st.host.k_q:st.k_q; sources[1]=mode?st.host.v_q:st.v_q;
            sources[2]=mode?st.host.k_scale:st.k_scale; sources[3]=mode?st.host.v_scale:st.v_scale;
        } else {
            alloc(st.k_pool,slot_rows*per); alloc(st.v_pool,slot_rows*per);
            if (mode) {alloc(st.host.k_pool,sizes[0],true); alloc(st.host.v_pool,sizes[1],true);}
            sources[0]=mode?st.host.k_pool:st.k_pool; sources[1]=mode?st.host.v_pool:st.v_pool;
        }
        alloc(st.idx_pooled,sizes[4]); sources[4]=st.idx_pooled;
        alloc(st.page_table,st.n_pages*4);
        if (mode==1) {
            auto& m=st.map;
            m.page_table=st.page_table; m.n_blocks=st.n_pages; m.n_slots=st.n_slots;
            alloc(m.slot_block,st.n_slots*4); alloc(m.slot_stamp,st.n_slots*4); alloc(m.slot_ref,st.n_slots*4);
            alloc(m.miss_block,st.n_slots*4); alloc(m.miss_slot,st.n_slots*4); alloc(m.ctl,kKvCtlInts*4);
            kv_stream_reset(m,nullptr);
        } else if (mode==2) {
            kv_ring_table(st.page_table,st.n_pages,st.n_slots,nullptr);
        } else {
            std::vector<int32_t> table(st.n_pages);
            for (int32_t i=0;i<st.n_pages;++i) table[i]=i;
            cuda_check(cudaMemcpy(st.page_table,table.data(),table.size()*4,cudaMemcpyHostToDevice));
        }
    }
    void fill(uint8_t salt) {
        for (size_t i=0;i<sources.size();++i) {
            std::vector<uint8_t> data(sizes[i]);
            for (size_t j=0;j<data.size();++j) data[j]=(uint8_t)(salt+i*31+j*7+j/257);
            if (!data.empty()) cuda_check(cudaMemcpy(sources[i],data.data(),data.size(),cudaMemcpyDefault));
        }
    }
    void fill_after(uint8_t salt, int64_t first_dirty) {
        for (size_t i=0;i<sources.size();++i) {
            const size_t offset = i == 4 ? size_t(first_dirty/4)*g.idx_key_dim*4
                                         : (sizes[i]/size_t(state.max_cells))*size_t((first_dirty/4)*4);
            if (sizes[i] > offset)
                cuda_check(cudaMemset(static_cast<uint8_t*>(sources[i])+offset, salt, sizes[i]-offset));
        }
    }
    ~Fixture() { for (void* p:device) cudaFree(p); for (void* p:host) cudaFreeHost(p); }
};

bool empty_map(const QsaState& st) {
    std::vector<int32_t> table(st.n_pages), slots(st.n_slots);
    cuda_check(cudaMemcpy(table.data(),st.map.page_table,table.size()*4,cudaMemcpyDeviceToHost));
    cuda_check(cudaMemcpy(slots.data(),st.map.slot_block,slots.size()*4,cudaMemcpyDeviceToHost));
    const auto counters=kv_stream_counters(st.map);
    return std::all_of(table.begin(),table.end(),[](int32_t x){return x==-1;}) &&
           std::all_of(slots.begin(),slots.end(),[](int32_t x){return x==-1;}) &&
           counters.calls==0 && counters.lookups==0 && counters.misses==0 && counters.overflow==0;
}

void direct_kv(int fmt, int source_mode, int target_mode) {
    Fixture source(fmt,source_mode), target(fmt,target_mode);
    const bool index=source_mode!=2 && target_mode!=2;
    std::string err;
    for (int64_t upto : {0,3,4,5,65}) {
        source.fill(29); target.fill(181);
        cuda_check(cudaDeviceSynchronize());
        ConversationKv expected, before, after;
        check(conversation_kv_save(expected,source.state,source.g,upto,index,err),"direct source reference");
        check(conversation_kv_save(before,target.state,target.g,96,index,err),"direct destination guard reference");
        const auto pointers=target.sources;
        if (target_mode==1) {
            const int32_t zero=0;
            cuda_check(cudaMemcpy(target.state.map.page_table,&zero,4,cudaMemcpyHostToDevice));
            cuda_check(cudaMemcpy(target.state.map.slot_block,&zero,4,cudaMemcpyHostToDevice));
        }
        check(conversation_kv_copy(source.state,target.state,source.g,upto,index,err),"direct authoritative pool copy");
        cuda_check(cudaDeviceSynchronize());
        uint64_t hash=0;
        check(conversation_kv_verify(expected,target.state,source.g,upto,index,hash,err),
              "direct copy preserves partial pages, scale runs, pooled rows and draft ring");
        check(target.sources==pointers,"direct copy keeps stationary pool addresses");
        if (target_mode==1) check(empty_map(target.state),"direct copy invalidates old streamed pages");
        check(conversation_kv_save(after,target.state,target.g,96,index,err),"read direct destination guards");
        const std::array<const ConversationBuffer*,5> full={&before.k,&before.v,&before.k_scale,&before.v_scale,&before.pooled};
        const std::array<const ConversationBuffer*,5> copied={&expected.k,&expected.v,&expected.k_scale,&expected.v_scale,&expected.pooled};
        const std::array<const ConversationBuffer*,5> got={&after.k,&after.v,&after.k_scale,&after.v_scale,&after.pooled};
        for (size_t i=0;i<full.size();++i) {
            const size_t offset=copied[i]->size(), bytes=full[i]->size()-offset;
            std::vector<uint8_t> want(bytes), actual(bytes);
            check(full[i]->read(want.data(),offset,bytes) && got[i]->read(actual.data(),offset,bytes) && want==actual,
                  "direct copy leaves pages and index rows beyond the rounded extent untouched");
        }
        ConversationKv source_after;
        check(conversation_kv_save(source_after,source.state,source.g,upto,index,err) && equal(source_after,expected),
              "direct copy leaves source authoritative payload intact");
        // Format identity (including INT8 rotation) is checked before any write.
        auto incompatible=target.state;
        incompatible.kv_rot=!incompatible.kv_rot;
        if (fmt==kKvInt8)
            check(!conversation_kv_copy(source.state,incompatible,source.g,upto,index,err),
                  "direct transfer refuses rotated/unrotated mismatch");
    }
}

void release_reinit(int fmt, int mode) {
    Fixture source(fmt,0);
    source.fill(37);
    // conversation_kv_save requires the producing device work to be complete.
    cuda_check(cudaDeviceSynchronize());
    const bool index=mode!=2;
    std::string err;
    ConversationKv image;
    check(conversation_kv_save(image,source.state,source.g,9,index,err),"capture native release/re-init reference");
    const int64_t cells=20484;
    const auto placement=strata::plan::qsa_resolve_placement(qsa_real_shapes(),cells,mode?20480:0,mode==2?16:0,fmt);
    const uint64_t bytes=qsa_state_bytes(source.g,cells,true,0,&placement);
    const uint64_t pinned_before=qsa_kv_host_bytes();
    void* arena=nullptr;
    cuda_check(cudaMalloc(&arena,bytes));
    QsaState state;
    check(qsa_state_init(source.g,cells,arena,state,nullptr,0,&placement,bytes)==bytes,"real allocator initializes requested residency");
    // qsa_state_init may initialize the new owner on the legacy stream.
    // Complete it before restore switches to the private copy stream.
    cuda_check(cudaDeviceSynchronize());
    check(state.kv_mode==mode && qsa_kv_format(state)==fmt,"native main/ring format and residency match placement");
    const uint64_t first_owner=state.allocation_id;
    uint64_t fingerprint=0;
    check(conversation_kv_restore(image,state,source.g,9,index,err),
          "native owner accepts the saved image");
    // conversation_kv_restore leaves ring refill on the legacy stream; the
    // diagnostic verifier requires the documented completion fence.
    cuda_check(cudaDeviceSynchronize());
    check(conversation_kv_verify(image,state,source.g,9,index,fingerprint,err),
          "native owner restores all authoritative and ring bytes");
    qsa_state_release(state);
    check(!state.allocation_id && !state.host_allocation && !state.host_step && !state.host_pos &&
          !state.host.present() && !state.max_cells && !state.map.page_table,"release clears all secondary allocations and map handles");
    check(qsa_kv_host_bytes()==pinned_before,"release returns native pinned KV accounting");
    qsa_state_release(state);
    check(qsa_kv_host_bytes()==pinned_before,"repeated release is accounting-neutral");
    check(qsa_state_init(source.g,cells,arena,state,nullptr,0,&placement,bytes)==bytes &&
          state.allocation_id!=first_owner,"same arena re-initializes with a new owner");
    qsa_state_zero(state,source.g,nullptr);
    cuda_check(cudaDeviceSynchronize());
    if (mode==1) check(empty_map(state),"re-initialized streamed owner starts with an empty map");
    check(!conversation_kv_verify(image,state,source.g,9,index,fingerprint,err),"fresh owner does not inherit released payload");
    check(conversation_kv_restore(image,state,source.g,9,index,err),
          "re-initialized native owner accepts the saved image");
    cuda_check(cudaDeviceSynchronize());
    check(conversation_kv_verify(image,state,source.g,9,index,fingerprint,err),
          "saved state survives release and re-initialization");
    qsa_state_release(state);
    cuda_check(cudaFree(arena));
    check(qsa_kv_host_bytes()==pinned_before,"re-initialized owner releases without a pinned leak");
}

void full_session(int fmt, int mode, int experts) {
    Fixture main(fmt,mode), draft(fmt==3?kKvInt8:fmt,2);
    auto& g=main.g;
    g.n_layers=4; g.n_expert=experts;
    g.ssm_state_size=2; g.ssm_v_heads=2; g.ssm_conv_channels=8;
    SessionState ss;
    ss.max_cells=96; ss.qsa_states=&main.state;
    ss.layer_hi=g.n_layers; ss.gdn_alloc=g.n_gdn_layers(); ss.qsa_alloc=g.n_qsa_layers();   // the whole-model carve
    std::string err;
    ConversationStateSizes sizes;
    check(conversation_state_sizes(g,sizes,err),"whole-session geometry sizes");
    main.alloc(ss.gdn_state,sizes.gdn); main.alloc(ss.ple_hist,sizes.ple);
    main.alloc(main.state.idx_tail,sizes.tail); main.alloc(main.state.idx_dead,sizes.dead);
    main.alloc(main.state.idx_block_pos,sizes.block_pos);
    std::vector<int32_t> ids(65);
    for (size_t i=0;i<ids.size();++i) ids[i]=(int32_t)i+1;
    std::vector<ConversationImageKey> images;
    std::vector<ConversationCheckpoint> checkpoints;
    auto fill=[&](uint8_t salt) {
        main.fill(salt); draft.fill(salt);
        for (const auto& [p,n] : std::vector<std::pair<void*,size_t>>{
                 {ss.gdn_state,sizes.gdn},{ss.ple_hist,sizes.ple},{main.state.idx_tail,sizes.tail},
                 {main.state.idx_dead,sizes.dead},{main.state.idx_block_pos,sizes.block_pos}})
            cuda_check(cudaMemset(p,salt,n));
        // A real indexer maintains its spare pooled row as a copy of idx_dead.
        cuda_check(cudaMemcpy(main.state.idx_pooled+(ids.size()/4)*g.idx_key_dim,
                              main.state.idx_dead,sizes.dead,cudaMemcpyDeviceToDevice));
        cuda_check(cudaDeviceSynchronize());
    };
    fill(13);
    ConversationCheckpoint checkpoint;
    checkpoint.ids.assign(ids.begin(),ids.begin()+3);
    check(conversation_checkpoint_save(checkpoint,ss,g,err),"save complete running checkpoint");
    checkpoint.used = 17;
    checkpoints.push_back(std::move(checkpoint));
    const ConversationView view{ids,images,checkpoints,true};
    SavedConversation a,b,restored;
    check(conversation_snapshot_save(a,view,ss,g,draft.state,err),"capture complete A");
    check(a.checkpoints[0].used == 17,"upstream checkpoint LRU stamp survives capture");
    if (prefix_backing_enabled() && fmt == kKvF16 && mode == 0 && experts == 512) {
        // One completed 64-token prefix spans a whole immutable extent in each
        // main K/V pool. Draft's last uncomputed cell keeps its prefix private.
        SavedConversation donor = a;
        donor.scope = ConversationScope{{1}};
        auto selected = donor.live;
        selected.ids.resize(64);
        selected.gdn.write()[0] = 37;
        selected.dead.write()[0] = 71;
        if (immutable_checkpoints_enabled()) selected.seal();
        donor.checkpoints.push_back(std::move(selected));
        const SavedConversation before = donor;
        SavedConversation sibling = donor;
        sibling.live.ids.back() += 100;
        check(sibling.kv[0].k.visit(ConversationBuffer::prefix_extent_bytes, 1,
                  [](uint8_t* p, size_t, size_t) { *p = 29; return true; }),
              "make branch tail distinct beyond the completed shared prefix");
        check(std::string_view(share_prefix_extents(sibling, donor).status) == "equal",
              "authorize shared sibling backing at the completed cut");
        check(probe_prefix_extents(sibling, donor).aliased_bytes ==
                  2 * ConversationBuffer::prefix_extent_bytes,
              "main K/V retain shared extents while draft's incomplete extent stays private");
        const auto& cut = sibling.checkpoints.back();
        fill(177);
        check(conversation_snapshot_restore(sibling, ss, g, &draft.state, err, &cut) ==
                  ConversationRestore::restored,
              "restore selected checkpoint from a shared sibling");
        std::vector<uint8_t> recurrent(sizes.gdn), spare(sizes.dead);
        cuda_check(cudaMemcpy(recurrent.data(), ss.gdn_state, sizes.gdn, cudaMemcpyDeviceToHost));
        cuda_check(cudaMemcpy(spare.data(), main.state.idx_pooled + 16 * g.idx_key_dim,
                              sizes.dead, cudaMemcpyDeviceToHost));
        check(recurrent == cut.gdn && spare == cut.dead &&
                  ss.ple_prev[0] == 63 && ss.ple_prev[1] == 64,
              "shared selected restore uses cut recurrence, spare row and PLE history");
        uint64_t fingerprint = 0;
        check(conversation_kv_verify(sibling.kv.back(), draft.state, g, 64, false,
                                    fingerprint, err, 65),
              "shared selected restore refills the earlier draft-ring pages");
        const ConversationView cut_view{cut.ids, sibling.live.imgs, sibling.checkpoints, sibling.cvec};
        ConversationKvReuse reuse{std::move(sibling.kv), 65, 64};
        SavedConversation reparked;
        size_t peak = 0, reused = 0;
        check(conversation_snapshot_capture_bytes(reuse, cut_view, ss, g, draft.state, peak, err),
              "price shared sibling repark before consuming its retained storage");
        check(conversation_snapshot_save(reparked, cut_view, ss, g, draft.state, err,
                                         std::move(reuse), &reused),
              "repark the selected sibling with retained shared backing");
        check(reparked.bytes() <= peak && reparked.live.ids == cut.ids &&
                  reparked.live.gdn == cut.gdn && reparked.live.dead == cut.dead,
              "repark preserves selected running state within its admitted peak");
        check(reparked.kv[0].k.visit(0, 1,
                  [](uint8_t* p, size_t, size_t) { *p = 91; return true; }),
              "rewrite the reparked branch through copy-on-write");
        check(same_snapshot(donor, before),
              "selected restore, shrink, repark and subsequent rewrite leave donor unchanged");
        check(conversation_snapshot_restore(a, ss, g, draft.state, err) == ConversationRestore::restored,
              "restore independent full-session reference after shared-sibling boundary");
    }
    {
        // Whole-session residency migration in both directions, including the
        // process's actual draft format (INT8, not K8V4, for a hybrid main).
        Fixture destination(fmt,1-mode), destination_draft(fmt==3?kKvInt8:fmt,mode?0:2);
        destination.g=g;
        SessionState target;
        target.max_cells=ss.max_cells; target.qsa_states=&destination.state;
        target.layer_hi=ss.layer_hi; target.gdn_alloc=ss.gdn_alloc; target.qsa_alloc=ss.qsa_alloc;
        destination.alloc(target.gdn_state,sizes.gdn); destination.alloc(target.ple_hist,sizes.ple);
        destination.alloc(destination.state.idx_tail,sizes.tail); destination.alloc(destination.state.idx_dead,sizes.dead);
        destination.alloc(destination.state.idx_block_pos,sizes.block_pos);
        destination.fill(177); destination_draft.fill(177);
        for (const auto& [p,n] : std::array<std::pair<void*,size_t>,5>{
                 {{target.gdn_state,sizes.gdn},{target.ple_hist,sizes.ple},{destination.state.idx_tail,sizes.tail},
                  {destination.state.idx_dead,sizes.dead},{destination.state.idx_block_pos,sizes.block_pos}}})
            cuda_check(cudaMemset(p,177,n));
        if (!mode) check(empty_map(destination.state),"resident-to-streamed destination map is initially empty");
        cuda_check(cudaDeviceSynchronize());
        check(conversation_session_copy(ss,target,g,ids,nullptr,err) &&
              conversation_kv_copy(draft.state,destination_draft.state,g,65,false,err),
              "direct full-session handoff across residency modes");
        cuda_check(cudaDeviceSynchronize());
        SavedConversation direct;
        check(conversation_snapshot_save(direct,view,target,g,destination_draft.state,err) && same_snapshot(direct,a),
              "direct session handoff equals host snapshot for GDN, PLE, KV, index and draft");
        check(target.ple_prev[0]==64 && target.ple_prev[1]==65,"direct handoff reconstructs PLE token window");
        const auto& cut=a.checkpoints[0];
        // The source has advanced past this row: the selected checkpoint's dead
        // key must win over its later pooled row, including at a partial page.
        cuda_check(cudaMemset(main.state.idx_pooled,177,sizes.dead));
        cuda_check(cudaDeviceSynchronize());
        check(conversation_session_copy(ss,target,g,cut.ids,&cut,err),"direct selected-cut handoff");
        std::vector<uint8_t> spare(sizes.dead);
        cuda_check(cudaMemcpy(spare.data(),destination.state.idx_pooled,sizes.dead,cudaMemcpyDeviceToHost));
        check(spare==cut.dead && target.ple_prev[0]==2 && target.ple_prev[1]==3,
              "selected cut repairs spare row after copying the longer-lived source pools");
        check(conversation_snapshot_restore(a,ss,g,draft.state,err)==ConversationRestore::restored,
              "restore source after selected-cut discriminator");
        check(conversation_snapshot_restore(a,target,g,destination_draft.state,err)==ConversationRestore::restored,
              mode?"streamed-to-resident complete snapshot":"resident-to-streamed complete snapshot");
        SavedConversation migrated;
        check(conversation_snapshot_save(migrated,view,target,g,destination_draft.state,err),"capture migrated complete session and draft");
        check(same_snapshot(migrated,a),
              "residency migration preserves all main, recurrence, indexer and draft payloads");
        uint64_t hash=0;
        check(conversation_kv_verify(a.kv.back(),destination_draft.state,g,65,false,hash,err),
              "draft migration verifies authoritative bytes and every restored ring run");
        strata::program::SessionFingerprint before,after;
        check(strata::program::session_fingerprint(ss,draft.state,g,65,before,err) &&
              strata::program::session_fingerprint(target,destination_draft.state,g,65,after,err),
              "fingerprint both sides of a residency migration");
        check(before.gdn==after.gdn && before.ple==after.ple && before.tail==after.tail && before.dead==after.dead &&
              before.kv==after.kv && before.pooled==after.pooled && before.pooled_full==after.pooled_full &&
              before.draft==after.draft && before.stale==after.stale,"fingerprint reads each role's authoritative pools");
        if (!mode) {
            check(empty_map(destination.state),"restoring into an empty map leaves no stale resident pages");
            const int32_t zero=0;
            cuda_check(cudaMemcpy(destination.state.map.page_table,&zero,4,cudaMemcpyHostToDevice));
            cuda_check(cudaMemcpy(destination.state.map.slot_block,&zero,4,cudaMemcpyHostToDevice));
            check(!empty_map(destination.state),"outgoing destination map really has a resident page");
            check(conversation_snapshot_restore(a,target,g,destination_draft.state,err)==ConversationRestore::restored &&
                  empty_map(destination.state),"restore invalidates an outgoing destination residency map");
        }
    }
    fill(177);
    check(conversation_snapshot_save(b,view,ss,g,draft.state,err),"capture complete B");
    {
        // disk save path: metadata + streamed K/V give the same file as the captured image
        SavedConversation meta;
        std::vector<SessionKvSource> sources;
        check(conversation_snapshot_sources(meta,sources,view,ss,g,draft.state,err),"disk-save sources");
        check(meta.kv.empty() && sources.size()==b.kv.size(),"one source per K/V layer, none captured");
        const SessionFileIdentity id{1,2};
        namespace fs=std::filesystem;
        const fs::path dir=fs::temp_directory_path()/("strata-snap-test-"+std::to_string(experts)+"-"+
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(dir);
        const std::string p1=(dir/"streamed.bin").string(), p2=(dir/"captured.bin").string();
        auto slurp=[](const std::string& p){ std::ifstream f(p,std::ios::binary);
            return std::vector<char>((std::istreambuf_iterator<char>(f)),std::istreambuf_iterator<char>()); };
        size_t n1=0,n2=0;
        check(session_file_write(p1,meta,sources,id,n1,err),"streamed session write");
        check(session_file_write(p2,b,id,n2,err),"captured session write");
        SavedConversation x,y;
        check(session_file_read(p1,id,x,n1,err) && session_file_read(p2,id,y,n2,err) && n1==n2,"both files read");
        check(x.live.gdn==y.live.gdn && x.live.dead==y.live.dead && x.live.ids==y.live.ids &&
              x.checkpoints.size()==y.checkpoints.size() && equal(x.kv[0],y.kv[0]) && equal(x.kv.back(),y.kv.back()),
              "streamed file equals captured file");
        check(slurp(p1)==slurp(p2),"streamed file is byte-identical to the captured file");
        std::error_code ec;
        fs::remove_all(dir,ec);
        auto empty_ids=std::vector<int32_t>{};
        const ConversationView none{empty_ids,images,checkpoints,true};
        check(!conversation_snapshot_sources(meta,sources,none,ss,g,draft.state,err),"empty session refused");
    }
    check(a.live.dead!=b.live.dead,"different opening-state spare keys in regression fixture");
    auto bad=a; bad.kv.back().k.pop_back();
    check(conversation_snapshot_restore(bad,ss,g,draft.state,err)==ConversationRestore::invalid,
          "reject invalid late draft before any main-layer write");
    check(conversation_snapshot_save(restored,view,ss,g,draft.state,err),"capture B after refused restore");
    check(restored.live.gdn==b.live.gdn && restored.live.dead==b.live.dead &&
          equal(restored.kv[0],b.kv[0]) && equal(restored.kv[1],b.kv[1]),"refusal preserves all tested state");
    check(conversation_snapshot_restore(a,ss,g,draft.state,err)==ConversationRestore::restored,"restore complete A");
    check(conversation_snapshot_save(restored,view,ss,g,draft.state,err),"capture restored A");
    check(same_snapshot(restored,a),"whole-session A/B/A exactness including spare key");
    check(ss.ple_prev[0]==64 && ss.ple_prev[1]==65,"PLE token window reconstructed");
    for (int64_t dirty : {65, 3, 0}) {
        check(conversation_snapshot_restore(a,ss,g,draft.state,err)==ConversationRestore::restored,"restore growth fixture base");
        ConversationKvReuse reuse{a.kv,65,dirty};
        const uint8_t* original = nullptr;
        reuse.kv[0].k.visit(0,1,[&](const uint8_t* p,size_t,size_t){original=p;return true;});
        main.fill_after(91,dirty); draft.fill_after(91,std::max<int64_t>(0,dirty-1));
        cuda_check(cudaMemset(ss.gdn_state,91,sizes.gdn));
        ids.resize(70);
        for (size_t i=65;i<ids.size();++i) ids[i]=int32_t(i+1);
        cuda_check(cudaMemcpy(main.state.idx_pooled+(ids.size()/4)*g.idx_key_dim,
                              main.state.idx_dead,sizes.dead,cudaMemcpyDeviceToDevice));
        SavedConversation fresh,incremental;
        check(conversation_snapshot_save(fresh,view,ss,g,draft.state,err),"full capture reference after growth or rewind");
        size_t peak=0,reused=0;
        check(conversation_snapshot_capture_bytes(reuse,view,ss,g,draft.state,peak,err),"admit incremental capture peak");
        check(conversation_snapshot_save(incremental,view,ss,g,draft.state,err,std::move(reuse),&reused),"capture with retained pages");
        check(incremental.bytes() <= peak,"incremental allocation stays within admitted bound");
        check(incremental.live.gdn==fresh.live.gdn && incremental.live.ple==fresh.live.ple &&
              incremental.live.dead==fresh.live.dead && equal(incremental.kv[0],fresh.kv[0]) &&
              equal(incremental.kv[1],fresh.kv[1]),"incremental capture equals full capture after growth or rewind");
        check((dirty>=4)==(reused>0),"only complete unchanged pages or rows are retained");
        incremental.kv[0].k.visit(0,1,[&](const uint8_t* p,size_t,size_t){
            check(p==original,"growth never reallocates the retained payload");return true;
        });
        check(conversation_snapshot_restore(incremental,ss,g,draft.state,err)==ConversationRestore::restored,"restore segmented incremental snapshot");
        uint64_t fingerprint=0;
        check(conversation_kv_verify(incremental.kv.back(),draft.state,g,70,false,fingerprint,err),"incremental draft authoritative and ring read-back");
        ids.resize(65);
    }
    {
        // a layer split's later stage: the same image WITHOUT the draft layer's K/V (draft == nullptr)
        check(conversation_snapshot_restore(a,ss,g,draft.state,err)==ConversationRestore::restored,"restore A before stage images");
        size_t with_draft=0,without=0;
        check(conversation_snapshot_bytes(view,ss,g,draft.state,with_draft,err) &&
              conversation_snapshot_bytes(view,ss,g,nullptr,without,err) && without<with_draft,
              "a stage image's estimate leaves the draft out");
        SavedConversation stage,stage_b,stage_back;
        check(conversation_snapshot_save(stage,view,ss,g,nullptr,err),"capture a stage image");
        check(stage.kv.size()==a.kv.size()-1 && equal(stage.kv[0],a.kv[0]) && stage.live.gdn==a.live.gdn,
              "a stage image holds the session's own layers, no draft");
        check(stage.bytes()<a.bytes(),"a stage image is smaller by the draft ring");
        check(conversation_snapshot_validate(stage,ss,g,nullptr,err),"validate a stage image without a draft");
        check(!conversation_snapshot_validate(stage,ss,g,draft.state,err),"a stage image is refused where a draft is expected");
        check(!conversation_snapshot_validate(a,ss,g,nullptr,err),"an image with a draft is refused where none is expected");
        fill(201);
        check(conversation_snapshot_save(stage_b,view,ss,g,nullptr,err),"capture another stage state");
        check(conversation_snapshot_restore(a,ss,g,nullptr,err)==ConversationRestore::invalid,
              "restoring a draft image as a stage image is refused before any write");
        check(conversation_snapshot_save(stage_back,view,ss,g,nullptr,err) && stage_back.live.gdn==stage_b.live.gdn &&
              equal(stage_back.kv[0],stage_b.kv[0]),"the refusal left the stage untouched");
        check(conversation_snapshot_restore(stage,ss,g,nullptr,err)==ConversationRestore::restored,"restore a stage image");
        check(conversation_snapshot_save(stage_back,view,ss,g,nullptr,err) && stage_back.live.gdn==stage.live.gdn &&
              stage_back.live.dead==stage.live.dead && equal(stage_back.kv[0],stage.kv[0]),"stage A/B/A exactness");
        // its retained K/V: the next park copies only what changed, and equals a full capture
        ConversationKvReuse reuse{stage.kv,65,65,{}};
        SavedConversation fresh,incremental;
        size_t peak=0,reused=0;
        check(conversation_snapshot_save(fresh,view,ss,g,nullptr,err),"full stage capture reference");
        check(conversation_snapshot_capture_bytes(reuse,view,ss,g,nullptr,peak,err),"admit an incremental stage capture");
        check(conversation_snapshot_save(incremental,view,ss,g,nullptr,err,std::move(reuse),&reused),"incremental stage capture");
        check(reused>0 && incremental.bytes()<=peak && equal(incremental.kv[0],fresh.kv[0]) &&
              incremental.live.gdn==fresh.live.gdn,"an incremental stage capture reuses pages and equals a full one");
        SavedConversation split_image=stage;
        split_image.stage_images.push_back(stage);
        check(!conversation_snapshot_validate(split_image,ss,g,draft.state,err),
              "a layer split's image is refused by the whole-session form");
        ConversationKvReuse wrong{a.kv,65,65,{}};
        check(!conversation_snapshot_capture_bytes(wrong,view,ss,g,nullptr,peak,err),"a draft image's K/V is not a stage's reuse");
    }
    check(conversation_snapshot_restore(a,ss,g,draft.state,err)==ConversationRestore::restored,"restore A after stage images");
    check(conversation_checkpoint_restore(a.checkpoints[0],ss,g,err),"restore early running checkpoint");
    std::vector<uint8_t> spare(sizes.dead);
    cuda_check(cudaMemcpy(spare.data(),main.state.idx_pooled,sizes.dead,cudaMemcpyDeviceToHost));
    check(spare==a.checkpoints[0].dead,"checkpoint rebuilds spare row over a later completed block");
    check(ss.ple_prev[0]==2 && ss.ple_prev[1]==3,"checkpoint PLE token window");
}
}

int main() {
    int devices=0;
    if (cudaGetDeviceCount(&devices)!=cudaSuccess || !devices) return 77;
    for (int fmt : std::array<int,4>{kKvF16,kKvInt8,kKvQ4,3}) for (int mode : {0,1,2}) {
        Fixture f(fmt,mode);
        if (fmt==3) {
            // Keep the under-provisioned rejection checks, separately from the
            // correctly backed hybrid owners exercised by the format/mode matrix.
            std::string err;
            ConversationKv invalid;
            const auto backed=f.state.host;
            f.state.host={};
            for (int unsupported : {1,2}) {
                f.state.kv_mode=unsupported;
                check(conversation_kv_bytes(f.state,f.g,9,true)==0,"hybrid streaming/ring rejected without authoritative backing");
                check(!conversation_kv_save(invalid,f.state,f.g,9,true,err),"unbacked hybrid streaming/ring capture fails closed");
            }
            f.state.kv_mode=mode;
            f.state.host=backed;
            if (mode) for (int part : {0,1,2}) {
                auto missing=f.state;
                if (part==0) missing.host.k_q=nullptr;
                if (part==1) missing.host.v_q4=nullptr;
                if (part==2) missing.host.k_scale=nullptr;
                check(conversation_kv_bytes(missing,f.g,9,mode!=2)==0,"hybrid missing host run is rejected before capture allocation");
                check(!conversation_kv_save(invalid,missing,f.g,9,mode!=2,err),"hybrid capture requires each authoritative host run");
            }
        }
        {
            std::string err;
            ConversationKv invalid;
            for (const int64_t bad : {-1LL, 97LL, (long long) std::numeric_limits<int64_t>::max()}) {
                check(!conversation_kv_save(invalid,f.state,f.g,bad,mode!=2,err),"reject invalid save extent before allocation");
                check(!conversation_kv_restore(invalid,f.state,f.g,bad,mode!=2,err),"reject invalid restore extent before copying");
                check(conversation_kv_bytes(f.state,f.g,bad,mode!=2)==0,"invalid extent cannot overflow byte estimate");
            }
        }
        for (int64_t upto : {0,1,3,4,5,63,64,65,96}) {
            const bool index=mode!=2;
            std::string err;
            f.fill(13);
            ConversationKv a,b,restored;
            cuda_check(cudaDeviceSynchronize());
            check(conversation_kv_save(a,f.state,f.g,upto,index,err),"save A");
            check(a.bytes()==conversation_kv_bytes(f.state,f.g,upto,index),"size estimate equals snapshot payload");
            f.fill(177);
            // Restore uses the private non-blocking copy stream, so finish the
            // legacy-stream fixture mutation before both save B and restore A.
            cuda_check(cudaDeviceSynchronize());
            check(conversation_kv_save(b,f.state,f.g,upto,index,err),"save B");
            check(upto==0 || !equal(a,b),"fixture changes state");
            check(conversation_kv_restore(a,f.state,f.g,upto,index,err),"restore A over B");
            cuda_check(cudaDeviceSynchronize());
            check(conversation_kv_save(restored,f.state,f.g,upto,index,err),"read back restored A");
            check(equal(a,restored),"A/B/A byte-exact K/V and indexer state");
            uint64_t fingerprint = 0;
            check(conversation_kv_verify(a,f.state,f.g,upto,index,fingerprint,err),"verify restored authoritative and resident bytes without rewriting them");
            {
                // disk save streams the same bytes as a captured image, without the host copy
                SessionKvSource src;
                check(conversation_kv_source(src,f.state,f.g,upto,index,err),"K/V source for a disk save");
                check(src.format==a.format && src.cells==a.cells && src.heads==a.heads && src.head_dim==a.head_dim &&
                      src.page_size==a.page_size && src.pooled_rows==a.pooled_rows && src.idx_dim==a.idx_dim,
                      "K/V source metadata equals the captured image");
                const std::array<const ConversationBuffer*,5> parts={&a.k,&a.v,&a.k_scale,&a.v_scale,&a.pooled};
                for (size_t i=0;i<5;++i) {
                    check(src.sizes[i]==parts[i]->size(),"K/V source part size");
                    std::vector<uint8_t> got(src.sizes[i]), want(src.sizes[i]);
                    const size_t half=src.sizes[i]/2;   // two reads: offsets are honoured
                    check(src.read(i,0,got.data(),half) && src.read(i,half,got.data()+half,src.sizes[i]-half),
                          "K/V source reads");
                    check(parts[i]->read(want.data(),0,want.size()) && got==want,"K/V source bytes equal the image");
                }
                SessionKvSource none;
                check(!conversation_kv_source(none,f.state,f.g,-1,index,err),"K/V source rejects an invalid extent");
            }
            if (mode==1) {
                std::vector<int32_t> table((size_t)f.state.n_pages);
                cuda_check(cudaMemcpy(table.data(),f.state.page_table,table.size()*4,cudaMemcpyDeviceToHost));
                check(std::all_of(table.begin(),table.end(),[](int32_t x){return x==-1;}),"stale VRAM pages invalidated");
            }
            if (mode==2 && upto>0) {
                const auto s=qsa_real_shapes();
                const int64_t b1=(upto+s.page_size-1)/s.page_size,b0=std::max<int64_t>(0,b1-f.state.n_slots);
                const size_t block=kv_block_bytes(s,fmt)/2;
                // Codes and scales are separate in INT8; check code bytes here.
                const size_t code_block=(fmt==kKvInt8 || fmt==3) ? s.page_size*s.n_head_kv*s.head_dim : block;
                const void* slots=fmt==kKvQ4?(void*)f.state.k_q4:(fmt==kKvInt8 || fmt==3)?(void*)f.state.k_q:(void*)f.state.k_pool;
                std::vector<uint8_t> got(code_block);
                for (int64_t page=b0;page<b1;++page) {
                    cuda_check(cudaMemcpy(got.data(),(const uint8_t*)slots+(page%f.state.n_slots)*code_block,code_block,cudaMemcpyDeviceToHost));
                    std::vector<uint8_t> expected(code_block);
                    check(a.k.read(expected.data(), page*code_block, code_block),"read segmented expected draft page");
                    check(got == expected,"draft ring contains restored pages");
                }
            }
            auto bad=a; bad.head_dim++;
            check(!conversation_kv_restore(bad,f.state,f.g,upto,index,err),"reject incompatible geometry");
            if (fmt==kKvInt8) {
                // #293: INT8 K/V saved through the Hadamard rotation (STRATA_KV_ROT=1) never restores into a state
                // without it, nor the reverse
                f.state.kv_rot=true;
                ConversationKv rotated;
                check(conversation_kv_save(rotated,f.state,f.g,upto,index,err),"save rotated INT8");
                check(!conversation_kv_restore(a,f.state,f.g,upto,index,err),"unrotated image into a rotated state rejected");
                f.state.kv_rot=false;
                check(!conversation_kv_restore(rotated,f.state,f.g,upto,index,err),"rotated image into an unrotated state rejected");
                check(conversation_kv_restore(a,f.state,f.g,upto,index,err),"unrotated image still restores");
            }
            if (!a.k.empty()) {
                bad=a; bad.k.pop_back();
                check(!conversation_kv_restore(bad,f.state,f.g,upto,index,err),"reject malformed payload");
            }
            if (upto==65 && mode==2) {
                // Two cuts cover complete and partial pages after the old ring has wrapped.
                for (int64_t cut : {4,5}) {
                    f.fill(177);
                    cuda_check(cudaDeviceSynchronize());
                    check(conversation_kv_restore(a,f.state,f.g,cut,false,err,upto),
                          "restore selected draft prefix from longer wrapped image");
                    cuda_check(cudaDeviceSynchronize());
                    check(conversation_kv_verify(a,f.state,f.g,cut,false,fingerprint,err,upto),
                          "selected draft prefix authoritative bytes and resident ring agree");
                }
            }
        }
    }
    for (int fmt : {kKvF16,kKvInt8,kKvQ4}) for (int mode : {0,1}) for (int experts : {256,512})
        full_session(fmt,mode,experts);
    for (int mode : {0,1}) for (int experts : {256,512}) full_session(3,mode,experts);
    for (int fmt : {kKvF16,kKvInt8,kKvQ4,kKvHybrid})
        for (int source : {0,1,2}) for (int target : {0,1,2}) direct_kv(fmt,source,target);
    for (int fmt : {kKvF16,kKvInt8,kKvQ4,kKvHybrid}) for (int mode : {0,1,2}) release_reinit(fmt,mode);
    std::printf("conversation_snapshot_test: %d checks passed\n",checks);
}
