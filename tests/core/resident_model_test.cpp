// Runs only in strata-resident-test after the normal protected full-model startup.
// No fake CUDA success: all transfers forward to the real runtime; one injected
// transfer failure proves that partial state is unavailable, not GPU recovery.
#include "strata/program/resident_slots.hpp"
#include "strata/program/session_fingerprint.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/mtp.hpp"
#include "strata/core/verify.hpp"
#include "conversation_snapshot_equivalence.hpp"
#include <cuda_runtime.h>
#include <array>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {
using strata::test::same_snapshot;
void* fail_destination = nullptr;
bool transfer_failed = false;

}
extern "C" cudaError_t __real_cudaMemcpy(void*, const void*, size_t, cudaMemcpyKind);
extern "C" cudaError_t __wrap_cudaMemcpy(void* dst, const void* src, size_t size, cudaMemcpyKind kind) {
    if (dst == fail_destination && size) {
        fail_destination = nullptr;
        transfer_failed = true;
        return cudaErrorInvalidValue;
    }
    return __real_cudaMemcpy(dst, src, size, kind);
}

// Snapshot transfers use a private stream. Forward async calls to the real
// runtime too, so the existing partial-write fixture reaches today's copy path.
extern "C" cudaError_t __real_cudaMemcpyAsync(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t);
extern "C" cudaError_t __wrap_cudaMemcpyAsync(void* dst, const void* src, size_t size,
                                               cudaMemcpyKind kind, cudaStream_t stream) {
    if (dst == fail_destination && size) {
        fail_destination = nullptr;
        transfer_failed = true;
        return cudaErrorInvalidValue;
    }
    return __real_cudaMemcpyAsync(dst, src, size, kind, stream);
}

int resident_model_checks(strata::program::ResidentSlots& owner, strata::core::ExpertDispatch& dispatch,
                          const strata::core::ModelGeometry& geometry, strata::core::SessionState& main,
                          strata::core::SessionState& a, strata::core::SessionState& b,
                          strata::core::MtpDrafter& draft, strata::core::Verifier& verifier,
                          strata::core::PoolMultiFn pool, void* pool_user) {
    using namespace strata;
    std::string error;
    int checks = 0;
    auto require = [&](bool ok, const char* message) {
        ++checks;
        if (!ok) throw std::runtime_error(std::string(message) + ": " + error);
    };
    try {
        if (std::getenv("STRATA_TEST_MIXED_RESIDENCY")) {
            // Exercise the production Verifier, not a single-sequence parity
            // kernel or a hand-written imitation of its row compaction.
            require(main.qsa_alloc>0 && main.qsa_states[main.qsa_primary()].kv_mode==0 &&
                    a.qsa_states[a.qsa_primary()].kv_mode==1 && b.qsa_states[b.qsa_primary()].kv_mode==1,
                    "mixed fixture needs resident working state and streamed slots");
            const std::vector<int32_t> prefix{11,22};
            const std::vector<core::ConversationImageKey> images;
            const std::vector<core::ConversationCheckpoint> checkpoints;
            auto capture = [&](core::SessionState& state, const std::vector<int32_t>& ids) {
                core::SavedConversation saved;
                require(core::conversation_snapshot_save(saved,{ids,images,checkpoints,true},state,geometry,nullptr,error),
                        "capture complete real mixed-owner state");
                return saved;
            };
            std::array<core::SavedConversation,2> opening{capture(main,prefix),capture(a,prefix)};
            const std::array<int32_t,2> previous{main.ple_prev[0],main.ple_prev[1]};
            const auto old_plan=dispatch.plan;
            const auto old_generation=dispatch.request_generation;
            const std::array<uint64_t,2> old_requests{dispatch.slot_requests[0],dispatch.slot_requests[1]};
            require(owner.release_resources(error),"release original batch bindings before mixed-owner binding");
            require(verifier.init_slots({&main,&a},error),"bind resident and streamed owners in one real verifier");
            kernels::SamplerParams sampling;
            sampling.greedy=true; sampling.temperature=0; sampling.penalty_last_n=0; sampling.seed=17;
            auto bind = [&] {
                for (int slot=0;slot<2;++slot) {
                    dispatch.slot_requests[slot]=1000+slot;
                    require(verifier.set_slot_context(slot,dispatch.slot_requests[slot],sampling,&prefix,true,error),
                            "bind independent real mixed-row identities");
                }
                dispatch.request_generation=1000;
                dispatch.plan=verifier.plan_sink();
            };
            bind();
            auto reset = [&] {
                require(verifier.context_idle(error),"drain mixed window before restoring its opening state");
                require(core::conversation_snapshot_restore(opening[0],main,geometry,nullptr,error)==core::ConversationRestore::restored &&
                        core::conversation_snapshot_restore(opening[1],a,geometry,nullptr,error)==core::ConversationRestore::restored,
                        "restore both mixed owners independently");
            };
            struct Observed {
                std::array<core::SavedConversation,2> state;
                std::array<std::vector<float>,2> residual;
                std::array<int32_t,2> picks{-1,-1};
            };
            auto window = [&](const std::vector<int>& slots, const std::array<int32_t,2>& tokens,
                              core::BatchRowMask other_mask) {
                core::BatchRow rows[2];
                int32_t output[2]={-7,-7};
                std::array<bool,2> active{};
                for (size_t t=0;t<slots.size();++t) {
                    const int slot=slots[t];
                    auto& row=rows[t];
                    row.slot=slot; row.state=slot?&a:&main; row.request_id=1000+slot;
                    row.cancellation_generation=row.request_id; row.token=tokens[slot];
                    row.position=2; row.sequence_end=3; row.output_offset=(int)t;
                    row.mask=core::BatchRowMask::active;
                    active[slot]=true;
                }
                int count=(int)slots.size();
                if (count==1) {
                    rows[1].slot=1-slots[0]; rows[1].state=slots[0]?&main:&a;
                    rows[1].mask=other_mask; rows[1].output_offset=1; count=2;
                }
                if (!count) {
                    for (int slot=0;slot<2;++slot) {
                        rows[slot].slot=slot; rows[slot].state=slot?&a:&main;
                        rows[slot].mask=other_mask; rows[slot].output_offset=slot;
                    }
                    count=2;
                }
                require(verifier.run_slot_rows(rows,count,pool,pool_user,output,error) &&
                        verifier.commit_slots(error) && verifier.context_idle(error) && !dispatch.failed,
                        "bounded production mixed window executes and commits");
                Observed observed;
                for (int slot=0;slot<2;++slot) {
                    auto ids=prefix;
                    if (active[slot]) ids.push_back(tokens[slot]);
                    observed.state[slot]=capture(slot?a:main,ids);
                }
                for (size_t t=0;t<slots.size();++t) {
                    const int slot=slots[t];
                    observed.picks[slot]=output[t];
                    observed.residual[slot].resize((size_t)geometry.hc*geometry.n_embd);
                    require(cudaMemcpy(observed.residual[slot].data(),verifier.final_R((int)t),
                                       observed.residual[slot].size()*4,cudaMemcpyDeviceToHost)==cudaSuccess,
                            "read the actual committed row residual");
                    require(verifier.validated_slot_request_id(slot)==1000+(uint64_t)slot,
                            "GPU commit canary validates the mixed row's owner");
                }
                if (slots.size()==1) require(output[1]==-1,"inactive or cancelled row never produces an output");
                if (slots.empty()) require(output[0]==-1 && output[1]==-1,"all-inert mixed window produces no outputs");
                return observed;
            };
            auto matches = [&](const Observed& got, const Observed& expected, int slot) {
                require(same_snapshot(got.state[slot],expected.state[slot]) &&
                        got.picks[slot]==expected.picks[slot] && got.residual[slot]==expected.residual[slot],
                        "mixed graph agrees byte-for-byte with the independent owner window");
            };
            const std::array<int32_t,2> first{33,44}, second{55,66};
            reset();
            const auto solo0=window({0},first,core::BatchRowMask::padding);
            require(same_snapshot(solo0.state[1],opening[1]),"unselected streamed owner stays unchanged");
            reset();
            const auto solo1=window({1},first,core::BatchRowMask::padding);
            require(same_snapshot(solo1.state[0],opening[0]),"unselected resident owner stays unchanged");
            reset();
            const size_t before=verifier.batch_resources().graph_keys;
            const auto captured=window({1,0},first,core::BatchRowMask::padding);
            const size_t captured_keys=verifier.batch_resources().graph_keys;
            require(captured_keys==before+1,"mixed row order captures exactly one new window/commit key");
            matches(captured,solo0,0); matches(captured,solo1,1);
            require(captured.residual[0]!=captured.residual[1],"distinct owners/tokens yield distinct actual residuals");
            reset();
            const auto replayed=window({1,0},first,core::BatchRowMask::padding);
            require(verifier.batch_resources().graph_keys==captured_keys,"second mixed window replays the retained graph");
            matches(replayed,captured,0); matches(replayed,captured,1);
            reset();
            const auto next0=window({0},second,core::BatchRowMask::padding);
            reset();
            const auto next1=window({1},second,core::BatchRowMask::padding);
            reset();
            const auto changed=window({1,0},second,core::BatchRowMask::padding);
            require(verifier.batch_resources().graph_keys==captured_keys,"changed tokens reuse the same mixed graph");
            matches(changed,next0,0); matches(changed,next1,1);
            require(changed.residual[0]!=captured.residual[0] && changed.residual[1]!=captured.residual[1],
                    "replay reads new row inputs rather than capture-time tokens");
            for (auto mask : {core::BatchRowMask::padding,core::BatchRowMask::finished,core::BatchRowMask::cancelled})
                for (int active : {0,1}) {
                    reset();
                    const auto inert=window({active},first,mask);
                    matches(inert,active?solo1:solo0,active);
                    require(same_snapshot(inert.state[1-active],opening[1-active]),
                            "inactive/finished/cancelled peer keeps every private running and KV byte");
                }
            for (auto mask : {core::BatchRowMask::padding,core::BatchRowMask::finished,core::BatchRowMask::cancelled}) {
                reset();
                const size_t keys=verifier.batch_resources().graph_keys;
                const auto inert=window({},first,mask);
                require(same_snapshot(inert.state[0],opening[0]) && same_snapshot(inert.state[1],opening[1]) &&
                        verifier.batch_resources().graph_keys==keys,"all-inert mixed window neither mutates owners nor captures a graph");
            }
            reset();
            require(verifier.release_batch_resources(error),"release real mixed-owner graphs and scratch");
            require(verifier.n_slots()==0 && verifier.batch_resources().graph_keys==0 &&
                    verifier.batch_resources().device_bytes==0 && verifier.batch_resources().pinned_bytes==0,
                    "release retires every mixed graph and batch allocation");
            require(same_snapshot(capture(main,prefix),opening[0]) && same_snapshot(capture(a,prefix),opening[1]),
                    "batch release preserves both residency owners' private state");
            require(verifier.init_slots({&main,&a},error),"re-initialize the same stable mixed owners");
            bind();
            const auto reinitialized=window({1,0},first,core::BatchRowMask::padding);
            matches(reinitialized,captured,0); matches(reinitialized,captured,1);
            reset();
            require(verifier.release_batch_resources(error),"retire temporary mixed owner bindings");
            dispatch.plan=old_plan; dispatch.request_generation=old_generation;
            dispatch.slot_requests[0]=old_requests[0]; dispatch.slot_requests[1]=old_requests[1];
            main.ple_prev[0]=previous[0]; main.ple_prev[1]=previous[1];
            require(owner.init_resources(error),"rebind original resident lifecycle owner");
            std::printf("MIXED_RESIDENCY_WINDOWS capture=1 replay=2 inactive=2 finished=2 cancelled=2 all_inert=3 reinitialized=1 failures=0\n");
            require(draft.slot_state(0) && draft.slot_state(1) && draft.kv_state().kv_mode==0 &&
                    draft.slot_state(0)->kv_mode==2 && draft.kv_state().kv_int8 && draft.slot_state(0)->kv_int8,
                    "hybrid process uses actual resident and ring INT8 draft states");
            core::ConversationKv original,peer,marker,changed_draft;
            require(core::conversation_kv_save(original,draft.kv_state(),geometry,2,false,error) &&
                    core::conversation_kv_save(peer,*draft.slot_state(1),geometry,2,false,error),"capture real working and independent peer draft");
            marker=original;
            for (auto* part : {&marker.k,&marker.v,&marker.k_scale,&marker.v_scale})
                require(part->visit(0,part->size(),[](uint8_t* p,size_t n,size_t){std::memset(p,0x3c,n);return true;}),
                        "mark each real draft payload run");
            require(core::conversation_kv_restore(marker,draft.kv_state(),geometry,2,false,error) &&
                    draft.save_slot(0,2,error),"save marked working draft into host-backed slot ring");
            uint64_t hash=0;
            require(core::conversation_kv_verify(marker,*draft.slot_state(0),geometry,2,false,hash,error),
                    "real draft save refills both codes and both scale runs of the ring");
            changed_draft=marker;
            for (auto* part : {&changed_draft.k,&changed_draft.v,&changed_draft.k_scale,&changed_draft.v_scale})
                require(part->visit(0,part->size(),[](uint8_t* p,size_t n,size_t){std::memset(p,0x5a,n);return true;}),
                        "replace working draft with distinct outgoing bytes");
            require(core::conversation_kv_restore(changed_draft,draft.kv_state(),geometry,2,false,error) &&
                    !core::conversation_kv_verify(marker,draft.kv_state(),geometry,2,false,hash,error),
                    "draft restore fixture really overwrites the working payload");
            require(draft.restore_slot(0,2,error) &&
                    core::conversation_kv_verify(marker,draft.kv_state(),geometry,2,false,hash,error),
                    "restore the saved ring draft over the overwritten resident working draft");
            require(core::conversation_kv_verify(peer,*draft.slot_state(1),geometry,2,false,hash,error),
                    "draft save/restore cannot alter the independent peer's payload");
            require(core::conversation_kv_restore(original,draft.kv_state(),geometry,2,false,error) &&
                    draft.save_slot(0,2,error),"restore original draft before resident lifecycle regressions");
            std::puts("DRAFT_RESIDENCY_ROUNDTRIP working=resident slot=ring format=int8 failures=0");
        }
        const uint64_t w4 = core::Verifier::planned_device_bytes(geometry, main.max_cells, 150000, 4);
        const uint64_t w8 = core::Verifier::planned_device_bytes(geometry, main.max_cells, 150000, 8);
        require(w4 > 0 && w8 > w4 && w8 != UINT64_MAX, "full eight-row verifier uses bounded actual arena sequence");
        require(core::Verifier::planned_device_bytes(geometry, main.max_cells, 150000, 9) == UINT64_MAX,
                "pricing refuses unsupported window width");
        const std::vector<int64_t> prompt{11, 22, 33, 44};
        const std::vector<int32_t> prefix{11, 22};
        program::ResidentRequest request;
        request.max_new = 8;
        request.scope.value[0] = 1;
        request.sampling.seed = 0; // chosen seed must survive repeated yields
        request.sampling.penalty_last_n = 2;
        require(!owner.init_resources(error), "repeated init refuses");
        const core::ExpertOperation operation{71, 0, 0, 1, 1};
        auto& ledger = dispatch.completion.at(0);
        require(ledger.begin(operation, 0) && ledger.retain_submission(operation), "retain submitted operation");
        require(!owner.begin(0, prompt, request, error), "begin refuses in-flight scratch");
        require(owner.working().request == 0 && dispatch.request_generation == 0, "refusal leaves identity unpublished");
        ledger.cancel();
        require(!owner.begin(0, prompt, request, error), "cancellation is not a consumer drain");
        require(ledger.drain_submission(operation) && ledger.release(operation), "consumer drain releases operation");
        require(owner.begin(-1, prompt, request, error), "begin request A");
        const uint64_t aid = owner.working().request, seed = owner.working().sampling.seed;
        require(seed != 0, "automatic seed resolved");
        std::vector<core::ConversationCheckpoint> checkpoint_chain(1);
        auto& checkpoint = checkpoint_chain.front();
        checkpoint.ids = prefix;
        require(core::conversation_checkpoint_save(checkpoint, main, geometry, error), "save real working checkpoint");
        program::ResidentAdmission admission{33, &checkpoint_chain};
        const auto* main_gdn=main.gdn_state;
        const auto* slot_gdn=a.gdn_state;
        const auto* slot_qsa=a.qsa_states;
        core::SavedConversation opening, parked;
        const std::vector<core::ConversationImageKey> images;
        const std::vector<core::ConversationCheckpoint> checks_at_open;
        const core::ConversationView view{prefix,images,checks_at_open,true};
        require(core::conversation_snapshot_save(opening,view,main,geometry,draft.kv_state(),error),
                "capture real working state before resident admission");
        require(owner.park(0, prefix, admission, true, error), "park request A");
        require(core::conversation_snapshot_save(parked,view,a,geometry,*draft.slot_state(0),error) &&
                same_snapshot(opening,parked), "resident admission preserves every main and draft payload byte");
        require(main.gdn_state==main_gdn && a.gdn_state==slot_gdn && a.qsa_states==slot_qsa,
                "resident admission preserves working and stationary slot addresses");
        const auto* history = &owner.slots()[0].ids;
        require(!owner.park(1, prefix, admission, true, error), "same request cannot own two slots");
        require(owner.slots()[1].request == 0 && owner.slots()[1].ids.empty(), "duplicate refusal leaves B empty");
        require(owner.release_resources(error) && !owner.release_resources(error), "release once only");
        require(!owner.available(0) && owner.slots()[0].ids == prefix, "released resources retain prefix but refuse admission");
        require(owner.begin(-1, prompt, request, error) && owner.working().request == aid &&
                owner.working().sampling.seed == seed && owner.working().resume_slot == 0,
                "released solo resend retains identity and automatic seed");
        require(!owner.begin(0, prompt, request, error), "released scratch still refuses BGEN admission");
        require(owner.resume(0, nullptr, error), "restore same yielded solo while scratch is released");
        core::SavedConversation resumed;
        require(core::conversation_snapshot_save(resumed,view,main,geometry,draft.kv_state(),error) &&
                same_snapshot(opening,resumed) && main.gdn_state==main_gdn && a.gdn_state==slot_gdn,
                "resident resume preserves payloads and addresses while scratch is released");
        require(owner.init_resources(error), "reinitialize retained contexts");
        require(owner.park(0, prefix, admission, true, error), "retain solo continuation before promotion");
        require(owner.begin(1, prompt, request, error) && owner.working().request != aid &&
                owner.working().resume_slot == -1, "different physical destination cannot claim parked identity");
        auto altered_request = request;
        altered_request.sampling.penalty_repeat += 0.1f;
        require(owner.begin(0, prompt, altered_request, error) && owner.working().request != aid &&
                owner.working().resume_slot == -1, "different settings cannot claim parked identity");
        const std::vector<int64_t> other_prompt{11, 22, 33, 45};
        require(owner.begin(0, other_prompt, request, error) && owner.working().request != aid &&
                owner.working().resume_slot == -1, "different full prompt cannot claim parked identity");
        require(owner.begin(0, prompt, request, error), "promote yielded solo request into its parked slot");
        require(owner.working().request == aid && owner.working().sampling.seed == seed && owner.working().resume_slot == 0,
                "solo-to-resident promotion retains identity and automatic seed");
        require(owner.resume(0, nullptr, error), "restore yielded request");
        auto aliased_source = owner.completed_snapshot_source(0, error);
        require(aliased_source.has_value(), "completed source exposes the actual mutable checkpoint chain");
        admission.checkpoints = &aliased_source->checkpoints;
        require(owner.park(0, prefix, admission, true, error), "yield same identity again");
        require(owner.slots()[0].request == aid && &owner.slots()[0].ids == history, "repark keeps stable owner/history object");
        require(owner.slots()[0].checks.size() == 1 && owner.slots()[0].checks[0].ids == prefix,
                "aliased checkpoint survives repark");
        admission.checkpoints = &checkpoint_chain;
        owner.cancel(0);
        require(!owner.slots()[0].partial && !owner.slots()[0].active, "cancel invalidates yielded continuation");
        require(owner.begin(0, prompt, request, error), "cancelled yield starts a new request");
        const uint64_t cid = owner.working().request;
        require(cid > aid && owner.working().resume_slot == -1, "cancelled request cannot resurrect identity");
        require(owner.admit(0, prefix, admission, error), "reuse A slot for C");
        require(owner.slots()[0].request == cid && owner.slots()[0].ids == prefix && !owner.slots()[0].stop,
                "C starts with only its private context");
        require(!owner.release_resources(error), "active request prevents resource release");
        require(ledger.begin(operation, 0) && ledger.retain_submission(operation), "retain C consumer");
        owner.cancel(0);
        require(!owner.finish(0, error) && !owner.available(0), "finish refuses a retained consumer");
        ledger.cancel();
        require(!owner.finish(0, error) && owner.slots()[0].active,
                "cancelled submission still prevents finish/reuse");
        require(ledger.drain_submission(operation) && ledger.release(operation), "drain C consumer");
        require(owner.finish(0, error), "finish C");
        require(owner.release_resources(error), "release batch scratch with C cached");
        require(owner.begin(-1, prompt, request, error) && owner.resume(0, nullptr, error),
                "solo restore uses cached C while batch scratch is released");
        require(owner.slots()[0].request == cid && owner.slots()[0].ids == prefix &&
                !owner.available(0) && owner.init_resources(error),
                "solo restoration retains C without making batch admission ready");
        require(owner.begin(1, prompt, request, error), "begin independent B");
        require(owner.admit(1, prefix, admission, error), "admit B");
        const auto bid = owner.slots()[1].request;
        program::SessionFingerprint before, after;
        require(program::session_fingerprint(b, *draft.slot_state(1), geometry, 2, before, error), "fingerprint B before failed A transition");
        require(owner.begin(0, prompt, request, error), "begin replacement A");
        if (std::getenv("STRATA_TEST_RING_COPY_REFUSAL")) {
            // A terminal fixture: a failed draft operation intentionally keeps
            // its scratch lease poisoned, so no later lifecycle test can reuse it.
            const auto* ring=draft.slot_state(0);
            require(ring && ring->kv_mode==2,"ring refusal needs an actual host-backed draft slot");
            auto payload=[](const core::QsaState& st, bool value, bool resident) -> void* {
                const bool host=st.kv_mode!=0 && !resident;
                if (st.kv_q4 || (st.kv_hybrid && value))
                    return value?(host?st.host.v_q4:st.v_q4):(host?st.host.k_q4:st.k_q4);
                if (st.kv_int8 || st.kv_hybrid)
                    return value?(host?st.host.v_q:st.v_q):(host?st.host.k_q:st.k_q);
                return value?(host?st.host.v_pool:st.v_pool):(host?st.host.k_pool:st.k_pool);
            };
            void* ring_k=payload(*ring,false,true);
            void* ring_v=payload(*ring,true,true);
            const uint32_t marker=0x3c3c3c3c;
            require(cudaMemcpy(payload(draft.kv_state(),false,false),&marker,sizeof(marker),cudaMemcpyDefault)==cudaSuccess &&
                    cudaMemcpy(payload(draft.kv_state(),true,false),&marker,sizeof(marker),cudaMemcpyDefault)==cudaSuccess &&
                    cudaMemset(ring_k,0xa5,sizeof(marker))==cudaSuccess &&
                    cudaMemset(ring_v,0xa5,sizeof(marker))==cudaSuccess && cudaDeviceSynchronize()==cudaSuccess,
                    "distinguish real ring pages from the incoming authoritative draft");
            require(cudaGetLastError()==cudaSuccess,"ring refusal starts with a clear runtime error slot");
            fail_destination=ring_v;
            transfer_failed=false;
            require(!owner.admit(0,prefix,admission,error) && transfer_failed &&
                    error.find(cudaGetErrorString(cudaErrorInvalidValue))!=std::string::npos,
                    "actual draft ring copy refusal reaches resident admission");
            require(cudaDeviceSynchronize()==cudaSuccess && cudaGetLastError()==cudaSuccess,
                    "software refusal leaves no runtime error for a second last-error read");
            uint32_t key=0,value=0;
            require(cudaMemcpy(&key,ring_k,sizeof(key),cudaMemcpyDeviceToHost)==cudaSuccess &&
                    cudaMemcpy(&value,ring_v,sizeof(value),cudaMemcpyDeviceToHost)==cudaSuccess &&
                    key==marker && value==0xa5a5a5a5,
                    "ring refusal happened after key mutation and before value mutation");
            require(owner.slots()[0].failed && !owner.slots()[0].active && !owner.available(0) &&
                    !owner.slots()[0].request && !dispatch.slot_requests[0],
                    "failed draft ring cannot publish its resident target");
            require(!draft.context_idle(error) && !draft.bind_request(0,999,error) &&
                    !draft.bind_request(8,999,error) && !draft.save_slot(0,2,error) &&
                    !draft.restore_slot(0,2,error),
                    "poisoned draft scratch refuses both rebinding and save/restore reuse");
            require(!owner.begin(-1,prompt,request,error) && !owner.admit(0,prefix,admission,error),
                    "resident owner cannot reuse a poisoned draft through a successor request");
            require(program::session_fingerprint(b,*draft.slot_state(1),geometry,2,after,error) &&
                    before.gdn==after.gdn && before.ple==after.ple && before.kv==after.kv && before.draft==after.draft &&
                    before.tail==after.tail && before.dead==after.dead && before.pooled==after.pooled &&
                    owner.slots()[1].request==bid && owner.slots()[1].active && owner.slots()[1].ids==prefix,
                    "draft ring refusal preserves independent peer state and identity");
            std::printf("RESIDENT_RING_REFUSAL checks=%d poisoned=1 rebind_refused=1 reuse_refused=1 failures=0\n",checks);
            return 0;
        }
        // GDN is copied before PLE. Mark it so the partial write is observable.
        require(cudaMemset(main.gdn_state, 0x3c, sizeof(float)) == cudaSuccess, "mark source state");
        fail_destination = a.ple_hist;
        require(fail_destination != nullptr, "full-model PLE transfer target exists");
        require(!owner.admit(0, prefix, admission, error) && transfer_failed, "actual partial transfer fails");
        uint32_t changed = 0;
        require(cudaMemcpy(&changed, a.gdn_state, sizeof(changed), cudaMemcpyDeviceToHost) == cudaSuccess && changed == 0x3c3c3c3c,
                "failed transfer occurred after destination mutation");
        require(owner.slots()[0].failed && !owner.available(0) && !owner.slots()[0].active,
                "partial target cannot be published or reused");
        require(program::session_fingerprint(b, *draft.slot_state(1), geometry, 2, after, error), "fingerprint unaffected B");
        require(before.gdn == after.gdn && before.ple == after.ple && before.kv == after.kv && before.draft == after.draft &&
                before.tail == after.tail && before.dead == after.dead && before.pooled == after.pooled &&
                owner.slots()[1].request == bid && owner.slots()[1].active && owner.slots()[1].ids == prefix,
                "failure cannot mutate B state or identity");
        owner.cancel(1);
        require(owner.finish(1, error), "cancel survivor after failed peer");
        require(owner.release_resources(error) && owner.init_resources(error) && !owner.available(0) && owner.available(1),
                "resource re-init restores valid contexts only");
        require(owner.begin(1, prompt, request, error), "select retained B for restore failure");
        transfer_failed = false;
        fail_destination = main.ple_hist;
        require(!owner.resume(1, nullptr, error) && transfer_failed, "partial working restore fails");
        require(!owner.begin(-1, prompt, request, error) && !owner.available(1),
                "partial restore poisons working reuse and its source context");
        require(owner.release_resources(error), "release test resources");
        std::printf("RESIDENT_MODEL_CHECKS checks=%d partial_transfer=1 failures=0\n", checks);
        return 0;
    } catch (const std::exception& exception) {
        fail_destination = nullptr;
        std::fprintf(stderr, "RESIDENT_MODEL_CHECKS checks=%d failures=1 error=%s\n", checks, exception.what());
        return 1;
    }
}

// Host counterpart of the single-device fixture: retain the actual three-stage,
// four-seat allocation and read each payload on its owning device. No inference
// performance or mixed/ring residency claim is made by this lifecycle fixture.
int resident_host_model_checks(strata::program::ResidentSlots& owner, strata::core::ExpertDispatch& dispatch,
                               const strata::core::ModelGeometry& geometry,
                               const std::vector<strata::program::ResidentStage>& stages,
                               strata::core::MtpDrafter& draft) {
    using namespace strata;
    std::string error;
    int checks = 0;
    auto require = [&](bool ok, const char* message) {
        ++checks;
        if (!ok) throw std::runtime_error(std::string(message) + ": " + error);
    };
    auto device = [&](int ordinal) {
        require(cudaSetDevice(ordinal) == cudaSuccess, "select actual payload device");
        require(cudaDeviceSynchronize() == cudaSuccess, "join payload device");
    };
    try {
        require(stages.size() == 3 && owner.slots().size() == 4, "retain three stages and four seats");
        const std::vector<int32_t> prefix{11, 22};
        const std::vector<int64_t> prompt{11, 22, 33, 44};
        const std::vector<core::ConversationImageKey> images;
        const std::vector<core::ConversationCheckpoint> empty_checks;
        const core::ConversationView view{prefix, images, empty_checks, true};
        size_t snapshot_bytes = core::conversation_kv_bytes(draft.kv_state(), geometry, 2, false);
        for (const auto& stage : stages) {
            require(stage.main && stage.slots.size() == 4, "complete stationary stage owners");
            size_t bytes = 0;
            require(core::conversation_snapshot_bytes(view, *stage.main, geometry, nullptr, bytes, error),
                    "price full-model diagnostic image");
            require(bytes <= SIZE_MAX - snapshot_bytes, "bounded diagnostic image sum");
            snapshot_bytes += bytes;
        }
        constexpr uint64_t fixture_allowance = uint64_t{2} << 30;
        uint64_t available = 0, physical = 0;
        // Eight complete images conservatively cover the retained comparisons,
        // checkpoint copies and an in-progress capture. No admission credit.
        require(snapshot_bytes <= fixture_allowance / 8 &&
                core::available_memory_bytes(available, nullptr, &physical) &&
                available >= fixture_allowance && physical >= fixture_allowance + (uint64_t{8} << 30),
                "diagnostic images fit explicit 2 GiB allowance above 8 GiB reserve");
        struct Image {
            std::array<core::SavedConversation, 3> main;
            core::ConversationKv draft;
        };
        auto capture = [&](int slot) {
            Image result;
            for (size_t i = 0; i < stages.size(); ++i) {
                device(stages[i].device);
                auto* state = slot < 0 ? stages[i].main : stages[i].slots[slot];
                require(core::conversation_snapshot_save(result.main[i], view, *state, geometry, nullptr, error),
                        "capture all authoritative stage payloads");
            }
            device(stages.back().device);
            const auto* state = slot < 0 ? &draft.kv_state() : draft.slot_state(slot);
            require(state && core::conversation_kv_save(result.draft, *state, geometry, 2, false, error),
                    "capture actual draft owner on final device");
            return result;
        };
        auto equal = [](const Image& a, const Image& b) {
            for (size_t i = 0; i < a.main.size(); ++i)
                if (!same_snapshot(a.main[i], b.main[i])) return false;
            return test::equal(a.draft, b.draft);
        };
        std::array<std::array<const void*, 3>, 3> addresses{};
        for (size_t i = 0; i < stages.size(); ++i)
            addresses[i] = {stages[i].main->gdn_state, stages[i].slots[0]->gdn_state, stages[i].slots[0]->qsa_states};
        auto stationary = [&] {
            for (size_t i = 0; i < stages.size(); ++i)
                require(addresses[i] == std::array<const void*, 3>{
                            stages[i].main->gdn_state, stages[i].slots[0]->gdn_state, stages[i].slots[0]->qsa_states},
                        "all graph-facing owner addresses remain stationary");
        };
        program::ResidentRequest request;
        request.max_new = 8;
        request.scope.value[0] = 1;
        request.sampling.seed = 0;
        request.sampling.penalty_last_n = 2;
        require(!owner.init_resources(error), "repeated initialization refuses");
        const core::ExpertOperation operation{71, 0, 0, 1, 1};
        auto& ledger = dispatch.completion.at(0);
        require(ledger.begin(operation, 0) && ledger.retain_submission(operation), "retain real consumer");
        require(!owner.begin(0, prompt, request, error), "live consumer prevents rebinding");
        ledger.cancel();
        require(!owner.begin(0, prompt, request, error), "cancel is not consumer retirement");
        require(ledger.drain_submission(operation) && ledger.release(operation), "retire retained consumer");
        require(owner.begin(-1, prompt, request, error), "begin working request");
        const auto identity = owner.working().request, seed = owner.working().sampling.seed;
        require(seed != 0, "resolved seed retained across yields");
        std::vector<core::ConversationCheckpoint> checkpoint_chain(1);
        auto& checkpoint = checkpoint_chain.front();
        checkpoint.ids = prefix;
        device(stages[0].device);
        require(core::conversation_checkpoint_save(checkpoint, *stages[0].main, geometry, error),
                "capture first-stage checkpoint");
        checkpoint.stage_parts.resize(stages.size() - 1);
        for (size_t i = 1; i < stages.size(); ++i) {
            device(stages[i].device);
            checkpoint.stage_parts[i - 1].ids = prefix;
            require(core::conversation_checkpoint_save(checkpoint.stage_parts[i - 1], *stages[i].main, geometry, error),
                    "capture every later-stage checkpoint");
        }
        program::ResidentAdmission admission{33, &checkpoint_chain};
        auto opening = capture(-1);
        require(owner.park(0, prefix, admission, true, error), "park working request");
        require(equal(opening, capture(0)), "admission preserves every stage and draft payload");
        stationary();
        require(!owner.park(1, prefix, admission, true, error) && !owner.slots()[1].request,
                "one request cannot own two seats");
        require(owner.release_resources(error) && !owner.release_resources(error), "release exactly once");
        require(owner.begin(-1, prompt, request, error) && owner.working().request == identity &&
                owner.working().sampling.seed == seed, "released resend preserves identity and seed");
        require(owner.resume(0, nullptr, error) && equal(opening, capture(-1)),
                "live resume preserves every stage and draft payload");
        require(owner.init_resources(error) && owner.park(0, prefix, admission, true, error),
                "reinitialize and retain the continuation");
        require(owner.begin(0, prompt, request, error) && owner.working().request == identity &&
                owner.working().sampling.seed == seed && owner.resume(0, &checkpoint, error) &&
                equal(opening, capture(-1)), "selected-cut resume preserves payload and identity");
        stationary();
        owner.cancel(0);
        require(owner.begin(0, prompt, request, error) && owner.working().request > identity &&
                owner.working().resume_slot == -1 && owner.admit(0, prefix, admission, error),
                "cancelled identity cannot resurrect in successor");
        owner.cancel(0);
        require(owner.finish(0, error), "finish cancelled successor with no submitted row consumer");
        require(owner.begin(1, prompt, request, error) && owner.admit(1, prefix, admission, error),
                "admit independent peer");
        opening = capture(1);
        const auto peer_identity = owner.slots()[1].request;
        require(owner.begin(0, prompt, request, error), "begin replacement before injected failure");
        transfer_failed = false;
        const bool draft_failure = std::getenv("STRATA_TEST_DRAFT_COPY_REFUSAL") != nullptr;
        if (draft_failure) {
            device(stages.back().device);
            const auto* target = draft.slot_state(0);
            require(target && target->kv_mode == 0 && draft.kv_state().kv_mode == 0,
                    "draft refusal exercises canonical resident owners, not a changed ring profile");
            auto payload = [](const core::QsaState& state, bool value) -> void* {
                if (state.kv_q4 || (state.kv_hybrid && value)) return value ? state.v_q4 : state.k_q4;
                if (state.kv_int8 || state.kv_hybrid) return value ? state.v_q : state.k_q;
                return value ? state.v_pool : state.k_pool;
            };
            require(cudaMemset(payload(draft.kv_state(), false), 0x3c, 4) == cudaSuccess &&
                    cudaMemset(payload(*target, false), 0xa5, 4) == cudaSuccess &&
                    cudaMemset(payload(*target, true), 0xa5, 4) == cudaSuccess &&
                    cudaDeviceSynchronize() == cudaSuccess, "mark distinct draft key/value payloads");
            fail_destination = payload(*target, true);
            require(!owner.admit(0, prefix, admission, error) && transfer_failed, "partial draft admission refuses");
            device(stages.back().device);
            uint32_t key = 0, value = 0;
            require(cudaMemcpy(&key, payload(*target, false), 4, cudaMemcpyDeviceToHost) == cudaSuccess &&
                    cudaMemcpy(&value, payload(*target, true), 4, cudaMemcpyDeviceToHost) == cudaSuccess &&
                    key == 0x3c3c3c3c && value == 0xa5a5a5a5, "draft refusal follows K mutation, before V mutation");
            require(!draft.context_idle(error) && !draft.bind_request(0, 999, error) &&
                    !draft.save_slot(0, 2, error) && !draft.restore_slot(0, 2, error) &&
                    !owner.begin(-1, prompt, request, error), "partial draft poisons all owner reuse");
        } else {
            device(stages[0].device);
            require(cudaMemset(stages[0].main->gdn_state, 0x3c, 4) == cudaSuccess, "mark main source");
            fail_destination = stages[0].slots[0]->ple_hist;
            require(fail_destination && !owner.admit(0, prefix, admission, error) && transfer_failed,
                    "partial main admission refuses");
            device(stages[0].device);
            uint32_t changed = 0;
            require(cudaMemcpy(&changed, stages[0].slots[0]->gdn_state, 4, cudaMemcpyDeviceToHost) == cudaSuccess &&
                    changed == 0x3c3c3c3c, "main refusal follows real destination mutation");
        }
        require(owner.slots()[0].failed && !owner.slots()[0].active && !owner.available(0) &&
                equal(opening, capture(1)) && owner.slots()[1].request == peer_identity && owner.slots()[1].active,
                "partial target is unavailable and every peer stage remains intact");
        if (!draft_failure) {
            owner.cancel(1);
            require(owner.finish(1, error) && owner.release_resources(error) && owner.init_resources(error) &&
                    !owner.available(0) && owner.available(1), "reinitialize only valid contexts");
            require(owner.begin(1, prompt, request, error), "select peer for working-resume failure");
            transfer_failed = false;
            fail_destination = stages[0].main->ple_hist;
            require(!owner.resume(1, nullptr, error) && transfer_failed &&
                    !owner.begin(-1, prompt, request, error) && !owner.available(1),
                    "partial working resume poisons reuse");
            require(owner.release_resources(error), "release fixture resources");
        }
        std::printf("RESIDENT_HOST_MODEL_CHECKS stages=3 seats=4 checks=%d draft_failure=%d failures=0\n",
                    checks, draft_failure);
        return 0;
    } catch (const std::exception& exception) {
        fail_destination = nullptr;
        std::fprintf(stderr, "RESIDENT_HOST_MODEL_CHECKS checks=%d failures=1 error=%s\n", checks, exception.what());
        return 1;
    }
}
