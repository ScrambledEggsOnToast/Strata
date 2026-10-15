#include "strata/core/conversation_cache.hpp"
#include "strata/core/conversation_reuse.hpp"
#include "strata/core/prefix_extent_probe.hpp"
#include "strata/program/prefix_selection.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <limits>

using namespace strata::core;

namespace {
const ConversationScope test_scope{{1}};
int checks = 0;
void check(bool value, const char* description) {
    ++checks;
    if (!value) { std::fprintf(stderr, "FAIL: %s\n", description); std::exit(1); }
}
SavedConversation image(std::initializer_list<int32_t> ids, bool cvec = true) {
    SavedConversation s;
    s.scope = test_scope;
    s.live.ids = ids;
    s.live.gdn.resize(64, 7);
    s.cvec = cvec;
    return s;
}
}

int main() {
    {
        auto make = [] {
            auto s = image({1,2,3,4,5,6,7,8,9});
            auto cut = s.live; cut.ids.resize(5); s.checkpoints.push_back(cut);
            for (int fmt : {3,1}) { // current profile: K8V4 main, INT8 draft
                const bool index = fmt == 3;
                auto shape = strata::kernels::qsa_real_shapes();
                shape.n_head_kv=1; shape.head_dim=64; shape.idx_dim=8;
                strata::plan::QsaImageBytes bytes;
                check(strata::plan::qsa_image_bytes(shape,9,fmt,index,bytes), "probe fixture layout");
                ConversationKv kv;
                kv.format=fmt; kv.cells=bytes.cells; kv.page_size=shape.page_size;
                kv.heads=1; kv.head_dim=64; kv.idx_dim=8; kv.pooled_rows=bytes.pooled_rows;
                kv.k.resize(bytes.k,7); kv.v.resize(bytes.v,7);
                kv.k_scale.resize(bytes.k_scale,7); kv.v_scale.resize(bytes.v_scale,7);
                kv.pooled.resize(bytes.pooled,7);
                s.kv.push_back(std::move(kv));
            }
            return s;
        };
        auto a=make(), b=make();
        b.live.ids[6]=17; // sibling branch after an actual common checkpoint
        auto r=probe_prefix_extents(a,b);
        check(std::string_view(r.status)=="equal" && r.cut==5 && r.eligible_bytes==968 &&
              r.duplicate_bytes==968 && r.aliased_bytes==0 && r.compared_bytes==968,
              "independent backing counted for complete K8V4 and draft pages plus completed index row");
        auto change = [](ConversationBuffer& data, size_t at) {
            check(data.visit(at,1,[](uint8_t* p,size_t,size_t) { *p=23; return true; }), "probe fixture mutation");
        };
        change(b.kv[0].k,4*64); change(b.kv[0].pooled,32); change(b.kv[1].k,4*64);
        r=probe_prefix_extents(a,b);
        check(std::string_view(r.status)=="equal" && r.duplicate_bytes==968,
              "partial pages, moving index row and draft tail excluded from duplicate witness");
        change(b.kv[0].k,0);
        r=probe_prefix_extents(a,b);
        check(std::string_view(r.status)=="different" && r.duplicate_bytes==0 && r.compared_bytes>0,
              "equal token prefixes cannot hide unequal authoritative KV");
        b=make(); b.scope=ConversationScope{{2}};
        r=probe_prefix_extents(a,b);
        check(std::string_view(r.status)=="scope" && r.compared_bytes==0, "foreign scope never reads KV");
        b=make(); b.cvec=!a.cvec;
        r=probe_prefix_extents(a,b);
        check(std::string_view(r.status)=="scope" && r.compared_bytes==0, "steering mismatch never reads KV");
        b=make(); b.checkpoints.clear();
        r=probe_prefix_extents(a,b);
        check(std::string_view(r.status)=="no-cut" && r.compared_bytes==0, "equal tokens do not manufacture recurrent cut");
        b=make(); b.kv[1].v.pop_back();
        r=probe_prefix_extents(a,b);
        check(std::string_view(r.status)!="equal" && r.compared_bytes==0, "malformed final draft payload rejected before comparisons");
        b=make();
        r=probe_prefix_extents(a,b,1);
        check(std::string_view(r.status)=="limit" && r.compared_bytes==0, "budget refusal is not a zero-opportunity finding");
        r=probe_prefix_extents(a,a);
        check(std::string_view(r.status)=="equal" && r.duplicate_bytes==0 && r.aliased_bytes==968,
              "already shared backing never counted as duplicated allocation");
        a.checkpoints[0].ids.resize(4); b.checkpoints[0].ids.resize(4);
        r=probe_prefix_extents(a,b);
        check(std::string_view(r.status)=="equal" && r.duplicate_bytes==440,
              "draft final page stays private even when recurrent cut aligns to page boundary");
        a.stage_images.push_back(make()); b.stage_images.push_back(make());
        r=probe_prefix_extents(a,b);
        check(std::string_view(r.status)=="no-cut", "every stage must hold the same recurrent cut");
        a.checkpoints[0].ids.push_back(5); b.checkpoints[0].ids.push_back(5);
        r=probe_prefix_extents(a,b);
        check(std::string_view(r.status)=="equal" && r.duplicate_bytes==1936,
              "complete staged siblings include each stage exactly once");
        const size_t charged=a.bytes();
        const auto* payload=a.live.gdn.data();
        (void) probe_prefix_extents(a,b);
        check(a.bytes()==charged && a.live.gdn.data()==payload && a.kv[0].k==b.kv[0].k,
              "observer does not change accounting, ownership or contents");
    }
    {
        auto make_segmented = [](bool shifted) {
            auto s = image({});
            s.live.ids.assign(4097, 1);
            s.checkpoints.push_back(s.live);
            s.checkpoints.back().ids.resize(4096);
            auto shape = strata::kernels::qsa_real_shapes();
            shape.n_head_kv = 129; shape.head_dim = 64; shape.idx_dim = 8;
            strata::plan::QsaImageBytes bytes;
            check(strata::plan::qsa_image_bytes(shape, 4097, 3, true, bytes), "segmented probe layout");
            ConversationKv kv;
            kv.format = 3; kv.cells = bytes.cells; kv.page_size = shape.page_size;
            kv.heads = 129; kv.head_dim = 64; kv.idx_dim = 8; kv.pooled_rows = bytes.pooled_rows;
            if (shifted) kv.k.resize(17, 7);
            kv.k.resize(bytes.k, 7); kv.v.resize(bytes.v, 7);
            kv.k_scale.resize(bytes.k_scale, 7); kv.v_scale.resize(bytes.v_scale, 7);
            kv.pooled.resize(bytes.pooled, 7);
            s.kv.push_back(std::move(kv));
            return s;
        };
        auto a = make_segmented(false), b = make_segmented(true);
        auto r = probe_prefix_extents(a, b);
        check(std::string_view(r.status) == "equal" && r.cut == 4096,
              "equal eligible bytes survive different segment boundaries");
        check(b.kv[0].k.visit(2 * ConversationBuffer::segment_bytes + 7, 1,
                             [](uint8_t* p, size_t, size_t) { *p = 23; return true; }),
              "third segment mutation lies inside the eligible prefix");
        r = probe_prefix_extents(a, b);
        check(std::string_view(r.status) == "different" && r.duplicate_bytes == 0,
              "a third-segment mismatch invalidates the complete-cut witness");
    }
    {
        // Explicit buffer policy keeps the rest of this executable on its normal
        // default path; no environment-dependent expectations in these fixtures.
        auto make = [](int branch) {
            auto s = image({});
            s.live.ids.assign(2305, 1);
            s.live.ids[2100] = branch;
            auto root = s.live; root.ids.resize(2048); s.checkpoints.push_back(root);
            auto deep = s.live; deep.ids.resize(2304); s.checkpoints.push_back(deep);
            for (int fmt : {3, 1}) {
                const bool index = fmt == 3;
                auto shape = strata::kernels::qsa_real_shapes();
                shape.n_head_kv = 1; shape.head_dim = 64; shape.idx_dim = 8;
                strata::plan::QsaImageBytes bytes;
                check(strata::plan::qsa_image_bytes(shape, 2305, fmt, index, bytes), "sharing fixture layout");
                ConversationKv kv;
                kv.format = fmt; kv.cells = bytes.cells; kv.page_size = shape.page_size;
                kv.heads = 1; kv.head_dim = 64; kv.idx_dim = 8; kv.pooled_rows = bytes.pooled_rows;
                const std::array<ConversationBuffer*, 5> parts{&kv.k,&kv.v,&kv.k_scale,&kv.v_scale,&kv.pooled};
                const std::array<uint64_t, 5> sizes{bytes.k,bytes.v,bytes.k_scale,bytes.v_scale,bytes.pooled};
                for (size_t i = 0; i < parts.size(); ++i) {
                    *parts[i] = ConversationBuffer(true);
                    parts[i]->resize(size_t(sizes[i]), 7);
                }
                s.kv.push_back(std::move(kv));
            }
            return s;
        };
        auto pointer = [](const ConversationBuffer& b, size_t at) {
            const uint8_t* out = nullptr;
            b.visit(at, 1, [&](const uint8_t* p, size_t, size_t) { out = p; return true; });
            return out;
        };
        auto a = make(2), b = make(3), c = make(4);
        PrefixBackingLedger before; before.add(a); before.add(b); before.add(c);
        const auto* tail = pointer(b.kv[0].k, 2048 * 64);
        auto r = share_prefix_extents(b, a);
        check(std::string_view(r.status) == "equal", "authorized completed common cut accepted");
        share_prefix_extents(c, a);
        auto aliases = probe_prefix_extents(a, b);
        check(aliases.aliased_bytes > 0 && aliases.duplicate_bytes > 0,
              "complete allocations share; subextent scales and private tails remain separate");
        PrefixBackingLedger after; after.add(a); after.add(b); after.add(c);
        check(!after.truncated && after.unique_payload < before.unique_payload &&
              before.unique_payload - after.unique_payload == 2 * aliases.aliased_bytes &&
              after.logical_payload == before.logical_payload,
              "unique allocation ledger counts three siblings once, not three repeated pairs");
        check(pointer(b.kv[0].k, 2048 * 64) == tail,
              "sharing frees full prefix allocations without relocating the private tail");
        check(pointer(a.kv[1].k, 2044 * 64) != pointer(b.kv[1].k, 2044 * 64),
              "draft last possible uncomputed page stays private at aligned cut");
        const size_t peak = b.kv[0].k.allocation_peak(b.kv[0].k.size());
        check(peak >= b.kv[0].k.bytes() + 2 * ConversationBuffer::prefix_extent_bytes,
              "same-size rewrite admission includes every shared extent replacement");
        const auto* original = pointer(a.kv[0].k, 0);
        check(b.kv[0].k.visit(0, 1, [](uint8_t* p, size_t, size_t) { *p = 29; return true; }),
              "mutable visitor can rewrite a shared prefix after detaching");
        check(*pointer(a.kv[0].k, 0) == 7 && *pointer(c.kv[0].k, 0) == 7 &&
              pointer(b.kv[0].k, 0) != original && *pointer(b.kv[0].k, 0) == 29,
              "prefix mutation leaves both other branches and their backing intact");
        check(peak >= b.kv[0].k.bytes(), "detachment stays inside admitted buffer peak");
        check(c.kv[0].k.allocation_peak(17) >= c.kv[0].k.bytes() + ConversationBuffer::prefix_extent_bytes,
              "partial shrink prices detachment before changing segment length");
        c.kv[0].k.resize(17);
        check(a.kv[0].k.size() > 17 && *pointer(a.kv[0].k, 0) == 7,
              "shrinking shared prefix cannot truncate the donor");
        for (int mismatch = 0; mismatch < 6; ++mismatch) {
            auto isolated = make(5);
            if (mismatch == 0) isolated.scope = ConversationScope{{2}};
            if (mismatch == 1) isolated.cvec = !a.cvec;
            if (mismatch == 2) isolated.geometry[0] = 1;
            if (mismatch == 3) isolated.layer_hi = 1;
            if (mismatch == 4) isolated.checkpoints.erase(isolated.checkpoints.begin());
            if (mismatch == 5) isolated.kv.back().v.pop_back();
            const auto* own = pointer(isolated.kv[0].k, 0);
            auto refused = share_prefix_extents(isolated, a);
            check(std::string_view(refused.status) != "equal" && refused.compared_bytes == 0 &&
                  pointer(isolated.kv[0].k, 0) == own,
                  "scope, steering, geometry, carve, checkpoint and format-size refusals preserve ownership");
        }
        auto unequal = make(6);
        unequal.kv.back().v.visit(0, 1, [](uint8_t* p, size_t, size_t) { *p = 99; return true; });
        const auto* unchanged = pointer(unequal.kv[0].k, 0);
        r = share_prefix_extents(unequal, a);
        check(std::string_view(r.status) == "different" && pointer(unequal.kv[0].k, 0) == unchanged,
              "late-layer mismatch rejects entire cut before sharing earlier layers");
        auto staged_a = make(2), staged_b = make(3);
        staged_a.stage_images.push_back(make(2)); staged_b.stage_images.push_back(make(3));
        staged_b.stage_images[0].checkpoints.erase(staged_b.stage_images[0].checkpoints.begin());
        r = share_prefix_extents(staged_b, staged_a);
        check(std::string_view(r.status) != "equal" && probe_prefix_extents(staged_a, staged_b).aliased_bytes == 0,
              "missing checkpoint on one stage prevents sharing every stage");
        staged_b.stage_images[0] = make(3);
        r = share_prefix_extents(staged_b, staged_a);
        check(std::string_view(r.status) == "equal" &&
              probe_prefix_extents(staged_a, staged_b).aliased_bytes == 2 * aliases.aliased_bytes,
              "all-stage identity shares each stage exactly once");
        auto owner = make(2), survivor = make(3);
        share_prefix_extents(survivor, owner);
        const auto* surviving = pointer(survivor.kv[0].k, 0);
        ConversationCache cache(32u << 20, 4);
        check(cache.put(std::move(owner)) && cache.put(std::move(survivor)), "shared siblings occupy original cache slots");
        check(cache.evict_oldest(), "donor can be evicted with live sibling references");
        auto remaining = cache.take(0);
        check(pointer(remaining.kv[0].k, 0) == surviving && *surviving == 7,
              "eviction and take preserve exactly the surviving allocation");
        cache.retain(std::move(remaining.kv), 2305);
        auto active = cache.take_reuse();
        check(pointer(active.kv[0].k, 0) == surviving, "active retained reuse keeps the same backing");
        const auto* same = pointer(active.kv[0].k, 0);
        active.kv[0].k.visit(0, 1, [](uint8_t* p, size_t, size_t) { *p = 31; return true; });
        check(pointer(active.kv[0].k, 0) == same && *same == 31,
              "last surviving owner writes without a redundant detach allocation");
        auto limited = make(7);
        r = share_prefix_extents(limited, a, 1);
        check(std::string_view(r.status) == "limit" && r.compared_bytes == 0,
              "comparison limit refuses sharing without a partial publication");
        PrefixBackingLedger bounded;
        for (size_t i = 0; i <= PrefixBackingLedger::max_backings; ++i) bounded.add(a.kv[0].k);
        check(bounded.truncated, "diagnostic reference bound reports incomplete coverage explicitly");
    }
    {
        auto source = image({1, 2, 3});
        source.live.stage_parts.resize(1);
        source.live.stage_parts[0].gdn.assign(32, 9);
        source.live.seal();
        const auto* root = source.live.gdn.data();
        const auto* stage = source.live.stage_parts[0].gdn.data();
        auto reader = source.live;
        check(reader.gdn.data()==root && reader.stage_parts[0].gdn.data()==stage,
              "sealed checkpoint copies retain exact root and stage allocations");
        reader.used = 17; reader.pinned = true;
        check(source.live.used!=reader.used && !source.live.pinned,
              "shared payload does not share LRU or pin metadata");
        reader.gdn.write()[0] = 42;
        check(reader.gdn.data()!=root && source.live.gdn[0]==7 && reader.gdn[0]==42,
              "mutating a sealed checkpoint detaches without changing another branch");
        auto survivor = source.live;
        source = {};
        check(survivor.gdn.data()==root && survivor.stage_parts[0].gdn.data()==stage,
              "evicting source preserves held root and stage readers");
        const size_t payload_charge = survivor.gdn.bytes();
        auto another = survivor;
        check(another.gdn.bytes()==payload_charge && payload_charge>=survivor.gdn.size(),
              "each shared reference retains its conservative payload charge");
    }

    {
        ConversationScope scope;
        check(conversation_scope_parse("0100000000000000000000000000000000000000000000000000000000000000", scope) &&
              scope == test_scope, "exact-width scope round trip");
        for (const auto text : {"", "01", "010000000000000000000000000000000000000000000000000000000000000",
                               "0000000000000000000000000000000000000000000000000000000000000000",
                               "z100000000000000000000000000000000000000000000000000000000000000"})
            check(!conversation_scope_parse(text, scope) && scope == test_scope,
                  "invalid scope is refused without overwriting a populated owner");
        ConversationCache cache(4096, 2);
        auto a = image({1, 2, 3});
        a.checkpoints.emplace_back();
        a.checkpoints.back().ids = {1, 2};
        check(cache.put(std::move(a)), "park conversation A with its scope");
        const ConversationScope other{{2}};
        const std::vector<int64_t> second_turn{1, 2, 3, 4, 5};
        check(cache.best(second_turn, {}, true, test_scope).tokens == 3,
              "parked then resumed turn two reads only its suffix");
        check(cache.best(second_turn, {}, true, other).tokens == 0 &&
              cache.best(second_turn, {}, true, {}).tokens == 0,
              "identical cold conversation and absent namespace cannot recover A");
        auto b = image({1, 2, 3});
        b.scope = other;
        b.checkpoints.emplace_back();
        b.checkpoints.back().ids = {1, 2};
        check(cache.put(std::move(b)) && cache.size() == 2 && cache.superseded() == 0,
              "identical authorized branches remain separate, not superseded across scopes");
        check(cache.bytes() == 2 * (image({1, 2, 3}).bytes() +
              sizeof(ConversationCheckpoint) + 2 * sizeof(int32_t)),
              "both 32-byte namespaces are charged inside the bounded parking budget");
        check(cache.evict_oldest() && cache.best(second_turn, {}, true, test_scope).tokens == 0 &&
              cache.best(second_turn, {}, true, other).tokens == 3,
              "eviction cannot resurrect the retired conversation's namespace");
        auto unscoped = image({1, 2, 3});
        unscoped.scope = {};
        check(!cache.put(std::move(unscoped)), "unscoped snapshot is not parkable");
    }
    {
        ConversationBuffer bytes;
        const size_t first = ConversationBuffer::segment_bytes + 17;
        const size_t peak = bytes.allocation_peak(first);
        bytes.resize(first, 7);
        check(bytes.bytes() <= peak, "segmented payload and directory fit admitted bytes");
        uint8_t* original = nullptr;
        bytes.visit(0, 1, [&](uint8_t* p, size_t, size_t) { original = p; return true; });
        const size_t grown_peak = bytes.allocation_peak(first + 99);
        bytes.resize(first + 99, 9);
        bytes.visit(0, 1, [&](uint8_t* p, size_t, size_t) {
            check(p == original, "appending preserves existing payload addresses"); return true;
        });
        check(bytes.bytes() <= grown_peak, "growth including transient directory fits admission");
        std::vector<uint8_t> tail(116);
        check(bytes.read(tail.data(), ConversationBuffer::segment_bytes, tail.size()), "read spans segment boundaries");
        check(std::all_of(tail.begin(), tail.begin()+17, [](auto v){return v==7;}) &&
              std::all_of(tail.begin()+17, tail.end(), [](auto v){return v==9;}), "growth preserves prefix and initializes only suffix");
        auto copy = bytes;
        bytes.resize(ConversationBuffer::segment_bytes + 10);
        bytes.resize(first, 7); bytes.resize(first + 99, 9);
        check(bytes == copy, "equality ignores differing segmentation after rewind and regrowth");
        check(!bytes.read(tail.data(), bytes.size()-1, 2), "range check rejects a truncated payload");
        check(bytes.allocation_peak(SIZE_MAX) == SIZE_MAX, "allocation estimate rejects overflow");
    }
    {
        ConversationBuffer bytes;
        for (size_t n=1;n<=4096;++n) {
            const size_t peak = bytes.allocation_peak(n*17);
            bytes.resize(n*17,7);
            check(bytes.bytes() <= peak,"small append stays within the predicted allocation peak");
        }
        size_t segments = 0;
        bytes.visit(0,bytes.size(),[&](const uint8_t*,size_t,size_t){++segments;return true;});
        check(segments <= 4,"thousands of small turns do not create thousands of restore transfers");
    }
    {
        ConversationKv layer;
        layer.k.resize(400);
        std::vector<ConversationKv> layers;
        layers.push_back(std::move(layer));
        const size_t retained = layers.capacity()*sizeof(ConversationKv) + layers[0].bytes();
        const size_t parked = image({1,2,3}).bytes();
        ConversationCache cache(retained + parked, 4);
        check(cache.put(image({1,2,3})), "park before retaining active storage");
        cache.retain(std::move(layers), 16);
        check(cache.bytes() == retained + parked && cache.size() == 1, "active retained storage consumes bytes but no parked slot");
        cache.limit_reuse(9); cache.limit_reuse(12);
        auto reuse = cache.take_reuse();
        check(reuse.unchanged_tokens == 9, "a later continuation cannot undo a rewind's dirty boundary");
        check(cache.bytes() == parked, "taking retained buffers releases their budget accounting");
        cache.retain(std::move(reuse.kv), 16);
        check(!cache.can_fit(parked),"retained storage is counted when checking a capture without eviction");
        check(cache.retained_bytes() == retained && cache.size() == 1 && cache.evictions() == 0,
              "optional reuse admission does not evict or release anything");
        check(!cache.can_fit(SIZE_MAX) && !cache.can_fit(1,SIZE_MAX),"non-mutating reservation rejects overflow");
        check(cache.reserve_snapshot(parked), "reservation can discard optional active buffers");
        check(cache.retained_bytes() == 0 && cache.size() == 1 && cache.evictions() == 0,
              "pressure drops retained storage before evicting parked conversations");
    }
    const std::vector<int64_t> a = {1, 2, 3, 4}, b = {9, 8, 7, 6};
    {
        ConversationCache cache(1024, 2);
        check(cache.put(image({1, 2, 3})), "park A");
        check(cache.put(image({9, 8, 7})), "park B");
        auto match = cache.best(a, {}, true, test_scope);
        check(match.tokens == 3 && match.live, "A/B/A: recover A");
        auto restored = cache.take(match.index);
        check(restored.live.ids == std::vector<int32_t>({1, 2, 3}), "taking selected A preserves identity");
        check(cache.size() == 1 && cache.best(b, {}, true, test_scope).tokens == 3, "B remains parked");
        check(cache.bytes() == image({9,8,7}).bytes(), "byte accounting after take");
        check(cache.put(std::move(restored)), "park returned A as newest");
        check(cache.put(image({5, 6})), "evict oldest by slot limit");
        check(cache.best(b, {}, true, test_scope).tokens == 0 && cache.best(a, {}, true, test_scope).tokens == 3, "B evicted before A");
        check(cache.evictions() == 1, "eviction counter");
    }
    {
        // evict_oldest(): the physical-RAM admission gate in generate.cpp frees parked conversations
        // one at a time and re-checks the host's free memory, so this has to be exactly the oldest-first
        // step make_room() takes - same accounting, same counters - and it has to stop at empty.
        ConversationCache cache(4096, 4);
        cache.put(image({1, 2, 3}));
        cache.put(image({9, 8, 7}));
        check(cache.size() == 2 && cache.slots() == 4, "slots() reports the configured limit");
        const size_t both = cache.bytes();
        check(cache.evict_oldest(), "evict_oldest reports an eviction");
        check(cache.size() == 1 && cache.evictions() == 1, "evict_oldest drops exactly one, oldest first");
        check(cache.best(a, {}, true, test_scope).tokens == 0 && cache.best(b, {}, true, test_scope).tokens == 3,
              "the oldest conversation went, the newest stayed");
        check(cache.bytes() == both - image({1, 2, 3}).bytes(), "evict_oldest releases the entry's bytes");
        cache.evict_oldest();
        check(cache.size() == 0 && cache.bytes() == 0, "evicting the last parked conversation empties the cache");
        check(!cache.evict_oldest(), "the gate's loop can reach an empty cache: false");
        check(cache.size() == 0 && cache.evictions() == 2, "evict_oldest on an empty cache is a no-op");
    }
    {   // the RAM gate's eviction skips a conversation that holds a pinned shared prefix, and says so (false)
        ConversationCache cache(1 << 20, 4);
        auto pinned = image({1, 2, 3});
        pinned.checkpoints.push_back(ConversationCheckpoint{});
        pinned.checkpoints.back().ids = {1, 2};
        pinned.checkpoints.back().pinned = true;
        pinned.checkpoints.back().gdn.resize(64, 1);
        cache.put(std::move(pinned));
        cache.put(image({9, 8, 7}));
        check(cache.evict_oldest() && cache.size() == 1, "the oldest unpinned conversation went (the pinned one is older)");
        check(!cache.evict_oldest() && cache.size() == 1, "only a pinned one is left: nothing more can go");
        // #879: after non-finite logits every parked conversation goes, the pinned one too
        cache.put(image({4, 5, 6}));
        std::vector<ConversationKv> kv(1);
        kv[0].cells = 3;
        kv[0].k.resize(64, 1);
        cache.retain(std::move(kv), 3);
        check(cache.retained_bytes() > 0, "a retained K/V before clear()");
        check(cache.clear() == 2, "clear() reports how many were parked");
        check(cache.size() == 0 && cache.bytes() == 0 && cache.retained_bytes() == 0, "clear() leaves nothing parked");
        check(cache.best(std::vector<int64_t>{1, 2, 3, 4}, {}, true, test_scope).tokens == 0, "a cleared cache matches nothing");
        check(cache.clear() == 0, "clear() on an empty cache");
    }
    {
        auto s = image({1, 2, 3});
        ConversationCheckpoint cp;
        cp.ids = {1, 2};
        s.checkpoints.push_back(cp);
        ConversationCache cache(4096, 4);
        cache.put(std::move(s));
        auto match = cache.best(std::vector<int64_t>{1, 2, 9, 4}, {}, true, test_scope);
        check(match.tokens == 2 && !match.live, "edited suffix falls back to parked checkpoint");
        check(cache.best(a, {}, true, test_scope).tokens == 3, "live prefix beats shorter checkpoint");
        check(cache.best(a, {}, false, test_scope).tokens == 0, "steering mode is isolated");
        check(cache.best(std::vector<int64_t>{1, 2}, {}, true, test_scope).tokens == 0, "equal-length checkpoint cannot consume last token");
        check(cache.best(std::vector<int64_t>{1}, {}, true, test_scope).tokens == 0, "short prompt cannot match");
        check(cache.best(std::vector<int64_t>{}, {}, true, test_scope).tokens == 0, "empty prompt cannot match");
    }
    {
        auto s = image({1, 2, 3});
        s.live.imgs = {{1, 123}};
        ConversationCache cache(1024, 3);
        cache.put(std::move(s));
        check(cache.best(a, {{1,123}}, true, test_scope).tokens == 3, "same image can resume");
        check(cache.best(a, {{1,124}}, true, test_scope).tokens == 0, "different image pixels invalidate same pad tokens");
        check(cache.best(a, {}, true, test_scope).tokens == 0, "missing image invalidates prefix");
        check(cache.best(a, {{1,123},{2,45}}, true, test_scope).tokens == 0, "additional image in cached prefix invalidates");
        check(cache.best(a, {{1,123},{3,45}}, true, test_scope).tokens == 3, "image after cached prefix does not invalidate");
    }
    {
        const size_t one = image({1,2,3}).bytes();
        ConversationCache cache(one*2, 8);
        cache.put(image({1,2,3})); cache.put(image({9,8,7}));
        check(cache.bytes() == one*2, "budget holds two exact-sized images");
        auto held = cache.take(cache.best(a, {}, true, test_scope).index);
        check(cache.reserve_snapshot(one, held.bytes()), "count in-flight image when reserving outgoing snapshot");
        check(cache.size() == 0, "in-flight reservation evicts otherwise fitting B");
        check(cache.put(image({5,6,7}), held.bytes()), "insert with in-flight accounting");
        check(cache.bytes()+held.bytes() <= one*2, "exchange obeys byte budget");
        check(!cache.reserve_snapshot(one+1, one), "oversized exchange rejected");
        check(cache.size() == 1, "oversized snapshot does not evict useful entries");
        auto huge = image({1}); huge.live.gdn.resize(one*3);
        check(!cache.put(std::move(huge)), "oversized image rejected");
        check(cache.size() == 1, "oversized put leaves cache unchanged");
        check(!cache.reserve_snapshot(0, one*2+1), "held larger than budget cannot underflow");
    }
    for (bool pinned : {false, true}) {
        ConversationCache cache(1 << 20, 2);
        auto donor = image({1, 2, 3, 4}), other = image({9, 8, 7});
        if (pinned) {
            donor.checkpoints = {donor.live}; donor.checkpoints.back().pinned = true;
        }
        other.checkpoints = {other.live}; other.checkpoints.back().pinned = true;
        check(cache.put(std::move(donor)) && cache.put(std::move(other)), "park a donor beside a pinned history");
        auto held = cache.take(cache.best(std::vector<int32_t>{1, 2, 3, 4, 5}, {}, true, test_scope).index);
        check(!cache.put(image({6, 5, 4}), held.bytes(), 1),
              "outgoing capture cannot spend a non-consuming donor's reserved entry");
        check(cache.best(std::vector<int32_t>{9, 8, 7, 6}, {}, true, test_scope).tokens == 3,
              "refused outgoing capture preserves the existing pinned history");
        check(cache.put(std::move(held)) &&
              cache.best(std::vector<int32_t>{1, 2, 3, 4, 5}, {}, true, test_scope).tokens == 4,
              "one-shot donor returns with its complete original prefix");
        check(cache.evictions() == 0 && cache.size() == 2, "borrow and return do not evict either history");
    }
    {
        ConversationCache cache(1 << 20, 2);
        check(cache.put(image({1, 2, 3})) && cache.put(image({9, 8, 7})), "fill ordinary donor table");
        auto held = cache.take(0);
        check(cache.put(image({6, 5, 4}), held.bytes(), 1), "reserved-entry capture may evict an unrelated unpinned history");
        check(cache.put(std::move(held)) && cache.size() == 2 && cache.evictions() == 1 &&
              cache.best(std::vector<int32_t>{1, 2, 3, 4}, {}, true, test_scope).tokens == 3 &&
              cache.best(std::vector<int32_t>{6, 5, 4, 3}, {}, true, test_scope).tokens == 3,
              "capture plus donor return obey the entry limit without consuming the donor");
        check(!cache.put(image({7, 6, 5}), 0, std::numeric_limits<size_t>::max()),
              "invalid held-entry reservations cannot underflow the publication limit");
    }
    {
        ConversationCache cache(1 << 20, 4);
        auto first = image({1, 2, 3, 8});
        auto root = first.live;
        root.ids = {1, 2};
        first.checkpoints = {root};
        auto second = image({1, 2, 4, 9});
        second.checkpoints = {root};
        check(cache.put(std::move(first)) && cache.put(std::move(second)), "park no-ID root siblings");
        check(cache.best(std::vector<int32_t>{1, 2, 3, 8, 10}, {}, true, test_scope).tokens == 4 &&
              cache.best(std::vector<int32_t>{1, 2, 4, 9, 10}, {}, true, test_scope).tokens == 4,
              "unrelated histories sharing their sole root retain their own complete prefixes");
    }
    {
        ConversationCache cache(1 << 20, 1);
        auto first = image({1, 2, 3, 8});
        auto pin = first.live;
        pin.ids = {1, 2, 3};
        pin.pinned = true;
        first.checkpoints = {pin};
        check(cache.put(std::move(first)), "park explicitly pinned history");
        auto second = image({1, 2, 3, 8, 9});
        pin.pinned = false;
        second.checkpoints = {pin};
        check(!cache.put(std::move(second)), "a matching but unpinned cut cannot displace the only pinned history");
        bool preserved = false;
        cache.for_each_image([&](size_t, const SavedConversation& saved) { preserved = saved.pinned(); });
        check(preserved && cache.best(std::vector<int32_t>{1, 2, 3, 7}, {}, true, test_scope).tokens == 3,
              "refused replacement leaves the explicitly pinned prefix available");
    }
    {
        ConversationCache cache(1 << 20, 2);
        check(cache.put(image({9, 8, 7})) && cache.put(image({1, 2, 3})), "fill the published entry table");
        auto next = image({1, 2, 3, 4});
        check(cache.reserve_snapshot(next.bytes()), "admit capture bytes at a full entry limit");
        check(cache.best(std::vector<int32_t>{9, 8, 7, 6}, {}, true, test_scope).tokens == 3 &&
              cache.best(std::vector<int32_t>{1, 2, 3, 5}, {}, true, test_scope).tokens == 3,
              "a failed or abandoned capture has not spent a published history entry");
        check(cache.put(std::move(next)), "publish a checkpoint-free prefix extension");
        check(cache.best(std::vector<int32_t>{9, 8, 7, 6}, {}, true, test_scope).tokens == 3 &&
              cache.best(std::vector<int32_t>{1, 2, 3, 4, 5}, {}, true, test_scope).tokens == 4,
              "replacement retains the unrelated waiting history and advances its own prefix");
        check(cache.best(std::vector<int32_t>{1, 2, 3, 99}, {}, true, test_scope).tokens == 0,
              "an old unpinned fork cold-reads rather than fabricating a recurrent checkpoint");
        check(cache.put(image({1, 2, 3, 4})), "publish an equal live snapshot");
        check(cache.best(std::vector<int32_t>{9, 8, 7, 6}, {}, true, test_scope).tokens == 3,
              "equal live snapshots do not consume the unrelated history's entry");
    }
    {
        ConversationCache cache(1 << 20, 1);
        auto first = image({1, 2, 3, 8});
        auto pin = first.live;
        pin.ids = {1, 2, 3}; pin.pinned = true;
        first.checkpoints = {pin};
        check(cache.put(std::move(first)), "fill a pinned entry table");
        auto next = image({1, 2, 3, 8, 9});
        next.checkpoints = {pin};
        check(cache.put(std::move(next)), "a full pinned table permits a pin-preserving lineage replacement");
        bool kept_pin = false;
        cache.for_each_image([&](size_t, const SavedConversation& saved) { kept_pin = saved.pinned(); });
        check(kept_pin && cache.best(std::vector<int32_t>{1, 2, 3, 99}, {}, true, test_scope).tokens == 3 &&
              cache.best(std::vector<int32_t>{1, 2, 3, 8, 9, 99}, {}, true, test_scope).tokens == 5,
              "replacement retains the real pinned cut and the latest complete history");
    }
    for (int change = 0; change < 10; ++change) {
        ConversationCache cache(1 << 20, 4);
        auto old = image({1, 2, 3, 4}), next = image({1, 2, 3, 4, 5});
        std::vector<ConversationImageKey> query_images;
        switch (change) {
        case 0: next.scope = ConversationScope{{2}}; break;
        case 1: next.cvec = false; break;
        case 2:
            old.live.imgs = {{1, 42}}; next.live.imgs = {{1, 43}};
            query_images = old.live.imgs; break;
        case 3: next.live.imgs = {{1, 42}}; break;
        case 4: next.geometry[0] = 1; break;
        case 5: next.layer_hi = 1; break;
        case 6:
            old.kv.resize(1); next.kv.resize(1);
            old.kv[0].format = 1; next.kv[0].format = 2; break;
        case 7: next.stage_images.push_back(image({1, 2, 3, 4, 5})); break;
        case 8:
            old.stage_images = {image({1, 2, 3, 4}), image({1, 2, 3, 4})};
            next.stage_images = {image({1, 2, 3, 4, 5}), image({1, 2, 3, 4, 5})};
            old.stage_images[0].layer_hi = next.stage_images[1].layer_hi = 1;
            old.stage_images[1].layer_hi = next.stage_images[0].layer_hi = 2; break;
        case 9: next.live.ids.back() = 99; next.live.ids[3] = 8; break;
        }
        check(cache.put(std::move(old)) && cache.put(std::move(next)), "park non-equivalent replacement candidates");
        check(cache.best(std::vector<int32_t>{1, 2, 3, 4, 99}, query_images, true, test_scope).tokens == 4,
              "scope, mode, image, shape, stage and branch mismatches preserve the earlier history");
    }
    {
        ConversationCache cache(1 << 20, 1);
        auto old = image({1, 2, 3, 4}), next = image({1, 2, 3, 4, 5});
        next.live.imgs = {{4, 42}}; // an image after the old cut does not change that prefix
        check(cache.put(std::move(old)) && cache.put(std::move(next)), "later images permit a complete prefix extension");
        check(cache.best(std::vector<int32_t>{1, 2, 3, 4, 5, 99}, {{4, 42}}, true, test_scope).tokens == 5,
              "the newly extended image history restores its actual image identity");
    }
    for (int missing = 0; missing < 4; ++missing) {
        ConversationCache cache(1 << 20, 1);
        auto old = image({1, 2, 3, 4}), next = image({1, 2, 3, 4, 5});
        auto pin = old.live; pin.ids = {1, 2, 3}; pin.pinned = true;
        old.checkpoints = next.checkpoints = {pin};
        old.stage_images.push_back(image({1, 2, 3, 4}));
        next.stage_images.push_back(image({1, 2, 3, 4, 5}));
        old.stage_images[0].scope = next.stage_images[0].scope = {};
        old.stage_images[0].checkpoints = next.stage_images[0].checkpoints = {pin};
        if (missing == 1) next.checkpoints.clear(); // live prefix alone is not a retained pin
        if (missing == 2) next.stage_images[0].checkpoints.clear();
        if (missing == 3) next.stage_images[0].checkpoints[0].pinned = false;
        check(cache.put(std::move(old)), "park a pinned layered history");
        check(cache.put(std::move(next)) == (missing == 0), "every stage must preserve the explicit pinned cut");
        check(cache.best(std::vector<int32_t>{1, 2, 3, 99}, {}, true, test_scope).tokens == 3,
              "layered replacement or refusal leaves a real pinned prefix available");
    }
    {
        // A genuinely appended history replaces its older live-prefix snapshots,
        // leaving the unrelated parent reusable even after many child turns.
        auto cp = [](std::vector<int32_t> ids) { ConversationCheckpoint c; c.ids = std::move(ids); return c; };
        auto with = [](std::vector<int32_t> prefix, std::initializer_list<int32_t> more) {
            prefix.insert(prefix.end(), more); return prefix;
        };
        const std::vector<int32_t> root = {1, 2, 3, 4};
        SavedConversation main = image({});
        main.live.ids = with(with(root, {10, 11, 12}), {13, 14});
        main.checkpoints = {cp(root), cp(with(root, {10, 11, 12}))};
        const size_t big = 1 << 20;
        ConversationCache cache(big, 4);
        check(cache.put(std::move(main)), "park MAIN");
        std::vector<int32_t> history = with(root, {20});
        std::vector<ConversationCheckpoint> chain = {cp(root)};
        for (int turn = 1; turn <= 9; ++turn) {
            chain.push_back(cp(history));                        // the turn boundary the next request resumes from
            SavedConversation sub = image({});
            sub.live.ids = with(history, {900, (int32_t) turn});
            sub.checkpoints = chain;
            check(cache.put(std::move(sub)), "park SUB turn");
            history = with(history, {900, (int32_t) turn});
        }
        const auto m = cache.best(with(with(root, {10, 11, 12}), {13, 14, 15}), {}, true, test_scope);
        check(m.tokens == 9 && m.live, "MAIN still restores in full");
        const auto s = cache.best(with(history, {40}), {}, true, test_scope);
        check(s.tokens == (int64_t) history.size() && s.live, "SUB resumes its complete appended history");
    }
    {
        // Shared roots and shorter rewind cuts do not replace complete divergent histories.
        auto cp = [](std::vector<int32_t> ids) { ConversationCheckpoint c; c.ids = std::move(ids); return c; };
        ConversationCache cache(1 << 20, 8);
        SavedConversation other = image({1, 2, 3, 4, 50, 51, 52});
        other.checkpoints = {cp({1, 2, 3, 4}), cp({1, 2, 3, 4, 50, 51})};
        cache.put(std::move(other));
        cache.put(image({1, 2, 3, 4, 7, 7}));                    // no checkpoints
        SavedConversation steered = image({1, 2, 3, 4, 60, 61}, false);
        steered.checkpoints = {cp({1, 2, 3, 4, 60})};
        cache.put(std::move(steered));
        SavedConversation branch = image({1, 2, 3, 4, 60, 70, 71});
        branch.checkpoints = {cp({1, 2, 3, 4}), cp({1, 2, 3, 4, 60, 70})};
        cache.put(std::move(branch));
        SavedConversation incoming = image({1, 2, 3, 4, 60, 80, 81});
        incoming.checkpoints = {cp({1, 2, 3, 4}), cp({1, 2, 3, 4, 60})};
        check(cache.put(std::move(incoming)), "park a conversation sharing roots with all of them");
        check(cache.size() == 5 && cache.superseded() == 0, "none of them is superseded");
        SavedConversation same_state = image({1, 2, 3, 4, 60, 70});   // live equal to the branch's deepest point
        same_state.checkpoints = {cp({1, 2, 3, 4})};
        check(cache.put(std::move(same_state)), "park a shorter rewind branch");
        check(cache.best(std::vector<int32_t>{1, 2, 3, 4, 60, 70, 71, 99}, {}, true, test_scope).tokens == 7,
              "a rewind matching only an older checkpoint preserves the longer branch");
        SavedConversation huge = image({1, 2, 3, 4, 60, 80});
        huge.checkpoints = {cp({1, 2, 3, 4}), cp({1, 2, 3, 4, 60})};
        huge.live.gdn.resize(2 << 20);
        check(!cache.put(std::move(huge)) &&
              cache.best(std::vector<int32_t>{1, 2, 3, 4, 60, 80, 81, 99}, {}, true, test_scope).tokens == 7,
              "oversized replacement preserves the useful complete branch");
    }
    {
        ConversationCache disabled(0,4), no_slots(1024,0);
        check(!disabled.enabled() && !no_slots.enabled(), "both disable switches");
        check(!disabled.put(image({1,2,3})) && !no_slots.put(image({1,2,3})), "disabled cache stores nothing");
        check(disabled.best(a, {}, true, test_scope).tokens == 0, "disabled cache has no matches");
    }
    {
        // --conversation-cache-min-tokens: a conversation shorter than the minimum is not worth one of the slots.
        // Parking is a switch-time decision, so without this a client's short side requests park themselves and,
        // once the cache is full, evict the long conversation they were interleaved with (oldest first).
        ConversationCache any(1024, 4, 0), floor(1024, 4, 12288);
        check(any.enabled() && floor.enabled(), "the minimum does not disable the cache");
        check(any.wants(0) && any.wants(1) && any.wants(1 << 20), "0 parks every length, as before the flag");
        check(!floor.wants(0) && !floor.wants(1) && !floor.wants(12287), "below the minimum is not parked");
        check(floor.wants(12288) && floor.wants(12289), "the minimum itself and above are parked");
        // the CLI rejects a negative, but the predicate must not invert for one that reaches it another way
        ConversationCache negative(1024, 4, -1);
        check(negative.wants(0) && negative.wants(1), "a negative minimum parks everything, like 0");
        // wants() answers for the length alone; the gate is enabled() && wants(), so a disabled cache parks
        // nothing whatever the length - keep the two independent rather than folding one into the other
        ConversationCache disabled_min(0, 4, 12288);
        check(!disabled_min.enabled() && disabled_min.wants(1 << 20), "a disabled cache still answers for length");
    }
    {
        // layer-split parking: checkpoints moved apart into stage parts and put back, no running state copied
        auto running = [](std::initializer_list<int32_t> ids, uint8_t salt, size_t parts) {
            ConversationCheckpoint c;
            c.ids = ids;
            c.gdn.assign(256, salt);
            c.tails.assign(32, uint8_t(salt + 1));
            c.used = salt;
            for (size_t k = 0; k < parts; ++k) {
                ConversationCheckpoint p;
                p.gdn.assign(128, uint8_t(salt + 10 + k));
                c.stage_parts.push_back(std::move(p));
            }
            return c;
        };
        std::vector<ConversationCheckpoint> checks;
        checks.push_back(running({1, 2}, 3, 2));
        checks.push_back(running({1, 2, 3}, 5, 1));       // an incomplete set: kept aside, put back in place
        checks.push_back(running({1, 2, 3, 4}, 7, 2));
        const auto original = checks;
        const uint8_t* gdn0 = checks[0].gdn.data();
        const uint8_t* part1 = checks[2].stage_parts[1].gdn.data();
        auto split = conversation_checkpoints_split(std::move(checks), 2, ConversationCheckpointRetention::all);
        check(checks.empty() && split.stage0.size() == 2 && split.rest.size() == 1 && split.parts.size() == 2 &&
              split.parts[0].size() == 2 && split.parts[1].size() == 2, "split: complete checkpoints apart, the other aside");
        check(split.stage0[0].stage_parts.empty() && split.parts[1][1].ids == original[2].ids &&
              split.parts[1][1].used == original[2].used && split.parts[0][0].gdn == original[0].stage_parts[0].gdn,
              "split: every stage part carries its checkpoint's ids and LRU stamp");
        check(split.stage0[0].gdn.data() == gdn0 && split.parts[1][1].gdn.data() == part1,
              "split moves the running state (same buffers, no copy)");
        std::vector<ConversationCheckpoint> back;
        check(conversation_checkpoints_merge(std::move(split), back) && back.size() == 3, "merge puts them back");
        bool same = true;
        for (size_t j = 0; j < back.size(); ++j) {
            same = same && back[j].ids == original[j].ids && back[j].gdn == original[j].gdn &&
                   back[j].tails == original[j].tails && back[j].used == original[j].used &&
                   back[j].stage_parts.size() == original[j].stage_parts.size();
            for (size_t k = 0; same && k < back[j].stage_parts.size(); ++k)
                same = back[j].stage_parts[k].gdn == original[j].stage_parts[k].gdn;
        }
        check(same, "merge(split(x)) == x, in the original order");
        check(back[0].gdn.data() == gdn0 && back[2].stage_parts[1].gdn.data() == part1, "merge moves them back too");

        auto torn = conversation_checkpoints_split(std::vector<ConversationCheckpoint>(original), 2, ConversationCheckpointRetention::all);
        torn.parts[1].pop_back();
        check(!conversation_checkpoints_merge(std::move(torn), back) && back.empty(), "a stage short of a part is refused");
        auto mixed = conversation_checkpoints_split(std::vector<ConversationCheckpoint>(original), 2, ConversationCheckpointRetention::all);
        mixed.parts[0][0].ids = {9, 9};
        check(!conversation_checkpoints_merge(std::move(mixed), back) && back.empty(), "a part of another checkpoint is refused");
        // a restored image: stage 0's checkpoints and each stage's, with no record of an incomplete one
        ConversationCheckpointSplit restored;
        restored.stage0.push_back(running({4, 5}, 11, 0));
        restored.parts.resize(2);
        for (auto& p : restored.parts) { auto c = running({4, 5}, 12, 0); p.push_back(std::move(c)); }
        check(conversation_checkpoints_merge(std::move(restored), back) && back.size() == 1 &&
              back[0].stage_parts.size() == 2, "a restored image's parts are all whole");
        auto none = conversation_checkpoints_split({}, 3, ConversationCheckpointRetention::all);
        check(conversation_checkpoints_merge(std::move(none), back) && back.empty(), "no checkpoints: nothing to do");
    }
    for (int pins = 0; pins <= 2; ++pins) {
        std::vector<ConversationCheckpoint> working(6);
        for (size_t i = 0; i < working.size(); ++i) {
            auto& cut = working[i];
            cut.ids.resize(i + 2);
            for (size_t j = 0; j < cut.ids.size(); ++j) cut.ids[j] = (int32_t)j + 1;
            cut.gdn = {(uint8_t)i}; cut.used = i;
            cut.pinned = (pins >= 1 && i == 1) || (pins == 2 && i == 3);
            cut.stage_parts.resize(2);
            for (auto& part : cut.stage_parts) {
                part.ids = cut.ids; part.used = cut.used; part.pinned = cut.pinned;
                part.gdn = {(uint8_t)(i + 10)};
            }
        }
        const auto original = working;
        auto selected = conversation_checkpoints_split(std::move(working), 2,
                                                       ConversationCheckpointRetention::parked_history);
        size_t at = 0;
        for (size_t i = 0; i < original.size(); ++i) {
            if (i != 0 && i + 1 != original.size() && !original[i].pinned) continue;
            check(at < selected.stage0.size() && selected.stage0[at].ids == original[i].ids &&
                  selected.stage0[at].gdn == original[i].gdn && selected.stage0[at].pinned == original[i].pinned,
                  "parked history contains its actual root, latest cut and every explicit pin");
            for (size_t stage = 0; stage < 2; ++stage)
                check(selected.parts[stage][at].ids == original[i].ids &&
                      selected.parts[stage][at].pinned == original[i].pinned &&
                      selected.parts[stage][at].gdn == original[i].stage_parts[stage].gdn,
                      "every retained cut keeps matching stage state and pin metadata");
            ++at;
        }
        check(at == selected.stage0.size(), "older unpinned cuts do not multiply across parked histories");
        check(conversation_checkpoints_merge(std::move(selected), working) && working.size() == original.size(),
              "capture restores the complete working checkpoint chain");
        for (size_t i = 0; i < original.size(); ++i) {
            check(working[i].ids == original[i].ids && working[i].gdn == original[i].gdn &&
                  working[i].pinned == original[i].pinned, "working rewind cuts and pins survive selective parking");
            for (size_t stage = 0; stage < 2; ++stage)
                check(working[i].stage_parts[stage].gdn == original[i].stage_parts[stage].gdn &&
                      working[i].stage_parts[stage].pinned == original[i].pinned,
                      "omitted and retained stage payloads both return to their working owners");
        }
    }
    {
        // the later stages' retained K/V, kept with the first stage's and limited together
        auto kv = [](size_t n) {
            std::vector<ConversationKv> v(2);
            for (auto& l : v) l.k.resize(n, 1);
            return v;
        };
        ConversationCache cache(1 << 20, 4);
        std::vector<std::vector<ConversationKv>> stage_kv;
        stage_kv.push_back(kv(1000));
        stage_kv.push_back(kv(2000));
        cache.retain(kv(500), 40, std::move(stage_kv));
        check(cache.retained_bytes() >= 2 * (500 + 1000 + 2000), "retained bytes count every stage");
        cache.limit_reuse(30);
        auto r = cache.take_reuse();
        check(r.kv.size() == 2 && r.stages.size() == 2 && r.unchanged_tokens == 30 &&
              r.stages[0].unchanged_tokens == 30 && r.stages[1].unchanged_tokens == 30 &&
              r.stages[1].captured_tokens == 40, "every stage's reuse is limited to the first rewrite");
        check(cache.retained_bytes() == 0, "taken once");
        std::vector<std::vector<ConversationKv>> big;
        big.push_back(kv(1 << 20));
        cache.retain(kv(500), 40, std::move(big));
        check(cache.retained_bytes() == 0, "a stage's K/V over the budget drops the whole reuse");
        std::vector<std::vector<ConversationKv>> one;
        one.push_back(kv(100));
        cache.retain(kv(500), 40, std::move(one));
        cache.limit_reuse(0);
        check(cache.retained_bytes() == 0, "namespace/steering changes and resident-owner mounts discard every stage's reuse");
        cache.retain(kv(500), 40);
        r = cache.take_reuse();
        check(r.kv.size() == 2 && r.stages.empty(), "no layer split: no stage reuse, as before");
    }
    {   // pin=N: a parked conversation that holds a pinned shared prefix is not the one evicted
        const size_t one = image({1,2,3}).bytes();
        auto pinned = image({1,2,3});
        pinned.checkpoints.push_back(ConversationCheckpoint{});
        pinned.checkpoints.back().ids = {1,2};
        pinned.checkpoints.back().pinned = true;
        pinned.checkpoints.back().gdn.resize(64, 1);
        const size_t pinned_bytes = pinned.bytes();
        ConversationCache big(pinned_bytes + one * 2 + 64, 3);
        check(pinned.pinned() && !image({1,2,3}).pinned(), "a conversation is pinned when one of its checkpoints is");
        check(big.put(std::move(pinned)), "the pinned conversation is parked first (the oldest)");
        check(big.put(image({7,8,9})) && big.put(image({4,5,6})) && big.size() == 3, "two more fill the slots");
        check(big.put(image({10,11,12})) && big.size() == 3 && big.evictions() == 1,
              "a fourth evicts the oldest unpinned one");
        check(big.best(std::vector<int32_t>{1,2,9,9}, {}, true, test_scope).tokens == 2, "the pinned prefix is still found");
        ConversationCache tiny(pinned_bytes + 8, 2);
        auto p2 = image({1,2,3});
        p2.checkpoints.push_back(ConversationCheckpoint{});
        p2.checkpoints.back().ids = {1,2};
        p2.checkpoints.back().pinned = true;
        p2.checkpoints.back().gdn.resize(64, 1);
        check(tiny.put(std::move(p2)), "a pinned one fits an empty cache");
        check(!tiny.put(image({7,8,9})) && tiny.size() == 1, "nothing else fits beside it: parking is refused, the pin stays");
    }
    {
        using namespace strata::program;
        PrefixCosts costs;
        PrefixWork resident{1000, 11}, parked{10000, 22}, park{5000, 33};
        check(!costs.transfer_ms(PrefixTransfer::resident_live, resident), "unobserved transfer is unknown");
        check(costs.transfer_ms(PrefixTransfer::park, {}) == 0.0, "no outgoing capture has zero cost");
        costs.transfer(PrefixTransfer::resident_live, resident, 10);
        costs.transfer(PrefixTransfer::parked, parked, 400);
        costs.transfer(PrefixTransfer::park, park, 5);
        costs.prefill(0, 2048, false, 409.6);
        const auto tail = costs.prefill_ms(1024, 2048, false);
        check(tail && std::abs(*tail - 204.8) < 1e-9, "reuse gain comes from observed fresh-prefill work");
        costs.prefill(1536, 2048, false, 102.4); // observed suffix geometry, not an extrapolated tiny batch
        auto choose = [&](const PrefixCosts& model, bool measured, double margin = 0) {
            PrefixChoice selection(measured, margin);
            selection.consider(0, 0, model.prefill_ms(0, 2048, false).value() +
                model.transfer_ms(PrefixTransfer::park, park).value());
            selection.consider(1, 1024, model.transfer_ms(PrefixTransfer::resident_live, resident).value() +
                model.prefill_ms(1024, 2048, false).value() + 5);
            selection.consider(2, 1792, model.transfer_ms(PrefixTransfer::parked, parked).value() +
                model.prefill_ms(1792, 2048, false).value() + 5);
            return selection;
        };
        check(choose(costs, true).selected() == 1 && choose(costs, true).using_cost(),
              "cheaper shallower resident source beats a deeper parked source");
        check(choose(costs, false).selected() == 2, "longest policy is the original rollback");
        PrefixCosts fast = costs;
        for (int i = 0; i < 8; ++i) fast.transfer(PrefixTransfer::parked, parked, 1);
        check(choose(fast, true).selected() == 2, "deeper parked image wins when the measured copy is worthwhile");
        PrefixChoice overpriced(true);
        overpriced.consider(0, 0, 10);
        overpriced.consider(1, 1000, 100);
        check(overpriced.selected() == 0, "full recomputation beats an expensive restore");
        check(choose(costs, true, 300).selected() == 2, "preregistered savings margin keeps longest");
        PrefixChoice tied(true);
        tied.consider(0, 0, 200);
        tied.consider(1, 1024, 100);
        tied.consider(2, 1792, 100);
        tied.consider(3, 1792, 100);
        check(tied.selected() == 2, "cost ties preserve longest and its original stable tie order");
        PrefixChoice unknown(true);
        unknown.consider(0, 0, 100);
        unknown.consider(1, 1024, 20);
        unknown.consider(2, 1792, {});
        check(unknown.selected() == 2 && unknown.unknown(), "unpriced longest uses deterministic longest fallback");
        unknown.consider(3, 2048, std::numeric_limits<double>::quiet_NaN());
        check(unknown.selected() == 3, "nonfinite costs cannot become a spurious cheapest source");
        PrefixChoice missing_alternative(true);
        missing_alternative.consider(0, 0, {});
        missing_alternative.consider(1, 1024, 20);
        missing_alternative.consider(2, 1792, 100);
        check(missing_alternative.selected() == 1,
              "unobserved alternatives cannot veto improvement over a priced longest prefix");
        check(!costs.transfer_ms(PrefixTransfer::parked, {10000, 23}),
              "observations never cross resolved device or residency paths");
        check(!costs.transfer_ms(PrefixTransfer::checkpoint, parked),
              "capture, restore and resident bounce costs are separate observations");
        check(!costs.transfer_ms(PrefixTransfer::parked, {40001, 22}) &&
              costs.transfer_ms(PrefixTransfer::parked, {40000, 22}) == 1600.0,
              "transfer prediction stops at its declared four-times size bound");
        check(!costs.prefill_ms(0, 65536, false) && !costs.prefill_ms(1024, 2048, true),
              "unknown context scale and window geometry remain unknown");
        check(!costs.prefill_ms(2000, 2048, false), "tiny batched parts cannot inherit a long-prefill rate");
        costs.prefill(1024, 2048, true, 512, 7);
        check(costs.prefill_ms(1536, 2048, true, 7) == 256.0 &&
              !costs.prefill_ms(1536, 2048, true, 8), "request arithmetic paths calibrate separately");
        costs.transfer(PrefixTransfer::parked, parked, std::numeric_limits<double>::infinity());
        check(costs.transfer_ms(PrefixTransfer::parked, parked) == 400.0, "failed/nonfinite timing is not learned");
        PrefixWork overflow{UINT64_MAX - 2, 1};
        overflow.add(3);
        check(!costs.transfer_ms(PrefixTransfer::parked, overflow), "byte overflow is unknown, never a cheap copy");
    }
    {
        // Price every compatible parked branch without taking, copying or evicting it.
        ConversationCache cache(4096, 4);
        auto shallow = image({1,2});
        auto deep = image({1,2,3,4});
        deep.checkpoints.push_back(ConversationCheckpoint{});
        deep.checkpoints.back().ids = {1,2,3};
        check(cache.put(std::move(shallow)) && cache.put(std::move(deep)), "park cost-selection branches");
        const size_t bytes = cache.bytes();
        const size_t size_before = cache.size(), evictions_before = cache.evictions();
        int visits = 0;
        auto denied = [&](auto, const auto&, const auto&) { ++visits; };
        cache.for_each_match(std::vector<int32_t>{1,2,3,4,5}, {}, true, ConversationScope{{2}}, denied);
        cache.for_each_match(std::vector<int32_t>{1,2,3,4,5}, {}, true, {}, denied);
        cache.for_each_match(std::vector<int32_t>{1,2,3,4,5}, {}, false, test_scope, denied);
        check(visits == 0, "identity and steering refusals happen before any economic comparison");
        auto economics = [&](const ConversationScope& scope) {
            strata::program::PrefixChoice selection(true);
            selection.consider(0, 0, 1000);
            size_t candidate = 0;
            cache.for_each_match(std::vector<int32_t>{1,2,3,4,5}, {}, true, scope,
                [&](auto match, const auto&, const auto&) {
                    selection.consider(++candidate, match.tokens, 1 + 100 * (4 - match.tokens));
                });
            return selection.selected();
        };
        check(economics(test_scope) == 1, "same-conversation long beneficial prefix stays reusable with cost on");
        check(economics(ConversationScope{{2}}) == 0 && economics({}) == 0,
              "even an attractive measured reuse gain cannot authorize a different or absent conversation");
        check(cache.size() == size_before && cache.bytes() == bytes && cache.evictions() == evictions_before &&
              cache.best(std::vector<int32_t>{1,2,3,4,5}, {}, true, test_scope).tokens == 4,
              "economic ranking leaves parked history, byte accounting and lifecycle state unchanged");
    }
    {
        // Freezing for an immutable publication lease: payloads are shared, not copied, and the alias never observes
        // a byte written through the image afterwards (copy-on-write detach).
        auto original = image({1, 2, 3, 4});
        original.live.ple.resize(1024, 11);
        original.checkpoints.push_back(ConversationCheckpoint{});
        original.checkpoints.back().ids = {1, 2};
        original.checkpoints.back().gdn.resize(64, 12);
        original.kv.push_back(ConversationKv{});
        original.kv.back().k.resize(4096, 13);
        const auto before_gdn = original.live.gdn.read();
        std::vector<uint8_t> before_k(original.kv.back().k.size());
        original.kv.back().k.read(before_k.data(), 0, before_k.size());
        auto lease = conversation_image_freeze_share(original);
        std::vector<uint8_t> leased_k(lease->kv.back().k.size());
        lease->kv.back().k.read(leased_k.data(), 0, leased_k.size());
        check(lease != nullptr && lease->scope == original.scope && lease->live.ids == original.live.ids &&
              lease->checkpoints.size() == original.checkpoints.size() &&
              lease->checkpoints.back().ids == original.checkpoints.back().ids &&
              lease->live.gdn.read() == before_gdn && lease->kv.size() == 1 &&
              leased_k == before_k,
              "a frozen alias carries the image's metadata and bytes");
        check(original.live.gdn.sealed() && original.live.ple.sealed() &&
              original.checkpoints.back().gdn.sealed() && original.kv.back().k.frozen(),
              "freezing seals every state array and freezes every buffer");
        check(original.live.gdn.data() == lease->live.gdn.data(),
              "the alias shares the frozen state instead of copying the payload");
        // A write through the image after the lease exists can only reach the lease's bytes if it did not detach.
        auto* gdn_bytes = original.live.gdn.write().data();
        gdn_bytes[0] = 0xEE;
        check(lease->live.gdn.read() == before_gdn, "a post-freeze write through the image detaches and cannot reach the lease");
        check(original.live.gdn.read() != before_gdn, "the detaching write is visible on the image's own side");
        check(original.live.gdn.write().data() != lease->live.gdn.data(),
              "the image and its lease hold different allocations after the detach");
        // Freezing reaches the nested owners too: a stage image's payloads and a checkpoint's stage parts.
        auto nested = image({8});
        nested.stage_images.push_back(image({8}));
        nested.stage_images.back().live.dead.resize(128, 4);
        nested.checkpoints.push_back(ConversationCheckpoint{});
        nested.checkpoints.back().stage_parts.push_back(ConversationCheckpoint{});
        nested.checkpoints.back().stage_parts.back().ple.resize(64, 5);
        auto nested_alias = conversation_image_freeze_share(nested);
        check(nested.stage_images.back().live.dead.sealed() &&
              nested.checkpoints.back().stage_parts.back().ple.sealed() &&
              nested_alias->stage_images.back().live.dead.data() ==
                  nested.stage_images.back().live.dead.data() &&
              nested_alias->checkpoints.back().stage_parts.back().ple.data() ==
                  nested.checkpoints.back().stage_parts.back().ple.data(),
              "freezing reaches stage images and checkpoint stage parts, sharing their allocations");
    }
    {
        // The unfrozen fallback is still correct: an alias of a buffer that was never frozen copies, so a later write
        // to the source cannot reach it either.
        auto source = image({5, 6, 7});
        source.live.tails.resize(256, 21);
        auto copy = source.frozen_alias();
        check(!source.live.tails.sealed(), "a capture that was never frozen is not reported as frozen");
        check(copy.live.tails.read() == source.live.tails.read(), "the fallback alias starts equal");
        check(copy.live.tails.data() != source.live.tails.data(),
              "the fallback alias owns its own bytes, so the source is free to move on");
        source.live.tails.write()[0] = 0x5A;
        check(copy.live.tails.read()[0] == 21, "a write to the source cannot reach an unfrozen alias");
    }
    {
        // Publication leases are charged to the same envelope as the parked payloads, and releasing them restores
        // admission exactly.  Every reference is charged in full, so the peak can never over-commit.
        auto one = image({1, 2, 3});
        one.live.gdn.resize(4096, 9);
        const size_t unit = one.bytes();
        ConversationCache cache(unit * 3, 4);
        check(cache.try_reserve_lease(unit * 3), "a lease may use the whole envelope when nothing is parked");
        check(cache.envelope_bytes() == unit * 3 && cache.leased_bytes() == unit * 3,
              "the lease is visible in the envelope and its own accounting");
        check(!cache.try_reserve_lease(1), "a lease cannot exceed the budget");
        check(!cache.can_fit(unit) && !cache.reserve_snapshot(unit) && !cache.put(image({9, 8, 7})),
              "parking and capture admission both stop while the envelope is fully leased");
        std::vector<ConversationKv> retained(1);
        retained[0].k.resize(64);
        cache.retain(std::move(retained), 3);
        check(cache.retained_bytes() == 0, "retained K/V cannot spend a leased envelope again");
        cache.release_lease(unit);
        check(cache.leased_bytes() == unit * 2 && cache.can_fit(unit),
              "releasing part of a lease restores exactly that much admission");
        cache.release_lease(unit * 4);
        check(cache.leased_bytes() == 0, "releasing more than was reserved never underflows");
        check(cache.put(image({9, 8, 7})), "parking works again once the leases are released");
        // A frozen, published image is charged in full even though its bytes are shared with the parked entry: the
        // bookkeeping stays conservative and the physical floor remains the real gate.
        auto parked = image({4, 5, 6});
        parked.live.gdn.resize(4096, 3);
        check(cache.put(std::move(parked)), "park the image that will be published");
        const size_t parked_bytes = cache.bytes();
        check(!cache.try_reserve_lease(unit * 3 - parked_bytes + 1),
              "a lease cannot spend bytes already occupied by parked histories");
        auto published = image({4, 5, 6, 7});
        published.live.gdn.resize(4096, 3);
        auto alias = conversation_image_freeze_share(published);
        check(cache.try_reserve_lease(alias->bytes()), "the publication lease is admitted beside the parked entry");
        check(cache.envelope_bytes() > parked_bytes && cache.bytes() == parked_bytes,
              "the lease changes the envelope, not the parked payload accounting");
        cache.release_lease(alias->bytes());
        check(cache.envelope_bytes() == parked_bytes,
              "releasing the publication lease returns the envelope to the parked payloads alone");
        check(published.live.gdn.data() == alias->live.gdn.data() &&
              published.live.gdn.read() == alias->live.gdn.read(),
              "the publication alias shares the frozen image's allocations with no payload copy");
    }
    {
        // The stored candidate's contract inside the existing selector (the demand path feeds its disk candidate
        // through the same PrefixChoice, never a second selector): an equal cut never outranks the RAM/live/resident
        // source enumerated first, and an unpriced longer candidate keeps the longest-prefix rollback rather than
        // inventing an SSD rate.
        strata::program::PrefixChoice choice(false);
        choice.consider(0, 1000, 5.0);          // the RAM/live cut, priced by real observations
        choice.consider(1, 1000, std::nullopt); // an equal stored cut, unpriced
        check(choice.selected() == 0, "an equal stored cut does not displace the RAM source enumerated first");
        strata::program::PrefixChoice cost(true);
        cost.consider(0, 1000, 5.0);
        cost.consider(1, 2000, std::nullopt);
        check(cost.longest() == 1 && cost.selected() == 1 && !cost.using_cost(),
              "an unpriced longer stored cut keeps the longest rollback instead of a measured choice");
        strata::program::PrefixChoice priced(true);
        priced.consider(0, 1000, 5.0);
        priced.consider(1, 2000, 9.0);
        check(priced.using_cost() && priced.selected() == 0,
              "a priced stored candidate competes in the existing cost policy like any other source");
    }
    {
        // The reuse intervals a capture may offer: each one names the donor head the retained bytes came from, ends on
        // that donor's own object grid, and claims nothing the capture did not keep.  The store re-validates all of it;
        // this is the caller-side half of the proof.
        const uint64_t chunk = 1u << 20;
        std::vector<ConversationStoreStreamInfo> streams;
        auto add_stream = [&](uint32_t stage, ConversationStoreStreamKind kind, uint64_t ordinal, uint64_t bytes) {
            ConversationStoreStreamInfo info;
            info.id = ConversationStoreStreamId{stage, uint16_t(kind), 0, ordinal};
            info.index = uint32_t(streams.size());
            info.bytes = bytes;
            streams.push_back(info);
        };
        add_stream(0, ConversationStoreStreamKind::kv_k, 0, 4 * chunk);
        add_stream(0, ConversationStoreStreamKind::kv_v, 0, 4 * chunk);
        add_stream(0, ConversationStoreStreamKind::kv_pooled, 0, chunk / 2);
        add_stream(1, ConversationStoreStreamKind::kv_k, 0, 2 * chunk);
        auto kept = [](uint32_t stage, uint32_t layer, uint32_t part, uint64_t bytes, uint64_t donor_bytes) {
            ConversationRetainedPrefix prefix;
            prefix.stage = stage;
            prefix.layer = layer;
            prefix.part = part;
            prefix.bytes = bytes;
            prefix.donor_bytes = donor_bytes;
            return prefix;
        };
        const std::vector<ConversationRetainedPrefix> reported = {
            kept(0, 0, 0, 3 * chunk + 4096, 4 * chunk),   // an unaligned prefix of this part's objects
            kept(0, 0, 4, chunk / 2, chunk / 2),          // the whole (short) pooled page: it ends both streams
            kept(1, 0, 0, 2 * chunk, 2 * chunk),          // a stage image's own stream, not the top image's
        };
        const auto reuse = conversation_store_reuse_intervals(streams, reported, 7, 5, 5, chunk);
        check(reuse.size() == 3, "every retained part may offer exactly one interval");
        check(reuse[0].head_id == 7 && reuse[0].offset == 0 && reuse[0].source_offset == 0 &&
              reuse[0].stream.stage == 0 && reuse[0].stream.ordinal == 0 &&
              reuse[0].stream.kind == uint16_t(ConversationStoreStreamKind::kv_k) &&
              reuse[0].bytes == 3 * chunk,
              "an unaligned retained prefix is offered down to the donor's own object grid");
        check(reuse[1].bytes == chunk / 2 && reuse[1].stream.kind == uint16_t(ConversationStoreStreamKind::kv_pooled),
              "a retained part that ends both streams needs no object grid");
        check(reuse[2].stream.stage == 1 && reuse[2].bytes == 2 * chunk,
              "a stage image's interval names the stage's own stream identity");
        check(conversation_store_reuse_intervals(streams, reported, 7, 4, 5, chunk).empty(),
              "a donor captured in another execution-domain epoch authorizes nothing");
        check(conversation_store_reuse_intervals(streams, reported, 0, 5, 5, chunk).empty(),
              "a capture with no named donor authorizes nothing");
        check(conversation_store_reuse_intervals(streams, reported, 7, 5, 0, chunk).empty(),
              "a capture outside any execution epoch authorizes nothing");
        const auto clamped = conversation_store_reuse_intervals(streams,
            {kept(0, 0, 0, 5 * chunk, 3 * chunk)}, 7, 5, 5, chunk);
        check(clamped.size() == 1 && clamped[0].bytes == 3 * chunk,
              "an interval is bounded by what the donor actually holds");
        const auto unknown = conversation_store_reuse_intervals(streams, {kept(0, 0, 1, chunk, chunk),
                                                                         kept(0, 3, 0, chunk, chunk)}, 7, 5, 5, chunk);
        check(unknown.size() == 1 && unknown[0].bytes == chunk,
              "a part whose stream identity is absent is skipped, never guessed at");
        check(conversation_store_reuse_intervals(streams, {kept(0, 0, 1, 4096, 4 * chunk)}, 7, 5, 5, chunk).empty(),
              "a retained prefix shorter than one object offers nothing");
        check(conversation_store_reuse_intervals(streams, {kept(0, 0, 1, 9 * chunk, 9 * chunk)}, 7, 5, 5, chunk).empty(),
              "a retained length longer than the outgoing stream offers nothing");
    }
    {
        // The retained K/V carries the store head it was restored from, so a later capture can name the donor; a
        // RAM-only restore carries none, and a capture then offers nothing.
        ConversationCache cache(1u << 20, 2);
        std::vector<ConversationKv> layers(1);
        layers[0].k.resize(512);
        cache.retain(std::move(layers), 8, {}, 91, 4);
        const ConversationKvReuse named = cache.take_reuse();
        check(named.donor_head == 91 && named.donor_epoch == 4 && named.captured_tokens == 8,
              "retained K/V keeps the head and epoch it was restored from");
        std::vector<ConversationKv> anonymous(1);
        anonymous[0].k.resize(512);
        cache.retain(std::move(anonymous), 8);
        check(cache.take_reuse().donor_head == 0, "a RAM-only restore retains its bytes without naming a donor");
    }
    std::printf("conversation_cache_test: %d checks passed\n", checks);
}
