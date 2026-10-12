#include "strata/core/conversation_cache.hpp"
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
        check(cache.make_room(parked), "reservation can discard optional active buffers");
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
        cache.put(image({1, 2, 3}));
        check(cache.best(a, {}, true, test_scope).index == 1, "ties prefer most recently parked");
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
        check(cache.make_room(one, held.bytes()), "count in-flight image when reserving outgoing snapshot");
        check(cache.size() == 0, "in-flight reservation evicts otherwise fitting B");
        check(cache.put(image({5,6,7}), held.bytes()), "insert with in-flight accounting");
        check(cache.bytes()+held.bytes() <= one*2, "exchange obeys byte budget");
        check(!cache.make_room(one+1, one), "oversized exchange rejected");
        check(cache.size() == 1, "oversized snapshot does not evict useful entries");
        auto huge = image({1}); huge.live.gdn.resize(one*3);
        check(!cache.put(std::move(huge)), "oversized image rejected");
        check(cache.size() == 1, "oversized put leaves cache unchanged");
        check(!cache.make_room(0, one*2+1), "held larger than budget cannot underflow");
    }
    {
        // #342: MAIN -> SUB (9 turns) -> MAIN with 4 slots.  Every SUB turn parks the previous turn's live state
        // (its reply as generated, which the next request re-rendered) with a chain that holds that turn's
        // boundary checkpoint; the copy a turn back adds only that stale tail and is dropped, so MAIN survives.
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
        std::vector<int32_t> history = with(root, {20});         // SUB's conversation so far, re-rendered
        std::vector<ConversationCheckpoint> chain = {cp(root)};
        for (int turn = 1; turn <= 9; ++turn) {
            chain.push_back(cp(history));                        // the turn boundary the next request resumes from
            SavedConversation sub = image({});
            sub.live.ids = with(history, {900, (int32_t) turn});   // + the reply with its thinking (stale tail)
            sub.checkpoints = chain;
            check(cache.put(std::move(sub)), "park SUB turn");
            check(cache.size() <= 2, "one parked copy of SUB at a time");
            history = with(history, {30, (int32_t) turn});       // the reply as the next request renders it
        }
        check(cache.evictions() == 0 && cache.superseded() == 8, "8 superseded copies dropped, nothing evicted");
        const auto m = cache.best(with(with(root, {10, 11, 12}), {13, 14, 15}), {}, true, test_scope);
        check(m.tokens == 9 && m.live, "MAIN still restores in full");
        const auto s = cache.best(with(history, {40}), {}, true, test_scope);
        check(s.tokens == (int64_t) history.size() - 2, "SUB resumes from its last turn boundary");
    }
    {
        // what is NOT superseded: another conversation sharing only the root, an entry without checkpoints, the
        // other steering mode, a branch whose deepest checkpoint the new chain does not hold
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
        check(cache.put(std::move(same_state)) && cache.superseded() == 1 && cache.size() == 5,
              "the live state equal to an entry's deepest checkpoint supersedes it");
        SavedConversation huge = image({1, 2, 3, 4, 60, 80});
        huge.checkpoints = {cp({1, 2, 3, 4}), cp({1, 2, 3, 4, 60})};
        huge.live.gdn.resize(2 << 20);
        check(!cache.put(std::move(huge)) && cache.size() == 5 && cache.superseded() == 1,
              "an oversized put drops nothing");
    }
    {
        ConversationCache disabled(0,4), no_slots(1024,0);
        check(!disabled.enabled() && !no_slots.enabled(), "both disable switches");
        check(!disabled.put(image({1,2,3})) && !no_slots.put(image({1,2,3})), "disabled cache stores nothing");
        check(disabled.best(a, {}, true, test_scope).tokens == 0, "disabled cache has no matches");
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
        auto split = conversation_checkpoints_split(std::move(checks), 2);
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

        auto torn = conversation_checkpoints_split(std::vector<ConversationCheckpoint>(original), 2);
        torn.parts[1].pop_back();
        check(!conversation_checkpoints_merge(std::move(torn), back) && back.empty(), "a stage short of a part is refused");
        auto mixed = conversation_checkpoints_split(std::vector<ConversationCheckpoint>(original), 2);
        mixed.parts[0][0].ids = {9, 9};
        check(!conversation_checkpoints_merge(std::move(mixed), back) && back.empty(), "a part of another checkpoint is refused");
        // a restored image: stage 0's checkpoints and each stage's, with no record of an incomplete one
        ConversationCheckpointSplit restored;
        restored.stage0.push_back(running({4, 5}, 11, 0));
        restored.parts.resize(2);
        for (auto& p : restored.parts) { auto c = running({4, 5}, 12, 0); p.push_back(std::move(c)); }
        check(conversation_checkpoints_merge(std::move(restored), back) && back.size() == 1 &&
              back[0].stage_parts.size() == 2, "a restored image's parts are all whole");
        auto none = conversation_checkpoints_split({}, 3);
        check(conversation_checkpoints_merge(std::move(none), back) && back.empty(), "no checkpoints: nothing to do");
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
        int visits = 0;
        std::vector<int64_t> order;
        cache.for_each_match(std::vector<int32_t>{1,2,3,4,5}, {}, true, test_scope,
            [&](auto, const auto&, const auto& c) { ++visits; order.push_back((int64_t) c.ids.size()); });
        check(visits == 3 && order == std::vector<int64_t>({4,3,2}),
              "all live/checkpoint candidates visit newest first in the old tie order");
        check(cache.size() == 2 && cache.bytes() == bytes && cache.evictions() == 0 &&
              cache.best(std::vector<int32_t>{1,2,3,4,5}, {}, true, test_scope).tokens == 4,
              "candidate enumeration changes no history, byte accounting or lifecycle state");
        visits = 0;
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
    }
    std::printf("conversation_cache_test: %d checks passed\n", checks);
}
