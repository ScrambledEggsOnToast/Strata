#include "strata/core/conversation_cache.hpp"
#include <cstdlib>
#include <cstdio>
#include <new>
static long fail_after=-1;
void* operator new(std::size_t n) {
    if(fail_after==0) throw std::bad_alloc();
    if(fail_after>0) --fail_after;
    if(void* p=std::malloc(n?n:1)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p,std::size_t) noexcept { std::free(p); }
using strata::core::ConversationCheckpoint;
bool equal(const ConversationCheckpoint& a,const ConversationCheckpoint& b) {
    if(a.ids!=b.ids || a.imgs!=b.imgs || a.gdn!=b.gdn || a.ple!=b.ple || a.tails!=b.tails || a.dead!=b.dead || a.block_pos!=b.block_pos || a.used!=b.used || a.pinned!=b.pinned || a.stage_parts.size()!=b.stage_parts.size()) return false;
    for(size_t i=0;i<a.stage_parts.size();++i) if(!equal(a.stage_parts[i],b.stage_parts[i])) return false;
    return true;
}
static bool equal_saved(const strata::core::SavedConversation& a, const strata::core::SavedConversation& b) {
    if (a.scope != b.scope || a.geometry != b.geometry || a.layer_lo != b.layer_lo || a.layer_hi != b.layer_hi ||
        a.cvec != b.cvec || !equal(a.live, b.live) || a.checkpoints.size() != b.checkpoints.size() ||
        !a.kv.empty() || !b.kv.empty() || !a.stage_images.empty() || !b.stage_images.empty()) return false;
    for (size_t i = 0; i < a.checkpoints.size(); ++i)
        if (!equal(a.checkpoints[i], b.checkpoints[i])) return false;
    return true;
}
int main() {
    {
        strata::core::CheckpointBytes source({1,2,3});
        source.seal();
        auto reader=source;
        const auto* storage=source.data();
        fail_after=0;
        bool refused=false;
        try { reader.write()[0]=9; } catch(const std::bad_alloc&) { refused=true; }
        fail_after=-1;
        if(!refused || source.data()!=storage || reader.data()!=storage || source[0]!=1 || reader[0]!=1) return 9;
    }

    // Every allocation failure while detaching a shared prefix preserves both
    // readers; sharing itself needs no new allocation after capture admission.
    for (bool shrink : {false, true}) for (size_t extents : {size_t{1}, size_t{2}}) {
        bool finished = false;
        int faults = 0;
        for (int n = 0; n < 16 && !finished; ++n) {
            strata::core::ConversationBuffer source(true), reader(true);
            const size_t extent = strata::core::ConversationBuffer::prefix_extent_bytes;
            const size_t original_size = extents * extent;
            source.resize(extent, 7); reader.resize(original_size, 7);
            fail_after = 0;
            try {
                if (reader.share_prefix(source, extent) != extent) return 10;
            } catch (const std::bad_alloc&) { return 11; }
            fail_after = -1;
            auto address = [](const auto& buffer) {
                const uint8_t* result = nullptr;
                buffer.visit(0, 1, [&](const uint8_t* p, size_t, size_t) { result = p; return true; });
                return result;
            };
            const auto* original = address(source);
            const size_t charged = reader.bytes();
            const size_t admitted = reader.allocation_peak(shrink ? 17 : original_size);
            fail_after = n;
            try {
                if (shrink) reader.resize(17);
                else reader.visit(0, 1, [](uint8_t* p, size_t, size_t) { *p = 9; return true; });
                fail_after = -1;
                if (address(reader) == original || reader.bytes() > admitted) return 12;
                finished = true;
            } catch (const std::bad_alloc&) {
                fail_after = -1;
                ++faults;
                if (address(reader) != original || reader.size() != original_size || reader.bytes() != charged) return 13;
                if (!std::as_const(reader).visit(0, reader.size(), [](const uint8_t* p, size_t n, size_t) {
                        return std::all_of(p, p + n, [](uint8_t byte) { return byte == 7; });
                    })) return 16;
            }
            if (address(source) != original || source.size() != extent || *original != 7) return 14;
        }
        if (!finished || !faults) return 15;
    }

    // Publication must allocate its deque storage before pruning/eviction.
    // Vary occupancy to cross both block and map-growth boundaries, and inject
    // every allocation failure before the first successful, nonthrowing commit.
    long publication_faults = 0, largest_fault_set = 0;
    for (int mode = 0; mode < 3; ++mode) for (size_t owners = 2; owners <= 64; ++owners) {
        bool finished = false;
        int faults = 0;
        for (long allowance = 0; allowance < 16 && !finished; ++allowance) {
            using namespace strata::core;
            std::vector<SavedConversation> original;
            SavedConversation candidate;
            candidate.scope = ConversationScope{{1}};
            candidate.live.gdn.resize(64, 9);
            candidate.live.ids.resize(mode == 2 ? 30000 : owners + 3);
            for (size_t i = 0; i < candidate.live.ids.size(); ++i) candidate.live.ids[i] = (int32_t)i + 1;
            size_t original_bytes = 0, ordinary_bytes = 0;
            for (size_t i = 0; i < owners; ++i) {
                SavedConversation saved;
                saved.scope = candidate.scope;
                saved.live.gdn.resize(64, (uint8_t)i);
                if (i % 3 == 0) {
                    saved.live.ids.resize(i + 2);
                    for (size_t j = 0; j < saved.live.ids.size(); ++j) saved.live.ids[j] = (int32_t)j + 1;
                    auto pin = saved.live; pin.ids.pop_back(); pin.pinned = true;
                    saved.checkpoints.push_back(pin);
                    candidate.checkpoints.push_back(std::move(pin));
                } else {
                    saved.live.ids = {(int32_t)i + 1000, 2};
                    ordinary_bytes += saved.bytes();
                }
                original_bytes += saved.bytes();
                original.push_back(std::move(saved));
            }
            if (mode == 1) { candidate.live.ids = {999, 998}; candidate.checkpoints.clear(); }
            const SavedConversation candidate_before = candidate;
            const size_t budget = mode == 2
                ? std::max(original_bytes, candidate.bytes() + ordinary_bytes / 2) : (1u << 20);
            ConversationCache cache(budget, owners);
            for (const auto& saved : original) {
                auto copy = saved;
                if (!cache.put(std::move(copy))) return 20;
            }
            std::vector<ConversationKv> reuse(1);
            cache.retain(std::move(reuse), 2);
            const size_t before_reuse = cache.retained_bytes();
            size_t keep_from = owners, room = budget - candidate.bytes();
            if (mode == 2) for (size_t i = owners; i-- > 0;) {
                if (i % 3 == 0) continue;
                if (original[i].bytes() > room) break;
                room -= original[i].bytes();
                keep_from = i;
            }
            const auto survives = [&](size_t i) {
                return mode == 1 ? i != 1 : i % 3 != 0 && (mode == 0 || i >= keep_from);
            };
            const size_t before_bytes = cache.bytes(), before_evictions = cache.evictions();
            const size_t before_superseded = cache.superseded();
            auto prompt = candidate.live.ids; prompt.push_back(123);
            fail_after = allowance;
            try {
                if (!cache.put(std::move(candidate))) { fail_after = -1; return 21; }
                // All remaining allocations are still fault-armed here.
                if (cache.bytes() > budget || cache.size() > owners ||
                    cache.best(prompt, {}, true, ConversationScope{{1}}).tokens != (int64_t)prompt.size() - 1) {
                    fail_after = -1; return 22;
                }
                bool pins_present = true;
                for (const auto& saved : original) for (const auto& pin : saved.checkpoints) {
                    bool found = false;
                    cache.for_each_image([&](size_t, const SavedConversation& entry) {
                        for (const auto& held : entry.checkpoints)
                            found = found || (held.pinned && held.ids == pin.ids && held.imgs == pin.imgs);
                    });
                    pins_present = pins_present && found;
                }
                size_t expected = 0;
                bool identities = true, candidate_seen = false;
                cache.for_each_image([&](size_t, const SavedConversation& entry) {
                    while (expected < owners && !survives(expected)) ++expected;
                    if (expected < owners) {
                        identities = identities && equal_saved(entry, original[expected++]);
                    } else {
                        identities = identities && !candidate_seen && equal_saved(entry, candidate_before);
                        candidate_seen = true;
                    }
                });
                while (expected < owners && !survives(expected)) ++expected;
                const bool reuse_correct = cache.retained_bytes() == (mode == 2 ? 0 : before_reuse);
                fail_after = -1;
                if (!pins_present || !identities || !candidate_seen || expected != owners || !reuse_correct) return 23;
                finished = true;
            } catch (const std::bad_alloc&) {
                fail_after = -1;
                ++faults;
                if (cache.bytes() != before_bytes || cache.size() != original.size() ||
                    cache.evictions() != before_evictions || cache.superseded() != before_superseded ||
                    cache.retained_bytes() != before_reuse || !equal_saved(candidate, candidate_before)) return 24;
                bool intact = true;
                cache.for_each_image([&](size_t index, const SavedConversation& saved) {
                    intact = intact && index < original.size() && equal_saved(saved, original[index]);
                });
                if (!intact) return 25;
            }
        }
        if (!finished) return 26;
        publication_faults += faults;
        largest_fault_set = std::max(largest_fault_set, (long)faults);
    }
    std::printf("cache publication: preserved %ld allocation failures; largest metadata-growth set=%ld\n",
                publication_faults, largest_fault_set);

    // #752: every stage count a layer split runs (1-3), with one checkpoint that has no stage parts (it stays in `rest`)
    long total_faults=0;
    for (auto retention : {strata::core::ConversationCheckpointRetention::all,
                           strata::core::ConversationCheckpointRetention::parked_history})
    for(size_t stages=1;stages<=3;++stages) {
        int faults=0;
        bool finished=false;
        for(int n=0;n<256 && !finished;++n) {
            std::vector<ConversationCheckpoint> checks(5);
            for(size_t i=0;i<checks.size();++i) {
                auto& c=checks[i]; c.ids={1,2,(int)i+3};c.gdn={1,2,3};c.ple={4};c.used=i;
                c.pinned = i == 1;
                if(i==2) continue;   // incomplete: no stage parts
                c.stage_parts.resize(stages);
                for(size_t k=0;k<stages;++k) {
                    auto& part = c.stage_parts[k];
                    part.ids = c.ids; part.used = c.used; part.pinned = c.pinned;
                    part.gdn={5,6,static_cast<uint8_t>(7+k)}; part.tails={static_cast<uint8_t>(k)};
                }
            }
            const auto backup=checks;
            fail_after=n;
            try {
                auto split=strata::core::conversation_checkpoints_split(std::move(checks),stages,
                    retention);
                fail_after=0; // Merge-back after a successful split must not allocate.
                if(!strata::core::conversation_checkpoints_merge(std::move(split),checks)) return 2;
                fail_after=-1;
                if(checks.size()!=backup.size()) return 6;
                finished=true;   // the split went through: the merge gives the running state back, in order
                for(size_t i=0;i<checks.size();++i) {
                    const auto& x=checks[i]; const auto& y=backup[i];
                    if (!equal(x, y)) return 7;
                }
            } catch(const std::bad_alloc&) {
                fail_after=-1;++faults;
                if(checks.size()!=backup.size()) return 4;
                for(size_t i=0;i<checks.size();++i) if(!equal(checks[i],backup[i])) {
                    std::fprintf(stderr,"stages %zu: allocation failure %d partially moved checkpoint %zu\n",stages,n,i);return 1;
                }
            }
        }
        if(!finished || faults==0) return 5;
        std::printf("stages %zu retention=%s: allocation-failure cases preserved: %d\n", stages,
                    retention == strata::core::ConversationCheckpointRetention::all ? "all" : "parked", faults);
        total_faults+=faults;
    }
    return total_faults>0?0:3;
}
