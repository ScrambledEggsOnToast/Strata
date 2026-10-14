#pragma once

#include "strata/core/conversation_cache.hpp"
#include "strata/plan/qsa_memory.hpp"
#include <array>
#include <cstring>

namespace strata::core {

// Functional-only observer. No ownership changes, allocations, CUDA or hashing.
// Exact comparisons establish duplicate payload, not a speedup or safe mutable alias.
struct PrefixExtentProbe {
    const char* status = "no-cut";
    int64_t cut = 0;
    uint64_t eligible_bytes = 0, compared_bytes = 0, duplicate_bytes = 0, aliased_bytes = 0;
};

namespace prefix_extent_detail {
inline const ConversationCheckpoint* checkpoint(const SavedConversation& image,
                                                const ConversationCheckpoint& cut) {
    for (const auto& c : image.checkpoints)
        if (c.ids == cut.ids && c.imgs == cut.imgs && c.stage_parts.empty()) return &c;
    return nullptr;
}
inline bool prefix(const SavedConversation& image, const ConversationCheckpoint& c) {
    if (c.ids.empty() || c.ids.size() > image.live.ids.size() || !c.stage_parts.empty() ||
        !std::equal(c.ids.begin(), c.ids.end(), image.live.ids.begin())) return false;
    size_t n = 0;
    for (const auto& key : image.live.imgs) {
        if (key.start < 0) return false;
        if (uint64_t(key.start) >= c.ids.size()) continue;
        if (n >= c.imgs.size() || !(key == c.imgs[n++])) return false;
    }
    return n == c.imgs.size();
}
inline std::array<const ConversationBuffer*, 5> buffers(const ConversationKv& kv) {
    return {&kv.k, &kv.v, &kv.k_scale, &kv.v_scale, &kv.pooled};
}
inline bool extents(const ConversationKv& kv, int64_t tokens, int64_t cut,
                    std::array<uint64_t, 5>& sizes) {
    auto shape = strata::kernels::qsa_real_shapes();
    shape.n_head_kv = kv.heads; shape.head_dim = kv.head_dim; shape.idx_dim = kv.idx_dim;
    const int fmt = kv.format >= 16 ? kv.format - 16 : kv.format;
    if (kv.format < 0 || kv.format > 19 || (kv.format >= 16 && fmt > 1) ||
        kv.page_size != shape.page_size || cut <= 0 || cut > tokens) return false;
    const bool index = kv.pooled_rows != 0;
    strata::plan::QsaImageBytes full, part;
    if (!strata::plan::qsa_image_bytes(shape,tokens,fmt,index,full)) return false;
    if (kv.cells != int64_t(full.cells) || kv.pooled_rows != int64_t(full.pooled_rows)) return false;
    const auto data = buffers(kv);
    const std::array<uint64_t, 5> expected{full.k,full.v,full.k_scale,full.v_scale,full.pooled};
    for (size_t i=0;i<data.size();++i) if (data[i]->size()!=expected[i]) return false;
    // Draft's last possible uncomputed cell is never included, even at a page boundary.
    const int64_t immutable_tokens = index ? cut : cut - 1;
    const int64_t cells = immutable_tokens / shape.page_size * shape.page_size;
    if (!strata::plan::qsa_image_bytes(shape,cells,fmt,false,part)) return false;
    sizes = {part.k,part.v,part.k_scale,part.v_scale,
             index ? uint64_t(cut / shape.idx_block) * uint64_t(kv.idx_dim) * sizeof(float) : 0};
    for (size_t i=0;i<data.size();++i) if (sizes[i]>data[i]->size()) return false;
    return true;
}
inline bool pair_layout(const SavedConversation& a, const SavedConversation& b,
                        const ConversationCheckpoint& cut) {
    if (a.cvec!=b.cvec || a.geometry!=b.geometry || a.layer_lo!=b.layer_lo || a.layer_hi!=b.layer_hi ||
        a.live.ids.size()>INT32_MAX || b.live.ids.size()>INT32_MAX ||
        a.kv.size()!=b.kv.size() || a.kv.size()>64 ||
        a.checkpoints.size()>64 || b.checkpoints.size()>64 ||
        !prefix(a,cut) || !prefix(b,cut) || !checkpoint(a,cut) || !checkpoint(b,cut)) return false;
    for (size_t j=0;j<a.kv.size();++j) {
        const auto& x=a.kv[j]; const auto& y=b.kv[j];
        std::array<uint64_t,5> xs,ys;
        if (x.format!=y.format || x.heads!=y.heads || x.head_dim!=y.head_dim || x.idx_dim!=y.idx_dim ||
            (x.pooled_rows==0)!=(y.pooled_rows==0) ||
            !extents(x,int64_t(a.live.ids.size()),int64_t(cut.ids.size()),xs) ||
            !extents(y,int64_t(b.live.ids.size()),int64_t(cut.ids.size()),ys) || xs!=ys) return false;
    }
    return true;
}
} // namespace prefix_extent_detail

inline PrefixExtentProbe probe_prefix_extents(const SavedConversation& a, const SavedConversation& b,
                                              uint64_t max_bytes = 64u << 20) {
    using namespace prefix_extent_detail;
    PrefixExtentProbe out;
    // Never inspect payload across authorization/steering namespaces.
    if (!a.scope.matches(b.scope) || a.cvec!=b.cvec) { out.status="scope"; return out; }
    if (a.stage_images.size()!=b.stage_images.size() || a.stage_images.size()>8 ||
        a.checkpoints.size()>64 || b.checkpoints.size()>64) { out.status="layout"; return out; }
    const ConversationCheckpoint* cut=nullptr;
    bool skipped_large_cut=false;
    for (const auto& c : a.checkpoints) {
        if (c.ids.size()>4096) { skipped_large_cut=true; continue; }
        if (c.ids.empty() || (cut && c.ids.size()<=cut->ids.size())) continue;
        if (!pair_layout(a,b,c)) continue;
        bool complete=true;
        for (size_t s=0;s<a.stage_images.size();++s)
            complete = complete && pair_layout(a.stage_images[s],b.stage_images[s],c);
        if (complete) cut=&c;
    }
    if (!cut) { if (skipped_large_cut) out.status="cut-limit"; return out; }
    out.cut=int64_t(cut->ids.size());
    auto stage = [](const SavedConversation& image, size_t s) -> const SavedConversation& {
        return s ? image.stage_images[s-1] : image;
    };
    for (size_t s=0;s<=a.stage_images.size();++s) {
        const auto& x=stage(a,s);
        for (const auto& kv:x.kv) {
            std::array<uint64_t,5> sizes;
            if (!extents(kv,int64_t(x.live.ids.size()),out.cut,sizes)) { out.status="layout"; return out; }
            for (auto n:sizes) {
                if (n>UINT64_MAX-out.eligible_bytes) { out.status="limit"; return out; }
                out.eligible_bytes+=n;
            }
        }
    }
    if (out.eligible_bytes>max_bytes) { out.status="limit"; return out; }
    for (size_t s=0;s<=a.stage_images.size();++s) {
        const auto& x=stage(a,s); const auto& y=stage(b,s);
        for (size_t j=0;j<x.kv.size();++j) {
            std::array<uint64_t,5> sizes;
            extents(x.kv[j],int64_t(x.live.ids.size()),out.cut,sizes);
            const auto left=buffers(x.kv[j]),right=buffers(y.kv[j]);
            for (size_t k=0;k<sizes.size();++k) {
                const bool same=left[k]->visit(0,size_t(sizes[k]),[&](const uint8_t* p,size_t n,size_t at) {
                    return right[k]->visit(at,n,[&](const uint8_t* q,size_t count,size_t offset) {
                        for (size_t done=0;done<count;) {
                            const size_t bytes=std::min<size_t>(65536,count-done);
                            const uint8_t* lp=p+(offset-at)+done;
                            const uint8_t* rp=q+done;
                            out.compared_bytes+=bytes;
                            if (std::memcmp(lp,rp,bytes)!=0) return false;
                            if (lp==rp) out.aliased_bytes+=bytes; else out.duplicate_bytes+=bytes;
                            done+=bytes;
                            progress_beat();
                        }
                        return true;
                    });
                });
                if (!same) {
                    out.status="different";
                    out.duplicate_bytes=out.aliased_bytes=0; // no complete-cut equivalence claim
                    return out;
                }
            }
        }
    }
    out.status="equal";
    return out;
}

// Retained-memory optimization only: capture has already completed. No capture
// or restore transfer is removed. The caller validates recurrent payloads against
// the loaded session before entering this pure host helper, just as for the probe.
inline PrefixExtentProbe share_prefix_extents(SavedConversation& target, const SavedConversation& source,
                                              uint64_t max_bytes = 64u << 20) {
    auto out = probe_prefix_extents(target, source, max_bytes);
    if (std::string_view(out.status) != "equal") return out;
    auto share_stage = [&](SavedConversation& dst, const SavedConversation& src) {
        for (size_t j = 0; j < dst.kv.size(); ++j) {
            auto& kv = dst.kv[j];
            std::array<uint64_t, 5> sizes;
            prefix_extent_detail::extents(kv, int64_t(dst.live.ids.size()), out.cut, sizes);
            const std::array<ConversationBuffer*, 5> buffers{&kv.k,&kv.v,&kv.k_scale,&kv.v_scale,&kv.pooled};
            const auto donors = prefix_extent_detail::buffers(src.kv[j]);
            for (size_t k = 0; k < buffers.size(); ++k) buffers[k]->share_prefix(*donors[k], size_t(sizes[k]));
        }
    };
    share_stage(target, source);
    for (size_t s = 0; s < target.stage_images.size(); ++s) share_stage(target.stage_images[s], source.stage_images[s]);
    // The returned comparison is BEFORE replacement. The independent observer
    // reports aliases afterwards; never interpret a sum of pairs as unique saving.
    return out;
}

// Bounded diagnostic scratch, caller-owned, with no retained registry or global
// payload references. Charge sizeof(PrefixBackingLedger) before capture when
// requested. Overflow means incomplete coverage, never a measured zero saving.
struct PrefixBackingLedger {
    static constexpr size_t max_backings = 8192;
    std::array<const void*, max_backings> identities{};
    size_t count = 0;
    uint64_t references = 0, logical_payload = 0, unique_payload = 0;
    uint64_t logical_owner = 0, unique_owner = 0, charged_buffers = 0;
    bool truncated = false;
    void add(const ConversationBuffer& buffer) {
        if (truncated) return;
        charged_buffers += buffer.bytes();
        buffer.for_each_backing([&](const void* identity, size_t capacity, size_t owner) {
            if (++references > max_backings) { truncated = true; return false; }
            logical_payload += capacity; logical_owner += owner;
            for (size_t i = 0; i < count; ++i) if (identities[i] == identity) return true;
            identities[count++] = identity;
            unique_payload += capacity; unique_owner += owner;
            return true;
        });
    }
    void add(const SavedConversation& image) {
        if (truncated) return;
        for (const auto& kv : image.kv)
            for (const auto* buffer : prefix_extent_detail::buffers(kv)) add(*buffer);
        for (const auto& stage : image.stage_images) add(stage);
    }
};

} // namespace strata::core
